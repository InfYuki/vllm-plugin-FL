# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""FLA decode with graph-sized scratch rows and masked paged writeback."""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger


logger = init_logger(__name__)


@dataclass(frozen=True)
class NativeDecodeOps:
    convolution: Callable | None = None
    recurrent: Callable | None = None


@lru_cache(maxsize=1)
def get_native_decode_ops():
    """Load the installed wheel lazily on the worker's selected device."""
    try:
        from fla_npu.ops import ascendc
    except (ImportError, OSError, RuntimeError) as exc:
        logger.warning("Qwen3.8 GDN AscendC decode unavailable: %s", exc)
        return NativeDecodeOps()
    convolution = getattr(ascendc, "causal_conv1d_update", None)
    recurrent = getattr(ascendc, "recurrent_gated_delta_rule", None)
    return NativeDecodeOps(
        convolution=convolution if callable(convolution) else None,
        recurrent=recurrent if callable(recurrent) else None,
    )


@lru_cache(maxsize=1)
def _fused_gating_enabled():
    return os.environ.get("VLLM_FL_QWEN38_FUSED_GDN_GATING") == "1"


@lru_cache(maxsize=1)
def _indexed_conv_enabled():
    return os.environ.get("VLLM_FL_QWEN38_INDEXED_GDN_CONV") == "1"


def _can_update_conv_pool(x, conv_state, indices):
    return (
        _indexed_conv_enabled()
        and is_forward_context_available()
        and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL
        and x.device.type == "npu"
        and x.shape == (8, 1280)
        and x.dtype == conv_state.dtype == torch.bfloat16
        and conv_state.device == x.device == indices.device
        and conv_state.ndim == 3
        and conv_state.shape[1:] == (1280, 3)
        and conv_state.stride()[1:] == (1, 1280)
        and conv_state.stride(0) >= 3840
        and conv_state.stride(0) % 16 == 0
        and conv_state.storage_offset() % 16 == 0
        and indices.shape == (8,)
        and indices.dtype == torch.int32
    )


@lru_cache(maxsize=1)
def _direct_ssm_enabled():
    return os.environ.get("VLLM_FL_QWEN38_DIRECT_GDN_SSM") == "1"


def _can_update_ssm_pool(x, ssm_state, indices, output):
    return (
        _direct_ssm_enabled()
        and is_forward_context_available()
        and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL
        and x.device.type == "npu"
        and x.device == ssm_state.device == indices.device == output.device
        and x.shape == (8, 1280)
        and x.dtype == output.dtype == torch.bfloat16
        and output.shape == (8, 6, 128)
        and output.is_contiguous()
        and indices.shape == (8,)
        and indices.dtype == torch.int32
        and indices.is_contiguous()
        and ssm_state.dtype == torch.float32
        and ssm_state.ndim == 4
        and ssm_state.shape[1:] == (6, 128, 128)
        and 8 < ssm_state.shape[0] <= 2**31
        and ssm_state.stride()[1:] == (16384, 128, 1)
        and ssm_state.stride(0) >= 98304
        and ssm_state.stride(0) % 8 == 0
        and ssm_state.storage_offset() % 8 == 0
    )


@lru_cache(maxsize=1)
def _fused_qkv_enabled():
    return os.environ.get("VLLM_FL_QWEN38_FUSED_GDN_QKV") == "1"


def gdn_decode_native(
    x,
    a,
    b,
    indices,
    conv_state,
    ssm_state,
    output,
    *,
    ops,
    conv_weights,
    conv_bias,
    activation,
    A_log,
    dt_bias,
    num_key_heads,
    num_value_heads,
    head_k_dim,
    head_v_dim,
    packed_beta,
):
    from .gdn_reference import (
        _decode_convolution,
        _decode_recurrence,
        _gating,
        _prepare_qkv,
    )
    from .gdn_state_io import gather_state_rows, scatter_state_rows_

    direct_ssm = (
        ops.recurrent is not None
        and _can_update_ssm_pool(x, ssm_state, indices, output)
        and (num_key_heads, num_value_heads, head_k_dim, head_v_dim) == (2, 6, 128, 128)
        and a.shape == b.shape == (8, 6)
    )
    if not direct_ssm:
        valid = indices > 0
    conv_rows = None
    indexed_conv = ops.convolution is not None and _can_update_conv_pool(x, conv_state, indices)
    if not indexed_conv:
        conv_rows = gather_state_rows(conv_state, indices)
    if ops.convolution is None:
        convolved = _decode_convolution(x, conv_rows, conv_weights, conv_bias, activation)
    elif indexed_conv:
        # Native UPDATE skips null slots. Initialize those lanes for the
        # subsequent per-lane Q/K normalization and scratch recurrence.
        convolved = ops.convolution(
            x.contiguous(),
            conv_state.transpose(-1, -2),
            conv_weights.T.contiguous(),
            bias=conv_bias,
            activation=activation,
            conv_state_indices=indices.clamp_min(0).contiguous(),
            null_block_id=0,
            out=torch.zeros_like(x, memory_format=torch.contiguous_format),
        )
        logger.info_once("Qwen3.8 GDN indexed convolution pool active (FULL [8])")
    else:
        # Omitting cache indices in dense UPDATE gives every lane a private
        # row, including lane 0. No graph padding index reaches the native op.
        conv_rows = conv_rows.transpose(-1, -2).contiguous()
        convolved = ops.convolution(
            x.contiguous(),
            conv_rows,
            conv_weights.T.contiguous(),
            bias=conv_bias,
            activation=activation,
            out=torch.empty_like(x, memory_format=torch.contiguous_format),
        )
        conv_rows = conv_rows.transpose(-1, -2).contiguous()

    fused_qkv = (
        direct_ssm
        and _fused_qkv_enabled()
        and convolved.device == x.device
        and convolved.shape == (8, 1280)
        and convolved.dtype == torch.bfloat16
        and convolved.is_contiguous()
    )
    if not fused_qkv:
        q, k, v = _prepare_qkv(
            convolved, num_key_heads, num_value_heads, head_k_dim, head_v_dim,
            expand_heads=ops.recurrent is None,
        )
    if not direct_ssm:
        state = gather_state_rows(ssm_state, indices)
        state = torch.where(valid[:, None, None, None], state, 0)
    if (
        _fused_gating_enabled()
        and is_forward_context_available()
        and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL
        and a.shape == b.shape == (8, 6)
        and A_log.shape == dt_bias.shape == (6,)
        and a.device.type == "npu"
        and a.device == b.device == A_log.device == dt_bias.device
        and a.dtype == b.dtype == torch.bfloat16
        and A_log.dtype == torch.float32
        and dt_bias.dtype in (torch.bfloat16, torch.float32)
    ):
        from .gdn_gating import fused_gating

        g, beta = fused_gating(a, b, A_log, dt_bias, packed_beta=packed_beta)
        logger.info_once("Qwen3.8 GDN fused gating active (FULL [8])")
    else:
        g, beta = _gating(a, b, A_log, dt_bias, packed_beta=packed_beta)
    if direct_ssm:
        from .gdn_direct_ssm import direct_ssm_recurrent

        prepared_qkv = None if fused_qkv else (
            q.to(x.dtype).contiguous(), k.to(x.dtype).contiguous(), v.contiguous(),
        )
        direct_ssm_recurrent(
            ops.recurrent,
            ssm_state,
            indices,
            beta.to(x.dtype).contiguous(),
            g.contiguous(),
            output,
            head_k_dim ** -0.5,
            prepared_qkv=prepared_qkv,
            convolved=convolved if fused_qkv else None,
        )
        if fused_qkv:
            logger.info_once("Qwen3.8 GDN fused QKV pack active (FULL [8])")
        logger.info_once("Qwen3.8 GDN direct FP32 SSM pool active (FULL [8])")
    else:
        if ops.recurrent is None:
            result, state = _decode_recurrence(q, k, v, state, g, beta, head_k_dim)
        else:
            n = x.shape[0]
            lane_ids = torch.arange(n, dtype=torch.int32, device=x.device)
            lengths = torch.ones(n + 1, dtype=torch.int32, device=x.device)
            lengths[:1].zero_()
            result = ops.recurrent(
                q.to(x.dtype).contiguous(),
                k.to(x.dtype).contiguous(),
                v.contiguous(),
                state,
                beta=beta.to(x.dtype).contiguous(),
                scale=head_k_dim ** -0.5,
                actual_seq_lengths=lengths,
                ssm_state_indices=lane_ids,
                g=g.contiguous(),
            )

        output.copy_(torch.where(valid[:, None, None], result, 0).to(output.dtype))

    if conv_rows is not None:
        scatter_state_rows_(conv_state, indices, conv_rows)
    if not direct_ssm:
        scatter_state_rows_(ssm_state, indices, state)
    if ops.convolution is not None:
        logger.info_once("Qwen3.8 GDN AscendC decode convolution active")
    if ops.recurrent is not None:
        logger.info_once("Qwen3.8 GDN AscendC decode recurrent active")
