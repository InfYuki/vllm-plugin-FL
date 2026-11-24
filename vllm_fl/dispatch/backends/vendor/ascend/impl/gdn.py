# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""Adapt upstream GDN metadata to FL-dispatched numerical operators."""

import os
from dataclasses import dataclass

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.utils.torch_utils import _encode_layer_name
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)

from vllm_fl.dispatch import CachedOp

from .gdn_native_decode import (
    NativeDecodeOps,
    gdn_decode_native,
    get_native_decode_ops,
)
from .gdn_phase6 import (
    Phase6PrefillMetadata,
    build_phase6_metadata,
    get_phase6_op,
    gdn_prefill_phase6,
)

_gdn_decode = CachedOp("gdn_decode")
_gdn_prefill = CachedOp("gdn_prefill")


@dataclass
class AscendGDNMetadata(GDNAttentionMetadata):
    prefill_token_ranges: tuple[tuple[int, int], ...] = ()
    phase6: Phase6PrefillMetadata | None = None


def _native_gdn_supported(vllm_config):
    model = vllm_config.model_config
    return (
        getattr(getattr(model, "hf_config", None), "model_type", None) == "qwen4_exp"
        and model.dtype == torch.bfloat16
        and vllm_config.quant_config is None
        and vllm_config.speculative_config is None
        and vllm_config.cache_config.mamba_cache_mode == "align"
        and vllm_config.parallel_config.prefill_context_parallel_size == 1
    )


def _native_prefill_requested(vllm_config):
    return (
        os.environ.get("VLLM_FL_QWEN38_NATIVE_PREFILL") == "1"
        and _native_gdn_supported(vllm_config)
    )


class AscendGDNMetadataBuilder(GDNAttentionMetadataBuilder):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._native_prefill = _native_prefill_requested(vllm_config)

    def build(self, common_prefix_len, common_attn_metadata, *args, **kwargs):
        metadata = super().build(common_prefix_len, common_attn_metadata, *args, **kwargs)
        if not metadata.num_prefills or metadata.spec_sequence_masks is not None:
            return metadata

        start = metadata.num_decodes
        stop = start + metadata.num_prefills + 1
        starts = common_attn_metadata.query_start_loc_cpu[start:stop].tolist()
        phase6 = (
            build_phase6_metadata(starts)
            if self._native_prefill
            and len(starts) == metadata.num_prefills + 1
            and starts[0] == metadata.num_decode_tokens
            and starts[-1] - starts[0] == metadata.num_prefill_tokens
            else None
        )
        return AscendGDNMetadata(
            **vars(metadata),
            prefill_token_ranges=tuple(zip(starts, starts[1:])),
            phase6=phase6,
        )


class AscendGDNBackend(GDNAttentionBackend):
    @staticmethod
    def get_builder_cls():
        return AscendGDNMetadataBuilder


class AscendQwenGatedDeltaNetAttention(QwenGatedDeltaNetAttention):
    """Non-speculative Ascend GDN with device-resident cache updates."""

    def __init__(
        self, config, vllm_config, prefix="", gqa_interleaved_layout=False,
        reduce_results=True,
    ):
        super().__init__(
            config, vllm_config, prefix, gqa_interleaved_layout, reduce_results
        )
        model = vllm_config.model_config
        self._use_native_prefill = _native_prefill_requested(vllm_config)
        self._use_fused_rmsnorm_gate = (
            os.environ.get("VLLM_FL_QWEN38_FUSED_GDN_RMSNORM_GATE") == "1"
            and _native_gdn_supported(vllm_config)
        )
        if self._use_fused_rmsnorm_gate:
            from . import gdn_rmsnorm_gate  # noqa: F401 — register the opaque op
        self._native_decode_mode = None
        if (
            os.environ.get("VLLM_FL_QWEN38_NATIVE_DECODE") == "1"
            and _native_gdn_supported(vllm_config)
        ):
            mode = os.environ.get("VLLM_FL_QWEN38_NATIVE_DECODE_OPS", "both")
            if mode not in ("both", "conv", "recurrent"):
                raise ValueError("VLLM_FL_QWEN38_NATIVE_DECODE_OPS must be both, conv, or recurrent")
            self._native_decode_mode = mode
        self._use_fused_prefill = (
            not self._use_native_prefill
            and os.environ.get("VLLM_FL_QWEN38_FUSED_GDN_PREFILL") == "1"
            and getattr(getattr(model, "hf_config", None), "model_type", None)
            == "qwen4_exp"
            and vllm_config.speculative_config is None
        )

    def _output_projection(self, core_attn_out, z):
        if not self._use_fused_rmsnorm_gate:
            return super()._output_projection(core_attn_out, z)

        # The compiled model can be first traced in NONE mode. Defer the
        # runtime graph-mode and tensor checks to the opaque operator.
        normalized = torch.ops.vllm.fl_qwen38_gdn_rmsnorm_gate(
            core_attn_out, z, self.norm.weight, _encode_layer_name(self.prefix),
        )
        output, _ = self.out_proj(normalized.flatten(-2))
        return output

    def get_attn_backend(self):
        return AscendGDNBackend

    def _warmup_prefill_kernels(self, *args, **kwargs):
        # There are no Triton prefill kernels in this reference implementation.
        return

    def _operator_kwargs(self):
        return dict(
            conv_weights=self.conv1d.weight.view(self.conv1d.weight.size(0), -1),
            conv_bias=self.conv1d.bias,
            activation=self.activation,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            num_key_heads=self.num_k_heads // self.tp_size,
            num_value_heads=self.num_v_heads // self.tp_size,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
        )

    def _forward_core(self, mixed_qkv, b, a, core_attn_out):
        context = get_forward_context()
        metadata = context.attn_metadata
        if metadata is None:
            return
        metadata = metadata[self.prefix]
        if metadata.spec_sequence_masks is not None:
            raise NotImplementedError(
                "Ascend reference GDN does not support speculative decoding"
            )
        conv_state, ssm_state = self.kv_cache
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        indices = metadata.non_spec_state_indices_tensor
        num_decode = metadata.num_decode_tokens
        op_kwargs = self._operator_kwargs()
        if metadata.num_prefills == 0:
            n = metadata.num_actual_tokens
            native_mode = getattr(self, "_native_decode_mode", None)
            if (
                native_mode is not None
                and context.cudagraph_runtime_mode in (CUDAGraphMode.NONE, CUDAGraphMode.FULL)
                and 0 < n <= 8
                and indices is not None
                and indices.ndim == 1
                and indices.shape[0] >= n
                and indices.dtype in (torch.int32, torch.int64)
                and mixed_qkv.device.type == "npu"
                and mixed_qkv.dtype == core_attn_out.dtype == conv_state.dtype == torch.bfloat16
                and ssm_state.dtype == torch.float32
                and conv_state.shape[1:] == (1280, 3)
                and ssm_state.shape[1:] == (6, 128, 128)
                and op_kwargs["conv_weights"].shape == (1280, 4)
                and (
                    op_kwargs["num_key_heads"],
                    op_kwargs["num_value_heads"],
                    op_kwargs["head_k_dim"],
                    op_kwargs["head_v_dim"],
                ) == (2, 6, 128, 128)
            ):
                available = get_native_decode_ops()
                ops = NativeDecodeOps(
                    convolution=available.convolution if native_mode != "recurrent" else None,
                    recurrent=available.recurrent if native_mode != "conv" else None,
                )
                if ops.convolution is not None or ops.recurrent is not None:
                    gdn_decode_native(
                        mixed_qkv[:n],
                        a[:n],
                        b[:n],
                        indices[:n],
                        conv_state,
                        ssm_state,
                        core_attn_out[:n],
                        ops=ops,
                        packed_beta=self.enable_packed_recurrent_decode,
                        **op_kwargs,
                    )
                    return
            _gdn_decode(
                mixed_qkv[:n],
                a[:n],
                b[:n],
                indices[:n],
                conv_state,
                ssm_state,
                core_attn_out[:n],
                packed_beta=self.enable_packed_recurrent_decode,
                **op_kwargs,
            )
            return
        if num_decode:
            _gdn_decode(
                mixed_qkv[:num_decode],
                a[:num_decode],
                b[:num_decode],
                indices[:num_decode],
                conv_state,
                ssm_state,
                core_attn_out[:num_decode],
                packed_beta=False,
                **op_kwargs,
            )

        # The scheduler already has query boundaries on CPU. Cache slots and
        # initial-state flags remain on device, including padding.
        if (
            getattr(self, "_use_native_prefill", False)
            and context.skip_compiled
            and context.cudagraph_runtime_mode == CUDAGraphMode.NONE
            and metadata.phase6 is not None
            and mixed_qkv.device.type == "npu"
            and mixed_qkv.dtype == torch.bfloat16
            and core_attn_out.dtype == torch.bfloat16
            and ssm_state.dtype == torch.float32
            and (
                op_kwargs["num_key_heads"],
                op_kwargs["num_value_heads"],
                op_kwargs["head_k_dim"],
                op_kwargs["head_v_dim"],
            ) == (2, 6, 128, 128)
            and not torch.npu.is_current_stream_capturing()
        ):
            phase6_op = get_phase6_op()
            if phase6_op is not None:
                end = num_decode + metadata.num_prefill_tokens
                gdn_prefill_phase6(
                    mixed_qkv[num_decode:end],
                    a[num_decode:end],
                    b[num_decode:end],
                    metadata.prefill_state_indices[: metadata.num_prefills],
                    metadata.prefill_has_initial_state[: metadata.num_prefills],
                    conv_state,
                    ssm_state,
                    core_attn_out[num_decode:end],
                    metadata.phase6,
                    phase6_op,
                    **op_kwargs,
                )
                return
        if self._use_fused_prefill and context.skip_compiled:
            op_kwargs["use_fused_chunk"] = True
        for seq, (start, end) in enumerate(metadata.prefill_token_ranges):
            if start == end:
                continue
            _gdn_prefill(
                mixed_qkv[start:end],
                a[start:end],
                b[start:end],
                metadata.prefill_state_indices[seq : seq + 1],
                metadata.prefill_has_initial_state[seq : seq + 1],
                conv_state,
                ssm_state,
                core_attn_out[start:end],
                **op_kwargs,
            )
