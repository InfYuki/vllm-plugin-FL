# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Tensor GDN operators with explicit weights, states and mutation contracts.

Convolution state uses [slots, channels, history]; recurrent state uses
[slots, value_heads, value_dim, key_dim]. Decode preserves padding slot zero.
Both operators update the supplied state/output buffers in place. They own
numerical computation, not model metadata, cache allocation or graph lifetime.
"""

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

from vllm.model_executor.layers.mamba.ops.cpu.causal_conv1d import (
    causal_conv1d_update_torch,
)

from vllm_fl.kv_cache.ascend.state import state_row_indices

from .gdn_state_io import gather_state_rows, scatter_state_rows_

logger = init_logger(__name__)


def _prepare_qkv(
    x, num_key_heads, num_value_heads, head_k_dim, head_v_dim, *, expand_heads=True
):
    q, k, v = torch.split(
        x,
        [num_key_heads * head_k_dim] * 2 + [num_value_heads * head_v_dim],
        dim=-1,
    )
    q = q.reshape(-1, num_key_heads, head_k_dim)
    k = k.reshape_as(q)
    v = v.reshape(-1, num_value_heads, head_v_dim)
    q = q.float() * torch.rsqrt(q.float().square().sum(-1, keepdim=True) + 1e-6)
    k = k.float() * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)
    repeats = num_value_heads // num_key_heads
    if expand_heads and repeats != 1:
        # Preserve grouped head order without the ACLNN scalar-repeat op,
        # whose op-info recording crashes during FULL static capture.
        q = q.unsqueeze(2).expand(-1, -1, repeats, -1).reshape(
            -1, num_value_heads, head_k_dim
        )
        k = k.unsqueeze(2).expand(-1, -1, repeats, -1).reshape(
            -1, num_value_heads, head_k_dim
        )
    return q, k, v


def _gating(a, b, A_log, dt_bias, *, packed_beta):
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float(), threshold=20)
    beta = b.float().sigmoid()
    if packed_beta:
        beta = beta.to(b.dtype).float()
    return g, beta


def _gather_rows(state, indices):
    if state.is_contiguous():
        return state.index_select(0, indices)
    # Index contiguous storage directly: indexing a strided NPU tensor can
    # materialize the entire logical cache, even for a single active row.
    storage = _storage_view(state)
    return storage.index_select(0, state_row_indices(state, indices)).view(
        indices.numel(), *state.shape[1:]
    )


def _storage_view(state):
    return state.as_strided(
        (state.untyped_storage().nbytes() // state.element_size(),),
        (1,),
        storage_offset=0,
    )


def _scatter_rows(state, indices, rows):
    if state.is_contiguous():
        state.index_copy_(0, indices, rows)
    else:
        _storage_view(state).index_copy_(
            0, state_row_indices(state, indices), rows.contiguous().flatten()
        )


def _decode_convolution(x, conv_rows, conv_weights, conv_bias, activation):
    return causal_conv1d_update_torch(
        x.unsqueeze(-1), conv_rows, conv_weights, conv_bias, activation
    ).squeeze(-1)


def _decode_recurrence(q, k, v, state, g, beta, head_k_dim):
    state = state * g.exp()[..., None, None]
    delta = (v.float() - (state * k.unsqueeze(-2)).sum(-1)) * beta[..., None]
    state = state + delta.unsqueeze(-1) * k.unsqueeze(-2)
    result = (state * (q * head_k_dim**-0.5).unsqueeze(-2)).sum(-1)
    return result, state


def gdn_decode(
    x,
    a,
    b,
    indices,
    conv_state,
    ssm_state,
    output,
    *,
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
    """Update one token per lane; indices <=0 are inactive padded lanes."""
    indices = indices.long()
    valid = indices > 0
    slots = indices.clamp_min(0)
    conv_old = _gather_rows(conv_state, slots)
    conv_new = conv_old.clone()
    x = _decode_convolution(x, conv_new, conv_weights, conv_bias, activation)
    _scatter_rows(
        conv_state, slots, torch.where(valid[:, None, None], conv_new, conv_old)
    )
    q, k, v = _prepare_qkv(x, num_key_heads, num_value_heads, head_k_dim, head_v_dim)
    old = _gather_rows(ssm_state, slots)
    state = torch.where(valid[:, None, None, None], old, 0).float()
    g, beta = _gating(a, b, A_log, dt_bias, packed_beta=packed_beta)
    result, state = _decode_recurrence(q, k, v, state, g, beta, head_k_dim)
    output.copy_(torch.where(valid[:, None, None], result, 0).to(output.dtype))
    _scatter_rows(
        ssm_state,
        slots,
        torch.where(valid[:, None, None, None], state.to(old.dtype), old),
    )


def _prefill_convolution(x, conv_rows, use_history, conv_weights, conv_bias, activation):
    """Convolve one sequence and update its gathered history row."""
    history_len = conv_weights.shape[1] - 1
    history = torch.where(use_history[:, None, None], conv_rows[..., :history_len], 0)
    seq_x = x.to(conv_rows.dtype).T.unsqueeze(0)
    conv_input = torch.cat((history, seq_x), dim=-1).to(conv_weights.dtype)
    convolved = F.conv1d(
        conv_input,
        conv_weights.unsqueeze(1),
        conv_bias,
        groups=conv_weights.shape[0],
    ).to(conv_rows.dtype)
    if activation in ("silu", "swish") or activation is True:
        convolved = F.silu(convolved)
    convolved = convolved.squeeze(0).T.to(x.dtype)
    if history_len:
        conv_rows[..., :history_len].copy_(conv_input[..., -history_len:])
    return convolved


def gdn_prefill(
    x,
    a,
    b,
    cache_index,
    has_initial_state,
    conv_state,
    ssm_state,
    output,
    *,
    conv_weights,
    conv_bias,
    activation,
    A_log,
    dt_bias,
    num_key_heads,
    num_value_heads,
    head_k_dim,
    head_v_dim,
    use_fused_chunk=False,
):
    """Update one prefill sequence with device-resident cache slot selection."""
    use_fused_chunk = (
        use_fused_chunk
        and x.device.type == "npu"
        and x.dtype == torch.bfloat16
        and (num_key_heads, num_value_heads, head_k_dim, head_v_dim) == (2, 6, 128, 128)
    )
    valid = cache_index > 0
    use_history = valid & has_initial_state
    conv_rows = gather_state_rows(conv_state, cache_index)
    convolved = _prefill_convolution(
        x, conv_rows, use_history, conv_weights, conv_bias, activation
    )
    if conv_weights.shape[1] > 1:
        scatter_state_rows_(conv_state, cache_index, conv_rows)

    g, beta = _gating(a, b, A_log, dt_bias, packed_beta=False)
    if use_fused_chunk:
        native_beta = beta.to(x.dtype).contiguous()
        # The A3 kernel can overflow outside its documented gate range.
        # Only eager prefill may do this scalar check.
        use_fused_chunk = bool(
            ((g >= -1) & (g <= 0) & (native_beta > 0) & (native_beta < 1)).all()
        )
        if not use_fused_chunk:
            logger.info_once("Qwen3.8 GDN prefill used reference for out-of-range gates")

    q, k, v = _prepare_qkv(
        convolved, num_key_heads, num_value_heads, head_k_dim, head_v_dim,
        expand_heads=not use_fused_chunk,
    )
    q, k = q.to(x.dtype), k.to(x.dtype)
    state_rows = gather_state_rows(ssm_state, cache_index)
    initial = torch.where(use_history[:, None, None, None], state_rows, 0)

    if use_fused_chunk:
        import torch_npu

        # CANN takes compact TND heads and [slot, value_head, value_dim, key_dim]
        # BF16 state. The serving cache retains its configured storage dtype.
        lengths = torch.full((1,), x.shape[0], dtype=torch.int32, device=x.device)
        result, final = torch_npu.npu_chunk_gated_delta_rule(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            beta=native_beta,
            initial_state=initial.to(x.dtype).contiguous(),
            actual_seq_lengths=lengths,
            scale=head_k_dim ** -0.5,
            g=g.contiguous(),
        )
        logger.info_once("Qwen3.8 GDN CANN fused prefill active")
    else:
        result, final = torch_chunk_gated_delta_rule(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            g.unsqueeze(0),
            beta.unsqueeze(0),
            initial_state=initial.float().transpose(-1, -2),
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )
        result = result.squeeze(0)
        final = final.transpose(-1, -2)

    output.copy_(torch.where(valid[:, None, None], result, output))
    scatter_state_rows_(ssm_state, cache_index, final.to(ssm_state.dtype).contiguous())
