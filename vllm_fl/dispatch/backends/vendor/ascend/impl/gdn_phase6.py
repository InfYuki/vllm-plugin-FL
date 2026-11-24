# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""Packed AscendC Phase6 GDN prefill with FP32 recurrent state."""

from dataclasses import dataclass
from functools import lru_cache

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)
_CHUNK_SIZE = 64


@dataclass(frozen=True)
class Phase6PrefillMetadata:
    cu_seqlens: tuple[int, ...]
    chunk_indices: tuple[int, ...]


def build_phase6_metadata(starts):
    """Build the kernel's flattened (sequence, local chunk) pairs on the host."""
    lengths = [end - start for start, end in zip(starts, starts[1:])]
    if not lengths or min(lengths) <= 0:
        # Phase6 on A3 requires strictly increasing sequence boundaries.
        return None
    return Phase6PrefillMetadata(
        cu_seqlens=tuple(start - starts[0] for start in starts),
        chunk_indices=tuple(
            value
            for seq, length in enumerate(lengths)
            for chunk in range((length + _CHUNK_SIZE - 1) // _CHUNK_SIZE)
            for value in (seq, chunk)
        ),
    )


@lru_cache(maxsize=1)
def get_phase6_op():
    try:
        from fla_npu.ops.ascendc import chunk_gated_delta_rule_fwd
    except (ImportError, OSError, RuntimeError) as exc:
        logger.warning("Qwen3.8 Phase6 prefill unavailable: %s", exc)
        return None
    if not callable(chunk_gated_delta_rule_fwd):
        logger.warning("Qwen3.8 Phase6 prefill unavailable: invalid wheel entry point")
        return None
    return chunk_gated_delta_rule_fwd


def gdn_prefill_phase6(
    x,
    a,
    b,
    indices,
    has_initial_state,
    conv_state,
    ssm_state,
    output,
    metadata,
    phase6_op,
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
):
    from .gdn_reference import _gating, _prefill_convolution, _prepare_qkv
    from .gdn_state_io import gather_state_rows, scatter_state_rows_

    ranges = tuple(zip(metadata.cu_seqlens, metadata.cu_seqlens[1:]))
    valid = indices > 0
    use_history = valid & has_initial_state
    conv_rows = gather_state_rows(conv_state, indices)
    convolved_rows = [
        _prefill_convolution(
            x[start:end], conv_rows[seq : seq + 1], use_history[seq : seq + 1],
            conv_weights, conv_bias, activation,
        )
        for seq, (start, end) in enumerate(ranges)
    ]
    convolved = convolved_rows[0] if len(ranges) == 1 else torch.cat(convolved_rows)

    q, k, v = _prepare_qkv(
        convolved, num_key_heads, num_value_heads, head_k_dim, head_v_dim,
        expand_heads=False,
    )
    g, beta = _gating(a, b, A_log, dt_bias, packed_beta=False)
    state_rows = gather_state_rows(ssm_state, indices)
    initial = torch.where(use_history[:, None, None, None], state_rows, 0)
    # Phase6 takes compact BNSD QKV and [sequence, head, key, value] state.
    initial = initial.transpose(-1, -2).contiguous()
    result, final, *_ = phase6_op(
        q.transpose(0, 1).unsqueeze(0).to(x.dtype).contiguous(),
        k.transpose(0, 1).unsqueeze(0).to(x.dtype).contiguous(),
        v.transpose(0, 1).unsqueeze(0).contiguous(),
        g.unsqueeze(0).contiguous(),
        beta.to(x.dtype).unsqueeze(0).contiguous(),
        initial_state=initial,
        output_final_state=True,
        chunk_size=_CHUNK_SIZE,
        cu_seqlens=metadata.cu_seqlens,
        chunk_indices=metadata.chunk_indices,
        scale=head_k_dim ** -0.5,
        layout="BNSD",
        use_exp2=False,
        use_qk_l2norm_in_kernel=False,
        allow_neg_eigval=False,
        disable_recompute=True,
        state_v_first=False,
    )
    if (
        result.shape != (1, *output.shape)
        or result.dtype != output.dtype
        or final is None
        or final.shape != initial.shape
        or final.dtype != torch.float32
    ):
        raise RuntimeError("Qwen3.8 Phase6 returned an incompatible output or state")

    result = result.squeeze(0)
    for seq, (start, end) in enumerate(ranges):
        out = output[start:end]
        out.copy_(torch.where(valid[seq : seq + 1, None, None], result[start:end], out))
    if conv_weights.shape[1] > 1:
        scatter_state_rows_(conv_state, indices, conv_rows)
    scatter_state_rows_(ssm_state, indices, final.transpose(-1, -2).contiguous())
    logger.info_once("Qwen3.8 GDN AscendC Phase6 prefill active")
