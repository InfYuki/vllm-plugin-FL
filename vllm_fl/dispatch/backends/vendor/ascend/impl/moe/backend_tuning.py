# SPDX-License-Identifier: Apache-2.0
"""DSv4-only, host-metadata bounds for the existing A3 fused MC2 operator.

This changes workspace capacity, not routing or token dropping. All ranks
derive the bound from DP-synchronized padded extents; no device counts are read.
"""
import torch
from vllm_fl.ascend_forward_context import _EXTRA_CTX, get_mc2_tokens_capacity


def bounded_decode_capacity(fused_input, moe_config, ceiling):
    if not _EXTRA_CTX.dsv4_backend_opt:
        return ceiling
    weights = fused_input.weights.w1
    first_weight = weights[0] if isinstance(weights, list) else weights
    if first_weight.dtype != torch.int8:
        return ceiling
    threshold = get_mc2_tokens_capacity()
    rows = max(int(_EXTRA_CTX.max_tokens_across_dp or 0),
               int(_EXTRA_CTX.padded_num_tokens or 0), 1)
    if threshold is None or rows > threshold:
        # Keep the original configured prefill/large-batch capacity.
        return ceiling
    if fused_input.hidden_states.shape[0] > rows:
        raise RuntimeError("MC2 local rows exceed the synchronized workspace bound")
    top_k = fused_input.topk_ids.shape[-1]
    experts = moe_config.num_local_experts
    # Every source EP rank has at most 'rows' inputs and each input has K
    # assignments. Assume ALL assignments arrive on one receiver (no balance
    # assumption). Reserve two 128-row M tiles per local expert for padding.
    bound = rows * moe_config.ep_size * top_k + experts * 256
    bound = max(256, ((bound + 255) // 256) * 256)
    return min(ceiling, bound)
