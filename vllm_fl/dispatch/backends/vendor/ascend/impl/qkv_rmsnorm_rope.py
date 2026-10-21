# SPDX-License-Identifier: Apache-2.0
"""Ascend implementation of the functional QKV norm/partial-RoPE contract.

Inputs are read-only; outputs are flattened (q, k, v) tensors in input dtype.
Weights already include any model-specific offset (for example Gemma's +1).
Cache insertion and attention metadata are deliberately outside this contract.
"""

import torch


def qkv_rmsnorm_rope(
    qkv: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: float,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from .graph_fusion_ops import ensure_graph_fusion_ops_registered

    ensure_graph_fusion_ops_registered()
    return torch.ops.vllm.qkv_rmsnorm_rope(
        input=qkv,
        cos_sin_cache=cos_sin_cache,
        positions=positions,
        q_weight=q_weight,
        k_weight=k_weight,
        q_hidden_size=q_hidden_size,
        kv_hidden_size=kv_hidden_size,
        head_dim=head_dim,
        eps=eps,
        q_bias=q_bias,
        k_bias=k_bias,
    )


def qkv_rmsnorm_mrope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    eps: float,
    mrope_section: list[int],
    is_interleaved: bool,
    rope_dim: int | None = None,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
    has_gate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Preserve the separate four-output MRoPE contract, including gate/bias."""
    from .linearnorm import split_qkv_rmsnorm_mrope  # noqa: F401

    return torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(
        qkv, q_weight, k_weight, cos_sin, num_q_heads, num_kv_heads,
        head_size, eps, mrope_section, is_interleaved, rope_dim,
        q_bias, k_bias, has_gate,
    )
