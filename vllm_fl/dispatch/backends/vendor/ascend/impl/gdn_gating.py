# SPDX-License-Identifier: Apache-2.0
"""Fused Qwen3.8 GDN decode gates, with FP32 outputs."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _fused_gating_kernel(
    A, B, ALog, DtBias, G, Beta,
    ROWS: tl.constexpr, NH: tl.constexpr,
    AS0: tl.constexpr, AS1: tl.constexpr,
    BS0: tl.constexpr, BS1: tl.constexpr,
    LOG_STRIDE: tl.constexpr, BIAS_STRIDE: tl.constexpr,
    PACKED_BETA: tl.constexpr, TILE: tl.constexpr,
):
    index = tl.program_id(0) * TILE + tl.arange(0, TILE)
    row = index // NH
    head = index % NH
    valid = row < ROWS
    a = tl.load(A + row * AS0 + head * AS1, valid, other=0).to(tl.float32)
    b = tl.load(B + row * BS0 + head * BS1, valid, other=0).to(tl.float32)
    a_log = tl.load(ALog + head * LOG_STRIDE, valid, other=0).to(tl.float32)
    bias = tl.load(DtBias + head * BIAS_STRIDE, valid, other=0).to(tl.float32)

    x = a + bias
    z = tl.exp(-tl.abs(x))
    one_plus = 1.0 + z
    delta = one_plus - 1.0
    # Compensate the rounding of 1+z and retain the negative softplus tail.
    denom = tl.where(delta == 0.0, 1.0, delta)
    log1p = tl.where(delta == 0.0, z, tl.log(one_plus) * (z / denom))
    softplus = tl.where(x > 20.0, x, tl.maximum(x, 0.0) + log1p)
    g = -tl.exp(a_log) * softplus
    beta = tl.sigmoid(b)
    if PACKED_BETA:
        beta = beta.to(B.dtype.element_ty).to(tl.float32)

    tl.store(G + index, g, valid)
    tl.store(Beta + index, beta, valid)


def fused_gating(a, b, a_log, dt_bias, *, packed_beta):
    g = torch.empty(a.shape, dtype=torch.float32, device=a.device)
    beta = torch.empty_like(g)
    tile = 16
    _fused_gating_kernel[(triton.cdiv(a.numel(), tile),)](
        a, b, a_log, dt_bias, g, beta,
        *a.shape, *a.stride(), *b.stride(), a_log.stride(0), dt_bias.stride(0),
        packed_beta, tile, num_warps=1, enable_fp_fusion=False,
    )
    return g, beta
