# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged QSA decode for the runner's existing all-visible FULL graph bound."""

from __future__ import annotations

import torch
from flaggems_vllm.ops.qwen4.qsa_attention import _qsa_sparse_paged_gqa_split_reduce_kernel
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_KV_TILE = 16
_SPLITS = 4


@triton.jit
def _visible_partials(
    query, key, value, block_table, request_ids, positions, seq_lens,
    partial, maxima, sums,
    q_row, q_head, q_dim,
    k_page, k_token, k_head, k_dim,
    v_page, v_token, v_head, v_dim,
    table_row, table_col, req_stride, pos_stride, len_stride,
    n_requests: tl.constexpr, n_pages: tl.constexpr, table_width: tl.constexpr,
    page_size: tl.constexpr, n_heads: tl.constexpr, head_dim: tl.constexpr,
    group_size: tl.constexpr, head_tile: tl.constexpr, kv_tile: tl.constexpr,
    splits: tl.constexpr, max_visible: tl.constexpr, scale: tl.constexpr,
):
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)
    req = tl.load(request_ids + row * req_stride).to(tl.int32)
    valid_req = (req >= 0) & (req < n_requests)
    safe_req = tl.minimum(tl.maximum(req, 0), n_requests - 1)
    position = tl.load(positions + row * pos_stride).to(tl.int32)
    seq_len = tl.load(seq_lens + safe_req * len_stride).to(tl.int32)
    visible = tl.minimum(tl.minimum(position + 1, seq_len), max_visible)
    visible = tl.where(valid_req, tl.maximum(visible, 0), 0)

    # Partition the live 16-token tiles. All cache pages are multiples of this
    # width, so one tile shares one page lookup. Replay reads device lengths.
    tiles = (visible + kv_tile - 1) // kv_tile
    first = tiles * split // splits
    end = tiles * (split + 1) // splits
    head = tl.arange(0, head_tile)
    query_head = kv_head * group_size + head
    dim = tl.arange(0, head_dim)
    q = tl.load(
        query + row * q_row + query_head[:, None] * q_head + dim[None, :] * q_dim,
        mask=head[:, None] < group_size,
        other=0.0,
    )
    q = (q * scale * 1.4426950408889634).to(q.dtype)
    maximum = tl.full((head_tile,), -1.0e20, tl.float32)
    total = tl.zeros((head_tile,), tl.float32)
    acc = tl.zeros((head_tile, head_dim), tl.float32)
    lanes = tl.arange(0, kv_tile)
    for tile in tl.range(first, end):
        token = tile * kv_tile
        logical_page = token // page_size
        physical_page = tl.load(
            block_table + safe_req * table_row + tl.minimum(logical_page, table_width - 1) * table_col
        ).to(tl.int64)
        valid_page = (logical_page < table_width) & (physical_page >= 0) & (physical_page < n_pages)
        safe_page = tl.where(valid_page, physical_page, 0)
        valid = valid_page & (token + lanes < visible)
        page_tokens = token % page_size + lanes
        k = tl.load(
            key + safe_page * k_page + page_tokens[None, :] * k_token
            + kv_head * k_head + dim[:, None] * k_dim,
            mask=valid[None, :], other=0.0,
        )
        v = tl.load(
            value + safe_page * v_page + page_tokens[:, None] * v_token
            + kv_head * v_head + dim[None, :] * v_dim,
            mask=valid[:, None], other=0.0,
        )
        scores = tl.dot(q, k)
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(maximum, tl.max(scores, axis=1))
        prior = tl.math.exp2(maximum - next_max)
        probs = tl.where(valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0)
        acc = tl.dot(probs.to(v.dtype), v, acc=acc * prior[:, None])
        total = total * prior + tl.sum(probs, axis=1)
        maximum = next_max

    stats_index = (row * n_heads + query_head) * splits + split
    tl.store(maxima + stats_index, maximum, mask=head < group_size)
    tl.store(sums + stats_index, total, mask=head < group_size)
    tl.store(partial + stats_index[:, None] * head_dim + dim[None, :], acc, mask=head[:, None] < group_size)


def supports_visible_decode(query: torch.Tensor, key_cache: torch.Tensor) -> bool:
    return (
        query.device.type == "npu"
        and query.dtype == key_cache.dtype == torch.bfloat16
        and query.shape[-1] == key_cache.shape[-1] == 256
        and key_cache.shape[0] > 0
        and key_cache.shape[1] >= _KV_TILE
        and key_cache.shape[1] % _KV_TILE == 0
        and query.shape[1] % key_cache.shape[2] == 0
    )


def visible_paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
    gate: torch.Tensor,
    softmax_scale: float,
    max_visible: int,
) -> None:
    """Compute exact all-visible token membership using the caller's graph bound."""
    logger.info_once("FL QSA short FULL decode uses direct visible paged attention")
    rows, heads, dim = query.shape
    kv_heads = key_cache.shape[2]
    group_size = heads // kv_heads
    partial = torch.empty((rows, heads, _SPLITS, dim), dtype=torch.float32, device=query.device)
    maxima = torch.empty((rows, heads, _SPLITS), dtype=torch.float32, device=query.device)
    sums = torch.empty_like(maxima)
    _visible_partials[(rows, kv_heads, _SPLITS)](
        query, key_cache, value_cache, block_table, token_to_req, logical_positions, seq_lens,
        partial, maxima, sums, *query.stride(), *key_cache.stride(), *value_cache.stride(),
        *block_table.stride(), token_to_req.stride(0), logical_positions.stride(0), seq_lens.stride(0),
        block_table.shape[0], key_cache.shape[0], block_table.shape[1], key_cache.shape[1],
        heads, dim, group_size, max(8, triton.next_power_of_2(group_size)), _KV_TILE,
        _SPLITS, max_visible, softmax_scale, num_warps=4, num_stages=2,
    )
    _qsa_sparse_paged_gqa_split_reduce_kernel[(rows, heads)](
        partial, maxima, sums, gate, output,
        *partial.stride(), *maxima.stride(), *sums.stride(), *gate.stride(), *output.stride(), rows,
        NUM_QUERY_HEADS=heads, NUM_SPLITS=_SPLITS, HEAD_DIM=dim,
        BLOCK_S=_SPLITS, BLOCK_D=dim, APPLY_GATE=True, num_warps=4, num_stages=2,
    )
