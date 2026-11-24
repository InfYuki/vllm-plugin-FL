# SPDX-License-Identifier: Apache-2.0
"""Eager Ascend QSA attention over FL's expanded logical-token indices.

The prefill phase and logical KV threshold follow vLLM-Ascend v0.32.0rc1
(`vllm_ascend/models/qwen4_exp/attention_dispatch.py`, commit 24487f4f).
The long-context path keeps FL's paged sparse attention and its selection
semantics, including the tail after a full set of selected groups.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from os import getenv as _gather_getenv

import torch
from vllm.triton_utils import tl, triton
from vllm.logger import init_logger

logger = init_logger(__name__)
_GATHER_LAYOUTS_REPORTED: set[tuple] = set()

FIA_MAX_LOGICAL_KV_LENGTH = 24 * 1024
_FIA_PAGE_SIZE = 128
_FIA_MASK_ALIGNMENT = 32
_FIA_MASK_BUDGET_BYTES = 64 * 1024 * 1024
_FIA_MAX_QUERY_TILE = 4096


@dataclass(frozen=True)
class _Request:
    index: int
    start: int
    end: int
    kv_length: int
    use_fia: bool
    is_prefill: bool

    @property
    def has_cached_prefix(self) -> bool:
        return self.kv_length != self.end - self.start


@dataclass(frozen=True)
class QSARequestPlan:
    requests: tuple[_Request, ...]
    num_actual_tokens: int


def build_qsa_request_plan(
    query_start_loc_cpu: torch.Tensor | None,
    seq_lens_cpu: torch.Tensor | None,
    num_actual_tokens: int,
    is_prefilling_cpu: torch.Tensor | None,
) -> QSARequestPlan | None:
    """Build one request plan from scheduler metadata shared across QSA layers."""
    inputs = (query_start_loc_cpu, seq_lens_cpu, is_prefilling_cpu)
    if any(t is None or t.device.type != "cpu" or t.ndim != 1 for t in inputs):
        return None
    starts = query_start_loc_cpu.tolist()
    lengths = seq_lens_cpu.tolist()
    phases = is_prefilling_cpu.tolist()
    if num_actual_tokens <= 0 or not starts or starts[0] != 0:
        return None
    if len(lengths) < len(starts) - 1 or len(phases) < len(starts) - 1:
        return None

    requests = []
    covered = 0
    has_prefill = False
    for index, (start, end) in enumerate(zip(starts, starts[1:])):
        if start != covered or end < start:
            return None
        end = min(end, num_actual_tokens)
        query_length = end - start
        if query_length:
            kv_length = int(lengths[index])
            if kv_length < query_length:
                return None
            is_prefill = bool(phases[index])
            has_prefill |= is_prefill
            cached_prefix = kv_length - query_length
            use_fia = is_prefill and cached_prefix < FIA_MAX_LOGICAL_KV_LENGTH
            requests.append(_Request(index, start, end, kv_length, use_fia, is_prefill))
        covered = end
        if covered == num_actual_tokens:
            break
    if covered != num_actual_tokens or not has_prefill:
        return None
    return QSARequestPlan(tuple(requests), num_actual_tokens)


@dataclass(frozen=True)
class _FIACache:
    key: torch.Tensor
    value: torch.Tensor
    logical_page_size: int
    block_stride: int

    def table(self, logical_table: torch.Tensor, kv_length: int) -> torch.Tensor:
        """Map FL blocks to dense 128-token FIA pages without copying K/V."""
        pages_per_block = self.logical_page_size // _FIA_PAGE_SIZE
        logical_pages = (kv_length + self.logical_page_size - 1) // self.logical_page_size
        fia_pages = (kv_length + _FIA_PAGE_SIZE - 1) // _FIA_PAGE_SIZE
        table = logical_table[:, :logical_pages].clamp_min(0).to(torch.int32)
        table = table * self.block_stride
        if pages_per_block == 1:
            return table.contiguous()
        offsets = torch.arange(pages_per_block, dtype=torch.int32, device=table.device)
        return (table.unsqueeze(-1) + offsets).flatten(1)[:, :fia_pages].contiguous()


def _paged_fia_cache(key_cache: torch.Tensor, value_cache: torch.Tensor) -> _FIACache | None:
    """Adapt allocator-backed FL [blocks, tokens, KV heads, dim] cache views."""
    if key_cache.ndim != 4 or key_cache.shape != value_cache.shape:
        return None
    if key_cache.stride() != value_cache.stride():
        return None
    blocks, tokens, heads, dim = key_cache.shape
    if min(blocks, tokens, heads, dim) <= 0 or tokens % _FIA_PAGE_SIZE:
        return None
    row_elements = heads * dim
    page_elements = _FIA_PAGE_SIZE * row_elements
    block_stride = key_cache.stride(0)
    if (
        key_cache.stride(3) != 1
        or (heads != 1 and key_cache.stride(2) != dim)
        or key_cache.stride(1) != row_elements
        or block_stride < tokens * row_elements
        or block_stride % page_elements
    ):
        return None

    # Include allocator gaps as unreferenced FIA pages. Each view ends at its
    # final logical K or V block, so it stays inside the original storage.
    stride_pages = block_stride // page_elements
    physical_pages = (blocks - 1) * stride_pages + tokens // _FIA_PAGE_SIZE
    shape = (physical_pages, _FIA_PAGE_SIZE, row_elements)
    strides = (page_elements, row_elements, 1)
    return _FIACache(
        key_cache.as_strided(shape, strides),
        value_cache.as_strided(shape, strides),
        tokens,
        stride_pages,
    )


@triton.jit
def _gather_fia_page(
    Key, Value, Table, DenseKey, DenseValue,
    TOKENS: tl.constexpr, PAGE: tl.constexpr, WIDTH: tl.constexpr,
    BLOCKS: tl.constexpr, KEY_BLOCK: tl.constexpr, VALUE_BLOCK: tl.constexpr,
    KEY_ROW: tl.constexpr, VALUE_ROW: tl.constexpr,
    TABLE_STRIDE: tl.constexpr, ROWS: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    channel = tl.arange(0, WIDTH)
    page = tl.load(Table + (row // PAGE) * TABLE_STRIDE, row < TOKENS, other=0)
    page = tl.maximum(page, 0).to(tl.int64)
    in_page = (row % PAGE).to(tl.int64)
    mask = ((row < TOKENS) & (page < BLOCKS))[:, None]
    key_offset = (page * KEY_BLOCK + in_page * KEY_ROW)[:, None] + channel[None, :]
    value_offset = (page * VALUE_BLOCK + in_page * VALUE_ROW)[:, None] + channel[None, :]
    out = row[:, None] * WIDTH + channel[None, :]
    key = tl.load(Key + key_offset, mask, other=0)
    value = tl.load(Value + value_offset, mask, other=0)
    tl.store(DenseKey + out, key, row[:, None] < TOKENS)
    tl.store(DenseValue + out, value, row[:, None] < TOKENS)


def _gather_fia_cache(
    key: torch.Tensor, value: torch.Tensor, table: torch.Tensor, kv_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    # FL align can choose a 400-token block, which cannot be reinterpreted
    # as a 128-token FIA page. Materialize only this request's visible K/V.
    page_size = key.shape[1]
    page_count = (kv_length + page_size - 1) // page_size
    width = key.shape[2] * key.shape[3]
    standard_layout = (
        key.stride()[1:] == value.stride()[1:] == (width, key.shape[3], 1)
    )
    strided_gather = _gather_getenv("VLLM_FL_QWEN38_QSA_FIA_STRIDED_GATHER") == "1"
    strided_layout = (
        strided_gather
        and key.shape[2] == 1
        and key.stride(3) == value.stride(3) == 1
        and table.ndim == 2
        and table.shape[0] > 0
        and 0 < page_count <= table.shape[1]
        and key.shape[0] > 0
        and value.device == table.device == key.device
        and table.dtype in (torch.int32, torch.int64)
    )
    use_fast_gather = (
        key.device.type == "npu"
        and key.shape == value.shape
        and key.dtype == value.dtype == torch.bfloat16
        and (standard_layout or strided_layout)
        and triton.next_power_of_2(width) == width
    )
    if strided_gather:
        # Layout-only logging avoids inspecting device data or synchronizing.
        path = "index_select"
        if use_fast_gather:
            path = "standard" if standard_layout else "strided"
        layout = (path, tuple(key.shape), tuple(value.shape), key.stride(), value.stride())
        if layout not in _GATHER_LAYOUTS_REPORTED:
            _GATHER_LAYOUTS_REPORTED.add(layout)
            logger.info(
                "Qwen3.8 strided FIA gather candidate path=%s key_shape=%s value_shape=%s "
                "key_stride=%s value_stride=%s key_offset=%s value_offset=%s "
                "table_shape=%s table_stride=%s kv_length=%s kernel=_gather_fia_page",
                *layout, key.storage_offset(), value.storage_offset(),
                tuple(table.shape), table.stride(), kv_length,
            )
    if use_fast_gather:
        # IndexSelect on a gapped view can materialize the whole logical
        # pool first. Each program here reads just the requested KV rows.
        dense_key = torch.empty((kv_length, *key.shape[2:]), device=key.device, dtype=key.dtype)
        dense_value = torch.empty_like(dense_key)
        rows = 16
        _gather_fia_page[(triton.cdiv(kv_length, rows),)](
            key, value, table, dense_key, dense_value,
            kv_length, page_size, width, key.shape[0],
            key.stride(0), value.stride(0), key.stride(1), value.stride(1),
            table.stride(1), rows,
        )
        return dense_key, dense_value
    pages = table[0, :page_count].clamp_min(0).to(torch.int64)
    return (
        key.index_select(0, pages).flatten(0, 1)[:kv_length],
        value.index_select(0, pages).flatten(0, 1)[:kv_length],
    )


def _selected_token_mask(
    logical_indices: torch.Tensor | None,
    visible_lens: torch.Tensor,
    mask_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = (logical_indices >= 0) & (logical_indices < visible_lens[:, None])
    has_selection = valid.any(dim=-1)
    # Send invalid IDs to a scratch column. Empty rows temporarily attend to
    # token zero so FIA need not define fully masked softmax behavior.
    scatter_indices = logical_indices.masked_fill(~valid, mask_width)
    empty_row_index = has_selection.to(logical_indices.dtype) * mask_width
    scatter_indices = torch.cat((scatter_indices, empty_row_index[:, None]), dim=1).to(torch.int64)
    mask = torch.ones(
        (logical_indices.shape[0], mask_width + _FIA_MASK_ALIGNMENT),
        dtype=torch.uint8,
        device=logical_indices.device,
    )
    mask.scatter_(1, scatter_indices, 0)
    return mask[:, :mask_width].contiguous().view(torch.bool), has_selection


def _run_fia(
    op: Callable,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    logical_indices: torch.Tensor | None,
    logical_positions: torch.Tensor,
    seq_len: torch.Tensor,
    kv_length: int,
    output: torch.Tensor,
    gate: torch.Tensor,
    softmax_scale: float,
    block_table: torch.Tensor | None,
) -> None:
    rows, query_heads, head_dim = query.shape
    if block_table is None:
        kv_heads = key.shape[1]
        key = key.reshape(1, kv_length, kv_heads * head_dim).contiguous()
        value = value.reshape(1, kv_length, kv_heads * head_dim).contiguous()
        mask_width = kv_length
    else:
        kv_heads = key.shape[-1] // head_dim
        mask_width = block_table.shape[1] * _FIA_PAGE_SIZE
    mask_width = (mask_width + _FIA_MASK_ALIGNMENT - 1) // _FIA_MASK_ALIGNMENT * _FIA_MASK_ALIGNMENT
    # Budget both mask copies and the token-sized temporary index/valid arrays.
    bytes_per_row = 2 * mask_width + _FIA_MASK_ALIGNMENT
    if logical_indices is not None:
        bytes_per_row += 16 * logical_indices.shape[1]
    query_tile = min(_FIA_MAX_QUERY_TILE, max(1, _FIA_MASK_BUDGET_BYTES // bytes_per_row))
    for start in range(0, rows, query_tile):
        span = slice(start, min(start + query_tile, rows))
        visible = torch.minimum(logical_positions[span].add(1).clamp_min(0), seq_len.clamp_max(kv_length))
        if logical_indices is None:
            columns = torch.arange(mask_width, dtype=torch.int32, device=query.device)
            mask = columns[None, :] >= visible.clamp_min(1)[:, None]
            has_selection = visible > 0
        else:
            mask, has_selection = _selected_token_mask(logical_indices[span], visible, mask_width)
        tile = query[span].contiguous()
        tile_rows = tile.shape[0]
        result, _ = op(
            query=tile.view(1, tile_rows, query_heads * head_dim),
            key=key,
            value=value,
            atten_mask=mask.view(1, 1, tile_rows, mask_width),
            actual_seq_qlen=[tile_rows],
            actual_seq_kvlen=[kv_length],
            num_query_heads=query_heads,
            num_key_value_heads=kv_heads,
            softmax_scale=softmax_scale,
            input_layout="BSH",
            sparse_mode=0,
            inner_precise=0,
            block_table=block_table,
            block_size=_FIA_PAGE_SIZE if block_table is not None else 0,
            return_softmax_lse=False,
        )
        gated = result.reshape_as(tile) * torch.sigmoid(gate[span])
        output[span].copy_(torch.where(has_selection[:, None, None], gated, 0))


def qsa_prefill_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    logical_indices: torch.Tensor | None,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    softmax_scale: float,
    output: torch.Tensor,
    gate: torch.Tensor,
    request_plan: QSARequestPlan | None,
    sparse_attention: Callable,
    split_workspace_factory: Callable | None = None,
) -> bool:
    """Write the whole live mixed batch, or return False before writing output.

    Call only in eager mode, after FL has updated its raw and main K/V caches.
    The original FL sparse kernel handles long prefill and decode spans with
    their original request IDs and gate. Its launch already parallelizes query
    rows and KV heads, with no dense workspace proportional to context length.
    """
    if (
        request_plan is None
        or query.device.type != "npu"
        or query.dtype != torch.bfloat16
        or request_plan.num_actual_tokens != query.shape[0]
        or query.shape[1] % key.shape[1]
    ):
        return False
    try:
        import torch_npu
    except ImportError:
        return False
    op = getattr(torch_npu, "npu_fused_infer_attention_score_v2", None)
    if not callable(op):
        return False

    paged = None
    if any(request.use_fia and request.has_cached_prefix for request in request_plan.requests):
        paged = _paged_fia_cache(key_cache, value_cache)
    sparse_start = None
    split_sparse = False

    def run_sparse(start: int, end: int, split: bool) -> None:
        assert logical_indices is not None
        span = slice(start, end)
        kwargs = {"gate": gate[span]}
        if split and split_workspace_factory is not None:
            kwargs["split_workspace"] = split_workspace_factory(query[span], 4)
        sparse_attention(
            query[span], key_cache, value_cache, logical_indices[span], block_table,
            token_to_req[span], softmax_scale, output[span], **kwargs,
        )
        if split:
            logger.info_once("FL QSA long prefill uses split paged sparse attention")

    for request in request_plan.requests:
        use_fia = request.use_fia or logical_indices is None
        if use_fia and request.has_cached_prefix:
            use_fia = (
                request.kv_length <= block_table.shape[1] * key_cache.shape[1]
                and request.index < block_table.shape[0]
            )
        if not use_fia and logical_indices is None:
            raise RuntimeError("All-visible FIA request exceeds the KV block table")
        if not use_fia:
            if sparse_start is None:
                sparse_start = request.start
            split_sparse |= request.is_prefill
            continue
        if sparse_start is not None:
            run_sparse(sparse_start, request.start, split_sparse)
            sparse_start = None
            split_sparse = False
        span = slice(request.start, request.end)
        table = None
        request_key, request_value = key[span], value[span]
        if request.has_cached_prefix:
            request_table = block_table[request.index : request.index + 1]
            if paged is not None:
                request_key, request_value = paged.key, paged.value
                table = paged.table(request_table, request.kv_length)
            else:
                request_key, request_value = _gather_fia_cache(
                    key_cache, value_cache, request_table, request.kv_length
                )
        logger.info_once("FL QSA prefill uses native FIA attention")
        _run_fia(
            op, query[span], request_key, request_value,
            None if logical_indices is None else logical_indices[span],
            logical_positions[span], seq_lens[request.index], request.kv_length,
            output[span], gate[span], softmax_scale, table,
        )
    if sparse_start is not None:
        run_sparse(sparse_start, request_plan.num_actual_tokens, split_sparse)
    return True
