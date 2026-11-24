# SPDX-License-Identifier: Apache-2.0
"""Eager prefill selection with CANN LightningIndexerV2 and FL's index ABI."""

from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.utils.torch_utils import canonicalize_singleton_dim_strides

from .qsa import expand_qsa_block_indices, qsa_select_all_paged_tokens

if TYPE_CHECKING:
    from ...common.qsa_cache import QSAForwardMetadata
    from ..indexer_qsa import QSAIndexer

logger = init_logger(__name__)

_HEAD_DIM = 128
_COMPRESS_RATIO = 4
_TOKEN_TOPK = 2048
_GROUP_TOPK = _TOKEN_TOPK // _COMPRESS_RATIO
_INVALID_GROUP = torch.iinfo(torch.int32).max


@lru_cache(maxsize=1)
def _load_lightning_indexer(library_path: str):
    try:
        torch.ops.load_library(library_path)
        namespace = torch.ops.vllm_fl_qsa
        return (
            namespace.npu_lightning_indexer_v2_metadata.default,
            namespace.npu_lightning_indexer_v2.default,
        )
    except (AttributeError, OSError, RuntimeError) as error:
        logger.warning("FL QSA LightningIndexerV2 extension unavailable: %s", error)
        return None


def _packed_max_lengths(
    metadata: QSAForwardMetadata,
    rows: int,
    page_capacity: int,
    allow_prefill: bool,
    allow_decode: bool,
) -> tuple[int, int] | None:
    common = metadata.common
    starts = common.query_start_loc_cpu
    lengths = common.seq_lens_cpu_upper_bound
    prefill = common.is_prefilling
    requests = common.num_reqs
    if (
        starts.device.type != "cpu"
        or starts.shape != (requests + 1,)
        or lengths is None
        or lengths.device.type != "cpu"
        or lengths.shape != (requests,)
        or prefill is None
        or prefill.device.type != "cpu"
        or prefill.shape != (requests,)
    ):
        return None
    has_prefill = bool(prefill.any())
    if has_prefill and (not allow_prefill or common.max_query_len <= 1):
        return None
    if not has_prefill and (not allow_decode or common.max_query_len != 1):
        return None
    boundaries = starts.tolist()
    query_lengths = [end - start for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)]
    sequence_lengths = lengths.tolist()
    # The runner's CPU lengths are exact with speculation disabled. QSA builds
    # each logical position as seq_len - query_len + local_query_row.
    if (
        boundaries[0] != 0
        or boundaries[-1] != rows
        or not all(
            0 < query_len <= seq_len <= page_capacity
            for query_len, seq_len in zip(query_lengths, sequence_lengths, strict=True)
        )
    ):
        return None
    return max(query_lengths), max(sequence_lengths)


def _repage_indexer_cache(
    key_cache: torch.Tensor, block_table: torch.Tensor, metadata: QSAForwardMetadata
) -> tuple[torch.Tensor, torch.Tensor]:
    # CANN 9.1 requires a multiple of 16; FL's 400-token align block has
    # 100 compressed rows. Pack live logical keys in 128-row native pages.
    # Reuse the row mapping across layers while reading each layer's own keys.
    page_size = key_cache.shape[1]
    physical_stride = key_cache.stride(0) // _HEAD_DIM
    mapping_key = (page_size, physical_stride)
    common = metadata.common
    mappings = getattr(common, "_fl_qsa_li_page_maps", None)
    if mappings is None:
        mappings = common._fl_qsa_li_page_maps = {}
    mapping = mappings.get(mapping_key)
    if mapping is None:
        native_page = 128
        lengths = common.seq_lens_cpu_upper_bound.tolist()
        pages = [max(1, (length // _COMPRESS_RATIO + native_page - 1) // native_page)
                 for length in lengths]
        native_table_cpu = torch.zeros((len(pages), max(pages)), dtype=torch.int32)
        row_ids = []
        offset = 0
        for req, (length, count) in enumerate(zip(lengths, pages, strict=True)):
            native_table_cpu[req, :count] = torch.arange(offset, offset + count, dtype=torch.int32)
            offset += count
            compressed_length = length // _COMPRESS_RATIO
            logical = torch.arange(count * native_page, device=key_cache.device, dtype=torch.int64)
            bounded = logical.clamp_max(max(0, compressed_length - 1))
            block = block_table[req].gather(0, bounded // page_size).to(torch.int64)
            row_ids.append(block.clamp_min(0) * physical_stride + bounded.remainder(page_size))
        mapping = torch.cat(row_ids), native_table_cpu.to(key_cache.device, non_blocking=True)
        mappings[mapping_key] = mapping
    row_ids, native_table = mapping
    physical_rows = (key_cache.shape[0] - 1) * physical_stride + page_size
    physical_cache = key_cache.as_strided(
        (physical_rows, _HEAD_DIM), (_HEAD_DIM, 1)
    )
    packed = physical_cache.index_select(0, row_ids).reshape(-1, 128, 1, _HEAD_DIM)
    return packed, native_table


def try_select_qsa_indices(
    indexer: QSAIndexer,
    query: torch.Tensor,
    metadata: QSAForwardMetadata,
    out: torch.Tensor | None = None,
    *,
    allow_prefill: bool = True,
    allow_decode: bool = False,
) -> torch.Tensor | None:
    """Return expanded logical token indices when the admitted eager ABI matches.

    A missing binding or an unsupported layout leaves selection to the caller.
    Once native work is submitted, failures propagate to the worker.
    """
    library_path = os.getenv("VLLM_FL_QWEN38_QSA_EXTENSION")
    if not library_path or query.device.type != "npu" or torch.compiler.is_compiling():
        return None
    context = get_forward_context()
    common = metadata.common
    if (
        context.cudagraph_runtime_mode != CUDAGraphMode.NONE
        or not context.skip_compiled
        or common.causal is not True
        or indexer.token_topk != _TOKEN_TOPK
        or indexer.compress_ratio != _COMPRESS_RATIO
        or metadata.compress_ratio != _COMPRESS_RATIO
        or query.dtype != torch.bfloat16
        or query.ndim != 3
        or not 1 <= query.shape[1] <= 64
        or query.shape[2] != _HEAD_DIM
    ):
        return None

    rows, heads, _ = query.shape
    requests = common.num_reqs
    key_cache = indexer.compressed_key_cache.kv_cache
    block_table = metadata.block_table
    query_starts = metadata.query_start_loc
    lengths = metadata.seq_lens
    positions = metadata.logical_positions
    row_requests = metadata.token_to_req
    if (
        not rows
        or not requests
        or rows != metadata.num_actual_tokens
        or key_cache.ndim != 4
        or key_cache.dtype != query.dtype
        or key_cache.shape[0] <= 0
        or key_cache.shape[1:] != (metadata.storage_block_size, 1, _HEAD_DIM)
        or not 0 < metadata.storage_block_size <= 1024
        or block_table.ndim != 2
        or block_table.shape[0] != requests
        or block_table.shape[1] <= 0
        or query_starts.shape != (requests + 1,)
        or lengths.shape != (requests,)
        or positions.shape != (rows,)
        or row_requests.shape != (rows,)
        or any(
            tensor.device != query.device
            for tensor in (key_cache, block_table, query_starts, lengths, positions, row_requests)
        )
        or any(
            tensor.dtype != torch.int32
            for tensor in (block_table, query_starts, lengths)
        )
        or any(
            tensor.dtype not in (torch.int32, torch.int64) or not tensor.is_contiguous()
            for tensor in (positions, row_requests)
        )
        or (
            out is not None
            and (
                out.shape != (rows, _TOKEN_TOPK + _COMPRESS_RATIO - 1)
                or out.dtype != torch.int32
                or out.device != query.device
            )
        )
    ):
        return None

    page_size = metadata.storage_block_size
    key_cache = canonicalize_singleton_dim_strides(key_cache)
    if (
        key_cache.stride()[1:] != (_HEAD_DIM, _HEAD_DIM, 1)
        or key_cache.stride(0) < page_size * _HEAD_DIM
        or key_cache.stride(0) % _HEAD_DIM
    ):
        return None
    raw_page_size = page_size * _COMPRESS_RATIO
    max_lengths = _packed_max_lengths(
        metadata, rows, block_table.shape[1] * raw_page_size, allow_prefill, allow_decode,
    )
    if max_lengths is None:
        return None
    max_query_len, max_sequence_len = max_lengths

    ops = _load_lightning_indexer(library_path)
    if ops is None:
        return None

    # Every complete group fits when the context is within the budget. The
    # existing expansion writes the causal partial group and invalid padding.
    if max_sequence_len <= _TOKEN_TOPK:
        return qsa_select_all_paged_tokens(
            positions,
            lengths,
            row_requests,
            _TOKEN_TOPK,
            _COMPRESS_RATIO,
            max_sequence_len,
            out,
        )

    if page_size % 16:
        key_cache, block_table = _repage_indexer_cache(key_cache, block_table, metadata)

    make_metadata, select_groups = ops
    compressed_lengths = lengths.floor_divide(_COMPRESS_RATIO)
    residual_lengths = lengths.remainder(_COMPRESS_RATIO)
    common_args = dict(
        cu_seqlens_q=query_starts.contiguous(),
        seqused_k=compressed_lengths,
        cmp_residual_k=residual_lengths,
        max_seqlen_q=max_query_len,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=_COMPRESS_RATIO,
    )
    schedule = make_metadata(
        heads,
        1,
        _HEAD_DIM,
        _GROUP_TOPK,
        batch_size=requests,
        max_seqlen_k=max_sequence_len // _COMPRESS_RATIO,
        **common_args,
    )
    weights = torch.full(
        (rows, heads),
        1.0 / math.sqrt(_HEAD_DIM),
        dtype=torch.float32,
        device=query.device,
    )
    groups, _ = select_groups(
        query.contiguous(),
        key_cache,
        weights,
        _GROUP_TOPK,
        block_table=block_table.contiguous(),
        metadata=schedule,
        return_value=0,
        **common_args,
    )
    if groups.shape != (rows, 1, _GROUP_TOPK) or groups.dtype != torch.int32:
        raise RuntimeError("FL QSA LightningIndexerV2 returned an unexpected index layout")
    # Native selection can resolve cutoff ties differently. Visit its selected
    # groups in FL's ascending logical order and keep invalid entries last.
    groups = groups.squeeze(1)
    ordered = groups.masked_fill(groups < 0, _INVALID_GROUP).sort(dim=-1).values
    ordered.masked_fill_(ordered == _INVALID_GROUP, -1)
    if max_query_len == 1:
        logger.info_once("FL QSA eager decode uses native request-batched LightningIndexerV2.")
    else:
        logger.info_once("FL QSA prefill uses native request-batched LightningIndexerV2.")
    return expand_qsa_block_indices(
        ordered, positions, lengths, row_requests, _COMPRESS_RATIO, _TOKEN_TOPK, out
    )
