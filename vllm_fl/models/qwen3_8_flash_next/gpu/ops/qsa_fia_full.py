# SPDX-License-Identifier: Apache-2.0
"""Fixed-capacity FIA for the Qwen3.8 eight-row FULL decode graph."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from os import getenv

import torch
import torch_npu
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_ROWS = 8
_HEADS = 3
_HEAD_DIM = 256
_FL_PAGE = 400
_FIA_PAGE = 128
_CAPACITY = 2048
_TILE = 16
_ROW_COPY_TILE = 32
_ROW_PSE_TILE = 1024
_REPACK_ENV = "VLLM_FL_QWEN38_FULL_FIA_REPACK"
_SCALE = _HEAD_DIM**-0.5


@dataclass(frozen=True)
class _FullFIAWorkspace:
    key: torch.Tensor
    value: torch.Tensor
    table: torch.Tensor
    bias: torch.Tensor
    live: torch.Tensor
    attention: torch.Tensor
    lse: torch.Tensor
    workspace: torch.Tensor
    zero: torch.Tensor
    operation: Callable[..., None]
    repack: str


# One strong owner per worker device. The worker serializes the FULL graphs
# sharing this storage, and their QSA repack, FIA and epilogue use one stream.
# Graph teardown must complete before any owner is dropped or overwritten.
_WORKSPACES: dict[torch.device, _FullFIAWorkspace] = {}


@triton.jit
def _repack_kv_and_pse(
    key, value, table, requests, positions, lengths,
    scratch_key, scratch_value, bias, live,
    key_block: tl.constexpr, value_block: tl.constexpr,
    key_row: tl.constexpr, value_row: tl.constexpr,
    table_row: tl.constexpr, table_col: tl.constexpr,
    req_stride: tl.constexpr, pos_stride: tl.constexpr, len_stride: tl.constexpr,
    n_requests: tl.constexpr, table_width: tl.constexpr, n_blocks: tl.constexpr,
    page: tl.constexpr, cap: tl.constexpr, max_visible: tl.constexpr,
    width: tl.constexpr, heads: tl.constexpr, tile: tl.constexpr,
    page_lanes: tl.constexpr, head_lanes: tl.constexpr,
):
    tile_id = tl.program_id(0)
    row = tl.program_id(1)
    req = tl.load(requests + row * req_stride).to(tl.int32)
    req_valid = (req >= 0) & (req < n_requests)
    safe_req = tl.minimum(tl.maximum(req, 0), n_requests - 1)
    length = tl.load(lengths + safe_req * len_stride).to(tl.int32)
    position = tl.load(positions + row * pos_stride).to(tl.int32)
    visible = tl.minimum(tl.minimum(length, position + 1), max_visible)
    visible = tl.where(req_valid, tl.maximum(visible, 0), 0)
    request_table = table + safe_req * table_row

    token = tile_id * tile + tl.arange(0, tile)
    logical_page = token // page
    in_page = token % page
    physical_page = tl.load(
        request_table + logical_page * table_col, logical_page < table_width, other=-1
    )
    valid = (token < visible) & (physical_page >= 0) & (physical_page < n_blocks)
    safe_page = tl.where(valid, physical_page, 0).to(tl.int64)
    channel = tl.arange(0, width)
    key_base = safe_page * key_block + in_page * key_row
    value_base = safe_page * value_block + in_page * value_row
    k = tl.load(key + key_base[:, None] + channel[None, :], valid[:, None], other=0.0)
    v = tl.load(value + value_base[:, None] + channel[None, :], valid[:, None], other=0.0)
    dest = (row * cap + token[:, None]) * width + channel[None, :]
    tl.store(scratch_key + dest, k, token[:, None] < cap)
    tl.store(scratch_value + dest, v, token[:, None] < cap)

    # A missing first page can still leave a later page visible.
    candidate = tl.arange(0, page_lanes)
    candidate_visible = (candidate * page < visible) & (candidate < table_width)
    candidate_page = tl.load(
        request_table + candidate * table_col, candidate_visible, other=-1
    )
    present = candidate_visible & (candidate_page >= 0) & (candidate_page < n_blocks)
    has_tokens = tl.sum(present.to(tl.int32), axis=0) > 0
    if tile_id == 0:
        tl.store(live + row, has_tokens)

    # A finite dummy token prevents all-masked softmax on empty rows.
    allowed = valid | ((~has_tokens) & (token == 0))
    pse = tl.where(allowed, 0.0, -float("inf"))
    head = tl.arange(0, head_lanes)
    bias_offset = (row * heads + head[:, None]) * cap + token[None, :]
    tl.store(bias + bias_offset, pse[None, :],
             (head[:, None] < heads) & (token[None, :] < cap))


@triton.jit
def _prepare_fia_row_vectors(
    table, requests, positions, lengths, native_table, bias, live,
    table_row: tl.constexpr, table_col: tl.constexpr,
    req_stride: tl.constexpr, pos_stride: tl.constexpr, len_stride: tl.constexpr,
    n_requests: tl.constexpr, table_width: tl.constexpr, n_blocks: tl.constexpr,
    page: tl.constexpr, native_page: tl.constexpr, cap: tl.constexpr,
    max_visible: tl.constexpr, heads: tl.constexpr, page_lanes: tl.constexpr,
    pse_tile: tl.constexpr,
):
    chunk = tl.program_id(0)
    row = tl.program_id(1)
    req = tl.load(requests + row * req_stride).to(tl.int32)
    req_valid = (req >= 0) & (req < n_requests)
    safe_req = tl.minimum(tl.maximum(req, 0), n_requests - 1)
    length = tl.load(lengths + safe_req * len_stride).to(tl.int32)
    position = tl.load(positions + row * pos_stride).to(tl.int32)
    visible = tl.minimum(tl.minimum(length, position + 1), max_visible)
    visible = tl.where(req_valid, tl.maximum(visible, 0), 0)

    source = tl.arange(0, page_lanes)
    in_window = (source * page < visible) & (source < table_width)
    physical = tl.load(table + safe_req * table_row + source * table_col, in_window, other=-1)
    present = in_window & (physical >= 0) & (physical < n_blocks)
    page_bits = tl.sum(present.to(tl.int32) << source, axis=0)
    has_tokens = page_bits != 0

    token = chunk * pse_tile + tl.arange(0, pse_tile)
    valid = (token < visible) & (((page_bits >> (token // page)) & 1) != 0)
    # A finite dummy token prevents all-masked softmax on empty rows.
    allowed = valid | ((~has_tokens) & (token == 0))
    pse = tl.where(allowed, 0.0, -float("inf"))
    for head in tl.static_range(heads):
        tl.store(bias + (row * heads + head) * cap + token, pse)

    if chunk == 0:
        tl.store(live + row, has_tokens)
        native = tl.arange(0, cap // native_page)
        first = native * native_page
        last = tl.maximum(tl.minimum(first + native_page, visible) - 1, 0)
        # A 128-token page can intersect at most two 400-token pages.
        first_valid = ((page_bits >> (first // page)) & 1) != 0
        last_valid = ((page_bits >> (last // page)) & 1) != 0
        active = (first < visible) & (first_valid | last_valid)
        dest = row * (cap // native_page) + native
        tl.store(native_table + dest, tl.where(active, dest + 1, 0))


@triton.jit
def _copy_fia_pages(
    key, value, table, requests, positions, lengths, native_table,
    scratch_key, scratch_value,
    key_block: tl.constexpr, value_block: tl.constexpr,
    key_row: tl.constexpr, value_row: tl.constexpr,
    table_row: tl.constexpr, table_col: tl.constexpr,
    req_stride: tl.constexpr, pos_stride: tl.constexpr, len_stride: tl.constexpr,
    table_width: tl.constexpr, n_blocks: tl.constexpr,
    page: tl.constexpr, native_page: tl.constexpr, cap: tl.constexpr,
    max_visible: tl.constexpr, width: tl.constexpr, tile: tl.constexpr,
):
    tile_id = tl.program_id(0)
    row = tl.program_id(1)
    native = tile_id // (native_page // tile)
    dest_page = tl.load(native_table + row * (cap // native_page) + native)
    # Page zero is immutable. A nonzero table entry implies a valid request.
    if dest_page > 0:
        req = tl.load(requests + row * req_stride).to(tl.int32)
        length = tl.load(lengths + req * len_stride).to(tl.int32)
        position = tl.load(positions + row * pos_stride).to(tl.int32)
        visible = tl.minimum(tl.minimum(length, position + 1), max_visible)
        lane = tl.arange(0, tile)
        token = tile_id * tile + lane
        logical_page = token // page
        in_page = token % page
        physical = tl.load(
            table + req * table_row + logical_page * table_col,
            logical_page < table_width, other=-1,
        )
        valid = (token < visible) & (physical >= 0) & (physical < n_blocks)
        safe_page = tl.where(valid, physical, 0).to(tl.int64)
        channel = tl.arange(0, width)
        key_base = safe_page * key_block + in_page * key_row
        value_base = safe_page * value_block + in_page * value_row
        k = tl.load(key + key_base[:, None] + channel[None, :], valid[:, None], other=0.0)
        v = tl.load(value + value_base[:, None] + channel[None, :], valid[:, None], other=0.0)
        dest_token = dest_page * native_page + (tile_id % (native_page // tile)) * tile + lane
        dest = dest_token[:, None] * width + channel[None, :]
        # Initialize masked tails and holes in each page FIA can read.
        tl.store(scratch_key + dest, k)
        tl.store(scratch_value + dest, v)


def _launch_repack(
    key: torch.Tensor,
    value: torch.Tensor,
    table: torch.Tensor,
    requests: torch.Tensor,
    positions: torch.Tensor,
    lengths: torch.Tensor,
    state: _FullFIAWorkspace,
    max_visible: int,
) -> None:
    if state.repack == "row":
        _prepare_fia_row_vectors[(_CAPACITY // _ROW_PSE_TILE, _ROWS)](
            table, requests, positions, lengths, state.table, state.bias, state.live,
            *table.stride(), requests.stride(0), positions.stride(0), lengths.stride(0),
            table.shape[0], table.shape[1], key.shape[0],
            _FL_PAGE, _FIA_PAGE, _CAPACITY, max_visible, _HEADS,
            triton.next_power_of_2(triton.cdiv(_CAPACITY, _FL_PAGE)), _ROW_PSE_TILE,
            num_warps=4, num_stages=2,
        )
        _copy_fia_pages[(_CAPACITY // _ROW_COPY_TILE, _ROWS)](
            key, value, table, requests, positions, lengths, state.table, state.key, state.value,
            key.stride(0), value.stride(0), key.stride(1), value.stride(1),
            *table.stride(), requests.stride(0), positions.stride(0), lengths.stride(0),
            table.shape[1], key.shape[0], _FL_PAGE, _FIA_PAGE, _CAPACITY, max_visible,
            _HEAD_DIM, _ROW_COPY_TILE, num_warps=4, num_stages=2,
        )
        return
    _repack_kv_and_pse[(_CAPACITY // _TILE, _ROWS)](
        key, value, table, requests, positions, lengths,
        state.key, state.value, state.bias, state.live,
        key.stride(0), value.stride(0), key.stride(1), value.stride(1),
        *table.stride(), requests.stride(0), positions.stride(0), lengths.stride(0),
        table.shape[0], table.shape[1], key.shape[0], _FL_PAGE, _CAPACITY, max_visible,
        _HEAD_DIM, _HEADS, _TILE, triton.next_power_of_2(triton.cdiv(_CAPACITY, _FL_PAGE)),
        triton.next_power_of_2(_HEADS), num_warps=4, num_stages=2,
    )


def _native_arguments(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    table: torch.Tensor, bias: torch.Tensor,
) -> dict:
    return {
        "query": query, "key": key, "value": value, "block_table": table,
        "input_layout": "TND", "block_size": _FIA_PAGE, "num_key_value_heads": 1,
        "sparse_mode": 0, "pre_tokens": 2147483647, "next_tokens": 0,
        "pse_shift": bias, "atten_mask": None,
        "actual_seq_lengths": list(range(1, _ROWS + 1)),
        "actual_seq_lengths_kv": [_CAPACITY] * _ROWS,
        "num_heads": _HEADS, "scale": _SCALE, "softmax_lse_flag": False,
    }


@torch.inference_mode()
def prepare_full_fia_workspace(device: torch.device | str) -> None:
    """Allocate one worker-owned workspace outside capture, before FULL warmup."""
    device = torch.device(device)
    if device.type != "npu":
        raise ValueError(f"Qwen3.8 FULL FIA requires an NPU device, got {device}")
    if device.index is None:
        device = torch.device("npu", torch.npu.current_device())
    repack = getenv(_REPACK_ENV, "fused")
    if repack not in ("fused", "row"):
        raise ValueError(f"{_REPACK_ENV} must be 'fused' or 'row', got {repack!r}")
    state = _WORKSPACES.get(device)
    if state is not None:
        if state.repack != repack:
            raise RuntimeError(f"Changing {_REPACK_ENV} requires restarting the Qwen3.8 worker")
        return
    if torch.npu.is_current_stream_capturing():
        raise RuntimeError("Prepare the Qwen3.8 FULL FIA workspace before ACL graph capture")

    pages = _ROWS * (_CAPACITY // _FIA_PAGE)
    first_page = int(repack == "row")
    options = {"dtype": torch.bfloat16, "device": device}
    key = torch.zeros((pages + first_page, _FIA_PAGE, _HEAD_DIM), **options)
    value = torch.zeros_like(key)
    table = torch.arange(first_page, pages + first_page, dtype=torch.int32, device=device)
    table = table.view(_ROWS, -1)
    bias = torch.zeros((_ROWS, _HEADS, 1, _CAPACITY), **options)
    live = torch.zeros((_ROWS,), dtype=torch.bool, device=device)
    zero = torch.zeros((), **options)
    dummy_query = torch.zeros((_ROWS, _HEADS, _HEAD_DIM), **options)
    kwargs = _native_arguments(dummy_query, key, value, table, bias)

    packet = torch.ops.npu.npu_fused_infer_attention_score
    eager = getattr(torch_npu, "npu_fused_infer_attention_score", packet.default)
    workspace_name = "_npu_fused_infer_attention_score_get_max_workspace"
    get_workspace = getattr(torch_npu, workspace_name, None)
    if get_workspace is None:
        get_workspace = getattr(torch.ops.npu, workspace_name)
    attention, lse = eager(**kwargs)
    workspace = get_workspace(**kwargs)
    out_op = packet.out
    arg_names = {argument.name for argument in out_op._schema.arguments}
    if "out" in arg_names:
        out_buffers = {"out": [attention, lse]}
    elif {"attention_out", "softmax_lse"} <= arg_names:
        out_buffers = {"attention_out": attention, "softmax_lse": lse}
    else:
        raise RuntimeError(f"Unsupported legacy FIA output schema: {out_op._schema}")

    del kwargs["query"]

    def native(query: torch.Tensor) -> None:
        out_op(query=query, **kwargs, workspace=workspace, **out_buffers)

    native(dummy_query)
    torch.npu.synchronize(device)
    _WORKSPACES[device] = _FullFIAWorkspace(
        key, value, table, bias, live, attention, lse, workspace, zero, native, repack
    )
    logger.info(
        "Qwen3.8 FULL FIA prepared on %s, repack=%s, native workspace bytes=%d, K/V scratch bytes=%d",
        device, repack, workspace.numel() * workspace.element_size(),
        (key.numel() + value.numel()) * key.element_size(),
    )


def supports_full_fia_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
    gate: torch.Tensor,
    max_visible: int,
) -> bool:
    """Check capture-time shapes and layouts. All replay metadata stays on NPU."""
    q_shape = (_ROWS, _HEADS, _HEAD_DIM)
    if (
        query.device.type != "npu"
        or not 0 < max_visible <= _CAPACITY
        or query.shape != q_shape
        or output.shape != q_shape
        or gate.shape != q_shape
        or block_table.ndim != 2
        or min(block_table.shape) <= 0
        or token_to_req.shape != (_ROWS,)
        or logical_positions.shape != (_ROWS,)
        or seq_lens.shape != (block_table.shape[0],)
        or key_cache.ndim != 4
        or key_cache.shape != value_cache.shape
        or key_cache.shape[1:] != (_FL_PAGE, 1, _HEAD_DIM)
        or key_cache.shape[0] == 0
    ):
        return False
    data = (query, key_cache, value_cache, output, gate)
    metadata = (block_table, token_to_req, logical_positions, seq_lens)
    return (
        all(t.dtype == torch.bfloat16 for t in data)
        and all(t.device == query.device for t in (*data, *metadata))
        and query.is_contiguous()
        and output.is_contiguous()
        and key_cache.stride(-1) == value_cache.stride(-1) == 1
        and block_table.dtype == token_to_req.dtype == seq_lens.dtype == torch.int32
        and logical_positions.dtype in (torch.int32, torch.int64)
    )


def full_fia_attention(
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
    """Attend an all-visible FULL batch using the caller's output and metadata."""
    if softmax_scale != _SCALE or not supports_full_fia_attention(
        query, key_cache, value_cache, block_table, token_to_req,
        logical_positions, seq_lens, output, gate, max_visible,
    ):
        raise ValueError("Qwen3.8 FULL FIA requires the BF16 eight-row QSA decode layout")
    state = _WORKSPACES.get(query.device)
    if state is None:
        raise RuntimeError("Call prepare_full_fia_workspace before Qwen3.8 FULL graph capture")
    if state.repack == "row":
        logger.info_once(
            "Qwen3.8 FULL decode uses fixed-capacity legacy FIA with row-vector QSA repack "
            "(PSE tile 1024, copy tile 32)"
        )
    else:
        logger.info_once("Qwen3.8 FULL decode uses fixed-capacity legacy FIA with fused QSA repack")
    _launch_repack(
        key_cache, value_cache, block_table, token_to_req, logical_positions,
        seq_lens, state, max_visible,
    )
    state.operation(query)
    gate_factor = gate.float().sigmoid().bfloat16()
    torch.where(state.live[:, None, None], state.attention * gate_factor, state.zero, out=output)
