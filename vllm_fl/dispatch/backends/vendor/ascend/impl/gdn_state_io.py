"""Copy selected GDN rows without materializing the scheduler's padded pool."""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_rows(Pool, Rows, Slots, SLOT_STRIDE: tl.constexpr,
               POOL_STRIDES: tl.constexpr, ROW_SHAPE: tl.constexpr,
               ROW_SIZE: tl.constexpr, SCATTER: tl.constexpr,
               BLOCK: tl.constexpr):
    lane = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(Slots + lane * SLOT_STRIDE).to(tl.int64)
    # Gather padding from slot zero for compatibility; scatter never writes it.
    selected = tl.maximum(slot, 0)
    physical = selected * POOL_STRIDES[0]
    # Literal tuple indices avoid backend-dependent static-loop index lowering.
    if len(ROW_SHAPE) == 2:
        physical += (offset // ROW_SHAPE[1]) * POOL_STRIDES[1]
        physical += (offset % ROW_SHAPE[1]) * POOL_STRIDES[2]
    else:
        physical += (offset // (ROW_SHAPE[1] * ROW_SHAPE[2])) * POOL_STRIDES[1]
        physical += ((offset // ROW_SHAPE[2]) % ROW_SHAPE[1]) * POOL_STRIDES[2]
        physical += (offset % ROW_SHAPE[2]) * POOL_STRIDES[3]
    mask = offset < ROW_SIZE
    if SCATTER:
        mask = mask & (slot > 0)
        values = tl.load(Rows + lane * ROW_SIZE + offset, mask=mask, other=0)
        tl.store(Pool + physical, values, mask=mask)
    else:
        values = tl.load(Pool + physical, mask=mask, other=0)
        tl.store(Rows + lane * ROW_SIZE + offset, values, mask=mask)


@triton.jit
def _copy_dense_ssm_rows(
    Pool, Rows, Slots, SLOT_STRIDE: tl.constexpr,
    POOL_ROW_STRIDE: tl.constexpr, ROW_SIZE: tl.constexpr,
    SCATTER: tl.constexpr, BLOCK: tl.constexpr,
):
    lane = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(Slots + lane * SLOT_STRIDE).to(tl.int64)
    physical = tl.maximum(slot, 0) * POOL_ROW_STRIDE + offset
    mask = offset < ROW_SIZE
    if SCATTER:
        mask = mask & (slot > 0)
        values = tl.load(Rows + lane * ROW_SIZE + offset, mask=mask, other=0)
        tl.store(Pool + physical, values, mask=mask)
    else:
        values = tl.load(Pool + physical, mask=mask, other=0)
        tl.store(Rows + lane * ROW_SIZE + offset, values, mask=mask)


def _launch_row_copy(pool, indices, rows, *, scatter):
    width = math.prod(pool.shape[1:])
    if pool.ndim == 4 and pool.stride()[1:] == (
        pool.shape[2] * pool.shape[3], pool.shape[3], 1
    ):
        # The allocator may pad between slots. The FP32 state inside each
        # slot is packed, so move whole contiguous tiles without index math.
        block = 2048
        _copy_dense_ssm_rows[(indices.numel(), triton.cdiv(width, block))](
            pool, rows, indices, indices.stride(0), pool.stride(0),
            width, scatter, block,
        )
        return
    block = 256
    _copy_rows[(indices.numel(), triton.cdiv(width, block))](
        pool, rows, indices, indices.stride(0), pool.stride(), tuple(pool.shape[1:]),
        width, scatter, block,
    )


def _check(pool, indices):
    if indices.ndim != 1 or pool.ndim not in (3, 4):
        raise ValueError("GDN row copies require 1D indices and a 3D/4D state pool")
    if indices.device != pool.device or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("GDN indices must be colocated int32/int64")


def gather_state_rows(pool, indices):
    """Indices must be in range or <=0 (padding); allocation scales with lanes."""
    _check(pool, indices)
    rows = torch.empty((indices.numel(), *pool.shape[1:]), device=pool.device, dtype=pool.dtype)
    if rows.numel():
        _launch_row_copy(pool, indices, rows, scatter=False)
    return rows


def scatter_state_rows_(pool, indices, rows):
    """Write positive slots only; positive indices must be unique and in range."""
    _check(pool, indices)
    if (rows.shape != (indices.numel(), *pool.shape[1:]) or not rows.is_contiguous()
            or rows.device != pool.device or rows.dtype != pool.dtype):
        raise ValueError("GDN source rows must be contiguous and match the state pool")
    if rows.numel():
        _launch_row_copy(pool, indices, rows, scatter=True)
