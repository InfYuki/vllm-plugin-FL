# SPDX-License-Identifier: Apache-2.0
"""Packed direct FP32 SSM updates for Qwen3.8 decode."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _pack_active_suffix(
    Slots, Q, K, V, Beta, Gate,
    PackedQ, PackedK, PackedV, PackedBeta, PackedGate,
    Lengths, PackedSlots, LaneToPacked,
    POOL_ROWS: tl.constexpr, N: tl.constexpr, QK_WIDTH: tl.constexpr,
    V_WIDTH: tl.constexpr, V_TILE: tl.constexpr,
    HEADS: tl.constexpr, GATE_TILE: tl.constexpr,
):
    lane = tl.program_id(0)
    lanes = tl.arange(0, N)
    slots = tl.load(Slots + lanes)
    valid = (slots > 0) & (slots < POOL_ROWS)
    inactive = N - tl.sum(valid.to(tl.int32), 0)

    if lane == N:
        # One program writes all the small buffers. Neighboring lanes can share
        # a 32-byte storage block, especially for the six gate values per lane.
        earlier = lanes[None, :] < lanes[:, None]
        active_before = tl.sum((earlier & valid[None, :]).to(tl.int32), 1)
        dest = tl.where(valid, inactive + active_before, -1)
        tl.store(LaneToPacked + lanes, dest)

        # Pick the lowest unused IDs from 1..N for the ignored physical prefix.
        # Keep all indices unique and in range, as the FLA ABI requires.
        candidates = lanes + 1
        occupied = tl.sum(
            ((candidates[:, None] == slots[None, :]) & valid[None, :]).to(tl.int32), 1,
        ) > 0
        free_before = tl.sum((earlier & ~occupied[None, :]).to(tl.int32), 1)
        free_match = (lanes[:, None] == free_before[None, :]) & ~occupied[None, :]
        free_ids = tl.sum(tl.where(free_match, candidates[None, :], 0), 1)

        active_match = lanes[:, None] == dest[None, :]
        active_ids = tl.sum(tl.where(active_match, slots[None, :], 0), 1)
        active_dest = lanes >= inactive
        tl.store(PackedSlots + lanes, tl.where(active_dest, active_ids, free_ids))
        length_pos = tl.arange(0, 2 * N)
        length = tl.where(length_pos == 0, inactive, (length_pos > inactive).to(tl.int32))
        tl.store(Lengths + length_pos, length, length_pos <= N)

        gate_dest = tl.arange(0, GATE_TILE)
        gate_row = gate_dest // HEADS
        gate_match = gate_row[:, None] == dest[None, :]
        gate_lane = tl.sum(tl.where(gate_match, lanes[None, :], 0), 1)
        gate_source = gate_lane * HEADS + gate_dest % HEADS
        gate_mask = (gate_row >= inactive) & (gate_dest < N * HEADS)
        gate = tl.load(Gate + gate_source, gate_mask, other=0)
        beta = tl.load(Beta + gate_source, gate_mask, other=0)
        tl.store(PackedGate + gate_dest, gate, gate_dest < N * HEADS)
        tl.store(PackedBeta + gate_dest, beta, gate_dest < N * HEADS)
    else:
        slot = tl.load(Slots + lane)
        if (slot > 0) & (slot < POOL_ROWS):
            active_before = tl.sum(((lanes < lane) & valid).to(tl.int32), 0)
            dest = inactive + active_before
            qk = tl.arange(0, QK_WIDTH)
            v = tl.arange(0, V_TILE)
            tl.store(PackedQ + dest * QK_WIDTH + qk, tl.load(Q + lane * QK_WIDTH + qk))
            tl.store(PackedK + dest * QK_WIDTH + qk, tl.load(K + lane * QK_WIDTH + qk))
            value = tl.load(V + lane * V_WIDTH + v, v < V_WIDTH, other=0)
            tl.store(PackedV + dest * V_WIDTH + v, value, v < V_WIDTH)


@triton.jit
def _unpack_active_suffix(
    PackedOutput, LaneToPacked, Output, WIDTH: tl.constexpr, TILE: tl.constexpr,
):
    lane = tl.program_id(0)
    source = tl.load(LaneToPacked + lane)
    column = tl.arange(0, TILE)
    safe_source = tl.maximum(source, 0)
    value = tl.load(
        PackedOutput + safe_source * WIDTH + column,
        (source >= 0) & (column < WIDTH), other=0,
    )
    tl.store(Output + lane * WIDTH + column, value, column < WIDTH)


@triton.jit
def _pack_qkv_active_suffix(
    Slots, X, Beta, Gate,
    PackedQ, PackedK, PackedV, PackedBeta, PackedGate,
    Lengths, PackedSlots, LaneToPacked,
    POOL_ROWS: tl.constexpr, N: tl.constexpr, KEY_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr, V_WIDTH: tl.constexpr, V_TILE: tl.constexpr,
    HEADS: tl.constexpr, GATE_TILE: tl.constexpr,
):
    lane = tl.program_id(0)
    lanes = tl.arange(0, N)
    slots = tl.load(Slots + lanes)
    valid = (slots > 0) & (slots < POOL_ROWS)
    inactive = N - tl.sum(valid.to(tl.int32), 0)

    if lane == N:
        # Match the validated metadata pack, including its distinct prefix IDs.
        earlier = lanes[None, :] < lanes[:, None]
        active_before = tl.sum((earlier & valid[None, :]).to(tl.int32), 1)
        dest = tl.where(valid, inactive + active_before, -1)
        tl.store(LaneToPacked + lanes, dest)

        candidates = lanes + 1
        occupied = tl.sum(
            ((candidates[:, None] == slots[None, :]) & valid[None, :]).to(tl.int32), 1,
        ) > 0
        free_before = tl.sum((earlier & ~occupied[None, :]).to(tl.int32), 1)
        free_match = (lanes[:, None] == free_before[None, :]) & ~occupied[None, :]
        free_ids = tl.sum(tl.where(free_match, candidates[None, :], 0), 1)

        active_match = lanes[:, None] == dest[None, :]
        active_ids = tl.sum(tl.where(active_match, slots[None, :], 0), 1)
        active_dest = lanes >= inactive
        tl.store(PackedSlots + lanes, tl.where(active_dest, active_ids, free_ids))
        length_pos = tl.arange(0, 2 * N)
        length = tl.where(length_pos == 0, inactive, (length_pos > inactive).to(tl.int32))
        tl.store(Lengths + length_pos, length, length_pos <= N)

        gate_dest = tl.arange(0, GATE_TILE)
        gate_row = gate_dest // HEADS
        gate_match = gate_row[:, None] == dest[None, :]
        gate_lane = tl.sum(tl.where(gate_match, lanes[None, :], 0), 1)
        gate_source = gate_lane * HEADS + gate_dest % HEADS
        gate_mask = (gate_row >= inactive) & (gate_dest < N * HEADS)
        gate = tl.load(Gate + gate_source, gate_mask, other=0)
        beta = tl.load(Beta + gate_source, gate_mask, other=0)
        tl.store(PackedGate + gate_dest, gate, gate_dest < N * HEADS)
        tl.store(PackedBeta + gate_dest, beta, gate_dest < N * HEADS)
    else:
        slot = tl.load(Slots + lane)
        if (slot > 0) & (slot < POOL_ROWS):
            active_before = tl.sum(((lanes < lane) & valid).to(tl.int32), 0)
            dest = inactive + active_before
            qk_width: tl.constexpr = KEY_HEADS * HEAD_DIM
            row_width: tl.constexpr = 2 * qk_width + V_WIDTH
            head = tl.arange(0, KEY_HEADS)
            col = tl.arange(0, HEAD_DIM)
            qk_offset = head[:, None] * HEAD_DIM + col[None, :]
            row = X + lane * row_width
            q = tl.load(row + qk_offset).to(tl.float32)
            k = tl.load(row + qk_width + qk_offset).to(tl.float32)
            q_norm = tl.rsqrt(tl.sum(q * q, 1) + 1e-6)
            k_norm = tl.rsqrt(tl.sum(k * k, 1) + 1e-6)
            q = (q * q_norm[:, None]).to(tl.bfloat16)
            k = (k * k_norm[:, None]).to(tl.bfloat16)
            tl.store(PackedQ + dest * qk_width + qk_offset, q)
            tl.store(PackedK + dest * qk_width + qk_offset, k)

            v = tl.arange(0, V_TILE)
            value = tl.load(row + 2 * qk_width + v, v < V_WIDTH, other=0)
            tl.store(PackedV + dest * V_WIDTH + v, value, v < V_WIDTH)


def direct_ssm_recurrent(
    recurrent, state, indices, beta, g, output, scale, *, prepared_qkv, convolved,
):
    """Pack live lanes, update their pool rows, and restore the original lane order."""
    lanes = 8
    key_heads = 2
    value_heads = 6
    head_dim = 128
    value_width = value_heads * head_dim
    value_tile = triton.next_power_of_2(value_width)
    packed_q = output.new_empty((lanes, key_heads, head_dim))
    packed_k = torch.empty_like(packed_q)
    packed_v = torch.empty_like(output)
    packed_beta = torch.empty_like(beta)
    packed_gate = torch.empty_like(g)
    lengths = indices.new_empty((lanes + 1,))
    packed_slots = torch.empty_like(indices)
    lane_to_packed = torch.empty_like(indices)

    # Active indices are unique scheduler-owned slots. FLA ignores the prefix
    # before it reads any Q/K/V data or SSM slots for those inactive lanes.
    gate_tile = triton.next_power_of_2(lanes * value_heads)
    if convolved is None:
        q, k, v = prepared_qkv
        _pack_active_suffix[(lanes + 1,)](
            indices, q, k, v, beta, g,
            packed_q, packed_k, packed_v, packed_beta, packed_gate,
            lengths, packed_slots, lane_to_packed,
            state.shape[0], lanes, key_heads * head_dim, value_width, value_tile,
            value_heads, gate_tile,
        )
    else:
        _pack_qkv_active_suffix[(lanes + 1,)](
            indices, convolved, beta, g,
            packed_q, packed_k, packed_v, packed_beta, packed_gate,
            lengths, packed_slots, lane_to_packed,
            state.shape[0], lanes, key_heads, head_dim, value_width, value_tile,
            value_heads, gate_tile,
        )
    packed_output = recurrent(
        packed_q, packed_k, packed_v, state, beta=packed_beta, g=packed_gate,
        scale=scale, actual_seq_lengths=lengths, ssm_state_indices=packed_slots,
    )
    _unpack_active_suffix[(lanes,)](
        packed_output, lane_to_packed, output, value_width, value_tile,
    )
