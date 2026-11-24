# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM custom-op boundaries for FlagGems-vllm HyperConnection kernels."""

from functools import lru_cache
from os import getenv

import torch
from flaggems_vllm import (
    qwen4_grouped_gemma_rmsnorm,
    qwen4_hc_combine_norm,
    qwen4_hc_gate_reduce as _baseline_hc_gate_reduce,
    qwen4_hc_inject_combine,
)
from flaggems_vllm.ops.qwen4.hyperconnection import can_use_hc_combine_norm_triton

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


@triton.jit
def _hc_gate_reduce_persistent_kernel(
    logits, normed, output, rows, work_items,
    hidden: tl.constexpr, branches: tl.constexpr,
    row_tile: tl.constexpr, hidden_tile: tl.constexpr,
    hidden_tiles: tl.constexpr, workers: tl.constexpr,
):
    worker = tl.program_id(0)
    chunk = (work_items + workers - 1) // workers
    begin = worker * chunk
    end = tl.minimum(begin + chunk, work_items)
    for tile in tl.range(begin, end):
        row = (tile // hidden_tiles) * row_tile + tl.arange(0, row_tile)
        channel = (tile % hidden_tiles) * hidden_tile + tl.arange(0, hidden_tile)
        active = (row[:, None] < rows) & (channel[None, :] < hidden)
        input_row = row[:, None] * (hidden * branches)
        accumulator = tl.zeros((row_tile, hidden_tile), dtype=tl.float32)
        for branch in tl.static_range(0, branches):
            offset = input_row + branch * hidden + channel[None, :]
            gate = tl.load(logits + offset, active, other=0.0).to(tl.float32)
            value = tl.load(normed + offset, active, other=0.0).to(tl.float32)
            accumulator += tl.sigmoid(gate) * value
        tl.store(
            output + row[:, None] * hidden + channel[None, :],
            accumulator / branches,
            active,
        )


@lru_cache(maxsize=None)
def _a3_vector_cores(device: torch.device) -> int:
    if not torch.npu.get_device_name(device).startswith("Ascend910_93"):
        return 0
    return torch.npu.get_device_properties(device).vector_core_num


def qwen4_hc_gate_reduce(
    logits: torch.Tensor, normed: torch.Tensor, hc_count: int,
) -> torch.Tensor:
    if (
        getenv("VLLM_FL_QWEN38_PERSISTENT_HC_GATE") != "1"
        or logits.device.type != "npu"
        or logits.device != normed.device
        or logits.dtype != torch.bfloat16
        or normed.dtype != torch.bfloat16
        or logits.ndim != 2
        or logits.shape != normed.shape
        or hc_count != 4
        or logits.shape[1] != 10240
        or not 224 <= logits.shape[0] <= 4096
        or not logits.is_contiguous()
        or not normed.is_contiguous()
    ):
        return _baseline_hc_gate_reduce(logits, normed, hc_count)
    aiv_cores = _a3_vector_cores(logits.device)
    if aiv_cores < 1:
        return _baseline_hc_gate_reduce(logits, normed, hc_count)

    rows = logits.shape[0]
    hidden = 2560
    row_tile, hidden_tile = 8, 512
    hidden_tiles = triton.cdiv(hidden, hidden_tile)
    work_items = triton.cdiv(rows, row_tile) * hidden_tiles
    workers = min(aiv_cores, work_items)
    output = torch.empty((rows, hidden), dtype=normed.dtype, device=normed.device)
    _hc_gate_reduce_persistent_kernel[(workers,)](
        logits, normed, output, rows, work_items, hidden, hc_count,
        row_tile, hidden_tile, hidden_tiles, workers, num_warps=4,
    )
    logger.info_once("Qwen3.8 A3 uses persistent HyperConnection gate reduction")
    return output


def qwen4_grouped_gemma_rmsnorm_fake(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    hc_count: int,
    eps: float,
) -> torch.Tensor:
    del weight, hc_count, eps
    return torch.empty_like(hidden_states)


def qwen4_hc_gate_reduce_fake(
    logits: torch.Tensor,
    normed: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    del normed
    return torch.empty(
        (*logits.shape[:-1], logits.shape[-1] // hc_count),
        dtype=logits.dtype,
        device=logits.device,
    )


def qwen4_hc_inject_combine_fake(
    injection_logits: torch.Tensor,
    block_output: torch.Tensor,
    residual: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    del injection_logits, block_output, hc_count
    return torch.empty_like(residual)


def qwen4_hc_combine_norm_fake(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del block_output, injection_logits, norm_weight, eps, hc_count
    return torch.empty_like(residual), torch.empty_like(residual)


direct_register_custom_op(
    op_name="qwen4_grouped_gemma_rmsnorm",
    op_func=qwen4_grouped_gemma_rmsnorm,
    mutates_args=[],
    fake_impl=qwen4_grouped_gemma_rmsnorm_fake,
)
direct_register_custom_op(
    op_name="qwen4_hc_gate_reduce",
    op_func=qwen4_hc_gate_reduce,
    mutates_args=[],
    fake_impl=qwen4_hc_gate_reduce_fake,
)
direct_register_custom_op(
    op_name="qwen4_hc_inject_combine",
    op_func=qwen4_hc_inject_combine,
    mutates_args=[],
    fake_impl=qwen4_hc_inject_combine_fake,
)
direct_register_custom_op(
    op_name="qwen4_hc_combine_norm",
    op_func=qwen4_hc_combine_norm,
    mutates_args=[],
    fake_impl=qwen4_hc_combine_norm_fake,
)


__all__ = [
    "can_use_hc_combine_norm_triton",
    "qwen4_grouped_gemma_rmsnorm",
    "qwen4_hc_combine_norm",
    "qwen4_hc_gate_reduce",
    "qwen4_hc_inject_combine",
]
