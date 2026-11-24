# SPDX-License-Identifier: Apache-2.0
"""Sigmoid RMSNormGated dispatch for the Qwen3.8 Ascend GDN output."""

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import (
    LayerNameType,
    _resolve_layer_name,
    direct_register_custom_op,
)


logger = init_logger(__name__)


@triton.jit
def _rmsnorm_gate(
    X, Z, W, Output,
    X_TOKEN: tl.constexpr, X_HEAD: tl.constexpr,
    Z_TOKEN: tl.constexpr, Z_HEAD: tl.constexpr,
    HEADS: tl.constexpr, WIDTH: tl.constexpr, EPSILON: tl.constexpr,
    SIGMOID_ONLY: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // HEADS
    head = row % HEADS
    col = tl.arange(0, WIDTH)
    x = tl.load(X + token * X_TOKEN + head * X_HEAD + col).to(tl.float32)
    z = tl.load(Z + token * Z_TOKEN + head * Z_HEAD + col).to(tl.float32)
    w = tl.load(W + col).to(tl.float32)
    variance = tl.sum(x * x, 0) / WIDTH
    normed = x * tl.rsqrt(variance + EPSILON)
    weighted = normed * w
    gate = tl.sigmoid(z)
    if not SIGMOID_ONLY:
        gate = z * gate
    tl.store(Output + row * WIDTH + col, (weighted * gate).to(tl.bfloat16))


def fused_rmsnorm_gate(x, z, weight):
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    _rmsnorm_gate[(48,)](
        x, z, weight, output, x.stride(0), x.stride(1),
        z.stride(0), z.stride(1), 6, 128, 1e-6, True,
    )
    return output


def fl_qwen38_gdn_rmsnorm_gate(
    x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, layer_name: LayerNameType,
) -> torch.Tensor:
    context = get_forward_context()
    norm = context.no_compile_layers[_resolve_layer_name(layer_name)].norm
    if (
        context.cudagraph_runtime_mode == CUDAGraphMode.FULL
        and x.device.type == "npu"
        and x.device == z.device == weight.device
        and x.shape == z.shape == (8, 6, 128)
        and x.dtype == z.dtype == torch.bfloat16
        and x.is_contiguous()
        and z.stride() in ((768, 128, 1), (2048, 128, 1))
        and x.storage_offset() % 16 == z.storage_offset() % 16 == 0
        and weight.shape == (128,)
        and weight.is_contiguous()
        and weight.dtype in (torch.bfloat16, torch.float32)
        and norm.group_size is None
        and norm.norm_before_gate
        and norm.activation == "sigmoid"
        and norm.eps == 1e-6
        and norm.bias is None
    ):
        logger.info_once("Qwen3.8 GDN fused sigmoid RMSNormGated active (FULL [8])")
        return fused_rmsnorm_gate(x, z, weight)
    return norm(x, z)


def fl_qwen38_gdn_rmsnorm_gate_fake(
    x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, layer_name: LayerNameType,
) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="fl_qwen38_gdn_rmsnorm_gate",
    op_func=fl_qwen38_gdn_rmsnorm_gate,
    fake_impl=fl_qwen38_gdn_rmsnorm_gate_fake,
)
