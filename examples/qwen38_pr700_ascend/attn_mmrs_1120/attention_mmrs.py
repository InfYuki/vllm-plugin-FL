# SPDX-License-Identifier: Apache-2.0
"""TP8 output projection for the BF16 Qwen attention layers in this candidate."""

from functools import lru_cache

import torch
import torch.nn.functional as F
import torch_npu
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import direct_register_custom_op
from vllm_ascend.models.common.ops.sequence_parallel import sp_reduce_scatter
from vllm_ascend.ops.linear import AscendRowParallelLinear, AscendUnquantizedLinearMethod
from vllm_ascend.ops.linear_op import CustomRowParallelOp

# Initializing the official ops package first also supports a cold import from model_state.
from vllm_ascend.device.device_op import DeviceOperator

TP_SIZE = 8
MIN_PREFILL_ROWS = 1024
logger = init_logger(__name__)
_logged_kinds: set[str] = set()


class LongPrefillMetadata(dict):
    """The same per-layer metadata objects, marked only for this real eager step."""


def mark_long_prefill(metadata, batch, mode, for_capture):
    if (
        mode == CUDAGraphMode.NONE and not for_capture and not batch.is_dummy
        and batch.num_tokens >= MIN_PREFILL_ROWS and batch.num_tokens > batch.num_reqs
        and isinstance(metadata, dict) and metadata
    ):
        return LongPrefillMetadata(metadata)
    return metadata


@lru_cache(maxsize=1)
def _tp_hcom(tp):
    # Keep the process's HCCL communicator outside persistent FX graphs.
    backend = tp.device_group._get_backend(torch.device("npu"))
    return backend.get_hccl_comm_name(tp.rank_in_group)


def _project_and_shard(x: torch.Tensor, weight: torch.Tensor, kind: str) -> torch.Tensor:
    # This choice must see the current step, even when the eager model runs cached FX.
    rows = x.shape[0]
    ctx = get_forward_context() if rows >= MIN_PREFILL_ROWS and is_forward_context_available() else None
    if (ctx is None or ctx.cudagraph_runtime_mode != CUDAGraphMode.NONE
            or not isinstance(ctx.attn_metadata, LongPrefillMetadata) or torch.npu.is_current_stream_capturing()):
        return sp_reduce_scatter(torch.ops.vllm.unquantized_gemm(x, weight, None))

    # Shape, dtype and TP domain are fixed when binding the bias-free projection.
    # Zero input rows produce the same zero partials as padding after the original GEMM.
    pad = (-rows) % TP_SIZE
    if torch_npu.get_npu_format(x) != 2:  # ACL_FORMAT_ND
        x = torch_npu.npu_format_cast(x, 2)
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
    output = DeviceOperator.npu_mm_reduce_scatter_base(
        x.contiguous(), weight.t(), _tp_hcom(get_tp_group()), TP_SIZE,
        reduce_op="sum", bias=None, comm_turn=0, comm_mode="aiv",
    )
    if kind not in _logged_kinds:
        logger.info("Qwen attention MMRS executed kind=%s rows=%d padded_rows=%d TP8 AIV", kind, rows, rows + pad)
        _logged_kinds.add(kind)
    return output


def _project_and_shard_fake(x: torch.Tensor, weight: torch.Tensor, kind: str) -> torch.Tensor:
    return x.new_empty((cdiv(x.shape[0], TP_SIZE), weight.shape[0]))


direct_register_custom_op(
    op_name="fl_qwen38_attn_project_rs", op_func=_project_and_shard, mutates_args=[],
    fake_impl=_project_and_shard_fake, dispatch_key="PrivateUse1",
)


class QwenAttentionRowOp(CustomRowParallelOp):
    def __init__(self, layer, kind):
        super().__init__(layer)
        self.update_attrs()
        self.kind = kind
        _tp_hcom(get_tp_group())

    def apply_impl(self, input_):
        return torch.ops.vllm.fl_qwen38_attn_project_rs(input_, self.layer.weight, self.kind), None


def bind_attention_projection(layer, kind, config):
    """Only the target decoder's projection may take ownership of the outer ReduceScatter."""
    if not (
        isinstance(layer, AscendRowParallelLinear) and type(layer.quant_method) is AscendUnquantizedLinearMethod
        and layer.custom_op is None and layer.reduce_results is False and layer.input_is_parallel
        and layer.return_bias and layer.bias is None and layer.tp_size == TP_SIZE
        and tuple(layer.weight.shape) == (2560, 768) and layer.weight.dtype == torch.bfloat16
        and config.quant_config is None and config.lora_config is None and config.speculative_config is None
    ):
        raise ValueError(f"Unsupported Qwen attention MMRS projection at {layer.prefix}")
    layer.custom_op = QwenAttentionRowOp(layer, kind)
    logger.info_once("Qwen attention MMRS bound kind=%s TP8 BF16", kind)
