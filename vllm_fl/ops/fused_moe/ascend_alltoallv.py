# SPDX-License-Identifier: Apache-2.0
"""A3 BF16 routed-expert exchange based on vLLM-Ascend's AllToAllV dispatcher."""

from contextlib import contextmanager
from contextvars import ContextVar
from os import getenv

import torch
import torch.distributed as dist
import torch_npu
from torch.nn import functional as F
from vllm.config import CUDAGraphMode
from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import FusedTopKRouter
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

logger = init_logger("vllm.fl.qwen38_alltoallv")
_balance_profile = ContextVar("qwen38_alltoallv_balance_profile", default=False)


@contextmanager
def balance_alltoallv_profile():
    token = _balance_profile.set(True)
    try:
        yield
    finally:
        _balance_profile.reset(token)


def _profile_expert_ids(ids, rank, local_experts, num_experts):
    # Distribute the dummy batch across experts with distinct experts per row.
    # Stagger the start for each rank to balance even a short profile batch.
    offset = rank * local_experts
    positions = torch.arange(ids.numel(), dtype=torch.int32, device=ids.device)
    return ((positions + offset) % num_experts).view_as(ids)


def bf16_expert_mlp(expanded, group_list, gate_up_weight, down_weight, *, group_list_type=1):
    kwargs = dict(
        split_item=2, group_type=0, group_list_type=group_list_type, group_list=group_list,
    )
    gate_up = torch_npu.npu_grouped_matmul(
        x=[expanded], weight=[gate_up_weight.transpose(1, 2)], **kwargs
    )[0]
    return torch_npu.npu_grouped_matmul(
        x=[torch_npu.npu_swiglu(gate_up)], weight=[down_weight.transpose(1, 2)], **kwargs
    )[0]


def _exchange(rows, send, receive, group):
    output = rows.new_empty((sum(receive), rows.shape[1]))
    request = dist.all_to_all_single(
        output, rows.contiguous(), output_split_sizes=receive,
        input_split_sizes=send, group=group, async_op=True,
    )
    request.wait()
    return output


def alltoallv_experts(hidden_states, weights, ids, gate_up_weight, down_weight, group):
    """Route TP-sharded rows to contiguous EP expert ranges and restore source rows."""
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    local_experts = gate_up_weight.shape[0]
    num_experts = world * local_experts
    # vLLM 0.32 uses -1 for padded routes. Retain a fixed route count
    # while preventing padded rows from contributing to the result.
    weights = weights.masked_fill(ids < 0, 0)
    ids = ids.to(torch.int32).clamp_min(0).contiguous()
    num_routes = ids.numel()
    if num_routes:
        counts = torch.histc(ids, bins=num_experts, min=0, max=num_experts)
        counts = counts.to(torch.int32).contiguous()
    else:
        counts = ids.new_zeros(num_experts)
    global_counts = counts.new_empty(world * num_experts)
    dist.all_gather_into_tensor(global_counts, counts, group=group)

    by_source = global_counts.view(world, world, local_experts)
    receive_counts = by_source[:, rank].contiguous()
    send_splits = by_source[rank].sum(dim=-1, dtype=torch.int32)
    receive_splits = receive_counts.sum(dim=-1, dtype=torch.int32)
    host_splits = torch.empty((2, world), dtype=torch.int32, device="cpu", pin_memory=True)
    host_splits[0].copy_(send_splits, non_blocking=True)
    host_splits[1].copy_(receive_splits, non_blocking=True)
    splits_ready = torch.npu.Event()
    splits_ready.record()

    # Dispatch rows in global expert order. Keep the source permutation and
    # weights here so the return exchange needs only the BF16 payload.
    if num_routes:
        rows, source_order = torch_npu.npu_moe_token_permute(
            tokens=hidden_states, indices=ids, num_out_tokens=num_routes,
        )
    else:
        rows = hidden_states
    # The permutation does not depend on the splits. Submit it before waiting
    # for the two host vectors required by all_to_all_single.
    splits_ready.synchronize()
    send, receive = host_splits.tolist()
    incoming = _exchange(rows, send, receive, group)

    if incoming.shape[0]:
        labels = torch.arange(local_experts, dtype=torch.int32, device=ids.device).repeat(world)
        labels = labels.repeat_interleave(
            receive_counts.flatten(), output_size=incoming.shape[0],
        )
        expert_rows, receive_order = torch_npu.npu_moe_token_permute(incoming, labels)
        expert_counts = receive_counts.sum(dim=0, dtype=torch.int64)
        expert_output = bf16_expert_mlp(expert_rows, expert_counts, gate_up_weight, down_weight)
        incoming = torch_npu.npu_moe_token_unpermute(expert_output, receive_order)

    returned = _exchange(incoming, receive, send, group)
    if not num_routes:
        return hidden_states
    return torch_npu.npu_moe_token_unpermute(
        permuted_tokens=returned,
        sorted_indices=source_order,
        probs=weights.to(hidden_states.dtype),
        restore_shape=hidden_states.shape,
    )


class AscendPrefillAllToAllVMoERunner(MoERunner):
    """Select the configured BF16 exchange using coordinated DP row counts."""

    def __init__(self, *args, use_alltoallv=True, use_mc2=False, **kwargs):
        super().__init__(*args, **kwargs)
        moe = self.moe_config
        if (
            not moe.is_sequence_parallel
            or (moe.sp_size, moe.dp_size, moe.ep_size) != (8, 2, 16)
            or moe.in_dtype != torch.bfloat16
            or moe.num_local_experts * moe.ep_size != moe.num_experts
            or moe.activation != MoEActivation.SILU
            or not moe.is_act_and_mul
            or moe.has_bias
            or moe.swiglu_limit is not None
            or self.routed_experts.apply_router_weight_on_input
        ):
            raise ValueError("Unsupported Qwen3.8 BF16 expert exchange layout")
        group = get_ep_group()
        if list(group.ranks) != list(range(16)):
            raise ValueError("Qwen3.8 AllToAllV needs the ordered 16-rank EP group")
        self._alltoallv_group = group.device_group
        self._alltoallv_rank = group.rank_in_group
        self._use_alltoallv = use_alltoallv
        self._native_decode = getenv("VLLM_FL_QWEN38_NATIVE_DECODE") == "1"
        self._mc2 = None
        if use_mc2:
            from .ascend_mc2 import BF16MC2

            self._mc2 = BF16MC2(group, moe.num_experts)
        if use_alltoallv:
            logger.info_once("Qwen3.8 BF16 prefill AllToAllV runner selected")

    def _forward_impl(self, hidden_states, router_logits, shared_experts_input, input_ids=None):
        context = get_forward_context()
        args = (hidden_states, router_logits, shared_experts_input, input_ids)
        mode = context.cudagraph_runtime_mode
        if context.dp_metadata is None or mode not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL):
            return super()._forward_impl(*args)
        if self._mc2 is None and mode != CUDAGraphMode.NONE:
            return super()._forward_impl(*args)
        with self._sequence_parallel_context():
            max_rows = max(context.dp_metadata.get_chunk_sizes_across_dp_rank())
        profiling = _balance_profile.get()
        # The two DPs must choose the same collective. The common SP
        # row bound also covers idle ranks and graph padding.
        eager_mc2 = (
            self._native_decode
            and mode == CUDAGraphMode.NONE
            and "qwen38_mc2_active_mask" in context.additional_kwargs
        )
        use_mc2 = (
            self._mc2 is not None
            and 0 < max_rows <= 2
            and (mode == CUDAGraphMode.FULL or eager_mc2)
        )
        use_alltoallv = self._use_alltoallv and mode == CUDAGraphMode.NONE and max_rows > 4
        if not (use_mc2 or use_alltoallv):
            return super()._forward_impl(*args)

        mc2_mask = None
        self.routed_experts._ensure_moe_quant_config_init()
        if use_mc2:
            # The runner refreshes this captured input in-place before replay.
            mc2_mask = context.additional_kwargs["qwen38_mc2_active_mask"][:max_rows]
            if mc2_mask.numel() != max_rows:
                raise ValueError("MC2 active mask is smaller than the coordinated SP batch")
            logger.info_once("Qwen3.8 BF16 MC2 dispatch active (%s)", mode.name)
        elif profiling:
            logger.info_once("Qwen3.8 BF16 prefill AllToAllV profile dispatch active")
        else:
            logger.info_once("Qwen3.8 BF16 prefill AllToAllV dispatch active (async splits)")
        if hidden_states.shape[0] == 0:
            # A rank with no source rows still receives and computes expert rows.
            shape = (0, self.moe_config.experts_per_token)
            ids = hidden_states.new_empty(shape, dtype=torch.int32)
            weights = hidden_states.new_empty(shape, dtype=torch.float32)
            routed = self._exchange_experts(hidden_states, weights, ids, mc2_mask)
            return (shared_experts_input, routed) if self.shared_experts else routed

        shared = self.shared_experts
        shared_async = shared is not None and shared.maybe_forward_async(shared_experts_input)
        shared_output = (
            shared._layer(shared_experts_input)
            if shared is not None and not shared_async
            else None
        )
        if self.gate is not None:
            if self._fse_fuse_gate:
                self._maybe_fuse_gate_weights()
                router_logits = F.linear(hidden_states, self._combined_gate_weight)
            else:
                router_logits, _ = self.gate(hidden_states)
        padding = context.is_padding
        if use_mc2:
            # FL's general padding mask marks every idle-DP dummy token and
            # sends -1 IDs to its dispatchers. MC2 dummy rows instead follow
            # the dedicated TP-local active mask and require valid expert IDs.
            context.is_padding = None
        try:
            router = self.router
            if (
                use_mc2
                and self._native_decode
                and not profiling
                and isinstance(router, FusedTopKRouter)
                and router.scoring_func == "softmax"
                and router.eplb_state is None
                and getattr(router, "capture_fn", None) is None
            ):
                weights, ids, _ = torch_npu.npu_moe_gating_top_k(
                    router_logits.float().contiguous(),
                    k=router.top_k,
                    k_group=1,
                    group_count=1,
                    group_select_mode=1,
                    renorm=int(router.renormalize),
                    norm_type=0,
                    out_flag=False,
                )
                logger.info_once("Qwen3.8 BF16 native fused decode router active")
            else:
                weights, ids = router.select_experts(
                    hidden_states=hidden_states, router_logits=router_logits,
                    topk_indices_dtype=torch.int32, input_ids=input_ids,
                )
        finally:
            if use_mc2:
                context.is_padding = padding
        if profiling:
            ids = _profile_expert_ids(
                ids, self._alltoallv_rank, self.moe_config.num_local_experts,
                self.moe_config.num_experts,
            )
        routed = self._exchange_experts(hidden_states, weights, ids, mc2_mask)
        if shared_async:
            shared.wait()
            shared_output = shared.output
        return (shared_output, routed) if shared is not None else routed

    def _exchange_experts(self, hidden_states, weights, ids, mc2_mask):
        experts = self.routed_experts
        if mc2_mask is not None:
            return self._mc2(
                hidden_states, weights, ids, experts.w13_weight, experts.w2_weight, mc2_mask,
            )
        return alltoallv_experts(
            hidden_states, weights, ids, experts.w13_weight, experts.w2_weight,
            self._alltoallv_group,
        )
