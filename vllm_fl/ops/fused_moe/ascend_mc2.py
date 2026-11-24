# SPDX-License-Identifier: Apache-2.0
"""A3 ordinary BF16 MC2, following vLLM-Ascend v0.32.0rc1's dispatcher."""

from functools import lru_cache

import torch
import torch.distributed as dist
import torch_npu
from torch.nn import functional as F

from .ascend_alltoallv import bf16_expert_mlp


@lru_cache(maxsize=1)
def _mc2_group(ep_device_group):
    # Model construction is ordered across the complete EP world. Use a
    # separate communicator for native MC2, as the official dispatcher does.
    return dist.new_group(ranks=dist.get_process_group_ranks(ep_device_group), backend="hccl")


class BF16MC2:
    def __init__(self, ep_group, num_experts):
        if not torch.npu.get_device_name().startswith(("Ascend910_93", "Ascend910C")):
            raise ValueError("Qwen3.8 BF16 MC2 requires Ascend A3")
        if dist.get_world_size() != 16 or ep_group.world_size != 16:
            raise ValueError("Qwen3.8 BF16 MC2 requires one 16-rank EP world")
        if not all(
            callable(getattr(torch_npu, name, None))
            for name in ("npu_moe_distribute_dispatch_v2", "npu_moe_distribute_combine_v2")
        ):
            raise RuntimeError("Qwen3.8 BF16 MC2 needs torch_npu dispatch/combine v2")
        group = _mc2_group(ep_group.device_group)
        rank = dist.get_rank(group)
        if rank != ep_group.rank_in_group:
            raise ValueError("MC2 group ordering must match the EP weights")
        comm = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        self._kwargs = dict(
            group_ep=comm, ep_world_size=ep_group.world_size, ep_rank_id=rank,
            group_tp=comm, tp_world_size=1, tp_rank_id=0,
            expert_shard_type=0, shared_expert_rank_num=0, moe_expert_num=num_experts,
            global_bs=0, comm_alg="",
        )

    def __call__(self, hidden_states, weights, ids, w13, w2, active_mask):
        rows = active_mask.numel()
        local_rows, top_k = ids.shape
        if active_mask.ndim != 1 or active_mask.dtype != torch.bool or active_mask.device != ids.device:
            raise ValueError("MC2 requires a rank-local device bool mask")
        if local_rows > rows:
            raise ValueError("MC2 received more SP rows than coordinated by DP metadata")
        ids = ids.to(torch.int32)
        pad = rows - local_rows
        if pad:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad))
            ids = F.pad(ids, (0, 0, 0, pad))
            weights = F.pad(weights, (0, 0, 0, pad))
        # Give masked-out graph rows distinct in-range IDs, including dummy rows.
        dummy_ids = torch.arange(top_k, device=ids.device, dtype=torch.int32)
        ids = torch.where(active_mask[:, None], ids, dummy_ids)
        kwargs = dict(
            self._kwargs,
            expert_ids=ids.contiguous(),
            x_active_mask=active_mask,
        )
        dispatch = torch_npu.npu_moe_distribute_dispatch_v2(
            x=hidden_states.contiguous(), quant_mode=0, expert_token_nums_type=0, **kwargs,
        )
        expanded, _, assist, prefix_counts, ep_counts, tp_counts, scales = dispatch[:7]
        expert_output = bf16_expert_mlp(
            expanded, prefix_counts, w13, w2, group_list_type=0,
        )
        result = torch_npu.npu_moe_distribute_combine_v2(
            expand_x=expert_output,
            assist_info_for_combine=assist,
            ep_send_counts=ep_counts,
            tp_send_counts=tp_counts,
            expand_scales=scales,
            expert_scales=weights.to(torch.float32).contiguous(),
            comm_quant_mode=0,
            **kwargs,
        )
        # The native combine does not initialize output for masked rows.
        # Clear them before passing the graph output to the next MoE layer.
        return result[:local_rows].masked_fill(~active_mask[:local_rows, None], 0)
