# SPDX-License-Identifier: Apache-2.0
"""DSv4 SWA-only merged draft graph, separate from target graph dispatch.

Metadata is built outside the graph. Every captured tensor address is retained;
only tensors whose addresses changed are copied before replay. Eligibility and
shape admission are agreed by all DP ranks in the proposer's packed collective.
"""
from dataclasses import fields, is_dataclass

import torch
from vllm.forward_context import override_forward_context
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


def tensor_leaves(value, path=()):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif value.__class__.__name__ == "RopeDataProxy":
        # FL RoPE proxy owns persistent per-step tensor views behind _data.
        yield from tensor_leaves(value._data, path + ("rope",))
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            yield from tensor_leaves(getattr(value, field.name), path + (field.name,))
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from tensor_leaves(child, path + (key,))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from tensor_leaves(child, path + (index,))


class MergedDraftGraph:
    # Exact shapes, no request/token padding and no mid-run cache eviction.
    # Limit additional graph/stream resources; unseen shapes use eager after this.
    MAX_GRAPHS = 4

    def __init__(self):
        self.entries = {}

    def run(self, key, runnable, inputs, contexts):
        if key is None or (key not in self.entries and len(self.entries) >= self.MAX_GRAPHS):
            return runnable(inputs, contexts)
        # Keep only the tensor-bearing metadata/slot mapping, not model modules,
        # config objects or process groups, in the replay binding traversal.
        bindings = (inputs, [(ctx.attn_metadata, ctx.slot_mapping,
                              ctx.additional_kwargs.get("mc2_mask")) for ctx in contexts])
        current = dict(tensor_leaves(bindings))
        if key not in self.entries:
            # Warm every operator/collective before capture. Same token positions
            # overwrite the same draft KV slots; no scheduler state is advanced.
            runnable(inputs, contexts)
            runnable(inputs, contexts)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            from vllm.model_executor.offloader.base import get_offloader
            get_offloader().sync_prev_onload()
            for ctx in contexts:
                ctx.capturing = True
                ctx.additional_kwargs["capturing"] = True
            try:
                with torch.npu.graph(graph, pool=current_platform.get_global_graph_pool()):
                    output = runnable(inputs, contexts)
                    get_offloader().join_after_forward()
            finally:
                for ctx in contexts:
                    ctx.capturing = False
                    ctx.additional_kwargs["capturing"] = False
            # Strong references own both graph input storage and output lifetime.
            self.entries[key] = (graph, output, current, inputs, contexts)
            logger.info("DSv4 merged MTP graph captured: first=%d, batch=%d, K=%d", *key)
            return output
        graph, output, captured, _, _ = self.entries[key]
        if current.keys() != captured.keys():
            raise RuntimeError("DSv4 graph metadata structure changed; use MTP_MERGED_GRAPH=0")
        copied = set()
        for path, source in current.items():
            destination = captured[path]
            if (destination.shape != source.shape or destination.stride() != source.stride()
                    or destination.dtype != source.dtype or destination.device != source.device):
                raise RuntimeError(f"DSv4 graph input layout changed at {path}; use MTP_MERGED_GRAPH=0")
            if destination.data_ptr() == source.data_ptr():
                continue
            pair = (destination.data_ptr(), source.data_ptr(), tuple(source.shape), tuple(source.stride()))
            if pair not in copied:
                destination.copy_(source, non_blocking=True)
                copied.add(pair)
        # DSA has no external attention-task parameter updates. Inputs, target
        # replay and this replay are ordered on the same worker stream.
        from vllm.model_executor.offloader.base import get_offloader
        get_offloader().sync_prev_onload()
        graph.replay()
        return output


def run_merged_draft(proposer, inputs, contexts):
    first_inputs, indices, positions = inputs
    ids_by_step = []
    contexts[0].moe_layer_index = 0
    with override_forward_context(contexts[0]):
        hidden = proposer.model(**first_inputs)
        hidden = hidden[indices]
        ids = proposer._greedy_sample(hidden)
    ids_by_step.append(ids)
    for step, next_positions in enumerate(positions, 1):
        contexts[step].moe_layer_index = 0
        with override_forward_context(contexts[step]):
            hidden = proposer.model(input_ids=ids, positions=next_positions,
                                    hidden_states=hidden, inputs_embeds=None)
            ids = proposer._greedy_sample(hidden)
        ids_by_step.append(ids)
    return torch.stack(ids_by_step, dim=1)
