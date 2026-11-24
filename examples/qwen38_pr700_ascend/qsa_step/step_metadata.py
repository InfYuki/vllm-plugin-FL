# SPDX-License-Identifier: Apache-2.0
"""Prepare the BF16 Qwen LightningIndexerV2 schedule once per MRV2 step."""

import torch
from vllm.logger import logger

from vllm_ascend.ops.lightning_indexer_v2 import lightning_indexer_v2_metadata


def _view(tensor):
    # data_ptr includes the storage offset. Distinct sliced views must not alias.
    return tensor.data_ptr(), tensor.shape, tensor.stride(), tensor.dtype, tensor.device


class Qwen4ExpStepMetadata:
    def __init__(self, model):
        self._consumers = []
        for layer in model.modules():
            if type(layer).__name__ != "AscendQwen4ExpQSAAttention":
                continue
            indexer = layer.indexer
            if indexer.skip_topk:
                continue
            geometry = (indexer.index_n_heads, indexer.index_head_dim, indexer.token_topk, indexer.compress_ratio)
            self._consumers.append((indexer.compressed_key_cache.prefix, geometry))
        self._graph_buffers = {}
        self._logged = set()

    def prepare(self, metadata_by_layer, *, graph_tokens, phase=None):
        # The builder binds the same metadata instance to all layers of an
        # attention group. Its identities and lengths can change every step.
        seen = set()
        generated = {}
        consumers = 0
        for prefix, geometry in self._consumers:
            md = metadata_by_layer.get(prefix)
            if md is None:
                continue
            consumers += 1
            if id(md) in seen:
                continue
            seen.add(id(md))
            if graph_tokens is None:
                requests = getattr(md, "qsa_num_live_requests", md.seq_lens.shape[0])
                rows = getattr(md, "qsa_num_live_tokens", md.num_actual_tokens)
            else:
                # Captured operator arguments retain the padded shape even
                # when fewer requests are live on replay (including dummy DP).
                requests = md.seq_lens.shape[0]
                rows = graph_tokens
            md.qsa_lightning_v2_shape = (requests, rows)
            md.qsa_lightning_v2_metadata = None
            if not requests or not rows:
                continue
            lengths = md.seq_lens[:requests]
            starts = md.query_start_loc[: requests + 1]
            heads, dim, topk, ratio = geometry
            key = geometry, rows, _view(lengths), _view(starts)
            schedule = generated.get(key)
            if schedule is None:
                schedule = lightning_indexer_v2_metadata(
                    heads, 1, dim, topk // ratio, batch_size=requests,
                    cu_seqlens_q=starts, seqused_k=lengths.floor_divide(ratio),
                    cmp_residual_k=lengths.remainder(ratio),
                    layout_q="TND", layout_k="PA_BBND", mask_mode=3, cmp_ratio=ratio,
                )
                generated[key] = schedule
            if graph_tokens is not None:
                # The first compressed-cache owner names a stable builder
                # group. Graph storage must survive changes of input addresses
                # and live counts. Independent groups keep independent outputs.
                bucket = prefix, geometry, requests, graph_tokens, schedule.device
                buffer = self._graph_buffers.get(bucket)
                if buffer is None:
                    buffer = torch.empty_like(schedule)
                    self._graph_buffers[bucket] = buffer
                buffer.copy_(schedule)
                schedule = buffer
            md.qsa_lightning_v2_metadata = schedule
        if phase is not None and phase not in self._logged and generated:
            logger.info(
                "Qwen BF16 QSA step metadata: phase=%s consumers=%d schedules=%d",
                phase, consumers, len(generated),
            )
            self._logged.add(phase)
