# SPDX-License-Identifier: Apache-2.0
"""Same-layout M3 cache, indexer and sparse-attend kernel bundle.

Methods mutate caches/output and must be called directly once selected: no
dispatcher exception fallback is safe after a partial cache write. Metadata,
allocation and the enclosing opaque op remain owned by the model/runtime.
"""

import torch


class MiniMaxM3SparseKernels:
    cache_contract = "minimax_m3_paged_v1"
    replay_contract = "persistent_metadata_v1"

    @staticmethod
    def insert_main_cache(cache, slots, key, value, block_size, *, plane_major):
        from .linearnorm.minimax_m3_qknorm_rope import _insert_kv

        _insert_kv(cache, slots, key, value, block_size, plane_major=plane_major)

    @staticmethod
    def insert_index_cache(cache, slots, key, block_size):
        from .linearnorm.minimax_m3_qknorm_rope import _insert_index_k

        _insert_index_k(cache, slots, key, block_size)

    @staticmethod
    def run_attention(layer, query, index_query, output):
        # Keep the existing indexer + attend ordering and capture boundary.
        layer._run_attention(query, index_query, output)

    @staticmethod
    def attend(
        query, cache, topk, block_table, seq_lens, query_lens,
        num_kv_heads, scale, block_size, output, *, plane_major,
    ):
        if cache.ndim != 5 or cache.shape[0 if plane_major else 1] != 2:
            raise ValueError("MiniMax main cache does not match its declared layout")
        key, value = (cache[0], cache[1]) if plane_major else (cache[:, 0], cache[:, 1])
        result = torch.ops._C_ascend.npu_sparse_attention_score(
            query, key, value, topk, block_table,
            selectNumIdx=(topk >= 0).sum(dim=-1).to(dtype=torch.int32),
            actualSeqLengths=query_lens,
            actualSeqLengthsKv=seq_lens,
            numKeyValueHeads=num_kv_heads,
            scaleValue=scale,
            blockSize=block_size,
            topK=topk.shape[-1],
            innerPrecise=4,
        )
        output.copy_(result)
