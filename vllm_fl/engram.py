# SPDX-License-Identifier: Apache-2.0
"""Select the Ascend Qwen PLE implementation using native vLLM EngramConfig."""


def qwen_ple_offload_enabled(vllm_config):
    config = vllm_config.engram_config
    if config is None:
        return False
    if config.dp_shared_memory or config.embedding_across_dp or config.use_thp:
        raise ValueError(
            "FL Ascend Qwen PLE requires private host tables without THP or DP sharding"
        )
    return config.cpu_offload
