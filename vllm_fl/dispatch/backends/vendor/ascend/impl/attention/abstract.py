# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod
from typing import Any, Generic, TypeVar

import torch
from vllm.v1.attention.backend import AttentionImpl


class AttentionMetadata:
    pass


T = TypeVar("T", bound=AttentionMetadata)


class DSAAttentionImpl(AttentionImpl[T], Generic[T]):
    @abstractmethod
    def __init__(
        self,
        dim: int,
        n_heads: int,
        scale: float,
        n_local_heads: int,
        q_lora_rank: int,
        o_lora_rank: int,
        head_dim: int,
        rope_head_dim: int | None,
        nope_head_dim: int,
        n_groups: int,
        n_local_groups: int,
        window_size: int,
        compress_ratio: int,
        **extra_impl_args: Any,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(  # type: ignore[override]
        self,
        layer_name: str,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor | None, ...] | None,
        attn_metadata: list[T] | None,
        need_gather_q_kv: bool = False,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Write output and update the provider's shared/auxiliary caches.

        Profiling supplies None for caches/metadata. Projection, indexer and
        compressor modules arrive through extra_impl_args at construction;
        this is deliberately not the generic Q/K/V AttentionImpl ABI.
        """
        raise NotImplementedError
