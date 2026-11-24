# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-vendor QSA owner with FlagTree/Triton kernels."""

from __future__ import annotations

from dataclasses import replace
from os import getenv
from typing import Any, ClassVar, cast

import torch
from torch import nn

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config.cache import CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import (
    set_default_quant_scales,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    canonicalize_singleton_dim_strides,
    direct_register_custom_op,
    get_dtype_size,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionType,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

from ..common.qsa_cache import QSAForwardMetadata, QSAMetadataBuilder
from .indexer_qsa import QSAIndexer
from .nvidia_fast_paths import has_native_cache_update, native_cache_update
from .quantization import without_modelopt_fp4


def _prepared_qsa_token_to_req(
    metadata: dict[str, Any],
    indexer: QSAIndexer,
    num_tokens: int,
) -> torch.Tensor:
    """Use the request map filled by the compressed metadata producer.

    On GPU the raw cache owns a separate zero-initialized buffer which is
    never prepared. Reading it would route every token to request row zero.
    Call this after the indexer has prepared the compressed metadata.
    """
    compressed = cast(
        QSAForwardMetadata, metadata[indexer.compressed_key_cache.prefix]
    )
    if compressed.num_actual_tokens != num_tokens:
        raise RuntimeError("QSA main and compressed metadata token counts disagree")
    return compressed.token_to_req[:num_tokens]


def _unpack_qsa_kv_cache(
    kv_cache: torch.Tensor,
    num_kv_heads: int,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return allocator-backed K/V views in [blocks, tokens, heads, dim]."""
    if kv_cache.ndim == 4 and kv_cache.shape[1] == 2:
        if kv_cache.shape[-1] != num_kv_heads * head_size:
            raise ValueError("QSA packed K/V width does not match the cache spec")
        key_cache, value_cache = kv_cache.unflatten(
            -1, (num_kv_heads, head_size)
        ).unbind(1)
    elif (
        kv_cache.ndim == 5
        and kv_cache.shape[1] == 2
        and kv_cache.shape[-2:] == (num_kv_heads, head_size)
    ):
        key_cache, value_cache = kv_cache.unbind(1)
    else:
        raise ValueError(f"Invalid QSA K/V cache shape: {tuple(kv_cache.shape)}")
    return (
        canonicalize_singleton_dim_strides(key_cache),
        canonicalize_singleton_dim_strides(value_cache),
    )


class Qwen3_8FlashNextQSAAttentionBackend(AttentionBackend):
    """Main K/V cache owner for the Triton QSA transaction.

    QSA performs cache update and sparse attention in its model custom op, so
    it needs vLLM only for cache allocation and device-side metadata building.
    Owning those interfaces directly avoids coupling the model to a vendor's
    FlashAttention extension or cache-update ABI.
    """

    # QSA device operations are captured by the model, with no task updater.
    model_owned_graph_execution: ClassVar[bool] = True

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return "QWEN38_FLASH_NEXT_QSA_FLAGTREE"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError(
            "QSA executes out-of-band through its model-owned Triton transaction"
        )

    @staticmethod
    def get_builder_cls() -> type[QSAMetadataBuilder]:
        return QSAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        del cache_dtype_str
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @classmethod
    def indexes_kv_by_block_stride(cls) -> bool:
        # The allocator currently owns an identity/layered layout.  Marking
        # this as block-indexed makes vLLM apply the cross-layer uniform
        # packing contract, which this cache does not satisfy.
        return False

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3, 4, 5)
        return (0, 1, 2, 3, 4)

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        if spec.head_size != spec.head_size_v:
            raise ValueError("QSA K/V slots require equal key and value widths")
        slot_bytes = spec.num_kv_heads * spec.head_size * get_dtype_size(spec.dtype)
        return replace(spec, num_head_slots=2, state_content_bytes=slot_bytes)

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # Block-outermost layouts support mixed page sizes without changing the
        # logical block sizes of QSA side caches or Mamba align states.
        return (KVCacheLayout.BLNHC, KVCacheLayout.BLHNC)

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False


class Qwen3_8FlashNextQSAAttention(Qwen3NextAttention, AttentionLayerBase):
    """Merged Qwen full-attention owner with a QSA index side branch."""

    supports_dcp = False

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Any,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = False,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        if cache_config is None:
            raise ValueError("Qwen3.8-Flash-Next QSA requires a paged KV cache")
        if model_config.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen3.8-Flash-Next QSA currently requires BF16")
        if cache_config.cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA requires a BF16 main KV cache"
            )
        if getattr(quant_config, "kv_cache_scheme", None) is not None:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA does not support KV quantization"
            )
        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.prefill_context_parallel_size > 1
            or parallel_config.decode_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA does not support context parallelism"
            )
        if not getattr(config, "is_causal", True):
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA requires causal decoder attention"
            )

        self.config = config
        self.hidden_size = int(config.hidden_size)
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = int(config.num_attention_heads)
        if self.total_num_heads % tp_size:
            raise ValueError("QSA attention heads must be divisible by TP size")
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError("QSA KV heads must be divisible by TP size")
        elif tp_size % self.total_num_kv_heads:
            raise ValueError("TP size must be divisible by replicated QSA KV heads")
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(config.head_dim or self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        if self.dual_chunk_attention_config is not None:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA does not support dual-chunk RoPE"
            )
        # Qwen3.8-Flash-Next full-attention checkpoints always pack a sigmoid output
        # gate next to Q, even when an inherited config default says otherwise.
        self.attn_output_gate = True

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=without_modelopt_fp4(quant_config),
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        mm_config = model_config.multimodal_config
        text_only = mm_config is None or mm_config.language_model_only
        self.use_fused_qk_norm_rope_gate = (
            self.attn_output_gate
            and getattr(self.rotary_emb, "is_neox_style", False)
            and current_platform.is_cuda()
            and text_only
        )

        self.layer_name = f"{prefix}.attn"
        self.attn_type = AttentionType.DECODER
        self.kv_cache_dtype = cache_config.cache_dtype
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, model_config
        )
        if self.kv_cache_torch_dtype != torch.bfloat16:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA requires BF16 cache storage"
            )
        self.kv_sharing_target_layer_name = None
        self.kv_cache = torch.tensor([])
        set_default_quant_scales(self, register_buffer=True)

        self.attn_backend = Qwen3_8FlashNextQSAAttentionBackend
        self.indexer = QSAIndexer(
            vllm_config=vllm_config,
            config=config,
            layer_id=layer_id,
            rotary_emb=self.rotary_emb,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer",
        )
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.register_buffer(
            "topk_indices_buffer",
            torch.empty(
                max_tokens,
                self.indexer.output_width,
                dtype=torch.int32,
            ),
            persistent=False,
        )
        # Keep one caller-owned partial workspace per layer and graph bucket.
        # The global vLLM workspace is shared by sequential layers; retaining
        # these views here prevents a later rows/TopK bucket from resizing the
        # allocation referenced by an already-captured graph.
        self._qsa_split_workspaces: dict[
            tuple[int, int, int, int, int, int],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}

        static_context = vllm_config.compilation_config.static_forward_context
        if self.layer_name in static_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        static_context[self.layer_name] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    def _run_qsa(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        context = get_forward_context()
        metadata = context.attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            output.zero_()
            return
        main_metadata = cast(QSAForwardMetadata, metadata[self.layer_name])
        if self.kv_cache.numel() == 0:
            raise RuntimeError("QSA main K/V cache is not bound")

        num_tokens = main_metadata.num_actual_tokens
        side_metadata = cast(
            QSAForwardMetadata,
            metadata[self.indexer.raw_key_cache.prefix],
        )
        if side_metadata.num_actual_tokens != num_tokens:
            raise RuntimeError("QSA main and side metadata token counts disagree")
        key_cache, value_cache = _unpack_qsa_kv_cache(
            self.kv_cache, self.num_kv_heads, self.head_dim
        )
        if key_cache.dtype != torch.bfloat16 or query.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen3.8-Flash-Next QSA requires BF16 Q/K/V")

        common = main_metadata.common
        visible_metadata = None
        prefill_plan = None
        all_visible_fia = False
        if (
            num_tokens
            and query.device.type == "npu"
            and self.indexer._use_native_prefill
            and context.skip_compiled
            and context.cudagraph_runtime_mode == CUDAGraphMode.NONE
            and common.causal is True
        ):
            import torch_npu
            from .ops.qsa_attention_native import build_qsa_request_plan

            if not hasattr(common, "_fl_qsa_prefill_plan"):
                common._fl_qsa_prefill_plan = build_qsa_request_plan(
                    common.query_start_loc_cpu,
                    common.seq_lens_cpu_upper_bound,
                    num_tokens,
                    common.is_prefilling,
                )
            prefill_plan = common._fl_qsa_prefill_plan
            all_visible_fia = (
                prefill_plan is not None
                and callable(getattr(torch_npu, "npu_fused_infer_attention_score_v2", None))
                and query.shape[1] % key.shape[1] == 0
                and all(
                    request.kv_length <= self.indexer.token_topk
                    and request.kv_length <= main_metadata.block_table.shape[1] * key_cache.shape[1]
                    and request.index < main_metadata.block_table.shape[0]
                    for request in prefill_plan.requests
                )
            )
        if (
            num_tokens
            and self.indexer._use_native_decode
            and self.indexer._qsa_graph_max_seq_len
            and context.cudagraph_runtime_mode == CUDAGraphMode.FULL
            and common.causal is True
            and main_metadata.block_table.numel()
        ):
            from .ops.qsa_decode import supports_visible_decode, visible_paged_attention

            if supports_visible_decode(query, key_cache):
                visible_metadata = self.indexer.prepare_visible_decode(hidden_states, positions)

        if all_visible_fia:
            self.indexer.prepare_visible_decode(hidden_states, positions)
            selected = None
            token_to_req = _prepared_qsa_token_to_req(metadata, self.indexer, num_tokens)
        elif visible_metadata is None:
            selected = self.indexer(
                hidden_states,
                positions,
                self.topk_indices_buffer[:num_tokens],
            )
            token_to_req = _prepared_qsa_token_to_req(metadata, self.indexer, num_tokens)
            if selected.shape != (num_tokens, self.indexer.output_width):
                raise RuntimeError("QSA indexer returned an invalid selection shape")

        from .ops.qsa import (
            qsa_prepare_split_workspace,
            qsa_sparse_paged_attention,
            qsa_store_kv_cache_rows,
        )

        slot_mapping = main_metadata.slot_mapping[:num_tokens]
        if num_tokens and has_native_cache_update():
            # Isolated NVIDIA fast path: keep the model-owned QSA backend and
            # use only vLLM's measured fused cache-update operator.
            native_cache_update(
                key[:num_tokens],
                value[:num_tokens],
                key_cache,
                value_cache,
                slot_mapping,
                self.kv_cache_dtype,
                self._k_scale,
                self._v_scale,
            )
        else:
            # Cross-vendor fallback: one stride-aware launch writes K and V
            # without flattening the allocator-owned head/dim layout.
            qsa_store_kv_cache_rows(
                key_cache,
                value_cache,
                slot_mapping,
                key[:num_tokens],
                value[:num_tokens],
            )

        output.zero_()
        if visible_metadata is not None:
            attention = visible_paged_attention
            if getenv("VLLM_FL_QWEN38_FULL_FIA") == "1":
                from .ops.qsa_fia_full import full_fia_attention

                attention = full_fia_attention
            attention(
                query[:num_tokens], key_cache, value_cache, main_metadata.block_table,
                visible_metadata.token_to_req[:num_tokens],
                visible_metadata.logical_positions[:num_tokens], visible_metadata.seq_lens,
                output[:num_tokens], gate[:num_tokens], self.scaling,
                self.indexer._qsa_graph_max_seq_len,
            )
            return
        if prefill_plan is not None:
            from .ops.qsa import _qsa_get_split_workspace
            from .ops.qsa_attention_native import qsa_prefill_attention

            compressed_metadata = cast(QSAForwardMetadata, metadata[self.indexer.compressed_key_cache.prefix])
            if qsa_prefill_attention(
                query[:num_tokens],
                key[:num_tokens],
                value[:num_tokens],
                key_cache,
                value_cache,
                selected,
                main_metadata.block_table,
                token_to_req,
                compressed_metadata.logical_positions[:num_tokens],
                main_metadata.seq_lens,
                self.scaling,
                output[:num_tokens],
                gate[:num_tokens],
                prefill_plan,
                qsa_sparse_paged_attention,
                _qsa_get_split_workspace,
            ):
                return
            if all_visible_fia:
                raise RuntimeError("All-visible QSA FIA rejected an admitted request plan")
        if num_tokens:
            phases = common.is_prefilling
            if (
                self.indexer._use_native_decode
                and query.device.type == "npu"
                and context.cudagraph_runtime_mode == CUDAGraphMode.NONE
                and context.skip_compiled
                and common.causal is True
                and common.max_query_len == 1
                and phases is not None
                and phases.device.type == "cpu"
                and not phases.any()
            ):
                from .ops.qsa import _qsa_get_split_workspace

                split_workspace = _qsa_get_split_workspace(query[:num_tokens], 4)
            else:
                _, split_workspace = qsa_prepare_split_workspace(
                    query[:num_tokens],
                    key_cache,
                    self.topk_indices_buffer.shape[1],
                    self._qsa_split_workspaces,
                )
            qsa_sparse_paged_attention(
                query[:num_tokens],
                key_cache,
                value_cache,
                self.topk_indices_buffer[:num_tokens],
                main_metadata.block_table,
                token_to_req,
                self.scaling,
                output[:num_tokens],
                gate=gate[:num_tokens],
                split_workspace=split_workspace,
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        if gate is None:
            raise RuntimeError("Qwen3.8-Flash-Next QSA requires its output gate")
        num_tokens = hidden_states.shape[0]
        query = q.view(num_tokens, self.num_heads, self.head_dim)
        key = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        value = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        gate = gate.view_as(query)
        attn_output = torch.empty_like(query)
        encoded_layer_name = _encode_layer_name(self.layer_name)
        if current_platform.opaque_attention_op():
            torch.ops.vllm.qwen3_8_flash_next_qsa_with_output(
                hidden_states,
                positions,
                query,
                key,
                value,
                gate,
                attn_output,
                encoded_layer_name,
            )
        else:
            qwen3_8_flash_next_qsa_with_output(
                hidden_states,
                positions,
                query,
                key,
                value,
                gate,
                attn_output,
                encoded_layer_name,
            )
        flat_output = attn_output.view(num_tokens, -1)
        output, _ = self.o_proj(flat_output)
        return output


def qwen3_8_flash_next_qsa_with_output(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Run the complete QSA state/update/attend transaction."""

    layer_name = _resolve_layer_name(layer_name)
    layer = get_forward_context().no_compile_layers[layer_name]
    if not isinstance(layer, Qwen3_8FlashNextQSAAttention):
        raise TypeError(f"{layer_name} is not a Qwen3.8-Flash-Next QSA owner")
    layer._run_qsa(
        hidden_states,
        positions,
        query,
        key,
        value,
        gate,
        output,
    )


def qwen3_8_flash_next_qsa_with_output_fake(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    del hidden_states, positions, query, key, value, gate, output, layer_name


direct_register_custom_op(
    op_name="qwen3_8_flash_next_qsa_with_output",
    op_func=qwen3_8_flash_next_qsa_with_output,
    mutates_args=["output"],
    fake_impl=qwen3_8_flash_next_qsa_with_output_fake,
)


__all__ = [
    "QSAIndexer",
    "Qwen3_8FlashNextQSAAttention",
    "Qwen3_8FlashNextQSAAttentionBackend",
    "qwen3_8_flash_next_qsa_with_output",
]
