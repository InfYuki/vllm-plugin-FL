# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-vendor inference-only Qwen3.8-Flash-Next model."""

import inspect
from os import getenv
from collections.abc import Iterable
from functools import cache, partial
from itertools import islice
from unittest.mock import patch

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.platforms import current_platform
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)

try:
    from vllm.model_executor.layers.fused_moe import (
        fused_moe_make_expert_params_mapping,
    )
except ImportError:  # vLLM builds without the legacy fused-MoE helper
    fused_moe_make_expert_params_mapping = None
try:
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateCopyFuncsByType,
    )
except ImportError:  # vLLM 0.24 type-only compatibility
    MambaStateCopyFuncsByType = dict
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsPP,
    _require_is_multimodal,
)
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextMLP,
    Qwen3NextModel,
    Qwen3NextSparseMoeBlock,
    QwenNextMixtureOfExperts,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    StageMissingLayer,
    WeightsMapper,
    _merge_multimodal_embeddings,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
)

try:
    from vllm.model_executor.models.utils import maybe_fuse_shared_experts
except ImportError:  # vLLM 0.24 predates optional AITER shared-expert fusion
    def maybe_fuse_shared_experts(weights, **_kwargs):
        return weights
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.sequence import IntermediateTensors
from vllm.tokenizers.registry import cached_tokenizer_from_config
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec

from ..common.hyperconnection import (
    GatedResidualSimple,
    HyperConnectionConfig,
    pack_gated_hc_projection_weights,
)
from ..config import Qwen3_8FlashNextConfig, Qwen3_8FlashNextTextConfig
from .ple_layer import Qwen3_8FlashNextPLELayer, validate_ple_embedding_weights
from .qsa import Qwen3_8FlashNextQSAAttention
from .quantization import with_platform_quantization

_HAS_NATIVE_STACKED_WEIGHTS_MAPPER = hasattr(
    WeightsMapper(), "orig_to_new_stacked"
)


def _get_qwen35_hf_to_vllm_mapper() -> WeightsMapper:
    """Return the native packed-weight mapper when the vLLM API provides it.

    The Qwen3.8 reference was authored against the post-0.24 mapper API.  The
    FlagOS Day0 base image uses vLLM 0.24, where Qwen3Next/Qwen3.5 still packs
    weights in handwritten loaders and therefore exposes no class mapper.
    """

    mapper = getattr(Qwen3_5Model, "hf_to_vllm_mapper", None)
    if mapper is None:
        mapper = getattr(Qwen3NextModel, "hf_to_vllm_mapper", None)
    return mapper if mapper is not None else WeightsMapper()


_QWEN35_HF_TO_VLLM_MAPPER = _get_qwen35_hf_to_vllm_mapper()


_STACKED_WEIGHT_MAPPINGS = (
    (".q_proj", ".qkv_proj", "q"),
    (".k_proj", ".qkv_proj", "k"),
    (".v_proj", ".qkv_proj", "v"),
    (".mlp.gate_proj", ".mlp.gate_up_proj", 0),
    (".mlp.up_proj", ".mlp.gate_up_proj", 1),
    (".shared_expert.gate_proj", ".shared_expert.gate_up_proj", 0),
    (".shared_expert.up_proj", ".shared_expert.gate_up_proj", 1),
    (".in_proj_qkv", ".in_proj_qkvz", (0, 1, 2)),
    (".in_proj_z", ".in_proj_qkvz", 3),
    (".in_proj_b", ".in_proj_ba", 0),
    (".in_proj_a", ".in_proj_ba", 1),
)


def _map_stacked_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    """Attach shard metadata when the native mapper cannot stack weights."""

    for name, weight in weights:
        for old, new, shard_id in _STACKED_WEIGHT_MAPPINGS:
            if old in name:
                name = name.replace(old, new, 1)
                weight.shard_id = shard_id
                break
        yield name, weight


def _get_expert_mappings(
    model: nn.Module,
    num_experts: int,
    num_redundant_experts: int,
) -> tuple[
    list[tuple[str, str, int, str]],
    list[tuple[str, str, int, str]],
] | None:
    """Probe for the fused-MoE mapping API.

    vLLM 0.24 exposes ``fused_moe_make_expert_params_mapping`` but does not
    teach ``AutoWeightsLoader`` how to turn a Hugging Face fused 3-D
    ``experts.{gate_up_proj,down_proj}`` tensor into the
    ``experts.routed_experts.{w13,w2}_weight`` parameters.  Newer vLLM builds
    carry that knowledge through the native mapper and never enter this
    compatibility path.  Keeping the probe structural avoids coupling the
    plugin to a version string.

    The first mapping handles ordinary one-expert-at-a-time checkpoints.  The
    second mapping deliberately uses ``gate_up_proj`` for both the gate and up
    names and is reduced to generic names for fused 3-D checkpoints, matching
    the loader in vLLM's Qwen3.5 implementation.
    """

    helper = fused_moe_make_expert_params_mapping
    if not callable(helper):
        return None

    try:
        regular_mapping = list(
            helper(
                model,
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=num_experts,
                num_redundant_experts=num_redundant_experts,
            )
        )
        fused_base_mapping = list(
            helper(
                model,
                ckpt_gate_proj_name="gate_up_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="gate_up_proj",
                num_experts=1,
            )
        )
    except (AttributeError, TypeError):
        # A future helper with a different ABI must stay on the native path
        # instead of being partially handled by this legacy adapter.
        return None

    fused_mapping: list[tuple[str, str, int, str]] = []
    for param_name, checkpoint_name, _, shard_id in fused_base_mapping:
        parts = checkpoint_name.split(".")
        if len(parts) < 3:
            return None
        fused_mapping.append(
            (
                f"{param_name}weight",
                f"{parts[0]}.{parts[2]}",
                0,
                shard_id,
            )
        )
    if not regular_mapping or not fused_mapping:
        return None
    return regular_mapping, fused_mapping


def _call_expert_weight_loader(
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    param_name: str,
    shard_id: str,
    expert_id: int,
) -> bool:
    """Call the legacy FusedMoE weight loader with capability probing."""

    weight_loader = getattr(param, "weight_loader", None)
    if weight_loader is None:
        raise TypeError(f"Fused expert parameter {param_name!r} has no weight_loader")
    try:
        loaded = weight_loader(
            param,
            loaded_weight,
            param_name,
            shard_id=shard_id,
            expert_id=expert_id,
            return_success=True,
        )
    except TypeError as exc:
        # Keep support for the older callable form without weakening errors
        # from inside a real loader implementation.
        if "return_success" not in str(exc):
            raise
        weight_loader(
            param,
            loaded_weight,
            param_name,
            shard_id=shard_id,
            expert_id=expert_id,
        )
        return True
    return True if loaded is None else bool(loaded)


def _load_fused_expert_weight(
    name: str,
    loaded_weight: torch.Tensor,
    params_dict: dict[str, nn.Parameter],
    fused_mapping: list[tuple[str, str, int, str]],
    num_experts: int,
) -> tuple[bool, set[str]]:
    """Load a fused HF expert tensor through the available expert loader."""

    if "mlp.experts.gate_up_proj" not in name and "mlp.experts.down_proj" not in name:
        return False, set()
    if loaded_weight.ndim != 3:
        raise ValueError(
            f"Expected a fused 3-D MoE tensor for {name!r}, got "
            f"shape={tuple(loaded_weight.shape)}"
        )

    loaded_params: set[str] = set()
    for param_name, weight_name, _, shard_id in fused_mapping:
        if weight_name not in name:
            continue
        name_mapped = name.replace(weight_name, param_name, 1)
        param = params_dict.get(name_mapped)
        # A pipeline-parallel rank can legitimately not own this layer.
        if param is None:
            return True, loaded_params

        if "gate_up_proj" in name:
            split_weights = loaded_weight.chunk(2, dim=-2)
            shard_weights = (
                ("w1", split_weights[0]),
                ("w3", split_weights[1]),
            )
        else:
            shard_weights = ((shard_id, loaded_weight),)

        loaded_local_expert = False
        for actual_shard_id, shard_weight in shard_weights:
            for expert_id in range(num_experts):
                if _call_expert_weight_loader(
                    param,
                    shard_weight[expert_id],
                    name_mapped,
                    actual_shard_id,
                    expert_id,
                ):
                    loaded_local_expert = True
        if loaded_local_expert:
            loaded_params.add(name_mapped)
        return True, loaded_params
    return True, loaded_params


def _load_single_expert_weight(
    name: str,
    loaded_weight: torch.Tensor,
    params_dict: dict[str, nn.Parameter],
    expert_mapping: list[tuple[str, str, int, str]],
) -> tuple[bool, set[str]]:
    """Load a non-fused expert tensor when the old ABI supplies one expert."""

    loaded_params: set[str] = set()
    for param_name, weight_name, expert_id, shard_id in expert_mapping:
        if weight_name not in name:
            continue
        name_mapped = name.replace(weight_name, param_name, 1)
        param = params_dict.get(name_mapped)
        if param is None:
            return True, loaded_params
        if _call_expert_weight_loader(
            param,
            loaded_weight,
            name_mapped,
            shard_id,
            expert_id,
        ):
            loaded_params.add(name_mapped)
        return True, loaded_params
    return False, loaded_params


class _StackedAutoWeightsLoader(AutoWeightsLoader):
    """Teach the vLLM 0.24 auto-loader to forward packed shard identifiers."""

    def _load_param(
        self,
        base_prefix: str,
        param: nn.Parameter,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[str]:
        for weight_name, weight_data in weights:
            shard_id = getattr(weight_data, "shard_id", None)
            if shard_id is None:
                yield from super()._load_param(
                    base_prefix, param, [(weight_name, weight_data)]
                )
                continue

            weight_qualname = self._get_qualname(base_prefix, weight_name)
            if self._can_skip(weight_qualname):
                continue
            if weight_name != "":
                if self._can_ignore_unexpected(weight_qualname):
                    continue
                raise ValueError(
                    f"Attempted to load nested weight {weight_qualname!r} "
                    f"into a single parameter {base_prefix!r}"
                )
            weight_loader = getattr(param, "weight_loader", None)
            if weight_loader is None:
                raise TypeError(
                    f"Packed parameter {weight_qualname!r} has no weight_loader"
                )
            weight_loader(param, weight_data, shard_id)
            yield weight_qualname


def _remap_qsa_cache_scale_name(
    name: str,
    qsa_layer_ids: frozenset[int],
) -> str:
    """Map serialized main-cache scales onto the merged QSA owner.

    Regular attention keeps cache scales below its ``attn`` child. QSA owns
    that cache directly, so only QSA layers need the final path component
    moved to the owner's persistent ``_k_scale``/``_v_scale`` buffers.
    """

    scale_suffixes = {
        "k_proj.k_scale": "_k_scale",
        "k_proj.output_scale": "_k_scale",
        "attn.k_scale": "_k_scale",
        "attn._k_scale": "_k_scale",
        "k_scale": "_k_scale",
        "_k_scale": "_k_scale",
        "v_proj.v_scale": "_v_scale",
        "v_proj.output_scale": "_v_scale",
        "attn.v_scale": "_v_scale",
        "attn._v_scale": "_v_scale",
        "v_scale": "_v_scale",
        "_v_scale": "_v_scale",
    }
    for layer_id in qsa_layer_ids:
        marker = f"layers.{layer_id}.self_attn."
        marker_start = name.find(marker)
        if marker_start < 0 or (marker_start > 0 and name[marker_start - 1] != "."):
            continue
        suffix = name[marker_start + len(marker) :]
        mapped_suffix = scale_suffixes.get(suffix)
        if mapped_suffix is not None:
            return f"{name[: marker_start + len(marker)]}{mapped_suffix}"
    return name


_QWEN38_FLASH_NEXT_IGNORED_MISSING_SUFFIXES = [
    ".bias",
    "_bias",
    ".k_scale",
    "_k_scale",
    ".v_scale",
    "_v_scale",
    "_weight_scale",
    "_input_scale",
]


logger = init_logger(__name__)


def _validate_hc_moe_parallelism(vllm_config: VllmConfig) -> None:
    parallel = vllm_config.parallel_config
    if parallel.use_sequence_parallel_moe and (
        current_platform.device_type != "npu"
        or current_platform.dist_backend != "hccl"
        or parallel.all2all_backend != "allgather_reducescatter"
    ):
        raise NotImplementedError(
            "Qwen3.8-Flash-Next HC sequence-parallel MoE requires Ascend HCCL AllGather"
        )


@cache
def _shared_expert_npu_stream(device: int):
    return torch.npu.Stream(device=device)


class _AscendSharedExperts(SharedExperts):
    def __init__(self, *args, graph_overlap=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._graph_overlap = graph_overlap
        if graph_overlap:
            # Create the worker's auxiliary stream before ACL graph capture.
            _shared_expert_npu_stream(torch.npu.current_device())

    def _determine_shared_experts_order(self, hidden_states):
        mode = get_forward_context().cudagraph_runtime_mode
        if mode == CUDAGraphMode.NONE or (
            self._graph_overlap
            and mode == CUDAGraphMode.FULL
            and torch.npu.is_current_stream_capturing()
        ):
            return SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
        return SharedExpertsOrder.NO_OVERLAP

    def maybe_forward_async(self, shared_experts_input):
        if self._determine_shared_experts_order(shared_experts_input) != (
            SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
        ):
            return False
        self._npu_stream = _shared_expert_npu_stream(shared_experts_input.device.index)
        idx = self._output_idx
        if self._output[idx] is not None:
            raise RuntimeError("Shared expert output has not been consumed")
        shared_experts_input.record_stream(self._npu_stream)
        main_stream = torch.npu.current_stream()
        if get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL:
            logger.info_once("Qwen3.8 Ascend shared expert ACL graph overlap active")
            # Make the auxiliary stream a dependent stream of this capture
            # before submitting any shared-expert kernels to it.
            self._npu_stream.wait_event(main_stream.record_event())
        else:
            self._npu_stream.wait_stream(main_stream)
        with torch.npu.stream(self._npu_stream):
            self._output[idx] = self._layer(shared_experts_input)
        return True

    def wait(self):
        main_stream = torch.npu.current_stream()
        # Rejoin inside moe_forward_shared, before the runner returns either
        # output or submits a collective consuming the combined result.
        main_stream.wait_stream(self._npu_stream)
        output = self._output[self._output_idx]
        if output is None:
            raise RuntimeError("Shared expert work was not enqueued")
        output.record_stream(main_stream)


class Qwen3_8FlashNextSparseMoeBlock(Qwen3NextSparseMoeBlock):
    """Qwen3Next MoE with Qwen3.8-Flash-Next HC validation."""

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        _validate_hc_moe_parallelism(vllm_config)
        parallel = vllm_config.parallel_config
        options = vllm_config.additional_config or {}
        use_alltoallv = options.get("enable_qwen38_prefill_alltoallv", False)
        use_mc2 = options.get("enable_qwen38_mc2", False)
        if (use_alltoallv or use_mc2) and not (
            current_platform.device_type == "npu"
            and parallel.tensor_parallel_size == 8
            and parallel.data_parallel_size == 2
            and parallel.enable_expert_parallel
            and parallel.use_sequence_parallel_moe
            and parallel.expert_placement_strategy == "linear"
            and not parallel.enable_eplb
            and not parallel.use_ubatching
            and not parallel.enable_elastic_ep
            and vllm_config.quant_config is None
            and vllm_config.model_config.dtype == torch.bfloat16
        ):
            raise ValueError("Qwen3.8 Ascend expert exchange requires BF16 TP8×DP2/EP16")
        if use_mc2 and (
            vllm_config.speculative_config is not None
            or options.get("enable_fused_mc2", 0)
            or parallel.data_parallel_size_local != 2
            or parallel.pipeline_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.all2all_backend != "allgather_reducescatter"
        ):
            raise ValueError("Qwen3.8 BF16 MC2 requires local EP16 without MTP or fused MC2")
        if use_alltoallv or use_mc2:
            from vllm.model_executor.models import qwen3_next
            from vllm_fl.ops.fused_moe.ascend_alltoallv import (
                AscendPrefillAllToAllVMoERunner,
            )

            # The parent resolves FusedMoE in its own module. OOT routing can
            # leave this binding on the upstream factory instead of FusedMoEFL.
            factory = partial(
                qwen3_next.FusedMoEFactory,
                runner_cls=partial(
                    AscendPrefillAllToAllVMoERunner,
                    use_alltoallv=use_alltoallv,
                    use_mc2=use_mc2,
                ),
            )
            with patch.object(qwen3_next, "FusedMoEFactory", factory):
                super().__init__(vllm_config=vllm_config, prefix=prefix)
            if not isinstance(self.experts, AscendPrefillAllToAllVMoERunner):
                raise RuntimeError("Qwen3.8 did not construct the Ascend expert exchange runner")
        else:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        # The current FusedMoEFactory owns its final tensor-parallel
        # reduction. Do not reduce the result a second time in the HC caller.
        self.requires_tp_all_reduce = False

        shared_dp = options.get("enable_shared_expert_dp", False)
        overlap = options.get("multistream_overlap_shared_expert", False)
        graph_overlap = options.get("enable_qwen38_graph_shared_expert_overlap", False)
        if graph_overlap and not overlap:
            raise ValueError("Qwen3.8 graph overlap requires multistream_overlap_shared_expert")
        if shared_dp or overlap:
            if (
                current_platform.device_type != "npu"
                or not self.is_sequence_parallel
                or self.shared_expert is None
            ):
                raise ValueError(
                    "Qwen3.8 Ascend shared expert DP requires MoE sequence "
                    "parallelism and a shared expert"
                )
            logger.info_once("Qwen3.8 Ascend shared expert token sharding is active")
        if overlap:
            if not shared_dp:
                raise ValueError("Shared expert overlap requires enable_shared_expert_dp")
            runner = self.experts
            shared = runner.shared_experts
            if shared is None or shared.enable_dbo:
                raise ValueError("Ascend shared expert overlap requires no DBO")
            if graph_overlap and (
                vllm_config.quant_config is not None
                or vllm_config.model_config.dtype != torch.bfloat16
                or self.shared_expert.gate_up_proj.tp_size != 1
                or self.shared_expert.down_proj.tp_size != 1
                or self.shared_expert.down_proj.reduce_results
            ):
                raise ValueError("Qwen3.8 graph overlap requires a local BF16 shared MLP")
            runner._shared_experts = _AscendSharedExperts(
                shared._layer, shared._moe_config, shared.enable_dbo,
                shared._mk_can_overlap_shared_experts,
                graph_overlap=graph_overlap,
            )


class Qwen3_8FlashNextDecoderLayer(Qwen3NextDecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
        force_disable_ple: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        config: Qwen3_8FlashNextTextConfig = vllm_config.model_config.hf_text_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)
        _validate_hc_moe_parallelism(vllm_config)
        self.ple: Qwen3_8FlashNextPLELayer | None = None
        ple_layer_ids = config.ple_layer_ids
        if (self.layer_idx + 1) in ple_layer_ids and not force_disable_ple:
            ple_layer_ids_sorted = sorted(set(ple_layer_ids))
            ple_dense_layer_id_map = {
                abs_id: idx for idx, abs_id in enumerate(ple_layer_ids_sorted)
            }
            ple_dense_layer_id = ple_dense_layer_id_map[self.layer_idx + 1]
            self.ple = Qwen3_8FlashNextPLELayer(
                config,
                vllm_config=vllm_config,
                layer_idx=self.layer_idx,
                ple_dense_layer_id=ple_dense_layer_id,
                prefix=f"{prefix}.ple",
            )

        if layer_type == "linear_attention":
            gdn_kwargs = {
                "vllm_config": vllm_config,
                "prefix": f"{prefix}.linear_attn",
                "gqa_interleaved_layout": False,
            }
            if "reduce_results" in inspect.signature(
                QwenGatedDeltaNetAttention.__init__
            ).parameters:
                gdn_kwargs["reduce_results"] = False
            self.linear_attn = QwenGatedDeltaNetAttention(config, **gdn_kwargs)
            # vLLM 0.24 predates the constructor option. HC owns the single TP
            # reduction after each branch, so disable the GDN output linear's
            # internal reduction explicitly to avoid a double all-reduce.
            if "reduce_results" not in inspect.signature(
                QwenGatedDeltaNetAttention.__init__
            ).parameters:
                self.linear_attn.out_proj.reduce_results = False
            self._gdn_requires_output_buffer = "output" in inspect.signature(
                QwenGatedDeltaNetAttention.forward
            ).parameters
        elif layer_type == "full_attention":
            use_qsa = getattr(config, "indexer_n_heads", None) is not None
            if not use_qsa:
                self.self_attn = Qwen3NextAttention(
                    config,
                    model_config=model_config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    reduce_results=False,
                    prefix=f"{prefix}.self_attn",
                )
            else:
                self.self_attn = Qwen3_8FlashNextQSAAttention(
                    vllm_config=vllm_config,
                    config=config,
                    layer_id=self.layer_idx,
                    quant_config=quant_config,
                    reduce_results=False,
                    prefix=f"{prefix}.self_attn",
                )
        else:
            raise ValueError(f"Invalid layer_type {layer_type}")

        mlp_only_layers = getattr(config, "mlp_only_layers", [])
        num_experts = getattr(config, "num_experts", 0) or 0
        absolute_layer_id = self.layer_idx + 1
        is_moe_layer = self.layer_idx not in mlp_only_layers and (
            num_experts > 0 and absolute_layer_id % config.decoder_sparse_step == 0
        )
        if is_moe_layer:
            self.mlp = Qwen3_8FlashNextSparseMoeBlock(
                vllm_config=vllm_config, prefix=f"{prefix}.mlp"
            )
        else:
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.mlp",
            )

        hc_config = HyperConnectionConfig(
            hc_count=config.hc_count,
            hidden_size=config.hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
            hc_per_branch_norm=True,
        )
        self.attn_hyper_connection = GatedResidualSimple(
            hc_config,
            layer_idx=self.layer_idx,
            role="attn",
        )
        self.mlp_hyper_connection = GatedResidualSimple(
            hc_config,
            layer_idx=self.layer_idx,
            role="mlp",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor | None = None,
        prev_injection: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        ngram_context: torch.Tensor | None = None,
        sequence_parallel_decode: bool = False,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # Keep a clear failure mode for old callers that still try to pass a
        # separate residual tensor.  HC's pending state is a block output plus
        # injection logits, not the ordinary vLLM residual ABI.
        legacy_residual = kwargs.pop("residual", None)
        if legacy_residual is not None:
            raise ValueError("HC layers use prev_block_output/prev_injection")
        del kwargs
        if self.ple is not None:
            # PLE adds to the materialized multi-stream state.  It therefore
            # terminates a pending delayed combine before its external add.
            if prev_block_output is not None and prev_injection is not None:
                hidden_states = self.attn_hyper_connection.combine_pending(
                    hidden_states,
                    prev_block_output,
                    prev_injection,
                )
                prev_block_output = prev_injection = None
            if input_ids is None:
                raise ValueError("PLE requires input_ids")
            if query_start_loc is None:
                raise ValueError("ngram PLE requires query_start_loc")
            if ngram_context is None:
                raise ValueError("ngram PLE requires ngram_context")
            if sequence_parallel_decode:
                hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=0)
            hidden_states = hidden_states + self.ple(
                hidden_states,
                input_ids,
                query_start_loc,
                ngram_context,
            )

        hidden_states, mixed, injection = self.attn_hyper_connection.combine_and_mix(
            hidden_states,
            prev_block_output,
            prev_injection,
        )
        if sequence_parallel_decode:
            if self.ple is None:
                mixed = tensor_model_parallel_all_gather(mixed, dim=0)
            else:
                hidden_states = sequence_parallel_chunk(hidden_states)
                injection = sequence_parallel_chunk(injection)
        if self.layer_type == "linear_attention":
            if self._gdn_requires_output_buffer:
                # vLLM 0.24's pluggable GDN ABI writes into a caller-provided
                # buffer; newer vLLM returns the projected tensor directly.
                self_attention_output = torch.empty_like(mixed)
                self.linear_attn(mixed, self_attention_output)
            else:
                self_attention_output = self.linear_attn(hidden_states=mixed)
        elif self.layer_type == "full_attention":
            self_attention_output = self.self_attn(
                hidden_states=mixed,
                positions=positions,
            )
        else:
            raise ValueError("Invalid layer_type")
        if sequence_parallel_decode:
            self_attention_output = tensor_model_parallel_reduce_scatter(
                self_attention_output, dim=0
            )
        elif get_tensor_model_parallel_world_size() > 1:
            self_attention_output = tensor_model_parallel_all_reduce(
                self_attention_output
            )

        # Consume the attention injection at the next HC boundary and leave
        # the MLP output itself pending for the following layer/final mixer.
        hidden_states, mixed, next_injection = (
            self.mlp_hyper_connection.combine_and_mix(
                hidden_states,
                self_attention_output,
                injection,
            )
        )
        if sequence_parallel_decode and isinstance(self.mlp, Qwen3_8FlashNextSparseMoeBlock):
            block_output = self.mlp(mixed, already_sequence_parallel=True)
        else:
            block_output = self.mlp(mixed)
        if get_tensor_model_parallel_world_size() > 1 and getattr(
            self.mlp, "requires_tp_all_reduce", True
        ):
            block_output = tensor_model_parallel_all_reduce(block_output)
        return hidden_states, block_output, next_injection


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "query_start_loc": 0,
        "ngram_context": 0,
        "deepstack_input_embeds": 0,
    }
)
class Qwen3_8FlashNextModel(nn.Module):
    hf_to_vllm_mapper = _QWEN35_HF_TO_VLLM_MAPPER

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        vllm_config = with_platform_quantization(vllm_config)
        super().__init__()
        config: Qwen3_8FlashNextTextConfig = vllm_config.model_config.hf_text_config
        self.config = config
        parallel = vllm_config.parallel_config
        self._sequence_parallel_decode = (
            getenv("VLLM_FL_QWEN38_NATIVE_DECODE") == "1"
            and getenv("VLLM_FL_QWEN38_SP_DECODE", "1") == "1"
            and current_platform.device_type == "npu"
            and vllm_config.quant_config is None
            and vllm_config.speculative_config is None
            and vllm_config.lora_config is None
            and parallel.tensor_parallel_size == 8
            and parallel.data_parallel_size == 2
            and parallel.pipeline_parallel_size == 1
            and parallel.use_sequence_parallel_moe
            and not parallel.enable_eplb
            and not parallel.use_ubatching
            and getattr(config, "decoder_sparse_step", 1) == 1
            and not getattr(config, "mlp_only_layers", ())
            and bool((vllm_config.additional_config or {}).get("enable_qwen38_mc2"))
        )
        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )
        self.vocab_size = config.vocab_size
        self._qsa_layer_ids = frozenset(
            layer_idx
            for layer_idx, layer_type in enumerate(config.layer_types)
            if layer_type == "full_attention"
            and getattr(config, "indexer_n_heads", None) is not None
        )
        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, config.hidden_size)

        def get_layer(prefix: str) -> Qwen3_8FlashNextDecoderLayer:
            layer_idx = extract_layer_index(prefix)
            return Qwen3_8FlashNextDecoderLayer(
                vllm_config,
                layer_type=config.layer_types[layer_idx],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        intermediate_size = config.hidden_size * config.hc_count
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], intermediate_size
        )

        self.hyper_connection_mixer: GatedResidualSimple | None
        if get_pp_group().is_last_rank:
            hc_config = HyperConnectionConfig(
                hc_count=config.hc_count,
                hidden_size=config.hidden_size,
                params_dtype=torch.bfloat16,
                hc_lowrank=config.hc_lowrank,
                rms_norm_eps=config.rms_norm_eps,
                hc_per_branch_norm=True,
            )
            self.hyper_connection_mixer = GatedResidualSimple(
                hc_config, use_combine=False, role="final"
            )
        else:
            self.hyper_connection_mixer = None

        spec_config = vllm_config.speculative_config
        # MTP HC multi-stream outputs: when speculative method=="mtp" and the
        # model uses HC with hc_count>1, retain the pre-final-mixer multi-stream
        # residual [T, hc_count*H] so the MTP drafter can feed a real
        # multi-stream backbone hidden on its first step (scheme A). Derived
        # purely from config (NOT node identity) so P/D nodes stay consistent.
        needs_mtp_hidden = (
            spec_config is not None
            and getattr(spec_config, "method", None) == "mtp"
            and get_pp_group().is_last_rank
        )
        if needs_mtp_hidden:
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.hc_count * config.hidden_size,
                dtype=vllm_config.model_config.dtype,
            )
        else:
            self._mtp_hidden_buffer = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        ngram_context: torch.Tensor | None = None,
        deepstack_input_embeds: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError("input_ids or inputs_embeds is required")
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = hidden_states.repeat(1, self.config.hc_count)
        else:
            if intermediate_tensors is None:
                raise ValueError("pipeline stage requires intermediate tensors")
            hidden_states = intermediate_tensors["hidden_states"]

        sequence_parallel_decode = (
            self._sequence_parallel_decode
            and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL
            and positions.shape[-1] == 8
        )
        if sequence_parallel_decode:
            hidden_states = sequence_parallel_chunk(hidden_states)
            logger.info_once("Qwen3.8 FULL decode keeps token shards across HC and MoE layers")

        block_output: torch.Tensor | None = None
        injection: torch.Tensor | None = None
        last_layer = None
        for layer_idx, layer in islice(
            enumerate(self.layers), self.start_layer, self.end_layer
        ):
            last_layer = layer
            hidden_states, block_output, injection = layer(
                hidden_states=hidden_states,
                prev_block_output=block_output,
                prev_injection=injection,
                positions=positions,
                input_ids=input_ids,
                query_start_loc=query_start_loc,
                ngram_context=ngram_context,
                sequence_parallel_decode=sequence_parallel_decode,
            )
            if deepstack_input_embeds is not None and layer_idx < len(
                deepstack_input_embeds
            ):
                deepstack_embed = deepstack_input_embeds[
                    f"deepstack_input_embeds_{layer_idx}"
                ]
                deepstack_embed = (
                    deepstack_embed.unsqueeze(-2)
                    .expand(
                        *deepstack_embed.shape[:-1],
                        self.config.hc_count,
                        self.config.hidden_size,
                    )
                    .flatten(-2)
                )
                if block_output is not None and injection is not None:
                    # Deepstack is an external add and must see the fully
                    # materialized state, so it explicitly ends the delayed
                    # combine chain for this layer.
                    hidden_states = layer.mlp_hyper_connection.combine_pending(
                        hidden_states,
                        block_output,
                        injection,
                    )
                    block_output = injection = None
                if sequence_parallel_decode:
                    deepstack_embed = sequence_parallel_chunk(deepstack_embed)
                hidden_states = hidden_states + deepstack_embed

        if not get_pp_group().is_last_rank:
            # PP transports one tensor.  Materialize the pending MLP block at
            # the stage boundary instead of silently dropping its injection.
            if last_layer is not None and block_output is not None:
                assert injection is not None
                hidden_states = last_layer.mlp_hyper_connection.combine_pending(
                    hidden_states,
                    block_output,
                    injection,
                )
            return IntermediateTensors({"hidden_states": hidden_states})

        assert self.hyper_connection_mixer is not None
        multi_hidden, sample_hidden_states, _ = (
            self.hyper_connection_mixer.combine_and_mix(
                hidden_states,
                block_output,
                injection,
            )
        )
        if self._mtp_hidden_buffer is not None:
            # Capture the pre-final-mixer multi-stream state
            # [T, hc_count*H] for the MTP drafter (zero extra compute:
            # this tensor is needed by the final mixer regardless).
            num_tokens = multi_hidden.shape[0]
            self._mtp_hidden_buffer[:num_tokens].copy_(multi_hidden)
        if sequence_parallel_decode:
            sample_hidden_states = tensor_model_parallel_all_gather(sample_hidden_states, dim=0)
        return sample_hidden_states

    @validate_ple_embedding_weights
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = (
            (
                _remap_qsa_cache_scale_name(name, self._qsa_layer_ids),
                weight,
            )
            for name, weight in weights
        )
        weights = maybe_fuse_shared_experts(
            weights,
            n_routed_experts=getattr(self.config, "num_experts", 0) or 0,
            n_shared_experts=1,
            ckpt_prefix="mlp.shared_expert",
        )
        # Non-persistent PLE state rebuilt in __init__; skip any ckpt
        # column for them.
        skip_substrs = [
            "hashstats_",
            "token_lookup",
            "hyper_connection_mixer.block_inject_weight",
        ]
        mapper = self.hf_to_vllm_mapper
        loader_cls = AutoWeightsLoader
        legacy_loader_kwargs = {}
        legacy_loaded: set[str] = set()
        if not _HAS_NATIVE_STACKED_WEIGHTS_MAPPER:
            expert_mappings = _get_expert_mappings(
                self,
                num_experts=getattr(self.config, "num_experts", 0) or 0,
                num_redundant_experts=self.num_redundant_experts,
            )
            if expert_mappings is not None:
                regular_expert_mapping, fused_expert_mapping = expert_mappings
                params_dict = dict(self.named_parameters())
                legacy_input = weights

                def _legacy_weights() -> Iterable[tuple[str, torch.Tensor]]:
                    for name, loaded_weight in legacy_input:
                        handled, loaded = _load_fused_expert_weight(
                            name,
                            loaded_weight,
                            params_dict,
                            fused_expert_mapping,
                            getattr(self.config, "num_experts", 0) or 0,
                        )
                        if handled:
                            legacy_loaded.update(loaded)
                            continue
                        handled, loaded = _load_single_expert_weight(
                            name,
                            loaded_weight,
                            params_dict,
                            regular_expert_mapping,
                        )
                        if handled:
                            legacy_loaded.update(loaded)
                            continue
                        yield from _map_stacked_weights(
                            ((name, loaded_weight),)
                        )

                weights = _legacy_weights()
            else:
                weights = _map_stacked_weights(weights)
            mapper = None
            loader_cls = _StackedAutoWeightsLoader
            legacy_loader_kwargs["skip_substrs"] = skip_substrs
        else:
            mapper = mapper | WeightsMapper(
                orig_to_new_substr={name: None for name in skip_substrs},
            )
        loader = loader_cls(
            self,
            ignore_unexpected_suffixes=_QWEN38_FLASH_NEXT_IGNORED_MISSING_SUFFIXES.copy(),
            **legacy_loader_kwargs,
        )
        loaded = loader.load_weights(
            weights,
            mapper=mapper,
        )
        loaded.update(legacy_loaded)
        pack_gated_hc_projection_weights(self)
        return loaded


class Qwen3_8FlashNextForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsPP,
    QwenNextMixtureOfExperts,
    IsHybrid,
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"model.language_model.": "model."}
    )
    requires_raw_input_tokens = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        vllm_config = with_platform_quantization(vllm_config)
        super().__init__()
        config: Qwen3_8FlashNextTextConfig = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.quant_config = vllm_config.quant_config
        self.config = config
        self.scheduler_config = vllm_config.scheduler_config
        if vllm_config.cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3.8-Flash-Next currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )
        self.model = Qwen3_8FlashNextModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        # Set MoE hyperparameters
        if (getattr(config, "num_experts", 0) or 0) > 0:
            QwenNextMixtureOfExperts.set_moe_parameters(self)

    @staticmethod
    def get_legacy_model_state_cls():
        from .legacy_model_state import Qwen3_8FlashNextLegacyModelState

        return Qwen3_8FlashNextLegacyModelState

    @staticmethod
    def get_model_state_cls():
        from .model_state import Qwen3_8FlashNextModelState

        return Qwen3_8FlashNextModelState

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        # Preserve model-state inputs (query_start_loc / ngram_context).
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **kwargs,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    @classmethod
    def get_ple_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_ple_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int]]:
        hf_config = vllm_config.model_config.hf_text_config
        conv_kernel_size = hf_config.ple_conv_kernel_size
        short_conv_dilation = hf_config.ngram_size
        conv_state_len = (conv_kernel_size - 1) * short_conv_dilation
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        hc_count = hf_config.hc_count
        hc_hidden_size = hf_config.hidden_size * hc_count
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=1,
            intermediate_size=hc_hidden_size,
            conv_kernel=conv_state_len + num_spec + 1,
        )

    @classmethod
    def get_gdn_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_gdn_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return cls.get_gdn_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return cls.get_gdn_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    @classmethod
    def get_mamba_state_copy_funcs(
        cls,
        mamba_types: set[MambaAttentionBackendEnum],
    ) -> MambaStateCopyFuncsByType:
        copy_funcs_by_type = {
            MambaAttentionBackendEnum.GDN_ATTN: cls.get_mamba_state_copy_func(),
            MambaAttentionBackendEnum.SHORT_CONV: (
                MambaStateCopyFuncCalculator.short_conv_state_copy_func()
            ),
        }
        missing_types = mamba_types - copy_funcs_by_type.keys()
        assert not missing_types, f"missing state copy funcs for {missing_types}"
        return {
            mamba_type: copy_funcs_by_type[mamba_type] for mamba_type in mamba_types
        }

    @classmethod
    def get_mamba_specs_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[MambaSpec, ...]:
        """Return all MambaSpecs for this model (GDN layers + PLE layer).

        The PLE layer uses a separate short_conv MambaSpec whose page_size_bytes
        may exceed the GDN spec; callers should take the maximum.
        """
        return (
            MambaSpec(
                shapes=cls.get_gdn_mamba_state_shape_from_config(vllm_config),
                dtypes=cls.get_gdn_mamba_state_dtype_from_config(vllm_config),
                block_size=-1,
            ),
            MambaSpec(
                shapes=cls.get_ple_mamba_state_shape_from_config(vllm_config),
                dtypes=cls.get_ple_mamba_state_dtype_from_config(vllm_config),
                block_size=-1,
            ),
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        return self.model._mtp_hidden_buffer

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        del mm_features
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.unsqueeze(0).expand(3, -1), 0

    @validate_ple_embedding_weights
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mapper = self.hf_to_vllm_mapper | WeightsMapper(
            orig_to_new_substr={"mtp.": None},
        )
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=_QWEN38_FLASH_NEXT_IGNORED_MISSING_SUFFIXES.copy(),
        )
        loaded = loader.load_weights(weights, mapper=mapper)
        pack_gated_hc_projection_weights(self)
        return loaded


class Qwen3_8FlashNextMixtureOfExperts(MixtureOfExperts):
    """Expose Qwen3.8-Flash-Next routed experts through vLLM's EPLB protocol."""

    language_model: Qwen3_8FlashNextForCausalLM

    def _set_moe_parameters(self) -> None:
        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        language_model = getattr(self, "model", None)
        if language_model is None:
            language_model = self.language_model.model
        for layer in language_model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)

        self.num_moe_layers = len(self.moe_layers)
        if example_moe is None:
            self.num_expert_groups = 1
            self.num_shared_experts = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_redundant_experts = 0
            return

        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.moe_mlp_layers:
            moe.n_physical_experts = num_physical_experts
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()


class Qwen3_8FlashNextProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self) -> Qwen3_8FlashNextConfig:
        return self.ctx.get_hf_config(Qwen3_8FlashNextConfig)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_8FlashNextProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_8FlashNextForConditionalGeneration(
    Qwen3_5ForConditionalGeneration,
    HasInnerState,
    Qwen3_8FlashNextMixtureOfExperts,
):
    """Qwen3-VL vision tower backed by the Qwen3.8-Flash-Next language model."""

    requires_raw_input_tokens = True
    # Do not inherit the multimodal-pruning capability from the Qwen3.5/Qwen3VL
    # base: some vLLM 0.24 builds default it to True, but this Day0 model never
    # initializes EVS pruning and ``recompute_mrope_positions`` raises. Declare
    # it here so the runner cannot select a pruning path the model cannot run.
    supports_multimodal_pruning = False

    packed_modules_mapping = Qwen3_5ForConditionalGeneration.packed_modules_mapping

    @staticmethod
    def get_legacy_model_state_cls():
        from .legacy_model_state import Qwen3_8FlashNextLegacyModelState

        return Qwen3_8FlashNextLegacyModelState

    @staticmethod
    def get_model_state_cls():
        from .model_state import Qwen3_8FlashNextModelState

        return Qwen3_8FlashNextModelState

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None:
        vllm_config = with_platform_quantization(vllm_config)
        nn.Module.__init__(self)
        config: Qwen3_8FlashNextConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        if multimodal_config is None:
            raise ValueError(
                "Qwen3_8FlashNextForConditionalGeneration requires multimodal_config"
            )

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.language_model_only = multimodal_config.language_model_only
        if self.language_model_only:
            self.use_data_parallel = False
            self.is_multimodal_pruning_enabled = False
            self.video_pruning_method = None
            self.video_pruning_rate = 0.0
            self._tokenizer = None
            self.visual = StageMissingLayer("vision_tower")
            self._tower_model_names = []
        else:
            self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
            # vLLM 0.24's Qwen3.5 multimodal base does not provide the newer
            # _init_video_pruning() helper and does not support EVS pruning.
            # Preserve the existing day0 compatibility behavior.
            self.is_multimodal_pruning_enabled = False
            self.video_pruning_method = None
            self.video_pruning_rate = 0.0
            self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)

            with self._mark_tower_model(vllm_config, {"image", "video"}):
                self.visual = Qwen3_VisionTransformer(
                    config.vision_config,
                    norm_eps=config.text_config.rms_norm_eps,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "visual"),
                )

        self.use_deepstack = (
            not self.language_model_only
            and bool(config.vision_config.deepstack_visual_indexes)
            and not isinstance(self.visual, StageMissingLayer)
        )
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        if self.use_deepstack:
            self.deepstack_input_embeds = [
                torch.zeros(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    config.text_config.hidden_size,
                )
                for _ in range(self.deepstack_num_level)
            ]
            self.deepstack_input_embeds_num_tokens = 0

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_8FlashNextForCausalLM(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        if not get_pp_group().is_first_rank and self.use_deepstack:
            assert self.language_model.model.start_layer >= len(
                config.vision_config.deepstack_visual_indexes
            ), (
                "start_layer should be greater than or equal to "
                "len(deepstack_visual_indexes)"
            )
        self._set_moe_parameters()

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
        )
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        if self.language_model_only:
            raise ValueError(
                "Qwen3.8-Flash-Next language_model_only does not accept "
                "multimodal embeddings"
            )

        is_multimodal = _require_is_multimodal(is_multimodal)
        if self.use_deepstack:
            deepstack_input_embeds, multimodal_embeddings = (
                self._compute_deepstack_embeds(
                    inputs_embeds=inputs_embeds,
                    multimodal_embeddings=multimodal_embeddings,
                    is_multimodal=is_multimodal,
                )
            )
        else:
            deepstack_input_embeds = None

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )
        if deepstack_input_embeds is not None:
            self._set_deepstack_input_embeds(deepstack_input_embeds)
        return inputs_embeds

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        return self.language_model.get_mtp_target_hidden_states()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        if inputs_embeds is not None and get_pp_group().is_first_rank:
            deepstack_input_embeds = self._get_deepstack_input_embeds(
                inputs_embeds.size(0)
            )
        else:
            deepstack_input_embeds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            query_start_loc=kwargs.get("query_start_loc"),
            ngram_context=kwargs.get("ngram_context"),
            deepstack_input_embeds=deepstack_input_embeds,
        )
        if inputs_embeds is not None and get_pp_group().is_first_rank:
            self._clear_deepstack_input_embeds(inputs_embeds.size(0))
        return hidden_states

    @validate_ple_embedding_weights
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mapper = self.hf_to_vllm_mapper | WeightsMapper(
            orig_to_new_substr={"mtp.": None},
            orig_to_new_prefix={"visual.": None} if self.language_model_only else {},
        )
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=_QWEN38_FLASH_NEXT_IGNORED_MISSING_SUFFIXES.copy(),
        )
        loaded = loader.load_weights(weights, mapper=mapper)
        pack_gated_hc_projection_weights(self)
        return loaded

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return Qwen3_8FlashNextForCausalLM.get_mamba_state_dtype_from_config(
            vllm_config
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return Qwen3_8FlashNextForCausalLM.get_mamba_state_shape_from_config(
            vllm_config
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return Qwen3_8FlashNextForCausalLM.get_mamba_state_copy_func()

    @classmethod
    def get_mamba_state_copy_funcs(
        cls,
        mamba_types: set[MambaAttentionBackendEnum],
    ) -> MambaStateCopyFuncsByType:
        return Qwen3_8FlashNextForCausalLM.get_mamba_state_copy_funcs(mamba_types)

    @classmethod
    def get_mamba_specs_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[MambaSpec, ...]:
        return Qwen3_8FlashNextForCausalLM.get_mamba_specs_from_config(vllm_config)


__all__ = [
    "Qwen3_8FlashNextDecoderLayer",
    "Qwen3_8FlashNextForCausalLM",
    "Qwen3_8FlashNextForConditionalGeneration",
    "Qwen3_8FlashNextModel",
    "Qwen3_8FlashNextSparseMoeBlock",
]
