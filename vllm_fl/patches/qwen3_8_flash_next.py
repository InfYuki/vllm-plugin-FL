# Copyright (c) 2026 BAAI. All rights reserved.
"""Register FL's Qwen3.8-Flash-Next implementation on Ascend."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.model_executor.models.config import (
    Qwen3_5ForConditionalGenerationConfig,
    _strip_qwen4_exp_mrope,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class Qwen3_8FlashNextForConditionalGenerationConfig(
    Qwen3_5ForConditionalGenerationConfig
):
    """Apply the hybrid cache contract and enforce the supported runtime."""

    @staticmethod
    def verify_and_update_config(vllm_config: VllmConfig) -> None:
        Qwen3_5ForConditionalGenerationConfig.verify_and_update_config(vllm_config)
        model_config = vllm_config.model_config
        text_config = model_config.hf_text_config
        parallel_config = vllm_config.parallel_config

        if int(text_config.hc_count) <= 1:
            raise ValueError("Qwen4Exp requires hc_count > 1")

        uses_ple = bool(text_config.ple_layer_ids)
        uses_qsa = getattr(text_config, "indexer_n_heads", None) is not None
        if (uses_ple or uses_qsa) and parallel_config.use_ubatching:
            raise NotImplementedError(
                "Qwen4Exp PLE/QSA does not support dual-batch overlap or "
                "microbatching"
            )
        if uses_ple and parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "Qwen4Exp PLE requires pipeline_parallel_size=1 because raw "
                "token n-gram context is not broadcast between PP stages"
            )
        if vllm_config.speculative_config is not None:
            raise NotImplementedError(
                "FL Qwen4Exp on Ascend does not support speculative decoding"
            )

        multimodal_config = model_config.multimodal_config
        if multimodal_config is not None and multimodal_config.language_model_only:
            _strip_qwen4_exp_mrope(model_config)


class Qwen3_8FlashNextForCausalLMConfig(Qwen3_8FlashNextForConditionalGenerationConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: VllmConfig) -> None:
        Qwen3_8FlashNextForConditionalGenerationConfig.verify_and_update_config(
            vllm_config
        )
        _strip_qwen4_exp_mrope(vllm_config.model_config)


_ARCHITECTURES = {
    "Qwen3_8FlashNextForCausalLM": Qwen3_8FlashNextForCausalLMConfig,
    "Qwen3_8FlashNextForConditionalGeneration": (
        Qwen3_8FlashNextForConditionalGenerationConfig
    ),
    "Qwen4ExpForCausalLM": Qwen3_8FlashNextForCausalLMConfig,
    "Qwen4ExpForConditionalGeneration": Qwen3_8FlashNextForConditionalGenerationConfig,
}


def apply_qwen3_8_flash_next_patches() -> None:
    """Register the checkpoint aliases and the FL compilation boundaries."""
    from vllm.config.compilation import CompilationConfig
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models import config as model_config
    from vllm.transformers_utils import config as transformers_config
    from vllm_fl.models.qwen3_8_flash_next.config import (
        Qwen3_8FlashNextConfig,
        Qwen3_8FlashNextTextConfig,
    )

    # Both names use the concrete config classes to keep text and vision
    # sub-configs typed when Transformers loads a composite checkpoint.
    transformers_config._CONFIG_REGISTRY.update(
        {
            "qwen3_8_flash_next": Qwen3_8FlashNextConfig,
            "qwen3_8_flash_next_text": Qwen3_8FlashNextTextConfig,
            "qwen4_exp": Qwen3_8FlashNextConfig,
            "qwen4_exp_text": Qwen3_8FlashNextTextConfig,
        }
    )
    model_config.MODELS_CONFIG_MAP.update(_ARCHITECTURES)
    module = "vllm_fl.models.qwen3_8_flash_next"
    for architecture in _ARCHITECTURES:
        ModelRegistry.register_model(architecture, f"{module}:{architecture}")

    for op in (
        "vllm::qwen3_8_flash_next_ple_short_conv",
        "vllm::qwen3_8_flash_next_qsa_with_output",
    ):
        if op not in CompilationConfig._attention_ops:
            CompilationConfig._attention_ops.append(op)
