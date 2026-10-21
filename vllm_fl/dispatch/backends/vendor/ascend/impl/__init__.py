# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend operator implementations.
https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/torch_npu_list.md
"""

from importlib import import_module

__all__ = [
    "silu_and_mul_ascend",
    "rms_norm_ascend",
    "rotary_embedding_ascend",
    "AscendAttentionBackend",
    "AscendAttentionBackendImpl",
    "AscendAttentionMetadataBuilder",
    "AscendMetadata",
    "AscendAttentionState",
    "AscendMLABackend",
    "is_torch_npu_available",
    "AttentionMaskBuilder",
    "get_attention_mask_builder",
]


def __getattr__(name):
    """Preserve exports without loading unrelated device implementations."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = {
        "silu_and_mul_ascend": ".activation",
        "rms_norm_ascend": ".normalization",
        "rotary_embedding_ascend": ".rotary",
        "AttentionMaskBuilder": ".attention.standard_mask",
        "get_attention_mask_builder": ".attention.standard_mask",
    }.get(name, ".attention.attention")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
