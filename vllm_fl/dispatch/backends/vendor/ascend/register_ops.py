# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend backend operator registrations.

This module registers all VENDOR (Ascend) implementations.
"""

from __future__ import annotations

import functools

from vllm_fl.dispatch.registry import OpRegistry
from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl


def _bind_is_available(fn, is_available_fn):
    """Wrap a function and bind _is_available attribute for OpImpl.is_available() check."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper._is_available = is_available_fn
    return wrapper


def register_builtins(registry: OpRegistry) -> None:
    """
    Register all Ascend (VENDOR) operator implementations.

    Args:
        registry: Registry to register into
    """
    from .ascend import AscendBackend

    backend = AscendBackend()
    is_avail = backend.is_available

    from vllm_fl.dispatch.backends.vendor.ascend.attention_selector import (
        native_dsa_backend,
        native_sfa_backend,
    )

    impls = [
        OpImpl(
            op_name="minimax_m3_sparse_kernels",
            impl_id="vendor.ascend",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.minimax_m3_sparse_kernels, is_avail),
            vendor="ascend",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="qkv_rmsnorm_mrope",
            impl_id="vendor.ascend",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.qkv_rmsnorm_mrope, is_avail),
            vendor="ascend",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="sparse_mla_attention_backend",
            impl_id="vendor.ascend",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(native_sfa_backend, is_avail),
            vendor="ascend",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="deepseek_v4_attention_backend",
            impl_id="vendor.ascend",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(native_dsa_backend, is_avail),
            vendor="ascend",
            priority=BackendPriority.VENDOR,
        ),
        OpImpl(
            op_name="qkv_rmsnorm_rope",
            impl_id="vendor.ascend",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.qkv_rmsnorm_rope, is_avail),
            vendor="ascend",
            priority=BackendPriority.VENDOR,
        ),
        # Activation
        OpImpl(
            op_name="silu_and_mul",
            impl_id="vendor.ascend",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.silu_and_mul, is_avail),
            vendor="ascend",
            priority=BackendPriority.VENDOR,
        ),
        # Activation (GELU variant; MiniMax-M3 vision tower / projector)
        OpImpl(
            op_name="gelu_and_mul",
            impl_id="vendor.ascend",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.gelu_and_mul, is_avail),
            vendor="ascend",
            priority=BackendPriority.VENDOR,
        ),
        # Normalization
        OpImpl(
            op_name="rms_norm",
            impl_id="vendor.ascend",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.rms_norm, is_avail),
            vendor="ascend",
            priority=BackendPriority.VENDOR,
        ),
        # Rotary Embedding
        OpImpl(
            op_name="rotary_embedding",
            impl_id="vendor.ascend",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.rotary_embedding, is_avail),
            vendor="ascend",
            priority=BackendPriority.VENDOR,
        ),
        # Attention Backend
        OpImpl(
            op_name="attention_backend",
            impl_id="vendor.ascend",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.attention_backend, is_avail),
            vendor="ascend",
            priority=BackendPriority.VENDOR,
        ),
        # MoE router: current-v0.24 A2/A3 native softmax top-k.
        OpImpl(
            op_name="topk_softmax",
            impl_id="vendor.ascend",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.topk_softmax, is_avail),
            vendor="ascend",
            priority=BackendPriority.VENDOR,
        ),
    ]

    impls.append(OpImpl(
        op_name="fused_experts",
        impl_id="vendor.ascend",
        kind=BackendImplKind.VENDOR,
        fn=_bind_is_available(backend.fused_experts, is_avail),
        vendor="ascend",
        priority=BackendPriority.VENDOR,
    ))
    registry.register_many(impls)

    # Keep model call contracts independent of the selected HC implementation.
    # Optional providers can register the same op names through FL discovery.
    registry.register_many([
        OpImpl(
            op_name=op_name,
            impl_id="vendor.ascend",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(fn, backend.is_hc_available),
            vendor="ascend",
            priority=BackendPriority.VENDOR,
        )
        for op_name, fn in (
            ("deepseek_v4_hc_pre", backend.deepseek_v4_hc_pre),
            ("deepseek_v4_hc_post", backend.deepseek_v4_hc_post),
        )
    ])
