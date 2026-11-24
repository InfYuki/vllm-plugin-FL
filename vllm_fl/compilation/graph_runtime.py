# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NPU task lifecycle hooks for FL's existing graph wrapper.

Task ordering follows vllm-ascend v0.24.0rc1 acl_graph.py and
model_runner_v1.py. Graph allocation, wrapper dispatch and input buffers remain
owned by FL's existing GraphWrapper and vLLM runner.
"""

import os
from contextlib import contextmanager
from typing import Any

import torch
from vllm.config import CUDAGraphMode
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


class GraphRuntimeController:
    """Hold ordinary-attention task updaters for one NPU runner."""

    def __init__(self, *, vllm_config: Any, full_graph_wrapper_type: Any = None):
        self.vllm_config = vllm_config
        self.wrapper_type = full_graph_wrapper_type
        self.attention_impls: tuple[type, ...] | None = None
        self.update_stream = None
        self._active = False

    @staticmethod
    def _batch_descriptor(context: Any) -> Any:
        descriptor = getattr(context, "batch_descriptor", None)
        num_tokens = getattr(descriptor, "num_tokens", None)
        if not isinstance(num_tokens, int) or isinstance(num_tokens, bool):
            raise RuntimeError(
                "Ascend FULL graphs require integer BatchDescriptor.num_tokens"
            )
        return descriptor

    @contextmanager
    def capture_scope(self, context: Any):
        from vllm_fl.compilation.graph import weak_ref_tensors
        from vllm_fl.compilation.graph_params import (
            prepare_graph_params,
            weak_ref_workspace,
        )

        assert not self._active
        descriptor = self._batch_descriptor(context)
        prepare_graph_params(descriptor)
        self._active = True
        context.capturing = True
        try:
            yield
        finally:
            try:
                weak_ref_workspace(descriptor, weak_ref_tensors)
            finally:
                context.capturing = False
                self._active = False

    def _use_native_decode_replay(self, context: Any) -> bool:
        if (
            os.getenv("VLLM_FL_QWEN38_NATIVE_DECODE") != "1"
            or current_platform.device_type != "npu"
            or self.attention_impls != ()
            or context.cudagraph_runtime_mode != CUDAGraphMode.FULL
            or not getattr(context.batch_descriptor, "uniform", False)
            or context.ubatch_slices is not None
        ):
            return False

        config = self.vllm_config
        model = config.model_config
        parallel = config.parallel_config
        if (
            getattr(model.hf_config, "model_type", None) != "qwen4_exp"
            or model.dtype != torch.bfloat16
            or model.quantization is not None
            or config.speculative_config is not None
            or config.kv_transfer_config is not None
            or config.ec_transfer_config is not None
            or config.lora_config is not None
            or config.compilation_config.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY
            or parallel.pipeline_parallel_size != 1
            or parallel.tensor_parallel_size != 8
            or parallel.data_parallel_size != 2
            or parallel.data_parallel_size_local != 2
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or not parallel.enable_expert_parallel
            or parallel.enable_eplb
            or parallel.use_ubatching
        ):
            return False
        device = current_platform.torch_device_fn
        return device.current_stream() == device.default_stream()

    @contextmanager
    def replay_scope(
        self,
        context: Any,
        *,
        synchronize_device: bool = False,
        allow_native_decode: bool = False,
    ):
        assert not self._active
        self._active = True
        try:
            native_decode = allow_native_decode and self._use_native_decode_replay(
                context
            )
            if native_decode:
                # On this path the runner stages PLE, attention and MC2 inputs
                # on the replay stream. Shared-expert work rejoins that stream
                # inside the graph. There are no external attention updates.
                logger.info_once("FL Qwen3.8 native FULL decode replay is active.")
            else:
                device = current_platform.torch_device_fn
                if synchronize_device:
                    device.synchronize()
                # Ordinary attention updaters release external graph events on
                # their own stream after the previous replay has finished.
                device.current_stream().synchronize()
            yield
            self._update_attention(context)
        finally:
            self._active = False

    def _update_attention(self, context: Any) -> None:
        if self.attention_impls is None:
            raise RuntimeError(
                "Ascend attention backends must be bound before FULL replay"
            )
        if self.attention_impls:
            if self.update_stream is None:
                self.update_stream = current_platform.torch_device_fn.Stream()
            for impl in self.attention_impls:
                impl.update_graph_params(
                    self.update_stream,
                    context,
                    self._batch_descriptor(context),
                    self.vllm_config,
                )

    def bind_attention_groups(self, attn_groups: Any) -> None:
        """Resolve backend factories while the runner's config is in scope."""
        from vllm.v1.attention.backend import AttentionBackend

        implementations = []
        seen_backends = set()
        for groups in attn_groups:
            for group in groups:
                backend = group.backend
                if backend in seen_backends:
                    continue
                seen_backends.add(backend)
                # Model-owned transactions capture their device operations directly
                # and do not create ordinary-attention tasks requiring updates.
                # Only an explicit capability opts out; factory failures on other
                # backends must still fail graph initialization.
                if getattr(backend, "model_owned_graph_execution", False) is True:
                    continue
                factory = getattr(backend, "get_impl_cls", None)
                if factory is None or factory is AttentionBackend.get_impl_cls:
                    continue
                try:
                    impl = factory()
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to resolve Ascend graph backend {backend.__name__}"
                    ) from exc
                if not callable(getattr(impl, "update_graph_params", None)):
                    raise RuntimeError(
                        f"Ascend graph backend {backend.__name__} lacks update_graph_params"
                    )
                if impl not in implementations:
                    implementations.append(impl)
        self.attention_impls = tuple(implementations)

    def clear_decoder_graphs(self) -> None:
        from vllm_fl.compilation.graph_params import clear_graph_params

        if self.wrapper_type is not None:
            self.wrapper_type.clear_all_graphs()
        clear_graph_params()
