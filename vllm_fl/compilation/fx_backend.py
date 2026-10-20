# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional FX rewrites before eager execution; device capture is independent.

Reuse vLLM's eager compiler contract and PyTorch's forward normalization. No
device compiler, generated-code cache or global backend patches are installed.
The platform owns the pass manager and its vendor-specific fusion rules.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from functorch.compile import make_boxed_func
from torch import fx
from torch._dynamo.backends.common import aot_autograd
from torch._inductor.compile_fx import graph_returns_tuple, make_graph_return_tuple

from vllm.compilation.compiler_interface import EagerAdaptor
from vllm.compilation.passes.inductor_pass import pass_context
from vllm.config.utils import Range
from vllm.platforms import current_platform


class FXFusionAdaptor(EagerAdaptor):
    """Apply configured normalized-forward passes and execute the resulting FX.

    This is not the ``aot_eager`` torch.compile backend: vLLM owns compilation,
    while AOTAutograd only normalizes the forward for existing fusion patterns.
    Alias/input-mutation handling and output unwrapping remain PyTorch-owned.
    Like EagerAdaptor, this adapter has no persistent compiled artifacts.
    """

    name = "fl_fx_fusion"

    def compute_hash(self, vllm_config):
        from vllm_fl.compilation.config import validate_ascend_compilation_options

        options = validate_ascend_compilation_options(
            (vllm_config.additional_config or {}).get("ascend_compilation_config", {})
        )
        self._npugraph_compiler = None
        if options["enable_npugraph_ex"]:
            # Preserve explicit DSv4 npugraph/static-kernel configurations.
            # The default remains upstream's FX fusion / eager compiler.
            from vllm_fl.compilation.compiler_interface import AscendCompiler

            self._npugraph_compiler = AscendCompiler()
            return self._npugraph_compiler.compute_hash(vllm_config)
        return super().compute_hash(vllm_config)

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        compile_range: Range,
        key: str | None = None,
    ):
        npugraph_compiler = getattr(self, "_npugraph_compiler", None)
        if npugraph_compiler is not None:
            return npugraph_compiler.compile(
                graph, example_inputs, compiler_config, compile_range, key
            )
        manager = compiler_config[current_platform.pass_key]
        if not manager.passes:
            return super().compile(
                graph, example_inputs, compiler_config, compile_range, key
            )

        # Preserve the caller's graph and current tracing-mode ownership, as
        # in the matching rc1 normalized-FX path, without its device backend.
        graph = copy.deepcopy(graph)
        from torch._guards import detect_fake_mode

        fake_mode = detect_fake_mode()
        if fake_mode is not None:
            example_inputs = [
                fake_mode.from_tensor(value)
                if isinstance(value, torch.Tensor)
                and hasattr(value, "fake_mode")
                and value.fake_mode is not fake_mode
                else value
                for value in example_inputs
            ]

        # vLLM supplies its IR functionalization here. It must run before AOT
        # tracing, not on the normalized graph passed to vendor fusion rules.
        pre_grad = compiler_config.get("pre_grad_custom_pass")
        if pre_grad is not None:
            with pass_context(compile_range):
                pre_grad(graph.graph)
            graph.recompile()

        def optimize(normalized: fx.GraphModule, inputs: list[Any]):
            del inputs
            with pass_context(compile_range):
                manager(normalized.graph)
            normalized.recompile()
            return make_boxed_func(normalized)

        backend = aot_autograd(fw_compiler=optimize)
        if not graph_returns_tuple(graph):
            compiled = make_graph_return_tuple(graph, example_inputs, backend)
        else:
            compiled = backend(graph, example_inputs)
        return compiled, None
