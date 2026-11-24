# Copyright (c) 2026 BAAI. All rights reserved.
"""npugraph_ex FX adapter. FL's outer wrapper owns ACL graph capture."""

import copy
import os
import shutil
from hashlib import sha256

import torch
import torch_npu
from torch._guards import detect_fake_mode
from torch._inductor.compile_fx import graph_returns_tuple, make_graph_return_tuple
from vllm.compilation.compiler_interface import CompilerInterface
from vllm.logger import init_logger

logger = init_logger(__name__)

_STATIC_INCOMPATIBLE_OPS = [
    "TopKV2", "GenericBlockSparseAttention", "QuantBatchMatmulV3",
    "TransposeQuantBatchMatMul", "QuantLightningIndexer", "PoolKeyIndexer",
]


class AscendGraphCompiler(CompilerInterface):
    name = "FLAscendGraphCompiler"

    def compute_hash(self, vllm_config):
        self.vllm_config = vllm_config
        self.static_sizes = ()
        ascend = (vllm_config.additional_config or {}).get("ascend_compilation_config", {})
        base = f"fl-npugraph-ex-{torch_npu.__version__}"
        if not ascend.get("enable_static_kernel", False):
            return base
        query_len = vllm_config.num_speculative_tokens + 1
        max_tokens = vllm_config.scheduler_config.max_num_seqs * query_len
        capture_sizes = vllm_config.compilation_config.cudagraph_capture_sizes or []
        self.static_sizes = tuple(sorted(size for size in set(capture_sizes)
                                         if query_len <= size <= max_tokens))
        if not self.static_sizes:
            raise ValueError("FL static kernels require at least one decode capture size")
        return sha256(f"{base}:static:{self.static_sizes}".encode()).hexdigest()[:16]

    def initialize_cache(self, cache_dir, disable_cache=False, prefix=""):
        if not self.static_sizes:
            return
        from npugraph_ex._acl_concrete_graph import static_kernel

        if not shutil.which("op_compiler") or not static_kernel._use_process_isolation:
            raise RuntimeError("FL static kernels require op_compiler and isolated package installation")
        parallel = self.vllm_config.parallel_config
        if parallel.nnodes != 1 or parallel.data_parallel_external_lb:
            raise ValueError("FL static kernels currently require single-host internal DP")
        world_size = parallel.local_world_size * parallel.data_parallel_size_local
        configured = os.environ.setdefault("LOCAL_WORLD_SIZE", str(world_size))
        if configured != str(world_size):
            raise ValueError(f"LOCAL_WORLD_SIZE={configured} does not match {world_size} local FL ranks")
        static_kernel._set_static_kernel_blacklist(_STATIC_INCOMPATIBLE_OPS)
        # npugraph_ex owns the per-process package and its atexit cleanup.
        logger.info_once("FL static kernel decode sizes=%s, local world=%s", self.static_sizes, world_size)

    def compile(self, graph, example_inputs, compiler_config, compile_range, key=None):
        # torch_npu registers this bundled module as a top-level alias.
        import npugraph_ex
        from npugraph_ex.configs.compiler_config import _process_kwargs_options

        graph = copy.deepcopy(graph)
        fake_mode = detect_fake_mode()
        if fake_mode is not None:
            example_inputs = [
                fake_mode.from_tensor(value)
                if isinstance(value, torch.Tensor)
                and getattr(value, "fake_mode", fake_mode) is not fake_mode
                else value
                for value in example_inputs
            ]
        config = npugraph_ex.CompilerConfig()
        options = {
            "force_eager": True,
            "inplace_pass": False,
            "clone_input": False,
            "clone_output": False,
        }
        if self.static_sizes:
            options["static_kernel_compile"] = True
            options["_vllm_aclnn_static_kernel_sym_range"] = list(self.static_sizes)
        _process_kwargs_options(config, {"options": options})
        torch.npu.set_compile_mode(jit_compile=False)
        backend = npugraph_ex.get_npu_backend(compiler_config=config)
        logger.info_once("FL Ascend graph compiler selected npugraph_ex")
        if graph_returns_tuple(graph):
            return backend(graph, example_inputs), None
        return make_graph_return_tuple(graph, example_inputs, backend), None
