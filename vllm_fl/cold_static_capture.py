"""Keep Qwen's collective static-kernel builds inside coordinated V2 startup/capture."""

from contextvars import ContextVar
from functools import wraps
import logging
from weakref import WeakValueDictionary

logger = logging.getLogger("vllm.plugins.fl")
_NPU_WORKER = "vllm_ascend.worker.worker.NPUWorker"
_active_capture = ContextVar("fl_ascend_static_capture", default=None)
_ready_runners = WeakValueDictionary()


def _install_eager_compile_guard():
    from npugraph_ex._acl_concrete_graph.acl_graph import AclGraph
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context, is_forward_context_available

    native = AclGraph.run_eagerly_compile
    if getattr(native, "_fl_capture_guard", False):
        return

    @wraps(native)
    def run_eagerly_compile(graph, *args, **kwargs):
        if is_forward_context_available():
            context = get_forward_context()
            if context.cudagraph_runtime_mode == CUDAGraphMode.NONE:
                # vLLM places this exact config-owned dictionary in each forward context.
                owner = _ready_runners.get(id(context.no_compile_layers))
                if (owner is not None and owner is not _active_capture.get()
                        and owner.vllm_config.compilation_config.static_forward_context is context.no_compile_layers):
                    return None
        return native(graph, *args, **kwargs)

    run_eagerly_compile._fl_capture_guard = True
    AclGraph.run_eagerly_compile = run_eagerly_compile


def _attach_capture(runner):
    native = runner.capture_model
    if getattr(native, "_fl_capture_guard", False):
        return
    _install_eager_compile_guard()

    @wraps(native)
    def capture_model(*, profile_only=False):
        token = _active_capture.set(runner)
        try:
            result = native(profile_only=profile_only)
        finally:
            _active_capture.reset(token)
        if not profile_only:
            counts = runner.cudagraph_manager.captured_token_counts()
            layers = runner.vllm_config.compilation_config.static_forward_context
            if counts and _ready_runners.get(id(layers)) is not runner:
                _ready_runners[id(layers)] = runner
                logger.info("FL enabled the eager-prefill static compile guard after official V2 "
                            "FULL_DECODE_ONLY capture, captured token counts %s", counts)
        return result

    capture_model._fl_capture_guard = True
    runner.capture_model = capture_model


def _is_target(runner):
    if (type(runner).__module__, type(runner).__name__) != ("vllm_ascend.worker.v2.model_runner", "NPUModelRunner"):
        return False
    from vllm.config import CUDAGraphMode
    from vllm_ascend.ascend_config import get_ascend_config

    config = runner.vllm_config
    if (not config.use_v2_model_runner or config.parallel_config.data_parallel_size <= 1
            or config.speculative_config is not None
            or config.compilation_config.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY
            or getattr(config.model_config.hf_config, "model_type", None) != "qwen4_exp"):
        return False
    compilation = get_ascend_config().ascend_compilation_config
    return compilation.enable_npugraph_ex and compilation.enable_static_kernel


def _wrap_worker_init(worker):
    native = worker.init_device
    if getattr(native, "_fl_capture_guard", False):
        return

    @wraps(native)
    def init_device(self, *args, **kwargs):
        result = native(self, *args, **kwargs)
        runner = self.model_runner
        if _is_target(runner):
            _attach_capture(runner)
        return result

    init_device._fl_capture_guard = True
    worker.init_device = init_device


def install_worker_hook():
    # WorkerWrapperBase.init_worker loads general plugins before resolving the worker.
    # API processes need neither the NPU worker nor the torch-npu compiler here.
    from vllm.v1.worker import worker_base

    resolve = worker_base.resolve_obj_by_qualname
    if getattr(resolve, "_fl_capture_guard", False):
        return

    @wraps(resolve)
    def resolve_worker(qualname):
        obj = resolve(qualname)
        if qualname == _NPU_WORKER:
            _wrap_worker_init(obj)
            if worker_base.resolve_obj_by_qualname is resolve_worker:
                worker_base.resolve_obj_by_qualname = resolve
        return obj

    resolve_worker._fl_capture_guard = True
    worker_base.resolve_obj_by_qualname = resolve_worker
