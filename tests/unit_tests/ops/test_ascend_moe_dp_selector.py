"""Exercise the production selector without loading accelerator dependencies."""

import ast
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest


@pytest.fixture
def selector(monkeypatch):
    source = Path(__file__).resolve().parents[3] / "vllm_fl/ops/fused_moe/fused_moe_utils.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "select_unquantized_moe_backend_oot")
    # Only isolate imports; run the actual function, including every NPU guard.
    function.returns = None
    for argument in function.args.args + function.args.kwonlyargs:
        argument.annotation = None
    platform = NS(device_type="npu", dist_backend="hccl",
                  is_cpu=lambda: False, is_tpu=lambda: False)
    experts = object()
    namespace = {"torch": NS(bfloat16="bf16"),
                 "current_platform": platform,
                 "UnquantizedMoeBackend": NS(TRITON="triton"),
                 "TritonExpertsFL": experts}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    parallel = NS(dp_size=2, tp_size=16, ep_size=1, use_ep=False, enable_eplb=False,
                  pcp_size=1, use_batched_activation_format=False,
                  is_sequence_parallel=False, all2all_backend="allgather_reducescatter")
    config = NS(moe_parallel_config=parallel, is_lora_enabled=False,
                moe_backend="auto", in_dtype="bf16", num_experts=512)
    engine = NS(
        model_config=NS(hf_config=NS(model_type="qwen4_exp")),
        parallel_config=NS(expert_placement_strategy="linear"),
        compilation_config=NS(pass_config=NS()),
    )
    vllm_config = ModuleType("vllm.config")
    vllm_config.get_current_vllm_config_or_none = lambda: engine
    monkeypatch.setitem(sys.modules, "vllm.config", vllm_config)
    return namespace[function.name], config, platform, experts, engine


@pytest.mark.parametrize("backend", ["auto", "triton"])
def test_dp_without_ep_keeps_flattened_parallel_configuration(selector, backend):
    select, config, _, experts, _ = selector
    config.moe_backend = backend
    before = vars(config.moe_parallel_config).copy()
    assert select(config) == ("triton", experts)
    assert vars(config.moe_parallel_config) == before


@pytest.mark.parametrize("owner,attribute,value", [
    ("config", "in_dtype", "fp16"),
    ("platform", "dist_backend", "nccl"),
    ("parallel", "all2all_backend", "deepep_high_throughput"),
    ("engine_parallel", "expert_placement_strategy", "round_robin"),
    ("parallel", "enable_eplb", True),
    ("parallel", "pcp_size", 2),
    ("parallel", "use_batched_activation_format", True),
    ("config", "is_lora_enabled", True),
    ("config", "moe_backend", "flashinfer_trtllm"),
])
def test_unsupported_dp_contract_is_rejected(selector, owner, attribute, value):
    select, config, platform, _, engine = selector
    target = {
        "config": config,
        "platform": platform,
        "parallel": config.moe_parallel_config,
        "engine_parallel": engine.parallel_config,
    }[owner]
    setattr(target, attribute, value)
    with pytest.raises(NotImplementedError):
        select(config)


def test_qwen_ep16_with_vllm032_pass_config(selector):
    select, config, _, experts, _ = selector
    config.moe_parallel_config.use_ep = True
    config.moe_parallel_config.ep_size = 16
    config.moe_parallel_config.is_sequence_parallel = True
    assert select(config) == ("triton", experts)


def test_existing_single_dp_does_not_gain_bf16_restriction(selector):
    select, config, _, experts, _ = selector
    config.moe_parallel_config.dp_size = 1
    config.in_dtype = "fp16"
    assert select(config) == ("triton", experts)
