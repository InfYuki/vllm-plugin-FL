# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.v1.worker import utils as upstream_utils

from vllm_fl.worker import kv_cache_utils


@pytest.fixture
def iluvatar(monkeypatch):
    platform = SimpleNamespace(
        vendor_name="iluvatar",
        device_type="cuda",
        is_cuda_alike=lambda: False,
        is_xpu=lambda: False,
        is_cpu=lambda: False,
    )
    monkeypatch.setattr(kv_cache_utils, "current_platform", platform)
    return platform


def test_upstream_rejection_is_limited_to_cache_binding(iluvatar, monkeypatch):
    monkeypatch.setattr(upstream_utils, "current_platform", iluvatar)
    caches = {
        "model.layers.0.main_attn": torch.empty(1),
        "model.layers.0.index_attn": torch.empty(1),
    }
    context = {name: SimpleNamespace(kv_cache=None) for name in caches}
    with pytest.raises(NotImplementedError):
        upstream_utils.bind_kv_cache(caches, context, [])

    runner_caches = []
    result = kv_cache_utils.bind_kv_cache(caches, context, runner_caches)

    assert result is None
    assert len(runner_caches) == 2
    assert all(context[name].kv_cache is tensor for name, tensor in caches.items())
    assert not iluvatar.is_cuda_alike()


def test_multiple_caches_keep_layer_order_and_tensor_identity(iluvatar):
    caches = {
        "model.layers.10.main_attn": torch.empty(2, 3),
        "model.layers.2.index_attn": torch.empty(3, 2).t(),
        "model.layers.10.index_attn": torch.empty(2, 3, dtype=torch.uint8),
        "model.layers.2.main_attn": torch.empty(2, 3),
    }
    context = {name: SimpleNamespace(kv_cache=None) for name in caches}
    runner_caches = []

    kv_cache_utils.bind_kv_cache(caches, context, runner_caches)

    expected = [
        "model.layers.2.index_attn",
        "model.layers.2.main_attn",
        "model.layers.10.main_attn",
        "model.layers.10.index_attn",
    ]
    assert len(runner_caches) == len(caches)
    for tensor, name in zip(runner_caches, expected):
        assert tensor is caches[name]
    for name, tensor in caches.items():
        assert context[name].kv_cache is tensor
    assert not caches["model.layers.2.index_attn"].is_contiguous()


def test_shared_cache_aliases_are_preserved(iluvatar):
    shared = torch.empty(2, 3)
    caches = {"model.layers.0.main_attn": shared, "model.layers.1.main_attn": shared}
    context = {name: SimpleNamespace() for name in caches}
    runner_caches = []

    kv_cache_utils.bind_kv_cache(caches, context, runner_caches)

    assert len(runner_caches) == 2
    assert runner_caches[0] is runner_caches[1] is shared
    assert all(layer.kv_cache is shared for layer in context.values())


@pytest.mark.parametrize("positional", [False, True])
def test_multiple_attention_module_indices(iluvatar, positional):
    caches = {
        "model.layers.1.attn.0": torch.empty(1),
        "model.layers.0.attn.1": torch.empty(1),
        "model.layers.0.attn.0": torch.empty(1),
    }
    context = {name: SimpleNamespace() for name in caches}
    runner_caches = []

    if positional:
        result = kv_cache_utils.bind_kv_cache(caches, context, runner_caches, 2)
    else:
        result = kv_cache_utils.bind_kv_cache(
            caches, context, runner_caches, num_attn_module=2
        )

    assert result is None
    assert len(runner_caches) == len(caches)
    for tensor, name in zip(runner_caches, sorted(caches)):
        assert tensor is caches[name]
        assert context[name].kv_cache is caches[name]


def test_existing_runner_cache_is_rejected(iluvatar):
    with pytest.raises(AssertionError):
        kv_cache_utils.bind_kv_cache({}, {}, [torch.empty(1)])


def test_empty_cache_binding(iluvatar):
    runner_caches = []
    kv_cache_utils.bind_kv_cache({}, {}, runner_caches)
    assert runner_caches == []


@pytest.mark.parametrize(
    ("vendor", "device_type"),
    [
        ("nvidia", "cuda"),
        ("metax", "cuda"),
        ("hygon", "cuda"),
        ("ascend", "npu"),
        ("mthreads", "musa"),
        ("iluvatar", "cpu"),
        (None, "cpu"),
    ],
)
@pytest.mark.parametrize("arguments", ["default", "positional", "keyword"])
def test_other_platforms_delegate_to_upstream(
    monkeypatch, vendor, device_type, arguments
):
    platform = SimpleNamespace(vendor_name=vendor, device_type=device_type)
    monkeypatch.setattr(kv_cache_utils, "current_platform", platform)
    upstream = Mock(return_value=None)
    monkeypatch.setattr(kv_cache_utils, "upstream_bind_kv_cache", upstream)
    extract = Mock(side_effect=AssertionError("Iluvatar binding must not run"))
    monkeypatch.setattr(kv_cache_utils, "extract_layer_index", extract)
    caches = {"model.layers.0.attn": torch.empty(1)}
    context = {name: SimpleNamespace(kv_cache=None) for name in caches}
    runner_caches = []

    if arguments == "default":
        result = kv_cache_utils.bind_kv_cache(caches, context, runner_caches)
    elif arguments == "positional":
        result = kv_cache_utils.bind_kv_cache(caches, context, runner_caches, 2)
    else:
        result = kv_cache_utils.bind_kv_cache(
            caches, context, runner_caches, num_attn_module=2
        )

    assert result is None
    upstream.assert_called_once_with(
        caches, context, runner_caches, 1 if arguments == "default" else 2
    )
    assert upstream.call_args.args[0] is caches
    assert upstream.call_args.args[1] is context
    assert upstream.call_args.args[2] is runner_caches
    extract.assert_not_called()


def _runner_binding_tail():
    """Execute the real runner's binding branch without initializing devices."""
    source = Path(__file__).resolve().parents[3] / "vllm_fl/worker/model_runner.py"
    tree = ast.parse(source.read_text())
    runner = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelRunnerFL"
    )
    method = next(
        node for node in runner.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "initialize_kv_cache_tensors"
    )
    binding = next(
        node for node in method.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute)
        and node.test.attr == "use_compress"
        and any(
            isinstance(child, ast.Name) and child.id == "bind_cache"
            for child in ast.walk(node)
        )
    )
    scaffold = ast.parse("def bind_runner(self, kv_caches):\n    pass\n")
    scaffold.body[0].body = binding.orelse
    ast.fix_missing_locations(scaffold)
    namespace = {"bind_kv_cache": kv_cache_utils.bind_kv_cache}
    exec(compile(scaffold, str(source), "exec"), namespace)
    return namespace["bind_runner"]


@pytest.mark.parametrize("cache_count", [1, 2])
@pytest.mark.parametrize(
    ("vendor", "architecture", "ascend_attention", "sfa", "local_binding"),
    [
        ("ascend", "MiniMaxM3ForCausalLM", True, False, True),
        ("ascend", "Glm4MoeForCausalLM", True, True, True),
        ("nvidia", "MiniMaxM3ForCausalLM", False, False, False),
        ("ascend", "Qwen3ForCausalLM", True, False, False),
        ("nvidia", "Qwen3ForCausalLM", False, False, False),
    ],
)
def test_runner_binding_routes_ascend_models_without_stale_upstream_alias(
    monkeypatch, cache_count, vendor, architecture, ascend_attention, sfa,
    local_binding,
):
    from vllm_fl.kv_cache.ascend import kv_cache_binding

    platform = SimpleNamespace(
        vendor_name=vendor, device_type="npu" if vendor == "ascend" else "cuda"
    )
    monkeypatch.setattr(kv_cache_utils, "current_platform", platform)
    caches = {"model.layers.0.main_attn": torch.empty(2, 3)}
    if cache_count == 2:
        caches["model.layers.0.index_attn"] = torch.empty(2, 3, dtype=torch.uint8)
    context = {name: SimpleNamespace(kv_cache=None) for name in caches}
    runner = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(
            model_type="test", architectures=[architecture]
        )),
        compilation_config=SimpleNamespace(static_forward_context=context),
        kv_caches=[], use_ascend_attention=ascend_attention, use_ascend_sfa=sfa,
    )
    original_helper = kv_cache_binding.bind_sfa_kv_cache
    helper = Mock(wraps=original_helper)
    monkeypatch.setattr(kv_cache_binding, "bind_sfa_kv_cache", helper)
    wrapper = Mock(wraps=kv_cache_utils.bind_kv_cache)
    bind_runner = _runner_binding_tail()
    bind_runner.__globals__["bind_kv_cache"] = wrapper

    if local_binding:
        # Simulate the wrapper importing the rejecting binder before M3's
        # process-global patch. Explicit Ascend routing must avoid that alias.
        upstream = Mock(side_effect=NotImplementedError("stale NPU binder"))
    else:
        upstream = Mock(wraps=original_helper)
    monkeypatch.setattr(kv_cache_utils, "upstream_bind_kv_cache", upstream)

    assert bind_runner(runner, caches) is None
    if local_binding:
        helper.assert_called_once_with(caches, context, runner.kv_caches, 1)
        wrapper.assert_not_called()
        upstream.assert_not_called()
    else:
        helper.assert_not_called()
        wrapper.assert_called_once_with(caches, context, runner.kv_caches, 1)
        upstream.assert_called_once_with(caches, context, runner.kv_caches, 1)
    assert len(runner.kv_caches) == cache_count
    for bound, (name, tensor) in zip(runner.kv_caches, caches.items()):
        assert bound is tensor
        assert context[name].kv_cache is tensor


@pytest.mark.parametrize("error_type", [RuntimeError, NotImplementedError])
def test_upstream_failure_after_mutation_propagates_without_retry(
    monkeypatch, error_type
):
    monkeypatch.setattr(
        kv_cache_utils,
        "current_platform",
        SimpleNamespace(vendor_name="nvidia", device_type="cuda"),
    )
    tensor = torch.empty(1)
    caches = {"model.layers.0.attn": tensor}
    context = {name: SimpleNamespace(kv_cache=None) for name in caches}
    runner_caches = []
    error = error_type("native binding failed after mutation")

    def fail_after_mutation(caches, context, runner_caches, num_attn_module):
        runner_caches.append(tensor)
        context["model.layers.0.attn"].kv_cache = tensor
        raise error

    upstream = Mock(side_effect=fail_after_mutation)
    extract = Mock(side_effect=AssertionError("Iluvatar binding must not run"))
    monkeypatch.setattr(kv_cache_utils, "upstream_bind_kv_cache", upstream)
    monkeypatch.setattr(kv_cache_utils, "extract_layer_index", extract)

    with pytest.raises(error_type) as raised:
        kv_cache_utils.bind_kv_cache(caches, context, runner_caches)

    assert raised.value is error
    upstream.assert_called_once_with(caches, context, runner_caches, 1)
    assert len(runner_caches) == 1
    assert runner_caches[0] is tensor
    assert context["model.layers.0.attn"].kv_cache is tensor
    extract.assert_not_called()
