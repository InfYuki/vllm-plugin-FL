# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU contract tests of production task methods, without NPU/vLLM imports."""

import __future__

import ast
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, Optional
from unittest.mock import Mock

import pytest
import torch


def _load_definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    exec(
        compile(tree, str(path), "exec", flags=__future__.annotations.compiler_flag),
        namespace,
    )


@pytest.fixture
def runtime():
    root = Path(__file__).resolve().parents[3]
    state = SimpleNamespace(
        events={2: []},
        handles={2: []},
        attention_params={2: []},
        workspaces={2: []},
    )
    calls = []
    stream = object()
    npu = SimpleNamespace(
        current_stream=lambda: stream,
        stream=lambda value: nullcontext(),
        ExternalEvent=lambda: Mock(),
        graph_task_group_begin=Mock(),
        graph_task_group_end=Mock(side_effect=["pa", "fia8", "fia64"]),
        graph_task_update_begin=Mock(),
        graph_task_update_end=Mock(),
    )

    class TorchProxy:
        def __getattr__(self, name):
            return getattr(torch, name)

    torch_proxy = TorchProxy()
    torch_proxy.npu = npu

    class FIA:
        def __call__(self, **kwargs):
            calls.append(("eager", kwargs))
            return torch.zeros_like(kwargs["query"]), None

        def out(self, **kwargs):
            calls.append(("fia", kwargs))

    ops = SimpleNamespace(
        npu_fused_infer_attention_score=FIA(),
        _npu_fused_infer_attention_score_get_max_workspace=Mock(
            side_effect=lambda **kw: torch.empty(9)
        ),
        _npu_paged_attention_get_workspace=Mock(
            side_effect=lambda **kw: torch.empty(7)
        ),
        _npu_paged_attention=lambda **kw: calls.append(("pa", kw)),
    )
    ns = dict(
        torch=torch_proxy,
        torch_npu=ops,
        dataclass=dataclass,
        field=field,
        Enum=Enum,
        Any=Any,
        ClassVar=ClassVar,
        List=list,
        Optional=Optional,
        Tuple=tuple,
        Type=type,
        AttentionImpl=object,
        AttentionLayer=object,
        nn=SimpleNamespace(Module=object),
        VllmConfig=object,
        CommonAttentionMetadata=object,
        AttentionCGSupport=SimpleNamespace(ALWAYS=1),
        AttentionMaskBuilder=object,
        _SWA_INT_MAX=2147483647,
        get_ascend_graph_params=lambda: state,
        is_ascend_graph_capturing=lambda: ns["capturing"],
        weak_ref_tensors=lambda tensor: tensor,
        capturing=True,
    )
    _load_definitions(
        root / "vllm_fl/dispatch/backends/vendor/ascend/impl/attention.py",
        {
            "AscendAttentionState",
            "AscendMetadata",
            "AscendAttentionMetadataBuilder",
            "AscendAttentionBackendImpl",
            "PagedAttentionGraphParam",
            "FusedInferAttentionGraphParam",
        },
        ns,
    )
    return SimpleNamespace(ns=ns, state=state, calls=calls, npu=npu, ops=ops)


def _metadata(rt, length=6):
    return rt.ns["AscendMetadata"](
        attn_state=rt.ns["AscendAttentionState"].DecodeOnly,
        actual_seq_lengths_q=[1, 2],
        seq_lens_list=[length, length + 1],
        seq_lens=torch.tensor([length, length + 1], dtype=torch.int32),
        block_tables=torch.tensor([[0], [1]], dtype=torch.int32),
        attn_mask=torch.ones(4, 4, dtype=torch.int8),
    )


def _impl(rt, window, heads=4):
    impl = object.__new__(rt.ns["AscendAttentionBackendImpl"])
    impl.sliding_window = window
    impl.num_heads, impl.num_kv_heads, impl.head_size = heads, 2, 8
    impl.scale = 0.125
    impl.key_cache = torch.empty(4, 128, 2, 8)
    impl.value_cache = torch.empty_like(impl.key_cache)
    return impl


@pytest.mark.parametrize("window", [8, 64])
def test_fia_capture_update_live_metadata_and_fixed_buffers(runtime, window):
    rt = runtime
    impl = _impl(rt, window)
    query = torch.empty(2, 4, 8)
    output = torch.empty_like(query)
    captured = _metadata(rt)
    assert impl.forward_impl(query, query, query, (), captured, output, "fia") is output
    param = rt.state.attention_params[2][0]
    assert isinstance(param, rt.ns["FusedInferAttentionGraphParam"])
    capture = rt.calls[-1][1]
    live = _metadata(rt, 133)
    live.actual_seq_lengths_q = [2]
    live.block_tables = torch.tensor([[2, 3], [1, 0]], dtype=torch.int32)
    update_stream = object()
    impl.update_graph_params(
        update_stream, SimpleNamespace(attn_metadata={"fia": live}), 2
    )
    update = rt.calls[-1][1]
    assert update["actual_seq_lengths"] is live.actual_seq_lengths_q
    assert update["actual_seq_lengths_kv"] is live.seq_lens_list
    assert update["block_table"] is live.block_tables
    assert update["out"][0] is capture["out"][0] is output
    assert update["out"][1] is capture["out"][1] is param.softmax_lse
    assert update["workspace"] is capture["workspace"]
    assert (update["sparse_mode"], update["pre_tokens"], update["next_tokens"]) == (
        4,
        window,
        0,
    )
    rt.npu.graph_task_update_begin.assert_called_once_with(update_stream, "pa")
    rt.state.events[2][0].record.assert_called_once_with(update_stream)
    rt.ns["capturing"] = False
    impl.forward_impl(query, query, query, (), live, output, "fia")
    eager = rt.calls[-1][1]
    assert (eager["sparse_mode"], eager["pre_tokens"], eager["next_tokens"]) == (
        4,
        window,
        0,
    )


def test_mixed_tasks_use_layer_metadata_and_distinct_workspaces(runtime):
    rt = runtime
    layers = [("pa", None, 4), ("fia8", 8, 4), ("fia64", 64, 8)]
    for name, window, heads in layers:
        query = torch.empty(2, heads, 8)
        _impl(rt, window, heads).forward_impl(
            query, query, query, (), _metadata(rt), torch.empty_like(query), name
        )
    assert len(rt.state.workspaces[2]) == 3
    assert len({id(workspace) for workspace in rt.state.workspaces[2]}) == 3
    live = {name: _metadata(rt, 129 + idx) for idx, (name, _, _) in enumerate(layers)}
    rt.ns["AscendAttentionBackendImpl"].update_graph_params(
        object(), SimpleNamespace(attn_metadata=live), 2
    )
    for (kind, kwargs), (name, window, _) in zip(rt.calls[-3:], layers):
        assert kwargs["block_table"] is live[name].block_tables
        if kind == "pa":
            assert kwargs["context_lens"] is live[name].seq_lens
        else:
            assert kwargs["actual_seq_lengths_kv"] is live[name].seq_lens_list
            assert kwargs["pre_tokens"] == window
    for event in rt.state.events[2]:
        event.record.assert_called_once()


def test_decode_builder_mask_for_sliding_window_only(runtime):
    cls = runtime.ns["AscendAttentionMetadataBuilder"]
    builder = object.__new__(cls)
    builder.model_config = SimpleNamespace(runner_type="generate", use_mla=False)
    mask = object()
    builder._get_mask_builder = lambda: SimpleNamespace(
        get_splitfuse_attn_mask=lambda: mask
    )
    state = runtime.ns["AscendAttentionState"].DecodeOnly
    builder.sliding_window = None
    assert builder._make_attention_mask(state) is None
    builder.sliding_window = 8
    assert builder._make_attention_mask(state) is mask


def test_builder_reads_window_from_cache_spec(runtime):
    runtime.ns["cdiv"] = lambda numerator, denominator: (
        (numerator + denominator - 1) // denominator
    )
    runtime.ns["AscendAttentionBackend"] = SimpleNamespace(
        get_supported_block_size=lambda: [128]
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=4096),
        speculative_config=None,
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
    )
    builder = runtime.ns["AscendAttentionMetadataBuilder"](
        SimpleNamespace(sliding_window=64), ["fia"], config, torch.device("cpu")
    )
    assert builder.sliding_window == 64


def test_weak_workspace_lifecycle_retains_each_task(runtime):
    root = Path(__file__).resolve().parents[3]
    owned = [torch.empty(7), torch.empty(9), torch.empty(11)]
    runtime.state.workspaces[2] = owned
    weakened = []

    def weak_ref(value):
        weakened.append(value)
        return ("weak", id(value))

    namespace = dict(_ascend_graph_params=runtime.state, weak_ref_tensors=weak_ref)
    _load_definitions(
        root / "vllm_fl/compilation/graph.py",
        {"weak_ref_ascend_graph_workspaces"},
        namespace,
    )
    namespace["weak_ref_ascend_graph_workspaces"]()
    assert len(weakened) == 3
    assert all(actual is expected for actual, expected in zip(weakened, owned))
    assert runtime.state.workspaces[2] == [("weak", id(value)) for value in owned]


def test_missing_live_layer_fails_instead_of_replaying_stale_lengths(runtime):
    rt = runtime
    query = torch.empty(2, 4, 8)
    impl = _impl(rt, 8)
    impl.forward_impl(
        query, query, query, (), _metadata(rt), torch.empty_like(query), "fia"
    )
    with pytest.raises(RuntimeError, match="Missing Ascend graph metadata"):
        impl.update_graph_params(object(), SimpleNamespace(attn_metadata={}), 2)
