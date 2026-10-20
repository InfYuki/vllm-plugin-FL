# Copyright (c) 2026 BAAI. All rights reserved.

from contextlib import contextmanager
from types import MethodType, SimpleNamespace

import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm_fl.compilation.graph_params import (
    clear_graph_params,
    get_graph_params,
    prepare_graph_params,
)

import vllm_fl.attention.ascend.attention as attention


def _config(*, pa_shape_list=None, speculative=False, quant=False):
    return SimpleNamespace(
        speculative_config=object() if speculative else None,
        quant_config=object() if quant else None,
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace()),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY
        ),
        additional_config=(
            {} if pa_shape_list is None else {"pa_shape_list": pa_shape_list}
        ),
    )


def test_using_paged_attention_defaults_to_fia_and_honors_explicit_shapes() -> None:
    assert attention.using_paged_attention(1, _config()) is False
    assert attention.using_paged_attention(1, _config(pa_shape_list=[1])) is True
    assert attention.using_paged_attention(
        1, _config(pa_shape_list=[1], speculative=True)
    ) is False
    assert attention.using_paged_attention(7, _config(), head_size=512) is True


def test_using_paged_attention_rejects_invalid_shape_config() -> None:
    with pytest.raises(TypeError, match="must contain integers"):
        attention.using_paged_attention(1, _config(pa_shape_list=[True]))


class _FakeEvent:
    def __init__(self, calls):
        self.calls = calls

    def wait(self, stream):
        self.calls.append(("event-wait", stream))

    def reset(self, stream):
        self.calls.append(("event-reset", stream))

    def record(self, stream):
        self.calls.append(("event-record", stream))


@pytest.fixture
def graph_task_runtime(monkeypatch):
    clear_graph_params()
    calls = []
    capture_stream = object()

    @contextmanager
    def stream_context(stream):
        calls.append(("stream-enter", stream))
        try:
            yield
        finally:
            calls.append(("stream-exit", stream))

    monkeypatch.setattr(
        attention.torch,
        "npu",
        SimpleNamespace(
            current_stream=lambda: capture_stream,
            ExternalEvent=lambda: _FakeEvent(calls),
            graph_task_group_begin=lambda stream: calls.append(
                ("capture-begin", stream)
            ),
            graph_task_group_end=lambda stream: calls.append(
                ("capture-end", stream)
            ) or "handle",
            graph_task_update_begin=lambda stream, handle: calls.append(
                ("update-begin", stream, handle)
            ),
            graph_task_update_end=lambda stream: calls.append(
                ("update-end", stream)
            ),
            stream=stream_context,
        ),
    )
    monkeypatch.setattr(
        attention.AscendAttentionBackendImpl,
        "_weak_ref_tensor",
        staticmethod(lambda tensor: tensor),
    )
    yield calls, capture_stream
    clear_graph_params()


def _impl(config):
    impl = SimpleNamespace(
        vllm_config=config,
        speculative_config=None,
        sliding_window=None,
        sinks=None,
        num_kv_heads=2,
        num_heads=4,
        head_size=8,
        scale=0.125,
        key_cache=torch.empty(2, 16, 2, 8),
        value_cache=torch.empty(2, 16, 2, 8),
        _weak_ref_tensor=lambda tensor: tensor,
        _require_regular_graph_capture=(
            lambda metadata, layer: layer.layer_name
        ),
    )
    impl._get_fia_params = MethodType(
        attention.AscendAttentionBackendImpl._get_fia_params, impl
    )
    return impl


def _metadata(seq_len=6):
    return SimpleNamespace(
        attn_state=attention.AscendAttentionState.DecodeOnly,
        block_tables=torch.tensor([[0]], dtype=torch.int32),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32),
        seq_lens_list=[seq_len],
        actual_seq_lengths_q=[1],
        attn_mask=None,
        causal=True,
    )


@pytest.mark.parametrize(
    "model_window,layer_window",
    [("absent", None), (None, None), (16, None), (16, 16)],
)
def test_fia_capture_and_replay_rebind_live_lengths_and_tables(
    monkeypatch, graph_task_runtime, model_window, layer_window
) -> None:
    calls, capture_stream = graph_task_runtime
    op_calls = []
    workspace = torch.empty(8)
    out_op = SimpleNamespace(
        out=lambda **kwargs: (
            calls.append("fia-op"),
            op_calls.append(("fia", kwargs)),
        )
    )
    monkeypatch.setattr(
        attention.torch_npu,
        "_npu_fused_infer_attention_score_get_max_workspace",
        lambda **kwargs: op_calls.append(("workspace", kwargs)) or workspace,
    )
    monkeypatch.setattr(
        attention.torch_npu,
        "npu_fused_infer_attention_score",
        out_op,
    )
    prepare_graph_params(1)
    config = _config()
    if model_window != "absent":
        config.model_config.hf_text_config.sliding_window = model_window
    impl = _impl(config)
    impl.sliding_window = layer_window
    query = torch.empty(1, 4, 8)
    output = torch.empty_like(query)
    captured = _metadata(6)
    layer = SimpleNamespace(layer_name="model.layers.0.self_attn.attn")

    result = attention.AscendAttentionBackendImpl._capture_fused_infer_attention(
        impl, query, object(), object(), captured, output, layer
    )
    assert result is output
    params = get_graph_params()
    assert isinstance(params.attn_params[1][0], attention.FusedInferAttentionGraphParam)
    assert params.workspaces[1] is workspace
    assert params.handles[1] == ["handle"]
    expected_sparse = 4 if layer_window else 3
    expected_pre = layer_window or attention._SWA_INT_MAX
    expected_next = 0 if layer_window else attention._SWA_INT_MAX
    for _, kwargs in op_calls:
        assert kwargs["sparse_mode"] == expected_sparse
        assert kwargs["pre_tokens"] == expected_pre
        assert kwargs["next_tokens"] == expected_next
    param = params.attn_params[1][0]
    assert param.block_table is captured.block_tables
    assert (param.sparse_mode, param.pre_tokens, param.next_tokens) == (
        expected_sparse, expected_pre, expected_next
    )

    live = _metadata(9)
    live.block_tables = torch.tensor([[3]], dtype=torch.int32)
    # The retained table is a static buffer; subsequent input preparation
    # updates its contents in place rather than replacing its address.
    captured.block_tables.fill_(5)
    update_stream = object()
    attention.AscendAttentionBackendImpl.update_graph_params(
        update_stream,
        SimpleNamespace(attn_metadata={layer.layer_name: live}),
        1,
        config,
    )
    replay_kwargs = op_calls[-1][1]
    assert replay_kwargs["actual_seq_lengths"] is live.actual_seq_lengths_q
    assert replay_kwargs["actual_seq_lengths_kv"] is live.seq_lens_list
    expected_table = (
        live.block_tables if model_window == "absent" else captured.block_tables
    )
    assert replay_kwargs["block_table"] is expected_table
    assert replay_kwargs["block_table"].item() == (
        3 if model_window == "absent" else 5
    )
    assert replay_kwargs["sparse_mode"] == expected_sparse
    assert replay_kwargs["pre_tokens"] == expected_pre
    assert replay_kwargs["next_tokens"] == expected_next
    assert ("capture-begin", capture_stream) in calls
    assert ("update-begin", update_stream, "handle") in calls
    assert ("event-record", update_stream) in calls
    assert calls == [
        ("event-wait", capture_stream),
        ("event-reset", capture_stream),
        ("capture-begin", capture_stream),
        "fia-op",
        ("capture-end", capture_stream),
        ("stream-enter", update_stream),
        ("update-begin", update_stream, "handle"),
        "fia-op",
        ("update-end", update_stream),
        ("event-record", update_stream),
        ("stream-exit", update_stream),
    ]


def test_pa_capture_and_replay_rebind_live_context_tensor(
    monkeypatch, graph_task_runtime
) -> None:
    calls, _ = graph_task_runtime
    op_calls = []
    workspace = torch.empty(8)
    monkeypatch.setattr(
        attention.torch_npu,
        "_npu_paged_attention_get_workspace",
        lambda **kwargs: op_calls.append(("workspace", kwargs)) or workspace,
    )
    monkeypatch.setattr(
        attention.torch_npu,
        "_npu_paged_attention",
        lambda **kwargs: (
            calls.append("pa-op"),
            op_calls.append(("pa", kwargs)),
        ),
    )
    prepare_graph_params(1)
    config = _config(pa_shape_list=[1])
    impl = _impl(config)
    query = torch.empty(1, 4, 8)
    output = torch.empty_like(query)
    captured = _metadata(6)
    layer = SimpleNamespace(layer_name="model.layers.0.self_attn.attn")

    attention.AscendAttentionBackendImpl._capture_paged_attention(
        impl, query, captured, output, layer
    )
    assert isinstance(
        get_graph_params().attn_params[1][0], attention.PagedAttentionGraphParam
    )

    live = _metadata(11)
    update_stream = object()
    attention.AscendAttentionBackendImpl.update_graph_params(
        update_stream,
        SimpleNamespace(attn_metadata={layer.layer_name: live}),
        1,
        config,
    )
    replay_kwargs = op_calls[-1][1]
    assert replay_kwargs["context_lens"] is live.seq_lens
    assert replay_kwargs["block_table"] is live.block_tables
    assert ("update-begin", update_stream, "handle") in calls
    assert calls == [
        ("event-wait", graph_task_runtime[1]),
        ("event-reset", graph_task_runtime[1]),
        ("capture-begin", graph_task_runtime[1]),
        "pa-op",
        ("capture-end", graph_task_runtime[1]),
        ("stream-enter", update_stream),
        ("update-begin", update_stream, "handle"),
        "pa-op",
        ("update-end", update_stream),
        ("event-record", update_stream),
        ("stream-exit", update_stream),
    ]


def test_replay_dispatches_from_captured_pa_type_for_large_head_fallback(
    monkeypatch, graph_task_runtime
) -> None:
    calls, _ = graph_task_runtime
    op_calls = []
    monkeypatch.setattr(
        attention.torch_npu,
        "_npu_paged_attention_get_workspace",
        lambda **kwargs: torch.empty(1),
    )
    monkeypatch.setattr(
        attention.torch_npu,
        "_npu_paged_attention",
        lambda **kwargs: op_calls.append(kwargs),
    )
    prepare_graph_params(1)
    query = torch.empty(1, 1, 512)
    params = get_graph_params()
    event = _FakeEvent(calls)
    params.attn_params[1].append(
        attention.PagedAttentionGraphParam(
            "layer", query, query, query, 1, 1, 1.0, query, query, query
        )
    )
    params.handles[1].append("handle")
    params.events[1].append(event)
    live = _metadata(13)

    # pa_shape_list remains empty: head-size fallback selected PA at capture.
    attention.AscendAttentionBackendImpl.update_graph_params(
        object(), SimpleNamespace(attn_metadata={"layer": live}), 1, _config()
    )
    assert op_calls[-1]["context_lens"] is live.seq_lens


def test_replay_rejects_mixed_attention_task_kinds(graph_task_runtime) -> None:
    prepare_graph_params(1)
    query = torch.empty(1, 1, 8)
    params = get_graph_params()
    params.attn_params[1].extend(
        [
            attention.PagedAttentionGraphParam(
                "layer", query, query, query, 1, 1, 1.0, query, query, query
            ),
            attention.FusedInferAttentionGraphParam(
                "layer", query, query, query, query, None, 1, [1], [1],
                1, 1, 1.0, query, query, 3, 1, 1,
            ),
        ]
    )
    params.handles[1].extend([object(), object()])
    params.events[1].extend([object(), object()])

    with pytest.raises(RuntimeError, match="cannot mix"):
        attention.AscendAttentionBackendImpl.update_graph_params(
            object(),
            SimpleNamespace(attn_metadata={"layer": _metadata()}),
            1,
            _config(),
        )


def test_regular_graph_capture_rejects_unsupported_only_on_capture_path() -> None:
    impl = SimpleNamespace(
        vllm_config=_config(speculative=True),
        sliding_window=None,
        sinks=None,
    )
    metadata = _metadata()

    with pytest.raises(NotImplementedError, match="speculative"):
        attention.AscendAttentionBackendImpl._require_regular_graph_capture(
            impl, metadata, SimpleNamespace(layer_name="layer")
        )


@pytest.mark.parametrize("exclusion", ["speculative", "quant", "sinks"])
def test_swa_capture_retains_unsupported_feature_guards(exclusion) -> None:
    impl = _impl(_config(
        speculative=exclusion == "speculative", quant=exclusion == "quant"
    ))
    impl.sliding_window = 16
    impl.sinks = object() if exclusion == "sinks" else None
    with pytest.raises(NotImplementedError):
        attention.AscendAttentionBackendImpl._require_regular_graph_capture(
            impl, _metadata(), SimpleNamespace(layer_name="layer")
        )


def test_regular_graph_capture_accepts_swa_decode() -> None:
    impl = _impl(_config())
    impl.sliding_window = 16
    assert attention.AscendAttentionBackendImpl._require_regular_graph_capture(
        impl, _metadata(), SimpleNamespace(layer_name="layer")
    ) == "layer"


@pytest.mark.parametrize(
    "state",
    [attention.AscendAttentionState.PrefillNoCache,
     attention.AscendAttentionState.PrefillCacheHit,
     attention.AscendAttentionState.DecodeOnly,
     attention.AscendAttentionState.ChunkedPrefill],
)
@pytest.mark.parametrize("window", [None, 16])
@pytest.mark.parametrize("causal", [True, False])
def test_fia_eager_preserves_swa_parameters_and_cache_paths(
    monkeypatch, state, window, causal
) -> None:
    monkeypatch.setattr(attention, "get_forward_context", lambda: SimpleNamespace())
    op_calls = []
    query = torch.empty(2, 4, 8)
    key = torch.empty(2, 2, 8)
    value = torch.empty_like(key)
    output = torch.full_like(query, -1)
    expected = torch.full((1, 4, 8), 7.0)
    monkeypatch.setattr(
        attention.torch_npu, "npu_fused_infer_attention_score",
        lambda **kwargs: (op_calls.append(kwargs) or expected, None),
    )
    impl = _impl(_config())
    impl.sliding_window = window
    metadata = _metadata()
    metadata.attn_state = state
    metadata.causal = causal
    metadata.attn_mask = torch.ones(2, 2, dtype=torch.bool)
    result = attention.AscendAttentionBackendImpl.forward_fused_infer_attention(
        impl, query, key, value, metadata, output
    )
    kwargs = op_calls[0]
    assert result is output
    assert torch.equal(output[:1], expected)
    assert torch.all(output[1:] == -1)
    if causal:
        assert kwargs["atten_mask"] is metadata.attn_mask
    else:
        assert "atten_mask" not in kwargs
    assert kwargs["sparse_mode"] == (
        (4 if window is not None else 3) if causal else 0
    )
    if window is not None and causal:
        assert kwargs["pre_tokens"] == window
        assert kwargs["next_tokens"] == 0
    else:
        assert "pre_tokens" not in kwargs and "next_tokens" not in kwargs
    if state == attention.AscendAttentionState.PrefillNoCache:
        assert kwargs["block_table"] is None
        assert kwargs["key"].shape[0] == 1
        assert kwargs["actual_seq_lengths_kv"] is metadata.actual_seq_lengths_q
    else:
        assert kwargs["block_size"] == 16
        assert kwargs["key"].data_ptr() == impl.key_cache.data_ptr()
        assert kwargs["actual_seq_lengths_kv"] is metadata.seq_lens_list


def test_swa_decode_uses_fia_even_when_paged_attention_shape_matches() -> None:
    impl = _impl(_config(pa_shape_list=[1]))
    impl.sliding_window = 16
    calls = []
    impl.forward_paged_attention = lambda *args: pytest.fail("SWA selected PA")
    impl.forward_fused_infer_attention = lambda *args: calls.append(args) or args[4]
    output = torch.empty(1, 4, 8)
    result = attention.AscendAttentionBackendImpl.forward_impl(
        impl, output, object(), object(), (), _metadata(), output
    )
    assert result is output
    assert len(calls) == 1


def test_swa_model_replay_uses_each_layers_live_lengths_and_captured_table(
    monkeypatch, graph_task_runtime
) -> None:
    op_calls = []
    monkeypatch.setattr(
        attention.torch_npu, "_npu_fused_infer_attention_score_get_max_workspace",
        lambda **kwargs: torch.empty(1),
    )
    monkeypatch.setattr(
        attention.torch_npu, "npu_fused_infer_attention_score",
        SimpleNamespace(out=lambda **kwargs: op_calls.append(kwargs)),
    )
    prepare_graph_params(1)
    config = _config()
    config.model_config.hf_text_config.sliding_window = 16
    captured = {}
    for name, window in [("local", 16), ("global", None)]:
        impl = _impl(config)
        impl.sliding_window = window
        captured[name] = _metadata()
        query = torch.empty(1, 4, 8)
        attention.AscendAttentionBackendImpl._capture_fused_infer_attention(
            impl, query, object(), object(), captured[name],
            torch.empty_like(query), SimpleNamespace(layer_name=name),
        )
    # Metadata dictionary order is independent of capture order.
    live = {"global": _metadata(20), "local": _metadata(10)}
    live["global"].actual_seq_lengths_q = [2]
    attention.AscendAttentionBackendImpl.update_graph_params(
        object(), SimpleNamespace(attn_metadata=live), 1, config
    )
    for kwargs, name, mode in zip(op_calls[-2:], ["local", "global"], [4, 3]):
        assert kwargs["actual_seq_lengths_kv"] is live[name].seq_lens_list
        assert kwargs["actual_seq_lengths"] is live[name].actual_seq_lengths_q
        assert kwargs["block_table"] is captured[name].block_tables
        assert kwargs["sparse_mode"] == mode
