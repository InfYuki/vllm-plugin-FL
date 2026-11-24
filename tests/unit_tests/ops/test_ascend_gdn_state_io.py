"""Run on Ascend: row correctness, bounded allocation, graph replay and GDN math."""
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")
pytest.importorskip("triton")
if not torch.npu.is_available():
    pytest.skip("Requires Ascend NPU", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[3]
IMPL = ROOT / "vllm_fl/dispatch/backends/vendor/ascend/impl"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


io = load("gdn_state_io_test_target", IMPL / "gdn_state_io.py")


def pool(shape, slots=64, gap=137, dtype=torch.float32):
    width = 1
    strides = []
    for size in reversed(shape):
        strides.insert(0, width)
        width *= size
    backing = torch.arange(slots * (width + gap), dtype=torch.float32).remainder_(97).to(dtype).npu()
    view = backing.as_strided((slots, *shape), (width + gap, *strides))
    return backing, view


@pytest.mark.parametrize("shape", [(5, 3), (2, 7, 9)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_padded_pool_preserves_all_unselected_storage(shape, dtype):
    backing, state = pool(shape, dtype=dtype)
    before = backing.cpu().clone()
    indices = torch.tensor([3, 0, -1, 17, 0], device="npu", dtype=torch.int64)
    got = io.gather_state_rows(state, indices)
    expected = state.cpu().index_select(0, indices.cpu().clamp_min(0))
    torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)
    rows = torch.full_like(got, -4)
    io.scatter_state_rows_(state, indices, rows)
    expected_backing = before.clone()
    expected_view = expected_backing.as_strided(state.shape, state.stride())
    expected_view[3].fill_(-4)
    expected_view[17].fill_(-4)
    torch.testing.assert_close(backing.cpu(), expected_backing, rtol=0, atol=0)


def test_noncontiguous_inner_dimensions():
    backing, original = pool((2, 7, 9))
    state = original.transpose(-1, -2)
    before = backing.cpu().clone()
    indices = torch.tensor([2, -1, 5], device="npu", dtype=torch.int32)
    rows = io.gather_state_rows(state, indices)
    torch.testing.assert_close(rows.cpu(), state.cpu()[[2, 0, 5]], rtol=0, atol=0)
    rows.fill_(7)
    io.scatter_state_rows_(state, indices, rows)
    expected = before.as_strided(state.shape, state.stride())
    expected[2].fill_(7)
    expected[5].fill_(7)
    torch.testing.assert_close(backing.cpu(), before, rtol=0, atol=0)


def test_graph_replay_reads_new_indices_and_preserves_padding():
    backing, state = pool((2, 7, 9))
    indices = torch.tensor([2, 0, -1, 6], device="npu")
    for _ in range(3):
        warm = io.gather_state_rows(state, indices)
        io.scatter_state_rows_(state, indices, warm)
    torch.npu.synchronize()
    before = backing.cpu().clone()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        rows = io.gather_state_rows(state, indices)
        rows.add_(1)
        io.scatter_state_rows_(state, indices, rows)
    addresses = (state.data_ptr(), rows.data_ptr(), indices.data_ptr())
    backing.copy_(before.npu())
    indices.copy_(torch.tensor([9, -1, 0, 11], device="npu"))
    graph.replay()
    torch.npu.synchronize()
    expected = before.clone()
    expected_view = expected.as_strided(state.shape, state.stride())
    expected_view[9].add_(1)
    expected_view[11].add_(1)
    torch.testing.assert_close(backing.cpu(), expected, rtol=0, atol=0)
    assert addresses == (state.data_ptr(), rows.data_ptr(), indices.data_ptr())


def test_workspace_does_not_scale_with_pool_size():
    peaks = []
    for slots in (32, 512):
        backing, state = pool((2, 64, 64), slots=slots, gap=4096)
        indices = torch.tensor([2, 0, -1, 7], device="npu")
        warm = io.gather_state_rows(state, indices)
        io.scatter_state_rows_(state, indices, warm)
        torch.npu.synchronize()
        del warm
        torch.npu.reset_peak_memory_stats()
        baseline = torch.npu.memory_allocated()
        rows = io.gather_state_rows(state, indices)
        io.scatter_state_rows_(state, indices, rows)
        torch.npu.synchronize()
        peaks.append(torch.npu.max_memory_allocated() - baseline)
        assert rows.numel() == 4 * 2 * 64 * 64
        del rows, state, backing, indices
    # Allow allocator granularity, but reject full 24-MiB pool materialization.
    assert max(peaks) < 4 * 1024 * 1024, peaks
    assert abs(peaks[1] - peaks[0]) < 1024 * 1024, peaks


def dense_decode_reference(module, x, a, b, indices, conv, ssm, output, **kw):
    """Original dense tensor recurrence; deliberately uses native indexed copies."""
    slots = indices.clamp_min(0)
    valid = indices > 0
    conv_old = conv.index_select(0, slots)
    conv_new = conv_old.clone()
    x = module.causal_conv1d_update_torch(x.unsqueeze(-1), conv_new,
        kw["conv_weights"], kw["conv_bias"], kw["activation"]).squeeze(-1)
    conv.index_copy_(0, slots, torch.where(valid[:, None, None], conv_new, conv_old))
    q, k, v = module._prepare_qkv(x, kw["num_key_heads"], kw["num_value_heads"], kw["head_k_dim"], kw["head_v_dim"])
    old = ssm.index_select(0, slots)
    state = torch.where(valid[:, None, None, None], old, 0).float()
    g, beta = module._gating(a, b, kw["A_log"], kw["dt_bias"], packed_beta=kw["packed_beta"])
    state = state * g.exp()[..., None, None]
    delta = (v.float() - (state * k.unsqueeze(-2)).sum(-1)) * beta[..., None]
    state = state + delta.unsqueeze(-1) * k.unsqueeze(-2)
    result = (state * (q * kw["head_k_dim"]**-0.5).unsqueeze(-2)).sum(-1)
    output.copy_(torch.where(valid[:, None, None], result, 0).to(output.dtype))
    ssm.index_copy_(0, slots, torch.where(valid[:, None, None, None], state.to(old.dtype), old))


@pytest.mark.parametrize("packed_beta", [False, True])
def test_complete_decode_matches_original_dense_recurrence(packed_beta):
    # Load by path with a synthetic package to avoid plugin activation side effects.
    import sys
    import types
    package = types.ModuleType("gdn_validation_package")
    package.__path__ = [str(IMPL)]
    sys.modules[package.__name__] = package
    module = load(package.__name__ + ".gdn_reference", IMPL / "gdn_reference.py")
    torch.manual_seed(38)
    lanes, hk, hv, dk, dv, history = 5, 1, 2, 8, 8, 3
    channels = 2 * hk * dk + hv * dv
    cb, conv = pool((channels, history), dtype=torch.bfloat16)
    sb, ssm = pool((hv, dv, dk))
    # Small states and weights make tolerances sensitive to incorrect row addressing.
    cb.mul_(0.001)
    sb.mul_(0.001)
    indices = torch.tensor([3, 0, -1, 17, 0], device="npu")
    x = torch.randn(lanes, channels, device="npu", dtype=torch.bfloat16) * 0.1
    a = torch.randn(lanes, hv, device="npu", dtype=torch.bfloat16) * 0.1
    b = torch.randn_like(a)
    kw = dict(conv_weights=torch.randn(channels, history + 1, device="npu", dtype=torch.bfloat16) * 0.1,
              conv_bias=None, activation="silu", A_log=torch.zeros(hv, device="npu"),
              dt_bias=torch.zeros(hv, device="npu"), num_key_heads=hk, num_value_heads=hv,
              head_k_dim=dk, head_v_dim=dv, packed_beta=packed_beta)
    dense_conv = conv.cpu().contiguous().npu()
    dense_ssm = ssm.cpu().contiguous().npu()
    got = torch.empty(lanes, hv, dv, device="npu", dtype=torch.bfloat16)
    expected = torch.empty_like(got)
    cb_before, sb_before = cb.cpu(), sb.cpu()
    dense_decode_reference(module, x.clone(), a, b, indices, dense_conv, dense_ssm, expected, **kw)
    module.gdn_decode(x.clone(), a, b, indices, conv, ssm, got, **kw)
    torch.testing.assert_close(got.cpu(), expected.cpu(), rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(conv.cpu(), dense_conv.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(ssm.cpu(), dense_ssm.cpu(), rtol=1e-5, atol=1e-6)
    cb_expected, sb_expected = cb_before.clone(), sb_before.clone()
    for target, state, dense in ((cb_expected, conv, dense_conv), (sb_expected, ssm, dense_ssm)):
        view = target.as_strided(state.shape, state.stride())
        for slot in (3, 17):
            view[slot].copy_(dense[slot].cpu())
    torch.testing.assert_close(cb.cpu(), cb_expected, rtol=0, atol=0)
    torch.testing.assert_close(sb.cpu(), sb_expected, rtol=1e-5, atol=1e-6)
