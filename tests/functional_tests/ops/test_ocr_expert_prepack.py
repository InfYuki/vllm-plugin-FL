# SPDX-License-Identifier: Apache-2.0
"""NPU integration coverage for the load/reload contract of TP1 OCR experts."""

from types import SimpleNamespace

import pytest
import torch


def test_ocr_expert_prepack_reload_and_restore(monkeypatch):
    torch_npu = pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    from vllm_fl.dispatch.backends.vendor.ascend.impl.fused_moe import (
        prepack_ocr_expert_weights,
    )

    def make_weight(shape):
        return torch.nn.Parameter(
            torch.zeros(shape, device="npu", dtype=torch.bfloat16), requires_grad=False
        )

    layer = SimpleNamespace(
        w13_weight=make_weight((64, 1792, 1280)),
        w2_weight=make_weight((64, 1280, 896)),
    )
    weights = (layer.w13_weight, layer.w2_weight)
    source_ptrs = [w.data_ptr() for w in weights]
    previous = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    cast = torch_npu.npu_format_cast

    def fail_second(view, fmt):
        if view.shape[-1] == 1280:
            raise RuntimeError("second cast failed")
        return cast(view, fmt)

    with monkeypatch.context() as patch:
        patch.setattr(torch_npu, "npu_format_cast", fail_second)
        with pytest.raises(RuntimeError, match="second cast failed"):
            prepack_ocr_expert_weights(layer)
    assert torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT") == previous
    assert [w.data_ptr() for w in weights] == source_ptrs

    for weight in weights:
        weight[0, 5, 3] = 2
    prepack_ocr_expert_weights(layer)
    packed_ptrs = [w.data_ptr() for w in weights]
    for weight in weights:
        gmm = weight.transpose(1, 2)
        assert gmm.is_contiguous() and torch_npu.get_npu_format(gmm) == 29
        assert gmm[0, 3, 5].item() == 2
        weight[0, 5, 3] = 7
    prepack_ocr_expert_weights(layer)
    assert [w.data_ptr() for w in weights] == packed_ptrs
    assert all(w.transpose(1, 2)[0, 3, 5].item() == 7 for w in weights)
    assert torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT") == previous
