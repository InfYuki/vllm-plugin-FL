"""Backend task-updater contracts; these tests do not capture device graphs."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_fl.compilation.graph_runtime import GraphRuntimeController


def _bind(*backends):
    controller = GraphRuntimeController(vllm_config=SimpleNamespace())
    controller.bind_attention_groups(
        [[SimpleNamespace(backend=backend) for backend in backends]]
    )
    return controller


@pytest.mark.parametrize("side_cache", [False, True])
def test_qsa_model_owned_graph_does_not_resolve_attention_impl(monkeypatch, side_cache):
    if side_cache:
        from vllm_fl.models.qwen3_8_flash_next.common.qsa_cache import QSAStateBackend
        backend = QSAStateBackend
    else:
        from vllm_fl.models.qwen3_8_flash_next.gpu.qsa import (
            Qwen3_8FlashNextQSAAttentionBackend,
        )
        backend = Qwen3_8FlashNextQSAAttentionBackend
    factory = Mock(side_effect=AssertionError("QSA has no attention implementation"))
    monkeypatch.setattr(backend, "get_impl_cls", factory)
    assert _bind(backend).attention_impls == ()
    factory.assert_not_called()


def test_unmarked_factory_error_remains_fatal():
    class OrdinaryBackend:
        @staticmethod
        def get_impl_cls():
            raise NotImplementedError("missing implementation")

    with pytest.raises(RuntimeError, match="Failed to resolve Ascend graph backend") as exc:
        _bind(OrdinaryBackend)
    assert isinstance(exc.value.__cause__, NotImplementedError)


def test_ordinary_backend_requires_task_updater():
    class OrdinaryBackend:
        @staticmethod
        def get_impl_cls():
            return type("IncompleteImpl", (), {})

    with pytest.raises(RuntimeError, match="lacks update_graph_params"):
        _bind(OrdinaryBackend)


def test_mixed_backends_keep_ordinary_updater_and_deduplicate():
    class ModelOwnedBackend:
        model_owned_graph_execution = True
        get_impl_cls = Mock(side_effect=AssertionError("must not be called"))

    class OrdinaryImpl:
        update_graph_params = Mock()

    class OrdinaryBackend:
        get_impl_cls = Mock(return_value=OrdinaryImpl)

    controller = _bind(ModelOwnedBackend, OrdinaryBackend, OrdinaryBackend)
    assert controller.attention_impls == (OrdinaryImpl,)
    ModelOwnedBackend.get_impl_cls.assert_not_called()
    OrdinaryBackend.get_impl_cls.assert_called_once_with()
