# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8 input state for the FL NPU Model Runner V2."""

from vllm_fl.worker.v2.mamba_align import FLMambaHybridModelState

from .model_state import Qwen3_8FlashNextModelState


class Qwen3_8FlashNextNPUModelState(
    Qwen3_8FlashNextModelState, FLMambaHybridModelState
):
    """Combine Qwen PLE graph inputs with FL hybrid-cache state handling.

    The Qwen class allocates stable query-offset and n-gram buffers. Its
    cooperative super calls run the FL hybrid initialization and input hooks,
    while attention metadata is built by the FL hybrid state.
    """
