# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.config.model import LogprobsMode
from vllm.v1.sample.ops.topk_topp_sampler import (
    TopKTopPSampler,
    apply_top_k_top_p_pytorch,
    random_sample,
)
from vllm.v1.sample.sampler import Sampler


class AscendTopKTopPSampler(TopKTopPSampler):
    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Use the same NPU-compatible filter as vLLM-Ascend V1. The upstream
        # Triton filter does not compile on the current Ascend backend.
        logits = apply_top_k_top_p_pytorch(logits, k, p)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        return random_sample(probs, generators, self.use_fp64_gumbel), logits_to_return


class AscendSampler(Sampler):
    def __init__(
        self,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        use_fp64_gumbel: bool = False,
    ) -> None:
        super().__init__(logprobs_mode, use_fp64_gumbel)
        self.topk_topp_sampler = AscendTopKTopPSampler(logprobs_mode, use_fp64_gumbel)
