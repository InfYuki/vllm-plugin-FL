# SPDX-License-Identifier: Apache-2.0
"""Native implementations of the backend-independent V4 HC call contract.

Both operations are functional: inputs are not rebound or mutated. Providers
must preserve the pre-operation's (hidden, post, combination) return order and
the post-operation's unbatched result shape. Native singleton-batch adaptation
belongs here rather than in the decoder. Importing this module does not load
the optional native extension.
"""

import torch


def is_hc_native_available() -> bool:
    """Inspect registered symbols without loading a native library."""
    namespace = torch.ops._C_ascend
    return hasattr(namespace, "npu_hc_pre_v2") and hasattr(namespace, "npu_hc_post")


def hc_pre_native(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops._C_ascend.npu_hc_pre_v2(
        x, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, norm_eps, hc_eps
    )


def hc_post_native(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    combination: torch.Tensor,
) -> torch.Tensor:
    return torch.ops._C_ascend.npu_hc_post(
        x.unsqueeze(0), residual.unsqueeze(0), post.unsqueeze(0),
        combination.unsqueeze(0),
    ).squeeze(0)
