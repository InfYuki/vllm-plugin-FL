# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM-Ascend v0.24.0rc1 hybrid cache configuration and views.
"""Dense Ascend hybrid cache views over the scheduler's shared byte pools.

Each pool stores all convolution states, then equally sized SSM/K slabs and
the V slab. Cache groups retain ownership of their scheduler-assigned block
IDs; the views must not allocate independent storage or change those IDs.
"""

import math
from dataclasses import replace

from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec


def pad_attention_specs(specs):
    """Account for convolution bytes before the shared-pool planner runs."""
    page_size = max(
        (s.page_size_bytes for s in specs.values() if isinstance(s, MambaSpec)),
        default=0,
    )
    return {
        name: replace(spec, page_size_padded=page_size)
        if isinstance(spec, AttentionSpec) and spec.page_size_bytes < page_size
        else spec
        for name, spec in specs.items()
    }


def dense_attention_view(raw, spec, shape):
    """Keep planar K/V at the pool tail, without per-forward cache copies."""
    payload = math.prod(shape) * get_dtype_size(spec.dtype)
    offset = raw.nbytes - payload
    if raw.element_size() != 1 or offset < 0 or offset % get_dtype_size(spec.dtype):
        raise ValueError("Invalid Ascend hybrid attention backing")
    return raw[offset:].view(spec.dtype).view(shape)


def dense_mamba_views(raw, spec, num_blocks):
    """Group each state type densely so NPU gathers only requested rows."""
    if raw.element_size() != 1:
        raise ValueError("Ascend hybrid cache requires byte backing")
    states = []
    offset = 0
    for shape, dtype in zip(spec.shapes, spec.dtypes):
        end = offset + num_blocks * math.prod(shape) * get_dtype_size(dtype)
        if offset % get_dtype_size(dtype) or end > raw.nbytes:
            raise ValueError("Invalid Ascend hybrid state backing")
        states.append(raw[offset:end].view(dtype).view(num_blocks, *shape))
        offset = end
    return states
