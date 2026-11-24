# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM-Ascend v0.32.0rc1's Qwen4Exp PLE host lookup and Engram UVA.
"""Private, CANN-mapped BF16 PLE shards for FL."""

import ctypes as C
from functools import cache

import torch
from torch import nn
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)
CHUNK_ROWS = 1 << 22
_HOST_ALIGNMENT = 64 * 1024
_MAPPED_PINNED = 0x2 | 0x10000000
_MAX_PROGRAMS_PER_LAUNCH = 65535


@cache
def _host_library():
    lib = C.CDLL("libascendcl.so")
    pointer = C.c_void_p
    signatures = {
        "aclrtMallocHost": [C.POINTER(pointer), C.c_size_t, C.c_uint32],
        "aclrtFreeHost": [pointer],
        "aclrtHostRegisterV2": [pointer, C.c_uint64, C.c_uint32],
        "aclrtHostGetDevicePointer": [pointer, C.POINTER(pointer), C.c_uint32],
        "aclrtHostUnregister": [pointer],
    }
    for name, args in signatures.items():
        call = getattr(lib, name)
        call.argtypes = args
        call.restype = C.c_int
    return lib


def _check(name, code):
    if code:
        raise RuntimeError(f"Qwen PLE host offload: {name} failed, rc={code}")


class PLEHostTable:
    """Own the allocation and its device mapping through graph teardown."""

    def __init__(self, rows: int, width: int, device: torch.device, chunk_rows: int = CHUNK_ROWS):
        self.lib = _host_library()
        self.device = device
        self.allocation = C.c_void_p()
        self.address = C.c_void_p()
        self.registered = False
        self.tensor = None
        self.ptrs = None
        row_bytes = width * 2
        size = rows * row_bytes
        if rows <= 0 or width <= 0 or chunk_rows <= 0:
            raise ValueError("PLE table shape and chunk size must be positive")
        _check("aclrtMallocHost", self.lib.aclrtMallocHost(
            C.byref(self.allocation), size + _HOST_ALIGNMENT - 1, 0
        ))
        try:
            base = self.allocation.value
            if base is None:
                raise RuntimeError("aclrtMallocHost returned NULL")
            aligned = (base + _HOST_ALIGNMENT - 1) // _HOST_ALIGNMENT * _HOST_ALIGNMENT
            self.address = C.c_void_p(aligned)
            _check("aclrtHostRegisterV2", self.lib.aclrtHostRegisterV2(
                self.address, size, _MAPPED_PINNED
            ))
            self.registered = True
            device_address = C.c_void_p()
            _check("aclrtHostGetDevicePointer", self.lib.aclrtHostGetDevicePointer(
                self.address, C.byref(device_address), 0
            ))
            if device_address.value is None:
                raise RuntimeError("aclrtHostGetDevicePointer returned NULL")
            self.tensor = torch.frombuffer(
                (C.c_char * size).from_address(aligned), dtype=torch.bfloat16,
            ).view(rows, width)
            self.ptrs = torch.tensor(
                [device_address.value + i * row_bytes for i in range(0, rows, chunk_rows)],
                device=device,
                dtype=torch.int64,
            )
        except Exception:
            try:
                self.close()
            except Exception as cleanup_error:
                cleanup_error.host_table = self
                raise
            raise

    def close(self):
        if not self.allocation.value:
            return
        if self.ptrs is not None:
            torch.npu.synchronize(self.device)
        if self.registered:
            _check("aclrtHostUnregister", self.lib.aclrtHostUnregister(self.address))
            self.registered = False
        _check("aclrtFreeHost", self.lib.aclrtFreeHost(self.allocation))
        self.allocation = C.c_void_p()
        self.tensor = None
        self.ptrs = None


@triton.jit
def _ple_host_lookup_kernel(
    ptrs, ids, output, start, end,
    WIDTH: tl.constexpr, CHUNK: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    index = tl.load(ids + row).to(tl.int64)
    owned = (index >= start) & (index < end)
    local = tl.where(owned, index - start, 0)
    pointer = tl.load(ptrs + local // CHUNK).to(tl.pointer_type(tl.bfloat16))
    col = tl.arange(0, BLOCK)
    offset = (local % CHUNK) * WIDTH + col
    values = tl.load(pointer + offset, mask=owned & (col < WIDTH), other=0)
    tl.store(output + row.to(tl.int64) * WIDTH + col, values, mask=col < WIDTH)


def _fake_lookup(
    ptrs: torch.Tensor, ids: torch.Tensor, start: int, end: int, width: int, chunk_rows: int,
) -> torch.Tensor:
    return torch.empty((*ids.shape, width), dtype=torch.bfloat16, device=ids.device)


def _lookup(
    ptrs: torch.Tensor, ids: torch.Tensor, start: int, end: int, width: int, chunk_rows: int,
) -> torch.Tensor:
    output = _fake_lookup(ptrs, ids, start, end, width, chunk_rows)
    flat_ids = ids.contiguous().view(-1)
    flat_output = output.view(flat_ids.numel(), width)
    for first in range(0, flat_ids.numel(), _MAX_PROGRAMS_PER_LAUNCH):
        last = min(first + _MAX_PROGRAMS_PER_LAUNCH, flat_ids.numel())
        _ple_host_lookup_kernel[(last - first,)](
            ptrs, flat_ids[first:last], flat_output[first:last], start, end,
            width, chunk_rows, triton.next_power_of_2(width),
        )
    return output


direct_register_custom_op(
    op_name="qwen38_ple_host_lookup",
    op_func=_lookup,
    mutates_args=[],
    fake_impl=_fake_lookup,
)


class QwenPLEHostEmbedding(VocabParallelEmbedding):
    def __init__(self, *args, **kwargs):
        device = torch.device("npu", torch.npu.current_device())
        # Reuse vLLM's partition and weight metadata without allocating the
        # full shard in HBM, including transiently during initialization.
        with torch.device("meta"):
            super().__init__(*args, **kwargs)
        layout = self.weight
        if layout.dtype != torch.bfloat16:
            raise ValueError("Qwen PLE host offload requires a BF16 table")
        self._ple_host_table = PLEHostTable(*layout.shape, device)
        try:
            weight = nn.Parameter(self._ple_host_table.tensor, requires_grad=False)
            weight.__dict__.update(vars(layout))
            self.weight = weight
            self.weight.data.zero_()
            # Prevent the generic quant-method post-loader from staging the
            # entire CPU parameter onto the accelerator.
            self.quant_method = None
        except Exception:
            self._ple_host_table.close()
            raise
        logger.info("Qwen PLE host offload mapped %d BF16 rows on %s", layout.shape[0], device)

    def forward(self, input_):
        shard = self.shard_indices
        output = torch.ops.vllm.qwen38_ple_host_lookup(
            self._ple_host_table.ptrs, input_, shard.org_vocab_start_index,
            shard.org_vocab_end_index, self.embedding_dim, CHUNK_ROWS,
        )
        if self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output)
        return output

    def close_host_offload(self):
        self._ple_host_table.close()
        self.register_parameter("weight", None)
