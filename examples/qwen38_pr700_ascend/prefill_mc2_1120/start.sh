#!/usr/bin/env bash
# Candidate for the BF16 TP8 x DP2 service on port 8098.
set -eo pipefail
source /workspace/env.sh

export VLLM_PLUGINS=fl VLLM_FL_ASCEND_RUNTIME=official
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ENGINE_READY_TIMEOUT_S=4800
# CANN 9.2.1 A3 DispatchV2/CombineV2: EP16, 32 local experts, K=10,
# H=2560 and 512 rows/TP rank require 2866 MiB with no dedicated shared ranks.
# HCCL uses separate send and receive buffers.
export HCCL_BUFFSIZE=3072

args=(
    "$model"
    --host 0.0.0.0
    --port 8098
    --served-model-name qwen3.8-flash-next
    --trust-remote-code
    --dtype bfloat16
    --tensor-parallel-size 8
    --data-parallel-size 2
    --enable-expert-parallel
    --max-model-len 131072
    --max-num-seqs 32
    --max-num-batched-tokens 4096
    --gpu-memory-utilization 0.93
    --enable-prefix-caching
    --mamba-cache-mode align
    --async-scheduling
    --enable-auto-tool-choice
    --tool-call-parser qwen3_xml
    --reasoning-parser qwen3
    --engram-config '{"cpu_offload":true,"dp_shared_memory":false}'
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    --additional-config '{"enable_cpu_binding":true,"enable_shared_expert_dp":true,"enable_fused_mc2":0,"enable_prefill_mc2":true,"enable_flashcomm1":true,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"fuse_norm_quant":false,"enable_npugraph_ex":true,"enable_static_kernel":true,"enable_super_kernel":false}}'
)

if [[ $# == 1 && $1 == --check ]]; then
    echo "Qwen ordinary prefill MC2: capacity 4096 tokens, HCCL_BUFFSIZE=$HCCL_BUFFSIZE MiB"
    exec python -B /workspace/check_runtime.py "${args[@]}"
elif (( $# )); then
    echo "Usage: bash /workspace/prefill_mc2_1120/start.sh [--check]" >&2
    exit 2
fi

python - <<'PY'
import socket
with socket.socket() as check:
    check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        check.bind(("0.0.0.0", 8098))
    except OSError as error:
        raise SystemExit(f"Cannot bind Qwen port 8098: {error}") from error
PY
idle=$(npu-smi info | awk '/No running processes found in NPU/ {n++} END {print n+0}')
if [[ "$idle" != 8 ]]; then
    echo "Qwen TP8 x DP2 requires eight idle A3 boards (16 logical chips). Idle boards: $idle." >&2
    exit 1
fi
run=$(mktemp -d /workspace/service_$(date +%Y%m%d_%H%M%S)_XXXXXX)
ln -sfnT "${run##*/}" /workspace/latest_service
echo "Service log: $run/service.log"
cd "$run"
printf '%s\n' "$$" > server.pid
exec > >(tee "$run/service.log") 2>&1
echo "Qwen ordinary prefill MC2 configured: enable_prefill_mc2=true enable_fused_mc2=0 max_step_tokens=4096 HCCL_BUFFSIZE=$HCCL_BUFFSIZE MiB"
exec vllm serve "${args[@]}"
