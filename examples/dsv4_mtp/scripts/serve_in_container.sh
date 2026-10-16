#!/usr/bin/env bash
set -Eeuo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
VARIANT=${1:-mtp1}
[[ "$VARIANT" == baseline || "$VARIANT" == mtp1 || "$VARIANT" == mtp2 || "$VARIANT" == mtp3 ]] || exit 2
MODEL=/public-flash/models/DeepSeek-V4-Flash-w8a8-mtp/
PORT=${PORT:-1112}
RUN_DIR="$HERE/runs/$VARIANT"
mkdir -p "$RUN_DIR"
python - "$RUN_DIR/identity.json" <<'PY'
import json,pathlib,sys,os
p=pathlib.Path(sys.argv[1])
if p.exists():
 r=json.loads(p.read_text()); proc=pathlib.Path(f"/proc/{r['pid']}/stat")
 if proc.exists():
  f=proc.read_text().rsplit(')',1)[1].split()
  if f[19]==r['starttime'] and f[0]!='Z': raise SystemExit('Recorded service is still alive; stop it explicitly first.')
PY
if [[ -f "$RUN_DIR/server.log" ]]; then cp -p "$RUN_DIR/server.log" "$RUN_DIR/server.$(date +%Y%m%d-%H%M%S).log"; fi
python - "$PORT" <<'PY'
import socket,sys
with socket.socket() as s: s.bind(('0.0.0.0',int(sys.argv[1])))
PY
unset VLLM_VERSION
unset LD_PRELOAD
unset VLLM_ASCEND_ENABLE_FLASHCOMM1
unset VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE
unset ASCEND_LAUNCH_BLOCKING
unset VLLM_FL_PLATFORM
unset VLLM_ASCEND_ENABLE_FUSED_MC2
unset VLLM_ASCEND_ENABLE_MC2_HIERARCHY_COMM


# ---------------------------------------------------------------------------
# vllm-plugin-FL / Ascend 配置
# ---------------------------------------------------------------------------

# 本容器同时装了 FL 插件（v024 分支）和镜像自带的 vllm-ascend，两者都注册
# vllm.platform_plugins。不指定的话 vLLM 会直接报
# "Only one platform plugin can be activated, but got: ['fl', 'ascend']"。
# 所以这里显式走 FL 插件路径，并补上 FL 插件需要的 VLLM_FL_VENDOR。
export VLLM_PLUGINS=fl
export VLLM_FL_VENDOR=ascend

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export TASK_QUEUE_ENABLE=1
export OMP_NUM_THREADS=10
export HCCL_BUFFSIZE=2048
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
VLLM_ASCEND_BALANCE_SCHEDULING=1
VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export USE_FLAGGEMS=0



export FL_DSV4_CPU_PREFILL_BOUNDS=0
export FL_DSV4_FUSED_MC2_ADAPTIVE_CAPACITY=0
export FL_DSV4_SKIP_UNUSED_MTP_STASH=0
export FL_DSV4_HC_RESIDUAL_ALIAS=0
export FL_DSV4_FUSED_SHARED_OVERLAP=0
export FL_DSV4_LARGE_BATCH_ALLTOALL=0
export FL_DSV4_DEFER_PROLOG_JOIN=0
export FL_DSV4_SHARE_PROLOG_QUANT=0
export FL_DSV4_EARLY_INDEXER_WEIGHTS=0
# Drop inherited patch switches and source/library overrides from old experiments.
for name in ${!FL_DSV4_@}; do unset "$name"; done
export FL_DSV4_CPU_PREFILL_BOUNDS=0
export FL_DSV4_FUSED_MC2_ADAPTIVE_CAPACITY=0
export FL_DSV4_SKIP_UNUSED_MTP_STASH=0
export FL_DSV4_HC_RESIDUAL_ALIAS=0
export FL_DSV4_FUSED_SHARED_OVERLAP=0
export FL_DSV4_LARGE_BATCH_ALLTOALL=0
export FL_DSV4_DEFER_PROLOG_JOIN=0
export FL_DSV4_SHARE_PROLOG_QUANT=0
export FL_DSV4_EARLY_INDEXER_WEIGHTS=0
for name in PYTHONPATH LD_LIBRARY_PATH ASCEND_CUSTOM_OPP_PATH; do
 value=$(python - "$name" <<'PY'
import os,sys
print(':'.join(x for x in os.environ.get(sys.argv[1],'').split(':') if x and 'vllm-plugin-FL' not in x))
PY
)
 export "$name=$value"
done
cd "$HERE"
SPEC_ARGS=()
if [[ "$VARIANT" != baseline ]]; then
 K=${VARIANT#mtp}
 SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$K,\"enforce_eager\":true,\"draft_sample_method\":\"greedy\",\"rejection_sample_method\":\"standard\"}")
fi
COMPILE_CONFIG=null
GRAPH=${GRAPH:-1}
[[ "$GRAPH" == 0 || "$GRAPH" == 1 ]] || exit 2
if [[ "$GRAPH" == 0 ]]; then
 COMPILE_ARGS=(--enforce-eager)
else
 COMPILE_CONFIG=$(python "$HERE/launch_config.py" compile "$VARIANT")
 COMPILE_ARGS=(--compilation-config "$COMPILE_CONFIG")
fi
ASYNC_SCHEDULING=$(python "$HERE/launch_config.py" async "$VARIANT" "${ASYNC_SCHEDULING:-auto}")
SCHED_ARGS=(--no-async-scheduling)
if [[ "$ASYNC_SCHEDULING" == 1 ]]; then SCHED_ARGS=(--async-scheduling); fi
ADDITIONAL_CONFIG=$(python "$HERE/launch_config.py" additional "$VARIANT")
python "$HERE/check_source.py"
export MTP_FUSED_GREEDY="${MTP_FUSED_GREEDY:-1}"
[[ "$MTP_FUSED_GREEDY" == 0 || "$MTP_FUSED_GREEDY" == 1 ]] || exit 2
export MSMONITOR_USE_DAEMON=0
PROFILE_ARGS=()
if [[ "${PROFILE:-0}" == 1 ]]; then
 PROFILE_ARGS=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$RUN_DIR/traces\",\"torch_profiler_with_stack\":false,\"ignore_frontend\":true,\"max_iterations\":16}")
fi
python -m pip freeze > "$RUN_DIR/environment.freeze.txt"
python "$HERE/check_source.py" > "$RUN_DIR/source_check.json"
cp "$HERE/manifest.json" "$RUN_DIR/source_manifest.json"
cp "$0" "$RUN_DIR/launch_script.sh"
cd "$RUN_DIR"
LOG="$RUN_DIR/server.log"
setsid nohup vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --served-model-name dsv4 \
    --api-server-count 1 \
    --data-parallel-size 4 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --max-model-len 66560 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.90 \
    --tokenizer-mode deepseek_v4 \
    --quantization ascend \
    --block-size 32 \
    --enable-chunked-prefill \
    "${SCHED_ARGS[@]}" \
    --no-enable-prefix-caching \
    --no-enable-log-requests \
    --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
    "${COMPILE_ARGS[@]}" \
    "${SPEC_ARGS[@]}" \
    --additional-config "$ADDITIONAL_CONFIG" \
    "${PROFILE_ARGS[@]}" > "$LOG" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$RUN_DIR/serve.pid"
python - "$PID" "$RUN_DIR/identity.json" "$VARIANT" "$PORT" "$GRAPH" "$ASYNC_SCHEDULING" "$COMPILE_CONFIG" <<'PY'
import hashlib,json,pathlib,sys,time,os
pid=int(sys.argv[1]);stamp=pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
path=pathlib.Path(sys.argv[2]); record={'pid':pid,'starttime':stamp}
path.write_text(json.dumps(record))
record.update(variant=sys.argv[3],port=int(sys.argv[4]),graph=int(sys.argv[5]),experiment='dsv4_mtp_infyuki_20261015',timestamp=time.time(),manifest_sha256=hashlib.sha256((path.parents[2]/'manifest.json').read_bytes()).hexdigest())
record.update(dsv4_mtp_fused_greedy=sys.argv[3].startswith('mtp') and os.getenv('MTP_FUSED_GREEDY','1')=='1',dsv4_mtp_fast_metadata=sys.argv[3].startswith('mtp') and os.getenv('MTP_FAST_METADATA','1')=='1',async_scheduling=sys.argv[6]=='1',compilation_config=json.loads(sys.argv[7]),enable_dsv4_mtp=sys.argv[3].startswith('mtp'),launch_script_sha256=hashlib.sha256((path.parents[2]/'serve_in_container.sh').read_bytes()).hexdigest(),launch_config_sha256=hashlib.sha256((path.parents[2]/'launch_config.py').read_bytes()).hexdigest())
path.with_name('launch_metadata.json').write_text(json.dumps(record,indent=2))
PY
echo "variant=$VARIANT graph=$GRAPH pid=$PID log=$LOG"
for ((i=0;i<360;i++)); do
 kill -0 "$PID" 2>/dev/null || { tail -n 60 "$LOG"; exit 1; }
 if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then echo "READY: $VARIANT at http://127.0.0.1:$PORT (you may now run the client)"; exit 0; fi
 sleep 5
done
echo 'Startup timeout; inspect log and recorded PID.' >&2
exit 1
