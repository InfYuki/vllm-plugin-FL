#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VARIANT=${1:-baseline}
MODE=${2:-performance}
case "$VARIANT" in baseline|mtp1|mtp2|mtp3) ;; *) echo 'variant: baseline|mtp1|mtp2|mtp3' >&2; exit 2;; esac
case "$MODE" in accuracy|performance) ;; *) echo 'mode: accuracy|performance' >&2; exit 2;; esac
CONTAINER=${CONTAINER:-fl-infyuki-v024-hzj}
# This directory is bind-mounted at both paths. Allow execution on host or in container.
if [[ ! -x /opt/aisbench-venv/bin/ais_bench ]]; then
  TARGET=${CONTAINER_DIR:-/workspace/model_test/dsv4-mtp-infyuki-20261015/aisbench}
  exec docker exec -i \
    -e PORT="${PORT:-1112}" -e HOST_IP="${HOST_IP:-127.0.0.1}" \
    -e MODEL_NAME="${MODEL_NAME:-dsv4}" \
    -e MODEL_PATH="${MODEL_PATH:-/public-flash/models/DeepSeek-V4-Flash-w8a8-mtp}" \
    -e CC="${CC:-64}" -e NUM_REQUESTS="${NUM_REQUESTS:-256}" \
    -e OUTPUT_LEN="${OUTPUT_LEN:-1024}" -e SEED="${SEED:-1024}" \
    -e ACC_CC="${ACC_CC:-8}" -e ACC_MAX_OUT="${ACC_MAX_OUT:-8192}" \
    -e ACC_NUM="${ACC_NUM:-1319}" -e WAIT_SECONDS="${WAIT_SECONDS:-300}" \
    -e API_KEY="${API_KEY:-}" \
    "$CONTAINER" bash "$TARGET/run.sh" "$VARIANT" "$MODE"
fi
export PORT=${PORT:-1112} HOST_IP=${HOST_IP:-127.0.0.1} MODEL_NAME=${MODEL_NAME:-dsv4}
export MODEL_PATH=${MODEL_PATH:-/public-flash/models/DeepSeek-V4-Flash-w8a8-mtp}
export CC=${CC:-64} NUM_REQUESTS=${NUM_REQUESTS:-256} OUTPUT_LEN=${OUTPUT_LEN:-1024} SEED=${SEED:-1024}
export ACC_CC=${ACC_CC:-8} ACC_MAX_OUT=${ACC_MAX_OUT:-8192} ACC_NUM=${ACC_NUM:-1319}
export WAIT_SECONDS=${WAIT_SECONDS:-300} API_KEY=${API_KEY:-}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}localhost,127.0.0.1,::1,$HOST_IP"
export no_proxy="$NO_PROXY"
PY=/opt/aisbench-venv/bin/python
AIS=/opt/aisbench-venv/bin/ais_bench
OUT="$HERE/results/$VARIANT/$(date +%Y%m%d-%H%M%S)-$MODE-$$"
mkdir -p "$OUT"
exec > >(tee "$OUT/console.log") 2>&1
echo "variant=$VARIANT mode=$MODE endpoint=http://$HOST_IP:$PORT results=$OUT"
"$PY" "$HERE/check_service.py" "$VARIANT" "$OUT"
if [[ "$MODE" == accuracy ]]; then
  "$PY" "$HERE/make_config.py" accuracy "$OUT/config.py"
  printf '%s\n' "$AIS $OUT/config.py --mode all --num-prompts $ACC_NUM --num-warmups 0 --max-num-workers 1 --debug --dump-eval-details -w $OUT/work" > "$OUT/command.txt"
  "$AIS" "$OUT/config.py" --mode all --num-prompts "$ACC_NUM" --num-warmups 0 \
    --max-num-workers 1 --debug --dump-eval-details -w "$OUT/work"
  "$PY" "$HERE/check_accuracy_result.py" "$OUT" "$ACC_NUM"
else
  # Prepare data on CPU only. Both rounds and all variants reuse identical prompts.
  "$PY" "$HERE/prepare_perf_data.py"
  for spec in 1024:1 1024:2 4096:1 4096:2; do
    INPUT_LEN=${spec%:*}; ROUND=${spec#*:}
    export INPUT_LEN
    ROUND_OUT="$OUT/${INPUT_LEN}_r$ROUND"
    mkdir -p "$ROUND_OUT"
    echo "===== input=$INPUT_LEN output=$OUTPUT_LEN concurrency=$CC requests=$NUM_REQUESTS round=$ROUND/2 ====="
    "$PY" "$HERE/make_config.py" performance "$ROUND_OUT/config.py"
    printf '%s\n' "$AIS $ROUND_OUT/config.py --mode perf --num-prompts $NUM_REQUESTS --num-warmups 0 --max-num-workers 1 --debug -w $ROUND_OUT/work" > "$ROUND_OUT/command.txt"
    "$AIS" "$ROUND_OUT/config.py" --mode perf --num-prompts "$NUM_REQUESTS" --num-warmups 0 \
      --max-num-workers 1 --debug -w "$ROUND_OUT/work" 2>&1 | tee "$ROUND_OUT/console.log"
    "$PY" "$HERE/check_perf_result.py" "$ROUND_OUT" "$NUM_REQUESTS" "$INPUT_LEN" "$OUTPUT_LEN"
  done
  "$PY" "$HERE/summarize_performance.py" "$OUT"
fi
echo "Completed: $OUT"
