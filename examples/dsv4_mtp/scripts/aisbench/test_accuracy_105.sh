#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR=/workspace/model_test/dsv4-mtp-infyuki-20261015/aisbench
if [[ ! -f /.dockerenv ]]; then
  exec docker exec -i \
    -e HOST_IP="${HOST_IP:-10.55.0.166}" -e PORT="${PORT:-11223}" \
    -e MODEL_NAME="${MODEL_NAME:-dsv4}" \
    -e MODEL_PATH="${MODEL_PATH:-/public-flash/models/DeepSeek-V4-Flash-w8a8-mtp}" \
    -e ACC_NUM="${ACC_NUM:-1319}" -e ACC_CC="${ACC_CC:-8}" \
    -e ACC_MAX_OUT="${ACC_MAX_OUT:-8192}" -e SEED="${SEED:-1024}" \
    -e API_KEY="${API_KEY:-}" \
    "${CONTAINER:-fl-infyuki-v024-hzj}" bash "$TASK_DIR/test_accuracy_105.sh" "$@"
fi
if (( $# > 1 )) || { (( $# == 1 )) && [[ "$1" != --check-only ]]; }; then
  echo 'Usage: bash test_accuracy_105.sh [--check-only]' >&2
  exit 2
fi
export HOST_IP="${HOST_IP:-10.55.0.166}" PORT="${PORT:-11223}"
export MODEL_NAME="${MODEL_NAME:-dsv4}"
export MODEL_PATH="${MODEL_PATH:-/public-flash/models/DeepSeek-V4-Flash-w8a8-mtp}"
export ACC_NUM="${ACC_NUM:-1319}" ACC_CC="${ACC_CC:-8}"
export ACC_MAX_OUT="${ACC_MAX_OUT:-8192}" SEED="${SEED:-1024}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export NO_PROXY="${NO_PROXY:-},localhost,127.0.0.1,${HOST_IP}"
export no_proxy="${no_proxy:-},localhost,127.0.0.1,${HOST_IP}"
PY=/opt/aisbench-venv/bin/python
AIS=/opt/aisbench-venv/bin/ais_bench
[[ -x "$PY" && -x "$AIS" ]] || { echo 'AISBench environment is missing' >&2; exit 1; }
mkdir -p "$HERE/results"
OUT="$(mktemp -d "$HERE/results/remote105-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
exec > >(tee "$OUT/run.log") 2>&1
echo "Target: http://${HOST_IP}:${PORT} model=${MODEL_NAME}"
echo "GSM8K: questions=${ACC_NUM} concurrency=${ACC_CC} max_output=${ACC_MAX_OUT}"
echo "Results: $OUT"
"$PY" "$HERE/make_config.py" accuracy "$OUT/config.py"
"$PY" "$HERE/check_remote_105.py" "$OUT"
if [[ "${1:-}" == --check-only ]]; then
  echo 'Preflight passed. No inference requests were sent.'
  exit 0
fi
cd "$HERE"
"$AIS" "$OUT/config.py" --mode all --num-prompts "$ACC_NUM" \
  --num-warmups 0 --max-num-workers 1 --debug --dump-eval-details -w "$OUT/work"
"$PY" "$HERE/check_accuracy_result.py" "$OUT" "$ACC_NUM"
echo "Accuracy summary: $OUT/accuracy_summary.json"
