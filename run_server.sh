#!/usr/bin/env bash
set -euo pipefail
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")"

mode=${1:-candidate}
usage='Usage: bash run_server.sh [candidate|baseline]'
if (( $# > 1 )); then echo "$usage" >&2; exit 2; fi
mask_mib=8192
case "$mode" in
  candidate) default_port=19751; budget=12288 ;;
  baseline)  default_port=19752; budget=12288 ;;
  *) echo "$usage" >&2; exit 2 ;;
esac
export VLLM_FL_OCR_DIRECT_ROUTE=1
export VLLM_FL_OCR_ASYNC=0
if [[ $mode == candidate ]]; then export VLLM_FL_OCR_ASYNC=1; fi
port=${PORT:-$default_port}
devices=${DEVICES:-0}
seqs=64
block_size=256

export PYTHONPATH="$PWD:/vllm-workspace/vllm:/workspace/benchmark-deps/.venv/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export USE_FLAGGEMS=0 VLLM_FL_VENDOR=ascend VLLM_PLUGINS=fl VLLM_FL_PREFER_ENABLED=1 VLLM_FL_STRICT=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_FL_RSWA_GROUPED_MASK=1 VLLM_FL_RSWA_CAUSAL_PREFILL=1
export VLLM_FL_OCR_PREFILL_MOE_TOP6=1 VLLM_FL_OCR_HEAD_MAJOR_KV=1
export VLLM_FL_OCR_EXPERT_PREPACK=1
export VLLM_FL_OCR_EXTENDED_TOP6=1
export VLLM_FL_OCR_TOP6_4096=1
export VLLM_FL_OCR_SKIP_HIDDEN_PAGES=1
export VLLM_FL_OCR_WIDE_PREFILL=1
export VLLM_FL_OCR_RSWA_VALID_ROWS=1

python -P - <<'PY'
from importlib.util import find_spec
from pathlib import Path

spec = find_spec("vllm_fl")
expected = Path.cwd() / "vllm_fl" / "__init__.py"
if spec is None or not spec.origin or Path(spec.origin).resolve() != expected:
    raise SystemExit(f"Expected FL at {expected}. Python resolved {spec!r}")
PY
printf 'FL=%s mode=%s TP=1 device=%s port=%s seqs=%s budget=%s mask_MiB=%s block_size=%s skip_hidden_pages=%s extended_top6=1 rswa_valid_rows=1 sam_rel_pos=0 memory=0.95\n' \
  "$PWD" "$mode" "$devices" "$port" "$seqs" "$budget" "$mask_mib" "$block_size" "$VLLM_FL_OCR_SKIP_HIDDEN_PAGES"
# Set before Python imports prometheus_client so worker counters reach /metrics.
metrics_dir=$(mktemp -d)
export PROMETHEUS_MULTIPROC_DIR="$metrics_dir"
server_pid=
stop_server() {
  if [[ -n $server_pid ]]; then
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" || true
  fi
  rm -f -- "$metrics_dir"/*.db
  rmdir -- "$metrics_dir"
}
trap stop_server EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
python examples/unlimited_ocr/serve.py /public-flash/models/q4_models/Unlimited-OCR \
  --devices "$devices" --tp 1 --block-size "$block_size" --port "$port" \
  --max-num-seqs "$seqs" --max-num-batched-tokens "$budget" --rswa-mask-mib "$mask_mib" \
  --gpu-memory-utilization 0.95 &
server_pid=$!
wait "$server_pid"
server_pid=
