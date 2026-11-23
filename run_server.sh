#!/usr/bin/env bash
set -euo pipefail
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")"

mode=${1:-candidate}
case "$mode" in
  candidate) head_major=1; default_port=19690 ;;
  baseline)  head_major=0; default_port=19691 ;;
  *) echo 'Usage: bash run_server.sh [candidate|baseline]' >&2; exit 2 ;;
esac
port=${PORT:-$default_port}
devices=${DEVICES:-0}
seqs=${MAX_NUM_SEQS:-64}
budget=${MAX_NUM_BATCHED_TOKENS:-1024}
mask_mib=${RSWA_MASK_MAX_MIB:-2048}

export PYTHONPATH="$PWD:/vllm-workspace/vllm:/workspace/benchmark-deps/.venv/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export USE_FLAGGEMS=0 VLLM_FL_VENDOR=ascend VLLM_PLUGINS=fl VLLM_FL_PREFER_ENABLED=1 VLLM_FL_STRICT=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_FL_RSWA_CAUSAL_PREFILL=1 VLLM_FL_OCR_PREFILL_MOE_TOP6=1
export VLLM_FL_OCR_HEAD_MAJOR_KV=$head_major

python -P - <<'PY'
from importlib.util import find_spec
from pathlib import Path

spec = find_spec("vllm_fl")
expected = Path.cwd() / "vllm_fl" / "__init__.py"
if spec is None or not spec.origin or Path(spec.origin).resolve() != expected:
    raise SystemExit(f"Expected FL at {expected}. Python resolved {spec!r}")
PY
printf 'mode=%s FL=%s TP=1 device=%s port=%s seqs=%s budget=%s mask_MiB=%s causal_prefill=1 moe_top6=1 head_major_kv=%s memory=0.95\n' \
  "$mode" "$PWD" "$devices" "$port" "$seqs" "$budget" "$mask_mib" "$head_major"
exec python examples/unlimited_ocr/serve.py /public-flash/models/q4_models/Unlimited-OCR \
  --devices "$devices" --tp 1 --port "$port" \
  --max-num-seqs "$seqs" --max-num-batched-tokens "$budget" --rswa-mask-mib "$mask_mib" \
  --gpu-memory-utilization 0.95
