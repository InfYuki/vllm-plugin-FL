#!/usr/bin/env bash
set -euo pipefail
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")"

case "${1:-2048}" in
  2048|candidate) budget=2048; default_port=19692 ;;
  1024|baseline)  budget=1024; default_port=19693 ;;
  *) echo 'Usage: bash run_server.sh [2048|1024]' >&2; exit 2 ;;
esac
port=${PORT:-$default_port}
devices=${DEVICES:-0}
seqs=64
mask_mib=4096

export PYTHONPATH="$PWD:/vllm-workspace/vllm:/workspace/benchmark-deps/.venv/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export USE_FLAGGEMS=0 VLLM_FL_VENDOR=ascend VLLM_PLUGINS=fl VLLM_FL_PREFER_ENABLED=1 VLLM_FL_STRICT=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_FL_RSWA_GROUPED_MASK=1 VLLM_FL_RSWA_CAUSAL_PREFILL=1
export VLLM_FL_OCR_PREFILL_MOE_TOP6=1 VLLM_FL_OCR_HEAD_MAJOR_KV=1

python -P - <<'PY'
from importlib.util import find_spec
from pathlib import Path

spec = find_spec("vllm_fl")
expected = Path.cwd() / "vllm_fl" / "__init__.py"
if spec is None or not spec.origin or Path(spec.origin).resolve() != expected:
    raise SystemExit(f"Expected FL at {expected}. Python resolved {spec!r}")
PY
printf 'FL=%s TP=1 device=%s port=%s seqs=%s max_num_batched_tokens=%s mask_MiB=%s causal_prefill=1 moe_top6=1 head_major_kv=1 memory=0.95\n' \
  "$PWD" "$devices" "$port" "$seqs" "$budget" "$mask_mib"
exec python examples/unlimited_ocr/serve.py /public-flash/models/q4_models/Unlimited-OCR \
  --devices "$devices" --tp 1 --port "$port" \
  --max-num-seqs "$seqs" --max-num-batched-tokens "$budget" --rswa-mask-mib "$mask_mib" \
  --gpu-memory-utilization 0.95
