#!/usr/bin/env bash
set -euo pipefail
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")"

mode=${1:-candidate}
usage='Usage: bash run_server.sh [candidate|baseline] (both use a 2048-token budget)'
if (( $# > 1 )); then echo "$usage" >&2; exit 2; fi
case "$mode" in
  candidate) block_size=256; default_port=19700 ;;
  baseline)  block_size=128; default_port=19701 ;;
  *) echo "$usage" >&2; exit 2 ;;
esac
port=${PORT:-$default_port}
devices=${DEVICES:-0}
seqs=64
budget=2048
mask_mib=4096

export PYTHONPATH="$PWD:/vllm-workspace/vllm:/workspace/benchmark-deps/.venv/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export USE_FLAGGEMS=0 VLLM_FL_VENDOR=ascend VLLM_PLUGINS=fl VLLM_FL_PREFER_ENABLED=1 VLLM_FL_STRICT=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_FL_RSWA_GROUPED_MASK=1 VLLM_FL_RSWA_CAUSAL_PREFILL=1
export VLLM_FL_OCR_PREFILL_MOE_TOP6=1 VLLM_FL_OCR_HEAD_MAJOR_KV=1
export VLLM_FL_OCR_EXPERT_PREPACK=1

python -P - <<'PY'
from importlib.util import find_spec
from pathlib import Path

spec = find_spec("vllm_fl")
expected = Path.cwd() / "vllm_fl" / "__init__.py"
if spec is None or not spec.origin or Path(spec.origin).resolve() != expected:
    raise SystemExit(f"Expected FL at {expected}. Python resolved {spec!r}")
PY
printf 'FL=%s mode=%s TP=1 device=%s port=%s seqs=%s max_num_batched_tokens=%s mask_MiB=%s causal_prefill=1 moe_top6=1 head_major_kv=1 expert_prepack=1 block_size=%s memory=0.95\n' \
  "$PWD" "$mode" "$devices" "$port" "$seqs" "$budget" "$mask_mib" "$block_size"
exec python examples/unlimited_ocr/serve.py /public-flash/models/q4_models/Unlimited-OCR \
  --devices "$devices" --tp 1 --block-size "$block_size" --port "$port" \
  --max-num-seqs "$seqs" --max-num-batched-tokens "$budget" --rswa-mask-mib "$mask_mib" \
  --gpu-memory-utilization 0.95
