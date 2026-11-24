#!/usr/bin/env bash
# Launch the accepted MC2 service with the separate attention projection implementation.
set -eo pipefail
candidate=/workspace/attn_mmrs_1120
test -r "$candidate/runtime/vllm_ascend/models/qwen4_exp/attention_mmrs.py" || {
    echo "Missing candidate runtime; run python -B $candidate/prepare_runtime.py" >&2
    exit 1
}
export PYTHONPATH="$candidate/runtime${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_FL_QWEN38_ATTN_MMRS=1
# Keep FX containing the new custom op separate from the accepted MC2 graphs.
export VLLM_CACHE_ROOT="$candidate/cache"
echo "Qwen BF16 attention MMRS candidate. Real eager prefill >=1024 rows, TP8, ordinary prefill MC2."
exec bash /workspace/prefill_mc2_1120/start.sh "$@"
