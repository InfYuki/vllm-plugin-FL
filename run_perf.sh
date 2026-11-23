#!/usr/bin/env bash
set -euo pipefail
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")"

n=${1:-16384}
mode=${2:-candidate}
case "$n" in
  1024|4096|16384) ;;
  *) echo 'Input length must be 1024, 4096 or 16384' >&2; exit 2 ;;
esac
case "$mode" in
  candidate) default_port=19690 ;;
  baseline)  default_port=19691 ;;
  *) echo 'Mode must be candidate or baseline' >&2; exit 2 ;;
esac
port=${PORT:-$default_port}
base_url=${BASE_URL:-http://127.0.0.1:$port}
base_url=${base_url%/}

export PYTHONPATH="$PWD:/vllm-workspace/vllm:/workspace/benchmark-deps/.venv/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
out="/workspace/results/tp1-kv-head-major-$mode-$n-$(date +%Y%m%d-%H%M%S)-$$"
curl -fsS "$base_url/v1/models" | python -c 'import json, sys; sys.exit(0 if any(m["id"] == "Unlimited-OCR" for m in json.load(sys.stdin)["data"]) else "Unlimited-OCR model was not found")'
mkdir -p "$out"
printf 'mode=%s endpoint=%s input=%s results=%s\n' "$mode" "$base_url" "$n" "$out"
python -m evalscope.cli.cli perf \
  --model Unlimited-OCR --url "$base_url/v1/chat/completions" --api openai --api-key EMPTY \
  --dataset random_vl --tokenizer-path /workspace/results/candidate-16384/tokenizer_export \
  --prefix-length 0 --apply-chat-template --min-prompt-length "$n" --max-prompt-length "$n" \
  --image-width 2048 --image-height 2048 --image-format RGB --image-num 1 \
  --parallel 64 --number "${NUMBER:-1024}" --no-test-connection --stream --seed 42 --temperature 0 \
  --min-tokens 1024 --max-tokens 1024 \
  --extra-args '{"ignore_eos":true,"stop":[],"stop_token_ids":[],"skip_special_tokens":false,"vllm_xargs":{"ngram_size":35,"window_size":128},"mm_processor_kwargs":{"image_size":640,"base_size":1024,"crop_mode":true}}' \
  --total-timeout 1800 --outputs-dir "$out" 2>&1 | tee "$out/client.log"
