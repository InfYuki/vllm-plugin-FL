#!/usr/bin/env bash
set -euo pipefail
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")"

n=${1:-4096}
mode=${2:-candidate}
usage='Usage: bash run_perf.sh [1024|4096|16384] [candidate|baseline]'
if (( $# > 2 )); then echo "$usage" >&2; exit 2; fi
case "$n" in
  1024|4096|16384) ;;
  *) echo "$usage" >&2; exit 2 ;;
esac
mask_mib=8192
case "$mode" in
  candidate) default_port=19743; budget=6144 ;;
  baseline)  default_port=19744; budget=4096 ;;
  *) echo "$usage" >&2; exit 2 ;;
esac
port=${PORT:-$default_port}
base_url=${BASE_URL:-http://127.0.0.1:$port}
base_url=${base_url%/}

export PYTHONPATH="$PWD:/vllm-workspace/vllm:/workspace/benchmark-deps/.venv/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
out="/workspace/results/tp1-prefill6144-$mode-input-$n-$(date +%Y%m%d-%H%M%S)-$$"
curl -fsS "$base_url/v1/models" | python -c 'import json, sys; sys.exit(0 if any(m["id"] == "Unlimited-OCR" for m in json.load(sys.stdin)["data"]) else "Unlimited-OCR model was not found")'
mkdir -p "$out"
snapshot() {
  curl --max-time 5 -fsS "$base_url/metrics" > "$out/metrics-$1.prom" || echo "Metrics unavailable ($1)" >&2
}
snapshot before
# Keep each visible physical chip separate so a client cannot guess the server device.
sample_hbm() {
  local stamp
  while :; do
    stamp=$(date +%s)
    timeout 5 npu-smi info | awk -F'|' -v t="$stamp" '
      $3 ~ /[[:xdigit:]][[:xdigit:]]:[[:xdigit:]][[:xdigit:]]:/ {
        gsub(/\//, " / ", $4)
        split($2, chip, " "); split($4, usage, " ")
        if (chip[2] ~ /^[0-9]+$/ && usage[5] ~ /^[0-9]+$/ && usage[6] == "/")
          print t, chip[2], usage[5], usage[7]
      }' || return
    sleep 2
  done
}
sampler=
if command -v npu-smi >/dev/null; then
  npu-smi info -m > "$out/device-map.txt" || true
  sample_hbm > "$out/hbm-samples.txt" 2>"$out/hbm-sampler.log" &
  sampler=$!
fi
finish() {
  if [[ -n $sampler ]]; then
    kill "$sampler" 2>/dev/null || true
    wait "$sampler" 2>/dev/null || true
    awk '{if ($3 > peak[$2]) peak[$2]=$3} END {for (id in peak) print id, peak[id]}' \
      "$out/hbm-samples.txt" | sort -n > "$out/hbm-sampled-peak-physical-id-mib.txt"
    [[ -s $out/hbm-samples.txt ]] || echo "HBM samples unavailable" >&2
  fi
  snapshot after
}
trap finish EXIT
printf 'mode=%s configured_budget=%s mask_MiB=%s endpoint=%s input=%s results=%s\n' "$mode" "$budget" "$mask_mib" "$base_url" "$n" "$out"
python -m evalscope.cli.cli perf \
  --model Unlimited-OCR --url "$base_url/v1/chat/completions" --api openai --api-key EMPTY \
  --dataset random_vl --tokenizer-path /workspace/results/candidate-16384/tokenizer_export \
  --prefix-length 0 --apply-chat-template --min-prompt-length "$n" --max-prompt-length "$n" \
  --image-width 2048 --image-height 2048 --image-format RGB --image-num 1 \
  --parallel 64 --number "${NUMBER:-1024}" --no-test-connection --stream --seed 42 --temperature 0 \
  --min-tokens 1024 --max-tokens 1024 \
  --extra-args '{"ignore_eos":true,"stop":[],"stop_token_ids":[],"skip_special_tokens":false,"vllm_xargs":{"ngram_size":35,"window_size":128},"mm_processor_kwargs":{"image_size":640,"base_size":1024,"crop_mode":true}}' \
  --total-timeout 1800 --outputs-dir "$out" 2>&1 | tee "$out/client.log"
