#!/usr/bin/env bash
# 宿主机执行：bash test_vllm_bench.sh mtp2
set -euo pipefail
cd -- "$(dirname -- "$0")"

VARIANT=${1:-mtp2}   # 仅用于结果目录命名；请先自行启动对应服务
PORT=${PORT:-1112}
WARMUP_REQUESTS=${WARMUP_REQUESTS:-64}
CONTAINER=${CONTAINER:-fl-infyuki-v024-hzj}
MODEL=/public-flash/models/DeepSeek-V4-Flash-w8a8-mtp
OUT="vllm_bench_results/$VARIANT/$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$OUT"

# 固定 CC64、256 请求、输出1024，1K/4K各两轮，每轮正式计时前先预热64请求。
for ROUND in 1024_r1 1024_r2 4096_r1 4096_r2; do
    INPUT_LEN=${ROUND%_r*}
    echo "===== $VARIANT $ROUND: input=$INPUT_LEN output=1024 CC=64 ====="
    docker exec \
        -e no_proxy=localhost,127.0.0.1,::1 \
        -e NO_PROXY=localhost,127.0.0.1,::1 \
        "$CONTAINER" vllm bench serve \
        --backend vllm \
        --base-url "http://127.0.0.1:$PORT" \
        --endpoint /v1/completions \
        --model "$MODEL" \
        --tokenizer "$MODEL" \
        --served-model-name dsv4 \
        --dataset-name random \
        --random-input-len "$INPUT_LEN" \
        --random-output-len 1024 \
        --random-range-ratio 0 \
        --num-prompts 256 \
        --max-concurrency 64 \
        --request-rate inf \
        --seed 1024 \
        --temperature 0 --top-p 1 \
        --extra-body '{"seed":1024}' \
        --ignore-eos \
        --num-warmups "$WARMUP_REQUESTS" \
        --ready-check-timeout-sec 0 \
        --save-result --save-detailed \
        --result-dir "/workspace/model_test/dsv4-mtp-infyuki-20261015/$OUT" \
        --result-filename "$ROUND.json" \
        2>&1 | tee "$OUT/$ROUND.log"
done

echo "结果目录：$(pwd)/$OUT"
