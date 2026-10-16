#!/usr/bin/env bash
set -Eeuo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
VARIANT=${1:-mtp1}
MODE=${2:-performance}
case "$VARIANT" in baseline|mtp1|mtp2|mtp3) ;; *) echo 'Usage: bash run_client.sh baseline|mtp1|mtp2|mtp3 [correctness|performance|correctness_full]'; exit 2;; esac
PORT=${PORT:-1112}
LOG="$HERE/runs/$VARIANT/server.log"
MODEL=/public-flash/models/DeepSeek-V4-Flash-w8a8-mtp/
BASE="http://127.0.0.1:$PORT"
case "$MODE" in correctness|performance|correctness_full) ;; *) echo "Unknown mode: $MODE"; exit 2;; esac
# Check the recorded service to avoid putting a baseline label on an MTP run.
python - "$HERE" "$VARIANT" "$PORT" <<'PY'
import hashlib,json,pathlib,sys
here=pathlib.Path(sys.argv[1]); variant=sys.argv[2]
p=here/'runs'/variant/'launch_metadata.json'
if not p.exists(): raise SystemExit(f'No launch record: {p}; run start_service.sh {variant} first.')
m=json.loads(p.read_text())
if m['manifest_sha256']!=hashlib.sha256((here/'manifest.json').read_bytes()).hexdigest(): raise SystemExit('Running service uses the previous source manifest; restart this variant before testing.')
if m['variant']!=variant or m['port']!=int(sys.argv[3]): raise SystemExit('Variant/PORT differs from launch record.')
proc=pathlib.Path(f"/proc/{m['pid']}/stat")
if not proc.exists(): raise SystemExit('Recorded service has exited; inspect '+str(p.with_name('server.log')))
s=proc.read_text().rsplit(')',1)[1].split()
if s[19]!=m['starttime'] or s[0]=='Z': raise SystemExit('Recorded service has exited; inspect '+str(p.with_name('server.log')))
PY
echo "Waiting for $VARIANT: $BASE/v1/models"
deadline=$((SECONDS + ${READY_TIMEOUT:-1800}))
until curl -fsS --connect-timeout 2 --max-time 5 "$BASE/v1/models" >/dev/null 2>&1; do
  if (( SECONDS >= deadline )); then
    echo "ERROR: port $PORT is not ready. See $LOG" >&2
    tail -n 40 "$LOG" >&2 || true
    exit 1
  fi
  echo "Service is still starting; log: $LOG"
  sleep 5
done
echo "Service is ready. variant=$VARIANT port=$PORT"
if [[ "$MODE" == correctness_full ]]; then
  exec python "$HERE/test_client.py" "$VARIANT" correctness "${@:3}"
fi
RUN_DIR="$HERE/results/$VARIANT/$(date +%Y%m%d-%H%M%S)-$MODE-$$"
mkdir -p "$RUN_DIR"
cp "$HERE/runs/$VARIANT/launch_metadata.json" "$RUN_DIR/"
# Metrics help inspect MTP acceptance; collection is optional.
curl -fsS --max-time 10 "$BASE/metrics" > "$RUN_DIR/before.prom" || true
if [[ "$MODE" == correctness ]]; then
  echo 'Smoke check only: three short requests. This is not token-by-token correctness validation.'
  for prompt in 'Answer briefly: what is 2 plus 2?' '请简要介绍一下你自己。' 'Continue the sequence: 1, 2, 3, 4,'; do
    echo "===== $prompt ====="
    payload=$(python - "$prompt" <<'PY'
import json,sys
print(json.dumps(dict(model='dsv4',prompt=sys.argv[1],max_tokens=128,temperature=0,seed=1024)))
PY
)
    curl --fail-with-body -sS --connect-timeout 5 --max-time 600 "$BASE/v1/completions" \
      -H 'Content-Type: application/json' -d "$payload" | tee -a "$RUN_DIR/responses.jsonl"
    printf '\n' | tee -a "$RUN_DIR/responses.jsonl"
  done
else
  CC=${CC:-64}; OUT=${OUT:-1024}; REQ=${REQ:-256}
  read -r -a LENGTHS <<< "${INPUT_LENS:-1024 1024 4096}"
  echo "Benchmark: CC=$CC OUT=$OUT REQ=$REQ input lengths=${LENGTHS[*]}"
  echo "Logs: $RUN_DIR"
  round=0
  for length in "${LENGTHS[@]}"; do
    round=$((round + 1))
    tag="${length}-run${round}"
    echo "===== input=$length / output=$OUT / Run $round : $(date -Is) ====="
    vllm bench serve \
      --backend vllm \
      --model "$MODEL" \
      --served-model-name dsv4 \
      --tokenizer "$MODEL" \
      --tokenizer-mode deepseek_v4 \
      --endpoint /v1/completions \
      --host 127.0.0.1 \
      --port "$PORT" \
      --dataset-name random \
      --temperature 0 \
      --seed 1024 \
      --random-input-len "$length" \
      --random-output-len "$OUT" \
      --num-prompts "$REQ" \
      --max-concurrency "$CC" \
      --ignore-eos \
      --save-result --result-dir "$RUN_DIR" --result-filename "$tag.json" \
      2>&1 | tee "$RUN_DIR/${tag}_bench.log"
  done
fi
curl -fsS --max-time 10 "$BASE/metrics" > "$RUN_DIR/after.prom" || true
echo "Finished: $(date -Is). Results: $RUN_DIR"
