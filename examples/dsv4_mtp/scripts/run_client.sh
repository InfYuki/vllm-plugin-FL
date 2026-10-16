#!/usr/bin/env bash
set -Eeuo pipefail
exec docker exec -i \
  -e PORT="${PORT:-1112}" -e READY_TIMEOUT="${READY_TIMEOUT:-1800}" \
  -e CC="${CC:-64}" -e OUT="${OUT:-1024}" -e REQ="${REQ:-256}" \
  -e INPUT_LENS="${INPUT_LENS:-1024 1024 4096}" \
  -e VLLM_PLUGINS=fl -e VLLM_FL_VENDOR=ascend -e USE_FLAGGEMS=0 \
  "${CONTAINER:-fl-infyuki-v024-hzj}" \
  bash /workspace/model_test/dsv4-mtp-infyuki-20261015/client_in_container.sh "$@"
