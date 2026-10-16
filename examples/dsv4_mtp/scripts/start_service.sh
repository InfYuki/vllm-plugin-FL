#!/usr/bin/env bash
set -Eeuo pipefail
exec docker exec \
  -e MTP_PREPARE_STEPS="${MTP_PREPARE_STEPS:-1}" \
  -e MTP_FUSED_INPUTS="${MTP_FUSED_INPUTS:-1}" \
  -e MTP_LOCAL_ARGMAX="${MTP_LOCAL_ARGMAX:-1}" \
  -e MTP_FUSED_GREEDY="${MTP_FUSED_GREEDY:-1}" -e MTP_FAST_METADATA="${MTP_FAST_METADATA:-1}" -e PORT="${PORT:-1112}" -e GRAPH="${GRAPH:-1}" -e ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-auto}" "${CONTAINER:-fl-infyuki-v024-hzj}" bash /workspace/model_test/dsv4-mtp-infyuki-20261015/serve_in_container.sh "${1:-mtp1}"
