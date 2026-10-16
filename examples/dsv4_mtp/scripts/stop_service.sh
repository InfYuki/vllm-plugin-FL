#!/usr/bin/env bash
set -Eeuo pipefail
exec docker exec "${CONTAINER:-fl-infyuki-v024-hzj}" python /workspace/model_test/dsv4-mtp-infyuki-20261015/stop_service.py "${1:-mtp1}"
