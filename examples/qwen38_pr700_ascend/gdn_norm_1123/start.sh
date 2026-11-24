#!/usr/bin/env bash
set -eo pipefail
base=/workspace/prefill_mc2_1120/start.sh
# Inject the isolated environment immediately after the accepted launcher initializes CANN.
test "$(grep -xc 'source /workspace/env.sh' "$base")" = 1
root=/workspace/gdn_norm_1123
# Source this launcher's environment branch after CANN setup in the accepted launcher.
if [[ ${1:-} == --environment && ${BASH_SOURCE[0]} != "$0" ]]; then
    source /workspace/gdn_chunk64_1123/env.sh
    export PYTHONPATH="/workspace/gdn_norm_1123/runtime:$PYTHONPATH"
    export VLLM_CACHE_ROOT=/workspace/gdn_norm_1123/cache
    python -B - <<'PY'
from importlib.metadata import version
from pathlib import Path

import fla_npu
import vllm_ascend.ops  # Register the production ops before resolving the adaptor.
from vllm_ascend.device.device_op import DeviceOperator

assert version("flash-linear-attention-npu-a3").split("+")[0] == "26.10.0"
assert Path(fla_npu.__file__).resolve().is_relative_to("/workspace/gdn_chunk64_1123/python")
method = Path(DeviceOperator.fla_gdn_prefill.__code__.co_filename)
assert method.resolve() == Path("/workspace/gdn_norm_1123/runtime/vllm_ascend/device/device_op.py")
PY
    return
fi
python -B "$root/prepare_runtime.py"
sed '/^source \/workspace\/env.sh$/a source /workspace/gdn_norm_1123/start.sh --environment' "$base" | bash -s -- "$@"
