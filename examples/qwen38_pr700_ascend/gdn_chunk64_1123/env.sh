#!/usr/bin/env bash
# Source after the workspace environment so its FLA paths cannot take precedence.
root=/workspace/gdn_chunk64_1123
vendor="$root/python/fla_npu/opp/vendors/fla_npu_transformer"
test -d "$vendor/op_api/lib" || { echo "Missing private FLA vendor: $vendor" >&2; return 1; }
# Remove earlier FLA registrations while keeping other CANN vendors in their original order.
replace_fla_vendor() {
    local variable=$1 entry resolved joined=$2
    local -a entries
    IFS=: read -ra entries <<< "${!variable:-}"
    for entry in "${entries[@]}"; do
        [[ -n "$entry" ]] || continue
        resolved=$(readlink -m -- "$entry")
        case "$resolved" in
            */vendors/fla_npu_transformer|*/vendors/fla_npu_transformer/*) continue ;;
        esac
        joined="$joined:$entry"
    done
    export "$variable=$joined"
}
replace_fla_vendor ASCEND_CUSTOM_OPP_PATH "$vendor"
replace_fla_vendor LD_LIBRARY_PATH "$vendor/op_api/lib"
unset -f replace_fla_vendor
export FLA_NPU_DISABLE_PTH=1
export PYTHONPATH="$root/python:/workspace/attn_mmrs_1120/runtime${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_CACHE_ROOT="$root/cache"
export VLLM_FL_QWEN38_ATTN_MMRS=1
