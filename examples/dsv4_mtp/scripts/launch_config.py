"""Launch settings only. Imports no FL, torch, vLLM or device runtime."""
import json
import math
import os
import sys

VARIANTS = ('baseline', 'mtp1', 'mtp2', 'mtp3')


def compilation_config(variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    result = {'cudagraph_mode': 'FULL_DECODE_ONLY'}
    if variant == 'mtp2':
        # Target verification has K+1=3 query tokens; FlashComm1 requires TP4
        # token alignment. Capture only LCM(3,4)=12 multiples, through 64 reqs.
        quantum = math.lcm(3, 4)
        # Keep every TP4/three-query aligned extent. In particular 60 avoids
        # the previous 48 -> 72 graph jump at 17..20 requests per DP engine.
        sizes = list(range(quantum, 64 * 3 + 1, quantum))
        result.update(cudagraph_capture_sizes=sizes,
                      max_cudagraph_capture_size=sizes[-1])
    return result


def additional_config(variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    flag = os.getenv('MTP_FAST_METADATA', '1')
    if variant.startswith('mtp') and flag not in ('0', '1'):
        raise ValueError('MTP_FAST_METADATA must be 0 or 1')
    result = dict(enable_dsv4_mtp=variant.startswith('mtp'),
                enable_fused_mc2=True, enable_flashcomm1=True,
                enable_shared_expert_dp=True, enable_mlapo=True,
                enable_balance_scheduling=True,
                ascend_compilation_config=dict(enable_npugraph_ex=True,
                                               enable_static_kernel=False),
                enable_cpu_binding=True, multistream_overlap_shared_expert=False)

    if variant.startswith('mtp'):
        result['dsv4_mtp_fast_metadata'] = flag == '1'
    return result

def async_scheduling(variant, override='auto'):
    if variant not in VARIANTS:
        raise ValueError(variant)
    if override not in ('auto', '0', '1'):
        raise ValueError('ASYNC_SCHEDULING must be auto, 0 or 1')
    enabled = True if override == 'auto' else override == '1'
    return enabled


if __name__ == '__main__':
    mode, variant = sys.argv[1:3]
    if mode == 'compile':
        print(json.dumps(compilation_config(variant)))
    elif mode == 'additional':
        print(json.dumps(additional_config(variant)))
    elif mode == 'async':
        print(int(async_scheduling(variant, sys.argv[3] if len(sys.argv)>3 else 'auto')))
    else:
        raise SystemExit('Unknown configuration mode')
