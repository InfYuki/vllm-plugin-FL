# Qwen3.8-Flash-Next on Ascend A3

This branch extends FL PR #700 (0b85bb80f) with the official Ascend runtime bridge and the Qwen graph capture handling. The example contains the runtime changes for the accepted prefill MC2 and attention MMRS baseline and the optional GDN Q/K normalization fusion.

Use a dedicated A3 container with vLLM 5df63a7, an editable vLLM-Ascend checkout at 24487f4, CANN 9.2.1, and the BF16 model mounted at /models/Qwen3.8-Flash-Next. The scripts expect the FL checkout at /workspace/vllm-plugin-FL and the container's existing /workspace/env.sh for CANN and model setup. For the optional GDN candidate, install the A3 FLA 26.10.0 wheel into /workspace/gdn_chunk64_1123/python.

From this example directory in the container:

```bash
cp -r qsa_step prefill_mc2_1120 attn_mmrs_1120 gdn_chunk64_1123 gdn_norm_1123 /workspace/
cp check_runtime.py stop.sh /workspace/
python -B /workspace/qsa_step/manage.py apply
python -B /workspace/attn_mmrs_1120/prepare_runtime.py
bash /workspace/attn_mmrs_1120/start.sh --check
```

Start the accepted baseline with `bash /workspace/attn_mmrs_1120/start.sh`, or start the GDN candidate with `bash /workspace/gdn_norm_1123/start.sh`. Both use port 8098, TP8, DP2, EP16, BF16 and MTP disabled.
