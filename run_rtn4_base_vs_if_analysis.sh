#!/usr/bin/env bash
set -euo pipefail

python -m quant_fp_phase1.scripts.run_rtn4_base_vs_if_analysis \
  --base-model "${BASE_MODEL:-NousResearch/Llama-2-7b-hf}" \
  --if-model "${IF_MODEL:-cnut1648/LLaMA2-7B-fingerprinted-SFT}" \
  --fingerprint-data "${FINGERPRINT_DATA:-Model-Fingerprint/dataset/llama_fingerprint_chat}" \
  --output-dir "${OUTPUT_DIR:-quant_fp_phase1/rtn4_base_vs_if_analysis}" \
  --group-size "${GROUP_SIZE:-128}" \
  --dtype "${DTYPE:-bf16}" \
  --device-map "${DEVICE_MAP:-auto}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-30}"
