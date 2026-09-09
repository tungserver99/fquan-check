#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-NousResearch/Llama-2-7b-hf}"
IF_MODEL="${IF_MODEL:-cnut1648/LLaMA2-7B-fingerprinted-SFT}"
FINGERPRINT_DATA="${FINGERPRINT_DATA:-Model-Fingerprint/dataset/llama_fingerprint_chat}"
OUTPUT_DIR="${OUTPUT_DIR:-quant_fp_phase1}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
CACHE_DIR="${CACHE_DIR:-./dataset_cache}"
GROUP_SIZE="${GROUP_SIZE:-128}"
MARGIN_BITS="${MARGIN_BITS:-4 3}"
DELTA_BITS="${DELTA_BITS:-3 4}"
RUN_PLOTS="${RUN_PLOTS:-0}"

read -r -a margin_bits <<< "$MARGIN_BITS"
read -r -a delta_bits <<< "$DELTA_BITS"

stages=(margin delta)
if [[ "$RUN_PLOTS" == "1" ]]; then
  stages+=(plots)
fi

python -m quant_fp_phase1.scripts.run_phase1 \
  --base-model "$BASE_MODEL" \
  --if-model "$IF_MODEL" \
  --fingerprint-data "$FINGERPRINT_DATA" \
  --output-dir "$OUTPUT_DIR" \
  --stages "${stages[@]}" \
  --baseline-bits "${margin_bits[@]}" \
  --delta-bits "${delta_bits[@]}" \
  --group-size "$GROUP_SIZE" \
  --dtype "$DTYPE" \
  --device-map "$DEVICE_MAP" \
  --cache-dir "$CACHE_DIR"
