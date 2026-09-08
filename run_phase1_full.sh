#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-NousResearch/Llama-2-7b-hf}"
IF_MODEL="${IF_MODEL:-cnut1648/LLaMA2-7B-fingerprinted-SFT}"
FINGERPRINT_DATA="${FINGERPRINT_DATA:-Model-Fingerprint/dataset/llama_fingerprint_chat}"
OUTPUT_DIR="${OUTPUT_DIR:-quant_fp_phase1}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
SEQLEN="${SEQLEN:-2048}"
CACHE_DIR="${CACHE_DIR:-./dataset_cache}"

python -m quant_fp_phase1.scripts.run_phase1 \
  --base-model "$BASE_MODEL" \
  --if-model "$IF_MODEL" \
  --fingerprint-data "$FINGERPRINT_DATA" \
  --output-dir "$OUTPUT_DIR" \
  --stages all \
  --baseline-bits 4 3 \
  --sweep-bits 8 6 5 4 3 \
  --delta-bits 3 4 \
  --group-size 128 \
  --dtype "$DTYPE" \
  --device-map "$DEVICE_MAP" \
  --seqlen "$SEQLEN" \
  --cache-dir "$CACHE_DIR" \
  --run-ppl

