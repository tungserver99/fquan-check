# Phase 1 Summary

This file is a results template. Fill it after running `run_phase1_full.sh` on
the GPU server.

## Configuration

- Base model:
- IF-SFT model:
- Fingerprint data:
- RTN: affine min-max copied from `far-round`, group size 128, transformer linear layers only (`lm_head`/embeddings excluded)
- RTN bits:
- Torch / Transformers / CUDA:

## Baseline

See `results/baseline.csv`.

## Margin Analysis

See `results/fingerprint_margin_per_sample.csv` and
`results/fingerprint_margin_drop_analysis.csv`.

## Delta Survival

See `results/fp_delta_all_tensors.csv`, `results/fp_delta_quantized_tensors_only.csv`, legacy aliases `results/fp_delta_by_tensor.csv` / `results/fp_delta_by_block.csv`, and `results/delta_survival_rtn*_by_*.csv`.

## Blockwise RTN3

See `results/blockwise_rtn3.csv`.

## Correlations

- Verification vs margin:
- Fingerprint drop vs PPL change:
- Delta survival vs blockwise margin drop:

## Hypothesis Status

- Margin erosion:
- Utility-vs-fingerprint selectivity:
- Delta distortion:
- Block-level connection:

## Caveats

- No new fingerprint training is performed.
- No AWQ, GPTQ, activation-aware quantization, recovery, distillation, or
  selective quantization method is implemented in Phase 1.
