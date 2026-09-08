# IF-SFT Quantization Phase 1

This directory contains the Phase 1 analysis runners described in
`if_sft_quantization_phase1_spec.md`.

## Full Run

```bash
BASE_MODEL=NousResearch/Llama-2-7b-hf \
IF_MODEL=cnut1648/LLaMA2-7B-fingerprinted-SFT \
FINGERPRINT_DATA=Model-Fingerprint/dataset/llama_fingerprint_chat \
OUTPUT_DIR=quant_fp_phase1 \
bash run_phase1_full.sh
```

The shell script deliberately does not activate or reference any Python
environment. Run it inside the environment you want to use on the server.

## Important Defaults

- RTN is weight-only, symmetric, per-group dequant-in-place.
- All RTN variants use `group_size=128`.
- Baseline bits: `4 3`.
- Sweep bits: `8 6 5 4 3`.
- PPL uses the repository-level `eval_ppl.py`.
- IF verification uses the original IF-SFT target string `ハリネズミ` and the
  first 8 fingerprint rows, matching `Model-Fingerprint/report_FSR_sft_chat.py`.

## Outputs

Primary CSV outputs are written under `quant_fp_phase1/results/`:

- `baseline.csv`
- `fingerprint_margin_per_sample.csv`
- `fingerprint_margin_drop_analysis.csv`
- `bitwidth_sweep.csv`
- `fp_delta_by_tensor.csv`
- `fp_delta_by_block.csv`
- `delta_survival_rtn3_by_tensor.csv`
- `delta_survival_rtn3_by_block.csv`
- `delta_survival_rtn4_by_tensor.csv`
- `delta_survival_rtn4_by_block.csv`
- `blockwise_rtn3.csv`

Token-level margin JSON files are written under
`quant_fp_phase1/results/token_level/`.

Plots are written under `quant_fp_phase1/plots/`.

## Stage-Only Runs

```bash
python -m quant_fp_phase1.scripts.run_phase1 --stages baseline ...
python -m quant_fp_phase1.scripts.run_phase1 --stages margin ...
python -m quant_fp_phase1.scripts.run_phase1 --stages sweep ...
python -m quant_fp_phase1.scripts.run_phase1 --stages delta ...
python -m quant_fp_phase1.scripts.run_phase1 --stages blockwise ...
python -m quant_fp_phase1.scripts.run_phase1 --stages plots ...
```

For a quick smoke run on a GPU server, add `--max-margin-samples 1` and omit
`--run-ppl`.

