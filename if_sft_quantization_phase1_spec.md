# Phase 1 Experimental Specification
## Quantization-Induced Fingerprint Degradation Analysis for IF-SFT on LLaMA-2-7B

## 1. Objective

The goal of this phase is **not** to design a new quantization method yet.

The goal is to understand **why an existing IF-SFT fingerprint survives or degrades under post-training quantization**, with particular focus on whether quantization:

1. distorts the parameter update introduced by fingerprint fine-tuning;
2. reduces the confidence / behavioral margin of the fingerprint response;
3. affects some transformer blocks much more strongly than others;
4. can reduce fingerprint verification while normal language-model utility remains relatively stable.

This phase should establish a mechanism before any later model-only selective quantization method is designed.

---

## 2. Important Constraints

### 2.1 Do NOT retrain the fingerprint model

The IF-SFT checkpoint for **LLaMA-2-7B** already exists and must be reused directly.

Use:

- `M_base`: original LLaMA-2-7B checkpoint used before fingerprint embedding.
- `M_if`: existing IF-SFT fingerprinted LLaMA-2-7B checkpoint.

Do **not**:

- rerun IF fingerprint training;
- recreate fingerprint query/response pairs unless already required by the public evaluation code;
- modify IF training hyperparameters;
- introduce a new fingerprint embedding method.

### 2.2 Reuse public source code

Whenever possible, reuse the official/public implementations for:

- IF-SFT checkpoint loading;
- IF verification;
- model evaluation;
- RTN quantization.

Do not reimplement a paper method from scratch unless absolutely necessary.

### 2.3 Keep the quantization implementation fixed

The purpose of Phase 1 is mechanistic analysis.

Therefore, use the **same RTN implementation and configuration already used in the previous IF experiments**.

Do not mix quantization libraries/configurations across experiments.

In particular, keep fixed:

- weight grouping;
- symmetric/asymmetric setting;
- per-channel/per-group setting;
- scale computation;
- zero-point handling;
- excluded layers, if any;
- dtype of non-quantized modules.

Only the requested experimental variable should change.

---

# 3. Main Research Hypothesis

Let:

\[
\theta_0
\]

be the parameters of the original LLaMA-2-7B model.

Let:

\[
\theta_F
\]

be the parameters of the IF-SFT model.

Fingerprint fine-tuning introduces:

\[
\Delta_F = \theta_F - \theta_0.
\]

The working hypothesis is:

> IF-SFT creates a special high-confidence behavior through the fine-tuning update \(\Delta_F\). Quantization changes or removes part of this update, which reduces the behavioral margin of the fingerprint response. Verification fails when this margin crosses the decision boundary.

The proposed causal chain is:

\[
\Delta_F
\rightarrow
\text{fingerprint behavior}
\rightarrow
\text{quantization distortion}
\rightarrow
\text{margin erosion}
\rightarrow
\text{fingerprint degradation}.
\]

Phase 1 must test each part of this chain.

---

# 4. Models to Evaluate

Minimum required models:

1. **IF-FP**
   - Existing IF-SFT checkpoint.
   - Full precision.

2. **IF-RTN4**
   - Existing IF-SFT checkpoint quantized with RTN 4-bit.

3. **IF-RTN3**
   - Existing IF-SFT checkpoint quantized with RTN 3-bit.

Optional for the bit-width sweep:

4. IF-RTN8
5. IF-RTN6
6. IF-RTN5

Do not add AWQ/GPTQ in Phase 1.

---

# 5. Experiment 0 — Baseline Reproduction

## Goal

Confirm that the current code reproduces the already observed IF behavior before any new analysis is added.

## Run

Evaluate:

- IF-FP
- IF-RTN4
- IF-RTN3

## Metrics

At minimum report:

### Fingerprint

- IF verification score / fingerprint accuracy.

### Utility

- WikiText-2 perplexity.
- Existing lightweight downstream utility benchmarks already available in the current repository.

Do not add a large benchmark suite only for this phase.

## Expected output

Create:

```text
results/baseline.csv
```

Recommended columns:

```text
model
quantizer
bits
group_size
fingerprint_score
wikitext2_ppl
utility_task_1
utility_task_2
...
```

Also save the complete quantization configuration.

Example:

```text
results/configs/rtn3.json
results/configs/rtn4.json
```

This is essential for reproducibility.

---

# 6. Experiment 1 — Fingerprint Behavioral Margin Analysis

## Priority

**Highest priority experiment.**

If fingerprint degradation cannot be explained by changes in target-token confidence / margin, do not force the hypothesis.

---

## 6.1 Definition

For each IF fingerprint query-response pair:

\[
x \rightarrow y_1, y_2, \ldots, y_T
\]

run the model using **teacher forcing**.

At target position \(t\), compute:

\[
m_t =
z(y_t) -
\max_{v \neq y_t} z(v)
\]

where:

- \(z(y_t)\) is the logit of the correct fingerprint target token;
- the second term is the highest competing-token logit.

Interpretation:

- \(m_t > 0\): correct target token is top-1.
- \(m_t < 0\): another token has crossed above the fingerprint target.

Also record:

\[
p_t = p(y_t \mid x,y_{<t})
\]

and token-level NLL:

\[
\text{NLL}_t = -\log p_t.
\]

---

## 6.2 Models

Run the exact same IF query-response pairs on:

- IF-FP
- IF-RTN4
- IF-RTN3

---

## 6.3 Required metrics per fingerprint sample

For each fingerprint sample, compute:

### Mean target margin

\[
\bar m =
\frac{1}{T}
\sum_{t=1}^{T} m_t
\]

### Minimum target margin

\[
m_{\min}
=
\min_t m_t
\]

### Mean target-token probability

\[
\bar p =
\frac{1}{T}
\sum_t p_t
\]

### Sequence NLL

\[
\text{SeqNLL}
=
\sum_t -\log p_t
\]

Optionally normalize by sequence length.

### Negative-margin ratio

\[
R_{\text{neg}}
=
\frac{
\#\{t : m_t < 0\}
}{
T
}
\]

This is particularly important because it directly measures how many fingerprint target tokens have crossed the decision boundary.

---

## 6.4 Required saved output

Create one row per sample per model:

```text
results/fingerprint_margin_per_sample.csv
```

Recommended columns:

```text
sample_id
model_variant
bits
verified
response_length
mean_margin
min_margin
mean_target_probability
sequence_nll
negative_margin_ratio
```

Also save token-level values:

```text
results/token_level/
    sample_<id>_fp.json
    sample_<id>_rtn4.json
    sample_<id>_rtn3.json
```

Each token record should contain:

```text
position
target_token_id
target_token_text
target_logit
best_competing_token_id
best_competing_token_text
best_competing_logit
margin
target_probability
nll
```

---

## 6.5 Required analysis

Split RTN3 samples into:

- fingerprints that still verify successfully;
- fingerprints that fail verification.

Compare their FP and RTN3:

- mean margin;
- minimum margin;
- margin drop;
- negative-margin ratio;
- target probability.

Key question:

> Do failed fingerprints have either a smaller initial margin or a much larger quantization-induced margin drop?

Required derived value:

\[
\Delta m
=
m_{\text{FP}} - m_{\text{quantized}}.
\]

Save it in the analysis table.

---

# 7. Experiment 2 — RTN Bit-Width Sweep

## Goal

Determine whether fingerprint degradation changes smoothly with quantization strength, and whether fingerprint degrades faster than normal utility.

---

## 7.1 Quantization settings

Minimum:

- FP
- RTN4
- RTN3

Preferred if implementation is straightforward:

- FP
- RTN8
- RTN6
- RTN5
- RTN4
- RTN3

Do not prioritize RTN2 unless the model remains usable; 2-bit is optional.

All non-bit-width settings must remain identical.

---

## 7.2 Metrics

For each bit width:

### Fingerprint

- IF verification score.
- mean fingerprint margin.
- median fingerprint margin.
- minimum-margin distribution.
- negative-margin ratio.
- sequence NLL.

### Utility

- WikiText-2 PPL.
- same lightweight utility tasks as Experiment 0.

---

## 7.3 Required outputs

```text
results/bitwidth_sweep.csv
```

Recommended columns:

```text
bits
fingerprint_score
mean_margin
median_margin
mean_min_margin
negative_margin_ratio
mean_sequence_nll
wikitext2_ppl
utility_task_1
utility_task_2
...
```

Required plots:

```text
plots/bitwidth_vs_fingerprint_score.png
plots/bitwidth_vs_margin.png
plots/bitwidth_vs_ppl.png
plots/utility_drop_vs_fingerprint_drop.png
```

The last plot is important.

We want to see whether there is a region where:

\[
\text{fingerprint degradation is substantial}
\]

while:

\[
\text{utility degradation is still small}.
\]

---

# 8. Experiment 3 — Fingerprint Fine-Tuning Delta Analysis

## Goal

Measure how much of the parameter change introduced by IF-SFT survives quantization.

This is **mechanistic analysis**, not a new method.

---

## 8.1 Compute the original fingerprint update

Load:

\[
\theta_0 = M_{\text{base}}
\]

and:

\[
\theta_F = M_{\text{if}}.
\]

For every matching tensor:

\[
\Delta_F^{(l)}
=
\theta_F^{(l)}
-
\theta_0^{(l)}.
\]

Do not assume every tensor should be analyzed identically.

At minimum separate:

- attention projections;
- MLP projections;
- layer norms;
- embeddings;
- LM head.

If embeddings / LM head are tied, handle them correctly.

---

## 8.2 Report FP delta statistics

For every tensor and transformer block, compute:

\[
\|\Delta_F^{(l)}\|_2
\]

and relative update magnitude:

\[
r_l =
\frac{
\|\Delta_F^{(l)}\|_2
}{
\|\theta_0^{(l)}\|_2 + \epsilon
}.
\]

Also report:

- mean absolute delta;
- max absolute delta;
- standard deviation of delta.

Create:

```text
results/fp_delta_by_tensor.csv
results/fp_delta_by_block.csv
```

---

## 8.3 Quantized delta comparison

For this analysis, quantize the **base model and fingerprint model with exactly matched RTN settings**.

Define:

\[
\Delta_F^Q =
Q(\theta_F) -
Q(\theta_0).
\]

Important:

Use matching quantization configuration.

If possible, also support an analysis mode where both models use the same externally defined scale/grid for a given group, but keep this as an additional analysis rather than silently changing the normal RTN pipeline.

---

## 8.4 Delta survival metrics

For each tensor/block compute:

### Norm survival ratio

\[
S_l =
\frac{
\|\Delta_{F,l}^{Q}\|_2
}{
\|\Delta_{F,l}\|_2 + \epsilon
}.
\]

Interpretation:

- high \(S_l\): much of the fine-tuning-induced difference remains after quantization;
- low \(S_l\): quantization strongly collapses/distorts the fingerprint-induced difference.

### Direction preservation

\[
C_l =
\cos(
\Delta_{F,l},
\Delta_{F,l}^{Q}
).
\]

### Optional coordinate collapse rate

\[
R_l^{\text{zero}}
=
\frac{
\#\{i :
\Delta_{F,i}\neq0
\land
\Delta^Q_{F,i}=0
\}
}{
\#\{i : \Delta_{F,i}\neq0\}
}.
\]

Do not use this as the main metric; it is secondary because very small and very large updates should not receive identical importance.

---

## 8.5 Required output

```text
results/delta_survival_rtn3_by_tensor.csv
results/delta_survival_rtn4_by_tensor.csv

results/delta_survival_rtn3_by_block.csv
results/delta_survival_rtn4_by_block.csv
```

Recommended columns:

```text
layer
module
fp_delta_l2
fp_relative_delta
quantized_delta_l2
delta_norm_survival
delta_cosine_similarity
coordinate_collapse_rate
```

Required plot:

```text
plots/block_delta_survival_rtn3.png
plots/block_delta_survival_rtn4.png
```

---

# 9. Experiment 4 — Block-Wise RTN3 Quantization

## Goal

Identify transformer blocks where quantization disproportionately damages fingerprint behavior while leaving normal model behavior relatively stable.

This should be done **after Experiments 1–3**, because its purpose is to connect parameter-level distortion to behavioral margin degradation.

---

## 9.1 Construction

LLaMA-2-7B has multiple transformer blocks.

For block \(l\), construct:

\[
M^{(l)}
\]

such that:

- block \(l\) is RTN3 quantized;
- all other transformer blocks remain FP;
- embeddings / LM head remain FP unless they are explicitly being tested separately.

Do this one block at a time.

Do not cumulatively quantize blocks.

Example:

```text
model_block_00_rtn3:
    block 0 = RTN3
    blocks 1...N = FP

model_block_01_rtn3:
    block 1 = RTN3
    all others = FP
```

---

## 9.2 Metrics per block

For every block-wise model, evaluate:

### Fingerprint behavior

- IF verification score.
- mean fingerprint margin.
- minimum fingerprint margin.
- negative-margin ratio.
- sequence NLL.

### Utility

- WikiText-2 PPL.

Do not run a heavy full benchmark suite per block.

---

## 9.3 Required output

```text
results/blockwise_rtn3.csv
```

Recommended columns:

```text
block_id
fingerprint_score
mean_margin
margin_drop_from_fp
negative_margin_ratio
sequence_nll
wikitext2_ppl
ppl_increase_from_fp
delta_norm_survival_rtn3
delta_cosine_similarity_rtn3
```

The final two columns should be joined from Experiment 3.

---

# 10. Main Cross-Experiment Analysis

The final Phase 1 analysis should test the following relationships.

---

## Question A

Does RTN quantization reduce fingerprint behavioral margin?

Compare:

\[
m_{\text{FP}},
m_{\text{RTN4}},
m_{\text{RTN3}}.
\]

---

## Question B

Does fingerprint verification fail when margin crosses or approaches zero?

Check correlation between:

- IF verification result;
- mean margin;
- minimum margin;
- negative-margin ratio.

---

## Question C

Does stronger quantization reduce fingerprint faster than general utility?

Compare:

\[
\Delta \text{Fingerprint}
\]

against:

\[
\Delta \text{PPL / utility}.
\]

---

## Question D

Does RTN collapse or distort the fine-tuning-induced parameter difference?

Analyze:

\[
\Delta_F
\]

versus:

\[
\Delta_F^Q.
\]

---

## Question E

Are blocks with strong delta distortion also the blocks that cause strong fingerprint-margin degradation?

For each block compare:

\[
\text{delta survival}
\]

with:

\[
\text{margin drop under block-wise quantization}.
\]

Useful simple correlations:

- Pearson correlation;
- Spearman rank correlation.

Do not introduce a learned predictor or weighted score in Phase 1.

---

# 11. Expected Key Figures

At minimum generate the following figures.

## Figure 1

Fingerprint score across:

- FP
- RTN4
- RTN3

## Figure 2

Distribution of per-sample fingerprint margins:

- FP
- RTN4
- RTN3

Prefer boxplot / violin / histogram.

## Figure 3

FP margin vs RTN3 margin per fingerprint sample.

Each point = one fingerprint sample.

Mark whether the sample still verifies under RTN3.

## Figure 4

Bit width vs:

- fingerprint score;
- fingerprint margin;
- WikiText-2 PPL.

## Figure 5

Per-block fingerprint fine-tuning delta magnitude.

## Figure 6

Per-block RTN3 delta survival ratio.

## Figure 7

Block-wise RTN3:

\[
x = \text{PPL increase}
\]

\[
y = \text{fingerprint margin drop}
\]

Each point = one transformer block.

## Figure 8

\[
x = \text{delta survival ratio}
\]

\[
y = \text{fingerprint margin drop}
\]

This figure directly connects parameter-level and behavior-level effects.

---

# 12. Suggested Repository Structure

Use the existing repository when possible.

Add a self-contained analysis directory such as:

```text
quant_fp_phase1/
│
├── configs/
│   ├── rtn3.yaml
│   ├── rtn4.yaml
│   └── sweep.yaml
│
├── scripts/
│   ├── eval_baseline.py
│   ├── analyze_margin.py
│   ├── sweep_bits.py
│   ├── analyze_delta.py
│   ├── run_blockwise_quant.py
│   └── aggregate_results.py
│
├── src/
│   ├── quantization.py
│   ├── fingerprint_eval.py
│   ├── margin_metrics.py
│   ├── delta_metrics.py
│   └── utility_eval.py
│
├── results/
│   ├── configs/
│   ├── token_level/
│   ├── baseline.csv
│   ├── fingerprint_margin_per_sample.csv
│   ├── bitwidth_sweep.csv
│   ├── fp_delta_by_tensor.csv
│   ├── fp_delta_by_block.csv
│   ├── delta_survival_rtn3_by_block.csv
│   ├── delta_survival_rtn4_by_block.csv
│   └── blockwise_rtn3.csv
│
└── plots/
```

The exact folder names can be adapted to the existing codebase, but outputs must remain structured and reproducible.

---

# 13. Implementation Requirements

## Determinism

Set and log:

- random seed;
- torch version;
- transformers version;
- quantization library/version;
- CUDA version;
- model revision/hash if available.

---

## Precision

For FP reference calculations:

- use the original checkpoint dtype;
- avoid silently casting to lower precision beyond what the checkpoint normally uses.

For logits/margin analysis:

- perform the final margin calculation in FP32 when practical, even if model inference uses FP16/BF16.

---

## Fingerprint evaluation

The official IF verification result remains the primary fingerprint metric.

Margin analysis is an explanatory metric, not a replacement for the official verifier.

---

## Teacher forcing

For margin analysis, use the known target fingerprint response and teacher forcing.

Do not use free generation to compute token-level margins.

Free generation can still be used by the official verifier if required by IF's original evaluation protocol.

---

## Memory

Do not keep multiple full LLaMA-2-7B model copies on GPU simultaneously unless necessary.

For delta analysis:

- process tensor-by-tensor / layer-by-layer;
- move tensors to CPU if needed;
- avoid storing redundant full-size copies of \(\Delta_F\).

---

# 14. What NOT to Implement in Phase 1

Do not implement any of the following yet:

- new fingerprint training;
- new fingerprint dataset;
- AWQ;
- GPTQ;
- activation-aware quantization;
- selective rounding;
- weight ranking;
- neuron ranking;
- optimization over fingerprint score;
- fine-tuning after quantization;
- model recovery;
- distillation;
- weighted heuristic score;
- attack-specific objective;
- any method requiring fingerprint verifier feedback during quantization.

Phase 1 is **analysis only**.

---

# 15. Minimum Viable Execution Order

Implement and run in this order.

## Step 1

Baseline:

```text
IF-FP
IF-RTN4
IF-RTN3
```

Confirm the known behavior.

## Step 2

Margin analysis on FP / RTN4 / RTN3.

This is the first critical result.

## Step 3

RTN bit-width sweep.

## Step 4

Compute:

\[
\Delta_F =
\theta_F - \theta_0
\]

and its layer/block statistics.

## Step 5

Compute quantized delta survival for RTN3 and RTN4.

## Step 6

Run block-wise RTN3 analysis.

## Step 7

Aggregate correlations and generate all plots.

---

# 16. Decision Criteria After Phase 1

Do not automatically continue to designing a new quantization procedure.

Use the following decision logic.

---

## Case A — Strong positive evidence

Continue if results show most of the following:

1. fingerprint margin clearly decreases with stronger RTN;
2. failed IF samples have smaller / more strongly eroded margins;
3. fingerprint degradation occurs before severe utility degradation;
4. quantization measurably distorts the IF fine-tuning delta;
5. specific blocks show strong fingerprint-margin damage with limited PPL impact;
6. block-level delta distortion correlates with fingerprint-margin degradation.

Then Phase 2 can investigate a model-only quantization procedure that identifies such fragile regions without using secret fingerprint feedback.

---

## Case B — Margin explains behavior, but delta survival does not

If:

- margin erosion strongly predicts fingerprint degradation;
- parameter-delta collapse does not correlate well;

then continue focusing on **activation / functional sensitivity**, not raw parameter difference.

Do not force the delta hypothesis.

---

## Case C — Delta distortion exists but fingerprint margin remains robust

Then IF-SFT likely has high behavioral redundancy / high margin.

The next question becomes:

> Why does a heavily distorted fine-tuning update still preserve the fingerprint mapping?

This would motivate redundancy / distributed representation analysis.

---

## Case D — Fingerprint only degrades when utility collapses

Then standard RTN is not giving useful selectivity for IF.

Do not proceed directly to a selective quantization method based on this mechanism.

The hypothesis should be reconsidered before further engineering.

---

# 17. Final Deliverables from Codex

Codex should return:

1. runnable scripts for all Phase 1 experiments;
2. a README with exact commands;
3. saved quantization configs;
4. CSV files with all raw/aggregated metrics;
5. generated plots;
6. a short `PHASE1_SUMMARY.md` containing:
   - experiment configuration;
   - baseline results;
   - main margin results;
   - main delta-survival results;
   - block-wise results;
   - correlations;
   - whether each hypothesis is supported / unsupported;
   - any implementation caveats.

Do not write a new removal method in `PHASE1_SUMMARY.md`.

The purpose of this phase is to establish the mechanism first.

---

# 18. Short Summary for the Implementation Team

Use the existing IF-SFT LLaMA-2-7B checkpoint.

Do **not** retrain IF.

The main experimental sequence is:

```text
Existing IF-SFT checkpoint
        ↓
FP / RTN4 / RTN3 baseline
        ↓
Fingerprint target-token margin analysis
        ↓
RTN bit-width sweep
        ↓
Compare base vs IF-SFT parameter delta
        ↓
Measure how much of that delta survives RTN
        ↓
Quantize one transformer block at a time
        ↓
Connect:
parameter distortion
→ fingerprint margin erosion
→ verification degradation
→ utility change
```

The central question is:

> Does quantization degrade IF because it distorts the fine-tuning-induced behavior enough to erode the fingerprint response margin, while normal language-model behavior remains comparatively stable?

Only after this question is answered should a new model-only quantization procedure be designed.
