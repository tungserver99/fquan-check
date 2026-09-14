# BASE-RTN4 vs IF-RTN4 Exact Difference and Causal Localization

## 1. Goal

This analysis compares exactly two models:

- `BASE-RTN4`: `NousResearch/Llama-2-7b-hf` quantized with the current Phase-1 RTN4 implementation.
- `IF-RTN4`: the IF-SFT fingerprinted LLaMA-2-7B checkpoint quantized with the **same** RTN4 implementation.

Known behavior from the current experiments:

- `BASE-RTN4`: fingerprint verification = `0/8`.
- `IF-RTN4`: fingerprint verification = `8/8`.

The central question is:

> **After applying the same RTN4 quantization to the base model and the fingerprinted model, where exactly do the two quantized models still differ, and which of those residual differences are functionally necessary for preserving the fingerprint?**

This is an **analysis task only**.

Do **not** design a new quantization method in this phase.

---

# 2. Scope

The analysis has only three stages:

1. **Exact difference map**
   - Identify every place where `BASE-RTN4` and `IF-RTN4` differ after quantization.

2. **Localization**
   - Aggregate those differences by transformer block, module, output row, and RTN group.

3. **Causal swap**
   - Starting from `IF-RTN4`, replace selected quantized regions with the corresponding regions from `BASE-RTN4`.
   - Measure whether fingerprint verification decreases.

The intended flow is:

```text
BASE checkpoint ──RTN4──> BASE-RTN4 ─┐
                                     ├─> exact difference map
IF-SFT checkpoint ─RTN4──> IF-RTN4 ──┘
                                     │
                                     ├─> block/module/group localization
                                     │
                                     └─> causal BASE→IF replacement tests
```

---

# 3. Important Constraints

## 3.1 Use exactly the existing RTN4 configuration

Do not change the quantization algorithm.

Use the same Phase-1 RTN path already used in the current experiments:

```text
bits = 4
group_size = 128
quantization = asymmetric affine min-max RTN
granularity = per-output-row / per-input-group
integer zero-point
same zero-point rounding rule as current implementation
same tensor discovery
same excluded modules
same dtype handling
```

The existing implementation quantizes transformer linear layers and excludes modules such as embeddings and `lm_head`.

Do not introduce:

- RTN3;
- AWQ;
- GPTQ;
- mixed-bit quantization;
- shared-grid quantization;
- alternative clipping;
- activation-aware quantization;
- calibration-based optimization;
- any new rounding method.

The purpose is to compare the two **existing RTN4 models** cleanly.

---

## 3.2 Quantize both models inside the same analysis code path

Do not compare two checkpoints produced by unrelated scripts if this can be avoided.

The preferred implementation is:

```python
base_model = load_base_model()
if_model = load_if_model()

base_rtn4, base_quant_state = quantize_rtn4(base_model)
if_rtn4, if_quant_state = quantize_rtn4(if_model)
```

Both calls must use exactly the same quantization function and configuration.

The analysis must log the configuration once and confirm that it is identical for both models.

---

## 3.3 Preserve the quantizer internals

The comparison must not rely only on the final dequantized floating-point weights.

For every quantized group, retain:

```text
integer quantization code q
scale s
zero-point z
dequantized weight w_q
```

This is important because the base and fingerprinted models are quantized using their own asymmetric min/max ranges.

Therefore:

```text
q_base == q_if
```

does **not** necessarily imply:

```text
dequant_weight_base == dequant_weight_if
```

if their scale or zero-point differs.

The analysis must therefore separately report:

1. integer-code differences;
2. scale differences;
3. zero-point differences;
4. actual dequantized-weight differences.

---

# 4. Model Naming

Use these names consistently:

```text
BASE-FP   = original base checkpoint
IF-FP     = original IF-SFT checkpoint

BASE-RTN4 = RTN4(BASE-FP)
IF-RTN4   = RTN4(IF-FP)
```

The main comparison in this document is only:

```text
BASE-RTN4  vs  IF-RTN4
```

---

# 5. Sanity Check Before Analysis

Before comparing parameters, reproduce the behavioral control.

Run the same 8 official IF fingerprint samples on:

```text
BASE-RTN4
IF-RTN4
```

Save:

```text
results/rtn4_behavior_control.csv
```

Columns:

```text
sample_id
dataset_index
model_variant
verified
generated_text
expected_text
```

Expected current observation:

```text
BASE-RTN4: 0/8 verified
IF-RTN4:   8/8 verified
```

If this is not reproduced, stop and debug before continuing.

---

# 6. Experiment 1 — Exact RTN4 Difference Map

## 6.1 Goal

Determine exactly where the two quantized models differ.

For every quantized tensor:

\[
W_B^Q = Q_4(W_{\text{base}})
\]

and

\[
W_F^Q = Q_4(W_{\text{IF}})
\]

define:

\[
D = W_F^Q - W_B^Q.
\]

The analysis must preserve coordinate-level location information.

---

## 6.2 Correct RTN group indexing

The current quantization is:

```text
per-output-row / per-input-group
group_size = 128
```

Therefore, a quantization group is **not** an arbitrary flat chunk of 128 weights.

For a weight matrix:

\[
W \in \mathbb{R}^{d_{out} \times d_{in}},
\]

a group is identified by:

```text
output_row = r
group_id   = floor(input_col / 128)
```

and contains:

```text
W[r, group_id*128 : min((group_id+1)*128, d_in)]
```

All group-level outputs and swap operations must use this exact indexing.

---

## 6.3 Coordinate-level fields

For every quantized coordinate `(output_row, input_col)`, record:

```text
tensor_name
block_id
module_type

output_row
input_col
group_id
offset_in_group

base_qcode
if_qcode
qcode_diff
abs_qcode_diff

base_scale
if_scale
scale_diff
relative_scale_diff

base_zero_point
if_zero_point
zero_point_diff

base_dequant_weight
if_dequant_weight
dequant_diff
abs_dequant_diff
```

Definitions:

\[
\Delta q_i = q_{F,i} - q_{B,i}
\]

\[
\Delta s_g = s_{F,g} - s_{B,g}
\]

\[
\Delta z_g = z_{F,g} - z_{B,g}
\]

\[
D_i = W^Q_{F,i} - W^Q_{B,i}.
\]

Do not threshold `dequant_diff` during raw-data creation.

Save the numerical difference itself.

This avoids hiding small but real differences behind an arbitrary tolerance.

---

## 6.4 Exact code-difference mask

Define:

\[
M_i^{code}
=
\mathbf{1}
[q_{F,i} \neq q_{B,i}].
\]

This is the cleanest exact indicator that the two weights landed on different integer RTN codes.

Also record:

```text
same_qcode = (base_qcode == if_qcode)
```

---

## 6.5 Group-grid difference

For every `(tensor, output_row, group_id)`, record whether the affine quantization grid differs:

\[
M_g^{scale}
=
\mathbf{1}[s_{F,g} \neq s_{B,g}]
\]

and

\[
M_g^{zp}
=
\mathbf{1}[z_{F,g} \neq z_{B,g}].
\]

Also report:

\[
R_g^{scale}
=
\frac{|s_{F,g}-s_{B,g}|}
{|s_{B,g}|+\epsilon}.
\]

Use a small numerical epsilon only to avoid division by zero.

Do not use this as a learned score.

---

## 6.6 Required raw outputs

Prefer Parquet or compressed NumPy files rather than CSV for coordinate-level data.

Required:

```text
results/rtn4_exact_diff_coordinates.parquet
results/rtn4_exact_diff_groups.parquet
```

The coordinate file should contain one row per quantized weight.

The group file should contain one row per:

```text
tensor_name
output_row
group_id
```

with:

```text
block_id
module_type

num_weights_in_group

base_scale
if_scale
scale_diff
relative_scale_diff

base_zero_point
if_zero_point
zero_point_diff

num_qcode_different
qcode_diff_ratio

dequant_diff_l1
dequant_diff_l2
dequant_diff_max
dequant_diff_mean_abs
```

---

# 7. Experiment 2 — Localization of the Residual RTN4 Difference

## 7.1 Goal

Use the exact map from Experiment 1 to answer:

> Are the remaining differences distributed across the whole model, or concentrated in particular blocks/modules/groups?

Do not invent a weighted importance score.

Use simple, interpretable statistics.

---

## 7.2 Tensor-level summary

For every quantized tensor, compute:

\[
R_T^{code}
=
\frac{
\#\{i:q_{F,i}\neq q_{B,i}\}
}{
N_T
}.
\]

Also compute:

\[
\|D_T\|_1,
\]

\[
\|D_T\|_2,
\]

\[
\max_i |D_{T,i}|,
\]

and:

\[
\operatorname{mean}_i |D_{T,i}|.
\]

Save:

```text
results/rtn4_diff_by_tensor.csv
```

Columns:

```text
tensor_name
block_id
module_type
num_weights

num_qcode_different
qcode_diff_ratio

dequant_diff_l1
dequant_diff_l2
dequant_diff_max
dequant_diff_mean_abs

num_groups
num_groups_scale_different
fraction_groups_scale_different
num_groups_zp_different
fraction_groups_zp_different
mean_relative_scale_diff
```

Sort a copy of the table by:

```text
qcode_diff_ratio
dequant_diff_l2
```

but do not combine them into a new score.

---

## 7.3 Block-level summary

Aggregate all quantized tensors inside each transformer block.

For block \(l\):

\[
R_l^{code}
=
\frac{
\#\{q_F \neq q_B\}
}{
N_l
}.
\]

Also compute:

\[
D_l^{L2}
=
\sqrt{
\sum_{i\in l}
(W_{F,i}^Q-W_{B,i}^Q)^2
}.
\]

Save:

```text
results/rtn4_diff_by_block.csv
```

Columns:

```text
block_id
num_weights
num_qcode_different
qcode_diff_ratio
dequant_diff_l1
dequant_diff_l2
dequant_diff_max
dequant_diff_mean_abs
```

---

## 7.4 Module-level summary

Use only the existing transformer linear module taxonomy:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

Aggregate both:

1. globally by module type;
2. by `(block_id, module_type)`.

Save:

```text
results/rtn4_diff_by_module.csv
results/rtn4_diff_by_block_module.csv
```

---

## 7.5 Output-row summary

Within each tensor, also aggregate by output row.

This is useful because the quantizer is row-wise.

Save:

```text
results/rtn4_diff_by_row.parquet
```

Columns:

```text
tensor_name
block_id
module_type
output_row

num_weights
num_qcode_different
qcode_diff_ratio

dequant_diff_l1
dequant_diff_l2
dequant_diff_max
dequant_diff_mean_abs

num_groups
```

---

## 7.6 Group-level ranking

The most fine-grained structural unit should be the RTN group:

```text
tensor_name
output_row
group_id
```

For every group rank separately by:

```text
qcode_diff_ratio
dequant_diff_l2
dequant_diff_max
```

Do not combine these three metrics.

Create:

```text
results/rtn4_diff_groups_top_by_qcode_ratio.csv
results/rtn4_diff_groups_top_by_l2.csv
results/rtn4_diff_groups_top_by_maxdiff.csv
```

Each file can contain the top 100 groups.

---

# 8. Required Figures

Keep the plots minimal.

## Figure 1 — Difference ratio by block

x-axis:

```text
block 0 ... block 31
```

y-axis:

\[
R_l^{code}.
\]

Output:

```text
plots/rtn4_qcode_diff_ratio_by_block.png
```

---

## Figure 2 — Block × module heatmap

Rows:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

Columns:

```text
block 0 ... block 31
```

Generate two heatmaps:

1. `qcode_diff_ratio`;
2. `dequant_diff_l2`.

Outputs:

```text
plots/rtn4_block_module_qcode_diff_heatmap.png
plots/rtn4_block_module_l2_diff_heatmap.png
```

---

## Figure 3 — Group-difference distribution

Plot the distribution of:

\[
R_g^{code}.
\]

Output:

```text
plots/rtn4_group_qcode_diff_distribution.png
```

The purpose is to distinguish between:

```text
many groups with small differences
```

and:

```text
a small number of highly different groups.
```

Do not over-interpret the plot automatically.

---

# 9. Experiment 3 — Causal Swap Analysis

## 9.1 Goal

Experiments 1–2 tell us **where** the two RTN4 models differ.

Experiment 3 asks:

> Which of those differences are actually necessary for the fingerprint behavior of IF-RTN4?

Use `IF-RTN4` as the starting model because it currently verifies `8/8`.

Then replace selected quantized regions with the corresponding values from `BASE-RTN4`.

This produces a direct causal intervention:

\[
\text{IF-RTN4 region}
\leftarrow
\text{BASE-RTN4 region}.
\]

---

# 10. Causal Swap Level 1 — Transformer Block

## 10.1 Construction

For each block \(l\), create a temporary model:

\[
M_l^{swap}
\]

where:

```text
block l                = BASE-RTN4 block l
all other blocks       = IF-RTN4
non-quantized modules  = unchanged from IF-RTN4
```

This is a one-block-at-a-time replacement.

Do not cumulatively replace blocks.

---

## 10.2 Evaluation

Run only the official 8 fingerprint samples.

Primary metric:

```text
verified_count / 8
```

Also save the raw generated text.

Do not run WikiText PPL or a large utility suite at this stage.

Save:

```text
results/rtn4_block_swap.csv
```

Columns:

```text
block_id

baseline_if_verified_count
verified_count_after_swap
fingerprint_drop

block_qcode_diff_ratio
block_dequant_diff_l2
```

where:

\[
\text{fingerprint_drop}
=
8-\text{verified_count_after_swap}.
\]

The last two columns are joined from Experiment 2 only for interpretation.

Do not combine them into a score.

---

# 11. Causal Swap Level 2 — Module

Only proceed to module-level swaps inside blocks that show a fingerprint change during the block-swap experiment.

For each selected block, independently replace:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

from `IF-RTN4` with the corresponding `BASE-RTN4` quantized weights.

All other parameters remain from `IF-RTN4`.

Save:

```text
results/rtn4_module_swap.csv
```

Columns:

```text
block_id
module_type

baseline_if_verified_count
verified_count_after_swap
fingerprint_drop

module_qcode_diff_ratio
module_dequant_diff_l2
```

Again, do not cumulatively swap modules.

---

# 12. Causal Swap Level 3 — RTN Group

Only proceed to group-level swaps inside modules that show a fingerprint change during module-level swapping.

A group is exactly:

```text
tensor_name
output_row
group_id
```

with up to 128 contiguous input weights.

For one group \(g\):

\[
W_{IF-RTN4}[g]
\leftarrow
W_{BASE-RTN4}[g].
\]

Everything else remains from `IF-RTN4`.

Run the same 8 fingerprint samples.

Save:

```text
results/rtn4_group_swap.csv
```

Columns:

```text
tensor_name
block_id
module_type
output_row
group_id

num_weights
num_qcode_different
qcode_diff_ratio
dequant_diff_l2
dequant_diff_max

baseline_if_verified_count
verified_count_after_swap
fingerprint_drop
```

This is the most important causal localization table.

---

# 13. Important Rule for Group-Swap Search

Do not test every RTN group in the entire LLaMA-2-7B model blindly.

Use the hierarchy:

```text
block swap
    ↓
module swap inside affected blocks
    ↓
group swap inside affected modules
```

This keeps the search focused and computationally manageable.

If no single block swap changes fingerprint verification, report:

> No single transformer block is individually necessary under the current one-block replacement test.

Do **not** fabricate a finer localization result in that case.

Do not automatically start combinatorial multi-block search in this phase.

That would be a separate follow-up experiment.

---

# 14. Optional Reverse Swap for Strong Candidates

This is optional and should only be run for the strongest candidates found above.

Start from `BASE-RTN4`, which currently verifies `0/8`.

Replace one candidate region with the corresponding region from `IF-RTN4`:

\[
\text{BASE-RTN4 region}
\leftarrow
\text{IF-RTN4 region}.
\]

This tests whether a region is not only necessary, but partially sufficient.

Examples:

```text
BASE-RTN4 + IF block l
BASE-RTN4 + IF module (l, m)
```

Do not run this exhaustively.

Only use it to validate one or a few strong candidates.

Save, if used:

```text
results/rtn4_reverse_swap_candidates.csv
```

---

# 15. What Not to Analyze in This Phase

Do not add unrelated analyses.

Specifically, do not implement:

```text
RTN3
bit-width sweep
token-final / EOS analysis
new fingerprint training
new quantization method
AWQ / GPTQ
mixed-bit quantization
gradient-based optimization
Hessian-based optimization
activation sensitivity
new weighted ranking score
new removal objective
fine-tuning after quantization
```

Do not turn this phase into a general fingerprint robustness study.

The scope is strictly:

```text
BASE-RTN4 vs IF-RTN4
```

---

# 16. Main Questions the Final Analysis Must Answer

The final `SUMMARY.md` must answer only these questions.

## Question 1

How different are the two quantized checkpoints?

Report:

```text
global qcode difference ratio
global dequantized-weight L1/L2 difference
fraction of groups with different scale
fraction of groups with different zero-point
```

---

## Question 2

Where are the differences located?

Report the strongest locations at:

```text
block
block × module
output row
RTN group
```

Do not state that a location is fingerprint-important merely because its numerical difference is large.

---

## Question 3

Which differences are causally related to fingerprint preservation?

Use only the swap experiments for this conclusion.

Report:

```text
block swaps that reduce verification
module swaps that reduce verification
group swaps that reduce verification
```

Numerical parameter difference alone is not causal evidence.

---

# 17. Recommended Final Repository Outputs

```text
results/
├── rtn4_behavior_control.csv
├── rtn4_exact_diff_coordinates.parquet
├── rtn4_exact_diff_groups.parquet
├── rtn4_diff_by_tensor.csv
├── rtn4_diff_by_block.csv
├── rtn4_diff_by_module.csv
├── rtn4_diff_by_block_module.csv
├── rtn4_diff_by_row.parquet
├── rtn4_diff_groups_top_by_qcode_ratio.csv
├── rtn4_diff_groups_top_by_l2.csv
├── rtn4_diff_groups_top_by_maxdiff.csv
├── rtn4_block_swap.csv
├── rtn4_module_swap.csv
├── rtn4_group_swap.csv
└── rtn4_reverse_swap_candidates.csv    # optional

plots/
├── rtn4_qcode_diff_ratio_by_block.png
├── rtn4_block_module_qcode_diff_heatmap.png
├── rtn4_block_module_l2_diff_heatmap.png
└── rtn4_group_qcode_diff_distribution.png

RTN4_BASE_VS_IF_SUMMARY.md
```

---

# 18. Suggested Implementation Structure

Keep the code small and analysis-focused.

```text
scripts/
├── compare_rtn4_exact.py
├── aggregate_rtn4_differences.py
├── run_rtn4_block_swap.py
├── run_rtn4_module_swap.py
├── run_rtn4_group_swap.py
└── summarize_rtn4_analysis.py

src/
├── rtn4_quant_state.py
├── rtn4_difference.py
├── swap_utils.py
└── fingerprint_eval.py
```

Reuse the existing quantization and fingerprint-evaluation code wherever possible.

Do not reimplement the IF verifier unnecessarily.

---

# 19. Implementation Checks

Before accepting the results, verify all of the following.

### Quantization consistency

```text
same RTN4 function
same bits = 4
same group_size = 128
same asymmetric min/max logic
same zero-point logic
same excluded modules
same dtype
same model tensor discovery
```

### Tensor matching

For every compared tensor:

```text
same tensor name
same shape
same module type
same block index
```

If a tensor cannot be matched, log it explicitly.

Do not silently skip mismatches.

### Swap correctness

After each swap:

1. verify that only the intended block/module/group changed;
2. confirm all other quantized weights remain bitwise/numerically identical to the original `IF-RTN4` state;
3. run fingerprint evaluation;
4. discard/reload the model before the next independent swap if necessary.

Do not accidentally accumulate swaps across experiments.

---

# 20. Minimal Execution Order

Run in this order:

```text
1. Quantize BASE and IF with exactly the same RTN4 path.
2. Reproduce BASE-RTN4 = 0/8 and IF-RTN4 = 8/8.
3. Dump exact RTN4 quantization states.
4. Build coordinate-level and group-level difference maps.
5. Aggregate by tensor/block/module/row/group.
6. Generate the four required plots.
7. Run one-block-at-a-time BASE→IF swaps.
8. For affected blocks, run module swaps.
9. For affected modules, run group swaps.
10. Write RTN4_BASE_VS_IF_SUMMARY.md.
```

---

# 21. Interpretation Rules

Use conservative wording.

### If differences are widespread

Conclude:

> The residual RTN4 difference between BASE and IF remains distributed across many regions.

Do not claim a sparse fingerprint carrier.

### If differences are numerically concentrated

Conclude:

> The residual RTN4 parameter difference is concentrated in specific regions.

Do not claim these regions preserve fingerprint until swap experiments support it.

### If swapping a region reduces verification

Then it is valid to say:

> The corresponding BASE-RTN4 vs IF-RTN4 difference is functionally involved in maintaining the fingerprint under RTN4.

### If one group swap alone breaks fingerprint

This is a particularly strong result because it localizes a necessary residual difference to a concrete:

```text
block
module
output row
RTN group
```

### If no single group/block/module swap changes verification

Do not interpret that as failure.

It indicates that fingerprint preservation may depend on distributed or redundant residual differences.

Stop this phase there rather than introducing an uncontrolled combinatorial search.

---

# 22. Short Instruction for Codex

The key requirement is:

> **Compare BASE-RTN4 and IF-RTN4 directly. First map their exact residual differences after the same asymmetric RTN4 quantization. Then localize those differences structurally. Finally, replace IF-RTN4 regions with their BASE-RTN4 counterparts to determine which residual quantized differences are causally necessary for fingerprint preservation. Do not design a new quantization method and do not add RTN3 or unrelated analyses.**

The analysis must distinguish:

```text
integer code difference
quantization-grid difference
actual dequantized-weight difference
```

and the group indexing must follow the real RTN implementation:

```text
one output row × one contiguous 128-input-weight group.
```
