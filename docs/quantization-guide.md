# Quantization Guide: DWQ, JANG, and the GPTQ Pipeline

This document covers the Flash-MoE quantization pipeline, including the DWQ (Data-aware Weight Quantization) and JANG (Jang Adaptive N-bit Grading) approaches, their relationship, and the full GPTQ-based implementation.

## DWQ vs JANG

DWQ and JANG solve different problems and are **complementary, not competing**. JANG decides HOW MANY bits each tensor gets. DWQ decides HOW to use those bits optimally.

### What is DWQ (Data-aware Weight Quantization)?

DWQ is GPTQ-style calibration that uses real input statistics (the Hessian proxy H = X^T @ X) to guide error compensation during quantization. Instead of independently rounding each weight to its nearest quantization level (RTN -- Round To Nearest), DWQ processes weights column-by-column and compensates downstream columns for each rounding error. The compensation is weighted by input importance: weights that participate in high-variance input dimensions receive more aggressive error correction.

The result: same number of bits, but the quantized matrix produces outputs much closer to the original.

### What is JANG (Jang Adaptive N-bit Grading)?

JANG assigns different bit widths to different tensors or layer types based on their sensitivity to quantization error. The core insight is that not all weights are equally important:

- **Router weights** (expert gating): extremely sensitive to quantization -- small errors flip expert selection entirely. Assigned **8-bit**.
- **Attention projections** (Q, K, V, O): moderate sensitivity. Assigned **4-bit**.
- **Hot experts** (top ~25% by activation frequency): handle ~80% of tokens. Assigned **4-bit**.
- **Cold experts** (remaining ~75%): infrequently activated. Assigned **2-bit**.

### Comparison

| | DWQ (GPTQ-style) | JANG (Adaptive N-bit) |
|---|---|---|
| **What it decides** | How to quantize (optimal scale/bias values) | How many bits per tensor |
| **Error reduction type** | Mathematical -- minimizes output reconstruction error | Structural -- allocates bits where they matter most |
| **Our 2-bit problem** | Fixes broken JSON at 2-bit by compensating rounding errors | Avoids the problem by keeping sensitive tensors at higher bit widths |
| **Model size impact** | None -- same bit width, better values | Large -- mixed precision reduces total size |
| **Offline cost** | Moderate -- needs calibration data + Hessian computation | Low -- needs expert frequency profiling |

### Why Neither Alone Is Sufficient

- **JANG 2-bit without DWQ**: Cold experts at 2-bit with naive RTN quantization still produce broken JSON. The `\name\` instead of `"name"` problem persists because RTN 2-bit has too much uncorrelated rounding error.
- **DWQ 4-bit without JANG**: Every expert stays at 4-bit. The model is 209GB. No size reduction.

### The Ideal Pipeline

```
JANG first (architecture decision)  -->  DWQ/GPTQ second (math optimization)
  "Which tensors get 2-bit?"               "Make those 2-bit values as good as possible"
```

JANG is the structural decision: assign bit budgets based on sensitivity. DWQ is the mathematical optimization: within each bit budget, find the quantized values that minimize output error. Together, they produce a model that is both smaller (JANG) and higher quality (DWQ).

## Quantization Formats

### 4-bit Format

- 8 values packed per `uint32`, LSB-first (bits 0-3 = value 0, bits 4-7 = value 1, etc.)
- Scale and bias per group of 64 values, stored as `bf16`
- Dequantization: `value = uint4_nibble * scale + bias`

### 2-bit Format

- 16 values packed per `uint32`, LSB-first (bits 0-1 = value 0, bits 2-3 = value 1, etc.)
- Scale and bias per group of 64 values, stored as `bf16`
- Dequantization: `value = uint2_pair * scale + bias`

### bf16 Conversion

Scale and bias are stored as `bf16` (bfloat16). Conversion from `float32` is a simple truncation: `bf16 = float32_bits >> 16`. No rounding is applied.

### RTN vs GPTQ: Same Format, Different Values

The on-disk format is identical for RTN and GPTQ quantized weights. Both produce the same packed uint4/uint2 values with bf16 scale and bias per group. The Metal dequant formula is the same. The only difference is in the scale/bias values themselves -- GPTQ produces values that minimize output reconstruction error rather than per-weight rounding error.

## Our Implementation: 4-Phase Pipeline

### Phase 0: MSE-Optimal Clipping

Before any GPTQ calibration, we apply optimal clipping to the RTN baseline. For each group of 64 values, a grid search over 20 clipping ratios finds the ratio that minimizes mean squared error between the original float32 values and their quantized reconstruction. This alone achieves 15-30% RMSE reduction over naive min/max scaling.

Implemented in `repack_experts_2bit.py`.

### Phase 1: Calibration Collection

Collect real input activations for each expert to build the Hessian proxy.

1. Run inference with `--collect-activations` flag. This dumps expert input vectors (the x fed into each expert MLP) to disk.
2. `build_hessian.py` accumulates H = X^T @ X per expert in an online fashion (no need to store all activations in memory).
3. `calibrate.sh` orchestrates the collection across diverse prompts. Minimum 16K tokens recommended for stable Hessian estimates.

### Phase 2: GPTQ Requantization

`gptq_requantize.py` applies the blocked GPTQ algorithm to requantize experts from 4-bit to 2-bit using the collected Hessians.

Key details:
- **Block size**: 128 columns. Processing in blocks amortizes the Cholesky factorization cost and improves numerical stability.
- **Automatic fallback**: Experts without calibration data (never activated during calibration) fall back to MSE-optimal clipping from Phase 0.
- **Safety check**: GPTQ output is only used if it beats RTN RMSE. If the Hessian is degenerate or the calibration data is insufficient, the RTN result is kept.

### Phase 3: Sensitivity Analysis

`sensitivity_analysis.py` computes a sensitivity score per expert:

```
sensitivity = frequency * quant_error * layer_weight
```

- `frequency`: how often the expert is activated (from profiling)
- `quant_error`: RMSE between float32 and quantized output
- `layer_weight`: earlier layers are weighted higher (errors compound through the network)

Experts are ranked by sensitivity. The most sensitive experts are assigned 4-bit until the target disk budget (in GB) is reached. The rest get GPTQ 2-bit. The output is a `hot_experts.json` manifest consumed by `repack_experts_tiered.py`.

The updated `repack_experts_tiered.py` accepts `--gptq-dir` (directory of GPTQ-requantized 2-bit experts) and `--hot-experts` (JSON manifest from sensitivity analysis) to build the final tiered model.

## GPTQ Algorithm (Technical Details)

The GPTQ algorithm minimizes the layer-wise reconstruction error:

```
argmin_Q  || W @ X - Q @ X ||^2
```

where W is the original weight matrix, Q is the quantized matrix, and X is the calibration input. This is equivalent to minimizing the error weighted by the Hessian H = X^T @ X.

### Step-by-Step

1. **Collect calibration activations** X per expert (Phase 1).
2. **Compute Hessian proxy** H = X^T @ X. This is a d_in x d_in matrix capturing input correlations.
3. **Add damping**: H += lambda * mean(diag(H)) * I. Prevents numerical instability from near-zero eigenvalues.
4. **Cholesky factorize** H to get H_inv efficiently.
5. **Process columns left-to-right in blocks of 128**:
   - For each column j in the block:
     - Quantize w_j to the nearest quantization level: q_j = quantize(w_j)
     - Compute the quantization error: e_j = w_j - q_j
     - Compensate remaining columns: w_{j+1..n} += e_j * H_inv[j, j+1..n] / H_inv[j, j]
6. **Net effect**: Errors in important input dimensions (high H diagonal values) are compensated more aggressively. Weights connected to correlated inputs are adjusted together.

### Why GPTQ Works Better Than RTN

RTN (Round To Nearest) independently rounds each weight to the nearest quantization level. This is locally optimal per weight but globally suboptimal because it ignores correlations.

GPTQ accounts for weight-to-weight correlations through the Hessian. When weight w_j is rounded down, GPTQ adjusts w_{j+1..n} to compensate, with the adjustment magnitude guided by input statistics. Weights connected to high-variance inputs get larger corrections. The result: same number of bits, but the quantized matrix produces outputs much closer to the original when evaluated on real inputs.

At 4-bit, RTN is already quite good (the quantization grid is fine enough). At 2-bit, the grid is coarse (only 4 levels), and RTN's uncorrelated rounding errors accumulate catastrophically -- this is why RTN 2-bit breaks JSON output. GPTQ's error compensation keeps the accumulated error bounded.

## Full Pipeline Commands

```bash
# Step 1: Calibration -- collect expert input activations
cd metal_infer && ./calibrate.sh

# Step 2: GPTQ requantization -- apply blocked GPTQ to 2-bit experts
python gptq_requantize.py --hessian-dir calibration/ --parallel 8

# Step 3: Sensitivity analysis -- rank experts, decide 4-bit vs 2-bit
python sensitivity_analysis.py --packed-dir ../packed_experts/ --target-gb 150

# Step 4: Build tiered model -- assemble final mixed-precision expert pack
python ../repack_experts_tiered.py --hot-experts hot_experts.json --gptq-dir packed_experts_gptq_2bit/
```

### What Each Step Produces

| Step | Input | Output |
|------|-------|--------|
| calibrate.sh | Diverse prompts + inference engine | `calibration/` directory with per-expert Hessian matrices |
| gptq_requantize.py | 4-bit experts + Hessians | `packed_experts_gptq_2bit/` directory |
| sensitivity_analysis.py | Packed experts + frequency data | `hot_experts.json` manifest |
| repack_experts_tiered.py | 4-bit experts + GPTQ 2-bit experts + manifest | `packed_experts_tiered/` directory (production model) |

## Model Size Estimates

| Configuration | Size | Quality | Notes |
|---------------|------|---------|-------|
| All 4-bit | 209 GB | Excellent | Current production. Full tool calling. |
| All RTN 2-bit | 120 GB | Broken JSON | `\name\` instead of `"name"` in JSON output |
| All GPTQ 2-bit | 120 GB | Good (expected) | GPTQ error compensation fixes JSON |
| 20% hot 4-bit + 80% GPTQ 2-bit | 134 GB | Very good (expected) | Best size/quality tradeoff |

The 20/80 tiered configuration with GPTQ is the target for production deployment on storage-constrained devices (256GB iPhones, smaller SSDs).

## Implementation Status

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 0: MSE-Optimal Clipping | **Done** | Working, 15-30% RMSE reduction |
| Phase 1: Calibration Collection | **Done** | Calibration data collected, Hessians built |
| Phase 2: GPTQ Requantization | **Done** | Blocked GPTQ working; inline script also created |
| Phase 3: Sensitivity Analysis | **In Progress** | Script exists, 64 vs 256 expert count mismatch needs fixing |

### Inline GPTQ Script

`gptq_tiered_inline.py` provides a single-script pipeline that combines Hessian accumulation, GPTQ requantization, and tiered repacking. Key features:

- **Batch Hessian accumulation**: Processes calibration data in batches to fit in memory
- **Estimated runtime**: ~2 hours for the 35B model on a single GPU
- **bf16 overflow issue**: Identified with 256+ experts (397B model). The Hessian accumulation can overflow bf16 range when expert count is large. Not yet fixed -- workaround is to use float32 accumulation, but this increases memory.

### Known Issues

- **64 vs 256 expert count mismatch**: The tiered repacking script assumes 64 experts per layer (122B model) but the 35B model has 256 experts per layer. The `--hot-experts` manifest needs to handle variable expert counts.
- **bf16 overflow with 256+ experts**: Hessian diagonal values can exceed bf16 range (~65K) when accumulating over many tokens. Affects the 35B model (256 experts) but not the 122B (64 experts).
- **Modal cloud repacking**: Works for 4-bit models. Tiered repacking on Modal needs fixes for path handling.

## Files

| File | Location | Purpose |
|------|----------|---------|
| `gptq_requantize.py` | `metal_infer/` | Blocked GPTQ requantization with automatic RTN fallback |
| `gptq_tiered_inline.py` | `metal_infer/` | Single-script inline GPTQ pipeline with batch Hessian accumulation |
| `build_hessian.py` | `metal_infer/` | Online Hessian accumulation (H = X^T @ X) per expert |
| `sensitivity_analysis.py` | `metal_infer/` | Expert sensitivity scoring and bit-width assignment |
| `calibrate.sh` | `metal_infer/` | Calibration runner (diverse prompts, 16K+ tokens) |
| `repack_experts_2bit.py` | `metal_infer/` | RTN 2-bit repacking with MSE-optimal clipping (Phase 0) |
| `repack_experts_tiered.py` | root | Tiered repacking with `--gptq-dir` and `--hot-experts` support |
