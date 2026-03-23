# Plan: Apply Vulkan Fork Learnings to Flash-MoE

## Context

Analysis of the [Vulkan fork](https://github.com/fluxism/flash-moe-vulkan) revealed several techniques worth adopting. The fork achieves 0.75 tok/s on AMD RDNA 3.5 (vs our 4.4 tok/s on M3 Max) — the gap is mostly hardware (discrete GPU + PCIe vs unified memory + 17.5 GB/s NVMe).

**Key finding: GPU linear attention is already implemented** in our codebase. The 5 Metal kernels (`gated_delta_net_step`, `conv1d_step`, `rms_norm_qk`, `compute_decay_beta`, `gated_rms_norm`) exist in `shaders.metal` and are enabled by default (`gpu_linear_attn_enabled = 1`). The Vulkan fork's highest-impact optimization — porting linear attention to GPU — was already done.

The remaining opportunities are incremental optimizations and a major maintainability refactor.

---

## Phase 1: Delta-Net Kernel Fusion -- COMPLETE

**Impact: +3-5% on linear attention layers (~0.5-1% overall tok/s)**
**Effort: 1 commit, ~50 lines**

The `gated_delta_net_step` kernel (`shaders.metal:1057`) makes 3 passes over the 128×128 state matrix per thread:

| Pass | Operation | Memory |
|------|-----------|--------|
| 1 | Decay + compute kv_mem | Read+write state, read k |
| 2 | Delta update | Read+write state, read k |
| 3 | Output query | Read state, read q |

Pass 2 and 3 can be fused — after updating `S[vi][ki] += delta`, the state is final and can immediately compute `out += S[vi][ki] * q[ki]` in the same loop. This eliminates one full pass over the state matrix: 128 reads per thread × 128 threads × 64 heads = ~1M fewer device memory reads per token.

### Changes
- `metal_infer/shaders.metal` (~line 1057): Add `gated_delta_net_step_fused` kernel
- `metal_infer/infer.m` (~line 1494): Pipeline creation for fused kernel
- `metal_infer/infer.m` (~line 4656): Use fused pipeline in GPU dispatch

### Verification
```bash
./infer --prompt "Hello" --tokens 50 --timing
# Compare cmd1_wait times before/after
```

---

## Phase 2: CMD1+CMD2 Merging for Linear Attention Layers -- COMPLETE

**Impact: +1-2% tok/s by eliminating one GPU sync point**
**Effort: 1 commit, ~100 lines**

For linear attention layers with GPU path enabled, the CPU phase between CMD1 and CMD2 is empty — the GPU already computed everything in CMD1. Currently:

```
CMD1 (commit+wait) → CPU (empty for linear attn) → CMD2 (commit+wait) → routing → CMD3
```

CMD2 (o_proj + residual + norm + routing) can be appended to CMD1 as additional encoders with pipeline barriers:

```
CMD1+CMD2 (commit+wait) → routing → CMD3
```

Savings: ~0.05-0.1ms per layer × 45 linear attention layers = 2.25-4.5ms per token.

### Changes
- `metal_infer/infer.m` (~lines 4600-5500): When `gpu_linear_attn == 1`, continue encoding CMD2 dispatches into CMD1 instead of creating a new command buffer.

### Verification
```bash
./infer --prompt "Hello" --tokens 50 --timing
# cmd1_wait + cmd2_wait should merge into single wait phase
```

---

## Phase 3: Modular Decomposition of infer.m -- COMPLETE

**Impact: 0% tok/s, major maintainability improvement**
**Effort: 5-8 commits, ~200 lines changed (mostly file moves)**

Split the 8081-line monolith into focused modules. The Vulkan fork's 6-file decomposition (`vk_compute.c`, `io_ring.c`, `weights.c`, `linear_attn.c`, `full_attn.c`, `infer.c`) demonstrates the right boundaries. The iOS unity build (`#include "infer.m"`) must continue to work.

### Proposed split

| New File | Content | Approx Lines |
|----------|---------|-------------|
| `config.h` | ModelConfig struct, constants, macros | ~200 |
| `timing.h` | LayerTimingAccum, timing functions | ~250 |
| `weights.m` | WeightFile, JSON parser, tensor lookup, layer cache | ~1200 |
| `cpu_kernels.m` | cpu_rms_norm, cpu_softmax, cpu_topk, BLAS delta-net fallback | ~700 |
| `metal_setup.m` | MetalCtx struct, metal_setup(), buffer allocation | ~600 |
| `gpu_dispatch.m` | gpu_encode_batch_matvec, gpu_flush_batch_results | ~600 |
| `expert_io.m` | Parallel pread, GCD dispatch, I/O pool | ~700 |
| `layer_forward.m` | fused_layer_forward, deferred expert completion | ~2000 |
| `generate.m` | Inference loop, sampling, chat, CLI args | ~1600 |
| `infer.m` | Unity build: `#include` all above in order | ~50 |

### Approach
1. Extract pure headers first (`config.h`, `timing.h`) — no compilation changes
2. Extract pure functions (`cpu_kernels.m`) — no static global dependencies
3. Extract Metal modules (`metal_setup.m`, `gpu_dispatch.m`) — share globals via `extern`
4. Update Makefile for individual `.m` compilation on macOS
5. Keep `infer.m` as unity include for iOS

### Critical constraint
Many functions use `static` globals (`g_metal`, `g_timing`, `cfg`, `g_deferred`). These need `extern` declarations in a shared header with definitions in one translation unit.

### Verification
```bash
make clean && make && ./infer --prompt "Hello" --tokens 20 --timing
# Output must be byte-identical to pre-refactor with fixed seed
```

---

## Phase 4: Dynamic SIMD Width in Shaders (Future-proofing) -- COMPLETE

**Impact: 0% on current hardware, prevents breakage on future Apple Silicon**
**Effort: 1 commit, ~30 lines**

Replace hardcoded SIMD assumptions:
- `ROWS_PER_TG = 8` → `threads_per_threadgroup / simd_size`
- `threadgroup float q_shared[4]` → `[128 / simd_size]`
- Use Metal's `[[threads_per_simdgroup]]` attribute

Low risk — Apple Silicon is always SIMD 32 today, this is purely defensive.

---

## Execution Summary

All 4 phases have been completed.

| # | Phase | Effort | tok/s Impact | Status |
|---|-------|--------|-------------|--------|
| 1 | Delta-Net Kernel Fusion | ~50 lines | +0.5-1% | **Complete** |
| 2 | CMD1+CMD2 Merging | ~100 lines | +1-2% | **Complete** |
| 3 | Modular Decomposition | 5-8 commits | 0% (maintenance) | **Complete** |
| 4 | Dynamic SIMD | ~30 lines | 0% (future-proof) | **Complete** |
| | **Total** | | **+1.5-3% tok/s** | |

### Phase 3 Final Structure

The 8081-line `infer.m` was split into 9 focused modules, all included via unity build:

| File | Content | Lines |
|------|---------|-------|
| `config.h` | ModelConfig struct, constants, macros | 438 |
| `timing.h` | Timing, telemetry, tracking globals | 256 |
| `weights.h` | Tensor manifest, hash table, mmap, bf16 conversion | 205 |
| `cpu_kernels.h` | Vocabulary, tokenizer, CPU compute kernels | 387 |
| `metal_ctx.h` | MetalCtx, metal_setup(), buffer management | 603 |
| `gpu_dispatch.h` | BatchMatvecSpec, batched GPU matmul, expert forward | 721 |
| `expert_io.h` | I/O thread pool, parallel pread, cache | 827 |
| `layer_forward.h` | RoPE, KVCache, attention, MoE, fused pipeline | 3068 |
| `generate.h` | Inference loop, sampling, HTTP serve, main() | 1717 |
| `infer.m` | Unity build entry point (#includes all above) | 86 |

The iOS unity build (`FlashMoEEngine.m` -> `#include "infer.m"`) continues to work unchanged.

---

## What We're NOT Doing (and why)

| Technique | Why Not |
|-----------|---------|
| Vulkan integration | Metal is optimal for Apple Silicon |
| io_uring | Apple doesn't support it; GCD + pread is equivalent |
| Pipeline cache to disk | Metal compilation is already fast (<1ms) |
| Full float32 shared memory | Half precision saves occupancy, 12% speedup proven |
| Sub-allocation / memory arena | Not needed with Metal's simpler buffer model |

---

## What We Can Offer the Vulkan Fork

1. **Half-precision shared memory** in dequant kernel — halves shared mem usage
2. **58 failed experiments** — saves weeks of dead-end exploration (LZ4, mmap, GPU LUT, temporal prediction, speculative routing, spin-poll)
3. **Trust the OS page cache** — 71% hit rate, every custom cache was slower
4. **Delta-net kernel fusion** (Phase 1 above) — applies to their GLSL kernel too
5. **FMA kernel details** — maps to single GPU FMA instruction per element
