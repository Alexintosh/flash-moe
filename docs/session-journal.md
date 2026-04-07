# Session Journal

Complete record of the development session. Covers all optimizations implemented, features built, bugs fixed, research analyzed, and current status of every feature branch.

## Optimizations Implemented

### Delta-Net Kernel Fusion
- **What:** Merged pass 2 and pass 3 in `gated_delta_net_step()` into a single pass
- **Impact:** Eliminates ~1M device memory reads per token (45 linear attention layers)
- **Status:** Done, merged to develop
- **Branch:** `feature/delta-net-fusion`

### CMD1+CMD2 Merging
- **What:** Combine GPU command buffers CMD1 (attention projections) and CMD2 (o_proj + norm + routing + shared) for linear attention layers into a single command buffer, eliminating one GPU sync point per layer
- **Impact:** Saves 2.25-4.5ms/token across 45 linear attention layers
- **Status:** Done, but was fixed twice
- **Bug #1:** First implementation had a fallback clobber bug -- the `else` branch for full-attention layers re-ran o_proj with a sentinel pointer, producing NaN output
- **Bug #2:** After the fix, needed to ensure the merge only applies to linear attention layers (not full attention layers which need separate CMD1/CMD2 for different dispatch patterns)
- **Branch:** `feature/combined-fixes` (merged with other fixes)

### Modular Decomposition
- **What:** Split the monolithic 8081-line `infer.m` into 9 focused header modules
- **Files:** `config.h`, `timing.h`, `fp8.h`, `weights.h`, `cpu_kernels.h`, `metal_ctx.h`, `gpu_dispatch.h`, `expert_io.h`, `layer_forward.h`, `generate.h`
- **Impact:** 0% performance impact, major maintainability improvement
- **Status:** Done, merged to develop

### Dynamic SIMD Width
- **What:** Added `[[threads_per_simdgroup]]` parameter to all dequant Metal kernels instead of hardcoding SIMD width = 32
- **Impact:** Future-proofing for Apple hardware with non-32 SIMD width
- **Status:** Done, merged to develop

### FP8 KV Cache
- **What:** Opt-in quantization of KV cache from float32 to FP8 E4M3 (1 sign, 4 exponent, 3 mantissa bits)
- **Impact:** 4x KV memory reduction (~60KB to ~15KB per position for 397B)
- **Key detail:** FP8/sliding window flags must be set BEFORE `metal_setup()` to ensure correct buffer allocation
- **Status:** Done, merged to develop

### Fused Online Softmax Attention
- **What:** Single-kernel FlashAttention-style implementation replacing the 3-dispatch pipeline (Q@K^T, softmax, scores@V)
- **Impact:** Reduces 3 GPU dispatches to 1 per full-attention layer
- **Bug fixed:** Unnormalized accumulator -- online softmax accumulator was not properly rescaled when the running max changed. Fixed by multiplying accumulator by `exp(old_max - new_max)` on each max update.
- **Status:** Done, merged to develop

### FP16 Accumulation
- **What:** Optional half-precision accumulation in dequant matvec kernels. Apple GPU has dedicated fp16 ALUs at 2x throughput.
- **Impact:** Theoretical 2x compute throughput, but fp16 has only ~3 decimal digits of precision
- **Risk:** Sums of 512+ elements may lose accuracy
- **Status:** Done, default OFF. Toggle in Expert Settings.
- **Branch:** `feature/fp16-accumulation`

### Metal Function Constants
- **What:** Compile-time specialization of fused attention kernel via `[[function_constant(0)]]`
- **Impact:** Zero overhead for disabled code paths (FP8 vs float32)
- **Status:** Done, merged to develop

### Wired Memory Limit
- **What:** Query Metal's `recommendedMaxWorkingSetSize` at startup to constrain GPU buffer totals
- **Impact:** Prevents Metal from evicting buffers to system memory (severe latency spikes)
- **Status:** Done, merged to develop

### OOM Prevention
- **What:** Comprehensive allocation hardening: 30+ pre-allocated scratch buffers, 40+ Metal nil checks, calloc guards, posix_memalign checks, iOS memory pressure handler, 500MB pre-flight check
- **Impact:** Eliminates ~300 malloc/free per token, prevents OOM crashes
- **Status:** Done, merged to develop

### Sliding Window Attention
- **What:** Circular KV buffer for full attention layers. Write position cycles via `cache_pos = kv->len % window_size`
- **Impact:** Fixed memory KV regardless of conversation length (e.g., 40MB with window 4096 + FP8)
- **Status:** Done, merged to develop

### H2O KV Cache Eviction
- **What:** Heavy Hitter Oracle eviction -- tracks cumulative attention scores, protects sink tokens + recent tokens, evicts low-scoring positions
- **Impact:** Smarter than sliding window -- keeps important tokens regardless of age
- **Status:** Implemented, UI toggle coming soon

### RoPE Scaling
- **What:** Three context extension methods: Linear, NTK-aware, YaRN
- **Impact:** Extends effective context length beyond model's native training length
- **Status:** Done
- **Branch:** `feature/rope-scaling`

### Expert Double Buffering
- **What:** Two I/O buffers (buf_A, buf_B) allow overlapping I/O for next layer with GPU compute for current layer
- **Bug fixed:** Data race where CMD3 read buf_B while prefetch I/O was still writing to it. Added I/O completion fence.
- **Status:** Done
- **Branch:** `feature/fix-expert-prefetch`

### Fused Expert Kernel
- **What:** Combined gate+up+SwiGLU in one Metal dispatch per expert
- **Bug fixed:** Divergent `simd_sum` across threads = undefined behavior. Threads in same SIMD group took different branches. Fixed to ensure uniform control flow before SIMD reduction.
- **Status:** Done
- **Branch:** `feature/fix-fused-expert`

### Batched Prefill
- **What:** GEMM-based prompt processing for all 40 layers with 13 new Metal kernels
- **Impact:** 17.3 tok/s prefill throughput on iPhone 17 (58 tokens in 3357ms)
- **Kernels added:** batched RoPE, batched RMS norm, batched attention (Q@K^T with causal mask), batched SwiGLU, batched MoE combine
- **Bug #1:** KV cache sync -- GPU wrote K/V during prefill but CPU mirror not updated
- **Bug #2:** Conv output stride mismatch in batched conv1d kernel for GatedDeltaNet
- **Status:** Working, post-projection GPU kernels need more testing
- **Branch:** `feature/batched-prefill-v2`

### Expert Callback APIs
- **What:** C function pointer callbacks for distributed inference (flashswarm)
- **Callbacks:** `expert_read_cb`, `expert_compute_cb`, `expert_bitmap_cb`, `expert_timing_cb`
- **Impact:** Enables offloading expert I/O and compute to remote nodes
- **Status:** Done
- **Branch:** `feature/expert-callback-api`

### GPU Background Handling
- **What:** iOS background/foreground lifecycle management for Metal resources
- **Impact:** Prevents jetsam kills from background GPU memory pressure
- **Status:** Done
- **Branch:** `feature/gpu-background-handling`

### OpenAI-Compatible API Server
- **What:** `--openai-api` flag starts HTTP server implementing `/v1/chat/completions`
- **Impact:** Integration with any OpenAI-compatible client
- **Status:** Done
- **Branch:** `feature/openai-api`

## iOS App Features

### Chat Interface
- Streaming token display with typing animation
- Message bubbles with text selection (long press to copy)
- Collapsible `<think>` blocks (DisclosureGroup)
- Special token stripping
- Auto-scroll, keyboard dismiss
- New chat / reset conversation
- KV cache reuse across turns

### Model Management
- On-device model scanning
- HuggingFace download catalog with per-model K recommendations
- Background URLSession downloads with pause/resume
- Swipe-to-delete, import from Files, export to Files
- Custom HuggingFace URL download
- Model info sheet
- Reload model button

### Expert Settings (15+ toggles)
Organized in collapsible sections (Speed, GPU Pipeline, Context, Generation):
- Active Experts (K), I/O Fanout, Expert Prefetch
- CMD1+CMD2 Merge, Fused Attention, Fused Expert Kernel, FP16 Accumulation
- FP8 KV Cache, Max Context Length, Sliding Window, H2O Budget, RoPE Scaling
- Thinking Mode, Thinking Budget, Max Generation Tokens (2048)
- Each toggle has info modal with analogy + technical details

### Benchmark Mode
- Configurable test matrix: prompts, token counts, settings combos
- Records tok/s, TTFT, total time, memory per config
- Card-based results UI with sorting
- `BenchmarkView.swift`

### Profiler Overlay
- Thermal state, temperature, TTFT, tok/s, tokens generated

### Universal App
- Same codebase for iPhone and Mac
- `#if os(iOS)` for platform-specific UI
- macOS sandbox entitlements

## Quantization Work

### Tiered RTN 2-bit
- **Status:** Working, production on HuggingFace (13.4GB for 35B)
- Hot experts (top ~25%) at 4-bit, cold experts requantized to 2-bit
- 34% disk reduction, no quality loss for hot path

### GPTQ Pipeline
- **Phase 0 (MSE-Optimal Clipping):** Done, 15-30% RMSE reduction
- **Phase 1 (Calibration):** Done, calibration data collected, Hessians built
- **Phase 2 (GPTQ Requantization):** Done, blocked GPTQ working
- **Phase 3 (Sensitivity Analysis):** In progress, 64 vs 256 expert count mismatch

### Inline GPTQ Script
- `gptq_tiered_inline.py`: single-script pipeline with batch Hessian accumulation
- Estimated ~2h for 35B model
- bf16 overflow issue with 256+ experts (identified, not fixed)

### Modal Cloud Repacking
- 4-bit repacking works on Modal
- Tiered repacking needs path fixes

## Research Analysis Performed

### Vulkan Fork (fluxism/flash-moe-vulkan)
- Extracted 4 optimization phases, all implemented:
  1. Delta-net kernel fusion
  2. CMD1+CMD2 merging
  3. Modular decomposition
  4. Dynamic SIMD width
- Key finding: GPU linear attention was already in our code

### vllm-metal
- Adopted: wired memory limit API, FP8 KV cache, fused attention pattern
- Learned: Metal function constants for compile-time specialization

### jangq
- Documented JANG + DWQ approach (complementary, not competing)
- JANG decides bit assignment, DWQ optimizes within each assignment
- Informed our GPTQ pipeline design

### reap-expert-swap
- Page cache priming idea evaluated
- Conclusion: "Trust the OS" is better -- custom priming adds overhead

### vmlx
- JANG loader architecture studied
- SSD streaming comparison: similar approach, different implementation

### Anemll Fork
- Batched prefill approach adopted
- PR #4 KV cache fix ported to our implementation
- Key insight: layer-first ordering for prefill

### TurboQuant
- In research queue, not yet analyzed

## Feature Branches Ready for Testing

| Branch | Feature | Status |
|--------|---------|--------|
| `feature/fix-fused-expert` | SIMD reduction fix in fused expert kernel | Ready |
| `feature/fix-expert-prefetch` | buf_B data race fix in expert double buffering | Ready |
| `feature/expert-callback-api` | Distributed inference C callback APIs | Ready |
| `feature/gpu-background-handling` | iOS background pause/resume for Metal | Ready |
| `feature/batched-prefill-v2` | Batched GEMM + partial GPU post-projection | Ready (needs more testing) |
| `feature/rope-scaling` | Linear/NTK/YaRN context extension | Ready |
| `feature/openai-api` | OpenAI-compatible HTTP server | Ready |
| `feature/combined-fixes` | Merged fix branches for testing together | Ready |

## Known Issues

1. **Batched prefill post-projection GPU kernels** -- need more on-device testing for correctness
2. **GPTQ tiered repacking: 64 vs 256 expert count mismatch** -- sensitivity analysis script assumes 64 experts per layer
3. **bf16 overflow in GPTQ with 256+ experts** -- Hessian diagonal exceeds bf16 range
4. **Expert prefetch toggle** -- missing from some feature branch builds
5. **Bundle ID reverts in merges** -- must always be `com.alexintosh.flashmoe`
6. **Model reloads twice** -- reload button triggers double load from state change race
7. **Thinking content visible when OFF** -- `<think>` content needs stripping when thinking disabled
8. **H2O UI toggle** -- implemented in engine but Expert Settings toggle not yet wired up

## Performance Summary

| Platform | Model | Config | tok/s | Notes |
|----------|-------|--------|-------|-------|
| iPhone 17 (A19, 12GB) | 35B tiered | K=4, Fused Att, CMD Merge, Fanout 2 | **11** | Best measured |
| iPhone 17 (A19, 12GB) | 35B tiered | Batched prefill (58 tokens) | **17.3 prefill** | TTFT improvement |
| iPhone 17 (A19, 12GB) | 35B tiered | K=8, baseline | 5.5 | Default config |
| MacBook M3 Max (48GB) | 397B 4-bit | K=4 | 4.4 | Production |
| MacBook M3 Max (48GB) | 35B 4-bit | Autoresearch optimizations | 9.7 | +34.7% from baseline |
| iPhone 17 (A19, 12GB) | 397B | Any config | OOM | 5.5GB weights exceed Metal 4GB limit |
