# Flash-MoE: Running Massive MoE Models on a Laptop

> **[Read the paper](paper/flash_moe.pdf)** — Full technical details, 90+ experiments, and the story of how an AI and a human built this in 24 hours.

Pure C/Metal inference engine for **Qwen3.5 Mixture-of-Experts** models on Apple Silicon. Runs models from 35B to 397B parameters on machines with as little as 24GB RAM, streaming expert weights from SSD through a custom Metal compute pipeline.

No Python runtime. No frameworks. Just C, Objective-C, and hand-tuned Metal shaders. Model architecture is auto-detected from HuggingFace `config.json` — switch models with a single `--model` flag.

## Compatible Models

Any **Qwen3.5 MoE** model with MLX quantization (`model_type: qwen3_5_moe`) is supported. Use the model manager to discover and download compatible models:

| Model | Params | Active | Quant | Disk | Min RAM |
|-------|--------|--------|-------|------|---------|
| Qwen3.5-35B-A3B | 35B | 3B | 4-bit | ~18GB | 24GB |
| Qwen3.5-35B-A3B | 35B | 3B | 8-bit | ~35GB | 48GB |
| Qwen3.5-122B-A10B | 122B | 10B | 4-bit | ~65GB | 48GB |
| Qwen3.5-397B-A17B | 397B | 17B | 4-bit | ~209GB | 48GB |
| Qwen3.5-397B-A17B | 397B | 17B | 6-bit | ~280GB | 64GB |
| Qwen3.5-397B-A17B | 397B | 17B | 8-bit | ~397GB | 96GB |

The engine auto-detects architecture, dimensions, expert counts, quantization, and layer types from `config.json`. No recompilation needed.

## Development Workflow (MANDATORY)

**NEVER merge feature branches into `develop` without explicit approval.** The flow is:

1. **Create `feature/branch-name`** from `develop`
2. Implement, commit, push to the feature branch
3. **Notify** the user that it's ready for testing
4. User tests on device and requests changes if needed
5. Iterate on the feature branch until the user confirms it works
6. **Only merge when the user explicitly says "merge it"**

**Why:** Merging untested code (batched prefill, fused expert, CMD merge) caused repeated gibberish output and crashes that wasted hours of debugging and device restarts. QA happens on the feature branch, not on develop.

**Also:**
- Always add a **settings toggle** (default OFF for experimental features) so broken features can be disabled without reverting code
- Before any `git rebase` or destructive git operation, **ask the user first**
- The `develop` branch should always produce correct output

## Results

![Progress](progress.png)

### MacBook Pro M3 Max (48GB)

| Configuration | tok/s | Quality | Notes |
|--------------|-------|---------|-------|
| 4-bit + autoresearch optimizations | **9.7** | Excellent | +34.7% from half-precision x_shared + SIMD + FMA |
| 4-bit experts, FMA kernel | **4.36** | Excellent | Before autoresearch. Full tool calling. 209GB on disk. |
| **Tiered (hot=4bit, cold=2bit)** | **4.36+** | **Excellent** | **33% smaller on disk. Auto-detected.** |
| 2-bit experts, trust OS | 5.74 | Good* | 120GB on disk. *Breaks JSON/tool calling. |

### iPhone 17 (12GB, A19)

| Model | K | tok/s | Notes |
|-------|---|-------|-------|
| Qwen3.5-35B-A3B (4-bit) | 8 | **5.5** | 19.5GB download. Full quality. Full GPU path. |
| Qwen3.5-35B-A3B (tiered) | 8 | **5.5+** | 13.4GB download. Same quality. |
| Qwen3.5-397B-A17B (4-bit) | 4 | ~0.003* | *CPU fallback only — Metal 4GB per-buffer limit blocks GPU path. |
| Qwen3.5-397B-A17B (4-bit) | 4 | ~1-2** | **Projected with split weight files enabling GPU path. |

*2-bit quantization produces `\name\` instead of `"name"` in JSON output, making tool calling unreliable. 4-bit is the production configuration.

**Tiered mode** keeps frequently-activated experts (top ~25%) at 4-bit quality while requantizing cold experts to 2-bit — reducing disk footprint by ~34% without quality loss. Hot experts are profiled from real workloads. See [docs/tiered-expert-quantization.md](docs/tiered-expert-quantization.md) for the full experiment writeup.

## Hardware

### Development (MacBook Pro)
- **Machine**: MacBook Pro, Apple M3 Max
- **Chip**: 16-core CPU (12P + 4E), 40-core GPU, 16-core ANE
- **Memory**: 48 GB unified (~400 GB/s bandwidth)
- **SSD**: 1TB Apple Fabric, **17.5 GB/s sequential read** (measured)
- **macOS**: 26.2 (Darwin 25.2.0)

### Mobile (iPhone 17)
- **Chip**: A19, ~10-core GPU
- **Memory**: 12 GB unified
- **SSD**: ~2.5-3 GB/s NVMe
- **iOS**: 18+

## Architecture

Qwen3.5 MoE models use a hybrid attention architecture with GatedDeltaNet (linear attention) and standard full attention layers, each containing a Mixture-of-Experts MLP. Model dimensions, expert counts, and layer types vary per model and are read from `config.json` at startup. For example, the 397B model has 60 layers (45 linear + 15 full), 512 experts (K=4 active), hidden dim 4096; the 35B model has 40 layers (30 linear + 10 full), 256 experts (K=8 active), hidden dim 2048.

### Key Techniques

1. **SSD Expert Streaming** — Expert weights (209GB at 4-bit) are read from NVMe SSD on demand via parallel `pread()` with GCD dispatch groups. Only the K=4 active experts per layer are loaded (~6.75MB each). The OS page cache manages caching — no custom cache needed ("Trust the OS" principle). Inspired by Apple's "LLM in a Flash" paper.

1. **Tiered Expert Quantization** — Expert usage follows a Zipfian distribution: ~25% of experts handle ~80% of activations. Hot experts stay at 4-bit; cold experts are requantized to 2-bit (44% smaller each). This shrinks total expert disk by ~34%, improving OS page cache hit rates without quality degradation. Per-expert Metal kernel dispatch selects the right dequant shader at runtime.

2. **FMA-Optimized Dequant Kernel** — The inner loop of the 4-bit dequantized matrix-vector multiply rearranges the math from `(nibble * scale + bias) * x` to `fma(nibble, scale*x, bias*x)`. Pre-computing `scale*x` and `bias*x` lets the GPU fused multiply-add unit do dequant+multiply in one instruction. 12% faster than the naive formulation.

3. **Metal Compute Shaders** — Hand-written Metal kernels for:
   - 4-bit and 2-bit dequantized matrix-vector multiply (tiled, SIMD-reduced, shared input cache, FMA-optimized)
   - Fused SwiGLU activation
   - RMS normalization (two-pass: sum-of-squares reduction + apply)
   - Batched GPU attention (Q@K^T, softmax, scores@V) for full attention layers
   - GPU RoPE (fused with Q deinterleave and K normalization)
   - MoE combine + residual + sigmoid gate (fused kernel)

4. **Deferred GPU Expert Compute** — CMD3 (expert forward pass) is submitted without waiting. The GPU executes it while the CPU prepares the next layer. The combine + residual + norm are also on GPU, feeding directly into the next layer's attention projections.

5. **Accelerate BLAS for Linear Attention** — The GatedDeltaNet recurrence uses `cblas_sscal`, `cblas_sgemv`, and `cblas_sger` for the 64-head × 128×128 state matrix update. 64% faster than scalar code.

6. **Trust the OS** — No custom expert cache. The OS page cache (~35GB) manages expert data caching via standard LRU. Every custom caching approach we tested (Metal LRU, malloc cache, LZ4 compressed cache) was slower due to GPU memory pressure or overhead. The page cache achieves ~71% hit rate naturally.

7. **K-Reduction for Mobile** — MoE models have a natural inference knob: activate fewer experts per token than trained. K=4 instead of K=10 on the 397B model cuts I/O by 60% per token with graceful quality degradation. The router still picks the *best* K experts from all 512 — it's not random subsampling. This makes the 397B viable on iPhone (12GB RAM, 2.5 GB/s NVMe) at ~1-2 tok/s.

8. **Half-Precision Shared Memory** — Storing the threadgroup input cache as `half` instead of `float` in Metal dequant kernels halves shared memory usage (16KB → 8KB), doubling GPU occupancy. Since inputs are already approximate from 4-bit dequantization, the precision loss is negligible. +12% tok/s on the main kernel.

9. **iOS Unity Build** — The entire 7,500-line inference engine compiles into the iOS app via `#include "infer.m"`. No fork, no separate codebase. A thin C API (`FlashMoEEngine.h`) wraps the static globals, and a Swift `@Observable` bridge provides `AsyncStream<Token>` generation with automatic memory-adaptive context sizing.

10. **FP8 E4M3 KV Cache** — Opt-in quantization of the KV cache from float32 to FP8 E4M3 (1 sign, 4 exponent, 3 mantissa bits). Per-position dynamic scales stored separately. Reduces KV memory from ~60KB/position to ~15KB/position for the 397B model (4x reduction), enabling longer context on memory-constrained devices. GPU inline dequant in the fused attention kernel reads FP8 bytes and scales on the fly. Enabled with `--fp8-kv` flag; default off to preserve float32 precision. Note: FP8/sliding window flags must be set BEFORE `metal_setup()` to ensure correct buffer allocation.

11. **Fused Online Softmax Attention** — Single-kernel FlashAttention-style implementation replaces the previous 3-dispatch pipeline (Q@K^T, softmax, scores@V) for full-attention layers. Iterates over KV positions in blocks of `BLOCK_SIZE=64`, maintaining online softmax state (running max `m`, running sum `l`, output accumulator `o`) per head. Each block computes partial scores, updates the running statistics, and rescales the accumulator — never materializing the full attention matrix. Reduces 3 GPU dispatches to 1 per full-attention layer.

12. **Metal Function Constants** — Compile-time specialization of the fused attention kernel via Metal `[[function_constant(0)]]`. The `USE_FP8_KV` boolean constant eliminates dead branches at pipeline creation time, so the FP8 dequant path has zero overhead when disabled and the float32 path has zero overhead when FP8 is active. Both variants share a single source kernel (`fused_attention_fc`).

13. **OOM Prevention** — Comprehensive allocation hardening across the engine: 30+ static scratch buffers pre-allocated at model load (eliminates ~300 malloc/free per token), all 40+ Metal buffer allocations checked for nil with actionable error messages, `calloc` guards on all CPU allocations with early-return on failure, `posix_memalign` for 2MB-aligned expert I/O buffers with error checking. On iOS: dispatch-source memory pressure handler cancels generation on critical pressure, `didReceiveMemoryWarning` observer as a second line of defense, pre-flight 500MB availability check before every generation call, and adaptive context length sizing via `os_proc_available_memory()`. See [docs/oom-prevention.md](docs/oom-prevention.md).

14. **Wired Memory Limit** — Metal's `recommendedMaxWorkingSetSize` API queried at startup and stored in `MetalCtx.recommended_working_set`. Used to constrain KV cache allocation so GPU buffer totals stay within the device's wired memory budget, preventing Metal from evicting buffers to system memory (which causes severe latency spikes).

15. **Universal App** — The SwiftUI shell compiles for both iPhone and Mac (via "Designed for iPad" / Mac Catalyst compatibility). Views use `#if os(iOS)` conditional compilation for platform-specific UI (toolbar placement, keyboard dismiss, document picker). The same C inference engine, Metal shaders, and Swift bridge run on both platforms without modification.

16. **Sliding Window Attention** — Circular KV buffer for full attention layers. Write position cycles via `cache_pos = kv->len % window_size`. The 30 GatedDeltaNet linear attention layers maintain full context through their 128x128 state matrices (O(1) memory), while only the 10 full attention layers are windowed. With window 4096 + FP8: fixed 40MB KV regardless of conversation length. Enabled via `--sliding-window N` flag.

17. **FP16 Accumulation** (experimental) — Optional half-precision accumulation in dequant matvec kernels. Apple's GPU has dedicated fp16 ALUs at 2x throughput. The FMA becomes `fma(half(nibble), half(scale*x), half(bias*x))` with final promotion to float32 via `simd_sum`. Risk: fp16 has ~3 decimal digits of precision; sums of 512+ elements may lose accuracy. Default OFF; toggle in Expert Settings.

18. **H2O KV Cache Eviction** (in progress) — Heavy Hitter Oracle eviction for the full-attention KV cache. Tracks cumulative post-softmax attention scores per position. When the cache exceeds the budget, it protects sink tokens (first N, typically 4) and recent tokens (25% of budget), then keeps the highest-scoring "heavy hitter" positions. Compacts both CPU and GPU caches in-place so GPU kernels see a shorter contiguous sequence. Replaces sliding window when both are configured (H2O is strictly better). See [docs/context-optimization.md](docs/context-optimization.md).

19. **Custom HuggingFace URL Download** — Users can paste any HuggingFace model URL (e.g. `mlx-community/Qwen3.5-35B-A3B-4bit`) in the iOS/Mac app to resolve and download compatible models not in the built-in catalog. The URL is validated, config.json is fetched to verify compatibility, and the model is added to the download list.

20. **macOS Sandbox Entitlements** — The universal app includes sandbox entitlements for file access (`com.apple.security.files.user-selected.read-write`), networking (`com.apple.security.network.client`), extended virtual addressing, and increased memory limits.

21. **Paper-Guided Autoresearch v2** — Automated experiment loop that reads the research paper, identifies optimization opportunities, implements them, benchmarks with quality gates, and logs results. See `autoresearch/program_v2.md`.

### GPTQ/JANG Quantization Pipeline

A 4-phase pipeline for producing high-quality 2-bit experts using GPTQ (Data-aware Weight Quantization) with optional JANG (Jang Adaptive N-bit Grading) mixed-precision assignment. GPTQ uses calibration data to build a Hessian proxy (H = X^T @ X) per expert, then applies blocked column-wise error compensation during quantization. The result: same 2-bit format, but output reconstruction error is dramatically lower than RTN (Round To Nearest). This fixes the broken JSON problem at 2-bit.

**Phases:**
- **Phase 0: MSE-Optimal Clipping** — Grid search over 20 clipping ratios per group of 64 values. 15-30% RMSE reduction. In `repack_experts_2bit.py`.
- **Phase 1: Calibration Collection** — `--collect-activations` flag dumps expert input vectors. `build_hessian.py` accumulates H = X^T @ X per expert online. `calibrate.sh` runner. 16K tokens minimum.
- **Phase 2: GPTQ Requantization** — `gptq_requantize.py`. Blocked GPTQ (block_size=128). Automatic fallback to MSE-clip for uncalibrated experts. Safety check: only uses GPTQ if it beats RTN RMSE.
- **Phase 3: Sensitivity Analysis** — `sensitivity_analysis.py` computes freq x quant_error x layer_weight per expert. Assigns 4-bit to most sensitive experts until target GB budget. `repack_experts_tiered.py` updated with `--gptq-dir` and `--hot-experts` flags.

See [docs/quantization-guide.md](docs/quantization-guide.md) for the full technical writeup including the GPTQ algorithm, DWQ vs JANG comparison, and pipeline commands.

### Expert Settings UI

The iOS/Mac app includes a comprehensive Expert Settings panel with info modals for every toggle. Each setting has an analogy (plain-language explanation) and technical details. The UI uses a compact layout with an info icon to the left of each label. Settings include: Active Experts (K), I/O Fanout, CMD1+CMD2 Merge, Fused Attention, Fused Expert Kernel, Expert Prefetch, FP16 Accumulation, FP8 KV Cache, Max Context Length (4K-32K), Sliding Window, Thinking Mode, and H2O Budget (coming soon). Max generation tokens bumped to 2048. See [docs/expert-settings-guide.md](docs/expert-settings-guide.md).

### Model Management

- Downloaded models are hidden from the download catalog list (no duplicate entries)
- Trash icon removed from download catalog rows
- Custom HuggingFace URL download support
- Import, export, and delete models on-device

### iOS-Specific Constraints

- **Metal 4GB per-buffer limit** — iOS Metal buffers cannot exceed 4096 MB, regardless of entitlements. The 35B model (2.5GB weights) fits in a single buffer. The 397B model (5.5GB weights) does not. Attempted workarounds: two overlapping buffers (OOM), staging buffer with memcpy per dispatch (data corruption from in-flight command buffer aliasing), CPU fallback (works, 6 min/token). Solution: split `model_weights.bin` into two <4GB files at packing time.
- **K-reduction quality varies by model** — K=2 and K=4 on the 397B produce gibberish/degenerate output. The model was trained with K=10 and needs K=6+ for coherence (untested — needs GPU path). The 35B at default K=8 works perfectly.
- **Debug build overhead** — Metal API Validation adds ~2GB of `MTLDebugComputeCommandEncoder` proxies, causing OOM on iPhone. Must build Release for on-device testing.
- **Bundle ID migration** — Switching from personal to paid developer team requires a new bundle ID (Apple takes 24-48h to release old ones). Moving 300GB of model data between app containers requires `UIDocumentPickerViewController` with `.moveToService`.
- **File Provider Storage penalty** — Models accessed via Files app integration go through the file coordination layer, adding I/O latency to every `pread`. Always import models to the app's own Documents directory.
- **`isExcludedFromBackup`** — Must be set on all model files to prevent iOS from purging 200GB+ of data during storage pressure events.

See [FlashMoE-iOS/IOS_PORT.md](FlashMoE-iOS/IOS_PORT.md) for the full iOS porting story and [FlashMoE-iOS/397B_ANALYSIS.md](FlashMoE-iOS/397B_ANALYSIS.md) for the 397B memory/performance analysis.

### Pipeline Per Layer (4.28ms average at 4-bit)

```
CMD3(prev) → CMD1: attention projections + delta-net  [1.22ms GPU]
           → CPU: flush results                       [0.01ms CPU]
           → CMD2: o_proj + norm + routing + shared    [0.55ms GPU]
           → CPU: softmax + topK routing               [0.003ms]
           → I/O: parallel pread K=4 experts           [2.41ms SSD]
           → CMD3: expert forward + combine + norm     [0.04ms encode, DEFERRED]
```

### Unified Memory Constraint

On Apple Silicon, SSD DMA and GPU compute share the same memory controller and cannot be profitably overlapped. The GPU's dequant kernels are bandwidth-saturated at ~418 GiB/s. Even small background SSD DMA causes disproportionate GPU latency spikes through memory controller arbitration. The serial pipeline (GPU → SSD → GPU) is hardware-optimal.

## Model Manager

The model manager helps you find, download, and validate compatible models:

```bash
# List local models and search HuggingFace for compatible ones
python model_manager.py

# Search HuggingFace only
python model_manager.py --search

# List local models only
python model_manager.py --local

# Download a specific model
python model_manager.py --download mlx-community/Qwen3.5-35B-A3B-4bit

# Check if a local model is compatible
python model_manager.py --check /path/to/model
```

After downloading, prepare the model for inference:

```bash
MODEL=~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit

# 1. Build expert index (maps tensor names to byte offsets)
python build_expert_index.py --model $MODEL

# 2. Pack expert weights into per-layer binary files
python repack_experts.py --index expert_index.json

# 3. Extract non-expert weights into a single mmap-friendly binary
python metal_infer/extract_weights.py --model $MODEL

# 4. Run inference (auto-detects weights in model directory)
cd metal_infer && ./infer --model $MODEL --prompt "Hello" --tokens 20
```

Pre-packed models are available on HuggingFace (no repacking needed):
- `alexintosh/Qwen3.5-35B-A3B-Q4-FlashMoE` (19.5 GB, 4-bit)
- `alexintosh/Qwen3.5-35B-A3B-Q4-Tiered-FlashMoE` (13.4 GB, tiered)

### Tiered Expert Quantization (Optional)

Reduces expert disk footprint by ~34% by keeping hot experts at 4-bit and requantizing cold experts to 2-bit. Recommended for memory-constrained setups:

```bash
# 1. Profile expert usage (run a few diverse prompts)
./infer --model <MODEL> --prompt "Explain quantum computing" --tokens 200 --freq 2>&1 | tee /tmp/freq1.txt
./infer --model <MODEL> --prompt "Write a Python function" --tokens 200 --freq 2>&1 | tee /tmp/freq2.txt

# 2. Generate hot expert manifest (80% coverage threshold)
python profile_experts.py --freq-output /tmp/freq1.txt /tmp/freq2.txt --coverage 0.8

# 3. Repack experts (creates packed_experts_tiered/)
python repack_experts_tiered.py --model <MODEL>

# 4. Run with --tiered (or auto-detected if packed_experts_tiered/ exists)
cd metal_infer && ./infer --model <MODEL> --tiered --prompt "Hello" --tokens 20
```

## Quick Start

```bash
cd metal_infer
make

# Run with a specific model (auto-detects architecture from config.json)
./infer --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit \
  --prompt "Explain quantum computing" --tokens 100

# Or set FLASH_MOE_MODEL to avoid passing --model every time
export FLASH_MOE_MODEL=~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit
./infer --prompt "Explain quantum computing" --tokens 100

# 2-bit inference (faster but breaks tool calling)
./infer --prompt "Explain quantum computing" --tokens 100 --2bit

# Tiered mode (hot=4-bit, cold=2-bit, auto-detected if packed_experts_tiered/ exists)
./infer --prompt "Explain quantum computing" --tokens 100 --tiered

# Interactive chat with tool calling (start server first, then chat client)
./infer --serve &
./chat

# Per-layer timing breakdown
./infer --prompt "Hello" --tokens 20 --timing
```

## Project Structure

```
model_manager.py       # Model discovery, download, and compatibility checking
build_expert_index.py  # Build expert_index.json from safetensors (step 1 of packing)
repack_experts.py      # 4-bit expert packing from safetensors (step 2)
profile_experts.py     # Expert frequency profiling → hot_experts.json
repack_experts_tiered.py  # Tiered repacking (hot=4-bit, cold=2-bit)
progress.py            # Results visualization (Q2/Q4 tracks)
results.tsv            # Experiment log (58 experiments)

metal_infer/
  infer.m              # Unity build entry point (86 lines, #includes all modules)
  config.h             # ModelConfig struct, constants, macros (438 lines)
  timing.h             # Timing, telemetry, tracking globals (256 lines)
  fp8.h                # FP8 E4M3 encode/decode, per-tensor dynamic scale, g_use_fp8_kv flag (119 lines)
  weights.h            # Tensor manifest, hash table, mmap, bf16 conversion (205 lines)
  cpu_kernels.h        # Vocabulary, tokenizer, CPU compute kernels (387 lines)
  metal_ctx.h          # MetalCtx, metal_setup(), buffer management, wired memory query (603 lines)
  gpu_dispatch.h       # BatchMatvecSpec, batched GPU matmul, expert forward (721 lines)
  expert_io.h          # I/O thread pool, parallel pread, cache (827 lines)
  layer_forward.h      # RoPE, KVCache, attention, MoE, fused pipeline, scratch buffers (3068 lines)
  generate.h           # Inference loop, sampling, HTTP serve, main() (1717 lines)
  shaders.metal        # Metal compute kernels (~1500 lines, includes fused attention + FP8 variants)
  chat.m               # Interactive chat TUI with tool calling
  tokenizer.h          # C BPE tokenizer (single-header, 449 lines)
  main.m               # MoE-only benchmark
  Makefile             # Build system
  extract_weights.py   # Creates model_weights.bin from safetensors
  repack_experts_2bit.py  # 4-bit → 2-bit expert requantization (with MSE-optimal clipping)
  gptq_requantize.py   # Blocked GPTQ 2-bit requantization with Hessian-guided error compensation
  build_hessian.py     # Online Hessian accumulation (H = X^T @ X) per expert from calibration data
  sensitivity_analysis.py  # Expert sensitivity scoring (freq × quant_error × layer_weight) and bit-width assignment
  calibrate.sh         # Calibration runner — collects expert activations over diverse prompts
  train_predictor.py   # Expert routing prediction analysis
  model_weights.bin    # Non-expert weights (model-specific, mmap'd)
  model_weights.json   # Tensor manifest
  vocab.bin            # Vocabulary for token decoding
  tokenizer.bin        # Pre-exported BPE tokenizer data

FlashMoE-iOS/              # Native iOS app
  FlashMoEEngine/
    FlashMoEEngine.h       # C API (create/load/generate/cancel/reset/destroy)
    FlashMoEEngine.m       # Unity build wrapping infer.m for iOS
  Bridge/
    FlashMoEBridge.swift   # @Observable async Swift wrapper
  Views/
    ChatView.swift         # Streaming chat UI with thinking disclosure
    ModelListView.swift    # Model discovery + download catalog
    ModelDownloadRow.swift # Download progress with pause/resume
    ProfilerView.swift     # Resource monitoring overlay
  Services/
    DownloadManager.swift  # Background URLSession model downloads
  Models/
    ModelCatalog.swift     # HuggingFace model registry with K recommendations
  App/
    FlashMoEApp.swift      # SwiftUI app entry point
  IOS_PORT.md              # Full iOS porting documentation
  397B_ANALYSIS.md         # 397B on iPhone: memory, Metal limits, K-reduction quality
  project.yml              # XcodeGen config (iOS 18+, iPhone only)
  copy_model_to_iphone.sh  # Push models to device over USB (pymobiledevice3)

autoresearch/              # Automated experiment loop
  program.md               # Agent instructions for autonomous optimization
  program_v2.md            # Paper-guided autoresearch v2 instructions
  benchmark.sh             # Measurement harness with quality gates
  prepare.sh               # Baseline setup
  experiments.tsv          # Experiment log
  findings.md              # Autoresearch findings and results

docs/
  context-optimization.md  # FP8 KV + sliding window + H2O context management
  expert-settings-guide.md # All Expert Settings with analogies and technical details
  ios-port.md              # iOS port overview
  optimization-experiments-q4.md  # Q4 optimization experiments
  vulkan-learnings-plan.md # Vulkan fork analysis (all 4 phases complete)
  oom-prevention.md        # OOM prevention architecture
  tiered-expert-quantization.md   # Tiered quantization experiment writeup
  quantization-guide.md  # DWQ/JANG comparison, GPTQ pipeline, quantization formats
```

## What We Tried (and What Worked)

### Autoresearch Wins (Automated Experiment Loop)
| Approach | Result | Impact |
|----------|--------|--------|
| Half-precision x_shared (v3 kernel) | Halve shared mem → 2× occupancy | **+12.1% tok/s** |
| FMA 2-bit dequant kernel | fma(nibble, scale*x, bias*x) | **+6.2% tok/s** |
| Half-precision x_shared (2-bit kernel) | Same occupancy trick | **+3.3% tok/s** |
| SIMD reduction in rms_norm_qk | simd_sum replaces serial loop | **+2.1% tok/s** |

### Vulkan Fork Analysis (Phases 1-4, All Complete)

Analyzed the [Vulkan fork](https://github.com/fluxism/flash-moe-vulkan) and identified 4 optimization phases. Key finding: GPU linear attention was already implemented in our code. All 4 phases have been completed:

| Phase | Optimization | Impact | Status |
|-------|-------------|--------|--------|
| 1 | Delta-net kernel fusion (merge pass 2+3 in gated_delta_net_step) | Eliminates ~1M device memory reads/token | **Done** |
| 2 | CMD1+CMD2 merging for linear attention layers | Saves 2.25-4.5ms/token (45 layers x 1 sync point) | **Done** |
| 3 | Modular decomposition (8081-line infer.m -> 9 focused modules) | 0% perf, major maintainability | **Done** |
| 4 | Dynamic SIMD width (`[[threads_per_simdgroup]]` in all dequant kernels) | Future-proofing for non-32 SIMD hardware | **Done** |

Full analysis: [docs/vulkan-learnings-plan.md](docs/vulkan-learnings-plan.md)

### Kept (Manual)
| Approach | Result | Impact |
|----------|--------|--------|
| FMA dequant kernel | GPU compute -12% | **+12% tok/s** |
| Trust OS page cache | Deleted Metal LRU → +38% | **Foundational** |
| GPU combine+norm in CMD3 | Eliminates CPU round-trip | **Pipeline** |
| BLAS delta-net (Accelerate) | cpu_attn 0.78→0.28ms | **+64% attn** |
| F_NOCACHE for 2-bit | +3% from avoiding page thrash | **2-bit only** |
| GPU fused attention (RoPE) | +2% for full-attn layers | **Small** |
| C BPE tokenizer | 180ms vs 3500ms startup | **20x startup** |
| Deferred CMD3 execution | GPU/CPU overlap | **Pipeline** |
| Tiered expert quant (hot=4b, cold=2b) | -34% disk, same quality | **Cache hit rate** |

### Discarded (58 experiments, highlights)
| Approach | Result | Why |
|----------|--------|-----|
| LZ4 expert compression | -13% | Decompress overhead > warm cache savings |
| F_RDADVISE prefetch | net 0% | Unified memory: SSD DMA slows GPU -73% |
| Temporal expert prediction | -18% | 25% hit rate, SSD bandwidth waste |
| MLP routing predictor | 31% accuracy | Worse than temporal baseline |
| GPU LUT dequant kernel | -2% | Indirect register access serializes |
| GPU private buffer compression | -20% pipeline | Blit cost 4×7MB > matvec savings |
| Spin-poll GPU wait | -23% | CPU thermal competes with GPU |
| Expert file clustering | 0% | NVMe ignores scatter at 7MB granularity |
| dispatch_io | -70% | dispatch_data management overhead |
| mmap expert files | -5x | Per-page fault overhead on cold data |
| Speculative early routing | -38% | Cache pollution + overhead |
| MTP speculative decoding | break-even | MoE I/O scales per-token (unlike dense) |

### Discarded (iOS-specific)
| Approach | Result | Why |
|----------|--------|-----|
| Single 5.5GB Metal buffer (397B weights) | Crash | Metal hard limit: 4096 MB per buffer, not configurable |
| Two overlapping Metal buffers (~3GB each) | OOM kill | Metal tracks ~8GB shared memory on 12GB device |
| 50MB staging buffer + memcpy per dispatch | Data corruption | In-flight command buffers alias single staging buffer; later memcpys overwrite earlier tensor data before GPU reads |
| K=2 on 397B (trained K=10) | Gibberish | Only 20% of trained expert capacity fires, output distribution collapses |
| K=4 on 397B (trained K=10) | Degenerate ("!!!!") | 40% capacity insufficient for 512-expert model |
| File Provider Storage for model access | +latency | File coordination layer adds overhead to every pread |

## Notable Bug Fixes

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| Metal shader loading on iOS | `newDefaultLibrary` returns nil when shaders.metal in Resources instead of Sources | Fall back to pre-compiled metallib in bundle; move shader to Sources build phase |
| KV cache OOM (2GB per cache) | `MAX_SEQ_LEN=1M` used for allocation | Adaptive runtime cap via `os_proc_available_memory()` |
| ARC heap corruption on model switch | MetalCtx `free()` without nil-ing `id<>` fields | Nil all Objective-C fields before `free` |
| Expert mmap jetsam kills on iOS | mmap'ing 112GB of expert files | Disabled expert mmap on iOS, pread-only |
| 2-bit auto-detection missing in iOS | iOS load path skipped 2-bit directory check | Added 2-bit auto-detection in `flashmoe_load()` |
| String format mismatch warnings | `%d` for `size_t`, `%f` for `int` | Corrected format specifiers throughout |
| MAX_K buffer overflow on 397B | Hardcoded `MAX_K=8`, 397B needs K=10 | Bumped to `MAX_K=16` with runtime cap |

## Safety

The engine explicitly controls memory:
- Non-expert weights: model-dependent (e.g., 5.5GB for 397B, ~1.5GB for 35B, mmap'd read-only)
- Metal scratch buffers: ~200MB (desktop), ~500MB (397B on iPhone with reduced context)
- Expert data streams from SSD on demand — no full model load required
- No custom caches. Trust the OS page cache for expert LRU.
- iOS: adaptive context length via `os_proc_available_memory()`, KV caches sized to fit device
- Wired memory budget: `recommendedMaxWorkingSetSize` constrains Metal buffer totals
- OOM prevention: 30+ pre-allocated scratch buffers, 40+ Metal nil checks, calloc guards, posix_memalign checks. iOS adds memory pressure handler, didReceiveMemoryWarning observer, and 500MB pre-flight check. See [docs/oom-prevention.md](docs/oom-prevention.md).
- FP8 KV cache (opt-in): reduces KV memory 4x for longer context on constrained devices
- Minimum RAM: 8GB iPhone (35B), 12GB iPhone (397B with K=4), 24GB Mac (35B), 48GB Mac (397B)
