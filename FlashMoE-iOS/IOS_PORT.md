# Flash-MoE iOS Port: From MacBook to iPhone

How we took a 397B-parameter MoE inference engine written in C/Metal and made it run on an iPhone — getting 5.5 tok/s on the 35B model, with a path to running 397B.

## The Challenge

The desktop Flash-MoE engine was designed for a MacBook Pro with 48GB unified memory, 40-core GPU, and 17.5 GB/s NVMe. iPhones have 8-12GB RAM, ~10-core GPU, ~2.5 GB/s NVMe, and no filesystem access to `shaders.metal` at runtime. Every assumption in the 7,000-line inference engine needed to be re-examined.

## Results

| Device | Model | K | tok/s | Notes |
|--------|-------|---|-------|-------|
| iPhone 17 (12GB) | Qwen3.5-35B-A3B | 8 | **5.5** | Full quality, 19.5GB download |
| MacBook Pro M3 Max (48GB) | Qwen3.5-35B-A3B | 8 | **9.7** | After autoresearch optimizations |
| MacBook Pro M3 Max (48GB) | Qwen3.5-397B-A17B | 4 | **4.4** | K-reduced from K=10 |

iPhone achieves **57% of laptop speed** on the 35B model with 17% of the memory.

## What We Built

A native iOS app (SwiftUI + Objective-C/C) that downloads pre-packed models from HuggingFace and runs them locally with streaming token generation, chat templates, and a full conversational UI.

**Architecture**: Swift (UI + async bridge) → Objective-C wrapper → C inference engine → Metal GPU shaders

## Performance Optimizations (Autoresearch)

Before the iOS port, we ran Karpathy's autoresearch pattern — an autonomous experiment loop that modifies Metal shaders, benchmarks, and keeps/discards based on tok/s. 10 experiments, 4 kept:

| Experiment | Description | Impact | Why It Works |
|-----------|-------------|--------|-------------|
| SIMD reduction | Replace serial thread-0 accumulation in `rms_norm_qk`/`gated_rms_norm` with `simd_sum` + shared memory reduction | +2.1% | Eliminates serial bottleneck in GPU reduction — 4 SIMD groups of 32 threads each contribute partial sums |
| FMA 2-bit kernel | Apply `fma(nibble, scale*x, bias*x)` pattern to 2-bit dequant (same trick as 4-bit v3) | +6.2% | GPU fused multiply-add does dequant+multiply in one instruction instead of two |
| Half-precision x_shared (v3) | Store threadgroup shared memory input cache as `half` instead of `float` | +12.1% | Halves shared memory from 16KB to 8KB → doubles GPU occupancy. Input values are already approximate from dequantization so half precision loses nothing |
| Half-precision x_shared (2-bit) | Same trick applied to 2-bit kernel | +3.3% | Same occupancy benefit, smaller because 2-bit kernel is more I/O bound |

**Combined: +15.3% theoretical, +34.7% real-world** (from 7.2 to 9.7 tok/s on MacBook Pro).

The half-precision x_shared insight was the biggest single win: dequantized values are already approximate (4-bit → float), so storing the shared input cache at half precision loses no meaningful accuracy while dramatically improving GPU core occupancy.

6 experiments were discarded:
- Single compute encoder for batch matvecs (−5%): GPU serializes dispatches that were previously pipelinable
- FMA in matvec_fast without shared memory (−15%): pre-computing scale*x requires reading x twice from device memory
- Extended x_shared to 8192 floats (−7%): 32KB maxes out threadgroup memory, killing occupancy
- Three others with marginal negative or neutral impact

## Problems Solved (iOS Port)

### 1. Metal Shader Loading

**Problem**: The desktop engine compiles `shaders.metal` from source at runtime via `newLibraryWithSource:`. On iOS, there's no filesystem path to the shader file.

**Fix**: Runtime fallback — try `[device newDefaultLibrary]` first (loads pre-compiled `default.metallib` from the app bundle), fall back to source compilation for macOS CLI:

```objc
ctx->library = [ctx->device newDefaultLibrary];
if (ctx->library) {
    // iOS: loaded from bundle
} else {
    // macOS: compile from source
    NSString *src = [NSString stringWithContentsOfFile:@"shaders.metal" ...];
    ctx->library = [ctx->device newLibraryWithSource:src ...];
}
```

**Xcode fix**: Moved `shaders.metal` from the **Resources** build phase to **Sources** so Xcode's Metal compiler produces `default.metallib` in the app bundle.

### 2. Memory: KV Cache OOM

**Problem**: The model's `max_position_embeddings` is 131,072 (128k context). KV cache allocation per full-attention layer: `131072 * kv_heads * head_dim * 4 bytes`. For the 35B (10 full-attn layers): **~2.5GB just for KV caches**. `calloc` silently returns NULL on iPhone, causing `EXC_BAD_ACCESS` crashes.

**Fix**: Adaptive context length based on `os_proc_available_memory()`. Budget 25% of available memory for KV caches, clamp to power-of-2 sizes (512–8192). On a 12GB iPhone this yields ~2048 context, keeping KV caches under ~40MB total. Context window is limited but sufficient for chat.

### 3. Memory: Metal Debug Layer

**Problem**: Debug builds wrap every Metal object with `MTLDebugComputeCommandEncoder` validation proxies, roughly doubling GPU memory usage. The app crashes with `NSMallocException` trying to allocate debug wrappers.

**Fix**: Build in **Release** mode and disable Metal API Validation in the Xcode scheme. The debug overhead is too large for iPhone's memory budget.

### 4. Tokenizer Not Found

**Problem**: `init_tokenizer()` searches for `tokenizer.bin` at relative filesystem paths (`./tokenizer.bin`, `./metal_infer/tokenizer.bin`). These don't exist on iOS.

**Fix**: Extended the search to check the model directory (where it's downloaded) and the app bundle:

```objc
// Try model directory (downloaded with model)
snprintf(model_tok, sizeof(model_tok), "%s/tokenizer.bin", cfg.model_path);
if (access(model_tok, R_OK) == 0) { bpe_load(&g_tokenizer, model_tok); }

// Try app bundle
NSString *bundlePath = [[NSBundle mainBundle] pathForResource:@"tokenizer" ofType:@"bin"];
if (bundlePath) { bpe_load(&g_tokenizer, [bundlePath UTF8String]); }
```

### 5. Missing Info.plist

**Problem**: Xcode target configs had `INFOPLIST_KEY_*` entries but never set `GENERATE_INFOPLIST_FILE = YES`, so no `Info.plist` was produced.

**Fix**: Added `GENERATE_INFOPLIST_FILE = YES` to both Debug and Release target build settings.

### 6. Chat Template (Garbage Output)

**Problem**: The model received raw text ("Hi") instead of Qwen's chat template format. Without the `<|im_start|>` / `<|im_end|>` markers, the model treats input as a continuation of arbitrary text, producing incoherent output.

**Fix**: Added `buildChatPrompt()` in `ChatView.swift` that formats the full conversation history using Qwen's `<|im_start|>system/user/assistant<|im_end|>` template.

### 7. Special Token Leakage

**Problem**: End-of-turn tokens like `<|im_end|>` appear as visible text in the chat UI.

**Fix**: Strip special tokens from the token stream before displaying.

### 8. KV Cache Reuse Across Turns

**Problem**: Each new message re-tokenizes and re-prefills the entire conversation history. On a model generating at 5.5 tok/s, re-prefilling 200+ tokens of history adds seconds of latency per turn.

**Fix**: Added `flashmoe_generate_continuation()` — a new C API that reuses existing KV cache state. The engine tracks conversation position and only processes the new user turn tokens. Falls back to full re-prefill if context fills up.

## K-Reduction: Running 397B on iPhone

### The Insight

Mixture-of-Experts gives a natural knob that dense models don't have: you can activate **fewer experts per token** at inference time, even if the model was trained with more. K=4 instead of K=10 means:

- **60% less I/O per token** (4 expert reads instead of 10, per layer)
- Storage unchanged (all 512 experts still on disk — routing decides which 4 to use)
- Quality degrades gracefully — you're still selecting the *best* 4 from 512 options

### 397B on iPhone: Memory Analysis

```
Non-expert weights (mmap'd):     5.5 GB  (virtual, not all resident)
Metal buffers:                    ~500 MB (KV cache at 2048 ctx + delta-net + expert buffers)
iOS overhead:                     ~2 GB
                                  --------
Resident estimate:                ~3 GB
Available for page cache:         ~9 GB on 12GB iPhone

Expert I/O per token (K=4):      4 × 60 layers × 6.75 MB = 1.6 GB
iPhone NVMe throughput:           ~2.5 GB/s
I/O time per token:               ~0.65s
GPU compute per token:            ~0.3-0.5s (60 layers, 32 heads, head_dim=256)
                                  --------
Expected:                         ~1-1.5 tok/s
```

With tiered experts (hot=4-bit, cold=2-bit): I/O drops ~34%, storage drops from 208GB to ~140GB, expected speed improves to ~1.5-2 tok/s.

### Critical Bug: MAX_K Overflow

The engine hardcodes `#define MAX_K 8` for multi-expert buffer arrays. The 397B model needs K=10. Without K-reduction (or bumping MAX_K), loading the 397B model causes a buffer overflow crash. Fix: bump MAX_K to 16 and add a runtime cap.

## iOS App Architecture

### Engine Layer (C/Objective-C)

- **FlashMoEEngine.m** — iOS wrapper around `infer.m` (unity build via `#include`)
- **infer.m** — The full 7,000-line inference engine, shared with macOS
- **shaders.metal** — Metal compute kernels, compiled into `default.metallib`

### Bridge Layer (Swift/ObjC Interop)

- **FlashMoEBridge.swift** — `@Observable` class wrapping the C API
  - `loadModel(at:)` → background thread → `flashmoe_load()` with adaptive memory config
  - `generate(prompt:)` → `AsyncStream<GenerationToken>` via C callback bridge
  - `generateContinuation(userMessage:)` → reuses KV cache for multi-turn
  - State machine: `idle → loading → ready → generating`

### UI Layer (SwiftUI)

- **ChatView** — Streaming chat with thinking block disclosure, text selection, braille spinner
- **ModelListView** — On-device models + downloadable catalog with auto K-reduction
- **ModelDownloadRow** — Per-model download progress with pause/resume/delete
- **ProfilerView** — Resource monitoring overlay (memory, tok/s, cache stats)

### Model Management

- **ModelCatalog.swift** — Static registry of pre-packed HuggingFace repos with recommended K values
- **DownloadManager.swift** — Background `URLSession` downloads with state persistence

## Key Design Decisions

| Decision | Why |
|----------|-----|
| Unity build (`#include "infer.m"`) | Share 100% of inference code with macOS, no fork to maintain |
| Runtime Metal library fallback | Single codepath works on both iOS (pre-compiled) and macOS (source) |
| Adaptive context cap | Scale KV cache to available memory instead of hardcoded limit |
| K-reduction for large models | MoE's natural knob — fewer experts = proportionally less I/O, no weight modification needed |
| Pre-packed HuggingFace models | No on-device conversion needed — download and run |
| Background URLSession | Downloads survive app suspension (not force-quit) |
| Trust the OS page cache | Same philosophy as desktop — no custom expert cache on iOS either |
| Auto-detect weights in model dir | `--model /path` finds model_weights.bin, vocab.bin etc. automatically |

## Pre-Packed Models on HuggingFace

Models are published pre-packed (repacked experts + extracted non-expert weights) so the iOS app can download and run directly:

| Repo | Model | Quant | Size | iPhone Min |
|------|-------|-------|------|-----------|
| `alexintosh/Qwen3.5-35B-A3B-Q4-FlashMoE` | 35B | 4-bit | 19.5 GB | 128 GB storage, 8 GB RAM |
| `alexintosh/Qwen3.5-35B-A3B-Q4-Tiered-FlashMoE` | 35B | Tiered | 13.4 GB | 128 GB storage, 8 GB RAM |
| `alexintosh/Qwen3.5-397B-A17B-Q4-FlashMoE` | 397B | 4-bit | ~214 GB | 256 GB storage, 12 GB RAM |

Each repo contains:
- `config.json` — Model architecture config
- `model_weights.bin` — Non-expert weights (mmap'd at runtime)
- `model_weights.json` — Tensor name → offset manifest
- `tokenizer.bin` / `tokenizer.json` — Pre-exported BPE tokenizer
- `vocab.bin` — Token vocabulary
- `packed_experts/layer_XX.bin` — One file per layer

### Preparing a New Model

```bash
# 1. Download from HuggingFace
python model_manager.py --download mlx-community/Qwen3.5-35B-A3B-4bit

# 2. Build expert index (maps tensor names to byte offsets in safetensors)
python build_expert_index.py --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit

# 3. Repack experts into per-layer binary files
python repack_experts.py --index expert_index.json

# 4. Extract non-expert weights into a single mmap-friendly binary
python metal_infer/extract_weights.py --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit

# 5. Upload to HuggingFace
# (see upload script in project — stages files with symlinks, uploads via huggingface_hub API)
```

## What's Next

- **Run 397B on iPhone** — needs MAX_K bump, tiered repack, and HuggingFace upload
- **Adaptive K** — auto-select K based on device RAM and model size
- **Thermal throttling awareness** — monitor thermal state and reduce K if throttling
- **Download resumption** — handle interrupted downloads more gracefully
- **Background inference** — continue generation when app is backgrounded
- **Smaller models** — 7B/14B variants for devices with less storage
