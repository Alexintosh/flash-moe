# iOS Port Overview

This document summarizes the Flash-MoE iOS port. For full details, see:
- [FlashMoE-iOS/IOS_PORT.md](../FlashMoE-iOS/IOS_PORT.md) -- complete porting story, problems solved, architecture
- [FlashMoE-iOS/397B_ANALYSIS.md](../FlashMoE-iOS/397B_ANALYSIS.md) -- 397B on iPhone: memory budget, Metal limits, K-reduction quality

## What We Built

A native SwiftUI iOS app that runs Qwen3.5 MoE models on iPhone, sharing 100% of the C/Metal inference engine with the macOS CLI via unity build (`#include "infer.m"`).

### Results

| Device | Model | K | tok/s | Notes |
|--------|-------|---|-------|-------|
| iPhone 17 (12GB, A19) | Qwen3.5-35B-A3B | 8 | **5.5** | Full quality, full GPU path |
| iPhone 17 (12GB, A19) | Qwen3.5-35B-A3B (tiered) | 8 | **5.5+** | 13.4GB download, same quality |
| iPhone 17 (12GB, A19) | Qwen3.5-35B-A3B | 4 | **~11*** | *Projected with K=4 + all optimizations (CMD merge, fused expert, prefetch) |
| iPhone 17 (12GB, A19) | Qwen3.5-397B-A17B | 4 | ~0.003 | CPU fallback only (Metal 4GB buffer limit) |

iPhone achieves 57% of laptop speed on the 35B model with 17% of the memory. With K=4 and all Expert Settings optimizations enabled, projections reach ~11 tok/s on the 35B.

## iOS App Features

### Chat Interface
- Streaming token display with typing animation
- Message bubbles with text selection (long press to copy)
- Collapsible `<think>` blocks (DisclosureGroup)
- Special token stripping (`<|endoftext|>`, `<|im_end|>`, `<|im_start|>`)
- Tap outside keyboard to dismiss
- Auto-scroll to latest message via `ScrollViewReader` + `scrollTo(:anchor:.bottom)`
- New chat / reset conversation
- KV cache reuse across conversation turns (continuation mode)

### Model Management
- On-device model scanning (Documents directory)
- HuggingFace download catalog with per-model K recommendations
- Background URLSession downloads with progress tracking
- Swipe-to-delete for downloaded models
- Import from Files app (UIDocumentPickerViewController) with bookmark or move-to-Documents
- Export/Move model to Files app for cross-app access
- Model info sheet (layers, experts, hidden dim, vocab, file sizes)

### Expert Settings
All settings include info modals with plain-language analogies and technical explanations. Compact UI layout with info icon to the left of each label.

- **Active Experts (K)** — K value picker (2-10) for K-reduction (fewer experts = less I/O, lower quality)
- **I/O Fanout** — Chunks picker (off/2/4/8) for splitting expert reads into parallel chunks
- **CMD1+CMD2 Merge** — Combine GPU command buffers for linear attention layers (saves one sync per layer)
- **Fused Attention** — Single-kernel FlashAttention-style online softmax (replaces 3-dispatch pipeline)
- **Fused Expert Kernel** — Combined gate+up+SwiGLU in one Metal dispatch per expert
- **Expert Prefetch** — Overlap next-layer expert I/O with current-layer GPU compute
- **FP16 Accumulation** — Experimental half-precision accumulation in dequant kernels (default OFF)
- **FP8 KV Cache** — 4x KV memory reduction via FP8 E4M3 quantization
- **Max Context Length** — Selector from 4K to 32K positions (Auto mode uses `os_proc_available_memory()`)
- **Sliding Window** — Circular KV buffer for full attention layers (0/2048/4096/8192)
- **Thinking Mode** — Enable/disable `<think>` chain-of-thought with configurable budget
- **H2O Budget** — Heavy Hitter Oracle KV eviction (coming soon)
- Max generation tokens bumped from 500 to 2048

### Profiler
- Resource monitoring overlay
- Thermal state indicator (Cool/Warm/Hot/Critical)
- Temperature display in Celsius
- TTFT display in minutes when >500s
- tok/s and tokens generated counters

### Quantization Support
- 4-bit experts (full quality, production)
- 2-bit experts (faster, breaks JSON/tool calling)
- Tiered quantization (4-bit hot / 2-bit cold experts, auto-detected)

### Context Management
- **FP8 KV Cache** — FP8 E4M3 quantization: float32 (4 bytes) to uint8 (1 byte) per element. Per-position dynamic scales in separate Metal buffers. 4x memory reduction (~60KB to ~15KB per position for 397B). GPU inline dequant in fused attention kernel. FP8/sliding window flags are set BEFORE `metal_setup()` to ensure correct buffer allocation.
- **Sliding Window Attention** — Circular KV buffer for full attention layers. Only the 10 full attention layers are windowed; the 30 GatedDeltaNet layers maintain full context via 128x128 state matrices. With window 4096 + FP8: fixed 40MB KV regardless of conversation length.
- **Max Context Length** — Configurable from 4K to 32K. Auto mode uses `os_proc_available_memory()` to pick the largest safe value. Memory cost: num_full_attn_layers x 2 x kv_heads x head_dim x bytes_per_elem x positions.
- **H2O KV Cache Eviction** (in progress) — Heavy Hitter Oracle: tracks cumulative attention scores, protects sink tokens + recent tokens, evicts low-scoring positions. Replaces sliding window when both are configured. See [context-optimization.md](context-optimization.md).

### Custom URL Download
- Paste any HuggingFace model URL (e.g. `mlx-community/Qwen3.5-35B-A3B-4bit`) in the download section
- URL is validated and config.json is fetched to verify Qwen3.5 MoE compatibility
- Custom models appear in the download list alongside catalog entries
- Downloaded models are hidden from the catalog (no duplicate entries)
- Trash icon removed from catalog download rows

### Universal App Support
- Same SwiftUI shell compiles for both iPhone and Mac destinations
- `#if os(iOS)` conditional compilation for platform-specific UI (toolbar, keyboard dismiss, document picker)
- C inference engine, Metal shaders, and Swift bridge are fully cross-platform
- No code fork — one codebase serves both platforms
- macOS sandbox entitlements: app-sandbox, extended-virtual-addressing, increased-memory-limit, user-selected file read-write, network client

### Model Management
- On-device model scanning (Documents directory)
- HuggingFace download catalog with per-model K recommendations
- Background URLSession downloads with progress tracking
- Swipe-to-delete for downloaded models
- Import from Files app (UIDocumentPickerViewController) with bookmark or move-to-Documents
- Export/Move model to Files app for cross-app access
- Model info sheet (layers, experts, hidden dim, vocab, file sizes)
- Custom HuggingFace URL download (see above)

## Architecture

```
SwiftUI (UI + @Observable state)
    -> Swift async bridge (AsyncStream<GenerationToken>)
        -> Objective-C wrapper (FlashMoEEngine.h C API)
            -> C inference engine (7,500+ lines, unity build)
                -> Metal GPU shaders (1,300 lines)
```

### Engine C API (FlashMoEEngine.h)

```
flashmoe_create()                  -- allocate context
flashmoe_load(ctx, config)         -- load model, allocate Metal resources
flashmoe_generate(ctx, prompt, max_tokens, callback, user_data)
flashmoe_generate_continuation()   -- reuse KV cache for multi-turn
flashmoe_cancel()                  -- thread-safe cancellation
flashmoe_reset()                   -- clear KV cache and position
flashmoe_unload()                  -- release model resources
flashmoe_destroy()                 -- free context
flashmoe_get_stats()               -- model info + generation stats
flashmoe_validate_model()          -- check model directory validity
flashmoe_turn_count()              -- conversation turn count
flashmoe_last_error()              -- human-readable error string
```

### Key Files

```
FlashMoE-iOS/
  FlashMoEEngine/
    FlashMoEEngine.h       -- C API (create/load/generate/cancel/reset/destroy)
    FlashMoEEngine.m       -- Unity build wrapping infer.m (#define CHAT_MODE 1)
  Bridge/
    FlashMoEBridge.swift   -- @Observable async Swift wrapper
  Views/
    ChatView.swift         -- Streaming chat UI with thinking disclosure
    ModelListView.swift    -- Model discovery + download catalog
    ModelDownloadRow.swift -- Download progress with pause/resume
    ProfilerView.swift     -- Resource monitoring overlay
  Services/
    DownloadManager.swift  -- Background URLSession model downloads
  Models/
    ModelCatalog.swift     -- HuggingFace model registry with K recommendations
  App/
    FlashMoEApp.swift      -- SwiftUI app entry point
  IOS_PORT.md              -- Full porting documentation
  397B_ANALYSIS.md         -- 397B memory/performance analysis
  project.yml              -- XcodeGen config (iOS 18+, iPhone only)
  copy_model_to_iphone.sh  -- Push models to device over USB
```

## iOS-Specific Constraints and Solutions

### Metal 4GB Per-Buffer Limit

iOS Metal buffers cannot exceed 4096 MB regardless of entitlements. The 35B model weights (~2.5GB) fit; the 397B weights (~5.5GB) do not.

**Attempted workarounds (all failed):**
- Single 5.5GB Metal buffer -- Metal assertion crash
- Two overlapping ~3GB Metal buffers -- OOM kill (8GB shared memory on 12GB device)
- 50MB staging buffer with memcpy per dispatch -- data corruption from in-flight command buffer aliasing
- CPU fallback -- works but 6 min/token

**Solution:** Split `model_weights.bin` into two <4GB files at the Python packing stage (pending implementation).

### Memory Management
- Adaptive context length via `os_proc_available_memory()` -- reduces 262144 to 8192 based on available memory
- Wired memory budget: `recommendedMaxWorkingSetSize` constrains Metal buffer totals to stay within device budget
- KV cache sizing: `MAX_SEQ_LEN` (1M) replaced with runtime `g_kv_seq_len` (4096) per cache
- FP8 KV cache: 4x memory reduction when opted in (`bytes_per_elem = g_use_fp8_kv ? 1 : sizeof(float)`)
- Expert mmap disabled on iOS -- jetsam kills from 112GB mapped address space
- Debug vs Release: Metal debug wrappers add ~2GB overhead, must build Release for on-device testing
- `isExcludedFromBackup` on all model files to prevent iOS purging 200GB+ of data

### OOM Prevention (8 Protections)

1. **Memory pressure dispatch source** — `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE` handler cancels generation on `DISPATCH_MEMORYPRESSURE_CRITICAL`. This is actionable (sets `atomic_store(&ctx->cancelled, 1)`), not just logging.
2. **`didReceiveMemoryWarning` observer** — `UIApplicationDidReceiveMemoryWarningNotification` as a second line of defense, also cancels generation.
3. **Pre-flight 500MB check** — `os_proc_available_memory() < 500MB` returns error before starting generation (checked in both `flashmoe_generate` and `flashmoe_generate_continuation`).
4. **Adaptive context length** — Runtime context cap based on available memory and Metal wired budget at model load time.
5. **Pre-allocated scratch buffers** — 30+ static scratch buffers allocated once at model load, reused across all layers per token. Eliminates ~300 malloc/free per token.
6. **Metal buffer nil checks** — All 40+ `newBufferWithLength` calls checked for nil with actionable error messages and early return.
7. **calloc guards** — All CPU allocations checked for NULL with error reporting.
8. **posix_memalign for expert I/O** — 2MB-aligned expert data buffers with error checking.

See [docs/oom-prevention.md](../docs/oom-prevention.md) for the full architecture document.

### ARC Cleanup
MetalCtx `free()` without nil-ing `id<>` Objective-C fields caused heap corruption on model switch. Fix: nil all `id<>` fields before `free`.

### 2-Bit Auto-Detection
iOS load path was missing 2-bit directory check. Added auto-detection in `flashmoe_load()`.

## 397B on iPhone -- What We Tried

| Approach | Result | Notes |
|----------|--------|-------|
| Metal 4GB buffer workarounds | All failed | See above |
| K=2 on 397B (trained K=10) | Gibberish | 20% of trained expert capacity |
| K=4 on 397B (trained K=10) | Degenerate ("!!!!") | 40% capacity insufficient |
| K=6+ on 397B | Untested | Needs GPU path (split weights) |
| File Provider Storage | +latency | File coordination overhead on every pread |

### Performance Projections (After Split Weights Enable GPU Path)

| Configuration | Expert I/O | Expected tok/s |
|--------------|-----------|----------------|
| K=10, 4-bit | 4.1 GB/token | ~0.5 |
| K=4, 4-bit | 1.6 GB/token | ~1.0 |
| K=4, tiered | 1.1 GB/token | ~1.2 |

## GPTQ 2-bit: Path to 397B on iPhone

The GPTQ quantization pipeline opens a realistic path to running the full 397B model on 256GB iPhones.

**Key insight**: The 20% hot 4-bit + 80% GPTQ 2-bit tiered configuration produces a **134GB** model. This fits on a 256GB iPhone with room for the OS, apps, and user data. Unlike RTN 2-bit (which breaks JSON/tool calling), GPTQ 2-bit uses Hessian-guided error compensation to preserve output quality at 2-bit precision.

| Component | Size | Notes |
|-----------|------|-------|
| Hot experts (20%, 4-bit) | ~42 GB | Top ~25% by activation frequency, full quality |
| Cold experts (80%, GPTQ 2-bit) | ~87 GB | GPTQ error compensation fixes JSON output |
| Non-expert weights | 5.5 GB | Needs split into two <4GB files for Metal, or CPU fallback |
| **Total** | **~134 GB** | Fits on 256GB iPhone |

**Expected quality**: Production-grade JSON and tool calling. GPTQ's column-wise error compensation keeps accumulated quantization error bounded even at 2-bit, avoiding the `\name\` corruption seen with RTN 2-bit.

**Remaining requirements**:
- Split `model_weights.bin` into two <4GB files (Metal per-buffer limit on iOS)
- Alternatively, use CPU fallback for non-expert weight projections (5.5GB exceeds 4GB Metal limit)
- Test K=6+ with GPU path on device for coherent output
- Validate GPTQ 2-bit JSON quality on real tool-calling workloads

See [docs/quantization-guide.md](quantization-guide.md) for the full GPTQ pipeline documentation.

## Next Steps

1. **Split `model_weights.bin` into two <4GB files** -- enables GPU path on iOS for 397B
2. **Test K=6/8/10 with GPU path** -- find minimum viable K for coherent 397B output
3. **Upload split 397B model to HuggingFace**
4. **Adaptive K** -- auto-select based on device RAM and thermal state
5. **Thermal throttling awareness** -- monitor `ProcessInfo.ThermalState`, reduce K when throttling
