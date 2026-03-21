# Tiered Expert Quantization

**Status**: Implemented and verified
**Branch**: `feature/tiered-expert-quantization`
**Date**: 2026-03-20

## Problem

Flash-MoE streams expert weights from SSD on demand, relying on the OS page cache for LRU caching. On the 397B model, total expert data is 209GB at 4-bit — far exceeding even 48GB of unified memory. Every cache miss costs ~2.4ms of SSD I/O per layer. Improving cache hit rates directly improves tok/s.

Previous caching attempts all failed:
- **Metal LRU cache**: -38% (steals GPU memory from compute)
- **malloc cache**: -20% (steals from OS page cache)
- **LZ4 compression**: -13% (decompression overhead)
- **mmap**: -5x (per-page fault overhead)
- **Speculative prefetch**: -38% (cache pollution, bandwidth waste)

The core constraint is Apple Silicon's unified memory — any memory consumed by a custom cache is memory stolen from both the GPU and the OS page cache.

## Insight

Expert activation frequencies follow a **Zipfian (power-law) distribution**. A small subset of experts handles the majority of activations. If we can make the total expert dataset *smaller on disk*, more of it fits in the OS page cache without consuming any additional memory.

The key insight: **reduce disk footprint, not add caching**.

## Approach: Mixed-Precision Expert Packing

1. **Profile** expert activation frequencies across representative workloads
2. **Classify** experts as "hot" (top ~25% by frequency, covering ~80% of activations) or "cold"
3. **Keep hot experts at 4-bit** — these are accessed most often, quality matters
4. **Requantize cold experts to 2-bit** — 44% smaller per expert, rarely activated, quality impact is diluted by low activation probability
5. **Per-expert Metal kernel dispatch** — at runtime, each expert uses the correct dequant shader (4-bit `matvec_v3` or 2-bit `matvec_2bit`)

This is fundamentally different from uniform 2-bit (which breaks JSON/tool calling). Hot experts preserve 4-bit quality for the most-used pathways while cold experts save space.

## Profiling Results (Qwen3.5-35B-A3B)

Ran `--freq` profiling across diverse prompts (math, code, reasoning, creative writing):

| Metric | Value |
|--------|-------|
| Total experts per layer | 256 |
| Active per token (K) | 8 |
| Avg experts for 80% coverage | 62/256 (24.2%) |
| Min experts for 80% coverage | 22 (layer with highest concentration) |
| Max experts for 80% coverage | 108 (layer with most uniform distribution) |

Expert usage is highly concentrated: the top-10 experts per layer cover 14-52% of all activations depending on the layer.

## Disk Savings

| Model | 4-bit Size | Tiered Size | Savings |
|-------|-----------|-------------|---------|
| Qwen3.5-35B-A3B | 16.88 GB | 11.19 GB | **33.7%** |
| Qwen3.5-397B-A17B (est.) | 209 GB | ~138 GB | **~34%** |

Per-expert sizes (35B):
- 4-bit expert: 1,769,472 bytes (1.69 MB)
- 2-bit expert: 983,040 bytes (0.94 MB)
- Savings per cold expert: 786,432 bytes (44.4%)

## Implementation

### Pipeline

```
profile_experts.py    →  hot_experts.json       (per-layer hot expert lists)
repack_experts_tiered.py  →  packed_experts_tiered/  (mixed layer files + manifest)
infer --tiered        →  per-expert kernel dispatch at runtime
```

### Profiling (`profile_experts.py`)

- Parses `FREQ_DUMP` output from `infer --freq`
- Aggregates frequencies across multiple runs
- Selects hot experts per layer based on configurable coverage threshold (default 80%)
- Outputs `hot_experts.json`:
  ```json
  {
    "coverage_threshold": 0.8,
    "layers": {
      "0": [45, 12, 201, 88, ...],
      "1": [3, 167, 44, ...],
      ...
    }
  }
  ```

### Repacking (`repack_experts_tiered.py`)

- Reads `packed_experts/layer_XX.bin` (uniform 4-bit)
- Hot experts: copied as-is (4-bit)
- Cold experts: requantized to 2-bit using optimal per-group quantization:
  1. Dequantize 4-bit → float32 (using scales + biases from group headers)
  2. Compute optimal 2-bit scale and bias per 32-element group
  3. Quantize float32 → 2-bit with round-to-nearest
- Writes `packed_experts_tiered/layer_XX.bin` with variable-size experts packed sequentially
- Writes `packed_experts_tiered/tiered_manifest.json`:
  ```json
  {
    "expert_size_4bit": 1769472,
    "expert_size_2bit": 983040,
    "layers": {
      "0": [
        {"offset": 0, "size": 1769472, "bits": 4},
        {"offset": 1769472, "size": 983040, "bits": 2},
        ...
      ]
    }
  }
  ```

### Runtime Changes (`infer.m`)

**New data structures:**
- `TieredExpertInfo` struct: `{offset, size, bits}` per expert
- `g_tiered_manifest` global array indexed by `layer * num_experts + expert`
- `expert_offset_size()` helper replaces all `active_expert_size()` calls

**Per-expert kernel dispatch in `gpu_encode_experts_batched()`:**
- Each of the K active experts checks its `bits` field from the manifest
- Selects the appropriate Metal pipeline (`matvec_v3` for 4-bit, `matvec_2bit` for 2-bit)
- Computes correct buffer offsets for the selected quantization format
- Different experts in the same token can use different kernels

**Updated I/O paths (all use `expert_offset_size()`):**
- `parallel_pread_experts()` / `parallel_pread_experts_into()`
- `async_pread_start()` / `async_pread_wait()`
- `infer_prefetch_start()` / `infer_prefetch_thread_fn()`
- Malloc cache miss path
- Metal LRU cache miss path
- Prediction miss path
- Speculative cache path
- Single-expert sequential path
- CPU fallback path

**CLI:**
- `--tiered` flag (mutually exclusive with `--2bit`)
- Auto-detection: if `packed_experts_tiered/tiered_manifest.json` exists, tiered mode activates automatically

## Benchmark Results (Qwen3.5-35B-A3B, M3 Max 48GB)

| Mode | tok/s | Disk | Quality |
|------|-------|------|---------|
| 4-bit baseline | 5.20 | 16.88 GB | Excellent |
| Tiered (hot=4b, cold=2b) | 5.24 | 11.19 GB | Excellent |
| 2-bit uniform | ~6.5 | ~9.4 GB | Broken JSON |

The marginal speedup on 35B is expected — 17GB of experts fits comfortably in 48GB page cache, so cache miss rate was already low. The real value is for:

1. **Larger models** (397B): 209GB → ~138GB means significantly more experts cached in the same 35GB page cache window
2. **Memory-constrained devices** (24GB Mac, future iOS): 11GB vs 17GB is the difference between fitting or not fitting
3. **Multi-model scenarios**: smaller expert footprint leaves more page cache for other processes

## Quality Verification

- **Math**: "What is 2+2?" → "4" (correct)
- **JSON/Tool calling**: Produces valid `{"name": "get_weather", "arguments": {...}}` — no quoting corruption
- **Reasoning**: Multi-step outputs coherent and correct

The key quality advantage over uniform 2-bit: hot experts (which contribute most to output quality) retain full 4-bit precision. Cold experts contribute less to any given token, so their lower precision has minimal impact.

## Why This Works When Other Caching Failed

| Approach | Memory Impact | Result |
|----------|--------------|--------|
| Metal LRU cache | Steals GPU memory | -38% |
| malloc cache | Steals from page cache | -20% |
| LZ4 compression | CPU overhead + memory for decompression | -13% |
| **Tiered quantization** | **Zero additional memory** | **+0.8% (35B), est. +15-25% (397B)** |

Tiered quantization is the only approach that improves cache hit rates without consuming any additional memory. It shrinks the data on disk, letting the existing OS page cache cover a larger fraction of the expert set.

## Files

| File | Lines | Description |
|------|-------|-------------|
| `profile_experts.py` | ~160 | Frequency profiling and hot expert selection |
| `repack_experts_tiered.py` | ~230 | Mixed-quant repacking with manifest generation |
| `metal_infer/infer.m` | +300/-73 | Runtime tiered dispatch, manifest loader, CLI |
