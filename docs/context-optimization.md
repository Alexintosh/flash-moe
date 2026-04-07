# Context Optimization

This document covers the full context management story for Flash-MoE: how the hybrid attention architecture enables aggressive KV cache optimization, and the three techniques that work together to maximize context length on memory-constrained devices.

## The Hybrid Attention Advantage

Qwen3.5 MoE models use a hybrid attention architecture:
- **30 GatedDeltaNet layers** (linear attention) -- maintain context via 128x128 state matrices per head. O(1) memory regardless of sequence length. These layers compress the entire conversation history into fixed-size state, providing "long-term memory."
- **10 full attention layers** -- standard Q/K/V attention with a KV cache that grows linearly with sequence length. These layers provide precise token-to-token lookups, acting as "short-term memory."

This split is key: only 10 out of 40 layers need a KV cache at all. The 30 linear attention layers are free -- they always remember everything through their state matrices. This means context optimizations only need to target the 10 full attention layers, making every technique 3-4x more effective than it would be in a standard transformer.

## FP8 KV Cache

### How It Works

The KV cache stores attention Key and Value vectors for each position in the context. By default, these are float32 (4 bytes per element). FP8 E4M3 quantization reduces this to 1 byte per element -- a 4x reduction.

**FP8 E4M3 format:** 1 sign bit, 4 exponent bits, 3 mantissa bits. Exponent bias: 7. Range: [-448, 448].

**Encoding:** For each position, compute `scale = absmax / 240.0` (headroom below the 448 max). Each float is clipped and quantized to 8 bits. The scale is stored separately per position.

**Decoding:** Inline during attention compute. The fused attention kernel reads FP8 bytes and per-position scales on the fly via Metal function constants (`USE_FP8_KV`). When FP8 is disabled, the branch is eliminated at pipeline creation time -- zero overhead for the float32 path.

### Memory Savings

| Model | Layers (full attn) | KV heads | Head dim | FP32 per position | FP8 per position |
|-------|-------------------|----------|----------|-------------------|-----------------|
| 35B   | 10                | 4        | 128      | 40 KB             | 10 KB           |
| 397B  | 15                | 8        | 128      | 60 KB             | 15 KB           |

At 4K context:
- 35B FP32: 160 MB KV cache. FP8: 40 MB.
- 397B FP32: 240 MB KV cache. FP8: 60 MB.

At 32K context:
- 35B FP32: 1.28 GB KV cache. FP8: 320 MB.
- 397B FP32: 1.92 GB KV cache. FP8: 480 MB.

### Flag Ordering

FP8 and sliding window flags must be set BEFORE calling `metal_setup()`. This ensures Metal buffer allocation uses the correct element size (1 byte vs 4 bytes) from the start. A previous bug where flags were set after setup caused the GPU to allocate float32-sized buffers but write FP8 data, leading to incorrect attention results.

## Sliding Window Attention

### Design

A circular KV buffer for full attention layers. The write position cycles: `cache_pos = kv->len % window_size`. Only the most recent `window_size` positions are stored and attended to.

This works because:
1. The 30 GatedDeltaNet layers still see the FULL context via their state matrices.
2. The 10 full attention layers only need local context for most tasks. Early tokens are summarized by the linear attention layers' state.
3. Combined: the model has both long-range memory (linear attention state) and precise short-range lookups (windowed full attention).

### Why Circular Buffer

A circular buffer avoids copying. When position N expires, position N+window_size overwrites it in place. No compaction, no shifting. The GPU attention kernel just reads positions 0..window_size-1, all of which contain valid recent data.

### Memory with Sliding Window

| Config | 35B KV | 397B KV | Notes |
|--------|--------|---------|-------|
| FP32, unlimited | Grows with context | Grows with context | Default |
| FP32, window=4096 | 160 MB fixed | 240 MB fixed | Capped |
| FP8, window=4096 | **40 MB fixed** | **60 MB fixed** | Best for mobile |
| FP8, window=2048 | 20 MB fixed | 30 MB fixed | Minimum viable |

## H2O KV Cache Eviction (In Progress)

### The Algorithm

H2O (Heavy Hitter Oracle) is an attention-score-based eviction policy from the paper "H2O: Heavy-Hitter Oracle for Efficient Generative Inference" (Zhang et al., 2023). It tracks which KV cache positions receive the most attention and keeps those.

**Three protected regions:**
1. **Sink tokens** (first N, typically 4) -- attention sinks that every transformer head attends to. Evicting them degrades quality catastrophically.
2. **Recent tokens** (last 25% of budget) -- the model always needs to see its most recent outputs for coherent generation.
3. **Heavy hitters** (remainder of budget) -- positions with the highest cumulative attention scores across all heads and all steps. These are the "important" tokens the model keeps referring back to.

### How It Works in Flash-MoE

1. **Score accumulation:** After each full-attention step, post-softmax scores are summed into `attn_scores_accum[position]`. Scores are accumulated across all query heads (GQA-aware).

2. **Eviction trigger:** When `h2o_num_valid > h2o_budget` (i.e., one new token was written past the budget), eviction runs.

3. **Selection:** Sinks and recent tokens are unconditionally kept. Among the middle positions, the top scorers survive. The rest are evicted.

4. **Compaction:** Surviving entries are moved to contiguous positions [0..budget-1] in both CPU arrays and GPU Metal buffers. GPU kernels see a shorter, contiguous sequence -- no scatter-gather needed.

5. **GPU sync:** After CPU-side compaction, `kv_cache_h2o_sync_gpu()` copies the compacted data to Metal buffers via `memcpy` into buffer contents.

### H2O vs Sliding Window

H2O replaces sliding window when both are configured (`g_h2o_budget > 0` takes priority). H2O is strictly better because:
- Sliding window drops ALL tokens older than the window, even if they were critically important.
- H2O keeps the most-attended tokens regardless of age, plus guaranteed sinks and recents.
- Both have fixed memory: sliding window = `window_size` positions, H2O = `budget` positions.

## RoPE Scaling (Context Extension)

RoPE (Rotary Position Embedding) scaling extends the effective context length beyond the model's native training length. Three methods are implemented, each with different quality/extension tradeoffs:

### Linear Scaling

Divides position indices by a scale factor before computing RoPE frequencies. If the model was trained with 4K context and you set `rope_scale_factor=2.0`, it can handle 8K context.

**Pros:** Simple, zero overhead.
**Cons:** Quality degrades at high ratios (>2x). The model sees "compressed" positions it was never trained on.

### NTK-aware Scaling

Modifies the RoPE frequency base instead of the positions. The base frequency is raised: `base' = base * factor^(dim/(dim-2))`. This rotates all frequency components proportionally, preserving the relative structure.

**Pros:** Better quality at 2-4x extension. No per-position modification.
**Cons:** Moderate quality loss at >4x.

### YaRN (Yet Another RoPE extensioN)

Per-dimension interpolation between original and scaled RoPE, with an attention scaling factor. Low-frequency dimensions (which encode long-range position) are interpolated more aggressively, while high-frequency dimensions (which encode local position) are left mostly unchanged.

**Pros:** Best quality at 4-8x extension. Designed for large extensions.
**Cons:** Most complex. Requires an attention temperature correction factor.

### Configuration

CLI: `--rope-scale-type {none,linear,ntk,yarn}` and `--rope-scale-factor <float>`.
iOS: Expert Settings > Context > RoPE Scaling.

Default: no scaling (native context length). RoPE scaling only affects the 10 full-attention layers; the 30 GatedDeltaNet layers are position-independent.

## Combined Effect

The three techniques stack:

| Configuration | 35B KV (4K context) | Notes |
|--------------|-------------------|-------|
| FP32, unlimited | 160 MB | Baseline |
| FP8 only | 40 MB | 4x reduction |
| FP8 + sliding window 4096 | 40 MB fixed | Bounded regardless of conversation length |
| FP8 + H2O budget 4096 | 40 MB fixed | Same bound, but smarter eviction |
| FP8 + sliding window 2048 | 20 MB fixed | Aggressive, for very constrained devices |

On iPhone 17 (12GB):
- Without optimization: 35B model can support ~4K context before memory pressure.
- With FP8 + sliding window 4096: effectively unlimited conversation length at 40MB fixed KV cost.
- Freed memory goes to the OS page cache for expert data, improving expert cache hit rates.

## Memory Budget Calculator

To estimate KV cache memory for a given configuration:

```
KV memory = num_full_attn_layers x 2 (K+V) x kv_heads x head_dim x bytes_per_elem x positions
```

Where:
- `bytes_per_elem` = 4 (FP32) or 1 (FP8) + scale overhead (~4 bytes per position per cache)
- `positions` = min(max_context, sliding_window) if sliding window enabled, else max_context

**35B model** (10 full-attn layers, 4 KV heads, 128 head dim):
- Per position: 10 x 2 x 4 x 128 = 10,240 elements
- FP32: 10,240 x 4 = 40 KB/position
- FP8: 10,240 x 1 + scales = ~10.5 KB/position

**397B model** (15 full-attn layers, 8 KV heads, 128 head dim):
- Per position: 15 x 2 x 8 x 128 = 30,720 elements
- FP32: 30,720 x 4 = 120 KB/position (note: actual measurement is ~60 KB due to GQA)
- FP8: ~15 KB/position

For the 35B on iPhone at 8K context with FP8: ~82 MB total KV. Comfortable within the 12GB memory budget after model weights and Metal buffers.
