# Expert Settings Guide

This document describes every toggle and picker in the Expert Settings panel of the Flash-MoE iOS/Mac app. Each setting includes a plain-language analogy and technical explanation (the same content shown in the in-app info modals).

## Active Experts (K)

**Analogy:** Imagine asking a question to a room of 256 specialists. K controls how many you consult. K=8 means you ask 8 experts and combine their answers. K=4 means you only ask 4 -- faster (less reading from disk) but you might miss a specialist who had a great insight.

**Technical:** Each transformer layer routes the token to K out of 256 experts via a learned gating network. Each expert is a ~1.7MB weight matrix loaded from SSD via `pread()`. Lower K = fewer SSD reads per layer = proportionally less I/O time (the dominant bottleneck at 56% of per-token latency). Quality degrades gracefully because the router still picks the best K from the full vocabulary.

**Options:** 2, 3, 4, 5, 6, 7, 8, 9, 10 (model default varies: K=8 for 35B, K=10 for 397B)

---

## I/O Fanout

**Analogy:** Think of reading a book page. Instead of reading the whole page in one go, you split it into strips and read them all simultaneously with multiple eyes. Fanout splits each expert weight file read into parallel chunks so the SSD controller can serve them concurrently.

**Technical:** Each expert (~1.7MB for 35B) is read via `pread()`. Fanout splits this into N page-aligned chunks dispatched via `GCD dispatch_group_async`. NVMe controllers have multiple queues and can serve parallel reads faster than a single large read. Best value depends on expert size vs NVMe page size (4KB). Diminishing returns above 4 chunks.

**Options:** Off, 2, 4, 8

---

## CMD1+CMD2 Merge

**Analogy:** Like combining two errands into one trip instead of driving home between them. CMD1 (attention projections) and CMD2 (output projection + normalization) are separate GPU tasks. Merging them avoids the roundtrip of "submit, wait, create new, submit" for each of the 30 linear attention layers.

**Technical:** For linear attention layers (GatedDeltaNet), the CPU phase between CMD1 and CMD2 is empty -- the GPU already computed everything. CMD2's dispatches (o_proj matmul, residual add, RMS norm, routing, shared expert) are appended to CMD1 with pipeline barriers. Saves ~0.05-0.1ms per layer x 30 layers = 1.5-3ms per token.

**Default:** ON

---

## Fused Attention

**Analogy:** Standard attention is like a three-step cooking recipe: measure all ingredients (Q@K scores), mix them (softmax), then combine (scores@V). Fused attention does all three in one pass -- like a skilled chef who seasons, mixes, and plates in a single flowing motion. Less cleanup between steps.

**Technical:** Replaces 3 separate GPU kernel dispatches (attn_scores, attn_softmax, attn_values) with a single fused kernel using FlashAttention-2 online softmax. Processes KV positions in blocks of 64, maintaining running max/sum/output. Eliminates 2 command encoder transitions per full-attention layer (10 layers). Uses unnormalized accumulation with single final division.

**Default:** OFF (experimental)

---

## Fused Expert Kernel

**Analogy:** Each expert normally does three separate calculations: gate, up, and activation. It's like washing, drying, and folding laundry in three separate trips. The fused kernel does all three in one pass through the data -- one trip, everything done.

**Technical:** Combines gate_proj matmul + up_proj matmul + SiLU activation into a single Metal compute kernel (`fused_gate_up_swiglu`). Both gate and up dot products are computed in one loop over the input vector, then SiLU is applied immediately. Reduces from 3 GPU dispatches to 1 per expert, saving command encoder overhead for K experts x 40 layers.

**Default:** ON

---

## Expert Prefetch

**Analogy:** While the kitchen (GPU) is cooking layer 5's dish, the waiter (CPU) runs ahead to the pantry (SSD) to grab ingredients for layer 6. When the kitchen finishes layer 5, the ingredients for layer 6 are already on the counter -- no waiting.

**Technical:** After CMD3(N) is submitted (deferred GPU execution), the system predicts which experts layer N+1 will need based on routing history. Those experts are `pread()` into Set B buffers asynchronously. When layer N+1 reaches its I/O phase, prefetch hits skip the pread entirely. Misses fall through to normal loading. Overlaps ~2.4ms of I/O with GPU compute time.

**Default:** OFF (experimental, not yet validated on all configurations)

---

## FP16 Accumulation

**Analogy:** Imagine counting coins on a kitchen scale that rounds to one decimal. Each coin adds a tiny rounding error. After 500 coins, you might be off by one. But the scale reads twice as fast. FP16 does math at 2x the speed of FP32, but accumulates small rounding errors over hundreds of additions.

**Technical:** The dequant matvec inner loop changes from float32 to float16 accumulation. Apple's A-series GPU has dedicated fp16 ALUs at 2x throughput. The FMA becomes half-precision: `fma(half(nibble), half(scale*x), half(bias*x))`. Final output is promoted to float32 via `simd_sum`. Risk: fp16 has ~3 decimal digits; sums of 512+ elements may lose precision.

**Default:** OFF (experimental)

---

## FP8 KV Cache

**Analogy:** The KV cache is like a notebook where the model writes down what it's seen. FP32 uses a full page per note. FP8 uses a quarter page -- same content, just more compressed handwriting. You fit 4x more notes in the same notebook, so the model can remember 4x more conversation.

**Technical:** Stores attention Key and Value vectors in FP8 E4M3 format (1 byte vs 4 bytes per element) with per-position dynamic scaling. Encoding: absmax/240 scale factor, each float clipped and quantized to 8-bit (1 sign, 4 exponent, 3 mantissa). Decoding is inline during attention compute via Metal function constants. 4x memory reduction enables 4x longer context at the same memory budget.

**Default:** OFF

See [context-optimization.md](context-optimization.md) for detailed memory savings tables.

---

## Max Context Length

**Analogy:** Context length is how far back the model can "see" in the conversation. Like a person's short-term memory -- 4K tokens is the last few minutes, 32K is the last hour. More context = better understanding of the conversation, but uses more memory.

**Technical:** Sets the maximum sequence length for KV cache allocation. Memory cost: `num_full_attn_layers x 2 (K+V) x kv_heads x head_dim x bytes_per_elem x positions`. For the 35B with 10 full-attn layers: 40KB/pos (FP32) or 10KB/pos (FP8). Auto mode uses `os_proc_available_memory()` to pick the largest safe value.

**Options:** Auto, 4K, 8K, 16K, 32K

---

## Sliding Window

**Analogy:** Instead of remembering everything forever (which fills up memory), the full-attention layers only look at the last N tokens -- like a window sliding along the conversation. But the 30 linear attention layers still remember everything through their state matrices. It's like having both short-term and long-term memory working together.

**Technical:** Implements a circular KV buffer for full attention layers. Write: `cache_pos = kv->len % window_size`. Read: attend only to the most recent `window_size` positions. The 30 GatedDeltaNet layers maintain full context via their 128x128 state matrices (O(1) memory). Only the 10 full attention layers are windowed. With window 4096 + FP8: fixed 40MB KV regardless of conversation length.

**Options:** Off, 2048, 4096, 8192

See [context-optimization.md](context-optimization.md) for the full context optimization story.

---

## Thinking Mode

**Analogy:** Like a student who shows their work before giving the final answer. The model reasons step-by-step inside `<think>` tags before responding. This usually produces better answers, but takes more tokens (and time). At low K values, the model may get stuck thinking forever -- disable it for speed.

**Technical:** The chat template includes a `<think>` tag after the assistant turn header. The model generates reasoning tokens inside the think block, then emits `</think>` before the actual response. Think budget caps the maximum thinking tokens and force-emits `</think>`. Set to -1 to disable thinking entirely (removes `<think>` from the template).

**Default:** ON (with configurable budget, default 2048 tokens)

---

## H2O Budget (Coming Soon)

**Analogy:** Instead of sliding window's simple "forget old stuff" approach, H2O watches which parts of the conversation the model keeps looking back at (the "heavy hitters") and keeps those. It's like a librarian who notices which reference books get used most and keeps those on the desk, while shelving the rarely-used ones.

**Technical:** Heavy Hitter Oracle eviction policy. Tracks cumulative post-softmax attention scores per KV position. Budget = total positions to keep. Protected regions: sink tokens (first 4) + recent tokens (25% of budget). Remaining budget goes to positions with highest cumulative attention scores. After eviction, both CPU and GPU caches are compacted to contiguous positions. Replaces sliding window when both are configured (H2O is strictly better).

**Status:** Implementation complete in the engine (`kv_cache_evict_h2o`, `kv_cache_h2o_accumulate_scores`). UI toggle pending.
