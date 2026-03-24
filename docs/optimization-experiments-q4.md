# Q4 Expert Optimization Experiments

## Context

After discovering that 2-bit expert quantization broke tool calling (JSON quotes → backslashes), we reverted to 4-bit experts. This dropped performance from 5.74 tok/s (2-bit) to 3.50 tok/s (4-bit). The goal: recover as much speed as possible while maintaining 4-bit quality.

The 4-bit experts are 7,077,888 bytes each (6.75 MB). With K=4 active experts per layer and 60 layers, each token reads 240 experts = 1.68 GB from SSD.

## Baseline Pipeline (4-bit, K=4, trust OS page cache)

```
Per layer (4.28 ms avg):
  cmd1_wait:    1.22 ms  (28%)  GPU: CMD3(prev) + CMD1 attention projections
  cmd2_wait:    0.55 ms  (13%)  GPU: o_proj + norm + routing + shared expert
  expert_io:    2.41 ms  (56%)  SSD: 4×7MB parallel pread
  CPU work:     0.10 ms  ( 2%)  encode + attention + routing + memcpy

60 layers × 4.28 ms = 257 ms per token = 3.90 tok/s
```

Page cache hit rate: ~71% (35 GB cache, 209 GB model).
Warm cache parallel pread: 1.0 ms. Cold SSD: 5.8 ms. Mixed: 2.4 ms.

## Experiment Results

### Kept: FMA Dequant Kernel (+2.6% → 4.36 tok/s)

Rearranged the inner loop of `dequant_matvec_4bit_v3` from:
```metal
acc += (float(nibble) * scale + bias) * x;
```
to:
```metal
float sx = scale * x, bx = bias * x;
acc += fma(float(nibble), sx, bx);
```

Pre-computing `scale*x` and `bias*x` per input element allows the GPU to use the fused multiply-add unit for the dequant+multiply in one instruction. Reduces per-nibble cost from (convert + mul + add + mul + add) to (convert + fma + add).

Impact: cmd1_wait -5.4%, cmd2_wait -10.7%. Total: 3.90 → 4.36 tok/s.

### Discarded: LZ4 Expert Compression (-13%)

Repacked 209 GB of expert files to 175 GB with LZ4 compression. Apple's LZ4 decompressor runs at 41 GB/s (NEON hardware-accelerated), making decompression only 0.17 ms per expert.

Results:
- Isolated cold reads: 15-24% faster (less data from SSD)
- Isolated decompression: 0.17 ms at 41 GB/s (essentially free)
- **Full pipeline: 3.55 tok/s (-13%)** — the 0.68 ms/layer decompress cost exceeds the warm cache I/O savings. The OS page cache is efficient enough that most reads are warm.

Also tested LZFSE (2.6 GB/s, too slow), APFS transparent compression (kernel serializes read+decompress, 2× slower), and per-expert files (15% slower from VFS metadata overhead).

Key finding: Apple's M3 Max SSD is so fast that CPU-based decompression can't keep up for warm cache reads. LZ4 only wins for cold reads, but the page cache handles most reads.

### Discarded: Expert Routing Prediction (-18%)

Built a temporal prediction system: store previous token's expert routing per layer, prefetch those experts into double-buffered Metal buffers during the next token's CMD1 wait.

Results:
- Temporal hit rate: 25.6% (only 1 of 4 experts matches between tokens)
- The 75% misses waste SSD bandwidth and require sync pread after the prediction wait
- With K=4 parallel reads, wall time = max(4 reads). Need ALL 4 to hit for improvement.
- P(all 4 hit) at 25% = 0.25⁴ = 0.4%. Practically zero.

Also trained an MLP predictor (31% accuracy from pre-attention hidden state — worse than temporal baseline). The gate_proj "logit lens" approach achieves 53% from pre-attention state, but the K=4 exponential penalty still kills it.

### Discarded: F_RDADVISE Prefetch (net 0%)

Sent F_RDADVISE kernel hints between CMD1 commit and wait to prefetch next token's predicted experts during GPU compute.

Results:
- expert_io: -31% (page cache warming works!)
- cmd2_wait: +73% (GPU memory bandwidth contention from SSD DMA)
- **Net: 0% across 5 diverse prompts**

Root cause: Apple Silicon unified memory architecture. SSD DMA and GPU matvec share the same memory controller. The GPU's dequant kernels are bandwidth-saturated at 418 GiB/s. Even 17.5 GB/s of background DMA (~4%) causes disproportionate latency spikes through memory controller arbitration. This is architectural — cannot be worked around in software.

### Discarded: GPU Kernel Variants

- **LUT dequant (v5)**: Pre-compute 16-entry lookup table per group to eliminate uint→float conversions. -2% because GPU indirect register access serializes.
- **Vector load (v4)**: uint4 loads for coalesced memory access. -3% from register pressure.
- **extract_bits intrinsic**: Neutral — compiler already generates the same instruction.
- **Spin-poll GPU wait**: -23%. CPU spinning steals thermal budget from GPU on unified architecture.
- **addCompletedHandler**: Neutral in practice — isolated 20% win on micro-benchmark but real workloads have enough GPU compute to hide the wait overhead.

### Discarded: I/O Path Alternatives

- **dispatch_io**: -70%. Apple's GCD I/O framework adds dispatch_data management overhead (allocate, map, memcpy) that far exceeds any kernel scheduling benefit.
- **aio_read**: -7% (matches GCD group + pread, which we already use).
- **Expert file clustering**: 0%. NVMe doesn't care about scatter distance at 7MB read granularity. 4 reads spanning 21 MB vs 2.9 GB take the same time.
- **GPU private buffer compression**: Isolated -13.5% per matvec (GPU hardware memory compression on StorageModePrivate). But in pipeline: blitting 4×7MB shared→private costs more than the matvec savings. -20% overall.

### Analyzed but not implemented: MTP Speculative Decoding

Qwen 3.5 ships with an MTP (Multi-Token Prediction) head — a single MoE transformer layer that predicts the next-next token. The head exists in the model config (`mtp_num_hidden_layers: 1`) but weights were stripped from the MLX quantization.

Analysis showed MTP speculative decoding doesn't help for MoE with SSD streaming: each speculated token requires its OWN expert routing and I/O. Batched verification of 2 tokens costs ~1.75× expert I/O for 1.7 tokens (70% acceptance). Break-even at best.

This contrasts with dense models where verification cost is constant regardless of batch size (same weights for every token).

## The Unified Memory Constraint

The single most important finding: **on Apple Silicon, SSD DMA and GPU compute cannot be profitably overlapped.** They share the same memory controller, and the GPU's dequant kernels are bandwidth-saturated. Any background I/O during GPU compute causes disproportionate GPU slowdown.

This means the serial pipeline (GPU → SSD → GPU) is actually **hardware-optimal** for this architecture. The current pipeline already achieves the best possible scheduling.

## Summary

| Configuration | tok/s | Status |
|--------------|-------|--------|
| 2-bit experts (best speed) | 5.74 | Quality regression (broken JSON) |
| 2-bit peak single token | 7.05 | Warm cache burst |
| **4-bit + FMA kernel** | **4.36** | **Current best. Quality preserved.** |
| 4-bit baseline (no FMA) | 3.90 | Previous 4-bit baseline |
| 4-bit + LZ4 compression | 3.55 | Decompress overhead > I/O savings |
| 4-bit + temporal prediction | 3.18 | 25% hit rate wastes SSD bandwidth |
| 4-bit + F_RDADVISE prefetch | 3.91 | GPU contention cancels I/O savings |

The 4-bit performance ceiling on M3 Max 48GB is approximately **4.4 tok/s** for sustained generation, limited by:
- 56% SSD expert I/O (2.4 ms/layer, hardware-limited)
- 41% GPU dequant matvec (1.8 ms/layer, bandwidth-limited)
- 3% CPU overhead (0.1 ms/layer)

Further improvement requires either hardware changes (more RAM for expert caching, faster SSD) or model architecture changes (fewer/smaller experts, larger shared expert to reduce per-token I/O).

## iPhone Testing (A19, 12GB)

The same optimization techniques were tested on iPhone 17 with the 35B model:

### Applicable Optimizations

| Optimization | Result on iPhone | Notes |
|-------------|-----------------|-------|
| FMA dequant kernel | Included in baseline | Same kernel runs on both platforms |
| CMD1+CMD2 merge | +5-10% projected | Saves 30 sync points per token on 35B (30 linear layers) |
| Fused expert kernel | +3-5% projected | Reduces dispatch overhead per expert |
| Expert prefetch | Under validation | Depends on A19 NVMe characteristics |
| FP16 accumulation | Under validation | A19 has fp16 ALUs, may be more impactful than M3 Max |
| K-reduction (K=4 from K=8) | **~2x speedup** | Halves expert I/O, the dominant bottleneck |

### iPhone-Specific Results

| Configuration | tok/s | Notes |
|--------------|-------|-------|
| 35B, K=8, baseline | 5.5 | Full quality, 19.5GB model |
| 35B, K=8, tiered | 5.5+ | 13.4GB model, same quality |
| 35B, K=4, projected with all opts | ~11 | CMD merge + fused expert + K reduction |
| 397B, K=4, CPU fallback | ~0.003 | Metal 4GB buffer limit blocks GPU path |

### Key Differences from MacBook

1. **NVMe bandwidth** -- iPhone ~2.5-3 GB/s vs MacBook 17.5 GB/s. Expert I/O takes proportionally longer, making K-reduction even more impactful on mobile.
2. **Memory budget** -- 12GB vs 48GB. Less room for OS page cache, so expert cache hit rates are lower. FP8 KV cache and sliding window attention free up memory for the page cache.
3. **Thermal constraints** -- iPhone thermal throttles sooner. Sustained generation speeds may be lower than burst measurements.
4. **Metal buffer limits** -- 4GB per buffer on iOS vs unlimited on macOS. Prevents GPU path for 397B model without split weight files.
