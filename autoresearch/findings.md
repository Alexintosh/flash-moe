# Autoresearch v2 Findings

Structured results from paper-guided experiments. Read this BEFORE starting new experiments to avoid repeating dead ends.

## Historical Context (from v1 and manual optimization)

Before this log, 58+ experiments were run. Key learnings:

- **FMA dequant kernel**: +12% from rearranging `(nibble * scale + bias) * x` → `fma(nibble, scale*x, bias*x)`. KEPT.
- **Trust OS page cache**: Every custom cache was slower. Metal LRU (-38%), malloc (-20%), LZ4 (-13%).
- **Deferred CMD3**: GPU/CPU overlap for expert forward pass. KEPT.
- **BLAS delta-net**: cblas_sscal/sgemv/sger for 64-head recurrence. +64% on attention. KEPT.
- **F_NOCACHE for 2-bit**: +3% from avoiding page cache thrash with smaller working set. 2-bit only.
- **GPU fused attention (RoPE)**: +2% for full-attention layers. KEPT.
- **C BPE tokenizer**: 180ms vs 3500ms startup. 20x improvement. KEPT.
- **Half-precision x_shared in 2-bit**: 6.48 tok/s (best from v1 experiments). KEPT.

### Failed approaches (DO NOT RETRY):
- LZ4 expert compression: -13% (decompression overhead > savings)
- F_RDADVISE prefetch: net 0% (unified memory: SSD DMA slows GPU -73%)
- Temporal expert prediction: -18% (25% hit rate, bandwidth waste)
- MLP routing predictor: 31% accuracy (worse than temporal)
- GPU LUT dequant kernel: -2% (indirect register access serializes)
- GPU private buffer compression: -20% (blit cost 4x7MB > savings)
- Spin-poll GPU wait: -23% (CPU thermal competes with GPU)
- Expert file clustering: 0% (NVMe ignores scatter at 7MB granularity)
- dispatch_io: -70% (dispatch_data management overhead)
- mmap expert files: -5x (per-page fault overhead on cold data)
- Speculative early routing: -38% (cache pollution + overhead)
- MTP speculative decoding: break-even (MoE I/O scales per-token)

---

_New experiments start below. Each entry follows the structured format._

