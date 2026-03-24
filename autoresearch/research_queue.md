# Research Queue

Prioritized techniques to test, extracted from papers. Checked items have been tested — see `findings.md` for results.

## Priority 1 — Expert I/O Optimization (56% of bottleneck)

- [ ] **Frequency-based expert pinning** — Keep the top-N most frequently activated experts in a persistent mmap'd buffer. OS page cache does LRU, but frequency-based pinning is different (sticky hot set). From "LLM in a Flash" (Alizadeh et al., Apple, 2023). Risk: we tested custom caches before and they were slower, but pinning != caching.
- [ ] **Expert co-activation clustering** — Reorder expert indices within each layer file so frequently co-activated experts are physically adjacent. Sequential NVMe reads are faster than scattered reads at small granularity. From "MoE-Infinity" (Xue et al., 2024). Note: we tested file-level clustering at 7MB granularity (0% improvement), but sub-expert byte-level adjacency within a single file is untested.
- [ ] **Attention-to-routing correlation prefetch** — Use layer N's attention output distribution to predict layer N+1's expert routing before the actual router MLP runs. Gives a 1-2ms head start on pread. From "Pre-gated MoE" (Hwang et al., 2024). Risk: prediction accuracy may be too low (our temporal predictor hit 25%).
- [ ] **NVMe page-aligned expert reads** — Ensure pread offsets and sizes are 4KB-aligned to avoid partial page reads. From "FlashNeuron" (Song et al., 2021). Quick check: are our current expert offsets already aligned?
- [ ] **Compressed expert transfer** — Use hardware-accelerated LZ4/zstd decompression (Accelerate.framework) on expert weights during SSD→memory transfer. Different from our prior test (software LZ4 was -13%) because hardware decompression has near-zero CPU cost. Needs investigation of Accelerate's compression APIs.

## Priority 2 — GPU Compute Optimization (30% of bottleneck)

- [ ] **Register-tiled GEMV** — Process multiple output rows per thread (thread coarsening) to amortize shared memory loads. Each thread computes 2-4 output elements instead of 1. From "AWQ" (Lin et al., 2023) and general GPU optimization literature. Currently ROWS_PER_TG=8 with 1 row per SIMD group — could do 2 rows per SIMD group.
- [ ] **Persistent threadgroups** — Keep threadgroups alive across multiple matvec dispatches within a command buffer using `[[threadgroup_size_in_memory]]`. Avoids threadgroup allocation/deallocation overhead. From CUTLASS (NVIDIA, 2022) — Metal equivalent is `dispatchThreadgroups:threadsPerThreadgroup:` with persistent launch.
- [ ] **SIMD shuffle reduction** — Replace `threadgroup float shared[32]` reductions with `simd_shuffle_xor` cascade. Avoids shared memory roundtrip for the final reduction step. From "FlashDecoding++" (Hong et al., 2024).
- [ ] **Mixed-precision with Kahan compensated summation** — Accumulate in fp16 for speed but use Kahan summation to recover fp32 accuracy. Best of both worlds: fp16 throughput with fp32 error bounds. From numerical analysis literature.
- [ ] **Vectorized nibble extraction** — Use `as_type<uchar4>()` to extract 4 nibbles simultaneously instead of sequential shift+mask. May improve instruction-level parallelism. From GPU programming guides.

## Priority 3 — Pipeline Optimization (10% of bottleneck)

- [ ] **Background command buffer encoding** — Encode CMD1 for layer N+1 on a background thread while GPU executes CMD3 for layer N. GCD dispatch_async + MTLCommandBuffer from shared queue. From general Metal best practices.
- [ ] **Layer-wise adaptive K** — Not all layers need the same number of experts. Early layers may work with K=2 while later layers need K=4. Measure per-layer routing entropy to auto-select K. From "Adaptive Computation in MoE" (various, 2024).
- [ ] **Speculative token execution** — Generate 2 candidate next tokens, run both through the first few layers, pick the winner early. For MoE, expert I/O scales per-token so this is risky. From "Sequoia" (Chen et al., 2024). Note: we tested MTP speculative decoding before (break-even) — this is a different approach.

## Priority 4 — Novel Techniques (from paper search)

- [ ] **Layer circuit duplication (adaptive depth)** — Repeat specific contiguous layer blocks (e.g. layers 20-25) during inference for harder prompts without any extra weights. "Functional circuits" across consecutive layers amplify reasoning ability. Use math/logic probes as heatmaps to identify which layer blocks to repeat. For simple prompts run once, for complex prompts run the circuit twice. Zero memory overhead. From DNH blog post (2025) on dense transformer layer duplication — needs adaptation for hybrid GatedDeltaNet + full attention architecture. Our 30 linear + 10 full attention layers may have different circuit patterns. Key question: does delta-net state accumulation remain stable through repeated layers?
- [ ] **Workload-aware page cache priming** — Use calibration-derived expert activation mass data to `madvise(MADV_WILLNEED)` on the hottest expert files at startup, improving cold-start cache hit rate from ~71% to potentially 85%+. From REAP-swap (Sero et al., 2025). Low effort (~20 lines), needs freq_data.json from calibration run.
- [ ] **MSE-optimal clipping for 2-bit** — Search 20 clipping ratios (0.7-1.0 of weight range) to minimize reconstruction MSE instead of using raw min/max. 15-30% RMSE reduction at 2-bit. May fix JSON corruption. From JANG (2025). Already implemented in `repack_experts_2bit.py --mse-clip` — needs benchmarking.
- [ ] **3-bit expert quantization** — Sweet spot between 2-bit (broken JSON) and 4-bit (209GB). 25% less I/O per expert with likely good-enough quality. Requires new Metal kernel for cross-byte-boundary bit extraction. From JANG adaptive quantization profiles.
- [ ] **DWQ/GPTQ error compensation** — Column-by-column Hessian-guided quantization that distributes rounding errors optimally. Makes 2-bit usable for JSON/tool calling. Already implemented in `gptq_requantize.py` — needs end-to-end validation with tool calling test.
- [ ] **Kahan compensated fp16 accumulation** — Use Kahan summation with fp16 accumulators to get fp16 throughput with fp32 accuracy bounds. Different from our current fp16 toggle which has no error compensation. From numerical analysis literature.
