# Flash-MoE Autoresearch

Autonomous optimization of MoE inference throughput on Apple Silicon.

## Objective

**Maximize tok/s** on the fixed benchmark while maintaining output quality. The benchmark generates 200 tokens with a fixed prompt. Your metric is `tok_s` — higher is better.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar21`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from the current branch.
3. **Read the in-scope files** for full context:
   - `CLAUDE.md` — architecture overview, what worked, what failed.
   - `metal_infer/infer.m` — the inference engine (~7500 lines, Objective-C). This is your primary target.
   - `metal_infer/shaders.metal` — Metal compute kernels (~1300 lines). Secondary target.
   - `autoresearch/benchmark.sh` — the fixed benchmark harness. Do not modify.
4. **Verify model**: Confirm `FLASH_MOE_MODEL` is set and `autoresearch/baseline.txt` exists. If not, tell the human to run `bash autoresearch/prepare.sh`.
5. **Initialize experiments.tsv**: If starting fresh, create it with the header row plus baseline from `baseline.txt`.
6. **Confirm and go**: Confirm setup looks good, then start experimenting.

## Files You May Modify

- `metal_infer/infer.m` — the inference engine. Everything is fair game: pipeline scheduling, buffer management, dispatch patterns, kernel launch parameters, memory layout, data flow.
- `metal_infer/shaders.metal` — Metal compute kernels. Fair game: threadgroup sizes, tiling strategy, SIMD utilization, register pressure, kernel fusion, shared memory patterns.

## Files You Must NOT Modify

- `autoresearch/benchmark.sh` — the measurement harness is sacred
- `autoresearch/prepare.sh`
- `metal_infer/Makefile`
- `metal_infer/chat.m`, `main.m`, `tokenizer.h`
- Any Python files (`*.py`)
- `CLAUDE.md`

## The Experiment Loop

Each experiment: modify → build → benchmark → decide → repeat. Each iteration takes ~1-2 minutes.

**LOOP FOREVER:**

1. **Propose** a single, focused optimization hypothesis. Write it down.
2. **Implement** the change in `infer.m` and/or `shaders.metal`.
3. **Commit**: `git add metal_infer/infer.m metal_infer/shaders.metal && git commit -m "<description>"`
4. **Benchmark**: `bash autoresearch/benchmark.sh 2>/dev/null` (stderr has progress, stdout has the BENCH_RESULT line)
   - To see progress: `bash autoresearch/benchmark.sh` (without redirect)
5. **Parse** the BENCH_RESULT line: `tok_s=X.XX math=PASS/FAIL json=PASS/FAIL status=OK/...`
6. **If crashed or build failed**: `tail -20` the build output, attempt a quick fix. If unfixable after 2 tries, revert and log as `crash`.
7. **Record** in `autoresearch/experiments.tsv`
8. **Decide**:
   - **KEEP** if `tok_s >= previous_best * 0.995` AND both quality gates pass. Advance the branch.
   - **DISCARD** if `tok_s < previous_best * 0.995` OR any quality gate fails. Revert: `git reset --hard HEAD~1`
9. **Repeat**

## Architecture Constraints — READ CAREFULLY

These are hard-won lessons from 58+ prior experiments. Violating them will waste your time.

### The Unified Memory Constraint

On Apple Silicon, **SSD DMA and GPU compute share the same memory controller**. They cannot be profitably overlapped. The GPU's dequant kernels are bandwidth-saturated at ~418 GiB/s. Even small background SSD DMA causes disproportionate GPU latency spikes through memory controller arbitration. The serial pipeline (GPU → SSD → GPU) is hardware-optimal.

**Do NOT attempt:**
- Overlapping SSD reads with GPU compute
- dispatch_io (70% slower due to dispatch_data overhead)
- F_RDADVISE / speculative prefetch (net 0% — bandwidth waste)
- mmap for expert files (5x slower — per-page fault overhead)

### The Caching Constraint

**Trust the OS page cache.** Every custom expert cache we tried was slower:
- Metal LRU cache: -38% (steals GPU memory)
- malloc cache: -20% (steals from page cache)
- LZ4 compressed cache: -13% (decompression overhead)
- Speculative early routing: -38% (cache pollution)

The OS page cache (~35GB) does LRU better than we can. Do not add caching layers.

### GPU Constraints

- Metal GPU is **memory-bandwidth-bound** at ~418 GiB/s. The dequant kernels are already saturated. Pure ALU tricks (LUT, bit shifts) don't help because the bottleneck is reading data.
- **No spin-polling.** CPU thermal throttling competes with GPU on unified architecture. Use proper `waitUntilCompleted` or command buffer completion handlers.
- Expert files are ~6.75MB each. NVMe doesn't care about scatter at this granularity.

### What Already Works Well

Don't reinvent these — they're already optimized:
- **FMA dequant kernel**: `fma(nibble, scale*x, bias*x)` — pre-computes `scale*x` and `bias*x`
- **Deferred CMD3**: expert forward pass submitted without waiting, GPU/CPU overlap
- **BLAS delta-net**: Accelerate framework for the 64-head state recurrence
- **GCD parallel pread**: dispatch groups for K=4 expert I/O
- **GPU fused attention**: RoPE + QK norm fused
- **Fused moe_combine_residual**: single kernel for combine + residual + sigmoid gate

## Promising Areas to Explore

Ideas that haven't been tried or were partially explored:

1. **Shader occupancy tuning** — threadgroup sizes, simdgroup counts, register pressure in the dequant kernels. The current defaults may not be optimal for all matrix dimensions.

2. **Command buffer batching** — reducing Metal encode overhead by batching more operations into fewer command buffers. Currently 3 command buffers per layer.

3. **Pipeline stage rebalancing** — the bottleneck is expert I/O at 2.41ms per layer. Can any GPU work be restructured to reduce total wall time?

4. **Reduced precision in non-critical paths** — bf16/f16 for intermediate buffers where full f32 isn't needed. Apple GPU has native f16 ALU at 2x throughput.

5. **Kernel fusion** — combining small sequential kernels (e.g., norm + projection) to reduce dispatch overhead and intermediate buffer traffic.

6. **Attention layer optimization** — the full-attention layers use batched GPU attention. Room for improvement in tiling or memory access patterns.

7. **Thread coarsening in dequant kernels** — processing more output elements per thread to amortize the shared memory load of the input vector.

8. **SIMD shuffle patterns** — using simd_shuffle instead of threadgroup memory for reductions in the dequant kernels.

9. **Encode parallelism** — can command buffer encoding happen on a background thread while the previous buffer executes?

10. **Quantization-aware optimizations** — the 4-bit nibble extraction uses shifts and masks; explore whether packed byte operations are faster.

## Recording Results

Append to `autoresearch/experiments.tsv` (tab-separated, NOT comma-separated):

```
id	timestamp	commit	tok_s	math	json	status	notes
```

Example:
```
001	2026-03-21T02:15:00	abc1234	4.42	PASS	PASS	keep	Increased threadgroup size in matvec_v3 from 256 to 512
002	2026-03-21T02:17:30	def5678	4.35	PASS	PASS	discard	Tried simd_shuffle for reduction — register pressure
003	2026-03-21T02:19:45	ghi9012	0.00	SKIP	SKIP	crash	Fused norm+proj kernel — indexing bug
```

**Do NOT commit experiments.tsv** — leave it untracked by git.

## Decision Rules

- **Keep** if: `tok_s >= previous_best * 0.995` AND both quality gates PASS (0.5% noise margin to avoid discarding neutral-but-correct simplifications)
- **Discard** if: `tok_s < previous_best * 0.995` OR any quality gate FAIL
- If **3 consecutive discards** with similar approaches: move to a different optimization area
- If **build fails**: quick fix attempt (2 tries max), then revert and move on
- After every **keep**: update your mental baseline to the new tok_s
- Always **revert before starting a new experiment** (clean slate from last keep)

## Simplicity Criterion

All else being equal, simpler is better. A 0.5% improvement that adds 50 lines of complexity? Borderline. A 0.5% improvement from *removing* code? Definitely keep. An improvement of ~0 but simpler code? Keep. The goal is a lean, fast engine — not a pile of micro-optimizations.

## Safety

- This is a primary dev machine with 48GB unified RAM
- Do not allocate more than ~200MB of new Metal buffers
- Do not create files larger than 10MB
- Do not modify the build system or add dependencies
- If build fails, revert immediately
- Do not touch files outside `metal_infer/infer.m` and `metal_infer/shaders.metal`

## NEVER STOP

Once the loop begins, do NOT pause to ask the human for permission. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human may be asleep. You run autonomously until manually interrupted. If you run out of ideas, think harder — re-read the source for new angles, try combining near-misses, try more radical changes. Each experiment takes ~1-2 minutes, so you can run ~30-60 per hour. The loop runs until the human stops you.
