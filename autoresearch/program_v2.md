# Flash-MoE Autoresearch v2 — Paper-Guided Optimization

Autonomous, research-driven optimization of MoE inference throughput on Apple Silicon.

## Objective

**Maximize tok/s** on the fixed benchmark while maintaining output quality. Each experiment is guided by a specific technique from a research paper — not random parameter tuning.

## Setup

1. **Read context files** (in this order):
   - `autoresearch/findings.md` — what previous agents learned. DO NOT repeat failed approaches.
   - `autoresearch/research_queue.md` — prioritized technique queue with paper citations.
   - `autoresearch/seed_papers.md` — curated list of relevant papers.
   - `CLAUDE.md` — architecture overview, constraints, what worked/failed historically.
   - The modular source files (see Files You May Modify below).
   - `autoresearch/benchmark.sh` — the fixed benchmark harness. Do not modify.

2. **Create branch**: `git checkout -b autoresearch/<tag>` from `develop`. Propose a tag based on today's date.

3. **Verify environment**: Confirm `FLASH_MOE_MODEL` is set and `autoresearch/baseline.txt` exists. If not, tell the human to run `bash autoresearch/prepare.sh`.

4. **Run baseline**: `bash autoresearch/benchmark.sh 2>/dev/null` to establish the current starting point.

## The Research-Experiment Loop

**LOOP FOREVER:**

### Step 1: Pick a Technique

Read `research_queue.md`. Pick the **top-ranked unchecked technique**. If all are checked, go to Step 1b.

### Step 1b: Search for New Papers (every 10 experiments or when queue is empty)

Use WebSearch to find recent papers on:
- "MoE inference optimization 2024 2025"
- "Apple Silicon Metal GPU kernel optimization"
- "SSD-based LLM inference offloading"
- "expert routing prediction mixture of experts"
- "quantized matrix vector multiply GPU"
- "speculative decoding sparse models"

For each promising paper:
- Read the abstract and method section (use WebFetch on arxiv/paper URL)
- Extract the specific technique that could apply to Flash-MoE
- Add to `research_queue.md` with paper citation and expected impact
- Rank by: (a) relevance to our bottleneck breakdown, (b) implementation effort, (c) theoretical speedup

### Step 2: Read the Paper

Use WebSearch/WebFetch to find and read the specific paper section describing the technique. Understand:
- The exact algorithm or code change
- Why it works (theoretical basis)
- What hardware assumptions it makes (may need adaptation for Apple Silicon)
- Expected speedup and under what conditions

### Step 3: Form Hypothesis

Write a specific, testable hypothesis:
```
HYPOTHESIS: [Technique] from [Paper] should improve tok/s by ~X% because [reason].
IMPLEMENTATION: Change [specific function/kernel] to [specific change].
RISK: [What could go wrong — quality, stability, platform-specific issues]
```

### Step 4: Implement

Make the change in the appropriate module file(s). Keep changes minimal and focused — one technique per experiment.

### Step 5: Build & Benchmark

```bash
git add metal_infer/ && git commit -m "experiment: [technique] (from [paper])"
bash autoresearch/benchmark.sh 2>/dev/null
```

Parse the BENCH_RESULT line: `tok_s=X.XX math=PASS/FAIL json=PASS/FAIL status=OK/...`

### Step 6: Record & Decide

Append to `autoresearch/experiments.tsv`:
```
id	timestamp	commit	tok_s	math	json	status	paper	notes
```

**KEEP** if: `tok_s >= previous_best * 0.995` AND both quality gates PASS
**DISCARD** if: `tok_s < previous_best * 0.995` OR any quality gate FAIL → `git reset --hard HEAD~1`

### Step 7: Write Findings

Append to `autoresearch/findings.md`:
```markdown
## Experiment N: [Technique Name]
**Paper:** [Title] ([Author], [Year])
**Hypothesis:** [What we expected and why]
**Implementation:** [What we changed, 1-2 paragraphs with file/line references]
**Result:** tok_s=X.XX (baseline: Y.YY, delta: +/-Z%), math=P/F, json=P/F → KEEP/DISCARD
**Analysis:** [Why it worked or didn't. What we learned about the system.]
**Next:** [What this result suggests trying next]
```

### Step 8: Update Queue

In `research_queue.md`:
- Check off `[x]` the technique you just tested
- If the paper suggested related techniques, add them
- If findings suggest a new direction, add it with rationale

### Step 9: Repeat

Go to Step 1.

## Files You May Modify

```
metal_infer/infer.m          — unity build entry point
metal_infer/config.h         — model config, macros
metal_infer/timing.h         — timing globals, telemetry
metal_infer/fp8.h            — FP8 E4M3 support
metal_infer/weights.h        — weight loading, tensor manifest
metal_infer/cpu_kernels.h    — CPU compute kernels, tokenizer
metal_infer/metal_ctx.h      — Metal setup, buffer allocation
metal_infer/gpu_dispatch.h   — GPU dispatch, batched matmul
metal_infer/expert_io.h      — Expert I/O, parallel pread, cache
metal_infer/layer_forward.h  — Forward pass, attention, MoE routing
metal_infer/generate.h       — Generation loop, sampling
metal_infer/shaders.metal    — Metal compute kernels
```

## Files You Must NOT Modify

- `autoresearch/benchmark.sh` — the measurement harness is sacred
- `autoresearch/prepare.sh`
- `metal_infer/Makefile`
- `metal_infer/chat.m`, `main.m`, `tokenizer.h`
- Any Python files (`*.py`)
- Any iOS app files (`FlashMoE-iOS/`)
- `CLAUDE.md`

## Architecture Constraints — READ CAREFULLY

These are hard-won lessons from 58+ prior experiments. Violating them will waste time.

### The Unified Memory Constraint

On Apple Silicon, **SSD DMA and GPU compute share the same memory controller**. They cannot be profitably overlapped. The GPU's dequant kernels are bandwidth-saturated at ~418 GiB/s. Even small background SSD DMA causes disproportionate GPU latency spikes.

**Do NOT attempt:**
- Overlapping SSD reads with GPU compute (validated: net negative)
- dispatch_io (70% slower)
- F_RDADVISE / speculative prefetch (net 0%)
- mmap for expert files (5x slower)

### The Caching Constraint

**Trust the OS page cache.** Every custom expert cache we tried was slower:
- Metal LRU cache: -38%
- malloc cache: -20%
- LZ4 compressed cache: -13%
- Speculative early routing: -38%

### GPU Constraints

- Metal GPU is **memory-bandwidth-bound** at ~418 GiB/s
- No spin-polling (CPU thermal competes with GPU)
- Expert files are ~6.75MB each; NVMe doesn't care about scatter at this granularity

### What Already Works Well (don't reinvent)

- FMA dequant kernel: `fma(nibble, scale*x, bias*x)`
- Deferred CMD3: expert forward pass submitted without waiting
- BLAS delta-net: Accelerate framework for 64-head state recurrence
- GCD parallel pread: dispatch groups for K experts
- GPU fused attention: RoPE + QK norm fused
- Fused moe_combine_residual: single kernel for combine + residual + sigmoid gate
- CMD1+CMD2 merge: fewer command buffers for linear attention layers
- Delta-net kernel fusion: merged pass 2+3 in gated_delta_net_step
- FP16 accumulation kernels: half-precision variants of all dequant kernels
- FP8 KV cache: E4M3 quantized attention cache

### Bottleneck Breakdown (397B, 4-bit, M3 Max)

| Phase | Time/Layer | % Total | Notes |
|-------|-----------|---------|-------|
| Expert I/O (pread) | 2.41ms | 56% | THE bottleneck |
| GPU compute (CMD1+CMD2+CMD3) | 1.81ms | 42% | Bandwidth-saturated |
| CPU routing | 0.003ms | <1% | Negligible |
| Misc overhead | 0.05ms | 1% | Command encoding |

## Decision Rules

- **Keep** if: `tok_s >= previous_best * 0.995` AND both quality gates PASS
- **Discard** if: `tok_s < previous_best * 0.995` OR any quality gate FAIL
- If **3 consecutive discards** with similar approaches: move to a different area
- If **build fails**: quick fix (2 tries max), then revert
- After every **keep**: update baseline to new tok_s

## Simplicity Criterion

Simpler is better. A 0.5% improvement that adds 50 lines? Borderline. A 0.5% improvement from *removing* code? Definitely keep. The goal is a lean, fast engine.

## Safety

- Primary dev machine with 48GB unified RAM
- Do not allocate more than ~200MB of new Metal buffers
- Do not create files larger than 10MB
- Do not modify the build system or add dependencies
- If build fails, revert immediately

## NEVER STOP

Once the loop begins, do NOT pause to ask the human. The human may be asleep. Run autonomously until manually interrupted. If you run out of queued techniques, search for more papers. Each experiment takes ~2-5 minutes, so you can run ~12-30 per hour. The loop runs until the human stops you.
