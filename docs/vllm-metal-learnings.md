# Learnings from vllm-metal

**Status**: Key techniques adopted and implemented
**Source**: [vllm-project/vllm-metal](https://github.com/vllm-project/vllm-metal) -- Metal backend for vLLM
**Relevant files**: `metal_ctx.h`, `layer_forward.h`, `shaders.metal`, `fp8.h`

## Context

vllm-metal is a Metal backend for the vLLM serving framework, targeting Apple Silicon for batched LLM inference. While Flash-MoE is a single-sequence MoE inference engine (optimized for SSD expert streaming), vllm-metal solves related problems on the same hardware. We studied their Metal shader patterns, memory management, and kernel design to identify techniques applicable to our use case.

## What We Adopted

### 1. Wired Memory Limit API

**Source**: vllm-metal queries `MTLDevice.recommendedMaxWorkingSetSize` to stay within the device's wired memory budget.

**Our implementation** (`metal_ctx.h`): At `metal_setup()`, we query the Metal device and store the result:
```c
ctx->recommended_working_set = (size_t)[ctx->device recommendedMaxWorkingSetSize];
```

This value is used to constrain KV cache allocation in `FlashMoEEngine.m`. When the total Metal buffer footprint (fixed GPU buffers + per-layer KV caches) would exceed the wired budget, the context length is reduced to fit. This prevents Metal from evicting buffers to system memory, which causes severe latency spikes on the memory-bandwidth-sensitive dequant kernels.

### 2. FP8 E4M3 KV Cache

**Source**: vllm-metal implements FP8 E4M3 quantized KV caches with per-position scales and GPU inline dequantization.

**Our implementation**:
- `fp8.h`: CPU-side FP8 E4M3 encode/decode with per-tensor dynamic scaling (scale = absmax / 240.0, giving headroom below the 448 max representable value)
- `metal_ctx.h`: Allocates FP8 KV buffers (`uint8_t` instead of `float`) and separate scale buffers (`buf_kv_k_scales[i]`, `buf_kv_v_scales[i]`) when `g_use_fp8_kv` is enabled
- `shaders.metal`: Three kernel variants for FP8 attention:
  - `fused_attention_fp8` -- dedicated FP8 kernel (Kernel 9c)
  - `fused_attention_fc` -- function-constant specialized kernel (Kernel 9d, preferred)
  - GPU inline dequant reads `uchar*` KV bytes and `float*` scales, converting on the fly
- `layer_forward.h`: KV write path encodes float32 projections to FP8 before GPU upload; attention dispatch selects the FP8-aware kernel

**Memory savings**: For the 397B model (60 layers, 8 KV heads, 128 head_dim):
- Float32: ~60KB per position (60 layers x 8 heads x 128 dim x 4 bytes x 2 for K+V)
- FP8: ~15KB per position (same layout but 1 byte per element + small scale overhead)
- 4x reduction enables significantly longer context on memory-constrained devices

**Trade-off**: FP8 E4M3 has only 3 mantissa bits (vs 23 for float32), introducing quantization noise in the attention scores. For long-context scenarios where memory is the bottleneck, the trade-off is favorable. For short prompts with ample memory, float32 preserves maximum precision. Hence the opt-in design (`--fp8-kv` flag, default off).

### 3. Fused Online Softmax Attention (FlashAttention-style)

**Source**: vllm-metal implements a single-kernel attention with online softmax, processing KV positions in blocks.

**Our implementation** (`shaders.metal`, Kernels 9b/9c/9d):

The previous attention pipeline used 3 separate GPU dispatches per full-attention layer:
1. Q@K^T (score computation)
2. Softmax (normalization)
3. Scores@V (weighted value sum)

The fused kernel replaces all three with a single dispatch that iterates over KV positions in blocks of `BLOCK_SIZE=64`:

```
For each block of 64 KV positions:
  1. Compute partial scores: S_block = Q @ K_block^T
  2. Find block max: m_block = max(S_block)
  3. Update online softmax:
     m_new = max(m_old, m_block)
     correction = exp(m_old - m_new)
     l = l * correction + sum(exp(S_block - m_new))
     O = O * correction + exp(S_block - m_new) @ V_block
  4. Final: O = O / l
```

**Benefits**:
- 3 GPU dispatches reduced to 1 per full-attention layer (15 layers on 397B)
- Never materializes the full N x N attention matrix (memory savings for long sequences)
- Better cache utilization: K and V are read once per block instead of separately

**BLOCK_SIZE=64**: Chosen to balance threadgroup shared memory usage against block iteration count. 64 positions x head_dim floats fits comfortably in Metal's 32KB threadgroup memory limit while keeping the block count reasonable for typical context lengths.

### 4. Metal Function Constants (Compile-Time Specialization)

**Source**: vllm-metal uses function constants to specialize kernels at pipeline creation time.

**Our implementation** (`shaders.metal`, `metal_ctx.h`):

```metal
constant bool USE_FP8_KV [[function_constant(0)]];
```

The `fused_attention_fc` kernel (Kernel 9d) uses `USE_FP8_KV` to select between float32 and FP8 KV read paths at compile time. Metal's shader compiler eliminates dead branches based on the constant value, producing two specialized variants from a single source kernel.

Pipeline creation in `metal_ctx.h`:
```objc
MTLFunctionConstantValues *fc = [[MTLFunctionConstantValues alloc] init];
bool use_fp8 = g_use_fp8_kv;
[fc setConstantValue:&use_fp8 type:MTLDataTypeBool atIndex:0];
```

**Benefit**: Zero runtime overhead for the inactive path. When `USE_FP8_KV=false`, the FP8 dequant code is completely eliminated by the compiler. When `USE_FP8_KV=true`, the float32 read path is eliminated.

## What vllm-metal Has That We Don't Need

| vllm-metal Feature | Why We Skip It |
|---------------------|---------------|
| Batched inference (multi-sequence) | Flash-MoE is single-sequence; MoE expert I/O scales per-token, making batching counterproductive for SSD-streamed models |
| PagedAttention | Designed for serving many concurrent sequences; single-sequence uses contiguous KV cache |
| Continuous batching scheduler | Server-side concern; Flash-MoE is a local inference engine |
| Prefill chunking | Our bottleneck is expert I/O, not attention compute |

## What We Have That vllm-metal Doesn't

| Flash-MoE Feature | Why vllm-metal Doesn't Need It |
|--------------------|-------------------------------|
| SSD expert streaming (pread + GCD) | vllm-metal serves dense models that fit in memory |
| Tiered expert quantization (hot=4bit, cold=2bit) | No MoE expert management |
| GatedDeltaNet linear attention (BLAS recurrence) | Targets standard transformer architectures |
| Trust-the-OS page cache strategy | No multi-hundred-GB expert data to cache |
| K-reduction for mobile | No MoE routing to reduce |
| iOS universal app with SwiftUI | Server-side framework, not a mobile app |
