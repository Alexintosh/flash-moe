# OOM Prevention Architecture

**Status**: Implemented across all code paths
**Relevant files**: `metal_ctx.h`, `layer_forward.h`, `generate.h`, `FlashMoEEngine.m`

## Problem

Flash-MoE runs on devices with as little as 12GB unified memory (iPhone 17) while managing hundreds of Metal buffers, CPU scratch allocations, and multi-gigabyte model weights. A single failed allocation can crash the app or corrupt inference output. On iOS, the system kills apps that exceed their memory budget (jetsam) with no warning beyond memory pressure notifications.

## Design Principle: Pre-Allocate, Check Everything, Fail Gracefully

The engine follows three rules:
1. **Pre-allocate at load time** -- all scratch buffers allocated once during `flashmoe_load()`, reused across all layers and tokens. No per-token malloc/free.
2. **Check every allocation** -- every `newBufferWithLength`, `calloc`, and `posix_memalign` return value is tested. Failures produce actionable error messages and early return (not abort).
3. **Monitor at runtime** -- iOS memory pressure and warning handlers cancel generation before the system kills the process.

## Allocation Hardening

### 1. Static Scratch Buffers (layer_forward.h)

30+ scratch buffers pre-allocated in `init_layer_scratch()` at model load:

```
s_normed, s_residual, s_attn_proj, s_h_post, s_h_mid,
s_gate_scores, s_spec_gate_scores, s_shared_gate, s_shared_up,
s_moe_out, s_shared_out, s_q_proj_out, s_k_proj_out, s_v_proj_out,
s_q, s_q_gate, s_attn_out, s_qkv_proj_out, s_z_proj_out,
s_beta_proj_out, s_alpha_proj_out, s_conv_out, s_out_vals, s_gated_out,
s_expert_out_cpu, s_gate_proj_out, s_up_proj_out, s_act_out,
s_shared_act, s_k_dequant, s_v_dequant
```

All checked in a single compound `if` statement. If any allocation fails, `init_layer_scratch()` returns -1 and the engine refuses to start inference.

**Impact**: Eliminates ~300 malloc/free per token. Each token traverses all layers, and each layer previously allocated temporary buffers for attention projections, expert routing, and MoE combine. Pre-allocation converts heap allocation to pointer reuse.

### 2. Metal Buffer Nil Checks (metal_ctx.h)

All 40+ Metal buffer allocations in `metal_setup()` are grouped and checked:

- Input/output buffers (`buf_input`, `buf_output`)
- Expert data buffers (`buf_expert_data`, `buf_expert_input`, `buf_expert_gate`, etc.)
- Multi-expert double-buffered slots (`buf_multi_expert_data[k]`, `buf_multi_expert_data_B[k]`)
- Shared expert buffers (`buf_shared_gate`, `buf_shared_up`, `buf_shared_act`, `buf_shared_out`)
- Residual and normalization buffers (`buf_residual`, `buf_h_mid`, `buf_sum_sq`)
- MoE combine buffers (`buf_moe_hidden`, `buf_combine_params`, `buf_cmd3_sum_sq`)
- KV cache buffers (`buf_kv_k[i]`, `buf_kv_v[i]` for each layer)
- FP8 scale buffers (`buf_kv_k_scales[i]`, `buf_kv_v_scales[i]`) when FP8 KV is enabled
- Attention scratch (`buf_attn_q`, `buf_attn_scores`, `buf_attn_out`, `buf_attn_gate`)
- Delta-net state (`buf_delta_state[i]`, `buf_conv_state[i]` per layer)
- Delta-net scratch (`buf_delta_q`, `buf_delta_k`, `buf_delta_v`, etc.)

Each check prints the buffer name, requested size, and device name to stderr, then returns early. Metal returns nil when the device cannot satisfy the allocation (typically when wired memory is exhausted).

### 3. calloc Guards

All CPU heap allocations use `calloc()` (zero-initialized) and check for NULL:
- Layer file descriptor arrays in `FlashMoEEngine.m`
- Tracking arrays (`alloc_tracking_arrays()`)
- Vocabulary and tokenizer data

### 4. posix_memalign for Expert I/O (metal_ctx.h)

Expert data double-buffers use 2MB-aligned allocation via `posix_memalign()`:
```c
int pa_ret1 = posix_memalign(&aligned_data,   2*1024*1024, expert_alloc_size);
int pa_ret2 = posix_memalign(&aligned_data_b, 2*1024*1024, expert_alloc_size);
```
2MB alignment matches macOS huge page size, enabling the OS to use huge pages for expert I/O buffers. The return value is checked and logged.

## iOS Memory Management

### 5. Memory Pressure Dispatch Source (FlashMoEEngine.m)

A GCD dispatch source monitors system memory pressure:
```
DISPATCH_SOURCE_TYPE_MEMORYPRESSURE
  DISPATCH_MEMORYPRESSURE_WARN | DISPATCH_MEMORYPRESSURE_CRITICAL
```

On `DISPATCH_MEMORYPRESSURE_CRITICAL`: sets `atomic_store(&ctx->cancelled, 1)`, which causes the generation loop to exit at the next token boundary. This is an **actionable** handler -- it stops generation, not just logs.

On `DISPATCH_MEMORYPRESSURE_WARN`: logs a warning (generation continues but the system is under pressure).

### 6. didReceiveMemoryWarning Observer (FlashMoEEngine.m)

A second line of defense via `UIApplicationDidReceiveMemoryWarningNotification`:
```objc
[[NSNotificationCenter defaultCenter]
    addObserverForName:UIApplicationDidReceiveMemoryWarningNotification ...]
```
Also cancels generation via `atomic_store(&ctx->cancelled, 1)`. This fires when UIKit receives a memory warning from the system, which may arrive on a different schedule than the dispatch source.

Both handlers are registered at model load and removed at unload.

### 7. Pre-Flight 500MB Check (FlashMoEEngine.m)

Before starting any generation (both `flashmoe_generate` and `flashmoe_generate_continuation`):
```c
size_t avail = os_proc_available_memory();
if (avail < 500 * 1024 * 1024) {
    snprintf(ctx->last_error, ..., "Insufficient memory (%.0f MB available, need 500+ MB)");
    return -1;
}
```

**Why 500MB**: Generation needs memory for expert I/O buffers (~27MB per token for K=4 at 4-bit), Metal command buffer overhead, and headroom for the OS page cache to service expert reads. 500MB provides a safety margin that prevents starting a generation that would immediately trigger memory pressure.

### 8. Adaptive Context Length (FlashMoEEngine.m)

At model load, `os_proc_available_memory()` determines the maximum KV cache size:
```
available_memory -> bytes_per_position (float32 or FP8) -> max_context
```
With FP8 KV cache enabled, `bytes_per_elem = 1` instead of `sizeof(float) = 4`, enabling 4x longer context within the same memory budget.

Additionally, `recommendedMaxWorkingSetSize` from the Metal device constrains the total GPU buffer allocation to stay within the wired memory budget.

## Prompt Length Caps

All prompt input paths cap token count to `cfg.max_seq_len`:
- System prompt tokenization (chat mode)
- User prompt tokenization (single-shot mode)
- HTTP API prompt tokenization (serve mode)

Excess tokens are silently truncated with a log message, preventing KV cache overflow.

## Architecture Decision: Why Pre-Allocate

The alternative -- allocating per-token -- was the original approach and caused two problems:

1. **Fragmentation**: 300 malloc/free per token over thousands of tokens fragments the heap. On iOS with 12GB, fragmentation can cause allocation failures even when total memory is available.

2. **Latency variance**: malloc can trigger page faults, madvise calls, or zone coalescing at unpredictable times. Pre-allocation moves all of this to model load, where latency is acceptable.

The pre-allocation cost is modest: ~2-3MB of scratch buffers for the 397B model (hidden_dim=4096, moe_intermediate=24576). This is negligible compared to the 200MB+ of Metal buffers.
