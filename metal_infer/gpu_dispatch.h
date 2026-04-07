// gpu_dispatch.h — Batched GPU matmul, encode-only variants, expert GPU forward
// Part of Flash-MoE modular decomposition (unity build).
// Included by infer.m — do NOT compile separately.

// ============================================================================
// Batched GPU matmul: encode N independent matmuls sharing the same input
// into ONE command buffer, reducing dispatch overhead by N-1 round-trips.
// ============================================================================

typedef struct {
    const void *W;           // packed weights (pointer into mmap'd file)
    const void *scales;      // scales (pointer into mmap'd file)
    const void *biases;      // biases (pointer into mmap'd file)
    float *out_cpu;          // CPU output pointer (result copied here after GPU finishes)
    uint32_t out_dim;
    uint32_t in_dim;
    uint32_t group_size;
    int batch_slot;          // which batch_out[slot] to use for GPU output
} BatchMatvecSpec;

// Run N matmuls in a single command buffer. All share the same input vector.
// The input is copied once; all outputs go to preallocated batch_out slots.
static void gpu_batch_matvec(
    MetalCtx *ctx,
    const float *x_f32, uint32_t x_dim,  // shared input
    BatchMatvecSpec *specs, int num_specs
) {
    // Copy input once
    memcpy([ctx->buf_input contents], x_f32, x_dim * sizeof(float));

    id<MTLCommandBuffer> cmdbuf = [ctx->queue commandBuffer];

    // Reset staging ONCE before the loop — all specs pack into the same staging buffer.
    // Each spec's data must remain valid until the command buffer completes.
    metal_staging_reset(ctx);

    for (int i = 0; i < num_specs; i++) {
        BatchMatvecSpec *s = &specs[i];
        id<MTLBuffer> w_buf, s_buf, b_buf;
        NSUInteger w_off, s_off, b_off;
        size_t w_size = (size_t)s->out_dim * s->in_dim / 8;
        size_t num_groups = (s->in_dim + s->group_size - 1) / s->group_size;
        size_t sb_size = (size_t)s->out_dim * num_groups * sizeof(uint16_t);
        metal_find_chunk_sized(ctx, s->W, w_size, &w_buf, &w_off);
        metal_find_chunk_sized(ctx, s->scales, sb_size, &s_buf, &s_off);
        metal_find_chunk_sized(ctx, s->biases, sb_size, &b_buf, &b_off);

        id<MTLBuffer> o_buf = ctx->batch_out[s->batch_slot];

        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        int use_v3 = (s->in_dim <= 4096);
        id<MTLComputePipelineState> pipe;
        if (g_use_fp16_accum && use_v3 && ctx->matvec_v3_fp16) {
            pipe = ctx->matvec_v3_fp16;
        } else {
            pipe = use_v3 ? ctx->matvec_v3 : ctx->matvec_fast;
        }
        [enc setComputePipelineState:pipe];
        [enc setBuffer:w_buf        offset:w_off atIndex:0];
        [enc setBuffer:s_buf        offset:s_off atIndex:1];
        [enc setBuffer:b_buf        offset:b_off atIndex:2];
        [enc setBuffer:ctx->buf_input offset:0   atIndex:3];
        [enc setBuffer:o_buf        offset:0     atIndex:4];
        [enc setBytes:&s->out_dim   length:4     atIndex:5];
        [enc setBytes:&s->in_dim    length:4     atIndex:6];
        [enc setBytes:&s->group_size length:4    atIndex:7];

        if (use_v3) {
            uint32_t num_tgs = (s->out_dim + 7) / 8;
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else {
            [enc dispatchThreadgroups:MTLSizeMake(s->out_dim, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }
        [enc endEncoding];
    }

    [cmdbuf commit];
    [cmdbuf waitUntilCompleted];

    // Copy results back to CPU
    for (int i = 0; i < num_specs; i++) {
        BatchMatvecSpec *s = &specs[i];
        memcpy(s->out_cpu, [ctx->batch_out[s->batch_slot] contents],
               s->out_dim * sizeof(float));
    }
}

// ============================================================================
// Encode-only variants: add dispatches to an EXISTING command buffer.
// These do NOT commit — the caller batches multiple encode calls into one
// command buffer and commits once, eliminating per-dispatch overhead.
// ============================================================================

// Encode N matmuls into cmdbuf. Input must already be in ctx->buf_input.
static void gpu_encode_batch_matvec(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    BatchMatvecSpec *specs, int num_specs
) {
    // Reset staging ONCE — all specs accumulate in the same staging buffer.
    // The caller's command buffer reads all staged data after commit.
    metal_staging_reset(ctx);

    for (int i = 0; i < num_specs; i++) {
        BatchMatvecSpec *s = &specs[i];
        id<MTLBuffer> w_buf, s_buf, b_buf;
        NSUInteger w_off, s_off, b_off;
        size_t w_size = (size_t)s->out_dim * s->in_dim / 8;
        size_t num_groups = (s->in_dim + s->group_size - 1) / s->group_size;
        size_t sb_size = (size_t)s->out_dim * num_groups * sizeof(uint16_t);
        metal_find_chunk_sized(ctx, s->W, w_size, &w_buf, &w_off);
        metal_find_chunk_sized(ctx, s->scales, sb_size, &s_buf, &s_off);
        metal_find_chunk_sized(ctx, s->biases, sb_size, &b_buf, &b_off);

        id<MTLBuffer> o_buf = ctx->batch_out[s->batch_slot];

        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        int use_v3 = (s->in_dim <= 4096);
        id<MTLComputePipelineState> pipe;
        if (g_use_fp16_accum && use_v3 && ctx->matvec_v3_fp16) {
            pipe = ctx->matvec_v3_fp16;
        } else {
            pipe = use_v3 ? ctx->matvec_v3 : ctx->matvec_fast;
        }
        [enc setComputePipelineState:pipe];
        [enc setBuffer:w_buf        offset:w_off atIndex:0];
        [enc setBuffer:s_buf        offset:s_off atIndex:1];
        [enc setBuffer:b_buf        offset:b_off atIndex:2];
        [enc setBuffer:ctx->buf_input offset:0   atIndex:3];
        [enc setBuffer:o_buf        offset:0     atIndex:4];
        [enc setBytes:&s->out_dim   length:4     atIndex:5];
        [enc setBytes:&s->in_dim    length:4     atIndex:6];
        [enc setBytes:&s->group_size length:4    atIndex:7];

        if (use_v3) {
            uint32_t num_tgs = (s->out_dim + 7) / 8;
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else {
            [enc dispatchThreadgroups:MTLSizeMake(s->out_dim, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }
        [enc endEncoding];
    }
}

// Copy batch results from GPU buffers back to CPU pointers.
static void gpu_flush_batch_results(MetalCtx *ctx, BatchMatvecSpec *specs, int num_specs) {
    for (int i = 0; i < num_specs; i++) {
        BatchMatvecSpec *s = &specs[i];
        memcpy(s->out_cpu, [ctx->batch_out[s->batch_slot] contents],
               s->out_dim * sizeof(float));
    }
}

// Encode a single matvec reading from buf_expert_act into buf_expert_out,
// using weight pointers into the mmap'd weight file.
// Used for shared expert down_proj which reads from a different input than
// the attention projections.
static void gpu_encode_dequant_matvec_with_io_bufs(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    const void *W, const void *scales, const void *biases,
    id<MTLBuffer> in_buf, id<MTLBuffer> out_buf,
    uint32_t out_dim, uint32_t in_dim, uint32_t group_size
) {
    id<MTLBuffer> w_buf, s_buf, b_buf;
    NSUInteger w_off, s_off, b_off;
    size_t w_size = (size_t)out_dim * in_dim / 8;  // 4-bit packed
    size_t num_groups = (in_dim + group_size - 1) / group_size;
    size_t sb_size = (size_t)out_dim * num_groups * sizeof(uint16_t);
    metal_staging_reset(ctx);
    metal_find_chunk_sized(ctx, W, w_size, &w_buf, &w_off);
    metal_find_chunk_sized(ctx, scales, sb_size, &s_buf, &s_off);
    metal_find_chunk_sized(ctx, biases, sb_size, &b_buf, &b_off);

    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
    int use_v3 = (in_dim <= 4096);
    id<MTLComputePipelineState> pipe;
    if (g_use_fp16_accum && use_v3 && ctx->matvec_v3_fp16) {
        pipe = ctx->matvec_v3_fp16;
    } else {
        pipe = use_v3 ? ctx->matvec_v3 : ctx->matvec_fast;
    }
    [enc setComputePipelineState:pipe];
    [enc setBuffer:w_buf offset:w_off atIndex:0];
    [enc setBuffer:s_buf offset:s_off atIndex:1];
    [enc setBuffer:b_buf offset:b_off atIndex:2];
    [enc setBuffer:in_buf      offset:0     atIndex:3];
    [enc setBuffer:out_buf     offset:0     atIndex:4];
    [enc setBytes:&out_dim     length:4     atIndex:5];
    [enc setBytes:&in_dim      length:4     atIndex:6];
    [enc setBytes:&group_size  length:4     atIndex:7];

    if (use_v3) {
        uint32_t num_tgs = (out_dim + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
        [enc dispatchThreadgroups:MTLSizeMake(out_dim, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }
    [enc endEncoding];
}

// Encode one expert forward using multi-expert slot k.
// Expert data must already be in buf_multi_expert_data[k].
// Input must already be in buf_multi_expert_input.
static void gpu_encode_expert_forward_slot(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    int k  // slot index
) {
    NSUInteger gate_w_off, gate_s_off, gate_b_off;
    NSUInteger up_w_off, up_s_off, up_b_off;
    NSUInteger down_w_off, down_s_off, down_b_off;
    if (g_use_2bit) {
        gate_w_off = cfg.gate_w_off_2; gate_s_off = cfg.gate_s_off_2; gate_b_off = cfg.gate_b_off_2;
        up_w_off   = cfg.up_w_off_2;   up_s_off   = cfg.up_s_off_2;   up_b_off   = cfg.up_b_off_2;
        down_w_off = cfg.down_w_off_2; down_s_off = cfg.down_s_off_2; down_b_off = cfg.down_b_off_2;
    } else {
        gate_w_off = cfg.gate_w_off_4; gate_s_off = cfg.gate_s_off_4; gate_b_off = cfg.gate_b_off_4;
        up_w_off   = cfg.up_w_off_4;   up_s_off   = cfg.up_s_off_4;   up_b_off   = cfg.up_b_off_4;
        down_w_off = cfg.down_w_off_4;  down_s_off = cfg.down_s_off_4;  down_b_off = cfg.down_b_off_4;
    }
    id<MTLComputePipelineState> expert_pipe;
    if (g_use_2bit) {
        expert_pipe = (g_use_fp16_accum && ctx->matvec_2bit_fp16) ? ctx->matvec_2bit_fp16 : ctx->matvec_2bit;
    } else {
        expert_pipe = (g_use_fp16_accum && ctx->matvec_v3_fp16) ? ctx->matvec_v3_fp16 : ctx->matvec_v3;
    }

    uint32_t gate_up_out = cfg.moe_intermediate;
    uint32_t gate_up_in  = cfg.hidden_dim;
    uint32_t down_out    = cfg.hidden_dim;
    uint32_t down_in     = cfg.moe_intermediate;
    uint32_t gs          = cfg.group_size;

    uint32_t num_tgs = (gate_up_out + 7) / 8;

    // 4-bit: fused gate+up+SwiGLU; 2-bit: separate dispatches
    // NOTE: fused kernel = 1 TG per row (not ROWS_PER_TG=8)
    if (!g_use_2bit && g_fused_expert_enabled && ctx->fused_gate_up) {
        // fused_gate_up_swiglu: data[k] -> act[k] directly
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        id<MTLComputePipelineState> fused_pipe = (g_use_fp16_accum && ctx->fused_gate_up_fp16)
            ? ctx->fused_gate_up_fp16 : ctx->fused_gate_up;
        [enc setComputePipelineState:fused_pipe];
        [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:gate_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:gate_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:gate_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:up_w_off    atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:up_s_off    atIndex:4];
        [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:up_b_off    atIndex:5];
        [enc setBuffer:ctx->buf_multi_expert_input     offset:0           atIndex:6];
        [enc setBuffer:ctx->buf_multi_expert_act[k]    offset:0           atIndex:7];
        [enc setBytes:&gate_up_out length:4 atIndex:8];
        [enc setBytes:&gate_up_in  length:4 atIndex:9];
        [enc setBytes:&gs          length:4 atIndex:10];
        [enc dispatchThreadgroups:MTLSizeMake(gate_up_out, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    } else {
        // gate_proj: data[k] -> gate[k]
        {
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            [enc setComputePipelineState:expert_pipe];
            [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:gate_w_off  atIndex:0];
            [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:gate_s_off  atIndex:1];
            [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:gate_b_off  atIndex:2];
            [enc setBuffer:ctx->buf_multi_expert_input     offset:0           atIndex:3];
            [enc setBuffer:ctx->buf_multi_expert_gate[k]   offset:0           atIndex:4];
            [enc setBytes:&gate_up_out length:4 atIndex:5];
            [enc setBytes:&gate_up_in  length:4 atIndex:6];
            [enc setBytes:&gs          length:4 atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }
        // up_proj: data[k] -> up[k]
        {
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            [enc setComputePipelineState:expert_pipe];
            [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:up_w_off  atIndex:0];
            [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:up_s_off  atIndex:1];
            [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:up_b_off  atIndex:2];
            [enc setBuffer:ctx->buf_multi_expert_input     offset:0          atIndex:3];
            [enc setBuffer:ctx->buf_multi_expert_up[k]     offset:0          atIndex:4];
            [enc setBytes:&gate_up_out length:4 atIndex:5];
            [enc setBytes:&gate_up_in  length:4 atIndex:6];
            [enc setBytes:&gs          length:4 atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }
        // SwiGLU: gate[k], up[k] -> act[k]
        {
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            [enc setComputePipelineState:ctx->swiglu];
            [enc setBuffer:ctx->buf_multi_expert_gate[k] offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_multi_expert_up[k]   offset:0 atIndex:1];
            [enc setBuffer:ctx->buf_multi_expert_act[k]  offset:0 atIndex:2];
            [enc setBytes:&gate_up_out length:4 atIndex:3];
            uint32_t swiglu_tgs = (gate_up_out + 255) / 256;
            [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }
    }
    // down_proj: act[k] -> out[k]
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:ctx->buf_multi_expert_data[k] offset:down_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_multi_expert_data[k] offset:down_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_multi_expert_data[k] offset:down_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_multi_expert_act[k]  offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_out[k]  offset:0           atIndex:4];
        [enc setBytes:&down_out length:4 atIndex:5];
        [enc setBytes:&down_in  length:4 atIndex:6];
        [enc setBytes:&gs       length:4 atIndex:7];
        uint32_t num_tgs = (down_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
}

// Encode one expert forward using explicit data buffer (for double buffering).
// Expert data must already be in data_buf.
// Input must already be in buf_multi_expert_input.
// Uses slot k's gate/up/act/out scratch buffers.
static void gpu_encode_expert_forward_slot_buf(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    int k,                  // slot index (for gate/up/act/out scratch)
    id<MTLBuffer> data_buf  // expert weight data buffer (from either set A or B)
) {
    NSUInteger gate_w_off, gate_s_off, gate_b_off;
    NSUInteger up_w_off, up_s_off, up_b_off;
    NSUInteger down_w_off, down_s_off, down_b_off;
    if (g_use_2bit) {
        gate_w_off = cfg.gate_w_off_2; gate_s_off = cfg.gate_s_off_2; gate_b_off = cfg.gate_b_off_2;
        up_w_off   = cfg.up_w_off_2;   up_s_off   = cfg.up_s_off_2;   up_b_off   = cfg.up_b_off_2;
        down_w_off = cfg.down_w_off_2; down_s_off = cfg.down_s_off_2; down_b_off = cfg.down_b_off_2;
    } else {
        gate_w_off = cfg.gate_w_off_4; gate_s_off = cfg.gate_s_off_4; gate_b_off = cfg.gate_b_off_4;
        up_w_off   = cfg.up_w_off_4;   up_s_off   = cfg.up_s_off_4;   up_b_off   = cfg.up_b_off_4;
        down_w_off = cfg.down_w_off_4;  down_s_off = cfg.down_s_off_4;  down_b_off = cfg.down_b_off_4;
    }
    id<MTLComputePipelineState> expert_pipe;
    if (g_use_2bit) {
        expert_pipe = (g_use_fp16_accum && ctx->matvec_2bit_fp16) ? ctx->matvec_2bit_fp16 : ctx->matvec_2bit;
    } else {
        expert_pipe = (g_use_fp16_accum && ctx->matvec_v3_fp16) ? ctx->matvec_v3_fp16 : ctx->matvec_v3;
    }

    uint32_t gate_up_out = cfg.moe_intermediate;
    uint32_t gate_up_in  = cfg.hidden_dim;
    uint32_t down_out    = cfg.hidden_dim;
    uint32_t down_in     = cfg.moe_intermediate;
    uint32_t gs          = cfg.group_size;

    uint32_t num_tgs = (gate_up_out + 7) / 8;

    // 4-bit: fused gate+up+SwiGLU; 2-bit: separate dispatches
    // NOTE: fused kernel = 1 TG per row (not ROWS_PER_TG=8)
    if (!g_use_2bit && g_fused_expert_enabled && ctx->fused_gate_up) {
        // fused_gate_up_swiglu: data_buf -> act[k] directly
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        id<MTLComputePipelineState> fused_pipe = (g_use_fp16_accum && ctx->fused_gate_up_fp16)
            ? ctx->fused_gate_up_fp16 : ctx->fused_gate_up;
        [enc setComputePipelineState:fused_pipe];
        [enc setBuffer:data_buf                        offset:gate_w_off  atIndex:0];
        [enc setBuffer:data_buf                        offset:gate_s_off  atIndex:1];
        [enc setBuffer:data_buf                        offset:gate_b_off  atIndex:2];
        [enc setBuffer:data_buf                        offset:up_w_off    atIndex:3];
        [enc setBuffer:data_buf                        offset:up_s_off    atIndex:4];
        [enc setBuffer:data_buf                        offset:up_b_off    atIndex:5];
        [enc setBuffer:ctx->buf_multi_expert_input     offset:0           atIndex:6];
        [enc setBuffer:ctx->buf_multi_expert_act[k]    offset:0           atIndex:7];
        [enc setBytes:&gate_up_out length:4 atIndex:8];
        [enc setBytes:&gate_up_in  length:4 atIndex:9];
        [enc setBytes:&gs          length:4 atIndex:10];
        [enc dispatchThreadgroups:MTLSizeMake(gate_up_out, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    } else {
        // gate_proj
        {
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            [enc setComputePipelineState:expert_pipe];
            [enc setBuffer:data_buf                        offset:gate_w_off  atIndex:0];
            [enc setBuffer:data_buf                        offset:gate_s_off  atIndex:1];
            [enc setBuffer:data_buf                        offset:gate_b_off  atIndex:2];
            [enc setBuffer:ctx->buf_multi_expert_input     offset:0           atIndex:3];
            [enc setBuffer:ctx->buf_multi_expert_gate[k]   offset:0           atIndex:4];
            [enc setBytes:&gate_up_out length:4 atIndex:5];
            [enc setBytes:&gate_up_in  length:4 atIndex:6];
            [enc setBytes:&gs          length:4 atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }
        // up_proj
        {
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            [enc setComputePipelineState:expert_pipe];
            [enc setBuffer:data_buf                        offset:up_w_off  atIndex:0];
            [enc setBuffer:data_buf                        offset:up_s_off  atIndex:1];
            [enc setBuffer:data_buf                        offset:up_b_off  atIndex:2];
            [enc setBuffer:ctx->buf_multi_expert_input     offset:0          atIndex:3];
            [enc setBuffer:ctx->buf_multi_expert_up[k]     offset:0          atIndex:4];
            [enc setBytes:&gate_up_out length:4 atIndex:5];
            [enc setBytes:&gate_up_in  length:4 atIndex:6];
            [enc setBytes:&gs          length:4 atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }
        // SwiGLU
        {
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            [enc setComputePipelineState:ctx->swiglu];
            [enc setBuffer:ctx->buf_multi_expert_gate[k] offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_multi_expert_up[k]   offset:0 atIndex:1];
            [enc setBuffer:ctx->buf_multi_expert_act[k]  offset:0 atIndex:2];
            [enc setBytes:&gate_up_out length:4 atIndex:3];
            uint32_t swiglu_tgs = (gate_up_out + 255) / 256;
            [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }
    }
    // down_proj
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:data_buf                        offset:down_w_off  atIndex:0];
        [enc setBuffer:data_buf                        offset:down_s_off  atIndex:1];
        [enc setBuffer:data_buf                        offset:down_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_multi_expert_act[k]    offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_out[k]    offset:0           atIndex:4];
        [enc setBytes:&down_out length:4 atIndex:5];
        [enc setBytes:&down_in  length:4 atIndex:6];
        [enc setBytes:&gs       length:4 atIndex:7];
        uint32_t num_tgs = (down_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
}

// Batched expert encoding: encode K experts using 2 encoders per expert + 2 for shared.
// 4-bit path: fused_gate_up_swiglu (1 dispatch) + down_proj = 2 encoders per expert.
// 2-bit fallback: gate+up (1 enc) + SwiGLU+down (1 enc) = 2 encoders per expert.
// With K=4: 10 encoders total (K*2 + 2 shared).
// Each expert gets its own encoder pair for GPU parallelism across experts.
static void gpu_encode_experts_batched(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    int K,                       // number of experts to encode
    const int *valid,            // which experts are valid [MAX_K]
    id<MTLBuffer> __strong *expert_bufs,   // per-expert weight data buffers [MAX_K]
    int layer_idx,               // layer index (for tiered manifest lookup)
    const int *expert_indices    // expert indices (for tiered per-expert quant)
) {
    uint32_t gate_up_out = cfg.moe_intermediate;
    uint32_t gate_up_in  = cfg.hidden_dim;
    uint32_t down_out    = cfg.hidden_dim;
    uint32_t down_in     = cfg.moe_intermediate;
    uint32_t gs          = cfg.group_size;
    // Threadgroup count is the same for 2-bit and 4-bit (based on out_dim).
    // The kernel handles packed_cols internally.
    uint32_t gate_up_tgs = (gate_up_out + 7) / 8;
    uint32_t down_tgs    = (down_out + 7) / 8;
    uint32_t swiglu_tgs  = (gate_up_out + 255) / 256;

    // Per-expert: Encoder A (gate+up), Encoder B (SwiGLU+down)
    // Separate encoders per expert enables GPU parallelism across experts.
    // Within each encoder, operations serialize (gate then up, SwiGLU then down).
    for (int k = 0; k < K; k++) {
        if (!valid[k]) continue;

        // Per-expert quantization selection (tiered: each expert may differ)
        int use_2bit_k;
        if (g_use_tiered && g_tiered_manifest) {
            use_2bit_k = (TIERED(layer_idx, expert_indices[k]).bits == 2);
        } else {
            use_2bit_k = g_use_2bit;
        }

        NSUInteger gate_w_off, gate_s_off, gate_b_off;
        NSUInteger up_w_off, up_s_off, up_b_off;
        NSUInteger down_w_off, down_s_off, down_b_off;
        id<MTLComputePipelineState> expert_pipe;

        if (use_2bit_k) {
            gate_w_off = cfg.gate_w_off_2; gate_s_off = cfg.gate_s_off_2; gate_b_off = cfg.gate_b_off_2;
            up_w_off   = cfg.up_w_off_2;   up_s_off   = cfg.up_s_off_2;   up_b_off   = cfg.up_b_off_2;
            down_w_off = cfg.down_w_off_2; down_s_off = cfg.down_s_off_2; down_b_off = cfg.down_b_off_2;
            expert_pipe = (g_use_fp16_accum && ctx->matvec_2bit_fp16) ? ctx->matvec_2bit_fp16 : ctx->matvec_2bit;
        } else {
            gate_w_off = cfg.gate_w_off_4; gate_s_off = cfg.gate_s_off_4; gate_b_off = cfg.gate_b_off_4;
            up_w_off   = cfg.up_w_off_4;   up_s_off   = cfg.up_s_off_4;   up_b_off   = cfg.up_b_off_4;
            down_w_off = cfg.down_w_off_4; down_s_off = cfg.down_s_off_4; down_b_off = cfg.down_b_off_4;
            expert_pipe = (g_use_fp16_accum && ctx->matvec_v3_fp16) ? ctx->matvec_v3_fp16 : ctx->matvec_v3;
        }

        // 4-bit path: fused gate+up+SwiGLU kernel (1 dispatch instead of 3)
        // 2-bit path: fallback to separate gate, up, SwiGLU dispatches
        if (!use_2bit_k && g_fused_expert_enabled && ctx->fused_gate_up) {
            // Encoder A: fused_gate_up_swiglu -> act[k] directly
            // NOTE: fused kernel uses 1 threadgroup per output row (like matvec_fast),
            // NOT ROWS_PER_TG=8 rows per threadgroup (like matvec_v3).
            {
                id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
                id<MTLComputePipelineState> fused_pipe = (g_use_fp16_accum && ctx->fused_gate_up_fp16)
                    ? ctx->fused_gate_up_fp16 : ctx->fused_gate_up;
                [enc setComputePipelineState:fused_pipe];
                [enc setBuffer:expert_bufs[k]                  offset:gate_w_off  atIndex:0];
                [enc setBuffer:expert_bufs[k]                  offset:gate_s_off  atIndex:1];
                [enc setBuffer:expert_bufs[k]                  offset:gate_b_off  atIndex:2];
                [enc setBuffer:expert_bufs[k]                  offset:up_w_off    atIndex:3];
                [enc setBuffer:expert_bufs[k]                  offset:up_s_off    atIndex:4];
                [enc setBuffer:expert_bufs[k]                  offset:up_b_off    atIndex:5];
                [enc setBuffer:ctx->buf_multi_expert_input     offset:0           atIndex:6];
                [enc setBuffer:ctx->buf_multi_expert_act[k]    offset:0           atIndex:7];
                [enc setBytes:&gate_up_out length:4 atIndex:8];
                [enc setBytes:&gate_up_in  length:4 atIndex:9];
                [enc setBytes:&gs          length:4 atIndex:10];
                [enc dispatchThreadgroups:MTLSizeMake(gate_up_out, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }

            // Encoder B: down_proj only (reads from act[k])
            {
                id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
                [enc setComputePipelineState:expert_pipe];
                [enc setBuffer:expert_bufs[k]                  offset:down_w_off  atIndex:0];
                [enc setBuffer:expert_bufs[k]                  offset:down_s_off  atIndex:1];
                [enc setBuffer:expert_bufs[k]                  offset:down_b_off  atIndex:2];
                [enc setBuffer:ctx->buf_multi_expert_act[k]    offset:0           atIndex:3];
                [enc setBuffer:ctx->buf_multi_expert_out[k]    offset:0           atIndex:4];
                [enc setBytes:&down_out length:4 atIndex:5];
                [enc setBytes:&down_in  length:4 atIndex:6];
                [enc setBytes:&gs       length:4 atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(down_tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
        } else {
            // Fallback: separate gate + up + SwiGLU (for 2-bit or if fused pipeline unavailable)
            // Encoder A: gate_proj + up_proj
            {
                id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
                // gate_proj
                [enc setComputePipelineState:expert_pipe];
                [enc setBuffer:expert_bufs[k]                  offset:gate_w_off  atIndex:0];
                [enc setBuffer:expert_bufs[k]                  offset:gate_s_off  atIndex:1];
                [enc setBuffer:expert_bufs[k]                  offset:gate_b_off  atIndex:2];
                [enc setBuffer:ctx->buf_multi_expert_input     offset:0           atIndex:3];
                [enc setBuffer:ctx->buf_multi_expert_gate[k]   offset:0           atIndex:4];
                [enc setBytes:&gate_up_out length:4 atIndex:5];
                [enc setBytes:&gate_up_in  length:4 atIndex:6];
                [enc setBytes:&gs          length:4 atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(gate_up_tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                // up_proj (same encoder, serialized after gate)
                [enc setBuffer:expert_bufs[k]                  offset:up_w_off  atIndex:0];
                [enc setBuffer:expert_bufs[k]                  offset:up_s_off  atIndex:1];
                [enc setBuffer:expert_bufs[k]                  offset:up_b_off  atIndex:2];
                [enc setBuffer:ctx->buf_multi_expert_up[k]     offset:0          atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(gate_up_tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }

            // Encoder B: SwiGLU + down_proj
            {
                id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
                // SwiGLU
                [enc setComputePipelineState:ctx->swiglu];
                [enc setBuffer:ctx->buf_multi_expert_gate[k] offset:0 atIndex:0];
                [enc setBuffer:ctx->buf_multi_expert_up[k]   offset:0 atIndex:1];
                [enc setBuffer:ctx->buf_multi_expert_act[k]  offset:0 atIndex:2];
                [enc setBytes:&gate_up_out length:4 atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                // down_proj (same encoder, serialized after SwiGLU)
                [enc setComputePipelineState:expert_pipe];
                [enc setBuffer:expert_bufs[k]                  offset:down_w_off  atIndex:0];
                [enc setBuffer:expert_bufs[k]                  offset:down_s_off  atIndex:1];
                [enc setBuffer:expert_bufs[k]                  offset:down_b_off  atIndex:2];
                [enc setBuffer:ctx->buf_multi_expert_act[k]    offset:0           atIndex:3];
                [enc setBuffer:ctx->buf_multi_expert_out[k]    offset:0           atIndex:4];
                [enc setBytes:&down_out length:4 atIndex:5];
                [enc setBytes:&down_in  length:4 atIndex:6];
                [enc setBytes:&gs       length:4 atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(down_tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
        }
    }
}

// Encode one expert forward (gate+up+swiglu+down) into cmdbuf.
// Expert data must already be in buf_expert_data.
// Input must already be in buf_expert_input.
__attribute__((unused))
static void gpu_encode_expert_forward(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf
) {
    NSUInteger gate_w_off = cfg.gate_w_off_4;
    NSUInteger gate_s_off = cfg.gate_s_off_4;
    NSUInteger gate_b_off = cfg.gate_b_off_4;
    NSUInteger up_w_off   = cfg.up_w_off_4;
    NSUInteger up_s_off   = cfg.up_s_off_4;
    NSUInteger up_b_off   = cfg.up_b_off_4;
    NSUInteger down_w_off = cfg.down_w_off_4;
    NSUInteger down_s_off = cfg.down_s_off_4;
    NSUInteger down_b_off = cfg.down_b_off_4;

    uint32_t gate_up_out = cfg.moe_intermediate;
    uint32_t gate_up_in  = cfg.hidden_dim;
    uint32_t down_out    = cfg.hidden_dim;
    uint32_t down_in     = cfg.moe_intermediate;
    uint32_t gs          = cfg.group_size;

    uint32_t num_tgs = (gate_up_out + 7) / 8;

    // Always 4-bit in this path: use fused kernel if available
    // NOTE: fused kernel = 1 TG per row (not ROWS_PER_TG=8)
    if (g_fused_expert_enabled && ctx->fused_gate_up) {
        // fused_gate_up_swiglu: expert_data -> expert_act directly
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        id<MTLComputePipelineState> fused_pipe = (g_use_fp16_accum && ctx->fused_gate_up_fp16)
            ? ctx->fused_gate_up_fp16 : ctx->fused_gate_up;
        [enc setComputePipelineState:fused_pipe];
        [enc setBuffer:ctx->buf_expert_data  offset:gate_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_expert_data  offset:gate_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_expert_data  offset:gate_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_expert_data  offset:up_w_off    atIndex:3];
        [enc setBuffer:ctx->buf_expert_data  offset:up_s_off    atIndex:4];
        [enc setBuffer:ctx->buf_expert_data  offset:up_b_off    atIndex:5];
        [enc setBuffer:ctx->buf_expert_input offset:0           atIndex:6];
        [enc setBuffer:ctx->buf_expert_act   offset:0           atIndex:7];
        [enc setBytes:&gate_up_out length:4 atIndex:8];
        [enc setBytes:&gate_up_in  length:4 atIndex:9];
        [enc setBytes:&gs          length:4 atIndex:10];
        [enc dispatchThreadgroups:MTLSizeMake(gate_up_out, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    } else {
        id<MTLComputePipelineState> mv_pipe = (g_use_fp16_accum && ctx->matvec_v3_fp16) ? ctx->matvec_v3_fp16 : ctx->matvec_v3;
        // gate_proj
        {
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            [enc setComputePipelineState:mv_pipe];
            [enc setBuffer:ctx->buf_expert_data  offset:gate_w_off  atIndex:0];
            [enc setBuffer:ctx->buf_expert_data  offset:gate_s_off  atIndex:1];
            [enc setBuffer:ctx->buf_expert_data  offset:gate_b_off  atIndex:2];
            [enc setBuffer:ctx->buf_expert_input offset:0           atIndex:3];
            [enc setBuffer:ctx->buf_expert_gate  offset:0           atIndex:4];
            [enc setBytes:&gate_up_out length:4 atIndex:5];
            [enc setBytes:&gate_up_in  length:4 atIndex:6];
            [enc setBytes:&gs          length:4 atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }
        // up_proj
        {
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            [enc setComputePipelineState:mv_pipe];
            [enc setBuffer:ctx->buf_expert_data  offset:up_w_off  atIndex:0];
            [enc setBuffer:ctx->buf_expert_data  offset:up_s_off  atIndex:1];
            [enc setBuffer:ctx->buf_expert_data  offset:up_b_off  atIndex:2];
            [enc setBuffer:ctx->buf_expert_input offset:0          atIndex:3];
            [enc setBuffer:ctx->buf_expert_up    offset:0          atIndex:4];
            [enc setBytes:&gate_up_out length:4 atIndex:5];
            [enc setBytes:&gate_up_in  length:4 atIndex:6];
            [enc setBytes:&gs          length:4 atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }
        // SwiGLU
        {
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            [enc setComputePipelineState:ctx->swiglu];
            [enc setBuffer:ctx->buf_expert_gate offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_expert_up   offset:0 atIndex:1];
            [enc setBuffer:ctx->buf_expert_act  offset:0 atIndex:2];
            [enc setBytes:&gate_up_out length:4 atIndex:3];
            uint32_t swiglu_tgs = (gate_up_out + 255) / 256;
            [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }
    }
    // down_proj
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        id<MTLComputePipelineState> down_pipe = (g_use_fp16_accum && ctx->matvec_v3_fp16) ? ctx->matvec_v3_fp16 : ctx->matvec_v3;
        [enc setComputePipelineState:down_pipe];
        [enc setBuffer:ctx->buf_expert_data offset:down_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_expert_data offset:down_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_expert_data offset:down_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_expert_act  offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_expert_out  offset:0           atIndex:4];
        [enc setBytes:&down_out length:4 atIndex:5];
        [enc setBytes:&down_in  length:4 atIndex:6];
        [enc setBytes:&gs       length:4 atIndex:7];
        uint32_t num_tgs = (down_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
}

// Batched wrapper: takes N matmul specs sharing the same input, dispatches
// via GPU batch if available, otherwise falls back to CPU.
static void fast_batch_matvec(
    const float *x, uint32_t x_dim,
    BatchMatvecSpec *specs, int num_specs
) {
    if (g_metal && (g_metal->wf_num_chunks > 0 || g_metal->wf_staging)) {
        gpu_batch_matvec(g_metal, x, x_dim, specs, num_specs);
    } else {
        for (int i = 0; i < num_specs; i++) {
            BatchMatvecSpec *s = &specs[i];
            cpu_dequant_matvec(s->W, s->scales, s->biases, x, s->out_cpu,
                               s->out_dim, s->in_dim, s->group_size);
        }
    }
}

// ============================================================================
// GPU expert forward: gate+up matvec -> SwiGLU -> down matvec
// All 3 matmuls + activation in a single command buffer submission.
// Expert data is copied into a reusable Metal buffer.
// ============================================================================

// expert_data_already_in_buffer: if true, expert data is already in buf_expert_data
//   (pread'd directly into it), skip the copy.
__attribute__((unused))
static void gpu_expert_forward(
    MetalCtx *ctx,
    const void *expert_data,     // cfg.expert_size_4bit bytes (may be buf_expert_data contents)
    const float *h_post,         // [cfg.hidden_dim] input
    float *expert_out,           // [cfg.hidden_dim] output
    int expert_data_already_in_buffer
) {
    // Expert layout offsets — select based on quantization mode
    NSUInteger gate_w_off, gate_s_off, gate_b_off;
    NSUInteger up_w_off, up_s_off, up_b_off;
    NSUInteger down_w_off, down_s_off, down_b_off;
    if (g_use_2bit) {
        gate_w_off = cfg.gate_w_off_2; gate_s_off = cfg.gate_s_off_2; gate_b_off = cfg.gate_b_off_2;
        up_w_off   = cfg.up_w_off_2;   up_s_off   = cfg.up_s_off_2;   up_b_off   = cfg.up_b_off_2;
        down_w_off = cfg.down_w_off_2; down_s_off = cfg.down_s_off_2; down_b_off = cfg.down_b_off_2;
    } else {
        gate_w_off = cfg.gate_w_off_4; gate_s_off = cfg.gate_s_off_4; gate_b_off = cfg.gate_b_off_4;
        up_w_off   = cfg.up_w_off_4;   up_s_off   = cfg.up_s_off_4;   up_b_off   = cfg.up_b_off_4;
        down_w_off = cfg.down_w_off_4;  down_s_off = cfg.down_s_off_4;  down_b_off = cfg.down_b_off_4;
    }
    id<MTLComputePipelineState> expert_pipe;
    if (g_use_2bit) {
        expert_pipe = (g_use_fp16_accum && ctx->matvec_2bit_fp16) ? ctx->matvec_2bit_fp16 : ctx->matvec_2bit;
    } else {
        expert_pipe = (g_use_fp16_accum && ctx->matvec_v3_fp16) ? ctx->matvec_v3_fp16 : ctx->matvec_v3;
    }

    // Copy expert weights into Metal buffer only if not already there
    if (!expert_data_already_in_buffer) {
        memcpy([ctx->buf_expert_data contents], expert_data, active_expert_size());
    }
    memcpy([ctx->buf_expert_input contents], h_post, cfg.hidden_dim * sizeof(float));

    uint32_t gate_up_out = cfg.moe_intermediate;  // 1024
    uint32_t gate_up_in  = cfg.hidden_dim;        // 4096
    uint32_t down_out    = cfg.hidden_dim;        // 4096
    uint32_t down_in     = cfg.moe_intermediate;  // 1024
    uint32_t gs          = cfg.group_size;        // 64

    // Build one command buffer with all 4 dispatches:
    // 1. gate_proj matvec (h_post -> gate_out)
    // 2. up_proj matvec (h_post -> up_out)
    // 3. SwiGLU (gate_out, up_out -> act_out)
    // 4. down_proj matvec (act_out -> expert_out)

    id<MTLCommandBuffer> cmdbuf = [ctx->queue commandBuffer];

    // --- Dispatch 1: gate_proj [4096] -> [1024] ---
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:ctx->buf_expert_data  offset:gate_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_expert_data  offset:gate_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_expert_data  offset:gate_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_expert_input offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_expert_gate  offset:0           atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }

    // --- Dispatch 2: up_proj [4096] -> [1024] ---
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:ctx->buf_expert_data  offset:up_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_expert_data  offset:up_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_expert_data  offset:up_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_expert_input offset:0          atIndex:3];
        [enc setBuffer:ctx->buf_expert_up    offset:0          atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }

    // --- Dispatch 3: SwiGLU(gate, up) -> act ---
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->swiglu];
        [enc setBuffer:ctx->buf_expert_gate offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_expert_up   offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_expert_act  offset:0 atIndex:2];
        [enc setBytes:&gate_up_out length:4 atIndex:3];
        uint32_t swiglu_tgs = (gate_up_out + 255) / 256;
        [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }

    // --- Dispatch 4: down_proj [1024] -> [4096] ---
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:ctx->buf_expert_data offset:down_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_expert_data offset:down_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_expert_data offset:down_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_expert_act  offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_expert_out  offset:0           atIndex:4];
        [enc setBytes:&down_out length:4 atIndex:5];
        [enc setBytes:&down_in  length:4 atIndex:6];
        [enc setBytes:&gs       length:4 atIndex:7];
        uint32_t num_tgs = (down_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }

    [cmdbuf commit];
    [cmdbuf waitUntilCompleted];

    // Copy result back to CPU
    memcpy(expert_out, [ctx->buf_expert_out contents], cfg.hidden_dim * sizeof(float));
}

// ============================================================================
// Fused expert kernel validation: run BOTH fused and separate paths on the
// same expert data, compare the SwiGLU activation output element-by-element.
// Enabled by --validate-fused flag (sets g_fused_expert_validate=1).
// Runs once for the first expert of the first layer, then auto-disables.
// ============================================================================
__attribute__((unused))
static void gpu_validate_fused_expert(
    MetalCtx *ctx,
    id<MTLBuffer> expert_buf  // expert weight data (already loaded)
) {
    if (!ctx->fused_gate_up) {
        fprintf(stderr, "[validate-fused] fused_gate_up pipeline not available, skipping.\n");
        return;
    }

    NSUInteger gate_w_off = cfg.gate_w_off_4, gate_s_off = cfg.gate_s_off_4, gate_b_off = cfg.gate_b_off_4;
    NSUInteger up_w_off   = cfg.up_w_off_4,   up_s_off   = cfg.up_s_off_4,   up_b_off   = cfg.up_b_off_4;

    uint32_t out_dim = cfg.moe_intermediate;
    uint32_t in_dim  = cfg.hidden_dim;
    uint32_t gs      = cfg.group_size;
    uint32_t num_tgs = (out_dim + 7) / 8;
    uint32_t swiglu_tgs = (out_dim + 255) / 256;

    // Allocate temporary buffer for the fused path output
    id<MTLBuffer> fused_out_buf = [ctx->device newBufferWithLength:out_dim * sizeof(float)
                                                           options:MTLResourceStorageModeShared];

    id<MTLComputePipelineState> mv_pipe = (g_use_fp16_accum && ctx->matvec_v3_fp16)
        ? ctx->matvec_v3_fp16 : ctx->matvec_v3;

    // --- Path A: separate gate + up + SwiGLU ---
    {
        id<MTLCommandBuffer> cmdbuf = [ctx->queue commandBuffer];
        // gate_proj -> buf_multi_expert_gate[0]
        {
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            [enc setComputePipelineState:mv_pipe];
            [enc setBuffer:expert_buf                      offset:gate_w_off  atIndex:0];
            [enc setBuffer:expert_buf                      offset:gate_s_off  atIndex:1];
            [enc setBuffer:expert_buf                      offset:gate_b_off  atIndex:2];
            [enc setBuffer:ctx->buf_multi_expert_input     offset:0           atIndex:3];
            [enc setBuffer:ctx->buf_multi_expert_gate[0]   offset:0           atIndex:4];
            [enc setBytes:&out_dim length:4 atIndex:5];
            [enc setBytes:&in_dim  length:4 atIndex:6];
            [enc setBytes:&gs      length:4 atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }
        // up_proj -> buf_multi_expert_up[0]
        {
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            [enc setComputePipelineState:mv_pipe];
            [enc setBuffer:expert_buf                      offset:up_w_off  atIndex:0];
            [enc setBuffer:expert_buf                      offset:up_s_off  atIndex:1];
            [enc setBuffer:expert_buf                      offset:up_b_off  atIndex:2];
            [enc setBuffer:ctx->buf_multi_expert_input     offset:0          atIndex:3];
            [enc setBuffer:ctx->buf_multi_expert_up[0]     offset:0          atIndex:4];
            [enc setBytes:&out_dim length:4 atIndex:5];
            [enc setBytes:&in_dim  length:4 atIndex:6];
            [enc setBytes:&gs      length:4 atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }
        // SwiGLU -> buf_multi_expert_act[0]
        {
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            [enc setComputePipelineState:ctx->swiglu];
            [enc setBuffer:ctx->buf_multi_expert_gate[0] offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_multi_expert_up[0]   offset:0 atIndex:1];
            [enc setBuffer:ctx->buf_multi_expert_act[0]  offset:0 atIndex:2];
            [enc setBytes:&out_dim length:4 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }
        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];
    }

    // Copy separate-path result
    float *separate_result = (float *)malloc(out_dim * sizeof(float));
    memcpy(separate_result, [ctx->buf_multi_expert_act[0] contents], out_dim * sizeof(float));

    // --- Path B: fused gate+up+SwiGLU ---
    {
        id<MTLCommandBuffer> cmdbuf = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        id<MTLComputePipelineState> fused_pipe = (g_use_fp16_accum && ctx->fused_gate_up_fp16)
            ? ctx->fused_gate_up_fp16 : ctx->fused_gate_up;
        [enc setComputePipelineState:fused_pipe];
        [enc setBuffer:expert_buf                      offset:gate_w_off  atIndex:0];
        [enc setBuffer:expert_buf                      offset:gate_s_off  atIndex:1];
        [enc setBuffer:expert_buf                      offset:gate_b_off  atIndex:2];
        [enc setBuffer:expert_buf                      offset:up_w_off    atIndex:3];
        [enc setBuffer:expert_buf                      offset:up_s_off    atIndex:4];
        [enc setBuffer:expert_buf                      offset:up_b_off    atIndex:5];
        [enc setBuffer:ctx->buf_multi_expert_input     offset:0           atIndex:6];
        [enc setBuffer:fused_out_buf                   offset:0           atIndex:7];
        [enc setBytes:&out_dim length:4 atIndex:8];
        [enc setBytes:&in_dim  length:4 atIndex:9];
        [enc setBytes:&gs      length:4 atIndex:10];
        [enc dispatchThreadgroups:MTLSizeMake(out_dim, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];
    }

    float *fused_result = (float *)[fused_out_buf contents];

    // --- Compare ---
    float max_abs_err = 0.0f, max_rel_err = 0.0f;
    int worst_idx = 0;
    int num_large = 0;
    for (uint32_t i = 0; i < out_dim; i++) {
        float diff = fabsf(fused_result[i] - separate_result[i]);
        float denom = fmaxf(fabsf(separate_result[i]), 1e-8f);
        float rel = diff / denom;
        if (diff > max_abs_err) { max_abs_err = diff; worst_idx = (int)i; }
        if (rel > max_rel_err) max_rel_err = rel;
        if (rel > 0.01f) num_large++;
    }
    fprintf(stderr, "[validate-fused] Comparing fused vs separate SwiGLU output (%u elements):\n", out_dim);
    fprintf(stderr, "  max_abs_err = %.6e  at index %d (fused=%.6f, separate=%.6f)\n",
            max_abs_err, worst_idx, fused_result[worst_idx], separate_result[worst_idx]);
    fprintf(stderr, "  max_rel_err = %.6e\n", max_rel_err);
    fprintf(stderr, "  elements with >1%% relative error: %d / %u\n", num_large, out_dim);
    if (max_rel_err < 0.001f) {
        fprintf(stderr, "  PASS: fused kernel matches separate path (max rel err < 0.1%%)\n");
    } else if (num_large == 0) {
        fprintf(stderr, "  PASS: minor numerical differences (all within 1%%)\n");
    } else {
        fprintf(stderr, "  FAIL: significant divergence detected (%d elements > 1%% error)\n", num_large);
    }

    free(separate_result);
    // fused_out_buf is ARC-managed, will be released automatically
    g_fused_expert_validate = 0;  // auto-disable after first run
}
