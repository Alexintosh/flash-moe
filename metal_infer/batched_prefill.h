// batched_prefill.h — Batched prefill: GEMV->GEMM conversion for N-token prefill
// Part of Flash-MoE modular decomposition (unity build).
// Included by infer.m — do NOT compile separately.
//
// Converts single-token GEMV operations to batched GEMM, amortizing weight reads
// across multiple tokens. Based on the Anemll fork's batched prefill approach.
//
// Strategy: layer-first — process all N tokens through one layer before moving
// to the next. Ping-pong between two [N * hidden_dim] GPU buffers.
//
// The key optimization: dequant_gemm_4bit_batch reads each weight row ONCE for
// all N tokens, giving ~Nx throughput improvement for projection-dominated layers.
//
// Expert handling modes:
//   g_prefill_skip_experts:      Skip ALL routed experts (shared only, fastest)
//   g_prefill_experts_full_only: Routed experts only at full attention layers
//   Default:                     Serial tail per token for routed experts after batched layer

// ============================================================================
// Helper: Encode a batched GEMM dispatch for dequant_gemm_4bit_batch
// ============================================================================
// Dispatches the batched 4-bit GEMM kernel. Reads weight row once for all N tokens.
//
// Parameters:
//   ctx:       Metal context
//   cmdbuf:    Command buffer to encode into
//   W/S/B:     Weight/scale/bias pointers (mmap'd, will be resolved via metal_find_chunk_sized)
//   in_buf:    Input buffer [N, in_dim] float
//   in_off:    Byte offset into in_buf
//   out_buf:   Output buffer [N, out_dim] float
//   out_off:   Byte offset into out_buf
//   out_dim:   Output dimension (rows in weight matrix)
//   in_dim:    Input dimension (cols in weight matrix)
//   group_size: Quantization group size
//   batch_n:   Number of tokens in batch
//
// Dispatch: (out_dim + 7) / 8 threadgroups of 256 threads
//   Each threadgroup handles 8 output rows (256/32 SIMD groups).

static void gpu_encode_pfb_gemm(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    const void *W, const void *S, const void *B,
    id<MTLBuffer> in_buf, NSUInteger in_off,
    id<MTLBuffer> out_buf, NSUInteger out_off,
    uint32_t out_dim, uint32_t in_dim, uint32_t group_size,
    uint32_t batch_n
) {
    id<MTLBuffer> w_buf, s_buf, b_buf;
    NSUInteger w_off, s_off, b_off;
    size_t w_size = (size_t)out_dim * in_dim / 8;  // 4-bit packed
    size_t num_groups = (in_dim + group_size - 1) / group_size;
    size_t sb_size = (size_t)out_dim * num_groups * sizeof(uint16_t);
    metal_find_chunk_sized(ctx, W, w_size, &w_buf, &w_off);
    metal_find_chunk_sized(ctx, S, sb_size, &s_buf, &s_off);
    metal_find_chunk_sized(ctx, B, sb_size, &b_buf, &b_off);

    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
    [enc setComputePipelineState:ctx->pfb_gemm_4bit];
    [enc setBuffer:w_buf   offset:w_off   atIndex:0];
    [enc setBuffer:s_buf   offset:s_off   atIndex:1];
    [enc setBuffer:b_buf   offset:b_off   atIndex:2];
    [enc setBuffer:in_buf  offset:in_off  atIndex:3];
    [enc setBuffer:out_buf offset:out_off atIndex:4];
    [enc setBytes:&out_dim    length:4 atIndex:5];
    [enc setBytes:&in_dim     length:4 atIndex:6];
    [enc setBytes:&group_size length:4 atIndex:7];
    [enc setBytes:&batch_n    length:4 atIndex:8];

    uint32_t num_tgs = (out_dim + 7) / 8;  // 8 rows per threadgroup (256 threads / 32 simd_size)
    [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
}

// ============================================================================
// Helper: CPU deinterleave conv1d output [N, conv_dim] into separate Q, K, V
// ============================================================================
// Conv1d outputs [N, conv_dim] where conv_dim = total_key*2 + total_value.
// Within each token: [Q(total_key), K(total_key), V(total_value)].
// Subsequent batched kernels (rms_norm_qk, delta_net_step) expect separate
// contiguous [N, total_key] and [N, total_value] arrays — they use total_key
// or total_value as the per-token stride, not conv_dim. Without this split,
// tokens t>=1 read from wrong offsets (stride mismatch: total_key vs conv_dim).

static void cpu_deinterleave_conv_batch(
    const float *conv_out,   // [N, conv_dim]
    float *q_out,            // [N, total_key]
    float *k_out,            // [N, total_key]
    float *v_out,            // [N, total_value]
    int batch_n, int total_key, int total_value, int conv_dim
) {
    for (int t = 0; t < batch_n; t++) {
        const float *src = conv_out + (size_t)t * conv_dim;
        memcpy(q_out + (size_t)t * total_key, src, total_key * sizeof(float));
        memcpy(k_out + (size_t)t * total_key, src + total_key, total_key * sizeof(float));
        memcpy(v_out + (size_t)t * total_value, src + 2 * total_key, total_value * sizeof(float));
    }
}

// ============================================================================
// Helper: CPU deinterleave Q projection output
// ============================================================================
// Q projection outputs [N, num_heads * head_dim * 2] with interleaved layout:
//   [head0_q(head_dim), head0_gate(head_dim), head1_q(head_dim), head1_gate(head_dim), ...]
// This splits it into separate Q [N, num_heads * head_dim] and gate [N, num_heads * head_dim].

static void cpu_deinterleave_q_batch(
    const float *q_proj,  // [N, num_heads * head_dim * 2]
    float *q_out,         // [N, num_heads * head_dim]
    float *gate_out,      // [N, num_heads * head_dim]
    int batch_n, int num_heads, int head_dim
) {
    int q_dim = num_heads * head_dim;
    int q_proj_dim = num_heads * head_dim * 2;
    for (int t = 0; t < batch_n; t++) {
        const float *src = q_proj + (size_t)t * q_proj_dim;
        float *q_dst = q_out + (size_t)t * q_dim;
        float *g_dst = gate_out + (size_t)t * q_dim;
        for (int h = 0; h < num_heads; h++) {
            memcpy(q_dst + h * head_dim, src + h * 2 * head_dim, head_dim * sizeof(float));
            memcpy(g_dst + h * head_dim, src + h * 2 * head_dim + head_dim, head_dim * sizeof(float));
        }
    }
}

// ============================================================================
// Batched prefill entry point
// ============================================================================
// Processes tokens [0, num_tokens-1] through all layers using batched GPU dispatch.
// The last prefill token is NOT processed here — caller handles it with
// fused_layer_forward + complete_deferred_experts for full hidden state output.
//
// Parameters:
//   wf:          Weight file (mmap'd non-expert weights)
//   embed_batch: [num_tokens, hidden_dim] pre-embedded tokens
//   num_tokens:  Total tokens to prefill (excluding the last one)
//   start_pos:   Starting RoPE position
//   K:           Number of active experts per token
//   layer_fds:   [num_layers] file descriptors for expert data
//   layer_mmaps: [num_layers] mmap'd expert data (or MAP_FAILED)
//   kv_caches:   [num_layers] KV caches (full attention layers)
//   layer_states: [num_layers] delta-net state (linear attention layers)
//
// Returns the number of tokens actually processed.

static int batched_prefill(
    WeightFile *wf,
    float *embed_batch,
    int num_tokens,
    int start_pos,
    int K,
    int *layer_fds,
    void **layer_mmaps,
    KVCache **kv_caches,
    void **layer_states __attribute__((unused))
) {
    MetalCtx *ctx = g_metal;
    if (!ctx || !ctx->pfb_gemm_4bit || g_prefill_batch <= 1) {
        return 0;  // fallback to token-by-token
    }

    int pfb = g_prefill_batch;
    if (pfb > MAX_PFB) pfb = MAX_PFB;
    int gpu_chunk = pfb < MAX_PFB_GPU ? pfb : MAX_PFB_GPU;

    if (!layer_cache_built) build_layer_cache(wf);

    printf("[batched_prefill] %d tokens, pfb=%d, gpu_chunk=%d, skip_experts=%d, full_only=%d\n",
           num_tokens, pfb, gpu_chunk, g_prefill_skip_experts, g_prefill_experts_full_only);

    double t_start = now_ms();

    // Allocate host-side ping-pong buffers for batched hidden states
    size_t batch_hidden_size = (size_t)num_tokens * cfg.hidden_dim * sizeof(float);
    float *hidden_A = (float *)malloc(batch_hidden_size);
    float *hidden_B = (float *)malloc(batch_hidden_size);
    if (!hidden_A || !hidden_B) {
        fprintf(stderr, "[batched_prefill] ERROR: Failed to allocate batch buffers (%.1f MB)\n",
                batch_hidden_size * 2 / 1e6);
        free(hidden_A); free(hidden_B);
        return 0;
    }

    // Initialize hidden_A with embedded tokens
    memcpy(hidden_A, embed_batch, (size_t)num_tokens * cfg.hidden_dim * sizeof(float));

    // Dimension shortcuts
    uint32_t hidden_dim = cfg.hidden_dim;
    uint32_t gs = cfg.group_size;
    uint32_t num_heads = cfg.num_attn_heads;
    uint32_t head_dim = cfg.head_dim;
    uint32_t num_kv_heads = cfg.num_kv_heads;
    uint32_t heads_per_kv = num_heads / num_kv_heads;
    uint32_t q_proj_dim = num_heads * head_dim * 2;   // interleaved Q + gate
    uint32_t q_dim = num_heads * head_dim;             // deinterleaved Q
    uint32_t kv_dim = num_kv_heads * head_dim;
    float rms_eps = cfg.rms_norm_eps;
    float rope_theta = cfg.rope_theta;
    uint32_t rotary_dim = cfg.rotary_dim;
    uint32_t shared_inter = cfg.shared_intermediate;

    // Linear attention dimensions
    uint32_t lin_conv_dim = cfg.linear_conv_dim;
    uint32_t lin_total_key = cfg.linear_total_key;
    uint32_t lin_total_value = cfg.linear_total_value;
    uint32_t lin_num_k_heads = cfg.linear_num_k_heads;
    uint32_t lin_num_v_heads = cfg.linear_num_v_heads;
    uint32_t lin_key_dim = cfg.linear_key_dim;
    uint32_t lin_value_dim = cfg.linear_value_dim;

    // Buffer assignments for batched pipeline:
    //   buf_pfb_input  = hidden states [N, hidden_dim] (input to each step)
    //   buf_pfb_out[0] = Q projection / QKV projection / attn output
    //   buf_pfb_out[1] = K projection / Z projection
    //   buf_pfb_out[2] = V projection / beta projection
    //   buf_pfb_out[3] = alpha projection / gate scores
    //   buf_pfb_out[4] = Q (deinterleaved) / conv output / shared gate
    //   buf_pfb_out[5] = gate (deinterleaved) / shared up
    //   buf_pfb_out[6] = normed hidden / shared act (SwiGLU output)
    //   buf_pfb_out[7] = shared down / combine scratch

    // Layer-first processing: all tokens through layer L before moving to L+1
    for (int layer = 0; layer < cfg.num_layers; layer++) {
        int is_full = cfg.is_full_attn[layer];
        LayerWeightCache *lc = &layer_cache[layer];

        // Determine expert handling for this layer
        int do_experts = 1;
        if (g_prefill_skip_experts) {
            do_experts = 0;
        } else if (g_prefill_experts_full_only && !is_full) {
            do_experts = 0;
        }

        // Process tokens in GPU-sized chunks (MAX_PFB_GPU = 32)
        for (int chunk_start = 0; chunk_start < num_tokens; chunk_start += gpu_chunk) {
            int chunk_n = num_tokens - chunk_start;
            if (chunk_n > gpu_chunk) chunk_n = gpu_chunk;
            uint32_t bn = (uint32_t)chunk_n;

            float *h_in  = hidden_A + (size_t)chunk_start * hidden_dim;
            float *h_out = hidden_B + (size_t)chunk_start * hidden_dim;
            int pos_base = start_pos + chunk_start;

            // ================================================================
            // Step 1: Upload hidden states + RMS norm
            // ================================================================
            // Copy chunk hidden states into GPU input buffer
            memcpy([ctx->buf_pfb_input contents], h_in,
                   (size_t)chunk_n * hidden_dim * sizeof(float));

            {
                id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
                metal_staging_reset(ctx);

                // RMS norm: buf_pfb_input -> buf_pfb_out[6] (normed)
                {
                    id<MTLBuffer> nw_buf; NSUInteger nw_off;
                    metal_find_chunk_sized(ctx, lc->input_norm_w,
                        (size_t)hidden_dim * 2, &nw_buf, &nw_off);  // bf16

                    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                    [enc setComputePipelineState:ctx->pfb_rms_norm];
                    [enc setBuffer:ctx->buf_pfb_input offset:0       atIndex:0]; // x [N, dim]
                    [enc setBuffer:nw_buf             offset:nw_off  atIndex:1]; // weight [dim] bf16
                    [enc setBuffer:ctx->buf_pfb_out[6] offset:0     atIndex:2]; // out [N, dim]
                    [enc setBytes:&hidden_dim length:4 atIndex:3];
                    [enc setBytes:&rms_eps   length:4 atIndex:4];
                    [enc setBytes:&bn        length:4 atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake(bn, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc endEncoding];
                }

                // ================================================================
                // Step 2: Attention projections (batched GEMM)
                // ================================================================
                if (is_full && lc->q_w && lc->k_w && lc->v_w) {
                    // Full attention: Q, K, V projections
                    // Q -> buf_pfb_out[0] [N, q_proj_dim] (interleaved Q+gate)
                    gpu_encode_pfb_gemm(ctx, cmd, lc->q_w, lc->q_s, lc->q_b,
                        ctx->buf_pfb_out[6], 0, ctx->buf_pfb_out[0], 0,
                        q_proj_dim, hidden_dim, gs, bn);
                    // K -> buf_pfb_out[1] [N, kv_dim]
                    gpu_encode_pfb_gemm(ctx, cmd, lc->k_w, lc->k_s, lc->k_b,
                        ctx->buf_pfb_out[6], 0, ctx->buf_pfb_out[1], 0,
                        kv_dim, hidden_dim, gs, bn);
                    // V -> buf_pfb_out[2] [N, kv_dim]
                    gpu_encode_pfb_gemm(ctx, cmd, lc->v_w, lc->v_s, lc->v_b,
                        ctx->buf_pfb_out[6], 0, ctx->buf_pfb_out[2], 0,
                        kv_dim, hidden_dim, gs, bn);
                } else if (!is_full && lc->qkv_w && lc->z_w && lc->b_w && lc->a_w) {
                    // Linear attention: QKV, Z, beta, alpha projections
                    // QKV -> buf_pfb_out[0] [N, conv_dim]
                    gpu_encode_pfb_gemm(ctx, cmd, lc->qkv_w, lc->qkv_s, lc->qkv_b,
                        ctx->buf_pfb_out[6], 0, ctx->buf_pfb_out[0], 0,
                        lin_conv_dim, hidden_dim, gs, bn);
                    // Z -> buf_pfb_out[1] [N, total_value]
                    gpu_encode_pfb_gemm(ctx, cmd, lc->z_w, lc->z_s, lc->z_b,
                        ctx->buf_pfb_out[6], 0, ctx->buf_pfb_out[1], 0,
                        lin_total_value, hidden_dim, gs, bn);
                    // beta -> buf_pfb_out[2] [N, num_v_heads]
                    gpu_encode_pfb_gemm(ctx, cmd, lc->b_w, lc->b_s, lc->b_b,
                        ctx->buf_pfb_out[6], 0, ctx->buf_pfb_out[2], 0,
                        lin_num_v_heads, hidden_dim, gs, bn);
                    // alpha -> buf_pfb_out[3] [N, num_v_heads]
                    gpu_encode_pfb_gemm(ctx, cmd, lc->a_w, lc->a_s, lc->a_b,
                        ctx->buf_pfb_out[6], 0, ctx->buf_pfb_out[3], 0,
                        lin_num_v_heads, hidden_dim, gs, bn);
                }

                [cmd commit];
                [cmd waitUntilCompleted];
            }

            // ================================================================
            // Step 3: Attention computation
            // ================================================================
            if (is_full && lc->q_w) {
                // ---- Full attention path ----
                // CPU deinterleave Q projection: buf_pfb_out[0] -> buf_pfb_out[4] (Q) + buf_pfb_out[5] (gate)
                {
                    float *q_proj_data = (float *)[ctx->buf_pfb_out[0] contents];
                    float *q_data = (float *)[ctx->buf_pfb_out[4] contents];
                    float *gate_data = (float *)[ctx->buf_pfb_out[5] contents];
                    cpu_deinterleave_q_batch(q_proj_data, q_data, gate_data,
                                             chunk_n, num_heads, head_dim);
                }

                // GPU: Q RoPE+norm, K RoPE+norm+cache write, causal attention
                {
                    id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
                    metal_staging_reset(ctx);

                    int fa_idx = cfg.full_attn_index[layer];
                    uint32_t cache_start = (uint32_t)pos_base;

                    // Q RoPE + norm: buf_pfb_out[4] in-place
                    {
                        id<MTLBuffer> qnw_buf; NSUInteger qnw_off;
                        metal_find_chunk_sized(ctx, lc->q_norm_w,
                            (size_t)head_dim * 2, &qnw_buf, &qnw_off);

                        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                        [enc setComputePipelineState:ctx->pfb_q_rope_norm];
                        [enc setBuffer:ctx->buf_pfb_out[4] offset:0       atIndex:0]; // Q [N, q_dim] in/out
                        [enc setBuffer:qnw_buf             offset:qnw_off atIndex:1]; // norm_w [head_dim] bf16
                        [enc setBytes:&head_dim    length:4 atIndex:2];
                        [enc setBytes:&num_heads   length:4 atIndex:3];
                        [enc setBytes:&bn          length:4 atIndex:4];
                        [enc setBytes:&cache_start length:4 atIndex:5];
                        [enc setBytes:&rope_theta  length:4 atIndex:6];
                        [enc setBytes:&rms_eps     length:4 atIndex:7];
                        [enc setBytes:&rotary_dim  length:4 atIndex:8];
                        uint32_t num_tgs = bn * num_heads;
                        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        [enc endEncoding];
                    }

                    // K RoPE + norm + KV cache write: buf_pfb_out[1] (K), buf_pfb_out[2] (V)
                    {
                        id<MTLBuffer> knw_buf; NSUInteger knw_off;
                        metal_find_chunk_sized(ctx, lc->k_norm_w,
                            (size_t)head_dim * 2, &knw_buf, &knw_off);

                        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                        [enc setComputePipelineState:ctx->pfb_kv_cache];
                        [enc setBuffer:ctx->buf_pfb_out[1]        offset:0       atIndex:0]; // K [N, kv_dim]
                        [enc setBuffer:ctx->buf_pfb_out[2]        offset:0       atIndex:1]; // V [N, kv_dim]
                        [enc setBuffer:ctx->buf_kv_k[fa_idx]      offset:0       atIndex:2]; // K cache
                        [enc setBuffer:ctx->buf_kv_v[fa_idx]      offset:0       atIndex:3]; // V cache
                        [enc setBuffer:knw_buf                    offset:knw_off atIndex:4]; // k_norm_w bf16
                        [enc setBytes:&head_dim     length:4 atIndex:5];
                        [enc setBytes:&kv_dim       length:4 atIndex:6];
                        [enc setBytes:&num_kv_heads length:4 atIndex:7];
                        [enc setBytes:&bn           length:4 atIndex:8];
                        [enc setBytes:&cache_start  length:4 atIndex:9];
                        [enc setBytes:&rope_theta   length:4 atIndex:10];
                        [enc setBytes:&rms_eps      length:4 atIndex:11];
                        [enc setBytes:&rotary_dim   length:4 atIndex:12];
                        uint32_t num_tgs = bn * num_kv_heads;
                        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        [enc endEncoding];
                    }

                    // Causal attention: Q (buf_pfb_out[4]), gate (buf_pfb_out[5]) -> buf_pfb_out[0]
                    {
                        float scale = 1.0f / sqrtf((float)head_dim);
                        scale *= rope_yarn_attn_factor(g_rope_scale_factor);

                        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                        [enc setComputePipelineState:ctx->pfb_causal_attn];
                        [enc setBuffer:ctx->buf_pfb_out[4]   offset:0 atIndex:0];  // Q [N, q_dim]
                        [enc setBuffer:ctx->buf_kv_k[fa_idx] offset:0 atIndex:1];  // K cache
                        [enc setBuffer:ctx->buf_kv_v[fa_idx] offset:0 atIndex:2];  // V cache
                        [enc setBuffer:ctx->buf_pfb_out[5]   offset:0 atIndex:3];  // gate [N, q_dim]
                        [enc setBuffer:ctx->buf_pfb_out[0]   offset:0 atIndex:4];  // output [N, q_dim]
                        [enc setBytes:&head_dim     length:4 atIndex:5];
                        [enc setBytes:&kv_dim       length:4 atIndex:6];
                        [enc setBytes:&num_heads    length:4 atIndex:7];
                        [enc setBytes:&heads_per_kv length:4 atIndex:8];
                        [enc setBytes:&cache_start  length:4 atIndex:9];
                        [enc setBytes:&bn           length:4 atIndex:10];
                        [enc setBytes:&scale        length:4 atIndex:11];
                        uint32_t num_tgs = bn * num_heads;
                        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        [enc endEncoding];
                    }

                    [cmd commit];
                    [cmd waitUntilCompleted];
                }

                // Also update host-side KV cache length tracking
                if (kv_caches[layer]) {
                    kv_caches[layer]->len += chunk_n;
                }

                // ================================================================
                // Step 4: O projection (batched GEMM)
                // ================================================================
                // attn output in buf_pfb_out[0] [N, q_dim] -> o_proj -> buf_pfb_out[4] [N, hidden_dim]
                {
                    id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
                    metal_staging_reset(ctx);

                    gpu_encode_pfb_gemm(ctx, cmd, lc->o_w, lc->o_s, lc->o_b,
                        ctx->buf_pfb_out[0], 0, ctx->buf_pfb_out[4], 0,
                        hidden_dim, q_dim, gs, bn);

                    [cmd commit];
                    [cmd waitUntilCompleted];
                }

            } else if (!is_full && lc->qkv_w) {
                // ---- Linear attention path ----
                int linear_layer_idx = cfg.linear_index[layer];

                // L1: conv1d_step_batched: buf_pfb_out[0] (qkv) -> buf_pfb_out[4] (conv output)
                // Must complete before CPU deinterleave, so separate command buffer.
                {
                    id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
                    metal_staging_reset(ctx);

                    id<MTLBuffer> cw_buf; NSUInteger cw_off;
                    metal_find_chunk_sized(ctx, lc->conv1d_w,
                        (size_t)lin_conv_dim * 4 * 2, &cw_buf, &cw_off);  // conv_dim * kernel_size * bf16

                    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                    [enc setComputePipelineState:ctx->pfb_conv1d_step];
                    [enc setBuffer:ctx->buf_conv_state[linear_layer_idx] offset:0 atIndex:0]; // conv state
                    [enc setBuffer:ctx->buf_pfb_out[0]  offset:0       atIndex:1]; // input [N, conv_dim]
                    [enc setBuffer:cw_buf               offset:cw_off  atIndex:2]; // weights bf16
                    [enc setBuffer:ctx->buf_pfb_out[4]  offset:0       atIndex:3]; // output [N, conv_dim]
                    [enc setBytes:&lin_conv_dim length:4 atIndex:4];
                    [enc setBytes:&bn           length:4 atIndex:5];
                    uint32_t tgs = (lin_conv_dim + 255) / 256;
                    [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc endEncoding];

                    [cmd commit];
                    [cmd waitUntilCompleted];
                }

                // CPU deinterleave: conv output [N, conv_dim] -> separate Q, K, V arrays.
                // The conv output packs [Q(total_key), K(total_key), V(total_value)] per token.
                // Batched kernels below expect contiguous [N, total_key] or [N, total_value]
                // arrays with stride = total_key/total_value, NOT conv_dim.
                // Without this split, token t>=1 reads from wrong offsets (stride mismatch).
                //
                // Buffer assignment after deinterleave:
                //   buf_pfb_out[0] = Q [N, total_key]   (overwrite, was QKV proj input)
                //   buf_pfb_out[5] = K [N, total_key]   (free slot)
                //   buf_pfb_out[6] = V [N, total_value]  (overwrite, normed input no longer needed)
                {
                    float *conv_data = (float *)[ctx->buf_pfb_out[4] contents];
                    float *q_data    = (float *)[ctx->buf_pfb_out[0] contents];
                    float *k_data    = (float *)[ctx->buf_pfb_out[5] contents];
                    float *v_data    = (float *)[ctx->buf_pfb_out[6] contents];
                    cpu_deinterleave_conv_batch(conv_data, q_data, k_data, v_data,
                        chunk_n, lin_total_key, lin_total_value, lin_conv_dim);
                }

                // GPU: rms_norm_qk -> compute_decay_beta -> delta_net_step -> gated_rms_norm
                // Now using separate contiguous Q/K/V buffers.
                {
                    id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
                    metal_staging_reset(ctx);

                    // L2: rms_norm_qk_batched: normalize Q and K in-place
                    //   Q in buf_pfb_out[0] [N, total_key], K in buf_pfb_out[5] [N, total_key]
                    {
                        float inv_scale = 1.0f / sqrtf((float)lin_key_dim);

                        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                        [enc setComputePipelineState:ctx->pfb_rms_norm_qk];
                        [enc setBuffer:ctx->buf_pfb_out[0] offset:0 atIndex:0]; // q [N, total_key]
                        [enc setBuffer:ctx->buf_pfb_out[5] offset:0 atIndex:1]; // k [N, total_key]
                        [enc setBytes:&lin_key_dim  length:4 atIndex:2];
                        [enc setBytes:&inv_scale    length:4 atIndex:3];
                        [enc setBytes:&lin_num_k_heads length:4 atIndex:4];
                        [enc setBytes:&bn           length:4 atIndex:5];
                        uint32_t num_tgs = bn * lin_num_k_heads;
                        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(lin_key_dim, 1, 1)];
                        [enc endEncoding];
                    }

                    // L3: compute_decay_beta_batched
                    // Output g_decay -> buf_pfb_out[3] (overwrite alpha, no longer needed)
                    // Output beta_gate -> buf_pfb_out[2] (overwrite beta, no longer needed)
                    // These buffers are [N, num_v_heads] sized from the GEMM output, large enough.
                    {
                        id<MTLBuffer> alog_buf, dtb_buf; NSUInteger alog_off, dtb_off;
                        metal_find_chunk_sized(ctx, lc->A_log,
                            (size_t)lin_num_v_heads * 4, &alog_buf, &alog_off);
                        metal_find_chunk_sized(ctx, lc->dt_bias,
                            (size_t)lin_num_v_heads * 2, &dtb_buf, &dtb_off);

                        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                        [enc setComputePipelineState:ctx->pfb_compute_decay_beta];
                        [enc setBuffer:ctx->buf_pfb_out[3]    offset:0         atIndex:0]; // alpha [N, num_v_heads]
                        [enc setBuffer:ctx->buf_pfb_out[2]    offset:0         atIndex:1]; // beta [N, num_v_heads]
                        [enc setBuffer:alog_buf               offset:alog_off  atIndex:2]; // A_log
                        [enc setBuffer:dtb_buf                offset:dtb_off   atIndex:3]; // dt_bias bf16
                        [enc setBuffer:ctx->buf_pfb_out[3]    offset:0         atIndex:4]; // g_decay output (in-place on alpha)
                        [enc setBuffer:ctx->buf_pfb_out[2]    offset:0         atIndex:5]; // beta_gate output (in-place on beta)
                        [enc setBytes:&lin_num_v_heads length:4 atIndex:6];
                        [enc setBytes:&bn              length:4 atIndex:7];
                        uint32_t total = bn * lin_num_v_heads;
                        uint32_t tgs = (total + 255) / 256;
                        [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        [enc endEncoding];
                    }

                    // L4: gated_delta_net_step_batched (processes tokens sequentially through recurrence)
                    // Q from buf_pfb_out[0], K from buf_pfb_out[5], V from buf_pfb_out[6]
                    // g_decay from buf_pfb_out[3], beta_gate from buf_pfb_out[2]
                    // output [N, total_value] -> buf_pfb_out[4] (conv output no longer needed)
                    {
                        uint32_t khpv = lin_num_v_heads / lin_num_k_heads;

                        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                        [enc setComputePipelineState:ctx->pfb_delta_net_step];
                        [enc setBuffer:ctx->buf_delta_state[linear_layer_idx] offset:0 atIndex:0]; // state
                        [enc setBuffer:ctx->buf_pfb_out[0] offset:0 atIndex:1]; // q [N, total_key]
                        [enc setBuffer:ctx->buf_pfb_out[5] offset:0 atIndex:2]; // k [N, total_key]
                        [enc setBuffer:ctx->buf_pfb_out[6] offset:0 atIndex:3]; // v [N, total_value]
                        [enc setBuffer:ctx->buf_pfb_out[3] offset:0 atIndex:4]; // g_decay [N, num_v_heads]
                        [enc setBuffer:ctx->buf_pfb_out[2] offset:0 atIndex:5]; // beta_gate [N, num_v_heads]
                        [enc setBuffer:ctx->buf_pfb_out[4] offset:0 atIndex:6]; // output [N, total_value]
                        [enc setBytes:&khpv            length:4 atIndex:7];
                        [enc setBytes:&lin_key_dim     length:4 atIndex:8];
                        [enc setBytes:&lin_value_dim   length:4 atIndex:9];
                        [enc setBytes:&lin_total_key   length:4 atIndex:10];
                        [enc setBytes:&lin_total_value length:4 atIndex:11];
                        [enc setBytes:&lin_num_v_heads length:4 atIndex:12];
                        [enc setBytes:&bn              length:4 atIndex:13];
                        [enc dispatchThreadgroups:MTLSizeMake(lin_num_v_heads, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(lin_value_dim, 1, 1)];
                        [enc endEncoding];
                    }

                    // L5: gated_rms_norm_batched: delta output + z -> buf_pfb_out[0]
                    // delta output in buf_pfb_out[4], z in buf_pfb_out[1]
                    {
                        id<MTLBuffer> gnw_buf; NSUInteger gnw_off;
                        metal_find_chunk_sized(ctx, lc->gated_norm_w,
                            (size_t)lin_total_value * 2, &gnw_buf, &gnw_off);

                        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                        [enc setComputePipelineState:ctx->pfb_gated_rms_norm];
                        [enc setBuffer:ctx->buf_pfb_out[4]   offset:0       atIndex:0]; // values [N, total_value]
                        [enc setBuffer:ctx->buf_pfb_out[1]   offset:0       atIndex:1]; // z [N, total_value]
                        [enc setBuffer:gnw_buf               offset:gnw_off atIndex:2]; // weight bf16
                        [enc setBuffer:ctx->buf_pfb_out[0]   offset:0       atIndex:3]; // output [N, total_value]
                        [enc setBytes:&lin_value_dim   length:4 atIndex:4];
                        [enc setBytes:&rms_eps         length:4 atIndex:5];
                        [enc setBytes:&lin_num_v_heads length:4 atIndex:6];
                        [enc setBytes:&bn              length:4 atIndex:7];
                        uint32_t num_tgs = bn * lin_num_v_heads;
                        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(lin_value_dim, 1, 1)];
                        [enc endEncoding];
                    }

                    [cmd commit];
                    [cmd waitUntilCompleted];
                }

                // ================================================================
                // Step 4: O projection (linear attention: out_proj)
                // ================================================================
                // gated_rms_norm output in buf_pfb_out[0] [N, total_value] -> out_proj -> buf_pfb_out[4] [N, hidden_dim]
                {
                    id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
                    metal_staging_reset(ctx);

                    gpu_encode_pfb_gemm(ctx, cmd, lc->out_proj_w, lc->out_proj_s, lc->out_proj_b,
                        ctx->buf_pfb_out[0], 0, ctx->buf_pfb_out[4], 0,
                        hidden_dim, lin_total_value, gs, bn);

                    [cmd commit];
                    [cmd waitUntilCompleted];
                }
            }

            // ================================================================
            // Step 5: Residual + post-attention norm
            // ================================================================
            // residual (buf_pfb_input) + o_proj (buf_pfb_out[4]) -> normed (buf_pfb_out[6])
            // Also stores updated hidden state for next steps
            {
                id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
                metal_staging_reset(ctx);

                id<MTLBuffer> panw_buf; NSUInteger panw_off;
                metal_find_chunk_sized(ctx, lc->post_attn_norm_w,
                    (size_t)hidden_dim * 2, &panw_buf, &panw_off);

                // residual_norm: residual + x -> out (normed), hidden (updated h_mid)
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:ctx->pfb_residual_norm];
                [enc setBuffer:ctx->buf_pfb_input  offset:0       atIndex:0]; // residual [N, dim]
                [enc setBuffer:ctx->buf_pfb_out[4] offset:0       atIndex:1]; // x (o_proj) [N, dim]
                [enc setBuffer:panw_buf            offset:panw_off atIndex:2]; // weight [dim] bf16
                [enc setBuffer:ctx->buf_pfb_out[6] offset:0       atIndex:3]; // out (normed) [N, dim]
                [enc setBuffer:ctx->buf_pfb_out[7] offset:0       atIndex:4]; // hidden (h_mid) [N, dim]
                [enc setBytes:&hidden_dim length:4 atIndex:5];
                [enc setBytes:&rms_eps   length:4 atIndex:6];
                [enc setBytes:&bn        length:4 atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(bn, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];

                [cmd commit];
                [cmd waitUntilCompleted];
            }

            // buf_pfb_out[6] now has h_post (post-attn normed), buf_pfb_out[7] has h_mid

            // ================================================================
            // Step 6: MoE routing (CPU: softmax + topK per token)
            // ================================================================
            // First, batched GEMM for routing gate + shared expert projections
            {
                id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
                metal_staging_reset(ctx);

                // gate scores: buf_pfb_out[6] (h_post) -> buf_pfb_out[0] [N, num_experts]
                if (lc->gate_w && lc->gate_s && lc->gate_b) {
                    gpu_encode_pfb_gemm(ctx, cmd, lc->gate_w, lc->gate_s, lc->gate_b,
                        ctx->buf_pfb_out[6], 0, ctx->buf_pfb_out[0], 0,
                        (uint32_t)cfg.num_experts, hidden_dim, gs, bn);
                }
                // shared gate_proj: buf_pfb_out[6] -> buf_pfb_out[4] [N, shared_intermediate]
                if (lc->sg_w && lc->sg_s && lc->sg_b) {
                    gpu_encode_pfb_gemm(ctx, cmd, lc->sg_w, lc->sg_s, lc->sg_b,
                        ctx->buf_pfb_out[6], 0, ctx->buf_pfb_out[4], 0,
                        shared_inter, hidden_dim, gs, bn);
                }
                // shared up_proj: buf_pfb_out[6] -> buf_pfb_out[5] [N, shared_intermediate]
                if (lc->su_w && lc->su_s && lc->su_b) {
                    gpu_encode_pfb_gemm(ctx, cmd, lc->su_w, lc->su_s, lc->su_b,
                        ctx->buf_pfb_out[6], 0, ctx->buf_pfb_out[5], 0,
                        shared_inter, hidden_dim, gs, bn);
                }
                // shared_expert_gate: buf_pfb_out[6] -> buf_pfb_out[3] [N, 1]
                if (lc->seg_w && lc->seg_s && lc->seg_b) {
                    gpu_encode_pfb_gemm(ctx, cmd, lc->seg_w, lc->seg_s, lc->seg_b,
                        ctx->buf_pfb_out[6], 0, ctx->buf_pfb_out[3], 0,
                        1, hidden_dim, gs, bn);
                }

                [cmd commit];
                [cmd waitUntilCompleted];
            }

            // ================================================================
            // Step 7: Shared expert SwiGLU + down_proj (batched)
            // ================================================================
            // SwiGLU: buf_pfb_out[4] (gate) + buf_pfb_out[5] (up) -> buf_pfb_out[4] (act)
            {
                id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];

                uint32_t total_swiglu = bn * shared_inter;
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:ctx->pfb_swiglu];
                [enc setBuffer:ctx->buf_pfb_out[4] offset:0 atIndex:0]; // gate [N * shared_inter]
                [enc setBuffer:ctx->buf_pfb_out[5] offset:0 atIndex:1]; // up [N * shared_inter]
                [enc setBuffer:ctx->buf_pfb_out[4] offset:0 atIndex:2]; // out [N * shared_inter] (in-place on gate)
                [enc setBytes:&total_swiglu length:4 atIndex:3];
                uint32_t tgs = (total_swiglu + 255) / 256;
                [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];

                // shared down_proj: buf_pfb_out[4] (act) [N, shared_inter] -> buf_pfb_out[5] [N, hidden_dim]
                metal_staging_reset(ctx);
                if (lc->sd_w && lc->sd_s && lc->sd_b) {
                    gpu_encode_pfb_gemm(ctx, cmd, lc->sd_w, lc->sd_s, lc->sd_b,
                        ctx->buf_pfb_out[4], 0, ctx->buf_pfb_out[5], 0,
                        hidden_dim, shared_inter, gs, bn);
                }

                [cmd commit];
                [cmd waitUntilCompleted];
            }

            // ================================================================
            // Step 8: Routed experts (serial per-token tail)
            // ================================================================
            // For each token: CPU softmax+topK on gate scores, then pread+GPU expert forward
            // This is the I/O-bound tail — cannot be batched across tokens because each
            // token routes to different experts.

            // Read back routing gate scores from GPU
            float *gate_scores_batch = (float *)[ctx->buf_pfb_out[0] contents];
            float *shared_gate_scores = (float *)[ctx->buf_pfb_out[3] contents];
            float *h_mid_batch = (float *)[ctx->buf_pfb_out[7] contents];
            float *shared_down_batch = (float *)[ctx->buf_pfb_out[5] contents];

            for (int t = 0; t < chunk_n; t++) {
                float *gate_t = gate_scores_batch + (size_t)t * cfg.num_experts;
                float *h_mid_t = h_mid_batch + (size_t)t * hidden_dim;
                float *shared_down_t = shared_down_batch + (size_t)t * hidden_dim;
                float shared_gate_t = shared_gate_scores[t];

                // ================================================================
                // Step 9: Combine for this token
                // ================================================================
                // hidden = h_mid + sigmoid(shared_gate) * shared_down + moe_out
                float shared_weight = 1.0f / (1.0f + expf(-shared_gate_t));

                if (do_experts) {
                    // CPU softmax + topK
                    float gate_local[cfg.num_experts];
                    memcpy(gate_local, gate_t, cfg.num_experts * sizeof(float));
                    cpu_softmax(gate_local, cfg.num_experts);

                    int expert_indices[64];
                    float expert_weights[64];
                    cpu_topk(gate_local, cfg.num_experts, K, expert_indices, expert_weights);
                    cpu_normalize_weights(expert_weights, K);

                    // Serial routed expert forward (pread from SSD + GPU)
                    void *mmap_base = layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL;
                    int packed_fd = layer_fds[layer];
                    float moe_out[hidden_dim];
                    memset(moe_out, 0, hidden_dim * sizeof(float));

                    for (int k = 0; k < K; k++) {
                        off_t eoff; size_t esz;
                        expert_offset_size(layer, expert_indices[k], &eoff, &esz);

                        // Load expert data into GPU buffer
                        void *dst = [ctx->buf_multi_expert_data[0] contents];
                        ssize_t rd;
                        if (mmap_base) {
                            memcpy(dst, (char *)mmap_base + eoff, esz);
                            rd = (ssize_t)esz;
                        } else {
                            rd = pread(packed_fd, dst, esz, eoff);
                        }
                        if (rd != (ssize_t)esz) continue;

                        // GPU expert forward: gate+up+SwiGLU+down
                        id<MTLCommandBuffer> ecmd = [ctx->queue commandBuffer];
                        // Copy h_post for this token into expert input buffer
                        float *h_post_t = (float *)[ctx->buf_pfb_out[6] contents] + (size_t)t * hidden_dim;
                        memcpy([ctx->buf_multi_expert_input contents], h_post_t, hidden_dim * sizeof(float));

                        gpu_encode_expert_forward_slot(ctx, ecmd, 0);
                        [ecmd commit];
                        [ecmd waitUntilCompleted];

                        // Accumulate weighted expert output
                        float *expert_result = (float *)[ctx->buf_multi_expert_out[0] contents];
                        for (int i = 0; i < (int)hidden_dim; i++) {
                            moe_out[i] += expert_weights[k] * expert_result[i];
                        }
                    }

                    // Combine: hidden = h_mid + shared + moe
                    for (int i = 0; i < (int)hidden_dim; i++) {
                        h_out[(size_t)t * hidden_dim + i] = h_mid_t[i]
                            + shared_weight * shared_down_t[i]
                            + moe_out[i];
                    }
                } else {
                    // Skip routed experts: hidden = h_mid + sigmoid(gate) * shared_down
                    for (int i = 0; i < (int)hidden_dim; i++) {
                        h_out[(size_t)t * hidden_dim + i] = h_mid_t[i]
                            + shared_weight * shared_down_t[i];
                    }
                }
            }
        }

        // Swap ping-pong buffers
        float *tmp = hidden_A;
        hidden_A = hidden_B;
        hidden_B = tmp;
    }

    free(hidden_A);
    free(hidden_B);

    double elapsed = now_ms() - t_start;
    printf("[batched_prefill] %d tokens in %.0f ms (%.1f ms/tok, %.1f tok/s)\n",
           num_tokens, elapsed, elapsed / num_tokens,
           num_tokens * 1000.0 / elapsed);

    return num_tokens;
}
