// batched_prefill.h — Layer-first batched prefill with batched GEMM dispatch
// Part of Flash-MoE modular decomposition (unity build).
// Included by infer.m — do NOT compile separately.
//
// Strategy: LAYER-FIRST ordering. Full attention layers use batched GEMM
// (dequant_gemm_4bit_batch) for Q/K/V projections, reading each weight row
// ONCE for all N tokens. Linear attention layers use per-token fused_layer_forward.
//
// The batched GEMM kernel reads each weight row ONCE and multiplies against
// N input vectors simultaneously. For a 4096x16384 Q projection (~32MB weights),
// processing 32 tokens reads 32MB once instead of 32x = 1GB.
//
// Incremental approach:
//   Step 1: validate_gemm_output() — compares batched GEMM vs per-token reference
//   Step 2+3: Batched GEMM for Q/K/V projections (full attn layers)
//   Step 4: Batched GEMM for shared expert projections (future)
//
// Linear attention layers still use per-token fused_layer_forward to avoid
// the conv1d state issues that caused previous failures.

// ============================================================================
// Step 1: GEMM validation — compare batched output vs per-token reference
// ============================================================================
// Runs the same projection per-token using cpu_dequant_matvec and compares
// against the batched GEMM output. Prints max absolute error.
// Returns max error. If > 0.01, prints WARNING.

static float validate_gemm_output(
    const char *label,
    const float *gemm_out,   // [N, out_dim] from batched GEMM
    const float *input,      // [N, in_dim] input vectors (normed hidden states)
    int N,
    const uint32_t *W, const uint16_t *scales, const uint16_t *biases,
    int out_dim, int in_dim, int group_size
) {
    float max_err = 0.0f;
    float *ref = (float *)malloc(out_dim * sizeof(float));
    if (!ref) {
        fprintf(stderr, "[validate_gemm] ERROR: alloc failed for %s\n", label);
        return -1.0f;
    }

    for (int t = 0; t < N; t++) {
        const float *x_t = input + (size_t)t * in_dim;
        const float *y_t = gemm_out + (size_t)t * out_dim;

        // Reference: per-token CPU dequant matvec
        cpu_dequant_matvec(W, scales, biases, x_t, ref, out_dim, in_dim, group_size);

        for (int i = 0; i < out_dim; i++) {
            float err = fabsf(y_t[i] - ref[i]);
            if (err > max_err) max_err = err;
        }
    }

    free(ref);

    if (max_err > 0.01f) {
        fprintf(stderr, "[validate_gemm] WARNING: %s max_err=%.6f (N=%d, out=%d, in=%d) — MISMATCH!\n",
                label, max_err, N, out_dim, in_dim);
    } else {
        fprintf(stderr, "[validate_gemm] %s max_err=%.6f (N=%d) — OK\n", label, max_err, N);
    }
    return max_err;
}

// ============================================================================
// Batched GEMM dispatch helper
// ============================================================================
// Encodes the dequant_gemm_4bit_batch kernel for one projection across N tokens.
// Input: in_buf [N, in_dim]
// Output: ctx->buf_pfb_out[out_slot] [N, out_dim]
// Weight data comes from the mmap'd weight file (zero-copy via Metal buffers).
// NOTE: caller must have called metal_staging_reset before the first call in a
// command buffer. Subsequent calls accumulate into the same staging buffer.

static void encode_batched_gemm(
    MetalCtx *ctx,
    id<MTLComputeCommandEncoder> enc,
    const uint32_t *W, const uint16_t *scales, const uint16_t *biases,
    id<MTLBuffer> in_buf, NSUInteger in_off,
    int out_dim, int in_dim, int group_size,
    int N,                   // number of tokens in batch
    int out_slot             // which buf_pfb_out[slot] to write to
) {
    id<MTLBuffer> w_buf, s_buf, b_buf;
    NSUInteger w_off, s_off, b_off;
    size_t w_size = (size_t)out_dim * in_dim / 8;  // 4-bit packed
    size_t num_groups = (in_dim + group_size - 1) / group_size;
    size_t sb_size = (size_t)out_dim * num_groups * sizeof(uint16_t);
    metal_find_chunk_sized(ctx, W, w_size, &w_buf, &w_off);
    metal_find_chunk_sized(ctx, scales, sb_size, &s_buf, &s_off);
    metal_find_chunk_sized(ctx, biases, sb_size, &b_buf, &b_off);

    uint32_t u_out_dim = (uint32_t)out_dim;
    uint32_t u_in_dim = (uint32_t)in_dim;
    uint32_t u_group_size = (uint32_t)group_size;
    uint32_t u_batch_n = (uint32_t)N;

    [enc setComputePipelineState:ctx->pfb_gemm_4bit];
    [enc setBuffer:w_buf   offset:w_off  atIndex:0];
    [enc setBuffer:s_buf   offset:s_off  atIndex:1];
    [enc setBuffer:b_buf   offset:b_off  atIndex:2];
    [enc setBuffer:in_buf  offset:in_off atIndex:3];
    [enc setBuffer:ctx->buf_pfb_out[out_slot] offset:0 atIndex:4];
    [enc setBytes:&u_out_dim    length:4 atIndex:5];
    [enc setBytes:&u_in_dim     length:4 atIndex:6];
    [enc setBytes:&u_group_size length:4 atIndex:7];
    [enc setBytes:&u_batch_n    length:4 atIndex:8];

    // 256 threads per threadgroup, 8 SIMD groups = 8 rows per threadgroup
    uint32_t num_tgs = (u_out_dim + 7) / 8;
    [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// ============================================================================
// Batched RMS norm dispatch helper
// ============================================================================
// Encodes prefill_rms_norm_bf16 for N tokens.

static void encode_batched_rms_norm(
    MetalCtx *ctx,
    id<MTLComputeCommandEncoder> enc,
    id<MTLBuffer> src_buf, NSUInteger src_offset,
    id<MTLBuffer> dst_buf, NSUInteger dst_offset,
    const uint16_t *norm_weight,
    int dim, float eps, int N
) {
    id<MTLBuffer> nw_buf; NSUInteger nw_off;
    size_t nw_size = (size_t)dim * sizeof(uint16_t);
    metal_find_chunk_sized(ctx, norm_weight, nw_size, &nw_buf, &nw_off);

    uint32_t u_dim = (uint32_t)dim;
    uint32_t u_batch_n = (uint32_t)N;

    [enc setComputePipelineState:ctx->pfb_rms_norm];
    [enc setBuffer:src_buf offset:src_offset atIndex:0];
    [enc setBuffer:nw_buf  offset:nw_off     atIndex:1];
    [enc setBuffer:dst_buf offset:dst_offset atIndex:2];
    [enc setBytes:&u_dim     length:4             atIndex:3];
    [enc setBytes:&eps       length:sizeof(float) atIndex:4];
    [enc setBytes:&u_batch_n length:4             atIndex:5];

    // One threadgroup per token, 256 threads per threadgroup
    [enc dispatchThreadgroups:MTLSizeMake(u_batch_n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// ============================================================================
// Batched prefill entry point (v2 — layer-first, batched GEMM for full attn)
// ============================================================================
// Processes tokens [0, num_tokens-1] through all layers in layer-first order.
// Full attention layers use batched GEMM for Q/K/V projections.
// Linear attention layers use per-token fused_layer_forward (safe path).
//
// The last prefill token is NOT processed here — caller handles it with
// fused_layer_forward + complete_deferred_experts for full hidden state output.

static int batched_prefill(
    WeightFile *wf,
    float *embed_batch,
    int num_tokens,
    int start_pos,
    int K,
    int *layer_fds,
    void **layer_mmaps,
    KVCache **kv_caches,
    void **layer_states
) {
    if (num_tokens <= 0) return 0;
    if (g_prefill_batch <= 1) return 0;  // batched prefill disabled

    double t_start = now_ms();

    // Ensure layer cache and scratch buffers are built
    if (!layer_cache_built) build_layer_cache(wf);
    if (init_layer_scratch() != 0) {
        fprintf(stderr, "[batched_prefill] ERROR: scratch buffer allocation failed\n");
        return 0;
    }

    // Check if batched GEMM is available for full attention layers
    int have_gemm = (g_metal && g_metal->pfb_gemm_4bit && g_metal->pfb_rms_norm &&
                     g_metal->buf_pfb_input && g_metal->buf_pfb_out[0] &&
                     (g_metal->wf_num_chunks > 0 || g_metal->wf_staging));

    // Dimensions
    int q_proj_dim = cfg.num_attn_heads * cfg.head_dim * 2;
    int kv_dim = cfg.num_kv_heads * cfg.head_dim;
    int hidden_dim = cfg.hidden_dim;
    int gs = cfg.group_size;

    // Allocate per-token hidden state buffers [num_tokens, hidden_dim]
    size_t hbuf_size = (size_t)num_tokens * hidden_dim * sizeof(float);
    float *hidden_states = (float *)malloc(hbuf_size);
    if (!hidden_states) {
        fprintf(stderr, "[batched_prefill] ERROR: Failed to allocate hidden buffer (%.1f MB)\n",
                hbuf_size / 1e6);
        return 0;
    }
    memcpy(hidden_states, embed_batch, hbuf_size);

    // Per-token working buffer for fused_layer_forward (it works in-place)
    float *token_hidden = (float *)malloc(hidden_dim * sizeof(float));
    if (!token_hidden) {
        fprintf(stderr, "[batched_prefill] ERROR: Failed to allocate token hidden buffer\n");
        free(hidden_states);
        return 0;
    }

    // Validation: run on the first full attention layer of the first call only
    static int gemm_validated = 0;

    int gemm_layers = 0, fallback_layers = 0;

    // Layer-first: for each layer, process all tokens through it
    for (int layer = 0; layer < cfg.num_layers; layer++) {
        int is_full = cfg.is_full_attn[layer];
        const void *mmap_base = (layer_mmaps[layer] != MAP_FAILED) ? layer_mmaps[layer] : NULL;
        LayerWeightCache *lc = &layer_cache[layer];

        // ================================================================
        // FULL ATTENTION LAYERS: batched GEMM for Q/K/V projections
        // ================================================================
        if (is_full && have_gemm && num_tokens <= MAX_PFB_GPU &&
            lc->q_w && lc->q_s && lc->q_b &&
            lc->k_w && lc->k_s && lc->k_b &&
            lc->v_w && lc->v_s && lc->v_b &&
            lc->input_norm_w) {

            gemm_layers++;
            MetalCtx *ctx = g_metal;

            // Upload hidden states [N, hidden_dim] into GPU input buffer
            memcpy([ctx->buf_pfb_input contents], hidden_states,
                   (size_t)num_tokens * hidden_dim * sizeof(float));

            // ---- GPU: RMS norm + Q/K/V GEMM in one command buffer ----
            metal_staging_reset(ctx);
            id<MTLCommandBuffer> cmdbuf = [ctx->queue commandBuffer];

            // RMS norm: buf_pfb_input (raw hidden) -> buf_pfb_out[7] (normed)
            {
                id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
                encode_batched_rms_norm(ctx, enc,
                    ctx->buf_pfb_input, 0,
                    ctx->buf_pfb_out[7], 0,
                    lc->input_norm_w,
                    hidden_dim, cfg.rms_norm_eps, num_tokens);
                [enc endEncoding];
            }

            // Q projection: buf_pfb_out[7] (normed) -> buf_pfb_out[0] [N, q_proj_dim]
            {
                id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
                encode_batched_gemm(ctx, enc,
                    lc->q_w, lc->q_s, lc->q_b,
                    ctx->buf_pfb_out[7], 0,
                    q_proj_dim, hidden_dim, gs,
                    num_tokens, 0);
                [enc endEncoding];
            }

            // K projection: buf_pfb_out[7] (normed) -> buf_pfb_out[1] [N, kv_dim]
            {
                id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
                encode_batched_gemm(ctx, enc,
                    lc->k_w, lc->k_s, lc->k_b,
                    ctx->buf_pfb_out[7], 0,
                    kv_dim, hidden_dim, gs,
                    num_tokens, 1);
                [enc endEncoding];
            }

            // V projection: buf_pfb_out[7] (normed) -> buf_pfb_out[2] [N, kv_dim]
            {
                id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
                encode_batched_gemm(ctx, enc,
                    lc->v_w, lc->v_s, lc->v_b,
                    ctx->buf_pfb_out[7], 0,
                    kv_dim, hidden_dim, gs,
                    num_tokens, 2);
                [enc endEncoding];
            }

            [cmdbuf commit];
            [cmdbuf waitUntilCompleted];

            // Read back GPU results
            float *q_batch = (float *)[ctx->buf_pfb_out[0] contents];  // [N, q_proj_dim]
            float *k_batch = (float *)[ctx->buf_pfb_out[1] contents];  // [N, kv_dim]
            float *v_batch = (float *)[ctx->buf_pfb_out[2] contents];  // [N, kv_dim]
            float *normed_batch = (float *)[ctx->buf_pfb_out[7] contents]; // [N, hidden_dim]

            // ---- Validate GEMM on first full attention layer (once) ----
            if (!gemm_validated) {
                validate_gemm_output("Q_proj", q_batch, normed_batch, num_tokens,
                    lc->q_w, lc->q_s, lc->q_b, q_proj_dim, hidden_dim, gs);
                validate_gemm_output("K_proj", k_batch, normed_batch, num_tokens,
                    lc->k_w, lc->k_s, lc->k_b, kv_dim, hidden_dim, gs);
                validate_gemm_output("V_proj", v_batch, normed_batch, num_tokens,
                    lc->v_w, lc->v_s, lc->v_b, kv_dim, hidden_dim, gs);
                gemm_validated = 1;
            }

            // ---- Per-token: use pre-computed Q/K/V via fused_layer_forward ----
            for (int t = 0; t < num_tokens; t++) {
                int pos = start_pos + t;
                float *h = hidden_states + (size_t)t * hidden_dim;

                // Copy hidden state into the per-token working buffer
                memcpy(token_hidden, h, hidden_dim * sizeof(float));

                // Pre-fill scratch buffers with batched GEMM results for this token
                memcpy(s_q_proj_out, q_batch + (size_t)t * q_proj_dim,
                       q_proj_dim * sizeof(float));
                memcpy(s_k_proj_out, k_batch + (size_t)t * kv_dim,
                       kv_dim * sizeof(float));
                memcpy(s_v_proj_out, v_batch + (size_t)t * kv_dim,
                       kv_dim * sizeof(float));

                // Pre-fill normed and residual for the bypass path
                memcpy(s_normed, normed_batch + (size_t)t * hidden_dim,
                       hidden_dim * sizeof(float));
                cpu_vec_copy(s_residual, token_hidden, hidden_dim);

                // Set bypass flag: fused_layer_forward will skip norm + Q/K/V dispatch
                g_pfb_precomputed_qkv = 1;

                fused_layer_forward(wf, layer, token_hidden,
                                    kv_caches[layer], NULL,
                                    pos, mmap_base,
                                    K, layer_fds[layer]);

                g_pfb_precomputed_qkv = 0;

                // Complete deferred expert work
                complete_deferred_experts();

                // Copy result back
                memcpy(h, token_hidden, hidden_dim * sizeof(float));
            }

        } else {
            // ================================================================
            // FALLBACK: per-token fused_layer_forward (linear attn or no GEMM)
            // ================================================================
            fallback_layers++;

            for (int t = 0; t < num_tokens; t++) {
                int pos = start_pos + t;
                float *h = hidden_states + (size_t)t * hidden_dim;

                memcpy(token_hidden, h, hidden_dim * sizeof(float));

                fused_layer_forward(wf, layer, token_hidden,
                                    is_full ? kv_caches[layer] : NULL,
                                    is_full ? NULL : (LinearAttnState *)layer_states[layer],
                                    pos, mmap_base,
                                    K, layer_fds[layer]);
                complete_deferred_experts();

                memcpy(h, token_hidden, hidden_dim * sizeof(float));
            }
        }
    }

    free(token_hidden);
    free(hidden_states);

    double elapsed = now_ms() - t_start;
    fprintf(stderr, "[batched_prefill] %d tokens, layer-first, %.1f ms (GEMM: %d layers, fallback: %d layers)\n",
            num_tokens, elapsed, gemm_layers, fallback_layers);

    return num_tokens;
}
