// batched_prefill.h — Layer-first batched prefill with GEMM optimization (v2)
// Part of Flash-MoE modular decomposition (unity build).
// Included by infer.m — do NOT compile separately.
//
// Strategy: LAYER-FIRST ordering with BATCHED GEMM for projection matmuls.
//
// For each layer, all N tokens' hidden states are batched into a single GEMM
// dispatch (dequant_gemm_4bit_batch) for the projection weights. This amortizes
// weight reads across the batch — each weight row is read ONCE for N tokens
// instead of N times.
//
// After the batched GEMM, the per-token sequential work (conv1d, delta-net,
// attention, routing, experts) uses the proven fused_layer_forward() with
// g_pfb_precomputed_qkv=1 to skip the already-computed projections.
//
// Supported layer types:
//   - Full attention (10 layers): batched Q/K/V projections
//   - Linear attention (30 layers): batched QKV/Z/beta/alpha projections
//     The sequential part (conv1d, delta-net recurrence) runs per-token.

// ============================================================================
// Helper: dispatch one batched GEMM (N tokens, one projection matrix)
// ============================================================================
// Dispatches dequant_gemm_4bit_batch: X[N, in_dim] @ W^T -> Y[N, out_dim]
// where X is in buf_pfb_input and Y goes to buf_pfb_out[slot].
//
// The GEMM kernel reads each weight row once and multiplies against all N inputs.
// For N=16, this is ~16x fewer weight reads vs N separate matvecs.

static void pfb_encode_gemm(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    const void *W, const void *scales, const void *biases,
    uint32_t out_dim, uint32_t in_dim, uint32_t group_size,
    int out_slot,   // which buf_pfb_out[slot] to write to
    uint32_t batch_n
) {
    id<MTLBuffer> w_buf, s_buf, b_buf;
    NSUInteger w_off, s_off, b_off;
    size_t w_size = (size_t)out_dim * in_dim / 8;
    size_t num_groups = (in_dim + group_size - 1) / group_size;
    size_t sb_size = (size_t)out_dim * num_groups * sizeof(uint16_t);

    metal_find_chunk_sized(ctx, W, w_size, &w_buf, &w_off);
    metal_find_chunk_sized(ctx, scales, sb_size, &s_buf, &s_off);
    metal_find_chunk_sized(ctx, biases, sb_size, &b_buf, &b_off);

    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
    [enc setComputePipelineState:ctx->pfb_gemm_4bit];
    [enc setBuffer:w_buf                    offset:w_off  atIndex:0];
    [enc setBuffer:s_buf                    offset:s_off  atIndex:1];
    [enc setBuffer:b_buf                    offset:b_off  atIndex:2];
    [enc setBuffer:ctx->buf_pfb_input       offset:0      atIndex:3];
    [enc setBuffer:ctx->buf_pfb_out[out_slot] offset:0    atIndex:4];
    [enc setBytes:&out_dim                  length:4      atIndex:5];
    [enc setBytes:&in_dim                   length:4      atIndex:6];
    [enc setBytes:&group_size               length:4      atIndex:7];
    [enc setBytes:&batch_n                  length:4      atIndex:8];

    // 256 threads per TG, 8 SIMD groups = 8 rows per TG
    uint32_t simd_size = 32;
    uint32_t rows_per_tg = 256 / simd_size;  // 8
    uint32_t num_tgs = (out_dim + rows_per_tg - 1) / rows_per_tg;
    [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
}

// ============================================================================
// Helper: dispatch batched RMS norm (N tokens)
// ============================================================================
// Normalizes X[N, dim] in-place (or in -> out) using weight[dim] (bf16).

static void pfb_encode_rms_norm(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    id<MTLBuffer> in_buf,
    const void *norm_weight_ptr,  // bf16 weight pointer in mmap'd file
    id<MTLBuffer> out_buf,
    uint32_t dim,
    float eps,
    uint32_t batch_n
) {
    id<MTLBuffer> nw_buf; NSUInteger nw_off;
    metal_find_chunk_sized(ctx, norm_weight_ptr, (size_t)dim * 2, &nw_buf, &nw_off);

    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
    [enc setComputePipelineState:ctx->pfb_rms_norm];
    [enc setBuffer:in_buf   offset:0      atIndex:0];
    [enc setBuffer:nw_buf   offset:nw_off atIndex:1];
    [enc setBuffer:out_buf  offset:0      atIndex:2];
    [enc setBytes:&dim      length:4      atIndex:3];
    [enc setBytes:&eps      length:4      atIndex:4];
    [enc setBytes:&batch_n  length:4      atIndex:5];
    // One threadgroup per token, 256 threads per TG
    [enc dispatchThreadgroups:MTLSizeMake(batch_n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
}

// ============================================================================
// Batched prefill entry point (v2 — layer-first with batched GEMM)
// ============================================================================
// Processes tokens [0, num_tokens-1] through all layers in layer-first order.
// The last prefill token is NOT processed here — caller handles it with
// fused_layer_forward + complete_deferred_experts for full hidden state output.
//
// For each layer:
//   1. Batched RMS norm: hidden[N, dim] -> normed[N, dim] on GPU
//   2. Batched GEMM: normed[N, dim] @ W^T -> proj[N, proj_dim] for each projection
//   3. Per-token: copy projection slice, set g_pfb_precomputed_qkv=1, call
//      fused_layer_forward (which skips norm + projections, runs sequential part)
//   4. Complete deferred experts for each token before moving to next
//
// Falls back to per-token matvec when GPU GEMM is unavailable.

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

    // Check GPU GEMM availability
    int have_gpu_gemm = (g_metal && g_metal->pfb_gemm_4bit &&
                         g_metal->pfb_rms_norm &&
                         g_metal->buf_pfb_input && g_metal->buf_pfb_out[0]);

    // Allocate per-token hidden state buffers
    size_t hbuf_size = (size_t)num_tokens * cfg.hidden_dim * sizeof(float);
    float *hidden_states = (float *)malloc(hbuf_size);
    if (!hidden_states) {
        fprintf(stderr, "[batched_prefill] ERROR: Failed to allocate hidden buffer (%.1f MB)\n",
                hbuf_size / 1e6);
        return 0;
    }

    // Initialize with embedded tokens
    memcpy(hidden_states, embed_batch, hbuf_size);

    // A single per-token hidden buffer for fused_layer_forward (it works in-place)
    float *token_hidden = (float *)malloc(cfg.hidden_dim * sizeof(float));
    if (!token_hidden) {
        fprintf(stderr, "[batched_prefill] ERROR: Failed to allocate token hidden buffer\n");
        free(hidden_states);
        return 0;
    }

    // Ensure layer cache is built
    if (!layer_cache_built) build_layer_cache(wf);
    if (init_layer_scratch() != 0) {
        fprintf(stderr, "[batched_prefill] ERROR: scratch buffer allocation failed\n");
        free(hidden_states); free(token_hidden);
        return 0;
    }

    int gemm_layers = 0, fallback_layers = 0;

    // Layer-first: for each layer, process all tokens through it
    for (int layer = 0; layer < cfg.num_layers; layer++) {
        int is_full = cfg.is_full_attn[layer];
        const void *mmap_base = (layer_mmaps[layer] != MAP_FAILED) ? layer_mmaps[layer] : NULL;
        LayerWeightCache *lc = &layer_cache[layer];

        // Determine batch size for GPU dispatch (capped at MAX_PFB_GPU)
        int gpu_batch = (num_tokens > MAX_PFB_GPU) ? MAX_PFB_GPU : num_tokens;

        // Check if we can use batched GEMM for this layer
        int can_gemm = 0;
        if (have_gpu_gemm && lc->input_norm_w) {
            if (is_full) {
                can_gemm = (lc->q_w && lc->q_s && lc->q_b &&
                            lc->k_w && lc->k_s && lc->k_b &&
                            lc->v_w && lc->v_s && lc->v_b);
            } else {
                can_gemm = (lc->qkv_w && lc->qkv_s && lc->qkv_b &&
                            lc->z_w && lc->z_s && lc->z_b &&
                            lc->b_w && lc->b_s && lc->b_b &&
                            lc->a_w && lc->a_s && lc->a_b);
            }
        }

        if (can_gemm) {
            gemm_layers++;

            // Process tokens in GPU-batch-sized chunks
            for (int chunk_start = 0; chunk_start < num_tokens; chunk_start += gpu_batch) {
                int chunk_n = num_tokens - chunk_start;
                if (chunk_n > gpu_batch) chunk_n = gpu_batch;
                uint32_t batch_n = (uint32_t)chunk_n;

                // ---- Step 1: Copy chunk of hidden states to GPU input buffer ----
                float *pfb_input = (float *)[g_metal->buf_pfb_input contents];
                for (int t = 0; t < chunk_n; t++) {
                    float *h = hidden_states + (size_t)(chunk_start + t) * cfg.hidden_dim;
                    memcpy(pfb_input + (size_t)t * cfg.hidden_dim, h,
                           cfg.hidden_dim * sizeof(float));
                }

                // ---- Step 2: Batched RMS norm + GEMM projections ----
                id<MTLCommandBuffer> cmd_pfb = [g_metal->queue commandBuffer];
                metal_staging_reset(g_metal);

                // RMS norm: buf_pfb_input -> buf_pfb_out[7] (use slot 7 as normed scratch)
                pfb_encode_rms_norm(g_metal, cmd_pfb,
                                    g_metal->buf_pfb_input, lc->input_norm_w,
                                    g_metal->buf_pfb_out[7],
                                    (uint32_t)cfg.hidden_dim, cfg.rms_norm_eps, batch_n);

                // Swap: normed output becomes GEMM input
                // We need to copy buf_pfb_out[7] -> buf_pfb_input for the GEMM kernel.
                // Instead, encode a blit copy within the same command buffer.
                {
                    id<MTLBlitCommandEncoder> blit = [cmd_pfb blitCommandEncoder];
                    size_t copy_size = (size_t)batch_n * cfg.hidden_dim * sizeof(float);
                    [blit copyFromBuffer:g_metal->buf_pfb_out[7] sourceOffset:0
                                toBuffer:g_metal->buf_pfb_input destinationOffset:0
                                    size:copy_size];
                    [blit endEncoding];
                }

                if (is_full) {
                    // Full attention: Q, K, V projections -> slots 0, 1, 2
                    int q_proj_dim = cfg.num_attn_heads * cfg.head_dim * 2;
                    int kv_dim = cfg.num_kv_heads * cfg.head_dim;

                    pfb_encode_gemm(g_metal, cmd_pfb,
                                    lc->q_w, lc->q_s, lc->q_b,
                                    (uint32_t)q_proj_dim, (uint32_t)cfg.hidden_dim,
                                    (uint32_t)cfg.group_size, 0, batch_n);
                    pfb_encode_gemm(g_metal, cmd_pfb,
                                    lc->k_w, lc->k_s, lc->k_b,
                                    (uint32_t)kv_dim, (uint32_t)cfg.hidden_dim,
                                    (uint32_t)cfg.group_size, 1, batch_n);
                    pfb_encode_gemm(g_metal, cmd_pfb,
                                    lc->v_w, lc->v_s, lc->v_b,
                                    (uint32_t)kv_dim, (uint32_t)cfg.hidden_dim,
                                    (uint32_t)cfg.group_size, 2, batch_n);
                } else {
                    // Linear attention: QKV, Z, beta, alpha projections -> slots 0, 1, 2, 3
                    int qkv_dim = cfg.linear_conv_dim;
                    int z_dim = cfg.linear_total_value;

                    pfb_encode_gemm(g_metal, cmd_pfb,
                                    lc->qkv_w, lc->qkv_s, lc->qkv_b,
                                    (uint32_t)qkv_dim, (uint32_t)cfg.hidden_dim,
                                    (uint32_t)cfg.group_size, 0, batch_n);
                    pfb_encode_gemm(g_metal, cmd_pfb,
                                    lc->z_w, lc->z_s, lc->z_b,
                                    (uint32_t)z_dim, (uint32_t)cfg.hidden_dim,
                                    (uint32_t)cfg.group_size, 1, batch_n);
                    pfb_encode_gemm(g_metal, cmd_pfb,
                                    lc->b_w, lc->b_s, lc->b_b,
                                    (uint32_t)cfg.linear_num_v_heads, (uint32_t)cfg.hidden_dim,
                                    (uint32_t)cfg.group_size, 2, batch_n);
                    pfb_encode_gemm(g_metal, cmd_pfb,
                                    lc->a_w, lc->a_s, lc->a_b,
                                    (uint32_t)cfg.linear_num_v_heads, (uint32_t)cfg.hidden_dim,
                                    (uint32_t)cfg.group_size, 3, batch_n);
                }

                [cmd_pfb commit];
                [cmd_pfb waitUntilCompleted];

                // ---- Step 3: Read back normed input and projections, then per-token forward ----
                float *normed_buf = (float *)[g_metal->buf_pfb_input contents];  // normed is now in pfb_input (after blit)

                for (int t = 0; t < chunk_n; t++) {
                    int global_t = chunk_start + t;
                    int pos = start_pos + global_t;
                    float *h = hidden_states + (size_t)global_t * cfg.hidden_dim;

                    // Set up residual (= hidden state before attention)
                    memcpy(s_residual, h, cfg.hidden_dim * sizeof(float));

                    // Copy normed input for this token
                    memcpy(s_normed, normed_buf + (size_t)t * cfg.hidden_dim,
                           cfg.hidden_dim * sizeof(float));

                    // Copy projection results for this token
                    if (is_full) {
                        int q_proj_dim = cfg.num_attn_heads * cfg.head_dim * 2;
                        int kv_dim = cfg.num_kv_heads * cfg.head_dim;
                        float *q_buf = (float *)[g_metal->buf_pfb_out[0] contents];
                        float *k_buf = (float *)[g_metal->buf_pfb_out[1] contents];
                        float *v_buf = (float *)[g_metal->buf_pfb_out[2] contents];
                        memcpy(s_q_proj_out, q_buf + (size_t)t * q_proj_dim,
                               q_proj_dim * sizeof(float));
                        memcpy(s_k_proj_out, k_buf + (size_t)t * kv_dim,
                               kv_dim * sizeof(float));
                        memcpy(s_v_proj_out, v_buf + (size_t)t * kv_dim,
                               kv_dim * sizeof(float));
                    } else {
                        int qkv_dim = cfg.linear_conv_dim;
                        int z_dim = cfg.linear_total_value;
                        float *qkv_buf = (float *)[g_metal->buf_pfb_out[0] contents];
                        float *z_buf   = (float *)[g_metal->buf_pfb_out[1] contents];
                        float *b_buf   = (float *)[g_metal->buf_pfb_out[2] contents];
                        float *a_buf   = (float *)[g_metal->buf_pfb_out[3] contents];
                        memcpy(s_qkv_proj_out, qkv_buf + (size_t)t * qkv_dim,
                               qkv_dim * sizeof(float));
                        memcpy(s_z_proj_out, z_buf + (size_t)t * z_dim,
                               z_dim * sizeof(float));
                        memcpy(s_beta_proj_out, b_buf + (size_t)t * cfg.linear_num_v_heads,
                               cfg.linear_num_v_heads * sizeof(float));
                        memcpy(s_alpha_proj_out, a_buf + (size_t)t * cfg.linear_num_v_heads,
                               cfg.linear_num_v_heads * sizeof(float));
                    }

                    // Copy hidden state for per-token forward
                    memcpy(token_hidden, h, cfg.hidden_dim * sizeof(float));

                    // Run per-token forward with precomputed projections
                    g_pfb_precomputed_qkv = 1;
                    fused_layer_forward(wf, layer, token_hidden,
                                        is_full ? kv_caches[layer] : NULL,
                                        is_full ? NULL : (LinearAttnState *)layer_states[layer],
                                        pos,
                                        mmap_base,
                                        K, layer_fds[layer]);
                    g_pfb_precomputed_qkv = 0;

                    // Complete deferred expert work
                    complete_deferred_experts();

                    // Copy result back
                    memcpy(h, token_hidden, cfg.hidden_dim * sizeof(float));
                }
            }
        } else {
            // ---- FALLBACK: per-token matvec (no batched GEMM available) ----
            fallback_layers++;
            for (int t = 0; t < num_tokens; t++) {
                int pos = start_pos + t;
                float *h = hidden_states + (size_t)t * cfg.hidden_dim;

                memcpy(token_hidden, h, cfg.hidden_dim * sizeof(float));

                fused_layer_forward(wf, layer, token_hidden,
                                    is_full ? kv_caches[layer] : NULL,
                                    is_full ? NULL : (LinearAttnState *)layer_states[layer],
                                    pos,
                                    mmap_base,
                                    K, layer_fds[layer]);

                complete_deferred_experts();

                memcpy(h, token_hidden, cfg.hidden_dim * sizeof(float));
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

// ============================================================================
// Validation: compare batched GEMM output against per-token matvec for one layer.
// Runs on the first full-attention AND first linear-attention layer to verify
// both code paths produce matching results.
// ============================================================================

static void pfb_validate_layer(WeightFile *wf, float *test_hidden, int layer_idx) {
    if (!g_metal || !g_metal->pfb_gemm_4bit || !g_metal->pfb_rms_norm ||
        !g_metal->buf_pfb_input || !g_metal->buf_pfb_out[0]) {
        fprintf(stderr, "[pfb_validate] Skipping: GPU GEMM not available\n");
        return;
    }
    if (!layer_cache_built) build_layer_cache(wf);
    LayerWeightCache *lc = &layer_cache[layer_idx];
    int is_full = cfg.is_full_attn[layer_idx];

    fprintf(stderr, "[pfb_validate] Validating layer %d (%s attention)...\n",
            layer_idx, is_full ? "full" : "linear");

    // ---- Reference: single-token matvec on CPU ----
    float *normed_ref = calloc(cfg.hidden_dim, sizeof(float));
    cpu_rms_norm(test_hidden, lc->input_norm_w, normed_ref, cfg.hidden_dim, cfg.rms_norm_eps);

    int num_projs;
    int proj_dims[4];
    float *ref_out[4];
    BatchMatvecSpec ref_specs[4];

    if (is_full) {
        int q_proj_dim = cfg.num_attn_heads * cfg.head_dim * 2;
        int kv_dim = cfg.num_kv_heads * cfg.head_dim;
        num_projs = 3;
        proj_dims[0] = q_proj_dim; proj_dims[1] = kv_dim; proj_dims[2] = kv_dim;
        for (int i = 0; i < num_projs; i++) ref_out[i] = calloc(proj_dims[i], sizeof(float));
        ref_specs[0] = (BatchMatvecSpec){ lc->q_w, lc->q_s, lc->q_b, ref_out[0], (uint32_t)q_proj_dim, cfg.hidden_dim, cfg.group_size, 0 };
        ref_specs[1] = (BatchMatvecSpec){ lc->k_w, lc->k_s, lc->k_b, ref_out[1], (uint32_t)kv_dim,     cfg.hidden_dim, cfg.group_size, 1 };
        ref_specs[2] = (BatchMatvecSpec){ lc->v_w, lc->v_s, lc->v_b, ref_out[2], (uint32_t)kv_dim,     cfg.hidden_dim, cfg.group_size, 2 };
    } else {
        int qkv_dim = cfg.linear_conv_dim;
        int z_dim = cfg.linear_total_value;
        num_projs = 4;
        proj_dims[0] = qkv_dim; proj_dims[1] = z_dim;
        proj_dims[2] = cfg.linear_num_v_heads; proj_dims[3] = cfg.linear_num_v_heads;
        for (int i = 0; i < num_projs; i++) ref_out[i] = calloc(proj_dims[i], sizeof(float));
        ref_specs[0] = (BatchMatvecSpec){ lc->qkv_w, lc->qkv_s, lc->qkv_b, ref_out[0], (uint32_t)qkv_dim,            cfg.hidden_dim, cfg.group_size, 0 };
        ref_specs[1] = (BatchMatvecSpec){ lc->z_w,   lc->z_s,   lc->z_b,   ref_out[1], (uint32_t)z_dim,              cfg.hidden_dim, cfg.group_size, 1 };
        ref_specs[2] = (BatchMatvecSpec){ lc->b_w,   lc->b_s,   lc->b_b,   ref_out[2], (uint32_t)cfg.linear_num_v_heads, cfg.hidden_dim, cfg.group_size, 2 };
        ref_specs[3] = (BatchMatvecSpec){ lc->a_w,   lc->a_s,   lc->a_b,   ref_out[3], (uint32_t)cfg.linear_num_v_heads, cfg.hidden_dim, cfg.group_size, 3 };
    }

    // Run reference matvecs (single-token GPU path)
    for (int i = 0; i < num_projs; i++) {
        cpu_dequant_matvec(ref_specs[i].W, ref_specs[i].scales, ref_specs[i].biases,
                           normed_ref, ref_specs[i].out_cpu,
                           ref_specs[i].out_dim, ref_specs[i].in_dim, ref_specs[i].group_size);
    }

    // ---- Batched GEMM: single token batch (N=1) ----
    memcpy([g_metal->buf_pfb_input contents], test_hidden, cfg.hidden_dim * sizeof(float));
    uint32_t batch_n = 1;

    id<MTLCommandBuffer> cmd = [g_metal->queue commandBuffer];
    metal_staging_reset(g_metal);

    pfb_encode_rms_norm(g_metal, cmd,
                        g_metal->buf_pfb_input, lc->input_norm_w,
                        g_metal->buf_pfb_out[7],
                        (uint32_t)cfg.hidden_dim, cfg.rms_norm_eps, batch_n);
    {
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        [blit copyFromBuffer:g_metal->buf_pfb_out[7] sourceOffset:0
                    toBuffer:g_metal->buf_pfb_input destinationOffset:0
                        size:cfg.hidden_dim * sizeof(float)];
        [blit endEncoding];
    }

    for (int i = 0; i < num_projs; i++) {
        pfb_encode_gemm(g_metal, cmd,
                        ref_specs[i].W, ref_specs[i].scales, ref_specs[i].biases,
                        ref_specs[i].out_dim, ref_specs[i].in_dim,
                        ref_specs[i].group_size, i, batch_n);
    }

    [cmd commit];
    [cmd waitUntilCompleted];

    // ---- Compare results ----
    const char *proj_names_full[] = {"Q", "K", "V"};
    const char *proj_names_linear[] = {"QKV", "Z", "beta", "alpha"};
    const char **proj_names = is_full ? proj_names_full : proj_names_linear;
    int all_ok = 1;

    for (int i = 0; i < num_projs; i++) {
        float *gemm_out = (float *)[g_metal->buf_pfb_out[i] contents];
        float max_err = 0.0f;
        float rms_err = 0.0f;
        float ref_rms = 0.0f;
        for (int j = 0; j < proj_dims[i]; j++) {
            float err = fabsf(gemm_out[j] - ref_out[i][j]);
            if (err > max_err) max_err = err;
            rms_err += err * err;
            ref_rms += ref_out[i][j] * ref_out[i][j];
        }
        rms_err = sqrtf(rms_err / proj_dims[i]);
        ref_rms = sqrtf(ref_rms / proj_dims[i]);
        float rel_err = (ref_rms > 1e-8f) ? rms_err / ref_rms : rms_err;

        if (rel_err > 0.01f || max_err > 1.0f) {
            fprintf(stderr, "[pfb_validate] FAIL layer %d %s: max_err=%.6f rms_err=%.6f rel=%.4f%%\n",
                    layer_idx, proj_names[i], max_err, rms_err, rel_err * 100.0f);
            all_ok = 0;
        } else {
            fprintf(stderr, "[pfb_validate]   OK  layer %d %s: max_err=%.6f rms_err=%.6f rel=%.4f%%\n",
                    layer_idx, proj_names[i], max_err, rms_err, rel_err * 100.0f);
        }
    }

    if (all_ok) {
        fprintf(stderr, "[pfb_validate] Layer %d PASSED (%s attention, %d projections)\n",
                layer_idx, is_full ? "full" : "linear", num_projs);
    } else {
        fprintf(stderr, "[pfb_validate] Layer %d FAILED\n", layer_idx);
    }

    // Cleanup
    free(normed_ref);
    for (int i = 0; i < num_projs; i++) free(ref_out[i]);
}

// Run validation on the first full-attention and first linear-attention layer.
static void pfb_validate_all(WeightFile *wf) {
    // Generate a deterministic test hidden state
    float *test_hidden = calloc(cfg.hidden_dim, sizeof(float));
    for (int i = 0; i < cfg.hidden_dim; i++) {
        // Simple PRNG: reproducible but not all-zeros
        test_hidden[i] = sinf((float)i * 0.01f) * 0.1f;
    }

    int found_full = 0, found_linear = 0;
    for (int i = 0; i < cfg.num_layers && (!found_full || !found_linear); i++) {
        if (cfg.is_full_attn[i] && !found_full) {
            pfb_validate_layer(wf, test_hidden, i);
            found_full = 1;
        } else if (!cfg.is_full_attn[i] && !found_linear) {
            pfb_validate_layer(wf, test_hidden, i);
            found_linear = 1;
        }
    }

    free(test_hidden);
}
