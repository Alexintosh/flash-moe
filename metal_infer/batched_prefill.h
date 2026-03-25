// batched_prefill.h — Layer-first batched prefill (v2)
// Part of Flash-MoE modular decomposition (unity build).
// Included by infer.m — do NOT compile separately.
//
// Strategy: LAYER-FIRST ordering using the PROVEN per-token fused_layer_forward.
//
// The previous batched prefill tried to re-implement the entire layer pipeline
// (attention, norm, routing, experts, combine) in batched form and broke
// repeatedly. This v2 takes a different approach:
//
//   1. Use fused_layer_forward() for each token — the same code path that works
//      perfectly in single-token generation.
//   2. Process ALL tokens through layer L before moving to layer L+1
//      (layer-first ordering), instead of the default token-first ordering
//      (token T through all layers, then token T+1 through all layers).
//
// Why layer-first helps:
//   - Layer weights (non-expert) stay in the Metal buffer cache across tokens
//   - Expert routing patterns within a layer are similar across consecutive tokens,
//     so the OS page cache gets better hit rates on expert data
//   - The per-token path reads each weight row once per token, but with layer-first
//     the weights are more likely to be cache-hot for subsequent tokens
//
// This is guaranteed correct because it uses the exact same fused_layer_forward()
// that single-token generation uses. The only difference is traversal order.
//
// Future optimization: replace the inner per-token matvecs with batched GEMM
// (dequant_gemm_4bit_batch) once this layer-first ordering is validated.

// ============================================================================
// Batched prefill entry point (v2 — layer-first, per-token forward)
// ============================================================================
// Processes tokens [0, num_tokens-1] through all layers in layer-first order.
// The last prefill token is NOT processed here — caller handles it with
// fused_layer_forward + complete_deferred_experts for full hidden state output.
//
// Parameters:
//   wf:           Weight file (mmap'd non-expert weights)
//   embed_batch:  [num_tokens, hidden_dim] pre-embedded tokens
//   num_tokens:   Number of tokens to prefill
//   start_pos:    Starting RoPE position
//   K:            Number of active experts per token
//   layer_fds:    [num_layers] file descriptors for expert data
//   layer_mmaps:  [num_layers] mmap'd expert data (or MAP_FAILED)
//   kv_caches:    [num_layers] KV caches (full attention layers)
//   layer_states: [num_layers] delta-net state (linear attention layers)
//
// Returns the number of tokens actually processed (num_tokens on success, 0 on failure).

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

    // Allocate per-token hidden state buffers.
    // We need num_tokens separate hidden buffers because layer-first ordering
    // means token T's output from layer L is the input to layer L+1, and we
    // process all tokens through layer L before moving to L+1.
    //
    // Layout: hidden_states[t] points to the hidden state for token t.
    // After processing layer L, hidden_states[t] contains the output of layer L
    // for token t, ready to be input to layer L+1.
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

    // Layer-first: for each layer, process all tokens through it
    for (int layer = 0; layer < cfg.num_layers; layer++) {
        int is_full = cfg.is_full_attn[layer];
        const void *mmap_base = (layer_mmaps[layer] != MAP_FAILED) ? layer_mmaps[layer] : NULL;

        for (int t = 0; t < num_tokens; t++) {
            int pos = start_pos + t;
            float *h = hidden_states + (size_t)t * cfg.hidden_dim;

            // Copy into the per-token working buffer
            memcpy(token_hidden, h, cfg.hidden_dim * sizeof(float));

            // Run the proven per-token forward pass for this layer
            fused_layer_forward(wf, layer, token_hidden,
                                is_full ? kv_caches[layer] : NULL,
                                is_full ? NULL : (LinearAttnState *)layer_states[layer],
                                pos,
                                mmap_base,
                                K, layer_fds[layer]);

            // Complete deferred expert work — experts contribute ~80% of each
            // layer's computation. Discarding them corrupts the hidden state.
            complete_deferred_experts();

            // Copy result back (fused_layer_forward modifies token_hidden in-place,
            // and complete_deferred_experts applies expert output to it)
            memcpy(h, token_hidden, cfg.hidden_dim * sizeof(float));
        }
    }

    free(token_hidden);
    free(hidden_states);

    double elapsed = now_ms() - t_start;
    fprintf(stderr, "[batched_prefill] %d tokens, layer-first, %.1f ms\n",
            num_tokens, elapsed);

    return num_tokens;
}
