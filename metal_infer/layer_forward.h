// layer_forward.h — RoPE, KVCache, attention, MoE forward, embedding, lm_head,
// LayerWeightCache, DeferredExpertState, fused_layer_forward
// Part of Flash-MoE modular decomposition (unity build).
// Included by infer.m — do NOT compile separately.

// ============================================================================
// Rotary position embedding (for full attention layers)
// ============================================================================

static void apply_rotary_emb(float *q, float *k, int pos, int num_heads, int num_kv_heads,
                              int head_dim, int rotary_dim) {
    // Apply RoPE to the first rotary_dim dimensions of each head
    // NON-TRADITIONAL (MLX default): pairs are (x[i], x[i + half_dim])
    // where half_dim = rotary_dim / 2
    int half = rotary_dim / 2;
    for (int h = 0; h < num_heads; h++) {
        float *qh = q + h * head_dim;
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(cfg.rope_theta, (float)(2 * i) / rotary_dim);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);

            float q0 = qh[i];
            float q1 = qh[i + half];
            qh[i]        = q0 * cos_a - q1 * sin_a;
            qh[i + half]  = q0 * sin_a + q1 * cos_a;
        }
    }
    for (int h = 0; h < num_kv_heads; h++) {
        float *kh = k + h * head_dim;
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(cfg.rope_theta, (float)(2 * i) / rotary_dim);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);

            float k0 = kh[i];
            float k1 = kh[i + half];
            kh[i]        = k0 * cos_a - k1 * sin_a;
            kh[i + half]  = k0 * sin_a + k1 * cos_a;
        }
    }
}

// ============================================================================
// KV Cache for full attention layers
// ============================================================================

typedef struct {
    float *k_cache;      // [max_seq, num_kv_heads * head_dim] (NULL when use_fp8=1)
    float *v_cache;      // [max_seq, num_kv_heads * head_dim] (NULL when use_fp8=1)
    uint8_t *k_cache_fp8;  // [max_seq, num_kv_heads * head_dim] FP8 E4M3 (NULL when use_fp8=0)
    uint8_t *v_cache_fp8;  // [max_seq, num_kv_heads * head_dim] FP8 E4M3 (NULL when use_fp8=0)
    float *k_scales;     // [max_seq] per-position K scale (FP8 only)
    float *v_scales;     // [max_seq] per-position V scale (FP8 only)
    int len;             // total tokens written (monotonically increasing)
    int use_fp8;         // 1 = FP8 E4M3, 0 = float32
    int window_size;     // >0: sliding window (circular buffer), 0: unlimited
    int capacity;        // allocated size (= window_size if sliding, else max_seq)
} KVCache;

static KVCache *kv_cache_new(void) {
    int max_seq = g_kv_seq_len > 0 ? g_kv_seq_len : cfg.max_seq_len;
    KVCache *c = calloc(1, sizeof(KVCache));
    c->len = 0;
    c->use_fp8 = g_use_fp8_kv;
    c->window_size = g_sliding_window;  // 0 = unlimited, >0 = circular buffer
    // Capacity: if sliding window, only allocate window_size positions
    int seq = (c->window_size > 0 && c->window_size < max_seq) ? c->window_size : max_seq;
    c->capacity = seq;
    size_t kv_dim = (size_t)cfg.num_kv_heads * cfg.head_dim;

    if (c->use_fp8) {
        // FP8 E4M3: 1 byte per element + per-position scale
        c->k_cache_fp8 = calloc((size_t)seq * kv_dim, sizeof(uint8_t));
        c->v_cache_fp8 = calloc((size_t)seq * kv_dim, sizeof(uint8_t));
        c->k_scales = calloc(seq, sizeof(float));
        c->v_scales = calloc(seq, sizeof(float));
        c->k_cache = NULL;
        c->v_cache = NULL;
        if (!c->k_cache_fp8 || !c->v_cache_fp8 || !c->k_scales || !c->v_scales) {
            fprintf(stderr, "ERROR: FP8 KV cache alloc failed (seq=%d, %.1f MB each)\n",
                    seq, (double)seq * kv_dim * sizeof(uint8_t) / 1e6);
            free(c->k_cache_fp8); free(c->v_cache_fp8);
            free(c->k_scales); free(c->v_scales); free(c);
            return NULL;
        }
    } else {
        // Float32 path (original)
        c->k_cache = calloc((size_t)seq * kv_dim, sizeof(float));
        c->v_cache = calloc((size_t)seq * kv_dim, sizeof(float));
        c->k_cache_fp8 = NULL;
        c->v_cache_fp8 = NULL;
        c->k_scales = NULL;
        c->v_scales = NULL;
        if (!c->k_cache || !c->v_cache) {
            fprintf(stderr, "ERROR: KV cache alloc failed (seq=%d, %.1f MB each)\n",
                    seq, (double)seq * kv_dim * sizeof(float) / 1e6);
            free(c->k_cache); free(c->v_cache); free(c);
            return NULL;
        }
    }
    return c;
}

static void kv_cache_free(KVCache *c) {
    if (c) {
        free(c->k_cache);
        free(c->v_cache);
        free(c->k_cache_fp8);
        free(c->v_cache_fp8);
        free(c->k_scales);
        free(c->v_scales);
        free(c);
    }
}

// ============================================================================
// Linear attention state (GatedDeltaNet recurrent state)
// ============================================================================

typedef struct {
    float *conv_state;  // [(kernel_size-1) * conv_dim] for conv1d
    float *ssm_state;   // [num_v_heads, head_v_dim, head_k_dim] recurrent state
} LinearAttnState;

static LinearAttnState *linear_attn_state_new(void) {
    LinearAttnState *s = calloc(1, sizeof(LinearAttnState));
    s->conv_state = calloc((cfg.conv_kernel_size - 1) * cfg.linear_conv_dim, sizeof(float));
    s->ssm_state = calloc(cfg.linear_num_v_heads * cfg.linear_value_dim * cfg.linear_key_dim, sizeof(float));
    return s;
}

static void linear_attn_state_free(LinearAttnState *s) {
    if (s) {
        free(s->conv_state);
        free(s->ssm_state);
        free(s);
    }
}

// ============================================================================
// Full attention layer forward (single token, incremental)
// ============================================================================

static int fa_debug_count = 0;

static float vec_rms(const float *v, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += v[i] * v[i];
    return sqrtf(sum / n);
}

__attribute__((unused))
static void full_attention_forward(
    WeightFile *wf,
    int layer_idx,
    float *hidden,       // [cfg.hidden_dim] in/out
    KVCache *kv,
    int pos              // position in sequence
) {
    fa_debug_count++;
    int do_debug = 0;  // set to (fa_debug_count <= N) to enable debug

    char name[256];
    float *normed = malloc(cfg.hidden_dim * sizeof(float));
    float *residual = malloc(cfg.hidden_dim * sizeof(float));
    if (!normed || !residual) {
        fprintf(stderr, "ERROR: full_attention_forward alloc failed\n");
        free(normed); free(residual); return;
    }
    cpu_vec_copy(residual, hidden, cfg.hidden_dim);

    if (do_debug) {
        fprintf(stderr, "[FA-DBG] layer=%d pos=%d hidden_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                layer_idx, pos, vec_rms(hidden, cfg.hidden_dim),
                hidden[0], hidden[1], hidden[2], hidden[3], hidden[4]);
    }

    // ---- Input LayerNorm ----
    snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", layer_idx);
    uint16_t *norm_w = get_tensor_ptr(wf, name);
    cpu_rms_norm(hidden, norm_w, normed, cfg.hidden_dim, cfg.rms_norm_eps);

    if (do_debug) {
        fprintf(stderr, "[FA-DBG] normed_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(normed, cfg.hidden_dim), normed[0], normed[1], normed[2], normed[3], normed[4]);
    }

    // ---- QKV Projection ----
    // CRITICAL: Q projection outputs num_heads * head_dim * 2 = 16384
    // The second half is a sigmoid gate applied after attention
    int q_proj_dim = cfg.num_attn_heads * cfg.head_dim * 2;  // 32 * 256 * 2 = 16384
    int q_dim = cfg.num_attn_heads * cfg.head_dim;            // 32 * 256 = 8192
    int kv_dim = cfg.num_kv_heads * cfg.head_dim;             // 2 * 256 = 512

    float *q_proj_out = calloc(q_proj_dim, sizeof(float));
    float *k = calloc(kv_dim, sizeof(float));
    float *v = calloc(kv_dim, sizeof(float));
    if (!q_proj_out || !k || !v) {
        fprintf(stderr, "ERROR: full_attention_forward QKV alloc failed\n");
        free(normed); free(residual); free(q_proj_out); free(k); free(v); return;
    }

    // Batch Q/K/V projections into a single GPU command buffer
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", layer_idx);
    uint32_t *qw = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.scales", layer_idx);
    uint16_t *qs = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.biases", layer_idx);
    uint16_t *qb = get_tensor_ptr(wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", layer_idx);
    uint32_t *kw = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.scales", layer_idx);
    uint16_t *ks = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.biases", layer_idx);
    uint16_t *kb = get_tensor_ptr(wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", layer_idx);
    uint32_t *vw = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.scales", layer_idx);
    uint16_t *vs = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.biases", layer_idx);
    uint16_t *vb = get_tensor_ptr(wf, name);

    // Batch Q/K/V into one command buffer (3 dispatches, 1 commit)
    if (qw && qs && qb && kw && ks && kb && vw && vs && vb) {
        BatchMatvecSpec qkv_specs[3] = {
            { qw, qs, qb, q_proj_out, (uint32_t)q_proj_dim, cfg.hidden_dim, cfg.group_size, 0 },
            { kw, ks, kb, k,          (uint32_t)kv_dim,     cfg.hidden_dim, cfg.group_size, 1 },
            { vw, vs, vb, v,          (uint32_t)kv_dim,     cfg.hidden_dim, cfg.group_size, 2 },
        };
        fast_batch_matvec(normed, cfg.hidden_dim, qkv_specs, 3);
    }

    if (do_debug) {
        fprintf(stderr, "[FA-DBG] q_proj first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                q_proj_out[0], q_proj_out[1], q_proj_out[2], q_proj_out[3], q_proj_out[4]);
    }

    // Split q_proj_out into queries and gate
    float *q = calloc(q_dim, sizeof(float));
    float *q_gate = calloc(q_dim, sizeof(float));
    for (int h = 0; h < cfg.num_attn_heads; h++) {
        float *src = q_proj_out + h * (2 * cfg.head_dim);
        memcpy(q + h * cfg.head_dim, src, cfg.head_dim * sizeof(float));
        memcpy(q_gate + h * cfg.head_dim, src + cfg.head_dim, cfg.head_dim * sizeof(float));
    }
    free(q_proj_out);

    if (do_debug) {
        fprintf(stderr, "[FA-DBG] v_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(v, kv_dim), v[0], v[1], v[2], v[3], v[4]);
        fprintf(stderr, "[FA-DBG] q_gate_rms=%.6f gate_first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(q_gate, q_dim), q_gate[0], q_gate[1], q_gate[2], q_gate[3], q_gate[4]);
        float gate_sigmoid_sum = 0.0f;
        for (int i = 0; i < q_dim; i++) {
            gate_sigmoid_sum += 1.0f / (1.0f + expf(-q_gate[i]));
        }
        fprintf(stderr, "[FA-DBG] gate_sigmoid_mean=%.6f\n", gate_sigmoid_sum / q_dim);
    }

    // ---- Q/K RMSNorm ----
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_norm.weight", layer_idx);
    uint16_t *qnorm_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_norm.weight", layer_idx);
    uint16_t *knorm_w = get_tensor_ptr(wf, name);

    // Apply per-head Q norm
    if (qnorm_w) {
        for (int h = 0; h < cfg.num_attn_heads; h++) {
            float *qh = q + h * cfg.head_dim;
            float sum_sq = 0.0f;
            for (int i = 0; i < cfg.head_dim; i++) sum_sq += qh[i] * qh[i];
            float inv_rms = 1.0f / sqrtf(sum_sq / cfg.head_dim + cfg.rms_norm_eps);
            for (int i = 0; i < cfg.head_dim; i++) {
                qh[i] = qh[i] * inv_rms * bf16_to_f32(qnorm_w[i]);
            }
        }
    }
    // Apply per-head K norm
    if (knorm_w) {
        for (int h = 0; h < cfg.num_kv_heads; h++) {
            float *kh = k + h * cfg.head_dim;
            float sum_sq = 0.0f;
            for (int i = 0; i < cfg.head_dim; i++) sum_sq += kh[i] * kh[i];
            float inv_rms = 1.0f / sqrtf(sum_sq / cfg.head_dim + cfg.rms_norm_eps);
            for (int i = 0; i < cfg.head_dim; i++) {
                kh[i] = kh[i] * inv_rms * bf16_to_f32(knorm_w[i]);
            }
        }
    }


    // ---- RoPE ----
    apply_rotary_emb(q, k, pos, cfg.num_attn_heads, cfg.num_kv_heads, cfg.head_dim, cfg.rotary_dim);

    // ---- Update KV cache (circular buffer for sliding window) ----
    int cache_pos;
    if (kv->window_size > 0) {
        cache_pos = kv->len % kv->capacity;  // circular write
    } else {
        cache_pos = kv->len;
        if (cache_pos >= kv->capacity) {
            fprintf(stderr, "ERROR: KV cache overflow (pos=%d >= cap=%d)\n", cache_pos, kv->capacity);
            return;
        }
    }
    if (kv->use_fp8) {
        if (!kv->k_cache_fp8 || !kv->v_cache_fp8) {
            fprintf(stderr, "ERROR: FP8 KV cache is NULL (alloc failed)\n");
            return;
        }
        kv->k_scales[cache_pos] = fp8_encode_vec(k, kv->k_cache_fp8 + cache_pos * kv_dim, kv_dim);
        kv->v_scales[cache_pos] = fp8_encode_vec(v, kv->v_cache_fp8 + cache_pos * kv_dim, kv_dim);
    } else {
        if (!kv->k_cache || !kv->v_cache) {
            fprintf(stderr, "ERROR: KV cache is NULL (alloc failed)\n");
            return;
        }
        memcpy(kv->k_cache + cache_pos * kv_dim, k, kv_dim * sizeof(float));
        memcpy(kv->v_cache + cache_pos * kv_dim, v, kv_dim * sizeof(float));
    }
    kv->len++;

    // ---- Scaled dot-product attention ----
    // GQA: cfg.num_attn_heads=32 heads, cfg.num_kv_heads=2 kv heads
    // Each group of 16 query heads shares 1 kv head
    int heads_per_kv = cfg.num_attn_heads / cfg.num_kv_heads;
    float scale = 1.0f / sqrtf((float)cfg.head_dim);

    float *attn_out = calloc(q_dim, sizeof(float));

    // Temp buffer for dequantized K/V when using FP8
    float *k_dequant = kv->use_fp8 ? malloc(kv_dim * sizeof(float)) : NULL;
    float *v_dequant = kv->use_fp8 ? malloc(kv_dim * sizeof(float)) : NULL;

    // Determine attention range: all entries, or sliding window
    int attn_len;  // number of positions to attend over
    if (kv->window_size > 0 && kv->len > kv->window_size) {
        attn_len = kv->window_size;  // attend only within window
    } else {
        attn_len = kv->len;          // attend to everything
    }

    for (int h = 0; h < cfg.num_attn_heads; h++) {
        int kv_h = h / heads_per_kv;
        float *qh = q + h * cfg.head_dim;

        // Compute attention scores for positions within window
        float *scores = malloc(attn_len * sizeof(float));
        if (!scores) {
            fprintf(stderr, "ERROR: attention scores alloc failed (len=%d)\n", attn_len);
            continue;
        }
        for (int i = 0; i < attn_len; i++) {
            // Map logical index i → physical position in circular buffer
            int p;
            if (kv->window_size > 0 && kv->len > kv->window_size) {
                // Circular: oldest valid entry is at (kv->len - window_size)
                p = (kv->len - kv->window_size + i) % kv->capacity;
            } else {
                p = i;
            }
            float dot = 0.0f;
            if (kv->use_fp8) {
                fp8_decode_vec(kv->k_cache_fp8 + p * kv_dim, k_dequant, kv_dim, kv->k_scales[p]);
                float *kp = k_dequant + kv_h * cfg.head_dim;
                for (int d = 0; d < cfg.head_dim; d++) dot += qh[d] * kp[d];
            } else {
                float *kp = kv->k_cache + p * kv_dim + kv_h * cfg.head_dim;
                for (int d = 0; d < cfg.head_dim; d++) dot += qh[d] * kp[d];
            }
            scores[i] = dot * scale;
        }

        // Softmax over window
        cpu_softmax(scores, attn_len);

        // Weighted sum of values within window
        float *oh = attn_out + h * cfg.head_dim;
        for (int i = 0; i < attn_len; i++) {
            int p;
            if (kv->window_size > 0 && kv->len > kv->window_size) {
                p = (kv->len - kv->window_size + i) % kv->capacity;
            } else {
                p = i;
            }
            if (kv->use_fp8) {
                fp8_decode_vec(kv->v_cache_fp8 + p * kv_dim, v_dequant, kv_dim, kv->v_scales[p]);
                float *vp = v_dequant + kv_h * cfg.head_dim;
                for (int d = 0; d < cfg.head_dim; d++) oh[d] += scores[i] * vp[d];
            } else {
                float *vp = kv->v_cache + p * kv_dim + kv_h * cfg.head_dim;
                for (int d = 0; d < cfg.head_dim; d++) oh[d] += scores[i] * vp[d];
            }
        }
        free(scores);
    }
    free(k_dequant);
    free(v_dequant);


    // ---- Apply sigmoid gate to attention output ----
    // MLX: return self.o_proj(output * mx.sigmoid(gate))
    // gate is reshaped to [B, L, num_heads*head_dim] = flat [q_dim]
    for (int i = 0; i < q_dim; i++) {
        float g = 1.0f / (1.0f + expf(-q_gate[i]));
        attn_out[i] *= g;
    }

    // ---- Output projection ----
    float *attn_projected = calloc(cfg.hidden_dim, sizeof(float));
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", layer_idx);
    uint32_t *ow = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.scales", layer_idx);
    uint16_t *os_ptr = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.biases", layer_idx);
    uint16_t *ob = get_tensor_ptr(wf, name);
    if (ow && os_ptr && ob) fast_dequant_matvec(ow, os_ptr, ob, attn_out, attn_projected, cfg.hidden_dim, q_dim, cfg.group_size);

    if (do_debug) {
        fprintf(stderr, "[FA-DBG] attn_out_rms=%.6f o_proj first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(attn_out, q_dim),
                attn_projected[0], attn_projected[1], attn_projected[2], attn_projected[3], attn_projected[4]);
    }

    // ---- Residual connection ----
    for (int i = 0; i < cfg.hidden_dim; i++) {
        hidden[i] = residual[i] + attn_projected[i];
    }

    if (do_debug) {
        fprintf(stderr, "[FA-DBG] AFTER layer=%d hidden_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                layer_idx, vec_rms(hidden, cfg.hidden_dim),
                hidden[0], hidden[1], hidden[2], hidden[3], hidden[4]);
    }

    free(normed);
    free(residual);
    free(q);
    free(q_gate);
    free(k);
    free(v);
    free(attn_out);
    free(attn_projected);
}

// ============================================================================
// Linear attention layer forward (GatedDeltaNet, single token, incremental)
// ============================================================================

// RMS norm without weights (just normalize)
static void cpu_rms_norm_bare(const float *x, float *out, int dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(sum_sq / dim + eps);
    for (int i = 0; i < dim; i++) out[i] = x[i] * inv_rms;
}

// RMSNormGated: out = rms_norm(x) * silu(z)
static void cpu_rms_norm_gated(const float *x, const float *z, const uint16_t *w_bf16,
                                float *out, int dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(sum_sq / dim + eps);
    for (int i = 0; i < dim; i++) {
        float w = bf16_to_f32(w_bf16[i]);
        float silu_z = z[i] / (1.0f + expf(-z[i]));
        out[i] = x[i] * inv_rms * w * silu_z;
    }
}

static int linear_attn_bypass = 0;  // set to 1 to skip linear attention (identity)
static int gpu_linear_attn_enabled = 1;  // fused GPU delta-net path (can disable via --cpu-linear)

__attribute__((unused))
static void linear_attention_forward(
    WeightFile *wf,
    int layer_idx,
    float *hidden,           // [cfg.hidden_dim] in/out
    LinearAttnState *state
) {
    // If bypass is enabled, just pass through (identity)
    if (linear_attn_bypass) {
        (void)wf; (void)layer_idx; (void)state;
        return;
    }

    static int la_debug_count = 0;
    la_debug_count++;
    int la_debug = 0;  // set to (la_debug_count <= N) to enable debug

    if (la_debug) {
        fprintf(stderr, "[LA-DBG] layer=%d hidden_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                layer_idx, vec_rms(hidden, cfg.hidden_dim),
                hidden[0], hidden[1], hidden[2], hidden[3], hidden[4]);
    }

    char name[256];
    float *normed = malloc(cfg.hidden_dim * sizeof(float));
    float *residual = malloc(cfg.hidden_dim * sizeof(float));
    if (!normed || !residual) {
        fprintf(stderr, "ERROR: linear_attention_forward alloc failed\n");
        free(normed); free(residual); return;
    }
    cpu_vec_copy(residual, hidden, cfg.hidden_dim);

    // ---- Input LayerNorm ----
    snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", layer_idx);
    uint16_t *norm_w = get_tensor_ptr(wf, name);
    cpu_rms_norm(hidden, norm_w, normed, cfg.hidden_dim, cfg.rms_norm_eps);

    // ---- Batch QKV + Z + B + A projections (4 matmuls, 1 command buffer) ----
    int qkv_dim = cfg.linear_conv_dim;  // 12288
    float *qkv = calloc(qkv_dim, sizeof(float));
    int z_dim = cfg.linear_total_value;  // 8192
    float *z = calloc(z_dim, sizeof(float));
    float *beta = calloc(cfg.linear_num_v_heads, sizeof(float));
    float *alpha = calloc(cfg.linear_num_v_heads, sizeof(float));

    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.weight", layer_idx);
    uint32_t *qkv_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.scales", layer_idx);
    uint16_t *qkv_s = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.biases", layer_idx);
    uint16_t *qkv_b = get_tensor_ptr(wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.weight", layer_idx);
    uint32_t *z_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.scales", layer_idx);
    uint16_t *z_s = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.biases", layer_idx);
    uint16_t *z_b = get_tensor_ptr(wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.weight", layer_idx);
    uint32_t *b_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.scales", layer_idx);
    uint16_t *b_s = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.biases", layer_idx);
    uint16_t *b_b = get_tensor_ptr(wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.weight", layer_idx);
    uint32_t *a_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.scales", layer_idx);
    uint16_t *a_s = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.biases", layer_idx);
    uint16_t *a_b = get_tensor_ptr(wf, name);

    if (qkv_w && qkv_s && qkv_b && z_w && z_s && z_b &&
        b_w && b_s && b_b && a_w && a_s && a_b) {
        BatchMatvecSpec la_specs[4] = {
            { qkv_w, qkv_s, qkv_b, qkv,   (uint32_t)qkv_dim,         cfg.hidden_dim, cfg.group_size, 0 },
            { z_w,   z_s,   z_b,   z,      (uint32_t)z_dim,           cfg.hidden_dim, cfg.group_size, 1 },
            { b_w,   b_s,   b_b,   beta,   (uint32_t)cfg.linear_num_v_heads, cfg.hidden_dim, cfg.group_size, 2 },
            { a_w,   a_s,   a_b,   alpha,  (uint32_t)cfg.linear_num_v_heads, cfg.hidden_dim, cfg.group_size, 3 },
        };
        fast_batch_matvec(normed, cfg.hidden_dim, la_specs, 4);
    }

    // ---- Conv1d step ----
    // conv_state holds last (kernel_size-1) inputs for each of the conv_dim channels
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.conv1d.weight", layer_idx);
    uint16_t *conv_w = get_tensor_ptr(wf, name);

    float *conv_out = calloc(qkv_dim, sizeof(float));
    if (conv_w) {
        cpu_conv1d_step(state->conv_state, qkv, conv_w, conv_out,
                        qkv_dim, cfg.conv_kernel_size);
    }

    // Update conv state: shift left, append new input
    memmove(state->conv_state, state->conv_state + qkv_dim,
            (cfg.conv_kernel_size - 2) * qkv_dim * sizeof(float));
    memcpy(state->conv_state + (cfg.conv_kernel_size - 2) * qkv_dim, qkv,
           qkv_dim * sizeof(float));

    // ---- Split conv_out into q, k, v ----
    // q: [num_k_heads * head_k_dim] = [2048]
    // k: [num_k_heads * head_k_dim] = [2048]
    // v: [num_v_heads * head_v_dim] = [8192]
    float *lin_q = conv_out;  // first cfg.linear_total_key elements
    float *lin_k = conv_out + cfg.linear_total_key;  // next cfg.linear_total_key
    float *lin_v = conv_out + 2 * cfg.linear_total_key;  // rest = cfg.linear_total_value

    // ---- RMS normalize q and k (bare, no weights) ----
    // q: scale = key_dim^(-0.5), normalize per head then scale by key_dim^(-1.0)
    // Actually from the code:
    //   inv_scale = k.shape[-1] ** -0.5 = head_k_dim^(-0.5) = 128^(-0.5)
    //   q = (inv_scale**2) * rms_norm(q) = (1/128) * rms_norm(q)
    //   k = inv_scale * rms_norm(k) = (1/sqrt(128)) * rms_norm(k)
    float inv_scale = 1.0f / sqrtf((float)cfg.linear_key_dim);

    for (int h = 0; h < cfg.linear_num_k_heads; h++) {
        float *qh = lin_q + h * cfg.linear_key_dim;
        cpu_rms_norm_bare(qh, qh, cfg.linear_key_dim, 1e-6f);
        float q_scale = inv_scale * inv_scale;  // inv_scale^2 = 1/head_k_dim
        for (int d = 0; d < cfg.linear_key_dim; d++) qh[d] *= q_scale;
    }
    for (int h = 0; h < cfg.linear_num_k_heads; h++) {
        float *kh = lin_k + h * cfg.linear_key_dim;
        cpu_rms_norm_bare(kh, kh, cfg.linear_key_dim, 1e-6f);
        for (int d = 0; d < cfg.linear_key_dim; d++) kh[d] *= inv_scale;
    }

    // ---- Gated delta net recurrence ----
    // From gated_delta.py:
    //   g = exp(-exp(A_log) * softplus(a + dt_bias))   -- per-head decay
    //   beta_gate = sigmoid(b)                          -- per-head beta (NO dt_bias)
    //   For each v_head:
    //     state = state * g                             -- decay
    //     kv_mem = sum(state * k, axis=key_dim)         -- predict v from state
    //     delta = (v - kv_mem) * beta_gate              -- error signal
    //     state = state + outer(delta, k)               -- update state
    //     output = sum(state * q, axis=key_dim)         -- read from state

    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.A_log", layer_idx);
    float *A_log = get_tensor_ptr(wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.dt_bias", layer_idx);
    uint16_t *dt_bias_bf16 = get_tensor_ptr(wf, name);

    float *out_values = calloc(cfg.linear_total_value, sizeof(float));  // [num_v_heads * head_v_dim]

    int k_heads_per_v = cfg.linear_num_v_heads / cfg.linear_num_k_heads;  // 64/16 = 4

    // Precompute per-head decay (g) and beta
    float g_decay[cfg.linear_num_v_heads];
    float beta_gate[cfg.linear_num_v_heads];
    for (int vh = 0; vh < cfg.linear_num_v_heads; vh++) {
        // g = exp(-exp(A_log) * softplus(a + dt_bias))
        float a_val = alpha[vh];
        float dt_b = dt_bias_bf16 ? bf16_to_f32(dt_bias_bf16[vh]) : 0.0f;
        float A_val = A_log ? expf(A_log[vh]) : 1.0f;
        float softplus_val = logf(1.0f + expf(a_val + dt_b));  // softplus(a + dt_bias)
        g_decay[vh] = expf(-A_val * softplus_val);

        // beta = sigmoid(b)  (just b, NO dt_bias)
        beta_gate[vh] = cpu_sigmoid(beta[vh]);
    }

    for (int vh = 0; vh < cfg.linear_num_v_heads; vh++) {
        int kh = vh / k_heads_per_v;  // which k head this v head maps to

        float g = g_decay[vh];
        float b_gate = beta_gate[vh];

        // state is [head_v_dim, head_k_dim]
        float *S = state->ssm_state + vh * cfg.linear_value_dim * cfg.linear_key_dim;
        float *v_h = lin_v + vh * cfg.linear_value_dim;
        float *k_h = lin_k + kh * cfg.linear_key_dim;

        // Step 1: Decay state
        for (int vi = 0; vi < cfg.linear_value_dim; vi++) {
            for (int ki = 0; ki < cfg.linear_key_dim; ki++) {
                S[vi * cfg.linear_key_dim + ki] *= g;
            }
        }

        // Step 2: Compute kv_mem[vi] = sum_ki(S[vi,ki] * k[ki])
        // Then delta[vi] = (v[vi] - kv_mem[vi]) * beta
        // Then state[vi,ki] += k[ki] * delta[vi]
        for (int vi = 0; vi < cfg.linear_value_dim; vi++) {
            float kv_mem = 0.0f;
            for (int ki = 0; ki < cfg.linear_key_dim; ki++) {
                kv_mem += S[vi * cfg.linear_key_dim + ki] * k_h[ki];
            }
            float delta = (v_h[vi] - kv_mem) * b_gate;
            for (int ki = 0; ki < cfg.linear_key_dim; ki++) {
                S[vi * cfg.linear_key_dim + ki] += k_h[ki] * delta;
            }
        }

        // Step 3: Output: y[vi] = sum_ki(S[vi,ki] * q[ki])
        float *q_h = lin_q + kh * cfg.linear_key_dim;
        float *o_h = out_values + vh * cfg.linear_value_dim;
        for (int vi = 0; vi < cfg.linear_value_dim; vi++) {
            float sum = 0.0f;
            for (int ki = 0; ki < cfg.linear_key_dim; ki++) {
                sum += S[vi * cfg.linear_key_dim + ki] * q_h[ki];
            }
            o_h[vi] = sum;
        }
    }

    // ---- RMSNormGated: out = rms_norm(out_values_per_head) * silu(z_per_head) * weight ----
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.norm.weight", layer_idx);
    uint16_t *gated_norm_w = get_tensor_ptr(wf, name);

    float *gated_out = calloc(cfg.linear_total_value, sizeof(float));
    for (int vh = 0; vh < cfg.linear_num_v_heads; vh++) {
        float *oh = out_values + vh * cfg.linear_value_dim;
        float *zh = z + vh * cfg.linear_value_dim;
        float *gh = gated_out + vh * cfg.linear_value_dim;
        if (gated_norm_w) {
            cpu_rms_norm_gated(oh, zh, gated_norm_w, gh, cfg.linear_value_dim, cfg.rms_norm_eps);
        } else {
            memcpy(gh, oh, cfg.linear_value_dim * sizeof(float));
        }
    }

    // ---- Output projection: [value_dim=8192] -> [hidden_dim=4096] ----
    float *attn_out = calloc(cfg.hidden_dim, sizeof(float));
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.weight", layer_idx);
    uint32_t *out_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.scales", layer_idx);
    uint16_t *out_s = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.biases", layer_idx);
    uint16_t *out_b = get_tensor_ptr(wf, name);
    if (out_w && out_s && out_b) {
        fast_dequant_matvec(out_w, out_s, out_b, gated_out, attn_out, cfg.hidden_dim,
                            cfg.linear_total_value, cfg.group_size);
    }

    // ---- Residual ----
    for (int i = 0; i < cfg.hidden_dim; i++) {
        hidden[i] = residual[i] + attn_out[i];
    }

    if (la_debug) {
        fprintf(stderr, "[LA-DBG] AFTER layer=%d out_proj_rms=%.6f gated_rms=%.6f hidden_rms=%.6f\n",
                layer_idx, vec_rms(attn_out, cfg.hidden_dim),
                vec_rms(gated_out, cfg.linear_total_value),
                vec_rms(hidden, cfg.hidden_dim));
    }

    free(normed);
    free(residual);
    free(qkv);
    free(z);
    free(beta);
    free(alpha);
    free(conv_out);
    free(out_values);
    free(gated_out);
    free(attn_out);
}

// ============================================================================
// MoE forward (routing + expert computation + shared expert)
// ============================================================================

static int moe_debug_count = 0;

__attribute__((unused))
static void moe_forward(
    WeightFile *wf,
    int layer_idx,
    float *hidden,         // [cfg.hidden_dim] in/out
    const char *model_path __attribute__((unused)),
    int K,                 // number of active experts (e.g. 4)
    int packed_fd          // fd for this layer's packed expert file (-1 if not available)
) {
    moe_debug_count++;
    int moe_debug = 0;  // set to (moe_debug_count <= N) to enable debug
    int moe_dump = 0;

    char name[256];
    float *h_post = malloc(cfg.hidden_dim * sizeof(float));
    float *h_mid = malloc(cfg.hidden_dim * sizeof(float));
    if (!h_post || !h_mid) {
        fprintf(stderr, "ERROR: moe_forward alloc failed\n");
        free(h_post); free(h_mid); return;
    }
    cpu_vec_copy(h_mid, hidden, cfg.hidden_dim);

    // ---- Post-attention LayerNorm ----
    snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", layer_idx);
    uint16_t *norm_w = get_tensor_ptr(wf, name);
    cpu_rms_norm(hidden, norm_w, h_post, cfg.hidden_dim, cfg.rms_norm_eps);

    // ---- Batch routing gate + shared expert gate/up + shared_expert_gate (4 matmuls, 1 commit) ----
    float *gate_scores = calloc(cfg.num_experts, sizeof(float));
    float *shared_gate = calloc(cfg.shared_intermediate, sizeof(float));
    float *shared_up = calloc(cfg.shared_intermediate, sizeof(float));
    float shared_gate_score = 0.0f;

    snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.weight", layer_idx);
    uint32_t *gate_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.scales", layer_idx);
    uint16_t *gate_s = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.biases", layer_idx);
    uint16_t *gate_b = get_tensor_ptr(wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.weight", layer_idx);
    uint32_t *sgw = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.scales", layer_idx);
    uint16_t *sgs = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.biases", layer_idx);
    uint16_t *sgb = get_tensor_ptr(wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.weight", layer_idx);
    uint32_t *suw = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.scales", layer_idx);
    uint16_t *sus = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.biases", layer_idx);
    uint16_t *sub = get_tensor_ptr(wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.weight", layer_idx);
    uint32_t *seg_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.scales", layer_idx);
    uint16_t *seg_s = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.biases", layer_idx);
    uint16_t *seg_b = get_tensor_ptr(wf, name);

    // All 4 matmuls share h_post as input -- batch into one command buffer
    if (gate_w && gate_s && gate_b && sgw && sgs && sgb &&
        suw && sus && sub && seg_w && seg_s && seg_b) {
        BatchMatvecSpec moe_specs[4] = {
            { gate_w, gate_s, gate_b, gate_scores,        (uint32_t)cfg.num_experts,        cfg.hidden_dim, cfg.group_size, 0 },
            { sgw,    sgs,    sgb,    shared_gate,         (uint32_t)cfg.shared_intermediate, cfg.hidden_dim, cfg.group_size, 1 },
            { suw,    sus,    sub,    shared_up,           (uint32_t)cfg.shared_intermediate, cfg.hidden_dim, cfg.group_size, 2 },
            { seg_w,  seg_s,  seg_b,  &shared_gate_score,  1,                            cfg.hidden_dim, cfg.group_size, 3 },
        };
        fast_batch_matvec(h_post, cfg.hidden_dim, moe_specs, 4);
    }

    // Softmax routing scores
    cpu_softmax(gate_scores, cfg.num_experts);

    // Top-K expert selection
    int expert_indices[64];
    float expert_weights[64];
    cpu_topk(gate_scores, cfg.num_experts, K, expert_indices, expert_weights);
    cpu_normalize_weights(expert_weights, K);

    if (moe_dump) {
        fprintf(stderr, "[MOE-DUMP] routing: K=%d experts=[", K);
        for (int k = 0; k < K; k++) fprintf(stderr, "%d(%.4f)%s", expert_indices[k], expert_weights[k], k<K-1?",":"");
        fprintf(stderr, "]\n");
    }

    // ---- Routed expert computation ----
    float *moe_out = calloc(cfg.hidden_dim, sizeof(float));

    if (packed_fd >= 0) {
        float *expert_out = malloc(cfg.hidden_dim * sizeof(float));
        if (!expert_out) {
            fprintf(stderr, "ERROR: moe_forward expert_out alloc failed\n");
            free(h_post); free(h_mid); free(gate_scores); free(moe_out);
            free(shared_gate); free(shared_up); return;
        }

        for (int k = 0; k < K; k++) {
            int eidx = expert_indices[k];
            off_t expert_offset; size_t esz;
            expert_offset_size(layer_idx, eidx, &expert_offset, &esz);

            if (g_metal && g_metal->buf_expert_data) {
                // GPU path: pread directly into Metal buffer, run gate+up+swiglu+down on GPU
                void *expert_buf_ptr = [g_metal->buf_expert_data contents];
                ssize_t nread = pread(packed_fd, expert_buf_ptr, esz, expert_offset);
                if (nread != (ssize_t)esz) {
                    fprintf(stderr, "WARNING: layer %d expert %d pread: %zd/%zu\n",
                            layer_idx, eidx, nread, esz);
                    continue;
                }

                gpu_expert_forward(g_metal, expert_buf_ptr, h_post, expert_out, 1 /*already in buffer*/);
            } else {
                // CPU fallback
                void *expert_data = malloc(esz);
                if (!expert_data) {
                    fprintf(stderr, "WARNING: layer %d expert %d malloc(%zu) failed, skipping\n",
                            layer_idx, eidx, esz);
                    continue;
                }
                ssize_t nread = pread(packed_fd, expert_data, esz, expert_offset);
                if (nread != (ssize_t)esz) {
                    fprintf(stderr, "WARNING: layer %d expert %d pread: %zd/%zu\n",
                            layer_idx, eidx, nread, esz);
                    free(expert_data);
                    continue;
                }

                uint32_t *gw = (uint32_t *)expert_data;
                uint16_t *gs_p = (uint16_t *)((char *)expert_data + (g_use_2bit ? cfg.gate_s_off_2 : cfg.gate_s_off_4));
                uint16_t *gb_p = (uint16_t *)((char *)expert_data + (g_use_2bit ? cfg.gate_b_off_2 : cfg.gate_b_off_4));
                uint32_t *uw = (uint32_t *)((char *)expert_data + (g_use_2bit ? cfg.up_w_off_2 : cfg.up_w_off_4));
                uint16_t *us_p = (uint16_t *)((char *)expert_data + (g_use_2bit ? cfg.up_s_off_2 : cfg.up_s_off_4));
                uint16_t *ub_p = (uint16_t *)((char *)expert_data + (g_use_2bit ? cfg.up_b_off_2 : cfg.up_b_off_4));
                uint32_t *dw = (uint32_t *)((char *)expert_data + (g_use_2bit ? cfg.down_w_off_2 : cfg.down_w_off_4));
                uint16_t *ds_p = (uint16_t *)((char *)expert_data + (g_use_2bit ? cfg.down_s_off_2 : cfg.down_s_off_4));
                uint16_t *db_p = (uint16_t *)((char *)expert_data + (g_use_2bit ? cfg.down_b_off_2 : cfg.down_b_off_4));

                float *gate_proj_out = malloc(cfg.moe_intermediate * sizeof(float));
                float *up_proj_out = malloc(cfg.moe_intermediate * sizeof(float));
                float *act_out = malloc(cfg.moe_intermediate * sizeof(float));
                if (!gate_proj_out || !up_proj_out || !act_out) {
                    fprintf(stderr, "WARNING: expert MLP alloc failed, skipping expert %d\n", eidx);
                    free(gate_proj_out); free(up_proj_out); free(act_out);
                    free(expert_data); continue;
                }

                cpu_dequant_matvec(gw, gs_p, gb_p, h_post, gate_proj_out,
                                   cfg.moe_intermediate, cfg.hidden_dim, cfg.group_size);
                cpu_dequant_matvec(uw, us_p, ub_p, h_post, up_proj_out,
                                   cfg.moe_intermediate, cfg.hidden_dim, cfg.group_size);
                cpu_swiglu(gate_proj_out, up_proj_out, act_out, cfg.moe_intermediate);
                cpu_dequant_matvec(dw, ds_p, db_p, act_out, expert_out,
                                   cfg.hidden_dim, cfg.moe_intermediate, cfg.group_size);

                free(gate_proj_out);
                free(up_proj_out);
                free(act_out);
                free(expert_data);
            }

            // Accumulate weighted
            if (moe_dump) {
                fprintf(stderr, "[MOE-DUMP] expert[%d] out_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        eidx, vec_rms(expert_out, cfg.hidden_dim),
                        expert_out[0], expert_out[1], expert_out[2], expert_out[3], expert_out[4]);
            }
            cpu_vec_madd(moe_out, expert_out, expert_weights[k], cfg.hidden_dim);
        }

        free(expert_out);
    }

    // ---- Shared expert SwiGLU (gate_proj + up_proj already computed above) ----
    float *shared_out = calloc(cfg.hidden_dim, sizeof(float));
    float *shared_act = calloc(cfg.shared_intermediate, sizeof(float));
    cpu_swiglu(shared_gate, shared_up, shared_act, cfg.shared_intermediate);

    if (moe_dump) {
        fprintf(stderr, "[MOE-DUMP] layer=%d h_post_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                layer_idx, vec_rms(h_post, cfg.hidden_dim), h_post[0], h_post[1], h_post[2], h_post[3], h_post[4]);
        fprintf(stderr, "[MOE-DUMP] gate_proj_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(shared_gate, cfg.shared_intermediate),
                shared_gate[0], shared_gate[1], shared_gate[2], shared_gate[3], shared_gate[4]);
        fprintf(stderr, "[MOE-DUMP] up_proj_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(shared_up, cfg.shared_intermediate),
                shared_up[0], shared_up[1], shared_up[2], shared_up[3], shared_up[4]);
        fprintf(stderr, "[MOE-DUMP] swiglu_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(shared_act, cfg.shared_intermediate),
                shared_act[0], shared_act[1], shared_act[2], shared_act[3], shared_act[4]);
    }

    // shared_expert down_proj (separate dispatch — different input than h_post)
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.weight", layer_idx);
    uint32_t *sdw = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.scales", layer_idx);
    uint16_t *sds = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.biases", layer_idx);
    uint16_t *sdb = get_tensor_ptr(wf, name);
    if (sdw && sds && sdb) {
        fast_dequant_matvec(sdw, sds, sdb, shared_act, shared_out, cfg.hidden_dim,
                            cfg.shared_intermediate, cfg.group_size);
    }

    // ---- Shared expert gate (sigmoid) -- already computed above ----
    float shared_weight = cpu_sigmoid(shared_gate_score);

    // Scale shared expert output
    for (int i = 0; i < cfg.hidden_dim; i++) {
        shared_out[i] *= shared_weight;
    }

    // ---- Combine: hidden = h_mid + moe_out + shared_out ----
    for (int i = 0; i < cfg.hidden_dim; i++) {
        hidden[i] = h_mid[i] + moe_out[i] + shared_out[i];
    }

    if (moe_debug) {
        fprintf(stderr, "[MOE-DBG] layer=%d h_mid_rms=%.4f moe_rms=%.4f shared_rms=%.4f shared_gate=%.4f hidden_rms=%.4f\n",
                layer_idx, vec_rms(h_mid, cfg.hidden_dim), vec_rms(moe_out, cfg.hidden_dim),
                vec_rms(shared_out, cfg.hidden_dim), shared_weight,
                vec_rms(hidden, cfg.hidden_dim));
    }

    free(h_post);
    free(h_mid);
    free(gate_scores);
    free(moe_out);
    free(shared_out);
    free(shared_gate);
    free(shared_up);
    free(shared_act);
}

// ============================================================================
// Embedding lookup (4-bit quantized)
// ============================================================================

static void embed_lookup(WeightFile *wf, int token_id, float *out) {
    // Embedding: weight[vocab_size, hidden_dim/8] (U32), scales[vocab_size, groups], biases[vocab_size, groups]
    // For embedding lookup, we just need one row.
    // But the embedding is quantized: each row has hidden_dim/8 uint32 values (packed 4-bit)
    // plus scales and biases per group

    TensorInfo *w_info = get_tensor_info(wf, "model.embed_tokens.weight");
    TensorInfo *s_info = get_tensor_info(wf, "model.embed_tokens.scales");
    TensorInfo *b_info = get_tensor_info(wf, "model.embed_tokens.biases");

    if (!w_info || !s_info || !b_info) {
        fprintf(stderr, "ERROR: embedding tensors not found\n");
        memset(out, 0, cfg.hidden_dim * sizeof(float));
        return;
    }

    // w shape: [248320, 512] U32 -> each row has 512 uint32 = 4096 packed 4-bit values
    int packed_cols = w_info->shape[1];  // 512
    int num_groups = s_info->shape[1];   // 64

    uint32_t *W = (uint32_t *)((char *)wf->data + w_info->offset);
    uint16_t *S = (uint16_t *)((char *)wf->data + s_info->offset);
    uint16_t *B = (uint16_t *)((char *)wf->data + b_info->offset);

    const uint32_t *w_row = W + (size_t)token_id * packed_cols;
    const uint16_t *s_row = S + (size_t)token_id * num_groups;
    const uint16_t *b_row = B + (size_t)token_id * num_groups;

    int group_size = cfg.hidden_dim / num_groups;  // 4096/64 = 64
    int packed_per_group = group_size / 8;     // 8

    for (int g = 0; g < num_groups; g++) {
        float scale = bf16_to_f32(s_row[g]);
        float bias = bf16_to_f32(b_row[g]);

        for (int p = 0; p < packed_per_group; p++) {
            uint32_t packed = w_row[g * packed_per_group + p];
            int base = g * group_size + p * 8;

            for (int n = 0; n < 8; n++) {
                uint32_t nibble = (packed >> (n * 4)) & 0xF;
                out[base + n] = (float)nibble * scale + bias;
            }
        }
    }
}

// ============================================================================
// LM head (logits projection)
// ============================================================================

static void lm_head_forward(WeightFile *wf, const float *hidden, float *logits) {
    // lm_head: [hidden_dim=4096] -> [vocab_size=248320]
    // This is a HUGE matmul. For 248320 output dims, it will be slow on CPU.
    // Optimization: only compute top candidates

    TensorInfo *w_info = get_tensor_info(wf, "lm_head.weight");
    TensorInfo *s_info = get_tensor_info(wf, "lm_head.scales");
    TensorInfo *b_info = get_tensor_info(wf, "lm_head.biases");

    if (!w_info || !s_info || !b_info) {
        fprintf(stderr, "ERROR: lm_head tensors not found\n");
        return;
    }

    uint32_t *W = (uint32_t *)((char *)wf->data + w_info->offset);
    uint16_t *S = (uint16_t *)((char *)wf->data + s_info->offset);
    uint16_t *B = (uint16_t *)((char *)wf->data + b_info->offset);

    // Full matmul — use GPU if available (248320 output rows!)
    fast_dequant_matvec(W, S, B, hidden, logits, cfg.vocab_size, cfg.hidden_dim, cfg.group_size);
}

// ============================================================================
// Per-layer weight pointer cache — built once, eliminates 40+ snprintf+lookup
// per layer per token. With 40 layers and 15 tokens = 24,000 lookups saved.
// ============================================================================

typedef struct {
    // Input/post-attention layer norms
    uint16_t *input_norm_w;
    uint16_t *post_attn_norm_w;

    // Full attention weights (non-NULL only for full attention layers)
    uint32_t *q_w; uint16_t *q_s, *q_b;
    uint32_t *k_w; uint16_t *k_s, *k_b;
    uint32_t *v_w; uint16_t *v_s, *v_b;
    uint32_t *o_w; uint16_t *o_s, *o_b;
    uint16_t *q_norm_w, *k_norm_w;

    // Linear attention weights (non-NULL only for linear attention layers)
    uint32_t *qkv_w; uint16_t *qkv_s, *qkv_b;
    uint32_t *z_w;   uint16_t *z_s, *z_b;
    uint32_t *b_w;   uint16_t *b_s, *b_b;
    uint32_t *a_w;   uint16_t *a_s, *a_b;
    uint16_t *conv1d_w;
    float *A_log;
    uint16_t *dt_bias;
    uint16_t *gated_norm_w;
    uint32_t *out_proj_w; uint16_t *out_proj_s, *out_proj_b;

    // MoE routing + shared expert weights
    uint32_t *gate_w; uint16_t *gate_s, *gate_b;
    uint32_t *sg_w;   uint16_t *sg_s, *sg_b;   // shared gate_proj
    uint32_t *su_w;   uint16_t *su_s, *su_b;   // shared up_proj
    uint32_t *sd_w;   uint16_t *sd_s, *sd_b;   // shared down_proj
    uint32_t *seg_w;  uint16_t *seg_s, *seg_b; // shared_expert_gate
} LayerWeightCache;

static LayerWeightCache *layer_cache = NULL;
static int layer_cache_built = 0;

// Allocate all dynamic tracking arrays (must be called after load_model_config)
static void alloc_tracking_arrays(void) {
    int nl = cfg.num_layers;
    int ne = cfg.num_experts;
    int seen_bytes_per_layer = (ne + 7) / 8;

    g_expert_freq            = calloc(nl * ne, sizeof(int));
    g_expert_seen            = calloc(nl * seen_bytes_per_layer, sizeof(uint8_t));
    g_lz4_index              = calloc(nl, sizeof(void *));
    g_cache_seen             = calloc(nl * ne, sizeof(uint8_t));
    g_cache_last_touch_token = calloc(nl * ne, sizeof(uint64_t));
    g_cache_last_evict_token = calloc(nl * ne, sizeof(uint64_t));
    g_pred_experts           = calloc(nl * MAX_K, sizeof(int));
    g_pred_count             = calloc(nl, sizeof(int));
    layer_cache              = calloc(nl, sizeof(LayerWeightCache));
}

static void build_layer_cache(WeightFile *wf) {
    if (layer_cache_built) return;
    char name[256];

    for (int i = 0; i < cfg.num_layers; i++) {
        LayerWeightCache *lc = &layer_cache[i];
        int is_full = cfg.is_full_attn[i];

        // Norms
        snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", i);
        lc->input_norm_w = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", i);
        lc->post_attn_norm_w = get_tensor_ptr(wf, name);

        if (is_full) {
            // Full attention
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", i);
            lc->q_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.scales", i);
            lc->q_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.biases", i);
            lc->q_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", i);
            lc->k_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.scales", i);
            lc->k_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.biases", i);
            lc->k_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", i);
            lc->v_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.scales", i);
            lc->v_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.biases", i);
            lc->v_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", i);
            lc->o_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.scales", i);
            lc->o_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.biases", i);
            lc->o_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_norm.weight", i);
            lc->q_norm_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_norm.weight", i);
            lc->k_norm_w = get_tensor_ptr(wf, name);
        } else {
            // Linear attention
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.weight", i);
            lc->qkv_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.scales", i);
            lc->qkv_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.biases", i);
            lc->qkv_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.weight", i);
            lc->z_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.scales", i);
            lc->z_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.biases", i);
            lc->z_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.weight", i);
            lc->b_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.scales", i);
            lc->b_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.biases", i);
            lc->b_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.weight", i);
            lc->a_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.scales", i);
            lc->a_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.biases", i);
            lc->a_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.conv1d.weight", i);
            lc->conv1d_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.A_log", i);
            lc->A_log = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.dt_bias", i);
            lc->dt_bias = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.norm.weight", i);
            lc->gated_norm_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.weight", i);
            lc->out_proj_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.scales", i);
            lc->out_proj_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.biases", i);
            lc->out_proj_b = get_tensor_ptr(wf, name);
        }

        // MoE weights (same for all layers)
        snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.weight", i);
        lc->gate_w = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.scales", i);
        lc->gate_s = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.biases", i);
        lc->gate_b = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.weight", i);
        lc->sg_w = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.scales", i);
        lc->sg_s = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.biases", i);
        lc->sg_b = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.weight", i);
        lc->su_w = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.scales", i);
        lc->su_s = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.biases", i);
        lc->su_b = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.weight", i);
        lc->sd_w = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.scales", i);
        lc->sd_s = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.biases", i);
        lc->sd_b = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.weight", i);
        lc->seg_w = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.scales", i);
        lc->seg_s = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.biases", i);
        lc->seg_b = get_tensor_ptr(wf, name);
    }

    layer_cache_built = 1;
    printf("[cache] Pre-computed weight pointers for %d layers\n", cfg.num_layers);
}

// ============================================================================
// Deferred expert state: holds state for async GPU expert compute.
// GPU experts are submitted async (commit without wait), and the wait+combine
// happens at the start of the NEXT layer. This overlaps ~1ms of GPU expert
// compute with the next layer's attention+routing CPU/GPU work.
// ============================================================================

typedef struct {
    int active;                         // 1 if there's a deferred GPU expert to wait for
    int gpu_combined;                   // 1 if CMD3 includes combine+residual+norm on GPU
                                        // (next layer can skip deferred_wait+finalize+input_norm
                                        //  and submit CMD1 immediately -- buf_input is ready)
    id<MTLCommandBuffer> cmd_experts;   // the async command buffer (committed but not waited)
    float expert_weights[MAX_K];        // routing weights for weighted accumulation
    int valid[MAX_K];                   // which experts loaded successfully
    int actual_K;                       // number of experts
    float *h_mid;                           // [hidden_dim] saved h_mid for final combine
    float shared_gate_score;            // saved shared expert gate score
    float *hidden;                      // pointer to hidden state (for writing final result)
    int layer_idx;                      // which layer produced this deferred state
} DeferredExpertState;

static DeferredExpertState g_deferred = { .active = 0, .h_mid = NULL };

// Wait for the deferred GPU expert command buffer to complete.
// Split from finalize so timing can be measured independently.
static void wait_deferred_experts_gpu(void) {
    if (!g_deferred.active) return;
    [g_deferred.cmd_experts waitUntilCompleted];
}

// CPU readback + accumulate + combine after GPU is done.
// Must be called after wait_deferred_experts_gpu().
// When gpu_combined=1, the GPU already computed the combine+residual+norm
// in CMD3, so we just need to read back the hidden state from buf_moe_hidden.
static void finalize_deferred_experts(void) {
    if (!g_deferred.active) return;

    if (g_deferred.gpu_combined) {
        // GPU-side combine: hidden state is already in buf_moe_hidden.
        // buf_input already has the normalized input for the next layer's CMD1.
        // Just read back hidden (needed for the residual connection in future layers).
        memcpy(g_deferred.hidden, [g_metal->buf_moe_hidden contents],
               cfg.hidden_dim * sizeof(float));
    } else {
        // CPU-side combine (original path)
        // Read back and accumulate routed expert outputs
        float moe_out[cfg.hidden_dim];
        memset(moe_out, 0, sizeof(moe_out));
        for (int k = 0; k < g_deferred.actual_K; k++) {
            if (!g_deferred.valid[k]) continue;
            float *expert_result = (float *)[g_metal->buf_multi_expert_out[k] contents];
            cpu_vec_madd(moe_out, expert_result, g_deferred.expert_weights[k], cfg.hidden_dim);
        }

        // Read shared expert result
        float shared_out[cfg.hidden_dim];
        memcpy(shared_out, [g_metal->buf_shared_out contents], cfg.hidden_dim * sizeof(float));

        // Apply shared expert gate
        float shared_weight = cpu_sigmoid(g_deferred.shared_gate_score);
        for (int i = 0; i < cfg.hidden_dim; i++) {
            shared_out[i] *= shared_weight;
        }

        // Final combine: hidden = h_mid + moe_out + shared_out
        for (int i = 0; i < cfg.hidden_dim; i++) {
            g_deferred.hidden[i] = g_deferred.h_mid[i] + moe_out[i] + shared_out[i];
        }
    }

    g_deferred.active = 0;
    g_deferred.gpu_combined = 0;
    g_deferred.cmd_experts = nil;
}

// Complete the deferred GPU expert compute: wait for GPU, read back, accumulate, combine.
// Must be called before the next layer modifies static scratch buffers.
static void complete_deferred_experts(void) {
    wait_deferred_experts_gpu();
    finalize_deferred_experts();
}

// Discard the deferred GPU expert result: wait for GPU to finish (for buffer safety)
// but skip the CPU readback/combine. Used during prefill for intermediate tokens
// where the hidden state will be immediately overwritten by the next token's embedding.
// This saves ~0.1-0.2ms per prefill token (avoids unnecessary memcpy + combine work).
static void discard_deferred_experts(void) {
    wait_deferred_experts_gpu();
    // Clear deferred state without reading back results
    if (g_deferred.active) {
        g_deferred.active = 0;
        g_deferred.gpu_combined = 0;
        g_deferred.cmd_experts = nil;
    }
}

// ============================================================================
// Fused layer forward: GPU/CPU overlap + deferred expert pipeline
//
// Pipeline per layer (3 cmd buffers, GPU-side combine in CMD3):
//
//   FAST PATH (when previous CMD3 did GPU-side combine):
//     CMD1: submit immediately (buf_input already populated by CMD3(N-1))
//     WAIT: CMD1 complete (implies CMD3(N-1) also done, queue is serial)
//     CPU:  finalize deferred (read back hidden from buf_moe_hidden)
//
//   SLOW PATH (first layer, or last layer's CMD3 without GPU combine):
//     [DEFERRED] Wait for PREVIOUS layer's CMD3 (if any) + CPU combine
//     CPU:  input_norm(hidden) -> normed -> buf_input
//     CMD1: attention projections (commit)
//     WAIT: CMD1 complete
//
//   Then (both paths):
//     CPU:  attention compute (RoPE/softmax/delta-net)
//     CMD2: o_proj + residual + norm + routing + shared expert projs (8 encoders, 1 commit)
//     WAIT: CMD2 complete
//     CPU:  softmax + top-K routing
//     I/O:  parallel pread K experts (4 pthreads)
//     CMD3: K expert forwards + shared SwiGLU + shared down
//           + moe_combine_residual + rms_norm -> buf_input (ASYNC commit, NO wait)
//     RETURN: GPU experts + combine running async
//
// GPU-side combine eliminates the 0.83ms deferred_wait + CPU combine + input_norm
// at the start of each layer, allowing CMD1 to be submitted immediately.
//
// Key optimizations:
//   1. Parallel pread (4 threads) instead of sequential: ~4x I/O speedup
//   2. o_proj fused into CMD2 with routing (saves 1 commit+wait)
//   3. Deferred CMD3 (expert GPU compute overlapped with next layer)
//   4. GPU-side combine in CMD3 (eliminates CPU deferred_wait + combine + norm)
// ============================================================================

// Static scratch buffers — allocated once, reused across all 40 layers per token.
// Eliminates ~300 malloc/free per token. Pre-allocated at model load for OOM safety.
static float *s_normed    = NULL;   // [cfg.hidden_dim]
static float *s_residual  = NULL;   // [cfg.hidden_dim]
static float *s_attn_proj = NULL;   // [cfg.hidden_dim]
static float *s_h_post    = NULL;   // [cfg.hidden_dim]
static float *s_h_mid     = NULL;   // [cfg.hidden_dim]
static float *s_gate_scores = NULL; // [cfg.num_experts]
static float *s_spec_gate_scores = NULL; // [cfg.num_experts] speculative routing scratch
static int s_spec_indices[MAX_K];         // speculative routing predicted expert indices
static int s_spec_count = 0;              // number of speculative predictions this layer
static float *s_shared_gate = NULL; // [cfg.shared_intermediate]
static float *s_shared_up  = NULL;  // [cfg.shared_intermediate]
static float g_merged_shared_gate_score = 0.0f; // CMD1+CMD2 merge: carries shared gate score
static float *s_moe_out   = NULL;   // [cfg.hidden_dim]
static float *s_shared_out = NULL;  // [cfg.hidden_dim]
// Full attention scratch
static float *s_q_proj_out = NULL;  // [cfg.num_attn_heads * cfg.head_dim * 2]
static float *s_k_proj_out = NULL;  // [cfg.num_kv_heads * cfg.head_dim]
static float *s_v_proj_out = NULL;  // [cfg.num_kv_heads * cfg.head_dim]
static float *s_q         = NULL;   // [cfg.num_attn_heads * cfg.head_dim]
static float *s_q_gate    = NULL;   // [cfg.num_attn_heads * cfg.head_dim]
static float *s_attn_out  = NULL;   // [cfg.num_attn_heads * cfg.head_dim]
// Linear attention scratch
static float *s_qkv_proj_out = NULL;   // [cfg.linear_conv_dim]
static float *s_z_proj_out   = NULL;   // [cfg.linear_total_value]
static float *s_beta_proj_out = NULL;  // [cfg.linear_num_v_heads]
static float *s_alpha_proj_out = NULL; // [cfg.linear_num_v_heads]
static float *s_conv_out  = NULL;   // [cfg.linear_conv_dim]
static float *s_out_vals  = NULL;   // [cfg.linear_total_value]
static float *s_gated_out = NULL;   // [cfg.linear_total_value]
// CPU fallback scratch (MoE expert compute + shared expert + FP8 attention)
static float *s_expert_out_cpu = NULL;   // [cfg.hidden_dim]
static float *s_gate_proj_out  = NULL;   // [cfg.moe_intermediate]
static float *s_up_proj_out    = NULL;   // [cfg.moe_intermediate]
static float *s_act_out        = NULL;   // [cfg.moe_intermediate]
static float *s_shared_act     = NULL;   // [cfg.shared_intermediate]
static float *s_k_dequant      = NULL;   // [cfg.num_kv_heads * cfg.head_dim] (FP8 attention)
static float *s_v_dequant      = NULL;   // [cfg.num_kv_heads * cfg.head_dim] (FP8 attention)
static int    s_scratch_initialized = 0;

static int init_layer_scratch(void) {
    if (s_scratch_initialized) return 0;  // already initialized

    s_normed     = calloc(cfg.hidden_dim, sizeof(float));
    s_residual   = calloc(cfg.hidden_dim, sizeof(float));
    s_attn_proj  = calloc(cfg.hidden_dim, sizeof(float));
    s_h_post     = calloc(cfg.hidden_dim, sizeof(float));
    s_h_mid      = calloc(cfg.hidden_dim, sizeof(float));
    s_gate_scores = calloc(cfg.num_experts, sizeof(float));
    s_spec_gate_scores = calloc(cfg.num_experts, sizeof(float));
    s_shared_gate = calloc(cfg.shared_intermediate, sizeof(float));
    s_shared_up  = calloc(cfg.shared_intermediate, sizeof(float));
    s_moe_out    = calloc(cfg.hidden_dim, sizeof(float));
    s_shared_out = calloc(cfg.hidden_dim, sizeof(float));
    s_q_proj_out = calloc(cfg.num_attn_heads * cfg.head_dim * 2, sizeof(float));
    s_k_proj_out = calloc(cfg.num_kv_heads * cfg.head_dim, sizeof(float));
    s_v_proj_out = calloc(cfg.num_kv_heads * cfg.head_dim, sizeof(float));
    s_q          = calloc(cfg.num_attn_heads * cfg.head_dim, sizeof(float));
    s_q_gate     = calloc(cfg.num_attn_heads * cfg.head_dim, sizeof(float));
    s_attn_out   = calloc(cfg.num_attn_heads * cfg.head_dim, sizeof(float));
    s_qkv_proj_out = calloc(cfg.linear_conv_dim, sizeof(float));
    s_z_proj_out   = calloc(cfg.linear_total_value, sizeof(float));
    s_beta_proj_out = calloc(cfg.linear_num_v_heads, sizeof(float));
    s_alpha_proj_out = calloc(cfg.linear_num_v_heads, sizeof(float));
    s_conv_out   = calloc(cfg.linear_conv_dim, sizeof(float));
    s_out_vals   = calloc(cfg.linear_total_value, sizeof(float));
    s_gated_out  = calloc(cfg.linear_total_value, sizeof(float));
    // CPU fallback scratch buffers
    s_expert_out_cpu = calloc(cfg.hidden_dim, sizeof(float));
    s_gate_proj_out  = calloc(cfg.moe_intermediate, sizeof(float));
    s_up_proj_out    = calloc(cfg.moe_intermediate, sizeof(float));
    s_act_out        = calloc(cfg.moe_intermediate, sizeof(float));
    s_shared_act     = calloc(cfg.shared_intermediate, sizeof(float));
    s_k_dequant      = calloc(cfg.num_kv_heads * cfg.head_dim, sizeof(float));
    s_v_dequant      = calloc(cfg.num_kv_heads * cfg.head_dim, sizeof(float));

    // Verify all allocations succeeded
    if (!s_normed || !s_residual || !s_attn_proj || !s_h_post || !s_h_mid ||
        !s_gate_scores || !s_spec_gate_scores || !s_shared_gate || !s_shared_up ||
        !s_moe_out || !s_shared_out || !s_q_proj_out || !s_k_proj_out || !s_v_proj_out ||
        !s_q || !s_q_gate || !s_attn_out || !s_qkv_proj_out || !s_z_proj_out ||
        !s_beta_proj_out || !s_alpha_proj_out || !s_conv_out || !s_out_vals || !s_gated_out ||
        !s_expert_out_cpu || !s_gate_proj_out || !s_up_proj_out || !s_act_out ||
        !s_shared_act || !s_k_dequant || !s_v_dequant) {
        fprintf(stderr, "ERROR: Failed to allocate layer scratch buffers (hidden_dim=%d, moe_intermediate=%d)\n",
                cfg.hidden_dim, cfg.moe_intermediate);
        return -1;
    }

    s_scratch_initialized = 1;
    return 0;
}

static void free_layer_scratch(void) {
    if (!s_scratch_initialized) return;
    free(s_normed);     free(s_residual);   free(s_attn_proj);
    free(s_h_post);     free(s_h_mid);      free(s_gate_scores);
    free(s_spec_gate_scores); free(s_shared_gate); free(s_shared_up);
    free(s_moe_out);    free(s_shared_out);
    free(s_q_proj_out); free(s_k_proj_out); free(s_v_proj_out);
    free(s_q);          free(s_q_gate);     free(s_attn_out);
    free(s_qkv_proj_out); free(s_z_proj_out);
    free(s_beta_proj_out); free(s_alpha_proj_out);
    free(s_conv_out);   free(s_out_vals);   free(s_gated_out);
    free(s_expert_out_cpu); free(s_gate_proj_out); free(s_up_proj_out);
    free(s_act_out);    free(s_shared_act);
    free(s_k_dequant);  free(s_v_dequant);
    s_normed = s_residual = s_attn_proj = s_h_post = s_h_mid = NULL;
    s_gate_scores = s_spec_gate_scores = s_shared_gate = s_shared_up = NULL;
    s_moe_out = s_shared_out = NULL;
    s_q_proj_out = s_k_proj_out = s_v_proj_out = s_q = s_q_gate = s_attn_out = NULL;
    s_qkv_proj_out = s_z_proj_out = s_beta_proj_out = s_alpha_proj_out = NULL;
    s_conv_out = s_out_vals = s_gated_out = NULL;
    s_expert_out_cpu = s_gate_proj_out = s_up_proj_out = s_act_out = NULL;
    s_shared_act = s_k_dequant = s_v_dequant = NULL;
    s_scratch_initialized = 0;
}

static void fused_layer_forward(
    WeightFile *wf,
    int layer_idx,
    float *hidden,           // [cfg.hidden_dim] in/out
    KVCache *kv,             // non-NULL for full attention layers
    LinearAttnState *la_state, // non-NULL for linear attention layers
    int pos,                 // position for RoPE
    const void *mmap_base,   // mmap'd layer file (NULL if not available)
    int K,                   // number of active experts
    int packed_fd            // fd for packed expert file
) {
    double t_layer_start = 0, t0 = 0, t1 = 0;
    if (g_timing_enabled) { t_layer_start = now_ms(); }
    int pred_started = 0;  // set to 1 if we started prediction preads during CMD1_wait

    if (init_layer_scratch() != 0) {
        fprintf(stderr, "FATAL: scratch buffer allocation failed, cannot proceed\n");
        return;
    }
    if (!layer_cache_built) build_layer_cache(wf);
    LayerWeightCache *lc = &layer_cache[layer_idx];
    int is_full = (kv != NULL);

    // =====================================================================
    // PHASE 1: Deferred completion + CMD1 (attention projections)
    // =====================================================================

    // ---- Prepare attention projection specs (doesn't depend on hidden) ----
    int num_attn_specs = 0;
    BatchMatvecSpec attn_specs[5];
    float *q_proj_out = NULL, *k_out = NULL, *v_out = NULL;
    float *qkv_out = NULL, *z_out = NULL, *beta_out = NULL, *alpha_out = NULL;

    if (is_full) {
        int q_proj_dim = cfg.num_attn_heads * cfg.head_dim * 2;
        int kv_dim = cfg.num_kv_heads * cfg.head_dim;

        q_proj_out = s_q_proj_out;
        k_out = s_k_proj_out;
        v_out = s_v_proj_out;

        if (lc->q_w && lc->q_s && lc->q_b && lc->k_w && lc->k_s && lc->k_b &&
            lc->v_w && lc->v_s && lc->v_b) {
            attn_specs[0] = (BatchMatvecSpec){ lc->q_w, lc->q_s, lc->q_b, q_proj_out, (uint32_t)q_proj_dim, cfg.hidden_dim, cfg.group_size, 0 };
            attn_specs[1] = (BatchMatvecSpec){ lc->k_w, lc->k_s, lc->k_b, k_out,      (uint32_t)kv_dim,     cfg.hidden_dim, cfg.group_size, 1 };
            attn_specs[2] = (BatchMatvecSpec){ lc->v_w, lc->v_s, lc->v_b, v_out,      (uint32_t)kv_dim,     cfg.hidden_dim, cfg.group_size, 2 };
            num_attn_specs = 3;
        }
    } else {
        int qkv_dim = cfg.linear_conv_dim;
        int z_dim = cfg.linear_total_value;

        qkv_out = s_qkv_proj_out;
        z_out = s_z_proj_out;
        beta_out = s_beta_proj_out;
        alpha_out = s_alpha_proj_out;

        if (lc->qkv_w && lc->qkv_s && lc->qkv_b && lc->z_w && lc->z_s && lc->z_b &&
            lc->b_w && lc->b_s && lc->b_b && lc->a_w && lc->a_s && lc->a_b) {
            attn_specs[0] = (BatchMatvecSpec){ lc->qkv_w, lc->qkv_s, lc->qkv_b, qkv_out,   (uint32_t)qkv_dim,            cfg.hidden_dim, cfg.group_size, 0 };
            attn_specs[1] = (BatchMatvecSpec){ lc->z_w,   lc->z_s,   lc->z_b,   z_out,      (uint32_t)z_dim,              cfg.hidden_dim, cfg.group_size, 1 };
            attn_specs[2] = (BatchMatvecSpec){ lc->b_w,   lc->b_s,   lc->b_b,   beta_out,   (uint32_t)cfg.linear_num_v_heads, cfg.hidden_dim, cfg.group_size, 2 };
            attn_specs[3] = (BatchMatvecSpec){ lc->a_w,   lc->a_s,   lc->a_b,   alpha_out,  (uint32_t)cfg.linear_num_v_heads, cfg.hidden_dim, cfg.group_size, 3 };
            num_attn_specs = 4;
        }
    }

    // ---- Deferred completion + CMD1 (sequential) ----
    float *normed = s_normed;
    float *residual = s_residual;
    id<MTLCommandBuffer> cmd1 = nil;
    int gpu_linear_attn = 0;  // set to 1 if GPU handles entire linear attention pipeline

    // Pre-compute linear_layer_idx for GPU linear attention encoding in CMD1
    int linear_layer_idx = -1;
    if (!is_full) {
        linear_layer_idx = cfg.linear_index[layer_idx];
    }
    // Can we run the full linear attention pipeline on GPU in CMD1?
    int can_gpu_linear = (gpu_linear_attn_enabled &&
                          !is_full && g_metal && g_metal->delta_net_step &&
                          g_metal->conv1d_step && g_metal->rms_norm_qk &&
                          g_metal->compute_decay_beta && g_metal->gated_rms_norm &&
                          g_metal->wf_buf &&
                          linear_layer_idx >= 0 && linear_layer_idx < cfg.num_linear_layers &&
                          lc->conv1d_w && lc->A_log && lc->dt_bias && lc->gated_norm_w &&
                          !linear_attn_bypass);

    // Check if previous layer's CMD3 already computed combine+residual+norm on GPU.
    // If so, buf_input already contains the normalized input for this layer's CMD1.
    // We can submit CMD1 immediately — the GPU queue serializes CMD3(N-1) then CMD1(N).
    int prev_gpu_combined = (g_deferred.active && g_deferred.gpu_combined);
    int cmd1_cmd2_merged = 0;  // set to 1 when CMD2 is encoded into CMD1 (linear attn only)

    if (prev_gpu_combined && g_metal && g_metal->wf_buf && num_attn_specs > 0) {
        // ---- FAST PATH: GPU-combined previous CMD3 ----
        // buf_input already has the normalized hidden state from CMD3(N-1).
        // Submit CMD1 immediately — GPU runs CMD3(N-1) then CMD1(N) back-to-back.
        if (g_timing_enabled) { t0 = now_ms(); }

        cmd1 = [g_metal->queue commandBuffer];
        gpu_encode_batch_matvec(g_metal, cmd1, attn_specs, num_attn_specs);

        // GPU linear attention: encode conv1d + normalize + decay/beta + delta-net + gated_norm into CMD1
        if (can_gpu_linear && num_attn_specs == 4) {
            // batch_out[0]=qkv(12288), [1]=z(8192), [2]=beta(64), [3]=alpha(64)
            uint32_t conv_dim = cfg.linear_conv_dim;
            // Enc L1: conv1d_step — input=batch_out[0], weights=conv1d_w, state=buf_conv_state, output=buf_conv_output
            // NOTE: staging was already reset by gpu_encode_batch_matvec above.
            // All tensors below accumulate into the same staging buffer — no resets
            // until cmd1 is committed, to avoid overwriting data the GPU hasn't read yet.
            {
                id<MTLBuffer> conv1d_w_buf; NSUInteger conv1d_w_off;
                metal_find_chunk_sized(g_metal, lc->conv1d_w,
                    (size_t)cfg.linear_conv_dim * cfg.linear_conv_dim * 2,
                    &conv1d_w_buf, &conv1d_w_off);
                id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                [enc setComputePipelineState:g_metal->conv1d_step];
                [enc setBuffer:g_metal->buf_conv_state[linear_layer_idx] offset:0 atIndex:0];
                [enc setBuffer:g_metal->batch_out[0]    offset:0            atIndex:1]; // qkv projection output
                [enc setBuffer:conv1d_w_buf offset:conv1d_w_off atIndex:2]; // conv weights (bf16)
                [enc setBuffer:g_metal->buf_conv_output offset:0            atIndex:3]; // conv output
                [enc setBytes:&conv_dim length:4 atIndex:4];
                uint32_t tgs = (conv_dim + 255) / 256;
                [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }

            // Enc L2: rms_norm_qk — normalize q and k in conv_output in-place
            {
                uint32_t key_dim = cfg.linear_key_dim;  // 128
                float inv_scale = 1.0f / sqrtf((float)cfg.linear_key_dim);
                id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                [enc setComputePipelineState:g_metal->rms_norm_qk];
                [enc setBuffer:g_metal->buf_conv_output offset:0 atIndex:0];  // q at offset 0
                [enc setBuffer:g_metal->buf_conv_output offset:cfg.linear_total_key * sizeof(float) atIndex:1];  // k at offset 2048 floats
                [enc setBytes:&key_dim   length:4 atIndex:2];
                [enc setBytes:&inv_scale length:4 atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(cfg.linear_num_k_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(cfg.linear_key_dim, 1, 1)];
                [enc endEncoding];
            }

            // Enc L3: compute_decay_beta — alpha=batch_out[3], beta=batch_out[2], A_log+dt_bias from wf_buf
            {
                id<MTLBuffer> alog_buf, dtb_buf; NSUInteger alog_off, dtb_off;
                metal_find_chunk_sized(g_metal, lc->A_log,
                    (size_t)cfg.linear_num_v_heads * 4, &alog_buf, &alog_off);  // float
                metal_find_chunk_sized(g_metal, lc->dt_bias,
                    (size_t)cfg.linear_num_v_heads * 2, &dtb_buf, &dtb_off);    // bf16
                id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                [enc setComputePipelineState:g_metal->compute_decay_beta];
                [enc setBuffer:g_metal->batch_out[3]       offset:0          atIndex:0]; // alpha
                [enc setBuffer:g_metal->batch_out[2]       offset:0          atIndex:1]; // beta
                [enc setBuffer:alog_buf  offset:alog_off  atIndex:2]; // A_log
                [enc setBuffer:dtb_buf   offset:dtb_off   atIndex:3]; // dt_bias (bf16)
                [enc setBuffer:g_metal->buf_delta_g_decay  offset:0          atIndex:4]; // g_decay output
                [enc setBuffer:g_metal->buf_delta_beta     offset:0          atIndex:5]; // beta_gate output
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(cfg.linear_num_v_heads, 1, 1)];
                [enc endEncoding];
            }

            // Enc L4: gated_delta_net_step — the main recurrence
            // Use fused kernel (pass 2+3 merged) when available, original as fallback
            {
                uint32_t khpv = cfg.linear_num_v_heads / cfg.linear_num_k_heads;  // 4
                id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                [enc setComputePipelineState:g_metal->delta_net_step_fused ? g_metal->delta_net_step_fused : g_metal->delta_net_step];
                [enc setBuffer:g_metal->buf_delta_state[linear_layer_idx] offset:0 atIndex:0]; // persistent state
                [enc setBuffer:g_metal->buf_conv_output offset:0 atIndex:1]; // q (first 2048 floats)
                [enc setBuffer:g_metal->buf_conv_output offset:cfg.linear_total_key * sizeof(float) atIndex:2]; // k (next 2048)
                [enc setBuffer:g_metal->buf_conv_output offset:2 * cfg.linear_total_key * sizeof(float) atIndex:3]; // v (next 8192)
                [enc setBuffer:g_metal->buf_delta_g_decay offset:0 atIndex:4];
                [enc setBuffer:g_metal->buf_delta_beta    offset:0 atIndex:5];
                [enc setBuffer:g_metal->buf_delta_output  offset:0 atIndex:6]; // output [8192]
                [enc setBytes:&khpv length:sizeof(khpv) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(cfg.linear_num_v_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [enc endEncoding];
            }

            // Enc L5: gated_rms_norm — normalize+gate delta-net output -> batch_out[6] for CMD2 o_proj
            {
                id<MTLBuffer> gnw_buf; NSUInteger gnw_off;
                metal_find_chunk_sized(g_metal, lc->gated_norm_w,
                    (size_t)cfg.linear_total_value * 2, &gnw_buf, &gnw_off);  // bf16
                uint32_t value_dim = cfg.linear_value_dim;  // 128
                float eps = cfg.rms_norm_eps;
                id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                [enc setComputePipelineState:g_metal->gated_rms_norm];
                [enc setBuffer:g_metal->buf_delta_output offset:0          atIndex:0]; // values [8192]
                [enc setBuffer:g_metal->batch_out[1]     offset:0          atIndex:1]; // z (z projection output) [8192]
                [enc setBuffer:gnw_buf offset:gnw_off atIndex:2]; // weight (bf16)
                [enc setBuffer:g_metal->batch_out[6]     offset:0          atIndex:3]; // output -> batch_out[6] for CMD2
                [enc setBytes:&value_dim length:4 atIndex:4];
                [enc setBytes:&eps       length:4 atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(cfg.linear_num_v_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(cfg.linear_value_dim, 1, 1)];
                [enc endEncoding];
            }

            gpu_linear_attn = 1;
        }

        // ---- CMD1+CMD2 merge for linear attention layers ----
        // When gpu_linear_attn is active, CMD2 (o_proj + residual + norm + routing)
        // can be encoded directly into CMD1, eliminating one commit+wait cycle.
        // buf_moe_hidden (from CMD3(N-1)) is used as the residual source on GPU —
        // serial queue ordering guarantees CMD3(N-1) completes before CMD1 starts.
        // CMD1+CMD2 merge: encode CMD2 dispatches into CMD1 for linear attention layers,
        // eliminating one commit+wait cycle per layer (45 layers x ~0.05-0.1ms).
        if (g_cmd_merge_enabled && gpu_linear_attn && g_metal->wf_buf &&
            lc->gate_w && lc->gate_s && lc->gate_b &&
            lc->sg_w && lc->sg_s && lc->sg_b &&
            lc->su_w && lc->su_s && lc->su_b &&
            lc->seg_w && lc->seg_s && lc->seg_b &&
            g_metal->residual_add && g_metal->rms_norm_sum &&
            g_metal->rms_norm_apply_bf16 && lc->post_attn_norm_w &&
            g_deferred.gpu_combined && g_metal->buf_moe_hidden) {
            // batch_out[6] already has the result from CMD1 gated_rms_norm.
            // buf_moe_hidden has the pre-attention hidden state (residual) from CMD3(N-1).

            // ---- o_proj matvec into cmd1 ----
            // For linear attention: out_proj (not o_proj), in_dim = total_value
            {
                uint32_t o_out_dim = cfg.hidden_dim;
                uint32_t o_in_dim = (uint32_t)cfg.linear_total_value;
                uint32_t o_gs = cfg.group_size;
                id<MTLBuffer> ow_buf, os_buf, ob_buf;
                NSUInteger ow_off, os_off, ob_off;
                size_t m_oproj_w_size = (size_t)o_out_dim * o_in_dim / 8;
                size_t m_oproj_ng = (o_in_dim + o_gs - 1) / o_gs;
                size_t m_oproj_sb_size = (size_t)o_out_dim * m_oproj_ng * sizeof(uint16_t);
                metal_staging_reset(g_metal);
                metal_find_chunk_sized(g_metal, lc->out_proj_w, m_oproj_w_size, &ow_buf, &ow_off);
                metal_find_chunk_sized(g_metal, lc->out_proj_s, m_oproj_sb_size, &os_buf, &os_off);
                metal_find_chunk_sized(g_metal, lc->out_proj_b, m_oproj_sb_size, &ob_buf, &ob_off);
                id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                [enc setComputePipelineState:g_metal->matvec_fast];
                [enc setBuffer:ow_buf offset:ow_off atIndex:0];
                [enc setBuffer:os_buf offset:os_off atIndex:1];
                [enc setBuffer:ob_buf offset:ob_off atIndex:2];
                [enc setBuffer:g_metal->batch_out[6] offset:0 atIndex:3];
                [enc setBuffer:g_metal->buf_output   offset:0 atIndex:4];
                [enc setBytes:&o_out_dim  length:4 atIndex:5];
                [enc setBytes:&o_in_dim   length:4 atIndex:6];
                [enc setBytes:&o_gs       length:4 atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(o_out_dim, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                [enc endEncoding];
            }
            // ---- residual_add (buf_output + buf_moe_hidden -> buf_h_mid) ----
            {
                id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                uint32_t dim = cfg.hidden_dim;
                [enc setComputePipelineState:g_metal->residual_add];
                [enc setBuffer:g_metal->buf_moe_hidden offset:0 atIndex:0]; // residual from CMD3(N-1)
                [enc setBuffer:g_metal->buf_output     offset:0 atIndex:1]; // o_proj result
                [enc setBuffer:g_metal->buf_h_mid      offset:0 atIndex:2]; // out
                [enc setBytes:&dim length:4 atIndex:3];
                uint32_t tgs = (dim + 255) / 256;
                [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
            // ---- rms_norm_sum_sq (buf_h_mid -> buf_sum_sq) ----
            {
                id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                uint32_t dim = cfg.hidden_dim;
                [enc setComputePipelineState:g_metal->rms_norm_sum];
                [enc setBuffer:g_metal->buf_h_mid  offset:0 atIndex:0];
                [enc setBuffer:g_metal->buf_sum_sq offset:0 atIndex:1];
                [enc setBytes:&dim length:4 atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
            // ---- rms_norm_apply_bf16 (buf_h_mid + norm_w -> buf_input) ----
            {
                id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                id<MTLBuffer> panw_buf; NSUInteger panw_off;
                metal_find_chunk_sized(g_metal, lc->post_attn_norm_w,
                    (size_t)cfg.hidden_dim * 2, &panw_buf, &panw_off);
                uint32_t dim = cfg.hidden_dim;
                float eps = cfg.rms_norm_eps;
                [enc setComputePipelineState:g_metal->rms_norm_apply_bf16];
                [enc setBuffer:g_metal->buf_h_mid  offset:0       atIndex:0];
                [enc setBuffer:panw_buf offset:panw_off atIndex:1];
                [enc setBuffer:g_metal->buf_sum_sq offset:0       atIndex:2];
                [enc setBuffer:g_metal->buf_input  offset:0       atIndex:3];
                [enc setBytes:&dim length:4 atIndex:4];
                [enc setBytes:&eps length:4 atIndex:5];
                uint32_t tgs = (dim + 255) / 256;
                [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
            // ---- routing + shared expert projections (using lc-> pointers directly) ----
            // Output buffers: allocate on stack for gpu_flush_batch_results
            float *m_gate_scores = s_gate_scores;
            memset(m_gate_scores, 0, cfg.num_experts * sizeof(float));
            float *m_shared_gate = s_shared_gate;
            memset(m_shared_gate, 0, cfg.shared_intermediate * sizeof(float));
            float *m_shared_up = s_shared_up;
            memset(m_shared_up, 0, cfg.shared_intermediate * sizeof(float));
            float m_shared_gate_score = 0.0f;
            BatchMatvecSpec moe_specs_merged[4] = {
                { lc->gate_w, lc->gate_s, lc->gate_b, m_gate_scores,      (uint32_t)cfg.num_experts,         cfg.hidden_dim, cfg.group_size, 0 },
                { lc->sg_w,   lc->sg_s,   lc->sg_b,   m_shared_gate,      (uint32_t)cfg.shared_intermediate, cfg.hidden_dim, cfg.group_size, 1 },
                { lc->su_w,   lc->su_s,   lc->su_b,   m_shared_up,        (uint32_t)cfg.shared_intermediate, cfg.hidden_dim, cfg.group_size, 2 },
                { lc->seg_w,  lc->seg_s,  lc->seg_b,  &m_shared_gate_score, 1,                              cfg.hidden_dim, cfg.group_size, 3 },
            };
            gpu_encode_batch_matvec(g_metal, cmd1, moe_specs_merged, 4);

            cmd1_cmd2_merged = 1;
        }

        [cmd1 commit];

        if (g_timing_enabled) { t1 = now_ms(); g_timing.cmd1_submit += t1 - t0; }

        // Wait for CMD1 (implies CMD3(N-1) also done, since queue is serial)
        if (g_timing_enabled) { t0 = now_ms(); }
        [cmd1 waitUntilCompleted];
        if (!gpu_linear_attn) {
            gpu_flush_batch_results(g_metal, attn_specs, num_attn_specs);
        }
        if (cmd1_cmd2_merged) {
            // Read back merged CMD2 results: routing scores + shared expert outputs.
            // The output pointers in moe_specs_merged point to s_gate_scores, s_shared_gate, etc.
            // which are the same static arrays that gate_scores/shared_gate/etc. will alias later.
            // gpu_flush reads GPU batch_out[0..3] → those CPU arrays.
            float m_sgs = 0.0f;
            BatchMatvecSpec moe_rb[4] = {
                { NULL, NULL, NULL, s_gate_scores,   (uint32_t)cfg.num_experts,         cfg.hidden_dim, cfg.group_size, 0 },
                { NULL, NULL, NULL, s_shared_gate,   (uint32_t)cfg.shared_intermediate, cfg.hidden_dim, cfg.group_size, 1 },
                { NULL, NULL, NULL, s_shared_up,     (uint32_t)cfg.shared_intermediate, cfg.hidden_dim, cfg.group_size, 2 },
                { NULL, NULL, NULL, &m_sgs,            1,                              cfg.hidden_dim, cfg.group_size, 3 },
            };
            gpu_flush_batch_results(g_metal, moe_rb, 4);
            g_merged_shared_gate_score = m_sgs;
            // Read h_mid and h_post from GPU
            memcpy(s_h_mid, [g_metal->buf_h_mid contents], cfg.hidden_dim * sizeof(float));
            memcpy(s_h_post, [g_metal->buf_input contents], cfg.hidden_dim * sizeof(float));
            memcpy(hidden, s_h_mid, cfg.hidden_dim * sizeof(float));
        }
        if (g_timing_enabled) { t1 = now_ms(); g_timing.cmd1_wait += t1 - t0; }

        // Now CMD3(N-1) is done. Read back hidden state from GPU.
        if (g_timing_enabled) { t0 = now_ms(); }
        if (!cmd1_cmd2_merged) {
            finalize_deferred_experts();  // reads buf_moe_hidden -> hidden
        } else {
            // Merged path: hidden already set from buf_h_mid above.
            // Clear deferred state so CMD3(N) can proceed.
            g_deferred.active = 0;
            g_deferred.cmd_experts = nil;
        }

        // Start predicted expert preads AFTER CMD1_wait.
        // CMD3(N-1) is guaranteed done (serial queue), so buf_B is safe to overwrite.
        // Predictions overlap with CPU attn + CMD2 + routing (~0.6ms head start).
        // Predicted experts that hit page cache (same as previous token) complete in ~0.1ms.
        // Skip if cross-layer prefetch already has buf_B in-flight for this layer.
        if (g_pred_enabled && g_pred_generating && g_pred_valid && packed_fd >= 0 &&
            g_metal->buf_multi_expert_data_B[0] && PRED_COUNT(layer_idx) > 0 &&
            !(g_prefetch_active && g_prefetch_layer == layer_idx)) {
            async_pread_start(packed_fd, &PRED_EXPERT(layer_idx, 0),
                              PRED_COUNT(layer_idx),
                              g_metal->buf_multi_expert_data_B, mmap_base,
                              layer_idx);
            pred_started = 1;
        }
        if (!cmd1_cmd2_merged) {
            // Set up residual for CMD2 (residual = hidden before this layer's attention)
            cpu_vec_copy(residual, hidden, cfg.hidden_dim);
        }
        if (g_timing_enabled) { t1 = now_ms(); g_timing.deferred_cpu += t1 - t0; }

        // No input_norm needed — CMD3 already computed it into buf_input.
        // normed is only needed if speculative routing is enabled (currently disabled).
        // Skip the readback to avoid unnecessary overhead.
    } else {
        // ---- ORIGINAL PATH: CPU deferred completion + input norm ----
        // Complete deferred experts from previous layer
        if (g_timing_enabled) { t0 = now_ms(); }
        wait_deferred_experts_gpu();
        if (g_timing_enabled) { t1 = now_ms(); g_timing.deferred_wait += t1 - t0; }

        if (g_timing_enabled) { t0 = now_ms(); }
        finalize_deferred_experts();
        if (g_timing_enabled) { t1 = now_ms(); g_timing.deferred_cpu += t1 - t0; }

        // Input norm
        if (g_timing_enabled) { t0 = now_ms(); }
        cpu_vec_copy(residual, hidden, cfg.hidden_dim);
        cpu_rms_norm(hidden, lc->input_norm_w, normed, cfg.hidden_dim, cfg.rms_norm_eps);
        if (g_timing_enabled) { t1 = now_ms(); g_timing.input_norm += t1 - t0; }

        // Submit CMD1: attention projections
        if (g_timing_enabled) { t0 = now_ms(); }
        if (g_metal && g_metal->wf_buf && num_attn_specs > 0) {
            memcpy([g_metal->buf_input contents], normed, cfg.hidden_dim * sizeof(float));
            cmd1 = [g_metal->queue commandBuffer];
            gpu_encode_batch_matvec(g_metal, cmd1, attn_specs, num_attn_specs);

            // GPU linear attention: encode conv1d + normalize + decay/beta + delta-net + gated_norm into CMD1
            if (can_gpu_linear && num_attn_specs == 4) {
                uint32_t conv_dim = cfg.linear_conv_dim;

                // Enc L1: conv1d_step
                // NOTE: staging was reset by gpu_encode_batch_matvec — don't reset again
                {
                    id<MTLBuffer> conv1d_w_buf; NSUInteger conv1d_w_off;
                    metal_find_chunk_sized(g_metal, lc->conv1d_w,
                        (size_t)cfg.linear_conv_dim * cfg.linear_conv_dim * 2,
                        &conv1d_w_buf, &conv1d_w_off);
                    id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                    [enc setComputePipelineState:g_metal->conv1d_step];
                    [enc setBuffer:g_metal->buf_conv_state[linear_layer_idx] offset:0 atIndex:0];
                    [enc setBuffer:g_metal->batch_out[0]    offset:0            atIndex:1];
                    [enc setBuffer:conv1d_w_buf offset:conv1d_w_off atIndex:2];
                    [enc setBuffer:g_metal->buf_conv_output offset:0            atIndex:3];
                    [enc setBytes:&conv_dim length:4 atIndex:4];
                    uint32_t tgs = (conv_dim + 255) / 256;
                    [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc endEncoding];
                }

                // Enc L2: rms_norm_qk
                {
                    uint32_t key_dim = cfg.linear_key_dim;
                    float inv_scale = 1.0f / sqrtf((float)cfg.linear_key_dim);
                    id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                    [enc setComputePipelineState:g_metal->rms_norm_qk];
                    [enc setBuffer:g_metal->buf_conv_output offset:0 atIndex:0];
                    [enc setBuffer:g_metal->buf_conv_output offset:cfg.linear_total_key * sizeof(float) atIndex:1];
                    [enc setBytes:&key_dim   length:4 atIndex:2];
                    [enc setBytes:&inv_scale length:4 atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(cfg.linear_num_k_heads, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(cfg.linear_key_dim, 1, 1)];
                    [enc endEncoding];
                }

                // Enc L3: compute_decay_beta
                {
                    id<MTLBuffer> alog_buf, dtb_buf; NSUInteger alog_off, dtb_off;
                    metal_find_chunk_sized(g_metal, lc->A_log,
                        (size_t)cfg.linear_num_v_heads * 4, &alog_buf, &alog_off);  // float
                    metal_find_chunk_sized(g_metal, lc->dt_bias,
                        (size_t)cfg.linear_num_v_heads * 2, &dtb_buf, &dtb_off);    // bf16
                    id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                    [enc setComputePipelineState:g_metal->compute_decay_beta];
                    [enc setBuffer:g_metal->batch_out[3]       offset:0          atIndex:0];
                    [enc setBuffer:g_metal->batch_out[2]       offset:0          atIndex:1];
                    [enc setBuffer:alog_buf  offset:alog_off  atIndex:2];
                    [enc setBuffer:dtb_buf   offset:dtb_off   atIndex:3];
                    [enc setBuffer:g_metal->buf_delta_g_decay  offset:0          atIndex:4];
                    [enc setBuffer:g_metal->buf_delta_beta     offset:0          atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(cfg.linear_num_v_heads, 1, 1)];
                    [enc endEncoding];
                }

                // Enc L4: gated_delta_net_step (fused pass 2+3 when available)
                {
                    uint32_t khpv = cfg.linear_num_v_heads / cfg.linear_num_k_heads;
                    id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                    [enc setComputePipelineState:g_metal->delta_net_step_fused ? g_metal->delta_net_step_fused : g_metal->delta_net_step];
                    [enc setBuffer:g_metal->buf_delta_state[linear_layer_idx] offset:0 atIndex:0];
                    [enc setBuffer:g_metal->buf_conv_output offset:0 atIndex:1];
                    [enc setBuffer:g_metal->buf_conv_output offset:cfg.linear_total_key * sizeof(float) atIndex:2];
                    [enc setBuffer:g_metal->buf_conv_output offset:2 * cfg.linear_total_key * sizeof(float) atIndex:3];
                    [enc setBuffer:g_metal->buf_delta_g_decay offset:0 atIndex:4];
                    [enc setBuffer:g_metal->buf_delta_beta    offset:0 atIndex:5];
                    [enc setBuffer:g_metal->buf_delta_output  offset:0 atIndex:6];
                    [enc setBytes:&khpv length:sizeof(khpv) atIndex:7];
                    [enc dispatchThreadgroups:MTLSizeMake(cfg.linear_num_v_heads, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    [enc endEncoding];
                }

                // Enc L5: gated_rms_norm -> batch_out[6]
                {
                    id<MTLBuffer> gnw_buf; NSUInteger gnw_off;
                    metal_find_chunk_sized(g_metal, lc->gated_norm_w,
                        (size_t)cfg.linear_total_value * 2, &gnw_buf, &gnw_off);  // bf16
                    uint32_t value_dim = cfg.linear_value_dim;
                    float eps = cfg.rms_norm_eps;
                    id<MTLComputeCommandEncoder> enc = [cmd1 computeCommandEncoder];
                    [enc setComputePipelineState:g_metal->gated_rms_norm];
                    [enc setBuffer:g_metal->buf_delta_output offset:0          atIndex:0];
                    [enc setBuffer:g_metal->batch_out[1]     offset:0          atIndex:1];
                    [enc setBuffer:gnw_buf offset:gnw_off atIndex:2];
                    [enc setBuffer:g_metal->batch_out[6]     offset:0          atIndex:3];
                    [enc setBytes:&value_dim length:4 atIndex:4];
                    [enc setBytes:&eps       length:4 atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake(cfg.linear_num_v_heads, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(cfg.linear_value_dim, 1, 1)];
                    [enc endEncoding];
                }

                gpu_linear_attn = 1;
            }

            [cmd1 commit];
        } else {
            for (int i = 0; i < num_attn_specs; i++) {
                BatchMatvecSpec *s = &attn_specs[i];
                cpu_dequant_matvec(s->W, s->scales, s->biases, normed, s->out_cpu,
                                   s->out_dim, s->in_dim, s->group_size);
            }
        }
        if (g_timing_enabled) { t1 = now_ms(); g_timing.cmd1_submit += t1 - t0; }

        // Wait for CMD1
        if (g_timing_enabled) { t0 = now_ms(); }
        if (cmd1) {
            [cmd1 waitUntilCompleted];
            if (!gpu_linear_attn) {
                gpu_flush_batch_results(g_metal, attn_specs, num_attn_specs);
            }
        }
        if (g_timing_enabled) { t1 = now_ms(); g_timing.cmd1_wait += t1 - t0; }
    }

    // =====================================================================
    // SPECULATIVE EARLY ROUTING — overlap expert I/O with CPU attention
    // =====================================================================
    // Compute approximate routing using the PRE-attention normed hidden state.
    // The real routing (in CMD2/PHASE 3) uses the POST-attention state, so this
    // is an approximation. Fire off async pread for predicted cache misses via
    // dispatch_group so the I/O runs concurrently with CPU attention compute.
    // After CPU attention, we wait for the group to finish. When the real routing
    // happens later, predicted experts are already in the LRU cache as hits.

    dispatch_group_t spec_group = NULL;
    int spec_preload_count = 0;
    int spec_routing_enabled = 0;  // DISABLED: cache pollution + overhead makes it slower

    if (g_timing_enabled) { t0 = now_ms(); }
    s_spec_count = 0;

    if (spec_routing_enabled && (g_expert_cache || g_malloc_cache) && packed_fd >= 0 && lc->gate_w) {
        float *spec_scores = s_spec_gate_scores;
        memset(spec_scores, 0, cfg.num_experts * sizeof(float));

        // Gate projection matvec on pre-attention normed input (CPU, ~0.1ms for 512x4096)
        cpu_dequant_matvec(lc->gate_w, lc->gate_s, lc->gate_b,
                           normed, spec_scores,
                           cfg.num_experts, cfg.hidden_dim, cfg.group_size);
        cpu_softmax(spec_scores, cfg.num_experts);

        int spec_K = (K > MAX_K) ? MAX_K : K;
        float spec_weights[MAX_K];
        cpu_topk(spec_scores, cfg.num_experts, spec_K, s_spec_indices, spec_weights);
        s_spec_count = spec_K;

        g_spec_route_attempts += spec_K;

        // Initialize GCD queue if needed
        if (!g_io_gcd_queue)
            g_io_gcd_queue = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);

        // Check cache for each predicted expert, start async I/O for misses
        if (g_malloc_cache) {
            spec_group = dispatch_group_create();
            for (int k = 0; k < spec_K; k++) {
                int eidx = s_spec_indices[k];
                id<MTLBuffer> cached = malloc_cache_lookup(g_malloc_cache, layer_idx, eidx);
                if (!cached) {
                    int cidx = -1;
                    id<MTLBuffer> buf = malloc_cache_insert(g_malloc_cache, layer_idx, eidx, &cidx);
                    if (buf && cidx >= 0) {
                        int fd_copy = packed_fd;
                        void *dst = g_malloc_cache->data[cidx];
                        off_t eoff; size_t esz;
                        expert_offset_size(layer_idx, eidx, &eoff, &esz);
                        off_t offset = eoff;
                        size_t sz = esz;
                        dispatch_group_async(spec_group, g_io_gcd_queue, ^{
                            pread(fd_copy, dst, sz, offset);
                        });
                        spec_preload_count++;
                        g_spec_route_preloads++;
                    }
                }
            }
        } else if (g_expert_cache) {
            spec_group = dispatch_group_create();
            for (int k = 0; k < spec_K; k++) {
                int eidx = s_spec_indices[k];
                id<MTLBuffer> cached = expert_cache_lookup(g_expert_cache, layer_idx, eidx);
                if (!cached) {
                    id<MTLBuffer> buf = expert_cache_insert(g_expert_cache, layer_idx, eidx);
                    if (buf) {
                        int fd_copy = packed_fd;
                        void *dst = [buf contents];
                        off_t eoff; size_t esz;
                        expert_offset_size(layer_idx, eidx, &eoff, &esz);
                        off_t offset = eoff;
                        size_t sz = esz;
                        dispatch_group_async(spec_group, g_io_gcd_queue, ^{
                            pread(fd_copy, dst, sz, offset);
                        });
                        spec_preload_count++;
                        g_spec_route_preloads++;
                    }
                }
            }
        }
    }
    (void)spec_preload_count;  // tracked via g_spec_route_preloads

    if (g_timing_enabled) { t1 = now_ms(); g_timing.spec_route += t1 - t0; }

    // =====================================================================
    // PHASE 2: CPU attention compute
    // =====================================================================

    if (g_timing_enabled) { t0 = now_ms(); }

    float *attn_projected = s_attn_proj;
    memset(attn_projected, 0, cfg.hidden_dim * sizeof(float));

    // Pre-lookup o_proj / out_proj weights (used after attention compute)
    // These are looked up NOW to avoid repeated snprintf later.
    uint32_t *oproj_w = NULL;
    uint16_t *oproj_s = NULL, *oproj_b = NULL;
    int oproj_in_dim = 0;

    if (is_full) {
        oproj_w = lc->o_w; oproj_s = lc->o_s; oproj_b = lc->o_b;
        oproj_in_dim = cfg.num_attn_heads * cfg.head_dim;
    } else if (!linear_attn_bypass) {
        oproj_w = lc->out_proj_w; oproj_s = lc->out_proj_s; oproj_b = lc->out_proj_b;
        oproj_in_dim = cfg.linear_total_value;
    }

    // All MoE weight pointers from cache (zero snprintf overhead)
    uint32_t *gate_w = lc->gate_w; uint16_t *gate_s = lc->gate_s, *gate_b = lc->gate_b;
    uint32_t *sgw = lc->sg_w;     uint16_t *sgs = lc->sg_s,       *sgb = lc->sg_b;
    uint32_t *suw = lc->su_w;     uint16_t *sus = lc->su_s,       *sub = lc->su_b;
    uint32_t *seg_w = lc->seg_w;  uint16_t *seg_s = lc->seg_s,   *seg_b = lc->seg_b;
    uint32_t *sdw = lc->sd_w;     uint16_t *sds = lc->sd_s,       *sdb = lc->sd_b;

    // ---- CPU attention compute (produces attn_out for o_proj) ----
    float *attn_out_for_oproj = NULL;

    if (is_full) {
        // ---- Full attention CPU compute ----
        int q_proj_dim = cfg.num_attn_heads * cfg.head_dim * 2;
        int q_dim = cfg.num_attn_heads * cfg.head_dim;
        int kv_dim = cfg.num_kv_heads * cfg.head_dim;
        (void)q_proj_dim;

        float *q = s_q;
        float *q_gate = s_q_gate;
        for (int h = 0; h < cfg.num_attn_heads; h++) {
            float *src = q_proj_out + h * (2 * cfg.head_dim);
            memcpy(q + h * cfg.head_dim, src, cfg.head_dim * sizeof(float));
            memcpy(q_gate + h * cfg.head_dim, src + cfg.head_dim, cfg.head_dim * sizeof(float));
        }

        // Q/K RMSNorm
        uint16_t *qnorm_w = lc->q_norm_w;
        uint16_t *knorm_w = lc->k_norm_w;
        if (qnorm_w) {
            for (int h = 0; h < cfg.num_attn_heads; h++) {
                float *qh = q + h * cfg.head_dim;
                float sum_sq = 0.0f;
                for (int i = 0; i < cfg.head_dim; i++) sum_sq += qh[i] * qh[i];
                float inv_rms = 1.0f / sqrtf(sum_sq / cfg.head_dim + cfg.rms_norm_eps);
                for (int i = 0; i < cfg.head_dim; i++) qh[i] = qh[i] * inv_rms * bf16_to_f32(qnorm_w[i]);
            }
        }
        if (knorm_w) {
            for (int h = 0; h < cfg.num_kv_heads; h++) {
                float *kh = k_out + h * cfg.head_dim;
                float sum_sq = 0.0f;
                for (int i = 0; i < cfg.head_dim; i++) sum_sq += kh[i] * kh[i];
                float inv_rms = 1.0f / sqrtf(sum_sq / cfg.head_dim + cfg.rms_norm_eps);
                for (int i = 0; i < cfg.head_dim; i++) kh[i] = kh[i] * inv_rms * bf16_to_f32(knorm_w[i]);
            }
        }

        // RoPE
        apply_rotary_emb(q, k_out, pos, cfg.num_attn_heads, cfg.num_kv_heads, cfg.head_dim, cfg.rotary_dim);

        // Update KV cache (CPU + GPU mirror) — circular buffer for sliding window
        int cache_pos;
        if (kv->window_size > 0) {
            cache_pos = kv->len % kv->capacity;  // circular write
        } else {
            cache_pos = kv->len;
        }
        int fa_idx = cfg.full_attn_index[layer_idx];
        int kv_cache_ok = kv->use_fp8
            ? (kv->k_cache_fp8 != NULL && kv->v_cache_fp8 != NULL)
            : (kv->k_cache != NULL && kv->v_cache != NULL);
        if (!kv_cache_ok) {
            fprintf(stderr, "ERROR: KV cache is NULL at layer %d\n", layer_idx);
        } else if (kv->window_size == 0 && cache_pos >= kv->capacity) {
            fprintf(stderr, "ERROR: KV cache overflow at layer %d (pos=%d >= cap=%d)\n",
                    layer_idx, cache_pos, kv->capacity);
        } else {
            if (kv->use_fp8) {
                // FP8 CPU cache: quantize K and V
                kv->k_scales[cache_pos] = fp8_encode_vec(k_out, kv->k_cache_fp8 + cache_pos * kv_dim, kv_dim);
                kv->v_scales[cache_pos] = fp8_encode_vec(v_out, kv->v_cache_fp8 + cache_pos * kv_dim, kv_dim);
            } else {
                memcpy(kv->k_cache + cache_pos * kv_dim, k_out, kv_dim * sizeof(float));
                memcpy(kv->v_cache + cache_pos * kv_dim, v_out, kv_dim * sizeof(float));
            }
            // GPU mirror: always float32 for GPU attention kernels
            // When FP8, GPU still stores float32 (dequant happens on CPU write)
            if (g_metal && g_metal->attn_scores_pipe && fa_idx >= 0 && fa_idx < cfg.num_full_attn_layers) {
                if (kv->use_fp8) {
                    // GPU KV buffers are uchar when FP8 — write quantized data + scales
                    memcpy((uint8_t *)[g_metal->buf_kv_k[fa_idx] contents] + cache_pos * kv_dim,
                           kv->k_cache_fp8 + cache_pos * kv_dim, kv_dim * sizeof(uint8_t));
                    memcpy((uint8_t *)[g_metal->buf_kv_v[fa_idx] contents] + cache_pos * kv_dim,
                           kv->v_cache_fp8 + cache_pos * kv_dim, kv_dim * sizeof(uint8_t));
                    // Write per-position scales to GPU scale buffers
                    ((float *)[g_metal->buf_kv_k_scales[fa_idx] contents])[cache_pos] = kv->k_scales[cache_pos];
                    ((float *)[g_metal->buf_kv_v_scales[fa_idx] contents])[cache_pos] = kv->v_scales[cache_pos];
                } else {
                    memcpy((float *)[g_metal->buf_kv_k[fa_idx] contents] + cache_pos * kv_dim,
                           k_out, kv_dim * sizeof(float));
                    memcpy((float *)[g_metal->buf_kv_v[fa_idx] contents] + cache_pos * kv_dim,
                           v_out, kv_dim * sizeof(float));
                }
            }
            kv->len++;
        }

        // Scaled dot-product attention (GQA) — GPU or CPU
        int heads_per_kv = cfg.num_attn_heads / cfg.num_kv_heads;
        float scale = 1.0f / sqrtf((float)cfg.head_dim);
        float *attn_out = s_attn_out;
        memset(attn_out, 0, q_dim * sizeof(float));

        // Effective attention length (sliding window or full)
        int attn_seq_len;
        if (kv->window_size > 0 && kv->len > kv->window_size) {
            attn_seq_len = kv->window_size;
        } else {
            attn_seq_len = kv->len;
        }

        // GPU attention: defer dispatches to CMD2 (fused into single cmd buffer).
        // Only enabled when seq_len >= 32 (below that, CPU is faster).
        int gpu_attn_ready = (g_metal && g_metal->attn_scores_pipe &&
                              fa_idx >= 0 && fa_idx < cfg.num_full_attn_layers &&
                              attn_seq_len >= 32 && attn_seq_len < GPU_KV_SEQ);

        if (gpu_attn_ready) {
            // Copy Q and gate to GPU; attention dispatches will be in CMD2
            memcpy([g_metal->buf_attn_q contents], q, q_dim * sizeof(float));
            memcpy([g_metal->buf_attn_gate contents], q_gate, q_dim * sizeof(float));
            // attn_out_for_oproj will be set to NULL below — CMD2 reads buf_attn_out
        } else {
            // CPU fallback (supports both float32 and FP8 KV cache + sliding window)
            float *k_tmp = kv->use_fp8 ? s_k_dequant : NULL;
            float *v_tmp = kv->use_fp8 ? s_v_dequant : NULL;
            for (int h = 0; h < cfg.num_attn_heads; h++) {
                int kv_h = h / heads_per_kv;
                float *qh = q + h * cfg.head_dim;
                float *scores = malloc(attn_seq_len * sizeof(float));
                if (!scores) {
                    fprintf(stderr, "ERROR: attention scores alloc failed (len=%d)\n", attn_seq_len);
                    continue;
                }
                for (int i = 0; i < attn_seq_len; i++) {
                    int p;
                    if (kv->window_size > 0 && kv->len > kv->window_size) {
                        p = (kv->len - kv->window_size + i) % kv->capacity;
                    } else {
                        p = i;
                    }
                    float dot = 0.0f;
                    if (kv->use_fp8) {
                        fp8_decode_vec(kv->k_cache_fp8 + p * kv_dim, k_tmp, kv_dim, kv->k_scales[p]);
                        float *kp = k_tmp + kv_h * cfg.head_dim;
                        for (int d = 0; d < cfg.head_dim; d++) dot += qh[d] * kp[d];
                    } else {
                        float *kp = kv->k_cache + p * kv_dim + kv_h * cfg.head_dim;
                        for (int d = 0; d < cfg.head_dim; d++) dot += qh[d] * kp[d];
                    }
                    scores[i] = dot * scale;
                }
                cpu_softmax(scores, attn_seq_len);
                float *oh = attn_out + h * cfg.head_dim;
                for (int i = 0; i < attn_seq_len; i++) {
                    int p;
                    if (kv->window_size > 0 && kv->len > kv->window_size) {
                        p = (kv->len - kv->window_size + i) % kv->capacity;
                    } else {
                        p = i;
                    }
                    if (kv->use_fp8) {
                        fp8_decode_vec(kv->v_cache_fp8 + p * kv_dim, v_tmp, kv_dim, kv->v_scales[p]);
                        float *vp = v_tmp + kv_h * cfg.head_dim;
                        for (int d = 0; d < cfg.head_dim; d++) oh[d] += scores[i] * vp[d];
                    } else {
                        float *vp = kv->v_cache + p * kv_dim + kv_h * cfg.head_dim;
                        for (int d = 0; d < cfg.head_dim; d++) oh[d] += scores[i] * vp[d];
                    }
                }
                free(scores);
            }
            // k_tmp, v_tmp are static scratch buffers — no free needed
            for (int i = 0; i < q_dim; i++) {
                float g = 1.0f / (1.0f + expf(-q_gate[i]));
                attn_out[i] *= g;
            }
        }

        if (gpu_attn_ready) {
            attn_out_for_oproj = NULL;  // signal CMD2 to use GPU buf_attn_out
        } else {
            attn_out_for_oproj = attn_out;
        }
        // q_proj_out, k_out, v_out, q, q_gate, attn_out are static scratch.
    } else if (gpu_linear_attn) {
        // ---- GPU linear attention: already computed in CMD1 ----
        // batch_out[6] already contains gated_rms_norm output (8192 floats)
        // Set a non-NULL sentinel so CMD2 enters fused path, but skip the memcpy
        static float gpu_linear_sentinel;
        attn_out_for_oproj = &gpu_linear_sentinel;
    } else {
        // ---- Linear attention CPU compute ----
        if (!linear_attn_bypass) {
            int qkv_dim = cfg.linear_conv_dim;

            // Conv1d step
            uint16_t *conv_w = lc->conv1d_w;
            float *conv_out = s_conv_out;
            memset(conv_out, 0, qkv_dim * sizeof(float));
            if (conv_w) {
                cpu_conv1d_step(la_state->conv_state, qkv_out, conv_w, conv_out,
                                qkv_dim, cfg.conv_kernel_size);
            }
            // Update conv state
            memmove(la_state->conv_state, la_state->conv_state + qkv_dim,
                    (cfg.conv_kernel_size - 2) * qkv_dim * sizeof(float));
            memcpy(la_state->conv_state + (cfg.conv_kernel_size - 2) * qkv_dim, qkv_out,
                   qkv_dim * sizeof(float));

            // Split into q, k, v
            float *lin_q = conv_out;
            float *lin_k = conv_out + cfg.linear_total_key;
            float *lin_v = conv_out + 2 * cfg.linear_total_key;

            // RMS normalize q and k
            float inv_scale = 1.0f / sqrtf((float)cfg.linear_key_dim);
            for (int h = 0; h < cfg.linear_num_k_heads; h++) {
                float *qh = lin_q + h * cfg.linear_key_dim;
                cpu_rms_norm_bare(qh, qh, cfg.linear_key_dim, 1e-6f);
                float q_scale = inv_scale * inv_scale;
                for (int d = 0; d < cfg.linear_key_dim; d++) qh[d] *= q_scale;
            }
            for (int h = 0; h < cfg.linear_num_k_heads; h++) {
                float *kh = lin_k + h * cfg.linear_key_dim;
                cpu_rms_norm_bare(kh, kh, cfg.linear_key_dim, 1e-6f);
                for (int d = 0; d < cfg.linear_key_dim; d++) kh[d] *= inv_scale;
            }

            // Gated delta net recurrence
            float *A_log = lc->A_log;
            uint16_t *dt_bias_bf16 = lc->dt_bias;

            float *out_values = s_out_vals;
            memset(out_values, 0, cfg.linear_total_value * sizeof(float));
            int k_heads_per_v = cfg.linear_num_v_heads / cfg.linear_num_k_heads;

            float g_decay[cfg.linear_num_v_heads];
            float beta_gate_arr[cfg.linear_num_v_heads];
            for (int vh = 0; vh < cfg.linear_num_v_heads; vh++) {
                float a_val = alpha_out[vh];
                float dt_b = dt_bias_bf16 ? bf16_to_f32(dt_bias_bf16[vh]) : 0.0f;
                float A_val = A_log ? expf(A_log[vh]) : 1.0f;
                float softplus_val = logf(1.0f + expf(a_val + dt_b));
                g_decay[vh] = expf(-A_val * softplus_val);
                beta_gate_arr[vh] = cpu_sigmoid(beta_out[vh]);
            }

            // Compute linear_layer_idx: count of non-full-attention layers before this one.
            // Full attention at (layer_idx+1) % 4 == 0, i.e. layers 3,7,11,...
            // linear_layer_idx = layer_idx - number_of_full_layers_at_or_before
            //                  = cfg.linear_index[layer_idx]
            int linear_layer_idx = cfg.linear_index[layer_idx];

            // GPU delta-net path (falls back to CPU if pipeline unavailable)
            if (g_metal && g_metal->delta_net_step &&
                linear_layer_idx >= 0 && linear_layer_idx < cfg.num_linear_layers) {
                // Upload CPU-computed data to GPU scratch buffers
                memcpy([g_metal->buf_delta_q contents], lin_q, cfg.linear_total_key * sizeof(float));
                memcpy([g_metal->buf_delta_k contents], lin_k, cfg.linear_total_key * sizeof(float));
                memcpy([g_metal->buf_delta_v contents], lin_v, cfg.linear_total_value * sizeof(float));
                memcpy([g_metal->buf_delta_g_decay contents], g_decay, cfg.linear_num_v_heads * sizeof(float));
                memcpy([g_metal->buf_delta_beta contents], beta_gate_arr, cfg.linear_num_v_heads * sizeof(float));

                id<MTLCommandBuffer> cmd_dn = [g_metal->queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd_dn computeCommandEncoder];
                [enc setComputePipelineState:g_metal->delta_net_step];
                [enc setBuffer:g_metal->buf_delta_state[linear_layer_idx] offset:0 atIndex:0];
                [enc setBuffer:g_metal->buf_delta_q       offset:0 atIndex:1];
                [enc setBuffer:g_metal->buf_delta_k       offset:0 atIndex:2];
                [enc setBuffer:g_metal->buf_delta_v       offset:0 atIndex:3];
                [enc setBuffer:g_metal->buf_delta_g_decay offset:0 atIndex:4];
                [enc setBuffer:g_metal->buf_delta_beta    offset:0 atIndex:5];
                [enc setBuffer:g_metal->buf_delta_output  offset:0 atIndex:6];
                uint32_t khpv = (uint32_t)k_heads_per_v;
                [enc setBytes:&khpv length:sizeof(khpv) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(cfg.linear_num_v_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [enc endEncoding];
                [cmd_dn commit];
                [cmd_dn waitUntilCompleted];

                // Read back GPU result
                memcpy(out_values, [g_metal->buf_delta_output contents], cfg.linear_total_value * sizeof(float));
            } else {
                // CPU delta-net with Accelerate BLAS
                for (int vh = 0; vh < cfg.linear_num_v_heads; vh++) {
                    int kh = vh / k_heads_per_v;
                    float g = g_decay[vh];
                    float b_gate = beta_gate_arr[vh];
                    float *S = la_state->ssm_state + vh * cfg.linear_value_dim * cfg.linear_key_dim;
                    float *v_h = lin_v + vh * cfg.linear_value_dim;
                    float *k_h = lin_k + kh * cfg.linear_key_dim;

                    // Step 1: Decay S *= g (BLAS sscal on entire state matrix)
                    cblas_sscal(cfg.linear_value_dim * cfg.linear_key_dim, g, S, 1);

                    // Step 2: kv_mem = S @ k (each row dot k)
                    // S is [VALUE_DIM x KEY_DIM] row-major, k is [KEY_DIM]
                    // kv_mem[vi] = sum_ki(S[vi,ki] * k[ki]) = matrix-vector: S @ k
                    float kv_mem_vec[cfg.linear_value_dim];
                    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                                cfg.linear_value_dim, cfg.linear_key_dim,
                                1.0f, S, cfg.linear_key_dim, k_h, 1,
                                0.0f, kv_mem_vec, 1);

                    // Step 3: delta = (v - kv_mem) * beta, then rank-1 update S += k * delta^T
                    // delta[vi] = (v[vi] - kv_mem[vi]) * beta
                    float delta_vec[cfg.linear_value_dim];
                    for (int vi = 0; vi < cfg.linear_value_dim; vi++) {
                        delta_vec[vi] = (v_h[vi] - kv_mem_vec[vi]) * b_gate;
                    }
                    // S += delta @ k^T (rank-1 update: sger)
                    // S[vi,ki] += delta[vi] * k[ki]
                    cblas_sger(CblasRowMajor, cfg.linear_value_dim, cfg.linear_key_dim,
                               1.0f, delta_vec, 1, k_h, 1, S, cfg.linear_key_dim);

                    // Step 4: output = S @ q (matrix-vector multiply)
                    float *q_h = lin_q + kh * cfg.linear_key_dim;
                    float *o_h = out_values + vh * cfg.linear_value_dim;
                    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                                cfg.linear_value_dim, cfg.linear_key_dim,
                                1.0f, S, cfg.linear_key_dim, q_h, 1,
                                0.0f, o_h, 1);
                }
            }

            // RMSNormGated
            uint16_t *gated_norm_w = lc->gated_norm_w;
            float *gated_out = s_gated_out;
            memset(gated_out, 0, cfg.linear_total_value * sizeof(float));
            for (int vh = 0; vh < cfg.linear_num_v_heads; vh++) {
                float *oh = out_values + vh * cfg.linear_value_dim;
                float *zh = z_out + vh * cfg.linear_value_dim;
                float *gh = gated_out + vh * cfg.linear_value_dim;
                if (gated_norm_w) {
                    cpu_rms_norm_gated(oh, zh, gated_norm_w, gh, cfg.linear_value_dim, cfg.rms_norm_eps);
                } else {
                    memcpy(gh, oh, cfg.linear_value_dim * sizeof(float));
                }
            }

            attn_out_for_oproj = gated_out;

            // conv_out, out_values are static — no free needed
            // gated_out is static — freed/released after CMD2 submission below
        }
        // else: linear_attn_bypass — attn_projected stays zero
        // qkv_out, z_out, beta_out, alpha_out are static scratch.
    }

    // =====================================================================
    // PHASE 3: FULLY FUSED CMD2 — o_proj + residual + norm + routing (1 cmd buffer)
    //   Eliminates 1 GPU round-trip vs old 2-buffer approach.
    //   GPU handles residual_add + rms_norm between o_proj and routing,
    //   so no CPU intervention is needed. 8 encoders, 1 commit+wait.
    //   Buffer flow: batch_out[6]->buf_output->buf_h_mid->buf_input->batch_out[0-3]
    // =====================================================================

    if (g_timing_enabled) { t1 = now_ms(); g_timing.cpu_attn += t1 - t0; }

    // Wait for speculative expert I/O to complete (overlapped with CPU attention)
    if (spec_group) {
        dispatch_group_wait(spec_group, DISPATCH_TIME_FOREVER);
        spec_group = NULL;  // ARC releases the group
    }

    if (g_timing_enabled) { t0 = now_ms(); }

    float *h_post = s_h_post;
    float *h_mid = s_h_mid;
    float *gate_scores = s_gate_scores;
    float *shared_gate = s_shared_gate;
    float *shared_up = s_shared_up;
    float shared_gate_score;
    if (cmd1_cmd2_merged) {
        // Merged path: results already in s_* arrays from gpu_flush in CMD1 readback
        shared_gate_score = g_merged_shared_gate_score;
    } else {
        memset(gate_scores, 0, cfg.num_experts * sizeof(float));
        memset(shared_gate, 0, cfg.shared_intermediate * sizeof(float));
        memset(shared_up, 0, cfg.shared_intermediate * sizeof(float));
        shared_gate_score = 0.0f;
    }

    int have_moe_weights = (gate_w && gate_s && gate_b && sgw && sgs && sgb &&
                            suw && sus && sub && seg_w && seg_s && seg_b);

    // gpu_attn_fuse: attention dispatches fused into CMD2 (full-attn layers only).
    // Only enabled when seq_len >= 32 — below that, CPU attention is faster
    // because GPU command encoder overhead dominates at short sequences.
    int gpu_attn_fuse = (is_full && !attn_out_for_oproj && g_metal && g_metal->attn_scores_pipe
                         && kv && kv->len >= 32 && kv->len < GPU_KV_SEQ);

    if (!cmd1_cmd2_merged &&
        (attn_out_for_oproj || gpu_attn_fuse) && oproj_w && oproj_s && oproj_b &&
        g_metal && g_metal->wf_buf && have_moe_weights &&
        g_metal->residual_add && g_metal->rms_norm_sum &&
        g_metal->rms_norm_apply_bf16 && lc->post_attn_norm_w) {
        // ---- FULLY FUSED CMD2 ----
        // For GPU attention (full-attn layers): attention dispatches are prepended,
        //   o_proj reads from buf_attn_out instead of batch_out[6].
        // For CPU attention / linear attn: o_proj reads from batch_out[6] as before.
        //
        // GPU attn path (12 encoders):
        //   Enc 1-4: attn_scores + softmax + values + sigmoid -> buf_attn_out
        //   Enc 5:   o_proj (buf_attn_out -> buf_output)
        //   Enc 6-8: residual + norm -> buf_input
        //   Enc 9-12: routing + shared expert
        //
        // CPU attn path (8 encoders, unchanged):
        //   Enc 1:   o_proj (batch_out[6] -> buf_output)
        //   Enc 2-4: residual + norm -> buf_input
        //   Enc 5-8: routing + shared expert

        if (!gpu_attn_fuse && !gpu_linear_attn) {
            // CPU/linear attn: copy attention output to GPU input buffer
            memcpy([g_metal->batch_out[6] contents], attn_out_for_oproj,
                   oproj_in_dim * sizeof(float));
        }
        // gpu_linear_attn: batch_out[6] already has the result from CMD1 gated_rms_norm
        // Copy residual into GPU buffer for residual_add kernel
        memcpy([g_metal->buf_residual contents], residual, cfg.hidden_dim * sizeof(float));

        attn_out_for_oproj = NULL;

        id<MTLCommandBuffer> cmd_fused = [g_metal->queue commandBuffer];

        // ---- GPU attention dispatches (only for full-attn layers with GPU path) ----
        if (gpu_attn_fuse) {
            int fa_idx = cfg.full_attn_index[layer_idx];
            int kv_dim = cfg.num_kv_heads * cfg.head_dim;
            int heads_per_kv = cfg.num_attn_heads / cfg.num_kv_heads;
            float scale = 1.0f / sqrtf((float)cfg.head_dim);
            uint32_t hd = cfg.head_dim;
            uint32_t kvd = (uint32_t)kv_dim;
            uint32_t sl = (uint32_t)kv->len;
            uint32_t seq_stride = GPU_KV_SEQ;
            uint32_t hpkv = (uint32_t)heads_per_kv;

            // Try fused online softmax attention (single kernel replaces 3-kernel pipeline)
            // Priority: function-constant specialized > type-specific > 3-kernel fallback
            int use_fused = 0;
            if (g_fused_attention_enabled && g_metal->fused_attention_fc_pipe) {
                // Function-constant specialized path: single kernel handles both FP8 and float32.
                // Buffer layout is unified: buffers 1-4 are K_cache, K_scales, V_cache, V_scales.
                // For float32 mode, K_scales and V_scales are unused (compiler eliminated FP8 code).
                id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
                [enc setComputePipelineState:g_metal->fused_attention_fc_pipe];
                [enc setBuffer:g_metal->buf_attn_q           offset:0 atIndex:0];
                [enc setBuffer:g_metal->buf_kv_k[fa_idx]     offset:0 atIndex:1];
                // K_scales: use real scales for FP8, dummy buffer for float32
                if (g_use_fp8_kv && g_metal->buf_kv_k_scales[fa_idx]) {
                    [enc setBuffer:g_metal->buf_kv_k_scales[fa_idx] offset:0 atIndex:2];
                } else {
                    [enc setBuffer:g_metal->buf_attn_q offset:0 atIndex:2];  // dummy, not read
                }
                [enc setBuffer:g_metal->buf_kv_v[fa_idx]     offset:0 atIndex:3];
                if (g_use_fp8_kv && g_metal->buf_kv_v_scales[fa_idx]) {
                    [enc setBuffer:g_metal->buf_kv_v_scales[fa_idx] offset:0 atIndex:4];
                } else {
                    [enc setBuffer:g_metal->buf_attn_q offset:0 atIndex:4];  // dummy, not read
                }
                [enc setBuffer:g_metal->buf_attn_out         offset:0 atIndex:5];
                uint32_t nh = (uint32_t)cfg.num_attn_heads;
                uint32_t nkvh = (uint32_t)cfg.num_kv_heads;
                [enc setBytes:&hd        length:4 atIndex:6];
                [enc setBytes:&kvd       length:4 atIndex:7];
                [enc setBytes:&sl        length:4 atIndex:8];
                [enc setBytes:&nh        length:4 atIndex:9];
                [enc setBytes:&nkvh      length:4 atIndex:10];
                [enc setBytes:&scale     length:4 atIndex:11];
                [enc dispatchThreadgroups:MTLSizeMake(cfg.num_attn_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
                use_fused = 1;
            } else if (g_use_fp8_kv && g_metal->fused_attention_fp8_pipe) {
                // Fused FP8 path: Q, K_fp8, K_scales, V_fp8, V_scales, out
                id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
                [enc setComputePipelineState:g_metal->fused_attention_fp8_pipe];
                [enc setBuffer:g_metal->buf_attn_q              offset:0 atIndex:0];
                [enc setBuffer:g_metal->buf_kv_k[fa_idx]        offset:0 atIndex:1];
                [enc setBuffer:g_metal->buf_kv_k_scales[fa_idx] offset:0 atIndex:2];
                [enc setBuffer:g_metal->buf_kv_v[fa_idx]        offset:0 atIndex:3];
                [enc setBuffer:g_metal->buf_kv_v_scales[fa_idx] offset:0 atIndex:4];
                [enc setBuffer:g_metal->buf_attn_out            offset:0 atIndex:5];
                uint32_t nh = (uint32_t)cfg.num_attn_heads;
                uint32_t nkvh = (uint32_t)cfg.num_kv_heads;
                [enc setBytes:&hd        length:4 atIndex:6];
                [enc setBytes:&kvd       length:4 atIndex:7];
                [enc setBytes:&sl        length:4 atIndex:8];
                [enc setBytes:&nh        length:4 atIndex:9];
                [enc setBytes:&nkvh      length:4 atIndex:10];
                [enc setBytes:&scale     length:4 atIndex:11];
                [enc dispatchThreadgroups:MTLSizeMake(cfg.num_attn_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
                use_fused = 1;
            } else if (!g_use_fp8_kv && g_metal->fused_attention_pipe) {
                // Fused float32 path: Q, K, V, out
                id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
                [enc setComputePipelineState:g_metal->fused_attention_pipe];
                [enc setBuffer:g_metal->buf_attn_q           offset:0 atIndex:0];
                [enc setBuffer:g_metal->buf_kv_k[fa_idx]     offset:0 atIndex:1];
                [enc setBuffer:g_metal->buf_kv_v[fa_idx]     offset:0 atIndex:2];
                [enc setBuffer:g_metal->buf_attn_out         offset:0 atIndex:3];
                uint32_t nh = (uint32_t)cfg.num_attn_heads;
                uint32_t nkvh = (uint32_t)cfg.num_kv_heads;
                [enc setBytes:&hd        length:4 atIndex:4];
                [enc setBytes:&kvd       length:4 atIndex:5];
                [enc setBytes:&sl        length:4 atIndex:6];
                [enc setBytes:&nh        length:4 atIndex:7];
                [enc setBytes:&nkvh      length:4 atIndex:8];
                [enc setBytes:&scale     length:4 atIndex:9];
                [enc dispatchThreadgroups:MTLSizeMake(cfg.num_attn_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
                use_fused = 1;
            }

            if (!use_fused) {
            // Fallback: 3-kernel attention pipeline (scores -> softmax -> values)
            // Enc A1: attn_scores_batched (float32 or FP8 variant)
            if (g_use_fp8_kv && g_metal->attn_scores_fp8_pipe) {
                id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
                [enc setComputePipelineState:g_metal->attn_scores_fp8_pipe];
                [enc setBuffer:g_metal->buf_attn_q              offset:0 atIndex:0];
                [enc setBuffer:g_metal->buf_kv_k[fa_idx]        offset:0 atIndex:1];  // uchar KV
                [enc setBuffer:g_metal->buf_kv_k_scales[fa_idx] offset:0 atIndex:2];  // per-pos scales
                [enc setBuffer:g_metal->buf_attn_scores         offset:0 atIndex:3];
                [enc setBytes:&hd        length:4 atIndex:4];
                [enc setBytes:&kvd       length:4 atIndex:5];
                [enc setBytes:&sl        length:4 atIndex:6];
                [enc setBytes:&seq_stride length:4 atIndex:7];
                [enc setBytes:&scale     length:4 atIndex:8];
                [enc setBytes:&hpkv      length:4 atIndex:9];
                [enc setBytes:&sl        length:4 atIndex:10];
                uint32_t total_tgs = sl * cfg.num_attn_heads;
                [enc dispatchThreadgroups:MTLSizeMake(total_tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            } else {
                id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
                [enc setComputePipelineState:g_metal->attn_scores_pipe];
                [enc setBuffer:g_metal->buf_attn_q          offset:0 atIndex:0];
                [enc setBuffer:g_metal->buf_kv_k[fa_idx]    offset:0 atIndex:1];
                [enc setBuffer:g_metal->buf_attn_scores     offset:0 atIndex:2];
                [enc setBytes:&hd        length:4 atIndex:3];
                [enc setBytes:&kvd       length:4 atIndex:4];
                [enc setBytes:&sl        length:4 atIndex:5];
                [enc setBytes:&seq_stride length:4 atIndex:6];
                [enc setBytes:&scale     length:4 atIndex:7];
                [enc setBytes:&hpkv      length:4 atIndex:8];
                [enc setBytes:&sl        length:4 atIndex:9];
                uint32_t total_tgs = sl * cfg.num_attn_heads;
                [enc dispatchThreadgroups:MTLSizeMake(total_tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
            // Enc A2: attn_softmax_batched (unchanged — operates on float scores)
            {
                id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
                [enc setComputePipelineState:g_metal->attn_softmax_pipe];
                [enc setBuffer:g_metal->buf_attn_scores offset:0 atIndex:0];
                [enc setBytes:&sl         length:4 atIndex:1];
                [enc setBytes:&seq_stride  length:4 atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(cfg.num_attn_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
            // Enc A3: attn_values_batched (float32 or FP8 variant)
            if (g_use_fp8_kv && g_metal->attn_values_fp8_pipe) {
                id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
                [enc setComputePipelineState:g_metal->attn_values_fp8_pipe];
                [enc setBuffer:g_metal->buf_attn_scores         offset:0 atIndex:0];
                [enc setBuffer:g_metal->buf_kv_v[fa_idx]        offset:0 atIndex:1];  // uchar KV
                [enc setBuffer:g_metal->buf_kv_v_scales[fa_idx] offset:0 atIndex:2];  // per-pos scales
                [enc setBuffer:g_metal->buf_attn_out            offset:0 atIndex:3];
                [enc setBytes:&hd        length:4 atIndex:4];
                [enc setBytes:&kvd       length:4 atIndex:5];
                [enc setBytes:&sl        length:4 atIndex:6];
                [enc setBytes:&seq_stride length:4 atIndex:7];
                [enc setBytes:&hpkv      length:4 atIndex:8];
                uint32_t total_threads = cfg.head_dim * cfg.num_attn_heads;
                uint32_t tgs = (total_threads + 255) / 256;
                [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            } else {
                id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
                [enc setComputePipelineState:g_metal->attn_values_pipe];
                [enc setBuffer:g_metal->buf_attn_scores   offset:0 atIndex:0];
                [enc setBuffer:g_metal->buf_kv_v[fa_idx]  offset:0 atIndex:1];
                [enc setBuffer:g_metal->buf_attn_out      offset:0 atIndex:2];
                [enc setBytes:&hd        length:4 atIndex:3];
                [enc setBytes:&kvd       length:4 atIndex:4];
                [enc setBytes:&sl        length:4 atIndex:5];
                [enc setBytes:&seq_stride length:4 atIndex:6];
                [enc setBytes:&hpkv      length:4 atIndex:7];
                uint32_t total_threads = cfg.head_dim * cfg.num_attn_heads;
                uint32_t tgs = (total_threads + 255) / 256;
                [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
            } // end !use_fused fallback
            // Enc A4: sigmoid_gate
            {
                uint32_t qdim = cfg.num_attn_heads * cfg.head_dim;
                id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
                [enc setComputePipelineState:g_metal->sigmoid_gate_pipe];
                [enc setBuffer:g_metal->buf_attn_out  offset:0 atIndex:0];
                [enc setBuffer:g_metal->buf_attn_gate offset:0 atIndex:1];
                [enc setBytes:&qdim length:4 atIndex:2];
                uint32_t tgs = (qdim + 255) / 256;
                [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
        }

        // ---- o_proj matvec ----
        {
            // For GPU attention: o_proj reads from buf_attn_out
            // For CPU attention: o_proj reads from batch_out[6]
            id<MTLBuffer> oproj_input = gpu_attn_fuse ? g_metal->buf_attn_out : g_metal->batch_out[6];

            uint32_t o_out_dim = cfg.hidden_dim;
            uint32_t o_in_dim = (uint32_t)oproj_in_dim;
            uint32_t o_gs = cfg.group_size;
            id<MTLBuffer> ow_buf, os_buf, ob_buf;
            NSUInteger ow_off, os_off, ob_off;
            size_t oproj_w_size = (size_t)o_out_dim * o_in_dim / 8;  // 4-bit packed
            size_t oproj_ng = (o_in_dim + o_gs - 1) / o_gs;
            size_t oproj_sb_size = (size_t)o_out_dim * oproj_ng * sizeof(uint16_t);
            metal_staging_reset(g_metal);
            metal_find_chunk_sized(g_metal, oproj_w, oproj_w_size, &ow_buf, &ow_off);
            metal_find_chunk_sized(g_metal, oproj_s, oproj_sb_size, &os_buf, &os_off);
            metal_find_chunk_sized(g_metal, oproj_b, oproj_sb_size, &ob_buf, &ob_off);
            id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
            [enc setComputePipelineState:g_metal->matvec_fast];
            [enc setBuffer:ow_buf offset:ow_off atIndex:0];
            [enc setBuffer:os_buf offset:os_off atIndex:1];
            [enc setBuffer:ob_buf offset:ob_off atIndex:2];
            [enc setBuffer:oproj_input      offset:0    atIndex:3];
            [enc setBuffer:g_metal->buf_output offset:0 atIndex:4];
            [enc setBytes:&o_out_dim  length:4 atIndex:5];
            [enc setBytes:&o_in_dim   length:4 atIndex:6];
            [enc setBytes:&o_gs       length:4 atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(o_out_dim, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [enc endEncoding];
        }

        // ---- Enc 2: residual_add (buf_output + buf_residual -> buf_h_mid) ----
        {
            id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
            uint32_t dim = cfg.hidden_dim;
            [enc setComputePipelineState:g_metal->residual_add];
            [enc setBuffer:g_metal->buf_residual offset:0 atIndex:0];  // a = residual
            [enc setBuffer:g_metal->buf_output   offset:0 atIndex:1];  // b = o_proj result
            [enc setBuffer:g_metal->buf_h_mid    offset:0 atIndex:2];  // out = h_mid
            [enc setBytes:&dim length:4 atIndex:3];
            uint32_t tgs = (dim + 255) / 256;
            [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        // ---- Enc 3: rms_norm_sum_sq (buf_h_mid -> buf_sum_sq) ----
        {
            id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
            uint32_t dim = cfg.hidden_dim;
            [enc setComputePipelineState:g_metal->rms_norm_sum];
            [enc setBuffer:g_metal->buf_h_mid  offset:0 atIndex:0];
            [enc setBuffer:g_metal->buf_sum_sq offset:0 atIndex:1];
            [enc setBytes:&dim length:4 atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        // ---- Enc 4: rms_norm_apply_bf16 (buf_h_mid + norm_w -> buf_input) ----
        {
            id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
            id<MTLBuffer> panw_buf; NSUInteger panw_off;
            // Don't reset staging — o_proj data is still being read by earlier encoders in cmd_fused
            metal_find_chunk_sized(g_metal, lc->post_attn_norm_w,
                (size_t)cfg.hidden_dim * 2, &panw_buf, &panw_off);  // bf16
            uint32_t dim = cfg.hidden_dim;
            float eps = cfg.rms_norm_eps;
            [enc setComputePipelineState:g_metal->rms_norm_apply_bf16];
            [enc setBuffer:g_metal->buf_h_mid  offset:0       atIndex:0];  // x
            [enc setBuffer:panw_buf offset:panw_off atIndex:1]; // weight (bf16)
            [enc setBuffer:g_metal->buf_sum_sq offset:0       atIndex:2];  // sum_sq
            [enc setBuffer:g_metal->buf_input  offset:0       atIndex:3];  // out = h_post
            [enc setBytes:&dim length:4 atIndex:4];
            [enc setBytes:&eps length:4 atIndex:5];
            uint32_t tgs = (dim + 255) / 256;
            [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        // ---- Enc 5-8: routing + shared expert projections (read buf_input) ----
        BatchMatvecSpec moe_specs[4] = {
            { gate_w, gate_s, gate_b, gate_scores,        (uint32_t)cfg.num_experts,        cfg.hidden_dim, cfg.group_size, 0 },
            { sgw,    sgs,    sgb,    shared_gate,         (uint32_t)cfg.shared_intermediate, cfg.hidden_dim, cfg.group_size, 1 },
            { suw,    sus,    sub,    shared_up,           (uint32_t)cfg.shared_intermediate, cfg.hidden_dim, cfg.group_size, 2 },
            { seg_w,  seg_s,  seg_b,  &shared_gate_score,  1,                            cfg.hidden_dim, cfg.group_size, 3 },
        };
        // buf_input already contains h_post from Enc 4 output -- no memcpy needed
        gpu_encode_batch_matvec(g_metal, cmd_fused, moe_specs, 4);

        if (g_timing_enabled) { t1 = now_ms(); g_timing.cmd2_encode += t1 - t0; }

        // ---- Single commit+wait for all 8 encoders ----
        if (g_timing_enabled) { t0 = now_ms(); }
        [cmd_fused commit];
        [cmd_fused waitUntilCompleted];

        // Read back results
        gpu_flush_batch_results(g_metal, moe_specs, 4);
        // Read h_mid from GPU buffer (needed for final combine)
        memcpy(h_mid, [g_metal->buf_h_mid contents], cfg.hidden_dim * sizeof(float));
        // Read h_post from buf_input (needed for expert input)
        memcpy(h_post, [g_metal->buf_input contents], cfg.hidden_dim * sizeof(float));
        // Update hidden state to h_mid (= residual + o_proj)
        memcpy(hidden, h_mid, cfg.hidden_dim * sizeof(float));
        if (g_timing_enabled) { t1 = now_ms(); g_timing.cmd2_wait += t1 - t0; }

    } else if (!cmd1_cmd2_merged) {
        // ---- Non-fused fallback path ----
        // Skipped when cmd1_cmd2_merged: the merge path already computed o_proj,
        // residual, norm, and routing on GPU within CMD1. Running the fallback
        // would clobber those results (and read garbage via attn_out_for_oproj
        // which points to a sentinel, not real attention output).
        // O projection
        if (attn_out_for_oproj && oproj_w && oproj_s && oproj_b) {
            fast_dequant_matvec(oproj_w, oproj_s, oproj_b, attn_out_for_oproj,
                                attn_projected, cfg.hidden_dim, oproj_in_dim, cfg.group_size);
        }
        // attn_out_for_oproj is static — no free needed
        attn_out_for_oproj = NULL;

        // Residual connection
        for (int i = 0; i < cfg.hidden_dim; i++) {
            hidden[i] = residual[i] + attn_projected[i];
        }
        // attn_projected, normed, residual are static — no free needed

        cpu_vec_copy(h_mid, hidden, cfg.hidden_dim);

        // Post-attention norm
        cpu_rms_norm(hidden, lc->post_attn_norm_w, h_post, cfg.hidden_dim, cfg.rms_norm_eps);

        // Routing + shared expert batch
        if (have_moe_weights) {
            BatchMatvecSpec moe_specs[4] = {
                { gate_w, gate_s, gate_b, gate_scores,        (uint32_t)cfg.num_experts,        cfg.hidden_dim, cfg.group_size, 0 },
                { sgw,    sgs,    sgb,    shared_gate,         (uint32_t)cfg.shared_intermediate, cfg.hidden_dim, cfg.group_size, 1 },
                { suw,    sus,    sub,    shared_up,           (uint32_t)cfg.shared_intermediate, cfg.hidden_dim, cfg.group_size, 2 },
                { seg_w,  seg_s,  seg_b,  &shared_gate_score,  1,                            cfg.hidden_dim, cfg.group_size, 3 },
            };
            fast_batch_matvec(h_post, cfg.hidden_dim, moe_specs, 4);
        }
        if (g_timing_enabled) { t1 = now_ms(); g_timing.cmd2_encode += t1 - t0; }
    }

    // ---- Softmax + top-K (CPU) ----
    if (g_timing_enabled) { t0 = now_ms(); }
    cpu_softmax(gate_scores, cfg.num_experts);
    int expert_indices[64];
    float expert_weights[64];
    cpu_topk(gate_scores, cfg.num_experts, K, expert_indices, expert_weights);
    cpu_normalize_weights(expert_weights, K);
    if (g_freq_tracking) {
        for (int k = 0; k < K; k++) {
            FREQ(layer_idx, expert_indices[k])++;
        }
        if (layer_idx == 0) g_freq_total_tokens++;
    }

    // Track speculative routing prediction accuracy
    if (s_spec_count > 0) {
        int cmp_K = (K > MAX_K) ? MAX_K : K;
        for (int s = 0; s < s_spec_count; s++) {
            for (int r = 0; r < cmp_K; r++) {
                if (s_spec_indices[s] == expert_indices[r]) {
                    g_spec_route_hits++;
                    break;
                }
            }
        }
    }

    if (g_timing_enabled) { t1 = now_ms(); g_timing.routing_cpu += t1 - t0; }

    // Log routing data for predictor training
    if (g_routing_log) {
        int32_t li = layer_idx;
        int32_t ki = (K > MAX_K) ? MAX_K : K;
        fwrite(&li, sizeof(int32_t), 1, g_routing_log);
        fwrite(&ki, sizeof(int32_t), 1, g_routing_log);
        fwrite(hidden, sizeof(float), cfg.hidden_dim, g_routing_log);
        fwrite(expert_indices, sizeof(int32_t), ki, g_routing_log);
        g_routing_log_samples++;
    }

    // ---- Parallel pread + GPU experts ----
    if (g_timing_enabled) { t0 = now_ms(); }
    float *moe_out = s_moe_out;
    memset(moe_out, 0, cfg.hidden_dim * sizeof(float));
    float *shared_out = s_shared_out;
    memset(shared_out, 0, cfg.hidden_dim * sizeof(float));

    int actual_K = (K > MAX_K) ? MAX_K : K;

    if (packed_fd >= 0 && g_metal && g_metal->buf_multi_expert_data[0]) {
        // GPU multi-expert path with LRU cache + parallel I/O:
        // For each expert:
        //   - Cache HIT:  dispatch directly from cached Metal buffer (skip pread)
        //   - Cache MISS: pread into cache buffer, then dispatch from it
        // Falls back to original parallel_pread_experts when cache is disabled.

        int valid[MAX_K];
        id<MTLBuffer> expert_bufs[MAX_K];  // buffer to dispatch from per expert

        // Cross-layer prefetch: check if we have pre-loaded experts from the previous layer's CMD3.
        // If the prefetch targeted THIS layer and completed, match hits into valid[]/expert_bufs[].
        int prefetch_hits = 0;
        int prefetch_handled __attribute__((unused)) = 0;  // 1 if cross-layer prefetch handled all I/O
        if (g_prefetch_active && g_prefetch_layer == layer_idx) {
            async_pread_wait();
            g_prefetch_active = 0;
            g_prefetch_layer = -1;

            for (int k = 0; k < actual_K; k++) {
                valid[k] = 0;
                for (int p = 0; p < g_async_pread.num_experts; p++) {
                    if (expert_indices[k] == PRED_EXPERT(layer_idx, p) && g_async_pread.valid[p]) {
                        expert_bufs[k] = g_metal->buf_multi_expert_data_B[p];
                        valid[k] = 1;
                        prefetch_hits++;
                        break;
                    }
                }
            }
            g_prefetch_hits_total += prefetch_hits;
            g_prefetch_misses_total += (actual_K - prefetch_hits);

            if (prefetch_hits == actual_K) {
                // All experts pre-loaded — skip I/O entirely
                prefetch_handled = 1;
                pred_started = 0;  // no prediction path needed
                goto experts_loaded;
            } else if (prefetch_hits > 0) {
                // Partial hits: pread misses into buf_A, keep hits in buf_B
                int miss_ei[MAX_K];
                int miss_k_slots[MAX_K];
                int miss_count = 0;
                for (int k = 0; k < actual_K; k++) {
                    if (!valid[k]) {
                        miss_ei[miss_count] = expert_indices[k];
                        miss_k_slots[miss_count] = k;
                        expert_bufs[k] = g_metal->buf_multi_expert_data[k];
                        miss_count++;
                    }
                }
                if (miss_count > 0) {
                    InferPreadTask tasks[MAX_K];
                    for (int m = 0; m < miss_count; m++) {
                        int k = miss_k_slots[m];
                        off_t eoff; size_t esz;
                        expert_offset_size(layer_idx, miss_ei[m], &eoff, &esz);
                        tasks[m].fd = packed_fd;
                        tasks[m].dst = [g_metal->buf_multi_expert_data[k] contents];
                        tasks[m].offset = eoff;
                        tasks[m].size = esz;
                        tasks[m].result = 0;
                        tasks[m].mmap_base = mmap_base;
                        tasks[m].lz4_comp_buf = NULL;
                        tasks[m].lz4_comp_size = 0;
                    }
                    io_pool_dispatch(tasks, miss_count);
                    for (int m = 0; m < miss_count; m++) {
                        int k = miss_k_slots[m];
                        valid[k] = (tasks[m].result == (ssize_t)tasks[m].size);
                    }
                }
                prefetch_handled = 1;
                pred_started = 0;
                goto experts_loaded;
            }
            // prefetch_hits == 0: fall through to normal I/O paths
        }
        // Drain stale prefetch targeting a different layer (shouldn't happen in normal flow)
        if (g_prefetch_active && g_prefetch_layer != layer_idx) {
            async_pread_wait();
            g_prefetch_active = 0;
            g_prefetch_layer = -1;
        }

        if (g_malloc_cache) {
            // ---- Malloc cache path (zero-copy Metal buffer wrappers) ----
            // Phase 1: check cache for each expert, collect misses
            int miss_indices[MAX_K];
            int miss_cache_idx[MAX_K];  // cache entry index for each miss
            int num_misses = 0;

            for (int k = 0; k < actual_K; k++) {
                id<MTLBuffer> cached = malloc_cache_lookup(g_malloc_cache, layer_idx, expert_indices[k]);
                if (cached) {
                    // Cache hit: zero-copy dispatch directly from cache buffer
                    expert_bufs[k] = cached;
                    valid[k] = 1;
                } else {
                    // Cache miss: insert entry (get buffer to pread into)
                    int cidx = -1;
                    id<MTLBuffer> buf = malloc_cache_insert(g_malloc_cache, layer_idx, expert_indices[k], &cidx);
                    expert_bufs[k] = buf;
                    miss_indices[num_misses] = k;
                    miss_cache_idx[num_misses] = cidx;
                    num_misses++;
                    valid[k] = 0;
                }
            }

            // Phase 2: parallel pread misses directly into cache buffers (zero-copy)
            if (num_misses > 0) {
                InferPreadTask tasks[MAX_K];
                for (int m = 0; m < num_misses; m++) {
                    int k = miss_indices[m];
                    int cidx = miss_cache_idx[m];
                    off_t eoff; size_t esz;
                    expert_offset_size(layer_idx, expert_indices[k], &eoff, &esz);
                    tasks[m].fd = expert_pick_fd(layer_idx, expert_indices[k], packed_fd);
                    tasks[m].dst = g_malloc_cache->data[cidx];
                    tasks[m].offset = eoff;
                    tasks[m].size = esz;
                    tasks[m].result = 0;
                    tasks[m].mmap_base = NULL;  // always pread for cache population
                }

                io_pool_dispatch(tasks, num_misses);

                // Mark valid
                for (int m = 0; m < num_misses; m++) {
                    int k = miss_indices[m];
                    valid[k] = (tasks[m].result == (ssize_t)tasks[m].size);
                    if (!valid[k]) {
                        fprintf(stderr, "WARNING: expert %d pread: %zd/%zu\n",
                                expert_indices[k], tasks[m].result, tasks[m].size);
                    }
                }
            }
        } else if (g_expert_cache) {
            // ---- Metal buffer LRU cache path ----
            // Phase 1: check cache for each expert, collect misses
            int miss_indices[MAX_K];       // indices into expert_indices[] for misses
            id<MTLBuffer> miss_bufs[MAX_K]; // cache buffers to pread into
            int num_misses = 0;

            for (int k = 0; k < actual_K; k++) {
                id<MTLBuffer> cached = expert_cache_lookup(g_expert_cache, layer_idx, expert_indices[k]);
                if (cached) {
                    // Cache hit: use this buffer directly for GPU dispatch
                    expert_bufs[k] = cached;
                    valid[k] = 1;
                } else {
                    // Cache miss: insert into cache (allocates or evicts), will pread below
                    id<MTLBuffer> buf = expert_cache_insert(g_expert_cache, layer_idx, expert_indices[k]);
                    if (buf) {
                        expert_bufs[k] = buf;
                        miss_indices[num_misses] = k;
                        miss_bufs[num_misses] = buf;
                        num_misses++;
                        valid[k] = 0;  // not yet loaded
                    } else {
                        expert_bufs[k] = nil;
                        valid[k] = 0;
                    }
                }
            }

            // Phase 2: parallel pread all cache misses
            if (num_misses > 0) {
                InferPreadTask tasks[MAX_K];
                for (int m = 0; m < num_misses; m++) {
                    int k = miss_indices[m];
                    off_t eoff; size_t esz;
                    expert_offset_size(layer_idx, expert_indices[k], &eoff, &esz);
                    tasks[m].fd = expert_pick_fd(layer_idx, expert_indices[k], packed_fd);
                    tasks[m].dst = [miss_bufs[m] contents];
                    tasks[m].offset = eoff;
                    tasks[m].size = esz;
                    tasks[m].result = 0;
                    tasks[m].mmap_base = mmap_base;
                }

                io_pool_dispatch(tasks, num_misses);

                // Mark successfully loaded misses as valid
                for (int m = 0; m < num_misses; m++) {
                    int k = miss_indices[m];
                    valid[k] = (tasks[m].result == (ssize_t)tasks[m].size);
                    if (!valid[k]) {
                        fprintf(stderr, "WARNING: expert %d pread: %zd/%zu\n",
                                expert_indices[k], tasks[m].result, tasks[m].size);
                    }
                }
            }
        } else if (pred_started) {
            // ---- Prediction path: predicted experts already loading into buf_B ----
            // Wait for predicted preads (they've had ~1.6ms: CMD1_wait + attn + CMD2 + routing)
            async_pread_wait();
            g_pred_layers++;

            // Match predictions against actual routing
            int miss_ei[MAX_K];       // actual expert indices for misses
            int miss_k_slots[MAX_K];  // which k-slot each miss maps to
            int miss_count = 0;
            int hit_count = 0;

            for (int k = 0; k < actual_K; k++) {
                int found = 0;
                for (int p = 0; p < PRED_COUNT(layer_idx); p++) {
                    if (expert_indices[k] == PRED_EXPERT(layer_idx, p) &&
                        g_async_pread.valid[p]) {
                        // Hit! This expert was pre-loaded into buf_B[p]
                        expert_bufs[k] = g_metal->buf_multi_expert_data_B[p];
                        valid[k] = 1;
                        found = 1;
                        hit_count++;
                        break;
                    }
                }
                if (!found) {
                    miss_ei[miss_count] = expert_indices[k];
                    miss_k_slots[miss_count] = k;
                    expert_bufs[k] = g_metal->buf_multi_expert_data[k];
                    miss_count++;
                }
            }
            g_pred_hits += hit_count;
            g_pred_misses += miss_count;

            // Parallel sync-pread misses into buf_A
            if (miss_count > 0) {
                InferPreadTask tasks[MAX_K];
                for (int m = 0; m < miss_count; m++) {
                    int k = miss_k_slots[m];
                    off_t eoff; size_t esz;
                    expert_offset_size(layer_idx, miss_ei[m], &eoff, &esz);
                    tasks[m].fd = packed_fd;
                    tasks[m].dst = [g_metal->buf_multi_expert_data[k] contents];
                    tasks[m].offset = eoff;
                    tasks[m].size = esz;
                    tasks[m].result = 0;
                }
                io_pool_dispatch(tasks, miss_count);
                for (int m = 0; m < miss_count; m++) {
                    int k = miss_k_slots[m];
                    valid[k] = (tasks[m].result == (ssize_t)tasks[m].size);
                }
            }
        } else if (g_use_lz4 && g_lz4_index[layer_idx]) {
            // ---- LZ4 compressed path: read compressed + decompress via io_pool ----
            // Note: LZ4 + tiered is not supported (LZ4 path uses its own offsets)
            size_t esz = active_expert_size();
            InferPreadTask tasks[MAX_K];
            for (int k = 0; k < actual_K; k++) {
                LZ4IndexEntry *ie = &g_lz4_index[layer_idx][expert_indices[k]];
                tasks[k].fd = packed_fd;
                tasks[k].dst = [g_metal->buf_multi_expert_data[k] contents];
                tasks[k].offset = ie->offset;
                tasks[k].size = esz;
                tasks[k].result = 0;
                tasks[k].mmap_base = NULL;
                tasks[k].lz4_comp_buf = g_lz4_comp_bufs[k];
                tasks[k].lz4_comp_size = ie->comp_size;
                expert_bufs[k] = g_metal->buf_multi_expert_data[k];
            }
            io_pool_dispatch(tasks, actual_K);
            for (int k = 0; k < actual_K; k++) {
                valid[k] = (tasks[k].result == (ssize_t)esz);
            }
        } else {
            // ---- No cache, no prediction, no LZ4: ASYNC parallel pread ----
            async_pread_start(packed_fd, expert_indices, actual_K,
                              g_metal->buf_multi_expert_data, mmap_base,
                              layer_idx);
            for (int k = 0; k < actual_K; k++) {
                expert_bufs[k] = g_metal->buf_multi_expert_data[k];
            }
        }

        experts_loaded:  // Cross-layer prefetch jumps here when all experts are pre-loaded
        // Shared expert prep (doesn't need expert data — can overlap with async pread)
        memcpy([g_metal->buf_multi_expert_input contents], h_post, cfg.hidden_dim * sizeof(float));
        memcpy([g_metal->buf_shared_gate contents], shared_gate,
               cfg.shared_intermediate * sizeof(float));
        memcpy([g_metal->buf_shared_up contents], shared_up,
               cfg.shared_intermediate * sizeof(float));

        // Wait for non-prediction async pread to complete
        if (!pred_started && g_async_pread.active) {
            async_pread_wait();
            for (int k = 0; k < actual_K; k++) {
                valid[k] = g_async_pread.valid[k];
            }
        }

        if (g_timing_enabled) { t1 = now_ms(); g_timing.expert_io += t1 - t0; }

        // Store this layer's routing for next token's temporal prediction.
        // MUST happen AFTER the prediction hit check above (which reads g_pred_experts).
        // Also store when expert_prefetch is enabled (cross-layer prefetch needs predictions).
        if ((g_pred_enabled || g_expert_prefetch_enabled) && g_pred_generating) {
            for (int k = 0; k < actual_K; k++) {
                PRED_EXPERT(layer_idx, k) = expert_indices[k];
            }
            PRED_COUNT(layer_idx) = actual_K;
            if (layer_idx == cfg.num_layers - 1) {
                g_pred_valid = 1;
            }
        }

        if (g_timing_enabled) { t0 = now_ms(); }

        // Step 3: encode ALL experts + shared expert into ONE command buffer.
        // Batched encoding: 4 encoders for K experts + 2 for shared = 6 total
        // (vs. 4*K + 2 = 18 with old per-expert encoding).
        id<MTLCommandBuffer> cmd_experts = [g_metal->queue commandBuffer];

        gpu_encode_experts_batched(g_metal, cmd_experts, actual_K, valid, expert_bufs,
                                   layer_idx, expert_indices);

        // Shared expert SwiGLU + down_proj (2 more encoders)
        // Note: shared_gate/up already copied to GPU buffers above (before async pread wait)

        // SwiGLU dispatch
        {
            id<MTLComputeCommandEncoder> enc = [cmd_experts computeCommandEncoder];
            [enc setComputePipelineState:g_metal->swiglu];
            [enc setBuffer:g_metal->buf_shared_gate offset:0 atIndex:0];
            [enc setBuffer:g_metal->buf_shared_up   offset:0 atIndex:1];
            [enc setBuffer:g_metal->buf_shared_act  offset:0 atIndex:2];
            uint32_t dim = cfg.shared_intermediate;
            [enc setBytes:&dim length:4 atIndex:3];
            uint32_t swiglu_tgs = (dim + 255) / 256;
            [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        // Shared down_proj dispatch
        if (sdw && sds && sdb) {
            if (g_metal->wf_num_chunks > 0) {
                // GPU path: weight data accessible via Metal buffer
                gpu_encode_dequant_matvec_with_io_bufs(
                    g_metal, cmd_experts, sdw, sds, sdb,
                    g_metal->buf_shared_act, g_metal->buf_shared_out,
                    cfg.hidden_dim, cfg.shared_intermediate, cfg.group_size);
            } else {
                // CPU fallback: weight file too large for Metal buffers
                float *shared_act_cpu = (float *)[g_metal->buf_shared_act contents];
                float *shared_out_cpu = (float *)[g_metal->buf_shared_out contents];
                cpu_dequant_matvec((const uint32_t *)sdw, (const uint16_t *)sds,
                                   (const uint16_t *)sdb, shared_act_cpu, shared_out_cpu,
                                   cfg.hidden_dim, cfg.shared_intermediate, cfg.group_size);
            }
        }

        // Step 4: GPU-side combine + residual + norm (if not last layer)
        // Appends dispatches to CMD3 so the next layer's CMD1 can submit immediately
        // without waiting for CMD3 to complete + CPU readback.
        //
        // For non-last layers with the combine pipeline available:
        //   Enc C1: moe_combine_residual (expert_outs + h_mid + shared_out -> buf_moe_hidden)
        //   Enc C2: rms_norm_sum_sq (buf_moe_hidden -> buf_cmd3_sum_sq)
        //   Enc C3: rms_norm_apply_bf16 (buf_moe_hidden + next_layer_norm_w -> buf_input)
        //
        // This makes CMD3 self-contained: it produces buf_input for the next layer's CMD1.
        // The next layer skips deferred_wait + finalize + input_norm entirely at layer start.

        int gpu_combine = (g_metal->moe_combine_residual &&
                           g_metal->rms_norm_sum &&
                           g_metal->rms_norm_apply_bf16 &&
                           g_metal->wf_buf &&
                           layer_idx < cfg.num_layers - 1 &&
                           layer_cache[layer_idx + 1].input_norm_w != NULL);

        if (gpu_combine) {
            // Copy h_mid from buf_h_mid (populated by CMD2) — it's still valid on GPU.
            // h_mid is already in buf_h_mid from CMD2's residual_add dispatch.

            // Prepare combine params: expert_weights[0..K-1] + shared_gate_score
            {
                float *params = (float *)[g_metal->buf_combine_params contents];
                // Zero all 10 slots first (unused experts get weight=0)
                memset(params, 0, 10 * sizeof(float));
                for (int k = 0; k < actual_K; k++) {
                    params[k] = valid[k] ? expert_weights[k] : 0.0f;
                }
                params[8] = shared_gate_score;
            }

            // Enc C1: moe_combine_residual
            {
                id<MTLComputeCommandEncoder> enc = [cmd_experts computeCommandEncoder];
                [enc setComputePipelineState:g_metal->moe_combine_residual];
                [enc setBuffer:g_metal->buf_h_mid         offset:0 atIndex:0];   // h_mid
                [enc setBuffer:g_metal->buf_shared_out    offset:0 atIndex:1];   // shared_out
                [enc setBuffer:g_metal->buf_moe_hidden    offset:0 atIndex:2];   // output: hidden
                // Bind all 8 expert output buffers (unused ones have weight=0 in params)
                for (int k = 0; k < MAX_K; k++) {
                    [enc setBuffer:g_metal->buf_multi_expert_out[k] offset:0 atIndex:(3 + k)];
                }
                [enc setBuffer:g_metal->buf_combine_params offset:0 atIndex:11]; // params
                uint32_t dim = cfg.hidden_dim;
                uint32_t k_val = (uint32_t)actual_K;
                [enc setBytes:&dim   length:4 atIndex:12];
                [enc setBytes:&k_val length:4 atIndex:13];
                uint32_t tgs = (dim + 255) / 256;
                [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }

            // Enc C2: rms_norm_sum_sq (buf_moe_hidden -> buf_cmd3_sum_sq)
            {
                id<MTLComputeCommandEncoder> enc = [cmd_experts computeCommandEncoder];
                uint32_t dim = cfg.hidden_dim;
                [enc setComputePipelineState:g_metal->rms_norm_sum];
                [enc setBuffer:g_metal->buf_moe_hidden  offset:0 atIndex:0];
                [enc setBuffer:g_metal->buf_cmd3_sum_sq offset:0 atIndex:1];
                [enc setBytes:&dim length:4 atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }

            // Enc C3: rms_norm_apply_bf16 (buf_moe_hidden + next_norm_w -> buf_input)
            {
                uint16_t *next_norm_w = layer_cache[layer_idx + 1].input_norm_w;
                id<MTLBuffer> nnw_buf; NSUInteger nnw_off;
                metal_staging_reset(g_metal);
                metal_find_chunk_sized(g_metal, next_norm_w,
                    (size_t)cfg.hidden_dim * 2, &nnw_buf, &nnw_off);  // bf16
                id<MTLComputeCommandEncoder> enc = [cmd_experts computeCommandEncoder];
                uint32_t dim = cfg.hidden_dim;
                float eps = cfg.rms_norm_eps;
                [enc setComputePipelineState:g_metal->rms_norm_apply_bf16];
                [enc setBuffer:g_metal->buf_moe_hidden  offset:0       atIndex:0]; // x
                [enc setBuffer:nnw_buf offset:nnw_off atIndex:1]; // weight (bf16)
                [enc setBuffer:g_metal->buf_cmd3_sum_sq offset:0       atIndex:2]; // sum_sq
                [enc setBuffer:g_metal->buf_input       offset:0       atIndex:3]; // out = normed
                [enc setBytes:&dim length:4 atIndex:4];
                [enc setBytes:&eps length:4 atIndex:5];
                uint32_t tgs = (dim + 255) / 256;
                [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
        }

        // DEFERRED commit — submit async, don't wait.
        [cmd_experts commit];
        if (g_timing_enabled) {
            t1 = now_ms();
            g_timing.cmd3_encode += t1 - t0;
            g_timing.count++;
            g_timing.total += t1 - t_layer_start;
        }

        // Save state for deferred completion
        g_deferred.active = 1;
        g_deferred.gpu_combined = gpu_combine;
        g_deferred.cmd_experts = cmd_experts;
        g_deferred.actual_K = actual_K;
        g_deferred.shared_gate_score = shared_gate_score;
        g_deferred.hidden = hidden;
        g_deferred.layer_idx = layer_idx;
        if (!gpu_combine) {
            // Only need to save h_mid for CPU-side combine path
            memcpy(g_deferred.h_mid, h_mid, cfg.hidden_dim * sizeof(float));
        }
        for (int k = 0; k < actual_K; k++) {
            g_deferred.expert_weights[k] = expert_weights[k];
            g_deferred.valid[k] = valid[k];
        }

        // Cross-layer prefetch: start pread'ing next layer's predicted experts into Set B.
        // CMD3 GPU execution overlaps with this I/O (~2.4ms of GPU time to hide behind).
        // Uses temporal prediction: last token's expert choices for layer N+1.
        if (g_expert_prefetch_enabled && g_layer_fds_global &&
            layer_idx + 1 < cfg.num_layers &&
            g_pred_count && PRED_COUNT(layer_idx + 1) > 0 &&
            !g_prefetch_active && g_pred_generating && g_pred_valid) {
            int next = layer_idx + 1;
            int next_fd = g_layer_fds_global[next];
            void *next_mmap = (g_layer_mmaps_global && g_layer_mmaps_global[next] != MAP_FAILED)
                               ? g_layer_mmaps_global[next] : NULL;
            if (next_fd >= 0 && g_metal->buf_multi_expert_data_B[0]) {
                async_pread_start(next_fd, &PRED_EXPERT(next, 0),
                                  PRED_COUNT(next),
                                  g_metal->buf_multi_expert_data_B,
                                  next_mmap, next);
                g_prefetch_active = 1;
                g_prefetch_layer = next;
            }
        }

        // Return immediately — GPU experts are running async.
        // The next call to fused_layer_forward() or complete_deferred_experts()
        // will wait for the GPU and apply the final combine.
        return;

    } else if (packed_fd >= 0) {
        // CPU fallback for experts (uses pre-allocated scratch buffers)
        float *expert_out_cpu = s_expert_out_cpu;
        for (int k = 0; k < K; k++) {
            int eidx = expert_indices[k];
            off_t expert_offset; size_t esz;
            expert_offset_size(layer_idx, eidx, &expert_offset, &esz);
            void *expert_data = malloc(esz);
            if (!expert_data) {
                fprintf(stderr, "WARNING: layer %d expert %d malloc(%zu) failed, skipping\n",
                        layer_idx, eidx, esz);
                continue;
            }
            ssize_t nread = pread(packed_fd, expert_data, esz, expert_offset);
            if (nread != (ssize_t)esz) {
                fprintf(stderr, "WARNING: layer %d expert %d pread: %zd/%zu\n",
                        layer_idx, eidx, nread, esz);
                free(expert_data);
                continue;
            }

            // CPU fallback offsets — determine quant per expert
            int use_2bit_k = g_use_2bit;
            if (g_use_tiered && g_tiered_manifest)
                use_2bit_k = (TIERED(layer_idx, eidx).bits == 2);
            uint32_t *gw = (uint32_t *)expert_data;
            uint16_t *gs_p = (uint16_t *)((char *)expert_data + (use_2bit_k ? cfg.gate_s_off_2 : cfg.gate_s_off_4));
            uint16_t *gb_p = (uint16_t *)((char *)expert_data + (use_2bit_k ? cfg.gate_b_off_2 : cfg.gate_b_off_4));
            uint32_t *uw = (uint32_t *)((char *)expert_data + (use_2bit_k ? cfg.up_w_off_2 : cfg.up_w_off_4));
            uint16_t *us_p = (uint16_t *)((char *)expert_data + (use_2bit_k ? cfg.up_s_off_2 : cfg.up_s_off_4));
            uint16_t *ub_p = (uint16_t *)((char *)expert_data + (use_2bit_k ? cfg.up_b_off_2 : cfg.up_b_off_4));
            uint32_t *dw = (uint32_t *)((char *)expert_data + (use_2bit_k ? cfg.down_w_off_2 : cfg.down_w_off_4));
            uint16_t *ds_p = (uint16_t *)((char *)expert_data + (use_2bit_k ? cfg.down_s_off_2 : cfg.down_s_off_4));
            uint16_t *db_p = (uint16_t *)((char *)expert_data + (use_2bit_k ? cfg.down_b_off_2 : cfg.down_b_off_4));

            float *gate_proj_out = s_gate_proj_out;
            float *up_proj_out = s_up_proj_out;
            float *act_out = s_act_out;

            cpu_dequant_matvec(gw, gs_p, gb_p, h_post, gate_proj_out,
                               cfg.moe_intermediate, cfg.hidden_dim, cfg.group_size);
            cpu_dequant_matvec(uw, us_p, ub_p, h_post, up_proj_out,
                               cfg.moe_intermediate, cfg.hidden_dim, cfg.group_size);
            cpu_swiglu(gate_proj_out, up_proj_out, act_out, cfg.moe_intermediate);
            cpu_dequant_matvec(dw, ds_p, db_p, act_out, expert_out_cpu,
                               cfg.hidden_dim, cfg.moe_intermediate, cfg.group_size);

            free(expert_data);

            cpu_vec_madd(moe_out, expert_out_cpu, expert_weights[k], cfg.hidden_dim);
        }

        // CPU shared expert (uses pre-allocated scratch)
        memset(s_shared_act, 0, cfg.shared_intermediate * sizeof(float));
        cpu_swiglu(shared_gate, shared_up, s_shared_act, cfg.shared_intermediate);
        if (sdw && sds && sdb) {
            cpu_dequant_matvec(sdw, sds, sdb, s_shared_act, shared_out,
                               cfg.hidden_dim, cfg.shared_intermediate, cfg.group_size);
        }
    } else {
        // No experts available -- still need shared expert (uses pre-allocated scratch)
        memset(s_shared_act, 0, cfg.shared_intermediate * sizeof(float));
        cpu_swiglu(shared_gate, shared_up, s_shared_act, cfg.shared_intermediate);
        if (sdw && sds && sdb) {
            fast_dequant_matvec(sdw, sds, sdb, s_shared_act, shared_out,
                                cfg.hidden_dim, cfg.shared_intermediate, cfg.group_size);
        }
    }

    // ---- Shared expert gate ----
    float shared_weight = cpu_sigmoid(shared_gate_score);
    for (int i = 0; i < cfg.hidden_dim; i++) {
        shared_out[i] *= shared_weight;
    }

    // ---- Final combine: hidden = h_mid + moe_out + shared_out ----
    for (int i = 0; i < cfg.hidden_dim; i++) {
        hidden[i] = h_mid[i] + moe_out[i] + shared_out[i];
    }

    if (g_timing_enabled) {
        t1 = now_ms();
        g_timing.cmd3_encode += t1 - t0;  // includes CPU expert compute for non-GPU paths
        g_timing.count++;
        g_timing.total += t1 - t_layer_start;
    }

    // h_post, h_mid, gate_scores, moe_out, shared_out, shared_gate, shared_up
    // are all static scratch buffers — no free needed.
}
