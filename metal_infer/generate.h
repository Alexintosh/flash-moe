// generate.h — Frequency analysis, tokenization helpers, HTTP serve mode, main()
// Part of Flash-MoE modular decomposition (unity build).
// Included by infer.m — do NOT compile separately.

// ============================================================================
// Main inference loop
// ============================================================================

// ============================================================================
// Expert frequency analysis (--freq)
// ============================================================================

static int freq_cmp_desc(const void *a, const void *b) {
    return *(const int *)b - *(const int *)a;
}

static void freq_print_analysis(int K) {
    if (!g_freq_tracking || g_freq_total_tokens == 0) return;

    int total_activations_per_layer = g_freq_total_tokens * K;

    fprintf(stderr, "\n=== Expert Frequency Analysis ===\n");
    fprintf(stderr, "Tokens tracked: %d, K=%d, activations/layer=%d\n\n",
            g_freq_total_tokens, K, total_activations_per_layer);

    // Per-layer analysis
    int experts_for_80_total = 0;  // sum across layers for overall estimate

    for (int l = 0; l < cfg.num_layers; l++) {
        // Count unique experts and sort frequencies descending
        int sorted[cfg.num_experts];
        memcpy(sorted, &FREQ(l, 0), cfg.num_experts * sizeof(int));
        qsort(sorted, cfg.num_experts, sizeof(int), freq_cmp_desc);

        int unique = 0;
        for (int e = 0; e < cfg.num_experts; e++) {
            if (sorted[e] > 0) unique++;
        }

        // Compute cumulative coverage thresholds
        int cum = 0;
        int top10_cov = 0, top30_cov = 0, top60_cov = 0;
        int n_for_50 = 0, n_for_80 = 0, n_for_90 = 0;
        for (int e = 0; e < cfg.num_experts; e++) {
            cum += sorted[e];
            if (e == 9)  top10_cov = cum;
            if (e == 29) top30_cov = cum;
            if (e == 59) top60_cov = cum;
            if (n_for_50 == 0 && cum * 100 >= total_activations_per_layer * 50)
                n_for_50 = e + 1;
            if (n_for_80 == 0 && cum * 100 >= total_activations_per_layer * 80)
                n_for_80 = e + 1;
            if (n_for_90 == 0 && cum * 100 >= total_activations_per_layer * 90)
                n_for_90 = e + 1;
        }

        double pct10 = 100.0 * top10_cov / total_activations_per_layer;
        double pct30 = 100.0 * top30_cov / total_activations_per_layer;
        double pct60 = 100.0 * top60_cov / total_activations_per_layer;

        fprintf(stderr, "Layer %2d: %3d unique experts, "
                "top-10 cover %.0f%%, top-30 cover %.0f%%, top-60 cover %.0f%% "
                "(50%%@%d, 80%%@%d, 90%%@%d)\n",
                l, unique, pct10, pct30, pct60, n_for_50, n_for_80, n_for_90);

        experts_for_80_total += n_for_80;
    }

    // Overall summary: average experts needed for 80% across all layers
    double avg_experts_80 = (double)experts_for_80_total / cfg.num_layers;
    // Expert size in GB: each expert is active_expert_size() bytes
    double expert_gb = (double)active_expert_size() / (1024.0 * 1024.0 * 1024.0);
    double total_pin_gb = avg_experts_80 * cfg.num_layers * expert_gb;

    fprintf(stderr, "\n--- Overall Summary ---\n");
    fprintf(stderr, "To achieve 80%% hit rate across all layers, need %d experts pinned "
            "(avg %.0f/layer, %.2f GB)\n",
            experts_for_80_total, avg_experts_80, total_pin_gb);
    fprintf(stderr, "Expert size: %zu bytes (%.3f MB), %d layers x %d experts = %d total\n",
            active_expert_size(), (double)active_expert_size() / (1024.0 * 1024.0),
            cfg.num_layers, cfg.num_experts, cfg.num_layers * cfg.num_experts);

    // Raw frequency dump for profile_experts.py
    fprintf(stderr, "\n--- Raw Frequency Dump (for profile_experts.py) ---\n");
    for (int l = 0; l < cfg.num_layers; l++) {
        fprintf(stderr, "FREQ_DUMP layer=%d:", l);
        for (int e = 0; e < cfg.num_experts; e++) {
            int f = FREQ(l, e);
            if (f > 0) fprintf(stderr, " %d:%d", e, f);
        }
        fprintf(stderr, "\n");
    }

    // JSON frequency dump (--freq-json FILE)
    if (g_freq_json_path) {
        FILE *jf = fopen(g_freq_json_path, "w");
        if (jf) {
            fprintf(jf, "{\n  \"tokens\": %d,\n  \"K\": %d,\n  \"layers\": {\n",
                    g_freq_total_tokens, K);
            for (int l = 0; l < cfg.num_layers; l++) {
                // Sort experts by frequency descending for this layer
                typedef struct { int idx; int freq; } EF;
                EF sorted_ef[cfg.num_experts];
                for (int e = 0; e < cfg.num_experts; e++) {
                    sorted_ef[e].idx = e;
                    sorted_ef[e].freq = FREQ(l, e);
                }
                // Simple insertion sort (only ~256-512 experts)
                for (int i = 1; i < cfg.num_experts; i++) {
                    EF tmp = sorted_ef[i];
                    int j = i - 1;
                    while (j >= 0 && sorted_ef[j].freq < tmp.freq) {
                        sorted_ef[j+1] = sorted_ef[j]; j--;
                    }
                    sorted_ef[j+1] = tmp;
                }
                fprintf(jf, "    \"%d\": [", l);
                int first = 1;
                for (int e = 0; e < cfg.num_experts; e++) {
                    if (sorted_ef[e].freq > 0) {
                        if (!first) fprintf(jf, ", ");
                        fprintf(jf, "%d", sorted_ef[e].idx);
                        first = 0;
                    }
                }
                fprintf(jf, "]%s\n", l < cfg.num_layers - 1 ? "," : "");
            }
            fprintf(jf, "  }\n}\n");
            fclose(jf);
            fprintf(stderr, "\nFrequency data written to %s\n", g_freq_json_path);
        }
    }
}

// Tokenize a continuation turn (available in both CLI and iOS modes).
// Prefixes with \n<|im_start|>user\n to start new turn, assumes prior assistant
// turn's EOS/<|im_end|> is already in the KV cache state.
static PromptTokens *tokenize_continuation_turn_shared(const char *user_content) {
    const char *prefix = "\n<|im_start|>user\n";
    const char *suffix = "<|im_end|>\n<|im_start|>assistant\n";

    size_t prompt_len = strlen(prefix) + strlen(user_content) + strlen(suffix) + 1;
    char *prompt = malloc(prompt_len);
    if (!prompt) return NULL;
    snprintf(prompt, prompt_len, "%s%s%s", prefix, user_content, suffix);
    PromptTokens *pt = encode_prompt_text_to_tokens(prompt);
    free(prompt);
    return pt;
}

#ifndef CHAT_MODE

// ============================================================================
// HTTP Serve Mode — OpenAI-compatible /v1/chat/completions (SSE streaming)
// ============================================================================

// Read exactly n bytes from fd, returns 0 on success, -1 on error/EOF
static int read_exact(int fd, char *buf, int n) {
    int got = 0;
    while (got < n) {
        ssize_t r = read(fd, buf + got, n - got);
        if (r <= 0) return -1;
        got += (int)r;
    }
    return 0;
}

// Read HTTP request into buf (up to bufsz-1). Returns total bytes read, or -1.
// Reads headers, then Content-Length body if present.
static int read_http_request(int fd, char *buf, int bufsz) {
    int total = 0;
    // Read until we find \r\n\r\n (end of headers)
    while (total < bufsz - 1) {
        ssize_t r = read(fd, buf + total, 1);
        if (r <= 0) return -1;
        total++;
        if (total >= 4 &&
            buf[total-4] == '\r' && buf[total-3] == '\n' &&
            buf[total-2] == '\r' && buf[total-1] == '\n') {
            break;
        }
    }
    buf[total] = '\0';

    // Find Content-Length
    const char *cl = strcasestr(buf, "Content-Length:");
    if (cl) {
        int content_len = atoi(cl + 15);
        if (content_len > 0 && total + content_len < bufsz - 1) {
            if (read_exact(fd, buf + total, content_len) < 0) return -1;
            total += content_len;
            buf[total] = '\0';
        }
    }
    return total;
}

// Extract the last "content" value from an OpenAI messages array.
// Minimal JSON parsing: find last "content":" and extract the string value.
// Returns pointer into buf (null-terminated in place), or NULL.
static char *extract_last_content(char *buf) {
    char *last = NULL;
    char *p = buf;
    for (;;) {
        p = strstr(p, "\"content\"");
        if (!p) break;
        p += 9; // skip "content"
        // Skip whitespace and colon
        while (*p == ' ' || *p == '\t' || *p == ':') p++;
        if (*p == '"') {
            p++; // skip opening quote
            last = p;
            // Find closing quote (handle escapes)
            while (*p && !(*p == '"' && *(p-1) != '\\')) p++;
        }
    }
    if (last) {
        // Null-terminate the content string (overwrite closing quote)
        char *end = last;
        while (*end && !(*end == '"' && (end == last || *(end-1) != '\\'))) end++;
        *end = '\0';
        // Unescape \\n -> \n, \\" -> ", \\\\ -> backslash inline
        char *r = last, *w = last;
        while (*r) {
            if (*r == '\\' && *(r+1)) {
                r++;
                switch (*r) {
                    case 'n':  *w++ = '\n'; r++; break;
                    case 't':  *w++ = '\t'; r++; break;
                    case '"':  *w++ = '"';  r++; break;
                    case '\\': *w++ = '\\'; r++; break;
                    default:   *w++ = '\\'; *w++ = *r++; break;
                }
            } else {
                *w++ = *r++;
            }
        }
        *w = '\0';
    }
    return last;
}

// Extract "max_tokens" or "max_completion_tokens" from JSON body. Returns value or default.
static int extract_max_tokens(const char *buf, int default_val) {
    const char *p = strstr(buf, "\"max_completion_tokens\"");
    if (!p) p = strstr(buf, "\"max_tokens\"");
    if (!p) return default_val;
    p = strchr(p, ':');
    if (!p) return default_val;
    return atoi(p + 1);
}

// Save a conversation turn to ~/.flash-moe/sessions/<session_id>.jsonl
// Shared data store with the chat client.
static void server_save_turn(const char *session_id, const char *role, const char *content) {
    if (!session_id || !session_id[0] || !content) return;
    const char *home = getenv("HOME");
    if (!home) home = "/tmp";
    char dir[1024], path[1024];
    snprintf(dir, sizeof(dir), "%s/.flash-moe/sessions", home);
    mkdir(dir, 0755);
    char parent[1024];
    snprintf(parent, sizeof(parent), "%s/.flash-moe", home);
    mkdir(parent, 0755);
    mkdir(dir, 0755);
    snprintf(path, sizeof(path), "%s/%s.jsonl", dir, session_id);
    FILE *f = fopen(path, "a");
    if (!f) return;
    // JSON-escape content
    size_t clen = strlen(content);
    char *escaped = malloc(clen * 2 + 1);
    int j = 0;
    for (size_t i = 0; i < clen; i++) {
        switch (content[i]) {
            case '"': escaped[j++]='\\'; escaped[j++]='"'; break;
            case '\\': escaped[j++]='\\'; escaped[j++]='\\'; break;
            case '\n': escaped[j++]='\\'; escaped[j++]='n'; break;
            case '\r': escaped[j++]='\\'; escaped[j++]='r'; break;
            case '\t': escaped[j++]='\\'; escaped[j++]='t'; break;
            default: escaped[j++]=content[i]; break;
        }
    }
    escaped[j] = 0;
    fprintf(f, "{\"role\":\"%s\",\"content\":\"%s\"}\n", role, escaped);
    free(escaped);
    fclose(f);
}

// Extract "session_id" string from JSON body. Copies into out_buf (max out_size).
// Returns 1 if found, 0 if missing.
static int extract_session_id(const char *buf, char *out_buf, int out_size) {
    const char *p = strstr(buf, "\"session_id\"");
    if (!p) return 0;
    p += 12; // skip "session_id"
    while (*p == ' ' || *p == '\t' || *p == ':') p++;
    if (*p != '"') return 0;
    p++; // skip opening quote
    int i = 0;
    while (*p && *p != '"' && i < out_size - 1) {
        out_buf[i++] = *p++;
    }
    out_buf[i] = '\0';
    return i > 0 ? 1 : 0;
}

// Write a full HTTP response string to fd
static void http_write(int fd, const char *data, int len) {
    int sent = 0;
    while (sent < len) {
        ssize_t w = write(fd, data + sent, len - sent);
        if (w <= 0) break;
        sent += (int)w;
    }
}

static void http_write_str(int fd, const char *s) {
    http_write(fd, s, (int)strlen(s));
}

// Send an SSE chunk with a token delta
// Returns 0 on success, -1 if client disconnected
static int sse_send_delta(int fd, const char *request_id, const char *token_text) {
    char chunk[4096];
    // Escape the token text for JSON
    char escaped[2048];
    char *w = escaped;
    for (const char *r = token_text; *r && w < escaped + sizeof(escaped) - 8; r++) {
        switch (*r) {
            case '"':  *w++ = '\\'; *w++ = '"';  break;
            case '\\': *w++ = '\\'; *w++ = '\\'; break;
            case '\n': *w++ = '\\'; *w++ = 'n';  break;
            case '\r': *w++ = '\\'; *w++ = 'r';  break;
            case '\t': *w++ = '\\'; *w++ = 't';  break;
            default:   *w++ = *r; break;
        }
    }
    *w = '\0';
    int n = snprintf(chunk, sizeof(chunk),
        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
        "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}\n\n",
        request_id, escaped);
    ssize_t wr = write(fd, chunk, n);
    return (wr <= 0) ? -1 : 0;
}

static void sse_send_done(int fd, const char *request_id) {
    char chunk[1024];
    int n = snprintf(chunk, sizeof(chunk),
        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
        "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
        "data: [DONE]\n\n",
        request_id);
    http_write(fd, chunk, n);
}

static const char *SSE_HEADERS =
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: text/event-stream\r\n"
    "Cache-Control: no-cache\r\n"
    "Connection: close\r\n"
    "Access-Control-Allow-Origin: *\r\n"
    "\r\n";

static const char *CORS_RESPONSE =
    "HTTP/1.1 204 No Content\r\n"
    "Access-Control-Allow-Origin: *\r\n"
    "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
    "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
    "Access-Control-Max-Age: 86400\r\n"
    "\r\n";

// Tokenize a user turn (system prompt already cached in KV).
// Only encodes: <|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n
static PromptTokens *tokenize_user_turn(const char *user_content) {
    const char *prefix = "<|im_start|>user\n";
    const char *suffix = "<|im_end|>\n<|im_start|>assistant\n";

    size_t prompt_len = strlen(prefix) + strlen(user_content) + strlen(suffix) + 1;
    char *prompt = malloc(prompt_len);
    if (!prompt) return NULL;
    snprintf(prompt, prompt_len, "%s%s%s", prefix, user_content, suffix);
    PromptTokens *pt = encode_prompt_text_to_tokens(prompt);
    free(prompt);
    return pt;
}

// Tokenize a continuation turn for session caching.
// Prefixes with <|im_end|>\n to close the previous assistant turn, then the new user turn.
// Used when the KV cache already contains the prior conversation state.
static PromptTokens *tokenize_continuation_turn(const char *user_content) {
    // EOS/<|im_end|> is already in the state (fed through model at end of generation)
    // Just need the newline + new user turn + assistant prompt
    const char *prefix = "\n<|im_start|>user\n";
    const char *suffix = "<|im_end|>\n<|im_start|>assistant\n";

    size_t prompt_len = strlen(prefix) + strlen(user_content) + strlen(suffix) + 1;
    char *prompt = malloc(prompt_len);
    if (!prompt) return NULL;
    snprintf(prompt, prompt_len, "%s%s%s", prefix, user_content, suffix);
    PromptTokens *pt = encode_prompt_text_to_tokens(prompt);
    free(prompt);
    return pt;
}

// Load custom system prompt from ~/.flash-moe/system.md, or use default
static char *load_system_prompt(void) {
    const char *home = getenv("HOME");
    if (home) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/.flash-moe/system.md", home);
        FILE *f = fopen(path, "r");
        if (f) {
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            char *buf = malloc(sz + 1);
            size_t n = fread(buf, 1, sz, f);
            buf[n] = 0;
            fclose(f);
            fprintf(stderr, "[serve] Loaded custom system prompt from %s (%ld bytes)\n", path, sz);
            return buf;
        }
    }
    return strdup("You are a helpful assistant. /think");
}

// Tokenize a full chat message (system prompt + user turn) for first-time use.
static PromptTokens *tokenize_chat_message(const char *user_content) {
    static char *sys_prompt_text = NULL;
    if (!sys_prompt_text) sys_prompt_text = load_system_prompt();

    // Build: <|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n
    size_t sys_len = strlen(sys_prompt_text);
    size_t user_len = strlen(user_content);
    size_t total = 30 + sys_len + 30 + user_len + 40;  // generous padding for tags
    char *prompt = malloc(total);
    if (!prompt) return NULL;
    snprintf(prompt, total, "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
             sys_prompt_text, user_content);
    PromptTokens *pt = encode_prompt_text_to_tokens(prompt);
    free(prompt);
    return pt;
}

// Keep old signature for backward compat (unused but prevents compiler warning)
__attribute__((unused))
static PromptTokens *tokenize_chat_message_old(const char *user_content) {
    const char *prefix =
        "<|im_start|>system\nYou are a helpful assistant. /think<|im_end|>\n"
        "<|im_start|>user\n";
    const char *suffix = "<|im_end|>\n<|im_start|>assistant\n";

    size_t prompt_len = strlen(prefix) + strlen(user_content) + strlen(suffix) + 1;
    char *prompt = malloc(prompt_len);
    if (!prompt) return NULL;

    snprintf(prompt, prompt_len, "%s%s%s", prefix, user_content, suffix);
    PromptTokens *pt = encode_prompt_text_to_tokens(prompt);
    free(prompt);
    return pt;
}

// The main serve loop. Model state must already be initialized.
// Sync CPU linear attention state → GPU buffers
static void sync_cpu_to_gpu_delta_state_serve(void **layer_states) {
    if (!g_metal || !g_metal->delta_net_step || !layer_states) return;
    int li = 0;
    for (int i = 0; i < cfg.num_layers; i++) {
        if (cfg.is_full_attn[i]) continue;
        if (!layer_states[i]) { li++; continue; }
        LinearAttnState *la = (LinearAttnState *)layer_states[i];
        if (li < cfg.num_linear_layers) {
            if (g_metal->buf_delta_state[li] && la->ssm_state)
                memcpy([g_metal->buf_delta_state[li] contents], la->ssm_state,
                       cfg.linear_num_v_heads * cfg.linear_value_dim * cfg.linear_key_dim * sizeof(float));
            if (g_metal->buf_conv_state[li] && la->conv_state)
                memcpy([g_metal->buf_conv_state[li] contents], la->conv_state,
                       (cfg.conv_kernel_size - 1) * cfg.linear_conv_dim * sizeof(float));
        }
        li++;
    }
}

static void serve_loop(
    int port,
    WeightFile *wf, Vocabulary *vocab,
    void **layer_states, KVCache **kv_caches,
    void **layer_mmaps, int *layer_fds,
    float *hidden, float *logits,
    uint16_t *final_norm_w, int K)
{
    // Ignore SIGPIPE (client disconnect mid-write)
    signal(SIGPIPE, SIG_IGN);

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket"); return; }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind"); close(server_fd); return;
    }
    if (listen(server_fd, 8) < 0) {
        perror("listen"); close(server_fd); return;
    }

    printf("[serve] Listening on http://0.0.0.0:%d\n", port);
    printf("[serve] Endpoints: POST /v1/chat/completions, GET /v1/models, GET /health\n");
    fflush(stdout);

    static uint64_t req_counter = 0;

    // ---- System prompt cache: prefill system prompt once at startup ----
    // Tokenize the system prompt and run it through all 40 layers.
    // Save the resulting KV cache + linear attention state as a snapshot.
    // On each request, restore the snapshot instead of re-prefilling.
    fprintf(stderr, "[serve] Pre-caching system prompt...\n");
    PromptTokens *sys_pt = tokenize_chat_message("");  // empty user = just system prompt
    int sys_pos = 0;
    if (sys_pt && sys_pt->count > 0) {
        // Cap system prompt length to max_seq_len
        if (sys_pt->count > cfg.max_seq_len) {
            fprintf(stderr, "WARNING: system prompt (%d tokens) exceeds max context (%d), truncating\n",
                    sys_pt->count, cfg.max_seq_len);
            sys_pt->count = cfg.max_seq_len;
        }
        // Pre-embed all system prompt tokens
        float *sys_embed_batch = NULL;
        if (sys_pt->count > 1) {
            sys_embed_batch = malloc((size_t)sys_pt->count * cfg.hidden_dim * sizeof(float));
            if (!sys_embed_batch) {
                fprintf(stderr, "ERROR: Failed to allocate system prompt embed batch (%.1f MB)\n",
                        (double)sys_pt->count * cfg.hidden_dim * 4 / 1e6);
                // Fall back to per-token embedding (slower but works)
            }
        }
        if (sys_embed_batch) {
            for (int i = 0; i < sys_pt->count; i++) {
                embed_lookup(wf, sys_pt->ids[i], sys_embed_batch + (size_t)i * cfg.hidden_dim);
            }
        }
        // Intermediate system prompt tokens: discard last-layer expert output
        for (int i = 0; i < sys_pt->count - 1; i++) {
            cache_telemetry_note_token();
            if (sys_embed_batch) {
                memcpy(hidden, sys_embed_batch + (size_t)i * cfg.hidden_dim,
                       cfg.hidden_dim * sizeof(float));
            } else {
                embed_lookup(wf, sys_pt->ids[i], hidden);
            }
            for (int layer = 0; layer < cfg.num_layers; layer++) {
                int is_full = cfg.is_full_attn[layer];
                fused_layer_forward(wf, layer, hidden,
                                    is_full ? kv_caches[layer] : NULL,
                                    is_full ? NULL : layer_states[layer],
                                    sys_pos,
                                    layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                    K, layer_fds[layer]);
            }
            discard_deferred_experts();
            sys_pos++;
        }
        // Last system prompt token: full completion
        {
            cache_telemetry_note_token();
            if (sys_embed_batch) {
                memcpy(hidden, sys_embed_batch + (size_t)(sys_pt->count - 1) * cfg.hidden_dim,
                       cfg.hidden_dim * sizeof(float));
            } else {
                embed_lookup(wf, sys_pt->ids[0], hidden);
            }
            for (int layer = 0; layer < cfg.num_layers; layer++) {
                int is_full = cfg.is_full_attn[layer];
                fused_layer_forward(wf, layer, hidden,
                                    is_full ? kv_caches[layer] : NULL,
                                    is_full ? NULL : layer_states[layer],
                                    sys_pos,
                                    layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                    K, layer_fds[layer]);
            }
            complete_deferred_experts();
            sys_pos++;
        }
        if (sys_embed_batch) { free(sys_embed_batch); sys_embed_batch = NULL; }
        // Sync CPU state → GPU for delta-net
        sync_cpu_to_gpu_delta_state_serve(layer_states);
        fprintf(stderr, "[serve] System prompt cached: %d tokens prefilled\n", sys_pos);
    }
    free(sys_pt);

    // Save snapshot of KV caches + linear attention state after system prompt
    // These are restored at the start of each request instead of resetting to zero
    typedef struct {
        float *k_snapshot;
        float *v_snapshot;
        int len;
    } KVSnapshot;
    KVSnapshot kv_snapshots[cfg.num_layers];
    memset(kv_snapshots, 0, sizeof(kv_snapshots));

    // Linear attention snapshots
    float *la_conv_snapshots[cfg.num_layers];
    float *la_ssm_snapshots[cfg.num_layers];
    memset(la_conv_snapshots, 0, sizeof(la_conv_snapshots));
    memset(la_ssm_snapshots, 0, sizeof(la_ssm_snapshots));

    size_t kv_dim = cfg.num_kv_heads * cfg.head_dim;
    size_t conv_state_size = (cfg.conv_kernel_size - 1) * cfg.linear_conv_dim * sizeof(float);
    size_t ssm_state_size = cfg.linear_num_v_heads * cfg.linear_value_dim * cfg.linear_key_dim * sizeof(float);

    for (int i = 0; i < cfg.num_layers; i++) {
        if (kv_caches[i]) {
            size_t sz = sys_pos * kv_dim * sizeof(float);
            kv_snapshots[i].k_snapshot = malloc(sz);
            kv_snapshots[i].v_snapshot = malloc(sz);
            memcpy(kv_snapshots[i].k_snapshot, kv_caches[i]->k_cache, sz);
            memcpy(kv_snapshots[i].v_snapshot, kv_caches[i]->v_cache, sz);
            kv_snapshots[i].len = kv_caches[i]->len;
        }
        if (layer_states[i]) {
            LinearAttnState *s = (LinearAttnState *)layer_states[i];
            la_conv_snapshots[i] = malloc(conv_state_size);
            la_ssm_snapshots[i] = malloc(ssm_state_size);
            memcpy(la_conv_snapshots[i], s->conv_state, conv_state_size);
            memcpy(la_ssm_snapshots[i], s->ssm_state, ssm_state_size);
        }
    }
    // Also snapshot GPU delta-net state
    void **gpu_delta_snapshots = calloc(cfg.num_linear_layers, sizeof(void *));
    void **gpu_conv_snapshots = calloc(cfg.num_linear_layers, sizeof(void *));
    // already zeroed by calloc
    // already zeroed by calloc
    if (g_metal && g_metal->delta_net_step) {
        for (int i = 0; i < cfg.num_linear_layers; i++) {
            if (g_metal->buf_delta_state[i]) {
                size_t sz = (size_t)cfg.linear_num_v_heads*cfg.linear_value_dim*cfg.linear_key_dim*sizeof(float);
                gpu_delta_snapshots[i] = malloc(sz);
                memcpy(gpu_delta_snapshots[i], [g_metal->buf_delta_state[i] contents], sz);
            }
            if (g_metal->buf_conv_state[i]) {
                size_t sz = (cfg.conv_kernel_size-1)*(size_t)cfg.linear_conv_dim*sizeof(float);
                gpu_conv_snapshots[i] = malloc(sz);
                memcpy(gpu_conv_snapshots[i], [g_metal->buf_conv_state[i] contents], sz);
            }
        }
    }
    int sys_prompt_len = sys_pos;  // number of tokens in system prompt cache

    // ---- Session state: track one active conversation session ----
    // The KV caches + linear attention state ARE the session.
    // We just track whether to restore from snapshot (new session) or continue (same session).
    char active_session_id[64] = {0};
    int session_pos = 0;  // RoPE position after last generation for the active session

    for (;;) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0) { perror("accept"); continue; }

        // Read HTTP request
        char *reqbuf = malloc(1024 * 1024); // 1MB max request
        int reqlen = read_http_request(client_fd, reqbuf, 1024 * 1024);
        if (reqlen <= 0) { free(reqbuf); close(client_fd); continue; }

        // Parse method and path from first line
        char method[16] = {0}, path[256] = {0};
        sscanf(reqbuf, "%15s %255s", method, path);

        // Handle CORS preflight
        if (strcmp(method, "OPTIONS") == 0) {
            http_write_str(client_fd, CORS_RESPONSE);
            free(reqbuf); close(client_fd);
            continue;
        }

        // GET /health
        if (strcmp(method, "GET") == 0 && strcmp(path, "/health") == 0) {
            const char *resp =
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: application/json\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Connection: close\r\n"
                "\r\n"
                "{\"status\":\"ok\",\"model\":\"qwen3.5-35b-a3b\"}\n";
            http_write_str(client_fd, resp);
            free(reqbuf); close(client_fd);
            continue;
        }

        // GET /v1/models
        if (strcmp(method, "GET") == 0 && strcmp(path, "/v1/models") == 0) {
            const char *resp =
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: application/json\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Connection: close\r\n"
                "\r\n"
                "{\"object\":\"list\",\"data\":[{\"id\":\"qwen3.5-35b-a3b\","
                "\"object\":\"model\",\"owned_by\":\"local\"}]}\n";
            http_write_str(client_fd, resp);
            free(reqbuf); close(client_fd);
            continue;
        }

        // POST /v1/chat/completions
        if (strcmp(method, "POST") == 0 && strcmp(path, "/v1/chat/completions") == 0) {
            // Find body (after \r\n\r\n)
            char *body = strstr(reqbuf, "\r\n\r\n");
            if (!body) {
                http_write_str(client_fd,
                    "HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n"
                    "{\"error\":\"no body\"}\n");
                free(reqbuf); close(client_fd); continue;
            }
            body += 4;

            // Extract session_id and max_tokens BEFORE content extraction
            // (extract_last_content mutates the body buffer in place)
            int max_gen = extract_max_tokens(body, 8192);
            if (max_gen > 32768) max_gen = 32768;
            char req_session_id[64] = {0};
            int has_session = extract_session_id(body, req_session_id, sizeof(req_session_id));

            // Extract user content from messages (mutates body — must be last)
            char *content = extract_last_content(body);
            if (!content || strlen(content) == 0) {
                http_write_str(client_fd,
                    "HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n"
                    "{\"error\":\"no content in messages\"}\n");
                free(reqbuf); close(client_fd); continue;
            }
            int is_continuation = (has_session &&
                                   active_session_id[0] != '\0' &&
                                   strcmp(req_session_id, active_session_id) == 0);

            // Session persistence is handled by the client (chat.m)

            char request_id[64];
            snprintf(request_id, sizeof(request_id), "chatcmpl-%llu", ++req_counter);

            fprintf(stderr, "[serve] %s content=%zu chars, max_tokens=%d, session=%s%s\n",
                    request_id, strlen(content), max_gen,
                    has_session ? req_session_id : "(none)",
                    is_continuation ? " [CONTINUE]" : " [NEW]");

            // ---- Tokenize ----
            // Continuation: prefix with <|im_end|>\n to close prior assistant turn
            // New session: just the user turn (system prompt restored from snapshot)
            PromptTokens *pt;
            if (is_continuation) {
                pt = tokenize_continuation_turn(content);
            } else {
                pt = tokenize_user_turn(content);
            }
            if (!pt) {
                http_write_str(client_fd,
                    "HTTP/1.1 500 Internal Server Error\r\nConnection: close\r\n\r\n"
                    "{\"error\":\"tokenization failed\"}\n");
                free(reqbuf); close(client_fd); continue;
            }

            fprintf(stderr, "[serve] %s prompt=%d tokens%s\n", request_id, pt->count,
                    is_continuation ? " (continuation — skipping snapshot restore)" : "");

            int pos;
            if (is_continuation) {
                // ---- Continue from existing session state ----
                // The KV caches + linear attention state already contain the full
                // conversation history. Just set pos to where we left off.
                pos = session_pos;
            } else {
                // ---- Restore state from system prompt snapshot ----
                // Instead of resetting to zero, restore to the cached system prompt state.
                // This skips re-prefilling the system prompt tokens (~20 tokens, ~6s saved).
                for (int i = 0; i < cfg.num_layers; i++) {
                    if (kv_caches[i] && kv_snapshots[i].k_snapshot) {
                        size_t sz = sys_prompt_len * kv_dim * sizeof(float);
                        memcpy(kv_caches[i]->k_cache, kv_snapshots[i].k_snapshot, sz);
                        memcpy(kv_caches[i]->v_cache, kv_snapshots[i].v_snapshot, sz);
                        kv_caches[i]->len = kv_snapshots[i].len;
                        // Also restore GPU KV mirror
                        if (g_metal) {
                            int fa_idx = cfg.full_attn_index[i];
                            if (fa_idx >= 0 && fa_idx < cfg.num_full_attn_layers) {
                                memcpy([g_metal->buf_kv_k[fa_idx] contents],
                                       kv_snapshots[i].k_snapshot, sz);
                                memcpy([g_metal->buf_kv_v[fa_idx] contents],
                                       kv_snapshots[i].v_snapshot, sz);
                            }
                        }
                    } else if (kv_caches[i]) {
                        kv_caches[i]->len = 0;
                    }
                    if (layer_states[i] && la_conv_snapshots[i]) {
                        LinearAttnState *s = (LinearAttnState *)layer_states[i];
                        memcpy(s->conv_state, la_conv_snapshots[i], conv_state_size);
                        memcpy(s->ssm_state, la_ssm_snapshots[i], ssm_state_size);
                    } else if (layer_states[i]) {
                        LinearAttnState *s = (LinearAttnState *)layer_states[i];
                        memset(s->conv_state, 0, conv_state_size);
                        memset(s->ssm_state, 0, ssm_state_size);
                    }
                }
                // Restore GPU delta-net state
                if (g_metal && g_metal->delta_net_step) {
                    for (int i = 0; i < cfg.num_linear_layers; i++) {
                        if (gpu_delta_snapshots[i] && g_metal->buf_delta_state[i])
                            memcpy([g_metal->buf_delta_state[i] contents],
                                   gpu_delta_snapshots[i], (size_t)cfg.linear_num_v_heads*cfg.linear_value_dim*cfg.linear_key_dim*sizeof(float));
                        if (gpu_conv_snapshots[i] && g_metal->buf_conv_state[i])
                            memcpy([g_metal->buf_conv_state[i] contents],
                                   gpu_conv_snapshots[i], (cfg.conv_kernel_size-1)*(size_t)cfg.linear_conv_dim*sizeof(float));
                    }
                } else {
                    reset_delta_net_state();
                }
                pos = sys_prompt_len;  // start after cached system prompt
                // Update active session
                if (has_session) {
                    strncpy(active_session_id, req_session_id, sizeof(active_session_id) - 1);
                    active_session_id[sizeof(active_session_id) - 1] = '\0';
                } else {
                    active_session_id[0] = '\0';
                }
            }
            if (g_cache_telemetry_enabled) cache_telemetry_reset();

            // ---- Send SSE headers ----
            http_write_str(client_fd, SSE_HEADERS);

            // ---- Batch prefill ----
            double t_prefill = now_ms();
            // Cap prompt length to max_seq_len
            if (pt->count > cfg.max_seq_len) {
                fprintf(stderr, "WARNING: request prompt (%d tokens) exceeds max context (%d), truncating\n",
                        pt->count, cfg.max_seq_len);
                pt->count = cfg.max_seq_len;
            }
            // Pre-embed all request tokens
            float *serve_embed_batch = NULL;
            if (pt->count > 1) {
                serve_embed_batch = malloc((size_t)pt->count * cfg.hidden_dim * sizeof(float));
                if (!serve_embed_batch) {
                    fprintf(stderr, "ERROR: Failed to allocate serve embed batch (%.1f MB)\n",
                            (double)pt->count * cfg.hidden_dim * 4 / 1e6);
                    // Fall back to per-token embedding (slower but works)
                }
                if (serve_embed_batch) {
                    for (int i = 0; i < pt->count; i++) {
                        embed_lookup(wf, pt->ids[i], serve_embed_batch + (size_t)i * cfg.hidden_dim);
                    }
                }
            }
            // Intermediate prefill tokens: discard last-layer expert output
            for (int i = 0; i < pt->count - 1; i++) {
                cache_telemetry_note_token();
                if (serve_embed_batch) {
                    memcpy(hidden, serve_embed_batch + (size_t)i * cfg.hidden_dim,
                           cfg.hidden_dim * sizeof(float));
                } else {
                    embed_lookup(wf, pt->ids[i], hidden);
                }
                for (int layer = 0; layer < cfg.num_layers; layer++) {
                    int is_full = cfg.is_full_attn[layer];
                    fused_layer_forward(wf, layer, hidden,
                                        is_full ? kv_caches[layer] : NULL,
                                        is_full ? NULL : layer_states[layer],
                                        pos,
                                        layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                        K, layer_fds[layer]);
                }
                discard_deferred_experts();
                pos++;
            }
            // Last prefill token: full completion (need hidden for logits)
            {
                cache_telemetry_note_token();
                if (serve_embed_batch) {
                    memcpy(hidden, serve_embed_batch + (size_t)(pt->count - 1) * cfg.hidden_dim,
                           cfg.hidden_dim * sizeof(float));
                } else {
                    embed_lookup(wf, pt->ids[0], hidden);
                }
                for (int layer = 0; layer < cfg.num_layers; layer++) {
                    int is_full = cfg.is_full_attn[layer];
                    fused_layer_forward(wf, layer, hidden,
                                        is_full ? kv_caches[layer] : NULL,
                                        is_full ? NULL : layer_states[layer],
                                        pos,
                                        layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                        K, layer_fds[layer]);
                }
                complete_deferred_experts();
                pos++;
            }
            if (serve_embed_batch) { free(serve_embed_batch); serve_embed_batch = NULL; }
            double prefill_ms = now_ms() - t_prefill;
            fprintf(stderr, "[serve] %s prefill=%d tokens in %.0fms\n",
                    request_id, pt->count, prefill_ms);

            // ---- Final norm + LM head for first token ----
            if (final_norm_w) {
                float *normed = malloc(cfg.hidden_dim * sizeof(float));
                cpu_rms_norm(hidden, final_norm_w, normed, cfg.hidden_dim, cfg.rms_norm_eps);
                memcpy(hidden, normed, cfg.hidden_dim * sizeof(float));
                free(normed);
            }
            lm_head_forward(wf, hidden, logits);
            int next_token = cpu_argmax(logits, cfg.vocab_size);

            // ---- Auto-regressive generation with SSE streaming ----
            if (g_pred_enabled) {
                g_pred_generating = 1;
                g_pred_valid = 0;
            }
            double t_gen = now_ms();
            int gen_count = 0;
            int in_think = 0;
            int think_tokens = 0;
            // Accumulate response for session persistence
            char *gen_response = calloc(1, 256 * 1024);
            int gen_resp_len = 0;

            for (int gen = 0; gen < max_gen; gen++) {
                if (next_token == cfg.eos_token_ids[0] || next_token == cfg.eos_token_ids[1]) {
                    // Feed EOS through the model so session state includes it
                    cache_telemetry_note_token();
                    embed_lookup(wf, next_token, hidden);
                    for (int layer = 0; layer < cfg.num_layers; layer++) {
                        int is_full = cfg.is_full_attn[layer];
                        fused_layer_forward(wf, layer, hidden,
                                            is_full ? kv_caches[layer] : NULL,
                                            is_full ? NULL : layer_states[layer],
                                            pos,
                                            layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                            K, layer_fds[layer]);
                    }
                    discard_deferred_experts();
                    pos++;
                    break;
                }

                // Think budget enforcement
                if (next_token == cfg.think_start_token) in_think = 1;
                if (next_token == cfg.think_end_token) in_think = 0;
                if (in_think) {
                    think_tokens++;
                    if (g_think_budget > 0 && think_tokens >= g_think_budget) {
                        next_token = cfg.think_end_token;  // force end thinking
                        in_think = 0;
                    }
                }

                const char *tok_str = decode_token(vocab, next_token);
                // Accumulate non-thinking response for session persistence
                if (!in_think && tok_str && gen_resp_len + (int)strlen(tok_str) < 256*1024 - 1) {
                    int tlen = (int)strlen(tok_str);
                    memcpy(gen_response + gen_resp_len, tok_str, tlen);
                    gen_resp_len += tlen;
                    gen_response[gen_resp_len] = 0;
                }
                if (sse_send_delta(client_fd, request_id, tok_str) < 0) {
                    fprintf(stderr, "[serve] %s client disconnected, stopping generation\n", request_id);
                    break;
                }
                gen_count++;

                // Generate next
                cache_telemetry_note_token();
                embed_lookup(wf, next_token, hidden);
                for (int layer = 0; layer < cfg.num_layers; layer++) {
                    int is_full = cfg.is_full_attn[layer];
                    fused_layer_forward(wf, layer, hidden,
                                        is_full ? kv_caches[layer] : NULL,
                                        is_full ? NULL : layer_states[layer],
                                        pos,
                                        layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                        K, layer_fds[layer]);
                }
                complete_deferred_experts();
                pos++;

                if (final_norm_w) {
                    float *normed = malloc(cfg.hidden_dim * sizeof(float));
                    cpu_rms_norm(hidden, final_norm_w, normed, cfg.hidden_dim, cfg.rms_norm_eps);
                    memcpy(hidden, normed, cfg.hidden_dim * sizeof(float));
                    free(normed);
                }
                lm_head_forward(wf, hidden, logits);
                next_token = cpu_argmax(logits, cfg.vocab_size);
            }

            sse_send_done(client_fd, request_id);

            // ---- Save session state ----
            free(gen_response);
            // The KV caches + linear attention state already contain this conversation.
            // Just record the position so the next request can continue from here.
            session_pos = pos;
            fprintf(stderr, "[serve] %s session_pos=%d (session=%s)\n",
                    request_id, session_pos,
                    active_session_id[0] ? active_session_id : "(none)");

            double gen_ms = now_ms() - t_gen;
            fprintf(stderr, "[serve] %s generated=%d tokens in %.0fms (%.2f tok/s)\n",
                    request_id, gen_count, gen_ms,
                    gen_count > 0 ? gen_count * 1000.0 / gen_ms : 0.0);
            if (g_expert_cache) {
                cache_telemetry_print(g_expert_cache->hits, g_expert_cache->misses);
            } else if (g_malloc_cache) {
                cache_telemetry_print(g_malloc_cache->hits, g_malloc_cache->misses);
            }

            free(pt->ids);
            free(pt);
            free(reqbuf);
            close(client_fd);
            continue;
        }

        // Unknown endpoint
        const char *resp404 =
            "HTTP/1.1 404 Not Found\r\n"
            "Content-Type: application/json\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Connection: close\r\n"
            "\r\n"
            "{\"error\":\"not found\"}\n";
        http_write_str(client_fd, resp404);
        free(reqbuf);
        close(client_fd);
    }
}

// ============================================================================

static void print_usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  --model PATH         Model path\n");
    printf("  --weights PATH       model_weights.bin path\n");
    printf("  --manifest PATH      model_weights.json path\n");
    printf("  --vocab PATH         vocab.bin path\n");
    printf("  --prompt-tokens PATH prompt_tokens.bin path\n");
    printf("  --prompt TEXT         Prompt text (requires encode_prompt.py)\n");
    printf("  --tokens N           Max tokens to generate (default: 20)\n");
    printf("  --k N                Active experts per layer (default: 4)\n");
    printf("  --cache-entries N    Expert LRU cache size (default: 2500, 0 = disabled)\n");
    printf("  --malloc-cache N     Malloc expert cache entries (e.g., 2581 = 17GB for 80%% hit)\n");
    printf("  --cpu-linear         Disable fused GPU delta-net and use the older CPU/hybrid linear path\n");
    printf("  --timing             Enable per-layer timing breakdown\n");
    printf("  --freq               Enable expert frequency tracking + analysis\n");
    printf("  --cache-telemetry    Report cold vs eviction misses and reuse distance\n");
    printf("  --2bit               Use 2-bit quantized experts (packed_experts_2bit/)\n");
    printf("  --tiered             Use tiered quantization: hot=4-bit, cold=2-bit (packed_experts_tiered/)\n");
    printf("  --gpu-linear         Alias for the fused GPU delta-net path (default)\n");
    printf("  --predict            Enable temporal expert prediction (prefetch during CMD1_wait)\n");
    printf("  --no-prefetch        Disable cross-layer expert prefetch (default: ON)\n");
    printf("  --collect-routing F  Log routing data to binary file F (for predictor training)\n");
    printf("  --collect-activations F  Dump expert activations to binary file F (for GPTQ calibration)\n");
    printf("  --think-budget N     Max thinking tokens before force </think> (default: 2048, 0=unlimited)\n");
    printf("  --fp16               Use half-precision accumulation in dequant kernels (experimental)\n");
    printf("  --serve PORT         Run HTTP server (OpenAI-compatible API)\n");
    printf("  --help               This message\n");
}

int main(int argc, char **argv) {
    @autoreleasepool {
        const char *model_path = getenv("FLASH_MOE_MODEL");
        const char *weights_path = NULL;
        const char *manifest_path = NULL;
        const char *vocab_path = NULL;
        const char *prompt_tokens_path = NULL;
        const char *prompt_text = NULL;
        int max_tokens = 20;
        int K = 8;
        int cache_entries = 0;  // default 0: trust OS page cache (38% faster than Metal LRU)
        int malloc_cache_entries = 0;  // 0 = disabled (override with --malloc-cache)
        int serve_port = 0;  // 0 = disabled, >0 = HTTP serve mode

        static struct option long_options[] = {
            {"model",         required_argument, 0, 'm'},
            {"weights",       required_argument, 0, 'w'},
            {"manifest",      required_argument, 0, 'j'},
            {"vocab",         required_argument, 0, 'v'},
            {"prompt-tokens", required_argument, 0, 'p'},
            {"prompt",        required_argument, 0, 'P'},
            {"tokens",        required_argument, 0, 't'},
            {"k",             required_argument, 0, 'k'},
            {"cache-entries",  required_argument, 0, 'C'},
            {"malloc-cache",   required_argument, 0, 'M'},
            {"cpu-linear",    no_argument,       0, 'L'},
            {"skip-linear",   no_argument,       0, 'S'},
            {"timing",        no_argument,       0, 'T'},
            {"freq",          no_argument,       0, 'F'},
            {"freq-json",     required_argument, 0, 0x100},
            {"cache-telemetry", no_argument,     0, 'E'},
            {"2bit",          no_argument,       0, '2'},
            {"tiered",        no_argument,       0, 'Q'},
            {"gpu-linear",    no_argument,       0, 'G'},
            {"think-budget",  required_argument, 0, 'B'},
            {"serve",         required_argument, 0, 'R'},
            {"predict",       no_argument,       0, 'D'},
            {"no-prefetch",   no_argument,       0, 'X'},
            {"collect-routing", required_argument, 0, 'Z'},
            {"collect-activations", required_argument, 0, 'A'},
            {"fp16",          no_argument,       0, 'H'},
            {"help",          no_argument,       0, 'h'},
            {0, 0, 0, 0}
        };

        int c;
        while ((c = getopt_long(argc, argv, "m:w:j:v:p:P:t:k:C:M:R:B:A:LSTFE2GHh", long_options, NULL)) != -1) {
            switch (c) {
                case 'm': model_path = optarg; break;
                case 'w': weights_path = optarg; break;
                case 'j': manifest_path = optarg; break;
                case 'v': vocab_path = optarg; break;
                case 'p': prompt_tokens_path = optarg; break;
                case 'P': prompt_text = optarg; break;
                case 't': max_tokens = atoi(optarg); break;
                case 'k': K = atoi(optarg); break;
                case 'C': cache_entries = atoi(optarg); break;
                case 'M': malloc_cache_entries = atoi(optarg); break;
                case 'L': gpu_linear_attn_enabled = 0; break;
                case 'S': linear_attn_bypass = 1; break;
                case 'T': g_timing_enabled = 1; break;
                case 'F': g_freq_tracking = 1; break;
                case 0x100: g_freq_json_path = optarg; g_freq_tracking = 1; break;
                case 'E': g_cache_telemetry_enabled = 1; break;
                case '2': g_use_2bit = 1; break;
                case 'Q': g_use_tiered = 1; break;
                case 'G': gpu_linear_attn_enabled = 1; break;
                case 'D': g_pred_enabled = 1; break;
                case 'X': g_expert_prefetch_enabled = 0; break;
                case 'H': g_use_fp16_accum = 1; break;
                case 'Z':
                    g_routing_log = fopen(optarg, "wb");
                    if (!g_routing_log) {
                        fprintf(stderr, "ERROR: cannot open routing log: %s\n", optarg);
                        return 1;
                    }
                    break;
                case 'A':
                    g_activation_dump = fopen(optarg, "wb");
                    if (!g_activation_dump) {
                        fprintf(stderr, "ERROR: cannot open activation dump file: %s\n", optarg);
                        return 1;
                    }
                    break;
                case 'B': g_think_budget = atoi(optarg); break;
                case 'R': serve_port = atoi(optarg); break;
                case 'h': print_usage(argv[0]); return 0;
                default:  print_usage(argv[0]); return 1;
            }
        }

        // ---- Load model configuration from HF config.json ----
        load_model_config(model_path ? model_path : "");
        alloc_tracking_arrays();
        g_deferred.h_mid = calloc(cfg.hidden_dim, sizeof(float));

        // Cap K to MAX_K (buffer overflow safety)
        if (K > MAX_K) {
            fprintf(stderr, "WARNING: K=%d exceeds MAX_K=%d, capping to %d\n", K, MAX_K, MAX_K);
            K = MAX_K;
        }

        // Build default paths — check model directory first, then relative paths
        char default_weights[1024] = {0}, default_manifest[1024] = {0}, default_vocab[1024] = {0};

        if (!weights_path) {
            // 1. Try <model_path>/model_weights.bin
            if (model_path) {
                snprintf(default_weights, sizeof(default_weights),
                         "%s/model_weights.bin", model_path);
                if (access(default_weights, R_OK) != 0)
                    default_weights[0] = '\0';
            }
            // 2. Try relative paths
            if (!default_weights[0]) {
                snprintf(default_weights, sizeof(default_weights),
                         "metal_infer/model_weights.bin");
                if (access(default_weights, R_OK) != 0) {
                    snprintf(default_weights, sizeof(default_weights),
                             "model_weights.bin");
                }
            }
            weights_path = default_weights;
        }
        if (!manifest_path) {
            if (model_path) {
                snprintf(default_manifest, sizeof(default_manifest),
                         "%s/model_weights.json", model_path);
                if (access(default_manifest, R_OK) != 0)
                    default_manifest[0] = '\0';
            }
            if (!default_manifest[0]) {
                snprintf(default_manifest, sizeof(default_manifest),
                         "metal_infer/model_weights.json");
                if (access(default_manifest, R_OK) != 0) {
                    snprintf(default_manifest, sizeof(default_manifest),
                             "model_weights.json");
                }
            }
            manifest_path = default_manifest;
        }
        if (!vocab_path) {
            if (model_path) {
                snprintf(default_vocab, sizeof(default_vocab),
                         "%s/vocab.bin", model_path);
                if (access(default_vocab, R_OK) != 0)
                    default_vocab[0] = '\0';
            }
            if (!default_vocab[0]) {
                snprintf(default_vocab, sizeof(default_vocab),
                         "metal_infer/vocab.bin");
                if (access(default_vocab, R_OK) != 0) {
                    snprintf(default_vocab, sizeof(default_vocab),
                             "vocab.bin");
                }
            }
            vocab_path = default_vocab;
        }

        // ---- Initialize Metal ----
        g_metal = metal_setup();
        if (!g_metal) {
            fprintf(stderr, "WARNING: Metal init failed, falling back to CPU\n");
        }

        // ---- Initialize persistent I/O thread pool ----
        io_pool_init();

        // ---- Initialize malloc expert cache (if requested) ----
        if (malloc_cache_entries > 0) {
            g_malloc_cache = malloc_cache_init(malloc_cache_entries, g_metal ? g_metal->device : MTLCreateSystemDefaultDevice());
            cache_entries = 0;  // disable Metal LRU cache when malloc cache is active
        }

        // ---- Initialize expert LRU cache ----
        if (cache_entries > 0 && g_metal) {
            g_expert_cache = expert_cache_new(g_metal->device, cache_entries);
        }

        printf("=== Flash-MoE Metal Inference Engine ===\n");
        printf("Config:   %s/config.json\n", cfg.model_path);
        printf("Model:    %s\n", model_path);
        printf("Weights:  %s\n", weights_path);
        printf("Manifest: %s\n", manifest_path);
        printf("Vocab:    %s\n", vocab_path);
        printf("K:        %d experts/layer\n", K);
        printf("Quant:    %s\n",
               g_use_tiered ? "tiered (hot=4-bit, cold=2-bit)" :
               g_use_2bit ? "2-bit experts" :
               "4-bit experts");
        printf("Linear:   %s\n", gpu_linear_attn_enabled ? "fused GPU delta-net" : "CPU/hybrid fallback");
        printf("Tokens:   %d\n", max_tokens);
        if (g_malloc_cache) {
            printf("Cache:    malloc %d entries (%.1f GB)\n",
                   malloc_cache_entries, (double)malloc_cache_entries * active_expert_size() / 1e9);
        } else {
            printf("Cache:    %d entries%s\n", cache_entries,
                   cache_entries > 0 ? "" : " (disabled)");
        }

        double t0 = now_ms();

        // ---- Load weights ----
        WeightFile *wf = open_weights(weights_path, manifest_path);
        if (!wf) {
            fprintf(stderr, "ERROR: Failed to load weights\n");
            return 1;
        }

        // Wrap weight file for Metal GPU access
        if (g_metal) {
            metal_set_weights(g_metal, wf->data, wf->size);
        }

        // ---- Load vocabulary ----
        Vocabulary *vocab = load_vocab(vocab_path);
        if (!vocab) {
            fprintf(stderr, "ERROR: Failed to load vocabulary\n");
            return 1;
        }

        // ---- Get prompt tokens (skip in serve mode) ----
        PromptTokens *pt = NULL;
        if (serve_port == 0) {
            if (prompt_text) {
                pt = encode_prompt_text_to_tokens(prompt_text);
                if (!pt) {
                    fprintf(stderr, "ERROR: Failed to encode prompt. Make sure encode_prompt.py exists.\n");
                    return 1;
                }
            } else if (!prompt_tokens_path) {
                pt = encode_prompt_text_to_tokens("Hello, what is");
                if (!pt) {
                    fprintf(stderr, "ERROR: No prompt tokens and encode_prompt.py not found\n");
                    return 1;
                }
            } else {
                pt = load_prompt_tokens(prompt_tokens_path);
            }

            if (!pt) {
                fprintf(stderr, "ERROR: Failed to load prompt tokens from %s\n", prompt_tokens_path);
                return 1;
            }
            printf("[prompt] %d tokens:", pt->count);
            for (int i = 0; i < pt->count && i < 20; i++) {
                printf(" %d", pt->ids[i]);
            }
            printf("\n");
        }

        // ---- Mutual exclusion: --tiered and --2bit cannot coexist ----
        if (g_use_tiered && g_use_2bit) {
            fprintf(stderr, "ERROR: --tiered and --2bit are mutually exclusive\n");
            exit(1);
        }

        // ---- Auto-detect tiered experts (takes priority over 2-bit auto-detect) ----
        if (!g_use_2bit && !g_use_tiered) {
            char probe[1024];
            snprintf(probe, sizeof(probe), "%s/packed_experts_tiered/tiered_manifest.json", model_path);
            if (access(probe, F_OK) == 0) {
                if (load_tiered_manifest(model_path)) {
                    g_use_tiered = 1;
                }
            }
        }

        // ---- Load tiered manifest if --tiered was explicitly set ----
        if (g_use_tiered && !g_tiered_manifest) {
            if (!load_tiered_manifest(model_path)) {
                fprintf(stderr, "ERROR: --tiered specified but no tiered_manifest.json found in %s/packed_experts_tiered/\n", model_path);
                exit(1);
            }
        }

        // ---- Auto-detect 2-bit experts ----
        if (!g_use_2bit && !g_use_tiered) {
            char probe[1024];
            snprintf(probe, sizeof(probe), "%s/packed_experts_2bit/layer_00.bin", model_path);
            int pfd = open(probe, O_RDONLY);
            if (pfd >= 0) {
                close(pfd);
                snprintf(probe, sizeof(probe), "%s/packed_experts/layer_00.bin", model_path);
                int pfd4 = open(probe, O_RDONLY);
                if (pfd4 < 0) {
                    g_use_2bit = 1;
                    printf("[auto] Using 2-bit experts (4-bit not found)\n");
                } else {
                    close(pfd4);
                }
            }
        }

        // ---- Open + mmap packed expert files ----
        // Tiered I/O: two fds per layer file.
        //   layer_fds[i]      = warm fd (page cached) — for experts seen before
        //   layer_fds_cold[i] = cold fd (F_NOCACHE)   — for first-time expert reads
        // Seen-expert bitset tracks which (layer, expert) pairs have been read before.
        // First read goes through cold fd (no page cache pollution).
        // Subsequent reads go through warm fd (page cache hit = 32 GB/s vs 5.5 GB/s).
        int *layer_fds = calloc(cfg.num_layers, sizeof(int));
        int *layer_fds_cold = calloc(cfg.num_layers, sizeof(int));
        void **layer_mmaps = calloc(cfg.num_layers, sizeof(void *));
        size_t *layer_mmap_sizes = calloc(cfg.num_layers, sizeof(size_t));
        int expert_layers_available = 0;

        // Reset the global seen-expert bitset
        memset(g_expert_seen, 0, cfg.num_layers * ((cfg.num_experts + 7) / 8));

        for (int i = 0; i < cfg.num_layers; i++) {
            char path[1024];
            snprintf(path, sizeof(path), "%s/%s/layer_%02d.bin", model_path,
                     g_use_tiered ? "packed_experts_tiered" :
                     g_use_2bit ? "packed_experts_2bit" : "packed_experts", i);
            layer_fds[i] = open(path, O_RDONLY);
            layer_fds_cold[i] = -1;  // no longer used (trust OS page cache)
            layer_mmaps[i] = MAP_FAILED;
            layer_mmap_sizes[i] = 0;
            if (layer_fds[i] >= 0) {
                expert_layers_available++;
                // Disable readahead: expert reads are random (different offsets per token).
                // Read-ahead prefetches adjacent data we won't use, wasting SSD bandwidth.
                fcntl(layer_fds[i], F_RDAHEAD, 0);
                struct stat st;
                if (fstat(layer_fds[i], &st) == 0 && st.st_size > 0) {
#if TARGET_OS_IPHONE || TARGET_OS_IOS
                    // iOS: never mmap expert files. The async pread path (GCD dispatch_group)
                    // doesn't use mmap, and mmap'ing ~18GB+ of expert layer files exhausts
                    // iOS virtual address space (limited even with extended-virtual-addressing).
                    (void)st;
#else
                    if (g_cache_io_split <= 1) {
                        // macOS: mmap when fanout is disabled. With cache-io-split the
                        // pread fanout path is used exclusively and mmap just wastes
                        // virtual address space and adds VM overhead.
                        layer_mmaps[i] = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, layer_fds[i], 0);
                    }
#endif
                    if (layer_mmaps[i] != MAP_FAILED) {
                        layer_mmap_sizes[i] = st.st_size;
                    }
                }
            }
        }
        const char *io_mode;
#if TARGET_OS_IPHONE || TARGET_OS_IOS
        io_mode = g_cache_io_split > 1 ? "pread fanout" : "pread (no mmap)";
#else
        io_mode = g_cache_io_split > 1 ? "pread fanout" : "mmap'd";
#endif
        printf("[experts] %d/%d packed layer files available (%s)\n",
               expert_layers_available, cfg.num_layers, io_mode);
        if (g_cache_io_split > 1) {
            printf("[fanout] cache-io-split=%d → %d page-aligned chunks per expert\n",
                   g_cache_io_split, active_cache_io_split(active_expert_size()));
        }

        // ---- LZ4 compressed experts: auto-detect and load ----
        {
            char lz4_probe[1024];
            snprintf(lz4_probe, sizeof(lz4_probe), "%s/packed_experts_lz4/layer_00.bin", model_path);
            if (!g_use_2bit && access(lz4_probe, R_OK) == 0) {
                int lz4_layers = 0;
                for (int i = 0; i < cfg.num_layers; i++) {
                    char lz4_path[1024];
                    snprintf(lz4_path, sizeof(lz4_path), "%s/packed_experts_lz4/layer_%02d.bin", model_path, i);
                    int lz4_fd = open(lz4_path, O_RDONLY);
                    if (lz4_fd >= 0) {
                        // Load index header (cfg.num_experts entries × 16 bytes)
                        g_lz4_index[i] = malloc(cfg.num_experts * sizeof(LZ4IndexEntry));
                        ssize_t nr = pread(lz4_fd, g_lz4_index[i],
                                           cfg.num_experts * sizeof(LZ4IndexEntry), 0);
                        if (nr == cfg.num_experts * (ssize_t)sizeof(LZ4IndexEntry)) {
                            // Replace the raw fd with the LZ4 fd
                            close(layer_fds[i]);
                            layer_fds[i] = lz4_fd;
                            fcntl(lz4_fd, F_RDAHEAD, 1);
                            lz4_layers++;
                        } else {
                            free(g_lz4_index[i]);
                            g_lz4_index[i] = NULL;
                            close(lz4_fd);
                        }
                    }
                }
                if (lz4_layers > 0) {
                    g_use_lz4 = 1;
                    // Allocate compressed read buffers (one per expert slot)
                    for (int k = 0; k < MAX_K; k++) {
                        g_lz4_comp_bufs[k] = malloc(cfg.expert_size_4bit + 4096);
                    }
                    printf("[lz4] %d/%d layers using LZ4 compressed experts\n",
                           lz4_layers, cfg.num_layers);
                }
            }
        }

        // Wire up tiered I/O globals
        g_layer_fds_cold = layer_fds_cold;
        if (!g_use_lz4)
            printf("[tiered-io] Cold fds (F_NOCACHE) + warm fds (page cached) active\n");

        // Wire up cross-layer prefetch globals
        g_layer_fds_global = layer_fds;
        g_layer_mmaps_global = (void **)layer_mmaps;
        g_layer_mmap_sizes_global = layer_mmap_sizes;
        g_prefetch_active = 0;
        g_prefetch_layer = -1;
        g_prefetch_hits_total = 0;
        g_prefetch_misses_total = 0;

        // Warm page cache hint
        if (expert_layers_available > 0) {
            double t_warm = now_ms();
            for (int i = 0; i < cfg.num_layers; i++) {
                if (layer_fds[i] >= 0) {
                    char dummy[4096];
                    pread(layer_fds[i], dummy, sizeof(dummy), 0);
                }
            }
            printf("[warmup] Page cache hint: %.1f ms\n", now_ms() - t_warm);
        }

        if (g_expert_prefetch_enabled)
            printf("[prefetch] Cross-layer expert prefetch enabled (overlap I/O with CMD3 GPU)\n");

        // ---- Allocate per-layer state ----
        void **layer_states = calloc(cfg.num_layers, sizeof(void *));
        KVCache **kv_caches = calloc(cfg.num_layers, sizeof(KVCache *));

        for (int i = 0; i < cfg.num_layers; i++) {
            int is_full = cfg.is_full_attn[i];
            if (is_full) {
                kv_caches[i] = kv_cache_new();
            } else {
                layer_states[i] = linear_attn_state_new();
            }
        }

        double t_init = now_ms();
        printf("[init] Setup: %.1f ms\n\n", t_init - t0);

        // ---- Allocate working buffers ----
        float *hidden = calloc(cfg.hidden_dim, sizeof(float));
        float *logits = calloc(cfg.vocab_size, sizeof(float));
        uint16_t *final_norm_w = get_tensor_ptr(wf, "model.norm.weight");

        // ---- Serve mode: enter HTTP server loop (never returns) ----
        if (serve_port > 0) {
            reset_delta_net_state();
            serve_loop(serve_port, wf, vocab,
                       layer_states, kv_caches,
                       (void **)layer_mmaps, layer_fds,
                       hidden, logits, final_norm_w, K);
            // serve_loop never returns, but cleanup just in case
            free(hidden); free(logits);
            return 0;
        }

        // ---- Generate tokens ----
        reset_delta_net_state();  // zero GPU delta-net state before generation
        if (g_cache_telemetry_enabled) cache_telemetry_reset();
        printf("--- Generating %d tokens ---\n", max_tokens);
        int pos = 0;  // position counter for RoPE

        // ---- Cap prompt length to max_seq_len ----
        if (pt->count > cfg.max_seq_len) {
            fprintf(stderr, "WARNING: prompt (%d tokens) exceeds max context (%d), truncating\n",
                    pt->count, cfg.max_seq_len);
            pt->count = cfg.max_seq_len;
        }

        // ---- Batch prefill: pre-embed all prompt tokens ----
        // Embedding all tokens upfront into a batch buffer avoids interleaving
        // embed_lookup with GPU work, and enables the optimized prefill loop below.
        float *embed_batch = NULL;
        if (pt->count > 1) {
            embed_batch = malloc((size_t)pt->count * cfg.hidden_dim * sizeof(float));
            if (!embed_batch) {
                fprintf(stderr, "ERROR: Failed to allocate embed batch (%.1f MB)\n",
                        (double)pt->count * cfg.hidden_dim * 4 / 1e6);
                free(hidden); free(logits);
                return -1;
            }
            double t_embed = now_ms();
            for (int i = 0; i < pt->count; i++) {
                embed_lookup(wf, pt->ids[i], embed_batch + (size_t)i * cfg.hidden_dim);
            }
            double embed_ms = now_ms() - t_embed;
            printf("  [prefill] batch embed %d tokens: %.1f ms\n", pt->count, embed_ms);
        }

        // ---- Batch prefill loop ----
        // Process all prompt tokens through the model. For intermediate tokens
        // (not the last), we use discard_deferred_experts() which waits for the GPU
        // but skips the CPU readback/combine of the last layer's expert outputs.
        // This is safe because the hidden state from intermediate prefill tokens
        // is immediately overwritten by the next token's embedding — the recurrent
        // state (KV cache, delta-net state) is already updated inside fused_layer_forward.
        if (pt->count > 1) {
            double t_prefill_batch = now_ms();
            double first_tok_ms = 0;

            for (int token_idx = 0; token_idx < pt->count - 1; token_idx++) {
                double t_tok = now_ms();

                // Load pre-embedded token from batch buffer
                cache_telemetry_note_token();
                memcpy(hidden, embed_batch + (size_t)token_idx * cfg.hidden_dim,
                       cfg.hidden_dim * sizeof(float));

                // Run through all 60 transformer layers
                for (int layer = 0; layer < cfg.num_layers; layer++) {
                    int is_full = cfg.is_full_attn[layer];
                    fused_layer_forward(wf, layer, hidden,
                                        is_full ? kv_caches[layer] : NULL,
                                        is_full ? NULL : layer_states[layer],
                                        pos,
                                        layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                        K, layer_fds[layer]);
                }

                // Discard last layer's expert output — hidden will be overwritten
                // by the next token's embedding. Only wait for GPU (buffer safety).
                discard_deferred_experts();
                pos++;

                if (token_idx == 0) {
                    first_tok_ms = now_ms() - t_tok;
                }
            }

            double prefill_batch_ms = now_ms() - t_prefill_batch;
            double avg_ms = (pt->count > 2) ?
                (prefill_batch_ms - first_tok_ms) / (pt->count - 2) : first_tok_ms;
            printf("  [prefill] %d/%d tokens: %.0f ms (first: %.0f ms, rest avg: %.0f ms)\n",
                   pt->count - 1, pt->count, prefill_batch_ms, first_tok_ms, avg_ms);
        }

        // ---- Last prefill token (or single-token prompt) ----
        // This one needs full completion since we need hidden state for logits.
        {
            cache_telemetry_note_token();
            if (embed_batch) {
                memcpy(hidden, embed_batch + (size_t)(pt->count - 1) * cfg.hidden_dim,
                       cfg.hidden_dim * sizeof(float));
            } else {
                embed_lookup(wf, pt->ids[0], hidden);
            }

            for (int layer = 0; layer < cfg.num_layers; layer++) {
                int is_full = cfg.is_full_attn[layer];
                fused_layer_forward(wf, layer, hidden,
                                    is_full ? kv_caches[layer] : NULL,
                                    is_full ? NULL : layer_states[layer],
                                    pos,
                                    layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                    K, layer_fds[layer]);
            }
            // Full completion — need hidden state for final norm + lm_head
            complete_deferred_experts();
            pos++;
        }

        if (embed_batch) { free(embed_batch); embed_batch = NULL; }

        // ---- Final norm ----
        if (final_norm_w) {
            float *normed = malloc(cfg.hidden_dim * sizeof(float));
            cpu_rms_norm(hidden, final_norm_w, normed, cfg.hidden_dim, cfg.rms_norm_eps);
            memcpy(hidden, normed, cfg.hidden_dim * sizeof(float));
            free(normed);
        }

        // ---- LM head ----
        double t_lm = now_ms();
        lm_head_forward(wf, hidden, logits);
        double lm_ms = now_ms() - t_lm;

        // ---- Sample first token ----
        int next_token = cpu_argmax(logits, cfg.vocab_size);
        double ttft_ms = now_ms() - t0;

        // Debug: show top-5 logits for first token
        {
            // Find top 5 manually
            int top5[5] = {0,0,0,0,0};
            float topv[5] = {-1e30f,-1e30f,-1e30f,-1e30f,-1e30f};
            for (int i = 0; i < cfg.vocab_size; i++) {
                int min_k = 0;
                for (int k = 1; k < 5; k++) if (topv[k] < topv[min_k]) min_k = k;
                if (logits[i] > topv[min_k]) { topv[min_k] = logits[i]; top5[min_k] = i; }
            }
            fprintf(stderr, "[debug] Top 5 logits (next_token=%d):\n", next_token);
            for (int i = 0; i < 5; i++) {
                fprintf(stderr, "  token %d (\"%s\") logit=%.4f\n",
                        top5[i], decode_token(vocab, top5[i]), topv[i]);
            }
            fprintf(stderr, "[debug] hidden rms after final_norm=%.4f, logits rms=%.4f\n",
                    vec_rms(hidden, cfg.hidden_dim), vec_rms(logits, cfg.vocab_size));
        }
        printf("[ttft] %.0f ms (prefill %d tokens + lm_head %.0f ms)\n",
               ttft_ms, pt->count, lm_ms);

        printf("\n--- Output ---\n");
        printf("%s", decode_token(vocab, next_token));
        fflush(stdout);

        int total_generated = 1;
        int in_think = (next_token == cfg.think_start_token) ? 1 : 0;
        int think_tokens = 0;

        // ---- Auto-regressive generation ----
        if (g_timing_enabled) timing_reset();
        if (g_pred_enabled) {
            g_pred_generating = 1;  // enable prediction storage/use during generation
            g_pred_valid = 0;       // reset — first gen token builds predictions
        }
        for (int gen = 1; gen < max_tokens; gen++) {
            double t_gen_start = now_ms();

            // Check EOS
            if (next_token == cfg.eos_token_ids[0] || next_token == cfg.eos_token_ids[1]) {
                fprintf(stderr, "\n[eos] Token %d at position %d\n", next_token, gen);
                break;
            }

            // Think budget enforcement
            if (next_token == cfg.think_start_token) in_think = 1;
            if (next_token == cfg.think_end_token) in_think = 0;
            if (in_think) think_tokens++;

            // Embed the just-generated token (next iteration)
            cache_telemetry_note_token();
            embed_lookup(wf, next_token, hidden);

            // Run 40 layers (fused: 1+K cmd buffers per layer)
            for (int layer = 0; layer < cfg.num_layers; layer++) {
                int is_full = cfg.is_full_attn[layer];
                fused_layer_forward(wf, layer, hidden,
                                    is_full ? kv_caches[layer] : NULL,
                                    is_full ? NULL : layer_states[layer],
                                    pos,
                                    layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                    K, layer_fds[layer]);
            }
            // Complete last layer's deferred GPU experts before final norm
            complete_deferred_experts();
            pos++;

            // Final norm
            if (final_norm_w) {
                float *normed = malloc(cfg.hidden_dim * sizeof(float));
                cpu_rms_norm(hidden, final_norm_w, normed, cfg.hidden_dim, cfg.rms_norm_eps);
                memcpy(hidden, normed, cfg.hidden_dim * sizeof(float));
                free(normed);
            }

            // LM head
            lm_head_forward(wf, hidden, logits);

            // Greedy sample
            next_token = cpu_argmax(logits, cfg.vocab_size);

            // Think budget: force end thinking if over budget
            if (in_think && g_think_budget > 0 && think_tokens >= g_think_budget) {
                next_token = cfg.think_end_token;
                in_think = 0;
            }
            total_generated++;

            // Print decoded token
            printf("%s", decode_token(vocab, next_token));
            fflush(stdout);

            double t_gen_end = now_ms();
            double tok_time = t_gen_end - t_gen_start;

            // Print progress to stderr
            fprintf(stderr, "  [gen %d/%d] token_id=%d (%.0f ms, %.2f tok/s)\n",
                    gen, max_tokens, next_token, tok_time, 1000.0 / tok_time);
        }

        if (g_timing_enabled) timing_print();
        printf("\n\n--- Statistics ---\n");
        double total_time = now_ms() - t0;
        printf("Total time:     %.1f s\n", total_time / 1000.0);
        printf("TTFT:           %.0f ms\n", ttft_ms);
        printf("Tokens:         %d generated\n", total_generated);
        if (total_generated > 1) {
            double gen_time = total_time - ttft_ms;
            printf("Generation:     %.1f s (%.2f tok/s)\n",
                   gen_time / 1000.0, (total_generated - 1) * 1000.0 / gen_time);
        }
        printf("Config:         K=%d experts, %d layers\n", K, cfg.num_layers);
        if (g_expert_cache) {
            uint64_t total = g_expert_cache->hits + g_expert_cache->misses;
            printf("Expert cache:   %llu hits, %llu misses (%.1f%% hit rate), %d/%d entries used\n",
                   g_expert_cache->hits, g_expert_cache->misses,
                   total > 0 ? 100.0 * g_expert_cache->hits / total : 0.0,
                   g_expert_cache->num_entries, g_expert_cache->max_entries);
            cache_telemetry_print(g_expert_cache->hits, g_expert_cache->misses);
        } else if (g_malloc_cache) {
            uint64_t total = g_malloc_cache->hits + g_malloc_cache->misses;
            printf("Expert cache:   malloc %llu hits, %llu misses (%.1f%% hit rate), %d/%d entries used\n",
                   g_malloc_cache->hits, g_malloc_cache->misses,
                   total > 0 ? 100.0 * g_malloc_cache->hits / total : 0.0,
                   g_malloc_cache->num_entries, g_malloc_cache->max_entries);
            cache_telemetry_print(g_malloc_cache->hits, g_malloc_cache->misses);
        }

        if (g_spec_route_attempts > 0) {
            printf("Spec routing:   %llu attempts, %llu preloads, %llu hits (%.1f%% prediction accuracy)\n",
                   g_spec_route_attempts, g_spec_route_preloads, g_spec_route_hits,
                   g_spec_route_attempts > 0
                       ? 100.0 * g_spec_route_hits / g_spec_route_attempts : 0.0);
        }

        if (g_freq_tracking) freq_print_analysis(K);
        if (g_routing_log) {
            fclose(g_routing_log);
            fprintf(stderr, "[routing] Logged %d samples to routing data file\n",
                    g_routing_log_samples);
            g_routing_log = NULL;
        }
        if (g_activation_dump) {
            fclose(g_activation_dump);
            fprintf(stderr, "[activations] Dumped %d activation records for GPTQ calibration\n",
                    g_activation_dump_samples);
            g_activation_dump = NULL;
        }

        // ---- Cleanup ----
        // Drain any in-flight cross-layer prefetch
        if (g_prefetch_active) {
            async_pread_wait();
            g_prefetch_active = 0;
            g_prefetch_layer = -1;
        }
        g_layer_fds_global = NULL;
        g_layer_mmaps_global = NULL;
        g_layer_mmap_sizes_global = NULL;
        io_pool_shutdown();
        if (g_malloc_cache) {
            malloc_cache_free(g_malloc_cache);
            g_malloc_cache = NULL;
        }
        if (g_expert_cache) {
            expert_cache_free(g_expert_cache);
            g_expert_cache = NULL;
        }
        for (int i = 0; i < cfg.num_layers; i++) {
            if (kv_caches[i]) kv_cache_free(kv_caches[i]);
            if (layer_states[i]) linear_attn_state_free(layer_states[i]);
            if (layer_mmaps[i] != MAP_FAILED) munmap(layer_mmaps[i], layer_mmap_sizes[i]);
            if (layer_fds[i] >= 0) close(layer_fds[i]);
            if (layer_fds_cold[i] >= 0) close(layer_fds_cold[i]);
        }
        free(layer_states);
        free(kv_caches);
        free(hidden);
        free(logits);

        return 0;
    }
}
#endif // CHAT_MODE
