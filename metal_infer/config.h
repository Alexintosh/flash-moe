// config.h — ModelConfig struct, macros, config parsing
// Part of Flash-MoE modular decomposition (unity build).
// Included by infer.m — do NOT compile separately.

// ============================================================================
// Runtime model configuration (populated from HuggingFace config.json)
// ============================================================================

typedef struct {
    // Core architecture
    int hidden_dim;
    int num_layers;
    int num_attn_heads;
    int num_kv_heads;
    int head_dim;
    int vocab_size;
    float rms_norm_eps;

    // MoE
    int num_experts;
    int num_experts_per_tok;
    int moe_intermediate;
    int shared_intermediate;
    int group_size;
    int bits;

    // Linear attention (GatedDeltaNet)
    int linear_num_v_heads;
    int linear_num_k_heads;
    int linear_key_dim;
    int linear_value_dim;
    int conv_kernel_size;

    // Full attention
    float rope_theta;
    float partial_rotary;

    // Layer type map
    int num_full_attn_layers;
    int num_linear_layers;
    bool *is_full_attn;       // [num_layers]
    int *full_attn_index;     // [num_layers] — index into full-attn buffers, or -1
    int *linear_index;        // [num_layers] — index into linear-attn buffers, or -1

    // Derived: expert byte offsets (4-bit)
    size_t expert_size_4bit;
    size_t gate_w_off_4, gate_s_off_4, gate_b_off_4;
    size_t up_w_off_4, up_s_off_4, up_b_off_4;
    size_t down_w_off_4, down_s_off_4, down_b_off_4;

    // Derived: expert byte offsets (2-bit)
    size_t expert_size_2bit;
    size_t gate_w_off_2, gate_s_off_2, gate_b_off_2;
    size_t up_w_off_2, up_s_off_2, up_b_off_2;
    size_t down_w_off_2, down_s_off_2, down_b_off_2;

    // Derived dimensions
    int linear_total_key;
    int linear_total_value;
    int linear_conv_dim;
    int rotary_dim;

    // Special tokens
    int eos_token_ids[8];
    int num_eos_tokens;
    int think_start_token;
    int think_end_token;

    // Context limits
    int max_seq_len;
    int gpu_kv_seq;

    // Model path (resolved)
    char model_path[1024];
} ModelConfig;

static ModelConfig cfg;

// ---- Tiered expert quantization manifest ----
// Per-expert metadata: offset in layer file, size, and quant bits (2 or 4)
typedef struct {
    size_t offset;   // byte offset in layer_XX.bin
    size_t size;     // bytes to read (expert_size_4bit or expert_size_2bit)
    int bits;        // 2 or 4
} TieredExpertInfo;

// Global tiered manifest: NULL if not using tiered mode
static TieredExpertInfo *g_tiered_manifest = NULL;  // [num_layers * num_experts]
static int g_use_tiered = 0;

// Access helper
#define TIERED(l, e) g_tiered_manifest[(l) * cfg.num_experts + (e)]

static void compute_expert_offsets(ModelConfig *c) {
    int mid = c->moe_intermediate;
    int hid = c->hidden_dim;
    int gs = c->group_size;

    for (int b = 4; b >= 2; b -= 2) {
        int vals_per_u32 = 32 / b;
        // gate_proj [mid, hid]
        size_t gw = (size_t)mid * ((hid + vals_per_u32 - 1) / vals_per_u32) * 4;
        size_t gs_sz = (size_t)mid * ((hid + gs - 1) / gs) * 2;
        size_t gb = gs_sz;
        // up_proj [mid, hid] — same shape
        size_t uw = gw, us = gs_sz, ub = gb;
        // down_proj [hid, mid]
        size_t dw = (size_t)hid * ((mid + vals_per_u32 - 1) / vals_per_u32) * 4;
        size_t ds = (size_t)hid * ((mid + gs - 1) / gs) * 2;
        size_t db = ds;

        size_t off = 0;
        if (b == 4) {
            c->gate_w_off_4 = off; off += gw;
            c->gate_s_off_4 = off; off += gs_sz;
            c->gate_b_off_4 = off; off += gb;
            c->up_w_off_4   = off; off += uw;
            c->up_s_off_4   = off; off += us;
            c->up_b_off_4   = off; off += ub;
            c->down_w_off_4 = off; off += dw;
            c->down_s_off_4 = off; off += ds;
            c->down_b_off_4 = off; off += db;
            c->expert_size_4bit = off;
        } else {
            c->gate_w_off_2 = off; off += gw;
            c->gate_s_off_2 = off; off += gs_sz;
            c->gate_b_off_2 = off; off += gb;
            c->up_w_off_2   = off; off += uw;
            c->up_s_off_2   = off; off += us;
            c->up_b_off_2   = off; off += ub;
            c->down_w_off_2 = off; off += dw;
            c->down_s_off_2 = off; off += ds;
            c->down_b_off_2 = off; off += db;
            c->expert_size_2bit = off;
        }
    }
}

static void load_model_config(const char *model_dir) {
    memset(&cfg, 0, sizeof(cfg));
    cfg.think_start_token = -1;
    cfg.think_end_token = -1;
    cfg.gpu_kv_seq = 8192;

    if (!model_dir || !model_dir[0]) {
        fprintf(stderr, "FATAL: --model path required\n");
        exit(1);
    }

    // Resolve HF snapshot directory
    NSString *base = [NSString stringWithUTF8String:model_dir];
    NSString *configPath = [base stringByAppendingPathComponent:@"config.json"];
    NSFileManager *fm = [NSFileManager defaultManager];

    if (![fm fileExistsAtPath:configPath]) {
        NSString *snapDir = [base stringByAppendingPathComponent:@"snapshots"];
        if ([fm fileExistsAtPath:snapDir]) {
            NSArray *snaps = [[fm contentsOfDirectoryAtPath:snapDir error:nil]
                              sortedArrayUsingSelector:@selector(compare:)];
            for (NSString *snap in snaps) {
                NSString *candidate = [[snapDir stringByAppendingPathComponent:snap]
                                        stringByAppendingPathComponent:@"config.json"];
                if ([fm fileExistsAtPath:candidate]) {
                    base = [snapDir stringByAppendingPathComponent:snap];
                    configPath = candidate;
                    break;
                }
            }
        }
    }

    if (![fm fileExistsAtPath:configPath]) {
        fprintf(stderr, "FATAL: config.json not found in %s\n", model_dir);
        exit(1);
    }

    strlcpy(cfg.model_path, [base UTF8String], sizeof(cfg.model_path));

    // Parse config.json
    NSData *data = [NSData dataWithContentsOfFile:configPath];
    NSError *jsonErr = nil;
    NSDictionary *root = [NSJSONSerialization JSONObjectWithData:data options:0 error:&jsonErr];
    if (!root) {
        fprintf(stderr, "FATAL: failed to parse config.json: %s\n", [[jsonErr localizedDescription] UTF8String]);
        exit(1);
    }
    NSDictionary *tc = root[@"text_config"];
    if (!tc) { fprintf(stderr, "FATAL: config.json missing text_config\n"); exit(1); }

    cfg.hidden_dim       = [tc[@"hidden_size"] intValue];
    cfg.num_layers       = [tc[@"num_hidden_layers"] intValue];
    cfg.num_attn_heads   = [tc[@"num_attention_heads"] intValue];
    cfg.num_kv_heads     = [tc[@"num_key_value_heads"] intValue];
    cfg.head_dim         = tc[@"head_dim"] ? [tc[@"head_dim"] intValue] : (cfg.hidden_dim / cfg.num_attn_heads);
    cfg.vocab_size       = [tc[@"vocab_size"] intValue];
    cfg.rms_norm_eps     = [tc[@"rms_norm_eps"] floatValue];
    cfg.num_experts      = [tc[@"num_experts"] intValue];
    cfg.num_experts_per_tok = [tc[@"num_experts_per_tok"] intValue];
    cfg.moe_intermediate = [tc[@"moe_intermediate_size"] intValue];
    cfg.shared_intermediate = [tc[@"shared_expert_intermediate_size"] intValue];
    cfg.linear_num_v_heads = [tc[@"linear_num_value_heads"] intValue];
    cfg.linear_num_k_heads = [tc[@"linear_num_key_heads"] intValue];
    cfg.linear_key_dim   = tc[@"linear_key_head_dim"] ? [tc[@"linear_key_head_dim"] intValue] : 128;
    cfg.linear_value_dim = tc[@"linear_value_head_dim"] ? [tc[@"linear_value_head_dim"] intValue] : 128;
    cfg.conv_kernel_size = tc[@"linear_conv_kernel_dim"] ? [tc[@"linear_conv_kernel_dim"] intValue] : 4;
    cfg.max_seq_len      = [tc[@"max_position_embeddings"] intValue];

    // Quantization
    NSDictionary *qc = root[@"quantization_config"] ?: root[@"quantization"];
    if (qc) {
        cfg.group_size = [qc[@"group_size"] intValue];
        cfg.bits       = [qc[@"bits"] intValue];
    } else {
        cfg.group_size = 64;
        cfg.bits       = 4;
        fprintf(stderr, "[config] WARNING: no quantization_config, defaulting to 4-bit group_size=64\n");
    }

    // RoPE parameters
    NSDictionary *rope = tc[@"rope_parameters"];
    if (rope) {
        cfg.rope_theta    = [rope[@"rope_theta"] floatValue];
        cfg.partial_rotary = [rope[@"partial_rotary_factor"] floatValue];
    } else {
        cfg.rope_theta    = 10000000.0f;
        cfg.partial_rotary = 0.25f;
    }

    // Layer types
    NSArray *layerTypes = tc[@"layer_types"];
    cfg.is_full_attn    = calloc(cfg.num_layers, sizeof(bool));
    cfg.full_attn_index = malloc(cfg.num_layers * sizeof(int));
    cfg.linear_index    = malloc(cfg.num_layers * sizeof(int));

    if (layerTypes && [layerTypes count] == (NSUInteger)cfg.num_layers) {
        for (int i = 0; i < cfg.num_layers; i++) {
            cfg.is_full_attn[i] = [layerTypes[i] isEqualToString:@"full_attention"];
        }
    } else {
        int interval = tc[@"full_attention_interval"] ? [tc[@"full_attention_interval"] intValue] : 4;
        for (int i = 0; i < cfg.num_layers; i++) {
            cfg.is_full_attn[i] = ((i + 1) % interval == 0);
        }
        fprintf(stderr, "[config] Using full_attn_interval=%d (no explicit layer_types)\n", interval);
    }

    int full_count = 0, linear_count = 0;
    for (int i = 0; i < cfg.num_layers; i++) {
        if (cfg.is_full_attn[i]) {
            cfg.full_attn_index[i] = full_count++;
            cfg.linear_index[i] = -1;
        } else {
            cfg.linear_index[i] = linear_count++;
            cfg.full_attn_index[i] = -1;
        }
    }
    cfg.num_full_attn_layers = full_count;
    cfg.num_linear_layers = linear_count;

    // EOS tokens (can be int or array in config.json)
    id eosVal = root[@"eos_token_id"];
    if ([eosVal isKindOfClass:[NSArray class]]) {
        NSArray *arr = (NSArray *)eosVal;
        cfg.num_eos_tokens = (int)[arr count];
        if (cfg.num_eos_tokens > 8) cfg.num_eos_tokens = 8;
        for (int i = 0; i < cfg.num_eos_tokens; i++)
            cfg.eos_token_ids[i] = [arr[i] intValue];
    } else if (eosVal) {
        cfg.num_eos_tokens = 1;
        cfg.eos_token_ids[0] = [eosVal intValue];
    }

    // Think tokens from tokenizer.json added_tokens
    NSString *tokPath = [base stringByAppendingPathComponent:@"tokenizer.json"];
    if ([fm fileExistsAtPath:tokPath]) {
        NSData *tokData = [NSData dataWithContentsOfFile:tokPath];
        NSDictionary *tokRoot = [NSJSONSerialization JSONObjectWithData:tokData options:0 error:nil];
        NSArray *addedTokens = tokRoot[@"added_tokens"];
        if (addedTokens) {
            for (NSDictionary *tok in addedTokens) {
                NSString *content = tok[@"content"];
                int tid = [tok[@"id"] intValue];
                if ([content isEqualToString:@"<think>"]) cfg.think_start_token = tid;
                else if ([content isEqualToString:@"</think>"]) cfg.think_end_token = tid;
            }
        }
    } else {
        fprintf(stderr, "[config] WARNING: tokenizer.json not found, think tokens disabled\n");
    }

    // Derived dimensions
    cfg.linear_total_key   = cfg.linear_num_k_heads * cfg.linear_key_dim;
    cfg.linear_total_value = cfg.linear_num_v_heads * cfg.linear_value_dim;
    cfg.linear_conv_dim    = cfg.linear_total_key * 2 + cfg.linear_total_value;
    cfg.rotary_dim         = (int)(cfg.head_dim * cfg.partial_rotary);

    // Expert byte offsets
    compute_expert_offsets(&cfg);

    // Summary
    fprintf(stderr, "[config] %d layers (%d linear + %d full), hidden=%d, heads=%d, kv_heads=%d, head_dim=%d\n",
            cfg.num_layers, cfg.num_linear_layers, cfg.num_full_attn_layers,
            cfg.hidden_dim, cfg.num_attn_heads, cfg.num_kv_heads, cfg.head_dim);
    fprintf(stderr, "[config] %d experts (K=%d), moe_intermediate=%d, shared=%d\n",
            cfg.num_experts, cfg.num_experts_per_tok, cfg.moe_intermediate, cfg.shared_intermediate);
    fprintf(stderr, "[config] %d-bit quantization, group_size=%d, expert_size=%zu bytes\n",
            cfg.bits, cfg.group_size, cfg.expert_size_4bit);
    fprintf(stderr, "[config] EOS tokens: [");
    for (int i = 0; i < cfg.num_eos_tokens; i++)
        fprintf(stderr, "%s%d", i ? ", " : "", cfg.eos_token_ids[i]);
    fprintf(stderr, "], think: %d/%d\n", cfg.think_start_token, cfg.think_end_token);
}

// ============================================================================
// Tiered manifest loader
// ============================================================================

static int load_tiered_manifest(const char *model_path) {
    char manifest_path[1024];
    snprintf(manifest_path, sizeof(manifest_path),
             "%s/packed_experts_tiered/tiered_manifest.json", model_path);

    NSData *data = [NSData dataWithContentsOfFile:
        [NSString stringWithUTF8String:manifest_path]];
    if (!data) return 0;  // No tiered manifest found

    NSError *err = nil;
    NSDictionary *root = [NSJSONSerialization JSONObjectWithData:data options:0 error:&err];
    if (!root || err) {
        fprintf(stderr, "[tiered] Failed to parse %s: %s\n",
                manifest_path, [[err localizedDescription] UTF8String]);
        return 0;
    }

    int num_layers = [root[@"num_layers"] intValue];
    int num_experts = [root[@"num_experts"] intValue];

    if (num_layers != cfg.num_layers || num_experts != cfg.num_experts) {
        fprintf(stderr, "[tiered] Manifest mismatch: %dx%d vs config %dx%d\n",
                num_layers, num_experts, cfg.num_layers, cfg.num_experts);
        return 0;
    }

    // Validate manifest expert sizes match runtime computation
    size_t manifest_size_4 = [root[@"expert_size_4bit"] unsignedLongLongValue];
    size_t manifest_size_2 = [root[@"expert_size_2bit"] unsignedLongLongValue];
    if (manifest_size_4 && manifest_size_4 != cfg.expert_size_4bit) {
        fprintf(stderr, "[tiered] expert_size_4bit mismatch: manifest=%zu vs config=%zu\n",
                manifest_size_4, cfg.expert_size_4bit);
        return 0;
    }
    if (manifest_size_2 && manifest_size_2 != cfg.expert_size_2bit) {
        fprintf(stderr, "[tiered] expert_size_2bit mismatch: manifest=%zu vs config=%zu\n",
                manifest_size_2, cfg.expert_size_2bit);
        return 0;
    }

    g_tiered_manifest = calloc(num_layers * num_experts, sizeof(TieredExpertInfo));

    NSDictionary *layers = root[@"layers"];
    int errors = 0;
    for (int l = 0; l < num_layers; l++) {
        NSString *lkey = [NSString stringWithFormat:@"%d", l];
        NSDictionary *layer = layers[lkey];
        if (!layer) {
            // Missing layer: default all experts to 4-bit with sequential offsets
            fprintf(stderr, "[tiered] WARNING: layer %d missing from manifest, defaulting to 4-bit\n", l);
            for (int e = 0; e < num_experts; e++) {
                TIERED(l, e).offset = (size_t)e * cfg.expert_size_4bit;
                TIERED(l, e).size = cfg.expert_size_4bit;
                TIERED(l, e).bits = 4;
            }
            continue;
        }

        NSArray *experts = layer[@"experts"];
        for (int e = 0; e < num_experts && e < (int)[experts count]; e++) {
            NSDictionary *exp = experts[e];
            int bits = [exp[@"bits"] intValue];
            if (bits != 2 && bits != 4) {
                fprintf(stderr, "[tiered] ERROR: layer %d expert %d has invalid bits=%d\n", l, e, bits);
                errors++;
                bits = 4;  // fallback
            }
            TIERED(l, e).offset = [exp[@"offset"] unsignedLongLongValue];
            TIERED(l, e).size = [exp[@"size"] unsignedLongLongValue];
            TIERED(l, e).bits = bits;
        }
    }

    if (errors > 0) {
        fprintf(stderr, "[tiered] WARNING: %d invalid entries found in manifest\n", errors);
    }

    // Print summary
    int hot = 0, cold = 0;
    for (int l = 0; l < num_layers; l++) {
        for (int e = 0; e < num_experts; e++) {
            if (TIERED(l, e).bits == 4) hot++;
            else cold++;
        }
    }
    double threshold = [root[@"threshold"] doubleValue];
    printf("[tiered] Loaded manifest: %d hot (4-bit) + %d cold (2-bit), threshold=%.0f%%\n",
           hot, cold, threshold * 100);

    return 1;
}

// ============================================================================
// Dynamic tracking arrays (allocated after config is loaded)
// Declarations here, alloc_tracking_arrays() defined after types below.
// ============================================================================

static int *g_expert_freq = NULL;
static uint8_t *g_expert_seen = NULL;
static void **g_lz4_index = NULL;  // actually LZ4IndexEntry**, cast at use site
static uint8_t *g_cache_seen = NULL;
static uint64_t *g_cache_last_touch_token = NULL;
static uint64_t *g_cache_last_evict_token = NULL;
static int *g_pred_experts = NULL;
static int *g_pred_count = NULL;

// GPU KV cache sequence length — set from cfg.max_seq_len in metal_setup()
// Default 8192 for desktop, iOS overrides via max_seq_len cap
static int GPU_KV_SEQ = 8192;

// Helper macros for flattened 2D access
#define FREQ(l, e)           g_expert_freq[(l) * cfg.num_experts + (e)]
#define EXPERT_SEEN_BYTE(l, e) g_expert_seen[(l) * ((cfg.num_experts + 7) / 8) + ((e) >> 3)]
#define CACHE_SEEN(l, e)     g_cache_seen[(l) * cfg.num_experts + (e)]
#define CACHE_TOUCH(l, e)    g_cache_last_touch_token[(l) * cfg.num_experts + (e)]
#define CACHE_EVICT(l, e)    g_cache_last_evict_token[(l) * cfg.num_experts + (e)]
#define PRED_EXPERT(l, k)    g_pred_experts[(l) * MAX_K + (k)]
#define PRED_COUNT(l)        g_pred_count[(l)]

// Forward declaration — defined after LayerWeightCache and LZ4IndexEntry
static void alloc_tracking_arrays(void);
