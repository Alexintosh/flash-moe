// metal_ctx.h — MetalCtx struct, metal_setup(), weight buffer resolution, dequant matvec
// Part of Flash-MoE modular decomposition (unity build).
// Included by infer.m — do NOT compile separately.

// ============================================================================
// Metal context for GPU-accelerated matmuls
// ============================================================================

// Maximum number of batched matmul output slots.
// Used for encoding multiple matmuls into one command buffer.
#define MAX_BATCH_SLOTS 8

typedef struct {
    id<MTLDevice>               device;
    id<MTLCommandQueue>         queue;
    id<MTLLibrary>              library;
    id<MTLComputePipelineState> matvec_v3;
    id<MTLComputePipelineState> matvec_v5;  // LUT dequant variant
    id<MTLComputePipelineState> matvec_fast;  // for in_dim > 4096
    id<MTLComputePipelineState> matvec_2bit;  // 2-bit expert dequant kernel
    id<MTLComputePipelineState> rms_norm_sum;
    id<MTLComputePipelineState> rms_norm_apply;
    id<MTLComputePipelineState> rms_norm_apply_bf16;
    id<MTLComputePipelineState> residual_add;
    id<MTLComputePipelineState> swiglu;
    // GPU attention pipelines
    id<MTLComputePipelineState> attn_scores_pipe;
    id<MTLComputePipelineState> attn_softmax_pipe;
    id<MTLComputePipelineState> attn_values_pipe;
    id<MTLComputePipelineState> sigmoid_gate_pipe;
    // FP8 E4M3 KV cache attention pipelines (opt-in via g_use_fp8_kv)
    id<MTLComputePipelineState> attn_scores_fp8_pipe;
    id<MTLComputePipelineState> attn_values_fp8_pipe;
    // Reusable buffers for attention matmuls
    id<MTLBuffer> buf_input;     // input vector [cfg.hidden_dim or max projection input]
    id<MTLBuffer> buf_output;    // output vector [max projection output]
    id<MTLBuffer> wf_buf;        // the mmap'd weight file as a Metal buffer (first chunk or sole)
    // Split weight file into multiple Metal buffers for files > 4GB (Metal limit on iOS)
    #define MAX_WF_CHUNKS 4
    id<MTLBuffer> wf_chunks[MAX_WF_CHUNKS];
    size_t wf_chunk_offsets[MAX_WF_CHUNKS]; // byte offset of each chunk from mmap base
    size_t wf_chunk_sizes[MAX_WF_CHUNKS];
    int wf_num_chunks;
    void *wf_mmap_base;         // base of mmap'd weight file
    size_t wf_mmap_size;        // size of mmap'd weight file
    // iOS staging mode: when wf_num_chunks == 0, GPU dispatches copy tensor data
    // into this reusable staging buffer instead of using zero-copy Metal buffers.
    // This avoids Metal tracking the full 5.5GB weight file as GPU memory.
    id<MTLBuffer> wf_staging;   // reusable staging buffer (~50MB, iOS only)
    size_t wf_staging_used;     // bytes currently packed into staging buffer
    // Batched matmul output slots (preallocated, reused across dispatches)
    id<MTLBuffer> batch_out[MAX_BATCH_SLOTS];
    // Reusable buffers for expert computation (avoids per-expert alloc)
    // Legacy single-expert buffers (kept for gpu_expert_forward compat)
    id<MTLBuffer> buf_expert_data;   // holds one expert's packed weights (cfg.expert_size_4bit bytes)
    id<MTLBuffer> buf_expert_input;  // h_post input [cfg.hidden_dim floats]
    id<MTLBuffer> buf_expert_gate;   // gate_proj output [cfg.moe_intermediate floats]
    id<MTLBuffer> buf_expert_up;     // up_proj output [cfg.moe_intermediate floats]
    id<MTLBuffer> buf_expert_act;    // SwiGLU output [cfg.moe_intermediate floats]
    id<MTLBuffer> buf_expert_out;    // down_proj output [cfg.hidden_dim floats]
    // Multi-expert buffers: K independent sets so all experts can be encoded
    // into a SINGLE command buffer (no per-expert commit+wait).
    // Each expert k uses slot [k].
    // Double-buffered: set A (data) for GPU compute, set B (data_B) for background pread.
    // Gate/up/act/out only need one set (GPU uses them after pread completes).
    #define MAX_K 16
    id<MTLBuffer> buf_multi_expert_data[MAX_K];   // [cfg.expert_size_4bit bytes] each — buffer set A
    id<MTLBuffer> buf_multi_expert_data_B[MAX_K]; // [cfg.expert_size_4bit bytes] each — buffer set B (prefetch)
    id<MTLBuffer> buf_multi_expert_gate[MAX_K];   // [cfg.moe_intermediate floats]
    id<MTLBuffer> buf_multi_expert_up[MAX_K];     // [cfg.moe_intermediate floats]
    id<MTLBuffer> buf_multi_expert_act[MAX_K];    // [cfg.moe_intermediate floats]
    id<MTLBuffer> buf_multi_expert_out[MAX_K];    // [cfg.hidden_dim floats]
    id<MTLBuffer> buf_multi_expert_input;         // [cfg.hidden_dim floats] (shared, read-only during dispatch)
    // Shared expert buffers for fused CMD2 (shared gate/up computed in CMD1,
    // SwiGLU + down_proj in CMD2 alongside routed experts)
    id<MTLBuffer> buf_shared_gate;   // [cfg.shared_intermediate floats]
    id<MTLBuffer> buf_shared_up;     // [cfg.shared_intermediate floats]
    id<MTLBuffer> buf_shared_act;    // [cfg.shared_intermediate floats] (SwiGLU output)
    id<MTLBuffer> buf_shared_out;    // [cfg.hidden_dim floats] (down_proj output)
    // Fused o_proj+norm+routing buffers (eliminates 1 cmd buffer per layer)
    id<MTLBuffer> buf_residual;     // [cfg.hidden_dim floats] holds residual for GPU add
    id<MTLBuffer> buf_h_mid;        // [cfg.hidden_dim floats] residual+oproj result
    id<MTLBuffer> buf_sum_sq;       // [1 float] for RMS norm reduction
    // GPU attention buffers (for full attention layers)
    id<MTLBuffer> __strong *buf_kv_k;  // K cache per full-attn layer (float or uchar when FP8)
    id<MTLBuffer> __strong *buf_kv_v;  // V cache per full-attn layer (float or uchar when FP8)
    // FP8 per-position scale buffers (1 float per cached position per layer)
    id<MTLBuffer> __strong *buf_kv_k_scales;  // [gpu_kv floats] per full-attn layer (FP8 only)
    id<MTLBuffer> __strong *buf_kv_v_scales;  // [gpu_kv floats] per full-attn layer (FP8 only)
    id<MTLBuffer> buf_attn_q;       // [cfg.num_attn_heads * cfg.head_dim floats] all query heads
    id<MTLBuffer> buf_attn_scores;  // [cfg.num_attn_heads * cfg.max_seq_len floats] all heads' scores
    id<MTLBuffer> buf_attn_out;     // [cfg.num_attn_heads * cfg.head_dim floats] full attention output
    id<MTLBuffer> buf_attn_gate;    // [cfg.num_attn_heads * cfg.head_dim floats] sigmoid gate
    // CMD3 GPU-side combine buffers (weighted_sum + residual + norm on GPU)
    id<MTLComputePipelineState> moe_combine_residual;  // fused combine kernel
    id<MTLBuffer> buf_moe_hidden;     // [cfg.hidden_dim floats] GPU combine output (hidden state)
    id<MTLBuffer> buf_combine_params; // [10 floats] expert weights[8] + shared_gate_score + padding
    id<MTLBuffer> buf_cmd3_sum_sq;    // [1 float] for RMS norm reduction in CMD3
    // Shared event for CPU-GPU synchronization (async pipeline)
    id<MTLSharedEvent> pipeline_event;   // CPU signals when buf_input is ready
    uint64_t event_value;                // monotonically increasing event counter
    // GPU delta-net (gated_delta_net_step) and conv1d pipelines
    id<MTLComputePipelineState> delta_net_step;  // gated_delta_net_step kernel
    id<MTLComputePipelineState> delta_net_step_fused;  // gated_delta_net_step_fused kernel (pass 2+3 merged)
    id<MTLComputePipelineState> conv1d_step;     // conv1d_step kernel
    id<MTLComputePipelineState> rms_norm_qk;     // per-head RMS normalize for q and k
    id<MTLComputePipelineState> compute_decay_beta; // g_decay and beta_gate for delta-net
    id<MTLComputePipelineState> gated_rms_norm;  // z-gated output normalization
    // Persistent GPU state buffers for linear attention layers
    id<MTLBuffer> __strong *buf_delta_state;   // [v_heads*v_dim*k_dim] float per layer
    id<MTLBuffer> __strong *buf_conv_state;     // [(kernel-1)*conv_dim] float per layer
    // Scratch buffers for delta-net inputs/outputs
    id<MTLBuffer> buf_delta_q;        // [cfg.linear_total_key=2048] float
    id<MTLBuffer> buf_delta_k;        // [cfg.linear_total_key=2048] float
    id<MTLBuffer> buf_delta_v;        // [cfg.linear_total_value=4096] float
    id<MTLBuffer> buf_delta_g_decay;  // [cfg.linear_num_v_heads=32] float
    id<MTLBuffer> buf_delta_beta;     // [cfg.linear_num_v_heads=32] float
    id<MTLBuffer> buf_delta_output;   // [cfg.linear_total_value=4096] float
    id<MTLBuffer> buf_conv_input;     // [cfg.linear_conv_dim=8192] float
    id<MTLBuffer> buf_conv_output;    // [cfg.linear_conv_dim=8192] float
    // Wired memory budget from Metal device
    size_t recommended_working_set;   // [ctx->device recommendedMaxWorkingSetSize]
} MetalCtx;

static MetalCtx *g_metal = NULL;

static MetalCtx *metal_setup(void) {
    // Set GPU KV cache size from model config (avoids over-allocating on iOS)
    if (cfg.max_seq_len > 0 && cfg.max_seq_len < GPU_KV_SEQ) {
        GPU_KV_SEQ = cfg.max_seq_len;
    }
    printf("[metal] GPU_KV_SEQ = %d\n", GPU_KV_SEQ);

    MetalCtx *ctx = calloc(1, sizeof(MetalCtx));
    // Allocate dynamic buffer arrays based on config
    ctx->buf_kv_k       = (__strong id<MTLBuffer> *)calloc(cfg.num_full_attn_layers, sizeof(id<MTLBuffer>));
    ctx->buf_kv_v       = (__strong id<MTLBuffer> *)calloc(cfg.num_full_attn_layers, sizeof(id<MTLBuffer>));
    ctx->buf_kv_k_scales = (__strong id<MTLBuffer> *)calloc(cfg.num_full_attn_layers, sizeof(id<MTLBuffer>));
    ctx->buf_kv_v_scales = (__strong id<MTLBuffer> *)calloc(cfg.num_full_attn_layers, sizeof(id<MTLBuffer>));
    ctx->buf_delta_state = (__strong id<MTLBuffer> *)calloc(cfg.num_linear_layers, sizeof(id<MTLBuffer>));
    ctx->buf_conv_state  = (__strong id<MTLBuffer> *)calloc(cfg.num_linear_layers, sizeof(id<MTLBuffer>));
    ctx->device = MTLCreateSystemDefaultDevice();
    if (!ctx->device) {
        fprintf(stderr, "ERROR: No Metal device\n");
        free(ctx); return NULL;
    }
    printf("[metal] Device: %s\n", [[ctx->device name] UTF8String]);

    ctx->recommended_working_set = (size_t)[ctx->device recommendedMaxWorkingSetSize];
    printf("[metal] Recommended working set: %.1f GB\n", ctx->recommended_working_set / 1e9);

    ctx->queue = [ctx->device newCommandQueue];
    if (!ctx->queue) {
        fprintf(stderr, "ERROR: No command queue\n");
        free(ctx); return NULL;
    }

    // Load Metal shaders
    NSError *error = nil;
    double t0 = now_ms();

    // Try pre-compiled default.metallib first (iOS app bundle, or macOS with embedded metallib)
    ctx->library = [ctx->device newDefaultLibrary];
    if (ctx->library) {
        printf("[metal] Loaded pre-compiled Metal library: %.0f ms\n", now_ms() - t0);
    } else {
        // Fallback: compile shaders from source at runtime (macOS CLI)
        NSArray *paths = @[@"shaders.metal", @"metal_infer/shaders.metal"];
        NSString *src = nil;
        for (NSString *p in paths) {
            src = [NSString stringWithContentsOfFile:p encoding:NSUTF8StringEncoding error:&error];
            if (src) break;
        }
        if (!src) {
            fprintf(stderr, "ERROR: Cannot find shaders.metal\n");
            free(ctx); return NULL;
        }

        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
        opts.mathMode = MTLMathModeFast;
        opts.languageVersion = MTLLanguageVersion3_1;
        ctx->library = [ctx->device newLibraryWithSource:src options:opts error:&error];
        if (!ctx->library) {
            fprintf(stderr, "ERROR: Shader compile failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            free(ctx); return NULL;
        }
        printf("[metal] Shader compile: %.0f ms\n", now_ms() - t0);
    }

    // Create pipelines
    id<MTLComputePipelineState> (^makePipe)(NSString *) = ^(NSString *name) {
        id<MTLFunction> fn = [ctx->library newFunctionWithName:name];
        if (!fn) { fprintf(stderr, "ERROR: shader '%s' not found\n", [name UTF8String]); return (id<MTLComputePipelineState>)nil; }
        NSError *e2 = nil;
        id<MTLComputePipelineState> ps = [ctx->device newComputePipelineStateWithFunction:fn error:&e2];
        if (!ps) { fprintf(stderr, "ERROR: pipeline '%s': %s\n", [name UTF8String], [[e2 localizedDescription] UTF8String]); }
        return ps;
    };

    ctx->matvec_v3     = makePipe(@"dequant_matvec_4bit_v3");
    ctx->matvec_v5     = makePipe(@"dequant_matvec_4bit_v5");  // LUT variant (no uint→float conversions)
    ctx->matvec_fast   = makePipe(@"dequant_matvec_4bit_fast");
    ctx->matvec_2bit   = makePipe(@"dequant_matvec_2bit");
    ctx->rms_norm_sum  = makePipe(@"rms_norm_sum_sq");
    ctx->rms_norm_apply = makePipe(@"rms_norm_apply");
    ctx->rms_norm_apply_bf16 = makePipe(@"rms_norm_apply_bf16");
    ctx->residual_add  = makePipe(@"residual_add");
    ctx->swiglu        = makePipe(@"swiglu_fused");
    ctx->attn_scores_pipe  = makePipe(@"attn_scores_batched");
    ctx->attn_softmax_pipe = makePipe(@"attn_softmax_batched");
    ctx->attn_values_pipe  = makePipe(@"attn_values_batched");
    ctx->sigmoid_gate_pipe = makePipe(@"sigmoid_gate");
    // FP8 E4M3 KV cache attention kernels (optional — only needed when g_use_fp8_kv)
    ctx->attn_scores_fp8_pipe = makePipe(@"attn_scores_fp8");
    ctx->attn_values_fp8_pipe = makePipe(@"attn_values_fp8");
    ctx->moe_combine_residual = makePipe(@"moe_combine_residual");
    ctx->delta_net_step    = makePipe(@"gated_delta_net_step");
    ctx->delta_net_step_fused = makePipe(@"gated_delta_net_step_fused");
    ctx->conv1d_step       = makePipe(@"conv1d_step");
    ctx->rms_norm_qk       = makePipe(@"rms_norm_qk");
    ctx->compute_decay_beta = makePipe(@"compute_decay_beta");
    ctx->gated_rms_norm    = makePipe(@"gated_rms_norm");
    if (!ctx->moe_combine_residual) fprintf(stderr, "[metal] WARNING: moe_combine_residual pipeline failed\n");
    if (!ctx->delta_net_step) fprintf(stderr, "[metal] WARNING: gated_delta_net_step pipeline failed (CPU fallback)\n");
    if (!ctx->delta_net_step_fused) fprintf(stderr, "[metal] WARNING: gated_delta_net_step_fused pipeline failed (using unfused fallback)\n");
    if (!ctx->conv1d_step)    fprintf(stderr, "[metal] WARNING: conv1d_step pipeline failed (CPU fallback)\n");
    if (!ctx->rms_norm_qk)       fprintf(stderr, "[metal] WARNING: rms_norm_qk pipeline failed (CPU fallback)\n");
    if (!ctx->compute_decay_beta) fprintf(stderr, "[metal] WARNING: compute_decay_beta pipeline failed (CPU fallback)\n");
    if (!ctx->gated_rms_norm)     fprintf(stderr, "[metal] WARNING: gated_rms_norm pipeline failed (CPU fallback)\n");

    if (!ctx->matvec_v3 || !ctx->matvec_fast) {
        fprintf(stderr, "ERROR: Required Metal pipeline missing\n");
        free(ctx); return NULL;
    }

    // Allocate reusable buffers (large enough for biggest projection)
    // Q proj output is 16384 floats, lm_head output is 248320 floats
    // o_proj input is 8192, linear attn out_proj input is 8192
    size_t max_out = cfg.vocab_size * sizeof(float);  // lm_head is largest
    size_t max_in = cfg.linear_total_value * sizeof(float);  // 8192 floats (linear_attn out_proj)
    if (max_in < (size_t)(cfg.num_attn_heads * cfg.head_dim) * sizeof(float)) {
        max_in = (size_t)(cfg.num_attn_heads * cfg.head_dim) * sizeof(float);  // o_proj input = 8192
    }
    ctx->buf_input  = [ctx->device newBufferWithLength:max_in  options:MTLResourceStorageModeShared];
    ctx->buf_output = [ctx->device newBufferWithLength:max_out options:MTLResourceStorageModeShared];

    // Batched matmul output slots — each large enough for the biggest projection
    // q_proj = 16384 floats, qkv_proj = 12288, z_proj = 8192, o_proj = 4096
    // lm_head (248320) uses buf_output directly, not batched.
    {
        size_t slot_size = (size_t)(cfg.num_attn_heads * cfg.head_dim * 2) * sizeof(float);  // 16384 floats
        if (slot_size < (size_t)cfg.linear_conv_dim * sizeof(float))
            slot_size = (size_t)cfg.linear_conv_dim * sizeof(float);  // 12288 floats
        for (int i = 0; i < MAX_BATCH_SLOTS; i++) {
            ctx->batch_out[i] = [ctx->device newBufferWithLength:slot_size
                                                         options:MTLResourceStorageModeShared];
        }
    }

    // Expert computation buffers (reused across all experts and layers)
    ctx->buf_expert_data  = [ctx->device newBufferWithLength:cfg.expert_size_4bit
                                                     options:MTLResourceStorageModeShared];
    ctx->buf_expert_input = [ctx->device newBufferWithLength:cfg.hidden_dim * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    ctx->buf_expert_gate  = [ctx->device newBufferWithLength:cfg.moe_intermediate * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    ctx->buf_expert_up    = [ctx->device newBufferWithLength:cfg.moe_intermediate * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    ctx->buf_expert_act   = [ctx->device newBufferWithLength:cfg.moe_intermediate * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    ctx->buf_expert_out   = [ctx->device newBufferWithLength:cfg.hidden_dim * sizeof(float)
                                                     options:MTLResourceStorageModeShared];

    // Multi-expert buffers: K independent slots (double-buffered data)
    // Expert data buffers use 2MB-aligned backing memory for DMA efficiency.
    // The pread DMA controller transfers 3.6x faster with 2MB alignment vs 16KB.
    ctx->buf_multi_expert_input = [ctx->device newBufferWithLength:cfg.hidden_dim * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
    size_t expert_alloc_size = (cfg.expert_size_4bit + 2*1024*1024 - 1) & ~(2*1024*1024 - 1);  // round up to 2MB
    for (int k = 0; k < MAX_K; k++) {
        // 2MB-aligned allocation for optimal DMA throughput
        void *aligned_data = NULL, *aligned_data_b = NULL;
        posix_memalign(&aligned_data,   2*1024*1024, expert_alloc_size);
        posix_memalign(&aligned_data_b, 2*1024*1024, expert_alloc_size);
        memset(aligned_data, 0, expert_alloc_size);
        memset(aligned_data_b, 0, expert_alloc_size);
        ctx->buf_multi_expert_data[k] = [ctx->device newBufferWithBytesNoCopy:aligned_data
                                                                       length:expert_alloc_size
                                                                      options:MTLResourceStorageModeShared
                                                                  deallocator:nil];
        ctx->buf_multi_expert_data_B[k] = [ctx->device newBufferWithBytesNoCopy:aligned_data_b
                                                                         length:expert_alloc_size
                                                                        options:MTLResourceStorageModeShared
                                                                    deallocator:nil];
        ctx->buf_multi_expert_gate[k] = [ctx->device newBufferWithLength:cfg.moe_intermediate * sizeof(float)
                                                                 options:MTLResourceStorageModeShared];
        ctx->buf_multi_expert_up[k]   = [ctx->device newBufferWithLength:cfg.moe_intermediate * sizeof(float)
                                                                 options:MTLResourceStorageModeShared];
        ctx->buf_multi_expert_act[k]  = [ctx->device newBufferWithLength:cfg.moe_intermediate * sizeof(float)
                                                                 options:MTLResourceStorageModeShared];
        ctx->buf_multi_expert_out[k]  = [ctx->device newBufferWithLength:cfg.hidden_dim * sizeof(float)
                                                                 options:MTLResourceStorageModeShared];
    }

    // Shared expert buffers (for fused CMD2)
    ctx->buf_shared_gate = [ctx->device newBufferWithLength:cfg.shared_intermediate * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    ctx->buf_shared_up   = [ctx->device newBufferWithLength:cfg.shared_intermediate * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    ctx->buf_shared_act  = [ctx->device newBufferWithLength:cfg.shared_intermediate * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    ctx->buf_shared_out  = [ctx->device newBufferWithLength:cfg.hidden_dim * sizeof(float)
                                                    options:MTLResourceStorageModeShared];

    // Fused o_proj+norm+routing buffers
    ctx->buf_residual = [ctx->device newBufferWithLength:cfg.hidden_dim * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
    ctx->buf_h_mid    = [ctx->device newBufferWithLength:cfg.hidden_dim * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
    ctx->buf_sum_sq   = [ctx->device newBufferWithLength:sizeof(float)
                                                 options:MTLResourceStorageModeShared];

    // CMD3 GPU-side combine buffers
    ctx->buf_moe_hidden    = [ctx->device newBufferWithLength:cfg.hidden_dim * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
    ctx->buf_combine_params = [ctx->device newBufferWithLength:10 * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
    ctx->buf_cmd3_sum_sq    = [ctx->device newBufferWithLength:sizeof(float)
                                                        options:MTLResourceStorageModeShared];

    // GPU attention buffers — sized to min(g_kv_seq_len, GPU_KV_SEQ)
    // When FP8 KV is enabled, KV buffers use uint8_t (1 byte) instead of float (4 bytes)
    {
        size_t kv_dim = cfg.num_kv_heads * cfg.head_dim;  // 512
        int gpu_kv = GPU_KV_SEQ;
        if (g_kv_seq_len > 0 && g_kv_seq_len < gpu_kv) gpu_kv = g_kv_seq_len;
        size_t elem_size = g_use_fp8_kv ? sizeof(uint8_t) : sizeof(float);
        size_t kv_cache_size = (size_t)gpu_kv * kv_dim * elem_size;
        for (int i = 0; i < cfg.num_full_attn_layers; i++) {
            ctx->buf_kv_k[i] = [ctx->device newBufferWithLength:kv_cache_size
                                                        options:MTLResourceStorageModeShared];
            ctx->buf_kv_v[i] = [ctx->device newBufferWithLength:kv_cache_size
                                                        options:MTLResourceStorageModeShared];
        }
        // FP8: allocate per-position scale buffers (1 float per position per layer)
        if (g_use_fp8_kv) {
            size_t scale_buf_size = (size_t)gpu_kv * sizeof(float);
            for (int i = 0; i < cfg.num_full_attn_layers; i++) {
                ctx->buf_kv_k_scales[i] = [ctx->device newBufferWithLength:scale_buf_size
                                                            options:MTLResourceStorageModeShared];
                ctx->buf_kv_v_scales[i] = [ctx->device newBufferWithLength:scale_buf_size
                                                            options:MTLResourceStorageModeShared];
            }
        }
        ctx->buf_attn_q      = [ctx->device newBufferWithLength:cfg.num_attn_heads * cfg.head_dim * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        ctx->buf_attn_scores = [ctx->device newBufferWithLength:(size_t)cfg.num_attn_heads * GPU_KV_SEQ * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        ctx->buf_attn_out    = [ctx->device newBufferWithLength:cfg.num_attn_heads * cfg.head_dim * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        ctx->buf_attn_gate   = [ctx->device newBufferWithLength:cfg.num_attn_heads * cfg.head_dim * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        printf("[metal] GPU attention buffers: %d KV caches (%.1f MB each%s), scores buf %.1f MB\n",
               cfg.num_full_attn_layers, kv_cache_size / 1e6,
               g_use_fp8_kv ? ", FP8 E4M3" : "",
               (double)(cfg.num_attn_heads * cfg.max_seq_len * sizeof(float)) / 1e6);
    }

    // Persistent GPU state buffers for delta-net (linear attention layers)
    if (ctx->delta_net_step) {
        for (int i = 0; i < cfg.num_linear_layers; i++) {
            ctx->buf_delta_state[i] = [ctx->device newBufferWithLength:(size_t)cfg.linear_num_v_heads*cfg.linear_value_dim*cfg.linear_key_dim*sizeof(float)
                                                               options:MTLResourceStorageModeShared];
            memset([ctx->buf_delta_state[i] contents], 0, (size_t)cfg.linear_num_v_heads*cfg.linear_value_dim*cfg.linear_key_dim*sizeof(float));
            ctx->buf_conv_state[i] = [ctx->device newBufferWithLength:(cfg.conv_kernel_size-1)*(size_t)cfg.linear_conv_dim*sizeof(float)
                                                              options:MTLResourceStorageModeShared];
            memset([ctx->buf_conv_state[i] contents], 0, (cfg.conv_kernel_size-1)*(size_t)cfg.linear_conv_dim*sizeof(float));
        }
        // Scratch buffers for delta-net inputs/outputs (allocated once, reused)
        ctx->buf_delta_q       = [ctx->device newBufferWithLength:cfg.linear_total_key*sizeof(float)    options:MTLResourceStorageModeShared];
        ctx->buf_delta_k       = [ctx->device newBufferWithLength:cfg.linear_total_key*sizeof(float)    options:MTLResourceStorageModeShared];
        ctx->buf_delta_v       = [ctx->device newBufferWithLength:cfg.linear_total_value*sizeof(float)  options:MTLResourceStorageModeShared];
        ctx->buf_delta_g_decay = [ctx->device newBufferWithLength:cfg.linear_num_v_heads*sizeof(float)  options:MTLResourceStorageModeShared];
        ctx->buf_delta_beta    = [ctx->device newBufferWithLength:cfg.linear_num_v_heads*sizeof(float)  options:MTLResourceStorageModeShared];
        ctx->buf_delta_output  = [ctx->device newBufferWithLength:cfg.linear_total_value*sizeof(float)  options:MTLResourceStorageModeShared];
        ctx->buf_conv_input    = [ctx->device newBufferWithLength:cfg.linear_conv_dim*sizeof(float)     options:MTLResourceStorageModeShared];
        ctx->buf_conv_output   = [ctx->device newBufferWithLength:cfg.linear_conv_dim*sizeof(float)     options:MTLResourceStorageModeShared];
        size_t state_bytes = (size_t)cfg.linear_num_v_heads*cfg.linear_value_dim*cfg.linear_key_dim*sizeof(float);
        size_t conv_bytes = (cfg.conv_kernel_size-1)*(size_t)cfg.linear_conv_dim*sizeof(float);
        printf("[metal] Delta-net GPU buffers: %d layers (%.1f MB state + %.1f MB scratch)\n",
               cfg.num_linear_layers,
               cfg.num_linear_layers * (state_bytes + conv_bytes) / 1e6,
               (cfg.linear_total_key*2+cfg.linear_total_value*2+cfg.linear_num_v_heads*2+cfg.linear_conv_dim*2) * sizeof(float) / 1e6);
    }

    // Create shared event for CPU-GPU async pipeline
    ctx->pipeline_event = [ctx->device newSharedEvent];
    ctx->event_value = 0;

    // Report total GPU allocation and warn if over budget
    {
        size_t allocated = (size_t)[ctx->device currentAllocatedSize];
        printf("[metal] Current GPU allocation: %.1f MB\n", allocated / 1e6);
        if (allocated > ctx->recommended_working_set) {
            fprintf(stderr, "[metal] WARNING: GPU allocation (%.1f MB) exceeds recommended working set (%.1f MB)\n",
                    allocated / 1e6, ctx->recommended_working_set / 1e6);
        }
    }

    printf("[metal] Inference pipelines ready (multi-expert[%d] + shared buffers allocated)\n", MAX_K);
    return ctx;
}

// Reset delta-net and conv GPU state buffers (call at start of new generation)
static void reset_delta_net_state(void) {
    if (!g_metal || !g_metal->delta_net_step) return;
    for (int i = 0; i < cfg.num_linear_layers; i++) {
        if (g_metal->buf_delta_state[i])
            memset([g_metal->buf_delta_state[i] contents], 0, (size_t)cfg.linear_num_v_heads*cfg.linear_value_dim*cfg.linear_key_dim*sizeof(float));
        if (g_metal->buf_conv_state[i])
            memset([g_metal->buf_conv_state[i] contents], 0, (cfg.conv_kernel_size-1)*(size_t)cfg.linear_conv_dim*sizeof(float));
    }
}

// Wrap the mmap'd weight file as Metal buffer(s) (zero-copy on unified memory).
// Metal enforces a hard 4GB per-buffer limit. Files >4GB get two overlapping buffers.
#define METAL_MAX_BUF ((size_t)4096 * 1024 * 1024 - 16384)  // 4GB - 1 page
#define WF_STAGING_SIZE ((size_t)50 * 1024 * 1024)  // legacy, kept for compile compat
static void metal_set_weights(MetalCtx *ctx, void *data, size_t size) {
    size_t page_size = 16384;
    ctx->wf_mmap_base = data;
    ctx->wf_mmap_size = size;
    ctx->wf_num_chunks = 0;
    ctx->wf_staging = nil;
    ctx->wf_staging_used = 0;
    ctx->wf_buf = nil;

    size_t aligned_size = (size + page_size - 1) & ~(page_size - 1);

    if (size <= METAL_MAX_BUF) {
        // Fits in one buffer — zero-copy wrap
        ctx->wf_buf = [ctx->device newBufferWithBytesNoCopy:data
                                                     length:aligned_size
                                                    options:MTLResourceStorageModeShared
                                                deallocator:nil];
        if (ctx->wf_buf) {
            ctx->wf_chunks[0] = ctx->wf_buf;
            ctx->wf_chunk_offsets[0] = 0;
            ctx->wf_chunk_sizes[0] = aligned_size;
            ctx->wf_num_chunks = 1;
            printf("[metal] Weight file wrapped as single Metal buffer (%.2f GB)\n", size / 1e9);
        } else {
            fprintf(stderr, "WARNING: Cannot wrap weight file (%.2f GB) — CPU fallback\n", size / 1e9);
        }
    } else {
        // >4GB: too large for Metal buffers on memory-constrained devices.
        // CPU fallback for non-expert matmuls. The mmap is still used — just no Metal wrapper.
        // All guards (g_metal->wf_buf, wf_num_chunks > 0) will route to CPU path.
        printf("[metal] Weight file %.2f GB exceeds 4GB Metal buffer limit.\n", size / 1e9);
        printf("[metal]   wf_buf=%s wf_num_chunks=%d — CPU fallback for non-expert matmuls.\n",
               ctx->wf_buf ? "SET" : "nil", ctx->wf_num_chunks);
    }
    printf("[metal] metal_set_weights done: wf_buf=%s wf_num_chunks=%d\n",
           ctx->wf_buf ? "SET" : "nil", ctx->wf_num_chunks);
}

// GPU dequant matvec: out[out_dim] = W_4bit * x[in_dim]
// W_packed, scales, biases are pointers into mmap'd weight file
// x_f32 is CPU float array, result written back to out_f32
//
// We wrap the ENTIRE mmap'd weight file as a single Metal buffer and use
// byte offsets to point each shader argument at the right tensor.
// This avoids per-tensor buffer creation and the page-alignment constraint.

// ============================================================================
// Weight buffer resolution: chunk mode (macOS) vs staging mode (iOS)
// ============================================================================

// Reset staging buffer offset — call at start of each command buffer encode
static inline void metal_staging_reset(MetalCtx *ctx) {
    ctx->wf_staging_used = 0;
}

// Stage a tensor into the staging buffer. Returns buffer + offset within staging.
// Tensors are packed sequentially; call metal_staging_reset() between command buffers.
static inline void metal_stage(MetalCtx *ctx, const void *ptr, size_t size,
                                id<MTLBuffer> *out_buf, NSUInteger *out_offset) {
    if (ctx->wf_staging_used + size > WF_STAGING_SIZE) {
        fprintf(stderr, "ERROR: staging buffer overflow: used=%zu + size=%zu > %zu\n",
                ctx->wf_staging_used, size, (size_t)WF_STAGING_SIZE);
        // Reset and overwrite from beginning (data corruption but won't crash)
        ctx->wf_staging_used = 0;
    }
    memcpy((char *)[ctx->wf_staging contents] + ctx->wf_staging_used, ptr, size);
    *out_buf = ctx->wf_staging;
    *out_offset = (NSUInteger)ctx->wf_staging_used;
    ctx->wf_staging_used += size;
    // Align to 16 bytes for Metal buffer offset requirements
    ctx->wf_staging_used = (ctx->wf_staging_used + 15) & ~(size_t)15;
}

// Find which Metal buffer chunk contains a given pointer, return the buffer and offset within it.
// In staging mode (wf_num_chunks == 0, iOS), copies tensor data into staging buffer.
// The `size` parameter is only used in staging mode — pass 0 on macOS chunk path.
static inline void metal_find_chunk_sized(MetalCtx *ctx, const void *ptr, size_t size,
                                           id<MTLBuffer> *out_buf, NSUInteger *out_offset) {
    // No Metal weight buffers (>4GB on iOS) — should not be called
    if (ctx->wf_num_chunks == 0 && !ctx->wf_staging) {
        fprintf(stderr, "BUG: metal_find_chunk called with no weight buffers! Using buf_input as dummy.\n");
        *out_buf = ctx->buf_input;
        *out_offset = 0;
        return;
    }
    // Staging mode (iOS): copy tensor into staging buffer
    if (ctx->wf_staging && ctx->wf_num_chunks == 0) {
        metal_stage(ctx, ptr, size, out_buf, out_offset);
        return;
    }
    // Chunk mode: find the zero-copy Metal buffer containing this pointer
    size_t abs_off = (const char *)ptr - (const char *)ctx->wf_mmap_base;
    for (int i = ctx->wf_num_chunks - 1; i >= 0; i--) {
        if (abs_off >= ctx->wf_chunk_offsets[i]) {
            NSUInteger local_off = (NSUInteger)(abs_off - ctx->wf_chunk_offsets[i]);
            if (local_off < [ctx->wf_chunks[i] length]) {
                *out_buf = ctx->wf_chunks[i];
                *out_offset = local_off;
                return;
            }
        }
    }
    fprintf(stderr, "ERROR: metal_find_chunk: ptr offset %zu (%.2f GB) not in any chunk!\n",
            abs_off, abs_off / 1e9);
    for (int i = 0; i < ctx->wf_num_chunks; i++) {
        fprintf(stderr, "  chunk %d: offset=%zu size=%zu\n",
                i, ctx->wf_chunk_offsets[i], ctx->wf_chunk_sizes[i]);
    }
    *out_buf = ctx->wf_chunks[0];
    *out_offset = (NSUInteger)abs_off;
}

// Legacy wrapper — used by sites that don't know tensor size (macOS only, chunk mode)
static inline void metal_find_chunk(MetalCtx *ctx, const void *ptr,
                                     id<MTLBuffer> *out_buf, NSUInteger *out_offset) {
    metal_find_chunk_sized(ctx, ptr, 0, out_buf, out_offset);
}

// Convenience macros for macOS chunk mode (no size needed)
#define WF_OFF(ctx, ptr) ({ \
    id<MTLBuffer> _b; NSUInteger _o; \
    metal_find_chunk((ctx), (ptr), &_b, &_o); _o; })
#define WF_BUF(ctx, ptr) ({ \
    id<MTLBuffer> _b; NSUInteger _o; \
    metal_find_chunk((ctx), (ptr), &_b, &_o); _b; })

static void gpu_dequant_matvec(
    MetalCtx *ctx,
    const void *W_packed, const void *scales, const void *biases,
    const float *x_f32, float *out_f32,
    uint32_t out_dim, uint32_t in_dim, uint32_t group_size
) {
    // Copy input to Metal buffer
    memcpy([ctx->buf_input contents], x_f32, in_dim * sizeof(float));

    size_t o_size = (size_t)out_dim * sizeof(float);

    // Find correct Metal buffer chunk and offset for each tensor
    id<MTLBuffer> w_buf, s_buf, b_buf;
    NSUInteger w_off, s_off, b_off;
    size_t w_size = (size_t)out_dim * in_dim / 8;  // 4-bit packed
    size_t num_groups = (in_dim + group_size - 1) / group_size;
    size_t sb_size = (size_t)out_dim * num_groups * sizeof(uint16_t);
    metal_staging_reset(ctx);
    metal_find_chunk_sized(ctx, W_packed, w_size, &w_buf, &w_off);
    metal_find_chunk_sized(ctx, scales, sb_size, &s_buf, &s_off);
    metal_find_chunk_sized(ctx, biases, sb_size, &b_buf, &b_off);

    // Ensure output buffer is large enough
    id<MTLBuffer> o_buf = ctx->buf_output;
    if (o_size > [o_buf length]) {
        o_buf = [ctx->device newBufferWithLength:o_size options:MTLResourceStorageModeShared];
    }

    id<MTLCommandBuffer> cmdbuf = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];

    // v3 shader uses x_shared[4096], so can only handle in_dim <= 4096
    // For larger in_dim (e.g. o_proj with in_dim=8192), use matvec_fast
    int use_v3 = (in_dim <= 4096);
    [enc setComputePipelineState: use_v3 ? ctx->matvec_v3 : ctx->matvec_fast];
    [enc setBuffer:w_buf        offset:w_off atIndex:0];
    [enc setBuffer:s_buf        offset:s_off atIndex:1];
    [enc setBuffer:b_buf        offset:b_off atIndex:2];
    [enc setBuffer:ctx->buf_input offset:0   atIndex:3];
    [enc setBuffer:o_buf        offset:0     atIndex:4];
    [enc setBytes:&out_dim      length:4     atIndex:5];
    [enc setBytes:&in_dim       length:4     atIndex:6];
    [enc setBytes:&group_size   length:4     atIndex:7];

    if (use_v3) {
        // v3: tiled threadgroups, 256 threads, 8 rows per TG
        uint32_t num_tgs = (out_dim + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
        // fast: one threadgroup per output row, 64 threads per TG
        NSUInteger tg_size = 64;
        [enc dispatchThreadgroups:MTLSizeMake(out_dim, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
    }
    [enc endEncoding];
    [cmdbuf commit];
    [cmdbuf waitUntilCompleted];

    // Copy result back
    memcpy(out_f32, [o_buf contents], o_size);
}

// Wrapper: use GPU if available and weight buffer is set, CPU otherwise
static void fast_dequant_matvec(
    const uint32_t *W, const uint16_t *scales, const uint16_t *biases,
    const float *x, float *out,
    int out_dim, int in_dim, int group_size
) {
    int use_gpu = g_metal && (g_metal->wf_num_chunks > 0 || g_metal->wf_staging);
    if (use_gpu) {
        // In staging mode, check if tensors fit in staging buffer.
        // LM head (~500MB) won't fit — fall back to CPU for that.
        size_t w_size = (size_t)out_dim * in_dim / 8;
        size_t num_groups = ((unsigned)in_dim + group_size - 1) / group_size;
        size_t sb_size = (size_t)out_dim * num_groups * sizeof(uint16_t);
        size_t total = w_size + sb_size + sb_size;
        if (g_metal->wf_staging && g_metal->wf_num_chunks == 0 && total > WF_STAGING_SIZE) {
            cpu_dequant_matvec(W, scales, biases, x, out, out_dim, in_dim, group_size);
            return;
        }
        gpu_dequant_matvec(g_metal, W, scales, biases, x, out,
                           (uint32_t)out_dim, (uint32_t)in_dim, (uint32_t)group_size);
    } else {
        cpu_dequant_matvec(W, scales, biases, x, out, out_dim, in_dim, group_size);
    }
}
