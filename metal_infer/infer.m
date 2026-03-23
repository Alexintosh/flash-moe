/*
 * infer.m — Qwen3.5 MoE inference engine using Metal
 *
 * Unity build entry point: includes all modules in dependency order.
 * Each module is a .h file that is NOT compiled independently.
 *
 * Full forward pass: embedding -> N transformer layers -> norm -> lm_head -> sample
 * Model architecture loaded at runtime from HuggingFace config.json (--model flag).
 * Non-expert weights loaded from model_weights.bin (mmap'd at startup)
 * Expert weights loaded from packed_experts/ per layer per token (pread)
 *
 * Supported: Qwen3.5-35B-A3B, Qwen3.5-397B-A17B, and compatible MoE variants.
 * Architecture auto-detected from config.json:
 *   - N layers: mix of linear attention (GatedDeltaNet) + full attention
 *   - Configurable hidden_size, head_dim, num_attention_heads, num_kv_heads
 *   - Variable experts/layer and active experts (K)
 *   - Shared expert per layer (always active)
 *   - Linear attention: conv1d + gated delta recurrence
 *   - Full attention: standard QKV + scaled dot product + RoPE
 *
 * Command buffer optimization (fused_layer_forward):
 *   Per-layer Metal command buffer structure:
 *     CMD1: attention input projections (3-4 dispatches, 1 commit)
 *     CPU:  attention compute (RoPE/softmax/delta-net)
 *     CMD2: o_proj + residual_add + rms_norm + routing + shared gate/up (8 encoders, 1 commit)
 *           GPU handles residual connection and post-attn norm internally,
 *           eliminating the CPU round-trip that previously split this into 2 cmd buffers.
 *     CPU:  softmax + top-K + pread all K experts (4 pthreads parallel)
 *     CMD3: all K expert forwards + shared SwiGLU + shared down
 *           + GPU-side combine + residual_add + rms_norm -> buf_input (DEFERRED commit)
 *           Batched encoding: 4 encoders for K experts + 2 shared + 3 combine = 9 total
 *   Total: 3 cmd buffers per layer. CMD3 is submitted async (commit without wait).
 *   GPU-side combine in CMD3: for non-last layers, CMD3 also computes:
 *     moe_combine_residual (weighted sum + residual + shared gate -> hidden)
 *     rms_norm (hidden -> buf_input using NEXT layer's input_norm weights)
 *   This allows the next layer's CMD1 to submit immediately without waiting
 *   for CMD3 completion — the GPU queue serializes CMD3(N-1) then CMD1(N).
 *   Saves ~0.83ms/layer deferred_wait + CPU combine + input_norm overhead.
 *   Multi-expert buffers (MAX_K=16 independent slots) allow all K expert
 *   forwards to be encoded into a single command buffer.
 *   Batched encoding: 2 encoders per expert (gate+up fused, SwiGLU+down fused)
 *   + 2 for shared expert = K*2 + 2 total encoders in CMD3.
 *   Double-buffered expert data (buf_multi_expert_data / data_B) for future
 *   async pread overlap with GPU compute.
 *
 * Build:  clang -O2 -Wall -fobjc-arc -framework Metal -framework Foundation -lpthread infer.m -o infer
 * Run:    ./infer --prompt "Explain relativity" --tokens 50
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <math.h>
#include <getopt.h>
#include <pthread.h>
#include <errno.h>
#include <dispatch/dispatch.h>
#include <Accelerate/Accelerate.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/wait.h>
#include <compression.h>

// ============================================================================
// Unity build: all modules included in dependency order.
// Each .h file contains static functions/globals — NOT compiled independently.
// ============================================================================

#include "config.h"          // ModelConfig, macros, load_model_config()
#include "timing.h"          // LayerTimingAccum, cache telemetry, expert tracking globals
#include "fp8.h"             // FP8 E4M3 encode/decode, g_use_fp8_kv flag
#include "weights.h"         // bf16 conversion, TensorManifest, WeightFile, hash table
#include "cpu_kernels.h"     // Vocabulary, tokenizer, CPU compute kernels
#include "metal_ctx.h"       // MetalCtx, metal_setup(), weight buffer resolution
#include "gpu_dispatch.h"    // BatchMatvecSpec, batched GPU matmul, expert GPU forward
#include "expert_io.h"       // I/O thread pool, parallel pread, LRU cache, prefetch
#include "layer_forward.h"   // RoPE, KVCache, attention, MoE, fused_layer_forward
#include "generate.h"        // Frequency analysis, HTTP serve, main()
