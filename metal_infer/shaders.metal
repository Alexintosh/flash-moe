/*
 * shaders.metal — Optimized Metal compute shaders for 4-bit quantized MoE inference
 *
 * Core operations:
 *   1. dequant_matvec_4bit: Naive 4-bit affine dequant matvec (reference)
 *   2. dequant_matvec_4bit_fast: SIMD-optimized with simd_sum reduction
 *   3. dequant_matvec_4bit_v3: Fully optimized — tiled threadgroup, vector loads,
 *      coalesced access, shared input cache. Target: <0.1ms per matmul.
 *   4. swiglu_fused / swiglu_fused_vec4: SwiGLU activation
 *   5. weighted_sum: combine expert outputs with routing weights
 *   6. rms_norm: RMS normalization
 *
 * Quantization format (MLX affine 4-bit, group_size=64):
 *   - Weights stored as uint32, each holding 8 x 4-bit values
 *   - Per-group scale and bias in bfloat16
 *   - Dequantized value = uint4_val * scale + bias
 *   - Groups of 64 elements share one (scale, bias) pair
 *
 * Matrix layout for expert projections:
 *   gate_proj/up_proj: [1024, 512] uint32 = [1024, 4096] logical (out=1024, in=4096)
 *   down_proj: [4096, 128] uint32 = [4096, 1024] logical (out=4096, in=1024)
 *
 *   Scales/biases: [out_dim, in_dim/group_size]
 *   gate/up scales: [1024, 64]   (4096/64 = 64 groups)
 *   down scales:    [4096, 16]   (1024/64 = 16 groups)
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Function constants for compile-time specialization
// ============================================================================
// Metal function constants allow the compiler to eliminate dead branches at
// pipeline creation time. USE_FP8_KV controls the KV read path in the fused
// attention kernel — the compiler removes the unused float32 or FP8 code path.

constant bool USE_FP8_KV [[function_constant(0)]];

// ============================================================================
// BFloat16 helpers
// ============================================================================

inline float bf16_to_f32(uint16_t bf16) {
    return as_type<float>(uint(bf16) << 16);
}

inline uint16_t f32_to_bf16(float f) {
    return uint16_t(as_type<uint>(f) >> 16);
}


// ============================================================================
// Kernel 1: 4-bit dequantized matrix-vector multiply (NAIVE — reference)
// ============================================================================

kernel void dequant_matvec_4bit(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;

    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;

    float acc = 0.0f;

    device const uint32_t* w_row = W_packed + tid * packed_cols;
    device const uint16_t* s_row = scales + tid * num_groups;
    device const uint16_t* b_row = biases + tid * num_groups;

    for (uint g = 0; g < num_groups; g++) {
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint base_packed = g * packed_per_group;
        uint base_x = g * group_size;

        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t packed = w_row[base_packed + p];
            uint x_base = base_x + p * 8;

            for (uint n = 0; n < 8; n++) {
                uint nibble = (packed >> (n * 4)) & 0xF;
                float w_val = float(nibble) * scale + bias;
                acc += w_val * x[x_base + n];
            }
        }
    }

    out[tid] = acc;
}


// ============================================================================
// Kernel 1b: 4-bit dequant matvec — SIMD-optimized (legacy, kept for compat)
// ============================================================================

kernel void dequant_matvec_4bit_fast(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tgid >= out_dim) return;

    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;

    device const uint32_t* w_row = W_packed + tgid * packed_cols;
    device const uint16_t* s_row = scales + tgid * num_groups;
    device const uint16_t* b_row = biases + tgid * num_groups;

    float acc = 0.0f;
    for (uint g = lid; g < num_groups; g += tg_size) {
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint base_packed = g * packed_per_group;
        uint base_x = g * group_size;

        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t packed = w_row[base_packed + p];
            uint x_base = base_x + p * 8;

            acc += (float((packed >>  0) & 0xF) * scale + bias) * x[x_base + 0];
            acc += (float((packed >>  4) & 0xF) * scale + bias) * x[x_base + 1];
            acc += (float((packed >>  8) & 0xF) * scale + bias) * x[x_base + 2];
            acc += (float((packed >> 12) & 0xF) * scale + bias) * x[x_base + 3];
            acc += (float((packed >> 16) & 0xF) * scale + bias) * x[x_base + 4];
            acc += (float((packed >> 20) & 0xF) * scale + bias) * x[x_base + 5];
            acc += (float((packed >> 24) & 0xF) * scale + bias) * x[x_base + 6];
            acc += (float((packed >> 28) & 0xF) * scale + bias) * x[x_base + 7];
        }
    }

    threadgroup float shared[32];
    float simd_val = simd_sum(acc);

    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;
    uint num_simd_groups = (tg_size + 31) / 32;

    if (simd_lane == 0) {
        shared[simd_group] = simd_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0 && simd_lane < num_simd_groups) {
        float val = shared[simd_lane];
        val = simd_sum(val);
        if (simd_lane == 0) {
            out[tgid] = val;
        }
    }
}

// ============================================================================
// Fused gate+up+SwiGLU: reads x ONCE, computes silu(gate(x)) * up(x)
// Saves one input read + one kernel dispatch per expert
// ============================================================================
kernel void fused_gate_up_swiglu(
    device const uint32_t* gate_W    [[buffer(0)]],
    device const uint16_t* gate_s    [[buffer(1)]],
    device const uint16_t* gate_b    [[buffer(2)]],
    device const uint32_t* up_W      [[buffer(3)]],
    device const uint16_t* up_s      [[buffer(4)]],
    device const uint16_t* up_b      [[buffer(5)]],
    device const float*    x         [[buffer(6)]],
    device float*          out       [[buffer(7)]],
    constant uint&         out_dim   [[buffer(8)]],
    constant uint&         in_dim    [[buffer(9)]],
    constant uint&         group_size [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_size  [[threads_per_simdgroup]]
) {
    if (tgid >= out_dim) return;
    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;
    device const uint32_t* gr = gate_W + tgid * packed_cols;
    device const uint16_t* gs = gate_s + tgid * num_groups;
    device const uint16_t* gb = gate_b + tgid * num_groups;
    device const uint32_t* ur = up_W   + tgid * packed_cols;
    device const uint16_t* us = up_s   + tgid * num_groups;
    device const uint16_t* ub = up_b   + tgid * num_groups;
    float ga = 0.0f, ua = 0.0f;
    for (uint g = lid; g < num_groups; g += tg_size) {
        float gsc = bf16_to_f32(gs[g]), gbi = bf16_to_f32(gb[g]);
        float usc = bf16_to_f32(us[g]), ubi = bf16_to_f32(ub[g]);
        uint bp = g * packed_per_group, bx = g * group_size;
        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t gp = gr[bp+p], up = ur[bp+p];
            for (uint i = 0; i < 8; i++) {
                float xv = x[bx + p*8 + i];
                ga += (float((gp>>(i*4))&0xF)*gsc+gbi)*xv;
                ua += (float((up>>(i*4))&0xF)*usc+ubi)*xv;
            }
        }
    }
    // Reduction using dynamic SIMD width (future-proof for non-32 SIMD sizes)
    uint num_simd_groups = tg_size / simd_size;
    threadgroup float sg[32], su[32];  // max 256/simd_size groups
    float rg = simd_sum(ga), ru = simd_sum(ua);
    if (simd_lane == 0) { sg[simd_group] = rg; su[simd_group] = ru; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0 && simd_lane < num_simd_groups) {
        float vg = simd_sum(sg[simd_lane]), vu = simd_sum(su[simd_lane]);
        if (simd_lane == 0) out[tgid] = (vg / (1.0f + exp(-vg))) * vu;
    }
}

// ============================================================================
// Kernel 1c: FULLY OPTIMIZED 4-bit dequant matvec
// ============================================================================
//
// Design for M3 Max (40-core GPU, SIMD width 32):
//
// Strategy: Each threadgroup handles ROWS_PER_TG output rows.
//   - Threadgroup size = 256 (8 SIMD groups of 32)
//   - Each SIMD group handles one output row
//   - Within a SIMD group, 32 threads split the input dimension
//   - Each thread processes in_dim/32 input elements using vector loads
//   - Reduction via simd_sum (single instruction)
//
// Memory optimizations:
//   - Input vector x cached in threadgroup shared memory (loaded once)
//   - uint4 vector loads for weights (128 bits = 32 nibbles per load)
//   - float4 vector loads for x (128 bits = 4 floats per load)
//   - Coalesced weight reads: adjacent threads read adjacent uint4 vectors
//
// For gate/up_proj [1024, 4096]: 1024/8 = 128 threadgroups, 256 threads each
//   - 128 * 256 = 32768 threads across 40 cores = good occupancy
//   - Each thread processes 4096/32 = 128 input elements = 16 uint32 packed words
//     = 4 uint4 loads per thread per row
//
// For down_proj [4096, 1024]: 4096/8 = 512 threadgroups
//   - Each thread processes 1024/32 = 32 input elements = 4 uint32 packed words
//     = 1 uint4 load per thread per row

// Number of output rows per threadgroup = number of SIMD groups.
// On current Apple Silicon SIMD width is always 32, so 256/32 = 8.
// Using [[threads_per_simdgroup]] for future-proofing.
#define ROWS_PER_TG 8  // compile-time default for array sizing

kernel void dequant_matvec_4bit_v3(
    device const uint32_t* W_packed   [[buffer(0)]],  // [out_dim, in_dim/8]
    device const uint16_t* scales     [[buffer(1)]],  // [out_dim, num_groups] bf16
    device const uint16_t* biases     [[buffer(2)]],  // [out_dim, num_groups] bf16
    device const float*    x          [[buffer(3)]],  // [in_dim]
    device float*          out        [[buffer(4)]],  // [out_dim]
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid   [[threadgroup_position_in_grid]],     // which tile of rows
    uint lid    [[thread_position_in_threadgroup]],    // 0..255
    uint simd_lane  [[thread_index_in_simdgroup]],    // 0..31
    uint simd_group [[simdgroup_index_in_threadgroup]], // 0..7
    uint simd_size  [[threads_per_simdgroup]]          // 32 on Apple Silicon
) {
    // Which output row this SIMD group handles (dynamic SIMD width)
    uint rows_per_tg = 256 / simd_size;
    uint row = tgid * rows_per_tg + simd_group;

    uint packed_cols = in_dim / 8;      // uint32 columns per row
    uint num_groups  = in_dim / group_size;

    // ---- Cache input vector in threadgroup shared memory ----
    // Max in_dim = 4096, so we need 4096 floats = 16KB shared memory
    // This is well within the 32KB threadgroup memory limit on M3
    threadgroup half x_shared[4096];

    // Cooperative load: 256 threads load 4096 values (16 per thread)
    // Store as half to halve shared memory (8KB vs 16KB) for better occupancy
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = half(x[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Now safe to bail out for out-of-bounds rows
    if (row >= out_dim) return;

    // ---- Pointer setup for this row ----
    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    // ---- Each lane processes a strided slice of the packed columns ----
    // Lane k processes columns: k, k+32, k+64, ...
    // This gives coalesced reads: adjacent lanes read adjacent uint32 words.

    float acc = 0.0f;

    // Process packed columns in strides of 32 (one per SIMD lane)
    for (uint col = simd_lane; col < packed_cols; col += 32) {
        // Determine which group this column belongs to
        // packed_per_group = group_size / 8 = 64 / 8 = 8
        uint g = col / (group_size / 8);
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint32_t packed = w_row[col];
        uint x_base = col * 8;

        // Dequantize 8 nibbles and multiply with cached x
        // Rearranged: (nibble * scale + bias) * x = nibble * (scale*x) + bias*x
        // Pre-compute scale*x and bias*x, then use FMA for dequant+multiply in one op.
        // This reduces per-nibble from (convert + mul + add + mul + add) to (convert + FMA + add).
        float x0 = float(x_shared[x_base + 0]), x1 = float(x_shared[x_base + 1]);
        float x2 = float(x_shared[x_base + 2]), x3 = float(x_shared[x_base + 3]);
        float x4 = float(x_shared[x_base + 4]), x5 = float(x_shared[x_base + 5]);
        float x6 = float(x_shared[x_base + 6]), x7 = float(x_shared[x_base + 7]);
        float sx0 = scale * x0;  float bx0 = bias * x0;
        float sx1 = scale * x1;  float bx1 = bias * x1;
        float sx2 = scale * x2;  float bx2 = bias * x2;
        float sx3 = scale * x3;  float bx3 = bias * x3;
        float sx4 = scale * x4;  float bx4 = bias * x4;
        float sx5 = scale * x5;  float bx5 = bias * x5;
        float sx6 = scale * x6;  float bx6 = bias * x6;
        float sx7 = scale * x7;  float bx7 = bias * x7;

        acc += fma(float((packed >>  0) & 0xF), sx0, bx0);
        acc += fma(float((packed >>  4) & 0xF), sx1, bx1);
        acc += fma(float((packed >>  8) & 0xF), sx2, bx2);
        acc += fma(float((packed >> 12) & 0xF), sx3, bx3);
        acc += fma(float((packed >> 16) & 0xF), sx4, bx4);
        acc += fma(float((packed >> 20) & 0xF), sx5, bx5);
        acc += fma(float((packed >> 24) & 0xF), sx6, bx6);
        acc += fma(float((packed >> 28) & 0xF), sx7, bx7);
    }

    // ---- SIMD reduction: sum across 32 lanes ----
    float sum = simd_sum(acc);

    // Lane 0 writes the result
    if (simd_lane == 0) {
        out[row] = sum;
    }
}


// ============================================================================
// Kernel 1f: 4-bit dequant matvec with LUT (eliminates uint→float conversions)
// ============================================================================
// Instead of converting each nibble to float (expensive conversion instruction),
// pre-compute a 16-entry LUT per group: lut[v] = float(v) * scale + bias.
// Then inner loop is just: acc += lut[nibble] * x_shared[i] — pure math, no conversions.
// The LUT is recomputed every group_size/8 iterations (amortized).

#define ROWS_PER_TG_V5 8

kernel void dequant_matvec_4bit_v5(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG_V5 + simd_group;
    uint packed_cols = in_dim / 8;
    uint num_groups  = in_dim / group_size;
    uint packed_per_group = group_size / 8;

    threadgroup float x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    float acc = 0.0f;
    uint prev_g = 0xFFFFFFFF;
    float lut[16];

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / packed_per_group;

        // Rebuild LUT when group changes
        if (g != prev_g) {
            float scale = bf16_to_f32(s_row[g]);
            float bias  = bf16_to_f32(b_row[g]);
            for (uint v = 0; v < 16; v++) {
                lut[v] = float(v) * scale + bias;
            }
            prev_g = g;
        }

        uint32_t packed = w_row[col];
        uint x_base = col * 8;

        acc += lut[(packed >>  0) & 0xF] * x_shared[x_base + 0];
        acc += lut[(packed >>  4) & 0xF] * x_shared[x_base + 1];
        acc += lut[(packed >>  8) & 0xF] * x_shared[x_base + 2];
        acc += lut[(packed >> 12) & 0xF] * x_shared[x_base + 3];
        acc += lut[(packed >> 16) & 0xF] * x_shared[x_base + 4];
        acc += lut[(packed >> 20) & 0xF] * x_shared[x_base + 5];
        acc += lut[(packed >> 24) & 0xF] * x_shared[x_base + 6];
        acc += lut[(packed >> 28) & 0xF] * x_shared[x_base + 7];
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[row] = sum;
    }
}

// ============================================================================
// Kernel 1e: 2-bit affine dequant matvec (same structure as v3)
// ============================================================================
// Packs 16 x 2-bit values per uint32. Each value is 0-3, dequantized as:
//   val = uint2 * scale + bias (same affine quantization, just 2-bit range)
// Same group structure: group_size elements share one (scale, bias) pair.
// packed_cols = in_dim / 16 (16 values per uint32, vs 8 for 4-bit)

kernel void dequant_matvec_2bit(
    device const uint32_t* W_packed   [[buffer(0)]],  // [out_dim, in_dim/16]
    device const uint16_t* scales     [[buffer(1)]],  // [out_dim, num_groups] bf16
    device const uint16_t* biases     [[buffer(2)]],  // [out_dim, num_groups] bf16
    device const float*    x          [[buffer(3)]],  // [in_dim]
    device float*          out        [[buffer(4)]],  // [out_dim]
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint lid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_size  [[threads_per_simdgroup]]
) {
    uint row = tgid * (256 / simd_size) + simd_group;
    uint packed_cols = in_dim / 16;  // 16 values per uint32 for 2-bit
    uint num_groups  = in_dim / group_size;

    threadgroup half x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = half(x[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= out_dim) return;

    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    float acc = 0.0f;

    // Each lane processes strided columns (16 values per uint32)
    // FMA optimization with half-precision x_shared
    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / (group_size / 16);
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint32_t packed = w_row[col];
        uint x_base = col * 16;

        // Load x values from half shared mem and pre-compute scale*x, bias*x for FMA
        float xv0  = float(x_shared[x_base +  0]), xv1  = float(x_shared[x_base +  1]);
        float xv2  = float(x_shared[x_base +  2]), xv3  = float(x_shared[x_base +  3]);
        float xv4  = float(x_shared[x_base +  4]), xv5  = float(x_shared[x_base +  5]);
        float xv6  = float(x_shared[x_base +  6]), xv7  = float(x_shared[x_base +  7]);
        float xv8  = float(x_shared[x_base +  8]), xv9  = float(x_shared[x_base +  9]);
        float xv10 = float(x_shared[x_base + 10]), xv11 = float(x_shared[x_base + 11]);
        float xv12 = float(x_shared[x_base + 12]), xv13 = float(x_shared[x_base + 13]);
        float xv14 = float(x_shared[x_base + 14]), xv15 = float(x_shared[x_base + 15]);
        float sx0  = scale * xv0;   float bx0  = bias * xv0;
        float sx1  = scale * xv1;   float bx1  = bias * xv1;
        float sx2  = scale * xv2;   float bx2  = bias * xv2;
        float sx3  = scale * xv3;   float bx3  = bias * xv3;
        float sx4  = scale * xv4;   float bx4  = bias * xv4;
        float sx5  = scale * xv5;   float bx5  = bias * xv5;
        float sx6  = scale * xv6;   float bx6  = bias * xv6;
        float sx7  = scale * xv7;   float bx7  = bias * xv7;
        float sx8  = scale * xv8;   float bx8  = bias * xv8;
        float sx9  = scale * xv9;   float bx9  = bias * xv9;
        float sx10 = scale * xv10;  float bx10 = bias * xv10;
        float sx11 = scale * xv11;  float bx11 = bias * xv11;
        float sx12 = scale * xv12;  float bx12 = bias * xv12;
        float sx13 = scale * xv13;  float bx13 = bias * xv13;
        float sx14 = scale * xv14;  float bx14 = bias * xv14;
        float sx15 = scale * xv15;  float bx15 = bias * xv15;

        acc += fma(float((packed >>  0) & 0x3), sx0,  bx0);
        acc += fma(float((packed >>  2) & 0x3), sx1,  bx1);
        acc += fma(float((packed >>  4) & 0x3), sx2,  bx2);
        acc += fma(float((packed >>  6) & 0x3), sx3,  bx3);
        acc += fma(float((packed >>  8) & 0x3), sx4,  bx4);
        acc += fma(float((packed >> 10) & 0x3), sx5,  bx5);
        acc += fma(float((packed >> 12) & 0x3), sx6,  bx6);
        acc += fma(float((packed >> 14) & 0x3), sx7,  bx7);
        acc += fma(float((packed >> 16) & 0x3), sx8,  bx8);
        acc += fma(float((packed >> 18) & 0x3), sx9,  bx9);
        acc += fma(float((packed >> 20) & 0x3), sx10, bx10);
        acc += fma(float((packed >> 22) & 0x3), sx11, bx11);
        acc += fma(float((packed >> 24) & 0x3), sx12, bx12);
        acc += fma(float((packed >> 26) & 0x3), sx13, bx13);
        acc += fma(float((packed >> 28) & 0x3), sx14, bx14);
        acc += fma(float((packed >> 30) & 0x3), sx15, bx15);
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[row] = sum;
    }
}


// ============================================================================
// Kernel 1d: FULLY OPTIMIZED with uint4 vector loads
// ============================================================================
//
// Same structure as v3 but uses uint4 loads (128-bit / 16 bytes) to maximize
// memory bandwidth per thread. Each uint4 = 4 uint32 = 32 nibbles.
//
// For gate/up (packed_cols=512): each thread processes 512/32 = 16 uint32
//   = 4 uint4 loads per thread
// For down (packed_cols=128): each thread processes 128/32 = 4 uint32
//   = 1 uint4 load per thread

kernel void dequant_matvec_4bit_v4(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_size  [[threads_per_simdgroup]]
) {
    uint row = tgid * (256 / simd_size) + simd_group;

    uint packed_cols = in_dim / 8;
    uint num_groups  = in_dim / group_size;

    // Cache input vector — ALL threads must participate before the barrier
    threadgroup float x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    // Pointers — cast to uint4 for vector loads
    device const uint4* w_row_v = (device const uint4*)(W_packed + row * packed_cols);
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    uint vec4_cols = packed_cols / 4;  // number of uint4 vectors per row

    float acc = 0.0f;

    // Each lane processes vec4_cols / 32 vectors (coalesced: adjacent lanes read adjacent uint4)
    for (uint vi = simd_lane; vi < vec4_cols; vi += 32) {
        uint4 packed4 = w_row_v[vi];

        // Each uint4 covers 4 * 8 = 32 input elements
        // Starting packed column index = vi * 4
        uint base_col = vi * 4;
        uint x_base = base_col * 8;  // starting input element

        // Process each of the 4 uint32 words in the uint4
        // Unroll all 4 words x 8 nibbles = 32 multiply-adds
        #pragma unroll
        for (uint w = 0; w < 4; w++) {
            uint32_t packed = packed4[w];
            uint col = base_col + w;
            uint g = col / (group_size / 8);
            float scale = bf16_to_f32(s_row[g]);
            float bias  = bf16_to_f32(b_row[g]);

            uint xb = x_base + w * 8;
            acc += (float((packed >>  0) & 0xF) * scale + bias) * x_shared[xb + 0];
            acc += (float((packed >>  4) & 0xF) * scale + bias) * x_shared[xb + 1];
            acc += (float((packed >>  8) & 0xF) * scale + bias) * x_shared[xb + 2];
            acc += (float((packed >> 12) & 0xF) * scale + bias) * x_shared[xb + 3];
            acc += (float((packed >> 16) & 0xF) * scale + bias) * x_shared[xb + 4];
            acc += (float((packed >> 20) & 0xF) * scale + bias) * x_shared[xb + 5];
            acc += (float((packed >> 24) & 0xF) * scale + bias) * x_shared[xb + 6];
            acc += (float((packed >> 28) & 0xF) * scale + bias) * x_shared[xb + 7];
        }
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[row] = sum;
    }
}


// ============================================================================
// Kernel 1e: Multi-expert batched matvec
// ============================================================================
//
// Dispatch multiple experts simultaneously. The grid's Y dimension indexes
// the expert, so K experts' matmuls run as parallel threadgroups.
//
// Buffer layout: W_packed, scales, biases are arrays of K experts concatenated.
// x_inputs:  K input vectors concatenated [K * in_dim]
// out:       K output vectors concatenated [K * out_dim]
// expert_offsets: byte offset into W_packed buffer for each expert's weights
//                 (allows non-contiguous expert data in a shared buffer)

kernel void dequant_matvec_4bit_batched(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x_inputs   [[buffer(3)]],  // [K, in_dim]
    device float*          out        [[buffer(4)]],  // [K, out_dim]
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    // Per-expert offsets into the weight/scale/bias buffers (in elements)
    device const uint*     w_offsets  [[buffer(8)]],  // [K] offset in uint32 elements
    device const uint*     s_offsets  [[buffer(9)]],  // [K] offset in uint16 elements
    device const uint*     b_offsets  [[buffer(10)]], // [K] offset in uint16 elements
    constant uint&         num_row_tiles [[buffer(11)]], // ceil(out_dim / ROWS_PER_TG)
    uint tgid_flat [[threadgroup_position_in_grid]],  // linearized (row_tile + expert * num_row_tiles)
    uint lid       [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_size  [[threads_per_simdgroup]]
) {
    // De-linearize: tgid_flat = row_tile + expert_k * num_row_tiles
    uint expert_k = tgid_flat / num_row_tiles;
    uint row_tile = tgid_flat % num_row_tiles;
    uint row = row_tile * (256 / simd_size) + simd_group;
    if (row >= out_dim) return;

    uint packed_cols = in_dim / 8;
    uint num_groups  = in_dim / group_size;

    // Cache this expert's input vector
    threadgroup float x_shared[4096];
    device const float* x_k = x_inputs + expert_k * in_dim;
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x_k[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Point to this expert's weights
    device const uint32_t* w_row = W_packed + w_offsets[expert_k] + row * packed_cols;
    device const uint16_t* s_row = scales   + s_offsets[expert_k] + row * num_groups;
    device const uint16_t* b_row = biases   + b_offsets[expert_k] + row * num_groups;

    float acc = 0.0f;

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / (group_size / 8);
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint32_t packed = w_row[col];
        uint x_base = col * 8;

        acc += (float((packed >>  0) & 0xF) * scale + bias) * x_shared[x_base + 0];
        acc += (float((packed >>  4) & 0xF) * scale + bias) * x_shared[x_base + 1];
        acc += (float((packed >>  8) & 0xF) * scale + bias) * x_shared[x_base + 2];
        acc += (float((packed >> 12) & 0xF) * scale + bias) * x_shared[x_base + 3];
        acc += (float((packed >> 16) & 0xF) * scale + bias) * x_shared[x_base + 4];
        acc += (float((packed >> 20) & 0xF) * scale + bias) * x_shared[x_base + 5];
        acc += (float((packed >> 24) & 0xF) * scale + bias) * x_shared[x_base + 6];
        acc += (float((packed >> 28) & 0xF) * scale + bias) * x_shared[x_base + 7];
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[expert_k * out_dim + row] = sum;
    }
}


// ============================================================================
// Kernel 2: SwiGLU activation
// ============================================================================

kernel void swiglu_fused(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device float*       out  [[buffer(2)]],
    constant uint&      dim  [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    float g = gate[tid];
    float silu_g = g / (1.0f + exp(-g));
    out[tid] = silu_g * up[tid];
}

// Vectorized SwiGLU: process 4 elements per thread
kernel void swiglu_fused_vec4(
    device const float4* gate [[buffer(0)]],
    device const float4* up   [[buffer(1)]],
    device float4*       out  [[buffer(2)]],
    constant uint&       dim  [[buffer(3)]],  // original dim (must be multiple of 4)
    uint tid [[thread_position_in_grid]]
) {
    uint vec_dim = dim / 4;
    if (tid >= vec_dim) return;

    float4 g = gate[tid];
    float4 silu_g = g / (1.0f + exp(-g));
    out[tid] = silu_g * up[tid];
}


// ============================================================================
// Kernel 2b: Batched SwiGLU for K experts
// ============================================================================

kernel void swiglu_fused_batched(
    device const float* gate [[buffer(0)]],  // [K * dim]
    device const float* up   [[buffer(1)]],  // [K * dim]
    device float*       out  [[buffer(2)]],  // [K * dim]
    constant uint&      dim  [[buffer(3)]],
    constant uint&      K    [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = K * dim;
    if (tid >= total) return;

    float g = gate[tid];
    float silu_g = g / (1.0f + exp(-g));
    out[tid] = silu_g * up[tid];
}


// ============================================================================
// Kernel 3: Weighted sum of expert outputs
// ============================================================================

kernel void weighted_sum(
    device const float* expert_outs [[buffer(0)]],
    device const float* weights     [[buffer(1)]],
    device float*       out         [[buffer(2)]],
    constant uint&      K           [[buffer(3)]],
    constant uint&      dim         [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    float acc = 0.0f;
    for (uint k = 0; k < K; k++) {
        acc += weights[k] * expert_outs[k * dim + tid];
    }
    out[tid] = acc;
}


// ============================================================================
// Kernel 4: RMS Normalization
// ============================================================================

kernel void rms_norm_sum_sq(
    device const float* x       [[buffer(0)]],
    device float*       sum_sq  [[buffer(1)]],
    constant uint&      dim     [[buffer(2)]],
    uint tid  [[thread_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float shared[32];

    float acc = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float val = x[i];
        acc += val * val;
    }

    float simd_val = simd_sum(acc);
    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;

    if (simd_lane == 0) {
        shared[simd_group] = simd_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        float val = (simd_lane < (tg_size + 31) / 32) ? shared[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            sum_sq[0] = val;
        }
    }
}

kernel void rms_norm_apply(
    device const float* x       [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device const float* sum_sq  [[buffer(2)]],
    device float*       out     [[buffer(3)]],
    constant uint&      dim     [[buffer(4)]],
    constant float&     eps     [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    float rms = rsqrt(sum_sq[0] / float(dim) + eps);
    out[tid] = x[tid] * rms * weight[tid];
}


// ============================================================================
// Kernel 4b: RMS Normalization with bf16 weights
// ============================================================================
// Same as rms_norm_apply but reads weights as bfloat16 (uint16_t) and
// converts to float32 inline. Used in the fused o_proj+norm+routing path
// where norm weights come directly from the mmap'd weight file (bf16).

kernel void rms_norm_apply_bf16(
    device const float*    x       [[buffer(0)]],
    device const uint16_t* weight  [[buffer(1)]],  // bf16 weights
    device const float*    sum_sq  [[buffer(2)]],
    device float*          out     [[buffer(3)]],
    constant uint&         dim     [[buffer(4)]],
    constant float&        eps     [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    float rms = rsqrt(sum_sq[0] / float(dim) + eps);
    float w = bf16_to_f32(weight[tid]);
    out[tid] = x[tid] * rms * w;
}


// ============================================================================
// Kernel 5: Residual add
// ============================================================================
// out[i] = a[i] + b[i]
// Used to fuse the residual connection into a GPU command buffer,
// eliminating a CPU round-trip between o_proj and routing.

kernel void residual_add(
    device const float* a   [[buffer(0)]],
    device const float* b   [[buffer(1)]],
    device float*       out [[buffer(2)]],
    constant uint&      dim [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    out[tid] = a[tid] + b[tid];
}


// ============================================================================
// Kernel 6: Batched GPU attention scores (Q @ K^T, scaled) — all heads at once
// ============================================================================
//
// Computes scores[h, p] = sum_d(Q[h, d] * K[p, kv_h*head_dim + d]) * scale
// for all heads h in [0, num_heads) and positions p in [0, seq_len).
//
// Grid: linearized (pos + h * num_seq_tgs) — one threadgroup per (position, head).
// Each threadgroup of 256 threads reduces over head_dim=256.
//
// GQA mapping: kv_head = h / heads_per_kv (e.g. 16 query heads share 1 KV head)
//
// Output layout: scores[h * seq_stride + p] where seq_stride = MAX_SEQ_LEN

kernel void attn_scores_batched(
    device const float* Q          [[buffer(0)]],  // [num_heads, head_dim]
    device const float* K_cache    [[buffer(1)]],  // [max_seq, kv_dim]
    device float*       scores     [[buffer(2)]],  // [num_heads, seq_stride]
    constant uint&      head_dim   [[buffer(3)]],  // 256
    constant uint&      kv_dim     [[buffer(4)]],  // 512
    constant uint&      seq_len    [[buffer(5)]],  // current seq length
    constant uint&      seq_stride [[buffer(6)]],  // MAX_SEQ_LEN
    constant float&     scale      [[buffer(7)]],  // 1/sqrt(head_dim)
    constant uint&      heads_per_kv [[buffer(8)]], // 16 (GQA ratio)
    constant uint&      num_seq_tgs  [[buffer(9)]],  // = seq_len
    uint tgid  [[threadgroup_position_in_grid]],    // linearized: pos + h * num_seq_tgs
    uint lid   [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint pos = tgid % num_seq_tgs;
    uint h = tgid / num_seq_tgs;
    if (pos >= seq_len) return;

    uint kv_h = h / heads_per_kv;
    device const float* qh = Q + h * head_dim;
    device const float* kp = K_cache + pos * kv_dim + kv_h * head_dim;

    float acc = 0.0f;
    for (uint d = lid; d < head_dim; d += tg_size) {
        acc += qh[d] * kp[d];
    }

    // SIMD reduction
    float simd_val = simd_sum(acc);
    threadgroup float shared[32];
    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;
    uint num_simd_groups = (tg_size + 31) / 32;
    if (simd_lane == 0) shared[simd_group] = simd_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0 && simd_lane < num_simd_groups) {
        float val = simd_sum(shared[simd_lane]);
        if (simd_lane == 0) {
            scores[h * seq_stride + pos] = val * scale;
        }
    }
}


// ============================================================================
// Kernel 7: Batched softmax — one threadgroup per head
// ============================================================================

kernel void attn_softmax_batched(
    device float*    scores     [[buffer(0)]],  // [num_heads, seq_stride]
    constant uint&   seq_len    [[buffer(1)]],
    constant uint&   seq_stride [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],     // head index
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    device float* s = scores + tgid * seq_stride;

    // Pass 1: find max
    threadgroup float shared_max[32];
    float local_max = -1e30f;
    for (uint i = lid; i < seq_len; i += tg_size) {
        local_max = max(local_max, s[i]);
    }
    float sm = simd_max(local_max);
    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;
    uint num_simd_groups = (tg_size + 31) / 32;
    if (simd_lane == 0) shared_max[simd_group] = sm;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_max = -1e30f;
    if (simd_group == 0 && simd_lane < num_simd_groups) {
        global_max = simd_max(shared_max[simd_lane]);
    }
    threadgroup float broadcast_max;
    if (lid == 0) broadcast_max = global_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_max = broadcast_max;

    // Pass 2: exp and sum
    threadgroup float shared_sum[32];
    float local_sum = 0.0f;
    for (uint i = lid; i < seq_len; i += tg_size) {
        float val = exp(s[i] - global_max);
        s[i] = val;
        local_sum += val;
    }
    float simd_s = simd_sum(local_sum);
    if (simd_lane == 0) shared_sum[simd_group] = simd_s;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_sum = 0.0f;
    if (simd_group == 0 && simd_lane < num_simd_groups) {
        global_sum = simd_sum(shared_sum[simd_lane]);
    }
    threadgroup float broadcast_sum;
    if (lid == 0) broadcast_sum = global_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_sum = broadcast_sum;

    // Pass 3: normalize
    float inv_sum = 1.0f / global_sum;
    for (uint i = lid; i < seq_len; i += tg_size) {
        s[i] *= inv_sum;
    }
}


// ============================================================================
// Kernel 8: Batched attention value aggregation (scores @ V) — all heads
// ============================================================================
//
// For each head h: output[h*head_dim + d] = sum_p(scores[h*seq_stride+p] * V[p*kv_dim + kv_h*head_dim + d])
//
// Grid: linearized over (head_dim * num_heads) — one thread per (dimension, head).

kernel void attn_values_batched(
    device const float* scores   [[buffer(0)]],  // [num_heads, seq_stride]
    device const float* V_cache  [[buffer(1)]],  // [max_seq, kv_dim]
    device float*       out      [[buffer(2)]],  // [num_heads, head_dim]
    constant uint&      head_dim  [[buffer(3)]],  // 256
    constant uint&      kv_dim    [[buffer(4)]],  // 512
    constant uint&      seq_len   [[buffer(5)]],
    constant uint&      seq_stride [[buffer(6)]],
    constant uint&      heads_per_kv [[buffer(7)]],
    uint tid [[thread_position_in_grid]]          // linearized: d + h * head_dim
) {
    uint d = tid % head_dim;
    uint h = tid / head_dim;

    uint kv_h = h / heads_per_kv;
    device const float* s = scores + h * seq_stride;

    float acc = 0.0f;
    for (uint p = 0; p < seq_len; p++) {
        acc += s[p] * V_cache[p * kv_dim + kv_h * head_dim + d];
    }
    out[h * head_dim + d] = acc;
}


// ============================================================================
// Kernel 8b: FP8 E4M3 attention scores — K cache is uchar with per-pos scale
// ============================================================================
//
// Same structure as attn_scores_batched, but K_cache is device uchar* and
// each position p has a float scale in scales[p].
// Dequant: k_float = fp8_e4m3_decode(k_byte) * scale[p]
//
// FP8 E4M3 decode (inline device function):

static inline float fp8_e4m3_to_float(uchar x) {
    if (x == 0x7F) return 0.0f;  // NaN -> 0 for safe accumulation
    uchar sign = (x >> 7) & 1;
    uchar exp_biased = (x >> 3) & 0xF;
    uchar mantissa = x & 0x7;
    float val;
    if (exp_biased == 0) {
        val = float(mantissa) * 0.001953125f;  // 2^-9 (subnormal)
    } else {
        val = (1.0f + float(mantissa) / 8.0f) * exp2(float(exp_biased) - 7.0f);
    }
    return sign ? -val : val;
}

kernel void attn_scores_fp8(
    device const float* Q            [[buffer(0)]],   // [num_heads, head_dim]
    device const uchar* K_cache_fp8  [[buffer(1)]],   // [max_seq, kv_dim] uint8
    device const float* K_scales     [[buffer(2)]],   // [max_seq] per-pos scale
    device float*       scores       [[buffer(3)]],   // [num_heads, seq_stride]
    constant uint&      head_dim     [[buffer(4)]],
    constant uint&      kv_dim       [[buffer(5)]],
    constant uint&      seq_len      [[buffer(6)]],
    constant uint&      seq_stride   [[buffer(7)]],
    constant float&     scale        [[buffer(8)]],   // 1/sqrt(head_dim)
    constant uint&      heads_per_kv [[buffer(9)]],
    constant uint&      num_seq_tgs  [[buffer(10)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint pos = tgid % num_seq_tgs;
    uint h = tgid / num_seq_tgs;
    if (pos >= seq_len) return;

    uint kv_h = h / heads_per_kv;
    device const float* qh = Q + h * head_dim;
    device const uchar* kp_fp8 = K_cache_fp8 + pos * kv_dim + kv_h * head_dim;
    float k_scale = K_scales[pos];

    float acc = 0.0f;
    for (uint d = lid; d < head_dim; d += tg_size) {
        float k_val = fp8_e4m3_to_float(kp_fp8[d]) * k_scale;
        acc += qh[d] * k_val;
    }

    float simd_val = simd_sum(acc);
    threadgroup float shared[32];
    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;
    uint num_simd_groups = (tg_size + 31) / 32;
    if (simd_lane == 0) shared[simd_group] = simd_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0 && simd_lane < num_simd_groups) {
        float val = simd_sum(shared[simd_lane]);
        if (simd_lane == 0) {
            scores[h * seq_stride + pos] = val * scale;
        }
    }
}


// ============================================================================
// Kernel 8c: FP8 E4M3 attention values — V cache is uchar with per-pos scale
// ============================================================================

kernel void attn_values_fp8(
    device const float* scores       [[buffer(0)]],   // [num_heads, seq_stride]
    device const uchar* V_cache_fp8  [[buffer(1)]],   // [max_seq, kv_dim] uint8
    device const float* V_scales     [[buffer(2)]],   // [max_seq] per-pos scale
    device float*       out          [[buffer(3)]],   // [num_heads, head_dim]
    constant uint&      head_dim     [[buffer(4)]],
    constant uint&      kv_dim       [[buffer(5)]],
    constant uint&      seq_len      [[buffer(6)]],
    constant uint&      seq_stride   [[buffer(7)]],
    constant uint&      heads_per_kv [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    uint d = tid % head_dim;
    uint h = tid / head_dim;

    uint kv_h = h / heads_per_kv;
    device const float* s = scores + h * seq_stride;

    float acc = 0.0f;
    for (uint p = 0; p < seq_len; p++) {
        float v_val = fp8_e4m3_to_float(V_cache_fp8[p * kv_dim + kv_h * head_dim + d]) * V_scales[p];
        acc += s[p] * v_val;
    }
    out[h * head_dim + d] = acc;
}


// ============================================================================
// Kernel 9: Sigmoid element-wise gate
// ============================================================================
// out[i] = x[i] * sigmoid(gate[i])

kernel void sigmoid_gate(
    device float*       x_out  [[buffer(0)]],  // [dim] in/out
    device const float* gate   [[buffer(1)]],  // [dim] gate values
    constant uint&      dim    [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    float g = 1.0f / (1.0f + exp(-gate[tid]));
    x_out[tid] = x_out[tid] * g;
}


// ============================================================================
// Kernel 9b: Fused online softmax attention — single kernel replaces
// attn_scores + attn_softmax + attn_values for full-attention layers
// ============================================================================
//
// FlashAttention-style online softmax: iterate over KV positions in blocks,
// maintaining running max and sum for numerically stable softmax without
// materializing the full scores matrix.
//
// One threadgroup per query head. 256 threads per threadgroup.
// Supports GQA: multiple query heads share one KV head.
// Supports both float32 and FP8 E4M3 KV caches (controlled by use_fp8 param).
//
// Algorithm per head:
//   m = -inf, l = 0, O = 0
//   For each block of BLOCK_SIZE=32 KV positions:
//     Compute QK dot products via shared memory + SIMD reduction
//     Update online softmax: m_new = max(m, block_max)
//     Rescale: O *= exp(m_old - m_new) * (l_old / l_new)
//     Accumulate: O += softmax_weight * V[pos]
//   Output O (already normalized)

#define FUSED_ATTN_BLOCK_SIZE 64

kernel void fused_attention_online(
    device const float* Q          [[buffer(0)]],   // [num_heads, head_dim]
    device const float* K_cache    [[buffer(1)]],   // [max_seq, kv_dim] float32
    device const float* V_cache    [[buffer(2)]],   // [max_seq, kv_dim] float32
    device float*       out        [[buffer(3)]],   // [num_heads, head_dim]
    constant uint&      head_dim   [[buffer(4)]],   // 128 or 256
    constant uint&      kv_dim     [[buffer(5)]],   // num_kv_heads * head_dim
    constant uint&      seq_len    [[buffer(6)]],   // current sequence length
    constant uint&      num_heads  [[buffer(7)]],   // total query heads
    constant uint&      num_kv_heads [[buffer(8)]], // KV heads (GQA)
    constant float&     scale      [[buffer(9)]],   // 1/sqrt(head_dim)
    uint tgid  [[threadgroup_position_in_grid]],    // head index
    uint lid   [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tgid >= num_heads) return;

    uint heads_per_kv = num_heads / num_kv_heads;
    uint kv_h = tgid / heads_per_kv;

    // Pointer to this head's query vector
    device const float* qh = Q + tgid * head_dim;

    // Online softmax state (FlashAttention-2 style: unnormalized accumulator)
    // m: running max of QK scores
    // l: running sum of exp(score - m)
    // o: unnormalized weighted V accumulator (divided by l at the end)
    float m_prev = -1e30f;
    float l_prev = 0.0f;

    // Each thread accumulates its slice of the output vector
    // Thread lid handles dimensions lid, lid+tg_size, lid+2*tg_size, ...
    float o_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // max 4 dims per thread (head_dim<=1024)

    // Shared memory for QK dot product reduction
    threadgroup float shared_qk[FUSED_ATTN_BLOCK_SIZE];
    // Shared memory for SIMD reduction of dot products
    threadgroup float simd_scratch[32];

    // Process KV positions in blocks
    for (uint block_start = 0; block_start < seq_len; block_start += FUSED_ATTN_BLOCK_SIZE) {
        uint block_end = min(block_start + FUSED_ATTN_BLOCK_SIZE, seq_len);
        uint block_len = block_end - block_start;

        // Phase 1: Compute QK dot products for this block
        float block_scores[FUSED_ATTN_BLOCK_SIZE];
        float block_max = -1e30f;

        for (uint b = 0; b < block_len; b++) {
            uint pos = block_start + b;
            device const float* kp = K_cache + pos * kv_dim + kv_h * head_dim;

            // Parallel dot product: each thread handles a stride of head_dim
            float dot = 0.0f;
            for (uint d = lid; d < head_dim; d += tg_size) {
                dot += qh[d] * kp[d];
            }

            // SIMD reduction within each simd_group
            float simd_val = simd_sum(dot);
            uint simd_lane = lid % 32;
            uint simd_group = lid / 32;
            uint num_simd_groups = (tg_size + 31) / 32;
            if (simd_lane == 0) simd_scratch[simd_group] = simd_val;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Final reduction across simd_groups
            float total = 0.0f;
            if (lid < num_simd_groups) {
                total = simd_sum(simd_scratch[lid]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lid == 0) {
                shared_qk[b] = total * scale;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float s = shared_qk[b];
            block_scores[b] = s;
            block_max = max(block_max, s);
        }

        // Phase 2: Online softmax update (FlashAttention-2 algorithm)
        float m_new = max(m_prev, block_max);
        float correction = exp(m_prev - m_new);

        // Compute softmax weights for this block (unnormalized)
        float block_sum = 0.0f;
        float block_weights[FUSED_ATTN_BLOCK_SIZE];
        for (uint b = 0; b < block_len; b++) {
            float w = exp(block_scores[b] - m_new);
            block_weights[b] = w;
            block_sum += w;
        }

        // Update running sum: l = l_prev * correction + block_sum
        float l_new = l_prev * correction + block_sum;

        // Phase 3: Rescale previous accumulator and add V contributions
        for (uint dim_idx = 0; dim_idx < 4; dim_idx++) {
            uint d = lid + dim_idx * tg_size;
            if (d >= head_dim) break;

            // Rescale previous unnormalized accumulator by correction factor
            float o_val = o_acc[dim_idx] * correction;

            // Add new V contributions (unnormalized — will divide by l at the end)
            for (uint b = 0; b < block_len; b++) {
                uint pos = block_start + b;
                float v_val = V_cache[pos * kv_dim + kv_h * head_dim + d];
                o_val += block_weights[b] * v_val;
            }
            o_acc[dim_idx] = o_val;
        }

        m_prev = m_new;
        l_prev = l_new;
    }

    // Write output — normalize by dividing by the total softmax sum
    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    for (uint dim_idx = 0; dim_idx < 4; dim_idx++) {
        uint d = lid + dim_idx * tg_size;
        if (d >= head_dim) break;
        out[tgid * head_dim + d] = o_acc[dim_idx] * inv_l;
    }
}


// ============================================================================
// Kernel 9c: Fused online softmax attention — FP8 E4M3 KV cache variant
// ============================================================================

kernel void fused_attention_online_fp8(
    device const float* Q              [[buffer(0)]],   // [num_heads, head_dim]
    device const uchar* K_cache_fp8    [[buffer(1)]],   // [max_seq, kv_dim] uint8
    device const float* K_scales       [[buffer(2)]],   // [max_seq] per-pos scale
    device const uchar* V_cache_fp8    [[buffer(3)]],   // [max_seq, kv_dim] uint8
    device const float* V_scales       [[buffer(4)]],   // [max_seq] per-pos scale
    device float*       out            [[buffer(5)]],   // [num_heads, head_dim]
    constant uint&      head_dim       [[buffer(6)]],
    constant uint&      kv_dim         [[buffer(7)]],
    constant uint&      seq_len        [[buffer(8)]],
    constant uint&      num_heads      [[buffer(9)]],
    constant uint&      num_kv_heads   [[buffer(10)]],
    constant float&     scale          [[buffer(11)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tgid >= num_heads) return;

    uint heads_per_kv = num_heads / num_kv_heads;
    uint kv_h = tgid / heads_per_kv;

    device const float* qh = Q + tgid * head_dim;

    float m_prev = -1e30f;
    float l_prev = 0.0f;
    float o_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    threadgroup float shared_qk[FUSED_ATTN_BLOCK_SIZE];
    threadgroup float simd_scratch[32];

    for (uint block_start = 0; block_start < seq_len; block_start += FUSED_ATTN_BLOCK_SIZE) {
        uint block_end = min(block_start + FUSED_ATTN_BLOCK_SIZE, seq_len);
        uint block_len = block_end - block_start;

        float block_scores[FUSED_ATTN_BLOCK_SIZE];
        float block_max = -1e30f;

        for (uint b = 0; b < block_len; b++) {
            uint pos = block_start + b;
            device const uchar* kp_fp8 = K_cache_fp8 + pos * kv_dim + kv_h * head_dim;
            float k_scale = K_scales[pos];

            float dot = 0.0f;
            for (uint d = lid; d < head_dim; d += tg_size) {
                float k_val = fp8_e4m3_to_float(kp_fp8[d]) * k_scale;
                dot += qh[d] * k_val;
            }

            float simd_val = simd_sum(dot);
            uint simd_lane = lid % 32;
            uint simd_group = lid / 32;
            uint num_simd_groups = (tg_size + 31) / 32;
            if (simd_lane == 0) simd_scratch[simd_group] = simd_val;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float total = 0.0f;
            if (lid < num_simd_groups) {
                total = simd_sum(simd_scratch[lid]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lid == 0) {
                shared_qk[b] = total * scale;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float s = shared_qk[b];
            block_scores[b] = s;
            block_max = max(block_max, s);
        }

        float m_new = max(m_prev, block_max);
        float correction = exp(m_prev - m_new);

        float block_sum = 0.0f;
        float block_weights[FUSED_ATTN_BLOCK_SIZE];
        for (uint b = 0; b < block_len; b++) {
            float w = exp(block_scores[b] - m_new);
            block_weights[b] = w;
            block_sum += w;
        }

        float l_new = l_prev * correction + block_sum;

        // Rescale previous accumulator and add V contributions (unnormalized)
        for (uint dim_idx = 0; dim_idx < 4; dim_idx++) {
            uint d = lid + dim_idx * tg_size;
            if (d >= head_dim) break;

            float o_val = o_acc[dim_idx] * correction;

            for (uint b = 0; b < block_len; b++) {
                uint pos = block_start + b;
                float v_val = fp8_e4m3_to_float(V_cache_fp8[pos * kv_dim + kv_h * head_dim + d]) * V_scales[pos];
                o_val += block_weights[b] * v_val;
            }
            o_acc[dim_idx] = o_val;
        }

        m_prev = m_new;
        l_prev = l_new;
    }

    // Final normalization: divide by total softmax sum
    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    for (uint dim_idx = 0; dim_idx < 4; dim_idx++) {
        uint d = lid + dim_idx * tg_size;
        if (d >= head_dim) break;
        out[tgid * head_dim + d] = o_acc[dim_idx] * inv_l;
    }
}


// ============================================================================
// Kernel 9d: Fused online softmax attention — function-constant specialized
// ============================================================================
//
// Single kernel that handles both float32 and FP8 KV caches. The Metal compiler
// uses function constants (USE_FP8_KV, FC_HEAD_DIM) to eliminate dead branches
// at pipeline creation time, producing code equivalent to the separate kernels
// but from a single source.
//
// Buffer layout unifies both paths:
//   buffer(1): K cache — float* when !USE_FP8_KV, uchar* when USE_FP8_KV
//   buffer(2): K scales (FP8 only, ignored otherwise)
//   buffer(3): V cache — float* when !USE_FP8_KV, uchar* when USE_FP8_KV
//   buffer(4): V scales (FP8 only, ignored otherwise)

kernel void fused_attention_online_fc(
    device const float* Q              [[buffer(0)]],
    device const void*  K_cache_raw    [[buffer(1)]],   // float* or uchar* depending on USE_FP8_KV
    device const float* K_scales       [[buffer(2)]],   // per-pos scales (FP8 only)
    device const void*  V_cache_raw    [[buffer(3)]],   // float* or uchar* depending on USE_FP8_KV
    device const float* V_scales       [[buffer(4)]],   // per-pos scales (FP8 only)
    device float*       out            [[buffer(5)]],
    constant uint&      head_dim       [[buffer(6)]],
    constant uint&      kv_dim         [[buffer(7)]],
    constant uint&      seq_len        [[buffer(8)]],
    constant uint&      num_heads      [[buffer(9)]],
    constant uint&      num_kv_heads   [[buffer(10)]],
    constant float&     scale          [[buffer(11)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tgid >= num_heads) return;

    uint heads_per_kv = num_heads / num_kv_heads;
    uint kv_h = tgid / heads_per_kv;

    device const float* qh = Q + tgid * head_dim;

    float m_prev = -1e30f;
    float l_prev = 0.0f;
    float o_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    threadgroup float shared_qk[FUSED_ATTN_BLOCK_SIZE];
    threadgroup float simd_scratch[32];

    for (uint block_start = 0; block_start < seq_len; block_start += FUSED_ATTN_BLOCK_SIZE) {
        uint block_end = min(block_start + FUSED_ATTN_BLOCK_SIZE, seq_len);
        uint block_len = block_end - block_start;

        float block_scores[FUSED_ATTN_BLOCK_SIZE];
        float block_max = -1e30f;

        for (uint b = 0; b < block_len; b++) {
            uint pos = block_start + b;
            float dot = 0.0f;

            // Function constant: compiler eliminates the dead branch
            if (USE_FP8_KV) {
                device const uchar* kp_fp8 = (device const uchar*)K_cache_raw + pos * kv_dim + kv_h * head_dim;
                float k_scale = K_scales[pos];
                for (uint d = lid; d < head_dim; d += tg_size) {
                    float k_val = fp8_e4m3_to_float(kp_fp8[d]) * k_scale;
                    dot += qh[d] * k_val;
                }
            } else {
                device const float* kp = (device const float*)K_cache_raw + pos * kv_dim + kv_h * head_dim;
                for (uint d = lid; d < head_dim; d += tg_size) {
                    dot += qh[d] * kp[d];
                }
            }

            float simd_val = simd_sum(dot);
            uint simd_lane = lid % 32;
            uint simd_group = lid / 32;
            uint num_simd_groups = (tg_size + 31) / 32;
            if (simd_lane == 0) simd_scratch[simd_group] = simd_val;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float total = 0.0f;
            if (lid < num_simd_groups) {
                total = simd_sum(simd_scratch[lid]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lid == 0) {
                shared_qk[b] = total * scale;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float s = shared_qk[b];
            block_scores[b] = s;
            block_max = max(block_max, s);
        }

        float m_new = max(m_prev, block_max);
        float correction = exp(m_prev - m_new);

        float block_sum = 0.0f;
        float block_weights[FUSED_ATTN_BLOCK_SIZE];
        for (uint b = 0; b < block_len; b++) {
            float w = exp(block_scores[b] - m_new);
            block_weights[b] = w;
            block_sum += w;
        }

        float l_new = l_prev * correction + block_sum;

        // Rescale previous accumulator and add V contributions (unnormalized)
        for (uint dim_idx = 0; dim_idx < 4; dim_idx++) {
            uint d = lid + dim_idx * tg_size;
            if (d >= head_dim) break;

            float o_val = o_acc[dim_idx] * correction;

            for (uint b = 0; b < block_len; b++) {
                uint pos = block_start + b;
                float v_val;
                if (USE_FP8_KV) {
                    v_val = fp8_e4m3_to_float(((device const uchar*)V_cache_raw)[pos * kv_dim + kv_h * head_dim + d]) * V_scales[pos];
                } else {
                    v_val = ((device const float*)V_cache_raw)[pos * kv_dim + kv_h * head_dim + d];
                }
                o_val += block_weights[b] * v_val;
            }
            o_acc[dim_idx] = o_val;
        }

        m_prev = m_new;
        l_prev = l_new;
    }

    // Final normalization: divide by total softmax sum
    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    for (uint dim_idx = 0; dim_idx < 4; dim_idx++) {
        uint d = lid + dim_idx * tg_size;
        if (d >= head_dim) break;
        out[tgid * head_dim + d] = o_acc[dim_idx] * inv_l;
    }
}


// ============================================================================
// Kernel 10: GatedDeltaNet linear attention step (single token, all heads)
// ============================================================================
//
// Implements the GatedDeltaNet recurrence for autoregressive generation:
//   1. State decay:  S[vi][ki] *= g_decay
//   2. Memory read:  kv_mem[vi] = sum_ki(S[vi][ki] * k[ki])
//   3. Delta:        delta[vi] = (v[vi] - kv_mem[vi]) * beta_gate
//   4. State update: S[vi][ki] += k[ki] * delta[vi]
//   5. Output:       out[vi] = sum_ki(S[vi][ki] * q[ki])
//
// Dispatch: 64 threadgroups (one per v-head), 128 threads each (one per vi).
// Each thread owns one row S[head_id][vi][:] of the 128x128 state matrix.
//
// State layout: [64 * 128 * 128] float = 4MB total, persisted across tokens.
// k-head sharing: 4 v-heads share 1 k-head (64 v-heads / 16 k-heads).

kernel void gated_delta_net_step(
    device float *state,             // [64 * 128 * 128] persistent state
    device const float *q,           // [2048] (16 k-heads * 128)
    device const float *k,           // [2048] (16 k-heads * 128)
    device const float *v,           // [8192] (64 v-heads * 128)
    device const float *g_decay,     // [64] per v-head
    device const float *beta_gate,   // [64] per v-head
    device float *output,            // [8192] (64 v-heads * 128)
    constant uint &k_heads_per_v,    // = 4
    uint head_id [[threadgroup_position_in_grid]],
    uint vi [[thread_position_in_threadgroup]]
) {
    uint kh = head_id / k_heads_per_v;
    float g = g_decay[head_id];
    float beta = beta_gate[head_id];

    uint state_base = head_id * 128 * 128 + vi * 128;
    uint k_base = kh * 128;
    uint v_base = head_id * 128;

    // Step 1+2: Decay state row and compute kv_mem = dot(S[vi][:], k[:])
    float kv_mem = 0.0f;
    for (uint ki = 0; ki < 128; ki++) {
        float s = state[state_base + ki] * g;
        state[state_base + ki] = s;
        kv_mem += s * k[k_base + ki];
    }

    // Step 3+4: Delta update — S[vi][ki] += k[ki] * delta
    float delta = (v[v_base + vi] - kv_mem) * beta;
    for (uint ki = 0; ki < 128; ki++) {
        state[state_base + ki] += k[k_base + ki] * delta;
    }

    // Step 5: Output — out[vi] = dot(S[vi][:], q[:])
    float out_val = 0.0f;
    for (uint ki = 0; ki < 128; ki++) {
        out_val += state[state_base + ki] * q[k_base + ki];
    }
    output[v_base + vi] = out_val;
}


// ============================================================================
// Kernel 10b: Fused GatedDeltaNet recurrence step (pass 2+3 merged)
// ============================================================================
//
// Same as gated_delta_net_step but fuses the delta update (pass 2) and output
// query (pass 3) into a single loop over the 128-element state row. This
// eliminates 128 state reads per thread × 128 threads × 64 heads = ~1M fewer
// device memory reads per token.
//
// Pass 1 (decay + kv_mem) remains separate because it must complete before
// computing delta = (v[vi] - kv_mem) * beta.
//
// Dispatch: identical to gated_delta_net_step.

kernel void gated_delta_net_step_fused(
    device float *state,             // [64 * 128 * 128] persistent state
    device const float *q,           // [2048] (16 k-heads * 128)
    device const float *k,           // [2048] (16 k-heads * 128)
    device const float *v,           // [8192] (64 v-heads * 128)
    device const float *g_decay,     // [64] per v-head
    device const float *beta_gate,   // [64] per v-head
    device float *output,            // [8192] (64 v-heads * 128)
    constant uint &k_heads_per_v,    // = 4
    uint head_id [[threadgroup_position_in_grid]],
    uint vi [[thread_position_in_threadgroup]]
) {
    uint kh = head_id / k_heads_per_v;
    float g = g_decay[head_id];
    float beta = beta_gate[head_id];

    uint state_base = head_id * 128 * 128 + vi * 128;
    uint k_base = kh * 128;
    uint v_base = head_id * 128;

    // Pass 1: Decay state row and compute kv_mem = dot(S[vi][:], k[:])
    float kv_mem = 0.0f;
    for (uint ki = 0; ki < 128; ki++) {
        float s = state[state_base + ki] * g;
        state[state_base + ki] = s;
        kv_mem += s * k[k_base + ki];
    }

    // Pass 2+3 fused: Delta update + output query in single loop
    // After S[vi][ki] += k[ki] * delta, state is final — immediately
    // accumulate out += S[vi][ki] * q[ki] in the same iteration.
    float delta = (v[v_base + vi] - kv_mem) * beta;
    float out_val = 0.0f;
    for (uint ki = 0; ki < 128; ki++) {
        float s = state[state_base + ki] + k[k_base + ki] * delta;
        state[state_base + ki] = s;
        out_val += s * q[k_base + ki];
    }
    output[v_base + vi] = out_val;
}


// ============================================================================
// Kernel 11: Conv1d depthwise step (single token, incremental inference)
// ============================================================================
//
// Depthwise 1D convolution for one new input token:
//   output[c] = sum_k(history[k][c] * weight[c][k]) + input[c] * weight[c][3]
//   then SiLU activation: output[c] = output[c] / (1 + exp(-output[c]))
//
// After computing, shifts the history buffer left and appends the new input.
//
// Weight layout: [channels * kernel_size] bf16, weight[c * kernel_size + k]
// Conv state layout: [(kernel_size-1) * channels] row-major, state[k * channels + c]
// kernel_size = 4 (hardcoded), so 3 history slots + 1 new input.
//
// Dispatch: conv_dim threads (12288), one per channel.

kernel void conv1d_step(
    device float *conv_state,         // [(kernel_size-1) * conv_dim] = [3 * conv_dim]
    device const float *input,        // [conv_dim] current input
    device const uint16_t *weights,   // [conv_dim * 4] bf16 as uint16
    device float *output,             // [conv_dim] convolution output
    constant uint &conv_dim,          // = 12288
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= conv_dim) return;

    // Convolution: dot product of history + new input with weights
    // weight layout: weight[c * 4 + k] for channel c, position k
    uint w_base = idx * 4;
    float acc = 0.0f;

    // 3 history slots (k=0,1,2)
    acc += conv_state[0 * conv_dim + idx] * bf16_to_f32(weights[w_base + 0]);
    acc += conv_state[1 * conv_dim + idx] * bf16_to_f32(weights[w_base + 1]);
    acc += conv_state[2 * conv_dim + idx] * bf16_to_f32(weights[w_base + 2]);

    // New input (k=3)
    float inp = input[idx];
    acc += inp * bf16_to_f32(weights[w_base + 3]);

    // SiLU activation
    output[idx] = acc / (1.0f + exp(-acc));

    // Shift history: move slots 1,2 -> 0,1, append input at slot 2
    conv_state[0 * conv_dim + idx] = conv_state[1 * conv_dim + idx];
    conv_state[1 * conv_dim + idx] = conv_state[2 * conv_dim + idx];
    conv_state[2 * conv_dim + idx] = inp;
}


// ============================================================================
// Kernel 12: Per-head RMS normalize for q and k vectors
// ============================================================================
// q: [num_k_heads * key_dim], k: [num_k_heads * key_dim]
// Normalize each head independently, then scale by 1/sqrt(key_dim)^2 for q, 1/sqrt(key_dim) for k
// Dispatch: num_k_heads threadgroups, key_dim threads each

kernel void rms_norm_qk(
    device float *q,              // [num_k_heads * key_dim] in/out
    device float *k,              // [num_k_heads * key_dim] in/out
    constant uint &key_dim,       // = 128
    constant float &inv_scale,    // = 1/sqrt(key_dim)
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    uint base = head * key_dim;
    uint simd_lane = tid % 32;
    uint simd_group = tid / 32;

    // RMS norm for q — SIMD-accelerated reduction
    float qval = (tid < key_dim) ? q[base + tid] : 0;
    float q_sq = qval * qval;
    float q_simd = simd_sum(q_sq);
    threadgroup float q_shared[4];  // 128 threads = 4 SIMD groups
    if (simd_lane == 0) q_shared[simd_group] = q_simd;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q_total = q_shared[0] + q_shared[1] + q_shared[2] + q_shared[3];
    float q_inv_rms = rsqrt(q_total / float(key_dim) + 1e-6f);
    if (tid < key_dim) {
        q[base + tid] = qval * q_inv_rms * inv_scale * inv_scale;
    }

    // RMS norm for k — SIMD-accelerated reduction
    float kval = (tid < key_dim) ? k[base + tid] : 0;
    float k_sq = kval * kval;
    float k_simd = simd_sum(k_sq);
    threadgroup float k_shared[4];
    if (simd_lane == 0) k_shared[simd_group] = k_simd;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float k_total = k_shared[0] + k_shared[1] + k_shared[2] + k_shared[3];
    float k_inv_rms = rsqrt(k_total / float(key_dim) + 1e-6f);
    if (tid < key_dim) {
        k[base + tid] = kval * k_inv_rms * inv_scale;
    }
}


// ============================================================================
// Kernel 13: Compute g_decay and beta_gate for GatedDeltaNet
// ============================================================================
// Per v-head: g_decay = exp(-A * softplus(alpha + dt_bias)), beta_gate = sigmoid(beta)
// Dispatch: num_v_heads threads (64)

kernel void compute_decay_beta(
    device const float *alpha_out,   // [num_v_heads] from projection
    device const float *beta_out,    // [num_v_heads] from projection
    device const float *A_log,       // [num_v_heads] log of decay base (persistent)
    device const uint16_t *dt_bias,  // [num_v_heads] bf16
    device float *g_decay,           // [num_v_heads] output
    device float *beta_gate,         // [num_v_heads] output
    uint idx [[thread_position_in_grid]]
) {
    float a_val = alpha_out[idx];
    float dt_b = bf16_to_f32(dt_bias[idx]);
    float A_val = exp(A_log[idx]);
    float softplus_val = log(1.0f + exp(a_val + dt_b));
    g_decay[idx] = exp(-A_val * softplus_val);
    beta_gate[idx] = 1.0f / (1.0f + exp(-beta_out[idx]));
}


// ============================================================================
// Kernel 14: Gated RMS norm (z-gated output normalization)
// ============================================================================
// output[i] = rms_norm(values[i]) * SiLU(z[i]) * weight[i]
// Per v-head: normalize values, gate with z, scale with weight
// Dispatch: num_v_heads threadgroups, value_dim threads each

kernel void gated_rms_norm(
    device const float *values,       // [num_v_heads * value_dim] delta-net output
    device const float *z,            // [num_v_heads * value_dim] gate values
    device const uint16_t *weight,    // [value_dim] bf16 norm weights (shared across heads)
    device float *output,             // [num_v_heads * value_dim]
    constant uint &value_dim,         // = 128
    constant float &eps,              // = 1e-6
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    uint base = head * value_dim;

    float val = (tid < value_dim) ? values[base + tid] : 0;
    uint simd_lane = tid % 32;
    uint simd_group = tid / 32;

    // RMS norm reduction — SIMD-accelerated
    float sq = val * val;
    float simd_s = simd_sum(sq);
    threadgroup float partial[4];  // 128 threads = 4 SIMD groups
    if (simd_lane == 0) partial[simd_group] = simd_s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = partial[0] + partial[1] + partial[2] + partial[3];
    float inv_rms = rsqrt(total / float(value_dim) + eps);

    if (tid < value_dim) {
        float normed = val * inv_rms;
        float zval = z[base + tid];
        float gate = zval / (1.0f + exp(-zval));  // SiLU
        float w = bf16_to_f32(weight[tid]);
        output[base + tid] = normed * gate * w;
    }
}


// ============================================================================
// Kernel 12: MoE combine + residual + shared expert gate (fused)
// ============================================================================
// Fused operation for CMD3 GPU-side combine:
//   hidden[i] = h_mid[i] + sum_k(expert_weight[k] * expert_out[k][i])
//               + sigmoid(shared_gate_score) * shared_out[i]
//
// All 8 expert output buffers are always bound (unused ones have weight=0).
// This avoids variable buffer bindings and keeps the dispatch simple.
//
// Dispatch: (dim + 255) / 256 threadgroups, 256 threads each.

kernel void moe_combine_residual(
    device const float* h_mid       [[buffer(0)]],   // [dim]
    device const float* shared_out  [[buffer(1)]],   // [dim]
    device float*       hidden_out  [[buffer(2)]],   // [dim] output
    device const float* expert_out0 [[buffer(3)]],   // [dim] expert 0
    device const float* expert_out1 [[buffer(4)]],   // [dim] expert 1
    device const float* expert_out2 [[buffer(5)]],   // [dim] expert 2
    device const float* expert_out3 [[buffer(6)]],   // [dim] expert 3
    device const float* expert_out4 [[buffer(7)]],   // [dim] expert 4
    device const float* expert_out5 [[buffer(8)]],   // [dim] expert 5
    device const float* expert_out6 [[buffer(9)]],   // [dim] expert 6
    device const float* expert_out7 [[buffer(10)]],  // [dim] expert 7
    device const float* params      [[buffer(11)]],  // [10]: weights[0..7], shared_gate_score, (unused)
    constant uint&      dim         [[buffer(12)]],
    constant uint&      K           [[buffer(13)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    // Read expert weights and shared gate from params buffer
    float shared_gate = 1.0f / (1.0f + exp(-params[8]));  // sigmoid(shared_gate_score)

    // Weighted sum of expert outputs
    float moe = 0.0f;
    // Unrolled for MAX_K=8 with branch on K to avoid reading invalid buffers
    if (K > 0) moe += params[0] * expert_out0[tid];
    if (K > 1) moe += params[1] * expert_out1[tid];
    if (K > 2) moe += params[2] * expert_out2[tid];
    if (K > 3) moe += params[3] * expert_out3[tid];
    if (K > 4) moe += params[4] * expert_out4[tid];
    if (K > 5) moe += params[5] * expert_out5[tid];
    if (K > 6) moe += params[6] * expert_out6[tid];
    if (K > 7) moe += params[7] * expert_out7[tid];

    hidden_out[tid] = h_mid[tid] + moe + shared_gate * shared_out[tid];
}


// ============================================================================
// FP16 ACCUMULATION VARIANTS — Experimental half-precision inner loops
// ============================================================================
// These kernels accumulate dot products in half precision for ~2x ALU throughput
// on Apple Silicon fp16 units. Output buffers remain float32.
// Default OFF — toggled via g_use_fp16_accum / --fp16 flag.

// ---- fp16 variant of dequant_matvec_4bit_v3 ----
kernel void dequant_matvec_4bit_v3_fp16(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_size  [[threads_per_simdgroup]]
) {
    uint rows_per_tg = 256 / simd_size;
    uint row = tgid * rows_per_tg + simd_group;

    uint packed_cols = in_dim / 8;
    uint num_groups  = in_dim / group_size;

    threadgroup half x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = half(x[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    half acc = 0.0h;

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / (group_size / 8);
        half scale = half(bf16_to_f32(s_row[g]));
        half bias  = half(bf16_to_f32(b_row[g]));

        uint32_t packed = w_row[col];
        uint x_base = col * 8;

        half sx0 = scale * x_shared[x_base + 0];  half bx0 = bias * x_shared[x_base + 0];
        half sx1 = scale * x_shared[x_base + 1];  half bx1 = bias * x_shared[x_base + 1];
        half sx2 = scale * x_shared[x_base + 2];  half bx2 = bias * x_shared[x_base + 2];
        half sx3 = scale * x_shared[x_base + 3];  half bx3 = bias * x_shared[x_base + 3];
        half sx4 = scale * x_shared[x_base + 4];  half bx4 = bias * x_shared[x_base + 4];
        half sx5 = scale * x_shared[x_base + 5];  half bx5 = bias * x_shared[x_base + 5];
        half sx6 = scale * x_shared[x_base + 6];  half bx6 = bias * x_shared[x_base + 6];
        half sx7 = scale * x_shared[x_base + 7];  half bx7 = bias * x_shared[x_base + 7];

        acc += fma(half((packed >>  0) & 0xF), sx0, bx0);
        acc += fma(half((packed >>  4) & 0xF), sx1, bx1);
        acc += fma(half((packed >>  8) & 0xF), sx2, bx2);
        acc += fma(half((packed >> 12) & 0xF), sx3, bx3);
        acc += fma(half((packed >> 16) & 0xF), sx4, bx4);
        acc += fma(half((packed >> 20) & 0xF), sx5, bx5);
        acc += fma(half((packed >> 24) & 0xF), sx6, bx6);
        acc += fma(half((packed >> 28) & 0xF), sx7, bx7);
    }

    // Promote to float for SIMD reduction and output write
    float sum = float(simd_sum(acc));

    if (simd_lane == 0) {
        out[row] = sum;
    }
}

// ---- fp16 variant of dequant_matvec_2bit ----
kernel void dequant_matvec_2bit_fp16(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint lid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_size  [[threads_per_simdgroup]]
) {
    uint row = tgid * (256 / simd_size) + simd_group;
    uint packed_cols = in_dim / 16;
    uint num_groups  = in_dim / group_size;

    threadgroup half x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = half(x[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= out_dim) return;

    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    half acc = 0.0h;

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / (group_size / 16);
        half scale = half(bf16_to_f32(s_row[g]));
        half bias  = half(bf16_to_f32(b_row[g]));

        uint32_t packed = w_row[col];
        uint x_base = col * 16;

        half sx0  = scale * x_shared[x_base +  0];  half bx0  = bias * x_shared[x_base +  0];
        half sx1  = scale * x_shared[x_base +  1];  half bx1  = bias * x_shared[x_base +  1];
        half sx2  = scale * x_shared[x_base +  2];  half bx2  = bias * x_shared[x_base +  2];
        half sx3  = scale * x_shared[x_base +  3];  half bx3  = bias * x_shared[x_base +  3];
        half sx4  = scale * x_shared[x_base +  4];  half bx4  = bias * x_shared[x_base +  4];
        half sx5  = scale * x_shared[x_base +  5];  half bx5  = bias * x_shared[x_base +  5];
        half sx6  = scale * x_shared[x_base +  6];  half bx6  = bias * x_shared[x_base +  6];
        half sx7  = scale * x_shared[x_base +  7];  half bx7  = bias * x_shared[x_base +  7];
        half sx8  = scale * x_shared[x_base +  8];  half bx8  = bias * x_shared[x_base +  8];
        half sx9  = scale * x_shared[x_base +  9];  half bx9  = bias * x_shared[x_base +  9];
        half sx10 = scale * x_shared[x_base + 10];  half bx10 = bias * x_shared[x_base + 10];
        half sx11 = scale * x_shared[x_base + 11];  half bx11 = bias * x_shared[x_base + 11];
        half sx12 = scale * x_shared[x_base + 12];  half bx12 = bias * x_shared[x_base + 12];
        half sx13 = scale * x_shared[x_base + 13];  half bx13 = bias * x_shared[x_base + 13];
        half sx14 = scale * x_shared[x_base + 14];  half bx14 = bias * x_shared[x_base + 14];
        half sx15 = scale * x_shared[x_base + 15];  half bx15 = bias * x_shared[x_base + 15];

        acc += fma(half((packed >>  0) & 0x3), sx0,  bx0);
        acc += fma(half((packed >>  2) & 0x3), sx1,  bx1);
        acc += fma(half((packed >>  4) & 0x3), sx2,  bx2);
        acc += fma(half((packed >>  6) & 0x3), sx3,  bx3);
        acc += fma(half((packed >>  8) & 0x3), sx4,  bx4);
        acc += fma(half((packed >> 10) & 0x3), sx5,  bx5);
        acc += fma(half((packed >> 12) & 0x3), sx6,  bx6);
        acc += fma(half((packed >> 14) & 0x3), sx7,  bx7);
        acc += fma(half((packed >> 16) & 0x3), sx8,  bx8);
        acc += fma(half((packed >> 18) & 0x3), sx9,  bx9);
        acc += fma(half((packed >> 20) & 0x3), sx10, bx10);
        acc += fma(half((packed >> 22) & 0x3), sx11, bx11);
        acc += fma(half((packed >> 24) & 0x3), sx12, bx12);
        acc += fma(half((packed >> 26) & 0x3), sx13, bx13);
        acc += fma(half((packed >> 28) & 0x3), sx14, bx14);
        acc += fma(half((packed >> 30) & 0x3), sx15, bx15);
    }

    float sum = float(simd_sum(acc));
    if (simd_lane == 0) {
        out[row] = sum;
    }
}

// ---- fp16 variant of fused_gate_up_swiglu ----
// NOTE: Only dot product accumulation is fp16. SiLU activation stays float32
// to avoid exp() overflow in half precision.
kernel void fused_gate_up_swiglu_fp16(
    device const uint32_t* gate_W    [[buffer(0)]],
    device const uint16_t* gate_s    [[buffer(1)]],
    device const uint16_t* gate_b    [[buffer(2)]],
    device const uint32_t* up_W      [[buffer(3)]],
    device const uint16_t* up_s      [[buffer(4)]],
    device const uint16_t* up_b      [[buffer(5)]],
    device const float*    x         [[buffer(6)]],
    device float*          out       [[buffer(7)]],
    constant uint&         out_dim   [[buffer(8)]],
    constant uint&         in_dim    [[buffer(9)]],
    constant uint&         group_size [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_size  [[threads_per_simdgroup]]
) {
    if (tgid >= out_dim) return;
    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;
    device const uint32_t* gr = gate_W + tgid * packed_cols;
    device const uint16_t* gs = gate_s + tgid * num_groups;
    device const uint16_t* gb = gate_b + tgid * num_groups;
    device const uint32_t* ur = up_W   + tgid * packed_cols;
    device const uint16_t* us = up_s   + tgid * num_groups;
    device const uint16_t* ub = up_b   + tgid * num_groups;
    half ga = 0.0h, ua = 0.0h;
    for (uint g = lid; g < num_groups; g += tg_size) {
        half gsc = half(bf16_to_f32(gs[g])), gbi = half(bf16_to_f32(gb[g]));
        half usc = half(bf16_to_f32(us[g])), ubi = half(bf16_to_f32(ub[g]));
        uint bp = g * packed_per_group, bx = g * group_size;
        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t gp = gr[bp+p], up = ur[bp+p];
            for (uint i = 0; i < 8; i++) {
                half xv = half(x[bx + p*8 + i]);
                ga += (half((gp>>(i*4))&0xF)*gsc+gbi)*xv;
                ua += (half((up>>(i*4))&0xF)*usc+ubi)*xv;
            }
        }
    }
    // Reduction using dynamic SIMD width
    uint num_simd_groups = tg_size / simd_size;
    threadgroup float sg[32], su[32];
    // Promote to float for reduction — SiLU MUST be computed in float32
    float rg = float(simd_sum(ga)), ru = float(simd_sum(ua));
    if (simd_lane == 0) { sg[simd_group] = rg; su[simd_group] = ru; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0 && simd_lane < num_simd_groups) {
        float vg = simd_sum(sg[simd_lane]), vu = simd_sum(su[simd_lane]);
        // SiLU activation in float32 to avoid exp() overflow in half
        if (simd_lane == 0) out[tgid] = (vg / (1.0f + exp(-vg))) * vu;
    }
}


// ============================================================================
// BATCHED PREFILL KERNELS
// ============================================================================
// Convert GEMV (1 token) to GEMM (N tokens) — amortize weight reads across batch.
// MAX_PFB_GPU = 32 is the maximum batch size per GPU dispatch.

#define MAX_PFB_GPU 32

// ============================================================================
// Kernel P1: 4-bit dequantized GEMM for batched prefill
// ============================================================================
// Input:  X[N, in_dim], weights W[out_dim, in_dim/8] (4-bit packed), scales/biases (bf16)
// Output: Y[N, out_dim]
// Read each weight row ONCE, multiply against all N input vectors.
// 256 threads per threadgroup, 8 SIMD groups = 8 rows per threadgroup.

kernel void dequant_gemm_4bit_batch(
    device const uint32_t* W_packed   [[buffer(0)]],  // [out_dim, in_dim/8]
    device const uint16_t* scales     [[buffer(1)]],  // [out_dim, num_groups] bf16
    device const uint16_t* biases     [[buffer(2)]],  // [out_dim, num_groups] bf16
    device const float*    X          [[buffer(3)]],  // [N, in_dim]
    device float*          Y          [[buffer(4)]],  // [N, out_dim]
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    constant uint&         batch_n    [[buffer(8)]],  // number of tokens in batch
    uint tgid       [[threadgroup_position_in_grid]],
    uint lid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_size  [[threads_per_simdgroup]]
) {
    // Which output row this SIMD group handles
    uint row = tgid * (256 / simd_size) + simd_group;
    if (row >= out_dim) return;

    uint packed_cols = in_dim / 8;
    uint num_groups  = in_dim / group_size;

    // Per-token accumulators (one per batch token)
    float acc[MAX_PFB_GPU];
    for (uint t = 0; t < batch_n; t++) acc[t] = 0.0f;

    // Pointer setup for this row's weights
    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    // Each lane processes a strided slice of packed columns
    for (uint col = simd_lane; col < packed_cols; col += simd_size) {
        uint g = col / (group_size / 8);
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint32_t packed = w_row[col];
        uint x_base = col * 8;

        // Extract 8 nibbles
        float n0 = float((packed >>  0) & 0xF);
        float n1 = float((packed >>  4) & 0xF);
        float n2 = float((packed >>  8) & 0xF);
        float n3 = float((packed >> 12) & 0xF);
        float n4 = float((packed >> 16) & 0xF);
        float n5 = float((packed >> 20) & 0xF);
        float n6 = float((packed >> 24) & 0xF);
        float n7 = float((packed >> 28) & 0xF);

        // For each token in the batch, compute FMA-optimized dot product
        for (uint t = 0; t < batch_n; t++) {
            device const float* x_t = X + t * in_dim + x_base;
            float x0 = x_t[0], x1 = x_t[1], x2 = x_t[2], x3 = x_t[3];
            float x4 = x_t[4], x5 = x_t[5], x6 = x_t[6], x7 = x_t[7];

            // FMA pattern: fma(nibble, scale*x, bias*x)
            acc[t] += fma(n0, scale * x0, bias * x0);
            acc[t] += fma(n1, scale * x1, bias * x1);
            acc[t] += fma(n2, scale * x2, bias * x2);
            acc[t] += fma(n3, scale * x3, bias * x3);
            acc[t] += fma(n4, scale * x4, bias * x4);
            acc[t] += fma(n5, scale * x5, bias * x5);
            acc[t] += fma(n6, scale * x6, bias * x6);
            acc[t] += fma(n7, scale * x7, bias * x7);
        }
    }

    // SIMD reduction and write: for each token
    for (uint t = 0; t < batch_n; t++) {
        float sum = simd_sum(acc[t]);
        if (simd_lane == 0) {
            Y[t * out_dim + row] = sum;
        }
    }
}


// ============================================================================
// Kernel P2: Batched prefill causal attention (online softmax, FlashAttention-2)
// ============================================================================
// Grid: N * num_heads threadgroups, 256 threads (= head_dim)
// One threadgroup per (query_token, head) pair.
// Uses unnormalized accumulator pattern (proven stable).
// Causal masking: query t attends to positions [0, cache_start + t].
// Fused sigmoid gate at the end.

kernel void prefill_causal_attn(
    device const float* Q         [[buffer(0)]],  // [N, num_heads * head_dim]
    device const float* K_cache   [[buffer(1)]],  // [max_seq, kv_dim]
    device const float* V_cache   [[buffer(2)]],  // [max_seq, kv_dim]
    device const float* gate_buf  [[buffer(3)]],  // [N, num_heads * head_dim] sigmoid gate
    device float*       output    [[buffer(4)]],  // [N, num_heads * head_dim]
    constant uint&      head_dim   [[buffer(5)]],
    constant uint&      kv_dim     [[buffer(6)]],
    constant uint&      num_heads  [[buffer(7)]],
    constant uint&      heads_per_kv [[buffer(8)]],
    constant uint&      cache_start [[buffer(9)]],  // starting position in KV cache
    constant uint&      batch_n    [[buffer(10)]],
    constant float&     scale      [[buffer(11)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    // Decode (token_idx, head) from linearized threadgroup ID
    uint token_idx = tgid / num_heads;
    uint h = tgid % num_heads;
    if (token_idx >= batch_n) return;

    uint kv_h = h / heads_per_kv;
    uint d = lid;  // thread maps to dimension (assumes tg_size >= head_dim)

    // Query vector for this token and head
    device const float* q_h = Q + token_idx * (num_heads * head_dim) + h * head_dim;

    // Causal: attend to positions [0, cache_start + token_idx]
    uint seq_end = cache_start + token_idx + 1;

    // Online softmax with unnormalized accumulator (FlashAttention-2)
    float m = -1e30f;  // running max
    float l = 0.0f;    // running sum of exp
    float o_acc = 0.0f; // unnormalized output accumulator

    for (uint p = 0; p < seq_end; p++) {
        // Compute Q @ K^T for position p — reduction across head_dim
        device const float* k_p = K_cache + p * kv_dim + kv_h * head_dim;
        float dot = 0.0f;
        if (d < head_dim) {
            dot = q_h[d] * k_p[d];
        }
        // Reduce dot product across threads
        float score = simd_sum(dot);

        // Broadcast score from lane 0 to all threads (needed for cross-SIMD)
        threadgroup float shared_score;
        // Full threadgroup reduction for head_dim=256 (8 SIMD groups)
        uint simd_lane_l = lid % 32;
        uint simd_group_l = lid / 32;
        threadgroup float partial_scores[8];
        float s_reduced = simd_sum(dot);
        if (simd_lane_l == 0) partial_scores[simd_group_l] = s_reduced;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            float total = 0.0f;
            uint num_sg = (tg_size + 31) / 32;
            for (uint i = 0; i < num_sg; i++) total += partial_scores[i];
            shared_score = total * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        score = shared_score;

        // Online softmax update
        float m_new = max(m, score);
        float correction = exp(m - m_new);
        float w = exp(score - m_new);

        // Update unnormalized accumulator: o = o * correction + w * v
        device const float* v_p = V_cache + p * kv_dim + kv_h * head_dim;
        if (d < head_dim) {
            o_acc = o_acc * correction + w * v_p[d];
        }

        l = l * correction + w;
        m = m_new;
    }

    // Normalize
    if (d < head_dim && l > 0.0f) {
        float result = o_acc / l;
        // Fused sigmoid gate
        float g = gate_buf[token_idx * (num_heads * head_dim) + h * head_dim + d];
        float sig = 1.0f / (1.0f + exp(-g));
        output[token_idx * (num_heads * head_dim) + h * head_dim + d] = result * sig;
    }
}


// ============================================================================
// Kernel P3: Fused Q deinterleave + RMS norm + RoPE for batched prefill
// ============================================================================
// Grid: [num_heads, batch_n, 1], 256 threads (= head_dim)

kernel void prefill_q_rope_norm_bf16(
    device float*          Q_out      [[buffer(0)]],  // [N, num_heads * head_dim] in/out
    device const uint16_t* norm_w     [[buffer(1)]],  // [head_dim] bf16 per-head norm weights
    constant uint&         head_dim   [[buffer(2)]],
    constant uint&         num_heads  [[buffer(3)]],
    constant uint&         batch_n    [[buffer(4)]],
    constant uint&         cache_start [[buffer(5)]],  // starting RoPE position
    constant float&        rope_theta  [[buffer(6)]],
    constant float&        eps        [[buffer(7)]],
    constant uint&         rotary_dim [[buffer(8)]],
    uint tgid_flat [[threadgroup_position_in_grid]],
    uint lid       [[thread_position_in_threadgroup]]
) {
    const uint tg_size = 256;  // always dispatched with 256 threads
    uint h = tgid_flat % num_heads;
    uint t = tgid_flat / num_heads;
    if (h >= num_heads || t >= batch_n) return;

    uint base = t * (num_heads * head_dim) + h * head_dim;
    device float* q = Q_out + base;

    uint d = lid;
    if (d >= head_dim) return;

    float val = q[d];

    // RMS norm reduction
    float sq = val * val;
    uint simd_lane_l = lid % 32;
    uint simd_group_l = lid / 32;
    float s_sq = simd_sum(sq);
    threadgroup float partial[8];
    if (simd_lane_l == 0) partial[simd_group_l] = s_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total_sq = 0.0f;
    if (simd_group_l == 0 && simd_lane_l < ((tg_size + 31) / 32)) {
        total_sq = simd_sum(partial[simd_lane_l]);
    }
    threadgroup float broadcast_sq;
    if (lid == 0) broadcast_sq = total_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = rsqrt(broadcast_sq / float(head_dim) + eps);

    // Apply norm weight
    float w = bf16_to_f32(norm_w[d]);
    val = val * inv_rms * w;

    // RoPE
    uint pos = cache_start + t;
    if (d < rotary_dim) {
        uint half_dim = rotary_dim / 2;
        uint pair_idx = d % half_dim;
        float freq = 1.0f / pow(rope_theta, float(2 * pair_idx) / float(rotary_dim));
        float angle = float(pos) * freq;
        float cos_a = cos(angle);
        float sin_a = sin(angle);

        // Read the paired element
        float val_pair;
        if (d < half_dim) {
            val_pair = q[d + half_dim] * inv_rms * bf16_to_f32(norm_w[d + half_dim]);
        } else {
            val_pair = q[d - half_dim] * inv_rms * bf16_to_f32(norm_w[d - half_dim]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (d < half_dim) {
            val = val * cos_a - val_pair * sin_a;
        } else {
            val = val_pair * sin_a + val * cos_a;
        }
    }

    q[d] = val;
}


// ============================================================================
// Kernel P4: K norm + RoPE + KV cache write for batched prefill
// ============================================================================
// Grid: [num_kv_heads, batch_n, 1], 256 threads

kernel void prefill_kv_cache_bf16(
    device const float*    K_in       [[buffer(0)]],  // [N, num_kv_heads * head_dim]
    device const float*    V_in       [[buffer(1)]],  // [N, num_kv_heads * head_dim]
    device float*          K_cache    [[buffer(2)]],  // [max_seq, kv_dim]
    device float*          V_cache    [[buffer(3)]],  // [max_seq, kv_dim]
    device const uint16_t* k_norm_w   [[buffer(4)]],  // [head_dim] bf16
    constant uint&         head_dim   [[buffer(5)]],
    constant uint&         kv_dim     [[buffer(6)]],
    constant uint&         num_kv_heads [[buffer(7)]],
    constant uint&         batch_n    [[buffer(8)]],
    constant uint&         cache_start [[buffer(9)]],
    constant float&        rope_theta  [[buffer(10)]],
    constant float&        eps        [[buffer(11)]],
    constant uint&         rotary_dim [[buffer(12)]],
    uint tgid_flat [[threadgroup_position_in_grid]],
    uint lid       [[thread_position_in_threadgroup]]
) {
    const uint tg_size = 256;  // always dispatched with 256 threads
    uint kv_h = tgid_flat % num_kv_heads;
    uint t = tgid_flat / num_kv_heads;
    if (kv_h >= num_kv_heads || t >= batch_n) return;

    uint d = lid;
    if (d >= head_dim) return;

    uint in_base = t * (num_kv_heads * head_dim) + kv_h * head_dim;
    uint cache_pos = cache_start + t;

    // K: RMS norm
    float k_val = K_in[in_base + d];
    float sq = k_val * k_val;
    uint simd_lane_l = lid % 32;
    uint simd_group_l = lid / 32;
    float s_sq = simd_sum(sq);
    threadgroup float partial[8];
    if (simd_lane_l == 0) partial[simd_group_l] = s_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total_sq = 0.0f;
    if (simd_group_l == 0 && simd_lane_l < ((tg_size + 31) / 32)) {
        total_sq = simd_sum(partial[simd_lane_l]);
    }
    threadgroup float broadcast_sq;
    if (lid == 0) broadcast_sq = total_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = rsqrt(broadcast_sq / float(head_dim) + eps);

    float w = bf16_to_f32(k_norm_w[d]);
    k_val = k_val * inv_rms * w;

    // RoPE for K
    uint pos = cache_start + t;
    if (d < rotary_dim) {
        uint half_dim = rotary_dim / 2;
        uint pair_idx = d % half_dim;
        float freq = 1.0f / pow(rope_theta, float(2 * pair_idx) / float(rotary_dim));
        float angle = float(pos) * freq;
        float cos_a = cos(angle);
        float sin_a = sin(angle);

        float k_pair;
        float k_raw_pair = K_in[in_base + (d < half_dim ? d + half_dim : d - half_dim)];
        // Need to norm the pair too
        k_pair = k_raw_pair * inv_rms * bf16_to_f32(k_norm_w[d < half_dim ? d + half_dim : d - half_dim]);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (d < half_dim) {
            k_val = k_val * cos_a - k_pair * sin_a;
        } else {
            k_val = k_pair * sin_a + k_val * cos_a;
        }
    }

    // Write K to cache
    K_cache[cache_pos * kv_dim + kv_h * head_dim + d] = k_val;

    // V: just write directly (no norm or RoPE)
    V_cache[cache_pos * kv_dim + kv_h * head_dim + d] = V_in[in_base + d];
}


// ============================================================================
// Kernel P5: RMS norm for N tokens (one threadgroup per token)
// ============================================================================

kernel void prefill_rms_norm_bf16(
    device const float*    x      [[buffer(0)]],  // [N, dim]
    device const uint16_t* weight [[buffer(1)]],  // [dim] bf16
    device float*          out    [[buffer(2)]],  // [N, dim]
    constant uint&         dim    [[buffer(3)]],
    constant float&        eps    [[buffer(4)]],
    constant uint&         batch_n [[buffer(5)]],
    uint tgid   [[threadgroup_position_in_grid]],  // token index
    uint lid    [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tgid >= batch_n) return;

    device const float* x_t = x + tgid * dim;
    device float* out_t = out + tgid * dim;

    // Sum of squares reduction
    float acc = 0.0f;
    for (uint i = lid; i < dim; i += tg_size) {
        float v = x_t[i];
        acc += v * v;
    }

    float simd_val = simd_sum(acc);
    uint simd_lane_l = lid % 32;
    uint simd_group_l = lid / 32;
    threadgroup float shared[8];
    if (simd_lane_l == 0) shared[simd_group_l] = simd_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = 0.0f;
    if (simd_group_l == 0) {
        uint num_sg = (tg_size + 31) / 32;
        float v = (simd_lane_l < num_sg) ? shared[simd_lane_l] : 0.0f;
        total = simd_sum(v);
    }
    threadgroup float broadcast;
    if (lid == 0) broadcast = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms = rsqrt(broadcast / float(dim) + eps);

    // Apply norm + weight
    for (uint i = lid; i < dim; i += tg_size) {
        out_t[i] = x_t[i] * inv_rms * bf16_to_f32(weight[i]);
    }
}


// ============================================================================
// Kernel P6: Fused residual + RMS norm for N tokens
// ============================================================================

kernel void prefill_residual_norm_bf16(
    device const float*    residual [[buffer(0)]],  // [N, dim]
    device const float*    x        [[buffer(1)]],  // [N, dim] (to add)
    device const uint16_t* weight   [[buffer(2)]],  // [dim] bf16
    device float*          out      [[buffer(3)]],  // [N, dim]
    device float*          hidden   [[buffer(4)]],  // [N, dim] store residual+x for next layer
    constant uint&         dim      [[buffer(5)]],
    constant float&        eps      [[buffer(6)]],
    constant uint&         batch_n  [[buffer(7)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tgid >= batch_n) return;

    device const float* r_t = residual + tgid * dim;
    device const float* x_t = x + tgid * dim;
    device float* out_t = out + tgid * dim;
    device float* h_t = hidden + tgid * dim;

    // Compute residual + x, and sum of squares
    float acc = 0.0f;
    for (uint i = lid; i < dim; i += tg_size) {
        float v = r_t[i] + x_t[i];
        h_t[i] = v;
        acc += v * v;
    }

    float simd_val = simd_sum(acc);
    uint simd_lane_l = lid % 32;
    uint simd_group_l = lid / 32;
    threadgroup float shared[8];
    if (simd_lane_l == 0) shared[simd_group_l] = simd_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = 0.0f;
    if (simd_group_l == 0) {
        uint num_sg = (tg_size + 31) / 32;
        float v = (simd_lane_l < num_sg) ? shared[simd_lane_l] : 0.0f;
        total = simd_sum(v);
    }
    threadgroup float broadcast;
    if (lid == 0) broadcast = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms = rsqrt(broadcast / float(dim) + eps);

    for (uint i = lid; i < dim; i += tg_size) {
        out_t[i] = h_t[i] * inv_rms * bf16_to_f32(weight[i]);
    }
}


// ============================================================================
// Kernel P7: Elementwise SwiGLU for N*dim elements
// ============================================================================

kernel void prefill_swiglu(
    device const float* gate [[buffer(0)]],  // [N * dim]
    device const float* up   [[buffer(1)]],  // [N * dim]
    device float*       out  [[buffer(2)]],  // [N * dim]
    constant uint&      total_elems [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= total_elems) return;
    float g = gate[tid];
    float silu_g = g / (1.0f + exp(-g));
    out[tid] = silu_g * up[tid];
}


// ============================================================================
// Kernel P8: Batched combine: hidden = h_mid + sigmoid(gate) * shared_out
// ============================================================================

kernel void prefill_combine(
    device const float* h_mid       [[buffer(0)]],  // [N, dim]
    device const float* shared_out  [[buffer(1)]],  // [N, dim]
    device const float* gate_scores [[buffer(2)]],  // [N] per-token sigmoid gate score
    device float*       hidden_out  [[buffer(3)]],  // [N, dim]
    constant uint&      dim         [[buffer(4)]],
    constant uint&      batch_n     [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = batch_n * dim;
    if (tid >= total) return;

    uint t = tid / dim;
    float shared_gate = 1.0f / (1.0f + exp(-gate_scores[t]));
    hidden_out[tid] = h_mid[tid] + shared_gate * shared_out[tid];
}


// ============================================================================
// Kernel P9: Conv1d step batched — process N tokens sequentially through conv state
// ============================================================================

kernel void conv1d_step_batched(
    device float *conv_state,         // [(kernel_size-1) * conv_dim]
    device const float *input,        // [N, conv_dim]
    device const uint16_t *weights,   // [conv_dim * 4] bf16
    device float *output,             // [N, conv_dim]
    constant uint &conv_dim,
    constant uint &batch_n,
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= conv_dim) return;

    uint w_base = idx * 4;
    float w0 = bf16_to_f32(weights[w_base + 0]);
    float w1 = bf16_to_f32(weights[w_base + 1]);
    float w2 = bf16_to_f32(weights[w_base + 2]);
    float w3 = bf16_to_f32(weights[w_base + 3]);

    // Process tokens sequentially (conv state is causal)
    for (uint t = 0; t < batch_n; t++) {
        float inp = input[t * conv_dim + idx];

        float acc = conv_state[0 * conv_dim + idx] * w0
                  + conv_state[1 * conv_dim + idx] * w1
                  + conv_state[2 * conv_dim + idx] * w2
                  + inp * w3;

        // SiLU activation
        output[t * conv_dim + idx] = acc / (1.0f + exp(-acc));

        // Shift history
        conv_state[0 * conv_dim + idx] = conv_state[1 * conv_dim + idx];
        conv_state[1 * conv_dim + idx] = conv_state[2 * conv_dim + idx];
        conv_state[2 * conv_dim + idx] = inp;
    }
}


// ============================================================================
// Kernel P10: Per-head RMS norm for Q and K, batched
// ============================================================================

kernel void rms_norm_qk_batched(
    device float *q,              // [N, num_k_heads * key_dim] in/out
    device float *k,              // [N, num_k_heads * key_dim] in/out
    constant uint &key_dim,
    constant float &inv_scale,
    constant uint &num_k_heads,
    constant uint &batch_n,
    uint tgid_flat [[threadgroup_position_in_grid]],
    uint tid       [[thread_position_in_threadgroup]]
) {
    uint head = tgid_flat % num_k_heads;
    uint t = tgid_flat / num_k_heads;
    if (head >= num_k_heads || t >= batch_n) return;

    uint stride = num_k_heads * key_dim;
    uint base = t * stride + head * key_dim;
    uint simd_lane_l = tid % 32;
    uint simd_group_l = tid / 32;

    // RMS norm for q
    float qval = (tid < key_dim) ? q[base + tid] : 0;
    float q_sq = qval * qval;
    float q_simd = simd_sum(q_sq);
    threadgroup float q_shared[4];
    if (simd_lane_l == 0) q_shared[simd_group_l] = q_simd;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q_total = q_shared[0] + q_shared[1] + q_shared[2] + q_shared[3];
    float q_inv_rms = rsqrt(q_total / float(key_dim) + 1e-6f);
    if (tid < key_dim) {
        q[base + tid] = qval * q_inv_rms * inv_scale * inv_scale;
    }

    // RMS norm for k
    float kval = (tid < key_dim) ? k[base + tid] : 0;
    float k_sq = kval * kval;
    float k_simd = simd_sum(k_sq);
    threadgroup float k_shared[4];
    if (simd_lane_l == 0) k_shared[simd_group_l] = k_simd;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float k_total = k_shared[0] + k_shared[1] + k_shared[2] + k_shared[3];
    float k_inv_rms = rsqrt(k_total / float(key_dim) + 1e-6f);
    if (tid < key_dim) {
        k[base + tid] = kval * k_inv_rms * inv_scale;
    }
}


// ============================================================================
// Kernel P11: Compute g_decay and beta_gate, batched
// ============================================================================

kernel void compute_decay_beta_batched(
    device const float *alpha_out,   // [N, num_v_heads]
    device const float *beta_out,    // [N, num_v_heads]
    device const float *A_log,       // [num_v_heads] persistent
    device const uint16_t *dt_bias,  // [num_v_heads] bf16
    device float *g_decay_out,       // [N, num_v_heads]
    device float *beta_gate_out,     // [N, num_v_heads]
    constant uint &num_v_heads,
    constant uint &batch_n,
    uint tid [[thread_position_in_grid]]
) {
    uint total = batch_n * num_v_heads;
    if (tid >= total) return;

    uint t = tid / num_v_heads;
    uint h = tid % num_v_heads;

    float a_val = alpha_out[t * num_v_heads + h];
    float dt_b = bf16_to_f32(dt_bias[h]);
    float A_val = exp(A_log[h]);
    float softplus_val = log(1.0f + exp(a_val + dt_b));
    g_decay_out[t * num_v_heads + h] = exp(-A_val * softplus_val);
    beta_gate_out[t * num_v_heads + h] = 1.0f / (1.0f + exp(-beta_out[t * num_v_heads + h]));
}


// ============================================================================
// Kernel P12: GatedDeltaNet step, batched — sequential per head within batch
// ============================================================================
// One workgroup per v-head. Tokens processed sequentially (causal state update).
// Each thread owns one row S[head_id][vi][:] of the state matrix.

kernel void gated_delta_net_step_batched(
    device float *state,             // [64 * 128 * 128] persistent state
    device const float *q,           // [N, total_key]
    device const float *k,           // [N, total_key]
    device const float *v,           // [N, total_value]
    device const float *g_decay_buf, // [N, num_v_heads]
    device const float *beta_buf,    // [N, num_v_heads]
    device float *output,            // [N, total_value]
    constant uint &k_heads_per_v,
    constant uint &key_dim,
    constant uint &value_dim,
    constant uint &total_key,
    constant uint &total_value,
    constant uint &num_v_heads,
    constant uint &batch_n,
    uint head_id [[threadgroup_position_in_grid]],
    uint vi [[thread_position_in_threadgroup]]
) {
    if (vi >= value_dim) return;

    uint kh = head_id / k_heads_per_v;

    for (uint t = 0; t < batch_n; t++) {
        float g = g_decay_buf[t * num_v_heads + head_id];
        float beta = beta_buf[t * num_v_heads + head_id];

        uint state_base = head_id * value_dim * key_dim + vi * key_dim;
        uint k_base = t * total_key + kh * key_dim;
        uint v_base = t * total_value + head_id * value_dim;

        // Step 1+2: Decay + memory read
        float kv_mem = 0.0f;
        for (uint ki = 0; ki < key_dim; ki++) {
            float s = state[state_base + ki] * g;
            state[state_base + ki] = s;
            kv_mem += s * k[k_base + ki];
        }

        // Step 3+4: Delta update
        float delta = (v[v_base + vi] - kv_mem) * beta;
        for (uint ki = 0; ki < key_dim; ki++) {
            state[state_base + ki] += k[k_base + ki] * delta;
        }

        // Step 5: Output
        float out_val = 0.0f;
        uint q_base = t * total_key + kh * key_dim;
        for (uint ki = 0; ki < key_dim; ki++) {
            out_val += state[state_base + ki] * q[q_base + ki];
        }
        output[v_base + vi] = out_val;
    }
}


// ============================================================================
// Kernel P13: Gated RMS norm, batched
// ============================================================================

kernel void gated_rms_norm_batched(
    device const float *values,       // [N, num_v_heads * value_dim]
    device const float *z,            // [N, num_v_heads * value_dim]
    device const uint16_t *weight,    // [value_dim] bf16
    device float *output,             // [N, num_v_heads * value_dim]
    constant uint &value_dim,
    constant float &eps,
    constant uint &num_v_heads,
    constant uint &batch_n,
    uint tgid_flat [[threadgroup_position_in_grid]],
    uint tid       [[thread_position_in_threadgroup]]
) {
    uint head = tgid_flat % num_v_heads;
    uint t = tgid_flat / num_v_heads;
    if (head >= num_v_heads || t >= batch_n) return;

    uint stride = num_v_heads * value_dim;
    uint base = t * stride + head * value_dim;

    float val = (tid < value_dim) ? values[base + tid] : 0;
    uint simd_lane_l = tid % 32;
    uint simd_group_l = tid / 32;

    float sq = val * val;
    float simd_s = simd_sum(sq);
    threadgroup float partial_s[4];
    if (simd_lane_l == 0) partial_s[simd_group_l] = simd_s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = partial_s[0] + partial_s[1] + partial_s[2] + partial_s[3];
    float inv_rms = rsqrt(total / float(value_dim) + eps);

    if (tid < value_dim) {
        float normed = val * inv_rms;
        float zval = z[base + tid];
        float gate = zval / (1.0f + exp(-zval));  // SiLU
        float w = bf16_to_f32(weight[tid]);
        output[base + tid] = normed * gate * w;
    }
}
