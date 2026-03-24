#!/usr/bin/env python3
"""
gptq_requantize.py -- GPTQ error-compensated requantization of 4-bit experts to 2-bit.

Drop-in replacement for repack_experts_2bit.py that uses GPTQ (Frantar et al., 2022)
instead of naive round-to-nearest (RTN) quantization. GPTQ compensates each column's
quantization error using the inverse Hessian, distributing the error across remaining
columns in proportion to their second-order sensitivity.

Algorithm per expert weight matrix W [out_dim, in_dim]:
  1. Load 4-bit dequantized weights from packed expert data
  2. Load Hessian H [in_dim, in_dim] from calibration/layer_LL/expert_EEE.npy
  3. Add damping: H += 0.01 * diag(H).mean() * I
  4. Cholesky factorize H
  5. Blocked GPTQ (block_size=128):
     - For each block of 128 columns, compute local H_inv block
     - Quantize each column, compensate remaining columns in block
     - After block: propagate accumulated error to remaining columns
  6. Pack to 2-bit format (uint32 LSB-first, bf16 scale/bias per group of 64)

Requires per-expert Hessian matrices from Phase 1 calibration (build_hessian.py).
Experts without Hessians fall back to MSE-optimal clipping (Phase 0 method).

Usage:
    python gptq_requantize.py \\
        --input-dir packed_experts/ \\
        --hessian-dir calibration/ \\
        --output-dir packed_experts_gptq_2bit/ \\
        --num-layers 60 \\
        --num-experts 512 \\
        --hidden-dim 4096 \\
        --moe-intermediate 1024 \\
        --group-size 64 \\
        --block-size 128 \\
        --parallel 8 \\
        --fallback mse-clip
"""

import argparse
import multiprocessing
import os
import sys
import time
import json

import numpy as np

# Try scipy for Cholesky; fall back to numpy if unavailable
try:
    from scipy.linalg import cholesky, cho_solve
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================================
# 4-bit expert layout (matches repack_experts_2bit.py / infer.m)
# ============================================================================

EXPERT_SIZE_4BIT = 7_077_888
GROUP_SIZE = 64

GATE_W_OFF_4 = 0
GATE_W_SIZE_4 = 2_097_152   # [1024, 512] uint32
GATE_S_OFF_4 = 2_097_152
GATE_S_SIZE_4 = 131_072     # [1024, 64] uint16
GATE_B_OFF_4 = 2_228_224
GATE_B_SIZE_4 = 131_072

UP_W_OFF_4 = 2_359_296
UP_W_SIZE_4 = 2_097_152
UP_S_OFF_4 = 4_456_448
UP_S_SIZE_4 = 131_072
UP_B_OFF_4 = 4_587_520
UP_B_SIZE_4 = 131_072

DOWN_W_OFF_4 = 4_718_592
DOWN_W_SIZE_4 = 2_097_152   # [4096, 128] uint32
DOWN_S_OFF_4 = 6_815_744
DOWN_S_SIZE_4 = 131_072     # [4096, 16] uint16
DOWN_B_OFF_4 = 6_946_816
DOWN_B_SIZE_4 = 131_072

# Projection descriptors: (name, out_dim, in_dim, w_off, s_off, b_off)
PROJS_4BIT = [
    ("gate", 1024, 4096, GATE_W_OFF_4, GATE_S_OFF_4, GATE_B_OFF_4),
    ("up",   1024, 4096, UP_W_OFF_4,   UP_S_OFF_4,   UP_B_OFF_4),
    ("down", 4096, 1024, DOWN_W_OFF_4, DOWN_S_OFF_4,  DOWN_B_OFF_4),
]


# ============================================================================
# 2-bit expert layout
# ============================================================================

GATE_W_SIZE_2 = 1_048_576   # [1024, 256] uint32
UP_W_SIZE_2   = 1_048_576
DOWN_W_SIZE_2 = 1_048_576   # [4096, 64] uint32

GATE_W_OFF_2 = 0
GATE_S_OFF_2 = GATE_W_OFF_2 + GATE_W_SIZE_2
GATE_B_OFF_2 = GATE_S_OFF_2 + GATE_S_SIZE_4
UP_W_OFF_2   = GATE_B_OFF_2 + GATE_B_SIZE_4
UP_S_OFF_2   = UP_W_OFF_2   + UP_W_SIZE_2
UP_B_OFF_2   = UP_S_OFF_2   + UP_S_SIZE_4
DOWN_W_OFF_2 = UP_B_OFF_2   + UP_B_SIZE_4
DOWN_S_OFF_2 = DOWN_W_OFF_2 + DOWN_W_SIZE_2
DOWN_B_OFF_2 = DOWN_S_OFF_2 + DOWN_S_SIZE_4
EXPERT_SIZE_2BIT = DOWN_B_OFF_2 + DOWN_B_SIZE_4

assert EXPERT_SIZE_2BIT == 3_932_160

PROJS_2BIT_OFFSETS = {
    "gate": (GATE_W_OFF_2, GATE_S_OFF_2, GATE_B_OFF_2),
    "up":   (UP_W_OFF_2,   UP_S_OFF_2,   UP_B_OFF_2),
    "down": (DOWN_W_OFF_2, DOWN_S_OFF_2, DOWN_B_OFF_2),
}


# ============================================================================
# bf16 <-> f32 conversion helpers
# ============================================================================

def bf16_to_f32(bf16_u16):
    """Convert array of uint16 (bf16 bit pattern) to float32."""
    return (bf16_u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(f32):
    """Convert float32 array to uint16 (bf16 bit pattern). Truncates (no rounding)."""
    return (f32.view(np.uint32) >> 16).astype(np.uint16)


# ============================================================================
# Unpack / pack helpers
# ============================================================================

def unpack_4bit(packed):
    """Unpack 4-bit values from uint32 array. 8 values per uint32, LSB first."""
    shape = packed.shape
    flat = packed.ravel()
    n = flat.size
    out = np.empty(n * 8, dtype=np.uint8)
    for i in range(8):
        out[i::8] = ((flat >> (i * 4)) & 0xF).astype(np.uint8)
    return out.reshape(shape[:-1] + (shape[-1] * 8,))


def pack_2bit(vals):
    """Pack 2-bit values into uint32 array. 16 values per uint32, LSB first."""
    shape = vals.shape
    assert shape[-1] % 16 == 0
    n_packed = shape[-1] // 16
    flat = vals.reshape(-1, shape[-1])
    rows = flat.shape[0]
    out = np.zeros((rows, n_packed), dtype=np.uint32)
    for i in range(16):
        out |= flat[:, i::16].astype(np.uint32) << (i * 2)
    return out.reshape(shape[:-1] + (n_packed,))


# ============================================================================
# Dequantize a full projection from 4-bit packed format
# ============================================================================

def dequantize_4bit_projection(expert_blob, name, out_dim, in_dim, w_off, s_off, b_off):
    """
    Dequantize a 4-bit packed projection to float32.

    Returns: W [out_dim, in_dim] float32
    """
    packed_cols_4 = in_dim // 8
    num_groups = in_dim // GROUP_SIZE

    w_end = w_off + out_dim * packed_cols_4 * 4
    s_end = s_off + out_dim * num_groups * 2
    b_end = b_off + out_dim * num_groups * 2

    packed_4bit = np.frombuffer(
        expert_blob[w_off:w_end], dtype=np.uint32
    ).reshape(out_dim, packed_cols_4)
    scales_bf16 = np.frombuffer(
        expert_blob[s_off:s_end], dtype=np.uint16
    ).reshape(out_dim, num_groups)
    biases_bf16 = np.frombuffer(
        expert_blob[b_off:b_end], dtype=np.uint16
    ).reshape(out_dim, num_groups)

    # Unpack to [out_dim, in_dim] uint8
    vals_4bit = unpack_4bit(packed_4bit)
    assert vals_4bit.shape == (out_dim, in_dim)

    # Dequantize
    scales_f32 = bf16_to_f32(scales_bf16)
    biases_f32 = bf16_to_f32(biases_bf16)

    vals_grouped = vals_4bit.reshape(out_dim, num_groups, GROUP_SIZE).astype(np.float32)
    s = scales_f32[:, :, np.newaxis]
    b = biases_f32[:, :, np.newaxis]
    dequant = vals_grouped * s + b  # [out_dim, num_groups, GROUP_SIZE]

    return dequant.reshape(out_dim, in_dim)


# ============================================================================
# MSE-optimal 2-bit clipping (Phase 0 fallback)
# ============================================================================

def quantize_2bit_mse_optimal(W, group_size=64):
    """
    Quantize W [out_dim, in_dim] to 2-bit using MSE-optimal clipping per group.

    Returns: (quant_int [out_dim, in_dim] uint8 in [0,3],
              scales [out_dim, num_groups] float32,
              biases [out_dim, num_groups] float32,
              rmse float)
    """
    out_dim, in_dim = W.shape
    num_groups = in_dim // group_size

    W_grouped = W.reshape(out_dim, num_groups, group_size)

    f_min = W_grouped.min(axis=2, keepdims=True)
    f_max = W_grouped.max(axis=2, keepdims=True)
    f_mean = W_grouped.mean(axis=2, keepdims=True)

    best_mse = np.full_like(f_min, np.inf)
    best_s2 = (f_max - f_min) / 3.0
    best_b2 = f_min.copy()

    for r in np.linspace(0.7, 1.0, 20):
        c_min = f_mean - r * (f_mean - f_min)
        c_max = f_mean + r * (f_max - f_mean)
        s_try = (c_max - c_min) / 3.0
        s_safe = np.where(s_try == 0.0, 1.0, s_try)
        q_try = np.clip(np.round((W_grouped - c_min) / s_safe), 0, 3)
        recon_try = q_try * s_try + c_min
        mse_try = np.mean((W_grouped - recon_try) ** 2, axis=2, keepdims=True)
        improved = mse_try < best_mse
        best_mse = np.where(improved, mse_try, best_mse)
        best_s2 = np.where(improved, s_try, best_s2)
        best_b2 = np.where(improved, c_min, best_b2)

    s2 = best_s2
    b2 = best_b2
    degenerate = (s2 == 0.0)
    s2_safe = np.where(degenerate, 1.0, s2)

    vals_2bit_f = (W_grouped - b2) / s2_safe
    quant_int = np.clip(np.round(vals_2bit_f), 0, 3).astype(np.uint8)

    recon = quant_int.astype(np.float32) * s2 + b2
    rmse = float(np.sqrt(np.mean((W_grouped - recon) ** 2)))

    quant_flat = quant_int.reshape(out_dim, in_dim)
    scales = s2.squeeze(axis=2)
    biases = b2.squeeze(axis=2)

    return quant_flat, scales, biases, rmse


# ============================================================================
# GPTQ blocked quantization
# ============================================================================

def gptq_quantize_2bit(W, H, group_size=64, block_size=128):
    """
    GPTQ error-compensated quantization of W [out_dim, in_dim] to 2-bit.

    Uses the Hessian H [in_dim, in_dim] to compensate quantization errors
    across columns within blocks.

    Returns: (quant_int [out_dim, in_dim] uint8 in [0,3],
              scales [out_dim, num_groups] float32,
              biases [out_dim, num_groups] float32,
              rmse float)
    """
    out_dim, in_dim = W.shape
    num_groups = in_dim // group_size
    W = W.copy().astype(np.float32)

    # Step 1: Damp the Hessian
    damp = 0.01 * np.mean(np.diag(H))
    H_damped = H + damp * np.eye(in_dim, dtype=np.float32)

    # Step 2: Cholesky factorization of H
    # We need H_inv columns within each block. Use Cholesky of H directly.
    # GPTQ uses: H_inv = inv(H), then processes columns.
    # For blocked GPTQ: we need H_inv[block, block] for each block.
    try:
        if HAS_SCIPY:
            L = cholesky(H_damped, lower=True)
            # Compute full inverse via Cholesky
            I_mat = np.eye(in_dim, dtype=np.float32)
            H_inv = cho_solve((L, True), I_mat)
        else:
            L = np.linalg.cholesky(H_damped)
            H_inv = np.linalg.inv(H_damped)
    except (np.linalg.LinAlgError, Exception):
        # Cholesky failed — add more damping and retry
        damp2 = 0.1 * np.mean(np.diag(H))
        H_damped = H + damp2 * np.eye(in_dim, dtype=np.float32)
        try:
            H_inv = np.linalg.inv(H_damped)
        except np.linalg.LinAlgError:
            # Give up on GPTQ, return None to signal fallback
            return None

    # Step 3: Allocate output quantized values and scales/biases
    quant_int = np.zeros((out_dim, in_dim), dtype=np.uint8)
    scales = np.zeros((out_dim, num_groups), dtype=np.float32)
    biases = np.zeros((out_dim, num_groups), dtype=np.float32)

    # Step 4: Blocked GPTQ
    num_blocks = (in_dim + block_size - 1) // block_size

    for block_idx in range(num_blocks):
        col_start = block_idx * block_size
        col_end = min(col_start + block_size, in_dim)
        block_cols = col_end - col_start

        # Extract the local H_inv block for error compensation
        H_inv_block = H_inv[col_start:col_end, col_start:col_end]

        # Error accumulator for propagation to next block
        # err_accum[row, :] tracks cumulative error for propagation
        err_block = np.zeros((out_dim, block_cols), dtype=np.float32)

        for j_local in range(block_cols):
            j = col_start + j_local
            g = j // group_size  # which group this column belongs to

            # If this is the first column in its group, compute group params
            g_start = g * group_size
            if j == g_start:
                # Compute MSE-optimal clipping for this group
                g_end = min(g_start + group_size, in_dim)
                g_len = g_end - g_start

                group_vals = W[:, g_start:g_end]  # [out_dim, g_len]
                g_min = group_vals.min(axis=1, keepdims=True)
                g_max = group_vals.max(axis=1, keepdims=True)
                g_mean = group_vals.mean(axis=1, keepdims=True)

                best_mse_g = np.full((out_dim, 1), np.inf, dtype=np.float32)
                best_s_g = (g_max - g_min) / 3.0
                best_b_g = g_min.copy()

                for r in np.linspace(0.7, 1.0, 20):
                    c_min = g_mean - r * (g_mean - g_min)
                    c_max = g_mean + r * (g_max - g_mean)
                    s_try = (c_max - c_min) / 3.0
                    s_safe = np.where(s_try == 0.0, 1.0, s_try)
                    q_try = np.clip(np.round((group_vals - c_min) / s_safe), 0, 3)
                    recon_try = q_try * s_try + c_min
                    mse_try = np.mean((group_vals - recon_try) ** 2, axis=1, keepdims=True)
                    improved = mse_try < best_mse_g
                    best_mse_g = np.where(improved, mse_try, best_mse_g)
                    best_s_g = np.where(improved, s_try, best_s_g)
                    best_b_g = np.where(improved, c_min, best_b_g)

                # Store group params
                cur_scale = best_s_g.squeeze(axis=1)  # [out_dim]
                cur_bias = best_b_g.squeeze(axis=1)    # [out_dim]
                scales[:, g] = cur_scale
                biases[:, g] = cur_bias
            else:
                cur_scale = scales[:, g]
                cur_bias = biases[:, g]

            # Quantize column j for all rows
            w_col = W[:, j]  # [out_dim]
            s_safe = np.where(cur_scale == 0.0, 1.0, cur_scale)
            q_col = np.clip(np.round((w_col - cur_bias) / s_safe), 0, 3).astype(np.uint8)
            quant_int[:, j] = q_col

            # Dequantized value
            w_hat = q_col.astype(np.float32) * cur_scale + cur_bias

            # Quantization error
            delta = w_col - w_hat  # [out_dim]

            # Store error for this column
            err_block[:, j_local] = delta

            # Compensate remaining columns in this block
            if j_local < block_cols - 1:
                h_inv_jj = H_inv_block[j_local, j_local]
                if abs(h_inv_jj) > 1e-10:
                    # delta[row] * H_inv[j, j+1:block_end] / H_inv[j, j]
                    h_inv_row = H_inv_block[j_local, j_local + 1:block_cols]  # [remaining_cols]
                    # W[:, j+1:col_end] -= outer(delta, h_inv_row / h_inv_jj)
                    compensation = np.outer(delta, h_inv_row / h_inv_jj)
                    W[:, j + 1:col_end] -= compensation

        # After processing the block: propagate accumulated error to remaining columns
        if col_end < in_dim:
            # Propagate: W[:, col_end:] -= err_block @ H_inv[block, col_end:]
            H_inv_cross = H_inv[col_start:col_end, col_end:]  # [block_cols, remaining]
            W[:, col_end:] -= err_block @ H_inv_cross

    # Compute RMSE vs original (before GPTQ modified W)
    # We need the original W for this — but GPTQ modifies W in-place.
    # Instead, reconstruct and compare to the pre-GPTQ values.
    # The caller computes RMSE separately against the original dequantized weights.
    recon = np.zeros_like(W)
    for g in range(num_groups):
        g_start = g * group_size
        g_end = min(g_start + group_size, in_dim)
        s = scales[:, g:g+1]
        b = biases[:, g:g+1]
        recon[:, g_start:g_end] = quant_int[:, g_start:g_end].astype(np.float32) * s + b

    # RMSE will be computed by caller against original weights
    # Return 0.0 as placeholder; caller computes actual RMSE
    return quant_int, scales, biases, 0.0


# ============================================================================
# Process one expert: read 4-bit blob, apply GPTQ or fallback
# ============================================================================

def process_expert(args_tuple):
    """
    Process a single (layer, expert) pair. Designed for multiprocessing.Pool.

    Args tuple: (layer_idx, expert_idx, input_dir, hessian_dir, output_expert_size,
                 group_size, block_size, fallback_mode)

    Returns: (layer_idx, expert_idx, expert_2bit_bytes, rtn_rmse, gptq_rmse, used_gptq)
    """
    (layer_idx, expert_idx, input_dir, hessian_dir,
     group_size, block_size, fallback_mode) = args_tuple

    # Read the expert blob from the layer file
    input_path = os.path.join(input_dir, f"layer_{layer_idx:02d}.bin")
    with open(input_path, "rb") as f:
        f.seek(expert_idx * EXPERT_SIZE_4BIT)
        expert_blob = f.read(EXPERT_SIZE_4BIT)

    if len(expert_blob) != EXPERT_SIZE_4BIT:
        raise ValueError(f"Short read for layer {layer_idx} expert {expert_idx}")

    # Dequantize all three projections from 4-bit
    projs_original = {}
    for name, out_dim, in_dim, w_off, s_off, b_off in PROJS_4BIT:
        W = dequantize_4bit_projection(expert_blob, name, out_dim, in_dim, w_off, s_off, b_off)
        projs_original[name] = W

    # Load Hessian if available
    hessian_path = os.path.join(hessian_dir, f"layer_{layer_idx:02d}", f"expert_{expert_idx:03d}.npy")
    has_hessian = os.path.isfile(hessian_path)

    H = None
    if has_hessian:
        H = np.load(hessian_path).astype(np.float32)

    # Process each projection
    output = bytearray(EXPERT_SIZE_2BIT)
    proj_rtn_rmse = {}
    proj_gptq_rmse = {}
    used_gptq = False

    for name, out_dim, in_dim, w_off, s_off, b_off in PROJS_4BIT:
        W_orig = projs_original[name]  # [out_dim, in_dim]

        # Determine if we can use GPTQ for this projection
        use_gptq_for_proj = False
        H_proj = None

        if H is not None:
            # The Hessian from calibration is H = X^T X where X are the inputs
            # to this expert (h_post vectors, dimension = hidden_dim = 4096).
            # gate_proj and up_proj: input is h_post [4096], so H is [4096, 4096] -- matches
            # down_proj: input is intermediate after SwiGLU [1024], H would need to be [1024, 1024]
            # But our calibration only collected h_post, so H is [4096, 4096]
            if name in ("gate", "up") and H.shape == (in_dim, in_dim):
                H_proj = H
                use_gptq_for_proj = True
            elif name == "down" and H.shape == (in_dim, in_dim):
                # If intermediate Hessian is available (would be [1024, 1024])
                H_proj = H
                use_gptq_for_proj = True
            # else: H shape doesn't match, fall back to MSE-clip

        # First: compute RTN baseline RMSE (MSE-optimal clipping)
        _, _, _, rtn_rmse = quantize_2bit_mse_optimal(W_orig, group_size)
        proj_rtn_rmse[name] = rtn_rmse

        # Try GPTQ
        gptq_result = None
        if use_gptq_for_proj:
            gptq_result = gptq_quantize_2bit(W_orig, H_proj, group_size, block_size)

        if gptq_result is not None:
            quant_int, scales_f32, biases_f32, _ = gptq_result
            # Compute actual GPTQ RMSE against original weights
            num_groups = in_dim // group_size
            recon = np.zeros_like(W_orig)
            for g in range(num_groups):
                gs = g * group_size
                ge = min(gs + group_size, in_dim)
                recon[:, gs:ge] = (quant_int[:, gs:ge].astype(np.float32)
                                   * scales_f32[:, g:g+1] + biases_f32[:, g:g+1])
            gptq_rmse = float(np.sqrt(np.mean((W_orig - recon) ** 2)))
            proj_gptq_rmse[name] = gptq_rmse

            # Use GPTQ result if it's better than RTN
            if gptq_rmse < rtn_rmse:
                used_gptq = True
            else:
                # GPTQ worse than RTN (can happen with poor Hessians); fall back
                quant_int, scales_f32, biases_f32, _ = quantize_2bit_mse_optimal(W_orig, group_size)
                proj_gptq_rmse[name] = rtn_rmse
        else:
            # Fallback: MSE-optimal clipping
            quant_int, scales_f32, biases_f32, _ = quantize_2bit_mse_optimal(W_orig, group_size)
            proj_gptq_rmse[name] = rtn_rmse

        # Pack 2-bit and write into output blob
        packed_2bit = pack_2bit(quant_int.reshape(out_dim, in_dim))
        new_scales_bf16 = f32_to_bf16(scales_f32.astype(np.float32))
        new_biases_bf16 = f32_to_bf16(biases_f32.astype(np.float32))

        w_off_2, s_off_2, b_off_2 = PROJS_2BIT_OFFSETS[name]
        w_data = packed_2bit.tobytes()
        s_data = new_scales_bf16.tobytes()
        b_data = new_biases_bf16.tobytes()

        output[w_off_2:w_off_2 + len(w_data)] = w_data
        output[s_off_2:s_off_2 + len(s_data)] = s_data
        output[b_off_2:b_off_2 + len(b_data)] = b_data

    return (layer_idx, expert_idx, bytes(output),
            proj_rtn_rmse, proj_gptq_rmse, used_gptq)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GPTQ error-compensated requantization of 4-bit experts to 2-bit")
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing layer_XX.bin 4-bit packed expert files")
    parser.add_argument("--hessian-dir", required=True,
                        help="Directory containing calibration Hessians (layer_LL/expert_EEE.npy)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for 2-bit packed expert files")
    parser.add_argument("--num-layers", type=int, default=60,
                        help="Number of layers (default: 60)")
    parser.add_argument("--num-experts", type=int, default=512,
                        help="Number of experts per layer (default: 512)")
    parser.add_argument("--hidden-dim", type=int, default=4096,
                        help="Model hidden dimension (default: 4096)")
    parser.add_argument("--moe-intermediate", type=int, default=1024,
                        help="MoE intermediate dimension (default: 1024)")
    parser.add_argument("--group-size", type=int, default=64,
                        help="Quantization group size (default: 64)")
    parser.add_argument("--block-size", type=int, default=128,
                        help="GPTQ block size (default: 128)")
    parser.add_argument("--parallel", type=int, default=8,
                        help="Number of parallel workers (default: 8)")
    parser.add_argument("--fallback", choices=["mse-clip", "abort"], default="mse-clip",
                        help="Fallback for uncalibrated experts (default: mse-clip)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Process only this layer (for testing)")
    args = parser.parse_args()

    input_dir = args.input_dir
    hessian_dir = args.hessian_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Determine layers to process
    if args.layer is not None:
        layers = [args.layer]
    else:
        layers = []
        for i in range(args.num_layers):
            layer_path = os.path.join(input_dir, f"layer_{i:02d}.bin")
            if os.path.isfile(layer_path):
                layers.append(i)

    if not layers:
        print(f"ERROR: No layer_XX.bin files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"GPTQ 2-bit Requantization")
    print(f"========================")
    print(f"Input:         {input_dir}")
    print(f"Hessians:      {hessian_dir}")
    print(f"Output:        {output_dir}")
    print(f"Layers:        {layers[0]}-{layers[-1]} ({len(layers)} layers)")
    print(f"Experts:       {args.num_experts}")
    print(f"Hidden dim:    {args.hidden_dim}")
    print(f"MoE inter:     {args.moe_intermediate}")
    print(f"Group size:    {args.group_size}")
    print(f"Block size:    {args.block_size}")
    print(f"Parallel:      {args.parallel}")
    print(f"Fallback:      {args.fallback}")
    print(f"4-bit size:    {EXPERT_SIZE_4BIT:,} bytes/expert")
    print(f"2-bit size:    {EXPERT_SIZE_2BIT:,} bytes/expert")
    print(f"Savings:       {1 - EXPERT_SIZE_2BIT / EXPERT_SIZE_4BIT:.1%}")
    print(f"scipy:         {'yes' if HAS_SCIPY else 'no (using numpy fallback)'}")
    print()

    total_t0 = time.time()
    total_experts = 0
    total_gptq = 0
    total_fallback = 0
    global_rtn_rmse = {"gate": 0.0, "up": 0.0, "down": 0.0}
    global_gptq_rmse = {"gate": 0.0, "up": 0.0, "down": 0.0}

    for layer_idx in layers:
        layer_t0 = time.time()

        input_path = os.path.join(input_dir, f"layer_{layer_idx:02d}.bin")
        output_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.bin")

        actual_size = os.path.getsize(input_path)
        expected_size = args.num_experts * EXPERT_SIZE_4BIT
        if actual_size != expected_size:
            if actual_size % EXPERT_SIZE_4BIT != 0:
                print(f"ERROR: layer_{layer_idx:02d}.bin size {actual_size} not a multiple "
                      f"of {EXPERT_SIZE_4BIT}, skipping", file=sys.stderr)
                continue
            num_experts = actual_size // EXPERT_SIZE_4BIT
            print(f"WARNING: layer_{layer_idx:02d}.bin has {num_experts} experts "
                  f"(expected {args.num_experts})")
        else:
            num_experts = args.num_experts

        # Count available Hessians for this layer
        hessian_layer_dir = os.path.join(hessian_dir, f"layer_{layer_idx:02d}")
        hessian_count = 0
        if os.path.isdir(hessian_layer_dir):
            for f in os.listdir(hessian_layer_dir):
                if f.startswith("expert_") and f.endswith(".npy"):
                    hessian_count += 1

        print(f"=== Layer {layer_idx:02d} ({num_experts} experts, "
              f"{hessian_count} Hessians available) ===")

        # Build work items for this layer
        work_items = []
        for eidx in range(num_experts):
            work_items.append((
                layer_idx, eidx, input_dir, hessian_dir,
                args.group_size, args.block_size, args.fallback
            ))

        # Process with multiprocessing pool
        layer_gptq = 0
        layer_fallback = 0
        layer_rtn_rmse = {"gate": 0.0, "up": 0.0, "down": 0.0}
        layer_gptq_rmse = {"gate": 0.0, "up": 0.0, "down": 0.0}

        # Collect all results, then write sequentially
        results = [None] * num_experts

        # Use min of parallel and num_experts to avoid idle workers
        num_workers = min(args.parallel, num_experts)

        if num_workers > 1:
            with multiprocessing.Pool(num_workers) as pool:
                for result in pool.imap_unordered(process_expert, work_items):
                    _, eidx, expert_2bit, rtn_rmses, gptq_rmses, used_gptq = result
                    results[eidx] = expert_2bit

                    for p in ("gate", "up", "down"):
                        layer_rtn_rmse[p] += rtn_rmses[p]
                        layer_gptq_rmse[p] += gptq_rmses[p]

                    if used_gptq:
                        layer_gptq += 1
                    else:
                        layer_fallback += 1

                    done = layer_gptq + layer_fallback
                    if done % 32 == 0 or done == num_experts:
                        elapsed = time.time() - layer_t0
                        rate = done / elapsed if elapsed > 0 else 0
                        eta = (num_experts - done) / rate if rate > 0 else 0
                        print(f"  [{done:3d}/{num_experts}] "
                              f"{elapsed:.1f}s, {rate:.1f} exp/s, ETA {eta:.0f}s "
                              f"(GPTQ: {layer_gptq}, fallback: {layer_fallback})")
        else:
            for item in work_items:
                result = process_expert(item)
                _, eidx, expert_2bit, rtn_rmses, gptq_rmses, used_gptq = result
                results[eidx] = expert_2bit

                for p in ("gate", "up", "down"):
                    layer_rtn_rmse[p] += rtn_rmses[p]
                    layer_gptq_rmse[p] += gptq_rmses[p]

                if used_gptq:
                    layer_gptq += 1
                else:
                    layer_fallback += 1

                done = layer_gptq + layer_fallback
                if done % 32 == 0 or done == num_experts:
                    elapsed = time.time() - layer_t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (num_experts - done) / rate if rate > 0 else 0
                    print(f"  [{done:3d}/{num_experts}] "
                          f"{elapsed:.1f}s, {rate:.1f} exp/s, ETA {eta:.0f}s "
                          f"(GPTQ: {layer_gptq}, fallback: {layer_fallback})")

        # Write all experts sequentially to output file
        with open(output_path, "wb") as fout:
            for eidx in range(num_experts):
                assert results[eidx] is not None, f"Missing result for expert {eidx}"
                assert len(results[eidx]) == EXPERT_SIZE_2BIT
                fout.write(results[eidx])

        layer_elapsed = time.time() - layer_t0
        total_experts += num_experts
        total_gptq += layer_gptq
        total_fallback += layer_fallback

        # Accumulate global stats
        for p in ("gate", "up", "down"):
            global_rtn_rmse[p] += layer_rtn_rmse[p]
            global_gptq_rmse[p] += layer_gptq_rmse[p]

        # Per-layer stats
        avg_rtn = {p: layer_rtn_rmse[p] / num_experts for p in layer_rtn_rmse}
        avg_gptq = {p: layer_gptq_rmse[p] / num_experts for p in layer_gptq_rmse}
        print(f"\n  Layer {layer_idx:02d} done in {layer_elapsed:.1f}s "
              f"({num_experts / layer_elapsed:.1f} experts/s)")
        print(f"  GPTQ: {layer_gptq}/{num_experts}, fallback: {layer_fallback}/{num_experts}")
        print(f"  Avg RTN  RMSE:  gate={avg_rtn['gate']:.6f}  "
              f"up={avg_rtn['up']:.6f}  down={avg_rtn['down']:.6f}")
        print(f"  Avg GPTQ RMSE:  gate={avg_gptq['gate']:.6f}  "
              f"up={avg_gptq['up']:.6f}  down={avg_gptq['down']:.6f}")
        for p in ("gate", "up", "down"):
            if avg_rtn[p] > 0:
                improvement = (1 - avg_gptq[p] / avg_rtn[p]) * 100
                print(f"  {p:4s} improvement: {improvement:+.1f}%")

        out_size = os.path.getsize(output_path)
        print(f"  Output: {output_path} ({out_size / 1e9:.2f} GB)")
        print()

    # Summary
    total_elapsed = time.time() - total_t0
    print(f"{'='*60}")
    print(f"GPTQ Requantization Summary")
    print(f"{'='*60}")
    print(f"Total experts:  {total_experts}")
    print(f"GPTQ applied:   {total_gptq} ({100*total_gptq/max(total_experts,1):.1f}%)")
    print(f"Fallback (RTN): {total_fallback} ({100*total_fallback/max(total_experts,1):.1f}%)")
    print(f"Total time:     {total_elapsed:.1f}s")
    if total_experts > 0:
        avg_rtn = {p: global_rtn_rmse[p] / total_experts for p in global_rtn_rmse}
        avg_gptq = {p: global_gptq_rmse[p] / total_experts for p in global_gptq_rmse}
        print(f"\nAvg RTN  RMSE:  gate={avg_rtn['gate']:.6f}  "
              f"up={avg_rtn['up']:.6f}  down={avg_rtn['down']:.6f}")
        print(f"Avg GPTQ RMSE:  gate={avg_gptq['gate']:.6f}  "
              f"up={avg_gptq['up']:.6f}  down={avg_gptq['down']:.6f}")
        for p in ("gate", "up", "down"):
            if avg_rtn[p] > 0:
                improvement = (1 - avg_gptq[p] / avg_rtn[p]) * 100
                print(f"  {p:4s} improvement: {improvement:+.1f}%")


if __name__ == "__main__":
    main()
