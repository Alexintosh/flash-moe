#!/usr/bin/env python3
"""
sensitivity_analysis.py -- JANG sensitivity-based tiered bit assignment.

Computes per-expert sensitivity scores using:
    sensitivity[layer][expert] = freq * quant_error * layer_weight

And assigns 4-bit to the highest-sensitivity experts until a target model
size budget is reached. Remaining experts get 2-bit.

Sensitivity inputs:
  - freq: activation frequency per (layer, expert), from --freq-file JSON
  - quant_error: RMSE of 2-bit vs 4-bit reconstruction per expert
  - layer_weight: 1.5 for first/last 3 layers, 1.0 otherwise

Output: hot_experts.json with {layer_idx: [list of 4-bit expert indices]}

Usage:
    python sensitivity_analysis.py \\
        --freq-file freq_data.json \\
        --packed-dir packed_experts/ \\
        --num-layers 60 \\
        --num-experts 512 \\
        --hidden-dim 4096 \\
        --moe-intermediate 1024 \\
        --target-gb 150 \\
        --output hot_experts.json
"""

import argparse
import json
import multiprocessing
import os
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np


# ============================================================================
# Constants
# ============================================================================

GROUP_SIZE = 64

EXPERT_SIZE_4BIT = 7_077_888
EXPERT_SIZE_2BIT = 3_932_160


# ============================================================================
# bf16 <-> f32 helpers
# ============================================================================

def bf16_to_f32(bf16_u16):
    """Convert uint16 (bf16 bit pattern) array to float32."""
    return (bf16_u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(f32):
    """Convert float32 array to uint16 (bf16 bit pattern). Truncates."""
    return (f32.view(np.uint32) >> 16).astype(np.uint16)


# ============================================================================
# 4-bit unpacking
# ============================================================================

def unpack_4bit(packed):
    """Extract 8 x 4-bit values from each uint32 (LSB first).

    Input:  [..., N] uint32
    Output: [..., N*8] uint8, values in [0, 15]
    """
    shape = packed.shape
    flat = packed.ravel()
    out = np.empty(flat.size * 8, dtype=np.uint8)
    for i in range(8):
        out[i::8] = ((flat >> (i * 4)) & 0xF).astype(np.uint8)
    return out.reshape(shape[:-1] + (shape[-1] * 8,))


# ============================================================================
# 2-bit packing
# ============================================================================

def pack_2bit(vals):
    """Pack 16 x 2-bit values into each uint32 (LSB first).

    Input:  [..., M] uint8, values in [0, 3], M divisible by 16
    Output: [..., M/16] uint32
    """
    shape = vals.shape
    assert shape[-1] % 16 == 0
    n_packed = shape[-1] // 16
    flat = vals.reshape(-1, shape[-1])
    out = np.zeros((flat.shape[0], n_packed), dtype=np.uint32)
    for i in range(16):
        out |= flat[:, i::16].astype(np.uint32) << (i * 2)
    return out.reshape(shape[:-1] + (n_packed,))


# ============================================================================
# Expert layout offsets
# ============================================================================

def compute_offsets(moe_intermediate, hidden_dim, group_size, bits):
    """Compute byte offsets within an expert blob for a given bit width.

    Returns (offsets_dict, total_size).
    offsets_dict has keys: gate_w, gate_s, gate_b, up_w, up_s, up_b,
                           down_w, down_s, down_b
    """
    vals_per_u32 = 32 // bits
    mid, hid, gs = moe_intermediate, hidden_dim, group_size

    def proj_sizes(out_d, in_d):
        w = out_d * ((in_d + vals_per_u32 - 1) // vals_per_u32) * 4
        s = out_d * ((in_d + gs - 1) // gs) * 2
        return w, s, s  # weight, scales, biases (scales and biases same size)

    gw, gs_sz, gb = proj_sizes(mid, hid)
    uw, us, ub = proj_sizes(mid, hid)
    dw, ds, db = proj_sizes(hid, mid)

    offsets = {}
    off = 0
    for name, w, s, b in [("gate", gw, gs_sz, gb),
                           ("up", uw, us, ub),
                           ("down", dw, ds, db)]:
        offsets[f"{name}_w"] = off; off += w
        offsets[f"{name}_s"] = off; off += s
        offsets[f"{name}_b"] = off; off += b

    return offsets, off


# ============================================================================
# Compute RMSE for a single expert (4-bit -> 2-bit requant)
# ============================================================================

def compute_expert_rmse(expert_blob, off4, moe_intermediate, hidden_dim,
                        group_size=64):
    """Dequantize 4-bit, requantize to 2-bit with MSE-optimal clipping,
    compute RMSE of reconstruction vs 4-bit dequantized values.

    Returns: float RMSE
    """
    projs = [
        ("gate", moe_intermediate, hidden_dim),
        ("up", moe_intermediate, hidden_dim),
        ("down", hidden_dim, moe_intermediate),
    ]

    total_sse = 0.0
    total_count = 0

    for name, out_dim, in_dim in projs:
        num_groups = in_dim // group_size
        packed_cols_4 = in_dim // 8

        # Read 4-bit components
        w_start = off4[f"{name}_w"]
        s_start = off4[f"{name}_s"]
        b_start = off4[f"{name}_b"]

        w_bytes = out_dim * packed_cols_4 * 4
        s_bytes = out_dim * num_groups * 2

        packed_4bit = np.frombuffer(
            expert_blob[w_start:w_start + w_bytes], dtype=np.uint32
        ).reshape(out_dim, packed_cols_4)
        scales = np.frombuffer(
            expert_blob[s_start:s_start + s_bytes], dtype=np.uint16
        ).reshape(out_dim, num_groups)
        biases = np.frombuffer(
            expert_blob[b_start:b_start + s_bytes], dtype=np.uint16
        ).reshape(out_dim, num_groups)

        # Dequantize 4-bit -> float32
        vals_4bit = unpack_4bit(packed_4bit).reshape(
            out_dim, num_groups, group_size).astype(np.float32)
        s_f32 = bf16_to_f32(scales)[:, :, np.newaxis]
        b_f32 = bf16_to_f32(biases)[:, :, np.newaxis]
        dequant = vals_4bit * s_f32 + b_f32

        # MSE-optimal 2-bit requantization
        f_min = dequant.min(axis=2, keepdims=True)
        f_max = dequant.max(axis=2, keepdims=True)
        f_mean = dequant.mean(axis=2, keepdims=True)

        best_mse = np.full_like(f_min, np.inf)
        best_s2 = (f_max - f_min) / 3.0
        best_b2 = f_min.copy()

        for r in np.linspace(0.7, 1.0, 20):
            c_min = f_mean - r * (f_mean - f_min)
            c_max = f_mean + r * (f_max - f_mean)
            s_try = (c_max - c_min) / 3.0
            s_safe = np.where(s_try == 0.0, 1.0, s_try)
            q_try = np.clip(np.round((dequant - c_min) / s_safe), 0, 3)
            recon_try = q_try * s_try + c_min
            mse_try = np.mean((dequant - recon_try) ** 2, axis=2,
                              keepdims=True)
            improved = mse_try < best_mse
            best_mse = np.where(improved, mse_try, best_mse)
            best_s2 = np.where(improved, s_try, best_s2)
            best_b2 = np.where(improved, c_min, best_b2)

        s2 = best_s2
        b2 = best_b2
        s2_safe = np.where(s2 == 0.0, 1.0, s2)
        vals_2bit = np.clip(np.round((dequant - b2) / s2_safe), 0, 3)
        recon = vals_2bit * s2 + b2

        sse = np.sum((dequant - recon) ** 2)
        count = out_dim * in_dim

        total_sse += float(sse)
        total_count += count

    return float(np.sqrt(total_sse / total_count)) if total_count > 0 else 0.0


# ============================================================================
# Worker function for multiprocessing
# ============================================================================

def _worker_compute_rmse(args_tuple, packed_dir, off4, size4,
                         moe_intermediate, hidden_dim, group_size):
    """Worker that computes RMSE for a single (layer, expert) pair.

    args_tuple: (layer_idx, expert_idx)
    Returns: (layer_idx, expert_idx, rmse)
    """
    layer_idx, expert_idx = args_tuple
    layer_path = os.path.join(packed_dir, f"layer_{layer_idx:02d}.bin")

    if not os.path.exists(layer_path):
        return (layer_idx, expert_idx, 0.0)

    # Read just this expert's blob
    with open(layer_path, "rb") as f:
        f.seek(expert_idx * size4)
        expert_blob = f.read(size4)

    if len(expert_blob) != size4:
        return (layer_idx, expert_idx, 0.0)

    rmse = compute_expert_rmse(expert_blob, off4, moe_intermediate,
                               hidden_dim, group_size)
    return (layer_idx, expert_idx, rmse)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="JANG sensitivity-based tiered bit assignment for MoE experts."
    )
    parser.add_argument("--freq-file", default=None,
                        help="JSON with activation counts: {\"layer_idx\": {\"expert_idx\": count}}. "
                             "If not provided, uses uniform frequency.")
    parser.add_argument("--packed-dir", required=True,
                        help="Directory with 4-bit packed expert files (layer_XX.bin)")
    parser.add_argument("--num-layers", type=int, default=60,
                        help="Number of transformer layers (default: 60)")
    parser.add_argument("--num-experts", type=int, default=512,
                        help="Number of experts per layer (default: 512)")
    parser.add_argument("--hidden-dim", type=int, default=4096,
                        help="Model hidden dimension (default: 4096)")
    parser.add_argument("--moe-intermediate", type=int, default=1024,
                        help="MoE intermediate size (default: 1024)")
    parser.add_argument("--group-size", type=int, default=64,
                        help="Quantization group size (default: 64)")
    parser.add_argument("--target-gb", type=float, required=True,
                        help="Target total model size in GB for tiered packing")
    parser.add_argument("--output", default="hot_experts.json",
                        help="Output JSON path (default: hot_experts.json)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count)")
    args = parser.parse_args()

    packed_dir = args.packed_dir
    num_layers = args.num_layers
    num_experts = args.num_experts
    hidden_dim = args.hidden_dim
    moe_intermediate = args.moe_intermediate
    group_size = args.group_size
    target_bytes = args.target_gb * 1024 * 1024 * 1024

    # Compute expert sizes
    off4, size4 = compute_offsets(moe_intermediate, hidden_dim, group_size, 4)
    off2, size2 = compute_offsets(moe_intermediate, hidden_dim, group_size, 2)

    assert size4 == EXPERT_SIZE_4BIT, \
        f"Computed 4-bit size {size4} != expected {EXPERT_SIZE_4BIT}"
    assert size2 == EXPERT_SIZE_2BIT, \
        f"Computed 2-bit size {size2} != expected {EXPERT_SIZE_2BIT}"

    print(f"Configuration:")
    print(f"  Packed dir:       {packed_dir}")
    print(f"  Layers:           {num_layers}")
    print(f"  Experts/layer:    {num_experts}")
    print(f"  Hidden dim:       {hidden_dim}")
    print(f"  MoE intermediate: {moe_intermediate}")
    print(f"  4-bit expert:     {size4:,} bytes ({size4/1024/1024:.2f} MB)")
    print(f"  2-bit expert:     {size2:,} bytes ({size2/1024/1024:.2f} MB)")
    print(f"  Target size:      {args.target_gb:.1f} GB")
    print()

    # ---- Load frequency data ----
    freq = {}
    if args.freq_file:
        print(f"Loading frequency data from {args.freq_file}...")
        with open(args.freq_file, "r") as f:
            freq_raw = json.load(f)
        # Normalize: freq_raw can be {str(layer): {str(expert): count}}
        for layer_str, expert_counts in freq_raw.items():
            layer_idx = int(layer_str)
            freq[layer_idx] = {}
            for expert_str, count in expert_counts.items():
                freq[layer_idx][int(expert_str)] = float(count)
        print(f"  Loaded frequency data for {len(freq)} layers")
    else:
        print("No --freq-file provided. Using uniform frequency (1.0 for all).")
        for l in range(num_layers):
            freq[l] = {e: 1.0 for e in range(num_experts)}
    print()

    # ---- Determine which layer files exist ----
    existing_layers = []
    for l in range(num_layers):
        layer_path = os.path.join(packed_dir, f"layer_{l:02d}.bin")
        if os.path.exists(layer_path):
            existing_layers.append(l)
    print(f"Found {len(existing_layers)} layer files in {packed_dir}")
    if not existing_layers:
        print("ERROR: No layer files found.", file=sys.stderr)
        sys.exit(1)
    print()

    # ---- Compute quantization error for all experts (multiprocessing) ----
    total_experts = len(existing_layers) * num_experts
    print(f"Computing quantization error for {total_experts} experts...")
    print(f"  Estimated time: ~{total_experts * 0.1 / 60:.0f} minutes")
    print()

    # Build work items
    work_items = []
    for l in existing_layers:
        for e in range(num_experts):
            work_items.append((l, e))

    # Create worker with bound parameters
    worker_fn = partial(
        _worker_compute_rmse,
        packed_dir=packed_dir,
        off4=off4,
        size4=size4,
        moe_intermediate=moe_intermediate,
        hidden_dim=hidden_dim,
        group_size=group_size,
    )

    # Run with multiprocessing
    n_workers = args.workers or min(multiprocessing.cpu_count(), 12)
    print(f"Using {n_workers} worker processes")

    quant_error = {}  # (layer, expert) -> rmse
    t0 = time.time()
    completed = 0

    with multiprocessing.Pool(n_workers) as pool:
        for result in pool.imap_unordered(worker_fn, work_items,
                                          chunksize=16):
            layer_idx, expert_idx, rmse = result
            quant_error[(layer_idx, expert_idx)] = rmse
            completed += 1

            if completed % 100 == 0 or completed == total_experts:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total_experts - completed) / rate if rate > 0 else 0
                print(f"  [{completed:6d}/{total_experts}] "
                      f"{elapsed:.0f}s elapsed, {rate:.1f} experts/s, "
                      f"ETA {eta:.0f}s")

    elapsed_total = time.time() - t0
    print(f"\nQuantization error computation done in {elapsed_total:.0f}s")
    print()

    # ---- Compute sensitivity scores ----
    print("Computing sensitivity scores...")

    # Layer weights: 1.5 for first/last 3 layers, 1.0 otherwise
    layer_weight = {}
    for l in range(num_layers):
        if l < 3 or l >= num_layers - 3:
            layer_weight[l] = 1.5
        else:
            layer_weight[l] = 1.0

    # Compute sensitivity for each (layer, expert)
    sensitivity_list = []  # [(layer, expert, sensitivity)]
    for l in existing_layers:
        for e in range(num_experts):
            f = freq.get(l, {}).get(e, 1.0)
            qe = quant_error.get((l, e), 0.0)
            lw = layer_weight.get(l, 1.0)
            sens = f * qe * lw
            sensitivity_list.append((l, e, sens))

    # Sort by sensitivity descending
    sensitivity_list.sort(key=lambda x: x[2], reverse=True)

    # Print top/bottom sensitivity stats
    print(f"  Total (layer, expert) pairs: {len(sensitivity_list)}")
    if sensitivity_list:
        print(f"  Max sensitivity: {sensitivity_list[0][2]:.6f} "
              f"(layer {sensitivity_list[0][0]}, expert {sensitivity_list[0][1]})")
        print(f"  Min sensitivity: {sensitivity_list[-1][2]:.6f} "
              f"(layer {sensitivity_list[-1][0]}, expert {sensitivity_list[-1][1]})")
        # Mean
        mean_sens = sum(s for _, _, s in sensitivity_list) / len(sensitivity_list)
        print(f"  Mean sensitivity: {mean_sens:.6f}")
    print()

    # ---- Assign bits based on target size ----
    print(f"Assigning bits to meet target of {args.target_gb:.1f} GB...")

    # Start with all experts at 2-bit (minimum size)
    all_2bit_size = len(sensitivity_list) * size2
    size_increase_per_upgrade = size4 - size2  # bytes gained by upgrading to 4-bit

    print(f"  All 2-bit size: {all_2bit_size / 1024**3:.2f} GB")
    print(f"  All 4-bit size: {len(sensitivity_list) * size4 / 1024**3:.2f} GB")
    print(f"  Per-expert upgrade cost: {size_increase_per_upgrade / 1024**2:.2f} MB")

    if all_2bit_size > target_bytes:
        print(f"WARNING: Even all-2bit ({all_2bit_size / 1024**3:.2f} GB) "
              f"exceeds target ({args.target_gb:.1f} GB).")
        print("  Assigning all experts as 2-bit.")
        hot_experts = {str(l): [] for l in existing_layers}
        num_hot = 0
    else:
        # Budget for upgrades
        upgrade_budget = target_bytes - all_2bit_size
        max_upgrades = int(upgrade_budget / size_increase_per_upgrade)

        print(f"  Upgrade budget: {upgrade_budget / 1024**3:.2f} GB "
              f"({max_upgrades} experts can be 4-bit)")

        # Assign 4-bit to top-sensitivity experts
        hot_experts = {str(l): [] for l in existing_layers}
        num_hot = 0
        for layer_idx, expert_idx, sens in sensitivity_list:
            if num_hot >= max_upgrades:
                break
            hot_experts[str(layer_idx)].append(expert_idx)
            num_hot += 1

        # Sort expert lists within each layer for deterministic output
        for l_str in hot_experts:
            hot_experts[l_str].sort()

    # Compute actual size
    actual_size = 0
    for l in existing_layers:
        n_hot = len(hot_experts[str(l)])
        n_cold = num_experts - n_hot
        actual_size += n_hot * size4 + n_cold * size2

    print(f"\n  4-bit experts: {num_hot}")
    print(f"  2-bit experts: {len(sensitivity_list) - num_hot}")
    print(f"  Actual size:   {actual_size / 1024**3:.2f} GB")

    # Per-layer breakdown
    print(f"\n  Per-layer breakdown:")
    for l in existing_layers:
        n_hot = len(hot_experts[str(l)])
        lw = layer_weight.get(l, 1.0)
        marker = " *" if lw > 1.0 else ""
        print(f"    Layer {l:2d}: {n_hot:3d} hot (4-bit), "
              f"{num_experts - n_hot:3d} cold (2-bit){marker}")

    # ---- Write output ----
    output_data = {
        "description": "JANG sensitivity-based tiered bit assignment",
        "target_gb": args.target_gb,
        "actual_gb": actual_size / 1024**3,
        "num_hot_total": num_hot,
        "num_cold_total": len(sensitivity_list) - num_hot,
        "expert_size_4bit": size4,
        "expert_size_2bit": size2,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "threshold": None,  # compatibility with repack_experts_tiered.py
        "hot_experts": hot_experts,
    }

    output_path = args.output
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nWrote {output_path}")
    print(f"  {num_hot} experts at 4-bit, "
          f"{len(sensitivity_list) - num_hot} at 2-bit")
    print(f"  Target: {args.target_gb:.1f} GB, Actual: {actual_size / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
