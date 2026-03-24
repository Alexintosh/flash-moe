#!/usr/bin/env python3
"""
build_hessian.py -- Streaming Hessian accumulator for GPTQ calibration.

Reads the binary activation dump produced by `./infer --collect-activations FILE`
and accumulates H[layer][expert] += outer(h_post, h_post) in float32.

Binary format per record (one per token per layer):
    int32  layer_idx
    int32  K
    K x {
        int32       expert_idx
        float32[D]  h_post        (D = hidden_dim)
    }

Two-pass approach to fit in 48GB Mac memory:
  Pass 1: scan file to count samples per (layer, expert)
  Pass 2: for each layer, re-read file and accumulate Hessians for that layer only

Output: calibration/layer_LL/expert_EEE.npy  (for experts with >= min_samples)
        calibration/layer_LL/counts.json      (sample counts)

Usage:
    python build_hessian.py <activation_dump.bin> --hidden-dim 4096 --output-dir calibration/ --min-samples 64
"""

import argparse
import json
import os
import struct
import sys

import numpy as np


def scan_counts(filepath, hidden_dim):
    """Pass 1: Count samples per (layer, expert) without accumulating Hessians."""
    counts = {}  # (layer, expert) -> int
    record_count = 0

    with open(filepath, "rb") as f:
        while True:
            header = f.read(8)  # int32 layer_idx + int32 K
            if len(header) < 8:
                break

            layer_idx, K = struct.unpack("<ii", header)

            for _ in range(K):
                eidx_bytes = f.read(4)
                if len(eidx_bytes) < 4:
                    return counts, record_count
                expert_idx = struct.unpack("<i", eidx_bytes)[0]

                # Skip the h_post vector (hidden_dim * 4 bytes)
                skip_bytes = hidden_dim * 4
                f.seek(skip_bytes, 1)

                key = (layer_idx, expert_idx)
                counts[key] = counts.get(key, 0) + 1

            record_count += 1

    return counts, record_count


def accumulate_layer(filepath, target_layer, hidden_dim, expert_set):
    """Pass 2: Accumulate Hessians for a single layer.

    Only processes experts in expert_set (those with enough samples).
    Returns dict: expert_idx -> H (hidden_dim x hidden_dim float32 ndarray).
    """
    hessians = {}
    sample_counts = {}

    for eidx in expert_set:
        hessians[eidx] = np.zeros((hidden_dim, hidden_dim), dtype=np.float32)
        sample_counts[eidx] = 0

    h_post = np.empty(hidden_dim, dtype=np.float32)

    with open(filepath, "rb") as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                break

            layer_idx, K = struct.unpack("<ii", header)

            for _ in range(K):
                eidx_bytes = f.read(4)
                if len(eidx_bytes) < 4:
                    return hessians, sample_counts
                expert_idx = struct.unpack("<i", eidx_bytes)[0]

                vec_bytes = hidden_dim * 4
                if layer_idx == target_layer and expert_idx in expert_set:
                    raw = f.read(vec_bytes)
                    if len(raw) < vec_bytes:
                        return hessians, sample_counts
                    np.frombuffer(raw, dtype=np.float32, count=hidden_dim, offset=0).copyto(h_post)
                    # Rank-1 update: H += outer(h_post, h_post)
                    # Use np.outer for clarity; BLAS sger underneath via numpy
                    hessians[expert_idx] += np.outer(h_post, h_post)
                    sample_counts[expert_idx] += 1
                else:
                    f.seek(vec_bytes, 1)

    return hessians, sample_counts


def main():
    parser = argparse.ArgumentParser(
        description="Build per-expert Hessian matrices from activation dumps for GPTQ calibration."
    )
    parser.add_argument("activation_dump", help="Path to binary activation dump file")
    parser.add_argument("--hidden-dim", type=int, default=4096, help="Model hidden dimension (default: 4096)")
    parser.add_argument("--output-dir", default="calibration/", help="Output directory (default: calibration/)")
    parser.add_argument("--min-samples", type=int, default=64, help="Minimum samples to output Hessian (default: 64)")
    args = parser.parse_args()

    if not os.path.isfile(args.activation_dump):
        print(f"ERROR: File not found: {args.activation_dump}", file=sys.stderr)
        sys.exit(1)

    file_size = os.path.getsize(args.activation_dump)
    print(f"Activation dump: {args.activation_dump} ({file_size / 1e9:.2f} GB)")
    print(f"Hidden dim: {args.hidden_dim}, Min samples: {args.min_samples}")

    # ---- Pass 1: Count samples ----
    print("\nPass 1: Scanning for sample counts...")
    counts, record_count = scan_counts(args.activation_dump, args.hidden_dim)
    print(f"  Found {record_count} records across {len(counts)} (layer, expert) pairs")

    if not counts:
        print("ERROR: No records found in activation dump.", file=sys.stderr)
        sys.exit(1)

    # Determine unique layers
    layers = sorted(set(l for l, _ in counts.keys()))
    print(f"  Layers: {len(layers)} ({min(layers)}-{max(layers)})")

    # Count experts per layer meeting threshold
    for layer in layers:
        layer_experts = {e for (l, e), c in counts.items() if l == layer and c >= args.min_samples}
        total_experts = sum(1 for (l, _) in counts.keys() if l == layer)
        print(f"  Layer {layer:2d}: {len(layer_experts)}/{total_experts} experts with >= {args.min_samples} samples")

    # ---- Pass 2: Accumulate Hessians one layer at a time ----
    print(f"\nPass 2: Accumulating Hessians (one layer at a time)...")
    hessian_size_mb = args.hidden_dim * args.hidden_dim * 4 / 1e6

    total_saved = 0
    for layer in layers:
        # Experts meeting the threshold for this layer
        expert_set = {e for (l, e), c in counts.items() if l == layer and c >= args.min_samples}
        if not expert_set:
            print(f"  Layer {layer:2d}: skipped (no experts with >= {args.min_samples} samples)")
            continue

        est_mem_gb = len(expert_set) * hessian_size_mb / 1000
        print(f"  Layer {layer:2d}: accumulating {len(expert_set)} experts (~{est_mem_gb:.1f} GB Hessian memory)...")

        hessians, sample_counts = accumulate_layer(
            args.activation_dump, layer, args.hidden_dim, expert_set
        )

        # Save outputs
        layer_dir = os.path.join(args.output_dir, f"layer_{layer:02d}")
        os.makedirs(layer_dir, exist_ok=True)

        layer_counts = {}
        for eidx in sorted(expert_set):
            n = sample_counts[eidx]
            layer_counts[str(eidx)] = n
            if n >= args.min_samples:
                out_path = os.path.join(layer_dir, f"expert_{eidx:03d}.npy")
                np.save(out_path, hessians[eidx])
                total_saved += 1

        counts_path = os.path.join(layer_dir, "counts.json")
        with open(counts_path, "w") as f:
            json.dump(layer_counts, f, indent=2)

        # Free Hessians for this layer before moving to next
        del hessians

    print(f"\nDone. Saved {total_saved} Hessian matrices to {args.output_dir}")


if __name__ == "__main__":
    main()
