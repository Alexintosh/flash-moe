#!/usr/bin/env python3
"""Repack experts into tiered quantization: hot @ 4-bit, cold @ 2-bit.

Reads packed_experts/ (all 4-bit) and a hot_experts.json (from sensitivity
analysis or threshold-based profiling), produces packed_experts_tiered/
with mixed-quant layer files + manifest.

Cold experts can use either:
  - GPTQ-quantized 2-bit data from --gptq-dir (best quality)
  - Inline MSE-optimal clipping requantization (fallback)

File format per layer:
  [expert_0_data][expert_1_data]...[expert_N_data]
  Each expert is either 4-bit or 2-bit format (variable size).
  tiered_manifest.json records per-expert: {offset, size, bits}

Usage:
    # Sensitivity-based with GPTQ 2-bit data:
    python repack_experts_tiered.py \\
      --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit \\
      --hot-experts hot_experts.json \\
      --gptq-dir packed_experts_gptq_2bit/

    # Sensitivity-based without GPTQ (inline RTN requant):
    python repack_experts_tiered.py \\
      --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit \\
      --hot-experts hot_experts.json

    # Legacy threshold-based (backward compatible):
    python repack_experts_tiered.py \\
      --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit \\
      --threshold 100
"""
import argparse
import json
import numpy as np
import os
import sys
from pathlib import Path


def bf16_to_f32(bf16_u16):
    return (bf16_u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(f32):
    return (f32.view(np.uint32) >> 16).astype(np.uint16)


def unpack_4bit(packed):
    """Extract 8 x 4-bit values from each uint32 (LSB first)."""
    flat = packed.ravel()
    out = np.empty(flat.size * 8, dtype=np.uint8)
    for i in range(8):
        out[i::8] = ((flat >> (i * 4)) & 0xF).astype(np.uint8)
    return out.reshape(packed.shape[:-1] + (packed.shape[-1] * 8,))


def pack_2bit(vals):
    """Pack 16 x 2-bit values into each uint32 (LSB first)."""
    shape = vals.shape
    n_packed = shape[-1] // 16
    flat = vals.reshape(-1, shape[-1])
    out = np.zeros((flat.shape[0], n_packed), dtype=np.uint32)
    for i in range(16):
        out |= flat[:, i::16].astype(np.uint32) << (i * 2)
    return out.reshape(shape[:-1] + (n_packed,))


def requantize_projection(packed_4bit, scales_bf16, biases_bf16, out_dim, in_dim, group_size=64):
    """4-bit -> 2-bit with optimal per-group quantization."""
    num_groups = in_dim // group_size
    vals_4bit = unpack_4bit(packed_4bit).reshape(out_dim, in_dim).astype(np.float32)
    scales_f32 = bf16_to_f32(scales_bf16)
    biases_f32 = bf16_to_f32(biases_bf16)

    vals_grouped = vals_4bit.reshape(out_dim, num_groups, group_size)
    s = scales_f32[:, :, np.newaxis]
    b = biases_f32[:, :, np.newaxis]
    dequant = vals_grouped * s + b

    f_min = dequant.min(axis=2, keepdims=True)
    f_max = dequant.max(axis=2, keepdims=True)
    s2 = (f_max - f_min) / 3.0
    b2 = f_min
    s2_safe = np.where(s2 == 0.0, 1.0, s2)
    vals_2bit = np.clip(np.round((dequant - b2) / s2_safe), 0, 3).astype(np.uint8)

    packed_2bit = pack_2bit(vals_2bit.reshape(out_dim, in_dim))
    new_scales = f32_to_bf16(s2.squeeze(axis=2))
    new_biases = f32_to_bf16(b2.squeeze(axis=2))

    return packed_2bit, new_scales, new_biases


def compute_offsets(moe_intermediate, hidden_dim, group_size, bits):
    """Compute expert byte offsets for given bit width. Mirrors compute_expert_offsets() in infer.m."""
    vals_per_u32 = 32 // bits
    mid, hid, gs = moe_intermediate, hidden_dim, group_size

    def proj_sizes(out_d, in_d):
        w = out_d * ((in_d + vals_per_u32 - 1) // vals_per_u32) * 4
        s = out_d * ((in_d + gs - 1) // gs) * 2
        return w, s, s  # weight, scales, biases

    gw, gs_sz, gb = proj_sizes(mid, hid)
    uw, us, ub = proj_sizes(mid, hid)
    dw, ds, db = proj_sizes(hid, mid)

    offsets = {}
    off = 0
    for name, w, s, b in [("gate", gw, gs_sz, gb), ("up", uw, us, ub), ("down", dw, ds, db)]:
        offsets[f"{name}_w"] = off; off += w
        offsets[f"{name}_s"] = off; off += s
        offsets[f"{name}_b"] = off; off += b

    return offsets, off  # offsets dict + total expert size


def requantize_expert(expert_4bit_blob, off4, size4, off2, size2,
                       moe_intermediate, hidden_dim, group_size=64):
    """Requantize a single expert from 4-bit to 2-bit (inline MSE-optimal clipping)."""
    output = bytearray(size2)

    projs = [
        ("gate", moe_intermediate, hidden_dim),
        ("up", moe_intermediate, hidden_dim),
        ("down", hidden_dim, moe_intermediate),
    ]

    for name, out_dim, in_dim in projs:
        # Read 4-bit components
        w_start = off4[f"{name}_w"]
        s_start = off4[f"{name}_s"]
        b_start = off4[f"{name}_b"]

        packed_cols_4 = (in_dim + 7) // 8
        w_bytes = out_dim * packed_cols_4 * 4
        num_groups = (in_dim + group_size - 1) // group_size
        s_bytes = out_dim * num_groups * 2

        packed_4bit = np.frombuffer(
            expert_4bit_blob[w_start:w_start + w_bytes], dtype=np.uint32
        ).reshape(out_dim, packed_cols_4)
        scales = np.frombuffer(
            expert_4bit_blob[s_start:s_start + s_bytes], dtype=np.uint16
        ).reshape(out_dim, num_groups)
        biases = np.frombuffer(
            expert_4bit_blob[b_start:b_start + s_bytes], dtype=np.uint16
        ).reshape(out_dim, num_groups)

        packed_2bit, new_scales, new_biases = requantize_projection(
            packed_4bit, scales, biases, out_dim, in_dim, group_size
        )

        # Write 2-bit components
        w2_start = off2[f"{name}_w"]
        s2_start = off2[f"{name}_s"]
        b2_start = off2[f"{name}_b"]
        output[w2_start:w2_start + packed_2bit.nbytes] = packed_2bit.tobytes()
        output[s2_start:s2_start + new_scales.nbytes] = new_scales.tobytes()
        output[b2_start:b2_start + new_biases.nbytes] = new_biases.tobytes()

    return bytes(output)


def load_gptq_expert(gptq_dir, layer_idx, expert_idx, size2):
    """Load a pre-quantized GPTQ 2-bit expert blob from the gptq directory.

    Expected layout: gptq_dir/layer_XX.bin with experts packed sequentially,
    each of size2 bytes.

    Returns bytes or None if not available.
    """
    layer_path = Path(gptq_dir) / f"layer_{layer_idx:02d}.bin"
    if not layer_path.exists():
        return None

    file_size = layer_path.stat().st_size
    expert_offset = expert_idx * size2
    if expert_offset + size2 > file_size:
        return None

    with open(layer_path, "rb") as f:
        f.seek(expert_offset)
        blob = f.read(size2)

    if len(blob) != size2:
        return None

    return blob


def build_hot_experts_from_threshold(freq_data, threshold):
    """Build hot_experts dict from frequency data using a threshold.

    Legacy backward-compatible path: any expert with activation count >= threshold
    is considered hot.

    freq_data: dict with structure matching profile_experts.py output
    threshold: minimum activation count

    Returns: dict {str(layer_idx): [list of hot expert indices]}
    """
    hot_experts = {}
    # freq_data may have various structures; try common ones
    if "expert_counts" in freq_data:
        # profile_experts.py format: {"expert_counts": {"layer": {"expert": count}}}
        for layer_str, expert_counts in freq_data["expert_counts"].items():
            hot = []
            for expert_str, count in expert_counts.items():
                if count >= threshold:
                    hot.append(int(expert_str))
            hot_experts[layer_str] = sorted(hot)
    elif "hot_experts" in freq_data:
        # Already in hot_experts format
        return freq_data["hot_experts"]
    else:
        # Assume flat {layer: {expert: count}} structure
        for layer_str, expert_counts in freq_data.items():
            if not isinstance(expert_counts, dict):
                continue
            hot = []
            for expert_str, count in expert_counts.items():
                if isinstance(count, (int, float)) and count >= threshold:
                    hot.append(int(expert_str))
            hot_experts[layer_str] = sorted(hot)

    return hot_experts


def main():
    parser = argparse.ArgumentParser(description="Repack experts with tiered quantization")
    parser.add_argument("--model", required=True, help="Path to model directory with packed_experts/")
    parser.add_argument("--hot-experts", default=None,
                        help="Path to hot_experts.json from sensitivity_analysis.py or profile_experts.py")
    parser.add_argument("--threshold", type=int, default=None,
                        help="Legacy: activation count threshold for hot/cold split. "
                             "Requires --hot-experts to point to frequency data.")
    parser.add_argument("--gptq-dir", default=None,
                        help="Directory with GPTQ-quantized 2-bit expert files (layer_XX.bin). "
                             "If provided, cold experts use GPTQ data instead of inline RTN requant.")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true", help="Compute sizes without writing files")
    args = parser.parse_args()

    # Validate arguments
    if args.hot_experts is None and args.threshold is None:
        print("ERROR: Must provide either --hot-experts or --threshold.", file=sys.stderr)
        sys.exit(1)

    model_path = Path(args.model)

    # Load hot expert assignment
    if args.hot_experts is not None:
        hot_file_data = json.loads(Path(args.hot_experts).read_text())
    else:
        hot_file_data = None

    if args.threshold is not None:
        # Legacy threshold-based mode
        if hot_file_data is None:
            print("ERROR: --threshold requires --hot-experts pointing to frequency data.",
                  file=sys.stderr)
            sys.exit(1)
        hot_expert_map = build_hot_experts_from_threshold(hot_file_data, args.threshold)
        threshold_value = args.threshold
        print(f"Using threshold-based assignment (threshold={args.threshold})")
    else:
        # Sensitivity-based mode (from sensitivity_analysis.py output)
        if "hot_experts" in hot_file_data:
            hot_expert_map = hot_file_data["hot_experts"]
        else:
            print("ERROR: --hot-experts file must contain 'hot_experts' key.", file=sys.stderr)
            sys.exit(1)
        threshold_value = hot_file_data.get("threshold", None)
        desc = hot_file_data.get("description", "unknown source")
        print(f"Using hot expert assignment from: {desc}")

    # Read model config
    config_path = model_path / "config.json"
    if not config_path.exists():
        # Try snapshots subdir (HF cache layout)
        candidates = list(model_path.glob("snapshots/*/config.json"))
        config_path = candidates[0] if candidates else None
    if config_path is None or not config_path.exists():
        print("ERROR: config.json not found"); sys.exit(1)
    config = json.loads(config_path.read_text())
    tc = config.get("text_config", config)

    num_experts = tc["num_experts"]
    num_layers = tc["num_hidden_layers"]
    moe_intermediate = tc.get("moe_intermediate_size", tc.get("moe_intermediate"))
    hidden_dim = tc["hidden_size"]
    group_size = args.group_size

    # Compute offsets for both formats
    off4, size4 = compute_offsets(moe_intermediate, hidden_dim, group_size, 4)
    off2, size2 = compute_offsets(moe_intermediate, hidden_dim, group_size, 2)

    print(f"Model: {model_path}")
    print(f"  {num_layers} layers, {num_experts} experts/layer")
    print(f"  4-bit expert: {size4:,} bytes ({size4/1024/1024:.2f} MB)")
    print(f"  2-bit expert: {size2:,} bytes ({size2/1024/1024:.2f} MB)")
    print(f"  Reduction per cold expert: {(size4-size2)/1024/1024:.2f} MB ({100*(size4-size2)/size4:.1f}%)")

    # GPTQ info
    gptq_dir = args.gptq_dir
    if gptq_dir:
        gptq_path = Path(gptq_dir)
        if gptq_path.exists():
            gptq_files = list(gptq_path.glob("layer_*.bin"))
            print(f"  GPTQ dir: {gptq_dir} ({len(gptq_files)} layer files)")
        else:
            print(f"  WARNING: GPTQ dir {gptq_dir} does not exist, falling back to inline requant")
            gptq_dir = None
    else:
        print(f"  GPTQ dir: not provided (using inline MSE-optimal clipping)")

    # Compute total sizes
    total_4bit = num_layers * num_experts * size4
    total_hot = 0
    total_cold = 0
    for l in range(num_layers):
        hot_set = set(hot_expert_map.get(str(l), []))
        n_hot = len(hot_set)
        n_cold = num_experts - n_hot
        total_hot += n_hot * size4
        total_cold += n_cold * size2

    total_tiered = total_hot + total_cold
    print(f"\n  All 4-bit:    {total_4bit/1024/1024/1024:.2f} GB")
    print(f"  Tiered total: {total_tiered/1024/1024/1024:.2f} GB ({100*total_tiered/total_4bit:.1f}%)")
    print(f"  Savings:      {(total_4bit-total_tiered)/1024/1024/1024:.2f} GB")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    # Create output directory
    experts_dir = model_path / "packed_experts"
    tiered_dir = model_path / "packed_experts_tiered"
    tiered_dir.mkdir(exist_ok=True)

    manifest = {
        "expert_size_4bit": size4,
        "expert_size_2bit": size2,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "threshold": threshold_value,
        "gptq_source": gptq_dir if gptq_dir else None,
        "layers": {},
    }

    gptq_used = 0
    gptq_fallback = 0

    for l in range(num_layers):
        src_path = experts_dir / f"layer_{l:02d}.bin"
        dst_path = tiered_dir / f"layer_{l:02d}.bin"

        if not src_path.exists():
            print(f"  WARNING: {src_path} not found, skipping layer {l}")
            continue

        hot_set = set(hot_expert_map.get(str(l), []))
        n_hot = len(hot_set)
        n_cold = num_experts - n_hot
        print(f"  Layer {l:2d}: {n_hot} hot (4-bit), {n_cold} cold (2-bit)...",
              end="", flush=True)

        # Read entire source layer file
        src_data = src_path.read_bytes()

        # Build tiered layer: sequential experts with variable sizes
        layer_manifest = {"experts": []}
        output_offset = 0
        layer_chunks = []

        for e in range(num_experts):
            expert_start = e * size4
            expert_blob = src_data[expert_start:expert_start + size4]

            if e in hot_set:
                # Keep 4-bit
                layer_chunks.append(expert_blob)
                layer_manifest["experts"].append({
                    "offset": output_offset,
                    "size": size4,
                    "bits": 4,
                })
                output_offset += size4
            else:
                # Cold expert: try GPTQ data first, then fall back to inline requant
                blob_2bit = None
                if gptq_dir:
                    blob_2bit = load_gptq_expert(gptq_dir, l, e, size2)
                    if blob_2bit is not None:
                        gptq_used += 1
                    else:
                        gptq_fallback += 1

                if blob_2bit is None:
                    # Inline requantization (MSE-optimal clipping)
                    blob_2bit = requantize_expert(
                        expert_blob, off4, size4, off2, size2,
                        moe_intermediate, hidden_dim, group_size
                    )

                layer_chunks.append(blob_2bit)
                layer_manifest["experts"].append({
                    "offset": output_offset,
                    "size": size2,
                    "bits": 2,
                })
                output_offset += size2

        # Write tiered layer file
        with open(dst_path, "wb") as f:
            for chunk in layer_chunks:
                f.write(chunk)

        layer_manifest["file_size"] = output_offset
        manifest["layers"][str(l)] = layer_manifest

        print(f" {output_offset/1024/1024:.1f} MB (was {num_experts * size4 / 1024/1024:.1f} MB)")

    # Write manifest
    manifest_path = tiered_dir / "tiered_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest: {manifest_path}")
    print(f"Total tiered: {total_tiered/1024/1024/1024:.2f} GB (was {total_4bit/1024/1024/1024:.2f} GB)")

    if gptq_dir:
        total_cold_experts = gptq_used + gptq_fallback
        print(f"\nGPTQ 2-bit source: {gptq_used}/{total_cold_experts} cold experts "
              f"({gptq_fallback} fell back to inline requant)")


if __name__ == "__main__":
    main()
