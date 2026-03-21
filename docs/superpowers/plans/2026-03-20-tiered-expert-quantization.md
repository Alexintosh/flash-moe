# Tiered Expert Quantization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce total expert disk footprint by keeping frequently-used ("hot") experts at 4-bit quality while requantizing rarely-used ("cold") experts to 2-bit, improving OS page cache hit rates and throughput.

**Architecture:** A profiling pass identifies hot experts per layer from frequency data. A repacking script creates tiered layer files where each expert's quantization is chosen independently. The runtime loads a manifest to determine per-expert offset and quant type, dispatching to the correct Metal dequant kernel per expert within the same token.

**Tech Stack:** Python (NumPy) for profiling/repacking, Objective-C/Metal for runtime, JSON manifests.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `profile_experts.py` | Create | Run inference with `--freq`, aggregate stats, output `hot_experts.json` |
| `repack_experts_tiered.py` | Create | Repack layer files with per-expert 4-bit/2-bit quantization + manifest |
| `metal_infer/infer.m` | Modify | Load tiered manifest, per-expert quant dispatch in `gpu_encode_experts_batched` and pread |
| `metal_infer/infer.m` | Modify | Add `--tiered` CLI flag and `packed_experts_tiered/` auto-detection |

---

### Task 1: Expert Profiling Script

**Files:**
- Create: `profile_experts.py`

This script wraps the existing `--freq` output from the inference engine. It runs inference across diverse prompts, parses the frequency output, and produces a `hot_experts.json` manifest identifying hot experts per layer.

- [ ] **Step 1: Create `profile_experts.py` — frequency output parser**

```python
#!/usr/bin/env python3
"""Profile expert usage and generate hot_experts.json for tiered quantization.

Runs inference with --freq across diverse prompts, parses frequency output,
and identifies hot experts per layer based on configurable coverage threshold.

Usage:
    python profile_experts.py --model /path/to/model --threshold 0.8
    python profile_experts.py --freq-output freq_log.txt --threshold 0.8
"""
import argparse
import json
import subprocess
import re
import sys
from pathlib import Path

# Diverse prompts covering different domains to get representative expert usage
DEFAULT_PROMPTS = [
    "Explain the theory of relativity in detail, covering both special and general relativity",
    "Write a Python function to implement a red-black tree with insert and delete operations",
    "Translate the following to French: The quick brown fox jumps over the lazy dog",
    "What are the key differences between TCP and UDP protocols?",
    "Describe the process of photosynthesis at the molecular level",
    '{"name": "get_weather", "arguments": {"location": "San Francisco", "units": "celsius"}}',
    "Solve the integral of x^2 * e^x dx step by step",
    "Write a haiku about autumn leaves falling",
]

def run_inference_with_freq(model_path: str, prompt: str, tokens: int = 200) -> str:
    """Run inference with --freq flag and capture stderr output."""
    cmd = [
        "./metal_infer/infer",
        "--model", model_path,
        "--prompt", prompt,
        "--tokens", str(tokens),
        "--freq",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return result.stderr

def parse_freq_output(stderr: str, num_layers: int, num_experts: int) -> dict:
    """Parse frequency analysis from stderr into per-layer frequency arrays.

    Returns: {layer_idx: {expert_idx: count, ...}, ...}
    """
    # The --freq output prints per-layer lines like:
    # Layer  0: 219 unique experts, top-10 cover 14%, ...
    # But doesn't print raw frequencies. We need to modify the approach.
    # Instead, we'll add a --freq-json flag or parse the raw freq data.
    # For now, we parse the coverage thresholds.
    freqs = {}
    pattern = r"Layer\s+(\d+):\s+(\d+) unique experts.*?50%@(\d+), 80%@(\d+), 90%@(\d+)"
    for m in re.finditer(pattern, stderr):
        layer = int(m.group(1))
        freqs[layer] = {
            "unique": int(m.group(2)),
            "n_for_50": int(m.group(3)),
            "n_for_80": int(m.group(4)),
            "n_for_90": int(m.group(5)),
        }
    return freqs

def parse_raw_freq_dump(stderr: str) -> dict:
    """Parse raw frequency dump (--freq-dump) into per-layer expert frequencies.

    Expected format (one line per layer):
    FREQ_DUMP layer=0: 0:15 1:3 2:0 3:42 ...

    Returns: {layer_idx: [(expert_idx, count), ...] sorted descending by count}
    """
    freqs = {}
    for line in stderr.splitlines():
        m = re.match(r"FREQ_DUMP layer=(\d+):\s*(.*)", line)
        if not m:
            continue
        layer = int(m.group(1))
        pairs = []
        for pair in m.group(2).strip().split():
            eidx, count = pair.split(":")
            pairs.append((int(eidx), int(count)))
        # Sort by count descending
        pairs.sort(key=lambda x: -x[1])
        freqs[layer] = pairs
    return freqs

def select_hot_experts(freq_data: dict, num_experts: int, threshold: float) -> dict:
    """Select hot experts per layer to achieve the target activation coverage.

    Args:
        freq_data: {layer: [(expert_idx, count), ...]} sorted descending
        num_experts: total experts per layer
        threshold: target coverage (e.g. 0.8 = 80%)

    Returns: {layer: [list of hot expert indices]}
    """
    hot_map = {}
    for layer, pairs in sorted(freq_data.items()):
        total = sum(c for _, c in pairs)
        if total == 0:
            hot_map[layer] = list(range(num_experts))  # all hot if no data
            continue
        cumulative = 0
        hot = []
        for eidx, count in pairs:
            hot.append(eidx)
            cumulative += count
            if cumulative / total >= threshold:
                break
        hot_map[layer] = sorted(hot)
    return hot_map

def main():
    parser = argparse.ArgumentParser(description="Profile experts and generate hot_experts.json")
    parser.add_argument("--model", type=str, help="Path to model directory")
    parser.add_argument("--freq-output", type=str, help="Pre-captured freq output file (skip inference)")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Coverage threshold (0.0-1.0, default 0.8 = keep top experts covering 80%%)")
    parser.add_argument("--tokens", type=int, default=200, help="Tokens per prompt")
    parser.add_argument("--output", type=str, default="hot_experts.json", help="Output manifest path")
    args = parser.parse_args()

    if args.freq_output:
        stderr = Path(args.freq_output).read_text()
    elif args.model:
        print(f"Running {len(DEFAULT_PROMPTS)} diverse prompts with --freq...")
        all_stderr = []
        for i, prompt in enumerate(DEFAULT_PROMPTS):
            print(f"  [{i+1}/{len(DEFAULT_PROMPTS)}] {prompt[:60]}...")
            stderr = run_inference_with_freq(args.model, prompt, args.tokens)
            all_stderr.append(stderr)
        stderr = "\n".join(all_stderr)
    else:
        parser.error("Need --model or --freq-output")

    freq_data = parse_raw_freq_dump(stderr)
    if not freq_data:
        print("ERROR: No FREQ_DUMP data found. Make sure infer was built with --freq-dump support.")
        print("       (This requires Task 2 modifications to infer.m)")
        sys.exit(1)

    # Read model config for num_experts
    if args.model:
        config_path = Path(args.model) / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            tc = config.get("text_config", config)
            num_experts = tc.get("num_experts", 256)
        else:
            num_experts = 256
    else:
        num_experts = 256

    hot_map = select_hot_experts(freq_data, num_experts, args.threshold)

    # Compute stats
    total_hot = sum(len(v) for v in hot_map.values())
    total_experts = len(hot_map) * num_experts

    manifest = {
        "threshold": args.threshold,
        "num_layers": len(hot_map),
        "num_experts": num_experts,
        "total_hot": total_hot,
        "total_cold": total_experts - total_hot,
        "hot_experts": {str(k): v for k, v in hot_map.items()},
    }

    Path(args.output).write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {args.output}")
    print(f"  Layers: {len(hot_map)}")
    print(f"  Hot experts: {total_hot} ({100*total_hot/total_experts:.1f}%)")
    print(f"  Cold experts: {total_experts - total_hot} ({100*(total_experts-total_hot)/total_experts:.1f}%)")
    avg_hot = total_hot / len(hot_map)
    print(f"  Avg hot/layer: {avg_hot:.0f} / {num_experts}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add `--freq-dump` raw output to `infer.m`**

Add a raw frequency dump at the end of `freq_print_analysis()` in `metal_infer/infer.m` (after line ~5953). This outputs machine-parseable frequency data:

```c
// At end of freq_print_analysis(), add:
fprintf(stderr, "\n--- Raw Frequency Dump (for profile_experts.py) ---\n");
for (int l = 0; l < cfg.num_layers; l++) {
    fprintf(stderr, "FREQ_DUMP layer=%d:", l);
    for (int e = 0; e < cfg.num_experts; e++) {
        int f = FREQ(l, e);
        if (f > 0) fprintf(stderr, " %d:%d", e, f);
    }
    fprintf(stderr, "\n");
}
```

- [ ] **Step 3: Test profiling script**

```bash
# First, rebuild infer with the --freq-dump output
cd metal_infer && make

# Run a single prompt to capture freq data
./infer --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit \
  --prompt "Explain quantum computing" --tokens 100 --freq 2> /tmp/freq_out.txt

# Verify FREQ_DUMP lines exist
grep "FREQ_DUMP" /tmp/freq_out.txt | head -3

# Run profiling script on captured output
cd .. && python profile_experts.py --freq-output /tmp/freq_out.txt --threshold 0.8 --output /tmp/hot_experts.json

# Inspect output
cat /tmp/hot_experts.json | python -m json.tool | head -30
```

Expected: `hot_experts.json` with per-layer lists of hot expert indices, total_hot < total_experts.

- [ ] **Step 4: Run full profiling across diverse prompts**

```bash
python profile_experts.py \
  --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit \
  --threshold 0.8 --tokens 200 \
  --output hot_experts.json
```

- [ ] **Step 5: Commit**

```bash
git add profile_experts.py
git commit -m "feat: add expert profiling script for tiered quantization"
```

---

### Task 2: Tiered Repacking Script

**Files:**
- Create: `repack_experts_tiered.py`

Takes `hot_experts.json` + existing 4-bit packed expert files, creates new tiered layer files where hot experts stay 4-bit and cold experts are requantized to 2-bit. Writes a `tiered_manifest.json` with per-expert metadata.

- [ ] **Step 1: Create `repack_experts_tiered.py`**

```python
#!/usr/bin/env python3
"""Repack experts into tiered quantization: hot @ 4-bit, cold @ 2-bit.

Reads packed_experts/ (all 4-bit) and hot_experts.json,
produces packed_experts_tiered/ with mixed-quant layer files + manifest.

File format per layer:
  [expert_0_data][expert_1_data]...[expert_N_data]

  Where each expert is either:
  - 4-bit format (expert_size_4bit bytes) if hot
  - 2-bit format (expert_size_2bit bytes) if cold

  The tiered_manifest.json records per-expert: {offset, size, bits}

Usage:
    python repack_experts_tiered.py \
      --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit \
      --hot-experts hot_experts.json
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
    """4-bit -> 2-bit with optimal per-group quantization. Returns (packed_2bit, new_scales, new_biases, rmse)."""
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

    recon = vals_2bit.astype(np.float32) * s2 + b2
    rmse = float(np.sqrt(np.mean((dequant - recon) ** 2)))

    packed_2bit = pack_2bit(vals_2bit.reshape(out_dim, in_dim))
    new_scales = f32_to_bf16(s2.squeeze(axis=2))
    new_biases = f32_to_bf16(b2.squeeze(axis=2))

    return packed_2bit, new_scales, new_biases, rmse

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

def requantize_expert(expert_4bit_blob, off4, size4, off2, size2, moe_intermediate, hidden_dim, group_size=64):
    """Requantize a single expert from 4-bit to 2-bit."""
    output = bytearray(size2)
    total_rmse = 0.0

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

        packed_4bit = np.frombuffer(expert_4bit_blob[w_start:w_start+w_bytes], dtype=np.uint32).reshape(out_dim, packed_cols_4)
        scales = np.frombuffer(expert_4bit_blob[s_start:s_start+s_bytes], dtype=np.uint16).reshape(out_dim, num_groups)
        biases = np.frombuffer(expert_4bit_blob[b_start:b_start+s_bytes], dtype=np.uint16).reshape(out_dim, num_groups)

        packed_2bit, new_scales, new_biases, rmse = requantize_projection(
            packed_4bit, scales, biases, out_dim, in_dim, group_size
        )
        total_rmse += rmse

        # Write 2-bit components
        w2_start = off2[f"{name}_w"]
        s2_start = off2[f"{name}_s"]
        b2_start = off2[f"{name}_b"]
        output[w2_start:w2_start+packed_2bit.nbytes] = packed_2bit.tobytes()
        output[s2_start:s2_start+new_scales.nbytes] = new_scales.tobytes()
        output[b2_start:b2_start+new_biases.nbytes] = new_biases.tobytes()

    return bytes(output), total_rmse / 3.0

def main():
    parser = argparse.ArgumentParser(description="Repack experts with tiered quantization")
    parser.add_argument("--model", required=True, help="Path to model directory with packed_experts/")
    parser.add_argument("--hot-experts", required=True, help="Path to hot_experts.json from profile_experts.py")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true", help="Compute sizes without writing files")
    args = parser.parse_args()

    model_path = Path(args.model)
    hot_data = json.loads(Path(args.hot_experts).read_text())

    # Read model config
    # Try snapshots/ subdir first (HF cache layout), then direct
    config_candidates = list(model_path.glob("snapshots/*/config.json")) + [model_path / "config.json"]
    config_path = next((p for p in config_candidates if p.exists()), None)
    if config_path is None:
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

    # Compute total sizes
    total_4bit = num_layers * num_experts * size4
    total_hot = 0
    total_cold = 0
    for l in range(num_layers):
        hot_set = set(hot_data["hot_experts"].get(str(l), []))
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
        "threshold": hot_data["threshold"],
        "layers": {},
    }

    for l in range(num_layers):
        src_path = experts_dir / f"layer_{l:02d}.bin"
        dst_path = tiered_dir / f"layer_{l:02d}.bin"

        if not src_path.exists():
            print(f"  WARNING: {src_path} not found, skipping layer {l}")
            continue

        hot_set = set(hot_data["hot_experts"].get(str(l), []))
        print(f"  Layer {l:2d}: {len(hot_set)} hot (4-bit), {num_experts - len(hot_set)} cold (2-bit)...", end="", flush=True)

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
                # Requantize to 2-bit
                blob_2bit, rmse = requantize_expert(
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

        hot_bytes = len(hot_set) * size4
        cold_bytes = (num_experts - len(hot_set)) * size2
        print(f" {output_offset/1024/1024:.1f} MB (was {num_experts * size4 / 1024/1024:.1f} MB)")

    # Write manifest
    manifest_path = tiered_dir / "tiered_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest: {manifest_path}")
    print(f"Total tiered: {total_tiered/1024/1024/1024:.2f} GB (was {total_4bit/1024/1024/1024:.2f} GB)")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test repacking script (dry run)**

```bash
python repack_experts_tiered.py \
  --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit \
  --hot-experts hot_experts.json \
  --dry-run
```

Expected: Prints size calculations showing ~39% reduction for 35B model.

- [ ] **Step 3: Run actual repacking**

```bash
python repack_experts_tiered.py \
  --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit \
  --hot-experts hot_experts.json
```

Expected: Creates `packed_experts_tiered/` directory with layer files and `tiered_manifest.json`.

- [ ] **Step 4: Verify output integrity**

```bash
# Check that tiered files are smaller than originals
du -sh ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit/packed_experts/
du -sh ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit/packed_experts_tiered/

# Verify manifest structure
python -c "
import json
m = json.load(open('$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit/packed_experts_tiered/tiered_manifest.json'))
print(f'Layers: {m[\"num_layers\"]}')
# Check first layer has mix of 4-bit and 2-bit experts
l0 = m['layers']['0']['experts']
bits_4 = sum(1 for e in l0 if e['bits'] == 4)
bits_2 = sum(1 for e in l0 if e['bits'] == 2)
print(f'Layer 0: {bits_4} @ 4-bit, {bits_2} @ 2-bit')
# Verify offsets are sequential
for i, e in enumerate(l0):
    if i > 0:
        prev = l0[i-1]
        assert e['offset'] == prev['offset'] + prev['size'], f'Gap at expert {i}'
print('Offsets OK: sequential, no gaps')
"
```

- [ ] **Step 5: Commit**

```bash
git add repack_experts_tiered.py
git commit -m "feat: add tiered repacking script (hot=4-bit, cold=2-bit)"
```

---

### Task 3: Runtime — Load Tiered Manifest

**Files:**
- Modify: `metal_infer/infer.m` (add tiered manifest loading + per-expert metadata)

Add a tiered manifest data structure and loader. The manifest tells the engine per-expert: byte offset in the layer file, byte size, and quantization bits.

- [ ] **Step 1: Add tiered manifest data structures to `infer.m`**

After the ModelConfig struct (around line 140), add:

```c
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
```

- [ ] **Step 2: Add manifest loader function**

Add after the `load_model_config()` function:

```c
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

    g_tiered_manifest = calloc(num_layers * num_experts, sizeof(TieredExpertInfo));

    NSDictionary *layers = root[@"layers"];
    for (int l = 0; l < num_layers; l++) {
        NSString *lkey = [NSString stringWithFormat:@"%d", l];
        NSDictionary *layer = layers[lkey];
        if (!layer) continue;

        NSArray *experts = layer[@"experts"];
        for (int e = 0; e < num_experts && e < (int)[experts count]; e++) {
            NSDictionary *exp = experts[e];
            TIERED(l, e).offset = [exp[@"offset"] unsignedLongLongValue];
            TIERED(l, e).size = [exp[@"size"] unsignedLongLongValue];
            TIERED(l, e).bits = [exp[@"bits"] intValue];
        }
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
```

- [ ] **Step 3: Add `--tiered` CLI flag and auto-detection**

In the CLI options section (around line 6855), add the flag:

```c
// In long_options array:
{"tiered", no_argument, 0, 'D'},

// In the help text:
printf("  --tiered             Use tiered expert quantization (packed_experts_tiered/)\n");

// In the switch:
case 'D': g_use_tiered = 1; break;
```

In the model loading section (around line 7050, where `--2bit` auto-detection happens), add tiered auto-detection:

```c
// ---- Auto-detect tiered experts ----
if (!g_use_2bit) {  // tiered takes priority over 2-bit, but not if 2-bit explicitly set
    char probe[1024];
    snprintf(probe, sizeof(probe), "%s/packed_experts_tiered/tiered_manifest.json", model_path);
    if (access(probe, F_OK) == 0) {
        if (load_tiered_manifest(model_path)) {
            g_use_tiered = 1;
        }
    }
}
```

- [ ] **Step 4: Update expert file opening for tiered mode**

In the layer file opening loop (around line 7080), add tiered path:

```c
for (int i = 0; i < cfg.num_layers; i++) {
    char path[1024];
    if (g_use_tiered) {
        snprintf(path, sizeof(path), "%s/packed_experts_tiered/layer_%02d.bin", model_path, i);
    } else {
        snprintf(path, sizeof(path), "%s/%s/layer_%02d.bin", model_path,
                 g_use_2bit ? "packed_experts_2bit" : "packed_experts", i);
    }
    // ... rest of opening logic unchanged ...
}
```

- [ ] **Step 5: Rebuild and test manifest loading**

```bash
cd metal_infer && make

# Test that tiered manifest is detected and loaded
./infer --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit \
  --prompt "Hello" --tokens 5 2>&1 | grep -i tiered
```

Expected: `[tiered] Loaded manifest: N hot (4-bit) + M cold (2-bit), threshold=80%`

- [ ] **Step 6: Commit**

```bash
git add metal_infer/infer.m
git commit -m "feat: add tiered manifest loading and --tiered flag"
```

---

### Task 4: Runtime — Per-Expert Quantization Dispatch

**Files:**
- Modify: `metal_infer/infer.m` — three functions need per-expert quant awareness

This is the core runtime change. Three areas need modification:
1. **Expert pread** — read correct number of bytes per expert (variable sizes)
2. **GPU kernel dispatch** — select 4-bit or 2-bit dequant kernel per expert
3. **Async pread** — same variable-size reads

- [ ] **Step 1: Modify `parallel_pread_experts` for tiered mode**

Replace the single `esz = active_expert_size()` with per-expert size lookup in `parallel_pread_experts()` (line 3430):

```c
static int parallel_pread_experts(
    int packed_fd,
    int *expert_indices,
    int K,
    int *valid,
    const void *mmap_base,
    int layer_idx  // NEW parameter: needed for tiered manifest lookup
) {
    InferPreadTask tasks[MAX_K];
    for (int k = 0; k < K; k++) {
        size_t esz;
        off_t offset;
        if (g_use_tiered && g_tiered_manifest) {
            TieredExpertInfo *ti = &TIERED(layer_idx, expert_indices[k]);
            esz = ti->size;
            offset = (off_t)ti->offset;
        } else {
            esz = active_expert_size();
            offset = (off_t)expert_indices[k] * esz;
        }
        tasks[k].fd = packed_fd;
        tasks[k].dst = [g_metal->buf_multi_expert_data[k] contents];
        tasks[k].offset = offset;
        tasks[k].size = esz;
        tasks[k].result = 0;
        tasks[k].mmap_base = mmap_base;
    }

    io_pool_dispatch(tasks, K);

    int loaded = 0;
    for (int k = 0; k < K; k++) {
        valid[k] = (tasks[k].result == (ssize_t)tasks[k].size);
        if (valid[k]) loaded++;
        else {
            fprintf(stderr, "WARNING: expert %d pread: %zd/%zu\n",
                    expert_indices[k], tasks[k].result, tasks[k].size);
        }
    }
    return loaded;
}
```

Note: Also update `async_pread_start()` (line 3379) with the same pattern, adding `layer_idx` parameter.

- [ ] **Step 2: Update all callers of pread functions to pass `layer_idx`**

Search for all call sites of `parallel_pread_experts`, `async_pread_start`, and `parallel_pread_experts_into` and add the `layer_idx` argument. Key call sites:

- Around line 5594: `async_pread_start(packed_fd, expert_indices, actual_K, ...)`
- Other call sites found via grep for `parallel_pread_experts\|async_pread_start`

- [ ] **Step 3: Modify `gpu_encode_experts_batched` for per-expert kernel selection**

This is the key change. Currently (line 2001), offsets and pipeline are selected once based on `g_use_2bit`. For tiered mode, move selection inside the per-expert loop:

```c
static void gpu_encode_experts_batched(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    int K,
    const int *valid,
    id<MTLBuffer> __strong *expert_bufs,
    int layer_idx,           // NEW: needed for tiered lookup
    const int *expert_indices // NEW: needed to look up per-expert quant
) {
    uint32_t gate_up_out = cfg.moe_intermediate;
    uint32_t gate_up_in  = cfg.hidden_dim;
    uint32_t down_out    = cfg.hidden_dim;
    uint32_t down_in     = cfg.moe_intermediate;
    uint32_t gs          = cfg.group_size;
    uint32_t gate_up_tgs = (gate_up_out + 7) / 8;
    uint32_t down_tgs    = (down_out + 7) / 8;
    uint32_t swiglu_tgs  = (gate_up_out + 255) / 256;

    for (int k = 0; k < K; k++) {
        if (!valid[k]) continue;

        // Per-expert quantization selection
        int use_2bit_k;
        if (g_use_tiered && g_tiered_manifest) {
            use_2bit_k = (TIERED(layer_idx, expert_indices[k]).bits == 2);
        } else {
            use_2bit_k = g_use_2bit;
        }

        NSUInteger gate_w_off, gate_s_off, gate_b_off;
        NSUInteger up_w_off, up_s_off, up_b_off;
        NSUInteger down_w_off, down_s_off, down_b_off;
        id<MTLComputePipelineState> expert_pipe;

        if (use_2bit_k) {
            gate_w_off = cfg.gate_w_off_2; gate_s_off = cfg.gate_s_off_2; gate_b_off = cfg.gate_b_off_2;
            up_w_off   = cfg.up_w_off_2;   up_s_off   = cfg.up_s_off_2;   up_b_off   = cfg.up_b_off_2;
            down_w_off = cfg.down_w_off_2; down_s_off = cfg.down_s_off_2; down_b_off = cfg.down_b_off_2;
            expert_pipe = ctx->matvec_2bit;
        } else {
            gate_w_off = cfg.gate_w_off_4; gate_s_off = cfg.gate_s_off_4; gate_b_off = cfg.gate_b_off_4;
            up_w_off   = cfg.up_w_off_4;   up_s_off   = cfg.up_s_off_4;   up_b_off   = cfg.up_b_off_4;
            down_w_off = cfg.down_w_off_4; down_s_off = cfg.down_s_off_4; down_b_off = cfg.down_b_off_4;
            expert_pipe = ctx->matvec_v3;
        }

        // Encoder A: gate_proj + up_proj (unchanged except using per-expert offsets/pipe)
        {
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            [enc setComputePipelineState:expert_pipe];
            [enc setBuffer:expert_bufs[k]                  offset:gate_w_off  atIndex:0];
            [enc setBuffer:expert_bufs[k]                  offset:gate_s_off  atIndex:1];
            [enc setBuffer:expert_bufs[k]                  offset:gate_b_off  atIndex:2];
            [enc setBuffer:ctx->buf_multi_expert_input     offset:0           atIndex:3];
            [enc setBuffer:ctx->buf_multi_expert_gate[k]   offset:0           atIndex:4];
            [enc setBytes:&gate_up_out length:4 atIndex:5];
            [enc setBytes:&gate_up_in  length:4 atIndex:6];
            [enc setBytes:&gs          length:4 atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(gate_up_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc setBuffer:expert_bufs[k]                  offset:up_w_off  atIndex:0];
            [enc setBuffer:expert_bufs[k]                  offset:up_s_off  atIndex:1];
            [enc setBuffer:expert_bufs[k]                  offset:up_b_off  atIndex:2];
            [enc setBuffer:ctx->buf_multi_expert_up[k]     offset:0          atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(gate_up_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        // Encoder B: SwiGLU + down_proj
        {
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            [enc setComputePipelineState:ctx->swiglu];
            [enc setBuffer:ctx->buf_multi_expert_gate[k] offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_multi_expert_up[k]   offset:0 atIndex:1];
            [enc setBuffer:ctx->buf_multi_expert_act[k]  offset:0 atIndex:2];
            [enc setBytes:&gate_up_out length:4 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc setComputePipelineState:expert_pipe];
            [enc setBuffer:expert_bufs[k]                  offset:down_w_off  atIndex:0];
            [enc setBuffer:expert_bufs[k]                  offset:down_s_off  atIndex:1];
            [enc setBuffer:expert_bufs[k]                  offset:down_b_off  atIndex:2];
            [enc setBuffer:ctx->buf_multi_expert_act[k]    offset:0           atIndex:3];
            [enc setBuffer:ctx->buf_multi_expert_out[k]    offset:0           atIndex:4];
            [enc setBytes:&down_out length:4 atIndex:5];
            [enc setBytes:&down_in  length:4 atIndex:6];
            [enc setBytes:&gs       length:4 atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(down_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }
    }
}
```

- [ ] **Step 4: Update call site of `gpu_encode_experts_batched`**

At line ~5637, update the call to pass `layer_idx` and `expert_indices`:

```c
// Old:
gpu_encode_experts_batched(g_metal, cmd_experts, actual_K, valid, expert_bufs);

// New:
gpu_encode_experts_batched(g_metal, cmd_experts, actual_K, valid, expert_bufs,
                           layer_idx, expert_indices);
```

- [ ] **Step 5: Ensure Metal expert data buffers are sized for max (4-bit)**

In `metal_setup()` where `buf_multi_expert_data[k]` is allocated, ensure the buffer size uses `expert_size_4bit` (the larger size), not `active_expert_size()`, when in tiered mode:

```c
// Buffer allocation for expert data — must fit the largest expert
size_t expert_buf_size = g_use_tiered ? cfg.expert_size_4bit : active_expert_size();
for (int k = 0; k < cfg.num_experts_per_tok; k++) {
    ctx->buf_multi_expert_data[k] = [dev newBufferWithLength:expert_buf_size
                                         options:MTLResourceStorageModeShared];
}
```

- [ ] **Step 6: Update banner/status output**

In the startup banner section, add tiered mode indication:

```c
printf("Quant:    %s\n",
       g_use_tiered ? "tiered (hot=4-bit, cold=2-bit)" :
       g_use_2bit ? "2-bit experts" :
       "4-bit experts");
```

- [ ] **Step 7: Rebuild and verify compilation**

```bash
cd metal_infer && make 2>&1 | tail -5
```

Expected: Clean compile with 0 new warnings.

- [ ] **Step 8: Commit**

```bash
git add metal_infer/infer.m
git commit -m "feat: per-expert quantization dispatch for tiered mode"
```

---

### Task 5: Integration Testing and Benchmarking

**Files:**
- No new files

- [ ] **Step 1: Verify tiered mode generates correct output**

```bash
cd metal_infer

# 4-bit baseline
./infer --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit \
  --prompt "What is 2+2?" --tokens 20 2>&1 | grep -E "gen|Output"

# Tiered mode (auto-detected from packed_experts_tiered/)
./infer --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit \
  --tiered --prompt "What is 2+2?" --tokens 20 2>&1 | grep -E "gen|Output"
```

Expected: Both produce "4" or equivalent correct answer. Tiered output shows `[tiered]` in startup.

- [ ] **Step 2: Benchmark 4-bit baseline**

```bash
./infer --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit \
  --prompt "Explain quantum computing in detail" --tokens 200 --timing 2>&1 | tail -20
```

Record: tok/s, avg layer time, expert I/O time.

- [ ] **Step 3: Benchmark tiered mode**

```bash
./infer --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit \
  --tiered --prompt "Explain quantum computing in detail" --tokens 200 --timing 2>&1 | tail -20
```

Record: tok/s, avg layer time, expert I/O time. Compare with baseline.

- [ ] **Step 4: Verify tool calling still works (4-bit hot experts handle JSON)**

```bash
./infer --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit \
  --tiered --prompt 'Call the get_weather function for Tokyo with units celsius. Respond only with valid JSON: {"name": "get_weather", "arguments": {...}}' \
  --tokens 50 2>&1
```

Expected: Valid JSON output (the key quality metric — 2-bit breaks JSON, but tiered should preserve it since hot experts handle structured output).

- [ ] **Step 5: Run with `--freq` to verify expert routing unchanged**

```bash
./infer --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit \
  --tiered --prompt "Explain relativity" --tokens 100 --freq 2>&1 | grep "Overall Summary"
```

Expected: Similar 80% coverage numbers as baseline (routing is independent of quant).

- [ ] **Step 6: Commit benchmark results**

```bash
# Add results to results.tsv
echo -e "TIERED: hot=4-bit cold=2-bit, 80% threshold, 35B-A3B. X.XX tok/s (vs Y.YY baseline). ZZ% disk reduction.\tKEPT" >> ../results.tsv
git add ../results.tsv
git commit -m "bench: tiered expert quantization results"
```

---

### Task 6: Backward Compatibility Verification

**Files:**
- No new files

Verify all three modes still work: plain 4-bit, plain 2-bit, and tiered.

- [ ] **Step 1: Test plain 4-bit (no tiered directory)**

```bash
# Temporarily rename tiered dir to test fallback
MODEL=~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit
mv "$MODEL/packed_experts_tiered" "$MODEL/packed_experts_tiered.bak"

cd metal_infer && ./infer --model "$MODEL" \
  --prompt "Hello" --tokens 10 2>&1 | grep -E "Quant|experts.*available"

mv "$MODEL/packed_experts_tiered.bak" "$MODEL/packed_experts_tiered"
```

Expected: `Quant: 4-bit experts`, no tiered messages.

- [ ] **Step 2: Test explicit `--2bit` flag overrides tiered**

```bash
cd metal_infer && ./infer --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit \
  --2bit --prompt "Hello" --tokens 10 2>&1 | grep -E "Quant|tiered"
```

Expected: `Quant: 2-bit experts`, no tiered (--2bit takes priority).

- [ ] **Step 3: Test explicit `--tiered` flag**

```bash
cd metal_infer && ./infer --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit \
  --tiered --prompt "Hello" --tokens 10 2>&1 | grep -E "Quant|tiered"
```

Expected: `Quant: tiered (hot=4-bit, cold=2-bit)`.

- [ ] **Step 4: Final commit with any fixes**

```bash
git add -A && git commit -m "fix: backward compatibility for all quant modes"
```
