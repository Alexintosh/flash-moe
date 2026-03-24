"""
Cloud repack pipeline for Flash-MoE models.

Downloads an MLX model from HuggingFace, repacks expert weights into
per-layer binary files, extracts non-expert weights (with optional split
for iOS Metal 4GB limit), and uploads the result to a new HF repo.

Supports three modes:
  - 4bit (default): All experts at 4-bit quantization.
  - tiered: Hot experts at 4-bit, cold experts requantized to 2-bit (MSE-optimal).
  - gptq: GPTQ error-compensated 2-bit (placeholder, requires calibration data).

Runs on Modal (https://modal.com) — no local disk or GPU needed.

Usage:
    # Install modal client
    pip install modal
    modal setup  # one-time auth

    # 4-bit repack (default)
    modal run cloud_repack.py --source mlx-community/Qwen3.5-122B-A10B-4bit \
                              --dest alexintosh/Qwen3.5-122B-A10B-Q4-FlashMoE \
                              --split 3.5

    # Tiered repack (top 20% hot at 4-bit, rest at 2-bit)
    modal run cloud_repack.py --source mlx-community/Qwen3.5-35B-A3B-4bit \
                              --dest alexintosh/Qwen3.5-35B-A3B-Q4-Tiered-FlashMoE \
                              --mode tiered --hot-ratio 0.2 --split 3.5

    # Repack 397B (needs large volume)
    modal run cloud_repack.py --source mlx-community/Qwen3.5-397B-A17B-4bit \
                              --dest alexintosh/Qwen3.5-397B-A17B-Q4-FlashMoE \
                              --split 3.5 --volume-size 500
"""

import modal
import os

app = modal.App("flash-moe-repack")

# Volume for storing model data during repacking
vol = modal.Volume.from_name("flash-moe-repack-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface_hub",
        "safetensors",
        "numpy",
        "torch",  # needed for safetensors loading
        "scipy",  # for future GPTQ support
    )
)

# Higher-resource image config for tiered repacking (needs more RAM for requantization)
tiered_image = image


@app.function(
    image=image,
    volumes={"/data": vol},
    timeout=7200,  # 2 hours max
    memory=16384,  # 16 GB RAM
    cpu=4,
)
def repack_model(source: str, dest: str, split_gb: float = 0, hf_token: str = ""):
    """Download, repack, and upload a Flash-MoE model."""
    import json
    import struct
    import time
    from pathlib import Path
    from collections import defaultdict
    from huggingface_hub import HfApi, snapshot_download

    WORK = Path("/data/work")
    MODEL_DIR = WORK / "model"
    OUTPUT_DIR = WORK / "output"

    # Clean previous runs
    import shutil
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Step 1: Download model from HuggingFace
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Step 1: Downloading {source}")
    print(f"{'='*60}")

    t0 = time.time()
    model_path = Path(snapshot_download(
        source,
        local_dir=str(MODEL_DIR),
        token=hf_token or None,
    ))
    print(f"Downloaded in {time.time()-t0:.0f}s to {model_path}")

    # ----------------------------------------------------------------
    # Step 2: Parse config
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Step 2: Parsing model config")
    print(f"{'='*60}")

    config = json.load(open(model_path / "config.json"))
    tc = config.get("text_config", config)

    num_layers = tc["num_hidden_layers"]
    num_experts = tc["num_experts"]
    hidden_size = tc["hidden_size"]
    moe_intermediate = tc["moe_intermediate_size"]
    group_size = config.get("quantization", {}).get("group_size", 64)
    bits = config.get("quantization", {}).get("bits", 4)

    print(f"  Layers: {num_layers}")
    print(f"  Experts: {num_experts}")
    print(f"  Hidden: {hidden_size}")
    print(f"  MoE intermediate: {moe_intermediate}")
    print(f"  Quantization: {bits}-bit, group_size={group_size}")

    # Compute expert size
    vals_per_u32 = 32 // bits
    # gate_proj / up_proj: [moe_intermediate, hidden_size]
    gate_w = moe_intermediate * ((hidden_size + vals_per_u32 - 1) // vals_per_u32) * 4
    gate_s = moe_intermediate * ((hidden_size + group_size - 1) // group_size) * 2
    gate_b = gate_s
    # down_proj: [hidden_size, moe_intermediate]
    down_w = hidden_size * ((moe_intermediate + vals_per_u32 - 1) // vals_per_u32) * 4
    down_s = hidden_size * ((moe_intermediate + group_size - 1) // group_size) * 2
    down_b = down_s

    expert_size = 3 * (gate_w + gate_s + gate_b)  # gate + up (same shape) + down (same total)
    # Actually: gate + up have same layout, down is different
    expert_size = 2 * (gate_w + gate_s + gate_b) + (down_w + down_s + down_b)

    print(f"  Expert size: {expert_size} bytes ({expert_size/1e6:.2f} MB)")
    print(f"  Total expert data: {expert_size * num_experts * num_layers / 1e9:.1f} GB")

    # Component layout within each expert block
    components = []
    off = 0
    for proj, rows, cols in [("gate_proj", moe_intermediate, hidden_size),
                              ("up_proj", moe_intermediate, hidden_size),
                              ("down_proj", hidden_size, moe_intermediate)]:
        w_size = rows * ((cols + vals_per_u32 - 1) // vals_per_u32) * 4
        s_size = rows * ((cols + group_size - 1) // group_size) * 2
        b_size = s_size
        components.append((f"{proj}.weight", off, w_size)); off += w_size
        components.append((f"{proj}.scales", off, s_size)); off += s_size
        components.append((f"{proj}.biases", off, b_size)); off += b_size

    assert off == expert_size, f"Component layout mismatch: {off} != {expert_size}"

    # ----------------------------------------------------------------
    # Step 3: Build expert index from safetensors
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Step 3: Building expert index")
    print(f"{'='*60}")

    index_file = model_path / "model.safetensors.index.json"
    with open(index_file) as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]

    # Parse safetensors headers
    header_cache = {}
    def get_header(filename):
        if filename not in header_cache:
            filepath = model_path / filename
            with open(filepath, 'rb') as f:
                header_len = struct.unpack('<Q', f.read(8))[0]
                header = json.loads(f.read(header_len))
                data_start = 8 + header_len
            header_cache[filename] = (header, data_start)
        return header_cache[filename]

    # Build expert reads: for each layer/expert/component, record file + offset + size
    import re
    expert_pattern = re.compile(
        r'language_model\.model\.layers\.(\d+)\.mlp\.switch_mlp\.'
        r'(gate_proj|up_proj|down_proj)\.(weight|scales|biases)'
    )

    expert_reads = {}  # (layer, expert_idx, component_name) -> (file, offset, size)

    for tensor_name, filename in weight_map.items():
        m = expert_pattern.match(tensor_name)
        if not m:
            continue
        layer = int(m.group(1))
        proj = m.group(2)
        attr = m.group(3)
        comp_name = f"{proj}.{attr}"

        header, data_start = get_header(filename)
        meta = header[tensor_name]
        t_off = meta['data_offsets']
        byte_len = t_off[1] - t_off[0]
        shape = meta['shape']

        # shape[0] = num_experts, rest = per-expert dims
        per_expert_size = byte_len // shape[0]

        for e in range(shape[0]):
            key = (layer, e, comp_name)
            expert_reads[key] = {
                'file': filename,
                'offset': data_start + t_off[0] + e * per_expert_size,
                'size': per_expert_size,
            }

    print(f"  Indexed {len(expert_reads)} expert components across {num_layers} layers")

    # ----------------------------------------------------------------
    # Step 4: Repack experts into per-layer binary files
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Step 4: Repacking experts")
    print(f"{'='*60}")

    expert_dir = OUTPUT_DIR / "packed_experts"
    expert_dir.mkdir(exist_ok=True)

    # Open all needed safetensors files
    file_handles = {}
    for key, info in expert_reads.items():
        fname = info['file']
        if fname not in file_handles:
            file_handles[fname] = open(model_path / fname, 'rb')

    t_repack = time.time()
    for layer in range(num_layers):
        layer_path = expert_dir / f"layer_{layer:02d}.bin"
        t_layer = time.time()

        with open(layer_path, 'wb') as out_f:
            for expert in range(num_experts):
                expert_block = bytearray(expert_size)

                for comp_name, comp_offset, comp_size in components:
                    key = (layer, expert, comp_name)
                    if key not in expert_reads:
                        print(f"  WARNING: missing {key}")
                        continue
                    info = expert_reads[key]
                    fh = file_handles[info['file']]
                    fh.seek(info['offset'])
                    data = fh.read(info['size'])
                    assert len(data) == comp_size, f"Size mismatch for {key}: {len(data)} != {comp_size}"
                    expert_block[comp_offset:comp_offset+comp_size] = data

                out_f.write(expert_block)

        layer_size = os.path.getsize(layer_path)
        layer_ms = (time.time() - t_layer) * 1000
        print(f"  Layer {layer:2d}/{num_layers}: {layer_size/1e9:.2f} GB ({layer_ms:.0f}ms)")

    # Write layout.json
    layout = {
        "expert_size": expert_size,
        "num_experts": num_experts,
        "num_layers": num_layers,
        "components": [{"name": c[0], "offset": c[1], "size": c[2]} for c in components],
    }
    with open(expert_dir / "layout.json", 'w') as f:
        json.dump(layout, f, indent=2)

    for fh in file_handles.values():
        fh.close()

    repack_time = time.time() - t_repack
    total_expert_gb = sum(os.path.getsize(expert_dir / f) for f in os.listdir(expert_dir)) / 1e9
    print(f"\n  Repacked {num_layers} layers ({total_expert_gb:.1f} GB) in {repack_time:.0f}s")

    # ----------------------------------------------------------------
    # Step 5: Extract non-expert weights
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Step 5: Extracting non-expert weights")
    print(f"{'='*60}")

    expert_tensor_pattern = re.compile(r'\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$')
    vision_pattern = re.compile(r'^(vision_tower|model\.visual)')

    tensors_to_extract = {}
    for name, filename in weight_map.items():
        if vision_pattern.match(name):
            continue
        if expert_tensor_pattern.search(name):
            continue
        tensors_to_extract[name] = filename

    def sanitize_name(name):
        if name.startswith("language_model."):
            return name[len("language_model."):]
        return name

    all_tensors = sorted([(sanitize_name(n), n, tensors_to_extract[n]) for n in tensors_to_extract])

    ALIGN = 64
    split_bytes = int(split_gb * 1e9) if split_gb > 0 else 0

    manifest = {
        "model": source,
        "num_tensors": len(all_tensors),
        "tensors": {},
        "config": {
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
            "num_attention_heads": tc.get("num_attention_heads", 32),
            "num_key_value_heads": tc.get("num_key_value_heads", 2),
            "head_dim": tc.get("head_dim", 128),
            "vocab_size": tc.get("vocab_size", 248320),
            "rms_norm_eps": tc.get("rms_norm_eps", 1e-6),
            "num_experts": num_experts,
            "num_experts_per_tok": tc.get("num_experts_per_tok", 8),
            "moe_intermediate_size": moe_intermediate,
            "shared_expert_intermediate_size": tc.get("shared_expert_intermediate_size", moe_intermediate),
            "linear_num_value_heads": tc.get("linear_num_value_heads", 32),
            "linear_num_key_heads": tc.get("linear_num_key_heads", 16),
            "linear_key_head_dim": tc.get("linear_key_head_dim", 128),
            "linear_value_head_dim": tc.get("linear_value_head_dim", 128),
            "linear_conv_kernel_dim": tc.get("linear_conv_kernel_dim", 4),
            "partial_rotary_factor": tc.get("rope_parameters", {}).get("partial_rotary_factor",
                                     tc.get("partial_rotary_factor", 0.5)),
            "rope_theta": tc.get("rope_parameters", {}).get("rope_theta",
                          tc.get("rope_theta", 10000000.0)),
        }
    }

    # Layer types
    layer_types = tc.get("layer_types", None)
    if layer_types is None:
        interval = tc.get("full_attention_interval", 4)
        layer_types = []
        for i in range(num_layers):
            if (i + 1) % interval == 0:
                layer_types.append("full_attention")
            else:
                layer_types.append("linear_attention")
    manifest["config"]["layer_types"] = layer_types

    # Write weight files
    offset = 0
    total_bytes = 0
    chunk_idx = 0
    chunk_offset = 0
    chunk_paths = []
    split_offsets = [0]

    cur_path = OUTPUT_DIR / "model_weights.bin"
    chunk_paths.append(cur_path)
    out_f = open(cur_path, 'wb')

    for i, (san_name, orig_name, filename) in enumerate(all_tensors):
        header, data_start = get_header(filename)

        if orig_name not in header:
            continue

        meta = header[orig_name]
        t_off = meta['data_offsets']
        byte_len = t_off[1] - t_off[0]

        # Split check
        if split_bytes > 0 and chunk_offset > 0 and chunk_offset + byte_len + ALIGN > split_bytes:
            out_f.close()
            chunk_idx += 1
            cur_path = OUTPUT_DIR / f"model_weights_{chunk_idx}.bin"
            chunk_paths.append(cur_path)
            split_offsets.append(offset)
            out_f = open(cur_path, 'wb')
            chunk_offset = 0

        # Align
        if offset % ALIGN != 0:
            pad = ALIGN - (offset % ALIGN)
            out_f.write(b'\x00' * pad)
            offset += pad
            chunk_offset += pad

        # Read and write tensor
        with open(model_path / filename, 'rb') as sf:
            sf.seek(data_start + t_off[0])
            data = sf.read(byte_len)

        out_f.write(data)

        manifest["tensors"][san_name] = {
            "offset": offset,
            "size": byte_len,
            "shape": meta['shape'],
            "dtype": meta['dtype'],
        }

        offset += byte_len
        chunk_offset += byte_len
        total_bytes += byte_len

        if (i + 1) % 200 == 0 or i == len(all_tensors) - 1:
            print(f"  [{i+1}/{len(all_tensors)}] {total_bytes/1e9:.2f} GB")

    out_f.close()

    if len(chunk_paths) > 1:
        manifest["split"] = {
            "num_chunks": len(chunk_paths),
            "chunk_files": [p.name for p in chunk_paths],
            "split_offsets": split_offsets,
        }

    # Write manifest
    with open(OUTPUT_DIR / "model_weights.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Extracted {total_bytes/1e9:.2f} GB in {len(chunk_paths)} chunk(s):")
    for p in chunk_paths:
        print(f"    {p.name}: {os.path.getsize(p)/1e9:.2f} GB")

    # ----------------------------------------------------------------
    # Step 6: Copy config + tokenizer files
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Step 6: Copying config and tokenizer")
    print(f"{'='*60}")

    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
        src = model_path / fname
        if src.exists():
            shutil.copy2(src, OUTPUT_DIR / fname)
            print(f"  Copied {fname}")

    # ----------------------------------------------------------------
    # Step 7: Upload to HuggingFace
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Step 7: Uploading to {dest}")
    print(f"{'='*60}")

    api = HfApi(token=hf_token or None)
    api.create_repo(dest, repo_type="model", exist_ok=True)

    # Write README
    readme = f"""---
license: apache-2.0
tags:
  - flash-moe
  - qwen3
  - moe
  - metal
  - apple-silicon
  - ios
---

# {dest.split('/')[-1]} — Pre-packed for Flash-MoE

Pre-packed weights for [Flash-MoE](https://github.com/Alexintosh/flash-moe) inference engine.
Source model: [{source}](https://huggingface.co/{source})

## File Format

All `.bin` files are raw numeric arrays (packed quantized weights + float16 scales/biases).
No pickle, no executable code.

## Contents

- `config.json` — Model architecture
- `model_weights*.bin` — Non-expert weights ({total_bytes/1e9:.1f} GB total{f', split into {len(chunk_paths)} chunks for iOS Metal 4GB limit' if len(chunk_paths) > 1 else ''})
- `model_weights.json` — Tensor manifest
- `packed_experts/layer_XX.bin` — Per-layer expert weights ({num_layers} files)

## Usage

```bash
git clone https://github.com/Alexintosh/flash-moe
cd flash-moe/metal_infer && make
./infer --model /path/to/this/repo --prompt "Hello" --tokens 100
```
"""
    with open(OUTPUT_DIR / "README.md", 'w') as f:
        f.write(readme)

    # Upload
    t_upload = time.time()
    api.upload_folder(
        folder_path=str(OUTPUT_DIR),
        repo_id=dest,
        repo_type="model",
        commit_message=f"Pre-packed Flash-MoE weights from {source}",
    )
    print(f"  Uploaded in {time.time()-t_upload:.0f}s")

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE in {total_time/60:.0f} minutes")
    print(f"Model available at: https://huggingface.co/{dest}")
    print(f"{'='*60}")


@app.function(
    image=tiered_image,
    volumes={"/data": vol},
    timeout=14400,  # 4 hours max (tiered requant is slower)
    memory=32768,   # 32 GB RAM (needs 4-bit data + 2-bit requant buffers)
    cpu=8,          # parallel requantization
)
def repack_tiered(
    source: str,
    dest: str,
    split_gb: float = 0,
    hf_token: str = "",
    hot_ratio: float = 0.2,
    freq_file: str = "",
    freq_data_json: str = "",  # JSON string of freq data (uploaded from local)
):
    """Download, repack with tiered quantization (hot=4-bit, cold=2-bit), and upload.

    Hot experts are kept at 4-bit. Cold experts are requantized to 2-bit using
    MSE-optimal clipping. The hot set is determined by either:
      - top `hot_ratio` fraction of experts by index (default: top 20%)
      - explicit frequency file (--freq-file path to hot_experts.json)
    """
    import json
    import struct
    import time
    import re
    import shutil
    import numpy as np
    from pathlib import Path
    from huggingface_hub import HfApi, snapshot_download

    # ==================================================================
    # Inlined 2-bit requantization helpers (self-contained for Modal)
    # ==================================================================

    def bf16_to_f32(bf16_arr):
        """Convert uint16 bf16 bit patterns to float32."""
        return (bf16_arr.astype(np.uint32) << 16).view(np.float32)

    def f32_to_bf16(f32_arr):
        """Convert float32 to uint16 bf16 bit patterns (truncation, no rounding)."""
        return (f32_arr.view(np.uint32) >> 16).astype(np.uint16)

    def unpack_4bit(packed, out_dim, in_dim):
        """Extract 8 x 4-bit nibbles per uint32, LSB-first."""
        packed_cols = in_dim // 8
        result = np.zeros((out_dim, in_dim), dtype=np.uint8)
        for i in range(8):
            result[:, i::8] = (packed[:, :packed_cols] >> (i * 4)) & 0xF
        return result

    def pack_2bit(vals, out_dim, in_dim):
        """Pack 16 x 2-bit values per uint32, LSB-first."""
        packed_cols = in_dim // 16
        result = np.zeros((out_dim, packed_cols), dtype=np.uint32)
        for i in range(16):
            result |= vals[:, i::16].astype(np.uint32) << (i * 2)
        return result

    def requantize_to_2bit(packed_4bit, scales_bf16, biases_bf16, out_dim, in_dim, group_size=64):
        """Requantize a 4-bit projection to 2-bit with MSE-optimal clipping."""
        # 1. Unpack 4-bit
        vals_4bit = unpack_4bit(packed_4bit, out_dim, in_dim)
        # 2. Dequantize
        scales = bf16_to_f32(scales_bf16)
        biases = bf16_to_f32(biases_bf16)
        num_groups = in_dim // group_size
        vals_grouped = vals_4bit.reshape(out_dim, num_groups, group_size).astype(np.float32)
        s = scales[:, :, np.newaxis]
        b = biases[:, :, np.newaxis]
        dequant = vals_grouped * s + b
        # 3. MSE-optimal clipping
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
            recon = q_try * s_try + c_min
            mse = np.mean((dequant - recon)**2, axis=2, keepdims=True)
            improved = mse < best_mse
            best_mse = np.where(improved, mse, best_mse)
            best_s2 = np.where(improved, s_try, best_s2)
            best_b2 = np.where(improved, c_min, best_b2)
        s2, b2 = best_s2, best_b2
        s2_safe = np.where(s2 == 0, 1.0, s2)
        vals_2bit = np.clip(np.round((dequant - b2) / s2_safe), 0, 3).astype(np.uint8)
        vals_2bit_flat = vals_2bit.reshape(out_dim, in_dim)
        packed_2bit = pack_2bit(vals_2bit_flat, out_dim, in_dim)
        new_scales = f32_to_bf16(s2.squeeze(2))
        new_biases = f32_to_bf16(b2.squeeze(2))
        return packed_2bit, new_scales, new_biases

    # ==================================================================
    # Setup
    # ==================================================================

    WORK = Path("/data/work")
    MODEL_DIR = WORK / "model"
    OUTPUT_DIR = WORK / "output"

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Download model
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Step 1: Downloading {source}")
    print(f"{'='*60}")

    t0 = time.time()
    model_path = Path(snapshot_download(
        source,
        local_dir=str(MODEL_DIR),
        token=hf_token or None,
    ))
    print(f"Downloaded in {time.time()-t0:.0f}s to {model_path}")

    # ------------------------------------------------------------------
    # Step 2: Parse config
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Step 2: Parsing model config")
    print(f"{'='*60}")

    config = json.load(open(model_path / "config.json"))
    tc = config.get("text_config", config)

    num_layers = tc["num_hidden_layers"]
    num_experts = tc["num_experts"]
    hidden_size = tc["hidden_size"]
    moe_intermediate = tc["moe_intermediate_size"]
    group_size = config.get("quantization", {}).get("group_size", 64)
    bits = config.get("quantization", {}).get("bits", 4)

    print(f"  Layers: {num_layers}")
    print(f"  Experts: {num_experts}")
    print(f"  Hidden: {hidden_size}")
    print(f"  MoE intermediate: {moe_intermediate}")
    print(f"  Quantization: {bits}-bit, group_size={group_size}")

    # Compute 4-bit expert size
    vals_per_u32_4 = 32 // 4  # always 8 for source 4-bit
    gate_w_4 = moe_intermediate * ((hidden_size + vals_per_u32_4 - 1) // vals_per_u32_4) * 4
    gate_s_4 = moe_intermediate * ((hidden_size + group_size - 1) // group_size) * 2
    gate_b_4 = gate_s_4
    down_w_4 = hidden_size * ((moe_intermediate + vals_per_u32_4 - 1) // vals_per_u32_4) * 4
    down_s_4 = hidden_size * ((moe_intermediate + group_size - 1) // group_size) * 2
    down_b_4 = down_s_4
    expert_size_4bit = 2 * (gate_w_4 + gate_s_4 + gate_b_4) + (down_w_4 + down_s_4 + down_b_4)

    # Compute 2-bit expert size
    vals_per_u32_2 = 32 // 2  # 16 for 2-bit
    gate_w_2 = moe_intermediate * ((hidden_size + vals_per_u32_2 - 1) // vals_per_u32_2) * 4
    gate_s_2 = gate_s_4  # scales/biases same shape (group_size preserved)
    gate_b_2 = gate_b_4
    down_w_2 = hidden_size * ((moe_intermediate + vals_per_u32_2 - 1) // vals_per_u32_2) * 4
    down_s_2 = down_s_4
    down_b_2 = down_b_4
    expert_size_2bit = 2 * (gate_w_2 + gate_s_2 + gate_b_2) + (down_w_2 + down_s_2 + down_b_2)

    print(f"  4-bit expert size: {expert_size_4bit} bytes ({expert_size_4bit/1e6:.2f} MB)")
    print(f"  2-bit expert size: {expert_size_2bit} bytes ({expert_size_2bit/1e6:.2f} MB)")

    # Component layouts for 4-bit (source) and 2-bit (target)
    components_4bit = []
    off = 0
    for proj, rows, cols in [("gate_proj", moe_intermediate, hidden_size),
                              ("up_proj", moe_intermediate, hidden_size),
                              ("down_proj", hidden_size, moe_intermediate)]:
        w_size = rows * ((cols + vals_per_u32_4 - 1) // vals_per_u32_4) * 4
        s_size = rows * ((cols + group_size - 1) // group_size) * 2
        b_size = s_size
        components_4bit.append((f"{proj}.weight", off, w_size, rows, cols)); off += w_size
        components_4bit.append((f"{proj}.scales", off, s_size, rows, cols)); off += s_size
        components_4bit.append((f"{proj}.biases", off, b_size, rows, cols)); off += b_size
    assert off == expert_size_4bit

    components_2bit = []
    off = 0
    for proj, rows, cols in [("gate_proj", moe_intermediate, hidden_size),
                              ("up_proj", moe_intermediate, hidden_size),
                              ("down_proj", hidden_size, moe_intermediate)]:
        w_size = rows * ((cols + vals_per_u32_2 - 1) // vals_per_u32_2) * 4
        s_size = rows * ((cols + group_size - 1) // group_size) * 2
        b_size = s_size
        components_2bit.append((f"{proj}.weight", off, w_size, rows, cols)); off += w_size
        components_2bit.append((f"{proj}.scales", off, s_size, rows, cols)); off += s_size
        components_2bit.append((f"{proj}.biases", off, b_size, rows, cols)); off += b_size
    assert off == expert_size_2bit

    # Projection descriptors for requantization: (proj_name, out_dim, in_dim)
    proj_descs = [
        ("gate_proj", moe_intermediate, hidden_size),
        ("up_proj", moe_intermediate, hidden_size),
        ("down_proj", hidden_size, moe_intermediate),
    ]

    # ------------------------------------------------------------------
    # Step 3: Build expert index from safetensors
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Step 3: Building expert index")
    print(f"{'='*60}")

    index_file = model_path / "model.safetensors.index.json"
    with open(index_file) as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]

    header_cache = {}
    def get_header(filename):
        if filename not in header_cache:
            filepath = model_path / filename
            with open(filepath, 'rb') as f:
                header_len = struct.unpack('<Q', f.read(8))[0]
                header = json.loads(f.read(header_len))
                data_start = 8 + header_len
            header_cache[filename] = (header, data_start)
        return header_cache[filename]

    expert_pattern = re.compile(
        r'language_model\.model\.layers\.(\d+)\.mlp\.switch_mlp\.'
        r'(gate_proj|up_proj|down_proj)\.(weight|scales|biases)'
    )

    expert_reads = {}
    for tensor_name, filename in weight_map.items():
        m = expert_pattern.match(tensor_name)
        if not m:
            continue
        layer = int(m.group(1))
        proj = m.group(2)
        attr = m.group(3)
        comp_name = f"{proj}.{attr}"

        header, data_start = get_header(filename)
        meta = header[tensor_name]
        t_off = meta['data_offsets']
        byte_len = t_off[1] - t_off[0]
        shape = meta['shape']
        per_expert_size = byte_len // shape[0]

        for e in range(shape[0]):
            key = (layer, e, comp_name)
            expert_reads[key] = {
                'file': filename,
                'offset': data_start + t_off[0] + e * per_expert_size,
                'size': per_expert_size,
            }

    print(f"  Indexed {len(expert_reads)} expert components across {num_layers} layers")

    # ------------------------------------------------------------------
    # Step 4: Determine hot expert set
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Step 4: Determining hot expert set")
    print(f"{'='*60}")

    num_hot = max(1, int(num_experts * hot_ratio))

    # Per-layer hot sets: maps layer_idx -> set of hot expert indices
    # If freq_file has per-layer data, use it. Otherwise use a single global set.
    per_layer_hot = {}
    hot_data = None

    # Try uploaded JSON string first (from local machine), then file path
    if freq_data_json:
        hot_data = json.loads(freq_data_json)
        print(f"  Using frequency data uploaded from local machine")
    elif freq_file:
        freq_path = Path(freq_file)
        if freq_path.exists():
            hot_data = json.load(open(freq_path))

            # Format 1: {"layers": {"0": [sorted expert indices], ...}} (from --freq-json)
            if isinstance(hot_data, dict) and "layers" in hot_data:
                print(f"  Using per-layer frequency data from {freq_file}")
                for l_str, expert_list in hot_data["layers"].items():
                    l = int(l_str)
                    per_layer_hot[l] = set(expert_list[:num_hot])

            # Format 2: {"hot": [list of indices]} — global hot set
            elif isinstance(hot_data, dict) and "hot" in hot_data:
                global_hot = set(hot_data["hot"][:num_hot])
                for l in range(num_layers):
                    per_layer_hot[l] = global_hot

            # Format 3: flat list of indices — global hot set
            elif isinstance(hot_data, list):
                global_hot = set(hot_data[:num_hot])
                for l in range(num_layers):
                    per_layer_hot[l] = global_hot

            else:
                print(f"  WARNING: Unrecognized freq file format, falling back to index-based")
        else:
            print(f"  WARNING: freq file {freq_file} not found, falling back to index-based")

    # Fallback: index-based (first num_hot experts are hot)
    if not per_layer_hot:
        global_hot = set(range(num_hot))
        for l in range(num_layers):
            per_layer_hot[l] = global_hot

    # Summary
    hot_counts = [len(per_layer_hot.get(l, set())) for l in range(num_layers)]
    avg_hot = sum(hot_counts) / len(hot_counts)
    avg_cold = num_experts - avg_hot
    print(f"  Hot experts (4-bit): avg {avg_hot:.0f}/layer ({avg_hot/num_experts:.0%})")
    print(f"  Cold experts (2-bit): avg {avg_cold:.0f}/layer ({avg_cold/num_experts:.0%})")
    if freq_file and hot_data and isinstance(hot_data, dict) and "layers" in hot_data:
        print(f"  Per-layer assignment: YES (different experts per layer)")

    # For backward compat
    hot_set = per_layer_hot.get(0, set(range(num_hot)))

    # ------------------------------------------------------------------
    # Step 5: Tiered repack — hot at 4-bit, cold requantized to 2-bit
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Step 5: Tiered repacking experts")
    print(f"{'='*60}")

    expert_dir = OUTPUT_DIR / "packed_experts_tiered"
    expert_dir.mkdir(exist_ok=True)

    # Open all needed safetensors files
    file_handles = {}
    for key, info in expert_reads.items():
        fname = info['file']
        if fname not in file_handles:
            file_handles[fname] = open(model_path / fname, 'rb')

    # Build tiered manifest
    tiered_manifest = {
        "expert_size_4bit": expert_size_4bit,
        "expert_size_2bit": expert_size_2bit,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "hot_ratio": hot_ratio,
        "hot_set": sorted(hot_set),
        "layers": {},
    }

    t_repack = time.time()
    total_hot = 0
    total_cold = 0

    for layer in range(num_layers):
        layer_path = expert_dir / f"layer_{layer:02d}.bin"
        t_layer = time.time()
        layer_manifest = []
        current_offset = 0

        with open(layer_path, 'wb') as out_f:
            for expert in range(num_experts):
                is_hot = expert in per_layer_hot.get(layer, set())

                # Read 4-bit expert data from safetensors
                expert_block_4bit = bytearray(expert_size_4bit)
                for comp_name, comp_offset, comp_size, _, _ in components_4bit:
                    key = (layer, expert, comp_name)
                    if key not in expert_reads:
                        print(f"  WARNING: missing {key}")
                        continue
                    info = expert_reads[key]
                    fh = file_handles[info['file']]
                    fh.seek(info['offset'])
                    data = fh.read(info['size'])
                    assert len(data) == comp_size, f"Size mismatch for {key}: {len(data)} != {comp_size}"
                    expert_block_4bit[comp_offset:comp_offset+comp_size] = data

                if is_hot:
                    # Keep 4-bit as-is
                    out_f.write(expert_block_4bit)
                    layer_manifest.append({
                        "offset": current_offset,
                        "size": expert_size_4bit,
                        "bits": 4,
                    })
                    current_offset += expert_size_4bit
                    total_hot += 1
                else:
                    # Requantize to 2-bit with MSE-optimal clipping
                    expert_block_2bit = bytearray(expert_size_2bit)

                    for proj_name, out_dim, in_dim in proj_descs:
                        # Find 4-bit component offsets
                        w_off_4 = s_off_4 = b_off_4 = 0
                        w_size_4 = s_size_4 = 0
                        for cn, co, cs, _, _ in components_4bit:
                            if cn == f"{proj_name}.weight":
                                w_off_4, w_size_4 = co, cs
                            elif cn == f"{proj_name}.scales":
                                s_off_4, s_size_4 = co, cs
                            elif cn == f"{proj_name}.biases":
                                b_off_4 = co

                        # Find 2-bit component offsets
                        w_off_2 = s_off_2 = b_off_2 = 0
                        w_size_2 = s_size_2 = 0
                        for cn, co, cs, _, _ in components_2bit:
                            if cn == f"{proj_name}.weight":
                                w_off_2, w_size_2 = co, cs
                            elif cn == f"{proj_name}.scales":
                                s_off_2, s_size_2 = co, cs
                            elif cn == f"{proj_name}.biases":
                                b_off_2 = co

                        packed_cols_4 = in_dim // 8
                        num_groups = in_dim // group_size

                        # Read 4-bit components
                        packed_4bit = np.frombuffer(
                            bytes(expert_block_4bit[w_off_4:w_off_4+w_size_4]),
                            dtype=np.uint32
                        ).reshape(out_dim, packed_cols_4)
                        scales_bf16 = np.frombuffer(
                            bytes(expert_block_4bit[s_off_4:s_off_4+s_size_4]),
                            dtype=np.uint16
                        ).reshape(out_dim, num_groups)
                        biases_bf16 = np.frombuffer(
                            bytes(expert_block_4bit[b_off_4:b_off_4+s_size_4]),
                            dtype=np.uint16
                        ).reshape(out_dim, num_groups)

                        # Requantize
                        packed_2bit, new_scales, new_biases = requantize_to_2bit(
                            packed_4bit, scales_bf16, biases_bf16,
                            out_dim, in_dim, group_size
                        )

                        # Write into 2-bit blob
                        w_data = packed_2bit.tobytes()
                        s_data = new_scales.tobytes()
                        b_data = new_biases.tobytes()
                        expert_block_2bit[w_off_2:w_off_2+len(w_data)] = w_data
                        expert_block_2bit[s_off_2:s_off_2+len(s_data)] = s_data
                        expert_block_2bit[b_off_2:b_off_2+len(b_data)] = b_data

                    out_f.write(expert_block_2bit)
                    layer_manifest.append({
                        "offset": current_offset,
                        "size": expert_size_2bit,
                        "bits": 2,
                    })
                    current_offset += expert_size_2bit
                    total_cold += 1

        tiered_manifest["layers"][str(layer)] = layer_manifest
        layer_size = os.path.getsize(layer_path)
        layer_ms = (time.time() - t_layer) * 1000
        print(f"  Layer {layer:2d}/{num_layers}: {layer_size/1e9:.2f} GB ({layer_ms:.0f}ms)")

    # Write tiered manifest
    with open(expert_dir / "tiered_manifest.json", 'w') as f:
        json.dump(tiered_manifest, f, indent=2)

    # Write layout.json (compatible with 4-bit-only tools)
    layout = {
        "expert_size_4bit": expert_size_4bit,
        "expert_size_2bit": expert_size_2bit,
        "num_experts": num_experts,
        "num_layers": num_layers,
        "tiered": True,
        "hot_ratio": hot_ratio,
        "components_4bit": [{"name": c[0], "offset": c[1], "size": c[2]} for c in components_4bit],
        "components_2bit": [{"name": c[0], "offset": c[1], "size": c[2]} for c in components_2bit],
    }
    with open(expert_dir / "layout.json", 'w') as f:
        json.dump(layout, f, indent=2)

    for fh in file_handles.values():
        fh.close()

    repack_time = time.time() - t_repack
    total_expert_gb = sum(
        os.path.getsize(expert_dir / f)
        for f in os.listdir(expert_dir)
        if f.endswith('.bin')
    ) / 1e9
    print(f"\n  Repacked {num_layers} layers ({total_expert_gb:.1f} GB) in {repack_time:.0f}s")
    print(f"  Hot experts: {total_hot} ({total_hot/(total_hot+total_cold):.0%})")
    print(f"  Cold experts: {total_cold} ({total_cold/(total_hot+total_cold):.0%})")

    # ------------------------------------------------------------------
    # Step 6: Extract non-expert weights (same as 4-bit mode)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Step 6: Extracting non-expert weights")
    print(f"{'='*60}")

    expert_tensor_pattern = re.compile(r'\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$')
    vision_pattern = re.compile(r'^(vision_tower|model\.visual)')

    tensors_to_extract = {}
    for name, filename in weight_map.items():
        if vision_pattern.match(name):
            continue
        if expert_tensor_pattern.search(name):
            continue
        tensors_to_extract[name] = filename

    def sanitize_name(name):
        if name.startswith("language_model."):
            return name[len("language_model."):]
        return name

    all_tensors = sorted([(sanitize_name(n), n, tensors_to_extract[n]) for n in tensors_to_extract])

    ALIGN = 64
    split_bytes = int(split_gb * 1e9) if split_gb > 0 else 0

    manifest = {
        "model": source,
        "num_tensors": len(all_tensors),
        "tensors": {},
        "config": {
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
            "num_attention_heads": tc.get("num_attention_heads", 32),
            "num_key_value_heads": tc.get("num_key_value_heads", 2),
            "head_dim": tc.get("head_dim", 128),
            "vocab_size": tc.get("vocab_size", 248320),
            "rms_norm_eps": tc.get("rms_norm_eps", 1e-6),
            "num_experts": num_experts,
            "num_experts_per_tok": tc.get("num_experts_per_tok", 8),
            "moe_intermediate_size": moe_intermediate,
            "shared_expert_intermediate_size": tc.get("shared_expert_intermediate_size", moe_intermediate),
            "linear_num_value_heads": tc.get("linear_num_value_heads", 32),
            "linear_num_key_heads": tc.get("linear_num_key_heads", 16),
            "linear_key_head_dim": tc.get("linear_key_head_dim", 128),
            "linear_value_head_dim": tc.get("linear_value_head_dim", 128),
            "linear_conv_kernel_dim": tc.get("linear_conv_kernel_dim", 4),
            "partial_rotary_factor": tc.get("rope_parameters", {}).get("partial_rotary_factor",
                                     tc.get("partial_rotary_factor", 0.5)),
            "rope_theta": tc.get("rope_parameters", {}).get("rope_theta",
                          tc.get("rope_theta", 10000000.0)),
        }
    }

    layer_types = tc.get("layer_types", None)
    if layer_types is None:
        interval = tc.get("full_attention_interval", 4)
        layer_types = []
        for i in range(num_layers):
            if (i + 1) % interval == 0:
                layer_types.append("full_attention")
            else:
                layer_types.append("linear_attention")
    manifest["config"]["layer_types"] = layer_types

    offset = 0
    total_bytes = 0
    chunk_idx = 0
    chunk_offset = 0
    chunk_paths = []
    split_offsets = [0]

    cur_path = OUTPUT_DIR / "model_weights.bin"
    chunk_paths.append(cur_path)
    out_f = open(cur_path, 'wb')

    for i, (san_name, orig_name, filename) in enumerate(all_tensors):
        header, data_start = get_header(filename)
        if orig_name not in header:
            continue
        meta = header[orig_name]
        t_off = meta['data_offsets']
        byte_len = t_off[1] - t_off[0]

        if split_bytes > 0 and chunk_offset > 0 and chunk_offset + byte_len + ALIGN > split_bytes:
            out_f.close()
            chunk_idx += 1
            cur_path = OUTPUT_DIR / f"model_weights_{chunk_idx}.bin"
            chunk_paths.append(cur_path)
            split_offsets.append(offset)
            out_f = open(cur_path, 'wb')
            chunk_offset = 0

        if offset % ALIGN != 0:
            pad = ALIGN - (offset % ALIGN)
            out_f.write(b'\x00' * pad)
            offset += pad
            chunk_offset += pad

        with open(model_path / filename, 'rb') as sf:
            sf.seek(data_start + t_off[0])
            data = sf.read(byte_len)

        out_f.write(data)
        manifest["tensors"][san_name] = {
            "offset": offset,
            "size": byte_len,
            "shape": meta['shape'],
            "dtype": meta['dtype'],
        }
        offset += byte_len
        chunk_offset += byte_len
        total_bytes += byte_len

        if (i + 1) % 200 == 0 or i == len(all_tensors) - 1:
            print(f"  [{i+1}/{len(all_tensors)}] {total_bytes/1e9:.2f} GB")

    out_f.close()

    if len(chunk_paths) > 1:
        manifest["split"] = {
            "num_chunks": len(chunk_paths),
            "chunk_files": [p.name for p in chunk_paths],
            "split_offsets": split_offsets,
        }

    with open(OUTPUT_DIR / "model_weights.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Extracted {total_bytes/1e9:.2f} GB in {len(chunk_paths)} chunk(s):")
    for p in chunk_paths:
        print(f"    {p.name}: {os.path.getsize(p)/1e9:.2f} GB")

    # ------------------------------------------------------------------
    # Step 7: Copy config + tokenizer files
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Step 7: Copying config and tokenizer")
    print(f"{'='*60}")

    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
        src = model_path / fname
        if src.exists():
            shutil.copy2(src, OUTPUT_DIR / fname)
            print(f"  Copied {fname}")

    # ------------------------------------------------------------------
    # Step 8: Upload to HuggingFace
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Step 8: Uploading to {dest}")
    print(f"{'='*60}")

    api = HfApi(token=hf_token or None)
    api.create_repo(dest, repo_type="model", exist_ok=True)

    hot_pct = len(hot_set) / num_experts * 100
    readme = f"""---
license: apache-2.0
tags:
  - flash-moe
  - qwen3
  - moe
  - metal
  - apple-silicon
  - ios
  - tiered-quantization
---

# {dest.split('/')[-1]} — Pre-packed Tiered Quantization for Flash-MoE

Pre-packed weights with **tiered quantization** for [Flash-MoE](https://github.com/Alexintosh/flash-moe) inference engine.
Source model: [{source}](https://huggingface.co/{source})

## Tiered Quantization

- **Hot experts ({hot_pct:.0f}%)**: Kept at 4-bit ({expert_size_4bit/1e6:.2f} MB each)
- **Cold experts ({100-hot_pct:.0f}%)**: Requantized to 2-bit with MSE-optimal clipping ({expert_size_2bit/1e6:.2f} MB each)

Hot experts preserve full quality for frequently-routed paths. Cold experts use MSE-optimal
clipping to minimize reconstruction error during 4-bit to 2-bit requantization.

## File Format

All `.bin` files are raw numeric arrays (packed quantized weights + float16 scales/biases).
No pickle, no executable code.

## Contents

- `config.json` — Model architecture
- `model_weights*.bin` — Non-expert weights ({total_bytes/1e9:.1f} GB total{f', split into {len(chunk_paths)} chunks for iOS Metal 4GB limit' if len(chunk_paths) > 1 else ''})
- `model_weights.json` — Tensor manifest
- `packed_experts_tiered/layer_XX.bin` — Per-layer tiered expert weights ({num_layers} files)
- `packed_experts_tiered/tiered_manifest.json` — Per-expert offset/size/bits manifest

## Usage

```bash
git clone https://github.com/Alexintosh/flash-moe
cd flash-moe/metal_infer && make
./infer --model /path/to/this/repo --prompt "Hello" --tokens 100
```
"""
    with open(OUTPUT_DIR / "README.md", 'w') as f:
        f.write(readme)

    t_upload = time.time()
    api.upload_folder(
        folder_path=str(OUTPUT_DIR),
        repo_id=dest,
        repo_type="model",
        commit_message=f"Pre-packed tiered Flash-MoE weights from {source} (hot_ratio={hot_ratio})",
    )
    print(f"  Uploaded in {time.time()-t_upload:.0f}s")

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE in {total_time/60:.0f} minutes")
    print(f"Model available at: https://huggingface.co/{dest}")
    print(f"{'='*60}")


@app.function(
    image=tiered_image,
    volumes={"/data": vol},
    timeout=14400,
    memory=32768,
    cpu=8,
)
def repack_gptq(
    source: str,
    dest: str,
    split_gb: float = 0,
    hf_token: str = "",
    hot_ratio: float = 0.2,
    hessian_volume: str = "",
):
    """GPTQ-quantized tiered repacking. Requires pre-computed Hessians.

    GPTQ (Generative Pre-trained Transformer Quantization) uses error compensation
    during quantization: each weight's rounding error is propagated to subsequent
    weights using the inverse Hessian, minimizing the overall output error.

    This produces significantly better 2-bit quantization than MSE-optimal clipping
    alone, but requires calibration data collected from actual model inference.

    Run calibration locally first:
        ./infer --collect-activations cal.bin --tokens 16384
        python build_hessian.py cal.bin --output-dir calibration/

    Then upload calibration data and run this function:
        modal run cloud_repack.py --mode gptq \
            --source mlx-community/Qwen3.5-397B-A17B-4bit \
            --dest alexintosh/Qwen3.5-397B-A17B-GPTQ-Tiered-FlashMoE \
            --hessian-volume calibration/
    """
    raise NotImplementedError(
        "GPTQ cloud repacking requires calibration data. "
        "Run calibration locally first, then use --hessian-volume.\n\n"
        "Steps:\n"
        "  1. ./infer --collect-activations cal.bin --tokens 16384\n"
        "  2. python build_hessian.py cal.bin --output-dir calibration/\n"
        "  3. modal run cloud_repack.py --mode gptq --hessian-volume calibration/ ..."
    )


@app.local_entrypoint()
def main(
    source: str = "mlx-community/Qwen3.5-122B-A10B-4bit",
    dest: str = "alexintosh/Qwen3.5-122B-A10B-Q4-FlashMoE",
    split: float = 0,
    hf_token: str = "",
    mode: str = "4bit",
    hot_ratio: float = 0.2,
    freq_file: str = "",
):
    """Repack a HuggingFace MLX model for Flash-MoE and upload.

    Modes:
      - 4bit (default): All experts at 4-bit quantization.
      - tiered: Hot experts at 4-bit, cold at 2-bit with MSE-optimal clipping.
      - gptq: GPTQ error-compensated 2-bit (requires calibration data).
    """
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN", "")

    print(f"Source: {source}")
    print(f"Dest:   {dest}")
    print(f"Mode:   {mode}")
    print(f"Split:  {split} GB" if split > 0 else "Split:  disabled")
    if mode in ("tiered", "gptq"):
        print(f"Hot ratio: {hot_ratio}")
        if freq_file:
            print(f"Freq file: {freq_file}")
    print()

    if mode == "4bit":
        repack_model.remote(source=source, dest=dest, split_gb=split, hf_token=hf_token)
    elif mode == "tiered":
        # Upload freq file contents to Modal (local path won't exist on remote)
        freq_data_json = ""
        if freq_file:
            local_freq = Path(freq_file)
            if local_freq.exists():
                freq_data_json = local_freq.read_text()
                print(f"Uploading freq data ({len(freq_data_json)} bytes) to Modal...")
            else:
                print(f"WARNING: freq file {freq_file} not found locally, using index-based assignment")
        repack_tiered.remote(
            source=source, dest=dest, split_gb=split, hf_token=hf_token,
            hot_ratio=hot_ratio, freq_data_json=freq_data_json,
        )
    elif mode == "gptq":
        repack_gptq.remote(
            source=source, dest=dest, split_gb=split, hf_token=hf_token,
            hot_ratio=hot_ratio,
        )
    else:
        print(f"ERROR: Unknown mode '{mode}'. Must be one of: 4bit, tiered, gptq")
