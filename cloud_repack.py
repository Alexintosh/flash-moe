"""
Cloud repack pipeline for Flash-MoE models.

Downloads an MLX model from HuggingFace, repacks expert weights into
per-layer binary files, extracts non-expert weights (with optional split
for iOS Metal 4GB limit), and uploads the result to a new HF repo.

Runs on Modal (https://modal.com) — no local disk or GPU needed.

Usage:
    # Install modal client
    pip install modal
    modal setup  # one-time auth

    # Repack the 122B model
    modal run cloud_repack.py --source mlx-community/Qwen3.5-122B-A10B-4bit \
                              --dest alexintosh/Qwen3.5-122B-A10B-Q4-FlashMoE \
                              --split 3.5

    # Repack the 35B model
    modal run cloud_repack.py --source mlx-community/Qwen3.5-35B-A3B-4bit \
                              --dest alexintosh/Qwen3.5-35B-A3B-Q4-FlashMoE

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
    )
)


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


@app.local_entrypoint()
def main(
    source: str = "mlx-community/Qwen3.5-122B-A10B-4bit",
    dest: str = "alexintosh/Qwen3.5-122B-A10B-Q4-FlashMoE",
    split: float = 0,
    hf_token: str = "",
):
    """Repack a HuggingFace MLX model for Flash-MoE and upload."""
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN", "")

    print(f"Source: {source}")
    print(f"Dest:   {dest}")
    print(f"Split:  {split} GB" if split > 0 else "Split:  disabled")
    print()

    repack_model.remote(source=source, dest=dest, split_gb=split, hf_token=hf_token)
