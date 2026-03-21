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


def parse_raw_freq_dump(stderr: str) -> dict:
    """Parse raw frequency dump (FREQ_DUMP) into per-layer expert frequencies.

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
            if ":" not in pair:
                continue
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
        print("ERROR: No FREQ_DUMP data found in output.")
        print("       Make sure infer was built with --freq-dump support (freq_print_analysis raw dump).")
        sys.exit(1)

    # Read model config for num_experts
    num_experts = 256  # default
    if args.model:
        config_path = Path(args.model) / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            tc = config.get("text_config", config)
            num_experts = tc.get("num_experts", 256)

    # If multiple runs, aggregate frequencies across all runs
    # parse_raw_freq_dump returns last run's data; for aggregation we need
    # to sum across all FREQ_DUMP blocks
    all_freqs = {}
    for line in stderr.splitlines():
        m = re.match(r"FREQ_DUMP layer=(\d+):\s*(.*)", line)
        if not m:
            continue
        layer = int(m.group(1))
        if layer not in all_freqs:
            all_freqs[layer] = {}
        for pair in m.group(2).strip().split():
            if ":" not in pair:
                continue
            eidx, count = pair.split(":")
            eidx, count = int(eidx), int(count)
            all_freqs[layer][eidx] = all_freqs[layer].get(eidx, 0) + count

    # Convert to sorted list format
    aggregated = {}
    for layer, experts in all_freqs.items():
        pairs = sorted(experts.items(), key=lambda x: -x[1])
        aggregated[layer] = pairs

    hot_map = select_hot_experts(aggregated if aggregated else freq_data,
                                  num_experts, args.threshold)

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
