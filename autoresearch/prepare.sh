#!/bin/bash
# prepare.sh — One-time setup for Flash-MoE autoresearch.
# Validates environment, builds, runs baseline benchmark.
#
# Usage: bash autoresearch/prepare.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================"
echo "  Flash-MoE Autoresearch — Setup"
echo "========================================"
echo ""

# --- Check model ---
if [ -z "${FLASH_MOE_MODEL:-}" ]; then
    echo "ERROR: Set FLASH_MOE_MODEL to your model path."
    echo "  export FLASH_MOE_MODEL=~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit"
    exit 1
fi

if [ ! -f "$FLASH_MOE_MODEL/config.json" ]; then
    echo "ERROR: No config.json in $FLASH_MOE_MODEL"
    exit 1
fi

echo "Model: $FLASH_MOE_MODEL"
echo ""

# --- Check source files ---
for f in metal_infer/infer.m metal_infer/shaders.metal metal_infer/Makefile; do
    if [ ! -f "$PROJECT_DIR/$f" ]; then
        echo "ERROR: Missing $f"
        exit 1
    fi
done
echo "Source files: OK"

# --- Build ---
echo ""
echo "Building..."
cd "$PROJECT_DIR/metal_infer"
make clean && make infer
echo "Build: OK"
echo ""

# --- Check for existing baseline ---
BASELINE_FILE="$SCRIPT_DIR/baseline.txt"
TSV_FILE="$SCRIPT_DIR/experiments.tsv"

if [ -f "$BASELINE_FILE" ]; then
    EXISTING=$(cat "$BASELINE_FILE")
    echo "Existing baseline found: $EXISTING tok/s"
    echo -n "Overwrite? [y/N] "
    read -r REPLY
    if [ "$REPLY" != "y" ] && [ "$REPLY" != "Y" ]; then
        echo "Keeping existing baseline."
        exit 0
    fi
fi

# --- Run baseline ---
echo "Running baseline benchmark..."
echo ""

RESULT=$(bash "$SCRIPT_DIR/benchmark.sh")
echo "$RESULT"

# Parse tok/s
TOK_S=$(echo "$RESULT" | grep "BENCH_RESULT" | grep -oE 'tok_s=[0-9.]+' | cut -d= -f2)
MATH=$(echo "$RESULT" | grep "BENCH_RESULT" | grep -oE 'math=[A-Z]+' | cut -d= -f2)
JSON=$(echo "$RESULT" | grep "BENCH_RESULT" | grep -oE 'json=[A-Z]+' | cut -d= -f2)
STATUS=$(echo "$RESULT" | grep "BENCH_RESULT" | grep -oE 'status=[A-Z_]+' | cut -d= -f2)

if [ "$STATUS" != "OK" ]; then
    echo ""
    echo "ERROR: Baseline failed with status=$STATUS"
    echo "Fix the issue before starting autoresearch."
    exit 1
fi

# Save baseline
echo "$TOK_S" > "$BASELINE_FILE"

# Record current commit
COMMIT=$(git -C "$PROJECT_DIR" rev-parse --short HEAD)
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%S)

# Initialize TSV
echo -e "id\ttimestamp\tcommit\ttok_s\tmath\tjson\tstatus\tnotes" > "$TSV_FILE"
echo -e "000\t$TIMESTAMP\t$COMMIT\t$TOK_S\t$MATH\t$JSON\tbaseline\tUnmodified baseline" >> "$TSV_FILE"

echo ""
echo "========================================"
echo "  Baseline: $TOK_S tok/s"
echo "  Math: $MATH  JSON: $JSON"
echo "  Saved to: autoresearch/baseline.txt"
echo "  TSV: autoresearch/experiments.tsv"
echo "========================================"
echo ""
echo "Ready. Start the agent with program.md."
