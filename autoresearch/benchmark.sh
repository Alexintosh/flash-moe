#!/bin/bash
# benchmark.sh — Fixed harness for Flash-MoE autoresearch.
# DO NOT MODIFY. This is the ground truth measurement.
#
# Usage: bash autoresearch/benchmark.sh
# Requires: FLASH_MOE_MODEL env var set to model path
#
# Exit codes: 0=all pass, 1=build fail, 2=quality fail, 3=runtime error

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
METAL_DIR="$PROJECT_DIR/metal_infer"

# --- Config ---
BENCH_TOKENS=200
BENCH_PROMPT="Explain quantum computing in simple terms"
MATH_PROMPT="What is 2+2? Answer with just the number."
JSON_PROMPT='You are a helpful assistant. Return a JSON object with keys "name" and "value" for the capital of France. Output ONLY valid JSON, nothing else.'
WARMUP_TOKENS=5

# --- Validate environment ---
if [ -z "${FLASH_MOE_MODEL:-}" ]; then
    echo "ERROR: FLASH_MOE_MODEL not set" >&2
    exit 1
fi
if [ ! -f "$FLASH_MOE_MODEL/config.json" ]; then
    echo "ERROR: No config.json in $FLASH_MOE_MODEL" >&2
    exit 1
fi

# --- Step 1: BUILD ---
echo "=== BUILD ===" >&2
cd "$METAL_DIR"

# Touch infer.m to ensure shaders.metal changes trigger rebuild
# (shaders are compiled at runtime from source embedded in infer.m's build)
touch infer.m
if ! make infer 2>&1 | tail -5 >&2; then
    echo "BENCH_RESULT tok_s=0.00 math=SKIP json=SKIP status=BUILD_FAIL"
    exit 1
fi
echo "Build OK" >&2

# --- Step 2: WARMUP (prime page cache) ---
echo "=== WARMUP ===" >&2
./infer --model "$FLASH_MOE_MODEL" --prompt "Hi" --tokens "$WARMUP_TOKENS" > /dev/null 2>&1 || true

# --- Step 3: PERFORMANCE BENCHMARK ---
echo "=== BENCHMARK ($BENCH_TOKENS tokens) ===" >&2
BENCH_OUT=$(mktemp)
BENCH_ERR=$(mktemp)
MATH_OUT=$(mktemp)
MATH_ERR=$(mktemp)
JSON_OUT=$(mktemp)
JSON_ERR=$(mktemp)
trap "rm -f $BENCH_OUT $BENCH_ERR $MATH_OUT $MATH_ERR $JSON_OUT $JSON_ERR 2>/dev/null" EXIT

if ! ./infer --model "$FLASH_MOE_MODEL" \
    --prompt "$BENCH_PROMPT" \
    --tokens "$BENCH_TOKENS" \
    > "$BENCH_OUT" 2> "$BENCH_ERR"; then
    echo "BENCH_RESULT tok_s=0.00 math=SKIP json=SKIP status=RUNTIME_ERROR"
    cat "$BENCH_ERR" >&2
    exit 3
fi

# Parse tok/s from "Generation:     X.X s (Y.YY tok/s)"
TOK_S=$(grep "Generation:" "$BENCH_OUT" | grep -oE '[0-9]+\.[0-9]+ tok/s' | grep -oE '[0-9]+\.[0-9]+')
if [ -z "$TOK_S" ]; then
    echo "ERROR: Could not parse tok/s from output" >&2
    cat "$BENCH_OUT" >&2
    echo "BENCH_RESULT tok_s=0.00 math=SKIP json=SKIP status=PARSE_ERROR"
    exit 3
fi
echo "Performance: $TOK_S tok/s" >&2

# --- Step 4: QUALITY GATE — MATH ---
echo "=== QUALITY: MATH ===" >&2
MATH_PASS="FAIL"
if ./infer --model "$FLASH_MOE_MODEL" \
    --prompt "$MATH_PROMPT" \
    --tokens 20 \
    --think-budget 256 \
    > "$MATH_OUT" 2> "$MATH_ERR"; then
    # Extract text between "--- Output ---" and "--- Statistics ---"
    MATH_TEXT=$(sed -n '/--- Output ---/,/--- Statistics ---/{/--- Output ---/d;/--- Statistics ---/d;p;}' "$MATH_OUT")
    if echo "$MATH_TEXT" | grep -q "4"; then
        MATH_PASS="PASS"
    fi
fi
echo "Math gate: $MATH_PASS" >&2

# --- Step 5: QUALITY GATE — JSON ---
echo "=== QUALITY: JSON ===" >&2
JSON_PASS="FAIL"
if ./infer --model "$FLASH_MOE_MODEL" \
    --prompt "$JSON_PROMPT" \
    --tokens 50 \
    --think-budget 256 \
    > "$JSON_OUT" 2> "$JSON_ERR"; then
    JSON_TEXT=$(sed -n '/--- Output ---/,/--- Statistics ---/{/--- Output ---/d;/--- Statistics ---/d;p;}' "$JSON_OUT")
    # Check for valid JSON structure: has braces and a key
    if echo "$JSON_TEXT" | grep -q '{' && echo "$JSON_TEXT" | grep -q '}' && echo "$JSON_TEXT" | grep -q '"name"'; then
        JSON_PASS="PASS"
    fi
fi
echo "JSON gate: $JSON_PASS" >&2

# --- Step 6: STRUCTURED OUTPUT ---
if [ "$MATH_PASS" = "PASS" ] && [ "$JSON_PASS" = "PASS" ]; then
    STATUS="OK"
    EXIT_CODE=0
else
    STATUS="QUALITY_FAIL"
    EXIT_CODE=2
fi

echo "BENCH_RESULT tok_s=$TOK_S math=$MATH_PASS json=$JSON_PASS status=$STATUS"
exit $EXIT_CODE
