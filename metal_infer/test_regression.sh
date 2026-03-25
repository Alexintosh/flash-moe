#!/bin/bash
# Flash-MoE Regression Test Runner
# Usage: ./test_regression.sh [model_path]
#
# Starts the OpenAI-compatible API server, runs the regression test suite,
# then shuts the server down. Exit code reflects test results.

set -euo pipefail

MODEL=${1:-"../models/qwen3.5-35b-a3b-q4"}
PORT=8099  # avoid conflict with dev server
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Flash-MoE Regression Test Runner"
echo "================================"
echo "Model: $MODEL"
echo "Port:  $PORT"
echo ""

# Start the inference server in the background
echo "Starting server..."
"$SCRIPT_DIR/infer" --model "$MODEL" --openai-api "$PORT" &
PID=$!

# Ensure we clean up the server on exit (normal or error)
cleanup() {
    if kill -0 "$PID" 2>/dev/null; then
        echo ""
        echo "Stopping server (PID $PID)..."
        kill "$PID" 2>/dev/null
        wait "$PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Wait for server to become ready (up to 30 seconds)
echo "Waiting for server to start..."
for i in $(seq 1 30); do
    if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "ERROR: Server process died during startup"
        exit 1
    fi
    sleep 1
done

# Verify server is actually responding
if ! curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
    echo "ERROR: Server failed to start within 30 seconds"
    exit 1
fi

echo ""

# Run the test suite
python3 "$SCRIPT_DIR/test_regression.py" --port "$PORT"
EXIT=$?

exit $EXIT
