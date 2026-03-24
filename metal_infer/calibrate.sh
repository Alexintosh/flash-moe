#!/bin/bash
# calibrate.sh -- Run GPTQ calibration: generate tokens to collect expert activations,
# then build per-expert Hessian matrices for quantization.
#
# Usage: ./calibrate.sh
# Prerequisites: ./infer must be built (make), packed_experts/ directory must exist.

set -euo pipefail

TOKENS=16384
DUMP_FILE="calibration_dump.bin"
OUTPUT_DIR="calibration/"
PROMPT="Explain the history of computing from the invention of the transistor through modern AI systems. Cover the key breakthroughs in hardware, software, networking, and algorithms that enabled each major advance. Discuss the contributions of pioneers like Turing, von Neumann, Shannon, Dijkstra, Knuth, and others."

echo "=== GPTQ Calibration Phase 1 ==="
echo "Generating ${TOKENS} tokens to collect expert activations..."
echo ""

./infer --prompt "${PROMPT}" --tokens ${TOKENS} --collect-activations "${DUMP_FILE}"

echo ""
echo "Activation dump saved to ${DUMP_FILE} ($(du -h "${DUMP_FILE}" | cut -f1))"
echo ""
echo "Building per-expert Hessian matrices..."
echo ""

python3 build_hessian.py "${DUMP_FILE}" --output-dir "${OUTPUT_DIR}"

echo ""
echo "Calibration complete. Hessians saved to ${OUTPUT_DIR}"
