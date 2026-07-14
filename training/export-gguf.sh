#!/usr/bin/env bash
# export-gguf.sh — Export merged model to GGUF format for llama.cpp
#
# Usage:
#   ./export-gguf.sh ./fongen-merged/ ./fongen-gguf/ [q4_k_m|q5_k_m|q8_0|f16]
set -euo pipefail

MERGED_DIR="${1:-./fongen-merged}"
GGUF_DIR="${2:-./fongen-gguf}"
QUANT="${3:-q4_k_m}"

if [[ ! -d "${MERGED_DIR}" ]]; then
  echo "ERROR: Merged model directory not found: ${MERGED_DIR}"
  exit 1
fi

mkdir -p "${GGUF_DIR}"

# Clone llama.cpp if not present
LLAMA_DIR="${LLAMA_CPP_DIR:-./llama.cpp}"
if [[ ! -d "${LLAMA_DIR}" ]]; then
  echo "[INFO] Cloning llama.cpp..."
  git clone https://github.com/ggml-org/llama.cpp "${LLAMA_DIR}" --depth 1
fi

# Install gguf python deps
pip install -e "${LLAMA_DIR}/gguf-py" --quiet 2>/dev/null || pip install gguf --quiet 2>/dev/null || true

# Convert to GGUF (unquantized f16 first)
echo "[INFO] Converting to GGUF (f16)..."
python "${LLAMA_DIR}/convert_hf_to_gguf.py" \
  "${MERGED_DIR}" \
  --outfile "${GGUF_DIR}/fongen-mc-f16.gguf" \
  --outtype f16

# Quantize
echo "[INFO] Quantizing to ${QUANT}..."
"${LLAMA_DIR}/build/bin/llama-quantize" \
  "${GGUF_DIR}/fongen-mc-f16.gguf" \
  "${GGUF_DIR}/fongen-mc-${QUANT}.gguf" \
  "${QUANT}" 2>/dev/null || {
  echo "[WARN] Pre-built llama-quantize not found. Building llama.cpp..."
  echo "[INFO] Alternative: use Python-based quantization or build llama.cpp manually."
  echo "[INFO] For now, the f16 GGUF is available at: ${GGUF_DIR}/fongen-mc-f16.gguf"
}

echo ""
echo "[DONE] GGUF export complete"
echo "[INFO] Files in ${GGUF_DIR}/:"
ls -lh "${GGUF_DIR}/" 2>/dev/null || true
echo ""
echo "Next: serve with llama-server"
echo "  llama-server -m ${GGUF_DIR}/fongen-mc-${QUANT}.gguf --port 8000 --host 0.0.0.0"
