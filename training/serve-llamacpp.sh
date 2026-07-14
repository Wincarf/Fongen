#!/usr/bin/env bash
# serve-llamacpp.sh — Serve Fongen model with llama.cpp (reliable fallback)
# Works on AMD (HIP/Vulkan), NVIDIA, or CPU.
#
# Usage:
#   ./serve-llamacpp.sh ./fongen-gguf/fongen-mc-q4_k_m.gguf 8000
set -euo pipefail

GGUF_FILE="${1:?Usage: $0 <gguf-file> [port]}"
PORT="${2:-8000}"

if [[ ! -f "${GGUF_FILE}" ]]; then
  echo "ERROR: GGUF file not found: ${GGUF_FILE}"
  exit 1
fi

LLAMA_DIR="${LLAMA_CPP_DIR:-./llama.cpp}"
SERVER_BIN="${LLAMA_DIR}/build/bin/llama-server"

if [[ ! -f "${SERVER_BIN}" ]]; then
  echo "ERROR: llama-server not found at ${SERVER_BIN}"
  echo "Build llama.cpp first:"
  echo "  cd ${LLAMA_DIR} && cmake -B build -DGGML_HIP=ON && cmake --build build --config Release"
  echo "  Or for Vulkan: cmake -B build -DGGML_VULKAN=ON && cmake --build build"
  echo "  Or for CPU:    cmake -B build && cmake --build build"
  exit 1
fi

echo "[INFO] Starting llama-server for ${GGUF_FILE} on port ${PORT}"
echo "[INFO] OpenAI-compatible API: http://0.0.0.0:${PORT}/v1"
echo "[INFO] GGUF size: $(du -h "${GGUF_FILE}" | cut -f1)"

"${SERVER_BIN}" \
  -m "${GGUF_FILE}" \
  --port "${PORT}" \
  --host 0.0.0.0 \
  --alias "fongen-mc" \
  --ctx-size 4096 \
  --n-gpu-layers 999
