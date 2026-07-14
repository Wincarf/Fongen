#!/usr/bin/env bash
# serve-vllm.sh — Serve Fongen model with vLLM (OpenAI-compatible API)
# Run on AMD MI300X with ROCm.
#
# Usage:
#   ./serve-vllm.sh ./fongen-merged/ 8000
#   ./serve-vllm.sh ./fongen-merged/ 8000 --dtype bfloat16
set -euo pipefail

MODEL_DIR="${1:-./fongen-merged}"
PORT="${2:-8000}"
EXTRA_ARGS="${@:3}"

echo "[INFO] Starting vLLM server for ${MODEL_DIR} on port ${PORT}"
echo "[INFO] OpenAI-compatible API: http://0.0.0.0:${PORT}/v1"

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_DIR}" \
  --port "${PORT}" \
  --host 0.0.0.0 \
  --served-model-name "fongen-mc" \
  --trust-remote-code \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  ${EXTRA_ARGS}
