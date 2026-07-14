#!/usr/bin/env bash
# run-demo.sh — Start demo Minecraft server and run Fongen agent
#
# Usage:
#   ./run-demo.sh                    # Start server + agent with mock planner
#   ./run-demo.sh --model            # Start server + agent with real model
#   ./run-demo.sh --server-only      # Start only the Minecraft server
#   ./run-demo.sh --agent-only       # Run only the agent (server already running)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
FRAMEWORK_DIR="${FRAMEWORK_DIR:-${PROJECT_DIR}/../Onyxz}"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"
ENV_FILE="${PROJECT_DIR}/.env"

use_model=false
server_only=false
agent_only=false

for arg in "$@"; do
  case "$arg" in
    --model) use_model=true ;;
    --server-only) server_only=true ;;
    --agent-only) agent_only=true ;;
    --help|-h)
      echo "Usage: $0 [--model|--server-only|--agent-only]"
      echo "  (default: start server + agent with mock planner)"
      exit 0
      ;;
  esac
done

start_server() {
  echo "[Demo] Starting Minecraft demo server on port 25575..."
  docker compose -f "${COMPOSE_FILE}" up -d
  echo "[Demo] Waiting for server to be ready..."
  for i in $(seq 1 60); do
    if docker compose -f "${COMPOSE_FILE}" exec -T fongen-mc mc-health 2>/dev/null; then
      echo "[Demo] Server is ready!"
      return 0
    fi
    echo -n "."
    sleep 5
  done
  echo ""
  echo "[ERROR] Server failed to start in 5 minutes"
  docker compose -f "${COMPOSE_FILE}" logs --tail 20
  exit 1
}

stop_server() {
  echo "[Demo] Stopping Minecraft demo server..."
  docker compose -f "${COMPOSE_FILE}" down
}

run_agent() {
  echo "[Demo] Starting Fongen agent..."
  if $use_model; then
    echo "[Demo] Using real model (ensure MODEL_BASE_URL is set in .env)"
  else
    echo "[Demo] Using mock planner (set MODEL_BASE_URL in .env for real model)"
  fi

  # Set demo defaults
  export MC_HOST="${MC_HOST:-127.0.0.1}"
  export MC_PORT="${MC_PORT:-25575}"
  export MC_USERNAME="${MC_USERNAME:-Fongen}"
  export MC_AUTH="${MC_AUTH:-offline}"
  export MC_VERSION="${MC_VERSION:-1.21.4}"

  if ! $use_model; then
    export MODEL_BASE_URL="${MODEL_BASE_URL:-}"
  fi

  cd "${FRAMEWORK_DIR}"
  npx tsx src/index.ts --demo
}

# Cleanup on exit
cleanup() {
  if ! $agent_only; then
    stop_server
  fi
}
trap cleanup EXIT

# Main
if ! $agent_only; then
  start_server
fi

if ! $server_only; then
  run_agent
fi

# If server-only, keep running
if $server_only; then
  echo "[Demo] Server running on port 25575. Press Ctrl+C to stop."
  trap cleanup EXIT
  while true; do sleep 60; done
fi
