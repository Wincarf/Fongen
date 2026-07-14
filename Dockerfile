# Fongen — Embodied Minecraft Agent
# Multi-stage build: Onyxz (TS body + reflex + schemas) + Fongen (demo/training)
#
# Build context: /opt/fongen-workspace (includes both Onyxz/ and Fongen/)

# ──────────────────────────────────────────────────────
# Stage 1: Onyxz body + TS runtime (TypeScript)
# ──────────────────────────────────────────────────────
FROM node:22-slim AS onyxz-build

WORKDIR /opt/onyxz

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ripgrep \
    && rm -rf /var/lib/apt/lists/*

# Copy Onyxz package files and install TS deps
COPY Onyxz/package.json Onyxz/package-lock.json* ./
RUN npm install

# Copy Onyxz source (schemas, reflex, body, index.ts, brain, agent, bridge)
COPY Onyxz/src/ ./src/
COPY Onyxz/tsconfig.json ./
COPY Onyxz/.env.example* ./

# Copy Onyxz Python backend
COPY Onyxz/backend/ /opt/onyxz/backend/
RUN pip install --no-cache-dir -r /opt/onyxz/backend/requirements.txt

# Copy Onyxz docs
COPY Onyxz/docs/ ./docs/
COPY Onyxz/README.md ./

# ──────────────────────────────────────────────────────
# Stage 2: Final runtime image
# ──────────────────────────────────────────────────────
FROM node:22-slim

WORKDIR /opt/fongen

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 git curl ripgrep ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy Onyxz (TS body + schemas + reflex + Python backend)
COPY --from=onyxz-build /opt/onyxz /opt/onyxz
WORKDIR /opt/onyxz

# Copy Fongen product wrapper (demo, training, config)
COPY Fongen/demo/ /opt/fongen/demo/
COPY Fongen/training/ /opt/fongen/training/
COPY Fongen/docs/ /opt/fongen/docs/
COPY Fongen/.env.example /opt/fongen/.env.example
COPY Fongen/README.md /opt/fongen/README.md

# Default env
ENV MC_HOST=127.0.0.1 \
    MC_PORT=25565 \
    MC_USERNAME=Fongen \
    MC_AUTH=offline \
    MC_VERSION=1.21.4 \
    MODEL_NAME=fongen-mc \
    LOG_LEVEL=info \
    PLANNER_ENABLED=true \
    PLANNER_MODEL=gemini-2.0-flash \
    PLAN_INTERVAL_MS=30000 \
    MAX_STEPS=50 \
    WATCHDOG_TIMEOUT_MS=60000 \
    FRAMEWORK_TS_DIR=/opt/onyxz \
    AUTO_START_PLANNER=false \
    MEMORY_DIR=/opt/onyxz/backend/data/memory

WORKDIR /opt/onyxz
# Entrypoint: run the full Fongen agent (body + reflex + planner)
CMD ["npx", "tsx", "src/index.ts", "--demo"]
