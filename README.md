# Fongen

> An embodied AI agent that lives in Minecraft. Fongen sees the world, plans what to do, and acts — mining, crafting, exploring, surviving, and chatting with players. Built on the Onyxz framework for Minecraft embodied agents.

**Author:** Wincarf · **License:** MIT

---

## Quick Start

```bash
cd Onyxz/backend
cp .env.example .env          # put your Gemini API key in LLM_API_KEY
docker compose up --build     # launches MC server + agent
```

Join the server at `localhost:25599`, API on `localhost:8000`. The agent connects, observes the world, plans multi-step goals with Gemini, and executes them autonomously.

To run against an existing Minecraft server:

```bash
docker compose -f docker-compose.external.yml up --build
```

---

## What is Fongen?

Fongen is an embodied Minecraft agent that runs a continuous **Observe-Plan-Act** loop:

1. **Observe** — extracts the full game state: health, inventory, nearby blocks, entities, terrain, equipment
2. **Plan** — sends the observation to Gemini and gets back a sequence of up to 8 high-level goals with reasoning
3. **Act** — executes each goal through Mineflayer: pathfinding, mining, crafting, placing, combat, smelting
4. **Verify** — a SafetyGuard reflex layer monitors health/food and can emergency-stop at any time

The agent also has a **chat brain** — a separate LLM call that produces humanized, personality-driven chat messages with realistic typing speed and reaction delays.

Fongen is the **agent**. Onyxz is the **framework it runs on** (schemas, reflex, body, planner, bridge). This repo holds the demo scripts and the training pipeline for a future fine-tuned reflex model.

```
    +----------+     +----------+     +----------+
    | OBSERVE  |---->| PLAN     |---->| ACT      |
    | (see)    |     | (think)  |     | (move)   |
    +----------+     +----------+     +----------+
         ^                                   |
         +------------- verify <-------------+
```

---

## Architecture

### Reflex Layer (SafetyGuard)

A rule-based survival system that runs independently of the planner:
- Emergency stop when health < 6 or food < 2
- Step cap per goal (default 50)
- Watchdog timeout per goal (default 60s)
- Game-agnostic — operates through an `EmergencyStoppable` interface

### Planner (FastAPI + Gemini)

The strategic brain — runs every 15-45 seconds:
- Observes the world via MCP, asks Gemini for multi-step goal sequences (up to 8 goals)
- **Score system** — rewards exploration, discovery, goal completion; penalizes damage and death
- **World memory** — persists known locations of crafting tables, furnaces, ores
- **Dynamic hints** — generates contextual crafting tips based on inventory
- **Minecraft encyclopedia** — system prompt includes ore generation, recipes, combat strategies
- **Sleep-polling** — wakes up immediately if player chat or danger is detected during the wait interval

### Chat Brain (Gemini)

Separate LLM call for social interaction:
- Humanized typing speed (configurable CPS)
- Randomized reaction delays
- Ambient replies (occasionally initiates conversation)
- Configurable persona (name, language, tone)

### Body (Mineflayer)

The Minecraft adapter:
- State extractor reads the world into a typed `Observation`
- Action executor turns `Goal` objects into 13 different Minecraft actions
- Handles connection lifecycle, kicks, errors, graceful shutdown

### MCP Bridge

Connects the Python planner to the TypeScript body over stdio:
- `get_state()` — current world observation
- `set_goal(goal)` — send a goal, block until done
- `get_goal_status()` — query current goal status
- `chat(message)` — send a chat message

```
+-------------------------------------------------------------+
|  REFLEX LAYER (SafetyGuard)                                 |
|  HP<6 -> stop, food<2 -> eat, step cap, watchdog            |
+--------------------------+----------------------------------+
                           |
+--------------------------v----------------------------------+
|  PLANNER (FastAPI + Gemini, every 15-45s)                   |
|  Observe -> Multi-step Plan (up to 8 goals) -> Execute      |
|  Score system + World memory + Dynamic hints                |
+--------------------------+----------------------------------+
                           | MCP stdio bridge
+--------------------------v----------------------------------+
|  BODY (Mineflayer)                                          |
|  13 intents: GOTO, MINE, CRAFT, PLACE, FOLLOW, SURVIVE,     |
|  EQUIP, SMELT, DROP, ATTACK, DEPOSIT, WITHDRAW, IDLE       |
+-------------------------------------------------------------+

+-------------------------------------------------------------+
|  CHAT BRAIN (Gemini, humanized replies)                     |
|  Persona + typing speed + ambient replies                   |
+-------------------------------------------------------------+
```

---

## Intents

The agent acts through 13 high-level intents:

| Intent | Description |
|--------|-------------|
| `GOTO` | Navigate to coordinates, player, or landmark |
| `MINE_TASK` | Mine a specific block type |
| `CRAFT_TASK` | Craft an item using a known recipe |
| `PLACE_TASK` | Place a block at a specific location |
| `FOLLOW_PLAYER` | Follow a named player at safe distance |
| `SURVIVE` | Eat, flee danger, find shelter |
| `EQUIP_TASK` | Equip weapon, tool, armor, or shield |
| `SMELT_TASK` | Smelt ores or cook food in a furnace |
| `DROP_TASK` | Drop items for the player to pick up |
| `ATTACK_TASK` | Hunt/attack a nearby entity |
| `DEPOSIT_TASK` | Deposit items into a chest |
| `WITHDRAW_TASK` | Withdraw items from a chest |
| `IDLE` | Stop and wait |

---

## Safety

The agent includes a **SafetyGuard** that:
- Triggers emergency stop when health < 6 or food < 2
- Limits total steps per session (configurable, default 50)
- Watchdog timeout on each goal execution (default 60s)
- Runs independently of the planner
- Chat command `Fongen stop` for manual override

---

## Manual Setup

### Prerequisites
- Node.js 22+
- Python 3.11+
- Docker (for demo Minecraft server)

### Install and run

```bash
# Clone the Onyxz framework repository (agent runtime: schemas, reflex, body, planner)
git clone https://github.com/wincarf/onyxz

# Clone this repository
git clone https://github.com/wincarf/fongen

cd Onyxz
npm install

cd ../Fongen
cp .env.example .env
# Edit .env: set PLANNER_API_KEY to your Gemini API key
# If empty, the agent uses a rule-based fallback planner (no API needed)

./demo/run-demo.sh               # start MC server + agent
./demo/run-demo.sh --server-only # just the server
./demo/run-demo.sh --agent-only  # agent against an existing server
```

Connect to the server at `localhost:25575` to watch Fongen act in real-time.

---

## Training Pipeline

The training pipeline for a future fine-tuned reflex model. See [`training/README.md`](training/README.md) for full details.

```bash
cd training

# Option A: Synthetic dataset (fast, no download)
python generate-synthetic.py --output ./training-data/ --count 2000

# Option B: PLAICraft data (research-backed, requires download)
./download-plaicraft.sh metadata
python convert-plaicraft.py --input ./plaicraft-data/ --output ./training-data/

# Fine-tune with LoRA BF16 (AMD MI300X + ROCm)
python train-lora.py --data ./training-data/ --output ./fongen-lora/ --epochs 3

# Merge and serve
python merge-lora.py --adapters ./fongen-lora/ --output ./fongen-merged/
./serve-vllm.sh ./fongen-merged/ 8000
```

---

## Roadmap

1. **Fine-tuned reflex model** — train a vision model on PLAICraft data to emit UMAS action tokens in < 100ms, replacing the rule-based SafetyGuard as the fast reflex layer.
2. **Subsumption** — survival reflexes override in-flight planner goals (creeper nearby -> flee mid-task).
3. **Hermes integration** — connect the Hermes Agent runtime for persistent SQLite memory, skills, and provider routing.
4. **Social voice agent** — real-time speech, decoupled from movement.
5. **Raw-input core** — C++ screen capture + keystroke injection, replacing Mineflayer.
6. **UMAS expansion** — from 13 intents toward a full ~150-token action taxonomy.

---

## License

MIT — see [LICENSE](LICENSE).
