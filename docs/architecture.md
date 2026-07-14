# Fongen Architecture

## System Overview

Fongen is an embodied agent prototype with a clear separation between **mind** (planning) and **body** (execution).

```
┌─────────────────────────────────────────────────────────────┐
│                        AGENT LOOP                           │
│                   (src/loop/agent-loop.ts)                  │
│                                                             │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐  │
│   │ OBSERVE  │──▶│  PLAN    │──▶│   ACT    │──▶│ VERIFY │  │
│   └──────────┘   └──────────┘   └──────────┘   └────────┘  │
│        ▲                                            │       │
│        └────────────────────────────────────────────┘       │
│                      (every ~5 seconds)                     │
└─────────────────────────────────────────────────────────────┘
```

## Component Map

### Body Layer (`src/body/`)

The body layer connects to Minecraft via Mineflayer and handles all world interaction.

| File | Responsibility |
|------|---------------|
| `minecraft-body.ts` | Mineflayer connection lifecycle, event handling, observation/action API |
| `state-extractor.ts` | Extracts compact world state: HP, food, position, inventory, nearby blocks/entities |
| `action-executor.ts` | Executes high-level intents (GOTO, MINE_TASK, CRAFT_TASK, etc.) via Mineflayer |
| `safety-guard.ts` | Watchdog, emergency stop, step limits, health/food thresholds |

### Mind Layer (`src/mind/`)

The mind layer handles planning via an LLM (Gemma 4 12B) and manages short-term memory.

| File | Responsibility |
|------|---------------|
| `gemma-provider.ts` | OpenAI-compatible API client to Gemma (vLLM/llama.cpp), with mock fallback |
| `planner.ts` | Coordinates observation → prompt → model → parsed response |
| `tool-bridge.ts` | System prompt construction, intent descriptions, observation formatting |
| `memory.ts` | Short-term history of goals and results (last N entries) |

### Schema Layer (`src/schemas/`)

Zod schemas that define the contract between mind and body.

| File | Exports |
|------|---------|
| `actions.ts` | `ActionEnum`, `ActionSchema` — 17 low-level primitives |
| `intents.ts` | `IntentEnum`, `GoalSchema`, `PlannerResponseSchema`, `GoalResultSchema` — 7 high-level intents |
| `observation.ts` | `ObservationSchema` — full world state structure |

### Training Layer (`training/`)

Python scripts for AMD MI300X (ROCm) fine-tuning pipeline.

| File | Responsibility |
|------|---------------|
| `download-plaicraft.sh` | Download PLAICraft metadata or full 621GB dataset |
| `convert-plaicraft.py` | Convert keyboard/mouse SQLite → JSONL training examples |
| `generate-synthetic.py` | Generate synthetic planning examples (fallback) |
| `train-lora.py` | Unsloth LoRA BF16 fine-tune (with PEFT fallback) |
| `merge-lora.py` | Merge LoRA adapters into base model |
| `export-gguf.sh` | Export merged model to GGUF for llama.cpp |
| `serve-vllm.sh` | vLLM serving (OpenAI-compatible API) |
| `serve-llamacpp.sh` | llama.cpp serving (reliable fallback) |

## Data Flow

```
1. OBSERVE
   MinecraftBody.observe()
   → StateExtractor.extractObservation(bot, eventLog)
   → Observation { health, position, inventory, nearby_blocks, ... }

2. PLAN
   Planner.decide(observation)
   → ShortTermMemory.recordObservation(observation)
   → ToolBridge.buildPlannerPrompt(observation, memory)
   → GemmaProvider.plan(systemPrompt, userPrompt)
   → Zod parse → PlannerResponse { thought, goal, confidence }

3. ACT
   MinecraftBody.act(goal)
   → SafetyGuard.resetWatchdog()
   → ActionExecutor.executeGoal(bot, goal, guard)
     → GOTO: Pathfinder navigation
     → MINE_TASK: find block → navigate → dig
     → CRAFT_TASK: find recipe → craft
     → SURVIVE: eat / flee / sneak
     → etc.
   → GoalResult { success, message, steps_taken, elapsed_ms }

4. VERIFY
   Planner.recordExecution(goal, result)
   → ShortTermMemory.record(goal, result)
   → (next planning cycle uses this history)
```

## Safety Architecture

```
                    ┌─────────────────────┐
                    │    SafetyGuard      │
                    ├─────────────────────┤
                    │ • maxSteps: 50      │
                    │ • watchdog: 60s     │
                    │ • health < 6 → stop │
                    │ • food < 2 → stop   │
                    │ • emergencyStop     │
                    │ • stepCount tracking│
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
      AgentLoop checks    ActionExecutor    MinecraftBody
      before each tick    checks before     event handlers
                          each action       trigger on damage
```

## Model Integration

The planner communicates with the model via an OpenAI-compatible API:

```
Fongen Runtime (TypeScript)
    │
    │ POST /v1/chat/completions
    │ { model, messages: [system, user], response_format: json }
    ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   vLLM      │ OR  │ llama-server│ OR  │  Mock       │
│  (ROCm)     │     │  (GGUF)     │     │  Planner    │
│  MI300X     │     │  any GPU    │     │  (no model) │
└─────────────┘     └─────────────┘     └─────────────┘
    │                    │
    │ Gemma 4 12B        │ Gemma 4 12B
    │ + LoRA adapters    │ GGUF quantized
    │ BF16               │ Q4_K_M
```

## Fallback Chain

```
Plan A: MI300X + ROCm + Unsloth + vLLM
   │ fail
   ▼
Plan B: LoRA BF16 + adamw_torch (no QLoRA)
   │ fail
   ▼
Plan C: llama.cpp GGUF (HIP or Vulkan)
   │ fail
   ▼
Plan D: Gemma 4 E4B (smaller model)
   │ fail
   ▼
Plan E: Mock planner (rule-based, no model needed)
```

The mock planner ensures the demo always works — even without GPU access, the observe-plan-act loop runs with rule-based goal selection.
