# Fongen Pitch Deck Outline

*For a 5-10 minute video presentation.*

---

## Slide 1: Title

**Fongen**
*An Embodied AI Agent in Minecraft*

---

## Slide 2: The Problem

Current AI agents are **disembodied** — they live in chat windows, not in worlds.

- LLMs can reason about Minecraft, but can't *play* it
- Game bots can play, but can't *reason* or *adapt*
- No open pipeline connects human behavior data → model training → embodied execution

**What if an AI could observe a 3D world, think about what to do, and actually do it?**

---

## Slide 3: Our Solution

**Fongen** — a complete embodied agent stack:

1. **PLAICraft** provides time-align human Minecraft behavior data
2. **Gemma 4 12B** is fine-tuned as the reflex model (fongen-mc) for UMAS action tokens and vision-based survival
3. **Hermes Agent** runtime with **Gemini 3.5 Flash** planner handles the agent loop
4. **Mineflayer body** executes goals in a live Minecraft world

```
Observe → Plan → Act → Verify → (repeat)
```

---

## Slide 4: Architecture

[Insert architecture diagram from docs/architecture.md]

Key design decisions:
- **Mind/body separation** — planner doesn't control raw keystrokes, it emits goals
- **Safety first** — watchdog, emergency stop, step limits
- **Model-agnostic** — any OpenAI-compatible endpoint works (vLLM, llama.cpp, mock)
- **Structured output** — Zod-validated JSON responses only

---

## Slide 5: PLAICraft Data

- 10,000+ hours of Minecraft gameplay from 10,000+ participants
- 200-hour public subset with time-aligned modalities:
  - Screen video (30 FPS)
  - Keyboard/mouse events (SQLite)
  - Audio (game + microphone)
- We convert keyboard/mouse windows → action labels → training examples
- **Fallback**: synthetic dataset with rule-based scenarios (2000+ examples in seconds)

---

## Slide 6: AMD Technology

- **AMD Instinct MI300X** (192GB HBM) via AMD Developer Cloud
- **ROCm** + PyTorch for training
- **Unsloth** LoRA BF16 fine-tuning (AMD playbook: adamw_torch, no QLoRA initially)
- **vLLM ROCm** for fast serving
- **llama.cpp** (HIP/Vulkan) as reliable inference fallback

The project uses AMD's full stack: from training on MI300X to inference with ROCm-optimized runtimes.

---

## Slide 7: Demo

[Screen recording of Fongen playing Minecraft]

- Agent spawns in a Minecraft world
- Observes: "I see oak logs 5 blocks north, my health is full"
- Plans: "I should collect wood for crafting" → MINE_TASK oak_log x4
- Acts: Pathfinds to tree, mines 4 logs
- Verifies: "Successfully mined 4/4 oak logs"
- Replans: "Now I should craft planks" → CRAFT_TASK oak_planks x4

---

## Slide 8: What Makes This Different

| Traditional Bots | Fongen |
|-----------------|--------|
| Scripted behavior | LLM-planned goals |
| No reasoning | Observes, thinks, explains decisions |
| No adaptation | Replans on failure, learns from history |
| Fixed actions | Extensible intent schema |
| No data pipeline | PLAICraft + synthetic training |

Fongen is not a bot — it's a **prototype of general embodied AI**.

---

## Slide 9: Roadmap

Future work:

1. **Dedicated dataset** — collect consented gameplay from players
2. **Raw-input engine** — C++ core for OS-level screen/input (no API wrappers)
3. **Reflex model** — fongen-mc (Gemma 4 12B) for real-time vision-based survival at FPS
4. **Mixture of Agents** — reflex (Gemma/fongen-mc) + planner (Gemini 3.5 Flash) + social (Gemini 3.5 Flash Live / GPT Realtime 2.1 mini)
5. **Long-term memory** — persistent context across sessions
6. **Voice agent** — real-time speech interaction

---

## Slide 10: Links

**Links:**
- GitHub: [repo URL]
- Demo video: [video URL]
- Live demo: [app URL if available]

---

*Fongen explores Minecraft.*
