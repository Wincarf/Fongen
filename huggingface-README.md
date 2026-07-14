---
library_name: vllm
license: mit
language:
  - en
base_model: google/gemma-4-12B-it
tags:
  - minecraft
  - embodied-agent
  - vla
  - vision-language-action
  - fongen
  - plaicraft
  - hermes
pipeline_tag: text-generation
---

# Fongen-MC

> A fine-tuned Gemma 4 12B model for **Minecraft reflexes** — vision-based survival, combat, and movement via UMAS action tokens.
> Part of the **Fongen** embodied agent.

## Model Details

Fongen-MC is the **reflex agent** (motor cortex) of Fongen, an embodied AI agent that lives inside Minecraft. It takes visual frames and game state as input and outputs a single UMAS action token — the next immediate action (move, mine, attack, flee, etc.).

Planning is handled separately by **Gemini 3.5 Flash** via the Hermes Agent runtime. This model is purely for fast, reactive survival.

| Property | Value |
|----------|-------|
| **Base model** | `google/gemma-4-12B-it` |
| **Parameters** | 11.95B (dense) |
| **Fine-tuning** | LoRA BF16 (Unsloth / PEFT) |
| **Training data** | PLAICraft (UBC/PLAI) — time-aligned keyboard/mouse + screen video |
| **Training goal** | UMAS action token classification (single-token output) |
| **Context length** | 4096 (tuned for short observation prompts) |
| **Output** | Single UMAS action token (e.g., `<|NAV_FWD|>`, `<|SURV_FLEE_180|>`) |

## Architecture

```
Visual Frame + Game State     Fongen-MC           UMAS Action Token
┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Screen frame │───▶│  Gemma 4 12B     │───▶│ <|SURV_FLEE_180|>│
│ HP, food,    │    │  + LoRA adapter  │    │ (single token)   │
│ position,    │    │  KV-cache        │    └────────┬─────────┘
│ threats...   │    └──────────────────┘             │
└──────────────┘                                     ▼
                                           Mineflayer Body
                                           (executes action)
```

Fongen uses a **dual-agent architecture**:
- **Reflex** (this model) — < 100ms vision-based survival via UMAS action tokens
- **Planner** (Gemini 3.5 Flash via Hermes) — 30-60s strategic planning
- **Social** (future) — Gemini 3.5 Flash Live / GPT Realtime 2.1 mini

## Action Tokens (UMAS)

The model outputs a single UMAS (Universal Mineflayer Action Space) token per forward pass:

| Category | Examples | Description |
|----------|----------|-------------|
| `NAV_` | `<|NAV_FWD|>`, `<|NAV_JUMP|>`, `<|NAV_SNEAK_FWD|>` | Navigation |
| `CAM_` | `<|CAM_LOCK_THREAT|>`, `<|CAM_YAW_L_30|>` | Camera/aiming |
| `ACT_` | `<|ACT_ATK_MELEE|>`, `<|ACT_MINE_TARGET|>` | Contextual interaction |
| `INV_` | `<|INV_EQUIP_MELEE_BEST|>`, `<|INV_EQUIP_FOOD_BEST|>` | Smart inventory |
| `SURV_` | `<|SURV_FLEE_180|>`, `<|SURV_WATER_BUCKET_MLG|>` | Survival instincts (override planner) |
| `SYS_` | `<|SYS_YIELD_TO_MIND|>`, `<|SYS_STUCK|>` | System orchestration |

> `<|SYS_YIELD_TO_MIND|>` is the most common output (~60-70%) — "no threats, continue planner directive".

## Training Pipeline

```bash
# Synthetic dataset (fast fallback)
python generate-synthetic.py --output ./training-data/ --count 2000

# PLAICraft data (research-backed)
./download-plaicraft.sh metadata
python convert-plaicraft.py --input ./plaicraft-data/ --output ./training-data/

# LoRA fine-tune (AMD MI300X / ROCm)
python train-lora.py --data ./training-data/ --output ./fongen-lora/ --epochs 3

# Merge adapters
python merge-lora.py --adapters ./fongen-lora/ --output ./fongen-merged/
```

Full training docs: see `training/README.md` in this repository.

## Serving

```bash
# vLLM (fastest, ROCm)
./serve-vllm.sh ./fongen-merged/ 8000
# Served as: fongen-mc

# llama.cpp (reliable fallback)
./serve-llamacpp.sh ./fongen-gguf/fongen-mc-q4_k_m.gguf 8000
# Alias: fongen-mc
```

Both expose an **OpenAI-compatible API** at `http://0.0.0.0:8000/v1`.

## Citation

```bibtex
@misc{fongen-mc,
  title={Fongen-MC: A Fine-tuned Gemma 4 12B for Minecraft Embodied Planning},
  year={2026}
}

@article{plaicraft,
  title={PLAICraft: A Large Open Multimodal Behavior Dataset for Minecraft},
  author={Baker, Bowen and others},
  journal={arXiv:2505.12707},
  year={2025}
}
```

## License

MIT — see [LICENSE](LICENSE).
