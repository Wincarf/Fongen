# Fongen Training Pipeline

Fine-tune **Gemma 4 12B** on PLAICraft Minecraft behavior data to create **`fongen-mc`** — the **reflex model** for the Fongen embodied agent.

The reflex model outputs UMAS action tokens (single-token classification) for vision-based survival, combat, and movement. It does NOT do planning — planning is handled by Gemini 3.5 Flash via Hermes.

## Quick Start (AMD MI300X)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic dataset (fast, no download needed)
python generate-synthetic.py --output ./training-data/ --count 2000

# OR: Download + convert PLAICraft data (slow, 621GB)
./download-plaicraft.sh metadata
python convert-plaicraft.py --input ./plaicraft-data/ --output ./training-data/

# 3. Fine-tune with LoRA
python train-lora.py \
  --data ./training-data/ \
  --output ./fongen-lora/ \
  --model google/gemma-4-12B-it \
  --epochs 3 \
  --batch-size 1 \
  --grad-accum 16

# 4. Merge LoRA adapters into base model
python merge-lora.py \
  --adapters ./fongen-lora/ \
  --model google/gemma-4-12B-it \
  --output ./fongen-merged/

# 5. Export to GGUF (for llama.cpp fallback)
./export-gguf.sh ./fongen-merged/ ./fongen-gguf/ q4_k_m

# 6. Serve the model
# Option A: vLLM (fastest on MI300X)
./serve-vllm.sh ./fongen-merged/ 8000

# Option B: llama.cpp (most reliable fallback)
./serve-llamacpp.sh ./fongen-gguf/fongen-mc-q4_k_m.gguf 8000
```

## Pipeline Overview

```
PLAICraft data ─┐
                ├─→ convert-plaicraft.py ─→ JSONL training data
Synthetic data ─┘                           │
                                             ↓
                                    train-lora.py (Unsloth LoRA BF16)
                                             │
                                             ↓
                                    merge-lora.py
                                             │
                                    ┌────────┴────────┐
                                    ↓                 ↓
                              serve-vllm.sh     export-gguf.sh
                              (MI300X, fast)          │
                                              serve-llamacpp.sh
                                              (any GPU/CPU)
```

## Dataset Formats

### Action Imitation (low-level)
```json
{
  "messages": [
    {"role": "system", "content": "You are Fongen..."},
    {"role": "user", "content": "{\"recent_actions\": [\"MOVE_FORWARD\", \"JUMP\"]}"},
    {"role": "assistant", "content": "{\"action\": \"MINE_TARGET\", \"reason\": \"...\"}"}
  ]
}
```

### Planning (high-level, used by default)
```json
{
  "messages": [
    {"role": "system", "content": "You are Fongen..."},
    {"role": "user", "content": "{\"health\": 20, \"nearby_blocks\": [...]}"},
    {"role": "assistant", "content": "{\"thought\": \"...\", \"goal\": {\"intent\": \"MINE_TASK\", ...}}"}
  ]
}
```

## AMD ROCm Notes

- Use official ROCm Docker images when possible
- Start with LoRA BF16 (not QLoRA) per AMD Unsloth playbook
- `adamw_torch` optimizer is more stable on ROCm
- vLLM ROCm works well on MI300X; keep llama.cpp as fallback
- If Unsloth fails on ROCm, use `--no-unsloth` flag for PEFT fallback

## Fallback Matrix

| Problem | Fallback |
|---------|----------|
| PLAICraft too slow | `generate-synthetic.py` |
| Unsloth fails on ROCm | `--no-unsloth` (PEFT) |
| 12B too large | `--model google/gemma-4-12B-it` → E4B variant |
| vLLM fails | `serve-llamacpp.sh` (llama.cpp) |
| No GPU at all | llama.cpp CPU mode, or rule-based SafetyGuard fallback |

## Citation

PLAICraft: Baker et al., "PLAICraft: A Large Open Multimodal Behavior Dataset for Minecraft" (arXiv:2505.12707)
