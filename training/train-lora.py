#!/usr/bin/env python3
"""
train-lora.py — Fine-tune Gemma 4 12B with LoRA on Fongen training data.

Primary path: Unsloth LoRA BF16 on AMD MI300X (ROCm).
Fallback: HuggingFace PEFT + Transformers if Unsloth unavailable.

Usage:
  python train-lora.py --data ./training-data/ --output ./fongen-lora/
  python train-lora.py --data ./training-data/ --model google/gemma-4-12B-it --epochs 3
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_jsonl_dataset(data_dir: str, max_samples: int | None = None):
    """Load JSONL files with chat messages format."""
    from datasets import Dataset

    data_path = Path(data_dir)
    files = list(data_path.glob("*.jsonl"))
    if not files:
        print(f"[ERROR] No .jsonl files found in {data_dir}")
        sys.exit(1)

    all_examples = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                example = json.loads(line)
                if "messages" in example:
                    all_examples.append(example)

    if max_samples:
        all_examples = all_examples[:max_samples]

    print(f"[INFO] Loaded {len(all_examples)} examples from {len(files)} files")
    return Dataset.from_list(all_examples)


def train_with_unsloth(model_name, dataset, output_dir, config):
    """Train using Unsloth (preferred for AMD ROCm)."""
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainingArguments

    print(f"[INFO] Loading model with Unsloth: {model_name}")

    max_seq_length = config["max_seq_length"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto: BF16 on MI300X
        load_in_4bit=config["load_in_4bit"],
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Ensure chat template is set
    if tokenizer.chat_template is None:
        print("[WARN] No chat template found, using default Gemma template")
        from unsloth.chat_templates import get_chat_template
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="gemma",
        )

    # Formatting function for chat messages
    def formatting_func(examples):
        convos = examples["messages"]
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in convos
        ]
        return texts

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        formatting_func=formatting_func,
        train_dataset=dataset,
        SFTConfig=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation"],
            warmup_steps=config["warmup_steps"],
            max_steps=config["max_steps"] or -1,
            num_train_epochs=config["epochs"],
            learning_rate=config["learning_rate"],
            fp16=False,
            bf16=True,
            logging_steps=10,
            optim="adamw_torch",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            output_dir=output_dir,
            save_strategy="epoch",
            save_total_limit=2,
            report_to="none",
            max_seq_length=max_seq_length,
        ),
    )

    print("[INFO] Starting training...")
    trainer.train()

    print(f"[INFO] Saving LoRA adapters to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer


def train_with_peft(model_name, dataset, output_dir, config):
    """Fallback: Train using HuggingFace PEFT + Transformers."""
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        DataCollatorForSeq2Seq,
    )
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig

    print(f"[INFO] Loading model with HuggingFace: {model_name}")

    max_seq_length = config["max_seq_length"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() or torch.backends.mps.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",  # Safer for ROCm
    )

    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def formatting_func(examples):
        convos = examples["messages"]
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in convos
        ]
        return texts

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        formatting_func=formatting_func,
        train_dataset=dataset,
        SFTConfig=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation"],
            warmup_steps=config["warmup_steps"],
            max_steps=config["max_steps"] or -1,
            num_train_epochs=config["epochs"],
            learning_rate=config["learning_rate"],
            fp16=False,
            bf16=True,
            logging_steps=10,
            optim="adamw_torch",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            output_dir=output_dir,
            save_strategy="epoch",
            save_total_limit=2,
            report_to="none",
            max_seq_length=max_seq_length,
        ),
    )

    print("[INFO] Starting training (PEFT fallback)...")
    trainer.train()

    print(f"[INFO] Saving LoRA adapters to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 4 12B with LoRA for Fongen")
    parser.add_argument("--data", required=True, help="Directory with JSONL training data")
    parser.add_argument("--output", default="./fongen-lora", help="Output directory for LoRA adapters")
    parser.add_argument("--model", default="google/gemma-4-12B-it", help="Base model ID")
    parser.add_argument("--epochs", type=float, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout")
    parser.add_argument("--max-steps", type=int, default=0, help="Max steps (0 = use epochs)")
    parser.add_argument("--warmup-steps", type=int, default=20, help="Warmup steps")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples (for quick test)")
    parser.add_argument("--no-unsloth", action="store_true", help="Skip Unsloth, use PEFT directly")
    parser.add_argument("--qlora", action="store_true", help="Use 4-bit QLoRA (experimental on ROCm)")
    args = parser.parse_args()

    config = {
        "max_seq_length": args.max_seq_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.grad_accum,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "warmup_steps": args.warmup_steps,
        "load_in_4bit": args.qlora,
    }

    print("=" * 60)
    print("  Fongen LoRA Fine-Tuning")
    print(f"  Model: {args.model}")
    print(f"  Data: {args.data}")
    print(f"  Output: {args.output}")
    print(f"  Config: {config}")
    print("=" * 60)

    dataset = load_jsonl_dataset(args.data, args.max_samples)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    use_unsloth = not args.no_unsloth
    if use_unsloth:
        try:
            import unsloth  # noqa: F401
        except ImportError:
            print("[WARN] Unsloth not available, falling back to HuggingFace PEFT")
            use_unsloth = False

    if use_unsloth:
        try:
            model, tokenizer = train_with_unsloth(args.model, dataset, args.output, config)
        except Exception as e:
            print(f"[ERROR] Unsloth training failed: {e}")
            print("[INFO] Falling back to HuggingFace PEFT...")
            model, tokenizer = train_with_peft(args.model, dataset, args.output, config)
    else:
        model, tokenizer = train_with_peft(args.model, dataset, args.output, config)

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print(f"  Adapters saved to: {args.output}")
    print("=" * 60)
    print("\nNext steps:")
    print(f"  1. Merge adapters: python merge-lora.py --adapters {args.output} --model {args.model}")
    print(f"  2. Export GGUF: ./export-gguf.sh {args.output}-merged")
    print(f"  3. Serve with vLLM or llama.cpp")


if __name__ == "__main__":
    main()
