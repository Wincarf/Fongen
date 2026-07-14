#!/usr/bin/env python3
"""
merge-lora.py — Merge LoRA adapters into base model for serving.

Usage:
  python merge-lora.py --adapters ./fongen-lora/ --model google/gemma-4-12B-it --output ./fongen-merged/
"""
import argparse
from pathlib import Path


def merge_with_unsloth(adapters_path, model_name, output_path):
    from unsloth import FastLanguageModel

    print(f"[INFO] Loading adapters with Unsloth: {adapters_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapters_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
    )

    print(f"[INFO] Merging and saving to: {output_path}")
    model.save_pretrained_merged(
        output_path,
        tokenizer,
        save_method="merged_16bit" if False else "merged_4bit" if False else "merged_16bit",
    )


def merge_with_peft(adapters_path, model_name, output_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"[INFO] Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dtype = torch.bfloat16
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="cpu",
    )

    print(f"[INFO] Loading LoRA adapters: {adapters_path}")
    model = PeftModel.from_pretrained(base_model, adapters_path)

    print("[INFO] Merging adapters...")
    model = model.merge_and_unload()

    print(f"[INFO] Saving merged model to: {output_path}")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into base model")
    parser.add_argument("--adapters", required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--model", default="google/gemma-4-12B-it", help="Base model ID")
    parser.add_argument("--output", required=True, help="Output directory for merged model")
    parser.add_argument("--no-unsloth", action="store_true", help="Use PEFT instead of Unsloth")
    args = parser.parse_args()

    use_unsloth = not args.no_unsloth
    if use_unsloth:
        try:
            import unsloth  # noqa: F401
        except ImportError:
            use_unsloth = False

    if use_unsloth:
        try:
            merge_with_unsloth(args.adapters, args.model, args.output)
        except Exception as e:
            print(f"[ERROR] Unsloth merge failed: {e}")
            print("[INFO] Falling back to PEFT...")
            merge_with_peft(args.adapters, args.model, args.output)
    else:
        merge_with_peft(args.adapters, args.model, args.output)

    print(f"\n[DONE] Merged model saved to: {args.output}")
    print("Next: export to GGUF with export-gguf.sh")


if __name__ == "__main__":
    main()
