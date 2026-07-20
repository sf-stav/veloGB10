"""
Download and save original Qwen3.5-0.8B model for conversion.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def download_model(model_name: str, output_dir: str):
    print(f"Downloading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

    import os
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir, safe_serialization=True)
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--output", default="./qwen3.5-0.8b-original")
    args = parser.parse_args()
    download_model(args.model, args.output)
