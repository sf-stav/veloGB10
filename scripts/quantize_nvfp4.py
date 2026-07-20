"""
Convert Qwen3.5-0.8B to a custom NVFP4-like packed format for the inference engine.
This is a helper script - run it once to prepare weights.
"""
import torch
import struct
import numpy as np
from pathlib import Path


def quantize_tensor_simple(tensor: torch.Tensor, block_size: int = 16) -> bytes:
    """
    Simple per-block INT4 quantization (stand-in for true NVFP4).
    Returns packed bytes: [scale (4 bytes float32)] [packed_64bits] ...
    """
    tensor = tensor.contiguous().float()
    num_elements = tensor.numel()
    num_blocks = (num_elements + block_size - 1) // block_size
    output = bytearray()

    for b in range(num_blocks):
        start = b * block_size
        end = min(start + block_size, num_elements)
        block = tensor[start:end]

        max_val = block.abs().max().item()
        if max_val == 0:
            scale = 0.0
        else:
            scale = max_val / 7.0  # Map to [-7, 7]

        # Quantize
        quantized = (block / max(scale, 1e-10)).round().clamp(-7, 7).to(torch.int8)

        # Pack into 4-bit values (signed)
        packed = []
        for i in range(0, len(quantized), 2):
            if i + 1 < len(quantized):
                low = ((quantized[i].item() + 8) & 0x0F)  # map -7..7 to 0..15
                high = ((quantized[i+1].item() + 8) & 0x0F)
                packed.append((high << 4) | low)
            else:
                low = ((quantized[i].item() + 8) & 0x0F)
                packed.append(low)

        # Pad to 8 bytes if needed
        while len(packed) < 8:
            packed.append(0)

        output.extend(struct.pack('<f', scale))
        output.extend(struct.pack('<8B', *packed[:8]))

    return bytes(output)


def convert_model(model_name: str, output_dir: str):
    """
    Download model and convert to packed INT4 format.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Downloading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save tokenizer
    tokenizer.save_pretrained(output_path / "tokenizer")
    print("Tokenizer saved")

    # Convert each linear layer
    weight_data = {}

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weight = module.weight.data
            print(f"Converting {name}: {weight.shape}")

            # Transpose to [out_features, in_features]
            weight_t = weight.t().contiguous()

            # Quantize
            packed = quantize_tensor_simple(weight_t, block_size=16)

            # Store metadata
            weight_data[name] = {
                'shape': list(weight_t.shape),
                'dtype': 'int4_packed',
                'num_blocks': (weight_t.numel() + 15) // 16,
                'data_size': len(packed),
            }

            # Save weight file
            safe_name = name.replace('.', '_').replace('/', '_')
            with open(output_path / f"{safe_name}.bin", "wb") as f:
                f.write(packed)

    # Save metadata
    import json
    with open(output_path / "weights_metadata.json", "w") as f:
        json.dump(weight_data, f, indent=2)

    print(f"\nConverted {len(weight_data)} layers")
    print(f"Output directory: {output_path}")
    model.save_pretrained(output_path / "original", safe_serialization=True)
    print("Original model also saved for reference")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--output", default="./qwen3.5-0.8b-packed")
    args = parser.parse_args()
    convert_model(args.model, args.output)
