#!/usr/bin/env bash
set -euo pipefail

# Weight-only baseline: every W4 MR-GPTQ GEMM consumes BF16/A16 activations.
unset GB10_W4A4_PREFILL
unset GB10_W4A4_LMHEAD_NARROW

exec /home/kedric/workspace/veloGB10/target/release/gb10_inference \
  --server \
  --model-dir /home/kedric/models/Qwen3.8-27B-MR-GPTQ-LMHEAD-W4A16 \
  --max-seq-len 226114 --max-batch 2 \
  --prefix-cache on --mtp auto \
  "$@"
