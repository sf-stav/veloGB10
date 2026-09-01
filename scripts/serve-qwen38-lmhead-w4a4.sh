#!/usr/bin/env bash
set -euo pipefail

# A4 for wide trunk prefill plus the explicit narrow LM-head path (the head runs at N=1).
export GB10_W4A4_PREFILL=attn,mlp,gdn,lmhead
export GB10_W4A4_LMHEAD_NARROW=1

exec /home/kedric/workspace/veloGB10/target/release/gb10_inference \
  --server \
  --model-dir /home/kedric/models/Qwen3.8-27B-MR-GPTQ-LMHEAD-W4A4 \
  --max-seq-len 226114 --max-batch 2 \
  --prefix-cache on --mtp auto \
  "$@"
