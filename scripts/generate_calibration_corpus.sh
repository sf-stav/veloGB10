#!/bin/sh
set -eu

usage() {
    cat <<'EOF'
Generate a complete GPTQ calibration corpus from raw, reproducible sources.

Usage:
  scripts/generate_calibration_corpus.sh MODEL_DIR OUTPUT_JSONL [SOURCE_ROOT]

Defaults:
  SOURCE_ROOT=$HOME/models/calibration-sources
  NSAMPLES=512
  SEQLEN=2048
  RESERVE_SEQUENCES=0
  MACA_LENGTHS=  # e.g. 256,512,1024,2048,4096
  TOKEN_BUDGET=  # default NSAMPLES * SEQLEN
  SEED=20260830
  BOOTSTRAP=1   # download/clone and checksum missing raw sources
  LONG_NSAMPLES=64
  LONG_SEQLEN=8192
  VISION_DIR=   # optional directory of representative images
  EXCLUDE_JSONL= # optional held-out benchmark JSONL

Composition in the exact NSAMPLES * SEQLEN prefix consumed by GPTQ:
  15% general long-context, multi-turn
  25% code (TypeScript/JavaScript, Go, shell, JSON/YAML/TOML,
            Python, Rust, CUDA/C/C++, SQL and web)
  25% multilingual (French, Japanese, Korean, German, Spanish,
                    Chinese, Arabic, Portuguese, Russian and code-switching)
  20% tools and structured conversations
  10% mathematical reasoning (verified solutions, algebra, geometry,
      combinatorics, number theory, calculus and LaTeX)
   5% defensive prompt-injection examples

The output, its manifest, the 8192-token companion, and OUTPUT_JSONL.sources/
are never overwritten. The raw vision pool remains in the sources directory.
EOF
}

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    usage >&2
    exit 2
fi

model_dir=$1
output_jsonl=$2
source_root=${3:-$HOME/models/calibration-sources}
nsamples=${NSAMPLES:-512}
seqlen=${SEQLEN:-2048}
reserve_sequences=${RESERVE_SEQUENCES:-0}
maca_lengths=${MACA_LENGTHS:-}
token_budget=${TOKEN_BUDGET:-}
seed=${SEED:-20260830}
bootstrap=${BOOTSTRAP:-1}
long_nsamples=${LONG_NSAMPLES:-64}
long_seqlen=${LONG_SEQLEN:-8192}
long_reserve_sequences=${LONG_RESERVE_SEQUENCES:-0}
long_output=${LONG_OUTPUT_JSONL:-${output_jsonl%.jsonl}.long-8192.jsonl}
vision_dir=${VISION_DIR:-}
exclude_jsonl=${EXCLUDE_JSONL:-}

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(dirname -- "$script_dir")
staging_dir="$output_jsonl.sources"
manifest="$output_jsonl.manifest.json"
injections="$repo_dir/assets/calibration/prompt_injection.jsonl"

if [ ! -f "$model_dir/tokenizer.json" ]; then
    echo "missing tokenizer: $model_dir/tokenizer.json" >&2
    exit 1
fi
long_manifest="$long_output.manifest.json"
for target in "$output_jsonl" "$manifest" "$long_output" "$long_manifest" "$staging_dir"; do
    if [ -e "$target" ]; then
        echo "refusing to overwrite: $target" >&2
        exit 1
    fi
done
mkdir -p "$(dirname -- "$output_jsonl")"

if [ "$bootstrap" = 1 ]; then
    "$script_dir/bootstrap_calibration_sources.sh" "$source_root"
fi

set -- \
    --source-root "$source_root" \
    --repo-root "$repo_dir" \
    --output-dir "$staging_dir" \
    --injection-corpus "$injections" \
    --seed "$seed"
if [ -n "$vision_dir" ]; then
    set -- "$@" --vision-dir "$vision_dir"
fi
if [ -n "$exclude_jsonl" ]; then
    set -- "$@" --exclude-jsonl "$exclude_jsonl"
fi
cd "$repo_dir"
cargo build --release --bin calib_sources
"$repo_dir/target/release/calib_sources" prepare "$@"

cargo build --release --bin calib_compose

set -- "$repo_dir/target/release/calib_compose" \
    --model-dir "$model_dir" \
    --output "$output_jsonl" \
    --nsamples "$nsamples" \
    --seqlen "$seqlen" \
    --reserve-sequences "$reserve_sequences" \
    --source "general_long_multiturn=15:2048:$staging_dir/general_long_multiturn.jsonl" \
    --source "code=25:1024:$staging_dir/code.jsonl" \
    --source "multilingual=25:768:$staging_dir/multilingual.jsonl" \
    --source "tools_structured=20:2048:$staging_dir/tools_structured.jsonl" \
    --source "math_reasoning=10:2048:$staging_dir/math_reasoning.jsonl" \
    --source "prompt_injection=5:768:$staging_dir/prompt_injection.jsonl"
if [ -n "$maca_lengths" ]; then
    set -- "$@" --maca-lengths "$maca_lengths"
fi
if [ -n "$token_budget" ]; then
    set -- "$@" --token-budget "$token_budget"
fi
"$@"

"$repo_dir/target/release/calib_compose" \
    --model-dir "$model_dir" \
    --output "$long_output" \
    --nsamples "$long_nsamples" \
    --seqlen "$long_seqlen" \
    --reserve-sequences "$long_reserve_sequences" \
    --source "general_long_context=100:8192:$staging_dir/general_long_context.jsonl"

echo "[generate] main:   $output_jsonl"
echo "[generate] long:   $long_output"
echo "[generate] vision: $staging_dir/vision_multimodal.jsonl"
