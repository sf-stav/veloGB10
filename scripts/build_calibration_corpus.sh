#!/bin/sh
set -eu

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "usage: $0 MODEL_DIR INPUT_JSONL OUTPUT_JSONL [PROMPT_INJECTION_PERCENT]" >&2
    exit 2
fi

model_dir=$1
input_jsonl=$2
output_jsonl=$3
percent=${4:-5}
nsamples=${NSAMPLES:-512}
seqlen=${SEQLEN:-2048}
chunk_tokens=${CHUNK_TOKENS:-512}
reserve_sequences=${RESERVE_SEQUENCES:-8}

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(dirname -- "$script_dir")
injection_corpus=${INJECTION_CORPUS:-$repo_dir/assets/calibration/prompt_injection.jsonl}

for required in "$model_dir/tokenizer.json" "$input_jsonl" "$injection_corpus"; do
    if [ ! -f "$required" ]; then
        echo "missing required file: $required" >&2
        exit 1
    fi
done

if [ -e "$output_jsonl" ]; then
    echo "refusing to overwrite: $output_jsonl" >&2
    exit 1
fi

cd "$repo_dir"
cargo build --release --bin calib_mix

exec "$repo_dir/target/release/calib_mix" \
    --model-dir "$model_dir" \
    --input "$input_jsonl" \
    --injections "$injection_corpus" \
    --output "$output_jsonl" \
    --percent "$percent" \
    --nsamples "$nsamples" \
    --seqlen "$seqlen" \
    --chunk-tokens "$chunk_tokens" \
    --reserve-sequences "$reserve_sequences"
