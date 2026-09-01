#!/bin/sh
set -eu

if [ "$#" -ne 3 ]; then
    echo "usage: $0 MODEL_DIR PRETOKENIZED_CORPUS_JSONL OUTPUT_DIR" >&2
    exit 2
fi

model_dir=$1
corpus=$2
output_dir=$3
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(dirname -- "$script_dir")
binary="$repo_dir/target/release/gb10_inference"
igs_binary="$repo_dir/target/release/calib_igs"

if [ ! -f "$model_dir/config.json" ] || [ ! -f "$corpus" ]; then
    echo "missing model config or corpus" >&2
    exit 1
fi
if [ -e "$output_dir" ]; then
    echo "refusing to overwrite: $output_dir" >&2
    exit 1
fi
command -v jq >/dev/null 2>&1 || { echo "jq is required" >&2; exit 1; }

seqlen=$(jq -r '.input_ids | length' "$corpus" | head -n 1)
case "$seqlen" in ''|*[!0-9]*) echo "corpus has no valid input_ids" >&2; exit 1 ;; esac

if [ ! -x "$binary" ] || [ ! -x "$igs_binary" ]; then
    cargo build --release --manifest-path "$repo_dir/Cargo.toml" \
        --bin gb10_inference --bin calib_igs
fi
mkdir -p "$output_dir"

for category in $(jq -r '.primary_category' "$corpus" | sort -u); do
    case "$category" in *[!A-Za-z0-9_-]*) echo "unsafe category name: $category" >&2; exit 1 ;; esac
    category_corpus="$output_dir/$category.jsonl"
    category_output="$output_dir/$category"
    jq -c --arg category "$category" 'select(.primary_category == $category)' "$corpus" > "$category_corpus"
    count=$(wc -l < "$category_corpus" | tr -d ' ')
    if [ "$count" -eq 0 ]; then
        continue
    fi
    echo "[igs-audit] $category: $count samples x $seqlen tokens"
    "$binary" --calib-igs \
        --model-dir "$model_dir" \
        --out "$category_output" \
        --calib "$category_corpus" \
        --nsamples "$count" \
        --seqlen "$seqlen"
done

"$igs_binary" audit \
    --root "$output_dir" \
    --output "$output_dir/report.json"
