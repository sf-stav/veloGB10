#!/bin/sh
set -eu

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "usage: $0 MODEL_DIR OUTPUT_JSONL [SOURCE_ROOT]" >&2
    exit 2
fi

model_dir=$1
output_jsonl=$2
source_root=${3:-$HOME/models/calibration-sources}
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(dirname -- "$script_dir")
staging_dir="$output_jsonl.sources"
manifest="$output_jsonl.manifest.json"
long_output=${LONG_OUTPUT_JSONL:-${output_jsonl%.jsonl}.long-8192.jsonl}
seed=${SEED:-20260831}
reserve_sequences=${RESERVE_SEQUENCES:-1536}
exclude_jsonl=${EXCLUDE_JSONL:-}

for required in "$model_dir/tokenizer.json" "$repo_dir/assets/calibration/prompt_injection.jsonl"; do
    test -f "$required" || { echo "missing required file: $required" >&2; exit 1; }
done
for target in "$output_jsonl" "$manifest" "$long_output" "$long_output.manifest.json" "$staging_dir"; do
    test ! -e "$target" || { echo "refusing to overwrite: $target" >&2; exit 1; }
done
mkdir -p "$(dirname -- "$output_jsonl")"
"$script_dir/bootstrap_calibration_sources.sh" "$source_root"

set -- \
    --source-root "$source_root" --repo-root "$repo_dir" --output-dir "$staging_dir" \
    --injection-corpus "$repo_dir/assets/calibration/prompt_injection.jsonl" \
    --agentic-reliability-corpus "$source_root/toolace/sequential-tool-use.parquet" \
    --schema-function-corpus "$source_root/johin/function-calling.jsonl" --seed "$seed"
if [ -n "$exclude_jsonl" ]; then
    set -- "$@" --exclude-jsonl "$exclude_jsonl"
fi
cd "$repo_dir"
cargo build --release --bin calib_sources
"$repo_dir/target/release/calib_sources" prepare "$@"
cargo build --release --bin calib_compose
"$repo_dir/target/release/calib_compose" --model-dir "$model_dir" --output "$output_jsonl" \
    --nsamples 661 --seqlen 4096 --token-budget 1048576 --maca-lengths 256,512,1024,2048,4096 \
    --reserve-sequences "$reserve_sequences" \
    --source "general_long_multiturn=14:2048:$staging_dir/general_long_multiturn.jsonl" \
    --source "code=23:1024:$staging_dir/code.jsonl" \
    --source "multilingual=19:768:$staging_dir/multilingual.jsonl" \
    --source "tools_structured=15:2048:$staging_dir/tools_structured.jsonl" \
    --source "agentic_reliability=10:2048:$staging_dir/agentic_reliability.jsonl" \
    --source "schema_function=9:1024:$staging_dir/schema_function.jsonl" \
    --source "math_reasoning=5:2048:$staging_dir/math_reasoning.jsonl" \
    --source "prompt_injection=5:768:$staging_dir/prompt_injection.jsonl"

"$repo_dir/target/release/calib_compose" --model-dir "$model_dir" --output "$long_output" \
    --nsamples 64 --seqlen 8192 \
    --source "general_long_context=100:8192:$staging_dir/general_long_context.jsonl"

echo "[generate-public-v9] main: $output_jsonl"
echo "[generate-public-v9] long: $long_output"
