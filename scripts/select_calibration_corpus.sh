#!/bin/sh
set -eu

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 PROFILE_MODEL_DIR CANDIDATES.jsonl SELECTED.jsonl [REFERENCE_PROFILES.jsonl]" >&2
    exit 2
fi

profile_model_dir=$1
candidates=$2
selected=$3
reference_profiles=${4:-}
profiles=${PROFILES_JSONL:-${candidates%.jsonl}.profiles.jsonl}
manifest="$candidates.manifest.json"

if [ ! -f "$manifest" ]; then
    echo "missing candidate manifest: $manifest" >&2
    exit 1
fi
if [ -e "$profiles" ] || [ -e "$profiles.manifest.json" ] || [ -e "$selected" ] || [ -e "$selected.manifest.json" ]; then
    echo "refusing to overwrite an existing profile/selection output" >&2
    exit 1
fi

candidate_count=$(jq -er '.records' "$manifest")
selected_count=$(jq -er '.nsamples' "$manifest")
max_seqlen=$(jq -er '.seqlen' "$manifest")

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(dirname -- "$script_dir")
cd "$repo_dir"
cargo build --release --bin gb10_inference --bin calib_select

set -- "$repo_dir/target/release/gb10_inference" --calib-profile \
    --model-dir "$profile_model_dir" \
    --calib "$candidates" \
    --out "$profiles" \
    --nsamples "$candidate_count" \
    --seqlen "$max_seqlen" \
    --profile-layers "${PROFILE_LAYERS:-auto}" \
    --profile-sketch-dim "${PROFILE_SKETCH_DIM:-16}"
if [ -n "${PROFILE_BASE:-}" ]; then
    set -- "$@" --base "$PROFILE_BASE"
fi
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} GB10_PLE_OFFLOAD=${GB10_PLE_OFFLOAD:-ssd} "$@"

set -- "$repo_dir/target/release/calib_select" \
    --candidates "$candidates" \
    --profiles "$profiles" \
    --output "$selected" \
    --nsamples "$selected_count" \
    --cola-weight "${COLA_WEIGHT:-1}" \
    --acdm-weight "${ACDM_WEIGHT:-1}" \
    --expert-weight "${EXPERT_WEIGHT:-1}" \
    --kmeans-iters "${KMEANS_ITERS:-6}" \
    --seed "${SELECTION_SEED:-20260831}"
if [ -n "$reference_profiles" ]; then
    set -- "$@" --reference-profiles "$reference_profiles"
fi
"$@"

echo "[select] profiles: $profiles"
echo "[select] selected: $selected"
