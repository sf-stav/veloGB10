#!/usr/bin/env bash
set -euo pipefail

if (( $# < 5 || $# > 6 )); then
    echo "Usage: $0 SRC BASE CANDIDATES.jsonl CALIB.jsonl FINAL_DIR [REFERENCE_PROFILES.jsonl]" >&2
    exit 2
fi

src=$1
base=$2
candidates=$3
calib=$4
final=$5
reference_profiles=${6:-}

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(dirname -- "$script_dir")
profiles=${PROFILES_JSONL:-${candidates%.jsonl}.profiles.jsonl}
gptq_log=${GPTQ_LOG:-/var/tmp/maca-layerwise-gptq.log}
selection_mode=${SELECTION_MODE:-direct}

for required in "$src/config.json" "$candidates" "$candidates.manifest.json"; do
    if [[ ! -e "$required" ]]; then
        echo "missing required input: $required" >&2
        exit 1
    fi
done

cd "$repo_dir"

if [[ ! -e "$base/config.json" ]]; then
    echo "[pipeline] creating NVFP4 base: $base"
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    GB10_QUANT_SHARD_GB=${GB10_QUANT_SHARD_GB:-4} \
        ./target/release/gb10_inference --quantize \
        --model-dir "$src" \
        --out "$base" \
        --recipe all
else
    echo "[pipeline] reusing NVFP4 base: $base"
fi

unset GB10_W4A4_PREFILL GB10_W4A4_VERIFY GB10_W4A4_LMHEAD_NARROW
unset GB10_W4A4_CHECK GB10_W4A4_TRACE

case "$selection_mode" in
direct)
    # The non-reserve prefix is already token-budgeted, length-balanced and
    # category-balanced by generate_calibration_corpus.sh.  Feeding it straight
    # to GPTQ avoids an additional full-model profiling pass.
    calib_input=$candidates
    nsamples=$(jq -er '.nsamples' "$candidates.manifest.json")
    echo "[pipeline] direct MaCa corpus: first $nsamples budgeted candidates"
    ;;
profiled)
    if [[ ! -e "$calib" || ! -e "$calib.manifest.json" ]]; then
        echo "[pipeline] sequential BF16 layer profiling and corpus selection"
        select_args=("$src" "$candidates" "$calib")
        if [[ -n "$reference_profiles" ]]; then
            select_args+=("$reference_profiles")
        fi
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
        GB10_PLE_OFFLOAD=${GB10_PLE_OFFLOAD:-ssd} \
        PROFILE_BASE="$base" \
        PROFILE_LAYERS=${PROFILE_LAYERS:-auto} \
        PROFILE_SKETCH_DIM=${PROFILE_SKETCH_DIM:-16} \
        COLA_WEIGHT=${COLA_WEIGHT:-1} \
        ACDM_WEIGHT=${ACDM_WEIGHT:-1} \
        EXPERT_WEIGHT=${EXPERT_WEIGHT:-1} \
        KMEANS_ITERS=${KMEANS_ITERS:-6} \
        SELECTION_SEED=${SELECTION_SEED:-20260831} \
        PROFILES_JSONL="$profiles" \
            "$script_dir/select_calibration_corpus.sh" "${select_args[@]}"
    else
        echo "[pipeline] reusing selected calibration corpus: $calib"
    fi
    calib_input=$calib
    nsamples=$(jq -er '.selected_count' "$calib.manifest.json")
    ;;
*)
    echo "invalid SELECTION_MODE=$selection_mode (expected direct or profiled)" >&2
    exit 2
    ;;
esac

if [[ -e "$final" ]]; then
    echo "refusing to overwrite final output: $final" >&2
    exit 1
fi

seqlen=$(jq -er '.seqlen' "$candidates.manifest.json")

echo "[pipeline] starting layer-wise MR-GPTQ: nsamples=$nsamples seqlen=$seqlen"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
GB10_PLE_OFFLOAD=${GB10_PLE_OFFLOAD:-ssd} \
    ./target/release/gb10_inference --gptq \
    --model-dir "$src" \
    --base "$base" \
    --out "$final" \
    --calib "$calib_input" \
    --nsamples "$nsamples" \
    --seqlen "$seqlen" \
    --maca \
    --damp 0.01 \
    --clip 7 \
    --rotate \
    --scale-iters 4 \
    --gptq-groups expert,attn,mlp,gdn,lmhead \
    --rtn-groups mtp,embed \
    2>&1 | tee "$gptq_log"

echo "[pipeline] complete: $final"
