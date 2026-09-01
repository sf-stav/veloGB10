#!/usr/bin/env bash
set -euo pipefail

if (( $# != 5 )); then
    echo "Usage: $0 QUANT_SERVICE FINAL_DIR MAIN_CALIB.jsonl LONG_CALIB.jsonl WORK_DIR" >&2
    exit 2
fi

quant_service=$1
final=$2
main_calib=$3
long_calib=$4
work_dir=$5

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(dirname -- "$script_dir")
main_out=$work_dir/main
long_out=$work_dir/long

echo "[igs-pipeline] waiting for $quant_service"
while systemctl --user is-active --quiet "$quant_service"; do
    sleep 30
done

quant_result=$(systemctl --user show "$quant_service" --property=Result --value)
quant_status=$(systemctl --user show "$quant_service" --property=ExecMainStatus --value)
if [[ "$quant_result" != success || "$quant_status" != 0 ]]; then
    echo "quantization did not finish successfully: result=$quant_result status=$quant_status" >&2
    exit 1
fi

for required in \
    "$final/config.json" \
    "$final/model.safetensors.index.json" \
    "$main_calib" \
    "$main_calib.manifest.json" \
    "$long_calib" \
    "$long_calib.manifest.json"; do
    if [[ ! -s "$required" ]]; then
        echo "missing required input: $required" >&2
        exit 1
    fi
done
for output in "$work_dir" "$final/input_global_scale.json" "$final/input_global_scale.stats.json"; do
    if [[ -e "$output" ]]; then
        echo "refusing to overwrite IGS output: $output" >&2
        exit 1
    fi
done

main_nsamples=$(jq -er '.nsamples' "$main_calib.manifest.json")
main_seqlen=$(jq -er '.seqlen' "$main_calib.manifest.json")
long_nsamples=$(jq -er '.nsamples' "$long_calib.manifest.json")
long_seqlen=$(jq -er '.seqlen' "$long_calib.manifest.json")

cd "$repo_dir"
cargo build --release --bin gb10_inference --bin calib_igs

unset GB10_W4A4_PREFILL GB10_W4A4_VERIFY GB10_W4A4_LMHEAD_NARROW
unset GB10_W4A4_CHECK GB10_W4A4_TRACE

echo "[igs-pipeline] main: $main_nsamples samples, max seqlen $main_seqlen"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
GB10_PLE_OFFLOAD=${GB10_PLE_OFFLOAD:-ssd} \
    ./target/release/gb10_inference --calib-igs \
    --model-dir "$final" \
    --out "$main_out" \
    --calib "$main_calib" \
    --nsamples "$main_nsamples" \
    --seqlen "$main_seqlen" \
    --igs-method headroom \
    --igs-anchor-percentile 1 \
    --igs-upper-percentile 99.99 \
    --igs-rho 16384

echo "[igs-pipeline] long: $long_nsamples samples, seqlen $long_seqlen"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
GB10_PLE_OFFLOAD=${GB10_PLE_OFFLOAD:-ssd} \
    ./target/release/gb10_inference --calib-igs \
    --model-dir "$final" \
    --out "$long_out" \
    --calib "$long_calib" \
    --nsamples "$long_nsamples" \
    --seqlen "$long_seqlen" \
    --igs-method headroom \
    --igs-anchor-percentile 1 \
    --igs-upper-percentile 99.99 \
    --igs-rho 16384

./target/release/calib_igs merge \
    --output "$final/input_global_scale.json" \
    "$main_out/input_global_scale.json" \
    "$long_out/input_global_scale.json"

scale_count=$(jq -er 'length' "$final/input_global_scale.json")
jq -e 'to_entries | all(.value; type == "number" and isfinite and . > 0)' \
    "$final/input_global_scale.json" >/dev/null
echo "[igs-pipeline] complete: $scale_count positive finite scales -> $final/input_global_scale.json"
