# Qwen3.8-27B v5 reproduction: corpus, MR-GPTQ, W4A4 and serving

This is the canonical recipe used for the validated `Qwen3.8-27B-MR-GPTQ-NVFP4-v5` build.
It produces W4 MR-GPTQ weights, W4A4 prefill for the transformer trunk, and deliberately keeps
the main, MTP, and DFlash2 output-head activations in A16.

For the newer variable-length MaCa pipeline with activation-aware COLA/ACDM selection and MoE
expert balancing, use [`MACA_COLA_ACDM_MOE_CALIBRATION.md`](MACA_COLA_ACDM_MOE_CALIBRATION.md).

The generator builds the corpus from pinned, verified raw sources and never overwrites an existing
output. Its main output contains one exact, pre-tokenized sample per JSONL record; GPTQ consumes
`input_ids` directly, so the manifest token percentages are the percentages used by calibration.

## Prerequisites and RTN base

Run from the repository root and make sure no other large GPU process is active:

```bash
cd "$HOME/workspace/veloGB10"

export GPU_ID=0
export SRC="$HOME/models/Qwen3.8-27B"
export BASE="$HOME/models/Qwen3.8-27B-NVFP4-base"
export FINAL="$HOME/models/Qwen3.8-27B-MR-GPTQ-NVFP4-v5"
export CALIB="$HOME/models/calibration-sources/qwen38-calibration-v5-mt15-code25-multi25-tools20-math10-pi5.jsonl"
export LONG_CALIB="$HOME/models/calibration-sources/qwen38-calibration-v5-mt15-code25-multi25-tools20-math10-pi5.long-8192.jsonl"
export DRAFT="$HOME/models/Qwen3.8-27B-DFlash2"

cargo build --release
test -f "$SRC/tokenizer.json"
test -f assets/calibration/prompt_injection.jsonl
```

The prompt-injection seed asset is part of the reproducible input and must be versioned with the
repository. Its expected SHA-256 for this build is
`678d46da4a54019b83c8e19188cb00a2d47f2939b08a7e0a558eeb21af2916c6`.

Create the reusable RTN NVFP4 base only when it does not already exist:

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" GB10_QUANT_SHARD_GB=4 \
./target/release/gb10_inference --quantize \
  --model-dir "$SRC" \
  --out "$BASE" \
  --recipe all \
  2>&1 | tee /var/tmp/qwen38_27b_nvfp4_base.log
```

## Generate from zero

```bash
cd ~/workspace/veloGB10

SEED=20260830 BOOTSTRAP=1 scripts/generate_calibration_corpus.sh \
  "$HOME/models/Qwen3.8-27B" \
  "$HOME/models/calibration-sources/qwen38-calibration-v5-mt15-code25-multi25-tools20-math10-pi5.jsonl"
```

Optional environment variables:

- `SEED=20260830` is the canonical deterministic seed;
- `BOOTSTRAP=1` downloads or checks all pinned raw sources;
- `EXCLUDE_JSONL=/path/to/held-out-benchmark.jsonl` removes exact and near-duplicate benchmark
  material;
- `VISION_DIR=/path/to/representative/images` adds local PNG/JPEG/WebP images;
- `NSAMPLES`, `SEQLEN`, `LONG_NSAMPLES`, and `LONG_SEQLEN` override the defaults.

The default outputs are:

- `qwen38-calibration-v5-mt15-code25-multi25-tools20-math10-pi5.jsonl`: 512 x 2048, exact 15% long multi-turn, 25% code,
  25% multilingual, 20% tools/structured, 10% verified mathematical reasoning, and 5% prompt-injection defense;
- `qwen38-calibration-v5-mt15-code25-multi25-tools20-math10-pi5.long-8192.jsonl`: 64 x 8192 long-context samples;
- `qwen38-calibration-v5-mt15-code25-multi25-tools20-math10-pi5.jsonl.sources/vision_multimodal.jsonl`: optional raw multimodal pool;
- one manifest beside each composed corpus plus `sources.manifest.json` for source hashes,
  licensing metadata, deduplication counts, languages, scenarios, and code-language coverage.

For the exact validated build, verify the composed outputs before GPTQ:

```bash
sha256sum "$CALIB" "$LONG_CALIB"
```

Expected SHA-256 values:

- main: `62fa1f7ba19ce5f689bcb7650b57fa8e33a931615aab1e82e9da595f42a93084`;
- long: `765234e3c59bd00173f2dc927e818d1b49ca66b212f0d09de745849b254b2bc1`.

Prompt-injection examples protect activation coverage for that traffic. Calibration is not
fine-tuning: it cannot teach a missing safety behavior or guarantee resistance to attacks.

## Main MR-GPTQ pass

Use the main 512 x 2048 corpus for the layer-wise Hessian pass. Sampling parameters such as temperature, top-p, top-k, and min-p are not calibration inputs and must not be added.

```bash
unset GB10_W4A4_PREFILL
unset GB10_W4A4_LMHEAD_NARROW
unset GB10_W4A4_CHECK
unset GB10_W4A4_TRACE

CUDA_VISIBLE_DEVICES="$GPU_ID" GB10_PLE_OFFLOAD=ssd \
./target/release/gb10_inference --gptq \
  --model-dir "$SRC" \
  --base "$BASE" \
  --out "$FINAL" \
  --calib "$CALIB" \
  --nsamples 512 \
  --seqlen 2048 \
  --damp 0.01 \
  --clip 7 \
  --rotate \
  --scale-iters 4 \
  --gptq-groups attn,mlp,gdn,lmhead \
  --rtn-groups mtp,embed \
  2>&1 | tee /var/tmp/qwen38_27b_mr_gptq_nvfp4_v5.log
```

Validate the resulting metadata before collecting A4 activation scales:

```bash
test -f "$FINAL/config.json"
test -f "$FINAL/model.safetensors.index.json"
jq ".quantization_config" "$FINAL/config.json"
```

Static activation-order GPTQ is enabled by default. Do not pass `--no-act-order` for this recipe.

## W4A4 activation-scale passes

These passes do not re-run GPTQ and do not modify weights. The exact v5 text/tool build merges
the main 512 x 2048 corpus with the 64 x 8192 long-context companion. Each pass records a
512-bin log2 histogram of per-16 activation block maxima, plus the literal maximum. The default
`headroom` policy follows NVIDIA ModelOpt: P1 anchor, P99.99 upper bound, and
`selected_amax = max(upper, 16384 * anchor)`.

```bash
mkdir -p /tmp/qwen38-igs-v5
unset GB10_W4A4_PREFILL
unset GB10_W4A4_VERIFY

CUDA_VISIBLE_DEVICES="$GPU_ID" \
./target/release/gb10_inference --calib-igs \
  --model-dir "$FINAL" \
  --out /tmp/qwen38-igs-v5/main \
  --calib "$CALIB" \
  --nsamples 512 \
  --seqlen 2048 \
  --igs-method headroom \
  --igs-anchor-percentile 1 \
  --igs-upper-percentile 99.99 \
  --igs-rho 16384

CUDA_VISIBLE_DEVICES="$GPU_ID" \
./target/release/gb10_inference --calib-igs \
  --model-dir "$FINAL" \
  --out /tmp/qwen38-igs-v5/long \
  --calib "$LONG_CALIB" \
  --nsamples 64 \
  --seqlen 8192 \
  --igs-method headroom \
  --igs-anchor-percentile 1 \
  --igs-upper-percentile 99.99 \
  --igs-rho 16384
```

Each output directory contains both `input_global_scale.json` and the mergeable
`input_global_scale.stats.json`. Merge the histograms first and derive the scale only once from
the union; do not take the minimum of already-derived headroom scales:

```bash
cargo build --release --bin calib_igs

target/release/calib_igs merge \
  --output "$FINAL/input_global_scale.json" \
  /tmp/qwen38-igs-v5/main/input_global_scale.json \
  /tmp/qwen38-igs-v5/long/input_global_scale.json

test "$(jq length "$FINAL/input_global_scale.json")" -eq 496
```

The merge also writes `$FINAL/input_global_scale.stats.json` and a manifest. Inspect
`range_exceeds_e4m3_stems` in the manifest: a non-empty list indicates a distribution too wide
for one E4M3 global scale to cover without compromise. A complete dense artifact contains 496
positive finite scales. Add a real representative vision corpus and a third merge input only for a
multimodal deployment; the validated tool-eval build used the main and long text corpora only.

## Serve v5: W4A4 trunk with W4A16 heads

The `lm_head` weight remains W4 MR-GPTQ, but its input activation stays BF16/A16. The MTP path
and DFlash2 borrow the same main output head, so this also keeps their head GEMMs in W4A16.
Do not set `GB10_W4A4_LMHEAD_NARROW=1`, and do not list `lmhead` in
`GB10_W4A4_PREFILL` for this validated profile.

```bash
unset GB10_W4A4_LMHEAD_NARROW

CUDA_VISIBLE_DEVICES="$GPU_ID" \
GB10_W4A4_PREFILL=attn,mlp,gdn \
./target/release/gb10_inference \
  --server \
  --model-dir "$FINAL" \
  --draft-dir "$DRAFT" \
  --spec-source dflash2-auto \
  --port 9000 \
  --max-seq-len 226114 \
  --max-batch 1 \
  --prefix-cache on \
  --mtp auto
```

This profile means:

- transformer prefill for `attn`, `mlp`, and `gdn`: W4A4;
- decode and speculative verification: W4A16;
- main `lm_head`, the MTP head use, and the head borrowed by DFlash2: W4A16;
- DFlash2-auto: per-request DFlash2 routing with MTP as the automatic fallback;
- prefix cache: enabled.

Expected startup evidence includes `MR-GPTQ transform: hadamard16`, `W4A4 prefill ON` for the
three trunk groups, `borrowed lm_head uses hadamard16`, and `DFlash2 round RESIDENT`. If the
W4A4 environment variable is omitted, the artifact serves entirely as W4A16.

Optional held-out tool evaluation:

```bash
tool-eval-bench \
  --base-url http://127.0.0.1:9000 \
  --model qwen3.8-27b \
  --hardmode \
  --label velo-qwen3.8-27B-v5-W4A4-trunk-W4A16-heads-HA16-dflash2 \
  --output-dir ../teb
```

## Domain audit

Run this on an existing quantized artifact when comparing calibration domains. It does not alter
the artifact:

```bash
scripts/audit_calibration_igs.sh \
  "$FINAL" \
  "$HOME/models/calibration-sources/qwen38-calibration-v5-mt15-code25-multi25-tools20-math10-pi5.jsonl" \
  /tmp/qwen38-domain-audit
```

`report.json` reconstructs per-domain activation maxima and flags tensors whose largest domain
maximum is at least 1.5 times the smallest. This is a coverage diagnostic, not an accuracy score;
quality still needs held-out perplexity/task evaluation and the user's serving benchmark.
