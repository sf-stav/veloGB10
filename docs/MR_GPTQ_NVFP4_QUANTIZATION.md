# MR-GPTQ NVFP4 Quantization for Qwen3.8

For the exact validated dense v5 recipe, including corpus hashes, A4 scale collection, W4A16
heads, DFlash2, and the production launch command, use [`CALIBRATION_CORPUS.md`](CALIBRATION_CORPUS.md).


This guide describes the complete procedure used to produce an NVFP4 artifact for W4A16 or
W4A4 inference. It covers dense models such as `Qwen3.8-27B`, the
`Qwen3.8-Flash-Next` MoE model, RTN base conversion, MR-GPTQ calibration, validation, and
W4A4 smoke testing.

The commands assume that the repository is located at `~/workspace/veloGB10` and that all
models are stored under `~/models`.

## 1. Hardware and storage requirements

Check the GPU, memory, and available disk space:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
free -h
df -h "$HOME/models"
```

Recommendations:

- approximately 128 GB of available or unified memory;
- no other large process running on the selected GPU;
- at least 150 GB of free storage for the 27B model;
- at least 650 GB for Flash-Next, preferably 750 GB.

On a multi-GPU machine, select the target GPU with:

```bash
export GPU_ID=0
```

If several GPUs share host memory, ensure that parallel jobs will not exhaust the system's
available memory.

## 2. Build and test

```bash
cd ~/workspace/veloGB10

cargo build --release
cargo check --release
cargo test --release --lib gptq::tests -- --nocapture
cargo test --release --test gptq_kernels_test -- --nocapture
```

Do not start a long quantization job if the build or the targeted GPTQ tests fail.

## 3. Prepare the source model

### Dense 27B model

The BF16 checkpoint must be available at:

```text
~/models/Qwen3.8-27B
```

### Flash-Next MoE model

Download the official checkpoint with:

```bash
mkdir -p "$HOME/models"

hf download Qwen/Qwen3.8-Flash-Next \
  --local-dir "$HOME/models/Qwen3.8-Flash-Next"
```

The checkpoint must then be available at:

```text
~/models/Qwen3.8-Flash-Next
```

### Checkpoint validation

Set `SRC` to the desired model, then check the required metadata and look for incomplete
downloads:

```bash
export SRC="$HOME/models/Qwen3.8-27B"

test -f "$SRC/config.json"
test -f "$SRC/model.safetensors.index.json"

find "$SRC" \
  -type f \
  \( -name "*.incomplete" -o -name "*.part" \)
```

The final command must not report any file.

## 4. Calibration corpus

The JSONL calibration corpus may contain raw text records:

```json
{"text":"Document, code sample, or conversation..."}
```

It may also contain chat messages:

```json
{"messages":[{"role":"user","content":"Question..."},{"role":"assistant","content":"Answer..."}]}
```

For `--nsamples 512 --seqlen 2048`, the corpus must provide at least 1,048,576 usable tokens.
A target of approximately 1.3 million tokens is recommended.

Suggested composition:

- 15% general long-context, multi-turn conversations;
- 25% code, including shell, Go, TypeScript, JSON, Python, Rust, CUDA/C++, SQL, and web;
- 25% multilingual material;
- 20% tool calls and structured conversations;
- 10% verified mathematical reasoning;
- 5% defensive prompt-injection examples.

Do not include benchmark samples that will later be used to compare the artifacts.

The local mixed calibration corpus can be selected with:

```bash
export CALIB="$HOME/models/calibration-sources/qwen38-calibration-v5-mt15-code25-multi25-tools20-math10-pi5.jsonl"
test -f "$CALIB"
```

The quantizer must report exactly:

```text
[gptq] 512 samples × 2048 tokens
```

If it cannot build 512 complete sequences, expand the corpus before continuing.

## 5. Disable inference modes during calibration

```bash
unset GB10_W4A4_PREFILL
unset GB10_W4A4_LMHEAD_NARROW
unset GB10_W4A4_CHECK
unset GB10_W4A4_TRACE
unset RUST_INFER_FAKE_QUANT

set -o pipefail
```

This prevents inference-only modes from changing the activations collected during calibration.

## 6. Create the RTN NVFP4 base artifact

The RTN base allows the quantizer to load the model within the available memory budget. It also
provides weights that are not recalibrated, the MTP head, and, for Flash-Next, the PLE table.

A complete and validated base artifact can be reused. It does not need to be regenerated before
every GPTQ pass.

### Dense 27B base

```bash
export SRC="$HOME/models/Qwen3.8-27B"
export BASE="$HOME/models/Qwen3.8-27B-NVFP4-base"

test ! -e "$BASE"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
GB10_QUANT_SHARD_GB=4 \
./target/release/gb10_inference --quantize \
  --model-dir "$SRC" \
  --out "$BASE" \
  --recipe all \
  2>&1 | tee /var/tmp/qwen38_27b_nvfp4_base.log
```

### Flash-Next MoE base

```bash
export SRC="$HOME/models/Qwen3.8-Flash-Next"
export BASE="$HOME/models/Qwen3.8-Flash-Next-NVFP4-BASE"

test ! -e "$BASE"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
GB10_QUANT_SHARD_GB=4 \
./target/release/gb10_inference --quantize \
  --model-dir "$SRC" \
  --out "$BASE" \
  --recipe all \
  2>&1 | tee /var/tmp/qwen38_flash_next_nvfp4_base.log
```

After creating the base, validate its main files and size:

```bash
test -f "$BASE/config.json"
test -f "$BASE/model.safetensors.index.json"
du -sh "$BASE"
```

Continue only if the command exited successfully and every shard referenced by the index exists.

## 7. MR-GPTQ for the dense 27B model

The 27B checkpoint is a dense hybrid model. GPTQ calibrates these groups:

- `attn`: full-attention projections;
- `mlp`: dense MLP projections;
- `gdn`: linear-attention projections;
- `lmhead`: output head, calibrated from the final-mixer outputs.

The embedding and MTP groups use RTN NVFP4.

```bash
cd ~/workspace/veloGB10

export GPU_ID=0
export SRC="$HOME/models/Qwen3.8-27B"
export BASE="$HOME/models/Qwen3.8-27B-NVFP4-base"
export FINAL="$HOME/models/Qwen3.8-27B-MR-GPTQ-NVFP4-v5"
export CALIB="$HOME/models/calibration-sources/qwen38-calibration-v5-mt15-code25-multi25-tools20-math10-pi5.jsonl"

test -d "$SRC"
test -d "$BASE"
test -f "$CALIB"
test ! -e "$FINAL"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
GB10_PLE_OFFLOAD=ssd \
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

Do not add:

- `--no-act-order`, because static activation order must remain enabled;
- `--mxfp4`, because it is incompatible with this MR-GPTQ recipe;
- `--gptq-lmhead`, because `lmhead` is already handled by the main GPTQ pass.

## 8. MR-GPTQ for the Flash-Next MoE model

The primary groups use MR-GPTQ:

- routed experts;
- attention projections;
- shared MLP projections;
- `lm_head`.

Sensitive groups or groups outside the main Hessian-capture path use RTN NVFP4:

- GDN projections;
- hyper-connection mixers;
- PLE projections;
- routers;
- embeddings;
- MTP.

```bash
cd ~/workspace/veloGB10

export GPU_ID=0
export SRC="$HOME/models/Qwen3.8-Flash-Next"
export BASE="$HOME/models/Qwen3.8-Flash-Next-NVFP4-BASE"
export FINAL="$HOME/models/Qwen3.8-Flash-Next-MR-GPTQ-NVFP4"
export CALIB="/var/tmp/qwen38_flash_next_calib.jsonl"

test -d "$SRC"
test -d "$BASE"
test -f "$CALIB"
test ! -e "$FINAL"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
GB10_PLE_OFFLOAD=ssd \
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
  --gptq-groups expert,attn,mlp,lmhead \
  --rtn-groups gdn,hc,ple,router,embed,mtp \
  2>&1 | tee /var/tmp/qwen38_flash_next_mr_gptq.log
```

The PLE table and required MTP components are reused from the NVFP4 base artifact.

## 9. Main parameter reference

### `--rotate`

Enables MR-GPTQ with a Hadamard16 micro-rotation. The rotation is applied to weights before
quantization and to the corresponding activations during inference.

### `--scale-iters 4`

Runs up to four iterations of alternating NVFP4 global-scale optimization. A candidate iteration
is retained only when it decreases the quantization SSE.

### Static activation order

Static activation order is enabled by default. It orders columns according to the importance
estimated from the Hessian during GPTQ, without requiring a dynamic runtime permutation.

### `--damp 0.01`

Adds 1% damping to the Hessian to stabilize its factorization and limit extreme compensation
updates.

### `--clip 7`

Tests all seven available clipping ratios and selects the value that minimizes the error for each
processed weight.

## 10. Monitoring a quantization job

Monitor:

- `layer N/64` progress for the 27B model or `layer N/48` for Flash-Next;
- available host or unified memory;
- free disk space;
- CUDA and cuSOLVER errors;
- scale-optimization lines and their SSE values;
- unexpected BF16 fallbacks.

Examples:

```bash
tail -f /var/tmp/qwen38_27b_mr_gptq_nvfp4_v5.log
```

```bash
tail -f /var/tmp/qwen38_flash_next_mr_gptq.log
```

Do not delete a partial artifact without inspecting it first. A fresh run requires an output path
that is absent or empty.

## 11. Validate the final artifact

```bash
test -f "$FINAL/config.json"
test -f "$FINAL/model.safetensors.index.json"

du -sh "$BASE" "$FINAL"
jq '.quantization_config' "$FINAL/config.json"
```

Check that `quantization_config` records:

- the `nvfp4-pack-quantized` format;
- `nsamples = 512`;
- `seqlen = 2048`;
- `damp = 0.01`;
- seven clipping ratios;
- the `hadamard16` transform;
- four scale-optimization iterations;
- static activation order;
- the expected GPTQ and RTN groups.

Normalization parameters, convolutions, non-matrix parameters, and small matrices that are not
compatible with NVFP4 block dimensions may legitimately remain in BF16 or FP32.

## 12. W4A4 smoke test

The conservative mode enables W4A4 only for the primary groups:

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" \
GB10_W4A4_PREFILL=1 \
./target/release/gb10_inference --server \
  --model-dir "$FINAL" \
  --ple-offload ssd \
  --port 9001 \
  --max-seq-len 8192 \
  --max-batch 1 \
  --prefix-cache on \
  --mtp auto
```

`GB10_W4A4_PREFILL=1` enables `expert`, `mlp`, and `attn` when those groups are present.

For the validated dense 27B profile, enable A4 only on the transformer trunk and explicitly keep
the main, MTP, and DFlash2 head activations in A16:

```bash
unset GB10_W4A4_LMHEAD_NARROW
export GB10_W4A4_PREFILL=attn,mlp,gdn
```

The `lm_head` weight is still W4 MR-GPTQ. Only its input activation stays BF16/A16. To run a
separate experimental A4 head comparison, both list `lmhead` and explicitly enable the narrow
head path:

```bash
export GB10_W4A4_PREFILL=attn,mlp,gdn,lmhead
export GB10_W4A4_LMHEAD_NARROW=1
```

Do not use the experimental narrow mode for the validated v5 serving profile.

Test at least:

- a general question in French;
- a general question in English;
- shell generation;
- Go generation;
- TypeScript and JSON generation;
- a tool-calling request if the server exposes tools.

## 13. Context-length errors during serving

A request must satisfy:

```text
prompt_tokens + max_new_tokens + internal headroom <= max_seq_len
```

If the server rejects a request with `context_length_exceeded`, lower `max_tokens` or increase
`--max-seq-len`. Do not automatically allocate the entire nominal remaining context without
preserving the headroom required by MTP and internal buffers.
