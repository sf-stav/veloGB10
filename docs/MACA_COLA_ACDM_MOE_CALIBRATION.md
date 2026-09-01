# MaCa + COLA/ACDM + MoE expert-balanced calibration

This pipeline builds a variable-length calibration candidate pool, profiles its activations on the
unquantized model, selects the final GPTQ prefix, and preserves the requested category and sequence-
length composition exactly.

The three selection signals are complementary:

- **COLA** clusters deterministic CountSketch summaries of activations and rewards representative
  candidates across the activation space.
- **ACDM** measures standardized per-layer mean/std distance to a task-reference activation
  centroid. Without a reference profile file, the candidate-pool centroid is used and the manifest
  labels the result as a proxy rather than task-aligned ACDM.
- **MoE balance** uses exact router counts for every profiled layer and rewards routes to experts
  that are still underrepresented. Dense models simply report `kind: dense` and ignore this term.

The selected manifest records the weights, dimensions, exact quota target/actual values, expert
coverage statistics, hashes, and whether ACDM used a true task reference or the proxy.

## 1. Generate a candidate pool with MaCa lengths

Run from the repository root. `RESERVE_SEQUENCES` adds candidates beyond the exact GPTQ token
budget; those extra records are available to COLA/ACDM without changing the final budget.

For the public-only v9 recipe (pinned revisions, SHA-256 verification, no local checkout content,
and an entirely Rust data-transformation path), use
[`PUBLIC_CALIBRATION_DATASETS.md`](PUBLIC_CALIBRATION_DATASETS.md) instead of the legacy recipe
shown below.

```bash
export SRC="$HOME/models/Qwen3.8-Flash-Next"
export BASE="$HOME/models/Qwen3.8-Flash-Next-NVFP4-base"
export CANDIDATES="$HOME/models/calibration-sources/qwen38-maca-candidates.jsonl"
export CALIB="$HOME/models/calibration-sources/qwen38-maca-cola-acdm-moe.jsonl"

MACA_LENGTHS=256,512,1024,2048,4096 \
TOKEN_BUDGET=1048576 \
RESERVE_SEQUENCES=1536 \
SEED=20260831 \
scripts/generate_calibration_corpus.sh "$SRC" "$CANDIDATES"
```

The composer repeats complete MaCa length cycles, then fills an exactly representable remainder.
It does not pad short rows to `--seqlen`. The candidate manifest contains the derived `nsamples`,
the exact token total, and the length histogram.

## 2. Profile and select

Profiling must see the original BF16 activations. Disable serving-only W4A4 routes and do not point
`PROFILE_MODEL` at an already quantized artifact.

```bash
export PROFILE_MODEL="$SRC"
export PROFILE_BASE="$BASE"
unset GB10_W4A4_PREFILL GB10_W4A4_VERIFY

CUDA_VISIBLE_DEVICES=0 \
PROFILE_LAYERS=auto \
PROFILE_SKETCH_DIM=16 \
COLA_WEIGHT=1 \
ACDM_WEIGHT=1 \
EXPERT_WEIGHT=1 \
KMEANS_ITERS=6 \
SELECTION_SEED=20260831 \
scripts/select_calibration_corpus.sh \
  "$PROFILE_MODEL" "$CANDIDATES" "$CALIB"
```

For checkpoints that do not fit resident in BF16, `PROFILE_BASE` enables exact sequential
layer profiling: the NVFP4 base supplies only the memory-resident skeleton and PLE table, all
of its transformer layers are dropped, then one source BF16 layer at a time is loaded and run
over the candidate pool. Hidden states stay resident between layers. COLA sketches therefore see
the same BF16-layer trajectory used by sequential GPTQ, while peak memory remains bounded by the
base skeleton, one BF16 layer, and the candidate hidden states.

`PROFILE_LAYERS=auto` samples up to eight layers across the network. For an MoE model, exact expert
routing counts are still captured at every layer reached by the profiler. The selector preserves
`primary_category × sequence_length` quotas from the original consumed prefix and performs a
deterministic repair pass if greedy selection displaced a quota.

For true task-aligned ACDM, first profile a held-out set representative of production traffic with
the same model, layers, sketch dimension, and maximum sequence length. Then pass that profile file
as the fourth argument:

```bash
scripts/select_calibration_corpus.sh \
  "$PROFILE_MODEL" "$CANDIDATES" "$CALIB" \
  "$HOME/models/calibration-sources/production-reference.profiles.jsonl"
```

Do not use benchmark evaluation questions as the reference set; that would leak the benchmark into
calibration selection.

## 3. Run MR-GPTQ with sequence normalization

Read the actual selected count and maximum length from the generated manifests instead of assuming
512 fixed-size rows.

```bash
export FINAL="$HOME/models/Qwen3.8-Flash-Next-MR-GPTQ-NVFP4-maca"

NSAMPLES=$(jq -er '.selected_count' "$CALIB.manifest.json")
SEQLEN=$(jq -er '.seqlen' "$CANDIDATES.manifest.json")

unset GB10_W4A4_PREFILL GB10_W4A4_VERIFY
CUDA_VISIBLE_DEVICES=0 GB10_PLE_OFFLOAD=ssd \
./target/release/gb10_inference --gptq \
  --model-dir "$SRC" \
  --base "$BASE" \
  --out "$FINAL" \
  --calib "$CALIB" \
  --nsamples "$NSAMPLES" \
  --seqlen "$SEQLEN" \
  --maca \
  --damp 0.01 \
  --clip 7 \
  --rotate \
  --scale-iters 4 \
  --gptq-groups expert,attn,mlp,gdn,lmhead \
  --rtn-groups mtp,embed
```

With `--maca`, each sequence contributes `1 / sequence_length` to Hessian accumulation. The raw
token count is still retained for under-calibration checks. Down-projection replay stores the same
weight for each captured segment, including the all-token fallback. DFlash2 calibration currently
rejects `--maca`; use fixed-length rows for `--gptq-dflash2`.

## References

- MaCa: <https://arxiv.org/abs/2602.07465>
- COLA: <https://arxiv.org/abs/2510.10618>
- ACDM: <https://openreview.net/forum?id=pfw3saHzGU>
- MoEQuant: <https://arxiv.org/abs/2505.03804>
