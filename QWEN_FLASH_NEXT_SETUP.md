# Qwen3.8-Flash-Next (qwen4_exp) NVFP4 — setup

Qwen3.8-Flash-Next is a 176B-parameter hybrid MoE (48 layers: 36 GatedDeltaNet + 12 gated GQA
attention, 512 experts top-10 + shared expert) with three things no other supported family has:

- **hyper-connections** — the residual stream is 4 copies of the hidden (`hc_count: 4`, 10240
  wide); every sublayer reads a gated mix of the 4 streams and writes back a per-stream weighted
  injection. There are no layer norms; the hc norms play that role;
- a **PLE n-gram table** — a 320M-row × 160 embedding table (51 GB in FP8, 102 GB bf16) hashed by
  the token's 2-gram / 3-gram context and injected once, at layer 1;
- a **QSA sparse-attention indexer** on the attention layers: past 2051 visible tokens each query
  attends only to the 512 best-scoring 4-token blocks (+ the tail) — see §3.

veloGB10 serves it as `Family::Qwen4Exp` inside the regular engine (same server, batching,
speculative verify, TP transport). **Everything is NVFP4**, the PLE table included, in a
row-record layout the engine can either keep on the GPU or stream from the SSD.

## 1. Quantize (once, ~10 min)

```bash
./target/release/gb10_inference --quantize \
    --model-dir ~/models/Qwen3.8-Flash-Next \
    --out ~/models/Qwen3.8-Flash-Next-NVFP4-velo \
    --recipe all
```

Output: 360 GB bf16 → ~97 GB:

| Part | Format | Size |
|---|---|---|
| routed experts (48 × 512), MTP experts | NVFP4 (compressed-tensors `weight_packed`) | ~68 GB |
| attention, GDN, shared experts, hyper-connections, PLE projections, router, embed, lm_head | NVFP4 | ~5 GB |
| PLE n-gram table | NVFP4 row records, `ple_ngram_nvfp4.bin` (96 B/row) + `ple_ngram_nvfp4.json` | 30.7 GB |
| norms, conv1d, `block_inject_weight` (M=4), vision tower | bf16 / f32 copy-through | ~1.4 GB |

Recipe groups specific to this family: `hc` (hyper-connection mixers), `ple` (PLE key/value
projections), `pletable` (the n-gram table). `all` includes them. `all,gdn:bf16` keeps the GDN
projections in bf16 if you want to trade ~4 GB for GDN precision (the earlier SGLang bring-up on
this box measured the GDN as the quantization-sensitive block — see `scripts/qwen4exp/README.md`).

Shards are written at 4 GB (`GB10_QUANT_SHARD_GB`): the loader holds a whole shard plus its
parts in host memory while it repacks, and 12 GB shards pushed a 128 GB box over the edge.

### 1b. Calibrated quantization: `--gptq` (GPTQ → NVFP4, optionally micro-rotated)

`--recipe all` is round-to-nearest. `--gptq` re-quantizes the GEMM weights with GPTQ (Hessian-
weighted error compensation from calibration activations), one layer at a time, on one GB10:

The same path supports Qwen3.5 dense checkpoints. For a dense hybrid model, use
`--gptq-groups attn,mlp,gdn`; `expert`, `hc`, and `ple` only apply to the MoE family.

```bash
GB10_PLE_OFFLOAD=ssd ./target/release/gb10_inference --gptq \
    --model-dir ~/models/Qwen3.8-Flash-Next \                 # bf16 source (read shard by shard)
    --base ~/models/Qwen3.8-Flash-Next-NVFP4-velo \           # an existing artifact: embeddings, norms, PLE table, MTP head
    --out ~/models/Qwen3.8-Flash-Next-GPTQ-velo \
    --calib /var/tmp/calib.jsonl --nsamples 128 --seqlen 1024 \
    [--gptq-groups expert,attn,mlp] [--rtn-groups mtp] [--fp8-groups ...] [--damp 0.01] [--clip 7] [--rotate]
    # groups: expert attn mlp gdn lmhead (GPTQ); any group for --rtn-groups / --fp8-groups.
    # `lmhead` GPTQ uses the final-mixer outputs of every calibration token as its Hessian;
    # embed / lm_head / final mixer are served during the calibration as the artifact will carry them.
```

How it fits in 128 GB: the `--base` artifact is loaded as the model; for each layer its linears are
swapped for the bf16 originals (~2.5 GB), the engine's own prefill runs the calibration samples
through that single layer (`prefill_batch_range`) with Hessian taps in `gemm_act` / `moe_batch`
(per routed expert for the MoE — 512 × 2560² f32 = 13 GB), GPTQ runs on the GPU (cuSOLVER
Cholesky, a row-parallel 128-column block sweep with NVFP4 16-group scales and a clip search,
cuBLAS propagation), the quantized layer is swapped back in and re-run so the next layer calibrates
on quantized inputs, and the records stream to 4 GB output shards. Peak ≈ 95 GB. Groups not in
`--gptq-groups`/`--rtn-groups` are written **bf16** (default: GDN, hyper-connections, router, PLE
projections, lm_head, embed — the quantization-sensitive ones); the MTP head and the PLE table
come from the base artifact. `--calib` is a jsonl (`{"text": …}`) or plain text; `random` gives
seeded random ids (synthetic-model smoke test). `--rotate` is MR-GPTQ: a 16-point Hadamard
micro-rotation (R = H16/4) is applied to every 16-block of the input dim before quantizing
(W' = W·R, H' = R·H·R); the artifact's config carries `quantization_config.transform =
{type: hadamard16, groups: […]}` and the engine rotates the matching activations at serve time
(`gptq_rotate_act_b` before the dense GEMMs, on the gathered expert inputs and on the silu
output before the down projection in the MoE paths; not supported together with `--mxfp4`).

Synthetic-model check (`scripts/qwen4exp/ref_bf16.py`, the HF reference on the bf16 weights vs
the engine's logits, same prompt): RTN 20.7 % relL2 / argmax 7/8 → GPTQ 12.0 % / 8/8 →
MR-GPTQ 11.8 % / 8/8.

Every GPTQ'd triple also carries `{stem}.input_global_scale` (F32 [1], compressed-tensors
convention: 6·448 / the activation |x| max seen by the calibration, after the MR rotation when
there is one) — the per-tensor activation scale of the W4A4 prefill below. An artifact quantized
before this existed can be calibrated in place without re-quantizing:
`--calib-igs --model-dir <artifact> --calib <jsonl> --nsamples 512 --seqlen 1024` runs a plain
prefill of the calibration set through the served model (~15 min) and writes
`<artifact>/input_global_scale.json` (stem → scale), which the loader merges over the tensors.

## 2. Serve

### Single GB10 — recommended: PLE table on the SSD

```bash
./target/release/gb10_inference --server \
    --model-dir ~/models/Qwen3.8-Flash-Next-NVFP4-velo \
    --ple-offload ssd \
    --port 9000 --max-seq-len 2048 --max-batch 2 --prefix-cache on
```

Device-resident footprint ≈ 70 GB + KV. The 16 PLE rows a token needs (16 × 96 B) are read from
`ple_ngram_nvfp4.bin` with `pread` on a thread fan-out (`GB10_PLE_SSD_THREADS`, default 32) at
the point of the forward where layer 1 consumes them; an application-level row cache
(`GB10_PLE_SSD_CACHE_ROWS`, default 262144) and the OS page cache keep a conversation's hot
n-grams in memory. Cost: one host round-trip per forward (CUDA decode graphs are disabled in this
mode). The logits are bit-identical to the resident mode.

### Single GB10 — PLE table on the GPU

Omit `--ple-offload`. The table is uploaded at load (~30 s); footprint ≈ 100 GB + KV: it fits, but
leaves **no room for anything else** on the box. The load-time guard refuses if the measured
budget (`[mem] qwen4_exp footprint: ...` line) does not fit in `MemAvailable`; do not force it.

### W4A4 prefill (`GB10_W4A4_PREFILL`)

`GB10_W4A4_PREFILL=1` (= groups `expert,mlp,attn`; or a comma list of quantizer groups) runs the
prefill GEMMs of those NVFP4 tensors on the Blackwell block-scaled FP4 tensor cores with the
**activations quantized too** (E2M1 + UE4M3 scale per 16, times the tensor's
`input_global_scale`; 1.0 when the artifact has none). The kernels (`kernels/gpu_w4a4.cu`) read
the engine's standard tiled weights directly — no second weight copy, so the 97-GB expert set
costs zero extra bytes — and only widths above the verify width (17+ tokens; the 128+-token
grouped MoE arm for the experts) take the path: decode and the MTP verify keep the W4A16 chain,
the lossless-verify contract is untouched. GDN, hyper-connections, PLE, router and lm_head stay
on bf16 activations by default (`gdn`/`hc`/`ple`/`lmhead` can be added to the list).

Measured on the RTN `all` artifact (batch 2, MTP, PLE on the SSD): TTFT 2 809 tokens 3.25 s →
2.41 s, 6 613 tokens 7.44 s → 5.33 s (−26/−28 %); decode unchanged. Accuracy: the A4 rounding is a
~4–14 % relL2 perturbation per GEMM (largest on the down projections), which the RTN artifact
does not always absorb (one 1 178-token prompt flipped into a reasoning ramble) while the
MR-GPTQ artifact — rotated weights, so the rotated activations are close to Gaussian per
16-block, exactly the MR-GPTQ W4A4 regime of Egiazarian et al. 2025 — answers correctly; the
tool-eval / GSM8K numbers are in the CHANGELOG. `GB10_W4A4_CHECK=1` recomputes every W4A4 GEMM
through the bf16 chain on fake-quantized inputs and reports the per-row relL2 (kernel check: dense
≤ 1e-2, MoE bit-identical); `GB10_W4A4_TRACE=1` logs each dispatch. Exclusive with `--mxfp4`.

### Memory safety

Two host hangs happened during the bring-up (unified memory exhausted → the kernel's OOM path
cannot reclaim device pages → no SSH). Root causes, both fixed: (a) the server used to load the
**vision tower** after the model, reading the 97 GB artifact's shards into host RAM to find
`model.visual.*` (+1.2 GB/s of RSS until OOM) — the tower is now skipped on this family; (b) a
GPU-resident PLE table forced past the memory guard left no headroom. Three defenses now exist,
keep all of them:

1. the load guard above (`GB10_LOAD_FORCE` is ignored on this family; only
   `GB10_LOAD_FORCE=unsafe` bypasses it);
2. a host-memory watchdog in every mode: if `MemAvailable` stays under `GB10_MEM_WATCHDOG_GB`
   (default 5 GB) for 400 ms the process prints a line and **exits** — losing the process frees
   the memory, hanging the box does not;
3. `scripts/memlog.sh /var/tmp/memlog.txt &` — a 1 Hz memory trace that survives a crash.

And a rule: while the model loads or serves, do not start other GPU jobs (probes, docker images
with torch, ollama models) on the same box.

## 3. What works, what does not (yet)

| | Status |
|---|---|
| prefill, greedy / sampled decode, continuous batching, prefix cache, OpenAI server, tool calls | works (validated against the HF reference on a synthetic model: logits cos ≥ 0.9999, argmax identical) |
| PLE on SSD (`--ple-offload ssd`) | works, bit-identical to resident |
| MTP speculative decoding (`--mtp auto`) | works, greedy output byte-identical to `--mtp off` (lossless); acceptance 64–78 %, ~3.1 tokens per verify forward, 27.8 → 47.5 tok/s end-to-end on a code answer (single GB10, PLE on SSD) |
| context > 2048 (QSA sparse attention) | works — `--max-seq-len` up to the model max; selection lists identical to the HF reference on the synthetic model (7/8 positions; the 8th is a score tie within bf16 noise), argmax 8/8. Real model, 7.3K-token needle prompt (`--max-seq-len 8192`, server): needle found, TTFT 8.2 s (~890 tok/s prefill; 7.9 s with dense attention forced), decode unchanged (~30 tok/s; 47 tok/s with `--mtp auto`, 71 % acceptance), output byte-identical with and without MTP, min 37 GB host memory free — see §3 |
| TP=2 / TP=4 | not brought up for this family yet |

Vision: `--vision-cpu` forces the CPU reference tower. Positions: the engine assigns sequential 1-D
positions to image tokens (the reference uses interleaved 3-D MRoPE: (t, h, w) per image patch and
`max(h, w)/2` positions per image for the following text). This is the existing behaviour on
Qwen3.5 as well; implementing MRoPE (3-D rope gather in the attention and QSA-indexer kernels,
per-lane position deltas after images) is the remaining fidelity item for image input.
| `--gptq` calibrated re-quantization (GPTQ / MR-GPTQ → NVFP4, one layer at a time) | works — see §1b; real-model A/B pending |
| vision input (OpenAI `image_url`, PNG/JPEG/WebP/GIF) | works — same tower as Qwen3.5 (identical HF code, 2560-wide merger), GPU tower, loaded from the one shard that holds `model.visual.*` (6 s, ~1.3 GB); image embeddings spliced before the hyper-connection expansion. Same limitation as the other families: image tokens use 1-D positions (no 3-D MRoPE) — see §3 |

### QSA sparse attention (long context)

`Qwen4ExpTextQSAIndexer`: on every full-attention layer (and the MTP head's), a small projection
gives each token 4 query heads and one raw key (128-d). A query's visible tokens are cut into
blocks of 4; a block's key is the k-layernorm of the mean of its raw keys, roped at the block start;
score = Σ_heads relu(q·k)/√128; the 512 best blocks plus the tail (visible mod 4) are the only keys
the attention sees. Below 2051 visible tokens every block is selected, so dense attention is exact
there and the engine keeps its dense kernels: **with `--max-seq-len <= 2051` nothing changes**
(graphs included). Above it the indexer is live:

- raw keys are cached per position like the KV (`[slot][pos][128]` bf16, ~1 KB/token over the 12
  layers); block keys are recomputed on the fly, so MTP rollbacks, prefix-cache reuse and tree
  compaction need no extra bookkeeping — whatever holds for the KV cache holds here;
- selection runs in the same rank space as the verify kernels (`path`-aware), so a verify column
  selects exactly what the decode at that position would: MTP stays lossless;
- the top-k is a deterministic radix select (ties → lowest block), the selected keys are visited in
  ascending order by a copy of the dense split-K kernel with one address indirection;
- prefill: a window whose queries all see ≤ 2051 tokens stays on the dense tensor-core path (raw
  keys are still cached); past that, scores are one cuBLAS GEMM batched over the indexer heads
  (bf16 in, f32 out, 1/√hd in alpha) + a relu-sum/causal combine (~4 ms per layer on a 7.3K
  prompt), and the attention over each query's list runs `gqa_attn_sel_prefill2` (one warp per
  query × kv head × 4-head group, 4 gathered keys in flight: 86 ms per layer at 7.3K — the whole
  sparse prefill costs 8.2 s vs 7.9 s with dense attention forced). A/B knobs: `GB10_QSA_SCALAR=1`
  (scalar score kernel), `GB10_QSA_SEL_V1=1` (one-key-at-a-time attention, 250 ms/layer),
  `GB10_QSA_TIME=1` (per-phase timing, syncs). Decode adds one small kernel chain per attention
  layer (no measurable tok/s change).

Flags / env: `--max-seq-len N` (KV + raw keys sized to N; the RoPE tables cap it at the model's
262144); `GB10_Q4_DENSE_ATTN=1` forces dense attention at any length (A/B only — NOT the reference
model past 2051); `GB10_QSA_DUMP=1` prints every selection list and score row (validation, syncs).
The sparse kernels support `--kv-cache bf16` and `--kv-cache k8v4`; `q4` and `tq` are refused when
the indexer is live. The k8v4 path uses its packed selected-attention reader while preserving the
bf16 raw-key cache used by the QSA indexer.

## 4. Probes

```bash
# prefill + greedy decode without the server; --chat renders the chat template
./target/release/gb10_inference --probe-q4 --model-dir <dir> --ple-offload ssd --chat \
    --prompt "Bonjour" --max-new-tokens 64 --max-seq-len 2048
# reference-oracle comparison (tiny synthetic model): scripts/qwen4exp/README.md
# long context: the same probe with a long --prompt and --max-seq-len 8192 (QSA live past 2051 tokens)
```
