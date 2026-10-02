# Changelog

High-level release notes for veloGB10. Minor bug fixes and small optimizations are grouped under
generic language where they aren't individually notable.

## v0.7.1 — TP=4 on the EXL3 path, image input, automatic shard shipping

A fast follow to v0.7.0: the EXL3 path gains a fourth node and images, the node sync becomes
shard-level, and two defaults change that alter long-context output bytes.

### TP=4 on the EXL3 path

`--tp 4` now serves Qwen3.8-Flash-Next from EXL3 packs across three peer nodes plus the head:
attention, GDN and the KV cache split by head, the routed experts dealt across the ranks, the
vocab-parallel LM head, sharded MTP-head screening, a per-step lockstep agreement check and a
watchdog. Measured on pure code: **~221 tok/s decode** (min/max 173/240, p50/p90/p99 221/234/237) and
**~3.2K tok/s prefill** — TTFT 0.64 s at ~2.1K input tokens, 10.9 s at ~34.9K. Against a single GB10
that is **×1.6 decode and ×2.0–2.2 prefill**.

### Image input, on every topology

Images are served end to end on the EXL3 path at TP=1, TP=2 and TP=4: the pack's **original bf16
vision tower** is loaded (`vision_tower_bf16.safetensors`), each image is resized so its longer side
is at most `--image-max-edge` (default 1024, aspect preserved), 3-axis mrope is applied at all six
main-attention RoPE sites **and** the QSA indexer — so images work past the 2,051-token dense window —
and image rows ship to the TP nodes as a digest-checked binary payload. Parts are returned in client
order. **Video and audio parts return `400`** for now; video is planned.

### Automatic shard shipping

A multi-node run no longer needs the model copied to the peer machines. The head plans each rank's
shard and ships only that, through the TP blob cache: **~57.5 GiB per node at TP=2** and **~46.8 GiB
at TP=4** for the EXL3 pack, and 66.0 GiB for NVFP4 (was 98.5 replicated). TP=2 output is
token-identical to the previous path for both formats, and the live TP=4 bank is identical.
`--tp-sync-only` runs a node that syncs and exits.

### Default changes (output-changing on long context)

- **`--prefill-chunk 4095`**, up from 2048 — a large part of the TP=4 prefill gain. Ragged-tail
  folding is always on; `--prefill-chunk 2048` restores the previous grid.
- **`--qsa-key-rope full`** — the EXL3 indexer had left rotary dims 32..63 of every pooled key at
  zero since the sparse-attention work, while the reference fills them. `half` keeps the old bytes
  for A/B comparison only.

Plus minor bug fixes and optimizations, including a clear error instead of a panic when `--model-dir`
does not exist.

## v0.7.0 — EXL3 packs (Qwen3.8-Flash-Next), TP=2 serving, CLI-only options

A large release: a second quantization format with its own model family, two-node serving as a
shipped mode, and one breaking change to how the engine is configured.

### New: EXL3 packs are served directly

`--model-dir` now accepts an **EXL3 (ExLlamaV3 trellis)** pack alongside the NVFP4/FP8 families.
The model shipped on this path is **Qwen3.8-Flash-Next**: a 125B MoE with ~6B active parameters,
a Gated DeltaNet + Qwen Sparse Attention hybrid, 512 experts (top-10), a 51B n-gram embedding
table, an MTP draft head, and a 262,144-token context.

- The weights are published as
  **[doth4580/Qwen3.8-Flash-Next-EXL3-3.05bpw](https://huggingface.co/doth4580/Qwen3.8-Flash-Next-EXL3-3.05bpw)**
  — EXL3 3.05 bpw, about 85 GB including the 32.6 GB n-gram table.
- Native MTP speculative decoding is built in: greedy output is bitwise identical to
  non-speculative decoding, and sampled output is distribution-exact.
- **TP=1 and TP=2 only.** Any other world size exits with `--tp 2 only`.
- **Text only — no vision on the EXL3 path** (image requests are rejected). Vision remains on the
  NVFP4 / Qwen3.8 27B path added in v0.6.0.
- q8 KV by default, prefix cache with prefill checkpoints, penalties, streaming, loop detection.

### TP=2

TP=2 ships as a served mode: sequence-parallel prefill, a vocab-parallel LM head, and prefill
communication overlap — each on by default with an `off` value. Measured on two GB10 (DGX Spark)
over ConnectX-7, one request at a time, on the same engine code as this release but not on the
release binary itself; decode includes MTP speculation:

| | TP=2 | vs TP=1 |
|---|---|---|
| Decode, greedy (C code / Python / prose) | ~192 / ~160 / ~110 tok/s | ×1.38–1.42 |
| Decode, thinking-on, sampled | ~95 tok/s | ×1.36 |
| Prefill (2K / 32K / 128K / 256K) | 2,485 / 2,376 / 2,279 / 2,138 tok/s | ×1.56–1.69 |

### CLI-1: every option is a command-line flag (breaking)

- **The engine reads no environment variables.** Leaving a `GB10_*` variable set now refuses
  startup and names the replacement flag — for example `GB10_TP_GRAPH` suggests `--tp-graph`.
- `--print-config` prints every option's resolved value for a given command line; `--help-diag`
  lists the diagnostics and test drills.
- Migration table for every removed variable: **[docs/ENV_TO_FLAGS.md](docs/ENV_TO_FLAGS.md)**.
- Under TP every option is set on the head; the head ships its resolved options to the node, and a
  node's own command line carries only the per-box options.

### Speed

Prefill and decode levers on both paths, all bitwise: a pipelined expert kernel, a tensor-core GDN
chunk scan, hyper-connection fusion, router coalescing, sparse-attention selection, MoE gate/up
folding, and shared-expert overlap. On the EXL3 path, prefill TTFT is down roughly 27–30% against
the previous release. Autotune tables are now stamped with a build id and validated at load, so a
stale table cannot be applied to a changed build. NVFP4 W4A4 prefill kernels were contributed via
community PR #4.

### Reliability

- **A node that fails during boot now fails the head loudly** instead of leaving it hanging
  (PACK-FIX). The EXL3 TP pack-manifest check was removed by owner decision — each box loads its
  own pack at the path the head names.
- Autotune's preflight no longer requires specific machine addresses; it checks only that the GPU
  is idle (no other engine, no resident compute applications).
- Streaming loop detection, and CPU-affinity pinning by default.

Plus minor bug fixes and optimizations.

## v0.6.1 — `response_format` served again (quality-regression hotfix)

- **`response_format` requests are served, not rejected.** v0.6.0's P13 W2 change turned any
  `response_format` request into a loud HTTP 400; that was a hard failure of requests that used to
  succeed and cost measured quality on the public benchmark (the hardmode leg fell 150/176 →
  85/100 because json_schema scenarios returned `[server error 400]` instead of a verdict). Such a
  request is now served again, and the truth is advertised instead of hidden: every reply for a
  schema request carries an **`x-json-schema-enforced: none`** header, and the reason is logged
  loudly once per distinct reason. Nothing is silently ignored, and nothing that used to work is
  refused. (Schema *enforcement* remains off — it is not yet proven, and the switch is documented as
  gated on evidence.)
- **JSON-schema FSM fix.** An INTEGER prefix that can never return into range is now refused at the
  digit rather than at termination (previously an unbounded digit run after `maximum` was allowed).
  Deliberately narrow: `Ty::Integer` only, so a float case like `70e-1 == 7` is never rejected.
- Phase-B decode-mask arming + schema-blind-lane routing groundwork (inert while no lane carries a
  schema); 4 new tests. Minor bug fixes and optimizations.

## v0.6.0 — DFlash v1 drafter lane, DSpark, tool-render parity, TP/FP8 correctness

A large release: a new drafter serving lane, a new draft-model family, reference-exact tool
rendering, and a broad sweep of TP/FP8/losslessness correctness fixes.

### Drafters & speculation
- **DFlash v1 drafter lane** (`--spec-source dflash`) — the v1 drafter is now config-driven and
  wired into the API serving path, including under **TP=2/TP=4** (SPMD lane on every rank + artifact
  shipping). The draft round's attention was DRAM-bound on ~144× redundant KV traffic; a tiled
  DFlash attention kernel fixes it.
- **DSpark drafter support** — new kernel module `gpu_dspark` and a native DSpark round, with a
  signed ring-identity guard for the prefix carry.
- **DF2 block-16** — runtime `--df2-block 8|16`. Measured at TP=4: **+23.03% pooled / +9.47%
  row-medians** on the mission prompt set (GO on that workload; NO-GO on the recipe's own payload).
  Opt-in — the default stays block 8.
- **DF2 carry across a prefix-cache hit** — keeps DFlash2 in the draft seat across a cache hit
  (flag-gated, default off; measured slower at TP=4).
- **One artifact flag for every drafter**: `--draft-dir`; `--spec-source` is the only selector.

### Tool calling & template rendering
- **Reference-exact tool rendering** — `serde_json` and `minijinja` now preserve key insertion order
  (`preserve_order`). Without it the `<tools>` block was re-sorted and diverged from the reference
  (transformers/vLLM) at byte 83 on every tool-using turn (−345/−1,346 tokens on 12/52 tools).
- JSON-schema handling made explicit (loud floor) and tool-call serialization reworked.

### Serving & scheduler
- **Two-lane prefill cursor** (`--prefill-sched <inline|cursor>`) — the prefill window loop moved out
  of `admit()` into the step loop.
- Scheduler fixes: a silent request drop in the TP head path, and slot exhaustion now fails loudly.
- k8v8 multi-lane fast-fail; thinking toggle; `/health` telemetry.

### TP correctness
- **MTP head kept replicated under TP** — F6 sharding broke the draft chain (a TP=2/TP=4 serving
  regression).
- **F9 fix**: warm prefix-reuse stream divergence (TP2 FP8 27B).
- Batch-invariant verify-attention partition under TP + a binv probe.
- Shard block-128 FP8 (`W::Fp8Blk`) on the in-place path.
- TP-mode GDN state probe (`StateTp`) + logits/extent probe fixes for the sharded path.

### FP8 & losslessness
- **pf8 prefill OOB class fixed** — the full-attention / batch-mixer prefill-width W8A8 lane output
  buffers are padded to tp64 (FIX #2, #2b, #3, #4).
- **`ignore_eos` / `min_new` honored on every spec-path emit loop** (12 sites) — the stop rule is a
  property of the request, not the serving path (fixes the d2/d4-vs-d0 early-stop ladder failures).
- Depth-2 attention key partition keyed on the request constant; residual depth-2 verify losslessness
  diagnostics.

### Safety & observability
- **Release-live tripwire** — a pool/OOB tripwire with `--pool-census` and `--tripwire-selftest`.
- **`DISPATCH_ASSERT`** — the wide-verify dispatch rule as a machine-checked test.
- New `tel` module + telemetry reporting (e.g. dflash2-tree as its own `/health` mode).

### Launchers
- TP=4 production launcher (world=4) + a DRYRUN-asserting test; an NVFP4 TP=2 production launcher
  with a boot-identity line on both lanes. Minor bug fixes and optimizations.

## v0.5.5 — FP8 prefill levers on, prompt-truncation + max_tokens fixes

- **FP8 prefill levers now on by default.** The tensor-core flash-attention prefill
  (`GB10_FA_PREFILL`) and the tensor-core chunked GDN scan (`GB10_GDN_CHUNK2`) are now default-on for
  the FP8 path (value-checked; `=0` restores the legacy path). Big prefill speedup on the FP8
  Qwen3.8 27B configuration. NVFP4/MXF4/other-model paths are untouched.
- **`truncate_prompt_tokens` on `/v1/chat/completions`.** vLLM's left-truncation field now works on
  the chat path (it previously only worked on `/v1/tokenize`, so an over-length chat couldn't be
  rescued). Keeps the LAST `n` prompt tokens instead of the over-length 400.
- **`max_tokens`-omitted fix.** The scheduler now clamps before rejecting instead of rejecting when
  `prompt_len + max_new + depth + 8 > kv_stride` — a request that omits `max_tokens` no longer
  terminates with 0 tokens. Minor bug fixes and optimizations.

## v0.5.4 — Built-in OpenTelemetry, FP8 support, DFlash2 tree mode

- **Built-in OpenTelemetry.** `--otel-endpoint <URL>` streams OTLP/HTTP-JSON generation telemetry
  (the actual SSE chunk bytes, with `model.id` / `topology` / `request.id` / `token.index` / `event`
  attributes), off by default, near-zero decode interference. `request.id` is now the conversation
  key so a reply's turns join one continuous session; `generation.id` stays per-POST. Companion
  flags: `--otel-batch-size`, `--otel-batch-interval-ms`, `--otel-include-tokens`,
  `--otel-model-id`, `--otel-topology`.
- **FP8 support expanded.** Direct load of Qwen fine-grained block-128 FP8 (`weight_scale_inv`);
  DFlash2 FP8 drafter bake (`--df2-bake-fp8`) plus weight-only NVFP4 (`--df2-bake-nvfp4`);
  `--df2-quant-fidelity` gate tool. Fixed FP8 kernel bugs across `gpu_batch.cu`.
- **DFlash2 tree verification.** New opt-in `--spec-source dflash2-tree` mode (additive; MTP and
  DFlash2 unchanged). Minor bug fixes and optimizations.

## v0.5.2 — vLLM-compatible tokenize / detokenize endpoints

- **`POST /v1/tokenize`** — vLLM-compatible tokenization: `{tokens, count, max_model_len}`, a pure
  tokenizer call (no forward / KV / GPU). Accepts a `prompt` string or a chat `messages` array; the
  `messages` mode renders exactly as the chat path so its count equals `usage.prompt_tokens`. Empty
  prompt → `{tokens: [], count: 0}`; over-length → `400 context_length_exceeded`.
- **`POST /v1/detokenize`** — vLLM-compatible decode half of the pair (`{model, prompt}`) for
  exact-N prompt building.
- Added so our engine can be benchmarked more correctly. Minor bug fixes and optimizations.

## v0.5.1 — Vision robustness, reasoning-effort, graceful-load fixes

- **Vision generalization + boot fix.** The GPU vision tower now bootstraps opportunistically: a
  non-vision or geometry-incompatible model serves text-only instead of crashing at startup (fixes a
  v0.5.0 boot crash on non-27B packs). Vision is generalized across the Qwen3.5/3.8 VL family, so
  all vision-tower models serve images.
- **OpenAI `reasoning_effort`.** Full level table (`none/low/medium/high/xhigh/max`) with
  per-family normalization, plus `--reasoning-effort`; the `high` mapping no longer silently drops
  thinking (regression fix).
- **Tool-call + reasoning-mode fix.** Tool-call markup is held back in reasoning mode too, fixing a
  first-call double-emit leak.
- **Graceful model-load exit.** Corrupted / stale / wrong-format checkpoints exit with a clear
  actionable message instead of a panic/OOM/core-dump.
- **`--output-prompts [n]`** — human-readable chat-request logging; `--vision-cpu` now listed in
  `--help`. Minor bug fixes and optimizations.

## v0.5.0 — Vision support

- **Vision support.** Image input is now supported end-to-end on a GPU vision tower
  (`gpu_vision` kernels), with a `--vision-cpu` escape hatch to the CPU reference path. PNG/JPEG/WebP/GIF
  decoding added. The engine now ships and requires the `gpu_vision.ptx` kernel artifact in addition
  to the existing PTX set.
- **Better tool-call support.** A single canonical serializer now handles streaming and
  non-streaming tool-call output identically, repairs malformed tool-call tags, and no longer drops
  or leaks text around tool-call boundaries. New tool-call compliance and serializer test suites.
- **Prefill/TTFT optimizations.** New opt-in prefill levers (tensor-core flash-attention prefill,
  v2 W4A4 prefill GEMM, GDN tensor-core chunked scan), all env-gated **default off**, so the default
  serving path is unchanged. Minor bug fixes and optimizations.
- **Model-id fix.** `/v1/models` and responses now report the model card's `base_model`
  (e.g. `Qwen/Qwen3.8-27B`) instead of a local directory fragment. `--model-name` still overrides.

## v0.4.2

- Fix: accept OpenAI multipart `content` (string | array | null) to unblock agent clients that send
  content parts; request-schema only.

## v0.4.1

- Fix: `--draft-dir` is now mandatory only when `--spec-source` explicitly names a DFlash2 mode;
  plain-MTP launches no longer require it.

## v0.4.0

- **Qwen3.8 27B NVFP4** support with native **DFlash 2** speculative decoding, full 256K context.
- **TP=4** serving (plus TP=2 and single-node).
- New DSV4 / DFlash2 / DSpark / MXFP4 kernel set.
- README Update section with the Qwen3.8 27B performance table and live throughput traces; new
  `QWEN_27B_SETUP.md` and `MANAGING_CACHE.md` docs.

## v0.3.1

- **KAT-Coder** model support; supported-models table in the README.

## v0.3.0

- README generalization and load-pipeline features. Minor bug fixes and optimizations.

## v0.2.0

- **Tencent Hy3 (hy_v3)** family support, 4-bit KV cache, FR-Spec draft head, model-name family fix.

## v0.1.0

- Initial public release: from-scratch Rust + CUDA engine for Qwen3.5/3.6 on single and TP=2 GB10,
  with NVFP4/FP8 quantization, MTP speculative decoding, an OpenAI-compatible server, and prebuilt
  release binaries.
