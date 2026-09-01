# Changelog

High-level release notes for veloGB10. Minor bug fixes and small optimizations are grouped under
generic language where they aren't individually notable.

## Unreleased — Qwen3.8-Flash-Next (qwen4_exp)

- **Qwen3.8-Flash-Next support** (`model_type: qwen4_exp`, 176B-A10B): hyper-connection residual
  streams, PLE n-gram injection, sigmoid-gated GatedDeltaNet, MoE 512×10 + shared expert, and its
  MTP head — served by the regular `GpuModel` engine (server, batching, prefix cache, verify).
- **Everything NVFP4**, including the 320M-row PLE n-gram table, quantized by `--quantize --recipe
  all` into 96-byte row records (`ple_ngram_nvfp4.bin`) that the engine keeps on the GPU or
  streams from the SSD (`--ple-offload ssd`, bit-identical).
- New quantizer groups `hc`, `ple`, `pletable`; 4 GB output shards (`GB10_QUANT_SHARD_GB`).
- Host-memory watchdog (`GB10_MEM_WATCHDOG_GB`) and an exact load-time memory guard for this
  family; `GB10_LOAD_FORCE` no longer bypasses it (`=unsafe` does).
- `--probe-q4` (prefill + greedy decode, optional logits dump) and `scripts/qwen4exp/` (HF
  reference oracle on a synthetic model, quantization round-trip check).
- **QSA sparse attention** (`Qwen4ExpTextQSAIndexer`): past 2051 visible tokens every attention
  layer (and the MTP head) attends to the 512 best-scoring 4-token blocks + tail, selected by a
  deterministic radix top-k in the verify kernels' rank space — MTP stays lossless. Raw indexer
  keys are cached per position like the KV; below the limit the dense kernels are unchanged.
  `GB10_Q4_DENSE_ATTN=1` (A/B) forces dense; `GB10_QSA_DUMP=1` dumps selections for the oracle
  check (`scripts/qwen4exp/compare_qsa.py`).
- **`--gptq`**: calibrated GPTQ / MR-GPTQ (`--rotate`, 16-point Hadamard micro-rotation)
  re-quantization to NVFP4, one layer at a time on one GB10 — the base artifact is loaded, each
  layer's bf16 weights are swapped in from the source shards, the engine's own prefill runs the
  calibration set with Hessian taps (per routed expert), GPTQ runs on the GPU (cuSOLVER Cholesky,
  row-parallel block sweep with NVFP4 group scales + clip search), the quantized layer is re-run
  for the next layer. Rotated artifacts are served with an activation micro-rotation before the
  matching GEMMs. The loader now decides q/k/v and GDN fusion per tensor group, so mixed
  bf16/NVFP4 artifacts load.
- **NVFP4 W4A4 prefill** (`GB10_W4A4_PREFILL`, `kernels/gpu_w4a4.cu`, `src/w4a4.rs`): the prefill
  GEMMs of the experts / shared expert / attention run on the block-scaled FP4 tensor cores with
  E2M1 activations (per-16 UE4M3 scales × the tensor's `input_global_scale`), reading the standard
  tiled weights — no repack, no second copy; decode / verify keep W4A16 (MTP contract intact).
  TTFT −26 % at 2.8K tokens, −28 % at 6.6K on Qwen3.8-Flash-Next. `GB10_W4A4_CHECK` fake-quant
  self-check. `e4m3_ceil` no longer saturates to 0x7F (NaN in E4M3) — also fixed in gpu_mxfp4.cu.
- **`input_global_scale`** written by `--gptq` for every calibrated tensor (activation amax from
  the Hessian taps) and `--calib-igs` to calibrate an existing artifact in place
  (`input_global_scale.json` sidecar merged by the loader).
- **GPTQ fixes**: the dense Hessians (attention, indexer, shared expert) were keyed by the
  quantizer's weight copy and never accumulated (those tensors silently got RTN); an empty
  Hessian is now an error, the QSA indexer reuses q_proj's; MR-GPTQ rotation is applied on the
  ≥128-token MoE prefill arm too (it was missing, breaking every rotated artifact past ~128
  tokens); RTN groups are no longer marked rotated; the MoE router (`--gptq-groups router`) is
  now marked rotated at load too (a router-GPTQ artifact routed on unrotated logits and answered
  EOS to everything); `--gptq-refmt` copies config.json instead of hard-linking (it edited the
  input artifact); output directories are guarded.
- **Vision** on this family: the Qwen3.5 tower with a 2560-wide merger (`TowerDims::out_hidden`
  read from `vision_config` and validated against the checkpoint); image embeddings spliced before
  the hyper-connection expansion.
  `VisualTower::load` now reads only the shards holding `model.visual.*` (every family).
- Limits: no TP for this family; QSA supports bf16 and k8v4 KV caches (q4/tq remain unsupported);
  image tokens use 1-D positions (no MRoPE — as on Qwen3.5).

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
