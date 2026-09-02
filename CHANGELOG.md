# Changelog

High-level release notes for veloGB10. Minor bug fixes and small optimizations are grouped under
generic language where they aren't individually notable.

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
