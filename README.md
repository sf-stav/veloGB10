<p align="center">
  <img src="assets/velogb10_logo.png" alt="veloGB10" width="480">
</p>

# veloGB10

**An inference engine optimized to the absolute maximum for one NVIDIA DGX Spark (GB10) — and for
a pair of them (TP=2). Nothing else is targeted; nothing is generic.**

veloGB10 (`gb10_inference`) is a from-scratch Rust + CUDA inference engine for the Qwen3.5/3.6
model family — hybrid GatedDeltaNet + GQA architectures, dense and MoE. Every kernel, precision
choice, and scheduling decision is specialized for the GB10 (Grace Blackwell, sm_121, 128 GB
unified LPDDR5x @ 255 GB/s measured) and for two GB10s linked over ConnectX-7. One binary serves
every supported model via `--model-dir`. No Python runtime, no framework serving stack.

```
cargo build --release
./gb10_inference --server --model-dir /path/to/model        # single node
./gb10_inference --server --model-dir /path/to/model --tp --nodes <peer-ip>   # head, TP=2
./gb10_inference --node                                     # second node: that's it
```

Prebuilt binaries (binary + the required PTX kernels + SHA256 checksums + provenance notes) are on
this repository's **Releases** page — no build needed if you're on a GB10.

## Running

**Single node, single user (maximum speed):**

```bash
./gb10_inference --server --model-dir=/path/to/model --port=9000 \
  --max-seq-len=32768 --max-batch=1 --prefix-cache=on --mtp=auto
```

**Single node, ~4 concurrent users (maximum aggregate throughput):**

```bash
./gb10_inference --server --model-dir=/path/to/model --port=9000 \
  --max-seq-len=32768 --max-batch=4 --mtp-lanes=on --prefix-cache=on
```

**Two nodes, TP=2** (start the peer first — it needs no model copy and no configuration; the head
ships weights, settings, and calibration at sync):

```bash
./gb10_inference --node --port 29500                                    # on the second GB10
./gb10_inference --server --model-dir=/path/to/model --tp \
  --nodes <peer-ip>:29500 --port=9000 --max-seq-len=32768 --prefix-cache=on   # on the head
```

---

## Purpose

The DGX Spark has one scarce resource: **~255 GB/s of measured, sustainable memory bandwidth**.
Every design decision in this engine is subordinate to spending it well. The result is an
engine that runs large models — up to **122B on a single node, larger across two** — at speeds
that hold up under an agentic workload, not just on a benchmark prompt.

Two properties are treated as non-negotiable and are enforced by gates, not by hope:

- **Correctness is bitwise.** The serving GEMM is batch-invariant: a speculative verify of width
  N produces results bit-identical to N separate decodes. Greedy speculative decoding is therefore
  *exactly* lossless — same tokens, same bytes — and stochastic decoding is distribution-exact.
- **Numbers are measured.** Decode rooflines, TP speedups, and acceptance rates in this README come
  from the engine's own gates on this hardware. Where a number is an estimate, it says so.

## What it does today

- **OpenAI-compatible server** — streaming, tool calling (with schema-aware argument coercion),
  seedable sampling, continuous batching, prefix caching.
- **MTP speculative decoding** — native multi-token prediction heads with an auto-depth policy
  that measures its own cost/acceptance trade-off live and re-picks depth (or disables itself)
  per workload. No configuration required.
- **Two-node TP=2 serving** — see below.
- **NVFP4 / FP8 mixed-precision quantization** — offline quantizer producing HF-compatible
  compressed-tensors artifacts; NVFP4 tensor-core GEMMs for the serving path.
- **Long context** — chunked prefill; 32K-class envelopes validated end-to-end on TP=2;
  model-context up to 256K on the 27B. The hybrid GDN layers carry a fixed-size recurrent state,
  so KV memory grows only on the periodic full-attention layers.
- **Model coverage** — Qwen3.5 0.8B/2B/4B/9B, Qwen3.6 27B (dense hybrid), Qwen3.6 35B (MoE),
  Qwen3.5 122B (MoE hybrid). One binary; the model is a directory, not a build.

## Unique aspects

### Engineered to the roofline

The engine is ~94% GEMM and weight-bandwidth-bound, and it is tuned as such: on 9B the LM head
sustains **229 GB/s — 90% of the machine's measured 255 GB/s pure-read ceiling** — and the whole
decode step runs at 72% of it. Optimization here means *fewer bytes* (NVFP4, fused projections,
frequency-ranked draft vocabularies), not fewer launches.

### Bitwise-lossless speculation

The quantized serving GEMM always runs one fixed shape (N padded to 16), so decode and verify
execute an identical instruction sequence and column 0 is bit-identical *by construction*, not by
argument. That is what makes greedy MTP lossless rather than approximately lossless — and it is
gated as such, at contexts up to 27K, under statistical process control rather than pass/fail
coin flips.

### TP=2 as a **performance** mode

Two-node tensor parallelism is usually about *capacity* — splitting a model that doesn't fit.
Here it is primarily a **speed** mode for a single user: split a model that *does* fit across two
DGX Sparks and go measurably faster, because each node streams half the weights per token.
Measured: **1.42–1.51× on 27B at 6–10K context** (the regime agentic workloads actually live in),
**1.34× on 122B**, with the speedup *growing* with context.

And it is built to be trusted, not just to run:

- **Zero-configuration node** — start `--node` on the second box and `--server --tp --nodes <ip>`
  on the head. The head communicates the model, all settings, and its calibration table at sync;
  the node reproduces the head byte-for-byte. There is nothing to keep in sync by hand.
- **Per-step agreement guard** — both ranks hash their state every decode step and abort loudly on
  any divergence. Silent desync is not a failure mode this system has.
- **Deterministic everything** — auto-depth decisions are a pure function of bit-identical token
  history; output is byte-identical to the single-node build (gated, incl. live depth switches).

### Hybrid-native long context

The GatedDeltaNet layers make prefix caching, MTP rollback, and KV management *different* here —
the recurrent state exists at exactly one point in the sequence. The engine handles this natively
(periodic GDN checkpoints, fed-not-emitted cache invariants), which is what makes both prefix
caching (99% prefill skip on cache hits) and lossless speculation work on this architecture.

## Benchmarks

> **Preliminary.** All throughput and TTFT numbers below were measured with
> **[`tool-eval-bench --perf`](https://github.com/SeraphimSerapis/tool-eval-bench/)** (OpenAI
> server path, pp2048 + tg128, 3 runs per cell) and veloGB10's own gate benches. A full uniform
> sweep across all models × modes × contexts is in progress; these tables will be regenerated
> from it. Single-stream decode, greedy, NVFP4, unless noted.

### Single-node decode (tok/s, greedy)

| Model | Plain | MTP (auto depth) | Source |
|---|---:|---:|---|
| Qwen3.6 27B | ~12 | **32.1 / 39.9 / 36.9** @ d0 / 4K / 8K | server sweep, pp2048+tg128 ×3 |
| Qwen3.6 35B (MoE) | — | **91.5** (d2) · ~112 (server auto) | gate bench |
| Qwen3.5 122B (MoE) | 26.2 | **37.6** (d2) · 37.3 (d4) | gate bench |
| Qwen3.5 9B | ~41 | **70.9** (auto) | gate bench |
| Qwen3.5 0.8B–4B | — | sweep pending | — |

MTP acceptance is workload-dependent (~35–85% across the family; prose accepts higher than code).

Multi-client batching is weight-amortized and nearly free: 9B serves 4 concurrent clients at
34 tok/s *each* (~136 tok/s aggregate, 3.2× single-stream) with byte-identical output.

### TP=2 decode (server path, pp2048 + tg128, greedy)

| Model | tg tok/s @ d0 | @ 4K ctx | @ 8K ctx | TTFT (s) @ d0 / 4K / 8K |
|---|---:|---:|---:|---|
| Qwen3.6 27B (nvfp4-full) | **50.2** | **46.6** | **44.1** | 6.8 / 19.6 / 33.0 |
| Qwen3.6 27B (nvfp4-mixed) | **42.7** | **47.5** | **43.4** | 6.8 / 19.6 / 33.0 |
| Qwen3.5 122B | **56.6** | **54.7** | **49.5** | 5.0 / 14.4 / 24.6 |
| Qwen3.5 9B | **90.4** | **102.3** | **97.0** | 2.9 / 8.6 / 14.4 |
| Qwen3.5 0.8B | **206.5** | **217.1** | **211.3** | 1.5 / 4.3 / 7.2 |

Against single-node 27B on the same harness (32.1 / 39.9 / 36.9): **TP=2 is 1.2–1.6×**. On the
depth-matched bench harness the like-for-like ratio is 1.42–1.51× at 6–10K context (35B: 1.15×,
122B: 1.30–1.34×) — every leg greedy-lossless verified.

*"Greedy-lossless verified" (`LOSSLESS_OK` in the engine's gates): speculative output bit-identical
to non-speculative decoding — speculation changes speed, never the tokens.*

TP=2 also halves per-node memory (122B: 39 GB/rank vs 73 GB replicated), which is what makes
large-model + long-context + multi-lane combinations fit.

### Prefill

| Model | tok/s | Note |
|---|---:|---|
| Qwen3.5 122B | **702** | grouped-MoE GEMM with N=16 weight reuse |
| Qwen3.6 27B | ~730 | 2.8 s TTFT on a 2048-token prompt |

### Quality ([tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench/), agentic scenario suite)

| Model | Single node | TP=2 |
|---|---:|---:|
| Qwen3.6 27B | 93/100 | 92/100 |
| Qwen3.5 122B | 88/100 | 88/100 |

## Requirements

- 1–2× NVIDIA DGX Spark (GB10); TP=2 uses the ConnectX-7 interconnect between them
- NVFP4/FP8-quantized model artifacts (offline quantizer included)
- Rust toolchain + CUDA (sm_121a) to build; runtime is the binary plus its PTX kernel artifacts
- To reproduce the benchmarks: [`tool-eval-bench`](https://github.com/SeraphimSerapis/tool-eval-bench/)
  with the `--perf` flag, pointed at a running veloGB10 server

> **Cluster scope:** veloGB10 is designed, measured, and gated on exactly one and two GB10
> machines — that is the hardware we have. **TP>2 work has not been done because we have no access
> to more than two GB10 machines**: the weight sharding, the transport, and the lockstep serving
> protocol are built for two ranks, so TP>2 is engineering work, not a configuration flag, and we
> have had no hardware to develop or validate it on. If you have a bigger rig and want to help
> make TP>2 (or expert/pipeline parallelism) real, open an issue — we'd like to hear from you.

## Status

Actively developed. The correctness gates are the contract: greedy losslessness (SPRT-tested),
batch invariance, distribution-exact stochastic sampling, and TP=2 byte-identity all have to be
green for a build to be called stable. Larger models (MoE up to 400B-class across two nodes) are
on the roadmap; the two-node runtime is already the proving ground for them.

## Next areas of research

New architectures, in order of appearance on the roadmap — all targeted at the 2× GB10 cluster
via the TP=2 runtime:

- **Tencent Hy3 (295B-A21B MoE)** — pure-GQA MoE with a native MTP layer; the most direct port
  from the current engine family. **Next release.**
- **DeepSeek-V4-Flash-DSpark (284B-A13B MoE)** — compressed sparse/heavily-compressed attention
  with a 1M-token context design point and a native speculative decoding module; the strongest
  long-context economics of any model evaluated so far. **Next release.**
- **Step 3.7 Flash** — under evaluation.
- **Qwen3.5 397B MoE (NVFP4) we may work on this if not superceeded ** — the same `qwen3_5_moe`
  architecture the engine already serves at 122B, scaled up: no port required, the work is the 
  TP=2 capacity bring-up (215 GB of weights, ~108 GB/node) plus the gates at that size. The 
  closest big-model item on the list.
- **New Qwen and DeepSeek releases** — tracked as they land; the engine's kernel family (NVFP4
  tensor-core GEMM, grouped-MoE GEMM, batch-invariant verify, TP=2) is built to absorb new
  family members quickly.
