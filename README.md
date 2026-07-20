<p align="center">
  <img src="assets/velogb10_logo.png" alt="veloGB10" width="480">
</p>

# veloGB10

**A GB10-specific inference engine for one or two GB10-based systems — NVIDIA DGX Spark and
compatible OEM machines built around the NVIDIA GB10 chipset.**

veloGB10 (`gb10_inference`) is a from-scratch Rust + CUDA inference engine for the Qwen3.5/3.6
model family, including hybrid GatedDeltaNet + GQA architectures, dense models, and MoE models.

The implementation is intentionally specialized for GB10 systems:

- **One GB10 machine** — single-node inference
- **Two GB10 machines** — tensor-parallel inference (TP=2) **for performance**, not just
  capacity: two machines decode a single request measurably faster than one can
- NVIDIA DGX Spark and compatible GB10 OEM systems (Grace Blackwell, sm_121)
- 128 GB unified LPDDR5x memory, ~255 GB/s measured sustained bandwidth
- ConnectX-7 networking for two-node inference
- GB10-specific kernels, precision paths, memory management, and scheduling

This project does not aim to provide generic GPU portability or support arbitrary hardware. The
same binary supports all supported models through `--model-dir`; no Python runtime or framework
serving stack is required.

**Headline** (greedy, MTP-speculative, bitwise-lossless — full tables in
[Benchmarks](#benchmarks)): Qwen3.6 27B at **~40 tok/s** on one GB10 and **~50 tok/s on two** ·
Qwen3.6 35B MoE at **~112 tok/s** · Qwen3.5 122B MoE at **~39 tok/s** on one GB10 and **~57
tok/s on two**.

Prebuilt binaries for GB10 systems are on the **Releases** page — each release includes the
inference binary, the required PTX kernels, SHA-256 checksums, and build provenance notes. If you
run an NVIDIA DGX Spark or a compatible OEM GB10 machine, you can use a release binary without
compiling anything.

## Building from source

**System prerequisites** (on the GB10 itself):

- **NVIDIA DGX Spark (GB10, sm_121)** with the CUDA toolkit — `nvcc` available (`CUDA_HOME` is
  honored). The build compiles the two kernel modules to PTX and **fails loudly** if nvcc fails;
  on a machine without nvcc it falls back to the checked-in PTX in `src/ptx/` with a warning, so
  the Rust side can still be compiled anywhere.
- **Rust stable toolchain** (`rustup`).
- **libibverbs + rdma-core dev headers** (for the TP=2 transport shim):
  `sudo apt install libibverbs-dev rdma-core`

**Build:**

```bash
cargo build --release
```

This produces `target/release/gb10_inference` plus the two PTX kernel artifacts in `src/ptx/`.
**The binary is not self-contained — it loads `src/ptx/*.ptx` relative to its working directory**,
so run it from a directory that has both (a build-fingerprint handshake refuses to run mismatched
binary/PTX pairs, so the two never silently drift apart).

Don't want to build? Use the prebuilt package on the **Releases** page instead.

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

**Where the node's copy of the model lives:** on first sync the node fetches the model from the
head into a content-addressed cache at `~/.cache/gb10_tp/` on the node machine:

- `blobs/` — the model artifacts, each named by its SHA-256. Identical blobs are stored once and
  shared across models.
- `models/<model-name>/` — symlinks into `blobs/`; this directory is what the node presents to the
  loader as the model. The `[node] manifest '<model>': N artifacts, X cached, Y to fetch` log line
  counts exactly these blobs.
- `hashcache.json` — memoized file hashes so later syncs skip re-hashing the model.

Only missing blobs are transferred, so the second start of the same model syncs nothing. The cache
is safe to delete (it just re-fetches over the network) — but keep an eye on disk headroom: a 122B
recipe is ~76 GB.

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
> server path, pp2048 + tg128, 3 runs per cell) and veloGB10's own built-in benchmarks. A full benchmark
> run across all models × modes × contexts is in progress; these tables will be regenerated
> from it. Single-stream decode, greedy, NVFP4, unless noted.

### Qwen3.5 family (tok/s, greedy, MTP auto unless noted)

| Model (recipe) | Single node | TP=2 |
|---|---:|---:|
| 0.8B (mixed) | **182–217** | **182–201** ¹ |
| 2B (mixed) | **150–169** | **159–166** ¹ |
| 4B (mixed) | **97–112** | **112–115** ¹ |
| 9B (full) | **71–83** | **83–90** |
| 27B (full) | **31–32** | **40–42** |
| 122B MoE (mixed) | **40–43** | **46–51** |
| 122B MoE (gdn4) | **39–48** | **49.5–54** |

### Qwen3.6 family (tok/s, greedy, MTP auto unless noted)

| Model (recipe) | Single node | TP=2 |
|---|---:|---:|
| 27B (full) | **29–36** | **42–43** |
| 35B MoE (full) | **91.5–112** | **105.5–108.8** |
| 35B MoE (mixed) | — pending | — pending |

**Notes.** "Pending" cells land with the full benchmark run (tool-eval-bench `--perf`);
ranges are across 0–8K context. ¹ TP=2 on the small models (0.8B, 2B, 4B) is unoptimized —
barriers dominate at these sizes: TTFT is several times slower for little or no decode gain; run
them single-node. **TP=2 vs single**, same harness: 27B **1.2–1.6×** (ratio grows
with context — a matched-depth comparison is 1.42–1.51× at 6–10K); 122B **1.1–1.3×**; 9B is
wash at short context but **~1.26× at 8K** (TP decode *rises* with context there); 35B
~1.15× (at this size the barriers eat most of the win — TP's value on the 35B is memory, not
speed). MTP acceptance is workload-dependent (~35–85% across the family; prose accepts higher
than code).

Multi-client batching is weight-amortized and nearly free: 9B serves 4 concurrent clients at
34 tok/s *each* (~136 tok/s aggregate, 3.2× single-stream) with byte-identical output.

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

## Command-line reference

Complete surface of `gb10_inference` (same content as `--help`). Square brackets show defaults.

### Modes

| Mode | What it does |
|---|---|
| `--server` | OpenAI-compatible HTTP server — the normal way to run (endpoints: `POST /v1/chat/completions`, `GET /v1/models[/:id]`, `GET /health`) |
| *(no mode)* | Interactive CLI: load model, generate from `--prompt` |
| `--help`, `-h` | Print help |

### Server flags (`--server`)

| Flag | Default | Meaning |
|---|---|---|
| `--model-dir <DIR>` | required | Model directory (`config.json` + safetensors + tokenizer). The normal way to load |
| `--model-name <NAME>` | dir name | Name reported by `/v1/models` |
| `--model <FILE>` | — | Legacy: single `.safetensors` file (use `--model-dir`) |
| `--tokenizer <FILE>` | — | Legacy: tokenizer.json path (implied by `--model-dir`) |
| `--port <N>` | 8000 | Listen port |
| `--max-batch <N>` | 8 | Max concurrent sequences (lanes) |
| `--max-tokens <N>` | 8192 | Generation cap when a request omits `max_tokens` |
| `--max-seq-len <N>` | 4096 | **The context size.** KV cache is allocated to exactly this; prompts longer are rejected, over-long generations clamped. Clamped to the model's `max_position_embeddings` (256K this family). KV ≈ 64 KB/token/lane on 27B (hybrid GDN keeps this small); above ~12K, CUDA graphs are skipped (measured zero cost) |
| `--mtp <auto\|on\|off>` | auto | MTP speculative decoding. `auto` measures whether it pays and self-tunes depth from live acceptance; greedy verify is bitwise-lossless, temp>0 distribution-exact. `on`/`off` force it (benchmarking) |
| `--mtp-depth <N>` | auto | Pin draft depth instead of auto-picking (benchmarking) |
| `--ngram-draft <N>` | 0 | EXPERIMENTAL prompt-lookup drafting, n-gram order N (0 = off) |
| `--prefix-cache <on\|off>` | off | Reuse a conversation's cached prefix (~3× faster follow-up turns). Not bit-exact across reuse; greedy MTP stays lossless |
| `--default-repetition-penalty <F>` | 1.0 | Repetition penalty (1.0 = off) |
| `--default-presence-penalty <F>` | 1.5 (2.0 on 2B) | Presence penalty |
| `--default-frequency-penalty <F>` | 0.0 | Frequency penalty |

`temperature` / `top_p` / `top_k` / `seed` are **per-request** only (defaults 0.7 / 0.8 / 20) —
every request may override in its JSON body. There are no MTP env vars; speculation is auto-tuned
per request.

### TP=2 flags (head) and node mode

| Flag | Default | Meaning |
|---|---|---|
| `--node [--port 29500] [--rdma-dev d1[,d2]] [--once]` | — | Run the **node** (peer) side: resident supervisor, zero configuration — model, config, cost table and stop tokens ship from the head at sync |
| `--tp` | off | Enable TP=2 on `--server` (sync + RDMA bring-up first) |
| `--nodes <ip[:port],...>` | — | Explicit node address(es); skips UDP discovery |
| `--discover-wait <S>` | 3 | Discovery broadcast window (instead of `--nodes`) |
| `--rdma-dev <d1[,d2]>` | platform defaults | RoCE devices (also `GB10_RDMA_DEV`) |
| `--head --model-dir <DIR>` | — | One-shot bench/generate head (use `--server --tp` for serving) |

TP environment variables (read on the head, shipped to the node at sync; a node never needs them):

| Env var | Meaning |
|---|---|
| `GB10_TP_SHARD_MIXERS=1` | Shard attention/GDN mixers **and** MoE experts (~half weight bytes per rank — the win). Default: FFN-only |
| `GB10_TP_GRAPH=1` | CUDA-graph the TP decode (bench path) |
| `GB10_TP_FP32_PARTIALS=1` | FP32 all-reduce partials (~2× barrier payload; kills the bf16-partial acceptance dip on small models) |
| `GB10_TP_MTP=1`, `GB10_TP_MTP_DEPTH=N` | Bench rig: run `--bench-mtp` under TP |
| `GB10_TP_CACHE=<dir>` | Node's model blob cache (`~/.cache/gb10_tp`) |
| `GB10_TP_TAIL_DRILL=1`, `GB10_TP_AGREE_DRILL=N` | Fault-injection drills for the transport/agree guard |

Other single-node env vars: `GB10_RDMA_DEV` (device override), `RUST_INFER_ZERO_KV=1` (restore
cold-admit KV zeroing), `RUST_INFER_PREFILL_SCALAR=1` (scalar prefill path),
`GB10_NO_DECODE_GRAPHS=1` (disable decode graphs), `RUST_INFER_CPU_SAMPLE=1` (CPU sampling),
`GB10_TP_TRACE=1` (per-barrier timing histograms at exit).

### Probes (diagnostics)

`--bench-mtp-sample` (stochastic distribution gate), `--bench-tree` (tree verify),
`--bench-lanes` (batched verify), `--bench-prefill` (TTFT proxy), `--probe-binv` (batch
invariance), `--probe-state` (GDN state divergence), `--probe-reject` (rollback),
`--probe-gemm` (cuBLAS audit), `--probe-bandwidth` / `--probe-bandwidth-sustained` (roofline;
idle GB10 ≈ 255 GB/s), `--tp-barrier-bench` (transport gates, no model), `--net-test` (2-proc
transport audit), `--sweep-gemm`.

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

Beyond new models:

- **Advanced KV-cache handling** — rotated/codebook KV-cache quantization in the TurboQuant
  family (deterministic variant, so greedy speculative decoding stays bitwise-lossless), aimed
  at much longer effective contexts and faster long-context decode. Most relevant to the fat-KV
  architectures (full-GQA models like Hy3) and to multi-lane long-context serving; the GDN
  hybrids need it least — which is exactly what makes it portable upside.

## Sponsorship & support

veloGB10 is a **one-man project by [Stav Katsoulis](https://github.com/sf-stav)** — kernels,
scheduler, transport, gates, docs, and releases are all done in one person's limited time. Bug reports and well-formed issues are always free and
welcome. If you need something specific and soon — a model port, a feature, tuning for your
workload, TP>2 — **special work requests are taken on at a price**: open an issue describing the
work and it will be quoted. This is also the most direct way to make the "next areas of research"
above happen faster.

## Acknowledgements

- [`cudarc`](https://github.com/coreylowman/cudarc) — the Rust CUDA driver-API bindings the whole
  engine's GPU control plane is built on.
- Hugging Face `tokenizers` and `safetensors`; `minijinja` (chat templates); `axum` + `tokio`
  (serving).
- Alibaba's Qwen team — the Qwen3.5/3.6 model family this engine exists to serve, and the hybrid
  GatedDeltaNet architecture that shapes its best ideas.
- [`tool-eval-bench`](https://github.com/SeraphimSerapis/tool-eval-bench/) — the benchmark
  harness behind every number in this README.
- NVIDIA — the DGX Spark. One (or two) of them is all it takes.
