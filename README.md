<p align="center">
  <img src="assets/velogb10_logo.png" alt="veloGB10" width="480">
</p>

# veloGB10

**A GB10-specific inference engine for one or two GB10-based systems — NVIDIA DGX Spark and
compatible OEM machines built around the NVIDIA GB10 chipset.**

veloGB10 (`gb10_inference`) is a from-scratch Rust + CUDA inference engine for a hand-selected
set of large language models — currently including the Qwen3.5/3.6/3.8 family and Tencent Hy3 — with
support for hybrid GatedDeltaNet + GQA architectures, dense models, and MoE models. More model
families are added deliberately rather than generically; each one is ported, measured, and gated
on real GB10 hardware before it ships.

The implementation is intentionally specialized for GB10 systems:

- **One GB10 machine** — single-node inference
- **Two or four GB10 machines** — tensor-parallel inference (TP=2 / TP=4) **for performance**, not
  just capacity: multiple machines decode a single request measurably faster than one can
- NVIDIA DGX Spark and compatible GB10 OEM systems (Grace Blackwell, sm_121)
- 128 GB unified LPDDR5x memory, ~255 GB/s measured sustained bandwidth
- ConnectX-7 networking for two-node inference
- GB10-specific kernels, precision paths, memory management, and scheduling

This project does not aim to provide generic GPU portability or support arbitrary hardware. The
same binary supports all supported models through `--model-dir`; no Python runtime or framework
serving stack is required.

**Headline** (greedy, MTP-speculative, bitwise-lossless — full tables in
[Benchmarks](#benchmarks)): Qwen3.8 27B at **~40 tok/s on one GB10 and ~56 tok/s on two** (and
**~85 tok/s on four** with DFlash 2) · Qwen3.6 27B at **~42 tok/s** on one GB10 and **~53 tok/s on two** ·
Qwen3.6 35B MoE at **~111 tok/s** on one GB10 and **~130 tok/s on two** · Qwen3.5 122B MoE at **~39 tok/s** on one GB10 and **~57
tok/s on two**.

Prebuilt binaries for GB10 systems are on the [**Releases** page](https://github.com/sf-stav/veloGB10/releases) — each release includes the
inference binary, the required PTX kernels, SHA-256 checksums, and build provenance notes. If you
run an NVIDIA DGX Spark or a compatible OEM GB10 machine, you can use a release binary without
compiling anything.

## Update — Qwen 3.8 27B NVFP4 with DFlash 2

**veloGB10 now fully supports the Qwen3.8 27B NVFP4 model, with native DFlash 2 speculative
decoding, at the model's full 256K context.**

The supported configuration combines our NVFP4-quantized
[Qwen3.8-27B-NVFP4-FULL](https://huggingface.co/doth4580/Qwen3.8-27B-NVFP4-FULL) target model with
the mirrored [Qwen3.8-27B-DFlash2](https://huggingface.co/doth4580/Qwen3.8-27B-DFlash2) drafter —
launched via `--spec-source dflash2-auto --draft-dir <dflash2 dir>` — and runs at **max-seq-len
262144** (full 256K context).

### Qwen 3.8 27B NVFP4 performance

Single-stream decode, greedy, NVFP4 with DFlash 2. Figures are representative; real numbers vary
with content type. **Highest performance is on code generation.** "Average" is a representative
blend across content types (see the note below — a code-heavy run averages much higher).

| Mode | Average | Bottoms | Peaks | Max sustained (code) |
|---|---:|---:|---:|---:|
| Single node | **> 40 tok/s** | ~11 tok/s | ~100 tok/s | **~70 tok/s** |
| TP=2 | **~56 tok/s** | ~18 tok/s | ~105 tok/s | **~85 tok/s** |
| TP=4 | **~85 tok/s** | ~32 tok/s | ~150 tok/s | **~125 tok/s** |

> **Averages span a wide range by content type.** On a mixed-content run (the `min_max` traces
> below) the session average is ~25 / ~38 / ~51 tok/s for single / TP=2 / TP=4, whereas on a
> code-heavy sustained run (the `max` traces) it is ~70 / ~85 / ~125 tok/s. The "Average" column
> above is a representative figure across that spread; **Peaks** are the top reached on
> code-generation content; **Max sustained (code)** is the steady-state rate held during a
> code-heavy run (read from the `max` traces).

> Peak rates are typically reached on **code content generation**; prose and mixed content sit lower
> in the ranges above.

#### Live throughput traces (veloGB10)

These are `TOKENS/SECOND` traces pulled straight from the engine's live stats panel while serving
the Qwen3.8 27B NVFP4 + DFlash 2 config. For each deployment mode there's a **peak-vs-lowest**
trace (the full spread of a session — where throughput bottoms out and where it tops out) and a
**sustained peak** trace (a segment holding its best rate). Download the images to see them full-size.

**Single node** — mixed-content average ~26 tok/s, code-heavy sustained ~70 tok/s, some content
types dip as low as ~11, and peaks reach ~100 on code. The peak-vs-lowest trace shows the swing
between content types; the sustained-peak trace is a stretch running near the ~70–100 tok/s band.

| Peak vs lowest | Sustained peak |
|---|---|
| ![Single node — peak vs lowest](assets/single_min_max.png) | ![Single node — sustained peak](assets/single_max.png) |

**TP=2** — the two-node setup lifts the ceiling: mixed average ~38, sustained ~85, bottoms ~18,
peaks ~105. Even the lowest points sit above the single-node average.

| Peak vs lowest | Sustained peak |
|---|---|
| ![TP=2 — peak vs lowest](assets/tp2_min_max.png) | ![TP=2 — sustained peak](assets/tp2_max.png) |

**TP=4** — four-node serving pushes well past 100 tok/s on the best content: mixed average ~51,
sustained ~125, bottoms ~32, peaks up to ~150. The sustained-peak trace holds a ~125 tok/s plateau
with spikes near 150.

| Peak vs lowest | Sustained peak |
|---|---|
| ![TP=4 — peak vs lowest](assets/tp4_min_max.png) | ![TP=4 — sustained peak](assets/tp4_max.png) |

Scaling up from one to four nodes roughly **doubles the ceiling** (single ~100 → TP=4 ~150) and
more than doubles the sustained average on code-heavy content (single ~70 → TP=4 ~125).

#### Comparison with other Qwen3.8 27B recipes

Same `TOKENS/SECOND` scale and format, but these are **community SGLang recipes** running the
Qwen3.8 27B on a single DGX Spark (not veloGB10). They are shown for side-by-side comparison only —
different kernels, quantization, and serving stacks, so treat the differences as informational, not
apples-to-apples. Each pair is the same **peak-vs-lowest** / **sustained peak** split as above.

**Mia AI Lab — Qwen3.8-27B-SGLang-DGX-Spark** ([repo](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark)):
single-node and two-node (TP=2) SGLang serving.

Single node:

| Peak vs lowest | Sustained peak |
|---|---|
| ![sglang_mia single — peak vs lowest](assets/sglang_mia_single_min_max.png) | ![sglang_mia single — sustained peak](assets/sglang_mia_single_max.png) |

TP=2:

| Peak vs lowest | Sustained peak |
|---|---|
| ![sglang_mia TP=2 — peak vs lowest](assets/sglang_mia_tp2_min_max.png) | ![sglang_mia TP=2 — sustained peak](assets/sglang_mia_tp2_max.png) |

**Hasso — dgx-spark-qwen38** ([repo](https://github.com/hasso5703/dgx-spark-qwen38)):
single-node SGLang serving.

| Peak vs lowest | Sustained peak |
|---|---|
| ![sglang_hasso single — peak vs lowest](assets/sglang_hasso_single_min_max.png) | ![sglang_hasso single — sustained peak](assets/sglang_hasso_single_max.png) |

### Recipe comparison (single-node scale)

Side-by-side figures for all the recipes shown above. veloGB10's own rows are included for
reference. **Peaks** are the top reached; **Max sustained (code)** is the steady-state rate held on
code-heavy content (read from the `max` traces). Averages are a representative blend across content
types;

| Recipe | Config | Average | Bottoms | Peaks | Max sustained (code) |
|---|---|---:|---:|---:|---:|
| **veloGB10** | Single node | **> 40 tok/s** | ~11 | ~100 | **~70** |
| **veloGB10** | TP=2 | **~56 tok/s** | ~18 | ~105 | **~85** |
| **veloGB10** | TP=4 | **~85 tok/s** | ~32 | ~150 | **~125** |
| Mia AI Lab | Single node | ~27–42 tok/s | ~15 | ~58 | ~42 |
| Mia AI Lab | TP=2 | ~21–37 tok/s | ~10–12 | ~45 | ~37 |
| Hasso | Single node | ~27–35 tok/s | ~19 | ~66 | ~35 |

> veloGB10's peaks are higher than either SGLang recipe even at **single node**, and the gap widens
> with TP=2/TP=4. These are different kernels, quantization, and serving stacks — informational
> comparison only, not a controlled benchmark.

### Getting started with Qwen 3.8 27B

Full, step-by-step setup instructions for single-node, TP=2, and TP=4 deployments (node layout,
required files, launch commands, and expected output) are in
**[QWEN_27B_SETUP.md](QWEN_27B_SETUP.md)**. The Tencent Hy3 model's two-node TP=2 bring-up is in
**[HY3_SETUP.md](HY3_SETUP.md)**. Managing the engine's TP model cache is documented in
**[MANAGING_CACHE.md](MANAGING_CACHE.md)**. If you want to see how the stack holds up under a long
run, there's an **8-hour endurance report** — throughput, latency, determinism, and thermals over a
mixed workload — in **[ENDURANCE_REPORT.md](ENDURANCE_REPORT.md)**. Release-by-release highlights are
in **[CHANGELOG.md](CHANGELOG.md)**.

### Notes

- **TP=4 does not eat the whole cluster.** Running the Qwen3.8 27B NVFP4 model across 4× DGX Spark
  does **not** mean you can't run anything else on those boxes. The model occupies roughly **45 GB on
  the head** and about **20 GB on each node**, so each GB10 still has plenty of headroom to run other
  processes. On a fully idle DGX Spark (~113 GB available) the steady-state estimate for the model is
  far under the machine's total memory.
- **Vision is supported.** Image input runs on the GPU vision tower, with a `--vision-cpu` escape
  to the CPU reference path. PNG/JPEG/WebP/GIF images are supported.

---

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

Don't want to build? Use the prebuilt package on the [**Releases** page](https://github.com/sf-stav/veloGB10/releases) instead.

## Running

> **Qwen3.8 27B with DFlash 2 requires the draft-model arguments.** Add
> `--spec-source dflash2-auto --draft-dir <path/to/Qwen3.8-27B-DFlash2>` to the launch command, and
> point `--model-dir` at the DFlash2-capable quantized model. The Qwen3.8 27B NVFP4 model is not a
> standalone drafter runner — it needs the DFlash2 drafter for speculative decoding. See
> [QWEN_27B_SETUP.md](QWEN_27B_SETUP.md) for the full Qwen3.8 27B command lines.

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

- **OpenAI-compatible server** — streaming, tool calling (schema-aware argument coercion, with a
  single canonical serializer across streaming and non-streaming), seedable sampling, continuous
  batching, prefix caching, and OpenAI `reasoning_effort` levels (`none/low/medium/high/xhigh/max`).
- **Vision** — image input on a GPU vision tower across the Qwen3.5/3.8 VL family
  (`--vision-cpu` for the CPU reference path); PNG/JPEG/WebP/GIF. The tower bootstraps
  opportunistically: a non-vision or incompatible model serves text-only, never a startup crash.
- **MTP speculative decoding** — native multi-token prediction heads with an auto-depth policy
  that measures its own cost/acceptance trade-off live and re-picks depth (or disables itself)
  per workload. No configuration required.
- **Two-node / four-node TP serving** — see below.
- **NVFP4 / FP8 mixed-precision quantization** — offline quantizer producing HF-compatible
  compressed-tensors artifacts; NVFP4 tensor-core GEMMs for the serving path.
- **Long context** — chunked prefill; 32K-class envelopes validated end-to-end on TP=2;
  model-context up to 256K on the 27B. The hybrid GDN layers carry a fixed-size recurrent state,
  so KV memory grows only on the periodic full-attention layers.
- **Supported models** — one binary loads any of these; the model is a directory, not a build.

  | Model | HF artifact | Architecture / recipe |
  |---|---|---|
  | Qwen3.5 0.8B | [doth4580/Qwen3.5-0.8B-NVFP4-MIXED](https://huggingface.co/doth4580/Qwen3.5-0.8B-NVFP4-MIXED) | dense hybrid, `nvfp4-mixed` |
  | Qwen3.5 2B | [doth4580/Qwen3.5-2B-NVFP4-MIXED](https://huggingface.co/doth4580/Qwen3.5-2B-NVFP4-MIXED) | dense hybrid, `nvfp4-mixed` |
  | Qwen3.5 4B | [doth4580/Qwen3.5-4B-NVFP4-MIXED](https://huggingface.co/doth4580/Qwen3.5-4B-NVFP4-MIXED) | dense hybrid, `nvfp4-mixed` |
  | Qwen3.5 9B | [doth4580/Qwen3.5-9B-NVFP4-FULL](https://huggingface.co/doth4580/Qwen3.5-9B-NVFP4-FULL) | dense hybrid, `nvfp4-full` |
  | Qwen3.6 27B | [doth4580/Qwen3.6-27B-NVFP4-FULL](https://huggingface.co/doth4580/Qwen3.6-27B-NVFP4-FULL) | dense hybrid, `nvfp4-full` |
  | Qwen3.6 35B MoE | [doth4580/Qwen3.6-35B-A3B-NVFP4-FULL](https://huggingface.co/doth4580/Qwen3.6-35B-A3B-NVFP4-FULL) | MoE hybrid, `nvfp4-full` |
  | Qwen3.5 122B MoE | [doth4580/Qwen3.5-122B-A10B-NVFP4-MIXED](https://huggingface.co/doth4580/Qwen3.5-122B-A10B-NVFP4-MIXED) / [GDN4](https://huggingface.co/doth4580/Qwen3.5-122B-A10B-NVFP4-GDN4) | MoE hybrid, `nvfp4-mixed` or `gdn4` |
  | Tencent Hy3 | [doth4580/Tencent-Hy3-295B-A21B-NVFP4](https://huggingface.co/doth4580/Tencent-Hy3-295B-A21B-NVFP4) | 295B-A21B pure-GQA MoE |
  | KAT-Coder-V2.5-Dev | [doth4580/Kwaipilot-KAT-Coder-V2.5-Dev-NVFP4-MIXED](https://huggingface.co/doth4580/Kwaipilot-KAT-Coder-V2.5-Dev-NVFP4-MIXED) | 35B-A3B MoE hybrid, code specialist, `nvfp4-mixed` |
  | Qwen3.8-Flash-Next | quantize locally from [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (`--recipe all`, see [QWEN_FLASH_NEXT_SETUP.md](QWEN_FLASH_NEXT_SETUP.md)) | 176B-A10B MoE hybrid with hyper-connections + PLE n-gram table, **all NVFP4** incl. the PLE table (GPU-resident or `--ple-offload ssd`); QSA sparse attention past 2051 tokens (`--max-seq-len` up to 262144); image input (same tower as Qwen3.5) |

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
| 27B (full) | **33–42** | **42–53** ³ |
| 35B MoE (full) | **98–111** | **118–130** ² |

**Notes.** "Pending" cells land with the full benchmark run (tool-eval-bench `--perf`);
ranges are across 0–8K context. ¹ TP=2 on the small models (0.8B, 2B, 4B) is unoptimized —
barriers dominate at these sizes: TTFT is several times slower for little or no decode gain; run
them single-node. **TP=2 vs single**, same harness: 27B **1.1–1.3×** (best-vs-best 1.26×;
a matched-depth comparison measured 1.42–1.51× at 6–10K); 122B **1.1–1.3×**; 9B is
wash at short context but **~1.26× at 8K** (TP decode *rises* with context there). ² 35B: TP=2
leads at every measured depth (1.07–1.20×) — and halves per-node memory besides. ³ 27B TP=2 is
quoted best-of-runs (measured spread 42–53 tok/s across sweeps
— MTP acceptance variance; we report best-vs-best). MTP acceptance is workload-dependent (~35–85% across the family; prose accepts higher
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
| Qwen3.6 27B | ~721 | 2.7 s TTFT on a 2048-token prompt |

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
| `--vision-cpu` | off | Force the CPU vision tower (reference path) instead of the GPU tower. Diagnostic/escape hatch |
| `--gptq --model-dir <bf16> --base <artifact> --out <dir> --calib <jsonl> [--maca]` | — | Calibrated GPTQ→NVFP4 re-quantization, one layer at a time on one GB10. `--maca` enables variable context lengths with per-sequence Hessian normalization. See `docs/MACA_COLA_ACDM_MOE_CALIBRATION.md` and the [public reproducible corpus recipe](docs/PUBLIC_CALIBRATION_DATASETS.md). |
| `--calib-profile --model-dir <artifact> --calib <jsonl> --out <profiles.jsonl>` | — | Collect activation sketches and exact MoE routing counts for COLA/ACDM/expert-aware corpus selection |
| `--ple-offload <ssd\|none>` | none | Qwen3.8-Flash-Next only: keep the 31 GB PLE n-gram table on the SSD and read the rows each forward needs (bit-identical to resident; decode graphs off) |
| `--reasoning-effort <e>` | template default | Reasoning level in the chat template (`none`/`no_think`/`low`/`medium`/`high`/`xhigh`/`max`); per-request `reasoning_effort` overrides |
| `--output-prompts [n]` | off | Log each chat request human-readable (params, messages, rendered prompt); optional render cap `n` |
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

### TP=2/TP=4 flags (head) and node mode

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
`GB10_TP_TRACE=1` (per-barrier timing histograms at exit). Opt-in prefill levers (default off):
`GB10_FA_PREFILL=1` (tensor-core flash-attention prefill), `GB10_MXFP4_PREFILL=1` (v2 W4A4 prefill
GEMM), `GB10_W4A4_PREFILL=1|<groups>` (NVFP4 W4A4 prefill on the standard tiled weights;
`gdn-in` and `gdn-out` select the two GDN projection sides independently, while `gdn` enables both — Qwen3.5 dense and qwen4_exp,
see QWEN_FLASH_NEXT_SETUP.md), `GB10_GDN_CHUNK=1` / `GB10_GDN_CHUNK2=1` (GDN tensor-core chunked scan); these change the
prefill path and are on by default only where the gates hold.

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
- Alibaba's Qwen team — the Qwen3.5/3.6 model family that shaped the engine's early design, and
  the hybrid GatedDeltaNet architecture that shapes its best ideas.
- Tencent — the Hy3 model family and its pure-GQA MoE architecture.
- [`tool-eval-bench`](https://github.com/SeraphimSerapis/tool-eval-bench/) — the benchmark
  harness behind every number in this README.
- NVIDIA — the DGX Spark. One (or two) of them is all it takes.

## AI full disclosure

This software is developed with strong assistance from open source LLM models (GLM & Kimi) and with experienced software architect humans (i.e. me) leading the technical direction, many ideas, testing, and extensive debugging over a long time. We say this openly because it shaped how the project was built. If you are not happy with AI-developed code, this software is not for you. The acknowledgement below is equally important: this would not exist without existing knowledge and source code written by hand by actual humans.

## Acknowledgements to the general community

This project does not link extensively against any other project (other than the obvious and documented usages). However, due to the fact that LLM code generation does not occur in a vacuum, this project exists thanks to the path opened by the many other projects and the kernels, quantization formats, open source AI/LLM ecosystem, and hard-won engineering knowledge developed there. We are thankful and indebted to everyone who contributed to this area of computing and its contributors. Their implementations, experiments, code, ideas, kernels, tests, and design choices were, even if implicit through the weight encoded memory of the models we use, an essential reference while building this specific inference source code.

## Roadmap / pending items

Areas that are in flight or planned. These are tracked openly — progress and timelines are as honest as I can make them, and this list changes as work lands.

- **Fix Tencent Hy3 support.** Hy3 regressed over the last few weeks as the engine evolved; restoring it to a fully working, gated state is a priority.
- **Broaden vision coverage.** Vision is supported; work remains to bring it to the full set of Qwen models and harden it further.
- **Qwen3.5 397B MoE (incl. Ornith 1.5).** Large-model port; the engine already serves this architecture at 122B, so the work is the TP=2/TP=4 capacity bring-up (large weight footprint) plus the correctness gates at that size.
- **DeepSeek V4 Flash DSpark.** Work has started but it's far from complete or optimized. The goal is to beat all competition on decode speed across 2× and 4× GB10.
- **Other Qwen 3.8 variants.** If a 122B or other Qwen 3.8-size model fits on 1×, 2×, or 4× Spark, it's likely to be picked up next.
