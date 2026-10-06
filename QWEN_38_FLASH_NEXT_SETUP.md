# Qwen 3.8 Flash-Next EXL3 — step-by-step setup

This guide walks through bringing up **Qwen3.8-Flash-Next** from an **EXL3 pack** on veloGB10, at
**TP=1, TP=2 and TP=4**. It covers the required files on each machine, the exact launch commands, the
log lines you should expect, and the measured performance.

- **Target model:** two EXL3 packs of the same model — **use either one**, they differ only in
  precision (and therefore in speed and in how much disk they need):
  - **[doth4580/Qwen3.8-Flash-Next-EXL3-3.05bpw](https://huggingface.co/doth4580/Qwen3.8-Flash-Next-EXL3-3.05bpw)**
    — 3.05 bpw, ~86 GB. The smaller and faster of the two; the examples below use it.
  - **[doth4580/Qwen3.8-Flash-Next-EXL3-4.05bpw](https://huggingface.co/doth4580/Qwen3.8-Flash-Next-EXL3-4.05bpw)**
    — 4.05 bpw (turboderp's `4.05bpw_h6_ng6`: 4-bit experts, 6-bit dense layers and head, 6-bit
    n-gram table), ~108 GB. Higher fidelity, roughly 0.86× the decode rate of 3.05 and not yet
    autotuned.
- **Format:** EXL3 (ExLlamaV3 trellis). Everything else in this guide is identical for both packs —
  only the directory changes.
- **Full context:** launch with `--max-seq-len 262144` (the model's full 256K).

> **The head ships the weights each node needs — you do not copy the pack to the node machines.**
> The head plans every rank's shard and ships it through the TP blob cache (`~/.cache/gb10_tp`), so
> an EXL3 node receives only its own rank's share — roughly **57.5 GiB at TP=2** and **46.8 GiB at
> TP=4** — rather than the whole pack. This is the same automatic path the NVFP4 models use, and it
> runs whether or not a node already has a copy of the model.

> **Images are supported; video is not yet.** Image parts are served at TP=1, TP=2 and TP=4 through
> the API: the pack's original bf16 vision tower is loaded, each image is resized so its longer side
> is at most `--image-max-edge` (default 1024, aspect preserved), and images work past the 2,051-token
> dense window. Video and audio parts return `400` for now.

---

## 1. The files you need

### Binary + PTX (every machine)

The engine is **the binary + a `src/ptx/` directory of kernel artifacts**. The binary loads the PTX
relative to its **current working directory**, so run it from a directory that contains both. The
binary is ~28 MB; the PTX files total ~63 MB. Do not mismatch a binary with foreign PTX — the engine
refuses to start on a build-id mismatch.

On every machine (nodes included), the directory must look like this:

```
.
├── gb10_inference
└── src
    └── ptx
        ├── exl3_bench.ptx
        ├── exl3_bench_k6.ptx
        ├── fused_decode.ptx
        ├── gemm_nvfp4.ptx
        ├── gpu_batch.ptx
        ├── gpu_batch_b3.ptx
        ├── gpu_dflash.ptx
        ├── gpu_dspark.ptx
        ├── gpu_dsv4.ptx
        ├── gpu_dsv4_attn.ptx
        ├── gpu_dsv4_comp.ptx
        ├── gpu_kernels.ptx
        ├── gpu_mxfp4.ptx
        ├── gpu_mxfp4_moe.ptx
        ├── gpu_vision.ptx
        ├── gpu_w4a4.ptx
        ├── mxfp4_bench.ptx
        ├── rms_norm.ptx
        └── silu_gate.ptx
```

**Copy the binary and the whole `src/ptx/` directory to every machine.** The release tarball already
has this layout.

### The EXL3 pack (head only)

Download **one** of the two packs **onto the head** — 3.05 bpw (~86 GB) or 4.05 bpw (~108 GB). The
launch commands are identical apart from `--model-dir`. The nodes get their rank's shard from the head
at sync time (§2), so they do not need the pack:

```
Qwen3.8-Flash-Next-EXL3-3.05bpw/          (or Qwen3.8-Flash-Next-EXL3-4.05bpw/)
├── config.json
├── config.json.native
├── model.safetensors.index.json
├── model-00001-of-00007.safetensors
├── ...
├── ngram_embedding.safetensors          (32.6 GB at 3.05, 39.0 GB at 4.05 — the n-gram table)
├── mtp_hyper_connection_mixer_patch.safetensors
├── vision_tower_bf16.safetensors        (the vision tower — see below)
├── quantization_config.json
├── chat_template.jinja
├── tokenizer.json, tokenizer_config.json, vocab.json, merges.txt
└── generation_config.json
```

**The vision tower matters more on 4.05.** On **3.05** `vision_tower_bf16.safetensors` is preferred,
with a fallback to the 5-bit tower inside the shards. On **4.05** it is **required for image input**:
that pack's tensor index lists no vision tensors, so without the file the server runs text-only. Both
Hugging Face repositories include it.

There is no drafter to download: **MTP speculative decoding is built into this model** and enabled
automatically.

---

## 2. Start the nodes

Start one node per peer machine, **before** the head. Ensure §1 is in place on each, then run:

```bash
./gb10_inference --node --port 29500
```

Expected output:

```
[node-resident] supervisor up on port 29500 — one process per head session; kill this process to stop the node
[node] <hostname> ready: discovery on UDP 29499, control on TCP 29500, cache ~/.cache/gb10_tp
```

The node needs **no other flags** and no model path — the head ships each rank its shard. For the
examples below we assume nodes are running at:

- **TP=2:** `192.0.2.12:29500`
- **TP=4:** `192.0.2.12:29500`, `192.0.2.13:29500`, `192.0.2.14:29500`

> **A single machine (TP=1) needs no node process.** Skip to §3 and use the TP=1 command.

---

## 3. Bring up the server on the head

Ensure the head directory has the binary + `src/ptx/` and the EXL3 pack from §1. Run the binary from
the directory holding `gb10_inference` and `src/ptx/`.

### TP=4 (three nodes + head)

```bash
./gb10_inference --server \
  --tp 4 \
  --model-dir ~/veloGB10/Qwen3.8-Flash-Next-EXL3-3.05bpw \
  --nodes 192.0.2.12:29500,192.0.2.13:29500,192.0.2.14:29500 \
  --port 9000 \
  --max-seq-len 262144 \
  --max-batch 1
```

### TP=2 (one node + head)

```bash
./gb10_inference --server \
  --tp 2 \
  --model-dir ~/veloGB10/Qwen3.8-Flash-Next-EXL3-3.05bpw \
  --nodes 192.0.2.12:29500 \
  --port 9000 \
  --max-seq-len 262144 \
  --max-batch 1
```

### Single machine (TP=1)

```bash
./gb10_inference --server \
  --model-dir ~/veloGB10/Qwen3.8-Flash-Next-EXL3-3.05bpw \
  --port 9000 \
  --max-seq-len 262144 \
  --max-batch 1
```

### What you should see

On a TP run the head plans and ships each rank's shard — only the missing bytes move, so the second
start of the same model transfers nothing:

```
[head] <host> — building manifest for ~/veloGB10/Qwen3.8-Flash-Next-EXL3-3.05bpw (world 4) ...
[head] B32 plan rank 1/4 (exl3, deal interleave): 65 files, 46.79 GiB
[head] manifest 'Qwen3.8-Flash-Next-EXL3-3.05bpw': 195 artifacts, 150.72 GB (B32: per-rank shards through the blob cache)
[head] 192.0.2.12 (rank 1) needs 65 / 65 artifacts (50.24 GB)
[head] 192.0.2.12 (rank 1) READY — model at ~/.cache/gb10_tp/models/Qwen3.8-Flash-Next-exl3-3.05bpw@exl3-w4-r1-interleave (50.24 GB in 98.3s = 0.51 GB/s)
[head] shipped config to 192.0.2.12 (rank 1/4)
...
[head] all 3 node(s) synced.
```

Then the pack loads on every rank:

```
[exl3-serve] <host>: loading EXL3 pack ~/veloGB10/Qwen3.8-Flash-Next-EXL3-3.05bpw (width ..., max_pos 262144, kv-cache q8)
[exl3-serve] M1 pack head loaded (MTP draft head for chain-verify serving)
[exl3-serve] vision ON: tower from vision_tower_bf16.safetensors (original bf16 tower): 27 blocks x 1152 wide -> 2560
[exl3-serve] MTP chain-verify ON (k=7, greedy + sampled lanes, dynamic draft stop at 0.4; ...)
```

When the server is ready to serve, it prints:

```
[exl3-serve] listening on 0.0.0.0:9000 (model id: Qwen3.8-Flash-Next-exl3-3.05bpw)
[exl3-serve] TP: scheduler pinned to core 9; spawned host workers use cpus 5-8,15-17
[exl3-serve] TP head scheduler thread cpu mask: 9
[exl3-serve] TP control: step-go over RDMA while live, TCP Step only with events (TP-E)
```

On a TP run the head also confirms every peer and prints the rank map:

```
[exl3-serve] TP node rank 1 READY (mirror armed)
[exl3-serve] TP=4 API up: head = rank 0; rank 1 = 192.0.2.12 (192.0.2.12:29500); rank 2 = ...; rank 3 = ...
```

Once the `listening on 0.0.0.0:9000` line appears, the server is up and you can connect with any
OpenAI-compatible client at `http://<head-ip>:9000/v1`.

**The first sync is the slow part** (a few minutes per node, ~0.5 GB/s on the examples above) and the
first load reads ~85 GB from disk. Later starts are much faster: the node cache is already populated
and the engine's own caches exist.

---

## 4. Performance

Measured with **VeloBenchmark 0.1.0** against the served model, one request at a time, reasoning
effort low. Decode includes MTP speculation. The TP=1 and TP=2 columns are 2026-09-30 runs; the TP=4
column is a 2026-10-02 run on the v0.7.1 build.

**Pure-code decode** — ANSI C sorting, ~3.2K output tokens:

| | TP=1 | TP=2 | **TP=4** |
|---|---:|---:|---:|
| Decode median | 137 tok/s | 186 tok/s | **221 tok/s** |
| Decode min / max | 83.1 / 144 | 122 / 199 | 173 / 240 |
| Decode p50 / p90 / p99 | 137 / 142 / 144 | 186 / 193 / 195 | 221 / 234 / 237 |

**Prefill** — one measurement per input size:

| Input tokens | TP=1 tok/s | TP=2 tok/s | **TP=4 tok/s** | **TP=4 TTFT** |
|---|---:|---:|---:|---:|
| ~550 | 1,323 | 2,001 | 2,285 | 0.24 s |
| ~2.1K | 1,470 | 2,319 | **3,250** | 0.64 s |
| ~6.2K | 1,517 | 2,441 | 3,227 | 1.92 s |
| ~10.3K | 1,541 | 2,452 | 3,226 | 3.19 s |
| ~18.5K | 1,541 | 2,451 | 3,246 | 5.69 s |
| ~34.9K | 1,525 | 2,432 | 3,197 | 10.90 s |

TP=4 against a single GB10: **×1.6** on this decode workload and **×2.0–2.2** on prefill.

### Running the 4.05 bpw pack instead

The commands above are identical for the other pack — point `--model-dir` at
`Qwen3.8-Flash-Next-EXL3-4.05bpw`. It is higher fidelity and slower, and it is **not autotuned yet**.
Untuned smoke figures (1,000 generated tokens / 2K-token prefill, single request), against the 3.05
numbers in the tables above on the same test:

| | TP=1 | TP=2 | TP=4 |
|---|---:|---:|---:|
| 3.05 bpw — decode | ~73 tok/s | ~102 tok/s | ~122 tok/s |
| **4.05 bpw — decode** | 63 tok/s | 89 tok/s | 120 tok/s |
| **4.05 bpw — prefill** | 1,399 tok/s | 2,612 tok/s | 3,496 tok/s |

Decode is about **0.86×** the 3.05 rate at TP=1 and TP=2. Two things to keep in mind with this pack:
its n-gram table is 39 GB rather than 32.6 GB, so `--ple-ram auto` is more likely to place it on SSD
(see §5), and on 4.05 `vision_tower_bf16.safetensors` is **required** for image input (§1). The 3.05
tables above stay the measured reference; the 4.05 rows are untuned smoke tests, not a matched run.

---

## 5. Notes

- **Images only.** Image parts are served at every topology; **video and audio parts return `400`**
  until video support lands. Images are resized so the longer side is at most `--image-max-edge`
  (default 1024); the QSA indexer follows the image positions, so images work past 2,051 tokens.
- **262,144 tokens is the maximum.** YaRN is not implemented, so there is no 1M context.
- **Concurrency.** Every number in §4 is `--max-batch 1` (one request). `--max-batch N` serves N
  requests at once (every lane's KV is allocated up front: about 5.6 GB per lane at the full 262,144-token
  context with the default q8 KV cache, 2.8 GB at 131K). With one busy request the engine speculates
  exactly as before. From about four busy requests (earlier on low-acceptance prose) it shares one batched
  step across all of them (`--spec-lanes-max auto`, the default): on one node roughly ×1.45 aggregate at 8
  requests and ×1.9 at 16, with TP=2 and TP=4 gaining proportionally (numbers in the CHANGELOG). **Two or
  three concurrent requests still take turns** — the aggregate stays at the single-request rate — and a
  packed multi-request verify for that range is in development. Each request's greedy output is
  byte-identical to running it alone. `--spec-lanes-max N` shares the step only above N busy requests; `0`
  keeps serial speculation.
- **Memory and lane count.** The weights, the n-gram table and every lane's KV share the same unified
  memory. `--ple-ram auto` (the default) keeps the n-gram table in RAM when it fits with 16 GiB to spare
  after everything else is allocated, and otherwise reads it from SSD (identical output, decode ~1–4%
  slower). If a configuration cannot fit even then, the server refuses to start **before loading** and
  prints the arithmetic and the fix (`--ple-ram ssd`, a smaller `--max-batch` or `--max-seq-len`, or
  `--tp 2`). Rule of thumb on one 128 GB Spark at `--max-seq-len 131072`: one lane with the table in RAM;
  up to about 16 lanes with the table on SSD.
- **Monitoring and binding.** `GET /metrics` serves Prometheus text format (requests, tokens, requests
  running/waiting, time-to-first-token and decode-rate histograms, speculation, engine health).
  `--host 127.0.0.1` binds the API to the local machine only (default: all interfaces).
- **The TP=2/TP=4 nodes need no pack copy** — the head ships each rank's shard through the blob cache
  at `~/.cache/gb10_tp`. The cache is safe to delete; the next run re-syncs what is missing.
- **Prefix caching changes wording, not correctness.** A cached turn is not bit-identical to a cold
  one — reuse re-chunks the prefill. Pass `--prefix-tail-ckpt 0` if you need bit-identical resumes.
- **Seeded sampled requests are not byte-reproducible across runs.** Greedy decoding is
  bitwise-lossless with respect to non-speculative decoding: MTP never changes a greedy token.
- **If a TP link fails mid-serve the server stops** and must be restarted; there is no auto-restart.
  A node that fails during *boot* fails the head loudly instead of hanging.
- **Environment variables are gone.** The engine reads none: leaving a `GB10_*` variable set refuses
  startup and names the replacement flag. The full env→flag table is in **docs/ENV_TO_FLAGS.md**.
- **New in v0.7.1, on by default:** `--prefill-chunk 4095` (wider prefill chunks; `--prefill-chunk
  2048` restores the previous grid) and `--qsa-key-rope full` (the indexer's pooled keys carry their
  full rotary dimensions; `half` keeps the old bytes for A/B only). Both change long-context output
  bytes versus v0.7.0.

---

## 6. The 4.05 bpw pack

A higher-fidelity pack — turboderp's `4.05bpw_h6_ng6`: 4-bit experts, 6-bit dense layers and head, a
6-bit n-gram table — is at
[doth4580/Qwen3.8-Flash-Next-EXL3-4.05bpw](https://huggingface.co/doth4580/Qwen3.8-Flash-Next-EXL3-4.05bpw)
(~108 GB; **needs v0.7.2 or newer**, which adds `src/ptx/exl3_bench_k6.ptx` to the deploy set — copy the
whole `src/ptx/` directory as always). Download it like the 3.05 pack and use **the same launch commands**
with `--model-dir` pointing at it.

- **TP=1:** the weights (~68 GB) plus the 39 GB n-gram table do not fit one GB10 together, so the table
  is read from SSD — `--ple-ram auto` chooses that by itself (you can force `--ple-ram ssd`).
- **TP=2 / TP=4:** each node receives about 71 GiB / 57 GiB (the n-gram table goes to every rank); the
  head ships it through the blob cache as for 3.05.
- **Images** work at every topology; keep `vision_tower_bf16.safetensors` in the model directory (it is in
  the Hugging Face repo) — without it the server runs text-only.
- **Untuned smoke-test figures** (1,000 generated tokens, 2K prefill, single request): TP=1 63 tok/s decode
  and 1,399 prefill; TP=2 89 and 2,612; TP=4 120 and 3,496. Decode is about 0.86× the 3.05 pack on one or
  two nodes. VeloBenchmark figures will follow.

## 7. Third-party packs and the exllamav3 1.5.x sidecar layout (v0.7.3)

Since v0.7.3, veloGB10 loads both n-gram sidecar layouts written by exllamav3: the original
single-tensor `ngram_embedding.trellis` and the exllamav3 1.5.x layout of 128 `shard_N.trellis`
tensors (K=5 and K=6), including packs whose `model.safetensors.index.json` does not list them
(the loader falls back to the sidecar's own safetensors header and validates the layout; a boot
line names the layout in use). Packs whose index lists the shards, and single-tensor packs,
load exactly as before (byte-identity gated). No flags needed — the new layout just loads.

### Tested third-party packs

| repo | revision / commit | bpw | K | n-gram layout | TP | c=1 tok/s (best of 3) | MTP acceptance | checked |
|---|---|---|---|---|---|---|---|---|
| SharkWipf/Swift-1.5-Qwen3.8-Flash-Next-exl3 | `4.05bpw_h6_ng6` @ `7bbb89df` | 4.05 | 6 | 128 shards in sidecar, not in index | 1, 2 | 55.4 / 83.4 | up to 91.4% (TP1), 88.9% (TP2) | boot, 5 sanity prompts, tool call, 8K prompt recall, mtp-stats, byte-identity gates |
| (synthetic, from doth4580 4.05) | fixture `…-shardedfixture` | 4.05 | 6 | 128 shards, no index entries | 1 | — | — | `IDENT_ALL` = original (`5502280475b19ea3`) |
| (synthetic, from doth4580 3.05) | fixture `…-noindexfixture` | 3.05 | 5 | 128 shards, index stripped | 1 | — | — | `IDENT_ALL` = original (`bf61490644d5c0e3`) |
| doth4580/Qwen3.8-Flash-Next-EXL3-3.05bpw (regression) | current | 3.05 | 5 | 128 shards, index lists them | 1 | — | — | code+prose `IDENT_ALL` unchanged across builds |
| doth4580 4.05bpw (regression) | current | 4.05 | 6 | single trellis tensor | 1 | — | — | code `IDENT_ALL` unchanged across builds |

## 8. Using Flash-Next as a coding-agent backend

Two flags and one parser rule matter when an agent harness drives the server for hours.

**`--max-batch 2` (no behaviour change).** A coding agent periodically sends a
compaction/summary request in the middle of a long conversation. With one lane, that request's
different prompt overwrites the lane's cache and the NEXT normal request re-prefills ~90K
tokens (~80 s each in the trace we analyzed — about a quarter of the session's wall time). With
two lanes the summary takes the second lane and the main conversation's cache survives. Cost:
a second lane's KV (~5 GB at 131K f16 with `--kv-cache f16`).

**`--keep-tools-when-tool-choice-none` (opt-in, default off).** The engine default (llama.cpp
behaviour) removes the tool definitions from the prompt when `tool_choice:"none"` — but this
model's chat template renders the tools inside the system block at the very start, so the
whole prompt differs from token 0 and nothing can be reused. The flag keeps the tools block
in the prompt (byte-identical to a normal request, so the prefix cache and checkpoints
survive), appends a `do-not-call` instruction to the last message, and returns the reply as
plain content. In the replayed reporter trace (309-message conversation, ~89K prompt tokens):
with the flag off, one summary turn plus its follow-up meant three full ~80 s re-prefills;
with it on, the summary resumed at 99.9% cache (7.9 s) and the follow-up at 100% (1.3 s).
Caveat: the model still SEES the tools and is merely told not to call them — a stray
`<tool_call>` comes back as visible text instead of being executed. `--exit-on-fatal` pairs
well with a supervisor for unattended runs (see `docs/OPERATIONS.md`).

**Tool-call values that quote tool-call syntax.** A parameter value may itself contain the
text `</parameter>` (an agent writing a file that documents the tool-call format). The parser
ends a value only at the *structural* `</parameter>` — the one followed by the next
`<parameter=`, by the `</function>` closing the call, or by the end of the call block.
Embedded close tags that fail this lookahead stay part of the value. Two residual ambiguities
are inherent to the format: (1) a value that ends with the *complete* closing sequence
(`</parameter>` + `</function>`) right at the end of the call block is indistinguishable from
the structural close and will be cut there; (2) an embedded `</tool_call>` still ends the
whole tool-call block. Advice for agent authors: when a tool must write text that quotes
tool-call syntax, prefer an edit/patch tool with a diff, or break the closing sequence (write
`< /parameter>` or insert a blank line inside the tag) — the parser cannot recover a complete
closing sequence at the very end of a value.

## 9. The stack we measured on, and what decode speed to expect

Measured on four DGX Spark units running DGX OS 7.5.0 (Ubuntu 24.04.4), kernel
`6.17.0-1029-nvidia`, NVIDIA driver 580.173.02, CUDA 13.0. We have NOT run DGX OS 7.6.0 /
kernel `7.0.0-1019-nvidia` — if you are on that stack and see anything odd, tell us; we
cannot yet say whether it behaves identically.

Decode speed depends on the workload through MTP acceptance: on near-pure code completions
the draft accepts often and a single request runs ~137 tok/s (3.05 bpw, TP=1), while mixed
code+prose sessions measure ~73 tok/s (3.05) and ~63 tok/s (4.05) — the same engine path,
different acceptance. The figures in section 4 and the 4.05 smoke numbers are the same story:
expect the pure-code end only when your workload is mostly code; agent conversations with
tools and prose sit at the lower end. All are single-request numbers; concurrency changes the
picture as described in the README (v0.7.2 section).

