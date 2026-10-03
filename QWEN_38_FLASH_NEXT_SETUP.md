# Qwen 3.8 Flash-Next EXL3 — step-by-step setup

This guide walks through bringing up **Qwen3.8-Flash-Next** from an **EXL3 pack** on veloGB10, at
**TP=1, TP=2 and TP=4**. It covers the required files on each machine, the exact launch commands, the
log lines you should expect, and the measured performance.

- **Target model:** `doth4580/Qwen3.8-Flash-Next-EXL3-3.05bpw` — https://huggingface.co/doth4580/Qwen3.8-Flash-Next-EXL3-3.05bpw
- **Format:** EXL3 (ExLlamaV3 trellis), 3.05 bpw. About 85 GB, including a 32.6 GB n-gram table.
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

Download the pack from `doth4580/Qwen3.8-Flash-Next-EXL3-3.05bpw` **onto the head**. The nodes get
their rank's shard from the head at sync time (§2), so they do not need the pack:

```
Qwen3.8-Flash-Next-EXL3-3.05bpw/
├── config.json
├── config.json.native
├── model.safetensors.index.json
├── model-00001-of-00007.safetensors
├── ...
├── ngram_embedding.safetensors          (32.6 GB — the n-gram table)
├── mtp_hyper_connection_mixer_patch.safetensors
├── vision_tower_bf16.safetensors        (the original bf16 vision tower — used for images when present;
│                                         the 5-bit tower inside the shards is the fallback)
├── quantization_config.json
├── chat_template.jinja
├── tokenizer.json, tokenizer_config.json, vocab.json, merges.txt
└── generation_config.json
```

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

- **TP=2:** `192.168.177.12:29500`
- **TP=4:** `192.168.177.12:29500`, `192.168.177.13:29500`, `192.168.177.14:29500`

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
  --nodes 192.168.177.12:29500,192.168.177.13:29500,192.168.177.14:29500 \
  --port 9000 \
  --max-seq-len 262144 \
  --max-batch 1
```

### TP=2 (one node + head)

```bash
./gb10_inference --server \
  --tp 2 \
  --model-dir ~/veloGB10/Qwen3.8-Flash-Next-EXL3-3.05bpw \
  --nodes 192.168.177.12:29500 \
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
[head] 192.168.177.12 (rank 1) needs 65 / 65 artifacts (50.24 GB)
[head] 192.168.177.12 (rank 1) READY — model at ~/.cache/gb10_tp/models/Qwen3.8-Flash-Next-exl3-3.05bpw@exl3-w4-r1-interleave (50.24 GB in 98.3s = 0.51 GB/s)
[head] shipped config to 192.168.177.12 (rank 1/4)
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
[exl3-serve] TP=4 API up: head = rank 0; rank 1 = 192.168.177.12 (192.168.177.12:29500); rank 2 = ...; rank 3 = ...
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

