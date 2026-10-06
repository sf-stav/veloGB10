# Qwen 3.8 27B NVFP4 — step-by-step setup

This guide walks through bringing up the **Qwen3.8 27B NVFP4** model with **DFlash 2** speculative
decoding on veloGB10, in three deployment shapes: **single node**, **TP=2**, and **TP=4**. It covers
the required files on each machine, the exact launch commands, and the log lines you should expect.

- **Target model:** `doth4580/Qwen3.8-27B-NVFP4-FULL` — https://huggingface.co/doth4580/Qwen3.8-27B-NVFP4-FULL
- **Drafter (requires download):** `doth4580/Qwen3.8-27B-DFlash2-NVFP4-RTNcal-FIXED` —
  https://huggingface.co/doth4580/Qwen3.8-27B-DFlash2-NVFP4-RTNcal-FIXED — an NVFP4 drafter that is
  slightly faster than the bf16 one.
- **Full context:** launch with `--max-seq-len 262144` (the model's full 256K).

> **Use only our quantized models.** Both the target model and the drafter are published from this
> account, and both are built and tuned specifically for the veloGB10 engine. An alternative for the
> target is to run `--quantize all` on the original Qwen bf16 model — the result is identical to our
> NVFP4 artifact.

> **The older bf16 drafter still works.** `doth4580/Qwen3.8-27B-DFlash2` remains valid — point
> `--draft-dir` at it instead. The examples below use the newer NVFP4 drafter.

---

## 1. The files you need

### Binary + PTX (every machine)

The engine is **the binary + a `src/ptx/` directory of kernel artifacts**. The binary loads the PTX
relative to its **current working directory**, so run it from a directory that contains both. The
binary is ~27 MB; the PTX files total ~37 MB. Do not mismatch a binary with foreign PTX — the
engine refuses to start on a build-id mismatch.

On every machine (nodes included), the directory must look like this:

```
.
├── gb10_inference
└── src
    └── ptx
        ├── exl3_bench.ptx
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

**Copy the binary and the `src/ptx/` directory to each node machine.**

### The drafter (on every machine that will serve)

The DFlash 2 drafter checkpoint must be present locally:

- `Qwen3.8-27B-DFlash2-NVFP4-RTNcal-FIXED/` — `config.json`, `model.safetensors`,
  `nvfp4.safetensors`, `README.md`, `LICENSE` (the full directory as downloaded from
  `doth4580/Qwen3.8-27B-DFlash2-NVFP4-RTNcal-FIXED`). This is the NVFP4 drafter the examples use.
- The older bf16 drafter still works: `Qwen3.8-27B-DFlash2/` — `config.json`, `model.safetensors`,
  `README.md`, `dl.log` (as downloaded from `doth4580/Qwen3.8-27B-DFlash2`). Point `--draft-dir` at
  it instead; nothing else changes.

### The target model (head only)

You do **not** need to copy the target model to the node machines. The head ships what each node
needs automatically (see §4). Only the **head** needs the full model directory plus the drafter.

---

## 2. Start the node(s)

On each node machine, first ensure the directory from §1 is in place (binary + `src/ptx/`), then run:

```bash
./gb10_inference --node --port 29500
```

Expected output:

```
[node-resident] supervisor up on port 29500 — one process per head session; kill this process to stop the node
[node] <hostname> ready: discovery on UDP 29499, control on TCP 29500, cache ~/.cache/gb10_tp
```

At this point the nodes are waiting for the head to launch.

For the examples below we assume nodes are running at:

- **TP=4:** `192.0.2.12:29500`, `192.0.2.13:29500`, `192.0.2.14:29500`
- **TP=2:** `192.0.2.12:29500`

> **You do NOT need to copy the model to the nodes.** Required model shards and files are transferred
> automatically by the head, and in most cases only the files that are actually needed are copied —
> not necessarily the whole model.

---

## 3. Bring up the server on the head

Ensure the head directory has the binary + `src/ptx/`, the target model directory, and the drafter
directory. For this example, copy the models next to the binary, each in its own subdirectory:

```
.
├── 3.8-27b-nvfp4-full-all
│   ├── LICENSE
│   ├── README.md
│   ├── chat_template.jinja
│   ├── config.json
│   ├── generation_config.json
│   ├── merges.txt
│   ├── model-00001.safetensors
│   ├── model-00002.safetensors
│   ├── model.safetensors.index.json
│   ├── preprocessor_config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.json
├── Qwen3.8-27B-DFlash2-NVFP4-RTNcal-FIXED
│   ├── LICENSE
│   ├── README.md
│   ├── config.json
│   ├── model.safetensors
│   └── nvfp4.safetensors
├── gb10_inference
└── src
    └── ptx
        └── ... (as in §1)
```

`3.8-27b-nvfp4-full-all` is the main model directory;
`Qwen3.8-27B-DFlash2-NVFP4-RTNcal-FIXED` is the DFlash 2 drafter directory (the older bf16
`Qwen3.8-27B-DFlash2` directory works equally well here). In a single-node or TP deployment, verify
the full directory tree from §1 and these two model directories before launching.

### TP=4 (three nodes + head)

```bash
./gb10_inference --server \
  --model-dir ~/veloGB10/3.8-27b-nvfp4-full-all \
  --tp 4 \
  --nodes 192.0.2.12:29500,192.0.2.13:29500,192.0.2.14:29500 \
  --port 9000 \
  --kv-cache k8v8 \
  --max-seq-len 262144 \
  --max-batch 1 \
  --max-tokens 65536 \
  --prefix-cache on \
  --default-presence-penalty 0.0 \
  --mtp=auto \
  --fp8-prefill on \
  --reasoning-effort low \
  --df2-block 16 \
  --df2-round-shard on \
  --spec-source dflash2 \
  --draft-dir ~/models/maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal-FIXED
```

### TP=2 (one node + head)

```bash
./gb10_inference --server \
  --model-dir ~/veloGB10/3.8-27b-nvfp4-full-all \
  --tp 2 \
  --nodes 192.0.2.12:29500 \
  --port 9000 \
  --kv-cache k8v8 \
  --max-seq-len 262144 \
  --max-batch 1 \
  --max-tokens 65536 \
  --prefix-cache on \
  --default-presence-penalty 0.0 \
  --mtp=auto \
  --fp8-prefill on \
  --reasoning-effort low \
  --df2-block 16 \
  --df2-round-shard on \
  --spec-source dflash2 \
  --draft-dir ~/models/maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal-FIXED
```

### Single node

```bash
./gb10_inference --server \
  --model-dir ~/veloGB10/3.8-27b-nvfp4-full-all \
  --port 9000 \
  --kv-cache k8v8 \
  --max-seq-len 262144 \
  --max-batch 1 \
  --max-tokens 65536 \
  --prefix-cache on \
  --default-presence-penalty 0.0 \
  --mtp=auto \
  --fp8-prefill on \
  --reasoning-effort low \
  --df2-block 16 \
  --df2-round-shard on \
  --spec-source dflash2 \
  --draft-dir ~/models/maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal-FIXED
```

### Enabling concurrency with `--max-batch`

All three examples above use `--max-batch 1` (one request handled at a time — maximum per-request
speed). To serve **multiple concurrent clients** instead, raise `--max-batch` to the number of
simultaneous requests you want to handle, e.g.:

```bash
./gb10_inference --server \
  --model-dir ~/veloGB10/3.8-27b-nvfp4-full-all \
  --tp 4 \
  --nodes 192.0.2.12:29500,192.0.2.13:29500,192.0.2.14:29500 \
  --port 9000 \
  --max-seq-len 262144 \
  --max-batch 4 \
  --max-tokens 65536 \
  --prefix-cache on \
  --default-presence-penalty 0.0 \
  --mtp=auto \
  --fp8-prefill on \
  --reasoning-effort low \
  --df2-block 16 \
  --df2-round-shard on \
  --spec-source dflash2 \
  --draft-dir ~/models/maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal-FIXED
```

> The concurrency example leaves `--kv-cache` at its default. Since v0.7.2 `k8v8` (used in the
> single-request examples) also works with `--max-batch` greater than 1 — earlier versions refused it.

- `--max-batch N` is the max concurrent sequences (lanes) the server will run. With `N > 1` the
  scheduler batches the concurrent greedy lanes into a single verify forward, so you trade a little
  per-request latency for much higher aggregate throughput across clients.
- With DFlash 2 the drafter serves one request at a time: above one lane the requests run as a plain
  batch (no per-request speculation), and a request that shared a batched step does not resume
  speculating for the rest of that request.
- Memory scales with the batch: the KV cache is allocated per-lane, so `--max-batch 8` costs roughly
  8× the KV memory of `--max-batch 1` at the same `--max-seq-len`.

### What you should see

During the head bring-up you should see lines like:

```
[tp] config installed: world=4 shard_mixers=true shard_mtp=true graph=false ...
[head] <hostname> — building manifest for ~/veloGB10/3.8-27b-nvfp4-full-all (world 4) ...
[head] manifest '3.8-27b-nvfp4-full-all': 39 artifacts, 48.95 GB
[head] draft manifest 'Qwen3.8-27B-DFlash2': 6 artifacts, 3.85 GB — ships to every node
[head] 192.0.2.12 (rank 1) needs 0 / 13 artifacts (0.00 GB)
[head] 192.0.2.12 (rank 1) READY — model at ~/.cache/gb10_tp/models/3.8-27b-nvfp4-full-all (0.00 GB in 0.0s = 0.00 GB/s)
[head] shipped config to 192.0.2.12 (rank 1/4)
[head] 192.0.2.12 drafter: 5 / 6 artifacts (3.85 GB)
[head] 192.0.2.12 drafter READY at ~/.cache/gb10_tp/models/Qwen3.8-27B-DFlash2 (3.85 GB in 7.6s)
...
[head] 3 node(s) synced; all control streams RETAINED for the serving session
[tp] rank 0/4 — bringing up RDMA data-plane link on rocep1s0f1 (listening) ...
[tp] rank 0/4 — link UP
[tp] rank 0/4 — data-plane all-reduce link SANE
[df2] DFlash2 round RESIDENT (spec-source=dflash2-auto) — serving via the S4F integrated round (b==1 lanes); MTP remains the fallback (standing directive)
TP -- all 3 node(s) READY; binding HTTP
OpenAI-compatible server running on http://0.0.0.0:9000
Serving model: 3.8-27b-nvfp4-full-all  (GET /v1/models)
POST /v1/chat/completions   max_batch=1  default max_tokens=65536
```

Once you see the final three lines, the server is up and you can connect with any OpenAI-compatible
client.

**Checking the nodes connected properly** — look for the `READY` lines listing your node IPs:

- **TP=4:** three `READY` lines (ranks 1, 2, 3) — if you get all 3, your nodes are configured correctly.
- **TP=2:** one `READY` line for rank 1.

**Checking DFlash 2 is enabled** — look for this line:

```
[df2] DFlash2 round RESIDENT (spec-source=dflash2-auto) — serving via the S4F integrated round (b==1 lanes); MTP remains the fallback (standing directive)
```

---

## 4. Notes

- **First load is slower.** The first time the model loads, the engine creates caches for various
  things. Subsequent loads are significantly faster.
- **Model transfer is automatic.** The head transfers model shards and config to the nodes on demand;
  in most cases only the needed files are copied, not the whole model.
- **Cache management.** The engine caches transferred model blobs on the node at
  `~/.cache/gb10_tp`. To inspect or reclaim that cache, see **MANAGING_CACHE.md**.
