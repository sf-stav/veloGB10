# DeepSeek-V4.1-Flash EXL3 + Engram — step-by-step setup

> **Preview (v0.8.0):** DeepSeek-V4.1-Flash support (and Velo-Fabric multi-node execution) is in **preview** state in veloGB10 v0.8.0. Expect rough edges and evolving flags across upcoming v0.8.x releases.

This guide walks through bringing up **DeepSeek-V4.1-Flash-1017** (`EXL3 2.9 bpw` + native FP8 Engram tables + 3-stage DSpark speculation) on **veloGB10** over **Velo-Fabric** (dual-rail 200GbE RoCEv2 Ethernet, no NCCL, no MPI), at **TP=2, TP=3, TP=4, TP=6, and TP=8**. It covers the required files on each machine, the exact launch commands, expected log lines, measured performance, and operational notes.

- **Target model:** **[doth4580/DeepSeek-V4.1-Flash-1017-Mia-EXL3-2.9bpw-Engram](https://huggingface.co/doth4580/DeepSeek-V4.1-Flash-1017-Mia-EXL3-2.9bpw-Engram)** — self-contained bundle (~386 GiB total on disk) combining:
  - **EXL3 2.9 bpw EAQ weights** quantized by **Mia-AiLab** ([`Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw`](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw)): 39 shards (~210.6 GB, 2.97 bpw effective; `lm_head` 6-bit, 3-stage DSpark drafter 4-bit).
  - **Native FP8 Engram tables** (`engram_source/model-00047-of-00048.safetensors` and `model-00048-of-00048.safetensors`, ~190 GiB) from [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), auto-detected with no extra flags.
  - **Precomputed manifests & SHA-256 digests** (`shard_hashes.json` + `engram_source/engram_head0..7.manifest.json`) to avoid hashing hundreds of GB on first boot.
- **Architecture:** 40 trunk layers + 3 DSpark speculative stages (block 5, Markov rank 256), hidden `5120`, `64` Q heads / `8` output groups (`head_dim 512`), `384` routed experts top-6 + `1` shared expert (`moe_inter 2304` = `18` EXL3 128-blocks), `8`-head Engram tables on layers `[1, 14]`, vocab `129,280`, ViT 32L + 3×3 vision aligner on rank 0.
- **Minimum topology:** **TP=2** (two 128 GB DGX Spark nodes). The 210.6 GB EXL3 weights exceed a single 128 GB box. Supported topologies are **TP=2, TP=3, TP=4, TP=6, and TP=8**.
- **Full context:** trained up to **1,048,576 tokens (1M)** (`--max-seq-len 1048576` on TP=3/4/6/8; `--max-seq-len 524288` recommended on TP=2 due to unified-memory headroom).

> **The head ships only the shards each node needs — you do not copy the 386 GiB pack to the node machines.**
> The head plans every rank's EXL3 weight segments and owned Engram heads and streams only missing blobs over RoCEv2 bulk RDMA into each node's content-addressed cache (`~/.cache/gb10_tp`). Warm restarts transfer nothing.

> **Multimodal vision is supported.** The bundled BF16 ViT 32L + 3×3 aligner runs on rank 0 (max 1,024 tokens/image) and splices image features across all TP ranks automatically.

---

## 1. The files you need

### Binary (every machine)

`gb10_inference` is a **single self-contained static binary** (~429 MB) with all Blackwell CUDA PTX kernels (`gpu_dsv41`, `gpu_inkling2`, etc.) embedded inside. Copy the exact same binary to the head and every peer node (for example, `~/stable_build/gb10_inference`). Head and nodes **must** run the same build — the engine refuses to boot on a build-id mismatch.

### Model pack (head only)

Download the self-contained pack onto the **head** machine using the Hugging Face CLI:

```bash
hf download doth4580/DeepSeek-V4.1-Flash-1017-Mia-EXL3-2.9bpw-Engram \
  --local-dir ./DeepSeek-V4.1-Flash-1017-Mia-EXL3-2.9bpw-Engram
```

Expected directory layout (~386 GiB):

```text
DeepSeek-V4.1-Flash-1017-Mia-EXL3-2.9bpw-Engram/
├── config.json
├── quantization_config.json
├── chat_template.jinja
├── tokenizer.json, tokenizer_config.json
├── model.safetensors.index.json
├── model-00001-of-00039.safetensors ... model-00039-of-00039.safetensors  (~197 GiB)
├── shard_hashes.json                                                       (precomputed digests)
└── engram_source/                                                          (~190 GiB, auto-detected)
    ├── config.json
    ├── model.safetensors.index.json
    ├── engram_head0.manifest.json ... engram_head7.manifest.json
    ├── model-00047-of-00048.safetensors
    └── model-00048-of-00048.safetensors
```

> **Alternative (separate EXL3 + base Engram directories):** If you downloaded `Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw` without the bundled `engram_source/` folder, pass `--engram-source /path/to/deepseek-ai/DeepSeek-V4.1-Flash-1017` on the head command line.

There is no separate drafter to download: the **3-stage DSpark speculative drafter** is built into the checkpoint and enabled automatically.

---

## 2. Start the nodes

Start one worker node process on **each peer machine** before starting the head:

```bash
~/stable_build/gb10_inference --node --port 29500
```

Expected output:

```text
[node-resident] 2026-11-24 21:00:00 supervisor up on port 29500 — one process per head session; kill this process to stop the node
[node] 2026-11-24 21:00:00 <hostname> ready: discovery on UDP 29499, control on TCP 29500, cache ~/.cache/gb10_tp
```

Nodes require **no model path and no other flags** — the head ships each rank's weights, Engram heads, and run options automatically. The examples below assume the head is `.11` and peer nodes are listening on port `29500` at:

- **TP=2:** `<peer-ip>:29500` (1 peer)
- **TP=3:** `<peer-ip>:29500` through `<peer-ip>:29500` (2 peers)
- **TP=4:** `<peer-ip>:29500` through `<peer-ip>:29500` (3 peers)
- **TP=6:** `<peer-ip>:29500` through `<peer-ip>:29500` (5 peers)
- **TP=8:** `<peer-ip>:29500` through `<peer-ip>:29500` (7 peers)

---

## 3. Bring up the server on the head

Replace the IP addresses and `--model-dir` path with your own. All multi-node topologies automatically enable dual-rail RoCEv2 (when both ConnectX-7 ports are up), P7SP sequence-parallel prefill, and DSpark speculation.

### TP=8 (7 nodes + head — highest throughput, fastest boot)

```bash
~/stable_build/gb10_inference --server --port 9000 \
  --model-dir /models/DeepSeek-V4.1-Flash-1017-Mia-EXL3-2.9bpw-Engram \
  --max-batch 8 --max-seq-len 1048576 --max-tokens 131072 \
  --prefix-cache on --tp 8 \
  --nodes <peer-ip>:29500,<peer-ip>:29500,<peer-ip>:29500,<peer-ip>:29500,<peer-ip>:29500,<peer-ip>:29500,<peer-ip>:29500 \
  --thermal-guard off
```

### TP=6 (5 nodes + head — exact `[3 × 6]` MoE split)

```bash
~/stable_build/gb10_inference --server --port 9000 \
  --model-dir /models/DeepSeek-V4.1-Flash-1017-Mia-EXL3-2.9bpw-Engram \
  --max-batch 8 --max-seq-len 1048576 --max-tokens 131072 \
  --prefix-cache on --tp 6 \
  --nodes <peer-ip>:29500,<peer-ip>:29500,<peer-ip>:29500,<peer-ip>:29500,<peer-ip>:29500 \
  --thermal-guard off
```

### TP=4 (3 nodes + head)

```bash
~/stable_build/gb10_inference --server --port 9000 \
  --model-dir /models/DeepSeek-V4.1-Flash-1017-Mia-EXL3-2.9bpw-Engram \
  --max-batch 8 --max-seq-len 1048576 --max-tokens 131072 \
  --prefix-cache on --tp 4 \
  --nodes <peer-ip>:29500,<peer-ip>:29500,<peer-ip>:29500 \
  --thermal-guard off
```

### TP=3 (2 nodes + head — 3-node sweet spot, exact `[6, 6, 6]` MoE split)

```bash
~/stable_build/gb10_inference --server --port 9000 \
  --model-dir /models/DeepSeek-V4.1-Flash-1017-Mia-EXL3-2.9bpw-Engram \
  --max-batch 8 --max-seq-len 1048576 --max-tokens 131072 \
  --prefix-cache on --tp 3 \
  --nodes <peer-ip>:29500,<peer-ip>:29500 \
  --thermal-guard off
```

### TP=2 (1 node + head — minimum footprint, 512K context)

```bash
~/stable_build/gb10_inference --server --port 9000 \
  --model-dir /models/DeepSeek-V4.1-Flash-1017-Mia-EXL3-2.9bpw-Engram \
  --max-batch 8 --max-seq-len 524288 --max-tokens 131072 \
  --prefix-cache on --tp 2 \
  --nodes <peer-ip>:29500 \
  --thermal-guard off
```

> **TP=2 memory note:** At TP=2, each rank holds ~105 GiB of EXL3 weights out of 128 GB unified memory, so the engine automatically sizes/disables the in-RAM prefix pool to stay within memory limits (and may reduce lanes if memory is tight). For heavy concurrency or full 1M context + prefix pooling, **TP=3** or higher is recommended.

### What you should see

On boot, the head prints the resolved configuration, syncs each peer's missing weight/Engram shards into `~/.cache/gb10_tp`, brings up the dual-rail Velo-Fabric RDMA mesh, loads and calibrates the DSpark drafter, and prints the ready line:

```text
gb10_inference v0.7.10 (veloGB10) — commit 71385304, kernel build 51a214cddc5395e4, source build 06155bd058b6c9af
configuration:
  mode           server (OpenAI-compatible HTTP API)
  model          /models/DeepSeek-V4.1-Flash-1017-Mia-EXL3-2.9bpw-Engram
  model params   deepseek_v41, 40 layers, hidden 5120, 384 experts (6 per token), vocab 129280, trained context 1048576
  weights        EXL3 2.9 bpw (lm_head 6 bits, DSpark drafter 4 bits)
  engram         layers [1, 14], 8 heads x n-gram orders 2..=4 (24 rows per token per layer), tables: DeepSeek's FP8 tables (native shards), owned by head per rank
  topology       TP=8, this box is the head; nodes: <peer-ip>:29500, ...
  ...
[boot] 2026-11-24 21:00:02 stage weights begin
[boot] 2026-11-24 21:01:01 stage weights done 59.0s
SERVING family=dsv41 model=DeepSeek-V4.1-Flash-1017-Mia-EXL3-2.9bpw-Engram tp=8 ctx=1048576 batch=8 kv=bf16 prefix=on mtp=auto spec=dspark port=9000 url=http://0.0.0.0:9000 boot=61.0s
```

Once `SERVING ...` appears, connect any OpenAI-compatible client to `http://<head-ip>:9000/v1`:

```bash
curl -s http://127.0.0.1:9000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"DeepSeek-V4.1-Flash-1017-Mia-EXL3-2.9bpw-Engram","temperature":0,"max_tokens":64,"messages":[{"role":"user","content":"What is 17 * 23? Answer with the number only."}]}'
```

**First boot vs warm boot:** Cold first boot packs and ships each peer's shards over RoCEv2 bulk RDMA into `~/.cache/gb10_tp`. Subsequent boots verify cached blobs locally and start in **~61 s (TP=8)** to **~152 s (TP=3)**.

---

## 4. Performance

Measured live on DGX Spark (GB10) nodes over dual-port 200GbE ConnectX-7 RoCEv2 via the OpenAI-compatible HTTP API (`/v1/chat/completions`, streaming, greedy `128 tok` decode, `4.4K–14.9K` cold prefill, 4-user concurrent batch):

| Topology | Mode | Warm Boot | 4.4K–5.2K Cold Prefill (`tok/s` / TTFT) | 8.0K–14.9K Cold Prefill (`tok/s`) | Turn-2 Prefix Hit TTFT | 1-User Decode (`tok/s`) | 4-User Batch (`agg tok/s` / per-user) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **TP=2** | FP32 (default) | `129 s` | `1,886` (`2.44 s`) | — | *(RAM-bound)* | `48.9` | `91.2` (`22.8`/u) |
| **TP=3** | FP32 + P7SP + LAGA (default) | `152 s` | **`2,261 – 2,275`** (`1.94 s`) | **`2,518 – 2,547`** (@ 8.0K) | **`157 – 220 ms`** | **`64.4`** | **`146.6`** (`36.7`/u) |
| **TP=3** | `--dsv41-wire-bf16 on` | `152 s` | `1,794 – 1,890` *(pre-SP/LAGA)* | **`2,177`** (@ 14.9K) | **`113 – 225 ms`** | **`68.1 – 70.6`** | **`145.0 – 183.7`** (`36.3–45.9`/u) |
| **TP=4** | FP32 + P7SP + LAGA (default) | `72 s` | **`2,764`** (`1.66 s`) | **`3,169 – 3,231`** (@ 8.0K) | **`142 ms`** | `58.5` | `135.1` (`33.8`/u) |
| **TP=6** | FP32 + P7SP *(pre-LAGA)* | `~95 s` | `2,147` (@ 4.4K) | **`2,873`** (@ 8.0K) | — | **`70.5 – 71.6`** | — |
| **TP=8** | FP32 + P7SP + LAGA (default) | **`61 s`** | **`3,094`** (`1.47 s`) | — | **`108 – 155 ms`** | **`87.6 – 92.6`** | **`219.9`** (`55.0`/u) |

See [docs/VELO_FABRIC_BENCHMARKS.md](docs/VELO_FABRIC_BENCHMARKS.md) and [VELO_FABRIC.md](VELO_FABRIC.md) for detailed micro-benchmarks and LAGA A/B wire reductions (`−18.8%` to `−19.3%` total cluster collective bytes).

---

## 5. Topology Sharding & Operational Notes

### Sharding geometry

| Topology | Q Heads / Output Groups per Rank | MoE 128-Blocks per Rank | Engram Heads per Rank | Weights / Rank (Rank 0) | Prefix Pool |
| --- | --- | --- | --- | --- | --- |
| **TP=2** | `[32, 32]` / `[4, 4]` | `[9, 9]` (exact) | `[0..4, 4..8]` | `~105 GiB` | Off *(RAM-bound)* |
| **TP=3** | `[24, 24, 16]` / `[3, 3, 2]` | `[6, 6, 6]` (exact) | `[0..2, 2..5, 5..8]` | `69.6 GiB` | On (16 GiB cap) |
| **TP=4** | `[16 × 4]` / `[2 × 4]` | `[5, 4, 5, 4]` | `[0..2, 2..4, 4..6, 6..8]` | `~53 GiB` | On (16 GiB cap) |
| **TP=6** | `[16, 16, 8, 8, 8, 8]` / `[2, 2, 1, 1, 1, 1]` | `[3 × 6]` (exact) | `[0..1, 1..2, 2..4, 4..5, 5..6, 6..8]` | `~38 GiB` | On (full) |
| **TP=8** | `[8 × 8]` / `[1 × 8]` | `[3, 2, 2, 2, 3, 2, 2, 2]` | `[0..1, 1..2, ..., 7..8]` | `30.1 GiB` | On (full) |

### Key flags & notes

- **Optional BF16 decode wire (`--dsv41-wire-bf16 on`):** Default is `off` (full FP32 wire, exact canonical reference numerics). Turning it `on` halves decode/verify collective bytes (still bit-identical across all ranks) for **+5–6% single-user decode speed** at TP=3 (`64.4 → 68.1–70.6 tok/s`).
- **LAGA BF16 attention-output All-to-All (`--dsv41-sp-laga auto`):** Enabled automatically on **TP=3, TP=4, TP=6, and TP=8** (and TP=2 when `--dsv41-attn-whole on`), cutting total forward-pass collective bytes by ~19%. Pass `--dsv41-sp-laga off` for A/B comparison.
- **DSpark speculation:** Enabled automatically (lossless greedy, distribution-exact sampled). Pass `--mtp off` (or `--dsv41-draft-len 0`) to force plain non-speculative decode.
- **Concurrency (`--max-batch`):** Supports `1..=8` concurrent lanes (default `4`). Requests beyond `--max-batch` queue cleanly.
- **Thermal pacing (`--thermal-guard`):** Default is `on` (paces prefill above 85 °C to prevent DGX Spark thermal shutdowns under sustained heavy load). The commands above pass `--thermal-guard off` for maximum unthrottled speed in well-cooled setups; omit it if your boxes run warm.
- **Terminal output (`--color` & `--log-level`):** `--color auto` (default) enables ANSI colors on interactive TTYs and stays plain when piped/redirected (`always` / `never` override). Every `[tag]` log line includes a local `YYYY-MM-DD HH:MM:SS` timestamp. Pass `--log-level debug` for verbose per-round telemetry.
- **Live telemetry (`/metrics`):** Prometheus metrics are served on the same HTTP port:
  ```bash
  curl -s http://127.0.0.1:9000/metrics | grep -E "velo_fabric|dsv41_sp|dsv41_tp_aborts|fatal_total"
  ```
- **No environment variables:** veloGB10 reads zero `GB10_*` environment variables; all configuration is passed via CLI flags and automatically shipped from the head to every TP node.
- **TP failure handling:** If a peer link drops mid-serve, the engine stops cleanly (`--exit-on-fatal on`, exit code 70) so a supervisor can restart it. The `~/.cache/gb10_tp` blob cache on peer nodes is safe to delete; the next boot re-ships only what is missing.

---

## 6. Credits & References

- **Model & Engram Tables:** DeepSeek-AI ([`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)).
- **EXL3 2.9 bpw Quantization:** **Mia-AiLab** ([`Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw`](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw)); bundled pack at [`doth4580/DeepSeek-V4.1-Flash-1017-Mia-EXL3-2.9bpw-Engram`](https://huggingface.co/doth4580/DeepSeek-V4.1-Flash-1017-Mia-EXL3-2.9bpw-Engram).
- **Velo-Fabric Architecture & Full Benchmarks:** [VELO_FABRIC.md](VELO_FABRIC.md), [docs/VELO_FABRIC_BENCHMARKS.md](docs/VELO_FABRIC_BENCHMARKS.md).
