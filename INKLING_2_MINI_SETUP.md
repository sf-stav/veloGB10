# Inkling-2-Mini NVFP4 — step-by-step setup

This guide walks through bringing up **Inkling-2-Mini** (stock `NVFP4 / MXFP8` looped MoE + 3-layer MTP speculation) on **veloGB10** over **Velo-Fabric** (dual-rail 200GbE RoCEv2 Ethernet, no NCCL, no MPI), at **TP=1, TP=2, TP=3, TP=4, TP=6, and TP=8**. It covers the required files on each machine, the exact launch commands, expected log lines, measured performance, and operational notes.

- **Target model:** **[thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)** — stock NVFP4 / MXFP8 checkpoint (~19 GB total on disk: `model-00001-of-00001.safetensors` ~19 GB + bundled `mtp.safetensors` 607 MB).
- **Architecture:** 24 layers (48 looped visits), `128` routed experts top-8 (`moe_inter 1152`), hidden `2304`, `24` Q heads / `8` KV heads, vocab `201,024`, 3-layer MTP speculative drafter, 23-layer BF16 vision tower (`0.64 GB`).
- **Supported topologies:** **TP=1** (single DGX Spark) and multi-node **TP=2, TP=3, TP=4, TP=6, and TP=8**.
- **Full context:** trained up to **262,144 tokens (256K)** (defaults to `--max-seq-len 262144` when omitted; smaller values like `32768` or `65536` reduce KV memory per lane for higher batch sizes).

> **The head ships only the shards each node needs — you do not copy the model to the node machines.**
> On multi-node runs (`--tp 2|3|4|6|8`), the head plans every rank's shard (whole-KV-head attention + balanced 64-channel MoE inner splits + vocab-parallel LM head/MTP) and streams missing blobs over RoCEv2 bulk RDMA into each node's `~/.cache/gb10_tp` cache. Warm 8-node restarts boot in **~13 seconds**.

> **Multimodal vision is supported.** The bundled 23-layer BF16 vision tower runs natively on GPU (content-keyed tower cache included) and accepts image inputs at every topology. Cap per-image LM tokens with `--image-max-tokens` (note: `--image-max-edge` does not apply to Inkling-2).

---

## 1. The files you need

### Binary (every machine)

`gb10_inference` is a **single self-contained static binary** (~429 MB) with all Blackwell CUDA PTX kernels (`gpu_inkling2`, `gpu_dsv41`, etc.) embedded inside. Copy the exact same binary to the head and every peer node (for example, `~/stable_build/gb10_inference`). Head and nodes **must** run the same build — the engine refuses to boot on a build-id mismatch.

### Model pack (head only)

Download the stock checkpoint onto the **head** machine using the Hugging Face CLI:

```bash
hf download thinkingmachines/Inkling-2-Mini --local-dir ./Inkling-2-Mini
```

Expected directory layout (~19 GB):

```text
Inkling-2-Mini/
├── config.json
├── generation_config.json
├── hf_quant_config.json
├── processor_config.json
├── chat_template.jinja
├── tokenizer.json, tokenizer_config.json, special_tokens_map.json
├── tiktoken/
├── model.safetensors.index.json
├── model-00001-of-00001.safetensors   (~19 GB — NVFP4/MXFP8 trunk + BF16 vision tower)
└── mtp.safetensors                    (607 MB — bundled 3-layer MTP speculative drafter)
```

> **Stock NVFP4/MXFP8 pack required:** Only the stock `thinkingmachines/Inkling-2-Mini` NVFP4/MXFP8 layout is served by the GPU engine (third-party EXL3 Inkling-2 packs are not yet wired to the GPU forward).

---

## 2. Start the nodes (multi-node TP=2/3/4/6/8 only)

> **Single machine (TP=1) needs no node process.** Skip directly to §3 and use the TP=1 command.

On **each peer machine**, start one worker node process before starting the head:

```bash
~/stable_build/gb10_inference --node --port 29500
```

Expected output:

```text
[node-resident] 2026-11-24 21:00:00 supervisor up on port 29500 — one process per head session; kill this process to stop the node
[node] 2026-11-24 21:00:00 <hostname> ready: discovery on UDP 29499, control on TCP 29500, cache ~/.cache/gb10_tp
```

Nodes require **no model path and no other flags** — the head ships each rank's weights and run options automatically. The examples below assume the head is `.11` and peer nodes are listening on port `29500` at:

- **TP=2:** `<peer-ip>:29500` (1 peer)
- **TP=3:** `<peer-ip>:29500` through `<peer-ip>:29500` (2 peers)
- **TP=4:** `<peer-ip>:29500` through `<peer-ip>:29500` (3 peers)
- **TP=6:** `<peer-ip>:29500` through `<peer-ip>:29500` (5 peers)
- **TP=8:** `<peer-ip>:29500` through `<peer-ip>:29500` (7 peers)

---

## 3. Bring up the server on the head

Replace the IP addresses and `--model-dir` path with your own. On **TP=3, TP=4, TP=6, and TP=8**, Velo-Fabric automatically engages **DeepSpeed-Ulysses** 1-RTT dual-rail RDMA All-to-All sequence-parallel attention with localized `gate_attn` (bit-exact FP32 by default).

### Single machine (TP=1)

```bash
~/stable_build/gb10_inference --server --port 9000 \
  --model-dir /models/thinkingmachines/Inkling-2-Mini \
  --max-batch 8 --max-seq-len 32768 --mtp on \
  --inkling2-expert-act native --prefix-cache on --thermal-guard off
```

### TP=2 (1 node + head — fused `grid.y=2` dual-rail AG+RS)

```bash
~/stable_build/gb10_inference --server --port 9000 \
  --model-dir /models/thinkingmachines/Inkling-2-Mini \
  --max-batch 8 --max-seq-len 32768 --mtp on \
  --inkling2-expert-act native --prefix-cache on --tp 2 \
  --nodes <peer-ip>:29500 \
  --thermal-guard off
```

### TP=3 (2 nodes + head — arbitrary-W Ulysses SP, exact `[384 × 3]` MoE)

```bash
~/stable_build/gb10_inference --server --port 9000 \
  --model-dir /models/thinkingmachines/Inkling-2-Mini \
  --max-batch 8 --max-seq-len 32768 --mtp on \
  --inkling2-expert-act native --prefix-cache on --tp 3 \
  --nodes <peer-ip>:29500,<peer-ip>:29500 \
  --thermal-guard off
```

### TP=4 (3 nodes + head — Ulysses SP)

```bash
~/stable_build/gb10_inference --server --port 9000 \
  --model-dir /models/thinkingmachines/Inkling-2-Mini \
  --max-batch 8 --max-seq-len 32768 --mtp on \
  --inkling2-expert-act native --prefix-cache on --tp 4 \
  --nodes <peer-ip>:29500,<peer-ip>:29500,<peer-ip>:29500 \
  --thermal-guard off
```

### TP=6 (5 nodes + head — fastest cold prefill, exact `[192 × 6]` MoE)

```bash
~/stable_build/gb10_inference --server --port 9000 \
  --model-dir /models/thinkingmachines/Inkling-2-Mini \
  --max-batch 8 --max-seq-len 32768 --mtp on \
  --inkling2-expert-act native --prefix-cache on --tp 6 \
  --nodes <peer-ip>:29500,<peer-ip>:29500,<peer-ip>:29500,<peer-ip>:29500,<peer-ip>:29500 \
  --thermal-guard off
```

### TP=8 (7 nodes + head — fastest 1-user decode & 4-user batch)

```bash
~/stable_build/gb10_inference --server --port 9000 \
  --model-dir /models/thinkingmachines/Inkling-2-Mini \
  --max-batch 8 --max-seq-len 32768 --mtp on \
  --inkling2-expert-act native --prefix-cache on --tp 8 \
  --nodes <peer-ip>:29500,<peer-ip>:29500,<peer-ip>:29500,<peer-ip>:29500,<peer-ip>:29500,<peer-ip>:29500,<peer-ip>:29500 \
  --thermal-guard off
```

> **Optional fast prefill wire (`--inkling2-ulysses-wire bf16`):** Add `--inkling2-ulysses-wire bf16` on **TP=3, TP=4, TP=6, or TP=8** to compress Q/K/V and ungated attention outputs to BF16 across the Ulysses All-to-All (while keeping `r`/`s`/`kd`/`vd` in FP32). This cuts Ulysses wire by **47%**, boosts prefill by **+7–12%** (up to **5,086 tok/s** at TP=6), and remains 100% deterministic across ranks.
>
> **Full 256K context:** Raise `--max-seq-len` up to `262144` (the model's trained window) whenever needed; adjust `--max-batch` accordingly for unified-memory headroom.

### What you should see

```text
gb10_inference v0.7.10 (veloGB10) — commit 71385304, kernel build 51a214cddc5395e4, source build 06155bd058b6c9af
configuration:
  mode           server (OpenAI-compatible HTTP API)
  model          /models/thinkingmachines/Inkling-2-Mini
  topology       TP=6, this box is the head; nodes: <peer-ip>:29500, ...
  ...
[boot] 2026-11-24 21:00:02 notice: --inkling2-expert-act native: expert activations quantised as the pack declares (NVFP4 'four over six' L0-19, MXFP8 L20-23) on the fp4/fp8 tensor cores; ...
[boot] 2026-11-24 21:00:08 notice: inkling2 TP=6: ...
SERVING family=inkling2 model=Inkling-2-Mini tp=6 ctx=32768 batch=8 kv=bf16 prefix=on mtp=on spec=mtp port=9000 url=http://0.0.0.0:9000 boot=13.0s
```

Once `SERVING ...` appears, connect any OpenAI-compatible client to `http://<head-ip>:9000/v1`:

```bash
curl -s http://127.0.0.1:9000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Inkling-2-Mini","temperature":0,"max_tokens":64,"messages":[{"role":"user","content":"What is 17 * 23? Answer with the number only."}]}'
```

---

## 4. Performance

Measured live on DGX Spark (GB10) nodes over dual-port 200GbE ConnectX-7 RoCEv2 via the OpenAI-compatible HTTP API (`/v1/chat/completions`, streaming, `--mtp on`):

| Topology | 4.87K–5.5K Cold SP Prefill (`tok/s`, FP32 / BF16 wire) | 16K Cold SP Prefill (`tok/s`, FP32 / BF16 wire) | Turn-2 Prefix Hit TTFT | 1-User Decode (`tok/s`) | 4-User Batch (`agg tok/s` / per-user) |
| --- | --- | --- | --- | --- | --- |
| **TP=1** (1 Spark) | `2,429` (`2.01 s`) | — | — | `43.0` | `59.7` (`14.9`/u) |
| **TP=2** (fused dual-rail AG+RS) | **`3,759 – 3,766`** (`1.45 s`) | — | — | `62.6` (+46%) | `83.2` (`20.8`/u) |
| **TP=3** (Ulysses + U1/U7a) | **`3,781 – 3,810`** / **`4,133 – 4,178`** | — | `112.5 ms` | `62.4` (+45%) | `167.2` (`41.8`/u) |
| **TP=4** (Ulysses + U1) | **`4,287 – 4,330`** (`1.27 s`) | **`4,254 – 4,285`** (`3.71 s`) | **`73.0 ms`** | **`64.7 – 76.1`** | `175.7` (`43.9`/u) |
| **TP=6** (Ulysses + U1/U7a) | **`4,617 – 4,637`** / **`5,008 – 5,086`** | **`4,450 – 4,475`** / **`4,881 – 4,887`** | `108.9 ms` | `70.6` (+64%) | `193.8` (`48.5`/u) |
| **TP=8** (Ulysses + U1/U7a) | **`4,436 – 4,496`** / **`4,738 – 4,805`** | **`4,478 – 4,505`** / **`4,723 – 4,737`** | **`70.0 ms`** | **`64.3 – 89.1`** (`2.07×` TP1) | **`194.0`** (`48.5`/u, `+225%`) |
| **TP=8** (`--mtp off`) | `3,603` *(pre-Ulysses)* | — | `105.0 ms` | `85.7` (`1.99×` TP1) | `120.9` (`30.2`/u) |

*Single-user speculative decode speed varies with prompt MTP draft acceptance (`64.3 tok/s` on dense low-acceptance technical prose up to `89.1 tok/s` on general prose/code at TP=8).*

### Attach-time RDMA all-reduce latency (dual-rail RoCEv2, FP32)

| Topology | 1-Row p50 (`9.2 KB`) | 4-Row p50 (`36.9 KB`) | 8-Row p50 (`73.7 KB`) | 96 Collectives / Token (1-Row) |
| --- | --- | --- | --- | --- |
| **TP=2** | **`15.9 – 16.2 µs`** | **`18.8 – 19.0 µs`** | **`22.6 – 22.9 µs`** | `1.53 – 1.55 ms` |
| **TP=3** | **`21.5 – 21.9 µs`** | **`25.4 – 25.5 µs`** | **`32.6 – 33.3 µs`** | `2.07 – 2.10 ms` |
| **TP=4** | **`22.8 µs`** | **`30.0 µs`** | **`38.9 µs`** | `2.19 ms` |
| **TP=6** | **`28.5 – 28.6 µs`** | **`36.9 µs`** | **`54.1 – 54.5 µs`** | `2.73 – 2.75 ms` |
| **TP=8** | **`31.4 – 32.3 µs`** | **`44.2 – 44.7 µs`** | **`65.1 – 68.9 µs`** | `3.01 – 3.10 ms` |

See [docs/VELO_FABRIC_BENCHMARKS.md](docs/VELO_FABRIC_BENCHMARKS.md) and [VELO_FABRIC.md](VELO_FABRIC.md) for full details.

---

## 5. Topology Sharding & Operational Notes

### Sharding geometry

| Topology | Q Heads / KV Heads per Rank | MoE Inner Channels per Rank (64-ch aligned) | Attention SP Mode | Vocab-Parallel MTP |
| --- | --- | --- | --- | --- |
| **TP=1** | `24` / `8` | `1152` | Single-node | Local |
| **TP=2** | `[12, 12]` / `[4, 4]` | `[576, 576]` (exact) | SP Prefill (fused dual-rail AG+RS) | Yes (`≤ 1 KiB` accept) |
| **TP=3** | `[9, 9, 6]` / `[3, 3, 2]` | `[384, 384, 384]` (exact) | **DeepSpeed-Ulysses** (arbitrary-W 1-RTT A2A + local gate) | Yes (`≤ 1 KiB` accept) |
| **TP=4** | `[6 × 4]` / `[2 × 4]` | `[320, 320, 256, 256]` | **DeepSpeed-Ulysses** (1-RTT A2A + local gate) | Yes (`≤ 1 KiB` accept) |
| **TP=6** | `[6, 6, 3, 3, 3, 3]` / `[2, 2, 1, 1, 1, 1]` | `[192 × 6]` (exact) | **DeepSpeed-Ulysses** (arbitrary-W 1-RTT A2A + local gate) | Yes (`≤ 1 KiB` accept) |
| **TP=8** | `[3 × 8]` / `[1 × 8]` | `[192 × 4, 128 × 4]` | **DeepSpeed-Ulysses** (1-RTT A2A + local gate) | Yes (`≤ 1 KiB` accept) |

### Key flags & notes

- **Expert activation precision (`--inkling2-expert-act a16|native`):** Default is `a16` (16-bit expert activations, matching the `transformers` reference). Passing `--inkling2-expert-act native` runs the pack-declared NVFP4 / MXFP8 activations on Blackwell FP4/FP8 tensor cores for significantly higher throughput (exported on `/metrics` as `velogb10_inkling2_expert_act_native`).
- **Ulysses wire compression (`--inkling2-ulysses-wire fp32|bf16`):** Default is `fp32` (bit-exact across TP1/TP2/TP3/TP4/TP6/TP8). Passing `--inkling2-ulysses-wire bf16` cuts Ulysses A2A wire by 47% for +7–12% faster prefill on TP=3/4/6/8 (exported on `/metrics` as `velogb10_inkling2_ulysses_wire_bf16`). TP=2 stays on fused dual-rail AG+RS intentionally because AG+RS moves fewer bytes at `W = 2`.
- **MTP speculation (`--mtp auto|on|off`):** Uses the bundled `mtp.safetensors` drafter (up to 7 drafts, tunable via `--inkling2-mtp-max-drafts N`). Lossless for greedy decoding and distribution-exact for sampled requests (including `top_p` / `top_k`). Pass `--mtp off` for plain decoding.
- **Reasoning effort & tool calls:** Supports per-request `reasoning_effort` (named or numeric, or default `--reasoning-effort`) and OpenAI tool calling with hardened catalogue validation.
- **Concurrency (`--max-batch`):** Supports `1..=16` concurrent lanes (default `4`). Prefill chunks interleave cleanly with active decode lanes so running streams stay responsive.
- **Thermal pacing (`--thermal-guard`):** Default is `on` (paces prefill above 85 °C). The commands above pass `--thermal-guard off` for maximum unthrottled throughput in well-cooled setups; omit it if your boxes run warm.
- **Terminal output (`--color` & `--log-level`):** `--color auto` (default) enables ANSI colors on interactive TTYs and stays plain when piped/redirected (`always` / `never` override). Every `[tag]` log line includes a local `YYYY-MM-DD HH:MM:SS` timestamp.
- **Live telemetry (`/metrics`):**
  ```bash
  curl -s http://127.0.0.1:9000/metrics | grep -E "velo_fabric|inkling2_(tp|spec|expert_act|ulysses_wire|prefill|decode)"
  ```
- **No environment variables:** veloGB10 reads zero `GB10_*` environment variables; all flags are passed on the head command line and shipped automatically to every peer node.

---

## 6. Credits & References

- **Model:** Thinking Machines Lab ([`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini)).
- **DeepSpeed-Ulysses Sequence-Parallel Attention:** Sam Ade Jacobs et al. (2023), *"DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models"*, [arXiv:2309.14509](https://arxiv.org/abs/2309.14509).
- **Velo-Fabric Architecture & Full Benchmarks:** [VELO_FABRIC.md](VELO_FABRIC.md), [docs/VELO_FABRIC_BENCHMARKS.md](docs/VELO_FABRIC_BENCHMARKS.md).
