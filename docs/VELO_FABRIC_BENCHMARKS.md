# Velo-Fabric Multi-Node Performance Benchmarks (DGX Spark / GB10)

This document collects live, end-to-end serving performance and RDMA collective micro-benchmarks for **veloGB10** running on **Velo-Fabric** across **1, 2, 3, 4, 6, and 8 NVIDIA DGX Spark (GB10)** nodes connected over dual-port 200GbE RoCEv2 Ethernet—with **no NCCL, no MPI, and no host-sync breaks inside the decode loop**.

It covers the two flagship MoE architectures running on Velo-Fabric:
1. **DeepSeek-V4.1-Flash** (`EXL3 2.9 bpw` + native FP8 Engram tables + 3-stage DSpark speculation) — **TP2, TP3, TP4, TP6, TP8**
2. **Inkling-2-Mini** (`NVFP4` looped MoE + 3-layer MTP speculation) — **TP1, TP2, TP3, TP4, TP6, TP8**

> **Note on topology coverage:** DeepSeek-V4.1-Flash (`210.6 GB` EXL3 weights + `~200 GB` native Engram tables) exceeds a single 128 GB DGX Spark and therefore starts at **TP2**. All five multi-node topologies (**TP2, TP3, TP4, TP6, TP8**) are live-verified on dual-rail RoCEv2 for both models, including arbitrary-W DeepSpeed-Ulysses (`TP3/TP4/TP6/TP8`) on Inkling-2-Mini and P7SP sequence-parallel prefill (`TP2/TP3/TP4/TP6/TP8`) + LAGA BF16 output All-to-All (`TP3/TP4/TP6/TP8`, plus `TP2` with `--dsv41-attn-whole on`) on DeepSeek-V4.1-Flash.

---

## 1. Testbed & Methodology

| Parameter | Configuration |
| --- | --- |
| **Compute Node** | NVIDIA DGX Spark (GB10 — Blackwell GPU + ARM Grace CPU, 128 GB unified coherent LPDDR5x, ~273 GB/s/node) |
| **Interconnect** | Dual-port 200GbE ConnectX-7 RoCEv2 (`rocep1s0f1` + `roceP2p1s0f1`, auto-detected by Velo-Fabric) |
| **Engine Build** | `veloGB10` (`gb10_inference v0.7.10`, `master` merge `ea650efd`, kernel build `51a214cddc5395e4`) |
| **Transport** | Velo-Fabric PDL zero-CPU-bounce RDMA (`GPU-validated direct NIC tail`, dual-rail split ≥ 16 KiB, canonical FP32 reduction, CUDA Graph capture) |
| **Serving Interface** | Production OpenAI-compatible HTTP API (`/v1/chat/completions`, `stream: true`, port `9000`) |
| **Reliability Gate** | Every run verified `0` TP aborts (`velogb10_dsv41_tp_aborts_total 0`), `0` fatal errors, and exact greedy math verification (`17 * 23 = 391`) |

### Benchmark Workloads
- **1-User Decode (`128 tok`):** Streaming greedy generation (`temperature: 0.0`, `max_tokens: 128`); decode speed is `(completion_tokens - 1) / (total_time - ttft)`.
- **Cold Prefill (`4.5K–5.5K` and `16K` tokens):** Fresh prompt with cold prefix cache; throughput is `prompt_tokens / ttft`.
- **Turn-2 Prefix-Cache Hit:** Follow-up turn appending a short question to the exact long prompt + assistant turn; reports TTFT in milliseconds.
- **4-User Concurrent Batch (`4 × 128 tok`):** Four simultaneous streaming requests over `ThreadPoolExecutor(4)`; reports wall-clock aggregate throughput (`total_completion_tokens / wall_s`) and per-user speed (`agg / 4`).

---

## 2. DeepSeek-V4.1-Flash (EXL3 2.9 bpw + Native FP8 Engram + DSpark)

- **Architecture:** 40 trunk layers + 3 DSpark speculative stages (block 5, Markov rank 256), hidden `5120`, `64` Q heads / `8` output groups (`head_dim 512`), `384` routed experts top-6 + `1` shared expert (`moe_inter 2304` = `18` EXL3 128-blocks), `8`-head Engram tables on layers `[1, 14]`, vocab `129,280`, `1,048,576` trained context, ViT 32L + 3×3 vision aligner on rank 0.
- **Checkpoint:** [doth4580/DeepSeek-V4.1-Flash-1017-Mia-EXL3-2.9bpw-Engram](https://huggingface.co/doth4580/DeepSeek-V4.1-Flash-1017-Mia-EXL3-2.9bpw-Engram) (`210.6 GB` EXL3 2.9 bpw weights quantized by **Mia-AiLab** + native FP8 Engram tables).

### 2.1 Topology & Sharding Geometry

| Topology | Nodes | Q Heads / Output Groups per Rank | MoE 128-Blocks per Rank | Engram Heads per Rank | Vocab Slice per Rank | Weights / Rank (Rank 0) | Prefix Cache |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **TP2** | 2 | `[32, 32]` / `[4, 4]` | `[9, 9]` (exact) | `[0..4, 4..8]` | `[64640, 64640]` | `~105 GiB` | Off *(RAM-bound)* |
| **TP3** | 3 | `[24, 24, 16]` / `[3, 3, 2]` | `[6, 6, 6]` (exact) | `[0..2, 2..5, 5..8]` | `[43136, 43136, 43008]` | `69.6 GiB` | On (16 GiB cap) |
| **TP4** | 4 | `[16 × 4]` / `[2 × 4]` | `[5, 4, 5, 4]` | `[0..2, 2..4, 4..6, 6..8]` | `[32384, 32384, 32256, 32256]` | `~53 GiB` | On (16 GiB cap) |
| **TP6** | 6 | `[16, 16, 8, 8, 8, 8]` / `[2, 2, 1, 1, 1, 1]` | `[3 × 6]` (exact) | `[0..1, 1..2, 2..4, 4..5, 5..6, 6..8]` | `21,504–21,632` | `~38 GiB` | On (full) |
| **TP8** | 8 | `[8 × 8]` / `[1 × 8]` | `[3, 2, 2, 2, 3, 2, 2, 2]` | `[0..1, 1..2, ..., 7..8]` | `16,128–16,256` | `30.1 GiB` | On (full) |

### 2.2 End-to-End Serving Performance (TP2 / TP3 / TP4 / TP6 / TP8)

| Topology | Mode | Warm Boot | 4.4K–5.2K Cold Prefill (`tok/s` / TTFT) | 8.0K–14.9K Cold Prefill (`tok/s` / TTFT) | Turn-2 Prefix Hit TTFT | 1-User Decode (`tok/s`) | 4-User Batch (`agg tok/s` / per-user) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **TP2** (2 Sparks) | FP32 (default) | `129 s` | `1,886` (`2.439 s` @ 4.6K) | — | *(cache off)* | `48.9` | `91.2` (`22.8`/u) |
| **TP3** (3 Sparks) | FP32 + **P7SP + LAGA** (default) | `152 s` | **`2,261 – 2,275`** (`1.94 s` @ 4.4K, **+20.9% vs pre-SP/LAGA**) | **`2,518 – 2,547`** (`3.14 s` @ 8.0K) | **`157 – 220 ms`** | **`64.4`** (+32% vs TP2) | **`146.6`** (`36.7`/u, +61% vs TP2) |
| **TP3** (3 Sparks) | BF16 (`--dsv41-wire-bf16 on`) | `152 s` | `1,794 – 1,890` *(pre-SP/LAGA)* | **`2,177`** (`6.840 s` @ 14.9K) | **`113 – 225 ms`** (`124 ms` @ 14.9K) | **`68.1 – 70.6`** (+39–44% vs TP2) | **`145.0 – 183.7`** (`36.3–45.9`/u)¹ |
| **TP4** (4 Sparks) | FP32 + **P7SP + LAGA** (default) | `72 s` | **`2,764`** (`1.664 s` @ 4.6K) | **`3,169 – 3,231`** (@ 8.0K, **−19.3% wire**) | `142 ms` (4.63K) | `58.5` (+20% vs TP2) | `135.1` (`33.8`/u, +48% vs TP2) |
| **TP6** (6 Sparks) | FP32 + **P7SP** *(pre-LAGA)* | `~95 s` | `2,147` (@ 4.4K) | **`2,873`** (@ 8.0K) | — | **`70.5 – 71.6`** (+46% vs TP2) | — |
| **TP8** (8 Sparks) | FP32 + **P7SP + LAGA** (default) | **`61 s`** | **`3,094`** (`1.467 s` @ 4.54K, +64% vs TP2) | — | **`155 ms`** (4.57K) | **`87.6`** (+79% vs TP2)¹ᵃ | **`219.9`** (`55.0`/u, **+141% vs TP2**) |

#### LAGA BF16 Attention-Output All-to-All A/B (`--dsv41-sp-laga auto` vs `off`)
Replacing FP32 `wo_b` reduce-scatter (`20 KiB/tok`) with a 1-RTT BF16 All-to-All on `wo_a` output (`16 KiB/tok`, generalized to uneven `o_groups` on TP3/TP6 via `D41LagaGroups`) + local whole `wo_b` cuts total collective wire across the entire forward pass:

| Topology | `velogb10_dsv41_tp_bytes_total` (LAGA OFF → ON) | Wire Delta | Cold Prefill (OFF → ON) | LAGA Layers / Aborts |
| --- | --- | --- | --- | --- |
| **TP3** (`.11–.13`) | `30.31 GB` → **`24.60 GB`** | **−18.8%** | 4.4K: `2,163–2,169` → **`2,261–2,275`** (**+4.7%**); 8.0K: `2,368–2,410` → **`2,518–2,547`** (**+6.0%**) | `285` / `0` (`61/61` agrees) |
| **TP4** (`.11–.14`) | `30.35 GB` → **`24.48 GB`** | **−19.3%** | 8.0K: `2,446–3,168` → **`3,169–3,231`**; 492/492 char-exact greedy parity | `80` / `0` (`89/89` agrees) |

¹ *On TP3 with `--dsv41-wire-bf16 on`, the first cold-policy 4-user batch measured `145.0 agg tok/s` (`36.3 tok/s`/u); after several rounds of online DSpark acceptance/cost calibration and warm CUDA graphs, sustained 4-user batch throughput climbed to `166.1 – 183.7 agg tok/s` (`41.5 – 45.9 tok/s`/u) and 1-user decode reached `70.1 – 70.6 tok/s`.*
¹ᵃ *Latest TP8 run (`FP32` wire, RoCEv2 technical prompt): `87.6 tok/s` 1-user decode (`109.9 ms` TTFT), `3,094.2 tok/s` 4.54K prefill, `155.4 ms` turn-2 prefix hit, `219.9 agg tok/s` 4-user batch (`0` aborts, `0` fatals, `531` CUDA graph replays). An earlier general-prose TP8 run reached `92.6 tok/s` 1-user decode and `108 ms` turn-2 prefix hit.*

#### Key Takeaways on DeepSeek-V4.1-Flash
- **TP3 is a sweet spot for 3-node clusters:** Because `moe_inter = 2304` divides cleanly into `[6, 6, 6]` EXL3 128-blocks and `main_proj` divides cleanly into `[5120, 5120, 5120]`, TP3 has zero MoE straggler imbalance, drops per-rank weights to `69.6 GiB` (unlocking prefix caching and 1M context headroom that TP2 lacks), and delivers **`64.4 – 70.6 tok/s` single-user** and **`146.6 – 183.7 agg tok/s` 4-user**.
- **TP8 approaches 90 tok/s single-user over pure Ethernet:** Cutting per-rank weights to `30.1 GiB` brings warm boot down to **`61 s`**, 4.54K cold prefill to **`3,094 tok/s`** (`1.467 s`), 1-user decode to **`87.6 tok/s`** (up to `92.6 tok/s` on high-acceptance prose), and 4-user batch to **`219.9 agg tok/s`** (`55.0 tok/s`/user).
- **Prefix caching:** Turn-2 prefix-cache hits cut TTFT by **10–20×** across TP3, TP4, and TP8 (`2.41 s → 220 ms` at TP3; `6.84 s → 124 ms` at 14.9K on TP3; `1.47 s → 155 ms` on latest TP8).
- **FP32 wire vs `--dsv41-wire-bf16 on`:** FP32 wire remains the default for exact reference numerics (especially on the 25-float mHC `mix` rider) and faster prefill (`1,882` vs `1,794 tok/s` at TP3). Opt-in `--dsv41-wire-bf16 on` trades a slight BF16 rounding of partial sums (still bit-identical across all ranks) for **+5.7% single-user decode** at TP3 (`64.4 → 68.1 tok/s`).

### 2.3 DeepSeek-V4.1-Flash TP3 Collective Latency (`--probe-dsv41-tp`, FP32 Wire, Dual-Rail)

Measured via GPU events on `.11`–`.13` (`DSV41_TP_SMOKE_OK`, `0` mismatching floats across all 348 checks, bitwise-equal head exchanges and CUDA graph replays):

| Verify Rows (Lanes) | Layer Reduce (`hidden + mix`, p50 / p99) | Layer Reduce + Engram `wkv` (p50) | Head Exchange (p50) | 81 Collectives Total (p50) | All 94 Collectives in 1 CUDA Graph (p50) |
| --- | --- | --- | --- | --- | --- |
| **1** (1 lane) | `26.5 / 45.2 µs` (20.6 KB) | `43.1 µs` (123 KB) | `20.5 µs` | `2.12 ms` | **`2.19 ms`** |
| **6** (1 lane, k=5) | `45.1 / 73.5 µs` (124 KB) | `106.5 µs` (738 KB) | `22.5 µs` | `3.76 ms` | **`3.89 ms`** |
| **12** (2 lanes) | `67.7 / 90.0 µs` (247 KB) | `198.7 µs` (1.48 MB) | `28.6 µs` | `5.79 ms` | **`5.87 ms`** |
| **24** (4 lanes) | `96.2 / 127.3 µs` (495 KB) | `350.2 µs` (2.95 MB) | `30.7 µs` | `8.22 ms` | **`8.54 ms`** |

---

## 3. Inkling-2-Mini (NVFP4 Looped MoE + MTP Speculation)

- **Architecture:** 24 layers (48 looped visits), `128` routed experts top-8 (`moe_inter 1152`), hidden `2304`, `24` Q heads / `8` KV heads, vocab `201,024`, 3-layer MTP speculative drafter, **`96` all-reduces per generated token**.
- Because per-layer compute is compact and collective frequency is very high (`96`/step), Inkling-2-Mini is a demanding stress test for Ethernet fabric latency and unsharded Amdahl's-Law bottlenecks.

### 3.1 Topology & Sharding Geometry

| Topology | Nodes | Q Heads / KV Heads per Rank | MoE Inner Channels per Rank (64-ch aligned) | Attention SP Mode | Vocab-Parallel MTP |
| --- | --- | --- | --- | --- | --- |
| **TP1** | 1 | `24` / `8` | `1152` | Single-node | Local |
| **TP2** | 2 | `[12, 12]` / `[4, 4]` | `[576, 576]` (exact) | SP Prefill (fused dual-rail AG+RS) | Yes (`≤ 1 KiB` accept) |
| **TP3** | 3 | `[9, 9, 6]` / `[3, 3, 2]` | `[384, 384, 384]` (exact) | **DeepSpeed-Ulysses** (arbitrary-W 1-RTT A2A + local gate) | Yes (`≤ 1 KiB` accept) |
| **TP4** | 4 | `[6 × 4]` / `[2 × 4]` | `[320, 320, 256, 256]` | **DeepSpeed-Ulysses** (1-RTT A2A + local gate) | Yes (`≤ 1 KiB` accept) |
| **TP6** | 6 | `[6, 6, 3, 3, 3, 3]` / `[2, 2, 1, 1, 1, 1]` | `[192 × 6]` (exact) | **DeepSpeed-Ulysses** (arbitrary-W 1-RTT A2A + local gate) | Yes (`≤ 1 KiB` accept) |
| **TP8** | 8 | `[3 × 8]` / `[1 × 8]` | `[192 × 4, 128 × 4]` | **DeepSpeed-Ulysses** (1-RTT A2A + local gate) | Yes (`≤ 1 KiB` accept) |

### 3.2 End-to-End Serving Performance (TP1 / TP2 / TP3 / TP4 / TP6 / TP8)

All multi-node runs use `--mtp on` (default `a16` or `--inkling2-expert-act native` where noted; FP32 Ulysses wire is default, `--inkling2-ulysses-wire bf16` is opt-in).

| Topology | 4.87K–5.5K Cold SP Prefill (`tok/s`, FP32 / BF16 wire) | 16K Cold SP Prefill (`tok/s`, FP32 / BF16 wire) | Turn-2 Prefix Hit TTFT | 1-User Decode (`tok/s`) | 4-User Batch (`agg tok/s` / per-user) |
| --- | --- | --- | --- | --- | --- |
| **TP1** (1 Spark, `--mtp on`) | `2,429` (`2.005 s` @ 4.87K) | — | — | `43.0` | `59.7` (`14.9`/u) |
| **TP2** (2 Sparks, fused dual-rail AG+RS) | **`3,759 – 3,766`** (`1.45 s` @ 5.5K, +55%) | — | — | `62.6` (+46% vs TP1) | `83.2` (`20.8`/u, +39% vs TP1) |
| **TP3** (3 Sparks, **Ulysses + U1/U7a**) | **`3,781 – 3,810`** / **`4,133 – 4,178`** (+72% BF16) | — | `112.5 ms` | `62.4` (+45% vs TP1) | `167.2` (`41.8`/u, **+180% vs TP1**) |
| **TP4** (4 Sparks, **Ulysses + U1**) | **`4,287 – 4,330`** (`1.27 s` @ 5.5K, +78%) | **`4,254 – 4,285`** (`3.71 s` @ 15.9K) | **`73.0 ms`** | **`64.7 – 76.1`** (+50–77% vs TP1)² | `175.7` (`43.9`/u, **+194% vs TP1**) |
| **TP6** (6 Sparks, **Ulysses + U1/U7a**) | **`4,617 – 4,637`** / **`5,008 – 5,086`** (**+109% vs TP1**) | **`4,450 – 4,475`** / **`4,881 – 4,887`** | `108.9 ms` | `70.6` (+64% vs TP1) | `193.8` (`48.5`/u, **+225% vs TP1**) |
| **TP8** (8 Sparks, **Ulysses + U1/U7a**) | **`4,436 – 4,496`** / **`4,738 – 4,805`** (**+98% vs TP1**) | **`4,478 – 4,505`** / **`4,723 – 4,737`** | **`70.0 ms`** | **`64.3 / 89.1`** (**2.07× TP1**)² | **`194.0`** (`48.5`/u, **+225% vs TP1**) |
| **TP8** (8 Sparks, `--mtp off`) | `3,603` (`1.352 s`, pre-Ulysses) | — | `105.0 ms` | `85.7` (1.99× TP1) | `120.9` (`30.2`/u, +103% vs TP1) |

² *Single-user speculative decode speed varies with MTP draft acceptance on the prompt: in the dense RoCEv2 technical prompt (`30.1%` draft acceptance at TP8), 1-user decode measured `64.7 tok/s` (TP4) and `64.3 tok/s` (TP8, `90.0 ms` TTFT); on general prose/code generation with higher MTP acceptance, the same TP4 and TP8 builds measured `76.1 tok/s` (TP4) and `89.1 tok/s` (TP8, `2.07×` TP1). Greedy outputs across TP2/TP3/TP6 FP32 verified 409/409 char-exact.*

### 3.3 DeepSpeed-Ulysses Sequence-Parallel Attention Gains (TP3, TP4, TP6, & TP8)

By replacing the pre-attention full-hidden all-gather (`xb`) and post-attention reduce-scatter (`ao`) with a **1-RTT dual-rail RDMA All-to-All (`TP_LEN_FLAG_A2A`)** head/sequence transpose (adapted from DeepSpeed-Ulysses, Jacobs et al., [arXiv:2309.14509](https://arxiv.org/abs/2309.14509)), localizing `gate_attn` post-A2A #2 (**−35.4% A2A #1 wire, bit-exact**), generalizing to uneven heads via `InkUlyHeads` (`TP3/TP6`), and offering opt-in BF16 wire compression (`--inkling2-ulysses-wire bf16`, **−47% Ulysses wire**):

| Topology | Pre-Ulysses Cold SP Prefill (5.5K) | Ulysses + U1 FP32 (5.5K / 15.9K, bit-exact) | Ulysses + U1 + U7a BF16 (5.5K / 15.9K) | Gain vs Pre-Ulysses |
| --- | --- | --- | --- | --- |
| **TP3** | `3,707 tok/s` | **`3,781 – 3,810`** / — | **`4,133 – 4,178`** / — | **+2.4% FP32 / +12.1% BF16** |
| **TP4** | `~3,650 tok/s` | **`4,287 – 4,330`** / **`4,254 – 4,285`** | — | **+18.0% FP32** |
| **TP6** | `4,078` / `4,034 tok/s` | **`4,617 – 4,637`** / **`4,450 – 4,475`** | **`5,008 – 5,086`** / **`4,881 – 4,887`** | **+13.5% FP32 / +23.8% BF16** |
| **TP8** | `3,556 – 3,603 tok/s` | **`4,436 – 4,496`** / **`4,478 – 4,505`** | **`4,738 – 4,805`** / **`4,723 – 4,737`** | **+25.0% FP32 / +33.0% BF16** |

*(TP2 intentionally stays on fused dual-rail AG+RS: `3,759–3,766 tok/s` vs `3,507 tok/s` with FP32 Ulysses, since at W=2 non-Ulysses wire `6,912 B/tok` is smaller than FP32 Ulysses `8,664 B/tok`.)*

### 3.4 Inkling-2-Mini Attach-Time RDMA All-Reduce Latency (Dual-Rail RoCEv2, FP32)

| Topology | 1-Row p50 (`9.2 KB`) | 4-Row p50 (`36.9 KB`) | 8-Row p50 (`73.7 KB`) | 96 Collectives / Token (1-Row) |
| --- | --- | --- | --- | --- |
| **TP2** (2 Sparks, fused `grid.y=2` dual-rail) | **`15.9 – 16.2 µs`** | **`18.8 – 19.0 µs`** | **`22.6 – 22.9 µs`** | `1.53 – 1.55 ms` |
| **TP3** (3 Sparks) | **`21.5 – 21.9 µs`** | **`25.4 – 25.5 µs`** | **`32.6 – 33.3 µs`** | `2.07 – 2.10 ms` |
| **TP4** (4 Sparks) | **`22.8 µs`** | **`30.0 µs`** | **`38.9 µs`** | `2.19 ms` |
| **TP6** (6 Sparks) | **`28.5 – 28.6 µs`** | **`36.9 µs`** | **`54.1 – 54.5 µs`** | `2.73 – 2.75 ms` |
| **TP8** (8 Sparks) | **`31.4 – 32.3 µs`** | **`44.2 – 44.7 µs`** | **`65.1 – 68.9 µs`** | `3.01 – 3.10 ms` |

---

## 4. Reproducible Launch Commands

### Worker Nodes (run once per peer box)
```bash
gb10_inference --node --port 29500
```

### DeepSeek-V4.1-Flash (EXL3 2.9 bpw + Engram)

**TP3 (3 Sparks — recommended 3-node setup, port 9000):**
```bash
gb10_inference --server --port 9000 \
  --model-dir /path/to/DeepSeek-V4.1-Flash-1017-Mia-EXL3-2.9bpw-Engram \
  --tp 3 --nodes <node1>:29500,<node2>:29500 \
  --max-batch 4 --max-tokens 131072 --max-seq-len 1048576 \
  --prefix-cache on --thermal-guard off
# Optional: add --dsv41-wire-bf16 on for +5-6% 1-user decode speed
# LAGA BF16 attention-output All-to-All is auto-enabled on TP3/4/6/8 (and TP2 with --dsv41-attn-whole on; --dsv41-sp-laga off to disable)
```
*(If using a standalone EXL3 pack without bundled `engram_source/`, add `--engram-source /path/to/DeepSeek-V4.1-Flash-1017`. For TP2, TP4, TP6, or TP8, set `--tp 2|4|6|8` with `W-1` peer addresses.)*

### Inkling-2-Mini (NVFP4 Looped MoE)

**TP1 (1 Spark, port 9000):**
```bash
gb10_inference --server --port 9000 \
  --model-dir /path/to/Inkling-2-Mini \
  --max-batch 8 --max-seq-len 32768 --mtp on \
  --inkling2-expert-act native --prefix-cache on --thermal-guard off
```

**TP2 / TP3 / TP4 / TP6 / TP8 (multi-node, port 9000):**
```bash
gb10_inference --server --port 9000 \
  --model-dir /path/to/Inkling-2-Mini \
  --tp 6 --nodes <node1>:29500,<node2>:29500,<node3>:29500,<node4>:29500,<node5>:29500 \
  --max-batch 8 --max-seq-len 32768 --mtp on \
  --inkling2-expert-act native --prefix-cache on --thermal-guard off
# Optional: add --inkling2-ulysses-wire bf16 for -47% Ulysses wire (+7-12% prefill, up to 5,086 tok/s at TP6)
```

### Live Telemetry Check
```bash
curl -s http://127.0.0.1:9000/metrics | grep -E "velo_fabric|sp_laga|sp_prefill|aborts_total|fatal_total"
```

---

## 5. Credits & References

- **DeepSpeed-Ulysses:** Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang, Shuaiwen Leon Song, Samyam Rajbhandari, and Yuxiong He (2023), *"DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models"*, [arXiv:2309.14509](https://arxiv.org/abs/2309.14509).
- **DeepSeek-V4.1-Flash EXL3 2.9 bpw Quantization:** **Mia-AiLab** (original EXL3-EAQ-v1 2.9 bpw weights); DeepSeek-AI (base model & native FP8 Engram tables).
- **Inkling-2-Mini:** Thinking Machines.
- **Velo-Fabric Architecture Overview:** See [VELO_FABRIC.md](../VELO_FABRIC.md).
