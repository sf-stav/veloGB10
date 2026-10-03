# Qwen 3.8 27B NVFP4: an 8-hour endurance run on NVIDIA GB10

**veloGB10's Qwen 3.8 27B NVFP4 + DFlash 2 configuration, served across four NVIDIA GB10 nodes
(tensor-parallel ×4).**

## What this is

An 8-hour stability soak. The engine was left running continuously and driven with a mixed
workload. The run measured three things:

1. Does throughput drift down over hours (thermal throttling, memory growth, a slow leak)?
2. Does the server stay deterministic (same prompt in → same text out)?
3. Does anything crash, hang, or wedge overnight?

Eight contiguous hours of steady serving in a single session. The engine was left unattended and
driven the whole time.

## The rig

| Piece | What it is |
|---|---|
| Target model | **Qwen 3.8 27B NVFP4** (FULL, NVFP4-quantized) |
| Drafter | **Qwen 3.8 27B DFlash 2** (block speculator) |
| Engine | **veloGB10** |
| Hardware | **4 × NVIDIA GB10** (Grace Blackwell Superchip), tensor-parallel **TP=4**, unified LPDDR5X |
| Decoding | **DFlash 2** speculative. A block is drafted, the trunk verifies it, and the longest matching prefix is accepted |

The configuration is the same one described in the README's *[Qwen 3.8 27B NVFP4 with
DFlash 2](README.md#qwen-38-27b-nvfp4-with-dflash-2)* section, at a 256K-capable context
headroom.

## What we ran

For **8 contiguous hours**, the engine served a rotating mix of five content types (a code task, two
prose/technical-narrative prompts, a short factual question, and a short descriptive prompt):

| Content | Prompt shape | Character |
|---|---|---|
| **code** | Build an LRUCache, O(1) get/put | dense code generation |
| **essay** | History of computing, Babbage → GPUs | long prose |
| **synth1** | Energy ↔ entropy across scales | technical narrative |
| **chat_t0** | Stack vs. queue | short factual |
| **chat_t1_off** | A quiet morning in a mountain village | short descriptive |

The batch mixer cycles through these five content types, so the run accumulates a mix of content
types rather than a single repeated workload. Every request records **client throughput**
(tokens/sec) and **time-to-first-token**. Hardware telemetry (GPU temperature, SM clock, power draw)
is sampled on **all four nodes every 5 minutes**. Once an hour, a seed-fixed, temperature-0 request
is served **twice**; the two completions are hashed and compared. This is a determinism canary.

## The results

### Throughput over 8 hours

Per-content-type mean throughput (tokens/sec), first half of the run vs. last half:

| Content | First half | Last half | Drift |
|---|---:|---:|---:|
| code | 97.74 | 97.59 | −0.2% |
| essay | 48.88 | 48.80 | −0.2% |
| synth1 | 49.07 | 49.00 | −0.1% |
| chat_t0 | 73.76 | 73.65 | −0.1% |
| chat_t1_off | 38.13 | 38.10 | −0.1% |

The largest measured drift is **0.2%**. Throughput at hour 8 matches hour 1. The content-type spread
(code at ~98, prose at ~49) reflects the compression of code relative to prose.

### Time-to-first-token

Mean **202.6 ms** across all ~2,400 requests. The first-half mean and last-half mean differ by
**+0.0 ms** drift; the first-hour spread is **σ = 9.4 ms**. No latency change across eight hours.

### Determinism

Every hourly canary (a seed-fixed, greedy request served twice and hash-compared) returned
**byte-for-byte identical**. Eight checks, eight matches, zero failures.

### Thermals

Mean GPU temperature per node over the run (min..max, mean):

| Node | Min | Max | Mean |
|---|---:|---:|---:|
| 1 | 64 | 77 | 73.7 |
| 2 | 58 | 69 | 65.4 |
| 3 | 59 | 70 | 68.8 |
| 4 | 60 | 72 | 71.4 |

All four nodes held moderate temperatures; the head node (node 1) ran warmest. SM clocks and power
draw were stable throughout. The thermal profile is a sustained GB10 load without a thermal runaway
or a throttling cliff.

### Reliability

Over eight hours and ~2,400 requests: **no crashes, no hangs, no wedges, no kill-condition trip, no
watchdog firing**.

## Summary

- **No performance drift.** Throughput and TTFT held flat to within a couple of tenths of a percent
  over eight hours.
- **No nondeterminism.** A seed-fixed prompt produced the same output at hour 1 and at hour 8.
- **No thermal problems.** The GB10s held a steady, moderate temperature profile under sustained
  load; power and clocks were stable.
- **No reliability incidents.** Zero crashes or hangs.

Two caveats, stated so the data is read correctly:

1. **A short warm-up transient.** The drafter's per-verify acceptance (tokens produced per
   speculative step) rose from ~2.9 to a stable ~3.5 during roughly the first half-hour, then held
   flat. This is a warm-up, not a drift; the client-visible throughput curve was flat the whole
   time, so a first-half vs. last-half comparison is not a drift signal.
2. **`nvidia-smi` reported *N/A* for the "memory used" field on these nodes** in the sampling path.
   This is a GB10 unified-memory reporting behavior. The memory-monotonic watch was effectively a
   no-op this run. The **thermal / clock / power** curves are the ones that carried the drift
   signal, and they were clean. "No memory leak" is not independently measured here; re-check if
   memory growth is a specific concern.

## Bottom line

After **eight unattended hours** and ~2,400 served requests across five content types, the
Qwen 3.8 27B NVFP4 + DFlash 2 stack on 4× GB10 showed **no throughput drift, no determinism
failures, and no crashes**.

---

*Everything above is measured on the engine's live telemetry during a single continuous 8-hour
session. Figures are representative of this run and this configuration; normal run-to-run
variation applies.*
