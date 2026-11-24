<!-- Ported from github.com/AbsolomBane/veloGB10 branch feature/kv-pool-tp-0710 (tip 9df3b15), by
@AbsolomBane (veloGB10 issue #43). Differences in the integrated v0.8.0 port:
- On a TP rank (--tp / --head / --node) --kv-pool is NOT refused at startup: the server boots with the standard
  per-lane KV allocation, prints one notice line and sets velogb10_kv_vmm_fallback{reason="tp"}.
  The TP groundwork (dry runs, the head's capacity gate, the lane-set agree) is in, inert until the pool runs under TP.
- Models whose KV is not in the EXL3 position planes (other families) get the same fallback with reason="model".
- /metrics: velogb10_kv_vmm_{mode,pool_bytes,mapped_bytes,reserved_bytes,evictions_total,length_ends_total,waits_total}.
- The figures below are the contributor's (one ASUS GX10, v0.7.9 / v0.7.10 base).
-->

# Shared KV pool (AbsolomBane/veloGB10, branch `feature/kv-pool-tp-0710`, on v0.7.10)

> **Status:** the pool is tested on one GB10 (ASUS GX10), TP=1 (every figure below). This branch also carries the
> **TP=2/4 groundwork**, written and covered by CPU tests only: **never run on TP hardware**, and the pool is still
> refused at startup with any TP role. If you have a TP=2 or TP=4 rig and want to contribute, see
> [TP=2/4: help wanted](#tp24-help-wanted) below.
>
> The GPU figures were measured on the pool code before the TP commits. Those commits touch the one-box path only
> through read-only refactors (the lane pick, the eviction order) and the group read-ahead exhaustion, and that part
> is covered by CPU tests, not yet re-measured on the GPU.
>
> The branch starts with `feature/fit-floor-gb`, its first commit ([`8217aec`](https://github.com/AbsolomBane/veloGB10/commit/8217aec)): `--fit-floor-gb`, the boot plan's free-memory
> floor as a registry flag (default 20 = upstream's fixed value). It is independent of the pool and useful on its own;
> the pool recipe below uses `--fit-floor-gb 5`.

Designed, directed and tested by [@AbsolomBane](https://github.com/AbsolomBane).
Base: veloGB10 v0.7.10 by sf-stav (Apache-2.0) — this branch is the port of the `kv-pool` work from v0.7.9. Everything below is opt-in: without `--kv-pool` (default off) the engine behaves as v0.7.10.

## What it does

Upstream velo gives every lane a fixed KV slab sized for `--max-seq-len`. 8 lanes x 1M would need ~150 GB.
This fork backs the six positional planes (trunk K/V, QSA keys, QSA pooled keys, and the same three for the MTP head)
with CUDA virtual memory: each lane reserves address space for its full context, but physical 2 MiB pages come
from **one shared pool** and are mapped only for the rows a lane actually uses.

* One long session can grow to 1M while other lanes stay small, all from the same physical budget.
* Pages are created once at boot (no `cuMemCreate` under memory pressure); serving only maps/unmaps existing handles
  (~0.06 ms per 2 MiB). CUDA graphs survive remapping.
* When the pool is full, idle lanes are evicted (snapshot to kv-spill SSD, pages returned). **The longest idle
  session is evicted last** — short agent/sub-agent sessions go first. If nothing is evictable, a DECODING lane that
  needs pages finishes with `length` (only that lane; the others keep decoding in the same round). A NEW request whose
  prompt does not fit waits at the head of the queue while other lanes still decode, and is admitted after a lane
  finishes (an ended lane is idle and evictable); only when no lane is decoding does it finish
  `error: prefill failed: kv-vmm: pool exhausted`. A failed mapping is rolled back, so no page is left half-owned.

## Scope and limits

* Tested on one GB10 (ASUS GX10) (TP=1) only.
* **Lazy mode costs two diagnostic routes.** The WP16 head-KV poison gate (`exl3_forward/wp16.rs:419`)
  returns early under lazy, so the PFX1 (b) "every head-KV row a draft can read was written first"
  invariant can no longer be PROVEN with the pool in lazy — the reset path still zeroes/poisons only
  the mapped rows (`exl3_forward/slot_state.rs:246-259`), the same guarantee over a smaller set. And
  the autotune self-test and the xtp self-test refuse lazy (`exl3_autotune/digest.rs:301`,
  `exl3_forward/xtp/gates.rs:100`), so the bit-exactness evidence for lazy is `--kv-pool full`
  plus the served-path smokes.
* **One box only: `--kv-pool full|lazy` is refused at startup with `--tp` / `--head` / `--node`.** The notes below
  describe why (kept for whoever adds a TP event for it).
* Diagnostic probes that memset or read back WHOLE position planes refuse lazy with a message: the autotune and xtp
  self-tests, the packstate probe, `--probe-exl3-packpf`, and `--probe-exl3-prefillx` with `--prefillx-qsa-keys`,
  `--prefillx-rel` or `--prefillx-tier`. Use `--kv-pool full` (or off) for them.
* A waiting request blocks the queue behind it (FIFO), as a request waiting for a free lane does today.
* **TP=2/4: groundwork only, never run on TP hardware** (see the next section). The pool stays refused under TP.

## TP=2/4: help wanted

We have one GB10, so nothing here has run at TP>1. What the branch adds for TP (CPU tests only, `cargo test --lib` 1238/1238):

* **Dry-run page queries** in `KvVmm` (`pages_needed`, `pages_freed`, `pages_needed_union`) and their `FwdModel` twins
  (`vmm_pages_needed`, `vmm_pages_freed`), built on the same page tables and unit math as the real map/unmap.
* **A lane-set agree after the decode map-ahead** (`step.rs`): every rank compares the survivor count and the ordered
  slot ids before any return, so a lane ended on one rank only cannot reach the next collective.
* **A spill vote on a failed capture** (`pool.rs`): a rank whose D2H capture fails still votes `false` (upstream v0.7.10
  returns before the vote; `spill_agree_tests` shows what that does to the next votes).
* **The head's capacity gate** (`tp_pool_gate.rs`): before an Admit ships, the pages the prompt needs on the lane the apply
  will pick (restore target included); held while a lane runs, refused with an error when nothing can make room; one
  Admit per iteration while it decides. Victims are not credited yet (`TP_EVICTS = false`): eviction stays single-box.
* **Group read-ahead exhaustion ends a lane** (`finish=length`) instead of failing the step (single box too).

Still missing before the refusal can be lifted: replicated eviction under TP (and `TP_EVICTS`), a per-slot cache
fingerprint in the round digest, and a run on real TP hardware. On a TP=2 rig, with the refusal in
`kv_vmm::tp_topology` removed locally, the checks that matter are: greedy outputs identical to plain v0.7.10 TP; a pool
small enough to fill on both ranks (lanes must end `finish=length` on the same lane on every rank, no hang, no fatal); a
prompt that does not fit (held, never a fatal); eviction and a cross-lane restore once eviction is replicated; and the
needle at 1M. Reports (logs of both ranks) are welcome as issues on this fork.

## Usage — the recipe we run in production (the memory figures in this section were measured on the v0.7.9 base)

One GB10 (ASUS GX10, 128 GB), Qwen3.8-Flash-Next EXL3 3.05: **8 lanes x 1M context each, one shared 30 GB KV pool.**

```
mkdir -p ~/velo-spill
./gb10_inference --server --model-dir <flashnext-exl3-305> --port 18301 \
  --kv-pool lazy --kv-pool-gb 30 --fit-floor-gb 5 \
  --max-seq-len 1048571 --max-batch 8 \
  --prefix-cache on --prefix-ckpt-mem-gb 2 --ple-ram ssd --lane-order rr \
  --rope-yarn-factor 4 --kv-spill ~/velo-spill --kv-spill-gb 40 --image-max-edge 1600
```

| | |
|---|---|
| Context per lane | 1,048,571 tokens (max_pos a multiple of 4; on v0.7.9 that also sidestepped the QSA pooled-key bug, which v0.7.10 fixes upstream) |
| Shared pool | 30 GB physical, about 1.56M tokens across all lanes (q8 KV, ~18.3 KB/token) |
| Address space | 8 x 1M reserved (153.6 GB virtual, no physical cost); ~3.4 GB mapped at boot |
| One lane at 1M | ~0.4-0.55M tokens left for the other 7 lanes; idle lanes beyond that go to the SSD spill tier, smallest first, the long session last |
| Free memory while serving | ~16-17 GiB MemAvailable (measured) |

Pool size trade-off on this box: 20 GB gives ~1.04M tokens total (a 1M session leaves almost nothing, frequent
evictions); 40 GB gives ~2.1M but only ~7 GiB stays free, below the margin we run with. 30 GB is the balance.

`--kv-pool full` maps everything at boot (same memory as upstream, useful for A/B). The boot memory plan counts the
pool (+0.3 GB per lane) instead of `lanes x max_seq_len`. `--fit-floor-gb 5` lowers the boot plan's free-memory
floor from upstream's 20 GB; the pool itself is what bounds KV memory.

## Fixes in this branch

1. **Shared KV pool** (`src/kv_vmm.rs`; hooks in `exl3_forward/{mod,load,slot_state,prefill,wp16,tensors}.rs` and `exl3_serve/{prefill,step,fit}.rs` — v0.7.10 split both files this work was written against).
2. **Mixed-row prefetch mapping**: the mixed-row pooled scorer prefetches the first lane up to the group's longest
   row count; the group is mapped before each step (`vmm_ensure_group`).
3. **QSA pooled-key geometry — NOT ours any more, and dropped on purpose.** The fork's fix B was a copy-time
   `qsa_pool_lane` helper plus spill payload version 4 -> 5; the bug (the cross-lane copy stepping
   `ceil(max_pos/ratio)` blocks a lane while the kernels step `floor`, so a session restored into another lane
   read shifted pooled keys) is real and was reported by @AbsolomBane. **sf-stav shipped the upstream fix for
   issue #41 in v0.7.10** and this branch uses it: that fix moves the *allocation* to the kernels' floor stride
   (`qsa_pool_plane_elems`, `exl3_forward/scratch.rs:44`, called at `exl3_forward/load.rs:881` and `:1043` here —
   `:876` / `:1038` in pristine v0.7.10, five lines lower because our own `kvvmm` block sits above them at
   `load.rs:424-431`), while the copies derive the lane stride from the plane size (`pool_planes` at
   `exl3_forward/slot_state.rs:524`, `pool_units` at `:560`; `:317` / `:420` pristine, before our 91-line `vmm_*`
   block). Payload version 5 is unchanged and the layout is the same one this fork produced, so no spill tier
   is invalidated. Upstream's test `pooled_plane_copy_addresses_every_lane_where_the_kernels_do`
   (`exl3_forward/spill_region_tests.rs:47`) pins the geometry.
4. **Eviction with priority** (longest session kept).
5. Diagnostic self-tests that memset whole planes (xtp, autotune self-test, packstate) refuse to run in lazy mode
   with a clear message.
6. `--fit-floor-gb`: configurable free-memory floor for the boot plan (default 20, as upstream; never below `--mem-watchdog-gb`).
7. Upstream conventions kept: all three settings are registry flags (`src/opts_table.rs`, CLI-1, no environment reads), the
   pool is refused at startup with `--tp` / `--head` / `--node`, and `cargo test --lib` passes in full.

Scripts (decode streams, needle, VeloBench runner, live monitor): branch `feature/bench-tools`.

## Tested on v0.7.10 (2026-11-18, one GB10 (ASUS GX10), Qwen3.8-Flash-Next EXL3 3.05, YaRN x4, q8 KV, PLE on SSD)

Reference = plain v0.7.10 (0642b78) built from the tag on the same box, same flags. Greedy = temperature 0, seed 42.
Pool runs use `--kv-pool lazy`. Every row below was measured on this branch (the first rows on `c0d872d`, the
rows marked *new* on the final branch with the flags, the wait and the rollback).

| Test | Result |
|---|---|
| Perplexity, 32 windows (`--perplexity`, q8, YaRN 2): plain v0.7.10 / pool off / full / lazy | **9.6374** on all four |
| Greedy outputs, 6 prompts (code, prose, tool call, 30k, 100k, 300k tokens), 2 x 1,048,571 window: plain v0.7.10 vs pool off, vs pool lazy (30 GB), and vs pool lazy *new* | **6/6 identical** in each comparison (serial requests, warm prefix cache) |
| Cross-lane restore, `--max-seq-len 262144` (max_pos 262,145, the unaligned case of issue #41), *new*: 15k prompt on lane 0, its lane kept busy by a long request, the prompt sent again | restored from the SSD tier **onto lane 1**, answer **identical** to the first and to plain v0.7.10 |
| Same at `--max-seq-len 1048571`, *new* | restored onto lane 1, identical to plain v0.7.10 |
| SSD restore, 100k prompt, 1M window | 67.3 s cold -> 6.9 s (1,875 MiB read), identical |
| Priority eviction, 20 GB pool, 8 lanes: one 555,717-token session, then 3 agents of 121k, then one agent of 302k | the 3 agents were evicted to SSD (2.4-3.2 s each), the long session stayed; sent again: **3.9 s** (489 s cold), identical |
| Pool full while decoding, 5 GB pool, 8 x 30k prompts (c0d872d) | exactly one lane ended `finish=length` by the pool, the others kept decoding; the ended lane was evicted for them next. 5 prompts that did not fit at admission got HTTP 500 (fixed below) |
| Same, *new* (prompts that do not fit wait in the queue) | **8/8 completed, 0 errors**; 5 waited and were admitted as lanes finished; idle lanes evicted in ~0.7 s |
| Needle, 1,046,957 tokens, 6-digit code at 55%, 8 lanes x 1M, 30 GB pool | **found**; prefill 1,074 s (~975 tok/s, debug logging on); sent again from the prefix cache: 2.6 s, identical |
| Decode, 8 lanes x 1M, 30 GB pool, 8 streams (1,200 tokens each, temperature 0.6), *new*, normal logging | **127.3 tok/s** aggregate, 16.0 GiB free after (v0.7.9 fork, same posture: 126.2 tok/s, 16.3 GiB); with a 26 GB pool: 132.1 tok/s, 14.7 GiB free |
| Same with `--log-level debug` (c0d872d), 1/2/4/8 streams | 60.2 / 89.3 / 90.9 / 107.4 tok/s, ~10 GiB free: debug logging alone costs ~15% at 8 streams and ~5 GiB here |
| Startup refusals, *new* | `--kv-pool lazy --tp 2` and `--kv-pool maybe` refused before the model loads |
| `cargo test --lib`, *new* | 1225 passed, 0 failed |

**Isolation: is the pool slower, or v0.7.10?** Same box, same flags, 8 lanes x 131,072 (fits without the pool), 8 streams
of 1,200 tokens (temperature 0.6), then a cold 100k-token prefill:

| Build | Free at boot | Decode, 8 streams | Cold 100k prefill |
|---|---|---|---|
| plain v0.7.9 | 27.99 GiB | 130.3 tok/s | 67.7 s |
| plain v0.7.10 | 25.34 GiB | 131.8 tok/s | 75.0 s |
| this branch, `--kv-pool off` (`--fit-floor-gb 5`) | 22.61 GiB | 131.7 tok/s | 77.3 s |
| this branch, `--kv-pool lazy --kv-pool-gb 30` | 12.07 GiB (the 30 GB pool is allocated at boot) | 131.9 tok/s | 73.1 s |

The pool costs no decode or prefill speed. v0.7.10 itself prefills ~11% slower than v0.7.9 here and keeps ~2.7 GiB
less free at boot. The ~2.7 GiB between plain v0.7.10 and `--kv-pool off` is `--fit-floor-gb 5`: with the default
floor (20) the same build boots with 25.08 GiB free and 92 decode graphs, as plain v0.7.10 (25.34 GiB, 90). The 30 GB
pool is sized for 8 x 1M; at 8 x 131k it is mostly unused headroom.

## Side by side: fork vs official v0.7.9 (same box, same day, 2026-11-17 — measured on the v0.7.9 base, NOT re-run on v0.7.10)

Fork = this branch, 8 lanes x 1M, 30 GB pool. Official = veloGB10 v0.7.9, 2 lanes x 1M (`--max-batch 2`), the
configuration we ran in production before the pool. Same model, same flags otherwise (YaRN x4, q8 KV, PLE on SSD,
kv-spill 40, prefix cache, `--lane-order rr`), `--max-seq-len 1048571` on both.

### VeloBench (sf-stav/VeloBench v0.4.0, built-in tests, temperature 0)

Reproduce: run VeloBench, add the server as a provider, then `POST /api/run` with
`{"provider":..., "testname":"prebuilt-pure-code-c", "mode":"concurrent", "workers":N, "temperature":0}`
(and `prebuilt-creative-scifi`, `prebuilt-prefill-scaling`). Figures are VeloBench's own `endToEndTokS`,
`wallMs` and `avgTtftMs`. Where a cell has two values, the test was run twice.

| Test | Workers | Fork tok/s | Official tok/s | Fork wall | Official wall | Fork avg TTFT | Official avg TTFT |
|---|---|---|---|---|---|---|---|
| pure-code-c | 1 | 75.5 | 81.2 | 54 s | 50 s | 1.6 s | 1.6 s |
| pure-code-c | 2 | 129.5 / 134.9 | 126.6 / 128.7 | 61-63 s | 64-65 s | 0.2-0.3 s | 0.2-0.5 s |
| pure-code-c | 4 | 111.5 / 125.1 | 128.7 / 134.2 | 131-147 s | 122-127 s | 0.2-0.3 s | 30-32 s |
| pure-code-c | 8 | **168.4** | 126.0 | **195 s** | 260 s | **0.6 s** | 97.7 s |
| creative-scifi | 1 | 62.7 | 63.1 | 33 s | 32 s | 0.4 s | 0.4 s |
| creative-scifi | 2 | 83.0 / 104.2 | 102.6 / 102.8 | 39-49 s | 40 s | 0.2-0.3 s | 0.2-0.6 s |
| creative-scifi | 4 | 114.5 / 120.7 | 103.6 / 109.6 | 68-72 s | 75-79 s | 0.2-0.4 s | 19-19.5 s |
| creative-scifi | 8 | **170.4** | 104.0 | **96 s** | 157 s | **0.7 s** | 59.1 s |

Prefill scaling (single request): 1K 691 vs 556 tok/s, 8K 1,538 vs 1,544, 64K 1,792 vs 1,791 (fork vs official).

Note: with these built-in tests the model spends its whole token budget thinking (100% reasoning tokens), on both
servers; it does not change the speed comparison.

### Our decode test (`bench/decode_streams.py`, 1,200 tokens per stream, aggregate tok/s — the run that produced the `par8` tables)

| Streams | Fork | Official |
|---|---|---|
| 1 | 61.8 | 62.0 |
| 2 | 88.1 / 86.3 / 65.2 | 80.8 / 86.6 |
| 4 | 95.1 / 98.9 | - (2 lanes) |
| 8 | 126.2 | - (2 lanes) |

### Reading it honestly

* **1-2 agents: same speed.** The pool costs nothing on decode or prefill.
* **4 agents:** on code the official build is ~7-15% faster in aggregate (two lanes batch more efficiently than
  four), but two of the four agents wait ~30 s for their first token; on prose the fork is ahead. Every fork agent
  starts in under a second.
* **8 agents: the fork wins clearly** — +34% (code) and +64% (prose) aggregate, the whole batch done 25-39%
  sooner, and first token in under a second instead of 1-1.6 minutes in the queue.
* Free memory while serving: ~16-17 GiB on the fork (8 lanes) vs ~13 GiB on the official build (2 lanes).

## Measured (GB10 / ASUS GX10, Qwen3.8-Flash-Next EXL3 3.05, YaRN x4, q8 KV, PLE on SSD) — every row below is from the v0.7.9 base and has NOT been re-run on v0.7.10

| Test | Result |
|---|---|
| Perplexity, off / full / lazy | 9.6374 / 9.6374 / 9.6374 |
| Greedy outputs at 30k / 100k / 300k + 3 short, off vs lazy | identical (sha match), same speed |
| Boot 4 x 1M and 8 x 1M lanes on a 20 GB pool | starts; 153.6 GB address space, 3.4 GB mapped at boot |
| Decode, 8 lanes, 20 GB pool, 1 / 2 / 4 / 8 streams | 64.2 / 82.6 / 91.8 / 125.3 tok/s aggregate (8 streams rerun: 90.7) |
| 800k session + 3 short sessions in a 20 GB pool | all complete (eviction) |
| Priority: 555,717-token session + 3 x 121k agents, then a 302k agent | the 3 agents evicted (~2.4 s each), long session kept; long session re-sent: answer in 3.3 s, identical sha |
| Prefill of the 555k session | 422 s (~1,317 tok/s) |
| Cross-lane restore, unaligned `max_pos`: v0.7.9 vs fix B (measured on the v0.7.9 base; v0.7.10 fixes it upstream) | diverges vs identical |
| 31-minute Hermes agent run, 30 GB pool, 6 lanes busy (main + 4 parallel sub-agents) | 0 errors, no eviction needed |
| Needle at 1M: 1,046,957 tokens, 6-digit code at 55%, greedy, 30 GB pool / 8 lanes | found; prefill 881 s (~1,189 tok/s); re-sent from prefix cache in 2.7 s, identical |
| Diagnostics in lazy mode | refuse cleanly; unchanged with the pool off |

## Scope & Support

Hardware: GB10. Only. PRs for other GPUs welcome but not tested and not
maintained here.

Models: EXL3 quants. If a model panics velo and the cause is in the quant
itself (fractional or mixed expert bitwidths, shards missing a module,
tensor shapes that don't match the engine's lane stride), the issue
belongs upstream with the EXL3 project or the quant's author — we'll help
identify it, not patch around it.
