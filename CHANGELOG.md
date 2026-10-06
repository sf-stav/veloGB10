# Changelog

High-level release notes for veloGB10. Minor bug fixes and small optimizations are grouped under
generic language where they aren't individually notable.

## v0.7.3 — the fixes you reported: TP=2 pre-verify race, exllamav3 1.5.x pack loading, cached-token accounting; tool-call parsing; long-running hardening

This release closes the three open issues from the v0.7.2 feedback (#8, #9, #10), makes the
server tell the truth about cached prompt tokens, stops a stalled log reader (or a slow OTLP
receiver, or a client that stops reading a stream) from stalling the engine, and adds the
operational surface (gauges, admission bounds, an operations guide) for running the server for
hours at a time. The two tarball launcher scripts changed interface: they take flags now, not
environment variables.

### Fixed — TP=2 `TP pre-verify FAILED` (issue #10)

At world size 2, the host-side control exchange staged its RDMA frames into the same ring the
device doorbell epochs use: when the exchange's slot counter happened to alias the slot holding
the last draft-pass epoch payload, the first bytes of that payload were overwritten and the
pre-verify hash compare fired — the `TP pre-verify FAILED` abort a few of you hit over
v0.7.0–v0.7.2, typically hours into a run. World sizes above 2 already used dedicated control
rings; world 2 now does too, which closes the whole aliasing class (not just the pre-verify
instance). The bug was confirmed with a deterministic fault-injection probe that fails on
v0.7.2 and passes on v0.7.3; outputs are bit-identical to the previous build
(`--exl3-tp-ident` hashes unchanged at TP=2 and TP=1), and decode speed is unchanged within
measurement noise. The `--tp-dh-shard 0` workaround is no longer needed. Cost: ~11.5 MB of
pinned host memory per rank. Verified on the release candidate: the fault-injection probe logs
0 clobbered slots over 142 aligned rounds (v0.7.2: 14), and a 30-minute stress checkpoint
(c=1, ~16K-prompt sampled requests, temperature 1) logged 38,273 pre-verify rounds with 0 aborts
and 0 clobbered slots on both ranks — that configuration runs ~1,270 rounds/min, so the longer
(~2 h / ~200,000-round) soak runs after this release and was not waited for. TP=2 output identity:
IDENT_ALL 871f083d58a54281 (code) / 38458d4ce6310c6f (prose) at 10/12/16 lanes, with r16 on,
`--exl3-r16 off` and `--router-coal 0` alike; decode medians unchanged (c=1 82.07 vs 82.10 tok/s,
4 concurrent 127.57 vs 127.60).

### Fixed — packs from exllamav3 1.5.x with the sharded n-gram sidecar (issue #9)

exllamav3 1.5.x writes the n-gram table as 128 shard tensors in the sidecar, and most public
Flash-Next EXL3 packs now use that layout; some of them also omit the shards from
`model.safetensors.index.json`. v0.7.2 refused those packs at boot (`ple shard 0 not in
index`). v0.7.3 loads both layouts as shipped — single-tensor and 128-shard, K=5 and K=6,
index-listed or not (the loader falls back to the sidecar's own safetensors header and
validates the layout; a boot line names the one in use). No flags, no workarounds: the
header-rewrite workaround some of you used is no longer needed. Packs that loaded before are
byte-identical before and after (gated). A tested-packs table is in the Flash-Next setup guide.

### Server surface (issue #8)

- `/v1/models` and `/v1/models/{id}` now report `max_model_len` = the served `--max-seq-len`.
- `usage.prompt_tokens_details.cached_tokens` — the number of prompt tokens served from the
  prefix cache (prompt-end hit or checkpoint resume, whichever is deeper) — on the chat
  non-streaming response, the streaming usage chunk (`stream_options.include_usage`), and the
  legacy non-streaming `/v1/completions` response; EXL3 and NVFP4 paths alike. The matching
  counter is `velogb10_prompt_tokens_cached_total`.
- `--model-name` / `--served-model-name` are honored on the EXL3 server (they used to be
  silently ignored there); the resolution order is `--model-name` > `--served-model-name` >
  the model card's `base_model:` > the directory name. Model ids containing `/` work.
- Known gaps, stated plainly: the minimal DSV4 server does not report the new fields, and the
  legacy **streaming** `/v1/completions` path emits no usage chunk.

Release artifacts: `velogb10-v0.7.3-gb10-sm121.tar.gz` + `SHA256SUMS.txt` + `PROVENANCE.txt`
(binary sha256 `949cab4758ae8af04565a826f28290c4f4280b179d95c93257361c6e09af706f`; built from this
tree at the release-source commit `d7529bd` with a clean checkout; dev repo `rel/cutb` @ `3a4cab1`).

### Tool calls: literal `</parameter>` inside a value

A parameter value that itself contains the text `</parameter>` (an agent writing a file *about*
the tool format) used to be silently truncated at that point. The parser now ends a value only
at the *structural* close — the `</parameter>` followed by the next `<parameter=`, by the
`</function>` that closes the call, or by the end of the call block. Two residual ambiguities
are inherent to the format and documented in the Flash-Next guide, together with advice for
agent authors (prefer a diff-style edit tool, or break the closing sequence when quoting
tool-call syntax).

### `--keep-tools-when-tool-choice-none` (opt-in, default off)

vLLM-compatible handling of `tool_choice: "none"`: the tool definitions stay in the prompt
(byte-identical tools block, so the prefix cache and checkpoints survive a mid-conversation
summary/compaction turn), a do-not-call instruction is appended, and the reply comes back as
plain content. In the replayed reporter trace (309-message conversation, ~89K prompt tokens),
the compaction request and the following turn went from three full ~80 s re-prefills to
7.9 s and 1.3 s. The model still *sees* the tools and is merely told not to call them — a
stray call returns as visible text. The default stays off (byte-identical legacy behaviour);
`--max-batch 2` remains the no-flag alternative. Details in the Flash-Next guide.

### Telemetry on Flash-Next

`--otel-endpoint` now works on the EXL3 (Flash-Next) server — it was silently ignored there.
The `--otel-*` flags are shared by both servers; an unusable endpoint is refused before the
model load. Under TP, only the head emits (nodes have no HTTP hooks). The sender is fully
async: a slow or absent receiver delays nothing but itself.

### Long-running hardening

- Log lines no longer go straight to a locked stdout: a bounded non-blocking queue (4096
  lines, drop-and-count, one recovery summary) means a stalled log reader can no longer stall
  the scheduler (`velogb10_log_lines_dropped_total` counts the losses).
- `--max-waiting` (default 256): 503 + `Retry-After: 5` instead of an unbounded admit queue.
- `--stream-backlog-events` (default 65536 ≈ 11 min at 100 tok/s): a stream whose consumer
  stopped reading is cancelled like a client disconnect.
- New gauges (RSS, threads, fds, memory headroom, scheduler progress/busy/age, graph-cache
  entries, rejected requests, cancelled streams, dropped log lines, cached prompt tokens) and
  exit-path log flushing, plus `=`-form flag parsing on both servers.
- New `docs/OPERATIONS.md`: supervisor rationale, systemd units, log handling, the gauge and
  alert table (alert on `scheduler_busy == 1 AND age > 120` only), exit codes, soak watch-list.
- `--exit-on-fatal` is now the default on the EXL3 server (it was opt-in): on a sticky CUDA
  error or a scheduler-thread panic the in-flight request finishes with an error and the process
  exits 70 for a supervisor to restart, instead of staying up as a DEAD engine answering 503.
  `--exit-on-fatal off` restores the old behaviour; the systemd examples in docs/OPERATIONS.md
  assume the default. The NVFP4 server is unchanged (it never had the option).

### Kernels

New in v0.7.3: fast paths for 9–16-row router folds and HC mixes (v0.7.2 ran the legacy
kernel chain at those widths), live at default flags: bit-identical output, 8/8 identity suite.
In isolation the kernels measure hc ~7.2 → ~6 ms and router ~4.8 → ~1.3 ms at 10 rows. Measured
end to end on one box (TP=1, alternating boots): decode at c=1 is unchanged (91.2/90.1/92.0 vs
91.7/91.4/91.3 tok/s, inside the 1.9 tok/s run-to-run spread), and aggregate throughput at
`--max-batch 16` with 10 concurrent requests measures 145.4/148.8 vs 139.3/141.2 tok/s
(2 reps per side, +4.4 % / +5.4 %, beyond the spread) — the expected effect of the row-batched
router/HC kernels on that workload; no claim is made for TP ≥ 2 (expert-parallel ranks keep
the legacy router at 9–16 rows) or for other workloads.

### `--lane-order fcfs` / `--lane-quantum` (opt-in; default `rr` unchanged)

For several concurrent requests in the serial-speculation arm: `rr` (today's behaviour) gives
every busy lane a round per scheduler step — n requests each run at 1/n speed and all finish
late; `fcfs` runs one lane to completion at a time (first-come-first-served, bounded by
`--lane-quantum`, default 256 generated tokens). Aggregate throughput is unchanged. Measured:
with equal-length turns at TP=1, mean completion time drops 23% at 2 and 30% at 3 concurrent
requests; two 512-token turns at `--lane-quantum 1024` finish 20–24% sooner on the mean
(8.2–8.5 s vs 10.4–10.8 s). The cost is the later lanes' time-to-first-token, and with mixed
lengths `rr` wins the mean (4.24 s vs 4.72 s at quantum 256).
Greedy output is byte-identical in both orders; TP=2 output identity verified.

### Launcher interface changed (tarball scripts)

`run_tp_server.sh` and `run_tp_node.sh` took `MODEL_DIR=`, `PORT=`, `NODE=`… environment
variables and carried stale defaults. They are plain flag scripts now:
`run_tp_server.sh --model-dir DIR --node IP:29500 [--port 9000] [--max-seq-len N]
[--max-batch N] [--tp 2] [-- <engine flags>]` and `run_tp_node.sh [--port 29500]
[--rdma-dev DEV]`; `--help` and `--dry-run` on both, missing required flags fail loudly, and
the server always binds `--host 0.0.0.0`. The old environment overrides are gone.

### Not in this release

The packed multi-request speculative verify (two and three concurrent requests on the EXL3
path) is still probe-only: it exists as hidden `--probe-exl3-pack*` / `--pack3-*` diagnostics,
is not wired into serving, and is not a supported feature. Two and three concurrent requests
still share the single-request aggregate rate.

## v0.7.2 — concurrent requests on the EXL3 path, Qwen3.8-Flash-Next 4.05 bpw, `/metrics`, `--host`, and the bugs you reported

This release answers a community benchmark of Flash-Next as a coding-agent executor (thank you — it
was exactly the right test), adds the higher-fidelity 4.05 bpw pack, and fixes what was found along
the way. It adds a **fourth PTX file** to the deploy set (`src/ptx/exl3_bench_k6.ptx`); copy the whole
`src/ptx/` directory as before — the release tarball already has it.

### Concurrent requests on the EXL3 path (Qwen3.8-Flash-Next)

Speculative (MTP) rounds run one request at a time, so with several busy requests the aggregate used
to stay flat at the single-request rate. Now each scheduling round picks the better of two arms by
estimated aggregate tokens/s: serial speculation, or **one shared batched step for every busy request**
(`--spec-lanes-max auto`, the default; `--spec-lanes-max N` shares the step only above N busy requests,
`0` keeps serial speculation). A single busy request always speculates exactly as before, and every
request's greedy output is byte-identical to running it alone, at every topology.

Aggregate generated tokens/s on a code workload (256-token requests, one lab box per node, untuned
build, indicative absolute values), v0.7.1 behaviour → v0.7.2, at 4 / 8 / 16 busy requests:

| | 4 requests | 8 requests | 16 requests |
|---|---:|---:|---:|
| TP=1 | 91 → 97 | 89 → 129 | 88 → 168 |
| TP=2 | 130 → 135 | 125 → 202 | 124 → 249 |
| TP=4 | 159 → 173 | 152 → 238 | 154 → 293 |

Prose (lower draft acceptance) gains earlier: at TP=1, 4 requests 67 → 89 and 8 requests 68 → 119. The
TP=2 and TP=4 figures are single passes.

**Two and three concurrent requests are not faster yet:** their aggregate stays at the single-request
rate (each request gets its turn). A packed multi-request speculative verify for that range is in
development and is the next item on the list.

- Under the hood: a K-chunked expert gate/up for 10–16-row calls, bit-identical to the paths it
  replaces (plain batched decode at 10–16 requests is ~20–25% faster); single-request speed and output
  are unchanged.
- Fixed on the way: at TP=2 with `--max-batch 16` the server could stop with `TP pre-verify FAILED ...`
  once several requests were busy (the two ranks could hold different sets of captured CUDA graphs).

### New: Qwen3.8-Flash-Next at 4.05 bpw (EXL3)

[`doth4580/Qwen3.8-Flash-Next-EXL3-4.05bpw`](https://huggingface.co/doth4580/Qwen3.8-Flash-Next-EXL3-4.05bpw)
(turboderp's `4.05bpw_h6_ng6`: 4-bit experts, 6-bit dense layers and head, a 6-bit n-gram table;
~108 GB) serves at **TP=1** (the n-gram table is read from SSD — see below), **TP=2 and TP=4**, with
image input. Untuned smoke-test figures (1,000 generated tokens / 2K-token prefill, single request):
TP=1 63 tok/s decode and 1,399 prefill; TP=2 89 and 2,612; TP=4 120 and 3,496 (the 3.05 pack measures
~73 / 102 / 122 decode on the same test). The pack was checked against the exllamav3 reference
implementation itself: every 4-, 5- and 6-bit weight tensor class (15.8M trellis blocks, including the
`lm_head`) and the 6-bit n-gram path decode bit-identically to exllamav3's. The model directory needs
`vision_tower_bf16.safetensors` for image input (it is in the Hugging Face repo).

### Memory: the n-gram table can live on SSD

`--ple-ram ram|ssd|auto` (default `auto`; `on`/`off` still work). `ssd` reads the n-gram table from
disk through the page cache and frees its 30–39 GB for requests and context: identical output, decode
~1–4% slower and cold prefill up to ~10% slower (measured on a PCIe Gen 5 drive). `auto` now decides
**after** the boot has allocated everything else (RAM if the table plus 16 GiB stay free, otherwise
SSD, logged with the reason), which fixes `--max-batch 8` at 131K being stopped by the memory watchdog
at boot on one Spark. A configuration that cannot fit at all is refused **before** the load, with the
arithmetic and the fix (`--ple-ram ssd`, a smaller `--max-batch` / `--max-seq-len`, or `--tp 2`).

### New flags and endpoints

- `--host <addr>` — the bind address of the HTTP API (default `0.0.0.0`; `127.0.0.1` = this machine only).
- `GET /metrics` — Prometheus text format, on by default, no measurable cost: requests by finish
  reason, prompt/generated tokens, requests running/waiting, time-to-first-token / request-duration /
  prefill and decode-rate histograms, speculation totals, engine alive, build info.

### Fixed

- **Tool-call history** is rendered the way the model writes it: `arguments` that are a JSON object are
  parsed before the chat template runs, even for templates with an `arguments is string` branch (they
  used to render `{"command": ...}` inside `<function=...>`, a mixed XML/JSON history the model copied
  in long sessions; issue #6).
- **Tool-call parsing:** text inside a parameter value that merely looks like a function tag
  (`<function=x>` or a bare `function=x>` line, for instance a file an agent is writing about this
  format) is payload and no longer produces a second, phantom tool call.
- **NVFP4 with DFlash2 or DSpark and `--max-batch 2`:** the scheduler could stop with
  `df2 ring nprev 512 != lane pos 713` when a second request arrived while the first was decoding (the
  drafter ring is shared). A request that loses the ring now continues on MTP.
- **NVFP4 `--kv-cache k8v8` now serves any `--max-batch`** (it was refused above one request). Plain
  batched decode runs the same attention kernel a lone request runs, so a request's text does not
  depend on how many others are busy; tree verification and `--mtp-lanes` use new int8 readers. KV
  compaction after a tree-verified step on k8v8 used the bf16 copy kernel on int8 data; fixed.
- A missing `--model-dir` is a clear error instead of a panic; TP shard shipping includes sidecar
  files the tensor index omits.

### Known limits

- Two and three concurrent requests on the EXL3 path do not scale yet (above).
- On the NVFP4 path, above one request the lanes run as a plain batch (the MTP head holds one request's
  state): two requests together are slower in aggregate than one speculating request, and a request
  that shared a batched step does not speculate again for the rest of that request.
- Seeded sampled requests are not byte-reproducible across runs. Video and audio parts return `400`.
- 4.05 bpw is untuned (no autotune decisions were found for it; it decodes ~0.86× as fast as 3.05 on
  one or two nodes).

## v0.7.1 — TP=4 on the EXL3 path, image input, automatic shard shipping

A fast follow to v0.7.0: the EXL3 path gains a fourth node and images, the node sync becomes
shard-level, and two defaults change that alter long-context output bytes.

### TP=4 on the EXL3 path

`--tp 4` now serves Qwen3.8-Flash-Next from EXL3 packs across three peer nodes plus the head:
attention, GDN and the KV cache split by head, the routed experts dealt across the ranks, the
vocab-parallel LM head, sharded MTP-head screening, a per-step lockstep agreement check and a
watchdog. Measured on pure code: **~221 tok/s decode** (min/max 173/240, p50/p90/p99 221/234/237) and
**~3.2K tok/s prefill** — TTFT 0.64 s at ~2.1K input tokens, 10.9 s at ~34.9K. Against a single GB10
that is **×1.6 decode and ×2.0–2.2 prefill**.

### Image input, on every topology

Images are served end to end on the EXL3 path at TP=1, TP=2 and TP=4: the pack's **original bf16
vision tower** is loaded (`vision_tower_bf16.safetensors`), each image is resized so its longer side
is at most `--image-max-edge` (default 1024, aspect preserved), 3-axis mrope is applied at all six
main-attention RoPE sites **and** the QSA indexer — so images work past the 2,051-token dense window —
and image rows ship to the TP nodes as a digest-checked binary payload. Parts are returned in client
order. **Video and audio parts return `400`** for now; video is planned.

### Automatic shard shipping

A multi-node run no longer needs the model copied to the peer machines. The head plans each rank's
shard and ships only that, through the TP blob cache: **~57.5 GiB per node at TP=2** and **~46.8 GiB
at TP=4** for the EXL3 pack, and 66.0 GiB for NVFP4 (was 98.5 replicated). TP=2 output is
token-identical to the previous path for both formats, and the live TP=4 bank is identical.
`--tp-sync-only` runs a node that syncs and exits.

### Default changes (output-changing on long context)

- **`--prefill-chunk 4095`**, up from 2048 — a large part of the TP=4 prefill gain. Ragged-tail
  folding is always on; `--prefill-chunk 2048` restores the previous grid.
- **`--qsa-key-rope full`** — the EXL3 indexer had left rotary dims 32..63 of every pooled key at
  zero since the sparse-attention work, while the reference fills them. `half` keeps the old bytes
  for A/B comparison only.

Plus minor bug fixes and optimizations, including a clear error instead of a panic when `--model-dir`
does not exist.

## v0.7.0 — EXL3 packs (Qwen3.8-Flash-Next), TP=2 serving, CLI-only options

A large release: a second quantization format with its own model family, two-node serving as a
shipped mode, and one breaking change to how the engine is configured.

### New: EXL3 packs are served directly

`--model-dir` now accepts an **EXL3 (ExLlamaV3 trellis)** pack alongside the NVFP4/FP8 families.
The model shipped on this path is **Qwen3.8-Flash-Next**: a 125B MoE with ~6B active parameters,
a Gated DeltaNet + Qwen Sparse Attention hybrid, 512 experts (top-10), a 51B n-gram embedding
table, an MTP draft head, and a 262,144-token context.

- The weights are published as
  **[doth4580/Qwen3.8-Flash-Next-EXL3-3.05bpw](https://huggingface.co/doth4580/Qwen3.8-Flash-Next-EXL3-3.05bpw)**
  — EXL3 3.05 bpw, about 85 GB including the 32.6 GB n-gram table.
- Native MTP speculative decoding is built in: greedy output is bitwise identical to
  non-speculative decoding, and sampled output is distribution-exact.
- **TP=1 and TP=2 only.** Any other world size exits with `--tp 2 only`.
- **Text only — no vision on the EXL3 path** (image requests are rejected). Vision remains on the
  NVFP4 / Qwen3.8 27B path added in v0.6.0.
- q8 KV by default, prefix cache with prefill checkpoints, penalties, streaming, loop detection.

### TP=2

TP=2 ships as a served mode: sequence-parallel prefill, a vocab-parallel LM head, and prefill
communication overlap — each on by default with an `off` value. Measured on two GB10 (DGX Spark)
over ConnectX-7, one request at a time, on the same engine code as this release but not on the
release binary itself; decode includes MTP speculation:

| | TP=2 | vs TP=1 |
|---|---|---|
| Decode, greedy (C code / Python / prose) | ~192 / ~160 / ~110 tok/s | ×1.38–1.42 |
| Decode, thinking-on, sampled | ~95 tok/s | ×1.36 |
| Prefill (2K / 32K / 128K / 256K) | 2,485 / 2,376 / 2,279 / 2,138 tok/s | ×1.56–1.69 |

### CLI-1: every option is a command-line flag (breaking)

- **The engine reads no environment variables.** Leaving a `GB10_*` variable set now refuses
  startup and names the replacement flag — for example `GB10_TP_GRAPH` suggests `--tp-graph`.
- `--print-config` prints every option's resolved value for a given command line; `--help-diag`
  lists the diagnostics and test drills.
- Migration table for every removed variable: **[docs/ENV_TO_FLAGS.md](docs/ENV_TO_FLAGS.md)**.
- Under TP every option is set on the head; the head ships its resolved options to the node, and a
  node's own command line carries only the per-box options.

### Speed

Prefill and decode levers on both paths, all bitwise: a pipelined expert kernel, a tensor-core GDN
chunk scan, hyper-connection fusion, router coalescing, sparse-attention selection, MoE gate/up
folding, and shared-expert overlap. On the EXL3 path, prefill TTFT is down roughly 27–30% against
the previous release. Autotune tables are now stamped with a build id and validated at load, so a
stale table cannot be applied to a changed build. NVFP4 W4A4 prefill kernels were contributed via
community PR #4.

### Reliability

- **A node that fails during boot now fails the head loudly** instead of leaving it hanging
  (PACK-FIX). The EXL3 TP pack-manifest check was removed by owner decision — each box loads its
  own pack at the path the head names.
- Autotune's preflight no longer requires specific machine addresses; it checks only that the GPU
  is idle (no other engine, no resident compute applications).
- Streaming loop detection, and CPU-affinity pinning by default.

Plus minor bug fixes and optimizations.

## v0.6.1 — `response_format` served again (quality-regression hotfix)

- **`response_format` requests are served, not rejected.** v0.6.0's P13 W2 change turned any
  `response_format` request into a loud HTTP 400; that was a hard failure of requests that used to
  succeed and cost measured quality on the public benchmark (the hardmode leg fell 150/176 →
  85/100 because json_schema scenarios returned `[server error 400]` instead of a verdict). Such a
  request is now served again, and the truth is advertised instead of hidden: every reply for a
  schema request carries an **`x-json-schema-enforced: none`** header, and the reason is logged
  loudly once per distinct reason. Nothing is silently ignored, and nothing that used to work is
  refused. (Schema *enforcement* remains off — it is not yet proven, and the switch is documented as
  gated on evidence.)
- **JSON-schema FSM fix.** An INTEGER prefix that can never return into range is now refused at the
  digit rather than at termination (previously an unbounded digit run after `maximum` was allowed).
  Deliberately narrow: `Ty::Integer` only, so a float case like `70e-1 == 7` is never rejected.
- Phase-B decode-mask arming + schema-blind-lane routing groundwork (inert while no lane carries a
  schema); 4 new tests. Minor bug fixes and optimizations.

## v0.6.0 — DFlash v1 drafter lane, DSpark, tool-render parity, TP/FP8 correctness

A large release: a new drafter serving lane, a new draft-model family, reference-exact tool
rendering, and a broad sweep of TP/FP8/losslessness correctness fixes.

### Drafters & speculation
- **DFlash v1 drafter lane** (`--spec-source dflash`) — the v1 drafter is now config-driven and
  wired into the API serving path, including under **TP=2/TP=4** (SPMD lane on every rank + artifact
  shipping). The draft round's attention was DRAM-bound on ~144× redundant KV traffic; a tiled
  DFlash attention kernel fixes it.
- **DSpark drafter support** — new kernel module `gpu_dspark` and a native DSpark round, with a
  signed ring-identity guard for the prefix carry.
- **DF2 block-16** — runtime `--df2-block 8|16`. Measured at TP=4: **+23.03% pooled / +9.47%
  row-medians** on the mission prompt set (GO on that workload; NO-GO on the recipe's own payload).
  Opt-in — the default stays block 8.
- **DF2 carry across a prefix-cache hit** — keeps DFlash2 in the draft seat across a cache hit
  (flag-gated, default off; measured slower at TP=4).
- **One artifact flag for every drafter**: `--draft-dir`; `--spec-source` is the only selector.

### Tool calling & template rendering
- **Reference-exact tool rendering** — `serde_json` and `minijinja` now preserve key insertion order
  (`preserve_order`). Without it the `<tools>` block was re-sorted and diverged from the reference
  (transformers/vLLM) at byte 83 on every tool-using turn (−345/−1,346 tokens on 12/52 tools).
- JSON-schema handling made explicit (loud floor) and tool-call serialization reworked.

### Serving & scheduler
- **Two-lane prefill cursor** (`--prefill-sched <inline|cursor>`) — the prefill window loop moved out
  of `admit()` into the step loop.
- Scheduler fixes: a silent request drop in the TP head path, and slot exhaustion now fails loudly.
- k8v8 multi-lane fast-fail; thinking toggle; `/health` telemetry.

### TP correctness
- **MTP head kept replicated under TP** — F6 sharding broke the draft chain (a TP=2/TP=4 serving
  regression).
- **F9 fix**: warm prefix-reuse stream divergence (TP2 FP8 27B).
- Batch-invariant verify-attention partition under TP + a binv probe.
- Shard block-128 FP8 (`W::Fp8Blk`) on the in-place path.
- TP-mode GDN state probe (`StateTp`) + logits/extent probe fixes for the sharded path.

### FP8 & losslessness
- **pf8 prefill OOB class fixed** — the full-attention / batch-mixer prefill-width W8A8 lane output
  buffers are padded to tp64 (FIX #2, #2b, #3, #4).
- **`ignore_eos` / `min_new` honored on every spec-path emit loop** (12 sites) — the stop rule is a
  property of the request, not the serving path (fixes the d2/d4-vs-d0 early-stop ladder failures).
- Depth-2 attention key partition keyed on the request constant; residual depth-2 verify losslessness
  diagnostics.

### Safety & observability
- **Release-live tripwire** — a pool/OOB tripwire with `--pool-census` and `--tripwire-selftest`.
- **`DISPATCH_ASSERT`** — the wide-verify dispatch rule as a machine-checked test.
- New `tel` module + telemetry reporting (e.g. dflash2-tree as its own `/health` mode).

### Launchers
- TP=4 production launcher (world=4) + a DRYRUN-asserting test; an NVFP4 TP=2 production launcher
  with a boot-identity line on both lanes. Minor bug fixes and optimizations.

## v0.5.5 — FP8 prefill levers on, prompt-truncation + max_tokens fixes

- **FP8 prefill levers now on by default.** The tensor-core flash-attention prefill
  (`GB10_FA_PREFILL`) and the tensor-core chunked GDN scan (`GB10_GDN_CHUNK2`) are now default-on for
  the FP8 path (value-checked; `=0` restores the legacy path). Big prefill speedup on the FP8
  Qwen3.8 27B configuration. NVFP4/MXF4/other-model paths are untouched.
- **`truncate_prompt_tokens` on `/v1/chat/completions`.** vLLM's left-truncation field now works on
  the chat path (it previously only worked on `/v1/tokenize`, so an over-length chat couldn't be
  rescued). Keeps the LAST `n` prompt tokens instead of the over-length 400.
- **`max_tokens`-omitted fix.** The scheduler now clamps before rejecting instead of rejecting when
  `prompt_len + max_new + depth + 8 > kv_stride` — a request that omits `max_tokens` no longer
  terminates with 0 tokens. Minor bug fixes and optimizations.

## v0.5.4 — Built-in OpenTelemetry, FP8 support, DFlash2 tree mode

- **Built-in OpenTelemetry.** `--otel-endpoint <URL>` streams OTLP/HTTP-JSON generation telemetry
  (the actual SSE chunk bytes, with `model.id` / `topology` / `request.id` / `token.index` / `event`
  attributes), off by default, near-zero decode interference. `request.id` is now the conversation
  key so a reply's turns join one continuous session; `generation.id` stays per-POST. Companion
  flags: `--otel-batch-size`, `--otel-batch-interval-ms`, `--otel-include-tokens`,
  `--otel-model-id`, `--otel-topology`.
- **FP8 support expanded.** Direct load of Qwen fine-grained block-128 FP8 (`weight_scale_inv`);
  DFlash2 FP8 drafter bake (`--df2-bake-fp8`) plus weight-only NVFP4 (`--df2-bake-nvfp4`);
  `--df2-quant-fidelity` gate tool. Fixed FP8 kernel bugs across `gpu_batch.cu`.
- **DFlash2 tree verification.** New opt-in `--spec-source dflash2-tree` mode (additive; MTP and
  DFlash2 unchanged). Minor bug fixes and optimizations.

## v0.5.2 — vLLM-compatible tokenize / detokenize endpoints

- **`POST /v1/tokenize`** — vLLM-compatible tokenization: `{tokens, count, max_model_len}`, a pure
  tokenizer call (no forward / KV / GPU). Accepts a `prompt` string or a chat `messages` array; the
  `messages` mode renders exactly as the chat path so its count equals `usage.prompt_tokens`. Empty
  prompt → `{tokens: [], count: 0}`; over-length → `400 context_length_exceeded`.
- **`POST /v1/detokenize`** — vLLM-compatible decode half of the pair (`{model, prompt}`) for
  exact-N prompt building.
- Added so our engine can be benchmarked more correctly. Minor bug fixes and optimizations.

## v0.5.1 — Vision robustness, reasoning-effort, graceful-load fixes

- **Vision generalization + boot fix.** The GPU vision tower now bootstraps opportunistically: a
  non-vision or geometry-incompatible model serves text-only instead of crashing at startup (fixes a
  v0.5.0 boot crash on non-27B packs). Vision is generalized across the Qwen3.5/3.8 VL family, so
  all vision-tower models serve images.
- **OpenAI `reasoning_effort`.** Full level table (`none/low/medium/high/xhigh/max`) with
  per-family normalization, plus `--reasoning-effort`; the `high` mapping no longer silently drops
  thinking (regression fix).
- **Tool-call + reasoning-mode fix.** Tool-call markup is held back in reasoning mode too, fixing a
  first-call double-emit leak.
- **Graceful model-load exit.** Corrupted / stale / wrong-format checkpoints exit with a clear
  actionable message instead of a panic/OOM/core-dump.
- **`--output-prompts [n]`** — human-readable chat-request logging; `--vision-cpu` now listed in
  `--help`. Minor bug fixes and optimizations.

## v0.5.0 — Vision support

- **Vision support.** Image input is now supported end-to-end on a GPU vision tower
  (`gpu_vision` kernels), with a `--vision-cpu` escape hatch to the CPU reference path. PNG/JPEG/WebP/GIF
  decoding added. The engine now ships and requires the `gpu_vision.ptx` kernel artifact in addition
  to the existing PTX set.
- **Better tool-call support.** A single canonical serializer now handles streaming and
  non-streaming tool-call output identically, repairs malformed tool-call tags, and no longer drops
  or leaks text around tool-call boundaries. New tool-call compliance and serializer test suites.
- **Prefill/TTFT optimizations.** New opt-in prefill levers (tensor-core flash-attention prefill,
  v2 W4A4 prefill GEMM, GDN tensor-core chunked scan), all env-gated **default off**, so the default
  serving path is unchanged. Minor bug fixes and optimizations.
- **Model-id fix.** `/v1/models` and responses now report the model card's `base_model`
  (e.g. `Qwen/Qwen3.8-27B`) instead of a local directory fragment. `--model-name` still overrides.

## v0.4.2

- Fix: accept OpenAI multipart `content` (string | array | null) to unblock agent clients that send
  content parts; request-schema only.

## v0.4.1

- Fix: `--draft-dir` is now mandatory only when `--spec-source` explicitly names a DFlash2 mode;
  plain-MTP launches no longer require it.

## v0.4.0

- **Qwen3.8 27B NVFP4** support with native **DFlash 2** speculative decoding, full 256K context.
- **TP=4** serving (plus TP=2 and single-node).
- New DSV4 / DFlash2 / DSpark / MXFP4 kernel set.
- README Update section with the Qwen3.8 27B performance table and live throughput traces; new
  `QWEN_27B_SETUP.md` and `MANAGING_CACHE.md` docs.

## v0.3.1

- **KAT-Coder** model support; supported-models table in the README.

## v0.3.0

- README generalization and load-pipeline features. Minor bug fixes and optimizations.

## v0.2.0

- **Tencent Hy3 (hy_v3)** family support, 4-bit KV cache, FR-Spec draft head, model-name family fix.

## v0.1.0

- Initial public release: from-scratch Rust + CUDA engine for Qwen3.5/3.6 on single and TP=2 GB10,
  with NVFP4/FP8 quantization, MTP speculative decoding, an OpenAI-compatible server, and prebuilt
  release binaries.
