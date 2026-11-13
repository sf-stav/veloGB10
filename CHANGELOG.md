## v0.7.9 — NVIDIA FP4-Local support, EXL3 long-context attention, prefix pool, and twelve community issues

The largest release since v0.7.4. It adds a second model family (NVIDIA's
[Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local)), rewrites
the EXL3 long-context attention path, adds a cross-lane prefix pool, and closes twelve issues and
pull requests from the community.

**Thanks to [@andersc](https://github.com/andersc)** for pull requests
[#30](https://github.com/sf-stav/veloGB10/pull/30) (per-token log probabilities),
[#31](https://github.com/sf-stav/veloGB10/pull/31) (JSON-schema constrained decoding) and
[#32](https://github.com/sf-stav/veloGB10/pull/32) (PLE prefill prefetch, fewer shuffles and
`ldmatrix` in the prefill MoE kernels), all ported with authorship preserved.

**Thanks to [@lesserevil](https://github.com/lesserevil)** for pull request
[#27](https://github.com/sf-stav/veloGB10/pull/27) (6–8-bit generic trellises and 6-bit expert
prefill).

**Thanks to [@jeffpeng3](https://github.com/jeffpeng3)** for pull request
[#36](https://github.com/sf-stav/veloGB10/pull/36) (`--kv-cache q4k`, a 4-bit packed KV cache) and
for issues [#33](https://github.com/sf-stav/veloGB10/issues/33),
[#34](https://github.com/sf-stav/veloGB10/issues/34) and
[#35](https://github.com/sf-stav/veloGB10/issues/35).

**Thanks to [@Gcool100](https://github.com/Gcool100)** for issue
[#26](https://github.com/sf-stav/veloGB10/issues/26) — a detailed, reproducible TP=2 report that
found three separate defects.

**Thanks to [@YulRustem](https://github.com/YulRustem)** for issue
[#28](https://github.com/sf-stav/veloGB10/issues/28) (grammar-constrained decoding).

**Thanks to [@liorm0505](https://github.com/liorm0505)** for issues
[#37](https://github.com/sf-stav/veloGB10/issues/37) (the OpenAI `developer` role) and
[#38](https://github.com/sf-stav/veloGB10/issues/38) (PLE sidecar K=8 / turboderp's ng8 packs).

### What this build does NOT do for the posture

Read this first. The posture is `--max-seq-len 262144 --max-batch 4 --max-tokens 65536
--prefix-cache on --mtp=auto`, thinking on, 1–4 concurrent streamed requests.

- **FP4-Local is TP=1/2/4 on GB10 only.** It is a separate model family; nothing here changes the
  EXL3 or NVFP4 paths' numerics.
- **`--kv-cache q4k` is EXL3-only.** It is not available on the NVFP4/BF16/FP8 servers.
- **q4k at TP=2/TP=4 was never run on hardware.** TP=1 is measured; the multi-machine cells are a
  named gap (`docs/KV_CACHE_Q4K.md` §8).
- **EXL3 fractional expert widths (K=2.5/3.5) are TP=1 for all four packs, TP=2 for 3.68+2.68
  only.** TP=4 is owed.
- **Fractional-width prefill runs on the wide-tiles HALF body** — there is no fractional
  `xq_moe_pf` body yet. Correct, not optimal.
- **`--json-schema-enforce` is a V1 subset** (EXL3 TP=1). An unsupported schema falls back with a
  warning rather than refusing.
- **`--chat-template` and `--model-dir <repo-id>` are new surfaces**; the repo-id form resolves
  from the local HF cache only and never touches the network.
- **Lane packing still never packs a request carrying a sampler penalty** (the v0.7.4 rule), and
  TP=4 lane packing was exercised by hand, not by an automated gate chain.
- **No long soak and no 262K-context cell was run for this release.**

### New: NVIDIA Qwen3.8-Flash-Next-FP4-Local (NVFP4)

A second model family on the EXL3 trunk: ModelOpt NVFP4 packs, CSF scale handling, a dedicated
forward path (`src/fp4l/`, `src/exl3_forward/fp4l_fwd.rs`, `kernels/gpu_fp4l.cu`), and the shared
n-gram (PLE) read-ahead. Flags: `--fp4l-plain`, `--fp4l-norms`, `--fp4l-e2m1`, `--fp4l-lm-head`,
`--fp4l-draft-head`, `--fp4l-prefill-dense`, `--fp4l-mtp-sidecar` (`auto|off` — the pack decides;
a sidecar is used exactly when the pack ships an `mtp-*.safetensors`). A ModelOpt pack with
unserved formats is refused **before** TP shipping, with a plain message.

Quality receipts: GSM8K-250 FP4-Local 96.8 % vs EXL3 4.05 96.0 %; against the reference the
most-exact fraction is 0.0098, closer than NVIDIA's own recommended DGX Spark vLLM recipe
(0.0155 / 0.0113) on the same samples.

### New: EXL3 long-context attention (split-K dense, on by default)

`--exl3-dense-splitk` is **on by default** for full-attention packs. It decodes q8 K/V once per
block in the trunk split-K, verifies from 2 rows with a shared read, and pipelines prefill on
tensor cores (`--exl3-dense-pf-tc`, cross-checked by `--exl3-pf-tc-xcheck`). `--exl3-dense-splitk
off` keeps the old kernels for A/B.

### New: prefix pool

Prompt state is kept in device memory across lanes: when a lane is taken over, the outgoing
conversation's positional rows, recurrent snapshot and draft-head planes are copied device-to-device
into a pool shared by all lanes, and a later prompt starting with a kept prompt is restored with one
copy instead of a re-prefill. Entries carry the lane's WP16 checkpoints; the memory floor shifts out
one LRU entry at a time; the pool survives memory pressure and shares rows between related prompts.
Flags: `--prefix-pool-gb`, `--prefix-pool-floor-gb`.

### New: self-update and boot version notification

`--update` downloads the latest release tarball, verifies it against `SHA256SUMS`, and atomically
replaces the running binary (`.prev` backup kept). It refuses on a dirty tree or when the executable
lives in a checkout or `target/`, and refuses on a TP `--node`. The boot banner now reports when a
newer release exists; `--no-update-check` disables that. Zero new dependencies, zero environment
reads.

### New: CPU idle waits

`--gpu-wait spin|hybrid` sleeps until just before each call site's predicted end and then spins
(~0.1–0.2 core instead of a full core); `--proxy-idle-us` makes the TP proxy's idle tier actually
wait (kernel WFE sequence) with deep sleep after 1 s. MTP calibration and bench timers always spin
(`SpinScope`), so measurements are unaffected. Both features are opt-in.

### New: EXL3 width coverage — the bit-width ladder is now complete

The EXL3 fast-lane kernels now cover **every integer width from 2 to 8 bits, plus the fractional
1.5 / 2.5 / 3.5**, on both the dense/quad path and the expert (routed and MTP) path. Fractions on
the expert path are the piece that was missing before, not the low integers:

| K (bpw) | `k2` | quad/dense | experts |
|---|---|---|---|
| 1.5 | 3 | already (v0.7.8) | **new** |
| 2 | 4 | already (v0.7.8) | **new** |
| 2.5 | 5 | **new** | **new** |
| 3 | 6 | already (v0.7.8) | yes |
| 3.5 | 7 | **new** | **new** |
| 4 | 8 | already (v0.7.8) | yes |
| 5 | 10 | already (v0.7.8) | yes |
| 6 | 12 | already (v0.7.8) | yes |
| 7 | 14 | **new** | **new** |
| 8 | 16 | **new** | **new** |

So the accurate headline is: v0.7.8 served 2–6 bit on the dense path; **v0.7.9 adds 7- and 8-bit
and makes the fractional widths (1.5 / 2.5 / 3.5) work on the expert path as well as the dense
one.** Turboderp's fractional packs (2.18 / 2.68 / 3.18 / 3.68 bpw) and the ng8 packs (which need
K=7/8 experts to serve *with* speculation rather than with the draft head skipped) both became
servable through this.

- **K=7 and K=8 routed and MTP experts** (`k2` 14/16) — what lets turboderp's ng8 packs serve
  *with* speculation rather than with the draft head skipped.
- **Fractional expert widths K=2.5/3.5** (`k2` 5/7) — turboderp 2.18 / 2.68 / 3.18 / 3.68 bpw.
- **K=1.5 and K=2 experts** admitted (coop case-2 bodies); the expert gate was widened to
  `k2 3|4|5|6|7|8|10|12|14|16` and a width the engine cannot serve now **refuses loudly at load**
  naming the nearest working path, instead of reaching a `default: break` and doing nothing.
- **MTP draft sidecars** with plain-map mirrors and a mixer-patch fallback.

A caveat that belongs with this: the fast lane is a *kernel* claim. Fractional expert widths are
served at TP=1 (all four turboderp packs) and TP=2 (3.68 + 2.68 only); TP=4 remains owed, and
fractional *prefill* runs on the wide-tiles HALF body rather than a fractional `xq_moe_pf` body.

### New: other surfaces

- `--model-dir` accepts a **Hugging Face repo id**, resolved from the local cache (no network).
- `--chat-template <PATH>` serves a custom chat template, winning over `chat_template.jinja` and
  `tokenizer_config.json`.
- `--kv-cache q4k` — 4-bit packed KV on the EXL3 path (144 B/row at head_dim 256 vs q8's 272),
  admitted to lane packing.
- `--seedless-seed` — a sampled request that sends no seed gets a fresh one drawn on the head so
  every TP rank agrees.
- `--reasoning-config` — vLLM-compatible `thinking_token_budget`.
- `--version`, a configuration block and boot progress on the banner.
- RDMA rails are auto-detected; `--rdma-dev` is needed only on a box with two active port pairs.

### Fixes

- **#26 — TP=2 packs with 5-bit experts failed at node boot.** Three separate defects: the
  host-routed MoE prefill was not EP-wired (now unreachable under EP, with a graceful fallback);
  the PLE sidecar rejected 3-bit n-gram tables (now `K = 1..=8`); and a boot failure double-panicked
  in `cudarc`'s destructor (now a clean `eprintln` + exit). A TP head width preflight now refuses an
  unservable pack in seconds, before the 57.5 GiB first ship. Verified on the reporter's own packs
  at TP=2: 3.05 bpw 47.1 tok/s, 4.05 bpw 48.4 tok/s, both coherent.
- **#27 — 6–8-bit generic trellises and 6-bit expert prefill** (PR by @lesserevil).
- **#28 / #31 — grammar-constrained decoding** via `--json-schema-enforce`.
- **#30 — per-token log probabilities** (`logprobs` / `top_logprobs`), keeping speculation.
- **#32 — PLE prefill prefetch** and fewer shuffles / `ldmatrix` in the prefill MoE kernels.
- **#33 — `--model-dir` accepts an HF repo id.**
- **#34 — `--chat-template`.**
- **#35 — `xq_attn_decode_group` resolved one KV slot for the whole group**, so a packed verify's
  segment B read and wrote segment A's K/V. Now resolved per row.
- **#36 — `--kv-cache q4k`** (PR by @jeffpeng3).
- **#37 — HTTP 500 on the OpenAI `developer` role.** Normalized to `system` at request ingest; the
  pack's template stays strict and unknown roles still fail loudly.
- **#38 — PLE sidecar rejected K=8 rows**, blocking turboderp's self-calibrated ng8 packs. The
  admission set is now `K = 1..=8` and pinned by a test so a narrowed check cannot return unnoticed.
- **TP boot barrier before MTP calibration.** Ranks finish per-rank setup up to ~20 s apart, so the
  calibration's first all-reduce ran under the 10 s per-step watchdog and fast ranks declared slow
  ones dead. Seen on 27B-FP8+DF2 TP=4 and 122B-MIXED+DF2 TP=4.
- **EXL3 packs now ship to TP nodes**; `--kv-cache bf16` falls back to f16 on EXL3; TP nodes install
  the head's DFlash2 draft block.
- **1.66 bpw k2=3 serving** fixed (no-op root cause plus fractional bodies), with a k2=16 width
  guard.
- **Single-node fit guard** now uses measured transient accounting; the 122B NVFP4 TP=1 loads
  without `--load-force`.
- **EXL3 TP=1 clamps the context** instead of refusing when not even one lane fits.
- **Sampler fixes**: whole-output penalties, random seeds for seedless requests, Qwen3.5 card
  defaults by model identity.
- **Probe bitwidths** section 7 now compares against the unrounded f32 oracle.

### Verified

Unit suite 1112 passed / 0 failed. FP4-Local identity and quality checks at TP=1/2/4; EXL3
split-K attention A/B on hardware; q4k pack probes (`--probe-exl3-pack2` r_mean 1.469 / 0 differing
live-state words, `--probe-exl3-packstate` PASS) and a serving boot with 2 concurrent sampled
requests showing `velogb10_lane_pack_active{reason=""} 1` over 70 packed lane rounds; #26 TP=2
end-to-end on the reporter's packs. Not re-run: TP=4 lane packing under an automated gate chain, any
262K-context cell, and q4k at TP=2/4.

---

## v0.7.8 — fix: `top_k` client compatibility (PR #29 by @andersc) + the NVFP4 sampler cap made explicit

Fixes the client-compatibility blocker reported in [#29](https://github.com/sf-stav/veloGB10/pull/29):
a request with `top_k: -1` — the vLLM/llama.cpp/TGI convention for "no top-k truncation" — was
rejected with HTTP 400 (`top_k must be non-negative`), and `--cpu-sample` panicked outright on
`top_k: 0` or `top_k` ≥ vocabulary.

**Thanks to [@andersc](https://github.com/andersc)** for the original pull request
[#29](https://github.com/sf-stav/veloGB10/pull/29).

What changed:

- `top_k` below 0 is accepted on `/v1/chat/completions` and `/v1/completions` and means **no top-k
  truncation**; the full unsigned range is parsed.
- The CPU sampler (`--cpu-sample`) treats `top_k` 0 as no truncation and no longer panics on
  `top_k` ≥ vocabulary.
- The NVFP4-family GPU samplers now serve `top_k` 0 / > 64 as **top-64** instead of silently
  falling back to argmax. The cap is announced once per boot and counted in
  `velogb10_top_k_capped_total` (shown by `scripts/owner_check.py`). The EXL3 server and
  `--cpu-sample` apply no such limit.
- Test hygiene: the `--help` snapshot fixtures (stale since v0.7.6/v0.7.7) were refreshed and the
  direct-print baseline rows updated, so a clean checkout passes the test suite again.

Receipts (merge gates, dev report §9): unit suite 986 passed (the one failure is the pre-existing
environment-dependent case, unchanged). Flash-Next TP=1 at the default posture: all 12 recorded
greedy hashes reproduced byte-identically, `top_k` -1 / 0 / u64::MAX return 200 on both endpoints;
on the 4B NVFP4 pack the GPU cap verified (`top_k` 1 = greedy, 0 / > 64 = top-64, counter 0 → 3,
one boot line); `--cpu-sample --mtp=off`: every cell 200, no panic. Non-negative `top_k` values are
byte-identical to v0.7.7 on the served EXL3 path. Not re-run at TP=2 / TP=4.

---

## v0.7.7 — fix: TP=2 MTP lockstep crash on first generation (the `--mtp` blocker)

Fixes the v0.7.5/v0.7.6 crash reported on the issue tracker: with MTP enabled at TP=2, the first
generation dies with `TP pre-verify FAILED` / `TP agree() FAILED at mtp-round` and exit 70.
`--mtp off` worked around it; v0.7.4 was unaffected.

**Root cause:** the fractional-K kernel work re-keyed the EXL3 decode GEMM kernels by `k2` (= 2K)
instead of `bits`, but five launch sites still passed `q.bits`. Those launches silently dispatched
to nothing, so the MTP head's prime / re-prime GEMMs computed nothing on affected packs, and the
ranks diverged at the next mtp-round digest check. The fix passes `q.k2` at the five sites — one
token per site, no kernel change, and no behaviour change for any launch that was already
computing.

Also in this release, all from the same investigation:
- the `--otel-*` option set is allowlisted; the previous `unknown option --otel-include-tokens`
  warning was a false positive — the value **was** applied, and the flag can stay in your launch
  files;
- the probe suite itself was repaired (several EXL3 probe launch sites carried the same stale-key
  bug and had been reporting false kernel-equivalence FAILs);
- on an `mtp-round` agree failure the server now dumps per-rank sampler state alongside the
  watchdog dump;
- `Cargo.lock` now records the crate version in lockstep with `Cargo.toml`.

**What this build does NOT do:** no performance work is included (every verification number in the
investigation is a correctness receipt); lane packing is unchanged (requests carrying a sampler
penalty still never pack — the v0.7.4 rule); TP=4 was verified only at the reported posture
(12 requests per cell); no long soak and no long-context cell was run. On packs whose MTP head
uses a 5-bit key, the head's prime / re-prime now computes correctly, which can only raise
speculative acceptance.

Verified: unit suite (26 k2/WP21/PSK/chain/quad/option-guard tests), losslessness x5, the
`--probe-exl3-binv` suite, TP=2 end-to-end at the reported flags (16/16 clean, speculation
engaged), TP=4 12/12 on two packs.

## v0.7.6 — LOADFAST staged loader for Naive TP + TP cold-boot bring-up fix

_Released 2026-10-27 · tag `v0.7.6` · asset `velogb10-v0.7.6-gb10-sm121.tar.gz` (+ `SHA256SUMS.txt`, `PROVENANCE.txt`) · binary sha256 `cf1501159f68ae0cb6f3c748cab8759285f3b144c7e363dbb456c20cf3e7a265`_

### What is in this build

- **LOADFAST: staged-layer loader for Naive TP.** After the first boot, ranks stage decoded weights
  from a per-rank local cache instead of re-decoding the pack: sha256-verified, resumable, staged by
  a claim-bounded parallel pipeline (deadlock-free by construction; unit-test regression included),
  and rank-0 stages from the local cache on subsequent boots. New flags: `--naive-load-cache`,
  `--naive-load-verify`, `--naive-load-workers`.
- **TP cold-boot bring-up fix (ISSUES 153).** A cold-cache TP head ships its world-1 ranks
  sequentially (about 19 minutes at 0.10 GB/s), and the old fixed dial/accept budgets (300 s dial /
  600 s accept / 20 s for world == 2) expired mid-ship: every node died with "nway connect:
  Connection refused" on the FIRST cold boot (a warm blob cache hides it — a warm sync takes
  seconds). The dial/accept loops now retry while the head's control stream is alive — the node
  registers its fd and the shim peeks it with MSG_PEEK without consuming protocol bytes — abort the
  instant the stream closes or resets, and keep a 1-hour hard cap with 60-second progress notes.
  Unit test: `head_liveness_peek_sees_peer_close`.
- **Issue #19 corrected: YaRN is already supported** (E5/YARN1 + YARN512K receipts) — the close-out
  retracted the wrong answer; no code change needed.
- The Flash-Next TP scaling assessment (agent-trace profile, per-turn budget, static findings
  S1–S6: double render+tokenize per request on the head, short-suffix chunk fixed cost, TP=4
  GPU-side receive, sampled full-row logit gather, deep-context KV-pair attention) is published in
  the dev reports; it proposes levers, none of which are code in this build.

### What this build does NOT do (plain words)

- LOADFAST pays only on REPEAT boots of a Naive TP server from the same pack on the same rank
  (the first boot is unchanged); no startup numbers are claimed here.
- The cold-boot fix changes retry/abort timing only — no bandwidth or protocol change.
- Everything the v0.7.5 notes list as not-done remains true unless addressed above.

## v0.7.5 — EXL3 everywhere + the Naive-N0.5-Flash family + DFlash2 drafts under TP everywhere + reduced draft vocabulary

_Released 2026-10-27 · tag `v0.7.5` · asset `velogb10-v0.7.5-gb10-sm121.tar.gz` (+ `SHA256SUMS.txt`, `PROVENANCE.txt`) · binary sha256 `7ff7ef98e8909ed408589623b018324c509b5c22d6b9ccd8636800ec512038f4`_

### Delta vs v0.7.4 — the full release delta list (2026-10-12 → 2026-10-27; 8 release commits, 155 files changed)

#### New model / serving support
1. **All five EXL3 quantization packs serve end-to-end** — dense 27B @ 3.05 & 3.5 bpw, dense 9B @ 3.5, 35B-A3B MoE @ 3.5, 27B Flash-Next @ 3.05. TP=2 on all five; TP=4 on 9B / 35B-A3B / 27B-3.05 / 27B-3.5 (losslessness ×5 each, both test machines). *(answers issues #9, #10)*
2. **Naive-N0.5-Flash family** (new architecture): TP=2/TP=4 serving, DSpark-drafter speculation, expert-parallel MoE, lane reservation, prefix reuse, TP levers.
3. **Flash-Next 1.5 / 2.0 bpw packs** (KFIX1 GEMM pipeline + draft-round CUDA graph).

#### New features
4. **Reduced MTP draft vocabulary — `--draft-vocab`** (FR-Spec): draft head built over an id-order vocab subset for all Qwen NVFP4/FP8/bf16 heads. **Default ON for Qwen3.5/3.6/3.8** (owner decision 2026-10-27), off elsewhere; known trade: non-Latin scripts mostly outside the subset (documented in README).
5. **DFlash2 drafts head-authoritative under TP on every path** — sampled drafts carry their drawn weights + per-position 16-candidate table; drafter ships through the EXL3 TP sync; boot agreement hashes the DF2 outcome; `--df2-k`, `--df2-mask-override`; bake inventories follow artifact geometry.
6. **KV-spill hardened** (TP=4 context hand-off; still off by default): capture vote accepts its own frames, failed vote stops admission with a real message, whole-plane snapshots, payloads carry only used rows, single-strip restore (every capture was rejected before), restore resends reasoning+content, peer-miss names its reason. TP=4 verified: capture→restore across 4 ranks, 959 prompt tokens restored with no re-prefill.
7. **Lane reservation** — one scoped admit per TP iteration with apply-time requeue; never refuse.
8. **Unsent sampler fields resolve from the pack's `generation_config.json`** on both servers, with one boot line naming the resolved values.

#### Performance / responsiveness
9. **`--prefill-interleave` defaults to 1** — a joining stream no longer stalls running streams (9.1 s → 1.7 s join at TP=2); `0` restores old behavior; engagement counters on `/metrics` + `owner_check.py`. *(answers community PR #23, which proposed the same)*
10. **Large prompts tokenize off the single request thread**; the debug generation log no longer re-tokenizes every reply.

#### Fixes
11. **Graceful `--max-batch` boot** — the server starts on the most lanes that fit, never truncates context, prints one line saying what it serves and why, counts what it queued on `/metrics`. *(the OOM-at-boot class, issue #1)*
12. **Streaming fix** (ISSUES 149 / issue #18): Qwen no-think render with `reasoning_effort low|high` no longer streams the reasoning label as visible content.
13. **EXL3 DF2 tap fix** — hyper-connection mean capture; band-attention smem bounded by the window (drafter conditioning correctness).
14. **Four static-review fixes** in the DFlash2 ring/verify paths (a measured re-alignment was rejected and backed out — it cost sampled acceptance).
15. **MTP calibration cache keyed on `--draft-vocab`** — a full-vocab calibration overpriced the reduced head; each setting keeps its own calibration file.
16. **`--quantize --help` documents the recipe syntax** and the three recipes (`all`; `all,gdn:fp8` — default; `all:fp8`). *(issue #3)*
17. Issue #2 (35B-A3B NVFP4 serving) verified in the matrix; #7/#8 (server surface) and #20/#24 (vision) were closed just before the cut and are cross-referenced.

#### Packaging / release format
18. **EMB1 one-binary release** — kernels embedded in the binary, tarball ships no external PTX (this is the first release cut on that contract).

#### Known limitations stated in this release (plain words)
- Serving-matrix numbers are **single-rep smokes**, not benchmarks; the README's measured tables remain the reference.
- **Naive-N0.5-Flash @TP=2 and 35B-A3B @TP=2 deterministically emit no visible answer** for a pure-code prompt at an 8192-token budget (thinking-dominated; prompt-seeded sampler reproduces the stream) — TP=1/TP=4 are fine; a higher budget completes.
- Some Flash-Next / 27B-exl3 cells hit the 8192 budget mid-answer (truncated code).
- **TP=2 head's over-cap counter is still not wired** (TP=1 counts every over-cap arrival; TP=2 serves correctly but never increments).
- `--kv-spill` off by default; TP=4 verified by the quick rung only — the full LANERES3 ladder (concurrency cells + churn ×2) was descoped.
- Reduced draft head: full **sampled** benchmarking still pending; non-Latin workloads don't benefit (falls back to full-vocab behavior).

Re-cut of the previously deleted v0.7.5: this build contains everything that cut contained (all
five EXL3 packs serving everywhere, graceful `--max-batch`, the stable cut) plus everything since
— the Naive-N0.5-Flash family, `--prefill-interleave` on by default, DFlash2 drafts under TP on
every path, and the KV-spill hardening. Verified end-to-end before staging: a 12-cell serving
matrix across four model families at TP=1/2/4 with the owner's exact launch flags (9 bootable
cells green — boot, gates, streamed generation with per-cell decode tok/s and code-emission
checks), the LANERES3-TP4 quick rung (TP=4 boot+serve, KV-spill capture/restore across all four
ranks, lane reservation), and a 969/0 unit suite on the naive port before it merged.

This is the **final v0.7.5 cut (2026-10-27)**: on top of all of the above it folds in the reduced
draft vocabulary work that landed after the re-cut candidate (below), so one release carries the
whole line.

### Reduced draft vocabulary (`--draft-vocab`) — new in this final cut

- **FR-Spec reduced MTP draft head + `--draft-vocab`**: the draft head can now be built over a
  reduced, id-order subset of the vocabulary (special-token tail) instead of the full vocab, for
  all Qwen NVFP4/FP8/bf16 heads (ISSUES 148). Started opt-in; since 2026-10-27 (owner decision)
  **`--draft-vocab` is ON by default for the Qwen3.5/3.6/3.8 family**; off elsewhere (hy_v3
  unverified, qwen4_exp never). `--draft-vocab off` disables it; an explicit setting keeps its
  own MTP calibration file.
- **The known trade, in plain words**: the reduced draft head covers an id-order subset, so
  non-Latin scripts are mostly OUTSIDE it — drafting for such text does not benefit and falls
  back to full-vocab behavior. Documented in the README.
- **MTP calibration cache is keyed on `--draft-vocab`**: a full-vocab calibration overprices the
  reduced draft head; each setting keeps its own calibration file (default file name and stamp
  unchanged).
- **Streaming fix (ISSUES 149)**: a Qwen no-think render with `reasoning_effort low|high` no
  longer streams the reasoning label as visible content (the streaming symptom reported in issue #18).
- **`--quantize --help` now documents the recipe syntax** and the three documented recipes
  (`all`; `all,gdn:fp8` - the default; `all:fp8`). Answers the question in issue #3.
- Verified so far: greedy-identical output with the flag on vs off across 12 configurations
  (TP1 on .11, TP2 on .11+.13); the default-on flip is owner-ordered (2026-10-27). Full sampled
  benchmarking of the reduced head is still pending.

### What is in this build

- **All five EXL3 packs serve** (from the first v0.7.5 cut, unchanged): dense 27B at 3.05 and
  3.5 bpw, dense 9B at 3.5, the 35B-A3B MoE at 3.5, the 27B Flash-Next at 3.05 — losslessness
  probe x5 each, on both test machines; TP=2 on all five, TP=4 on 9B/35B/27B-3.05/27B-3.5.
- **`--max-batch` boots anyway** — the server starts on the most lanes that fit, keeps the
  context length (never truncated), prints one line saying what it serves and why, and counts
  what it had to queue on `/metrics`.
- **The Naive-N0.5-Flash family (new in this re-cut):** TP=2/TP=4 serving, DSpark-drafter
  speculation, expert-parallel MoE, lane reservation, prefix reuse, TP levers.
- **`--prefill-interleave` defaults to 1** — a joining stream no longer stalls the running
  streams (9.1 s to 1.7 s join at TP=2); `--prefill-interleave 0` restores the old behavior;
  engagement counters on `/metrics` and in `scripts/owner_check.py`. Answers community PR #23, which proposed the same feature.
- **DFlash2 drafts are head-authoritative under TP on every path** — sampled drafts carry their
  drawn weights and the 16-candidate table per position, greedy drafts too; the drafter ships
  through the EXL3 TP sync and the boot agreement hashes the DF2 outcome; `--df2-k` caps verify
  width; `--df2-mask-override`; bake inventories follow the artifact geometry.
- **KV-spill hardened and off by default:** the TP=4 capture vote accepts its own frames and a
  failed vote stops admission with its real message; whole-plane snapshots; payloads carry only
  the rows the conversation uses; single-strip restore (every capture was rejected before);
  restore resends reasoning+content; the peer-miss names its reason. Verified at TP=4:
  capture-then-restore across 4 ranks, 959 prompt tokens restored with no re-prefill.
- **Unsent sampler fields resolve from the pack's `generation_config.json`** on both servers,
  with one boot line naming the resolved values.
- **Large prompts tokenize off the single request thread**; the debug generation log no longer
  re-tokenizes every reply.
- **Lane reservation:** one scoped admit per TP iteration with apply-time requeue — never refuse.
- **Flash-Next 1.5/2.0 bpw packs** (KFIX1 GEMM pipeline, draft-round CUDA graph) and the EXL3 DF2
  tap fix (hyper-connection mean; band-attention smem bounded by the window).
- Four static-review fixes in the DFlash2 ring/verify paths (one measured re-alignment was
  rejected and backed out — it cost sampled acceptance).

- **Community items this release answers**: PR #23 (`--prefill-interleave` - proposed upstream,
  shipped natively); issues #9 and #10 (EXL3 pack loads and the TP draft-hash instability, both
  closed during bring-up and re-verified by this release's matrix); issues #1 and #2 (the
  OOM-at-boot class - the graceful `--max-batch` boot - and the 35B-A3B NVFP4 class, verified
  serving in the matrix). Server-surface items #7/#8 and vision items #20/#24 were closed before
  this cut.

### What this build does NOT do (plain words)

- **The serving-matrix numbers are single-rep smokes**, not benchmarks — the benchmark tables in
  this README remain the measured reference.
- **Naive-N0.5-Flash at TP=2 and the 35B-A3B at TP=2 deterministically produce no visible answer
  for a pure-code prompt at an 8192-token budget** — the run is thinking-dominated and the
  prompt-seeded sampler reproduces the same stream; TP=1/TP=4 of the same models emit the answer.
  A higher token budget completes them.
- Some Flash-Next and 27B-exl3 cells hit the 8192-token budget mid-answer (truncated code).
- **The TP=2 head's over-cap counter is still not wired** — the TP=1 server counts every arrival
  beyond its lane cap; the TP=2 head queues and serves the same traffic correctly but never
  increments the counter.
- **`--kv-spill` is off by default** — turn it on explicitly; at TP=4 it is verified by the quick
  rung (capture/restore across 4 ranks), not by the full concurrency ladder.
- **The full LANERES3-TP4 ladder (concurrency cells + churn x2) was descoped** — TP=4 is verified
  by the 3-rung quick test (all PASS) plus the serving matrix's TP=4 cells.
- Everything the v0.7.4-era notes list as not-done remains true unless addressed above.


## v0.7.4 — tool-call parsing fixes, speculation that re-arms, per-row attention regime, vision-tower recovery, option-parsing guards, lane packing on by default, first stream chunk at the first token

This is a fix release for the EXL3 (Qwen3.8-Flash-Next) serving path, with a few opt-in
additions. It fixes defects found after v0.7.3: speculative decoding could switch itself off for
the rest of a long request; a request's greedy text could depend on what else was being served;
tool-call parsing returned wrong answers for responses that quote tool calls; a GPU vision-tower
error disabled images until restart (issue #20); and the first streamed chunk reached a
client's clock long before any output existed. It makes things that used to fail silently fail
loudly or say what they did (an option that swallows the next flag, a head and a node of
different builds, a `--max-batch` above the server's 16-request cap), adds KV occupancy gauges
(issue #24), ports five community pull requests (#15, #16, #17, #21, #23), turns on by default
`--lane-pack` (two concurrent requests share one speculative verification; `--lane-pack off`
restores the previous behaviour), and adds a perplexity measurement tool.

**It is not the speed release for sampled requests.** Most clients send no sampler settings,
which means sampling, and a request that samples is slower than the greedy figures published
for earlier releases. That work is the next release. "What is not included in this build" and
"What's next", at the end of this entry, say what is and is not covered, and on what kind of
request each figure was measured.

Several defaults change what a client receives in narrow cases. They are listed together in the
next section, each with the flag that restores the v0.7.3 reading.

**Two defaults were set at the cut by the maintainer:** `--lane-pack` is on (listed in the table
below with its restore flag), and `--tool-unclosed` stays `drop`, the v0.7.3 reading
(`--tool-unclosed complete` is available).

**Artifacts.** `velogb10-v0.7.4-gb10-sm121.tar.gz` (the binary, all 19 PTX files and the two TP
launchers; binary sha256 `21a54db93d53c067e5045a2fcf49f1e92f28d02f5089f0cbba91312663b49f0b`),
`SHA256SUMS.txt`, `PROVENANCE.txt`. Built from the public tree at commit `f06f1c1` (the
release source), a clean checkout; the kernel files and the PTX are byte-identical to v0.7.3.

### Behaviour changes at default flags

| what a client sees at default flags | restore the v0.7.3 reading with |
|---|---|
| Greedy text can differ from v0.7.3 for a request whose context crosses about 2,051 tokens under speculation, and for a short request that shared a decode step with a longer one (the v0.7.3 bytes of those depended on timing). Details under "a request's greedy text could depend on what else was being served" | `--qsa-row-regime off` |
| A `</tool_call>` inside an argument value is text, and so is a quoted complete block; before, the first `</tool_call>` ended the block and cut the call there | `--tool-block-end first` |
| A string argument loses exactly one wrapping newline instead of all of them (a file the model ends with a blank line arrives with its final newline) | `--tool-value-trim all` |
| **`--lane-pack` is ON by default** (it was off): two concurrent requests share one speculative verification; greedy bytes are unchanged, sampled requests draw from the same distribution; it holds about 1.8 GiB more device memory per machine (a second 16-row scratch and its graphs, captured at boot), also for a single request; a posture it cannot serve boots with it inactive and one line saying why; a request with a penalty or an image never packs | `--lane-pack off` |
| Streamed call whose argument holds `<think>`: no raw call markup as content and no split into reasoning. Thinking off and an argument holding `</think>`: the call is kept (it was thrown away). A Qwen call whose value mentions the Hy3 call tag stays a Qwen call. A JSON-bodied block whose string quotes `</tool_call>` is read whole | none: defect fixes |
| A streamed chat response carries an SSE comment line `: keep-alive` every 10 s while it waits in the queue, prefills or buffers a tool call; the role chunk leaves with the first generated token (data events, bytes and order are unchanged) | `--sse-keepalive 0` for the lines; the role chunk's time has no flag |
| A non-streaming chat response that ends inside its think block returns that text as `reasoning_content` with an empty `content` (as the streaming API always did) | `--unclosed-think content` |
| A streamed response no longer carries its own stop string; an empty stop string (`stop: ""`) is ignored (before: an empty reply with finish `stop`); a streamed reply that ended in `<` or in the start of `<tool_call` / `<think>` / `</think>` now sends those characters | none: defect fixes |
| After an engine error a stream carries nothing but the finish chunk (a held but unparsed tool-call span is dropped; a complete call is still delivered) | none |
| An option that takes a free-form value and is followed by another `--flag`, or is last on the line, exits 2 naming both flags (before: the next flag was taken as its value, silently); the same for the flags the mode dispatch reads by hand, with a short documented list of flags whose value is optional | write `--option=value` |
| A head and a node of different builds are refused at the hello (protocol 6); a v0.7.4 machine does not pair with a v0.7.3 one | run the same binary on every machine |

### Fixed — speculation could stay off for the rest of a long request

On the EXL3 server a per-request control compares the cost of speculative (MTP) rounds with
plain decoding and turns speculation off for a request when it stops paying. Two things were
wrong with it. The decision used a 2 % margin over a window of about 16 rounds, against a
plain-decoding sample that is refreshed only once every 128 rounds (or 1,024). And it was
final: nothing turned speculation back on for that request. In an engine log shared by a user,
a request that had averaged 18.3 ms per token with speculation was switched off at round 250 on
a short stretch measuring 27.84 ms per token against a plain sample of 27.22; the remaining
1,640 of its 2,592 tokens (63 %) ran at plain speed. A second user's log shows the same after
161 rounds (24.58 against 23.35 ms per token). Only generations of 64 rounds or more could be
affected.

What changed:

- The cost is averaged over about 64 rounds and the margin is 10 %.
- Before switching off, the engine takes a fresh plain-decoding measurement at the current
  context, and switches off only if the margin still holds.
- A request that was switched off keeps its draft head in step, and speculation is tried again
  for a short probe (at most 32 rounds; a probe that is not paying fails after 16) once 256
  plain steps have passed. Each failed probe doubles the wait, up to 4,096 steps; a successful
  probe resets it.
- `--mtp-reprobe <auto|off>` (default `auto`). `off` restores the previous one-way behaviour
  exactly.

Measured (3.05 bpw pack):

- Eight long sampled requests of the kind in the first log (up to 3,200 tokens each,
  temperature 0.7 / top_p 0.8 / top_k 20, one machine; run on the first build of the fix,
  before its review follow-ups, which changed the cost estimator's seeding and the hold-off
  after a re-arm): 8 of 8 served with 0 off events; the six requests served with speculation
  ran 2.57–4.62 tokens per round at 10.3–15.4 ms per token. The temperature-1.3 workload
  below, the A/B and the identity cells were repeated on the final build.
- A workload where speculation does not pay (temperature 1.3, 8 × 3,000 tokens): speculation
  is switched off (for example after 207 rounds, 28.95 against 26.19 ms per token at 1.10
  tokens per round), a probe follows after 256 plain steps and fails, and the waits double:
  512, 1,024, 2,048. The same sequence runs on two machines (TP=2: 7 switch-offs over 8 requests, all served; 12
  re-probes, 11 of them failed and 1 successful, so 18 off events counting the failed probes),
  with the same lines in both ranks' logs.
- Greedy output is unchanged: six recorded identity cells (one machine and TP=2) give their
  recorded hashes while a test drill forces the off / re-arm cycle continuously (the head log
  of one TP=2 run holds 416 off events and 398 re-probes, 233 of them successful, with both
  ranks in step throughout).
- Cost when speculation is healthy: none measurable. Same binary with `auto` and with `off`,
  alternating boots, one request of 256 tokens: code 84.5 / 83.7 against 84.5 / 84.3 tok/s,
  prose 60.0 / 59.2 against 59.9 / 59.9 tok/s (inside the ±1 % run-to-run band).
- Cost of keeping the draft head in step while off: requests that ran almost entirely plain
  measured a median of 26.45 ms per token with it and 26.58 without, a difference below the
  request-to-request spread (about 4 ms per token).
- Upper bound of the probing cost on a request where speculation never pays (arithmetic, not a
  measurement): a probe of 32 rounds per 256 plain steps (12.5 %) in the first period after a
  switch-off, falling to 32 per 4,096 (0.78 %) once the wait has reached its cap.

What does not change: when draft acceptance stays low for a whole request, the probes fail and
the request still runs mostly at plain-decoding speed. The fix is for a temporary trough that
used to cost the rest of the request.

Observability: the per-request `[mtp-stats]` log line now ends with
`off_events=N reprobes=N reprobe_ok=N`; `/metrics` adds `velogb10_spec_mtp_off_events_total`
(events, not requests: one request can go off several times), `velogb10_spec_mtp_reprobes_total`
and `velogb10_spec_mtp_reprobe_ok_total`; the log prints `MTP re-probe (slot N): back ON after
N plain steps`, `MTP re-probe FAILED (...next probe in N plain steps)` and `MTP re-probe OK`
next to the existing `MTP off for this request` line.

### Fixed — a request's greedy text could depend on what else was being served

A decode step chose dense or sparse attention once for all of its rows, from the longest
request in the step. The sparse path starts at 2,052 tokens of context on Flash-Next, so a
request at or below 2,051 tokens that shared a batched step with a longer one was computed by
the sparse kernels: the same attention, through a different rounding path. A second site of
the same rule: a speculative round whose rows straddled the 2,051-token switch.

Measured before the fix:

- Two running requests (short prompts, 1,500 generated tokens) keep their texts when they run
  alone; with a request of about 20,000 tokens joining, both texts changed in 3 of 3 runs. It
  reproduces on the released v0.7.3 binary.
- Two requests with prompts of about 1,500 tokens, each crossing the switch while generating,
  were not stable from run to run even without a third request (two different texts in three
  runs for one of them).
- Size of the effect: after one step the logits differ by one 16-bit-float unit (relative L2
  1.4e-3). Under teacher forcing the difference settles at a KL divergence of about 5e-4 to
  8e-4 nats with no trend over 1,600 positions, and the top-1 token differs at 1 to 1.5 % of
  positions, every one of them a near-tie. It is harmless for quality and real for
  reproducibility: a free-running greedy text departs at its first near-tie.

What changed (host side only; no kernel changed):

- In a mixed step the rows at or below the switch keep the dense kernels they run alone, and
  the scheduler orders such rows first.
- A speculative round stops at the switch instead of straddling it (a few shorter rounds, once
  per request that crosses 2,052 tokens).
- `--qsa-row-regime <on|off>` (default `on`). `0` restores the previous rule.

Measured on the fix (3.05 bpw pack unless stated):

- Bit level (`--probe-exl3-regime`, a new diagnostic): a short row inside a mixed step is
  bit-identical to the same row alone. 0 of 1,600 steps differ (before: 1,600 of 1,600), in
  two-row and in three-row mixed steps.
- Served: with a ~20,000-token request joining, 0 of 5 runs change a running request's text.
  The two crossing requests give the same texts alone, with a joiner, and with speculation off.
- Texts that were stable keep their bytes: the recorded identity cells are unchanged on one
  machine and at TP=2, and the requests' alone texts equal those of v0.7.3.
- Also checked on the release candidate: the 4.05 bpw pack's identity cell; 16 sampled requests
  crossing the switch, alone and next to long requests (16 of 16 served, no error);
  `--lane-order fcfs` (same texts alone and joined); TP=2 with and without speculation (same
  texts alone and joined, both ranks in step).
- Cost: about 0.8 % of aggregate tokens/s with four concurrent requests of mixed length (63.7
  to 63.2 tok/s, two boots per side); nothing at one request (61.8 / 61.5 against 61.9 / 61.7)
  or with four short requests (91.4 / 90.9 against 91.0 / 91.2). The first step of a new
  (rows, short rows) shape costs about 8 ms once per server lifetime (40.8 ms against 33.05 ms
  for a replay).

**Behaviour change.** With the default (`--qsa-row-regime` on), two kinds of greedy request
can produce different bytes than on v0.7.3: a request whose context crosses about 2,051
tokens while it generates under speculation, and a short request that used to share a decode
step with a longer one. The two paths round the same attention differently, so the bytes
change only where a near-tie resolves differently: a given request may or may not differ.
The v0.7.3 bytes of those requests depended on timing and on the other requests in flight;
the v0.7.4 bytes do not. Requests that never come near the switch and never shared a step
with a longer request keep their bytes. `--qsa-row-regime 0` restores the old rule for
comparison.

New gauge: `velogb10_graph_cache_mixed_step_entries` (mixed-step CUDA graphs captured since
boot; they are captured on first use, one per step shape seen, and are also counted in
`velogb10_graph_cache_entries`).

The defect itself was found while testing pull request #23.

### Fixed — a GPU vision-tower error disabled images until restart (issue #20, pull request #16)

Reported by [@liorm0505](https://github.com/liorm0505) (issue #20); fix contributed by
[@seanlinmt](https://github.com/seanlinmt) (pull request #16), ported into this tree.

A cuBLAS error inside the GPU vision tower was a panic. The panic poisoned the tower's lock,
and every later image request failed until the server was restarted. Reproduced before the
port at TP=2 with two and three simultaneous images: 19 of 20 and 30 of 30 requests failed
with HTTP 400.

Now:

- Tower errors (matrix multiply, attention, readback, synchronisation) are returned as errors.
  The request is then served by the CPU tower, and the next request uses the GPU tower again.
- A poisoned lock is recovered, and a slow CPU fallback does not hold the lock: other image
  requests keep using the GPU tower meanwhile.
- The tower's cuBLAS handle is pinned to the tower's own stream and its per-image scratch is
  released (memory flat over 100 images: 69,869 to 69,903 MiB).
- A failed image preparation is an HTTP 400 with its cause, never a 200 without the image.

Measured: 202 image requests on one machine (100 one at a time, 102 three at a time), 0
failures; at TP=2, 100 requests in groups of two and three, 0 failures. With a failure
injected by a test drill, the request is answered by the CPU tower with the same text
(53.8 s on one machine) and the following requests return to the GPU tower (274 ms, then
104 ms). On the release candidate at TP=2: 1,138 image and 1,145 text requests over six
boots, no wrong answer, both ranks in step, six injected failures all answered by the CPU
tower, and 360 other image requests served by the GPU tower while those fallbacks ran.

Limits: a failed kernel launch inside the tower is still a panic (the worker's panic handler and
the lock recovery catch it, loudly; sending it through the CPU fallback is not done). A GPU
fault that the CPU tower cannot cover is reported as HTTP 400 with the error text, not as a 5xx.
The two findings of the TP=2 run, a tower error while the scheduler was capturing CUDA graphs
and the slow CPU answer, are the next two sections.

### Fixed — an image request that coincided with a CUDA graph capture could lose the capture, the image worker, or the server

Found by the TP=2 verification: once in 1,138 image requests, about ten seconds after a boot's
first request, the GPU tower was refused by the CUDA driver while the scheduler was capturing
graphs for newly used step shapes. The mechanism, reproduced without a model: while another
thread records a CUDA graph, any call on the legacy default stream (an allocation, a copy, a
synchronisation, a free; every "sync" helper of the CUDA binding is one) is refused and also
invalidates the capture in progress. The tower made about eighteen such calls per image. On the
served box the unfixed build lost a graph capture and an image worker in one boot and aborted
the whole server process in another: two of five test boots with image traffic, the two that ran with
`--graph-precapture off` (every graph is then captured under traffic); the fixed build ran the same
five boots (three at default flags, two capture-heavy) with zero failures of any kind.

Now the tower, after its construction, uses only its own stream: its own allocation, zero-fill,
copies and drains (same kernels, same order, same arguments; the weights are still uploaded
once at boot). Two matrix-multiply results in the tower's forward pass that were discarded
without being checked are now propagated, so a failure there goes to the CPU fallback instead
of returning embeddings computed from stale scratch.

New counters: `velogb10_graph_capture_invalidated_total` and `velogb10_graph_capture_failed_total`
(a capture that produced no graph, by cause; they also count the capture sites that used to
return silently). Measured (3.05 pack, one machine, `--max-seq-len 32768 --max-batch 4`): the
embeddings of three image sizes are bit-identical on the fixed and the previous build; text
speed is unchanged (medians 70.6 to 72.5 tok/s over four interleaved boots).
Not covered: a capture begun in the driver's `GLOBAL` mode by another thread would still
refuse the tower's own-stream calls (no served image path has one); image traffic on one
machine against graph capture beyond the five boots above, and TP=4.

### Added — `--vision-cpu-threads <N>` (default 1) and a drill flag that arms in any position

The CPU vision tower answers an image request only when the GPU tower failed. It is a
single-threaded scalar reference, so the answer takes minutes, and it grows with the image
(173 s for 448 x 448 at one thread on one machine; the 173 to 218 s seen at TP=2 is the image
size, not the topology). `--vision-cpu-threads N` (1 to 64, default 1: unchanged) splits the
tower's token rows over N threads; the embeddings are bit-identical for every N (tested at 1, 2,
3, 7, 16 and 64 threads; on the real tower the same digest at 1, 4 and 8). 448 x 448 on one
machine: 172.9 s at 1 thread, 44.8 s at 4, 31.0 s at 8. Not measured at the default
`--image-max-edge 1024`: projected from those figures it is many minutes at one thread. The
default stays 1 on purpose: more threads take worker cores for the duration of a rare fallback.

`--vision-fail-gemm` (the test drill that fails the first GPU matrix multiply of the tower) was
armed only when another argument followed it; as the last argument it was silently inert. It
arms in any position now. `--vision-cpu` and `--parity` on the NVFP4 server had the same defect
and are fixed too.

### Fixed — EXL3 served windows above the trained length (pull request #21)

Reported and fixed by [@ccgauvin94](https://github.com/ccgauvin94) (pull request #21), ported
into this tree.

The EXL3 server did not clamp `--max-seq-len`. Above the trained window (262,144 tokens on
Flash-Next) a scratch buffer of the sparse attention kept its trained-window size while the
scorer wrote one block per served position, so above 262,147 a speculative round could write
past it (at `--max-seq-len 524288` every round of five or more rows did, from the first one).
v0.7.3 is affected. The defect is shown by the code path and its index arithmetic: in a 96-token probe
at a 524,288 window the v0.7.3 binary showed no visible symptom, and the later CUDA
illegal-address error described in the pull request was not reproduced here.

Now: the buffer follows the served window; the EXL3 server clamps `--max-seq-len` to the
trained window and prints the limit (`--max-seq-len N exceeds the model's trained window
262144 (max_position_embeddings) - clamping to 262144.`); the bit-packed launch arguments are
checked in release builds; the loader refuses a window they cannot carry.

Measured: a boot with `--max-seq-len 524288` clamps and serves; a request of 255,656 prompt
tokens plus 2,048 generated ends at position 257,704 of 262,144 with a correct answer (234 s);
the recorded identity cells are unchanged; at TP=2 both ranks clamp to the same window.

Behaviour changes: (1) an EXL3 `--max-seq-len` above the trained window is now clamped where
it used to boot at the requested size. (2) The same checks are active on the NVFP4 path, where
`--rope-yarn-factor 2 --max-seq-len 524288` used to overflow a packed field silently and now
stops at the first attention launch with `bs_packed field overflow … lower --max-seq-len`.
This second case was not run on a GPU.

### Fixed — TP network bring-up (pull request #15) and TP image payload order (pull request #17)

Contributed by [@seanlinmt](https://github.com/seanlinmt) (pull requests #15 and #17), ported
into this tree.

- **RoCE GID index.** The TP data plane used a fixed GID index (3). It is now detected: the
  first IPv4-mapped RoCE v2 GID in sysfs, printed at bring-up with the sysfs evidence
  (`[tp] gid_idx=3 auto-detected on <device> (sysfs type 'RoCE v2', gid ...)`); if none is
  found the old fixed value is used. `--gid-idx <N>` sets it explicitly for every rail; a
  wrong N fails at bind and is never re-scanned.
- **Bring-up.** The CUDA device is selected explicitly in both bring-up paths, and a failed
  pinned-memory allocation prints the CUDA error text and code.
- **Image payloads under TP.** All image payloads of a scheduler step are read right after the
  step frame, in admission order, before any prefill runs (a prefill uses the same control
  stream).
- **`--ple-headroom-per-lane-gb <F>`** (opt-in, default 0): extra headroom for the
  `--ple-ram auto` decision, per additional lane on top of the fixed 16 GiB (for example 2
  with `--max-batch 10` gives 16 + 2 × 9 = 34 GiB). The default reproduces the previous
  decision exactly.

Measured at TP=2: auto-detection picks index 3 on both ranks of the test pair (the value the
fixed setting used), the recorded identity cells are unchanged, a boot with `--gid-idx 3`
behaves the same, and after the head process is killed the node re-arms by itself (4 s) and
serves the next head. On the release candidate the node re-armed across six consecutive head
sessions, and 1,117 steps shipped image payloads (21 of them with two or more) with every
answer naming its own image.

Not ported: the environment-variable form of the GID option in #15 (the engine takes flags
only); `--served-model-name` from #15 and the telemetry wiring from #17 (both already in
v0.7.3); the 2 GiB-per-lane default of #17 (the option is there, the default stays 0); and a
transport-side fallback scan for the GID, which was removed in review so that the transport
binds exactly the index the engine resolved.

### Fixed — tool calls that quote tool calls, and five more parser defects

When a model writes a document ABOUT tool calls (a file, a test fixture, a prompt), the quoted
calls contain the same tags as a real call, and the parser has to decide which tags are
structure and which are text. The parser was reviewed by an outside expert and changed. What
v0.7.4 does at default flags (each change only alters responses that were wrong before; the
restore flags are in the behaviour table above):

- A `</tool_call>` inside an argument value is text, and so is a quoted complete block inside a
  value (`--tool-block-end after-function`; `first` restores v0.7.3). This follows the popular
  engines: in vLLM's Qwen3 parser only `</function>` ends a call, and `</tool_call>` is looked
  for after it.
- A string argument loses exactly one wrapping newline, the one the chat template writes
  before and after the value, instead of all of them, as vLLM does (`--tool-value-trim one`;
  `all` restores v0.7.3).
- A streamed call whose argument holds `<think>` no longer leaks raw call markup as content or
  splits into reasoning; with thinking off, an argument holding `</think>` no longer loses the
  call. A Qwen call whose value mentions the Hy3 call tag stays a Qwen call (before: no call and
  the raw markup). A JSON-bodied block whose string quotes `</tool_call>` is read whole.
- A hostile block (thousands of parameters, or junk after every close tag) took seconds to
  parse; the parse is linear in the block for those shapes. One third shape remains quadratic:
  a block made of a very large number of bare `function=` lines with no `<function=` tag (about
  6 s for 500 KB, the same as v0.7.3).
- **A block the model never closed** (no `</tool_call>`) is dropped, as in v0.7.3
  (`--tool-unclosed drop`, the default). `--tool-unclosed complete` is available: when the model
  ended by itself (finish `stop`) and a `<function=...>` element inside the block is closed with
  `</function>`, that call is returned, as vLLM does; a call cut inside an argument, or any
  response cut by the token limit, cancelled by the client or ended by an engine error, is
  dropped whole under both settings, because it would run with missing arguments. It is not the
  default because the evidence does not support it: in the 36 recorded outputs with an unclosed
  wrapper that were replayed (one machine, 3.05 and 4.05 packs), no response had a closed
  function element without its wrapper close (every real one was cut inside its element), and
  the one output `complete` changed was a piece of prose about the call format whose stray tags
  parse as a junk call. `velogb10_toolcall_unclosed_total{outcome="returned|dropped_cut|dropped_length"}`
  counts what happens either way.

**Not fixed by default: a value that quotes a call with two or more parameters.** The quoted
`</parameter>` followed by a `<parameter=` looks exactly like a real parameter boundary, so at
default flags such a document is read as a cut value plus extra calls, and a value that quotes
two calls comes back as two calls (the second made of the quoted text, which an agent would
execute). Measured with the v0.7.4 binary (one machine, 3.05 pack, streamed, default flags, 30
offered tools): a response that quoted 40 two-parameter calls came back as 40 calls; the right
answer was one `write_file`. On recorded outputs that copy such documents (non-streamed, mostly
thinking off, client-set temperatures; 7 documents x 3 phrasings, resampled) the v0.7.3 reading
returned a phantom call in 348 of 1,000 and a cut file in 506 of 1,000.

Two opt-in readings handle it: `--tool-quote-pairing on` pairs each quoted open tag with its
close and reads the document whole; `--tool-quote-pairing cost` picks the cheapest consistent
reading and reads more. Neither is the default, because each has a class it reads worse than
the default: `on` mishandles a value that mentions an open tag once, unclosed, followed by a
real close; `cost` mishandles a real second call that follows a first argument holding a lone
open tag. We do not yet know how often either class happens on real traffic. **To help us
measure it:** start the server with `--tool-shadow-cost on` (diagnostic only: it also reads every
tool-call response under the `cost` rule and counts how often the result would differ; it
changes nothing that is returned and costs about 0.05 ms per response, 6 ms in the worst shape
measured: 64 blocks of 8 KB with 100 tools offered). The counters are on `/metrics` as
`velogb10_toolcall_*`. The counts, or a sample of responses where the readings differed with anything private removed, are what is
needed to choose the default for the next release. The Flash-Next guide, section 8, has the
advice for agent authors (escape or split quoted tags, or write the file through a path that
does not go through a tool-call value).

Where this was and was not run: one machine, EXL3 3.05 pack, streamed, thinking on, 30 offered
tools, a handful of requests per case (not a frequency). Never on two or four machines and not
on the NVFP4 server for these changes (the parser changes are head-only and independent of the
topology; that is an argument, not a run). The Hy3 and DeepSeek V4 tool-call formats keep the
old value and block rules.

### Fixed — the first streamed chunk was not a first response (forum thread, post #46)

The chat stream sent a role-only `choices` chunk about 5 ms after the request arrived and the
first content or reasoning delta 1.6 to 1.7 s later (a 2,111-token prompt). A client that takes
the first chunk with a `choices` entry as the first response, as the llama-benchy tool does,
measured prompt processing at 431,582 to 514,059 tok/s on one machine of ours; the figure is
1,292 to 1,362 tok/s after the change (c=1, 4.05 pack, one machine, the same tool).

Now no chunk with a `choices` entry leaves the EXL3 server before the first generated token, and
the role chunk leaves with that token (also when the reply is only a buffered tool call). Data
events, bytes and order are the same as before on ten request shapes; only the role chunk's time
moved. While a request waits in the queue or a long prefill runs, a streamed response now carries
an SSE comment line (`: keep-alive`) every 10 s (`--sse-keepalive N`; `0` = off). The lines stop
when the whole server has made no progress for 60 s (no decode round and no prefill chunk of any
request), so a hung engine cannot hold a client open; a request that gets no event while other
requests progress keeps receiving them. `/health` and the DEAD state remain the liveness signals.
The same rule serves `/v1/completions` streams. Errors are unchanged on purpose: the status
and headers still leave at once, and a failure before the first token is an SSE `error` event.

Checked: one machine (EXL3, 4.05 pack, TP=1) and, in the integration gates, TP=2 (3.05 pack:
role chunk with the first token, `prompt_ms` 117.9 against the role chunk at 119.6 ms; a waiting
request received frames every 10 s through 90 s). The NVFP4 server shares the handler and was
checked on one machine (role chunk with the first token; frames at `--sse-keepalive 2`, none at
0). The DeepSeek V4 server's streamer was moved into the library with its tests and its
`timings` block for greedy requests (which reported a fictional `prompt_per_second`) was
corrected; **neither was run on a GPU**, and that server sends no keep-alive frames.

Also in this group (all found by reviews of the streaming path, all without a flag):
`stop: ""` and empty entries of a stop list are ignored; a streamed reply no longer contains its
stop string (the non-streaming response never did); a streamed reply that ended in `<` or in
the first characters of `<tool_call`, `<think>` or `</think>` lost those characters, and text
after a `<tool_call` that was not a call tag was lost too; a stop string costs time linear in
its length per token (it was quadratic: 308 ms per call at 200 KB); after an in-band engine
error a stream carries only the finish chunk.
`--unclosed-think reasoning|content` (default `reasoning`) is described in the behaviour table.

### Fixed — the TP lockstep word could not tell a 15-row verify from a 16-row one; head and node must be the same build

The per-step agreement word that the TP ranks compare carried a step's verify width in 4 bits,
so a 15-row and a 16-row step were not distinguished by it. It carries 5 bits now (0 to 31).
Checked on two machines with a drill that makes one rank report a different width: the fix
aborts on it at the step, and the old word let it through in silence. No failure had been
attributed to this before; a 16-row step needs `--max-batch` of 16 or more.

The same work found that the stamp that makes a head refuse a node of a different build did
not cover 20 engine sources (`src/exl3_forward/*`, `src/dflash2/*`, `src/dspark/*`), the data
compiled in, `Cargo.toml`, `Cargo.lock` or `build.rs`: a head and a node that differed only in
the TP sharder were not refused. It covers them now (FNV-1a over every file's path, length and
bytes, so two machines that build one commit with different toolchains get the same stamp), and
the protocol version is 6. A head and a node of different builds are refused at the hello with
both stamps named, before any model is shipped; checked in both directions on two machines.
**A v0.7.4 machine therefore does not pair with a v0.7.3 one: run the same binary on every
machine of a TP group.**

### Changed — `--lane-pack <on|off>` is ON by default (EXL3; `--lane-pack off` restores the previous behaviour)

Two requests that run at the same time used to take turns, or to share one plain step with
speculation off. With lane packing, two of them share ONE speculative verification (a
packed round): the engine drafts for both, verifies both in one pass, and each request keeps
its own tokens. The flag was opt-in in the release candidates; the maintainer decided on
2026-10-12 to turn it on by default in this release.

- Greedy requests are byte-identical packed or alone (the recorded identity cells on one machine
  and at TP=2, with the packed-round counters equal on both ranks). Sampled requests join the same
  rounds and are verified by the real-q ratio rule that a lone sampled round uses;
  `--lane-pack-sampled-tp <on|off>` (default `on`) chooses whether they join under TP (`off`:
  only greedy requests pack under TP).
- A request that carries a repetition, presence or frequency penalty, or an image, never joins
  a packed round; neither does any request on a server booted with a non-zero
  `--default-*-penalty` (the boot prints a warning for that case).
- A posture the packed arm was never gated under boots with packing INACTIVE and one line
  saying what was asked, what the boot does instead and why, instead of refusing
  (`--lane-order fcfs`, `--kv-cache` other than q8, `--exl3-mtp-k` other than its default,
  `--qsa-row-regime off`, `--a-ctrl old`, `--max-batch 1`, `--graph-precapture off`, no live
  draft head (`--mtp off`), and the model and kernel postures the packed verify cannot serve).
  `velogb10_lane_pack_active{reason}` says which, and `--lane-pack off` silences the line. A
  packed scratch or graph that cannot be built at boot (device memory) is the same fallback: the
  boot continues without packing and says why; it never aborts. (Under TP a failure on one rank
  only is caught by the boot agreement, loudly, rather than served.)
- With the default on, the choice is made by every machine of a TP group from the same
  resolved value (the head's value ships to the nodes), with no flag typed; the boot agreement
  hashes the decision and its reason, so a head and a node that disagree refuse the boot.
- At four busy requests with lane packing active, one shared plain step is tried every 16 rounds
  while the engine has no measurement of it (then every 128), so the load-adaptive policy can
  pick the cheaper arm from a measurement; before this, a packing boot rode packed
  pairs for a whole request in 19 of 20 test boots although the shared step measured faster in
  3 of the 4 standard cells.
- Cost: about 1.7 to 1.8 GiB of device memory per machine (a second 16-row scratch and its
  graphs, captured at boot), also for a server that never has two requests at once. The boot
  line prints the free memory after the boot. Paid by default now; `--lane-pack off` gives it
  back. Not measured for this release with the default on: the free host memory before and
  after on the 4.05 pack at `--max-seq-len 262144 --max-batch 4` (the tight cell).

Measured (request class stated on each line; all unpenalized, no sampler fields; every number
below was taken with the flag typed `--lane-pack on`, before it became the default):

- One machine (TP=1, 3.05 pack, `--max-seq-len 32768 --max-batch 4`, 512-token requests with
  short prompts, thinking on and off, code and prose, three adjacent off/on pairs per cell):
  two requests together ran 1.15 to 1.34 times as fast (mean of three pairs; the weakest cell,
  code with thinking off, has the noisiest off arm); a single request is unchanged (0.97 to
  1.05, the noise floor); packed rounds 98 to 217 per cell.
- Two machines (TP=2, 3.05 pack, streamed, default sampling, window 262,144, five adjacent
  off/on pairs per cell, recomputed by an independent review): two requests together ran 1.17
  to 1.32 times as fast on short prompts (about 75 tokens, 512 forced tokens). On agent-style
  prompts (about 5,500 tokens, a 14-tool catalogue, first turn, nothing cached) the gain was
  about 12 % at best with thinking on (marginal) and within noise with thinking off, because
  the prompt dominates the wall clock; three requests: no gain; four requests: between 8 %
  slower (prose, thinking off) and about 14 % faster (code, thinking off).
- Not measured: four machines, a conversation whose prompt is already cached (warm multi-turn),
  a request with a penalty. The distribution check of the sampled packed round passes its
  stated rule (alpha 0.01) at p = 0.035, is a single experiment, and would not pass at 0.05; the
  48 byte-identity cells ran at a 32,768 window.
- Robustness (TP=2, 3.05 pack, `--lane-pack on`): a 30-minute stress of 782 streamed requests
  (default sampling plus a greedy share, one to three at once, 10 % cancelled): 0 errors, the
  greedy ones identical to alone, the round counters equal on both ranks; a deliberately
  injected width mismatch is caught on both ranks.
- Counters: `velogb10_lane_pack_rounds_total{arm="packed|serial|shared_plain"}`,
  `velogb10_lane_pack_lane_rounds_total{kind}`, `velogb10_lane_pack_skip_total{reason}` (why a
  request was kept out of a packed round), `velogb10_lane_pack_disabled`,
  `velogb10_lane_pack_active{reason}`.
- **What was exercised on the release binary.** The maintainer ran the previous build by hand
  on one machine, at `--max-batch 1`, at TP=2 and at TP=4, with packed rounds (the flag typed
  `--lane-pack on`), for daily use, and noted no crash or error; the speed benefit is clearest
  on one and two machines and smaller at TP=4 (no numbers are published). No automated gate
  chain was run on this exact binary. The gate results quoted in this section are those of the
  release candidate `7a4117643a52` and of the gate sessions named, carried over by argument: the
  packed arm is the same code the gates ran with the flag typed; what the cut changes is that it
  is on by default and how that is resolved and agreed between machines (read, not booted).

### Added — KV occupancy gauges (issue #24)

Requested by [@liorm0505](https://github.com/liorm0505) (issue #24). On the EXL3 server's
`/metrics`: `velogb10_kv_positions_{capacity,used,peak,cached}`, per lane
`velogb10_kv_lane_positions_{capacity,used,peak,cached}{lane}`, and
`velogb10_kv_allocated_bytes{kind}` (`trunk_kv`, `indexer_keys`, `draft_head`). A position is
one token of one lane's context, not a byte; `used` counts running requests, `cached` counts
the prompt positions that free lanes keep for the prefix cache. Reading them:
`max(velogb10_kv_lane_positions_peak) / velogb10_kv_lane_positions_capacity` is how much of the
window the longest context used, and `sum(velogb10_kv_allocated_bytes) /
velogb10_kv_positions_capacity` is the bytes one position costs. Under `--tp` the byte series are
the head's share. The NVFP4 server does not report them. Checked on one machine and at TP=2
(two packed lanes: every lane's peak equals its request's final position; 0 used when idle).

### Added — `--perplexity` on EXL3 packs: a measuring tool with checks that can fail

`--perplexity --model-dir <pack> --text <file>` (or `--ppl-ids <file>`) scores held-out text on
the SERVED chunked prefill, with every row of every window passed through the language-model
head, and prints one line per window and a final `RESULT:` line. It is a measuring tool: it
serves nothing, and it runs on one machine (TP=1). Options: `--window N`
(default 1024), `--ppl-csv <file>` (per-token NLL), `--ppl-dump-ids <file>` (the scored ids, to
feed another runtime the same tokens), `--ppl-label`, `--ppl-ref-windows N` (also run the first N
windows through the single-token decode path, a different kernel set, and report the
difference). The run exits 1 and prints no `RESULT:` line when a check fails: a language-model
head that wrote nothing (the buffers are poisoned before every launch and scanned afterwards),
a top-1 accuracy below `--ppl-min-top1` (0.02), or a perplexity outside `--ppl-min-ppl` (1.05)
and `--ppl-max-ppl` (1000). **A printed `RESULT:` means "not catastrophically broken" and
nothing more**: the bounds are wide, and correctness needs a reference of a different class.
The tool refuses what would make a measurement ambiguous: an option repeated with different
values (the registry options are last-wins while `--model-dir`, `--kv-cache`, `--prefill-chunk`
and `--max-seq-len` are first-wins elsewhere), and a value-less `--text`, `--ppl-ids`,
`--ppl-csv`, `--ppl-dump-ids`, `--ppl-label` or `--model-dir` followed by another flag
(it would swallow that flag). `--perplexity off` does not run the mode.

Measured (3.05 and 4.05 packs, three texts, 25 windows of 2,048 tokens = 51,175 scored tokens
per cell, KV q8; perplexity):

| text | 3.05 | 4.05 | ExLlamaV3 3.05 | ExLlamaV3 4.05 |
|---|---|---|---|---|
| code | 3.5264 | 3.5409 | 3.5483 | 3.5571 |
| prose | 6.3994 | 6.4184 | 6.4221 | 6.4839 |
| json | 2.3135 | 2.3179 | 2.3096 | 2.3162 |

The 4.05 pack does not score better than the 3.05 pack on these texts, in either runtime (+0.13
to +0.56 % in the first matrix; very slightly worse in 9 of 9 engine cells in the repeated
measurement, about 0.1 to 0.2 % in perplexity, t = 0.5 to 2.1 per cell, sign test p about 0.002).
Single-pass reference differences below about 0.5 % are noise: the ExLlamaV3 reference is not
deterministic from call to call, so any comparison uses `--repeats 3` or more with its standard
deviation (the reference script of this work does). KV precision (q8, fp8, f16) does not
separate the cells beyond 0.22 %. In the first window the engine's loss is below the reference's
by 0.05 to 0.44 nats per token from position 256 on, in every KV format, and on the first row of
windows 2 to 25 it is lower in 18 of 18 cells; the candidate cause is the start token (the
pack's `eos_token_id` 248044 against the engine's 248046), which is open. The probe that
compares the served chunked prefill with token-by-token decode reports a recurrent-state
difference (largest absolute value 0.73) that is byte-identical to the one recorded two weeks
earlier; it has not been investigated as a defect, and nothing here claims it harmless.

Found on the way: the generic (non-EXL3) loader read tensor bytes with a cast that requires
the data's file offset to be aligned for its dtype, and a safetensors header length need not be a
multiple of 8. On such a file v0.7.3 panicked at the first tensor it cast, and the panic left
the loader's worker threads waiting for a flag nobody would set: the process then hung. Fixed:
an unaligned read is copied, an aligned one is borrowed as before (the same perplexity to every
printed digit on an aligned model; the misaligned copy of the same model now loads), and the
panic no longer hangs the process. An EXL3 pack handed to a loader that cannot read it is now
refused at once with a message naming the entry points that can.

### Added — `--prefill-interleave <N>` (EXL3, one machine, default 0; pull request #23)

Contributed by [@adriangrassi](https://github.com/adriangrassi) (pull request #23), ported
into this tree.

A long prompt that joins while other requests are generating used to freeze those streams for
its whole prefill. With `--prefill-interleave N` the server runs up to N decode steps for the
running requests after each prefill chunk of the joining prompt except the last.

Measured (two running streams, a joining prompt of about 20,000 tokens, one machine):

| `--prefill-chunk` | interleave | longest stall of the running streams | joining request's time to first token |
|---|---|---|---|
| 4095 (default) | 0 | 14.1 s | 14.1 s |
| 4095 (default) | 1 | 5.4 s | 14.5 s |
| 2048 | 0 | 15.1 s | 15.1 s |
| 2048 | 1 | 2.7 s | 15.6 s |
| 1024 | 0 | 16.6–17.0 s | 16.75 s (mean of 3) |
| 1024 | 1 | 1.35–1.47 s | 18.21 s (mean of 3, +8.7 %) |
| 1024 | 2 | 1.35–1.55 s | 19.57 s (mean of 3, +17 %) |

The last chunk is never interleaved and the tail of a prompt is merged into it, so the longest
stall is the largest chunk of the grid: at the default chunk width the flag alone only brings
the stall to 5.4 s. The useful settings combine it with a smaller `--prefill-chunk`, which
prefills more slowly and changes the output of long prompts (as that flag is documented to
do). The prefill rate itself is not slowed (1,210 against 1,203 tok/s in the server's prefill
line); the added time to first token is the decode steps run for the other requests.

Identity: with the per-row attention regime of this release, the running requests' greedy
texts are the same in all nine joined runs and equal to their texts alone, and each joining
request's text is the same at interleave 0, 1 and 2 (three joining documents, chunk 1024, one
machine). Stress: 60 minutes at interleave 1 with long prompts joining, being cancelled during
prefill and during generation, and re-sent (334 joining requests): 0 server errors, 0 panics.

The option is ignored under TP, with a line on every rank
(`--prefill-interleave N IGNORED under TP (single-box only)`); with the flag set, TP=2 output
equals the recorded hashes. The server's prefill line excludes the interleaved decode time and
a new line reports it (`prefill-interleave slot=N: ... scheduler step(s) ... ms of decode`).
Not tested with the flag on: an image request as the joining request, and `--lane-order fcfs`.
The NVFP4 server does not read this flag (it has its own `--prefill-sched`).

### Changed

- **`--mtp off` is honoured on the EXL3 server.** It used to be ignored there (the boot banner
  kept saying `MTP chain-verify ON`). With `--mtp off` the banner reads `MTP chain-verify OFF:
  MTP OFF (--mtp off)`, no draft head runs, and the `[mtp-stats]` lines show `rounds=0
  drafted=0`. `--exl3-no-mtp` still works. On the EXL3 server `on` and `auto` mean the same
  (speculation on, with the per-request control); an invalid value is refused at startup.
- **`--help` says what the servers do.** `--max-tokens` is the default for a chat request that
  omits `max_tokens`, not a cap: unset, the EXL3 server has no limit and the NVFP4 server uses
  8192. `--default-presence-penalty` is 0.0 on the EXL3 server (1.5, or 2.0 on 2B, on NVFP4).
  The sampler defaults are stated per endpoint (see "Known limits"). `--lane-order` carries the
  measured numbers. Thirteen lines of `--help` that printed mis-encoded characters are fixed.
- **`--mtp-lanes` and `--prefill-sched` are listed in `--help`.** Both are NVFP4-server flags
  that were parsed but listed nowhere; they now refuse an unknown value at startup (before, an
  unknown value silently meant the default). `--prefill-sched` defaults to `cursor`.
- **TP launcher (`run_tp_server.sh`).** `--tp` now defaults to 1 + the number of `--node`
  flags, and an explicit `--tp` that disagrees with that count is an error (exit 2). Two
  `--node` flags are refused, because the engine serves a power-of-two world only (TP=2 or
  TP=4 in practice).
- **`[gen]` log line.** Three fields are appended: `stop_kind` (`eos`, `stop_string`, `length`,
  `loop`, `cancel`, `error`, `other`), and for a request that ends on a stop token `stop_p` and
  `stop_rank`: the probability the sampler gave that token and its rank among the logits (for a
  greedy request, the softmax probability at temperature 1). The cost is one read of a logits
  row, once per request, on its last step, and only for a stop-token finish. The two fields
  read `na` under TP, on the NVFP4 server, when the first generated token is the stop token,
  and for every other kind of finish. A request that runs out of tokens inside its thinking
  block is now counted as reasoning tokens in this line (it was counted as content).
- **Streaming `/v1/completions` sends a usage chunk** when the request carries
  `stream_options: {"include_usage": true}`: empty `choices`, top-level `usage` with
  `prompt_tokens_details.cached_tokens`, after the finish chunk and before `[DONE]`. Without
  the option the stream is byte-identical to before. (The chat endpoint sends its usage chunk
  unless told not to; this endpoint sends it only when asked.)
- **Benchmark and probe rounds** (`--bench-mtp` and two diagnostics) stop at the 2,051-token
  switch like the server, so a losslessness run across the switch measures what is served.
- **Flash-Next guide, section 8: tool-call values that quote tool-call syntax** now
  describes the parser as it is (the block rule, the newline rule, the three inherent ambiguities
  and the opt-in readings), with the advice for agent authors.
- **`--max-batch` above 16 on the EXL3 server** boots with 16 lanes and prints one line saying
  so (`--max-batch 32: this server serves at most 16 requests at the same time; using 16;
  further requests wait in the queue`); `velogb10_max_batch` reports 16 and the new
  `velogb10_max_batch_requested` what the command line asked. Nothing is rejected: the 17th
  request waits. At TP=2 the same value used to be refused before the network step; one
  resolver now serves one machine, two and four, and both ranks resolve the same width.
- **`--exl3-recon-min 0` (or above 32) on the 4.05 pack** boots and serves. The pack has 6-bit
  modules that the wide trellis kernel cannot decode (3-, 4- and 5-bit only); those modules now
  go through the reconstruct + cuBLASLt arm whatever the option says, with one line at the
  first use and `velogb10_exl3_recon_forced_total{bits}`. Before, the boot's warm-up reached the
  wide arm and died. The greedy text equals the default boot's (readiness hash
  `6809e8685d17412e`). A pipelined prefill MoE kernel on a pack with 5-bit routed experts, and the
  host-routed MoE arm on an expert width it has no path for, are refused with a message that
  names the option (`--pf-moe-kernel off`) instead of computing garbage; no shipped pack is
  affected. A plain step of 17 or more rows on a pack with 6-bit modules (`--max-batch` above 16
  on such a pack) now fails loudly where it used to answer with garbage.
- **Options that swallowed the next flag.** An option that takes a free-form value (text or
  path) and is followed by another `--flag`, or is last on the line, exits 2 before any model
  load, naming both flags and the escape: `--rdma-dev needs a value, but the next argument is
  --tp; to give a value that really starts with "--" write --rdma-dev=--tp`. The `--option=value`
  form is accepted everywhere. The flags that the mode dispatch reads by hand (among them
  `--model-dir`, `--port`, `--nodes`, `--tp`, `--max-batch`, `--max-tokens`, `--prefix-cache`) are
  covered by the same rule. Flags whose value is optional never refuse: `--tp` (bare `--tp` is
  two machines and prints one info line on stderr; write `--tp 4` for four), `--head`,
  `--probe-tq`, `--output-prompts`, `--probe-bandwidth-sustained`, `--splitk-gemm`.
  Checked against 1,685 launch lines of the repository's scripts, docs and model cards: no real
  launch line is refused. Integer, float, choice and boolean options already failed loudly on a
  swallowed flag and are unchanged.
- **Unknown flags warn.** A `--token` that no option or mode reads prints one startup line on
  the head (`[opts] unknown option --max-bach: ignored; did you mean --max-batch?`, with up to
  three nearest names) and the boot continues exactly as before; a misspelt flag used to
  disappear silently. Nodes do not warn.
- **`--help` and `--help-diag`** gained the options of this release, the TP drill paragraph and
  the corrected `--lane-pack`, `--unclosed-think` and `--vision-cpu` texts. The release binary
  carries no build-host directory in its panic locations or other strings.
- **The server prints what it does when a limit or an unsupported combination is applied**
  rather than refusing: lane packing with an ungated option, a `--max-batch` above 16, an
  `--exl3-recon-min` the pack cannot serve. The refusals that remain are the ones where
  continuing would serve wrong output or misstate what is served: a head and a node of
  different builds, a TP world above 4, a pack the engine cannot read, a deliberately broken
  KV control, and a wrong value of an option (the engine cannot know what was meant).

### Investigated — issue #18: no engine defect found

Reported by [@hatemismail](https://github.com/hatemismail) (issue #18), with a complete
reproducer: on one 33K-token request with three tool definitions, the model's reasoning ran to
the 8,192-token budget on veloGB10 while the reporter's reference stack finished in
564–1,147 output tokens (the reporter's figures).

Result, on the reporter's request, the 4.05 bpw pack, temperature 1.0 / top_p 0.95 / top_k 20:

| prompt (identical token ids on both runtimes) | veloGB10 v0.7.3 | ExLlamaV3 1.4.7, same pack |
|---|---|---|
| the request as veloGB10 and Hugging Face transformers render it (33,415 tokens) | 25 of 35 requests reach the budget | 7 of 10 |
| the same request with the tool definitions serialised as the reference stack does (33,445 tokens, reconstructed from the reporter's description) | 0 of 15 | 0 of 10 |
| the same three tool definitions in reverse order (33,415 tokens) | 0 of 10 | 0 of 10 |

On identical token ids a second runtime on the same weights behaves as veloGB10 does, on each
rendering. No engine setting moved the rate on the original rendering (KV cache precision,
speculation on and off, the sampling rule, sparse against dense attention). The difference the
reporter saw is reproduced by the difference in how the two stacks serialise the tool
definitions; veloGB10 passes the client's tool JSON through the model's chat template
unchanged. Of ten renderings of the tools block tried (the original included), six reach the
budget often (4 to 8 of 10 requests), three almost never do (0 of 15, 0 of 10, 1 of 10) and
one is in between (3 of 10). Why these few tokens matter is not known, and no mechanism is
claimed. Not measured: the reporter's
reference weights (NVFP4) fed the original rendering. The issue stays open for that
measurement. The Flash-Next guide has a short note on this behaviour.

### Verification on two machines (TP=2)

- **The soak owed from v0.7.3.** The published v0.7.3 binary, in the configuration of the
  issue #10 report (one request at a time, ~16K-token sampled prompts, a client cancel every
  eighth request): 203,129 pre-verify rounds in 159.8 minutes, 0 aborts, 0 overwritten slots,
  the round counters equal on both ranks, 368 completed requests and 52 client cancels with 0
  request errors, memory flat.
- **A v0.7.4 development build, before the later changes** (the release binary is built from
  the public tree and gated again): the six recorded
  TP=2 identity cells give their recorded hashes; the same configuration as the soak above for
  45.5 minutes: 58,057 pre-verify rounds, 0 aborts, 0 request errors; the speculation re-probe,
  the image robustness run, the prefill-interleave identity and the attention-regime checks
  are reported in their sections above. That build did not contain the lane packing, the
  parser changes, the streaming changes, the lockstep-word change or the option-parsing
  guards; those were gated separately on later builds, on the request classes named in their
  sections.
- **The integration build** (one TP=2 pair, 3.05 pack): a 20-minute run at `--max-seq-len 65536
  --max-batch 16` with 10 to 16 greedy requests arriving and leaving (1,547 requests, 0 failed,
  119 waves); a 10-minute image run (`--vision-fail-gemm`): 203 text and 201 image requests,
  0 wrong, one injected failure served by the CPU tower in 36.3 s while eight GPU images were
  answered; the banked identity cells again; the head-and-node mismatch refused in both
  directions; the width drill caught.
- **No drift across releases.** The published v0.7.2 and v0.7.3 binaries and the candidate
  produce the same per-request texts in the 1/2/3/5/6/8-concurrent identity cell (`IDENT_ALL
  44fe40977a57e9d9`), at every concurrency level of that cell.
- **What the TP=2 checks do not show.** At default flags with three busy requests the TP=2
  scheduler kept serial speculation, so those runs exercise the round stop at the switch and
  not the mixed step; the mixed step at TP=2 was exercised with speculation off. More busy
  requests at `--max-batch 16` were not tried.

### Known limits

- **Image traffic that was not covered by the verification:** one machine (TP=1) against
  graph capture beyond the five boots of the vision section, a continuous image hour
  without restarts, and TP=4.
- **The CPU-tower answer is slow** (minutes for a 448 x 448 image at one thread, longer for a
  larger image; `--vision-cpu-threads` shortens it). It runs only when the GPU tower failed. A
  client with a 60 to 120 s timeout gives up before the answer arrives.
- **Sampler defaults differ between the two endpoints.** When a request omits the fields:
  `/v1/chat/completions` on the EXL3 server uses temperature 1.0 / top_p 0.95 / top_k 20 when
  the prompt opens a thinking block and 0.7 / 0.8 / 20 otherwise; on the NVFP4 server it uses
  0.7 / 0.8 / 20 in both cases. `/v1/completions` uses 0.7 / 0.8 / 20 on both servers. The
  penalties follow the `--default-*-penalty` flags on the chat endpoint (EXL3: 1.0 / 0.0 /
  0.0; NVFP4: 1.0 / 1.5, or 2.0 on 2B / 0.0) and are fixed at 1.0 / 0.0 / 0.0 on
  `/v1/completions`, which ignores those flags. `max_tokens`: the chat endpoint follows
  `--max-tokens` (EXL3 unlimited when unset, NVFP4 8192); `/v1/completions` uses 16 and ignores
  the flag. Greedy decoding needs an explicit `temperature: 0` on either endpoint. The chat
  values are constants in the engine; they are not read from the pack's
  `generation_config.json`.
- **The tuner's speculative rounds are not stopped at the 2,051-token switch.** The server and
  the benchmark rounds are; the four rounds inside the autotune harness are not.
- **`--prefill-interleave` at the default chunk width** leaves a stall of about 5.4 s for a
  20K-token joining prompt (see its section); it is ignored under TP.
- **Tool calls:** a value that quotes a call with two or more parameters is still read wrongly at
  default flags (see "Fixed — tool calls that quote tool calls"). A quoted call whose last argument is
  blank lets the quoted call after it through as a call, under every setting. The Hy3 and DeepSeek
  V4 tool-call formats keep the old value and block rules. A tool call written inside an
  unclosed think block is returned as a call (the parse of an unclosed block is deliberate: the
  model sometimes jumps straight to a call).
- **A non-streaming chat response that ends inside its thinking block** now returns its text in
  `reasoning_content` (see the behaviour table); a client that reads `message.content` of a call
  with a small `max_tokens` on a thinking prompt gets an empty string.
- **Options that are still read by hand outside the option registry** (173 value-taking flags
  of the mode dispatch) keep their own value readers, now guarded by the one swallow check; moving them into
  the registry is planned.
- **A 6-bit pack (the 4.05 pack) with `--max-batch` above 16**, and the host-routed MoE arm
  (`--exl3-moe-devroute 0`, a diagnostic), have no boot-time refusal; the first fails loudly at the
  first 17-row step, the second has no bit-width guard (no shipped pack has 6-bit routed experts).
- **The count of ignored unknown options is not exported by v0.7.4.** The engine keeps the
  count but does not render it on `/metrics` (a known gap, fixed in the next release). The
  `[opts] unknown option ...` warning line printed at startup is the signal.
  `scripts/owner_check.py` (shipped in the repository from this release) says "NOT EXPORTED by
  this build" for a series that is absent and never reads an absent series as "all clear".
- **Not part of this release:** the other experiments of the lane-packing work (for example
  packing aware of the speculation arm), and the typing and ranging of the 173 hand-read flags
  in the option registry.
- **Still open from earlier releases:** the legacy minimal DSV4 server does not report the
  newer usage fields; a JSON schema sent in `response_format` is accepted but not enforced (the
  reply header `x-json-schema-enforced: none` says so); on several machines an error in a step stops
  the whole server, which then restarts.

### What is not included in this build

v0.7.4 is a fix release. It closes reported defects and it does **not** contain the performance
work for the kind of request most people actually send. We would rather say that here than
have it found out with a stopwatch.

**Speed with sampling (temperature above 0).** Most of the speed figures we have published were
measured with greedy requests (temperature 0), one request at a time, often on code. Requests
that sample, which is what a client gets when it sends no sampler settings, are slower: fewer of
the tokens the engine drafts ahead are accepted, so each round yields less.

- The headline decode figures for Qwen3.8-Flash-Next (137 / 186 / 221 tok/s on one / two / four
  machines) are for a single request generating code with low reasoning effort. On the same
  four-machine build we measured 228 tok/s for greedy code, 121 tok/s for greedy prose and about
  100 tok/s for a single sampled request at temperature 0.8.
- On one machine (3.05 bpw pack, 512-token replies, one request): a code prompt ran at about 94
  tok/s greedy and about 97 tok/s sampled with thinking off (temperature 0.7). With thinking on
  (temperature 1.0) the same prompt ran at 59 to 69 tok/s: the engine drafted 2.6 tokens ahead
  per round and 58 % of them were accepted, against 5.8 and 68 % with thinking off. We have not
  yet separated how much of that drop is the temperature and how much is the text itself
  (reasoning text is harder to draft than code).
- The settings that steer speculation (how far ahead to draft, when to stop drafting, when to
  switch speculation off) were chosen on greedy measurements. They have not been re-tuned for
  sampled text.

**Two to four requests at the same time.** With two, three or four requests running, the server
either lets them take turns or serves all of them in one shared step with speculation switched
off. In both cases the **total** is about what a single request gets, and each request gets a
fraction of it.

- Measured on one machine with sampled requests (4.05 bpw pack, a single run): each request ran
  at about 25 tok/s with two at once and about 15 tok/s with four.
- The concurrency gains published earlier (for eight and sixteen requests) were measured with
  greedy 256-token code requests.
- **New in this build, ON by default: lane packing (`--lane-pack`; `--lane-pack off` restores
  the previous behaviour)** (see its section for every figure and its request class). Two
  requests share one verification pass. These figures are what was measured, and they are
  smaller than the headline suggests: on one machine, with sampled requests and short prompts,
  two together ran 1.15 to 1.34 times faster; on two machines, with short prompts of about 75
  tokens and 512 forced tokens, 1.17 to 1.32 times faster, and on agent-style prompts of about
  5,500 tokens (a first turn, nothing cached) about 12 % at best with thinking on and within
  noise with thinking off; with three requests there is no gain; with four requests between 8 %
  slower and about 14 % faster depending on the text. A conversation whose prompt is already
  cached was not measured. A request that carries a repetition, presence or frequency penalty,
  or an image, never takes part. It costs about 1.7 to 1.8 GiB of device memory per machine.
  A failed packed round ends the engine under TP (loudly, on both ranks).

**A new request pauses the others while its prompt is read.** The prompt of an arriving request
is read in one go, and the requests that are already streaming stand still until that is done.
On one machine a prompt of about 20,000 tokens paused two running streams for about 14 seconds;
`--prefill-interleave` reduces that to about 5 seconds there (and to 1.4 to 2.7 s with a smaller
`--prefill-chunk`, see its section). On two and four machines that option is not available, and we
have not measured the pause. (On one machine at default flags, a fresh 4,566-token prefill
coincided with a 3.4 s gap in a second running stream.)

**The prompt cache belongs to a request slot.** A conversation's next turn reuses its cached
prompt only if the slot that holds it is free. With several conversations interleaved, one can
push another out, and the whole prompt is then read again (and the others pause, as above). In
one real agent session on one machine, such re-reads took 23 % of the wall time.

**Requests with a presence, repetition or frequency penalty.** A penalty adds a step to every
token. The only measurements we have are single runs: about 6 to 18 % slower per token with a
presence penalty of 1.5 on one machine, and 73.6 against 107.3 tok/s for two such requests on two
machines. Such requests also do not take part in `--lane-pack`, and on several machines they use
a heavier exchange between the machines.

**Tool calls that quote tool calls.** See "Fixed — tool calls that quote tool calls": a value that
quotes a call with two or more parameters is still read as a cut value plus extra calls at
default flags. Two opt-in readings exist and neither is the default yet; `--tool-shadow-cost on`
counts how often they would differ on a server's own traffic, which is how the default will be chosen.

**Not run on hardware for this release** (every statement above that names a request class was
measured on that class and no other):

- **What was exercised on this build, stated plainly.** The maintainer exercised a single
  machine, a single machine at `--max-batch 1`, TP=2 and TP=4 by hand, on the release build
  before lane packing became the default (the flag typed `--lane-pack on`). No automated gate
  chain was run on this exact binary, TP=4 included: the gates of the candidate
  `7a4117643a52` (TP=1 and a TP=2 pair) are carried over by argument, since the changes after
  it do not touch the serving path except as follows. The default path (the flag NOT typed)
  differs from the typed-flag path only in how the option is resolved and agreed between
  machines; that code was read, not booted by a test session; `--tool-unclosed drop` as the
  default is the v0.7.3 reading, which the in-tree differential tests pin.
- Four machines (TP=4): no gate chain of this release was run on four machines. The maintainer
  ran TP=4 with packed rounds by hand for daily use with no crash or error noted (no numbers
  are published). Until a gate chain is run, "not gated on four machines" is the accurate
  statement. A warm multi-turn agent conversation with lane packing, and the four-request off/on
  speed pairs after the shared-step probe of this release, were not measured.
- The DeepSeek V4 server (the streamer move and the greedy `timings` correction), DFlash under
  TP, images together with `--lane-pack`, and the NVFP4 `--rope-yarn-factor 2 --max-seq-len
  524288` guard.
- Long runs: the longest uninterrupted runs behind this release are 30 minutes at TP=2 with
  lane packing on (782 streamed requests), 45.5 minutes at TP=2 at default flags on a
  development build, and 159.8 minutes at TP=2 on the published v0.7.3 binary. No soak of the
  v0.7.4 release binary was run; the maintainer will schedule a long comprehensive soak after
  the release.

### What's next

The next release (v0.7.5) is about one thing: **speed for the requests people actually send.**
In this order:

1. **Measure first, and publish it.** A baseline at the settings a default client uses (sampling
   at the model's defaults, thinking on, one to four requests at once, on one, two and four
   machines, with realistic prompt sizes). From then on every speed figure we publish states its
   sampler, the number of concurrent requests and the number of machines, and greedy figures are
   labelled as such.
2. **Speculation tuned for sampled text.** Draft depth, the acceptance rule and the stop
   thresholds re-measured and re-set at those settings instead of greedy.
3. **Two to four concurrent requests.** Packing for three and four requests, on several machines,
   and for requests with a penalty (that code is written and has not yet run on hardware).
4. **A new request must not freeze the others**, on one machine and on several.
5. **A prompt cache that several conversations can share.**

Direction, without numbers: a production baseline at default sampling on one, two and four
machines; a review of the lane-packing results and the defaults that follow from it; a three-
or four-request pack; the routed-expert kernel; moving the hand-read flags into the registry;
YaRN (issues #19 and #22); a long soak and TP=4 gate numbers.

We will not put numbers on these before we have measured them. Also known and not in this
release: the Hy3 and DeepSeek V4 tool-call formats (parked until those models are in scope),
unclosed tool-call blocks cut inside an argument (still dropped), YaRN for the EXL3 path
(issues #19 and #22), and moving the hand-read flags into the option registry.

## v0.7.3 — reported-issue fixes: TP=2 pre-verify race, exllamav3 1.5.x pack loading, cached-token accounting; tool-call parsing; long-running hardening

This release closes the three open issues from the v0.7.2 feedback (#8, #9, #10), makes the
server tell the truth about cached prompt tokens, stops a stalled log reader (or a slow OTLP
receiver, or a client that stops reading a stream) from stalling the engine, and adds the
operational surface (gauges, admission bounds, an operations guide) for running the server for
hours at a time. The two tarball launcher scripts changed interface: they take flags now, not
environment variables.

### Fixed — TP=2 `TP pre-verify FAILED` (issue #10)

Reported by [@JashicTM](https://github.com/JashicTM) (issue #10).

At world size 2, the host-side control exchange staged its RDMA frames into the same ring the
device doorbell epochs use: when the exchange's slot counter happened to alias the slot holding
the last draft-pass epoch payload, the first bytes of that payload were overwritten and the
pre-verify hash compare fired — the `TP pre-verify FAILED` abort reported against
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

Reported by [@MushroomMan321](https://github.com/MushroomMan321) (issue #9).

exllamav3 1.5.x writes the n-gram table as 128 shard tensors in the sidecar, and most public
Flash-Next EXL3 packs now use that layout; some of them also omit the shards from
`model.safetensors.index.json`. v0.7.2 refused those packs at boot (`ple shard 0 not in
index`). v0.7.3 loads both layouts as shipped — single-tensor and 128-shard, K=5 and K=6,
index-listed or not (the loader falls back to the sidecar's own safetensors header and
validates the layout; a boot line names the one in use). No flags, no workarounds: the
header-rewrite workaround some users applied is no longer needed. Packs that loaded before are
byte-identical before and after (gated). A tested-packs table is in the Flash-Next setup guide.

### Server surface (issue #8)

Reported by [@liorm0505](https://github.com/liorm0505) (issue #8; the same report was first filed as #7 by [@liorzivsensors-cmd](https://github.com/liorzivsensors-cmd)).

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

## v0.7.2 — concurrent requests on the EXL3 path, Qwen3.8-Flash-Next 4.05 bpw, `/metrics`, `--host`, and reported-issue fixes

This release answers a community benchmark of Flash-Next as a coding-agent executor (an apt test of
the engine), adds the higher-fidelity 4.05 bpw pack, and fixes what was found along
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
  in long sessions; issue #6, reported by [@JashicTM](https://github.com/JashicTM)).
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
community PR #4 by [@kedric](https://github.com/kedric).

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
  v0.5.0 boot crash on non-27B packs; issue #2, reported by [@npw1980](https://github.com/npw1980)). Vision is generalized across the Qwen3.5/3.8 VL family, so
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
