//! TP=2 runtime context: rank/world + the RDMA data-plane link, brought up AFTER the cluster has
//! discovered the peer and synced the model. Head = rank 0 (listens for the QP handshake), node =
//! rank 1 (connects to the head's RoCE IP that it saw during the TCP sync).
//!
//! The link (`net::TpLink`) is the inference data plane for the sharded forward's all-reduces.

use crate::net::TpLink;
use anyhow::{Context, Result};
use std::net::IpAddr;

/// net_shim QP-bootstrap TCP port (distinct from the cluster control plane on 29500).
pub const TP_PORT: u16 = 29600;
/// One slot must hold the widest all-reduce payload we will ever ship: hidden * 4 (FP32) * verify batch.
/// 128 KB covers hidden=5120 at FP32 up to batch 6, or bf16 up to batch 12 — i.e. an MTP verify at any
/// depth we would plausibly run. Cost is ~4 MB pinned for both rings; the reason not to size it huge is
/// that the ring addresses are baked into a captured CUDA graph, so changing it later forces a
/// re-capture. Sized once, deliberately. (At 64 KB a batch-8 bf16 forward failed the payload guard.)
pub const TP_SLOT_BYTES: usize = 1024 * 1024;
pub const GID_IDX: i32 = 3;
const DEFAULT_RDMA_DEV: &str = "rocep1s0f1";

// ---------------------------------------------------------------------------------------------------
// TP settings distribution (TP item C)
// ---------------------------------------------------------------------------------------------------
// The contract is "nodes just run `--node`": the head ships its TP config to the node during the
// cluster sync (`Msg::Config`), and the node runs with ZERO GB10_TP_* env vars yet reproduces the
// head's behavior. Env vars remain as overrides for benches. Resolution rule at every consumer:
// env var present → env wins; else this process-global config (if installed); else default (same
// default as the no-env behavior). Flags are presence-based, matching the old `is_ok()` semantics.

/// Wire default for `df2_block` when an older peer omits the field: the shipped block (8).
fn df2_block_default() -> usize { crate::dflash2::BLOCK }

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TpConfig {
    pub config_version: u32,          // = 25 (v25: + ship_shards / sync_only, B32 per-rank shard shipping;
                                      //      v24: + exl3_pack_files, PACK-FIX's per-file pack manifest;
                                      //      v23: exl3_env -> opts, the CLI-1 option registry;
                                      //      v22: + exl3_seq_parallel, TP-SP1's --tp-seq-parallel;
                                      //      v21: + exl3_vp_sampled, TP-H2's --tp-vp-sampled;
                                      //      v20: exl3_pf_overlap gains the `fold` code bit 30, TP-I3 (c1);
                                      //      v19: + exl3_pf_overlap, TP-I2's --tp-prefill-overlap; v18: the EXL3 TP fields;
                                      //      v17: + pf_sched_inline, the P11 --prefill-sched escape;
                                      //      v16: + df2_carry, the DFlash2 prefix-cache carry;
                                      //      v15: + df2_block, the DFlash2 runtime draft block;
                                      //      v14: + df2_step_dump, the Phase-0 coverage-trace SPMD flag;
                                       //     v13: + df2_round_shard, the P2 round-sharding flag;
                                       //     v12: + spec_source/df2_draft_dir, the TP-DF2 leg
                                       //     — S9F; v11: + decode_ctx, the DecodeCtx probe branch riding
                                       //     the config — P0-2; v10: + shard_mtp, the MTP-block
                                       //     sharding flag; v9:
                                      //     + topology, the full rank->RoCE-IP map, and
                                      //     node_rank, the per-node rank shipped by the head;
                                      //     v9: + kv_k8v8 (int8 K+V blocks); v8: + world, the TP rank count; v7: + kv_k8v4, the
                                      //     int8-K / q4-V cache mode; reduce_fuse rode v6,
                                      //     gpu_recv rode v5)
    pub world: u32,                   // TP rank count — the --tp CLI flag is the only authority (bare = 2)
    pub shard_mixers: bool,           // [tp-shard-mixers]
    /// head's --tp-shard-mtp (--tp-shard-mtp env alias, harness/diagnostics). Shards the MTP
    /// draft block (fc m-slice + attn heads + FFN/experts) at attach — the draft path then carries
    /// reduce sites. SPMD-critical: a one-sided shard is a weight-layout + barrier-sequence
    /// mismatch, so it rides the config to every node. DEFAULT ON (user decision 2026-08-15:
    /// +8.8% under MTP load at TP=4, LOSSLESS-gated); --tp-shard-mtp=0 opts out.
    pub shard_mtp: bool,
    pub graph: bool,                  // --tp-graph
    pub fp32_partials: bool,          // --tp-fp32-partials
    pub trace: bool,                  // --tp-trace
    pub mtp: bool,                    // [tp-mtp]
    pub mtp_depth: Option<usize>,     // [tp-mtp-depth]
    pub batch_probe: Option<usize>,   // --tp-batch-probe
    pub step_probe: Option<usize>,    // --tp-step-probe
    /// head's --tp-decode-ctx (one-shot DecodeCtx probe branch). Presence-based with the
    /// dispatch's own parse-or-2048 default: the shipped value must equal what the head itself
    /// resolved, garbage values included, or head and node take different TpBranch programs —
    /// the 2026-08-15 "rank 1 selector 0x00, head 0x60" split-brain.
    pub decode_ctx: Option<usize>,    // --tp-decode-ctx
    pub prefill_payload: Option<usize>, // --tp-prefill-payload (all-reduce chunk cap, bytes)
    pub accept: Option<usize>,        // --tp-accept (bench_accept depth; node runs it too — SPMD)
    pub capture: Option<String>,      // --tp-capture (debug dump path; node runs it too — SPMD)
    // ---- v2: serving mode (TP item A). `from_opts` fills these with defaults (bench mode is
    // unaffected); the head's `--server --tp` branch fills them from the server args. The node
    // needs them to build a BatchScheduler identical to the head's with ZERO env of its own.
    pub mode_serve: bool,             // false = one-shot bench session; true = resident OpenAI server
    pub max_seq_len: usize,           // head's (clamped) --max-seq-len = both ranks' kv_stride
    pub max_batch: usize,             // head's --max-batch; also the all-reduce payload width
    pub prefix_cache: bool,           // head's --prefix-cache
    pub ngram_draft: usize,           // head's --ngram-draft
    pub tree_draft: bool,             // head's --tree-draft
    pub mtp_lanes: bool,              // head's --mtp-lanes
    /// P11 W5a — head's --prefill-sched. false (default) = the CURSOR schedule (one prefill window
    /// per scheduler iteration; SCHEDULER_2LANE, owner-blessed default-ON). true = "inline": drain
    /// the whole prefill inside admission (the pre-P10 posture; max decode gap = the full prefill).
    /// Scheduler-only (kernel ORDER, not numerics), but the node mirror must resolve the SAME value
    /// or the SPMD op sequence diverges — so it rides the config. #[serde(default)] keeps an older
    /// peer's config parseable (missing => cursor).
    #[serde(default)]
    pub pf_sched_inline: bool,        // head's --prefill-sched inline
    pub mtp_force: Option<bool>,      // head's --mtp=on|off (None = auto)
    pub mtp_depth_pin: Option<usize>, // head's --mtp-depth
    pub fp8_prefill: bool,          // head's --fp8-prefill (native e4m3 prefill GEMM
                                    // lane; node installs the env so both ranks dispatch
                                    // identically — a head-only lane would split-brain TP2)
    pub no_decode_graphs: bool,       // head's --no-decode-graphs (env-read; node installs as env)
    pub cpu_sample: bool,             // head's --cpu-sample (env-read; node installs as env)
    pub no_verify_graph: bool,        // head's --no-verify-graph (env-read; node installs as env)
    pub kv_quant: bool,               // head's [kv-quant] (4-bit KV cache; node installs as env)
    pub kv_tq: bool,                  // head's [kv-tq]=1 (3.5-bit TurboQuant KV, E4; node installs as env)
    pub kv_tq_b3: bool,               // head's [kv-tq]=3 (TurboQuant b=3 K variant; node installs as env)
    /// head's [kv-k8v4]=1 (int8-K + q4-V k8v4 cache; node installs as env). Mutually exclusive
    /// with kv_quant/kv_tq — the cache layout must match on both ranks (SPMD).
    pub kv_k8v4: bool,                // head's [kv-k8v4] (k8v4 KV cache; node installs as env)
    /// int8 K+V per-16-block cache (20 B/16 both channels) — the direct-read substrate for the
    /// p8b verify kernel. VALUE-based like kv_k8v4: only [kv-k8v8]=1 enables it.
    pub kv_k8v8: bool,                // head's [kv-k8v8] (k8v8 KV cache; node installs as env)
    pub fuse_residual: Option<bool>,  // head's --fuse-residual (None = default ON; node installs as env)
    pub device_loop: bool,            // head's --device-loop (device-resident token loop; node installs as env)
    pub gpu_recv: Option<bool>,       // v2 GPU-direct all-reduce receive (--tp-gpu-recv; node installs as env)
    /// AR landing 2: fused reduce+residual+norm epilogue (--tp-reduce-fuse; node installs as
    /// env). SPMD-relevant: the fused kernel replaces the K2 + norm two-launch chain at the
    /// mixer/FFN epilogue sites, so both ranks must fuse the same launches or the barrier chains
    /// diverge.
    pub reduce_fuse: Option<bool>,
    pub eos: Vec<u32>,                // head's stop-token set (node has no tokenizer)
    pub calib_prompt: Vec<u32>,       // head-encoded "The capital of France is" probe ids
    /// head's --bench-dspark / --dspark flag (DSpark speculation serve). The head sets it; the node
    /// reproduces it via the shipped config → both ranks take the DSpark branch (SPMD). The DSpark
    /// stages load from the replicated `{rank}/dspark.safetensors`.
    pub dspark: bool,
    /// head's --dspark-depth N (item 3.3 adaptive draft depth): None = the policy re-picks the
    /// drafted-row count every 128 steps from the measured r(D) table; Some(n) pins it. SPMD-
    /// relevant: the node must draft the same width as the head or the verify all-reduces diverge.
    pub dspark_depth: Option<u32>,
    /// head's --exact-gemm flag (tolerance-class fast paths DEFAULT ON; the locked bit-exact
    /// kernels selectable via this flag, item 2.5). Rides TpConfig so the zero-config node takes
    /// the same kernel path as the head (the olo/compressor dispatch is SPMD-relevant state).
    pub exact_gemm: bool,
    /// head's --splitk-gemm flag (NVFP4 serving-GEMM split-K; DEFAULT OFF — the split
    /// reassociates the fp32 k-sum and the A/B verdict is the user's call). Rides TpConfig so the
    /// zero-config node takes the same GEMM dispatch as the head: the split geometry is a function
    /// of the weight shape on BOTH ranks, so it is SPMD-consistent by construction, but the TP
    /// Cf split path still needs its two-box binv gate before any TP=2 claim.
    pub splitk_gemm: bool,
    /// head's --mxfp4 flag (QWEN_MXFP4_NATIVE_DESIGN.md — fp4 decode/verify GEMMs on the sm_121a
    /// OMMA path; DEFAULT OFF = the bit-exact bf16 chain). Rides TpConfig so the zero-config node
    /// builds the SAME OMMA repacks and dispatch as the head (SPMD-relevant: rank-local weights
    /// are sharded copies, and a chain divergence would desync the verify all-reduces).
    pub mxfp4: bool,
    /// head's --mxfp4-mtp-native escape hatch (the MTP head allowlist — acceptance-gated).
    /// SPMD-relevant: both ranks must make the same allowlist decision or the MTP draft chain
    /// diverges across ranks. Ships with the config; the node installs it as env before load.
    pub mxfp4_mtp_native: bool,
    /// head's --server-dspark <on|off> (item 3.4 — DSpark speculation in the persistent DSV4
    /// --server path). ON by default at the CLI (user decision 2026-08-05, 3.4 VERIFIED; pass
    /// off for greedy — byte-identical to pre-flag); ON routes every request through the
    /// DSpark draft/verify/rollback loop, both ranks SPMD. (The struct default here stays
    /// false: default() is the probe/bench fallback, not the serving path.)
    pub server_dspark: bool,
    /// head's --dspark-fp8-head <on|off> (item 1.7(i) / T3): fp8_bsb draft LM head + Markov W2
    /// (halve the draft's head reads). ON by default at the CLI (user decision 2026-08-05;
    /// pass off for bf16); rides TpConfig so the zero-config node builds the SAME fp8 arms as
    /// the head (a draft-logits divergence would desync the acceptance/verify SPMD sequence).
    /// Draft-side only — LOSSLESS preserved, acceptance may shift at near-ties.
    pub dspark_fp8_head: bool,
    /// E5: trunk YaRN rope factor (1.0 = legacy; see GpuModel::build_rope_tables).
    pub rope_yarn_factor: f32,
    /// E29-B3 DFlash drafter: --tp-dflash=1 routes the one-shot Generate through the
    /// draft-8-verify-accept loop. SPMD-relevant: the drafter runs on rank 0 ONLY (the node
    /// never loads it); the 7 draft tokens are shipped to the node over the link before the
    /// verify, and BOTH ranks verify + accept identically. Ships so the zero-config node takes
    /// the same Generate branch (a one-sided env would silently desync the verify all-reduces).
    pub dflash: bool,
    /// head's `--df2-capture` (S4F DFlash2 trunk tap capture; node installs as env
    /// [df2-capture]). DEFAULT OFF — with it off the capture is a strict no-op (zero
    /// launches, zero host work; the R1 timing-free proof). Ships on the config so a future
    /// TP deployment captures on BOTH ranks or neither (a one-sided capture is dead weight
    /// but config-shipped beats env drift).
    pub df2_capture: bool,
    /// E12 fold escape: head's --moe-no-fold (fold the shared expert into the grouped MoE
    /// launches as ONE extra slot; =1 restores the separate-launch shared MLP). SPMD-relevant:
    /// the fold changes the launch sequence on both ranks; a one-sided escape would desync the
    /// all-reduce epochs. Node installs it as env before load (the loader reads the env).
    pub moe_fold: bool,
    /// E8 shard escape: head's --e8-no-shard (=1 keeps the shared expert replicated instead of
    /// the paired ColSegs gate+up + row-parallel down). SPMD-relevant: the shard changes the
    /// WEIGHT LAYOUT at load — a one-sided escape is a weight-layout mismatch, not just a
    /// numerics change. Node installs it as env before load.
    pub e8_shard: bool,
    /// E9 PDL escape: head's --e9-no-fold (presence-based — any value disables the
    /// programmatic dependent-launch overlap; =1 restores the plain barrier path). SPMD-relevant:
    /// both ranks must make the same launch-attribute decision or the barrier chain diverges.
    /// Node installs it as env before load.
    pub e9_fold: bool,
    /// P3-1 one-shot all-peers push (world==4 only, DEFAULT OFF). SPMD-critical: it selects the
    /// ring LAYOUT at transport init (sender-indexed recv rings) — the node installs the env from
    /// this field before link bring-up. Head env: --tp-oneshot=1 (any value but "0").
    pub oneshot: bool,
    /// P4: full rank→RoCE-IP topology, indexed by rank (`topology[rank]` = that rank's RoCE IP;
    /// `topology[self_rank]` is this process's own and unused by the N-way transport). Populated
    /// by the head after discovery and shipped to every node via `Msg::Config`; `from_opts` leaves
    /// it empty (empty = no discovered topology — only valid for world==2 or non-cluster paths).
    pub topology: Vec<String>,
    /// P4: this node's rank, shipped by the head (the head sets it per-node). The node is
    /// zero-config: it reads its rank + world + topology from the shipped config rather than
    /// deriving anything locally. Head processes ignore it (rank 0); `from_opts` defaults to 1 for
    /// the world==2 single-node bench fallback.
    pub node_rank: i32,
    /// S9F (the TP-DF2 leg): the head's resolved `--spec-source` CLI name ("" = pre-v12 configs).
    /// Ships so the zero-config node builds the IDENTICAL MtpPolicy source — a one-sided source
    /// (head = DFlash2, node = MTP) would take different lane branches and desync the verify
    /// all-reduces. The node's actual round residency comes from the post-load CalibTable
    /// message (the head only knows its round load's outcome after its own load).
    pub spec_source: String,
    /// S9F: the head's resolved `--draft-dir` (the DFlash2 artifact dir). Ships so the node
    /// loads the SAME artifact bytes at the SAME path — the round's drafts must be bit-identical
    /// across ranks or the verify all-reduces diverge.
    pub df2_draft_dir: String,
    /// S9F+ (2026-08-29): the head's `--sha256 <hex|off>` override for the DFlash2 artifact pin.
    /// None = the published REAL_SHA256 pin (default). Some("off") = no sha check (the
    /// inventory/shape/dtype guard still runs). Some(hex) = pin to that exact artifact hash.
    /// Ships so the node loads the same artifact under the same pin — a one-sided pin is a
    /// round-load mismatch (head loads, node refuses, or vice versa) and must never happen.
    #[serde(default)]
    pub df2_sha_pin: Option<String>,
    /// P2 (v13): head's `--df2-round-shard <on|off>` — shard the DFlash2 drafter round across
    /// the TP ranks (qkv/gate/up col-split, o/down K-split, per-head ring KV, two trunk-class
    /// all-reduce sites per layer on the round's stream, in-capture). SPMD-critical: a one-sided
    /// shard is a weight-layout + barrier-sequence mismatch (the sharded round contributes 10
    /// logical all-reduces per step the unsharded round does not), so it rides the config to
    /// every node. Engages only at world > 2 (the quad campaign target); DEFAULT OFF until the
    /// Phase D quad truth flips it.
    #[serde(default)]
    pub df2_round_shard: bool,
    /// DF2 block-16: the head's resolved `--df2-block` (8 default, 16 opt-in). SPMD — every rank
    /// must resolve the SAME block, because the round's 15 MASK rows and the verify's 16 columns
    /// are collective: a one-sided block would desync the verify all-reduce. Absent/0 on the wire
    /// (an old head) means the default 8 via `df2_block_default()`.
    #[serde(default = "df2_block_default")]
    pub df2_block: usize,
    /// DF2_CARRY: keep DFlash2 in the draft seat across a prefix-cache hit (default OFF until
    /// `PLAN/DF2_CARRY_SPEC.md` §7 is green). SPMD — every rank must make the SAME admit
    /// decision, or the carried lane would desync the verify all-reduce. Absent (an old head)
    /// means off via serde's `bool` default.
    #[serde(default)]
    pub df2_carry: bool,
    /// head's --df2-step-dump (PLAN/25 Phase 0 coverage trace). SPMD: the node mirror must run
    /// the identical trace op sequence (eager keep-logits verify, logging MTP chain) — it sets
    /// `cov_trace` from this flag and leaves `step_dump` None (only the head writes records).
    #[serde(default)]
    pub df2_step_dump: bool,
    /// P3(a) close: route GREEDY (temp-0) General requests to GREEDY drafts — DEFAULT ON since
    /// the 2026-08-23 quad temp-0 sweep (prose tau +10.5% step-weighted, code control
    /// bit-identical). SAMPLED (temp>0) General keeps the real-q walk (greedy drafts regress
    /// flat sampled targets: chat_t1_off 1.010x -> 0.865x vs MTP). `--df2-prose-lane rq`
    /// restores the unconditional sampled real-q walk. Resolves in the head's
    /// `df2_effective_src`; SPMD-critical because the node runs the IDENTICAL `decode_step` —
    /// a one-sided routing choice would desync the verify all-reduces. The head ships this bit
    /// to every node at sync (no env side channel).
    #[serde(default)]
    pub df2_prose_lane_greedy: bool,
    /// Phase-7 F9 gate: --tp-binv-tp / the head's --probe-binv-tp — run the TP
    /// batch-invariance probe (verify-width sweep N=1..MAX_VERIFY through the real
    /// sharded verify path) instead of generation. SPMD-critical: a one-sided flag would
    /// run the probe on one rank and generation on the other, i.e. mismatched all-reduce
    /// epochs; it rides TpConfig plus the branch selector like every other probe.
    #[serde(default)]
    pub binv_tp: bool,
    /// Phase-7: --tp-state-tp — run the GDN decode-vs-verify state probe on the sharded
    /// path (the `--probe-state` gate is world=1 only). SPMD: both ranks run the identical
    /// forwards and compare their own state slices.
    #[serde(default)]
    pub state_tp: bool,
    // ---- v18 (TP-A, EXL3 TP bring-up — PLAN/TP2_DESIGN_EXL3_2026-09-28.md §5). All
    // #[serde(default)] so an older peer's config stays parseable (and reads as "not EXL3").
    /// The session runs the EXL3 engine (FwdModel) — the node routes to the EXL3 TP path.
    #[serde(default)]
    pub exl3: bool,
    /// The head's EXL3 pack dir. Legacy (D-T0-6): the node loads ITS OWN local copy at the same path. Nothing
    /// checks that copy against the head's — the manifest check was removed by the owner in 074c669
    /// (2026-09-30); `exl3_manifest` / `exl3_pack_files` below are no longer filled. With B32 `ship_shards`
    /// the node replaces this with the shard dir the head shipped it through the blob cache.
    #[serde(default)]
    pub exl3_pack_dir: String,
    /// sha256 over the pack's (relative path, sha256, size) list (cluster::pack_manifest) — PACK-FIX:
    /// over the files the engine READS only (index-referenced safetensors + sidecars + the loader /
    /// tokenizer / serve small files), never README/LICENSE/other files.
    #[serde(default)]
    pub exl3_manifest: String,
    /// v24 (PACK-FIX): the head's per-file manifest list behind `exl3_manifest` — a refusing node
    /// names each differing file (head hash vs node hash) in the failure it reports to the head.
    #[serde(default)]
    pub exl3_pack_files: Vec<(String, String, u64)>,
    /// The static expert deal ("interleave" | "contig") — fixes each rank's expert GROUPING, so
    /// it is numerics-affecting (SPMD).
    #[serde(default)]
    pub exl3_ep_deal: String,
    /// v23 (CLI-1): the head's resolved option registry — every set, non-node-local option as
    /// (flag, value) (`opts::snapshot`, filled at ship time in cluster::head_sync_one). The node
    /// installs exactly this set (`opts::install_head`, every mode, before anything reads an option)
    /// and clears any non-local option of its own the head did not set — "one knob, set on the head,
    /// rides TpConfig" (AGENTS §7). Replaces v18..v22's env snapshot `exl3_env`.
    #[serde(default)]
    pub opts: Vec<(String, String)>,
    /// The EXL3 TP program: "xtp" (the rung-3 cross-TP probe). Unknown = refuse.
    #[serde(default)]
    pub exl3_mode: String,
    /// v25 (B32): the head shipped this node ONLY its rank's shard dir through the blob cache (shard_plan) —
    /// always, for every model the shard plan covers (no switch, owner 2026-10-02); the node loads the assembled
    /// cache dir, never the head's path. false = a model the plan does not cover (its own existing path).
    #[serde(default)]
    pub ship_shards: bool,
    /// v25 (B32, `--tp-sync-only`): sync, report, exit (head and nodes) — no model load.
    #[serde(default)]
    pub sync_only: bool,
    // ---- v19 (TP-I2): CLI-flag knobs of the EXL3 TP engine (owner rule 2026-09-29: new options are
    // CLI flags riding TpConfig, never env vars).
    /// The head's resolved `--tp-prefill-overlap` code (rows per hook | dual << 31 | fold << 30 (v20); 0 = off). The node
    /// installs exactly this value (xtp::set_pf_overlap); None (a pre-v19 head) = off.
    #[serde(default)]
    pub exl3_pf_overlap: Option<u32>,
    /// v21 (TP-H2): the head's resolved `--tp-vp-sampled` code (1 = the vocab-parallel lm_head for sampled /
    /// penalized / ratio-rule rows, 0 = the replicated head for them). The node installs exactly this
    /// value (xtp::set_vp_sampled); None (a pre-v21 head) = off.
    #[serde(default)]
    pub exl3_vp_sampled: Option<u32>,
    /// v22 (TP-SP1): the head's resolved `--tp-seq-parallel` code (1 = sequence-parallel prefill, 0 = the
    /// replicated prefill). The node installs exactly this value (xtp::set_seq_parallel); None (a pre-v22
    /// head) = off.
    #[serde(default)]
    pub exl3_seq_parallel: Option<u32>,
}

impl TpConfig {
    /// Snapshot the GB10_TP_* env vars (flags = presence; probes/depth = parse). Serving fields get
    /// their v1-compatible defaults — a bench config is indistinguishable from before.
    pub fn from_opts() -> Self {
        eprintln!("[from_opts] [kv-k8v8]={:?}", crate::opts::var(crate::opt!("kv-k8v8")).ok());
        TpConfig {
            config_version: 25,
            // TP rank count. The --tp CLI flag is the single authority for a TP run (bare --tp = 2,
            // --tp N = N); `from_opts` only snapshots the bench `--head` path, which is always TP=2.
            // GB10_TP_WORLD is deliberately NOT read here — one source of truth, zero ambiguity.
            world: 2,
            shard_mixers: crate::opts::var(crate::opt!("tp-shard-mixers")).is_ok(),
            // Value-based, DEFAULT ON (user decision 2026-08-15: +8.8% measured under MTP load at
            // TP=4, LOSSLESS-gated; CLI flag --tp-shard-mtp still forces it). =0 is the opt-out.
            shard_mtp: crate::opts::var(crate::opt!("tp-shard-mtp")).ok().map_or(true, |v| v != "0"),
            graph: crate::opts::var(crate::opt!("tp-graph")).is_ok(),
            fp32_partials: crate::opts::var(crate::opt!("tp-fp32-partials")).is_ok(),
            trace: crate::opts::var(crate::opt!("tp-trace")).is_ok(),
            mtp: crate::opts::var(crate::opt!("tp-mtp")).is_ok(),
            mtp_depth: crate::opts::var(crate::opt!("tp-mtp-depth")).ok().and_then(|v| v.parse().ok()),
            batch_probe: crate::opts::var(crate::opt!("tp-batch-probe")).ok().and_then(|v| v.parse().ok()),
            step_probe: crate::opts::var(crate::opt!("tp-step-probe")).ok().and_then(|v| v.parse().ok()),
            // Same resolution as the dispatch (parse-or-2048): what ships is what the head ran.
            decode_ctx: crate::opts::var(crate::opt!("tp-decode-ctx")).ok().map(|v| v.parse().unwrap_or(2048)),
            prefill_payload: crate::opts::var(crate::opt!("tp-prefill-payload")).ok().and_then(|v| v.parse().ok()),
            // Presence-based like the branch ladder itself: set (even unparsable) means the branch,
            // defaulting depth to 2 — a garbage value must ship the SAME resolution the head makes,
            // or head and node would take different branches.
            accept: crate::opts::var(crate::opt!("tp-accept")).ok().map(|v| v.parse().unwrap_or(2)),
            capture: crate::opts::var(crate::opt!("tp-capture")).ok(),
            mode_serve: false,
            max_seq_len: 0,
            max_batch: 0,
            prefix_cache: false,
            ngram_draft: 0,
            tree_draft: false,
            mtp_lanes: false,
            pf_sched_inline: false,
            mtp_force: None,
            mtp_depth_pin: None,
            no_decode_graphs: false,
            fp8_prefill: false,
            cpu_sample: false,
            no_verify_graph: crate::opts::var(crate::opt!("no-verify-graph")).is_ok(),
            kv_quant: crate::opts::var(crate::opt!("kv-quant")).is_ok(),
            // VALUE-based (not presence): only [kv-tq]=1 (b=2 K) or =3 (b=3 K) enables TQ;
            // =0 restores the default path byte-for-byte (the E4 escape-hatch acceptance — see
            // gpu::kv_modes_from_opts).
            kv_tq: matches!(crate::opts::var(crate::opt!("kv-tq")).ok().as_deref(), Some("1") | Some("3")),
            kv_tq_b3: crate::opts::var(crate::opt!("kv-tq")).ok().as_deref() == Some("3"),
            // VALUE-based like kv_tq: only [kv-k8v4]=1 enables the k8v4 mode.
            kv_k8v4: crate::opts::var(crate::opt!("kv-k8v4")).ok().as_deref() == Some("1"),
            kv_k8v8: crate::opts::var(crate::opt!("kv-k8v8")).ok().as_deref() == Some("1"),
            fuse_residual: crate::opts::var(crate::opt!("fuse-residual")).ok().map(|v| v != "0"),
            device_loop: crate::opts::var(crate::opt!("device-loop")).map_or(false, |v| matches!(v.as_str(), "1" | "on" | "true")),
            gpu_recv: crate::opts::var(crate::opt!("tp-gpu-recv")).ok().map(|v| v != "0"),
            reduce_fuse: crate::opts::var(crate::opt!("tp-reduce-fuse")).ok().map(|v| v != "0"),
            eos: Vec::new(),
            calib_prompt: Vec::new(),
            dspark: false,
            dspark_depth: None,
            exact_gemm: crate::opts::var(crate::opt!("exact-gemm")).is_ok(),
            splitk_gemm: !crate::opts::var(crate::opt!("splitk-gemm")).is_ok_and(|v| v == "0"),  // default ON (E15); 0 = off
            mxfp4: crate::opts::var(crate::opt!("mxfp4")).is_ok(),
            mxfp4_mtp_native: crate::opts::var(crate::opt!("mxfp4-mtp-native")).is_ok(),
            server_dspark: false,
            dspark_fp8_head: crate::opts::var(crate::opt!("dspark-fp8-logits")).is_ok(),
            rope_yarn_factor: 1.0,
            dflash: crate::opts::var(crate::opt!("tp-dflash")).is_ok(),
            df2_capture: crate::opts::var(crate::opt!("df2-capture")).is_ok(),
            // E12/E8/E9 escapes: VALUE-based like [kv-tq] — only the explicit disable
            // (MOE_NO_FOLD=1 / E8_NO_SHARD=1) flips the flag; E9 is presence-based (any
            // --e9-no-fold disables — the backup gpu.rs resolution rule). The fold is
            // DEFAULT OFF (2026-08-11: fold-on + MTP degenerates hy3); --moe-fold=1 opts in.
            moe_fold: crate::opts::var(crate::opt!("moe-fold")).map_or(false, |v| v == "1")
                && crate::opts::var(crate::opt!("moe-no-fold")).map_or(true, |v| v != "1"),
            e8_shard: crate::opts::var(crate::opt!("e8-no-shard")).map_or(true, |v| v != "1"),
            e9_fold: !crate::opts::var(crate::opt!("e9-no-fold")).is_ok(),
            // P3-1 one-shot push (world==4): value-based, DEFAULT OFF (transport risk class —
            // gated by the P3-2 barrier bench + cell battery before any default change).
            oneshot: crate::opts::var(crate::opt!("tp-oneshot")).map_or(false, |v| v != "0"),
            // P4: no discovered topology in a from_opts snapshot (the head overwrites these before
            // shipping; the node reads them from the shipped config). world==2 never needs topology.
            topology: Vec::new(),
            node_rank: 1,
            // S9F (TP-DF2 leg): the from_opts snapshot predates the serving args — the head fills
            // these from its own --spec-source / --draft-dir before shipping; empty = "not set"
            // (the node falls back to the Mtp source, the pre-S9F behavior).
            spec_source: String::new(),
            df2_draft_dir: String::new(),
            df2_sha_pin: None,
            df2_round_shard: false,
            df2_block: crate::dflash2::BLOCK,
            df2_carry: false,
            df2_step_dump: false,
            df2_prose_lane_greedy: false,
            binv_tp: crate::opts::var(crate::opt!("tp-binv-tp")).is_ok(),
            state_tp: crate::opts::var(crate::opt!("tp-state-tp")).is_ok(),
            exl3: false,
            exl3_pack_dir: String::new(),
            exl3_manifest: String::new(),
            exl3_pack_files: Vec::new(),
            exl3_ep_deal: String::new(),
            opts: Vec::new(),
            exl3_pf_overlap: None,
            exl3_vp_sampled: None,
            exl3_seq_parallel: None,
            exl3_mode: String::new(),
            ship_shards: false, // set by the head per node when it ships shards (cluster::head_sync_one)
            sync_only: crate::opts::var(crate::opt!("tp-sync-only")).is_ok(),
        }
    }
}

static TP_CONFIG: std::sync::OnceLock<TpConfig> = std::sync::OnceLock::new();

/// Install the process-global TP config (head: its env snapshot; node: the head's `Msg::Config`).
pub fn set_tp_config(c: TpConfig) {
    // P3-1 one-shot: the ring-layout selector must be installed BEFORE any link bring-up (the
    // transport reads the env at init). set_tp_config is the single choke point both node paths
    // (bench one-shot + serving) pass through when the head's config arrives — install here so
    // the layout can never mismatch. The head installs its own env before its own bring-up.
    if c.oneshot && crate::opts::var(crate::opt!("tp-oneshot")).is_err() {
        crate::opts::set(crate::opt!("tp-oneshot"), "1");
    }
    eprintln!("[tp] config installed: world={} shard_mixers={} shard_mtp={} graph={} fp32_partials={} trace={} mtp={} \
               mtp_depth={:?} batch_probe={:?} step_probe={:?} mode_serve={} accept={:?} capture={:?} \
               dspark={} dspark_depth={:?} exact_gemm={} server_dspark={} fp8_head={} splitk_gemm={} \
               mxfp4={} mxfp4_mtp_native={} dflash={} moe_fold={} e8_shard={} e9_fold={} \
               gpu_recv={:?} reduce_fuse={:?} df2_round_shard={} df2_block={} df2_carry={} df2_prose_lane_greedy={}",
              c.world, c.shard_mixers, c.shard_mtp, c.graph, c.fp32_partials, c.trace, c.mtp,
              c.mtp_depth, c.batch_probe, c.step_probe, c.mode_serve, c.accept, c.capture,
              c.dspark, c.dspark_depth, c.exact_gemm, c.server_dspark, c.dspark_fp8_head,
              c.splitk_gemm, c.mxfp4, c.mxfp4_mtp_native, c.dflash, c.moe_fold, c.e8_shard, c.e9_fold,
              c.gpu_recv, c.reduce_fuse, c.df2_round_shard, c.df2_block, c.df2_carry, c.df2_prose_lane_greedy);
    // Phase-9 A.2: publish the boot context the status route reports (tp width, df2 block).
    crate::tel::set_boot(c.world as usize, c.df2_block as usize);
    if TP_CONFIG.set(c).is_err() {
        eprintln!("[tp] WARNING: TP config already installed — keeping the first");
    }
}

/// The installed TP config, if any (None on a plain single-box run — env/defaults apply there).
pub fn tp_config() -> Option<&'static TpConfig> { TP_CONFIG.get() }

// P4: a SEPARATE process-global store for the discovered topology. The head installs the real
// rank→IP list here (from `cluster::run_head`/`run_head_session`, AFTER discovery + rank
// assignment) before it calls `bring_up_head`; the node instead reads the topology it already
// received in the shipped `TpConfig` (`tp_config().topology`). Keeping it out of `TP_CONFIG` avoids
// re-ordering the existing early `set_tp_config` calls, which must not move.
static TP_TOPOLOGY: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();

/// Install the discovered rank→RoCE-IP topology (head only). Ignores the error and logs if it is
/// already set — a process brings up exactly one TP link.
pub fn set_topology(t: Vec<String>) {
    if let Err(_) = TP_TOPOLOGY.set(t) {
        eprintln!("[tp] WARNING: topology already installed — keeping the first");
    }
}

/// The installed discovered topology, if any.
pub fn topology() -> Option<&'static Vec<String>> { TP_TOPOLOGY.get() }

/// Resolve the N-way peer-IP list (indexed by rank) for `world > 2`. Precedence: the process-global
/// `TP_TOPOLOGY` (head), then the shipped `TpConfig::topology` (node), then fail loudly. A fabricated
/// placeholder is NOT acceptable here — it silently corrupts the doubling-partner map.
pub fn resolve_topology(world: i32) -> Result<Vec<IpAddr>> {
    let list: &Vec<String> = TP_TOPOLOGY
        .get()
        .or_else(|| tp_config().map(|c| &c.topology))
        .ok_or_else(|| anyhow::anyhow!(
            "world>2 requires a discovered topology (run via --head/--node cluster)"))?;
    if list.len() < world as usize {
        anyhow::bail!(
            "discovered topology has {} entries but world={world} (run via --head/--node cluster)",
            list.len());
    }
    let mut ips: Vec<IpAddr> = Vec::with_capacity(world as usize);
    for (i, s) in list.iter().take(world as usize).enumerate() {
        let ip: IpAddr = s.parse().with_context(|| format!("topology[{i}] is not an IP: '{s}'"))?;
        ips.push(ip);
    }
    Ok(ips)
}

pub fn rdma_dev() -> String {
    crate::opts::var(crate::opt!("rdma-dev")).ok()
        .and_then(|s| s.split(',').next().map(|x| x.trim().to_string()))
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| DEFAULT_RDMA_DEV.to_string())
}

pub struct TpContext {
    pub rank: i32,
    pub world: i32,
    pub link: TpLink,
}

impl TpContext {
    /// Recursive-doubling round count: `log2(world)`. A logical all-reduce at a model site consumes
    /// `rounds` consecutive device epochs (round = epoch % rounds; partner = rank ^ (1<<round)).
    pub fn rounds(&self) -> u32 { self.world.ilog2() }

    /// The partner rank for round `k` (recursive doubling).
    pub fn partner(&self, round: u32) -> i32 { self.rank ^ (1i32 << round) }

    /// Head side (rank 0): listen for the node's QP handshake on the RoCE device.
    pub fn bring_up_head(world: i32) -> Result<Self> {
        eprintln!("[tp] rank 0/{} — bringing up RDMA data-plane link on {} (listening) ...", world, rdma_dev());
        let link = if world == 2 {
            TpLink::connect(0, "", TP_PORT, &rdma_dev(), GID_IDX, TP_SLOT_BYTES)?
        } else {
            let ips = resolve_topology(world)?;
            TpLink::connect_nway(0, world, &ips, TP_PORT, &rdma_dev(), GID_IDX, TP_SLOT_BYTES)?
        };
        eprintln!("[tp] rank 0/{} — link UP", world);
        Ok(TpContext { rank: 0, world, link })
    }

    /// Node side: connect the QP to the head's RoCE IP (seen during the cluster sync).
    pub fn bring_up_node(head_ip: IpAddr, rank: i32, world: i32) -> Result<Self> {
        eprintln!("[tp] rank {}/{} — connecting RDMA data-plane link to head {head_ip} on {} ...", rank, world, rdma_dev());
        let link = if world == 2 {
            TpLink::connect(rank, &head_ip.to_string(), TP_PORT, &rdma_dev(), GID_IDX, TP_SLOT_BYTES)?
        } else {
            let ips = resolve_topology(world)?;
            TpLink::connect_nway(rank, world, &ips, TP_PORT, &rdma_dev(), GID_IDX, TP_SLOT_BYTES)?
        };
        eprintln!("[tp] rank {}/{} — link UP", rank, world);
        Ok(TpContext { rank, world, link })
    }

    /// Broadcast the prompt + decode budget from head (rank 0) to node (rank 1) over the live link,
    /// so the node runs the identical SPMD generate loop with zero manual config. Multi-frame (C3):
    /// the logical i32 stream `[total_n, max_new, tok…]` (total_n + 2 words) ships FRAME_I32 words
    /// per exchange; both sides derive the frame count from total_n (frame 0 carries it), so the
    /// number of symmetric exchange() calls matches by construction (lockstep-safe). The exchange is
    /// symmetric, so both fill and ship their whole slot; the head ignores what it reads back, the
    /// node reads the head's frames. A prompt that fits one frame stays byte-identical to the old
    /// single-frame format. Head calls with `Some((prompt, max_new))`; node with `None`.
    ///
    /// R2.3 prefix-cache: the head may pass `start_pos > 0` to signal a delta-prefill (the prompt
    /// is the NEW tokens only; the model state carries from the previous turn — no reset). The
    /// frame header is `[n, max_new, start_pos, tokens...]` (3 header words). `start_pos == 0`
    /// means full prefill (reset + forward at 0). Both ranks see the same start_pos and mirror.
    pub fn broadcast_prompt(&mut self, head: Option<(&[u32], usize, usize)>) -> Result<(Vec<u32>, usize, usize)> {
        let cap = TP_SLOT_BYTES / 4;                      // i32 slots per RDMA buffer
        const FRAME_I32: usize = TP_SLOT_BYTES / 4 - 3;   // stream words per frame (3 words headroom)
        let frames = |n: usize| (n + 3).div_ceil(FRAME_I32);
        // world==2 reads the gen-slotted recv ring; world>2 nodes read their dedicated head control slot.
        fn recv_frame<'a>(link: &'a TpLink, world: i32, n: usize) -> &'a [i32] {
            if world <= 2 { link.recv_host::<i32>(n) }
            else { link.ctrl_recv::<i32>(0, n) }
        }
        match head {
            Some((prompt, max_new, start_pos)) => {
                let word = |w: usize| -> i32 {
                    if w == 0 { prompt.len() as i32 }
                    else if w == 1 { max_new as i32 }
                    else if w == 2 { start_pos as i32 }
                    else { prompt[w - 3] as i32 }
                };
                for f in 0..frames(prompt.len()) {
                    let (lo, hi) = (f * FRAME_I32, ((f + 1) * FRAME_I32).min(prompt.len() + 3));
                    {
                        let slot = self.send_stage_mut::<i32>(cap);
                        for x in slot.iter_mut() { *x = 0; }
                        for (j, w) in (lo..hi).enumerate() { slot[j] = word(w); }
                    }
                    self.broadcast_frame(cap * 4)?;
                }
                Ok((prompt.to_vec(), max_new, start_pos))
            }
            None => {
                // Frame 0 carries [total_n, max_new, start_pos, first tokens…]; the frame count derives from total_n.
                { let slot = self.send_stage_mut::<i32>(cap); for x in slot.iter_mut() { *x = 0; } }
                self.broadcast_frame(cap * 4)?;
                let mut stream: Vec<i32> = Vec::new();
                let (n, max_new, start_pos) = {
                    let recv = recv_frame(&self.link, self.world, cap);
                    let n = recv[0] as usize;
                    anyhow::ensure!(n + 3 <= FRAME_I32 * 256, "bad prompt frame from head (n={n})");
                    stream.extend_from_slice(&recv[..(n + 3).min(FRAME_I32)]);
                    (n, recv[1] as usize, recv[2] as usize)
                };
                for _ in 1..frames(n) {
                    { let slot = self.send_stage_mut::<i32>(cap); for x in slot.iter_mut() { *x = 0; } }
                    self.broadcast_frame(cap * 4)?;
                    let recv = recv_frame(&self.link, self.world, cap);
                    let take = (n + 3 - stream.len()).min(FRAME_I32);
                    stream.extend_from_slice(&recv[..take]);
                }
                anyhow::ensure!(stream.len() == n + 3, "bad prompt stream from head ({} words for n={n})", stream.len());
                let prompt: Vec<u32> = stream[3..].iter().map(|&x| x as u32).collect();
                Ok((prompt, max_new, start_pos))
            }
        }
    }

    /// Consume the context, handing the link to a GpuModel for the sharded forward. Returns
    /// `(rank, world, link)` so the caller can call `GpuModel::attach_tp`.
    pub fn into_parts(self) -> (i32, i32, TpLink) { (self.rank, self.world, self.link) }

    /// P5 world>2 head-hub control broadcast: the head stages ONE frame into the send slot and fans it
    /// out to every node (each node does a single receive); the node stages its (zeroed) slot and does a
    /// single receive from the head. world==2 stays on the symmetric `exchange` fast path. The head's
    /// per-peer generation counters and the node's head counter stay in lockstep (one exchange per frame
    /// per pair), so the multi-frame prompt protocol is preserved exactly.
    fn broadcast_frame(&mut self, nbytes: usize) -> Result<()> {
        if self.world <= 2 {
            self.link.exchange(nbytes)
        } else if self.rank == 0 {
            for r in 1..self.world {
                self.link.exchange_one(r, nbytes)?;
            }
            Ok(())
        } else {
            self.link.exchange_one(0, nbytes)
        }
    }

    /// world>2 read of the control receive slot (sender `src`), for the just-completed exchange_one.
    fn ctrl_recv<T: Copy>(&self, src: i32, n: usize) -> &[T] {
        self.link.ctrl_recv::<T>(src, n)
    }

    /// The send staging buffer for the current world: world==2 uses the gen-slotted `send_host_mut`
    /// (byte-identical to pre-P5), world>2 uses the dedicated control SEND slot (`ctrl_send_mut`).
    fn send_stage_mut<T: Copy>(&mut self, n: usize) -> &mut [T] {
        if self.world <= 2 { self.link.send_host_mut::<T>(n) }
        else { self.link.ctrl_send_mut::<T>(n) }
    }

    /// Sanity: exchange a rank-stamped probe and confirm we received the peer's — proves the data-plane
    /// link is live end to end (discover → sync → RDMA), before any sharded forward runs. world==2 is the
    /// symmetric single exchange; world>2 the head fans its stamp out to every node and verifies each
    /// node's reply, while each node does one exchange against the head.
    pub fn sanity(&mut self) -> Result<()> {
        let stamp = 0xA0u8 + self.rank as u8;
        if self.world <= 2 {
            for b in self.link.send_host_mut::<u8>(16).iter_mut() { *b = stamp; }
            self.link.exchange(16)?;
            let peer = self.link.recv_host::<u8>(16)[0];
            let expect = 0xA0u8 + (1 - self.rank) as u8;
            if peer != expect { anyhow::bail!("tp sanity: peer stamp {peer:#x}, expected {expect:#x}"); }
            eprintln!("[tp] rank {}/{} — data-plane all-reduce link SANE (peer stamp {peer:#x})", self.rank, self.world);
            return Ok(());
        }
        // world > 2: head-hub. Each rank stages its own stamp; the head fans out and checks every node,
        // each node does one exchange and checks the head's stamp.
        for b in self.send_stage_mut::<u8>(16).iter_mut() { *b = stamp; }
        if self.rank == 0 {
            for r in 1..self.world {
                self.link.exchange_one(r, 16)?;
                let peer = self.ctrl_recv::<u8>(r, 16)[0];
                let expect = 0xA0u8 + r as u8;
                if peer != expect {
                    anyhow::bail!("tp sanity: rank {r} stamp {peer:#x}, expected {expect:#x}");
                }
            }
        } else {
            self.link.exchange_one(0, 16)?;
            let peer = self.ctrl_recv::<u8>(0, 16)[0];
            let expect = 0xA0u8;   // the head's rank stamp
            if peer != expect { anyhow::bail!("tp sanity: head stamp {peer:#x}, expected {expect:#x}"); }
        }
        eprintln!("[tp] rank {}/{} — data-plane all-reduce link SANE", self.rank, self.world);
        Ok(())
    }

    /// Prove both ranks resolved the SAME SPMD program before any kernel runs. The branch ladder
    /// (capture/accept/mtp/probes/generate) resolves from rank-local env + the shipped TpConfig —
    /// if those ever disagree (an env-only gate set on one rank), the two ranks would run different
    /// programs against one link: deterministically mismatched all-reduce epochs, i.e. silent
    /// garbage. One symmetric exchange of the selector makes that class fail LOUDLY at bring-up
    /// instead. The exchange is gen-sequenced and re-callable, and both ranks reach this point at
    /// the same program point, so the call count always matches.
    pub fn branch_check(&mut self, branch: &TpBranch) -> Result<()> {
        let s = branch.selector();
        if self.world <= 2 {
            for b in self.link.send_host_mut::<u8>(16).iter_mut() { *b = s; }
            self.link.exchange(16)?;
            let peer = self.link.recv_host::<u8>(16)[0];
            if peer != s {
                anyhow::bail!("TP BRANCH MISMATCH: this rank resolved selector {s:#04x}, the peer \
                               {peer:#04x} — the two ranks would run different programs against one \
                               link. The branch gates resolve env-first, then the shipped TpConfig, so \
                               the env differs between the boxes; set the knob on the head (it ships) \
                               or on both.");
            }
            eprintln!("[tp] rank {}/{} — branch selector {s:#04x} AGREED with peer", self.rank, self.world);
            return Ok(());
        }
        // world > 2: head-hub. The head verifies every node's selector equals its own; each node checks
        // the head's selector. All ranks must agree or the run refuses to start.
        for b in self.send_stage_mut::<u8>(16).iter_mut() { *b = s; }
        if self.rank == 0 {
            for r in 1..self.world {
                self.link.exchange_one(r, 16)?;
                let peer = self.ctrl_recv::<u8>(r, 16)[0];
                if peer != s {
                    anyhow::bail!("TP BRANCH MISMATCH: rank {r} resolved selector {peer:#04x}, head {s:#04x}");
                }
            }
        } else {
            self.link.exchange_one(0, 16)?;
            let peer = self.ctrl_recv::<u8>(0, 16)[0];
            if peer != s {
                anyhow::bail!("TP BRANCH MISMATCH: head resolved selector {peer:#04x}, this rank {s:#04x}");
            }
        }
        eprintln!("[tp] rank {}/{} — branch selector {s:#04x} AGREED", self.rank, self.world);
        Ok(())
    }
}

/// Which SPMD program both ranks run — the tp_serve branch ladder. Resolved in ONE place from
/// env-first, then the shipped TpConfig, then the no-env default; the selector exchange above
/// proves the resolution matched before any barrier fires.
pub enum TpBranch {
    Capture(String),
    Accept(usize),
    Mtp,
    StepProbe(usize),
    BatchProbe(usize),
    DecodeCtx(usize),
    BinvTp,
    StateTp,
    Generate,
    /// TP-A: the EXL3 cross-TP probe program (rung 3).
    Exl3Xtp,
}

impl TpBranch {
    /// One byte discriminating the program AND its parameters (two ranks at different accept depths
    /// or capture outputs would mismatch their barrier sequences just the same).
    fn selector(&self) -> u8 {
        match self {
            TpBranch::Generate => 0x00,
            TpBranch::Capture(p) => 0x10 ^ p.bytes().fold(0u8, |a, b| a.wrapping_mul(31).wrapping_add(b)),
            TpBranch::Accept(d) => 0x20 ^ (*d as u8),
            TpBranch::Mtp => 0x30,
            TpBranch::StepProbe(d) => 0x40 ^ (*d as u8),
            TpBranch::BatchProbe(n) => 0x50 ^ (*n as u8),
            TpBranch::DecodeCtx(c) => 0x60 ^ (*c as u8),
            TpBranch::BinvTp => 0x70,
            TpBranch::StateTp => 0x71,
            TpBranch::Exl3Xtp => 0x80,
        }
    }
}
