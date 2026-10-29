//! S-A3-e — EXL3 first-class serve backend (TP=1; S-A3-g owns TP).
//!
//! Consumes the SAME `BatchRequest` protocol server.rs speaks, so the entire HTTP surface
//! (/v1/chat/completions, /v1/completions, /v1/tokenize, /v1/models, /health) is 100% shared
//! with the NVFP4 serving path. The engine underneath is the S-A3-d EXL3 stack
//! (`FwdModel::forward_step`), with the width>1 kernel fixes (xq_hc_norm / xq_hc_inject) that
//! make DISTINCT concurrent lanes correct.
//!
//! Posture: MTP chain-verify for greedy AND sampled lanes (API-parity G2: sampled lanes sample
//! every verify row on device and accept while sample == draft — distribution-exact with the
//! argmax draft; greedy stays bitwise lossless). The §6 profitability control is PER REQUEST
//! (API-parity G1): a request whose drafts do not pay runs plain; the next request speculates.

use crate::batch::{BatchRequest, TokEvent};
use crate::exl3_forward::{FwdModel, Scratch};
use crate::server::{create_router, AppState};
use crate::tokenizer::{QwenTokenizer, ThinkingMode};
use anyhow::{Context as _, Result};
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedReceiver;

/// S1 (REL_V0_7_3): delegates to the ONE shared `=`-aware parser (`crate::arg_value`) so
/// `--lane-order=fcfs` and `--lane-order fcfs` behave identically on EXL3 (the old body
/// matched only the space form and silently ignored the `=` form).
fn arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    crate::arg_value(args, name)
}

// ===========================================================================================
// DFlash2 (DF2) speculation on the EXL3 serve path (ISSUES[94]).
//
// Until now the EXL3 server ACCEPTED --spec-source dflash2* / --draft-dir / --df2-block /
// --df2-round-shard and silently ignored them: the flags passed the shared validators in
// main.rs (run_server -> resolve_draft_dir / resolve_spec_source) and then fell on the floor,
// because build_serve only ever built the pack-MTP chain verifier. This section parses those
// flags FOR REAL, gates the engagement (source x artifact x pack dims x ctx x TP=1), loads
// the DF2 round through the forward-side contract (FwdModel::df2_* / dflash2::round), primes
// the ring at every prefill chunk, and dispatches the lone-lane speculative round to the DF2
// chain. No env vars anywhere: every knob is one of the four CLI flags above.
// ===========================================================================================

/// The pack dimensions the DF2 artifact was trained against (trunk df2_round_compatible):
/// hidden 5120 (the Qwen3.8-27B backbone the S4F artifact was trained on), vocab 248320 (the
/// EXL3 pack's head), more backbone layers than the artifact's deepest tap. A pack that
/// disagrees cannot use the artifact: the ring reads the backbone's hidden stream and the
/// selector's codebooks at these widths. (Doc bug fixed 2026-10-23: this trunk is the 27B,
/// NOT Flash-Next — Flash-Next is hidden 2560 / 48 layers / taps [5,15,25,35,45].)
/// DFLASH2-GEN: the DF2 round dims are LIVE geometry (installed from the draft config.json);
/// these consts remain only as the 27B default the geometry falls back to.
pub(crate) fn df2_want_hidden() -> usize { crate::dflash2::hidden() }
pub(crate) fn df2_want_vocab() -> usize { crate::dflash2::vocab() }
pub(crate) fn df2_min_layers() -> usize { *crate::dflash2::tap_layers().last().unwrap() }

/// EXL3 mirror of main.rs `resolve_spec_source` -- the SAME parse, the SAME default
/// (`dflash2-auto`, S8F), and the SAME error text, so a typo dies identically whichever
/// backend the pack routes to. (main.rs already validates the flag before the exl3 dispatch;
/// this mirror keeps the lib self-contained and the two checks can never disagree.)
pub fn resolve_spec_source_exl3(args: &[String]) -> crate::batch::SpecSource {
    use crate::batch::SpecSource;
    match arg(args, "--spec-source").map(str::to_lowercase) {
        None => SpecSource::DFlash2Auto,
        Some(s) => match SpecSource::from_cli(&s) {
            Some(src) => src,
            None => {
                eprintln!("--spec-source must be mtp|dspark|dflash2|dflash2-rq|dflash2-auto|dflash2-tree|none (got {s:?})");
                std::process::exit(1);
            }
        },
    }
}

/// The boot-time load of the DF2 round, mirroring the trunk `load_df2_round_dir` (main.rs)
/// for the single-process EXL3 path: borrow the served model's dequantized bf16 head + embed
/// (`df2_boot_ptrs`), pin-load the artifact against the SAME sha policy the trunk uses
/// (`--sha256` absent = the built-in published pin), attach the tap sink, and hand both sinks
/// to the model so its forward taps hiddens for the draft ring (prime sink cols =
/// `batch::PREFILL_CHUNK`, the window `df2_prime_window` consumes per prefill chunk).
/// `ar` is None: this path is TP=1, the round is never sharded.
fn df2_load_round(model: &mut Arc<FwdModel>, dir: &str, max_c: usize, sha_pin: Option<&str>)
    -> Result<crate::dflash2::round::Df2Round> {
    use crate::dflash2::capture::{Df2PrimeSink, Df2TapSink};
    use crate::dflash2::round::{BorrowedW, Df2Round};
    let (embed_ptr, head_ptr) = model.df2_boot_ptrs()?;
    // S-EXL3-DF2Q (2026-10-18): two-stage quantized head by default — the DSV4 serving default.
    // The plain Bf16 borrow costs ~126 ms/round through gemm_binv_b (measured, 1.5bpw trunk);
    // coarse-NVFP4 (0.63 GB) + exact bf16 re-rank of a top-256 shortlist costs ~3 ms and keeps
    // EXACT ranking. Escape hatch: --no-df2-q2stage (plain bf16, the old behavior).
    let head = if crate::opts::var(crate::opt!("no-df2-q2stage")).is_err() {
        match model.df2_q_head_ptrs() {
            Some(q) => BorrowedW::TwoStage { q, bf16_ptr: head_ptr },
            None => BorrowedW::Bf16 { ptr: head_ptr },
        }
    } else {
        BorrowedW::Bf16 { ptr: head_ptr }
    };
    let embed = BorrowedW::Bf16 { ptr: embed_ptr };
    let mut round = Df2Round::load_tp_pinned(dir, Some(head), Some(embed), max_c, None, sha_pin)?;
    let tap = std::sync::Arc::new(Df2TapSink::new(model.dev()));
    round.attach_sink(&tap);
    let prime = std::sync::Arc::new(Df2PrimeSink::new(model.dev(), crate::batch::PREFILL_CHUNK));
    let m = Arc::get_mut(model)
        .ok_or_else(|| anyhow::anyhow!("the model was already shared before the DF2 capture attach"))?;
    m.set_df2_capture(Some(tap), Some(prime));
    Ok(round)
}

fn parse_df2_block(v: Option<&str>) -> Result<usize> {
    match v {
        None => Ok(crate::dflash2::block()),
        Some(s) => match s.parse::<usize>() {
            Ok(n) if n >= 2 => Ok(n),
            _ => anyhow::bail!("--df2-block must be an integer >= 2 (got '{s}')"),
        },
    }
}

/// `--df2-round-shard on|off` -- the TP>2 round-shard frame. Parsed and REPORTED even though
/// the EXL3 serve path is TP=1 (the flag never shards here): the engaged boot line carries
/// its value, so an operator asking for it sees the truth instead of silence.
fn parse_df2_round_shard(v: Option<&str>) -> Result<bool> {
    match v {
        None | Some("off") => Ok(false),
        Some("on") => Ok(true),
        Some(other) => anyhow::bail!("--df2-round-shard must be on or off (got '{other}')"),
    }
}

/// The pure half of the boot gate: given the resolved source, the resolved --draft-dir, the
/// loaded model's dims, the resolved --max-seq-len and the TP world, return None (ENGAGE) or
/// Some(reason) (FALLBACK to pack MTP with exactly this reason in the boot line). Pure so the
/// reason table is unit-testable on the host (see `df2_gate_reason_table` below); the load
/// failure reason is produced by the caller (it needs the GPU).
///
/// Reasons (the boot line prints them verbatim inside the FALLBACK parens):
///   `source`                          -- the resolved spec source is not a DFlash2 source
///   `no --draft-dir`                  -- no artifact dir given (dflash2-auto default boots here)
///   `dims hidden=<h>x<want_h> vocab=<v>x<want_v> layers=<l> (need ><taps>)` -- the pack is
///   not the shape the DRAFT's config.json describes (27B defaults when no draft ships)
///   `ctx <got>><cap>`                 -- --max-seq-len beyond the DF2 ring's safe ceiling
///   (RETIRED 2026-10-21, EXL27D item 2) `tp=<n>>1` -- the TP-DF2 leg now serves DF2 at TP>1
pub(crate) fn df2_gate_reason(src: crate::batch::SpecSource, draft_dir: Option<&str>,
                              hidden: usize, vocab: usize, layers: usize,
                              max_c: usize, tp_world: usize) -> Option<String> {
    if !crate::batch::is_df2_src(src) { return Some("source".into()); }
    let Some(dir) = draft_dir else { return Some("no --draft-dir".into()); };
    // (a dir that does not exist / does not load is NOT this gate's business: it engages here
    //  and the load produces the `load failed: <e>` FALLBACK reason — the brief's table.)
    // DFLASH2-GEN: the want-dims come from the draft's OWN config.json when it ships one —
    // installed here so an Ornith-geometry draft is not refused against 27B defaults. A
    // missing/unreadable config keeps the 27B defaults and the load fails loudly later.
    // (FLASHNEXT-DF2-B: a dir whose config.json is not readable on THIS box leaves the 27B
    //  default geometry in place — the caller prints a second line naming that fact next to
    //  the dims reason; the reason table itself stays pure for the tests below.)
    let _ = crate::dflash2::geom::install_geom_from_dir(dir);
    let (want_h, want_v, need_layers) = (df2_want_hidden(), df2_want_vocab(), df2_min_layers());
    if hidden != want_h || vocab != want_v || layers <= need_layers {
        return Some(format!(
            "dims hidden={hidden}x{want_h} vocab={vocab}x{want_v} layers={layers} (need >{need_layers})"));
    }
    if max_c > crate::dflash2::max_ctx_safe() {
        return Some(format!("ctx {max_c}>{}", crate::dflash2::max_ctx_safe()));
    }
    // EXL27D item 2 (2026-10-21): the `tp={n}>1` reason is RETIRED. The TP-DF2 leg ships
    // the artifact and the head's argv to the node (TpConfig v12 `df2_draft_dir` /
    // `exl3_mode`), both ranks run the identical scheduler (tp_lockstep agree covers the
    // DF2 load outcome in the boot hash and the per-round accept/emitted words), so the
    // chain is no longer a single-process contract. `tp_world` stays a parameter (the
    // signature is pinned by the reason-table tests below) but no longer gates.
    let _ = tp_world;
    None
}

/// FLASHNEXT-DF2-C (ISSUES 116): the draft dir this rank loads. Under an EXL3 TP serve (world > 1)
/// the SHIPPED `TpConfig` decides: on the head it is the absolute `--draft-dir` the head put there
/// before its sync, on a node it is the blob-cache dir the sync assembled (`run_cluster_node_once`
/// rewrites it) or empty when the head shipped no drafter — so a node never reads the head's
/// filesystem path and a missing local copy can no longer turn into a silent 27B-default gate.
/// A single-box boot reads `--draft-dir` as before.
pub fn df2_draft_dir_for_boot(args: &[String], cfg: Option<&crate::tp::TpConfig>) -> Option<String> {
    match cfg {
        Some(c) if c.mode_serve && c.exl3 && c.world > 1 => {
            if c.df2_draft_dir.is_empty() { None } else { Some(c.df2_draft_dir.clone()) }
        }
        _ => arg(args, "--draft-dir").map(str::to_string),
    }
}

/// FLASHNEXT-DF2-C (ISSUES 116): the draft dir the EXL3 TP HEAD ships to its nodes — the
/// absolute `--draft-dir` when the resolved spec source is a DFlash2-class source (the sync's
/// `DraftManifest` reads `spec_source` + `df2_draft_dir`, exactly the nvfp4 head's contract);
/// None otherwise (no `--draft-dir`, a non-DF2 source, or a path that cannot be canonicalized —
/// the head's own gate then says why).
pub fn tp_head_draft_dir(args: &[String]) -> Option<String> {
    let src = resolve_spec_source_exl3(args);
    if !crate::batch::is_df2_src(src) { return None; }
    let dir = arg(args, "--draft-dir")?;
    std::fs::canonicalize(dir).ok().map(|p| p.to_string_lossy().to_string())
}

/// The engaged boot line, exactly as the owner's brief specifies it:
/// `[exl3-serve] df2: ENGAGED draft=<dir> sha256=<hex> block=<n> round-shard=<on|off> (DFlash2 chain on EXL3, TP=<n>)`
pub(crate) fn df2_boot_line_engaged(dir: &str, sha: &str, block: usize, shard: bool, tp: usize) -> String {
    // EXL27D item 2: the world size is in the line — the receipt must say which topology
    // the chain serves on (TP=1 stayed the literal before; a TP=2 boot printed "TP=1").
    // FLASHNEXT-DF2 Phase 4: the artifact sha the loader VERIFIED is in the line (the
    // receipt names which artifact drafted; `unverified` = a path that skipped the hash).
    format!("[exl3-serve] df2: ENGAGED draft={dir} sha256={sha} block={block} round-shard={} (DFlash2 chain on EXL3, TP={}; chain verify k<={} = the pack MTP wide-verify width)",
            if shard { "on" } else { "off" }, tp, crate::exl3_forward::MTP_MAX_K)
}

/// The fallback boot line: `[exl3-serve] df2: FALLBACK to pack MTP (<reason>)`.
pub(crate) fn df2_boot_line_fallback(reason: &str) -> String {
    format!("[exl3-serve] df2: FALLBACK to pack MTP ({reason})")
}

// ---- DF2 counters (the /metrics series `exl3_df2_*`) ---------------------------------------
// Promoted to Prometheus text by `df2_metrics_append` below. NOTE (metrics surface): the
// /metrics route renders `crate::metrics::render()` and metrics.rs is outside this change's
// file set, so the one-line hook that appends these series --
//     crate::exl3_serve::df2_metrics_append(&mut o);
// as the last statement of `metrics::render` before its final `render_toolcall(&mut o)` --
// lands with the merge of this branch (owner-visible deviation; the counters themselves are
// live NOW and the boot/once lines name the speculator, AGENTS SS4 "identical hashes is not
// a liveness receipt"). The DF2 round/accept counts ALSO ride the existing generic series
// (velogb10_spec_rounds_total & co.) because DF2 rounds feed the same per-request stats the
// pack-MTP path reports (`Lane::log_stats` -> `crate::tel::add_spec`).
static DF2_ENGAGED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DF2_ROUNDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DF2_ACCEPTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DF2_FALLBACK_MTP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// FLASHNEXT-DF2-117: sampled DF2 rounds whose draft rode the control plane (head sends / node adopts).
static DF2_SHIPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DF2_REANCHOR: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DF2_DDS_BYPASS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Mark the boot engaged (gauge 0/1; 1 exactly once per engaged boot).
pub(crate) fn df2_mark_engaged() {
    DF2_ENGAGED.store(1, std::sync::atomic::Ordering::Relaxed);
}

/// Count one served DF2 round and its accepted tokens. The FIRST round also logs the
/// AGENTS §4 once line — which speculator actually served a round (an engaged boot whose
/// rounds all bypass is visible here, not inferred from hashes).
pub(crate) fn df2_count_round(accepted: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    if DF2_ROUNDS.fetch_add(1, Relaxed) == 0 {
        println!("[exl3-serve] df2: first round served by the DFlash2 chain (speculator=dflash2)");
    }
    DF2_ACCEPTED.fetch_add(accepted as u64, Relaxed);
}

/// Count one round that fell back to the pack-MTP chain although DF2 was engaged and the lane
/// ran alone (the DF2 ring is stale for this lane -- see step_mtp's dispatch note).
pub(crate) fn df2_count_fallback_mtp() {
    DF2_FALLBACK_MTP.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}
pub(crate) fn df2_count_reanchor() {
    DF2_REANCHOR.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Count one lone-lane round the DDS/adaptive branch kept (DF2 has no `keep` closure in the
/// forward contract, so adaptive drafting is not trivially routable to it).
pub(crate) fn df2_count_dds_bypass() {
    DF2_DDS_BYPASS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Append the `exl3_df2_*` series to a Prometheus text body (see the metrics-surface note
/// above for the render hook).
pub fn df2_metrics_append(o: &mut String) {
    use std::sync::atomic::Ordering::Relaxed;
    o.push_str("# HELP exl3_df2_engaged 1 if the EXL3 server booted with the DFlash2 chain engaged (0 = pack MTP).\n# TYPE exl3_df2_engaged gauge\n");
    o.push_str(&format!("exl3_df2_engaged {}\n", DF2_ENGAGED.load(Relaxed)));
    o.push_str("# HELP exl3_df2_rounds_total Lone-lane speculative rounds served by the DFlash2 chain.\n# TYPE exl3_df2_rounds_total counter\n");
    o.push_str(&format!("exl3_df2_rounds_total {}\n", DF2_ROUNDS.load(Relaxed)));
    o.push_str("# HELP exl3_df2_tokens_accepted_total Draft tokens the EXL3 verify accepted from the DFlash2 chain.\n# TYPE exl3_df2_tokens_accepted_total counter\n");
    o.push_str(&format!("exl3_df2_tokens_accepted_total {}\n", DF2_ACCEPTED.load(Relaxed)));
    o.push_str("# HELP exl3_df2_shipped_drafts_total Sampled DF2 rounds whose draft was head-authoritative under TP (head: frames sent per node; node: frames adopted) — ISSUES 117.\n# TYPE exl3_df2_shipped_drafts_total counter\n");
    o.push_str(&format!("exl3_df2_shipped_drafts_total {}\n", DF2_SHIPPED.load(Relaxed)));
    o.push_str("# HELP exl3_df2_fallback_mtp_total Rounds that used the pack-MTP chain although DF2 was engaged (stale DF2 ring for the lane).\n# TYPE exl3_df2_fallback_mtp_total counter\n");
    o.push_str(&format!("exl3_df2_fallback_mtp_total {}\n", DF2_FALLBACK_MTP.load(Relaxed)));
    o.push_str("# HELP exl3_df2_reanchor_total Solo fallback rounds that rewound the DF2 ring to a lane behind the frontier (prefix-cached follow-up request).\n# TYPE exl3_df2_reanchor_total counter\n");
    o.push_str(&format!("exl3_df2_reanchor_total {}\n", DF2_REANCHOR.load(Relaxed)));
    o.push_str("# HELP exl3_df2_dds_bypass_total Lone-lane rounds kept on the DDS adaptive chain. Structural 0 since Fix1 (2026-10-16): an armed, ring-fresh DF2 chain preempts the adaptive branch; kept for continuity.\n# TYPE exl3_df2_dds_bypass_total counter\n");
    o.push_str(&format!("exl3_df2_dds_bypass_total {}\n", DF2_DDS_BYPASS.load(Relaxed)));
}

struct Lane {
    pos: usize,          // next trunk position to write
    last_tok: u32,       // last committed token
    generated: usize,
    max_new: usize,
    min_new: usize,
    ignore_eos: bool,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    min_p: f32,
    // WP15: this request's penalties (None = unpenalized: no penalty kernel runs for it).
    pen: Option<crate::exl3_forward::PenParams>,
    // WP08: the reference implementation's streaming loop detector (None = --loop-detect off) + log context.
    loop_det: Option<crate::loop_detect::LoopDetector>,
    prompt_hash: u64,
    history: Vec<u32>,
    // API-parity G2: sampler stream (uniform = hash(seed, ctr); ctr advances per sampled row).
    seed: u64,
    ctr: u32,
    // WP24: the real-q draft RNG counter (draft pass i of a round draws dctr + i; a separate
    // stream from the verify rows' `ctr`). Advances by the served depth per real-q round.
    dctr: u32,
    // WP24 gate (--probe-spec-sampling): true until the request's first MTP step (its first
    // round's d_0 feeds the gate's negative control). Bookkeeping only.
    gate_first: bool,
    // API-parity G1: this request's MTP round-time and tokens-per-round EMAs + rounds run.
    // cost = ema_ms / ema_tok (ratio of means). NOT an EMA of per-round ms/tok: that mean
    // of ratios over-weights accept-0 rounds (Jensen) and read ~34 ms/tok for rounds that
    // really cost ~22, turning MTP off on every request (2026-09-25, owner: "SUPER SLOW").
    ema_ms: f64,
    ema_tok: f64,
    // S-B9-REL-MTPREPROBE2 W1.1: rounds counted since the last estimator reset
    // (request start / re-arm) — the re-probe path blends a RUNNING MEAN over the
    // first 64 of them (see mtp_reprobe::blend_cost_ema), never a one-round seed.
    ema_n: usize,
    /// P18 SPEC8K: measured round-to-round WALL latency of this lane (ms): the end of one round to the
    /// end of the next, so it contains the wait for the rounds of the other busy lanes, which
    /// ema_ms (the time of one round, or for the lane of a packed round one SHARE of it) does not.
    /// Read only by the Auto arm of use_shared_plain, whose job is to price one arm against the other:
    /// both sides must be the aggregate that the n clients see. 0.0 until the second round of the lane.
    ema_wall_ms: f64,
    /// P18 SPEC8K: the instant round_account finished the previous round of this lane, the anchor for
    /// ema_wall_ms.
    last_wall: Option<std::time::Instant>,
    mtp_rounds: usize,
    wp23: Wp23Lane, // WP23: DDS cost-guard fit + per-width round counts (the [dds] line)
    // per-request round anatomy, logged at finish ([mtp-stats]): comparable to the reference implementation's
    // draft_accepted/draft_rejected counters (tokens = accepted + rounds).
    st_rounds: usize,
    st_drafted: usize,
    st_accepted: usize,
    st_ms: f64,
    st_plain: usize,
    st_ctrl: usize, // WP03: control rounds this request ran (the cadence gate reads it)
    // S-B9-CF-PACK4S: which arm ran each of this request's rounds (`[mtp-stats] packed=`, and
    // `arms=` with --round-trace): the provenance of every emitted token.
    arms: crate::lane_pack::ArmTrace,
    // the request's first generated token ids (seam token first; at most FIRST_TOKS), printed
    // with --round-trace as `toks=`: the distribution gate's statistic, read with `arms=`
    first_toks: Vec<u32>,
    // S-A3-f-d: chain-verify state. tap_row = taps_keep row holding the tap that
    // predicted last_tok (usize::MAX after a control round: resid holds it).
    mtp: Option<usize>,
    // S-B9-REL-MTPREPROBE: re-probe state for the §6 per-request MTP auto-off
    // (src/mtp_reprobe.rs). mtp=None + phase Off = speculation off but the draft head
    // stays primed and a re-probe comes after a backoff; force_control = a pending
    // confirm wants ONE fresh control round at the current context.
    reprobe: crate::mtp_reprobe::ReProbe,
    force_control: bool,
    // CF-FCFS: this request's admission order (the scheduler's monotonically increasing admit
    // counter at admit time). fcfs lane order is the ONLY thing that reads it. It is a pure
    // function of replicated state — every TP rank runs the same admits in the same head-driven
    // order, so every rank assigns the same value to the same request (AGENTS 2.10).
    admit_seq: u64,
    // CF-FCHS: tokens this lane has generated since it last became the fcfs front lane. Reset when
    // a lane rotates to the back; compared against --lane-quantum to bound one turn's monopoly.
    fcfs_credit: usize,
    // S-B9-CF-PACK4: an image request (its slot carries a 3-axis RoPE map) — never packed in v1.
    img: bool,
    // W2-EXL3: the request's compiled JSON-schema FSM (None = unconstrained), its current
    // state, and whether the constraint is live yet (false while the lane is still inside the
    // think block its prompt opened: the mask arms on the think-close token). A schema lane
    // never speculates (mtp = None for its whole life), so every token it emits comes from a
    // masked plain step.
    schema: Option<std::sync::Arc<crate::json_schema::SchemaMask>>,
    schema_state: u32,
    schema_armed: bool,
    tx: crate::server::TokTx,
}

/// W2-EXL3: what one emitted token does to a schema lane.
#[derive(Debug, PartialEq, Eq)]
enum SchemaStep {
    /// keep decoding (constraint not armed yet, or the document is still open)
    Continue,
    /// the JSON document is complete: the turn ends here ("stop")
    Complete,
    /// the token is not valid in the machine's state, or leaves no valid continuation: the
    /// turn ends with an error finish (`SCHEMA_DEAD_END`) and the event is logged
    Invalid,
}

/// The finish reason of a schema turn that reached a state no token can continue (or emitted a
/// token the state refuses): an error, so the client never reads the partial document as a
/// completed one.
const SCHEMA_DEAD_END: &str = "error: json-schema constrained decoding reached a state with no valid continuation";

/// W2-EXL3: advance a schema lane's FSM by one emitted token (pure; unit-tested).
fn schema_feed(sm: &crate::json_schema::SchemaMask, state: &mut u32, armed: &mut bool, tok: u32) -> SchemaStep {
    if !*armed {
        if sm.defer_until() == Some(tok) { *armed = true; }
        return SchemaStep::Continue;
    }
    match sm.step(*state, tok) {
        None => SchemaStep::Invalid,
        Some(ns) => {
            *state = ns;
            if sm.is_complete(ns) {
                SchemaStep::Complete
            } else if sm.allowed(ns).iter().all(|&w| w == 0) {
                // a state no token can continue (the machine should never reach one): end the
                // turn instead of letting a fully masked row pick id 0
                SchemaStep::Invalid
            } else {
                SchemaStep::Continue
            }
        }
    }
}

/// S-B9-CF-PACK4: host-side anatomy of the packed rounds since boot: wall ms of the three stages
/// (the head re-prime is enqueued, so its GPU time lands in the next round's first draft pass,
/// exactly as the lone round's device re-prime does). The sums are CUMULATIVE; the idle line
/// prints their per-round means, and only when a round ran since the previous line.
#[derive(Default, Clone, Copy)]
struct PackStats {
    rounds: u64,
    rows: u64,
    toks: u64,
    ms_draft: f64,
    ms_verify: f64,
    ms_finish: f64,
    // the verify's own parts (FwdModel::pack_verify): staging, graph replay + sync, readback
    ms_stage: f64,
    ms_graph: f64,
    ms_read: f64,
    // per packed width R (index = rows, 0..=16): rounds, graph-and-sync ms, tokens emitted — the served cost curve
    by_r: [(u64, f64, u64); 17],
    printed: u64,
    // S-B9-CF-PACK4S: packed rounds that replayed the real-q family, and lane rounds of sampled
    // lanes (real-q, match rule) and of greedy lanes served by packed verifies
    rq_rounds: u64,
    lanes_rq: u64,
    lanes_match: u64,
    lanes_greedy: u64,
    // S-B9-CF-PACK5: host ms of the packed round's two lockstep points (TP only): the pre-verify
    // (wasted-pass drain + exchange + agree) and the post-verify tp_round (fence A + exchange + fence B)
    ms_pre: f64,
    ms_post: f64,
    // PACK4-REVIEW A3: true from the moment a pair round starts its packed part (after the
    // control-cadence fallback) until it completes; an error that leaves it set came from the
    // packed round, and the arm is disabled for the rest of the boot (`pack_disable`)
    in_round: bool,
}

/// S-B9-REL-INTEG: the boot warning of lane packing (on by default since F10) under non-default penalty defaults (None = the defaults are
/// the no-op 1.0 / 0 / 0, so an unpenalized request packs). A request that sends no penalty field inherits the
/// server default, so with `--default-presence-penalty 1.5` EVERY such request is penalized and none of them packs
/// (`Skip::Penalty`); a request that sends `presence_penalty: 0` (and the other two neutral) overrides the default
/// and does pack. This is the owner's posture: his hand test showed 0 packed rounds for exactly this reason.
fn lane_pack_penalty_warning(rep: f32, pres: f32, freq: f32) -> Option<String> {
    if crate::exl3_forward::PenParams::new(rep, pres, freq, 1, 0).is_none() { return None; }
    Some(format!("[exl3-serve] lane-pack WARNING: this server's penalty defaults are repetition {rep}, presence {pres}, frequency {freq} \
(--default-repetition-penalty / --default-presence-penalty / --default-frequency-penalty). A request that sends no penalty \
field inherits them and is a PENALIZED request, and a penalized request never joins a packed verify (it runs its own \
lone round: /metrics velogb10_lane_pack_skip_total{{reason=\"penalty\"}}). With these defaults lane packing (--lane-pack, on by default) changes \
nothing for such requests; a request that sends presence_penalty 0, frequency_penalty 0 and repetition_penalty 1 itself \
does pack. The packed arm for penalized requests is not built"))
}

/// May SAMPLED lanes join a packed verify under TP? `--lane-pack-sampled-tp on|off`, default ON
/// (owner 2026-10-10: real requests are sampled, and a feature is not kept away from them because
/// its gates have not run: his TP=4 hand test showed 0 packed rounds and 898 `skip sampled`).
/// The sampled tail of a packed verify under TP (the match-rule and real-q families keyed by their
/// vocabulary-parallel tail) was built by S-B9-CF-PACK5 (`lane_pack::fams`, `FwdModel::pack_fam_key`)
/// and has NOT run its hardware gates. `off` = a sampled lane runs its own lone round under TP
/// (counted and logged as `skip sampled`). Spmd scope: the head ships the value, and the boot agree
/// carries `pack_sampled`, so ranks cannot differ.
fn pack_sampled_under_tp() -> bool {
    crate::opts::choice(crate::opt!("lane-pack-sampled-tp")) != "off"
}

struct Exl3Scheduler {
    model: Arc<FwdModel>,
    sc: Scratch,
    width: usize,
    lanes: Vec<Option<Lane>>,
    eos: Vec<u32>,
    chunk: usize, // S-A3-f-b: prefill chunk width (tokens per forward sweep)
    /// ISSUES[94] DF2: the loaded DFlash2 round (artifact + ring), present iff the boot gate
    /// ENGAGED it. One shared round per scheduler: the DF2 chain is a single token stream, so
    /// only a lane whose position matches the ring's `nprev` may run a DF2 round (the lone-lane
    /// case by construction; every other lane/step runs the pack-MTP paths unchanged).
    df2: Option<crate::dflash2::round::Df2Round>,
    /// The chain-verify depth the DF2 round uses: k = --df2-block - 1 (default
    /// `dflash2::block() - 1`).
    df2_k: usize,
    /// `--prefill-interleave N` (public PR #23): scheduler steps run for the lanes already decoding
    /// after each non-final prefill chunk of a joining prompt. 0 (default) = none: the whole prompt
    /// prefills before the next step, exactly as before. Always 0 under TP (build_serve).
    interleave: usize,
    psc: Option<crate::exl3_forward::PrefillScratch>,
    // S-A3-f-d MTP policy: depth k (--exl3-mtp-k, default MTP_MAX_K = 7 since WP23), auto-disable
    // when the measured per-token cost loses to plain (§6), control-round bookkeeping.
    mtp_k: usize,
    /// CF-P1e `--spec-lanes-max`: when the busy lanes share one plain batched step instead of serial speculation.
    spec_policy: SpecLanes,
    /// CF-FCFS `--lane-order` / `--lane-quantum`: the serial rounds' order, the fcfs quantum in
    /// generated tokens, the next admission's sequence number, and the first-16 rotation log count.
    lane_order: LaneOrder,
    lane_quantum: usize,
    admit_ctr: u64,
    fcfs_rots: usize,
    /// LANERES1 `--lane-reserve`: the lane-reservation policy. None = off = today's admission,
    /// byte-for-byte (the run loop and the pick keep their original shapes).
    reserve: Option<crate::lane_reserve::Cfg>,
    /// The policy's forced-take limiter state (one per scheduler).
    reserve_st: crate::lane_reserve::State,
    /// LANERES2 `--kv-spill`: the SSD spill tier for evicted lane snapshots. None = off =
    /// today's behavior, byte-for-byte (no capture, no restore, no counters).
    spill: Option<crate::kv_spill::Store>,
    /// The spill policy (cap, LRU, cost model); a zero-cap policy when the tier is off.
    spill_pol: crate::kv_spill::Policy,
    /// This scheduler's rank (0 on a single box): spill files are per-rank.
    spill_rank: usize,
    /// Per-slot release stamps (released_at, normal finish) for the policy. Stamped only while
    /// the policy is on; read to build SlotViews and the forced-take log line.
    rel: Vec<Option<(std::time::Instant, bool)>>,
    /// S-B9-CF-PACK4 `--lane-pack` (see step_packed): the mode, the packed verify's own scratch
    /// (Some only with the flag on), which regime families are captured [dense, sparse], the
    /// rotation counter, the skip-line budget, "a prefill is in flight" and the idle-line stats.
    lane_pack: crate::lane_pack::LanePack,
    scp: Option<Scratch>,
    pack_ready: [bool; 2],
    /// S-B9-CF-PACK5: sampled lanes may pack on this boot (TP=1: yes; under TP: `--lane-pack-sampled-tp`,
    /// see `pack_sampled_under_tp`). A function of the topology and a shipped option: the same on every rank.
    pack_sampled: bool,
    pack_rot: u64,
    pack_logs: usize,
    in_prefill: bool,
    pack_stats: PackStats,
    /// CF-P1e auto policy: learned scale of the shared-step time model, the last decision (hysteresis) and the
    /// consecutive shared-step count (stale-estimate refresh). All updated from lockstep-adopted values.
    shared_scale: f64,
    shared_last: bool,
    shared_streak: usize,
    /// F4 (PACK5G finding 1): rounds riding the packed/serial arm at >=4 busy lanes since the
    /// last shared plain step, under `--lane-pack on` + Auto (the shared-cost probe; see step()).
    shared_probe_ctr: u32,
    // S-C4XOVER: measured delivered rate per arm (tokens / ms of the rounds that ran),
    // the honest basis of the Auto pick: both modelled inputs were biased at four busy
    // lanes (the serial cycle estimate ~+30% optimistic vs delivery, the shared prior's
    // learned scale ~-18% pessimistic), which pinned the policy to the slower arm.
    spec_win_tok: [f64; 2],        // window accumulators: tokens the arm's rounds delivered
    spec_win_ms: [f64; 2],         // ... the ms those rounds cost
    spec_win_n: [u8; 2],           // ... the busiest lane count seen in the window
    spec_win_acc: [u32; 2],        // ... rounds accumulated since the last fold
    spec_arm_rate: [f64; 2],       // EWMA of delivered aggregate tok/ms (0 = never measured)
    spec_arm_n: [u8; 2],           // busy lanes the rate was measured at (last fold's tag)
    spec_arm_fold: [u64; 2],       // scheduler step of the last fold (freshness)
    spec_est_pick: Option<bool>,   // S-C4XOVER v3: the estimate path's last pick (None before one)
    spec_probe: Option<(u8, u32)>, // arm-rate probe window in flight: (arm, steps left)
    spec_probe_ctr: u32,           // rounds since the last probe of the idle arm
    spec_margin: f64,           // S-C4XOVER v2: last MEASURED arm margin (max/min - 1; 0 = none)
    spec_margin_n: u8,          // ...and the busy count it was measured at (eager again on change)
    policy_logs: usize,
    ctrl_round: usize,
    /// S-B9-REL-MTPREPROBE: the re-probe schedule (shrunken by --mtp-reprobe-drill) and the
    /// master switch (--mtp-reprobe=off = the previous one-way behaviour, exactly).
    reprobe_cfg: crate::mtp_reprobe::ReProbeCfg,
    reprobe_on: bool,
    // API-parity G1: plain decode cost (ms/token at m=1), refreshed by control rounds. It is
    // a property of the ENGINE (not of the content), so it persists across requests; the
    // MTP side is per request (Lane::ema_mtp) because acceptance is a property of content.
    ema_plain: f64,
    // API-parity DDS: dynamic draft stop target (0 = fixed depth) + the shared calibrator.
    draft_conf: f64,
    cal: DraftCal,
    // Prefix cache (2026-09-26): per slot, the prompt tokens whose recurrent state is snapshotted
    // (prompt[..snap]) — a new prompt that starts with them resumes prefill at `snap`.
    prefix_on: bool,
    cache: Vec<Option<Vec<u32>>>,
    // WP16: intermediate recurrent checkpoints at aligned chunk ends (None = off: --prefix-cache
    // off, --wp16-off=1, or a --prefix-ckpt-mem-gb cap that holds none) + the XCHECK diagnostic.
    wp16: Option<crate::exl3_forward::CkptStore<crate::exl3_forward::RecurCkpt>>,
    wp16_xcheck: bool,
    // WP08: stop_on_loop (window, min_reps) — the reference implementation's (300, 3) unless --loop-detect off.
    loop_cfg: Option<(usize, usize)>,
    // WP15: penalty window (sustain = decay = this; the reference implementation's penalty_range 1024).
    pen_range: usize,
    // WP24: --spec-sampling ratio (real-q: sampled drafts, accept min(1, p/q'), exact residual)
    // for sampled lanes; false = match (today). draft_temp = --draft-temperature (None = the
    // request's temperature).
    spec_ratio: bool,
    draft_temp: Option<f32>,
    // WP24 gate (--probe-spec-sampling) round bookkeeping; None in the server.
    gate: Option<GateStats>,
    // for the [loop] log's text tail only
    tok: Arc<QwenTokenizer>,
    // WP02 liveness. exit_on_fatal = --exit-on-fatal (default ON since v0.7.3, owner decision
    // 2026-10-06): a sticky CUDA error exits 70 so a supervisor restarts the server; off, the
    // engine goes DEAD (/health and every request answer 503). `fatal` is set by admit/step when
    // the context is poisoned.
    exit_on_fatal: bool,
    fatal: Option<String>,
    inject_panic: Option<usize>, // --inject-panic-steps=<n>: panic at decode step n (liveness gate)
    steps: usize,
    // TP-C: Some = this scheduler is one rank of a TP group (SPMD lockstep, head-authoritative timing)
    tp: Option<TpSync>,
    // TP-D: the last admit's resume points (prompt-end reuse, WP16/C1 checkpoint q) — proven
    // identical on every rank by the seam lockstep
    last_resume: (usize, usize),
    // VIS-2: the text config's interleaved mrope_section (rope_parameters.mrope_section; None = the
    // model declares no interleaved mrope -> image requests are refused)
    mrope_section: Option<[usize; 3]>,
    // VIS-4: the last image admit's (image tokens, rope delta, FNV of the fp16 rows) — proven equal on
    // every rank by the seam lockstep (None = a text admit)
    vis_seam: Option<(u32, i32, u64)>,
    // Public issue #24: the prefill in flight as /metrics sees it — (slot, positions written so
    // far). Set per completed chunk, cleared when the admit ends (the lane then carries `pos`).
    kv_pf: Option<(usize, usize)>,
}

/// WP24 gate: MTP round anatomy of one arm (non-plain rounds only) + the first-round drafts.
#[derive(Default, Debug)]
struct GateStats {
    rounds: u64,
    drafted: u64,
    accepted: u64,
    ratio_rounds: u64,
    d0: Vec<i32>,
    // TP-C harness: MTP round wall ms (the head's under TP), control rounds, plain steps
    ms: f64,
    ctrl: u64,
    plain: u64,
}

/// TP-C: the SPMD lockstep of a TP scheduler (None at TP=1 — every hook below is then a no-op and
/// the TP=1 serve path is unchanged). Every rank runs the same scheduler on the same requests;
/// the head (rank 0) is AUTHORITATIVE for every decision that reads a wall clock (design §4.2):
/// after each round the ranks agree on (step, accept, verify width, hash of emitted ids + drafts +
/// draft confidences) — the load-bearing agree() guard, AGENTS §2.10 — and the node ADOPTS the
/// head's measured round / plain-step ms and graph-capture flag. Everything downstream (the §6
/// per-request MTP-off, the control cadence, the WP23 cost guard's target, the cost fit) is then a
/// pure function of replicated state, so both ranks choose the same draft stop and the same verify
/// width by construction; the agree token proves it every round.
/// TP-E: per-call-site wall time of the host lockstep waits (always on; ~50 ns per call). Printed and
/// reset whenever the engine goes idle, so every line covers one request (or one idle-to-idle span).
const WP_SITES: [&str; 8] = ["fenceA", "xchg", "fenceB", "pv_xchg", "pv_agree", "ctl_recv", "ctl_send", "step_go"];
const WP_EDGES_US: [u64; 5] = [20, 50, 200, 1000, 2000];
#[derive(Default, Clone)]
pub(crate) struct WaitProf {
    n: [u64; 8],
    sum_ns: [u64; 8],
    max_ns: [u64; 8],
    hist: [[u64; 6]; 8],
    sleeps: [u64; 8],
    rounds: u64,
}

impl WaitProf {
    #[inline]
    fn rec(&mut self, site: usize, t0: std::time::Instant, s0: u64) {
        let ns = t0.elapsed().as_nanos() as u64;
        self.n[site] += 1;
        self.sum_ns[site] += ns;
        self.max_ns[site] = self.max_ns[site].max(ns);
        let us = ns / 1000;
        let b = WP_EDGES_US.iter().position(|&e| us < e).unwrap_or(WP_EDGES_US.len());
        self.hist[site][b] += 1;
        self.sleeps[site] += crate::net::wait_sleeps().saturating_sub(s0);
    }
    fn line(&self, rank: i32) -> String {
        let mut o = format!("[tp-wait] rank {rank} rounds {}", self.rounds);
        let mut tot = 0u64;
        for i in 0..WP_SITES.len() {
            if self.n[i] == 0 { continue; }
            tot += if i < 5 || i == 7 { self.sum_ns[i] } else { 0 };
            let h = &self.hist[i];
            o += &format!(" | {} n {} mean {:.1}us max {:.0}us sleeps {} h[<20,<50,<200,<1k,<2k,>=2k]us {}/{}/{}/{}/{}/{}",
                          WP_SITES[i], self.n[i], self.sum_ns[i] as f64 / self.n[i] as f64 / 1e3,
                          self.max_ns[i] as f64 / 1e3, self.sleeps[i], h[0], h[1], h[2], h[3], h[4], h[5]);
        }
        if self.rounds > 0 {
            o += &format!(" | lockstep(fenceA+xchg+fenceB+pv+go) {:.3} ms/round", tot as f64 / self.rounds as f64 / 1e6);
        }
        // TP-G: transport health (cumulative): tail-guard fires (MUST stay 0) and GPU-proved epochs
        o += &format!(" | xport tail_fires {} gpu_rx_skips {} abort {} reuse-gate binds {}", crate::net::traced_tail_fires(),
                      crate::net::traced_gpu_rx_skips(), crate::net::traced_abort_status(), crate::net::traced_gate_waits());
        o += &format!(" | k2-prefetch hints armed {} used {}", crate::exl3_forward::xtp::PF_ARMED.load(std::sync::atomic::Ordering::Relaxed),
                      crate::exl3_forward::xtp::PF_USED.load(std::sync::atomic::Ordering::Relaxed));
        o
    }
}

pub(crate) struct TpSync {
    pub rank: i32,
    /// TP-4D: the TP world (2 = the pairwise lockstep of TP-C..TP-I, byte-for-byte; > 2 = the head-hub lockstep
    /// of `tp_lockstep`, ONE merged hub op per lockstep point).
    pub world: i32,
    /// TP-4D (audit C5): per-wait deadline of a world > 2 lockstep hub op, ms (0 = unbounded; always 0 at world 2,
    /// which keeps the pairwise channel and its own 10 s `net_agree` timeout).
    pub deadline_ms: u64,
    /// TP-4D: rounds that reported a device-epoch skew between ranks (diagnostic print throttle)
    epoch_warns: u64,
    pub wp: WaitProf,
    /// agree_ext step counter (24-bit field on the wire)
    step: u64,
    /// G-T1-b graphed: digest the logits rows + the tap of every step/round and compare across
    /// ranks (a dtoh per step — the harness's identity pass only)
    pub ident: bool,
    pub ident_bufs: u64,
    pub ident_bad: u64,
    pub agrees: u64,
    /// TP-D: pre-verify agrees (DDS: the verify width + drafts, BEFORE the verify graph launches)
    pub pre_agrees: u64,
    /// TP-D: the serve control plane (rank 0: one stream per node; a node: the stream to the head).
    /// Empty in the TP-C harness (run_spec), where nothing is head-decided mid-step.
    ctl: Vec<std::net::TcpStream>,
    /// `--tp-width-drill N` (test drill, per box, one-shot; None = not armed): see `width_said`.
    drill_width: Option<usize>,
    /// rounds the drill has lied about: 0, then 1 for the rest of the boot (it is one-shot)
    pub drill_hits: u64,
}

impl TpSync {
    pub(crate) fn new(rank: i32, world: i32) -> Self {
        let deadline_ms = if world > 2 { lockstep_deadline_ms() } else { 0 };
        // S-B9-CF-AGREE16B (W3): the drill is armed (or declared inert) HERE, once, with a boot line
        // either way; `net::drill_plan` is the pure decision.
        let plan = crate::net::drill_plan(crate::opts::get(crate::opt!("tp-agree-drill")).as_deref(),
                                          crate::opts::get(crate::opt!("tp-width-drill")).as_deref(), true, Some((rank, world)));
        for l in &plan.lines { crate::dreprintln!("{l}"); }
        TpSync { rank, world, deadline_ms, epoch_warns: 0, step: 0, ident: false, ident_bufs: 0, ident_bad: 0, agrees: 0,
                 pre_agrees: 0, ctl: Vec::new(), wp: WaitProf::default(), drill_width: plan.width, drill_hits: 0 }
    }

    /// The verify width this rank REPORTS in a round's lockstep words: the real one, except ONCE under the
    /// `--tp-width-drill N` test drill, where the first round of width N is reported as `N - 1 - rank` on
    /// the box that carries the flag (on the head, N = 16: "this rank believes 15, the peer 16"). The
    /// forward itself is untouched: only the reported word lies, so a guard that cannot tell the two widths
    /// apart lets the drill pass in silence and one that can aborts the link at that round.
    ///
    /// S-B9-CF-AGREE16B: it was `--tp-agree-drill N`, it lied on EVERY N-row round, and every rank said
    /// N - 1. Two armed ranks then agreed on the lie, and a real 15-against-16 disagreement stayed masked
    /// for the whole boot. One shot bounds that to one round, and the rank term makes two armed ranks
    /// report different widths (`net::drill_plan` arms only where `N > rank`, so this cannot underflow).
    fn width_said(&mut self, m: usize) -> usize {
        match self.drill_width {
            Some(n) if n == m && self.drill_hits == 0 && m > self.rank as usize => {
                self.drill_hits = 1;
                let said = m - 1 - self.rank as usize;
                crate::deprintln!("[tp-agree] DRILL (one-shot, now spent): rank {} reports verify width {said} for a {m}-row round \
                           (lockstep step {}); from here on it reports the truth", self.rank, self.step);
                said
            }
            _ => m,
        }
    }

    /// TP-D: a boolean the HEAD decides inside a step (a prefill chunk boundary's "client gone"),
    /// shipped over the control stream; every rank returns the head's value at the same program
    /// point. No control plane (the harness) = false on every rank (nothing is cancellable there).
    fn head_flag(&mut self, v: bool) -> Result<bool> {
        if self.ctl.is_empty() {
            return Ok(false);
        }
        if self.rank == 0 {
            for s in self.ctl.iter_mut() {
                crate::tp_serve::send_serving(s, &crate::tp_serve::ServingMsg::HeadFlag { v })?;
            }
            Ok(v)
        } else {
            match crate::tp_serve::recv_serving(&mut self.ctl[0])? {
                crate::tp_serve::ServingMsg::HeadFlag { v } => Ok(v),
                other => anyhow::bail!("TP control plane out of step: expected HeadFlag, got {other:?}"),
            }
        }
    }

    /// FLASHNEXT-DF2-117 (ISSUES 117): ONE sampled DFlash2 draft per round for every rank. The head
    /// ships ITS walk's result (`mine`) to every node and keeps using it (returns None); a node
    /// blocks here at the same program point, adopts the head's draft (returns Some) and never
    /// draws its own. No control plane (the harness, TP=1) = None: the caller uses its own draw.
    /// `lane`/`p` ride the frame as the out-of-step tripwire.
    pub(crate) fn df2_draft(&mut self, lane: usize, p: usize,
                            mine: Option<&crate::dflash2::round::Df2SampleOut>)
                            -> Result<Option<crate::dflash2::round::Df2SampleOut>> {
        use crate::tp_serve::ServingMsg;
        if self.ctl.is_empty() { return Ok(None); }
        if self.rank == 0 {
            let m = mine.ok_or_else(|| anyhow::anyhow!("df2 draft ship: the head has no sampled draft to ship"))?;
            let msg = ServingMsg::Df2Draft {
                lane: lane as u32, p: p as u64, tokens: m.tokens.clone(),
                q_rows: m.q_rows.iter().map(|x| x.to_bits()).collect(), cand_tok: m.cand_tok.clone(),
                cand_q: m.cand_q.iter().map(|x| x.to_bits()).collect(),
            };
            for s in self.ctl.iter_mut() { crate::tp_serve::send_serving(s, &msg)?; }
            DF2_SHIPPED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(None)
        } else {
            match crate::tp_serve::recv_serving(&mut self.ctl[0])? {
                ServingMsg::Df2Draft { lane: l, p: pp, tokens, q_rows, cand_tok, cand_q } => {
                    anyhow::ensure!(l as usize == lane && pp as usize == p,
                        "TP control plane out of step: the head's DF2 draft is for lane {l} at p {pp}; this rank is at lane {lane} p {p}");
                    DF2_SHIPPED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    Ok(Some(crate::dflash2::round::Df2SampleOut {
                        tokens, q_rows: q_rows.into_iter().map(f32::from_bits).collect(), cand_tok,
                        cand_q: cand_q.into_iter().map(f32::from_bits).collect(),
                    }))
                }
                other => anyhow::bail!("TP control plane out of step: expected Df2Draft, got {other:?}"),
            }
        }
    }
    /// Whether this rank sits on a serve control plane (false in the harness / at TP=1).
    pub(crate) fn has_ctl(&self) -> bool { !self.ctl.is_empty() }

    /// `spill_agree` at world > 2: the head hub, generic over the transport so the four-rank mock runs the real
    /// frame + verdict (ISSUES 121: the verdict had never passed on hardware and no test reached it).
    pub(crate) fn spill_agree_hub<X: crate::net::HubXport>(&mut self, x: &mut X, ok: bool) -> Result<bool> {
        use crate::tp_lockstep as ls;
        let all = crate::net::hub_all(x, &ls::spill_frame(ok), ls::SPILL_WIRE)
            .map_err(|e| self.hub_fail("spill agree", &format!("{e:#}")))?;
        ls::spill_verdict(&all, self.world as usize).map_err(|why| self.hub_fail("spill agree", &why))
    }

    /// LANERES3: the spill tier's lockstep bit — the symmetric sibling of `head_flag`.
    /// Every rank contributes whether its local disk op succeeded (capture write /
    /// restore read) and the verdict is the AND of all ranks. Called from the
    /// admission-phase spill hooks, which every rank executes at the same program point
    /// of the same shipped admit sequence, so the rendezvous pairs up the same way
    /// HeadFlag frames pair up inside a prefill. This keeps the replicated
    /// `kv_spill::Policy` identical on every rank: a key no rank-local disk state can
    /// back on ALL ranks is dropped everywhere (counted), never acted on by one rank
    /// alone — a disk that diverged can never desync the pair.
    fn spill_agree(&mut self, ok: bool) -> Result<bool> {
        use crate::tp_serve::ServingMsg;
        if self.world > 2 {
            let mut x = crate::net::link_hub(self.deadline_ms)?;
            return self.spill_agree_hub(&mut x, ok);
        }
        if self.ctl.is_empty() {
            return Ok(ok); // no control plane (harness paths): local truth, ranks symmetric by construction
        }
        if self.rank == 0 {
            crate::tp_serve::send_serving(&mut self.ctl[0], &ServingMsg::SpillFlag { ok })?;
            match crate::tp_serve::recv_serving(&mut self.ctl[0])? {
                ServingMsg::SpillFlag { ok: peer } => Ok(ok && peer),
                other => anyhow::bail!("TP control plane out of step: expected SpillFlag, got {other:?}"),
            }
        } else {
            match crate::tp_serve::recv_serving(&mut self.ctl[0])? {
                ServingMsg::SpillFlag { ok: peer } => {
                    crate::tp_serve::send_serving(&mut self.ctl[0], &ServingMsg::SpillFlag { ok })?;
                    Ok(ok && peer)
                }
                other => anyhow::bail!("TP control plane out of step: expected SpillFlag, got {other:?}"),
            }
        }
    }

    /// TP-E: the per-step "go" of a LIVE engine over the RDMA link instead of the TCP control stream.
    /// Measured (TP-E [tp-wait]): the node waited ~0.53 ms per round in the TCP recv of an EMPTY Step
    /// (kernel TCP over the CX7 netdev: NIC interrupt moderation + wakeup), and the head then waited
    /// the same ~0.5 ms at its pre-verify exchange — the node started every round half a millisecond
    /// late. Now, whenever a lane is live (replicated state, so both ranks take this path together),
    /// the head publishes (step number, event count) in an RDMA exchange (~8 us); the TCP Step is sent
    /// and read ONLY when the step carries events (admit / cancel — a few per request). Shape =
    /// exchange (rendezvous) then agree (fence): the exchange reuses hot-path ring slot memory, and a
    /// plain / seam step may run collectives right after it (TP-C fence-B rule). The node adopts the
    /// head's words; the head's own words come back as the node's zeros (ignored).
    /// `--exl3-tp-fastctl=0` (diagnostic, rides TpConfig's CLI option-registry snapshot) = the TP-D TCP-every-step path.
    fn step_go(&mut self, step_no: u64, n_events: usize) -> Result<(u64, usize)> {
        if self.world > 2 {
            let mut x = crate::net::link_hub(self.deadline_ms)?;
            return self.step_go_hub(&mut x, step_no, n_events);
        }
        let tag = 0x60_000000u32 | (step_no as u32 & 0xFF_FFFF);
        let mine: [u32; 3] = if self.rank == 0 { [tag, step_no as u32, n_events as u32] } else { [0, 0, 0] };
        let (t0, s0) = (std::time::Instant::now(), crate::net::wait_sleeps());
        let peer = crate::net::exchange_u32s(&mine, 3 + 4)?;
        let (hs, hn) = if self.rank == 0 { (step_no, n_events) } else {
            anyhow::ensure!(peer[0] & 0xFF00_0000 == 0x6000_0000, "TP step-go: head sent tag {:#x}", peer[0]);
            (((step_no & !0xFFFF_FFFF) | peer[1] as u64), peer[2] as usize)
        };
        // the fence takes the next lockstep step number (a plain agree field, like fence A): its field
        // then differs from the stale fence-B (bit 23) token it replaces and from the next pre-verify's.
        self.step += 1;
        let r = tp_agree_eq(agree_field(self.step, AGREE_GO), 0, 0, go_fence_word(hs, hn));
        self.wp.rec(7, t0, s0);
        if let Err(why) = r {
            crate::net::abort_link();
            anyhow::bail!("TP step-go fence FAILED at control step {hs}: {why}");
        }
        Ok((hs, hn))
    }

    /// TP-D (design §4.2 belt-and-braces): agree on the DDS verify width + the drafts BEFORE the
    /// verify graph launches — the width selects the graph and therefore the barrier sequence, so a
    /// divergent draft stop must be caught before any mismatched barrier runs, not one round later.
    ///
    /// Shape = exchange (rendezvous) -> agree (fence): net_agree is ONE mirrored token slot, so two
    /// agrees with no rendezvous between them race (the peer, asleep in its spin, never sees the
    /// first token before the second overwrites it — the 2026-09-29 served smoke died exactly so:
    /// fence B of round n, then this agree after draft passes that have no collective). The
    /// exchange is the rendezvous that makes the fence-B token observed before this one is
    /// written; the agree after it is the fence that keeps the verify's ring epochs off the
    /// exchange slots (TP-C's fence-B rule).
    ///
    /// TP-H: `launched` = the head draft passes this rank issued this round (w, plus one when a
    /// speculative pass ran past the DDS stop). The sharded MTP head puts collectives in every draft
    /// pass, so a rank-local speculative-launch decision that diverged would mis-pair the verify's
    /// first barrier with the wasted pass's; the count is agreed HERE, before the verify launches
    /// (fail-stop, never a mis-paired all-reduce). Free at world 2 with a replicated head too.
    fn pre_verify(&mut self, w: usize, drafts: &[i32], launched: usize) -> Result<()> {
        if self.world > 2 {
            let mut x = crate::net::link_hub(self.deadline_ms)?;
            return self.pre_verify_hub(&mut x, w, drafts, launched);
        }
        let step = agree_field(self.step + 1, AGREE_PV);
        let h = fnv32(0x5EED, drafts.iter().map(|&d| d as u32));
        let tag = 0x9E00_0000u32 | (step as u32 & 0xFFFF);
        let mine = [tag, w as u32, h, launched as u32];
        // S-B9-REL-TP2RACE (issue #10): the reporter's diagnostic — `device_epoch & 7` next to the
        // exchange generation's slot `gen & 7` — plus the deterministic fault-injection hook. With
        // --tp-race-probe <ms> the node's proxy holds every epoch release, so its last screen epoch
        // sits validated-but-unconsumed when we get here; aligning our next exchange generation onto
        // that epoch's recv slot (gen & 7 == device_epoch & 7) makes our frame land exactly on the
        // withheld payload. Both ranks run the identical alignment (same device epoch, same lockstep
        // gen), so the paired generations stay equal. On the dedicated-slot transport this changes
        // nothing (the frame never touches the doorbell rings); on the old hot-ring transport the
        // node screens a corrupted candidate set and this bail or the proxy's CLOBBER line fires.
        let e_last = crate::net::traced_device_epoch();
        let g0 = crate::net::traced_xchg_gen();
        let g = if crate::net::race_probe_hold_ms() > 0 {
            let g1 = crate::net::traced_gen_add(crate::net::probe_gen_delta(g0, e_last & 7));
            crate::deprintln!("[tp-race-probe] rank {}: pre-verify aligned xchg gen {g0} -> {g1} onto epoch {e_last} \
                       (device_epoch&7 = {}, gen&7 = {}), hold {} ms on rank 1",
                      self.rank, e_last & 7, g1 & 7, crate::net::race_probe_hold_ms());
            g1
        } else { g0 };
        let (t0, s0) = (std::time::Instant::now(), crate::net::wait_sleeps());
        let peer = crate::net::exchange_u32s(&mine, 4 + 4)?; // + the 16-byte tail guard (as tp_round)
        self.wp.rec(3, t0, s0);
        if peer != mine {
            crate::net::abort_link();
            anyhow::bail!("TP pre-verify FAILED (round {}): this rank width {w} drafts {h:08x} head passes launched {launched}, \
                           peer {:?} — device_epoch&7 = {}, xchg gen&7 = {} — ranks chose different verify widths, drafts \
                           or speculative head-pass launches; link aborted before the verify",
                          self.step + 1, peer, e_last & 7, g & 7);
        }
        let (t0, s0) = (std::time::Instant::now(), crate::net::wait_sleeps());
        let r = tp_agree_eq(step, w, w + 1, h);
        self.wp.rec(4, t0, s0);
        if let Err(why) = r {
            crate::net::abort_link();
            anyhow::bail!("TP pre-verify agree FAILED (round {}, draft width {w}, drafts hash {h:08x}: {why}) — ranks chose \
                           different verify widths or drafts; link aborted before the verify", self.step + 1);
        }
        self.pre_agrees += 1;
        Ok(())
    }

    /// S-B9-CF-PACK5: the PACKED round's pre-verify lockstep point — ONE exchange + ONE agree for a
    /// pair (the serial arm does one per lane), after BOTH lanes drafted and before the packed verify
    /// graph (whose R-keyed barrier sequence every rank must run alike) launches.
    ///
    /// `r` = the total verify rows R = m_0 + m_1 (the explicit number the barrier sequence depends
    /// on), `m0` = the first segment's rows, `words` = `lane_pack::pre_words` (per segment: rows,
    /// slot, position, bonus token, drafts — the design's OQ1 combined hash: self-delimiting, so a
    /// single FNV over the concatenation catches every disagreement a per-request vector would),
    /// `launched` = the SUM of both lanes' head draft passes issued on this rank.
    ///
    /// Same shape and slot discipline as `pre_verify` (exchange = rendezvous, then agree = fence;
    /// the wire sizes are unchanged: [tag, R, hash, launched] + the 16-byte tail guard). The width
    /// the exchange and the agree carry is R (<= 16; AGREE16 made the token's 5-bit field exact
    /// up to 31), the accept byte carries m0; `--tp-width-drill N` lies about a pack of R = N rows
    /// HERE (before the verify), so the drill proves the guard fires before the verify.
    fn pre_verify_pack(&mut self, r: usize, m0: usize, words: &[i32], launched: usize) -> Result<()> {
        let r_said = self.width_said(r);
        if self.world > 2 {
            let mut x = crate::net::link_hub(self.deadline_ms)?;
            return self.pre_verify_hub(&mut x, r_said, words, launched);
        }
        let step = agree_field(self.step + 1, AGREE_PV);
        // a domain of its own (tag 0x9F.., seed 0x9AC5_0001): a rank that planned a PACK where its peer ran
        // a lone round (whose pre-verify uses 0x9E.. / 0x5EED) is caught by the exchange word itself,
        // exactly, not by two hashes over different word lists happening to differ
        let h = fnv32(0x9AC5_0001, words.iter().map(|&d| d as u32));
        let tag = 0x9F00_0000u32 | (step as u32 & 0xFFFF);
        let mine = [tag, r_said as u32, h, launched as u32];
        // the race probe's deterministic alignment, exactly as `pre_verify` applies it (the pack's
        // exchange has the same slot-aliasing exposure, so the REL-TP2RACE drills cover it)
        let e_last = crate::net::traced_device_epoch();
        let g0 = crate::net::traced_xchg_gen();
        let g = if crate::net::race_probe_hold_ms() > 0 {
            let g1 = crate::net::traced_gen_add(crate::net::probe_gen_delta(g0, e_last & 7));
            crate::dreprintln!("[tp-race-probe] rank {}: pack pre-verify aligned xchg gen {g0} -> {g1} onto epoch {e_last} \
                       (device_epoch&7 = {}, gen&7 = {}), hold {} ms on rank 1",
                      self.rank, e_last & 7, g1 & 7, crate::net::race_probe_hold_ms());
            g1
        } else { g0 };
        let (t0, s0) = (std::time::Instant::now(), crate::net::wait_sleeps());
        let peer = crate::net::exchange_u32s(&mine, 4 + 4)?; // + the 16-byte tail guard
        self.wp.rec(3, t0, s0);
        if peer != mine {
            crate::net::abort_link();
            anyhow::bail!("TP pack pre-verify FAILED (round {}): this rank R = {r_said} rows, pack hash {h:08x} (combined over every \
                           segment's rows, slot, position, bonus, sampler and drafts), head passes launched {launched}; peer {:?} \
                           [tag, R, hash, launched] — device_epoch&7 = {}, xchg gen&7 = {} — ranks chose different packs (row total, \
                           slots, drafts or speculative head-pass launches); this rank's pack: {}; link aborted before the packed verify",
                          self.step + 1, peer, e_last & 7, g & 7, crate::lane_pack::describe_pre_words(words));
        }
        let (t0, s0) = (std::time::Instant::now(), crate::net::wait_sleeps());
        let rr = tp_agree_eq(step, m0, r_said, h);
        self.wp.rec(4, t0, s0);
        if let Err(why) = rr {
            crate::net::abort_link();
            anyhow::bail!("TP pack pre-verify agree FAILED (round {}, R = {r_said} rows, first segment {m0} rows, pack hash {h:08x}: \
                           {why}) — ranks chose different packs; this rank's pack: {}; link aborted before the packed verify",
                          self.step + 1, crate::lane_pack::describe_pre_words(words));
        }
        self.pre_agrees += 1;
        Ok(())
    }
}

// ------------------------------------------------------------------------------------------------------
// TP-4D: the world > 2 lockstep — ONE merged head-hub op per lockstep point.
//
// World 2 keeps the TP-C..TP-I sequence above (fence A agree, u32 exchange, fence B agree). Those agrees and
// that exchange are PAIRWISE ops over the HOT-PATH ring memory (the exchange stages in send slot 0 and lands in
// recv slot gen & 7, the rings the device doorbell epochs use), which is the only reason fence A/B and
// `drain_sends` exist. At world > 2 every lockstep frame rides `net::hub_all` on the DEDICATED control slots
// (`net_exchange_one`: its own send staging slot, its own per-sender recv rings of depth TP_CTRL_RING, a stable
// copy per sender) — no byte of a hub frame shares memory with the hot path, so
//   * fence B has nothing to protect: after the op the next forward may start at once (its epochs cannot reach
//     the control slots), and the op's two rounds are already a barrier (a rank leaves round 2 only after the
//     head has heard from every rank);
//   * fence A's "both ranks finished the forward" is the op's own arrival condition (the frame is built from
//     the finished forward's host results);
//   * `drain_sends` (exl3_forward.rs, gated to world 2) has no aliasing left to drain.
// The claim is void if a hub frame ever shares memory with the hot path (the model in `net::hub_tests` assumes
// separate slots). Every rank ends up holding EVERY rank's frame, runs the same pure verdict
// (`tp_lockstep::*_verdict`) over the same data and therefore reaches the same verdict: all ranks abort together,
// none is left parked in a later exchange. Transport failure (dead peer, code 10 / deadline, code 12 / link
// abort) makes `hub_all` fail on the ranks that see it; the dead rank's block reads zero on the rest, which the
// verdict rejects (tags differ), so every live rank ends in an error (`net::hub_tests` kills / hangs a node).

/// The per-wait deadline of a world > 2 lockstep hub op: `--tp-lockstep-timeout-ms`, default 30 s, 0 = unbounded.
pub(crate) fn lockstep_deadline_ms() -> u64 {
    crate::opts::var(crate::opt!("tp-lockstep-timeout-ms")).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(30_000)
}

impl TpSync {
    fn hub_fail(&self, what: &str, why: &dyn std::fmt::Display) -> anyhow::Error {
        // keep an abort code a failed exchange already recorded (10 = peer dead, 12 = lockstep deadline)
        crate::net::abort_link_keep_code();
        anyhow::anyhow!("TP lockstep {what} FAILED at rank {} (world {}): {why}", self.rank, self.world)
    }

    /// world > 2 `step_go`: [tag, counter field, head step, head events] from every rank; the verdict proves all
    /// ranks are at the same lockstep counter and the nodes adopt the head's (step, events).
    pub(crate) fn step_go_hub<X: crate::net::HubXport>(&mut self, x: &mut X, step_no: u64, n_events: usize)
                                                       -> Result<(u64, usize)> {
        use crate::tp_lockstep as ls;
        let (t0, s0) = (std::time::Instant::now(), crate::net::wait_sleeps());
        self.step += 1;
        let field = agree_field(self.step, AGREE_GO) as u32;
        let frame = ls::go_frame(self.rank == 0, field, step_no, n_events);
        let all = crate::net::hub_all(x, &frame, ls::GO_WIRE).map_err(|e| self.hub_fail("step-go", &format!("{e:#}")))?;
        self.wp.rec(7, t0, s0);
        let (hs, hn) = ls::go_verdict(&all, self.world as usize)
            .map_err(|why| self.hub_fail("step-go", &format!("fence FAILED at control step {step_no}: {why}")))?;
        // same adoption rule as the pairwise path: the head's low 32 bits under this rank's own high bits
        let hs = if self.rank == 0 { step_no } else { (step_no & !0xFFFF_FFFF) | hs as u64 };
        Ok((hs, hn as usize))
    }

    /// world > 2 `pre_verify`: every rank's (width, drafts hash, head passes launched) at the same lockstep
    /// counter — compared by EVERY rank, BEFORE the verify graph (whose width selects the barrier sequence) launches.
    pub(crate) fn pre_verify_hub<X: crate::net::HubXport>(&mut self, x: &mut X, w: usize, drafts: &[i32], launched: usize)
                                                          -> Result<()> {
        use crate::tp_lockstep as ls;
        let field = agree_field(self.step + 1, AGREE_PV) as u32;
        let h = fnv32(0x5EED, drafts.iter().map(|&d| d as u32));
        let frame = ls::pv_frame(field, w, h, launched);
        let (t0, s0) = (std::time::Instant::now(), crate::net::wait_sleeps());
        let all = crate::net::hub_all(x, &frame, ls::PV_WIRE).map_err(|e| self.hub_fail("pre-verify", &format!("{e:#}")))?;
        self.wp.rec(3, t0, s0);
        ls::pv_verdict(&all, self.world as usize).map_err(|why| {
            self.hub_fail("pre-verify", &format!("round {}: this rank width {w} drafts {h:08x} head passes launched \
                {launched}; {why} — ranks chose different verify widths, drafts or speculative head-pass launches; \
                link aborted before the verify", self.step + 1))
        })?;
        self.pre_agrees += 1;
        Ok(())
    }

    /// world > 2 `tp_round`: the merged lockstep op (tripwire words + the head's timing + a device-epoch probe) and,
    /// with `ident` on, a second op with the digests of `digests(m)` compared against the head's. `digests` runs
    /// only after the verdict proved every rank chose the same width `m`.
    pub(crate) fn round_hub<X: crate::net::HubXport>(&mut self, x: &mut X, what: &str, a: usize, m: usize, words: &[u32],
                                                     ms: f64, plain_ms: f64, captured: bool, epoch: u64,
                                                     digests: impl FnOnce(usize) -> Result<Vec<u64>>)
                                                     -> Result<(f64, f64, bool)> {
        use crate::tp_lockstep as ls;
        self.step += 1;
        let (step, world) = (self.step, self.world as usize);
        let h = fnv32(0, words.iter().copied());
        let field = agree_field(step, AGREE_A) as u32;
        // S-B9-CF-AGREE16: the frame words are whole u32s; the real accept count and width ride them (the
        // saturation at 255 / at 15 inherited from the pairwise token made 15, 16 and 17 rows one word here too)
        let frame = ls::rnd_frame(&ls::RoundIn { step, field, accept: a, width: self.width_said(m), hash: h,
                                                 epoch, ms, plain_ms, captured });
        let (t0, s0) = (std::time::Instant::now(), crate::net::wait_sleeps());
        let all = crate::net::hub_all(x, &frame, ls::RND_WIRE)
            .map_err(|e| self.hub_fail("round", &format!("at {what} (lockstep step {step}): {e:#}")))?;
        self.wp.rec(1, t0, s0);
        self.wp.rounds += 1;
        let out = ls::rnd_verdict(&all, world).map_err(|why| {
            self.hub_fail("round", &format!("agree() FAILED at {what} (lockstep step {step}, accept {a}, width {m}, \
                hash {h:08x}): {why} — ranks diverged or the link aborted"))
        })?;
        if self.ident {
            let d = digests(m).map_err(|e| self.hub_fail("ident", &format!("digest failed at {what}: {e:#}")))?;
            let frame = ls::id_frame(step, &d);
            let all = crate::net::hub_all(x, &frame, ls::id_wire(d.len()))
                .map_err(|e| self.hub_fail("ident", &format!("at {what} (lockstep step {step}): {e:#}")))?;
            let bad = ls::id_verdict(&all, world, d.len())
                .map_err(|why| self.hub_fail("ident", &format!("step sync at {what}: {why}")))?;
            self.ident_bufs += d.len() as u64;
            self.ident_bad += bad as u64;
            if bad > 0 && self.ident_bad as usize == bad {
                crate::deprintln!("[tp-ident] FIRST MISMATCH at {what} (lockstep step {step}, width {m}): {bad} of {} digests differ \
                           (logits rows 0..{m}, resid, taps; some rank differs from the head)", d.len());
            }
        }
        self.agrees += 1;
        if !out.epoch_skew.is_empty() {
            self.epoch_warns += 1;
            if self.epoch_warns <= 5 || self.epoch_warns % 1000 == 0 {
                crate::deprintln!("[tp-round] EPOCH-PROBE at {what} (lockstep step {step}, #{}): device epoch differs from the head's on \
                           (rank, delta) {:?}", self.epoch_warns, out.epoch_skew);
            }
        }
        Ok((out.ms, out.plain_ms, out.captured))
    }
}

/// TP-E: the 24-bit agree step field = [kind:2 | lockstep counter mod 2^22]. `net_agree` returns the
/// peer's CURRENT token as soon as its step field matches, so the field of every agree must differ from
/// the peer's previous token (the stale value a waiting rank may read). Adjacent agrees always differ in
/// kind (B -> GO -> PV -> A -> B, or B -> A on plain rounds), and a same-kind collision needs the counter
/// 2^22 apart. (Before TP-E, fence A used the raw 24-bit counter and pre-verify `S & 0x3FFFFF | 1<<22`:
/// equal once S reached [2^22, 2^23) — ~4 M rounds — which the new comparison would turn into an abort.)
const AGREE_A: u64 = 0;
const AGREE_PV: u64 = 1;
const AGREE_B: u64 = 2;
const AGREE_GO: u64 = 3;
fn agree_field(counter: u64, kind: u64) -> u64 { (counter & 0x3F_FFFF) | (kind << 22) }

/// TP-E: the lockstep agree WITH the comparison. `net::agree_ext` only rendezvouses on the step field
/// and returns the PEER's (accept, hash); until TP-E the EXL3 callers checked `is_none()` alone, so a
/// divergent accept count / emitted-ids hash / boot config would have sailed through every "agree ok"
/// (only the timeout could fire). The NVFP4 path (batch.rs) always compared. Ok(()) = both ranks
/// published the identical token; Err = timeout/abort or a mismatch (message says which).
pub(crate) fn tp_agree_eq(step: u64, accept: usize, k_verify: usize, hash: u32) -> std::result::Result<(), String> {
    let (a, k) = tp_agree_fields(accept, k_verify)?;
    tp_agree_decide(a, k, hash, crate::net::agree_ext(step, a, k, hash))
}

/// The (accept, width) pair as the pairwise token carries it: accept in a byte, width in the five bits of
/// `net::agree_word`. S-B9-CF-AGREE16: a value that does not fit is REFUSED (the caller aborts the link),
/// never saturated — a saturation at 15 here made a 15-row and a 16-row verify one and the same word.
pub(crate) fn tp_agree_fields(accept: usize, k_verify: usize) -> std::result::Result<(u8, u8), String> {
    if accept > u8::MAX as usize || k_verify > crate::net::AGREE_WIDTH_MAX {
        return Err(format!("REFUSED: accept count {accept} / verify width {k_verify} do not fit the lockstep token (accept <= 255, \
                            width <= {}); refusing instead of sending a value the peer could mistake for another",
                           crate::net::AGREE_WIDTH_MAX));
    }
    Ok((accept as u8, k_verify as u8))
}

/// The step-go fence's digest of the (control step, event count) pair both ranks hold after the exchange.
/// S-B9-CF-AGREE16: it was `(hs as u32) ^ ((hn as u32) << 24) ^ 0x60`, where the event count shared bits
/// 24..31 with the step (`(hs ^ 1 << 24, hn ^ 1)` gave the same word) and an event count of 256 or more
/// shifted out. An FNV of the two whole words has neither alias, and it is a digest, which is what
/// `net::agree_word` folds.
pub(crate) fn go_fence_word(hs: u64, hn: usize) -> u32 {
    fnv32(0x60, [hs as u32, (hs >> 32) as u32, hn as u32, (hn as u64 >> 32) as u32].into_iter())
}

/// Boot check (TP=2 only): the pairwise lockstep token names a round's width in five bits, and a shared
/// plain step is as wide as the busy lanes, so `--max-batch` above 31 could reach a round the token
/// cannot name. Refuse at boot instead of aborting the link mid-serve at the first such round. World > 2
/// frames carry whole u32 words and TP=1 has no lockstep: both are unbounded here.
pub(crate) fn tp_lane_width_fits(world: u32, max_batch: usize) -> std::result::Result<(), String> {
    if world == 2 && max_batch > crate::net::AGREE_WIDTH_MAX {
        return Err(format!("--max-batch {max_batch} at TP=2: a plain step shared by more than {} lanes cannot be named in the \
                            pairwise lockstep token (5-bit verify-width field); use --max-batch <= {}",
                           crate::net::AGREE_WIDTH_MAX, crate::net::AGREE_WIDTH_MAX));
    }
    Ok(())
}

/// F2 (owner 2026-10-10: "it should output a message and clearly show the cap at 16 lanes"): the EXL3
/// server serves at most 16 requests at the same time — the PLE prefill kernels cap a plain step at
/// m <= 16 (a 17th lane used to error every in-flight request and go DEAD under TP), the packed
/// verify widths are 4..=16, and the old TP=2 rule capped at `AGREE_WIDTH_MAX`. A `--max-batch`
/// above 16 now BOOTS WITH 16: one plain line at the boot, `velogb10_max_batch` reports 16, the KV
/// is sized for 16, and the head ships 16 to the TP nodes (the same resolver on the same argv).
/// Further requests wait in the queue — `--max-waiting` semantics are unchanged.
/// PREHARD1 [ISSUES 104]: ONE authority — this is `batch::SERVE_MAX_LANES`, the cap the trunk
/// (NVFP4) serve path now shares; the two can no longer diverge.
pub(crate) const EXL3_MAX_LANES: usize = crate::batch::SERVE_MAX_LANES;

/// `--max-batch` as the serve resolves it from an argv: the ONE parser (`build_serve` and the
/// pre-network refusal below), so the two cannot come to read different defaults.
pub(crate) fn max_batch_arg(args: &[String]) -> usize {
    arg(args, "--max-batch").and_then(|s| s.parse().ok()).unwrap_or(8)
}

/// F2: the resolved lane count and the value the argv asked for. Every consumer of `--max-batch`
/// (`build_serve`, the pre-network refusal, the boot line and the metrics) reads the resolved half,
/// so the ranks cannot come to different lane counts.
pub(crate) fn resolved_max_batch(args: &[String]) -> (usize, usize) {
    let raw = max_batch_arg(args);
    (raw.min(EXL3_MAX_LANES), raw)
}

/// S-B9-CF-AGREE16B (W4): `tp_lane_width_fits` from the argv and the world alone, so it can run BEFORE
/// any network step. It used to run only inside `build_serve`, after the pack sync (minutes, on a cold
/// node) and the RDMA attach: a `--max-batch 32` TP=2 launch synced, linked, and was refused then. Call
/// sites, in boot order: the head before discovery and the pack sync (`run_exl3_tp_serve_head`), the
/// node before the link (`exl3_tp_node_prelink`, reported to the head), both ranks before the attach
/// (`run_tp_head_serve` / `run_tp_node_serve`, next to the otel refusal), and `build_serve` as the
/// backstop for a caller that reaches it another way.
pub fn tp_lane_width_refusal(args: &[String], world: u32) -> Result<()> {
    // F2: the 16-lane cap is applied FIRST, so this can no longer refuse anything — 16 always fits
    // the pairwise agree width (`AGREE_WIDTH_MAX` 31). The function stays as the backstop that keeps
    // a future caller that bypasses the cap from booting a width the agree token cannot name.
    let (width, _) = resolved_max_batch(args);
    tp_lane_width_fits(world, width).map_err(|e| anyhow::anyhow!(e))
}

/// The comparison half of `tp_agree_eq`, pure: `peer` is what `net::agree_ext` returned (None = link
/// aborted / peer timeout, Some = the peer's (accept, agree_word) for the same step field).
pub(crate) fn tp_agree_decide(accept: u8, k_verify: u8, hash: u32, peer: Option<(u8, u32)>) -> std::result::Result<(), String> {
    let word = crate::net::agree_word(k_verify, hash);
    match peer {
        None => Err("link aborted or peer timeout".into()),
        Some((pa, pw)) if pa == accept && pw == word => Ok(()),
        Some((pa, pw)) => Err(format!("MISMATCH: this rank (accept {accept}, width {k_verify}, word {word:08x}) vs peer (accept {pa}, \
                                       width {}, word {pw:08x}) [word = width in bits 27..31 | 27-bit digest]", pw >> 27)),
    }
}

fn fnv32(h: u32, words: impl Iterator<Item = u32>) -> u32 {
    let mut h = if h == 0 { 0x811c_9dc5 } else { h };
    for w in words {
        for b in w.to_le_bytes() {
            h ^= b as u32;
            h = h.wrapping_mul(0x0100_0193);
        }
    }
    h
}

/// WP02: a CUDA error that poisons the context — every later call fails too, so the engine is dead
/// (cudarc's DriverError prints the CUresult name; anyhow's {:?} carries the whole chain).
fn sticky_cuda_error(e: &anyhow::Error) -> bool {
    let s = format!("{e:?}");
    ["ILLEGAL_ADDRESS", "LAUNCH_FAILED", "MISALIGNED_ADDRESS", "ILLEGAL_INSTRUCTION", "ECC_UNCORRECTABLE",
     "HARDWARE_STACK_ERROR", "INVALID_PC", "INVALID_ADDRESS_SPACE", "CUDA_ERROR_ASSERT", "LAUNCH_TIMEOUT",
     "CONTEXT_IS_DESTROYED"].iter().any(|t| s.contains(t))
}

/// WP02: the scheduler thread's liveness guard. A panic unwinding out of the scheduler drops its
/// lanes' senders (every in-flight request then ends "error: engine stopped", never "length"),
/// then this marks the engine DEAD; with --exit-on-fatal it exits 70 for the supervisor. An abort
/// (a panic inside a CudaSlice drop, AGENTS §5a) never reaches here: the supervisor restarting on
/// abort is that half.
struct FatalGuard {
    exit_on_fatal: bool,
}

impl Drop for FatalGuard {
    fn drop(&mut self) {
        if std::thread::panicking() {
            crate::server::set_engine_dead("scheduler thread panicked");
            if self.exit_on_fatal {
                eprintln!("[exl3-serve] FATAL: scheduler thread panicked — exit 70 (--exit-on-fatal, default)");
                std::thread::sleep(std::time::Duration::from_millis(300));
                crate::logq::flush_and_exit(std::time::Duration::from_millis(300), 70); // H7
            }
        }
    }
}

/// Port of exllamav3's DraftConfidenceCalibrator (vcruz305 fork 523ecd3,
/// exllamav3/generator/draft_confidence.py): an online map from the draft head's argmax LOGIT
/// to the observed acceptance rate, in 1-logit bins of exponentially decayed (tested, accepted)
/// counts. One calibrator for the whole server, as in the reference implementation (one per generator).
pub(crate) struct DraftCal {
    bins: std::collections::BTreeMap<i64, (f64, f64)>,
    total: f64,
    // WP23: depth-keyed bins [depth bucket] -> logit bin -> (tested, accepted), fed alongside the
    // pooled map (which stays byte-for-byte the pre-WP23 calibrator). None = logit-only (K <= 5).
    dbins: Option<Vec<std::collections::BTreeMap<i64, (f64, f64)>>>,
}

impl DraftCal {
    const BIN_W: f64 = 1.0;
    const DECAY: f64 = 0.995;
    const MIN_COUNT: f64 = 8.0;
    const BURN_IN: f64 = 64.0;
    pub(crate) fn new() -> Self {
        let k = crate::exl3_forward::mtp_depth_default();
        let dbins = if wp23_on(crate::opt!("exl3-dds-depth-bins"), k) {
            Some(vec![Default::default(); WP23_DEPTH_BUCKETS])
        } else {
            None
        };
        Self { bins: Default::default(), total: 0.0, dbins }
    }
    /// `i` = the draft's pass index in its round (0 = the first draft); only the WP23 depth bins
    /// read it — the pooled map is updated exactly as before.
    pub(crate) fn add_label(&mut self, i: usize, score: f32, accepted: bool) {
        let idx = (score as f64 / Self::BIN_W).floor() as i64;
        let b = self.bins.entry(idx).or_insert((0.0, 0.0));
        b.0 += 1.0;
        if accepted { b.1 += 1.0; }
        self.total += 1.0;
        if let Some(d) = self.dbins.as_mut() {
            let b = d[wp23_depth_bucket(i)].entry(idx).or_insert((0.0, 0.0));
            b.0 += 1.0;
            if accepted { b.1 += 1.0; }
        }
    }
    /// Once per verification round, so the map tracks drift (prose vs code) over ~hundreds of rounds.
    pub(crate) fn decay_step(&mut self) {
        for b in self.bins.values_mut() { b.0 *= Self::DECAY; b.1 *= Self::DECAY; }
        self.total *= Self::DECAY;
        if let Some(d) = self.dbins.as_mut() {
            for m in d.iter_mut() {
                for b in m.values_mut() { b.0 *= Self::DECAY; b.1 *= Self::DECAY; }
            }
        }
    }
    /// WP23: acceptance estimate for pass `i`'s draft. Depth bins on: the nearest populated bin
    /// at or below the score in i's depth bucket; a bucket with none there (sparse at deep levels
    /// on prose) falls back to the pooled estimate. Depth bins off: exactly `estimate`.
    pub(crate) fn estimate_at(&self, i: usize, score: f32) -> f64 {
        let Some(d) = self.dbins.as_ref() else { return self.estimate(score) };
        if self.total < Self::BURN_IN { return 1.0; }
        let idx = (score as f64 / Self::BIN_W).floor() as i64;
        let near = d[wp23_depth_bucket(i)].range(..=idx).rev()
            .find(|(_, v)| v.0 >= Self::MIN_COUNT).map(|(_, v)| *v);
        match near { Some((t, a)) => a / t, None => self.estimate(score) }
    }
    /// Estimated conditional acceptance for a draft with this score: the nearest populated bin at
    /// or below it (else the lowest populated); optimistic 1.0 while still learning.
    fn estimate(&self, score: f32) -> f64 {
        if self.total < Self::BURN_IN { return 1.0; }
        let idx = (score as f64 / Self::BIN_W).floor() as i64;
        let mut below: Option<(f64, f64)> = None;
        let mut lowest: Option<(f64, f64)> = None;
        for (&k, &v) in self.bins.iter() {
            if v.0 < Self::MIN_COUNT { continue; }
            if lowest.is_none() { lowest = Some(v); }
            if k <= idx { below = Some(v); } else { break; }
        }
        match below.or(lowest) { Some((t, a)) => a / t, None => 1.0 }
    }
}

// ---- WP23 (PLAN/SURPASS_PLAN_2026-09-26.md): draft depth 6/7 — DDS depth bins + cost guard ----
// Draft-only: the verify decides every emitted token (greedy bitwise at any width <= 16 by batch
// invariance); these only choose how many drafts a round proposes.

/// The pre-WP23 depth ceiling: at a served depth <= this, both WP23 DDS parts default OFF, so
/// --exl3-mtp-k=5 reproduces the K5 build's DDS decisions (the brief's confound check).
const WP23_LEGACY_MAX_K: usize = 5;
/// Depth buckets of the WP23 bins: pass {0} | {1,2} | {3,4} | {5..} — levels 6/7 share a bucket so
/// it fills on code at ~1.5 labels/round (a per-level bucket at 1-logit bins stays under MIN_COUNT).
const WP23_DEPTH_BUCKETS: usize = 4;
fn wp23_depth_bucket(i: usize) -> usize {
    match i { 0 => 0, 1 | 2 => 1, 3 | 4 => 2, _ => 3 }
}

/// WP23 part switch: `name`=0|off / 1|on forces it (diagnostic A/B, never a serving default);
/// unset = on iff the served depth k exceeds the pre-WP23 ceiling.
fn wp23_on(name: crate::opts::OptId, k: usize) -> bool {
    match crate::opts::var(name).as_deref() {
        Ok("0") | Ok("off") => false,
        Ok("1") | Ok("on") => true,
        _ => k > WP23_LEGACY_MAX_K,
    }
}

/// WP23 cost guard switch (--exl3-dds-guard=0|1; default on iff k > 5). k = the scheduler's
/// served depth (one per process), so the env is read once.
pub(crate) fn wp23_guard_on(k: usize) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| wp23_on(crate::opt!("exl3-dds-guard"), k))
}

/// WP23 cost guard: the lane's least-squares slope of round ms on drafts per round (exponentially
/// weighted, this request only), shrunk toward the 6.0 ms/draft prior (PLAN WP23: a row past m=6
/// costs ~6.0-6.3 ms incl. its draft pass). With little spread in drafts/round (code that always
/// drafts to the cap) the prior dominates; DDS-varied rounds (prose, python) identify the slope.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CostFit {
    /// the prior slope ĉ is shrunk toward (ms per draft): 6.0 at TP=1; TP-I #7: dds_prior(tp)
    prior: f64,
    w: f64,
    sx: f64,
    sy: f64,
    sxx: f64,
    sxy: f64,
    n: usize,
}

impl Default for CostFit {
    fn default() -> Self {
        CostFit { prior: Self::PRIOR_SLOPE, w: 0.0, sx: 0.0, sy: 0.0, sxx: 0.0, sxy: 0.0, n: 0 }
    }
}

/// TP-I #7 (T2 lever): the WP23 cost guard's prior ĉ for a TP=2 lane. The 6.0 ms/draft seed was fitted
/// to TP=1 costs; a TP=2 draft costs a verify row at the TP=2 width slope (1.92 ms/row, TP-I re-ledger:
/// r0 union m=4 22.91 -> m=8 30.60 ms/verify) plus one sharded draft pass (0.76 ms) = ~2.7 ms. With
/// code that always drafts to the cap the fit never identifies the slope, so the prior IS the guard.
///   unset / on : 2.7 at TP (world 2) — only lanes of a TP scheduler; TP=1 keeps 6.0 by construction
///   0 / off    : 6.0 (the TP=1 constant, the pre-TP-I behaviour)
///   <ms>       : that prior (diagnostic)
/// --tp-dds-prior rides TpConfig's CLI option-registry snapshot; the value joins the TP boot agree hash. The guard's
/// timing inputs stay the HEAD's (tp_round), so both ranks' targets are identical.
pub(crate) const TP2_DDS_PRIOR: f64 = 2.7;
/// World-4 default: an ESTIMATE, not a measurement — TP-4A (24.7-17.59)/5 = 1.42 ms/row + 0.73 draft = 2.15; see TP-4F_PREP_REPORT.md.
pub(crate) const TP4_DDS_PRIOR: f64 = 2.1;
/// CLI-1 (--print-config): the TP=2 prior this process resolves.
pub fn dds_prior_tp() -> f64 { dds_prior(true) }
/// CLI-1 (--print-config): the TP=4 prior this process resolves.
pub fn dds_prior_tp4() -> f64 { dds_prior_w(4) }
pub(crate) fn dds_prior(tp: bool) -> f64 { dds_prior_w(if tp { 2 } else { 1 }) }
/// The prior for a scheduler of `world` ranks (1 = no TP): 6.0 / 2.7 / 2.1 at world 1 / 2 / 4 (any other world
/// keeps the TP=2 value); `--tp-dds-prior` (off = 6.0, a number = that prior) applies at every TP world.
pub(crate) fn dds_prior_w(world: usize) -> f64 {
    if world < 2 { return CostFit::PRIOR_SLOPE; }
    static RAW: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    resolve_dds_prior(RAW.get_or_init(|| crate::opts::var(crate::opt!("tp-dds-prior")).ok()).as_deref(), world)
}
/// Pure resolution of the prior from the raw `--tp-dds-prior` value (None = unset) at a TP `world` >= 2. The
/// world-2 arms are the pre-TP-4F code verbatim: on/unset/garbage = 2.7, off = 6.0, a positive number = itself.
pub(crate) fn resolve_dds_prior(raw: Option<&str>, world: usize) -> f64 {
    let dflt = if world == 4 { TP4_DDS_PRIOR } else { TP2_DDS_PRIOR };
    match raw.map(|v| v.trim().to_ascii_lowercase()) {
        Some(v) if v == "0" || v == "off" || v == "false" => CostFit::PRIOR_SLOPE,
        Some(v) if v == "1" || v == "on" || v.is_empty() => dflt,
        Some(v) => v.parse::<f64>().ok().filter(|x| *x > 0.0).unwrap_or(dflt),
        None => dflt,
    }
}
/// The prior's word in the TP boot agree hash (`tp_boot_agree`).
pub(crate) fn boot_prior_word(world: usize) -> u32 { (dds_prior_w(world) as f32).to_bits() }

impl Exl3Scheduler {
    /// Ranks of this scheduler's TP group for the world-aware DDS prior: 1 = no TP (an impossible tp-without-model
    /// pairing keeps the pre-TP-4F value, world 2).
    fn tp_world(&self) -> usize {
        if self.tp.is_none() { 1 } else { self.model.tp_rank().map_or(2, |(_, w)| w as usize) }
    }
}

#[cfg(test)]
mod dds_prior_w4_tests {
    use super::*;

    /// The pre-TP-4F `dds_prior(true)` body, verbatim (only the option read is replaced by `raw`): the frozen
    /// reference that world 2 must equal bit for bit.
    fn old_world2(raw: Option<&str>) -> f64 {
        match raw.map(|v| v.trim().to_ascii_lowercase()) {
            Some(v) if v == "0" || v == "off" || v == "false" => CostFit::PRIOR_SLOPE,
            Some(v) if v == "1" || v == "on" || v.is_empty() => TP2_DDS_PRIOR,
            Some(v) => v.parse::<f64>().ok().filter(|x| *x > 0.0).unwrap_or(TP2_DDS_PRIOR),
            None => TP2_DDS_PRIOR,
        }
    }

    const RAWS: [Option<&str>; 16] = [None, Some(""), Some(" "), Some("on"), Some("ON"), Some("1"), Some("off"), Some("OFF"),
        Some("0"), Some("false"), Some("2.7"), Some("3.5"), Some("0.0"), Some("-1"), Some("garbage"), Some(" 1.25 ")];

    #[test]
    fn world_1_and_2_are_bit_identical_to_the_pre_world_aware_prior() {
        assert_eq!(dds_prior(false).to_bits(), CostFit::PRIOR_SLOPE.to_bits());
        assert_eq!(dds_prior_w(1).to_bits(), 6.0f64.to_bits());
        assert_eq!(dds_prior_w(0).to_bits(), 6.0f64.to_bits());
        for raw in RAWS {
            assert_eq!(resolve_dds_prior(raw, 2).to_bits(), old_world2(raw).to_bits(), "raw {raw:?}");
        }
        // unset (the test process has no --tp-dds-prior): the runtime entry points are the old constants
        assert_eq!(dds_prior(true).to_bits(), 2.7f64.to_bits());
        assert_eq!(dds_prior_tp().to_bits(), 2.7f64.to_bits());
        assert_eq!(dds_prior_w(2).to_bits(), 2.7f64.to_bits());
    }

    #[test]
    fn world_2_boot_hash_word_and_hash_are_unchanged() {
        let sample = [128u32, 2048, 7, 1, 512, 0, 262144, 1, 0, 0x3F00_0000, 0, 0, 1, 2, 3];
        let hash = |word: u32| fnv32(0xB007, sample.iter().copied().chain([word]).chain([9, 9, 9]));
        assert_eq!(boot_prior_word(2), (2.7f64 as f32).to_bits(), "the world-2 word");
        assert_eq!(boot_prior_word(2), (dds_prior(true) as f32).to_bits(), "== the pre-change `(dds_prior(true) as f32).to_bits()`");
        assert_eq!(hash(boot_prior_word(2)), hash((dds_prior(true) as f32).to_bits()));
        // every world-2 value the flag can produce hashes exactly as the old code did
        for raw in RAWS {
            assert_eq!(hash((resolve_dds_prior(raw, 2) as f32).to_bits()), hash((old_world2(raw) as f32).to_bits()), "raw {raw:?}");
        }
        // world 4's word differs (its prior is 2.1): ranks must agree among themselves, and do, both resolve the same flag
        assert_eq!(boot_prior_word(4), (2.1f64 as f32).to_bits());
        assert_ne!(hash(boot_prior_word(4)), hash(boot_prior_word(2)));
    }

    #[test]
    fn world_4_default_is_the_derived_value_and_an_override_still_overrides() {
        assert_eq!(TP4_DDS_PRIOR, 2.1);
        assert_eq!(dds_prior_w(4).to_bits(), TP4_DDS_PRIOR.to_bits());
        assert_eq!(dds_prior_tp4().to_bits(), TP4_DDS_PRIOR.to_bits());
        // unset / on / unusable = the world's default (the TP=2 arms' "unusable falls back to default" rule)
        for raw in [None, Some(""), Some("on"), Some("1"), Some("garbage"), Some("-1"), Some("0.0")] {
            assert_eq!(resolve_dds_prior(raw, 4), TP4_DDS_PRIOR, "raw {raw:?}");
        }
        for raw in [Some("off"), Some("0"), Some("false")] { assert_eq!(resolve_dds_prior(raw, 4), 6.0, "raw {raw:?}"); }
        assert_eq!(resolve_dds_prior(Some("2.7"), 4), 2.7, "a numeric override overrides at W=4 (even to the W=2 value)");
        assert_eq!(resolve_dds_prior(Some("3.5"), 4), 3.5);
        assert_eq!(resolve_dds_prior(Some(" 1.25 "), 4), 1.25);
        // the override is world-independent: the same number at world 2 and 4
        for n in ["0.5", "1.9", "4.4"] { assert_eq!(resolve_dds_prior(Some(n), 2), resolve_dds_prior(Some(n), 4)); }
        // any other TP world keeps the TP=2 constant
        assert_eq!(resolve_dds_prior(None, 3), 2.7);
    }
}

impl CostFit {
    pub(crate) fn with_prior(prior: f64) -> Self { CostFit { prior, ..Default::default() } }
    const DECAY: f64 = 0.97;       // ~33-round memory: the context (and a row's cost) drifts slowly
    const PRIOR_SLOPE: f64 = 6.0;  // ms per draft (the brief's seed)
    const PRIOR_W: f64 = 4.0;      // prior weight in (drafts - mean)^2 units
    pub(crate) fn add(&mut self, x: f64, y: f64) {
        let d = Self::DECAY;
        self.w = d * self.w + 1.0;
        self.sx = d * self.sx + x;
        self.sy = d * self.sy + y;
        self.sxx = d * self.sxx + x * x;
        self.sxy = d * self.sxy + x * y;
        self.n += 1;
    }
    /// ĉ (ms per draft), >= 0.
    pub(crate) fn slope(&self) -> f64 {
        let (cxx, cxy) = if self.w > 0.0 {
            ((self.sxx - self.sx * self.sx / self.w).max(0.0), self.sxy - self.sx * self.sy / self.w)
        } else {
            (0.0, 0.0)
        };
        ((cxy + Self::PRIOR_W * self.prior) / (cxx + Self::PRIOR_W)).max(0.0)
    }
}

/// WP23: the DDS target for this round. `base` = --draft-confidence (0.4 served). Once the lane
/// has 16 rounds and its cost EMAs, a draft must be worth its cost: accept probability >= ĉ ms x
/// the lane's tokens/ms — max(base, clamp(ĉ·ema_tok/ema_ms, 0.15, 0.8)). The guard only RAISES
/// the target (long context: rows cost more; fast code rounds: every ms is worth more tokens).
pub(crate) fn wp23_target(base: f64, guard: bool, rounds: usize, c_hat: f64, ema_tok: f64, ema_ms: f64) -> f64 {
    if !guard || rounds < 16 || ema_ms <= 0.0 || ema_tok <= 0.0 {
        return base;
    }
    base.max((c_hat * ema_tok / ema_ms).clamp(0.15, 0.8))
}

/// WP23 per-lane state: the cost fit, the last round's target, rounds by verified draft width.
#[derive(Clone, Copy, Debug, Default)]
struct Wp23Lane {
    fit: CostFit,
    target: f64,
    widths: [usize; crate::exl3_forward::MTP_MAX_K + 1],
}

/// `--spec-lanes-max`: never / a fixed lane threshold / cost-based (default).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum SpecLanes { Never, Over(usize), Auto }

/// The switch position `step_plain` orders its lanes around: qsa_limit() when the pack has a live
/// sparse attention and the per-row regime is on; usize::MAX = no switch (no indexer, --qsa-exl3 0
/// or --qsa-row-regime 0), which makes every order below the identity.
fn regime_switch(qsa_live: bool, row_regime_on: bool, qsa_limit: usize) -> usize {
    if qsa_live && row_regime_on { qsa_limit } else { usize::MAX }
}

/// The scheduler side of the per-row regime (S-B9-REL-QSAREGIME; a free function since
/// S-B9-REL-SMALLFIXES 9a(iv) so the order is testable without a model): `slots` with the lanes
/// at/below the switch `lim` first, each class in its given order. `pos_of(slot)` = the lane's
/// next position.
fn regime_ordered_slots(slots: &[usize], pos_of: impl Fn(usize) -> usize, lim: usize) -> Vec<usize> {
    let pos: Vec<usize> = slots.iter().map(|&s| pos_of(s)).collect();
    crate::exl3_forward::regime_order(&pos, lim).into_iter().map(|i| slots[i]).collect()
}

/// S-B9-REL-SMALLFIXES 9a(iii): the idle-point line of the lazily captured mixed-step graphs
/// (None = nothing new since the last line).
fn mixed_graphs_idle_line(printed: u64, now: u64, lanes: usize) -> Option<String> {
    (now > printed).then(|| format!("[exl3-serve] mixed-step graphs idle: {printed}->{now} captured since boot (lazy, one per step shape \
seen, not capped; {lanes} lanes can form {} mixed step shapes per sparse grid bucket and penalty state)",
                                    crate::exl3_forward::mixed_step_shapes(lanes)))
}

/// S-B9-REL-SMALLFIXES item 5(a): the statistics a scheduler may report for a finishing stop
/// token. A greedy row's token is that row's argmax, so its rank must be 1: any other rank means
/// the logits row read back is not the row that chose the token, and nothing is reported rather
/// than a number about another row. A sampled row has no such check (any rank is possible).
fn stop_info_checked(stats: Option<(f64, u32)>, greedy: bool) -> Option<(f64, u32)> {
    match stats {
        Some((p, rank)) if p.is_finite() && (!greedy || rank == 1) => Some((p, rank)),
        _ => None,
    }
}

/// SURFACE2 item 4: does this boot speculate at all — the pack has a draft head and neither
/// `--mtp off` nor the hatch turned it off? The boot lines about the speculative rounds (their lane
/// order, the shared-step policy, the DDS depth posture, the sampling mode) are printed only then.
fn spec_live(has_head: bool, off: Option<crate::exl3_forward::MtpOff>) -> bool {
    has_head && off.is_none()
}

/// The boot line about the pack's draft head (None = the pack has none). The head is loaded either
/// way (it is part of the pack); with speculation off the line says it is not used.
fn head_loaded_line(has_head: bool, off: Option<crate::exl3_forward::MtpOff>) -> Option<String> {
    if !has_head { return None; }
    Some(match off {
        None => "[exl3-serve] M1 pack head loaded (MTP draft head for chain-verify serving)".to_string(),
        Some(o) => format!("[exl3-serve] M1 pack head loaded (MTP draft head; NOT used on this boot: {})", o.flag()),
    })
}

/// The boot line of the MTP posture (None = the pack has no MTP head: nothing to say).
/// S-B9-REL-SMALLFIXES item 1: `--mtp off` is honoured and says so ('MTP OFF (--mtp off)'); the
/// two OFF lines keep the `[exl3-serve] MTP chain-verify OFF` prefix the gate scripts match.
fn mtp_banner(has_head: bool, off: Option<crate::exl3_forward::MtpOff>, mtp_k: usize, draft_conf: f64) -> Option<String> {
    use crate::exl3_forward::MtpOff;
    if !has_head { return None; }
    Some(match off {
        None => format!("[exl3-serve] MTP chain-verify ON (k={mtp_k}, greedy + sampled lanes, dynamic draft stop {}; --mtp off / --exl3-mtp-k override)",
                        if draft_conf > 0.0 { format!("at {draft_conf}") } else { "OFF".to_string() }),
        Some(MtpOff::MtpFlag) => "[exl3-serve] MTP chain-verify OFF: MTP OFF (--mtp off) — plain decoding only, no draft head, no speculative rounds".to_string(),
        Some(MtpOff::Hatch) => "[exl3-serve] MTP chain-verify OFF (--exl3-no-mtp set — diagnostics hatch)".to_string(),
    })
}

/// CF-FCFS `--lane-order`: the order the SERIAL speculative rounds run in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum LaneOrder {
    /// Today's order, unchanged: every scheduler step gives each busy MTP-capable lane one round,
    /// in slot order. This arm must stay bit-for-bit identical to the pre-flag build.
    Rr,
    /// Run to completion, first come first served: only the front lane (the oldest busy
    /// MTP-capable lane that still has quantum left) steps; it does so until it finishes or
    /// exhausts its quantum, then it rotates to the back.
    Fcfs,
}

impl LaneOrder {
    pub(crate) fn parse(v: Option<&str>) -> Result<LaneOrder> {
        match v {
            None | Some("") | Some("rr") | Some("round-robin") => Ok(LaneOrder::Rr),
            Some("fcfs") => Ok(LaneOrder::Fcfs),
            Some(other) => anyhow::bail!("--lane-order: expected rr|fcfs, got {other:?}"),
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self { LaneOrder::Rr => "rr", LaneOrder::Fcfs => "fcfs" }
    }
}

/// CF-FCFS: the one lane that gets this scheduler step's speculative round, or None when this step
/// runs no speculative round at all.
///
/// PURE FUNCTION of replicated state (AGENTS 2.10, TP SPMD): it reads only `mtp_capable`,
/// `admit_seq` and `fcfs_credit` per lane — no wall clock, no rank-local counter, nothing the
/// head alone sees. Every rank admits the same requests in the same order (the head ships the
/// admits), so every rank's `admit_seq` and token counters are identical and every rank picks the
/// same lane. The rr arm returns None here and the caller keeps today's loop verbatim.
///
/// `lanes` is the busy-lane list in slot order (the caller's `slots`), already split into the
/// lanes this round would speculate (`capable`, MTP on and lane.mtp armed) and the MTP-off lanes
/// that batch separately. Rotation accounting (`fcfs_credit`, the rotation log) is the caller's.
fn fcfs_front<'a>(capable: &[(usize, u64, usize)], quantum: usize) -> Option<usize> {
    // Oldest first (admit_seq is unique per live request: the counter only ever increments).
    let mut order: Vec<&(usize, u64, usize)> = capable.iter().collect();
    order.sort_by_key(|(_, seq, _)| *seq);
    // A lane that is not the front yet starts with a full quantum; only a lane that already had a
    // turn can be over its quantum (its credit is reset when it rotated to the back).
    order.into_iter().find(|(_, _, credit)| *credit < quantum).map(|(slot, _, _)| *slot)
}

/// CF-FCFSB: charge the front lane for the tokens it just generated and rotate it to the BACK of
/// the queue when it used its quantum. PURE function of replicated state: on rotation the lane is
/// re-stamped with the next admission sequence number (`*admit_ctr`, then the counter advances) so
/// `fcfs_front` — which orders by `admit_seq` — now sees it as the youngest lane. Without the
/// re-stamp the rotated lane keeps the lowest `admit_seq` and is picked again on the very next
/// step: the "rotation" was a log line and fcfs degenerated to run-to-completion by admission
/// order. Returns true when the lane rotated. `rr` never calls this.
fn fcfs_charge(seq: &mut u64, credit: &mut usize, delta: usize, quantum: usize, admit_ctr: &mut u64) -> bool {
    *credit += delta;
    if *credit >= quantum {
        *credit = 0;
        *seq = *admit_ctr;
        *admit_ctr += 1;
        true
    } else {
        false
    }
}

impl SpecLanes {
    pub(crate) fn parse(v: Option<&str>) -> Result<SpecLanes> {
        match v.map(str::trim) {
            None | Some("auto") | Some("") => Ok(SpecLanes::Auto),
            Some("0") | Some("never") | Some("off") => Ok(SpecLanes::Never),
            Some(n) => n.parse::<usize>().map(SpecLanes::Over)
                .map_err(|_| anyhow::anyhow!("--spec-lanes-max must be auto, 0 or a lane count (got '{n}')")),
        }
    }
}

/// Model of one shared plain step over n lanes (ms, TP=1 code class, fitted on .14 2026-10-02: n=4 46.5, n=8 61.7,
/// n=16 90): the learned `shared_scale` corrects it for the topology / content actually served.
fn shared_step_prior_ms(n: usize) -> f64 { 31.5 + 3.6 * n as f64 }

/// F4 (S-B9-REL-FINALFIX, from PACK5G finding 1): under `--lane-pack on` the shared arm of the
/// `use_shared_plain` comparison is priced as `shared_scale * shared_step_prior_ms(n)`, but the
/// scale is only LEARNED by running shared rounds (`learn_shared_ms`) — and with the scale
/// unlearned the prior never wins at c=4, so an on-boot rides packed pairs for the whole request
/// (19/20 PACK5G boots) while the shared step measured FASTER in 3 of the 4 standard cells (only
/// code/thinking-off pays for pairs). The probe breaks the loop: one shared plain step early,
/// then periodically, so both arms are priced from measurement and the comparison picks the
/// cheaper of the two. Cadence counts rounds on the OTHER arm:
const SHARED_PROBE_FIRST: u32 = 16;  // while the scale has never been learned (boot start)
const SHARED_PROBE_EVERY: u32 = 128;
/// P18 SPEC8K: weight of a new sample in the round-to-round WALL latency EMA of a lane
/// (Lane::ema_wall_ms). The second round of a lane already measures the whole cycle under load, so
/// this only smooths it.
const WALL_EMA_W: f64 = 0.75;
/// P18 SPEC8K: a wall gap past this multiple of the round it measured is a stall (a prefill on
/// another lane, a TP seam, the idle gap between two requests), not the lateness of this lane.
const WALL_STALL_X: f64 = 6.0;

/// P18 SPEC8K: the ms that ONE lane contributes to the cycle time of the serial arm: the time from
/// one round of the lane to its NEXT round, which is what its client waits, NOT the ms of the round
/// itself. A packed round charges each lane one SHARE of the verify (lane_pack::lane_share_ms), so
/// summing ema_ms over n lanes prices n rounds as if they all ran at once; the shared-batching arm
/// then looks slower than a cycle that in truth is about n times longer (spec-lanes auto picked
/// serial speculation for four busy lanes at 8K prose, where the shared plain step measured faster).
/// ema_wall_ms holds the measured queueing, so this is the whole cycle for one lane; for a lone lane
/// it equals ema_ms.
// S-C4XOVER arm indices: 0 = the shared plain step arm, 1 = the serial speculation arm
// (packed pairs where two lanes pack); they index the measured-rate arrays and the
// /metrics arm labels (SPEC_LANES_TO), and pick as usize maps true->0 / false->1.
pub(crate) const ARM_SHARED: usize = 0;
pub(crate) const ARM_SERIAL: usize = 1;
// S-C4XOVER cadences: a measured arm rate is FRESH for 512 scheduler steps at a busy
// count within +-1 of now; the window folds every 8 accumulated rounds; the idle arm is
// probed (a 4-step window of it) when its rate has gone stale, at most every 256
// riding-arm rounds (16 while it has never been measured, so a boot converges in ~2 s).
const SPEC_ARM_FRESH_STEPS: u64 = 512;
const SPEC_ARM_N_TOL: i64 = 1;
const SPEC_ARM_FOLD_EVERY: u32 = 8;
const SPEC_ARM_PROBE_FIRST: u32 = 16;
const SPEC_ARM_PROBE_EVERY: u32 = 256;
const SPEC_ARM_PROBE_WIN: u32 = 4;
pub(crate) fn serial_cycle_ms(ema_ms: f64, ema_wall_ms: f64) -> f64 {
    if ema_wall_ms > 0.0 { ema_wall_ms } else { ema_ms }
} // (was the note on the SHARED_PROBE_EVERY line)
fn shared_probe_due(ctr: u32, scale_unlearned: bool) -> bool {
    ctr >= if scale_unlearned { SHARED_PROBE_FIRST } else { SHARED_PROBE_EVERY }
}

/// Generated token ids a request keeps for its `[mtp-stats] toks=` field (--round-trace).
const FIRST_TOKS: usize = 8;

/// `--round-trace` in the EXL3 server (S-B9-CF-PACK4S): `[mtp-stats]` also prints the request's
/// seed, its per-round arm trace and its first generated token ids. Diagnostics; no effect on any round.
fn round_trace_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::opts::var(crate::opt!("round-trace")).is_ok())
}

impl Lane {
    fn log_stats(&self, slot: usize, reason: &str) {
        if self.st_rounds == 0 && self.st_plain == 0 { return; }
        let r = self.st_rounds.max(1) as f64;
        crate::tel::add_spec(self.st_rounds as u64, self.st_drafted as u64, self.st_accepted as u64);
        crate::tel::add_mtp_reprobe(self.reprobe.off_events as u64, self.reprobe.reprobes as u64,
                                    self.reprobe.reprobe_ok as u64);
        self.log_wp23(slot);
        // S-B9: off_events/reprobes/reprobe_ok APPENDED (existing key order is frozen —
        // scripts parse it).
        // S-B9-CF-PACK4S: packed= APPENDED (lane rounds a packed verify served, of `rounds`); with
        // --round-trace also the request's seed and its per-round arm trace (lane_pack::ArmTrace:
        // the provenance of each emitted token — the sampled-lane distribution gate reads it).
        let trace = if round_trace_on() {
            format!(" seed={} arms={} toks={}", self.seed, self.arms.text(),
                    self.first_toks.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","))
        } else { String::new() };
        crate::dreprintln!("[mtp-stats] slot={slot} finish={reason} gen={} rounds={} drafted={} accepted={} ({:.1}%) \
                   tok/round={:.2} ms/round={:.1} plain_steps={} temp={} ctrl={} off_events={} reprobes={} reprobe_ok={} packed={}{trace}",
                  self.generated, self.st_rounds, self.st_drafted, self.st_accepted,
                  100.0 * self.st_accepted as f64 / self.st_drafted.max(1) as f64,
                  (self.st_accepted as f64 + self.st_rounds as f64) / r, self.st_ms / r,
                  self.st_plain, self.temperature, self.st_ctrl,
                  self.reprobe.off_events, self.reprobe.reprobes, self.reprobe.reprobe_ok, self.arms.packed);
    }

    /// WP02: the lane's client is gone (receiver dropped: disconnect, or a stop-string hit on the
    /// HTTP side) — its slot is freed now; always leaves a [mtp-stats] finish=cancelled line.
    fn log_cancel(&self, slot: usize) {
        if self.st_rounds == 0 && self.st_plain == 0 {
            crate::dreprintln!("[mtp-stats] slot={slot} finish=cancelled gen={} rounds=0 plain_steps=0 temp={}",
                      self.generated, self.temperature);
        } else {
            self.log_stats(slot, "cancelled");
        }
    }

    /// WP23: one line per finished request that ran MTP rounds — rounds by verified draft width
    /// (widths[w] = rounds that verified w drafts; w=6/7 are the new depths), the cost fit ĉ and
    /// the last round's DDS target. Printed just before [mtp-stats].
    fn log_wp23(&self, slot: usize) {
        if self.st_rounds == 0 { return; }
        let g = &self.wp23;
        crate::dreprintln!("[dds] slot={slot} widths={:?} c_hat={:.2} ms/draft (fit rounds {}) target={:.3}",
                  g.widths, g.fit.slope(), g.fit.n, g.target);
    }

    /// Sampler params for `n` consecutive rows starting at the lane's counter (None = greedy).
    fn samp_rows(&self, n: usize) -> Vec<Option<crate::exl3_forward::SampParams>> {
        (0..n).map(|i| crate::exl3_forward::SampParams::sampled(
            self.temperature, self.top_p, self.top_k, self.min_p, self.seed,
            self.ctr.wrapping_add(i as u32))).collect()
    }

    /// WP08: feed one emitted token (one that did not already end the request) to the loop
    /// detector — the reference implementation's job.py:1010-1013 order. True = end the response ("loop_detected").
    fn loop_feed(&mut self, t: u32) -> bool {
        self.loop_det.as_mut().map_or(false, |d| d.feed(t))
    }

    /// WP08: one line per detection, with what a replay needs (a loop can be an engine bug:
    /// replay greedy + plain before blaming the model).
    fn log_loop(&self, slot: usize, tok: &QwenTokenizer) {
        let Some(d) = self.loop_det.as_ref() else { return };
        let tail_ids = &self.history[self.history.len().saturating_sub(96)..];
        let text = tok.decode(tail_ids, false).unwrap_or_default();
        let n = text.chars().count();
        let tail: String = text.chars().skip(n.saturating_sub(200)).collect();
        crate::dreprintln!("[loop] detected slot={slot} period={} (detector {}) gen={} pos={} seed={} temp={} \
                   top_p={} top_k={} min_p={} pen={:?} mtp={} prompt_hash={:016x} tail={:?}",
                  d.fundamental_period().map_or("-".into(), |p| p.to_string()),
                  d.period().map_or("-".into(), |p| p.to_string()),
                  self.generated, self.pos, self.seed, self.temperature, self.top_p, self.top_k,
                  self.min_p, self.pen, self.mtp.is_some(), self.prompt_hash, tail);
    }
}

/// `--prefill-interleave` value from the command line: absent / empty = 0 (off).
fn parse_prefill_interleave(v: Option<&str>) -> Result<usize> {
    match v.map(str::trim).filter(|v| !v.is_empty()) {
        None => Ok(0),
        Some(v) => v.parse::<usize>().map_err(|e| anyhow::anyhow!("--prefill-interleave: {e} (decode steps per prefill chunk, 0 = off)")),
    }
}

/// The interleave the scheduler runs with, and the boot line that says so (None = flag off, no
/// line: default boot logs are unchanged). PERFFIX1: under TP too — every rank reads the value from
/// the head's shipped registry (`prefill-interleave` is SPMD, in the boot agree's digest) and runs the
/// SAME steps at the SAME chunk boundaries (the boundary is the per-chunk HeadFlag exchange; a step
/// is lockstep work on every rank, its cancels come from the head's outer loop), so no rank steps alone.
fn resolve_prefill_interleave(n: usize, tp: bool, chunk: usize) -> (usize, Option<String>) {
    match (n, tp) {
        (0, _) => (0, None),
        (n, true) => (n, Some(format!("[exl3-serve] prefill interleave ON under TP (--prefill-interleave {n}): every rank runs up to {n} \
                                       scheduler step(s) for the lanes already decoding after each non-final prefill chunk (chunk {chunk} \
                                       tokens), in lockstep; the chunk grid and the joining request's bytes are unchanged, its first token is later"))),
        (n, false) => (n, Some(format!("[exl3-serve] prefill interleave ON (--prefill-interleave {n}): up to {n} scheduler step(s) for the \
                                        lanes already decoding after each non-final prefill chunk (chunk {chunk} tokens); \
                                        the chunk grid and the joining request's bytes are unchanged, its first token is later"))),
    }
}

/// Public issue #24: one lane's (positions used, positions cached) as /metrics reports them.
/// A running request uses `pos` positions (prompt + generated) and a prefill in flight the rows it
/// has written (a resumed prefix included); neither reads as cached, because the prefix snapshot is
/// part of what the request holds. A free lane uses nothing and keeps its snapshot's positions.
fn kv_lane_occupancy(active_pos: Option<usize>, prefill_pos: Option<usize>, cache_len: Option<usize>) -> (usize, usize) {
    match active_pos.or(prefill_pos) {
        Some(p) => (p, 0),
        None => (0, cache_len.unwrap_or(0)),
    }
}

/// Does the hook run after the chunk that ended at `end` of a prefill to `n`? Never after the final
/// chunk (the seam step and the head prime follow it directly), never for a non-cancellable
/// internal prefill (`cancellable` false: the WP16 XCHECK's reference run). Every topology (PERFFIX1).
fn interleave_point(interleave: usize, cancellable: bool, end: usize, n: usize) -> bool {
    interleave > 0 && cancellable && end < n
}

/// C1 v2: prefill scratch rows — the merged-chunk cap with tail checkpoints on (2C - 1; under TP
/// min(2C - 1, 4,095) so `--prefill-chunk 4095` fits the all-reduce partial buffer, TP-4X1), else C.
fn psc_rows(c: usize, tp: bool) -> usize {
    if crate::exl3_forward::tail_ckpt_on() && !crate::exl3_forward::wp16_off() {
        crate::exl3_forward::merge_cap(c, tp).max(c)
    } else { c }
}

/// FNV-1a over the prompt ids — a replay key for the [loop] log (not a security hash).
fn prompt_hash(p: &[u32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &t in p {
        for b in t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

impl Exl3Scheduler {
    /// Prefill one lane's prompt into its slot, chunked (S-A3-f-b): C tokens per
    /// forward sweep through the wide-M path. The final sweep's last token is
    /// NOT processed here — see the S-A3-h seam contract below.
    /// NOTE: the sequential path primed the M1 draft head per prompt token; the
    /// chunked prefill skips that work entirely (pure TTFT win) and the head is
    /// primed once after the seam step (PLAN/S_A3_F_B_CHUNKED_PREFILL.md).
    /// S-A3-h item 1 seam contract (bench byte-parity): chunked prefill covers
    /// prompt[..plen-1] ONLY — the LAST prompt token runs through `forward_step`,
    /// the graphed/fused decode step, in `admit` (the exact bench `--prefill-chunk`
    /// contract, exl3_forward::run). The f-g prefill row and the f-f decode row for
    /// the same token differ in low bits (N.6 class): using the prefill row at the
    /// prompt/decode seam broke bench-vs-serve byte parity AND primed the M1 tap
    /// from a drifted hidden (bug5's self-consistent-but-wrong chains). One
    /// composition per position on both sides, by construction.
    /// WP02: returns Ok(false) when the client went away between chunks (no Finish is sent; the
    /// caller leaves cache[slot] = None, so the partial slot state is never reused). `tx` None =
    /// not cancellable (the WP16 XCHECK's fresh reference prefill).
    /// WP16: `aligned` = this run's chunk grid starts at 0 or at an aligned checkpoint (the caller
    /// did CkptStore::begin_aligned): its chunks extend the slot's checkpointed prefix and the
    /// chunk ends `wp16_due` picks become checkpoints. Never set for a prompt-end prefix-hit resume.
    /// C1 (`tail` Some only with prefix.tail_ckpt on): the grid is `run_grid(from, n, C, realign,
    /// tail)` — realigned to the C grid and split at the message boundary `tail`, which becomes a
    /// checkpoint of this (tracked) run. Knob off: realign = false, tail = None = today's grid.
    /// `key`: the prompt as the prefix cache / WP16 checkpoints see it (VIS-2: image rows carry
    /// pixel-hash ids so two different same-size images never share a cached prefix); == `prompt`
    /// for text.
    fn prefill_range(&mut self, slot: usize, prompt: &[u32], key: &[u32], from: usize, to: usize,
                     tx: Option<&crate::server::TokTx>, aligned: bool,
                     tail: Option<usize>, realign: bool) -> Result<bool> {
        let tp = std::time::Instant::now();
        let c = self.chunk.max(1);
        if self.psc.is_none() {
            self.psc = Some(self.model.prefill_scratch(psc_rows(c, self.tp.is_some()))?);
        }
        let mut pos = from;
        let n = to.min(prompt.len().saturating_sub(1)); // the seam token is a decode step
        if tx.is_some() {
            // issue #24: the request owns its slot from here; a resumed prefix (rows 0..from) counts
            self.kv_pf = Some((slot, from));
            self.publish_kv();
        }
        if pos >= n { return Ok(true); }
        let mut taken: Vec<usize> = Vec::new();
        let (mut il_points, mut il_steps, mut il_ms) = (0usize, 0usize, 0.0f64);
        // C1 v2 (knob on == realign): merge away the chunk the split / realign would add (psc holds
        // 2C - 1 rows then, see run())
        let absorb = crate::exl3_forward::absorb_rmin();
        let grid = crate::exl3_forward::plan_grid(from, n, c, realign, tail, psc_rows(c, self.tp.is_some()), absorb, aligned);
        let nchunks = grid.len();
        if absorb.is_some() || c > 2048 {
            // TP-4X1 receipts: the grid actually run (only with the new options on; default logs unchanged)
            let mut rows = Vec::with_capacity(grid.len());
            let mut s = from;
            for &e in &grid { rows.push(e - s); s = e; }
            let shown = if rows.len() > 16 {
                format!("{:?} .. {:?}", &rows[..4], &rows[rows.len() - 6..])
            } else { format!("{rows:?}") };
            crate::drprintln!("[exl3-serve] prefill grid slot={slot} n={} (pos {from}..{n}) C={c}: {nchunks} chunk(s), rows {shown}{}",
                     n - from, if absorb.is_some() { " (absorb-tail)" } else { "" });
        }
        for end in grid {
            // (re-borrowed per chunk: the interleave hook at the end of the body needs all of self)
            let psc = self.psc.as_mut().unwrap();
            if pos > from {
                let mut gone = tx.map_or(false, |t| t.is_closed());
                // TP-D: the HEAD decides (its client); every rank stops at this same chunk boundary
                if let (Some(t), true) = (self.tp.as_mut(), tx.is_some()) {
                    gone = t.head_flag(gone)?;
                }
                if gone {
                    crate::drprintln!("[exl3-serve] prefill slot={slot} cancelled at pos {pos} of {n} (client gone)");
                    return Ok(false);
                }
            }
            let toks: Vec<i32> = prompt[pos..end].iter().map(|&t| t as i32).collect();
            self.model.prefill_chunk(psc, &toks, pos, slot, false, None)?;
            crate::metrics::sched_touch(); // H3: prefill chunk COMPLETE (long prefill stays live)
            // ISSUES[94] DF2: prime the draft ring with this chunk's window — the prime sink
            // captured the backbone hiddens of rows pos..end during the sweep that just
            // finished (cols = what this chunk actually ran), so the ring is current for the
            // lane's first speculative round after the prefill.
            if let Some(round) = self.df2.as_mut() {
                self.model.df2_prime_window(round, pos, end - pos)?;
            }
            if aligned {
                if let Some(st) = self.wp16.as_mut() {
                    st.extend(slot, &key[pos..end]);
                    if crate::exl3_forward::wp16_due(end, n, c) || tail == Some(end) {
                        let model = &self.model;
                        // TP-4D (audit C2): at world > 2 the allocation outcome is agreed across ranks (a rank-local
                        // failure would desynchronise the checkpoint stores); world <= 2: the plain local alloc.
                        let n_before = st.n_alloc;
                        let mut tp_alloc = self.tp.as_mut().filter(|t| t.world > 2);
                        let mut hub_err: Option<anyhow::Error> = None;
                        let got = st.acquire(|| {
                            let r = model.ckpt_alloc();
                            match tp_alloc.as_mut() { Some(t) => t.wp16_alloc(n_before, r, &mut hub_err), None => r }
                        });
                        if let Some(e) = hub_err { return Err(e); }
                        if let Some(mut b) = got {
                            match model.ckpt_save(slot, psc, &mut b) {
                                Ok(()) => { st.insert(slot, end, b); taken.push(end); }
                                Err(e) => { st.release(b); return Err(e); }
                            }
                        }
                    }
                }
            }
            pos = end;
            if tx.is_some() {
                // issue #24: a request's prefill holds rows 0..end of its slot
                self.kv_pf = Some((slot, end));
                self.publish_kv();
            }
            // --prefill-interleave: this chunk's rows, checkpoint and head-fill carry are complete
            // (interleave_steps drains the device first), so the lanes that were decoding before
            // this prompt arrived get their steps now.
            if interleave_point(self.interleave, tx.is_some(), end, n) {
                let t_il = std::time::Instant::now();
                let ran = self.interleave_steps(slot)?;
                il_points += usize::from(ran > 0);
                il_steps += ran;
                if ran > 0 { il_ms += t_il.elapsed().as_secs_f64() * 1e3; }
            }
        }
        if il_steps > 0 {
            crate::metrics::prefill_interleave(il_points as u64, il_steps as u64);
            crate::drprintln!("[exl3-serve] prefill-interleave slot={slot}: {il_steps} scheduler step(s) at {il_points} of {} chunk boundaries, \
                              {il_ms:.1} ms of decode (not counted in the prefill line's ms / tok/s)", nchunks.saturating_sub(1));
        }
        // S-A3-f-d Item 2: serve-path prefill telemetry (TTFT's main term). With --prefill-interleave
        // the decode steps run between the chunks are taken out, so the line stays the prefill's own rate.
        let ms = tp.elapsed().as_secs_f64() * 1e3 - il_ms;
        let done = n - from;
        crate::drprintln!("[exl3-serve] prefill slot={} n={} (pos {}..{}) in {:.1} ms ({:.0} tok/s, C={})",
                 slot, done, from, n, ms, if ms > 0.0 { done as f64 * 1e3 / ms } else { 0.0 }, c);
        if let Some(t) = tail {
            crate::drprintln!("[exl3-serve] C1 tail slot={slot}: grid split + checkpoint at {t} (message boundary - {}; {nchunks} chunk(s){})",
                     crate::exl3_forward::TAIL_BACKOFF,
                     if realign && from % c != 0 { ", realigned to the C grid" } else { "" });
        } else if realign && from % c != 0 {
            crate::drprintln!("[exl3-serve] C1 slot={slot}: run from {from} realigned to the C grid ({nchunks} chunk(s))");
        }
        if !taken.is_empty() {
            if let Some(st) = self.wp16.as_ref() {
                let per = self.model.ckpt_layout().bytes() as f64;
                crate::drprintln!("[exl3-serve] WP16 slot={slot}: checkpoints taken at {taken:?}; slot holds {:?}; \
                          live {} x {:.0} MB = {:.2} GiB (allocated {} of cap {})",
                         st.positions(slot), st.live(), per / 1e6, st.live() as f64 * per / (1u64 << 30) as f64,
                         st.n_alloc, st.cap);
            }
        }
        Ok(true)
    }

    /// `--prefill-interleave N` (public PR #23): between two prefill chunks of the prompt joining
    /// `slot`, run up to N scheduler steps for the lanes that are already decoding, so a long
    /// prompt no longer freezes every running stream for its whole prefill. Returns the steps run.
    ///
    /// Why the nested `step()` cannot see or disturb the prefill in flight:
    /// - `lanes[slot]` is None until `admit` finishes, and `step()` (cancel sweep, re-probe arming,
    ///   shared-plain choice, fcfs front, every round) only walks the lanes that are Some. Nothing
    ///   in a step admits a request, so the joiner's slot cannot be handed to anyone else; a lane
    ///   that finishes or is cancelled inside a step frees a DIFFERENT index (it was Some when this
    ///   slot was picked) and `run` admits the next waiting request there after this admit returns.
    /// - the prefill carries its state across chunks in `self.psc` (head-fill carry, WP05 stage,
    ///   image splice) and in `slot`'s own rows of the model (GDN / conv state, PLE ring, KV and
    ///   QSA planes, head KV). A step works in `self.sc` and the stepped lanes' slots; it never
    ///   reads `self.psc`, `self.cache` or `self.wp16` (those are touched by admit / prefill only).
    /// - `admit` stages the joiner's sampler, penalties and seam step in `self.sc` only AFTER the
    ///   prefill returns, so nothing a nested step stages there is read by the joiner.
    /// A failing step fails every live lane (`fail_step`) and the error ends this prefill too.
    fn interleave_steps(&mut self, slot: usize) -> Result<usize> {
        anyhow::ensure!(self.lanes[slot].is_none(), "prefill-interleave: slot {slot} already holds a lane");
        if self.fatal.is_some() || self.lanes.iter().all(|l| l.is_none()) { return Ok(0); }
        // prefill_chunk ends synced, but a checkpoint taken at this chunk end queued its copies
        // after that: drain them so the first nested round's measured time (lane EMAs, the cost
        // guard, the shared-step scale) is the round's own.
        self.model.dev().synchronize()?;
        let mut ran = 0usize;
        // S-B9-CF-PACK4: a prompt is mid-prefill for these steps — the decoding lanes do not pack
        self.in_prefill = true;
        for _ in 0..self.interleave {
            if self.fatal.is_some() || self.lanes.iter().all(|l| l.is_none()) { break; }
            if let Err(e) = self.step() {
                self.in_prefill = false;
                self.fail_step(&e);
                // The joiner fails too (the PR's choice: after a failed step nothing is admitted on
                // top of it). Its client sees `{e}` of this error, so the root cause goes in the text.
                anyhow::bail!("a decode step between prefill chunks failed (--prefill-interleave): {e:#}");
            }
            ran += 1;
        }
        self.in_prefill = false;
        if let Some(f) = self.fatal.as_ref() {
            anyhow::bail!("engine stopped during a prefill-interleave step: {f}");
        }
        Ok(ran)
    }

    /// A failed decode step (from `run`, or from a prefill-interleave step): fail every live lane
    /// loudly (never emit garbage as if it were text) and mark the engine dead on a sticky CUDA error.
    fn fail_step(&mut self, e: &anyhow::Error) {
        eprintln!("[exl3-serve] decode step failed: {e:#}");
        for s in 0..self.lanes.len() {
            if let Some(lane) = self.take_lane(s, false) {
                let _ = lane.tx.send(TokEvent::Finish { reason: format!("error: {e}") });
            }
        }
        if sticky_cuda_error(e) {
            self.fatal = Some(format!("{e:#}"));
        }
    }

    /// Prefill prompt[..plen-1] into `slot`, resuming from a cached prefix when possible, and
    /// snapshot the recurrent state at the END of the chunked prefill (prompt[..plen-1]; the last
    /// token is the seam decode step). No extra split: a fresh prompt prefills bit-identically to
    /// --prefix-cache off (a message-boundary split cost a ~175 ms small chunk and moved chunk
    /// boundaries). The next chat turn re-renders this prompt verbatim up to plen-1 in both
    /// thinking modes (verified via /v1/tokenize: thinking on differs only at the final token).
    /// WP16: when the prompt-end snapshot does not match (a new conversation on a shared system
    /// prompt, a new question on the same document), resume from the deepest intermediate
    /// checkpoint <= min(LCP(prompt, the checkpointed prefix), plen-1) instead of prefilling from 0
    /// — the same chunk grid as a fresh prefill, so bit-identical to it. The prompt-end hit path
    /// is unchanged (whenever it matches it is the deeper resume and it still wins).
    /// C1 (prefix.tail_ckpt on): `ckpt_at` (the message boundary: this prompt without its
    /// generation prompt, server.rs) becomes a tail checkpoint of this run — the resume point of
    /// the next chat turn, which diverges right there — and every run is tracked (rows extend the
    /// checkpointed prefix) with its grid realigned to C. See wp16.rs T_PREFIX_TAIL_CKPT.
    fn prefill_cached(&mut self, slot: usize, prompt: &[u32], key: &[u32], ckpt_at: Option<usize>,
                      tx: &crate::server::TokTx) -> Result<bool> {
        anyhow::ensure!(key.len() == prompt.len(), "prefix key length {} != prompt length {}", key.len(), prompt.len());
        let plen = prompt.len();
        let end = plen.saturating_sub(1);
        self.last_resume = (0, 0);
        if !self.prefix_on || !self.model.prefix_snapshots() {
            // PFX1 (b): the chunked prefill below head-fills rows 0..plen-2 (no head-KV memset)
            self.model.reset_slot_for_prefill(slot)?;
            self.cache[slot] = None;
            return self.prefill_range(slot, prompt, key, 0, end, Some(tx), false, None, false);
        }
        let tail_on = self.wp16.is_some() && crate::exl3_forward::tail_ckpt_on();
        // LANERES2: this slot is being taken over by a different conversation when its cache
        // does not prefix the incoming prompt — capture the victim's snapshot to the SSD spill
        // tier BEFORE anything below resets/overwrites it. The D2H is the only in-round cost
        // (~0.1 s per 0.5-GiB snapshot); the disk write is the async part.
        if self.spill.is_some() {
            if let Some(old) = self.cache[slot].clone() {
                let own = old.len() <= end && key[..old.len()] == old[..];
                if !own {
                    self.spill_capture_slot(slot, &old)?;
                }
            }
        }
        let mut reuse = match &self.cache[slot] {
            Some(t) if t.len() <= end && key[..t.len()] == t[..] => t.len(),
            _ => 0,
        };
        // LANERES2: own-snapshot miss — try the SSD spill tier; a hit restores rows 0..n + the
        // snapshot state and the normal `reuse > 0` branch below runs unchanged. Cost-model
        // gated; every miss/drop/error falls back to the plain prefill and is counted.
        if reuse == 0 && self.spill.is_some() {
            if let Some(n) = self.spill_restore_slot(slot, key, end)? {
                reuse = n;
            }
        }
        if self.psc.is_none() {
            self.psc = Some(self.model.prefill_scratch(psc_rows(self.chunk.max(1), self.tp.is_some()))?);
        }
        let c = self.chunk.max(1);
        // WP16: only on a prompt-end miss (a hit is always the deeper, unchanged resume).
        let (q, lcp) = match (&self.wp16, reuse) {
            (Some(st), 0) => st.resume_point(slot, key, end, c),
            _ => (0, 0),
        };
        let mut tracked = reuse == 0 && self.wp16.is_some();
        if reuse > 0 {
            self.model.restore_slot(slot, self.psc.as_mut().unwrap())?;
            crate::drprintln!("[exl3-serve] prefix-cache hit slot={slot}: reuse {reuse} of {plen} prompt tokens");
            // an unaligned grid from `reuse`: no checkpoint is taken in it, none above it survives.
            // C1: with tail checkpoints the run is tracked instead (realigned grid, tail split) when
            // the slot's checkpointed prefix is exactly prompt[..reuse].
            if let Some(st) = self.wp16.as_mut() {
                if tail_on && st.prefix_ok(slot, key, reuse, c) {
                    st.begin_aligned(slot, reuse, c);
                    tracked = true;
                } else {
                    st.begin_unaligned(slot, reuse);
                }
            }
        } else if q > 0 {
            let st = self.wp16.as_mut().unwrap();
            let dropped = st.begin_aligned(slot, q, c);
            st.touch(slot, q);
            let ck = st.get(slot, q).context("WP16: resume checkpoint vanished")?;
            // replaces reset_slot: recurrent state + PLE ring/hist + head-fill carry at q; every
            // positional row < q (trunk/head KV, QSA planes) is the checkpointed prefix's own
            self.model.ckpt_restore(slot, self.psc.as_mut().unwrap(), ck)?;
            self.model.ckpt_poison_head_kv_from(slot, q)?; // PFX1 (b) poison gate only (no-op otherwise)
            crate::drprintln!("[exl3-serve] WP16 checkpoint resume slot={slot}: resume at {q} of {plen} prompt tokens \
                      (LCP {lcp}, C={c}; {dropped} checkpoint(s) above dropped) — prefill {q}..{end}");
        } else {
            self.model.reset_slot_for_prefill(slot)?; // PFX1 (b): prefill_range from 0 head-fills
            if let Some(st) = self.wp16.as_mut() { st.begin_aligned(slot, 0, c); }
        }
        self.cache[slot] = None;
        self.last_resume = (reuse, q);
        let from = reuse.max(q);
        // v0.7.3 (#8.2): report the cache-served prefix (prefill skipped) to the request
        // handler ONCE, before the first token: usage.prompt_tokens_details.cached_tokens
        // and the /metrics counter. from == 0 (fresh prefill) stays silent — the
        // handler's default is 0, so engines without a prefix cache report 0.
        if from > 0 {
            let _ = tx.send(TokEvent::Admitted { cached_tokens: from as u32 });
        }
        let tail = if tail_on && tracked { crate::exl3_forward::tail_point(ckpt_at, from, end, c, crate::exl3_forward::tail_ckpt_extra_chunk()) } else { None };
        match self.prefill_range(slot, prompt, key, from, end, Some(tx), tracked, tail, tail_on && tracked) {
            Ok(true) => {}
            // WP02: cancelled mid-prefill — no snapshot, cache stays None. WP16: the checkpoints
            // taken so far stay (they describe prompt[..pos], rows this run did write).
            Ok(false) => return Ok(false),
            Err(e) => {
                if let Some(st) = self.wp16.as_mut() { st.clear_slot(slot); } // slot state unknown
                return Err(e);
            }
        }
        if q > 0 && self.wp16_xcheck && key == prompt { // VIS-2: the xcheck's scratch prefill is text-only
            self.wp16_xcheck(slot, prompt, q)?;
        }
        self.model.snapshot_slot(slot, self.psc.as_ref().unwrap())?;
        self.cache[slot] = Some(key[..end].to_vec());
        Ok(true)
    }

    /// --wp16-xcheck=1 (diagnostic): after a checkpoint resume of `slot` (state R live), prefill
    /// the same prompt FRESH from 0 into a scratch slot — a free lane's slot, or (--max-batch 1)
    /// the resumed slot itself — and bit-compare GDN S / conv / PLE ring / PLE hist / head-fill
    /// carry and the seam step's first-token logits. The resumed state is then restored, so the
    /// served output stays the resumed path's (scratch mode: exactly; same-slot mode: the KV rows
    /// are the fresh run's rewrite of the same positions).
    fn wp16_xcheck(&mut self, slot: usize, prompt: &[u32], q: usize) -> Result<()> {
        let t0 = std::time::Instant::now();
        let plen = prompt.len();
        let end = plen - 1;
        let last = prompt[end] as i32;
        let model = self.model.clone();
        let v = model.cfg.vocab_size;
        let x = (0..self.lanes.len()).find(|&i| i != slot && self.lanes[i].is_none()).unwrap_or(slot);
        let mut r = model.ckpt_alloc()?;
        model.ckpt_save(slot, self.psc.as_ref().unwrap(), &mut r)?;
        // raw logits: greedy, unpenalized (admit re-stages both before the real seam step)
        model.set_sampling(&mut self.sc, &[None])?;
        model.set_penalties(&mut self.sc, &[None])?;
        let id_r = model.forward_step(&mut self.sc, &[last], &[end], &[slot], None)?[0];
        let lg_r: Vec<u16> = model.logits_host(&self.sc)?[..v].to_vec();
        if x != slot {
            self.cache[x] = None; // the scratch slot's rows get overwritten
            if let Some(st) = self.wp16.as_mut() { st.clear_slot(x); }
        }
        crate::dprintln!("[wp16-xcheck] slot={slot} resumed at {q}: fresh reference prefill 0..{end} into {} slot {x}",
                 if x != slot { "scratch" } else { "the same (no free lane)" });
        model.reset_slot_for_prefill(x)?;
        self.prefill_range(x, prompt, prompt, 0, end, None, false, None, false)?;
        let mut f = model.ckpt_alloc()?;
        model.ckpt_save(x, self.psc.as_ref().unwrap(), &mut f)?;
        let d = model.ckpt_diff(&r, &f)?;
        let id_f = model.forward_step(&mut self.sc, &[last], &[end], &[x], None)?[0];
        let lg_f: Vec<u16> = model.logits_host(&self.sc)?[..v].to_vec();
        let nl = lg_r.iter().zip(&lg_f).filter(|(a, b)| a != b).count();
        // serve the resumed state: recurrent state + PLE ring/hist + carry back to R (the seam step
        // above advanced them; the real seam step rewrites KV row plen-1 identically)
        model.ckpt_restore(slot, self.psc.as_mut().unwrap(), &r)?;
        if x != slot { model.reset_slot(x)?; }
        let exact = d.exact() && nl == 0 && id_r == id_f;
        crate::dprintln!("[wp16-xcheck] slot={slot} resume {q} vs fresh (slot {x}) at plen {plen}: {}; first-token logits \
                  differ {nl}/{v} (argmax {id_r} vs {id_f}) -> {} ({:.0} ms)",
                 d.line(), if exact { "EXACT" } else { "MISMATCH" }, t0.elapsed().as_secs_f64() * 1e3);
        Ok(())
    }

    /// S-A3-h: head prime after the seam decode step — snapshot the tap the seam
    /// step left in sc.resid (the f-f decode composition, byte-exact: the same tap
    /// class as the bench-mtp prime and every round's re-prime; the f-d
    /// tap_snapshot_from_prefill workaround fed round 1's verify restore a
    /// ≤1e-3-drifted prefill tap, whose chain was self-consistent but wrong), then
    /// feed the head the last prompt token with that tap (round 1's draft-extend
    /// reuses the snapshot via tap_row 0).
    /// WP03: the slot's own taps window (row 0) + a KV-only head row plen-1 paired with the stream
    /// after plen-2 (psc's head-fill carry) — FwdModel::mtp_seam_prime (--a-seam = old arm).
    fn mtp_prime(&mut self, slot: usize, last_prompt_tok: u32, plen: usize) -> Result<()> {
        let head = self.model.mtp.as_ref()
            .context("mtp_prime called without a draft head")?;
        self.model.mtp_seam_prime(&mut self.sc, self.psc.as_ref(), head, last_prompt_tok as i32, plen, slot)
    }

    /// TP-C: one lockstep point (a seam step, a plain batched step, an MTP round or a control
    /// step). (1) agree_ext(step, accept, verify width m, FNV of `words`) — the load-bearing
    /// tripwire (AGENTS §2.10): a divergence aborts the link on both ranks; (2) a u32 exchange
    /// that ships the head's (ms, plain ms, captured) — returned on EVERY rank, so every timing-
    /// derived decision downstream is the head's; (3) with `ident` on, the digests of the logits
    /// rows 0..m, the residual and the taps window are compared across ranks (G-T1-b graphed).
    ///
    /// S-B9-CF-PACK5: `digest_pack` = the round's rows live in the PACKED scratch (a packed verify):
    /// the `--exl3-tp-ident` digests then read it, not the served scratch (whose argmax / logits are
    /// only `--max-batch`-wide: a 16-row pack indexed past them).
    fn tp_round(&mut self, what: &str, a: usize, m: usize, words: &[u32], ms: f64, plain_ms: f64,
                captured: bool, digest_pack: bool) -> Result<(f64, f64, bool)> {
        let Some(t) = self.tp.as_mut() else { return Ok((ms, plain_ms, captured)) };
        if t.world > 2 {
            // TP-4D: ONE merged head-hub op (tp_lockstep::RND) instead of agree / exchange / agree
            let mut x = crate::net::link_hub(t.deadline_ms)?;
            let epoch = crate::net::traced_device_epoch();
            let (model, sc) = (&self.model, if digest_pack { self.scp.as_ref().unwrap_or(&self.sc) } else { &self.sc });
            return t.round_hub(&mut x, what, a, m, words, ms, plain_ms, captured, epoch, |m| model.tp_ident_digests(sc, m));
        }
        t.step += 1;
        let (step, rank, ident) = (t.step, t.rank, t.ident);
        let h = fnv32(0, words.iter().copied());
        // Fence A = the agree token itself: both ranks finished this step's forward, so every hot-
        // path epoch payload of it has been read on both sides (the exchange below reuses ring slot
        // memory: its send buffer is send slot 0, its receive lands in recv slot gen & 7).
        let (t0, s0) = (std::time::Instant::now(), crate::net::wait_sleeps());
        let r = tp_agree_eq(agree_field(step, AGREE_A), a, t.width_said(m), h);
        t.wp.rec(0, t0, s0);
        t.wp.rounds += 1;
        if let Err(why) = r {
            crate::net::abort_link();
            anyhow::bail!("TP agree() FAILED at {what} (lockstep step {step}, accept {a}, width {m}, hash {h:08x}): {why} — \
                           ranks diverged or the link aborted");
        }
        let digests = if ident {
            let dsc = if digest_pack { self.scp.as_ref().unwrap_or(&self.sc) } else { &self.sc };
            self.model.tp_ident_digests(dsc, m)?
        } else { Vec::new() };
        let tag = 0xC0DE_0000u32 | (step as u32 & 0xFFFF);
        let (mb, pb) = (ms.to_bits(), plain_ms.to_bits());
        let mut mine = vec![tag, mb as u32, (mb >> 32) as u32, pb as u32, (pb >> 32) as u32, captured as u32,
                            digests.len() as u32];
        mine.extend(digests.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]));
        let (t0, s0) = (std::time::Instant::now(), crate::net::wait_sleeps());
        let peer = crate::net::exchange_u32s(&mine, mine.len() + 4)?;
        let tw = self.tp.as_mut().unwrap();
        tw.wp.rec(1, t0, s0);
        // Fence B: nobody re-enters the hot path (whose next epochs reuse the ring slots) until both
        // ranks have read their exchange payload. (Without it a rank that finished its exchange first
        // raced into the next forward and overwrote the slower rank's unread exchange slot — the
        // 2026-09-29 smoke: "recv payload tail never landed", abort code 2.)
        let (t0, s0) = (std::time::Instant::now(), crate::net::wait_sleeps());
        let r = tp_agree_eq(agree_field(step, AGREE_B), 0, 0, tag);
        tw.wp.rec(2, t0, s0);
        if let Err(why) = r {
            crate::net::abort_link();
            anyhow::bail!("TP post-exchange fence FAILED at {what} (lockstep step {step}): {why}");
        }
        anyhow::ensure!(peer[0] == tag, "TP step sync at {what}: peer at {:#x}, this rank at {tag:#x}", peer[0]);
        anyhow::ensure!(peer[6] as usize == digests.len(), "TP step sync at {what}: peer sent {} digests, this rank {}",
                        peer[6], digests.len());
        let t = self.tp.as_mut().unwrap();
        t.agrees += 1;
        if ident {
            let bad = (0..digests.len()).filter(|&i| mine[7 + 2 * i] != peer[7 + 2 * i] || mine[8 + 2 * i] != peer[8 + 2 * i]).count();
            t.ident_bufs += digests.len() as u64;
            t.ident_bad += bad as u64;
            if bad > 0 && t.ident_bad as usize == bad {
                crate::deprintln!("[tp-ident] FIRST MISMATCH at {what} (lockstep step {step}, width {m}): {bad} of {} digests differ \
                           (logits rows 0..{m}, resid, taps)", digests.len());
            }
            // MTPAGREE1: name WHICH digests differ (index < m: logits row / vp key; m: resid; m+1: taps), every time,
            // at --log-level debug, so the first divergent round says trunk vs head.
            if bad > 0 {
                let which: Vec<String> = (0..digests.len()).filter(|&i| mine[7 + 2 * i] != peer[7 + 2 * i] || mine[8 + 2 * i] != peer[8 + 2 * i])
                    .map(|i| format!("{}:{:08x}{:08x}/{:08x}{:08x}", if i < m { format!("logits{i}") } else { ["resid", "taps", "x", "chain_xh", "head_kv"].get(i - m).map(|s| s.to_string()).unwrap_or_else(|| format!("d{i}")) },
                                     mine[8 + 2 * i], mine[7 + 2 * i], peer[8 + 2 * i], peer[7 + 2 * i])).collect();
                crate::deprintln!("[tp-ident] {what} step {step} width {m}: differing digests (mine/peer) {}", which.join(" "));
            }
        }
        if rank == 0 {
            Ok((ms, plain_ms, captured))
        } else {
            Ok((f64::from_bits(peer[1] as u64 | ((peer[2] as u64) << 32)),
                f64::from_bits(peer[3] as u64 | ((peer[4] as u64) << 32)), peer[5] != 0))
        }
    }

    /// One decode round over all active lanes (batched; lanes are independent slots).
    /// S-A3-f-d: lanes with a loaded M1 head run chain-verify rounds (speculation ON by
    /// default per AGENTS 1b — --exl3-no-mtp is the owner's diagnostics hatch, never a
    /// default). API-parity G2: sampled lanes speculate too (device-sampled verify rows).
    /// Lanes whose request turned MTP off (G1) batch through the plain decode path.
    fn step(&mut self) -> Result<()> {
        let before = self.steps;
        let r = self.step_rounds();
        // H3: round COMPLETE — the age gauge measures from here. Every arm of `step_rounds` ends
        // here (it used to be written on the rr arm only: a shared plain step and an fcfs step
        // left velogb10_scheduler_last_step_age_seconds growing while busy read 1).
        if r.is_ok() && self.steps != before { crate::metrics::sched_touch(); }
        self.publish_kv(); // issue #24: lane positions after this step (finished lanes read 0)
        r
    }

    /// Public issue #24: publish every lane's KV occupancy to /metrics — the positions a running
    /// request holds (`Lane.pos`: prompt + generated; the prefill in flight: the rows written so
    /// far) and, for a free lane, the prompt positions its prefix-cache snapshot keeps. Host-only:
    /// a few relaxed loads/stores per lane, once per scheduler step / prefill chunk / admit.
    fn publish_kv(&self) {
        let pf = self.kv_pf;
        crate::metrics::kv_publish(self.lanes.iter().zip(self.cache.iter()).enumerate()
            .map(|(s, (lane, cache))| kv_lane_occupancy(lane.as_ref().map(|l| l.pos), pf.filter(|p| p.0 == s).map(|p| p.1),
                                                        cache.as_ref().map(|c| c.len()))));
    }

    /// The rounds of one `step` (every arm may return early; `step` does the per-step publication).
    fn step_rounds(&mut self) -> Result<()> {
        crate::metrics::sched_step(); // v0.7.3 gauge: rounds + busy
        // WP02 cancel sweep (the NVFP4 decode_step sweep, batch.rs): a lane whose client is gone
        // (disconnect, or the HTTP side dropped the receiver on a stop-string hit) frees its slot
        // NOW instead of decoding to EOS/max_new while every later request waits behind it.
        // TP-D: under TP the HEAD sweeps (run_tp_head) and ships each cancel as a Step event — a
        // rank-local sweep would free a lane on one rank only.
        for s in 0..self.lanes.len() {
            if self.tp.is_none() {
                if let Some(l) = self.lanes[s].as_ref() {
                    let closed = l.tx.is_closed();
                    let over = !closed && crate::server::stream_backlog_exceeded(l.tx.len()); // LR-5
                    if closed || over {
                        let lane = self.take_lane(s, false).unwrap();
                        if over {
                            crate::metrics::stream_backlog_cancel();
                            crate::dreprintln!("[exl3-serve] slot {s}: stream cancelled — unconsumed event backlog at the --stream-backlog-events limit (reader stalled; LR-5)");
                        }
                        lane.log_cancel(s);
                    }
                }
            }
        }
        let slots: Vec<usize> = self.lanes.iter().enumerate()
            .filter(|(_, l)| l.is_some())
            .map(|(i, _)| i)
            .collect();
        if slots.is_empty() {
            return Ok(());
        }
        self.steps += 1;
        if self.inject_panic == Some(self.steps) {
            panic!("--inject-panic-steps: injected scheduler panic at decode step {} (liveness gate)", self.steps);
        }
        crate::tel::note_step();
        let mtp_on = self.model.mtp.is_some()
            && !crate::exl3_forward::mtp_disabled_by_opt();
        // S-B9-REL-MTPREPROBE: arm due re-probes BEFORE dispatch (pure lane state —
        // plain-step counts and agreed HEAD ms inputs — so every TP rank arms the same
        // lanes; lockstep holds through the transition). The seam is exactly a control
        // round's post-step state: the lane's plain steps kept its head row primed and
        // its tap snapshotted (step_plain), so Some(0) arms it cleanly.
        if mtp_on && self.reprobe_on {
            let mut armed: Vec<(usize, usize, usize)> = Vec::new(); // (slot, plain steps waited, off event)
            for &s in &slots {
                let due = self.lanes[s].as_ref()
                    .map_or(false, |l| l.mtp.is_none() && l.reprobe.probe_due(&self.reprobe_cfg));
                if due {
                    let (steps, ev) = {
                        let l = self.lanes[s].as_ref().unwrap();
                        (l.reprobe.plain_steps, l.reprobe.off_events)
                    };
                    let l = self.lanes[s].as_mut().unwrap();
                    l.mtp = Some(0);
                    l.ema_ms = 0.0;
                    l.ema_tok = 0.0;
                    l.ema_n = 0;
                    l.reprobe.armed_for_probe();
                    armed.push((s, steps, ev));
                }
            }
            for &(s, steps, ev) in &armed {
                crate::drprintln!("[exl3-serve] MTP re-probe (slot {s}): back ON after {steps} plain steps \
(off event {ev}; probe window {} rounds, a {:.0}% sustained loss turns it off again)",
                           self.reprobe_cfg.probe_window, 100.0 * self.reprobe_cfg.margin);
            }
        }
        // CF-P1e load-adaptive batching: past `--spec-lanes-max` busy lanes, serial speculative rounds stop
        // scaling (one lane per round), so every busy lane shares one batched plain step instead; the
        // lanes' MTP heads stay in sync (see step_plain's `sync_head`), so speculation resumes when load drops.
        // S-C4XOVER arm-rate probe (generalizes the F4 shared-cost probe): while one arm
        // rides, the idle arm's MEASURED rate goes stale; at most every 256 riding rounds (16
        // while it has never been measured) a 4-step window of the idle arm runs, refreshing
        // its measured rate (and, for shared, the learned scale) so the pick below stays
        // grounded in DELIVERY, not in the priors. Gated like F4 (lane packing on, Auto
        // policy) but at >=2 busy lanes with at least one MTP-capable; pure scheduler-state
        // decision (counters, rates, policy are rank-identical or adopted), so both TP ranks
        // take it in the same step, like the re-probe arming above.
        self.spec_probe_ctr = self.spec_probe_ctr.saturating_add(1);
        let mut probe_force: Option<bool> = None;
        if mtp_on && self.lane_pack.on() && self.spec_policy == SpecLanes::Auto && slots.len() >= 2
            && slots.iter().any(|&s| self.lanes[s].as_ref().map_or(false, |l| l.mtp.is_some())) {
            if let Some((arm, rem)) = self.spec_probe {
                self.spec_probe = if rem <= 1 { None } else { Some((arm, rem - 1)) };
                probe_force = Some(arm as usize == ARM_SHARED);
                self.shared_probe_ctr = 0;
            } else {
                let riding = if self.shared_last { ARM_SHARED } else { ARM_SERIAL };
                let idle = 1 - riding;
                let first = self.spec_arm_rate[idle] <= 0.0;
                let agrees = self.spec_est_pick == Some(riding == ARM_SHARED)
                    && self.spec_margin >= 0.12 && (self.spec_margin_n as i64 - slots.len() as i64).abs() <= 1;
                let backoff = if agrees { 8.0 }
                              else if self.spec_margin >= 0.12 && (self.spec_margin_n as i64 - slots.len() as i64).abs() <= 1 { 4.0 }
                              else if self.spec_margin >= 0.06 { 2.0 } else { 1.0 };
                let every = (SPEC_ARM_PROBE_EVERY as f64 * backoff) as u32;
                let due = self.spec_probe_ctr >= if first { SPEC_ARM_PROBE_FIRST } else { every };
                if due && !self.arm_fresh(idle, slots.len()) {
                    self.spec_probe = Some((idle as u8, SPEC_ARM_PROBE_WIN));
                    self.spec_probe_ctr = 0;
                    probe_force = Some(idle == ARM_SHARED);
                    self.shared_probe_ctr = 0;
                    crate::metrics::spec_lanes_probe(idle);
                    if self.policy_logs < 64 {
                        self.policy_logs += 1;
                        crate::dreprintln!("[exl3-serve] spec-lanes auto: probing the {} arm for {} steps with {} busy lanes \
                           (its measured rate went stale; the pick prices both arms by delivery)",
                            if idle == ARM_SHARED { "shared plain" } else { "serial speculation" }, SPEC_ARM_PROBE_WIN, slots.len());
                    }
                }
            }
        }
        if mtp_on && probe_force.unwrap_or_else(|| self.use_shared_plain(&slots)) {
            self.shared_streak += 1;
            self.shared_probe_ctr = 0; // a shared round ran: the scale just learned, no probe owed
            // Stale-estimate refresh: lanes' speculative-speed EMAs freeze while they ride shared steps, so every
            // 128th shared round one lane (rotating) runs a serial speculative round instead; the others skip
            // this iteration. ~0.5% of rounds; keeps the cost-based choice tracking the content.
            let refresh_every = if self.spec_margin >= 0.12 { 256 } else { 128 };
            if probe_force.is_none() && self.spec_policy == SpecLanes::Auto && self.shared_streak % refresh_every == 0 {
                let s = slots[(self.shared_streak / 128) % slots.len()];
                if self.lanes[s].as_ref().map_or(false, |l| l.mtp.is_some()) {
                    // shared-refresh: slots.len() >= 2 in this arm, so the DF2 chain (a single
                    // token stream) stays out of it — the pack-MTP round refreshes the EMA.
                    return self.step_mtp(s, false);
                }
            }
            crate::metrics::lane_pack_round(2); // arm = shared_plain
            return self.step_plain(&slots, true);
        }
        self.shared_streak = 0;
        // S-C4XOVER: F4's single-step shared probe is superseded by the arm-rate probe
        // above (same idea, windowed, both directions, feeding measured rates).
        // S-B9-CF-PACK4 `--lane-pack on`: two packable busy lanes share ONE packed speculative
        // verify. `pack_plan` is None with the flag off or fewer than two busy lanes, so both of
        // those take the loops below untouched (off == master; a lone request never packs).
        if mtp_on {
            if let Some(plan) = self.pack_plan(&slots) {
                return self.step_packed(&slots, plan);
            }
        }
        // CF-FCFS: the fcfs arm steps ONE lane (the front) and leaves the rest untouched this
        // scheduler step; the MTP-off lanes still batch together exactly as below. rr (the default)
        // runs the original loop, unchanged and in slot order.
        if self.lane_order == LaneOrder::Fcfs {
            let mut plain: Vec<usize> = Vec::new();
            let mut capable: Vec<(usize, u64, usize)> = Vec::new();
            for &s in &slots {
                let lane = self.lanes[s].as_ref().unwrap();
                if mtp_on && lane.mtp.is_some() {
                    capable.push((s, lane.admit_seq, lane.fcfs_credit));
                } else {
                    plain.push(s);
                }
            }
            let front = fcfs_front(&capable, self.lane_quantum);
            match front {
                Some(s) => {
                    let before = self.lanes[s].as_ref().unwrap().generated;
                    self.step_mtp(s, slots.len() == 1)?;
                    // step_mtp may have FINISHED this lane (stop / length / cancel / a dead client):
                    // it frees the slot (self.lanes[s] = None) before returning. The front lane is
                    // then simply gone — its turn ended early and the next step picks the next-oldest
                    // — so every post-step access must tolerate the slot being empty.
                    let lane = match self.lanes[s].as_mut() {
                        Some(lane) => lane,
                        None => {
                            if !plain.is_empty() {
                                self.step_plain(&plain, false)?;
                            }
                            return Ok(());
                        }
                    };
                    let delta = lane.generated - before;
                    let (mut seq, mut credit) = (lane.admit_seq, lane.fcfs_credit);
                    // fcfs_charge works on the replicated (seq, credit) tuple; BOTH fields are
                    // written back every step (the credit accumulation lives inside charge —
                    // forgetting this write-back on the no-rotate path pins credit at 0 and the
                    // front lane never reaches the quantum: run-to-completion again).
                    let rotated = fcfs_charge(&mut seq, &mut credit, delta, self.lane_quantum, &mut self.admit_ctr);
                    lane.fcfs_credit = credit;
                    if rotated {
                        lane.admit_seq = seq; // re-stamped: the queue position IS the sequence number
                        self.fcfs_rots += 1; // TOTAL rotations; the first 16 are logged and the total is printed at a clean idle point
                        if self.fcfs_rots <= 16 {
                            crate::dreprintln!("[exl3-serve] lane-order fcfs: slot {s} used its quantum ({} tokens) \
                                       and rotates to the back (rotation {}/16 logged)", self.lane_quantum, self.fcfs_rots);
                        }
                    }
                }
                None => {
                    // Unreachable with the charge-at-quantum bookkeeping: a lane's credit is reset
                    // the moment it reaches the quantum, so some lane always has credit < quantum.
                    // Kept as a guard: this step services no capable lane.
                }
            }
            if !plain.is_empty() {
                self.step_plain(&plain, false)?;
            }
            return Ok(());
        }
        let mut plain: Vec<usize> = Vec::new();
        for &s in &slots {
            let lane = self.lanes[s].as_ref().unwrap();
            if mtp_on && lane.mtp.is_some() {
                self.step_mtp(s, slots.len() == 1)?;
            } else {
                plain.push(s);
            }
        }
        if !plain.is_empty() {
            self.step_plain(&plain, false)?;
        }
        Ok(())
    }

    /// The depth of a speculative round for a lane at position `p` with `remaining` tokens of
    /// budget: (after the end-of-window and remaining-budget clamps, after the sparse-switch
    /// clamp as well). The second is the round's depth; first > second = the round is clamped
    /// at the switch. One definition for `step_mtp` and the packed arm's plan.
    fn round_depth(&self, p: usize, remaining: usize) -> (usize, usize) {
        let mut k_eff = self.mtp_k.min(self.model.max_pos().saturating_sub(p + 2));
        if crate::exl3_forward::a_flags().rem_clamp {
            k_eff = k_eff.min(remaining.saturating_sub(1));
        }
        (k_eff, self.model.regime_depth(p, k_eff))
    }

    /// S-B9-CF-PACK4: this step's packing (`lane_pack::plan`), or None = the flag is off or
    /// fewer than two lanes are busy, and the caller runs master's loops untouched. Every input
    /// is lane state plus the rotation counter (which advances once per planned step).
    fn pack_plan(&mut self, slots: &[usize]) -> Option<crate::lane_pack::Plan> {
        use crate::lane_pack::{lane_block, LaneFacts, LaneView, Skip};
        if !self.lane_pack.on() || slots.len() < 2 { return None; }
        let views: Vec<LaneView> = slots.iter().map(|&s| {
            let l = self.lanes[s].as_ref().unwrap();
            let (k_pre, kd_reg) = self.round_depth(l.pos, l.max_new.saturating_sub(l.generated));
            let facts = LaneFacts {
                mtp_armed: l.mtp.is_some(),
                force_control: l.force_control,
                probing: l.reprobe.phase == crate::mtp_reprobe::Phase::Probing,
                sampled: l.samp_rows(1)[0].is_some(),
                // PACK4S: sampled lanes pack (real-q and match rule); only the lone-verify
                // sampler dump (--wp24-dump records the lone round's rows) keeps them out; and
                // (PACK5) a topology whose packed sampled tail is not gated (TP=2)
                sampled_lone: crate::wp24::dump_on() || !self.pack_sampled,
                penalized: l.pen.is_some(),
                image: l.img,
                depth: 0, // set by lane_view from kd_reg (capped at PACK_DEPTH_MAX: PACK4-REVIEW A4)
            };
            // The verify's regime is its LAST row's (p + w for the w <= kd drafts the DDS keeps).
            // With the per-row regime on, a round never straddles the switch, so every w gives
            // the regime of p + kd; with --qsa-row-regime 0 a straddling round's regime depends
            // on w, which is only known after drafting: such a lane does not pack (and the boot
            // refuses that posture, PACK4-REVIEW B3). `lane_view` is the pure, unit-tested rule.
            crate::lane_pack::lane_view(s, facts, l.pos, k_pre, kd_reg, self.in_prefill, self.pack_ready,
                                        &|p| self.model.qsa_live(p))
        }).collect();
        let plan = crate::lane_pack::plan(self.lane_pack, &views, self.pack_rot);
        if plan.is_some() { self.pack_rot += 1; }
        plan
    }

    /// One lane round ran outside a pack: count it, and log the first few (64 lines in all, at
    /// most 8 per reason so one common reason cannot use the whole budget).
    fn note_skip(&mut self, s: usize, why: crate::lane_pack::Skip) {
        crate::metrics::lane_pack_skip(why.index());
        if self.pack_logs < 64 && crate::metrics::lane_pack_skips(why.index()) <= 8 {
            self.pack_logs += 1;
            crate::dreprintln!("[exl3-serve] lane-pack skip: {} (slot {s}; line {}/64, every skip is counted: /metrics \
velogb10_lane_pack_skip_total{{reason=\"{}\"}})", why.text(), self.pack_logs, why.label());
        }
    }

    /// S-B9-CF-PACK4: one scheduler step under `--lane-pack on` with at least two busy lanes.
    /// Lanes are visited in slot order, as in the serial loop: a lane that is the first slot of
    /// a planned pair runs the packed round with its partner; every other lane runs exactly
    /// what the serial loop runs for it (its lone speculative round, or the plain batch for
    /// MTP-off lanes), with the reason it stayed out of a pack counted.
    fn step_packed(&mut self, slots: &[usize], plan: crate::lane_pack::Plan) -> Result<()> {
        let mut plain: Vec<usize> = Vec::new();
        for &s in slots {
            match plan.pair_of(s) {
                Some([a, b]) => {
                    if s == a { // b ran with a (slots ascend, a < b)
                        if let Err(e) = self.step_pair(a, b) {
                            // PACK4-REVIEW A3: an error out of the PACKED part of the round leaves
                            // the arm off for the rest of the boot (the caller still fails every
                            // live lane, as for any decode-step error); the control-cadence
                            // fallback's serial rounds are the lone path's own and do not.
                            if std::mem::take(&mut self.pack_stats.in_round) { self.pack_disable(a, b, &e); }
                            return Err(e);
                        }
                    }
                }
                None => {
                    if let Some(why) = plan.skip_of(s) { self.note_skip(s, why); }
                    if self.lanes[s].as_ref().map_or(false, |l| l.mtp.is_some()) {
                        self.step_mtp(s, false)?; // packed arm: never a lone lane
                    } else if self.lanes[s].is_some() {
                        plain.push(s);
                    }
                }
            }
        }
        if !plain.is_empty() {
            self.step_plain(&plain, false)?;
        }
        // H3 (round COMPLETE): written by `step`, the wrapper, for every arm of `step_rounds` (this one included)
        Ok(())
    }

    /// S-B9-CF-PACK4: ONE packed speculative round for the lanes in slots `sa` < `sb` (design
    /// 3.2): each request drafts on the served scratch with the lone round's own draft half;
    /// ONE packed verify (graph replay on the packed scratch) accepts and commits per segment;
    /// then, per request, the head re-prime, the lone round's bookkeeping (`round_account`, fed
    /// the request's SHARE of the round time) and its emit loop (`round_emit`). Both lanes are
    /// unpenalized, MTP-armed, on one side of the sparse switch, with depth >= 1.
    /// S-B9-CF-PACK4S: each lane is greedy or sampled (real-q under `--spec-sampling ratio`,
    /// else the match rule), in any combination. A lane drafts under ITS OWN sampler staging,
    /// exactly as its lone round stages it; a real-q lane's draft lists go to its row offset
    /// of the packed scratch before the next lane drafts; the verify stages one sampler block
    /// per row and the lane's counters advance as in its lone round.
    fn step_pair(&mut self, sa: usize, sb: usize) -> Result<()> {
        use crate::lane_pack::{ctrl_period, lane_share_ms, pair_owes_control, PostLane, PreSeg, SegSamp, Skip};
        let k = self.mtp_k;
        let af = crate::exl3_forward::a_flags();
        // (slot, bonus token, position, tap row, depth, cost) of each lane, before the round
        let pre: Vec<(usize, i32, usize, usize, usize, f64)> = [sa, sb].iter().map(|&s| {
            let l = self.lanes[s].as_ref().unwrap();
            let (_, kd) = self.round_depth(l.pos, l.max_new.saturating_sub(l.generated));
            let kd = kd.min(crate::lane_pack::PACK_DEPTH_MAX); // PACK4-REVIEW A4: the plan's own depth
            (s, l.last_tok as i32, l.pos, l.mtp.unwrap(), kd,
             if l.ema_tok > 0.0 { l.ema_ms / l.ema_tok } else { 0.0 })
        }).collect();
        // The control cadence is the serial arm's, count for count: the pair would consume the
        // counter values ctrl_round + 1 and + 2; if either is a control round for its lane, the
        // two lanes run serial this step (the control round itself is step_mtp's).
        let (per_a, per_b) = (ctrl_period(af.ctrl == 0, self.ema_plain, pre[0].5), ctrl_period(af.ctrl == 0, self.ema_plain, pre[1].5));
        if af.ctrl != 3 && pair_owes_control(self.ctrl_round, per_a, per_b) {
            self.note_skip(sa, Skip::Control);
            self.note_skip(sb, Skip::Control);
            self.step_mtp(sa, false)?; // pair control: two lanes in this step
            return self.step_mtp(sb, false);
        }
        self.pack_stats.in_round = true; // PACK4-REVIEW A3: cleared when the round completes
        let xcheck = crate::exl3_forward::a_xcheck_on();
        if xcheck {
            for &(s, _, _, tap_row, _, _) in &pre { self.model.tap_xcheck_verify(&self.sc, s, tap_row)?; }
        }
        self.ctrl_round += 2;
        let head = self.model.mtp.as_ref().unwrap();
        let t0 = std::time::Instant::now();
        let g0 = self.model.graph_count(); // a round that captures a graph is not cost-fit
        // 1. draft: per request, sequential — the lone round's draft half, under the lone
        // round's own staging of the served scratch (which may still carry another lane's):
        // the lane's sampler rows and, for a real-q lane, its draft sampler (the draft tail and
        // the pass-graph keys read these), and no penalties (a penalized lane does not pack).
        let dds = self.draft_conf > 0.0;
        let mut drafts: Vec<crate::exl3_forward::PackDraft> = Vec::with_capacity(2);
        let mut at = 0usize; // the next segment's first row in the packed verify
        for &(s, b, p, tap_row, kd, _) in &pre {
            let (rows, rq) = (self.lanes[s].as_ref().unwrap().samp_rows(k + 1), self.rq_stage(s));
            let seg = SegSamp { samp: rows[0], ratio: rq.map_or(false, |r| r.mode == 2) };
            self.model.set_spec_sampling(&mut self.sc, &rows, rq)?;
            self.model.set_penalties(&mut self.sc, &vec![None; k + 1])?;
            let d = if dds {
                let target = {
                    let lane = self.lanes[s].as_mut().unwrap();
                    let t = wp23_target(self.draft_conf, wp23_guard_on(k), lane.mtp_rounds,
                                        lane.wp23.fit.slope(), lane.ema_tok, lane.ema_ms);
                    lane.wp23.target = t;
                    t
                };
                let cal = &self.cal;
                let mut reach = 1.0f64;
                let mut keep = |i: usize, c: f32| { reach *= cal.estimate_at(i, c); reach >= target };
                self.model.pack_draft(&mut self.sc, head, s, kd, b, p, tap_row, Some(&mut keep), seg, rq)?
            } else {
                // --draft-confidence 0: the fixed-depth chain of the round's depth (the lone mtp_round's)
                self.model.pack_draft(&mut self.sc, head, s, kd, b, p, tap_row, None, seg, rq)?
            };
            // the real-q list hand-off, issued BEFORE the next lane drafts: both chains write
            // the served list rows 0..w (no-op for a greedy or match-rule lane)
            let scp = self.scp.as_mut().context("lane-pack: the packed scratch is missing")?;
            self.model.pack_list_handoff(&self.sc, scp, at, &d)?;
            at += d.drafts.len() + 1;
            drafts.push(d);
        }
        // S-B9-CF-PACK5 (TP): the pack's ONE pre-verify lockstep point — after BOTH lanes drafted,
        // before the packed verify launches (the verify's R-keyed barrier sequence needs every rank
        // to have built the same pack). The pack was decided by `pack_plan` from replicated lane
        // state, so a divergence here is a bug (or a fault drill), never a legitimate difference.
        // `--exl3-tp-preverify=0` (diagnostic) skips it, as it skips the lone round's.
        let tpre0 = std::time::Instant::now();
        if self.tp.is_some() && tp_preverify_on() {
            self.model.pack_drain_wasted(&drafts)?;
            let segs = [PreSeg::of(drafts[0].slot, drafts[0].p, drafts[0].b, &drafts[0].drafts, &drafts[0].samp, drafts[0].launched, drafts[0].wasted),
                        PreSeg::of(drafts[1].slot, drafts[1].p, drafts[1].b, &drafts[1].drafts, &drafts[1].samp, drafts[1].launched, drafts[1].wasted)];
            let words = crate::lane_pack::pre_words(&segs);
            let r = crate::lane_pack::pack_rows_total(&segs);
            let launched = drafts[0].launched + drafts[1].launched;
            let t = self.tp.as_mut().context("lane-pack: TP state missing")?;
            t.pre_verify_pack(r, segs[0].drafts.len() + 1, &words, launched)?;
            self.pack_stats.ms_pre += tpre0.elapsed().as_secs_f64() * 1e3;
        }
        // S-B9-CF-PACK5: the decision trace of a packed round, on EVERY rank (head and node print the
        // same line: slots, rows, the combined pack hash, head passes, wasted passes), so a divergence
        // is a `diff` of two logs. The first 16 rounds, then every 4096th.
        {
            let n = self.pack_stats.rounds + 1;
            if n <= 16 || n % 4096 == 0 {
                let segs = [PreSeg::of(drafts[0].slot, drafts[0].p, drafts[0].b, &drafts[0].drafts, &drafts[0].samp, drafts[0].launched, drafts[0].wasted),
                            PreSeg::of(drafts[1].slot, drafts[1].p, drafts[1].b, &drafts[1].drafts, &drafts[1].samp, drafts[1].launched, drafts[1].wasted)];
                let h = fnv32(0x5EED, crate::lane_pack::pre_words(&segs).iter().map(|&d| d as u32));
                crate::drprintln!("[exl3-serve] lane-pack: round #{n}{}: slots {}+{} rows {}+{}={} pack hash {h:08x} head passes {}+{} wasted {}+{}",
                                 self.tp.as_ref().map_or(String::new(), |t| format!(" (TP rank {})", t.rank)),
                                 drafts[0].slot, drafts[1].slot, drafts[0].drafts.len() + 1, drafts[1].drafts.len() + 1,
                                 crate::lane_pack::pack_rows_total(&segs), drafts[0].launched, drafts[1].launched,
                                 drafts[0].wasted as u8, drafts[1].wasted as u8);
            }
        }
        let t1 = std::time::Instant::now();
        // 2. ONE packed verify: stage, replay, taps into the served windows, one readback
        let scp = self.scp.as_mut().context("lane-pack: the packed scratch is missing")?;
        let (outs, pvm) = self.model.pack_verify(&mut self.sc, scp, &drafts[0], &drafts[1])?;
        let t2 = std::time::Instant::now();
        // 3. per request: the head re-prime from the taps the verify left in its window
        for j in 0..2 {
            self.model.pack_finish(&mut self.sc, head, &drafts[j], &outs[j])?;
        }
        let t3 = std::time::Instant::now();
        let captured_local = self.model.graph_count() != g0;
        let ms_of = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1e3;
        // (--wp24-xcheck's host re-derivation is not round cost: the lone round's `diag_ms` rule)
        let ms_local = (ms_of(t0, t3) - pvm.diag).max(0.0);
        // S-B9-CF-PACK5 (TP): the packed round's POST-verify lockstep point, `tp_round` as the lone
        // round runs it: agree the pair's outcome (per lane the accept count, rows, emitted ids,
        // drafts and draft confidences; the total row count R as the width) and TAKE THE HEAD's
        // round ms and capture flag. `step_pair` fed its LOCAL clock to the cost model before; under
        // TP that would let the EMAs, the DDS fit and the section 6 control diverge between ranks.
        let tpost0 = std::time::Instant::now();
        let (ms_round, _plain_ms, captured) = if self.tp.is_some() {
            let lanes: Vec<PostLane> = (0..2).map(|j| PostLane {
                accepted: outs[j].a, rows: drafts[j].drafts.len() + 1,
                emitted: &outs[j].emitted, drafts: &outs[j].drafts, confs: &drafts[j].confs }).collect();
            let w = crate::lane_pack::post_words(&lanes);
            let r = lanes.iter().map(|l| l.rows).sum::<usize>();
            self.tp_round("pack-round", outs[0].a + outs[1].a, r, &w, ms_local, 0.0, captured_local, true)?
        } else {
            (ms_local, 0.0, captured_local)
        };
        if self.tp.is_some() { self.pack_stats.ms_post += tpost0.elapsed().as_secs_f64() * 1e3; }
        let ms_lane = lane_share_ms(ms_round, 2);
        crate::metrics::lane_pack_round(0); // arm = packed
        {
            let st = &mut self.pack_stats;
            st.rq_rounds += drafts.iter().any(|d| d.samp.ratio) as u64;
            for d in &drafts {
                let kind = if d.samp.samp.is_none() { 2 } else if d.samp.ratio { 0 } else { 1 };
                crate::metrics::lane_pack_lane_round(kind);
                match kind { 0 => st.lanes_rq += 1, 1 => st.lanes_match += 1, _ => st.lanes_greedy += 1 }
            }
            st.rounds += 1;
            st.rows += (drafts[0].drafts.len() + drafts[1].drafts.len() + 2) as u64;
            st.toks += (outs[0].a + outs[1].a + 2) as u64;
            st.ms_draft += ms_of(t0, t1);
            st.ms_verify += ms_of(t1, t2);
            st.ms_finish += ms_of(t2, t3);
            st.ms_stage += pvm.stage;
            st.ms_graph += pvm.graph;
            st.ms_read += pvm.read;
            let r = (drafts[0].drafts.len() + drafts[1].drafts.len() + 2).min(16);
            st.by_r[r].0 += 1;
            st.by_r[r].1 += pvm.graph;
            st.by_r[r].2 += (outs[0].a + outs[1].a + 2) as u64;
        }
        // every verified position labels the calibrator (the lone round's rule, per request;
        // a fixed-depth chain has no confidences and labels nothing, as in the lone mtp_round)
        for j in 0..2 {
            for (i, &c) in drafts[j].confs.iter().enumerate() {
                if i < outs[j].a { self.cal.add_label(i, c, true); } else { self.cal.add_label(i, c, false); break; }
            }
            if dds { self.cal.decay_step(); }
        }
        // 4. per request: the serial arm's bookkeeping and emit loop. A lane that finishes here
        // (stop / length / loop / a gone client) frees its slot; its partner's round is complete
        // on the device and in `outs`, so it is accounted and emitted regardless.
        let row0 = [0usize, drafts[0].drafts.len() + 1];
        for j in 0..2 {
            let s = pre[j].0;
            // the lane's real-q flag: ctr advances by k + 1 and (real-q) dctr by k, as in its lone round
            self.round_account(s, &outs[j], ms_lane, 0.0, captured, false, false, drafts[j].samp.ratio, xcheck, true)?;
            self.round_emit(s, &outs[j], Some(row0[j]))?;
        }
        self.pack_stats.in_round = false;
        Ok(())
    }

    /// PACK4-REVIEW A3: a packed round failed. The caller fails every live lane (a decode-step error
    /// always does: after an arbitrary error mid-round the shared staging — the pinned stage's
    /// in-flight copies, `sc.rp.pending`, the speculative-pass state — is not proven clean, and
    /// only the pair's own state could be spared, which this does not attempt). Without this the
    /// arm stayed on and a deterministic failure repeated for every later concurrent pair until
    /// a restart. Now the server degrades to the serial arm for the rest of the boot: one loud
    /// line, `velogb10_lane_pack_disabled 1`, and every later round is master's.
    fn pack_disable(&mut self, a: usize, b: usize, why: &anyhow::Error) {
        if !self.lane_pack.on() { return; }
        self.lane_pack = crate::lane_pack::LanePack::Off;
        crate::metrics::lane_pack_disabled();
        crate::reprintln!("[exl3-serve] LANE-PACK DISABLED for the rest of this boot: the packed round of slots {a} + {b} failed ({why:#}); \
every busy lane of this step fails with it, and every later round runs the serial arm (--lane-pack off semantics; \
/metrics velogb10_lane_pack_disabled 1). Restart the server to re-enable packing");
    }

    /// S-B9-CF-PACK4: the idle-point line of the packed arm (None = nothing new since the last).
    fn pack_idle_line(&mut self) -> Option<String> {
        let st = self.pack_stats;
        if st.rounds == st.printed { return None; }
        self.pack_stats.printed = st.rounds;
        let curve: Vec<String> = st.by_r.iter().enumerate().filter(|(_, c)| c.0 > 0)
            .map(|(r, c)| format!("R{r}:{}x {:.1}ms {:.2}tok", c.0, c.1 / c.0 as f64, c.2 as f64 / c.0 as f64)).collect();
        crate::drprintln!("[exl3-serve] lane-pack cost by width (rounds, graph-and-sync ms, tokens per round): {}", curve.join(" | "));
        let n = st.rounds.max(1) as f64;
        let skips: Vec<String> = crate::lane_pack::Skip::ALL.iter()
            .map(|w| (w.label(), crate::metrics::lane_pack_skips(w.index())))
            .filter(|(_, c)| *c > 0).map(|(l, c)| format!("{l} {c}")).collect();
        Some(format!("[exl3-serve] lane-pack idle{}: {} packed rounds since boot ({} on the real-q family; lane rounds in packs: {} real-q sampled, {} match-rule sampled, {} greedy; \
{:.2} rows and {:.2} tokens per packed round; \
host ms per round: drafts {:.2} + verify {:.2} (staging {:.2} + graph and sync {:.2} + readback {:.2}) + re-prime enqueue {:.2}{}); rounds by arm: packed {} serial {} shared_plain {}; \
lane rounds outside a pack: {}", self.tp.as_ref().map_or(String::new(), |t| format!(" (TP rank {})", t.rank)),
                     st.rounds, st.rq_rounds, st.lanes_rq, st.lanes_match, st.lanes_greedy, st.rows as f64 / n, st.toks as f64 / n,
                     st.ms_draft / n, st.ms_verify / n, st.ms_stage / n, st.ms_graph / n, st.ms_read / n, st.ms_finish / n,
                     if self.tp.is_some() { format!("; TP lockstep: pre-verify {:.2} ms (inside the drafts figure) + post-verify tp_round {:.2} ms (after the re-prime enqueue: NOT in the figures above) per round",
                                                    st.ms_pre / n, st.ms_post / n) } else { String::new() },
                     crate::metrics::lane_pack_rounds(0), crate::metrics::lane_pack_rounds(1), crate::metrics::lane_pack_rounds(2),
                     if skips.is_empty() { "none".to_string() } else { skips.join(", ") }))
    }

    /// S-C4XOVER: fold one round's (tokens, ms) into its arm's measured-rate window; every
    /// SPEC_ARM_FOLD_EVERY rounds the window rate folds into the arm's EWMA, tagged with the
    /// busiest lane count it saw. Rank-identical by construction: fed from the same agreed
    /// round-ms values the per-lane EMAs already use, on both TP ranks.
    fn arm_acc(&mut self, arm: usize, tok: f64, ms: f64) {
        if ms <= 0.0 || tok <= 0.0 { return; }
        let n = self.lanes.iter().filter(|l| l.is_some()).count() as u8;
        crate::metrics::spec_arm_charge(arm, ms, tok);
        self.spec_win_tok[arm] += tok;
        self.spec_win_ms[arm] += ms;
        self.spec_win_n[arm] = self.spec_win_n[arm].max(n);
        self.spec_win_acc[arm] += 1;
        if self.spec_win_acc[arm] >= SPEC_ARM_FOLD_EVERY {
            let rate = self.spec_win_tok[arm] / self.spec_win_ms[arm];
            self.spec_arm_rate[arm] = if self.spec_arm_rate[arm] > 0.0 {
                0.5 * self.spec_arm_rate[arm] + 0.5 * rate
            } else { rate };
            self.spec_arm_n[arm] = self.spec_win_n[arm];
            self.spec_arm_fold[arm] = self.steps as u64;
            self.spec_win_tok[arm] = 0.0;
            self.spec_win_ms[arm] = 0.0;
            self.spec_win_n[arm] = 0;
            self.spec_win_acc[arm] = 0;
            crate::metrics::spec_lanes_measured(arm, self.spec_arm_rate[arm] * 1e3);
        }
    }
    /// S-C4XOVER: is this arm's measured rate fresh at this busy-lane count?
    fn arm_fresh(&self, arm: usize, n: usize) -> bool {
        self.spec_arm_rate[arm] > 0.0
            && self.spec_arm_fold[arm].saturating_add(SPEC_ARM_FRESH_STEPS) > self.steps as u64
            && (self.spec_arm_n[arm] as i64 - n as i64).abs() <= SPEC_ARM_N_TOL
    }
    /// CF-P1e: do the busy lanes share one plain batched step this round (true) or run serial speculative rounds?
    /// `Auto` compares the estimated aggregate tokens/ms of the two arms (4% hysteresis): serial = the busy
    /// MTP-capable lanes' measured round speed (sum tokens / sum ms) plus one batched step for any MTP-off lanes;
    /// shared = n tokens per modelled shared-step time. A lane without an estimate yet falls back to a lane count of 4.
    fn use_shared_plain(&mut self, slots: &[usize]) -> bool {
        let n = slots.len();
        if n < 2 { return false; }
        match self.spec_policy {
            SpecLanes::Never => false,
            SpecLanes::Over(k) => n > k,
            SpecLanes::Auto => {
                // S-C4XOVER: measured delivered rates outrank the modelled estimates (both
                // were biased at four busy lanes: the serial cycle estimate ~+30% optimistic
                // vs delivery, the shared prior ~-18% pessimistic; together they pinned the
                // pick to the slower arm, ~13% at c4). When both arms have a fresh measured
                // rate at a compatible busy count, compare THOSE; the estimate path below is
                // the fallback for an arm that has not run recently.
                if self.arm_fresh(ARM_SERIAL, n) && self.arm_fresh(ARM_SHARED, n) {
                    let (s_rate, h_rate) = (self.spec_arm_rate[ARM_SERIAL], self.spec_arm_rate[ARM_SHARED]);
                    let pick = if h_rate > s_rate * 1.04 { true } else if s_rate > h_rate * 1.04 { false } else { self.shared_last };
                    if pick != self.shared_last && self.policy_logs < 64 {
                        self.policy_logs += 1;
                        crate::dreprintln!("[exl3-serve] spec-lanes auto (measured): {} with {n} busy lanes \
                           (serial speculation {:.0} tok/s delivered, one shared step {:.0} tok/s delivered)",
                            if pick { "SHARED plain step" } else { "serial speculation" }, s_rate * 1e3, h_rate * 1e3);
                    }
                    self.spec_margin = if s_rate.min(h_rate) > 0.0 { s_rate.max(h_rate) / s_rate.min(h_rate) - 1.0 } else { 0.0 };
                    self.spec_margin_n = n as u8;
                    crate::metrics::spec_lanes_decision(s_rate * 1e3, h_rate * 1e3, self.shared_scale, n);
                    if pick != self.shared_last {
                        crate::metrics::spec_lanes_switch(pick as usize); // 0 = shared, 1 = serial
                    }
                    self.shared_last = pick;
                    return pick;
                }
                let (mut tok, mut ms, mut capable, mut known) = (0.0f64, 0.0f64, 0usize, 0usize);
                for &s in slots {
                    let l = self.lanes[s].as_ref().unwrap();
                    if l.mtp.is_some() {
                        capable += 1;
                        if l.ema_tok > 0.0 && l.ema_ms > 0.0 { tok += l.ema_tok;
                            // P18 SPEC8K: price the serial arm by the measured CYCLE of the lane
                            // (serial_cycle_ms) instead of the ms of its own round, which for a packed
                            // lane is one SHARE of the verify. Both arms are then read, below, as the
                            // aggregate tokens per second that the n clients see.
                            ms += serial_cycle_ms(l.ema_ms, l.ema_wall_ms);
                            known += 1;
                        }
                    }
                }
                if capable == 0 { return false; }          // nothing speculates: the plain batch already runs
                if known < capable { return n > 4; }       // an estimate is missing: the fixed threshold
                let plain_only = n - capable;
                let shared_ms = self.shared_scale * shared_step_prior_ms(n);
                if plain_only > 0 {
                    tok += plain_only as f64;
                    ms += self.shared_scale * shared_step_prior_ms(plain_only);
                }
                // P18 SPEC8K: both sides are the AGGREGATE tokens per second that the n clients
                // see (n times a per-lane rate), which is the quantity the arm choice is about.
                let (serial_rate, shared_rate) = (n as f64 * tok / ms, n as f64 / shared_ms);
                let pick = if shared_rate > serial_rate * 1.04 { true } else if serial_rate > shared_rate * 1.04 { false } else { self.shared_last };
                if pick != self.shared_last && self.policy_logs < 64 {
                    self.policy_logs += 1;
                    crate::dreprintln!("[exl3-serve] spec-lanes auto: {} with {n} busy lanes (serial speculation {:.0} tok/s est, \
                               one shared step {:.0} tok/s est, scale {:.2}, serial cycle {:.1} ms/lane)",
                              if pick { "SHARED plain step" } else { "serial speculation" },
                              serial_rate * 1e3, shared_rate * 1e3, self.shared_scale, ms / n as f64);
                }
                // S-C4XOVER: the decision on /metrics (est gauges + flip counters; no behavior change).
                self.spec_est_pick = Some(pick);
                crate::metrics::spec_lanes_decision(serial_rate * 1e3, shared_rate * 1e3, self.shared_scale, n);
                if pick != self.shared_last {
                    crate::metrics::spec_lanes_switch(pick as usize); // 0 = shared, 1 = serial
                }
                self.shared_last = pick;
                pick
            }
        }
    }

    /// WP24: this lane's real-q draft sampler for its next MTP round — Some only for a SAMPLED
    /// lane (greedy stays on the bitwise match path) under --spec-sampling ratio (mode 2), or
    /// with --wp24-dump set (mode 1: today's drafts, top-32 lists recorded for rung 0).
    fn rq_stage(&self, s: usize) -> Option<crate::exl3_forward::RqStage> {
        let lane = self.lanes[s].as_ref()?;
        crate::exl3_forward::RqStage::for_lane(self.spec_ratio, self.draft_temp, lane.temperature,
                                               lane.top_p, lane.top_k, lane.min_p, lane.seed, lane.dctr)
    }

    /// One chain-verify round for one lane, with the §6 profitability control:
    /// every 128th round runs PLAIN as the control and the MTP path auto-disables
    /// (loudly) if its per-token cost loses to plain.
    /// ISSUES[94] DF2: `solo` = exactly ONE lane was active in this scheduler step (the serial
    /// arms pass slots.len() == 1; the packed/pair/shared-refresh arms pass false). Only a
    /// solo round may run the DFlash2 chain — the DF2 ring follows ONE token stream, so a
    /// multi-lane step keeps the pack-MTP machinery — and only when the ring is current for
    /// this lane (`nprev == p`): the prefill primed it and every DF2 round advanced it, but a
    /// lane that prefilled while another lane held the ring is stale until its next prefill;
    /// those rounds fall back to the pack MTP and count on `exl3_df2_fallback_mtp_total`.
    fn step_mtp(&mut self, s: usize, solo: bool) -> Result<()> {
        crate::metrics::lane_pack_round(1); // arm = serial (a lone speculative lane round)
        let k = self.mtp_k;
        let af = crate::exl3_forward::a_flags();
        let head = self.model.mtp.as_ref().unwrap();
        let (b, p, tap_row, rows, pen, remaining, lane_cost) = {
            let lane = self.lanes[s].as_ref().unwrap();
            (lane.last_tok as i32, lane.pos, lane.mtp.unwrap(), lane.samp_rows(k + 1), lane.pen,
             lane.max_new.saturating_sub(lane.generated),
             if lane.ema_tok > 0.0 { lane.ema_ms / lane.ema_tok } else { 0.0 })
        };
        if crate::exl3_forward::DF2_TRACE.load(std::sync::atomic::Ordering::Relaxed) {
            static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                let nv = self.df2.as_ref().map(|r| r.nprev());
                eprintln!("[df2trace] step_mtp s={s} solo={solo} p={p} nprev={nv:?} draft_conf={} df2_k={}",
                          self.draft_conf, self.df2_k);
            }
        }
        let xcheck = crate::exl3_forward::a_xcheck_on();
        if xcheck {
            self.model.tap_xcheck_verify(&self.sc, s, tap_row)?; // WP03: was this lane's tap clobbered?
        }
        // WP02 end-of-window clamp. Verify rows p..p+k (draft rows p..p+k-1, head re-prime
        // p+1..p+a) write the max_pos-row KV / indexer-key planes and read cos_tab: the last rounds
        // of a window-filling answer wrote up to p+k past them — p+7 at WP23's depth 7 — (K head 0
        // into head 1's rows 0-1, the last V head into the next slot or past the allocation).
        // (WP23 carried an identical copy of this clamp; p2 keeps this one.) p+k <= max_pos-2 also keeps every
        // written row's pooled-key block inside floor(max_pos/ratio). The remaining-budget half
        // drafts no token the request cannot emit (--a-remclamp=0 = off). k_eff == 0 -> a plain
        // step (with the budget clamp that is the request's last token).
        // (S-B9-CF-PACK4: the two clamps and the regime clamp below are `round_depth` — one
        // definition for this round and the packed arm's plan.)
        let (mut k_eff, kd) = self.round_depth(p, remaining);
        // S-B9-REL-QSAREGIME: a verify (and its draft chain) runs ONE attention regime, chosen from
        // its last row, so a round that straddles the sparse switch would put its rows at/below the
        // switch on the sparse kernels — not the bits a plain decode of those positions has (AGENTS
        // 2.4: column 0 of a verify is the decode's). Like the two clamps above, the round stops at
        // the switch: rows p..=qsa_limit only; at p == qsa_limit that is k_eff 0, a plain step. A
        // few short rounds once per request; a function of p alone (lockstep state under TP).
        {
            if kd < k_eff {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| crate::drprintln!("[exl3-serve] per-row attention regime: a speculative round at position {p} stops at the \
sparse switch (depth {k_eff} -> {kd}; --qsa-row-regime 0 = rounds may straddle it)"));
            }
            k_eff = kd;
        }
        // S-B9-REL-MTPREPROBE: a pending confirm forces ONE control round now (a fresh
        // plain sample at the CURRENT context) before an off may fire.
        let force_ctrl = {
            let l = self.lanes[s].as_mut().unwrap();
            std::mem::take(&mut l.force_control)
        };
        self.ctrl_round += 1;
        // WP03 cadence: 1/1024 while this request's MTP cost is clearly below plain (< 0.8x), else
        // 1/128. ema_plain stays context-local: it is refreshed only by control rounds at the
        // current context, never by a boot-time plain cost (under-priced at long context).
        let period = crate::lane_pack::ctrl_period(af.ctrl == 0, self.ema_plain, lane_cost);
        // (--a-ctrl off, gate diagnostics: no cadence control; see exl3_forward::AFlags)
        let control = (af.ctrl != 3 && self.ctrl_round % period == 0) || force_ctrl;
        let plain_step = control || k_eff == 0;
        // G2: stage this lane's sampler rows (verify rows 0..k, or row 0 of a plain step).
        // WP24: a sampled lane under --spec-sampling ratio stages real-q rows + the draft sampler
        // (plain steps stay plain); None = exactly set_sampling (the match rule, today).
        let rq = if plain_step { None } else { self.rq_stage(s) };
        let rq_ratio = rq.map_or(false, |r| r.mode == 2);
        // FLASHNEXT-DF2-B 2026-10-23: the greedy-lane divert to pack MTP that lived here
        // (interim for ISSUES work[112]: a graphed greedy DF2 draft overflowed the band
        // kernel's dynamic smem on replay) is GONE — the real fix sizes the smem to the
        // window bound (dflash2::band_smem); greedy lanes take the DF2 round like every other.
        self.model.set_spec_sampling(&mut self.sc, if plain_step { &rows[..1] } else { &rows }, rq)?;
        // WP15: the same penalty params on every row of the round (and the draft mirror's row 0).
        self.model.set_penalties(&mut self.sc, &vec![pen; if plain_step { 1 } else { k + 1 }])?;
        let t0 = std::time::Instant::now();
        let g0 = self.model.graph_count(); // WP23: a round that captures a graph is not cost-fit
        let mut df2_served = false;
        // EXL27D-reanchor (hoisted): a second request on the same server may start at a
        // position BEHIND the DF2 ring's frontier (its prefill came from the PREFIX
        // CACHE, so df2_prime_window never ran; df2_reprime only ADVANCES). Left alone,
        // every later request falls back to pack MTP forever (measured 2026-10-21:
        // request 2 = 0 DF2 rounds, 843 MTP drafts). Rewind the frontier to just PAST
        // this lane's position — `p`, not `p-1` — so the DF2 gate still misses THIS
        // round: the adaptive round that serves it refreshes the tap sink at the lane's
        // positions, and the Fix2 reprime advances the ring from there. The next round
        // is a DF2 round with fresh conditioning. Ring rows over a shared prefix are
        // exact (request 1 wrote them); a different prompt self-heals within a few
        // rounds. Counted on `exl3_df2_reanchor_total`.
        if !plain_step && solo && p >= 1
            && self.df2.as_ref().map_or(false, |r| r.nprev() + 1 > p)
        {
            // ISSUES 126: rewind to `p` so the gate (nprev == p, the trunk's invariant) matches THIS
            // round. Ring rows below p are the previous request's (exact over a shared prefix,
            // stale otherwise — ISSUES 114): that costs acceptance, never correctness (the verify
            // commits the target's own tokens). (The old one-behind convention rewound to p-1 and
            // admitted nprev+1 == p: the block then started one position early and the seam's tap
            // was never in context.)
            if let Some(r) = self.df2.as_mut() { r.rewind(p - 1); }
            df2_count_reanchor();
        }
        let (out, plain_ms, confs) = if plain_step {
            // plain round: consume b at p like a plain decode step. WP03: no draft pass runs, so
            // first write the head's row p KV-only (tap that predicted b, b), and after the step
            // snapshot its tap into row 0 of the lane's OWN window (sc.resid is shared: another
            // lane's round overwrites it before this lane's next round). --a-ctrl=old = neither.
            let ctrl_new = af.ctrl != 2;
            let tc = if ctrl_new {
                self.model.mtp_plain_prime(&mut self.sc, head, s, b, p, tap_row)?;
                self.model.dev().synchronize()?; // the control's plain-cost timing stays the step alone
                std::time::Instant::now()
            } else {
                t0
            };
            let ids = self.model.forward_step(&mut self.sc, &[b], &[p], &[s], None)?;
            let ms = tc.elapsed().as_secs_f64() * 1e3;
            if ctrl_new {
                self.model.tap_snapshot(&mut self.sc, s)?;
            }
            // synthesize an accept-0 round; e_0 becomes the new bonus
            (crate::exl3_forward::MtpRoundOut {
                a: 0,
                emitted: vec![ids[0]],
                pos_next: p + 1,
                last_tok: ids[0],
                // new: the snapshot row; old: the plain step leaves the true tap in sc.resid
                tap_row: if ctrl_new { 0 } else { usize::MAX },
                drafts: Vec::new(),
                diag_ms: 0.0,
            }, ms, Vec::new())
        } else if solo && self.df2.as_ref().is_some_and(|r| r.nprev() + 1 == p) {
            // ISSUES[94] DF2 (Fix1, 2026-10-16): the DFlash2 chain round owns the solo
            // speculative step whenever it is armed and its ring is fresh — INCLUDING under
            // `draft_conf > 0`: the WP23 adaptive draft-stop is pack-MTP-head machinery (it
            // reads the MTP head's per-position confidences through a keep hook the DF2 block
            // drafter has no analog of), so the adaptive branch keeps serving only the lanes
            // DF2 cannot (multi-lane rounds, stale ring). The DF2 round keeps the §6
            // profitability control (plain control rounds still measure it) and returns the
            // same MtpRoundOut, so `round_account` / `round_emit` are the MTP path's. The
            // plain CONTROL round above is untouched — the DF2 economics are still measured
            // against it.
            // ISSUES 127: ONE width for the whole DF2 round — the scheduler's k_eff (remaining budget, end of the KV
            // window, the QSA dense/sparse switch: a round never straddles it) caps --df2-k like it caps the MTP depth.
            // k_eff >= 1 here (k_eff == 0 is a plain step).
            let k_df2 = self.df2_k.min(crate::exl3_forward::MTP_MAX_K).min(k_eff);
            // EXL27D-RQ (item 1): a sampled lane re-stages its sampler rows at the DF2
            // verify width (k_df2+1; the shared staging above sized the MTP depth k_eff —
            // a mismatch would stage fewer ratio rows than the verify judges) and drafts
            // through the sampled selector walk. sel_seeds = rng(seed, DF2_SEL, j) per
            // position — the trunk lane's exact convention (batch.rs df2_lane_step_sample_rq);
            // tau = the staged draft temperature (--draft-temperature or the request's).
            let rq2 = if rq_ratio { self.rq_stage(s) } else { None };
            // ISSUES 127: sampler AND penalty rows at the DF2 verify width for greedy lanes too (they were staged at the
            // MTP depth k+1 above, so a wider DF2 verify judged its extra rows with stale rows).
            let (rows_df2, seed) = {
                let l = self.lanes[s].as_ref().unwrap();
                (l.samp_rows(k_df2 + 1), l.seed)
            };
            self.model.set_spec_sampling(&mut self.sc, &rows_df2, if rq_ratio { rq2 } else { rq })?;
            self.model.set_penalties(&mut self.sc, &vec![pen; k_df2 + 1])?;
            if rq_ratio {
                let seeds: Vec<u32> = (0..crate::dflash2::levels())
                    .map(|j| crate::batch::rng_u32(seed, crate::batch::RNG_DOM_DF2_SEL, j))
                    .collect();
                let tau = rq2.map(|r| r.tau).unwrap_or(1.0);
                // FLASHNEXT-DF2-117 (ISSUES 117): ONE sampled draft per round on every rank. Only
                // the head (or a single box) runs the selector walk; under TP the head ships its
                // tokens + candidate table to every node and the nodes adopt them instead of
                // drawing their own — a replicated walk over residuals that differ across ranks
                // in the low bits drafted differently and tripped the lockstep agree (TP=2 within
                // ~600 rounds, TP=4 after ~10k tokens on the owner's traffic). Greedy drafts never
                // needed this (a graph replay whose top-1 agreed over thousands of rounds).
                let draws_here = self.tp.as_ref().map_or(true, |t| t.rank == 0 || !t.has_ctl());
                let mine = if draws_here {
                    Some(self.model.df2_draft_sample(self.df2.as_mut().unwrap(), b as u32, &seeds, tau)?)
                } else { None };
                let sample = match self.tp.as_mut() {
                    Some(t) => match t.df2_draft(s, p, mine.as_ref())? {
                        Some(theirs) => theirs,
                        None => mine.expect("the head keeps the draft it shipped"),
                    },
                    None => mine.expect("single-box: the local draw"),
                };
                {
                    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                    if let Some(t) = self.tp.as_ref() {
                        if t.has_ctl() && !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                            crate::reprintln!("[exl3-serve] df2: DF2 drafts are head-authoritative under TP={} (greedy and sampled: the head's draft ships to every node per round; exl3_df2_shipped_drafts_total counts them — ISSUES 117)", t.world);
                        }
                    }
                }
                let o = self.model.df2_round(&mut self.sc, self.df2.as_mut().unwrap(), s, k_df2, b, p,
                                             Some((seeds.as_slice(), tau)), Some(sample),
                                             [t0, t0, t0])?;
                df2_count_round(o.a);
                df2_served = true;
                (o, 0.0, Vec::new())
            } else {
                // FLASHNEXT-DF2-117 (greedy twin): the greedy top-1 flips across ranks on near-ties as
                // well (TP=2 proof: one greedy round in ~1,500 after the sampled rounds were fixed), so
                // the greedy draft is head-authoritative too — tokens only, no candidate table.
                let draws_here = self.tp.as_ref().map_or(true, |t| t.rank == 0 || !t.has_ctl());
                let mine = if draws_here {
                    Some(crate::dflash2::round::Df2SampleOut {
                        tokens: self.model.df2_draft_greedy(self.df2.as_mut().unwrap(), b as u32)?,
                        q_rows: Vec::new(), cand_tok: Vec::new(), cand_q: Vec::new(),
                    })
                } else { None };
                let sample = match self.tp.as_mut() {
                    Some(t) => match t.df2_draft(s, p, mine.as_ref())? {
                        Some(theirs) => theirs,
                        None => mine.expect("the head keeps the draft it shipped"),
                    },
                    None => mine.expect("single-box: the local draw"),
                };
                let o = self.model.df2_round(&mut self.sc, self.df2.as_mut().unwrap(), s, k_df2, b, p,
                                             None, Some(sample), [t0, t0, t0])?;
                df2_count_round(o.a);
                df2_served = true;
                (o, 0.0, Vec::new())
            }
        } else if self.draft_conf > 0.0 {
            // API-parity DDS: draft until the running product of calibrated per-position
            // acceptance estimates falls below the target (the reference implementation's -dds -dc), then verify
            // only the proposed window; every verified position labels the calibrator.
            // WP23: the target carries the lane's cost guard, the estimate its draft's depth bucket
            // (both identity at k <= 5 unless forced: the K5 build's decisions exactly).
            let target = {
                let lane = self.lanes[s].as_mut().unwrap();
                let t = wp23_target(self.draft_conf, wp23_guard_on(k), lane.mtp_rounds,
                                    lane.wp23.fit.slope(), lane.ema_tok, lane.ema_ms);
                lane.wp23.target = t;
                t
            };
            let cal = &self.cal;
            let mut reach = 1.0f64;
            let mut keep = |i: usize, c: f32| { reach *= cal.estimate_at(i, c); reach >= target };
            // TP-D: agree the verify width + drafts before the verify launches (None at TP=1)
            let tp_on = self.tp.is_some();
            let tp = &mut self.tp;
            let mut pv = |w: usize, d: &[i32], n: usize| -> Result<()> {
                match tp.as_mut() { Some(t) => t.pre_verify(w, d, n), None => Ok(()) }
            };
            let pre: Option<&mut dyn FnMut(usize, &[i32], usize) -> Result<()>> =
                if tp_on && tp_preverify_on() { Some(&mut pv) } else { None };
            let (o, confs) = self.model.mtp_round_adaptive(&mut self.sc, head, s, k_eff, b, p, tap_row, &mut keep, pre)?;
            for (i, &c) in confs.iter().enumerate() {
                if i < o.a { self.cal.add_label(i, c, true); } else { self.cal.add_label(i, c, false); break; }
            }
            self.cal.decay_step();
            (o, 0.0, confs)
        } else {
            if solo && self.df2.is_some() {
                // DF2 engaged but the ring is stale for this lane (`nprev != p`: the lane
                // prefilled while another lane held the chain, or the artifact never primed
                // this window): the pack MTP serves the round and the fallback is COUNTED —
                // `exl3_df2_fallback_mtp_total` — never a silent skip.
                df2_count_fallback_mtp();
            }
            (self.model.mtp_round(&mut self.sc, head, s, k_eff, b, p, tap_row)?, 0.0, Vec::new())
        };
        // EXL27C-DF2-Fix2: after any NON-DF2 solo round (plain/control, adaptive or MTP
        // fallback), re-prime the DF2 ring from the taps that round's target forward captured
        // (every solo forward writes the staging sink), so the next solo round is DF2 again
        // instead of falling back forever. `df2_round` advanced the ring for its own rounds.
        if solo && self.df2.is_some() && !df2_served {
            if let Some(r) = self.df2.as_mut() {
                if let Err(e) = self.model.df2_reprime(r, out.a + 1) {
                    eprintln!("[exl3-serve] df2 reprime after fallback round failed: {e}");
                }
            }
        }
        // WP24: the dump / XCHECK host diagnostics are not round cost (they would trip the §6
        // per-request auto-off on a diagnostics run); 0 when off.
        let ms = (t0.elapsed().as_secs_f64() * 1e3 - out.diag_ms).max(0.0);
        // TP-C: lockstep — agree on (step, accept, verify width, emitted + drafts + confidences) and
        // take the HEAD's ms / plain ms / capture flag (identity at TP=1: no TP state).
        let captured_local = self.model.graph_count() != g0;
        // MTPAGREE1: --round-trace prints every speculative round's view on EVERY rank (and at TP=1), so two
        // runs or two ranks can be diffed field by field: tokens vs the per-position confidences (the f32
        // bits the TP mtp-round digest also hashes).
        if round_trace_on() {
            let r = self.tp.as_ref().map_or(0, |t| t.rank);
            eprintln!("[round-trace] rank={r} s={s} p={p} plain={plain_step} k_eff={k_eff} a={} drafts={:?} emitted={:?} confs_bits={:?}",
                      out.a, out.drafts, out.emitted, confs.iter().map(|c| format!("{:08x}", c.to_bits())).collect::<Vec<_>>());
        }
        let (ms, plain_ms, captured) = if self.tp.is_some() {
            let mut w: Vec<u32> = out.emitted.iter().map(|&x| x as u32).collect();
            w.extend(out.drafts.iter().map(|&x| x as u32));
            w.extend(confs.iter().map(|c| c.to_bits()));
            let m = if plain_step { 1 } else { out.drafts.len() + 1 };
            match self.tp_round(if plain_step { "plain-round" } else { "mtp-round" }, out.a, m, &w, ms, plain_ms, captured_local, false) {
                Ok(t) => t,
                Err(e) => {
                    // P7-TP2 bisect: both ranks print their round view at the mismatch; diff the
                    // logs to see whether the chain diverges at draft[0] (input/ring) or mid-chain.
                    eprintln!("[df2-tp2] p={:?} plain={} accept={} drafts={:?} emitted={:?} confs={:?}",
                              p, plain_step, out.a, out.drafts, out.emitted, confs);
                    // MTPAGREE1 (public v0.7.6 TP=2 report): the lane's sampler state next to the round
                    // view, so a rank diff names the subsystem (seed/counters = RNG stream; pen = the
                    // penalty rows; history tail = the committed stream the two ranks hold).
                    if let Some(lane) = self.lanes[s].as_ref() {
                        let ht: Vec<u32> = lane.history.iter().rev().take(8).copied().collect();
                        eprintln!("[df2-tp2] lane s={s} seed={} ctr={} dctr={} temp={} top_p={} top_k={} min_p={} pen={:?} \
                                   spec_ratio={} k_eff={} tap_row={} b={} pos={} generated={} history_tail(rev)={:?}",
                                  lane.seed, lane.ctr, lane.dctr, lane.temperature, lane.top_p, lane.top_k,
                                  lane.min_p, lane.pen, self.spec_ratio, k_eff, tap_row, b, lane.pos, lane.generated, ht);
                    }
                    return Err(e);
                }
            }
        } else {
            (ms, plain_ms, captured_local)
        };
        // S-B9-CF-PACK4: the round's bookkeeping and its emit loop are `round_account` /
        // `round_emit` (moved there unchanged, in order) so the packed arm runs the same code
        // per request.
        self.round_account(s, &out, ms, plain_ms, captured, control, plain_step, rq_ratio, xcheck, false)?;
        self.round_emit(s, &out, None)
    }

    /// One lane's round bookkeeping, exactly as the serial arm does it after its lockstep point:
    /// the tap cross-check record, the sampler counters, the per-request profitability control
    /// (cost EMAs, re-probe policy, control samples) and the WP23 width counts + cost fit.
    /// `ms` = this lane's round time (a packed round passes its SHARE, lane_pack::lane_share_ms).
    /// `packed` = a packed verify served the round (the request's arm trace records it).
    #[allow(clippy::too_many_arguments)]
    fn round_account(&mut self, s: usize, out: &crate::exl3_forward::MtpRoundOut, ms: f64, plain_ms: f64,
                     captured: bool, control: bool, plain_step: bool, rq_ratio: bool, xcheck: bool, packed: bool) -> Result<()> {
        let k = self.mtp_k;
        // P18 SPEC8K: sample the real round-to-round latency of the lane (queueing included) for the
        // load-adaptive choice of arm. Once per lane per round, here, whichever arm the round ran
        // (serial speculation, a packed pair, a shared plain step): this is the single point that every
        // arm reaches with the ms of its own round.
        if !captured {
            let now = std::time::Instant::now();
            let lane = self.lanes[s].as_mut().unwrap();
            if let Some(prev) = lane.last_wall {
                let wall = (now - prev).as_secs_f64() * 1e3;
                if wall > 0.0 && wall <= WALL_STALL_X * ms.max(1.0) {
                    lane.ema_wall_ms = if lane.ema_wall_ms <= 0.0 { wall } else {
                        WALL_EMA_W * lane.ema_wall_ms + (1.0 - WALL_EMA_W) * wall
                    };
                }
            }
            lane.last_wall = Some(now);
        }
        // S-C4XOVER: every round_account call is a serial-arm round (a lone speculative
        // round, a packed pair's lane share, or its plain control): fold its delivery
        // (emitted tokens, round ms) into the arm's measured rate.
        self.arm_acc(ARM_SERIAL, out.emitted.len() as f64, ms);
        if xcheck {
            self.model.tap_xcheck_record(&self.sc, s, out.tap_row)?;
        }
        // G2: advance the lane's sampler counter past every row this round consumed.
        {
            let lane = self.lanes[s].as_mut().unwrap();
            lane.arms.push(if packed { 'P' } else if control { 'c' } else if plain_step { 'p' } else { 'S' }, out.emitted.len());
            lane.ctr = lane.ctr.wrapping_add(if plain_step { 1 } else { (k + 1) as u32 });
            // WP24: past every draft counter this round could use (passes 0..k_eff-1 incl. a
            // HOST speculative pass the DDS then stopped before; k_eff <= k)
            if rq_ratio { lane.dctr = lane.dctr.wrapping_add(k as u32); }
            // WP24 gate bookkeeping (None in the server): round anatomy per arm + the request's
            // first-round draft d_0 (the negative control: d_0 ~ q', a law != p)
            if let Some(g) = self.gate.as_mut() {
                if !plain_step {
                    g.rounds += 1;
                    g.drafted += out.drafts.len() as u64;
                    g.accepted += out.a as u64;
                    g.ratio_rounds += rq_ratio as u64;
                    g.ms += ms;
                } else if control {
                    g.ctrl += 1;
                } else {
                    g.plain += 1;
                }
                if lane.gate_first {
                    if let Some(&d0) = out.drafts.first() { g.d0.push(d0); }
                }
            }
            lane.gate_first = false;
        }
        // API-parity G1: §6 profitability, scoped to THIS request. The plain reference is
        // an engine property (control rounds, persists); the MTP side is this request's
        // content. A request whose drafts do not pay finishes plain; nothing global dies
        // (the old process-lifetime `mtp_dead` turned one prose request into ~30 tok/s
        // for every later request).
        let toks = (out.a + 1) as f64;
        if control {
            self.ema_plain = if self.ema_plain <= 0.0 { plain_ms } else {
                0.875 * self.ema_plain + 0.125 * plain_ms
            };
            if let Some(lane) = self.lanes[s].as_mut() {
                lane.st_ctrl += 1;
                // S-B9: a pending confirm collects this fresh sample (current context)
                if lane.reprobe.confirm_pending {
                    lane.reprobe.confirm_sample(plain_ms);
                }
            }
        } else if plain_step {
            // WP02: the window/budget-clamped last step (k_eff == 0) — a plain step, not a round
            if let Some(lane) = self.lanes[s].as_mut() { lane.st_plain += 1; }
        } else {
            let ema_plain = self.ema_plain;
            let reprobe_on = self.reprobe_on;
            let cfg = self.reprobe_cfg;
            // legacy mode keeps today's estimator weight so --mtp-reprobe=off is exact
            let ema_w = if reprobe_on { cfg.ema_w } else { 1.0 / 16.0 };
            let lane = self.lanes[s].as_mut().unwrap();
            lane.mtp_rounds += 1;
            lane.st_rounds += 1;
            lane.st_drafted += out.drafts.len();
            lane.st_accepted += out.a;
            lane.st_ms += ms;
            // The first `warmup` rounds of a slot can include graph capture; keep them
            // out of the cost. --mtp-reprobe=off keeps the exact legacy estimator
            // (one-round seed + 1/16 EMA, hardcoded 4-round guard: bit-for-bit the
            // pre-branch behaviour). The re-probe policy blends via
            // mtp_reprobe::blend_cost_ema (W1.1): a RUNNING MEAN over the first 64
            // post-reset rounds and capture rounds excluded entirely — one
            // capture-poisoned or 1-token round must not steer a probe verdict
            // (at 1/64 a seed still carries 0.79-0.62 of itself at rounds 16-31).
            if reprobe_on {
                if lane.mtp_rounds > cfg.warmup {
                    crate::mtp_reprobe::blend_cost_ema(
                        &mut lane.ema_ms, &mut lane.ema_tok, &mut lane.ema_n,
                        ms, toks, captured, (cfg.ema_w.recip()) as usize, cfg.ema_w);
                }
            } else if lane.mtp_rounds > 4 {
                if lane.ema_tok <= 0.0 {
                    lane.ema_ms = ms;
                    lane.ema_tok = toks;
                } else {
                    lane.ema_ms = (1.0 - ema_w) * lane.ema_ms + ema_w * ms;
                    lane.ema_tok = (1.0 - ema_w) * lane.ema_tok + ema_w * toks;
                }
            }
            let cost = if lane.ema_tok > 0.0 { lane.ema_ms / lane.ema_tok } else { 0.0 };
            if !reprobe_on {
                // --mtp-reprobe=off: the previous one-way rule, exactly (2% margin, no confirm,
                // no re-arm — the frozen pre-fix behaviour for A/B and for users who want it).
                if lane.mtp_rounds >= 64 && ema_plain > 0.0 && cost > ema_plain * 1.02 {
                    crate::dreprintln!("[exl3-serve] MTP off for this request (slot {s}): {:.2} ms/tok ({:.1} ms/round, \
                           {:.2} tok/round) vs plain {:.2} after {} rounds (AGENTS §6, per-request)",
                           cost, lane.ema_ms, lane.ema_tok, ema_plain, lane.mtp_rounds);
                    // the lane is at a clean seam (round committed, last_tok unconsumed at
                    // lane.pos — the same state a control round leaves), so plain can take over.
                    lane.mtp = None;
                }
            } else {
                // S-B9-REL-MTPREPROBE: the §6 decision is the pure policy's (margin 10%, a
                // 1/64-round window that spans local troughs, and a plain baseline that must
                // survive a fresh confirm at the CURRENT context — see src/mtp_reprobe.rs).
                // The round left the lane at a clean seam (last_tok unconsumed at lane.pos,
                // the round's tap saved in out.tap_row), so an off can prime from here.
                use crate::mtp_reprobe::RoundVerdict as RV;
                let plain_ref = lane.reprobe.plain_reference(ema_plain);
                match lane.reprobe.observe_round(cost, ema_plain, &cfg) {
                    RV::Continue => {}
                    RV::ConfirmControl => {
                        // crossed the margin: demand ONE fresh plain sample before deciding
                        lane.force_control = true;
                    }
                    RV::Off => {
                        crate::dreprintln!("[exl3-serve] MTP off for this request (slot {s}): {:.2} ms/tok ({:.1} ms/round, \
{:.2} tok/round) vs plain {:.2} after {} rounds (AGENTS §6; re-probe in {} plain steps)",
                                   cost, lane.ema_ms, lane.ema_tok, plain_ref, lane.mtp_rounds, cfg.backoff_base);
                        lane.mtp = None;
                        lane.reprobe.went_off(out.tap_row, false, &cfg);
                    }
                    RV::ProbeFailed => {
                        let next_b = (lane.reprobe.backoff.saturating_mul(2)).min(cfg.backoff_cap).max(cfg.backoff_base);
                        crate::reprintln!("[exl3-serve] MTP re-probe FAILED (slot {s}): {:.2} ms/tok vs plain {:.2} after {} \
probe rounds; off again (next probe in {next_b} plain steps)", cost, plain_ref, lane.reprobe.probe_rounds);
                        lane.mtp = None;
                        lane.reprobe.went_off(out.tap_row, true, &cfg);
                    }
                    RV::ProbeSucceeded => {
                        crate::dreprintln!("[exl3-serve] MTP re-probe OK (slot {s}): stayed on {} probe rounds at {:.2} \
ms/tok vs plain {:.2}; backoff reset to {}", cfg.probe_window, cost, plain_ref, cfg.backoff_base);
                        lane.reprobe.probe_ok(&cfg);
                    }
                }
            }
        }
        // WP23: rounds by verified draft width, and the cost fit (ms on drafts) — MTP rounds of a
        // DDS lane past the EMAs' 4-round warm-up, minus any round that captured a graph (a lazy
        // first-use capture) or ran > 3x the lane's mean round (a host stall): outliers, not cost.
        if !plain_step {
            let dds = self.draft_conf > 0.0;
            if let Some(lane) = self.lanes[s].as_mut() {
                let w = out.drafts.len();
                if let Some(c) = lane.wp23.widths.get_mut(w) { *c += 1; }
                if dds && lane.mtp_rounds > 4 && !captured && (lane.ema_ms <= 0.0 || ms <= 3.0 * lane.ema_ms) {
                    lane.wp23.fit.add(w as f64, ms);
                }
            }
        }
        Ok(())
    }

    /// Emit one round's tokens for the lane in slot `s` (stop at EOS / max_new / a loop / a gone
    /// client mid-list; a finishing lane frees its slot). `packed_row0` = Some(the lane's first
    /// row in the packed verify) when the round's logits rows live in the packed scratch.
    fn round_emit(&mut self, s: usize, out: &crate::exl3_forward::MtpRoundOut, packed_row0: Option<usize>) -> Result<()> {
        // ---- emit the round's tokens (stop at EOS / max_new mid-list)
        for (j, &t32) in out.emitted.iter().enumerate() {
            let t = t32 as u32;
            let (finish, reason, generated) = {
                let lane = self.lanes[s].as_mut().unwrap();
                lane.last_tok = t;
                lane.pos += 1;
                lane.generated += 1;
                lane.history.push(t);
                if lane.first_toks.len() < FIRST_TOKS { lane.first_toks.push(t); }
                let fin = lane.generated >= lane.max_new
                    || (self.eos.contains(&t) && !lane.ignore_eos && lane.generated >= lane.min_new);
                let rsn = if lane.generated >= lane.max_new { "length" }
                    else if fin { "stop" } else { "" };
                // WP08: a loop ends the round mid-list like a stop token (rest dropped).
                if !fin && lane.loop_feed(t) {
                    (true, "loop_detected".to_string(), lane.generated)
                } else {
                    (fin, rsn.to_string(), lane.generated)
                }
            };
            if let Some(lane) = self.lanes[s].as_mut() {
                if lane.mtp.is_some() {
                    lane.mtp = Some(out.tap_row);
                }
            }
            // WP02: a failed send = the client is gone — cancel now, free the slot. TP-D: under TP
            // the head's next sweep ships the cancel instead (rank-local drops would desync).
            let sent = self.lanes[s].as_ref().map_or(false, |lane| lane.tx.send(crate::batch::TokEvent::Tok(t)).is_ok());
            if !sent && self.tp.is_none() {
                if let Some(lane) = self.take_lane(s, false) {
                    lane.log_cancel(s);
                }
                return Ok(());
            }
            if finish {
                self.publish_kv(); // issue #24: the lane's final position (the peaks include the round that ends the request)
                // item 5(a): emitted[j] is the choice of the round's logits row j (verify row j, or
                // row 0 of a plain step) — still in sc.logits: the re-prime after a verify is KV-only
                if reason == "stop" { self.send_stop_info(s, packed_row0.map_or(j, |r0| r0 + j), t, packed_row0.is_some()); }
                if let Some(lane) = self.take_lane(s, crate::lane_reserve::normal_finish(&reason)) {
                    if reason == "loop_detected" { lane.log_loop(s, &self.tok); }
                    lane.log_stats(s, &reason);
                    self.model.dump_expert_hist(); // TP-I #6 diagnostic (no-op unless --tp-ep-hist)
                    self.model.dump_esel_hist(); // S-A3-o diagnostic (no-op unless --exl3-esel-hist)
                    self.model.dump_route_log(); // CF-P1e step 0 diagnostic (no-op unless --exl3-route-log)
                    let _ = lane.tx.send(crate::batch::TokEvent::Finish { reason });
                    // after the Finish: the DHEADP line may sync the device once (penalized requests)
                    self.model.dhead_request_report(); // DHEADP fallback line (penalized requests) + DHEAD XCHECK lines (when on)
                }
                return Ok(()); // lane finished; remaining emitted tokens dropped
            }
        }
        // consumed all emitted. S-A3-h: the emit loop's per-token pos increments
        // leave lane.pos = out.pos_next = the position of last_tok — mtp_round's
        // next-p contract (identical to bench_mtp_exl3's `p = out.pos_next`, whose
        // chain is the LOSSLESS reference). The old `lane.pos = out.pos_next - 1`
        // clobber shifted every round 2+ one position back (RoPE/state seam) and
        // produced the residual bug-5 wrong-but-consistent chains.
        Ok(())
    }

    /// S-B9-REL-SMALLFIXES item 5(a): the lane in slot `s` is finishing on the stop token `tok`,
    /// the choice of logits row `row` of the step that just ran. Reads that ONE row back and sends
    /// the token's probability and rank ahead of the Finish (`stop_p` / `stop_rank` of the HTTP
    /// side's `[gen]` line). Cost: one host readback of vocab f16 values plus one host pass over
    /// them, on a request's FINISHING step and only when it ends on a stop token; no step of a
    /// running request pays anything. TP=1 only: under TP the greedy head is vocab-parallel and no
    /// rank holds a full logits row (the line says `na`). A greedy row checks itself — its token
    /// must be the row's top logit (`stop_info_checked`); if it is not, the row read is not the row
    /// that chose the token, nothing is sent and one line per boot says so.
    fn send_stop_info(&self, s: usize, row: usize, tok: u32, packed: bool) {
        if self.tp.is_some() { return; }
        let Some(lane) = self.lanes[s].as_ref() else { return };
        let samp = crate::exl3_forward::SampParams::sampled(lane.temperature, lane.top_p, lane.top_k, lane.min_p, 0, 0);
        // S-B9-CF-PACK4: a packed round's logits rows are in the packed scratch (row = row0 + j)
        let sc = match (packed, self.scp.as_ref()) { (true, Some(scp)) => scp, _ => &self.sc };
        let stats = match self.model.logits_row(sc, row) {
            Ok(r) => crate::exl3_forward::stop_token_stats(&r, tok as usize, samp.as_ref()),
            Err(_) => None,
        };
        match stop_info_checked(stats, samp.is_none()) {
            Some((p, rank)) => { let _ = lane.tx.send(TokEvent::StopInfo { p: p as f32, rank, sampled: samp.is_some() }); }
            None => {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| crate::drprintln!("[exl3-serve] stop_p unavailable for a request (slot {s}, logits row {row}, token {tok}, {:?}): \
the [gen] line says stop_p=na", stats));
            }
        }
    }

    /// Plain batched decode for the given slots (the pre-f-d step() body).
    /// CF-P1e: fold a measured shared-round time (ms, lockstep-adopted under TP) into the model's learned scale.
    /// The first rounds of a width can include a graph capture, so an outlier above 3x the model is ignored.
    fn learn_shared_ms(&mut self, n: usize, ms: f64) {
        if n < 2 { return; }
        let r = ms / shared_step_prior_ms(n);
        if !(0.05..3.0).contains(&r) { return; }
        self.shared_scale = if self.shared_streak <= 1 && self.shared_scale == 1.0 { r } else { 0.9 * self.shared_scale + 0.1 * r };
    }

    /// S-B9-REL-QSAREGIME: `slots` with the lanes at/below the sparse switch first (see step_plain).
    fn regime_ordered(&self, slots: &[usize]) -> Vec<usize> {
        let lim = regime_switch(self.model.qsa_live(usize::MAX), crate::exl3_forward::row_regime_on(), self.model.qsa_limit());
        regime_ordered_slots(slots, |s| self.lanes[s].as_ref().map_or(0, |l| l.pos), lim)
    }

    /// `sync_head` (CF-P1e): the lanes are MTP-capable but this round is a shared plain step — keep each lane's draft
    /// head exactly as a control round does (write the head's row p KV-only from the lane's saved tap BEFORE the
    /// step; snapshot the step's tap row into the lane's window AFTER it), so its next speculative round is valid.
    fn step_plain(&mut self, slots: &[usize], sync_head: bool) -> Result<()> {
        // S-B9-REL-QSAREGIME: lanes at/below the sparse-attention switch first (stable; the identity
        // unless the lanes straddle the switch), so forward_step keeps their rows on the dense
        // kernels when a longer lane shares the step — a lane's bits must not depend on its
        // neighbours. Everything below indexes rows through this one list. Under TP every rank
        // orders from the same lockstep positions.
        let ordered = self.regime_ordered(slots);
        let slots: &[usize] = &ordered;
        // S-B9-REL-MTPREPROBE: an MTP-off lane under the re-probe policy keeps the SAME
        // head sync an armed lane gets — prime its row p from the saved tap BEFORE the step
        // (the 1-layer head row a control round would write) and snapshot the step's tap
        // AFTER it — so its head KV stays contiguous and its re-arm is exactly a
        // control-round seam. --mtp-reprobe=off primes only when sync_head, as before.
        let reprobe_primes = self.reprobe_on && self.model.mtp.is_some();
        if sync_head || reprobe_primes {
            let model = self.model.clone();
            if let Some(head) = model.mtp.as_ref() {
                // S-C4PRIMEA: collect the step's primeable lanes. When EVERY lane has a real
                // tap row and there are >= 2, ONE batched call (xq_gather_tap_rows + a single
                // re-prime pass at m = lanes + xq_attn_kv_prefill_pairs) replaces the per-lane
                // mtp_plain_prime loop. Any other shape (a sentinel tap, a skipped lane, one
                // lane) runs the VERBATIM legacy loop below — byte-identical to the old code,
                // which is also what the control site and solo re-probe steps keep running.
                let mut lanes: Vec<(usize, i32, usize, usize)> = Vec::with_capacity(slots.len());
                let mut batch_ok = prime_batch_on();
                for &s in slots {
                    let (b, p, tr) = { let l = self.lanes[s].as_ref().unwrap(); (l.last_tok as i32, l.pos, l.mtp) };
                    let tr = tr.or({
                        // an off rearmable lane: its saved tap row (0 once a primed step snapshotted)
                        let l = self.lanes[s].as_ref().unwrap();
                        if l.reprobe.keep_primed() { Some(l.reprobe.prime_tap_row) } else { None }
                    });
                    match tr {
                        Some(tap_row) if tap_row != usize::MAX => lanes.push((s, b, p, tap_row)),
                        _ => batch_ok = false,
                    }
                }
                batch_ok = batch_ok && lanes.len() >= 2 && lanes.len() <= 16; // re-prime width cap; mtp_plain_prime_batched re-checks vs the scratch width
                if batch_ok {
                    if prime_batch_xcheck_take() {
                        // S-C4PRIMEA: ONE-TIME-per-boot GPU identity gate on the real path:
                        // the legacy per-lane head-KV rows vs the batched bytes of the same rows.
                        for &(s, b, p, tr) in &lanes { model.mtp_plain_prime(&mut self.sc, head, s, b, p, tr)?; }
                        let want = model.head_kv_row_bytes(&mut self.sc, head, &lanes)?;
                        model.mtp_plain_prime_batched(&mut self.sc, head, &lanes)?;
                        let got = model.head_kv_row_bytes(&mut self.sc, head, &lanes)?;
                        let same = want == got;
                        eprintln!("[prime-batch] XCHECK {}: {} lanes, {} head-KV row bytes, per-lane == batched",
                                  if same { "PASS" } else { "FAIL" }, lanes.len(), got.len());
                        anyhow::ensure!(same, "prime-batch identity xcheck FAILED (first batched call this boot)");
                    } else {
                        model.mtp_plain_prime_batched(&mut self.sc, head, &lanes)?;
                    }
                    crate::metrics::prime_batch_round(lanes.len() as u64);
                    prime_batch_boot_log(lanes.len());
                } else {
                    for &s in slots {
                        let (b, p, tr) = { let l = self.lanes[s].as_ref().unwrap(); (l.last_tok as i32, l.pos, l.mtp) };
                        let tr = tr.or({
                            // an off rearmable lane: its saved tap row (0 once a primed step snapshotted)
                            let l = self.lanes[s].as_ref().unwrap();
                            if l.reprobe.keep_primed() { Some(l.reprobe.prime_tap_row) } else { None }
                        });
                        if let Some(tap_row) = tr {
                            model.mtp_plain_prime(&mut self.sc, head, s, b, p, tap_row)?;
                        }
                    }
                }
            }
        }
        // S-B9-REL-MTPREPROBE2 W1.3: when this shared step is exactly one slot and that
        // lane is an off re-probe lane, its plain sample must time the STEP ALONE — the
        // same discipline as a control round (prime, synchronize, then time). Without
        // the sync the async mtp_plain_prime's head GPU time lands inside the sample
        // (stream-ordered before forward_step's readback), inflating off_plain_ema and
        // biasing the policy toward staying ON. Multi-lane shared steps keep the CF-P1e
        // costing (primes + step): their number feeds the shared estimate, not the policy.
        let solo_off_sample = self.reprobe_on && slots.len() == 1
            && self.lanes[slots[0]].as_ref().map_or(false, |l| l.reprobe.keep_primed());
        if (sync_head || reprobe_primes) && solo_off_sample {
            self.model.dev().synchronize()?;
        }
        let t_round = std::time::Instant::now();   // CF-P1e: the shared round's cost = head primes + the step
        let toks: Vec<i32> = slots.iter()
            .map(|&s| self.lanes[s].as_ref().unwrap().last_tok as i32)
            .collect();
        let poss: Vec<usize> = slots.iter()
            .map(|&s| self.lanes[s].as_ref().unwrap().pos)
            .collect();
        // API-parity G2: sampled lanes sample ON DEVICE (xq_sample_rows after the argmax;
        // the same sampler the MTP verify rows use) — no 248K-float host readback per token.
        let rows: Vec<Option<crate::exl3_forward::SampParams>> = slots.iter()
            .map(|&s| self.lanes[s].as_ref().unwrap().samp_rows(1)[0])
            .collect();
        self.model.set_sampling(&mut self.sc, &rows)?;
        let pens: Vec<Option<crate::exl3_forward::PenParams>> = slots.iter()
            .map(|&s| self.lanes[s].as_ref().unwrap().pen)
            .collect();
        self.model.set_penalties(&mut self.sc, &pens)?;
        // W2-EXL3: per-row schema masks for the armed schema lanes (nothing staged — and no
        // mask node in the step's graph — when no lane in this step carries a live constraint).
        {
            let masks: Vec<Option<std::sync::Arc<Vec<u32>>>> = slots.iter().map(|&s| {
                let l = self.lanes[s].as_ref().unwrap();
                match (&l.schema, l.schema_armed) {
                    (Some(sm), true) => Some(sm.allowed(l.schema_state)),
                    _ => None,
                }
            }).collect();
            let rows: Vec<Option<&[u32]>> = masks.iter().map(|m| m.as_deref().map(|v| &v[..])).collect();
            self.model.set_vmask(&mut self.sc, &rows)?;
        }
        let ids = self.model.forward_step(&mut self.sc, &toks, &poss, slots, None)?;
        // S-B9: one agreed ms for the step (the HEAD's under TP — a rank-local clock must
        // never drive a decision), reused by the shared estimate and the off-lane sampler.
        let own_ms = t_round.elapsed().as_secs_f64() * 1e3;
        let plain_ms_head;
        if self.tp.is_some() {
            // TP-C: lockstep on the batched plain step's emitted ids (no wall-clock decision here)
            let w: Vec<u32> = ids.iter().map(|&x| x as u32).collect();
            let (ms, _, _) = self.tp_round("plain-step", 0, slots.len(), &w, own_ms, 0.0, false, false)?;
            if sync_head {
                self.learn_shared_ms(slots.len(), ms);
                // S-C4XOVER: a shared plain step delivers one token to every busy lane;
                // fold it into the shared arm's measured rate (2+ lanes: a lone-lane
                // sync step is a reprobe control, not the shared arm at work).
                if slots.len() >= 2 { self.arm_acc(ARM_SHARED, slots.len() as f64, ms); }
            }
            plain_ms_head = ms;
        } else {
            if sync_head {
                self.learn_shared_ms(slots.len(), own_ms);
                if slots.len() >= 2 { self.arm_acc(ARM_SHARED, slots.len() as f64, own_ms); }
            }
            plain_ms_head = own_ms;
        }
        if let Some(g) = self.gate.as_mut() { g.plain += slots.len() as u64; }
        if sync_head || self.reprobe_on {
            let model = self.model.clone();
            // S-B9-REL-MTPREPROBE2 W1.5: --a-xcheck must see these plain-step snapshots
            // too (a control round records its row the same way) — without the record
            // the FIRST speculative round after a re-arm verifies against the lane's
            // last RECORDED (verify-era) row and logs a false "clobbered" line.
            let xcheck = crate::exl3_forward::a_xcheck_on();
            for (j, &s) in slots.iter().enumerate() {
                if self.lanes[s].as_ref().map_or(false, |l| l.mtp.is_some()) {
                    model.tap_snapshot_row(&mut self.sc, s, j)?;
                    if xcheck {
                        if let Err(e) = model.tap_xcheck_record(&self.sc, s, 0) {
                            eprintln!("[a-xcheck] record failed: {e:#}");
                        }
                    }
                    self.lanes[s].as_mut().unwrap().mtp = Some(0);
                } else if self.reprobe_on && self.lanes[s].as_ref().map_or(false, |l| l.reprobe.keep_primed()) {
                    // S-B9: keep the off lane's tap fresh (row 0) so its next head prime is valid
                    model.tap_snapshot_row(&mut self.sc, s, j)?;
                    if xcheck {
                        if let Err(e) = model.tap_xcheck_record(&self.sc, s, 0) {
                            eprintln!("[a-xcheck] record failed: {e:#}");
                        }
                    }
                    self.lanes[s].as_mut().unwrap().reprobe.prime_tap_row = 0;
                }
            }
            // S-B9: off lanes' plain steps count toward the re-probe backoff. A lane ALONE
            // in this batch also yields an exact per-token plain cost at the CURRENT
            // context (the engine's ema_plain can be stale-low: plain grows with context)
            // — it raises the probe's plain reference, so probes cannot fail on a stale
            // baseline.
            if self.reprobe_on {
                for &s in slots {
                    let off = self.lanes[s].as_ref()
                        .map_or(false, |l| l.mtp.is_none() && l.reprobe.keep_primed());
                    if off {
                        let sample = if slots.len() == 1 && plain_ms_head > 0.0 { plain_ms_head } else { 0.0 };
                        self.lanes[s].as_mut().unwrap().reprobe.observe_plain(sample, &self.reprobe_cfg);
                    }
                }
            }
        }

        for (j, &s) in slots.iter().enumerate() {
            let t = {
                let lane = self.lanes[s].as_mut().unwrap();
                let next = ids[j] as u32;
                lane.ctr = lane.ctr.wrapping_add(1);
                lane.st_plain += 1;
                lane.arms.push('p', 1);
                if lane.first_toks.len() < FIRST_TOKS { lane.first_toks.push(next); }
                lane.last_tok = next;
                lane.pos += 1;
                lane.generated += 1;
                lane.history.push(next);
                if lane.history.len() > 256 { lane.history.drain(0..128); }
                next
            };
            let (finish, reason) = {
                let lane = self.lanes[s].as_mut().unwrap();
                if lane.generated >= lane.max_new {
                    (true, "length")
                } else if self.eos.contains(&t) && !lane.ignore_eos && lane.generated >= lane.min_new {
                    (true, "stop")
                } else if let Some(sm) = lane.schema.clone().filter(|_| lane.schema_armed) {
                    // W2-EXL3: a schema turn ends at the end of its JSON document. The loop
                    // detector does not run on an armed lane: the mask bounds the output, and a
                    // valid document may legitimately repeat (a long array of equal values).
                    match schema_feed(&sm, &mut lane.schema_state, &mut lane.schema_armed, t) {
                        SchemaStep::Continue => (false, ""),
                        SchemaStep::Complete => (true, "stop"),
                        SchemaStep::Invalid => {
                            crate::reprintln!("[schema] slot {s}: token {t} leaves no valid continuation \
                                               — ending the turn with an error");
                            (true, SCHEMA_DEAD_END)
                        }
                    }
                } else if lane.loop_feed(t) {
                    (true, "loop_detected") // WP08
                } else if let Some(sm) = lane.schema.clone() {
                    // not armed yet (reasoning): this token may be the one that arms it
                    schema_feed(&sm, &mut lane.schema_state, &mut lane.schema_armed, t);
                    (false, "")
                } else {
                    (false, "")
                }
            };
            let tp_on = self.tp.is_some();
            // item 5(a): ids[j] is the choice of the step's logits row j
            if finish && reason == "stop" { self.send_stop_info(s, j, t, false); }
            if finish { self.publish_kv(); } // issue #24: the lane's final position (the peaks include the round that ends the request)
            let lane = self.lanes[s].as_mut().unwrap();
            // TP-D: under TP a failed send is not a cancel here (the head's next sweep ships it)
            if lane.tx.send(TokEvent::Tok(t)).is_err() && !tp_on {
                // WP02: the client is gone — cancel now, free the slot
                lane.log_cancel(s);
                self.release_lane(s, false);
                continue;
            }
            if finish {
                if reason == "loop_detected" { lane.log_loop(s, &self.tok); }
                lane.log_stats(s, reason);
                self.model.dump_expert_hist(); // TP-I #6 diagnostic (no-op unless --tp-ep-hist)
                self.model.dump_esel_hist(); // S-A3-o diagnostic (no-op unless --exl3-esel-hist)
                self.model.dump_route_log(); // CF-P1e step 0 diagnostic (no-op unless --exl3-route-log)
                let _ = lane.tx.send(TokEvent::Finish { reason: reason.to_string() });
                self.release_lane(s, crate::lane_reserve::normal_finish(reason));
                self.model.dhead_request_report(); // DHEADP fallback line (penalized requests) + DHEAD XCHECK lines (when on)
            }
        }
        Ok(())
    }

    fn run(mut self, mut rx: UnboundedReceiver<BatchRequest>) {
        // HOST / RO-7: the mask this thread inherited from `pub fn run`'s pin (--cpu-affinity)
        if let Ok(m) = crate::cpu_affinity::current_thread_affinity() {
            println!("[exl3-serve] scheduler thread cpu mask: {}", crate::cpu_affinity::format_cpu_list(&m));
        }
        // Requests beyond the lane count WAIT in arrival order (the standard server contract);
        // they used to be refused as "server busy", which reached a non-streaming client as an
        // empty HTTP 200. A client that hangs up while queued is dropped before admission.
        let mut waiting: std::collections::VecDeque<BatchRequest> = std::collections::VecDeque::new();
        let mut rots_printed = 0usize;
        let mut mixed_printed = 0u64;
        loop {
            if self.fatal.is_some() {
                return self.die(waiting, rx);
            }
            if waiting.is_empty() && self.lanes.iter().all(|l| l.is_none()) {
                // Clean idle point: report the TOTAL number of fcfs rotations this scheduler made
                // (the per-rotation log caps at 16 lines; G1 needs the true count).
                if self.lane_order == LaneOrder::Fcfs && self.fcfs_rots > rots_printed {
                    crate::drprintln!("[exl3-serve] lane-order fcfs idle: {rots_printed}->{} total rotations", self.fcfs_rots);
                    rots_printed = self.fcfs_rots;
                }
                // 9a(iii): the lazily captured mixed-step graphs, whenever their number grew
                if let Some(line) = mixed_graphs_idle_line(mixed_printed, crate::metrics::mixed_step_graphs(), self.width) {
                    crate::drprintln!("{line}");
                    mixed_printed = crate::metrics::mixed_step_graphs();
                }
                self.publish_kv(); // issue #24: the idle value (lanes freed outside a step included)
                if let Some(line) = self.pack_idle_line() { crate::drprintln!("{line}"); } // S-B9-CF-PACK4
                crate::metrics::sched_idle(); // H3: about to block for work (busy must not stick at 1)
                match rx.blocking_recv() {
                    Some(req) => waiting.push_back(req),
                    None => break,
                }
            }
            while let Ok(req) = rx.try_recv() {
                // MAXBATCH1: an arrival beyond what the cap can run at once — every lane busy,
                // or the queue already holds the overflow (a burst on an idle box lands here
                // too: all 12 drain before admission runs). Count it (the cap's engagement
                // counter; the 2026-10-21 soak showed the lanes-already-full-only test misses
                // exactly those bursts).
                if self.lanes.iter().filter(|l| l.is_some()).count() + waiting.len() >= self.lanes.len() {
                    crate::metrics::over_cap_queued();
                }
                waiting.push_back(req);
            }
            while self.fatal.is_none() && self.lanes.iter().any(|l| l.is_none()) {
                if self.reserve.is_some() {
                    // LANERES1: plan-driven admission — snapshot-first, reservations, bounded
                    // patience, rate-limited forced takes. Same queue, same rx; only WHICH
                    // waiting request goes to WHICH lane changes, and only while the policy is
                    // on (--lane-reserve). Off = the original body below, byte-for-byte.
                    while let Ok(req) = rx.try_recv() {
                        // MAXBATCH1: over-cap arrival (beyond what the cap runs at once) — count it.
                        if self.lanes.iter().filter(|l| l.is_some()).count() + waiting.len() >= self.lanes.len() {
                            crate::metrics::over_cap_queued();
                        }
                        waiting.push_back(req);
                    }
                    if !waiting.is_empty() {
                        waiting.retain(|r| !r.tx.is_closed());
                    }
                    let now = std::time::Instant::now();
                    let slots = self.slot_views();
                    let wvs: Vec<crate::lane_reserve::WaitView> = waiting.iter().map(|r| crate::lane_reserve::WaitView {
                        arrived_at: r.received_at,
                        own_slot: self.own_snapshot_slot(&r.prompt),
                    }).collect();
                    let Some(cfg) = self.reserve else { unreachable!("checked above") };
                    match crate::lane_reserve::plan(&cfg, &mut self.reserve_st, now, &slots, &wvs) {
                        crate::lane_reserve::Plan::Admit { waiting_index, allowed, kind } => {
                            if let Some(req) = waiting.remove(waiting_index) {
                                if req.tx.is_closed() {
                                    continue;
                                }
                                self.admit_scoped(req, &allowed, kind);
                                crate::metrics::lane_reserve_waiting(waiting.len() as u64);
                            }
                        }
                        crate::lane_reserve::Plan::Wait { until } => {
                            crate::metrics::lane_reserve_waiting(waiting.len() as u64);
                            if self.lanes.iter().any(|l| l.is_some()) {
                                break; // a round is running; re-plan after the next step
                            }
                            // Every lane free but reserved: bounded blocking wait for a new
                            // request or the next deadline (20 ms tick; no engine work exists).
                            loop {
                                if self.fatal.is_some() { break; }
                                if until.map_or(false, |d| std::time::Instant::now() >= d) { break; }
                                match rx.try_recv() {
                                    Ok(req) => {
                                        // MAXBATCH1: over-cap arrival (beyond what the cap runs at once) — count it.
                                        if self.lanes.iter().filter(|l| l.is_some()).count() + waiting.len() >= self.lanes.len() {
                                            crate::metrics::over_cap_queued();
                                        }
                                        waiting.push_back(req);
                                        break;
                                    }
                                    Err(_) => std::thread::sleep(std::time::Duration::from_millis(20)),
                                }
                            }
                        }
                    }
                    continue; // re-plan from the top of this while
                }
                let Some(req) = waiting.pop_front() else { break };
                if req.tx.is_closed() {
                    continue;
                }
                self.admit(req);
            }
            if !waiting.is_empty() {
                waiting.retain(|r| !r.tx.is_closed());
            }
            if self.fatal.is_some() || self.lanes.iter().all(|l| l.is_none()) {
                continue;
            }
            if let Err(e) = self.step() {
                self.fail_step(&e);
            }
        }
    }

    /// WP02 liveness: the CUDA context is poisoned (sticky error) — nothing on this engine can run
    /// again. Mark it DEAD (/health 503, new requests 503 at the HTTP layer), fail every live and
    /// queued request with an error Finish, then either exit 70 (--exit-on-fatal: the supervisor
    /// restarts the server) or keep answering every late request with the same error.
    fn die(mut self, mut waiting: std::collections::VecDeque<BatchRequest>,
           mut rx: UnboundedReceiver<BatchRequest>) {
        let why = self.fatal.clone().unwrap_or_default();
        crate::server::set_engine_dead(&why);
        let reason = format!("error: engine stopped (fatal CUDA error: {why})");
        eprintln!("[exl3-serve] FATAL: {reason}");
        for s in 0..self.lanes.len() {
            if let Some(lane) = self.lanes[s].take() {
                let _ = lane.tx.send(TokEvent::Finish { reason: reason.clone() });
            }
        }
        self.publish_kv(); // issue #24: a dead engine holds no request
        while let Ok(req) = rx.try_recv() {
            waiting.push_back(req);
        }
        for req in waiting.drain(..) {
            let _ = req.tx.send(TokEvent::Finish { reason: reason.clone() });
        }
        if self.exit_on_fatal {
            eprintln!("[exl3-serve] FATAL: exit 70 (--exit-on-fatal, default; a supervisor restarts the server)");
            std::thread::sleep(std::time::Duration::from_millis(300)); // let the error events flush
            crate::logq::flush_and_exit(std::time::Duration::from_millis(300), 70); // H7 + LR-1
        }
        while let Some(req) = rx.blocking_recv() {
            let _ = req.tx.send(TokEvent::Finish { reason: reason.clone() });
        }
    }

    /// VIS-2 (TP=1): arm `slot` for an image request — the 3-axis RoPE map + image-token rows
    /// (FwdModel::set_slot_vision) and the prefill splice data (psc.vis_img / vis_src). Returns the
    /// prefix-cache key (image rows replaced by pixel-hash ids, so a cached prefix is reused only
    /// for the SAME image) and a generated-token cap (none since VIS-3).
    fn vision_setup(&mut self, slot: usize, req: &BatchRequest) -> Result<(Vec<u32>, usize)> {
        let sec = self.mrope_section.context("bad request: this server has no vision tower for this model (text-only)")?;
        let embeds = req.image_embeds.as_ref().context("image spans without image embeddings")?;
        let h = self.model.cfg.hidden_size;
        let plen = req.prompt.len();
        let n_img: usize = req.image_spans.iter().map(|s| s.num_tokens).sum();
        anyhow::ensure!(embeds.len() == n_img * h, "image embeddings {} != {n_img} tokens x {h}", embeds.len());
        anyhow::ensure!(n_img <= crate::exl3_forward::VIS_MROPE_ROWS,
                        "bad request: {n_img} image tokens exceed the per-request limit of {}", crate::exl3_forward::VIS_MROPE_ROWS);
        let (pos3, delta) = crate::vision_encoder::mrope_positions(plen, &req.image_spans)?;
        self.model.set_slot_vision(slot, &pos3, delta, sec)?;
        let mut src = vec![-1i32; plen];
        let mut key = req.prompt.clone();
        let mut off = 0usize;
        for sp in &req.image_spans {
            let hh = (sp.pixel_hash as u32) ^ ((sp.pixel_hash >> 32) as u32);
            for j in 0..sp.num_tokens {
                src[sp.start + j] = (off + j) as i32;
                // ids >= 2^31 never collide with a vocabulary id; only the cache compares them
                key[sp.start + j] = 0x8000_0000 | (hh.wrapping_add((j as u32).wrapping_mul(0x9E37_79B1)) & 0x7FFF_FFFF);
            }
            off += sp.num_tokens;
        }
        // the embedding table is fp16: image rows enter the residual in the same precision class
        // (VIS-4: also the TP wire's precision, so every rank splices bit-identical rows)
        let rows: Vec<f32> = embeds.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect();
        let (_, digest) = crate::tp_serve::image_rows_wire(&rows);
        self.vis_seam = Some((n_img as u32, delta as i32, digest));
        if self.psc.is_none() {
            self.psc = Some(self.model.prefill_scratch(psc_rows(self.chunk.max(1), self.tp.is_some()))?);
        }
        let img = self.model.vis_rows_upload(&rows)?;
        let psc = self.psc.as_mut().unwrap();
        psc.vis_img = Some(img);
        psc.vis_src = src;
        crate::drprintln!("[exl3-serve] VIS-2 slot={slot}: {} image(s), {n_img} image tokens in a {plen}-token prompt; rope delta {delta}",
                 req.image_spans.len());
        // VIS-3: the QSA indexer ropes its pooled keys through the same per-slot map, so an image
        // request runs past the dense window like text (no generated-token cap)
        Ok((key, usize::MAX))
    }

    // ---------------------------------------------------------- LANERES2: --kv-spill
    /// Capture `slot`'s snapshot (the EVICTED conversation's — the caller checked the incoming
    /// prompt does not extend it) into the spill store. Never fails the round: a capture or
    /// store error counts as a recompute-miss and the snapshot is simply lost (today's
    /// behavior). Key = FNV-1a of the cached tokens, so a later request whose prompt extends
    /// them finds it by prefix scan. The disk write is async (the store's writer thread).
    /// Err only under TP, when the cross-rank agree itself failed (the link is aborted by then): the admission must stop
    /// there with this message (ISSUES 121: it used to be parked in `self.fatal` while the admission ran on, and the
    /// next lockstep's 'transport already aborted' message replaced it).
    fn spill_capture_slot(&mut self, slot: usize, old: &[u32]) -> Result<()> {
        let Some(st) = self.spill.clone() else { return Ok(()) };
        let key = crate::kv_spill::conv_key(old);
        let tokens = old.len() as u64;
        // LANERES3 (TP>=2): every decision in this hook must come out IDENTICALLY on all
        // ranks — the policy state it feeds decides future restores in lockstep. So under
        // TP the budget gate reads the deterministic policy bytes (not the live writer
        // queue), the write is synchronous at this lockstep point (no async completion to
        // race), and the outcome is agreed across ranks with one SpillFlag: a disk that
        // failed on ANY rank drops the key EVERYWHERE (counted), so the replicated policy
        // can never diverge. One capture is in flight at a time (admission is serial), so
        // the sync write costs the round one D2H + one local NVMe write at an eviction —
        // the same transient the single-box path pays, minus the queue.
        if self.tp.is_some() {
            let est = tokens * crate::kv_spill::KV_BYTES_PER_TOKEN;
            if self.spill_pol.bytes() + est > self.spill_pol.cap_bytes() {
                crate::metrics::kv_spill_miss_recompute();
                crate::drprintln!("[kv-spill] capture refused: policy over budget ({} + est {est} bytes)", self.spill_pol.bytes());
                return Ok(());
            }
            let mut buf = Vec::new();
            if let Err(e) = self.model.spill_capture(slot, old.len(), &mut buf) {
                crate::metrics::kv_spill_miss_recompute();
                crate::dreprintln!("[exl3-serve] kv spill: capture failed slot={slot} ({e}); snapshot dropped");
                return Ok(());
            }
            crate::kv_spill::finalize_header(&mut buf, tokens);
            let wrote = st.put_sync(key, self.spill_rank, &buf);
            let all_ok = match self.tp.as_mut().map(|t| t.spill_agree(wrote.is_ok())) {
                Some(Ok(v)) => v,
                Some(Err(e)) => anyhow::bail!("kv spill: capture agree failed slot={slot}: {e:#}"),
                None => wrote.is_ok(),
            };
            if !all_ok {
                st.remove(key, self.spill_rank);
                crate::metrics::kv_spill_capture_peer_drop();
                crate::dreprintln!("[kv-spill] capture dropped: a rank's write failed (agreed across ranks; re-prefill on return)");
                return Ok(());
            }
            let bytes = wrote.unwrap_or(0);
            for v in self.spill_pol.put(key, bytes, tokens) {
                st.remove(v, self.spill_rank);
            }
            crate::metrics::kv_spill_write(bytes, tokens);
            return Ok(());
        }
        // Pre-check the queue budget BEFORE the GiB-scale capture exists: a refusal here is
        // free; one after the D2H transiently pumps ~4 GB through malloc and can cross the
        // box's memwatch floor under churn (the b5x10 lesson).
        {
            let st = self.spill.as_ref().unwrap();
            if st.pending.load(std::sync::atomic::Ordering::Relaxed) + tokens * crate::kv_spill::KV_BYTES_PER_TOKEN
                > st.max_pending
            {
                crate::drprintln!("[kv-spill] capture refused: queue over budget (est {} bytes)", tokens * crate::kv_spill::KV_BYTES_PER_TOKEN);
                crate::metrics::kv_spill_miss_recompute();
                return Ok(());
            }
        }
        let mut buf = Vec::new();
        if let Err(e) = self.model.spill_capture(slot, old.len(), &mut buf) {
            crate::metrics::kv_spill_miss_recompute();
            crate::drprintln!("[exl3-serve] kv spill: capture failed slot={slot} ({e}); snapshot dropped");
            return Ok(());
        }
        crate::kv_spill::finalize_header(&mut buf, tokens);
        match st.put_prebuilt(key, self.spill_rank, buf) {
            Ok(bytes) => {
                for v in self.spill_pol.put(key, bytes, tokens) {
                    st.remove(v, self.spill_rank);
                }
                crate::metrics::kv_spill_write(bytes, tokens);
            }
            Err(e) => {
                crate::metrics::kv_spill_miss_recompute();
                crate::drprintln!("[exl3-serve] kv spill: store put failed ({e}); snapshot dropped");
            }
        }
        Ok(())
    }

    /// Own-snapshot miss: try the spill store. Returns the restored token count (the new
    /// `reuse`) or None (miss / cost-model drop / any error — re-prefill, counted). The read
    /// is synchronous in the admission path, like the prefill it replaces, only ~50x shorter.
    /// Err only under TP, when the cross-rank agree failed or a restore failed AFTER every rank agreed to it (the ranks'
    /// KV would differ): the admission stops there with this message (ISSUES 121, as `spill_capture_slot`).
    fn spill_restore_slot(&mut self, slot: usize, key: &[u32], end: usize) -> Result<Option<usize>> {
        if self.spill.is_none() || self.spill_pol.is_empty() {
            return Ok(None);
        }
        let st = self.spill.clone().unwrap();
        let Some((conv, _)) = self.spill_pol.lookup(key, crate::kv_spill::NVME_READ_BPS, crate::kv_spill::PREFILL_TPS) else {
            crate::metrics::kv_spill_miss_recompute();
            return Ok(None);
        };
        // LANERES3 (TP>=2): the lookup and cost model are pure replicated state (identical
        // on every rank); the local read's outcome is agreed with one SpillFlag, so a
        // snapshot only ONE rank can read is dropped on BOTH (counted) and both re-prefill
        // — the rounds stay identical, never a one-sided restore.
        let read = st.get(conv, self.spill_rank);
        // LANERES3: agree the PARSE outcome, not just the read — the payload must pass
        // `spill_restore_check` (magic/version/shape vs THIS model) on BOTH ranks before
        // either restores. A payload any rank cannot read OR parse (e.g. a file from a
        // pre-LANERES3 capture format) is dropped everywhere and both re-prefill: the
        // policy stays replicated-identical and neither rank acts alone.
        if self.tp.is_some() {
            let local_ok = matches!(&read, Ok((_, p)) if self.model.spill_restore_check(slot, p).is_ok());
            let all_ok = match self.tp.as_mut().map(|t| t.spill_agree(local_ok)) {
                Some(Ok(v)) => v,
                Some(Err(e)) => anyhow::bail!("kv spill: restore agree failed slot={slot}: {e:#}"),
                None => local_ok,
            };
            if !all_ok {
                // One rank could not read or parse it: drop everywhere (symmetric), re-prefill.
                let rerr = match &read {
                    Ok(_) => String::new(),
                    Err(e) => format!("read: {e}"),
                };
                let cerr = match &read {
                    Ok((_, p)) => match self.model.spill_restore_check(slot, p) {
                        Ok(_) => String::new(),
                        Err(e) => format!("check: {e:#}"),
                    },
                    Err(_) => String::new(),
                };
                crate::dreprintln!(
                    "[kv-spill] restore dropped: conv {conv:016x} slot={slot} rank={} \
                     local read=({}) local check=({}) - dropped on all ranks, re-prefill",
                    self.spill_rank, rerr, cerr);
                self.spill_pol.remove(conv);
                st.remove(conv, self.spill_rank);
                crate::metrics::kv_spill_restore_peer_miss();
                return Ok(None);
            }
        }
        let payload = match read {
            Ok((_, p)) => p,
            Err(_) => {
                self.spill_pol.remove(conv);
                st.remove(conv, self.spill_rank);
                crate::metrics::kv_spill_miss_recompute();
                return Ok(None);
            }
        };
        let n = match self.model.spill_restore(slot, &payload) {
            Ok(n) => n,
            Err(e) => {
                // LANERES3 (TP>=2): after the cross-rank agree a one-sided failure means
                // the ranks' KV would differ — fatal-grade, reported loudly, never a
                // silent re-prefill (single-box keeps the graceful re-prefill).
                if self.tp.is_some() {
                    anyhow::bail!("kv spill: restore failed slot={slot} ({e}) after the cross-rank agree — ranks would diverge");
                }
                crate::metrics::kv_spill_miss_recompute();
                crate::drprintln!("[exl3-serve] kv spill: restore failed slot={slot} ({e}); re-prefill");
                return Ok(None);
            }
        };
        self.spill_pol.remove(conv);
        st.remove(conv, self.spill_rank);
        crate::metrics::kv_spill_hit(n as u64);
        crate::drprintln!("[exl3-serve] kv spill: restored slot={slot}: {n} of {end} prompt tokens (read {} MiB)", payload.len() >> 20);
        Ok(Some(n))
    }

    // ---------------------------------------------------------- LANERES1: --lane-reserve
    /// The SlotViews the policy reads, built fresh at every plan/pick: busy, snapshot present,
    /// release time + normal-ness of the last release.
    fn slot_views(&self) -> Vec<crate::lane_reserve::SlotView> {
        (0..self.lanes.len()).map(|i| crate::lane_reserve::SlotView {
            busy: self.lanes[i].is_some(),
            has_snapshot: self.cache[i].is_some(),
            released_at: self.rel[i].map(|(t, _)| t),
            normal_finish: self.rel[i].map_or(false, |(_, n)| n),
        }).collect()
    }

    /// WP16 resume depth per slot for THIS prompt (0 = no grid / no usable checkpoint) — the
    /// second branch of the owner-change victim order.
    fn wp16_depths(&self, prompt: &[u32]) -> Vec<usize> {
        let end = prompt.len().saturating_sub(1);
        (0..self.lanes.len()).map(|i| self.wp16.as_ref()
            .map_or(0, |st| st.resume_point(i, prompt, end, self.chunk.max(1)).0)).collect()
    }

    /// The pick block's prompt-end hit test over FREE lanes (same test, same (len, slot) max):
    /// the "own slot" a waiting request is admitted into first under --lane-reserve.
    fn own_snapshot_slot(&self, prompt: &[u32]) -> Option<usize> {
        (0..self.lanes.len()).filter(|&i| self.lanes[i].is_none())
            .filter_map(|i| match &self.cache[i] {
                Some(t) if t.len() < prompt.len() && prompt[..t.len()] == t[..] => Some((t.len(), i)),
                _ => None,
            }).max().map(|(_, i)| i)
    }

    /// LANERES1: every lane release of the single-box paths goes through one helper so no site
    /// is missed. `normal` = the request completed its turn (stop / length / a stop string / a
    /// tool call / loop-detected); only a normal release is eligible for a reservation. Cancels,
    /// errors and failed prefills never reserve. No stamping while the policy is off.
    fn release_lane(&mut self, s: usize, normal: bool) {
        self.lanes[s] = None;
        if self.reserve.is_some() {
            self.rel[s] = Some((std::time::Instant::now(), normal));
        }
    }

    /// `release_lane` for the sites that still need the lane (`take()`): stamp, then hand over.
    fn take_lane(&mut self, s: usize, normal: bool) -> Option<Lane> {
        let lane = self.lanes[s].take();
        if lane.is_some() && self.reserve.is_some() {
            self.rel[s] = Some((std::time::Instant::now(), normal));
        }
        lane
    }

    fn admit(&mut self, req: BatchRequest) {
        self.admit_req(req, None);
        // issue #24: the prefill (if any) is over — the lane now carries its own position, or the
        // admit failed / was cancelled and the slot is free again.
        self.kv_pf = None;
        self.publish_kv();
    }

    /// LANERES1: `admit` through the reservation policy — the pick is restricted to `allowed`,
    /// `kind` drives the counter and the log line. Same post-conditions as `admit`.
    fn admit_scoped(&mut self, req: BatchRequest, allowed: &[usize], kind: crate::lane_reserve::Kind) {
        self.admit_req(req, Some((allowed, kind)));
        self.kv_pf = None;
        self.publish_kv();
    }

    fn admit_req(&mut self, req: BatchRequest, scoped: Option<(&[usize], crate::lane_reserve::Kind)>) {
        let t_admit = std::time::Instant::now();
        // prefix cache: prefer the free lane whose snapshot is the longest prefix of this prompt,
        // then a lane with no snapshot, then any free lane.
        // LANERES1: with the reservation policy on (`scoped`), every candidate list is restricted
        // to the policy's `allowed` lanes, and the fallback after the snapshot hit is the
        // policy's victim order (empty > deepest WP16 > least recently released) instead of
        // "slot 0". `scoped == None` keeps today's pick byte-for-byte.
        let pick = {
            let free: Vec<usize> = (0..self.lanes.len()).filter(|&i| self.lanes[i].is_none()).collect();
            let free: Vec<usize> = match scoped {
                Some((allowed, _)) => free.into_iter().filter(|i| allowed.contains(i)).collect(),
                None => free,
            };
            let hit = free.iter().copied().filter_map(|i| match &self.cache[i] {
                Some(t) if t.len() < req.prompt.len() && req.prompt[..t.len()] == t[..] => Some((t.len(), i)),
                _ => None,
            }).max();
            hit.map(|(_, i)| i)
                .or_else(|| {
                    // LANERES1: a scoped admit that did not come through the snapshot-hit arm
                    // picks its owner-change victim among the allowed lanes only.
                    if scoped.is_some() {
                        return crate::lane_reserve::victim(&free, &self.slot_views(), &self.wp16_depths(&req.prompt));
                    }
                    // WP16: else the free lane whose checkpoints resume this prompt deepest
                    let st = self.wp16.as_ref()?;
                    let end = req.prompt.len().saturating_sub(1);
                    free.iter().copied()
                        .map(|i| (st.resume_point(i, &req.prompt, end, self.chunk.max(1)).0, i))
                        .filter(|&(q, _)| q > 0).max().map(|(_, i)| i)
                })
                .or_else(|| free.iter().copied().find(|&i| self.cache[i].is_none()))
                .or_else(|| free.first().copied())
        };
        let free = match pick {
            Some(i) => i,
            None => {
                let _ = req.tx.send(TokEvent::Finish {
                    reason: "error: server busy (all lanes occupied)".into(),
                });
                return;
            }
        };
        if let Some((_, kind)) = scoped {
            crate::metrics::lane_reserve_admit(match kind {
                crate::lane_reserve::Kind::Own => 0,
                crate::lane_reserve::Kind::Unreserved => 1,
                crate::lane_reserve::Kind::Forced => 2,
            });
            match kind {
                crate::lane_reserve::Kind::Own => {
                    let snap = self.cache[free].as_ref().map_or(0, |t| t.len());
                    crate::drprintln!("[exl3-serve] lane-reserve: own slot={free} ({snap} of {} prompt tokens)", req.prompt.len());
                }
                crate::lane_reserve::Kind::Unreserved => {
                    crate::drprintln!("[exl3-serve] lane-reserve: unreserved slot={free}");
                }
                crate::lane_reserve::Kind::Forced => {
                    let waited = t_admit.duration_since(req.received_at).as_secs_f32();
                    let victim_age = self.rel[free].map_or("never ran".to_string(), |(t, _)| {
                        format!("{:.0} s ago", t_admit.duration_since(t).as_secs_f32())
                    });
                    crate::drprintln!("[exl3-serve] lane-reserve: forced slot={free} after {waited:.0} s (victim released {victim_age})");
                }
            }
        }
        if req.schema.is_some() && self.tp.is_some() {
            // W2-EXL3: the masked plain step is wired at TP=1 only (the head refuses it in
            // exl3_reject before an Admit ships; this is the structural backstop)
            let _ = req.tx.send(TokEvent::Finish {
                reason: "error: json-schema constraint not supported on the exl3 TP serve path yet".into(),
            });
            return;
        }
        if req.prompt.is_empty() {
            let _ = req.tx.send(TokEvent::Finish { reason: "error: empty prompt".into() });
            return;
        }
        let mut max_new = req.max_new.min(self.model.max_pos().saturating_sub(req.prompt.len() + 1));
        // VIS-2: an image request arms the slot's 3-axis RoPE map and the prefill splice; every other
        // request puts the slot back on plain text RoPE (a previous image request may have armed it).
        self.vis_seam = None;
        let key: Vec<u32> = if req.image_spans.is_empty() {
            if let Err(e) = self.model.clear_slot_vision(free) {
                let _ = req.tx.send(TokEvent::Finish { reason: format!("error: slot reset failed: {e}") });
                self.fatal = Some(format!("{e:#}"));
                return;
            }
            req.prompt.clone()
        } else {
            match self.vision_setup(free, &req) {
                Ok((key, cap)) => {
                    max_new = max_new.min(cap);
                    key
                }
                Err(e) => {
                    let _ = self.model.clear_slot_vision(free);
                    if let Some(p) = self.psc.as_mut() { p.vis_img = None; p.vis_src.clear(); }
                    let msg = format!("{e:#}");
                    let reason = if msg.starts_with("bad request:") || msg.starts_with("unprocessable:") {
                        format!("error: {msg}")
                    } else {
                        format!("error: vision setup failed: {msg}") // an engine fault, not the client's
                    };
                    let _ = req.tx.send(TokEvent::Finish { reason });
                    return;
                }
            }
        };
        let pre = self.prefill_cached(free, &req.prompt, &key, req.ckpt_at, &req.tx);
        if let Some(p) = self.psc.as_mut() { p.vis_img = None; p.vis_src.clear(); } // VIS-2: splice data is per prefill
        match pre {
            Ok(true) => {}
            // WP02: the client left mid-prefill — the slot is free again, nobody to tell
            Ok(false) => return,
            Err(e) => {
                crate::reprintln!("[exl3-serve] prefill failed (slot {free}): {e:#}");
                let _ = req.tx.send(TokEvent::Finish { reason: format!("error: prefill failed: {e}") });
                if self.tp.is_some() || sticky_cuda_error(&e) { self.fatal = Some(format!("{e:#}")); }
                return;
            }
        }
        // S-A3-h seam step: the LAST prompt token runs through forward_step — the
        // graphed/fused decode step the bench path uses for it — so its KV/GDN state
        // and the tap left in sc.resid are byte-class identical to bench ground
        // truth. Its argmax is the first generated token (the S-A3-e contract).
        let plen = req.prompt.len();
        // API-parity G2: the first generated token is SAMPLED for sampled requests (it was
        // the argmax). Seed: the request's, else fresh entropy; counter 0 = this token.
        let seed = req.seed.unwrap_or_else(|| {
            let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64).unwrap_or(0);
            t ^ ((free as u64) << 56) ^ 0x5DEECE66D
        });
        let first = crate::exl3_forward::SampParams::sampled(
            req.temperature, req.top_p, req.top_k, req.min_p, seed, 0);
        if let Err(e) = self.model.set_sampling(&mut self.sc, &[first]) {
            let _ = req.tx.send(TokEvent::Finish { reason: format!("error: sampler staging failed: {e}") });
            if self.tp.is_some() { self.fatal = Some(format!("{e:#}")); } // TP-D: never a one-rank skip
            return;
        }
        // WP15: penalties from the first generated token on (the seam step's row), over the
        // prompt tail + everything generated — the reference implementation's past_ids = the full sequence.
        let pen = crate::exl3_forward::PenParams::new(req.rep_penalty, req.presence_penalty,
                                                      req.frequency_penalty, self.pen_range, self.pen_range);
        if pen.is_some() {
            crate::dreprintln!("[exl3-serve] penalties slot={free}: {pen:?} (window {} + {} tokens)", self.pen_range, self.pen_range);
        }
        let staged = match pen {
            Some(_) => self.model.pen_seed(&mut self.sc, free, &req.prompt[..plen - 1])
                .and_then(|_| self.model.set_penalties(&mut self.sc, &[pen])),
            None => self.model.set_penalties(&mut self.sc, &[None]),
        };
        if let Err(e) = staged {
            let _ = req.tx.send(TokEvent::Finish { reason: format!("error: penalty staging failed: {e}") });
            if self.tp.is_some() { self.fatal = Some(format!("{e:#}")); } // TP-D: never a one-rank skip
            return;
        }
        // W2-EXL3: the seam step emits the request's FIRST token — mask it when the constraint
        // is live from the start (no think block to wait for).
        let mut schema_state = req.schema.as_ref().map_or(0, |sm| sm.start_state());
        let mut schema_armed = req.schema.as_ref().map_or(false, |sm| sm.defer_until().is_none());
        let vstaged = match req.schema.as_ref().filter(|_| schema_armed) {
            Some(sm) => {
                let bits = sm.allowed(schema_state);
                self.model.set_vmask(&mut self.sc, &[Some(&bits[..])])
            }
            None => self.model.set_vmask(&mut self.sc, &[None]),
        };
        if let Err(e) = vstaged {
            let _ = req.tx.send(TokEvent::Finish { reason: format!("error: schema mask staging failed: {e}") });
            return;
        }
        let last = match self.model.forward_step(
            &mut self.sc,
            &[*req.prompt.last().unwrap() as i32],
            &[plen - 1],
            &[free],
            None,
        ) {
            Ok(ids) => ids[0] as u32,
            Err(e) => {
                eprintln!("[exl3-serve] seam decode step failed (slot {free}): {e:#}");
                let _ = req.tx.send(TokEvent::Finish {
                    reason: format!("error: seam step failed: {e}"),
                });
                if self.tp.is_some() || sticky_cuda_error(&e) { self.fatal = Some(format!("{e:#}")); }
                return;
            }
        };
        // issue #24: the seam step wrote row plen - 1 — the request holds its whole prompt now (a
        // request that ends at its first token never becomes a lane; its context still counts in the peak)
        self.kv_pf = Some((free, plen));
        self.publish_kv();
        if self.tp.is_some() {
            // TP-C: lockstep on the seam step (the request's first token)
            // TP-D: + the slot, the prefix-cache / WP16-C1 resume points and the checkpoint store's
            // shape (live, allocated, this slot's checkpoint positions) — proven equal on every rank
            let (reuse, q) = self.last_resume;
            let mut w = vec![last, plen as u32, free as u32, reuse as u32, q as u32];
            if let Some((n, d, h)) = self.vis_seam {
                // VIS-4: the image rows, their count and the rope delta are identical on every rank
                w.extend([0x7155_0000, n, d as u32, h as u32, (h >> 32) as u32]);
            }
            if let Some(st) = self.wp16.as_ref() {
                w.extend([st.live() as u32, st.n_alloc as u32,
                          fnv32(0, st.positions(free).iter().map(|&x| x as u32))]);
            }
            if let Err(e) = self.tp_round("seam", 0, 1, &w, 0.0, 0.0, false, false) {
                crate::reprintln!("[exl3-serve] TP seam lockstep failed (slot {free}): {e:#}");
                let _ = req.tx.send(TokEvent::Finish { reason: format!("error: TP lockstep failed: {e}") });
                self.fatal = Some(format!("{e:#}"));
                return;
            }
        }
        if req.tx.send(TokEvent::Tok(last)).is_err() && self.tp.is_none() {
            // WP02: the client is gone — no lane (the prefix snapshot stays valid). TP-D: under TP
            // the lane is created on every rank and the head's next sweep cancels it.
            crate::dreprintln!("[mtp-stats] slot={free} finish=cancelled gen=1 rounds=0 plain_steps=0 temp={}", req.temperature);
            return;
        }
        // WP04-v2: engine-side TTFT (receipt = the handler's hand-off, AFTER tokenization and
        // template rendering), so a client TTFT can be split into HTTP/tokenizer vs engine time.
        crate::drprintln!("[exl3-serve] first token slot={free} {:.1} ms after receipt (queued {:.1} ms)",
                 req.received_at.elapsed().as_secs_f64() * 1e3,
                 t_admit.duration_since(req.received_at).as_secs_f64() * 1e3);
        if max_new <= 1 {
            let _ = req.tx.send(TokEvent::Finish { reason: "length".into() });
            return;
        }
        if self.eos.contains(&last) && !req.ignore_eos && 1 >= req.min_new {
            let _ = req.tx.send(TokEvent::Finish { reason: "stop".into() });
            return;
        }
        if let Some(sm) = req.schema.as_ref() {
            match schema_feed(sm, &mut schema_state, &mut schema_armed, last) {
                SchemaStep::Continue => {}
                SchemaStep::Complete => {
                    let _ = req.tx.send(TokEvent::Finish { reason: "stop".into() });
                    return;
                }
                SchemaStep::Invalid => {
                    crate::reprintln!("[schema] slot {free}: first token {last} leaves no valid continuation \
                                       — ending the turn with an error");
                    let _ = req.tx.send(TokEvent::Finish { reason: SCHEMA_DEAD_END.into() });
                    return;
                }
            }
        }
        // S-A3-f-d: greedy lanes on an M1 pack speculate by default (AGENTS 1b).
        // Prime the head once: snapshot the tap that predicted `last` (now the
        // seam decode step's sc.resid — exact, same composition as the bench-mtp
        // prime and every round's re-prime), then feed the head the last prompt
        // token with that tap (round 1's draft-extend reuses tap_row 0).
        // W2-EXL3: a schema lane never speculates (the verify rows are not masked), so it never
        // primes the head either; its reprobe stays Armed with mtp = None = plain for life.
        let mtp = if self.model.mtp.is_some() && !crate::exl3_forward::mtp_disabled_by_opt() && req.schema.is_none() {
            match self.mtp_prime(free, *req.prompt.last().unwrap(), plen) {
                Ok(()) => {
                    if crate::exl3_forward::a_xcheck_on() {
                        if let Err(e) = self.model.tap_xcheck_record(&self.sc, free, 0) {
                            eprintln!("[a-xcheck] record failed: {e:#}");
                        }
                    }
                    Some(0usize)
                }
                Err(e) => {
                    crate::reprintln!("[exl3-serve] MTP prime failed (lane runs plain): {e:#}");
                    if self.tp.is_some() || sticky_cuda_error(&e) { self.fatal = Some(format!("{e:#}")); }
                    None
                }
            }
        } else {
            None
        };
        // WP08: the reference implementation feeds the detector every emitted token, the first one included.
        let mut loop_det = self.loop_cfg
            .map(|(w, r)| crate::loop_detect::LoopDetector::for_stop_on_loop(w, r));
        if let Some(d) = loop_det.as_mut() { d.feed(last); }
        let is_img = !req.image_spans.is_empty(); // S-B9-CF-PACK4: an image lane never packs (v1)
        let admit_seq = self.admit_ctr;   // CF-FCFS: FCFS order; the same on every TP rank
        self.admit_ctr += 1;
        self.lanes[free] = Some(Lane {
            pos: req.prompt.len(),
            last_tok: last,
            generated: 1,
            max_new,
            min_new: req.min_new,
            ignore_eos: req.ignore_eos,
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            min_p: req.min_p,
            pen,
            loop_det,
            prompt_hash: prompt_hash(&req.prompt),
            history: req.prompt.clone(),
            seed,
            ctr: 1,
            dctr: 0,
            gate_first: true,
            ema_ms: 0.0,
            ema_tok: 0.0,
            ema_n: 0,
            ema_wall_ms: 0.0,
            last_wall: None,
            mtp_rounds: 0,
            // TP-I #7: a TP lane's cost prior is the TP=2 one (dds_prior; TP=1 = 6.0 unchanged)
            wp23: Wp23Lane { fit: CostFit::with_prior(dds_prior_w(self.tp_world())), ..Default::default() },
            st_rounds: 0,
            st_drafted: 0,
            st_accepted: 0,
            st_ms: 0.0,
            st_plain: 0,
            st_ctrl: 0,
            arms: Default::default(),
            first_toks: vec![last],
            reprobe: crate::mtp_reprobe::ReProbe::new(&self.reprobe_cfg),
            force_control: false,
            mtp,
            admit_seq,
            fcfs_credit: 0,
            img: is_img,
            schema: req.schema.clone(),
            schema_state,
            schema_armed,
            tx: req.tx,
        });
    }
}

// ---- TP-D: serving at TP=2 (design §5: run_tp_head / run_tp_mirror around the SPMD scheduler) ----

/// A request the EXL3 path refuses before admission (checked on the head BEFORE an Admit ships,
/// so a refused request never reaches the node).
fn exl3_reject(req: &BatchRequest) -> Option<String> {
    if req.schema.is_some() {
        return Some("error: json-schema constraint not supported on the exl3 serve path yet".into());
    }
    if req.prompt.is_empty() {
        return Some("error: empty prompt".into());
    }
    None
}

/// The seed a request without one gets (admit's TP=1 derivation, minus the slot term).
fn entropy_seed() -> u64 {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64).unwrap_or(0);
    t ^ 0x5DEECE66D
}

impl Exl3Scheduler {
    /// TP-D: when the engine goes idle, one line of lockstep counters (the session's agree receipts).
    fn tp_log_idle(&mut self, step_no: u64) {
        if self.lanes.iter().all(|l| l.is_none()) {
            // S-B9-CF-PACK5: the packed arm's counters on the node too (its mirror loop never reaches
            // the head's idle-point line); printed once per change, so the head's own call is a no-op
            if let Some(line) = self.pack_idle_line() { crate::drprintln!("{line}"); }
            if let Some(t) = self.tp.as_mut() {
                crate::drprintln!("[tp-serve] rank {} idle after control step {step_no}: lockstep agrees {} ok, pre-verify agrees {} ok{}",
                         t.rank, t.agrees, t.pre_agrees,
                         if t.ident { format!(", graphed digests {} compared, {} differ", t.ident_bufs, t.ident_bad) } else { String::new() });
                crate::drprintln!("{}", t.wp.line(t.rank)); // H5: twin of the line above - never direct on a served path
                t.wp = WaitProf::default();
            }
        }
    }

    /// VIS-4: the binary channel — raw image-row payloads written right after a Step frame, in the
    /// order of that Step's image admits (every node reads them back by `image_bytes`).
    fn tp_ship_raw(&mut self, payloads: &[Vec<u8>]) -> Result<()> {
        if payloads.is_empty() { return Ok(()); }
        use std::io::Write;
        let t = self.tp.as_mut().context("TP ship without TP state")?;
        let t0 = std::time::Instant::now();
        let total: usize = payloads.iter().map(|p| p.len()).sum();
        for s in t.ctl.iter_mut() {
            for p in payloads { s.write_all(p)?; }
            s.flush()?;
        }
        crate::drprintln!("[exl3-serve] VIS-4: shipped {} image payload(s), {:.1} MB to {} node(s) in {:.1} ms",
                 payloads.len(), total as f64 / 1e6, t.ctl.len(), t0.elapsed().as_secs_f64() * 1e3);
        Ok(())
    }

    fn tp_ship(&mut self, m: &crate::tp_serve::ServingMsg) -> Result<()> {
        let live = self.lanes.iter().any(|l| l.is_some());
        let t = self.tp.as_mut().context("TP ship without TP state")?;
        let (t0, s0) = (std::time::Instant::now(), crate::net::wait_sleeps());
        for s in t.ctl.iter_mut() {
            crate::tp_serve::send_serving(s, m)?;
        }
        if live { t.wp.rec(6, t0, s0); }
        Ok(())
    }

    /// TP-D: a TP rank cannot serve on after a failed lockstep, a failed step or a fatal admit:
    /// fail every live, pending and queued request loudly, mark the engine dead (/health 503 while
    /// the process winds down), and return the error — the caller exits 70.
    fn tp_fail(&mut self, why: String, waiting: &mut std::collections::VecDeque<BatchRequest>,
               rx: Option<&mut UnboundedReceiver<BatchRequest>>) -> anyhow::Error {
        crate::server::set_engine_dead(&why);
        let reason = format!("error: TP engine stopped ({why})");
        for s in 0..self.lanes.len() {
            if let Some(lane) = self.lanes[s].take() {
                let _ = lane.tx.send(TokEvent::Finish { reason: reason.clone() });
            }
        }
        self.publish_kv(); // issue #24: a dead engine holds no request
        if let Some(rx) = rx {
            while let Ok(req) = rx.try_recv() { waiting.push_back(req); }
        }
        for req in waiting.drain(..) {
            let _ = req.tx.send(TokEvent::Finish { reason: reason.clone() });
        }
        anyhow::anyhow!("{why}")
    }

    /// TP-D: the TP=2 HEAD's scheduler loop. Every scheduler-visible decision is taken HERE and
    /// shipped before it is applied: per iteration the head sweeps cancels (a closed client
    /// channel), admits waiting requests (validated here; the seed resolved here so sampled ranks
    /// draw the same stream), ships ONE `Step { events }` to every node, then applies exactly those
    /// events in order and runs the SPMD step the node runs. Everything else the step decides is a
    /// pure function of replicated state (slot pick, prefix-cache / WP16 / C1 resume points, EOS /
    /// max_new / loop stops) or is head-authoritative inside the step (round timing — TP-C's
    /// exchange; a prefill chunk's cancel — HeadFlag); the agree tripwires prove it every
    /// seam / round (and before every DDS verify).
    fn run_tp_head(mut self, mut rx: UnboundedReceiver<BatchRequest>, streams: Vec<std::net::TcpStream>) -> Result<()> {
        use crate::tp_serve::{ServingMsg, StepEvents, TpEvent, WireRequest};
        if let Ok(m) = crate::cpu_affinity::current_thread_affinity() {
            crate::dprintln!("[exl3-serve] TP head scheduler thread cpu mask: {}", crate::cpu_affinity::format_cpu_list(&m));
        }
        self.tp.as_mut().context("run_tp_head without TP state")?.ctl = streams;
        let mut waiting: std::collections::VecDeque<BatchRequest> = std::collections::VecDeque::new();
        let mut step_no: u64 = 0;
        let mut rots_printed = 0usize;
        let mut mixed_printed = 0u64;
        let fast_ctl = crate::opts::var(crate::opt!("exl3-tp-fastctl")).map_or(true, |v| v != "0");
        crate::dprintln!("[exl3-serve] TP control: {}", if fast_ctl { "step-go over RDMA while live, TCP Step only with events (TP-E)" }
                 else { "TCP Step every step (--exl3-tp-fastctl=0)" });
        loop {
            if waiting.is_empty() && self.lanes.iter().all(|l| l.is_none()) {
                // Clean idle point (TP twin of the single-process loop): total fcfs rotations.
                if self.lane_order == LaneOrder::Fcfs && self.fcfs_rots > rots_printed {
                    crate::drprintln!("[exl3-serve] lane-order fcfs idle: {rots_printed}->{} total rotations", self.fcfs_rots);
                    rots_printed = self.fcfs_rots;
                }
                // 9a(iii): the lazily captured mixed-step graphs, whenever their number grew
                if let Some(line) = mixed_graphs_idle_line(mixed_printed, crate::metrics::mixed_step_graphs(), self.width) {
                    crate::drprintln!("{line}");
                    mixed_printed = crate::metrics::mixed_step_graphs();
                }
                self.publish_kv(); // issue #24: the idle value (lanes freed outside a step included)
                if let Some(line) = self.pack_idle_line() { crate::drprintln!("{line}"); } // S-B9-CF-PACK4
                crate::metrics::sched_idle(); // H3: about to block for work (busy must not stick at 1)
                match rx.blocking_recv() {
                    Some(req) => waiting.push_back(req),
                    None => {
                        let _ = self.tp_ship(&ServingMsg::Shutdown);
                        return Ok(());
                    }
                }
            }
            while let Ok(req) = rx.try_recv() {
                waiting.push_back(req);
            }
            let mut events: Vec<TpEvent> = Vec::new();
            let mut admits: std::collections::VecDeque<BatchRequest> = std::collections::VecDeque::new();
            let mut payloads: Vec<Vec<u8>> = Vec::new(); // VIS-4: image rows of this step's admits, in order
            let mut free = 0usize;
            for s in 0..self.lanes.len() {
                match &self.lanes[s] {
                    None => free += 1,
                    Some(l) if l.tx.is_closed()
                             || crate::server::stream_backlog_exceeded(l.tx.len()) => { // LR-5
                        if !l.tx.is_closed() {
                            crate::metrics::stream_backlog_cancel();
                            crate::dreprintln!("[exl3-serve] slot {s}: stream cancelled — unconsumed event backlog at the --stream-backlog-events limit (reader stalled; LR-5; Cancel shipped to the mirrors)");
                        }
                        events.push(TpEvent::Cancel { lane: s });
                        free += 1;
                    }
                    Some(_) => {}
                }
            }
            // LANERES3: plan-driven admission under TP — the same single-box policy
            // (`lane_reserve::plan`), head-authoritative. The node never runs the policy:
            // each admit's scoping rides the wire (`WireRequest::reserve`), so both ranks
            // restrict `admit_req`'s pick identically and hold the same slot tables.
            // --lane-reserve 0 (default) takes the plain FIFO body: byte-for-byte today.
            if self.reserve.is_some() {
                // LANERES3 fix (2026-10-22): EXACTLY ONE scoped admit per iteration. The
                // plan's `allowed` set is computed from `self.lanes` HERE, but admits apply
                // later (the events-apply phase below) — a sweep of several admits plans
                // them all against the same stale view, two can target one lane, and the
                // second then finds `free ∩ allowed = {}` and the request was REFUSED
                // ("server busy"), the §1b-bis.8 anti-pattern (9/60 requests at conc 4 in
                // the first structural cells run; the FIFO path is immune because its pick
                // recomputes the free set at apply time). One admit per iteration keeps
                // every plan's view equal to the state its admit acts on; bursts take one
                // extra scheduler iteration per request.
                waiting.retain(|r| !r.tx.is_closed());
                if free > 0 && !waiting.is_empty() {
                    let now = std::time::Instant::now();
                    let slots = self.slot_views();
                    let wvs: Vec<crate::lane_reserve::WaitView> = waiting.iter().map(|r| crate::lane_reserve::WaitView {
                        arrived_at: r.received_at,
                        own_slot: self.own_snapshot_slot(&r.prompt),
                    }).collect();
                    if let Some(cfg) = self.reserve {
                        match crate::lane_reserve::plan(&cfg, &mut self.reserve_st, now, &slots, &wvs) {
                            crate::lane_reserve::Plan::Admit { waiting_index, allowed, kind } => {
                                if let Some(mut req) = waiting.remove(waiting_index) {
                                    if !req.tx.is_closed() {
                                        if let Some(why) = exl3_reject(&req) {
                                            let _ = req.tx.send(TokEvent::Finish { reason: why });
                                        } else {
                                            if req.seed.is_none() {
                                                req.seed = Some(entropy_seed()); // head-resolved: every rank samples one stream
                                            }
                                            let mut w = WireRequest::from(&req);
                                            w.reserve = Some(crate::tp_serve::Scoping { allowed: allowed.clone(), kind: kind as u8 });
                                            if let Some(e) = req.image_embeds.as_ref().filter(|_| !req.image_spans.is_empty()) {
                                                // VIS-4: the rows ride the binary channel right after this step's frame
                                                let (bytes, digest) = crate::tp_serve::image_rows_wire(e);
                                                w.image_bytes = bytes.len() as u64;
                                                w.image_digest = digest;
                                                payloads.push(bytes);
                                            }
                                            events.push(TpEvent::Admit(w));
                                            admits.push_back(req);
                                            free -= 1;
                                        }
                                    }
                                }
                            }
                            crate::lane_reserve::Plan::Wait { until } => {
                                // Admission holds: every free lane is reserved for a waiting
                                // conversation's own snapshot. With a round running there is
                                // nothing to wait for HERE — the next step re-plans. With every
                                // lane free the head ticks (20 ms) for a new arrival or the
                                // next deadline, the single-box shape (no busy spin, no lost
                                // work: the patience deadline still forces a take).
                                if self.lanes.iter().all(|l| l.is_none()) {
                                    loop {
                                        if self.fatal.is_some() {
                                            break;
                                        }
                                        if until.map_or(true, |d| std::time::Instant::now() >= d) {
                                            break;
                                        }
                                        match rx.try_recv() {
                                            Ok(req) => {
                                                waiting.push_back(req);
                                                break;
                                            }
                                            Err(_) => std::thread::sleep(std::time::Duration::from_millis(20)),
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                crate::metrics::lane_reserve_waiting(waiting.len() as u64);
            } else {
            while free > 0 {
                let Some(mut req) = waiting.pop_front() else { break };
                if req.tx.is_closed() {
                    continue;
                }
                if let Some(why) = exl3_reject(&req) {
                    let _ = req.tx.send(TokEvent::Finish { reason: why });
                    continue;
                }
                if req.seed.is_none() {
                    req.seed = Some(entropy_seed()); // head-resolved: every rank samples one stream
                }
                let mut w = WireRequest::from(&req);
                if let Some(e) = req.image_embeds.as_ref().filter(|_| !req.image_spans.is_empty()) {
                    // VIS-4: the rows ride the binary channel right after this step's frame
                    let (bytes, digest) = crate::tp_serve::image_rows_wire(e);
                    w.image_bytes = bytes.len() as u64;
                    w.image_digest = digest;
                    payloads.push(bytes);
                }
                events.push(TpEvent::Admit(w));
                admits.push_back(req);
                free -= 1;
            }
            }
            if !waiting.is_empty() {
                waiting.retain(|r| !r.tx.is_closed());
            }
            if events.is_empty() && self.lanes.iter().all(|l| l.is_none()) {
                continue;
            }
            step_no += 1;
            // TP-E: a live engine's step goes over RDMA; the TCP Step only when it carries events
            let go = fast_ctl && self.lanes.iter().any(|l| l.is_some());
            if go {
                let n = events.len();
                if let Err(e) = self.tp.as_mut().context("TP state")?.step_go(step_no, n) {
                    return Err(self.tp_fail(format!("step-go failed at step {step_no}: {e:#}"), &mut waiting, Some(&mut rx)));
                }
            }
            if !go || !events.is_empty() {
                let msg = ServingMsg::Step(StepEvents { step: step_no, events: events.clone() });
                if let Err(e) = self.tp_ship(&msg) {
                    return Err(self.tp_fail(format!("control plane send failed at step {step_no}: {e:#}"), &mut waiting, Some(&mut rx)));
                }
                if let Err(e) = self.tp_ship_raw(&payloads) {
                    return Err(self.tp_fail(format!("image payload send failed at step {step_no}: {e:#}"), &mut waiting, Some(&mut rx)));
                }
            }
            for ev in events {
                match ev {
                    TpEvent::Cancel { lane } => {
                        if let Some(l) = self.lanes[lane].take() {
                            l.log_cancel(lane);
                        }
                    }
                    TpEvent::Admit(w) => {
                        let req = admits.pop_front().context("admit queue out of step")?;
                        // LANERES3: apply the same scoping we shipped (`WireRequest::reserve`)
                        // or the head's own slot table diverges from the mirrors' from the
                        // very first scoped admit. A scoped admit whose allowed lanes are no
                        // longer free at apply time NEVER refuses (§1b-bis.8): the request
                        // returns to the waiting deque, the next iteration re-plans, and the
                        // mirror skips the event identically (same replicated lane state).
                        match w.reserve {
                            Some(sc) => {
                                if !crate::lane_reserve::allowed_free(&self.lanes, &sc.allowed) {
                                    crate::metrics::lane_reserve_tp_requeued();
                                    crate::dreprintln!("[exl3-serve] TP scoped admit requeued (no allowed lane free; replanning)");
                                    waiting.push_back(req);
                                } else {
                                    self.admit_scoped(req, &sc.allowed, crate::lane_reserve::Kind::from_wire(sc.kind));
                                }
                            }
                            None => self.admit(req),
                        }
                    }
                }
                if let Some(f) = self.fatal.clone() {
                    return Err(self.tp_fail(format!("admit failed at step {step_no}: {f}"), &mut waiting, Some(&mut rx)));
                }
            }
            if self.lanes.iter().any(|l| l.is_some()) {
                if let Err(e) = self.step() {
                    eprintln!("[exl3-serve] TP decode step failed: {e:#}");
                    return Err(self.tp_fail(format!("decode step {step_no} failed: {e:#}"), &mut waiting, Some(&mut rx)));
                }
            }
            self.tp_log_idle(step_no);
        }
    }

    /// TP-D: the node's mirror loop — apply each head `Step` (cancels, admits with the head's
    /// resolved sampling params + seed) in order, then run the identical SPMD step. The node's
    /// token channels have no receiver (sends are no-ops; no rank-local cancel exists under TP).
    /// Ok = the head shut the session down (or its stream closed); Err = a failed step (the node
    /// process exits and its supervisor re-arms).
    fn run_tp_mirror(mut self) -> Result<()> {
        use crate::tp_serve::{ServingMsg, TpEvent};
        let mut step_no: u64 = 0;
        let mut admitted: u64 = 0;
        let fast_ctl = crate::opts::var(crate::opt!("exl3-tp-fastctl")).map_or(true, |v| v != "0");
        crate::drprintln!("[exl3-serve] TP control (node): {}", if fast_ctl { "step-go over RDMA while live (TP-E)" } else { "TCP Step every step" });
        loop {
            let live = self.lanes.iter().any(|l| l.is_some());
            if fast_ctl && live {
                let (hs, n) = self.tp.as_mut().context("mirror without TP state")?.step_go(step_no + 1, 0)?;
                anyhow::ensure!(hs == step_no + 1, "TP mirror: head step-go {hs} after local step {step_no}");
                if n == 0 {
                    step_no = hs;
                    self.step().with_context(|| format!("TP node: decode step {step_no} failed"))?;
                    self.tp_log_idle(step_no);
                    continue;
                }
                // events ride the TCP Step below (the head sent it after the go)
            }
            let msg = {
                let t = self.tp.as_mut().context("mirror without TP state")?;
                let (t0, s0) = (std::time::Instant::now(), crate::net::wait_sleeps());
                let r = crate::tp_serve::recv_serving(&mut t.ctl[0]);
                if live { t.wp.rec(5, t0, s0); }  // mid-request only: the head's inter-step host time + TCP
                match r {
                    Ok(m) => m,
                    Err(e) => {
                        // H7: END-OF-LIFE line — direct print; the node exits right after the
                        // return and no logq flush can be guaranteed there.
                        crate::deprintln!("[exl3-serve] TP node: head control stream closed ({e:#}) — session over after {step_no} steps, {admitted} requests");
                        return Ok(());
                    }
                }
            };
            match msg {
                ServingMsg::Shutdown => {
                    // H7: END-OF-LIFE line — direct print (same reason as the closed-stream line).
                    crate::deprintln!("[exl3-serve] TP node: Shutdown from the head after {step_no} steps, {admitted} requests");
                    return Ok(());
                }
                ServingMsg::Step(se) => {
                    anyhow::ensure!(se.step == step_no + 1, "TP mirror: head step {} after local step {step_no}", se.step);
                    step_no = se.step;
                    // PR #17 (ported): read ALL binary image payloads right after the Step frame,
                    // in admit order, BEFORE any admit()/prefill runs — prefill multiplexes
                    // HeadFlag traffic on the same control stream, so a payload still unread when
                    // the first admit runs could interleave with HeadFlag frames (stream desync /
                    // digest mismatch with 2+ images admitted in one Step). Digest checks unchanged.
                    // S-B9-REL-PRPORT3 (review R9): the pre-read lives in
                    // read_step_image_payloads (unit-tested); admit-order specs, then read.
                    let specs: Vec<(u64, u64)> = se.events.iter().filter_map(|ev| match ev {
                        TpEvent::Admit(w) => Some((w.image_bytes as u64, w.image_digest as u64)),
                        _ => None,
                    }).collect();
                    let payloads = {
                        let t = self.tp.as_mut().context("mirror without TP state")?;
                        read_step_image_payloads(&mut t.ctl[0], step_no, &specs)?
                    };
                    let mut payload_iter = payloads.into_iter();
                    for ev in se.events {
                        match ev {
                            TpEvent::Cancel { lane } => {
                                if let Some(l) = self.lanes.get_mut(lane).and_then(|l| l.take()) {
                                    l.log_cancel(lane);
                                }
                            }
                            TpEvent::Admit(w) => {
                                let (tx, _rx, _bl) = crate::server::tok_channel(); // mirror dummy; LR-5 gated off under TP
                                let min_p = w.min_p;
                                // LANERES3: the head's reservation scoping rides the wire;
                                // the mirror restricts its pick identically so every rank
                                // lands the same lane (the pick is a pure function of
                                // replicated state, and the restriction is part of that
                                // state now).
                                let scoping = w.reserve.clone();
                                let mut req = w.into_request(tx);
                                req.min_p = min_p;
                                // PR #17: this admit's image rows were already read (in admit
                                // order) right after the Step frame — see payloads above.
                                req.image_embeds = payload_iter.next().flatten();
                                admitted += 1;
                                match scoping {
                                    Some(sc) => {
                                        if !crate::lane_reserve::allowed_free(&self.lanes, &sc.allowed) {
                                            // LANERES3: the head requeued this admit (no allowed lane free
                                            // at apply time); the mirror skips it identically or the slot
                                            // tables diverge. The request re-arrives as a fresh admit event.
                                            crate::metrics::lane_reserve_tp_requeued();
                                        } else {
                                            self.admit_scoped(req, &sc.allowed, crate::lane_reserve::Kind::from_wire(sc.kind));
                                        }
                                    }
                                    None => self.admit(req),
                                }
                            }
                        }
                        if let Some(f) = self.fatal.clone() {
                            anyhow::bail!("TP node: admit failed at step {step_no}: {f}");
                        }
                    }
                    if self.lanes.iter().any(|l| l.is_some()) {
                        self.step().with_context(|| format!("TP node: decode step {step_no} failed"))?;
                    }
                    self.tp_log_idle(step_no);
                }
                other => anyhow::bail!("TP mirror: unexpected control message {other:?}"),
            }
        }
    }
}

impl TpSync {
    /// TP-4D (audit C2) world > 2: the outcome of ONE WP16 checkpoint allocation attempt as a lockstep decision.
    /// `CkptStore::acquire` freezes its cap when THIS rank's `cuMemAlloc` fails; ranks whose allocation outcomes
    /// differ hold different checkpoint sets and every later store-dependent decision (resume point, chunk grid)
    /// goes rank-local. Every rank sends (buffers held, ok); Ok(ranks that failed) — if any, ALL ranks must treat
    /// the allocation as failed (and drop a buffer they did get) so the stores stay identical.
    pub(crate) fn alloc_agree_hub<X: crate::net::HubXport>(&mut self, x: &mut X, n_alloc: usize, ok: bool)
                                                           -> Result<Vec<usize>> {
        use crate::tp_lockstep as ls;
        let all = crate::net::hub_all(x, &ls::alloc_frame(n_alloc, ok), ls::ALLOC_WIRE)
            .map_err(|e| self.hub_fail("wp16 alloc", &format!("{e:#}")))?;
        ls::alloc_verdict(&all, self.world as usize).map_err(|why| self.hub_fail("wp16 alloc", &why))
    }

    /// The body of `CkptStore::acquire`'s allocation closure with the cross-rank agreement applied: `r` = this rank's
    /// local `ckpt_alloc` result, `n_before` = the buffers this rank held before it. Every rank runs this at the same
    /// program point (the store state is identical by invariant) and gets the same outcome: Ok = every rank got its
    /// buffer, Err = at least one rank's allocation failed (a buffer this rank did get is dropped) so `acquire` freezes
    /// the cap on ALL ranks. A transport failure of the agree itself lands in `hub_err` (the caller must return it:
    /// `acquire` would otherwise swallow it as an allocation failure).
    pub(crate) fn alloc_outcome<B, X: crate::net::HubXport>(&mut self, x: &mut X, n_before: usize, r: Result<B>,
                                                            hub_err: &mut Option<anyhow::Error>) -> Result<B> {
        match self.alloc_agree_hub(x, n_before, r.is_ok()) {
            Ok(failed) if failed.is_empty() => r,
            Ok(failed) => match r {
                Ok(_dropped) => Err(anyhow::anyhow!("checkpoint allocation failed on rank(s) {failed:?} (agreed across ranks)")),
                Err(e) => Err(e),
            },
            Err(e) => { let m = format!("{e:#}"); *hub_err = Some(e); Err(anyhow::anyhow!(m)) }
        }
    }

    /// World 2 keeps today's rank-local decision (NO exchange, no behaviour change: the gap stays open there, see
    /// PLAN/TP-4D_REPORT.md); world > 2 runs `alloc_outcome` over the hub's control slots.
    fn wp16_alloc<B>(&mut self, n_before: usize, r: Result<B>, hub_err: &mut Option<anyhow::Error>) -> Result<B> {
        if self.world <= 2 { return r; }
        match crate::net::link_hub(self.deadline_ms) {
            Ok(mut x) => self.alloc_outcome(&mut x, n_before, r, hub_err),
            Err(e) => { let m = format!("{e:#}"); *hub_err = Some(e); Err(anyhow::anyhow!(m)) }
        }
    }
}

/// TP-4D: the head's rank/host map (`rank r = host (addr)`, r >= 1), set once by main after the node sync so the API
/// startup line can name which box is which rank. Display only — nothing reads it for a decision.
static TP_RANK_MAP: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
pub fn set_tp_rank_map(map: Vec<String>) { let _ = TP_RANK_MAP.set(map); }

/// TP-D: bring up the EXL3 TP attachment from a TpContext (the sanity + branch handshakes).
fn tp_attach(mut ctx: crate::tp::TpContext) -> Result<crate::exl3_forward::xtp::TpAttach> {
    ctx.sanity()?;
    ctx.branch_check(&crate::tp::TpBranch::Exl3Xtp)?;
    let (rank, world, link) = ctx.into_parts();
    Ok(crate::exl3_forward::xtp::TpAttach { rank, world, link })
}

/// TP-D: --exl3-tp-preverify=0 (diagnostic, prices the belt): skip the pre-verify width/drafts
/// exchange + agree (the post-round agree still proves the width, one round late — TP-C's shape).
/// Rides TpConfig's CLI option-registry snapshot, so both ranks decide alike.
fn tp_preverify_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::opts::var(crate::opt!("exl3-tp-preverify")).map_or(true, |v| v != "0"))
}

/// TP-D: --exl3-tp-ident=1 (diagnostic; rides TpConfig's CLI option-registry snapshot, so both ranks agree):
/// the served scheduler digests the logits rows, residual and taps at every lockstep point and
/// compares them across ranks (TP-C's G-T1-b graphed identity, through the API). Costs a dtoh per
/// step — a gate knob, never a serving default.
fn tp_ident_knob(s: &mut Exl3Scheduler) {
    if crate::opts::var(crate::opt!("exl3-tp-ident")).map_or(false, |v| v == "1") {
        if let Some(t) = s.tp.as_mut() {
            t.ident = true;
            crate::dprintln!("[exl3-serve] TP ident digests ON (--exl3-tp-ident=1, diagnostic): logits rows + resid + taps compared across ranks every lockstep point");
        }
    }
}

/// TP-4D: the world > 2 boot-agree frame magic (class prefix; a frame of another class at this point is a desync).
const BOOT_AGREE_MAGIC: u32 = 0xB007_A612;

/// TP-4D: Ok when every rank sent the boot magic and rank 0's config hash; Err names every differing rank with both
/// hashes (`all[r] = [magic, hash]`).
fn boot_hash_verdict(all: &[Vec<u32>], my_rank: usize) -> std::result::Result<(), String> {
    let mut bad = Vec::new();
    for (r, f) in all.iter().enumerate() {
        if f[0] != BOOT_AGREE_MAGIC {
            bad.push(format!("rank {r} sent frame class {:#010x}, not a boot-agree frame", f[0]));
        } else if f[1] != all[0][1] {
            bad.push(format!("rank {r} hash {:08x} != head {:08x}", f[1], all[0][1]));
        }
    }
    if bad.is_empty() { Ok(()) } else { Err(format!("{} (this is rank {my_rank})", bad.join("; "))) }
}

/// TP-D: every resolved scheduler knob, hashed and AGREED across ranks before the first request
/// (both ranks resolved them from the same argv + shipped env; this proves it, including the
/// memory-derived WP16 checkpoint cap and the tune table actually frozen).
fn tp_boot_agree(s: &mut Exl3Scheduler) -> Result<()> {
    let tune = crate::exl3_tune::status().map(|t| format!("{}|{:?}|{}|{}", t.state, t.table_sha8, t.decisions_s, t.decisions_d))
        .unwrap_or_default();
    let mut w: Vec<u32> = vec![s.width as u32, s.chunk as u32, s.mtp_k as u32, s.prefix_on as u32,
                               s.pen_range as u32, s.spec_ratio as u32, s.model.max_pos() as u32,
                               s.model.mtp.is_some() as u32, crate::exl3_forward::mtp_disabled_by_opt() as u32];
    let dc = s.draft_conf.to_bits();
    w.extend([dc as u32, (dc >> 32) as u32, s.draft_temp.map_or(0, |t| t.to_bits())]);
    w.extend(s.loop_cfg.map_or([0, 0], |(a, b)| [a as u32, b as u32]));
    w.extend(s.wp16.as_ref().map_or([0, 0], |st| [1, st.cap as u32]));
    w.push(crate::exl3_forward::tail_ckpt_on() as u32);
    w.push(crate::exl3_forward::kv_fmt_from_opts() as u32);
    // TP-I #7: the WP23 cost-guard prior both ranks resolved (it only matters if the targets differ)
    w.push(boot_prior_word(s.tp_world()));
    // TP-I2 item 1: the prefill-overlap granularity both ranks resolved (--tp-prefill-overlap, TpConfig v19)
    w.push(crate::exl3_forward::xtp::pf_overlap_code());
    // TP-H2: the sampled-row vocab-parallel tail both ranks resolved (--tp-vp-sampled, TpConfig v21)
    w.push(crate::exl3_forward::xtp::vp_sampled_code());
    // TP-SP1: sequence-parallel prefill (--tp-seq-parallel, TpConfig v22) — an ON code only, so an OFF
    // boot's hash is the pre-SP posture's
    if crate::exl3_forward::xtp::seq_parallel_on() {
        w.push(0x5350_0000 | crate::exl3_forward::xtp::seq_parallel_code());
    }
    // CLI-1: every SPMD registry option either rank has set (both installed the head's registry
    // (TpConfig v23); this proves it). Pushed only when one is set, so a boot with no option set
    // keeps the pre-registry hash.
    let spmd = crate::opts::spmd_digest();
    if let Some(d) = spmd { w.extend([0x4F50_5453, d]); }
    // S-B9-CF-PACK5: with --lane-pack on, the resolved mode, whether sampled lanes may pack here and
    // THE PACKED GRAPH SET (count + digest of the (family, regime, width) presence table). `pack_plan`
    // reads `pack_ready`, derived from rank-local graph presence: a rank whose set differed (the
    // CF-P1e TP=2 --max-batch 16 death was a decision reading rank-local graph state) now refuses to
    // boot here instead of planning a pack its peer cannot replay. Pushed only when the arm is on,
    // so a --lane-pack off boot adds nothing to the hash.
    if s.lane_pack.on() {
        if let Some(scp) = s.scp.as_ref() {
            let (n, d, _) = s.model.pack_graph_set(scp, &crate::lane_pack::fams(s.spec_ratio, s.pack_sampled));
            w.extend([0x504B_0001, 1, s.pack_sampled as u32, n as u32, d, s.pack_ready[0] as u32, s.pack_ready[1] as u32]);
        }
    }
    // F1/F10: when the RESOLVED mode is on (the default: no flag needs to be typed), hash the fallback decision
    // (active or not, and why) — a rank that packs while its peer fell back would misstate what is served. Every
    // reason comes from shipped options and model facts and the mode itself from the registry (the head's typed
    // value ships in the snapshot, the row's default is the same constant in one binary), so both ranks hash the
    // same words; a --lane-pack off boot records nothing.
    if let Some(why) = crate::lane_pack::boot_pack_word() {
        let mut d = 0x811c9dc5u32;
        for b in why.bytes() { d = (d ^ b as u32).wrapping_mul(0x01000193); }
        w.extend([0x5F50_4B46, why.is_empty() as u32, d]);
    }
    // FLASHNEXT-DF2-C (ISSUES 116): the DFlash2 chain's boot outcome — pushed only on a rank that
    // ENGAGED it (a boot without DF2 keeps the pre-existing hash), so a head that engaged while a
    // node fell back (no artifact on that node, a different artifact, block or verify width)
    // refuses HERE with a named reason instead of diverging at the first round.
    if s.df2.is_some() {
        let sha = crate::dflash2::load::last_load_sha().unwrap_or("unverified");
        w.extend([0xDF20_0001, crate::dflash2::block() as u32, s.df2_k as u32,
                  fnv32(0xD5A, sha.bytes().map(|b| b as u32))]);
    }
    w.extend(s.eos.iter().copied());
    w.extend(tune.bytes().map(|b| b as u32));
    let h = fnv32(0xB007, w.iter().copied());
    let rank = s.tp.as_ref().map_or(0, |t| t.rank);
    let world = s.tp.as_ref().map_or(2, |t| t.world);
    crate::dprintln!("[exl3-serve] TP={world} rank {rank}: boot config hash {h:08x} (width {} chunk {} k {} dds {} prefix {} wp16 cap {} tail {} tune {tune} \
              dds-guard prior {} ms/draft (--tp-dds-prior; TP=1 6.0) prefill-overlap {} vp-sampled {}{})",
             s.width, s.chunk, s.mtp_k, s.draft_conf, s.prefix_on, s.wp16.as_ref().map_or(0, |st| st.cap),
             crate::exl3_forward::tail_ckpt_on(), dds_prior_w(s.tp_world()),
             crate::exl3_forward::xtp::pf_overlap_desc(crate::exl3_forward::xtp::pf_overlap_code()),
             crate::exl3_forward::xtp::vp_sampled_desc(crate::exl3_forward::xtp::vp_sampled_code()),
             if crate::exl3_forward::xtp::seq_parallel_on() {
                 format!(" seq-parallel {}", crate::exl3_forward::xtp::seq_parallel_desc(crate::exl3_forward::xtp::seq_parallel_code()))
             } else { String::new() });
    crate::dprintln!("[exl3-serve] TP={world} rank {rank}: options {} (spmd digest {})", crate::opts::set_summary(),
             spmd.map_or("none".to_string(), |d| format!("{d:08x}")));
    if world > 2 {
        // TP-4D: world > 2 — every rank's hash through the head hub (probe-tolerant, unbounded: the four model
        // loads finish at different times; a dead peer still aborts within the dead-peer probe): EVERY rank sees
        // EVERY hash and names the offender, instead of one head-side verdict. world 2 keeps the pairwise agree.
        let all = crate::net::exchange_u32s_all(&[BOOT_AGREE_MAGIC, h], 4)
            .context("TP boot agree (head hub; peer dead or link aborted)")?;
        if let Err(why) = boot_hash_verdict(&all, rank as usize) {
            crate::net::abort_link_keep_code();
            anyhow::bail!("TP boot agree FAILED: the ranks resolved different serve configs (hash {h:08x} on rank {rank}): {why}");
        }
        return Ok(());
    }
    if let Err(why) = tp_agree_eq(0xB0_0700, 0, 0, h) {
        crate::net::abort_link();
        anyhow::bail!("TP boot agree FAILED: the ranks resolved different serve configs (hash {h:08x} on rank {rank}): {why}");
    }
    Ok(())
}

/// TP-D: the TP=2 HEAD serve (rank 0): load this rank's shard, prove the config equal with the
/// node, wait for the node's mirror to arm (Ready), then serve HTTP with the scheduler running
/// `run_tp_head` over the retained control streams.
/// PACK-FIX: `watch` (armed by the caller before the link bring-up) keeps watching the node control
/// streams through this rank's load + boot agree — a node boot failure it reports (or its death)
/// exits the head at once with the node's reason; it is disarmed right before the Ready reads.
pub fn run_tp_head_serve(args: &[String], model_dir: &str, ctx: crate::tp::TpContext,
                         mut streams: Vec<std::net::TcpStream>, watch: crate::cluster::NodeWatch) -> Result<()> {
    // S7 (REL_V0_7_3): refuse a bad --otel-endpoint BEFORE the RDMA attach (inside
    // build_serve it runs after tp_attach). Returning Err (not exit) lets the caller report it.
    if let Some(cfg) = crate::otel::config_from_args(|f| arg(args, &format!("--{f}")).map(|s| s.to_string())) {
        if let Err(e) = cfg.hostport() { anyhow::bail!("[otel] {e}"); }
    }
    tp_lane_width_refusal(args, ctx.world as u32)?;   // W4: before the attach (the head's main refused it before the sync)
    // S-B9-CF-PACK5: --lane-pack runs at TP=1, TP=2 and TP=4; a TYPED `--lane-pack on` at a larger world is refused
    // here BEFORE the attach (PACK4-REVIEW B1: and before this, in main.rs, by xtp::world_gt2_flag_refusals,
    // ahead of the discovery). F10: `parse(None)` is the raw-argv meaning (off), so a defaulted-on boot is never
    // refused here; build_serve turns it into a fallback line.
    crate::lane_pack::tp_refusal(crate::lane_pack::LanePack::parse(arg(args, "--lane-pack"))?, ctx.world as u32)?;
    let attach = tp_attach(ctx)?;
    let mut parts = build_serve(args, model_dir, Some(attach))?;
    tp_ident_knob(&mut parts.sched);
    tp_boot_agree(&mut parts.sched)?;
    watch.disarm(&streams);
    for (i, s) in streams.iter_mut().enumerate() {
        match crate::tp_serve::recv_serving(s).context("node Ready")? {
            crate::tp_serve::ServingMsg::Ready => crate::dprintln!("[exl3-serve] TP node rank {} READY (mirror armed)", i + 1),
            other => anyhow::bail!("expected Ready from node rank {}, got {other:?}", i + 1),
        }
    }
    let world = parts.sched.tp.as_ref().map_or(2, |t| t.world);
    if world <= 2 {
        println!("[exl3-serve] TP={world} serving: every §1b feature as TP=1 (graphs, MTP k={}, DDS {}, WP27 forks, prefix cache {}, \
                  WP16 {}) — head-authoritative Step/Cancel/Admit(seed) + HeadFlag, agree per seam/round + pre-verify",
                 parts.sched.mtp_k, parts.sched.draft_conf, parts.sched.prefix_on,
                 parts.sched.wp16.as_ref().map_or("off".to_string(), |st| format!("cap {}", st.cap)));
    }
    if world > 2 {
        // TP-4D (review F1): at world > 2 some §1b features are OFF, INERT or refused, so the world-2 "every §1b feature as
        // TP=1" claim is not printed. The `TP=N serving: ` prefix is the launcher's exactly-once marker (tp4_launch.sh
        // SERVE_SERVING_RE); the per-feature state rides the next line. Option-resolved state, read from the same predicates
        // the load uses; the `TP-H ...` / `[exl3-tp] ...` load lines above show what attached.
        {
            use crate::exl3_forward::xtp;
            let s = &parts.sched;
            let mtp_on = s.model.mtp.is_some() && !crate::exl3_forward::mtp_disabled_by_opt();
            let graphs = if crate::opts::var(crate::opt!("exl3-no-graph")).is_ok()
                || crate::opts::var(crate::opt!("mtp-dump")).map_or(false, |v| !v.is_empty()) { "OFF (--exl3-no-graph / --mtp-dump)" }
                else if arg(args, "--graph-precapture") == Some("off") { "ON (lazy capture: --graph-precapture off)" }
                else { "ON (the greedy, unpenalized shapes are precaptured at boot; sampled and penalized shapes are captured during traffic)" };
            let mtp = if mtp_on { format!("ON k={}", s.mtp_k) }
                else if s.model.mtp.is_some() {
                    format!("OFF ({})", crate::exl3_forward::mtp_off_by_opt().map_or("--exl3-no-mtp", |o| o.flag()))
                } else { "OFF (pack has no MTP head)".to_string() };
            let dds = if mtp_on && s.draft_conf > 0.0 { format!("ON at {}", s.draft_conf) } else { "OFF".to_string() };
            let dec = xtp::dec_xport_mode() == 1;
            println!("[exl3-serve] TP={world} serving: head-authoritative Step/Cancel/Admit(seed) + HeadFlag, hub lockstep agree per \
                      seam/round + pre-verify — some TP=1 §1b features are OFF / INERT / refused at world {world}; per-feature state on the \
                      next line, table in PLAN/TP-4D_REPORT.md");
            crate::dprintln!("[exl3-serve] TP={world} features: graphs {graphs}; MTP {mtp}; DDS {dds}; prefix cache {}; WP16 {}; WP27 forks not \
                      world-gated (--wp27-off / --wp27-forks / PDL decide); TP-H greedy vocab-parallel head {}; sampled / penalized / \
                      ratio-rule rows = {}; sharded MTP-head screen {}; decode \
                      transport {}; prefill transport {}; prefill overlap {}; sequence-parallel prefill {}; EP deal '{}'; pre-verify \
                      drain SKIPPED at world > 2 (it runs at world 2 only; hardware-unproven); refused if passed: --tp-oneshot, --tp-dec-grecv != 0, \
                      --ep-deal freq outside world 4",
                     if s.prefix_on { "ON" } else { "OFF (--prefix-cache off)" },
                     s.wp16.as_ref().map_or("OFF".to_string(), |st| format!("ON (cap {} checkpoints)", st.cap)),
                     if xtp::vp_head_on() && dec { "ON" } else { "OFF (--tp-vp-head off / --tp-dec-xport 0)" },
                     if xtp::vp_head_on() && dec && xtp::vp_sampled_on() { "ON (vocab-parallel all-gather, TP-4H2)" }
                         else { "REPLICATED head (--tp-vp-sampled off / --tp-vp-head off / --tp-dec-xport 0)" },
                     if mtp_on && xtp::dh_shard_on() && dec { "ON (shown for the greedy, unpenalized requests)" } else { "OFF (no MTP head / --tp-dh-shard off / --tp-dec-xport 0)" },
                     if dec { format!("folded K1 + multi-block K2, {} exchange(s) per reduce{}, receive cpu_done only",
                                      if xtp::tp_reduce_single_for(world) { 1 } else { (world as u32).trailing_zeros() },
                                      if xtp::tp_reduce_single_for(world) { " (--tp-reduce single: ONE all-peers stage, the world-4 default)" } else { "" }) }
                         else { "serial K1/K2 + cvt (--tp-dec-xport 0)".to_string() },
                     match xtp::prefill_xport_mode() { 0 => "serial rounds-aware K1/K2", 1 => "single-rail pipelined (hardware-unproven at world > 2)", _ => "dual-rail pipelined (hardware-unproven at world > 2)" },
                     if xtp::pf_overlap_rows() == 0 { "OFF" } else { "ON (the hooks ride the TP-4E round-aware program: a landed row range is consumed only after its last round's K2m; fold is refused at world > 2)" },
                     if !xtp::seq_parallel_on() { "OFF" }
                         else if world == 4 { "ON (TP-4S quarter rows: recursive-halving reduce-scatter + recursive-doubling all-gather; hardware-unproven until PLAN/TP-4S_GATES.md rungs 1-2 pass)" }
                         else { "INERT" },
                     xtp::ep_deal_kind());
        }
        // TP-4D: the API startup line names the world, the lockstep protocol and the rank/host map
        crate::dprintln!("[exl3-serve] TP={world} API up: head = rank 0; {}; lockstep = head-hub (one merged op per point, per-wait \
                  deadline {} ms — --tp-lockstep-timeout-ms); Step/Cancel/Admit/HeadFlag over {} node control streams",
                 TP_RANK_MAP.get().map_or("rank/host map unavailable".to_string(), |m| m.join("; ")),
                 parts.sched.tp.as_ref().map_or(0, |t| t.deadline_ms), world - 1);
    }
    serve_http(parts, Some(streams))
}

/// TP-D: the TP=2 NODE serve (rank >= 1): the same boot as the head (same argv), then the mirror
/// loop on this thread, pinned to core 9.
/// PACK-FIX: a boot failure (attach, load, boot agree, pinning) is reported to the head over the
/// control stream before returning — the head watches the stream through its own load and exits
/// loudly with this reason, instead of loading for minutes and then losing the node in a probe.
pub fn run_tp_node_serve(args: &[String], model_dir: &str, ctx: crate::tp::TpContext,
                         mut stream: std::net::TcpStream) -> Result<()> {
    let node_world = ctx.world;   // TP-4Z1: the worker mask needs the world to know whether the single-stage proxy (core 17) is live
    // S7 (REL_V0_7_3): same hoisted refusal as on the head - BEFORE the attach, and reported
    // through the boot failure path instead of process::exit(1).
    if let Some(cfg) = crate::otel::config_from_args(|f| arg(args, &format!("--{f}")).map(|s| s.to_string())) {
        if let Err(e) = cfg.hostport() { anyhow::bail!("[otel] {e}"); }
    }
    tp_lane_width_refusal(args, node_world as u32)?;   // W4: before the attach (the prelink refused it before the link)
    // S-B9-CF-PACK5: --lane-pack runs at TP=1, TP=2 and TP=4; a TYPED `--lane-pack on` at a larger world is refused
    // here BEFORE the attach (and before the link, in the prelink, by xtp::world_gt2_flag_refusals: PACK4-REVIEW
    // B1); a defaulted-on boot is handled by build_serve's fallback (F10).
    crate::lane_pack::tp_refusal(crate::lane_pack::LanePack::parse(arg(args, "--lane-pack"))?, node_world as u32)?;
    let boot = (|| -> Result<(Exl3Scheduler, Option<Vec<usize>>)> {
        let attach = tp_attach(ctx)?;
        let parts = build_serve(args, model_dir, Some(attach))?;
        let mut sched = parts.sched;
        tp_ident_knob(&mut sched);
        tp_boot_agree(&mut sched)?;
        anyhow::ensure!(crate::net::pin_thread(9), "TP node mirror thread failed to pin to core 9 — TP refuses to run unpinned");
        Ok((sched, parts.cpu_aff.0))
    })();
    let (mut sched, cpu_mask) = match boot {
        Ok(x) => x,
        Err(e) => {
            crate::cluster::node_report_failure(&mut stream, &format!("TP node boot failed: {e:#}"));
            return Err(e);
        }
    };
    set_worker_mask(tp_worker_mask(&cpu_mask, true, node_world));
    crate::tp_serve::send_serving(&mut stream, &crate::tp_serve::ServingMsg::Ready)?;
    crate::dprintln!("[exl3-serve] TP node: mirror armed (Ready sent) — waiting for the head's steps");
    sched.tp.as_mut().context("node without TP state")?.ctl = vec![stream];
    sched.run_tp_mirror()
}

/// WP24: `--spec-sampling match|ratio` and `--draft-temperature <T>` (EXL3 serve, TP=1 — no
/// TpConfig transport needed while EXL3 has no TP path). match (default) = today: drafts are the
/// head's argmax, a sampled verify row accepts while its sample == the draft. ratio = real-q
/// speculative sampling for SAMPLED requests: each draft pass samples its draft from q' (the
/// head's top-32 under --draft-temperature and the request's top-k / top-p), a verify row accepts
/// it with probability min(1, p/q'), an exact residual on reject — distribution-exact vs plain
/// sampling (gate: --probe-spec-sampling), but seeded sampled bytes change. Greedy requests are
/// unaffected. --draft-temperature default = the request's temperature.
pub fn parse_spec_sampling(args: &[String]) -> Result<(bool, Option<f32>)> {
    let spec_ratio = match arg(args, "--spec-sampling") {
        None | Some("ratio") => true, // owner ruling 2026-09-27: ratio is the default (χ² gate PASS, +5.1% T=1 thinking)
        Some("match") => false,
        Some(v) => anyhow::bail!("--spec-sampling must be match or ratio (got '{v}')"),
    };
    let draft_temp: Option<f32> = match arg(args, "--draft-temperature") {
        None => None,
        Some(v) => {
            let t: f32 = v.parse().map_err(|_| anyhow::anyhow!("--draft-temperature must be a number"))?;
            anyhow::ensure!(t.is_finite() && t > 0.0, "--draft-temperature must be > 0 (omit it = the request's temperature)");
            Some(t)
        }
    };
    // (SURFACE2 item 4: nothing is drafted with speculation switched off, so the mode is not announced)
    if (spec_ratio || draft_temp.is_some()) && !crate::exl3_forward::mtp_disabled_by_opt() {
        crate::dprintln!("[exl3-serve] spec sampling: {} (draft temperature {}){}",
                 if spec_ratio { "ratio (real-q: sampled drafts on the full draft-slice head — on DHEAD when the DHEAD-RQ line says so, accept min(1, p/q'))" } else { "match" },
                 draft_temp.map_or("= request temperature".to_string(), |t| t.to_string()),
                 if !spec_ratio && draft_temp.is_some() { " — --draft-temperature only applies with --spec-sampling ratio" } else { "" });
    }
    Ok((spec_ratio, draft_temp))
}

/// Is the `--vision-fail-gemm` drill armed? Read through the option registry, as the NVFP4 server
/// does (`src/main.rs`). S-B9-REL-VISIONFIX H2: this path read it with `arg(args, ..).is_some()`,
/// the VALUE reader, which answers "is there a token after the flag": a bare flag at the END of
/// the command line was silently not armed (and `--print-config` still showed it as set), while a
/// flag followed by anything at all, `=0` included, was. The registry parses a `Ty::Flag` by
/// presence, wherever it stands.
fn vision_fail_gemm_armed() -> bool {
    crate::opts::on(crate::opt!("vision-fail-gemm"))
}

/// CF-P1d pre-load fit check: the parts of a serving footprint, in bytes (estimates; see `fit_plan`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FitParts {
    pub weights: u64,  // device weights on this rank (model shards / world + the vision tower on the head)
    pub table: u64,    // the PLE n-gram table (host RAM when RAM-resident)
    pub per_lane: u64, // one lane: KV + indexer keys + MTP-head KV over max_pos, + its recurrent state (x1.15)
    pub fixed: u64,    // scratch, prefill scratch, graphs, vendor workspaces, prefix-checkpoint cap
    pub floor: u64,    // the memory watchdog's floor (--mem-watchdog-gb)
}

/// What a configuration needs and whether it fits: Ok(table_in_ram_possible) or Err(the refusal text).
pub fn fit_plan(p: FitParts, lanes: usize, avail: u64, mode: &str, max_pos: usize) -> Result<bool, String> {
    let gb = |b: u64| b as f64 / 1e9;
    let base = p.weights + p.fixed + p.per_lane * lanes as u64 + p.floor;
    let with_table = base + p.table;
    let line = format!("model weights {:.1} GB + {lanes} lane(s) x {:.2} GB (KV for {max_pos} tokens) + {:.1} GB working memory \
                        + {:.1} GB safety floor{} = {:.1} GB; this machine has {:.1} GB available",
                       gb(p.weights), gb(p.per_lane), gb(p.fixed), gb(p.floor),
                       if mode == "ram" { format!(" + n-gram table {:.1} GB in RAM", gb(p.table)) } else { String::new() },
                       gb(if mode == "ram" { with_table } else { base }), gb(avail));
    let fits_ssd = base <= avail;
    if mode == "ram" && with_table > avail {
        let mut msg = format!("this configuration does not fit in memory: {line}.");
        if fits_ssd {
            msg += " It fits with the n-gram table on SSD: use --ple-ram ssd (or leave --ple-ram at auto) — identical output, decode ~1-2% slower.";
        }
        return Err(msg + &suggest(p, lanes, avail, max_pos));
    }
    if !fits_ssd {
        return Err(format!("this configuration does not fit in memory, even with the n-gram table on SSD: {line}.{}",
                           suggest(p, lanes, avail, max_pos)));
    }
    Ok(with_table <= avail)
}

/// MAXBATCH1: the largest lane count `<= width` whose memory plan fits — `fit_plan` Ok (the auto
/// arm accepts the n-gram table landing on SSD, exactly like an un-fitted boot). None when not
/// even one lane fits (the refusal case: nothing can be served on this box at this context).
pub fn fit_max_lanes(p: FitParts, width: usize, avail: u64, mode: &str, max_pos: usize) -> Option<usize> {
    (1..=width).rev().find(|&n| fit_plan(p, n, avail, mode, max_pos).is_ok())
}

fn suggest(p: FitParts, lanes: usize, avail: u64, max_pos: usize) -> String {
    let room = avail.saturating_sub(p.weights + p.fixed + p.floor);
    let mut out = String::new();
    if p.per_lane > 0 && room / p.per_lane < lanes as u64 {
        let n = room / p.per_lane;
        if n >= 1 { out += &format!(" At this context length, at most --max-batch {n} fits."); }
        let per_tok = p.per_lane as f64 / max_pos.max(1) as f64;
        let ctx = (room as f64 / lanes.max(1) as f64 / per_tok) as u64 / 1024 * 1024;
        if ctx >= 4096 { out += &format!(" With --max-batch {lanes}, at most --max-seq-len {ctx} fits."); }
    }
    if room < p.per_lane {
        out += " The model alone leaves too little memory on one box: run it over two boxes (--tp 2).";
    }
    out
}

/// Which limit trimmed a `--max-seq-len` request (public PR #21).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CtxLimit {
    /// The model's trained window (`max_position_embeddings`) times the rope extension factor.
    Rope,
    /// The launch-argument bit-packings: `max_pos` rides in 20 bits (`exl3_forward::MAX_POS_PACKED`).
    Packing,
}

/// The YaRN factor the EXL3 serve honours for `model_dir`: `--rope-yarn-factor` wins, then
/// the shipped TpConfig (TP node), then the pack config's `rope_scaling` block (YARN1). One
/// precedence rule with the Config-based `qwen::resolved_yarn` (shared `qwen::pick_yarn`).
pub fn yarn_factor(model_dir: &str) -> f64 {
    let (f, _) = crate::qwen::pick_yarn(
        crate::opts::var(crate::opt!("rope-yarn-factor")).ok().and_then(|v| v.parse::<f32>().ok()),
        crate::tp::tp_config().map(|c| c.rope_yarn_factor),
        pack_yarn_from_config(model_dir),
    );
    f as f64
}

/// The pack config's own YaRN factor — `rope_scaling {type: "yarn", factor}` from the same
/// config.json `native_window` opens (None = the pack ships no YaRN block).
fn pack_yarn_from_config(model_dir: &str) -> Option<f32> {
    let raw = std::fs::read(std::path::Path::new(model_dir).join("config.json")).ok()?;
    let cfg: serde_json::Value = serde_json::from_slice(&raw).ok()?;
    let t = cfg.get("text_config").unwrap_or(&cfg);
    let rs = t.get("rope_scaling")?;
    let is_yarn = rs.get("type").and_then(|v| v.as_str())
        .map_or(false, |s| s.eq_ignore_ascii_case("yarn"));
    if !is_yarn { return None; }
    rs.get("factor").and_then(|v| v.as_f64())
        .map(|f| f as f32).filter(|f| f.is_finite() && *f >= 1.0)
}

/// The widest `--max-seq-len` the EXL3 packings carry: `max_pos = max_seq_len + decode_headroom`
/// must stay below `MAX_POS_PACKED`.
pub fn packable_max_seq_len() -> usize {
    (crate::exl3_forward::MAX_POS_PACKED - 1).saturating_sub(crate::batch::decode_headroom(false))
}

/// The context a `--max-seq-len` request may use, and the limit that trimmed it (None = honoured as
/// asked). `native` is the pack's trained window when its config could be read; the packing ceiling
/// is a constant and applies either way, so an unreadable config never disables the guard.
pub fn clamp_ctx(max_seq_len: usize, native: Option<usize>, rope_factor: f64) -> (usize, Option<CtxLimit>) {
    let pack_eff = packable_max_seq_len();
    let (eff, why) = match native {
        Some(n) => {
            let rope_eff = ((n as f64) * rope_factor.max(1.0)).ceil() as usize;
            (rope_eff.min(pack_eff), if pack_eff < rope_eff { CtxLimit::Packing } else { CtxLimit::Rope })
        }
        None => (pack_eff, CtxLimit::Packing),
    };
    if max_seq_len > eff { (eff, Some(why)) } else { (max_seq_len, None) }
}

/// The pack's trained window from its config.json (`text_config` when nested), as `fit_parts` reads it.
fn native_window(model_dir: &str) -> Option<usize> {
    let cfg: serde_json::Value =
        serde_json::from_slice(&std::fs::read(std::path::Path::new(model_dir).join("config.json")).ok()?).ok()?;
    let t = cfg.get("text_config").unwrap_or(&cfg);
    t.get("max_position_embeddings").and_then(|v| v.as_u64()).map(|v| v as usize)
}

/// One line naming the limit that trimmed `asked` to `eff` (pure: the text is unit-tested).
pub fn clamp_ctx_message(asked: usize, eff: usize, why: CtxLimit, native: Option<usize>, rope_factor: f64) -> String {
    let limit = match why {
        CtxLimit::Rope if rope_factor > 1.0 =>
            format!("the model's trained window {} x rope factor {rope_factor}", native.unwrap_or(0)),
        CtxLimit::Rope => format!("the model's trained window {} (max_position_embeddings)", native.unwrap_or(0)),
        CtxLimit::Packing => format!("the launch-argument bit-packings (max_pos must stay below {} = 2^20)",
                                     crate::exl3_forward::MAX_POS_PACKED),
    };
    format!("[warn] --max-seq-len {asked} exceeds {limit} - clamping to {eff}.")
}

/// The context this EXL3 serve honours for `model_dir`: `--max-seq-len` trimmed to the rope and
/// packing ceilings, with one log line naming the limit when it was trimmed. Every rank of a TP serve
/// resolves it from the same argv and the same pack config (`build_serve`), and the head ships the
/// trimmed value in `TpConfig.max_seq_len`.
pub fn clamp_max_seq_len(model_dir: &str, max_seq_len: usize) -> usize {
    let (eff, msg) = resolve_max_seq_len(model_dir, max_seq_len);
    if let Some(m) = msg { crate::drprintln!("{m}"); }
    eff
}

/// `clamp_max_seq_len` without the log line: (window, the line to print when it was trimmed).
pub fn resolve_max_seq_len(model_dir: &str, max_seq_len: usize) -> (usize, Option<String>) {
    let native = native_window(model_dir);
    let yarn = yarn_factor(model_dir);
    let (eff, why) = clamp_ctx(max_seq_len, native, yarn);
    (eff, why.map(|w| clamp_ctx_message(max_seq_len, eff, w, native, yarn)))
}

#[cfg(test)]
mod ctx_clamp_tests {
    use super::*;

    /// YARN1: the pack config's own rope_scaling feeds the factor (native 262144 x 2 = 524288,
    /// still under the packing ceiling). The flag/TP precedence is unit-tested at
    /// qwen::yarn_tests; this covers the config-file source only.
    #[test]
    fn the_pack_rope_scaling_raises_the_ceiling() {
        let dir = std::env::temp_dir().join(format!("gb10_yarn_clamp_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let d = dir.to_str().unwrap();
        std::fs::write(dir.join("config.json"),
            r#"{"text_config":{"max_position_embeddings":262144,"rope_scaling":{"type":"yarn","factor":2.0}}}"#).unwrap();
        // 400_000 is UNDER the 524_288 ceiling: served as asked, no trim. 600_000 trims to it.
        assert_eq!(clamp_max_seq_len(d, 400_000), 400_000);
        assert_eq!(clamp_max_seq_len(d, 600_000), 524_288);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    const W: usize = 262_144; // the Qwen3.8 family's trained window

    /// Without a rope extension the context stops at the model's trained window.
    #[test]
    fn the_context_stops_at_the_trained_window() {
        assert_eq!(clamp_ctx(300_000, Some(W), 1.0), (W, Some(CtxLimit::Rope)));
        assert_eq!(clamp_ctx(W + 1, Some(W), 1.0), (W, Some(CtxLimit::Rope)));
        assert_eq!(clamp_ctx(W, Some(W), 1.0), (W, None));
        assert_eq!(clamp_ctx(32_768, Some(W), 1.0), (32_768, None));
    }

    /// A rope factor raises the ceiling until the packings stop it: max_pos = max_seq_len + headroom
    /// must stay below 2^20.
    #[test]
    fn a_rope_factor_raises_the_ceiling_to_the_packing_limit() {
        let top = packable_max_seq_len();
        assert_eq!(top + crate::batch::decode_headroom(false), crate::exl3_forward::MAX_POS_PACKED - 1);
        assert_eq!(clamp_ctx(600_000, Some(W), 2.0), (2 * W, Some(CtxLimit::Rope)));
        assert_eq!(clamp_ctx(1_048_576, Some(W), 4.0), (top, Some(CtxLimit::Packing)));
        assert_eq!(clamp_ctx(top, Some(W), 4.0), (top, None));
        // a factor below 1 never shrinks the trained window
        assert_eq!(clamp_ctx(W, Some(W), 0.5), (W, None));
    }

    /// An unreadable config must not disable the guard: the packing ceiling still applies.
    #[test]
    fn an_unknown_window_still_gets_the_packing_ceiling() {
        let top = packable_max_seq_len();
        assert_eq!(clamp_ctx(usize::MAX, None, 1.0), (top, Some(CtxLimit::Packing)));
        assert_eq!(clamp_ctx(W, None, 1.0), (W, None));
    }

    /// Whatever is asked, the window the clamp returns loads: `FwdModel::load_tp` accepts its max_pos.
    #[test]
    fn every_clamped_window_passes_the_loader_guard() {
        for native in [None, Some(4096), Some(W), Some(1 << 20), Some(1 << 24)] {
            for f in [1.0, 2.0, 4.0, 64.0] {
                for asked in [1usize, 4096, W, W + 3, 524_288, 1_048_574, 1_048_575, 1 << 20, 1 << 30] {
                    let (eff, _) = clamp_ctx(asked, native, f);
                    assert!(eff <= asked);
                    let max_pos = eff + crate::batch::decode_headroom(false);
                    assert!(crate::exl3_forward::check_max_pos_packable(max_pos).is_ok(), "{asked} {native:?} x{f} -> {eff}");
                }
            }
        }
    }

    /// The log line names the limit that applied (and only mentions a rope factor when one is set).
    #[test]
    fn the_message_names_the_limit() {
        let m = clamp_ctx_message(300_000, W, CtxLimit::Rope, Some(W), 1.0);
        assert_eq!(m, "[warn] --max-seq-len 300000 exceeds the model's trained window 262144 (max_position_embeddings) - clamping to 262144.");
        let m = clamp_ctx_message(600_000, 2 * W, CtxLimit::Rope, Some(W), 2.0);
        assert!(m.contains("trained window 262144 x rope factor 2") && m.ends_with("clamping to 524288."), "{m}");
        let m = clamp_ctx_message(2_000_000, 1_048_574, CtxLimit::Packing, None, 1.0);
        assert!(m.contains("bit-packings (max_pos must stay below 1048576 = 2^20)") && m.ends_with("clamping to 1048574."), "{m}");
    }

    /// Wiring pin: the serve builder and the TP head resolve --max-seq-len through the clamp (the
    /// served path itself is the GPU gate: a 524288 boot loads at 262145 on every rank).
    #[test]
    fn the_serve_and_the_tp_head_go_through_the_clamp() {
        let code = |src: &str| src.lines().filter(|l| !l.trim_start().starts_with("//")).collect::<Vec<_>>().join("\n");
        let serve = code(include_str!("exl3_serve.rs"));
        let needle = ["let max_seq_len: usize = clamp_max_seq_len(model_dir, ", "arg(args, \"--max-seq-len\")"].concat();
        assert_eq!(serve.matches(&needle).count(), 1, "build_serve");
        let main = code(include_str!("main.rs"));
        let needle = ["tpc.max_seq_len = gb10_inference::exl3_serve::", "resolve_max_seq_len("].concat();
        assert_eq!(main.matches(&needle).count(), 1, "run_exl3_tp_serve_head");
    }

    /// The trained window is read from the pack's config.json, flat or nested under text_config.
    #[test]
    fn the_trained_window_is_read_from_the_pack_config() {
        let dir = std::env::temp_dir().join(format!("gb10_ctx_clamp_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let d = dir.to_str().unwrap();
        assert_eq!(native_window(d), None, "no config.json");
        std::fs::write(dir.join("config.json"), r#"{"max_position_embeddings": 262144}"#).unwrap();
        assert_eq!(native_window(d), Some(W));
        std::fs::write(dir.join("config.json"), r#"{"text_config": {"max_position_embeddings": 131072}, "max_position_embeddings": 7}"#).unwrap();
        assert_eq!(native_window(d), Some(131_072));
        assert_eq!(clamp_max_seq_len(d, 200_000), 131_072);
        assert_eq!(clamp_max_seq_len(d, 65_536), 65_536);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

/// Estimate the footprint of an EXL3 pack from its files and config (no GPU, no weight reads).
pub fn fit_parts(model_dir: &str, max_pos: usize, world: usize, head: bool, kv_bytes: f64, ckpt_gb: f64) -> Option<FitParts> {
    let dir = std::path::Path::new(model_dir);
    let size = |n: &str| std::fs::metadata(dir.join(n)).map(|m| m.len()).unwrap_or(0);
    let idx: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json")).ok()?).ok()?;
    let all: std::collections::BTreeSet<String> = idx["weight_map"].as_object()?.values()
        .filter_map(|v| v.as_str().map(str::to_string)).collect();
    // the n-gram table is counted as `table`, not device weights (in a shipped shard dir it sits under seg/)
    let (ngram, files): (std::collections::BTreeSet<String>, std::collections::BTreeSet<String>) =
        all.into_iter().partition(|f| f.contains("ngram"));
    let shards: u64 = files.iter().map(|f| size(f)).sum();
    // A shipped per-rank shard dir (B32, `gb10_shard.json`) already holds this rank's share; a full pack is split by world.
    let share = if dir.join("gb10_shard.json").is_file() { 1 } else { world.max(1) as u64 };
    let mut weights = shards / share;
    if head { weights += size("vision_tower_bf16.safetensors"); }
    let mut table: u64 = ngram.iter().map(|f| size(f)).sum();
    if table == 0 {
        table = std::fs::read_dir(dir).ok()?.flatten()
            .filter(|e| e.file_name().to_string_lossy().contains("ngram"))
            .map(|e| e.metadata().map(|m| m.len()).unwrap_or(0)).sum();
    }
    let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json")).ok()?).ok()?;
    let t = cfg.get("text_config").unwrap_or(&cfg);
    let g = |k: &str| t[k].as_u64().unwrap_or(0);
    let types = t["layer_types"].as_array()?;
    let n_full = types.iter().filter(|v| v.as_str() == Some("full_attention")).count() as u64;
    let n_lin = types.len() as u64 - n_full;
    let kvh_hd = g("num_key_value_heads") * g("head_dim");
    // per token: trunk K+V (kv-cache format) + indexer keys (bf16) + the MTP head's own K+V (f16)
    // trunk KV heads are split across ranks up to their count (2 KV heads: TP=2 and TP=4 hold one each)
    let kv_split = (world.max(1) as u64).min(g("num_key_value_heads").max(1)) as f64;
    let per_tok = n_full as f64 * 2.0 * kvh_hd as f64 * kv_bytes / kv_split
        + (n_full * g("indexer_kv_heads") * g("indexer_head_dim") * 2) as f64
        + (g("mtp_num_hidden_layers").max(1) * 2 * kvh_hd * 2) as f64;
    // per lane fixed: GDN recurrent state (f32) + conv ring
    let state = n_lin * g("linear_num_value_heads") * g("linear_key_head_dim") * g("linear_value_head_dim") * 4
        + n_lin * (2 * g("linear_num_key_heads") * g("linear_key_head_dim") + g("linear_num_value_heads") * g("linear_value_head_dim"))
            * g("linear_conv_kernel_dim").max(1) * 2;
    let per_lane = ((per_tok * max_pos as f64 + state as f64) * 1.15) as u64;
    // Working memory beyond weights + table + lanes, measured on .14 2026-10-02 (Flash-Next 3.05 TP=1, 131K): 14.3 GB
    // at 1 lane and 14.8 GB at 8 lanes after the boot (scratch, prefill scratch, graphs, vendor workspaces, staging),
    // + the prefix-checkpoint pool, which grows to its cap while serving.
    let fixed = 15_000_000_000 + (ckpt_gb * (1u64 << 30) as f64) as u64;
    // MAXBATCH1 (2026-10-21, .12, 27B EXL3 3.05, TP=1, posture ctx 262144, 8-lane clamp): serving-time memory
    // grew ~6.4 GB BEYOND this model under one 32-concurrency cell (CUDA-graph captures at mixed widths,
    // prefix/snapshot churn, the queue) and MemAvailable fell to 4.4 GB — the memwatch emergency exit fired and
    // the server died mid-traffic. The 5 GB watchdog floor is the EMERGENCY bar (box protection), not a serving
    // plan. The serve fit reserves the owner's §5 serving-headroom bar instead: ~20 GB free per box on a serving
    // posture ("lower is good"); the watchdog keeps its own floor as the last resort below that.
    let watch_gb: f64 = crate::opts::var(crate::opt!("mem-watchdog-gb")).ok().and_then(|v| v.parse().ok()).unwrap_or(5.0);
    let floor_gb = watch_gb.max(20.0);
    Some(FitParts { weights, table, per_lane, fixed, floor: (floor_gb * 1e9) as u64 })
}

/// The served `--prefill-chunk` default (owner 2026-10-02, v0.7.1: steady 4,095-row chunks, TP-4X1).
pub const DEFAULT_PREFILL_CHUNK: usize = 4095;


/// EXL27 (owner scope add 2026-10-16): single-file EXL3 packs (one model.safetensors, no
/// model.safetensors.index.json) are SUPPORTED, not refused (§1b-bis 8). We synthesize the
/// one-shard index from the safetensors header — exactly what HF emits for a 1-shard model —
/// so the standard indexed-pack path serves them unchanged. Returns true if an index was written.
pub fn ensure_singlefile_index(dir: &str) -> std::io::Result<bool> {
    use std::io::Read;
    let p = std::path::Path::new(dir.trim_end_matches('/'));
    let idx = p.join("model.safetensors.index.json");
    let st = p.join("model.safetensors");
    if idx.exists() || !st.exists() {
        return Ok(false);
    }
    let mut f = std::fs::File::open(&st)?;
    let mut hlen = [0u8; 8];
    f.read_exact(&mut hlen)?;
    let n = u64::from_le_bytes(hlen) as usize;
    if n == 0 || n > 128_000_000 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad safetensors header length"));
    }
    let mut hdr = vec![0u8; n];
    f.read_exact(&mut hdr)?;
    let v: serde_json::Value = serde_json::from_slice(&hdr)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("header JSON: {e}")))?;
    let mut wm = serde_json::Map::new();
    for k in v.as_object().map(|o| o.keys()).into_iter().flatten() {
        if k == "__metadata__" {
            continue;
        }
        wm.insert(k.clone(), serde_json::Value::String("model.safetensors".to_string()));
    }
    if wm.is_empty() {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "no tensors in header"));
    }
    let total = std::fs::metadata(&st)?.len();
    let out = serde_json::json!({ "metadata": { "total_size": total }, "weight_map": wm });
    let tmp = p.join(".model.safetensors.index.json.tmp-exl27");
    std::fs::write(&tmp, serde_json::to_vec_pretty(&out)?)?;
    std::fs::rename(&tmp, &idx)?; // atomic: no partial index is ever observed
    Ok(true)
}

/// Detection: a dir is an EXL3 pack iff quantization_config.json exists and the safetensors
/// index carries `.trellis` entries (the A3 spec's quadruple manifest).
pub fn is_exl3_pack(dir: &str) -> bool {
    let dir = dir.trim_end_matches('/');
    let qc = std::path::Path::new(dir).join("quantization_config.json");
    if !qc.exists() {
        return false;
    }
    let idx = std::path::Path::new(dir).join("model.safetensors.index.json");
    if !idx.exists() {
        return false;
    }
    std::fs::read_to_string(&idx).map(|s| s.contains(".trellis")).unwrap_or(false)
}

/// Everything a booted serve needs: the scheduler, the HTTP state, and the host-side knobs the
/// launch applies (TP-D: shared by the TP=1 server, the TP=2 head and the TP=2 node mirror).
struct ServeParts {
    sched: Exl3Scheduler,
    state: AppState,
    port: u16,
    cpu_aff: (Option<Vec<usize>>, String),
    exit_on_fatal: bool,
    rx: UnboundedReceiver<BatchRequest>,
    /// OUT1: what the one SERVING line needs at listen time (tp, ctx, batch, kv, prefix,
    /// mtp-head-present, mtp-off).
    boot_facts: (u32, usize, usize, String, bool, bool, bool, bool),
}

/// Boot the EXL3 HTTP server (TP=1). Mirrors run_server's serving surface.
pub fn run(args: &[String], model_dir: &str) -> Result<()> {
    let parts = build_serve(args, model_dir, None)?;
    serve_http(parts, None)
}

/// TP-D: resolve every serve knob from `args`, load (TP: this rank's shard) and build the
/// scheduler + HTTP state. At TP the head and the node run this with the SAME args (the head ships
/// its argv in TpConfig.exl3_mode), so both resolve every knob identically; `tp_boot_agree` then
/// proves the resolved scheduler config equal across ranks before a request is served.
fn build_serve(args: &[String], model_dir: &str, tp: Option<crate::exl3_forward::xtp::TpAttach>) -> Result<ServeParts> {
    let tp_rank: Option<i32> = tp.as_ref().map(|a| a.rank);
    let tp_world: u32 = tp.as_ref().map_or(1, |a| a.world as u32);
    let who = match tp_rank { Some(r) => format!("TP={tp_world} rank {r}"), None => "TP=1".to_string() };
    // v0.7.3 ("tel"): --otel-* on the EXL3 server, the SAME shared parser as the NVFP4 path
    // (otel::config_from_args). Parsed + refused on EVERY rank BEFORE the model load (a bad
    // endpoint aborts the boot); the SINK is built later, only where the HTTP API lives.
    let otel_cfg = crate::otel::config_from_args(
        |f| arg(args, &format!("--{f}")).map(|s| s.to_string()));
    if let Some(cfg) = &otel_cfg {
        if let Err(e) = cfg.hostport() { crate::deprintln!("[otel] {e}"); crate::logq::flush_and_exit(std::time::Duration::from_millis(300), 1); } // H7
    }
    // S-B9-CF-PACK4: --lane-pack, parsed before the model load. F10 (owner 2026-10-12): the registry default is
    // ON, so the mode is resolved from the TYPED value (this head's argv; the node re-parses the head's argv) and
    // otherwise from the registry row: both ranks of one binary resolve the same mode with no flag typed.
    // Only a TYPED `--lane-pack on` above the largest world is refused (as before); a defaulted-on boot that
    // cannot pack, for that or any other reason, falls back with one line (AGENTS 1b-bis item 8).
    let lane_pack_typed = arg(args, "--lane-pack");
    let mut lane_pack = match lane_pack_typed {
        Some(v) => crate::lane_pack::LanePack::parse(Some(v))?,
        None => crate::lane_pack::LanePack::from_opts()?,
    };
    if lane_pack_typed.is_some() { crate::lane_pack::tp_refusal(lane_pack, tp_world)?; }
    let port: u16 = arg(args, "--port").and_then(|s| s.parse().ok()).unwrap_or(8000);
    // Public PR #21: trim --max-seq-len to what the engine can address (the trained window, then the
    // 20-bit packings) and say which limit applied; an unclamped wider window used to boot.
    // YARN1: set the trunk YaRN factor BEFORE both — the load (the EXL3 tables build at load)
    // and the clamp (its rope ceiling is native x factor). The EXL3 TP node re-parses this
    // same argv and the value also ships in TpConfig, so both ranks resolve identically.
    if let Some(f) = arg(args, "--rope-yarn-factor") {
        let f: f32 = f.parse().map_err(|_| anyhow::anyhow!(
            "--rope-yarn-factor must be a finite number >= 1.0 (got {f:?})"))?;
        if !f.is_finite() || f < 1.0 {
            anyhow::bail!("--rope-yarn-factor must be a finite number >= 1.0 (got {f})");
        }
        crate::opts::set(crate::opt!("rope-yarn-factor"), f.to_string());
    }
    let st_config = crate::boot_report::Stage::begin("config");
    let max_seq_len: usize = clamp_max_seq_len(model_dir, arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(4096));
    // F2 (owner 2026-10-10): at most 16 requests are served at once; a larger --max-batch boots
    // with 16 and further requests wait in the queue (--max-waiting semantics unchanged).
    // MAXBATCH1: `width` is `mut` — the memory fit below may boot it down to the largest lane
    // count that fits (§1b-bis#8 boot-anyway; the context length is never silently cut).
    let (mut width, raw_width) = resolved_max_batch(args);
    crate::metrics::set_max_batch_requested(raw_width);
    if raw_width > width {
        crate::boot_report::notice(&format!("[exl3-serve] --max-batch {raw_width}: this server serves at most {EXL3_MAX_LANES} \
                           requests at the same time; using {width}; further requests wait in the queue"));
    }
    // (backstop: the TP entry points resolve the same cap before any network step,
    // `tp_lane_width_refusal`; 16 always fits the pairwise agree width)
    tp_lane_width_fits(tp_world, width).map_err(|e| anyhow::anyhow!(e))?;
    // S-B9-CF-AGREE16B (W3): at TP=1 there is no TpSync to say what a lockstep drill flag does: say it here
    if tp_rank.is_none() {
        for l in crate::net::drill_plan(crate::opts::get(crate::opt!("tp-agree-drill")).as_deref(),
                                        crate::opts::get(crate::opt!("tp-width-drill")).as_deref(), true, None).lines {
            crate::boot_report::notice(&l);
        }
    }
    let max_pos = max_seq_len + crate::batch::decode_headroom(false);
    // --prefix-cache on|off (default on, as the NVFP4 server): per-slot recurrent-state snapshots
    // at the message boundary so a multi-turn re-send prefills only its new suffix.
    let prefix_on = !matches!(arg(args, "--prefix-cache"), Some("off"));
    if prefix_on { crate::opts::set(crate::opt!("exl3-prefix"), "1"); }
    // WP16: --prefix-ckpt-mem-gb <GiB> — the LRU byte cap of the intermediate recurrent checkpoints
    // (allocated lazily up to it). Default 4 (placeholder: the owner sets the final default);
    // 0 = no checkpoints (today's behaviour, as --wp16-off=1).
    let wp16_gb: f64 = match arg(args, "--prefix-ckpt-mem-gb") {
        None => 4.0,
        Some(v) => v.parse().map_err(|_| anyhow::anyhow!("--prefix-ckpt-mem-gb must be a number (GiB)"))?,
    };
    anyhow::ensure!(wp16_gb.is_finite() && wp16_gb >= 0.0, "--prefix-ckpt-mem-gb must be >= 0");
    // --ple-ram on|off|auto: where the ~30 GiB PLE n-gram table lives (RAM vs per-token pread
    // through the page cache). Output-identical either way; transported to the loader via env.
    if let Some(v) = arg(args, "--ple-ram") {
        anyhow::ensure!(matches!(v, "auto" | "ram" | "ssd" | "on" | "off"), "--ple-ram must be auto, ram or ssd (on/off are the old spellings of ram/ssd)");
        crate::opts::set(crate::opt!("ple-ram"), v);
    }
    // --kv-cache f32|f16|fp8|q8: attention KV storage format (PLAN/KV_CACHE_FORMATS.md). Default
    // q8 (owner decision 2026-09-26, after the WP25 quality gate); f32 is exact; f16 halves, fp8 (e4m3 + per-row f32 scale) and q8 (the reference implementation's H32-rotated int8 +
    // f16 per-32 scale, WP25) quarter the long-context KV bytes. Transported to the loader via
    // [kv-cache] (an explicit env value wins when the flag is absent).
    if let Some(v) = arg(args, "--kv-cache") {
        anyhow::ensure!(matches!(v, "f32" | "f16" | "fp8" | "q8"), "--kv-cache must be f32, f16, fp8 or q8");
        crate::opts::set(crate::opt!("kv-cache"), v);
    }
    // --spec-source <s> (ISSUES[94]): the mirror of the trunk resolve_spec_source (main.rs) —
    // same value set (mtp|dspark|dflash2|dflash2-rq|dflash2-auto|dflash2-tree|none), same
    // default (dflash2-auto), same error text — parsed HERE so the EXL3 serve path honors what
    // it was asked for instead of silently ignoring the DF2 family. CLI flags only, never env.
    let df2_src = resolve_spec_source_exl3(args);
    // --draft-dir <dir>: the DFlash2 artifact directory, the same flag the NVFP4 serving path
    // takes. (main's pre-dispatch resolve_draft_dir already exits on an explicit
    // artifact-REQUIRING source with no dir; dflash2-auto — the default — without a dir is the
    // silent-ignore case this gate fixes, with a loud FALLBACK line.)
    // FLASHNEXT-DF2-C (ISSUES 116): under an EXL3 TP serve the SHIPPED config decides the draft
    // dir (the head's absolute path on rank 0, the blob-cache dir the sync assembled on a node,
    // empty = the head shipped no drafter); a node never opens the head's filesystem path.
    let df2_draft_dir = df2_draft_dir_for_boot(args, crate::tp::tp_config());
    // --df2-block <n>: the artifact's block size (default: dflash2::block(), 8; 16 on the wide
    // selector artifact). The round's chain-verify depth is k = block - 1.
    let mut df2_block = parse_df2_block(arg(args, "--df2-block"))?;
    if args.iter().any(|a| a == "--df2-trace") {
        crate::exl3_forward::DF2_TRACE.store(true, std::sync::atomic::Ordering::Relaxed);
        println!("[exl3-serve] df2: trace on (--df2-trace: stage syncs, rounds slower)");
    }
    // --df2-round-shard on|off (default off): the TP>2-only round sharding — parsed for flag
    // parity and shown on the DF2 boot line; it never applies on this TP=1 path.
    let df2_shard = parse_df2_round_shard(arg(args, "--df2-round-shard"))?;
    let kv_fmt = crate::exl3_forward::kv_fmt_from_opts();
    anyhow::ensure!(kv_fmt != crate::exl3_forward::KV_Q4X,
                    "[kv-cache]=q4x is the WP25 probe's deliberately broken control; it is never served");
    let kv_name = crate::exl3_forward::kv_fmt_name(kv_fmt);
    // PACK4-REVIEW B3, now F1 (owner 2026-10-10: "support things properly or gracefully fall back"):
    // the postures --lane-pack on was never gated under no longer REFUSE the boot. The reasons are
    // collected here — before the model load, so nothing packed is ever built for a boot that will
    // not pack — and the boot decides just before the packed scratch: ONE plain line, packed rounds
    // OFF for this boot, and the decision rides /metrics and the TP boot agree (a rank-local
    // difference still dies loudly there).
    let mut pack_blockers: Vec<String> = Vec::new();
    if lane_pack.on() && tp_world > crate::lane_pack::LANE_PACK_MAX_WORLD {
        pack_blockers.push(format!("TP={tp_world}: the packed arm has a lockstep frame for at most {} machines",
                                   crate::lane_pack::LANE_PACK_MAX_WORLD));
    }
    if lane_pack.on() {
        pack_blockers.extend(crate::lane_pack::posture_fallbacks(&crate::lane_pack::Posture {
            kv: kv_name,
            mtp_k: crate::exl3_forward::mtp_depth_default(),
            row_regime: crate::exl3_forward::row_regime_on(),
            a_ctrl_old: crate::exl3_forward::a_flags().ctrl == 2,
            fcfs: LaneOrder::parse(crate::opt_or_arg(args, "--lane-order").as_deref().map(str::trim).filter(|v| !v.is_empty()))? == LaneOrder::Fcfs,
        }));
    }
    // --reasoning-effort / --thinking: the server-wide thinking defaults, as on the NVFP4 server.
    // Unset = the model's own chat template decides (Qwen3.8: thinking on at xhigh, the model
    // card's default). A request's reasoning_effort / chat_template_kwargs.enable_thinking wins.
    let reasoning_effort = match arg(args, "--reasoning-effort") {
        None => None,
        Some(e @ ("none" | "no_think" | "low" | "medium" | "high" | "xhigh")) => Some(e.to_string()),
        Some(other) => anyhow::bail!("--reasoning-effort must be none|low|medium|high|xhigh (got '{other}')"),
    };
    let thinking = match arg(args, "--thinking") {
        None => crate::tokenizer::ThinkingMode::Auto,
        Some(v) => crate::tokenizer::ThinkingMode::parse(v)
            .ok_or_else(|| anyhow::anyhow!("--thinking must be auto|on|off (got '{v}')"))?,
    };
    // HOST / RO-7: --cpu-affinity auto|off (default auto; src/cpu_affinity.rs). Resolved here so
    // a bad value fails before the load; applied after it (the load's threads keep every core).
    let cpu_aff = crate::cpu_affinity::resolve(
        crate::cpu_affinity::AffinityMode::parse(arg(args, "--cpu-affinity"))?)?;
    // MAXBATCH1 (owner ruling 2026-10-10 §1b-bis#8: "support it properly or fall back gracefully —
    // no refusals"): resolve the memory fit BEFORE the tune table is selected (the table is keyed
    // by the lane count that will actually serve). An asked lane count that does not fit memory
    // BOOTS ANYWAY on the nearest thing that works — the largest lane count that fits at the
    // asked context — with ONE clear line: what was asked, what it does instead, why. The context
    // length is never silently cut (full-context serving is the READY bar; a lane waits instead).
    // CF-P1d keeps its teeth: the refusal remains when not even ONE lane fits, with the
    // arithmetic and what to change (that boot could not serve anything at all).
    {
        let kvb = match kv_name { "f32" => 4.0, "f16" => 2.0, "fp8" => 1.0, _ => 1.0625 };
        let mode = match crate::opts::var(crate::opt!("ple-ram")).as_deref() { Ok("on") | Ok("ram") => "ram", Ok("off") | Ok("ssd") => "ssd", _ => "auto" };
        let world = tp_world.max(1) as usize;
        if let Some(parts) = fit_parts(model_dir, max_pos, world, tp_rank.map_or(true, |r| r == 0), kvb, wp16_gb) {
            let avail = std::fs::read_to_string("/proc/meminfo").ok()
                .and_then(|m| m.lines().find(|l| l.starts_with("MemAvailable:"))
                    .and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<u64>().ok()))
                .unwrap_or(0) * 1024;
            if avail > 0 {
                if let Err(ref msg) = fit_plan(parts, width, avail, mode, max_pos) {
                    if let Some(n) = fit_max_lanes(parts, width, avail, mode, max_pos) {
                        let gb = |b: u64| b as f64 / 1e9;
                        crate::boot_report::notice(&format!(
                            "[exl3-serve] --max-batch {raw_width}: {width} lanes do not fit memory at this context length on this box \
                             ({avail_g:.1} GB available; asked-for plan: weights {w_g:.1} GB + {width} x {pl_g:.2} GB per-lane KV + {fx_g:.1} GB \
                             working memory + {fl_g:.1} GB serving headroom); booting with {n} lanes — the most that fit (the context length is \
                             kept); further requests wait in the queue",
                            avail_g = gb(avail), w_g = gb(parts.weights), pl_g = gb(parts.per_lane),
                            fx_g = gb(parts.fixed), fl_g = gb(parts.floor)));
                        width = n;
                    } else {
                        eprintln!("error: {msg}");
                        std::process::exit(2);
                    }
                }
            }
        }
    }
    // TUNE T0 (PLAN/AUTOTUNE_DESIGN.md §7): resolve + freeze the tune table BEFORE the load (Load-scope
    // knobs are read there). --tune-table auto (default: a matching table from ./tune/, S section) |
    // off (built-in defaults) | <path>; --tune-draft on also applies section D (draft-only);
    // --tune-profile <MHz> picks the clock profile (default: detected, else the 2400 MHz stock one).
    // A missing or mismatched table is a loud FALLBACK to the defaults (== today, bitwise).
    {
        use crate::exl3_tune as tune;
        let sel = tune::TableSel::parse(arg(args, "--tune-table"))?;
        let draft_on = match arg(args, "--tune-draft") {
            None | Some("off") => false,
            Some("on") => true,
            Some(o) => anyhow::bail!("--tune-draft must be on or off (got '{o}')"),
        };
        let profile_mhz = match arg(args, "--tune-profile") {
            None => None,
            Some(v) => Some(v.parse::<u32>().map_err(|_| anyhow::anyhow!("--tune-profile must be MHz (got '{v}')"))?),
        };
        let posture = tune::Posture {
            lanes: width,
            mtp_max_k: crate::exl3_forward::mtp_depth_default(),
            dds: arg(args, "--draft-confidence").and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.4) > 0.0,
            kv_fmt: crate::exl3_forward::kv_fmt_from_opts(),
            max_pos,
            prefill_chunk: arg(args, "--prefill-chunk").and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_PREFILL_CHUNK),
        };
        tune::boot(&tune::BootReq { sel, model_dir, posture, profile_mhz, draft_on,
                                    tp: tp_world });
    }
    // (CF-P1d + MAXBATCH1: the memory fit ran above, before the tune table — boot-anyway clamp.)
    st_config.ok();
    let st_weights = crate::boot_report::Stage::begin("weights");
    if crate::boot_report::debug() {
        crate::dprintln!("[exl3-serve] {who}: loading EXL3 pack {model_dir} (width {width}, max_pos {max_pos}, kv-cache {kv_name})");
    }
    let mut model = FwdModel::load_tp(model_dir, width, max_pos, tp)?;
    st_weights.ok();
    // ---- ISSUES[94]: the DFlash2 (DF2) boot gate + round load. Exactly ONE line per boot —
    // ENGAGED or FALLBACK with the reason — and a load failure is a FALLBACK, never a dead
    // boot (the pack MTP this server already has serves every request unchanged).
    let mut df2: Option<crate::dflash2::round::Df2Round> = None;
    match df2_gate_reason(df2_src, df2_draft_dir.as_deref(), model.cfg.hidden_size,
                          model.cfg.vocab_size, model.cfg.num_layers, max_seq_len,
                          tp_world as usize) {
        Some(reason) => {
            println!("{}", df2_boot_line_fallback(&reason));
            // FLASHNEXT-DF2-B (TP=2 boot 2026-10-23): a node re-parses the head's argv and reads
            // --draft-dir from ITS OWN disk (the EXL3 TP sync ships the pack, not the drafter). A
            // node without the artifact fell to the 27B default geometry, reported a "dims
            // mismatch" and served MTP while the head ENGAGED — the pair diverged at the first
            // round. Say so, next to the reason.
            if let Some(dir) = df2_draft_dir.as_deref() {
                if !std::path::Path::new(dir).join("config.json").is_file() {
                    println!("[exl3-serve] df2: note — {dir}/config.json is not readable on this box, so the dims above assume the 27B default geometry; under TP every rank needs the artifact at this path (rank {})", tp_rank.map_or(0, |r| r as usize));
                }
            }
        }
        None => {
            // EXL27C: the unsharded round kernel's ring band is structurally 8 rows
            // (write_kv_b ring rows [RING, RING+8)); block 16 is the SHARDED (TP) shape.
            // Unsharded TP=1 boots clamp to 8 with one clear line (sanitizer: gqa_attn_band_ring_b
            // OOB __shared__ writes at block 16 unsharded, 2026-10-16).
            if !df2_shard && df2_block > 8 {
                crate::dflash2::set_block(8).ok();
                println!("[exl3-serve] df2: --df2-block {df2_block} without --df2-round-shard runs block 8 unsharded (the round kernel's band width)");
                df2_block = 8;
            }
            // sha_pin: None = the built-in published pin (load_tp_pinned applies it), the
            // trunk's --sha256 handling exactly (Some("off") disables, Some(hex) pins).
            let pin = arg(args, "--sha256");
            let dir = df2_draft_dir.clone().unwrap_or_default();
            match df2_load_round(&mut model, &dir, max_seq_len, pin) {
                Ok(mut r) => {
                    // S5F parity (batch.rs:1682): capture the draft-round CUDA graph once at boot.
                    // The eager EXL3 chain measured 149 ms/round (df2-trace, 2026-10-18, .12);
                    // the graph replay collapses launch/sync overhead. Opt-out: --no-df2-graph.
                    if crate::opts::var(crate::opt!("no-df2-graph")).is_err() {
                        if r.capture_round_graph() {
                            println!("[exl3-serve] df2: draft-round CUDA graph captured (eager fallback via --no-df2-graph)");
                        } else {
                            println!("[exl3-serve] df2: draft-round graph capture unsupported — staying eager");
                        }
                    }
                    let sha = crate::dflash2::load::last_load_sha().unwrap_or("unverified");
                    println!("{}", df2_boot_line_engaged(&dir, sha, df2_block, df2_shard, tp_world as usize));
                    df2_mark_engaged();
                    df2 = Some(r);
                }
                Err(e) => println!("{}", df2_boot_line_fallback(&format!("load failed: {e:#}"))),
            }
        }
    }
    // FLASHNEXT-DF2-B: `--df2-k <n>` caps the DF2 verify width (the walk still drafts block-1
    // tokens; only the first n are verified). Default = block-1. Applied and shown, never a
    // refusal (AGENTS §1b-bis.8): out-of-range values clamp into 1..=block-1.
    let df2_k_max = (df2_block - 1).min(crate::exl3_forward::MTP_MAX_K);
    let df2_k = match arg(args, "--df2-k") {
        None => df2_k_max,
        Some(v) => {
            let want: usize = v.parse().unwrap_or(df2_k_max);
            let k = want.clamp(1, df2_k_max);
            if df2.is_some() {
                println!("[exl3-serve] df2: verify width k={k} (--df2-k {v}{}; default {df2_k_max} = block-1; the chain still drafts {} and verifies the first k)",
                         if k != want { ", clamped" } else { "" }, df2_block - 1);
            }
            k
        }
    };
    // SURFACE2 item 4: the boot lines that describe HOW this server speculates print only when it
    // does. With --mtp off (or the hatch) the WP23 DDS line, the lane-order line, the shared-step
    // line and the spec-sampling line described rounds that never run.
    let spec_live = spec_live(model.mtp.is_some(), crate::exl3_forward::mtp_off_by_opt());
    if let Some(line) = head_loaded_line(model.mtp.is_some(), crate::exl3_forward::mtp_off_by_opt()) {
        if crate::boot_report::debug() {
            crate::dprintln!("{line}");
        }
    }
    let st_tok = crate::boot_report::Stage::begin("tokenizer");
    let tok = QwenTokenizer::from_file(&format!("{}/tokenizer.json", model_dir.trim_end_matches('/')))?;
    st_tok.ok();
    // Stop tokens = the §3 union: generation_config eos (int or list) + tokenizer/config eos.
    let cfg_eos: u32 = {
        let cfg: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(format!("{}/config.json", model_dir.trim_end_matches('/')))?,
        )?;
        cfg.get("eos_token_id").and_then(|e| e.as_u64()).unwrap_or(248046) as u32
    };
    let eos = tok.stop_token_ids(cfg_eos);
    crate::dprintln!("[exl3-serve] stop_ids={eos:?}");

    let mtp_k = crate::exl3_forward::mtp_depth_default();
    // API-parity DDS: dynamic draft stop (the reference implementation's -dds -dc 0.6). CLI --draft-confidence <p>;
    // 0 = fixed-depth chain. Output-neutral (greedy lossless, sampled distribution-exact).
    // Default 0.4, not the reference implementation's 0.6: our draft pass is cheaper relative to a verify row, so
    // longer drafts pay (sweep 2026-09-25, k5 greedy 1024 tok: 0.4 -> ansic 87.9 / python 77.6 /
    // prose 55.8; 0.6 -> 85.2 / 74.2 / 55.6; 0.2-0.7 all within +-2).
    let draft_conf: f64 = arg(args, "--draft-confidence").and_then(|s| s.parse().ok()).unwrap_or(0.4);
    anyhow::ensure!((0.0..1.0).contains(&draft_conf), "--draft-confidence must be in [0, 1)");
    // WP08 (owner decision 2026-09-26): the reference implementation's streaming loop detector, ON by default at its
    // launcher setting stop_on_loop = (300, 3) (chat.py -lw 300 -lmr 3): a response whose last
    // 300 tokens are one repeating sequence of period <= 100 ends as a normal stop
    // (stop_reason "loop_detected"). --loop-detect off disables; --loop-window / --loop-min-reps
    // tune it (the reference implementation's asserts: window > 1, 1 < reps < window). It only changes output when
    // the model is already looping.
    let loop_cfg = match arg(args, "--loop-detect") {
        None | Some("on") => {
            let w: usize = match arg(args, "--loop-window") {
                None => 300,
                Some(v) => v.parse().map_err(|_| anyhow::anyhow!("--loop-window must be an integer"))?,
            };
            let r: usize = match arg(args, "--loop-min-reps") {
                None => 3,
                Some(v) => v.parse().map_err(|_| anyhow::anyhow!("--loop-min-reps must be an integer"))?,
            };
            anyhow::ensure!(w > 1, "--loop-window must be > 1");
            anyhow::ensure!(r > 1 && r < w, "--loop-min-reps must be > 1 and < --loop-window");
            Some((w, r))
        }
        Some("off") => None,
        Some(v) => anyhow::bail!("--loop-detect must be on or off (got '{v}')"),
    };
    // WP15: penalty window, the reference implementation's -penr (penalty_range, default 1024) used as both its
    // full-strength (sustain) and fading (decay) spans. Penalties themselves are per request.
    let pen_range: usize = match arg(args, "--penalty-range") {
        None => 1024,
        Some(v) => v.parse().map_err(|_| anyhow::anyhow!("--penalty-range must be an integer"))?,
    };
    anyhow::ensure!(pen_range <= crate::exl3_forward::PEN_RANGE_MAX,
                    "--penalty-range must be <= {}", crate::exl3_forward::PEN_RANGE_MAX);
    // WP24 (owner-consent class: seeded sampled bytes change; default match = today bit for bit).
    let (spec_ratio, draft_temp) = parse_spec_sampling(args)?;
    // Server-wide penalty defaults for requests that send none (the shared server flags; the
    // reference implementation's defaults 1.0 / 0 / 0 when absent). Same validity rules as a request's values.
    let dflt = |name: &str, d: f32| -> Result<f32> {
        match arg(args, name) {
            None => Ok(d),
            Some(v) => v.parse::<f32>().map_err(|_| anyhow::anyhow!("{name} must be a number")),
        }
    };
    let default_rep_penalty = dflt("--default-repetition-penalty", 1.0)?;
    let default_presence_penalty = dflt("--default-presence-penalty", 0.0)?;
    let default_frequency_penalty = dflt("--default-frequency-penalty", 0.0)?;
    crate::server::validate_penalties(Some(default_rep_penalty), Some(default_presence_penalty),
                                      Some(default_frequency_penalty), None)
        .map_err(|e| anyhow::anyhow!("server penalty defaults: {e}"))?;
    crate::dprintln!("[exl3-serve] loop detector {} ; penalty window {pen_range} + {pen_range} tokens \
              (defaults rep {default_rep_penalty} presence {default_presence_penalty} frequency {default_frequency_penalty})",
             match loop_cfg { Some((w, r)) => format!("ON (window {w}, min reps {r})"), None => "OFF".into() });
    // S-B9-REL-INTEG (AGENTS 1b-bis): a penalized request never joins a packed verify, and a request that sends no
    // penalty field INHERITS these server defaults. Say so at boot, where the operator chose the flags.
    if lane_pack.on() {
        if let Some(w) = lane_pack_penalty_warning(default_rep_penalty, default_presence_penalty, default_frequency_penalty) {
            crate::drprintln!("{w}");
        }
    }
    let tok = Arc::new(tok);
    // BUG-5 FIX (width half): mtp_round verifies at m=k+1 rows — the scratch must
    // hold the verify width even when the lane count is 1 (w1 serve panicked on
    // sc.slots[2] with a 2-element buffer). Model width (state slots) stays = lanes.
    let mut sc = FwdModel::scratch(model.dev(), &model.cfg, width.max(mtp_k + 1), model.max_pos())?;
    // S-A3-f-b: chunked prefill width (CLI --prefill-chunk; user-facing knob).
    // S-A3-f-d Item 2: default 512 -> 2048 — the matched-C table (7838 toks,
    // fixed flash256 kernel) shows 112 tok/s @ C=512 vs 234 @ C=2048 (the KV
    // re-scan term ~N^2/2C); serve == bench at matched C, so the chunk width is
    // the whole serve-side lever. Scratch cost is ~4x of a few hundred MB.
    // v0.7.1 (owner 2026-10-02, D-TP4-6): default 2048 -> 4095 (TP-4X1: W=4 prefill -3% at 8K, -5.4% at 32K).
    let chunk: usize = arg(args, "--prefill-chunk").and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_PREFILL_CHUNK);
    if tp_rank.is_some() {
        anyhow::ensure!(psc_rows(chunk.max(1), true) <= crate::exl3_forward::xtp::TP_MAX_ROWS,
                        "TP: --prefill-chunk {chunk} gives prefill chunks up to {} rows, above the TP all-reduce \
                         partial buffer ({} rows; C1 tail checkpoints merge up to min(2C-1, {}) rows)",
                        psc_rows(chunk.max(1), true), crate::exl3_forward::xtp::TP_MAX_ROWS,
                        crate::exl3_forward::xtp::TP_MAX_ROWS - 1);
    }
    // S-A3-u: build the prefill scratch at startup and warm cuBLASLt (lazy kernel loads + plans
    // ~0.4 s) so request 1 does not pay it. --exl3-no-prefill-warmup skips (diagnostics).
    let mut psc0 = model.prefill_scratch(psc_rows(chunk.max(1), tp_rank.is_some()))?;
    if crate::opts::var(crate::opt!("exl3-no-prefill-warmup")).is_err() {
        model.prefill_warmup(&mut psc0, 0, &crate::exl3_forward::FwdModel::prefill_warmup_widths(chunk.max(1)))?;
    }
    // WP02 --exit-on-fatal (default ON since v0.7.3, owner decision 2026-10-06; was opt-in):
    // a sticky CUDA error or a scheduler panic exits 70 so a supervisor (serve_ours.sh SUPERVISE=1)
    // restarts the server. --exit-on-fatal off keeps the old behaviour: the engine goes DEAD and
    // /health + every request answer 503 until restarted. Registered Bool: bare flag = on.
    let exit_on_fatal = crate::opts::var(crate::opt!("exit-on-fatal")).map_or(true, |v| v != "0");
    let inject_panic: Option<usize> = crate::opts::var(crate::opt!("inject-panic-steps")).ok().and_then(|v| v.parse().ok());
    if let Some(n) = inject_panic {
        crate::deprintln!("[exl3-serve] DIAGNOSTIC: --inject-panic-steps={n} — injected scheduler fault at decode step {n}");
    }
    let _ = crate::exl3_forward::a_flags(); // log any GB10_A_* diagnostic escape at boot
    // WP04: --graph-precapture on|off (default on) — capture every decode graph the served config
    // reaches (step, verify widths, draft passes / chain, both attention regimes, every lane slot)
    // BEFORE listening, against the scheduler's own scratch, so no request pays a lazy capture.
    // Capture-only (WP04-v2): no kernel runs, so every device byte stays as an `off` boot has it.
    let precapture = match arg(args, "--graph-precapture") {
        None | Some("on") => true,
        Some("off") => false,
        Some(other) => anyhow::bail!("--graph-precapture must be on or off (got '{other}')"),
    };
    if precapture {
        model.precapture_decode_graphs(&mut sc, width, mtp_k, draft_conf > 0.0)?;
    } else {
        crate::dprintln!("[exl3-serve] graph precapture OFF (--graph-precapture off): decode graphs capture lazily");
    }
    // S-B9-CF-PACK4 `--lane-pack on`: the packed verify's own 16-row scratch and its graph family
    // (every exact width R = 4..16, each reachable regime), captured NOW — the served arm never
    // captures a packed graph while serving. With the flag off nothing here runs: no scratch,
    // no graph, and every device byte is master's (the host still prints one boot line and renders
    // the lane-pack /metrics series, all zero).
    // F1 (owner 2026-10-10): `--lane-pack on` never refuses a boot. Every reason it cannot pack on
    // this boot has been collected into `pack_blockers` (the posture fallbacks, `--max-batch 1`,
    // `--graph-precapture off`) plus `pack_supported`'s kernel/model postures here. A blocked boot
    // flips the flag OFF before anything packed runs — no packed scratch, no packed graph, the
    // served path is master's bit for bit — prints ONE plain line, and records the decision for
    // /metrics and the TP boot agree.
    if lane_pack.on() {
        if width < 2 {
            pack_blockers.push(format!("--max-batch {width}: a pack steps two requests per round (needs --max-batch >= 2)"));
        }
        if !precapture {
            pack_blockers.push("--graph-precapture off: the packed graphs are captured at boot, never while serving".to_string());
        }
        if let Err(e) = model.pack_supported() {
            pack_blockers.push(e.to_string());
        }
        if !pack_blockers.is_empty() {
            let why = pack_blockers.join("; ");
            crate::drprintln!("[exl3-serve] lane packing (--lane-pack, on by default) is NOT ACTIVE on this boot: {why}; requests are served without packed rounds (--lane-pack off silences this line)");
            crate::lane_pack::record_boot_pack(Some(why.clone()));
            lane_pack = crate::lane_pack::LanePack::Off;
        }
    }
    // F10: building the packed scratch and its graph family is the only allocation the default adds to a boot. It
    // must never abort one: any failure here (device memory, a capture) is a FALLBACK reason, the half-built
    // scratch is dropped, and the decision is recorded AFTER the build (the TP boot agree hashes it).
    let (scp, pack_ready, pack_sampled) = if lane_pack.on() {
        // Do sampled lanes pack on this boot? TP=1: yes (PACK4S). Under TP: `--lane-pack-sampled-tp`
        // (default on; off = a sampled lane runs its own lone round, counted `skip sampled`).
        let sampled_ok = tp_rank.is_none() || pack_sampled_under_tp();
        let fams = crate::lane_pack::fams(spec_ratio, sampled_ok);
        let free = || cudarc::driver::result::mem_get_info().map(|x| x.0).unwrap_or(0) as f64;
        let f0 = free();
        let built = crate::lane_pack::build_or_fall_back(|| {
            let mut scp = model.pack_scratch()?;
            model.dev().synchronize()?;
            let f1 = free();
            // S-B9-CF-PACK4S: under --spec-sampling ratio (the default) a sampled lane's verify rows
            // are real-q rows, so the boot also holds the rq family (the same widths and regimes);
            // under the match rule the greedy family serves sampled rows at TP=1 (a family of its own
            // under TP: a different verify tail).
            let tcap = std::time::Instant::now();
            let (n, graph_mb, note) = model.pack_precapture(&mut scp, &fams)?;
            let cap_s = tcap.elapsed().as_secs_f64();
            let fam_ready = |sparse: bool| fams.iter().all(|&f| model.pack_family_ready(&scp, sparse, f));
            let ready = [fam_ready(false), fam_ready(true)];
            anyhow::ensure!(ready[0], "the dense packed graph family is incomplete after the boot capture");
            Ok((scp, ready, n, graph_mb, note, cap_s, f1))
        });
        match built {
            Err(why) => {
                crate::drprintln!("[exl3-serve] lane packing (--lane-pack, on by default) is NOT ACTIVE on this boot: {why}; requests are served without packed rounds (--lane-pack off silences this line)");
                crate::lane_pack::record_boot_pack(Some(why));
                lane_pack = crate::lane_pack::LanePack::Off;
                (None, [false, false], false)
            }
            Ok((scp, ready, n, graph_mb, note, cap_s, f1)) => {
                crate::lane_pack::record_boot_pack(Some(String::new()));
                if tp_rank.is_some() {
                    crate::drprintln!("[exl3-serve] lane-pack under TP: sampled requests {} (--lane-pack-sampled-tp {})",
                                     if sampled_ok { "JOIN packed rounds (the pack ran them at TP=2 on one pair, 2026-10-10; TP=4 was exercised by hand, no gate chain)" } else { "run alone; only greedy requests pack" },
                                     if sampled_ok { "on" } else { "off" });
                }
                let f2 = free();
                let fam_names: Vec<&str> = fams.iter().map(|f| f.label()).collect();
                crate::drprintln!("[exl3-serve] lane-pack ON (the default; --lane-pack off turns it off){}: two requests share ONE packed speculative verify per round \
(three busy = one pair + one rotating lone round); {}; penalized requests and requests with images do not; \
draft chain: {}; packed scratch {:.0} MB + {n} packed verify graphs captured in {cap_s:.2} s, {graph_mb:+.0} MB \
(exact widths R = {}..{}, families: {}; {note}), {:.2} GiB of device memory in all (the graph delta is cuMemGetInfo-noisy; \
the free memory that matters is the 'after the boot' figure of the PLE line below)",
                         if tp_rank.is_some() { format!(" at TP={tp_world} rank {}", tp_rank.unwrap_or(0)) } else { String::new() },
                         if !sampled_ok { "GREEDY requests pack; SAMPLED requests run their own lone round (the sampled tail of a packed verify is not gated under TP)".to_string() }
                         else if crate::wp24::dump_on() { "greedy and sampled requests pack, in any combination (--wp24-dump is set: sampled requests stay on the lone path)".to_string() }
                         else if spec_ratio { "greedy and SAMPLED requests pack, in any combination (sampled rows verify by the real-q ratio rule, per-row sampler blocks)".to_string() }
                         else { "greedy and SAMPLED requests pack, in any combination (sampled rows verify by the match rule, per-row sampler blocks)".to_string() },
                         if draft_conf > 0.0 { "dynamic draft stop" } else { "FIXED depth (--draft-confidence 0)" },
                         (f0 - f1) / 1e6, crate::lane_pack::PACK_WIDTHS.start(), crate::lane_pack::PACK_WIDTHS.end(),
                         fam_names.join(" + "), (f0 - f2) / (1u64 << 30) as f64);
                // S-B9-CF-PACK5: the packed graph set, named on EVERY rank (head and node print this line);
                // the TP boot agree compares its count and digest (a rank holding a different set refuses
                // to boot instead of letting `pack_plan` decide rank-locally)
                let (gn, gh, gdesc) = model.pack_graph_set(&scp, &fams);
                crate::drprintln!("[exl3-serve] lane-pack packed graph set{}: {gn} graphs [{gdesc}] digest {gh:08x}; dense ready {}, sparse ready {}",
                         if tp_rank.is_some() { format!(" (TP={tp_world} rank {})", tp_rank.unwrap_or(0)) } else { String::new() },
                         ready[0], ready[1]);
                (Some(scp), ready, sampled_ok)
            }
        }
    } else {
        (None, [false, false], false)
    };
    // PFX1 (e): after the boot captures — the cuBLASLt plans of every prefill tail length are
    // built in the background (host-only work; requests are served meanwhile, lazily as before).
    let _ = model.lt_prewarm_start();
    // WP16: intermediate recurrent checkpoints — memory accounting at boot (allocated lazily).
    let wp16 = {
        let lay = model.ckpt_layout();
        let per = lay.bytes() as u64;
        let cap_bytes = (wp16_gb * (1u64 << 30) as f64) as u64;
        let cap = if per > 0 { (cap_bytes / per) as usize } else { 0 };
        if !prefix_on {
            crate::dprintln!("[exl3-serve] WP16 prefix checkpoints OFF (--prefix-cache off)");
            None
        } else if crate::exl3_forward::wp16_off() {
            crate::dprintln!("[exl3-serve] WP16 prefix checkpoints OFF (--wp16-off=1 — diagnostic escape: the prompt-end \
                      snapshot only, today's behaviour)");
            None
        } else if cap == 0 {
            crate::dprintln!("[exl3-serve] WP16 prefix checkpoints OFF (--prefix-ckpt-mem-gb {wp16_gb} holds no {:.1} MB checkpoint)",
                     per as f64 / 1e6);
            None
        } else {
            crate::dprintln!("[exl3-serve] WP16 prefix checkpoints ON: {}; cap {wp16_gb} GiB (--prefix-ckpt-mem-gb; default 4 is a \
                      placeholder — the owner sets the final default) = {cap} checkpoint(s) = {:.2} GiB at most, \
                      allocated lazily, LRU across slots, kept across a conversation's cache-hit turns. Taken at chunk \
                      ends of aligned prefills (grid from 0 or from an aligned checkpoint; never after a prompt-end \
                      prefix hit): end % {} == 0, plus every C={chunk} boundary in a prefill's last 2 chunks; keyed \
                      by C. Resumed on a prompt-end miss from the deepest checkpoint <= min(LCP, plen-1) \
                      (a 128K document holds ~18 = ~{:.1} GiB)",
                     lay.describe(), (cap as u64 * per) as f64 / (1u64 << 30) as f64,
                     crate::exl3_forward::WP16_STRIDE, 18.0 * per as f64 / (1u64 << 30) as f64);
            Some(crate::exl3_forward::CkptStore::new(width, cap))
        }
    };
    // C1: tail (message-boundary) checkpoints — opt-in, output-changing (registry prefix.tail_ckpt, N).
    let tail_ckpt_boot = wp16.is_some() && crate::exl3_forward::tail_ckpt_on();
    if tail_ckpt_boot {
        crate::dprintln!("[exl3-serve] C1 tail checkpoints ON (prefix.tail_ckpt = {} / --prefix-tail-ckpt=1|2: {}; OUTPUT-CHANGING, \
                  owner consent): every prefill also checkpoints at its message boundary (the prompt without its \
                  generation prompt) minus {} tokens when that saves >= {} tokens over the C grid point below it; runs \
                  are realigned to the C grid; the next chat turn resumes there",
                 if crate::exl3_forward::tail_ckpt_extra_chunk() { 2 } else { 1 },
                 if crate::exl3_forward::tail_ckpt_extra_chunk() {
                     "also splits costing one extra chunk when they promise >= 1024 tokens"
                 } else { "only splits absorbed into an existing chunk boundary" },
                 crate::exl3_forward::TAIL_BACKOFF, crate::exl3_forward::TAIL_MIN_GAIN);
    } else if crate::exl3_forward::tail_ckpt_on() {
        crate::dprintln!("[exl3-serve] C1 tail checkpoints requested but WP16 checkpoints are off — ignored");
    }
    let wp16_xcheck = wp16.is_some() && crate::exl3_forward::wp16_xcheck_on();
    if wp16_xcheck {
        crate::dprintln!("[exl3-serve] WP16 XCHECK ON (--wp16-xcheck=1, diagnostic): every checkpoint resume also prefills \
                  fresh from 0 into a scratch slot ({}) and bit-compares GDN S / conv / PLE ring + hist / head-fill \
                  carry and the first-token logits ([wp16-xcheck] lines)",
                 if width > 1 { "a free lane's slot" } else { "--max-batch 1: the resumed slot itself, then the resumed state is restored" });
    }
    // VIS-2: the vision tower (TP=1; TP > 1 serves images from VIS-4). Loaded when config.json declares
    // a vision_config and the text config an interleaved mrope; a load failure serves text-only with a
    // visible notice (image requests then answer 400).
    // VIS-4: the head (TP=1, or rank 0) runs the tower; a node only needs the mrope section (it splices
    // the rows the head ships).
    let (vision_tower, vision_gpu, mrope_section) = if tp_rank.map_or(true, |r| r == 0) {
        // PR #16 drill flag: --vision-fail-gemm (one-shot GEMM fault; see vision_gemm_drill).
        vision_boot(model_dir, vision_fail_gemm_armed())
    } else {
        let sec = mrope_section_of(model_dir);
        crate::dprintln!("[exl3-serve] vision (node): mrope section {sec:?}; image rows arrive from the head");
        (None, None, sec)
    };
    // CF-P1d: every lane's KV, the scratch, graphs and the vision tower now exist — `--ple-ram auto` decides RAM vs
    // SSD for the n-gram table on the memory that is actually left (each rank for itself; identical output either way).
    model.ple_promote_auto()?;
    let spec_policy = SpecLanes::parse(crate::opt_or_arg(args, "--spec-lanes-max").as_deref())?;
    // CF-FCFS: --lane-order reaches every rank through the SPMD option registry (Scope::Spmd), so
    // the head's value is installed on the nodes before the scheduler is built. The `arg()` read is
    // the TP=1 path's own value; under TP both ranks see the head's shipped value (identical).
    let lane_order = LaneOrder::parse(crate::opt_or_arg(args, "--lane-order").as_deref()
                                      .map(str::trim)
                                      .filter(|v| !v.is_empty()))?;
    let lane_quantum = match crate::opt_or_arg(args, "--lane-quantum").as_deref() {
        None => 256,
        Some(v) if v.trim().is_empty() => 256,
        Some(v) => v.trim().parse::<usize>().map_err(|e| anyhow::anyhow!("--lane-quantum: {e}"))?,
    };
    anyhow::ensure!(lane_quantum > 0, "--lane-quantum must be > 0");
    anyhow::ensure!(!(lane_pack.on() && lane_order == LaneOrder::Fcfs),
                    "--lane-pack on needs --lane-order rr: fcfs steps ONE lane per scheduler step, a pack steps two");
    // --prefill-interleave N (public PR #23; default 0 = off = no code path change). Every TP rank
    // parses the same argv and resolves 0, so the node needs nothing and the lockstep is untouched.
    let (interleave, il_line) = resolve_prefill_interleave(
        // PERFFIX1: from the registry (SPMD: a node gets the head's value; its own argv never carries the flag)
        // owner 2026-10-24: default 1 — an unset flag takes the REGISTRY default (never a hard-coded 0 that would drift)
        parse_prefill_interleave(Some(crate::opts::var(crate::opt!("prefill-interleave")).ok()
            .unwrap_or_else(|| crate::opt!("prefill-interleave").def().def.to_string()).as_str()))?, tp_rank.is_some(), chunk);
    if let Some(line) = il_line { crate::drprintln!("{line}"); }
    if width > 1 && spec_live {
        crate::dprintln!("[exl3-serve] load-adaptive batching (--spec-lanes-max): {} (--max-batch {width})", match spec_policy {
            SpecLanes::Auto => "auto — each round picks serial speculation or ONE shared plain step by estimated aggregate tokens/s".to_string(),
            SpecLanes::Over(k) => format!("one shared plain step with more than {k} busy lanes, serial speculation otherwise"),
            SpecLanes::Never => "off — always serial speculative rounds".to_string(),
        });
    }
    // S-B9-REL-MTPREPROBE: --mtp-reprobe=off restores the previous one-way behaviour;
    // --mtp-reprobe-drill shrinks the schedule for identity/stress runs.
    let reprobe_on = crate::opts::var(crate::opt!("mtp-reprobe")).map_or(true, |v| v != "off");
    let reprobe_cfg = if crate::opts::var(crate::opt!("mtp-reprobe-drill")).is_ok() {
        crate::mtp_reprobe::ReProbeCfg::drill()
    } else {
        crate::mtp_reprobe::ReProbeCfg::default()
    };
    // LANERES1: --lane-reserve R --lane-reserve-patience P --lane-reserve-swap S. LANERES3:
    // the policy is now head-authoritative at every topology — under TP the head runs
    // `lane_reserve::plan` in its admit sweep and ships each admit's scoping on the wire
    // (`WireRequest::reserve`), so the node restricts its lane pick identically and both
    // ranks hold the same slot tables. Default R=0 = today's admission, byte-for-byte.
    let lr_r = crate::opts::ranged(crate::opt!("lane-reserve")).map_err(|e| anyhow::anyhow!("{e}"))? as u64;
    let lr_p = crate::opts::ranged(crate::opt!("lane-reserve-patience")).map_err(|e| anyhow::anyhow!("{e}"))? as u64;
    let lr_s = crate::opts::ranged(crate::opt!("lane-reserve-swap")).map_err(|e| anyhow::anyhow!("{e}"))? as u64;
    let reserve_cfg = if lr_r > 0 {
        Some(crate::lane_reserve::Cfg::on(lr_r, lr_p, lr_s))
    } else {
        None
    };
    // LANERES2: --kv-spill <auto|off|PATH> --kv-spill-gb N. An SSD spill tier for evicted lane
    // snapshots: a hard eviction is captured to NVMe (async writer thread) and restored on the
    // conversation's next arrival when the read beats the re-prefill (kv_spill::Policy::lookup).
    // Default OFF since 2026-10-23 (owner: a snapshot copies the whole lane, so it cost more than it saved on typical
    // prompts; ISSUES 123); auto = /tmp/gb10_kv_spill; --kv-spill-gb 0 or "off" = no tier.
    // LANERES3: active at TP>=2 as well — every rank runs the same capture/restore hooks in
    // lockstep, writes ITS rank's shard to ITS OWN local SSD (the file name carries the rank;
    // snapshot bytes never cross the wire), and the outcome is agreed across ranks with one
    // SpillFlag rendezvous, so the replicated policy — and with it every later restore
    // decision — can never diverge on a disk fault (a key any rank cannot back is dropped
    // everywhere and counted).
    // owner 2026-10-23: default OFF (the registry default; read from the registry so the two cannot drift apart)
    let ks_mode = crate::opts::var(crate::opt!("kv-spill")).map(|v| v.to_string())
        .unwrap_or_else(|_| crate::opt!("kv-spill").def().def.to_string());
    let ks_gb = crate::opts::ranged(crate::opt!("kv-spill-gb")).map_err(|e| anyhow::anyhow!("{e}"))? as u64;
    let (spill_on, spill_dir) = if ks_mode == "off" || ks_gb == 0 {
        (false, String::new())
    } else {
        let d = if ks_mode == "auto" { "/tmp/gb10_kv_spill".to_string() } else { ks_mode.clone() };
        (true, d)
    };
    let (spill_store, spill_pol) = if spill_on {
        let st = crate::kv_spill::Store::spawn_with(std::path::PathBuf::from(&spill_dir),
            std::cmp::min(std::cmp::max((ks_gb << 30) / 4, 4 << 30), 6 << 30))
            .map_err(|e| anyhow::anyhow!("kv-spill: cannot create {spill_dir}: {e}"))?;
        (Some(st), crate::kv_spill::Policy::new(ks_gb << 30))
    } else {
        (None, crate::kv_spill::Policy::new(0))
    };
    crate::rprintln!("{}", if spill_on {
        let tp_note = if tp_rank.is_some() {
            "; TP: per-rank local shards, capture/restore agreed across ranks in-round"
        } else { "" };
        format!("[exl3-serve] kv spill: on, cap {ks_gb} GiB, dir {spill_dir} (rank {}; text-only snapshots; restores when the NVMe read beats the re-prefill it replaces){tp_note}", tp_rank.unwrap_or(0))
    } else {
        "[exl3-serve] kv spill: off (the default; --kv-spill auto or a directory turns it on)".to_string()
    });
    let sched = Exl3Scheduler {
        model: model.clone(), sc, width,
        lanes: (0..width).map(|_| None).collect(), eos: eos.clone(), chunk, interleave, psc: Some(psc0),
        df2, df2_k,
        mtp_k, spec_policy, lane_order, lane_quantum, admit_ctr: 0, fcfs_rots: 0,
        lane_pack, scp, pack_ready, pack_sampled, pack_rot: 0, pack_logs: 0, in_prefill: false, pack_stats: PackStats::default(),
        reprobe_on, reprobe_cfg,
        reserve: reserve_cfg, reserve_st: crate::lane_reserve::State::new(), rel: (0..width).map(|_| None).collect(),
        spill: spill_store, spill_pol, spill_rank: tp_rank.map(|r| r as usize).unwrap_or(0),
        shared_scale: 1.0, shared_last: false, shared_streak: 0, shared_probe_ctr: 0, spec_win_tok: [0.0; 2], spec_win_ms: [0.0; 2], spec_win_n: [0; 2], spec_win_acc: [0; 2], spec_arm_rate: [0.0; 2], spec_arm_n: [0; 2], spec_arm_fold: [0; 2], spec_probe: None, spec_probe_ctr: 0, spec_est_pick: None, spec_margin: 0.0, spec_margin_n: 0, policy_logs: 0, ctrl_round: 0, ema_plain: 0.0, draft_conf, cal: DraftCal::new(),
        prefix_on, cache: (0..width).map(|_| None).collect(),
        wp16, wp16_xcheck,
        loop_cfg, pen_range, spec_ratio, draft_temp, gate: None, tok: tok.clone(),
        tp: tp_rank.map(|r| TpSync::new(r, tp_world as i32)),
        exit_on_fatal, fatal: None, inject_panic, steps: 0, last_resume: (0, 0), mrope_section, vis_seam: None, kv_pf: None,
    };
    if width > 1 && spec_live {
        crate::dprintln!("[exl3-serve] serial lane order (--lane-order): {}{}", lane_order.name(),
                 if lane_order == LaneOrder::Fcfs {
                     format!(" — run to completion, one lane at a time, FCFS by admission order, \
                              quantum {} generated tokens (--lane-quantum)", lane_quantum)
                 } else {
                     " — every busy lane one round per scheduler step, in slot order (unchanged)".to_string()
                 });
        // after the lane-order line: `speculation_boot_lines_follow_the_switch` pins that line directly behind its guard
        crate::rprintln!("[exl3-serve] packed speculative verify (--lane-pack): {}", lane_pack.name());
    }
    // LANERES1: one line for the lane-reservation posture — active at every topology since
    // LANERES3 (head-authoritative under TP), or off.
    match reserve_cfg {
        Some(c) => crate::rprintln!("[exl3-serve] lane reservation: R {} s, patience {} s, swap {} s (--lane-reserve: keep a conversation's lane and its prefix cache; andrei.gnezdilov's design{})",
            c.reserve.as_secs(), c.patience.as_secs(), c.swap.as_secs(),
            if tp_rank.is_some() { "; TP: head-authoritative, the node applies the shipped admit scoping" } else { "" }),
        None => crate::rprintln!("[exl3-serve] lane reservation: off (--lane-reserve 0: today's admission, byte-for-byte)"),
    }
    println!("[exl3-serve] liveness: sticky CUDA error / scheduler thread crash -> {}",
             if exit_on_fatal { "exit 70 (--exit-on-fatal, default)" } else { "engine DEAD, /health 503 (--exit-on-fatal off)" });
    crate::dprintln!("[exl3-serve] prefill chunk width = {chunk} tokens/sweep (wide-M)");
    if let Some(line) = mtp_banner(model.mtp.is_some(), crate::exl3_forward::mtp_off_by_opt(), mtp_k, draft_conf) {
        crate::dprintln!("{line}");
    }
    // WP23: depth 6/7 posture (both parts default on iff k > 5; --exl3-mtp-k=5 = the K5 build).
    if spec_live && draft_conf > 0.0 {
        crate::dprintln!("[exl3-serve] WP23 DDS: depth ceiling k={mtp_k} (max {}), depth-keyed bins {}, cost guard {} \
                  (target = max({draft_conf}, clamp(c_hat*tok/ms, 0.15, 0.8)) after 16 rounds, c_hat seeded 6.0 ms/draft; \
                  --exl3-dds-depth-bins / --exl3-dds-guard = 0|1 force)",
                 crate::exl3_forward::MTP_MAX_K,
                 if wp23_on(crate::opt!("exl3-dds-depth-bins"), mtp_k) { "ON" } else { "off" },
                 if wp23_guard_on(mtp_k) { "ON" } else { "off" });
    }

    let (stx, srx) = tokio::sync::mpsc::unbounded_channel::<crate::batch::BatchRequest>();
    // v0.7.3 (#8.3): --model-name (alias --served-model-name) wins verbatim; the default
    // is the model card's `base_model:` line, else the directory name — the same
    // resolution order as the NVFP4 server (server::resolve_model_name).
    let model_name = crate::server::resolve_model_name(
        arg(args, "--model-name"), arg(args, "--served-model-name"), model_dir);
    crate::metrics::set_max_batch(width);
    // Public issue #24: the KV occupancy gauges — `width` lanes of the SERVED window (after the
    // clamp above). The model load registered the byte counts.
    crate::metrics::kv_pool_init(width, max_seq_len);
    // v0.7.3 ("tel"): the sink lives ONLY where the HTTP API lives — TP=1, or TP rank 0
    // (the head). A node never builds one: it has no HTTP hooks to fill the ring and no
    // sender to drain it, so under TP only the head emits. `--otel-model-id` /
    // `--otel-topology` override the auto values inside OtelSink::new.
    let otel = match (&otel_cfg, tp_rank) {
        (Some(cfg), None) | (Some(cfg), Some(0)) => Some(crate::otel::OtelSink::new(
            cfg.clone(), &model_name,
            // tp_rank (not tp — that was moved into the load) still tells single vs TP.
            &crate::otel::topology_from_world(if tp_rank.is_some() { Some(tp_world) } else { None }))),
        _ => None,
    };
    // FLASHNEXT-DF2-B (ISSUES work[113]): the pack's generation_config.json decides the
    // unsent-field sampler defaults (the Qwen3.8 card values only when the pack has none; a
    // Qwen3.8 pack's file holds exactly the card's thinking triple, so this is identity there).
    let (sampling_defaults, sd_src) = crate::server::SamplingDefaults::from_pack(
        model_dir, crate::server::SamplingDefaults::QWEN38_CARD);
    println!("[exl3-serve] {}", sampling_defaults.describe(sd_src));
    let state = AppState {
        sampling_defaults,
        scheduler: stx,
        tokenizer: tok,
        model_name: model_name.clone(),
        // No output cap in the request = generate until a stop token or the end of the context
        // window (the vLLM contract; was a silent 1024). --max-tokens N is the DEFAULT for a chat
        // request that omits max_tokens, not a cap: server.rs takes
        // `req.max_tokens.unwrap_or(default_max_tokens)` and clamps only to the context room.
        default_max_tokens: arg(args, "--max-tokens").and_then(|s| s.parse().ok()).unwrap_or(usize::MAX),
        default_rep_penalty,
        default_presence_penalty,
        default_frequency_penalty,
        reasoning_effort,
        thinking,
        output_prompts: 0,
        max_seq_len,
        decode_headroom: crate::batch::decode_headroom(false),
        // exl3 snapshots at the prompt end: no ckpt_at render needed — except C1 tail checkpoints,
        // which take one at the message boundary (ckpt_at) of every prompt
        prefix_cache: prefix_on && tail_ckpt_boot,
        // the EXL3 scheduler arms a schema after the think-close token (schema_feed)
        schema_defer: true,
        vision_tower,
        vision_gpu,
        vision_cpu: false,
        stop_ids: eos,
        otel,
    };
    let df2_engaged = sched.df2.is_some();
    Ok(ServeParts { sched, state, port, cpu_aff, exit_on_fatal, rx: srx,
                    boot_facts: (tp_world, max_seq_len, width, kv_name.to_string(), prefix_on,
                                 model.mtp.is_some(), crate::exl3_forward::mtp_off_by_opt().is_some(),
                                 df2_engaged) })
}

/// VIS-2: the text config's interleaved mrope_section (None = the model declares no interleaved mrope).
fn mrope_section_of(model_dir: &str) -> Option<[usize; 3]> {
    crate::vision_tower::vision_geometry(model_dir).ok()??;
    let raw = std::fs::read_to_string(std::path::Path::new(model_dir).join("config.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let rp = v.get("text_config").unwrap_or(&v).get("rope_parameters")?;
    if rp.get("mrope_interleaved").and_then(|x| x.as_bool()) != Some(true) { return None; }
    let a = rp.get("mrope_section")?.as_array()?;
    if a.len() != 3 { return None; }
    Some([a[0].as_u64()? as usize, a[1].as_u64()? as usize, a[2].as_u64()? as usize])
}

/// VIS-2: load the vision tower for the EXL3 server (CPU weights + the f32 GPU tower) and the text
/// config's interleaved mrope_section. Any failure = text-only with a printed reason.
#[allow(clippy::type_complexity)]
fn vision_boot(model_dir: &str, fail_first_gemm: bool) -> (Option<Arc<crate::vision_tower::VisualTower>>,
                                    Option<Arc<std::sync::Mutex<crate::vision_gpu::GpuVisualTower>>>,
                                    Option<[usize; 3]>) {
    // S-B9-REL-VISIONFIX2 (review M3): `--vision-cpu` is a registry flag, but only the NVFP4 server honours it.
    // This server always builds the GPU tower (the CPU tower runs only as the fallback after a GPU failure):
    // say so instead of ignoring the flag silently.
    if crate::opts::on(crate::opt!("vision-cpu")) {
        crate::rprintln!("[exl3-serve] vision: --vision-cpu is honoured by the NVFP4 server only; this server always builds the GPU tower \
(the CPU tower serves only after a GPU-tower failure)");
    }
    match crate::vision_tower::vision_geometry(model_dir) {
        Ok(Some(_)) => {}
        Ok(None) => return (None, None, None),
        Err(e) => { println!("[exl3-serve] vision: geometry probe failed ({e:#}) — text-only"); return (None, None, None); }
    }
    let Some(sec) = mrope_section_of(model_dir) else {
        crate::dprintln!("[exl3-serve] vision: the text config declares no interleaved mrope_section — text-only");
        return (None, None, None);
    };
    let t0 = std::time::Instant::now();
    let tower = match crate::vision_tower::VisualTower::load(model_dir) {
        Ok(t) => Arc::new(t),
        Err(e) => { println!("[exl3-serve] vision: tower load failed ({e:#}) — text-only"); return (None, None, None); }
    };
    let gpu = match cudarc::driver::CudaDevice::new(0).map_err(anyhow::Error::from)
        .and_then(|d| crate::vision_gpu::GpuVisualTower::new_with_drill(d, &tower, fail_first_gemm)) {
        Ok(g) => g,
        Err(e) => { crate::dprintln!("[exl3-serve] vision: GPU tower unavailable ({e:#}) — text-only"); return (None, None, None); }
    };
    let d = tower.dims;
    crate::dprintln!("[exl3-serve] vision ON: tower from {}: {} blocks x {} wide -> {} ({:.1} s); images resized to a longer side <= {} px \
              (--image-max-edge), mrope section {sec:?}; QSA indexer pooled keys follow the image positions (VIS-3)",
             tower.source, d.depth, d.hidden, d.out_hidden, t0.elapsed().as_secs_f64(),
             if tower.preproc.max_edge == 0 { "unlimited".to_string() } else { tower.preproc.max_edge.to_string() });
    (Some(tower), Some(Arc::new(std::sync::Mutex::new(gpu))), Some(sec))
}

/// TP-D: the host-thread mask at TP — the big cores minus the launch core (9, the pinned scheduler
/// thread) and the RDMA proxy core (19). TP=1: the resolved mask unchanged.
fn tp_worker_mask(m: &Option<Vec<usize>>, tp: bool, world: i32) -> Option<Vec<usize>> {
    let m = m.clone()?;
    if !tp { return Some(m); }
    // TP-F: the dual-rail prefill transport pins its second proxy on TP_AUX_PROXY_CORE (18).
    let aux = if crate::exl3_forward::xtp::prefill_xport_mode() == 2 { crate::exl3_forward::xtp::TP_AUX_PROXY_CORE as usize } else { usize::MAX };
    // TP-4F2: `--tp-reduce single` pins a third proxy (the single-stage reduce ctx) on TP_SINGLE_PROXY_CORE (17).
    let single = if crate::exl3_forward::xtp::tp_reduce_single_for(world) { crate::exl3_forward::xtp::TP_SINGLE_PROXY_CORE as usize } else { usize::MAX };
    let f: Vec<usize> = m.into_iter().filter(|&c| c != 9 && c != 19 && c != aux && c != single).collect();
    if f.is_empty() { None } else { Some(f) }
}

/// TP-D: threads the pinned (single-core) TP scheduler thread spawns — the prefill PLE row workers —
/// would inherit its one-core mask and serialize on core 9; they re-pin to this mask instead.
fn set_worker_mask(m: Option<Vec<usize>>) {
    if let Some(m) = m {
        crate::dprintln!("[exl3-serve] TP: scheduler pinned to core 9; spawned host workers use cpus {}",
                 crate::cpu_affinity::format_cpu_list(&m));
        crate::cpu_affinity::set_worker_mask(m);
    }
}

/// Spawn the scheduler thread and serve HTTP on this thread (never returns on success).
/// `tp_ctl` Some = the TP=2 head: the scheduler runs `run_tp_head` over the node control streams,
/// on a thread pinned to core 9 (the TP launch-thread rule), and the HTTP/worker threads keep the
/// big-core mask minus the launch (9) and proxy (19) cores.
fn serve_http(parts: ServeParts, tp_ctl: Option<Vec<std::net::TcpStream>>) -> Result<()> {
    let ServeParts { sched, state, port, cpu_aff, exit_on_fatal, rx: srx, boot_facts } = parts;
    let model_name = state.model_name.clone();
    // v0.7.3 ("tel"): the ONE sender task rides THIS runtime (serve_http is head/TP=1 only).
    let otel_sink = state.otel.clone();
    // HOST / RO-7: pin THIS thread (it runs the HTTP runtime below) to the big cores BEFORE the
    // scheduler thread is spawned — a new thread inherits its creator's mask (Linux), and so do
    // the threads either of them spawns later (prefill PLE workers, tokio's blocking pool).
    crate::dprintln!("[exl3-serve] {}", cpu_aff.1);
    let mask = tp_worker_mask(&cpu_aff.0, tp_ctl.is_some(), tp_ctl.as_ref().map_or(1, |c| c.len() as i32 + 1));
    if let Some(cpus) = mask.as_ref() {
        if let Err(e) = crate::cpu_affinity::pin_current_thread(cpus) {
            println!("[exl3-serve] cpu affinity: pinning FAILED ({e:#}) — threads left to the OS scheduler");
        }
    }
    match tp_ctl {
        None => {
            std::thread::spawn(move || {
                let _guard = FatalGuard { exit_on_fatal };
                sched.run(srx)
            });
        }
        Some(streams) => {
            std::thread::spawn(move || {
                // any exit of the TP scheduler thread ends the process (a TP head cannot serve on
                // without its lockstep): exit 70, the NVFP4 TP head's contract
                let _guard = FatalGuard { exit_on_fatal: true };
                if !crate::net::pin_thread(9) {
                    eprintln!("\n*** FATAL: TP head scheduler failed to pin to core 9 — TP refuses to run unpinned. Exiting. ***\n");
                    crate::logq::flush_and_exit(std::time::Duration::from_millis(300), 70); // H7
                }
                set_worker_mask(mask);
                let r = sched.run_tp_head(srx, streams);
                match r {
                    Ok(()) => eprintln!("[exl3-serve] TP head: request channel closed — session over"),
                    Err(e) => eprintln!("\n*** FATAL: the TP scheduler failed: {e:#}. Exiting (70). ***\n"),
                }
                std::thread::sleep(std::time::Duration::from_millis(300)); // let the error events flush
                crate::logq::flush_and_exit(std::time::Duration::from_millis(300), 70); // H7 + LR-1
            });
        }
    }
    let rt = tokio::runtime::Builder::new_current_thread().enable_all()
        .build().context("tokio runtime")?;
    rt.block_on(async move {
        // v0.7.3 ("tel"): run_sender is fully async (timed connect/write/read) — a slow or
        // absent receiver delays only itself; decode and SSE serving continue (the design
        // contract: ring, drop-on-full, one task off the compute path, zero cost when off).
        if let Some(sink) = otel_sink { tokio::spawn(crate::otel::run_sender(sink)); }
        let app = create_router(state);
        let listener = crate::server::http_listen(port).await;
        let url_l = listener.local_addr().map_or_else(|_| format!("port {port}"), |a| a.to_string());
        if crate::boot_report::debug() {
            println!("[exl3-serve] listening on {url_l} (model id: {model_name})");
        }
        {
            let (tp_w, ctx, batch, kv_name, prefix_on, mtp_head, mtp_off, df2_on) = &boot_facts;
            crate::boot_report::serving(&crate::boot_report::ServingLine {
                family: "exl3", model: &model_name, tp: *tp_w, ctx: *ctx, batch: *batch,
                kv: kv_name, prefix: if *prefix_on { "on" } else { "off" },
                mtp: if !*mtp_head { "none" } else if *mtp_off { "off" } else { "auto" },
                // ISSUES[94]: the DF2 chain is the engaged speculator — the SERVING line says
                // so (dflash2), the pack MTP otherwise.
                spec: if *df2_on { "dflash2" } else if *mtp_head && !*mtp_off { "mtp" } else { "plain" },
                port,
                url: if url_l.starts_with("port ") { format!("http://0.0.0.0:{port}") } else { format!("http://{url_l}") },
            });
        }
        // WP02: TCP_NODELAY — an SSE token event is a few dozen bytes; Nagle held them for the ACK
        axum::serve(listener, app).tcp_nodelay(true).await.unwrap();
    });
    Ok(())
}

// ---- WP24 gate, stage B (served level): see src/wp24.rs for stage A and the statistics ----

/// The gate's three arms: ratio MTP (under test), match MTP (the control: today's rule, known
/// exact), plain sampled decode with MTP off (the reference law).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GateArm {
    Plain = 0,
    Ratio = 1,
    Match = 2,
}

struct GateArmOut {
    seqs: Vec<Vec<u32>>,
    stats: GateStats,
    secs: f64,
}

const WP24_GATE_PROMPT: &str = "Write a short story about a lighthouse keeper who finds a message in a bottle.\n\nThe lighthouse keeper";

/// One arm of one condition: `trials` seeded requests of `prompt` through the REAL scheduler
/// (admit -> prefix-cache restore -> seam step -> prime -> step() rounds), `positions` tokens
/// each (ignore_eos, so every trial yields exactly that many), in waves of `width` lanes. The
/// plain arm clears each lane's MTP state after admit (the per-request §6 fallback's exact
/// hand-off: the lane then decodes through step_plain). Trial seeds: distinct per (condition,
/// arm, trial) — independent RNG streams everywhere (AGENTS §3).
#[allow(clippy::too_many_arguments)]
fn wp24_gate_arm(sched: &mut Exl3Scheduler, arm: GateArm, prompt: &[u32], t: f32, top_p: f32,
                 top_k: usize, trials: usize, positions: usize, base: u64) -> Result<GateArmOut> {
    sched.spec_ratio = arm == GateArm::Ratio;
    sched.cal = DraftCal::new(); // each arm learns its own DDS map (never another arm's labels)
    sched.gate = Some(GateStats::default());
    let t0 = std::time::Instant::now();
    let mut seqs: Vec<Vec<u32>> = Vec::with_capacity(trials);
    let mut done = 0usize;
    while done < trials {
        let nb = (trials - done).min(sched.width);
        let mut rxs = Vec::with_capacity(nb);
        for j in 0..nb {
            let (tx, rx, bl) = crate::server::tok_channel(); // LR-5 (H6): drained counted below
            let seed = crate::wp24::mix64(base ^ ((arm as u64 + 1) << 56) ^ (done + j) as u64);
            sched.admit(BatchRequest {
                prompt: prompt.to_vec(),
                max_new: positions,
                temperature: t,
                top_p,
                top_k,
                rep_penalty: 1.0,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                min_p: 0.0,
                min_new: 0,
                ignore_eos: true,
                tx,
                seed: Some(seed),
                ckpt_at: None,
                domain: crate::batch::Domain::General,
                received_at: std::time::Instant::now(),
                image_embeds: None,
                image_spans: Vec::new(),
                schema: None,
            });
            if let Some(f) = sched.fatal.as_ref() { anyhow::bail!("WP24 gate: engine fatal at admit: {f}"); }
            rxs.push((rx, bl));
        }
        if arm == GateArm::Plain {
            for l in sched.lanes.iter_mut().flatten() { l.mtp = None; }
        }
        let mut steps = 0usize;
        while sched.lanes.iter().any(|l| l.is_some()) {
            sched.step()?;
            if let Some(f) = sched.fatal.as_ref() { anyhow::bail!("WP24 gate: engine fatal: {f}"); }
            steps += 1;
            anyhow::ensure!(steps <= 4 * positions + 64, "WP24 gate: lanes still live after {steps} steps");
        }
        for (mut rx, bl) in rxs {
            let (mut toks, mut reason) = (Vec::with_capacity(positions), None);
            while let Some(ev) = crate::server::try_recv_counted(&mut rx, &bl) { // LR-5 (H6)
                match ev {
                    TokEvent::Tok(x) => toks.push(x),
                    TokEvent::Finish { reason: r } => reason = Some(r),
                    // #8.2: admission telemetry — no action for a gate drain.
                    TokEvent::Admitted { .. } => {}
                    TokEvent::StopInfo { .. } => {}
                }
            }
            anyhow::ensure!(reason.as_deref() == Some("length") && toks.len() == positions,
                            "WP24 gate: a {arm:?} trial ended {reason:?} after {} tokens (want {positions}, \"length\")",
                            toks.len());
            seqs.push(toks);
        }
        done += nb;
    }
    let stats = sched.gate.take().unwrap_or_default();
    Ok(GateArmOut { seqs, stats, secs: t0.elapsed().as_secs_f64() })
}

/// `--probe-spec-sampling --model-dir <EXL3 pack>` stage B: the served-level distribution gate.
/// Per condition (T in {0.7, 1.0} x top_p in {0.8, 0.95}, top_k 20, >= 2 seeds): `--trials`
/// requests per arm of one raw prompt (`--prompt`, default a story opening), each generating
/// `--positions` tokens (default 6: x1 = the seam step, identical code in every arm — a built-in
/// control; x2.. = MTP rounds; the remaining-budget clamp lets round 1 draft up to positions-2).
/// Tests (reference-free two-sample chi-square, wp24::gof2 — |z| < 4 passes):
///   GATE    ratio-vs-plain per position and on the joint (x2, x3);
///   CONTROL match-vs-plain, the same tests (today's rule is exact: a failure = the harness is
///           broken, and the condition FAILS);
///   NEG     the ratio arm's first-round drafts d_0 vs plain x2 (d_0 ~ q' != p: must be REJECTED
///           in at least one condition, else the gate had no power on this prompt).
/// Success signals asserted, not just the absence of errors: every trial returns exactly
/// `positions` tokens with finish "length"; the ratio arm ran real-q rounds (>0) and the match
/// arm none; both MTP arms drafted. Output: one line per condition, then RESULT. `--quick` = the
/// owner's config (T 1.0, top_p 0.95) only, as the first rung of the ladder.
pub fn wp24_served_gate(args: &[String], model_dir: &str) -> Result<bool> {
    let num = |name: &str, d: usize| -> Result<usize> {
        match crate::opt_or_arg(args, name) {
            None => Ok(d),
            Some(v) => v.parse().map_err(|_| anyhow::anyhow!("{name} must be an integer")),
        }
    };
    let trials = num("--trials", 600)?.max(50);
    let positions = num("--positions", 6)?.clamp(3, 32);
    let width = num("--max-batch", 4)?.clamp(1, 16);
    let n_seeds = num("--seeds", 2)?.max(2);
    let top_k = num("--top-k", 20)?;
    let max_seq_len = num("--max-seq-len", 4096)?;
    let draft_conf: f64 = match arg(args, "--draft-confidence") {
        None => 0.4,
        Some(v) => v.parse().map_err(|_| anyhow::anyhow!("--draft-confidence must be a number"))?,
    };
    anyhow::ensure!((0.0..1.0).contains(&draft_conf), "--draft-confidence must be in [0, 1)");
    let (_, draft_temp) = parse_spec_sampling(args)?; // --draft-temperature applies to the ratio arm
    let text = arg(args, "--prompt").unwrap_or(WP24_GATE_PROMPT).to_string();
    // the served posture: prefix snapshots on (run() sets this for --prefix-cache on), KV format
    // from --kv-cache / [kv-cache] / the default, graphs precaptured like the server boot
    crate::opts::set(crate::opt!("exl3-prefix"), "1");
    if let Some(v) = arg(args, "--kv-cache") {
        anyhow::ensure!(matches!(v, "f32" | "f16" | "fp8" | "q8"), "--kv-cache must be f32, f16, fp8 or q8");
        crate::opts::set(crate::opt!("kv-cache"), v);
    }
    if let Some(off) = crate::exl3_forward::mtp_off_by_opt() {
        anyhow::bail!("WP24 gate: {} is set — the MTP arms need the head", off.flag());
    }
    let max_pos = max_seq_len + crate::batch::decode_headroom(false);
    crate::dprintln!("WP24 served gate: loading {model_dir} (width {width}, max_pos {max_pos}, kv-cache {})",
             crate::exl3_forward::kv_fmt_name(crate::exl3_forward::kv_fmt_from_opts()));
    let model = FwdModel::load(model_dir, width, max_pos)?;
    anyhow::ensure!(model.mtp.is_some(), "WP24 gate: the pack has no MTP draft head");
    let tok = QwenTokenizer::from_file(&format!("{}/tokenizer.json", model_dir.trim_end_matches('/')))?;
    let cfg_eos: u32 = {
        let cfg: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(format!("{}/config.json", model_dir.trim_end_matches('/')))?)?;
        cfg.get("eos_token_id").and_then(|e| e.as_u64()).unwrap_or(248046) as u32
    };
    let eos = tok.stop_token_ids(cfg_eos);
    let prompt = tok.encode(&text, false)?;
    let mtp_k = crate::exl3_forward::mtp_depth_default();
    anyhow::ensure!(prompt.len() >= 2 && prompt.len() + positions + mtp_k + 4 < model.max_pos(),
                    "WP24 gate: prompt of {} tokens does not fit the window", prompt.len());
    let mut sc = FwdModel::scratch(model.dev(), &model.cfg, width.max(mtp_k + 1), model.max_pos())?;
    let chunk = 2048usize;
    let psc0 = model.prefill_scratch(chunk)?;
    model.precapture_decode_graphs(&mut sc, width, mtp_k, draft_conf > 0.0)?;
    let tok = Arc::new(tok);
    // S-B9-REL-MTPREPROBE policy flags (bench/probe paths construct the same scheduler).
    let reprobe_on = crate::opts::var(crate::opt!("mtp-reprobe")).map_or(true, |v| v != "off");
    let reprobe_cfg = if crate::opts::var(crate::opt!("mtp-reprobe-drill")).is_ok() {
        crate::mtp_reprobe::ReProbeCfg::drill()
    } else {
        crate::mtp_reprobe::ReProbeCfg::default()
    };
    let mut sched = Exl3Scheduler {
        spill: None, spill_pol: crate::kv_spill::Policy::new(0), spill_rank: 0,
        model: model.clone(), sc, width,
        lanes: (0..width).map(|_| None).collect(), eos, chunk, interleave: 0, psc: Some(psc0),
        df2: None, df2_k: crate::dflash2::block() - 1, // run_spec probe path: no DF2 artifact
        mtp_k, spec_policy: SpecLanes::Never, lane_order: LaneOrder::Rr, lane_quantum: 256, admit_ctr: 0, fcfs_rots: 0,
        lane_pack: crate::lane_pack::LanePack::Off, scp: None, pack_ready: [false, false], pack_sampled: false, pack_rot: 0, pack_logs: 0, in_prefill: false, pack_stats: PackStats::default(),
        reprobe_on, reprobe_cfg, shared_scale: 1.0, shared_last: false, shared_streak: 0, shared_probe_ctr: 0, spec_win_tok: [0.0; 2], spec_win_ms: [0.0; 2], spec_win_n: [0; 2], spec_win_acc: [0; 2], spec_arm_rate: [0.0; 2], spec_arm_n: [0; 2], spec_arm_fold: [0; 2], spec_probe: None, spec_probe_ctr: 0, spec_est_pick: None, spec_margin: 0.0, spec_margin_n: 0, policy_logs: 0, ctrl_round: 0, ema_plain: 0.0, draft_conf, cal: DraftCal::new(),
        reserve: None, reserve_st: crate::lane_reserve::State::new(), rel: (0..width).map(|_| None).collect(),
        prefix_on: true, cache: (0..width).map(|_| None).collect(),
        wp16: None, wp16_xcheck: false,
        loop_cfg: None, pen_range: 1024, spec_ratio: false, draft_temp, gate: None, tok: tok.clone(), tp: None,
        exit_on_fatal: false, fatal: None, inject_panic: None, steps: 0, last_resume: (0, 0), mrope_section: None, vis_seam: None, kv_pf: None,
    };
    let quick = args.iter().any(|a| a == "--quick");
    wp24_gate_conditions(&mut sched, &prompt, &text, trials, positions, n_seeds, top_k, draft_conf, draft_temp, quick)
}

/// The WP24 served gate's condition loop on a constructed scheduler (TP-C: shared by the TP=1
/// gate above and the TP=2 spec program, where every rank runs it in lockstep).
#[allow(clippy::too_many_arguments)]
fn wp24_gate_conditions(sched: &mut Exl3Scheduler, prompt: &[u32], text: &str, trials: usize, positions: usize,
                        n_seeds: usize, top_k: usize, draft_conf: f64, draft_temp: Option<f32>, quick: bool) -> Result<bool> {
    let mtp_k = sched.mtp_k;
    sched.draft_conf = draft_conf;
    sched.draft_temp = draft_temp;
    crate::dprintln!("WP24 served gate: prompt {} tokens {:?}; {trials} trials x 3 arms (plain | ratio | match) per condition, \
              {positions} positions, top_k {top_k}, {n_seeds} seeds, DDS {draft_conf}, depth {mtp_k}, draft temperature {}",
             prompt.len(), text.chars().take(60).collect::<String>(),
             draft_temp.map_or("= T".to_string(), |x| x.to_string()));
    type Hist = std::collections::HashMap<u64, u64>;
    let hist = |seqs: &[Vec<u32>], j: usize| -> Hist {
        let mut h = Hist::new();
        for s in seqs { *h.entry(s[j] as u64).or_insert(0) += 1; }
        h
    };
    let pair = |seqs: &[Vec<u32>], j: usize| -> Hist {
        let mut h = Hist::new();
        for s in seqs { *h.entry(((s[j] as u64) << 32) | s[j + 1] as u64).or_insert(0) += 1; }
        h
    };
    let t_all = std::time::Instant::now();
    let (mut all_ok, mut neg_rejected, mut conds) = (true, false, 0usize);
    // --quick: the owner's daily config only (T 1.0, top_p 0.95) — the ladder's first rung; the
    // full gate is all four configs
    let configs: &[(f32, f32)] = if quick { &[(1.0, 0.95)] } else { &[(0.7, 0.8), (0.7, 0.95), (1.0, 0.8), (1.0, 0.95)] };
    let total = configs.len() * n_seeds;
    for &(t, top_p) in configs {
        for si in 0..n_seeds {
            let base = crate::wp24::mix64(0x5EED_2400_0000_0000u64
                ^ (((t.to_bits() as u64) << 32) | top_p.to_bits() as u64)
                ^ (si as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let plain = wp24_gate_arm(sched, GateArm::Plain, prompt, t, top_p, top_k, trials, positions, base)?;
            let ratio = wp24_gate_arm(sched, GateArm::Ratio, prompt, t, top_p, top_k, trials, positions, base)?;
            let matc = wp24_gate_arm(sched, GateArm::Match, prompt, t, top_p, top_k, trials, positions, base)?;
            let (mut zr, mut zm) = (Vec::with_capacity(positions + 1), Vec::with_capacity(positions + 1));
            for j in 0..positions {
                zr.push(crate::wp24::gof2_maps(&hist(&ratio.seqs, j), &hist(&plain.seqs, j)).1);
                zm.push(crate::wp24::gof2_maps(&hist(&matc.seqs, j), &hist(&plain.seqs, j)).1);
            }
            zr.push(crate::wp24::gof2_maps(&pair(&ratio.seqs, 1), &pair(&plain.seqs, 1)).1);
            zm.push(crate::wp24::gof2_maps(&pair(&matc.seqs, 1), &pair(&plain.seqs, 1)).1);
            let mut d0h = Hist::new();
            for &d in &ratio.stats.d0 { *d0h.entry(d as u64).or_insert(0) += 1; }
            let neg = crate::wp24::gof2_maps(&d0h, &hist(&plain.seqs, 1));
            let signals = ratio.stats.ratio_rounds > 0 && ratio.stats.drafted > 0
                && matc.stats.ratio_rounds == 0 && matc.stats.drafted > 0;
            let gate_ok = zr.iter().all(|z| z.abs() < 4.0);
            let ctrl_ok = zm.iter().all(|z| z.abs() < 4.0);
            let ok = gate_ok && ctrl_ok && signals;
            all_ok &= ok;
            neg_rejected |= neg.1 >= 4.0 && !ratio.stats.d0.is_empty();
            conds += 1;
            let fz = |v: &[f64]| v.iter().map(|z| format!("{z:+.2}")).collect::<Vec<_>>().join(" ");
            let arm = |a: &GateArmOut| {
                let s = &a.stats;
                format!("rounds {} ratio-rounds {} accept {:.1}% tok/round {:.2} ({:.0} s)", s.rounds, s.ratio_rounds,
                        100.0 * s.accepted as f64 / s.drafted.max(1) as f64,
                        (s.accepted + s.rounds) as f64 / s.rounds.max(1) as f64, a.secs)
            };
            crate::dprintln!("  [served] T={t} top_p={top_p} seed#{si}: n={trials}/arm | GATE ratio-vs-plain z[pos 1..{positions}, pair 2-3] = {} | \
                      CONTROL match-vs-plain z = {} | NEG d0-vs-plain-x2 z={:+.1} ({} d0) | ratio: {} | match: {} | plain {:.0} s{} -> {}",
                     fz(&zr), fz(&zm), neg.1, ratio.stats.d0.len(), arm(&ratio), arm(&matc), plain.secs,
                     if signals { "" } else { " | SUCCESS SIGNALS MISSING (ratio rounds / drafts)" },
                     if ok { "PASS" } else if !ctrl_ok { "FAIL (control failed: harness suspect)" } else { "FAIL" });
            if conds == 1 {
                let per = t_all.elapsed().as_secs_f64();
                crate::dprintln!("  [served] projected total ~{:.0} min ({total} conditions at {:.0} s each)", per * total as f64 / 60.0, per);
            }
        }
    }
    if !neg_rejected {
        crate::dprintln!("  [served] negative control never rejected: the gate had no power to see q' != p on this prompt");
        crate::dprintln!("RESULT: SERVED_GATE_NO_POWER");
        return Ok(false);
    }
    crate::dprintln!("RESULT: {} ({conds} conditions{}, {:.1} min)",
             if all_ok { "SERVED_DISTRIBUTION_OK (ratio-mode MTP emits the plain sampler's law at every tested position)" }
             else { "SERVED_DISTRIBUTION_MISMATCH" },
             if quick { ", --quick: T 1.0 / top_p 0.95 only — run without --quick for the full gate" } else { "" },
             t_all.elapsed().as_secs_f64() / 60.0);
    Ok(all_ok)
}

#[cfg(test)]
mod wp23_tests {
    use super::*;

    fn cal(depth: bool) -> DraftCal {
        DraftCal {
            bins: Default::default(),
            total: 0.0,
            dbins: if depth { Some(vec![Default::default(); WP23_DEPTH_BUCKETS]) } else { None },
        }
    }

    /// deterministic label stream: (pass index, score, accepted), deeper passes less likely
    fn stream(n: usize) -> Vec<(usize, f32, bool)> {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        (0..n).map(|_| {
            let i = (next() % 7) as usize;
            let score = 8.0 + (next() % 2400) as f32 / 100.0;
            let acc = ((next() % 100) as f32 / 100.0) < (score / 32.0 - 0.06 * i as f32);
            (i, score, acc)
        }).collect()
    }

    fn feed(c: &mut DraftCal, labels: &[(usize, f32, bool)]) {
        for (j, &(i, sc, a)) in labels.iter().enumerate() {
            c.add_label(i, sc, a);
            if j % 3 == 2 { c.decay_step(); }
        }
    }

    #[test]
    fn depth_bins_off_is_the_legacy_calibrator() {
        let mut c = cal(false);
        feed(&mut c, &stream(4000));
        for i in 0..8 {
            for q in 0..400 {
                let sc = 4.0 + q as f32 * 0.1;
                assert_eq!(c.estimate_at(i, sc).to_bits(), c.estimate(sc).to_bits(), "i {i} score {sc}");
            }
        }
    }

    #[test]
    fn depth_bins_keep_the_pooled_map_identical() {
        let (mut a, mut b) = (cal(false), cal(true));
        let l = stream(4000);
        feed(&mut a, &l);
        feed(&mut b, &l);
        assert_eq!(a.total.to_bits(), b.total.to_bits());
        assert_eq!(a.bins.len(), b.bins.len());
        for ((ka, va), (kb, vb)) in a.bins.iter().zip(b.bins.iter()) {
            assert_eq!(ka, kb);
            assert_eq!((va.0.to_bits(), va.1.to_bits()), (vb.0.to_bits(), vb.1.to_bits()));
        }
    }

    #[test]
    fn depth_estimate_uses_its_bucket_else_the_pool() {
        let mut c = cal(true);
        // pass 0: logit 20 always accepted; pass 5 (bucket 3): logit 20 accepted 1 in 4
        for _ in 0..40 { c.add_label(0, 20.5, true); }
        for j in 0..40 { c.add_label(5, 20.5, j % 4 == 0); }
        assert!(c.total >= DraftCal::BURN_IN);
        let pooled = c.estimate(20.5);
        assert!((pooled - 50.0 / 80.0).abs() < 1e-12);
        assert!((c.estimate_at(0, 20.5) - 1.0).abs() < 1e-12);
        assert!((c.estimate_at(6, 20.5) - 0.25).abs() < 1e-12); // pass 6 shares bucket 3
        // bucket 2 (passes 3, 4) has no labels: the pooled estimate
        assert_eq!(c.estimate_at(3, 20.5).to_bits(), pooled.to_bits());
        // below every populated bin of the bucket: the pooled fallback (its lowest-bin rule)
        assert_eq!(c.estimate_at(5, 12.0).to_bits(), c.estimate(12.0).to_bits());
        // burn-in: optimistic 1.0
        let mut d = cal(true);
        for _ in 0..10 { d.add_label(5, 20.5, false); }
        assert_eq!(d.estimate_at(5, 20.5), 1.0);
    }

    #[test]
    fn cost_fit_prior_and_slope() {
        assert_eq!(CostFit::default().slope(), 6.0);
        // constant drafts/round: no spread, the prior stays
        let mut g = CostFit::default();
        for _ in 0..200 { g.add(7.0, 56.0); }
        assert!((g.slope() - 6.0).abs() < 1e-9, "{}", g.slope());
        // spread drafts with a true 9 ms/draft slope: identified (shrunk slightly toward 6)
        let mut h = CostFit::default();
        for r in 0..200 { let x = (r % 7 + 1) as f64; h.add(x, 30.0 + 9.0 * x); }
        let s = h.slope();
        assert!(s > 8.7 && s < 9.0, "{s}");
        // never negative
        let mut n = CostFit::default();
        for r in 0..200 { let x = (r % 7 + 1) as f64; n.add(x, 90.0 - 30.0 * x); }
        assert!(n.slope() >= 0.0);
    }

    #[test]
    fn guard_target_rules() {
        // off, too early, or no EMAs: the base target
        assert_eq!(wp23_target(0.4, false, 100, 6.0, 5.6, 56.0), 0.4);
        assert_eq!(wp23_target(0.4, true, 15, 6.0, 5.6, 56.0), 0.4);
        assert_eq!(wp23_target(0.4, true, 100, 6.0, 0.0, 0.0), 0.4);
        // AnsiC-like: 6 ms/draft at 0.1 tok/ms -> 0.6
        assert!((wp23_target(0.4, true, 16, 6.0, 5.6, 56.0) - 0.6).abs() < 1e-12);
        // prose-like: 6 * 2.7 / 45 = 0.36 < base -> base (the guard only raises)
        assert_eq!(wp23_target(0.4, true, 100, 6.0, 2.7, 45.0), 0.4);
        // long context, expensive rows: clamped at 0.8
        assert_eq!(wp23_target(0.4, true, 100, 20.0, 5.0, 60.0), 0.8);
        // an operator base above the guard wins
        assert_eq!(wp23_target(0.9, true, 100, 20.0, 5.0, 60.0), 0.9);
    }

    #[test]
    fn depth_buckets_cover_the_window() {
        let b: Vec<usize> = (0..crate::exl3_forward::MTP_MAX_K).map(wp23_depth_bucket).collect();
        assert_eq!(b, vec![0, 1, 1, 2, 2, 3, 3]);
        assert!(b.iter().all(|&x| x < WP23_DEPTH_BUCKETS));
    }
}

// ---------------------------------------------------------------------------------------------
// TP-C: the spec program (`--probe-exl3-tpspec`) — the SERVED scheduler, offline (no HTTP), at
// TP=1 (single process) or as one rank of TP=2 (SPMD, every rank runs this function). Every
// request goes through Exl3Scheduler::admit (prefix cache -> chunked prefill -> seam step ->
// head prime) and step() (graphed DDS rounds / plain steps / control rounds), exactly the
// serve's code; at TP=2 each step is a lockstep point (TpSync). Gates:
//   G-T1-e  greedy: every spec arm (DDS at the served depth, fixed chains at each --spec-depths)
//           emits exactly the plain arm's tokens, per prompt and rep (and plain is rep-stable);
//   G-T1-g  acceptance: tok/round of the DDS arm per prompt class (compare TP=2 vs the TP=1 run
//           of this same program);
//   speed   DDS ms/round and tok/s, plain ms/token (bring-up readings, not perf claims);
//   G-T1-b  (graphed) with ident on: logits rows / residual / taps digests identical across ranks
//           at every lockstep point of rep 0;
//   G-T1-h  the WP24 served distribution gate (ratio + match vs plain, chi-square) on the same
//           scheduler (--spec-samp quick|full).
// ---------------------------------------------------------------------------------------------

/// Options of the spec program; ride TpConfig.exl3_mode as
/// "spec:gen=N:reps=N:depths=1,2:dds=0|1:width=N:prefix=0|1:samp=0|1|2:trials=N:ident=0|1".
#[derive(Clone, Debug, PartialEq)]
pub struct SpecOpts {
    pub gen: usize,
    pub reps: usize,
    pub depths: Vec<usize>,
    pub dds: bool,
    pub width: usize,
    pub prefix: bool,
    /// 0 = no sampled gate, 1 = WP24 --quick, 2 = the full ladder
    pub samp: u8,
    pub trials: usize,
    pub ident: bool,
}

impl Default for SpecOpts {
    fn default() -> Self {
        SpecOpts { gen: 1024, reps: 3, depths: (1..=crate::exl3_forward::MTP_MAX_K).collect(), dds: true, width: 1,
                   prefix: true, samp: 0, trials: 600, ident: true }
    }
}

impl SpecOpts {
    pub fn parse(mode: &str) -> Result<Self> {
        let mut it = mode.split(':');
        anyhow::ensure!(it.next() == Some("spec"), "EXL3 TP: '{mode}' is not a spec program");
        let mut o = SpecOpts::default();
        for kv in it {
            let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
            let b = |v: &str| v == "1";
            match k {
                "gen" => o.gen = v.parse().context("spec gen=N")?,
                "reps" => o.reps = v.parse().context("spec reps=N")?,
                "depths" => o.depths = if v.is_empty() { Vec::new() } else {
                    v.split(',').map(|x| x.parse::<usize>().context("spec depths=a,b")).collect::<Result<_>>()?
                },
                "dds" => o.dds = b(v),
                "width" => o.width = v.parse().context("spec width=N")?,
                "prefix" => o.prefix = b(v),
                "samp" => o.samp = v.parse().context("spec samp=0|1|2")?,
                "trials" => o.trials = v.parse().context("spec trials=N")?,
                "ident" => o.ident = b(v),
                _ => anyhow::bail!("EXL3 TP: unknown spec option '{kv}' in '{mode}'"),
            }
        }
        anyhow::ensure!(o.depths.iter().all(|&d| (1..=crate::exl3_forward::MTP_MAX_K).contains(&d)),
                        "spec depths must be in 1..={}", crate::exl3_forward::MTP_MAX_K);
        anyhow::ensure!(o.width >= 1 && o.reps >= 1 && o.gen >= 2, "spec: width/reps >= 1, gen >= 2");
        Ok(o)
    }
    pub fn to_mode(&self) -> String {
        format!("spec:gen={}:reps={}:depths={}:dds={}:width={}:prefix={}:samp={}:trials={}:ident={}",
                self.gen, self.reps, self.depths.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(","),
                self.dds as u8, self.width, self.prefix as u8, self.samp, self.trials, self.ident as u8)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum SpecArm {
    Plain,
    Dds,
    Fixed(usize),
}

impl SpecArm {
    fn name(&self) -> String {
        match self { SpecArm::Plain => "plain".into(), SpecArm::Dds => "dds".into(), SpecArm::Fixed(d) => format!("d{d}") }
    }
}

struct SpecArmOut {
    toks: Vec<u32>,
    st: GateStats,
    secs: f64,
    reason: String,
}

/// One greedy request of `prompt` through the real scheduler under `arm` (plain = the §6 per-request
/// fallback's exact hand-off: the lane's MTP state cleared after admit, then step_plain).
fn spec_arm(sched: &mut Exl3Scheduler, arm: SpecArm, prompt: &[u32], gen: usize) -> Result<SpecArmOut> {
    let kdef = crate::exl3_forward::mtp_depth_default();
    match arm {
        SpecArm::Plain | SpecArm::Dds => { sched.mtp_k = kdef; sched.draft_conf = 0.4; }
        SpecArm::Fixed(d) => { sched.mtp_k = d; sched.draft_conf = 0.0; }
    }
    sched.cal = DraftCal::new();
    sched.gate = Some(GateStats::default());
    let (tx, mut rx, bl) = crate::server::tok_channel(); // LR-5 (H6): drained counted below
    let t0 = std::time::Instant::now();
    sched.admit(BatchRequest {
        prompt: prompt.to_vec(), max_new: gen, temperature: 0.0, top_p: 1.0, top_k: 0,
        rep_penalty: 1.0, presence_penalty: 0.0, frequency_penalty: 0.0, min_p: 0.0, min_new: 0,
        ignore_eos: false, tx, seed: Some(0x7C0D_E5EE_D000_0001), ckpt_at: None,
        domain: crate::batch::Domain::General, received_at: std::time::Instant::now(),
        image_embeds: None, image_spans: Vec::new(), schema: None,
    });
    if let Some(f) = sched.fatal.as_ref() { anyhow::bail!("spec: engine fatal at admit: {f}"); }
    if arm == SpecArm::Plain {
        for l in sched.lanes.iter_mut().flatten() { l.mtp = None; }
    }
    let mut steps = 0usize;
    while sched.lanes.iter().any(|l| l.is_some()) {
        sched.step()?;
        if let Some(f) = sched.fatal.as_ref() { anyhow::bail!("spec: engine fatal: {f}"); }
        steps += 1;
        anyhow::ensure!(steps <= 4 * gen + 64, "spec: lane still live after {steps} steps");
    }
    let secs = t0.elapsed().as_secs_f64();
    let (mut toks, mut reason) = (Vec::with_capacity(gen), String::new());
    while let Some(ev) = crate::server::try_recv_counted(&mut rx, &bl) { // LR-5 (H6)
        match ev {
            TokEvent::Tok(x) => toks.push(x),
            TokEvent::Finish { reason: r } => reason = r,
            // #8.2: admission telemetry — no action for a spec gate drain.
            TokEvent::Admitted { .. } => {}
            TokEvent::StopInfo { .. } => {}
        }
    }
    anyhow::ensure!(!reason.starts_with("error"), "spec: {} request failed: {reason}", arm.name());
    Ok(SpecArmOut { toks, st: sched.gate.take().unwrap_or_default(), secs, reason })
}

fn spec_pack(prompts: &[(String, Vec<u32>)]) -> Vec<u32> {
    let mut v = vec![prompts.len() as u32];
    for (n, ids) in prompts {
        let mut b = n.as_bytes().to_vec();
        b.resize(16, 0);
        v.extend(b.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])));
        v.push(ids.len() as u32);
        v.extend_from_slice(ids);
    }
    v
}

fn spec_unpack(v: &[u32]) -> Result<Vec<(String, Vec<u32>)>> {
    let n = *v.first().context("spec: empty prompt pack")? as usize;
    let mut i = 1;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        anyhow::ensure!(i + 5 <= v.len(), "spec: truncated prompt pack");
        let b: Vec<u8> = v[i..i + 4].iter().flat_map(|w| w.to_le_bytes()).take_while(|&c| c != 0).collect();
        let len = v[i + 4] as usize;
        i += 5;
        anyhow::ensure!(i + len <= v.len(), "spec: truncated prompt pack");
        out.push((String::from_utf8_lossy(&b).to_string(), v[i..i + len].to_vec()));
        i += len;
    }
    Ok(out)
}

/// TP-4D: fold the world > 2 end-of-program hub result: `all[r] = [rank r's verdict (1 = OK), rank r's agree count]`.
/// Ok only when EVERY rank reports OK and every rank's agree count equals this rank's (the same lockstep points were
/// crossed everywhere). The line names each rank, so one failing rank of four is visible in every rank's log.
fn tpspec_fold(all: &[Vec<u32>], my_rank: usize, agrees: u64) -> (bool, String) {
    let mut ok = true;
    let mut parts = Vec::new();
    for (r, f) in all.iter().enumerate() {
        let good = f[0] != 0 && f[1] as u64 == agrees;
        ok &= good;
        parts.push(format!("rank {r}{} {} (agrees {})", if r == my_rank { " (this)" } else { "" },
                           if f[0] != 0 { "OK" } else { "FAIL" }, f[1]));
    }
    (ok, format!("{} | this rank agrees {agrees}", parts.join(" | ")))
}

/// The spec program (see the section comment). `ctx` = Some on both TP ranks (the head passes
/// `prompts`, the node None — they arrive over the link); None = TP=1 (prompts required).
pub fn run_spec(model_dir: &str, ctx: Option<crate::tp::TpContext>, prompts: Option<Vec<(String, Vec<u32>)>>,
                max_pos: usize, o: &SpecOpts) -> Result<()> {
    use sha2::{Digest, Sha256};
    let (prompts, attach, rank, world) = match ctx {
        Some(mut ctx) => {
            ctx.sanity()?;
            ctx.branch_check(&crate::tp::TpBranch::Exl3Xtp)?;
            let packed = prompts.as_ref().map(|p| spec_pack(p));
            let (seq, _, _) = match &packed {
                Some(v) => ctx.broadcast_prompt(Some((v, 0, 0)))?,
                None => ctx.broadcast_prompt(None)?,
            };
            let prompts = spec_unpack(&seq)?;
            let (rank, world, link) = ctx.into_parts();
            anyhow::ensure!(crate::net::pin_thread(9), "launch thread failed to pin to core 9 — TP refuses to run unpinned");
            (prompts, Some(crate::exl3_forward::xtp::TpAttach { rank, world, link }), rank, world)
        }
        None => (prompts.context("spec (TP=1): no prompts")?, None, 0, 1),
    };
    let tp_on = attach.is_some();
    // "TP=2 rank N" at world 2 (the label the gate scripts grep), "TP=4 rank N" at world 4
    let who = if tp_on { format!("TP={world} rank {rank}") } else { "TP=1".to_string() };
    if o.prefix { crate::opts::set(crate::opt!("exl3-prefix"), "1"); }
    crate::dprintln!("[tpspec] {who}: program {} — {} prompt(s) {:?}; loading {model_dir} (width {}, max_pos {max_pos}, kv-cache {})",
             o.to_mode(), prompts.len(), prompts.iter().map(|(n, p)| format!("{n}:{}", p.len())).collect::<Vec<_>>(),
             o.width, crate::exl3_forward::kv_fmt_name(crate::exl3_forward::kv_fmt_from_opts()));
    let t_load = std::time::Instant::now();
    let model = FwdModel::load_tp(model_dir, o.width, max_pos, attach)?;
    anyhow::ensure!(model.mtp.is_some(), "spec: the pack has no MTP draft head");
    crate::dprintln!("[tpspec] {who}: loaded in {:.1}s (trunk per rank: {} q / {} kv heads, GDN {} k / {} v heads)",
             t_load.elapsed().as_secs_f32(), model.tc.num_heads, model.tc.num_kv_heads,
             model.tc.lin_num_k_heads, model.tc.lin_num_v_heads);
    let tok = QwenTokenizer::from_file(&format!("{}/tokenizer.json", model_dir.trim_end_matches('/')))?;
    let cfg_eos: u32 = {
        let cfg: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(format!("{}/config.json", model_dir.trim_end_matches('/')))?)?;
        cfg.get("eos_token_id").and_then(|e| e.as_u64()).unwrap_or(248046) as u32
    };
    let eos = tok.stop_token_ids(cfg_eos);
    let mtp_k = crate::exl3_forward::mtp_depth_default();
    let width = o.width;
    let mut sc = FwdModel::scratch(model.dev(), &model.cfg, width.max(mtp_k + 1), model.max_pos())?;
    let chunk = 2048usize;
    let psc0 = model.prefill_scratch(chunk)?;
    model.precapture_decode_graphs(&mut sc, width, mtp_k, true)?;
    let tok = Arc::new(tok);
    // S-B9-REL-MTPREPROBE policy flags (bench/probe paths construct the same scheduler).
    let reprobe_on = crate::opts::var(crate::opt!("mtp-reprobe")).map_or(true, |v| v != "off");
    let reprobe_cfg = if crate::opts::var(crate::opt!("mtp-reprobe-drill")).is_ok() {
        crate::mtp_reprobe::ReProbeCfg::drill()
    } else {
        crate::mtp_reprobe::ReProbeCfg::default()
    };
    let mut sched = Exl3Scheduler {
        spill: None, spill_pol: crate::kv_spill::Policy::new(0), spill_rank: 0,
        model: model.clone(), sc, width,
        lanes: (0..width).map(|_| None).collect(), eos: eos.clone(), chunk, interleave: 0, psc: Some(psc0),
        df2: None, df2_k: crate::dflash2::block() - 1, // this path loads no DF2 artifact
        mtp_k, spec_policy: SpecLanes::Never, lane_order: LaneOrder::Rr, lane_quantum: 256, admit_ctr: 0, fcfs_rots: 0,
        lane_pack: crate::lane_pack::LanePack::Off, scp: None, pack_ready: [false, false], pack_sampled: false, pack_rot: 0, pack_logs: 0, in_prefill: false, pack_stats: PackStats::default(),
        reprobe_on, reprobe_cfg, shared_scale: 1.0, shared_last: false, shared_streak: 0, shared_probe_ctr: 0, spec_win_tok: [0.0; 2], spec_win_ms: [0.0; 2], spec_win_n: [0; 2], spec_win_acc: [0; 2], spec_arm_rate: [0.0; 2], spec_arm_n: [0; 2], spec_arm_fold: [0; 2], spec_probe: None, spec_probe_ctr: 0, spec_est_pick: None, spec_margin: 0.0, spec_margin_n: 0, policy_logs: 0, ctrl_round: 0, ema_plain: 0.0, draft_conf: 0.4, cal: DraftCal::new(),
        reserve: None, reserve_st: crate::lane_reserve::State::new(), rel: (0..width).map(|_| None).collect(),
        prefix_on: o.prefix, cache: (0..width).map(|_| None).collect(),
        wp16: None, wp16_xcheck: false,
        loop_cfg: Some((300, 3)), pen_range: 1024, spec_ratio: true, draft_temp: None, gate: None, tok: tok.clone(),
        exit_on_fatal: false, fatal: None, inject_panic: None, steps: 0, last_resume: (0, 0), mrope_section: None, vis_seam: None, kv_pf: None,
        tp: if tp_on { Some(TpSync::new(rank, world)) } else { None },
    };
    let mut arms = vec![SpecArm::Plain];
    if o.dds { arms.push(SpecArm::Dds); }
    arms.extend(o.depths.iter().map(|&d| SpecArm::Fixed(d)));
    crate::dprintln!("[tpspec] {who}: arms {:?} x {} reps, gen {} greedy (thinking-off chat ids), served depth {mtp_k}, DDS 0.4, \
              prefix cache {}, loop detect (300, 3), precapture done ({} graphs)",
             arms.iter().map(|a| a.name()).collect::<Vec<_>>(), o.reps, o.gen, if o.prefix { "on" } else { "off" },
             model.graph_count());
    let fnv64 = |t: &[u32]| -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for &x in t { h ^= x as u64; h = h.wrapping_mul(0x0000_0100_0000_01b3); }
        h
    };
    let text_sha = |t: &[u32]| -> String {
        let body: Vec<u32> = t.iter().copied().filter(|x| !eos.contains(x)).collect();
        let s = tok.decode(&body, true).unwrap_or_default();
        Sha256::digest(s.as_bytes()).iter().take(8).map(|b| format!("{b:02x}")).collect()
    };
    let t_all = std::time::Instant::now();
    let mut e_ok = true;
    let (mut n_cmp, mut n_bad) = (0usize, 0usize);
    let mut plain_ref: std::collections::BTreeMap<String, Vec<u32>> = Default::default();
    // per (class, arm): (tok/round, ms/round, tok/s) per rep; plain: ms/token
    let mut acc: std::collections::BTreeMap<(String, String), Vec<(f64, f64, f64)>> = Default::default();
    for rep in 0..o.reps {
        if let Some(t) = sched.tp.as_mut() { t.ident = o.ident && rep == 0; }
        for (name, ids) in &prompts {
            let mut rf: Option<Vec<u32>> = None;
            for &arm in &arms {
                let r = spec_arm(&mut sched, arm, ids, o.gen)?;
                let st = &r.st;
                let same = match arm {
                    SpecArm::Plain => {
                        let prev = plain_ref.entry(name.clone()).or_insert_with(|| r.toks.clone());
                        rf = Some(r.toks.clone());
                        *prev == r.toks
                    }
                    _ => rf.as_ref().map_or(false, |p| *p == r.toks),
                };
                n_cmp += 1;
                if !same { n_bad += 1; e_ok = false; }
                let div = rf.as_ref().and_then(|p| p.iter().zip(&r.toks).position(|(a, b)| a != b));
                let n = r.toks.len();
                let (tpr, mpr) = if st.rounds > 0 {
                    ((st.accepted + st.rounds) as f64 / st.rounds as f64, st.ms / st.rounds as f64)
                } else { (0.0, 0.0) };
                let tps = if r.secs > 0.0 { n as f64 / r.secs } else { 0.0 };
                let key = (name.clone(), arm.name());
                acc.entry(key).or_default().push(if arm == SpecArm::Plain { (0.0, r.secs * 1e3 / n.max(1) as f64, tps) } else { (tpr, mpr, tps) });
                if arm == SpecArm::Plain && rep == 0 && rank == 0 {
                    let body: Vec<u32> = r.toks.iter().copied().filter(|x| !eos.contains(x)).collect();
                    let t = tok.decode(&body, true).unwrap_or_default();
                    let (hd, tl): (String, String) = (t.chars().take(160).collect(), t.chars().rev().take(120).collect::<Vec<_>>().into_iter().rev().collect());
                    crate::dprintln!("TPSPEC-TEXT {who} {name}: head {hd:?} ... tail {tl:?}");
                }
                crate::dprintln!("TPSPEC {who} rep {rep} {name} {:<5} n={n} ({}) ids {:016x} text {} | {} | rounds {} drafted {} accepted {} \
                          tok/round {tpr:.3} ms/round {mpr:.2} ctrl {} plain {} | {tps:.1} tok/s ({:.1} s)",
                         arm.name(), r.reason, fnv64(&r.toks), text_sha(&r.toks),
                         if arm == SpecArm::Plain { if same { "plain (rep-stable)".to_string() } else { "PLAIN DIFFERS FROM REP 0".to_string() } }
                         else if same { "== plain".to_string() } else { format!("MISMATCH vs plain at {:?}", div) },
                         st.rounds, st.drafted, st.accepted, st.ctrl, st.plain, r.secs);
            }
        }
        if let Some(t) = sched.tp.as_ref() {
            crate::dprintln!("[tpspec] {who} rep {rep}: lockstep agrees {} ok (agree_ext tripwire armed; proxy watchdog 10 s){}",
                     t.agrees, if t.ident_bufs > 0 { format!("; G-T1-b graphed digests {} compared, {} differ", t.ident_bufs, t.ident_bad) } else { String::new() });
        }
    }
    crate::dprintln!("TPSPEC {who} SUMMARY (mean over {} reps; DDS/fixed: tok/round | ms/round | tok/s; plain: ms/token | tok/s):", o.reps);
    for ((name, arm), v) in &acc {
        let k = v.len() as f64;
        let m = |f: fn(&(f64, f64, f64)) -> f64| v.iter().map(f).sum::<f64>() / k;
        let mn = |f: fn(&(f64, f64, f64)) -> f64| v.iter().map(f).fold(f64::INFINITY, f64::min);
        let mx = |f: fn(&(f64, f64, f64)) -> f64| v.iter().map(f).fold(f64::NEG_INFINITY, f64::max);
        if arm == "plain" {
            crate::dprintln!("  {name:<7} {arm:<5} ms/token {:.2} (min {:.2} max {:.2}) | {:.1} tok/s", m(|x| x.1), mn(|x| x.1), mx(|x| x.1), m(|x| x.2));
        } else {
            crate::dprintln!("  {name:<7} {arm:<5} tok/round {:.3} (min {:.3} max {:.3}) | ms/round {:.2} (min {:.2} max {:.2}) | {:.1} tok/s",
                     m(|x| x.0), mn(|x| x.0), mx(|x| x.0), m(|x| x.1), mn(|x| x.1), mx(|x| x.1), m(|x| x.2));
        }
    }
    let (ident_bufs, ident_bad, agrees) = sched.tp.as_ref().map_or((0, 0, 0), |t| (t.ident_bufs, t.ident_bad, t.agrees));
    let (sp_ok, sp_bad) = (crate::exl3_forward::xtp::SPINE_OK.load(std::sync::atomic::Ordering::Relaxed),
                           crate::exl3_forward::xtp::SPINE_BAD.load(std::sync::atomic::Ordering::Relaxed));
    if tp_on {
        crate::dprintln!("[tpspec] {who}: graphs captured {} ; TP-SPINE audit {sp_ok} graphs OK, {sp_bad} discarded (a discarded shape runs eager)",
                 model.graph_count());
    }
    crate::dprintln!("G-T1-e {who}: {} — {n_cmp} requests compared ({n_bad} differ) over {} prompt(s) x {} arm(s) x {} rep(s), gen {} ({:.1} min)",
             if e_ok { "LOSSLESS_OK (every spec arm == plain, byte-identical; plain rep-stable)" } else { "MISMATCH" },
             prompts.len(), arms.len(), o.reps, o.gen, t_all.elapsed().as_secs_f64() / 60.0);
    let mut samp_ok = true;
    if o.samp > 0 {
        let text = WP24_GATE_PROMPT;
        let p = tok.encode(text, false)?;
        sched.mtp_k = mtp_k;
        let ok = wp24_gate_conditions(&mut sched, &p, text, o.trials.max(50), 6, 2, 20, 0.4, None, o.samp == 1)?;
        samp_ok = ok;
        println!("G-T1-h {who}: {} (WP24 served gate on this scheduler, {}; trials {}/arm, width {width})",
                 if ok { "SERVED_DISTRIBUTION_OK" } else { "FAIL / NO_POWER" }, if o.samp == 1 { "--quick" } else { "full ladder" }, o.trials);
    }
    // (re-read after the sampled gate: its rounds are lockstep points too)
    let (ident_bufs, ident_bad, agrees) = sched.tp.as_ref().map_or((0, 0, 0), |t| (t.ident_bufs, t.ident_bad, t.agrees));
    let mut all_ok = e_ok && samp_ok && ident_bad == 0;
    if let Some(t) = sched.tp.as_ref() {
        if t.world > 2 {
            // TP-4D: world > 2 — every rank's verdict + agree count through the head hub (unbounded: the ranks
            // finish the gate's last rep at different times); every rank prints the same per-rank table
            let all = crate::net::exchange_u32s_all(&[all_ok as u32, t.agrees as u32], 4)?;
            let (ok, line) = tpspec_fold(&all, t.rank as usize, agrees);
            crate::dprintln!("[tpspec] {who}: {line} (graphed digests {ident_bufs} compared, {ident_bad} differ)");
            all_ok &= ok;
        } else {
            let peer = crate::net::exchange_u32s(&[all_ok as u32, t.agrees as u32], 4)?;
            println!("[tpspec] {who}: this rank {} | peer {} (peer agrees {}, this rank {agrees}; graphed digests {ident_bufs} compared, {ident_bad} differ)",
                     if all_ok { "OK" } else { "FAIL" }, if peer[0] != 0 { "OK" } else { "FAIL" }, peer[1]);
            all_ok &= peer[0] != 0 && peer[1] as u64 == agrees;
        }
    }
    println!("RESULT: {} ({who}; G-T1-e {}{}{})", if all_ok { "TPSPEC_OK" } else { "TPSPEC_FAIL" },
             if e_ok { "OK" } else { "MISMATCH" },
             if o.samp > 0 { if samp_ok { " ; G-T1-h OK" } else { " ; G-T1-h FAIL" } } else { "" },
             if tp_on { if ident_bad == 0 { " ; RANK_IDENTITY_OK (graphed)" } else { " ; RANK_IDENTITY_FAIL" } } else { "" });
    anyhow::ensure!(all_ok, "spec program gates failed ({who})");
    Ok(())
}

/// TP-4D: CPU models of every world > 2 lockstep op of the serve path — the REAL `TpSync::*_hub` functions and the
/// REAL `net::hub_all` over the in-memory `net::hub_mock` transport, one thread per rank (4 ranks). What these prove:
/// the merged hub ops are symmetric (every rank runs the same op sequence and reaches the same verdict), counters
/// stay in lockstep, nodes adopt the head's words, a divergent / dead / hung rank fails EVERY live rank (none is left
/// parked), and the WP16 allocation agree keeps four checkpoint stores identical. What they do NOT prove: anything
/// about RDMA placement, the proxy, the hot-path rings or the GPU (hardware questions for the W=4 gate run).
#[cfg(test)]
mod tp_hub_tests {
    use super::*;
    use crate::net::hub_mock::{run_ranks, Net};
    use std::time::Duration;

    const W: usize = 4;

    fn sync(rank: usize) -> TpSync { TpSync::new(rank as i32, W as i32) }

    #[test]
    fn step_go_hub_adopts_head_words_and_advances_counters_in_lockstep() {
        let net = Net::new(W, true);
        let out = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
            let mut t = sync(r);
            let mut got = Vec::new();
            for round in 0..5u64 {
                let (events, local_step) = if r == 0 { ((round % 3) as usize, 100 + round) } else { (0, 100 + round) };
                got.push(t.step_go_hub(hub, local_step, events).expect("a healthy step-go"));
            }
            (got, t.step)
        });
        for (r, (got, step)) in out.iter().enumerate() {
            assert_eq!(*step, 5, "rank {r}: one lockstep counter tick per step-go");
            let want: Vec<(u64, usize)> = (0..5u64).map(|i| (100 + i, (i % 3) as usize)).collect();
            assert_eq!(got, &want, "rank {r} adopts the head's (step, events)");
        }
        assert!(net.clobbers().is_empty(), "control ring clobbers: {:?}", net.clobbers());
    }

    #[test]
    fn a_rank_one_lockstep_point_ahead_fails_every_rank_and_names_it() {
        let net = Net::new(W, true);
        let out = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
            let mut t = sync(r);
            if r == 2 { t.step += 1; }
            t.step_go_hub(hub, 7, 0).map_err(|e| format!("{e:#}"))
        });
        for (r, o) in out.iter().enumerate() {
            let e = o.as_ref().expect_err(&format!("rank {r} must fail: the ranks are at different lockstep points"));
            assert!(e.contains("step-go") && e.contains("FAILED") && e.contains("rank 2 counter field"), "rank {r}: {e}");
        }
    }

    #[test]
    fn pre_verify_hub_compares_width_drafts_and_launches_on_every_rank() {
        let net = Net::new(W, true);
        let ok = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
            let mut t = sync(r);
            let res = t.pre_verify_hub(hub, 6, &[1, 2, 3, 4, 5], 7);
            (res.is_ok(), t.pre_agrees)
        });
        assert!(ok.iter().all(|&(good, n)| good && n == 1), "{ok:?}");
        for (what, r_bad) in [("drafts", 3usize), ("width", 1), ("launched", 2)] {
            let net = Net::new(W, true);
            let out = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
                let mut t = sync(r);
                let (w, d, l) = match (what, r == r_bad) {
                    ("drafts", true) => (6, vec![1, 2, 3, 4, 6], 7),
                    ("width", true) => (5, vec![1, 2, 3, 4, 5], 7),
                    ("launched", true) => (6, vec![1, 2, 3, 4, 5], 8),
                    _ => (6, vec![1, 2, 3, 4, 5], 7),
                };
                t.pre_verify_hub(hub, w, &d, l).map_err(|e| format!("{e:#}"))
            });
            for (r, o) in out.iter().enumerate() {
                let e = o.as_ref().expect_err(&format!("{what}: rank {r} must fail on a divergent rank {r_bad}"));
                assert!(e.contains("pre-verify") && e.contains(&format!("rank {r_bad}")), "{what} rank {r}: {e}");
            }
        }
        // S-B9-CF-AGREE16: the widths a packed verify would reach (today's pre-verify sees at most 7 drafts).
        // One rank drafted for a 15-row verify, the others for a 16-row one (and 16 vs 17, 15 vs 17): the SAME
        // drafts hash on every rank, so the width word alone must fail the op, on every rank, before the verify
        // launches. (This frame always carried the whole width: the case pins it, it was not red on master.)
        let drafts: Vec<i32> = (0..16).collect();
        for (w_most, w_bad, r_bad) in [(16usize, 15usize, 1usize), (15, 16, 2), (17, 16, 3), (16, 17, 1), (17, 15, 2)] {
            let net = Net::new(W, true);
            let out = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
                let mut t = sync(r);
                t.pre_verify_hub(hub, if r == r_bad { w_bad } else { w_most }, &drafts, 3).map_err(|e| format!("{e:#}"))
            });
            for (r, o) in out.iter().enumerate() {
                let e = o.as_ref().expect_err(&format!("width {w_most} vs {w_bad}: rank {r} must fail"));
                assert!(e.contains("pre-verify") && e.contains(&format!("rank {r_bad} width")), "{w_most} vs {w_bad}, rank {r}: {e}");
            }
        }
        for w in [15usize, 16, 17] {
            let net = Net::new(W, true);
            let ok = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| sync(r).pre_verify_hub(hub, w, &drafts, 3).is_ok());
            assert!(ok.iter().all(|&g| g), "every rank at width {w} must agree: {ok:?}");
        }
    }

    /// S-B9-CF-PACK5, the fault drill at unit scale (fake hub transport, every rank a thread): the
    /// PACKED round's pre-verify frame. The words are `lane_pack::pre_words` of a two-lane pack; R is
    /// the explicit row total. One rank disagreeing about the pack (a different draft in either lane,
    /// the same words split differently, another slot, another position) or about R must fail EVERY
    /// rank at this exchange, before any packed verify could launch, with the combined-hash message
    /// naming the offending rank; identical packs pass. (The world-2 path is the same words through
    /// `exchange_u32s` + `tp_agree_eq`, which needs the real link: the live drill covers it.)
    #[test]
    fn the_pack_pre_verify_catches_every_disagreement_about_the_pack_before_the_verify() {
        use crate::lane_pack::{pack_rows_total, pre_words, PreSeg};
        let (d0, d1) = ([11, 12, 13], [21, 22]);
        let mk = |slot1: usize, pos1: usize, e1: [i32; 2]| -> (usize, Vec<i32>) {
            let segs = [PreSeg::greedy(1, 100, 7, &d0), PreSeg::greedy(slot1, pos1, 8, &e1)];
            (pack_rows_total(&segs), pre_words(&segs))
        };
        let base = mk(2, 200, d1);
        let net = Net::new(W, true);
        let ok = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
            let mut t = sync(r);
            let res = t.pre_verify_hub(hub, base.0, &base.1, 5);
            (res.is_ok(), t.pre_agrees)
        });
        assert!(ok.iter().all(|&(good, n)| good && n == 1), "{ok:?}");
        assert_eq!(base.0, 7);
        for (what, r_bad, other) in [("a draft of lane 1", 3usize, mk(2, 200, [21, 23])), ("lane 1's slot", 1, mk(3, 200, d1)),
                                     ("lane 1's position", 2, mk(2, 201, d1))] {
            let net = Net::new(W, true);
            let out = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
                let mut t = sync(r);
                let (rows, words) = if r == r_bad { (other.0, &other.1) } else { (base.0, &base.1) };
                t.pre_verify_hub(hub, rows, words, 5).map_err(|e| format!("{e:#}"))
            });
            for (r, o) in out.iter().enumerate() {
                let e = o.as_ref().expect_err(&format!("{what}: rank {r} must fail on a divergent rank {r_bad}"));
                assert!(e.contains("pre-verify") && e.contains(&format!("rank {r_bad}")) && e.contains("link aborted before the verify"),
                        "{what} rank {r}: {e}");
            }
        }
        // the row total alone: SAME words, one rank believes R is one row wider (a 15- vs 16-row pack)
        for (r_most, r_bad, who) in [(15usize, 16usize, 2usize), (16, 15, 1)] {
            let net = Net::new(W, true);
            let out = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
                let mut t = sync(r);
                t.pre_verify_hub(hub, if r == who { r_bad } else { r_most }, &base.1, 5).map_err(|e| format!("{e:#}"))
            });
            for (r, o) in out.iter().enumerate() {
                let e = o.as_ref().expect_err(&format!("R {r_most} vs {r_bad}: rank {r} must fail"));
                assert!(e.contains(&format!("rank {who} width")), "R {r_most} vs {r_bad}, rank {r}: {e}");
            }
        }
    }

    /// S-B9-CF-AGREE16: the post-verify round of the world > 2 lockstep. `words` (emitted ids, drafts,
    /// confidences) are the same on every rank here, so the hash is equal and ONLY the width (then only the
    /// accept count) tells the ranks apart: every unequal pair in 0..=17 must fail every rank and name the
    /// offender; every equal pair must pass.
    #[test]
    fn round_hub_tells_every_width_and_accept_count_apart_0_to_17() {
        let words = [11u32, 22, 33];
        let run = |a_most: usize, a_bad: usize, m_most: usize, m_bad: usize| -> Vec<std::result::Result<(), String>> {
            let net = Net::new(W, true);
            run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
                let mut t = sync(r);
                let (a, m) = if r == 1 { (a_bad, m_bad) } else { (a_most, m_most) };
                t.round_hub(hub, "verify", a, m, &words, 1.0, 1.0, false, 0, |_| Ok(vec![])).map(|_| ()).map_err(|e| format!("{e:#}"))
            })
        };
        let mut silent: Vec<String> = Vec::new();
        for m_most in 0..=17usize {
            for m_bad in 0..=17usize {
                let out = run(3, 3, m_most, m_bad);
                if m_most == m_bad {
                    assert!(out.iter().all(|o| o.is_ok()), "width {m_most} on every rank must agree: {out:?}");
                } else if out.iter().any(|o| o.is_ok()) {
                    silent.push(format!("width {m_most} vs {m_bad}"));
                } else {
                    for (r, o) in out.iter().enumerate() {
                        let e = o.as_ref().unwrap_err();
                        assert!(e.contains("rank 1 verify width"), "width {m_most} vs {m_bad}, rank {r}: {e}");
                    }
                }
            }
        }
        for a_most in 0..=17usize {
            for a_bad in 0..=17usize {
                let out = run(a_most, a_bad, 17, 17);
                if a_most == a_bad {
                    assert!(out.iter().all(|o| o.is_ok()), "accept {a_most} on every rank must agree: {out:?}");
                } else if out.iter().any(|o| o.is_ok()) {
                    silent.push(format!("accept {a_most} vs {a_bad}"));
                } else {
                    for (r, o) in out.iter().enumerate() {
                        let e = o.as_ref().unwrap_err();
                        assert!(e.contains("rank 1 accept count"), "accept {a_most} vs {a_bad}, rank {r}: {e}");
                    }
                }
            }
        }
        assert!(silent.is_empty(), "{} unequal pairs passed the round in silence: {silent:?}", silent.len());
        // beyond a byte: the frame words are whole u32s, so 255 vs 256 accepted ids are told apart as well
        assert!(run(255, 256, 17, 17).iter().all(|o| o.is_err()), "accept 255 vs 256 must fail the round");
        assert!(run(3, 3, 31, 32).iter().all(|o| o.is_err()), "width 31 vs 32 must fail the round");
    }

    /// S-B9-CF-AGREE16, the fault drill at unit scale: every rank runs a 16-row round, ONE rank carries
    /// `--tp-width-drill 16` (the flag's name since AGREE16B) and therefore REPORTS 15 - rank. The guard
    /// must fail the round on every rank.
    #[test]
    fn the_width_drill_on_one_rank_fails_a_16_row_round_on_every_rank() {
        for drill_rank in [0usize, 2] {
            let net = Net::new(W, true);
            let out = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
                let mut t = sync(r);
                if r == drill_rank { t.drill_width = Some(16); }
                // a 5-row round first: the drill leaves every width but its own alone
                let first = t.round_hub(hub, "verify", 2, 5, &[1, 2], 1.0, 1.0, false, 0, |_| Ok(vec![])).map(|_| ()).map_err(|e| format!("{e:#}"));
                let second = t.round_hub(hub, "verify", 2, 16, &[1, 2], 1.0, 1.0, false, 0, |_| Ok(vec![])).map(|_| ()).map_err(|e| format!("{e:#}"));
                (first, second, t.drill_hits)
            });
            for (r, (first, second, hits)) in out.iter().enumerate() {
                assert!(first.is_ok(), "rank {r}: the 5-row round is not the drill's width: {first:?}");
                let e = second.as_ref().expect_err(&format!("rank {r}: rank {drill_rank} reported 15 for a 16-row round — the round must fail"));
                assert!(e.contains("verify width"), "rank {r}: {e}");
                assert_eq!(*hits, (r == drill_rank) as u64, "rank {r}");
            }
        }
    }

    #[test]
    fn round_hub_ships_head_timing_and_reports_ident_digests() {
        let net = Net::new(W, true);
        let out = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
            let mut t = sync(r);
            t.ident = true;
            let words = [3u32, 1, 4, 1, 5];
            // rank 3's second digest differs: counted, but it is diagnostic (the round itself stays OK)
            let res = t.round_hub(hub, "verify", 3, 5, &words, if r == 0 { 12.5 } else { 99.0 }, 20.0, true, 1000 + r as u64,
                                  |m| { assert_eq!(m, 5); Ok(vec![0xAA, if r == 3 { 0xBC } else { 0xBB }, 0xCC]) });
            (res.expect("a healthy round"), t.agrees, t.ident_bufs, t.ident_bad, t.step)
        });
        for (r, (res, agrees, bufs, bad, step)) in out.iter().enumerate() {
            assert_eq!(*res, (12.5, 20.0, true), "rank {r} gets the head's timing, not its own");
            assert_eq!((*agrees, *bufs, *bad, *step), (1, 3, 1, 1), "rank {r}");
        }
    }

    #[test]
    fn round_hub_fails_every_rank_on_a_divergent_accept_count() {
        let net = Net::new(W, true);
        let out = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
            let mut t = sync(r);
            t.round_hub(hub, "verify", if r == 1 { 2 } else { 3 }, 5, &[1, 2, 3], 1.0, 1.0, false, 0, |_| Ok(vec![]))
                .map(|_| ()).map_err(|e| format!("{e:#}"))
        });
        for (r, o) in out.iter().enumerate() {
            let e = o.as_ref().expect_err(&format!("rank {r} must fail"));
            assert!(e.contains("rank 1 accept count"), "rank {r}: {e}");
        }
    }

    #[test]
    fn a_node_that_dies_fails_every_live_rank_and_a_hung_one_trips_the_deadline() {
        // dead node: rank 2 "crashes" before its step-go (its thread ends: the mock's dead-peer probe fires)
        let net = Net::new(W, true);
        let out = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
            if r == 2 { return Err("node crashed".to_string()); }
            sync(r).step_go_hub(hub, 9, 1).map(|_| ()).map_err(|e| format!("{e:#}"))
        });
        for (r, o) in out.iter().enumerate() {
            let e = o.as_ref().expect_err(&format!("rank {r}: a dead rank 2 must fail the op"));
            if r != 2 { assert!(e.contains("step-go") && e.contains("FAILED"), "rank {r}: {e}"); }
        }
        assert!(net.abort_code(0) == 10 || net.abort_code(0) == 0, "the head saw the dead peer (code 10) or a failed verdict");
        // hung node: alive but never posts; the head's deadline fires (code 12) and every other rank fails too
        let net = Net::new(W, true);
        let out = run_ranks(&net, Some(Duration::from_millis(300)), |r, hub| {
            if r == 3 { std::thread::sleep(Duration::from_millis(1500)); return Err("hung".to_string()); }
            sync(r).step_go_hub(hub, 9, 1).map(|_| ()).map_err(|e| format!("{e:#}"))
        });
        for r in 0..3 { assert!(out[r].is_err(), "rank {r} must not stay parked behind a hung rank 3"); }
        assert_eq!(net.abort_code(0), 12, "the head's wait hit the lockstep deadline");
    }

    #[test]
    fn checkpoint_allocation_outcome_is_agreed_so_four_stores_stay_identical() {
        use crate::exl3_forward::CkptStore;
        // rank 2's third allocation fails (its device memory is tighter); the others would succeed.
        let local = |r: usize, n_before: usize| -> Result<u32> {
            if r == 2 && n_before >= 2 { anyhow::bail!("CUDA_ERROR_OUT_OF_MEMORY (model)") } else { Ok(n_before as u32 + 1) }
        };
        // WITHOUT the agree (today's world-2 behaviour): the stores diverge — the C2 bug, reproduced
        let plain: Vec<(usize, usize)> = (0..W).map(|r| {
            let mut st: CkptStore<u32> = CkptStore::new(1, 8);
            st.begin_aligned(0, 0, 2048);
            for i in 0..5 {
                let n = st.n_alloc;
                if let Some(b) = st.acquire(|| local(r, n)) { st.insert(0, 2048 * (i + 1), b); }
            }
            (st.n_alloc, st.live())
        }).collect();
        assert!(plain.iter().any(|&p| p != plain[0]), "without the agree rank 2 freezes at 2 while the others grow: {plain:?}");
        // WITH the agree: every rank's store ends identical
        let net = Net::new(W, true);
        let agreed = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
            let mut t = sync(r);
            let mut st: CkptStore<u32> = CkptStore::new(1, 8);
            st.begin_aligned(0, 0, 2048);
            let mut hub_err = None;
            for i in 0..5 {
                let n = st.n_alloc;
                let got = st.acquire(|| { let l = local(r, n); t.alloc_outcome(hub, n, l, &mut hub_err) });
                assert!(hub_err.is_none(), "rank {r}: a healthy hub");
                if let Some(b) = got { st.insert(0, 2048 * (i + 1), b); }
            }
            (st.n_alloc, st.live(), st.positions(0))
        });
        assert!(agreed.iter().all(|a| *a == agreed[0]), "the four stores must be identical: {agreed:?}");
        assert_eq!(agreed[0].0, 2, "the cap froze at the two buffers every rank could allocate");
        assert!(net.clobbers().is_empty());
    }

    #[test]
    fn a_failed_alloc_agree_is_reported_not_swallowed_as_an_allocation_failure() {
        use crate::exl3_forward::CkptStore;
        // rank 3 dies before the allocation agree: the live ranks' `acquire` would freeze the cap on the Err, so the
        // transport failure must come back through `hub_err`
        let net = Net::new(W, true);
        let out = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
            if r == 3 { return None; }
            let mut t = sync(r);
            let mut st: CkptStore<u32> = CkptStore::new(1, 8);
            let mut hub_err = None;
            let _ = st.acquire(|| t.alloc_outcome(hub, 0, Ok(1u32), &mut hub_err));
            hub_err.map(|e| format!("{e:#}"))
        });
        for r in 0..3 {
            let e = out[r].as_ref().unwrap_or_else(|| panic!("rank {r}: the agree failure must surface in hub_err"));
            assert!(e.contains("wp16 alloc"), "rank {r}: {e}");
        }
    }

    #[test]
    fn boot_hash_verdict_names_every_differing_rank() {
        let ok: Vec<Vec<u32>> = (0..W).map(|_| vec![BOOT_AGREE_MAGIC, 0x1234]).collect();
        assert_eq!(boot_hash_verdict(&ok, 1), Ok(()));
        let mut bad = ok.clone();
        bad[2][1] = 0x9999;
        bad[3][0] = 0x6000_0001;
        let e = boot_hash_verdict(&bad, 1).unwrap_err();
        assert!(e.contains("rank 2 hash 00009999 != head 00001234") && e.contains("rank 3 sent frame class") && e.contains("this is rank 1"), "{e}");
        assert!(!e.contains("rank 1 hash") && !e.contains("rank 0"), "{e}");
    }

    #[test]
    fn tpspec_fold_requires_every_rank_ok_with_equal_agree_counts() {
        let good: Vec<Vec<u32>> = (0..W).map(|_| vec![1, 77]).collect();
        let (ok, line) = tpspec_fold(&good, 2, 77);
        assert!(ok && line.contains("rank 2 (this) OK (agrees 77)"), "{line}");
        let mut one_fail = good.clone();
        one_fail[3][0] = 0;
        let (ok, line) = tpspec_fold(&one_fail, 0, 77);
        assert!(!ok && line.contains("rank 3 FAIL"), "{line}");
        let mut skew = good.clone();
        skew[1][1] = 76;
        let (ok, _) = tpspec_fold(&skew, 0, 77);
        assert!(!ok, "a rank that crossed a different number of lockstep points must fail the fold");
    }
}

#[cfg(test)]
mod fit_tests {
    use super::{fit_parts, fit_plan, FitParts};
    const G: u64 = 1_000_000_000;

    #[test]
    fn fit_plan_refuses_with_the_arithmetic_and_the_way_out() {
        let p = FitParts { weights: 53 * G, table: 33 * G, per_lane: 3 * G, fixed: 19 * G, floor: 5 * G };
        // 1 lane: fits with the table in RAM.
        assert_eq!(fit_plan(p, 1, 124 * G, "auto", 131072), Ok(true));
        // 8 lanes: fits only on SSD; auto serves it, ram is refused and told to use ssd.
        assert_eq!(fit_plan(p, 8, 124 * G, "auto", 131072), Ok(false));
        let e = fit_plan(p, 8, 124 * G, "ram", 131072).unwrap_err();
        assert!(e.contains("--ple-ram ssd") && e.contains("does not fit"), "{e}");
        // 20 lanes: does not fit even on SSD; names the lane and context limits.
        let e = fit_plan(p, 20, 124 * G, "auto", 131072).unwrap_err();
        assert!(e.contains("even with the n-gram table on SSD") && e.contains("--max-batch 15"), "{e}");
        // the weights alone overflow: says TP=2.
        let e = fit_plan(FitParts { weights: 110 * G, ..p }, 1, 124 * G, "ssd", 131072).unwrap_err();
        assert!(e.contains("--tp 2"), "{e}");
    }

    /// MAXBATCH1: the boot-anyway clamp. The same arithmetic fit_plan refuses with becomes the
    /// answer to "what is the largest lane count that BOOTS": the .12 refusal of 2026-10-21
    /// (16 lanes, 262K context, 125.5 GB box) resolves to 8 lanes, the table landing on SSD
    /// exactly like a plain boot that only fits on SSD. None is reserved for "not even one lane
    /// can serve" — the one refusal CF-P1d keeps — and a forced `--ple-ram ram` is honoured
    /// (the explicit user demand), not silently downgraded to SSD.
    #[test]
    fn fit_max_lanes_picks_the_largest_lane_count_that_boots() {
        use super::fit_max_lanes;
        // The .12 2026-10-21 boot: weights 53.2, working 19.3, floor 5.0, per-lane 5.61 at 262145.
        let p = FitParts { weights: 53_200_000_000, table: 33_000_000_000, per_lane: 5_610_000_000,
                           fixed: 19_300_000_000, floor: 5_000_000_000 };
        assert_eq!(fit_max_lanes(p, 16, 125_500_000_000, "auto", 262145), Some(8),
                   "the refused boot clamps to the 8 lanes that fit (9 lanes = 127.5 GB > 125.5)");
        // A generous box: nothing is clamped away, the table fits in RAM.
        assert_eq!(fit_max_lanes(p, 6, 200 * G, "auto", 262145), Some(6));
        // The clamp never exceeds the asked width.
        assert_eq!(fit_max_lanes(p, 4, 200 * G, "auto", 262145), Some(4));
        // Weights alone overflow: None — CF-P1d's refusal (use --tp 2) survives.
        assert_eq!(fit_max_lanes(FitParts { weights: 110 * G, ..p }, 16, 124 * G, "ssd", 131072), None);
        // Forced --ple-ram ram with a table that cannot sit in RAM: no lane count satisfies the
        // explicit demand → None keeps the refusal; auto on the same box boots 5 lanes.
        assert_eq!(fit_max_lanes(p, 16, 110 * G, "ram", 262145), None);
        assert_eq!(fit_max_lanes(p, 16, 110 * G, "auto", 262145), Some(5));
    }

    /// The real Flash-Next 3.05 pack, where present: the estimate must bracket the 2026-10-02 measurement on .14
    /// (per lane at 131K ~2.8 GB measured; device weights ~53 GB; table 32.6 GB).
    #[test]
    fn flash_next_305_estimate_brackets_the_measurement() {
        let dir = std::path::Path::new(&std::env::var("HOME").unwrap_or_default()).join("models/Qwen3.8-Flash-Next-exl3-3.05bpw");
        if !dir.is_dir() { return; }
        let p = fit_parts(dir.to_str().unwrap(), 131073, 1, true, 1.0625, 4.0).unwrap();
        assert!((2_600_000_000..3_300_000_000).contains(&p.per_lane), "{p:?}");
        assert!((50 * G..56 * G).contains(&p.weights), "{p:?}");
        assert!((32 * G..34 * G).contains(&p.table), "{p:?}");
        // MAXBATCH1: the serve fit reserves the §5 serving-headroom bar (>= 20 GB), not the 5 GB
        // emergency-watchdog floor — a 2026-10-21 .12 boot at the 5 GB floor rode it and died mid-sweep.
        assert!(p.floor >= 20 * G, "serve-fit floor must reserve the serving headroom: {p:?}");
    }
}

#[cfg(test)]
mod spec_lanes_tests {
    use super::{fcfs_charge, fcfs_front, shared_step_prior_ms, LaneOrder, SpecLanes};

    /// CF-FCFS: the ordering function is a pure function of (admit_seq, credit) — no GPU, no
    /// scheduler. The credit/rotation bookkeeping itself is `fcfs_charge`, exercised by
    /// `fcfs_simulation_three_lanes` below, which drives this same pair exactly as step() does.
    #[test]
    fn fcfs_order_is_pure() {
        // (slot, admit_seq, fcfs_credit)
        let q = 4;
        // Two lanes admitted in order, both fresh: the OLDEST (lowest admit_seq) is the front, not
        // the lowest slot.
        let two = [(0usize, 0u64, 0usize), (1, 1, 0)];
        assert_eq!(fcfs_front(&two, q), Some(0));
        // Slot order and admission order disagree: admission order wins.
        let swapped = [(0usize, 7u64, 0usize), (1, 3, 0)];
        assert_eq!(fcfs_front(&swapped, q), Some(1));
        // A lane whose credit reached the quantum (as fcfs_charge leaves it: credit 0 AND the
        // HIGHEST admit_seq after rotation) would only be the front if it were the sole lane.
        let one_left = [(0usize, 0u64, 0usize)];
        assert_eq!(fcfs_front(&one_left, q), Some(0));
        // The front finishes (its lane is gone from the list) -> the next-oldest runs at once.
        let finished = [(1usize, 1u64, 0)];
        assert_eq!(fcfs_front(&finished, q), Some(1));
        // A new arrival does NOT jump the queue: it has the highest admit_seq.
        let arrived = [(0usize, 0u64, 0), (1, 1, 0), (2, 2, 0)];
        assert_eq!(fcfs_front(&arrived, q), Some(0));
        // A lane with a partially used credit is still the front (its quantum is not exhausted).
        let partial = [(0usize, 0u64, 3), (1, 1, 0)];
        assert_eq!(fcfs_front(&partial, q), Some(0));
        // No capable lane at all (all MTP-off) -> the caller runs the plain batch only.
        assert_eq!(fcfs_front(&[], q), None);
        // Defensive guard on an unreachable state (charge resets credit AT the quantum): still
        // answers, never panics.
        let all_done = [(0usize, 0u64, q), (1, 1, q + 3)];
        assert_eq!(fcfs_front(&all_done, q), None);
        // Purity: the same input always gives the same front, whatever the slot order.
        let a = [(5usize, 2u64, 0), (3, 9, 0), (7, 4, q)];
        let b = [(7usize, 4u64, q), (5, 2, 0), (3, 9, 0)];
        assert_eq!(fcfs_front(&a, q), fcfs_front(&b, q));
        assert_eq!(fcfs_front(&a, q), Some(5));
        // A huge quantum never rotates (one lane runs to completion).
        assert_eq!(fcfs_front(&two, 1 << 20), Some(0));
    }

    /// CF-FCFSB: a multi-step simulation that drives the SAME bookkeeping pair step() uses —
    /// `fcfs_front` to pick, `fcfs_charge` to charge and rotate — over 3 lanes whose turns are
    /// longer than the quantum, one lane finishing mid-turn and a new arrival. Asserts the actual
    /// SERVICE ORDER (who steps at each scheduler step): a lane that used its quantum goes BEHIND
    /// the others, and the old front only gets its next turn after every waiting lane had one.
    /// (Before the fix this test fails: the rotated lane kept the lowest admit_seq and stepped
    /// forever — the order degenerated to [0,0,0,...].)
    #[test]
    fn fcfs_simulation_three_lanes() {
        let q = 4usize;
        // (slot, admit_seq, credit, tokens_left). Lanes 0 and 1 have 10-token turns, lane 2 a
        // 5-token turn; each speculative round generates 2 tokens. Quantum 4 => 2 consecutive
        // rounds per turn, then rotate.
        let mut lanes: Vec<Option<(usize, u64, usize, usize)>> =
            vec![Some((0, 0, 0, 10)), Some((1, 1, 0, 10)), Some((2, 2, 0, 5))];
        let mut admit_ctr = 3u64;
        let mut rotations = 0usize;
        let mut service: Vec<usize> = Vec::new();
        let mut lane3_admitted = false;
        for _step in 0..100 {
            // Lane 3 is admitted once lane 2 has finished (the scheduler admits into a free slot).
            if !lane3_admitted && lanes[2].is_none() {
                lanes[2] = Some((2, admit_ctr, 0, 10));
                admit_ctr += 1;
                lane3_admitted = true;
            }
            let capable: Vec<(usize, u64, usize)> = lanes
                .iter()
                .filter_map(|l| l.as_ref().map(|(s, seq, c, _)| (*s, *seq, *c)))
                .collect();
            let Some(front) = fcfs_front(&capable, q) else { break };
            let lane = lanes
                .iter_mut()
                .find_map(|l| l.as_mut().filter(|(s, ..)| *s == front))
                .unwrap();
            let delta = lane.3.min(2);
            lane.3 -= delta;
            let (mut seq, mut credit) = (lane.1, lane.2);
            if fcfs_charge(&mut seq, &mut credit, delta, q, &mut admit_ctr) {
                rotations += 1;
            }
            lane.1 = seq;
            lane.2 = credit;
            service.push(front);
            if lane.3 == 0 {
                lanes.iter_mut().find(|l| matches!(l, Some((s, ..)) if *s == front)).unwrap().take();
            }
        }
        // Hand-derived from the bookkeeping above: two rounds per quantum, then rotate; lane 2's
        // last round is a 1-token partial (credit 1, no rotation); lanes 0/1 finish with credit 2.
        // The rotated lane is served again ONLY after both other lanes had their turns; the new
        // arrival (lane 3, admitted into slot 2) runs only after lanes 0 and 1 finished entirely.
        assert_eq!(service, [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 0, 1, 2, 2, 2, 2, 2]);
        assert!(lane3_admitted);
        assert_eq!(rotations, 7); // 5 in the first two cycles + 2 from the lane-3 turn
        assert_eq!(admit_ctr, 11); // 3 initial + 7 rotation re-stamps + 1 admission stamp
    }

    #[test]
    fn lane_order_parse() {
        assert_eq!(LaneOrder::parse(None).unwrap(), LaneOrder::Rr);
        assert_eq!(LaneOrder::parse(Some("")).unwrap(), LaneOrder::Rr);
        assert_eq!(LaneOrder::parse(Some("rr")).unwrap(), LaneOrder::Rr);
        assert_eq!(LaneOrder::parse(Some("round-robin")).unwrap(), LaneOrder::Rr);
        assert_eq!(LaneOrder::parse(Some("fcfs")).unwrap(), LaneOrder::Fcfs);
        assert!(LaneOrder::parse(Some("fifo")).is_err());
        assert!(LaneOrder::parse(Some("RR")).is_err());
    }

    #[test]
    fn spec_lanes_parse() {
        assert_eq!(SpecLanes::parse(None).unwrap(), SpecLanes::Auto);
        assert_eq!(SpecLanes::parse(Some("auto")).unwrap(), SpecLanes::Auto);
        assert_eq!(SpecLanes::parse(Some("0")).unwrap(), SpecLanes::Never);
        assert_eq!(SpecLanes::parse(Some("off")).unwrap(), SpecLanes::Never);
        assert_eq!(SpecLanes::parse(Some("4")).unwrap(), SpecLanes::Over(4));
        assert!(SpecLanes::parse(Some("many")).is_err());
        assert!(SpecLanes::parse(Some("-1")).is_err());
    }

    /// The prior reproduces the 2026-10-02 TP=1 measurements within 5% (n = 4: 46.5, 8: 61.7, 16: 90 ms).
    #[test]
    fn shared_step_prior_matches_measurements() {
        for (n, ms) in [(4usize, 46.5f64), (8, 61.7), (16, 90.0)] {
            let p = shared_step_prior_ms(n);
            assert!((p - ms).abs() / ms < 0.05, "n={n}: {p} vs {ms}");
        }
    }

    /// F4: the shared-cost probe fires EARLY while the scale has never been learned (the wrong
    /// arm must not own a quarter of a ~250-round request) and at the 1-in-128 cadence afterwards,
    /// mirroring the shared arm's own stale-estimate refresh.
    #[test]
    fn the_shared_probe_fires_early_then_every_128() {
        assert_eq!(super::SHARED_PROBE_FIRST, 16);
        assert_eq!(super::SHARED_PROBE_EVERY, 128);
        let due = super::shared_probe_due;
        assert!(!due(0, true) && !due(15, true), "not before 16 rounds on the other arm");
        assert!(due(16, true) && due(17, true), "due at 16 while the scale is unlearned");
        assert!(!due(15, false) && !due(127, false), "learned scale: 1-in-128 cadence");
        assert!(due(128, false) && due(1000, false));
    }

    /// S-C4XOVER: the arm-rate probe (which generalizes F4's shared-cost probe) sits BEFORE
    /// the arm pick (it can force either arm for a few steps), gated on `--lane-pack on` +
    /// the Auto policy + >=2 busy lanes with at least one MTP-capable lane, and only starts
    /// when the idle arm's measured rate has gone stale. A forced-shared window runs the
    /// shared arm's own step (not a new path) and the 128th-round refresh stays off while a
    /// probe window is in flight. Flag-off boots (G2 byte identity), `Never`/`Over` policies
    /// and single-lane steps never reach it.
    #[test]
    fn the_arm_rate_probe_is_flag_on_auto_two_lanes_and_preempts_the_pick() {
        let src = include_str!("exl3_serve.rs");
        let pick = src.find("if mtp_on && probe_force.unwrap_or_else(|| self.use_shared_plain(&slots)) {")
            .expect("arm pick in step()");
        let probe = src.find("S-C4XOVER arm-rate probe").expect("probe block in step()");
        assert!(probe < pick, "the probe preempts the arm pick when due");
        let gate = &src[probe..pick];
        assert!(gate.contains("self.lane_pack.on() && self.spec_policy == SpecLanes::Auto && slots.len() >= 2"),
                "the gate names the flag, the policy and the two lanes");
        assert!(gate.contains("l.mtp.is_some()"),
                "the gate requires an MTP-capable lane (else the plain batch already runs)");
        assert!(gate.contains("if due && !self.arm_fresh(idle, slots.len()) {"),
                "a probe only starts when the idle arm's measured rate is stale");
        let branch = &src[pick..pick + 1600];
        assert!(branch.contains("crate::metrics::lane_pack_round(2); // arm = shared_plain"),
                "a forced-shared probe counts itself as a shared_plain round on /metrics");
        assert!(branch.contains(&["return self.", "step_plain(&slots, true);"].concat()),
                "a forced-shared probe runs the shared arm's own step");
        assert!(branch.contains("probe_force.is_none() && self.spec_policy == SpecLanes::Auto && self.shared_streak % refresh_every == 0"),
                "the 128th-round refresh is suppressed while a probe window is in flight");
    }
}


#[cfg(test)]
mod prefill_interleave_tests {
    use super::*;

    #[test]
    fn the_flag_parses_to_zero_when_absent_and_refuses_garbage() {
        assert_eq!(parse_prefill_interleave(None).unwrap(), 0);
        assert_eq!(parse_prefill_interleave(Some("")).unwrap(), 0);
        assert_eq!(parse_prefill_interleave(Some(" 2 ")).unwrap(), 2);
        assert_eq!(parse_prefill_interleave(Some("0")).unwrap(), 0);
        for bad in ["on", "-1", "1.5"] {
            let e = parse_prefill_interleave(Some(bad)).unwrap_err().to_string();
            assert!(e.starts_with("--prefill-interleave:"), "{e}");
        }
        let a: Vec<String> = ["--prefill-interleave=3", "--port", "9000"].iter().map(|s| s.to_string()).collect();
        assert_eq!(parse_prefill_interleave(arg(&a, "--prefill-interleave")).unwrap(), 3);
    }

    /// Off is silent (default boot logs unchanged); on says so, at TP=1 and (PERFFIX1) under TP, where every
    /// rank runs the same steps at the same chunk boundaries.
    #[test]
    fn the_resolved_value_is_the_flag_at_every_topology_and_the_boot_line_says_so() {
        assert_eq!(resolve_prefill_interleave(0, false, 4095), (0, None));
        assert_eq!(resolve_prefill_interleave(0, true, 4095), (0, None));
        let (n, line) = resolve_prefill_interleave(2, false, 1024);
        assert_eq!(n, 2);
        let line = line.unwrap();
        assert!(line.starts_with("[exl3-serve] prefill interleave ON (--prefill-interleave 2)") && line.contains("chunk 1024 tokens"), "{line}");
        let (n, line) = resolve_prefill_interleave(2, true, 1024);
        assert_eq!(n, 2, "PERFFIX1: every rank runs the flag's value under TP");
        let line = line.unwrap();
        assert!(line.contains("prefill interleave ON under TP (--prefill-interleave 2)") && line.contains("in lockstep"), "{line}");
    }

    /// The hook never runs after the final chunk (the seam step follows it), for a non-cancellable
    /// internal prefill, or with the flag off (at every topology since PERFFIX1).
    #[test]
    fn the_hook_runs_only_between_chunks_of_a_cancellable_prefill() {
        assert!(interleave_point(1, true, 1024, 20_000));
        assert!(interleave_point(4, true, 19_999, 20_000));
        assert!(!interleave_point(1, true, 20_000, 20_000), "final chunk");
        assert!(!interleave_point(0, true, 1024, 20_000), "flag off");
        assert!(!interleave_point(1, false, 1024, 20_000), "WP16 XCHECK reference prefill");
    }

    /// Every served grid ends exactly at `n`, whatever its options (realign, tail split, absorb-tail),
    /// so no hook follows the last chunk and every earlier boundary is a hook point. (A property of
    /// `plan_grid` that the hook predicate relies on; the hook itself is covered by the GPU gate.)
    #[test]
    fn interleave_points_are_the_grid_minus_its_last_chunk() {
        use crate::exl3_forward::plan_grid;
        for (from, n, c, realign, tail, absorb) in [
            (0usize, 20_000usize, 1024usize, false, None, None),
            (0, 20_000, 4095, true, Some(19_900), Some(16)),
            (3_000, 20_000, 1024, true, Some(19_000), Some(16)),
            (0, 900, 1024, false, None, None),
            (0, 4095, 4095, true, None, Some(16)),
        ] {
            let grid = plan_grid(from, n, c, realign, tail, psc_rows(c, false), absorb, true);
            assert_eq!(grid.last().copied(), Some(n), "the grid ends at n");
            let points = grid.iter().filter(|&&e| interleave_point(1, true, e, n)).count();
            assert_eq!(points, grid.len() - 1, "grid {grid:?}");
        }
    }

    /// The registry row: a plain integer flag, no environment variable, default off.
    #[test]
    fn the_flag_is_registered_without_an_environment_variable() {
        let o = crate::opt!("prefill-interleave").def();
        assert_eq!((o.flag, o.env, o.def), ("prefill-interleave", "", "1"), "owner 2026-10-24: default 1");
        // the server's unset fallback is the registry default, not a literal
        let serve: String = include_str!("exl3_serve.rs").lines().filter(|l| !l.trim_start().starts_with("//")).collect::<Vec<_>>().join("\n");
        assert!(serve.contains(".unwrap_or_else(|| crate::opt!(\"prefill-interleave\").def().def.to_string())"), "unset -> registry default");
        assert!(matches!(o.ty, crate::opts::Ty::Int));
        assert_eq!(o.scope, crate::opts::Scope::Spmd);
    }
}

#[cfg(test)]
/// S1 (REL_V0_7_3 review): both spellings of every flag EXL3 reads with `arg()` must work.
/// The table-driven test drives the REAL parser through the REAL `arg()` — before the fix,
/// `--lane-order=fcfs` returned None (silently ignored) while `--print-config` showed it set.
mod s1_eq_aware_arg_tests {
    use super::*;

    const FLAGS: &[&str] = &[
        "--lane-order", "--lane-quantum", "--model-name", "--served-model-name",
        "--otel-endpoint", "--otel-model-id", "--otel-topology",
        "--otel-include-tokens", "--otel-batch-size", "--otel-batch-interval-ms",
        "--port", "--max-seq-len", "--max-batch", "--prefix-cache", "--prefill-chunk", "--prefill-interleave",
        "--kv-cache", "--ple-ram", "--reasoning-effort", "--thinking", "--cpu-affinity",
        "--tune-table", "--tune-draft", "--tune-profile", "--draft-confidence",
        "--draft-temperature", "--spec-sampling", "--host",
    ];

    fn owned(args: &[&str]) -> Vec<String> { args.iter().map(|s| s.to_string()).collect() }

    #[test]
    fn arg_accepts_both_spellings_of_every_flag() {
        for f in FLAGS {
            // space form
            let a = owned(&["serve", "/m", *f, "VALUE", "--after", "x"]);
            assert_eq!(arg(&a, f).as_deref(), Some("VALUE"), "{f} space form");
            // equals form
            let a = owned(&["serve", "/m", &format!("{f}=VALUE"), "--after", "x"]);
            assert_eq!(arg(&a, f).as_deref(), Some("VALUE"), "{f} = form (S1: was None)");
            // absent
            let a = owned(&["serve", "/m", "--other", "y"]);
            assert_eq!(arg(&a, f), None, "{f} absent");
        }
    }

    #[test]
    fn arg_equals_form_does_not_eat_the_next_token() {
        let a = owned(&["--lane-order=fcfs", "--lane-quantum", "4"]);
        assert_eq!(arg(&a, "--lane-order").as_deref(), Some("fcfs"));
        assert_eq!(arg(&a, "--lane-quantum").as_deref(), Some("4"));
        // a longer flag sharing a prefix must not match (the old strip bug class)
        let a = owned(&["--port=9000"]);
        assert_eq!(arg(&a, "--por"), None);
        assert_eq!(arg(&a, "--port").as_deref(), Some("9000"));
    }

    #[test]
    fn nvfp4_parse_arg_is_the_same_parser() {
        // the NVFP4-side reader is a thin delegate of the same shared fn; spot-check both
        // spellings through the EXL3 reader to prove ONE mechanism.
        let a = owned(&["--model-name=Qwen/Qwen3.8-Flash-Next"]);
        assert_eq!(arg(&a, "--model-name").as_deref(), Some("Qwen/Qwen3.8-Flash-Next"));
        let a = owned(&["--model-name", "Qwen/Qwen3.8-Flash-Next"]);
        assert_eq!(arg(&a, "--model-name").as_deref(), Some("Qwen/Qwen3.8-Flash-Next"));
    }
}

/// S-B9-REL-PRPORT3 (review R9): the PR #17 image-payload pre-read, extracted so the
/// admit-order alignment and the error paths are unit-testable. Reads every image-bearing
/// Admit's payload (in admit order) from the control stream BEFORE any admit() runs —
/// prefill multiplexes HeadFlag frames on the same stream, so a payload still unread when
/// the first admit runs could interleave with them (the stream desync / digest mismatch
/// master could hit with 2+ images admitted in one Step). Digest verified per payload;
/// little-endian f16 rows decoded to f32.
pub(crate) fn read_step_image_payloads<R: std::io::Read>(
    r: &mut R,
    step_no: u64,
    specs: &[(u64, u64)],
) -> anyhow::Result<Vec<Option<Vec<f32>>>> {
    use std::io::Read;
    let mut payloads: Vec<Option<Vec<f32>>> = Vec::new();
    for &(ib, idg) in specs {
        if ib > 0 {
            let mut buf = vec![0u8; ib as usize];
            r.read_exact(&mut buf).context("TP node: image payload read")?;
            anyhow::ensure!(crate::tp_serve::fnv64(&buf) == idg,
                            "TP node: image payload digest mismatch (step {step_no})");
            payloads.push(Some(buf.chunks_exact(2)
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect()));
        } else {
            payloads.push(None);
        }
    }
    Ok(payloads)
}

#[cfg(test)]
mod step_image_payload_tests {
    //! S-B9-REL-PRPORT3 (review R9): the admit-order alignment and error paths of the
    //! PR #17 payload pre-read.
    use super::*;

    fn f16bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter()
            .flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes())
            .collect()
    }

    fn spec(vals: &[f32]) -> (u64, u64) {
        let b = f16bytes(vals);
        (b.len() as u64, crate::tp_serve::fnv64(&b))
    }

    #[test]
    fn mixed_admits_stay_aligned_in_admit_order() {
        let a = spec(&[1.5, 2.5]);
        let b = spec(&[-0.5, 4.0]);
        let specs = [a, (0, 0), b]; // image, text, image — the text Admit must not shift rows
        let mut stream = std::io::Cursor::new([f16bytes(&[1.5, 2.5]), f16bytes(&[-0.5, 4.0])].concat());
        let out = read_step_image_payloads(&mut stream, 7, &specs).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].as_ref().unwrap(), &[1.5, 2.5]);
        assert!(out[1].is_none(), "a text Admit stays aligned (None), never shifts the next image");
        assert_eq!(out[2].as_ref().unwrap(), &[-0.5, 4.0]);
    }

    #[test]
    fn short_read_errors() {
        let a = spec(&[1.0, 2.0]);
        let mut stream = std::io::Cursor::new(vec![0u8; 2]); // 2 of 4 bytes
        let err = read_step_image_payloads(&mut stream, 9, &[a]).unwrap_err();
        assert!(format!("{err:#}").contains("image payload read"), "got: {err:#}");
    }

    #[test]
    fn digest_mismatch_errors() {
        let a = spec(&[1.0, 2.0]);
        let mut stream = std::io::Cursor::new(f16bytes(&[9.0, 9.0])); // right length, wrong bytes
        let err = read_step_image_payloads(&mut stream, 9, &[a]).unwrap_err();
        assert!(format!("{err:#}").contains("digest mismatch"), "got: {err:#}");
    }
}

/// S-B9-REL-SMALLFIXES item 1 (owner 2026-10-08): `--mtp off` is honoured by the EXL3 server.
/// Before, `--mtp` was documented in `--help` and ignored here: the banner stayed 'MTP chain-verify
/// ON' and the only working switch was the diagnostics hatch `--exl3-no-mtp`.
#[cfg(test)]
mod mtp_switch_tests {
    use super::*;
    use crate::exl3_forward::{mtp_disabled_by_opt, mtp_off_by_opt, mtp_off_of, MtpOff};
    use crate::opts::{Cat, Scope, Ty};

    /// The registry tests share one process-global store: serialize the ones that write `mtp`.
    static STORE: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn the_decision_function() {
        // not passed / auto / on (and the NVFP4 server's older spellings of on) = speculation ON
        for v in [None, Some("auto"), Some("on"), Some("1"), Some("true")] {
            assert_eq!(mtp_off_of(v, false), None, "--mtp {v:?} must leave the head on");
        }
        // off (and its older spellings) = OFF, attributed to --mtp
        for v in ["off", "0", "false"] {
            assert_eq!(mtp_off_of(Some(v), false), Some(MtpOff::MtpFlag), "--mtp {v}");
        }
        // the hatch keeps working, alone and next to any --mtp value (it names itself)
        for v in [None, Some("auto"), Some("on"), Some("off")] {
            assert_eq!(mtp_off_of(v, true), Some(MtpOff::Hatch), "--exl3-no-mtp with --mtp {v:?}");
        }
        assert_eq!(MtpOff::MtpFlag.flag(), "--mtp off");
        assert_eq!(MtpOff::Hatch.flag(), "--exl3-no-mtp");
    }

    #[test]
    fn mtp_is_a_registered_spmd_option() {
        let id = crate::opts::lookup("mtp").expect("--mtp must be in the registry");
        let o = id.def();
        assert_eq!(o.env, "", "a new option has no environment variable");
        assert_eq!(o.scope, Scope::Spmd, "--mtp must ride TpConfig to the nodes and enter the boot digest");
        assert_eq!(o.cat, Cat::Spec);
        assert_eq!(o.def, "auto");
        match o.ty {
            Ty::Choice(l) => for v in ["auto", "on", "off"] { assert!(l.contains(&v), "--mtp {v}"); },
            other => panic!("--mtp must be a Choice (validated at startup), got {other:?}"),
        }
        assert!(o.help.contains("MTP OFF (--mtp off)"), "the help names the banner: {}", o.help);
        // the hatch stays registered and Spmd
        let h = crate::opts::lookup("exl3-no-mtp").expect("--exl3-no-mtp stays");
        assert_eq!(h.def().scope, Scope::Spmd);
    }

    /// The real parser into the real store, read back by the gate every EXL3 site uses.
    #[test]
    fn the_command_line_reaches_the_gate() {
        let _g = STORE.lock().unwrap_or_else(|e| e.into_inner());
        let (mtp, hatch) = (crate::opt!("mtp"), crate::opt!("exl3-no-mtp"));
        let args = |a: &[&str]| -> Vec<String> { a.iter().map(|s| s.to_string()).collect() };
        let reset = || { crate::opts::test_set(mtp, None); crate::opts::test_set(hatch, None); };
        reset();
        assert!(!mtp_disabled_by_opt(), "a default boot speculates");
        for off in [&["--mtp", "off"][..], &["--mtp=off"][..]] {
            reset();
            crate::opts::parse_cli(&args(off)).unwrap();
            assert_eq!(mtp_off_by_opt(), Some(MtpOff::MtpFlag), "{off:?}");
            assert!(mtp_disabled_by_opt(), "{off:?} must turn the head off");
        }
        for on in [&["--mtp", "on"][..], &["--mtp=auto"][..], &["--mtp", "auto"][..]] {
            reset();
            crate::opts::parse_cli(&args(on)).unwrap();
            assert_eq!(mtp_off_by_opt(), None, "{on:?} is unchanged: the head runs");
        }
        reset();
        crate::opts::parse_cli(&args(&["--exl3-no-mtp"])).unwrap();
        assert_eq!(mtp_off_by_opt(), Some(MtpOff::Hatch));
        // an invalid value is refused by the startup parse, with the accepted spellings in the message
        reset();
        let e = crate::opts::parse_cli(&args(&["--mtp", "maybe"])).unwrap_err();
        assert!(e.contains("--mtp takes auto|on|off") && e.contains("'maybe'"), "{e}");
        assert!(crate::opts::parse_cli(&args(&["--mtp"])).is_err(), "a bare --mtp needs a value");
        // TP: the head's value is what a node installs (and an unset head clears a node's own)
        reset();
        crate::opts::parse_cli(&args(&["--mtp", "off"])).unwrap();
        let shipped = crate::opts::snapshot();
        assert!(shipped.contains(&("mtp".to_string(), "off".to_string())), "--mtp off must be in the TpConfig snapshot: {shipped:?}");
        assert!(crate::opts::spmd_digest().is_some(), "and in the boot agree digest");
        reset();
        let mine: Vec<(String, String)> = shipped.into_iter().filter(|(k, _)| k == "mtp").collect();
        // (install only this row: a full install_head would clear options other tests hold)
        crate::opts::test_set(mtp, mine.first().map(|(_, v)| v.as_str()));
        assert!(mtp_disabled_by_opt(), "a node with the head's --mtp off has its head off");
        reset();
    }

    /// SURFACE2 item 4: with speculation off, no boot line describes speculative rounds (the WP23
    /// DDS line still printed under `--mtp off`), and the head line says the head is not used.
    #[test]
    fn speculation_boot_lines_follow_the_switch() {
        assert!(spec_live(true, None));
        assert!(!spec_live(true, Some(MtpOff::MtpFlag)) && !spec_live(true, Some(MtpOff::Hatch)));
        assert!(!spec_live(false, None), "a pack without a draft head never speculates");
        assert_eq!(head_loaded_line(false, None), None);
        assert_eq!(head_loaded_line(true, None).unwrap(),
                   "[exl3-serve] M1 pack head loaded (MTP draft head for chain-verify serving)", "the default line is unchanged");
        let off = head_loaded_line(true, Some(MtpOff::MtpFlag)).unwrap();
        assert!(off.starts_with("[exl3-serve] M1 pack head loaded") && off.contains("NOT used on this boot: --mtp off"), "{off}");
        assert!(head_loaded_line(true, Some(MtpOff::Hatch)).unwrap().contains("NOT used on this boot: --exl3-no-mtp"));
        // source pin (the boot needs a GPU): the line before each of these prints is its guard
        let src = include_str!("exl3_serve.rs");
        for (line, guard) in [("WP23 DDS: depth ceiling", "if spec_live && draft_conf > 0.0 {"),
                              ("load-adaptive batching (--spec-lanes-max)", "if width > 1 && spec_live {"),
                              ("serial lane order (--lane-order)", "if width > 1 && spec_live {"),
                              ("spec sampling: {} (draft temperature", "&& !crate::exl3_forward::mtp_disabled_by_opt() {")] {
            let print = ["crate::dprintln!(\"[exl3-serve] ", line].concat();
            assert_eq!(src.matches(print.as_str()).count(), 1, "one `{line}` boot line");
            let before = &src[..src.find(print.as_str()).unwrap()];
            let guard_line = before.lines().rev().nth(1).unwrap();
            assert!(guard_line.trim_end().ends_with(guard), "`{line}` must sit behind `{guard}`, found `{}`", guard_line.trim());
        }
    }

    #[test]
    fn the_boot_banner() {
        assert_eq!(mtp_banner(false, None, 5, 0.4), None, "no MTP head in the pack: no line");
        assert_eq!(mtp_banner(false, Some(MtpOff::MtpFlag), 5, 0.4), None);
        let on = mtp_banner(true, None, 5, 0.4).unwrap();
        assert!(on.starts_with("[exl3-serve] MTP chain-verify ON (k=5,") && on.contains("dynamic draft stop at 0.4"), "{on}");
        assert!(on.contains("--mtp off"), "the ON line names the switch that turns it off: {on}");
        let off = mtp_banner(true, Some(MtpOff::MtpFlag), 5, 0.4).unwrap();
        assert!(off.contains("MTP OFF (--mtp off)"), "{off}");
        assert!(!off.contains("chain-verify ON"), "{off}");
        let hatch = mtp_banner(true, Some(MtpOff::Hatch), 5, 0.4).unwrap();
        assert_eq!(hatch, "[exl3-serve] MTP chain-verify OFF (--exl3-no-mtp set — diagnostics hatch)", "unchanged line");
        // both OFF lines keep the prefix the gate scripts match (scripts/i18d_chain.sh MOFF)
        for l in [&off, &hatch] { assert!(l.starts_with("[exl3-serve] MTP chain-verify OFF"), "{l}"); }
    }
}

/// S-B9-REL-SMALLFIXES item 5(a): the scheduler's side of `stop_p` / `stop_rank`.
#[cfg(test)]
mod stop_info_tests {
    use super::*;
    use crate::exl3_forward::{stop_token_stats, SampParams};

    fn row(logits: &[f32]) -> Vec<u16> { logits.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect() }
    fn near(a: f64, b: f64) -> bool { (a - b).abs() < 2e-3 } // f16 logits: three decimals
    fn samp_opt(t: f32, top_p: f32, top_k: usize, min_p: f32) -> Option<SampParams> {
        SampParams::sampled(t, top_p, top_k, min_p, 0, 0)
    }

    /// The S-B9-REL-SMALLFIXES computation, kept verbatim as the reference `stop_token_stats` must
    /// reproduce BIT FOR BIT (SURFACE2 item 2 replaced its full sort and its per-token exp calls).
    fn stop_token_stats_reference(row: &[u16], tok: usize, samp: Option<&SampParams>) -> Option<(f64, u32)> {
        let lg = |b: u16| half::f16::from_bits(b).to_f32() as f64;
        let xt = lg(*row.get(tok)?);
        let (mut lmax, mut rank) = (f64::NEG_INFINITY, 1u32);
        for &b in row {
            let x = lg(b);
            if x > lmax { lmax = x; }
            if x > xt { rank += 1; }
        }
        if !xt.is_finite() || !lmax.is_finite() { return None; }
        let Some(s) = samp else {
            let z: f64 = row.iter().map(|&b| (lg(b) - lmax).exp()).filter(|w| w.is_finite()).sum();
            return Some(((xt - lmax).exp() / z, rank));
        };
        let t = (s.temperature as f64).max(1e-6);
        let w = |x: f64| ((x - lmax) / t).exp();
        let tau = if s.top_k > 0 && (s.top_k as usize) < row.len() {
            let mut xs: Vec<f64> = row.iter().map(|&b| lg(b)).filter(|x| !x.is_nan()).collect();
            let k = (s.top_k as usize - 1).min(xs.len() - 1);
            *xs.select_nth_unstable_by(k, |a, b| b.total_cmp(a)).1
        } else {
            f64::NEG_INFINITY
        };
        let mfloor = if s.min_p > 0.0 { lmax + t * (s.min_p as f64).ln() } else { f64::NEG_INFINITY };
        let mut cand: Vec<f64> = row.iter().map(|&b| lg(b)).filter(|&x| x >= tau && x >= mfloor).collect();
        let mass: f64 = cand.iter().map(|&x| w(x)).sum();
        let mut theta = f64::NEG_INFINITY;
        if s.top_p < 1.0 && mass > 0.0 {
            cand.sort_unstable_by(|a, b| b.total_cmp(a));
            let (target, mut cum, mut i) = (s.top_p as f64 * mass, 0.0f64, 0usize);
            while i < cand.len() {
                let x = cand[i];
                while i < cand.len() && cand[i] == x { cum += w(x); i += 1; }
                theta = x;
                if cum >= target { break; }
            }
        }
        if !(xt >= tau && xt >= mfloor && xt >= theta) { return Some((0.0, rank)); }
        let z: f64 = cand.iter().filter(|&&x| x >= theta).map(|&x| w(x)).sum();
        Some((w(xt) / z, rank))
    }

    #[test]
    fn a_greedy_row_reports_the_models_probability_and_rank() {
        let r = row(&[1.0, 2.0, 3.0, 0.0]);
        let z = (1f64.exp() + 2f64.exp() + 3f64.exp() + 1.0);
        let (p, rank) = stop_token_stats(&r, 2, None).unwrap();
        assert!(near(p, 3f64.exp() / z) && rank == 1, "{p} {rank}");
        let (p, rank) = stop_token_stats(&r, 3, None).unwrap();
        assert!(near(p, 1.0 / z) && rank == 4, "{p} {rank}");
        // ties share the better rank (rank = 1 + the number of STRICTLY greater logits)
        let t = row(&[5.0, 5.0, 1.0]);
        assert_eq!(stop_token_stats(&t, 0, None).unwrap().1, 1);
        assert_eq!(stop_token_stats(&t, 1, None).unwrap().1, 1);
        assert_eq!(stop_token_stats(&t, 2, None).unwrap().1, 3);
        // out of range, or a masked token
        assert_eq!(stop_token_stats(&r, 4, None), None);
        assert_eq!(stop_token_stats(&row(&[1.0, f32::NEG_INFINITY]), 1, None), None);
    }

    #[test]
    fn a_sampled_row_reports_the_probability_in_the_law_the_sampler_draws_from() {
        let samp = |t: f32, top_p: f32, top_k: usize, min_p: f32| SampParams::sampled(t, top_p, top_k, min_p, 7, 0).unwrap();
        // temperature only: softmax(l / T)
        let r = row(&[0.0, 1.0, 2.0]);
        let s = samp(0.5, 1.0, 0, 0.0);
        let z = 0f64.exp() + 2f64.exp() + 4f64.exp();
        let (p, rank) = stop_token_stats(&r, 1, Some(&s)).unwrap();
        assert!(near(p, 2f64.exp() / z) && rank == 2, "{p} {rank}");
        // top-k = 2 over [3, 2, 1, 0]: the third token is truncated away (p = 0), its rank stays 3
        let r = row(&[3.0, 2.0, 1.0, 0.0]);
        let s = samp(1.0, 1.0, 2, 0.0);
        let (p, rank) = stop_token_stats(&r, 1, Some(&s)).unwrap();
        assert!(near(p, 2f64.exp() / (3f64.exp() + 2f64.exp())) && rank == 2, "{p} {rank}");
        assert_eq!(stop_token_stats(&r, 2, Some(&s)).unwrap(), (0.0, 3));
        // top-k keeps every tie of the k-th largest (the kernel's rule)
        let r = row(&[3.0, 2.0, 2.0, 0.0]);
        let (p, _) = stop_token_stats(&r, 2, Some(&s)).unwrap();
        assert!(near(p, 2f64.exp() / (3f64.exp() + 2.0 * 2f64.exp())), "{p}");
        // top-p 0.7 over probabilities .5 / .3 / .2: the prefix .5 + .3 reaches it; .2 is cut
        let r = row(&[0.5f32.ln(), 0.3f32.ln(), 0.2f32.ln()]);
        let s = samp(1.0, 0.7, 0, 0.0);
        let (p, rank) = stop_token_stats(&r, 1, Some(&s)).unwrap();
        assert!(near(p, 0.3 / 0.8) && rank == 2, "{p} {rank}");
        assert_eq!(stop_token_stats(&r, 2, Some(&s)).unwrap(), (0.0, 3));
        // min_p 0.2 over .6 / .3 / .1: the floor is 0.2 * 0.6 = 0.12, so .1 is cut
        let r = row(&[0.6f32.ln(), 0.3f32.ln(), 0.1f32.ln()]);
        let s = samp(1.0, 1.0, 0, 0.2);
        let (p, _) = stop_token_stats(&r, 1, Some(&s)).unwrap();
        assert!(near(p, 0.3 / 0.9), "{p}");
        assert_eq!(stop_token_stats(&r, 2, Some(&s)).unwrap().0, 0.0);
        // every kept probability sums to 1 (top-k 3 + top-p 0.9 + min_p together)
        let r = row(&[2.0, 1.5, 1.0, 0.5, 0.0, -1.0]);
        let s = samp(0.8, 0.9, 4, 0.05);
        let total: f64 = (0..6).map(|t| stop_token_stats(&r, t, Some(&s)).unwrap().0).sum();
        assert!((total - 1.0).abs() < 1e-9, "{total}");
    }

    /// A greedy token is its row's argmax: any other rank means the wrong row was read back.
    #[test]
    fn a_greedy_row_that_does_not_put_its_token_first_is_not_reported() {
        assert_eq!(stop_info_checked(Some((0.9, 1)), true), Some((0.9, 1)));
        assert_eq!(stop_info_checked(Some((0.01, 7)), true), None, "greedy + rank 7 = not the row that chose the token");
        assert_eq!(stop_info_checked(Some((0.01, 7)), false), Some((0.01, 7)), "a sample can have any rank");
        assert_eq!(stop_info_checked(Some((0.0, 40)), false), Some((0.0, 40)));
        assert_eq!(stop_info_checked(None, true), None);
        assert_eq!(stop_info_checked(Some((f64::NAN, 1)), false), None);
    }

    /// The readback is on the finishing step only: the two call sites sit behind `reason == "stop"`
    /// inside the finish branch, and nothing else in the scheduler calls it (source pin: the unit
    /// suite cannot run a step).
    #[test]
    fn the_readback_runs_only_when_a_request_finishes_on_a_stop_token() {
        let src = include_str!("exl3_serve.rs");
        // S-B9-CF-PACK4: the call gained a `packed` argument and step_mtp's emit loop moved into round_emit
        let call = format!("self.send_{}(s, ", "stop_info");
        assert_eq!(src.matches(call.as_str()).count(), 2, "round_emit (step_mtp's emit loop, packed or not) and step_plain");
        // PACK4-REVIEW B4: the ROW argument is pinned again (the pin had been loosened to the call's prefix): a
        // packed round reads the stop token's row at the lane's row offset + its index, a serial round row j
        let (packed_call, plain_call) = (format!("self.send_{}(s, packed_row0.map_or(j, |r0| r0 + j), t, packed_row0.is_some())", "stop_info"),
                                         format!("self.send_{}(s, j, t, false)", "stop_info"));
        assert_eq!(src.matches(packed_call.as_str()).count(), 1, "round_emit reads the packed row r0 + j");
        assert_eq!(src.matches(plain_call.as_str()).count(), 1, "step_plain reads row j of the shared step");
        for (i, _) in src.match_indices(call.as_str()) {
            let line_start = src[..i].rfind('\n').unwrap() + 1;
            let line = &src[line_start..i];
            assert!(line.contains("reason == \"stop\""), "a call not guarded by the stop reason: `{}`", line.trim());
        }
        let rb = format!(".logits_{}(", "row");
        assert_eq!(src.matches(rb.as_str()).count(), 1, "one readback site in the scheduler (send_stop_info)");
        let f = &src[src.find(format!("fn send_{}(&self", "stop_info").as_str()).expect("send_stop_info")..];
        assert!(f.find("if self.tp.is_some() { return; }").unwrap() < f.find(rb.as_str()).unwrap(), "no readback under TP");
    }

    /// A vocabulary-sized logits row shaped like a served one: a bulk around `mean` with spread
    /// `sd` (f16-rounded, so many exact ties), a few dozen plausible tokens above it and one clear
    /// winner at index 7. `peaked` false = a flat row (a wide nucleus: the top-p loop runs long).
    fn synth_row(seed: u64, n: usize, mean: f32, sd: f32, peaked: bool) -> Vec<u16> {
        let mut x = seed | 1;
        let mut u = move || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x >> 11) as f64 / (1u64 << 53) as f64 };
        let mut v: Vec<f32> = (0..n).map(|_| {
            let g = (-2.0 * u().max(1e-300).ln()).sqrt() * (std::f64::consts::TAU * u()).cos(); // Box-Muller
            mean + sd * g as f32
        }).collect();
        if peaked {
            for j in 0..48 { v[(j * 5003 + 11) % n] = 6.0 + 9.0 * (u() as f32); }
            v[7] = 21.5;
            v[9] = 19.0;
        }
        v.iter().map(|&f| half::f16::from_f32(f).to_bits()).collect()
    }

    /// SURFACE2 item 2: `stop_token_stats` no longer sorts the vocabulary nor takes one exp per
    /// token; (p, rank) must equal the first implementation's BIT FOR BIT, on served-shape rows
    /// and on the awkward ones (heavy ties, both zeros, infinities, NaN, masked rows, tiny rows).
    #[test]
    fn stop_token_stats_is_bit_identical_to_the_reference() {
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let mut rnd = move || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x };
        let bits = |r: Option<(f64, u32)>| r.map(|(p, k)| (p.to_bits(), k));
        let mut rows: Vec<Vec<u16>> = vec![
            synth_row(3, 248_320, -1.5, 2.6, true),
            synth_row(4, 248_320, 0.0, 3.0, false),
            synth_row(5, 40_000, 0.0, 0.02, false),                   // nearly every logit a tie
            row(&[0.0, -0.0, 0.0, -0.0, -1.0, -0.0]),                 // the maximum is a zero of either sign
            row(&[-0.0, 0.0, -3.0]),
            row(&[2.0, f32::NEG_INFINITY, 2.0, 1.0, f32::NEG_INFINITY, f32::NAN, 0.5]),
            row(&[f32::NAN, f32::NAN]),
            row(&[f32::NEG_INFINITY, f32::NEG_INFINITY]),
            row(&[1.0, f32::INFINITY]),
            row(&[3.25]),
            row(&[]),
        ];
        for i in 0..40 { // small rows from a small alphabet: ties, zeros and specials everywhere
            let n = 1 + (rnd() % 60) as usize;
            let vals = [0.0f32, -0.0, 1.0, 1.5, -2.0, 7.25, -7.25, 1e-3, f32::NEG_INFINITY, f32::NAN, 30.0, 0.5];
            let lim = if i % 4 == 0 { vals.len() } else { 8 };
            rows.push((0..n).map(|_| half::f16::from_f32(vals[(rnd() % lim as u64) as usize]).to_bits()).collect());
        }
        for _ in 0..3 { rows.push((0..3000).map(|_| (rnd() >> 13) as u16).collect()); } // raw f16 patterns
        let samplers: Vec<Option<SampParams>> = {
            let mut v = vec![None];
            // (an infinite temperature is not refused by the API: with a -inf logit the weight is NaN)
            for &t in &[1.0f32, 0.7, 0.05, 2.5, 1e-9, f32::INFINITY, 3e38] {
                for &top_p in &[1.0f32, 0.95, 0.8, 0.5, 1e-6, 0.999_999] {
                    for &top_k in &[0usize, 1, 2, 20, 64, 5000, usize::MAX] {
                        for &min_p in &[0.0f32, 0.05, 0.5, 1.0, 1.5] {
                            v.push(Some(SampParams { temperature: t, top_p, top_k: top_k.min(u32::MAX as usize) as u32,
                                                     min_p, seed: 0, counter: 0 }));
                        }
                    }
                }
            }
            v
        };
        let mut checked = 0usize;
        for (ri, r) in rows.iter().enumerate() {
            // the top token, a few random ones, the last one, and one past the end
            let top = (0..r.len()).max_by(|&a, &b| half::f16::from_bits(r[a]).to_f32().total_cmp(&half::f16::from_bits(r[b]).to_f32())).unwrap_or(0);
            let mut toks = vec![top, 0, r.len().saturating_sub(1), r.len()];
            for _ in 0..3 { toks.push((rnd() % r.len().max(1) as u64) as usize); }
            // every row takes a slice of the sampler grid, offset by the row index so the rows
            // cover all of it between them (the reference sorts 248K values per call on the big
            // rows; the new function's histogram is a fixed ~70 us per call on the small ones)
            let stride = if r.len() > 10_000 { 11 } else { 9 };
            for (si, sp) in samplers.iter().enumerate().filter(|(si, _)| (si + ri) % stride == 0) {
                for &tok in if r.len() > 10_000 { &toks[..2] } else { &toks[..] } {
                    let (got, want) = (stop_token_stats(r, tok, sp.as_ref()), stop_token_stats_reference(r, tok, sp.as_ref()));
                    assert_eq!(bits(got), bits(want), "row {ri} (len {}), sampler {si} {sp:?}, tok {tok}: {got:?} vs {want:?}", r.len());
                    checked += 1;
                }
            }
        }
        assert!(checked > 40_000, "{checked} comparisons");
    }

    /// SURFACE2 item 2 instrument (run by hand: `cargo test --release --lib stop_token_stats_cost --
    /// --ignored --nocapture`): the host pass of a finishing step, per sampler setting, on a
    /// vocabulary-sized row. Prints min / median of 15 runs in ms.
    #[test]
    #[ignore]
    fn stop_token_stats_cost() {
        const V: usize = 248_320;
        let time = |f: &dyn Fn() -> Option<(f64, u32)>| -> (f64, f64, Option<(f64, u32)>) {
            let mut ts: Vec<f64> = (0..15).map(|_| { let t = std::time::Instant::now(); std::hint::black_box(f()); t.elapsed().as_secs_f64() * 1e3 }).collect();
            ts.sort_by(|a, b| a.total_cmp(b));
            (ts[0], ts[7], f())
        };
        for (shape, row) in [("peaked", synth_row(1, V, -1.5, 2.6, true)), ("flat", synth_row(2, V, 0.0, 3.0, false))] {
            let distinct = row.iter().collect::<std::collections::HashSet<_>>().len();
            crate::dprintln!("row {shape}: {V} logits, {distinct} distinct f16 values");
            let cases: [(&str, Option<SampParams>, usize); 7] = [
                ("greedy (softmax at T=1)", None, 7),
                ("T=0.7 top_k=20 top_p=0.8", samp_opt(0.7, 0.8, 20, 0.0), 7),
                ("T=0.7 top_k=20 top_p=1.0", samp_opt(0.7, 1.0, 20, 0.0), 7),
                ("T=1.0 top_k=0  top_p=1.0", samp_opt(1.0, 1.0, 0, 0.0), 7),
                ("T=0.7 top_k=0  top_p=0.95", samp_opt(0.7, 0.95, 0, 0.0), 7),
                ("T=1.0 top_k=0  top_p=0.95", samp_opt(1.0, 0.95, 0, 0.0), 9),
                ("T=1.0 top_k=0  top_p=0.95 min_p=0.05", samp_opt(1.0, 0.95, 0, 0.05), 7),
            ];
            for (name, sp, tok) in cases {
                let (new_min, new_med, got) = time(&|| stop_token_stats(&row, tok, sp.as_ref()));
                let (old_min, old_med, want) = time(&|| stop_token_stats_reference(&row, tok, sp.as_ref()));
                assert_eq!(got.map(|(p, r)| (p.to_bits(), r)), want.map(|(p, r)| (p.to_bits(), r)), "{name}: not bit-exact");
                crate::dprintln!("  {name:<38} reference min {old_min:7.3} med {old_med:7.3} ms | now min {new_min:7.3} med {new_med:7.3} ms | (p, rank) = {got:?}");
            }
        }
    }
}

/// S-B9-CF-PACK4: `--lane-pack` wiring that no unit can run (the suite never builds a Scratch,
/// AGENTS 1a-ter): source pins, so a later edit cannot silently drop a refusal or the gate.
#[cfg(test)]
mod lane_pack_wiring_tests {
    #[test]
    fn tp_is_refused_before_the_attach_on_both_ranks_and_in_build_serve() {
        let src = include_str!("exl3_serve.rs");
        let call = format!("lane_pack::{}(", "tp_refusal");
        assert_eq!(src.matches(call.as_str()).count(), 3, "build_serve (any TP rank) + the head and node entry points");
        // the head and node entry points refuse BEFORE tp_attach, build_serve before the model load
        for entry in ["pub fn run_tp_head_serve(", "pub fn run_tp_node_serve("] {
            let f = &src[src.find(entry).unwrap_or_else(|| panic!("{entry}"))..];
            assert!(f.find(call.as_str()).unwrap() < f.find("tp_attach(ctx)").unwrap(), "{entry} refuses before the attach");
        }
        let b = &src[src.find("fn build_serve(").unwrap()..];
        assert!(b.find(call.as_str()).unwrap() < b.find("let model = ").expect("the model-load anchor of build_serve was renamed: update this pin"),
                "build_serve refuses before the load");
        assert!(src.contains("--lane-pack on needs --lane-order rr"), "fcfs steps one lane, a pack steps two");
    }

    /// PACK4-REVIEW A2: the reasoned pin. `pack_finish` re-primes eagerly; `head_reprime` takes
    /// `rp.pending` FIRST and, if a lone round left one behind (it failed between its
    /// `reprime_dev_enqueue` and its `head_reprime`), would run `reprime_dev_finish` on the wrong
    /// round and fail every live lane. The packed path therefore clears the hand-off first.
    #[test]
    fn a_packed_round_never_takes_a_stale_reprime_handoff() {
        let fwd = include_str!("exl3_forward.rs");
        let pf = &fwd[fwd.find("pub fn pack_finish(&self").unwrap()..];
        let pf = &pf[..pf.find("\n    }\n").unwrap()];
        assert!(pf.find("rp.pending = None;").expect("the stale hand-off is cleared")
                < pf.find("self.head_reprime(sc, head, d.slot").unwrap(), "cleared BEFORE head_reprime can take it");
        // the lone path is untouched: its enqueue still resets the hand-off itself, and head_reprime
        // still consumes it with take()
        let re = &fwd[fwd.find("fn reprime_dev_enqueue(&self").unwrap()..];
        assert!(re[..re.find("\n    }\n").unwrap()].contains("rp.pending = None;"));
        assert!(fwd.contains("sc.rp.as_mut().and_then(|r| r.pending.take())"));
    }

    /// PACK4-REVIEW A3: an error out of the packed part of a pair round switches the arm off (the
    /// planner then returns None: master's loops), loudly and visibly; an error of the control
    /// cadence's serial fallback does not.
    #[test]
    fn a_packed_round_error_turns_the_arm_off_for_the_boot() {
        let src = include_str!("exl3_serve.rs");
        let sk = &src[src.find("fn step_packed(&mut self").unwrap()..];
        let sk = &sk[..sk.find("\n    }\n").unwrap()];
        assert!(sk.contains("std::mem::take(&mut self.pack_stats.in_round)") && sk.contains("self.pack_disable(a, b, &e)"));
        let sp = &src[src.find("fn step_pair(&mut self").unwrap()..];
        let sp = &sp[..sp.find("\n    }\n").unwrap()];
        // the fallback's serial rounds run BEFORE the flag is raised; the packed part raises it and
        // lowers it on completion
        let fallback = sp.find(["return self.", "step_mtp(sb, false);"].concat().as_str()).expect("the control-cadence fallback");
        let raise = sp.find("self.pack_stats.in_round = true;").expect("flag raised");
        let lower = sp.find("self.pack_stats.in_round = false;").expect("flag lowered");
        assert!(fallback < raise && raise < lower);
        let pd = &src[src.find("fn pack_disable(&mut self").unwrap()..];
        let pd = &pd[..pd.find("\n    }\n").unwrap()];
        assert!(pd.contains("self.lane_pack = crate::lane_pack::LanePack::Off;") && pd.contains("crate::metrics::lane_pack_disabled();"));
        assert!(src[src.find("fn pack_plan(&mut self").unwrap()..].contains("if !self.lane_pack.on() || slots.len() < 2 { return None; }"),
                "the planner returns None once the mode is Off: master's loops");
    }

    /// PACK4-REVIEW A4: the planner and the pair round both use the capped depth (one definition:
    /// `lane_pack::PACK_DEPTH_MAX`), so the forward path never meets a pair `pack_rows` refuses.
    #[test]
    fn the_planner_and_the_pair_round_cap_the_depth_alike() {
        let src = include_str!("exl3_serve.rs");
        let pp = &src[src.find("fn pack_plan(&mut self").unwrap()..];
        let pp = &pp[..pp.find("\n    }\n").unwrap()];
        // the per-lane rule is the pure `lane_view` (PACK4-REVIEW B4), which applies the cap and the clamp flag
        assert!(pp.contains("crate::lane_pack::lane_view(s, facts, l.pos, k_pre, kd_reg, self.in_prefill, self.pack_ready"));
        let lp = include_str!("lane_pack.rs");
        let lv = &lp[lp.find("pub(crate) fn lane_view(").unwrap()..];
        let lv = &lv[..lv.find("\n}\n").unwrap()];
        assert!(lv.contains("kd_reg.min(PACK_DEPTH_MAX)") && lv.contains("clamped: kd_reg < k_pre"));
        let sp = &src[src.find("fn step_pair(&mut self").unwrap()..];
        let sp = &sp[..sp.find("\n    }\n").unwrap()];
        assert!(sp.contains("kd.min(crate::lane_pack::PACK_DEPTH_MAX)"));
        // PACK4-REVIEW B4: step_pair drafts at the round's CLAMPED depth (round_depth's regime-clamped value), never at mtp_k
        assert!(sp.contains("let (_, kd) = self.round_depth(l.pos, l.max_new.saturating_sub(l.generated));"));
        assert!(!sp.contains("self.mtp_k.min("), "the pair round must not recompute a depth of its own");
    }

    /// PACK4-REVIEW B2: the packed precapture decides nothing from a free-memory delta; a capture
    /// that did not land still fails the boot, per graph.
    #[test]
    fn the_packed_precapture_has_no_memory_delta_cap() {
        let fwd = include_str!("exl3_forward.rs");
        let pc = &fwd[fwd.find("pub fn pack_precapture(&self").unwrap()..];
        let pc = &pc[..pc.find("\n    }\n").unwrap()];
        assert!(!pc.contains("1usize << 30") && !pc.contains("precapture cap"), "no cuMemGetInfo-delta cap");
        assert!(pc.contains("did not capture (see the EXL3_GRAPH line above)"), "a failed capture still fails the boot");
    }

    /// S-B9-CF-PACK5: the packed round's lockstep points, in source order. After BOTH lanes drafted:
    /// the wasted-pass drain, then the ONE pre-verify exchange (before `pack_verify` launches the
    /// R-keyed graph); after the finishes: the post-verify `tp_round` whose head ms / capture flag
    /// feed `round_account` (never the local clock). The pack is built from lane state only.
    #[test]
    fn the_packed_round_runs_its_lockstep_points_in_order() {
        let src = include_str!("exl3_serve.rs");
        let sp = &src[src.find("fn step_pair(&mut self").unwrap()..];
        let sp = &sp[..sp.find("\n    }\n").unwrap()];
        let at = |pat: &str| sp.find(pat).unwrap_or_else(|| panic!("missing in step_pair: {pat}"));
        let drafts_done = at("drafts.push(d);"); // the end of the per-lane draft loop body: BOTH lanes have drafted after it
        let drain = at("self.model.pack_drain_wasted(&drafts)?;");
        let pre = at("t.pre_verify_pack(r, segs[0].drafts.len() + 1, &words, launched)?;");
        let verify = at("self.model.pack_verify(&mut self.sc, scp, &drafts[0], &drafts[1])?");
        let fin = at("self.model.pack_finish(&mut self.sc, head, &drafts[j], &outs[j])?;");
        let post = at("self.tp_round(\"pack-round\", outs[0].a + outs[1].a, r, &w, ms_local, 0.0, captured_local, true)?");
        let acct = at("self.round_account(s, &outs[j], ms_lane, 0.0, captured,");
        assert!(drafts_done < drain && drain < pre && pre < verify, "both drafts, then the drain, the pre-verify, then the verify");
        assert!(verify < fin && fin < post && post < acct, "verify, finishes, post-verify agree, then the bookkeeping");
        assert!(sp.contains("let ms_lane = lane_share_ms(ms_round, 2);"), "the lane's share of the AGREED (head) round time");
        assert!(sp.contains("if self.tp.is_some() && tp_preverify_on() {"), "the pre-verify honours --exl3-tp-preverify=0 like the lone round");
        // nothing in the pack is decided from rank-local clocks or graph presence: the drafts loop reads
        // lane state and the pre list, the dds target comes from lane state + the head-agreed EMAs
        assert!(!sp.contains("lane_share_ms(ms_of("), "a local clock must not be the lane's cost");
        // the arm goes off after a packed-round error (PACK4-REVIEW A3), and a lone round's stale
        // device re-prime hand-off cannot fail a pair (A2)
        let sk = &src[src.find("fn step_packed(&mut self").unwrap()..];
        let sk = &sk[..sk.find("\n    }\n").unwrap()];
        assert!(sk.contains("self.pack_disable(a, b, &e)"));
        let fwd = include_str!("exl3_forward.rs");
        let pf = &fwd[fwd.find("pub fn pack_finish(&self").unwrap()..];
        let pf = &pf[..pf.find("\n    }\n").unwrap()];
        assert!(pf.find("rp.pending = None;").unwrap() < pf.find("self.head_reprime(sc, head, d.slot").unwrap(),
                "pack_finish clears a stale re-prime hand-off before the eager re-prime");
    }

    #[test]
    fn the_packed_arm_sits_behind_the_flag_and_in_front_of_the_serial_loops() {
        let src = include_str!("exl3_serve.rs");
        // the arms live in `step_rounds`; `step` is the wrapper that publishes after any of them
        let st = &src[src.find("fn step_rounds(&mut self) -> Result<()>").unwrap()..];
        let st = &st[..st.find("\n    }\n").unwrap()];
        let (shared, pack, fcfs, serial) = (st.find(["return self.", "step_plain(&slots, true);"].concat().as_str()).unwrap(),
            st.find("self.pack_plan(&slots)").expect("the packed arm"), st.find("self.lane_order == LaneOrder::Fcfs").unwrap(),
            st.find("let mut plain: Vec<usize> = Vec::new();\n        for &s in &slots {").unwrap());
        assert!(shared < pack && pack < fcfs && fcfs < serial, "shared-plain, then the packed arm, then the serial loops");
        let pp = &src[src.find("fn pack_plan(&mut self").unwrap()..];
        let pp = &pp[..pp.find("\n    }\n").unwrap()];
        assert!(pp.contains("if !self.lane_pack.on() || slots.len() < 2 { return None; }"), "off or a lone request: master's path");
        // the boot precaptures the packed family (never while serving) and refuses a half-built one
        assert!(src.contains("model.pack_precapture(&mut scp, &fams)?"), "the families (greedy + real-q / match) are captured at boot");
        assert!(src.contains("crate::lane_pack::fams(spec_ratio, sampled_ok)"), "one list of families for capture, readiness and the boot agree");
        assert!(src.contains("the dense packed graph family is incomplete"));
    }

    /// S-B9-CF-PACK4S: the sampled-lane glue, pinned in source order. Per lane, inside ONE loop
    /// body: the lane's own sampler staging on the served scratch (the lone round's two calls),
    /// then its draft chain, then the hand-off of its real-q lists — so the first lane's lists
    /// leave the served scratch before the second lane's chain overwrites them.
    #[test]
    fn a_pair_stages_drafts_and_hands_off_per_lane_in_order() {
        let src = include_str!("exl3_serve.rs");
        let src = &src[..src.find("mod lane_pack_wiring_tests").unwrap()]; // the code, not these pins
        let sp = &src[src.find("fn step_pair(&mut self").unwrap()..];
        let sp = &sp[..sp.find("\n    }\n").unwrap()];
        let body = &sp[sp.find("for &(s, b, p, tap_row, kd, _) in &pre {").expect("the per-lane draft loop")..];
        let body = &body[..body.find("let t1 = std::time::Instant::now();").unwrap()];
        let at = |pat: &str| body.find(pat).unwrap_or_else(|| panic!("missing in the draft loop: {pat}"));
        let (rows, stage, pen) = (at("samp_rows(k + 1), self.rq_stage(s)"), at("self.model.set_spec_sampling(&mut self.sc, &rows, rq)?"),
                                  at("self.model.set_penalties(&mut self.sc, &vec![None; k + 1])?"));
        let (dds, fixed, hand, push) = (at("Some(&mut keep), seg, rq)?"), at("tap_row, None, seg, rq)?"),
                                        at("self.model.pack_list_handoff(&self.sc, scp, at, &d)?"), at("drafts.push(d);"));
        assert!(rows < stage && stage < pen && pen < dds && dds < fixed && fixed < hand && hand < push,
                "stage -> draft -> list hand-off, per lane");
        assert!(at("at += d.drafts.len() + 1;") > hand, "the next segment's row offset follows this lane's rows");
        // the lane's real-q flag reaches the accounting (ctr += k + 1, dctr += k, as in the lone round)
        assert!(sp.contains("captured, false, false, drafts[j].samp.ratio, xcheck, true)?"));
        // the all-greedy staging of PACK4 is gone: each lane stages its own
        assert!(!sp.contains("set_spec_sampling(&mut self.sc, &vec![None; k + 1], None)"));
        // the fixed-depth chain is allowed under lane-pack (the deterministic gate posture)
        assert!(!src.contains(&format!("{} on needs the dynamic draft stop", "--lane-pack")));
        // a control round never runs with --a-ctrl off, in the lone round and in the pair's check
        assert!(src.contains("let control = (af.ctrl != 3 && self.ctrl_round % period == 0) || force_ctrl;"));
        assert!(sp.contains("if af.ctrl != 3 && pair_owes_control(self.ctrl_round, per_a, per_b) {"));
    }

    /// S-B9-CF-PACK4S: the forward side. The real-q list follows the scratch width (16 rows for
    /// the packed scratch); the packed verify stages its sampler block BEFORE it picks the graph
    /// (the family is the staged pair's); the precapture leaves the scratch greedy.
    #[test]
    fn the_packed_verify_stages_per_row_samplers_and_picks_the_family() {
        let fwd = include_str!("exl3_forward.rs");
        assert_eq!(fwd.matches("rq_list: dev.htod_sync_copy(&vec![u32::MAX; crate::lane_pack::rq_list_rows(w) * crate::wp24::RQ_ROW])?").count(), 1,
                   "the list is sized from the scratch width");
        assert!(!fwd.contains("vec![u32::MAX; (MTP_MAX_K + 1) * crate::wp24::RQ_ROW]"), "the 8-row list is gone");
        let pv = &fwd[fwd.find("pub fn pack_verify(&self").unwrap()..];
        let pv = &pv[..pv.find("\n    }\n").unwrap()];
        let at = |pat: &str| pv.find(pat).unwrap_or_else(|| panic!("missing in pack_verify: {pat}"));
        let (samp, set, key, launch) = (at("crate::lane_pack::pack_samp(&meta.segs, &[d0.samp, d1.samp], scp.samp.len() / 8)?"),
                                        at("self.set_pack_sampling(scp, &ps)?"),
                                        at("pack_key(meta.r, q0, qb, scp.rq_live) | self.vp_key_st(scp.samp_live, false, scp.rq_live, meta.r) | sk(scp)"),
                                        at(".launch();"));
        assert!(samp < set && set < key && key < launch);
        let pc = &fwd[fwd.find("pub fn pack_precapture(&self").unwrap()..];
        let pc = &pc[..pc.find("\n    }\n").unwrap()];
        // the family's sampler shadows are set BEFORE the capture (the verify tail is chosen from them)
        // and restored to greedy after the last one; the graph key is the family's own (vp bits included)
        assert!(pc.find("scp.rq_live = rq;").unwrap() < pc.find("self.capture_verify_graph_packed(").unwrap());
        assert!(pc.find("scp.samp_live = samp;").unwrap() < pc.find("self.capture_verify_graph_packed(").unwrap());
        assert!(pc.find("self.capture_verify_graph_packed(").unwrap() < pc.find("scp.rq_live = false;").unwrap());
        assert!(pc.find("self.capture_verify_graph_packed(").unwrap() < pc.find("scp.samp_live = false;").unwrap());
        assert!(pc.contains("self.pack_fam_key(scp, r, sparse, qb, fam)"), "one key function for capture, readiness and replay");
        let fk = &fwd[fwd.find("fn pack_fam_key(&self").unwrap()..];
        let fk = &fk[..fk.find("\n    }\n").unwrap()];
        assert!(fk.contains("self.vp_key_st(samp, false, rq, r)"), "the vp tail bits are part of the family key");
        // the hand-off is a D2D copy on the COMPUTE stream (AGENTS 2.3), never a sync copy
        let ho = &fwd[fwd.find("pub fn pack_list_handoff(&self").unwrap()..];
        let ho = &ho[..ho.find("\n    }\n").unwrap()];
        assert!(ho.contains("cuMemcpyDtoDAsync_v2(dst, src, words * 4, self.stream.stream)") && !ho.contains("dtod_sync"));
    }
}

/// S-B9-REL-SMALLFIXES 9a(iii) + 9a(iv): the scheduler side of the per-row attention regime.
#[cfg(test)]
mod regime_sched_tests {
    use super::*;

    const LIM: usize = 2051; // Flash-Next: qsa_limit()

    /// Issue #24: what one lane reports to /metrics in each of its states.
    #[test]
    fn kv_lane_occupancy_states() {
        // a running request: its position is used, and its own prefix snapshot is not "cached"
        assert_eq!(kv_lane_occupancy(Some(1200), None, Some(900)), (1200, 0));
        // a prefill in flight (the lane does not exist yet): the rows written so far
        assert_eq!(kv_lane_occupancy(None, Some(4095), None), (4095, 0));
        assert_eq!(kv_lane_occupancy(None, Some(0), None), (0, 0));
        // a free lane: nothing used; the snapshot it keeps for the prefix cache
        assert_eq!(kv_lane_occupancy(None, None, Some(900)), (0, 900));
        assert_eq!(kv_lane_occupancy(None, None, None), (0, 0));
        // the lane wins over a stale prefill marker (admit clears it; a nested interleave step
        // publishes while it is still set for ANOTHER slot, which the caller filters by slot)
        assert_eq!(kv_lane_occupancy(Some(7), Some(3), None), (7, 0));
    }

    /// Issue #24 wiring: every scheduler function that frees or fills a lane ends in a publish, and
    /// the age gauge is written by the `step` wrapper (every arm), not by one arm of the rounds.
    #[test]
    fn kv_publish_points_are_wired() {
        let src = include_str!("exl3_serve.rs");
        let body = |name: &str| -> &str {
            let at = src.find(name).unwrap_or_else(|| panic!("{name} not found"));
            let end = src[at + name.len()..].find("\n    fn ").map_or(src.len(), |e| at + name.len() + e);
            &src[at..end]
        };
        let publish = ["self.", "publish_kv()"].concat();
        let touch = ["crate::metrics::", "sched_touch()"].concat();
        for f in ["fn step(&mut self) -> Result<()> {", "fn admit(&mut self, req: BatchRequest) {", "fn die(mut self,",
                  "fn tp_fail(&mut self,", "fn prefill_range(&mut self,"] {
            assert!(body(f).contains(&publish), "{f} must publish the KV gauges");
        }
        assert!(body("fn step(&mut self) -> Result<()> {").contains(&touch), "the step wrapper writes the age gauge");
        assert!(!body("fn step_rounds(&mut self) -> Result<()> {").contains(&touch),
                "step_rounds must not write the age gauge itself (the fcfs and shared-plain arms return early)");
        // a request's last round: its final position is published before the lane is dropped
        // (the post-step publication sees a free lane), in the speculative and the plain step
        assert_eq!(src.matches(&[publish.as_str(), "; // issue #24: the lane's final position"].concat()).count(), 1);
        assert_eq!(src.matches(&["if finish { ", publish.as_str(), "; } // issue #24: the lane's final position"].concat()).count(), 1);
        // both scheduler loops publish the idle value before they block
        let idle = ["crate::metrics::", "sched_idle()"].concat();
        let n_idle = src.matches(&idle).count();
        let paired = src.matches(&[publish.as_str(), "; // issue #24: the idle value"].concat()).count();
        assert!(n_idle >= 2 && paired == n_idle, "every sched_idle() site publishes first ({paired} of {n_idle})");
    }

    /// 9a(iv): step_plain's order — the lanes at/below the switch first, stable in each class.
    #[test]
    fn step_plain_orders_short_lanes_first() {
        // slot -> the lane's next position
        let pos = |s: usize| [100usize, 5000, 2051, 2052, 7, 30000][s];
        // slots 0, 2, 4 are at/below the switch (2051 itself is dense); 1, 3, 5 are past it
        assert_eq!(regime_ordered_slots(&[0, 1, 2, 3, 4, 5], pos, LIM), vec![0, 2, 4, 1, 3, 5]);
        // a subset, given in another order: each class keeps the GIVEN order (a stable partition)
        assert_eq!(regime_ordered_slots(&[5, 4, 3, 2], pos, LIM), vec![4, 2, 5, 3]);
        // one-sided steps are untouched
        assert_eq!(regime_ordered_slots(&[4, 0, 2], pos, LIM), vec![4, 0, 2]);
        assert_eq!(regime_ordered_slots(&[5, 1, 3], pos, LIM), vec![5, 1, 3]);
        assert_eq!(regime_ordered_slots(&[1], pos, LIM), vec![1]);
        assert_eq!(regime_ordered_slots(&[], pos, LIM), Vec::<usize>::new());
        // the result is always a permutation of the input
        let mut o = regime_ordered_slots(&[3, 0, 5, 2, 1, 4], pos, LIM);
        o.sort();
        assert_eq!(o, vec![0, 1, 2, 3, 4, 5]);
        // the dense rows lead, so forward_step's dense prefix covers every short lane of the step
        let ord = regime_ordered_slots(&[1, 0, 3, 2, 5, 4], pos, LIM);
        let p: Vec<usize> = ord.iter().map(|&s| pos(s)).collect();
        assert_eq!(crate::exl3_forward::dense_prefix(&p, LIM), (3, 0), "3 leading dense rows, none behind a long row");
    }

    /// 9a(iv): with the flag off (or no sparse attention) the order is the identity.
    #[test]
    fn the_flag_off_order_is_the_identity() {
        assert_eq!(regime_switch(true, true, LIM), LIM);
        assert_eq!(regime_switch(true, false, LIM), usize::MAX, "--qsa-row-regime 0");
        assert_eq!(regime_switch(false, true, LIM), usize::MAX, "no indexer / --qsa-exl3 0");
        assert_eq!(regime_switch(false, false, LIM), usize::MAX);
        let pos = |s: usize| [100usize, 5000, 2051, 2052, 7, 30000][s];
        for slots in [vec![0, 1, 2, 3, 4, 5], vec![5, 4, 3, 2, 1, 0], vec![1, 0, 3], vec![]] {
            assert_eq!(regime_ordered_slots(&slots, pos, regime_switch(true, false, LIM)), slots, "flag off: {slots:?}");
            assert_eq!(regime_ordered_slots(&slots, pos, regime_switch(false, true, LIM)), slots, "no switch: {slots:?}");
        }
    }

    /// The method the scheduler calls is exactly these two functions (source pin).
    #[test]
    fn regime_ordered_is_the_two_pure_functions() {
        let src = include_str!("exl3_serve.rs");
        let f = &src[src.find("fn regime_ordered(&self, slots: &[usize]) -> Vec<usize> {").expect("regime_ordered")..];
        let f = &f[..f.find("\n    }\n").unwrap()];
        assert!(f.contains("regime_switch(self.model.qsa_live(usize::MAX), crate::exl3_forward::row_regime_on(), self.model.qsa_limit())"));
        assert!(f.contains("regime_ordered_slots(slots, |s| self.lanes[s].as_ref().map_or(0, |l| l.pos), lim)"));
    }

    /// 9a(iii): the count of mixed step shapes, and the idle line that reports the graphs.
    #[test]
    fn mixed_step_graph_accounting() {
        use crate::exl3_forward::mixed_step_shapes;
        assert_eq!((mixed_step_shapes(0), mixed_step_shapes(1), mixed_step_shapes(2)), (0, 0, 1));
        assert_eq!(mixed_step_shapes(4), 6);
        assert_eq!(mixed_step_shapes(16), 120);
        // = the number of (width m in 2..=lanes, dense rows md in 1..m) pairs
        for lanes in 0..=16usize {
            let n = (2..=lanes).map(|m| m - 1).sum::<usize>();
            assert_eq!(mixed_step_shapes(lanes), n, "lanes {lanes}");
        }
        assert_eq!(mixed_graphs_idle_line(0, 0, 4), None);
        assert_eq!(mixed_graphs_idle_line(3, 3, 4), None, "nothing new: no line");
        let l = mixed_graphs_idle_line(0, 2, 4).unwrap();
        assert!(l.starts_with("[exl3-serve] mixed-step graphs idle: 0->2 captured since boot") && l.contains("4 lanes can form 6 mixed step shapes"), "{l}");
        // the gauge is rendered (the count itself is a process global other tests may bump)
        assert!(crate::metrics::render(true).contains("velogb10_graph_cache_mixed_step_entries "));
    }
}

/// S-B9-REL-VISIONFIX H2: the `--vision-fail-gemm` drill through the real argument path
/// (`opts::parse_cli`, what `main` runs on argv, then the accessor the serve path calls).
#[cfg(test)]
mod vision_drill_flag_tests {
    use super::*;

    /// The registry store is process-global: one test at a time writes the drill's row.
    static STORE: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn argv(tokens: &[&str]) -> Vec<String> { tokens.iter().map(|s| s.to_string()).collect() }

    /// Parse `tokens` as the process would and ask the serve path's question.
    fn armed_after_parsing(tokens: &[&str]) -> bool {
        let drill = crate::opt!("vision-fail-gemm");
        // hermetic: the command line may name no OTHER registered option (parse_cli would write
        // its row into the store every other test shares)
        for t in tokens.iter().filter_map(|t| t.strip_prefix("--")) {
            let name = t.split('=').next().unwrap();
            assert!(name == "vision-fail-gemm" || crate::opts::lookup(name).is_none(), "--{name} is a registry option: keep it out of this test");
        }
        crate::opts::test_set(drill, None);
        crate::opts::parse_cli(&argv(tokens)).expect("parse_cli");
        let armed = vision_fail_gemm_armed();
        crate::opts::test_set(drill, None);
        armed
    }

    #[test]
    fn the_drill_arms_wherever_the_flag_stands() {
        let _g = STORE.lock().unwrap_or_else(|e| e.into_inner());
        // LAST argument: the case that was silently not armed
        assert!(armed_after_parsing(&["--server", "--model-dir", "/m", "--vision-fail-gemm"]), "the flag as the last argument");
        // in the middle, and first
        assert!(armed_after_parsing(&["--server", "--vision-fail-gemm", "--model-dir", "/m"]), "the flag in the middle");
        assert!(armed_after_parsing(&["--vision-fail-gemm", "--server", "--model-dir", "/m"]), "the flag first");
        // alone
        assert!(armed_after_parsing(&["--vision-fail-gemm"]));
        // absent: not armed (the default of a server)
        assert!(!armed_after_parsing(&["--server", "--model-dir", "/m"]), "absent");
        assert!(!armed_after_parsing(&[]));
        // an explicit off is off (the old reader ARMED the drill on `--vision-fail-gemm=0`)
        assert!(!armed_after_parsing(&["--server", "--vision-fail-gemm=0", "--model-dir", "/m"]));
        assert!(!armed_after_parsing(&["--server", "--vision-fail-gemm", "off"]));
        assert!(armed_after_parsing(&["--server", "--vision-fail-gemm=1"]));
        // a look-alike is not the flag
        assert!(!armed_after_parsing(&["--server", "--vision-fail-gemm-x"]));
    }

    /// What the old reader answered on the same command lines: the bug, pinned as the reason.
    #[test]
    fn the_value_reader_misses_a_trailing_bare_flag() {
        let last = argv(&["--server", "--model-dir", "/m", "--vision-fail-gemm"]);
        assert!(arg(&last, "--vision-fail-gemm").is_none(), "a value reader sees no value after a trailing flag");
        let mid = argv(&["--server", "--vision-fail-gemm", "--model-dir", "/m"]);
        assert_eq!(arg(&mid, "--vision-fail-gemm"), Some("--model-dir"), "and takes the NEXT FLAG for the value in the middle");
        let zero = argv(&["--vision-fail-gemm=0"]);
        assert!(arg(&zero, "--vision-fail-gemm").is_some(), "and counts `=0` as given");
    }

    /// S-B9-REL-VISIONFIX2 V5: the other two members of H2's class, now registry flags: `--vision-cpu`
    /// (NVFP4 server; a CPU tower is a multi-minute path, and `=0` used to FORCE it) and `--parity`
    /// (a bench protocol flag; as the last argument it silently ran the non-parity recipe). Through
    /// the real argument path: last, middle, first, alone, absent, `=0`, `=off`, `=1`, look-alike.
    #[test]
    fn vision_cpu_and_parity_are_read_by_presence_wherever_they_stand() {
        let _g = STORE.lock().unwrap_or_else(|e| e.into_inner());
        fn on_after(flag: &str, tokens: &[&str]) -> bool {
            let id = crate::opts::lookup(flag).expect("a registry flag");
            assert!(matches!(id.def().ty, crate::opts::Ty::Flag), "--{flag} is a presence flag");
            for t in tokens.iter().filter_map(|t| t.strip_prefix("--")) {
                let name = t.split('=').next().unwrap();
                assert!(name == flag || crate::opts::lookup(name).is_none(), "--{name} is a registry option: keep it out of this test");
            }
            crate::opts::test_set(id, None);
            crate::opts::parse_cli(&argv(tokens)).expect("parse_cli");
            let r = crate::opts::on(id);
            crate::opts::test_set(id, None);
            r
        }
        for flag in ["vision-cpu", "parity"] {
            let f = format!("--{flag}");
            assert!(on_after(flag, &["--server", "--model-dir", "/m", &f]), "{f} as the LAST argument");
            assert!(on_after(flag, &["--server", &f, "--model-dir", "/m"]), "{f} in the middle");
            assert!(on_after(flag, &[&f, "--server", "--model-dir", "/m"]), "{f} first");
            assert!(on_after(flag, &[&f]), "{f} alone");
            assert!(!on_after(flag, &["--server", "--model-dir", "/m"]), "{f} absent");
            assert!(!on_after(flag, &[]), "{f} absent, no arguments");
            assert!(!on_after(flag, &["--server", &format!("{f}=0"), "--model-dir", "/m"]), "{f}=0 is off");
            assert!(!on_after(flag, &["--server", &format!("{f}=off")]), "{f}=off is off");
            assert!(!on_after(flag, &["--server", &f, "off"]), "{f} off is off");
            assert!(on_after(flag, &["--server", &format!("{f}=1")]), "{f}=1 is on");
            assert!(!on_after(flag, &["--server", &format!("{f}-x")]), "a look-alike is not the flag");
        }
        // the options exist in --help (Serve / Diag), each with a help text
        for flag in ["vision-cpu", "parity"] {
            let id = crate::opts::lookup(flag).unwrap();
            assert!(id.def().help.len() > 40 && id.def().env.is_empty(), "--{flag}: a help text and no environment variable");
        }
    }

    /// V4a: `--vision-cpu-threads` is range-checked when the command line is parsed (the process
    /// exits 2 there): a bad value used to fail INSIDE the tower load, which booted the server
    /// text-only with image requests rejected.
    #[test]
    fn a_bad_vision_cpu_threads_value_is_refused_at_startup() {
        let _g = STORE.lock().unwrap_or_else(|e| e.into_inner());
        let id = crate::opt!("vision-cpu-threads");
        for bad in ["0", "65", "-1", "abc", "1.5", ""] {
            crate::opts::test_set(id, None);
            let r = crate::opts::parse_cli(&argv(&["--server", "--vision-cpu-threads", bad]));
            assert!(r.is_err(), "--vision-cpu-threads {bad:?} must be refused, got {r:?}");
            let r = crate::opts::parse_cli(&argv(&[&format!("--vision-cpu-threads={bad}")]));
            assert!(r.is_err(), "--vision-cpu-threads={bad:?} must be refused");
            assert!(crate::opts::get(id).is_none(), "a refused value is not stored");
        }
        // The store is process-global and `VisualTower::load` (other test modules) reads this row: only the
        // default value is ever written to it here; the other values go through the pure accessor.
        crate::opts::test_set(id, None);
        crate::opts::parse_cli(&argv(&["--vision-cpu-threads", "1"])).expect("a good value");
        assert_eq!(crate::opts::ranged(id).unwrap(), 1);
        crate::opts::test_set(id, None);
        for (good, n) in [("1", 1i64), ("8", 8), ("64", 64), (" 4 ", 4)] {
            assert_eq!(crate::opts::ranged_from(id, Some(good)).unwrap(), n);
        }
        assert_eq!(crate::opts::ranged(id).unwrap(), 1, "unset = the default 1");
        // a value that reached the store another way is an Err for the accessor, never a silent default
        for bad in ["0", "65", "-1", "many", ""] {
            assert!(crate::opts::ranged_from(id, Some(bad)).is_err(), "{bad:?}");
        }
    }

    /// The serve path asks the registry, and nothing in the tree reads a registry flag with a value
    /// reader's `is_some()` again. The value readers stay right for options that TAKE a value. The
    /// two bare flags that were still misread (`--vision-cpu`, `--parity`) are registry flags since
    /// S-B9-REL-VISIONFIX2 V5 (the list of bare-and-still-misread names is empty and may not grow).
    #[test]
    fn no_registry_flag_is_read_with_a_value_readers_is_some() {
        const TAKES_A_VALUE: [&str; 4] = ["server-port", "bench-file", "model-dir", "model"];
        const BARE_AND_STILL_MISREAD: [&str; 0] = [];
        let serve = include_str!("exl3_serve.rs");
        assert_eq!(serve.matches(&["vision_boot(model_dir, ", "vision_fail_gemm_armed())"].concat()).count(), 1, "the serve path asks the registry");
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files: Vec<std::path::PathBuf> = Vec::new();
        let mut dirs = vec![root.clone()];
        while let Some(d) = dirs.pop() {
            for e in std::fs::read_dir(&d).unwrap().flatten() {
                let p = e.path();
                if p.is_dir() { dirs.push(p); } else if p.extension().map_or(false, |x| x == "rs") { files.push(p); }
            }
        }
        assert!(files.len() > 40);
        let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for f in &files {
            let text = std::fs::read_to_string(f).unwrap();
            for reader in ["arg(args, \"--", "parse_arg(args, \"--", "arg_value(args, \"--", "parse_arg(&args, \"--"] {
                for (at, _) in text.match_indices(reader) {
                    let rest = &text[at + reader.len()..];
                    let Some(q) = rest.find('"') else { continue };
                    let after = rest[q + 1..].trim_start();
                    let Some(after) = after.strip_prefix(')') else { continue };
                    let after = after.trim_start();
                    if !(after.starts_with(".is_some()") || after.starts_with(".is_none()")) { continue; }
                    let name = rest[..q].to_string();
                    if let Some(id) = crate::opts::lookup(&name) {
                        assert!(!matches!(id.def().ty, crate::opts::Ty::Flag | crate::opts::Ty::Bool),
                                "{}: --{name} is a registry flag read with a value reader's is_some(): use opts::on", f.display());
                    }
                    seen.insert(name);
                }
            }
        }
        let allowed: std::collections::BTreeSet<String> = TAKES_A_VALUE.iter().chain(BARE_AND_STILL_MISREAD.iter()).map(|s| s.to_string()).collect();
        assert_eq!(seen, allowed, "options read with <value reader>(..).is_some(): a new one must take a value, or be read by presence");
    }
}

// ------------------------------------------------------------------------------------------------------
// S-B9-CF-AGREE16: the lockstep words must tell every verify width apart (0..=31), 15 from 16 included.
// These tests drive the real field packing (`tp_agree_fields`), the real token (`net::agree_token`) and the
// real comparison (`tp_agree_decide`); only the wire is replaced: `net_agree` hands each rank the other
// rank's token for the same step field, which is what `pair` does.
// ------------------------------------------------------------------------------------------------------
#[cfg(test)]
mod tp_agree_width_tests {
    use super::*;

    type Said = (usize, usize, u32); // (accept, verify width, digest) as one rank passes them to tp_agree_eq

    /// What the two ranks of a TP=2 pair decide at one lockstep point: (this rank, the peer).
    fn pair(step: u64, a: Said, b: Said) -> (std::result::Result<(), String>, std::result::Result<(), String>) {
        let token = |s: Said| tp_agree_fields(s.0, s.1).map(|(a8, k8)| (a8, k8, crate::net::agree_token(step, a8, k8, s.2)));
        let side = |me: Said, other: Said| -> std::result::Result<(), String> {
            let (a8, k8, _) = token(me)?; // this rank refused its own fields: it aborts the link
            // a peer that refused never publishes a token: this rank's net_agree ends in an aborted link (None)
            let reply = token(other).ok().map(|(_, _, t)| crate::net::agree_token_reply(t));
            tp_agree_decide(a8, k8, me.2, reply)
        };
        (side(a, b), side(b, a))
    }

    /// Digests of the kind the lockstep points pass: FNVs of short id lists, plus the corner words.
    fn digests() -> Vec<u32> {
        let mut v = vec![0u32, u32::MAX, 0x8000_0000, 0x7800_0000, 0x0800_0000, 0xF800_0000, 0x07FF_FFFF, 1];
        v.extend((0..32).map(|b| 1u32 << b));
        for n in 0..24u32 {
            v.push(fnv32(0, (0..n).map(|i| 1000 + 7 * i)));
            v.push(fnv32(0x5EED, (0..n).map(|i| 31 * i + n)));
        }
        v
    }

    #[test]
    fn the_digest_can_set_and_clear_every_bit_of_its_word() {
        // the bits `h` can set: ALL 32. fnv32 is a full-width hash, so no bit of the 32-bit word is free for
        // a width unless the word reserves it (which is what the next test demands).
        let (mut any, mut all) = (0u32, u32::MAX);
        for n in 0..200u32 {
            let h = fnv32(0, (0..n % 17 + 1).map(|i| n.wrapping_mul(2654435761).wrapping_add(i)));
            any |= h;
            all &= h;
        }
        assert_eq!(any, u32::MAX, "the digest never set bits {:#010x}", !any);
        assert_eq!(all, 0, "the digest never cleared bits {all:#010x}");
    }

    #[test]
    fn the_width_owns_the_top_five_bits_whatever_the_digest_holds() {
        const DIGEST_MASK: u32 = (1 << 27) - 1;
        for h in digests() {
            for k in 0..=31u8 {
                let w = crate::net::agree_word(k, h);
                assert_eq!(w >> 27, k as u32, "width {k}, digest {h:#010x}: the word's top five bits are {:#04x}", w >> 27);
                assert_eq!(w & DIGEST_MASK, crate::net::agree_word(0, h) & DIGEST_MASK,
                           "width {k} changed the digest part of the word (digest {h:#010x})");
            }
            // the 27-bit digest part still hears every one of the 32 digest bits
            for b in 0..32 {
                assert_ne!(crate::net::agree_word(0, h ^ (1 << b)), crate::net::agree_word(0, h),
                           "digest bit {b} of {h:#010x} does not reach the word");
            }
        }
    }

    #[test]
    fn equal_inputs_agree_and_every_width_or_accept_difference_is_refused_0_to_17() {
        let step = agree_field(41, AGREE_A);
        let mut silent: Vec<String> = Vec::new(); // differing inputs that BOTH ranks accepted
        let mut loud: Vec<String> = Vec::new(); // equal inputs a rank refused
        for h in [0u32, 0xDEAD_BEEF, u32::MAX, 0x7800_0000, fnv32(0, 0..16), fnv32(0, 0..15)] {
            for (wa, wb) in (0..=17usize).flat_map(|a| (0..=17usize).map(move |b| (a, b))) {
                for (aa, ab) in (0..=17usize).flat_map(|a| (0..=17usize).map(move |b| (a, b))) {
                    let (ra, rb) = pair(step, (aa, wa, h), (ab, wb, h));
                    let same = wa == wb && aa == ab;
                    match (same, ra.is_ok() && rb.is_ok()) {
                        (true, false) => loud.push(format!("width {wa} accept {aa}: {ra:?} / {rb:?}")),
                        (false, true) => silent.push(format!("(width {wa}, accept {aa}) vs (width {wb}, accept {ab})")),
                        _ => {}
                    }
                }
            }
        }
        assert!(loud.is_empty(), "equal inputs must agree; refused: {:?}", &loud[..loud.len().min(6)]);
        let widths: std::collections::BTreeSet<(usize, usize)> = silent.iter()
            .filter_map(|s| { let n: Vec<usize> = s.split(|c: char| !c.is_ascii_digit()).filter_map(|t| t.parse().ok()).collect();
                              (n.len() == 4 && n[0] < n[2]).then(|| (n[0], n[2])) }).collect();
        assert!(silent.is_empty(), "{} differing (width, accept) pairs agreed in silence; width pairs that alias: {widths:?}; first: {:?}",
                silent.len(), &silent[..silent.len().min(4)]);
    }

    #[test]
    fn no_digest_difference_can_cancel_a_width_difference() {
        let step = agree_field(7, AGREE_PV);
        let mut silent: Vec<String> = Vec::new();
        let mut x = 0x9E37_79B9u32;
        for (ka, kb) in (0..=17usize).flat_map(|a| (0..=17usize).map(move |b| (a, b))).filter(|(a, b)| a != b) {
            for ha in digests() {
                // the exact differences that cancel a width XOR-folded at bit 27 (4-bit and 5-bit folds), every
                // other top-five-bit difference, and a spread of unrelated digests
                let mut others: Vec<u32> = vec![ha ^ ((((ka ^ kb) & 0xF) as u32) << 27), ha ^ ((((ka ^ kb) & 0x1F) as u32) << 27),
                                                ha ^ ((((ka.min(15) ^ kb.min(15)) & 0xF) as u32) << 27)];
                others.extend((1..32u32).map(|d| ha ^ (d << 27)));
                for _ in 0..8 { x ^= x << 13; x ^= x >> 17; x ^= x << 5; others.push(x); }
                for hb in others {
                    let (ra, rb) = pair(step, (3, ka, ha), (3, kb, hb));
                    if ra.is_ok() || rb.is_ok() {
                        silent.push(format!("width {ka} digest {ha:#010x} vs width {kb} digest {hb:#010x}"));
                    }
                }
            }
        }
        assert!(silent.is_empty(), "{} (width, digest) pairs with DIFFERENT widths produced equal words; first: {:?}",
                silent.len(), &silent[..silent.len().min(4)]);
    }

    #[test]
    fn a_value_the_token_cannot_carry_is_refused_not_clamped() {
        for a in 0..=255usize {
            for k in 0..=31usize {
                assert_eq!(tp_agree_fields(a, k), Ok((a as u8, k as u8)), "accept {a}, width {k} must ride the token exactly");
            }
        }
        for (a, k) in [(0usize, 32usize), (0, 33), (0, 255), (0, 256), (256, 0), (257, 16), (usize::MAX, 16), (3, usize::MAX)] {
            let e = tp_agree_fields(a, k).expect_err(&format!("accept {a}, width {k} cannot ride the token: it must be refused"));
            assert!(e.contains("REFUSED"), "{e}");
        }
        // a rank that refuses takes its peer down with it (the peer sees the aborted link), never one-sided
        let (ra, rb) = pair(agree_field(9, AGREE_A), (0, 32, 5), (0, 16, 5));
        assert!(ra.is_err() && rb.is_err(), "{ra:?} / {rb:?}");
        let (ra, rb) = pair(agree_field(9, AGREE_A), (256, 4, 5), (0, 4, 5));
        assert!(ra.is_err() && rb.is_err(), "{ra:?} / {rb:?}");
    }

    #[test]
    fn the_token_fields_do_not_overlap_and_the_reply_reads_them_back() {
        for (counter, kind) in [(0u64, AGREE_A), (1, AGREE_PV), (0x3F_FFFF, AGREE_B), (0x40_0000, AGREE_GO), (12345, AGREE_A)] {
            let step = agree_field(counter, kind);
            assert!(step <= 0xFF_FFFF, "the step field is 24 bits");
            for a in [0u8, 1, 15, 16, 17, 255] {
                for k in [0u8, 1, 15, 16, 17, 31] {
                    for h in [0u32, u32::MAX, 0xDEAD_BEEF] {
                        let t = crate::net::agree_token(step, a, k, h);
                        assert_eq!(t >> 40, step, "step field");
                        assert_eq!(crate::net::agree_token_reply(t), (a, crate::net::agree_word(k, h)));
                    }
                }
            }
        }
        // the boot agree's raw step word keeps its 24 bits too
        assert_eq!(crate::net::agree_token(0xB0_0700, 0, 0, 1) >> 40, 0xB0_0700);
    }

    #[test]
    fn the_step_go_fence_word_tells_step_and_event_count_apart() {
        // the pairs master's XOR word could not tell apart, and the one a folded XOR word could not
        let step = agree_field(3, AGREE_GO);
        let word = |hs: u64, hn: usize| crate::net::agree_token(step, 0, 0, go_fence_word(hs, hn));
        for ((hs_a, hn_a), (hs_b, hn_b)) in [((5u64, 0usize), (4u64, 8usize)), ((7, 0), (7 ^ (1 << 24), 1)), ((9, 0), (9, 256)),
                                             ((9, 1), (9, 257)), ((1, 0), (1 + (1 << 32), 0)), ((0, 1), (1, 0))] {
            assert_ne!(word(hs_a, hn_a), word(hs_b, hn_b), "(step {hs_a}, {hn_a} events) vs (step {hs_b}, {hn_b} events)");
        }
        // and a dense block: 64 steps x 64 event counts, all different
        let mut seen = std::collections::HashSet::new();
        for hs in 0..64u64 { for hn in 0..64usize { assert!(seen.insert(word(1000 + hs, hn)), "step {hs}, events {hn} repeats a word"); } }
    }

    #[test]
    fn a_lane_count_the_pairwise_token_cannot_name_is_refused_at_boot() {
        for w in [1usize, 4, 16, 31] { assert_eq!(tp_lane_width_fits(2, w), Ok(()), "TP=2 --max-batch {w}"); }
        for w in [32usize, 33, 64] {
            let e = tp_lane_width_fits(2, w).expect_err("TP=2 with more than 31 lanes must not boot");
            assert!(e.contains("--max-batch") && e.contains("31"), "{e}");
        }
        // TP=1 has no lockstep and the world > 2 frames carry whole u32 words: no bound from the token
        for world in [1u32, 4, 8] { assert_eq!(tp_lane_width_fits(world, 64), Ok(())); }
    }

    /// The pairwise lockstep points hand `tp_agree_eq` the real accept count and the real width: no
    /// saturation may come back between the scheduler and the token (source pin; the call sites need a
    /// live link, so they cannot be driven from a unit test).
    #[test]
    fn the_lockstep_call_sites_pass_the_real_width() {
        let src = include_str!("exl3_serve.rs");
        let body = |from: &str, to: &str| -> &str {
            let s = &src[src.find(from).unwrap_or_else(|| panic!("{from} not found"))..];
            &s[..s.find(to).unwrap_or_else(|| panic!("{to} not found after {from}"))]
        };
        let sat15 = ["min", "(15)"].concat();
        let sat255 = ["min", "(255)"].concat();
        for (name, text) in [("pre_verify", body("    fn pre_verify(&mut self, w: usize", "impl TpSync {")),
                             ("round_hub", body("    pub(crate) fn round_hub<X", "/// TP-E: the 24-bit agree step field")),
                             ("tp_agree_eq", body("pub(crate) fn tp_agree_eq(", "fn fnv32(")),
                             ("tp_round", body("    fn tp_round(&mut self, what: &str", "        let digests = if ident"))] {
            assert!(!text.contains(&sat15), "{name} saturates a verify width at 15");
            assert!(!text.contains(&sat255), "{name} saturates an accept count at 255");
        }
        assert!(body("    fn pre_verify(&mut self, w: usize", "impl TpSync {").contains("tp_agree_eq(step, w, w + 1, h)"));
        assert!(body("    fn tp_round(&mut self, what: &str", "        let digests = if ident")
                    .contains("tp_agree_eq(agree_field(step, AGREE_A), a, t.width_said(m), h)"));
    }
}

/// S-B9-CF-AGREE16B: the one-shot, rank-dependent width drill (W3) and the hoisted `--max-batch` refusal (W4).
#[cfg(test)]
mod agree16b_tests {
    use super::*;
    use crate::net::hub_mock::{run_ranks, Net};
    use std::time::Duration;

    fn armed(rank: usize, world: usize, n: usize) -> TpSync {
        let mut t = TpSync::new(rank as i32, world as i32);
        assert_eq!(t.drill_width, None, "no drill flag is set in the unit suite");
        t.drill_width = Some(n);
        t
    }

    /// W3: the drill lies ONCE (it used to lie on every N-row round), and only about its own width.
    #[test]
    fn the_width_drill_fires_once_then_tells_the_truth() {
        for world in [2usize, 4] {
            for rank in 0..world {
                let mut t = armed(rank, world, 16);
                for m in [0usize, 1, 5, 15, 17] { assert_eq!(t.width_said(m), m, "width {m} is not the drill's"); }
                assert_eq!(t.drill_hits, 0, "still armed");
                assert_eq!(t.width_said(16), 15 - rank, "rank {rank}: the first 16-row round is reported as 16 - 1 - rank");
                assert_eq!(t.drill_hits, 1);
                for _ in 0..8 { assert_eq!(t.width_said(16), 16, "rank {rank}: spent, the truth from here on"); }
                assert_eq!(t.drill_hits, 1);
            }
        }
        // not armed: the truth, always
        let mut t = TpSync::new(0, 2);
        for m in 0..=31 { assert_eq!(t.width_said(m), m); }
        // a width the rank term cannot go below zero for is left alone even if forced past `drill_plan`
        let mut t = armed(1, 2, 1);
        assert_eq!((t.width_said(1), t.drill_hits), (1, 0));
    }

    /// W3, TP=2 (the pairwise token): armed on the head only it is the 15-against-16 fault; armed on the
    /// node only, or on BOTH ranks, the two words still differ, in both directions. Before AGREE16B two
    /// armed ranks both said N - 1 and agreed.
    #[test]
    fn a_width_drill_on_one_rank_or_on_both_fails_the_pairwise_agree() {
        let words = |said: [usize; 2], h: u32| -> [(u8, u8, (u8, u32)); 2] {
            [0, 1].map(|r| {
                let (a, k) = tp_agree_fields(0, said[r]).unwrap();
                (a, k, crate::net::agree_token_reply(crate::net::agree_token(agree_field(87, AGREE_A), a, k, h)))
            })
        };
        let fails = |said: [usize; 2], h: u32| {
            let w = words(said, h);
            let (e0, e1) = (tp_agree_decide(w[0].0, w[0].1, h, Some(w[1].2)), tp_agree_decide(w[1].0, w[1].1, h, Some(w[0].2)));
            assert!(e0.is_err() && e1.is_err(), "widths said {said:?} (digest {h:08x}) must fail on both ranks: {e0:?} / {e1:?}");
            (e0.unwrap_err(), e1.unwrap_err())
        };
        for n in 2..=31usize {
            for h in [0u32, 0x99915f3e, 0xFFFF_FFFF] {
                let said = |on: [bool; 2]| [0, 1].map(|r| if on[r] { armed(r, 2, n).width_said(n) } else { n });
                assert_eq!(said([false, false]), [n, n]);
                assert_eq!(said([true, false]), [n - 1, n], "the head alone");
                assert_eq!(said([false, true]), [n, n - 2], "the node alone");
                assert_eq!(said([true, true]), [n - 1, n - 2], "both ranks: two different lies");
                for on in [[true, false], [false, true], [true, true]] { fails(said(on), h); }
                let w = words(said([false, false]), h);
                assert_eq!(tp_agree_decide(w[0].0, w[0].1, h, Some(w[1].2)), Ok(()), "no drill: the round agrees");
            }
        }
        // the gate's line, head armed with 16: "this rank (accept 0, width 15, ..) vs peer (accept 0, width 16, ..)"
        let (head, node) = fails([15, 16], 0x99915f3e);
        assert!(head.contains("this rank (accept 0, width 15, word 79915f2d) vs peer (accept 0, width 16, word 81915f2d)"), "{head}");
        assert!(node.contains("this rank (accept 0, width 16, word 81915f2d) vs peer (accept 0, width 15, word 79915f2d)"), "{node}");
    }

    /// W3, world 4 (the hub frames): the drill armed on EVERY rank still fails the 16-row round on every
    /// rank (each reports a different width), and a rank armed for another width does not disturb it.
    #[test]
    fn a_width_drill_on_every_rank_fails_the_hub_round_on_every_rank() {
        const W: usize = 4;
        let net = Net::new(W, true);
        let out = run_ranks(&net, Some(Duration::from_secs(5)), |r, hub| {
            let mut t = armed(r, W, 16);
            let first = t.round_hub(hub, "verify", 2, 5, &[1, 2], 1.0, 1.0, false, 0, |_| Ok(vec![])).map(|_| ()).map_err(|e| format!("{e:#}"));
            let second = t.round_hub(hub, "plain-step", 0, 16, &[1, 2], 1.0, 1.0, false, 0, |_| Ok(vec![])).map(|_| ()).map_err(|e| format!("{e:#}"));
            (first, second, t.drill_hits)
        });
        for (r, (first, second, hits)) in out.iter().enumerate() {
            assert!(first.is_ok(), "rank {r}: a 5-row round is not the drill's width: {first:?}");
            let e = second.as_ref().expect_err(&format!("rank {r}: every rank lied differently about the 16-row round — it must fail"));
            assert!(e.contains("verify width"), "rank {r}: {e}");
            assert_eq!(*hits, 1, "rank {r}: fired once");
        }
    }

    /// W4, now F2 (owner 2026-10-10): the lane count from the argv alone, through the ONE resolver the
    /// serve uses. A `--max-batch` above 16 no longer refuses anywhere: the cap resolves it to 16
    /// first (16 always fits the pairwise agree width), the boot prints the one plain line, and
    /// further requests wait in the queue.
    #[test]
    fn the_max_batch_refusal_reads_the_argv_the_serve_reads() {
        let argv = |v: &[&str]| -> Vec<String> { v.iter().map(|s| s.to_string()).collect() };
        assert_eq!(max_batch_arg(&argv(&["--server", "--tp", "2"])), 8, "the serve's default");
        assert_eq!(max_batch_arg(&argv(&["--max-batch", "16"])), 16);
        for ok in [&["--max-batch", "1"][..], &["--max-batch", "16"], &["--max-batch", "31"], &["--server"]] {
            assert!(tp_lane_width_refusal(&argv(ok), 2).is_ok(), "{ok:?} boots at TP=2");
        }
        for bad in ["32", "33", "64", "256", "287"] {
            let raw: usize = bad.parse().unwrap();
            for world in [1u32, 2, 4] {
                assert!(tp_lane_width_refusal(&argv(&["--server", "--max-batch", bad, "--tp", "2"]), world).is_ok(),
                        "--max-batch {bad} at world {world}: the 16-lane cap resolves it, it does not refuse");
            }
            assert_eq!(resolved_max_batch(&argv(&["--max-batch", bad])), (16, raw), "the one resolver caps at 16");
        }
        assert_eq!(resolved_max_batch(&argv(&["--server"])), (8, 8), "the default sits under the cap, uncapped");
    }

    /// W4, now F2, the call sites (pinned in the source: they sit in the boot path of a TP pair). The
    /// cap call sites remain as the BACKSTOP for a caller that bypasses the 16-lane resolver — the
    /// head before discovery and the pack sync, the node before the link, both before the RDMA
    /// attach; and `build_serve` resolves `--max-batch` through the same resolver before its backstop.
    #[test]
    fn the_max_batch_refusal_runs_before_any_network_step() {
        let code = |s: &str| s.lines().map(|l| l.split("//").next().unwrap_or("")).collect::<Vec<_>>().join("\n");
        let body = |src: &str, from: &str, to: &str| -> String {
            let a = src.find(from).unwrap_or_else(|| panic!("{from} not found"));
            let b = src[a + from.len()..].find(to).unwrap_or_else(|| panic!("{to} not found after {from}")) + a + from.len();
            code(&src[a..b])
        };
        let before = |text: &str, first: &str, then: &str, what: &str| {
            let (a, b) = (text.find(first).unwrap_or_else(|| panic!("{what}: `{first}` is missing")),
                          text.find(then).unwrap_or_else(|| panic!("{what}: `{then}` is missing")));
            assert!(a < b, "{what}: `{first}` must come before `{then}`");
        };
        let main = include_str!("main.rs");
        let head = body(main, "fn run_exl3_tp_serve_head(", "\nfn ");
        before(&head, "tp_lane_width_refusal(args, world as u32)", "run_head_retain(", "the head's main");
        before(&head, "tp_lane_width_refusal(args, world as u32)", "TpConfig::from_opts()", "the head's main");
        let prelink = body(main, "fn exl3_tp_node_prelink(", "\nfn ");
        before(&prelink, "tp_lane_width_refusal(&argv, tpc.world as u32)?", "Exl3NodeProgram::Serve(argv)", "the node's prelink");
        let serve = include_str!("exl3_serve.rs");
        let h = body(serve, "pub fn run_tp_head_serve(", "\npub fn ");
        before(&h, "tp_lane_width_refusal(args, ctx.world as u32)?", "tp_attach(ctx)", "run_tp_head_serve");
        let n = body(serve, "pub fn run_tp_node_serve(", "\nfn ");
        before(&n, "tp_lane_width_refusal(args, node_world as u32)?", "tp_attach(ctx)", "run_tp_node_serve");
        let b = body(serve, "fn build_serve(", "\nfn ");
        before(&b, "let (mut width, raw_width) = resolved_max_batch(args);", "tp_lane_width_fits(tp_world, width)", "build_serve (the 16-lane cap, then the backstop)");
        before(&b, "tp_lane_width_fits(tp_world, width)", "FwdModel::load", "build_serve (the backstop, before the load)");
    }

    /// MAXBATCH1: the boot-anyway memory clamp is pinned in the source — it runs in `build_serve`
    /// BEFORE the tune table is selected (the table is keyed by the lane count that will serve),
    /// it reassigns `width` to the clamped count, and the CF-P1d exit(2) refusal survives ONLY
    /// for the nothing-fits arm. The over-cap arrival hooks ride every rx arrival site.
    #[test]
    fn the_memory_fit_clamps_lanes_before_the_tune_table_and_keeps_the_nothing_fits_refusal() {
        let src = include_str!("exl3_serve.rs");
        let code = |s: &str| s.lines().map(|l| l.split("//").next().unwrap_or("")).collect::<Vec<_>>().join("\n");
        let body = |src: &str, from: &str, to: &str| -> String {
            let a = src.find(from).unwrap_or_else(|| panic!("{from} not found"));
            let b = src[a + from.len()..].find(to).unwrap_or_else(|| panic!("{to} not found after {from}")) + a + from.len();
            code(&src[a..b])
        };
        let b = body(src, "fn build_serve(", "\nfn ");
        let clamp = b.find("fit_max_lanes(parts, width, avail, mode, max_pos)").expect("the clamp call in build_serve");
        let tune = b.find("tune::boot(&tune::BootReq").expect("the tune table boot");
        assert!(clamp < tune, "the clamp runs before the tune table is picked (the table is keyed by the serving lane count)");
        let reassign = b.find("width = n;").expect("the clamped lane count is assigned to width");
        assert!(reassign > clamp, "the reassignment follows the clamp");
        assert!(b[clamp..].contains("std::process::exit(2)"), "the nothing-fits refusal survives (CF-P1d's teeth)");
        // The queue path F2 boots counts its over-cap arrivals: the three rx arrival sites in the
        // scheduler (top of loop, the reserve branch, the Plan::Wait arm) all carry the hook.
        assert!(src.matches("crate::metrics::over_cap_queued()").count() >= 3,
                "expected >= 3 over_cap_queued hooks, found {}", src.matches("crate::metrics::over_cap_queued()").count());
    }
}

/// S-B9-REL-INTEG F1 (review of the orchestrator's `2d52ad79`): `--lane-pack-sampled-tp` through the real
/// argument path (`opts::parse_cli`, then the accessor the boot reads), and the three properties the option
/// rests on: it rides to every rank (Spmd, in the TpConfig snapshot and the boot-agree digest), `off`
/// restores the earlier graph set (greedy family only), and a bad value is refused at startup.
#[cfg(test)]
mod lane_pack_sampled_tp_tests {
    use super::*;

    /// The registry store is process-global: one test at a time writes this option's row.
    static STORE: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn argv(tokens: &[&str]) -> Vec<String> { tokens.iter().map(|s| s.to_string()).collect() }

    #[test]
    fn the_option_reaches_the_boot_and_rides_to_the_nodes() {
        let _g = STORE.lock().unwrap_or_else(|e| e.into_inner());
        let id = crate::opt!("lane-pack-sampled-tp");
        crate::opts::test_set(id, None);
        // default: ON (sampled requests join packed rounds under TP)
        assert!(pack_sampled_under_tp(), "the default is on");
        assert_eq!(id.def().def, "on");
        // the head ships it: an SPMD option (the node installs the head's value; the boot agree digest covers it)
        assert_eq!(id.def().scope, crate::opts::Scope::Spmd);
        crate::opts::parse_cli(&argv(&["--lane-pack-sampled-tp", "off"])).unwrap();
        assert!(!pack_sampled_under_tp(), "--lane-pack-sampled-tp off");
        let shipped = crate::opts::snapshot();
        assert!(shipped.contains(&("lane-pack-sampled-tp".to_string(), "off".to_string())), "in the TpConfig snapshot: {shipped:?}");
        assert!(crate::opts::spmd_digest().is_some(), "and in the boot-agree digest");
        crate::opts::test_set(id, None);
        crate::opts::parse_cli(&argv(&["--lane-pack-sampled-tp=on"])).unwrap();
        assert!(pack_sampled_under_tp(), "--lane-pack-sampled-tp=on");
        // a value that is not on|off is refused when the command line is parsed, with the accepted spellings
        crate::opts::test_set(id, None);
        let e = crate::opts::parse_cli(&argv(&["--lane-pack-sampled-tp", "maybe"])).unwrap_err();
        assert!(e.contains("--lane-pack-sampled-tp") && e.contains("'maybe'"), "{e}");
        assert!(crate::opts::parse_cli(&argv(&["--lane-pack-sampled-tp"])).is_err(), "a bare flag needs a value");
        assert!(crate::opts::get(id).is_none(), "a refused value is not stored");
        // `off` keeps the graph set of the builds before the option: the greedy family alone; `on` adds the
        // family the boot's sampling rule asks for (real-q under --spec-sampling ratio, the default)
        use crate::lane_pack::{fams, PackFam};
        assert_eq!(fams(true, false), vec![PackFam::Greedy]);
        assert_eq!(fams(false, false), vec![PackFam::Greedy]);
        assert_eq!(fams(true, true), vec![PackFam::Greedy, PackFam::Rq]);
        assert_eq!(fams(false, true), vec![PackFam::Greedy, PackFam::Match]);
        // the boot's decision: TP=1 always packs sampled requests; under TP the option decides (and a skipped
        // sampled lane is counted as `sampled`: `sampled_lone = dump_on || !pack_sampled`)
        let src = include_str!("exl3_serve.rs");
        let code = &src[..src.find("mod lane_pack_sampled_tp_tests").unwrap()];
        assert!(code.contains("let sampled_ok = tp_rank.is_none() || pack_sampled_under_tp();"));
        assert!(code.contains("sampled_lone: crate::wp24::dump_on() || !self.pack_sampled,"));
        crate::opts::test_set(id, None);
    }
}

/// S-B9-REL-INTEG: the `--lane-pack on` boot warning for penalty defaults (the owner's posture has
/// `--default-presence-penalty 1.5`, under which no request packs).
#[cfg(test)]
mod lane_pack_penalty_warning_tests {
    use super::*;

    #[test]
    fn the_warning_follows_the_server_defaults() {
        // the no-op defaults (rep 1, presence 0, frequency 0): nothing to warn about
        assert!(lane_pack_penalty_warning(1.0, 0.0, 0.0).is_none());
        // the owner's posture, and each sibling on its own
        for (r, p, f) in [(1.0, 1.5, 0.0), (1.0, 0.0, 0.3), (1.1, 0.0, 0.0), (1.1, 1.5, 0.3)] {
            let w = lane_pack_penalty_warning(r, p, f).unwrap_or_else(|| panic!("({r}, {p}, {f}) must warn"));
            assert!(w.starts_with("[exl3-serve] lane-pack WARNING") && w.contains("never joins a packed verify")
                    && w.contains("--default-presence-penalty") && w.contains(&format!("presence {p}")), "{w}");
        }
        // it is asked exactly where the defaults are parsed, and only when the arm is on
        let src = include_str!("exl3_serve.rs");
        let code = &src[..src.find("mod lane_pack_penalty_warning_tests").unwrap()];
        let at = code.find("if let Some(w) = lane_pack_penalty_warning(").expect("the call");
        assert!(code[..at].trim_end().ends_with("if lane_pack.on() {"), "the warning is only for a boot with the arm on");
    }
}

// ---- S-C4PRIMEA: --exl3-prime-batch (default on; "0"/"off" disables) — batch the shared
// step's per-lane mtp_plain_prime head-row writes into ONE gather + ONE re-prime pass.
// --exl3-prime-batch-xcheck (default on): the FIRST batched call per boot bit-compares the
// legacy per-lane head-KV rows against the batched ones on the real path (one-time cost).
fn prime_batch_on() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        !matches!(crate::opts::var(crate::opt!("exl3-prime-batch")).as_deref(), Ok("0") | Ok("off"))
    })
}
fn prime_batch_xcheck_take() -> bool {
    static DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if matches!(crate::opts::var(crate::opt!("exl3-prime-batch-xcheck")).as_deref(), Ok("0") | Ok("off")) {
        return false;
    }
    !DONE.swap(true, std::sync::atomic::Ordering::Relaxed)
}
fn prime_batch_boot_log(lanes: usize) {
    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
        crate::deprintln!("PRIME-BATCH armed: shared-step head-row primes batched ({lanes} lanes this step; \
xq_gather_tap_rows + ONE re-prime pass + xq_attn_kv_prefill_pairs; --exl3-prime-batch off disables)");
    }
}

// ---- ISSUES[94] DF2 gate tests: the reason table is host-pure (df2_gate_reason touches no
// GPU and no filesystem), so the flag-parse fallback outcomes are pinned here exactly as the
// boot line prints them. A non-existent dir is NOT a gate reason: it engages and fails at the
// LOAD (reason `load failed: <e>`), which needs the GPU and cannot be tested on the host.
#[cfg(test)]
mod df2_gate_tests {
    use super::*;
    use crate::batch::SpecSource;

    const DIR: Option<&str> = Some("/a/df2/artifact/dir");

    #[test]
    fn df2_gate_engages_every_dflash2_source_on_the_27b_pack_shape() {
        for src in [SpecSource::DFlash2, SpecSource::DFlash2Rq, SpecSource::DFlash2Auto,
                    SpecSource::DFlash2Tree] {
            assert_eq!(df2_gate_reason(src, DIR, 5120, 248_320, 62, 262_144, 1), None,
                       "{src:?} + dir + dims + ctx within the pin must ENGAGE");
        }
    }

    #[test]
    fn df2_gate_source_wins_over_everything() {
        // a non-DF2 source never engages, even with a dir, the right dims and a sane ctx
        for src in [SpecSource::Mtp, SpecSource::Dspark, SpecSource::Plain] {
            assert_eq!(df2_gate_reason(src, DIR, 5120, 248_320, 62, 262_144, 1).as_deref(),
                       Some("source"), "{src:?}");
        }
        // the flag's own parse: unknown values die with the trunk's exact error text; the
        // default (no flag) is dflash2-auto, so a plain boot is a DF2-class boot that falls
        // back on the missing artifact, not on `source`.
        let mut args: Vec<String> = vec!["gb10_inference".into(), "--model-dir".into(), "/m".into()];
        assert!(matches!(resolve_spec_source_exl3(&args), SpecSource::DFlash2Auto));
        args.push("--spec-source".into()); args.push("dflash2-tree".into());
        assert!(matches!(resolve_spec_source_exl3(&args), SpecSource::DFlash2Tree));
        args.push("--spec-source=wharrgarbl".into());
        // (the error path eprintln!s and exits(1): not exercised here — it is the same text
        // main.rs prints, pinned by that file's own tests.)
    }

    #[test]
    fn df2_gate_missing_dir_is_the_default_boot_reason() {
        // dflash2-auto default + no --draft-dir: the old silent-ignore case, now a reason
        assert_eq!(df2_gate_reason(SpecSource::DFlash2Auto, None, 5120, 248_320, 62, 262_144, 1).as_deref(),
                   Some("no --draft-dir"));
        // every DF2-class source without a dir says the same thing
        for src in [SpecSource::DFlash2, SpecSource::DFlash2Rq, SpecSource::DFlash2Tree] {
            assert_eq!(df2_gate_reason(src, None, 5120, 248_320, 62, 262_144, 1).as_deref(),
                       Some("no --draft-dir"));
        }
    }

    #[test]
    fn df2_gate_dims_reason_reports_got_and_want() {
        assert_eq!(df2_gate_reason(SpecSource::DFlash2, DIR, 3072, 248_320, 62, 262_144, 1).as_deref(),
                   Some("dims hidden=3072x5120 vocab=248320x248320 layers=62 (need >61)"));
        assert_eq!(df2_gate_reason(SpecSource::DFlash2, DIR, 5120, 151_936, 62, 262_144, 1).as_deref(),
                   Some("dims hidden=5120x5120 vocab=151936x248320 layers=62 (need >61)"));
        // layers must be MORE than 61: the Flash-Next backbone
        assert_eq!(df2_gate_reason(SpecSource::DFlash2, DIR, 5120, 248_320, 61, 262_144, 1).as_deref(),
                   Some("dims hidden=5120x5120 vocab=248320x248320 layers=61 (need >61)"));
    }

    #[test]
    fn df2_gate_ctx_and_tp_reasons() {
        // one past the artifact's safe ceiling
        assert_eq!(df2_gate_reason(SpecSource::DFlash2, DIR, 5120, 248_320, 62, 262_145, 1).as_deref(),
                   Some("ctx 262145>262144"));
        // EXL27D item 2: the tp reason is RETIRED — a TP boot ENGAGES like TP=1 (the
        // TP-DF2 leg ships the artifact + argv; both ranks run the identical scheduler)
        assert_eq!(df2_gate_reason(SpecSource::DFlash2, DIR, 5120, 248_320, 62, 262_144, 2), None,
                   "tp=2 must ENGAGE now (the single-process contract is retired)");
    }

    #[test]
    fn df2_boot_lines_are_verbatim() {
        let sha = "0123456789abcdef".repeat(4);
        assert_eq!(df2_boot_line_engaged("/d", &sha, 8, false, 1),
                   format!("[exl3-serve] df2: ENGAGED draft=/d sha256={sha} block=8 round-shard=off (DFlash2 chain on EXL3, TP=1; chain verify k<=7 = the pack MTP wide-verify width)"));
        assert_eq!(df2_boot_line_engaged("/d2", &sha, 16, true, 2),
                   format!("[exl3-serve] df2: ENGAGED draft=/d2 sha256={sha} block=16 round-shard=on (DFlash2 chain on EXL3, TP=2; chain verify k<=7 = the pack MTP wide-verify width)"));
        assert_eq!(df2_boot_line_fallback("source"),
                   "[exl3-serve] df2: FALLBACK to pack MTP (source)");
        assert_eq!(df2_boot_line_fallback("no --draft-dir"),
                   "[exl3-serve] df2: FALLBACK to pack MTP (no --draft-dir)");
        assert_eq!(df2_boot_line_fallback("load failed: no such file"),
                   "[exl3-serve] df2: FALLBACK to pack MTP (load failed: no such file)");
    }

    #[test]
    fn df2_metrics_render_carries_the_exact_series_names() {
        let mut o = String::new();
        df2_metrics_append(&mut o);
        for series in ["exl3_df2_engaged ", "exl3_df2_rounds_total ",
                       "exl3_df2_tokens_accepted_total ", "exl3_df2_fallback_mtp_total ",
                       "exl3_df2_dds_bypass_total "] {
            assert!(o.contains(series), "missing series `{series}` in:\n{o}");
        }
    }
}

/// ISSUES 121: the world > 2 spill agree end to end over the four-rank mock hub (the real `hub_all`, frame and
/// verdict): healthy writes agree on every rank, one rank's failed write drops the snapshot on every rank, and no rank
/// aborts its link either way. The shipped build failed this on every rank at the first TP=4 capture.
#[cfg(test)]
mod spill_agree_tests {
    use super::*;
    use crate::net::hub_mock::{run_ranks, Net};

    #[test]
    fn ranks_agree_on_a_capture_and_drop_it_together_when_one_write_failed() {
        for world in [3usize, 4] {
            for failed in [None, Some(0usize), Some(world - 1)] {
                let net = Net::new(world, true);
                let outs = run_ranks(&net, None, |rank, hub| {
                    let mut t = TpSync::new(rank as i32, world as i32);
                    (0..3).map(|_| t.spill_agree_hub(hub, Some(rank) != failed).map_err(|e| format!("{e:#}")))
                          .collect::<Vec<_>>()
                });
                for (rank, v) in outs.iter().enumerate() {
                    for r in v {
                        assert_eq!(r, &Ok(failed.is_none()), "world {world}, failed write on {failed:?}: rank {rank}");
                    }
                }
                assert!((0..world).all(|r| net.abort_code(r) == 0), "no rank aborts its link on a healthy vote");
                assert!(net.clobbers().is_empty(), "{:?}", net.clobbers());
            }
        }
    }
}

/// ISSUES 127 / 128 (static review P0s): source pins for the DF2 width clamp and the draft slot staging (the ISSUES 126
/// ring re-alignment was measured and NOT kept: archive/reviewfix1-issue126-attempt). The behaviour needs a GPU; these keep a later edit from quietly restoring the old code.
#[cfg(test)]
mod review_p0_pins {
    fn code(src: &str) -> String {
        src.lines().filter(|l| !l.trim_start().starts_with("//")).collect::<Vec<_>>().join("\n")
    }

    #[test]
    fn df2_round_width_is_clamped_and_staged_for_greedy_and_sampled() {
        let s = code(include_str!("exl3_serve.rs"));
        // 127: one clamped width, staged for greedy and sampled alike
        assert!(s.contains("let k_df2 = self.df2_k.min(crate::exl3_forward::MTP_MAX_K).min(k_eff);"), "DF2 width clamp");
        let k = s.find("let k_df2 = self.df2_k").unwrap();
        let branch = &s[k..k + 2500];
        let stage = branch.find("self.model.set_penalties(&mut self.sc, &vec![pen; k_df2 + 1])?;").expect("penalty rows at k_df2+1");
        assert!(stage < branch.find("if rq_ratio {\n").unwrap(), "rows staged before the sampled/greedy split");
    }

    #[test]
    fn served_draft_chains_stage_the_lane_slot_in_every_regime() {
        let s = code(include_str!("exl3_forward.rs"));
        assert!(!s.contains(&["if dqsa { self.dev.htod_copy_into(vec![slot as i32; sc.slot_ids", ".len()], &mut sc.slot_ids)?; }"].concat()),
                "the fixed chain no longer stages the slot only for a sparse last pass");
        assert!(!s.contains(&["if self.qsa_live(p + kmax - 1) {\n            if stage {\n                let st = sc.wp05", ""].concat()),
                "the adaptive chain no longer stages the slot only for a sparse last pass");
    }
}
