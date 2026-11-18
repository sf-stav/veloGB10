//! DFlash2 (DF2) speculation on the EXL3 serve path: flag parsing and gating (`df2_gate_reason`), the round loader, boot lines,
//! the `exl3_df2_*` counters and their /metrics rendering. The round itself runs from `mtp_round.rs`; the boot wiring is in `builder.rs`.
//! No environment variables: every knob is a command-line flag.

use super::*;

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
pub(super) fn df2_load_round(model: &mut Arc<FwdModel>, dir: &str, max_c: usize, sha_pin: Option<&str>)
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

pub(super) fn parse_df2_block(v: Option<&str>) -> Result<usize> {
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
pub(super) fn parse_df2_round_shard(v: Option<&str>) -> Result<bool> {
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
pub(super) static DF2_SHIPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

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
