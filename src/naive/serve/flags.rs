//! Serve flags read from argv (lane count, PDL, fuse, drafter dir, speculation config) and `sample_params`.
//! Siblings: `head.rs` / `loops.rs` consume them; `mod.rs` has the map.

use super::*;

/// `--naive-pdl=0|1` (default 0 until the A/B on the posture lands): programmatic dependent launch for
/// every naive kernel (gpu.rs `set_pdl`). Read from the head's argv on every rank, so the ranks agree.
/// Default ON since 2026-10-20 (owner overnight: the A/B on THE POSTURE at TP4, build 9fed3324: plain step 21.2 -> 20.8 ms,
/// every speculative cell ~1% faster, 7/7 greedy + sampled streams byte-identical, cancel PASS). `--naive-pdl=0` turns it off.
/// The Naive server's lane cap (G1 serves 1..=4 requests per shared step).
pub const MAX_LANES: usize = 4;

/// G1 concurrent requests: the standard `--max-batch` (owner 2026-10-21; absent = the generic default 8), capped at
/// MAX_LANES; `--naive-lanes N` (N >= 1) still overrides it. Returns (lanes, what was asked). Head argv, every rank agrees.
pub fn lanes_asked(args: &[String]) -> (usize, usize) {
    let num = |f: &str| crate::arg_value(args, f).and_then(|s| s.parse::<usize>().ok());
    let asked = num("--naive-lanes").filter(|&n| n >= 1).or_else(|| num("--max-batch")).unwrap_or(8).max(1);
    (asked.min(MAX_LANES), asked)
}

pub fn lanes_flag(args: &[String]) -> usize { lanes_asked(args).0 }

pub fn pdl_flag(args: &[String]) -> bool {
    crate::arg_value(args, "--naive-pdl").map(|s| s == "1" || s == "on").unwrap_or(true)
}
/// `--naive-fuse-combine=0|1` (default 1): the F13 fused split-K combine. Head argv, every rank agrees.
pub fn fuse_flag(args: &[String]) -> bool {
    crate::arg_value(args, "--naive-fuse-combine").map(|s| s != "0" && s != "off").unwrap_or(true)
}

/// ISSUES 98: the drafter dir the HEAD ships to its nodes (through the cluster sync's draft round, the same
/// content-addressed blob cache the DFlash2 / DSpark drafters use): `--draft-dir` when it exists on the head, else the
/// drafter next to the pack; None with `--spec-source none|off` or when no drafter is on the head's disk. Prints nothing.
pub fn head_draft_dir(args: &[String], model_dir: &str) -> Option<String> {
    let src = crate::arg_value(args, "--spec-source").unwrap_or("dspark");
    if src == "none" || src == "off" { return None; }
    let ok = |p: &std::path::Path| p.join("config.json").is_file() && p.join("model.safetensors").is_file();
    if let Some(e) = crate::arg_value(args, "--draft-dir") {
        if ok(std::path::Path::new(e)) { return Some(e.to_string()); }
    }
    std::path::Path::new(model_dir.trim_end_matches('/')).parent().map(|p| p.join("Naive-N0.5-Flash-FP8-Draft"))
        .filter(|p| ok(p)).map(|p| p.to_string_lossy().to_string())
}

/// The speculative configuration from the head's argv: `--spec-source dspark|none` (default dspark when the
/// drafter dir exists), `--draft-dir <dir>` (default: `Naive-N0.5-Flash-FP8-Draft` next to the pack), `--spec-block B`.
pub fn spec_cfg(args: &[String], model_dir: &str) -> Option<SpecCfg> {
    let src = crate::arg_value(args, "--spec-source").unwrap_or("dspark");
    if src == "none" || src == "off" { return None; }
    let default_dir = std::path::Path::new(model_dir.trim_end_matches('/')).parent().map(|p| p.join("Naive-N0.5-Flash-FP8-Draft"));
    // `--draft-dir` is the head's path; a node rank (work item B: it loads its shard of the drafter) may not have
    // that path (the model reaches the node through the head's sync cache), so a path that does not exist on
    // THIS rank falls back to the drafter next to the rank's model dir (the sync cache's models/ directory).
    let explicit = crate::arg_value(args, "--draft-dir").map(|s| s.to_string());
    let default_ok = default_dir.filter(|p| p.join("config.json").is_file()).map(|p| p.to_string_lossy().to_string());
    let draft_dir = match (explicit, default_ok) {
        (Some(e), _) if std::path::Path::new(&e).join("config.json").is_file() => Some(e),
        (Some(e), Some(d)) => { println!("[naive-serve] --draft-dir {e} is not on this rank; using {d}"); Some(d) }
        (Some(e), None) => Some(e),
        (None, d) => d,
    };
    let block: usize = crate::arg_value(args, "--spec-block").and_then(|s| s.parse().ok()).unwrap_or(SPEC_BLOCK);
    // `--spec-conf T`: verify the longest prefix of drafts whose confidence head says >= T (0 = always the
    // whole block, the pre-adaptive behaviour). The drafter ships a trained confidence head for exactly this.
    // Work item D (first pass, 2026-10-20 sweep at TP4 with the sharded drafter, production sampling, 3.5 bpw): the
    // cheaper round pays with a HIGHER width threshold and a LOWER guard -- conf 0.45 + guard 0.5/0.8 gave thinking
    // prose 58.7/53.4/49.5 tok/s vs 54.8/47.1/46.5 at the old defaults (0.25, 0.8/1.1) and the C-code cell 74.3/75.9 vs
    // 73.4/65.1/63.0; conf alone (0.45 / 0.6) and the guard alone were each in between. Topology-aware: world 4 takes
    // the swept values; TP2 / world 1 keep the earlier calibration until swept. The flags override either.
    let world = crate::arg_value(args, "--tp").and_then(|s| s.parse::<usize>().ok()).unwrap_or(1);
    // TP2 (2026-10-20 sweep at TP2 .13+.14, build 46d2bb3b, production sampling, seed 100, prose x3 / C x2): conf 0.45 alone
    // wins (prose 40.0/36.9/37.6 vs 33.2/33.2/36.1 tok/s at the old defaults, C 60.2/36.9 vs 56.6/33.9); the lower guard
    // alone does nothing there (34.7/34.1/36.5) and conf + guard is within noise of conf alone (37.4/37.6/37.3, C
    // 55.7/39.6). The posture table at TP2 then showed thinking prose LOSING 3-6% to the plain step with guard 0.8/1.1
    // (36.2-37.4 vs 38.2-38.6 tok/s: the 26 ms plain step is cheap relative to a round); guard 1.2/1.6 on the same seeds
    // gave prose 37.9/36.5/38.1 (within noise of plain) and C 48.9/61.5/61.1 (unchanged), so world 2 takes conf 0.45 +
    // guard 1.2/1.6 (--spec-probe-every 48 on top lost 20-40% on two C trajectories: rejected). World 1 is unswept and
    // keeps the earlier calibration.
    let (d_conf, d_off, d_on) = if world >= 4 { (0.45f32, 0.5f32, 0.8f32) } else if world >= 2 { (0.45f32, 1.2f32, 1.6f32) } else { (SPEC_CONF, SPEC_GUARD_OFF, SPEC_GUARD_ON) };
    let conf: f32 = crate::arg_value(args, "--spec-conf").and_then(|s| s.parse().ok()).unwrap_or(d_conf);
    let guard_off: f32 = crate::arg_value(args, "--spec-guard-off").and_then(|s| s.parse().ok()).unwrap_or(d_off);
    let guard_on: f32 = crate::arg_value(args, "--spec-guard-on").and_then(|s| s.parse().ok()).unwrap_or(d_on);
    println!("[naive-serve] spec defaults for world {world}: --spec-conf {conf} --spec-guard-off {guard_off} --spec-guard-on {guard_on} (flags override)");
    let probe_every: usize = crate::arg_value(args, "--spec-probe-every").and_then(|s| s.parse().ok()).unwrap_or(SPEC_PROBE_EVERY);
    let draft_temp: f32 = crate::arg_value(args, "--spec-draft-temp").and_then(|s| s.parse().ok()).unwrap_or(SPEC_DRAFT_TEMP);
    let drafter = crate::arg_value(args, "--spec-drafter").unwrap_or(SPEC_DRAFTER).to_string();
    let sampler = crate::arg_value(args, "--spec-sampler").unwrap_or(SPEC_SAMPLER).to_string();
    let lane_ctx = crate::arg_value(args, "--naive-lane-ctx").map(|s| s == "on" || s == "1").unwrap_or(true);
    Some(SpecCfg { draft_dir, block, conf, guard_off, guard_on, probe_every, draft_temp, drafter, sampler, lane_ctx })
}


pub(super) fn sample_params(req: &BatchRequest) -> SampleParams {
    SampleParams { temperature: req.temperature, top_p: req.top_p, top_k: req.top_k, min_p: req.min_p,
                   rep_penalty: req.rep_penalty, presence_penalty: req.presence_penalty, frequency_penalty: req.frequency_penalty }
}

