//! `/metrics` rows of DSpark speculation (AGENTS §4: every new path ships a counter; §1b-bis: the engagement
//! counter is what "done" is read from). Names as leg GA's G3 reads them (`velogb10_dsv41_spec_{rounds,drafted,
//! accepted}_total`), plus the emitted tokens, the residual draws, the zero-mass fallbacks, the per-position
//! acceptance, the chosen draft lengths and the cut fallbacks. Nothing renders unless a DSpark round ran or the
//! source was announced in this process, so every other family's `/metrics` body stays byte-identical.
//! `scripts/owner_check.py` prints the section in plain words.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;

pub static ROUNDS: AtomicU64 = AtomicU64::new(0);
pub static DRAFTED: AtomicU64 = AtomicU64::new(0);
pub static ACCEPTED: AtomicU64 = AtomicU64::new(0);
pub static EMITTED: AtomicU64 = AtomicU64::new(0);
pub static SAMPLED_ROUNDS: AtomicU64 = AtomicU64::new(0);
pub static GREEDY_ROUNDS: AtomicU64 = AtomicU64::new(0);
pub static RESIDUAL_DRAWS: AtomicU64 = AtomicU64::new(0);
pub static ZERO_MASS: AtomicU64 = AtomicU64::new(0);
/// rows whose top-k / top-p cut could not be computed on the device (a host / wide-gather cut, counted)
pub static WIDE_CUTS: AtomicU64 = AtomicU64::new(0);
/// rounds the confidence-scheduled variable length shortened
pub static VARLEN_ROUNDS: AtomicU64 = AtomicU64::new(0);
static ACC_POS: Mutex<[(u64, u64); super::BLOCK]> = Mutex::new([(0, 0); super::BLOCK]);
static LEN_HIST: Mutex<[u64; super::BLOCK + 1]> = Mutex::new([0; super::BLOCK + 1]);
static SOURCE: Mutex<Option<String>> = Mutex::new(None);
/// the boot's draft-length policy as announced (`policy_label`: off / auto / fixed), so `/metrics` can say speculation is OFF
static POLICY: Mutex<Option<&'static str>> = Mutex::new(None);
static ANNOUNCED: AtomicBool = AtomicBool::new(false);
/// leg PS1: the draft-length rule of `auto` (conf / stats), the decisions by verified drafts c (one per round, all lanes
/// share c), the expected tokens the rule priced and the tokens those rounds emitted, the probes of the stats rule, the
/// round wall times, and the round-cost table the rule prices with (lanes, c, ms) plus the learned overhead per lane count.
static RULE: Mutex<Option<&'static str>> = Mutex::new(None);
static CHOSEN: Mutex<[u64; super::BLOCK + 1]> = Mutex::new([0; super::BLOCK + 1]);
static EXP_REAL: Mutex<(f64, u64, u64)> = Mutex::new((0.0, 0, 0));
pub static PROBES: AtomicU64 = AtomicU64::new(0);
static WALL: Mutex<(f64, u64)> = Mutex::new((0.0, 0));
static TABLE: Mutex<(Vec<(usize, usize, f64)>, Vec<(usize, f64)>)> = Mutex::new((Vec::new(), Vec::new()));

/// The rule `auto` uses (the boot calls it once; `None` = not `auto`).
pub fn set_rule(rule: Option<&'static str>) { if let Ok(mut r) = RULE.lock() { *r = rule; } }

/// One decided round: `c` drafts verified per lane, `expected` = the tokens the rule priced (summed over the round's
/// lanes), `realised` = the tokens the round emitted (summed over its lanes).
pub fn decision(c: usize, expected: f64, realised: u64) {
    if let Ok(mut h) = CHOSEN.lock() { h[c.min(super::BLOCK)] += 1; }
    if let Ok(mut e) = EXP_REAL.lock() { e.0 += expected; e.1 += realised; e.2 += 1; }
}

/// Leg PS3: a decided round whose lanes verified different counts: its tokens count in expected vs realised, its counts in
/// the per-lane histogram (`alloc::record`), not in the one-c histogram.
pub fn decision_uneven(expected: f64, realised: u64) {
    if let Ok(mut e) = EXP_REAL.lock() { e.0 += expected; e.1 += realised; e.2 += 1; }
}

/// The wall time of one round that verified drafts (draft step + choice + verify + bookkeeping, ms).
pub fn round_wall(ms: f64) { if let Ok(mut w) = WALL.lock() { w.0 += ms; w.1 += 1; } }

/// The round-cost table the rule prices with: (lanes, c, ms incl. the draft step), and the learned overhead per lane count.
pub fn publish_costs(table: Vec<(usize, usize, f64)>, overhead: Vec<(usize, f64)>) { if let Ok(mut t) = TABLE.lock() { *t = (table, overhead); } }

/// Leg PS2: the online round-cost model behind the table (`roundcost`): where it came from, when it last moved, how many
/// served rounds refreshed it, its speed factor, the host gap it prices, the fitted fixed + per-row cost per lane count.
#[derive(Clone, Debug)]
struct ModelInfo {
    cached: bool,
    boot_unix: u64,
    refreshed: Option<std::time::Instant>,
    published: std::time::Instant,
    served: u64,
    shifts: u64,
    rejected: u64,
    speed: f64,
    gap_ms: f64,
    fits: Vec<(usize, f64, f64)>,
}
static MODEL: Mutex<Option<ModelInfo>> = Mutex::new(None);

pub fn publish_model(m: &super::roundcost::RoundCost) {
    let now = std::time::Instant::now();
    let info = ModelInfo {
        cached: m.source == super::roundcost::Source::Cached, boot_unix: m.boot_unix,
        refreshed: m.age_s().map(|a| now - std::time::Duration::from_secs_f64(a)), published: now,
        served: m.served, shifts: m.shifts, rejected: m.rejected, speed: m.speed(), gap_ms: m.gap_ms(),
        fits: (1..=m.lanes()).filter_map(|l| m.fit(l).map(|(a, b)| (l, a, b))).collect(),
    };
    if let Ok(mut g) = MODEL.lock() { *g = Some(info); }
}

/// The once-per-boot line naming the speculation source and its policy; also makes `/metrics` show the section.
/// Returns the line the first time, None after (the caller prints it through its boot report).
pub fn announce(source: &str, policy: &str) -> Option<String> {
    if let Ok(mut s) = SOURCE.lock() { *s = Some(source.to_string()); }
    if let Ok(mut p) = POLICY.lock() { *p = Some(policy_label(policy)); }
    if ANNOUNCED.swap(true, Relaxed) { return None; }
    Some(format!("[dsv41 spec] speculation source: {source}; draft length {policy}; verify: DeepSeek's rule (sampled: accept u < min(1, q/p), \
                  residual max(q - alpha p, 0) alpha = exp(-5e-5), keyed counter draws; greedy: exact argmax)"))
}

/// The policy the boot announced, as one word for `/metrics`: `off` (draft length 0: `--mtp off` or `--dsv41-draft-len 0`),
/// `auto` (the measured-cost policy, which may itself pick 0 drafts on a round), `fixed` (a pinned length), `unknown`.
/// The callers pass the policy's `Debug` form (`Off`, `Auto`, `Fixed(3)`) or a sentence that starts with it.
pub fn policy_label(policy: &str) -> &'static str {
    let p = policy.trim().to_ascii_lowercase();
    if p.starts_with("off") { "off" } else if p.starts_with("auto") { "auto" } else if p.starts_with("fixed") { "fixed" } else { "unknown" }
}

/// One lane's round: `offered` drafts verified, `accepted` taken, `sampled` (vs greedy), `residual` (a rejection
/// drew from the residual), `zero_mass` (the residual was empty: the first max logit).
pub fn round(offered: usize, accepted: usize, sampled: bool, residual: bool, zero_mass: bool) {
    ROUNDS.fetch_add(1, Relaxed);
    DRAFTED.fetch_add(offered as u64, Relaxed);
    ACCEPTED.fetch_add(accepted as u64, Relaxed);
    EMITTED.fetch_add(accepted as u64 + 1, Relaxed);
    if sampled { SAMPLED_ROUNDS.fetch_add(1, Relaxed); } else { GREEDY_ROUNDS.fetch_add(1, Relaxed); }
    if residual { RESIDUAL_DRAWS.fetch_add(1, Relaxed); }
    if zero_mass { ZERO_MASS.fetch_add(1, Relaxed); }
    if let Ok(mut a) = ACC_POS.lock() { for i in 0..offered.min(super::BLOCK) { a[i].1 += 1; if i < accepted { a[i].0 += 1; } } }
    if let Ok(mut h) = LEN_HIST.lock() { h[offered.min(super::BLOCK)] += 1; }
}

/// Account the round results of `lanes` (read back after the verify): offered = k, accepted = n_acc, a residual
/// draw when a sampled lane rejected, an empty residual when the finish took the first max logit.
pub fn record_lanes(lanes: &[super::Lane]) {
    for l in lanes.iter().filter(|l| l.active != 0) {
        let sampled = l.inv_t > 0.0;
        round(l.k.max(0) as usize, l.n_acc.max(0) as usize, sampled, sampled && l.n_acc < l.k, l.zmass != 0);
    }
}

fn row(o: &mut String, name: &str, help: &str, kind: &str, v: u64) {
    o.push_str(&format!("# HELP {name} {help}\n# TYPE {name} {kind}\n{name} {v}\n"));
}

/// The section (empty unless a round ran or a source was announced in this process).
pub fn render(o: &mut String) {
    let src = SOURCE.lock().ok().and_then(|s| s.clone());
    if src.is_none() && ROUNDS.load(Relaxed) == 0 { return; }
    o.push_str("# HELP velogb10_dsv41_spec_source DeepSeek-V4.1 speculation source of this process (1 = in use).\n# TYPE velogb10_dsv41_spec_source gauge\n");
    o.push_str(&format!("velogb10_dsv41_spec_source{{source=\"{}\"}} 1\n", src.as_deref().unwrap_or("dspark")));
    if let Some(p) = POLICY.lock().ok().and_then(|p| *p) {
        o.push_str("# HELP velogb10_dsv41_spec_policy DeepSeek-V4.1 draft-length policy of this boot (off = speculation off: --mtp off or --dsv41-draft-len 0).\n# TYPE velogb10_dsv41_spec_policy gauge\n");
        o.push_str(&format!("velogb10_dsv41_spec_policy{{policy=\"{p}\"}} 1\n"));
    }
    row(o, "velogb10_dsv41_spec_rounds_total", "DSpark rounds (one per lane per verify).", "counter", ROUNDS.load(Relaxed));
    row(o, "velogb10_dsv41_spec_drafted_total", "Drafts verified.", "counter", DRAFTED.load(Relaxed));
    row(o, "velogb10_dsv41_spec_accepted_total", "Drafts accepted by the verify.", "counter", ACCEPTED.load(Relaxed));
    row(o, "velogb10_dsv41_spec_emitted_total", "Tokens emitted by DSpark rounds (accepted + one bonus / correction per round).", "counter", EMITTED.load(Relaxed));
    row(o, "velogb10_dsv41_spec_sampled_rounds_total", "Rounds of sampled requests (DeepSeek's rule).", "counter", SAMPLED_ROUNDS.load(Relaxed));
    row(o, "velogb10_dsv41_spec_greedy_rounds_total", "Rounds of greedy requests (exact argmax).", "counter", GREEDY_ROUNDS.load(Relaxed));
    row(o, "velogb10_dsv41_spec_residual_draws_total", "Rejections that drew from the residual max(q - alpha p, 0).", "counter", RESIDUAL_DRAWS.load(Relaxed));
    row(o, "velogb10_dsv41_spec_zero_mass_total", "Rejections whose residual was empty (the target's first max logit was taken).", "counter", ZERO_MASS.load(Relaxed));
    row(o, "velogb10_dsv41_spec_wide_cut_rows_total", "Rows whose top-k / top-p cut was computed off the device (counted fallback).", "counter", WIDE_CUTS.load(Relaxed));
    row(o, "velogb10_dsv41_spec_varlen_rounds_total", "Rounds the confidence-scheduled variable length shortened (--dsv41-draft-varlen).", "counter", VARLEN_ROUNDS.load(Relaxed));
    if let Ok(a) = ACC_POS.lock() {
        o.push_str("# HELP velogb10_dsv41_spec_position_total Drafts offered / accepted by position in the chain (1 = the first draft).\n# TYPE velogb10_dsv41_spec_position_total counter\n");
        for (i, (acc, off)) in a.iter().enumerate() {
            o.push_str(&format!("velogb10_dsv41_spec_position_total{{position=\"{}\",outcome=\"offered\"}} {off}\nvelogb10_dsv41_spec_position_total{{position=\"{}\",outcome=\"accepted\"}} {acc}\n", i + 1, i + 1));
        }
    }
    if let Ok(h) = LEN_HIST.lock() {
        o.push_str("# HELP velogb10_dsv41_spec_draft_len_total Rounds by the draft length verified.\n# TYPE velogb10_dsv41_spec_draft_len_total counter\n");
        for (k, n) in h.iter().enumerate() { o.push_str(&format!("velogb10_dsv41_spec_draft_len_total{{k=\"{k}\"}} {n}\n")); }
    }
    render_ps1(o);
    super::mcache::render(o);   // leg PH1
}

/// Leg PS1's rows: the rule, the decisions by c, expected vs realised tokens, the round walls, the cost table.
fn render_ps1(o: &mut String) {
    if let Some(r) = RULE.lock().ok().and_then(|r| *r) {
        o.push_str("# HELP velogb10_dsv41_spec_rule How --dsv41-draft-len auto decides (conf = the baseline recipe's per-round rule from the drafter's confidences; stats = acceptance by position).\n# TYPE velogb10_dsv41_spec_rule gauge\n");
        o.push_str(&format!("velogb10_dsv41_spec_rule{{rule=\"{r}\"}} 1\n"));
    }
    if let Ok(h) = CHOSEN.lock() {
        o.push_str("# HELP velogb10_dsv41_spec_chosen_k_total Rounds by the drafts per lane the auto rule chose to verify (one decision per round whose lanes share a count; leg PS3: rounds whose lanes verify different counts are in velogb10_dsv41_spec_lane_drafts_total only).\n# TYPE velogb10_dsv41_spec_chosen_k_total counter\n");
        for (k, n) in h.iter().enumerate() { o.push_str(&format!("velogb10_dsv41_spec_chosen_k_total{{k=\"{k}\"}} {n}\n")); }
    }
    if let Ok(e) = EXP_REAL.lock() {
        o.push_str(&format!("# HELP velogb10_dsv41_spec_expected_tokens_total Tokens the auto rule expected from its decided rounds (sum over lanes of 1 + chances).\n# TYPE velogb10_dsv41_spec_expected_tokens_total counter\nvelogb10_dsv41_spec_expected_tokens_total {:.3}\n", e.0));
        row(o, "velogb10_dsv41_spec_realised_tokens_total", "Tokens the auto rule's decided rounds emitted (sum over lanes).", "counter", e.1);
        row(o, "velogb10_dsv41_spec_decided_rounds_total", "Rounds whose verify count the auto rule decided.", "counter", e.2);
    }
    row(o, "velogb10_dsv41_spec_probes_total", "Rounds the stats rule offered all 5 drafts to re-measure deep positions.", "counter", PROBES.load(Relaxed));
    if let Ok(w) = WALL.lock() {
        o.push_str(&format!("# HELP velogb10_dsv41_spec_round_wall_ms_total Wall time of the rounds that drafted (draft + choice + verify), ms.\n# TYPE velogb10_dsv41_spec_round_wall_ms_total counter\nvelogb10_dsv41_spec_round_wall_ms_total {:.3}\n", w.0));
        row(o, "velogb10_dsv41_spec_timed_rounds_total", "Rounds counted in velogb10_dsv41_spec_round_wall_ms_total.", "counter", w.1);
    }
    if let Ok(t) = TABLE.lock() {
        if !t.0.is_empty() {
            o.push_str("# HELP velogb10_dsv41_spec_round_cost_ms The round cost the auto rule prices (measured on this engine: verify of c drafts per lane + the draft step), by lanes and c.\n# TYPE velogb10_dsv41_spec_round_cost_ms gauge\n");
            for (l, k, ms) in &t.0 { o.push_str(&format!("velogb10_dsv41_spec_round_cost_ms{{lanes=\"{l}\",k=\"{k}\"}} {ms:.3}\n")); }
        }
        if !t.1.is_empty() {
            o.push_str("# HELP velogb10_dsv41_spec_overhead_ms The per-round overhead the conf rule learned on top of the table, by lanes.\n# TYPE velogb10_dsv41_spec_overhead_ms gauge\n");
            for (l, ms) in &t.1 { o.push_str(&format!("velogb10_dsv41_spec_overhead_ms{{lanes=\"{l}\"}} {ms:.3}\n")); }
        }
    }
    render_model(o);
    super::alloc::render(o);
    crate::dsv41::tp::copy::render(o);
}

/// Leg PS2's rows: the online round-cost model.
fn render_model(o: &mut String) {
    let Some(m) = MODEL.lock().ok().and_then(|m| m.clone()) else { return };
    let unix = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let src = if m.served > 0 { "served" } else if m.cached { "boot_cache" } else { "boot" };
    o.push_str("# HELP velogb10_dsv41_spec_cost_source Where the conf rule's round-cost table comes from now: served = refreshed from the served rounds' own timings (leg PS2), boot = this boot's calibration, boot_cache = a calibration cached by an earlier boot of the same binary, pack, TP size, lanes, nodes and Spmd flags.\n# TYPE velogb10_dsv41_spec_cost_source gauge\n");
    o.push_str(&format!("velogb10_dsv41_spec_cost_source{{source=\"{src}\"}} 1\n"));
    let age = m.refreshed.map_or(-1.0, |t| t.elapsed().as_secs_f64());
    o.push_str(&format!("# HELP velogb10_dsv41_spec_cost_age_seconds Seconds since a served round last refreshed the table (-1 = none yet: the boot prior), as of the last publish {:.0} s ago.\n# TYPE velogb10_dsv41_spec_cost_age_seconds gauge\nvelogb10_dsv41_spec_cost_age_seconds {age:.1}\n", m.published.elapsed().as_secs_f64()));
    o.push_str(&format!("# HELP velogb10_dsv41_spec_cost_boot_age_seconds Seconds since the boot calibration under the table was measured (a cached one keeps its first time).\n# TYPE velogb10_dsv41_spec_cost_boot_age_seconds gauge\nvelogb10_dsv41_spec_cost_boot_age_seconds {}\n", unix.saturating_sub(m.boot_unix)));
    row(o, "velogb10_dsv41_spec_cost_served_rounds_total", "Served rounds folded into the round-cost table.", "counter", m.served);
    row(o, "velogb10_dsv41_spec_cost_shifts_total", "Times the table's speed factor jumped to a new level (6 rounds in a row off by more than 15 %: a clock step, a thermal cap, a long context).", "counter", m.shifts);
    row(o, "velogb10_dsv41_spec_cost_outliers_total", "Served rounds that did not move the speed factor (more than 15 % off, no run of 6).", "counter", m.rejected);
    o.push_str(&format!("# HELP velogb10_dsv41_spec_cost_speed_factor The table's speed factor (served round walls over the fitted shape; 1 = as at boot).\n# TYPE velogb10_dsv41_spec_cost_speed_factor gauge\nvelogb10_dsv41_spec_cost_speed_factor {:.4}\n", m.speed));
    o.push_str(&format!("# HELP velogb10_dsv41_spec_cost_gap_ms The host gap between consecutive rounds priced on top of every round (median of the last 32 under 50 ms).\n# TYPE velogb10_dsv41_spec_cost_gap_ms gauge\nvelogb10_dsv41_spec_cost_gap_ms {:.3}\n", m.gap_ms));
    o.push_str("# HELP velogb10_dsv41_spec_cost_fit_ms The table's fitted line per lane count: part=fixed (a round with no draft) and part=per_draft (each draft per request), speed factor applied.\n# TYPE velogb10_dsv41_spec_cost_fit_ms gauge\n");
    for (l, a, b) in &m.fits {
        o.push_str(&format!("velogb10_dsv41_spec_cost_fit_ms{{lanes=\"{l}\",part=\"fixed\"}} {a:.3}\nvelogb10_dsv41_spec_cost_fit_ms{{lanes=\"{l}\",part=\"per_draft\"}} {b:.3}\n"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dsv41_ds_metrics_count_and_render() {
        let l = announce("dspark (test)", "auto");
        round(5, 3, true, true, false);
        round(2, 2, false, false, false);
        let mut o = String::new();
        render(&mut o);
        for name in ["velogb10_dsv41_spec_rounds_total", "velogb10_dsv41_spec_drafted_total", "velogb10_dsv41_spec_accepted_total"] {
            assert!(o.lines().any(|x| x.starts_with(&format!("{name} "))), "{name} missing:\n{o}");
        }
        assert!(o.contains("velogb10_dsv41_spec_position_total{position=\"1\",outcome=\"offered\"}"));
        // leg PS1's receipts
        set_rule(Some("conf"));
        decision(3, 3.4, 4);
        round_wall(41.0);
        publish_costs(vec![(1, 0, 30.0), (1, 3, 44.0)], vec![(1, 0.5)]);
        let mut o = String::new();
        render(&mut o);
        for want in ["velogb10_dsv41_spec_rule{rule=\"conf\"} 1", "velogb10_dsv41_spec_chosen_k_total{k=\"3\"} ", "velogb10_dsv41_spec_expected_tokens_total ",
                     "velogb10_dsv41_spec_realised_tokens_total ", "velogb10_dsv41_spec_decided_rounds_total ", "velogb10_dsv41_spec_round_wall_ms_total ",
                     "velogb10_dsv41_spec_round_cost_ms{lanes=\"1\",k=\"3\"} 44.000", "velogb10_dsv41_spec_overhead_ms{lanes=\"1\"} 0.500", "velogb10_dsv41_spec_probes_total "] {
            assert!(o.contains(want), "{want} missing:\n{o}");
        }
        assert!(o.contains("velogb10_dsv41_spec_source{source=\"dspark (test)\"} 1"));
        assert!(l.is_none() || l.unwrap().contains("speculation source"));
        assert!(announce("again", "auto").is_none(), "the boot line prints once");
        assert!(o.contains("velogb10_dsv41_spec_policy{policy=\""), "the announced policy renders:\n{o}");
    }

    /// Leg PS2: the online round-cost model's receipts (source, age, served rounds, speed factor, fit) render, and
    /// `scripts/owner_check.py` says in plain words where the table comes from and how fresh it is.
    #[test]
    fn dsv41_ps2_cost_model_rows_render_and_owner_check_reads_them() {
        use super::super::roundcost::{RoundCost, Source, NC};
        let prior: Vec<[f64; NC]> = (1..=2).map(|l| std::array::from_fn(|c| 30.0 + 4.0 * (l * (c + 1)) as f64)).collect();
        let mut m = RoundCost::new(prior, Source::Measured, 1_763_600_000);
        for i in 0..40 { m.observe(1, i % NC, 31.0 + 4.0 * (i % NC + 1) as f64); }
        m.observe_gap(0.7);
        // the rows are process-global and another test announces `off`: retry the announce + render pair until it carries `auto`
        round(5, 3, true, true, false);
        let mut o = String::new();
        for _ in 0..50 {
            let _ = announce("dspark (TP serve, leg SE)", "Auto");
            set_rule(Some("conf"));
            publish_costs(m.rows(), Vec::new());
            publish_model(&m);
            o = String::from("velogb10_dsv41_serving 1\n");
            render(&mut o);
            if o.contains("velogb10_dsv41_spec_policy{policy=\"auto\"} 1") { break; }
        }
        for want in ["velogb10_dsv41_spec_cost_source{source=\"served\"} 1", "velogb10_dsv41_spec_cost_served_rounds_total 40", "velogb10_dsv41_spec_cost_speed_factor ",
                     "velogb10_dsv41_spec_cost_fit_ms{lanes=\"1\",part=\"fixed\"} ", "velogb10_dsv41_spec_cost_fit_ms{lanes=\"2\",part=\"per_draft\"} ",
                     "velogb10_dsv41_spec_cost_age_seconds ", "velogb10_dsv41_spec_cost_boot_age_seconds ", "velogb10_dsv41_spec_cost_gap_ms 0.700"] {
            assert!(o.contains(want), "{want} missing:\n{o}");
        }
        for l in o.lines().filter(|l| !l.starts_with('#') && l.starts_with("velogb10_dsv41_spec_cost")) {
            assert!(l.rsplit(' ').next().unwrap().parse::<f64>().is_ok(), "bad sample line {l}");
        }
        let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/owner_check.py");
        if !script.exists() || std::process::Command::new("python3").arg("--version").output().is_err() { return; }
        let dir = std::env::temp_dir().join(format!("ps2_oc_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("metrics.txt");
        std::fs::write(&f, &o).unwrap();
        let out = std::process::Command::new("python3").arg("-I").arg(&script).arg("--file").arg(&f).output().expect("run owner_check.py");
        let _ = std::fs::remove_dir_all(&dir);
        let text = String::from_utf8_lossy(&out.stdout).to_string();
        assert!(out.status.success(), "{text}\n{}", String::from_utf8_lossy(&out.stderr));
        for want in ["round-cost table source: refreshed from 40 served round(s)", "speed factor", "fitted fixed + per-draft cost: 1 request(s)"] {
            assert!(text.contains(want), "owner_check lacks {want:?}:\n{text}");
        }
    }

    #[test]
    fn dsv41_int8_spec_policy_label_names_off_from_the_announced_mode() {
        // the TP serve loop announces `format!("{:?}", policy.mode)`; --mtp off / --dsv41-draft-len 0 parse to Mode::Off
        assert_eq!(policy_label(&format!("{:?}", super::super::policy::Mode::Off)), "off");
        assert_eq!(policy_label(&format!("{:?}", super::super::policy::Mode::Auto)), "auto");
        assert_eq!(policy_label(&format!("{:?}", super::super::policy::Mode::Fixed(3))), "fixed");
        assert_eq!(policy_label("auto (measured round costs, --dsv41-draft-len)"), "auto");
        assert_eq!(policy_label("dspark"), "unknown");
    }

    /// `scripts/owner_check.py` on this module's REAL rendered rows with the policy off (UXF boot B: `--mtp off`): it must say
    /// speculation is off, never "ENGAGED: N round(s)". The rows are process-global and another test announces `auto`, so the
    /// announce + render pair is retried until the render carries `off`.
    #[test]
    fn dsv41_int8_owner_check_says_speculation_is_off_from_the_policy_row() {
        let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/owner_check.py");
        if !script.exists() || std::process::Command::new("python3").arg("--version").output().is_err() {
            return; // the public repository suite has neither the script nor (maybe) python3
        }
        let mut body = String::new();
        for _ in 0..50 {
            let _ = announce("dspark (TP serve, leg SE)", &format!("{:?}", super::super::policy::Mode::Off));
            round(0, 0, true, false, false);
            body = String::from("velogb10_dsv41_serving 1\n");
            render(&mut body);
            if body.contains("velogb10_dsv41_spec_policy{policy=\"off\"} 1") { break; }
        }
        assert!(body.contains("velogb10_dsv41_spec_policy{policy=\"off\"} 1"), "{body}");
        let dir = std::env::temp_dir().join(format!("int8_oc_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("metrics.txt");
        std::fs::write(&f, &body).unwrap();
        let out = std::process::Command::new("python3").arg("-I").arg(&script).arg("--file").arg(&f).output().expect("run owner_check.py");
        let _ = std::fs::remove_dir_all(&dir);
        let text = String::from_utf8_lossy(&out.stdout).to_string();
        assert!(out.status.success(), "{text}\n{}", String::from_utf8_lossy(&out.stderr));
        assert!(text.contains("SPECULATION IS OFF on this server"), "{text}");
        assert!(!text.contains("ENGAGED:"), "speculation off must not read as engaged:\n{text}");
    }
}
