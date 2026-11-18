//! Tool-call parser shadow counting: how long the parser took, which rule was served, where the other rules would have differed, and the `render_toolcall` series.
//! Facts only: nothing here says which reading is right (S-B9-TOOLPARSE3 T1). `render_toolcall` is called from `render.rs`. Map: `mod.rs`.

use super::*;

// ─── tool-call parser shadow counting (S-B9-TOOLPARSE3 T1) ──────────────────────────────
// What the parser's OTHER rules would have returned for the text it parsed. Facts only: which
// reading returns fewer calls and which rule was served; nothing here says which reading is right.
static TC_BLOCKS: AtomicU64 = AtomicU64::new(0);
static TC_COMPARED: [AtomicU64; 4] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
static TC_DIFF: [AtomicU64; 36] = [const { AtomicU64::new(0) }; 36];
static TC_SHADOW_FAILED: AtomicU64 = AtomicU64::new(0);
static TC_SKIPPED: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
/// responses that ended inside a `<tool_call>` block that never closed: [finish stop / length / other] x
/// [the text ends the way a finished call ends: no / yes]
static TC_UNCLOSED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
pub(super) static UNKNOWN_OPTIONS: AtomicU64 = AtomicU64::new(0);
/// F7b: distinct "--"-options this head ignored at startup with one "[opts] unknown option" line
/// each (warn, never refuse; a typo used to disappear silently). Head only; nodes do not warn.
pub fn set_unknown_options(n: u64) { UNKNOWN_OPTIONS.store(n, Relaxed); }
/// Time the thread that parses a response spent in the tool-call parser, per response that held a
/// `<tool_call>` block: [served parse, shadow comparison], a histogram in microsecond buckets, its sum
/// and its worst case. On the EXL3 server that thread is the one that serves every stream, /health and
/// /metrics: this is how long they waited.
const TC_TIME_LE_US: [u64; 11] = [50, 100, 250, 500, 1_000, 2_000, 5_000, 10_000, 25_000, 100_000, 1_000_000];
static TC_TIME_BUCKETS: [[AtomicU64; 12]; 2] = [[const { AtomicU64::new(0) }; 12], [const { AtomicU64::new(0) }; 12]];
static TC_TIME_SUM_US: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
static TC_TIME_MAX_US: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
/// blocks the cost model read by the v0.7.3 rule because they exceeded its work budget: [served, shadow]
static TC_COST_DECLINED: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// One response's time in the parser: `part` 0 = the served parse, 1 = the shadow comparison.
pub fn toolcall_time(part: usize, us: u64) {
    let b = TC_TIME_LE_US.iter().position(|&le| us <= le).unwrap_or(TC_TIME_LE_US.len());
    TC_TIME_BUCKETS[part][b].fetch_add(1, Relaxed);
    TC_TIME_SUM_US[part].fetch_add(us, Relaxed);
    TC_TIME_MAX_US[part].fetch_max(us, Relaxed);
}

/// `blocks` closed `<tool_call>` blocks were read by the served parse of one response (counted
/// whether or not the shadow comparison is made, and before it runs).
pub fn toolcall_blocks(blocks: usize) {
    if blocks > 0 { TC_BLOCKS.fetch_add(blocks as u64, Relaxed); }
}

/// A response ended inside a block the model never closed (F8). `outcome`: "returned" (the model
/// ended by itself, finish stop, and function-closed elements came back as calls), "dropped_cut"
/// (the model ended by itself but no complete element was there to return), "dropped_length" (the
/// token budget, the client or the engine cut the response: dropped whole, as v0.7.3).
pub fn toolcall_unclosed_outcome(outcome: &str) {
    let i = match outcome { "returned" => 0, "dropped_cut" => 1, _ => 2 };
    TC_UNCLOSED[i].fetch_add(1, Relaxed);
}

/// The shadow comparison was not made, or was cut short: `size` (text over `tools::SHADOW_MAX_BYTES`),
/// `blocks` (more than `tools::SHADOW_MAX_BLOCKS`), `time` (its wall-clock budget was spent).
pub fn toolcall_shadow_skipped(reason: &str) {
    let i = match reason { "size" => 0, "blocks" => 1, _ => 2 };
    TC_SKIPPED[i].fetch_add(1, Relaxed);
}

/// The cost model read a SERVED block by the v0.7.3 rule because it exceeded its work budget (events,
/// states or work units).
pub fn toolcall_cost_declined() { TC_COST_DECLINED[0].fetch_add(1, Relaxed); }

/// The shadow comparison panicked (a defect in a rule that is not served); the served result was
/// returned unchanged.
pub fn toolcall_shadow_failed() {
    TC_SHADOW_FAILED.fetch_add(1, Relaxed);
    crate::reprintln!("[toolcall-shadow] the comparison panicked; the served result is unaffected");
}

fn tc_pairing_name(p: crate::tools::QuotePairing) -> &'static str {
    match p { crate::tools::QuotePairing::Off => "off", crate::tools::QuotePairing::On => "on", crate::tools::QuotePairing::Cost => "cost" }
}

/// Count one response's shadow comparison and log one content-free line per disagreement.
pub fn toolcall_shadow(sh: &crate::tools::Shadow, served: &crate::tools::Rules) {
    if sh.cost_declined > 0 { TC_COST_DECLINED[1].fetch_add(sh.cost_declined as u64, Relaxed); }
    for (i, n) in sh.compared.iter().enumerate() { if *n > 0 { TC_COMPARED[i].fetch_add(*n as u64, Relaxed); } }
    for (n, d) in sh.diffs.iter().enumerate() {
        TC_DIFF[d.other.idx() * 9 + d.kind.idx() * 3 + d.fewer.idx()].fetch_add(1, Relaxed);
        // one content-free line per disagreement, at most 8 per response (the counters count them all)
        if n == 8 { crate::reprintln!("[toolcall-shadow] {} more disagreements of this response are counted, not logged", sh.diffs.len() - 8); }
        if n >= 8 { continue; }
        crate::reprintln!("[toolcall-shadow] block={} len={} served={}/{} other={} kind={} calls_served={} calls_other={} fewer={}",
                          d.hash, d.len, tc_pairing_name(served.pairing),
                          if served.block == crate::tools::BlockEnd::First { "first" } else { "after-function" },
                          d.other.as_str(), d.kind.as_str(), d.calls_served, d.calls_other, d.fewer.as_str());
    }
}

pub(super) fn render_toolcall(o: &mut String) {
    use crate::tools::{DiffKind, Fewer, Other};
    let served = crate::tools::Rules::served();
    o.push_str("# HELP velogb10_toolcall_rules_info The tool-call parser rules this server serves (value is always 1): pairing = --tool-quote-pairing (how the end of an argument value is found), block = --tool-block-end (how a <tool_call> block ends), unclosed = --tool-unclosed (what a block that never closed does when the model stopped by itself).\n# TYPE velogb10_toolcall_rules_info gauge\n");
    o.push_str(&format!("velogb10_toolcall_rules_info{{pairing=\"{}\",block=\"{}\",unclosed=\"{}\"}} 1\n", tc_pairing_name(served.pairing),
                        if served.block == crate::tools::BlockEnd::First { "first" } else { "after-function" },
                        if served.unclosed == crate::tools::Unclosed::Complete { "complete" } else { "drop" }));
    o.push_str("# HELP velogb10_toolcall_shadow_failed_total Shadow comparisons that panicked (a defect in a parser rule that is not served; the served result was returned unchanged). Expected 0.\n# TYPE velogb10_toolcall_shadow_failed_total counter\n");
    o.push_str(&format!("velogb10_toolcall_shadow_failed_total {}\n", TC_SHADOW_FAILED.load(Relaxed)));
    o.push_str("# HELP velogb10_toolcall_shadow_skipped_total Responses whose shadow comparison was not made or was cut short, by reason: size (text over 512 KiB), blocks (more than 64 <tool_call> blocks), time (the comparison's 100 ms wall-clock budget was spent between two rules). The served result is unaffected. The default comparison reads two cheap rules (quote pairing and the block rule); the cost model is read only under --tool-shadow-cost on, and has its own work budget (see cost_declined).\n# TYPE velogb10_toolcall_shadow_skipped_total counter\n");
    for (i, r) in ["size", "blocks", "time"].iter().enumerate() {
        o.push_str(&format!("velogb10_toolcall_shadow_skipped_total{{reason=\"{r}\"}} {}\n", TC_SKIPPED[i].load(Relaxed)));
    }
    o.push_str(&format!("# HELP velogb10_toolcall_cost_declined_total Blocks the cost model (--tool-quote-pairing cost, or --tool-shadow-cost on) did not read because they exceeded its work budget (events, states or work units: {} per block and {} per response, counted work and not a clock): they were read by the v0.7.3 rule. role=served: blocks of served responses, where the v0.7.3 reading is what the client got; role=shadow: blocks the comparison skipped. A block counted here is a block on which cost and v0.7.3 agree by construction.\n# TYPE velogb10_toolcall_cost_declined_total counter\n",
                        crate::tools::cost::BLOCK_UNITS, crate::tools::cost::RESPONSE_UNITS));
    o.push_str(&format!("velogb10_toolcall_cost_declined_total{{role=\"served\"}} {}\n", TC_COST_DECLINED[0].load(Relaxed)));
    o.push_str(&format!("velogb10_toolcall_cost_declined_total{{role=\"shadow\"}} {}\n", TC_COST_DECLINED[1].load(Relaxed)));
    o.push_str("# HELP velogb10_toolcall_blocks_total <tool_call> blocks (closed) read by the served parse of finished responses that held at least one. Counted for every such response, whether or not the shadow comparison was made.\n# TYPE velogb10_toolcall_blocks_total counter\n");
    o.push_str(&format!("velogb10_toolcall_blocks_total {}\n", TC_BLOCKS.load(Relaxed)));
    o.push_str("# HELP velogb10_toolcall_parse_seconds Time the thread that parses a response spent in the tool-call parser, per response that held a <tool_call> block. part=served: the parse that is returned; part=shadow: the comparison with the rules that are not served. On the EXL3 server this thread serves every stream, /health and /metrics: it is how long they waited. Histogram, with the worst case below.\n# TYPE velogb10_toolcall_parse_seconds histogram\n");
    for (pi, part) in ["served", "shadow"].iter().enumerate() {
        let mut cum = 0u64;
        for (bi, le) in TC_TIME_LE_US.iter().enumerate() {
            cum += TC_TIME_BUCKETS[pi][bi].load(Relaxed);
            o.push_str(&format!("velogb10_toolcall_parse_seconds_bucket{{part=\"{part}\",le=\"{}\"}} {cum}\n", *le as f64 / 1e6));
        }
        cum += TC_TIME_BUCKETS[pi][TC_TIME_LE_US.len()].load(Relaxed);
        o.push_str(&format!("velogb10_toolcall_parse_seconds_bucket{{part=\"{part}\",le=\"+Inf\"}} {cum}\n"));
        o.push_str(&format!("velogb10_toolcall_parse_seconds_sum{{part=\"{part}\"}} {}\n", TC_TIME_SUM_US[pi].load(Relaxed) as f64 / 1e6));
        o.push_str(&format!("velogb10_toolcall_parse_seconds_count{{part=\"{part}\"}} {cum}\n"));
    }
    o.push_str("# HELP velogb10_toolcall_parse_seconds_max The longest single response in the tool-call parser since the process started, by part (see velogb10_toolcall_parse_seconds).\n# TYPE velogb10_toolcall_parse_seconds_max gauge\n");
    for (pi, part) in ["served", "shadow"].iter().enumerate() {
        o.push_str(&format!("velogb10_toolcall_parse_seconds_max{{part=\"{part}\"}} {}\n", TC_TIME_MAX_US[pi].load(Relaxed) as f64 / 1e6));
    }
    o.push_str("# HELP velogb10_toolcall_unclosed_total Responses that ended inside a <tool_call> block the model never closed. outcome = returned: the model ended by itself (finish stop) and function elements that were closed inside the wrapper came back as calls (--tool-unclosed complete); dropped_cut: the model ended by itself but nothing complete was there to return (cut inside an argument, or a JSON body); dropped_length: the token budget, the client or the engine cut the response (dropped whole, as v0.7.3, and as EVERY finish under --tool-unclosed drop).\n# TYPE velogb10_toolcall_unclosed_total counter\n");
    for (i, name) in [(0usize, "returned"), (1, "dropped_cut"), (2, "dropped_length")].iter() {
        o.push_str(&format!("velogb10_toolcall_unclosed_total{{outcome=\"{name}\"}} {}\n", TC_UNCLOSED[*i].load(Relaxed)));
    }
    o.push_str("# HELP velogb10_toolcall_shadow_compared_total Blocks (other=\"block_end\": responses) that were also read under the rule named by other, which is NOT the served one. other=off / on / cost are the --tool-quote-pairing settings, other=block_end is the --tool-block-end setting that is not served. The comparison changes nothing that is returned.\n# TYPE velogb10_toolcall_shadow_compared_total counter\n");
    for other in Other::ALL {
        o.push_str(&format!("velogb10_toolcall_shadow_compared_total{{other=\"{}\"}} {}\n", other.as_str(), TC_COMPARED[other.idx()].load(Relaxed)));
    }
    o.push_str("# HELP velogb10_toolcall_parse_disagreement_total Blocks (other=\"block_end\": responses) where the rule named by other, read the same text differently from the rule that was served. kind=call_count: a different number of calls; kind=arg_boundary: the same number of calls, an argument or name differs; kind=content: the same calls, the prose differs. fewer says which of the two readings returned FEWER calls (served / other / same). It is a fact about the two readings; the server cannot know which one is right, and a nonzero count means only that the text was ambiguous under the rules.\n# TYPE velogb10_toolcall_parse_disagreement_total counter\n");
    for other in Other::ALL {
        for kind in DiffKind::ALL {
            for fewer in Fewer::ALL {
                o.push_str(&format!("velogb10_toolcall_parse_disagreement_total{{other=\"{}\",kind=\"{}\",fewer=\"{}\"}} {}\n",
                                    other.as_str(), kind.as_str(), fewer.as_str(),
                                    TC_DIFF[other.idx() * 9 + kind.idx() * 3 + fewer.idx()].load(Relaxed)));
            }
        }
    }
}
