//! DSV4.1 loader (leg LS; leg MEM1 made it predict what the TP forward really allocates): the MEMORY-FIT CHECK before
//! the load (UX row 7: graceful caps, never a refusal under load). Rank 0's resident bytes, with the forward's own
//! allocation formulas (`tp::mem`):
//!
//! ```text
//!   weights (the rank's selection from the shard plan)
//! + lanes x lane state(max_seq_len, chunk)          (`mem::lane_bytes`: windows, KV-source caches, frontier, latent rows)
//! + 2 activation arenas (512 KiB per chunk row, >= 256 MiB each) + the frames and fixed buffers (`mem::fixed_bytes`)
//! + the MoE / dense scratch at the chunk (`mem::moe_scratch_est`)
//! + the DSpark round, the Engram row cache (pinned) and its host index, the captured graphs (`mem::GRAPH_CAP` of them)
//! + the CUDA context, modules and the process's host memory (`mem::PROCESS_EST`)
//!   <= budget = what this box can give the process (MemAvailable + half of the swappable pages the free swap can take)
//!               - the memory watchdog's floor (--mem-watchdog-gb) - 0.5 GiB
//! ```
//!
//! The prefix pool is NOT in the sum: it is the flexible part. The lane loop's governor (`tp::mem::Governor`) gives it
//! what is free above the floor plus a reserve, and takes it back before the floor; the notice says how much it can expect.
//! When the asked posture does not fit it is CAPPED in this order, each step only as far as needed: the Engram row cache
//! (floor 512 MiB), the prefill chunk (floor 1,024 rows), the lanes (floor 1), the context (floor 8,192 tokens, in steps
//! of 1/16 of the ask). The boot prints ONE notice line with what was asked, what runs and the arithmetic, and counts it on
//! `/metrics`. Only when even the floor does not fit is there nothing to serve: the error names the shortfall and the fix
//! (a wider TP).

use crate::dsv41::config::Dsv41Config;
use crate::dsv41::tp::mem;
use anyhow::{bail, Result};

pub const GIB: u64 = 1 << 30;
/// What the budget keeps besides the floor (the arithmetic's own error).
pub const MARGIN: u64 = GIB / 2;
/// A box's other processes and kernel when the box is not this one (the dry run's planning figure).
pub const OTHERS: u64 = 8 * GIB;
const FLOOR_CHUNK: usize = 1024;
const FLOOR_CTX: usize = 8192;

#[derive(Clone, Debug)]
pub struct FitInput {
    pub weights: u64,
    pub lanes: usize,
    pub max_seq_len: usize,
    pub chunk: usize,
    pub engram_cache_mib: u64,
    pub mem_total: u64,
    /// the TP world (the MoE scratch is at the rank's inner width; the Engram output at its heads)
    pub world: usize,
    /// what this box can give the process at boot (MemAvailable + the swap the kernel can move other pages to)
    pub avail: u64,
    /// the memory watchdog's floor
    pub floor: u64,
    /// the prefix pool's asked budget (reported, never capped here: the governor sizes it at run time)
    pub pool: u64,
}

impl FitInput {
    /// A posture planned for a box that is not this one (the dry run): its memory minus `OTHERS`, the default floor.
    pub fn planned(weights: u64, lanes: usize, max_seq_len: usize, chunk: usize, engram_cache_mib: u64, mem_total: u64, world: usize) -> FitInput {
        FitInput { weights, lanes, max_seq_len, chunk, engram_cache_mib, mem_total, world, avail: mem_total.saturating_sub(OTHERS), floor: 5_000_000_000, pool: 0 }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Fit {
    pub lanes: usize,
    pub max_seq_len: usize,
    pub chunk: usize,
    pub engram_cache_mib: u64,
    pub total: u64,
    /// the same sum at the asked posture (before any cap)
    pub asked_total: u64,
    pub budget: u64,
    /// what was lowered: (what, asked, served)
    pub caps: Vec<(&'static str, u64, u64)>,
}

/// The Engram output columns a rank owns (n-gram orders x its heads).
fn n_own(cfg: &Dsv41Config, world: usize) -> usize { (cfg.engram_max_ngram.saturating_sub(1)) * cfg.engram_heads.div_ceil(world.max(1)).max(1) }

/// Bytes of the serving state besides the weights and the lanes at a chunk of `chunk` rows: arenas, frames and fixed
/// buffers, scratch, the DSpark round, the Engram cache and index, the graphs, the process.
pub fn workspace(cfg: &Dsv41Config, world: usize, chunk: usize, cache_mib: u64) -> u64 {
    let rows = chunk.max(crate::dsv41::tp::fwd::HEAD_MAX);
    let (pinned, host) = mem::engram_bytes(cache_mib);
    2 * mem::arena_bytes(rows) as u64 + mem::fixed_bytes(cfg, rows, n_own(cfg, world)) + mem::moe_scratch_est(cfg, world, rows) + mem::SPEC_EST
        + pinned + host + mem::GRAPH_CAP as u64 * mem::GRAPH_EST + mem::PROCESS_EST
}

/// Bytes of the resident set under `i` (the arithmetic above; the prefix pool excluded).
pub fn total(cfg: &Dsv41Config, i: &FitInput, lanes: usize, ctx: usize, chunk: usize, cache_mib: u64) -> u64 {
    let rows = chunk.max(crate::dsv41::tp::fwd::HEAD_MAX);
    i.weights + lanes as u64 * mem::lane_bytes(cfg, rows, ctx) + workspace(cfg, i.world, chunk, cache_mib)
}

/// What the budget allows: what the box can give minus the floor and the margin.
pub fn budget_of(i: &FitInput) -> u64 { i.avail.saturating_sub(i.floor).saturating_sub(MARGIN) }

/// Fit `i` into the box, capping as the module doc says. Err only when nothing is left to cap.
pub fn check(cfg: &Dsv41Config, i: &FitInput) -> Result<Fit> {
    let budget = budget_of(i);
    let (mut lanes, mut ctx, mut chunk, mut cache) = (i.lanes.max(1), i.max_seq_len, i.chunk, i.engram_cache_mib);
    let mut caps = Vec::new();
    let t = |l, c, k, m| total(cfg, i, l, c, k, m);
    let asked_total = t(lanes, ctx, chunk, cache);
    if t(lanes, ctx, chunk, cache) > budget {
        // 1. the Engram row cache
        let want = cache;
        while cache > mem::CACHE_FLOOR_MIB && t(lanes, ctx, chunk, cache) > budget { cache = (cache / 2).max(mem::CACHE_FLOOR_MIB); }
        if cache != want { caps.push(("engram-row-cache-mib", want, cache)); }
        // 2. the prefill chunk
        let want = chunk;
        while chunk > FLOOR_CHUNK && t(lanes, ctx, chunk, cache) > budget { chunk = (chunk / 2).max(FLOOR_CHUNK); }
        if chunk != want { caps.push(("prefill-chunk", want as u64, chunk as u64)); }
        // 3. the lanes
        let want = lanes;
        while lanes > 1 && t(lanes, ctx, chunk, cache) > budget { lanes -= 1; }
        if lanes != want { caps.push(("max-batch", want as u64, lanes as u64)); }
        // 4. the context
        let want = ctx;
        let step = (want / 16).max(1024);
        while ctx > FLOOR_CTX && t(lanes, ctx, chunk, cache) > budget { ctx = (ctx.saturating_sub(step)).max(FLOOR_CTX); }
        if ctx != want { caps.push(("max-seq-len", want as u64, ctx as u64)); }
        let used = t(lanes, ctx, chunk, cache);
        if used > budget {
            bail!("the weights of this rank ({:.1} GiB) plus the smallest serving state ({:.1} GiB: one lane at {} tokens, a {} row chunk, a {} MiB Engram cache) are {:.1} GiB; this box can give {:.1} GiB ({:.1} GiB available - the {:.1} GB memory-watchdog floor - {:.1} GiB margin). \
Nothing is left to cap: run on a wider TP (each doubling halves the weights per rank) or a smaller pack",
                  i.weights as f64 / GIB as f64, (used - i.weights) as f64 / GIB as f64, ctx, chunk, cache, used as f64 / GIB as f64, budget as f64 / GIB as f64,
                  i.avail as f64 / GIB as f64, i.floor as f64 / 1e9, MARGIN as f64 / GIB as f64);
        }
    }
    Ok(Fit { lanes, max_seq_len: ctx, chunk, engram_cache_mib: cache, total: t(lanes, ctx, chunk, cache), asked_total, budget, caps })
}

impl Fit {
    /// What remains for the prefix pool under the budget.
    pub fn pool_room(&self) -> u64 { self.budget.saturating_sub(self.total) }
    /// The one line a boot prints (None when nothing was capped).
    pub fn notice(&self, asked: &FitInput) -> Option<String> {
        if self.caps.is_empty() { return None; }
        let parts: Vec<String> = self.caps.iter().map(|(w, a, s)| format!("--{w} {a} -> {s}")).collect();
        Some(format!("memory fit: the asked posture needs {:.1} GiB of the {:.1} GiB this box can give ({:.1} GiB available - the {:.1} GB memory-watchdog floor - {:.1} GiB margin); \
capped {} so the boot needs {:.1} GiB; the prefix pool gets what is free at run time (about {:.1} GiB now)",
                     self.asked_total as f64 / GIB as f64, self.budget as f64 / GIB as f64, asked.avail as f64 / GIB as f64, asked.floor as f64 / 1e9,
                     MARGIN as f64 / GIB as f64, parts.join(", "), self.total as f64 / GIB as f64, self.pool_room() as f64 / GIB as f64))
    }
    /// The breakdown line (debug level / dry run).
    pub fn line(&self, cfg: &Dsv41Config, i: &FitInput) -> String {
        let rows = self.chunk.max(crate::dsv41::tp::fwd::HEAD_MAX);
        let kv = mem::lane_bytes(cfg, rows, self.max_seq_len);
        format!("weights {:.1} + {} lane(s) x {:.2} GiB ({} tokens, {} row chunk) + workspace {:.2} (arenas, frames, scratch, DSpark, Engram cache {} MiB + index, {} graphs, process) = {:.1} GiB \
of the {:.1} GiB budget ({:.1} GiB available, {:.1} GiB memory); prefix pool room {:.1} GiB (asked {:.1} GiB)",
                i.weights as f64 / GIB as f64, self.lanes, kv as f64 / GIB as f64, self.max_seq_len, rows, workspace(cfg, i.world, self.chunk, self.engram_cache_mib) as f64 / GIB as f64,
                self.engram_cache_mib, mem::GRAPH_CAP, self.total as f64 / GIB as f64, self.budget as f64 / GIB as f64, i.avail as f64 / GIB as f64, i.mem_total as f64 / GIB as f64,
                self.pool_room() as f64 / GIB as f64, i.pool as f64 / GIB as f64)
    }
}

/// This box's MemTotal in bytes (`/proc/meminfo`); None when it cannot be read.
pub fn mem_total() -> Option<u64> {
    let t = std::fs::read_to_string("/proc/meminfo").ok()?;
    let l = t.lines().find(|l| l.starts_with("MemTotal:"))?;
    Some(l.split_whitespace().nth(1)?.parse::<u64>().ok()? * 1024)
}

/// Leg PFX2: default per-rank Engram row cache when neither `--engram-row-cache-mib` nor `--dsv41-engram-ram-gib` was passed:
/// 1024 MiB at TP2 (frees +1.16 GiB on Rank 0, doubling `PrefixPool` room to ~2.0 GB and leaving safe UMA margin for Vision),
/// 2048 MiB at TP1 / TP4 / TP8.
pub fn default_engram_cache_mib(world: usize) -> u64 { if world == 2 { 1024 } else { 2048 } }

/// Leg P6-7 / PFX2: size the host Engram row cache (`--dsv41-engram-ram-gib unset|auto|<GiB>`) when `--engram-row-cache-mib`
/// was not explicitly passed. Default (`unset`) uses `default_engram_cache_mib(world)` (1024 MiB at TP2, 2048 MiB otherwise)
/// so GB10 unified LPDDR5X retains full UMA / page-cache headroom and never swaps; explicit `<GiB>` requests that size
/// (subject to `fit::check`), and `"auto"` expands only from non-swap `MemAvailable` above a 48 GiB UMA safety floor.
#[allow(clippy::too_many_arguments)]
pub fn resolve_engram_cache_mib(dir: &std::path::Path, world: usize, lanes: usize, ctx: usize, chunk: usize,
                                explicit_mib: Option<u64>, ram_gib: Option<&str>) -> u64 {
    if let Some(m) = explicit_mib { return m; }
    let def = default_engram_cache_mib(world);
    let Some(spec) = ram_gib.map(str::trim).filter(|s| !s.is_empty() && *s != "unset" && *s != "off") else {
        return def;
    };
    if spec != "auto" {
        if let Ok(g) = spec.parse::<f64>() {
            if g.is_finite() && g >= 0.0 { return (g * 1024.0).round() as u64; }
        }
        return def;
    }
    let Ok(c) = super::ship::census_of(dir) else { return def };
    let Ok(o) = super::ship::plan_opts(&c, world, &super::ship::deal()) else { return def };
    let Ok(sel) = super::plan::select(&c, &o, 0) else { return def };
    let weights = loaded_bytes(&c, &sel);
    let Some(mem_total) = mem_total() else { return def };
    let Some(avail) = mem::avail() else { return def };
    let inp = FitInput { weights, lanes, max_seq_len: ctx, chunk, engram_cache_mib: 0, mem_total, world, avail, floor: mem::floor(), pool: 0 };
    let base = total(&c.cfg, &inp, lanes.max(1), ctx, chunk, 0);
    let safe = avail.saturating_sub(48 * GIB).min(budget_of(&inp));
    let room_mib = (safe.saturating_sub(base) / (1024 * 1024)) * 85 / 100;
    let local_heads_mib = 110_184u64 / world.max(1) as u64;
    room_mib.min(local_heads_mib).max(def)
}

/// The boot-time check (the serve path calls this, LS2): rank 0's weights from the shard plan of `dir` at `world`,
/// what this box can give the process now, the asked posture. Never a refusal: `Err` carries ONE line (the pack cannot
/// be planned, or nothing is left to cap); the caller prints it and goes on. Records the result on `/metrics`.
#[allow(clippy::too_many_arguments)]
pub fn boot_check(dir: &std::path::Path, world: usize, lanes: usize, ctx: usize, chunk: usize, cache_mib: u64, pool: u64) -> Result<(Fit, FitInput, Dsv41Config)> {
    use anyhow::Context;
    let c = super::ship::census_of(dir)?;
    let o = super::ship::plan_opts(&c, world, &super::ship::deal())?;
    let sel = super::plan::select(&c, &o, 0).context("the shard plan of rank 0")?;
    let weights = loaded_bytes(&c, &sel);
    let mem_total = mem_total().context("reading /proc/meminfo")?;
    let avail = mem::avail().context("reading /proc/meminfo")? + mem::swap_reclaimable();
    let inp = FitInput { weights, lanes, max_seq_len: ctx, chunk, engram_cache_mib: cache_mib, mem_total, world, avail, floor: mem::floor(), pool };
    let f = check(&c.cfg, &inp)?;
    super::metrics_rows::fit(lanes, f.lanes, ctx, f.max_seq_len, f.caps.len(), f.total, f.budget);
    mem::set_predicted(f.total);
    Ok((f, inp, c.cfg.clone()))
}

/// The bytes of a rank's selection the TP serve loads: the trunk layers, the DSpark stages, the embedding / head / final
/// norm, and (on rank 0 when `--dsv41-vision` is not `off`) the vision tower, aligner and delimiter embeddings.
pub fn loaded_bytes(c: &crate::dsv41::census::Census, sel: &super::plan::RankSelection) -> u64 {
    use crate::dsv41::census::classify::TensorClass as C;
    let vis_on = sel.rank == 0 && crate::dsv41::vision::vision_enabled(&c.cfg);
    sel.holds.iter().filter(|h| {
        let e = &c.entries[h.entry as usize];
        match e.layer {
            Some(l) => (l as usize) < c.cfg.total_layers() && e.class != C::EngramTable,
            None => matches!(e.class, C::Embed | C::Head | C::FinalNorm) || (vis_on && crate::dsv41::vision::is_vision_global_class(e.class)),
        }
    }).map(|h| h.bytes).sum()
}

/// The default prefill chunk (T0 §7: 16K rows, 8K at TP2).
pub fn default_chunk(world: usize) -> usize { if world <= 2 { 8192 } else { 16384 } }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsv41::config::{test_config_json, Dsv41Config};

    fn cfg() -> Dsv41Config { Dsv41Config::from_value(&test_config_json(), None).unwrap() }
    fn inp(weights_gib: f64, lanes: usize, ctx: usize, chunk: usize, avail_gib: f64) -> FitInput {
        FitInput { weights: (weights_gib * GIB as f64) as u64, lanes, max_seq_len: ctx, chunk, engram_cache_mib: 2048, mem_total: 122 * GIB, world: 2,
                   avail: (avail_gib * GIB as f64) as u64, floor: 5_000_000_000, pool: 2 * GIB }
    }

    #[test]
    fn dsv41_mem1_the_sum_is_the_forwards_own_allocations() {
        let c = cfg();
        let i = inp(99.0, 4, 262144, 2048, 120.0);
        let f = check(&c, &i).unwrap();
        let want = i.weights + 4 * mem::lane_bytes(&c, 2048, 262144) + 2 * mem::arena_bytes(2048) as u64 + mem::fixed_bytes(&c, 2048, n_own(&c, 2))
            + mem::moe_scratch_est(&c, 2, 2048) + mem::SPEC_EST + mem::engram_bytes(2048).0 + mem::engram_bytes(2048).1 + mem::GRAPH_CAP as u64 * mem::GRAPH_EST + mem::PROCESS_EST;
        assert_eq!(f.total, want);
        assert!(f.caps.is_empty(), "{:?}", f.caps);
        assert_eq!(f.budget, budget_of(&i));
        assert_eq!(f.pool_room(), f.budget - f.total);
    }

    #[test]
    fn a_tight_posture_is_capped_in_order_and_never_refused() {
        let c = cfg();
        // little room: the cache goes first, then the chunk, then lanes
        let i = inp(99.0, 4, 1 << 20, 4096, 110.0);
        let f = check(&c, &i).unwrap();
        assert!(f.total <= f.budget);
        assert_eq!(f.caps[0].0, "engram-row-cache-mib", "{:?}", f.caps);
        assert!(f.engram_cache_mib >= mem::CACHE_FLOOR_MIB);
        assert!(f.lanes >= 1 && f.max_seq_len >= FLOOR_CTX && f.chunk >= FLOOR_CHUNK);
        let n = f.notice(&i).unwrap();
        assert!(n.contains("--engram-row-cache-mib 2048 ->") && n.contains("capped") && n.contains("memory-watchdog floor"), "{n}");
        // the floor moves the budget one for one
        let mut j = i.clone();
        j.floor = 0;
        assert_eq!(budget_of(&j), budget_of(&i) + 5_000_000_000);
    }

    #[test]
    fn nothing_left_to_cap_names_the_fix() {
        let c = cfg();
        let e = check(&c, &inp(119.0, 4, 1 << 20, 8192, 120.0)).unwrap_err().to_string();
        assert!(e.contains("wider TP") && e.contains("Nothing is left to cap"), "{e}");
    }

    #[test]
    fn mem_total_reads() { assert!(mem_total().unwrap() > GIB); }

    #[test]
    fn engram_cache_defaults_to_1024_at_tp2_2048_otherwise_and_honors_explicit_gib() {
        let p = std::path::Path::new("/nonexistent");
        assert_eq!(resolve_engram_cache_mib(p, 2, 4, 262144, 2048, None, None), 1024);
        assert_eq!(resolve_engram_cache_mib(p, 2, 4, 262144, 2048, None, Some("unset")), 1024);
        for w in [1usize, 4, 8] {
            assert_eq!(resolve_engram_cache_mib(p, w, 4, 262144, 2048, None, None), 2048, "world {w}");
            assert_eq!(resolve_engram_cache_mib(p, w, 4, 262144, 2048, None, Some("unset")), 2048, "world {w}");
        }
        assert_eq!(resolve_engram_cache_mib(p, 2, 4, 262144, 2048, None, Some("8")), 8192);
        assert_eq!(resolve_engram_cache_mib(p, 2, 4, 262144, 2048, Some(2048), None), 2048);
        assert_eq!(resolve_engram_cache_mib(p, 4, 4, 262144, 2048, None, Some("8")), 8192);
        assert_eq!(resolve_engram_cache_mib(p, 4, 4, 262144, 2048, Some(4096), Some("8")), 4096);
    }
}
