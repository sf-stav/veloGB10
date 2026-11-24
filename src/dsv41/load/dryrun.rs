//! DSV4.1 loader (leg LS): `--dsv41-shard-plan`, the dry run of the whole loader on a pack's headers: the
//! tensors each rank of each posture holds (bytes by kind), the Engram heads and their disk bytes, the proofs
//! (every tensor held by someone; a cut tensor's slices tile its bytes exactly once; the unique bytes equal the
//! census), the comparison with the census' preliminary per-rank table, and the memory-fit arithmetic.

use super::engram;
use super::fit::{self, FitInput, GIB};
use super::plan::{self, Bucket};
use super::ship;
use crate::dsv41::census::classify::TensorClass;
use crate::dsv41::census::report::rank_bytes;
use crate::dsv41::rankmap::{self, Posture};
use anyhow::Result;
use std::fmt::Write as _;
use std::path::Path;

fn gb(b: u64) -> String { format!("{:.2}", b as f64 / 1e9) }
fn gib(b: u64) -> String { format!("{:.1}", b as f64 / GIB as f64) }

/// One posture's table and its verdict (`ok` = every proof held).
pub fn posture_report(c: &crate::dsv41::census::Census, world: usize, deal: &str, lanes: usize, seq: usize, cache_mib: u64) -> Result<(String, bool)> {
    let o = ship::plan_opts(c, world, deal)?;
    let (uniq, held, sels) = plan::check_all(c, &o)?;
    let census_total: u64 = c.entries.iter().filter(|e| e.class != TensorClass::EngramTable).map(|e| e.bytes).sum();
    let mut s = String::new();
    let post = Posture::from_world(world);
    let _ = writeln!(s, "\n### TP{world}  (expert split {}, attention whole: {}, Engram wkv whole: {}, embedding on rank 0 only: {})\n", o.expert_split.name(), o.attn_whole, o.engram_wkv_whole, o.embed_rank0_only);
    let _ = writeln!(s, "| rank | box | replicated GB | split GB | experts GB | rank-0 only GB | **shard GB** | Engram heads | Engram disk GB | prelim. census GB | fit (lanes / ctx / chunk) |\n|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---|");
    let prelim = post.map(|p| rank_bytes(c, p));
    let boxes = post.map(rankmap::boxes);
    let mem = fit::mem_total().unwrap_or(122 * GIB);
    let mut ok = true;
    for sel in &sels {
        let b = sel.by_bucket();
        let g = |k: Bucket| b.get(&k).copied().unwrap_or(0);
        let heads = engram::heads_of_rank(c.cfg.engram_heads, world, sel.rank);
        let eng = engram::bytes_for_rank(&c.cfg, world, sel.rank);
        let host = boxes.map(|bx| bx[sel.rank].0.rsplit('.').next().map(|x| format!(".{x}")).unwrap_or_default()).unwrap_or_default();
        let pre = prelim.as_ref().map(|p| p[sel.rank].replicated + p[sel.rank].split + p[sel.rank].experts).map_or("-".into(), gb);
        let inp = FitInput::planned(sel.bytes(), lanes, seq, fit::default_chunk(world), cache_mib, mem, world);
        let fits = match fit::check(&c.cfg, &inp) {
            Ok(f) => { let tag = if f.caps.is_empty() { format!("{} / {} / {} -> {} GiB", f.lanes, f.max_seq_len, f.chunk, gib(f.total)) } else { format!("CAPPED {} / {} / {} -> {} GiB", f.lanes, f.max_seq_len, f.chunk, gib(f.total)) }; tag }
            Err(_) => "DOES NOT FIT (no cap left)".to_string(),
        };
        let _ = writeln!(s, "| {} | {host} | {} | {} | {} | {} | **{}** | {}..{} | {} | {pre} | {fits} |", sel.rank,
                         gb(g(Bucket::Replicated)), gb(g(Bucket::Split)), gb(g(Bucket::Experts)), gb(g(Bucket::RankZero)), gb(sel.bytes()),
                         heads.start, heads.end, gb(eng));
    }
    let _ = writeln!(s, "\nproofs: unique tensor bytes {} = census {} (Engram tables excluded): **{}**; bytes held over all ranks {} ({:.2}x the unique bytes: the replicated part is held {} times)",
                     uniq, census_total, if uniq == census_total { "EQUAL" } else { ok = false; "NOT EQUAL" }, held, held as f64 / uniq as f64, world);
    // the fit arithmetic of rank 0
    let w0 = sels[0].bytes();
    let inp = FitInput::planned(w0, lanes, seq, fit::default_chunk(world), cache_mib, mem, world);
    match fit::check(&c.cfg, &inp) {
        Ok(f) => { let _ = writeln!(s, "memory fit, rank 0: {}", f.line(&c.cfg, &inp)); if let Some(n) = f.notice(&inp) { let _ = writeln!(s, "  notice: {n}"); } }
        Err(e) => { let _ = writeln!(s, "memory fit, rank 0: {e}"); }
    }
    Ok((s, ok))
}

/// The whole dry run for `worlds`; returns (markdown, all proofs held).
pub fn run(model_dir: &Path, worlds: &[usize], deal: &str, lanes: usize, seq: Option<usize>, cache_mib: Option<u64>) -> Result<(String, bool)> {
    let c = ship::census_of(model_dir)?;
    let seq = seq.unwrap_or(c.cfg.max_pos);
    let mut s = String::new();
    let cache_desc = match cache_mib { Some(m) => format!("{m} MiB"), None => "1024 MiB (TP2) / 2048 MiB (TP4+)".into() };
    let _ = writeln!(s, "# DSV4.1 shard plan (dry run)\n\npack `{}`: {} tensors, {} bytes, {}; {} lane(s) x {} tokens, Engram row cache {cache_desc}, box memory {} GiB",
                     model_dir.display(), c.entries.len(), c.total_bytes(), c.weights_line(), lanes, seq, gib(fit::mem_total().unwrap_or(0)));
    let mut ok = true;
    for &w in worlds {
        if let Some(p) = Posture::from_world(w) {
            let bad = rankmap::posture_divides(&c.cfg, p);
            if !bad.is_empty() { let _ = writeln!(s, "\n### TP{w}: not a valid posture ({})", bad.join(", ")); continue; }
        }
        let cm = cache_mib.unwrap_or_else(|| fit::default_engram_cache_mib(w));
        match posture_report(&c, w, deal, lanes, seq, cm) { Ok((t, k)) => { s.push_str(&t); ok &= k; } Err(e) => { let _ = writeln!(s, "\n### TP{w}: PLAN FAILED: {e:#}"); ok = false; } }
    }
    Ok((s, ok))
}
