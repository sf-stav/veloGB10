//! DSV4.1 census (leg I0): the per-rank byte accounting and the markdown report.
//!
//! The per-rank split is the plan's PRELIMINARY one (`classify::split_of`): routed and DSpark
//! experts by EP, attention by `o_groups` (wq_b / sink / wo_a column-parallel, wo_b row-parallel),
//! the indexer's query heads column-parallel, a vocab-parallel LM head (and DSpark Markov head),
//! Engram tables by owned head, everything else replicated. Leg T0 owns the final design.

use super::classify::{split_of, Role, Split, TensorClass};
use super::{Census, Entry, LinearFormat};
use crate::dsv41::rankmap::{self, Posture};
use std::collections::BTreeMap;
use std::fmt::Write as _;

/// One rank's weight bytes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RankBytes {
    pub rank: usize,
    pub replicated: u64,
    /// column / row-parallel shares (attention by o_groups, indexer heads, vocab-parallel heads)
    pub split: u64,
    pub experts: u64,
    /// Engram tables held in this pack (0 for a pack without tables)
    pub engram_tables: u64,
}

impl RankBytes {
    pub fn total(&self) -> u64 { self.replicated + self.split + self.experts + self.engram_tables }
}

fn part(n: usize, w: usize, r: usize) -> std::ops::Range<usize> { (r * n / w)..((r + 1) * n / w) }

/// The bytes of tensor `e` that rank `r` of a `w`-way posture holds, and the bucket it counts in.
pub fn entry_share(c: &Census, e: &Entry, w: usize, r: usize, lay: &rankmap::EngramLayout) -> (u64, &'static str) {
    let cfg = &c.cfg;
    let even = |b: u64| b.div_ceil(w as u64);
    match split_of(e.class) {
        Split::Replicated => (e.bytes, "replicated"),
        Split::Expert => {
            let n = e.layer.map(|l| cfg.moe_of(l as usize).0).unwrap_or(cfg.n_routed);
            let mine = e.sub.map_or(false, |x| part(n, w, r).contains(&(x as usize)));
            (if mine { e.bytes } else { 0 }, "experts")
        }
        Split::Out if e.class == TensorClass::AttnWoA && e.sub.is_some() => {
            // EXL3 per-group slice: the whole slice module lives on the rank owning that group
            let mine = part(cfg.o_groups, w, r).contains(&(e.sub.unwrap_or(0) as usize));
            (if mine { e.bytes } else { 0 }, "split")
        }
        Split::Out => match e.role {
            Role::Suh | Role::Mul1 | Role::Scale2 | Role::InputScale => (e.bytes, "replicated"),
            _ => (even(e.bytes), "split"),
        },
        Split::In => match e.role {
            Role::Svh | Role::Mul1 | Role::Bias | Role::Scale2 | Role::InputScale => (e.bytes, "replicated"),
            _ => (even(e.bytes), "split"),
        },
        Split::EngramHeads => {
            let li = cfg.engram_layers.iter().position(|&l| Some(l as u16) == e.layer).unwrap_or(0);
            let rows = cfg.engram_rows.get(li).copied().unwrap_or(1).max(1);
            let heads = part(cfg.engram_heads, w, r);
            let mine: u64 = heads.map(|h| lay.head_rows(li, h)).sum();
            (e.bytes / rows * mine, "engram")
        }
    }
}

/// Every rank's bytes for posture `p`.
pub fn rank_bytes(c: &Census, p: Posture) -> Vec<RankBytes> {
    let w = p.world();
    let lay = rankmap::engram_layout(&c.cfg);
    (0..w).map(|r| {
        let mut b = RankBytes { rank: r, ..Default::default() };
        for e in &c.entries {
            let (n, bucket) = entry_share(c, e, w, r, &lay);
            match bucket { "replicated" => b.replicated += n, "split" => b.split += n, "experts" => b.experts += n, _ => b.engram_tables += n }
        }
        b
    }).collect()
}

/// Engram table bytes per rank when the pack carries no tables (DeepSeek's FP8 tables, by head).
pub fn native_engram_per_rank(c: &Census, p: Posture) -> Vec<u64> {
    let lay = rankmap::engram_layout(&c.cfg);
    rankmap::rank_map(&c.cfg, p).iter().map(|s| rankmap::engram_bytes_for(&c.cfg, &lay, s.engram_heads.clone())).collect()
}

fn gb(b: u64) -> String { format!("{:.2}", b as f64 / 1e9) }

/// The markdown section of one pack.
pub fn section(c: &Census, title: &str) -> String {
    let mut s = String::new();
    let cfg = &c.cfg;
    let total = c.total_bytes();
    let _ = writeln!(s, "## {title}\n");
    let _ = writeln!(s, "- dir: `{}`", c.dir.display());
    let _ = writeln!(s, "- quantization: `{}`{}", cfg.quant.method,
                     if cfg.quant.has_file { format!(" (quantization_config.json, {} tensor_storage entries)", cfg.quant.tensor_storage_entries) } else { String::new() });
    let _ = writeln!(s, "- files: {}; tensors: **{}**; modules: {}; bytes: **{}** ({} GB)", c.files.len(), c.entries.len(), c.modules.len(), total, gb(total));
    match c.index_total {
        Some(t) => {
            let _ = writeln!(s, "- index: {} keys, total_size {} -> census {} keys / {} bytes: **{}**", c.index_keys, t, c.entries.len(), total,
                             if c.index_keys == c.entries.len() && t == total && c.problems.is_empty() { "EQUAL" } else { "NOT EQUAL" });
        }
        None => { let _ = writeln!(s, "- index: none (single-file pack)"); }
    }
    if let Some(l) = &cfg.slice_layers { let _ = writeln!(s, "- slice of layers {l:?}"); }
    let _ = writeln!(s, "- weights: {}", c.weights_line());
    let _ = writeln!(s, "- problems: {}", if c.problems.is_empty() { "none".to_string() } else { c.problems.len().to_string() });
    for p in &c.problems { let _ = writeln!(s, "  - {p}"); }
    for n in &c.notes { let _ = writeln!(s, "- note: {n}"); }

    let _ = writeln!(s, "\n### Totals by class and format\n");
    let _ = writeln!(s, "| class | group | format | modules | tensors | GB |\n|---|---|---|---:|---:|---:|");
    for ((cl, f), (t, m, b)) in c.by_class_format() {
        let _ = writeln!(s, "| {} | {} | {} | {} | {} | {} |", cl.name(), cl.group(), f, m, t, gb(b));
    }
    let _ = writeln!(s, "\n### Totals by tensor format\n");
    let _ = writeln!(s, "| tensor format | tensors | GB | share |\n|---|---:|---:|---:|");
    for (f, (t, b)) in c.by_tensor_format() {
        let _ = writeln!(s, "| {f} | {t} | {} | {:.2}% |", gb(b), b as f64 * 100.0 / total.max(1) as f64);
    }
    // EXL3 bit widths per class
    let mut kh: BTreeMap<(TensorClass, u8), (usize, Vec<u16>)> = BTreeMap::new();
    for m in &c.modules {
        if let LinearFormat::Exl3 { k, .. } = m.format {
            let r = kh.entry((m.class, k)).or_default();
            r.0 += 1;
            if let Some(l) = m.layer { if !r.1.contains(&l) { r.1.push(l); } }
        }
    }
    if !kh.is_empty() {
        let _ = writeln!(s, "\n### EXL3 bit widths (K) per class\n");
        let _ = writeln!(s, "| class | K | modules | layers |\n|---|---:|---:|---|");
        for ((cl, k), (n, ls)) in kh {
            let _ = writeln!(s, "| {} | {k} | {n} | {} |", cl.name(), layer_ranges(&ls));
        }
    }
    // bytes per layer
    let groups = ["attention", "indexer+compressor", "hc+norms", "router", "routed experts", "shared expert", "engram module", "engram tables", "dspark extras"];
    let _ = writeln!(s, "\n### Bytes per layer (GB)\n");
    let _ = writeln!(s, "| layer | class | {} | total |\n|---|---|{}---:|", groups.join(" | "), "---:|".repeat(groups.len()));
    let mut per: BTreeMap<u16, BTreeMap<&'static str, u64>> = BTreeMap::new();
    let mut global: BTreeMap<&'static str, u64> = BTreeMap::new();
    for m in &c.modules {
        match m.layer { Some(l) => *per.entry(l).or_default().entry(m.class.group()).or_default() += m.bytes,
                        None => *global.entry(m.class.group()).or_default() += m.bytes }
    }
    for (l, g) in &per {
        let cls = match cfg.attn_class(*l as usize) {
            crate::dsv41::config::AttnClass::SwaOnly => "SWA".to_string(),
            crate::dsv41::config::AttnClass::DsparkStage => format!("DSpark {}", *l as usize - cfg.layers),
            crate::dsv41::config::AttnClass::Compressed { ratio, kv_source, index_source, candidate_source, uses_candidates } => {
                let mut t = format!("r{ratio}");
                if kv_source { t.push_str(" kv-src"); }
                if index_source { t.push_str(" idx-src"); }
                if candidate_source { t.push_str(" cand-src"); }
                if uses_candidates { t.push_str(" uses-cand"); }
                t
            }
        };
        let eng = if cfg.engram_layers.contains(&(*l as usize)) { " engram" } else { "" };
        let cells: Vec<String> = groups.iter().map(|k| g.get(k).map_or("-".into(), |b| gb(*b))).collect();
        let _ = writeln!(s, "| {l} | {cls}{eng} | {} | {} |", cells.join(" | "), gb(g.values().sum()));
    }
    let _ = writeln!(s, "\nGlobal (no layer): {}", global.iter().map(|(k, b)| format!("{k} {} GB", gb(*b))).collect::<Vec<_>>().join(", "));

    // per rank: device-resident weights; the Engram tables are read from NVMe per token, never resident
    let _ = writeln!(s, "\n### Preliminary per-rank weight bytes (GB; plan split, T0 refines)\n");
    let has_tables = c.modules.iter().any(|m| m.class == TensorClass::EngramTable);
    let _ = writeln!(s, "Resident = what the rank loads into (unified) memory. The Engram tables are NOT resident: each rank reads its own heads' rows from NVMe per token ({}).\n",
                     if has_tables { "this pack carries them" } else { "this pack has none: DeepSeek's FP8 tables from the native checkpoint, by head" });
    let _ = writeln!(s, "| posture | rank | box | replicated | split (attn / idx / head) | experts (EP) | **resident total** | Engram tables on disk |\n|---|---:|---|---:|---:|---:|---:|---:|");
    for p in [Posture::Tp1, Posture::Tp2, Posture::Tp4, Posture::Tp8] {
        let rb = rank_bytes(c, p);
        let native = native_engram_per_rank(c, p);
        let boxes = rankmap::boxes(p);
        for b in &rb {
            let tables = if has_tables { b.engram_tables } else { native[b.rank] };
            let host = boxes[b.rank].0.rsplit('.').next().map(|x| format!(".{x}")).unwrap_or_default();
            let _ = writeln!(s, "| {} | {} | {host} | {} | {} | {} | **{}** | {} |", p.name(), b.rank,
                             gb(b.replicated), gb(b.split), gb(b.experts), gb(b.replicated + b.split + b.experts), gb(tables));
        }
    }
    s
}

/// `[2,3,4,9]` -> "2-4, 9"
pub fn layer_ranges(ls: &[u16]) -> String {
    let mut v = ls.to_vec();
    v.sort();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < v.len() {
        let mut j = i;
        while j + 1 < v.len() && v[j + 1] == v[j] + 1 { j += 1; }
        out.push(if i == j { v[i].to_string() } else { format!("{}-{}", v[i], v[j]) });
        i = j + 1;
    }
    out.join(", ")
}
