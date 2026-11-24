//! DSV4.1 (leg I0): the decode / prefill state, in the model's own formats, sized from the config
//! x lanes x max_seq_len, with exact byte accounting. No bf16 KV shortcut (plan §2.3).
//!
//! Per lane, from `inference/model.py`:
//! * **main compressed KV**, one cache per KV source (layers 2, 8, 14 at ratio 2; 20 at ratio 1),
//!   `max_seq_len / ratio` rows of `head_dim` = 512 values, stored FP4 E2M1 with one E4M3 scale per
//!   16 (`fp4_act_quant(latent, 16, scale_dtype=e4m3)`): 256 + 32 = **288 B / row**;
//! * **index-K**, one cache per KV source (`Indexer.owns_k = layer in kv_source_layers`: 4 caches,
//!   NOT one per index source — the 4 non-KV index sources 24 / 28 / 32 / 36 read source 20's),
//!   `max_seq_len / ratio` rows of 128 values, FP4 with one E8M0 scale per 32: 64 + 4 = **68 B / row**;
//! * **SWA ring** per layer slot (40 trunk + 3 DSpark stages), window + speculative backtracking
//!   = 128 + 6 = **134 slots** (DeepSeek's own engine sizes it so; PLAN/DSV41_XREF_DEEPINFER.md §4.1;
//!   slot = position % 134) of 512 values, FP8 E4M3 with one E8M0 scale per 32
//!   (`act_quant(kv, 32, ue8m0)`): 512 + 16 = **528 B / slot**;
//! * **compressor frontier ring** per ratio>1 KV source: f32 `kv_state` + `score_state` rows
//!   [ratio + backtracking - 1 = 7, 512] (the incomplete group plus the rows a rejected draft rolls back);
//! * **no Engram n-gram state**: the hash windows are rebuilt from the lane's token ids (DeepSeek's
//!   engine keeps none either); a prefix state therefore needs only token ids for Engram;
//! * **DSpark carry**: the target layers' stream means `main_hidden` [3 x 5120] bf16 of the last
//!   committed position (the drafter's conditioning).
//! Transient per-forward buffers (not persistent, sized per row): the mHC stream [hc x 5120],
//! the layer-20 candidate blocks (2048 block ids per query), index scores. See `transient_per_row`.

use crate::dsv41::config::Dsv41Config;
use std::fmt::Write as _;

/// FP4 E2M1 values + E4M3 scale per 16: bytes of one main-KV row of `dim` values.
pub fn main_kv_row_bytes(dim: usize) -> usize { dim / 2 + dim / 16 }
/// FP4 E2M1 values + E8M0 scale per 32: bytes of one index-K row.
pub fn index_k_row_bytes(dim: usize) -> usize { dim / 2 + dim / 32 }
/// FP8 E4M3 values + E8M0 scale per 32: bytes of one SWA slot.
pub fn swa_row_bytes(dim: usize) -> usize { dim + dim / 32 }

/// One compressed main-KV cache (per KV source).
#[derive(Clone, Debug, PartialEq)]
pub struct MainKv { pub source: usize, pub ratio: u32, pub rows: usize, pub row_bytes: usize, pub readers: Vec<usize> }

/// One index-K cache (per KV source; read by the index sources whose KV source it is).
#[derive(Clone, Debug, PartialEq)]
pub struct IndexK { pub source: usize, pub ratio: u32, pub rows: usize, pub row_bytes: usize, pub index_sources: Vec<usize> }

/// One sliding-window ring (per layer slot).
#[derive(Clone, Debug, PartialEq)]
pub struct SwaRing { pub layer: usize, pub slots: usize, pub row_bytes: usize }

/// One compressor frontier (per ratio>1 KV source): f32 kv_state + score_state [ratio, head_dim].
#[derive(Clone, Debug, PartialEq)]
pub struct Frontier { pub layer: usize, pub ratio: u32, pub bytes: usize }

/// The per-lane persistent state of one lane at `max_seq_len`.
#[derive(Clone, Debug, PartialEq)]
pub struct LaneState {
    pub max_seq_len: usize,
    pub main_kv: Vec<MainKv>,
    pub index_k: Vec<IndexK>,
    pub swa: Vec<SwaRing>,
    pub frontiers: Vec<Frontier>,
    /// Engram n-gram state: always 0 (rebuilt from the token ids; kept as a field so the report says so)
    pub ngram_bytes: usize,
    /// DSpark: main_hidden of the last committed position, bf16
    pub dspark_carry_bytes: usize,
}

/// Byte totals of one lane.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StateBytes { pub main_kv: u64, pub index_k: u64, pub swa: u64, pub frontiers: u64, pub ngram: u64, pub dspark: u64 }

impl StateBytes {
    pub fn total(&self) -> u64 { self.main_kv + self.index_k + self.swa + self.frontiers + self.ngram + self.dspark }
}

impl LaneState {
    pub fn plan(cfg: &Dsv41Config, max_seq_len: usize) -> LaneState {
        let hd = cfg.head_dim;
        let main_kv = cfg.kv_sources.iter().map(|&s| {
            let r = cfg.ratio(s);
            let readers = (0..cfg.layers).filter(|&l| cfg.kv_source_of(l) == Some(s)).collect();
            MainKv { source: s, ratio: r, rows: max_seq_len / r.max(1) as usize, row_bytes: main_kv_row_bytes(hd), readers }
        }).collect();
        let index_k = cfg.kv_sources.iter().map(|&s| {
            let r = cfg.ratio(s);
            let idx = cfg.index_sources.iter().copied().filter(|&i| cfg.kv_source_of(i) == Some(s)).collect();
            IndexK { source: s, ratio: r, rows: max_seq_len / r.max(1) as usize, row_bytes: index_k_row_bytes(cfg.index_head_dim), index_sources: idx }
        }).collect();
        let swa = (0..cfg.total_layers()).map(|l| SwaRing { layer: l, slots: cfg.swa_slots(), row_bytes: swa_row_bytes(hd) }).collect();
        let frontiers = cfg.kv_sources.iter().filter(|&&s| cfg.ratio(s) > 1)
            .map(|&s| Frontier { layer: s, ratio: cfg.ratio(s), bytes: 2 * cfg.frontier_rows(cfg.ratio(s)) * hd * 4 }).collect();
        LaneState { max_seq_len, main_kv, index_k, swa, frontiers,
                    ngram_bytes: 0,
                    dspark_carry_bytes: cfg.dspark_targets.len() * cfg.hidden * 2 }
    }

    pub fn bytes(&self) -> StateBytes {
        StateBytes {
            main_kv: self.main_kv.iter().map(|c| (c.rows * c.row_bytes) as u64).sum(),
            index_k: self.index_k.iter().map(|c| (c.rows * c.row_bytes) as u64).sum(),
            swa: self.swa.iter().map(|c| (c.slots * c.row_bytes) as u64).sum(),
            frontiers: self.frontiers.iter().map(|f| f.bytes as u64).sum(),
            ngram: self.ngram_bytes as u64,
            dspark: self.dspark_carry_bytes as u64,
        }
    }
}

/// A prefix state (the contract for legs SV / CE, after PLAN/DSV41_XREF_DEEPINFER.md §3 / D8): to
/// resume a lane EXACTLY at `len` tokens it needs the main-KV and index-K rows of positions < len,
/// every one of the 43 SWA rings (134 slots each, encoder AND decoder AND DSpark), the compressor
/// frontier rings and the DSpark carry — and the token ids, from which the Engram hash windows are
/// rebuilt (no Engram state is stored). DeepSeek's engine stores only the encoder SWA and replays
/// the decoder (approximate below 2,541 kept rows); a full snapshot is our exact path.
pub fn prefix_state_bytes(cfg: &Dsv41Config, len: usize) -> u64 {
    let full = LaneState::plan(cfg, len).bytes();
    full.main_kv + full.index_k + full.swa + full.frontiers + full.dspark + (len * 4) as u64
}

/// What `inference/model.py` itself allocates per lane (its buffers in their reference dtypes:
/// bf16 caches under generate.py's default dtype, f32 compressor states, the int64 n-gram history).
pub fn reference_bytes_per_lane(cfg: &Dsv41Config, max_seq_len: usize) -> StateBytes {
    let hd = cfg.head_dim as u64;
    let s = max_seq_len as u64;
    let mut b = StateBytes::default();
    for &src in &cfg.kv_sources {
        let r = cfg.ratio(src).max(1) as u64;
        b.main_kv += s / r * hd * 2;                       // compress_kv_cache bf16
        b.index_k += s / r * cfg.index_head_dim as u64 * 2; // Indexer.k_cache bf16 (owns_k)
        if r > 1 { b.frontiers += 2 * r * hd * 4; }         // kv_state + score_state f32
    }
    b.swa = cfg.total_layers() as u64 * cfg.window as u64 * hd * 2; // window_kv_cache bf16
    b.ngram = s * 8;                                                // NgramHashState.cache int64
    b
}

/// Transient bytes per processed row (activations the forward keeps live; not persistent state).
pub fn transient_per_row(cfg: &Dsv41Config, max_seq_len: usize) -> Vec<(String, u64)> {
    let c20 = cfg.candidate_source.map(|c| cfg.ratio(c).max(1) as usize).unwrap_or(1);
    vec![
        ("mHC stream [hc x hidden] bf16".into(), (cfg.hc * cfg.hidden * 2) as u64),
        ("index scores of one index source, f32 over every compressed position at max_seq_len (TILE IT for prefill)".into(),
         (max_seq_len / c20 * 4) as u64),
        ("layer-20 candidate blocks (i32 block ids)".into(), (cfg.candidate_topk_blocks * 4) as u64),
        ("selected compressed positions (i32, index_topk) + SWA slots".into(), ((cfg.index_topk + cfg.window) * 4) as u64),
    ]
}

/// The state report (census / `--dsv41-census`): ours vs the reference, per lane and per posture.
pub fn report(cfg: &Dsv41Config, lanes: usize, max_seq_len: usize) -> String {
    let gb = |b: u64| format!("{:.3}", b as f64 / 1e9);
    let ln = LaneState::plan(cfg, max_seq_len);
    let ours = ln.bytes();
    let refb = reference_bytes_per_lane(cfg, max_seq_len);
    let mut s = String::new();
    let _ = writeln!(s, "## State per lane at max_seq_len {max_seq_len} (persistent)\n");
    let _ = writeln!(s, "| piece | layout | ours (GB) | inference/model.py buffers (GB) |\n|---|---|---:|---:|");
    let mk: Vec<String> = ln.main_kv.iter().map(|c| format!("src {} r{} {} rows", c.source, c.ratio, c.rows)).collect();
    let _ = writeln!(s, "| main compressed KV ({}) | FP4 E2M1 + E4M3/16, {} B/row: {} | {} | {} (bf16) |", ln.main_kv.len(), main_kv_row_bytes(cfg.head_dim), mk.join("; "), gb(ours.main_kv), gb(refb.main_kv));
    let ik: Vec<String> = ln.index_k.iter().map(|c| format!("src {} -> index sources {:?}", c.source, c.index_sources)).collect();
    let _ = writeln!(s, "| index-K ({} caches, one per KV source) | FP4 E2M1 + E8M0/32, {} B/row: {} | {} | {} (bf16) |", ln.index_k.len(), index_k_row_bytes(cfg.index_head_dim), ik.join("; "), gb(ours.index_k), gb(refb.index_k));
    let _ = writeln!(s, "| SWA rings ({} layer slots x {} = window {} + backtracking {}) | FP8 E4M3 + E8M0/32, {} B/slot (ref: {} slots) | {} | {} (bf16) |",
                     ln.swa.len(), cfg.swa_slots(), cfg.window, cfg.backtrack(), swa_row_bytes(cfg.head_dim), cfg.window, gb(ours.swa), gb(refb.swa));
    let _ = writeln!(s, "| compressor frontier rings ({}) | f32 kv_state + score_state [ratio + {} rows, {}] (ref: [ratio, {}]) | {} | {} |",
                     ln.frontiers.len(), cfg.backtrack() - 1, cfg.head_dim, cfg.head_dim, gb(ours.frontiers), gb(refb.frontiers));
    let _ = writeln!(s, "| Engram n-gram state | none: rebuilt from the token ids (ref: the whole int64 history) | {} | {} |", gb(ours.ngram), gb(refb.ngram));
    let _ = writeln!(s, "| DSpark carry | main_hidden [{} x {}] bf16 | {} | - |", cfg.dspark_targets.len(), cfg.hidden, gb(ours.dspark));
    let _ = writeln!(s, "| **total per lane** | | **{}** ({:.3} GiB) | **{}** |", gb(ours.total()), ours.total() as f64 / (1u64 << 30) as f64, gb(refb.total()));
    let _ = writeln!(s, "\n{} lanes: **{} GB** (ours) vs {} GB (reference buffers). Under the preliminary split every rank holds the full state (MQA: one KV head that every rank's query heads read; the index scores are all-reduced over index heads), so this is also the per-rank figure until T0 decides on sequence-parallel KV.",
                     lanes, gb(ours.total() * lanes as u64), gb(refb.total() * lanes as u64));
    let _ = writeln!(s, "\nTransient per processed row:");
    for (k, b) in transient_per_row(cfg, max_seq_len) { let _ = writeln!(s, "- {k}: {} KB", b / 1024); }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsv41::config::{test_config_json, Dsv41Config};

    #[test]
    fn dsv41_state_bytes_at_1m() {
        let c = Dsv41Config::from_value(&test_config_json(), None).unwrap();
        let l = LaneState::plan(&c, 1 << 20);
        assert_eq!(main_kv_row_bytes(512), 288);
        assert_eq!(index_k_row_bytes(128), 68);
        assert_eq!(swa_row_bytes(512), 528);
        assert_eq!(l.main_kv.len(), 4);
        assert_eq!(l.index_k.len(), 4, "index-K caches are owned by the 4 KV sources");
        assert_eq!(l.index_k[3].index_sources, vec![20, 24, 28, 32, 36]);
        assert_eq!(l.main_kv[0].readers, (2..8).collect::<Vec<_>>());
        assert_eq!(l.swa.len(), 43);
        assert!(l.swa.iter().all(|r| r.slots == 134), "window 128 + backtracking 6");
        let b = l.bytes();
        assert_eq!(b.main_kv, 3 * (1 << 19) * 288 + (1 << 20) * 288);
        assert_eq!(b.index_k, 3 * (1 << 19) * 68 + (1 << 20) * 68);
        assert_eq!(b.swa, 43 * 134 * 528);
        assert_eq!(b.frontiers, 3 * 2 * 7 * 512 * 4);
        assert_eq!(b.ngram, 0, "no Engram state: token ids suffice");
        // the plan's "~0.9 GB per 1M-token lane": 0.936 GB
        assert!((b.total() as f64 / 1e9 - 0.936).abs() < 0.001, "{}", b.total());
        let r = reference_bytes_per_lane(&c, 1 << 20);
        assert_eq!(r.main_kv, (3 * (1u64 << 19) + (1 << 20)) * 1024);
        assert!(report(&c, 4, 1 << 20).contains("**total per lane**"));
        assert_eq!(prefix_state_bytes(&c, 4096), (3 * 2048 + 4096) * (288 + 68) + 43 * 134 * 528 + 3 * 2 * 7 * 512 * 4 + 3 * 5120 * 2 + 4096 * 4);
    }
}
