//! DSV4.1 TP (leg TPC): what crosses the wire in a decode round, as pure data (PLAN/DSV41_TP_DESIGN.md 4.1).
//!
//! A **site** is one collective of the round. A **frame** is the fp32 buffer a site reduces, laid out in
//! sections so the producers write straight into it:
//!
//! | section | shape | written by |
//! |---|---|---|
//! | `hidden` | `[rows][hidden]` (5,120) | the sublayer's FP32 partial (`wo_b` K slice / the MoE inner-width combine) |
//! | `mix` | `[rows][mix_stride]` (32; 25 used) | the K-slice partials of the NEXT mHC mix: 24 `hc_fn` dots + the sum of squares (KM `mix_partial`, the rank's slots pre-summed in slot order) |
//! | `extra` | `[rows][extra]` | a rider: Engram `wkv` partial (25,600 = (hc+1) x hidden) on the FFN reduce of the layer before each Engram layer; DSpark `main_proj` partial (5,120) on the attention reduce of the last backbone layer |
//!
//! Head-style sites (the vocab-parallel head and the drafter's candidates / Markov steps) gather one
//! 16-word record per row and rank instead (`[rows][world][16]`, kernels/gpu_dsv41/05_tp.cuh); the residual
//! exchange of a sampled verify gathers 4 words.
//!
//! The reduction association is the transport's (`tp_xport::rd_tree_reference`): world 2 `p0 + p1`; world 4
//! `(p0 + p2) + (p1 + p3)` once the round phase is aligned at attach (R10; the single-stage ctx forms the same
//! association). [`tree_reference`] is the CPU model of it.

use anyhow::{ensure, Result};

/// The largest decode-round row count a frame is sized for (up to 32 verify rows across active lanes).
pub const MAX_ROWS: usize = 32;
/// Head-style record words per row and rank (`D41TP_HEAD_REC`).
pub const HEAD_REC: usize = 16;
/// Residual record words per row and rank (`D41TP_RES_REC`).
pub const RES_REC: usize = 4;
/// Mix rider stride per row: 25 used, padded to a 16 B multiple so every frame is a whole number of uint4
/// (the multi-block K1 copies 16 B per thread).
pub const MIX_STRIDE: usize = 32;

/// The model constants a frame depends on (from `Dsv41Config`; the defaults are DeepSeek-V4.1-Flash-1017's).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape {
    pub hidden: usize,
    pub hc: usize,
    pub layers: usize,
    /// Engram layer ids (their `wkv` partial rides the FFN reduce of layer id - 1)
    pub engram_layers: Vec<usize>,
    pub dspark_stages: usize,
    pub dspark_block: usize,
    pub vocab: usize,
}

impl Shape {
    pub fn of(cfg: &crate::dsv41::config::Dsv41Config) -> Shape {
        Shape { hidden: cfg.hidden, hc: cfg.hc, layers: cfg.layers, engram_layers: cfg.engram_layers.clone(),
                dspark_stages: cfg.dspark_stages, dspark_block: cfg.dspark_block, vocab: cfg.vocab }
    }
    /// DeepSeek-V4.1-Flash-1017 (config.json): for unit tests and the config-free smoke.
    pub fn v41_flash() -> Shape {
        Shape { hidden: 5120, hc: 4, layers: 40, engram_layers: vec![1, 14], dspark_stages: 3, dspark_block: 5, vocab: 129_280 }
    }
    /// 24 `hc_fn` dot products + 1 sum of squares.
    pub fn mix_used(&self) -> usize { (2 + self.hc) * self.hc + 1 }
    /// Engram `kv` width: (hc + 1) x hidden (`cpu::engram_mod::KOUT`).
    pub fn engram_kv(&self) -> usize { (self.hc + 1) * self.hidden }
}

/// One collective of the decode round.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Site {
    /// the attention sublayer's reduce of backbone layer l
    Attn(u16),
    /// the FFN sublayer's reduce of backbone layer l
    Ffn(u16),
    /// the target head exchange (greedy / sample / accept)
    Head,
    /// the residual exchange of a sampled verify (first rejection)
    HeadResidual,
    /// DSpark stage s, attention / FFN reduce (block rows)
    DraftAttn(u8),
    DraftFfn(u8),
    /// the drafter's head pass candidates (block rows)
    DraftHead,
    /// Markov step k of the draft chain (one row per lane)
    Markov(u8),
}

/// The class a site's latency is reported under.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Class {
    /// a layer reduce with the mix rider only
    Layer,
    /// a layer reduce carrying the Engram `wkv` partial
    LayerEngram,
    /// a layer reduce carrying the `main_proj` partial
    LayerMainProj,
    Head,
    HeadResidual,
    DraftLayer,
    DraftHead,
    Markov,
}

impl Class {
    pub fn name(self) -> &'static str {
        match self {
            Class::Layer => "layer reduce (hidden + mix)",
            Class::LayerEngram => "layer reduce + Engram wkv",
            Class::LayerMainProj => "layer reduce + main_proj",
            Class::Head => "head exchange",
            Class::HeadResidual => "head residual",
            Class::DraftLayer => "drafter stage reduce",
            Class::DraftHead => "drafter candidates",
            Class::Markov => "Markov step",
        }
    }
}

/// A site's frame at a row count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameSpec {
    pub rows: usize,
    pub hidden: usize,
    pub mix_used: usize,
    pub extra: usize,
    /// head-style: words per (row, rank) record (0 = a sum frame)
    pub rec: usize,
    pub world: usize,
}

impl FrameSpec {
    pub fn is_gather(&self) -> bool { self.rec > 0 }
    /// fp32 elements on the wire (every rank sends and receives this many per exchange)
    pub fn floats(&self) -> usize {
        if self.is_gather() { self.rows * self.world * self.rec } else { self.rows * (self.hidden + MIX_STRIDE + self.extra) }
    }
    pub fn bytes(&self) -> usize { self.floats() * 4 }
    /// element offsets of the sections (sum frames)
    pub fn hidden_off(&self) -> usize { 0 }
    pub fn mix_off(&self) -> usize { self.rows * self.hidden }
    pub fn extra_off(&self) -> usize { self.rows * (self.hidden + MIX_STRIDE) }
    /// payload per row per rank as T0 states it (the record of this rank only, for gathers)
    pub fn bytes_per_row_per_rank(&self) -> usize {
        if self.is_gather() { self.rec * 4 } else { (self.hidden + self.mix_used + self.extra) * 4 }
    }
}

/// The program of one decode round: which sites, in order, at which rows.
#[derive(Clone, Debug)]
pub struct Round {
    pub sites: Vec<(Site, usize)>,
}

/// Lanes behind a verify width (6 rows per lane; a 1-row round is one lane).
pub fn lanes_of(rows: usize) -> usize { rows.div_ceil(6).max(1) }

impl Shape {
    /// The rows a drafter site runs at, for a round whose verify width is `rows`.
    pub fn draft_rows(&self, rows: usize) -> usize { self.dspark_block * lanes_of(rows) }

    /// The frame of `site` at `rows` (the site's own row count, see `round`).
    pub fn frame(&self, site: Site, rows: usize, world: usize) -> FrameSpec {
        let sum = |extra: usize| FrameSpec { rows, hidden: self.hidden, mix_used: self.mix_used(), extra, rec: 0, world };
        match site {
            Site::Attn(l) if l as usize + 1 == self.layers && self.dspark_stages > 0 => sum(self.hidden),
            Site::Ffn(l) if self.engram_layers.iter().any(|&e| e == l as usize + 1) => sum(self.engram_kv()),
            Site::Attn(_) | Site::Ffn(_) | Site::DraftAttn(_) | Site::DraftFfn(_) => sum(0),
            Site::Head | Site::DraftHead | Site::Markov(_) => FrameSpec { rows, hidden: 0, mix_used: 0, extra: 0, rec: HEAD_REC, world },
            Site::HeadResidual => FrameSpec { rows, hidden: 0, mix_used: 0, extra: 0, rec: RES_REC, world },
        }
    }

    pub fn class(&self, site: Site) -> Class {
        match site {
            Site::Attn(l) if l as usize + 1 == self.layers && self.dspark_stages > 0 => Class::LayerMainProj,
            Site::Ffn(l) if self.engram_layers.iter().any(|&e| e == l as usize + 1) => Class::LayerEngram,
            Site::Attn(_) | Site::Ffn(_) => Class::Layer,
            Site::DraftAttn(_) | Site::DraftFfn(_) => Class::DraftLayer,
            Site::Head => Class::Head,
            Site::HeadResidual => Class::HeadResidual,
            Site::DraftHead => Class::DraftHead,
            Site::Markov(_) => Class::Markov,
        }
    }

    /// T0's decode round at verify width `rows`: 2 reduces per backbone layer, the head exchange, then the drafter
    /// tail (2 reduces per stage at block rows, the candidates, one Markov exchange per draft position). With
    /// `residual` the sampled-verify residual exchange follows the head (one more small collective than T0 counts).
    pub fn round(&self, rows: usize, residual: bool) -> Round {
        let mut s = Vec::new();
        for l in 0..self.layers as u16 { s.push((Site::Attn(l), rows)); s.push((Site::Ffn(l), rows)); }
        s.push((Site::Head, rows));
        if residual { s.push((Site::HeadResidual, rows)); }
        let d = self.draft_rows(rows);
        for st in 0..self.dspark_stages as u8 { s.push((Site::DraftAttn(st), d)); s.push((Site::DraftFfn(st), d)); }
        s.push((Site::DraftHead, d));
        for k in 0..self.dspark_block as u8 { s.push((Site::Markov(k), lanes_of(rows))); }
        Round { sites: s }
    }
}

impl Round {
    pub fn len(&self) -> usize { self.sites.len() }
    pub fn is_empty(&self) -> bool { self.sites.is_empty() }
    /// bytes one rank sends in the round (one exchange per site; a gather sends the whole zero-padded frame)
    pub fn bytes_sent(&self, shape: &Shape, world: usize) -> usize {
        self.sites.iter().map(|(s, r)| shape.frame(*s, *r, world).bytes()).sum()
    }
    /// the largest frame of the round, in fp32 elements
    pub fn max_floats(&self, shape: &Shape, world: usize) -> usize {
        self.sites.iter().map(|(s, r)| shape.frame(*s, *r, world).floats()).max().unwrap_or(0)
    }
}

/// The first round bit of the transport's association at `world` once the round phase is aligned (R10: the
/// device epoch is a multiple of `rounds` before a reduce, so the first exchange runs round (e0 + 1) % rounds).
pub fn first_round_bit(world: usize) -> usize {
    match world { 4 => 1, 8 => 1, _ => 0 }
}

/// CPU model of the reduction: rank partials summed in the transport's association (world 1, 2, 4, 8).
pub fn tree_reference(parts: &[Vec<f32>]) -> Result<Vec<f32>> {
    let w = parts.len();
    ensure!(matches!(w, 1 | 2 | 4 | 8) && parts.iter().all(|p| p.len() == parts[0].len()), "tree_reference: {w} partials");
    if w == 1 { return Ok(parts[0].clone()); }
    // e0 = 0 (aligned): rd_tree_reference's first round is (0 + 1) % rounds
    crate::tp_xport::rd_tree_reference(w, 0, parts)
}
