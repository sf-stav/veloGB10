//! DSV4.1 loader (leg LS): the per-rank TENSOR SELECTION. For every tensor of a census (native, any
//! EXL3 pack, mixed, slice) and a `(world, rank)`: is it absent, whole, or a slice, and how many bytes.
//! This is T0's split (PLAN/DSV41_TP_DESIGN.md §1, §3) as one table, the single place a later change
//! of the split is made. Pure: it reads the census, never a file.
//!
//! | class | rule |
//! |---|---|
//! | routed experts (`w1` `w3` out, `w2` in), shared experts | **every** expert at 1/N of the inner width (2,304); EXL3 in 128-wide Hadamard blocks with T0's rotation by expert id, native in 32-wide groups; `--dsv41-expert-split ep` = whole experts by id range instead |
//! | DSpark stage experts | as the trunk (inner width), or EP by id range |
//! | `wq_b` (out by head), `attn_sink` (by head), `wo_a` (by group), `wo_b` (in by group) | the rank's heads / groups; **whole at TP4 / TP8** (row-parallel prefill needs them; T0 §8 "attention weights whole") |
//! | `wq_a`, `wkv`, compressor, indexer, router gate, norms, `hc_*` base / scale, Markov embed, confidence | replicated |
//! | `hc_attn_fn`, `hc_ffn_fn` | K (20,480) cut by 1/N, whole where the attention is whole |
//! | Engram `wkv` | TP2 EXL3: the rank's heads of every n-gram order (input cut); whole at TP4 / TP8 and for native |
//! | `main_proj` (DSpark) | input K cut by 1/N |
//! | `head`, Markov head | vocabulary rows in whole 128-blocks (`splits::vocab_range`) |
//! | embedding | rank 0 only at TP2 (its rows ride the glue message), replicated otherwise |
//! | vision tower, image tokens | rank 0 |
//! | Engram tables | NOT in the shard: packed head files, shipped separately (`engram.rs`) |

use super::splits::{self, Axis, Cut, SliceSpec, TensorKind, HAD, QGROUP};
use crate::dsv41::census::classify::{Role, TensorClass as C};
use crate::dsv41::census::{Census, Entry, LinearFormat, Module};
use crate::dsv41::config::Dsv41Config;
use anyhow::{bail, Context, Result};

/// How the routed / shared / DSpark experts are divided.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertSplit { Inner, Ep }

impl ExpertSplit {
    pub fn parse(s: &str) -> Option<Self> { match s { "inner" | "auto" => Some(ExpertSplit::Inner), "ep" => Some(ExpertSplit::Ep), _ => None } }
    pub fn name(self) -> &'static str { match self { ExpertSplit::Inner => "inner", ExpertSplit::Ep => "ep" } }
}

/// Everything the table above varies with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanOpts {
    pub world: usize,
    pub expert_split: ExpertSplit,
    /// `wq_b` / `wo_a` / `wo_b` / sink / `hc_*_fn` whole on every rank (row-parallel prefill)
    pub attn_whole: bool,
    /// Engram `wkv` whole on every rank (always for native, whose FP8 layout has no regular input cut; TP4 / TP8)
    pub engram_wkv_whole: bool,
    pub embed_rank0_only: bool,
    /// leg PA1: `hc_*_fn` whole on every rank (TP2 with `--dsv41-hc-side`: each rank computes both K slices of the mix)
    pub hc_whole: bool,
}

impl PlanOpts {
    /// `attn_whole`: `auto` = TP3 and up (T0 §8 + arbitrary-W TP3/TP6), `on` / `off` force it.
    pub fn resolve(world: usize, expert_split: ExpertSplit, attn_whole: &str, native: bool) -> Result<PlanOpts> {
        let aw = match attn_whole { "on" => true, "off" => false, "auto" | "" => world >= 3, x => bail!("--dsv41-attn-whole takes auto|on|off, not {x:?}") };
        Ok(PlanOpts { world, expert_split, attn_whole: aw, engram_wkv_whole: world >= 3 || native, embed_rank0_only: world == 2, hc_whole: false })
    }
    /// leg PA1: hold `hc_*_fn` whole (see `hc_whole`).
    pub fn with_hc_whole(mut self, on: bool) -> PlanOpts { self.hc_whole = on; self }
}

/// What rank `r` does with one module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement { Absent, Whole, Cut(Cut) }

/// The accounting bucket of a held tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Bucket { Replicated, Split, Experts, RankZero }

impl Bucket { pub fn name(self) -> &'static str { match self { Bucket::Replicated => "replicated", Bucket::Split => "split", Bucket::Experts => "experts", Bucket::RankZero => "rank-0 only" } } }

fn is_exl3(m: &Module) -> bool { matches!(m.format, LinearFormat::Exl3 { .. }) }

/// The rank's cut of an expert-shaped module's inner width (`moe_inter`): EXL3 blocks of 128 rotated by `id`,
/// native groups of 32.
fn inner_cut(cfg: &Dsv41Config, m: &Module, axis: Axis, o: &PlanOpts, rank: usize, id: usize) -> Cut {
    let unit = if is_exl3(m) { HAD } else { QGROUP };
    let total = cfg.moe_inter as u64;
    let (a, n) = splits::part(total / unit, o.world, rank, id);
    Cut::range(axis, total, a * unit, n * unit)
}

/// The placement of module `m` on rank `rank`.
pub fn placement(cfg: &Dsv41Config, m: &Module, o: &PlanOpts, rank: usize) -> Placement {
    let w = o.world;
    if w == 1 { return if m.class == C::EngramTable { Placement::Absent } else { Placement::Whole }; }
    let group_rows = cfg.o_lora_rank as u64;
    match m.class {
        C::EngramTable => Placement::Absent,
        C::Embed | C::DsTiedEmbed => if o.embed_rank0_only && rank != 0 { Placement::Absent } else { Placement::Whole },
        C::VisPatchEmbed | C::VisNorm | C::VisQkv | C::VisWo | C::VisMlpW1 | C::VisMlpW2 | C::AlignerW1 | C::AlignerW2 | C::ImageSep =>
            if rank == 0 { Placement::Whole } else { Placement::Absent },
        C::Head | C::DsTiedHead | C::DsMarkovHead => {
            let r = splits::vocab_range(cfg.vocab, w, rank);
            Placement::Cut(Cut::range(Axis::Out, cfg.vocab as u64, r.start as u64, r.len() as u64))
        }
        C::AttnSink | C::AttnWqB | C::AttnWoB | C::HcAttnFn | C::HcFfnFn if o.attn_whole => Placement::Whole,
        C::HcAttnFn | C::HcFfnFn if o.hc_whole => Placement::Whole,
        C::AttnWoA if o.attn_whole => Placement::Whole,
        C::AttnSink => {
            let hpg = (cfg.heads / cfg.o_groups) as u64;
            let (g0, ng) = splits::part(cfg.o_groups as u64, w, rank, 0);
            Placement::Cut(Cut::range(Axis::Out, cfg.heads as u64, g0 * hpg, ng * hpg))
        }
        C::AttnWqB => {
            let hd = cfg.head_dim as u64;
            let hpg = (cfg.heads / cfg.o_groups) as u64;
            let (g0, ng) = splits::part(cfg.o_groups as u64, w, rank, 0);
            Placement::Cut(Cut::range(Axis::Out, cfg.heads as u64 * hd, g0 * hpg * hd, ng * hpg * hd))
        }
        C::AttnWoA => {
            let (a, n) = splits::part(cfg.o_groups as u64, w, rank, 0);
            match m.sub {
                // EXL3: one module per group
                Some(g) => if (a..a + n).contains(&(g as u64)) { Placement::Whole } else { Placement::Absent },
                // native: one [o_groups * o_lora, in] matrix, group rows contiguous
                None => Placement::Cut(Cut::range(Axis::Out, cfg.o_groups as u64 * group_rows, a * group_rows, n * group_rows)),
            }
        }
        C::AttnWoB => {
            let (a, n) = splits::part(cfg.o_groups as u64, w, rank, 0);
            Placement::Cut(Cut::range(Axis::In, cfg.o_groups as u64 * group_rows, a * group_rows, n * group_rows))
        }
        C::HcAttnFn | C::HcFfnFn => {
            let k = (cfg.hc * cfg.hidden) as u64;
            let (a, n) = splits::part(k / HAD, w, rank, 0);
            Placement::Cut(Cut::range(Axis::In, k, a * HAD, n * HAD))
        }
        C::DsMainProj => {
            let k = (cfg.dspark_targets.len() * cfg.hidden) as u64;
            let (a, n) = splits::part(k / HAD, w, rank, 0);
            Placement::Cut(Cut::range(Axis::In, k, a * HAD, n * HAD))
        }
        C::EngramWkv => {
            if o.engram_wkv_whole { return Placement::Whole; }
            // [n-gram orders][heads][head_dim]: this rank's heads of every order (same split as engram::heads_of_rank)
            let (orders, per) = ((cfg.engram_max_ngram - 1) as u64, cfg.engram_head_dim as u64);
            let er = super::engram::heads_of_rank(cfg.engram_heads, w, rank);
            Placement::Cut(Cut { axis: Axis::In, total: orders * cfg.engram_heads as u64 * per, groups: orders, start: er.start as u64 * per, len: er.len() as u64 * per })
        }
        C::ExpertW1 | C::ExpertW2 | C::ExpertW3 => {
            let layer = m.layer.unwrap_or(0) as usize;
            if o.expert_split == ExpertSplit::Ep {
                let n = cfg.moe_of(layer).0 as u64;
                let (a, c) = splits::part(n, w, rank, 0);
                return if (a..a + c).contains(&(m.sub.unwrap_or(0) as u64)) { Placement::Whole } else { Placement::Absent };
            }
            let axis = if m.class == C::ExpertW2 { Axis::In } else { Axis::Out };
            Placement::Cut(inner_cut(cfg, m, axis, o, rank, m.sub.unwrap_or(0) as usize))
        }
        C::SharedW1 | C::SharedW2 | C::SharedW3 => {
            let axis = if m.class == C::SharedW2 { Axis::In } else { Axis::Out };
            Placement::Cut(inner_cut(cfg, m, axis, o, rank, 0))
        }
        _ => Placement::Whole,
    }
}

/// How one tensor's bytes are laid out relative to its module's logical dimensions.
pub fn tensor_kind(e: &Entry, f: &LinearFormat) -> TensorKind {
    match (f, e.role) {
        (LinearFormat::Exl3 { .. }, Role::Trellis) => TensorKind::Exl3Trellis,
        (LinearFormat::Exl3 { .. }, Role::Suh) => TensorKind::Exl3Suh,
        (LinearFormat::Exl3 { .. }, Role::Svh) => TensorKind::Exl3Svh,
        (LinearFormat::Plain(_), Role::Weight) | (LinearFormat::Fp8Block { .. }, Role::Weight) => TensorKind::Matrix { bo: 1, bi: 1, pack: 1 },
        (LinearFormat::Fp8Block { bo, bi, .. }, Role::Scale) => TensorKind::Matrix { bo: *bo as u64, bi: *bi as u64, pack: 1 },
        (LinearFormat::Fp4 { .. }, Role::Weight) => TensorKind::Matrix { bo: 1, bi: 1, pack: 2 },
        (LinearFormat::Fp4 { block, .. }, Role::Scale) => TensorKind::Matrix { bo: 1, bi: *block as u64, pack: 1 },
        (_, Role::Bias) => TensorKind::Vector,
        (LinearFormat::Param(_), _) => match e.shape.len() { 1 => TensorKind::Vector, 2 => TensorKind::Matrix { bo: 1, bi: 1, pack: 1 }, _ => TensorKind::Whole },
        _ => TensorKind::Whole,
    }
}

/// One tensor as a rank holds it.
#[derive(Clone, Debug)]
pub struct Hold {
    /// index into `Census::entries`
    pub entry: u32,
    /// `None` = the whole tensor
    pub slice: Option<SliceSpec>,
    /// the held shape (the full shape when whole)
    pub shape: Vec<u64>,
    pub bytes: u64,
    pub bucket: Bucket,
}

/// One rank's selection.
#[derive(Clone, Debug)]
pub struct RankSelection {
    pub rank: usize,
    pub world: usize,
    pub holds: Vec<Hold>,
}

impl RankSelection {
    pub fn bytes(&self) -> u64 { self.holds.iter().map(|h| h.bytes).sum() }
    pub fn by_bucket(&self) -> std::collections::BTreeMap<Bucket, u64> {
        let mut m = std::collections::BTreeMap::new();
        for h in &self.holds { *m.entry(h.bucket).or_insert(0) += h.bytes; }
        m
    }
}

/// The bucket a module's held bytes count in.
fn bucket_of(m: &Module, p: &Placement) -> Bucket {
    match (p, m.class) {
        (Placement::Whole, C::VisPatchEmbed | C::VisNorm | C::VisQkv | C::VisWo | C::VisMlpW1 | C::VisMlpW2 | C::AlignerW1 | C::AlignerW2 | C::ImageSep) => Bucket::RankZero,
        (Placement::Whole, C::Embed | C::DsTiedEmbed) => Bucket::RankZero,
        (_, C::ExpertW1 | C::ExpertW2 | C::ExpertW3 | C::SharedW1 | C::SharedW2 | C::SharedW3) => Bucket::Experts,
        (Placement::Cut(_), _) => Bucket::Split,
        (Placement::Whole, C::AttnWqB | C::AttnWoA | C::AttnWoB | C::AttnSink | C::HcAttnFn | C::HcFfnFn | C::EngramWkv) => Bucket::Split,
        _ => Bucket::Replicated,
    }
}

/// The selection of rank `rank`: every tensor of the census, absent / whole / sliced.
pub fn select(c: &Census, o: &PlanOpts, rank: usize) -> Result<RankSelection> {
    let mut holds = Vec::new();
    let placements: Vec<Placement> = c.modules.iter().map(|m| placement(&c.cfg, m, o, rank)).collect();
    for (i, e) in c.entries.iter().enumerate() {
        let m = &c.modules[e.module as usize];
        let p = placements[e.module as usize];
        let bucket = bucket_of(m, &p);
        match p {
            Placement::Absent => {}
            Placement::Whole => holds.push(Hold { entry: i as u32, slice: None, shape: e.shape.clone(), bytes: e.bytes, bucket }),
            Placement::Cut(cut) => {
                let elt = splits::elt_bytes(&e.dtype).with_context(|| e.name.clone())?;
                match splits::cut_tensor(tensor_kind(e, &m.format), &e.shape, elt, &cut).with_context(|| format!("{}: cut {cut:?} on {:?}", e.name, m.format.label()))? {
                    Some((spec, shape)) => {
                        let (bytes, whole) = (spec.bytes(), spec.is_whole());
                        holds.push(Hold { entry: i as u32, slice: if whole { None } else { Some(spec) }, shape: if whole { e.shape.clone() } else { shape }, bytes, bucket });
                    }
                    // the cut leaves this tensor (mul1, the other axis's vector, a norm) whole on every rank
                    None => holds.push(Hold { entry: i as u32, slice: None, shape: e.shape.clone(), bytes: e.bytes, bucket: if bucket == Bucket::Experts { Bucket::Experts } else { Bucket::Replicated } }),
                }
            }
        }
    }
    Ok(RankSelection { rank, world: o.world, holds })
}

/// The proof the plan is a plan: over all ranks every tensor is held by someone, a cut tensor's slices tile
/// its bytes exactly once, a whole one is whole. Returns (bytes of every tensor once, bytes held in total
/// over the ranks, per-rank selections).
pub fn check_all(c: &Census, o: &PlanOpts) -> Result<(u64, u64, Vec<RankSelection>)> {
    let sels: Vec<RankSelection> = (0..o.world).map(|r| select(c, o, r)).collect::<Result<_>>()?;
    let mut copies = vec![0u32; c.entries.len()];
    let mut sliced = vec![0u64; c.entries.len()];
    for s in &sels {
        for h in &s.holds {
            copies[h.entry as usize] += 1;
            if h.slice.is_some() { sliced[h.entry as usize] += h.bytes; }
        }
    }
    let mut uniq = 0u64;
    for (i, e) in c.entries.iter().enumerate() {
        if e.class == C::EngramTable { if copies[i] != 0 { bail!("{}: Engram tables must not ship in the shard", e.name); } continue; }
        uniq += e.bytes;
        if copies[i] == 0 { bail!("{}: held by no rank at world {} ({:?})", e.name, o.world, o.expert_split); }
        if sliced[i] != 0 && sliced[i] != e.bytes && (copies[i] as usize) == o.world {
            bail!("{}: its slices hold {} B of {} B", e.name, sliced[i], e.bytes);
        }
    }
    let held: u64 = sels.iter().map(|s| s.bytes()).sum();
    Ok((uniq, held, sels))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsv41::config::{test_config_json, Dsv41Config};

    fn cfg() -> Dsv41Config { Dsv41Config::from_value(&test_config_json(), None).unwrap() }

    fn module(class: C, layer: u16, sub: Option<u16>, f: LinearFormat) -> Module {
        Module { key: String::new(), class, layer: Some(layer), sub, format: f, logical: None, tensors: vec![], bytes: 0 }
    }
    fn exl3() -> LinearFormat { LinearFormat::Exl3 { k: 3, mul1: true, suh: true, svh: true } }

    #[test]
    fn expert_inner_cuts_tile_the_2304_inner_width_for_every_expert_and_world() {
        let c = cfg();
        for w in [2usize, 4, 8] {
            let o = PlanOpts::resolve(w, ExpertSplit::Inner, "auto", false).unwrap();
            for e in 0..6u16 {
                let m = module(C::ExpertW1, 3, Some(e), exl3());
                let mut next = 0;
                for r in 0..w {
                    let Placement::Cut(cut) = placement(&c, &m, &o, r) else { panic!() };
                    assert_eq!((cut.axis, cut.total, cut.start % 128, cut.len % 128), (Axis::Out, 2304, 0, 0));
                    assert_eq!(cut.start, next, "w{w} e{e} r{r}");
                    next += cut.len;
                }
                assert_eq!(next, 2304);
            }
            // native: 32-wide groups, equal
            let m = module(C::ExpertW2, 3, Some(1), LinearFormat::Fp4 { block: 32, scale: "F8_E8M0".into(), scale2: false });
            let Placement::Cut(cut) = placement(&c, &m, &o, 0) else { panic!() };
            assert_eq!((cut.axis, cut.len), (Axis::In, 2304 / w as u64));
        }
    }

    #[test]
    fn ep_deals_whole_experts_and_inner_never_leaves_a_rank_without_an_expert() {
        let c = cfg();
        let o = PlanOpts::resolve(4, ExpertSplit::Ep, "auto", false).unwrap();
        let holders = |e: u16| (0..4).filter(|&r| placement(&c, &module(C::ExpertW1, 3, Some(e), exl3()), &o, r) == Placement::Whole).count();
        for e in [0u16, 95, 96, 383] { assert_eq!(holders(e), 1, "expert {e}"); }
        let o = PlanOpts::resolve(4, ExpertSplit::Inner, "auto", false).unwrap();
        assert!((0..4).all(|r| matches!(placement(&c, &module(C::ExpertW1, 3, Some(0), exl3()), &o, r), Placement::Cut(_))));
    }

    #[test]
    fn attention_is_sliced_at_tp2_and_whole_at_tp4() {
        let c = cfg();
        let tp2 = PlanOpts::resolve(2, ExpertSplit::Inner, "auto", false).unwrap();
        let tp4 = PlanOpts::resolve(4, ExpertSplit::Inner, "auto", false).unwrap();
        let wq = module(C::AttnWqB, 3, None, exl3());
        assert_eq!(placement(&c, &wq, &tp2, 1), Placement::Cut(Cut::range(Axis::Out, 32768, 32 * 512, 32 * 512)));
        assert_eq!(placement(&c, &wq, &tp4, 1), Placement::Whole);
        // EXL3 wo_a slices: the rank's groups whole, the others absent
        let wo = |g| module(C::AttnWoA, 3, Some(g), exl3());
        assert_eq!((0..8).filter(|g| placement(&c, &wo(*g), &tp2, 1) == Placement::Whole).collect::<Vec<_>>(), vec![4, 5, 6, 7]);
        assert_eq!(placement(&c, &wo(0), &tp4, 3), Placement::Whole);
        // wo_b: the input rows of its groups
        assert_eq!(placement(&c, &module(C::AttnWoB, 3, None, exl3()), &tp2, 0), Placement::Cut(Cut::range(Axis::In, 8192, 0, 4096)));
        let off = PlanOpts::resolve(4, ExpertSplit::Inner, "off", false).unwrap();
        assert!(matches!(placement(&c, &wq, &off, 1), Placement::Cut(_)));
        assert!(PlanOpts::resolve(4, ExpertSplit::Inner, "maybe", false).is_err());
    }

    #[test]
    fn engram_wkv_keeps_the_ranks_heads_of_every_order_at_tp2_and_is_whole_at_tp4() {
        let c = cfg();
        let m = module(C::EngramWkv, 1, None, exl3());
        let tp2 = PlanOpts::resolve(2, ExpertSplit::Inner, "auto", false).unwrap();
        assert_eq!(placement(&c, &m, &tp2, 1), Placement::Cut(Cut { axis: Axis::In, total: 6144, groups: 3, start: 1024, len: 1024 }));
        assert_eq!(placement(&c, &m, &PlanOpts::resolve(4, ExpertSplit::Inner, "auto", false).unwrap(), 1), Placement::Whole);
        assert_eq!(placement(&c, &m, &PlanOpts::resolve(2, ExpertSplit::Inner, "auto", true).unwrap(), 1), Placement::Whole, "native: no regular cut");
        // When engram_wkv_whole is off, placement must match engram::heads_of_rank for every world (including TP3/TP6).
        for w in [2usize, 3, 4, 6, 8] {
            let mut o = PlanOpts::resolve(w, ExpertSplit::Inner, "auto", false).unwrap();
            o.engram_wkv_whole = false;
            let per = c.engram_head_dim as u64;
            for r in 0..w {
                let er = super::super::engram::heads_of_rank(c.engram_heads, w, r);
                assert_eq!(
                    placement(&c, &m, &o, r),
                    Placement::Cut(Cut { axis: Axis::In, total: 6144, groups: 3, start: er.start as u64 * per, len: er.len() as u64 * per }),
                    "world {w} rank {r}"
                );
            }
        }
    }

    #[test]
    fn head_embed_vision_and_tables() {
        let c = cfg();
        let tp2 = PlanOpts::resolve(2, ExpertSplit::Inner, "auto", false).unwrap();
        let tp4 = PlanOpts::resolve(4, ExpertSplit::Inner, "auto", false).unwrap();
        let m = |cl| Module { key: String::new(), class: cl, layer: None, sub: None, format: exl3(), logical: None, tensors: vec![], bytes: 0 };
        assert_eq!(placement(&c, &m(C::Embed), &tp2, 1), Placement::Absent);
        assert_eq!(placement(&c, &m(C::Embed), &tp2, 0), Placement::Whole);
        assert_eq!(placement(&c, &m(C::Embed), &tp4, 3), Placement::Whole);
        assert_eq!(placement(&c, &m(C::VisQkv), &tp4, 1), Placement::Absent);
        assert_eq!(placement(&c, &m(C::EngramTable), &tp4, 0), Placement::Absent);
        assert_eq!(placement(&c, &m(C::Head), &tp4, 0), Placement::Cut(Cut::range(Axis::Out, 129280, 0, 253 * 128)));
        let one = PlanOpts::resolve(1, ExpertSplit::Inner, "auto", false).unwrap();
        assert_eq!(placement(&c, &m(C::Head), &one, 0), Placement::Whole);
    }
}
