//! DSV4.1 TP (leg TP): the full 40-layer trunk on one rank of a TP group (world 1, 2 or 4), on leg SL's
//! per-layer pieces (`fwd::layer::Fwd`) and leg TPC's comm layer (`Dsv41Comm`), split per T0
//! (`PLAN/DSV41_TP_DESIGN.md` §1, §3):
//!
//! | piece | split | here |
//! |---|---|---|
//! | `hc_attn_fn` / `hc_ffn_fn` | K slice per rank; the 25 partials (24 dots + sum of squares) ride the sublayer's reduce | [`TpModel::mix_partial`] / [`TpModel::mix_finish`] |
//! | `wq_a`, `wkv`, q / kv norms, window, compressor, indexer, router, norms, Engram gate | replicated | SL's pieces |
//! | `wq_b`, sink, sparse attention | the rank's heads | [`TpModel::attn_core`] |
//! | `wo_a` / `wo_b` | the rank's o_groups / their input range: an FP32 partial in the attention frame | [`TpModel::attn_core`] |
//! | routed + shared experts | inner width (KE2's fused call): an FP32 partial in the FFN frame | [`TpModel::ffn_core`] |
//! | Engram `wkv` | the rank's heads' columns; the partial rides the FFN reduce of the layer before | [`TpModel::engram_partial`] |
//! | head | the rank's vocabulary rows; TPC's head exchange (greedy / sample / probe log-prob) | [`TpModel::head`] |
//! | embedding | gathered where held; TP2: rank 0 gathers, the fp32 all-reduce broadcasts it (others add zeros) | [`TpModel::embed`] |
//!
//! Reduces: every frame is summed in place by `TpComm::all_reduce_raw` (the transport's fixed association,
//! the same bits on every rank), then rounded to bf16 once by the consumer. One call = `rows` tokens of the
//! lane at positions `p0..p0 + rows`: a decode step (1 row), a verify-shaped step (6 / 24 rows) or a prefill
//! chunk (this leg's prefill is the FULL non-CED one: all 40 layers on every prompt row, chunked).
//!
//! State per lane (one lane here): the SWA window per layer (ring of 134 + linear rows), the KV sources'
//! compressor caches (main KV, index-K planes, frontier ring), the Engram hash state. A call wider than the
//! backtrack (6) copies its window rows below p0 from the ring into linear rows first, so the ring writes of
//! the same call never feed a query.

use super::comm::{Dsv41Comm, HeadCall, TpComm, TpOut, TpRow};
use super::frame::{Site, MIX_STRIDE};
use super::tpk::TpfKernels;
use super::wload::{held_k, Global, RankSrc};
use crate::dsv41::census::classify::TensorClass as C;
use crate::dsv41::config::Dsv41Config;
use crate::dsv41::engram::hash::{EngramHasher, NgramHashState};
use crate::dsv41::engram::layout::{EngramConfig, EngramLayout};
use crate::dsv41::engram::prefetch::{EngramIo, EngramIoConfig};
use crate::dsv41::fwd::dev::{bf16, f32a, Buf, Gpu};
use crate::dsv41::fwd::glue;
use crate::dsv41::fwd::layer::{fmt_of, Bump, Fwd, SrcState, Union, Window};
use crate::dsv41::fwd::receipts;
use crate::dsv41::fwd::weights::LayerW;
use crate::dsv41::gpu::{attn, kv, mixers as km};
use crate::dsv41::load::splits;
use crate::dsv41::weights::exl3::moe::{moe_ex_ow, MoePath};
use crate::dsv41::weights::{Accum, BackendKind, DenseWeight, DevPtr, MoeWeight, Route, Scratch, Shard};
use anyhow::{ensure, Context, Result};
use cudarc::driver::CudaStream;
use std::collections::HashMap;
use std::sync::Arc;

/// The rank's K slice of one `hc_*_fn` as resident: the kernel reads `fn_ptr` (column `k0` of the slice) with row
/// stride `fn_ld`, over stream elements `[k0, k0 + klen)`.
#[derive(Clone, Copy, Debug)]
pub struct MixK { pub fn_ptr: u64, pub fn_ld: usize, pub k0: usize, pub klen: usize,
                  /// leg PA1: column 0 of the WHOLE tensor when the rank holds it whole (TP2 with `--dsv41-hc-side`), else 0
                  pub fn_base: u64 }

/// One Engram `wkv` call of a rank: the weight (whole over a K range, or the held grouped cut), the column offset of
/// its input inside the EIO row, and whether it adds into the partial.
pub struct EngramCall { pub w: (DenseWeight, BackendKind), pub x_off: usize, pub k: usize }

/// One trunk layer on a rank: SL's `LayerW` plus the TP views the forward calls.
pub struct TpLayer {
    pub w: LayerW,
    /// the shared expert as a one-expert table (KE2's fused routed + shared call)
    pub shared: MoeWeight,
    pub wq_b: (DenseWeight, BackendKind),
    pub wo_b: (DenseWeight, BackendKind),
    pub sink: u64,
    pub mix_a: MixK,
    pub mix_f: MixK,
    pub engram: Vec<EngramCall>,
}

/// The rank's ranges (T0 §1; `load::splits::part` = the shipped placement's own rule).
#[derive(Clone, Copy, Debug)]
pub struct Ranges { pub h0: usize, pub nh: usize, pub g0: usize, pub ng: usize, pub k0: usize, pub klen: usize, pub eh0: usize, pub neh: usize }

impl Ranges {
    pub fn of(cfg: &Dsv41Config, world: usize, rank: usize) -> Ranges {
        let (g0, ng) = splits::part(cfg.o_groups as u64, world, rank, 0);
        let hpg = (cfg.heads / cfg.o_groups) as u64;
        let (h0, nh) = (g0 * hpg, ng * hpg);
        let k = (cfg.hc * cfg.hidden) as u64;
        let (a, n) = splits::part(k / 128, world, rank, 0);
        let er = crate::dsv41::load::engram::heads_of_rank(cfg.engram_heads, world, rank);
        Ranges { h0: h0 as usize, nh: nh as usize, g0: g0 as usize, ng: ng as usize, k0: (a * 128) as usize, klen: (n * 128) as usize,
                 eh0: er.start, neh: er.len() }
    }
}

fn param_elt(dtype: &str) -> u64 { match dtype { "F32" => 4, _ => 2 } }

impl TpLayer {
    pub fn new(w: LayerW, cfg: &Dsv41Config, rg: &Ranges, world: usize) -> Result<TpLayer> {
        let l = w.layer;
        let hd = cfg.head_dim as u64;
        let mut wq_b = w.lin(C::AttnWqB)?.clone();
        if matches!(wq_b.0.shard, Shard::Whole) && world > 1 { wq_b.0.shard = Shard::Out(rg.h0 as u64 * hd..(rg.h0 + rg.nh) as u64 * hd); }
        let gin = (cfg.o_lora_rank) as u64;
        let mut wo_b = w.lin(C::AttnWoB)?.clone();
        if matches!(wo_b.0.shard, Shard::Whole) && world > 1 { wo_b.0.shard = Shard::In(rg.g0 as u64 * gin..(rg.g0 + rg.ng) as u64 * gin); }
        let s = w.param(C::AttnSink)?;
        let sink = if s.shape[0] as usize == cfg.heads { s.ptr.0 + rg.h0 as u64 * param_elt(&s.dtype) } else {
            ensure!(s.shape[0] as usize == rg.nh, "layer {l}: sink {:?} is neither the whole [{}] nor the rank's [{}]", s.shape, cfg.heads, rg.nh);
            s.ptr.0
        };
        ensure!(param_elt(&s.dtype) == 4, "layer {l}: the attention sink is {} (f32 expected)", s.dtype);
        let mixk = |c: C| -> Result<MixK> {
            let t = w.param(c)?;
            let kres = t.shape[1] as usize;
            let kfull = cfg.hc * cfg.hidden;
            if kres == kfull { Ok(MixK { fn_ptr: t.ptr.0 + 4 * rg.k0 as u64, fn_ld: kfull, k0: rg.k0, klen: rg.klen, fn_base: t.ptr.0 }) }
            else { ensure!(kres == rg.klen, "layer {l}: {} {:?} is neither whole nor the rank's K slice {}", c.name(), t.shape, rg.klen);
                   Ok(MixK { fn_ptr: t.ptr.0, fn_ld: kres, k0: rg.k0, klen: rg.klen, fn_base: 0 }) }
        };
        let (mix_a, mix_f) = (mixk(C::HcAttnFn)?, mixk(C::HcFfnFn)?);
        // the shared expert, as KE2's one-expert table
        let (s1, s2, s3) = (w.lin(C::SharedW1)?.0.clone(), w.lin(C::SharedW2)?.0.clone(), w.lin(C::SharedW3)?.0.clone());
        let shared = MoeWeight { layer: l as u16, n_experts: 1, local: 0..1, top_k: 1, dim: cfg.hidden, inter: cfg.moe_inter, swiglu_limit: cfg.swiglu_limit as f32,
                                 experts: vec![[s1, s2, s3]] };
        // Engram wkv (layers 1, 14): the rank's columns c = g * heads + h, h in [eh0, eh0 + neh), g = n-gram order
        let mut engram = Vec::new();
        if let Ok(ew) = w.lin(C::EngramWkv) {
            let per = cfg.engram_head_dim;
            let orders = cfg.engram_max_ngram - 1;
            let kfull = orders * cfg.engram_heads * per;
            let held = held_k(&ew.0).context("Engram wkv: no trellis (EXL3 expected)")? as usize;
            if held == kfull {
                for g in 0..orders {
                    let a = (g * cfg.engram_heads + rg.eh0) * per;
                    let mut v = ew.clone();
                    v.0.shard = Shard::In(a as u64..(a + rg.neh * per) as u64);
                    engram.push(EngramCall { w: v, x_off: g * rg.neh * per, k: rg.neh * per });
                }
            } else {
                ensure!(held == orders * rg.neh * per && ew.0.inn as usize == held, "layer {l}: Engram wkv held K {held} (inn {}) != {} for heads {}..{}", ew.0.inn,
                        orders * rg.neh * per, rg.eh0, rg.eh0 + rg.neh);
                engram.push(EngramCall { w: ew.clone(), x_off: 0, k: held });
            }
        }
        Ok(TpLayer { w, shared, wq_b, wo_b, sink, mix_a, mix_f, engram })
    }
}

/// One lane's decode state (leg FW: one per concurrent request, up to `MAX_LANES`).
pub struct Lane {
    pub pos: usize,
    pub win: Vec<Window>,
    pub src: HashMap<usize, SrcState>,
    pub hash: NgramHashState,
    pub cap_rows: usize,
    pub max_seq: usize,
    /// the tokens at positions [0, pos) (the Engram hash state after a speculative rollback is rebuilt from the tail)
    pub hist: Vec<u32>,
    /// sampling of this lane's head rows (TPC's exchange): temperature 0 = greedy
    pub temp: f32,
    pub seed: u64,
    /// what the state holds at `pos` (a CED prefill's mid-prompt state lacks the decoder rings): which resumes are exact
    pub kind: crate::dsv41::ced::schedule::StateKind,
    /// leg VIS1: per-prompt-position VL token type (`TEXT = -1`, `0..=3` image span); empty for text-only requests
    pub vl_types: Vec<i8>,
    /// leg VIS1: Rank-0 splice ops for `0..vl_types.len()` (`-1` text, `-2` start, `-3` newline, `-4` end, `>= 0` aligner row)
    pub vl_row_ops: Vec<i32>,
    /// leg VIS1: Rank-0 temporary `[uncached_aligner_rows, hidden]` BF16 buffer during prefill; freed when prefill completes
    pub vl_aligner: Option<Buf>,
}

/// Lanes one TP model serves at most (THE POSTURE: 4 default, up to 8 concurrent requests; KA addresses 8 per call).
pub const MAX_LANES: usize = 8;

/// The DSpark taps a call writes: call rows `[row0, row0 + n)` -> rows `dst0..` of `buf` (bf16 [..][ld], target t at
/// column t * dim); `rider`: the call's `main_proj` partial rides layer 39's attention reduce (every row tapped).
#[derive(Clone, Copy, Debug)]
pub struct TapReq { pub buf: u64, pub ld: usize, pub row0: usize, pub n: usize, pub dst0: usize, pub rider: bool }

/// One lane's rows inside a call: rows `[row0, row0 + n)` of the call are positions `[p0, p0 + n)` of lane `lane`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Seg { pub lane: usize, pub row0: usize, pub n: usize, pub p0: usize }

/// What a call returns from the head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadWant { None, Last, All }

pub struct TpModel<'a> {
    pub f: Fwd<'a>,
    /// the second bump arena: layer l's outputs stay valid while layer l + 1 runs (the two arenas alternate)
    pub(super) spare: Bump,
    pub comm: &'a TpComm,
    pub cs: &'a CudaStream,
    pub k: TpfKernels,
    pub cfg: &'a Dsv41Config,
    pub rg: Ranges,
    /// every trunk layer this rank's source holds (a slice pack holds some; the forward needs all 40)
    pub layers: Vec<Option<TpLayer>>,
    pub glob: Global,
    /// leg VIS1: Rank-0 resident Vision Tower + 3x3 Aligner + delimiter embeddings (`None` on ranks 1..N or when `--dsv41-vision off`)
    pub vision: Option<crate::dsv41::vision::tower::VisionWeights>,
    /// leg VIS1: persistent `u8 [max_rows]` image-mask buffer (`1` on image-span tokens, `0` on text) surviving `swap_bump()` across layers
    pub(super) vl_mask_buf: Buf,
    /// the lanes (leg FW); a call addresses any subset of them (`forward_segs`)
    pub lanes: Vec<Lane>,
    /// the segments of the call in flight (row -> lane); set by `forward_segs`
    pub segs: Vec<Seg>,
    /// i32 [max_rows] lane index of every row of the call (KA's per-row lane)
    pub(super) lane_buf: Buf,
    /// leg PG1: graph keys by total rows (the segment table and the per-row lane ids on the device)
    pub(super) rk: super::graph::RowKeys,
    pub eio: Option<EngramIo>,
    pub hasher: Option<EngramHasher>,
    /// the frames (A attention, B FFN), sized for `max_rows` with the Engram rider
    pub(super) frames: [Buf; 2],
    /// the embedding broadcast frame and the call's positions / ids (leg FW's prefill driver reads them)
    pub(super) pos_dev: u64,
    pub(super) ids_dev: u64,
    /// the embedding broadcast frame (f32 [max_rows][hidden])
    pub(super) bcast: Buf,
    /// EIO output [layers][max_tok][n_own][256] bf16
    pub(super) eout: Buf,
    eio_max: usize,
    /// leg PH1: the Engram row decode of a forward call runs on the compute stream inside the round (its graph), right
    /// before its first reader (layer 0's FFN `wkv` partial), instead of on the device stream before the call
    /// (`--dsv41-engram-in-graph`, default on); `eng_pending` = the call's decode step, submitted and not yet launched
    pub(super) eng_in_graph: bool,
    eng_pending: Option<crate::dsv41::gpu::engram::StepLaunch>,
    /// the attention decode scratch (<= 24 rows)
    pub(super) ds: (Buf, Buf),
    pub max_rows: usize,
    /// leg P4: a prefill call's last chunk of up to this many rows rides the chunk before it (`--dsv41-prefill-tail-merge`; the buffers hold `max_rows` = chunk + this)
    pub tail_merge: usize,
    pub world: usize,
    pub rank: usize,
    /// per-row head inputs / outputs (<= 24 rows per exchange)
    pub(super) rows_p: Buf,
    pub(super) outs: Buf,
    pub(super) pos_buf: Buf,
    pub(super) ids_buf: Buf,
    /// per index source: its selection [max_rows][512] i32 for the layers that reuse it
    pub(super) sel_bufs: HashMap<usize, Buf>,
    /// TP2: only rank 0 holds the embedding; its rows are broadcast by the fp32 all-reduce
    pub embed_bcast: bool,
    /// leg Q1: full-logit taps of the head (None = off; `tp::q1tap`)
    pub logit_tap: Option<super::q1tap::LogitTap>,
    /// 16 floats for the settle reduce
    pub(super) tiny: Buf,
    /// i32 [MAX_ROW_BLOCKS] merge tickets of the indexer's fused merge (zeroed once; the kernel re-zeroes them)
    pub(super) tick: Buf,
    /// layer 20's candidate union (u32 [MAX_ROW_BLOCKS row blocks][8 x 2048]) and counts, reused by every call
    pub(super) uni_bufs: Option<(Buf, Buf)>,
    /// the DSpark taps of the next call (leg FW): which rows, where to, and whether `main_proj` rides layer 39's reduce
    pub taps: Option<TapReq>,
    /// the DSpark drafter's `main_proj` (this rank's K slice), when the stages are loaded
    pub main_proj: Option<(DenseWeight, BackendKind)>,
    /// the last call's final stream and FFN record (the head's input; valid until the next call)
    pub(super) last: Option<(u64, u64)>,
    /// window keys below this position are masked (CE's opt-in truncated replay; 0 = none): the prefill entry with a floor
    pub win_lo: usize,
    /// the KC RoPE tables on the device ([0] plain, [1] compressed)
    pub(super) kc_freq: [Buf; 2],
    /// graph mode: the call is being captured (graph-safe kernels, capacity caps, prepared tables)
    pub gm: bool,
    pub gtab: Option<super::graph::GTab>,
    /// the captured decode / verify calls (None: graphs off)
    pub graphs: Option<super::graph::Graphs>,
    pub weights_bytes: u64,
    /// the MoE / dense scratch `f.scr` points into (leg MEM1: sized to the calls' largest bound, grown once by `load_stages`)
    pub(super) scr_buf: Buf,
    /// leg PF1 (no memset nodes in a round): `d41_mix_partial`'s last-block counters, zeroed once here and re-armed by the
    /// kernel's last arrival (every call of every layer reuses them: calls are stream-ordered); u32 [max_rows]
    pub(super) mix_cnt: Buf,
    /// leg PF1: the attention [0] and indexer [1] error words, zeroed once (the kernels only ever OR / CAS a code into them,
    /// so a nonzero word stays: sticky, read by the checks)
    pub(super) err_w: Buf,
    /// leg PF1: bf16 zeros [HEAD_MAX][hidden]: the embedding rows a rank without the table adds (a kernel node, not a memset)
    pub(super) zrows: Buf,
    /// leg PD1: the dense decode path's switches, hand-offs and the paced L2 prefetch (`tp::pd1`)
    pub pd1: super::pd1::Pd1,
    /// leg PP1: the tokens that follow the next prefill call's rows (the next call's chunk), sent by the head with the call: the
    /// call's last chunk reads their Engram rows into the cache while it computes (`--dsv41-pf-engram-ahead`); consumed by the call
    pub pf_ahead: Vec<u32>,
    /// leg PA1: the attention sublayer's fused helpers, attention prologue / merge and the hc side stream (`tp::pa1`)
    pub pa1: super::pa1::Pa1,
    /// leg P7COPY: lazy `(kv, score)` row stash (`<= 16` rows) per `(lane, src_layer)` for O(1) `WideCopyGuard` rollback
    pub(super) wide_stash: HashMap<(usize, usize), (Buf, Buf)>,
    /// leg P7SP: whether this rank's loaded shard holds `hc_*_fn` whole on every trunk layer, and the cluster-agreed flag
    pub(super) hc_whole_local: bool,
    pub(super) hc_whole_agreed: Option<bool>,
}

pub const HEAD_MAX: usize = super::frame::MAX_ROWS;
/// Row blocks of 8 rows one indexer call holds at most (the merge tickets and layer 20's persistent candidate union):
/// a non-graphed call of more than `8 * MAX_ROW_BLOCKS` rows must be split (the score path's calls are).
pub const MAX_ROW_BLOCKS: usize = 64;

/// The most rows one lane may have in a graph-captured call. Graph mode keeps a call's rows only in the lane's window
/// ring (`swa_slots()` slots, no linear rows) and writes them before any row attends: n rows overwrite position
/// p0 + n - 1 - slots, which row 0's window [p0 - window + 1, p0] must not hold, so n <= slots - window + 1. The cap is
/// the ring's design figure, the backtrack (a verify: DSpark block + 1). Leg FIXB: the former 8-row cap let a captured
/// 8-row call at P >= 127 read a future key in place of row 0's oldest window key (width 8 differed at a long context).
pub fn graph_seg_max(cfg: &Dsv41Config) -> usize { cfg.backtrack() }

impl<'a> TpModel<'a> {
    /// Load the rank's 40 trunk layers and globals, open the Engram runtime on the rank's heads, allocate the lane.
    #[allow(clippy::too_many_arguments)]
    pub fn new(g: &'a Gpu, cs: &'a CudaStream, cfg: &'a Dsv41Config, comm: &'a TpComm, src: &RankSrc, engram_dir: &std::path::Path, hash_dir: &std::path::Path,
               max_rows: usize, max_seq: usize, cache_mib: usize, n_lanes: usize, log: &mut dyn FnMut(String)) -> Result<TpModel<'a>> {
        ensure!((1..=MAX_LANES).contains(&n_lanes), "lanes {n_lanes} (1..={MAX_LANES})");
        let (world, rank) = (comm.world(), comm.rank());
        ensure!(src.world() == world && src.rank == rank, "the shard is rank {} of {}, the link rank {rank} of {world}", src.rank, src.world());
        let rg = Ranges::of(cfg, world, rank);
        crate::dsv41::weights::exl3::prepare()?;
        let bump_bytes = super::mem::arena_bytes(max_rows);
        // the scratch is sized once the weights are on the device (leg MEM1: the largest bound of the calls, not a fixed 1 GiB)
        let mut f = Fwd { g, cfg, scr: glue::Scr { ptr: 0, bytes: 0 }, b: Bump::new(g, bump_bytes)?, rope: crate::dsv41::fwd::rope_set(cfg)?, checks: false, wide_min: 0 };
        let spare = Bump::new(g, bump_bytes)?;
        let t0 = std::time::Instant::now();
        let mut layers = Vec::with_capacity(cfg.layers);
        let mut bytes = 0u64;
        let total: u64 = (0..cfg.layers).filter(|&l| src.has_layer(l)).map(|l| src.layer_bytes(l)).sum();
        let prog = crate::dsv41::load::progress(crate::dsv41::load::Phase::Upload, Some((rank, world)), total.max(1));
        for l in 0..cfg.layers {
            if !src.has_layer(l) { layers.push(None); continue; }
            let w = src.load_layer(g, l).with_context(|| format!("rank {rank}: layer {l}"))?;
            bytes += w.bytes;
            let tl = TpLayer::new(w, cfg, &rg, world)?;
            crate::dsv41::weights::exl3::moe::prepare_moe(&tl.w.moe.as_ref().context("no routed experts")?.0)?;
            crate::dsv41::weights::exl3::moe::prepare_moe(&tl.shared)?;
            layers.push(Some(tl));
            prog.set(bytes);
        }
        prog.finish();
        let glob = src.load_global(g)?;
        bytes += glob.bytes;
        let vision = src.load_vision(g)?;
        if let Some(ref v) = vision { bytes += v.bytes; }
        let vis_desc = match &vision {
            Some(v) => format!(" + vision ViT {}L + aligner ({:.2} GB)", v.vis.layers, v.bytes as f64 / 1e9),
            None => String::new(),
        };
        log(format!("rank {rank}/{world}: {} trunk layers + embedding {} / final norm / head (vocab {}..{}){vis_desc} on the device: {:.2} GB in {:.0} s, from {}",
                    layers.iter().filter(|x| x.is_some()).count(), if glob.embed.is_some() { "held" } else { "not held (rank 0 broadcasts its rows)" }, comm.vocab().start, comm.vocab().end,
                    bytes as f64 / 1e9, t0.elapsed().as_secs_f64(), src.how));
        super::mem::phase("weights loaded");
        // the scratch every linear / MoE call needs (the backends refuse a call whose scratch is short): the largest bound
        // over every weight at the rows the forward calls it at
        let scr = g.buf(super::mem::scratch_need(&layers, Some(&glob.head), cfg.layers, max_rows)?)?;
        f.scr = glue::Scr { ptr: scr.ptr(), bytes: scr.bytes };
        let k = TpfKernels::load(&g.dev)?;
        // Engram
        let (mut eio, mut hasher, mut eio_max, mut n_own) = (None, None, 0, 0);
        if let Some(ec) = EngramConfig::from_model_dir(hash_dir)? {
            let layout = Arc::new(EngramLayout::new(ec)?);
            let heads: Vec<usize> = layout.heads_for_rank(world, rank)?.collect();
            ensure!(heads == (rg.eh0..rg.eh0 + rg.neh).collect::<Vec<_>>(), "Engram heads {heads:?} != the TP ranges {}..{}", rg.eh0, rg.eh0 + rg.neh);
            let mut c = EngramIoConfig::new(engram_dir.to_path_buf(), heads.clone());
            c.prefill_max_tok = max_rows;
            c.decode_max_tok = HEAD_MAX;
            c.cache_mib = cache_mib;
            // leg HT1: how long a reader without work spins before it parks (--engram-park-us, default 250 ms as before)
            c.park_after = std::time::Duration::from_micros(crate::opts::lookup("engram-park-us").and_then(|o| crate::opts::ranged(o).ok()).unwrap_or(250_000) as u64);
            let io = match EngramIo::open(&g.dev, layout, c) {
                Ok(io) => io,
                // world 1 on a box that does not hold every head (a slice check): the Engram points are skipped, said here
                Err(e) if world == 1 => { log(format!("rank 0 (world 1): Engram runtime NOT open ({e:#}); the Engram points are skipped")); return Self::finish_new(f, spare, comm, cs, k, cfg, rg, layers, glob, vision, None, None, 0, 0, max_rows, max_seq, bytes, src, scr, n_lanes); }
                Err(e) => return Err(e.context(format!("Engram runtime on heads {heads:?} in {}", engram_dir.display()))),
            };
            n_own = io.n_own();
            eio_max = io.channel_max_tok(io.ch_prefill());
            log(format!("rank {rank}: Engram heads {heads:?} from {} (columns {:?}, row cache {cache_mib} MiB)", engram_dir.display(), io.own_cols()));
            eio = Some(io);
            hasher = Some(EngramHasher::from_model_dir(hash_dir)?.context("Engram hasher")?);
        }
        Self::finish_new(f, spare, comm, cs, k, cfg, rg, layers, glob, vision, eio, hasher, eio_max, n_own, max_rows, max_seq, bytes, src, scr, n_lanes)
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_new(f: Fwd<'a>, spare: Bump, comm: &'a TpComm, cs: &'a CudaStream, k: TpfKernels, cfg: &'a Dsv41Config, rg: Ranges, layers: Vec<Option<TpLayer>>, glob: Global,
                  vision: Option<crate::dsv41::vision::tower::VisionWeights>, eio: Option<EngramIo>, hasher: Option<EngramHasher>, eio_max: usize, n_own: usize,
                  max_rows: usize, max_seq: usize, bytes: u64, src: &RankSrc, scr: Buf, n_lanes: usize) -> Result<TpModel<'a>> {
        let g = f.g;
        let (world, rank) = (comm.world(), comm.rank());
        let h = cfg.hidden;
        let frame_floats = max_rows * (h + MIX_STRIDE + (cfg.hc + 1) * h);
        let frames = [g.buf(frame_floats * 4)?, g.buf(frame_floats * 4)?];
        let bcast = g.buf(max_rows * h * 4)?;
        let eout = g.buf((2 * eio_max.max(1) * n_own.max(1) * 256 * 2).max(256))?;
        let vl_mask_buf = g.buf(max_rows.max(256))?;
        // the lanes (leg FW): each holds its windows, its KV-source caches and its Engram hash state
        let mut lanes = Vec::with_capacity(n_lanes);
        for _ in 0..n_lanes { lanes.push(Self::new_lane(g, cfg, max_rows, max_seq)?); }
        let lane_buf = g.buf(max_rows.max(HEAD_MAX) * 4)?;
        let rk = super::graph::RowKeys { on: crate::opts::lookup("dsv41-graph-keys").and_then(crate::opts::get).as_deref() != Some("split"),
                                         seg: g.buf(2 * n_lanes.max(1) * 4)?, lid: g.buf(max_rows.max(HEAD_MAX) * 4)?, live: false };
        let (pp, cc) = attn::DecodeScratch::bytes(HEAD_MAX, cfg.heads, attn::MAX_SPLITS.max(attn::RECIPE_SPLITS));
        let ds = (g.buf(pp)?, g.buf(cc)?);
        let rows_p = g.buf(HEAD_MAX.max(max_rows) * std::mem::size_of::<TpRow>())?;
        let outs = g.buf(HEAD_MAX.max(max_rows) * std::mem::size_of::<TpOut>())?;
        let (pos_buf, ids_buf) = (g.buf(max_rows * 4)?, g.buf(max_rows * 4)?);
        let tiny = g.buf(64)?;
        let tick = g.buf(MAX_ROW_BLOCKS * 4)?;
        let draft_cap = HEAD_MAX.max(MAX_LANES * 5);
        let (mix_cnt, err_w, zrows) = (g.buf(world.clamp(1, 8) * max_rows.max(draft_cap) * 4)?, g.buf(2 * 4)?, g.buf(draft_cap * h * 2)?);
        let kc_freq = [g.buf(128)?, g.buf(128)?];
        g.h2d(kc_freq[0].ptr(), crate::dsv41::fwd::layer::bytemuck_f32(&f.rope.kc.plain))?;
        g.h2d(kc_freq[1].ptr(), crate::dsv41::fwd::layer::bytemuck_f32(&f.rope.kc.compressed))?;
        let uni_bufs = Some((g.buf(MAX_ROW_BLOCKS * 8 * crate::dsv41::gpu::indexer::CAND_BLOCKS * 4)?, g.buf(MAX_ROW_BLOCKS * 4)?));
        let mut sel_bufs = HashMap::new();
        for &s in &cfg.index_sources { sel_bufs.insert(s, g.buf(max_rows * 512 * 4)?); }
        let embed_bcast = world > 1 && src.opts.embed_rank0_only;
        ensure!(embed_bcast || glob.embed.is_some(), "rank {rank}: the plan says every rank holds the embedding, this one does not");
        g.sync()?;
        let (pos_dev, ids_dev) = (pos_buf.ptr(), ids_buf.ptr());
        let whole = layers.iter().flatten().all(|y| y.mix_a.fn_base != 0 && y.mix_f.fn_base != 0);
        Ok(TpModel { f, spare, comm, cs, k, cfg, rg, layers, glob, vision, vl_mask_buf, lanes, segs: Vec::new(), lane_buf, rk, pos_dev, ids_dev,
                     eio, hasher, frames, bcast, eout, eio_max, ds, max_rows, world, rank, rows_p, outs, pos_buf, ids_buf, sel_bufs, embed_bcast, tiny, tick, uni_bufs,
                     taps: None, main_proj: None, last: None, win_lo: 0, kc_freq, gm: false, gtab: None, graphs: None, logit_tap: None, weights_bytes: bytes, scr_buf: scr,
                     mix_cnt, err_w, zrows, pd1: super::pd1::Pd1::new(world)?, pf_ahead: Vec::new(), tail_merge: 0,
                     eng_in_graph: crate::opts::lookup("dsv41-engram-in-graph").and_then(crate::opts::get).as_deref() != Some("off"), eng_pending: None,
                     pa1: super::pa1::Pa1::new(g, world, max_rows, whole)?, wide_stash: HashMap::new(), hc_whole_local: whole, hc_whole_agreed: None })
    }

    /// One lane's state: per layer the SWA window (ring of 134 slots + linear rows for a prefill call), per KV
    /// source its compressor caches for `max_seq` positions, the Engram hash state.
    pub fn new_lane(g: &Gpu, cfg: &Dsv41Config, max_rows: usize, max_seq: usize) -> Result<Lane> {
        let hd = cfg.head_dim;
        let lin_rows = max_rows + cfg.window;
        let mut win = Vec::with_capacity(cfg.layers);
        for _ in 0..cfg.layers {
            let slots = cfg.swa_slots();
            win.push(Window { ring_q: g.buf(slots * hd)?, ring_s: g.buf(slots * hd / 32)?, lin_q: g.buf(lin_rows * hd)?, lin_s: g.buf(lin_rows * hd / 32)?, lin_base: 0, rows: lin_rows });
        }
        let mut srcs = HashMap::new();
        for &s in &cfg.kv_sources {
            let ratio = cfg.ratio(s);
            let cap = max_seq / ratio.max(1) as usize + 2;
            let rr = cfg.frontier_rows(ratio);
            srcs.insert(s, SrcState { layer: s, ratio, n_groups: 0, kv_main: g.buf(cap * 288)?, iq: g.buf(cap * 64)?, is: g.buf(cap * 4)?,
                                      front: if ratio >= 2 { Some((g.buf(rr * hd * 4)?, g.buf(rr * hd * 4)?)) } else { None }, latent: g.buf((max_rows + 1) * hd * 2)? });
        }
        Ok(Lane { pos: 0, win, src: srcs, hash: NgramHashState::new(), cap_rows: max_rows, max_seq, hist: Vec::new(), temp: 0.0, seed: 0,
                  kind: crate::dsv41::ced::schedule::StateKind::Full, vl_types: Vec::new(), vl_row_ops: Vec::new(), vl_aligner: None })
    }

    /// Capacity of a KV source's caches in groups (every lane is sized alike).
    pub fn src_cap(&self, s: usize) -> usize { self.lanes[0].src.get(&s).map_or(0, |x| x.kv_main.bytes / 288) }

    /// A 4-float all-reduce, synchronised: the ranks are in step after it. The engine's agree is a one-slot mailbox
    /// per peer, so two agree-type rendezvous (agree / boot barrier) must never follow each other without a reduce
    /// between them (a rank still polling for the first token would never see it once the second overwrote it).
    pub fn settle(&mut self) -> Result<()> {
        self.f.g.memset(self.tiny.ptr(), 64, 0)?;
        self.reduce(self.tiny.ptr(), 4)?;
        self.f.g.sync()?;
        self.comm.check("a settle reduce")
    }

    /// The rendezvous after per-rank work of uneven length (loads, dump reads): the boot barrier's long deadline, then a
    /// settle reduce (see `settle`).
    pub fn barrier(&mut self, tag: u32) -> Result<()> {
        crate::net::boot_barrier(tag)?;
        self.settle()
    }

    pub(super) fn swap_bump(&mut self) { std::mem::swap(&mut self.f.b, &mut self.spare); self.f.b.reset(); }

    pub(super) fn reduce(&self, ptr: u64, floats: usize) -> Result<()> { self.comm.all_reduce_raw(self.cs, ptr, floats) }
    /// Neural activation/residual partial sum reduce (allows opt-in `--dsv41-wire-bf16 on`).
    pub(super) fn reduce_act(&self, ptr: u64, floats: usize) -> Result<()> { self.comm.all_reduce_act(self.cs, ptr, floats) }

    // ------------------------------------------------------------------ mHC
    /// The rank's K-slice partial of a sublayer's mix over stream `x` [rows][hc * dim] -> `rider` [rows][32] (25 used).
    pub fn mix_partial(&mut self, mk: MixK, x: u64, rows: usize, rider: u64) -> Result<()> {
        let chunks = km::partial_chunks(mk.klen);
        let rg = km::pick_rg(rows);
        let ngroups = rows.div_ceil(rg);
        let part = self.f.b.take(chunks * rows * 32 * 4)?;
        // leg PF1: the persistent counters (zero between calls: the kernel's last arrival re-arms its group's counter)
        ensure!(ngroups * 4 <= self.mix_cnt.bytes, "mix_partial: {ngroups} row groups > the counter buffer");
        let cnt = self.mix_cnt.ptr();
        let k = self.cfg.hc * self.cfg.hidden;
        let a = km::MixPartialArgs { x, x_ld: k as i64, fn_: mk.fn_ptr, fn_ld: mk.fn_ld as i64, part, chunk_stride: (rows * 32) as i64, row_stride: 32,
                                     rank_part: rider, rp_row_stride: MIX_STRIDE as i64, counters: cnt, k0: mk.k0 as i32, klen: mk.klen as i32, rows: rows as i32, rg: rg as i32 };
        self.f.g.launch1("d41_mix_partial", (chunks as u32, ngroups as u32, 1), (128, 1, 1), 0, &a)?;
        receipts::kernel("tp.mix_partial (K slice)", "d41_mix_partial", rows);
        Ok(())
    }

    /// The mix record [rows][24] (pre, post, comb) from the REDUCED rider.
    pub fn mix_finish(&mut self, l: usize, attn_side: bool, rider: u64, rows: usize) -> Result<u64> {
        let (sc, bc) = if attn_side { (C::HcAttnScale, C::HcAttnBase) } else { (C::HcFfnScale, C::HcFfnBase) };
        let w = &self.layers[l].as_ref().unwrap().w;
        let (s, b) = (w.param(sc)?.ptr.0, w.param(bc)?.ptr.0);
        let out = self.f.b.take(rows * 24 * 4)?;
        let k = self.cfg.hc * self.cfg.hidden;
        let fa = km::MixFinishArgs { part: rider, slot_stride: (rows * MIX_STRIDE) as i64, row_stride: MIX_STRIDE as i64, scale: s, base: b, out, out_ld: 24, nslots: 1,
                                     rows: rows as i32, inv_k: 1.0 / k as f32, eps_norm: self.cfg.rms_eps as f32, eps_hc: self.cfg.hc_eps as f32,
                                     sinkhorn_iters: self.cfg.sinkhorn_iters as i32 };
        self.f.g.launch1("d41_mix_finish", ((rows as u32).div_ceil(8), 1, 1), (128, 1, 1), 0, &fa)?;
        Ok(out)
    }

    /// hc_post of a sublayer straight from its REDUCED f32 frame (rows at stride dim): the kernel rounds each value to bf16
    /// as `add_round(frame, 0)` did into the bf16 sublayer output it read before (same bits, one launch fewer: leg PF1, the
    /// external reference's `_hc_post_only` reads the summed partials itself).
    pub fn hc_post_f32(&mut self, frame: u64, resid: u64, rec: u64, rows: usize) -> Result<u64> {
        let d = self.cfg.hidden;
        ensure!(d % 4 == 0 && frame % 16 == 0, "hc_post_f32: dim {d} / frame alignment");
        let out = self.f.b.take(rows * 4 * d * 2)?;
        let p = km::HcPostArgs { xs: 0, xs_ld: d as i64, resid, r_ld: (4 * d) as i64, post: rec + 16, post_stride: 24, comb: rec + 32, comb_stride: 24,
                                 out, o_ld: (4 * d) as i64, dim: d as i32, rows: rows as i32 };
        let a = km::HcPost32Args { p, xs32: frame, x32_ld: d as i64 };
        self.f.g.launch1("d41_hc_post_f32", ((d as u32).div_ceil(1024), rows as u32, 1), (256, 1, 1), 0, &a)?;
        receipts::kernel("mhc.post (reduced f32 frame)", "d41_hc_post_f32", rows);
        Ok(out)
    }

    // ------------------------------------------------------------------ window / compressor (replicated)
    /// Window K/V of this call's rows into layer l's windows: the projection over every row of the call, then per
    /// segment the lane's ring + linear rows (a segment wider than the backtrack first copies its window rows below
    /// p0 from the ring into the front of the linear rows, so the ring writes of the same call never feed a query).
    pub fn window(&mut self, l: usize, ain: u64, rows: usize, compressed: bool) -> Result<()> {
        if self.pa1.swa_done.take() == Some(l) {
            // PA1: the fused q / kv norm launch already wrote this layer's window rows (graph mode)
            self.pd1.kv_raw = None;
            return Ok(());
        }
        let (d, hd) = (self.cfg.hidden, self.cfg.head_dim);
        let lw = &self.layers[l].as_ref().unwrap().w;
        let raw = match self.pd1.kv_raw.take() {
            Some((ll, p)) if ll == l => p,   // PD1: computed by the grouped front launch
            _ => { let raw = self.f.b.take(rows * hd * 2)?; self.f.lin_rows("attn.wkv", lw, C::AttnWkv, None, ain, d, d, rows, raw, hd, false)?; raw }
        };
        self.window_write_raw(l, raw, rows, compressed)
    }

    /// Write pre-projected window `raw` (`[rows][head_dim]` bf16) into layer `l`'s lane ring + linear rows.
    pub(super) fn window_write_raw(&mut self, l: usize, raw: u64, rows: usize, compressed: bool) -> Result<()> {
        let hd = self.cfg.head_dim;
        let lw = &self.layers[l].as_ref().unwrap().w;
        let norm = lw.param(C::AttnKvNorm)?;
        let (nptr, nfmt) = (norm.ptr.0, fmt_of(norm)?);
        let freq = self.kc_freq[compressed as usize].ptr();
        if self.gm {
            // graph mode (decode / verify): every row to its lane's ring, positions from the device (rows <= 6 per lane)
            ensure!(self.segs.iter().all(|sg| sg.n <= graph_seg_max(self.cfg)), "graph mode: a lane segment wider than {} rows would overwrite window keys its own rows read",
                    graph_seg_max(self.cfg));
            let a = self.swa_args(l, raw, rows, compressed)?;
            self.k.swa_rows(self.cs, &a)?;
            receipts::kernel("kv.swa_write (graph: device positions)", "d41fw_swa_rows", rows);
            return Ok(());
        }
        let segs = self.segs.clone();
        for sg in &segs {
            let kpre = if sg.n > self.cfg.backtrack() { sg.p0.min(self.cfg.window) } else { 0 };
            let w = &mut self.lanes[sg.lane].win[l];
            ensure!(kpre + sg.n <= w.rows, "layer {l}: window call of {} rows (+{kpre}) > {} linear rows", sg.n, w.rows);
            if kpre > 0 {
                self.k.ring_lin(self.cs, w.ring_q.ptr(), w.ring_s.ptr(), w.lin_q.ptr(), w.lin_s.ptr(), (sg.p0 - kpre) as i64, kpre, self.cfg.swa_slots(), hd, hd / 32)?;
            }
            let a = kv::SwaArgs { kv_raw: raw + (sg.row0 * hd * 2) as u64, norm_w: nptr, inv_freq: freq, ring_q: w.ring_q.ptr(), ring_s: w.ring_s.ptr(),
                                  chunk_q: w.lin_q.ptr() + (kpre * hd) as u64, chunk_s: w.lin_s.ptr() + (kpre * hd / 32) as u64,
                                  p0: sg.p0 as i64, rows: sg.n as i32, norm_fmt: nfmt, eps: self.cfg.rms_eps as f32 };
            self.f.g.launch1("d41_kc_swa_write", (sg.n as u32, 1, 1), (128, 1, 1), 0, &a)?;
            w.lin_base = (sg.p0 - kpre) as i64;
        }
        receipts::kernel("kv.swa_write", "d41_kc_swa_write", rows);
        Ok(())
    }

    /// The graph-mode window write's arguments (`d41fw_swa_rows`; leg PA1's fused launch takes the same).
    pub(super) fn swa_args(&self, l: usize, raw: u64, rows: usize, compressed: bool) -> Result<super::tpk::SwaRowsArgs> {
        let lw = &self.layers[l].as_ref().unwrap().w;
        let norm = lw.param(C::AttnKvNorm)?;
        let t = self.gtab.as_ref().context("graph tables")?;
        let (tq, ts) = t.ring_tab(l, rows);
        Ok(super::tpk::SwaRowsArgs { kv_raw: raw, norm_w: norm.ptr.0, inv_freq: self.kc_freq[compressed as usize].ptr(), pos: self.pos_dev, ring_q: tq, ring_s: ts,
                                     rows: rows as i32, norm_fmt: fmt_of(norm)?, eps: self.cfg.rms_eps as f32, pad: 0 })
    }

    /// The KV source l's compressor over this call's rows, incremental per lane (positions [p0, p0 + n) of each
    /// segment): the groups the segment completes, the frontier ring stash, the index-K planes of the new groups.
    /// The projections run once over every row of the call (row-local); the pooling per lane.
    pub fn compressor(&mut self, l: usize, ain: u64, rows: usize) -> Result<()> {
        let (d, hd) = (self.cfg.hidden, self.cfg.head_dim);
        let ratio = self.cfg.ratio(l);
        let lw = &self.layers[l].as_ref().unwrap().w;
        let pre = self.pd1.comp.take().filter(|c| c.0 == l);
        let (kvp, scp, es) = if let Some((_, kvp, scp)) = pre {
            (kvp, scp, if ratio >= 2 { 4usize } else { 2 })   // PD1: computed by the grouped front launch
        } else if ratio >= 2 {
            let (kvp, scp) = (self.f.b.take(rows * hd * 4)?, self.f.b.take(rows * hd * 4)?);
            self.f.lin_rows("comp.wkv", lw, C::CompWkv, None, ain, d, d, rows, kvp, hd, true)?;
            self.f.lin_rows("comp.wgate", lw, C::CompWgate, None, ain, d, d, rows, scp, hd, true)?;
            (kvp, scp, 4usize)
        } else {
            let kvp = self.f.b.take(rows * hd * 2)?;
            self.f.lin_rows("comp.wkv", lw, C::CompWkv, None, ain, d, d, rows, kvp, hd, false)?;
            (kvp, 0, 2usize)
        };
        self.compressor_write_proj(l, kvp, scp, es, rows)
    }

    /// Pool pre-projected compressor rows `(kvp, scp)` into lane `SrcState` caches and index-K planes.
    pub(super) fn compressor_write_proj(&mut self, l: usize, kvp: u64, scp: u64, es: usize, rows: usize) -> Result<()> {
        let hd = self.cfg.head_dim;
        let ratio = self.cfg.ratio(l);
        let rr = self.cfg.frontier_rows(ratio);
        let lw = &self.layers[l].as_ref().unwrap().w;
        let norm = lw.param(C::CompNorm)?;
        let (nptr, nfmt) = (norm.ptr.0, fmt_of(norm)?);
        let kn = lw.param(C::IdxKNorm)?;
        let (knp, knf) = (kn.ptr.0, fmt_of(kn)?);
        let freq = self.kc_freq[1].ptr();
        let segs = self.segs.clone();
        if self.gm && self.rk.live {
            // leg PG1: one launch set per lane SLOT (the highest lane of the call + 1: the graph's key), each reading its
            // lane's (row0, rows) from the segment table: the captured structure does not depend on the split. Grids for
            // the widest segment a graphed call holds; a lane without rows returns at once. The same per-group arithmetic.
            let smax = graph_seg_max(self.cfg);
            let gmax = smax.div_ceil(ratio as usize);
            let slots = super::graph::slot_class(segs.iter().map(|s| s.lane + 1).max().unwrap_or(0), self.lanes.len());
            for slot in 0..slots {
                let st = self.lanes[slot].src.get(&l).context("KV source state")?;
                let (kv_main, latent, iq, is_) = (st.kv_main.ptr(), st.latent.ptr(), st.iq.ptr(), st.is.ptr());
                let front = st.front.as_ref().map(|(a, b)| (a.ptr(), b.ptr()));
                let mut a = kv::CompressArgs { norm_w: nptr, inv_freq: freq, latent_out: latent, kv_main, p0: 0, rows: 0, ratio: ratio as i32,
                                               ring_rows: rr as i32, g_first: 0, n_groups: 0, norm_fmt: nfmt, eps: self.cfg.rms_eps as f32, ..Default::default() };
                if ratio >= 2 { let (fk, fs) = front.context("ratio >= 2 without a frontier ring")?; a.kv = kvp; a.score = scp; a.ring_kv = fk; a.ring_score = fs; } else { a.kv16 = kvp; }
                self.k.compress_tab(self.cs, &a, self.pos_dev, self.rk.seg.ptr(), slot, gmax)?;
                if let Some((fk, fs)) = front {
                    let s = kv::StashArgs { kv: kvp, score: scp, ring_kv: fk, ring_score: fs, p0: 0, rows: 0, ring_rows: rr as i32 };
                    self.k.stash_tab(self.cs, &s, self.pos_dev, self.rk.seg.ptr(), slot, smax.min(rr))?;
                }
                let id = self.cfg.index_head_dim;
                let kraw = self.f.b.take(gmax * id * 2)?;
                let lw = &self.layers[l].as_ref().unwrap().w;
                self.f.lin_rows("idx.wk", lw, C::IdxWk, None, latent, hd, hd, gmax, kraw, id, false)?;
                let ia = kv::IndexArgs { k_raw: kraw, norm_w: knp, inv_freq: freq, q_plane: iq, s_plane: is_, ratio: ratio as i32, g_first: 0, n_groups: 0, norm_fmt: knf,
                                         eps: self.cfg.rms_eps as f32 };
                self.k.index_k_tab(self.cs, &ia, self.pos_dev, self.rk.seg.ptr(), slot, gmax)?;
            }
            for sg in &segs {
                let (gf, ng) = kv::group_range(sg.p0 as u64, sg.n as u64, ratio);
                self.lanes[sg.lane].src.get_mut(&l).unwrap().n_groups = (gf + ng) as usize;
            }
            receipts::kernel("kv.compress (graph keyed by rows: per lane slot, segments from the device)", "d41fw_compress_tab+d41fw_stash_tab+d41fw_index_k_tab", rows);
            return Ok(());
        }
        if self.gm {
            for sg in &segs {
                let (gf, ng) = kv::group_range(sg.p0 as u64, sg.n as u64, ratio);
                let gmax = sg.n.div_ceil(ratio as usize);
                let st = self.lanes[sg.lane].src.get(&l).context("KV source state")?;
                let (kv_main, latent, iq, is_) = (st.kv_main.ptr(), st.latent.ptr(), st.iq.ptr(), st.is.ptr());
                let front = st.front.as_ref().map(|(a, b)| (a.ptr(), b.ptr()));
                let (kv_s, sc_s) = (kvp + (sg.row0 * hd * es) as u64, if scp != 0 { scp + (sg.row0 * hd * 4) as u64 } else { 0 });
                let mut a = kv::CompressArgs { norm_w: nptr, inv_freq: freq, latent_out: latent, kv_main, p0: 0, rows: sg.n as i32, ratio: ratio as i32,
                                               ring_rows: rr as i32, g_first: 0, n_groups: 0, norm_fmt: nfmt, eps: self.cfg.rms_eps as f32, ..Default::default() };
                if ratio >= 2 { let (fk, fs) = front.context("ratio >= 2 without a frontier ring")?; a.kv = kv_s; a.score = sc_s; a.ring_kv = fk; a.ring_score = fs; } else { a.kv16 = kv_s; }
                let p0d = self.pos_dev + (sg.row0 * 4) as u64;
                self.k.compress_seg(self.cs, &a, p0d, gmax)?;
                if let Some((fk, fs)) = front {
                    let s = kv::StashArgs { kv: kv_s, score: sc_s, ring_kv: fk, ring_score: fs, p0: 0, rows: sg.n as i32, ring_rows: rr as i32 };
                    self.k.stash_seg(self.cs, &s, p0d, sg.n.min(rr))?;
                }
                let id = self.cfg.index_head_dim;
                let kraw = self.f.b.take(gmax * id * 2)?;
                let lw = &self.layers[l].as_ref().unwrap().w;
                self.f.lin_rows("idx.wk", lw, C::IdxWk, None, latent, hd, hd, gmax, kraw, id, false)?;
                let ia = kv::IndexArgs { k_raw: kraw, norm_w: knp, inv_freq: freq, q_plane: iq, s_plane: is_, ratio: ratio as i32, g_first: 0, n_groups: 0, norm_fmt: knf,
                                         eps: self.cfg.rms_eps as f32 };
                self.k.index_k_seg(self.cs, &ia, p0d, sg.n, gmax)?;
                self.lanes[sg.lane].src.get_mut(&l).unwrap().n_groups = (gf + ng) as usize;
            }
            receipts::kernel("kv.compress (graph: device positions, per lane)", "d41fw_compress_seg+d41fw_stash_seg+d41fw_index_k_seg", rows);
            return Ok(());
        }
        for sg in &segs {
            let (gf, ng) = kv::group_range(sg.p0 as u64, sg.n as u64, ratio);
            let st = self.lanes[sg.lane].src.get(&l).context("KV source state")?;
            let cap = st.kv_main.bytes / 288;
            ensure!((gf + ng) as usize <= cap, "layer {l}: KV source groups {} > capacity {cap} (--max-seq-len)", gf + ng);
            let (kv_main, latent, iq, is_) = (st.kv_main.ptr(), st.latent.ptr(), st.iq.ptr(), st.is.ptr());
            let front = st.front.as_ref().map(|(a, b)| (a.ptr(), b.ptr()));
            let (kv_s, sc_s) = (kvp + (sg.row0 * hd * es) as u64, if scp != 0 { scp + (sg.row0 * hd * 4) as u64 } else { 0 });
            let mut a = kv::CompressArgs { norm_w: nptr, inv_freq: freq, latent_out: latent, kv_main, p0: sg.p0 as i64, rows: sg.n as i32, ratio: ratio as i32,
                                           ring_rows: rr as i32, g_first: gf as i32, n_groups: ng as i32, norm_fmt: nfmt, eps: self.cfg.rms_eps as f32, ..Default::default() };
            if ratio >= 2 {
                let (fk, fs) = front.context("ratio >= 2 without a frontier ring")?;
                a.kv = kv_s; a.score = sc_s; a.ring_kv = fk; a.ring_score = fs;
            } else {
                a.kv16 = kv_s;
            }
            if ng > 0 { self.f.g.launch1("d41_kc_compress", (ng as u32, 1, 1), (128, 1, 1), 0, &a)?; }
            if let Some((fk, fs)) = front {
                let s = kv::StashArgs { kv: kv_s, score: sc_s, ring_kv: fk, ring_score: fs, p0: sg.p0 as i64, rows: sg.n as i32, ring_rows: rr as i32 };
                self.f.g.launch1("d41_kc_stash", (sg.n.min(rr) as u32, 1, 1), (128, 1, 1), 0, &s)?;
                if sg.n > self.cfg.backtrack() && sg.n <= crate::dsv41::gpu::dspark::WIDE_RPL {
                    let need = crate::dsv41::gpu::dspark::WIDE_RPL * hd * 4;
                    if !self.wide_stash.contains_key(&(sg.lane, l)) {
                        self.wide_stash.insert((sg.lane, l), (self.f.g.buf(need)?, self.f.g.buf(need)?));
                    }
                    let (sk, ss) = &self.wide_stash[&(sg.lane, l)];
                    self.f.g.d2d(sk.ptr(), kv_s, sg.n * hd * 4)?;
                    self.f.g.d2d(ss.ptr(), sc_s, sg.n * hd * 4)?;
                }
            }
            if ng > 0 {
                let id = self.cfg.index_head_dim;
                let kraw = self.f.b.take(ng as usize * id * 2)?;
                let lw = &self.layers[l].as_ref().unwrap().w;
                self.f.lin_rows("idx.wk", lw, C::IdxWk, None, latent, hd, hd, ng as usize, kraw, id, false)?;
                let ia = kv::IndexArgs { k_raw: kraw, norm_w: knp, inv_freq: freq, q_plane: iq, s_plane: is_, ratio: ratio as i32, g_first: gf as i32,
                                         n_groups: ng as i32, norm_fmt: knf, eps: self.cfg.rms_eps as f32 };
                self.f.g.launch1("d41_kc_index_k", ((ng as u32).div_ceil(4), 1, 1), (32, 4, 1), 0, &ia)?;
            }
            self.lanes[sg.lane].src.get_mut(&l).unwrap().n_groups = (gf + ng) as usize;
        }
        receipts::kernel("kv.compress (incremental, per lane)", "d41_kc_compress+d41_kc_stash+d41_kc_index_k", rows);
        Ok(())
    }

    /// The indexer's selection for the call's rows of an index source: row blocks of at most 8 rows of ONE lane each
    /// (the lane's index-K planes per row block), positions per row from `pos`. `cand_out` = the candidate source
    /// (writes the union for the layers that use it); `cand_in` = a candidate user (its row blocks must equal the
    /// source's, i.e. the same call). Same arithmetic as SL's `Fwd::indexer` per row block (KI: a row's selection
    /// does not depend on the other row blocks).
    #[allow(clippy::too_many_arguments)]
    pub fn indexer_segs(&mut self, l: usize, s: usize, qr: u64, x: u64, pos: u64, rows: usize, cand_out: bool, cand_in: Option<&Union>) -> Result<(u64, Option<Union>)> {
        self.indexer_segs_into(l, s, qr, x, pos, rows, cand_out, cand_in, None)
    }

    /// `indexer_segs` writing the selection to `sel_out` (i32 [rows][512]) when given (leg PF1: the forward's kept selection
    /// buffer, so no device copy follows the indexer inside a round).
    #[allow(clippy::too_many_arguments)]
    pub fn indexer_segs_into(&mut self, l: usize, s: usize, qr: u64, x: u64, pos: u64, rows: usize, cand_out: bool, cand_in: Option<&Union>,
                             sel_out: Option<u64>) -> Result<(u64, Option<Union>)> {
        use crate::dsv41::gpu::indexer as ki;
        let (d, ql) = (self.cfg.hidden, self.cfg.q_lora_rank);
        let w = &self.layers[l].as_ref().unwrap().w;
        let iq = self.f.b.take(rows * ki::HEADS * ki::HEAD_DIM * 2)?;
        if let Some((_, xiq)) = self.pd1.xiq.take().filter(|x| x.0 == l) {
            // PD1: idx wq_b's rotated input written by the q norm
            let wi = w.lin(C::IdxWqB)?.0.clone();
            self.pd1_pre_call("idx.wq_b (PD1: rotated by the q norm)", &wi, &bf16(qr, rows, ql, ql), &bf16(iq, rows, ki::HEADS * ki::HEAD_DIM, ki::HEADS * ki::HEAD_DIM), xiq)?;
        } else {
            self.f.lin_rows("idx.wq_b", w, C::IdxWqB, None, qr, ql, ql, rows, iq, ki::HEADS * ki::HEAD_DIM, false)?;
        }
        let wp = self.f.b.take(rows * ki::HEADS * 2)?;
        let w = &self.layers[l].as_ref().unwrap().w;
        self.f.lin_rows("idx.weights_proj", w, C::IdxWeightsProj, None, x, d, d, rows, wp, ki::HEADS, false)?;
        let mut cap = 1usize;
        for sg in &self.segs { cap = cap.max(self.lanes[sg.lane].src.get(&s).context("index source state")?.n_groups); }
        // graph mode: the plan's shape follows the context class (part of the graph's key), the caps are the capacity
        let plan_keys = cap;
        if self.gm { cap = self.src_cap(s); }
        // decode / verify calls (every lane <= 8 rows, <= 24 rows): PK1's decode plan (row blocks per lane, key chunks,
        // the merge fused for 1-row blocks); prefill calls: 8-row blocks, one key chunk. Same selection bits either way.
        let decode = rows <= HEAD_MAX && self.segs.iter().all(|sg| sg.n <= 8);
        let (r, g, min_tiles, rbs, fused) = if decode {
            let p = ki::decode_plan(&self.segs.iter().map(|sg| (sg.row0, sg.n)).collect::<Vec<_>>(), plan_keys);
            (p.r, p.g, p.min_tiles, p.rbs, p.fused_merge)
        } else {
            let mut v = Vec::new();
            for sg in &self.segs { for i in (0..sg.n).step_by(8) { v.push(((sg.row0 + i) as i32, (sg.n - i).min(8) as i32)); } }
            (8usize, 1usize, 8usize, v, false)
        };
        let mut planes: Vec<(u64, u64)> = Vec::with_capacity(rbs.len());
        for &(r0, _) in &rbs {
            let sg = self.segs.iter().find(|sg| (r0 as usize) >= sg.row0 && (r0 as usize) < sg.row0 + sg.n).context("row block outside the call")?;
            let st = self.lanes[sg.lane].src.get(&s).context("index source state")?;
            planes.push((st.iq.ptr(), st.is.ptr()));
        }
        if let Some(u) = cand_in { ensure!(u.rbs == rbs, "layer {l}: mode 1 needs layer 20's row blocks (same call)"); }
        let n_rb = rbs.len();
        ensure!(n_rb <= MAX_ROW_BLOCKS, "indexer: {n_rb} row blocks (the merge tickets and the candidate union hold {MAX_ROW_BLOCKS}: a call of more than {} rows must be split)", 8 * MAX_ROW_BLOCKS);
        let ucap = r * ki::CAND_BLOCKS;
        let sz = ki::sizes(n_rb, r, g, rows, cap, ucap);
        let rbv: Vec<u64> = rbs.iter().zip(&planes).flat_map(|(&(r0, n), &(pq, ps))| [pq, ps, (r0 as u32 as u64) | ((n as u32 as u64) << 32), 0]).collect();
        let rbd = if self.gm {
            // graph mode: the row-block table was written before the capture (the same content: segments and planes)
            let t = self.gtab.as_ref().context("graph tables")?;
            let (ptr, want) = t.rbs_of(l).context("no row-block table for this index layer")?;
            ensure!(want == rbv, "layer {l}: the graph's row-block table differs from the call's");
            ptr
        } else {
            let rbd = self.f.b.take(rbv.len() * 8)?;
            self.f.g.h2d_async(rbd, &rbv)?;
            rbd
        };
        let sel = match sel_out { Some(p) => p, None => self.f.b.take(sz.sel_i32 * 4)? };
        let cnt = self.f.b.take(rows * 4)?;
        let part = self.f.b.take(sz.part_u64.max(1) * 8)?;
        let pcnt = self.f.b.take(sz.pcnt_i32.max(1) * 4)?;
        let bmax = if cand_out { self.f.b.take(sz.bmax_u16.max(1) * 2)? } else { 0 };
        let cand = self.f.b.take(sz.cand_i32 * 4)?;
        let ccnt = self.f.b.take(rows * 4)?;
        let bm = self.f.b.take(sz.bm_u32.max(1) * 4)?;
        let err = self.err_w.ptr() + 4;     // leg PF1: persistent, zeroed once (sticky)
        // the candidate union lives in persistent buffers (a cudaMalloc / cudaFree per call would sync the device mid-forward)
        let un = if cand_out {
            let (ub, cb) = self.uni_bufs.take().context("candidate union buffers in use")?;
            ensure!(sz.uni_u32 * 4 <= ub.bytes && n_rb * 4 <= cb.bytes, "candidate union: {} B / {n_rb} blocks exceed the persistent buffers", sz.uni_u32 * 4);
            Some(Union { uni: ub, ucnt: cb, rbs: rbs.clone() })
        } else { None };
        let (uni, ucnt) = match (&un, cand_in) { (Some(u), _) | (None, Some(u)) => (u.uni.ptr(), u.ucnt.ptr()), _ => (self.f.b.take(4)?, self.f.b.take(n_rb.max(1) * 4)?) };
        let mut a = ki::IdxArgs::default();
        a.iq = iq; a.w = wp; a.pos = pos; a.rbs = rbd; a.sel = sel; a.sel_cnt = cnt; a.part = part; a.pcnt = pcnt; a.bmax = bmax; a.cand = cand; a.cand_cnt = ccnt; a.bm = bm;
        a.uni = uni; a.ucnt = ucnt; a.err = err; a.tick = if fused && g > 1 { self.tick.ptr() } else { 0 };
        a.n_rb = n_rb as i32; a.rows_total = rows as i32; a.iq_ld = (ki::HEADS * ki::HEAD_DIM) as i32; a.w_ld = ki::HEADS as i32; a.in_f32 = 0;
        a.ratio = self.cfg.ratio(s) as i32; a.mode = cand_in.is_some() as i32; a.write_bmax = cand_out as i32;
        a.n_keys_cap = cap as i32; a.nb_cap = cap.div_ceil(ki::CAND_BLOCK) as i32; a.nbw_cap = (a.nb_cap as usize).div_ceil(32) as i32;
        a.g = g as i32; a.min_tiles = min_tiles as i32; a.dbg_cap = 0; a.ucap = ucap as i32; a.r = r as i32; a.inv_freq = self.f.rope.ki;
        ki::select(&self.f.g.l(), &a, cand_out)?;
        let (entry, _) = ki::score_entry(r);
        let path = match (decode, cand_out, cand_in.is_some()) {
            (true, true, _) => "tp.idx.select+cand (PK1 decode plan)", (true, false, true) => "tp.idx.select mode 1 (PK1 decode plan)", (true, ..) => "tp.idx.select (PK1 decode plan)",
            (false, true, _) => "tp.idx.select+cand (prefill shape, 8-row blocks)", (false, false, true) => "tp.idx.select mode 1 (prefill shape)", (false, ..) => "tp.idx.select (prefill shape, 8-row blocks)",
        };
        receipts::kernel(path, entry, rows);
        if decode { super::metrics::idx_plan(r, g); }
        if self.f.checks {
            let e = self.f.g.down::<i32>(err, 1)?[0];
            ensure!(e == 0, "indexer layer {l}: kernel error word {e}");
        }
        Ok((sel, un))
    }

    // ------------------------------------------------------------------ attention (the rank's heads)
    /// q path (rank's heads) + sparse attention over the window and the selection + wo_a / wo_b -> FP32 partial `y`
    /// [rows][dim] (overwritten). Returns qr (for the indexer, replicated) is the caller's: `qr`, `ain` are given.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_core(&mut self, l: usize, qr: u64, pos: u64, rows: usize, src: Option<usize>, sel: Option<(u64, usize, usize)>, y: u64) -> Result<()> {
        let (ql, hd) = (self.cfg.q_lora_rank, self.cfg.head_dim);
        let nh = self.rg.nh;
        let compressed = self.cfg.ratio(l) > 0;
        // wq_b over the rank's heads
        let q = self.f.b.take(rows * nh * hd * 2)?;
        let wq = self.layers[l].as_ref().unwrap().wq_b.clone();
        if let Some((_, xq)) = self.pd1.xq.take().filter(|x| x.0 == l) {
            // PD1: wq_b's rotated input written by the q norm
            self.pd1_pre_call("tp.attn.wq_b (heads; PD1: rotated by the q norm)", &wq.0, &bf16(qr, rows, ql, ql), &bf16(q, rows, nh * hd, nh * hd), xq)?;
        } else {
            self.lin_w("tp.attn.wq_b (heads)", &wq, &bf16(qr, rows, ql, ql), &{ let mut a = bf16(q, rows, nh * hd, nh * hd); a.cols = nh * hd; a }, Accum::Overwrite)?;
        }
        self.pd1_pf(0, l, rows)?;   // PD1 site A: this layer's wo_a groups + wo_b
        // PA1: q's RoPE rides the decode attention's q load (same per-element sequence) when the call takes the decode entry
        let qrope = self.pa1_qrope(compressed, rows);
        if !qrope { glue::rope_rows(self.f.g, q, nh * hd, pos, rows, nh, hd, false, &self.f.rope.q.of(compressed))?; }
        // attention
        let out = self.f.b.take(rows * nh * hd * 2)?;
        let err = self.err_w.ptr();         // leg PF1: persistent, zeroed once (sticky)
        // one KA lane per segment of the call (its ring, linear rows, KV-source cache); the per-row lane index buffer
        // maps rows to them (a single segment: lane 0, no buffer)
        // leg PG1 (rows-keyed graph): one KA lane per lane SLOT 0..=the highest lane of the call, the row -> lane map from the
        // device (`rk.lid`): the attention's arguments do not depend on the split
        let slot_lanes: Vec<usize> = if self.gm && self.rk.live { (0..super::graph::slot_class(self.segs.iter().map(|s| s.lane + 1).max().unwrap_or(0), self.lanes.len())).collect() }
                                     else { self.segs.iter().map(|s| s.lane).collect() };
        let mut lsrc = Vec::with_capacity(slot_lanes.len());
        for &lane in &slot_lanes {
            let ln = &self.lanes[lane];
            let w = &ln.win[l];
            // graph mode: kv_rows = the cache capacity (a bound only: the indexer never selects past the visible groups) and
            // no linear rows (a decode / verify call's own rows are in the ring: 134 = 128 + 6 slots)
            let kvs = src.map(|s| { let st = &ln.src[&s]; (st.kv_main.p(), if self.gm { st.kv_main.bytes / 288 } else { st.n_groups }) });
            let lin = if self.gm { None } else { Some((w.lin_q.p(), w.lin_s.p(), w.lin_base)) };
            lsrc.push(attn::LaneSrc { swa_q: w.ring_q.p(), swa_s: w.ring_s.p(), slots: self.cfg.swa_slots(), lin, kv: kvs, ext: None });
        }
        let lane_ptr = if self.gm && self.rk.live { DevPtr(self.rk.lid.ptr()) } else if self.segs.len() > 1 { DevPtr(self.lane_buf.ptr()) } else { DevPtr(0) };
        let (sp, sstride, scap) = sel.unwrap_or((0, 0, 0));
        let call = attn::AttnCall { rows, heads: nh, q: DevPtr(q), q_stride: nh * hd, sink: DevPtr(self.layers[l].as_ref().unwrap().sink), pos: DevPtr(pos), win_end: DevPtr(0), lane: lane_ptr,
                                    sel: DevPtr(sp), sel_stride: sstride, sel_cap: scap, n_sel: DevPtr(0), lanes: lsrc, out: DevPtr(out), out_stride: nh * hd,
                                    out_q: None, out_s: None, epi: attn::EPI_ROPE | attn::EPI_AQ, rope_freq: self.f.rope.ka.of(compressed), err: DevPtr(err) };
        if self.win_lo > 0 {
            attn::launch_prefill_floor(self.f.g.l(), &call, self.win_lo)?;
            receipts::kernel("tp.attn.sparse (heads, window floor: opt-in replay)", "d41_attn_prefill(floor)", rows);
        } else if rows <= HEAD_MAX {
            let s = attn::DecodeScratch { part: self.ds.0.p(), cnt: self.ds.1.p(), rows, groups: nh.div_ceil(attn::HG), nsplit: attn::MAX_SPLITS.max(attn::RECIPE_SPLITS) };
            let x = self.pa1.attn_x(qrope);
            if !self.pd1_attn(l, &call, &s, rows, x)? {
                attn::launch_decode_x(self.f.g.l(), &call, &s, None, x)?;
                super::pa1::Pa1::count_attn(x);
                receipts::kernel("tp.attn.sparse (heads)", if x & attn::X_DEFER != 0 { "d41_attn_decode+d41_attn_merge" } else { "d41_attn_decode" }, rows);
            }
        } else if self.pa1.recipe && attn::recipe_covers(&call) {
            // PA1 opt-in: the recipe rule for prefill rows too (8 splits a row, pieces of 16 rows): decode == prefill holds
            let s = attn::DecodeScratch { part: self.ds.0.p(), cnt: self.ds.1.p(), rows: HEAD_MAX, groups: nh.div_ceil(attn::HG), nsplit: attn::MAX_SPLITS.max(attn::RECIPE_SPLITS) };
            attn::launch_recipe(self.f.g.l(), &call, Some(&s), None, 0)?;
            receipts::kernel("tp.attn.sparse (heads; PA1 recipe rule, prefill rows)", "d41_attn_recipe+d41_attn_merge_recipe", rows);
        } else {
            attn::launch_prefill(self.f.g.l(), &call)?;
            receipts::kernel("tp.attn.sparse (heads)", "d41_attn_prefill", rows);
        }
        if self.f.checks {
            let e = self.f.g.down::<i32>(err, 1)?[0];
            ensure!(e == 0, "attention layer {l}: kernel error word {e}");
        }
        self.wo_partial(l, out, rows, y)
    }

    /// U6a (LAGA): q path + sparse attention + `wo_a` over the rank's o_groups -> compact BF16 `mid` `[rows][ng * olr]`
    /// (for 1-RTT All-to-All followed by local whole `wo_b` on `1/W` rows).
    #[allow(clippy::too_many_arguments)]
    pub fn attn_core_mid(&mut self, l: usize, qr: u64, pos: u64, rows: usize, src: Option<usize>, sel: Option<(u64, usize, usize)>) -> Result<u64> {
        let (ql, hd) = (self.cfg.q_lora_rank, self.cfg.head_dim);
        let nh = self.rg.nh;
        let compressed = self.cfg.ratio(l) > 0;
        let q = self.f.b.take(rows * nh * hd * 2)?;
        let wq = self.layers[l].as_ref().unwrap().wq_b.clone();
        self.lin_w("tp.attn.wq_b (heads)", &wq, &bf16(qr, rows, ql, ql), &{ let mut a = bf16(q, rows, nh * hd, nh * hd); a.cols = nh * hd; a }, Accum::Overwrite)?;
        glue::rope_rows(self.f.g, q, nh * hd, pos, rows, nh, hd, false, &self.f.rope.q.of(compressed))?;
        let out = self.f.b.take(rows * nh * hd * 2)?;
        let err = self.err_w.ptr();
        let slot_lanes: Vec<usize> = self.segs.iter().map(|s| s.lane).collect();
        let mut lsrc = Vec::with_capacity(slot_lanes.len());
        for &lane in &slot_lanes {
            let ln = &self.lanes[lane];
            let w = &ln.win[l];
            let kvs = src.map(|s| { let st = &ln.src[&s]; (st.kv_main.p(), st.n_groups) });
            let lin = Some((w.lin_q.p(), w.lin_s.p(), w.lin_base));
            lsrc.push(attn::LaneSrc { swa_q: w.ring_q.p(), swa_s: w.ring_s.p(), slots: self.cfg.swa_slots(), lin, kv: kvs, ext: None });
        }
        let lane_ptr = if self.segs.len() > 1 { DevPtr(self.lane_buf.ptr()) } else { DevPtr(0) };
        let (sp, sstride, scap) = sel.unwrap_or((0, 0, 0));
        let call = attn::AttnCall { rows, heads: nh, q: DevPtr(q), q_stride: nh * hd, sink: DevPtr(self.layers[l].as_ref().unwrap().sink), pos: DevPtr(pos), win_end: DevPtr(0), lane: lane_ptr,
                                    sel: DevPtr(sp), sel_stride: sstride, sel_cap: scap, n_sel: DevPtr(0), lanes: lsrc, out: DevPtr(out), out_stride: nh * hd,
                                    out_q: None, out_s: None, epi: attn::EPI_ROPE | attn::EPI_AQ, rope_freq: self.f.rope.ka.of(compressed), err: DevPtr(err) };
        if self.win_lo > 0 {
            attn::launch_prefill_floor(self.f.g.l(), &call, self.win_lo)?;
        } else if self.pa1.recipe && attn::recipe_covers(&call) {
            let s = attn::DecodeScratch { part: self.ds.0.p(), cnt: self.ds.1.p(), rows: HEAD_MAX, groups: nh.div_ceil(attn::HG), nsplit: attn::MAX_SPLITS.max(attn::RECIPE_SPLITS) };
            attn::launch_recipe(self.f.g.l(), &call, Some(&s), None, 0)?;
        } else {
            attn::launch_prefill(self.f.g.l(), &call)?;
        }
        if self.f.checks {
            let e = self.f.g.down::<i32>(err, 1)?[0];
            ensure!(e == 0, "attention layer {l}: kernel error word {e}");
        }
        let (ng, olr) = (self.rg.ng, self.cfg.o_lora_rank);
        let gin = nh * hd / ng;
        let mid = self.f.b.take(rows * ng * olr * 2)?;
        for i in 0..ng {
            let gi = self.rg.g0 + i;
            let lw = &self.layers[l].as_ref().unwrap().w;
            self.f.lin_rows("tp.attn.wo_a (groups, LAGA)", lw, C::AttnWoA, Some(gi as u16), out + (i * gin * 2) as u64, nh * hd, gin, rows, mid + (i * olr * 2) as u64, ng * olr, false)?;
        }
        Ok(mid)
    }

    /// U6a (LAGA): run whole `wo_b` (`Shard::Whole`, already resident) on `[rows, o_groups * olr]` BF16 -> BF16 `[rows, d]`.
    pub fn wo_b_whole(&mut self, l: usize, mid_full: u64, rows: usize, ao_bf16: u64) -> Result<()> {
        let d = self.cfg.hidden;
        let k_full = self.cfg.o_groups * self.cfg.o_lora_rank;
        let wo = self.layers[l].as_ref().unwrap().w.lin(C::AttnWoB)?.clone();
        let mut ya = bf16(ao_bf16, rows, d, d);
        ya.cols = d;
        self.lin_w("tp.attn.wo_b (whole, LAGA)", &wo, &bf16(mid_full, rows, k_full, k_full), &ya, Accum::Overwrite)
    }

    /// wo_a over the rank's o_groups (block-diagonal) of the attention output `out` [rows][nh * 512], wo_b over their
    /// input range -> FP32 partial `y` [rows][dim] (overwritten).
    pub fn wo_partial(&mut self, l: usize, out: u64, rows: usize, y: u64) -> Result<()> {
        if self.pd1_wo(l, out, rows, y)? { return Ok(()); }   // PD1: one wo_a launch for every group, wo_b without a prep
        let (hd, d) = (self.cfg.head_dim, self.cfg.hidden);
        let (nh, ng) = (self.rg.nh, self.rg.ng);
        let olr = self.cfg.o_lora_rank;
        let gin = nh * hd / ng;
        let mid = self.f.b.take(rows * ng * olr * 2)?;
        for i in 0..ng {
            let gi = self.rg.g0 + i;
            let lw = &self.layers[l].as_ref().unwrap().w;
            self.f.lin_rows("tp.attn.wo_a (groups)", lw, C::AttnWoA, Some(gi as u16), out + (i * gin * 2) as u64, nh * hd, gin, rows, mid + (i * olr * 2) as u64, ng * olr, false)?;
        }
        let wo = self.layers[l].as_ref().unwrap().wo_b.clone();
        let mut ya = f32a(y, rows, d, d);
        ya.cols = d;
        self.lin_w("tp.attn.wo_b (group inputs, fp32 partial)", &wo, &bf16(mid, rows, ng * olr, ng * olr), &ya, Accum::Overwrite)
    }

    /// A linear of the TP forward's own views (heads / groups / K slices) through the backend; a prefill piece with
    /// fewer rows than `Fwd::wide_min` runs padded to it (the wide row family, leg CE: a row's bits never depend on
    /// how many rows its call had, so CED and chunked prefills equal the every-layer prefill bit for bit).
    pub fn lin_w(&mut self, what: &'static str, w: &(DenseWeight, BackendKind), x: &crate::dsv41::weights::Act, y: &crate::dsv41::weights::Act, acc: Accum) -> Result<()> {
        use crate::dsv41::weights::{Act, ActDtype};
        let (m, n) = (self.f.wide_min, x.rows);
        if m == 0 || n >= m { glue::linear(self.f.g, what, w, x, y, acc, &self.f.scr)?; return Ok(()); }
        let es = |d: ActDtype| if matches!(d, ActDtype::F32) { 4usize } else { 2 };
        let (xs, ys) = (es(x.dtype), es(y.dtype));
        let (xb, yb) = (self.f.b.take(m * x.cols * xs)?, self.f.b.take(m * y.cols * ys)?);
        for r in 0..m { self.f.g.d2d(xb + (r * x.cols * xs) as u64, x.ptr.0 + (r.min(n - 1) * x.ld * xs) as u64, x.cols * xs)?; }
        if matches!(acc, Accum::Add) { for r in 0..n { self.f.g.d2d(yb + (r * y.cols * ys) as u64, y.ptr.0 + (r * y.ld * ys) as u64, y.cols * ys)?; } }
        let xa = Act { ptr: DevPtr(xb), rows: m, cols: x.cols, ld: x.cols, dtype: x.dtype };
        let ya = Act { ptr: DevPtr(yb), rows: m, cols: y.cols, ld: y.cols, dtype: y.dtype };
        glue::linear(self.f.g, what, w, &xa, &ya, acc, &self.f.scr)?;
        for r in 0..n { self.f.g.d2d(y.ptr.0 + (r * y.ld * ys) as u64, yb + (r * y.cols * ys) as u64, y.cols * ys)?; }
        Ok(())
    }

    /// q = q_norm(wq_a(ain)) (replicated: the indexer reads it too).
    pub fn q_lora(&mut self, l: usize, ain: u64, rows: usize) -> Result<u64> {
        let (d, ql) = (self.cfg.hidden, self.cfg.q_lora_rank);
        let lw = &self.layers[l].as_ref().unwrap().w;
        let qa = match self.pd1.qa.take() {
            Some((ll, p)) if ll == l => p,   // PD1: computed by the grouped front launch
            _ => { let qa = self.f.b.take(rows * ql * 2)?; self.f.lin_rows("attn.wq_a", lw, C::AttnWqA, None, ain, d, d, rows, qa, ql, false)?; qa }
        };
        let y = self.f.b.take(rows * ql * 2)?;
        if self.pd1_normrot(l, qa, y, rows)? { return Ok(y); }   // PD1: the q norm also writes wq_b's (idx wq_b's) rotated rows
        let lw = &self.layers[l].as_ref().unwrap().w;
        let w = lw.param(C::AttnQNorm)?;
        let a = km::RmsNormArgs { x: qa, x_ld: ql as i64, w: w.ptr.0, y, y_ld: ql as i64, w_fmt: fmt_of(w)?, dim: ql as i32, rows: rows as i32,
                                  eps: self.cfg.rms_eps as f32, inv_dim: 1.0 / ql as f32 };
        self.f.g.launch1("d41_rmsnorm", (rows as u32, 1, 1), (256, 1, 1), 0, &a)?;
        Ok(y)
    }

    // ------------------------------------------------------------------ FFN (inner width)
    /// Routed + shared experts of the rank's inner slices -> FP32 partial `y` [rows][dim] (overwritten).
    pub fn ffn_core(&mut self, l: usize, fin: u64, rows: usize, ids: u64, wts: u64, y: u64) -> Result<()> {
        let d = self.cfg.hidden;
        let lay = self.layers[l].as_ref().context("layer not loaded")?;
        let mw = &lay.w.moe.as_ref().context("no routed experts")?.0;
        let route = Route { ids: DevPtr(ids), weights: DevPtr(wts), top_k: mw.top_k };
        let s = Scratch { ptr: DevPtr(self.f.scr.ptr), bytes: self.f.scr.bytes };
        // leg PF1: y is OVERWRITTEN (the decode kernels' combine writes 0 + sum; any other path zeroes y first itself), so no
        // memset node sits between the router and the MoE call
        let cls = moe_ex_ow(&self.f.g.l(), mw, Some(&lay.shared), &bf16(fin, rows, d, d), &route, &f32a(y, rows, d, d), s, MoePath::Auto, true)?;
        receipts::count("tp.moe.routed+shared (inner slices)", BackendKind::Exl3, if rows <= 32 { "d41x_md_{prep,gu,dn} (fused shared)" } else { "d41x_mp_{plan,gu,dn,combine} (fused shared)" }, rows, cls);
        Ok(())
    }

    // ------------------------------------------------------------------ Engram
    /// The rank's `wkv` partial of Engram layer `el` over its columns' rows `e` ([rows][n_own][256] bf16) -> FP32 `y`
    /// [rows][(hc + 1) * dim] (overwritten).
    pub fn engram_partial(&mut self, el: usize, e: u64, rows: usize, y: u64) -> Result<()> {
        let kv_out = (self.cfg.hc + 1) * self.cfg.hidden;
        let ld = self.rg.neh * (self.cfg.engram_max_ngram - 1) * self.cfg.engram_head_dim;
        let calls: Vec<(usize, usize, (DenseWeight, BackendKind))> = self.layers[el].as_ref().unwrap().engram.iter().map(|c| (c.x_off, c.k, c.w.clone())).collect();
        ensure!(!calls.is_empty(), "layer {el} has no Engram wkv");
        for (i, (x_off, k, w)) in calls.into_iter().enumerate() {
            let x = crate::dsv41::weights::Act { ptr: DevPtr(e + (x_off * 2) as u64), rows, cols: k, ld, dtype: crate::dsv41::weights::ActDtype::Bf16 };
            let mut ya = f32a(y, rows, kv_out, kv_out);
            ya.cols = kv_out;
            self.lin_w("tp.engram.wkv (own columns, fp32 partial)", &w, &x, &ya, if i == 0 { Accum::Overwrite } else { Accum::Add })?;
        }
        Ok(())
    }

    /// The Engram gate of layer `el` on stream h with the reduced kv (bf16 [rows][(hc + 1) * dim]) -> new stream.
    pub fn engram_gate(&mut self, el: usize, h: u64, kvb: u64, rows: usize) -> Result<u64> {
        let d = self.cfg.hidden;
        let kv_out = (self.cfg.hc + 1) * d;
        let lw = &self.layers[el].as_ref().unwrap().w;
        let (wq, wk) = (lw.param(C::EngramQ)?, lw.param(C::EngramK)?);
        let out = self.f.b.take(rows * 4 * d * 2)?;
        let a = km::EngramGateArgs { h, h_ld: (4 * d) as i64, kv: kvb, kv_ld: kv_out as i64, wq: wq.ptr.0, wk: wk.ptr.0, gate_on: 0, out, o_ld: (4 * d) as i64,
                                     wq_fmt: fmt_of(wq)?, wk_fmt: fmt_of(wk)?, dim: d as i32, rows: rows as i32, eps: self.cfg.rms_eps as f32, inv_dim: 1.0 / d as f32,
                                     inv_sqrt_dim: (d as f64).powf(-0.5) as f32, clamp_min: 1e-6 };
        self.f.g.launch1("d41_engram_gate", (4, rows as u32, 1), (256, 1, 1), 0, &a)?;
        receipts::kernel("engram.gate", "d41_engram_gate", rows);
        Ok(out)
    }

    /// The Engram rows of this call's tokens on this rank's heads (both Engram layers): EIO's prefill channel. Each
    /// segment hashes with its lane's n-gram state (advanced here). Returns the [layers][max_tok][n_own][256] bf16
    /// output (layer li at + li * max_tok * n_own * 256 * 2).
    pub(super) fn engram_rows(&mut self, toks: &[u32]) -> Result<Option<(u64, usize)>> {
        self.engram_rows_vl(toks, None)
    }

    /// Leg VIS1: [`Self::engram_rows`] with optional `live` mask (`false` = DEAD image-span token -> `ZERO_SLOT`, lookback blocked).
    pub(super) fn engram_rows_vl(&mut self, toks: &[u32], live: Option<&[bool]>) -> Result<Option<(u64, usize)>> {
        let Some(hs) = self.hasher.as_ref() else { return Ok(None) };
        let per = hs.ids_per_token();
        let mut ids = vec![0u32; toks.len() * per];
        for sg in self.segs.clone() {
            let lv = live.map(|l| &l[sg.row0..sg.row0 + sg.n]);
            hs.hash_chunk(&mut self.lanes[sg.lane].hash, &toks[sg.row0..sg.row0 + sg.n], lv, &mut ids[sg.row0 * per..(sg.row0 + sg.n) * per])?;
        }
        Ok(Some(self.engram_rows_mode_vl(&ids, live, toks.len(), false)?))
    }

    /// Leg PH1: [`Self::engram_rows`] for a forward call whose body launches the decode itself (`--dsv41-engram-in-graph`):
    /// the rows are hashed and their reads submitted now, the decode step waits in the round for the readers' publish.
    pub(super) fn engram_rows_call(&mut self, toks: &[u32]) -> Result<Option<(u64, usize)>> {
        let Some(hs) = self.hasher.as_ref() else { return Ok(None) };
        let per = hs.ids_per_token();
        let mut ids = vec![0u32; toks.len() * per];
        for sg in self.segs.clone() {
            hs.hash_chunk(&mut self.lanes[sg.lane].hash, &toks[sg.row0..sg.row0 + sg.n], None, &mut ids[sg.row0 * per..(sg.row0 + sg.n) * per])?;
        }
        let defer = self.eng_in_graph;
        Ok(Some(self.engram_rows_mode(&ids, toks.len(), defer)?))
    }

    /// Leg PH1: launch the call's pending Engram decode step on the compute stream (captured when the round is).
    fn engram_decode_pending(&mut self) -> Result<()> {
        if let Some(sl) = self.eng_pending.take() {
            crate::dsv41::gpu::engram::launch_step_decode_on(&self.f.g.dev, self.cs, &sl)?;
            receipts::kernel("engram.rows (own heads; in the round, before layer 0's FFN)", "d41_engram_step_decode", sl.max_tok as usize);
        }
        Ok(())
    }

    /// The Engram rows of `n` tokens given their hash ids (every column of both layers per token, ENGa's order).
    pub fn engram_rows_ids(&mut self, ids: &[u32], n: usize) -> Result<(u64, usize)> { self.engram_rows_mode(ids, n, false) }

    /// `defer`: leave the decode step for the call's body (leg PH1); otherwise launch it now on the device stream.
    fn engram_rows_mode(&mut self, ids: &[u32], n: usize, defer: bool) -> Result<(u64, usize)> {
        self.engram_rows_mode_vl(ids, None, n, defer)
    }

    fn engram_rows_mode_vl(&mut self, ids: &[u32], live: Option<&[bool]>, n: usize, defer: bool) -> Result<(u64, usize)> {
        ensure!(self.eng_pending.is_none(), "an Engram decode step is still pending from an earlier call");
        let io = self.eio.as_mut().context("the Engram runtime is not open")?;
        ensure!(n <= self.eio_max, "{n} tokens > the Engram channel's {}", self.eio_max);
        let ch = io.ch_prefill();
        io.submit_prefill(ids, live)?;
        if defer {
            self.eng_pending = Some(io.step_launch(ch, self.eout.ptr(), 0, crate::dsv41::gpu::engram::MODE_DECODE));
        } else {
            io.launch(ch, self.eout.ptr(), 0, crate::dsv41::gpu::engram::MODE_DECODE)?;
            receipts::kernel("engram.rows (own heads)", "d41_engram_step_decode", n);
        }
        let stride = self.eio_max * io.n_own() * 256 * 2;
        Ok((self.eout.ptr(), stride))
    }

    /// Leg PP1 (the reference recipe's prompt worker reads a chunk ahead): read the Engram rows of `next` (the tokens right after
    /// the rows lane `lane`'s hash state has advanced through) into the row cache, so the step of the chunk that holds them finds
    /// them resident. Hashes with a COPY of the lane's state (the lane's state advances only when those rows are submitted). The
    /// same rows reach the GPU either way (same bits); a no-op without the Engram runtime, its row cache, or with
    /// `--dsv41-pf-engram-ahead off`. Returns the reads issued.
    pub(super) fn engram_ahead(&mut self, lane: usize, next: &[u32]) -> Result<usize> {
        self.engram_ahead_vl(lane, next, None)
    }

    pub(super) fn engram_ahead_vl(&mut self, lane: usize, next: &[u32], live: Option<&[bool]>) -> Result<usize> {
        if next.is_empty() || !crate::dsv41::gpu::pf_on("dsv41-pf-engram-ahead") { return Ok(0); }
        let (Some(hs), Some(io)) = (self.hasher.as_ref(), self.eio.as_mut()) else { return Ok(0) };
        let n = next.len().min(self.eio_max);
        let lv = live.map(|l| &l[..n]);
        let mut st = self.lanes[lane].hash;
        let mut ids = vec![0u32; n * hs.ids_per_token()];
        hs.hash_chunk(&mut st, &next[..n], lv, &mut ids)?;
        io.warm(&ids, lv, false)
    }

    /// Leg P6-7: warm row 0's Engram n-gram lookups while `draft()` executes (`--dsv41-engram-draft-warm on`, opt-in).
    pub fn engram_warm_pending(&mut self, lanes_toks: &[(usize, u32)]) -> Result<usize> {
        let on = crate::opts::lookup("dsv41-engram-draft-warm").and_then(crate::opts::get).as_deref() == Some("on");
        if lanes_toks.is_empty() || !on { return Ok(0); }
        let (Some(hs), Some(io)) = (self.hasher.as_ref(), self.eio.as_mut()) else { return Ok(0) };
        let mut total = 0usize;
        for &(lane, tok) in lanes_toks {
            total += crate::dsv41::gpu::dspark::prefetch::warm_token(io, hs, &self.lanes[lane].hash, tok)?;
        }
        Ok(total)
    }

    /// Leg P6-7: warm the verify block `[tok, drafts..]`'s Engram n-gram lookups before `forward_segs` runs (`--dsv41-engram-draft-warm on`, opt-in).
    pub fn engram_warm_drafts(&mut self, lanes_blocks: &[(usize, u32, &[u32])]) -> Result<usize> {
        let on = crate::opts::lookup("dsv41-engram-draft-warm").and_then(crate::opts::get).as_deref() == Some("on");
        if lanes_blocks.is_empty() || !on { return Ok(0); }
        let (Some(hs), Some(io)) = (self.hasher.as_ref(), self.eio.as_mut()) else { return Ok(0) };
        let mut total = 0usize;
        for &(lane, tok, drafts) in lanes_blocks {
            if !drafts.is_empty() {
                total += crate::dsv41::gpu::dspark::prefetch::warm_drafts(io, hs, &self.lanes[lane].hash, tok, drafts)?;
            }
        }
        Ok(total)
    }

    pub fn engram_check_pub(&mut self) -> Result<()> { self.engram_check() }

    /// Check the Engram channel after a call (a failed read must fail the call, never feed zeros).
    pub(super) fn engram_check(&mut self) -> Result<()> {
        if let Some(io) = self.eio.as_mut() {
            let ch = io.ch_prefill();
            let e = io.gpu_ctrl(ch).err;
            ensure!(e == 0, "Engram step error {e} ({})", crate::dsv41::gpu::engram::err_name(e));
            io.housekeep();
        }
        Ok(())
    }

    // ------------------------------------------------------------------ embedding
    /// The call's stream [rows][hc * dim] bf16: the embedding rows (gathered where held; broadcast by the fp32
    /// all-reduce when only rank 0 holds the table) repeated hc times.
    pub fn embed(&mut self, ids_dev: u64, rows: usize) -> Result<u64> {
        let d = self.cfg.hidden;
        let bc = self.bcast.ptr();
        let all_hold = !self.embed_bcast;
        match &self.glob.embed {
            Some(w) => {
                let e = self.f.b.take(rows * d * 2)?;
                glue::backend(w.1).gather(&self.f.g.l(), &w.0, DevPtr(ids_dev), rows, &bf16(e, rows, d, d))?;
                self.k.bf16_f32(self.cs, e, bc, rows * d)?;
            }
            None => {
                ensure!(!all_hold, "rank {}: no embedding table", self.rank);
                // leg PF1: zeros by a kernel (a PDL chain link) at decode / verify / draft widths; a memset for prefill-sized calls
                if rows <= HEAD_MAX.max(MAX_LANES * 5) { self.k.bf16_f32(self.cs, self.zrows.ptr(), bc, rows * d)?; } else { self.f.g.memset(bc, rows * d * 4, 0)?; }
            }
        }
        if !all_hold { self.reduce(bc, rows * d)?; }
        let h = self.f.b.take(rows * self.cfg.hc * d * 2)?;
        self.k.expand_hc(self.cs, bc, d, h, rows, d, self.cfg.hc)?;
        Ok(h)
    }

    /// Leg VIS1: prefill chunk `[c0, c0 + rows)` overlapping an image span. Rank 0 gathers text embeddings, splices
    /// `image_start` / `aligner_embeds` / `image_newline` / `image_end` into the BF16 rows, converts to F32 in `bcast`,
    /// and broadcasts across `world > 1` (since Vision weights reside only on Rank 0 even at TP4/TP8) before `expand_hc`.
    pub fn embed_vl_chunk(&mut self, lane: usize, ids_dev: u64, c0: usize, rows: usize) -> Result<u64> {
        let d = self.cfg.hidden;
        let bc = self.bcast.ptr();
        if self.rank == 0 {
            let w = self.glob.embed.as_ref().context("rank 0: no embedding table")?;
            let vis = self.vision.as_ref().context("rank 0: vision weights not loaded")?;
            let e = self.f.b.take(rows * d * 2)?;
            glue::backend(w.1).gather(&self.f.g.l(), &w.0, DevPtr(ids_dev), rows, &bf16(e, rows, d, d))?;
            let c1 = c0 + rows;
            ensure!(c1 <= self.lanes[lane].vl_row_ops.len(), "lane {lane}: vl_row_ops len {} < chunk end {c1}", self.lanes[lane].vl_row_ops.len());
            let row_ops = &self.lanes[lane].vl_row_ops[c0..c1];
            let align_ptr = self.lanes[lane].vl_aligner.as_ref().map_or(0, |b| b.ptr());
            if row_ops.iter().any(|&op| op >= 0) {
                ensure!(align_ptr != 0, "rank 0: missing vl_aligner buffer for uncached image rows in {c0}..{c1}");
            }
            vis.splice_chunk_embeddings(self.f.g, &mut self.f.b, e, align_ptr, row_ops)?;
            self.k.bf16_f32(self.cs, e, bc, rows * d)?;
        } else if rows <= HEAD_MAX.max(MAX_LANES * 5) {
            self.k.bf16_f32(self.cs, self.zrows.ptr(), bc, rows * d)?;
        } else {
            self.f.g.memset(bc, rows * d * 4, 0)?;
        }
        if self.world > 1 { self.reduce(bc, rows * d)?; }
        let h = self.f.b.take(rows * self.cfg.hc * d * 2)?;
        self.k.expand_hc(self.cs, bc, d, h, rows, d, self.cfg.hc)?;
        Ok(h)
    }

    // ------------------------------------------------------------------ the forward
    /// One call of `toks` at the lane 0's next positions (the single-lane form of `forward_segs`).
    pub fn forward(&mut self, toks: &[u32], want: HeadWant, probe: Option<&[i32]>) -> Result<Vec<TpOut>> {
        self.forward_segs(&[(0, toks)], want, probe)
    }

    /// Lay out a call: `parts` = (lane, tokens) in row order, each lane at most once, each at its lane's next positions.
    pub fn plan_call(&mut self, parts: &[(usize, &[u32])]) -> Result<Vec<u32>> {
        ensure!(!parts.is_empty(), "an empty call");
        let mut segs = Vec::with_capacity(parts.len());
        let mut toks = Vec::new();
        for &(lane, t) in parts {
            ensure!(lane < self.lanes.len(), "lane {lane} of {}", self.lanes.len());
            ensure!(!t.is_empty(), "lane {lane}: no rows");
            ensure!(!segs.iter().any(|s: &Seg| s.lane == lane), "lane {lane} twice in one call");
            segs.push(Seg { lane, row0: toks.len(), n: t.len(), p0: self.lanes[lane].pos });
            toks.extend_from_slice(t);
        }
        ensure!(toks.len() <= self.max_rows, "a call of {} rows (1..={})", toks.len(), self.max_rows);
        ensure!(segs.len() == 1 || toks.len() <= HEAD_MAX, "a mixed-lane call of {} rows (<= {HEAD_MAX}: decode / verify rounds)", toks.len());
        self.segs = segs;
        Ok(toks)
    }

    /// The call's per-row positions and token ids (and lanes, for KA) on the device.
    pub(super) fn upload_rows(&mut self, toks: &[u32]) -> Result<()> {
        let mut pos = Vec::with_capacity(toks.len());
        let mut lane = Vec::with_capacity(toks.len());
        for sg in &self.segs { for i in 0..sg.n { pos.push((sg.p0 + i) as i32); lane.push(self.segs.iter().position(|x| x == sg).unwrap() as i32); } }
        self.f.g.h2d_async(self.pos_buf.ptr(), &pos)?;
        self.f.g.h2d_async(self.ids_buf.ptr(), &toks.iter().map(|&t| t as i32).collect::<Vec<i32>>())?;
        if self.segs.len() > 1 { self.f.g.h2d_async(self.lane_buf.ptr(), &lane)?; }
        if self.rk.on && toks.len() <= HEAD_MAX {
            // leg PG1: the call's segmentation as data (a rows-keyed graph reads it; the eager path does not)
            let (t, lid) = super::graph::seg_table(&self.segs.iter().map(|s| (s.lane, s.row0, s.n)).collect::<Vec<_>>(), self.lanes.len());
            self.f.g.h2d_async(self.rk.seg.ptr(), &t)?;
            self.f.g.h2d_async(self.rk.lid.ptr(), &lid)?;
        }
        Ok(())
    }

    /// One call: every lane of `parts` gets its rows at its next positions (a prefill chunk of one lane, or a decode /
    /// verify round of up to `MAX_LANES` lanes, <= 24 rows). `want`: which rows get a head exchange (row order of the
    /// call); `probe[i]`: the token row i's log-probability is wanted for (teacher forcing; -1 = none). Every per-row
    /// piece is batch invariant (a lane's bytes do not depend on the other lanes of the call); the lane-specific pieces
    /// (window, compressor, indexer row blocks, attention sources, Engram hashing) run per segment.
    pub fn forward_segs(&mut self, parts: &[(usize, &[u32])], want: HeadWant, probe: Option<&[i32]>) -> Result<Vec<TpOut>> {
        ensure!(self.layers[..self.cfg.layers].iter().all(|x| x.is_some()), "the forward needs all {} trunk layers (this source holds a slice)", self.cfg.layers);
        let toks = self.plan_call(parts)?;
        let rows = toks.len();
        super::metrics::call(self.segs.len(), rows);
        self.upload_rows(&toks)?;
        let erows = self.engram_rows_call(&toks)?;
        super::hostprof::mark(super::hostprof::M_PLANNED);
        let head_rows: Vec<usize> = match want { HeadWant::None => vec![], HeadWant::Last => self.segs.iter().map(|s| s.row0 + s.n - 1).collect(), HeadWant::All => (0..rows).collect() };
        let graphable = self.graphs.is_some() && probe.is_none() && rows <= HEAD_MAX && self.segs.iter().all(|s| s.n <= graph_seg_max(self.cfg))
            && self.taps.map_or(true, |t| t.rider && t.row0 == 0 && t.n == rows && t.dst0 == 0);
        let mut replayed = false;
        if graphable {
            if !head_rows.is_empty() { self.head_params(&head_rows, &vec![-1; head_rows.len()])?; }
            let hr = head_rows.clone();
            replayed = self.graph_call(rows, &head_rows, &mut |m: &mut Self| {
                m.swap_bump();
                let (h, pre) = m.body(rows, erows)?;
                m.last = Some((h, pre));
                if !hr.is_empty() { m.head_launch(h, pre, &hr)?; }
                Ok(())
            })?;
        }
        super::hostprof::mark(super::hostprof::M_GRAPH);
        if !replayed {
            if self.graphs.is_some() { super::metrics::graph_eager(); }
            self.swap_bump();
            let (h, pre) = self.body(rows, erows)?;
            self.last = Some((h, pre));
        }
        // leg PH1: a replayed round decoded its step inside the graph; a body that never reached an Engram layer (none
        // in this call) still consumes its step, so the channel's device counter stays in step with the host's
        if replayed { self.eng_pending = None; } else { self.engram_decode_pending()?; }
        self.engram_check()?;
        for sg in self.segs.clone() {
            let ln = &mut self.lanes[sg.lane];
            ln.pos = sg.p0 + sg.n;
            ln.hist.extend_from_slice(&toks[sg.row0..sg.row0 + sg.n]);
            // the groups every KV source holds after the call (a replayed graph does not run the host side of the compressor)
            for st in ln.src.values_mut() { st.n_groups = ln.pos / st.ratio.max(1) as usize; }
        }
        if replayed {
            let outs = if head_rows.is_empty() { Vec::new() } else { self.head_read(head_rows.len())? };
            self.comm.check("after a forward call")?;
            super::hostprof::mark(super::hostprof::M_FWD_DONE);
            return Ok(outs);
        }
        // ---- head: one exchange per <= 24 head rows (the last rows of several lanes are gathered into one block)
        let (h, pre) = self.last.unwrap();
        let mut outs = Vec::with_capacity(head_rows.len());
        for chunk in head_rows.chunks(HEAD_MAX) {
            let pr: Vec<i32> = chunk.iter().map(|&r| probe.map_or(-1, |p| p[r])).collect();
            // the head's per-exchange buffers come from one mark of the arena: every exchange of this call reuses them
            let mark = self.f.b.off;
            self.head_params(chunk, &pr)?;
            self.head_launch(h, pre, chunk)?;
            outs.extend(self.head_read(chunk.len())?);
            self.f.b.off = mark;
        }
        self.comm.check("after a forward call")?;
        super::hostprof::mark(super::hostprof::M_FWD_DONE);
        Ok(outs)
    }

    /// The layers of the call laid out by `plan_call` (rows, positions and ids uploaded; `erows` = the Engram rows):
    /// the embedding, the 40 trunk layers with their reduces. Returns the final stream and FFN record.
    fn body(&mut self, rows: usize, erows: Option<(u64, usize)>) -> Result<(u64, u64)> {
        let cfg = self.cfg;
        let (d, hc) = (cfg.hidden, cfg.hc);
        let (pos, ids) = (self.pos_buf.ptr(), self.ids_buf.ptr());
        let mut h = self.embed(ids, rows)?;
        let mut pre = self.f.b.take(rows * 24 * 4)?;
        self.k.fill_pre(self.cs, pre, 24, rows, hc)?;
        let mut kv_next: Option<u64> = None;
        let (fa, fb) = (self.frames[0].ptr(), self.frames[1].ptr());
        let mut sel_of: HashMap<usize, u64> = HashMap::new();
        let mut uni: Option<Union> = None;
        let last_target = cfg.dspark_targets.iter().copied().max();
        self.pd1.in_body = true;
        self.pd1.stats = Default::default();
        for l in 0..cfg.layers {
            // NIGHT1: a prefill-sized call (scoring, the every-layer prefill) paces inside the pass like CED's chunks;
            // decode / verify calls (<= HEAD_MAX rows, graph captures) are paced once per op by the serve loop
            if rows > HEAD_MAX && l > 0 && l % crate::thermal::PASS_EVERY_LAYERS == 0 { let g = self.f.g; crate::thermal::pace_pass(self.gm, || g.sync())?; }
            // layer l's arena: the previous layer's outputs (h, pre, the Engram kv) stay valid in the other one
            self.swap_bump();
            // ---- Engram (layers 1, 14): the stream before the attention
            if let Some(kvb) = kv_next.take() {
                h = self.engram_gate(l, h, kvb, rows)?;
            }
            // ---- the DSpark taps: the mean over the hc copies of the stream at the input of the target layers
            let mut a_extra = 0usize;
            if let Some(ti) = cfg.dspark_targets.iter().position(|&t| t == l) {
                if let Some(tp) = self.taps {
                    self.k.tap(self.cs, h, hc, d, tp.row0, tp.n, tp.buf, tp.dst0, tp.ld, ti * d)?;
                    if tp.rider && Some(l) == last_target {
                        self.main_proj_partial(tp.buf, tp.ld, rows, fa + (rows * (d + MIX_STRIDE) * 4) as u64)?;
                        a_extra = d;
                    }
                }
            }
            // ---- attention sublayer
            let ratio = cfg.ratio(l);
            let compressed = ratio > 0;
            // PA1 (TP1 / TP2): the mix (both K slices) + Sinkhorn on the side stream, joined before this sublayer's hc_post
            let side_a = self.pa1_mix_side(l, true, self.layers[l].as_ref().unwrap().mix_a, h, rows)?;
            if side_a.is_none() { self.mix_partial(self.layers[l].as_ref().unwrap().mix_a, h, rows, fa + (rows * d * 4) as u64)?; }
            let ain = {
                let n = self.layers[l].as_ref().unwrap().w.param(C::AttnNorm)?.clone();
                // PA1: pre (the previous FFN's record) was joined before that sublayer's hc_post, an earlier launch on this
                // stream: its loads may precede the wait (layer 0's pre comes from fill_pre, the PDL predecessor: no)
                if side_a.is_some() && l > 0 { self.pa1_pre_norm(h, pre, &n, rows)? } else { self.f.hc_pre_norm(h, pre, 24, &n, rows)? }
            };
            self.pd1_front(l, ain, rows, compressed && cfg.kv_source_of(l) == Some(l))?;   // PD1: wq_a + wkv (+ compressor) in one launch
            let qr = self.q_lora(l, ain, rows)?;
            self.window(l, ain, rows, compressed)?;
            let mut sel = None;
            let mut src = None;
            if compressed {
                let s = cfg.kv_source_of(l).context("compressed layer without a KV source")?;
                if s == l { self.compressor(l, ain, rows)?; }
                src = Some(s);
                let isrc = cfg.index_source_of(l).context("no index source")?;
                if isrc == l {
                    let cand_out = cfg.candidate_source == Some(l);
                    let cin = if cfg.uses_candidates(l) { uni.take() } else { None };
                    // the selection outlives this layer's arena (layers up to the next index source read it): the indexer
                    // writes it straight into the kept buffer (leg PF1: no device copy inside the round)
                    let keep = self.sel_bufs.get(&l).context("index source without a selection buffer")?.ptr();
                    let (_, un) = self.indexer_segs_into(l, s, qr, ain, pos, rows, cand_out, cin.as_ref(), Some(keep))?;
                    if let Some(u) = un { uni = Some(u); } else if let Some(u) = cin { uni = Some(u); }
                    sel_of.insert(l, keep);
                }
                let sl = *sel_of.get(&isrc).with_context(|| format!("layer {l}: index source {isrc} has no selection in this call"))?;
                // a constant cap: -1 entries are masked keys and a fully masked block is an exact identity (KA), so a
                // row's attention does not depend on the other lanes' contexts
                sel = Some((sl, 512usize, 512usize));
            }
            self.attn_core(l, qr, pos, rows, src, sel, fa)?;
            let pf_after = self.pd1.pf.after_reduce.at(rows);
            if !pf_after { self.pd1_pf(1, l, rows)?; }   // PD1 site B: the FFN mix fn, the router gate, the shared expert
            // PA1: with the side mix the rider is not reduced (the frame's d floats only, unless a rider follows it)
            self.reduce_act(fa, rows * if side_a.is_some() && a_extra == 0 { d } else { d + MIX_STRIDE + a_extra })?;
            if pf_after { self.pd1_pf(1, l, rows)?; }
            let rec_a = match side_a { Some(r) => { self.pa1_join()?; r } None => self.mix_finish(l, true, fa + (rows * d * 4) as u64, rows)? };
            let xm = self.hc_post_f32(fa, h, rec_a, rows)?;
            // ---- FFN sublayer
            let engram_next = cfg.engram_layers.contains(&(l + 1));
            let extra = if engram_next { (hc + 1) * d } else { 0 };
            let side_f = self.pa1_mix_side(l, false, self.layers[l].as_ref().unwrap().mix_f, xm, rows)?;
            if side_f.is_none() { self.mix_partial(self.layers[l].as_ref().unwrap().mix_f, xm, rows, fb + (rows * d * 4) as u64)?; }
            let fin = {
                let n = self.layers[l].as_ref().unwrap().w.param(C::FfnNorm)?.clone();
                // PA1: rec_a was joined before the attention's hc_post (the launch before this one)
                if side_f.is_some() { self.pa1_pre_norm(xm, rec_a, &n, rows)? } else { self.f.hc_pre_norm(xm, rec_a, 24, &n, rows)? }
            };
            let (rid, rw) = self.f.router(&self.layers[l].as_ref().unwrap().w, fin, rows)?;
            self.ffn_core(l, fin, rows, rid, rw, fb)?;
            if !pf_after { self.pd1_pf(2, l, rows)?; }   // PD1 site C: the next layer's attention weights (an Engram layer's wkv; the head after the last)
            if engram_next {
                self.engram_decode_pending()?;   // leg PH1: the call's decode step, at its first reader
                let (e, stride) = erows.context("an Engram layer without the Engram runtime")?;
                let li = cfg.engram_layers.iter().position(|&x| x == l + 1).unwrap();
                self.engram_partial(l + 1, e + (li * stride) as u64, rows, fb + (rows * (d + MIX_STRIDE) * 4) as u64)?;
            }
            self.reduce_act(fb, rows * if side_f.is_some() && extra == 0 { d } else { d + MIX_STRIDE + extra })?;
            if pf_after { self.pd1_pf(2, l, rows)?; }
            let rec_f = match side_f { Some(r) => { self.pa1_join()?; r } None => self.mix_finish(l, false, fb + (rows * d * 4) as u64, rows)? };
            h = self.hc_post_f32(fb, xm, rec_f, rows)?;
            if engram_next {
                let kvb = self.f.b.take(rows * extra * 2)?;
                glue::add_round(self.f.g, fb + (rows * (d + MIX_STRIDE) * 4) as u64, 0, kvb, rows, extra)?;
                kv_next = Some(kvb);
            }
            pre = rec_f;
        }
        if let Some(u) = uni.take() { self.uni_bufs = Some((u.uni, u.ucnt)); }
        self.pd1_join()?;   // PD1: the prefetch stream joins the main stream at the end of the call's body
        Ok((h, pre))
    }

    /// The head exchange's row parameters (lane seed / temperature, the token's position, the probe) to the device.
    fn head_params(&mut self, rix: &[usize], probe: &[i32]) -> Result<()> {
        let rp: Vec<TpRow> = rix.iter().enumerate().map(|(i, &r)| {
            let (ln, p) = if self.segs.is_empty() { (0, r) } else { self.row_of(r) };
            TpRow { seed: self.lanes[ln].seed, pos: p as u64 + 1, temp: self.lanes[ln].temp, probe: probe[i], flags: 0, pad: 0 }
        }).collect();
        self.f.g.h2d_async(self.rows_p.ptr(), &rp)
    }

    /// The head of rows `rix` of the final stream h / record pre (<= 24; non-consecutive rows gathered first): final
    /// hc_pre + norm, the rank's vocabulary rows, TPC's exchange (no host sync: `head_read` collects the outputs).
    fn head_launch(&mut self, h: u64, pre: u64, rix: &[usize]) -> Result<()> {
        let (d, hc) = (self.cfg.hidden, self.cfg.hc);
        let n = rix.len();
        ensure!(n >= 1 && n <= HEAD_MAX, "head exchange of {n} rows");
        let consecutive = rix.windows(2).all(|w| w[1] == w[0] + 1);
        let (hh, pp) = if consecutive { (h + (rix[0] * hc * d * 2) as u64, pre + (rix[0] * 24 * 4) as u64) } else {
            let (hs, ps) = (self.f.b.take(n * hc * d * 2)?, self.f.b.take(n * 24 * 4)?);
            for (i, &r) in rix.iter().enumerate() {
                self.f.g.d2d(hs + (i * hc * d * 2) as u64, h + (r * hc * d * 2) as u64, hc * d * 2)?;
                self.f.g.d2d(ps + (i * 24 * 4) as u64, pre + (r * 24 * 4) as u64, 24 * 4)?;
            }
            (hs, ps)
        };
        let nrm = self.glob.norm.clone();
        let x = self.f.hc_pre_norm(hh, pp, 24, &nrm, n)?;
        let vl = self.comm.vocab().len();
        let lg = self.f.b.take(n * vl * 4)?;
        let mut ya = f32a(lg, n, vl, vl);
        ya.cols = vl;
        let hw = self.glob.head.clone();
        self.lin_w("tp.head (vocabulary rows, fp32)", &hw, &bf16(x, n, d, d), &ya, Accum::Overwrite)?;
        // leg Q1's full-logit tap (teacher-forced scoring calls; never under a graph capture)
        if !self.gm && self.logit_tap.is_some() {
            let p0 = if self.segs.is_empty() { rix[0] } else { self.row_of(rix[0]).1 };
            let g = self.f.g;
            if let Some(t) = self.logit_tap.as_mut() { t.take(g, lg, n, vl, p0)?; }
        }
        let call = HeadCall { site: Site::Head, logits: lg, ld: vl, rows: n, rows_p: self.rows_p.ptr(), out: self.outs.ptr() };
        self.comm.head(self.cs, &call)
    }

    /// The head outputs of the last `head_launch` (syncs the stream).
    fn head_read(&mut self, n: usize) -> Result<Vec<TpOut>> {
        self.f.g.sync()?;
        self.f.g.down::<TpOut>(self.outs.ptr(), n)
    }

    /// Swap the two bump arenas without resetting either (a graph replay restores the captured arena state).
    pub(super) fn swap_arenas_only(&mut self) { std::mem::swap(&mut self.f.b, &mut self.spare); }

    /// The lane and position of row `r` of the call in flight.
    fn row_of(&self, r: usize) -> (usize, usize) {
        let sg = self.segs.iter().find(|s| r >= s.row0 && r < s.row0 + s.n).expect("row of the call");
        (sg.lane, sg.p0 + (r - sg.row0))
    }

    /// The rank's partial of `main_proj` (the DSpark drafter's input projection, K-split: this rank's slice of the
    /// 3 x dim taps) over `rows` tap rows -> f32 `y` [rows][dim] (overwritten): rides layer 39's attention reduce.
    pub fn main_proj_partial(&mut self, taps: u64, ld: usize, rows: usize, y: u64) -> Result<()> {
        let mp = self.main_proj.clone().context("main_proj not loaded (the DSpark stages are off)")?;
        let d = self.cfg.hidden;
        let (k0, klen) = match &mp.0.shard { Shard::In(r) => (r.start as usize, (r.end - r.start) as usize), _ => (0, ld) };
        let x = crate::dsv41::weights::Act { ptr: DevPtr(taps + (k0 * 2) as u64), rows, cols: klen, ld, dtype: crate::dsv41::weights::ActDtype::Bf16 };
        let mut ya = f32a(y, rows, d, d);
        ya.cols = d;
        glue::linear(self.f.g, "tp.ds.main_proj (K slice, fp32 partial)", &mp, &x, &ya, Accum::Overwrite, &self.f.scr)?;
        Ok(())
    }

    /// The head's logits of `n` rows of the last call's final stream (rows `r0..r0 + n`) on this rank's vocabulary
    /// shard -> f32 [n][vlocal] at `out` (the DSpark round's target rows; no exchange here).
    pub fn head_logits(&mut self, r0: usize, n: usize, out: u64) -> Result<()> {
        let (h, pre) = self.last.context("no forward call before the head")?;
        let (d, hc) = (self.cfg.hidden, self.cfg.hc);
        let nrm = self.glob.norm.clone();
        let mark = self.f.b.off;
        let x = self.f.hc_pre_norm(h + (r0 * hc * d * 2) as u64, pre + (r0 * 24 * 4) as u64, 24, &nrm, n)?;
        let vl = self.comm.vocab().len();
        let mut ya = f32a(out, n, vl, vl);
        ya.cols = vl;
        let hw = self.glob.head.clone();
        self.lin_w("tp.head (vocabulary rows, fp32)", &hw, &bf16(x, n, d, d), &ya, Accum::Overwrite)?;
        self.f.b.off = mark;
        Ok(())
    }

    /// The head's logits of every segment of the last call, segment rows -> `out + (lane * lane_rows + i) * vlocal * 4`
    /// (leg PF1): consecutive segments whose rows follow each other both in the call and in `out` (lanes l, l + 1, ...
    /// each with `lane_rows` rows) share ONE head launch. Row i of an N-row head call is bit-identical to its 1-row call
    /// (18_*: rows <= 32 class; the head's hc_pre + norm is per row), so every lane's logits equal its own call's. Returns
    /// the number of head launches.
    pub fn head_logits_lanes(&mut self, lane_rows: usize, out: u64) -> Result<usize> {
        let vl = self.comm.vocab().len();
        let segs = self.segs.clone();
        let mut runs: Vec<(usize, usize, usize)> = Vec::new();     // (call row0, rows, first lane)
        for sg in &segs {
            ensure!(sg.n <= lane_rows, "lane {}: {} rows > the {lane_rows} rows a lane's logits hold", sg.lane, sg.n);
            if let Some(r) = runs.last_mut() {
                let (r0, n, l0) = *r;
                if r0 + n == sg.row0 && l0 * lane_rows + n == sg.lane * lane_rows && n % lane_rows == 0 && r0 + n + sg.n <= 32 { r.1 += sg.n; continue; }
            }
            runs.push((sg.row0, sg.n, sg.lane));
        }
        for &(r0, n, l0) in &runs { self.head_logits(r0, n, out + (l0 * lane_rows * vl * 4) as u64)?; }
        super::metrics::head_launches(runs.len(), segs.len());
        Ok(runs.len())
    }

    /// TPC's head exchange of row `r` of the last call's final stream (greedy / the lane's sampling, log-sum).
    pub fn head_last(&mut self, r: usize) -> Result<TpOut> {
        let (h, pre) = self.last.context("no forward call before the head")?;
        let (d, hc) = (self.cfg.hidden, self.cfg.hc);
        let mark = self.f.b.off;
        let o = self.head(h + (r * hc * d * 2) as u64, pre + (r * 24 * 4) as u64, 24, 1, &[r], &[-1])?;
        self.f.b.off = mark;
        o.first().copied().context("no head output")
    }

    /// The head exchange of `n` (<= 24) rows of stream h: final hc_pre + norm, the rank's vocabulary rows, TPC's exchange.
    /// `rix[i]` = row i's index in the call in flight (its lane and position give the row's sampling key); h / pre point
    /// at the first of n consecutive rows.
    pub fn head(&mut self, h: u64, pre: u64, pre_stride: usize, n: usize, rix: &[usize], probe: &[i32]) -> Result<Vec<TpOut>> {
        ensure!(pre_stride == 24 && rix.len() == n, "head: stride {pre_stride}, {} row ids for {n} rows", rix.len());
        let (d, hc) = (self.cfg.hidden, self.cfg.hc);
        self.head_params(rix, probe)?;
        // h / pre already point at the first row: present the rows as 0..n of that block
        let local: Vec<usize> = (0..n).collect();
        let _ = (d, hc);
        self.head_launch(h, pre, &local)?;
        self.head_read(n)
    }

    /// Rank 0's token ids to every rank (the others pass None): the length, then the ids as exact 16-bit halves through
    /// one fp32 all-reduce of the embedding broadcast frame (others add zeros).
    pub fn bcast_tokens(&mut self, mine: Option<Vec<u32>>) -> Result<Vec<u32>> {
        let n = self.bcast_small(&[mine.as_ref().map_or(0, |v| v.len()) as f32])?[0] as usize;
        let len = (2 * n).div_ceil(4) * 4;
        ensure!(len * 4 <= self.bcast.bytes, "{n} tokens exceed the broadcast frame");
        let b = self.bcast.ptr();
        self.f.g.memset(b, len * 4, 0)?;
        if self.rank == 0 {
            let v = mine.context("rank 0 must provide the tokens")?;
            ensure!(v.len() == n, "token count changed");
            let mut h = vec![0f32; len];
            for (i, &t) in v.iter().enumerate() { h[2 * i] = (t & 0xFFFF) as f32; h[2 * i + 1] = (t >> 16) as f32; }
            self.f.g.h2d_async(b, &h)?;
        }
        self.reduce(b, len)?;
        let h = self.f.g.down::<f32>(b, 2 * n)?;
        self.comm.check("a token broadcast")?;
        Ok((0..n).map(|i| h[2 * i] as u32 | ((h[2 * i + 1] as u32) << 16)).collect())
    }

    /// Forget lane `lane` (a new prompt).
    pub fn reset_lane_n(&mut self, lane: usize) -> Result<()> {
        let ln = self.lanes.get_mut(lane).context("no such lane")?;
        ln.pos = 0;
        ln.hash = NgramHashState::new();
        ln.hist.clear();
        ln.vl_types.clear();
        ln.vl_row_ops.clear();
        ln.vl_aligner = None;
        ln.kind = crate::dsv41::ced::schedule::StateKind::Full;
        for s in ln.src.values_mut() { s.n_groups = 0; }
        Ok(())
    }
    /// Forget lane 0 (TP's single-lane programs).
    pub fn reset_lane(&mut self) -> Result<()> { self.reset_lane_n(0) }

    /// Roll lane `lane` back to `pos` positions (a speculative verify's rejected rows): the compressor caches keep the
    /// groups the first `pos` positions complete (the frontier ring and the 134-slot rings are sized for the 6
    /// backtracking rows: what the rejected rows wrote is overwritten before anything reads it), the Engram hash
    /// state is rebuilt from the token tail.
    pub fn rollback(&mut self, lane: usize, pos: usize) -> Result<()> {
        let ln = self.lanes.get_mut(lane).context("no such lane")?;
        ensure!(pos <= ln.pos && ln.pos - pos <= self.cfg.backtrack(), "lane {lane}: rollback {} -> {pos} (at most {} rows)", ln.pos, self.cfg.backtrack());
        if pos == ln.pos { return Ok(()); }
        ln.pos = pos;
        ln.kind = crate::dsv41::ced::schedule::StateKind::Full;
        ln.hist.truncate(pos);
        if ln.vl_types.len() > pos { ln.vl_types.truncate(pos); }
        for s in ln.src.values_mut() { s.n_groups = pos / s.ratio.max(1) as usize; }
        self.rebuild_lane_hash(lane)
    }

    /// Rebuild `lanes[lane].hash` from the committed `hist[..pos]` tail (shared by `rollback` and `cpv` wide rollbacks).
    pub fn rebuild_lane_hash(&mut self, lane: usize) -> Result<()> {
        let ln = self.lanes.get_mut(lane).context("no such lane")?;
        let pos = ln.pos;
        if let Some(hs) = self.hasher.as_ref() {
            let t0 = ln.hist.len().saturating_sub(self.cfg.engram_max_ngram - 1);
            let tail = &ln.hist[t0..];
            if ln.vl_types.is_empty() {
                ln.hash = hs.state_at(pos as u64, tail, None)?;
            } else {
                let live_tail: Vec<bool> = (t0..ln.hist.len()).map(|p| ln.vl_types.get(p).copied().unwrap_or(-1) < 0).collect();
                ln.hash = hs.state_at(pos as u64, tail, Some(&live_tail))?;
            }
        } else {
            ln.hash = NgramHashState::new();
        }
        Ok(())
    }

    // ------------------------------------------------------------------ SP prefill helpers (leg P7SP)
    /// Cluster-agreed flag that every rank holds `hc_attn_fn` and `hc_ffn_fn` whole on all trunk layers.
    pub(super) fn hc_whole_ok(&mut self) -> Result<bool> {
        if !matches!(self.world, 2 | 3 | 4 | 6 | 8) { return Ok(false); }
        if self.pa1.side { return Ok(true); }
        if let Some(ok) = self.hc_whole_agreed { return Ok(ok); }
        let t = self.tiny.ptr();
        self.f.g.memset(t, 64, 0)?;
        let mut mine = [0f32; 8];
        mine[self.rank] = if self.hc_whole_local { 0.0 } else { 1.0 };
        self.f.g.h2d_async(t, &mine[..self.world])?;
        self.reduce(t, 16)?;
        let lacks = self.f.g.down::<f32>(t, self.world)?;
        self.comm.check("SP hc_whole agreement")?;
        let ok = lacks.iter().all(|&x| x == 0.0);
        self.hc_whole_agreed = Some(ok);
        Ok(ok)
    }

    /// Compute all `world` K slices of a sublayer's mHC mix + Sinkhorn locally on `rows` owned stream rows -> `out` (`[rows][24]` f32).
    /// Bit-identical to the K-sliced `mix_partial` + `reduce` + `mix_finish` (`p0 + p1` at TP2, `(p0 + p2) + (p1 + p3)` at TP4,
    /// `((p0 + p2) + (p4 + p6)) + ((p1 + p3) + (p5 + p7))` at TP8, canonical `0..W` sequential sum at TP3 / TP6).
    pub(super) fn mix_local_into(&mut self, l: usize, attn_side: bool, x: u64, rows: usize, out: u64) -> Result<()> {
        let halves = self.world;
        ensure!(matches!(halves, 2 | 3 | 4 | 6 | 8), "mix_local_into: world {halves}");
        let lay = self.layers[l].as_ref().unwrap();
        let mk = if attn_side { lay.mix_a } else { lay.mix_f };
        let k = self.cfg.hc * self.cfg.hidden;
        ensure!(mk.fn_base != 0 && mk.fn_ld == k, "layer {l}: mix_local_into needs whole hc_fn");
        let mut ks = [(0usize, 0usize); 8];
        let mut max_chunks = 0usize;
        let mut same_chunks = true;
        let mut k_sum = 0usize;
        for z in 0..halves {
            let (a, n) = splits::part((k / 128) as u64, halves, z, 0);
            let (k0, klen) = ((a * 128) as usize, (n * 128) as usize);
            ks[z] = (k0, klen);
            k_sum += klen;
            let ch = km::partial_chunks(klen);
            if z == 0 { max_chunks = ch; }
            else {
                if ch != max_chunks { same_chunks = false; }
                max_chunks = max_chunks.max(ch);
            }
        }
        ensure!(k_sum == k, "layer {l}: mix_local_into K split {k_sum} != {k}");
        let rg = km::pick_rg(rows);
        let ngroups = rows.div_ceil(rg);
        ensure!(halves * ngroups * 4 <= self.mix_cnt.bytes, "mix_local_into: {ngroups} row groups > counter buffer");
        let part = self.f.b.take(halves * max_chunks * rows * 32 * 4)?;
        let rider = self.f.b.take(halves * rows * MIX_STRIDE * 4)?;
        let mut h = [km::MixPartialArgs::default(); 8];
        for (z, a) in h.iter_mut().enumerate().take(halves) {
            let (k0, klen) = ks[z];
            let fnp = mk.fn_base + (4 * k0) as u64;
            *a = km::MixPartialArgs {
                x, x_ld: k as i64, fn_: fnp, fn_ld: mk.fn_ld as i64,
                part: part + (z * max_chunks * rows * 32 * 4) as u64, chunk_stride: (rows * 32) as i64, row_stride: 32,
                rank_part: rider + (z * rows * MIX_STRIDE * 4) as u64, rp_row_stride: MIX_STRIDE as i64,
                counters: self.mix_cnt.ptr() + (z * ngroups * 4) as u64, k0: k0 as i32, klen: klen as i32, rows: rows as i32, rg: rg as i32,
            };
        }
        let (sc, bc) = if attn_side { (C::HcAttnScale, C::HcAttnBase) } else { (C::HcFfnScale, C::HcFfnBase) };
        let w = &self.layers[l].as_ref().unwrap().w;
        let fa = km::MixFinishArgs {
            part: rider, slot_stride: (rows * MIX_STRIDE) as i64, row_stride: MIX_STRIDE as i64,
            scale: w.param(sc)?.ptr.0, base: w.param(bc)?.ptr.0, out, out_ld: 24, nslots: halves as i32,
            rows: rows as i32, inv_k: 1.0 / k as f32, eps_norm: self.cfg.rms_eps as f32, eps_hc: self.cfg.hc_eps as f32,
            sinkhorn_iters: self.cfg.sinkhorn_iters as i32,
        };
        let entry = if !same_chunks {
            for z in 0..halves {
                let ch = km::partial_chunks(ks[z].1);
                self.f.g.launch1("d41_mix_partial", (ch as u32, ngroups as u32, 1), (128, 1, 1), 0, &h[z])?;
            }
            "d41_mix_partial(per-rank)+d41_mix_finish"
        } else {
            match halves {
                2 => {
                    let h2 = [h[0], h[1]];
                    self.f.g.launch1("d41_mix_partial2", (max_chunks as u32, ngroups as u32, 2), (128, 1, 1), 0, &h2)?;
                    "d41_mix_partial2+d41_mix_finish"
                }
                4 => {
                    let h4 = [h[0], h[1], h[2], h[3]];
                    self.f.g.launch1("d41_mix_partial4", (max_chunks as u32, ngroups as u32, 4), (128, 1, 1), 0, &h4)?;
                    "d41_mix_partial4+d41_mix_finish"
                }
                _ => {
                    self.f.g.launch1("d41_mix_partial8", (max_chunks as u32, ngroups as u32, halves as u32), (128, 1, 1), 0, &h)?;
                    "d41_mix_partial8+d41_mix_finish"
                }
            }
        };
        self.f.g.launch1("d41_mix_finish", ((rows as u32).div_ceil(8), 1, 1), (128, 1, 1), 0, &fa)?;
        receipts::kernel("tp.sp.mix_local", entry, rows);
        Ok(())
    }

    pub(super) fn hc_pre_norm_into(&mut self, x: u64, pre: u64, pre_stride: usize, norm: &crate::dsv41::weights::DevTensor, rows: usize, y: u64) -> Result<()> {
        let d = self.cfg.hidden;
        let a = km::HcPreNormArgs {
            x, x_ld: (4 * d) as i64, pre, pre_stride: pre_stride as i64, w: norm.ptr.0, y_norm: y, yn_ld: d as i64, y_raw: 0, yr_ld: 0,
            w_fmt: fmt_of(norm)?, dim: d as i32, rows: rows as i32, eps: self.cfg.rms_eps as f32, inv_dim: 1.0 / d as f32,
        };
        self.f.g.launch1("d41_hc_pre_norm", (rows as u32, 1, 1), (256, 1, 1), 0, &a)?;
        receipts::kernel("mhc.pre_norm", "d41_hc_pre_norm", rows);
        Ok(())
    }

    pub(super) fn hc_post_into(&mut self, xs: u64, resid: u64, rec: u64, rows: usize, out: u64) -> Result<()> {
        let d = self.cfg.hidden;
        let a = km::HcPostArgs {
            xs, xs_ld: d as i64, resid, r_ld: (4 * d) as i64, post: rec + 16, post_stride: 24, comb: rec + 32, comb_stride: 24,
            out, o_ld: (4 * d) as i64, dim: d as i32, rows: rows as i32,
        };
        self.f.g.launch1("d41_hc_post", ((d as u32).div_ceil(1024), rows as u32, 1), (256, 1, 1), 0, &a)?;
        receipts::kernel("mhc.post", "d41_hc_post", rows);
        Ok(())
    }

    pub(super) fn engram_gate_into(&mut self, el: usize, h: u64, kvb: u64, rows: usize, out: u64) -> Result<()> {
        let d = self.cfg.hidden;
        let kv_out = (self.cfg.hc + 1) * d;
        let lw = &self.layers[el].as_ref().unwrap().w;
        let (wq, wk) = (lw.param(C::EngramQ)?, lw.param(C::EngramK)?);
        let a = km::EngramGateArgs {
            h, h_ld: (4 * d) as i64, kv: kvb, kv_ld: kv_out as i64, wq: wq.ptr.0, wk: wk.ptr.0, gate_on: 0, out, o_ld: (4 * d) as i64,
            wq_fmt: fmt_of(wq)?, wk_fmt: fmt_of(wk)?, dim: d as i32, rows: rows as i32, eps: self.cfg.rms_eps as f32, inv_dim: 1.0 / d as f32,
            inv_sqrt_dim: (d as f64).powf(-0.5) as f32, clamp_min: 1e-6,
        };
        self.f.g.launch1("d41_engram_gate", (4, rows as u32, 1), (256, 1, 1), 0, &a)?;
        receipts::kernel("engram.gate", "d41_engram_gate", rows);
        Ok(())
    }

    pub(super) fn q_lora_into(&mut self, l: usize, ain: u64, rows: usize, y: u64) -> Result<()> {
        let (d, ql) = (self.cfg.hidden, self.cfg.q_lora_rank);
        let qa = self.f.b.take(rows * ql * 2)?;
        let lw = &self.layers[l].as_ref().unwrap().w;
        self.f.lin_rows("attn.wq_a", lw, C::AttnWqA, None, ain, d, d, rows, qa, ql, false)?;
        let lw = &self.layers[l].as_ref().unwrap().w;
        let w = lw.param(C::AttnQNorm)?;
        let a = km::RmsNormArgs {
            x: qa, x_ld: ql as i64, w: w.ptr.0, y, y_ld: ql as i64, w_fmt: fmt_of(w)?, dim: ql as i32, rows: rows as i32,
            eps: self.cfg.rms_eps as f32, inv_dim: 1.0 / ql as f32,
        };
        self.f.g.launch1("d41_rmsnorm", (rows as u32, 1, 1), (256, 1, 1), 0, &a)?;
        Ok(())
    }

    pub(super) fn router_vl_into(&mut self, l: usize, x: u64, rows: usize, image_mask: u64, out_i: u64, out_w: u64) -> Result<()> {
        let (e, k) = self.cfg.moe_of(l);
        let d = self.cfg.hidden;
        let w = &self.layers[l].as_ref().unwrap().w;
        let gw = w.weight_tensor(C::GateWeight)?;
        let bias = w.param(C::GateBias)?;
        let bias_fmt = fmt_of(bias)?;
        let bias_vl = if image_mask != 0 {
            let bvl = w.param(C::GateBiasVl)?;
            ensure!(fmt_of(bvl)? == bias_fmt, "layer {l}: GateBiasVl dtype {} != GateBias dtype {}", bvl.dtype, bias.dtype);
            bvl.ptr.0
        } else {
            0
        };
        let sc = self.f.b.take(rows * e * 4)?;
        let mut r0 = 0usize;
        if rows > km::ROUTER_ROWS_MAX && !km::router_tf32_on() && crate::dsv41::gpu::pf_on("dsv41-pf-router-wide") {
            let a = km::RouterScoresArgs { x, x_ld: d as i64, w: gw.ptr.0, w_ld: d as i64, scores: sc, s_ld: e as i64, w_fmt: fmt_of(gw)?, dim: d as i32, e: e as i32, rows: rows as i32 };
            let (name, (gx, gy), block) = km::router_wide_launch(&a, rows, km::WIDE);
            self.f.g.launch1(name, (gx, gy, 1), (block, 1, 1), 0, &a)?;
            receipts::kernel("ffn.router_scores (PP1 wide)", name, rows);
            crate::dsv41::gpu::PF_ROUTER_WIDE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            r0 = rows;
        }
        while r0 < rows {
            let n = (rows - r0).min(km::ROUTER_ROWS_MAX);
            let a = km::RouterScoresArgs {
                x: x + (r0 * d * 2) as u64, x_ld: d as i64, w: gw.ptr.0, w_ld: d as i64, scores: sc + (r0 * e * 4) as u64, s_ld: e as i64,
                w_fmt: fmt_of(gw)?, dim: d as i32, e: e as i32, rows: n as i32,
            };
            let (name, grid, block) = km::router_launch(&a, n);
            self.f.g.launch1(name, (grid, 1, 1), (block, 1, 1), 0, &a)?;
            receipts::kernel("ffn.router_scores", name, n);
            r0 += n;
        }
        let a = km::RouterSelectArgs {
            scores: sc, s_ld: e as i64, bias: bias.ptr.0, bias_vl, image_mask, out_w, out_i, o_ld: k as i64, bias_fmt,
            e: e as i32, topk: k as i32, rows: rows as i32, route_scale: self.cfg.routed_scale as f32, temp: 1.0, norm_topk: self.cfg.norm_topk_prob as i32,
        };
        self.f.g.launch1("d41_router_select", (rows as u32, 1, 1), (128, 1, 1), 0, &a)?;
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::Ranges;
    use crate::dsv41::config::{test_config_json, Dsv41Config};

    /// The forward's per-rank ranges tile every axis once and agree with the shipped placement's own rule
    /// (`load::splits::part`): heads <-> o_groups (8 heads per group), the hc_fn K slices, the Engram heads.
    #[test]
    fn dsv41_tp_ranges_tile_and_match_the_placement() {
        let c = Dsv41Config::from_value(&test_config_json(), None).unwrap();
        for world in [1usize, 2, 3, 4, 6, 8] {
            let (mut h, mut g, mut k, mut e) = (0, 0, 0, 0);
            for r in 0..world {
                let rg = Ranges::of(&c, world, r);
                assert_eq!((rg.h0, rg.g0, rg.k0, rg.eh0), (h, g, k, e), "world {world} rank {r}: ranges are contiguous");
                assert_eq!(rg.nh, rg.ng * c.heads / c.o_groups, "world {world} rank {r}: the rank's heads are its groups' heads");
                assert_eq!(rg.h0, rg.g0 * c.heads / c.o_groups);
                assert!(rg.klen % 128 == 0 && rg.klen > 0);
                h += rg.nh; g += rg.ng; k += rg.klen; e += rg.neh;
            }
            assert_eq!((h, g, k, e), (c.heads, c.o_groups, c.hc * c.hidden, c.engram_heads), "world {world}: every axis covered once");
        }
    }

    /// A graph-mode call's rows sit only in the window ring: the per-lane cap keeps every row's write out of the slots
    /// row 0 still reads, and still covers a verify (DSpark block + 1 rows).
    #[test]
    fn dsv41_tp_graph_segments_fit_the_window_ring() {
        let c = Dsv41Config::from_value(&test_config_json(), None).unwrap();
        let n = super::graph_seg_max(&c);
        assert!(n + c.window <= c.swa_slots() + 1, "{n} rows in a {}-slot ring overwrite row 0's window of {}", c.swa_slots(), c.window);
        assert!(n >= c.dspark_block + 1, "a verify of {} rows must stay graphable", c.dspark_block + 1);
    }
}
