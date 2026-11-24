//! S3: the GPU model of the looped Inkling-2 text decoder (TP1, stock pack), built from the K1 / K2 kernels.
//!
//! Op order is cpu.rs `CpuModel::forward` / `layer` (the oracle), visit for visit: 6 pre-loop layers, the loop core
//! (physical 6..18) three times with cross-pass messages and MessageGates on passes 1 and 2, 6 post-loop layers, the
//! output AttnRes, the final norm, the LM head (softcap 30), the sampler.
//!
//! Rows. One call runs `segs` = (lane, p0, m) segments whose rows are concatenated (a prefill chunk is one segment of
//! m <= 2,048 rows; a decode step is one 1-row segment per active lane). Every row-wise op (projections, AttnRes,
//! MessageGate, norms, MoE) runs once over all rows (each row's result is independent of the others: the K1 / K2
//! batch-invariance gates); only the per-sequence ops (attention prep / attention, the four short convs with their
//! rings) loop over the segments.
//!
//! State per lane (`Lane`): one committed position, and per logical slot (48) the f16 K / V cache (sliding slots: a
//! 4,096-row ring, full slots: linear up to the served context) and the four 32-row conv-input rings (K1 design:
//! "commit = pos += accepted", no snapshots).
//!
//! Hidden-state buffers (the residual stream, AttnRes block states, the loop's saved / previous-pass states) live in a
//! small pool of [rows][2304] fp32 buffers addressed by index, so a block boundary or a loop reset is an index move,
//! never a copy.
//!
//! Attention kernels: prefill chunks (m > 16) use the tensor-core `ink_attn_tc` (2.6x faster per 2,048-row chunk);
//! decode uses `decode_tc` (default false = the CUDA-core `ink_attn`, 2x faster at M = 1, see K1 report). The
//! future verify must use the same kernel as decode (batch invariance holds within a kernel, M = 1..16).

pub mod prefix;
pub mod probe;
pub mod probe_s7;
pub mod serve;

use super::config::Inkling2Config;
use super::gpu::head::{self, HeadDev, HeadScratch};
use super::gpu::k1::{bf16_bits, AttnArgs, PrepArgs, ProjW, DREL, HD, LANES_MAXR, LANE_SLOT_U64, NH, RING};
use super::gpu::moe::{self, MoeLayerDev, MoeScratch};
use super::gpu::{dp, Ink2Kernels, KArgs};
use super::tp::kern as tpk;
use super::tp::shard::{self, Rank};
use super::tp::{Op, TpRank};
use super::pack::{Inkling2Pack, P};
use anyhow::{bail, ensure, Context, Result};
use cudarc::driver::{sys, CudaDevice, CudaSlice};
use std::path::Path;
use std::sync::Arc;

pub const D: usize = 2304;
/// Rows per prefill chunk (the plan's 2,048) = the work-buffer capacity.
pub const CHUNK: usize = 2048;
/// Sliding-slot K / V ring rows (>= window 1024 + CHUNK - 1, a power of two).
pub const SWA_CAP: usize = 4096;
/// Sampler CTAs per row.
const SMP_G: usize = 128;
/// Largest number of rows whose logits one call produces (S7: a 4-lane x 12-row verify).
pub const MAX_OUT: usize = LANES_MAXR;
const POOL: usize = 12;

struct GateW { down: ProjW, wf: CudaSlice<u16>, wg: CudaSlice<u16> }

/// DeepSpeed-Ulysses (Jacobs et al., arXiv:2309.14509) full-head projection weights held at TP3/TP4/TP6/TP8
/// so SP prefill runs 1-RTT dual-rail All-to-All instead of attention AG + FP32 RS (~1.5 GB total VRAM).
struct UlyssesW {
    proj_full: ProjW,
    wo_full: ProjW,
    oq: usize, og: usize, or_: usize, os: usize, ok: usize, okd: i32, ov: usize, ovd: i32,
    t1: usize, t2: usize,
}

/// Scratch buffers for DeepSpeed-Ulysses (arXiv:2309.14509) SP prefill All-to-All (allocated at TP3/TP4/TP6/TP8).
struct UlyssesWork {
    y_full: CudaSlice<f32>,
    a2a_src: CudaSlice<f32>,
    a2a_dst: CudaSlice<f32>,
    att_f32: CudaSlice<f32>,
    att_full: CudaSlice<u16>,
    hds: tpk::InkUlyHeads,
    max_nq: usize,
    max_nkv: usize,
}

struct LayerW {
    sliding: bool,
    proj: ProjW,
    /// q, gate, r, sink, k, kd(-1), v, vd(-1) column offsets in Y
    oq: usize, og: usize, or_: usize, os: usize, ok: usize, okd: i32, ov: usize, ovd: i32,
    t1: usize, t2: usize,
    wo: ProjW,
    ulysses: Option<UlyssesW>,
    cw_k: CudaSlice<f32>, cw_v: CudaSlice<f32>,
    attn_conv: CudaSlice<f32>, mlp_conv: CudaSlice<f32>,
    q_scale: CudaSlice<f32>, r_scale: CudaSlice<f32>, rel_proj: CudaSlice<f32>,
    /// sliding: [attn_norm]; full: [q, k, v]_input_norm
    norms: Vec<CudaSlice<f32>>,
    mlp_norm: CudaSlice<f32>,
    gates: Option<Vec<GateW>>,
    moe: MoeLayerDev,
    ext: usize,
}

pub struct GpuModel {
    pub cfg: Inkling2Config,
    pub dev: Arc<CudaDevice>,
    pub k: &'static Ink2Kernels,
    layers: Vec<LayerW>,
    embed: CudaSlice<u16>,
    embed_norm: CudaSlice<f32>,
    attn_q: Vec<CudaSlice<f32>>,
    mlp_q: Vec<CudaSlice<f32>>,
    out_q: CudaSlice<f32>,
    msg_q: Vec<CudaSlice<f32>>,
    norm: CudaSlice<f32>,
    head: HeadDev,
    ldy: usize,
    ldy_full: usize,
    /// decode attention kernel: tensor-core `ink_attn_tc` (true) or CUDA-core `ink_attn` (false)
    pub decode_tc: bool,
    /// S7 opt-in `--inkling2-attn-single-pass`: prefill attention with Q / P rounded once to f16 (changes outputs)
    pub attn_single_pass: bool,
    /// Q1: experts on the pack-declared NVFP4 / MXFP8 activations (`--inkling2-expert-act native`)
    pub expert_native: bool,
    /// S-INK2-TP2: what this rank holds (TP1: everything) and its transport / head control (None at TP1)
    pub rank: Rank,
    pub tp: Option<TpRank>,
    /// the whole unpadded vocabulary (the logits rows `logits_host` returns)
    vocab_full: usize,
    next_lane: std::cell::Cell<usize>,
}

/// One sequence's device state.
pub struct Lane {
    /// creation index (the head and the nodes create lanes in the same order: the id names a lane on every rank)
    pub id: usize,
    pub pos: usize,
    pub max_ctx: usize,
    kc: Vec<CudaSlice<u16>>,
    vc: Vec<CudaSlice<u16>>,
    cap: Vec<usize>,
    /// per slot: k-in, v-in, attn-out, mlp-out conv input rings [32][C]
    ring: Vec<[CudaSlice<f32>; 4]>,
}

/// A row segment of one call.
#[derive(Clone, Copy, Debug)]
pub struct Seg { pub lane: usize, pub p0: usize, pub m: usize }

/// How the rows of one call map to sequences.
///   * `Segs`: host-side segments (prefill chunks, the per-lane decode loop of `decode_tc`); positions by value.
///   * `Lanes(n)`: a decode step of `n` rows, row r = one token of lane `rinfo.lane[r]` at `rinfo.pos[r]`, read on
///     the device (85_lanes.cuh): no host value in any launch depends on a position, so the step is graph-able.
/// `Lanes(n, runs)`: `runs` = some lane carries more than one row (a verify): the multi-row attention is launched.
enum Rows<'a> { Segs(&'a [&'a Lane], &'a [Seg]), Lanes(usize, bool) }

use std::sync::atomic::{AtomicBool, Ordering};

/// U7a opt-in `--inkling2-ulysses-wire fp32|bf16` (default `fp32` for strict TP1 bit-identity).
static ULYSSES_WIRE_BF16: AtomicBool = AtomicBool::new(false);

pub fn ulysses_wire_bf16() -> bool {
    ULYSSES_WIRE_BF16.load(Ordering::Relaxed)
}

pub fn set_ulysses_wire_from_args(args: &[String]) -> Result<bool> {
    let v = crate::arg_value(args, "--inkling2-ulysses-wire");
    let bf16 = match v {
        None | Some("fp32") | Some("f32") | Some("off") => false,
        Some("bf16") | Some("on") => true,
        Some(o) => bail!("--inkling2-ulysses-wire {o}: expected fp32 (default, bit-exact) or bf16 (opt-in BF16 Q/K/V/att wire)"),
    };
    ULYSSES_WIRE_BF16.store(bf16, Ordering::Relaxed);
    if bf16 {
        crate::boot_report::notice("--inkling2-ulysses-wire bf16: Ulysses All-to-All ships Q/K/V and ungated attn output in BF16 (r/s/delta kept FP32); cuts Ulysses wire by ~47% vs FP32, deterministic cross-rank, differs slightly from TP1 FP32");
    }
    Ok(bf16)
}

/// DeepSpeed-Ulysses (arXiv:2309.14509) per-rank sequence block descriptor.
struct SpPlan {
    rank: usize,
    world: usize,
    blocks: Vec<(usize, usize)>,
    halo_blocks: Vec<(usize, usize)>,
    my_r0: usize,
    my_r1: usize,
    my_m: usize,
    off_f32: u64,
    off_bf16: u64,
}

/// Minimum rows for Inkling-2 sequence-parallel prefill (guarantees every rank owns >= 3 rows for conv halos at W <= 8
/// when `blocks.iter().all(|&(a, b)| b - a >= 3)` also holds).
const SP_MIN_ROWS: usize = 64;

impl SpPlan {
    fn for_chunk(rank: usize, world: usize, m: usize) -> Option<SpPlan> {
        if m < SP_MIN_ROWS || !matches!(world, 2 | 3 | 4 | 6 | 8) { return None; }
        let blocks: Vec<(usize, usize)> = (0..world).map(|r| crate::tp_xport::row_block(m, r, world)).collect();
        if !blocks.iter().all(|&(a, b)| b.saturating_sub(a) >= 3) { return None; }
        let (my_r0, my_r1) = blocks[rank];
        let halo_blocks: Vec<(usize, usize)> = (0..world).map(|r| (r * 3, (r + 1) * 3)).collect();
        Some(SpPlan {
            rank,
            world,
            blocks,
            halo_blocks,
            my_r0,
            my_r1,
            my_m: my_r1 - my_r0,
            off_f32: (my_r0 * D * 4) as u64,
            off_bf16: (my_r0 * D * 2) as u64,
        })
    }

    fn new(tp: &TpRank, rows: &Rows, m: usize) -> Option<SpPlan> {
        if !tp.x.sp_ready() { return None; }
        match rows {
            Rows::Segs(_, segs) if segs.len() == 1 => Self::for_chunk(tp.x.rank, tp.x.world, m),
            _ => None,
        }
    }
}

/// Decode-step CUDA graphs (one per row count) and their capture / launch stream (blocking: ordered against the
/// legacy NULL stream the rest of the engine uses).
struct Graphs { stream: sys::CUstream, exec: Vec<Option<(sys::CUgraph, sys::CUgraphExec)>> }
// owned by the one thread that drives the model (the serve worker / a probe); moved there once
unsafe impl Send for Graphs {}
impl Drop for Graphs {
    fn drop(&mut self) {
        unsafe {
            for (g, x) in self.exec.drain(..).flatten() { sys::cuGraphExecDestroy(x); sys::cuGraphDestroy(g); }
            if !self.stream.is_null() { sys::cuStreamDestroy_v2(self.stream); }
        }
    }
}

/// Work buffers (one set; calls are stream-ordered).
pub struct Work {
    pool: Vec<CudaSlice<f32>>,
    ar: Vec<CudaSlice<f32>>,
    mg: Vec<CudaSlice<f32>>,
    msg: Vec<CudaSlice<f32>>,
    xb: Vec<CudaSlice<u16>>,
    y: CudaSlice<f32>,
    att: CudaSlice<u16>,
    ao: CudaSlice<f32>,
    mn: CudaSlice<f32>,
    mo: CudaSlice<f32>,
    qf: CudaSlice<f32>,
    rf: CudaSlice<f32>,
    pm: CudaSlice<f32>,
    pl: CudaSlice<f32>,
    pacc: CudaSlice<f32>,
    cat: CudaSlice<u16>,
    z: CudaSlice<f32>,
    ids: CudaSlice<i32>,
    fin: CudaSlice<f32>,
    finn: CudaSlice<f32>,
    pub logits: CudaSlice<f32>,
    hs: HeadScratch,
    moe: MoeScratch,
    smp_max: CudaSlice<f32>,
    smp_idx: CudaSlice<i32>,
    smp_seed: CudaSlice<u64>,
    smp_it: CudaSlice<f32>,
    pub smp_out: CudaSlice<i32>,
    /// [lanes][slots][LANE_SLOT_U64] device table of every lane's caches / rings (`bind_lanes`)
    lt: Option<CudaSlice<u64>>,
    /// [4][LANES_MAXR] i32: ids, positions, lane index, index inside the lane's run of each row (uploaded per step)
    rinfo: CudaSlice<i32>,
    graphs: Graphs,
    /// S6 drafter taps (None = off): bf16 [rows][4][D], the residual ENTERING visits `tap_visits` (slots 0..3) and the
    /// final-norm output (slot 3, when tap_visits[3] == visits()); see src/inkling2/mtp.
    pub taps: Option<CudaSlice<u16>>,
    pub tap_visits: [usize; 4],
    /// S5 image splice for the next `prefill`: (first prompt row, rows, device address of [rows][D] f32 features)
    pub img: Vec<(usize, usize, u64)>,
    /// the part of `img` inside the current chunk: (chunk row, device address, rows); `forward` consumes it
    img_chunk: Vec<(usize, u64, usize)>,
    /// Q1: the native expert path's scratch (Some iff `expert_native`)
    actq: Option<super::actq::ActqScratch>,
    /// DeepSpeed-Ulysses (arXiv:2309.14509) SP prefill attention scratch (Some at TP3/TP4/TP6/TP8)
    ulysses: Option<UlyssesWork>,
    /// the last forward's block / hidden pool indices (for `logits_rows`)
    last: (Vec<usize>, usize),
}

impl Work {
    /// S5: the image rows of prompt rows [i, i + m) (the next `forward` chunk) from `img` = (first prompt row, rows,
    /// device address of [rows][D] f32 features); `forward` consumes it.
    pub fn set_img_chunk(&mut self, img: &[(usize, usize, u64)], i: usize, m: usize) {
        self.img_chunk.clear();
        for &(s0, n, src) in img {
            let (a, b) = (s0.max(i), (s0 + n).min(i + m));
            if a < b { self.img_chunk.push((a - i, src + ((a - s0) * D * 4) as u64, b - a)); }
        }
    }
}

/// Static conv taps `[C][1][4]` (tap 3 = current, D3) -> lag-major `[4][C]` (lag 0 = current).
fn lag_major(wc: &[f32], c: usize) -> Vec<f32> {
    let mut o = vec![0f32; 4 * c];
    for ch in 0..c { for lag in 0..4 { o[lag * c + ch] = wc[ch * 4 + 3 - lag]; } }
    o
}

impl GpuModel {
    pub fn load(dir: &Path, dev: &Arc<CudaDevice>) -> Result<GpuModel> { Self::load_rank(dir, dev, None) }

    /// S-INK2-TP2: load the slice of rank `rank` (None = TP1, the whole model; tp/shard.rs has the split).
    pub fn load_rank(dir: &Path, dev: &Arc<CudaDevice>, rank: Option<Rank>) -> Result<GpuModel> {
        let t0 = std::time::Instant::now();
        let pack = Inkling2Pack::open(dir)?;
        let cfg = pack.cfg.clone();
        let rk = rank.unwrap_or_else(|| Rank::solo(cfg.inter, cfg.unpadded_vocab));
        let solo = rk.is_solo();
        let (q0, q1, kv0, kv1) = (rk.h0 * HD, (rk.h0 + rk.nq) * HD, rk.g0 * HD, (rk.g0 + rk.nkv) * HD);
        // rows [a, b) of a [rows][D] projection weight (TP1: all)
        let sl = |v: Vec<f32>, a: usize, b: usize| -> Vec<f32> { if solo { v } else { shard::rows(&v, D, a, b) } };
        // rows [a, b) of a [rows][k] vector (the per-head clip scales)
        let sl_n = |v: Vec<f32>, k: usize, a: usize, b: usize| -> Vec<f32> { if solo { v } else { shard::rows(&v, k, a, b) } };
        ensure!(cfg.hidden == D && cfg.conv_k == 4, "inkling2 GPU model: hidden {} conv_k {} (built for 2304 / 4)", cfg.hidden, cfg.conv_k);
        let k = Ink2Kernels::global(dev)?;
        let native = super::actq::mode() == super::actq::ExpertAct::Native;
        let g = |n: &str| -> Result<Vec<f32>> { pack.plain(&format!("{P}{n}")) };
        let up = |v: &[f32]| -> Result<CudaSlice<f32>> { Ok(dev.htod_sync_copy(v)?) };
        let up16 = |v: &[f32]| -> Result<CudaSlice<u16>> { Ok(dev.htod_sync_copy(&v.iter().map(|&x| bf16_bits(x)).collect::<Vec<_>>())?) };
        let mut layers = Vec::with_capacity(cfg.layers);
        let mut ldy = 0;
        let mut ldy_full = 0;
        // Ulysses cuts wire bytes strictly for W >= 3 (at W=2, non-Ulysses AG x + RS ao is 6,912 B/tok vs FP32 Ulysses 8,664 B/tok).
        let want_ulysses = matches!(rk.world, 3 | 4 | 6 | 8);
        for l in 0..cfg.layers {
            let p = format!("layers.{l}.");
            let a = format!("{p}attn.");
            let sliding = cfg.is_sliding[l];
            let (r0, r1, d0, d1) = (rk.h0 * DREL, (rk.h0 + rk.nq) * DREL, rk.g0 * 4, (rk.g0 + rk.nkv) * 4);
            let wq_all = g(&format!("{a}wq_du.weight"))?;
            let wg_all = g(&format!("{a}gate_attn.weight"))?;
            let wr_all = g(&format!("{a}wr_du.weight"))?;
            let ws_all = g(&format!("{a}attn_sink_proj.weight"))?;
            let wk_all = g(&format!("{a}wk_dv.weight"))?;
            let wkd_all = if !sliding { Some(g(&format!("{a}k_sconv.weight_proj"))?) } else { None };
            let wv_all = g(&format!("{a}wv_dv.weight"))?;
            let wvd_all = if !sliding { Some(g(&format!("{a}v_sconv.weight_proj"))?) } else { None };
            let wo_full_h = g(&format!("{a}wo_ud.weight"))?;

            let ulysses = if want_ulysses {
                let mut fsegs: Vec<&[f32]> = vec![&wq_all, &wg_all, &wr_all, &ws_all, &wk_all];
                if let Some(ref kd) = wkd_all { fsegs.push(kd); }
                fsegs.push(&wv_all);
                if let Some(ref vd) = wvd_all { fsegs.push(vd); }
                let proj_full = ProjW::new(k, &fsegs, D)?;
                let fo = proj_full.offs.clone();
                let (okd_f, ov_f, ovd_f) = if sliding { (-1, fo[5], -1) } else { (fo[5] as i32, fo[6], fo[7] as i32) };
                let (t1_f, t2_f) = (fo[4] / 16, ov_f / 16);
                ldy_full = ldy_full.max(proj_full.n_pad);
                let wo_full = ProjW::new(k, &[&wo_full_h], NH * HD)?;
                Some(UlyssesW { proj_full, wo_full, oq: fo[0], og: fo[1], or_: fo[2], os: fo[3], ok: fo[4], okd: okd_f, ov: ov_f, ovd: ovd_f, t1: t1_f, t2: t2_f })
            } else { None };

            let mut segs: Vec<Vec<f32>> = vec![sl(wq_all, q0, q1), sl(wg_all, q0, q1),
                                               sl(wr_all, r0, r1),
                                               sl(ws_all, rk.h0, rk.h0 + rk.nq), sl(wk_all, kv0, kv1)];
            if let Some(kd) = wkd_all { segs.push(sl(kd, d0, d1)); }
            segs.push(sl(wv_all, kv0, kv1));
            if let Some(vd) = wvd_all { segs.push(sl(vd, d0, d1)); }
            let proj = ProjW::new(k, &segs.iter().map(|v| v.as_slice()).collect::<Vec<_>>(), D)?;
            let o = proj.offs.clone();
            let (okd, ov, ovd) = if sliding { (-1, o[5], -1) } else { (o[5] as i32, o[6], o[7] as i32) };
            let (t1, t2) = (o[4] / 16, ov / 16);
            ldy = ldy.max(proj.n_pad);
            let wo = if solo { ProjW::new(k, &[&wo_full_h], NH * HD)? } else { ProjW::new(k, &[&shard::cols(&wo_full_h, NH * HD, q0, q1)], rk.nq * HD)? };
            let conv_w = |s: &str| -> Result<Vec<f32>> {
                let w = if sliding { lag_major(&g(&format!("{a}{s}_sconv.weight"))?, 1024) } else { g(&format!("{a}{s}_sconv.static_w"))? };
                Ok(if solo { w } else { shard::cols(&w, 1024, kv0, kv1) })
            };
            let norms = if sliding { vec![up(&g(&format!("{p}attn_norm.weight"))?)?] } else {
                vec![up(&g(&format!("{p}q_input_norm.weight"))?)?, up(&g(&format!("{p}k_input_norm.weight"))?)?, up(&g(&format!("{p}v_input_norm.weight"))?)?]
            };
            let gates = if cfg.is_loop(l) {
                let mut v = Vec::new();
                for s in ["attn", "k", "v", "mlp"] {
                    let c = format!("{p}controllers.{s}.");
                    v.push(GateW { down: ProjW::new(k, &[&g(&format!("{c}down_weight"))?], 2 * D)?,
                                   wf: up16(&g(&format!("{c}forget_weight"))?)?, wg: up16(&g(&format!("{c}gain_weight"))?)? });
                }
                Some(v)
            } else { None };
            layers.push(LayerW {
                sliding, oq: o[0], og: o[1], or_: o[2], os: o[3], ok: o[4], okd, ov, ovd, t1, t2, proj, wo, ulysses,
                cw_k: up(&conv_w("k")?)?, cw_v: up(&conv_w("v")?)?,
                attn_conv: up(&lag_major(&g(&format!("{p}attn_sconv.weight"))?, D))?,
                mlp_conv: up(&lag_major(&g(&format!("{p}mlp_sconv.weight"))?, D))?,
                q_scale: up(&sl_n(g(&format!("{a}q_l2_clip.scale"))?, HD, rk.h0, rk.h0 + rk.nq))?,
                r_scale: up(&sl_n(g(&format!("{a}rel_l2_clip.scale"))?, DREL, rk.h0, rk.h0 + rk.nq))?,
                rel_proj: up(&g(&format!("{a}rel_logits_proj.proj"))?)?,
                norms, mlp_norm: up(&g(&format!("{p}mlp_norm.weight"))?)?, gates,
                moe: if native { super::actq::load_layer_range(dev, &pack, l, rk.c0, rk.c1)? } else { MoeLayerDev::load_range(dev, &pack, l, rk.c0, rk.c1)? }, ext: cfg.rel_ext(l),
            });
        }
        let attn_q = (0..cfg.layers - 1).map(|i| up(&g(&format!("attn_res.attn_queries.{i}"))?)).collect::<Result<Vec<_>>>()?;
        let mlp_q = (0..cfg.layers).map(|i| up(&g(&format!("attn_res.mlp_queries.{i}"))?)).collect::<Result<Vec<_>>>()?;
        let qm = g("middle_cycle_cross_pass_queries")?;
        let kvm = g("middle_cycle_cross_pass_kv_queries")?;
        let row = |m: &[f32], r: usize| m[r * D..(r + 1) * D].to_vec();
        // D8: [q, k, v, mlp] = [queries[2k], kv[2k], kv[2k+1], queries[2k+1]]
        let msg_q = (0..cfg.loop_len()).map(|kk| up(&[row(&qm, 2 * kk), row(&kvm, 2 * kk), row(&kvm, 2 * kk + 1), row(&qm, 2 * kk + 1)].concat()))
            .collect::<Result<Vec<_>>>()?;
        let eb = pack.st.raw(&format!("{P}embed.weight"))?;
        let embed = dev.htod_sync_copy(&eb.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect::<Vec<u16>>())?;
        let head = HeadDev::load(dev, &pack, rk.v0, rk.vn)?;
        let vocab_full = cfg.unpadded_vocab;
        let m = GpuModel {
            embed, embed_norm: up(&g("embed_norm.weight")?)?, attn_q, mlp_q, out_q: up(&g("attn_res.output_query")?)?, msg_q,
            norm: up(&g("norm.weight")?)?, head, ldy, ldy_full, layers, decode_tc: false, attn_single_pass: false, expert_native: native, k, dev: dev.clone(), cfg,
            rank: rk, tp: None, vocab_full, next_lane: std::cell::Cell::new(0),
        };
        dev.synchronize()?;
        eprintln!("INK2_MODEL loaded {} layers ({} visits) in {:.1}s", m.cfg.layers, m.cfg.visits(), t0.elapsed().as_secs_f64());
        Ok(m)
    }

    /// S-INK2-TP2: the head announces `op` to every node before it launches it (nodes replay the head's calls in
    /// order: the control stream is the only thing that decides what a node runs). No-op at TP1 and on a node.
    fn mirror(&self, op: impl FnOnce() -> Op) -> Result<()> {
        match &self.tp { Some(t) => t.announce(op()), None => Ok(()) }
    }
    /// Public control-plane mirror for sibling modules (`mtp::serve` vocab-parallel drafting & accept).
    pub fn mirror_op(&self, op: Op) -> Result<()> {
        match &self.tp { Some(t) => t.announce(op), None => Ok(()) }
    }
    /// Sum the partials of `n` floats at `p` over the ranks, in place (no-op at TP1).
    fn reduce(&self, p: u64, n: usize) -> Result<()> {
        match &self.tp { Some(t) => t.x.all_reduce(&self.dev, p, n), None => Ok(()) }
    }

    /// Logits row length of `logits_host` (the whole unpadded vocabulary on every rank count).
    pub fn vocab(&self) -> usize { self.vocab_full }
    /// S6 / Drop 1 drafter access: (embed table bf16, embed_norm weight, this rank's LM head slice `[v0..v0+vn)`).
    pub fn shared_for_mtp(&self) -> (u64, u64, &HeadDev) { (dp(&self.embed), dp(&self.embed_norm), &self.head) }

    /// A lane able to hold `max_ctx` positions (full slots cap = next power of two >= max_ctx).
    pub fn new_lane(&self, max_ctx: usize) -> Result<Lane> {
        let c = &self.cfg;
        let n = c.visits();
        let dev = &self.dev;
        let full_cap = max_ctx.max(16).next_power_of_two();
        let (mut kc, mut vc, mut cap, mut ring) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for slot in 0..n {
            let l = if slot < c.layers { slot } else { c.loop_start + (slot - c.layers) % c.loop_len() };
            let cp = if c.is_sliding[l] { SWA_CAP.min(full_cap.max(SWA_CAP)) } else { full_cap };
            let kvw = self.rank.kvw();
            let mut a = dev.alloc_zeros::<u16>(cp * kvw)?;
            let mut b = dev.alloc_zeros::<u16>(cp * kvw)?;
            dev.memset_zeros(&mut a)?;
            dev.memset_zeros(&mut b)?;
            kc.push(a);
            vc.push(b);
            cap.push(cp);
            let mut r: Vec<CudaSlice<f32>> = Vec::new();
            for cc in [self.rank.kvw(), self.rank.kvw(), D, D] {
                let mut x = dev.alloc_zeros::<f32>(RING * cc)?;
                dev.memset_zeros(&mut x)?;
                r.push(x);
            }
            let r: [CudaSlice<f32>; 4] = r.try_into().map_err(|_| anyhow::anyhow!("ring"))?;
            ring.push(r);
        }
        dev.synchronize()?;
        let id = self.next_lane.get();
        self.next_lane.set(id + 1);
        Ok(Lane { id, pos: 0, max_ctx, kc, vc, cap, ring })
    }

    pub fn new_work(&self) -> Result<Work> {
        let dev = &self.dev;
        let f = |n: usize| -> Result<CudaSlice<f32>> { let mut x = dev.alloc_zeros::<f32>(n)?; dev.memset_zeros(&mut x)?; Ok(x) };
        let h = |n: usize| -> Result<CudaSlice<u16>> { let mut x = dev.alloc_zeros::<u16>(n)?; dev.memset_zeros(&mut x)?; Ok(x) };
        let cm = CHUNK;
        let maxsplit = 64;
        let w = Work {
            pool: (0..POOL).map(|_| f(cm * D)).collect::<Result<_>>()?,
            ar: (0..3).map(|_| f(cm * D)).collect::<Result<_>>()?,
            mg: (0..3).map(|_| f(cm * D)).collect::<Result<_>>()?,
            msg: (0..4).map(|_| f(cm * D)).collect::<Result<_>>()?,
            xb: (0..3).map(|_| h(cm * D)).collect::<Result<_>>()?,
            y: f(cm * self.ldy)?,
            att: h(cm * NH * HD)?,
            ao: f(cm * D)?,
            mn: f(cm * D)?,
            mo: f(cm * D)?,
            qf: f(cm * NH * HD)?,
            rf: f(cm * NH * DREL)?,
            pm: f(LANES_MAXR * NH * maxsplit)?,
            pl: f(LANES_MAXR * NH * maxsplit)?,
            pacc: f(LANES_MAXR * NH * maxsplit * HD)?,
            cat: h(cm * 2 * D)?,
            z: f((cm * 64).max(3 * LANES_MAXR * 64 * super::gpu::k1::MSG_ZS))?,
            ids: dev.htod_sync_copy(&vec![0i32; cm])?,
            fin: f(MAX_OUT * D)?,
            finn: f(MAX_OUT * D)?,
            logits: f(MAX_OUT * self.head.n)?,
            hs: HeadScratch::new(dev, &self.head)?,
            moe: MoeScratch::new(dev, cm)?,
            smp_max: f(MAX_OUT * SMP_G)?,
            smp_idx: dev.htod_sync_copy(&vec![0i32; MAX_OUT * SMP_G])?,
            smp_seed: dev.htod_sync_copy(&vec![0u64; MAX_OUT])?,
            smp_it: f(MAX_OUT)?,
            smp_out: dev.htod_sync_copy(&vec![0i32; MAX_OUT])?,
            lt: None,
            rinfo: dev.htod_sync_copy(&vec![0i32; 4 * LANES_MAXR])?,
            graphs: Graphs { stream: std::ptr::null_mut(), exec: Vec::new() },
            taps: None,
            tap_visits: [usize::MAX; 4],
            img: Vec::new(),
            img_chunk: Vec::new(),
            actq: if self.expert_native { Some(super::actq::ActqScratch::new(dev, cm, self.cfg.gate_softcap, self.cfg.up_softcap)?) } else { None },
            ulysses: if self.ldy_full > 0 && matches!(self.rank.world, 3 | 4 | 6 | 8) {
                let (hds, max_nq, max_nkv) = tpk::InkUlyHeads::for_world(self.rank.world, self.cfg.inter, self.vocab_full)?;
                let max_my_m = cm.div_ceil(self.rank.world);
                // U1 + arbitrary-W: gate_attn stays local; sized by max_nq/max_nkv across ranks (TP2/TP3/TP4/TP6/TP8).
                let y_stride_max = (max_nq * 145 + max_nkv * 264 + 3) & !3;
                Some(UlyssesWork {
                    y_full: f(max_my_m * self.ldy_full)?,
                    a2a_src: f(self.rank.world * max_my_m * y_stride_max)?,
                    a2a_dst: f(self.rank.world * max_my_m * y_stride_max)?,
                    att_f32: f(cm * self.rank.nq * HD)?,
                    att_full: h(max_my_m * NH * HD)?,
                    hds,
                    max_nq,
                    max_nkv,
                })
            } else { None },
            last: (Vec::new(), 0),
        };
        dev.synchronize()?;
        Ok(w)
    }

    /// Run the rows of `segs` (ids concatenated in segment order) through the model; the logits of the rows in
    /// `out_rows` (indices into the concatenated rows, <= 16) land in `ws.logits` ([i][vocab], softcapped).
    /// Lane positions are NOT advanced (the caller commits).
    pub fn forward(&self, ws: &mut Work, lanes: &[&Lane], segs: &[Seg], ids: &[u32], out_rows: &[usize]) -> Result<()> {
        let m: usize = segs.iter().map(|s| s.m).sum();
        ensure!(m == ids.len() && m >= 1 && m <= CHUNK, "inkling2 forward: {m} rows vs {} ids (cap {CHUNK})", ids.len());
        ensure!(out_rows.len() <= MAX_OUT && out_rows.iter().all(|&r| r < m), "out rows {out_rows:?}");
        for s in segs {
            let ln = lanes[s.lane];
            ensure!(s.p0 + s.m <= ln.max_ctx, "lane {}: positions {}..{} beyond its context {}", s.lane, s.p0, s.p0 + s.m, ln.max_ctx);
        }
        if self.tp.as_ref().is_some_and(|t| t.ctl.is_some()) {
            // the head: image rows of this chunk go with the op (the tower runs on the head only)
            let mut img = Vec::new();
            for &(r0, src, n) in &ws.img_chunk {
                let mut h = vec![0f32; n * D];
                let r = unsafe { sys::cuMemcpyDtoH_v2(h.as_mut_ptr() as *mut std::ffi::c_void, src, n * D * 4) };
                ensure!(r == sys::CUresult::CUDA_SUCCESS, "image rows to host: {r:?}");
                img.push((r0, h));
            }
            self.mirror(|| Op::Forward { segs: segs.iter().map(|s| (lanes[s.lane].id, s.p0, s.m)).collect(), ids: ids.to_vec(), out: out_rows.to_vec(), img })?;
        }
        let idv: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
        self.dev.htod_sync_copy_into(&idv, &mut ws.ids.slice_mut(0..m))?;
        let idp = dp(&ws.ids);
        self.run(ws, &Rows::Segs(lanes, segs), m, idp, out_rows)
    }

    /// The forward body: launches only (no host <-> device copy, so a `Rows::Lanes` call can be captured).
    fn run(&self, ws: &mut Work, rows: &Rows, m: usize, ids: u64, out_rows: &[usize]) -> Result<()> {
        let c = &self.cfg;
        let k = self.k;
        let sp = self.tp.as_ref().and_then(|tp| SpPlan::new(tp, rows, m));
        let sp_ref = sp.as_ref();
        let mut live = [false; POOL];
        let alloc = |live: &mut [bool; POOL]| -> Result<usize> {
            let i = live.iter().position(|x| !x).context("inkling2: hidden-state pool exhausted")?;
            live[i] = true;
            Ok(i)
        };
        let mut h = alloc(&mut live)?;
        if let Some(sp) = sp_ref {
            k.embed_norm(ids + (sp.my_r0 * 4) as u64, dp(&self.embed), dp(&self.embed_norm), dp(&ws.pool[h]) + sp.off_f32, D, sp.my_m, c.eps)?;
            for (r0, src, n) in std::mem::take(&mut ws.img_chunk) {
                let lo = r0.max(sp.my_r0);
                let hi = (r0 + n).min(sp.my_r1);
                if hi > lo {
                    k.vis_rows_copy(dp(&ws.pool[h]) + (lo * D * 4) as u64, src + ((lo - r0) * D * 4) as u64, (hi - lo) * D)?;
                }
            }
        } else {
            k.embed_norm(ids, dp(&self.embed), dp(&self.embed_norm), dp(&ws.pool[h]), D, m, c.eps)?;
            // S5: image features replace the (normed) embeddings of the placeholder rows (the reference's masked_scatter)
            for (r0, src, n) in std::mem::take(&mut ws.img_chunk) {
                k.vis_rows_copy(dp(&ws.pool[h]) + (r0 * D * 4) as u64, src, n * D)?;
            }
        }
        let mut blocks: Vec<usize> = Vec::new();
        let relive = |live: &mut [bool; POOL], keep: &[usize]| { *live = [false; POOL]; for &i in keep { live[i] = true; } };
        // pre-loop
        for l in 0..c.loop_start {
            self.tap(ws, l, h, m, sp_ref)?;
            let hn = self.layer(ws, rows, l, 0, h, &blocks, false, m, sp_ref, &mut live, alloc)?;
            if c.is_boundary(l) { blocks.push(h); }
            h = hn;
            let mut keep = blocks.clone(); keep.push(h);
            relive(&mut live, &keep);
        }
        let (pre_h, pre_b) = (h, blocks.clone());
        let npre = pre_b.len();
        let mut prev: Vec<usize> = Vec::new(); // [prev_b..., prev_h]
        for pass in 0..c.passes {
            if pass > 0 {
                let mut pv: Vec<usize> = blocks[(npre + 1).min(blocks.len())..].to_vec();
                pv.push(h);
                prev = pv;
                h = pre_h;
                blocks = pre_b.clone();
                let mut keep = blocks.clone(); keep.push(h); keep.push(pre_h); keep.extend(&prev);
                relive(&mut live, &keep);
            }
            for (kk, l) in (c.loop_start..c.loop_end).enumerate() {
                if pass > 0 {
                    let (off, mr) = sp_ref.map_or((0u64, m), |s| (s.off_f32, s.my_m));
                    let vals: Vec<u64> = prev.iter().map(|&i| dp(&ws.pool[i]) + off).collect();
                    let outs: Vec<u64> = ws.msg.iter().map(dp).collect();
                    k.attnres(&vals, dp(&self.msg_q[kk]), 4, &outs, D, mr)?;
                }
                self.tap(ws, c.loop_start + pass * c.loop_len() + kk, h, m, sp_ref)?;
                let hn = self.layer(ws, rows, l, pass, h, &blocks, pass > 0, m, sp_ref, &mut live, alloc)?;
                if c.is_boundary(l) { blocks.push(h); }
                h = hn;
                let mut keep = blocks.clone(); keep.push(h); keep.push(pre_h); keep.extend(&prev);
                relive(&mut live, &keep);
            }
        }
        for l in c.loop_end..c.layers {
            self.tap(ws, l + (c.passes - 1) * c.loop_len(), h, m, sp_ref)?;
            let hn = self.layer(ws, rows, l, 0, h, &blocks, false, m, sp_ref, &mut live, alloc)?;
            if c.is_boundary(l) { blocks.push(h); }
            h = hn;
            let mut keep = blocks.clone(); keep.push(h);
            relive(&mut live, &keep);
        }
        // S6 drafter: the final-norm hidden of every row as tap slot 3
        if let Some(t) = ws.taps.as_ref() {
            if ws.tap_visits[3] == c.visits() {
                let (off, mr, toff) = sp_ref.map_or((0u64, m, 0u64), |s| (s.off_f32, s.my_m, (s.my_r0 * 4 * D * 2) as u64));
                let mut vals: Vec<u64> = blocks.iter().map(|&b| dp(&ws.pool[b]) + off).collect();
                vals.push(dp(&ws.pool[h]) + off);
                if sp_ref.is_none() && (mr <= 16 || (mr <= super::gpu::k1::VERIFY_MAX_ROWS && super::gpu::k1::verify_class())) {
                    k.attnres_cl_norm(&vals, dp(&self.out_q), 1, &[dp(&self.norm)], &[0], &[dp(&ws.mo)], D, mr, c.eps)?;
                } else {
                    k.attnres(&vals, dp(&self.out_q), 1, &[dp(&ws.ao)], D, mr)?;
                    k.rmsnorm(dp(&ws.ao), dp(&self.norm), 0, dp(&ws.mo), D, mr, c.eps)?;
                }
                crate::inkling2::mtp::gpu::tap_store(k, dp(&ws.mo), dp(t) + toff, D, 3, mr)?;
            }
            if let Some(sp) = sp_ref {
                let tp = self.tp.as_ref().unwrap();
                tp.x.all_gather_rows_multi(&self.dev, &[(dp(t), 4 * D * 2)], &sp.blocks)?;
            }
        }
        ws.last = (blocks.clone(), h);
        // output AttnRes + final norm on the requested rows, then the LM head
        if !out_rows.is_empty() {
            let n = out_rows.len();
            if let Some(sp) = sp_ref {
                let r = unsafe { sys::cuMemsetD32Async(dp(&ws.finn), 0, n * D, k.stream.get()) };
                ensure!(r == sys::CUresult::CUDA_SUCCESS, "inkling2 SP finn zero: {r:?}");
                for (i, &row) in out_rows.iter().enumerate() {
                    if row >= sp.my_r0 && row < sp.my_r1 {
                        let mut vals: Vec<u64> = blocks.iter().map(|&b| dp(&ws.pool[b]) + (row * D * 4) as u64).collect();
                        vals.push(dp(&ws.pool[h]) + (row * D * 4) as u64);
                        k.attnres_cl_norm(&vals, dp(&self.out_q), 1, &[dp(&self.norm)], &[0], &[dp(&ws.finn) + (i * D * 4) as u64], D, 1, c.eps)?;
                    }
                }
                self.reduce(dp(&ws.finn), n * D)?;
            } else {
                if out_rows.iter().enumerate().all(|(i, &r)| i == r) {
                    // Drop 3 (#4a): fused cluster AttnRes + final RMSNorm (zero ws.fin round-trip)
                    for r0 in (0..n).step_by(16) {
                        let mut vals: Vec<u64> = blocks.iter().map(|&b| dp(&ws.pool[b]) + (r0 * D * 4) as u64).collect();
                        vals.push(dp(&ws.pool[h]) + (r0 * D * 4) as u64);
                        k.attnres_cl_norm(&vals, dp(&self.out_q), 1, &[dp(&self.norm)], &[0], &[dp(&ws.finn) + (r0 * D * 4) as u64], D, (n - r0).min(16), c.eps)?;
                    }
                } else {
                    for (i, &r) in out_rows.iter().enumerate() {
                        let mut vals: Vec<u64> = blocks.iter().map(|&b| dp(&ws.pool[b]) + (r * D * 4) as u64).collect();
                        vals.push(dp(&ws.pool[h]) + (r * D * 4) as u64);
                        k.attnres_cl_norm(&vals, dp(&self.out_q), 1, &[dp(&self.norm)], &[0], &[dp(&ws.finn) + (i * D * 4) as u64], D, 1, c.eps)?;
                    }
                }
            }
            head::forward_rows(&self.dev, &self.head, &ws.hs, dp(&ws.finn), dp(&ws.logits), self.head.n, n)?;
        }
        Ok(())
    }

    /// S6 drafter tap: the residual entering logical visit `visit`, when it is one of `ws.tap_visits[..3]`.
    fn tap(&self, ws: &Work, visit: usize, h: usize, m: usize, sp: Option<&SpPlan>) -> Result<()> {
        if let (Some(t), Some(slot)) = (ws.taps.as_ref(), ws.tap_visits[..3].iter().position(|&v| v == visit)) {
            let (off, mr, toff) = sp.map_or((0u64, m, 0u64), |s| (s.off_f32, s.my_m, (s.my_r0 * 4 * D * 2) as u64));
            crate::inkling2::mtp::gpu::tap_store(self.k, dp(&ws.pool[h]) + off, dp(t) + toff, D, slot, mr)?;
        }
        Ok(())
    }

    /// One decoder-layer call (physical `l`, loop pass `pass`); returns the pool index of its output.
    #[allow(clippy::too_many_arguments)]
    fn layer(&self, ws: &mut Work, rows: &Rows, l: usize, pass: usize, h: usize, blocks: &[usize], msgs: bool,
             m: usize, sp: Option<&SpPlan>, live: &mut [bool; POOL], alloc: impl Fn(&mut [bool; POOL]) -> Result<usize>) -> Result<usize> {
        let c = &self.cfg;
        let k = self.k;
        let lw = &self.layers[l];
        let slot = c.cache_slot(l, pass);
        let hp = dp(&ws.pool[h]);
        let (off_f32, off_bf16, mr) = sp.map_or((0u64, 0u64, m), |s| (s.off_f32, s.off_bf16, s.my_m));
        let decode_fuse = sp.is_none() && (mr <= 16 || (mr <= super::gpu::k1::VERIFY_MAX_ROWS && super::gpu::k1::verify_class()));
        let three = msgs || !lw.sliding;
        let xb: Vec<u64> = ws.xb.iter().map(dp).collect();
        // --- attention inputs: AttnRes (1 query sliding, 3 full), MessageGates (loop passes), norms -> bf16
        if l > 0 && !msgs && decode_fuse {
            // Drop 3 (#4a): fused cluster AttnRes + RMSNorm -> bf16 xb directly (zero ws.ar round-trip)
            let mut vals: Vec<u64> = blocks.iter().map(|&b| dp(&ws.pool[b])).collect();
            vals.push(hp);
            if lw.sliding {
                k.attnres_cl_norm(&vals, dp(&self.attn_q[l - 1]), 1, &[dp(&lw.norms[0])], &[xb[0]], &[0], D, mr, c.eps)?;
            } else {
                k.attnres_cl_norm(
                    &vals, dp(&self.attn_q[l - 1]), 3,
                    &[dp(&lw.norms[0]), dp(&lw.norms[1]), dp(&lw.norms[2])],
                    &[xb[0], xb[1], xb[2]], &[0, 0, 0], D, mr, c.eps,
                )?;
            }
        } else {
            let mut xin: [u64; 3] = [hp + off_f32; 3];
            if l > 0 {
                let mut vals: Vec<u64> = blocks.iter().map(|&b| dp(&ws.pool[b]) + off_f32).collect();
                vals.push(hp + off_f32);
                let nq = if lw.sliding { 1 } else { 3 };
                let outs: Vec<u64> = ws.ar[..nq].iter().map(dp).collect();
                k.attnres(&vals, dp(&self.attn_q[l - 1]), nq, &outs, D, mr)?;
                xin = if nq == 1 { [outs[0]; 3] } else { [outs[0], outs[1], outs[2]] };
            }
            if msgs {
                let g = lw.gates.as_ref().context("messages on a non-loop layer")?;
                if matches!(rows, Rows::Lanes(_, _)) {
                    // S7: the three gates in one launch per stage (bit-identical)
                    k.msggate3([&g[0].down, &g[1].down, &g[2].down], [dp(&g[0].wf), dp(&g[1].wf), dp(&g[2].wf)], [dp(&g[0].wg), dp(&g[1].wg), dp(&g[2].wg)],
                               xin, [dp(&ws.msg[0]), dp(&ws.msg[1]), dp(&ws.msg[2])], dp(&ws.cat), dp(&ws.z),
                               [dp(&ws.mg[0]), dp(&ws.mg[1]), dp(&ws.mg[2])], D, mr, c.eps)?;
                } else {
                    for i in 0..3 {
                        k.msggate(&g[i].down, dp(&g[i].wf), dp(&g[i].wg), xin[i], dp(&ws.msg[i]), dp(&ws.cat), dp(&ws.z), dp(&ws.mg[i]), D, mr, c.eps)?;
                    }
                }
                xin = [dp(&ws.mg[0]), dp(&ws.mg[1]), dp(&ws.mg[2])];
            }
            if three {
                let nw = if lw.sliding {
                    [dp(&lw.norms[0]), dp(&lw.norms[0]), dp(&lw.norms[0])]
                } else {
                    [dp(&lw.norms[0]), dp(&lw.norms[1]), dp(&lw.norms[2])]
                };
                k.rmsnorm_x3(xin, nw, [xb[0] + off_bf16, xb[1] + off_bf16, xb[2] + off_bf16], [0, 0, 0], D, mr, c.eps)?;
            } else {
                k.rmsnorm(xin[0], dp(&lw.norms[0]), xb[0] + off_bf16, 0, D, mr, c.eps)?;
            }
        }
        let ldy = lw.proj.n_pad;
        let use_ulysses = sp.is_some()
            && lw.ulysses.is_some()
            && ws.ulysses.is_some()
            && self.tp.as_ref().is_some_and(|t| t.x.a2a_ready());
        if use_ulysses {
            // DeepSpeed-Ulysses (Jacobs et al., arXiv:2309.14509) + U1 localized gate_attn:
            // project `my_m` tokens across all heads locally and transpose `[my_m, H_all] -> [m, H_local]`
            // via 1-RTT All-to-All WITHOUT `gate_attn` (-35.4% A2A #1 wire; gate stays in `y_full` and is
            // applied post-A2A #2 in FP32 before BF16 round for 100% bit-exact TP1 parity).
            let sp = sp.unwrap();
            let tp = self.tp.as_ref().unwrap();
            let uw = lw.ulysses.as_ref().unwrap();
            let uws = ws.ulysses.as_ref().unwrap();
            let (x1_sp, x2_sp) = if three { (xb[1] + off_bf16, xb[2] + off_bf16) } else { (xb[0] + off_bf16, xb[0] + off_bf16) };
            k.proj(&uw.proj_full, xb[0] + off_bf16, x1_sp, x2_sp, uw.t1, uw.t2, dp(&uws.y_full), uw.proj_full.n_pad, sp.my_m, 0)?;
            let max_my_m = sp.blocks.iter().map(|&(a, b)| b - a).max().unwrap_or(0);
            let blk = tpk::InkUlyBlocks::from_blocks(&sp.blocks);
            if ulysses_wire_bf16() {
                let b16_n = (uws.max_nq + 2 * uws.max_nkv) * HD;
                let f32_n = uws.max_nq * 17 + if lw.sliding { 0 } else { uws.max_nkv * 8 };
                let row_bytes = (b16_n * 2 + f32_n * 4 + 15) & !15;
                tpk::ulysses_y_pack_b16(k, dp(&uws.y_full), uw.proj_full.n_pad, dp(&uws.a2a_src), row_bytes,
                                        max_my_m, sp.my_m, sp.world, uws.hds,
                                        uw.oq, uw.or_, uw.os, uw.ok, uw.okd, uw.ov, uw.ovd)?;
                tp.x.all_to_all(&self.dev, dp(&uws.a2a_src), dp(&uws.a2a_dst), max_my_m * row_bytes)?;
                tpk::ulysses_y_unpack_b16(k, dp(&uws.a2a_dst), row_bytes, max_my_m, dp(&ws.y), ldy,
                                          sp.world, self.rank.nq, self.rank.nkv,
                                          lw.oq, lw.or_, lw.os, lw.ok, lw.okd, lw.ov, lw.ovd, blk)?;
            } else {
                let raw_f = uws.max_nq * 145 + uws.max_nkv * if lw.sliding { 256 } else { 264 };
                let y_stride = (raw_f + 3) & !3;
                tpk::ulysses_y_pack(k, dp(&uws.y_full), uw.proj_full.n_pad, dp(&uws.a2a_src), y_stride,
                                    max_my_m, sp.my_m, sp.world, uws.hds,
                                    uw.oq, uw.or_, uw.os, uw.ok, uw.okd, uw.ov, uw.ovd)?;
                tp.x.all_to_all(&self.dev, dp(&uws.a2a_src), dp(&uws.a2a_dst), max_my_m * y_stride * 4)?;
                tpk::ulysses_y_unpack(k, dp(&uws.a2a_dst), y_stride, max_my_m, dp(&ws.y), ldy,
                                      sp.world, self.rank.nq, self.rank.nkv,
                                      lw.oq, lw.or_, lw.os, lw.ok, lw.okd, lw.ov, lw.ovd, blk)?;
            }
        } else {
            if let Some(sp) = sp {
                let tp = self.tp.as_ref().unwrap();
                if three {
                    tp.x.all_gather_rows_multi(&self.dev, &[(xb[0], D * 2), (xb[1], D * 2), (xb[2], D * 2)], &sp.blocks)?;
                } else {
                    tp.x.all_gather_rows_multi(&self.dev, &[(xb[0], D * 2)], &sp.blocks)?;
                }
            }
            let (x1, x2) = if three { (xb[1], xb[2]) } else { (xb[0], xb[0]) };
            k.proj(&lw.proj, xb[0], x1, x2, lw.t1, lw.t2, dp(&ws.y), ldy, m, 0)?;
        }
        // --- attention: all decode rows in one launch (Lanes), or per segment (its lane's caches / rings)
        let (lanes, segs): (&[&Lane], &[Seg]) = match rows { Rows::Segs(l, s) => (l, s), Rows::Lanes(_, _) => (&[], &[]) };
        if let Rows::Lanes(n, runs) = *rows {
            let tab = dp(ws.lt.as_ref().context("decode lanes: bind_lanes first")?);
            let rinfo = dp(&ws.rinfo);
            let ns = c.visits();
            let nq = self.rank.nq;
            tpk::attn_prep_lanes(k, nq, &PrepArgs {
                y: dp(&ws.y), ldy: ldy as i32, oq: lw.oq as i32, orr: lw.or_ as i32, ok: lw.ok as i32, ov: lw.ov as i32, okd: lw.okd, ovd: lw.ovd,
                q_scale: dp(&lw.q_scale), r_scale: dp(&lw.r_scale), qk_clip: c.qk_clip, rel_clip: c.rel_clip,
                full: (!lw.sliding) as i32, log_alpha: c.log_alpha, log_floor: c.log_floor,
                cw_k: dp(&lw.cw_k), cw_v: dp(&lw.cw_v), conv_div: if lw.sliding { 1.0 / 5f32.sqrt() } else { 1.0 / 3.0 },
                ring_k: 0, ring_v: 0, ring_mask: (RING - 1) as i32, ring_write: 1,
                kc: 0, vc: 0, cap_mask: 0, eps: c.eps, qf: dp(&ws.qf), rf: dp(&ws.rf), p0: 0, m: 1,
            }, tab, slot, ns, rinfo, n)?;
            tpk::attn_lanes(k, nq, &AttnArgs {
                qf: dp(&ws.qf), rf: dp(&ws.rf), kc: 0, vc: 0, cap_mask: 0, proj: dp(&lw.rel_proj), ext: lw.ext as i32,
                window: if lw.sliding { c.sliding_window as i32 } else { 0 }, p0: 0, m: 1, chunk: 0, c0: 0,
                y: dp(&ws.y), ldy: ldy as i32, os: lw.os as i32, og: lw.og as i32, out: dp(&ws.att),
                pm: dp(&ws.pm), pl: dp(&ws.pl), pacc: dp(&ws.pacc),
            }, tab, slot, ns, rinfo, n, runs)?;
        }
        let mut r0 = 0usize;
        for s in segs {
            let ln = lanes[s.lane];
            let cap = ln.cap[slot];
            let ybase = dp(&ws.y) + (r0 * ldy * 4) as u64;
            let rw = s.m <= RING - 3;
            let nq = self.rank.nq;
            let (qf, rf) = (dp(&ws.qf) + (r0 * nq * HD * 4) as u64, dp(&ws.rf) + (r0 * nq * DREL * 4) as u64);
            let ring = &ln.ring[slot];
            tpk::attn_prep(k, nq, &PrepArgs {
                y: ybase, ldy: ldy as i32, oq: lw.oq as i32, orr: lw.or_ as i32, ok: lw.ok as i32, ov: lw.ov as i32, okd: lw.okd, ovd: lw.ovd,
                q_scale: dp(&lw.q_scale), r_scale: dp(&lw.r_scale), qk_clip: c.qk_clip, rel_clip: c.rel_clip,
                full: (!lw.sliding) as i32, log_alpha: c.log_alpha, log_floor: c.log_floor,
                cw_k: dp(&lw.cw_k), cw_v: dp(&lw.cw_v), conv_div: if lw.sliding { 1.0 / 5f32.sqrt() } else { 1.0 / 3.0 },
                ring_k: dp(&ring[0]), ring_v: dp(&ring[1]), ring_mask: (RING - 1) as i32, ring_write: rw as i32,
                kc: dp(&ln.kc[slot]), vc: dp(&ln.vc[slot]), cap_mask: (cap - 1) as u32, eps: c.eps, qf, rf, p0: s.p0 as i32, m: s.m as i32,
            })?;
            if !rw {
                k.ring_store(ybase, ldy, lw.ok, dp(&ring[0]), self.rank.kvw(), s.p0, s.m)?;
                k.ring_store(ybase, ldy, lw.ov, dp(&ring[1]), self.rank.kvw(), s.p0, s.m)?;
            }
            let prefill = s.m > 16;
            super::gpu::k1::ATTN_SP.with(|c| c.set(self.attn_single_pass));
            let window = if lw.sliding { c.sliding_window } else { 0 };
            let groups = if prefill { vec![(0, s.m)] } else { super::gpu::k1::attn_groups(s.p0, s.m, window) };
            for (j0, mg) in groups {
                let (og_arg, out_ptr) = if use_ulysses {
                    (-1i32, dp(&ws.ulysses.as_ref().unwrap().att_f32) + ((r0 + j0) * nq * HD * 4) as u64)
                } else {
                    (lw.og as i32, dp(&ws.att) + ((r0 + j0) * nq * HD * 2) as u64)
                };
                tpk::attn(k, nq, &AttnArgs {
                    qf: qf + (j0 * nq * HD * 4) as u64, rf: rf + (j0 * nq * DREL * 4) as u64, kc: dp(&ln.kc[slot]), vc: dp(&ln.vc[slot]),
                    cap_mask: (cap - 1) as u32, proj: dp(&lw.rel_proj), ext: lw.ext as i32,
                    window: window as i32, p0: (s.p0 + j0) as i32, m: mg as i32, chunk: 0, c0: 0,
                    y: ybase + (j0 * ldy * 4) as u64, ldy: ldy as i32, os: lw.os as i32, og: og_arg,
                    out: out_ptr,
                    pm: dp(&ws.pm), pl: dp(&ws.pl), pacc: dp(&ws.pacc),
                }, prefill, if prefill { true } else { self.decode_tc })?;
            }
            r0 += s.m;
        }
        let res = alloc(live)?;
        let boundary = c.is_boundary(l);
        if use_ulysses {
            // DeepSpeed-Ulysses (Jacobs et al., arXiv:2309.14509) + U1 reverse All-to-All:
            // ungated FP32 `[m, H_local] -> [my_m, H_all]`, apply local `sigmoid(gate_attn)` -> BF16 `att_full`,
            // then local `wo_full` on `my_m` rows (zero ao RS, bit-exact TP1 parity!).
            let sp = sp.unwrap();
            let tp = self.tp.as_ref().unwrap();
            let uw = lw.ulysses.as_ref().unwrap();
            let uws = ws.ulysses.as_ref().unwrap();
            let max_my_m = sp.blocks.iter().map(|&(a, b)| b - a).max().unwrap_or(0);
            let nq_hd = self.rank.nq * HD;
            let att_stride = uws.max_nq * HD;
            let all_hd = NH * HD;
            let blk = tpk::InkUlyBlocks::from_blocks(&sp.blocks);
            if ulysses_wire_bf16() {
                tpk::ulysses_att_pack_b16(k, dp(&uws.att_f32), dp(&uws.a2a_src), max_my_m, sp.world, nq_hd, att_stride, blk)?;
                tp.x.all_to_all(&self.dev, dp(&uws.a2a_src), dp(&uws.a2a_dst), max_my_m * att_stride * 2)?;
                tpk::ulysses_att_unpack_b16(k, dp(&uws.a2a_dst), dp(&uws.y_full), uw.proj_full.n_pad, uw.og,
                                            dp(&uws.att_full), max_my_m, sp.my_m, sp.world, att_stride, all_hd, uws.hds)?;
            } else {
                tpk::ulysses_att_pack(k, dp(&uws.att_f32), dp(&uws.a2a_src), max_my_m, sp.world, nq_hd, att_stride, blk)?;
                tp.x.all_to_all(&self.dev, dp(&uws.a2a_src), dp(&uws.a2a_dst), max_my_m * att_stride * 4)?;
                tpk::ulysses_att_unpack(k, dp(&uws.a2a_dst), dp(&uws.y_full), uw.proj_full.n_pad, uw.og,
                                        dp(&uws.att_full), max_my_m, sp.my_m, sp.world, att_stride, all_hd, uws.hds)?;
            }
            k.proj(&uw.wo_full, dp(&uws.att_full), dp(&uws.att_full), dp(&uws.att_full), 1 << 30, 1 << 30, dp(&ws.ao) + sp.off_f32, D, sp.my_m, 0)?;
            self.conv_rows_sp(lanes[segs[0].lane], segs[0], slot, 2, &lw.attn_conv, dp(&ws.ao), if boundary { 0 } else { hp }, dp(&ws.pool[res]), sp)?;
        } else if let Some(sp) = sp {
            k.proj(&lw.wo, dp(&ws.att), dp(&ws.att), dp(&ws.att), 1 << 30, 1 << 30, dp(&ws.ao), D, m, 0)?;
            let tp = self.tp.as_ref().unwrap();
            tp.x.sp_reduce_scatter(&self.dev, dp(&ws.ao), D, &sp.blocks)?;
            self.conv_rows_sp(lanes[segs[0].lane], segs[0], slot, 2, &lw.attn_conv, dp(&ws.ao), if boundary { 0 } else { hp }, dp(&ws.pool[res]), sp)?;
        } else {
            k.proj(&lw.wo, dp(&ws.att), dp(&ws.att), dp(&ws.att), 1 << 30, 1 << 30, dp(&ws.ao), D, m, 0)?;
            self.reduce(dp(&ws.ao), m * D)?;
            // --- attn short conv + residual (block boundary: the residual restarts at y_attn)
            self.conv_rows(ws, rows, slot, 2, &lw.attn_conv, dp(&ws.ao), if boundary { 0 } else { hp }, dp(&ws.pool[res]))?;
        }
        // --- MLP input: AttnRes over blocks (+ this layer's input at a boundary) and the residual
        let mut vals: Vec<u64> = blocks.iter().map(|&b| dp(&ws.pool[b]) + off_f32).collect();
        if boundary { vals.push(hp + off_f32); }
        vals.push(dp(&ws.pool[res]) + off_f32);
        if !msgs && decode_fuse {
            // Drop 3 (#4a): fused cluster AttnRes + mlp_norm -> ws.mn directly (zero ws.ar round-trip)
            k.attnres_cl_norm(&vals, dp(&self.mlp_q[l]), 1, &[dp(&lw.mlp_norm)], &[0], &[dp(&ws.mn)], D, mr, c.eps)?;
        } else {
            k.attnres(&vals, dp(&self.mlp_q[l]), 1, &[dp(&ws.ar[0])], D, mr)?;
            let mut mi = dp(&ws.ar[0]);
            if msgs {
                let g = &lw.gates.as_ref().unwrap()[3];
                k.msggate(&g.down, dp(&g.wf), dp(&g.wg), mi, dp(&ws.msg[3]), dp(&ws.cat), dp(&ws.z), dp(&ws.mg[0]), D, mr, c.eps)?;
                mi = dp(&ws.mg[0]);
            }
            if let Some(sp) = sp {
                let tp = self.tp.as_ref().unwrap();
                let ob = if ws.actq.is_some() { dp(&ws.xb[0]) + off_bf16 } else { 0 };
                k.rmsnorm(mi, dp(&lw.mlp_norm), ob, dp(&ws.mn) + off_f32, D, mr, c.eps)?;
                moe::router_slice(&self.dev, &lw.moe, &ws.moe, dp(&ws.mn) + off_f32, sp.my_r0, sp.my_m)?;
                match ws.actq.as_ref() {
                    Some(aq) => {
                        tp.x.all_gather_rows_multi(
                            &self.dev,
                            &[(dp(&ws.xb[0]), D * 2), (dp(&ws.moe.ids), moe::TOPK * 4), (dp(&ws.moe.wts), moe::TOPK * 4)],
                            &sp.blocks,
                        )?;
                        super::actq::forward_routed_bf16(&self.dev, &lw.moe, &ws.moe, aq, dp(&ws.xb[0]), dp(&ws.mo), m)?;
                    }
                    None => {
                        tp.x.all_gather_rows_multi(
                            &self.dev,
                            &[(dp(&ws.moe.xh), D * 2), (dp(&ws.moe.ids), moe::TOPK * 4), (dp(&ws.moe.wts), moe::TOPK * 4)],
                            &sp.blocks,
                        )?;
                        moe::forward_routed(&self.dev, &lw.moe, &ws.moe, dp(&ws.mo), m)?;
                    }
                }
                tp.x.sp_reduce_scatter(&self.dev, dp(&ws.mo), D, &sp.blocks)?;
                let hn = alloc(live)?;
                self.conv_rows_sp(lanes[segs[0].lane], segs[0], slot, 3, &lw.mlp_conv, dp(&ws.mo), dp(&ws.pool[res]), dp(&ws.pool[hn]), sp)?;
                return Ok(hn);
            }
            k.rmsnorm(mi, dp(&lw.mlp_norm), 0, dp(&ws.mn), D, m, c.eps)?;
        }
        match ws.actq.as_ref() {
            Some(aq) => super::actq::forward(&self.dev, &lw.moe, &ws.moe, aq, dp(&ws.mn), dp(&ws.mo), m)?,
            None => moe::forward(&self.dev, &lw.moe, &ws.moe, dp(&ws.mn), dp(&ws.mo), m)?,
        }
        self.reduce(dp(&ws.mo), m * D)?;
        let hn = alloc(live)?;
        self.conv_rows(ws, rows, slot, 3, &lw.mlp_conv, dp(&ws.mo), dp(&ws.pool[res]), dp(&ws.pool[hn]))?;
        Ok(hn)
    }

    /// Drop 2 (#2): sequence-parallel short causal conv (`k = 4`) + residual over this rank's `[my_r0 .. my_r1)` slice.
    /// Exchanges the last 3 rows of every rank (`27 KiB/rank`) via `tp.halo`, computes `ink_sconv_sp` on `my_m` rows,
    /// and updates the conv ring on every rank with the final 3 rows (`seg.p0 + seg.m - 3 .. seg.p0 + seg.m`).
    #[allow(clippy::too_many_arguments)]
    fn conv_rows_sp(&self, lane: &Lane, seg: Seg, slot: usize, ri: usize, w: &CudaSlice<f32>,
                    x: u64, base: u64, out: u64, sp: &SpPlan) -> Result<()> {
        let tp = self.tp.as_ref().context("conv_rows_sp without TP")?;
        let halo = dp(&tp.halo);
        let dst_my = halo + (sp.rank * 3 * D * 4) as u64;
        let src_last3 = x + ((sp.my_r1 - 3) * D * 4) as u64;
        let r = unsafe { sys::cuMemcpyDtoDAsync_v2(dst_my, src_last3, 3 * D * 4, self.k.stream.get()) };
        ensure!(r == sys::CUresult::CUDA_SUCCESS, "inkling2 SP conv halo copy: {r:?}");
        tp.x.all_gather_rows_multi(&self.dev, &[(halo, D * 4)], &sp.halo_blocks)?;
        let halo_prev = if sp.rank == 0 { 0 } else { halo + ((sp.rank - 1) * 3 * D * 4) as u64 };
        let ring = dp(&lane.ring[slot][ri]);
        tpk::sconv_sp(
            self.k, x + sp.off_f32, halo_prev, ring, dp(w),
            if base == 0 { 0 } else { base + sp.off_f32 }, out + sp.off_f32,
            D, seg.p0 + sp.my_r0, sp.my_m,
        )?;
        let halo_end = halo + ((sp.world - 1) * 3 * D * 4) as u64;
        self.k.ring_store(halo_end, D, 0, ring, D, seg.p0 + seg.m - 3, 3)
    }

    #[allow(clippy::too_many_arguments)]
    fn conv_rows(&self, ws: &Work, rows: &Rows, slot: usize, ri: usize, w: &CudaSlice<f32>, x: u64, base: u64, out: u64) -> Result<()> {
        let (lanes, segs) = match *rows {
            Rows::Segs(l, s) => (l, s),
            Rows::Lanes(n, _) => {
                let tab = dp(ws.lt.as_ref().context("decode lanes: bind_lanes first")?);
                return self.k.sconv_lanes(x, D, tab, slot, self.cfg.visits(), ri, dp(w), base, out, D, dp(&ws.rinfo), n);
            }
        };
        let mut r0 = 0usize;
        for s in segs {
            let off = (r0 * D * 4) as u64;
            self.k.sconv(x + off, D, dp(&lanes[s.lane].ring[slot][ri]), dp(w), if base == 0 { 0 } else { base + off }, out + off, D, s.p0, s.m)?;
            r0 += s.m;
        }
        Ok(())
    }

    /// Write the device table of every lane's caches and conv rings (once, after the lanes exist; the lane buffers
    /// never move afterwards). Required by `decode_lanes`.
    pub fn bind_lanes(&self, ws: &mut Work, lanes: &[Lane]) -> Result<()> {
        let ns = self.cfg.visits();
        let mut t = vec![0u64; lanes.len().max(1) * ns * LANE_SLOT_U64];
        for (li, ln) in lanes.iter().enumerate() {
            for slot in 0..ns {
                let e = &mut t[(li * ns + slot) * LANE_SLOT_U64..][..LANE_SLOT_U64];
                e[0] = dp(&ln.kc[slot]);
                e[1] = dp(&ln.vc[slot]);
                for r in 0..4 { e[2 + r] = dp(&ln.ring[slot][r]); }
                e[6] = (ln.cap[slot] - 1) as u64;
            }
        }
        ws.lt = Some(self.dev.htod_sync_copy(&t)?);
        // a new binding invalidates captured graphs (they bake the table address)
        let st = std::mem::replace(&mut ws.graphs, Graphs { stream: std::ptr::null_mut(), exec: Vec::new() });
        drop(st);
        Ok(())
    }

    /// One decode step: row r = token `rows[r].1` of lane `rows[r].0` at that lane's position (not advanced). The
    /// logits of row r land in `ws.logits[r]`. `graph`: replay (capturing on first use) the CUDA graph of this row
    /// count; otherwise the same launches run eagerly (bit-identical). Needs `bind_lanes` and the CUDA-core decode
    /// attention (`decode_tc` false). Same as [`GpuModel::verify_lanes`] with one token per lane.
    pub fn decode_lanes(&self, ws: &mut Work, lanes: &[Lane], rows: &[(usize, u32)], graph: bool) -> Result<()> {
        let toks: Vec<[u32; 1]> = rows.iter().map(|r| [r.1]).collect();
        let segs: Vec<(usize, &[u32])> = rows.iter().zip(&toks).map(|(r, t)| (r.0, &t[..])).collect();
        self.verify_lanes(ws, lanes, &segs, graph)
    }

    /// S7 VERIFY / MULTI-ROW DECODE ENTRY (stable signature, the speculation leg calls this).
    ///
    /// `segs[i] = (lane, tokens)`: lane `lane` gets `tokens` (1..=16 of them) at positions `lane.pos ..
    /// lane.pos + tokens.len()`; the rows of all segments are concatenated in `segs` order (a lane appears at most
    /// once; total rows <= `MAX_OUT` = 48). The logits of EVERY row land in `ws.logits[row]` ([row][vocab],
    /// softcapped fp32), ready for `sample` / the drafter's acceptance.
    ///
    /// Commit rule (no snapshot, no rollback): the call writes the K / V cache and the conv rings of all rows; lane
    /// positions are NOT advanced: after accepting `a` rows of a segment (the caller's decision), set
    /// `lane.pos += a` (the next call overwrites the rejected rows before anything reads them).
    ///
    /// Exactness: every row is BIT-IDENTICAL to a 1-row `decode_lanes` of that token at that position (attention
    /// runs each row as its own decode row at its position, the convs read the earlier rows of the run from the
    /// call's own inputs, every other op is row-wise and batch-invariant), so greedy speculation is bitwise lossless.
    ///
    /// `graph`: replay (capturing on first use) the CUDA graph of this TOTAL row count (the graph does not depend on
    /// how the rows split into lanes or on positions: those are read on the device); `false` = the same launches
    /// eagerly (bit-identical). Needs `bind_lanes` and `decode_tc` false.
    pub fn verify_lanes(&self, ws: &mut Work, lanes: &[Lane], segs: &[(usize, &[u32])], graph: bool) -> Result<()> {
        let n: usize = segs.iter().map(|s| s.1.len()).sum();
        ensure!(n >= 1 && n <= MAX_OUT && !self.decode_tc, "verify_lanes: {n} rows (1..={MAX_OUT}), decode_tc {}", self.decode_tc);
        let mut ri = vec![0i32; 4 * LANES_MAXR];
        let mut r = 0usize;
        for (i, &(li, toks)) in segs.iter().enumerate() {
            ensure!(li < lanes.len() && (1..=16).contains(&toks.len()), "verify_lanes: segment {i}: lane {li}, {} tokens (1..=16)", toks.len());
            ensure!(segs[..i].iter().all(|s| s.0 != li), "verify_lanes: lane {li} appears twice");
            let ln = &lanes[li];
            ensure!(ln.pos + toks.len() <= ln.max_ctx, "lane {li}: positions {}..{} beyond its context {}", ln.pos, ln.pos + toks.len(), ln.max_ctx);
            for (t, &id) in toks.iter().enumerate() {
                ri[r] = id as i32;
                ri[LANES_MAXR + r] = (ln.pos + t) as i32;
                ri[2 * LANES_MAXR + r] = li as i32;
                ri[3 * LANES_MAXR + r] = t as i32;
                r += 1;
            }
        }
        self.mirror(|| Op::Verify { segs: segs.iter().map(|&(li, t)| (lanes[li].id, t.to_vec(), lanes[li].pos)).collect(), graph })?;
        self.dev.htod_sync_copy_into(&ri, &mut ws.rinfo)?;
        let outr: Vec<usize> = (0..n).collect();
        let idp = dp(&ws.rinfo);
        let runs = segs.iter().any(|s| s.1.len() > 1);
        // decode-class kernels for every op of a lanes call (verify rows == 1-row decodes, bitwise)
        struct Vc;
        impl Drop for Vc { fn drop(&mut self) { super::gpu::k1::VERIFY_CLASS.with(|c| c.set(false)); } }
        super::gpu::k1::VERIFY_CLASS.with(|c| c.set(true));
        let _vc = Vc;
        if !graph {
            return self.run(ws, &Rows::Lanes(n, runs), n, idp, &outr);
        }
        unsafe {
            if ws.graphs.stream.is_null() {
                self.dev.bind_to_thread()?;
                let mut st: sys::CUstream = std::ptr::null_mut();
                ensure!(sys::cuStreamCreate(&mut st, 0) == sys::CUresult::CUDA_SUCCESS, "cuStreamCreate");
                ws.graphs.stream = st;
            }
            let st = ws.graphs.stream;
            // one graph per (row count, multi-row runs or not)
            let gi = n + if runs { MAX_OUT + 1 } else { 0 } + 2 * (MAX_OUT + 1) * super::gpu::k1::verify_cfg();
            if ws.graphs.exec.len() <= gi { ws.graphs.exec.resize_with(gi + 1, || None); }
            if ws.graphs.exec[gi].is_none() {
                let old = self.k.stream.replace(st);
                let r = (|| -> Result<(sys::CUgraph, sys::CUgraphExec)> {
                    ensure!(sys::cuStreamBeginCapture_v2(st, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) == sys::CUresult::CUDA_SUCCESS, "begin capture");
                    let r = self.run(ws, &Rows::Lanes(n, runs), n, idp, &outr);
                    let mut g: sys::CUgraph = std::ptr::null_mut();
                    let e = sys::cuStreamEndCapture(st, &mut g);
                    r?;
                    ensure!(e == sys::CUresult::CUDA_SUCCESS, "end capture: {e:?}");
                    let mut x: sys::CUgraphExec = std::ptr::null_mut();
                    let e = sys::cuGraphInstantiateWithFlags(&mut x, g, 0);
                    ensure!(e == sys::CUresult::CUDA_SUCCESS, "graph instantiate: {e:?}");
                    Ok((g, x))
                })();
                self.k.stream.set(old);
                ws.graphs.exec[gi] = Some(r?);
            }
            let (_, x) = ws.graphs.exec[gi].unwrap();
            let e = sys::cuGraphLaunch(x, st);
            ensure!(e == sys::CUresult::CUDA_SUCCESS, "graph launch: {e:?}");
        }
        Ok(())
    }

    /// Sample one token per logits row `i < n` (temperature `t[i]`, <= 0 = greedy; seed per row) on the GPU
    /// (Gumbel-max: exact sampling from softmax(logits / T)); returns the ids.
    pub fn sample(&self, ws: &mut Work, t: &[f32], seeds: &[u64]) -> Result<Vec<u32>> {
        let n = t.len();
        ensure!(n >= 1 && n <= MAX_OUT && seeds.len() == n);
        self.mirror(|| Op::Sample { t: t.to_vec(), seeds: seeds.to_vec(), check: self.tp.as_ref().map(|x| x.last.borrow().clone()).unwrap_or_default() })?;
        let it: Vec<f32> = t.iter().map(|&x| if x > 0.0 { 1.0 / x } else { 0.0 }).collect();
        self.dev.htod_sync_copy_into(&it, &mut ws.smp_it.slice_mut(0..n))?;
        self.dev.htod_sync_copy_into(seeds, &mut ws.smp_seed.slice_mut(0..n))?;
        let v = self.head.n;
        if let Some(tp) = &self.tp {
            // vocab-parallel: each rank's best of its rows, exchanged, the best of the bests (ties: lowest id)
            let w = self.rank.world;
            tpk::gumbel_rec(self.k, dp(&ws.logits), v, v, self.rank.v0, dp(&ws.smp_seed), dp(&ws.smp_it), dp(&ws.smp_max), dp(&ws.smp_idx),
                            SMP_G, n, dp(&tp.rec), self.rank.rank, w)?;
            tp.x.all_reduce(&self.dev, dp(&tp.rec), n * w * 2)?;
            tpk::rank_pick(self.k, dp(&tp.rec), w, n, dp(&ws.smp_out))?;
            let mut o = vec![0i32; n];
            self.dev.dtoh_sync_copy_into(&ws.smp_out.slice(0..n), &mut o)?;
            tp.x.check("after a sampling exchange")?;
            let out: Vec<u32> = o.iter().map(|&x| x as u32).collect();
            *tp.last.borrow_mut() = out.clone();
            return Ok(out);
        }
        self.k.launch("ink3_gumbel_part", (SMP_G as u32, n as u32, 1), (256, 1, 1), 0,
                      KArgs::new().v(dp(&ws.logits)).v(v as i32).v(v as i32).v(dp(&ws.smp_seed)).v(dp(&ws.smp_it)).v(dp(&ws.smp_max)).v(dp(&ws.smp_idx)))?;
        self.k.launch("ink3_gumbel_pick", (n as u32, 1, 1), (256, 1, 1), 0,
                      KArgs::new().v(dp(&ws.smp_max)).v(dp(&ws.smp_idx)).v(SMP_G as i32).v(dp(&ws.smp_out)))?;
        let mut o = vec![0i32; n];
        self.dev.dtoh_sync_copy_into(&ws.smp_out.slice(0..n), &mut o)?;
        Ok(o.iter().map(|&x| x as u32).collect())
    }

    /// Q1 hook: the logits of rows `r0..r0 + n` (n <= 16) of the LAST forward call (output AttnRes, final norm, LM head
    /// on those rows) into `ws.logits[0..n]` (the probe's teacher-forced dump: every position of a prefill chunk).
    pub fn logits_rows(&self, ws: &mut Work, r0: usize, n: usize) -> Result<()> {
        ensure!(n >= 1 && n <= MAX_OUT, "logits_rows: {n} rows");
        let (blocks, h) = ws.last.clone();
        for i in 0..n {
            let r = r0 + i;
            let mut vals: Vec<u64> = blocks.iter().map(|&b| dp(&ws.pool[b]) + (r * D * 4) as u64).collect();
            vals.push(dp(&ws.pool[h]) + (r * D * 4) as u64);
            self.k.attnres(&vals, dp(&self.out_q), 1, &[dp(&ws.fin) + (i * D * 4) as u64], D, 1)?;
        }
        self.k.rmsnorm(dp(&ws.fin), dp(&self.norm), 0, dp(&ws.finn), D, n, self.cfg.eps)?;
        head::forward(&self.dev, &self.head, &ws.hs, dp(&ws.finn), dp(&ws.logits), self.head.n, n, false)
    }

    /// Device address of the full-vocabulary logits rows `0..n` ([n][vocab()] fp32): `ws.logits` at TP1; under TP
    /// gathered via 2-rail byte-exact all-gather (`pack` -> `all_gather_rows` -> `unpack`, sending only `(W-1)/W` of
    /// the vocabulary — 6.85x less wire/rail at TP8 vs 1-rail all-reduce). Only needed for CPU-sampled (`top_p`/`top_k`) rows.
    pub fn logits_full(&self, ws: &Work, n: usize) -> Result<u64> {
        let Some(tp) = &self.tp else { return Ok(dp(&ws.logits)) };
        self.mirror(|| Op::GatherLogits { n })?;
        self.gather_vocab_rows_inner(tp, dp(&ws.logits), n)
    }

    /// Gather `n` rows of any `[n][self.head.n]` vocab-sharded fp32 buffer (e.g. `ws.logits` or drafter `w.step`)
    /// into `tp.full_buf` `[n][vocab_full]` via 2-rail byte-exact all-gather.
    pub fn gather_vocab_rows(&self, src: u64, n: usize) -> Result<u64> {
        let Some(tp) = &self.tp else { return Ok(src) };
        self.gather_vocab_rows_inner(tp, src, n)
    }

    fn gather_vocab_rows_inner(&self, tp: &TpRank, src: u64, n: usize) -> Result<u64> {
        let v = self.vocab_full;
        let w = self.rank.world;
        let per = shard::vocab_range(v, w, 0).1;
        let full = tp.full_buf(&self.dev, MAX_OUT * v)?;
        let gather = tp.gather_buf(&self.dev, w * MAX_OUT * per)?;
        tpk::logits_pack(self.k, src, self.head.n, self.head.n, dp(&*gather), per, self.rank.rank, n)?;
        let blocks: Vec<(usize, usize)> = (0..w).map(|r| (r * n, (r + 1) * n)).collect();
        tp.x.all_gather_rows(&self.dev, dp(&*gather), per * 4, &blocks)?;
        tpk::logits_unpack(self.k, dp(&*gather), per, dp(&*full), v, n, w)?;
        tp.x.check("after a logits gather")?;
        Ok(dp(&*full))
    }

    /// The logits rows `0..n` (host copy).
    pub fn logits_host(&self, ws: &Work, n: usize) -> Result<Vec<f32>> {
        if let Some(tp) = &self.tp {
            // the CPU-sampled rows: 2-rail byte-exact all-gather of each rank's vocabulary slice
            self.mirror(|| Op::Logits { n })?;
            let v = self.vocab_full;
            let full_ptr = self.gather_vocab_rows_inner(tp, dp(&ws.logits), n)?;
            self.k.sync()?;
            tp.x.check("after a logits gather")?;
            let mut o = vec![0f32; n * v];
            let r = unsafe { sys::cuMemcpyDtoH_v2(o.as_mut_ptr() as *mut std::ffi::c_void, full_ptr, n * v * 4) };
            ensure!(r == sys::CUresult::CUDA_SUCCESS, "logits_host dtoh: {r:?}");
            return Ok(o);
        }
        self.k.sync()?;
        let mut o = vec![0f32; n * self.head.n];
        self.dev.dtoh_sync_copy_into(&ws.logits.slice(0..n * self.head.n), &mut o)?;
        Ok(o)
    }

    /// Prefill `ids` into `lane` from its current position (chunks of CHUNK rows); returns the logits of the last
    /// row in `ws.logits[0]`. Advances `lane.pos`.
    pub fn prefill(&self, ws: &mut Work, lanes: &mut [Lane], li: usize, ids: &[u32]) -> Result<()> {
        let mut i = 0;
        while i < ids.len() {
            let m = (ids.len() - i).min(CHUNK);
            let last = i + m == ids.len();
            let p0 = lanes[li].pos;
            let outr = [m - 1];
            let img = std::mem::take(&mut ws.img);
            ws.set_img_chunk(&img, i, m);
            ws.img = img;
            {
                let refs: Vec<&Lane> = lanes.iter().collect();
                self.forward(ws, &refs, &[Seg { lane: li, p0, m }], &ids[i..i + m], if last { &outr[..] } else { &[][..] })?;
            }
            lanes[li].pos += m;
            i += m;
        }
        Ok(())
    }
}

#[cfg(test)]
mod sp_tests {
    use super::*;

    #[test]
    fn sp_plan_geometry_and_halo_alignment() {
        for &w in &[2usize, 4, 8] {
            assert!(SpPlan::for_chunk(0, w, 32).is_none());
            for &m in &[73usize, 128, 300, 552, 1024, 2048] {
                let mut covered = 0usize;
                for r in 0..w {
                    let p = SpPlan::for_chunk(r, w, m).expect("valid SP plan");
                    assert_eq!(p.my_r0, covered);
                    assert!(p.my_m >= 3, "w={w} m={m} r={r} my_m={}", p.my_m);
                    assert_eq!(p.off_f32, (p.my_r0 * D * 4) as u64);
                    assert_eq!(p.off_bf16, (p.my_r0 * D * 2) as u64);
                    assert_eq!(p.halo_blocks[r], (r * 3, (r + 1) * 3));
                    covered = p.my_r1;
                }
                assert_eq!(covered, m, "w={w} m={m}");
            }
        }
        // Edge case: m = 65 at W = 8 gives rank 7 only 2 rows (< 3 halo), so SP safely declines.
        assert!(SpPlan::for_chunk(0, 8, 65).is_none());
        assert!(SpPlan::for_chunk(0, 8, 64).is_some());
        // Row byte sizes for every SP all-gather buffer must be 16 B-aligned.
        for rb in [D * 2, D * 4, moe::TOPK * 4, 4 * D * 2] {
            assert_eq!(rb % 16, 0, "row_bytes={rb}");
        }
    }
}
