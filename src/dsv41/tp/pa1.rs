//! DSV4.1 TP (leg PA1): the attention sublayer's small kernels as the external reference recipe structures them
//! (PLAN/DSV41_RX6_REPORT.md 2.1-2.3), only the parts that keep our output bits:
//!
//! 1. `--dsv41-qkv-fuse` (default on; decode / verify graph calls with PD1's normrot): the q norm (+ wq_b's rotated rows)
//!    and the window K/V write in ONE launch (`d41_q_kv_norm`, roles by blockIdx.y, each role today's device function),
//!    its constants L2-prefetched before the wait; q's RoPE after wq_b folded into the attention's q load
//!    (`D41_ATTN_X_QROPE`): two launches a layer fewer.
//! 2. `--dsv41-attn-wpf` (default on): the decode attention prefetches the row's window keys into L2 before its
//!    `griddepcontrol.wait` (`D41_ATTN_X_WPF`).
//! 3. `--dsv41-attn-merge` (default on): the decode attention's split merge + epilogue (+ PD1's wo_a rotation) as its own
//!    launch, one CTA per (row, head), in our merge order (`d41_attn_merge`, `D41_ATTN_X_DEFER`).
//! 4. `--dsv41-hc-side auto|on|off` (auto = on at TP1 / TP2): the sublayer's hc mix (both ranks' K slices computed
//!    locally, then rank 0's + rank 1's values added once: the world-2 all-reduce's one IEEE add) and its Sinkhorn on a
//!    side stream forked when the sublayer's stream input is ready and joined right before the sublayer's hc_post; the
//!    25-float rider leaves the reduce; the pre-norm loads its pre record and norm weights before its wait. TP2 holds the
//!    hc_fn tensors whole for it (load plan `hc_whole`, +78.6 MB a rank). TP4 / TP8: unchanged (the world-4 reduce's add
//!    order is not replicated: RX6 gap G2).
//!
//! Every switch keeps the output bits; `off` is the A/B. Engagement: `velogb10_dsv41_fw_pa1_*` on /metrics (launches the
//! host built: a graph replay re-runs a captured launch without counting) and `scripts/owner_check.py`.

use super::fwd::{MixK, TpModel};
use super::frame::MIX_STRIDE;
use crate::dsv41::census::classify::TensorClass as C;
use crate::dsv41::fwd::dev::Buf;
use crate::dsv41::fwd::receipts;
use crate::dsv41::gpu::mixers as km;
use anyhow::{ensure, Context, Result};
use cudarc::driver::sys;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

fn opt(name: &'static str) -> Option<String> { crate::opts::lookup(name).and_then(crate::opts::get) }
fn on(name: &'static str) -> bool { opt(name).map_or(true, |v| v != "off") }

/// `--dsv41-hc-side` resolved for a world size: `auto` = TP1 / TP2 (TP4 keeps the 100-byte sublayer-reduce rider by default);
/// explicit `on` = TP1, TP2 and TP4 (leg P6-5 replicates the world-4 `(p0 + p2) + (p1 + p3)` association locally);
/// TP8 and up keep the reduce path (an `on` there falls back with one line).
pub fn hc_side_for(world: usize) -> bool {
    let v = opt("dsv41-hc-side").unwrap_or_else(|| "auto".into());
    match v.as_str() {
        "off" => false,
        "on" if matches!(world, 1 | 2 | 4 | 8) => true,
        "on" => {
            static SAID: std::sync::Once = std::sync::Once::new();
            SAID.call_once(|| crate::logq::send(false, format!("[dsv41] --dsv41-hc-side on at TP{world}: the hc mix stays on the sublayer reduce (the world-{world} reduce's add order is not replicated locally; TP1 / TP2 / TP4 / TP8 only)")));
            FALLBACK.store(1, Relaxed);
            false
        }
        _ => matches!(world, 1 | 2),
    }
}
/// The load plan holds hc_fn whole on every rank (TP2 / TP4 / TP8 with the side stream: all ranks' K slices are computed locally).
pub fn hc_whole_for(world: usize) -> bool { matches!(world, 2 | 4 | 8) && hc_side_for(world) }

pub struct Pa1 {
    pub fuse: bool,
    pub wpf: bool,
    pub merge: bool,
    pub side: bool,
    /// P7ATTN (`--dsv41-attn-2cta`): 0 = off (512-thread 1 CTA/SM), 1 = auto (shape-dispatched), 2 = on (256-thread 2 CTAs/SM)
    pub attn_2cta: u8,
    /// INT12: the side stream was asked for this world size (`--dsv41-hc-side` auto / on at TP1 / TP2); `side` is what runs
    /// after the cluster agreed (`TpModel::pa1_agree`)
    pub(super) asked: bool,
    /// `--dsv41-attn-split recipe` (opt-in, different bits)
    pub recipe: bool,
    stream: u64,
    ev_fork: sys::CUevent,
    ev_join: sys::CUevent,
    /// the side stream's mix counters (2 halves x row groups), zero between calls (the kernel's last arrival re-arms)
    cnt: Option<Buf>,
    /// a side mix is in flight (its join is owed before the record is read)
    pending: bool,
    /// layer whose window rows the fused q / kv launch wrote (window() then skips its own launch)
    pub(super) swa_done: Option<usize>,
}

static BOOT: AtomicU64 = AtomicU64::new(0);
static FALLBACK: AtomicU64 = AtomicU64::new(0);
/// INT12: 1 = the side stream was asked at TP2 but a rank's shard holds the hc_fn K slice only; every rank runs the reduce path
static KSLICE_FALLBACK: AtomicU64 = AtomicU64::new(0);
static QKV: AtomicU64 = AtomicU64::new(0);
static QROPE: AtomicU64 = AtomicU64::new(0);
static WPF: AtomicU64 = AtomicU64::new(0);
static MERGE: AtomicU64 = AtomicU64::new(0);
static SIDE: AtomicU64 = AtomicU64::new(0);

impl Pa1 {
    /// `whole`: every trunk layer holds its hc_fn tensors whole (the plan's `hc_whole` at TP2 / `attn_whole` at TP4).
    pub fn new(g: &crate::dsv41::fwd::dev::Gpu, world: usize, max_rows: usize, whole: bool) -> Result<Pa1> {
        let asked = hc_side_for(world);
        // INT12 (owner rule: fall back gracefully, never refuse): a rank whose shard holds the hc_fn K slice only (a shard / stage
        // built before the plan held hc_fn whole) cannot take the side path; it says so here, and the cluster then agrees in
        // `TpModel::pa1_agree` (every rank must take the same path: the ranks' reduce sizes differ between the two paths)
        let side = asked && (!matches!(world, 2 | 4 | 8) || whole);
        let (mut st, mut e1, mut e2): (sys::CUstream, sys::CUevent, sys::CUevent) = (std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut());
        let mut cnt = None;
        if asked {
            // a BLOCKING stream (AGENTS 2.1): it reads and writes data, so it must order against cudarc's NULL-stream copies
            unsafe {
                ensure!(sys::cuStreamCreate(&mut st, sys::CUstream_flags::CU_STREAM_DEFAULT as u32) == sys::CUresult::CUDA_SUCCESS, "PA1 hc side stream");
                ensure!(sys::cuEventCreate(&mut e1, sys::CUevent_flags::CU_EVENT_DISABLE_TIMING as u32) == sys::CUresult::CUDA_SUCCESS, "PA1 fork event");
                ensure!(sys::cuEventCreate(&mut e2, sys::CUevent_flags::CU_EVENT_DISABLE_TIMING as u32) == sys::CUresult::CUDA_SUCCESS, "PA1 join event");
            }
            cnt = Some(g.buf(world.clamp(1, 8) * max_rows.max(super::fwd::HEAD_MAX) * 4)?);   // zeroed (Gpu::buf memsets)
        }
        let attn_2cta = match opt("dsv41-attn-2cta").as_deref() {
            Some("off") => 0u8,
            Some("on") => 2u8,
            _ => 1u8,
        };
        let p = Pa1 { recipe: opt("dsv41-attn-split").as_deref() == Some("recipe"), fuse: on("dsv41-qkv-fuse"), wpf: on("dsv41-attn-wpf"), merge: on("dsv41-attn-merge"), side, attn_2cta, asked, stream: st as u64, ev_fork: e1, ev_join: e2,
                      cnt, pending: false, swa_done: None };
        static SAID: std::sync::Once = std::sync::Once::new();
        SAID.call_once(|| crate::logq::send(false, p.line(world)));
        BOOT.store((p.fuse as u64) | (p.wpf as u64) << 1 | (p.merge as u64) << 2 | (p.side as u64) << 3 | 16 | (p.recipe as u64) << 5 | (p.attn_2cta as u64) << 6, Relaxed);
        Ok(p)
    }
    pub fn line(&self, world: usize) -> String {
        let f = |b: bool| if b { "ON" } else { "off" };
        let c2 = match self.attn_2cta { 0 => "off (1 CTA/SM)", 2 => "ON (2 CTAs/SM)", _ => "auto" };
        format!("[dsv41] PA1 attention sublayer (decode / verify): q / kv norm in one launch + q RoPE in the attention's q load {} (--dsv41-qkv-fuse), \
                 window keys L2-prefetched before the wait {} (--dsv41-attn-wpf), split merge + epilogue as its own per-head launch {} (--dsv41-attn-merge), \
                 2-CTA/SM 8-head kernel {c2} (--dsv41-attn-2cta), \
                 hc mix + Sinkhorn on a side stream {} (--dsv41-hc-side, TP{world}{}); attention split rule: {} (--dsv41-attn-split{})",
                f(self.fuse), f(self.wpf), f(self.merge), f(self.side), if !matches!(world, 1 | 2 | 4 | 8) { ": TP1 / TP2 / TP4 / TP8 only" } else { "" },
                if self.recipe { "RECIPE (opt-in: 8 splits of 32-key blocks round-robin for every row of every call; different bits from exact)" } else { "exact" },
                if self.recipe { " recipe" } else { ", default exact" })
    }
    /// The attention decode entry's PA1 flags (`attn::X_*`); `qrope` = this call folds q's RoPE into the q load.
    pub fn attn_x(&self, qrope: bool) -> u32 {
        use crate::dsv41::gpu::attn::{X_1CTA, X_2CTA, X_DEFER, X_QROPE, X_RECIPE, X_WPF};
        let c2 = match self.attn_2cta { 0 => X_1CTA, 2 => X_2CTA, _ => 0 };
        c2 | (if self.recipe { X_RECIPE } else { 0 }) | (if self.wpf { X_WPF } else { 0 }) | (if self.merge { X_DEFER } else { 0 }) | (if qrope { X_QROPE } else { 0 })
    }
    pub fn count_attn(x: u32) {
        use crate::dsv41::gpu::attn::{X_DEFER, X_QROPE, X_WPF};
        if x & X_WPF != 0 { WPF.fetch_add(1, Relaxed); }
        if x & X_DEFER != 0 { MERGE.fetch_add(1, Relaxed); }
        if x & X_QROPE != 0 { QROPE.fetch_add(1, Relaxed); }
    }
}
impl Drop for Pa1 {
    fn drop(&mut self) {
        unsafe {
            if !self.ev_fork.is_null() { sys::cuEventDestroy_v2(self.ev_fork); }
            if !self.ev_join.is_null() { sys::cuEventDestroy_v2(self.ev_join); }
            if self.stream != 0 { sys::cuStreamDestroy_v2(self.stream as sys::CUstream); }
        }
    }
}

pub fn append(o: &mut String) {
    use std::fmt::Write as _;
    let b = BOOT.load(Relaxed);
    if b == 0 { return; }
    let rows: [(&str, &str, u64); 14] = [
        ("velogb10_dsv41_fw_pa1_attn_rule", "attention split rule serving: 0 = exact (default), 1 = recipe (--dsv41-attn-split recipe, opt-in, different bits)", (b >> 5) & 1),
        ("velogb10_dsv41_fw_pa1_attn_2cta", "decode attention 2-CTA/SM mode (--dsv41-attn-2cta): 0 = off, 1 = auto (default), 2 = on", (b >> 6) & 3),
        ("velogb10_dsv41_fw_pa1_attn_recipe_launches_total", "attention launches the host built on the recipe split rule", crate::dsv41::gpu::attn::recipe_calls()),
        ("velogb10_dsv41_fw_pa1_qkv_fuse", "1 = PA1 fused q / kv norm launch + q RoPE in the attention's q load on (--dsv41-qkv-fuse)", b & 1),
        ("velogb10_dsv41_fw_pa1_attn_wpf", "1 = PA1 window-key L2 prefetch before the attention's wait on (--dsv41-attn-wpf)", (b >> 1) & 1),
        ("velogb10_dsv41_fw_pa1_attn_merge", "1 = PA1 per-head split merge + epilogue launch on (--dsv41-attn-merge)", (b >> 2) & 1),
        ("velogb10_dsv41_fw_pa1_hc_side", "1 = PA1 hc mix + Sinkhorn on a side stream (--dsv41-hc-side; TP1 / TP2 / TP4)", (b >> 3) & 1),
        ("velogb10_dsv41_fw_pa1_hc_side_fallback", "1 = --dsv41-hc-side on was asked at TP8 or wider and fell back to the reduce path", FALLBACK.load(Relaxed)),
        ("velogb10_dsv41_fw_pa1_hc_side_kslice_fallback", "1 = the hc side stream was asked at TP2 but a rank's shard holds the hc_fn K slice only (a stage from an older plan): every rank runs the hc mix on the reduce path (rebuild the stage with --dsv41-load-cache on to get the side stream)", KSLICE_FALLBACK.load(Relaxed)),
        ("velogb10_dsv41_fw_pa1_qkv_launches_total", "fused q / kv norm launches the host built (graph replays re-run captured launches without counting)", QKV.load(Relaxed)),
        ("velogb10_dsv41_fw_pa1_qrope_launches_total", "decode attention launches built with q's RoPE in the q load", QROPE.load(Relaxed)),
        ("velogb10_dsv41_fw_pa1_wpf_launches_total", "decode attention launches built with the window-key L2 prefetch before the wait", WPF.load(Relaxed)),
        ("velogb10_dsv41_fw_pa1_merge_launches_total", "decode attention launches built asking for the separate per-head merge launch (a call with one CTA per (row, head group) keeps the in-CTA epilogue)", MERGE.load(Relaxed)),
        ("velogb10_dsv41_fw_pa1_hc_side_mixes_total", "sublayer hc mixes the host built on the side stream", SIDE.load(Relaxed)),
    ];
    for (n, h, v) in rows { let _ = writeln!(o, "# HELP {n} {h}\n# TYPE {n} counter\n{n} {v}"); }
}

/// INT12: the cluster's hc side-stream decision from every rank's report (`lacks[r]` = rank r asked for the side stream and its
/// shard holds the hc_fn K slice only): the side stream runs only when no rank lacks the whole tensor.
pub fn side_agreed(asked: bool, lacks: &[f32]) -> bool { asked && lacks.iter().all(|&x| x == 0.0) }

impl TpModel<'_> {
    /// INT12 (owner rule 8: support it or fall back gracefully, never refuse): at TP2 with the hc side stream asked, every rank
    /// reports whether its shard holds hc_fn whole (an all-reduce over one slot per rank, the same sum on every rank); the head
    /// decides from it and tells the nodes (a broadcast), so every rank takes the same path. A rank holding the K slice only
    /// turns the side stream OFF for the whole cluster with one line and `velogb10_dsv41_fw_pa1_hc_side_kslice_fallback` 1.
    /// Called once on every rank (SPMD) right after the first boot barrier (the ranks load at different speeds), before any forward.
    pub fn pa1_agree(&mut self) -> Result<()> {
        if !matches!(self.world, 2 | 4 | 8) || !self.pa1.asked { return Ok(()); }
        let t = self.tiny.ptr();
        self.f.g.memset(t, 64, 0)?;
        let mut mine = [0f32; 8];
        mine[self.rank] = if self.pa1.side { 0.0 } else { 1.0 };
        self.f.g.h2d_async(t, &mine[..self.world])?;
        self.reduce(t, 16)?;
        let lacks = self.f.g.down::<f32>(t, self.world)?;
        super::comm::Dsv41Comm::check(self.comm, "the hc side-stream agreement")?;
        let head = side_agreed(true, &lacks);
        let side = self.bcast_small(&[if head { 1.0 } else { 0.0 }])?[0] != 0.0;
        if !side {
            self.pa1.side = false;
            KSLICE_FALLBACK.store(1, Relaxed);
            BOOT.fetch_and(!(1u64 << 3), Relaxed);
            let who: Vec<String> = lacks.iter().enumerate().filter(|(_, &x)| x != 0.0).map(|(r, _)| format!("rank {r}")).collect();
            crate::logq::send(false, format!("[dsv41] --dsv41-hc-side ({}) asked for the hc mix on a side stream at TP{}, but {} hold(s) the hc_fn K slice only \
                (a shard / stage from an older plan): every rank runs the hc mix on the sublayer reduce instead (same bits, without PA1's overlap). \
                To get the side stream, rebuild the stage (--dsv41-load-cache on). velogb10_dsv41_fw_pa1_hc_side_kslice_fallback 1",
                opt("dsv41-hc-side").unwrap_or_else(|| "auto".into()), self.world, who.join(", ")));
        }
        Ok(())
    }

    /// Hook (body, at a sublayer's start): fork the sublayer's hc mix (all ranks' K slices at TP2 / TP4, the one slice at TP1)
    /// and its Sinkhorn onto the side stream. Returns the record pointer (valid after `pa1_join`), or None when the side
    /// stream is off (the caller runs mix_partial on the reduce).
    pub(super) fn pa1_mix_side(&mut self, l: usize, attn_side: bool, mk: MixK, x: u64, rows: usize) -> Result<Option<u64>> {
        if !self.pa1.side || !matches!(self.world, 1 | 2 | 4 | 8) { return Ok(None); }
        ensure!(!self.pa1.pending, "PA1: a side mix is still pending (join missing)");
        let halves = self.world;
        let k = self.cfg.hc * self.cfg.hidden;
        let main = self.f.g.stream as sys::CUstream;
        let side = self.pa1.stream as sys::CUstream;
        let chunks = km::partial_chunks(mk.klen);
        let rg = km::pick_rg(rows);
        let ngroups = rows.div_ceil(rg);
        let cnt = self.pa1.cnt.as_ref().context("PA1 side counters")?;
        ensure!(halves * ngroups * 4 <= cnt.bytes, "PA1 side mix: {ngroups} row groups > the counter buffer");
        let part = self.f.b.take(halves * chunks * rows * 32 * 4)?;
        let rider = self.f.b.take(halves * rows * MIX_STRIDE * 4)?;
        let out = self.f.b.take(rows * 24 * 4)?;
        let mut h = [km::MixPartialArgs::default(); 8];
        for (z, a) in h.iter_mut().enumerate().take(halves) {
            // rank z's slice: its k0 and fn columns (the whole tensor at TP2 / TP4 / TP8, the resident one at TP1), its own partials,
            // counters and rider: exactly the launch rank z runs on the reduce path
            let (k0, fnp, fld) = if halves == 1 { (mk.k0, mk.fn_ptr, mk.fn_ld) } else {
                ensure!(mk.fn_base != 0 && mk.fn_ld == k && halves * mk.klen == k, "layer {l}: the side mix needs hc_fn whole on the rank (K {k}, slice {})", mk.klen);
                (z * mk.klen, mk.fn_base + (4 * z * mk.klen) as u64, mk.fn_ld)
            };
            *a = km::MixPartialArgs { x, x_ld: k as i64, fn_: fnp, fn_ld: fld as i64, part: part + (z * chunks * rows * 32 * 4) as u64, chunk_stride: (rows * 32) as i64,
                                      row_stride: 32, rank_part: rider + (z * rows * MIX_STRIDE * 4) as u64, rp_row_stride: MIX_STRIDE as i64,
                                      counters: cnt.ptr() + (z * ngroups * 4) as u64, k0: k0 as i32, klen: mk.klen as i32, rows: rows as i32, rg: rg as i32 };
        }
        let (sc, bc) = if attn_side { (C::HcAttnScale, C::HcAttnBase) } else { (C::HcFfnScale, C::HcFfnBase) };
        let w = &self.layers[l].as_ref().unwrap().w;
        let fa = km::MixFinishArgs { part: rider, slot_stride: (rows * MIX_STRIDE) as i64, row_stride: MIX_STRIDE as i64, scale: w.param(sc)?.ptr.0, base: w.param(bc)?.ptr.0,
                                     out, out_ld: 24, nslots: halves as i32, rows: rows as i32, inv_k: 1.0 / k as f32, eps_norm: self.cfg.rms_eps as f32,
                                     eps_hc: self.cfg.hc_eps as f32, sinkhorn_iters: self.cfg.sinkhorn_iters as i32 };
        unsafe {
            ensure!(sys::cuEventRecord(self.pa1.ev_fork, main) == sys::CUresult::CUDA_SUCCESS, "PA1 fork record");
            ensure!(sys::cuStreamWaitEvent(side, self.pa1.ev_fork, 0) == sys::CUresult::CUDA_SUCCESS, "PA1 fork wait");
        }
        match halves {
            1 => crate::dsv41::gpu::launch_args("d41_mix_partial", (chunks as u32, ngroups as u32, 1), (128, 1, 1), 0, self.pa1.stream, &h[0])?,
            2 => {
                let h2 = [h[0], h[1]];
                crate::dsv41::gpu::launch_args("d41_mix_partial2", (chunks as u32, ngroups as u32, 2), (128, 1, 1), 0, self.pa1.stream, &h2)?;
            }
            4 => {
                let h4 = [h[0], h[1], h[2], h[3]];
                crate::dsv41::gpu::launch_args("d41_mix_partial4", (chunks as u32, ngroups as u32, 4), (128, 1, 1), 0, self.pa1.stream, &h4)?;
            }
            _ => crate::dsv41::gpu::launch_args("d41_mix_partial8", (chunks as u32, ngroups as u32, 8), (128, 1, 1), 0, self.pa1.stream, &h)?,
        }
        crate::dsv41::gpu::launch_args("d41_mix_finish", ((rows as u32).div_ceil(8), 1, 1), (128, 1, 1), 0, self.pa1.stream, &fa)?;
        unsafe { ensure!(sys::cuEventRecord(self.pa1.ev_join, side) == sys::CUresult::CUDA_SUCCESS, "PA1 join record"); }
        self.pa1.pending = true;
        SIDE.fetch_add(1, Relaxed);
        let entry = match halves { 1 => "d41_mix_partial+d41_mix_finish", 2 => "d41_mix_partial2+d41_mix_finish", 4 => "d41_mix_partial4+d41_mix_finish", _ => "d41_mix_partial8+d41_mix_finish" };
        receipts::kernel("tp.mix (PA1 side stream: all K slices + Sinkhorn)", entry, rows);
        Ok(Some(out))
    }

    /// The side mix's rejoin: the main stream waits for the record (before the sublayer's hc_post reads post / comb).
    pub(super) fn pa1_join(&mut self) -> Result<()> {
        if std::mem::take(&mut self.pa1.pending) {
            unsafe { ensure!(sys::cuStreamWaitEvent(self.f.g.stream as sys::CUstream, self.pa1.ev_join, 0) == sys::CUresult::CUDA_SUCCESS, "PA1 join wait"); }
        }
        Ok(())
    }

    /// hc_pre + RMSNorm with the pre record and the norm weights loaded before the PDL wait (`d41_hc_pre_norm_pw`). Only
    /// for a pre record joined from the side stream before an earlier kernel of this stream (body: every layer but 0).
    pub(super) fn pa1_pre_norm(&mut self, x: u64, pre: u64, norm: &crate::dsv41::weights::DevTensor, rows: usize) -> Result<u64> {
        let d = self.cfg.hidden;
        let y = self.f.b.take(rows * d * 2)?;
        let a = km::HcPreNormArgs { x, x_ld: (4 * d) as i64, pre, pre_stride: 24, w: norm.ptr.0, y_norm: y, yn_ld: d as i64, y_raw: 0, yr_ld: 0,
                                    w_fmt: crate::dsv41::fwd::layer::fmt_of(norm)?, dim: d as i32, rows: rows as i32, eps: self.cfg.rms_eps as f32, inv_dim: 1.0 / d as f32 };
        self.f.g.launch1("d41_hc_pre_norm_pw", (rows as u32, 1, 1), (256, 1, 1), 0, &a)?;
        receipts::kernel("mhc.pre_norm (PA1: inputs before the wait)", "d41_hc_pre_norm_pw", rows);
        Ok(y)
    }

    /// Hook (pd1_normrot): the q norm + rotated rows AND this layer's window rows in one launch (graph calls, where the
    /// window write is `d41fw_swa_rows`). Returns false when it does not apply (the caller launches normrot alone).
    pub(super) fn pa1_qkv(&mut self, l: usize, nr: &crate::dsv41::weights::exl3::dense_dec::NormRotArgs, rows: usize) -> Result<bool> {
        self.pa1.swa_done = None;
        if !self.pa1.fuse || !self.gm { return Ok(false); }
        let Some(raw) = self.pd1.kv_raw.filter(|&(ll, _)| ll == l).map(|(_, p)| p) else { return Ok(false) };
        if !self.segs.iter().all(|sg| sg.n <= super::fwd::graph_seg_max(self.cfg)) { return Ok(false); }
        let compressed = self.cfg.ratio(l) > 0;
        let sw = self.swa_args(l, raw, rows, compressed)?;
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct QkvArgs { nr: crate::dsv41::weights::exl3::dense_dec::NormRotArgs, sw: super::tpk::SwaRowsArgs }
        const _: () = assert!(std::mem::size_of::<QkvArgs>() == 104 + 64);
        let a = QkvArgs { nr: *nr, sw };
        crate::dsv41::gpu::launch_args("d41_q_kv_norm", (rows as u32, 2, 1), (256, 1, 1), 0, self.f.g.stream, &a)?;
        receipts::kernel("tp.q_norm + kv.swa_write (PA1: one launch, + wq_b / idx wq_b rotated rows)", "d41_q_kv_norm", rows);
        QKV.fetch_add(1, Relaxed);
        self.pa1.swa_done = Some(l);
        Ok(true)
    }

    /// q's RoPE can ride the attention's q load: the q and attention RoPE tables of this layer class are the same table.
    pub(super) fn pa1_qrope(&self, compressed: bool, rows: usize) -> bool {
        let (a, b) = (self.f.rope.q.of(compressed), self.f.rope.ka.of(compressed));
        self.pa1.fuse && self.win_lo == 0 && rows <= super::fwd::HEAD_MAX && a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn int12_hc_side_falls_back_when_any_rank_lacks_the_whole_tensor() {
        assert!(super::side_agreed(true, &[0.0, 0.0]));
        assert!(!super::side_agreed(true, &[0.0, 1.0]), "the node lacks it: both off");
        assert!(!super::side_agreed(true, &[1.0, 0.0]), "the head lacks it: both off");
        assert!(!super::side_agreed(false, &[0.0, 0.0]), "not asked");
    }

    #[test]
    fn dsv41_pa1_flags() {
        for f in ["dsv41-qkv-fuse", "dsv41-attn-wpf", "dsv41-attn-merge", "dsv41-attn-2cta", "dsv41-hc-side", "dsv41-attn-split"] {
            let id = crate::opts::lookup(f).unwrap_or_else(|| panic!("--{f} is not an option"));
            assert_eq!(id.def().scope, crate::opts::Scope::Spmd, "--{f} rides TpConfig to the nodes");
            assert_eq!(id.def().env, "", "--{f}: no environment variable");
        }
        assert_eq!(crate::opts::lookup("dsv41-attn-2cta").unwrap().def().def, "auto");
        // defaults: auto = TP1 / TP2 (TP4 keeps the 100-byte reduce rider unless --dsv41-hc-side on is explicitly passed)
        assert!(super::hc_side_for(1) && super::hc_side_for(2) && !super::hc_side_for(4) && !super::hc_side_for(8));
        assert!(super::hc_whole_for(2) && !super::hc_whole_for(4) && !super::hc_whole_for(1) && !super::hc_whole_for(8));
        assert!(super::side_agreed(true, &[0.0, 0.0, 0.0, 0.0]));
        assert!(!super::side_agreed(true, &[0.0, 0.0, 1.0, 0.0]));
    }
}
