//! DSV4.1 TP (leg PD1): the dense decode path of the external reference recipe (PLAN/DSV41_RX4_REPORT.md A and C),
//! for decode / verify calls (rows <= 32, the dense decode class, `Fwd::wide_min` 0):
//!
//! 1. GROUPED LAUNCHES (`--dsv41-dense-group`, default on): a layer's dense linears in 4 launches as theirs are: wq_a +
//!    wkv (+ the compressor's wkv / wgate on a KV-source layer) in ONE launch on the attention input; wq_b; every wo_a
//!    group of the rank in ONE launch; wo_b. One launch per width when a group mixes widths (`dense_dec::run_group`).
//!    Each linear's bits equal its own call (per-segment partitions).
//! 2. PRODUCER-SIDE ROTATION (`--dsv41-dense-rot`, default on): the input transform fp16(H128(x . suh)) of a linear is
//!    computed by the kernel that produces x: the q norm writes wq_b's (and the index layer's idx wq_b's) rotated rows
//!    (`dense_dec::normrot`), the attention epilogue writes wo_a's (`attn::launch_decode_rot`), wo_a's epilogue writes
//!    wo_b's (`Call::rot`). The one prep left per layer is wq_a + wkv's (theirs keeps a `rot_many` there too), plus the
//!    indexer's wk (input from the compressor kernel) and the linears whose producer belongs to another leg (MoE, head).
//! 3. THE PACED L2 PREFETCH (`--dsv41-l2-prefetch`, sizes `--dsv41-l2-prefetch-mib A,B,C,E`, pacing
//!    `--dsv41-l2-prefetch-ns-per-mib`, `--dsv41-l2-prefetch-delay-ns`): forked on its own stream after wq_b (A: this
//!    layer's wo_a groups + wo_b), after wo_b (B: the FFN mix's hc fn slice, the router gate, the shared expert's gate /
//!    up / down) and after the MoE (C: the next layer's attention mix fn slice, wq_a, wkv, compressor, wq_b; E instead when
//!    the next layer is an Engram layer: its Engram wkv; after the last layer: the head), joined to the main stream at the
//!    end of the call's body (inside the captured graph).
//!
//! Engagement: `velogb10_dsv41_fw_pd1_*` on /metrics (calls served, prefetch launches and bytes counted per executed
//! call, graph replays included) and `scripts/owner_check.py`.

use super::fwd::TpModel;
use crate::dsv41::census::classify::TensorClass as C;
use crate::dsv41::fwd::dev::{bf16, f32a};
use crate::dsv41::fwd::receipts;
use crate::dsv41::weights::exl3::dense_dec::{self, Call, Pre, RotTo};
use crate::dsv41::weights::{Accum, Act, BackendKind, DenseWeight, DevPtr, Launch, Scratch};
use anyhow::{ensure, Context, Result};
use cudarc::driver::sys;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Row ceiling for `--dsv41-l2-prefetch-after-reduce auto|rows8`: narrow 1-user / small multi-lane decode forks sites B/C
/// after reduce (`<= 8` rows), while wider verify rounds (`> 8` rows) fork before reduce so prefetch overlaps wide compute.
pub const AFTER_REDUCE_MAX_ROWS: usize = 8;

/// When sites B and C fork relative to the sublayer RDMA reduce (`--dsv41-l2-prefetch-after-reduce`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AfterReduce { Off, On, Rows8 }
impl AfterReduce {
    pub fn parse(raw: Option<&str>) -> Self {
        match raw.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
            Some("auto") | Some("rows8") => Self::Rows8,
            Some("on") => Self::On,
            _ => Self::Off,
        }
    }
    pub fn at(self, rows: usize) -> bool {
        match self {
            Self::Off => false,
            Self::On => true,
            Self::Rows8 => rows <= AFTER_REDUCE_MAX_ROWS,
        }
    }
    pub fn code(self) -> u64 {
        match self { Self::Off => 0, Self::On => 1, Self::Rows8 => 2 }
    }
    pub fn desc(self) -> &'static str {
        match self { Self::Off => "before", Self::On => "after", Self::Rows8 => "after at rows <= 8, before above" }
    }
}

/// The paced prefetch's settings (the reference recipe's constants at TP1/TP2; topology-scaled by 2/world at TP4+).
#[derive(Clone, Copy, Debug)]
pub struct PfCfg { pub on: bool, pub after_reduce: AfterReduce, pub ranks: u8, pub head_delay_ns: u64, pub mib: [u64; 4], pub ns_per_mib: u64, pub delay_ns: u64 }
pub const PF_MIB_DEFAULT: [u64; 4] = [16, 8, 16, 4];
pub const PF_MIB_TP4: [u64; 4] = [8, 4, 8, 2];
pub const PF_NS_PER_MIB_DEFAULT: u64 = 4559;
pub const PF_DELAY_NS_DEFAULT: u64 = 3000;
pub fn default_pf_mib(_world: usize) -> [u64; 4] { PF_MIB_DEFAULT }

/// What one executed call did (stashed per captured graph; added on every replay).
#[derive(Clone, Copy, Debug, Default)]
pub struct CallStats { pub dense: u64, pub pf_launches: u64, pub pf_bytes: u64 }

pub struct Pd1 {
    pub group: bool,
    pub rot: bool,
    pub pf: PfCfg,
    stream: u64,
    ev_fork: sys::CUevent,
    ev_join: sys::CUevent,
    /// a prefetch was forked in this body (the join is owed)
    forked: bool,
    /// the forward's body is running (prefetch forks only there: they are joined at its end)
    pub(super) in_body: bool,
    /// this call's counts (executed eagerly: added at the body's end; captured: stashed in the graph entry)
    pub(super) stats: CallStats,
    // per-layer hand-offs between the hooks (layer, pointer)
    pub(super) qa: Option<(usize, u64)>,
    pub(super) kv_raw: Option<(usize, u64)>,
    pub(super) comp: Option<(usize, u64, u64)>,
    pub(super) want_rot_q: bool,
    pub(super) xq: Option<(usize, u64)>,
    pub(super) xiq: Option<(usize, u64)>,
    pub(super) xo: Option<(usize, u64)>,
}

fn opt(name: &'static str) -> Option<String> { crate::opts::lookup(name).and_then(crate::opts::get) }
fn parse_mib(v: &str) -> Result<[u64; 4]> {
    let x: Vec<u64> = v.split(',').map(|s| s.trim().parse::<u64>()).collect::<std::result::Result<_, _>>().with_context(|| format!("--dsv41-l2-prefetch-mib {v}"))?;
    ensure!(x.len() == 4 && x.iter().all(|&m| m <= 64), "--dsv41-l2-prefetch-mib {v}: four sizes in MiB (A after wq_b, B after wo_b, C after the MoE, E before an Engram layer), each <= 64");
    Ok([x[0], x[1], x[2], x[3]])
}

impl Pd1 {
    pub fn new(world: usize) -> Result<Pd1> {
        let on = |n: &'static str, def: bool| opt(n).map_or(def, |v| v != "off");
        let after_reduce = AfterReduce::parse(opt("dsv41-l2-prefetch-after-reduce").as_deref());
        let pf = PfCfg {
            on: on("dsv41-l2-prefetch", PF_DEFAULT_ON),
            after_reduce,
            ranks: match opt("dsv41-l2-prefetch-ranks").as_deref() { Some("node") => 1, Some("head") => 2, _ => 0 },
            head_delay_ns: opt("dsv41-l2-prefetch-head-delay-ns").and_then(|v| v.parse().ok()).unwrap_or(0),
            mib: match opt("dsv41-l2-prefetch-mib").filter(|v| v != "auto") { Some(v) => parse_mib(&v)?, None => default_pf_mib(world) },
            ns_per_mib: opt("dsv41-l2-prefetch-ns-per-mib").and_then(|v| v.parse().ok()).unwrap_or(PF_NS_PER_MIB_DEFAULT),
            delay_ns: opt("dsv41-l2-prefetch-delay-ns").and_then(|v| v.parse().ok()).unwrap_or(PF_DELAY_NS_DEFAULT),
        };
        let (mut st, mut e1, mut e2): (sys::CUstream, sys::CUevent, sys::CUevent) = (std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut());
        if pf.on {
            // a NON-blocking stream (the reference recipe's): the prefetch kernel reads and writes no data, so cudarc's
            // NULL-stream copies cannot race it (AGENTS 2.1 is about streams that touch data)
            unsafe {
                ensure!(sys::cuStreamCreate(&mut st, sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32) == sys::CUresult::CUDA_SUCCESS, "PD1 prefetch stream");
                ensure!(sys::cuEventCreate(&mut e1, sys::CUevent_flags::CU_EVENT_DISABLE_TIMING as u32) == sys::CUresult::CUDA_SUCCESS, "PD1 fork event");
                ensure!(sys::cuEventCreate(&mut e2, sys::CUevent_flags::CU_EVENT_DISABLE_TIMING as u32) == sys::CUresult::CUDA_SUCCESS, "PD1 join event");
            }
        }
        let p = Pd1 { group: on("dsv41-dense-group", true), rot: on("dsv41-dense-rot", true), pf, stream: st as u64, ev_fork: e1, ev_join: e2, forked: false, in_body: false,
                      stats: CallStats::default(), qa: None, kv_raw: None, comp: None, want_rot_q: false, xq: None, xiq: None, xo: None };
        static SAID: std::sync::Once = std::sync::Once::new();
        SAID.call_once(|| crate::logq::send(false, p.line()));
        BOOT.store((p.group as u64) | (p.rot as u64) << 1 | (p.pf.on as u64) << 2 | 8 | (p.pf.after_reduce.code() << 4), Relaxed);
        Ok(p)
    }
    /// The boot line.
    pub fn line(&self) -> String {
        format!("[dsv41] PD1 dense decode path (rows <= 32): grouped launches {} (--dsv41-dense-group), producer-side input rotation {} (--dsv41-dense-rot), paced L2 prefetch {} (--dsv41-l2-prefetch{}: {} / {} / {} / {} MiB after wq_b / wo_b / the MoE / before an Engram layer, {} ns a MiB, {} ns delay, sites B/C {} reduce; ranks code {} (--dsv41-l2-prefetch-ranks 0 all / 1 node / 2 head), head delay {} ns (0 = same))",
                if self.group { "ON" } else { "off" }, if self.rot { "ON" } else { "off" }, if self.pf.on { "ON" } else { "off" },
                if PF_DEFAULT_ON { ", default on" } else { ", default off" }, self.pf.mib[0], self.pf.mib[1], self.pf.mib[2], self.pf.mib[3], self.pf.ns_per_mib, self.pf.delay_ns,
                self.pf.after_reduce.desc(), self.pf.ranks, self.pf.head_delay_ns)
    }
    /// Reset the per-layer hand-offs (a hook that did not run leaves nothing behind for the next layer).
    pub(super) fn clear(&mut self) { self.qa = None; self.kv_raw = None; self.comp = None; self.want_rot_q = false; self.xq = None; self.xiq = None; self.xo = None; }
}
impl Drop for Pd1 {
    fn drop(&mut self) {
        unsafe {
            if !self.ev_fork.is_null() { sys::cuEventDestroy_v2(self.ev_fork); }
            if !self.ev_join.is_null() { sys::cuEventDestroy_v2(self.ev_join); }
            if self.stream != 0 { sys::cuStreamDestroy_v2(self.stream as sys::CUstream); }
        }
    }
}

/// The default of `--dsv41-l2-prefetch` (decided by the measurement in PLAN/DSV41_PD1_REPORT.md).
pub const PF_DEFAULT_ON: bool = true;

// ---- /metrics (executed calls: eager calls at their end, captured calls on every replay) ----
static BOOT: AtomicU64 = AtomicU64::new(0);
static CALLS: AtomicU64 = AtomicU64::new(0);
static PF_LAUNCHES: AtomicU64 = AtomicU64::new(0);
static PF_BYTES: AtomicU64 = AtomicU64::new(0);
static PF_CALLS: AtomicU64 = AtomicU64::new(0);
pub fn add(s: &CallStats) {
    if s.dense > 0 { CALLS.fetch_add(1, Relaxed); }
    if s.pf_launches > 0 { PF_CALLS.fetch_add(1, Relaxed); PF_LAUNCHES.fetch_add(s.pf_launches, Relaxed); PF_BYTES.fetch_add(s.pf_bytes, Relaxed); }
}
pub fn append(o: &mut String) {
    use std::fmt::Write as _;
    let b = BOOT.load(Relaxed);
    if b == 0 { return; }
    let rows: [(&str, &str, u64); 8] = [
        ("velogb10_dsv41_fw_pd1_dense_group", "1 = PD1 grouped dense launches on (--dsv41-dense-group)", b & 1),
        ("velogb10_dsv41_fw_pd1_dense_rot", "1 = PD1 producer-side input rotation on (--dsv41-dense-rot)", (b >> 1) & 1),
        ("velogb10_dsv41_fw_pd1_l2pf", "1 = PD1 paced L2 prefetch on (--dsv41-l2-prefetch)", (b >> 2) & 1),
        ("velogb10_dsv41_fw_pd1_l2pf_after_reduce_mode", "PD1 sites B/C fork timing (--dsv41-l2-prefetch-after-reduce: 0 = off/before, 1 = on/after, 2 = auto/rows<=8)", (b >> 4) & 3),
        ("velogb10_dsv41_fw_pd1_calls_total", "decode / verify calls executed on the PD1 dense path (graph replays included)", CALLS.load(Relaxed)),
        ("velogb10_dsv41_fw_pd1_l2pf_calls_total", "decode / verify calls executed with paced L2 prefetch launches", PF_CALLS.load(Relaxed)),
        ("velogb10_dsv41_fw_pd1_l2pf_launches_total", "paced L2 prefetch launches executed (graph replays included)", PF_LAUNCHES.load(Relaxed)),
        ("velogb10_dsv41_fw_pd1_l2pf_bytes_total", "bytes those launches asked into L2", PF_BYTES.load(Relaxed)),
    ];
    for (n, h, v) in rows { let _ = writeln!(o, "# HELP {n} {h}\n# TYPE {n} counter\n{n} {v}"); }
}

fn exl3_dd<'w>(w: &'w (DenseWeight, BackendKind), rows: usize) -> Option<&'w DenseWeight> {
    (w.1 == BackendKind::Exl3 && dense_dec::eligible(&w.0, rows)).then_some(&w.0)
}

impl TpModel<'_> {
    fn pd1_l(&self) -> Launch { self.f.g.l() }
    fn pd1_scr(&self) -> Scratch { Scratch { ptr: DevPtr(self.f.scr.ptr), bytes: self.f.scr.bytes } }
    /// A decode / verify call of the dense decode class.
    pub(super) fn pd1_dd(&self, rows: usize) -> bool { (1..=32).contains(&rows) && self.f.wide_min == 0 }

    /// Hook (body, before q_lora): wq_a + wkv (+ the compressor's projections on a KV-source layer) in ONE grouped launch
    /// on the attention input; q_lora / window / compressor then take their outputs.
    pub(super) fn pd1_front(&mut self, l: usize, ain: u64, rows: usize, comp: bool) -> Result<()> {
        self.pd1.clear();
        if !self.pd1.group || !self.pd1_dd(rows) { return Ok(()); }
        let (d, hd, ql) = (self.cfg.hidden, self.cfg.head_dim, self.cfg.q_lora_rank);
        let ratio = self.cfg.ratio(l);
        let lw = &self.layers[l].as_ref().unwrap().w;
        let (Some(wqa), Some(wkv)) = (exl3_dd(lw.lin(C::AttnWqA)?, rows), exl3_dd(lw.lin(C::AttnWkv)?, rows)) else { return Ok(()) };
        let (wqa, wkv) = (wqa.clone(), wkv.clone());
        let cw = if comp {
            let a = exl3_dd(lw.lin(C::CompWkv)?, rows).cloned();
            let b = if ratio >= 2 { exl3_dd(lw.lin(C::CompWgate)?, rows).cloned().map(Some) } else { Some(None) };
            match (a, b) { (Some(a), Some(b)) => Some((a, b)), _ => return Ok(()) }
        } else { None };
        let x = bf16(ain, rows, d, d);
        let qa = self.f.b.take(rows * ql * 2)?;
        let raw = self.f.b.take(rows * hd * 2)?;
        let (ya, yk) = (bf16(qa, rows, ql, ql), bf16(raw, rows, hd, hd));
        let mut outs: Vec<(DenseWeight, Act)> = vec![(wqa, ya), (wkv, yk)];
        let mut cp = None;
        if let Some((cwkv, cwg)) = cw {
            if ratio >= 2 {
                let (kvp, scp) = (self.f.b.take(rows * hd * 4)?, self.f.b.take(rows * hd * 4)?);
                outs.push((cwkv, f32a(kvp, rows, hd, hd)));
                outs.push((cwg.context("compressor wgate")?, f32a(scp, rows, hd, hd)));
                cp = Some((kvp, scp));
            } else {
                let kvp = self.f.b.take(rows * hd * 2)?;
                outs.push((cwkv, bf16(kvp, rows, hd, hd)));
                cp = Some((kvp, 0));
            }
        }
        let calls: Vec<Call> = outs.iter().map(|(w, y)| Call::new(w, &x, y, Accum::Overwrite)).collect();
        dense_dec::run_group(&self.pd1_l(), &calls, self.pd1_scr())?;
        receipts::kernel("tp.attn.front (PD1: wq_a + wkv [+ compressor] in one launch)", "d41x_dd_prep+d41x_dd_a*_k* (grouped)", rows);
        self.pd1.stats.dense += 1;
        self.pd1.qa = Some((l, qa));
        self.pd1.kv_raw = Some((l, raw));
        if let Some((a, b)) = cp { self.pd1.comp = Some((l, a, b)); }
        self.pd1.want_rot_q = self.pd1.rot;
        Ok(())
    }

    /// Hook (q_lora): the q norm that also writes wq_b's (and this index layer's idx wq_b's) rotated rows. Returns false when
    /// it did not run (the caller runs the plain norm).
    pub(super) fn pd1_normrot(&mut self, l: usize, qa: u64, y: u64, rows: usize) -> Result<bool> {
        if !std::mem::take(&mut self.pd1.want_rot_q) || !self.pd1_dd(rows) { return Ok(false); }
        let ql = self.cfg.q_lora_rank;
        let lay = self.layers[l].as_ref().unwrap();
        let Some(wqb) = exl3_dd(&lay.wq_b, rows) else { return Ok(false) };
        let mut targets = vec![];
        let (su, k) = dense_dec::suh_of(wqb)?;
        ensure!(k == ql, "layer {l}: wq_b input {k} != q_lora_rank {ql}");
        let xq = self.f.b.take(rows * ql * 2)?;
        targets.push((su, xq, ql));
        let mut xiq = None;
        let index_layer = self.cfg.ratio(l) > 0 && self.cfg.index_source_of(l) == Some(l);
        if index_layer {
            if let Ok(w) = lay.w.lin(C::IdxWqB) {
                if let Some(wi) = exl3_dd(w, rows) {
                    let (su2, k2) = dense_dec::suh_of(wi)?;
                    if k2 == ql { let p = self.f.b.take(rows * ql * 2)?; targets.push((su2, p, ql)); xiq = Some(p); }
                }
            }
        }
        let w = lay.w.param(C::AttnQNorm)?;
        let fmt = crate::dsv41::fwd::layer::fmt_of(w)?;
        let wp = w.ptr.0;
        let na = dense_dec::normrot_args(qa, ql, wp, fmt, y, ql, ql, rows, self.cfg.rms_eps as f32, &targets)?;
        if !self.pa1_qkv(l, &na, rows)? {   // PA1: + this layer's window rows in the same launch (graph calls)
            dense_dec::normrot(&self.pd1_l(), qa, ql, wp, fmt, y, ql, ql, rows, self.cfg.rms_eps as f32, &targets)?;
            receipts::kernel("tp.q_norm (PD1: + wq_b / idx wq_b rotated rows)", "d41x_dd_normrot", rows);
        }
        self.pd1.xq = Some((l, xq));
        if let Some(p) = xiq { self.pd1.xiq = Some((l, p)); }
        Ok(true)
    }

    /// A pre-rotated linear call (the hand-off `xh` written by its producer).
    pub(super) fn pd1_pre_call(&mut self, what: &'static str, w: &DenseWeight, x: &Act, y: &Act, xh: u64) -> Result<()> {
        let mut c = Call::new(w, x, y, Accum::Overwrite);
        c.pre = Some(Pre { xh, ld: x.cols });
        dense_dec::run(&self.pd1_l(), &[c], self.pd1_scr())?;
        receipts::kernel(what, "d41x_dd_a*_k* (pre-rotated input: no prep)", x.rows);
        Ok(())
    }

    /// Hook (attn_core): the decode attention with wo_a's rotated rows written by its epilogue. Returns false when it did
    /// not run (the caller launches the plain decode).
    pub(super) fn pd1_attn(&mut self, l: usize, call: &crate::dsv41::gpu::attn::AttnCall, s: &crate::dsv41::gpu::attn::DecodeScratch, rows: usize, x: u32) -> Result<bool> {
        self.pd1.xo = None;
        if !self.pd1.group || !self.pd1.rot || !self.pd1_dd(rows) || self.win_lo > 0 { return Ok(false); }
        let (nh, ng, hd) = (self.rg.nh, self.rg.ng, self.cfg.head_dim);
        if ng == 0 || nh % ng != 0 || ng > 8 { return Ok(false); }
        let lw = &self.layers[l].as_ref().unwrap().w;
        let mut suh = Vec::with_capacity(ng);
        for i in 0..ng {
            let Some(w) = lw.lin_sub(C::AttnWoA, (self.rg.g0 + i) as u16) else { return Ok(false) };
            let Some(w) = exl3_dd(w, rows) else { return Ok(false) };
            let (su, k) = dense_dec::suh_of(w)?;
            if k != nh * hd / ng { return Ok(false); }
            suh.push(su);
        }
        let xo = self.f.b.take(rows * nh * hd * 2)?;
        // PA1 flags `x` (window prefetch, separate merge, q RoPE in the q load): same outputs and rotated rows
        crate::dsv41::gpu::attn::launch_decode_x(self.f.g.l(), call, s, Some(&crate::dsv41::gpu::attn::RotOut { suh: &suh, hpg: nh / ng, xh: xo, ld: nh * hd }), x)?;
        super::pa1::Pa1::count_attn(x);
        receipts::kernel("tp.attn.sparse (heads; PD1: + wo_a rotated rows)", if x & crate::dsv41::gpu::attn::X_DEFER != 0 { "d41_attn_decode(rot_out)+d41_attn_merge" } else { "d41_attn_decode(rot_out)" }, rows);
        self.pd1.xo = Some((l, xo));
        Ok(true)
    }

    /// Hook (wo_partial): every wo_a group in ONE launch (input from the attention epilogue when it wrote it), its epilogue
    /// rotating into wo_b's input, then wo_b without a prep. Returns false when it did not run.
    pub(super) fn pd1_wo(&mut self, l: usize, out: u64, rows: usize, y: u64) -> Result<bool> {
        let xo = self.pd1.xo.take().filter(|&(ll, _)| ll == l).map(|(_, p)| p);
        if !self.pd1.group || !self.pd1_dd(rows) { return Ok(false); }
        let (hd, d) = (self.cfg.head_dim, self.cfg.hidden);
        let (nh, ng, olr) = (self.rg.nh, self.rg.ng, self.cfg.o_lora_rank);
        let gin = nh * hd / ng.max(1);
        let lay = self.layers[l].as_ref().unwrap();
        let Some(wob) = exl3_dd(&lay.wo_b, rows).cloned() else { return Ok(false) };
        let mut was = Vec::with_capacity(ng);
        for i in 0..ng {
            let Some(w) = lay.w.lin_sub(C::AttnWoA, (self.rg.g0 + i) as u16) else { return Ok(false) };
            let Some(w) = exl3_dd(w, rows) else { return Ok(false) };
            was.push(w.clone());
        }
        let rot = self.pd1.rot && dense_dec::bits_of(&wob).is_ok();
        let (wsu, wk) = dense_dec::suh_of(&wob)?;
        ensure!(wk == ng * olr, "layer {l}: wo_b input {wk} != {ng} groups x {olr}");
        let mid = self.f.b.take(rows * ng * olr * 2)?;
        let xb = if rot { self.f.b.take(rows * ng * olr * 2)? } else { 0 };
        let xs: Vec<Act> = (0..ng).map(|i| bf16(out + (i * gin * 2) as u64, rows, gin, nh * hd)).collect();
        let ys: Vec<Act> = (0..ng).map(|i| bf16(mid + (i * olr * 2) as u64, rows, olr, ng * olr)).collect();
        let calls: Vec<Call> = (0..ng).map(|i| {
            let mut c = Call::new(&was[i], &xs[i], &ys[i], Accum::Overwrite);
            if let Some(p) = xo { c.pre = Some(Pre { xh: p + (i * gin * 2) as u64, ld: nh * hd }); }
            if rot { c.rot = Some(RotTo { suh: wsu + (i * olr * 2) as u64, xh: xb + (i * olr * 2) as u64, ld: ng * olr }); }
            c
        }).collect();
        dense_dec::run_group(&self.pd1_l(), &calls, self.pd1_scr())?;
        receipts::kernel("tp.attn.wo_a (PD1: every group in one launch)", "d41x_dd_a*_k* (grouped)", rows);
        let xm = bf16(mid, rows, ng * olr, ng * olr);
        let mut ya = f32a(y, rows, d, d);
        ya.cols = d;
        let mut c = Call::new(&wob, &xm, &ya, Accum::Overwrite);
        if rot { c.pre = Some(Pre { xh: xb, ld: ng * olr }); }
        dense_dec::run(&self.pd1_l(), &[c], self.pd1_scr())?;
        receipts::kernel("tp.attn.wo_b (PD1: input rotated by wo_a's epilogue)", "d41x_dd_a*_k*", rows);
        self.pd1.stats.dense += 1;
        Ok(true)
    }

    // ------------------------------------------------------------------ the paced L2 prefetch
    /// Fork one paced prefetch launch at `site` (0 = after wq_b, 1 = after wo_b, 2 = after the MoE) of layer `l`.
    pub(super) fn pd1_pf(&mut self, site: usize, l: usize, rows: usize) -> Result<()> {
        if !self.pd1.pf.on || !self.pd1.in_body || !self.pd1_dd(rows) { return Ok(()); }
        // XL1: rank-asymmetric arrangements (0 = all, 1 = node only, 2 = head only)
        let head = self.rank == 0;
        if (self.pd1.pf.ranks == 1 && head) || (self.pd1.pf.ranks == 2 && !head) { return Ok(()); }
        let delay_ns = if head && self.pd1.pf.head_delay_ns > 0 { self.pd1.pf.head_delay_ns } else { self.pd1.pf.delay_ns };
        let (ranges, budget) = self.pd1_pf_ranges(site, l)?;
        if ranges.is_empty() || budget == 0 { return Ok(()); }
        let main = self.f.g.stream as sys::CUstream;
        let pf = self.pd1.stream as sys::CUstream;
        unsafe {
            ensure!(sys::cuEventRecord(self.pd1.ev_fork, main) == sys::CUresult::CUDA_SUCCESS, "PD1 fork record");
            ensure!(sys::cuStreamWaitEvent(pf, self.pd1.ev_fork, 0) == sys::CUresult::CUDA_SUCCESS, "PD1 fork wait");
        }
        let b = dense_dec::l2_paced(self.pd1.stream, &ranges, budget, delay_ns, self.pd1.pf.ns_per_mib)?;
        unsafe { ensure!(sys::cuEventRecord(self.pd1.ev_join, pf) == sys::CUresult::CUDA_SUCCESS, "PD1 join record"); }
        self.pd1.forked = true;
        self.pd1.stats.pf_launches += 1;
        self.pd1.stats.pf_bytes += b;
        Ok(())
    }

    /// The end of a body: the main stream waits for the prefetch stream (inside the capture, so the graph is joined), then
    /// the call's counts (an eager call: now; a captured call: the graph entry keeps them for its replays).
    pub(super) fn pd1_join(&mut self) -> Result<()> {
        if std::mem::take(&mut self.pd1.forked) {
            unsafe { ensure!(sys::cuStreamWaitEvent(self.f.g.stream as sys::CUstream, self.pd1.ev_join, 0) == sys::CUresult::CUDA_SUCCESS, "PD1 join wait"); }
        }
        self.pd1.in_body = false;
        self.pd1.clear();
        if !self.gm { add(&self.pd1.stats); }
        Ok(())
    }

    /// The ranges of a site, in the order the next kernels read them, and its budget (bytes).
    fn pd1_pf_ranges(&self, site: usize, l: usize) -> Result<(Vec<(u64, u64, u64, u64)>, u64)> {
        let mib = |i: usize| self.pd1.pf.mib[i] << 20;
        let lay = |l: usize| self.layers.get(l).and_then(|x| x.as_ref());
        let lin = |w: &(DenseWeight, BackendKind)| -> Option<(u64, u64, u64, u64)> { if w.1 == BackendKind::Exl3 { dense_dec::weight_range(&w.0).ok() } else { None } };
        let fnr = |l: usize, c: C| -> Option<(u64, u64, u64, u64)> {
            let t = lay(l)?.w.param(c).ok()?;
            let mk = if c == C::HcAttnFn { lay(l)?.mix_a } else { lay(l)?.mix_f };
            let rows = t.shape.first().copied()? as u64;
            Some(if mk.fn_ld == mk.klen { (mk.fn_ptr, rows * mk.klen as u64 * 4, rows * mk.klen as u64 * 4, 1) } else { (mk.fn_ptr, mk.klen as u64 * 4, mk.fn_ld as u64 * 4, rows) })
        };
        let mut r = Vec::new();
        let Some(ly) = lay(l) else { return Ok((r, 0)) };
        match site {
            0 => {
                for i in 0..self.rg.ng { if let Some(w) = ly.w.lin_sub(C::AttnWoA, (self.rg.g0 + i) as u16) { r.extend(lin(w)); } }
                r.extend(lin(&ly.wo_b));
                Ok((r, mib(0)))
            }
            1 => {
                r.extend(fnr(l, C::HcFfnFn));
                if let Ok(t) = ly.w.weight_tensor(C::GateWeight) { r.push((t.ptr.0, t.bytes, t.bytes, 1)); }
                for i in [0usize, 2, 1] {   // the shared expert's gate (w1), up (w3), down (w2): the order the MoE reads them
                    if let Ok(x) = dense_dec::weight_range(&ly.shared.experts[0][i]) { r.push(x); }
                }
                Ok((r, mib(1)))
            }
            _ => {
                let n = l + 1;
                if n >= self.cfg.layers {
                    r.extend(lin(&self.glob.head));
                    return Ok((r, mib(2)));
                }
                if self.cfg.engram_layers.contains(&n) {
                    for c in &ly.engram { r.extend(lin(&c.w)); }
                    if let Some(nl) = lay(n) { for c in &nl.engram { r.extend(lin(&c.w)); } }
                    return Ok((r, mib(3)));
                }
                let Some(nl) = lay(n) else { return Ok((r, 0)) };
                r.extend(fnr(n, C::HcAttnFn));
                for c in [C::AttnWqA, C::AttnWkv] { if let Ok(w) = nl.w.lin(c) { r.extend(lin(w)); } }
                if self.cfg.ratio(n) > 0 && self.cfg.kv_source_of(n) == Some(n) {
                    for c in [C::CompWkv, C::CompWgate] { if let Ok(w) = nl.w.lin(c) { r.extend(lin(w)); } }
                }
                r.extend(lin(&nl.wq_b));
                Ok((r, mib(2)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dsv41_pd1_prefetch_flags() {
        assert_eq!(parse_mib("16,8,16,4").unwrap(), [16, 8, 16, 4]);
        assert_eq!(parse_mib(" 8, 4 ,8,2").unwrap(), [8, 4, 8, 2]);
        assert!(parse_mib("16,8,16").is_err());
        assert!(parse_mib("16,8,16,65").is_err());
        assert!(parse_mib("a,b,c,d").is_err());
        // the reference recipe's pacing: 1 MiB per 4,559 ns = 230 GB/s, 3 us delay
        assert_eq!(PF_NS_PER_MIB_DEFAULT, (128 * 8192 / 230) as u64);
        assert_eq!(PF_DELAY_NS_DEFAULT, 3000);
        assert_eq!(PF_MIB_DEFAULT, [16, 8, 16, 4]);
        assert_eq!(default_pf_mib(2), [16, 8, 16, 4]);
        assert_eq!(default_pf_mib(4), [16, 8, 16, 4]);
        assert_eq!(AfterReduce::parse(None), AfterReduce::Off);
        assert_eq!(AfterReduce::parse(Some("auto")), AfterReduce::Rows8);
        assert_eq!(AfterReduce::parse(Some("rows8")), AfterReduce::Rows8);
        assert_eq!(AfterReduce::parse(Some("on")), AfterReduce::On);
        assert_eq!(AfterReduce::parse(Some(" OFF ")), AfterReduce::Off);
        for r in 1..=8 { assert!(AfterReduce::Rows8.at(r) && AfterReduce::On.at(r) && !AfterReduce::Off.at(r)); }
        for r in [9usize, 12, 16, 20, 24, 32] { assert!(!AfterReduce::Rows8.at(r) && AfterReduce::On.at(r) && !AfterReduce::Off.at(r)); }
        for f in ["dsv41-l2-prefetch", "dsv41-l2-prefetch-mib", "dsv41-l2-prefetch-after-reduce", "dsv41-l2-prefetch-ns-per-mib", "dsv41-l2-prefetch-delay-ns", "dsv41-l2-prefetch-ranks", "dsv41-l2-prefetch-head-delay-ns", "dsv41-dense-group", "dsv41-dense-rot"] {
            let id = crate::opts::lookup(f).unwrap_or_else(|| panic!("--{f} is not an option"));
            assert_eq!(id.def().scope, crate::opts::Scope::Spmd, "--{f} rides TpConfig to the nodes");
            assert_eq!(id.def().env, "", "--{f}: no environment variable");
        }
    }
}
