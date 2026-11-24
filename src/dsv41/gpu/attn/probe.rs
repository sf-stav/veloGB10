//! `--probe-dsv41-kernel --dsv41-kernel-set attn` (leg KA): the sparse attention core on the GPU against its
//! CPU references, on synthetic caches built in the engine's own formats:
//!   1. ABI: the host mirror of the argument structs equals the device's (sizes, offsets, constants);
//!   2. correctness: decode and prefill entries vs an exact f64 attention (rel-L2 per row) and vs the
//!      reference kernel's own arithmetic (block-64 online softmax, bf16 P), at the TP head ranges
//!      64 / 32 / 16 / 8, early rows (window partly before position 0), short / holed / padded selections,
//!      two lanes, ring and linear window sources, DSpark extra keys;
//!   3. invariance, bit for bit: decode == prefill; a row alone == the row in a 24-row call; heads of a
//!      64-head call == the same heads computed at 32 / 16 / 8 heads (TP); ring == linear window; a padded
//!      selection == the trimmed one; 200 repeated launches identical;
//!   4. the epilogue (inverse RoPE incl. the large-angle path, act_quant codes and scales) bit-exact against
//!      its CPU port applied to the kernel's own raw output;
//!   5. the error word for an out-of-range selection.
//! Then, when asked, the REFb fixtures (`attn/fixture.rs`) and the timing table (`attn/timing.rs`).
//! RESULT: DSV41_KERNEL_ATTN_OK only when every check holds.

use super::reference as cref;
use super::{AttnCall, DecodeScratch, LaneSrc, EPI_AQ, EPI_ROPE};
use crate::dsv41::weights::{DevPtr, Launch};
use anyhow::{bail, ensure, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DeviceRepr, LaunchAsync, LaunchConfig};
use std::sync::Arc;

/// A deterministic generator (xorshift64*).
pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 { self.0 ^= self.0 >> 12; self.0 ^= self.0 << 25; self.0 ^= self.0 >> 27; self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) }
    pub fn unif(&mut self) -> f32 { (self.next() >> 40) as f32 / (1u64 << 24) as f32 }
    pub fn gauss(&mut self) -> f32 { let (a, b) = (self.unif().max(1e-7), self.unif()); (-2.0 * a.ln()).sqrt() * (6.283_185_3 * b).cos() }
    pub fn below(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
}

/// Device buffers kept alive for a call.
pub struct Bufs { pub dev: Arc<CudaDevice>, keep_u8: Vec<CudaSlice<u8>>, keep_u16: Vec<CudaSlice<u16>>, keep_i32: Vec<CudaSlice<i32>>, keep_f32: Vec<CudaSlice<f32>> }
impl Bufs {
    pub fn new(dev: &Arc<CudaDevice>) -> Self { Bufs { dev: dev.clone(), keep_u8: vec![], keep_u16: vec![], keep_i32: vec![], keep_f32: vec![] } }
    fn ptr<T: DeviceRepr>(s: &CudaSlice<T>) -> DevPtr { DevPtr(*s.device_ptr()) }
    pub fn u8(&mut self, v: &[u8]) -> Result<DevPtr> { let s = self.dev.htod_sync_copy(if v.is_empty() { &[0u8][..] } else { v })?; let p = Self::ptr(&s); self.keep_u8.push(s); Ok(p) }
    pub fn u16(&mut self, v: &[u16]) -> Result<DevPtr> { let s = self.dev.htod_sync_copy(if v.is_empty() { &[0u16][..] } else { v })?; let p = Self::ptr(&s); self.keep_u16.push(s); Ok(p) }
    pub fn i32(&mut self, v: &[i32]) -> Result<DevPtr> { let s = self.dev.htod_sync_copy(if v.is_empty() { &[0i32][..] } else { v })?; let p = Self::ptr(&s); self.keep_i32.push(s); Ok(p) }
    pub fn f32(&mut self, v: &[f32]) -> Result<DevPtr> { let s = self.dev.htod_sync_copy(if v.is_empty() { &[0f32][..] } else { v })?; let p = Self::ptr(&s); self.keep_f32.push(s); Ok(p) }
    /// read back the most recent u16 / u8 / i32 buffer whose pointer is `p`
    pub fn get_u16(&self, p: DevPtr) -> Result<Vec<u16>> { let s = self.keep_u16.iter().find(|s| Self::ptr(s) == p).context("u16 buffer")?; Ok(self.dev.dtoh_sync_copy(s)?) }
    pub fn get_u8(&self, p: DevPtr) -> Result<Vec<u8>> { let s = self.keep_u8.iter().find(|s| Self::ptr(s) == p).context("u8 buffer")?; Ok(self.dev.dtoh_sync_copy(s)?) }
    pub fn get_i32(&self, p: DevPtr) -> Result<Vec<i32>> { let s = self.keep_i32.iter().find(|s| Self::ptr(s) == p).context("i32 buffer")?; Ok(self.dev.dtoh_sync_copy(s)?) }
}

/// One lane's synthetic caches: every position's window KV (FP8 planes) and a compressed cache.
pub struct HostLane {
    pub maxpos: usize,
    pub win_q: Vec<u8>, pub win_s: Vec<u8>, pub win: Vec<Vec<f32>>,
    pub kv: Vec<u8>, pub comp: Vec<Vec<f32>>,
    pub ext_q: Vec<u8>, pub ext_s: Vec<u8>, pub ext: Vec<Vec<f32>>,
}

pub fn kv_vec(rng: &mut Rng, amp: f32) -> Vec<f32> {
    (0..512).map(|d| rng.gauss() * amp * if d % 37 == 0 { 8.0 } else { 1.0 }).collect()
}

impl HostLane {
    pub fn build(rng: &mut Rng, maxpos: usize, ncomp: usize, n_ext: usize) -> HostLane {
        let mut l = HostLane { maxpos, win_q: vec![], win_s: vec![], win: vec![], kv: vec![], comp: vec![], ext_q: vec![], ext_s: vec![], ext: vec![] };
        for _ in 0..=maxpos {
            let (q, s) = cref::act_quant_fp8(&kv_vec(rng, 1.0));
            l.win.push(cref::fp8_row_decode(&q, &s));
            l.win_q.extend_from_slice(&q);
            l.win_s.extend_from_slice(&s);
        }
        for _ in 0..ncomp {
            let r = cref::kv_record_encode(&kv_vec(rng, 1.0));
            l.comp.push(cref::kv_record_decode(&r));
            l.kv.extend_from_slice(&r);
        }
        for _ in 0..n_ext {
            let (q, s) = cref::act_quant_fp8(&kv_vec(rng, 1.0));
            l.ext.push(cref::fp8_row_decode(&q, &s));
            l.ext_q.extend_from_slice(&q);
            l.ext_s.extend_from_slice(&s);
        }
        l
    }
    /// A ring of `slots` slots holding positions (last - slots, last] (slot = position % slots).
    pub fn ring(&self, last: usize, slots: usize) -> (Vec<u8>, Vec<u8>) {
        let (mut q, mut s) = (vec![0u8; slots * 512], vec![127u8; slots * 16]);
        for p in last.saturating_sub(slots - 1)..=last {
            let k = p % slots;
            q[k * 512..k * 512 + 512].copy_from_slice(&self.win_q[p * 512..p * 512 + 512]);
            s[k * 16..k * 16 + 16].copy_from_slice(&self.win_s[p * 16..p * 16 + 16]);
        }
        (q, s)
    }
}

/// How a lane serves its window in a call.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WinSrc { Ring { last: usize }, Lin { base: usize }, Both { last: usize, base: usize } }

/// One synthetic call: rows, their lanes, selections, q, sinks.
pub struct Case {
    pub heads: usize,
    pub pos: Vec<i32>, pub wend: Vec<i32>, pub lane: Vec<i32>,
    pub sel: Vec<i32>, pub nsel: Vec<i32>, pub sel_cap: usize,
    pub q: Vec<u16>, pub sink: Vec<f32>,
    pub use_kv: bool, pub use_ext: bool,
    pub win: Vec<WinSrc>,
    pub freq: [f32; 32],
}

impl Case {
    pub fn rows(&self) -> usize { self.pos.len() }
    /// the canonical key list of row r (window by position, then the selection, then the extra keys)
    pub fn keys<'a>(&self, lanes: &'a [HostLane], r: usize) -> Vec<&'a [f32]> {
        let l = &lanes[self.lane[r] as usize];
        let mut k: Vec<&[f32]> = Vec::new();
        for i in 0..128i64 {
            let w = self.wend[r] as i64 - 127 + i;
            if w >= 0 { k.push(&l.win[w as usize]); }
        }
        if self.use_kv {
            for e in 0..(self.nsel[r] as usize).min(self.sel_cap) {
                let x = self.sel[r * self.sel_cap + e];
                if x >= 0 && (x as usize) < l.comp.len() { k.push(&l.comp[x as usize]); }
            }
        }
        if self.use_ext { for e in &l.ext { k.push(e); } }
        k
    }
    /// the reference's decode-layout index list (window slots oldest first with -1 before position 0)
    pub fn ref_list<'a>(&self, lanes: &'a [HostLane], r: usize) -> Vec<Option<&'a [f32]>> {
        let l = &lanes[self.lane[r] as usize];
        let mut k: Vec<Option<&[f32]>> = (0..128i64).map(|i| { let w = self.wend[r] as i64 - 127 + i; if w >= 0 { Some(l.win[w as usize].as_slice()) } else { None } }).collect();
        if self.use_kv {
            for e in 0..(self.nsel[r] as usize).min(self.sel_cap) {
                let x = self.sel[r * self.sel_cap + e];
                k.push(if x >= 0 && (x as usize) < l.comp.len() { Some(l.comp[x as usize].as_slice()) } else { None });
            }
        }
        if self.use_ext { for e in &l.ext { k.push(Some(e.as_slice())); } }
        k
    }
    pub fn qrow(&self, r: usize) -> Vec<f32> { self.q[r * self.heads * 512..(r + 1) * self.heads * 512].iter().map(|b| cref::bf16_f32(*b)).collect() }
}

/// The result of one launch: bf16 out, E4M3 codes, E8M0 bytes, the error word.
pub struct Out { pub o: Vec<u16>, pub q: Vec<u8>, pub s: Vec<u8>, pub err: i32 }

/// The probe's compute stream: one BLOCKING stream (CU_STREAM_DEFAULT, AGENTS 2.1) for the process, so
/// cudarc's NULL-stream copies order against it both ways.
pub fn launch_of(dev: &Arc<CudaDevice>) -> Launch {
    static S: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let s = *S.get_or_init(|| {
        let _ = dev.bind_to_thread();
        cudarc::driver::result::stream::create(cudarc::driver::result::stream::StreamKind::Default).map(|s| s as u64).unwrap_or(0)
    });
    Launch { stream: s, pdl: false }
}

/// Wait for the probe's stream.
pub fn sync(dev: &Arc<CudaDevice>) -> Result<()> {
    let r = unsafe { cudarc::driver::sys::cuStreamSynchronize(launch_of(dev).stream as cudarc::driver::sys::CUstream) };
    ensure!(r == cudarc::driver::sys::CUresult::CUDA_SUCCESS, "stream synchronize: {r:?}");
    Ok(())
}

/// A case uploaded once (caches, rows, outputs, the decode scratch): launch it any number of times.
pub struct Prepared { pub b: Bufs, pub call: AttnCall, pub scratch: DecodeScratch, out: DevPtr, oq: DevPtr, os: DevPtr, err: DevPtr, cnt: DevPtr }

pub fn prepare(dev: &Arc<CudaDevice>, lanes: &[HostLane], c: &Case) -> Result<Prepared> {
    let mut b = Bufs::new(dev);
    let mut srcs = Vec::new();
    for (i, l) in lanes.iter().enumerate() {
        let w = c.win[i];
        let (rq, rs) = match w { WinSrc::Ring { last } | WinSrc::Both { last, .. } => l.ring(last, 134), WinSrc::Lin { .. } => (vec![0u8; 512], vec![127u8; 16]) };
        let lin = match w {
            WinSrc::Lin { base } | WinSrc::Both { base, .. } => Some((b.u8(&l.win_q[base * 512..])?, b.u8(&l.win_s[base * 16..])?, base as i64)),
            WinSrc::Ring { .. } => None,
        };
        let swa_q = b.u8(&rq)?;
        let swa_s = b.u8(&rs)?;
        let kv = if c.use_kv && !l.comp.is_empty() { Some((b.u8(&l.kv)?, l.comp.len())) } else { None };
        let ext = if c.use_ext && !l.ext.is_empty() { Some((b.u8(&l.ext_q)?, b.u8(&l.ext_s)?, l.ext.len())) } else { None };
        srcs.push(LaneSrc { swa_q, swa_s, slots: 134, lin, kv, ext });
    }
    let rows = c.rows();
    let n = rows * c.heads * 512;
    let out = b.u16(&vec![0xFFFFu16; n])?;
    let oq = b.u8(&vec![0u8; n])?;
    let os = b.u8(&vec![0u8; n / 32])?;
    let err = b.i32(&[0])?;
    let call = AttnCall {
        rows, heads: c.heads, q: b.u16(&c.q)?, q_stride: c.heads * 512, sink: b.f32(&c.sink)?, pos: b.i32(&c.pos)?,
        win_end: b.i32(&c.wend)?, lane: b.i32(&c.lane)?, sel: if c.use_kv { b.i32(&c.sel)? } else { DevPtr(0) }, sel_stride: c.sel_cap,
        sel_cap: c.sel_cap, n_sel: b.i32(&c.nsel)?, lanes: srcs, out, out_stride: c.heads * 512,
        out_q: Some((oq, c.heads * 512)), out_s: Some((os, c.heads * 16)), epi: 0, rope_freq: c.freq, err,
    };
    let ns = call.nsplit().max(super::RECIPE_SPLITS);   // PA1: the recipe rule's 8 split slots too
    let (pb, cb) = DecodeScratch::bytes(rows, c.heads, ns);
    let part = b.f32(&vec![0f32; pb / 4])?;
    let cnt = b.i32(&vec![0i32; cb / 4])?;
    let scratch = DecodeScratch { part, cnt, rows, groups: c.heads.div_ceil(16), nsplit: ns };
    Ok(Prepared { b, call, scratch, out, oq, os, err, cnt })
}

impl Prepared {
    /// One launch of the decode (split) or prefill entry, then the outputs.
    pub fn go(&mut self, decode: bool, epi: u32) -> Result<Out> {
        let dev = self.b.dev.clone();
        self.call.epi = epi;
        if decode { super::launch_decode(launch_of(&dev), &self.call, &self.scratch)?; } else { super::launch_prefill(launch_of(&dev), &self.call)?; }
        sync(&dev)?;
        if decode { ensure!(self.b.get_i32(self.cnt)?.iter().all(|&x| x == 0), "decode: arrival counters not reset after the launch"); }
        Ok(Out { o: self.b.get_u16(self.out)?, q: self.b.get_u8(self.oq)?, s: self.b.get_u8(self.os)?, err: self.b.get_i32(self.err)?[0] })
    }
}

/// Upload a case and run one entry (`decode` = the split entry with its scratch).
pub fn run_case(dev: &Arc<CudaDevice>, lanes: &[HostLane], c: &Case, decode: bool, epi: u32) -> Result<Out> {
    prepare(dev, lanes, c)?.go(decode, epi)
}

/// Rows at assorted positions over two lanes (lane 0: linear window from 0; lane 1: ring + linear past 2900).
pub fn mixed_case(rng: &mut Rng, lanes: &[HostLane], heads: usize, use_kv: bool, freq: [f32; 32]) -> Case {
    let ps = [3000i32, 2999, 2998, 2997, 2996, 2995, 63, 64, 10, 0, 127, 128, 1500, 2050];
    let mut c = Case { heads, pos: vec![], wend: vec![], lane: vec![], sel: vec![], nsel: vec![], sel_cap: 512, q: vec![], sink: vec![],
                       use_kv, use_ext: false, win: vec![WinSrc::Lin { base: 0 }, WinSrc::Both { last: 3000, base: 2900 }], freq };
    for (r, &p) in ps.iter().enumerate() {
        c.pos.push(p);
        c.wend.push(p);
        c.lane.push(if p >= 2990 { (r & 1) as i32 } else { 0 });
        let reach = ((p + 1) / 2) as usize;
        let n = reach.min(512);
        let mut pick: Vec<i32> = (0..reach as i32).collect();
        for i in 0..n { let j = i + rng.below(reach - i); pick.swap(i, j); }
        pick.truncate(n);
        pick.sort();
        if r == 3 && n > 10 { pick[5] = -1; pick[200.min(n - 1)] = -1; }
        let ns = match r { 4 => 512, 5 => 100, 13 => 129, _ => n } as i32;
        for i in 0..512 { c.sel.push(if i < n { pick[i] } else { -1 }); }
        c.nsel.push(ns);
    }
    c.q = (0..ps.len() * heads * 512).map(|_| cref::bf16_bits(rng.gauss() * 0.5)).collect();
    c.sink = (0..heads).map(|_| rng.gauss()).collect();
    let _ = lanes;
    c
}

fn worst_rel(c: &Case, lanes: &[HostLane], o: &[u16]) -> (f64, f64) {
    let (mut w64, mut wref) = (0f64, 0f64);
    let scale = super::softmax_scale();
    for r in 0..c.rows() {
        let q = c.qrow(r);
        let keys = c.keys(lanes, r);
        let f = cref::attend_f64(&q, c.heads, &keys, &c.sink, scale);
        let got: Vec<f32> = o[r * c.heads * 512..(r + 1) * c.heads * 512].iter().map(|b| cref::bf16_f32(*b)).collect();
        w64 = w64.max(cref::rel_l2(&got, &f));
        let lst = c.ref_list(lanes, r);
        let rl = cref::attend_ref_layout(&q, c.heads, &lst, &c.sink, scale);
        wref = wref.max(cref::rel_l2_f32(&got, &rl));
    }
    (w64, wref)
}

/// Sub-case with a subset of rows (batch invariance) or a head range (TP).
fn sub_rows(c: &Case, rows: &[usize]) -> Case {
    let mut s = Case { heads: c.heads, pos: vec![], wend: vec![], lane: vec![], sel: vec![], nsel: vec![], sel_cap: c.sel_cap, q: vec![], sink: c.sink.clone(),
                       use_kv: c.use_kv, use_ext: c.use_ext, win: c.win.clone(), freq: c.freq };
    for &r in rows {
        s.pos.push(c.pos[r]); s.wend.push(c.wend[r]); s.lane.push(c.lane[r]); s.nsel.push(c.nsel[r]);
        s.sel.extend_from_slice(&c.sel[r * c.sel_cap..(r + 1) * c.sel_cap]);
        s.q.extend_from_slice(&c.q[r * c.heads * 512..(r + 1) * c.heads * 512]);
    }
    s
}
fn sub_heads(c: &Case, h0: usize, h: usize) -> Case {
    let mut s = sub_rows(c, &(0..c.rows()).collect::<Vec<_>>());
    s.heads = h;
    s.sink = c.sink[h0..h0 + h].to_vec();
    s.q = (0..c.rows()).flat_map(|r| c.q[(r * c.heads + h0) * 512..(r * c.heads + h0 + h) * 512].to_vec()).collect();
    s
}
fn heads_of(o: &[u16], heads: usize, rows: usize, h0: usize, h: usize) -> Vec<u16> {
    (0..rows).flat_map(|r| o[(r * heads + h0) * 512..(r * heads + h0 + h) * 512].to_vec()).collect()
}

/// The synthetic suite. Returns report lines; any failure is an error naming it.
pub fn synthetic(dev: &Arc<CudaDevice>) -> Result<Vec<String>> {
    let mut lines = Vec::new();
    let mut rng = Rng(0x5EED_D41A);
    let lanes = vec![HostLane::build(&mut rng, 3000, 1500, 5), HostLane::build(&mut rng, 3000, 1500, 5)];
    let freq_c = super::rope_inv_freq(160000.0, Some((16.0, 65536.0, 32.0, 1.0)))?;
    // ---- 2 + 3a: correctness at every TP head range, decode == prefill ----
    let mut base64: Option<(Case, Vec<u16>)> = None;
    for &heads in &[64usize, 32, 16, 8] {
        for use_kv in [false, true] {
            let c = mixed_case(&mut Rng(0xC0FFEE + heads as u64), &lanes, heads, use_kv, freq_c);
            let d = run_case(dev, &lanes, &c, true, 0)?;
            let p = run_case(dev, &lanes, &c, false, 0)?;
            ensure!(d.err == 0 && p.err == 0, "heads {heads}: unexpected error word {} / {}", d.err, p.err);
            ensure!(d.o == p.o, "heads {heads} kv {use_kv}: decode and prefill differ");
            let (w64, wref) = worst_rel(&c, &lanes, &p.o);
            ensure!(w64 < 6e-3, "heads {heads} kv {use_kv}: rel-L2 vs the f64 attention {w64:.3e}");
            ensure!(wref < 6e-3, "heads {heads} kv {use_kv}: rel-L2 vs the reference-arithmetic emulation {wref:.3e}");
            lines.push(format!("heads {heads:2} {:>10}: {} rows, worst rel-L2 vs f64 {w64:.2e}, vs the reference's block arithmetic {wref:.2e}; decode == prefill bitwise",
                               if use_kv { "window+sel" } else { "window" }, c.rows()));
            if heads == 64 && use_kv { base64 = Some((c, p.o)); }
        }
    }
    let (c64, o64) = base64.context("base case")?;
    // ---- 3b: TP head ranges: heads of the 64-head call == the same heads at 32 / 16 / 8 ----
    for &(h0, h) in &[(0usize, 32usize), (32, 32), (16, 16), (48, 16), (8, 8), (56, 8)] {
        let s = sub_heads(&c64, h0, h);
        let o = run_case(dev, &lanes, &s, true, 0)?.o;
        ensure!(o == heads_of(&o64, 64, c64.rows(), h0, h), "TP: heads {h0}..{} computed at {h} heads differ from the 64-head call", h0 + h);
    }
    lines.push("TP head ranges: heads [0,32) [32,64) at 32, [16,32) [48,64) at 16, [8,16) [56,64) at 8 == the 64-head call, bitwise".into());
    // ---- 3c: batch invariance: each row alone == in the call ----
    for r in 0..c64.rows() {
        let s = sub_rows(&c64, &[r]);
        let o = run_case(dev, &lanes, &s, true, 0)?.o;
        ensure!(o == o64[r * 64 * 512..(r + 1) * 64 * 512], "row {r} alone differs from the row in the {}-row call", c64.rows());
    }
    lines.push(format!("batch invariance: each of the {} rows alone == the row in the call, bitwise", c64.rows()));
    // ---- 3d: ring vs linear window, padded vs trimmed selection ----
    {
        let mut a = sub_rows(&c64, &[0, 1, 2, 3]);
        a.lane = vec![1; 4];
        a.win = vec![WinSrc::Lin { base: 0 }, WinSrc::Ring { last: 3000 }];
        let ring = run_case(dev, &lanes, &a, true, 0)?.o;
        a.win = vec![WinSrc::Lin { base: 0 }, WinSrc::Lin { base: 1000 }];
        let lin = run_case(dev, &lanes, &a, true, 0)?.o;
        ensure!(ring == lin, "window from the ring != window from linear rows");
        let mut t = sub_rows(&c64, &[0, 2, 6, 8]);
        let trimmed = run_case(dev, &lanes, &t, false, 0)?.o;
        t.nsel = vec![512; 4];
        let padded = run_case(dev, &lanes, &t, false, 0)?.o;
        ensure!(trimmed == padded, "a selection padded with -1 to 512 != the trimmed selection");
        lines.push("window from the ring == from linear rows; selection padded with -1 == trimmed: bitwise".into());
    }
    // ---- 2b: DSpark rows: window ends at the main position, 5 extra keys attended by every row ----
    {
        let p = 2500i32;
        let mut c = Case { heads: 16, pos: (1..=5).map(|i| p + i).collect(), wend: vec![p; 5], lane: vec![0; 5], sel: vec![-1; 5 * 512], nsel: vec![0; 5],
                           sel_cap: 512, q: vec![], sink: vec![], use_kv: false, use_ext: true, win: vec![WinSrc::Ring { last: p as usize }, WinSrc::Lin { base: 0 }],
                           freq: super::rope_inv_freq(10000.0, None)? };
        c.q = (0..5 * 16 * 512).map(|_| cref::bf16_bits(rng.gauss() * 0.5)).collect();
        c.sink = (0..16).map(|_| rng.gauss()).collect();
        let d = run_case(dev, &lanes, &c, true, 0)?;
        let pf = run_case(dev, &lanes, &c, false, 0)?;
        ensure!(d.o == pf.o, "DSpark rows: decode != prefill");
        let (w64, wref) = worst_rel(&c, &lanes, &pf.o);
        ensure!(w64 < 6e-3 && wref < 6e-3, "DSpark rows: rel-L2 {w64:.2e} / {wref:.2e}");
        lines.push(format!("DSpark rows (window to the main position + 5 extra keys): worst rel-L2 vs f64 {w64:.2e}, vs block arithmetic {wref:.2e}; decode == prefill"));
    }
    // ---- 4: the epilogue, bit for bit against its CPU port on the kernel's raw output ----
    {
        let mut c = sub_rows(&c64, &(0..c64.rows()).collect::<Vec<_>>());
        c.pos = c.pos.iter().enumerate().map(|(i, p)| if i % 3 == 0 { p + 600_000 } else { *p }).collect(); // the large-angle path too
        for (epi, name) in [(EPI_ROPE, "inverse RoPE"), (EPI_AQ, "act_quant"), (EPI_ROPE | EPI_AQ, "inverse RoPE + act_quant")] {
            let raw = run_case(dev, &lanes, &c, false, 0)?;
            let got = run_case(dev, &lanes, &c, true, epi)?;
            for r in 0..c.rows() {
                let o: Vec<f32> = raw.o[r * 64 * 512..(r + 1) * 64 * 512].iter().map(|b| cref::bf16_f32(*b)).collect();
                let (x, q, s) = cref::epilogue(&o, 64, c.pos[r], &c.freq, epi & EPI_ROPE != 0, epi & EPI_AQ != 0);
                let want: Vec<u16> = x.iter().map(|v| cref::bf16_bits(*v)).collect();
                ensure!(got.o[r * 64 * 512..(r + 1) * 64 * 512] == want[..], "epilogue {name}: row {r} bf16 output differs from the CPU port");
                if epi & EPI_AQ != 0 {
                    ensure!(got.q[r * 64 * 512..(r + 1) * 64 * 512] == q[..], "epilogue {name}: row {r} E4M3 codes differ");
                    ensure!(got.s[r * 64 * 16..(r + 1) * 64 * 16] == s[..], "epilogue {name}: row {r} E8M0 bytes differ");
                }
            }
        }
        lines.push("epilogue (inverse RoPE incl. positions past 105,615; act_quant values, E4M3 codes, E8M0 bytes) == its CPU port, bitwise".into());
    }
    // ---- 4b (leg PD1): wo_a's rotated input written by the decode epilogue == d41x_dd_prep's math on the stored bf16 out,
    // bit for bit (a host emulation of the lane butterflies), and the attention's own outputs unchanged ----
    {
        let c = sub_rows(&c64, &(0..c64.rows()).collect::<Vec<_>>());
        let epi = EPI_ROPE | EPI_AQ;
        let base = run_case(dev, &lanes, &c, true, epi)?;
        let mut pr = prepare(dev, &lanes, &c)?;
        pr.call.epi = epi;
        let (hpg, ng) = (8usize, c.heads / 8);
        let suh: Vec<Vec<u16>> = (0..ng).map(|_| (0..hpg * 512).map(|_| half::f16::from_f32(if rng.next() & 1 == 0 { 1.0 } else { -1.0 } * (0.7 + 0.3 * rng.unif())).to_bits()).collect()).collect();
        let dsuh: Vec<DevPtr> = suh.iter().map(|v| pr.b.u16(v)).collect::<Result<_>>()?;
        let rows = c.rows();
        let ld = c.heads * 512 + 64;
        let xh = pr.b.u16(&vec![0xFFFFu16; rows * ld])?;
        let sp: Vec<u64> = dsuh.iter().map(|p| p.0).collect();
        super::launch_decode_rot(launch_of(dev), &pr.call, &pr.scratch, &super::RotOut { suh: &sp, hpg, xh: xh.0, ld })?;
        sync(dev)?;
        let o = pr.b.get_u16(pr.call.out)?;
        ensure!(o == base.o, "rot_out: the attention output changed");
        let got = pr.b.get_u16(xh)?;
        let mut bad = 0usize;
        for r in 0..rows { for h in 0..c.heads { for b in 0..4 {
            let x: Vec<f32> = (0..128).map(|j| cref::bf16_f32(o[r * c.heads * 512 + h * 512 + b * 128 + j])).collect();
            let sv: Vec<f32> = (0..128).map(|j| half::f16::from_bits(suh[h / hpg][(h % hpg) * 512 + b * 128 + j]).to_f32()).collect();
            let want = crate::dsv41::weights::exl3::dense_dec::prep128_emul(&x, &sv);
            if got[r * ld + h * 512 + b * 128..r * ld + h * 512 + b * 128 + 128] != want[..] { bad += 1; }
        } } }
        ensure!(bad == 0, "rot_out: {bad} (row, head, block) rotated rows differ from d41x_dd_prep's math");
        lines.push(format!("PD1 rot_out: wo_a's rotated input from the decode epilogue ({rows} rows x {} heads, {ng} groups of {hpg}) == d41x_dd_prep's math on the stored out, bitwise; out unchanged", c.heads));
    }
    // ---- 4c (leg PA1): the decode entry's X_* flags (window prefetch before the wait, the separate per-head merge, q's RoPE
    // in the q load) == the plain decode entry, bit for bit (out, E4M3 codes, E8M0 bytes, wo_a's rotated rows, error word) ----
    lines.extend(pa1_flags(dev, &lanes, &c64, &mut rng)?);
    lines.extend(pa1_recipe(dev, &lanes, &c64)?);
    // ---- 5: an out-of-range selection is reported and masked ----
    {
        let mut c = sub_rows(&c64, &[0]);
        let mut d = sub_rows(&c64, &[0]);
        c.sel[7] = 1_000_000;
        d.sel[7] = -1;
        let a = run_case(dev, &lanes, &c, true, 0)?;
        let b = run_case(dev, &lanes, &d, true, 0)?;
        ensure!(a.err == super::ERR_SEL_RANGE && b.err == 0, "error word {} / {}", a.err, b.err);
        ensure!(a.o == b.o, "an out-of-range selection is not treated as masked");
        lines.push("an out-of-range selection sets the error word and is masked".into());
    }
    // ---- 3e: determinism and the self-resetting counters (24 decode rows, 200 launches on one upload) ----
    {
        let rows: Vec<usize> = (0..24).map(|i| i % c64.rows()).collect();
        let c = sub_rows(&c64, &rows);
        let mut pr = prepare(dev, &lanes, &c)?;
        let first = pr.go(true, 0)?.o;
        for i in 1..200 {
            let o = pr.go(true, 0)?.o;
            ensure!(o == first, "launch {i}: decode output changed between launches on the same scratch");
        }
        let pf = pr.go(false, 0)?.o;
        ensure!(pf == first, "24 rows: prefill != decode");
        lines.push("determinism: 200 decode launches of a 24-row call on one scratch, every output bitwise identical (counters reset each time)".into());
    }
    Ok(lines)
}

/// `d41_attn_abi`: host mirror == device.
pub fn abi_check(dev: &Arc<CudaDevice>) -> Result<String> {
    let f = dev.get_func(crate::dsv41::gpu::MODULE, "d41_attn_abi").context("d41_attn_abi missing")?;
    let mut out = dev.alloc_zeros::<u32>(13)?;
    unsafe { f.launch(LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 }, (&mut out,)) }?;
    let dvc = dev.dtoh_sync_copy(&out)?;
    let host = super::host_abi();
    ensure!(dvc[..] == host[..], "attn ABI mismatch: device {dvc:?} host {host:?} (lane, args, q, out, rows, rope_freq, smem, threads, part, splits, lanes, lane[1], split blocks)");
    Ok(format!("ABI: d41_attn_lane {} B, d41_attn_args {} B, smem {} B, {} threads: host == device", host[0], host[1], host[6], host[7]))
}

/// The probe entry (cli.rs): `--probe-dsv41-kernel --dsv41-kernel-set attn [--dsv41-kernel-fixture DIR[,DIR..]]
/// [--dsv41-kernel-timing 4k,128k,1m]`.
pub fn run(fixtures: Option<&str>, timing: Option<&str>) -> Result<Vec<String>> {
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    crate::dsv41::gpu::load_module(&dev)?;
    let mut lines = vec![abi_check(&dev)?];
    lines.extend(synthetic(&dev)?);
    if let Some(f) = fixtures {
        for dir in f.split(',').filter(|s| !s.is_empty()) {
            lines.extend(super::fixture::run(&dev, std::path::Path::new(dir))?);
        }
    }
    if let Some(t) = timing {
        if t.split(',').any(|x| x == "pa1") { lines.extend(pa1_timing(&dev)?); }
        let rest: Vec<&str> = t.split(',').filter(|x| *x != "pa1").collect();
        if !rest.is_empty() { lines.extend(super::timing::run(&dev, &rest.join(","))?); }
    }
    if lines.iter().any(|l| l.contains("FAIL")) { bail!("a check failed:\n{}", lines.join("\n")); }
    lines.push("DSV41_KERNEL_ATTN_OK".into());
    Ok(lines)
}

/// Leg PA1: every `X_*` combination of the decode entry against the plain decode entry (PD1's rotation on and off), at the
/// TP head ranges 64 / 32 / 16 / 8, 1 / 3 / 14 / 24 rows (split groups G = splits .. 1), the epilogue on; X_QROPE gets the
/// un-rotated q and is compared with `d41f_rope_rows` then the plain entry.
fn pa1_flags(dev: &Arc<CudaDevice>, lanes: &[HostLane], c64: &Case, rng: &mut Rng) -> Result<Vec<String>> {
    use super::{X_DEFER, X_QROPE, X_WPF};
    let epi = EPI_ROPE | EPI_AQ;
    let mut n = 0usize;
    for &(h0, heads) in &[(0usize, 64usize), (0, 32), (16, 16), (56, 8)] {
        let hc = sub_heads(c64, h0, heads);
        for rows in [vec![0usize], vec![0, 7, 13], (0..hc.rows()).collect::<Vec<_>>(), (0..24).map(|i| i % hc.rows()).collect()] {
            let c = sub_rows(&hc, &rows);
            for rot in [false, true] {
                let (hpg, ng) = (8usize.min(heads), heads / 8usize.min(heads));
                let suh: Vec<Vec<u16>> = (0..ng).map(|_| (0..hpg * 512).map(|_| half::f16::from_f32(if rng.next() & 1 == 0 { 1.0 } else { -1.0 } * (0.7 + 0.3 * rng.unif())).to_bits()).collect()).collect();
                // one run: (q already rotated by d41f_rope_rows?, flags) -> (out, codes, scales, rotated rows, error word)
                let run = |pre_rope: bool, x: u32| -> Result<(Vec<u16>, Vec<u8>, Vec<u8>, Vec<u16>, i32)> {
                    let mut pr = prepare(dev, lanes, &c)?;
                    pr.call.epi = epi;
                    if pre_rope {
                        let a = crate::dsv41::fwd::glue::RopeArgs { x: pr.call.q.0, pos: pr.call.pos.0, ld: (heads * 512) as i64, rows: c.rows() as i32, heads: heads as i32,
                                                                    head_dim: 512, inverse: 0, freq: c.freq };
                        crate::dsv41::gpu::launch_args("d41f_rope_rows", (c.rows() as u32, ((heads * 32) as u32).div_ceil(256), 1), (256, 1, 1), 0, launch_of(dev).stream, &a)?;
                    }
                    let ld = heads * 512 + 64;
                    let xh = pr.b.u16(&vec![0xFFFFu16; c.rows() * ld])?;
                    let dsuh: Vec<DevPtr> = suh.iter().map(|v| pr.b.u16(v)).collect::<Result<_>>()?;
                    let sp: Vec<u64> = dsuh.iter().map(|p| p.0).collect();
                    let ro = super::RotOut { suh: &sp, hpg, xh: xh.0, ld };
                    super::launch_decode_x(launch_of(dev), &pr.call, &pr.scratch, if rot { Some(&ro) } else { None }, x)?;
                    sync(dev)?;
                    ensure!(pr.b.get_i32(pr.cnt)?.iter().all(|&v| v == 0), "PA1 flags {x:#x}: arrival counters not zero after the launch");
                    Ok((pr.b.get_u16(pr.out)?, pr.b.get_u8(pr.oq)?, pr.b.get_u8(pr.os)?, pr.b.get_u16(xh)?, pr.b.get_i32(pr.err)?[0]))
                };
                let base = run(true, 0)?;
                ensure!(base.4 == 0, "PA1: base error word {}", base.4);
                for x in [X_WPF, X_DEFER, X_WPF | X_DEFER, X_QROPE, X_QROPE | X_WPF | X_DEFER] {
                    let got = run(x & X_QROPE == 0, x)?;
                    ensure!(got == base, "PA1 flags {x:#x}: heads {heads} (from {h0}), {} rows, rot {rot}: differs from the plain decode entry", c.rows());
                    n += 1;
                }
            }
        }
    }
    Ok(vec![format!("PA1 decode flags (window prefetch before the wait, per-head merge launch, q RoPE in the q load, and their combinations): {n} cases == the plain decode entry (q rotated by d41f_rope_rows), out / E4M3 codes / E8M0 bytes / wo_a rotated rows / error word bitwise; heads 64 / 32 / 16 / 8, rows 1 / 3 / 14 / 24, rotation on and off")])
}

/// Leg PA1, `--dsv41-attn-split recipe` (opt-in, different bits): correctness against the f64 attention (same bound as the
/// exact rule), determinism, batch invariance inside each class (<= 16 rows: 8 splits; > 16: one split), the X_WPF / X_QROPE
/// flags bit-neutral, and the distance to the exact rule (raw o, bf16).
fn pa1_recipe(dev: &Arc<CudaDevice>, lanes: &[HostLane], c64: &Case) -> Result<Vec<String>> {
    use super::{X_QROPE, X_WPF};
    let mut lines = Vec::new();
    let go = |c: &Case, x: u32, pre_rope: bool, epi: u32| -> Result<Vec<u16>> {
        let mut pr = prepare(dev, lanes, c)?;
        pr.call.epi = epi;
        if pre_rope {
            let a = crate::dsv41::fwd::glue::RopeArgs { x: pr.call.q.0, pos: pr.call.pos.0, ld: (c.heads * 512) as i64, rows: c.rows() as i32, heads: c.heads as i32,
                                                        head_dim: 512, inverse: 0, freq: c.freq };
            crate::dsv41::gpu::launch_args("d41f_rope_rows", (c.rows() as u32, ((c.heads * 32) as u32).div_ceil(256), 1), (256, 1, 1), 0, launch_of(dev).stream, &a)?;
        }
        super::launch_recipe(launch_of(dev), &pr.call, Some(&pr.scratch), None, x)?;
        sync(dev)?;
        ensure!(pr.b.get_i32(pr.err)?[0] == 0, "recipe: error word");
        pr.b.get_u16(pr.out)
    };
    let (mut worst64, mut worstref, mut maxd, mut sumd, mut nd, mut ndiff) = (0f64, 0f64, 0f32, 0f64, 0usize, 0usize);
    for &(h0, heads) in &[(0usize, 64usize), (0, 32), (16, 16), (56, 8)] {
        let hc = sub_heads(c64, h0, heads);
        let n = hc.rows();
        // correctness (raw o) and the distance to the exact rule
        let o = go(&hc, 0, false, 0)?;
        let ex = run_case(dev, lanes, &hc, true, 0)?.o;
        let (w64, wref) = worst_rel(&hc, lanes, &o);
        ensure!(w64 < 6e-3, "recipe heads {heads}: rel-L2 vs the f64 attention {w64:.3e}");
        worst64 = worst64.max(w64); worstref = worstref.max(wref);
        for (a, b) in o.iter().zip(&ex) {
            let d = (cref::bf16_f32(*a) - cref::bf16_f32(*b)).abs();
            maxd = maxd.max(d); sumd += d as f64; nd += 1; ndiff += (a != b) as usize;
        }
        // determinism + the flags (epilogue on)
        let epi = EPI_ROPE | EPI_AQ;
        let base = go(&hc, 0, false, epi)?;
        ensure!(go(&hc, 0, false, epi)? == base, "recipe heads {heads}: two launches differ");
        ensure!(go(&hc, X_WPF, false, epi)? == base, "recipe heads {heads}: X_WPF changed the output");
        let rq = go(&hc, 0, true, epi)?;
        ensure!(go(&hc, X_QROPE | X_WPF, false, epi)? == rq, "recipe heads {heads}: X_QROPE differs from d41f_rope_rows + recipe");
        // batch invariance inside the classes: each row alone == the row in the n-row call (n <= 16), in 1..16-row calls,
        // and the rows of 17- and 24-row calls equal each other (one split)
        let w = heads * 512;
        for r in 0..n {
            let one = go(&sub_rows(&hc, &[r]), 0, false, epi)?;
            ensure!(one[..] == base[r * w..(r + 1) * w], "recipe heads {heads}: row {r} alone != in the {n}-row call");
        }
        for width in [2usize, 7, 16] {
            let rows: Vec<usize> = (0..width).map(|i| i % n).collect();
            let o = go(&sub_rows(&hc, &rows), 0, false, epi)?;
            for (i, &r) in rows.iter().enumerate() { ensure!(o[i * w..(i + 1) * w] == base[r * w..(r + 1) * w], "recipe heads {heads}: width {width} row {i}"); }
        }
        // > 16 rows (verify rounds of 4 lanes x 6, prefill chunks): the same 8-split rule a row: equal to the row alone
        for width in [17usize, 24, 40] {
            let rows: Vec<usize> = (0..width).map(|i| (i + 3) % n).collect();
            let o = go(&sub_rows(&hc, &rows), 0, false, epi)?;
            for (i, &r) in rows.iter().enumerate() { ensure!(o[i * w..(i + 1) * w] == base[r * w..(r + 1) * w], "recipe heads {heads}: width {width} row {i} != the row alone"); }
        }
        lines.push(format!("PA1 recipe rule, heads {heads:2}: rel-L2 vs f64 {w64:.2e}; deterministic; X_WPF / X_QROPE bit-neutral; each row alone == the row in 2 / 7 / {n} / 16 / 17 / 24 / 40-row calls (8 splits a row at every width)"));
    }
    lines.push(format!("PA1 recipe vs exact rule (raw o, bf16): {ndiff} / {nd} values differ, max |diff| {maxd:.3e}, mean |diff| {:.3e}; worst rel-L2 vs f64: recipe {worst64:.2e}, (vs reference block arithmetic {worstref:.2e})", sumd / nd.max(1) as f64));
    Ok(lines)
}

/// Leg PA1 (`--dsv41-kernel-timing pa1`): microseconds per call of the decode attention variants at the TP2 head range (32
/// heads), rows 1 / 2 / 6 / 24 of position 3000 with 512 picks, epilogue + wo_a rotation on, 300 back-to-back launches
/// (hot L2, no PDL). It ranks the variants' kernel time; it is not the served (PDL, cold-L2, in-graph) number.
fn pa1_timing(dev: &Arc<CudaDevice>) -> Result<Vec<String>> {
    use super::{X_DEFER, X_QROPE, X_RECIPE, X_WPF};
    let mut rng = Rng(0x5EED_D41A);
    let lanes = vec![HostLane::build(&mut rng, 3000, 1500, 5), HostLane::build(&mut rng, 3000, 1500, 5)];
    let freq_c = super::rope_inv_freq(160000.0, Some((16.0, 65536.0, 32.0, 1.0)))?;
    let c = mixed_case(&mut Rng(0xC0FFEE + 32), &lanes, 32, true, freq_c);
    let mut out = vec!["PA1 attention timing (32 heads = TP2, row of position 3000 with 512 picks, epilogue + rotation, hot L2, no PDL; us per call):".to_string()];
    for rows in [1usize, 2, 6, 24] {
        let cc = sub_rows(&c, &vec![0usize; rows]);
        let mut pr = prepare(dev, &lanes, &cc)?;
        pr.call.epi = EPI_ROPE | EPI_AQ;
        let ld = 32 * 512;
        let xh = pr.b.u16(&vec![0u16; rows * ld])?;
        let suh: Vec<u16> = (0..8 * 512).map(|_| half::f16::from_f32(1.0).to_bits()).collect();
        let sp: Vec<u64> = (0..4).map(|_| pr.b.u16(&suh).map(|p| p.0)).collect::<Result<_>>()?;
        let ro = super::RotOut { suh: &sp, hpg: 8, xh: xh.0, ld };
        let mut row = format!("  rows {rows:2}:");
        for (name, x) in [("exact (base)", 0u32), ("+wpf", X_WPF), ("+merge launch", X_DEFER), ("+wpf+merge+qrope", X_WPF | X_DEFER | X_QROPE), ("RECIPE rule", X_RECIPE | X_WPF)] {
            for _ in 0..20 { super::launch_decode_x(launch_of(dev), &pr.call, &pr.scratch, Some(&ro), x)?; }
            sync(dev)?;
            let t0 = std::time::Instant::now();
            for _ in 0..300 { super::launch_decode_x(launch_of(dev), &pr.call, &pr.scratch, Some(&ro), x)?; }
            sync(dev)?;
            row.push_str(&format!(" {name} {:.1}", t0.elapsed().as_secs_f64() * 1e6 / 300.0));
        }
        out.push(row);
    }
    Ok(out)
}
