//! DSV4.1 leg KI: the indexer's GPU gates and its timing table.
//! `gb10_inference --probe-dsv41-kernel --dsv41-kernel-set indexer [--max-seq-len N] [--dsv41-dump-ref DIR]`
//!
//! Gates (all on synthetic data unless DIR is given; every one asserts its success signal):
//! * ABI: the argument block / shared-memory sizes agree between Rust and the GPU.
//! * q prep: the kernel's RoPE + FP4 query (`dbg_q`) against `reference::q_prep` (cos / sin from the
//!   host; CUDA sincosf may differ by one ulp, so the gate is a mismatch fraction, reported).
//! * scores: every visible key's 16-bit score (`dbg`) BIT-EXACT against the host reference computed
//!   from the kernel's own prepared query.
//! * selections: the GPU top-512 lists EQUAL the host's (score desc, position asc), for decode,
//!   verify (rows sharing the key pass), chunked + merged, every row-block width (batch invariance),
//!   ratio 2 and 1, short / empty rows, heavy ties, f32 inputs, wide scale spreads.
//! * candidates: block maxima, the top-2048 blocks, the union, and mode-1 selections (layers 24-36).
//! * DIR (REFb fixture from tools/dsv41_ki/idx_fixture.py): real queries / keys of the 8 index layers;
//!   the selections against REFb's `topk_compressed_pos` (up to ties: overlap reported) and the
//!   bit-exact host gate on the same data.
//! Timing: decode / verify / multi-lane rounds from 4K to `--max-seq-len` (default 1M), cold L2,
//! per layer class and per round, with the key-stream GB/s against the 238 GB/s floor; prefill
//! throughput of the row-blocked kernel.

use super::reference as rf;
use super::*;
use crate::dsv41::weights::Launch;
use anyhow::{bail, ensure, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use std::sync::Arc;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 { self.next() % n.max(1) }
    fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
    fn normal(&mut self) -> f64 { let (a, b) = (self.unit().max(1e-300), self.unit()); (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos() }
}

fn dp<T>(s: &CudaSlice<T>) -> u64 { *s.device_ptr() }

/// One lane's index-K cache on the host.
#[derive(Clone)]
pub struct Lane { pub kq: Vec<u8>, pub ks: Vec<u8> }
impl Lane {
    pub fn n(&self) -> usize { self.ks.len() / 4 }
    fn dec(&self) -> Vec<i8> { rf::decode2(&self.kq) }
}

/// One query row (bf16 bits).
#[derive(Clone)]
pub struct RowIn { pub lane: usize, pub pos: i32, pub iq: Vec<u16>, pub w: Vec<u16> }

fn bf(x: f32) -> u16 { rf::bf16_bits(x) }
fn unbf(b: u16) -> f32 { f32::from_bits((b as u32) << 16) }

/// Random keys: random codes, scale exponents 127 + base +- spread / 2; `pool` > 0 repeats `pool`
/// distinct key rows (heavy score ties).
fn gen_lane(rng: &mut Rng, n: usize, spread: u32, pool: usize) -> Lane {
    let mut kq = vec![0u8; n * 64];
    let mut ks = vec![0u8; n * 4];
    let base = 124 + rng.below(5) as i32;
    let salt = rng.next();
    for i in 0..n {
        let src = if pool > 0 { (rng.below(pool as u64) as usize) * 7919 } else { i };
        let mut r2 = Rng(src as u64 ^ salt);
        for b in 0..64 { kq[i * 64 + b] = (r2.next() & 0xFF) as u8; }
        for g in 0..4 { ks[i * 4 + g] = (base + r2.below(spread as u64 + 1) as i32 - spread as i32 / 2).clamp(1, 254) as u8; }
    }
    Lane { kq, ks }
}

fn gen_row(rng: &mut Rng, lane: usize, pos: i32, mag: f64) -> RowIn {
    let iq = (0..HEADS * HEAD_DIM).map(|i| { let m = if (i % 128) >= 64 { mag * 0.5 } else { mag }; bf((rng.normal() * m) as f32) }).collect();
    let w = (0..HEADS).map(|_| bf((rng.normal() * 2.0) as f32)).collect();
    RowIn { lane, pos, iq, w }
}

/// Device-side state of one indexer call (buffers sized for the call).
struct Call {
    a: IdxArgs,
    _in32: Option<(CudaSlice<f32>, CudaSlice<f32>)>,
    rbs: CudaSlice<IdxRbRaw>,
    sel: CudaSlice<i32>, sel_cnt: CudaSlice<i32>, part: CudaSlice<u64>, pcnt: CudaSlice<i32>,
    bmax: CudaSlice<u16>, cand: CudaSlice<i32>, cand_cnt: CudaSlice<i32>, bm: CudaSlice<u32>,
    uni: CudaSlice<u32>, ucnt: CudaSlice<i32>, dbg: CudaSlice<u16>, dbgq: CudaSlice<u8>, err: CudaSlice<i32>, tick: CudaSlice<i32>,
    pos: CudaSlice<i32>, iq: CudaSlice<u16>, w: CudaSlice<u16>,
}

/// `IdxRb` as a plain-old-data device element.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct IdxRbRaw(pub [u64; 4]);
unsafe impl cudarc::driver::DeviceRepr for IdxRbRaw {}
unsafe impl cudarc::driver::ValidAsZeroBits for IdxRbRaw {}
fn rb_raw(r: &IdxRb) -> IdxRbRaw {
    IdxRbRaw([r.kq, r.ks, (r.row0 as u32 as u64) | ((r.nrows as u32 as u64) << 32), 0])
}

pub struct Gpu { pub dev: Arc<CudaDevice>, pub l: Launch }
impl Gpu {
    pub fn new() -> Result<Gpu> {
        let dev = CudaDevice::new(0).context("CUDA device 0")?;
        super::super::load_module(&dev)?;
        let l = Launch { stream: *dev.cu_stream() as u64, pdl: false };
        Ok(Gpu { dev, l })
    }
    fn sync(&self) -> Result<()> { self.dev.synchronize().context("synchronize") }
}

/// Upload lanes once (they are reused by several calls).
struct DLanes { kq: Vec<CudaSlice<u8>>, ks: Vec<CudaSlice<u8>> }
fn upload_lanes(g: &Gpu, lanes: &[Lane]) -> Result<DLanes> {
    let mut kq = Vec::new();
    let mut ks = Vec::new();
    for l in lanes { kq.push(g.dev.htod_sync_copy(&l.kq)?); ks.push(g.dev.htod_sync_copy(&l.ks)?); }
    Ok(DLanes { kq, ks })
}

/// One call's shape. `fused`: the last chunk CTA merges (tick) instead of the separate merge kernel.
struct Spec { ratio: i32, mode: i32, r: usize, g: usize, min_tiles: usize, write_bmax: bool, in_f32: bool, dbg: bool, fused: bool }

/// Build a call: rows grouped per lane in input order into row blocks of <= R rows.
fn build(gpu: &Gpu, dl: &DLanes, n_keys_cap: usize, rows: &[RowIn], sp: &Spec, inv: &[f32; 32], ucap: usize) -> Result<Call> {
    let dev = &gpu.dev;
    let mut rbs = Vec::new();
    let mut i = 0;
    while i < rows.len() {
        let lane = rows[i].lane;
        let mut j = i;
        while j < rows.len() && rows[j].lane == lane && j - i < sp.r { j += 1; }
        rbs.push(IdxRb { kq: dp(&dl.kq[lane]), ks: dp(&dl.ks[lane]), row0: i as i32, nrows: (j - i) as i32, pad0: 0, pad1: 0 });
        i = j;
    }
    let n_rb = rbs.len();
    let nr = rows.len();
    let sz = sizes(n_rb, sp.r, sp.g, nr, n_keys_cap, ucap);
    let nb_cap = n_keys_cap.div_ceil(CAND_BLOCK);
    let nbw_cap = nb_cap.div_ceil(32);
    let iq_h: Vec<u16> = rows.iter().flat_map(|r| r.iq.iter().copied()).collect();
    let w_h: Vec<u16> = rows.iter().flat_map(|r| r.w.iter().copied()).collect();
    let (iq, w, in32) = if sp.in_f32 {
        let f: Vec<f32> = iq_h.iter().map(|&b| unbf(b)).collect();
        let fw: Vec<f32> = w_h.iter().map(|&b| unbf(b)).collect();
        (dev.htod_sync_copy(&[0u16])?, dev.htod_sync_copy(&[0u16])?, Some((dev.htod_sync_copy(&f)?, dev.htod_sync_copy(&fw)?)))
    } else { (dev.htod_sync_copy(&iq_h)?, dev.htod_sync_copy(&w_h)?, None) };
    let pos_h: Vec<i32> = rows.iter().map(|r| r.pos).collect();
    let (iq_ptr, w_ptr) = match &in32 { Some((a, b)) => (dp(a), dp(b)), None => (dp(&iq), dp(&w)) };
    let dbg_cap = if sp.dbg { n_keys_cap } else { 0 };
    let c = Call {
        rbs: dev.htod_sync_copy(&rbs.iter().map(rb_raw).collect::<Vec<_>>())?,
        sel: dev.htod_sync_copy(&vec![-7i32; sz.sel_i32])?,
        sel_cnt: dev.htod_sync_copy(&vec![-7i32; nr])?,
        part: dev.htod_sync_copy(&vec![0u64; sz.part_u64.max(1)])?,
        pcnt: dev.htod_sync_copy(&vec![0i32; sz.pcnt_i32.max(1)])?,
        bmax: dev.htod_sync_copy(&vec![0u16; if sp.write_bmax { sz.bmax_u16 } else { 1 }])?,
        cand: dev.htod_sync_copy(&vec![0i32; sz.cand_i32])?,
        cand_cnt: dev.htod_sync_copy(&vec![0i32; nr])?,
        bm: dev.htod_sync_copy(&vec![0u32; sz.bm_u32.max(1)])?,
        uni: dev.htod_sync_copy(&vec![0u32; sz.uni_u32.max(1)])?,
        ucnt: dev.htod_sync_copy(&vec![0i32; n_rb])?,
        dbg: dev.htod_sync_copy(&vec![0xFFFFu16; (nr * dbg_cap).max(1)])?,
        dbgq: dev.htod_sync_copy(&vec![0u8; if sp.dbg { nr * DBG_Q_ROW } else { 1 }])?,
        err: dev.htod_sync_copy(&[0i32])?,
        tick: dev.htod_sync_copy(&vec![0i32; n_rb])?,
        pos: dev.htod_sync_copy(&pos_h)?,
        iq, w, _in32: in32, a: IdxArgs::default(),
    };
    let mut a = IdxArgs::default();
    a.iq = iq_ptr; a.w = w_ptr; a.pos = dp(&c.pos); a.rbs = dp(&c.rbs);
    a.sel = dp(&c.sel); a.sel_cnt = dp(&c.sel_cnt); a.part = dp(&c.part); a.pcnt = dp(&c.pcnt);
    a.bmax = if sp.write_bmax { dp(&c.bmax) } else { 0 }; a.cand = dp(&c.cand); a.cand_cnt = dp(&c.cand_cnt); a.bm = dp(&c.bm);
    a.uni = dp(&c.uni); a.ucnt = dp(&c.ucnt); a.dbg = if sp.dbg { dp(&c.dbg) } else { 0 }; a.err = dp(&c.err);
    a.dbg_q = if sp.dbg { dp(&c.dbgq) } else { 0 };
    a.tick = if sp.fused { dp(&c.tick) } else { 0 };
    a.n_rb = n_rb as i32; a.rows_total = nr as i32; a.iq_ld = (HEADS * HEAD_DIM) as i32; a.w_ld = HEADS as i32;
    a.in_f32 = sp.in_f32 as i32; a.ratio = sp.ratio; a.mode = sp.mode; a.write_bmax = sp.write_bmax as i32;
    a.n_keys_cap = n_keys_cap as i32; a.nb_cap = nb_cap as i32; a.nbw_cap = nbw_cap as i32; a.g = sp.g as i32;
    a.min_tiles = sp.min_tiles as i32; a.dbg_cap = dbg_cap as i32; a.ucap = ucap as i32; a.r = sp.r as i32;
    a.inv_freq = *inv;
    Ok(Call { a, ..c })
}

/// The host's prepared query of a row (cos / sin from the host libm).
fn host_qprep(r: &RowIn, inv: &[f32; 32]) -> rf::QPrep {
    let iq: Vec<f32> = r.iq.iter().map(|&b| unbf(b)).collect();
    let cs: Vec<(f32, f32)> = inv.iter().map(|&f| { let a = (r.pos as f32) * f; (a.cos(), a.sin()) }).collect();
    rf::q_prep(&iq, &cs)
}

fn gpu_qprep(dq: &[u8], row: usize) -> rf::QPrep {
    let b = &dq[row * DBG_Q_ROW..(row + 1) * DBG_Q_ROW];
    rf::QPrep { codes: b[..HEADS * 64].to_vec(), scales: b[HEADS * 64..].to_vec() }
}

/// Decoded-value mismatches between two prepared queries (codes of equal value count as equal).
fn qprep_mismatch(a: &rf::QPrep, b: &rf::QPrep) -> usize {
    let (da, db) = (rf::decode2(&a.codes), rf::decode2(&b.codes));
    let mut n = 0;
    for h in 0..HEADS {
        for i in 0..HEAD_DIM {
            let g = i / 32;
            let va = da[h * 128 + i] as f64 * 2f64.powi(a.scales[h * 4 + g] as i32);
            let vb = db[h * 128 + i] as f64 * 2f64.powi(b.scales[h * 4 + g] as i32);
            if va != vb { n += 1; }
        }
    }
    n
}

/// Host results of one row: visible score list and the expected selection.
struct HostRow { scores: Vec<(u32, u16)>, sel: Vec<u32> }

fn nvis(pos: i32, ratio: i32, cap: usize) -> usize { if pos < 0 { 0 } else { (((pos + 1) / ratio) as usize).min(cap) } }

/// Host reference for every row of a call (from the GPU's prepared queries), restricted to the
/// candidate blocks when `cand` is given (per row, ascending block ids).
fn host_rows(rows: &[RowIn], lanes_dec: &[Vec<i8>], lanes: &[Lane], q: &[rf::QPrep], ratio: i32, cap: usize, cand: Option<&[Vec<u32>]>) -> Vec<HostRow> {
    // rows in parallel (the host reference is the slow side of every gate)
    let nt = std::thread::available_parallelism().map_or(4, |n| n.get()).min(32).max(1);
    let per = rows.len().div_ceil(nt).max(1);
    let mut out: Vec<Option<HostRow>> = (0..rows.len()).map(|_| None).collect();
    std::thread::scope(|sc| {
        for (k, slot) in out.chunks_mut(per).enumerate() {
            let i0 = k * per;
            sc.spawn(move || {
                for (j, o) in slot.iter_mut().enumerate() {
                    *o = Some(host_row(i0 + j, &rows[i0 + j], lanes_dec, lanes, &q[i0 + j], ratio, cap, cand));
                }
            });
        }
    });
    out.into_iter().map(|o| o.expect("host row")).collect()
}

#[allow(clippy::too_many_arguments)]
fn host_row(i: usize, r: &RowIn, lanes_dec: &[Vec<i8>], lanes: &[Lane], q: &rf::QPrep, ratio: i32, cap: usize, cand: Option<&[Vec<u32>]>) -> HostRow {
    {
        let qd = rf::QDec::new(q, &r.w.iter().map(|&b| unbf(b)).collect::<Vec<_>>());
        let n = nvis(r.pos, ratio, cap);
        let kd = &lanes_dec[r.lane];
        let ks = &lanes[r.lane].ks;
        let positions: Vec<u32> = match cand {
            None => (0..n as u32).collect(),
            Some(c) => c[i].iter().flat_map(|&b| (b * 8..b * 8 + 8).filter(move |&p| (p as usize) < n)).collect(),
        };
        let scores: Vec<(u32, u16)> = positions.iter().map(|&p| {
            let p = p as usize;
            (p as u32, qd.score(&kd[p * 128..p * 128 + 128], &ks[p * 4..p * 4 + 4]))
        }).collect();
        let sel = rf::select_topk(&scores, TOPK);
        HostRow { scores, sel }
    }
}

struct Outcome { lines: Vec<String>, ok: bool }
static DIAG_DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Run one call (score + merge [+ cand + union]) and check it against the host. Returns the GPU's
/// selections for cross-call identity checks, and (when write_bmax) the GPU candidate lists.
#[allow(clippy::too_many_arguments)]
fn run_case(gpu: &Gpu, name: &str, lanes: &[Lane], dl: &DLanes, rows: &[RowIn], sp: &Spec, inv: &[f32; 32],
            prev: Option<&Call>, host_cand: Option<&[Vec<u32>]>, out: &mut Outcome) -> Result<CaseOut> {
    let cap = lanes.iter().map(Lane::n).max().unwrap_or(0);
    let ucap = sp.r * CAND_BLOCKS;
    let mut call = build(gpu, dl, cap, rows, sp, inv, ucap)?;
    if let Some(p) = prev { // mode 1 visits the union produced by the previous (layer-20) call
        call.a.uni = p.a.uni; call.a.ucnt = p.a.ucnt;
    }
    select(&gpu.l, &call.a, sp.write_bmax)?;
    gpu.sync()?;
    let err = gpu.dev.dtoh_sync_copy(&call.err)?[0];
    let sel = gpu.dev.dtoh_sync_copy(&call.sel)?;
    let cnt = gpu.dev.dtoh_sync_copy(&call.sel_cnt)?;
    let dbg = gpu.dev.dtoh_sync_copy(&call.dbg)?;
    let dq = gpu.dev.dtoh_sync_copy(&call.dbgq)?;
    let gq: Vec<rf::QPrep> = (0..rows.len()).map(|i| gpu_qprep(&dq, i)).collect();
    let qmis: usize = rows.iter().enumerate().map(|(i, r)| qprep_mismatch(&host_qprep(r, inv), &gq[i])).sum();
    let ldec: Vec<Vec<i8>> = lanes.iter().map(Lane::dec).collect();
    let host = host_rows(rows, &ldec, lanes, &gq, sp.ratio, cap, host_cand);
    let (mut smis, mut scnt, mut selbad) = (0usize, 0usize, 0usize);
    for (i, h) in host.iter().enumerate() {
        for &(p, s) in &h.scores {
            scnt += 1;
            if dbg[i * cap + p as usize] != s { smis += 1; }
        }
        let n = cnt[i] as usize;
        let got: Vec<u32> = sel[i * TOPK..i * TOPK + n.min(TOPK)].iter().map(|&x| x as u32).collect();
        let pad_ok = sel[i * TOPK + n.min(TOPK)..(i + 1) * TOPK].iter().all(|&x| x == -1);
        if got != h.sel || !pad_ok { selbad += 1; }
    }
    let qfrac = qmis as f64 / (rows.len() * HEADS * HEAD_DIM) as f64;
    if (qmis > 0 || smis > 0) && !DIAG_DONE.swap(true, std::sync::atomic::Ordering::Relaxed) {
        // one-time diagnostic: row 0, head 0 and head 17, the GPU's prepared query against the host's
        let hq = host_qprep(&rows[0], inv);
        for h in [0usize, 17] {
            let (dg, dh) = (rf::decode2(&gq[0].codes[h * 64..h * 64 + 64]), rf::decode2(&hq.codes[h * 64..h * 64 + 64]));
            out.lines.push(format!("DIAG row0 head{h} scales gpu {:?} host {:?}", &gq[0].scales[h * 4..h * 4 + 4], &hq.scales[h * 4..h * 4 + 4]));
            out.lines.push(format!("DIAG row0 head{h} gpu  {:?}", dg));
            out.lines.push(format!("DIAG row0 head{h} host {:?}", dh));
            let iq: Vec<f32> = rows[0].iq[h * 128..h * 128 + 128].iter().map(|&b| unbf(b)).collect();
            out.lines.push(format!("DIAG row0 head{h} iq {:?}", iq));
        }
        let wv: Vec<f32> = rows[0].w.iter().map(|&b| unbf(b)).collect();
        out.lines.push(format!("DIAG row0 pos {} w {:?}", rows[0].pos, wv));
        let n = host[0].scores.len().min(16);
        let g16: Vec<f32> = (0..n).map(|k| rf::unsortable(dbg[host[0].scores[k].0 as usize])).collect();
        let h16: Vec<f32> = (0..n).map(|k| rf::unsortable(host[0].scores[k].1)).collect();
        out.lines.push(format!("DIAG row0 scores gpu  {:?}", g16));
        out.lines.push(format!("DIAG row0 scores host {:?}", h16));
        let qd = rf::QDec::new(&gq[0], &rows[0].w.iter().map(|&b| unbf(b)).collect::<Vec<_>>());
        let kd = &ldec[rows[0].lane];
        let ks = &lanes[rows[0].lane].ks;
        let dots: Vec<f32> = (0..32).map(|h| rf::dot_dec(&qd.v2[h * 128..h * 128 + 128], &qd.sc[h * 4..h * 4 + 4], &kd[..128], &ks[..4])).collect();
        out.lines.push(format!("DIAG row0 key0 host dots {:?} kscales {:?}", dots, &ks[..4]));
    }
    let mut ok = err == 0 && smis == 0 && selbad == 0 && qfrac < 1e-3;
    let mut extra = String::new();
    let mut ret_call: Option<(Call, Vec<Vec<u32>>)> = None;
    if sp.write_bmax {
        let bmax = gpu.dev.dtoh_sync_copy(&call.bmax)?;
        let candv = gpu.dev.dtoh_sync_copy(&call.cand)?;
        let ccnt = gpu.dev.dtoh_sync_copy(&call.cand_cnt)?;
        let nb_cap = cap.div_ceil(8);
        let (mut bbad, mut cbad) = (0, 0);
        let mut host_lists = Vec::new();
        for (i, h) in host.iter().enumerate() {
            let n = nvis(rows[i].pos, sp.ratio, cap);
            let nb = n.div_ceil(8);
            let mut hb = vec![0u16; nb];
            for &(p, s) in &h.scores { let b = p as usize / 8; hb[b] = hb[b].max(s); }
            if bmax[i * nb_cap..i * nb_cap + nb] != hb[..] { bbad += 1; }
            let hl = rf::select_blocks(&hb, n, CAND_BLOCKS, CAND_BLOCK);
            let gl: Vec<u32> = candv[i * CAND_BLOCKS..i * CAND_BLOCKS + ccnt[i] as usize].iter().map(|&x| x as u32).collect();
            if gl != hl { cbad += 1; }
            host_lists.push(hl);
        }
        // the union of each row block, with row masks
        let uni = gpu.dev.dtoh_sync_copy(&call.uni)?;
        let ucnt = gpu.dev.dtoh_sync_copy(&call.ucnt)?;
        let mut ubad = 0;
        let mut i = 0;
        for (b, &uc) in ucnt.iter().enumerate() {
            let nr = { let mut j = i; while j < rows.len() && rows[j].lane == rows[i].lane && j - i < sp.r { j += 1; } j - i };
            let mut m: std::collections::BTreeMap<u32, u32> = Default::default();
            for r in 0..nr { for &blk in &host_lists[i + r] { *m.entry(blk).or_default() |= 1 << r; } }
            let want: Vec<u32> = m.iter().map(|(&blk, &mask)| (blk << 8) | mask).collect();
            if uni[b * ucap..b * ucap + uc as usize] != want[..] { ubad += 1; }
            i += nr;
        }
        ok &= bbad == 0 && cbad == 0 && ubad == 0;
        extra = format!(", block maxima {} rows bad, candidate lists {} bad, unions {} bad", bbad, cbad, ubad);
        ret_call = Some((call, host_lists));
    }
    let (call, host_cand) = match ret_call { Some((c, h)) => (Some(c), h), None => (None, Vec::new()) };
    out.lines.push(format!("{} {name}: rows {}, err {err}, q-prep value mismatches {qmis} ({:.1e}), scores {smis} of {scnt} differ, selections {selbad} of {} differ{extra}",
                           if ok { "PASS" } else { "FAIL" }, rows.len(), qfrac, rows.len()));
    out.ok &= ok;
    Ok(CaseOut { sel, call, host_cand })
}

/// What a case hands to the next one: its selections, and for a candidate source its call (whose
/// union a mode-1 call visits) with the host's candidate lists.
struct CaseOut { sel: Vec<i32>, call: Option<Call>, host_cand: Vec<Vec<u32>> }

fn abi_check(gpu: &Gpu, out: &mut Outcome) -> Result<()> {
    let f = gpu.dev.get_func(super::super::MODULE, "d41_idx_abi").context("d41_idx_abi")?;
    let mut o = gpu.dev.htod_sync_copy(&[0i32; 10])?;
    use cudarc::driver::{LaunchAsync, LaunchConfig};
    unsafe { f.launch(LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 }, (&mut o,)) }?;
    let v = gpu.dev.dtoh_sync_copy(&o)?;
    let want = [std::mem::size_of::<IdxArgs>() as i32, std::mem::size_of::<IdxRb>() as i32,
                score_smem(1) as i32, score_smem(2) as i32, score_smem(4) as i32, score_smem(8) as i32,
                score_threads(1) as i32, score_threads(2) as i32, score_threads(4) as i32, score_threads(8) as i32];
    let ok = v == want && [1, 2, 4, 8].iter().all(|&r| score_smem(r) <= 101_376);
    out.lines.push(format!("{} abi: args {} B, rb {} B, smem r1/r2/r4/r8 = {:?} (host {:?}), threads {:?}", if ok { "PASS" } else { "FAIL" }, v[0], v[1], &v[2..6], &want[2..6], &v[6..]));
    out.ok &= ok;
    Ok(())
}

/// The synthetic gate suite.
fn suite(gpu: &Gpu, inv: &[f32; 32], out: &mut Outcome) -> Result<()> {
    let mut rng = Rng(0xD5_41_6B_49);
    let base = |r: usize, g: usize, mt: usize, ratio: i32| Spec { ratio, mode: 0, r, g, min_tiles: mt, write_bmax: false, in_f32: false, dbg: true, fused: r == 1 };
    // 1. decode, ratio 2, one row, one chunk; then the same row chunked and merged
    let lanes = vec![gen_lane(&mut rng, 3000, 4, 0)];
    let dl = upload_lanes(gpu, &lanes)?;
    let rows = vec![gen_row(&mut rng, 0, 5999, 2.0)];
    let s1 = run_case(gpu, "decode r2 1 row 1 chunk", &lanes, &dl, &rows, &base(1, 1, 8, 2), inv, None, None, out)?.sel;
    let s2 = run_case(gpu, "decode r2 1 row 8 chunks, fused merge", &lanes, &dl, &rows, &base(1, 8, 1, 2), inv, None, None, out)?.sel;
    let mut sp = base(1, 8, 1, 2); sp.fused = false;
    let s3 = run_case(gpu, "decode r2 1 row 8 chunks, merge kernel", &lanes, &dl, &rows, &sp, inv, None, None, out)?.sel;
    let s4 = run_case(gpu, "decode r2 1 row 23 chunks (sorted lists), fused merge", &lanes, &dl, &rows, &base(1, 23, 1, 2), inv, None, None, out)?.sel;
    let same = s1 == s2 && s1 == s3 && s1 == s4;
    out.lines.push(format!("{} chunked (8 / 23, fused / kernel) == unchunked (bit for bit)", if same { "PASS" } else { "FAIL" }));
    out.ok &= same;
    // 2. verify: 6 consecutive rows of one lane, every row-block width -> identical selections
    let lanes = vec![gen_lane(&mut rng, 9000, 4, 0)];
    let dl = upload_lanes(gpu, &lanes)?;
    let rows: Vec<RowIn> = (0..6).map(|k| gen_row(&mut rng, 0, 8990 + k, 2.0)).collect();
    let mut sels = Vec::new();
    for (r, g) in [(8, 1), (8, 6), (4, 3), (2, 2), (1, 5)] {
        sels.push(run_case(gpu, &format!("verify r1 6 rows R{r} G{g}"), &lanes, &dl, &rows, &base(r, g, 1, 1), inv, None, None, out)?.sel);
    }
    let inv_ok = sels.windows(2).all(|w| w[0] == w[1]);
    out.lines.push(format!("{} row-block width / chunking invariance (R 8/4/2/1, G 1..6)", if inv_ok { "PASS" } else { "FAIL" }));
    out.ok &= inv_ok;
    // 3. four lanes x 6 rows (24 rows), ratio 1, different lengths, chunked
    let lanes: Vec<Lane> = (0..4).map(|i| gen_lane(&mut rng, 4000 + 1500 * i, 4, 0)).collect();
    let dl = upload_lanes(gpu, &lanes)?;
    let rows: Vec<RowIn> = (0..4).flat_map(|l| { let n = 4000 + 1500 * l as i32; (0..6).map(|k| (l, n - 6 + k)).collect::<Vec<_>>() })
        .map(|(l, p)| gen_row(&mut rng, l, p, 2.0)).collect();
    run_case(gpu, "4 lanes x 6 rows r1 R8 G6", &lanes, &dl, &rows, &base(8, 6, 1, 1), inv, None, None, out)?;
    // 3b. TP key stripes: 4 stripes x 2 chunks into one gathered list buffer, merged = unsplit
    {
        let rows: Vec<RowIn> = (0..6).map(|k| gen_row(&mut rng, 0, 8990 + k, 2.0)).collect();
        let lanes = vec![gen_lane(&mut rng, 9000, 4, 0)];
        let dl = upload_lanes(gpu, &lanes)?;
        let whole = run_case(gpu, "stripe reference (unsplit) r1 R8", &lanes, &dl, &rows, &base(8, 1, 8, 1), inv, None, None, out)?.sel;
        let (n, gper) = (4usize, 2usize);
        let mut sp = base(8, n * gper, 1, 1); sp.dbg = false;
        let call = build(gpu, &dl, 9000, &rows, &sp, inv, 8 * CAND_BLOCKS)?;
        let mut a = call.a;
        a.g = gper as i32;
        let stripe = 9000usize.div_ceil(n).div_ceil(TILE) * TILE;
        for k in 0..n { score(&gpu.l, &stripe_args(&a, k, n, stripe))?; }
        let mut am = a; am.lists = (n * gper) as i32;
        merge(&gpu.l, &am)?;
        gpu.sync()?;
        let got = gpu.dev.dtoh_sync_copy(&call.sel)?;
        let err = gpu.dev.dtoh_sync_copy(&call.err)?[0];
        let ok = got == whole && err == 0;
        out.lines.push(format!("{} TP key stripes: 4 stripes x 2 chunks, gathered and merged == unsplit (bit for bit), err {err}", if ok { "PASS" } else { "FAIL" }));
        out.ok &= ok;
    }
    // 4. heavy ties (16 distinct key rows), short / empty rows, 512 boundaries
    let lanes = vec![gen_lane(&mut rng, 4000, 2, 16)];
    let dl = upload_lanes(gpu, &lanes)?;
    let rows: Vec<RowIn> = [0, 1, 100, 1023, 1024, 1025, 3999].iter().map(|&p| gen_row(&mut rng, 0, p, 2.0)).collect();
    run_case(gpu, "ties + short rows r2 R8 G1", &lanes, &dl, &rows, &base(8, 1, 8, 2), inv, None, None, out)?;
    run_case(gpu, "ties + short rows r2 R8 G4", &lanes, &dl, &rows, &base(8, 4, 1, 2), inv, None, None, out)?;
    let rows1: Vec<RowIn> = [0, 1, 100, 511, 512, 513, 3999].iter().map(|&p| gen_row(&mut rng, 0, p, 2.0)).collect();
    run_case(gpu, "ties + short rows r1 R8 G3", &lanes, &dl, &rows1, &base(8, 3, 1, 1), inv, None, None, out)?;
    // 4b. a cache capacity that ends inside a 4-key scale chunk (the real caches' lengths are arbitrary)
    for cap in [2999usize, 3001, 3002] {
        let lanes = vec![gen_lane(&mut rng, cap, 4, 0)];
        let dl = upload_lanes(gpu, &lanes)?;
        let rows: Vec<RowIn> = [cap as i32 - 1, 2 * cap as i32 - 1].iter().zip([1, 2]).map(|(&p, _)| gen_row(&mut rng, 0, p, 2.0)).collect();
        run_case(gpu, &format!("cache capacity {cap} (ratio 1, every key visible) R8 G2"), &lanes, &dl, &rows[..1], &base(8, 2, 1, 1), inv, None, None, out)?;
        run_case(gpu, &format!("cache capacity {cap} (ratio 2, every key visible) R8 G1"), &lanes, &dl, &rows[1..], &base(8, 1, 8, 2), inv, None, None, out)?;
    }
    // 5. prefill-like: 64 consecutive rows in blocks of 4
    let lanes = vec![gen_lane(&mut rng, 5000, 4, 0)];
    let dl = upload_lanes(gpu, &lanes)?;
    let rows: Vec<RowIn> = (0..64).map(|k| gen_row(&mut rng, 0, 9900 + k, 2.0)).collect();
    run_case(gpu, "prefill 64 rows r2 R4", &lanes, &dl, &rows, &base(4, 1, 8, 2), inv, None, None, out)?;
    // 6. f32 inputs; wide scale spreads (the MMA's round-toward-zero accumulation)
    let mut sp = base(4, 2, 1, 1); sp.in_f32 = true;
    let lanes6 = vec![gen_lane(&mut rng, 3000, 4, 0)];
    let dl6 = upload_lanes(gpu, &lanes6)?;
    let rows6: Vec<RowIn> = (0..3).map(|k| gen_row(&mut rng, 0, 2990 + k, 3.0)).collect();
    run_case(gpu, "f32 inputs r1 R4 G2", &lanes6, &dl6, &rows6, &sp, inv, None, None, out)?;
    let lanes7 = vec![gen_lane(&mut rng, 3000, 16, 0)];
    let dl7 = upload_lanes(gpu, &lanes7)?;
    run_case(gpu, "scale spread 16 r1 R4", &lanes7, &dl7, &rows6, &base(4, 1, 8, 1), inv, None, None, out)?;
    // 7. candidate source (> 16,384 positions: 2,048 of ~3,800 blocks kept), union, then mode 1
    for (nl, nr) in [(1usize, 6usize), (2, 3)] {
        let lanes: Vec<Lane> = (0..nl).map(|i| gen_lane(&mut rng, 30000 + 700 * i, 4, 0)).collect();
        let dl = upload_lanes(gpu, &lanes)?;
        let rows: Vec<RowIn> = (0..nl).flat_map(|l| (0..nr).map(move |k| (l, 30000 + 700 * l as i32 - nr as i32 + k as i32)))
            .map(|(l, p)| gen_row(&mut rng, l, p, 2.0)).collect();
        let mut sp20 = base(8, 4, 1, 1); sp20.write_bmax = true;
        let o20 = run_case(gpu, &format!("candidate source {nl} lane(s) x {nr} rows"), &lanes, &dl, &rows, &sp20, inv, None, None, out)?;
        let hc = o20.host_cand;
        let c20 = o20.call.context("layer-20 call")?;
        let rows2: Vec<RowIn> = rows.iter().map(|r| gen_row(&mut rng, r.lane, r.pos, 2.0)).collect();
        let mut sp1 = base(8, 3, 1, 1); sp1.mode = 1;
        run_case(gpu, &format!("candidate user (mode 1) {nl} lane(s) x {nr} rows"), &lanes, &dl, &rows2, &sp1, inv, Some(&c20), Some(hc.as_slice()), out)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// REFb fixture (tools/dsv41_ki/idx_fixture.py): real queries and keys of the 8 index layers.

fn st_vec<T: Copy + Default>(h: &crate::dsv41::st::StHeader, name: &str) -> Result<Vec<T>> {
    let b = crate::dsv41::st::read_tensor(h, name).with_context(|| format!("fixture tensor {name}"))?;
    let n = b.len() / std::mem::size_of::<T>();
    let mut v = vec![T::default(); n];
    unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), v.as_mut_ptr() as *mut u8, n * std::mem::size_of::<T>()); }
    Ok(v)
}

fn fixture(gpu: &Gpu, dir: &std::path::Path, inv: &[f32; 32], out: &mut Outcome) -> Result<()> {
    let path = dir.join("idx_fixture.safetensors");
    let h = crate::dsv41::st::read_header(&path).with_context(|| format!("{}", path.display()))?;
    let layers: Vec<usize> = h.metadata.get("layers").context("fixture metadata 'layers'")?
        .split(',').map(|s| s.trim().parse()).collect::<std::result::Result<_, _>>()?;
    let name = h.metadata.get("name").cloned().unwrap_or_default();
    let mut c20: Option<(Call, Vec<Vec<u32>>)> = None;
    let mut keys20: Option<(Lane, DLanes)> = None;
    for &l in &layers {
        let ratio: i32 = st_vec::<i32>(&h, &format!("L{l}.ratio"))?[0];
        let pos: Vec<i32> = st_vec(&h, &format!("L{l}.pos"))?;
        let iq: Vec<u16> = st_vec(&h, &format!("L{l}.iq"))?;
        let w: Vec<u16> = st_vec(&h, &format!("L{l}.w"))?;
        let refsel: Vec<i32> = st_vec(&h, &format!("L{l}.ref"))?;
        let nr = pos.len();
        let rows: Vec<RowIn> = (0..nr).map(|i| RowIn { lane: 0, pos: pos[i], iq: iq[i * 4096..(i + 1) * 4096].to_vec(), w: w[i * 32..(i + 1) * 32].to_vec() }).collect();
        let owns = h.tensors.iter().any(|t| t.name == format!("L{l}.kq"));
        let (lane, dl) = if owns {
            let lane = Lane { kq: st_vec(&h, &format!("L{l}.kq"))?, ks: st_vec(&h, &format!("L{l}.ks"))? };
            let dl = upload_lanes(gpu, std::slice::from_ref(&lane))?;
            (lane, dl)
        } else {
            let (lane, _) = keys20.as_ref().context("layers above 20 need layer 20's keys first")?;
            let dl = upload_lanes(gpu, std::slice::from_ref(lane))?;
            (lane.clone(), dl)
        };
        let lanes = vec![lane.clone()];
        let mut sp = Spec { ratio, mode: 0, r: 8, g: 1, min_tiles: 8, write_bmax: l == 20, in_f32: false, dbg: true, fused: false };
        let o = if l > 20 {
            sp.mode = 1;
            let (p, hc) = c20.as_ref().context("layer 20 must run first")?;
            run_case(gpu, &format!("REFb {name} L{l} (mode 1)"), &lanes, &dl, &rows, &sp, inv, Some(p), Some(hc.as_slice()), out)?
        } else {
            run_case(gpu, &format!("REFb {name} L{l}"), &lanes, &dl, &rows, &sp, inv, None, None, out)?
        };
        let sel = o.sel;
        if l == 20 {
            c20 = o.call.map(|c| (c, o.host_cand));
            keys20 = Some((lane.clone(), dl));
        }
        // against REFb's own selections (bf16 scores tie often: equality up to ties, overlap reported)
        let (mut eq, mut ov_min, mut ov_sum) = (0usize, 1.0f64, 0.0f64);
        for i in 0..nr {
            let a: std::collections::BTreeSet<i32> = sel[i * TOPK..(i + 1) * TOPK].iter().copied().filter(|&x| x >= 0).collect();
            let b: std::collections::BTreeSet<i32> = refsel[i * TOPK..(i + 1) * TOPK].iter().copied().filter(|&x| x >= 0).collect();
            if a == b { eq += 1; }
            let ov = if b.is_empty() { 1.0 } else { a.intersection(&b).count() as f64 / b.len() as f64 };
            ov_min = ov_min.min(ov); ov_sum += ov;
        }
        out.lines.push(format!("INFO REFb {name} L{l}: selections equal to REFb's on {eq} of {nr} rows; mean overlap {:.5}, min {:.4}", ov_sum / nr.max(1) as f64, ov_min));
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Timing.

struct Ev(cudarc::driver::sys::CUevent);
impl Ev {
    fn new() -> Result<Ev> {
        let mut e = std::ptr::null_mut();
        let r = unsafe { cudarc::driver::sys::cuEventCreate(&mut e, 0) };
        ensure!(r == cudarc::driver::sys::CUresult::CUDA_SUCCESS, "cuEventCreate {r:?}");
        Ok(Ev(e))
    }
    fn rec(&self, l: &Launch) { unsafe { cudarc::driver::sys::cuEventRecord(self.0, l.stream as cudarc::driver::sys::CUstream); } }
    fn ms(&self, start: &Ev) -> f32 {
        let mut ms = 0f32;
        unsafe { cudarc::driver::sys::cuEventSynchronize(self.0); cudarc::driver::sys::cuEventElapsedTime(&mut ms, start.0, self.0); }
        ms
    }
}

/// Median wall time (ms, CUDA events) of `f`, cold L2 (a 64 MB write before every rep).
fn time_cold(gpu: &Gpu, flush: &mut CudaSlice<u8>, reps: usize, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    let (e0, e1) = (Ev::new()?, Ev::new()?);
    let mut v = Vec::new();
    f()?; // warm-up (module, launch path)
    for _ in 0..reps {
        gpu.dev.memset_zeros(flush)?;
        e0.rec(&gpu.l);
        f()?;
        e1.rec(&gpu.l);
        v.push(e1.ms(&e0) as f64);
    }
    gpu.sync()?;
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(v[v.len() / 2])
}

fn bench(gpu: &Gpu, inv: &[f32; 32], max_ctx: usize, out: &mut Outcome) -> Result<()> {
    let mut rng = Rng(0xBE_7C_41);
    let mut flush = gpu.dev.htod_sync_copy(&vec![1u8; 64 << 20])?;
    let ctxs: Vec<usize> = [4096usize, 16384, 65536, 131072, 262144, 524288, 1048576].into_iter().filter(|&c| c <= max_ctx).collect();
    let cmax = *ctxs.last().context("--max-seq-len below 4096")?;
    let nl_max = 4;
    let lanes: Vec<Lane> = (0..nl_max).map(|_| gen_lane(&mut rng, cmax, 4, 0)).collect();
    let dl = upload_lanes(gpu, &lanes)?;
    out.lines.push("TIMING decode / verify rounds (cold L2, median of 5; key stream GB/s vs the 238 GB/s floor; one round = 3 ratio-2 layers + layer 20 + 4 candidate layers):".into());
    out.lines.push("| lanes x rows | context | ratio-2 layer us (GB/s) | layer 20 us (GB/s) | cand layer us | round us |".into());
    out.lines.push("|---|---:|---:|---:|---:|---:|".into());
    for (nl, rpl) in [(1usize, 1usize), (1, 2), (1, 6), (2, 6), (4, 1), (4, 6)] {
        for &c in &ctxs {
            let rows: Vec<RowIn> = (0..nl).flat_map(|l| (0..rpl).map(move |k| (l, (c - rpl + k) as i32))).map(|(l, p)| gen_row(&mut rng, l, p, 2.0)).collect();
            let r = score_entry(rpl).1;
            let g = (96 / nl).clamp(1, 96);
            let mk = |ratio: i32, mode: i32, wb: bool| Spec { ratio, mode, r, g, min_tiles: 8, write_bmax: wb, in_f32: false, dbg: false, fused: r == 1 };
            let c2 = build(gpu, &dl, cmax, &rows, &mk(2, 0, false), inv, r * CAND_BLOCKS)?;
            let c20 = build(gpu, &dl, cmax, &rows, &mk(1, 0, true), inv, r * CAND_BLOCKS)?;
            let mut c1 = build(gpu, &dl, cmax, &rows, &mk(1, 1, false), inv, r * CAND_BLOCKS)?;
            c1.a.uni = c20.a.uni; c1.a.ucnt = c20.a.ucnt;
            let t2 = time_cold(gpu, &mut flush, 5, || select(&gpu.l, &c2.a, false))?;
            let t20 = time_cold(gpu, &mut flush, 5, || select(&gpu.l, &c20.a, true))?;
            let t1 = time_cold(gpu, &mut flush, 5, || select(&gpu.l, &c1.a, false))?;
            let err = gpu.dev.dtoh_sync_copy(&c1.err)?[0] | gpu.dev.dtoh_sync_copy(&c20.err)?[0] | gpu.dev.dtoh_sync_copy(&c2.err)?[0];
            ensure!(err == 0, "bench call reported error bits {err}");
            let b2 = (nl * (c / 2) * 68) as f64;
            let b20 = (nl * c * 68) as f64;
            let round = 3.0 * t2 + t20 + 4.0 * t1;
            out.lines.push(format!("| {nl} x {rpl} | {}K | {:.1} ({:.0}) | {:.1} ({:.0}) | {:.1} | {:.1} |", c / 1024, t2 * 1e3, b2 / (t2 * 1e-3) / 1e9,
                                   t20 * 1e3, b20 / (t20 * 1e-3) / 1e9, t1 * 1e3, round * 1e3));
        }
    }
    // chunk-count sweep at the longest context, one row
    let rows: Vec<RowIn> = vec![gen_row(&mut rng, 0, (cmax - 1) as i32, 2.0)];
    let mut line = format!("TIMING chunk sweep, 1 x 1 row, ratio 1, {}K keys:", cmax / 1024);
    for g in [12usize, 24, 48, 96, 192, 384] {
        let c = build(gpu, &dl, cmax, &rows, &Spec { ratio: 1, mode: 0, r: 1, g, min_tiles: 8, write_bmax: false, in_f32: false, dbg: false, fused: true }, inv, CAND_BLOCKS)?;
        let t = time_cold(gpu, &mut flush, 5, || select(&gpu.l, &c.a, false))?;
        line += &format!(" G{g} {:.1} us ({:.0} GB/s);", t * 1e3, (cmax * 68) as f64 / (t * 1e-3) / 1e9);
    }
    out.lines.push(line);
    // prefill: 1,024 consecutive rows at the end of the context, ratio 2, row blocks of 4, one chunk
    out.lines.push("TIMING prefill (ratio-2 layer, 1,024 rows at the end of the context, R8 blocks, one chunk; FP4 MMA work = 8,192 FLOP per key-row):".into());
    out.lines.push("| context | ms per 1,024 rows | key-rows / s | TFLOPS |".into());
    out.lines.push("|---:|---:|---:|---:|".into());
    for &c in &ctxs {
        let rows: Vec<RowIn> = (0..1024).map(|k| gen_row(&mut rng, 0, (c - 1024 + k) as i32, 2.0)).collect();
        let call = build(gpu, &dl, cmax, &rows, &Spec { ratio: 2, mode: 0, r: 8, g: 1, min_tiles: 8, write_bmax: false, in_f32: false, dbg: false, fused: false }, inv, 8 * CAND_BLOCKS)?;
        let t = time_cold(gpu, &mut flush, 3, || score(&gpu.l, &call.a))?;
        let kr: f64 = rows.iter().map(|r| nvis(r.pos, 2, cmax) as f64).sum();
        out.lines.push(format!("| {}K | {:.2} | {:.3e} | {:.0} |", c / 1024, t, kr / (t * 1e-3), kr * 8192.0 / (t * 1e-3) / 1e12));
    }
    Ok(())
}

/// The probe: gates, then the REFb fixture (if given), then timing. Returns the result lines; the
/// last line is `DSV41_KERNEL_INDEXER_OK` when every gate passed.
pub fn run(cfg: Option<&Dsv41Config>, fixture_dir: Option<&std::path::Path>, max_ctx: usize, timing: bool) -> Result<Vec<String>> {
    if let Some(c) = cfg { check_config(c)?; }
    let inv = match cfg { Some(c) => rope_inv_freq(c)?, None => {
        let v = rf::rope_inv_freq(64, 160000.0, 16.0, 65536.0, 32.0, 1.0);
        let mut a = [0f32; 32]; a.copy_from_slice(&v); a
    } };
    let gpu = Gpu::new()?;
    let mut out = Outcome { lines: Vec::new(), ok: true };
    abi_check(&gpu, &mut out)?;
    suite(&gpu, &inv, &mut out)?;
    if let Some(d) = fixture_dir { fixture(&gpu, d, &inv, &mut out)?; }
    if timing && out.ok && max_ctx >= 4096 { bench(&gpu, &inv, max_ctx, &mut out)?; }
    if !out.ok { out.lines.push("DSV41_KERNEL_INDEXER_FAIL".into()); bail!("{}", out.lines.join("\n")); }
    out.lines.push("DSV41_KERNEL_INDEXER_OK".into());
    Ok(out.lines)
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib dsv41_idx_gpu -- --ignored"]
    fn dsv41_idx_gpu_gates() {
        let lines = super::run(None, None, 0, false).unwrap();
        for l in &lines { eprintln!("{l}"); }
        assert_eq!(lines.last().map(String::as_str), Some("DSV41_KERNEL_INDEXER_OK"));
    }
}
