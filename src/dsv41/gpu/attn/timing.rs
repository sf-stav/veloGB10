//! The timing table of the sparse attention core (leg KA), `--dsv41-kernel-timing 4k,32k,128k,512k,1m`:
//! decode / verify calls at the TP head ranges (16 heads = TP4, 32 = TP2, 8 = TP8) and 1 / 2 / 6 / 12 / 24
//! rows (lanes of six verify rows), and 2,048-row prefill calls at 64 heads (TP4 / TP8 row-parallel) and 32
//! (TP2 head split), against caches of the ratio-1 decoder class sized to the context (one compressed
//! row per token: the largest cache; ratio-2 layers read half as many rows from the same 512 picks).
//! Selections are 512 random positions per row; the rows of one lane share 15 of every 16 picks (a lane's
//! verify rows, neighbouring prefill rows). Each cell is timed twice with CUDA events: HOT (back-to-back
//! launches, caches L2-resident where they fit) and COLD (a 128 MB write flushes the 24 MB L2 before every
//! launch, as a decode round's weight stream does between two visits to a layer). Values are random codes:
//! the timing does not depend on them. Reported: microseconds per call, TFLOP/s (4 x heads x 512 x keys x
//! rows), the unique KV bytes a call must read and that over the cold time (against the 238 GB/s roofline).

use super::probe::{launch_of, sync, Bufs, Rng};
use super::{AttnCall, DecodeScratch, LaneSrc};
use crate::dsv41::weights::DevPtr;
use anyhow::{bail, ensure, Context, Result};
use cudarc::driver::CudaDevice;
use std::collections::BTreeSet;
use std::sync::Arc;

fn parse_ctx(s: &str) -> Result<usize> {
    let t = s.trim().to_ascii_lowercase();
    let (n, m) = if let Some(x) = t.strip_suffix('m') { (x, 1usize << 20) } else if let Some(x) = t.strip_suffix('k') { (x, 1024) } else { (t.as_str(), 1) };
    Ok(n.parse::<usize>().with_context(|| format!("--dsv41-kernel-timing: bad context {s:?}"))? * m)
}

fn random_bytes(rng: &mut Rng, n: usize, f: impl Fn(u8) -> u8) -> Vec<u8> {
    let mut v = vec![0u8; n];
    for c in v.chunks_mut(8) { let x = rng.next().to_le_bytes(); for (a, b) in c.iter_mut().zip(x) { *a = f(b); } }
    v
}
// E4M3 codes without NaN; E8M0 scales near 2^-3; KV records with valid nibbles and E4M3 scales near 0.1
fn fp8_code(b: u8) -> u8 { if b & 0x7F == 0x7F { b ^ 1 } else { b } }

struct Ev(cudarc::driver::sys::CUevent);
impl Ev {
    fn new() -> Result<Ev> {
        let mut e = std::ptr::null_mut();
        let r = unsafe { cudarc::driver::sys::cuEventCreate(&mut e, 0) };
        ensure!(r == cudarc::driver::sys::CUresult::CUDA_SUCCESS, "cuEventCreate {r:?}");
        Ok(Ev(e))
    }
    fn rec(&self, st: u64) -> Result<()> {
        let r = unsafe { cudarc::driver::sys::cuEventRecord(self.0, st as cudarc::driver::sys::CUstream) };
        ensure!(r == cudarc::driver::sys::CUresult::CUDA_SUCCESS, "cuEventRecord {r:?}");
        Ok(())
    }
    fn ms(&self, start: &Ev) -> Result<f32> {
        unsafe { cudarc::driver::sys::cuEventSynchronize(self.0) };
        let mut ms = 0f32;
        let r = unsafe { cudarc::driver::sys::cuEventElapsedTime(&mut ms, start.0, self.0) };
        ensure!(r == cudarc::driver::sys::CUresult::CUDA_SUCCESS, "cuEventElapsedTime {r:?}");
        Ok(ms)
    }
}
impl Drop for Ev { fn drop(&mut self) { unsafe { cudarc::driver::sys::cuEventDestroy_v2(self.0) }; } }

struct Cell { name: String, heads: usize, rows: usize, keys: f64, unique: f64, hot_us: f64, cold_us: f64 }

/// Time `go` (one launch on the stream): hot = average of back-to-back launches, cold = median of single
/// launches each after an L2 flush.
fn time_it(dev: &Arc<CudaDevice>, flush: (u64, usize), go: &mut dyn FnMut() -> Result<()>) -> Result<(f64, f64)> {
    let st = launch_of(dev).stream;
    let (a, b) = (Ev::new()?, Ev::new()?);
    for _ in 0..20 { go()?; }
    sync(dev)?;
    let n = 50;
    a.rec(st)?;
    for _ in 0..n { go()?; }
    b.rec(st)?;
    let hot = b.ms(&a)? as f64 * 1e3 / n as f64;
    let mut cold = Vec::new();
    for _ in 0..25 {
        let r = unsafe { cudarc::driver::sys::cuMemsetD8Async(flush.0, 0x5A, flush.1, st as cudarc::driver::sys::CUstream) };
        ensure!(r == cudarc::driver::sys::CUresult::CUDA_SUCCESS, "L2 flush memset {r:?}");
        a.rec(st)?;
        go()?;
        b.rec(st)?;
        cold.push(b.ms(&a)? as f64 * 1e3);
    }
    cold.sort_by(|x, y| x.partial_cmp(y).unwrap());
    Ok((hot, cold[cold.len() / 2]))
}

/// Selections for `rows` rows in groups of `per_lane`: a group shares a base set of 512 picks below `reach`,
/// each row replaces one of every 16 picks with its own.
fn selections(rng: &mut Rng, rows: usize, per_lane: usize, reach: usize) -> (Vec<i32>, usize) {
    let n = reach.min(512);
    let mut out = vec![-1i32; rows * 512];
    let mut unique = 0usize;
    let mut r0 = 0;
    while r0 < rows {
        let mut base = BTreeSet::new();
        while base.len() < n { base.insert(rng.below(reach) as i32); }
        let base: Vec<i32> = base.into_iter().collect();
        let mut uni: BTreeSet<i32> = base.iter().cloned().collect();
        for r in r0..(r0 + per_lane).min(rows) {
            let mut s: BTreeSet<i32> = base.iter().cloned().collect();
            for k in (0..n).step_by(16) {
                let x = rng.below(reach) as i32;
                if !s.contains(&x) { s.remove(&base[k]); s.insert(x); }
            }
            uni.extend(s.iter().cloned());
            for (i, v) in s.into_iter().enumerate() { out[r * 512 + i] = v; }
        }
        unique += uni.len();
        r0 += per_lane;
    }
    (out, unique)
}

pub fn run(dev: &Arc<CudaDevice>, spec: &str) -> Result<Vec<String>> {
    let ctxs: Vec<usize> = spec.split(',').filter(|s| !s.trim().is_empty()).map(parse_ctx).collect::<Result<_>>()?;
    ensure!(!ctxs.is_empty(), "--dsv41-kernel-timing: no contexts");
    let mut rng = Rng(0x7131_D41A);
    let mut lines = vec!["timing (HOT = back-to-back launches, COLD = after a 128 MB L2 flush; keys = 128 window + 512 selected per row):".to_string()];
    lines.push("| context | entry | heads | rows | HOT us | COLD us | TFLOP/s (cold) | unique KV MB | GB/s (cold) |".into());
    lines.push("|---:|---|---:|---:|---:|---:|---:|---:|---:|".into());
    let mut flush_buf = Bufs::new(dev);
    let flush = (flush_buf.u8(&vec![0u8; 128 << 20])?.0, 128usize << 20);
    for &ctx in &ctxs {
        ensure!(ctx >= 4096, "--dsv41-kernel-timing: contexts of at least 4k (got {ctx})");
        let mut b = Bufs::new(dev);
        // four lanes: ring of 134 slots ending at ctx - 1, a ratio-1 compressed cache of ctx records
        let mut lanes = Vec::new();
        for _ in 0..4 {
            let rq = random_bytes(&mut rng, 134 * 512, fp8_code);
            let rs = random_bytes(&mut rng, 134 * 16, |x| 120 + (x & 3));
            let kv = random_bytes(&mut rng, ctx * 288, |x| x);
            let mut kv = kv;
            for r in kv.chunks_exact_mut(288) { for s in &mut r[256..] { *s = 0x28 + (*s & 7); } }
            lanes.push(LaneSrc { swa_q: b.u8(&rq)?, swa_s: b.u8(&rs)?, slots: 134, lin: None, kv: Some((b.u8(&kv)?, ctx)), ext: None });
        }
        let freq = super::rope_inv_freq(160000.0, Some((16.0, 65536.0, 32.0, 1.0)))?;
        let mut cells: Vec<Cell> = Vec::new();
        // ---- decode / verify ----
        for &heads in &[16usize, 32, 8] {
            for &rows in &[1usize, 2, 6, 12, 24] {
                let per = if rows <= 2 { rows } else { 6 };
                let nl = rows.div_ceil(per);
                let (sel, uni) = selections(&mut rng, rows, per, ctx);
                let pos: Vec<i32> = (0..rows).map(|r| (ctx - per + r % per) as i32).collect();
                let lane: Vec<i32> = (0..rows).map(|r| (r / per) as i32).collect();
                let q: Vec<u16> = (0..rows * heads * 512).map(|_| super::reference::bf16_bits(rng.gauss() * 0.5)).collect();
                let sink: Vec<f32> = (0..heads).map(|_| rng.gauss()).collect();
                let out = b.u16(&vec![0u16; rows * heads * 512])?;
                let call = AttnCall { rows, heads, q: b.u16(&q)?, q_stride: heads * 512, sink: b.f32(&sink)?, pos: b.i32(&pos)?, win_end: DevPtr(0),
                                      lane: b.i32(&lane)?, sel: b.i32(&sel)?, sel_stride: 512, sel_cap: 512, n_sel: DevPtr(0), lanes: lanes[..nl].to_vec(),
                                      out, out_stride: heads * 512, out_q: None, out_s: None, epi: super::EPI_ROPE | super::EPI_AQ, rope_freq: freq, err: DevPtr(0) };
                let ns = call.nsplit();
                let (pb, cb) = DecodeScratch::bytes(rows, heads, ns);
                let sc = DecodeScratch { part: b.f32(&vec![0f32; pb / 4])?, cnt: b.i32(&vec![0i32; cb / 4])?, rows, groups: heads.div_ceil(16), nsplit: ns };
                let l = launch_of(dev);
                let (hot, cold) = time_it(dev, flush, &mut || super::launch_decode(l, &call, &sc))?;
                let unique = (nl * (127 + per)) as f64 * 528.0 + uni as f64 * 288.0;
                cells.push(Cell { name: "decode".into(), heads, rows, keys: 640.0, unique, hot_us: hot, cold_us: cold });
            }
        }
        // ---- prefill (2,048 consecutive rows ending at ctx - 1, window from linear rows) ----
        {
            let rows = 2048usize;
            let p0 = ctx - rows;
            let lin_rows = rows + 127;
            let lq = random_bytes(&mut rng, lin_rows * 512, fp8_code);
            let ls = random_bytes(&mut rng, lin_rows * 16, |x| 120 + (x & 3));
            let mut l0 = lanes[0];
            l0.lin = Some((b.u8(&lq)?, b.u8(&ls)?, (p0 - 127) as i64));
            let (sel, uni) = selections(&mut rng, rows, 64, p0);
            for &heads in &[64usize, 32] {
                let pos: Vec<i32> = (0..rows).map(|r| (p0 + r) as i32).collect();
                let q: Vec<u16> = (0..rows * heads * 512).map(|_| super::reference::bf16_bits(rng.gauss() * 0.5)).collect();
                let sink: Vec<f32> = (0..heads).map(|_| rng.gauss()).collect();
                let out = b.u16(&vec![0u16; rows * heads * 512])?;
                let call = AttnCall { rows, heads, q: b.u16(&q)?, q_stride: heads * 512, sink: b.f32(&sink)?, pos: b.i32(&pos)?, win_end: DevPtr(0),
                                      lane: DevPtr(0), sel: b.i32(&sel)?, sel_stride: 512, sel_cap: 512, n_sel: DevPtr(0), lanes: vec![l0],
                                      out, out_stride: heads * 512, out_q: None, out_s: None, epi: super::EPI_ROPE | super::EPI_AQ, rope_freq: freq, err: DevPtr(0) };
                let l = launch_of(dev);
                let (hot, cold) = time_it(dev, flush, &mut || super::launch_prefill(l, &call))?;
                let unique = lin_rows as f64 * 528.0 + uni as f64 * 288.0;
                cells.push(Cell { name: "prefill".into(), heads, rows, keys: 640.0, unique, hot_us: hot, cold_us: cold });
            }
        }
        for c in &cells {
            let flops = 4.0 * c.heads as f64 * 512.0 * c.keys * c.rows as f64;
            lines.push(format!("| {} | {} | {} | {} | {:.1} | {:.1} | {:.2} | {:.2} | {:.1} |", if ctx >= 1 << 20 { format!("{}M", ctx >> 20) } else { format!("{}K", ctx >> 10) },
                               c.name, c.heads, c.rows, c.hot_us, c.cold_us, flops / (c.cold_us * 1e-6) / 1e12, c.unique / 1e6, c.unique / (c.cold_us * 1e-6) / 1e9));
        }
        sync(dev)?;
        drop(b);
    }
    if lines.len() < 4 { bail!("timing produced no cells"); }
    Ok(lines)
}
