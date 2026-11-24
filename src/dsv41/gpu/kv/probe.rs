//! Leg KC probe: `--probe-dsv41-kernel --dsv41-kernel-set kv --dsv41-dump-ref <fixture dir>`.
//!
//! The fixture (`tools/dsv41_ref/kc_fixture.py`) holds, per scenario, DeepSeek's reference cache contents
//! for the first n tokens (single shot, torch on the GPU, REFb's bitwise-exact fp4 / fp8 casts) and the
//! projection inputs. The probe REPLAYS a call sequence (chunked prefill, decode, verify rounds with
//! rejected rows that later calls recompute) through the four kernels into fresh caches and compares the
//! final caches BYTE FOR BYTE: main-KV records, index-K planes, SWA ring slots of the last 134 positions,
//! and, per call, the pre-RoPE latent and the SWA chunk rows. It also checks the arg-struct ABI, the Rust
//! rotary tables against torch's, and prints the byte-floor timing table.

use super::*;
use crate::dsv41::st;
use anyhow::{bail, Result};
use cudarc::driver::{CudaSlice, DevicePtr};
use std::path::Path;

const R_EXTRA: i32 = 5; // frontier ring rows beyond the ratio (backtracking 6 - 1)

struct Fx { h: st::StHeader }
impl Fx {
    fn bytes(&self, n: &str) -> Result<Vec<u8>> { st::read_tensor(&self.h, n) }
    fn shape(&self, n: &str) -> Result<Vec<u64>> { Ok(self.h.tensors.iter().find(|t| t.name == n).with_context(|| format!("no tensor {n}"))?.shape.clone()) }
}

fn ptr<T>(s: &CudaSlice<T>) -> u64 { *s.device_ptr() as u64 }
fn up(dev: &Arc<CudaDevice>, b: &[u8]) -> Result<CudaSlice<u8>> {
    let v: Vec<u8> = if b.is_empty() { vec![0u8; 16] } else { b.to_vec() };
    Ok(dev.htod_sync_copy(&v)?)
}
fn filled(dev: &Arc<CudaDevice>, n: usize, v: u8) -> Result<CudaSlice<u8>> { Ok(dev.htod_sync_copy(&vec![v; n.max(16)])?) }
fn down(dev: &Arc<CudaDevice>, s: &CudaSlice<u8>, n: usize) -> Result<Vec<u8>> { let mut v = dev.dtoh_sync_copy(s)?; v.truncate(n); Ok(v) }
fn rows<'a>(b: &'a [u8], row_bytes: usize, from: usize, n: usize) -> &'a [u8] { &b[from * row_bytes..(from + n) * row_bytes] }

/// Compare two byte slices; returns the number of differing bytes (and appends a description of the first few).
fn cmp(what: &str, got: &[u8], want: &[u8], row_bytes: usize, notes: &mut Vec<String>) -> usize {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let bad: Vec<usize> = (0..got.len()).filter(|&i| got[i] != want[i]).collect();
    if !bad.is_empty() && notes.len() < 24 {
        let i = bad[0];
        notes.push(format!("{what}: {} of {} bytes differ; first at row {} byte {} (got {:#04x}, want {:#04x})", bad.len(), got.len(), i / row_bytes, i % row_bytes, got[i], want[i]));
    }
    bad.len()
}

struct Tally { checks: usize, bytes: usize, bad: usize, notes: Vec<String>, lat_vals: usize, lat_ulp1: usize }
impl Tally {
    /// bf16 latent rows: the RMSNorm mean is a fp32 reduction whose order torch does not publish, so a value may sit one
    /// bf16 ulp off. Counted and reported (never hidden); anything beyond 1 ulp is a failure.
    fn add_bf16_ulp1(&mut self, what: &str, got: &[u8], want: &[u8]) {
        self.checks += 1;
        for (a, b) in got.chunks(2).zip(want.chunks(2)) {
            let (x, y) = (u16::from_le_bytes([a[0], a[1]]) as i32, u16::from_le_bytes([b[0], b[1]]) as i32);
            self.lat_vals += 1;
            if x != y {
                self.lat_ulp1 += 1;
                if (x - y).abs() > 1 { self.bad += 1; if self.notes.len() < 24 { self.notes.push(format!("{what}: bf16 values differ by {} ulp", (x - y).abs())); } }
            }
        }
    }
    fn add(&mut self, what: &str, got: &[u8], want: &[u8], row_bytes: usize) {
        self.checks += 1;
        self.bytes += got.len();
        self.bad += cmp(what, got, want, row_bytes, &mut self.notes);
    }
}

pub fn run(dir: &Path) -> Result<Vec<String>> {
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    let kk = KvKernels::new(&dev)?;
    let mut out = vec![];
    // 1. ABI of the argument structs
    {
        let f = dev.get_func(super::super::MODULE, "d41_kc_sizes").context("d41_kc_sizes")?;
        let mut o = dev.alloc_zeros::<u32>(6)?;
        unsafe { f.launch(LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 }, (&mut o,)) }?;
        let got = dev.dtoh_sync_copy(&o)?;
        ensure!(got == SIZES, "KC arg-struct ABI mismatch: device {got:?} host {SIZES:?}");
        out.push(format!("arg structs agree with the device (sizes {:?})", SIZES));
    }
    let fx = Fx { h: st::read_header(&dir.join("kc.safetensors"))? };
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("kc.json"))?)?;
    // 2. rotary tables: ours vs torch's
    let cfg_rope = |compressed: bool| -> Result<Vec<f32>> {
        if compressed { rope::inv_freq(64, 65536, 160000.0, 16.0, 32.0, 1.0) } else { rope::inv_freq(64, 0, 10000.0, 16.0, 32.0, 1.0) }
    };
    for (name, comp) in [("inv_freq.compress", true), ("inv_freq.plain", false)] {
        let want: Vec<f32> = fx.bytes(name)?.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let got = cfg_rope(comp)?;
        let eq = got.iter().zip(&want).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
        let max_ulp = got.iter().zip(&want).map(|(a, b)| (a.to_bits() as i64 - b.to_bits() as i64).abs()).max().unwrap_or(0);
        out.push(format!("rotary table {name}: {eq}/32 bit-equal to torch, max {max_ulp} ulp"));
        ensure!(max_ulp <= 1, "rotary table {name} differs from torch by {max_ulp} ulp");
    }
    // 3. scenarios
    let mut tally = Tally { checks: 0, bytes: 0, bad: 0, notes: vec![], lat_vals: 0, lat_ulp1: 0 };
    let inv = up(&dev, &fx.bytes("inv_freq.compress")?)?;
    let scenarios = meta["scenarios"].as_array().context("kc.json scenarios")?.clone();
    for sc in &scenarios {
        let name = sc["name"].as_str().unwrap().to_string();
        for fmt in [NORM_BF16, NORM_F32] {
            if fmt == NORM_F32 && !(name == "r2_chunk_decode_verify" || name == "r1_chunk_decode_verify") { continue; }
            replay(&dev, &kk, &fx, sc, &inv, fmt, &mut tally, &mut out)?;
        }
    }
    out.push(format!("compared {} caches / slices, {} bytes: {} bytes differ (cache records and planes, exact); pre-RoPE latent: {} of {} bf16 values one ulp off (RMSNorm reduction order)", tally.checks, tally.bytes, tally.bad, tally.lat_ulp1, tally.lat_vals));
    for n in &tally.notes { out.push(format!("  MISMATCH {n}")); }
    if tally.bad > 0 { bail!("{}\nKC cache contents differ from the reference in {} bytes", out.join("\n"), tally.bad); }
    // 4. timing / byte floor
    out.extend(timing(&dev, &kk)?);
    out.push("DSV41_KC_OK".into());
    Ok(out)
}

fn wbytes(b: &[u8], fmt: u32) -> Vec<u8> {
    if fmt == NORM_F32 { b.chunks(2).flat_map(|c| (f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).to_le_bytes()).collect() } else { b.to_vec() }
}

#[allow(clippy::too_many_arguments)]
fn replay(dev: &Arc<CudaDevice>, kk: &KvKernels, fx: &Fx, sc: &serde_json::Value, inv: &CudaSlice<u8>, nfmt: u32, t: &mut Tally, out: &mut Vec<String>) -> Result<()> {
    let name = sc["name"].as_str().unwrap();
    let ratio = sc["ratio"].as_i64().unwrap() as i32;
    let n = sc["n"].as_u64().unwrap() as usize;
    let groups = sc["groups"].as_u64().unwrap() as usize;
    let calls: Vec<(u64, u64, u64)> = sc["calls"].as_array().unwrap().iter().map(|c| (c[0].as_u64().unwrap(), c[1].as_u64().unwrap(), c[2].as_u64().unwrap())).collect();
    let tag = |s: &str| format!("{name}.{s}");
    let g = |alt: bool, s: &str| -> Result<Vec<u8>> { fx.bytes(&format!("{name}.{}{s}", if alt { "alt." } else { "" })) };
    let (nw, ikw, kvw) = (wbytes(&fx.bytes(&tag("norm_w"))?, nfmt), wbytes(&fx.bytes(&tag("ik_norm_w"))?, nfmt), wbytes(&fx.bytes(&tag("kv_norm_w"))?, nfmt));
    let (d_nw, d_ikw, d_kvw) = (up(dev, &nw)?, up(dev, &ikw)?, up(dev, &kvw)?);
    let r = ratio as usize;
    let ring_rows = if r >= 2 { ratio + R_EXTRA } else { 0 };
    let cap_groups = n / r + 8;
    let ring_kv = filled(dev, ring_rows as usize * DIM * 4, 0xFF)?;
    let ring_sc = filled(dev, ring_rows as usize * DIM * 4, 0xFF)?;
    let main = filled(dev, cap_groups * 288, 0xA5)?;
    let ik_q = filled(dev, cap_groups * 64, 0xA5)?;
    let ik_s = filled(dev, cap_groups * 4, 0xA5)?;
    let swa_rq = filled(dev, 134 * 512, 0xA5)?;
    let swa_rs = filled(dev, 134 * 16, 0xA5)?;
    let mut err = dev.alloc_zeros::<i32>(1)?;
    let (e_lat, e_main) = (fx.bytes(&tag("x.latent"))?, fx.bytes(&tag("x.kv_main"))?);
    let (e_ikq, e_iks) = (fx.bytes(&tag("x.ik_q"))?, fx.bytes(&tag("x.ik_s"))?);
    let (e_swq, e_sws) = (fx.bytes(&tag("x.swa_q"))?, fx.bytes(&tag("x.swa_s"))?);
    let (kv_p, kv_a, sc_p, sc_a, k16_p, k16_a);
    if r >= 2 { kv_p = g(false, "kv")?; kv_a = g(true, "kv")?; sc_p = g(false, "score")?; sc_a = g(true, "score")?; k16_p = vec![]; k16_a = vec![]; }
    else { kv_p = vec![]; kv_a = vec![]; sc_p = vec![]; sc_a = vec![]; k16_p = g(false, "kv16")?; k16_a = g(true, "kv16")?; }
    let (kr_p, kr_a, sw_p, sw_a) = (g(false, "k_raw")?, g(true, "k_raw")?, g(false, "swa_raw")?, g(true, "swa_raw")?);
    let pick = |p: &[u8], a: &[u8], rb: usize, first: usize, cnt: usize, acc_end: usize| -> Vec<u8> {
        // rows first..first+cnt: primary while the row index < acc_end, else the rejected-draft tensor
        let mut v = Vec::with_capacity(cnt * rb);
        for i in first..first + cnt { v.extend_from_slice(&(if i < acc_end { p } else { a })[i * rb..(i + 1) * rb]); }
        v
    };
    let _ = (&kvw, &ikw);
    let ctag = format!("{name}[{}]", if nfmt == NORM_F32 { "f32 norms" } else { "bf16 norms" });
    for &(p0, nrows, acc) in &calls {
        let (p0u, nr, acc_end) = (p0 as usize, nrows as usize, (p0 + acc) as usize);
        let (gf, gc) = group_range(p0, nrows, ratio as u32);
        let (gf, gc) = (gf as usize, gc as usize);
        // ---- compress + stash
        let (d_kv, d_sc, d_k16);
        if r >= 2 {
            d_kv = up(dev, &pick(&kv_p, &kv_a, 2048, p0u, nr, acc_end))?;
            d_sc = up(dev, &pick(&sc_p, &sc_a, 2048, p0u, nr, acc_end))?;
            d_k16 = up(dev, &[])?;
        } else { d_kv = up(dev, &[])?; d_sc = up(dev, &[])?; d_k16 = up(dev, &pick(&k16_p, &k16_a, 1024, p0u, nr, acc_end))?; }
        let lat = filled(dev, gc.max(1) * 1024, 0xCD)?;
        let ca = CompressArgs { kv: ptr(&d_kv), score: ptr(&d_sc), kv16: ptr(&d_k16), ring_kv: ptr(&ring_kv), ring_score: ptr(&ring_sc), norm_w: ptr(&d_nw),
            inv_freq: ptr(inv), latent_out: ptr(&lat), kv_main: ptr(&main), p0: p0 as i64, rows: nr as i32, ratio, ring_rows, g_first: gf as i32, n_groups: gc as i32,
            norm_fmt: nfmt, eps: 1e-20, err: ptr_i32(&err) };
        // the compress kernel reads the call's rows (kv at row q - p0): for r = 1 only kv16
        kk.compress(&ca)?;
        if r >= 2 { kk.stash(&StashArgs { kv: ptr(&d_kv), score: ptr(&d_sc), ring_kv: ptr(&ring_kv), ring_score: ptr(&ring_sc), p0: p0 as i64, rows: nr as i32, ring_rows })?; }
        dev.synchronize()?;
        // per-call latent check for groups whose members are all accepted
        let got_lat = down(dev, &lat, gc * 1024)?;
        for j in 0..gc {
            let gi = gf + j;
            if gi < groups && ((gi + 1) * r) <= acc_end { t.add_bf16_ulp1(&format!("{ctag} call p0={p0} latent g{gi}"), rows(&got_lat, 1024, j, 1), rows(&e_lat, 1024, gi, 1)); }
        }
        // ---- index-K: input row g from the accepted / rejected tensor by the group's last position
        if gc > 0 {
            let mut kr = Vec::with_capacity(gc * 256);
            for gi in gf..gf + gc { let src = if (gi + 1) * r <= acc_end { &kr_p } else { &kr_a }; kr.extend_from_slice(&src[gi * 256..(gi + 1) * 256]); }
            let d_kr = up(dev, &kr)?;
            kk.index_k(&IndexArgs { k_raw: ptr(&d_kr), norm_w: ptr(&d_ikw), inv_freq: ptr(inv), q_plane: ptr(&ik_q), s_plane: ptr(&ik_s), ratio, g_first: gf as i32, n_groups: gc as i32, norm_fmt: nfmt, eps: 1e-20 })?;
        }
        // ---- SWA
        let d_sw = up(dev, &pick(&sw_p, &sw_a, 1024, p0u, nr, acc_end))?;
        let ch_q = filled(dev, nr * 512, 0xCD)?;
        let ch_s = filled(dev, nr * 16, 0xCD)?;
        kk.swa_write(&SwaArgs { kv_raw: ptr(&d_sw), norm_w: ptr(&d_kvw), inv_freq: ptr(inv), ring_q: ptr(&swa_rq), ring_s: ptr(&swa_rs), chunk_q: ptr(&ch_q), chunk_s: ptr(&ch_s),
            p0: p0 as i64, rows: nr as i32, norm_fmt: nfmt, eps: 1e-20 })?;
        dev.synchronize()?;
        let (gq, gs) = (down(dev, &ch_q, nr * 512)?, down(dev, &ch_s, nr * 16)?);
        let accn = (acc as usize).min(nr);
        t.add(&format!("{ctag} call p0={p0} swa chunk q"), rows(&gq, 512, 0, accn), rows(&e_swq, 512, p0u, accn.min(n - p0u.min(n))), 512);
        t.add(&format!("{ctag} call p0={p0} swa chunk s"), rows(&gs, 16, 0, accn), rows(&e_sws, 16, p0u, accn.min(n - p0u.min(n))), 16);
    }
    ensure!(dev.dtoh_sync_copy(&err)?[0] == 0, "{name}: kernel error code {}", dev.dtoh_sync_copy(&err)?[0]);
    err = dev.alloc_zeros::<i32>(1)?; let _ = &err;
    // ---- final caches vs the single-shot reference
    t.add(&format!("{ctag} main-KV records [{groups}]"), &down(dev, &main, groups * 288)?, &e_main, 288);
    t.add(&format!("{ctag} index-K q [{groups}]"), &down(dev, &ik_q, groups * 64)?, &e_ikq, 64);
    t.add(&format!("{ctag} index-K s [{groups}]"), &down(dev, &ik_s, groups * 4)?, &e_iks, 4);
    let (rq, rs) = (down(dev, &swa_rq, 134 * 512)?, down(dev, &swa_rs, 134 * 16)?);
    let lo = n.saturating_sub(134);
    let (mut gq, mut gs, mut wq, mut ws) = (vec![], vec![], vec![], vec![]);
    for pos in lo..n {
        let slot = pos % 134;
        gq.extend_from_slice(&rq[slot * 512..(slot + 1) * 512]); gs.extend_from_slice(&rs[slot * 16..(slot + 1) * 16]);
        wq.extend_from_slice(&e_swq[pos * 512..(pos + 1) * 512]); ws.extend_from_slice(&e_sws[pos * 16..(pos + 1) * 16]);
    }
    t.add(&format!("{ctag} swa ring q [{} slots]", n - lo), &gq, &wq, 512);
    t.add(&format!("{ctag} swa ring s [{} slots]", n - lo), &gs, &ws, 16);
    out.push(format!("{ctag}: {} calls replayed, n = {n} tokens, {groups} groups: checked", calls.len()));
    Ok(())
}

fn ptr_i32(s: &CudaSlice<i32>) -> u64 { *s.device_ptr() as u64 }

/// Byte-floor table: per call shape, compulsory bytes / measured time of one launch (queued back to back,
/// so the launch cost is inside the number: these kernels are far below the bandwidth roofline by latency).
fn timing(dev: &Arc<CudaDevice>, kk: &KvKernels) -> Result<Vec<String>> {
    let mut v = vec!["byte floor (GB/s vs the 238 GB/s roofline): compulsory bytes / wall time per launch, 400 back-to-back launches, launch cost INCLUDED".to_string()];
    let inv = dev.htod_sync_copy(&vec![0u8; 128])?;
    let w = dev.htod_sync_copy(&vec![0x3Fu8; 1024])?;
    let ring = dev.htod_sync_copy(&vec![0u8; 7 * 2048])?;
    let big = dev.htod_sync_copy(&vec![0u8; 64 * 2048])?;
    let big16 = dev.htod_sync_copy(&vec![0u8; 64 * 1024])?;
    let main = dev.htod_sync_copy(&vec![0u8; 64 * 288])?;
    let lat = dev.htod_sync_copy(&vec![0u8; 64 * 1024])?;
    let (iq, is_) = (dev.htod_sync_copy(&vec![0u8; 64 * 64])?, dev.htod_sync_copy(&vec![0u8; 64 * 4])?);
    let (rq, rs) = (dev.htod_sync_copy(&vec![0u8; 134 * 512])?, dev.htod_sync_copy(&vec![0u8; 134 * 16])?);
    let (cq, cs) = (dev.htod_sync_copy(&vec![0u8; 64 * 512])?, dev.htod_sync_copy(&vec![0u8; 64 * 16])?);
    let it = 400u32;
    v.push("rows | compress r2 (+stash) | index-K | swa_write | (bytes, GB/s, us per launch)".into());
    for nr in [1usize, 2, 6, 12, 24] {
        let p0 = 1000i64; // even: the first group starts in the call
        let (gf, gc) = group_range(p0 as u64, nr as u64, 2);
        let ca = CompressArgs { kv: ptr(&big), score: ptr(&big), kv16: 0, ring_kv: ptr(&ring), ring_score: ptr(&ring), norm_w: ptr(&w), inv_freq: ptr(&inv), latent_out: ptr(&lat), kv_main: ptr(&main),
            p0, rows: nr as i32, ratio: 2, ring_rows: 7, g_first: 0, n_groups: gc as i32, norm_fmt: NORM_BF16, eps: 1e-20, err: 0 };
        let _ = gf;
        let sa = StashArgs { kv: ptr(&big), score: ptr(&big), ring_kv: ptr(&ring), ring_score: ptr(&ring), p0, rows: nr as i32, ring_rows: 7 };
        let ia = IndexArgs { k_raw: ptr(&lat), norm_w: ptr(&w), inv_freq: ptr(&inv), q_plane: ptr(&iq), s_plane: ptr(&is_), ratio: 2, g_first: 0, n_groups: gc as i32, norm_fmt: NORM_BF16, eps: 1e-20 };
        let wa = SwaArgs { kv_raw: ptr(&big16), norm_w: ptr(&w), inv_freq: ptr(&inv), ring_q: ptr(&rq), ring_s: ptr(&rs), chunk_q: ptr(&cq), chunk_s: ptr(&cs), p0, rows: nr as i32, norm_fmt: NORM_BF16, eps: 1e-20 };
        let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
            for _ in 0..20 { f()?; }
            dev.synchronize()?;
            let t0 = std::time::Instant::now();
            for _ in 0..it { f()?; }
            dev.synchronize()?;
            Ok(t0.elapsed().as_secs_f64() / it as f64)
        };
        let tc = time(&|| { kk.compress(&ca)?; kk.stash(&sa) })?;
        let ti = time(&|| kk.index_k(&ia))?;
        let tw = time(&|| kk.swa_write(&wa))?;
        let g = gc as f64;
        // compulsory bytes: compress reads 2 f32 rows per input row + a ring half-group, writes latent + record + the stash rows
        let bc = (nr * 2 * 2048) as f64 + 2.0 * 2048.0 + g * (1024.0 + 288.0) + nr.min(7) as f64 * 2.0 * 2048.0;
        let bi = g * (256.0 + 68.0);
        let bw = nr as f64 * (1024.0 + 528.0 * 2.0);
        v.push(format!("{nr:>4} | {:>6.0} B {:>5.2} GB/s {:>5.1} us | {:>5.0} B {:>5.2} GB/s {:>5.1} us | {:>6.0} B {:>5.2} GB/s {:>5.1} us",
            bc, bc / tc / 1e9, tc * 1e6, bi, bi / ti / 1e9, ti * 1e6, bw, bw / tw / 1e9, tw * 1e6));
    }
    Ok(v)
}
