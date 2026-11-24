//! DSV4.1 TP (leg PX2): `--probe-dsv41-xchg`, one GPU, seconds. The dsv41 exchange pairs (world 2: PX1, world 4: PX2)
//! against the shared kernels they were copied from (`tp_gate_copy_signal_mb` + `tp_wait_add` mode 2;
//! `tp_gate_copy_signal_os_mb` + `tp_wait_add_4way_f32`), on fake doorbell contexts in mapped host memory (no link, no
//! proxy: the probe plays the proxy and the peers by writing the gates, the tails and the peers' partials itself).
//! For every world, rank and frame width (1 / 6 / 24 rows, and the 6-row Engram-rider frame):
//! - stage: the send slot (payload + tail), the len tag, the epoch, the qp_mask entry, gpu_ready, the arrival counter and
//!   the reuse-gate count are byte-equal to the shared stage's;
//! - collect: the output is bit-equal to the shared collect's AND to the host reference (`p0 + p1` with rank 0's partial
//!   first; `(p0 + p2) + (p1 + p3)`);
//! - chained under PDL (stage then collect on one stream with the attribute, the collect reading the epoch the stage
//!   stored): the same output bits.
//!
//! The device's own `tp_dev_ctx` offsets are read first (`d41x_xc_ctx_layout`), and the fields are written at those
//! offsets, so a drift of `native/tp_doorbell.h` cannot make the probe lie. RESULT: `DSV41_XCHG_OK`.

use crate::dsv41::gpu::engram::MappedHost;
use anyhow::{bail, ensure, Context, Result};
use cudarc::driver::{CudaDevice, CudaFunction, DevicePtr, LaunchAsync, LaunchConfig};
use std::sync::Arc;

const REF_MODULE: &str = "d41px2_ref";
const REF_FNS: [&str; 5] = ["tp_gate_copy_signal_mb", "tp_wait_add", "tp_gate_copy_signal_os_mb", "tp_wait_add_4way_f32", "tp_wait_add_mb"];
/// `native/tp_doorbell.h`
const R: usize = 8;
const F_GPU_READY: usize = 0;
const F_CPU_DONE: usize = 128;
const F_TX_RETIRED: usize = 192;
const F_ABORT: usize = 256;
const NWAY_TX_OFF: usize = 16 * 64;
const NWAY_CPU_OFF: usize = 2 * 16 * 64;
const NWAY_FLAGS_BYTES: usize = 4 * 16 * 64;
const LEN_EPOCHS: usize = 4096;
const QPMASK_SLOTS: usize = 128;
/// one decode row of a layer frame (hidden + mix rider), and the Engram rider a row adds at layers 0 / 13
pub const ROW_FLOATS: usize = 5120 + 32;
pub const ENGRAM_ROW_FLOATS: usize = 25_600;

/// The device's tp_dev_ctx layout: sizeof, then the offsets in `d41x_xc_ctx_layout` order.
#[derive(Clone, Copy, Debug)]
struct Layout {
    size: usize,
    epoch: usize, flags: usize, send: usize, recv: usize, gpu_ts: usize, stride: usize, payload: usize, rank: usize,
    fp32: usize, gate_waits: usize, len_local: usize, world: usize, rounds: usize, nway_flags: usize, nway_recv: usize,
    oneshot_recv: usize, oneshot: usize, qp_mask: usize,
}

/// One fake context and its buffers in one mapped host region.
struct Fake {
    m: MappedHost,
    l: Layout,
    stride: usize,
    o_flags: usize, o_nway: usize, o_len: usize, o_qp: usize, o_send: usize, o_recv: usize, o_os: usize, o_nr: usize,
}

fn align(x: usize, a: usize) -> usize { x.div_ceil(a) * a }

impl Fake {
    fn new(dev: &Arc<CudaDevice>, l: Layout, world: usize, rank: usize, n: usize, epoch: u64) -> Result<Self> {
        let stride = align(n * 4 + 8, 64);
        let o_flags = 1024;
        let o_nway = o_flags + 1024;
        let o_len = o_nway + NWAY_FLAGS_BYTES;
        let o_qp = o_len + LEN_EPOCHS * 8;
        let o_send = align(o_qp + QPMASK_SLOTS * 4, 4096);
        let o_recv = o_send + R * stride;
        let o_os = o_recv + R * stride;
        let o_nr = o_os + 4 * R * stride;
        let len = o_nr + 2 * R * stride;
        ensure!(l.size <= o_flags, "tp_dev_ctx is {} B", l.size);
        let f = Fake { m: MappedHost::new(dev, len)?, l, stride, o_flags, o_nway, o_len, o_qp, o_send, o_recv, o_os, o_nr };
        let d = f.m.dev_ptr();
        f.w64(l.epoch, epoch);
        f.w64(l.flags, d + o_flags as u64);
        f.w64(l.send, d + o_send as u64);
        f.w64(l.recv, d + o_recv as u64);
        f.w64(l.gpu_ts, 0);
        f.w32(l.stride, stride as u32);
        f.w32(l.payload, (n * 4) as u32);
        f.w32(l.rank, rank as u32);
        f.w32(l.fp32, 1);
        f.w64(l.gate_waits, 0);
        f.w64(l.len_local, d + o_len as u64);
        f.w32(l.world, world as u32);
        f.w32(l.rounds, world.trailing_zeros());
        if world > 2 {
            f.w64(l.nway_flags, d + o_nway as u64);
            f.w64(l.oneshot_recv, d + o_os as u64);
            f.w64(l.nway_recv, d + o_nr as u64);
            f.w32(l.oneshot, 1);
            f.w64(l.qp_mask, d + o_qp as u64);
        }
        Ok(f)
    }
    fn ctx(&self) -> u64 { self.m.dev_ptr() }
    fn p(&self, off: usize) -> *mut u8 { unsafe { self.m.host_ptr().add(off) } }
    fn w64(&self, off: usize, v: u64) { unsafe { std::ptr::write_volatile(self.p(off) as *mut u64, v) } }
    fn w32(&self, off: usize, v: u32) { unsafe { std::ptr::write_volatile(self.p(off) as *mut u32, v) } }
    fn r64(&self, off: usize) -> u64 { unsafe { std::ptr::read_volatile(self.p(off) as *const u64) } }
    fn r32(&self, off: usize) -> u32 { unsafe { std::ptr::read_volatile(self.p(off) as *const u32) } }
    fn bytes(&self, off: usize, len: usize) -> Vec<u8> { unsafe { std::slice::from_raw_parts(self.p(off), len).to_vec() } }
    fn put(&self, off: usize, v: &[f32]) { unsafe { std::ptr::copy_nonoverlapping(v.as_ptr() as *const u8, self.p(off), v.len() * 4) } }
    /// a world-4 sender-indexed receive slot
    fn os_slot(&self, sender: usize, e: u64) -> usize { self.o_os + (sender * R + (e as usize % R)) * self.stride }
    fn recv_slot(&self, e: u64) -> usize { self.o_recv + (e as usize % R) * self.stride }
    fn send_slot(&self, e: u64) -> usize { self.o_send + (e as usize % R) * self.stride }
    /// a world-4 tree (round-keyed) receive slot: round(e) = e % 2, slot = (e / 2) % R (Phase 2)
    fn tree_slot(&self, e: u64) -> usize { self.o_nr + ((e as usize % 2) * R + ((e as usize / 2) % R)) * self.stride }
}

/// Keyed partials with full mantissas and exponents spread over 2^-8..2^7 (summation order shows in the bits).
fn partial(seed: u64, n: usize) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n).map(|_| {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        let sign = ((x >> 63) as u32) << 31;
        let exp = (127 - 8 + ((x >> 40) % 16) as u32) << 23;
        f32::from_bits(sign | exp | (x as u32 & 0x7f_ffff))
    }).collect()
}

/// The reference sums: world 2 `lower + upper` (rank 0's partial first), world 4 `(p0 + p2) + (p1 + p3)`.
pub fn reference(parts: &[Vec<f32>]) -> Vec<f32> {
    let n = parts[0].len();
    match parts.len() {
        2 => (0..n).map(|i| parts[0][i] + parts[1][i]).collect(),
        4 => (0..n).map(|i| (parts[0][i] + parts[2][i]) + (parts[1][i] + parts[3][i])).collect(),
        w => panic!("world {w}"),
    }
}

fn layout(dev: &Arc<CudaDevice>) -> Result<Layout> {
    let out = MappedHost::new(dev, 128)?;
    let mut o = out.dev_ptr();
    let mut p = [&mut o as *mut u64 as *mut std::ffi::c_void];
    let s = crate::gpu::fork_blocking_stream(dev);
    crate::dsv41::gpu::launch_raw("d41x_xc_ctx_layout", (1, 1, 1), (32, 1, 1), 0, s.stream as u64, false, &mut p)?;
    unsafe { cudarc::driver::result::stream::synchronize(s.stream) }?;
    let v: Vec<usize> = (0..19).map(|i| unsafe { std::ptr::read_volatile((out.host_ptr() as *const u32).add(i)) } as usize).collect();
    Ok(Layout { size: v[0], epoch: v[1], flags: v[2], send: v[3], recv: v[4], gpu_ts: v[5], stride: v[6], payload: v[7], rank: v[8],
                fp32: v[9], gate_waits: v[10], len_local: v[11], world: v[12], rounds: v[13], nway_flags: v[14], nway_recv: v[15],
                oneshot_recv: v[16], oneshot: v[17], qp_mask: v[18] })
}

struct Kit {
    dev: Arc<CudaDevice>,
    l: Layout,
    r: Vec<CudaFunction>,
    stream: cudarc::driver::CudaStream,
}

impl Kit {
    fn sync(&self) -> Result<()> { unsafe { cudarc::driver::result::stream::synchronize(self.stream.stream) }?; Ok(()) }
    /// Wait for the stream at most 10 s; on a timeout, raise the abort word of every fake (the kernels no-op out, I9) and
    /// fail instead of hanging.
    fn sync_or_abort(&self, fakes: &[&Fake], what: &str) -> Result<()> {
        use cudarc::driver::sys;
        let t0 = std::time::Instant::now();
        loop {
            let r = unsafe { sys::cuStreamQuery(self.stream.stream) };
            if r == sys::CUresult::CUDA_SUCCESS { return Ok(()); }
            ensure!(r == sys::CUresult::CUDA_ERROR_NOT_READY, "{what}: {r:?}");
            if t0.elapsed() > std::time::Duration::from_secs(10) {
                for f in fakes { f.w64(f.o_flags + F_ABORT, 99); }
                self.sync()?;
                bail!("{what}: the kernels did not finish in 10 s (a gate never opened); aborted");
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
    fn dbuf(&self, v: &[f32]) -> Result<cudarc::driver::CudaSlice<f32>> { Ok(self.dev.htod_sync_copy(v)?) }
    fn arrive(&self) -> Result<cudarc::driver::CudaSlice<u32>> { let mut a = self.dev.alloc_zeros::<u32>(1)?; self.dev.memset_zeros(&mut a)?; Ok(a) }

    fn stage_ref(&self, world: usize, f: &Fake, src: u64, n: usize, arr: u64) -> Result<()> {
        let cfg = LaunchConfig { grid_dim: (super::rdma::k1_mb_blocks(n), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            if world == 2 { self.r[0].clone().launch_on_stream(&self.stream, cfg, (f.ctx(), src, (n * 4) as u32, arr, 0i32)) }
            else { self.r[2].clone().launch_on_stream(&self.stream, cfg, (f.ctx(), src, (n * 4) as u32, arr)) }
        }?;
        Ok(())
    }
    fn collect_ref(&self, world: usize, f: &Fake, buf: u64, n: usize) -> Result<()> {
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            if world == 2 { self.r[1].clone().launch_on_stream(&self.stream, cfg, (f.ctx(), buf, buf, n as i32, 2i32)) }
            else { self.r[3].clone().launch_on_stream(&self.stream, cfg, (f.ctx(), buf, buf, n as i32)) }
        }?;
        Ok(())
    }
    fn stage_new(&self, world: usize, f: &Fake, src: u64, n: usize, arr: u64, pdl: bool) -> Result<()> {
        let (mut c, mut s, mut nb, mut a, mut tr) = (f.ctx(), src, (n * 4) as u32, arr, 0u64);
        let mut p = [&mut c as *mut u64 as *mut std::ffi::c_void, &mut s as *mut u64 as *mut _, &mut nb as *mut u32 as *mut _,
                     &mut a as *mut u64 as *mut _, &mut tr as *mut u64 as *mut _];
        let name = if world == 2 { super::xchg::STAGE } else { super::xchg::STAGE4 };
        crate::dsv41::gpu::launch_raw(name, (super::xchg::stage_blocks(n), 1, 1), (256, 1, 1), 0, self.stream.stream as u64, pdl, &mut p)
    }
    fn collect_new(&self, world: usize, f: &Fake, buf: u64, n: usize, pdl: bool, rel: u64) -> Result<()> {
        let (mut c, mut b, mut nn, mut tr, mut relp) = (f.ctx(), buf, n as i32, 0u64, rel);
        let mut p = [&mut c as *mut u64 as *mut std::ffi::c_void, &mut b as *mut u64 as *mut _, &mut nn as *mut i32 as *mut _,
                     &mut tr as *mut u64 as *mut _, &mut relp as *mut u64 as *mut _];
        let (name, blocks) = if world == 2 { (super::xchg::COLLECT, super::xchg::collect_blocks(n)) } else { (super::xchg::COLLECT4, super::xchg::collect4_blocks(n)) };
        crate::dsv41::gpu::launch_raw(name, (blocks, 1, 1), (256, 1, 1), 0, self.stream.stream as u64, pdl, &mut p)
    }

    fn rel_buf(&self) -> Result<cudarc::driver::CudaSlice<u64>> { let mut r = self.dev.alloc_zeros::<u64>(2)?; self.dev.memset_zeros(&mut r)?; Ok(r) }

    fn fake(&self, world: usize, rank: usize, n: usize, epoch: u64) -> Result<Fake> {
        let f = Fake::new(&self.dev, self.l, world, rank, n, epoch)?;
        // the reuse gate at e = epoch + 1 > R: everything up to e - R retired, on every QP the slot went out on
        let e = epoch + 1;
        if world == 2 { f.w64(f.o_flags + F_TX_RETIRED, e); } else {
            for p in 0..4 { if p != rank { f.w64(f.o_nway + p * 64 + NWAY_TX_OFF, e); } }
            let mask = 0xFu32 & !(1u32 << rank);
            for k in 0..QPMASK_SLOTS { f.w32(f.o_qp + 4 * k, mask); }
        }
        Ok(f)
    }
    /// The probe as the proxy / the peers: epoch `e`'s peer partials placed and released.
    fn deliver(&self, world: usize, rank: usize, f: &Fake, e: u64, parts: &[Vec<f32>]) {
        let wire = align(parts[0].len() * 4, 8);
        if world == 2 {
            f.put(f.recv_slot(e), &parts[1 - rank]);
            f.w64(f.recv_slot(e) + wire, e);
            f.w64(f.o_flags + F_CPU_DONE, e);
        } else {
            for p in 0..4 {
                if p == rank { continue; }
                f.put(f.os_slot(p, e), &parts[p]);
                f.w64(f.os_slot(p, e) + wire, e);
                f.w64(f.o_nway + p * 64 + NWAY_CPU_OFF, e);
            }
            f.w64(f.o_flags + F_CPU_DONE, e);
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    }
}

/// leg PX2's reduce-scatter pieces at world 4: the tree stage against the shared `tp_gate_copy_signal_mb` (world>2 arm),
/// the tree collect against the shared `tp_wait_add_mb` (world>2 arm, lag 0, fp32) and the host sum; the all-gather
/// collect's placement (a plain copy, -0.0 kept). Then the whole reduce-scatter + all-gather of a 24-row frame played for
/// all four ranks on four fakes (the probe moves each epoch's payload between them) against `(p0 + p2) + (p1 + p3)`.
fn tree_and_gather(kit: &Kit, lines: &mut Vec<String>) -> Result<usize> {
    let mut fails = 0;
    let n = 12 * ROW_FLOATS;
    for (pre, rank) in [(12u64, 0usize), (12, 3), (13, 1), (13, 2)] {
        let e = pre + 1;
        let partner = rank ^ (1 << (e as usize % 2));
        let parts: Vec<Vec<f32>> = (0..4).map(|q| partial(77 + q as u64, n)).collect();
        let (fa, fb) = (kit.fake(4, rank, n, pre)?, kit.fake(4, rank, n, pre)?);
        for f in [&fa, &fb] {   // the tree gate: partner(e - R)'s QP retired
            let tgt_partner = rank ^ (1 << ((e as usize - R) % 2));
            f.w64(f.o_nway + tgt_partner * 64 + NWAY_TX_OFF, e);
        }
        let src = kit.dbuf(&parts[rank])?;
        let (aa, ab) = (kit.arrive()?, kit.arrive()?);
        kit.stage_ref(2, &fa, *src.device_ptr(), n, *aa.device_ptr())?;   // tp_gate_copy_signal_mb takes its world>2 arm from the ctx
        let (mut c, mut s, mut nb, mut a, mut tr) = (fb.ctx(), *src.device_ptr(), (n * 4) as u32, *ab.device_ptr(), 0u64);
        let mut p = [&mut c as *mut u64 as *mut std::ffi::c_void, &mut s as *mut u64 as *mut _, &mut nb as *mut u32 as *mut _,
                     &mut a as *mut u64 as *mut _, &mut tr as *mut u64 as *mut _];
        crate::dsv41::gpu::launch_raw(super::xchg::TSTAGE, (super::xchg::stage_blocks(n), 1, 1), (256, 1, 1), 0, kit.stream.stream as u64, false, &mut p)?;
        kit.sync()?;
        let wire = align(n * 4, 8);
        let st_ok = fa.bytes(fa.send_slot(e), wire + 8) == fb.bytes(fb.send_slot(e), wire + 8)
            && fa.r64(fa.l.epoch) == e && fb.r64(fb.l.epoch) == e
            && fa.r32(fa.o_qp + 4 * (e as usize % QPMASK_SLOTS)) == fb.r32(fb.o_qp + 4 * (e as usize % QPMASK_SLOTS))
            && fa.r32(fb.o_qp + 4 * (e as usize % QPMASK_SLOTS)) == 1u32 << partner
            && fa.r64(fa.o_len + 8 * (e as usize % LEN_EPOCHS)) == fb.r64(fb.o_len + 8 * (e as usize % LEN_EPOCHS));
        // the tree collect: partner's partial in the round slot, partner's cpu_done
        for f in [&fa, &fb] {
            f.put(f.tree_slot(e), &parts[partner]);
            f.w64(f.tree_slot(e) + wire, e);
            f.w64(f.o_nway + partner * 64 + NWAY_CPU_OFF, e);
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        let (ba, bb) = (kit.dbuf(&parts[rank])?, kit.dbuf(&parts[rank])?);
        let rel_tree = kit.rel_buf()?;
        let cfg = LaunchConfig { grid_dim: (8, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe { kit.r[4].clone().launch_on_stream(&kit.stream, cfg, (fa.ctx(), *ba.device_ptr(), *ba.device_ptr(), n as i32, 0u32, 0i32)) }?;
        let (mut c, mut d, mut nn, mut tr, mut relp) = (fb.ctx(), *bb.device_ptr(), n as i32, 0u64, *rel_tree.device_ptr());
        let mut q = [&mut c as *mut u64 as *mut std::ffi::c_void, &mut d as *mut u64 as *mut _, &mut nn as *mut i32 as *mut _, &mut tr as *mut u64 as *mut _, &mut relp as *mut u64 as *mut _];
        crate::dsv41::gpu::launch_raw(super::xchg::TCOLLECT, (super::xchg::collect_blocks(n), 1, 1), (256, 1, 1), 0, kit.stream.stream as u64, false, &mut q)?;
        kit.sync_or_abort(&[&fa, &fb], "tree collect")?;
        let (oa, ob) = (kit.dev.dtoh_sync_copy(&ba)?, kit.dev.dtoh_sync_copy(&bb)?);
        let lower = rank & (1 << (e as usize % 2)) == 0;
        let want: Vec<f32> = (0..n).map(|i| if lower { parts[rank][i] + parts[partner][i] } else { parts[partner][i] + parts[rank][i] }).collect();
        let (d1, d2) = (bits_equal(&oa, &want), bits_equal(&ob, &want));
        let ok = st_ok && d1 == 0 && d2 == 0;
        if !ok { fails += 1; }
        crate::reprintln!("[xprobe] tree epoch {e} rank {rank} done");
        lines.push(format!("world 4 tree epoch {e} rank {rank} (partner {partner}, {n} floats): stage {} | collect: shared vs host {d1}, dsv41 vs host {d2} differing floats {}",
                           if st_ok { "byte-equal to the shared tree stage" } else { "DIFFERS" }, if ok { "BITWISE EQUAL" } else { "FAIL" }));
    }
    // the whole reduce-scatter + all-gather, four fakes, the probe as the wire
    let n = 24 * ROW_FLOATS;
    let mut parts: Vec<Vec<f32>> = (0..4).map(|q| partial(500 + q as u64, n)).collect();
    parts[2][5] = -0.0; parts[0][5] = -0.0; parts[1][5] = -0.0; parts[3][5] = -0.0;   // -0.0 + -0.0 sums stay -0.0
    let want = reference(&parts);
    let pre = 12u64;   // the primary tree starts on an odd epoch (R10): 13 pairs r ^ 2, 14 pairs r ^ 1
    let pf: Vec<Fake> = (0..4).map(|r| kit.fake(4, r, n, pre)).collect::<Result<_>>()?;
    let sf: Vec<Fake> = (0..4).map(|r| kit.fake(4, r, n, pre)).collect::<Result<_>>()?;
    let bufs: Vec<cudarc::driver::CudaSlice<f32>> = parts.iter().map(|p| kit.dbuf(p)).collect::<Result<_>>()?;
    let xs: Vec<super::xchg::Xchg> = (0..4).map(|r| super::xchg::Xchg::new(&kit.dev, 4, r, true, false)).collect::<Result<_>>()?;
    // epoch by epoch: every rank's stage, then the probe delivers, then every rank's collect (the gates are already open,
    // so the six launches are run as three stage / deliver / collect steps)
    let plans: Vec<super::xchg::RsAg> = (0..4).map(|r| super::xchg::rsag_plan(n, r).unwrap()).collect();
    for step in 0..3 {
        crate::reprintln!("[xprobe] reduce-scatter + all-gather step {step}");
        // primary epochs 13 (pairs r ^ 2) and 14 (r ^ 1); the single-stage ctx's first epoch is 13
        let e = if step < 2 { pre + 1 + step as u64 } else { pre + 1 };
        for r in 0..4 {
            let f = if step < 2 { &pf[r] } else { &sf[r] };
            if step == 2 { f.w64(f.l.epoch, pre); }
            let g = plans[r];
            let (off, len) = match step { 0 => (g.a_send, g.half), 1 => (g.b_send, g.quarter), _ => (g.b_keep, g.quarter) };
            let src = *bufs[r].device_ptr() + (off * 4) as u64;
            let (mut c, mut s, mut nb, mut a, mut tr) = (f.ctx(), src, (len * 4) as u32, xs[r].arrive_ptr(), 0u64);
            let mut p = [&mut c as *mut u64 as *mut std::ffi::c_void, &mut s as *mut u64 as *mut _, &mut nb as *mut u32 as *mut _,
                         &mut a as *mut u64 as *mut _, &mut tr as *mut u64 as *mut _];
            let name = if step < 2 { super::xchg::TSTAGE } else { super::xchg::STAGE4 };
            // the tree gate at e > R: the QP of partner(e - R) retired (only step 1 has e - R odd/even differently)
            for k in 0..4 { f.w64(f.o_nway + k * 64 + NWAY_TX_OFF, e); }
            crate::dsv41::gpu::launch_raw(name, (super::xchg::stage_blocks(len), 1, 1), (256, 1, 1), 0, kit.stream.stream as u64, true, &mut p)?;
        }
        kit.sync()?;
        let wire_of = |len: usize| align(len * 4, 8);
        for r in 0..4 {
            let g = plans[r];
            if step < 2 {
                let peer = r ^ if step == 0 { 2 } else { 1 };
                let (pfk, len) = (&pf[peer], if step == 0 { g.half } else { g.quarter });
                let payload = pfk.bytes(pfk.send_slot(e), len * 4);
                let f = &pf[r];
                unsafe { std::ptr::copy_nonoverlapping(payload.as_ptr(), f.p(f.tree_slot(e)), payload.len()) };
                f.w64(f.tree_slot(e) + wire_of(len), e);
                f.w64(f.o_nway + peer * 64 + NWAY_CPU_OFF, e);
            } else {
                for p in 0..4 {
                    if p == r { continue; }
                    let payload = sf[p].bytes(sf[p].send_slot(e), g.quarter * 4);
                    let f = &sf[r];
                    unsafe { std::ptr::copy_nonoverlapping(payload.as_ptr(), f.p(f.os_slot(p, e)), payload.len()) };
                    f.w64(f.os_slot(p, e) + wire_of(g.quarter), e);
                    f.w64(f.o_nway + p * 64 + NWAY_CPU_OFF, e);
                }
                sf[r].w64(sf[r].o_flags + F_CPU_DONE, e);
            }
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        for r in 0..4 {
            let g = plans[r];
            let (f, name, dst, len, blocks) = match step {
                0 => (&pf[r], super::xchg::TCOLLECT, g.a_keep, g.half, super::xchg::collect_blocks(g.half)),
                1 => (&pf[r], super::xchg::TCOLLECT, g.b_keep, g.quarter, super::xchg::collect_blocks(g.quarter)),
                _ => (&sf[r], super::xchg::GATHER4, 0, g.quarter, super::xchg::collect_blocks(3 * g.quarter)),
            };
            let (mut c, mut d, mut nn, mut tr, mut relp) = (f.ctx(), *bufs[r].device_ptr() + (dst * 4) as u64, len as i32, 0u64,
                                                            if step < 2 { xs[r].rel_ptr() + 16 } else { xs[r].rel_ptr() });
            let mut q = [&mut c as *mut u64 as *mut std::ffi::c_void, &mut d as *mut u64 as *mut _, &mut nn as *mut i32 as *mut _, &mut tr as *mut u64 as *mut _, &mut relp as *mut u64 as *mut _];
            crate::dsv41::gpu::launch_raw(name, (blocks, 1, 1), (256, 1, 1), 0, kit.stream.stream as u64, true, &mut q)?;
        }
        let all: Vec<&Fake> = pf.iter().chain(sf.iter()).collect();
        kit.sync_or_abort(&all, "reduce-scatter + all-gather")?;
    }
    let mut worst = 0;
    for r in 0..4 { worst = worst.max(bits_equal(&kit.dev.dtoh_sync_copy(&bufs[r])?, &want)); }
    let neg0 = kit.dev.dtoh_sync_copy(&bufs[1])?[5].to_bits() == (-0.0f32).to_bits();
    let ok = worst == 0 && neg0;
    if !ok { fails += 1; }
    let (one, rs) = super::xchg::wire_bytes_w4(n);
    lines.push(format!("world 4 reduce-scatter + all-gather, 24 rows ({n} floats), all four ranks played: worst rank {worst} floats differ from (p0 + p2) + (p1 + p3); -0.0 kept {neg0}; bytes a rank sends {rs} (one all-peers stage {one}) {}",
                       if ok { "BITWISE EQUAL" } else { "FAIL" }));
    Ok(fails)
}

fn bits_equal(a: &[f32], b: &[f32]) -> usize { a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count() }

/// `--probe-dsv41-xchg`: the lines of the report (the last is `DSV41_XCHG_OK` when every check holds).
pub fn run() -> Result<Vec<String>> {
    let dev = CudaDevice::new(0).context("CUDA device 0")?;
    dev.bind_to_thread()?;
    let l = layout(&dev)?;
    let mut lines = vec![format!("tp_dev_ctx on the device: {} B; epoch @{}, flags @{}, send @{}, recv @{}, world @{}, nway_flags @{}, oneshot_recv @{}, qp_mask @{}",
                                 l.size, l.epoch, l.flags, l.send, l.recv, l.world, l.nway_flags, l.oneshot_recv, l.qp_mask)];
    if dev.get_func(REF_MODULE, REF_FNS[0]).is_none() {
        dev.load_ptx(cudarc::nvrtc::Ptx::from_src(crate::ptx_embed::GPU_BATCH), REF_MODULE, &REF_FNS).context("load the shared pair (gpu_batch.ptx)")?;
    }
    let r = REF_FNS.iter().map(|n| dev.get_func(REF_MODULE, n).with_context(|| format!("{n} missing"))).collect::<Result<Vec<_>>>()?;
    let kit = Kit { stream: crate::gpu::fork_blocking_stream(&dev), dev: dev.clone(), l, r };
    let mut fails = 0usize;
    let widths: [(&str, usize); 4] = [("1 row", ROW_FLOATS), ("6 rows", 6 * ROW_FLOATS), ("24 rows", 24 * ROW_FLOATS),
                                      ("6 rows + Engram rider", 6 * (ROW_FLOATS + ENGRAM_ROW_FLOATS))];
    let pre = 12u64;
    let e = pre + 1;
    for world in [2usize, 4] {
        for (wname, n) in widths {
            for rank in 0..world {
                let parts: Vec<Vec<f32>> = (0..world).map(|q| partial(1000 * world as u64 + 7 * n as u64 + q as u64, n)).collect();
                let want = reference(&parts);
                // stage
                let (fa, fb) = (kit.fake(world, rank, n, pre)?, kit.fake(world, rank, n, pre)?);
                let src = kit.dbuf(&parts[rank])?;
                let (aa, ab) = (kit.arrive()?, kit.arrive()?);
                kit.stage_ref(world, &fa, *src.device_ptr(), n, *aa.device_ptr())?;
                kit.stage_new(world, &fb, *src.device_ptr(), n, *ab.device_ptr(), false)?;
                kit.sync()?;
                let wire = align(n * 4, 8);
                let mut diffs = Vec::new();
                if fa.bytes(fa.send_slot(e), wire + 8) != fb.bytes(fb.send_slot(e), wire + 8) { diffs.push("send slot"); }
                if fa.r64(fa.l.epoch) != e || fb.r64(fb.l.epoch) != e { diffs.push("epoch"); }
                if fa.r64(fa.o_len + 8 * (e as usize % LEN_EPOCHS)) != fb.r64(fb.o_len + 8 * (e as usize % LEN_EPOCHS)) { diffs.push("len tag"); }
                if fa.r64(fa.o_flags + F_GPU_READY) != e || fb.r64(fb.o_flags + F_GPU_READY) != e { diffs.push("gpu_ready"); }
                if fa.r32(fa.o_qp + 4 * (e as usize % QPMASK_SLOTS)) != fb.r32(fb.o_qp + 4 * (e as usize % QPMASK_SLOTS)) { diffs.push("qp_mask"); }
                if fa.r64(fa.l.gate_waits) != fb.r64(fb.l.gate_waits) { diffs.push("gate waits"); }
                if kit.dev.dtoh_sync_copy(&aa)?[0] != 0 || kit.dev.dtoh_sync_copy(&ab)?[0] != 0 { diffs.push("arrival counter"); }
                let sent: Vec<f32> = fb.bytes(fb.send_slot(e), n * 4).chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                if bits_equal(&sent, &parts[rank]) != 0 { diffs.push("payload != the partial"); }
                // collect (epoch already e on both fakes)
                kit.deliver(world, rank, &fa, e, &parts);
                kit.deliver(world, rank, &fb, e, &parts);
                let (ba, bb) = (kit.dbuf(&parts[rank])?, kit.dbuf(&parts[rank])?);
                let rel_b = kit.rel_buf()?;
                kit.collect_ref(world, &fa, *ba.device_ptr(), n)?;
                kit.collect_new(world, &fb, *bb.device_ptr(), n, false, *rel_b.device_ptr())?;
                kit.sync_or_abort(&[&fa, &fb], "collect")?;
                let (oa, ob) = (kit.dev.dtoh_sync_copy(&ba)?, kit.dev.dtoh_sync_copy(&bb)?);
                let (d_ref, d_new, d_pair) = (bits_equal(&oa, &want), bits_equal(&ob, &want), bits_equal(&oa, &ob));
                // chained under PDL: a fresh fake, stage then collect with the attribute, the collect reading the stored epoch
                let fc = kit.fake(world, rank, n, pre)?;
                kit.deliver(world, rank, &fc, e, &parts);
                let ac = kit.arrive()?;
                let rel_c = kit.rel_buf()?;
                let bc = kit.dbuf(&parts[rank])?;
                kit.stage_new(world, &fc, *bc.device_ptr(), n, *ac.device_ptr(), true)?;
                kit.collect_new(world, &fc, *bc.device_ptr(), n, true, *rel_c.device_ptr())?;
                kit.sync_or_abort(&[&fc], "PDL chain")?;
                let d_pdl = bits_equal(&kit.dev.dtoh_sync_copy(&bc)?, &want);
                let ok = diffs.is_empty() && d_ref == 0 && d_new == 0 && d_pair == 0 && d_pdl == 0;
                if !ok { fails += 1; }
                crate::reprintln!("[xprobe] world {world} rank {rank} {wname} done");
                lines.push(format!("world {world} rank {rank} {wname:22} ({n:6} floats): stage {} | collect: shared vs reference {d_ref}, dsv41 vs reference {d_new}, dsv41 vs shared {d_pair} differing floats; PDL chain {d_pdl} {}",
                                   if diffs.is_empty() { "byte-equal (slot, tail, len tag, epoch, qp_mask, gpu_ready, counters)".to_string() } else { format!("DIFFERS: {}", diffs.join(", ")) },
                                   if ok { "BITWISE EQUAL" } else { "FAIL" }));
            }
        }
    }
    fails += tree_and_gather(&kit, &mut lines)?;
    if fails > 0 { bail!("{fails} case(s) differ:\n{}", lines.join("\n")); }
    lines.push("DSV41_XCHG_OK".into());
    Ok(lines)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_sums_keep_the_transport_association() {
        // values where the association changes the bits: (a + c) + (b + d) != ((a + b) + c) + d
        let p = vec![vec![1.0e8f32], vec![1.0f32], vec![-1.0e8f32], vec![1.0f32]];
        assert_eq!(reference(&p)[0], 2.0);            // (1e8 - 1e8) + (1 + 1)
        assert_ne!(((p[0][0] + p[1][0]) + p[2][0]) + p[3][0], 2.0);
        let q = vec![vec![1.5f32], vec![2.25f32]];
        assert_eq!(reference(&q)[0], 3.75);
    }

    #[test]
    fn partials_spread_exponents_and_differ_by_seed() {
        let a = partial(1, 1000);
        let b = partial(2, 1000);
        assert!(a.iter().all(|x| x.is_finite() && x.abs() >= 2f32.powi(-8) && x.abs() < 2f32.powi(8)));
        assert!(bits_equal(&a, &b) > 990);
    }
}
