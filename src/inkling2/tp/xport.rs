//! The Inkling-2 TP transport (Phase B): full `velo-fabric` multi-rail RDMA collective backend
//! (`crate::dsv41::tp::rdma::RdmaXport`) across TP2, TP4, and TP8, plus deterministic loopback
//! simulation (`--probe-inkling2-tp-sim`).
//!
//! * **TP2 / TP4 / TP8 decode & verify (`1..=48` rows = `9 KiB..432 KiB <= 768 KiB`)**:
//!   - Zero-CPU-bounce GPU-validated PDL stage/collect kernels (single-CTA C2C poller w/ GPU-L2
//!     handoff + direct NIC tail validation, default ON; `--tp-gpu-recv off` fallback);
//!   - Dual-rail RoCEv2 (400 Gb/s) auto-negotiated across every rank (`>= 16 KiB` split across both
//!     rails; single-rail graceful fallback);
//!   - World-4 (`>= 384 KiB`) and World-8 (`>= 128 KiB`) dual-rail PDL RS+AG (`0.75F`/rail at W=4,
//!     `0.875F`/rail at W=8 — 4x less wire than 1-stage all-peers);
//!   - Every decode / verify frame (`1..=48` rows) launches directly onto `Ink2Kernels::stream` so
//!     it is 100% captured into Inkling-2's decode/verify CUDA graphs.
//! * **Wide prefill (`m = 512..2048` rows = `4.5..18.9 MiB`)**:
//!   - 2-rail pipelined transport (`Pipe`): 2-rail `reduce_pipe` at W=2 (`0.50F`/rail), and 2-rail
//!     pipelined `sp_reduce_scatter` + `sp_all_gather_multi` at W=4 (`0.75F`/rail, 2.67x less
//!     wire/rail than legacy 1-rail Pipe) and W=8 (`0.875F`/rail, 3.43x less wire/rail than legacy
//!     1-rail Pipe) — 100% bit-exact canonical tree sums.
//! * **Logits gather (`logits_full` / `logits_host`)**:
//!   - True 2-rail byte-exact `all_gather_rows` transferring only `(W-1)/W` of the vocabulary
//!     (4.0x / 5.3x / 6.85x less wire/rail at TP2 / TP4 / TP8 vs legacy 1-rail all-reduce).
//! * **`Loop`**: W ranks as W threads on ONE device with canonical tree sums (W=2/4/8) and
//!   matching `all_gather_rows` for `--probe-inkling2-tp-sim`.

use anyhow::{anyhow, ensure, Context, Result};
use cudarc::driver::{result, CudaDevice, DevicePtr};
use std::sync::Arc;

/// Largest decode/verify frame handled in one PDL exchange (768 KiB: holds all 1..=48 verify rows).
pub const SINGLE_CAP_BYTES: usize = crate::dsv41::tp::rdma::SINGLE_CAP_BYTES_XCHG4;
/// Default wide-pipe threshold (matched to `RdmaXport`).
pub const PIPE_MIN_BYTES: usize = crate::dsv41::tp::rdma::PIPE_MIN_BYTES_XCHG4;

enum Kind {
    Solo,
    Rdma {
        rdma: crate::dsv41::tp::rdma::RdmaXport,
        /// Blocking stream (CU_STREAM_DEFAULT): implicitly ordered against Ink2Kernels' legacy NULL stream.
        stream: cudarc::driver::CudaStream,
    },
    Loop {
        txs: Vec<std::sync::mpsc::Sender<Vec<f32>>>,
        rx: std::sync::Mutex<std::sync::mpsc::Receiver<Vec<f32>>>,
        a2a_txs: Vec<std::sync::mpsc::Sender<Vec<u32>>>,
        a2a_rx: std::sync::Mutex<std::sync::mpsc::Receiver<Vec<u32>>>,
        barrier: Arc<std::sync::Barrier>,
    },
}

pub struct Xport {
    pub rank: usize,
    pub world: usize,
    kind: Kind,
    pub describe: String,
}
// The proxy owns the link; device pointers / streams are driven by the single model worker thread.
unsafe impl Send for Xport {}

static REDUCES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// (reduces built, bytes) on this rank (graph replays re-run captured reduces without counting).
pub fn counts() -> (u64, u64) {
    (REDUCES.load(std::sync::atomic::Ordering::Relaxed), BYTES.load(std::sync::atomic::Ordering::Relaxed))
}

impl Xport {
    pub fn solo() -> Xport {
        Xport { rank: 0, world: 1, kind: Kind::Solo, describe: "TP1 (no transport)".into() }
    }

    /// `world` loopback transports for `world` ranks on ONE device (the exchange simulation).
    pub fn loopback_group(world: usize) -> Vec<Xport> {
        let mut txs = Vec::new();
        let mut rxs = Vec::new();
        let mut a2a_txs = Vec::new();
        let mut a2a_rxs = Vec::new();
        for _ in 0..world {
            let (t, r) = std::sync::mpsc::channel::<Vec<f32>>();
            txs.push(t);
            rxs.push(r);
            let (at, ar) = std::sync::mpsc::channel::<Vec<u32>>();
            a2a_txs.push(at);
            a2a_rxs.push(ar);
        }
        let barrier = Arc::new(std::sync::Barrier::new(world));
        rxs.into_iter().zip(a2a_rxs).enumerate().map(|(rank, (rx, a2a_rx))| Xport {
            rank,
            world,
            kind: Kind::Loop {
                txs: txs.clone(),
                rx: std::sync::Mutex::new(rx),
                a2a_txs: a2a_txs.clone(),
                a2a_rx: std::sync::Mutex::new(a2a_rx),
                barrier: barrier.clone(),
            },
            describe: format!("loopback simulation, rank {rank}/{world} (host channels, canonical sum + Ulysses A2A)"),
        }).collect()
    }

    /// Attach a brought-up link on the full `velo-fabric` multi-rail PDL / GPU-recv / RS+AG transport.
    pub fn attach(dev: &Arc<CudaDevice>, ctx: crate::tp::TpContext) -> Result<Xport> {
        let rdma = crate::dsv41::tp::rdma::RdmaXport::attach_with_branch(
            dev, ctx, 2304 * 4, &crate::tp::TpBranch::Inkling2,
        )?;
        let stream = crate::gpu::fork_blocking_stream(dev);
        let (rank, world) = (rdma.rank, rdma.world);
        let describe = format!("velo-fabric {}", rdma.describe);
        let mut x = Xport { rank, world, kind: Kind::Rdma { rdma, stream }, describe };
        // Bit-exact boot self-test across every transport tier (also loads GPU_DSV41 PTX before graph capture):
        //   world*2 (1-row vocab-parallel sampler record — tests non-16B-aligned 6 floats at TP3 too)
        //   2,304 (1 row, 9 KiB: rail-1 PDL) | 8*2,304 (72 KiB: dual-rail PDL)
        //   48*2,304 (432 KiB: dual-rail RS+AG) | 512*2,304 (4.5 MiB: 2-rail Pipe / wide RS+AG)
        for &n in &[world * 2, 2304usize, 8 * 2304, 48 * 2304, 512 * 2304] {
            let bad = x.self_test(dev, n)?;
            ensure!(bad == 0, "inkling2 TP: transport self-test ({n} floats): {bad} wrong floats (rank {rank}/{world})");
        }
        let ag_bad = x.self_test_all_gather(dev)?;
        ensure!(ag_bad == 0, "inkling2 TP: all_gather self-test: {ag_bad} wrong floats (rank {rank}/{world})");
        if x.sp_ready() {
            let sp_bad = x.self_test_sp(dev)?;
            ensure!(sp_bad == 0, "inkling2 TP: SP reduce-scatter + multi all-gather self-test: {sp_bad} wrong values (rank {rank}/{world})");
        }
        if x.a2a_ready() {
            let a2a_bad = x.self_test_all_to_all(dev)?;
            ensure!(a2a_bad == 0, "inkling2 TP: DeepSpeed-Ulysses all_to_all self-test: {a2a_bad} wrong values (rank {rank}/{world})");
        }
        // Exchange ledger at attach: 1-row plain decode, 4-row, and 8-row MTP verify (200 reps each)
        let us1 = x.time_frames(dev, 2304, 200)?;
        let us4 = x.time_frames(dev, 4 * 2304, 200)?;
        let us8 = x.time_frames(dev, 8 * 2304, 200)?;
        let sp_tag = if x.sp_ready() {
            if matches!(world, 3 | 4 | 6 | 8) && x.a2a_ready() {
                "; SP prefill: ON (1/W AttnRes+MsgGate+Norm+sconv+Router, DeepSpeed-Ulysses All-to-All Attn [arXiv:2309.14509], BF16/FP16 MoE AG)"
            } else {
                "; SP prefill: ON (1/W AttnRes+MsgGate+Norm+sconv+Router, BF16/FP16 AG)"
            }
        } else { "" };
        x.describe = format!(
            "{}{sp_tag}; measured at attach: {us1:.1} us/1-row, {us4:.1} us/4-row, {us8:.1} us/8-row (x 96 = {:.2} / {:.2} / {:.2} ms)",
            x.describe, us1 * 96.0 / 1e3, us4 * 96.0 / 1e3, us8 * 96.0 / 1e3
        );
        Ok(x)
    }

    /// Rank r fills n floats with `(r + 1) * (i % 7)`, reduces, checks the exact sum. Returns wrong count.
    pub fn self_test(&self, dev: &Arc<CudaDevice>, n: usize) -> Result<usize> {
        let w = self.world;
        let mine: Vec<f32> = (0..n).map(|i| (self.rank as f32 + 1.0) * (i % 7) as f32).collect();
        let want: Vec<f32> = (0..n).map(|i| (1..=w).map(|r| r as f32).sum::<f32>() * (i % 7) as f32).collect();
        let buf = dev.htod_sync_copy(&mine)?;
        self.all_reduce(dev, *buf.device_ptr(), n)?;
        dev.synchronize()?;
        let got = dev.dtoh_sync_copy(&buf)?;
        Ok(got.iter().zip(&want).filter(|(a, b)| a.to_bits() != b.to_bits()).count())
    }

    /// Rank r populates its own 2-row block of 128 floats, runs `all_gather_rows`, and checks every rank's block bit-exact.
    pub fn self_test_all_gather(&self, dev: &Arc<CudaDevice>) -> Result<usize> {
        let w = self.world;
        if w <= 1 { return Ok(0); }
        let rows_per_rank = 2usize;
        let row_floats = 128usize;
        let total = w * rows_per_rank * row_floats;
        let mut host = vec![0f32; total];
        let val = |r: usize, rr: usize, j: usize| ((r + 1) * 1000 + rr * 100 + (j % 37)) as f32;
        for rr in 0..rows_per_rank {
            for j in 0..row_floats {
                host[(self.rank * rows_per_rank + rr) * row_floats + j] = val(self.rank, rr, j);
            }
        }
        let buf = dev.htod_sync_copy(&host)?;
        let blocks: Vec<(usize, usize)> = (0..w).map(|r| (r * rows_per_rank, (r + 1) * rows_per_rank)).collect();
        self.all_gather_rows(dev, *buf.device_ptr(), row_floats * 4, &blocks)?;
        dev.synchronize()?;
        let got = dev.dtoh_sync_copy(&buf)?;
        let mut bad = 0usize;
        for r in 0..w {
            for rr in 0..rows_per_rank {
                for j in 0..row_floats {
                    let g = got[(r * rows_per_rank + rr) * row_floats + j];
                    if g.to_bits() != val(r, rr, j).to_bits() { bad += 1; }
                }
            }
        }
        Ok(bad)
    }

    /// Boot self-test for Drop 2 SP prefill collectives: `sp_reduce_scatter` + 2-buffer `all_gather_rows_multi`.
    pub fn self_test_sp(&self, dev: &Arc<CudaDevice>) -> Result<usize> {
        let w = self.world;
        if w <= 1 { return Ok(0); }
        let rows = w * 8;
        let rl = 2304usize;
        let blocks: Vec<(usize, usize)> = (0..w).map(|r| crate::tp_xport::row_block(rows, r, w)).collect();
        let (r0, r1) = blocks[self.rank];
        let mine: Vec<f32> = (0..rows * rl).map(|i| (self.rank as f32 + 1.0) * (i % 11) as f32).collect();
        let sum_r: f32 = (1..=w).map(|r| r as f32).sum();
        let buf = dev.htod_sync_copy(&mine)?;
        self.sp_reduce_scatter(dev, *buf.device_ptr(), rl, &blocks)?;
        dev.synchronize()?;
        let got = dev.dtoh_sync_copy(&buf)?;
        let mut bad = 0usize;
        for row in r0..r1 {
            for j in 0..rl {
                let idx = row * rl + j;
                let want = sum_r * (idx % 11) as f32;
                if got[idx].to_bits() != want.to_bits() { bad += 1; }
            }
        }
        // 2-buffer multi all-gather (e.g. BF16 activation [rows][D] + routing ids [rows][8])
        let mut b1 = vec![0u16; rows * rl];
        let mut b2 = vec![0i32; rows * 8];
        for row in r0..r1 {
            for j in 0..rl { b1[row * rl + j] = ((self.rank + 1) * 100 + (row % 31) + (j % 17)) as u16; }
            for j in 0..8 { b2[row * 8 + j] = ((self.rank + 1) * 1000 + row * 10 + j) as i32; }
        }
        let d1 = dev.htod_sync_copy(&b1)?;
        let d2 = dev.htod_sync_copy(&b2)?;
        self.all_gather_rows_multi(dev, &[(*d1.device_ptr(), rl * 2), (*d2.device_ptr(), 8 * 4)], &blocks)?;
        dev.synchronize()?;
        let g1 = dev.dtoh_sync_copy(&d1)?;
        let g2 = dev.dtoh_sync_copy(&d2)?;
        for (r, &(a, b)) in blocks.iter().enumerate() {
            for row in a..b {
                for j in 0..rl {
                    let want = ((r + 1) * 100 + (row % 31) + (j % 17)) as u16;
                    if g1[row * rl + j] != want { bad += 1; }
                }
                for j in 0..8 {
                    let want = ((r + 1) * 1000 + row * 10 + j) as i32;
                    if g2[row * 8 + j] != want { bad += 1; }
                }
            }
        }
        Ok(bad)
    }

    /// Boot self-test for 1-RTT All-to-All (DeepSpeed-Ulysses sequence-parallel attention, arXiv:2309.14509).
    pub fn self_test_all_to_all(&self, dev: &Arc<CudaDevice>) -> Result<usize> {
        let w = self.world;
        if w <= 1 { return Ok(0); }
        let slice_u32 = 1024usize; // 4096 B/peer (exercises single + dual rail paths cleanly)
        let mut host = vec![0u32; w * slice_u32];
        for p in 0..w {
            for j in 0..slice_u32 {
                host[p * slice_u32 + j] = ((self.rank as u32 + 1) << 24) | ((p as u32 + 1) << 16) | (j as u32);
            }
        }
        let d_src = dev.htod_sync_copy(&host)?;
        let mut d_dst = dev.alloc_zeros::<u32>(w * slice_u32)?;
        dev.memset_zeros(&mut d_dst)?;
        self.all_to_all(dev, *d_src.device_ptr(), *d_dst.device_ptr(), slice_u32 * 4)?;
        dev.synchronize()?;
        let got = dev.dtoh_sync_copy(&d_dst)?;
        let mut bad = 0usize;
        for p in 0..w {
            for j in 0..slice_u32 {
                let want = ((p as u32 + 1) << 24) | ((self.rank as u32 + 1) << 16) | (j as u32);
                if got[p * slice_u32 + j] != want { bad += 1; }
            }
        }
        Ok(bad)
    }

    /// True when sequence-parallel prefill collectives (`sp_reduce_scatter` / `all_gather_rows_multi`) are supported.
    pub fn sp_ready(&self) -> bool {
        match &self.kind {
            Kind::Solo => false,
            Kind::Rdma { rdma, .. } => rdma.sp_ready(),
            Kind::Loop { .. } => matches!(self.world, 2 | 3 | 4 | 6 | 8),
        }
    }

    /// True when 1-RTT All-to-All (DeepSpeed-Ulysses sequence-parallel attention, arXiv:2309.14509) is available.
    pub fn a2a_ready(&self) -> bool {
        match &self.kind {
            Kind::Solo => false,
            Kind::Rdma { rdma, .. } => rdma.a2a_ready(),
            Kind::Loop { .. } => matches!(self.world, 2 | 3 | 4 | 6 | 8),
        }
    }

    /// 1-RTT chunked All-to-All across all ranks (`src` and `dst` each hold `world` slices of `slice_bytes`).
    /// Powers DeepSpeed-Ulysses sequence-parallel attention (Jacobs et al., arXiv:2309.14509).
    pub fn all_to_all(&self, dev: &Arc<CudaDevice>, src: u64, dst: u64, slice_bytes: usize) -> Result<()> {
        ensure!(slice_bytes > 0 && slice_bytes % 16 == 0, "inkling2 TP all_to_all: slice_bytes {slice_bytes} must be 16 B aligned");
        let raw_stream = crate::inkling2::gpu::Ink2Kernels::global(dev)?.stream.get();
        unsafe { result::ctx::set_current(*dev.cu_primary_ctx()) }.map_err(|e| anyhow!("ctx bind: {e}"))?;
        match &self.kind {
            Kind::Solo => Ok(()),
            Kind::Rdma { rdma, stream } => {
                ensure!(raw_stream.is_null(), "inkling2 TP all_to_all cannot run inside CUDA graph capture");
                rdma.all_to_all(stream, src, dst, slice_bytes).context("inkling2 TP Ulysses all_to_all")?;
                REDUCES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                BYTES.fetch_add(((self.world - 1) * slice_bytes) as u64, std::sync::atomic::Ordering::Relaxed);
                Ok(())
            }
            Kind::Loop { a2a_txs, a2a_rx, barrier, .. } => {
                unsafe { result::stream::synchronize(raw_stream) }.map_err(|e| anyhow!("loop a2a sync: {e}"))?;
                dev.synchronize()?;
                let w = self.world;
                let slice_u32 = slice_bytes / 4;
                let mut host_src = vec![0u32; w * slice_u32];
                unsafe { result::memcpy_dtoh_sync(&mut host_src, src) }.map_err(|e| anyhow!("loop a2a dtoh: {e}"))?;
                let mut host_dst = vec![0u32; w * slice_u32];
                host_dst[self.rank * slice_u32..(self.rank + 1) * slice_u32]
                    .copy_from_slice(&host_src[self.rank * slice_u32..(self.rank + 1) * slice_u32]);
                for (p, tx) in a2a_txs.iter().enumerate() {
                    if p == self.rank { continue; }
                    let mut msg = Vec::with_capacity(slice_u32 + 1);
                    msg.extend_from_slice(&host_src[p * slice_u32..(p + 1) * slice_u32]);
                    msg.push(self.rank as u32);
                    tx.send(msg).map_err(|_| anyhow!("loopback a2a peer gone"))?;
                }
                let g = a2a_rx.lock().unwrap();
                for _ in 0..w - 1 {
                    let mut v = g.recv().map_err(|_| anyhow!("loopback a2a peer gone"))?;
                    let peer = v.pop().unwrap() as usize;
                    ensure!(v.len() == slice_u32, "loop a2a slice len mismatch from peer {peer}");
                    host_dst[peer * slice_u32..(peer + 1) * slice_u32].copy_from_slice(&v);
                }
                drop(g);
                barrier.wait();
                unsafe { result::memcpy_htod_sync(dst, &host_dst) }.map_err(|e| anyhow!("loop a2a htod: {e}"))?;
                Ok(())
            }
        }
    }

    /// Mean microseconds per all-reduce of `n` floats over `reps` back-to-back reduces (eager launches; a collective).
    pub fn time_frames(&self, dev: &Arc<CudaDevice>, n: usize, reps: usize) -> Result<f64> {
        let mut buf = dev.alloc_zeros::<f32>(n)?;
        dev.memset_zeros(&mut buf)?;
        dev.synchronize()?;
        let p = *buf.device_ptr();
        for _ in 0..8 { self.all_reduce(dev, p, n)?; }
        dev.synchronize()?;
        let t = std::time::Instant::now();
        for _ in 0..reps { self.all_reduce(dev, p, n)?; }
        dev.synchronize()?;
        self.check("timing the exchange")?;
        Ok(t.elapsed().as_secs_f64() * 1e6 / reps as f64)
    }

    pub fn abort_code(&self) -> u64 {
        match &self.kind {
            Kind::Rdma { rdma, .. } => rdma.abort_code(),
            Kind::Solo | Kind::Loop { .. } => 0,
        }
    }

    /// Fail loudly once the link has aborted (`what` = where it was noticed).
    pub fn check(&self, what: &str) -> Result<()> {
        let c = self.abort_code();
        ensure!(
            c == 0,
            "inkling2 TP rank {}: the transport ABORTED (code {c:#x}), noticed {what}: a peer stopped answering (dead or hung node); \
             every reduce since returned a partial sum, so this rank stops (watchdog dump: /tmp/tp_wdog.rank{}.dump)",
            self.rank, self.rank
        );
        Ok(())
    }

    /// In-place exact fp32 sum all-reduce of `n` floats at device address `base`, on Inkling-2's current stream.
    pub fn all_reduce(&self, dev: &Arc<CudaDevice>, base: u64, n: usize) -> Result<()> {
        if n == 0 { return Ok(()); }
        REDUCES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        BYTES.fetch_add((n * 4) as u64, std::sync::atomic::Ordering::Relaxed);
        let raw_stream = crate::inkling2::gpu::Ink2Kernels::global(dev)?.stream.get();
        unsafe { result::ctx::set_current(*dev.cu_primary_ctx()) }.map_err(|e| anyhow!("ctx bind: {e}"))?;
        match &self.kind {
            Kind::Solo => Ok(()),
            Kind::Rdma { rdma, stream } => {
                let wide_row_len = if n % 2304 == 0 { Some(2304) } else { None };
                rdma.all_reduce_raw(raw_stream, stream, base, n, wide_row_len)
            }
            Kind::Loop { txs, rx, barrier, .. } => {
                unsafe { result::stream::synchronize(raw_stream) }.map_err(|e| anyhow!("loop sync: {e}"))?;
                dev.synchronize()?;
                let w = self.world;
                let mut mine = vec![0f32; n];
                unsafe { result::memcpy_dtoh_sync(&mut mine, base) }.map_err(|e| anyhow!("loop dtoh: {e}"))?;
                let mut tagged = mine.clone();
                tagged.push(self.rank as f32);
                for (r, tx) in txs.iter().enumerate() {
                    if r != self.rank { tx.send(tagged.clone()).map_err(|_| anyhow!("loopback peer gone"))?; }
                }
                let mut parts: Vec<Option<Vec<f32>>> = vec![None; w];
                parts[self.rank] = Some(mine);
                let g = rx.lock().unwrap();
                for _ in 0..w - 1 {
                    let mut v = g.recv().map_err(|_| anyhow!("loopback peer gone"))?;
                    let r = v.pop().unwrap() as usize;
                    parts[r] = Some(v);
                }
                drop(g);
                barrier.wait();
                // Canonical association: W=2 p0+p1; W=4 (p0+p2)+(p1+p3); W=8 ((p0+p2)+(p4+p6))+((p1+p3)+(p5+p7))
                let sum: Vec<f32> = if w == 8 {
                    let p = |r: usize| parts[r].as_ref().unwrap();
                    (0..n).map(|i| ((p(0)[i] + p(2)[i]) + (p(4)[i] + p(6)[i])) + ((p(1)[i] + p(3)[i]) + (p(5)[i] + p(7)[i]))).collect()
                } else if w == 4 {
                    let p = |r: usize| parts[r].as_ref().unwrap();
                    (0..n).map(|i| (p(0)[i] + p(2)[i]) + (p(1)[i] + p(3)[i])).collect()
                } else {
                    let mut s = parts[0].as_ref().unwrap().clone();
                    for r in 1..w { for (a, b) in s.iter_mut().zip(parts[r].as_ref().unwrap()) { *a += *b; } }
                    s
                };
                unsafe { result::memcpy_htod_sync(base, &sum) }.map_err(|e| anyhow!("loop htod: {e}"))?;
                Ok(())
            }
        }
    }

    /// Drop 2 (#2): in-place canonical FP32 row-block reduce-scatter across `blocks`.
    /// After completion, `base + blocks[self.rank].0 * row_len * 4` holds this rank's exact canonical sum rows.
    pub fn sp_reduce_scatter(&self, dev: &Arc<CudaDevice>, base: u64, row_len: usize, blocks: &[(usize, usize)]) -> Result<()> {
        ensure!(blocks.len() == self.world, "inkling2 TP sp_reduce_scatter: {} blocks for world {}", blocks.len(), self.world);
        let raw_stream = crate::inkling2::gpu::Ink2Kernels::global(dev)?.stream.get();
        unsafe { result::ctx::set_current(*dev.cu_primary_ctx()) }.map_err(|e| anyhow!("ctx bind: {e}"))?;
        match &self.kind {
            Kind::Solo => Ok(()),
            Kind::Rdma { rdma, stream } => {
                ensure!(raw_stream.is_null(), "inkling2 TP sp_reduce_scatter cannot run inside CUDA graph capture");
                let wire = rdma.sp_reduce_scatter(stream, base, row_len, blocks)
                    .context("inkling2 TP 2-rail sp_reduce_scatter")?;
                REDUCES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                BYTES.fetch_add(wire as u64, std::sync::atomic::Ordering::Relaxed);
                Ok(())
            }
            Kind::Loop { txs, rx, barrier, .. } => {
                unsafe { result::stream::synchronize(raw_stream) }.map_err(|e| anyhow!("loop rs sync: {e}"))?;
                dev.synchronize()?;
                let w = self.world;
                let total_rows = blocks.iter().map(|b| b.1).max().unwrap_or(0);
                let n = total_rows * row_len;
                let mut mine = vec![0f32; n];
                unsafe { result::memcpy_dtoh_sync(&mut mine, base) }.map_err(|e| anyhow!("loop rs dtoh: {e}"))?;
                let mut tagged = mine.clone();
                tagged.push(self.rank as f32);
                for (r, tx) in txs.iter().enumerate() {
                    if r != self.rank { tx.send(tagged.clone()).map_err(|_| anyhow!("loopback peer gone"))?; }
                }
                let mut parts: Vec<Option<Vec<f32>>> = vec![None; w];
                parts[self.rank] = Some(mine);
                let g = rx.lock().unwrap();
                for _ in 0..w - 1 {
                    let mut v = g.recv().map_err(|_| anyhow!("loopback peer gone"))?;
                    let r = v.pop().unwrap() as usize;
                    parts[r] = Some(v);
                }
                drop(g);
                barrier.wait();
                // Poison non-owned rows with NaN so loopback simulation catches any accidental peer-row read!
                let (my_r0, my_r1) = blocks[self.rank];
                let mut out = vec![f32::NAN; n];
                let lo = my_r0 * row_len;
                let hi = my_r1 * row_len;
                if w == 8 {
                    let p = |r: usize| parts[r].as_ref().unwrap();
                    for i in lo..hi {
                        out[i] = ((p(0)[i] + p(2)[i]) + (p(4)[i] + p(6)[i])) + ((p(1)[i] + p(3)[i]) + (p(5)[i] + p(7)[i]));
                    }
                } else if w == 4 {
                    let p = |r: usize| parts[r].as_ref().unwrap();
                    for i in lo..hi { out[i] = (p(0)[i] + p(2)[i]) + (p(1)[i] + p(3)[i]); }
                } else {
                    for i in lo..hi {
                        let mut s = parts[0].as_ref().unwrap()[i];
                        for r in 1..w { s += parts[r].as_ref().unwrap()[i]; }
                        out[i] = s;
                    }
                }
                unsafe { result::memcpy_htod_sync(base, &out) }.map_err(|e| anyhow!("loop rs htod: {e}"))?;
                Ok(())
            }
        }
    }

    /// Byte-exact row-block all-gather of one buffer (`blocks.len() == self.world`, `row_bytes % 16 == 0`).
    pub fn all_gather_rows(&self, dev: &Arc<CudaDevice>, base: u64, row_bytes: usize, blocks: &[(usize, usize)]) -> Result<()> {
        self.all_gather_rows_multi(dev, &[(base, row_bytes)], blocks)
    }

    /// Byte-exact row-block all-gather of one or more row buffers (`(base, row_bytes)`, each `row_bytes % 16 == 0`).
    /// After completion, every rank holds all `blocks[0..world]` rows of every buffer.
    pub fn all_gather_rows_multi(&self, dev: &Arc<CudaDevice>, bufs: &[(u64, usize)], blocks: &[(usize, usize)]) -> Result<()> {
        ensure!(blocks.len() == self.world, "inkling2 TP all_gather: {} blocks for world {}", blocks.len(), self.world);
        ensure!(!bufs.is_empty(), "inkling2 TP all_gather: empty buffer list");
        let raw_stream = crate::inkling2::gpu::Ink2Kernels::global(dev)?.stream.get();
        unsafe { result::ctx::set_current(*dev.cu_primary_ctx()) }.map_err(|e| anyhow!("ctx bind: {e}"))?;
        match &self.kind {
            Kind::Solo => Ok(()),
            Kind::Rdma { rdma, stream } => {
                ensure!(raw_stream.is_null(), "inkling2 TP all_gather_rows_multi cannot run inside CUDA graph capture");
                let wire = rdma.sp_all_gather_multi(stream, bufs, blocks)
                    .context("inkling2 TP 2-rail all_gather_rows_multi")?;
                REDUCES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                BYTES.fetch_add(wire as u64, std::sync::atomic::Ordering::Relaxed);
                Ok(())
            }
            Kind::Loop { txs, rx, barrier, .. } => {
                for &(_, rb) in bufs {
                    ensure!(rb % 4 == 0, "inkling2 TP loop all_gather: row_bytes {rb} not a multiple of 4");
                }
                unsafe { result::stream::synchronize(raw_stream) }.map_err(|e| anyhow!("loop sync: {e}"))?;
                dev.synchronize()?;
                let (r0, r1) = blocks[self.rank];
                let my_rows = r1 - r0;
                let row_f_sum: usize = bufs.iter().map(|&(_, rb)| rb / 4).sum();
                let mut tagged = Vec::with_capacity(my_rows * row_f_sum + 1);
                for &(base, rb) in bufs {
                    let len = my_rows * (rb / 4);
                    let mut tmp = vec![0f32; len];
                    unsafe { result::memcpy_dtoh_sync(&mut tmp, base + (r0 * rb) as u64) }.map_err(|e| anyhow!("loop ag dtoh: {e}"))?;
                    tagged.extend_from_slice(&tmp);
                }
                tagged.push(self.rank as f32);
                for (r, tx) in txs.iter().enumerate() {
                    if r != self.rank { tx.send(tagged.clone()).map_err(|_| anyhow!("loopback peer gone"))?; }
                }
                let g = rx.lock().unwrap();
                for _ in 0..self.world - 1 {
                    let mut v = g.recv().map_err(|_| anyhow!("loopback peer gone"))?;
                    let peer = v.pop().unwrap() as usize;
                    let (p0, p1) = blocks[peer];
                    let prows = p1 - p0;
                    ensure!(v.len() == prows * row_f_sum, "loop ag size mismatch for peer {peer}");
                    let mut off = 0usize;
                    for &(base, rb) in bufs {
                        let len = prows * (rb / 4);
                        unsafe { result::memcpy_htod_sync(base + (p0 * rb) as u64, &v[off..off + len]) }
                            .map_err(|e| anyhow!("loop ag htod: {e}"))?;
                        off += len;
                    }
                }
                drop(g);
                barrier.wait();
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inkling2_verify_and_logits_geometries_fit_velo_fabric_tiers() {
        const D: usize = 2304;
        const MAX_OUT: usize = 48;
        const VOCAB: usize = 200_058;
        // Every verify batch 1..=48 fits in SINGLE_CAP_BYTES (768 KiB) -> 100% CUDA-graph capturable!
        assert!(MAX_OUT * D * 4 <= SINGLE_CAP_BYTES);
        // Every m >= 1 is 128 B-aligned (32 floats) -> dual_rsag_split and dual_rsag8_split always succeed.
        for m in [1usize, 2, 4, 8, 16, 32, 48] {
            let n = m * D;
            assert!(crate::dsv41::tp::xchg::dual_split(n).is_some(), "m={m}");
            assert!(crate::dsv41::tp::xchg::dual_rsag_split(n).is_some(), "m={m}");
            assert!(crate::dsv41::tp::xchg::dual_rsag8_split(n).is_some(), "m={m}");
        }
        // Logits all-gather: rank-0 slice width `per` is 512 B-aligned and fits one Pipe/single ring slot for all worlds.
        for w in [2usize, 3, 4, 6, 8] {
            let per = super::super::shard::vocab_range(VOCAB, w, 0).1;
            assert_eq!(per % 128, 0, "world {w}");
            let cap = if w.is_power_of_two() { crate::tp_xport::EPOCH_FLOATS * 4 } else { SINGLE_CAP_BYTES };
            assert!((per * 4) % 16 == 0 && per * 4 <= cap, "world {w} per={per}");
        }
    }
}
