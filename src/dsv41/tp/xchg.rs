//! DSV4.1 TP (leg PX1): the world-2 exchange as the external reference recipe runs it (PLAN/DSV41_RX4_REPORT.md B.2 /
//! B.5, PLAN/DSV41_PX1_REPORT.md): a dsv41 copy of the doorbell stage that is a PDL dependent (waits for the
//! producer, releases the collect at once) and a dsv41 copy of the collect that is a PDL dependent of the stage
//! (releases the consumer at once, polls the receive gate without sleeping, copies out with vector loads over
//! several blocks). Same protocol, rings, tags, proxy and wire bytes as the shared pair (`native/tp_doorbell.h`);
//! the shared kernels (`tp_gate_copy_signal_mb`, `tp_wait_add`) are not touched and stay selectable
//! (`--dsv41-xchg legacy`). The per-element sum is the shared mode-2 sum (rank 0's partial first): bit-identical.
//!
//! `--dsv41-xchg-trace <dir>` (diagnostics): per-epoch globaltimer stamps of both kernels into a mapped ring, the
//! proxy's own per-epoch stamps (`net::trace_data`), and a dump of both to `<dir>/xtr_r<rank>_<n>.csv` whenever the
//! link has been quiet for 1.5 s after new exchanges, with a GPU / host clock calibration taken at each dump.

use crate::dsv41::gpu::engram::MappedHost;
use anyhow::{ensure, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use std::sync::Arc;

pub const STAGE: &str = "d41x_xc_stage";
pub const COLLECT: &str = "d41x_xc_collect";
pub const COLLECT_G: &str = "d41x_xc_collect_g";
/// R3: fused dual-rail world-2 stage & collect (grid.y = rail)
pub const STAGE2D: &str = "d41x_xcd_stage";
pub const STAGE2D_B16: &str = "d41x_xcd_stage_b16";
pub const COLLECT2D: &str = "d41x_xcd_collect";
pub const COLLECT2D_G: &str = "d41x_xcd_collect_g";
pub const COLLECT2D_B16: &str = "d41x_xcd_collect_b16";
pub const A2A2D_COLLECT: &str = "d41x_a2a2d_collect";
/// leg PX2: the world-4 pair on the uniform one-shot single-stage ctx (`(p0 + p2) + (p1 + p3)`)
pub const STAGE4: &str = "d41x_xc4_stage";
pub const COLLECT4: &str = "d41x_xc4_collect";
pub const COLLECT4_G: &str = "d41x_xc4_collect_g";
/// Phase 2 / TP=8: world-8 single-stage collect (`((p0+p2)+(p4+p6)) + ((p1+p3)+(p5+p7))`, stages via STAGE4/STAGE4D)
pub const COLLECT8: &str = "d41x_xc8_collect";
pub const COLLECT8_G: &str = "d41x_xc8_collect_g";
/// leg PX2: the reduce-scatter (primary link tree) + all-gather (single-stage ctx) form for wide world-4/8 frames
pub const TSTAGE: &str = "d41x_xct_stage";
pub const TCOLLECT: &str = "d41x_xct_collect";
pub const GATHER4: &str = "d41x_xc4_gather";
pub const GATHER4_G: &str = "d41x_xc4_gather_g";
pub const GATHER8: &str = "d41x_xc8_gather";
/// leg TX1: the dual-rail world-4/8 pair, one launch each (grid.y = rail)
pub const STAGE4D: &str = "d41x_xc4d_stage";
pub const COLLECT4D: &str = "d41x_xc4d_collect";
pub const COLLECT4D_G: &str = "d41x_xc4d_collect_g";
pub const COLLECT8D: &str = "d41x_xc8d_collect";
pub const COLLECT8D_G: &str = "d41x_xc8d_collect_g";
/// Phase 1D + Phase A: dual-rail RS+AG kernels (grid.y = rail)
pub const TSTAGE_D: &str = "d41x_xctd_stage";
pub const TCOLLECT_D: &str = "d41x_xctd_collect";
pub const GATHER4D: &str = "d41x_xc4d_gather";
pub const GATHER4D_G: &str = "d41x_xc4d_gather_g";
pub const GATHER8D: &str = "d41x_xc8d_gather";
/// Phase A: opt-in BF16 wire compression (`--dsv41-wire-bf16 on`)
pub const STAGE_B16: &str = "d41x_xc_stage_b16";
pub const STAGE4_B16: &str = "d41x_xc4_stage_b16";
pub const STAGE4D_B16: &str = "d41x_xc4d_stage_b16";
pub const COLLECT_B16: &str = "d41x_xc_collect_b16";
pub const COLLECT4_B16: &str = "d41x_xc4_collect_b16";
pub const COLLECT4D_B16: &str = "d41x_xc4d_collect_b16";
pub const COLLECT8_B16: &str = "d41x_xc8_collect_b16";
pub const COLLECT8D_B16: &str = "d41x_xc8d_collect_b16";
/// Velo-Fabric Arbitrary-W (TP3 / TP6 / W in 3..=8) + 1-RTT All-to-All (DeepSpeed-Ulysses, Jacobs et al. arXiv:2309.14509)
pub const COLLECTN: &str = "d41x_xcn_collect";
pub const COLLECTND: &str = "d41x_xcnd_collect";
pub const COLLECTN_B16: &str = "d41x_xcn_collect_b16";
pub const COLLECTND_B16: &str = "d41x_xcnd_collect_b16";
pub const A2A_STAGE: &str = "d41x_a2a_stage";
pub const A2AD_STAGE: &str = "d41x_a2ad_stage";
pub const A2A_COLLECT: &str = "d41x_a2a_collect";
pub const A2AD_COLLECT: &str = "d41x_a2ad_collect";
pub const A2A2_COLLECT: &str = "d41x_a2a2_collect";
pub const RSN_COLLECT: &str = "d41x_rsn_collect";
pub const RSND_COLLECT: &str = "d41x_rsnd_collect";
pub const A2A_VAR_STAGE: &str = "d41x_a2a_var_stage";
pub const A2AD_VAR_STAGE: &str = "d41x_a2ad_var_stage";
pub const AG_VAR_STAGE: &str = "d41x_ag_var_stage";
pub const AGD_VAR_STAGE: &str = "d41x_agd_var_stage";
pub const AG_VAR_COLLECT: &str = "d41x_ag_var_collect";
pub const AGD_VAR_COLLECT: &str = "d41x_agd_var_collect";
/// U6a: DSV4.1 LAGA BF16 attention-output All-to-All pack / unpack
pub const LAGA_PACK_B16: &str = "d41x_laga_pack_b16";
pub const LAGA_UNPACK_B16: &str = "d41x_laga_unpack_b16";

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct D41LagaBlocks {
    pub r0: [i32; 8],
    pub m: [i32; 8],
}

impl D41LagaBlocks {
    pub fn from_blocks(blocks: &[(usize, usize)]) -> Self {
        let mut out = Self::default();
        for (p, &(a, b)) in blocks.iter().enumerate().take(8) {
            out.r0[p] = a as i32;
            out.m[p] = (b - a) as i32;
        }
        out
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct D41LagaGroups {
    pub g0: [i32; 8],
    pub ng: [i32; 8],
}

impl D41LagaGroups {
    pub fn for_world(cfg: &crate::dsv41::config::Dsv41Config, world: usize) -> (Self, usize) {
        let mut out = Self::default();
        let mut max_ng = 0usize;
        for p in 0..world.min(8) {
            let rg = crate::dsv41::tp::fwd::Ranges::of(cfg, world, p);
            out.g0[p] = rg.g0 as i32;
            out.ng[p] = rg.ng as i32;
            max_ng = max_ng.max(rg.ng);
        }
        (out, max_ng)
    }
}

pub fn laga_pack_b16(stream: u64, mid: u64, a2a_src: u64, max_my_n: usize, world: usize, my_cols_bf16: usize, stride_cols_bf16: usize, blocks: &[(usize, usize)]) -> Result<()> {
    ensure!(my_cols_bf16 % 8 == 0 && stride_cols_bf16 % 8 == 0, "laga_pack_b16: cols {my_cols_bf16}/{stride_cols_bf16} not multiple of 8");
    let pdl = crate::dsv41::gpu::pdl_on();
    let (mut m, mut dst) = (mid, a2a_src);
    let (mut mm, mut mv4, mut sv4) = (max_my_n as i32, (my_cols_bf16 / 8) as i32, (stride_cols_bf16 / 8) as i32);
    let mut blk = D41LagaBlocks::from_blocks(blocks);
    let mut p = [
        &mut m as *mut u64 as *mut std::ffi::c_void,
        &mut dst as *mut u64 as *mut std::ffi::c_void,
        &mut mm as *mut i32 as *mut std::ffi::c_void,
        &mut mv4 as *mut i32 as *mut std::ffi::c_void,
        &mut sv4 as *mut i32 as *mut std::ffi::c_void,
        &mut blk as *mut D41LagaBlocks as *mut std::ffi::c_void,
    ];
    crate::dsv41::gpu::launch_raw(LAGA_PACK_B16, (world as u32, max_my_n as u32, 1), (256, 1, 1), 0, stream, pdl, &mut p).context("dsv41 LAGA pack b16")
}

#[allow(clippy::too_many_arguments)]
pub fn laga_unpack_b16(stream: u64, a2a_dst: u64, mid_full: u64, max_my_n: usize, my_n: usize, world: usize, stride_cols_bf16: usize, all_cols_bf16: usize, olr_bf16: usize, grp: D41LagaGroups) -> Result<()> {
    ensure!(stride_cols_bf16 % 8 == 0 && all_cols_bf16 % 8 == 0 && olr_bf16 % 8 == 0, "laga_unpack_b16: cols not multiple of 8");
    let pdl = crate::dsv41::gpu::pdl_on();
    let (mut src, mut dst) = (a2a_dst, mid_full);
    let (mut mm, mut sv4, mut av4, mut ov4) = (max_my_n as i32, (stride_cols_bf16 / 8) as i32, (all_cols_bf16 / 8) as i32, (olr_bf16 / 8) as i32);
    let mut g = grp;
    let mut p = [
        &mut src as *mut u64 as *mut std::ffi::c_void,
        &mut dst as *mut u64 as *mut std::ffi::c_void,
        &mut mm as *mut i32 as *mut std::ffi::c_void,
        &mut sv4 as *mut i32 as *mut std::ffi::c_void,
        &mut av4 as *mut i32 as *mut std::ffi::c_void,
        &mut ov4 as *mut i32 as *mut std::ffi::c_void,
        &mut g as *mut D41LagaGroups as *mut std::ffi::c_void,
    ];
    crate::dsv41::gpu::launch_raw(LAGA_UNPACK_B16, (world as u32, my_n as u32, 1), (256, 1, 1), 0, stream, pdl, &mut p).context("dsv41 LAGA unpack b16")
}

/// Variable per-rank byte spans (passed by value to `d41x_{a2a,ag}_var_*` kernels for uneven TP3/TP6 row blocks).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct TpSpan8 {
    pub off: [u64; 8],
    pub len: [u32; 8],
}

/// Phase 1D: split an `n`-float frame into two 64 B-aligned (16-float) disjoint halves for dual-rail RS+AG.
/// Each half is a valid `rsag_plan` input (`>= 16` floats, `% 16 == 0`).
pub fn dual_rsag_split(n: usize) -> Option<(usize, usize)> {
    if n % 16 != 0 || n < 32 { return None; }
    let ha = ((n / 16) / 2) * 16;
    if ha == 0 || ha >= n { None } else { Some((ha, n - ha)) }
}

/// Phase A: split an `n`-float frame into two 128 B-aligned (32-float) disjoint halves for world-8 dual-rail RS+AG.
pub fn dual_rsag8_split(n: usize) -> Option<(usize, usize)> {
    if n % 32 != 0 || n < 64 { return None; }
    let ha = ((n / 32) / 2) * 32;
    if ha == 0 || ha >= n { None } else { Some((ha, n - ha)) }
}

/// Involution mapping world-8 rank `p` to its contiguous eighth slot in RS+AG (swaps bit 1 and bit 2).
#[inline]
pub fn slot_w8(p: usize) -> usize { ((p & 2) << 1) | ((p & 4) >> 1) | (p & 1) }

/// Phase A: world-8 reduce-scatter + all-gather plan for an `n`-float frame (`n % 32 == 0`):
/// 3 tree epochs on primary link (`^2` half, `^4` quarter, `^1` eighth) + 1 all-peers gather on single-stage ctx.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RsAg8 {
    pub a_send: usize, pub a_keep: usize, pub half: usize,
    pub b_send: usize, pub b_keep: usize, pub quarter: usize,
    pub c_send: usize, pub c_keep: usize, pub eighth: usize,
}
pub fn rsag8_plan(n: usize, rank: usize) -> Option<RsAg8> {
    if n % 32 != 0 || rank > 7 { return None; }
    let (half, quarter, eighth) = (n / 2, n / 4, n / 8);
    let h = (rank >> 1) & 1;
    let q = (h << 1) | ((rank >> 2) & 1);
    let o = (q << 1) | (rank & 1); // == slot_w8(rank)
    Some(RsAg8 {
        a_send: (1 - h) * half, a_keep: h * half, half,
        b_send: (q ^ 1) * quarter, b_keep: q * quarter, quarter,
        c_send: (o ^ 1) * eighth, c_keep: o * eighth, eighth,
    })
}
/// Bytes one rank puts on the wire at world 8: (one all-peers stage = 7F, RS+AG = 1.75F).
pub fn wire_bytes_w8(n: usize) -> (usize, usize) { (7 * n * 4, (n / 2 + n / 4 + n / 8 + 7 * (n / 8)) * 4) }

/// leg PX2: the reduce-scatter + all-gather plan of an `n`-float frame at world 4 for `rank` (float offsets / lengths):
/// epoch A (partner rank ^ 2) sends `a_send` and adds into `a_keep`; epoch B (partner rank ^ 1) sends `b_send` and adds
/// into `b_keep` (= quarter `rank`); the all-gather sends quarter `rank` and places quarter p at p * n / 4.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RsAg { pub a_send: usize, pub a_keep: usize, pub half: usize, pub b_send: usize, pub b_keep: usize, pub quarter: usize }
pub fn rsag_plan(n: usize, rank: usize) -> Option<RsAg> {
    if n % 16 != 0 || rank > 3 { return None; }
    let (half, quarter) = (n / 2, n / 4);
    let (b1, b0) = ((rank >> 1) & 1, rank & 1);
    Some(RsAg { a_send: (1 - b1) * half, a_keep: b1 * half, half, b_send: b1 * half + (1 - b0) * quarter, b_keep: rank * quarter, quarter })
}
/// Bytes one rank puts on the wire for an `n`-float frame at world 4: (one all-peers stage, reduce-scatter + all-gather).
pub fn wire_bytes_w4(n: usize) -> (usize, usize) { (3 * n * 4, (n / 2 + n / 4 + 3 * (n / 4)) * 4) }
/// float4s per thread per pass in the world-4 collect (`d41x_xc4_sum<2>`: 2 x 4 loads in flight a thread)
const C4_PER: usize = 2;
const CLOCK: &str = "d41x_xc_clock";
/// mirrors `D41X_XC_TR_EPOCHS` / `_STRIDE` and the stamp slots of `kernels/gpu_dsv41/07_xchg.cuh`
pub const TR_EPOCHS: usize = 4096;
pub const TR_STRIDE: usize = 8;
pub const TR_NAMES: [&str; TR_STRIDE] = ["s_in", "s_go", "s_db", "c_in", "c_go", "c_seen", "c_end0", "c_endl"];
/// collect threads per block and float4s per thread per pass (the kernel's `d41x_xc_sum<4>`)
const C_THREADS: usize = 256;
const C_PER: usize = 4;

/// `--dsv41-xchg pdl|legacy` (default pdl).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode { Pdl, Legacy }
pub fn mode_of(raw: Option<&str>) -> Mode {
    match raw.map(|v| v.trim().to_ascii_lowercase()).as_deref() { Some("legacy") => Mode::Legacy, _ => Mode::Pdl }
}
pub fn mode() -> Mode { mode_of(crate::opts::get(crate::opt!("dsv41-xchg")).as_deref()) }

/// The stage's grid for an `n`-float frame (the shared K1m's: one 16 B store per thread per block of 256, <= 32 blocks).
pub fn stage_blocks(n: usize) -> u32 { super::rdma::k1_mb_blocks(n) }
/// The collect's grid: enough blocks that each thread issues `C_PER` float4 loads per side in one pass, 1..=32.
pub fn collect_blocks(n: usize) -> u32 { ((n / 4).div_ceil(C_THREADS * C_PER)).clamp(1, 32) as u32 }
/// The world-4 collect's grid: each thread loads `C4_PER` float4s of each of the 4 partials in one pass, 1..=32 blocks.
pub fn collect4_blocks(n: usize) -> u32 { ((n / 4).div_ceil(C_THREADS * C4_PER)).clamp(1, 32) as u32 }
/// The world-8 collect's grid: each thread loads 1 float4 of each of the 8 partials in one pass, 1..=32 blocks.
pub fn collect8_blocks(n: usize) -> u32 { ((n / 4).div_ceil(C_THREADS)).clamp(1, 32) as u32 }

/// Exchanges built on the dsv41 pair (engagement receipt; a graph replay re-runs captured launches without counting).
static BUILT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ENGAGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub fn built() -> u64 { BUILT.load(std::sync::atomic::Ordering::Relaxed) }
static DUAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Exchanges built on both rails (leg TX1, `--dsv41-dual-rail`).
pub fn dual_built() -> u64 { DUAL.load(std::sync::atomic::Ordering::Relaxed) }
/// The two halves (floats) of an `n`-float frame for the dual-rail exchange: the first is half the frame's float4s
/// rounded down (16 B aligned), the second the rest. None when either half would be empty or `n` is not a float4 multiple.
pub fn dual_split(n: usize) -> Option<(usize, usize)> {
    if n % 4 != 0 { return None; }
    let ha = (n / 4 / 2) * 4;
    if ha == 0 || ha >= n { None } else { Some((ha, n - ha)) }
}
static RSAG: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub fn rsag_built() -> u64 { RSAG.load(std::sync::atomic::Ordering::Relaxed) }
static SP_LAYERS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SP_IDX_LAYERS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SP_LAGA_LAYERS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub fn sp_layer() { SP_LAYERS.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
pub fn sp_idx_layer() { SP_IDX_LAYERS.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
pub fn sp_laga_layer() { SP_LAGA_LAYERS.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
pub fn engaged() -> bool { ENGAGED.load(std::sync::atomic::Ordering::Relaxed) }
/// `/metrics` rows (leg PX1 / P7SP).
pub fn append(o: &mut String) {
    use std::fmt::Write as _;
    let _ = writeln!(o, "# HELP velogb10_dsv41_xchg_pdl 1 = the dsv41 PDL exchange pair serves this rank's world-2 exchanges (leg PX1, --dsv41-xchg pdl)\n# TYPE velogb10_dsv41_xchg_pdl gauge\nvelogb10_dsv41_xchg_pdl {}", u64::from(engaged()));
    let _ = writeln!(o, "# HELP velogb10_dsv41_xchg_rsag_built_total world-4 wide frames built as reduce-scatter + all-gather (leg PX2, --dsv41-tp4-rsag-min-kib)\n# TYPE velogb10_dsv41_xchg_rsag_built_total counter\nvelogb10_dsv41_xchg_rsag_built_total {}", rsag_built());
    let _ = writeln!(o, "# HELP velogb10_dsv41_xchg_dual_rail_built_total exchanges built as two halves on both RDMA rails (leg TX1, --dsv41-dual-rail)\n# TYPE velogb10_dsv41_xchg_dual_rail_built_total counter\nvelogb10_dsv41_xchg_dual_rail_built_total {}", dual_built());
    let _ = writeln!(o, "# HELP velogb10_dsv41_xchg_pdl_built_total exchanges built on the dsv41 PDL pair (graph replays re-run captured ones without counting)\n# TYPE velogb10_dsv41_xchg_pdl_built_total counter\nvelogb10_dsv41_xchg_pdl_built_total {}", built());
    let _ = writeln!(o, "# HELP velogb10_dsv41_sp_prefill_layers_total prefill layers executed on the sequence-parallel path (leg P7SP, --dsv41-sp-prefill)\n# TYPE velogb10_dsv41_sp_prefill_layers_total counter\nvelogb10_dsv41_sp_prefill_layers_total {}", SP_LAYERS.load(std::sync::atomic::Ordering::Relaxed));
    let _ = writeln!(o, "# HELP velogb10_dsv41_sp_prefill_idx_layers_total prefill indexer layers executed with mirror-zigzag query sharding (leg P7SP, --dsv41-sp-prefill)\n# TYPE velogb10_dsv41_sp_prefill_idx_layers_total counter\nvelogb10_dsv41_sp_prefill_idx_layers_total {}", SP_IDX_LAYERS.load(std::sync::atomic::Ordering::Relaxed));
    let _ = writeln!(o, "# HELP velogb10_dsv41_sp_laga_layers_total prefill layers executed with LAGA BF16 attention-output All-to-All + local whole wo_b (U6a, --dsv41-sp-laga)\n# TYPE velogb10_dsv41_sp_laga_layers_total counter\nvelogb10_dsv41_sp_laga_layers_total {}", SP_LAGA_LAYERS.load(std::sync::atomic::Ordering::Relaxed));
}

/// The trace ring and the dump watcher (`--dsv41-xchg-trace`).
pub struct Trace {
    ring: MappedHost,
}
impl Trace {
    pub fn dev_ptr(&self) -> u64 { self.ring.dev_ptr() }
    fn row(&self, e: u64) -> [u64; TR_STRIDE] {
        let p = self.ring.host_ptr() as *const u64;
        let i = (e as usize % TR_EPOCHS) * TR_STRIDE;
        let mut r = [0u64; TR_STRIDE];
        for (k, v) in r.iter_mut().enumerate() { *v = unsafe { std::ptr::read_volatile(p.add(i + k)) }; }
        r
    }
}

/// GPU %globaltimer minus host CLOCK_MONOTONIC_RAW (ns), best of the samples with the tightest host bracket, and that
/// bracket (ns). A 1-thread kernel writes the timer into a mapped word for 3 ms while this thread samples it.
pub fn calibrate(dev: &Arc<CudaDevice>) -> Result<(i64, u64)> {
    dev.bind_to_thread()?;
    let word = MappedHost::new(dev, 64)?;
    let stream = crate::gpu::fork_blocking_stream(dev);
    let mut out = word.dev_ptr();
    let mut ns = 3_000_000u64;
    let mut p = [&mut out as *mut u64 as *mut std::ffi::c_void, &mut ns as *mut u64 as *mut std::ffi::c_void];
    crate::dsv41::gpu::launch_raw(CLOCK, (1, 1, 1), (1, 1, 1), 0, stream.stream as u64, false, &mut p)?;
    let hp = word.host_ptr() as *const u64;
    let (mut best, mut off) = (u64::MAX, 0i64);
    let t_end = crate::net::now_ns() + 2_500_000;
    loop {
        let t1 = crate::net::now_ns();
        let g = unsafe { std::ptr::read_volatile(hp) };
        let t2 = crate::net::now_ns();
        if g != 0 && t2 - t1 < best { best = t2 - t1; off = g as i64 - ((t1 / 2 + t2 / 2) as i64); }
        if t2 > t_end { break; }
    }
    unsafe { cudarc::driver::result::stream::synchronize(stream.stream) }?;
    ensure!(best != u64::MAX, "dsv41 xchg trace: the clock kernel never wrote");
    Ok((off, best))
}

/// The world-2 exchange pair of one rank (`pair`), or only the trace of the shared pair (`--dsv41-xchg legacy` with
/// `--dsv41-xchg-trace`: the proxy's and the shared collect's stamps are dumped, the dsv41 columns stay 0).
pub struct Xchg {
    arrive: CudaSlice<u32>,   // 2 words: [0] the pair's counter, [1] the dual-rail stage's second group
    rel: CudaSlice<u64>,      // Phase 1A+1D: 4 device-L2 release words ([0..2] single-stage r1/r2, [2..4] tree r1/r2)
    trace: Option<Arc<Trace>>,
    /// the dsv41 PDL pair serves the exchanges (false = trace only)
    pub pair: bool,
    /// 2 = the primary link's pair (PX1), 4/8 = the single-stage ctx's pair (PX2 / Phase A)
    pub world: usize,
    pub rank: usize,
    /// Phase A: zero-CPU-bounce GPU-validated collect (direct NIC tail validation + TP_F_RX_DONE)
    pub gpu_recv: bool,
    /// Phase A: opt-in BF16 wire compression (`--dsv41-wire-bf16 on`)
    pub wire_bf16: bool,
}

impl Xchg {
    /// `pair`: `--dsv41-xchg pdl`; `trace_on`: `--dsv41-xchg-trace` (the ring is allocated here; `start_dump` starts
    /// the watcher once the proxy runs).
    pub fn new(dev: &Arc<CudaDevice>, world: usize, rank: usize, pair: bool, trace_on: bool) -> Result<Self> {
        Self::new_with_opts(dev, world, rank, pair, trace_on, false, false)
    }

    pub fn new_with_opts(dev: &Arc<CudaDevice>, world: usize, rank: usize, pair: bool, trace_on: bool,
                         gpu_recv: bool, wire_bf16: bool) -> Result<Self> {
        Self::new_with_tag(dev, world, rank, pair, trace_on, gpu_recv, wire_bf16, "dsv41")
    }

    pub fn new_with_tag(dev: &Arc<CudaDevice>, world: usize, rank: usize, pair: bool, trace_on: bool,
                        gpu_recv: bool, wire_bf16: bool, tag: &str) -> Result<Self> {
        let mut arrive = dev.alloc_zeros::<u32>(2)?;   // leg TX1: one arrival counter per rail group (the dual-rail stage)
        dev.memset_zeros(&mut arrive)?;   // AGENTS §2.2
        let mut rel = dev.alloc_zeros::<u64>(4)?;      // Phase 1A+1D: [0..2] single-stage r1/r2, [2..4] tree r1/r2
        dev.memset_zeros(&mut rel)?;      // AGENTS §2.2
        let trace = if trace_on {
            Some(Arc::new(Trace { ring: MappedHost::new(dev, TR_EPOCHS * TR_STRIDE * 8).context("xchg trace ring")? }))
        } else { None };
        dev.synchronize()?;
        if pair {
            ENGAGED.store(true, std::sync::atomic::Ordering::Relaxed);
            crate::reprintln!("[{tag} xchg] world-{world} exchanges on the velo-fabric PDL pair (receive: {}, wire: {}, PDL: {}, trace: {})",
                              if gpu_recv { "GPU-validated direct NIC tail (zero CPU bounce)" } else { "CPU bounce (shared cpu_done)" },
                              if wire_bf16 { "BF16 (--dsv41-wire-bf16 on, FP32 canonical sum)" } else { "FP32 (bit-exact)" },
                              if crate::dsv41::gpu::pdl_on() { "on" } else { "off" },
                              if trace_on { "on" } else { "off" });
        } else {
            crate::reprintln!("[{tag} xchg] world-{world} exchanges on the shared pair (--dsv41-xchg legacy); trace {}", if trace_on { "on (proxy + shared collect stamps)" } else { "off" });
        }
        Ok(Xchg { arrive, rel, trace, pair, world, rank, gpu_recv, wire_bf16 })
    }

    /// Stage + collect of `n` floats at `base`, in place, on `stream`. `allow_bf16` permits BF16 wire compression
    /// (for neural residual partials only — NEVER for exact/integer-packed exchanges like `TpXchg` or `head_combine`).
    pub fn exchange(&self, stream: u64, ctx: u64, base: u64, n: usize) -> Result<()> {
        self.exchange_impl(stream, ctx, base, n, false)
    }
    pub fn exchange_act(&self, stream: u64, ctx: u64, base: u64, n: usize) -> Result<()> {
        self.exchange_impl(stream, ctx, base, n, self.wire_bf16)
    }
    fn exchange_impl(&self, stream: u64, ctx: u64, base: u64, n: usize, use_bf16: bool) -> Result<()> {
        ensure!(n > 0 && (n % 4 == 0 || (!self.world.is_power_of_two() && !use_bf16)),
                "dsv41 xchg: {n} floats is not a positive 16 B multiple");
        let pdl = crate::dsv41::gpu::pdl_on();
        let tr = self.trace.as_ref().map_or(0u64, |t| t.dev_ptr());
        if use_bf16 {
            let (stage, collect, cblocks) = match self.world {
                8 => (STAGE4_B16, COLLECT8_B16, collect8_blocks(n)),
                4 => (STAGE4_B16, COLLECT4_B16, collect4_blocks(n)),
                2 => (STAGE_B16, COLLECT_B16, collect_blocks(n)),
                _ => (STAGE4_B16, COLLECTN_B16, collect8_blocks(n)),
            };
            let (mut c, mut src, mut nf, mut arr, mut trv) = (ctx, base, n as u32, *self.arrive.device_ptr(), tr);
            let mut p = [&mut c as *mut u64 as *mut std::ffi::c_void, &mut src as *mut u64 as *mut _, &mut nf as *mut u32 as *mut _,
                         &mut arr as *mut u64 as *mut _, &mut trv as *mut u64 as *mut _];
            crate::dsv41::gpu::launch_raw(stage, (stage_blocks(n), 1, 1), (256, 1, 1), 0, stream, pdl, &mut p).context("dsv41 xchg b16 stage")?;
            let (mut c2, mut buf, mut nn, mut gr, mut trv2, mut relp) = (ctx, base, n as i32, i32::from(self.gpu_recv), tr, *self.rel.device_ptr());
            let mut q = [&mut c2 as *mut u64 as *mut std::ffi::c_void, &mut buf as *mut u64 as *mut _, &mut nn as *mut i32 as *mut _,
                         &mut gr as *mut i32 as *mut _, &mut trv2 as *mut u64 as *mut _, &mut relp as *mut u64 as *mut _];
            crate::dsv41::gpu::launch_raw(collect, (cblocks, 1, 1), (C_THREADS as u32, 1, 1), 0, stream, pdl, &mut q).context("dsv41 xchg b16 collect")?;
            BUILT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(());
        }
        if self.world != 2 && self.world != 4 && self.world != 8 {
            let (mut c, mut src, mut nb, mut arr, mut trv) = (ctx, base, (n * 4) as u32, *self.arrive.device_ptr(), tr);
            let mut p = [&mut c as *mut u64 as *mut std::ffi::c_void, &mut src as *mut u64 as *mut _, &mut nb as *mut u32 as *mut _,
                         &mut arr as *mut u64 as *mut _, &mut trv as *mut u64 as *mut _];
            crate::dsv41::gpu::launch_raw(STAGE4, (stage_blocks(n), 1, 1), (256, 1, 1), 0, stream, pdl, &mut p).context("velo-fabric xchg-n stage")?;
            let (mut c2, mut buf, mut nn, mut gr, mut trv2, mut relp) = (ctx, base, n as i32, i32::from(self.gpu_recv), tr, *self.rel.device_ptr());
            let mut q = [&mut c2 as *mut u64 as *mut std::ffi::c_void, &mut buf as *mut u64 as *mut _, &mut nn as *mut i32 as *mut _,
                         &mut gr as *mut i32 as *mut _, &mut trv2 as *mut u64 as *mut _, &mut relp as *mut u64 as *mut _];
            crate::dsv41::gpu::launch_raw(COLLECTN, (collect8_blocks(n), 1, 1), (C_THREADS as u32, 1, 1), 0, stream, pdl, &mut q)
                .context("velo-fabric xchg-n collect")?;
            BUILT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(());
        }
        let (stage, collect, cblocks) = match (self.world, self.gpu_recv) {
            (8, true)  => (STAGE4, COLLECT8_G, collect8_blocks(n)),
            (8, false) => (STAGE4, COLLECT8, collect8_blocks(n)),
            (4, true)  => (STAGE4, COLLECT4_G, collect4_blocks(n)),
            (4, false) => (STAGE4, COLLECT4, collect4_blocks(n)),
            (_, true)  => (STAGE, COLLECT_G, collect_blocks(n)),
            (_, false) => (STAGE, COLLECT, collect_blocks(n)),
        };
        let (mut c, mut src, mut nb, mut arr, mut trv) = (ctx, base, (n * 4) as u32, *self.arrive.device_ptr(), tr);
        let mut p = [&mut c as *mut u64 as *mut std::ffi::c_void, &mut src as *mut u64 as *mut _, &mut nb as *mut u32 as *mut _,
                     &mut arr as *mut u64 as *mut _, &mut trv as *mut u64 as *mut _];
        crate::dsv41::gpu::launch_raw(stage, (stage_blocks(n), 1, 1), (256, 1, 1), 0, stream, pdl, &mut p).context("dsv41 xchg stage")?;
        let (mut c2, mut buf, mut nn, mut trv2, mut relp) = (ctx, base, n as i32, tr, *self.rel.device_ptr());
        let mut q = [&mut c2 as *mut u64 as *mut std::ffi::c_void, &mut buf as *mut u64 as *mut _, &mut nn as *mut i32 as *mut _,
                     &mut trv2 as *mut u64 as *mut _, &mut relp as *mut u64 as *mut _];
        crate::dsv41::gpu::launch_raw(collect, (cblocks, 1, 1), (C_THREADS as u32, 1, 1), 0, stream, pdl, &mut q)
            .context("dsv41 xchg collect")?;
        BUILT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// leg TX1 (`--dsv41-dual-rail`): split `n` floats in two 16 B-aligned halves across both rails.
    pub fn exchange_dual(&self, stream: u64, ctx: u64, ctx2: u64, base: u64, n: usize) -> Result<()> {
        self.exchange_dual_impl(stream, ctx, ctx2, base, n, false)
    }
    pub fn exchange_dual_act(&self, stream: u64, ctx: u64, ctx2: u64, base: u64, n: usize) -> Result<()> {
        self.exchange_dual_impl(stream, ctx, ctx2, base, n, self.wire_bf16)
    }
    fn exchange_dual_impl(&self, stream: u64, ctx: u64, ctx2: u64, base: u64, n: usize, use_bf16: bool) -> Result<()> {
        let (ha, hb) = dual_split(n).with_context(|| format!("dsv41 dual-rail xchg: {n} floats cannot be split in two 16 B halves"))?;
        let pdl = crate::dsv41::gpu::pdl_on();
        let arr = *self.arrive.device_ptr();
        let rel_base = *self.rel.device_ptr();
        let ptr = |v: &mut u64| v as *mut u64 as *mut std::ffi::c_void;
        if self.world >= 3 && self.world != 4 && self.world != 8 {
            if use_bf16 {
                let (mut c1, mut c2, mut s1, mut s2) = (ctx, ctx2, base, base + (ha * 4) as u64);
                let (mut nf1, mut nf2, mut a, mut tr) = (ha as u32, hb as u32, arr, 0u64);
                let mut p = [ptr(&mut c1), ptr(&mut c2), ptr(&mut s1), ptr(&mut s2), &mut nf1 as *mut u32 as *mut _, &mut nf2 as *mut u32 as *mut _, ptr(&mut a), ptr(&mut tr)];
                crate::dsv41::gpu::launch_raw(STAGE4D_B16, (stage_blocks(hb.max(ha)), 2, 1), (256, 1, 1), 0, stream, pdl, &mut p).context("velo-fabric dual xchg-n b16 stage")?;
                let (mut d1, mut d2, mut b1, mut b2) = (ctx, ctx2, base, base + (ha * 4) as u64);
                let (mut n1, mut n2, mut gr, mut tr2, mut rel2) = (ha as i32, hb as i32, i32::from(self.gpu_recv), 0u64, rel_base);
                let mut q = [ptr(&mut d1), ptr(&mut d2), ptr(&mut b1), ptr(&mut b2), &mut n1 as *mut i32 as *mut _, &mut n2 as *mut i32 as *mut _, &mut gr as *mut i32 as *mut _, ptr(&mut tr2), ptr(&mut rel2)];
                crate::dsv41::gpu::launch_raw(COLLECTND_B16, (collect8_blocks(hb.max(ha)), 2, 1), (C_THREADS as u32, 1, 1), 0, stream, pdl, &mut q).context("velo-fabric dual xchg-n b16 collect")?;
                BUILT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                DUAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(());
            }
            let (mut c1, mut c2, mut s1, mut s2) = (ctx, ctx2, base, base + (ha * 4) as u64);
            let (mut nb1, mut nb2, mut a, mut tr) = ((ha * 4) as u32, (hb * 4) as u32, arr, 0u64);
            let mut p = [ptr(&mut c1), ptr(&mut c2), ptr(&mut s1), ptr(&mut s2), &mut nb1 as *mut u32 as *mut _, &mut nb2 as *mut u32 as *mut _, ptr(&mut a), ptr(&mut tr)];
            crate::dsv41::gpu::launch_raw(STAGE4D, (stage_blocks(hb.max(ha)), 2, 1), (256, 1, 1), 0, stream, pdl, &mut p).context("velo-fabric dual xchg-n stage")?;
            let (mut d1, mut d2, mut b1, mut b2) = (ctx, ctx2, base, base + (ha * 4) as u64);
            let (mut n1, mut n2, mut gr, mut tr2, mut rel2) = (ha as i32, hb as i32, i32::from(self.gpu_recv), 0u64, rel_base);
            let mut q = [ptr(&mut d1), ptr(&mut d2), ptr(&mut b1), ptr(&mut b2), &mut n1 as *mut i32 as *mut _, &mut n2 as *mut i32 as *mut _, &mut gr as *mut i32 as *mut _, ptr(&mut tr2), ptr(&mut rel2)];
            crate::dsv41::gpu::launch_raw(COLLECTND, (collect8_blocks(hb.max(ha)), 2, 1), (C_THREADS as u32, 1, 1), 0, stream, pdl, &mut q).context("velo-fabric dual xchg-n collect")?;
            BUILT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            DUAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(());
        }
        if self.world == 4 || self.world == 8 {
            if use_bf16 {
                let (collect_d, cblocks) = if self.world == 8 {
                    (COLLECT8D_B16, collect8_blocks(hb.max(ha)))
                } else {
                    (COLLECT4D_B16, collect4_blocks(hb.max(ha)))
                };
                let (mut c1, mut c2, mut s1, mut s2) = (ctx, ctx2, base, base + (ha * 4) as u64);
                let (mut nf1, mut nf2, mut a, mut tr) = (ha as u32, hb as u32, arr, 0u64);
                let mut p = [ptr(&mut c1), ptr(&mut c2), ptr(&mut s1), ptr(&mut s2), &mut nf1 as *mut u32 as *mut _, &mut nf2 as *mut u32 as *mut _, ptr(&mut a), ptr(&mut tr)];
                crate::dsv41::gpu::launch_raw(STAGE4D_B16, (stage_blocks(hb.max(ha)), 2, 1), (256, 1, 1), 0, stream, pdl, &mut p).context("dsv41 dual xchg b16 stage (fused)")?;
                let (mut d1, mut d2, mut b1, mut b2) = (ctx, ctx2, base, base + (ha * 4) as u64);
                let (mut n1, mut n2, mut gr, mut tr2, mut rel2) = (ha as i32, hb as i32, i32::from(self.gpu_recv), 0u64, rel_base);
                let mut q = [ptr(&mut d1), ptr(&mut d2), ptr(&mut b1), ptr(&mut b2), &mut n1 as *mut i32 as *mut _, &mut n2 as *mut i32 as *mut _, &mut gr as *mut i32 as *mut _, ptr(&mut tr2), ptr(&mut rel2)];
                crate::dsv41::gpu::launch_raw(collect_d, (cblocks, 2, 1), (C_THREADS as u32, 1, 1), 0, stream, pdl, &mut q).context("dsv41 dual xchg b16 collect (fused)")?;
                BUILT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                DUAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(());
            }
            let (collect_d, cblocks) = match (self.world, self.gpu_recv) {
                (8, true)  => (COLLECT8D_G, collect8_blocks(hb.max(ha))),
                (8, false) => (COLLECT8D, collect8_blocks(hb.max(ha))),
                (_, true)  => (COLLECT4D_G, collect4_blocks(hb.max(ha))),
                (_, false) => (COLLECT4D, collect4_blocks(hb.max(ha))),
            };
            let (mut c1, mut c2, mut s1, mut s2) = (ctx, ctx2, base, base + (ha * 4) as u64);
            let (mut nb1, mut nb2, mut a, mut tr) = ((ha * 4) as u32, (hb * 4) as u32, arr, 0u64);
            let mut p = [ptr(&mut c1), ptr(&mut c2), ptr(&mut s1), ptr(&mut s2), &mut nb1 as *mut u32 as *mut _, &mut nb2 as *mut u32 as *mut _, ptr(&mut a), ptr(&mut tr)];
            crate::dsv41::gpu::launch_raw(STAGE4D, (stage_blocks(hb.max(ha)), 2, 1), (256, 1, 1), 0, stream, pdl, &mut p).context("dsv41 dual xchg stage (fused)")?;
            let (mut d1, mut d2, mut b1, mut b2) = (ctx, ctx2, base, base + (ha * 4) as u64);
            let (mut n1, mut n2, mut tr2, mut rel2) = (ha as i32, hb as i32, 0u64, rel_base);
            let mut q = [ptr(&mut d1), ptr(&mut d2), ptr(&mut b1), ptr(&mut b2), &mut n1 as *mut i32 as *mut _, &mut n2 as *mut i32 as *mut _, ptr(&mut tr2), ptr(&mut rel2)];
            crate::dsv41::gpu::launch_raw(collect_d, (cblocks, 2, 1), (C_THREADS as u32, 1, 1), 0, stream, pdl, &mut q).context("dsv41 dual xchg collect (fused)")?;
            BUILT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            DUAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(());
        }
        // World 2 dual-rail (R3): fused grid.y = 2 stage & collect (removes serial rail-1 -> rail-2 PDL dependency)
        let base_b = base + (ha * 4) as u64;
        if use_bf16 {
            let (mut c1, mut c2, mut s1, mut s2) = (ctx, ctx2, base, base_b);
            let (mut nf1, mut nf2, mut a, mut tr) = (ha as u32, hb as u32, arr, 0u64);
            let mut p = [ptr(&mut c1), ptr(&mut c2), ptr(&mut s1), ptr(&mut s2), &mut nf1 as *mut u32 as *mut _, &mut nf2 as *mut u32 as *mut _, ptr(&mut a), ptr(&mut tr)];
            crate::dsv41::gpu::launch_raw(STAGE2D_B16, (stage_blocks(hb.max(ha)), 2, 1), (256, 1, 1), 0, stream, pdl, &mut p).context("dsv41 dual xchg w2 b16 stage (fused)")?;
            let (mut d1, mut d2, mut b1, mut b2) = (ctx, ctx2, base, base_b);
            let (mut n1, mut n2, mut gr, mut tr2, mut rel2) = (ha as i32, hb as i32, i32::from(self.gpu_recv), 0u64, rel_base);
            let mut q = [ptr(&mut d1), ptr(&mut d2), ptr(&mut b1), ptr(&mut b2), &mut n1 as *mut i32 as *mut _, &mut n2 as *mut i32 as *mut _, &mut gr as *mut i32 as *mut _, ptr(&mut tr2), ptr(&mut rel2)];
            crate::dsv41::gpu::launch_raw(COLLECT2D_B16, (collect_blocks(hb.max(ha)), 2, 1), (C_THREADS as u32, 1, 1), 0, stream, pdl, &mut q).context("dsv41 dual xchg w2 b16 collect (fused)")?;
            BUILT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            DUAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(());
        }
        let collect_d = if self.gpu_recv { COLLECT2D_G } else { COLLECT2D };
        let (mut c1, mut c2, mut s1, mut s2) = (ctx, ctx2, base, base_b);
        let (mut nb1, mut nb2, mut a, mut tr) = ((ha * 4) as u32, (hb * 4) as u32, arr, 0u64);
        let mut p = [ptr(&mut c1), ptr(&mut c2), ptr(&mut s1), ptr(&mut s2), &mut nb1 as *mut u32 as *mut _, &mut nb2 as *mut u32 as *mut _, ptr(&mut a), ptr(&mut tr)];
        crate::dsv41::gpu::launch_raw(STAGE2D, (stage_blocks(hb.max(ha)), 2, 1), (256, 1, 1), 0, stream, pdl, &mut p).context("dsv41 dual xchg w2 stage (fused)")?;
        let (mut d1, mut d2, mut b1, mut b2) = (ctx, ctx2, base, base_b);
        let (mut n1, mut n2, mut tr2, mut rel2) = (ha as i32, hb as i32, 0u64, rel_base);
        let mut q = [ptr(&mut d1), ptr(&mut d2), ptr(&mut b1), ptr(&mut b2), &mut n1 as *mut i32 as *mut _, &mut n2 as *mut i32 as *mut _, ptr(&mut tr2), ptr(&mut rel2)];
        crate::dsv41::gpu::launch_raw(collect_d, (collect_blocks(hb.max(ha)), 2, 1), (C_THREADS as u32, 1, 1), 0, stream, pdl, &mut q).context("dsv41 dual xchg w2 collect (fused)")?;
        BUILT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        DUAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// the stages' arrival counter (the probe drives the kernels itself)
    pub fn arrive_ptr(&self) -> u64 { *self.arrive.device_ptr() }
    /// the collects' per-rail device-L2 release flags (the probe drives the kernels itself)
    pub fn rel_ptr(&self) -> u64 { *self.rel.device_ptr() }

    /// 1-RTT chunked All-to-All across all `world` ranks (`world in 2..=8`, single- or dual-rail).
    /// `src` holds `world` contiguous slices `[0..world)` of `slice_bytes` each (16 B multiple);
    /// `dst` receives `world` contiguous slices `[0..world)` from every peer (bit-exact, zero reduction).
    /// Powers DeepSpeed-Ulysses sequence-parallel attention (Jacobs et al., arXiv:2309.14509).
    #[allow(clippy::too_many_arguments)]
    pub fn all_to_all(
        &self,
        stream: u64,
        ctx1: u64,
        ctx2: Option<u64>,
        src: u64,
        dst: u64,
        slice_bytes: usize,
        slot_cap: usize,
        dual_min_bytes: usize,
    ) -> Result<()> {
        ensure!(slice_bytes > 0 && slice_bytes % 16 == 0, "velo-fabric all_to_all: slice_bytes {slice_bytes} must be a positive 16 B multiple");
        ensure!(src != dst, "velo-fabric all_to_all: src and dst must be distinct buffers");
        let pdl = crate::dsv41::gpu::pdl_on();
        let arr = *self.arrive.device_ptr();
        let rel_base = *self.rel.device_ptr();
        let ptr = |v: &mut u64| v as *mut u64 as *mut std::ffi::c_void;
        let peer_stride = slice_bytes as u64;
        if self.world == 2 {
            let peer = 1 - self.rank;
            // Copy own slice locally once on stream.
            let r = unsafe {
                cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    dst + (self.rank as u64) * peer_stride,
                    src + (self.rank as u64) * peer_stride,
                    slice_bytes,
                    stream as cudarc::driver::sys::CUstream,
                )
            };
            ensure!(r == cudarc::driver::sys::CUresult::CUDA_SUCCESS, "all_to_all w2 self copy: {r:?}");
            let max_ch = (slot_cap.saturating_sub(64)) & !15;
            let mut off = 0usize;
            while off < slice_bytes {
                let rem = slice_bytes - off;
                let use_dual = ctx2.is_some() && rem >= dual_min_bytes.max(32);
                if use_dual {
                    let c2 = ctx2.unwrap();
                    let ch = rem.min(max_ch * 2);
                    let ha = ((ch / 16) / 2) * 16;
                    let hb = ch - ha;
                    let s1 = src + (peer as u64) * peer_stride + off as u64;
                    let s2 = s1 + ha as u64;
                    let d1 = dst + (peer as u64) * peer_stride + off as u64;
                    let d2 = d1 + ha as u64;
                    let (mut ca, mut cb, mut sp1, mut sp2) = (ctx1, c2, s1, s2);
                    let (mut nb1, mut nb2, mut a, mut tr) = (ha as u32, hb as u32, arr, 0u64);
                    let mut p = [ptr(&mut ca), ptr(&mut cb), ptr(&mut sp1), ptr(&mut sp2), &mut nb1 as *mut u32 as *mut _, &mut nb2 as *mut u32 as *mut _, ptr(&mut a), ptr(&mut tr)];
                    crate::dsv41::gpu::launch_raw(STAGE2D, (stage_blocks(hb.max(ha) / 4), 2, 1), (256, 1, 1), 0, stream, pdl, &mut p).context("a2a w2 dual stage (fused)")?;
                    let (mut ca2, mut cb2, mut dp1, mut dp2) = (ctx1, c2, d1, d2);
                    let (mut nb1b, mut nb2b, mut gr, mut rr) = (ha as u32, hb as u32, i32::from(self.gpu_recv), rel_base);
                    let mut q = [ptr(&mut ca2), ptr(&mut cb2), ptr(&mut dp1), ptr(&mut dp2), &mut nb1b as *mut u32 as *mut _, &mut nb2b as *mut u32 as *mut _, &mut gr as *mut i32 as *mut _, ptr(&mut rr)];
                    crate::dsv41::gpu::launch_raw(A2A2D_COLLECT, (collect_blocks(hb.max(ha) / 4), 2, 1), (C_THREADS as u32, 1, 1), 0, stream, pdl, &mut q).context("a2a w2 dual collect (fused)")?;
                    off += ch;
                } else {
                    let ch = rem.min(max_ch);
                    let (mut cc, mut ssp, mut nb, mut aa, mut tr) = (ctx1, src + (peer as u64) * peer_stride + off as u64, ch as u32, arr, 0u64);
                    let mut p = [ptr(&mut cc), ptr(&mut ssp), &mut nb as *mut u32 as *mut _, ptr(&mut aa), ptr(&mut tr)];
                    crate::dsv41::gpu::launch_raw(STAGE, (stage_blocks(ch / 4), 1, 1), (256, 1, 1), 0, stream, pdl, &mut p).context("a2a w2 stage")?;
                    let (mut c2, mut ddp, mut nb2, mut gr, mut rr) = (ctx1, dst + (peer as u64) * peer_stride + off as u64, ch as u32, i32::from(self.gpu_recv), rel_base);
                    let mut q = [ptr(&mut c2), ptr(&mut ddp), &mut nb2 as *mut u32 as *mut _, &mut gr as *mut i32 as *mut _, ptr(&mut rr)];
                    crate::dsv41::gpu::launch_raw(A2A2_COLLECT, (collect_blocks(ch / 4), 1, 1), (C_THREADS as u32, 1, 1), 0, stream, pdl, &mut q).context("a2a w2 collect")?;
                    off += ch;
                }
            }
            BUILT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(());
        }
        // World >= 3 (TP3 / TP4 / TP6 / TP8): 1-RTT per-peer RDMA_WRITE via TP_LEN_FLAG_A2A
        let max_ch = ((slot_cap / self.world).saturating_sub(64)) & !15;
        ensure!(max_ch >= 16, "velo-fabric all_to_all: slot_cap {slot_cap} too small for world {}", self.world);
        let mut off = 0usize;
        while off < slice_bytes {
            let rem = slice_bytes - off;
            let use_dual = ctx2.is_some() && rem * self.world >= dual_min_bytes && rem >= 32;
            if use_dual {
                let c2 = ctx2.unwrap();
                let ch = rem.min(max_ch * 2);
                let ha = ((ch / 16) / 2) * 16;
                let hb = ch - ha;
                let (mut ca, mut cb, mut sbuf) = (ctx1, c2, src);
                let (mut ds1, mut ds2) = (dst + (self.rank as u64) * peer_stride + off as u64, dst + (self.rank as u64) * peer_stride + (off + ha) as u64);
                let (mut ps, mut o1, mut o2, mut ch1, mut ch2, mut a) = (peer_stride, off as u64, (off + ha) as u64, ha as u32, hb as u32, arr);
                let mut p = [ptr(&mut ca), ptr(&mut cb), ptr(&mut sbuf), ptr(&mut ds1), ptr(&mut ds2), ptr(&mut ps), ptr(&mut o1), ptr(&mut o2), &mut ch1 as *mut u32 as *mut _, &mut ch2 as *mut u32 as *mut _, ptr(&mut a)];
                let sb = stage_blocks((hb.max(ha) * self.world) / 4);
                crate::dsv41::gpu::launch_raw(A2AD_STAGE, (sb, 2, 1), (256, 1, 1), 0, stream, pdl, &mut p).context("a2a dual stage")?;
                let (mut ca2, mut cb2, mut dbuf) = (ctx1, c2, dst);
                let (mut ps2, mut co1, mut co2, mut cch1, mut cch2, mut gr, mut rel2) = (peer_stride, off as u64, (off + ha) as u64, ha as u32, hb as u32, i32::from(self.gpu_recv), rel_base);
                let mut q = [ptr(&mut ca2), ptr(&mut cb2), ptr(&mut dbuf), ptr(&mut ps2), ptr(&mut co1), ptr(&mut co2), &mut cch1 as *mut u32 as *mut _, &mut cch2 as *mut u32 as *mut _, &mut gr as *mut i32 as *mut _, ptr(&mut rel2)];
                let cb_blks = collect8_blocks((hb.max(ha) * self.world) / 4);
                crate::dsv41::gpu::launch_raw(A2AD_COLLECT, (cb_blks, 2, 1), (C_THREADS as u32, 1, 1), 0, stream, pdl, &mut q).context("a2a dual collect")?;
                off += ch;
            } else {
                let ch = rem.min(max_ch);
                let (mut ca, mut sbuf, mut ds, mut ps, mut o1, mut ch1, mut a) =
                    (ctx1, src, dst + (self.rank as u64) * peer_stride + off as u64, peer_stride, off as u64, ch as u32, arr);
                let mut p = [ptr(&mut ca), ptr(&mut sbuf), ptr(&mut ds), ptr(&mut ps), ptr(&mut o1), &mut ch1 as *mut u32 as *mut _, ptr(&mut a)];
                crate::dsv41::gpu::launch_raw(A2A_STAGE, (stage_blocks((ch * self.world) / 4), 1, 1), (256, 1, 1), 0, stream, pdl, &mut p).context("a2a stage")?;
                let (mut ca2, mut dbuf, mut ps2, mut co1, mut cch1, mut gr, mut relp) =
                    (ctx1, dst, peer_stride, off as u64, ch as u32, i32::from(self.gpu_recv), rel_base);
                let mut q = [ptr(&mut ca2), ptr(&mut dbuf), ptr(&mut ps2), ptr(&mut co1), &mut cch1 as *mut u32 as *mut _, &mut gr as *mut i32 as *mut _, ptr(&mut relp)];
                crate::dsv41::gpu::launch_raw(A2A_COLLECT, (collect8_blocks((ch * self.world) / 4), 1, 1), (C_THREADS as u32, 1, 1), 0, stream, pdl, &mut q).context("a2a collect")?;
                off += ch;
            }
        }
        BUILT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// 1-RTT Direct Reduce-Scatter for arbitrary `world >= 3` (TP3 / TP6) over variable row blocks `blocks`.
    /// Sums all ranks' slices for `self.rank` in canonical order `0..world` in-place at `base + blocks[rank].0 * row_len * 4`.
    #[allow(clippy::too_many_arguments)]
    pub fn direct_rs_rows(
        &self,
        stream: u64,
        ctx1: u64,
        ctx2: Option<u64>,
        base: u64,
        row_len: usize,
        blocks: &[(usize, usize)],
        slot_cap: usize,
        dual_min_bytes: usize,
    ) -> Result<usize> {
        ensure!(blocks.len() == self.world && self.world >= 3 && self.world <= 8, "direct_rs_rows: bad world {}", self.world);
        let row_bytes = row_len * 4;
        ensure!(row_bytes % 16 == 0, "direct_rs_rows: row_bytes {row_bytes} must be 16 B aligned");
        let mut spn = TpSpan8::default();
        let mut max_bytes = 0usize;
        for (p, &(r0, r1)) in blocks.iter().enumerate() {
            let b = (r1 - r0) * row_bytes;
            spn.off[p] = (r0 * row_bytes) as u64;
            spn.len[p] = b as u32;
            if b > max_bytes { max_bytes = b; }
        }
        if max_bytes == 0 { return Ok(0); }
        let pdl = crate::dsv41::gpu::pdl_on();
        let arr = *self.arrive.device_ptr();
        let rel_base = *self.rel.device_ptr();
        let ptr = |v: &mut u64| v as *mut u64 as *mut std::ffi::c_void;
        let max_ch = ((slot_cap / self.world).saturating_sub(64)) & !15;
        let my_off = spn.off[self.rank];
        let my_len = spn.len[self.rank] as usize;
        let mut sub = 0usize;
        while sub < max_bytes {
            let rem = max_bytes - sub;
            let use_dual = ctx2.is_some() && rem * self.world >= dual_min_bytes && rem >= 32;
            if use_dual {
                let c2 = ctx2.unwrap();
                let ch = rem.min(max_ch * 2);
                let ha = ((ch / 16) / 2) * 16;
                let hb = ch - ha;
                let (mut ca, mut cb, mut sbuf, mut ds1, mut ds2) = (ctx1, c2, base, 0u64, 0u64);
                let (mut sp_arg, mut sub1, mut sub2, mut ch1, mut ch2, mut a) = (spn, sub as u32, (sub + ha) as u32, ha as u32, hb as u32, arr);
                let mut p = [ptr(&mut ca), ptr(&mut cb), ptr(&mut sbuf), ptr(&mut ds1), ptr(&mut ds2), &mut sp_arg as *mut TpSpan8 as *mut _, &mut sub1 as *mut u32 as *mut _, &mut sub2 as *mut u32 as *mut _, &mut ch1 as *mut u32 as *mut _, &mut ch2 as *mut u32 as *mut _, ptr(&mut a)];
                crate::dsv41::gpu::launch_raw(A2AD_VAR_STAGE, (stage_blocks((hb.max(ha) * self.world) / 4), 2, 1), (256, 1, 1), 0, stream, pdl, &mut p).context("direct_rs dual stage")?;
                let n1 = (my_len.saturating_sub(sub).min(ha)) / 4;
                let n2 = (my_len.saturating_sub(sub + ha).min(hb)) / 4;
                let (mut ca2, mut cb2, mut d1, mut d2) = (ctx1, c2, base + my_off + sub as u64, base + my_off + (sub + ha) as u64);
                let (mut nn1, mut nn2, mut w1, mut w2, mut gr, mut rel2) = (n1 as i32, n2 as i32, ((ha + 7) & !7) as u32, ((hb + 7) & !7) as u32, i32::from(self.gpu_recv), rel_base);
                let mut q = [ptr(&mut ca2), ptr(&mut cb2), ptr(&mut d1), ptr(&mut d2), &mut nn1 as *mut i32 as *mut _, &mut nn2 as *mut i32 as *mut _, &mut w1 as *mut u32 as *mut _, &mut w2 as *mut u32 as *mut _, &mut gr as *mut i32 as *mut _, ptr(&mut rel2)];
                crate::dsv41::gpu::launch_raw(RSND_COLLECT, (collect8_blocks(n1.max(n2).max(4)), 2, 1), (C_THREADS as u32, 1, 1), 0, stream, pdl, &mut q).context("direct_rs dual collect")?;
                sub += ch;
            } else {
                let ch = rem.min(max_ch);
                let (mut ca, mut sbuf, mut ds, mut sp_arg, mut sub1, mut ch1, mut a) = (ctx1, base, 0u64, spn, sub as u32, ch as u32, arr);
                let mut p = [ptr(&mut ca), ptr(&mut sbuf), ptr(&mut ds), &mut sp_arg as *mut TpSpan8 as *mut _, &mut sub1 as *mut u32 as *mut _, &mut ch1 as *mut u32 as *mut _, ptr(&mut a)];
                crate::dsv41::gpu::launch_raw(A2A_VAR_STAGE, (stage_blocks((ch * self.world) / 4), 1, 1), (256, 1, 1), 0, stream, pdl, &mut p).context("direct_rs stage")?;
                let n1 = (my_len.saturating_sub(sub).min(ch)) / 4;
                let (mut ca2, mut d1, mut nn1, mut w1, mut gr, mut relp) = (ctx1, base + my_off + sub as u64, n1 as i32, ((ch + 7) & !7) as u32, i32::from(self.gpu_recv), rel_base);
                let mut q = [ptr(&mut ca2), ptr(&mut d1), &mut nn1 as *mut i32 as *mut _, &mut w1 as *mut u32 as *mut _, &mut gr as *mut i32 as *mut _, ptr(&mut relp)];
                crate::dsv41::gpu::launch_raw(RSN_COLLECT, (collect8_blocks(n1.max(4)), 1, 1), (C_THREADS as u32, 1, 1), 0, stream, pdl, &mut q).context("direct_rs collect")?;
                sub += ch;
            }
        }
        BUILT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok((self.world - 1) * max_bytes)
    }

    /// 1-RTT Direct All-Gather for arbitrary `world >= 3` (TP3 / TP6) over variable row blocks `blocks`.
    #[allow(clippy::too_many_arguments)]
    pub fn direct_ag_rows_multi(
        &self,
        stream: u64,
        ctx1: u64,
        ctx2: Option<u64>,
        bufs: &[(u64, usize)],
        blocks: &[(usize, usize)],
        slot_cap: usize,
        dual_min_bytes: usize,
    ) -> Result<usize> {
        ensure!(blocks.len() == self.world && self.world >= 3 && self.world <= 8, "direct_ag_rows_multi: bad world {}", self.world);
        let pdl = crate::dsv41::gpu::pdl_on();
        let arr = *self.arrive.device_ptr();
        let rel_base = *self.rel.device_ptr();
        let ptr = |v: &mut u64| v as *mut u64 as *mut std::ffi::c_void;
        let max_ch = (slot_cap.saturating_sub(64)) & !15;
        let mut total_wire = 0usize;
        for &(base, row_bytes) in bufs {
            ensure!(row_bytes % 16 == 0, "direct_ag_rows_multi: row_bytes {row_bytes} must be 16 B aligned");
            let mut spn = TpSpan8::default();
            let mut max_bytes = 0usize;
            for (p, &(r0, r1)) in blocks.iter().enumerate() {
                let b = (r1 - r0) * row_bytes;
                spn.off[p] = (r0 * row_bytes) as u64;
                spn.len[p] = b as u32;
                if b > max_bytes { max_bytes = b; }
            }
            if max_bytes == 0 { continue; }
            let my_off = spn.off[self.rank];
            let my_len = spn.len[self.rank] as usize;
            let mut sub = 0usize;
            while sub < max_bytes {
                let rem = max_bytes - sub;
                let use_dual = ctx2.is_some() && rem >= dual_min_bytes.max(32);
                if use_dual {
                    let c2 = ctx2.unwrap();
                    let ch = rem.min(max_ch * 2);
                    let ha = ((ch / 16) / 2) * 16;
                    let hb = ch - ha;
                    let av1 = my_len.saturating_sub(sub).min(ha);
                    let av2 = my_len.saturating_sub(sub + ha).min(hb);
                    let (mut ca, mut cb, mut s1, mut s2) = (ctx1, c2, base + my_off + sub as u64, base + my_off + (sub + ha) as u64);
                    let (mut a1, mut a2, mut ch1, mut ch2, mut ar) = (av1 as u32, av2 as u32, ha as u32, hb as u32, arr);
                    let mut p = [ptr(&mut ca), ptr(&mut cb), ptr(&mut s1), ptr(&mut s2), &mut a1 as *mut u32 as *mut _, &mut a2 as *mut u32 as *mut _, &mut ch1 as *mut u32 as *mut _, &mut ch2 as *mut u32 as *mut _, ptr(&mut ar)];
                    crate::dsv41::gpu::launch_raw(AGD_VAR_STAGE, (stage_blocks(hb.max(ha) / 4), 2, 1), (256, 1, 1), 0, stream, pdl, &mut p).context("direct_ag dual stage")?;
                    let (mut ca2, mut cb2, mut dbuf, mut sp_arg) = (ctx1, c2, base, spn);
                    let (mut sub1, mut sub2, mut cch1, mut cch2, mut gr, mut rel2) = (sub as u32, (sub + ha) as u32, ha as u32, hb as u32, i32::from(self.gpu_recv), rel_base);
                    let mut q = [ptr(&mut ca2), ptr(&mut cb2), ptr(&mut dbuf), &mut sp_arg as *mut TpSpan8 as *mut _, &mut sub1 as *mut u32 as *mut _, &mut sub2 as *mut u32 as *mut _, &mut cch1 as *mut u32 as *mut _, &mut cch2 as *mut u32 as *mut _, &mut gr as *mut i32 as *mut _, ptr(&mut rel2)];
                    crate::dsv41::gpu::launch_raw(AGD_VAR_COLLECT, (collect8_blocks((hb.max(ha) * self.world) / 4), 2, 1), (C_THREADS as u32, 1, 1), 0, stream, pdl, &mut q).context("direct_ag dual collect")?;
                    sub += ch;
                } else {
                    let ch = rem.min(max_ch);
                    let av1 = my_len.saturating_sub(sub).min(ch);
                    let (mut ca, mut s1, mut a1, mut ch1, mut ar) = (ctx1, base + my_off + sub as u64, av1 as u32, ch as u32, arr);
                    let mut p = [ptr(&mut ca), ptr(&mut s1), &mut a1 as *mut u32 as *mut _, &mut ch1 as *mut u32 as *mut _, ptr(&mut ar)];
                    crate::dsv41::gpu::launch_raw(AG_VAR_STAGE, (stage_blocks(ch / 4), 1, 1), (256, 1, 1), 0, stream, pdl, &mut p).context("direct_ag stage")?;
                    let (mut ca2, mut dbuf, mut sp_arg, mut sub1, mut cch1, mut gr, mut relp) = (ctx1, base, spn, sub as u32, ch as u32, i32::from(self.gpu_recv), rel_base);
                    let mut q = [ptr(&mut ca2), ptr(&mut dbuf), &mut sp_arg as *mut TpSpan8 as *mut _, &mut sub1 as *mut u32 as *mut _, &mut cch1 as *mut u32 as *mut _, &mut gr as *mut i32 as *mut _, ptr(&mut relp)];
                    crate::dsv41::gpu::launch_raw(AG_VAR_COLLECT, (collect8_blocks((ch * self.world) / 4), 1, 1), (C_THREADS as u32, 1, 1), 0, stream, pdl, &mut q).context("direct_ag collect")?;
                    sub += ch;
                }
            }
            total_wire += (self.world - 1) * max_bytes;
        }
        BUILT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(total_wire)
    }

    /// 2-RTT Direct Wide All-Reduce (RS + AG) for arbitrary `world >= 3` (TP3 / TP6) when `n * 4 > single_cap`.
    #[allow(clippy::too_many_arguments)]
    pub fn direct_wide_allreduce(
        &self,
        stream: u64,
        ctx1: u64,
        ctx2: Option<u64>,
        base: u64,
        n: usize,
        slot_cap: usize,
        dual_min_bytes: usize,
    ) -> Result<()> {
        ensure!(n % 4 == 0 && n > 0, "direct_wide_allreduce: {n} floats must be 16 B aligned");
        let n4 = n / 4; // units of 16 bytes (float4)
        let blocks: Vec<(usize, usize)> = (0..self.world).map(|r| crate::tp_xport::row_block(n4, r, self.world)).collect();
        self.direct_rs_rows(stream, ctx1, ctx2, base, 4, &blocks, slot_cap, dual_min_bytes)?;
        self.direct_ag_rows_multi(stream, ctx1, ctx2, &[(base, 16)], &blocks, slot_cap, dual_min_bytes)?;
        Ok(())
    }

    fn launch(&self, name: &str, blocks: u32, stream: u64, args: &mut [*mut std::ffi::c_void]) -> Result<()> {
        crate::dsv41::gpu::launch_raw(name, (blocks, 1, 1), (C_THREADS as u32, 1, 1), 0, stream, crate::dsv41::gpu::pdl_on(), args)
            .with_context(|| format!("dsv41 xchg {name}"))
    }

    /// leg PX2: the reduce-scatter + all-gather of an `n`-float frame at `base` at world 4 (`rsag_plan`): two tree epochs
    /// on the primary link `pctx`, one all-peers epoch on the single-stage ctx `sctx`. Six PDL launches on `stream`, no
    /// host step. The trace ring is not stamped here (the two contexts' epochs would collide in it).
    pub fn rsag(&self, stream: u64, pctx: u64, sctx: u64, base: u64, n: usize) -> Result<()> {
        ensure!(self.world == 4, "dsv41 rsag: world {}", self.world);
        let g = rsag_plan(n, self.rank).with_context(|| format!("dsv41 rsag: {n} floats is not a 64 B multiple"))?;
        let arr = *self.arrive.device_ptr();
        let rel_single = *self.rel.device_ptr();
        let rel_tree = rel_single + 16;   // dedicated rel[2] for primary tree epoch domain
        let f4 = |off: usize| base + (off * 4) as u64;
        let ptr = |v: &mut u64| v as *mut u64 as *mut std::ffi::c_void;
        for (send, keep, len) in [(g.a_send, g.a_keep, g.half), (g.b_send, g.b_keep, g.quarter)] {
            let (mut c, mut src, mut nb, mut a, mut tr) = (pctx, f4(send), (len * 4) as u32, arr, 0u64);
            let mut p = [ptr(&mut c), ptr(&mut src), &mut nb as *mut u32 as *mut _, ptr(&mut a), ptr(&mut tr)];
            self.launch(TSTAGE, stage_blocks(len), stream, &mut p)?;
            let (mut c2, mut dst, mut nn, mut tr2, mut relp) = (pctx, f4(keep), len as i32, 0u64, rel_tree);
            let mut q = [ptr(&mut c2), ptr(&mut dst), &mut nn as *mut i32 as *mut _, ptr(&mut tr2), ptr(&mut relp)];
            self.launch(TCOLLECT, collect_blocks(len), stream, &mut q)?;
        }
        let (mut c, mut src, mut nb, mut a, mut tr) = (sctx, f4(g.b_keep), (g.quarter * 4) as u32, arr, 0u64);
        let mut p = [ptr(&mut c), ptr(&mut src), &mut nb as *mut u32 as *mut _, ptr(&mut a), ptr(&mut tr)];
        self.launch(STAGE4, stage_blocks(g.quarter), stream, &mut p)?;
        let (mut c2, mut buf, mut nq, mut tr2, mut relp) = (sctx, base, g.quarter as i32, 0u64, rel_single);
        let mut q = [ptr(&mut c2), ptr(&mut buf), &mut nq as *mut i32 as *mut _, ptr(&mut tr2), ptr(&mut relp)];
        self.launch(if self.gpu_recv { GATHER4_G } else { GATHER4 }, collect_blocks(3 * g.quarter), stream, &mut q)?;
        BUILT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        RSAG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Phase 1D: dual-rail reduce-scatter + all-gather at world 4 (`grid.y = rail`).
    pub fn rsag_dual(&self, stream: u64, pctx1: u64, pctx2: u64, sctx1: u64, sctx2: u64, base: u64, n: usize) -> Result<()> {
        ensure!(self.world == 4, "dsv41 rsag_dual: world {}", self.world);
        let (ha, hb) = dual_rsag_split(n).with_context(|| format!("dsv41 rsag_dual: {n} floats cannot split into two 64 B halves"))?;
        let g1 = rsag_plan(ha, self.rank).with_context(|| format!("dsv41 rsag_dual: ha={ha}"))?;
        let g2 = rsag_plan(hb, self.rank).with_context(|| format!("dsv41 rsag_dual: hb={hb}"))?;
        let pdl = crate::dsv41::gpu::pdl_on();
        let arr = *self.arrive.device_ptr();
        let rel_single = *self.rel.device_ptr();
        let rel_tree = rel_single + 16;   // dedicated rel[2..4] for primary tree epoch domain
        let base2 = base + (ha * 4) as u64;
        let ptr = |v: &mut u64| v as *mut u64 as *mut std::ffi::c_void;
        for ((s1, k1, l1), (s2, k2, l2)) in [((g1.a_send, g1.a_keep, g1.half), (g2.a_send, g2.a_keep, g2.half)),
                                             ((g1.b_send, g1.b_keep, g1.quarter), (g2.b_send, g2.b_keep, g2.quarter))] {
            let (mut c1, mut c2, mut src1, mut src2) = (pctx1, pctx2, base + (s1 * 4) as u64, base2 + (s2 * 4) as u64);
            let (mut nb1, mut nb2, mut a, mut tr) = ((l1 * 4) as u32, (l2 * 4) as u32, arr, 0u64);
            let mut p = [ptr(&mut c1), ptr(&mut c2), ptr(&mut src1), ptr(&mut src2), &mut nb1 as *mut u32 as *mut _, &mut nb2 as *mut u32 as *mut _, ptr(&mut a), ptr(&mut tr)];
            crate::dsv41::gpu::launch_raw(TSTAGE_D, (stage_blocks(l1.max(l2)), 2, 1), (256, 1, 1), 0, stream, pdl, &mut p).context("dsv41 dual rsag tree stage")?;
            let (mut d1, mut d2, mut dst1, mut dst2) = (pctx1, pctx2, base + (k1 * 4) as u64, base2 + (k2 * 4) as u64);
            let (mut n1, mut n2, mut tr2, mut relp) = (l1 as i32, l2 as i32, 0u64, rel_tree);
            let mut q = [ptr(&mut d1), ptr(&mut d2), ptr(&mut dst1), ptr(&mut dst2), &mut n1 as *mut i32 as *mut _, &mut n2 as *mut i32 as *mut _, ptr(&mut tr2), ptr(&mut relp)];
            crate::dsv41::gpu::launch_raw(TCOLLECT_D, (collect_blocks(l1.max(l2)), 2, 1), (C_THREADS as u32, 1, 1), 0, stream, pdl, &mut q).context("dsv41 dual rsag tree collect")?;
        }
        let (mut c1, mut c2, mut src1, mut src2) = (sctx1, sctx2, base + (g1.b_keep * 4) as u64, base2 + (g2.b_keep * 4) as u64);
        let (mut nb1, mut nb2, mut a, mut tr) = ((g1.quarter * 4) as u32, (g2.quarter * 4) as u32, arr, 0u64);
        let mut p = [ptr(&mut c1), ptr(&mut c2), ptr(&mut src1), ptr(&mut src2), &mut nb1 as *mut u32 as *mut _, &mut nb2 as *mut u32 as *mut _, ptr(&mut a), ptr(&mut tr)];
        crate::dsv41::gpu::launch_raw(STAGE4D, (stage_blocks(g1.quarter.max(g2.quarter)), 2, 1), (256, 1, 1), 0, stream, pdl, &mut p).context("dsv41 dual rsag gather stage")?;
        let (mut d1, mut d2, mut b1, mut b2) = (sctx1, sctx2, base, base2);
        let (mut nq1, mut nq2, mut tr2, mut relp) = (g1.quarter as i32, g2.quarter as i32, 0u64, rel_single);
        let mut q = [ptr(&mut d1), ptr(&mut d2), ptr(&mut b1), ptr(&mut b2), &mut nq1 as *mut i32 as *mut _, &mut nq2 as *mut i32 as *mut _, ptr(&mut tr2), ptr(&mut relp)];
        crate::dsv41::gpu::launch_raw(if self.gpu_recv { GATHER4D_G } else { GATHER4D }, (collect_blocks(3 * g1.quarter.max(g2.quarter)), 2, 1), (C_THREADS as u32, 1, 1), 0, stream, pdl, &mut q).context("dsv41 dual rsag gather collect")?;
        BUILT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        RSAG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        DUAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Phase A: world-8 single-rail reduce-scatter + all-gather (`rsag8_plan`): 3 tree epochs on `pctx` + 1 all-peers gather on `sctx`.
    pub fn rsag8(&self, stream: u64, pctx: u64, sctx: u64, base: u64, n: usize) -> Result<()> {
        ensure!(self.world == 8, "dsv41 rsag8: world {}", self.world);
        let g = rsag8_plan(n, self.rank).with_context(|| format!("dsv41 rsag8: {n} floats is not a 128 B multiple"))?;
        let arr = *self.arrive.device_ptr();
        let rel_single = *self.rel.device_ptr();
        let rel_tree = rel_single + 16;
        let f4 = |off: usize| base + (off * 4) as u64;
        let ptr = |v: &mut u64| v as *mut u64 as *mut std::ffi::c_void;
        for (send, keep, len) in [(g.a_send, g.a_keep, g.half), (g.b_send, g.b_keep, g.quarter), (g.c_send, g.c_keep, g.eighth)] {
            let (mut c, mut src, mut nb, mut a, mut tr) = (pctx, f4(send), (len * 4) as u32, arr, 0u64);
            let mut p = [ptr(&mut c), ptr(&mut src), &mut nb as *mut u32 as *mut _, ptr(&mut a), ptr(&mut tr)];
            self.launch(TSTAGE, stage_blocks(len), stream, &mut p)?;
            let (mut c2, mut dst, mut nn, mut tr2, mut relp) = (pctx, f4(keep), len as i32, 0u64, rel_tree);
            let mut q = [ptr(&mut c2), ptr(&mut dst), &mut nn as *mut i32 as *mut _, ptr(&mut tr2), ptr(&mut relp)];
            self.launch(TCOLLECT, collect_blocks(len), stream, &mut q)?;
        }
        let (mut c, mut src, mut nb, mut a, mut tr) = (sctx, f4(g.c_keep), (g.eighth * 4) as u32, arr, 0u64);
        let mut p = [ptr(&mut c), ptr(&mut src), &mut nb as *mut u32 as *mut _, ptr(&mut a), ptr(&mut tr)];
        self.launch(STAGE4, stage_blocks(g.eighth), stream, &mut p)?;
        let (mut c2, mut buf, mut nq, mut gr, mut tr2, mut relp) = (sctx, base, g.eighth as i32, i32::from(self.gpu_recv), 0u64, rel_single);
        let mut q = [ptr(&mut c2), ptr(&mut buf), &mut nq as *mut i32 as *mut _, &mut gr as *mut i32 as *mut _, ptr(&mut tr2), ptr(&mut relp)];
        self.launch(GATHER8, collect_blocks(7 * g.eighth), stream, &mut q)?;
        BUILT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        RSAG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Phase A: world-8 dual-rail reduce-scatter + all-gather (`grid.y = rail`): 0.875F wire bytes per rail vs 3.5F
    /// for dual-rail single-stage (4x wire bandwidth reduction), bit-exact canonical world-8 tree sum.
    pub fn rsag8_dual(&self, stream: u64, pctx1: u64, pctx2: u64, sctx1: u64, sctx2: u64, base: u64, n: usize) -> Result<()> {
        ensure!(self.world == 8, "dsv41 rsag8_dual: world {}", self.world);
        let (ha, hb) = dual_rsag8_split(n).with_context(|| format!("dsv41 rsag8_dual: {n} floats cannot split into two 128 B halves"))?;
        let g1 = rsag8_plan(ha, self.rank).with_context(|| format!("dsv41 rsag8_dual: ha={ha}"))?;
        let g2 = rsag8_plan(hb, self.rank).with_context(|| format!("dsv41 rsag8_dual: hb={hb}"))?;
        let pdl = crate::dsv41::gpu::pdl_on();
        let arr = *self.arrive.device_ptr();
        let rel_single = *self.rel.device_ptr();
        let rel_tree = rel_single + 16;
        let base2 = base + (ha * 4) as u64;
        let ptr = |v: &mut u64| v as *mut u64 as *mut std::ffi::c_void;
        for ((s1, k1, l1), (s2, k2, l2)) in [
            ((g1.a_send, g1.a_keep, g1.half), (g2.a_send, g2.a_keep, g2.half)),
            ((g1.b_send, g1.b_keep, g1.quarter), (g2.b_send, g2.b_keep, g2.quarter)),
            ((g1.c_send, g1.c_keep, g1.eighth), (g2.c_send, g2.c_keep, g2.eighth)),
        ] {
            let (mut c1, mut c2, mut src1, mut src2) = (pctx1, pctx2, base + (s1 * 4) as u64, base2 + (s2 * 4) as u64);
            let (mut nb1, mut nb2, mut a, mut tr) = ((l1 * 4) as u32, (l2 * 4) as u32, arr, 0u64);
            let mut p = [ptr(&mut c1), ptr(&mut c2), ptr(&mut src1), ptr(&mut src2), &mut nb1 as *mut u32 as *mut _, &mut nb2 as *mut u32 as *mut _, ptr(&mut a), ptr(&mut tr)];
            crate::dsv41::gpu::launch_raw(TSTAGE_D, (stage_blocks(l1.max(l2)), 2, 1), (256, 1, 1), 0, stream, pdl, &mut p).context("dsv41 dual rsag8 tree stage")?;
            let (mut d1, mut d2, mut dst1, mut dst2) = (pctx1, pctx2, base + (k1 * 4) as u64, base2 + (k2 * 4) as u64);
            let (mut n1, mut n2, mut tr2, mut relp) = (l1 as i32, l2 as i32, 0u64, rel_tree);
            let mut q = [ptr(&mut d1), ptr(&mut d2), ptr(&mut dst1), ptr(&mut dst2), &mut n1 as *mut i32 as *mut _, &mut n2 as *mut i32 as *mut _, ptr(&mut tr2), ptr(&mut relp)];
            crate::dsv41::gpu::launch_raw(TCOLLECT_D, (collect_blocks(l1.max(l2)), 2, 1), (C_THREADS as u32, 1, 1), 0, stream, pdl, &mut q).context("dsv41 dual rsag8 tree collect")?;
        }
        let (mut c1, mut c2, mut src1, mut src2) = (sctx1, sctx2, base + (g1.c_keep * 4) as u64, base2 + (g2.c_keep * 4) as u64);
        let (mut nb1, mut nb2, mut a, mut tr) = ((g1.eighth * 4) as u32, (g2.eighth * 4) as u32, arr, 0u64);
        let mut p = [ptr(&mut c1), ptr(&mut c2), ptr(&mut src1), ptr(&mut src2), &mut nb1 as *mut u32 as *mut _, &mut nb2 as *mut u32 as *mut _, ptr(&mut a), ptr(&mut tr)];
        crate::dsv41::gpu::launch_raw(STAGE4D, (stage_blocks(g1.eighth.max(g2.eighth)), 2, 1), (256, 1, 1), 0, stream, pdl, &mut p).context("dsv41 dual rsag8 gather stage")?;
        let (mut d1, mut d2, mut b1, mut b2) = (sctx1, sctx2, base, base2);
        let (mut nq1, mut nq2, mut gr, mut tr2, mut relp) = (g1.eighth as i32, g2.eighth as i32, i32::from(self.gpu_recv), 0u64, rel_single);
        let mut q = [ptr(&mut d1), ptr(&mut d2), ptr(&mut b1), ptr(&mut b2), &mut nq1 as *mut i32 as *mut _, &mut nq2 as *mut i32 as *mut _, &mut gr as *mut i32 as *mut _, ptr(&mut tr2), ptr(&mut relp)];
        crate::dsv41::gpu::launch_raw(GATHER8D, (collect_blocks(7 * g1.eighth.max(g2.eighth)), 2, 1), (C_THREADS as u32, 1, 1), 0, stream, pdl, &mut q).context("dsv41 dual rsag8 gather collect")?;
        BUILT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        RSAG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        DUAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Start the trace dump watcher (after the proxy runs: `net::trace_data` reads the registered link).
    pub fn start_dump(&self, dev: &Arc<CudaDevice>, dir: &str, rank: usize, ctx_addr: usize) -> Result<()> {
        let Some(t) = self.trace.clone() else { return Ok(()) };
        std::fs::create_dir_all(dir).with_context(|| format!("--dsv41-xchg-trace {dir}"))?;
        let (dev, dir, world) = (dev.clone(), dir.to_string(), self.world);
        let (off0, br0) = calibrate(&dev)?;
        crate::reprintln!("[dsv41 xchg] trace on: rank {rank}, ring {TR_EPOCHS} epochs, dumps to {dir}; clock gpu - host {off0} ns (bracket {br0} ns)");
        std::thread::Builder::new().name("d41-xtrace".into()).spawn(move || {
            let (mut last, mut since, mut dumped, mut n) = (0u64, crate::net::now_ns(), 0u64, 0u32);
            loop {
                std::thread::sleep(std::time::Duration::from_millis(200));
                let e = crate::net::ctx_device_epoch(ctx_addr);
                let now = crate::net::now_ns();
                if e != last { last = e; since = now; continue; }
                if e > dumped + 64 && now - since > 1_500_000_000 {
                    let cal = calibrate(&dev).unwrap_or((0, u64::MAX));
                    let path = format!("{dir}/xtr_r{rank}_{n:03}.csv");
                    if let Err(err) = dump(&t, &path, rank, e, (off0, br0), cal, world == 2) { crate::reprintln!("[dsv41 xchg] trace dump {path}: {err:#}"); }
                    else { crate::reprintln!("[dsv41 xchg] trace dump {path} (epoch {e})"); }
                    dumped = e; n += 1;
                }
            }
        })?;
        Ok(())
    }
}

fn dump(t: &Trace, path: &str, rank: usize, dev_epoch: u64, cal0: (i64, u64), cal1: (i64, u64), proxy_cols: bool) -> Result<()> {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    // the proxy's stamps are the registered (primary) link's: indexed by ITS epochs, so only at world 2 (at world 4 the
    // pair runs on the single-stage ctx; its columns are written as 0)
    let td = if proxy_cols { crate::net::trace_data() } else { None };
    let (posted, retired, released, tail) = td.map_or((0, 0, 0, 0), |d| d.2);
    writeln!(f, "# rank {rank} dev_epoch {dev_epoch} cal_attach_off_ns {} cal_attach_bracket_ns {} cal_dump_off_ns {} cal_dump_bracket_ns {} \
                 posted {posted} retired {retired} released {released} tail_fires {tail} proxy_parks {} proxy_parked_s {:.3} mode {:?}",
             cal0.0, cal0.1, cal1.0, cal1.1, crate::net::proxy_parks(), crate::net::proxy_parked_secs(), mode())?;
    write!(f, "epoch")?;
    for n in TR_NAMES { write!(f, ",{n}")?; }
    writeln!(f, ",g_k1in,g_k1out,g_k2in,g_k2go,h_ready,h_posted,h_cqe,h_peerseen,h_released")?;
    let n = (dev_epoch as usize).min(TR_EPOCHS - 1).min(crate::net::GTS_EPOCHS - 1);
    for e in dev_epoch + 1 - n as u64..=dev_epoch {
        write!(f, "{e}")?;
        for v in t.row(e) { write!(f, ",{v}")?; }
        match td {
            Some((g, c, ..)) => {
                let gi = (e as usize % crate::net::GTS_EPOCHS) * crate::net::GTS_STRIDE;
                let ci = (e as usize % crate::net::GTS_EPOCHS) * crate::net::CTS_STRIDE;
                for k in 0..4 { write!(f, ",{}", g[gi + k])?; }
                for k in 0..5 { write!(f, ",{}", c[ci + k])?; }
            }
            None => { for _ in 0..9 { write!(f, ",0")?; } }
        }
        writeln!(f)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_defaults_to_pdl_and_parses_legacy() {
        assert_eq!(mode_of(None), Mode::Pdl);
        assert_eq!(mode_of(Some("pdl")), Mode::Pdl);
        assert_eq!(mode_of(Some(" Legacy ")), Mode::Legacy);
    }

    #[test]
    fn collect_grid_covers_every_float4_in_one_pass_up_to_32_blocks() {
        // 1 row frame (5,120 + 32 floats): 1,288 float4s -> 2 blocks of 256 x 4
        assert_eq!(collect_blocks(5152), 2);
        // 6 rows: 30,912 floats -> 7,728 float4s -> 8 blocks
        assert_eq!(collect_blocks(6 * 5152), 8);
        assert_eq!(collect_blocks(4), 1);
        // a full ring slot caps at 32 blocks (grid-stride covers the rest)
        assert_eq!(collect_blocks(262_000), 32);
        for n in [4usize, 5152, 30_912, 123_648, 262_000] {
            let b = collect_blocks(n) as usize;
            assert!(b * C_THREADS * C_PER >= (n / 4).min(32 * C_THREADS * C_PER), "{n}");
        }
    }

    #[test]
    fn metrics_rows_name_the_gauge_and_the_counter() {
        let mut o = String::new();
        append(&mut o);
        assert!(o.contains("\nvelogb10_dsv41_xchg_pdl ") && o.contains("\nvelogb10_dsv41_xchg_pdl_built_total "), "{o}");
        let oc = std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/owner_check.py")).unwrap();
        assert!(oc.contains("velogb10_dsv41_xchg_pdl_built_total"));
    }

    #[test]
    fn world4_routing_puts_every_decode_frame_on_one_stage() {
        use super::super::rdma::{route_cap, thresholds, Path, PIPE_MIN_BYTES, SINGLE_CAP_BYTES, SINGLE_CAP_BYTES_XCHG4};
        use super::super::xprobe::{ENGRAM_ROW_FLOATS, ROW_FLOATS};
        let frame = |rows: usize| rows * ROW_FLOATS * 4;
        let engram = |rows: usize| rows * (ROW_FLOATS + ENGRAM_ROW_FLOATS) * 4;
        // the dsv41 world-4 pair, no flags (leg P4): pipe threshold 384 KiB, single-stage bound 768 KiB
        let (pm, cap) = thresholds(4, false, true, None, None, None);
        assert_eq!((pm, cap), (super::super::rdma::PIPE_MIN_BYTES_XCHG4, SINGLE_CAP_BYTES_XCHG4));
        let r = |b: usize| route_cap(b, 4, true, true, pm, cap);
        for rows in [1, 2, 3, 6, 12] { assert_eq!(r(frame(rows)), Path::Single, "{rows} rows ({} B)", frame(rows)); }
        assert_eq!(r(frame(24)), Path::Pipe, "24 rows (494,592 B): above 384 KiB (leg P4); reduce-scatter takes it first from 192 KiB (--dsv41-tp4-rsag-min-kib)");
        assert_eq!(r(engram(1)), Path::Single);           // 123,008 B
        assert_eq!(r(engram(6)), Path::Pipe);             // 738,048 B (reduce-scatter + all-gather takes it first)
        assert_eq!(r(engram(12)), Path::Pipe);            // 1.48 MB
        assert_eq!(r(engram(24)), Path::Pipe);            // 2.95 MB
        // ... and before routing, the wide ones take reduce-scatter + all-gather (default 192 KiB)
        use super::super::rdma::{use_rsag, RSAG_MIN_BYTES};
        let rs = |floats: usize| use_rsag(floats, RSAG_MIN_BYTES, cap);
        for rows in [1, 2, 3, 6] { assert!(!rs(rows * ROW_FLOATS), "{rows} rows"); }
        // 600 KiB (leg TX1): the 12- and 24-row frames (247 / 495 KB) stay one-stage / pipe, only >= 600 KiB frames split
        for rows in [12, 24] { assert!(!rs(rows * ROW_FLOATS), "{rows} rows"); }
        assert!(!rs(ROW_FLOATS + ENGRAM_ROW_FLOATS));
        assert!(rs(6 * (ROW_FLOATS + ENGRAM_ROW_FLOATS)) && rs(12 * (ROW_FLOATS + ENGRAM_ROW_FLOATS)));
        assert!(!rs(24 * (ROW_FLOATS + ENGRAM_ROW_FLOATS)));   // its half (1.48 MB) exceeds a primary ring slot: the pipe
        assert!(!use_rsag(24 * ROW_FLOATS, 0, cap));           // 0 = never
        // the shared pair at world 4 keeps today's routing: 32 KiB pipe, 256 KiB bound
        let (pm, cap) = thresholds(4, false, false, None, None, None);
        assert_eq!((pm, cap), (PIPE_MIN_BYTES, SINGLE_CAP_BYTES));
        assert_eq!(route_cap(frame(1), 4, true, true, pm, cap), Path::Single);
        assert_eq!(route_cap(frame(6), 4, true, true, pm, cap), Path::Pipe);
        // the TP4 flag wins over the TP2 one at world 4; the TP2 one alone still applies (back-compat)
        assert_eq!(thresholds(4, false, true, Some(32), Some(256), None).0, 256 * 1024);
        assert_eq!(thresholds(4, false, true, Some(32), None, None).0, 32 * 1024);
        assert_eq!(thresholds(4, false, true, None, None, Some(1024)), (384 * 1024, 1024 * 1024));   // leg P4: 384 KiB under a wide bound
        assert_eq!(thresholds(4, false, true, None, None, Some(256)), (256 * 1024, 256 * 1024), "... and the bound itself under a narrow one");
        // world 2 is untouched by the TP4 flags
        assert_eq!(thresholds(2, true, false, None, Some(7), Some(1024)).0, super::super::rdma::PIPE_MIN_BYTES_XCHG);
        assert_eq!(thresholds(2, false, false, None, Some(7), None).0, PIPE_MIN_BYTES);
    }

    #[test]
    fn rsag_plan_gives_every_rank_its_own_quarter_and_the_pair_sums_in_order() {
        let n = 24 * 5152;
        let plans: Vec<RsAg> = (0..4).map(|r| rsag_plan(n, r).unwrap()).collect();
        for (r, g) in plans.iter().enumerate() {
            assert_eq!(g.b_keep, r * n / 4, "rank {r} owns quarter r");
            // epoch A: the pair (r, r ^ 2) splits the frame in halves; each sends what the other keeps
            let o = &plans[r ^ 2];
            assert_eq!(g.a_send, o.a_keep);
            assert_ne!(g.a_keep, o.a_keep);
            // epoch B: inside the kept half, the pair (r, r ^ 1) splits in quarters
            let o = &plans[r ^ 1];
            assert_eq!(g.a_keep, o.a_keep);
            assert_eq!(g.b_send, o.b_keep);
            assert!(g.b_keep >= g.a_keep && g.b_keep < g.a_keep + g.half);
        }
        // a host model of the three epochs reproduces (p0 + p2) + (p1 + p3) on every rank, bit for bit
        let parts: Vec<Vec<f32>> = (0..4).map(|q| (0..n).map(|i| (((i * 7 + q * 13) % 97) as f32 - 48.0) * 1.0e-3 + (q as f32) * 1.0e3).collect()).collect();
        let mut buf = parts.clone();
        for (lo_bit, len_of, keep_of, send_of) in [(2usize, 0usize, 0usize, 0usize), (1, 1, 1, 1)] {
            let _ = (len_of, keep_of, send_of);
            let snap = buf.clone();
            for r in 0..4 {
                let g = rsag_plan(n, r).unwrap();
                let (keep, send, len) = if lo_bit == 2 { (g.a_keep, g.a_send, g.half) } else { (g.b_keep, g.b_send, g.quarter) };
                let peer = r ^ lo_bit;
                let gp = rsag_plan(n, peer).unwrap();
                let psend = if lo_bit == 2 { gp.a_send } else { gp.b_send };
                assert_eq!(psend, keep);
                let _ = send;
                let lower = r & lo_bit == 0;
                for i in 0..len {
                    let (a, b) = (snap[r][keep + i], snap[peer][psend + i]);
                    buf[r][keep + i] = if lower { a + b } else { b + a };
                }
            }
        }
        let want = super::super::xprobe::reference(&parts);
        for r in 0..4 {
            let q = n / 4;
            for p in 0..4 { let src = buf[p][p * q..(p + 1) * q].to_vec(); buf[r][p * q..(p + 1) * q].copy_from_slice(&src); }
            assert!(buf[r].iter().zip(&want).all(|(x, y)| x.to_bits() == y.to_bits()), "rank {r}");
        }
        // the bytes one rank sends: 3 F on the one-stage path, 1.5 F with reduce-scatter + all-gather
        assert_eq!(wire_bytes_w4(n), (3 * n * 4, 3 * n * 2));
        assert!(rsag_plan(5153, 0).is_none());
    }

    #[test]
    fn dual_halves_are_16_byte_aligned_disjoint_and_cover_the_frame() {
        for n in [8usize, 12, 5152, 2 * 5152, 6 * 5152, 24 * 5152, 24 * 5152 + 32, 262_000] {
            let (a, b) = dual_split(n).unwrap();
            assert_eq!(a + b, n);
            assert!(a % 4 == 0 && b % 4 == 0 && a > 0 && b > 0, "{n}: {a} + {b}");
            assert!(a.abs_diff(b) <= 4, "{n}: halves {a} / {b}");
        }
        assert_eq!(dual_split(4), None);
        assert_eq!(dual_split(6), None);
    }

    #[test]
    fn world4_collect_grid_covers_the_frame() {
        assert_eq!(collect4_blocks(5152), 3);            // 1,288 float4s / (256 x 2)
        assert_eq!(collect4_blocks(6 * 5152), 16);
        assert_eq!(collect4_blocks(24 * 5152), 32);
        for n in [4usize, 5152, 30_912, 123_648, 184_512] {
            let b = collect4_blocks(n) as usize;
            assert!(b * C_THREADS * C4_PER >= (n / 4).min(32 * C_THREADS * C4_PER), "{n}");
        }
    }

    #[test]
    fn world4_kernels_keep_the_shared_association() {
        let k = include_str!("../../../kernels/gpu_dsv41/07_xchg.cuh");
        assert!(k.contains("(a0.x + a2.x) + (a1.x + a3.x)") && k.contains("(a0.w + a2.w) + (a1.w + a3.w)"));
        assert!(k.contains("extern \"C\" __global__ void d41x_xc4_stage") && k.contains("extern \"C\" __global__ void d41x_xc4_collect"));
    }

    #[test]
    fn stage_grid_is_the_shared_k1m_grid() {
        for n in [4usize, 5152, 30_912, 262_000] { assert_eq!(stage_blocks(n), super::super::rdma::k1_mb_blocks(n)); }
    }

    #[test]
    fn dual_rsag_split_produces_two_64b_aligned_rsag_halves() {
        use super::super::rdma::{use_rsag_dual, RSAG_MIN_BYTES, SINGLE_CAP_BYTES_XCHG4};
        use super::super::xprobe::{ENGRAM_ROW_FLOATS, ROW_FLOATS};
        for n in [32usize, 48, 6 * ROW_FLOATS, 12 * ROW_FLOATS, 24 * ROW_FLOATS,
                  6 * (ROW_FLOATS + ENGRAM_ROW_FLOATS), 12 * (ROW_FLOATS + ENGRAM_ROW_FLOATS),
                  24 * (ROW_FLOATS + ENGRAM_ROW_FLOATS)] {
            let (ha, hb) = dual_rsag_split(n).unwrap();
            assert_eq!(ha + hb, n);
            assert!(ha % 16 == 0 && hb % 16 == 0 && ha >= 16 && hb >= 16);
            assert!(rsag_plan(ha, 0).is_some() && rsag_plan(hb, 3).is_some());
        }
        assert_eq!(dual_rsag_split(16), None);
        assert_eq!(dual_rsag_split(24), None);
        // 24-row Engram (2.95 MB) now fits dual-rail RS+AG!
        assert!(use_rsag_dual(24 * (ROW_FLOATS + ENGRAM_ROW_FLOATS), RSAG_MIN_BYTES, SINGLE_CAP_BYTES_XCHG4));
        assert!(use_rsag_dual(6 * (ROW_FLOATS + ENGRAM_ROW_FLOATS), RSAG_MIN_BYTES, SINGLE_CAP_BYTES_XCHG4));
        assert!(!use_rsag_dual(24 * ROW_FLOATS, 0, SINGLE_CAP_BYTES_XCHG4));
    }

    #[test]
    fn trace_columns_match_the_kernel_slots() {
        assert_eq!(TR_NAMES.len(), TR_STRIDE);
        let k = include_str!("../../../kernels/gpu_dsv41/07_xchg.cuh");
        assert!(k.contains("#define D41X_XC_TR_EPOCHS 4096") && k.contains("#define D41X_XC_TR_STRIDE 8"));
        for (i, n) in ["S_IN", "S_GO", "S_DB", "C_IN", "C_GO", "C_SEEN", "C_END0", "C_ENDL"].iter().enumerate() {
            assert!(k.contains(&format!("#define D41X_XC_TR_{n} {i}")), "{n}");
        }
    }

    #[test]
    fn rsag8_plan_matches_canonical_world8_tree_bitwise() {
        let n = 6 * 5152; // 30,912 floats, % 32 == 0
        let parts: Vec<Vec<f32>> = (0..8)
            .map(|r| (0..n).map(|i| (((i * 11 + r * 17) % 101) as f32 - 50.0) * 1.0e-3 + (r as f32) * 1.0e2).collect())
            .collect();
        let mut buf = parts.clone();
        for (stage, lo_bit) in [(0usize, 2usize), (1, 4), (2, 1)] {
            let snap = buf.clone();
            for r in 0..8 {
                let g = rsag8_plan(n, r).unwrap();
                let (keep, send, len) = match stage {
                    0 => (g.a_keep, g.a_send, g.half),
                    1 => (g.b_keep, g.b_send, g.quarter),
                    _ => (g.c_keep, g.c_send, g.eighth),
                };
                let peer = r ^ lo_bit;
                let gp = rsag8_plan(n, peer).unwrap();
                let psend = match stage { 0 => gp.a_send, 1 => gp.b_send, _ => gp.c_send };
                assert_eq!(psend, keep, "stage {stage} rank {r} peer {peer}");
                let _ = send;
                let lower = (r & lo_bit) == 0;
                for i in 0..len {
                    let (a, b) = (snap[r][keep + i], snap[peer][psend + i]);
                    buf[r][keep + i] = if lower { a + b } else { b + a };
                }
            }
        }
        let want = crate::tp_xport::rd_tree_reference(8, 0, &parts).unwrap();
        let e8 = n / 8;
        let final_eighths: Vec<Vec<f32>> = (0..8).map(|p| {
            let off = slot_w8(p) * e8;
            buf[p][off..off + e8].to_vec()
        }).collect();
        for r in 0..8 {
            for p in 0..8 {
                let off = slot_w8(p) * e8;
                buf[r][off..off + e8].copy_from_slice(&final_eighths[p]);
            }
            assert!(buf[r].iter().zip(&want).all(|(x, y)| x.to_bits() == y.to_bits()), "rank {r} rsag8 != canonical tree");
        }
        assert_eq!(wire_bytes_w8(n), (7 * n * 4, 7 * n)); // 7F vs 1.75F = 4x wire reduction
        let (ha, hb) = dual_rsag8_split(n).unwrap();
        assert_eq!(ha + hb, n);
        assert!(ha % 32 == 0 && hb % 32 == 0 && rsag8_plan(ha, 7).is_some() && rsag8_plan(hb, 0).is_some());
    }

    #[test]
    fn a2a_tag_encoding_and_arbitrary_w_chunk_bounds_hold() {
        // Mirror native/tp_doorbell.h TP_LEN_FLAG_A2A / TP_LEN_TAG bit layout.
        const TP_LEN_FLAG_A2A: u64 = 1u64 << 19;
        const TP_TAIL_BYTES: usize = 8;
        const TP_CL: usize = 64;
        let encode = |e: u64, w: u64, a2a: bool| (e << 20) | (w >> 3) | if a2a { TP_LEN_FLAG_A2A } else { 0 };
        let dec_epoch = |t: u64| t >> 20;
        let dec_bytes = |t: u64| (t & ((1u64 << 19) - 1)) << 3;
        let dec_a2a = |t: u64| (t & TP_LEN_FLAG_A2A) != 0;
        for &e in &[1u64, 4095, 4096, 1_000_000, (1u64 << 43) - 1] {
            for &wire in &[8u64, 64, 4096, 262_080, 786_432, (1u64 << 22) - 8] {
                for a2a in [false, true] {
                    let t = encode(e, wire, a2a);
                    assert_eq!(dec_epoch(t), e);
                    assert_eq!(dec_bytes(t), wire);
                    assert_eq!(dec_a2a(t), a2a);
                    // Commit clears A2A flag without touching epoch or wire bytes.
                    let cm = t & !TP_LEN_FLAG_A2A;
                    assert_eq!(dec_epoch(cm), e);
                    assert_eq!(dec_bytes(cm), wire);
                    assert!(!dec_a2a(cm));
                }
            }
        }
        // Verify A2A / direct_rs_rows max_ch formula fits slot_stride across all worlds & slot caps.
        for &cap in &[256 * 1024usize, 384 * 1024, 768 * 1024, 1024 * 1024] {
            let slot_stride = (cap + TP_TAIL_BYTES + TP_CL - 1) & !(TP_CL - 1);
            for &w in &[3usize, 4, 6, 8] {
                let max_ch = ((cap / w).saturating_sub(64)) & !15;
                assert!(max_ch >= 16 && max_ch % 16 == 0, "w={w} cap={cap} max_ch={max_ch}");
                let wire = (max_ch + 7) & !7;
                let a2a_stride = (wire + TP_TAIL_BYTES + TP_CL - 1) & !(TP_CL - 1);
                assert!(w * a2a_stride <= slot_stride, "w={w} cap={cap}: {} > {slot_stride}", w * a2a_stride);
            }
        }
        // Verify TpSpan8 layout size/alignment matches CUDA struct TpSpan8 in 07_xchg.cuh.
        assert_eq!(std::mem::size_of::<TpSpan8>(), 8 * 8 + 8 * 4);
        assert_eq!(std::mem::align_of::<TpSpan8>(), 8);
    }
}
