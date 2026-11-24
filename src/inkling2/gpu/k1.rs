//! S2a (leg K1): host side of the non-MoE kernels — weight layouts made at load, argument structs
//! (layouts = the C structs in kernels/gpu_inkling2/10_attn.cuh / 30_attnres.cuh) and launch wrappers.
//!
//! Buffer conventions (one sequence, M rows at positions p0..p0+M-1):
//!   * residual stream, AttnRes values / outputs, MessageGate in/out: fp32 [M][2304];
//!   * every linear's input: bf16 [M][K] (written by the norm kernels; the bf16 reference casts there too);
//!   * projection outputs: fp32 [M][ldy];
//!   * K / V cache per logical slot: f16 [cap][8][128], cap a power of two, row = pos & (cap - 1);
//!   * conv input rings per (slot, conv): fp32 [32][C], row = pos & 31 (commit = advance pos, 20_conv.cuh).

use super::{dp, Ink2Kernels, KArgs};
use anyhow::{ensure, Result};
use cudarc::driver::DeviceRepr;

pub const HD: usize = 128;
pub const NH: usize = 24;
pub const NKV: usize = 8;
pub const DREL: usize = 16;
pub const RING: usize = 32;
/// MessageGate `down` (K = 4608 = 144 groups) at decode: 18 cross-CTA k-slices x 8 warps = 1 group per warp.
pub const MSG_ZS: usize = 18;
/// Largest dynamic smem `ink_attn` takes (16 rows): 4*(48*144 + 16*80) + 64*(130+128)*2.
pub const ATTN_SMEM_MAX: usize = attn_smem(16);
/// `ink_attn_tc` dynamic smem: K + V (64 x 136 f16 each) + proj (80 x 24 bf16) + bias (3 x 16 x 80 fp32)
/// + Q lo (3 x 16 x 136 f16).
pub const ATTN_TC_SMEM: usize = 2 * 64 * 136 * 2 + 80 * 24 * 2 + 3 * 16 * 80 * 4 + 3 * 16 * 136 * 2;
/// `ink_attn_tc2` (S4, 2 token tiles per CTA): K + V, proj 96 x 24 bf16, bias 6 x 16 x 80 fp32, Q lo 6 x 16 x 136 f16.
pub const ATTN_TC2_SMEM: usize = 2 * 64 * 136 * 2 + 96 * 24 * 2 + 6 * 16 * 80 * 4 + 6 * 16 * 136 * 2;
thread_local! { pub static ATTN_TC2: std::cell::Cell<bool> = const { std::cell::Cell::new(true) }; }
/// S7 opt-in (`--inkling2-attn-single-pass`): prefill attention with Q / P rounded once to f16 (ink_attn_tc2_sp,
/// half the MMAs; CHANGES outputs). Default off.
thread_local! { pub static ATTN_SP: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
pub const ATTN_TC2_SP_SMEM: usize = ATTN_TC2_SMEM - 6 * 16 * 136 * 2;
pub const fn attn_smem(m: usize) -> usize {
    let qcap = 3 * if m < 16 { m } else { 16 };
    4 * (qcap * 144 + 16 * 80) + 64 * (130 + 128) * 2
}

/// The exact bf16 bits of a bf16-valued f32 (the pack's plain tensors are bf16 widened exactly).
pub fn bf16_bits(x: f32) -> u16 { (x.to_bits() >> 16) as u16 }
/// Round-to-nearest-even f32 -> bf16 bits.
pub fn f32_to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() { return 0x7fc0; }
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}
pub fn bf16_to_f32(b: u16) -> f32 { f32::from_bits((b as u32) << 16) }

/// Pack a row-major bf16 weight `[n][k]` (as u16 bits) into the B32 fragment-native layout
/// (05_proj_gemm.cuh), N padded to a multiple of 16 with zero rows.
pub fn b32_pack(w: &[u16], n: usize, k: usize) -> Vec<u16> {
    assert_eq!(w.len(), n * k);
    assert!(k % 32 == 0, "K {k} % 32");
    let nt = n.div_ceil(16);
    let g = k / 32;
    let mut o = vec![0u16; nt * 16 * k];
    for t in 0..nt {
        for gg in 0..g {
            for h in 0..2 {
                for lane in 0..32 {
                    let (gid, tig) = (lane >> 2, lane & 3);
                    let row = 16 * t + 8 * h + gid;
                    let dst = (((t * g + gg) * 2 + h) * 32 + lane) * 8;
                    if row < n {
                        let src = row * k + 32 * gg + 8 * tig;
                        o[dst..dst + 8].copy_from_slice(&w[src..src + 8]);
                    }
                }
            }
        }
    }
    o
}

/// A fused projection weight: segments (rows each, padded to 16) concatenated, B32-packed.
pub struct ProjW {
    pub buf: cudarc::driver::CudaSlice<u16>,
    pub k: usize,
    pub n_pad: usize,
    /// row offset (column in Y) of each segment, in the order given
    pub offs: Vec<usize>,
}
impl ProjW {
    pub fn ntiles(&self) -> usize { self.n_pad / 16 }
    /// Build from segments of bf16-valued f32 rows (`[rows][k]` each).
    pub fn new(ks: &Ink2Kernels, segs: &[&[f32]], k: usize) -> Result<ProjW> {
        let mut rows: Vec<u16> = Vec::new();
        let mut offs = Vec::new();
        let mut n = 0usize;
        for s in segs {
            ensure!(s.len() % k == 0, "segment {} not a multiple of K {k}", s.len());
            offs.push(n);
            rows.extend(s.iter().map(|&x| bf16_bits(x)));
            let r = s.len() / k;
            let pad = r.div_ceil(16) * 16 - r;
            rows.extend(std::iter::repeat(0u16).take(pad * k));
            n += r + pad;
        }
        let packed = b32_pack(&rows, n, k);
        Ok(ProjW { buf: ks.dev.htod_sync_copy(&packed)?, k, n_pad: n, offs })
    }
    pub fn bytes(&self) -> usize { self.n_pad * self.k * 2 }
}

/// k-split for the decode GEMM: the smallest that gives >= `min_ctas` CTAs (default 192 = 4 per SM),
/// limited by (K/32) % KS == 0.
pub fn proj_ks(ntiles: usize, k: usize, min_ctas: usize) -> usize {
    let g = k / 32;
    let mut best = 1;
    for ks in [1usize, 2, 4, 8] {
        if g % ks != 0 { continue; }
        best = ks;
        if (ntiles * ks).div_ceil(8) >= min_ctas { break; }
    }
    best
}

impl Ink2Kernels {
    /// Y[M][ldy] = X . W^T with up to three activation segments (x1 from tile t1, x2 from tile t2).
    /// M <= 16 -> ink_proj_dec_ks{ks} (ks = 0: policy), else ink_proj_wide.
    #[allow(clippy::too_many_arguments)]
    pub fn proj(&self, w: &ProjW, x0: u64, x1: u64, x2: u64, t1: usize, t2: usize, y: u64, ldy: usize, m: usize, ks: usize) -> Result<()> {
        let nt = w.ntiles();
        let mut a = KArgs::new();
        a.v(dp(&w.buf)).v(x0).v(x1).v(x2).v(t1 as i32).v(t2 as i32).v(y).v(ldy as i32).v(m as i32).v(w.k as i32).v(nt as i32);
        if m <= 16 || (m <= VERIFY_MAX_ROWS && verify_class()) {
            let ks = if ks == 0 { proj_ks(nt, w.k, 192) } else { ks };
            ensure!((w.k / 32) % ks == 0, "K {} not divisible into {ks} slices", w.k);
            let tpb = 8 / ks;
            a.v(0i32);
            // S7: 17..48 verify rows run the decode kernel's arithmetic with three 16-row groups per weight fragment
            let name = if m <= 16 { format!("ink_proj_dec_ks{ks}") } else { format!("ink_proj_dec3_ks{ks}") };
            self.launch(&name, (nt.div_ceil(tpb) as u32, 1, 1), (256, 1, 1), 0, &mut a)
        } else if m < PROJ_TILE_MIN_M {
            self.launch("ink_proj_wide", (nt.div_ceil(8) as u32, m.div_ceil(64) as u32, 1), (256, 1, 1), 0, &mut a)
        } else {
            // S4: the shared-memory tiled GEMM (06_proj_tile.cuh), bit-identical to ink_proj_wide
            let st = PROJ_TILE_ST.with(|c| c.get());
            let name = match st { 2 => "ink_proj_tile2", 3 => "ink_proj_tile3", _ => "ink_proj_tile" };
            let (c1, c2) = (t1.min(nt), t2.max(t1).min(nt).max(t1.min(nt)));
            let nblk = c1.div_ceil(8) + (c2 - c1).div_ceil(8) + (nt - c2).div_ceil(8);
            self.launch(name, (m.div_ceil(128) as u32, nblk as u32, 1), (256, 1, 1), proj_tile_smem(st) as u32, &mut a)
        }
    }
    /// The K1 prefill GEMM (05 ink_proj_wide), kept as the reference of ink_proj_tile (probe bit-identity check).
    #[allow(clippy::too_many_arguments)]
    pub fn proj_wide_ref(&self, w: &ProjW, x0: u64, x1: u64, x2: u64, t1: usize, t2: usize, y: u64, ldy: usize, m: usize) -> Result<()> {
        let nt = w.ntiles();
        let mut a = KArgs::new();
        a.v(dp(&w.buf)).v(x0).v(x1).v(x2).v(t1 as i32).v(t2 as i32).v(y).v(ldy as i32).v(m as i32).v(w.k as i32).v(nt as i32);
        self.launch("ink_proj_wide", (nt.div_ceil(8) as u32, m.div_ceil(64) as u32, 1), (256, 1, 1), 0, &mut a)
    }
}
/// 06_proj_tile.cuh smem of `st` stages x (128 x 80 B A rows + 8 KB B).
pub const fn proj_tile_smem(st: usize) -> usize { st * (128 * 80 + 8 * 1024) }
/// 40_msggate.cuh InkMsg3 (pointer triples of the batched MessageGate).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Msg3 { pub h: [u64; 3], pub m: [u64; 3], pub cat: [u64; 3], pub down: [u64; 3], pub z: [u64; 3], pub wf: [u64; 3], pub wg: [u64; 3], pub out: [u64; 3] }
unsafe impl DeviceRepr for Msg3 {}
/// S7: largest row count of a decode-class (verify) call: the projections, MessageGate, AttnRes and MoE of a
/// `verify_lanes` call run the DECODE kernels' arithmetic up to this many rows (so each row is bit-identical to a
/// 1-row decode); the switch is `VERIFY_CLASS`, set only around a lanes (decode / verify) forward.
pub const VERIFY_MAX_ROWS: usize = 48;
thread_local! { pub static VERIFY_CLASS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
pub fn verify_class() -> bool { VERIFY_CLASS.with(|c| c.get()) }
thread_local! {
    /// S7 A/B switch (probe): serve multi-row verify runs with ink_attn_lanes_mr (default) or every row per-row.
    pub static ATTN_MR: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}
/// The verify-path switches as bits (graph cache key): mr attention, small router tiles, expert verify tiles.
pub fn verify_cfg() -> usize {
    ATTN_MR.with(|c| c.get()) as usize | (super::moe::ROUTER_SMALL.with(|c| c.get()) as usize) << 1
        | (super::moe::VERIFY_KERNELS.with(|c| c.get()) as usize) << 2
}
/// Rows from which the tiled GEMM beats ink_proj_wide (S4 probe: at 37 rows wide is 1.8x faster).
pub const PROJ_TILE_MIN_M: usize = 128;
thread_local! { pub static PROJ_TILE_ST: std::cell::Cell<usize> = const { std::cell::Cell::new(2) }; }

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PrepArgs {
    pub y: u64, pub ldy: i32,
    pub oq: i32, pub orr: i32, pub ok: i32, pub ov: i32, pub okd: i32, pub ovd: i32,
    pub q_scale: u64, pub r_scale: u64,
    pub qk_clip: f32, pub rel_clip: f32,
    pub full: i32, pub log_alpha: f32, pub log_floor: f32,
    pub cw_k: u64, pub cw_v: u64,
    pub conv_div: f32,
    pub ring_k: u64, pub ring_v: u64, pub ring_mask: i32, pub ring_write: i32,
    pub kc: u64, pub vc: u64, pub cap_mask: u32,
    pub eps: f32,
    pub qf: u64, pub rf: u64,
    pub p0: i32, pub m: i32,
}
unsafe impl DeviceRepr for PrepArgs {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AttnArgs {
    pub qf: u64, pub rf: u64,
    pub kc: u64, pub vc: u64, pub cap_mask: u32,
    pub proj: u64, pub ext: i32,
    pub window: i32,
    pub p0: i32, pub m: i32,
    pub chunk: i32, pub c0: i32,
    pub y: u64, pub ldy: i32, pub os: i32, pub og: i32,
    pub out: u64,
    pub pm: u64, pub pl: u64, pub pacc: u64,
}
unsafe impl DeviceRepr for AttnArgs {}

/// The split of one attention call: depends on p0 only (never on M), so row 0 of an M-row call sees
/// exactly the M = 1 call's chunks (batch invariance).
#[derive(Clone, Copy, Debug)]
pub struct AttnSplit { pub chunk: usize, pub c0: usize, pub nsplit: usize }
pub fn attn_split(p0: usize, m: usize, window: usize) -> AttnSplit {
    let kmin = if window > 0 { (p0 + 1).saturating_sub(window) } else { 0 };
    let span = p0 + 16 - kmin;
    let chunk = (span.div_ceil(16)).next_power_of_two().max(64);
    let c0 = kmin / chunk;
    // at least 2 splits: the in-place finalize (nsplit 1) rounds differently from ink_attn_combine, so a row must take
    // the combine path whatever the other rows of its call need (S6: a verify row == the decode of that row, bitwise;
    // an empty trailing chunk holds no key and the combine skips it)
    let nsplit = ((p0 + m - 1) / chunk - c0 + 1).max(2);
    AttnSplit { chunk, c0, nsplit }
}

/// Row groups `(j0, rows)` of a decode / verify segment (m <= 16 rows at p0..) that share the chunk size the M = 1
/// decode of each row would use (S6: rows across a chunk-size threshold get their own launch, so every verify row
/// is computed exactly as its decode).
pub fn attn_groups(p0: usize, m: usize, window: usize) -> Vec<(usize, usize)> {
    let mut g: Vec<(usize, usize)> = Vec::new();
    let mut last = usize::MAX;
    for j in 0..m {
        let c = attn_split(p0 + j, 1, window).chunk;
        if c == last { g.last_mut().unwrap().1 += 1; } else { g.push((j, 1)); last = c; }
    }
    g
}

impl Ink2Kernels {
    pub fn attn_prep(&self, a: &PrepArgs) -> Result<()> {
        ensure!(a.ring_write == 0 || (a.m as usize) <= RING - 3, "ring write with M {} > R - 3", a.m);
        self.launch("ink_attn_prep", (a.m as u32, 1, 1), (1024, 1, 1), 0, KArgs::new().v(*a))
    }
    /// Decode / verify (M <= 16): split-KV + combine (or in-place finalize when one chunk covers it).
    /// Prefill (M > 16 or `prefill`): one CTA per (16-row tile, kv head), no split.
    /// `a.pm/pl/pacc` must hold M*24*nsplit (+*128) floats for the split path.
    /// `tc`: the tensor-core kernel `ink_attn_tc` (11_attn_tc.cuh), else the CUDA-core `ink_attn`.
    pub fn attn(&self, a: &AttnArgs, prefill: bool, tc: bool) -> Result<AttnSplit> {
        let m = a.m as usize;
        let mut a = *a;
        let sp = if prefill || m > 16 {
            AttnSplit { chunk: 1 << 30, c0: 0, nsplit: 1 }
        } else {
            attn_split(a.p0 as usize, m, a.window as usize)
        };
        a.chunk = sp.chunk as i32;
        a.c0 = sp.c0 as i32;
        let grid = (sp.nsplit as u32, NKV as u32, m.div_ceil(16) as u32);
        if tc && sp.nsplit == 1 && m > 16 && ATTN_TC2.with(|c| c.get()) {
            // S4: two 16-token tiles per CTA (bit-identical to ink_attn_tc, prefill only)
            let (name, sm) = if ATTN_SP.with(|c| c.get()) { ("ink_attn_tc2_sp", ATTN_TC2_SP_SMEM) } else { ("ink_attn_tc2", ATTN_TC2_SMEM) };
            self.launch(name, (1, NKV as u32, m.div_ceil(32) as u32), (192, 1, 1), sm as u32, KArgs::new().v(a))?;
        } else if tc {
            self.launch("ink_attn_tc", grid, (96, 1, 1), ATTN_TC_SMEM as u32, KArgs::new().v(a))?;
        } else {
            self.launch("ink_attn", grid, (256, 1, 1), attn_smem(m) as u32, KArgs::new().v(a))?;
        }
        if sp.nsplit > 1 {
            self.launch("ink_attn_combine", (m as u32, NH as u32, 1), (128, 1, 1), 0, KArgs::new().v(a).v(sp.nsplit as i32))?;
        }
        Ok(sp)
    }
    /// Static conv + residual: out = base + (conv(x) + x)/sqrt(5); ring updated in-kernel when M <= R - 3,
    /// else by ink_ring_store after.
    #[allow(clippy::too_many_arguments)]
    pub fn sconv(&self, x: u64, ldx: usize, ring: u64, w: u64, base: u64, out: u64, c: usize, p0: usize, m: usize) -> Result<()> {
        let rw = m <= RING - 3;
        self.launch("ink_sconv", (c.div_ceil(256) as u32, m as u32, 1), (256, 1, 1), 0,
                    KArgs::new().v(x).v(ldx as i32).v(ring).v((RING - 1) as i32).v(rw as i32).v(w).v(base).v(out)
                        .v(c as i32).v(p0 as i32).v(1.0f32 / 5f32.sqrt()))?;
        if !rw { self.ring_store(x, ldx, 0, ring, c, p0, m)?; }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn ring_store(&self, x: u64, ldx: usize, off: usize, ring: u64, c: usize, p0: usize, m: usize) -> Result<()> {
        let rows = m.min(RING);
        self.launch("ink_ring_store", (c.div_ceil(256) as u32, rows as u32, 1), (256, 1, 1), 0,
                    KArgs::new().v(x).v(ldx as i32).v(off as i32).v(ring).v((RING - 1) as i32).v(c as i32).v(p0 as i32).v(m as i32).v(rows as i32))
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ArArgs { pub v: [u64; 6], pub nv: i32, pub q: u64, pub nq: i32, pub out: [u64; 4], pub d: i32 }
unsafe impl DeviceRepr for ArArgs {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ArNormArgs { pub ar: ArArgs, pub nw: [u64; 3], pub ob: [u64; 3], pub of: [u64; 3], pub eps: f32 }
unsafe impl DeviceRepr for ArNormArgs {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Norm3Args { pub x: [u64; 3], pub w: [u64; 3], pub ob: [u64; 3], pub of: [u64; 3] }
unsafe impl DeviceRepr for Norm3Args {}

impl Ink2Kernels {
    /// AttnRes: `vals` = blocks then the partial (<= 6), `nq` queries [nq][d] at `q`, outputs `outs`.
    pub fn attnres(&self, vals: &[u64], q: u64, nq: usize, outs: &[u64], d: usize, m: usize) -> Result<()> {
        ensure!(!vals.is_empty() && vals.len() <= 6 && nq >= 1 && nq <= 4 && outs.len() == nq && d % 256 == 0);
        let mut a = ArArgs { nv: vals.len() as i32, q, nq: nq as i32, d: d as i32, ..Default::default() };
        a.v[..vals.len()].copy_from_slice(vals);
        a.out[..nq].copy_from_slice(outs);
        // decode / verify rows: one row per 8-CTA cluster (latency); prefill: one CTA per row (throughput)
        if d == 8 * 288 && (m <= 16 || (m <= VERIFY_MAX_ROWS && verify_class())) {
            self.launch("ink_attnres_cl", (8, m as u32, 1), (288, 1, 1), 0, KArgs::new().v(a))
        } else {
            self.launch("ink_attnres", (m as u32, 1, 1), (256, 1, 1), 0, KArgs::new().v(a))
        }
    }
    /// Drop 3 (#4a): Fused cluster AttnRes + RMSNorm for decode/verify widths (`d == 2304`, `nq <= 3`).
    #[allow(clippy::too_many_arguments)]
    pub fn attnres_cl_norm(&self, vals: &[u64], q: u64, nq: usize, nw: &[u64], ob: &[u64], of: &[u64], d: usize, m: usize, eps: f32) -> Result<()> {
        ensure!(!vals.is_empty() && vals.len() <= 6 && nq >= 1 && nq <= 3 && nw.len() == nq && ob.len() == nq && of.len() == nq && d == 8 * 288);
        let mut ar = ArArgs { nv: vals.len() as i32, q, nq: nq as i32, d: d as i32, ..Default::default() };
        ar.v[..vals.len()].copy_from_slice(vals);
        let mut p = ArNormArgs { ar, eps, ..Default::default() };
        p.nw[..nq].copy_from_slice(nw);
        p.ob[..nq].copy_from_slice(ob);
        p.of[..nq].copy_from_slice(of);
        self.launch("ink_attnres_cl_norm", (8, m as u32, 1), (288, 1, 1), 0, KArgs::new().v(p))
    }
    /// Drop 3 (#4a): Batched 3-stream RMSNorm in one launch (bit-identical to 3 `rmsnorm` calls).
    pub fn rmsnorm_x3(&self, x: [u64; 3], w: [u64; 3], ob: [u64; 3], of: [u64; 3], d: usize, m: usize, eps: f32) -> Result<()> {
        let a = Norm3Args { x, w, ob, of };
        self.launch("ink_rmsnorm_x3", (m as u32, 1, 3), (256, 1, 1), 0, KArgs::new().v(a).v(d as i32).v(eps))
    }
    /// MessageGate: cat (bf16 scratch [M][2d]) -> z = down (fp32; at M <= 16 as MSG_ZS k-slice partials
    /// [MSG_ZS][M][64], so `z` holds MSG_ZS * M * 64 floats) -> out.
    #[allow(clippy::too_many_arguments)]
    pub fn msggate(&self, down: &ProjW, wf: u64, wg: u64, h: u64, msg: u64, cat: u64, z: u64, out: u64, d: usize, m: usize, eps: f32) -> Result<()> {
        self.launch("ink_rms2_cat_bf16", (m as u32, 1, 1), (256, 1, 1), 0, KArgs::new().v(h).v(msg).v(cat).v(d as i32).v(eps))?;
        let nzs = if m <= 16 || (m <= VERIFY_MAX_ROWS && verify_class()) {
            let (nt, g) = (down.ntiles(), down.k / 32);
            ensure!(g % (8 * MSG_ZS) == 0, "down K {} vs {MSG_ZS} slices", down.k);
            self.launch(if m <= 16 { "ink_proj_dec_ks8" } else { "ink_proj_dec3_ks8" }, (nt as u32, MSG_ZS as u32, 1), (256, 1, 1), 0,
                        KArgs::new().v(dp(&down.buf)).v(cat).v(cat).v(cat).v(1i32 << 30).v(1i32 << 30).v(z).v(64i32).v(m as i32)
                            .v(down.k as i32).v(nt as i32).v((m * 64) as i32))?;
            MSG_ZS
        } else {
            self.proj(down, cat, cat, cat, 1 << 30, 1 << 30, z, 64, m, 0)?;
            1
        };
        let (name, mt, bs) = match m { 1 => ("ink_msggate_out_m1", 1, 64), 2..=4 => ("ink_msggate_out_m4", 4, 64), _ => ("ink_msggate_out", 16, 256) };
        self.launch(name, (d.div_ceil(bs) as u32, m.div_ceil(mt) as u32, 1), (bs as u32, 1, 1), 0,
                    KArgs::new().v(z).v(wf).v(wg).v(h).v(msg).v(out).v(d as i32).v(m as i32).v(nzs as i32).v((m * 64) as i32))
    }
    /// S7: three MessageGates in one launch per stage (the q / k / v attention-input gates of a loop layer), decode /
    /// verify widths only (m <= 16, or <= VERIFY_MAX_ROWS under verify_class). Gate i: (down[i], wf[i], wg[i]) on
    /// (h[i], msg[i]) -> out[i]; cat / z scratch for gate i at `cat + i * m * 2d` (bf16) / `z + i * MSG_ZS * m * 64`
    /// (fp32). Bit-identical to three `msggate` calls.
    #[allow(clippy::too_many_arguments)]
    pub fn msggate3(&self, down: [&ProjW; 3], wf: [u64; 3], wg: [u64; 3], h: [u64; 3], msg: [u64; 3], cat: u64, z: u64, out: [u64; 3],
                    d: usize, m: usize, eps: f32) -> Result<()> {
        ensure!(m <= 16 || (m <= VERIFY_MAX_ROWS && verify_class()), "msggate3: {m} rows outside decode / verify widths");
        let (nt, g) = (down[0].ntiles(), down[0].k / 32);
        ensure!(g % (8 * MSG_ZS) == 0 && down.iter().all(|w| w.ntiles() == nt && w.k == down[0].k), "msggate3 shapes");
        let mut p = Msg3 { ..Default::default() };
        for i in 0..3 {
            p.h[i] = h[i]; p.m[i] = msg[i]; p.cat[i] = cat + (i * m * 2 * d * 2) as u64;
            p.down[i] = dp(&down[i].buf); p.z[i] = z + (i * MSG_ZS * m * 64 * 4) as u64;
            p.wf[i] = wf[i]; p.wg[i] = wg[i]; p.out[i] = out[i];
        }
        self.launch("ink_rms2_cat_bf16_x3", (m as u32, 1, 3), (256, 1, 1), 0, KArgs::new().v(p).v(d as i32).v(eps))?;
        self.launch("ink_msg_down_x3", (nt as u32, MSG_ZS as u32, 3), (256, 1, 1), 0,
                    KArgs::new().v(p).v(m as i32).v(down[0].k as i32).v(nt as i32).v((m * 64) as i32))?;
        let (name, mt, bs) = match m { 1 => ("ink_msggate_out_m1_x3", 1, 64), 2..=4 => ("ink_msggate_out_m4_x3", 4, 64), _ => ("ink_msggate_out_x3", 16, 256) };
        self.launch(name, (d.div_ceil(bs) as u32, m.div_ceil(mt) as u32, 3), (bs as u32, 1, 1), 0,
                    KArgs::new().v(p).v(d as i32).v(m as i32).v(MSG_ZS as i32).v((m * 64) as i32))
    }
    /// Weighted (w != 0) / unweighted RMSNorm; bf16 and / or fp32 outputs (0 = skip).
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm(&self, x: u64, w: u64, ob: u64, of: u64, d: usize, m: usize, eps: f32) -> Result<()> {
        self.launch("ink_rmsnorm", (m as u32, 1, 1), (256, 1, 1), 0, KArgs::new().v(x).v(w).v(ob).v(of).v(d as i32).v(eps))
    }
    pub fn embed_norm(&self, ids: u64, emb: u64, w: u64, out: u64, d: usize, m: usize, eps: f32) -> Result<()> {
        self.launch("ink_embed_norm", (m as u32, 1, 1), (256, 1, 1), 0, KArgs::new().v(ids).v(emb).v(w).v(out).v(d as i32).v(eps))
    }
}

/// S4 lane-batched decode (kernels/gpu_inkling2/85_lanes.cuh): rows of one or more lanes, every per-sequence value
/// read on the device from `tab` ([lanes][slots] InkLaneSlot, 8 u64 each) and `rinfo` ([4][LANES_MAXR] i32: ids,
/// positions, lane, index of the row inside its lane's run (S7 verify rows)), so the launches of a step do not
/// depend on positions or on how the rows split into lanes (one CUDA graph per row count).
pub const LANE_SLOT_U64: usize = 8;
/// rinfo row capacity (85_lanes.cuh IKL_RI).
pub const LANES_MAXR: usize = 48;
/// Max attention splits of one decode row (85_lanes.cuh IKL_MAXS).
pub const LANES_MAXS: usize = 18;
impl Ink2Kernels {
    pub fn attn_prep_lanes(&self, a: &PrepArgs, tab: u64, slot: usize, nslots: usize, rinfo: u64, n: usize) -> Result<()> {
        self.launch("ink_attn_prep_lanes", (n as u32, 1, 1), (1024, 1, 1), 0,
                    KArgs::new().v(*a).v(tab).v(slot as i32).v(nslots as i32).v(rinfo))
    }
    /// `runs`: some lane carries several consecutive rows (a verify): `ink_attn_lanes_mr` serves those runs (K / V
    /// read once per run, each row bit-identical to its 1-row decode); `ink_attn_lanes` serves every other row.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_lanes(&self, a: &AttnArgs, tab: u64, slot: usize, nslots: usize, rinfo: u64, n: usize, runs: bool) -> Result<()> {
        // a.c0 (unused by the lanes entries, positions are on the device) = 1: no multi-row runs (A/B switch)
        let runs = runs && ATTN_MR.with(|c| c.get());
        let mut a = *a;
        a.c0 = if runs { 0 } else { 1 };
        let a = &a;
        self.launch("ink_attn_lanes", (LANES_MAXS as u32, NKV as u32, n as u32), (256, 1, 1), attn_smem(1) as u32,
                    KArgs::new().v(*a).v(tab).v(slot as i32).v(nslots as i32).v(rinfo))?;
        if runs {
            self.launch("ink_attn_lanes_mr", (LANES_MAXS as u32 + 1, NKV as u32, n as u32), (256, 1, 1), attn_smem(16) as u32,
                        KArgs::new().v(*a).v(tab).v(slot as i32).v(nslots as i32).v(rinfo))?;
        }
        self.launch("ink_attn_combine_lanes", (n as u32, NH as u32, 1), (128, 1, 1), 0, KArgs::new().v(*a).v(rinfo))
    }
    #[allow(clippy::too_many_arguments)]
    pub fn sconv_lanes(&self, x: u64, ldx: usize, tab: u64, slot: usize, nslots: usize, ri: usize, w: u64, base: u64, out: u64,
                       c: usize, rinfo: u64, n: usize) -> Result<()> {
        self.launch("ink_sconv_lanes", (c.div_ceil(256) as u32, n as u32, 1), (256, 1, 1), 0,
                    KArgs::new().v(x).v(ldx as i32).v(tab).v(slot as i32).v(nslots as i32).v(ri as i32).v(w).v(base).v(out)
                        .v(c as i32).v(rinfo).v(1.0f32 / 5f32.sqrt()))
    }
}
