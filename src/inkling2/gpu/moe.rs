//! Inkling-2 MoE on the GPU: device weights of one layer's router + 128 routed experts (NVFP4 L0-19 / MXFP8
//! L20-23 of the stock pack), the scratch, and the dispatch (kernels/gpu_inkling2/61-64).
//!
//! Contract (f32 in, f32 out, so the trunk's own activation dtype is free):
//!   x [m][2304] fp32 = the MoE input (the `mlp_norm` output) -> out [m][2304] fp32 = sum_k w_k * expert_k(x).
//!   Router logits in fp32 from the stored bf16 gate (fp32 activations); the experts are W4A16 / W8A16 (f16 MMA,
//!   fp32 accumulation, exact weight dequant inside the B fragments), the expert input and the SwiGLU output rounded
//!   to f16 (|act| <= 100). Decode m <= 16: union-of-experts slots; prefill m > 16: device-routed 64-row tiles.
//!   Every token row is independent of the other rows (batch-invariant), and a prefill row is bit-identical to the
//!   same token's decode row.

use super::layout;
use super::dptr;
use crate::ink2_launch;
use crate::inkling2::pack::{ExpFmt, Inkling2Pack, P};
use anyhow::{ensure, Result};
use cudarc::driver::{CudaDevice, CudaSlice};
use std::sync::Arc;

pub const E: usize = 128;
pub const TOPK: usize = 8;
pub const H: usize = 2304;
pub const INTER: usize = 1152;
/// Prefill: compact rows per tile (the tile kernels' CTA height).
pub const PF_ROWS: usize = 64;
/// Router-logit tokens per CTA row at prefill.
const ROUTER_TPB: usize = 32;

pub struct ExpertsDev {
    pub fmt: ExpFmt,
    pub w13: CudaSlice<u8>,
    pub s13: CudaSlice<u8>,
    pub w2: CudaSlice<u8>,
    pub s2: CudaSlice<u8>,
    /// NVFP4: [E][2] = {w13 amax/den, w2 amax/den}; MXFP8: unused ([2] ones).
    pub gs: CudaSlice<f32>,
}

impl ExpertsDev {
    /// Weight bytes one expert streams per use (gate|up, down; codes + scales).
    pub fn bytes_per_expert(&self) -> (usize, usize) {
        match self.fmt {
            ExpFmt::Nvfp4 => (2 * INTER * H / 2 + 2 * INTER * H / 16, H * INTER / 2 + H * INTER / 16),
            ExpFmt::Mxfp8 => (2 * INTER * H + 2 * INTER * H / 32, H * INTER + H * INTER / 32),
        }
    }
}

pub struct MoeLayerDev {
    pub layer: usize,
    /// router gate weight [E][H] bf16 row-major (590 KB, read once per CTA)
    pub router_w: CudaSlice<u16>,
    pub router_b: CudaSlice<f32>,
    pub ex: ExpertsDev,
    /// inner channels held (INTER at TP1; a TP rank's whole-128-block slice [c0, c1), tp/shard.rs)
    pub inner: usize,
}

fn bf16_bits(st: &crate::fp4l::pack::StDir, name: &str) -> Result<Vec<u16>> {
    let b = st.raw(name)?;
    Ok(b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
}

impl MoeLayerDev {
    pub fn load(dev: &Arc<CudaDevice>, pack: &Inkling2Pack, l: usize) -> Result<MoeLayerDev> { Self::load_range(dev, pack, l, 0, INTER) }

    /// S-INK2-TP2: every expert's inner channels [c0, c1) only (gate / up rows 2c0..2c1 of the interleaved w13, down
    /// columns c0..c1); the kernels then see N = 2 (c1 - c0), K = c1 - c0. TP1 = [0, INTER).
    pub fn load_range(dev: &Arc<CudaDevice>, pack: &Inkling2Pack, l: usize, c0: usize, c1: usize) -> Result<MoeLayerDev> {
        let c = &pack.cfg;
        ensure!(c0 < c1 && c1 <= INTER && c0 % 64 == 0 && c1 % 64 == 0, "inkling2 MoE: inner slice [{c0}, {c1}) not whole 64-blocks of {INTER}");
        let inner = c1 - c0;
        let full = inner == INTER;
        ensure!(c.hidden == H && c.inter == INTER && c.n_experts == E && c.topk == TOPK,
                "inkling2 MoE kernels are built for hidden {H}, inter {INTER}, {E} experts, top-{TOPK}");
        let st = &pack.st;
        let p = format!("{P}layers.{l}.mlp.");
        let router_w = dev.htod_copy(bf16_bits(st, &format!("{p}gate.weight"))?)?;
        let router_b = dev.htod_copy(pack.plain(&format!("{p}gate.bias"))?)?;
        let fmt = pack.exp_fmt[l];
        let (n13, k13, n2, k2) = (2 * INTER, H, H, INTER);
        // the slice of one expert's stored bytes: w13 rows [2c0, 2c1) ([n13][rb] row-major), w2 columns [c0, c1)
        // (`per` = the stored bytes of one channel along K: 1/2 for NVFP4 codes, 1/16 / 1/32 for the scales)
        let cut = |proj: &str, b: &[u8], rb: usize, den: usize| -> Vec<u8> {
            if full { return b.to_vec(); }
            if proj == "w13" { b[2 * c0 * rb..2 * c1 * rb].to_vec() } else { crate::inkling2::tp::shard::byte_cols(b, rb, c0 / den, c1 / den) }
        };
        let (sn13, sk2) = (2 * inner, inner);
        let ex = match fmt {
            ExpFmt::Nvfp4 => {
                let mut gsv = vec![0f32; 2 * E];
                let mut out = Vec::new();
                for (pi, (proj, n, k, on, ok)) in [("w13", n13, k13, sn13, k13), ("w2", n2, k2, n2, sk2)].into_iter().enumerate() {
                    let b = format!("{p}experts.{proj}_weight");
                    let codes = st.raw(&b)?;
                    let scales = st.raw(&format!("{b}_scale"))?;
                    let amax = pack.plain(&format!("{b}_amax"))?;
                    let den = pack.plain(&format!("{b}_outer_scale_denominator"))?[0];
                    for e in 0..E { gsv[2 * e + pi] = amax[e] / den; }
                    let (ce, se) = (n * k / 2, n * k / 16);
                    let (oce, ose) = (on * ok / 2, on * ok / 16);
                    let mut w = vec![0u8; E * oce];
                    let mut s = vec![0u8; E * ose];
                    let cut = &cut;
                    std::thread::scope(|sco| {
                        for (e, (wc, sc)) in w.chunks_mut(oce).zip(s.chunks_mut(ose)).enumerate() {
                            let (codes, scales) = (&codes, &scales);
                            sco.spawn(move || {
                                let (mut cc, mut ss) = (cut(proj, &codes[e * ce..(e + 1) * ce], k / 2, 2), cut(proj, &scales[e * se..(e + 1) * se], k / 16, 16));
                                if proj == "w13" {
                                    cc = layout::w13_tile_rows(&cc, on, ok / 2);
                                    ss = layout::w13_tile_rows(&ss, on, ok / 16);
                                }
                                layout::nv4_frag_into(&cc, &ss, on, ok, wc, sc);
                            });
                        }
                    });
                    out.push((dev.htod_copy(w)?, dev.htod_copy(s)?));
                }
                let (w2, s2) = out.pop().unwrap();
                let (w13, s13) = out.pop().unwrap();
                ExpertsDev { fmt, w13, s13, w2, s2, gs: dev.htod_copy(gsv)? }
            }
            ExpFmt::Mxfp8 => {
                let mut out = Vec::new();
                for (proj, n, k, on, ok) in [("w13", n13, k13, sn13, k13), ("w2", n2, k2, n2, sk2)] {
                    let b = format!("{p}experts.{proj}_weight");
                    let codes = st.raw(&b)?;
                    let scales = st.raw(&format!("{b}_scale"))?;
                    ensure!(!scales.contains(&0xFF), "{b}_scale: a UE8M0 NaN scale (0xFF)");
                    let (ce, se) = (n * k, n * k / 32);
                    let (oce, ose) = (on * ok, on * ok / 32);
                    let mut w = vec![0u8; E * oce];
                    let mut s = vec![0u8; E * ose];
                    let cut = &cut;
                    std::thread::scope(|sco| {
                        for (e, (wc, sc)) in w.chunks_mut(oce).zip(s.chunks_mut(ose)).enumerate() {
                            let (codes, scales) = (&codes, &scales);
                            sco.spawn(move || {
                                let (mut cc, mut ss) = (cut(proj, &codes[e * ce..(e + 1) * ce], k, 1), cut(proj, &scales[e * se..(e + 1) * se], k / 32, 32));
                                if proj == "w13" {
                                    cc = layout::w13_tile_rows(&cc, on, ok);
                                    ss = layout::w13_tile_rows(&ss, on, ok / 32);
                                }
                                layout::mx8_frag_into(&cc, &ss, on, ok, wc, sc);
                            });
                        }
                    });
                    out.push((dev.htod_copy(w)?, dev.htod_copy(s)?));
                }
                let (w2, s2) = out.pop().unwrap();
                let (w13, s13) = out.pop().unwrap();
                ExpertsDev { fmt, w13, s13, w2, s2, gs: dev.htod_copy(vec![1f32; 2 * E])? }
            }
        };
        Ok(MoeLayerDev { layer: l, router_w, router_b, ex, inner })
    }
}

/// Per-call scratch, sized for `cap_m` tokens (decode needs cap_m >= 16 for the slot buffers).
pub struct MoeScratch {
    pub cap_m: usize,
    pub xh: CudaSlice<u16>,      // [cap_m][H] f16
    pub logits: CudaSlice<f32>,  // [cap_m][E]
    pub ids: CudaSlice<i32>,     // [cap_m][8] (ascending expert id)
    pub wts: CudaSlice<f32>,     // [cap_m][8]
    pub rowidx: CudaSlice<i32>,  // [cap_m][8]
    pub idxmap: CudaSlice<i32>,  // [E]
    pub esel: CudaSlice<i32>,    // [1]
    pub din: CudaSlice<u16>,     // decode [E][16][INTER] / prefill [8 cap_m][INTER] f16
    pub y: CudaSlice<f32>,       // decode [E][16][H] / prefill [8 cap_m][H]
    pub cnt: CudaSlice<i32>,
    pub offs: CudaSlice<i32>,
    pub rowtok: CudaSlice<i32>,  // [8 cap_m]
    pub tiles: CudaSlice<i32>,   // [2 max_tiles]
    pub ntiles: CudaSlice<i32>,
}

pub fn max_tiles(m: usize) -> usize { m * TOPK / PF_ROWS + E }

impl MoeScratch {
    pub fn new(dev: &Arc<CudaDevice>, cap_m: usize) -> Result<MoeScratch> {
        let rows = (TOPK * cap_m).max(E * 16);
        Ok(MoeScratch {
            cap_m,
            xh: dev.alloc_zeros(cap_m * H)?,
            logits: dev.alloc_zeros(cap_m * E)?,
            ids: dev.alloc_zeros(cap_m * TOPK)?,
            wts: dev.alloc_zeros(cap_m * TOPK)?,
            rowidx: dev.alloc_zeros(cap_m * TOPK)?,
            idxmap: dev.alloc_zeros(E)?,
            esel: dev.alloc_zeros(1)?,
            din: dev.alloc_zeros(rows * INTER)?,
            y: dev.alloc_zeros(rows * H)?,
            cnt: dev.alloc_zeros(E)?,
            offs: dev.alloc_zeros(E)?,
            rowtok: dev.alloc_zeros(TOPK * cap_m)?,
            tiles: dev.alloc_zeros(2 * max_tiles(cap_m))?,
            ntiles: dev.alloc_zeros(1)?,
        })
    }
}

/// The stages, separately launchable (the probe times them one by one).
pub fn router(dev: &Arc<CudaDevice>, lay: &MoeLayerDev, sc: &MoeScratch, x: u64, m: usize) -> Result<()> {
    ensure!(m >= 1 && m <= sc.cap_m, "inkling2 MoE: m {m} vs scratch {}", sc.cap_m);
    // decode: 8 experts per CTA (measured faster than 4 at M 1..16); prefill: 16 (fewer re-reads of the fp32 rows)
    // S7: decode / verify widths (<= 48 rows in a lanes call) use at most 4 tokens per CTA row (one interleaved
    // chain): more CTAs at 2..48 rows; a token's dot is the same chain whatever the grouping (gated: verify == decode)
    let small = ROUTER_SMALL.with(|c| c.get());
    let (tpb, wpb) = if m <= 16 || (m <= VERIFY_MAX_M && super::k1::verify_class()) { (if small { m.min(4) } else { m.min(16) }, 8) } else { (ROUTER_TPB, 16) };
    ink2_launch!(dev, "ink2_router_logits", ((E / wpb) as u32, m.div_ceil(tpb) as u32, 1), ((32 * wpb) as u32, 1, 1), 0,
                 (x, dptr(&lay.router_w, 0), dptr(&sc.logits, 0), dptr(&sc.xh, 0), m as i32, E as i32, tpb as i32))?;
    let build = (m <= 16) as i32;
    ink2_launch!(dev, "ink2_router_topk", (m.div_ceil(16) as u32, 1, 1), (512, 1, 1), 0,
                 (dptr(&sc.logits, 0), dptr(&lay.router_b, 0), dptr(&sc.ids, 0), dptr(&sc.wts, 0), dptr(&sc.rowidx, 0),
                  dptr(&sc.idxmap, 0), dptr(&sc.esel, 0), m as i32, 8.0f32, build))
}

/// Drop 2 (#2): sequence-parallel prefill router on this rank's `my_m` rows at global row offset `r0`.
/// Writes `sc.xh[r0..r0+my_m]` (f16), `sc.ids[r0..r0+my_m]`, `sc.wts[r0..r0+my_m]` (`build = 0` always).
pub fn router_slice(dev: &Arc<CudaDevice>, lay: &MoeLayerDev, sc: &MoeScratch, x: u64, r0: usize, my_m: usize) -> Result<()> {
    ensure!(my_m >= 1 && r0 + my_m <= sc.cap_m, "inkling2 MoE SP router: [{r0}..{}) vs scratch {}", r0 + my_m, sc.cap_m);
    let (tpb, wpb) = (ROUTER_TPB, 16);
    ink2_launch!(dev, "ink2_router_logits", ((E / wpb) as u32, my_m.div_ceil(tpb) as u32, 1), ((32 * wpb) as u32, 1, 1), 0,
                 (x, dptr(&lay.router_w, 0), dptr(&sc.logits, 0), dptr(&sc.xh, r0 * H), my_m as i32, E as i32, tpb as i32))?;
    ink2_launch!(dev, "ink2_router_topk", (my_m.div_ceil(16) as u32, 1, 1), (512, 1, 1), 0,
                 (dptr(&sc.logits, 0), dptr(&lay.router_b, 0), dptr(&sc.ids, r0 * TOPK), dptr(&sc.wts, r0 * TOPK),
                  dptr(&sc.rowidx, 0), dptr(&sc.idxmap, 0), dptr(&sc.esel, 0), my_m as i32, 8.0f32, 0i32))
}

/// Drop 2 (#2): prefill MoE dispatch after `sc.xh`, `sc.ids`, `sc.wts` have been all-gathered across `[0..m)`.
pub fn forward_routed(dev: &Arc<CudaDevice>, lay: &MoeLayerDev, sc: &MoeScratch, out: u64, m: usize) -> Result<()> {
    ensure!(m > 16 && m <= sc.cap_m, "inkling2 MoE SP forward_routed: m {m}");
    route_prefill(dev, sc, m)?;
    experts_prefill(dev, lay, sc, m, 3)?;
    combine(dev, sc, out, m)
}

/// Decode experts (m <= 16): gate|up (+ SwiGLU) then down over the union slots.
pub fn experts_decode(dev: &Arc<CudaDevice>, lay: &MoeLayerDev, sc: &MoeScratch, m: usize, which: u8) -> Result<()> {
    ensure!((1..=16).contains(&m), "inkling2 MoE decode: m {m} > 16");
    let emax = (TOPK * m).min(E) as u32;
    let ex = &lay.ex;
    let (gu, dn) = match ex.fmt { ExpFmt::Nvfp4 => ("ink2_nv4_gu_dec", "ink2_nv4_dn_dec"), ExpFmt::Mxfp8 => ("ink2_mx8_gu_dec", "ink2_mx8_dn_dec") };
    let (m_, idx, es) = (m as i32, dptr(&sc.idxmap, 0), dptr(&sc.esel, 0));
    if which & 1 != 0 {
        match ex.fmt {
            ExpFmt::Nvfp4 => ink2_launch!(dev, gu, (((2 * lay.inner) / 64) as u32, emax, 1), (128, 1, 1), 0,
                (dptr(&ex.w13, 0), dptr(&ex.s13, 0), dptr(&ex.gs, 0), dptr(&sc.xh, 0), dptr(&sc.din, 0), idx, es, m_, H as i32, (2 * lay.inner) as i32))?,
            ExpFmt::Mxfp8 => ink2_launch!(dev, gu, (((2 * lay.inner) / 64) as u32, emax, 1), (128, 1, 1), 0,
                (dptr(&ex.w13, 0), dptr(&ex.s13, 0), dptr(&sc.xh, 0), dptr(&sc.din, 0), idx, es, m_, H as i32, (2 * lay.inner) as i32))?,
        }
    }
    if which & 2 != 0 {
        match ex.fmt {
            ExpFmt::Nvfp4 => ink2_launch!(dev, dn, ((H / 64) as u32, emax, 1), (128, 1, 1), 0,
                (dptr(&ex.w2, 0), dptr(&ex.s2, 0), dptr(&ex.gs, 0), dptr(&sc.din, 0), dptr(&sc.y, 0), idx, es, m_, lay.inner as i32, H as i32))?,
            ExpFmt::Mxfp8 => ink2_launch!(dev, dn, ((H / 64) as u32, emax, 1), (128, 1, 1), 0,
                (dptr(&ex.w2, 0), dptr(&ex.s2, 0), dptr(&sc.din, 0), dptr(&sc.y, 0), idx, es, m_, lay.inner as i32, H as i32))?,
        }
    }
    Ok(())
}

/// Prefill routing (after `router` with m > 16): counts, compact order, tile list.
pub fn route_prefill(dev: &Arc<CudaDevice>, sc: &MoeScratch, m: usize) -> Result<()> {
    ink2_launch!(dev, "ink2_moe_pf_route", (E as u32, 1, 1), (256, 1, 1), 0,
                 (dptr(&sc.ids, 0), (m * TOPK) as i32, dptr(&sc.cnt, 0), dptr(&sc.offs, 0), dptr(&sc.rowtok, 0),
                  dptr(&sc.rowidx, 0), dptr(&sc.tiles, 0), dptr(&sc.ntiles, 0)))
}

pub fn experts_prefill(dev: &Arc<CudaDevice>, lay: &MoeLayerDev, sc: &MoeScratch, m: usize, which: u8) -> Result<()> {
    let mt = max_tiles(m) as u32;
    let ex = &lay.ex;
    let (tl, nt, of, cn) = (dptr(&sc.tiles, 0), dptr(&sc.ntiles, 0), dptr(&sc.offs, 0), dptr(&sc.cnt, 0));
    if which & 1 != 0 {
        match ex.fmt {
            ExpFmt::Nvfp4 => ink2_launch!(dev, "ink2_nv4_gu_pf", (((2 * lay.inner) / 128) as u32, mt, 1), (256, 1, 1), 0,
                (dptr(&ex.w13, 0), dptr(&ex.s13, 0), dptr(&ex.gs, 0), dptr(&sc.xh, 0), dptr(&sc.rowtok, 0), dptr(&sc.din, 0),
                 tl, nt, of, cn, H as i32, (2 * lay.inner) as i32))?,
            ExpFmt::Mxfp8 => ink2_launch!(dev, "ink2_mx8_gu_pf", (((2 * lay.inner) / 128) as u32, mt, 1), (256, 1, 1), 0,
                (dptr(&ex.w13, 0), dptr(&ex.s13, 0), dptr(&sc.xh, 0), dptr(&sc.rowtok, 0), dptr(&sc.din, 0),
                 tl, nt, of, cn, H as i32, (2 * lay.inner) as i32))?,
        }
    }
    if which & 2 != 0 {
        match ex.fmt {
            ExpFmt::Nvfp4 => ink2_launch!(dev, "ink2_nv4_dn_pf", ((H / 128) as u32, mt, 1), (256, 1, 1), 0,
                (dptr(&ex.w2, 0), dptr(&ex.s2, 0), dptr(&ex.gs, 0), dptr(&sc.din, 0), dptr(&sc.y, 0),
                 tl, nt, of, cn, lay.inner as i32, H as i32))?,
            ExpFmt::Mxfp8 => ink2_launch!(dev, "ink2_mx8_dn_pf", ((H / 128) as u32, mt, 1), (256, 1, 1), 0,
                (dptr(&ex.w2, 0), dptr(&ex.s2, 0), dptr(&sc.din, 0), dptr(&sc.y, 0),
                 tl, nt, of, cn, lay.inner as i32, H as i32))?,
        }
    }
    Ok(())
}

/// S7 verify widths (17..=48 rows, after `router` + `route_prefill`): each expert computes only the rows that picked
/// it, 16-row groups sharing every weight fragment (kernels/gpu_inkling2/66_experts_verify.cuh). Same compact row
/// layout as `experts_prefill` (Din / Y [8M][..], combine via rowidx); bit-identical rows.
pub const VERIFY_MAX_M: usize = 48;
pub fn experts_verify(dev: &Arc<CudaDevice>, lay: &MoeLayerDev, sc: &MoeScratch, m: usize, which: u8) -> Result<()> {
    ensure!(m <= VERIFY_MAX_M, "inkling2 MoE verify: m {m} > {VERIFY_MAX_M}");
    let ex = &lay.ex;
    let (rt, cn, of) = (dptr(&sc.rowtok, 0), dptr(&sc.cnt, 0), dptr(&sc.offs, 0));
    if which & 1 != 0 {
        match ex.fmt {
            ExpFmt::Nvfp4 => ink2_launch!(dev, "ink2_nv4_gu_vt", (((2 * lay.inner) / 64) as u32, E as u32, 1), (128, 1, 1), 0,
                (dptr(&ex.w13, 0), dptr(&ex.s13, 0), dptr(&ex.gs, 0), dptr(&sc.xh, 0), rt, dptr(&sc.din, 0), cn, of, H as i32, (2 * lay.inner) as i32))?,
            ExpFmt::Mxfp8 => ink2_launch!(dev, "ink2_mx8_gu_vt", (((2 * lay.inner) / 64) as u32, E as u32, 1), (128, 1, 1), 0,
                (dptr(&ex.w13, 0), dptr(&ex.s13, 0), dptr(&sc.xh, 0), rt, dptr(&sc.din, 0), cn, of, H as i32, (2 * lay.inner) as i32))?,
        }
    }
    if which & 2 != 0 {
        match ex.fmt {
            ExpFmt::Nvfp4 => ink2_launch!(dev, "ink2_nv4_dn_vt", ((H / 64) as u32, E as u32, 1), (128, 1, 1), 0,
                (dptr(&ex.w2, 0), dptr(&ex.s2, 0), dptr(&ex.gs, 0), dptr(&sc.din, 0), dptr(&sc.y, 0), cn, of, lay.inner as i32, H as i32))?,
            ExpFmt::Mxfp8 => ink2_launch!(dev, "ink2_mx8_dn_vt", ((H / 64) as u32, E as u32, 1), (128, 1, 1), 0,
                (dptr(&ex.w2, 0), dptr(&ex.s2, 0), dptr(&sc.din, 0), dptr(&sc.y, 0), cn, of, lay.inner as i32, H as i32))?,
        }
    }
    Ok(())
}

thread_local! {
    /// S7 A/B switch: router logits with <= 4 tokens per CTA row at decode / verify widths (default) or the S4 policy.
    pub static ROUTER_SMALL: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    /// S7: route 17..=48-row calls through the verify kernels (default) instead of the prefill tiles (probe A/B).
    pub static VERIFY_KERNELS: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

pub fn combine(dev: &Arc<CudaDevice>, sc: &MoeScratch, out: u64, m: usize) -> Result<()> {
    ink2_launch!(dev, "ink2_moe_combine", (m as u32, 1, 1), (256, 1, 1), 0,
                 (dptr(&sc.y, 0), dptr(&sc.rowidx, 0), dptr(&sc.wts, 0), out, H as i32))
}

/// The whole MoE block: x [m][H] fp32 -> out [m][H] fp32 (stream-ordered, no host sync).
pub fn forward(dev: &Arc<CudaDevice>, lay: &MoeLayerDev, sc: &MoeScratch, x: u64, out: u64, m: usize) -> Result<()> {
    router(dev, lay, sc, x, m)?;
    if m <= 16 {
        experts_decode(dev, lay, sc, m, 3)?;
    } else if m <= VERIFY_MAX_M && VERIFY_KERNELS.with(|c| c.get()) && super::k1::verify_class() {
        route_prefill(dev, sc, m)?;
        experts_verify(dev, lay, sc, m, 3)?;
    } else {
        route_prefill(dev, sc, m)?;
        experts_prefill(dev, lay, sc, m, 3)?;
    }
    combine(dev, sc, out, m)
}
