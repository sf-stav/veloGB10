//! Leg Q1: the expert-activation precision switch, `--inkling2-expert-act a16|native` (default `a16`).
//!
//!   * `a16` (default): K2's W4A16 / W8A16 experts — the exact checkpoint weights against f16 activations (what the
//!     transformers reference computes, up to its bf16 roundings), untouched by this module.
//!   * `native`: the pack-declared activation quantisation (`hf_quant_config.json` `required_activation_quantization`,
//!     enforced by the external reference SGLang): NVFP4 activations (block 16, e4m3 scales, per-token amax,
//!     `fouroversix_mae`) on layers 0-19 and MXFP8 activations (block 32, ue8m0) on 20-23, on the sm_121a
//!     block-scaled tensor cores (`mma .kind::mxf4nvf4` / `.kind::mxf8f6f4`), with the reference's expert-pipeline
//!     roundings (bf16 GEMM outputs, gamma before the down-projection quantiser, bf16 expert rows). The router, the
//!     MoE input / output (fp32) and everything outside the experts are the same in both modes.
//!
//! Files: [`cpu`] the quantisers' CPU model (bit-exact vs the reference's own kernels, fixture-gated);
//! [`layout`] the native weight layouts; [`probe`] `--probe-inkling2-actq` (GPU vs CPU / fixture / f64 GEMM checks,
//! timings) and `--probe-inkling2-logits` (teacher-forced logit dumps for the fidelity study).
//! Kernels: `kernels/gpu_inkling2/c0-c3_*.cuh`.

pub mod cpu;
pub mod layout;
pub mod probe;

use crate::ink2_launch;
use crate::inkling2::gpu::dptr;
use crate::inkling2::gpu::moe::{self, ExpertsDev, MoeLayerDev, MoeScratch, E, H, INTER, TOPK};
use crate::inkling2::pack::{ExpFmt, Inkling2Pack, P};
use anyhow::{bail, ensure, Result};
use cudarc::driver::{CudaDevice, CudaSlice};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

/// The expert activation precision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertAct { A16, Native }

static MODE: AtomicU8 = AtomicU8::new(0);
/// Prefill expert tiles: 64 compact rows per CTA (0, default) or 128 (1; an expert's tile pair, one CTA per SM).
/// Measured on the loaded .11 (probe): NVFP4 64 = 1.22x a16, 128 = 1.00x; MXFP8 1.33x / 1.37x: 64 kept. The probe times both.
pub static PF_TILE128: AtomicU8 = AtomicU8::new(0);

/// Parse `--inkling2-expert-act a16|native` (absent = a16) and make it the process mode (read by the model loader).
pub fn set_mode_from_args(args: &[String]) -> Result<ExpertAct> {
    let v = args.iter().position(|a| a == "--inkling2-expert-act").and_then(|i| args.get(i + 1)).map(|s| s.as_str());
    let m = match v {
        None | Some("a16") => ExpertAct::A16,
        Some("native") => ExpertAct::Native,
        Some(o) => bail!("--inkling2-expert-act {o}: expected a16 (16-bit activations, default) or native (the pack-declared NVFP4 / MXFP8 activations)"),
    };
    MODE.store((m == ExpertAct::Native) as u8, Ordering::Relaxed);
    if m == ExpertAct::Native {
        crate::boot_report::notice("--inkling2-expert-act native: expert activations quantised as the pack declares (NVFP4 'four over six' L0-19, MXFP8 L20-23) on the fp4/fp8 tensor cores; differs from the 16-bit transformers reference (see PLAN/INKLING2_Q1_REPORT.md)");
    }
    Ok(m)
}

pub fn mode() -> ExpertAct { if MODE.load(Ordering::Relaxed) == 1 { ExpertAct::Native } else { ExpertAct::A16 } }

fn bf16_bits(st: &crate::fp4l::pack::StDir, name: &str) -> Result<Vec<u16>> {
    let b = st.raw(name)?;
    Ok(b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
}

/// One layer's router (as K2) + experts in the native layouts (`ExpertsDev.gs` = [E][4] {w13 amax, w13 den, w2 amax,
/// w2 den} for NVFP4, unused for MXFP8).
pub fn load_layer(dev: &Arc<CudaDevice>, pack: &Inkling2Pack, l: usize) -> Result<MoeLayerDev> {
    load_layer_range(dev, pack, l, 0, INTER)
}

/// Drop 1 (#3a) + Drop 3 (#3b): inner-channel sharded native expert loader (`[c0, c1)` whole 64-blocks for TP1/TP2/TP4/TP8).
pub fn load_layer_range(dev: &Arc<CudaDevice>, pack: &Inkling2Pack, l: usize, c0: usize, c1: usize) -> Result<MoeLayerDev> {
    let c = &pack.cfg;
    ensure!(c0 < c1 && c1 <= INTER && c0 % 64 == 0 && c1 % 64 == 0,
            "inkling2 native experts: inner slice [{c0}, {c1}) not whole 64-blocks of {INTER}");
    let inner = c1 - c0;
    let full = inner == INTER;
    ensure!(c.hidden == H && c.inter == INTER && c.n_experts == E && c.topk == TOPK, "inkling2 native experts: shape");
    let st = &pack.st;
    let p = format!("{P}layers.{l}.mlp.");
    let router_w = dev.htod_copy(bf16_bits(st, &format!("{p}gate.weight"))?)?;
    let router_b = dev.htod_copy(pack.plain(&format!("{p}gate.bias"))?)?;
    let fmt = pack.exp_fmt[l];
    let mut gsv = vec![1f32; 4 * E];
    let cut = |proj: &str, b: &[u8], rb: usize, den: usize| -> Vec<u8> {
        if full { return b.to_vec(); }
        if proj == "w13" { b[2 * c0 * rb..2 * c1 * rb].to_vec() } else { crate::inkling2::tp::shard::byte_cols(b, rb, c0 / den, c1 / den) }
    };
    let mut out = Vec::new();
    for (pi, (proj, n, k, on, ok)) in [("w13", 2 * INTER, H, 2 * inner, H), ("w2", H, INTER, H, inner)].into_iter().enumerate() {
        let b = format!("{p}experts.{proj}_weight");
        let codes = st.raw(&b)?;
        let scales = st.raw(&format!("{b}_scale"))?;
        let (ce, se) = match fmt { ExpFmt::Nvfp4 => (n * k / 2, n * k / 16), ExpFmt::Mxfp8 => (n * k, n * k / 32) };
        let (oce, ose) = match fmt { ExpFmt::Nvfp4 => (on * ok / 2, on * ok / 16), ExpFmt::Mxfp8 => (on * ok, (on / 16) * ok.div_ceil(128) * 64) };
        if fmt == ExpFmt::Nvfp4 {
            let amax = pack.plain(&format!("{b}_amax"))?;
            let den = pack.plain(&format!("{b}_outer_scale_denominator"))?[0];
            for e in 0..E { gsv[4 * e + 2 * pi] = amax[e]; gsv[4 * e + 2 * pi + 1] = den; }
        } else {
            ensure!(!scales.contains(&0xFF), "{b}_scale: a UE8M0 NaN scale (0xFF)");
        }
        let mut w = vec![0u8; E * oce];
        let mut s = vec![0u8; E * ose];
        let cut = &cut;
        std::thread::scope(|sco| {
            for (e, (wc, sc)) in w.chunks_mut(oce).zip(s.chunks_mut(ose)).enumerate() {
                let (codes, scales) = (&codes, &scales);
                sco.spawn(move || {
                    let (rb, sb, rden, sden) = match fmt {
                        ExpFmt::Nvfp4 => (k / 2, k / 16, 2usize, 16usize),
                        ExpFmt::Mxfp8 => (k, k / 32, 1usize, 32usize),
                    };
                    let (orb, osb) = match fmt { ExpFmt::Nvfp4 => (ok / 2, ok / 16), ExpFmt::Mxfp8 => (ok, ok / 32) };
                    let mut cc = cut(proj, &codes[e * ce..(e + 1) * ce], rb, rden);
                    let mut ss = cut(proj, &scales[e * se..(e + 1) * se], sb, sden);
                    if proj == "w13" {
                        cc = crate::inkling2::gpu::layout::w13_tile_rows(&cc, on, orb);
                        ss = crate::inkling2::gpu::layout::w13_tile_rows(&ss, on, osb);
                    }
                    match fmt {
                        ExpFmt::Nvfp4 => layout::nv4_native_into(&cc, &ss, on, ok, wc, sc),
                        ExpFmt::Mxfp8 => layout::mx8_native_into(&cc, &ss, on, ok, wc, sc),
                    }
                });
            }
        });
        out.push((dev.htod_copy(w)?, dev.htod_copy(s)?));
    }
    let (w2, s2) = out.pop().unwrap();
    let (w13, s13) = out.pop().unwrap();
    Ok(MoeLayerDev { layer: l, router_w, router_b, ex: ExpertsDev { fmt, w13, s13, w2, s2, gs: dev.htod_copy(gsv)? }, inner })
}

/// Scratch of the native path (on top of K2's `MoeScratch`, whose router outputs / route / Y it reuses).
pub struct ActqScratch {
    pub cap_m: usize,
    pub xc: CudaSlice<u8>,   // token codes [cap_m][<= 2304]
    pub xs: CudaSlice<u8>,   // token scales [cap_m][<= 144]
    pub xa: CudaSlice<f32>,  // token amax [cap_m]
    pub act: CudaSlice<u16>, // bf16 down input [rows][INTER]
    pub ac: CudaSlice<u8>,   // its codes [rows][<= INTER]
    pub asc: CudaSlice<u8>,  // its scales [rows][<= 72]
    pub aa: CudaSlice<f32>,  // its amax [rows]
    pub gam: CudaSlice<f32>, // routing weight per expert row [rows]
    pub ones: CudaSlice<f32>,// [cap_m][8] = 1
    pub gate_cap: f32,
    pub up_cap: f32,
}

impl ActqScratch {
    pub fn new(dev: &Arc<CudaDevice>, cap_m: usize, gate_cap: f32, up_cap: f32) -> Result<ActqScratch> {
        let rows = (TOPK * cap_m).max(E * 16);
        let z8 = |n: usize| -> Result<CudaSlice<u8>> { let mut x = dev.alloc_zeros::<u8>(n)?; dev.memset_zeros(&mut x)?; Ok(x) };
        let zf = |n: usize| -> Result<CudaSlice<f32>> { let mut x = dev.alloc_zeros::<f32>(n)?; dev.memset_zeros(&mut x)?; Ok(x) };
        let mut act = dev.alloc_zeros::<u16>(rows * INTER)?;
        dev.memset_zeros(&mut act)?;
        let s = ActqScratch {
            cap_m, xc: z8(cap_m * H)?, xs: z8(cap_m * H / 16)?, xa: zf(cap_m)?, act,
            ac: z8(rows * INTER)?, asc: z8(rows * INTER / 16)?, aa: zf(rows)?, gam: zf(rows)?,
            ones: dev.htod_sync_copy(&vec![1f32; TOPK * cap_m])?, gate_cap, up_cap,
        };
        dev.synchronize()?;
        Ok(s)
    }
}

/// GB10 SM count (the reference's quantiser grid: min(rows, 4 x SMs)).
const SMS: usize = 48;

/// Quantise `rows` rows of the expert input / act (`f32_in`: fp32 x rounded to bf16; else bf16) for `fmt`.
/// `nrows_dev` (decode): rows = *nrows_dev * nmul on the device, `rows` = the upper bound for the grid.
#[allow(clippy::too_many_arguments)]
pub fn quant(dev: &Arc<CudaDevice>, fmt: ExpFmt, width: usize, f32_in: bool, x: u64, codes: u64, scales: u64, amax: u64,
             rows: usize, nrows_dev: u64, nmul: usize) -> Result<()> {
    let grid = (rows.min(4 * SMS).max(1) as u32, 1, 1);
    let name = match (fmt, width, f32_in) {
        (ExpFmt::Nvfp4, 2304, true) => "ink2q_nv4_quant_f2304",
        (ExpFmt::Nvfp4, 2304, false) => "ink2q_nv4_quant_b2304",
        (ExpFmt::Nvfp4, 1152, false) => "ink2q_nv4_quant_b1152",
        (ExpFmt::Nvfp4, 640, false) => "ink2q_nv4_quant_b640",
        (ExpFmt::Nvfp4, 576, false) => "ink2q_nv4_quant_b576",
        (ExpFmt::Nvfp4, 512, false) => "ink2q_nv4_quant_b512",
        (ExpFmt::Nvfp4, 384, false) => "ink2q_nv4_quant_b384",
        (ExpFmt::Nvfp4, 320, false) => "ink2q_nv4_quant_b320",
        (ExpFmt::Nvfp4, 256, false) => "ink2q_nv4_quant_b256",
        (ExpFmt::Nvfp4, 192, false) => "ink2q_nv4_quant_b192",
        (ExpFmt::Nvfp4, 128, false) => "ink2q_nv4_quant_b128",
        (ExpFmt::Mxfp8, 2304, true) => "ink2q_mx8_quant_f2304",
        (ExpFmt::Mxfp8, 2304, false) => "ink2q_mx8_quant_b2304",
        (ExpFmt::Mxfp8, 1152, false) => "ink2q_mx8_quant_b1152",
        (ExpFmt::Mxfp8, 640, false) => "ink2q_mx8_quant_b640",
        (ExpFmt::Mxfp8, 576, false) => "ink2q_mx8_quant_b576",
        (ExpFmt::Mxfp8, 512, false) => "ink2q_mx8_quant_b512",
        (ExpFmt::Mxfp8, 384, false) => "ink2q_mx8_quant_b384",
        (ExpFmt::Mxfp8, 320, false) => "ink2q_mx8_quant_b320",
        (ExpFmt::Mxfp8, 256, false) => "ink2q_mx8_quant_b256",
        (ExpFmt::Mxfp8, 192, false) => "ink2q_mx8_quant_b192",
        (ExpFmt::Mxfp8, 128, false) => "ink2q_mx8_quant_b128",
        o => bail!("ink2q quant: no kernel for {o:?}"),
    };
    let rows_arg = if nrows_dev != 0 { 0 } else { rows as i32 };
    match fmt {
        ExpFmt::Nvfp4 => {
            let thr = ((width / 16 * 4 + 31) / 32 * 32) as u32;
            ink2_launch!(dev, name, grid, (thr, 1, 1), 0, (x, codes, scales, amax, rows_arg, nrows_dev, nmul as i32))
        }
        ExpFmt::Mxfp8 => {
            let thr = ((width / 16 + 31) / 32 * 32) as u32;
            ink2_launch!(dev, name, grid, (thr, 1, 1), 0, (x, codes, scales, rows_arg, nrows_dev, nmul as i32))
        }
    }
}

/// The native MoE block: x [m][H] fp32 -> out [m][H] fp32 (stream-ordered).
pub fn forward(dev: &Arc<CudaDevice>, lay: &MoeLayerDev, sc: &MoeScratch, aq: &ActqScratch, x: u64, out: u64, m: usize) -> Result<()> {
    ensure!(m >= 1 && m <= sc.cap_m && m <= aq.cap_m, "inkling2 native MoE: m {m}");
    let ex = &lay.ex;
    let fmt = ex.fmt;
    let nv = fmt == ExpFmt::Nvfp4;
    let inner = lay.inner;
    let (gc, uc) = (aq.gate_cap, aq.up_cap);
    moe::router(dev, lay, sc, x, m)?;
    quant(dev, fmt, H, true, x, dptr(&aq.xc, 0), dptr(&aq.xs, 0), dptr(&aq.xa, 0), m, 0, 0)?;
    let (xc, xs, xa, ac, asc, aa, gam, act, y) = (dptr(&aq.xc, 0), dptr(&aq.xs, 0), dptr(&aq.xa, 0), dptr(&aq.ac, 0),
        dptr(&aq.asc, 0), dptr(&aq.aa, 0), dptr(&aq.gam, 0), dptr(&aq.act, 0), dptr(&sc.y, 0));
    let (w13, s13, w2, s2, gs) = (dptr(&ex.w13, 0), dptr(&ex.s13, 0), dptr(&ex.w2, 0), dptr(&ex.s2, 0), dptr(&ex.gs, 0));
    let (rowidx, wts) = (dptr(&sc.rowidx, 0), dptr(&sc.wts, 0));
    if m <= 16 {
        let emax = (TOPK * m).min(E);
        let (idx, es, m_) = (dptr(&sc.idxmap, 0), dptr(&sc.esel, 0), m as i32);
        ink2_launch!(dev, "ink2q_gamma", (1, 1, 1), (256, 1, 1), 0, (rowidx, wts, gam, (m * TOPK) as i32, (emax * m) as i32))?;
        let gg = ((2 * inner / 64) as u32, emax as u32, 1);
        if nv {
            ink2_launch!(dev, "ink2q_nv4_gu_dec", gg, (128, 1, 1), 0,
                         (w13, s13, gs, xc, xs, xa, gam, act, idx, es, m_, H as i32, (2 * inner) as i32, gc, uc))?;
        } else {
            ink2_launch!(dev, "ink2q_mx8_gu_dec", gg, (128, 1, 1), 0,
                         (w13, s13, xc, xs, gam, act, idx, es, m_, H as i32, (2 * inner) as i32, gc, uc))?;
        }
        quant(dev, fmt, inner, false, act, ac, asc, aa, emax * m, es, m)?;
        let gd = ((H / 64) as u32, emax as u32, 1);
        if nv {
            ink2_launch!(dev, "ink2q_nv4_dn_dec", gd, (128, 1, 1), 0, (w2, s2, gs, ac, asc, aa, y, idx, es, m_, inner as i32, H as i32))?;
        } else {
            ink2_launch!(dev, "ink2q_mx8_dn_dec", gd, (128, 1, 1), 0, (w2, s2, ac, asc, y, idx, es, m_, inner as i32, H as i32))?;
        }
    } else {
        moe::route_prefill(dev, sc, m)?;
        let mt = moe::max_tiles(m) as u32;
        let (tl, nt, of, cn, rt) = (dptr(&sc.tiles, 0), dptr(&sc.ntiles, 0), dptr(&sc.offs, 0), dptr(&sc.cnt, 0), dptr(&sc.rowtok, 0));
        ink2_launch!(dev, "ink2q_gamma_pf", (((m * TOPK).div_ceil(256)) as u32, 1, 1), (256, 1, 1), 0, (rowidx, wts, gam, (m * TOPK) as i32))?;
        let t128 = PF_TILE128.load(Ordering::Relaxed) == 1;
        let (gu, dn) = match (nv, t128) {
            (true, false) => ("ink2q_nv4_gu_pf", "ink2q_nv4_dn_pf"),
            (false, false) => ("ink2q_mx8_gu_pf", "ink2q_mx8_dn_pf"),
            (true, true) => ("ink2q_nv4_gu_pf128", "ink2q_nv4_dn_pf128"),
            (false, true) => ("ink2q_mx8_gu_pf128", "ink2q_mx8_dn_pf128"),
        };
        ink2_launch!(dev, gu, (((2 * inner) / 128) as u32, mt, 1), (256, 1, 1), 0,
                     (w13, s13, gs, xc, xs, xa, rt, gam, act, 0u64, tl, nt, of, cn, H as i32, (2 * inner) as i32, gc, uc))?;
        quant(dev, fmt, inner, false, act, ac, asc, aa, m * TOPK, 0, 0)?;
        ink2_launch!(dev, dn, ((H / 128) as u32, mt, 1), (256, 1, 1), 0,
                     (w2, s2, gs, ac, asc, aa, 0u64, 0u64, 0u64, y, tl, nt, of, cn, inner as i32, H as i32, 0f32, 0f32))?;
    }
    ink2_launch!(dev, "ink2_moe_combine", (m as u32, 1, 1), (256, 1, 1), 0, (y, rowidx, dptr(&aq.ones, 0), out, H as i32))
}

/// Drop 2 (#2): native prefill MoE dispatch after BF16 `xb` (`[m][H]`) + `sc.ids` + `sc.wts` have been all-gathered.
/// Bit-identical to `forward` (`ink2q_*_quant_b2304` on bf16(x) == `ink2q_*_quant_f2304` on f32 x).
pub fn forward_routed_bf16(dev: &Arc<CudaDevice>, lay: &MoeLayerDev, sc: &MoeScratch, aq: &ActqScratch, xb: u64, out: u64, m: usize) -> Result<()> {
    ensure!(m > 16 && m <= sc.cap_m && m <= aq.cap_m, "inkling2 native SP MoE: m {m}");
    let ex = &lay.ex;
    let fmt = ex.fmt;
    let nv = fmt == ExpFmt::Nvfp4;
    let inner = lay.inner;
    let (gc, uc) = (aq.gate_cap, aq.up_cap);
    quant(dev, fmt, H, false, xb, dptr(&aq.xc, 0), dptr(&aq.xs, 0), dptr(&aq.xa, 0), m, 0, 0)?;
    let (xc, xs, xa, ac, asc, aa, gam, act, y) = (dptr(&aq.xc, 0), dptr(&aq.xs, 0), dptr(&aq.xa, 0), dptr(&aq.ac, 0),
        dptr(&aq.asc, 0), dptr(&aq.aa, 0), dptr(&aq.gam, 0), dptr(&aq.act, 0), dptr(&sc.y, 0));
    let (w13, s13, w2, s2, gs) = (dptr(&ex.w13, 0), dptr(&ex.s13, 0), dptr(&ex.w2, 0), dptr(&ex.s2, 0), dptr(&ex.gs, 0));
    let (rowidx, wts) = (dptr(&sc.rowidx, 0), dptr(&sc.wts, 0));
    moe::route_prefill(dev, sc, m)?;
    let mt = moe::max_tiles(m) as u32;
    let (tl, nt, of, cn, rt) = (dptr(&sc.tiles, 0), dptr(&sc.ntiles, 0), dptr(&sc.offs, 0), dptr(&sc.cnt, 0), dptr(&sc.rowtok, 0));
    ink2_launch!(dev, "ink2q_gamma_pf", (((m * TOPK).div_ceil(256)) as u32, 1, 1), (256, 1, 1), 0, (rowidx, wts, gam, (m * TOPK) as i32))?;
    let t128 = PF_TILE128.load(Ordering::Relaxed) == 1;
    let (gu, dn) = match (nv, t128) {
        (true, false) => ("ink2q_nv4_gu_pf", "ink2q_nv4_dn_pf"),
        (false, false) => ("ink2q_mx8_gu_pf", "ink2q_mx8_dn_pf"),
        (true, true) => ("ink2q_nv4_gu_pf128", "ink2q_nv4_dn_pf128"),
        (false, true) => ("ink2q_mx8_gu_pf128", "ink2q_mx8_dn_pf128"),
    };
    ink2_launch!(dev, gu, (((2 * inner) / 128) as u32, mt, 1), (256, 1, 1), 0,
                 (w13, s13, gs, xc, xs, xa, rt, gam, act, 0u64, tl, nt, of, cn, H as i32, (2 * inner) as i32, gc, uc))?;
    quant(dev, fmt, inner, false, act, ac, asc, aa, m * TOPK, 0, 0)?;
    ink2_launch!(dev, dn, ((H / 128) as u32, mt, 1), (256, 1, 1), 0,
                 (w2, s2, gs, ac, asc, aa, 0u64, 0u64, 0u64, y, tl, nt, of, cn, inner as i32, H as i32, 0f32, 0f32))?;
    ink2_launch!(dev, "ink2_moe_combine", (m as u32, 1, 1), (256, 1, 1), 0, (y, rowidx, dptr(&aq.ones, 0), out, H as i32))
}
