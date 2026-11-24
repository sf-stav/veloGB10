//! KE2: host side of the prefill MoE (kernels/gpu_dsv41/16_exl3_moe_wide.cuh): > 32 rows, routed + shared expert in
//! one call: plan (3 launches), gate/up tiles (input transform fused, act by the second matrix of a pair), down tiles,
//! the ordered combine. PK2: gate/up tiles are 64 rows x a block pair (512 threads, one transform for both blocks); the
//! standalone harness tools/dsv41_pk2/exl3_moe_wide_bench.cu mirrors this launch sequence (keep the two in step). Rows are bit-identical to the decode kernels' (the probe gates it). Calls above SUB_ROWS rows
//! run as consecutive SUB_ROWS-row chunks (each streams the experts once; rows do not interact).

use super::launch::{act_code, kernel, Kernel};
use super::moe::MoeTable;
use super::moe_dec::launch_ex;
use crate::dsv41::weights::{Act, ActDtype, Launch, MoeWeight, Route, Scratch};
use anyhow::{ensure, Context, Result};
use cudarc::driver::sys;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

/// Rows per prefill chunk (T0 §7.1: 16,384 at TP4 / TP8). Scratch scales with it (~2.7 GB at TP4, see `scratch_bytes`).
pub const SUB_ROWS: usize = 16384;
const RB: usize = 256;
const TM: usize = 128;
/// PK2: gate / up tiles are 64 rows x one block PAIR (512 threads); down tiles are 64 rows x DN_NB output blocks
/// (256 x DN_NB threads). TM (128) is only the KE2 geometry constant the device reports.
const TMG: usize = 64;
const GU_THREADS: u32 = 512;
const DN_NB: usize = 1;
const MAXSL: usize = 1025;
const META_INTS: usize = 16 + 6 * MAXSL + 2 + 1024;
const ASTG: usize = TM * 256;
const ASTG_G: usize = TMG * 256;

/// kernels/gpu_dsv41/16_exl3_moe_wide.cuh `D41xMpArgs` (size checked against the device at first use).
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct MpArgs {
    pub ex: u64, pub sh: u64, pub ids: u64, pub wts: u64,
    pub x: u64, pub xcode: i32, pub ldx: i32,
    pub hist: u64, pub lrank: u64, pub pos: u64, pub prow: u64, pub pw: u64, pub meta: u64, pub pcnt: u64,
    pub gtab: u64, pub dtab: u64,
    pub graw: u64, pub xh2: u64, pub h2: u64,
    pub out: u64, pub ldo: i32,
    pub limit: f32, pub limit_sh: f32,
    pub n_experts: i32, pub rows: i32, pub topk: i32, pub dim: i32, pub inter_ld: i32, pub kmax: i32, pub kmin: i32, pub nrb: i32, pub ncnt: i32,
    /// PK3: 1 = bf16 per-pair contributions (`--dsv41-prefill-bf16-partials on`), 0 = the exact f32 h2 (default)
    pub h2bf: i32, pub pad_: i32,
}

pub fn check_geometry() -> Result<()> {
    static OK: OnceLock<()> = OnceLock::new();
    if OK.get().is_some() { return Ok(()); }
    let k = kernel("d41x_mp_sizes")?;
    let mut h = [0i32; 8];
    unsafe {
        let mut d: sys::CUdeviceptr = 0;
        ensure!(sys::cuMemAlloc_v2(&mut d, 32) == sys::CUresult::CUDA_SUCCESS, "cuMemAlloc (mp sizes)");
        let mut st: sys::CUstream = std::ptr::null_mut();
        ensure!(sys::cuStreamCreate(&mut st, 0) == sys::CUresult::CUDA_SUCCESS, "cuStreamCreate (mp sizes)");
        let mut p = d as u64;
        let mut params = [&mut p as *mut u64 as *mut c_void];
        ensure!(sys::cuLaunchKernel(k.0, 1, 1, 1, 32, 1, 1, 0, st, params.as_mut_ptr(), std::ptr::null_mut()) == sys::CUresult::CUDA_SUCCESS, "launch d41x_mp_sizes");
        ensure!(sys::cuStreamSynchronize(st) == sys::CUresult::CUDA_SUCCESS, "sync (mp sizes)");
        ensure!(sys::cuMemcpyDtoH_v2(h.as_mut_ptr() as *mut c_void, d, 32) == sys::CUresult::CUDA_SUCCESS, "read mp sizes");
        sys::cuStreamDestroy_v2(st);
        sys::cuMemFree_v2(d);
    }
    ensure!(h[0] as usize == std::mem::size_of::<MpArgs>(), "D41xMpArgs is {} B on the device, {} B in Rust", h[0], std::mem::size_of::<MpArgs>());
    ensure!(h[1] as usize == META_INTS && h[2] as usize == ASTG && h[3] as usize == TM && h[4] as usize == RB && h[5] as usize == TMG && h[6] as usize == ASTG_G
            && h[7] as usize == DN_NB, "prefill MoE geometry mismatch {:?}", &h[..8]);
    let _ = OK.set(());
    Ok(())
}

pub fn eligible(t: &MoeTable, ts: Option<&MoeTable>, w: &MoeWeight, sh: Option<&MoeWeight>) -> bool {
    t.int_k() && w.n_experts <= 1024 && w.top_k <= 8 && w.dim % 512 == 0 && w.inter % 128 == 0
        && ts.map_or(true, |s| s.int_k() && s.dim == t.dim)
        && sh.map_or(true, |s| s.n_experts == 1 && s.top_k == 1 && s.dim == w.dim)
}

fn a256(b: usize) -> usize { (b + 255) & !255 }
struct Lay { hist: usize, lrank: usize, pos: usize, prow: usize, pw: usize, meta: usize, pcnt: usize, gtab: usize, dtab: usize, graw: usize, xh2: usize, h2: usize, total: usize,
             inter_ld: usize, p: usize, ncnt: usize, gu_ub: usize, dn_ub: usize, nrb: usize }
fn layout(t: &MoeTable, ts: Option<&MoeTable>, rows: usize, topk: usize) -> Lay {
    let rows = rows.min(SUB_ROWS);
    let nb_max = t.nblk_max.max(ts.map_or(0, |s| s.nblk_max));
    let inter_ld = nb_max * 128;
    let pr = rows * topk;
    let p = pr + if ts.is_some() { rows } else { 0 };
    let nrb = rows.div_ceil(RB);
    // sum over slots of ceil(cnt / tile rows) <= ceil(p / tile rows) + slots; p counts the shared expert's rows too (PK2: KE2
    // bounded it by the routed pairs only, safe there by slack, short at 64-row tiles when every expert holds 9 blocks)
    let tiles_rows_g = p.div_ceil(TMG) + t.n_experts + 2;
    let ncnt = tiles_rows_g * nb_max.div_ceil(2);                // act-pair counters: (64-row tile, block pair)
    let (gu_ub, dn_ub) = (2 * ncnt, tiles_rows_g * (t.dim / (128 * DN_NB)));
    let mut o = 0usize;
    let mut take = |b: usize| { let s = o; o += a256(b); s };
    let hist = take(nrb * t.n_experts * 4);
    let lrank = take(rows * topk * 2);
    let pos = take(rows * topk * 4);
    let prow = take(p * 4);
    let pw = take(p * 4);
    let meta = take(META_INTS * 4);
    let pcnt = take(ncnt * 4);
    let gtab = take(gu_ub * 4);
    let dtab = take(dn_ub * 4);
    let graw = take(p * 2 * inter_ld * 4);
    let xh2 = take(p * inter_ld * 2);
    let h2 = take(p * t.dim * 4);
    Lay { hist, lrank, pos, prow, pw, meta, pcnt, gtab, dtab, graw, xh2, h2, total: o, inter_ld, p, ncnt, gu_ub, dn_ub, nrb }
}
pub fn scratch_bytes(t: &MoeTable, ts: Option<&MoeTable>, rows: usize, topk: usize) -> usize { layout(t, ts, rows, topk).total }

/// The > 48 KB dynamic smem opt-in. The attribute is ONE value per function: keep the largest ever asked (monotonic),
/// so a call with a smaller need never lowers it under a later, larger one.
fn opt_in(name: &str, smem: u32) -> Result<Kernel> {
    static SET: OnceLock<Mutex<HashMap<String, u32>>> = OnceLock::new();
    let k = kernel(name)?;
    let m = SET.get_or_init(|| Mutex::new(HashMap::new()));
    let mut g = m.lock().unwrap();
    let cur = g.get(name).copied().unwrap_or(0);   // SX1: opt in from the first request: a kernel with static smem cannot take 48 KB dynamic without it
    if smem > cur {
        let r = unsafe { sys::cuFuncSetAttribute(k.0, sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32) };
        ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuFuncSetAttribute({name}, smem {smem}): {r:?}");
        g.insert(name.to_string(), smem);
    }
    Ok(k)
}

/// y [rows, dim] f32 += routed (+ shared), rows > 32 (any count; chunks of SUB_ROWS). With `--dsv41-prefill-bf16-partials on`
/// (PK3, opt-in) every pair's contribution is stored as bf16 before the combine (changes bits; the scratch stays f32-sized).
#[allow(clippy::too_many_arguments)]
pub fn run(l: &Launch, t: &MoeTable, ts: Option<&MoeTable>, w: &MoeWeight, sh: Option<&MoeWeight>, x: &Act, route: &Route, y: &Act, s: Scratch) -> Result<()> {
    run_mode(l, t, ts, w, sh, x, route, y, s, crate::dsv41::weights::partials::on())
}

/// `run` with the partial precision given (the probes compare both modes in one process).
#[allow(clippy::too_many_arguments)]
pub fn run_mode(l: &Launch, t: &MoeTable, ts: Option<&MoeTable>, w: &MoeWeight, sh: Option<&MoeWeight>, x: &Act, route: &Route, y: &Act, s: Scratch,
                bf16_partials: bool) -> Result<()> {
    ensure!(y.dtype == ActDtype::F32 && x.cols == w.dim && y.cols == w.dim && y.rows == x.rows, "moe {}: y must be f32 [rows, dim]", w.layer);
    ensure!(route.top_k == w.top_k, "moe {}: route top_k {} != {}", w.layer, route.top_k, w.top_k);
    check_geometry()?;
    let xcode = act_code(x.dtype)?;
    let esz = |d: ActDtype| match d { ActDtype::F32 => 4u64, _ => 2 };
    // SX1: stages sized for the widest matrix of either table (w2 included); every tile reads its entry's own code
    let kmax = t.kmax().max(ts.map_or(0, |s| s.kmax())) as usize;
    let kmin = t.kmin().min(ts.map_or(i32::MAX, |s| s.kmin()));
    ensure!((2..=8).contains(&kmin) && kmax <= 8, "moe {}: prefill kernels need integer K 2..8, the call spans {kmin}..{kmax}", w.layer);
    let smem_g = (2 * ASTG_G + 2 * 2 * 2048 * kmax) as u32;
    let smem_d = (2 * ASTG_G + 2 * DN_NB * 2048 * kmax).max(64 * (128 * DN_NB + 8) * 4) as u32;
    // gate / up entries (PK3): f32 x -> d41x_mp_gu_x4 (every width); bf16 / f16 x -> d41x_mp_gu_x2 (K 2..4) and / or d41x_mp_gu_x2h
    // (K 5..8), the ones the call's widths need, both over the same tile table (each skips the other's tiles)
    // SX1: the gate / up entries the call's w1 / w3 widths need (x2: K 2..4, x2h: K 5..8; each skips the other's tiles);
    // the down kernel has a case for every K 2..8
    let (klo, khi) = (t.k13.0.min(ts.map_or(t.k13.0, |s| s.k13.0)), t.k13.1.max(ts.map_or(t.k13.1, |s| s.k13.1)));
    let mut gu_names: Vec<&'static str> = Vec::with_capacity(2);
    if xcode == 2 { gu_names.push("d41x_mp_gu_x4"); } else {
        if klo <= 4 { gu_names.push("d41x_mp_gu_x2"); }
        if khi >= 5 { gu_names.push("d41x_mp_gu_x2h"); }
    }
    let (k_p1, k_p2, k_p3) = (kernel("d41x_mp_plan1")?, kernel("d41x_mp_plan2")?, kernel("d41x_mp_plan3")?);
    let k_gus = gu_names.iter().map(|n| opt_in(n, smem_g)).collect::<Result<Vec<_>>>()?;
    let k_dn = opt_in("d41x_mp_dn", smem_d)?;
    let k_cb = kernel("d41x_mp_combine")?;
    let p1_smem = (t.n_experts * 32) as u32;
    let k_p1 = if p1_smem > 48 * 1024 { opt_in("d41x_mp_plan1", p1_smem)? } else { k_p1 };
    let mut done = 0usize;
    while done < x.rows {
        let n = (x.rows - done).min(SUB_ROWS);
        ensure!(n as u64 * x.ld as u64 <= u32::MAX as u64, "prefill MoE chunk {n} x ld {} exceeds the 32-bit row offsets", x.ld);
        let lay = layout(t, ts, n, w.top_k);
        ensure!(s.bytes >= lay.total, "moe {}: scratch {} < {}", w.layer, s.bytes, lay.total);
        let b = s.ptr.0;
        let at = |o: usize| b + o as u64;
        let mut a = MpArgs {
            ex: t.dev, sh: ts.map_or(0, |s| s.dev),
            ids: route.ids.0 + (done * w.top_k * 4) as u64, wts: route.weights.0 + (done * w.top_k * 4) as u64,
            x: x.ptr.0 + done as u64 * x.ld as u64 * esz(x.dtype), xcode, ldx: x.ld as i32,
            hist: at(lay.hist), lrank: at(lay.lrank), pos: at(lay.pos), prow: at(lay.prow), pw: at(lay.pw), meta: at(lay.meta), pcnt: at(lay.pcnt),
            gtab: at(lay.gtab), dtab: at(lay.dtab),
            graw: at(lay.graw), xh2: at(lay.xh2), h2: at(lay.h2),
            out: y.ptr.0 + done as u64 * y.ld as u64 * 4, ldo: y.ld as i32,
            limit: w.swiglu_limit, limit_sh: sh.map_or(0.0, |s| s.swiglu_limit),
            n_experts: w.n_experts as i32, rows: n as i32, topk: w.top_k as i32, dim: w.dim as i32, inter_ld: lay.inter_ld as i32,
            kmax: kmax as i32, kmin, nrb: lay.nrb as i32, ncnt: lay.ncnt as i32,
            h2bf: bf16_partials as i32, pad_: 0,
        };
        let _ = lay.p;
        let mut pp = [&mut a as *mut MpArgs as *mut c_void];
        launch_ex(&k_p1, (lay.nrb as u32, 1, 1), 256, p1_smem, l, &mut pp).with_context(|| format!("d41x_mp_plan1 grid {} smem {p1_smem}", lay.nrb))?;
        launch_ex(&k_p2, (1, 1, 1), 1024, 0, l, &mut pp).context("d41x_mp_plan2")?;
        launch_ex(&k_p3, (lay.nrb as u32, 1, 1), 256, 0, l, &mut pp).context("d41x_mp_plan3")?;
        for (k_gu, gu_name) in k_gus.iter().zip(&gu_names) {
            launch_ex(k_gu, (lay.gu_ub as u32, 1, 1), GU_THREADS, smem_g, l, &mut pp).with_context(|| format!("{gu_name} grid {} smem {smem_g}", lay.gu_ub))?;
        }
        launch_ex(&k_dn, (lay.dn_ub as u32, 1, 1), (256 * DN_NB) as u32, smem_d, l, &mut pp).with_context(|| format!("d41x_mp_dn grid {} smem {smem_d}", lay.dn_ub))?;
        launch_ex(&k_cb, (n as u32, (w.dim / 512) as u32, 1), 128, 0, l, &mut pp).context("d41x_mp_combine")?;
        done += n;
    }
    if bf16_partials { crate::dsv41::weights::partials::note_call("EXL3", x.rows); }
    Ok(())
}
