//! SL: host side of band 02-04 (`kernels/gpu_dsv41/02_fwd.cuh`) and the one place the forward dispatches a
//! linear to its backend (`linear`), counting every call for the engagement receipts.

use super::dev::Gpu;
use super::receipts;
use crate::dsv41::weights::exl3::Exl3Weights;
use crate::dsv41::weights::native::NativeWeights;
use crate::dsv41::weights::plain::PlainWeights;
use crate::dsv41::weights::{Accum, Act, BackendKind, DenseWeight, DevPtr, Dsv41Weights, MoeWeight, Route, RowClass, Scratch};
use anyhow::{ensure, Result};

/// Entry kernels of band 02-04 (the gpu_dsv41 loader collects them).
pub const FNS: &[&str] = &["d41f_rope_rows", "d41f_swiglu", "d41f_add_round", "d41f_rope_table"];

static EXL3: Exl3Weights = Exl3Weights;
static NATIVE: NativeWeights = NativeWeights;
static PLAIN: PlainWeights = PlainWeights;

pub fn backend(k: BackendKind) -> &'static dyn Dsv41Weights {
    match k { BackendKind::Exl3 => &EXL3, BackendKind::Native => &NATIVE, BackendKind::Plain => &PLAIN }
}

/// The scratch every linear call of a forward shares (sized once for the largest call).
pub struct Scr { pub ptr: u64, pub bytes: usize }

/// y (+)= x . W^T through the linear's backend; the served class is counted under `what`.
pub fn linear(g: &Gpu, what: &'static str, w: &(DenseWeight, BackendKind), x: &Act, y: &Act, acc: Accum, s: &Scr) -> Result<RowClass> {
    let b = backend(w.1);
    let need = b.scratch_bytes(&w.0, x.rows);
    ensure!(need <= s.bytes, "{}: scratch {} B < {need} B", w.0.key, s.bytes);
    let cls = b.dense(&g.l(), &w.0, x, y, acc, Scratch { ptr: DevPtr(s.ptr), bytes: s.bytes })?;
    receipts::count(what, w.1, entry_of(w.1, x.rows), x.rows, cls);
    Ok(cls)
}

/// Routed experts through the backend (y f32 += ...).
pub fn moe(g: &Gpu, w: &(MoeWeight, BackendKind), x: &Act, r: &Route, y: &Act, s: &Scr) -> Result<RowClass> {
    let b = backend(w.1);
    let cls = b.moe(&g.l(), &w.0, x, r, y, Scratch { ptr: DevPtr(s.ptr), bytes: s.bytes })?;
    let entry = match w.1 {
        BackendKind::Exl3 => if x.rows <= 32 { "d41x_moe_{route,prep1,gemm1,act,gemm2,post2}" } else { "d41x_moe_* in 32-row chunks" },
        BackendKind::Native => if x.rows <= 32 { "d41_mx_act_quant+d41_mx_fc1_g*+d41_mx_fc2_g*" } else { "d41_mx_{plan,fc1,fc2,combine}_w" },
        BackendKind::Plain => "-" };
    receipts::count("moe.routed", w.1, entry, x.rows, cls);
    Ok(cls)
}

/// The kernel family a backend's dense call ran at `rows` (the backends return only the class; these are
/// their dispatch rules: KE MT = 1 / 2 / 4-8 by rows, KN1 split-K GEMV family up to 32 rows, KE plain GEMV).
fn entry_of(b: BackendKind, rows: usize) -> &'static str {
    match (b, rows) {
        (BackendKind::Exl3, 0..=16) => "d41x_prep+d41x_gemm_m1+d41x_post",
        (BackendKind::Exl3, 17..=32) => "d41x_prep+d41x_gemm_m2+d41x_post",
        (BackendKind::Exl3, _) => "d41x_prep+d41x_gemm_m{4,8}+d41x_post",
        (BackendKind::Native, 0..=32) => "d41_fp8_act_quant+d41_fp8_gemv (split-K)",
        (BackendKind::Native, _) => "d41_fp8_act_quant+d41_fp8_gemm (tensor core)",
        (BackendKind::Plain, 0..=32) => "d41_ds_plain_r{8,16,24,32}/d41x_plain_gemv_w/d41_ds_conf",
        (BackendKind::Plain, _) => "d41x_plain_gemv_rg",
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RopeArgs { pub x: u64, pub pos: u64, pub ld: i64, pub rows: i32, pub heads: i32, pub head_dim: i32, pub inverse: i32, pub freq: [f32; 32] }

/// RoPE (torch-on-CUDA arithmetic) on the last 64 dims of each head of bf16 rows, in place.
pub fn rope_rows(g: &Gpu, x: u64, ld: usize, pos: u64, rows: usize, heads: usize, head_dim: usize, inverse: bool, freq: &[f32; 32]) -> Result<()> {
    if rows == 0 { return Ok(()); }
    let a = RopeArgs { x, pos, ld: ld as i64, rows: rows as i32, heads: heads as i32, head_dim: head_dim as i32, inverse: inverse as i32, freq: *freq };
    g.launch1("d41f_rope_rows", (rows as u32, ((heads * 32) as u32).div_ceil(256), 1), (256, 1, 1), 0, &a)?;
    receipts::kernel("rope.q", "d41f_rope_rows", rows);
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SwigluArgs { pub g: u64, pub u: u64, pub out: u64, pub ldg: i64, pub ldu: i64, pub ldo: i64, pub rows: i32, pub n: i32, pub limit: f32, pub pad: i32 }

/// out = bf16(silu(min(g, L)) * clamp(u, -L, L)) (the shared expert's activation).
pub fn swiglu(g: &Gpu, gate: u64, up: u64, out: u64, ld: usize, rows: usize, n: usize, limit: f32) -> Result<()> {
    let a = SwigluArgs { g: gate, u: up, out, ldg: ld as i64, ldu: ld as i64, ldo: ld as i64, rows: rows as i32, n: n as i32, limit, pad: 0 };
    g.launch1("d41f_swiglu", ((n as u32).div_ceil(256), rows as u32, 1), (256, 1, 1), 0, &a)
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AddArgs { pub y: u64, pub s: u64, pub out: u64, pub ldy: i64, pub lds: i64, pub ldo: i64, pub rows: i32, pub n: i32 }

/// out = bf16(y + f32(s)) (s = 0: a plain f32 -> bf16 cast).
pub fn add_round(g: &Gpu, y: u64, s: u64, out: u64, rows: usize, n: usize) -> Result<()> {
    let a = AddArgs { y, s, out, ldy: n as i64, lds: n as i64, ldo: n as i64, rows: rows as i32, n: n as i32 };
    g.launch1("d41f_add_round", ((n as u32).div_ceil(256), rows as u32, 1), (256, 1, 1), 0, &a)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dsv41_sl_glue_arg_layouts() {
        assert_eq!(std::mem::size_of::<RopeArgs>(), 8 + 8 + 8 + 16 + 128);
        assert_eq!(std::mem::size_of::<SwigluArgs>(), 3 * 8 + 3 * 8 + 16);
        assert_eq!(std::mem::size_of::<AddArgs>(), 6 * 8 + 8);
    }
}
