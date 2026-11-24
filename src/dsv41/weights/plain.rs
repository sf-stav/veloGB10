//! DSV4.1 plain-linear backend (BF16 / F16 / F32 matrices) — OWNED BY LEG KE (I0 stub).
//!
//! What it will hold: every linear a pack stores unquantized — in the EAQ v1 EXL3 pack the
//! embedding (BF16), the router gates (F16), the indexer weights_proj (F16), the DSpark Markov
//! embedding / head and confidence head; in mixed packs any other class (EAQ v1.1: 27 BF16
//! indexer / compressor linears; the dealignai pack: 27 BF16 shared_experts.w2); in the native
//! checkpoint the embedding, head, gates, compressor, indexer wk / weights_proj and the DSpark
//! heads. f32 accumulation; the head and the router compute in f32 as the reference does. All
//! three row classes. This is a first-class path (Revision 1), not a fallback.

use super::exl3::dense::tensor;
use super::exl3::launch::{act_code, kernel, launch};
use super::{Accum, Act, ActDtype, BackendKind, DenseWeight, DevPtr, Dsv41Weights, Launch, MoeWeight, Route, RowClass, Scratch, Shard};
use crate::dsv41::census::classify::Role;
use crate::dsv41::census::LinearFormat;
use anyhow::{bail, ensure, Result};
use std::ffi::c_void;

pub struct PlainWeights;

/// (pointer to the call's first row, row length K, rows in the call, code 0 bf16 / 1 f16) of a plain matrix.
fn rows_of(w: &DenseWeight) -> Result<(u64, usize, usize, i32)> {
    let t = tensor(w, Role::Weight)?;
    let code = match t.dtype.as_str() { "BF16" => 0, "F16" => 1, d => bail!("plain {}: dtype {d} (BF16 / F16 only)", w.key) };
    ensure!(t.shape.len() == 2, "plain {}: shape {:?}", w.key, t.shape);
    let (n, k) = (t.shape[0] as usize, t.shape[1] as usize);
    ensure!(k as u64 == w.inn, "plain {}: K {k} != {} (an input slice of a plain linear is not supported)", w.key, w.inn);
    Ok(match &w.shard {
        Shard::Whole => (t.ptr.0, k, n, code),
        Shard::Out(r) => {
            let (lo, hi) = (r.start as usize, r.end as usize);
            ensure!(hi <= w.out as usize && lo < hi, "plain {}: shard {lo}..{hi}", w.key);
            if n == w.out as usize { (t.ptr.0 + (lo * k * 2) as u64, k, hi - lo, code) } else { ensure!(n == hi - lo, "plain {}: resident rows {n}", w.key); (t.ptr.0, k, n, code) }
        }
        Shard::In(_) => bail!("plain {}: an input slice of a plain linear is not supported", w.key),
    })
}

#[cfg(test)]
mod pp1_tests {
    use super::gemv_row_blocks;
    #[test]
    fn dsv41_pp1_plain_gemv_row_blocks_leave_decode_launches_alone() {
        for rows in 0..=32 { assert_eq!(gemv_row_blocks(rows), 1, "rows {rows}: the decode / verify / head launch is unchanged"); }
        assert_eq!(gemv_row_blocks(2048), 256, "a 2,048-row prefill call: one block row per 8-row group");
        assert_eq!(gemv_row_blocks(33), 5);
        assert_eq!(gemv_row_blocks(8 * 65_535), 65_535);
        assert_eq!(gemv_row_blocks(8 * 65_535 + 1), 1, "beyond one grid: the looping kernel");
    }
}

/// Leg PP1: block rows (grid y) of a plain GEMV call of `rows` rows. Calls of up to 32 rows (decode, verify, the head) keep
/// `d41x_plain_gemv` on one block row: their launches are unchanged. A wider call (a prefill piece) takes `d41x_plain_gemv_rg`
/// with one block row per 8-row group, so a narrow weight (the indexer `weights_proj`, 32 columns: 4 blocks along x) no longer
/// runs a 2,048-row call on 4 blocks. Every (8-row group, column) is the same computation either way (same bits).
pub fn gemv_row_blocks(rows: usize) -> u32 {
    // (more than 65,535 row groups cannot be one grid's y: such a call keeps the one-block-row kernel, which loops over the rows)
    if rows <= 32 || rows.div_ceil(8) > 65_535 { 1 } else { rows.div_ceil(8) as u32 }
}

/// Leg P4: does a plain GEMV call take the one-warp-per-block entry (`d41x_plain_gemv_w`)? A narrow weight at decode / verify widths.
pub fn small_gemv(nn: usize, rows: usize) -> bool { (1..=32).contains(&rows) && nn <= 256 }

fn gemv(l: &Launch, w: &DenseWeight, x: &Act, y: &Act, acc: Accum) -> Result<RowClass> {
    let (wp, k, nn, wcode) = rows_of(w)?;
    ensure!(x.cols == k && y.cols == nn && x.rows == y.rows && k % 256 == 0, "plain {}: x [{}, {}] y [{}, {}] K {k} N {nn} (K must be a multiple of 256)", w.key, x.rows, x.cols, y.rows, y.cols);
    let (mut wptr, mut wc, mut kk, mut n, mut xp, mut xc, mut ldx, mut rows, mut yp, mut yc, mut ldy, mut ac) =
        (wp, wcode, k as i32, nn as i32, x.ptr.0, act_code(x.dtype)?, x.ld as i32, x.rows as i32, y.ptr.0, act_code(y.dtype)?, y.ld as i32, (acc == Accum::Add) as i32);
    // leg P6-8: single-column BF16 -> F32 projection (DsConfidence) runs 1 warp per row (`d41_ds_conf`), exact same xor tree.
    if nn == 1 && xc == 0 && yc == 2 && ac == 0 && (1..=32).contains(&x.rows) && wp % 16 == 0 {
        let a = crate::dsv41::gpu::dspark::ConfArgs { w: wp, x: x.ptr.0, y: y.ptr.0, wcode, k: k as i32, ldx: x.ld as i32, ldy: y.ld as i32, rows: x.rows as i32, pad: 0 };
        let mut p = [&a as *const _ as *mut c_void];
        crate::dsv41::gpu::launch_raw("d41_ds_conf", (x.rows as u32, 1, 1), (32, 1, 1), 0, l.stream, l.pdl, &mut p)?;
        return Ok(RowClass::of(x.rows));
    }
    // leg P4: a narrow weight (<= 256 columns) at 1..=32 rows: one warp per block over (column, 2-row group), not 4-32 blocks of 8 warps
    if small_gemv(nn, x.rows) && crate::dsv41::gpu::pf_on("dsv41-pf-gemv-small") {
        crate::dsv41::gpu::PF_GEMV_SMALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        launch(&kernel("d41x_plain_gemv_w")?, (nn as u32, x.rows.div_ceil(2) as u32, 1), 32, l.stream, &mut [
            &mut wptr as *mut _ as *mut c_void, &mut wc as *mut _ as *mut c_void, &mut kk as *mut _ as *mut c_void, &mut n as *mut _ as *mut c_void,
            &mut xp as *mut _ as *mut c_void, &mut xc as *mut _ as *mut c_void, &mut ldx as *mut _ as *mut c_void, &mut rows as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void, &mut yc as *mut _ as *mut c_void, &mut ldy as *mut _ as *mut c_void, &mut ac as *mut _ as *mut c_void])?;
        return Ok(RowClass::of(x.rows));
    }
    // leg P6-8: 1..=32 decode / verify rows on wide plain matrices: keep all rows in registers (`d41_ds_plain_r{8,16,24,32}`),
    // reading the weight matrix once with the exact same per-row fmaf + xor-tree arithmetic (0 generic fallbacks).
    if (1..=32).contains(&x.rows) && wp % 16 == 0 && (xc != 0 || (x.ptr.0 % 16 == 0 && x.ld % 8 == 0)) {
        let name = match x.rows { 1..=8 => "d41_ds_plain_r8", 9..=16 => "d41_ds_plain_r16", 17..=24 => "d41_ds_plain_r24", _ => "d41_ds_plain_r32" };
        let a = crate::dsv41::gpu::dspark::PlainArgs { w: wp, x: x.ptr.0, y: y.ptr.0, wcode, k: k as i32, nn: nn as i32, xcode: xc, ldx: x.ld as i32,
                                                       rows: x.rows as i32, ycode: yc, ldy: y.ld as i32, accum: ac, pad: 0 };
        let mut p = [&a as *const _ as *mut c_void];
        crate::dsv41::gpu::launch_raw(name, (nn.div_ceil(8) as u32, 1, 1), (256, 1, 1), 0, l.stream, l.pdl, &mut p)?;
        return Ok(RowClass::of(x.rows));
    }
    let gy = if crate::dsv41::gpu::pf_on("dsv41-pf-gemv-rows") { gemv_row_blocks(x.rows) } else { 1 };
    if gy > 1 { crate::dsv41::gpu::PF_GEMV_ROWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
    launch(&kernel(if gy > 1 { "d41x_plain_gemv_rg" } else { "d41x_plain_gemv" })?, (((nn + 7) / 8) as u32, gy, 1), 256, l.stream, &mut [
        &mut wptr as *mut _ as *mut c_void, &mut wc as *mut _ as *mut c_void, &mut kk as *mut _ as *mut c_void, &mut n as *mut _ as *mut c_void,
        &mut xp as *mut _ as *mut c_void, &mut xc as *mut _ as *mut c_void, &mut ldx as *mut _ as *mut c_void, &mut rows as *mut _ as *mut c_void,
        &mut yp as *mut _ as *mut c_void, &mut yc as *mut _ as *mut c_void, &mut ldy as *mut _ as *mut c_void, &mut ac as *mut _ as *mut c_void])?;
    Ok(if gy > 1 { RowClass::of(x.rows) } else { RowClass::Generic })
}

impl Dsv41Weights for PlainWeights {
    fn kind(&self) -> BackendKind { BackendKind::Plain }
    fn describe(&self) -> &'static str {
        "plain BF16 / F16 linears (embedding, head, gates, indexer / compressor / shared linears of mixed packs), f32 accumulation, warp-per-column GEMV"
    }
    fn holds(&self, f: &LinearFormat) -> bool { matches!(f, LinearFormat::Plain(_)) }
    fn scratch_bytes(&self, _w: &DenseWeight, _rows: usize) -> usize { 0 }
    fn dense(&self, l: &Launch, w: &DenseWeight, x: &Act, y: &Act, a: Accum, _s: Scratch) -> Result<RowClass> { gemv(l, w, x, y, a) }
    fn moe(&self, _l: &Launch, w: &MoeWeight, _x: &Act, _r: &Route, _y: &Act, _s: Scratch) -> Result<RowClass> {
        bail!("plain grouped MoE layer {}: not implemented (a BF16 expert pack would land here)", w.layer)
    }
    fn gather(&self, l: &Launch, w: &DenseWeight, ids: DevPtr, rows: usize, o: &Act) -> Result<()> {
        let (wp, dim, nrows, code) = rows_of(w)?;
        ensure!(code == 0 && o.dtype == ActDtype::Bf16 && o.cols == dim && o.rows == rows, "plain gather {}: BF16 table and BF16 [rows, dim] output required", w.key);
        let v0 = match &w.shard { Shard::Out(r) => r.start as i32, _ => 0 };
        let (mut e, mut idp, mut a0, mut a1, mut d, mut r, mut op, mut ldo) = (wp, ids.0, v0, v0 + nrows as i32, dim as i32, rows as i32, o.ptr.0, o.ld as i32);
        launch(&kernel("d41x_gather")?, (((dim + 255) / 256) as u32, rows as u32, 1), 256, l.stream, &mut [
            &mut e as *mut _ as *mut c_void, &mut idp as *mut _ as *mut c_void, &mut a0 as *mut _ as *mut c_void, &mut a1 as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void, &mut r as *mut _ as *mut c_void, &mut op as *mut _ as *mut c_void, &mut ldo as *mut _ as *mut c_void])
    }
    fn logits(&self, l: &Launch, w: &DenseWeight, x: &Act, o: &Act, _s: Scratch) -> Result<RowClass> {
        ensure!(o.dtype == ActDtype::F32, "plain head {}: logits are f32", w.key);
        gemv(l, w, x, o, Accum::Overwrite)
    }
}
