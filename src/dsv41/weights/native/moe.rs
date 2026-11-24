//! DSV4.1 native MXFP4 routed experts with FP8 activations — leg KN2.
//! Kernels: `kernels/gpu_dsv41/30_mxfp4_moe.cuh` (band head) + `31_mxfp4_decode.cuh` (grouped, <= 32
//! rows); the arithmetic contract and the fragment map are in the band head.
//!
//! ## What a native `MoeWeight` must hold (the contract for the loader, leg LS)
//! The loader packs every expert of a layer into the **mx1** layout (`pack::pack_w13` / `pack_w2`,
//! or `pack_layer` for a whole layer) at upload: two device buffers per layer, local experts at a
//! uniform stride:
//! - `experts[e][0]` (w1): one `DevTensor { role: Weight, dtype: "MX1_W13", ptr: w13 + e * e13,
//!   bytes: e13 }` — w1 and w3 interleaved per 8 inner rows; `shard` = `Shard::Out(i0..i0 + ns)`
//!   for an inner-width slice (T0 §3.1, the default) or `Shard::Whole`;
//! - `experts[e][1]` (w2): `DevTensor { dtype: "MX1_W2", ptr: w2 + e * e2, bytes: e2 }`, `shard`
//!   = `Shard::In(i0..i0 + ns)` or `Whole`;
//! - `experts[e][2]` (w3): no tensors (it lives in the w13 buffer); same `shard` as w1.
//! - `local` = the experts this rank holds: all of them with the inner split, a range under EP
//!   (T0's `--dsv41-expert-split ep`; routes to other ranks' experts are skipped here).
//!
//! ## What `moe` computes
//! `y f32 [rows, dim] += sum over this rank's routes, in ascending expert id, of
//! weight * W2_e(swiglu)` with the reference's arithmetic: whole experts (TP1, EP) give each
//! expert's output rounded to bf16 (REF_MAP §4 item 12: bit-exact to `model.py`'s MoE when `y`
//! starts at zero); an inner slice gives the FP32 partial (T0 §3.1; the reduce adds the ranks).
//! Rows <= 32 run the grouped kernels (`31`: GEMV / cooperative classes and the 17-32-row verify of
//! 3-4 lanes); more rows run the prefill kernels (`32`: route plan, token tiles of 64 per expert,
//! the activations staged in shared memory, ordered combine) in sub-chunks that fit the scratch.
//! Every path computes each output with the same instruction sequence: bit-identical results.

#[path = "mxfp4_launch.rs"]
pub mod launch;
#[path = "mxfp4_pack.rs"]
pub mod pack;
#[path = "mxfp4_probe.rs"]
pub mod probe;
#[path = "mxfp4_ref.rs"]
pub mod reference;

use crate::dsv41::weights::{Act, ActDtype, Launch, MoeWeight, Route, RowClass, Scratch, Shard};
use anyhow::{bail, ensure, Context, Result};

/// Entry kernels of band 30 (`kernels/gpu_dsv41/30_mxfp4_moe.cuh`); the gpu_dsv41 loader collects them.
pub const FNS: &[&str] = launch::FNS;

/// The device dtype tags of the packed tensors.
pub const DT_W13: &str = "MX1_W13";
pub const DT_W2: &str = "MX1_W2";

/// The packed view of a native MoE layer (validates the loader contract above).
pub fn view(w: &MoeWeight) -> Result<launch::PackedView> {
    let n_local = w.local.len();
    ensure!(n_local > 0 && w.experts.len() == n_local, "MoE layer {}: {} expert entries for local {:?}", w.layer, w.experts.len(), w.local);
    ensure!(w.inter % 32 == 0 && w.dim % 64 == 0, "MoE layer {}: dim {} / inter {}", w.layer, w.dim, w.inter);
    let t13 = |e: usize| w.experts[e][0].tensors.first().with_context(|| format!("layer {} expert {e}: no w13 tensor", w.layer));
    let t2 = |e: usize| w.experts[e][1].tensors.first().with_context(|| format!("layer {} expert {e}: no w2 tensor", w.layer));
    let (a, b) = (t13(0)?, t2(0)?);
    ensure!(a.dtype == DT_W13 && b.dtype == DT_W2, "MoE layer {}: tensors are {} / {}, not the mx1 layout ({DT_W13} / {DT_W2})", w.layer, a.dtype, b.dtype);
    let (i0, ns) = match &w.experts[0][0].shard {
        Shard::Whole => (0, w.inter),
        Shard::Out(r) => (r.start as usize, (r.end - r.start) as usize),
        Shard::In(_) => bail!("MoE layer {}: w1 cannot be split along its input", w.layer),
    };
    let geom = pack::ExpertGeom { dim: w.dim, inter: w.inter, i0, ns };
    geom.validate()?;
    match &w.experts[0][1].shard {
        Shard::Whole => ensure!(geom.is_whole(), "MoE layer {}: w1 sliced, w2 whole", w.layer),
        Shard::In(r) => ensure!(r.start as usize == i0 && (r.end - r.start) as usize == ns, "MoE layer {}: w2 slice {r:?} != w1 slice {i0}+{ns}", w.layer),
        Shard::Out(_) => bail!("MoE layer {}: w2 is split along its input (the inner width), not its output", w.layer),
    }
    ensure!(a.bytes as usize == geom.w13_bytes() && b.bytes as usize == geom.w2_bytes(),
            "MoE layer {}: packed sizes {} / {} != {} / {} for {geom:?}", w.layer, a.bytes, b.bytes, geom.w13_bytes(), geom.w2_bytes());
    for e in 1..n_local {   // uniform stride: the kernel addresses expert e as base + e * stride
        ensure!(t13(e)?.ptr.0 == a.ptr.0 + e as u64 * a.bytes && t2(e)?.ptr.0 == b.ptr.0 + e as u64 * b.bytes,
                "MoE layer {}: expert {e} is not at the uniform stride of the packed buffers", w.layer);
    }
    Ok(launch::PackedView {
        w13: a.ptr, e13: a.bytes, mt13: geom.w13_mtile() as u64, w2: b.ptr, e2: b.bytes, mt2: geom.w2_mtile() as u64,
        n_local, e_first: w.local.start, n_experts: w.n_experts, dim: w.dim, i0, ns, round_bf16: geom.is_whole(),
        wide_bf16: geom.is_whole() || crate::dsv41::weights::partials::on(), limit: w.swiglu_limit,
    })
}

/// Device scratch a `moe` call of `rows` rows needs (a prefill call of more than 32 rows: the
/// scratch of one sub-chunk of up to `WIDE_CHUNK_MAX` rows; less scratch runs smaller sub-chunks).
pub fn scratch_bytes(w: &MoeWeight, rows: usize) -> usize {
    let ns = match w.experts.first().map(|x| &x[0].shard) { Some(Shard::Out(r)) => (r.end - r.start) as usize, _ => w.inter };
    scratch_bytes_for(rows, w.top_k, w.dim, ns, w.local.len())
}
pub fn scratch_bytes_for(rows: usize, top_k: usize, dim: usize, ns: usize, n_local: usize) -> usize {
    let g = launch::ScratchPlan::of(rows, top_k, dim, ns).bytes;
    if rows <= launch::GROUPED_MAX_ROWS { g } else { g.max(launch::WidePlan::of(rows.min(launch::WIDE_CHUNK_MAX), top_k, dim, ns, n_local).bytes) }
}

/// The trait entry (`Dsv41Weights::moe` of the native backend).
pub fn moe(l: &Launch, w: &MoeWeight, x: &Act, r: &Route, y: &Act, s: Scratch) -> Result<RowClass> {
    let v = view(w)?;
    moe_view(l, &v, w.top_k, x, r, y, true, s, launch::Debug::default())
}

/// The same on an already validated view (the probe drives this one with debug outputs).
#[allow(clippy::too_many_arguments)]
pub fn moe_view(l: &Launch, v: &launch::PackedView, top_k: usize, x: &Act, r: &Route, y: &Act, accumulate: bool, s: Scratch,
                dbg: launch::Debug) -> Result<RowClass> {
    ensure!(x.dtype == ActDtype::Bf16 && x.cols == v.dim && x.ld >= v.dim, "MoE input must be bf16 [rows, {}] (got {:?} {}x{})", v.dim, x.dtype, x.rows, x.cols);
    ensure!(y.dtype == ActDtype::F32 && y.cols == v.dim && y.rows == x.rows && y.ld >= v.dim, "MoE output must be f32 [{}, {}]", x.rows, v.dim);
    ensure!(r.top_k == top_k && top_k > 0, "route top_k {} != layer top_k {top_k}", r.top_k);
    ensure!(l.stream != 0, "dsv41 launches never use the legacy default stream (AGENTS §2.1)");
    let rows = x.rows;
    if rows == 0 { return Ok(RowClass::of(1)); }
    if (dbg.gate_up != 0 || dbg.route_out != 0) && rows > launch::GROUPED_MAX_ROWS { bail!("debug outputs are decode-only (<= 32 rows)"); }
    if rows <= launch::GROUPED_MAX_ROWS {
        let sp = launch::ScratchPlan::of(rows, top_k, v.dim, v.ns);
        ensure!(s.bytes >= sp.bytes && s.ptr.0 != 0, "MoE scratch {} B < {} B", s.bytes, sp.bytes);
        launch::grouped(l.stream, l.pdl, v, x.ptr.0, x.ld, rows, top_k, r.ids.0, r.weights.0, y.ptr.0, y.ld, accumulate, s.ptr.0, &sp, dbg)?;
        return Ok(RowClass::of(rows));
    }
    // prefill: the wide kernels in sub-chunks that fit the scratch (same bits whatever the sub-chunk)
    let wp = launch::WidePlan::fit(rows, top_k, v.dim, v.ns, v.n_local, s.bytes)
        .with_context(|| format!("MoE scratch {} B cannot hold a {}-row wide sub-chunk (needs {} B)", s.bytes, launch::GROUPED_MAX_ROWS + 1,
                                 launch::WidePlan::of(launch::GROUPED_MAX_ROWS + 1, top_k, v.dim, v.ns, v.n_local).bytes))?;
    ensure!(s.ptr.0 != 0, "MoE scratch pointer is null");
    let mut r0 = 0;
    while r0 < rows {
        let n = (rows - r0).min(wp.rc);
        launch::wide(l.stream, l.pdl, v, x.ptr.0 + (r0 * x.ld * 2) as u64, x.ld, n, top_k,
                     r.ids.0 + (r0 * top_k * 4) as u64, r.weights.0 + (r0 * top_k * 4) as u64,
                     y.ptr.0 + (r0 * y.ld * 4) as u64, y.ld, accumulate, s.ptr.0, &wp)?;
        r0 += n;
    }
    if v.wide_bf16 && !v.round_bf16 { crate::dsv41::weights::partials::note_call("native MXFP4", rows); }
    Ok(RowClass::Wide)
}

/// Host-side packing of one layer for the loader: every local expert's raw checkpoint tensors ->
/// the two mx1 buffers (w13, w2) at a uniform stride (threads over experts).
pub fn pack_layer(experts: &[reference::RawExpert], geom: &pack::ExpertGeom) -> Result<(Vec<u8>, Vec<u8>)> {
    let (e13, e2) = (geom.w13_bytes(), geom.w2_bytes());
    let mut b13 = vec![0u8; e13 * experts.len()];
    let mut b2 = vec![0u8; e2 * experts.len()];
    if experts.is_empty() { return Ok((b13, b2)); }
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
    let chunk = experts.len().div_ceil(threads).max(1);
    std::thread::scope(|sc| -> Result<()> {
        let mut hs = Vec::new();
        for ((ex, o13), o2) in experts.chunks(chunk).zip(b13.chunks_mut(e13 * chunk)).zip(b2.chunks_mut(e2 * chunk)) {
            hs.push(sc.spawn(move || -> Result<()> {
                for (i, e) in ex.iter().enumerate() {
                    let (dim, inter) = (geom.dim, geom.inter);
                    pack::pack_w13(&mut o13[i * e13..(i + 1) * e13], &pack::RawMat::new(e.w1, e.s1, inter, dim)?,
                                   &pack::RawMat::new(e.w3, e.s3, inter, dim)?, geom)?;
                    pack::pack_w2(&mut o2[i * e2..(i + 1) * e2], &pack::RawMat::new(e.w2, e.s2, dim, inter)?, geom)?;
                }
                Ok(())
            }));
        }
        for h in hs { h.join().map_err(|_| anyhow::anyhow!("pack thread panicked"))??; }
        Ok(())
    })?;
    Ok((b13, b2))
}
