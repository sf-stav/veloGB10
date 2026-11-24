//! KM kernel tests. Every GPU test is `#[ignore]` (run under `flock /tmp/dsv41_gpu.lock`, see the report):
//! `cargo test --release --lib dsv41_km -- --ignored --test-threads 1 --nocapture`.
//! Real weights come from the native slice (layers 0 / 1 / 14: hc_fn, router gates, Engram q / k) when it exists; the stream and
//! activations are synthetic (REFb's dumps / OM's oracle supply real ones once they land).

use super::mix_ref::*;
use super::*;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use std::sync::Arc;

const DIM: usize = 5120;
const SLICE: &str = "/mnt/models/dsv41/slices/native-fp8-L0-1-2-14-20-21-24-40-41-42";

fn dev() -> Arc<CudaDevice> {
    let d = CudaDevice::new(0).expect("GPU");
    crate::dsv41::gpu::load_module(&d).expect("gpu_dsv41 module");
    d
}
fn p<T>(s: &CudaSlice<T>) -> u64 { *s.device_ptr() as u64 }
fn up_bf(d: &Arc<CudaDevice>, v: &[f32]) -> CudaSlice<u16> { d.htod_sync_copy(&v.iter().map(|&x| bf_bits(x)).collect::<Vec<_>>()).unwrap() }
fn down_bf(d: &Arc<CudaDevice>, s: &CudaSlice<u16>) -> Vec<f32> { d.dtoh_sync_copy(s).unwrap().into_iter().map(from_bits).collect() }

/// A tensor of the native slice as f32 (F32 / BF16 / F16), or None when the slice is not on this box.
fn real(name: &str, layer: usize) -> Option<(Vec<f32>, Vec<u64>)> {
    let path = std::path::PathBuf::from(SLICE).join(format!("model-L{layer:02}.safetensors"));
    let h = crate::dsv41::st::read_header(&path).ok()?;
    let t = h.tensors.iter().find(|t| t.name == name)?.clone();
    let b = crate::dsv41::st::read_tensor(&h, name).ok()?;
    let v: Vec<f32> = match t.dtype.as_str() {
        "F32" => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        "BF16" => b.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        "F16" => b.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        _ => return None,
    };
    Some((v, t.shape))
}
fn rand_stream(rng: &mut Rng, rows: usize, sc: f32) -> Vec<f32> { (0..rows * 4 * DIM).map(|_| bf(rng.normal() * sc)).collect() }

#[test]
#[ignore = "GPU"]
fn dsv41_km_hc_mix_matches_f64_and_the_split_sums() {
    let d = dev();
    let c = MixConsts::v41();
    let k = HC * DIM;
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let (fnw, scale, base, src) = match (real("layers.0.hc_attn_fn", 0), real("layers.0.hc_attn_scale", 0), real("layers.0.hc_attn_base", 0)) {
        (Some((f, _)), Some((s, _)), Some((b, _))) => (f, [s[0], s[1], s[2]], <[f32; 24]>::try_from(&b[..]).unwrap(), "real layer-0 hc_attn"),
        _ => ((0..24 * k).map(|_| rng.normal() * 0.01).collect(), [0.1, 0.1, 0.1], [0.0; 24], "synthetic"),
    };
    println!("hc mix weights: {src}");
    let rows = 24;
    let x = rand_stream(&mut rng, rows, 2.0);
    let xd = up_bf(&d, &x);
    let fd = d.htod_sync_copy(&fnw).unwrap();
    let sc = d.htod_sync_copy(&scale).unwrap();
    let bs = d.htod_sync_copy(&base).unwrap();
    let refs: Vec<[f64; 24]> = (0..rows).map(|r| hc_record(&x[r * k..(r + 1) * k], &fnw, k, &scale, &base, 1e-20, 1e-6, 20)).collect();
    let mut first: Option<Vec<f32>> = None;
    for splits in [1usize, 2, 4, 8] {
        let klen = k / splits;
        let nchunk = klen / PART_CHUNK;
        let slots = splits * nchunk;
        let (row_stride, slot_stride) = (32usize, rows * 32);
        let part = d.alloc_zeros::<f32>(slots * slot_stride).unwrap();
        let rpart = d.alloc_zeros::<f32>(splits * slot_stride).unwrap();
        let cnt = d.htod_sync_copy(&vec![0u32; 8 * 8]).unwrap();
        let out = d.alloc_zeros::<f32>(rows * 24).unwrap();
        for _launch in 0..2 {   // twice: the counters re-arm themselves
        for rank in 0..splits {
            mix_partial(&d, &MixPartialArgs { x: p(&xd), x_ld: k as i64, fn_: p(&fd) + (rank * klen * 4) as u64, fn_ld: k as i64,
                part: p(&part) + (rank * nchunk * slot_stride * 4) as u64, chunk_stride: slot_stride as i64, row_stride: row_stride as i64,
                rank_part: p(&rpart) + (rank * slot_stride * 4) as u64, rp_row_stride: row_stride as i64, counters: p(&cnt) + (rank * 8 * 4) as u64,
                k0: (rank * klen) as i32, klen: klen as i32, rows: rows as i32, rg: pick_rg(rows) as i32 }).unwrap();
        }
        }
        mix_finish(&d, &MixFinishArgs { part: p(&rpart), slot_stride: slot_stride as i64, row_stride: row_stride as i64, scale: p(&sc), base: p(&bs),
            out: p(&out), out_ld: 24, nslots: splits as i32, rows: rows as i32, inv_k: c.inv_stream(), eps_norm: c.norm_eps, eps_hc: c.hc_eps, sinkhorn_iters: 20 }).unwrap();
        let got = d.dtoh_sync_copy(&out).unwrap();
        let mut worst = 0f64;
        for r in 0..rows { for m in 0..24 { worst = worst.max((got[r * 24 + m] as f64 - refs[r][m]).abs()); } }
        // comb is doubly stochastic after the last column normalisation
        for r in 0..rows { for kk in 0..4 { let s: f32 = (0..4).map(|j| got[r * 24 + 8 + 4 * j + kk]).sum(); assert!((s - 1.0).abs() < 1e-5, "col sum {s}"); } }
        println!("splits {splits} ({splits} rank partials of {nchunk} chunks): max |gpu - f64| over {rows} rows x 24 = {worst:.3e}");
        assert!(worst < 5e-6, "split {splits}: {worst}");
        if let Some(f) = &first {
            let dd = f.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            println!("   vs the unsplit mix: max |diff| {dd:.3e} (summation order only)");
            assert!(dd < 5e-6);
        } else { first = Some(got.clone()); }
        // batch invariance: row 5 alone == row 5 of the 24-row call, bitwise
        let out1 = d.alloc_zeros::<f32>(24).unwrap();
        let part1 = d.alloc_zeros::<f32>(slots * 32).unwrap();
        let rpart1 = d.alloc_zeros::<f32>(splits * 32).unwrap();
        let cnt1 = d.htod_sync_copy(&vec![0u32; 8 * 8]).unwrap();
        for rank in 0..splits {
            mix_partial(&d, &MixPartialArgs { x: p(&xd) + (5 * k * 2) as u64, x_ld: k as i64, fn_: p(&fd) + (rank * klen * 4) as u64, fn_ld: k as i64,
                part: p(&part1) + (rank * nchunk * 32 * 4) as u64, chunk_stride: 32, row_stride: 32,
                rank_part: p(&rpart1) + (rank * 32 * 4) as u64, rp_row_stride: 32, counters: p(&cnt1) + (rank * 8 * 4) as u64, k0: (rank * klen) as i32, klen: klen as i32, rows: 1, rg: 1 }).unwrap();
        }
        mix_finish(&d, &MixFinishArgs { part: p(&rpart1), slot_stride: 32, row_stride: 32, scale: p(&sc), base: p(&bs), out: p(&out1), out_ld: 24, nslots: splits as i32, rows: 1,
            inv_k: c.inv_stream(), eps_norm: c.norm_eps, eps_hc: c.hc_eps, sinkhorn_iters: 20 }).unwrap();
        let g1 = d.dtoh_sync_copy(&out1).unwrap();
        assert!(g1.iter().zip(&got[5 * 24..6 * 24]).all(|(a, b)| a.to_bits() == b.to_bits()), "row 5 differs between a 1-row and a 24-row call");
    }
    println!("batch invariance (1 row == the same row in 24) holds bitwise at every split");
}

#[test]
#[ignore = "GPU"]
fn dsv41_km_hc_pre_norm_and_post() {
    let d = dev();
    let c = MixConsts::v41();
    let mut rng = Rng(77);
    let rows = 6;
    let x = rand_stream(&mut rng, rows, 3.0);
    let pre: Vec<f32> = (0..rows * 4).map(|_| 0.2 + rng.uni() as f32).collect();
    let wn: Vec<f32> = (0..DIM).map(|_| 0.5 + rng.uni() as f32).collect();
    let wn_bf: Vec<f32> = wn.iter().map(|&v| bf(v)).collect();
    let (xd, pd) = (up_bf(&d, &x), d.htod_sync_copy(&pre).unwrap());
    for (fmt, wv, wd_u16, wd_f32) in [(3u32, &wn, None, Some(d.htod_sync_copy(&wn).unwrap())), (1u32, &wn_bf, Some(up_bf(&d, &wn)), None)] {
        let yn = d.alloc_zeros::<u16>(rows * DIM).unwrap();
        let yr = d.alloc_zeros::<u16>(rows * DIM).unwrap();
        let wptr = wd_u16.as_ref().map(p).or(wd_f32.as_ref().map(p)).unwrap();
        hc_pre_norm(&d, &HcPreNormArgs { x: p(&xd), x_ld: (4 * DIM) as i64, pre: p(&pd), pre_stride: 4, w: wptr, y_norm: p(&yn), yn_ld: DIM as i64, y_raw: p(&yr), yr_ld: DIM as i64,
            w_fmt: fmt, dim: DIM as i32, rows: rows as i32, eps: c.norm_eps, inv_dim: c.inv_dim() }).unwrap();
        let (gn, gr) = (down_bf(&d, &yn), down_bf(&d, &yr));
        let (mut raw_bad, mut norm_bad, mut worst) = (0usize, 0usize, 0f64);
        for r in 0..rows {
            let y = hc_pre_y(&x[r * 4 * DIM..(r + 1) * 4 * DIM], DIM, &[pre[r * 4], pre[r * 4 + 1], pre[r * 4 + 2], pre[r * 4 + 3]]);
            let n = rmsnorm_ref(&y, Some(wv), 1e-20);
            for i in 0..DIM {
                if gr[r * DIM + i] != y[i] { raw_bad += 1; }
                let e = (gn[r * DIM + i] as f64 - n[i]).abs() / n[i].abs().max(1e-3);
                worst = worst.max(e);
                if e > 0.0078125 { norm_bad += 1; }   // > 1 bf16 ulp (2^-7 relative)
            }
        }
        println!("hc_pre_norm w_fmt {fmt}: y_raw mismatches {raw_bad} / {} (bitwise vs the fp32 chain), normed worst rel err {worst:.3e}, beyond 1 ulp {norm_bad}", rows * DIM);
        assert_eq!(raw_bad, 0);
        assert_eq!(norm_bad, 0);
    }
    // hc_post: bitwise against the same fp32 chain; also in place
    let mut recs = vec![0f32; rows * 24];
    for v in recs.iter_mut() { *v = 0.1 + rng.uni() as f32 * 0.6; }
    let xs: Vec<f32> = (0..rows * DIM).map(|_| bf(rng.normal() * 2.0)).collect();
    let (xsd, rd) = (up_bf(&d, &xs), d.htod_sync_copy(&recs).unwrap());
    let out = d.alloc_zeros::<u16>(rows * 4 * DIM).unwrap();
    let a = HcPostArgs { xs: p(&xsd), xs_ld: DIM as i64, resid: p(&xd), r_ld: (4 * DIM) as i64, post: p(&rd) + 16, post_stride: 24, comb: p(&rd) + 32, comb_stride: 24,
        out: p(&out), o_ld: (4 * DIM) as i64, dim: DIM as i32, rows: rows as i32 };
    hc_post(&d, &a).unwrap();
    let got = down_bf(&d, &out);
    let mut bad = 0;
    for r in 0..rows {
        let want = hc_post_ref(&xs[r * DIM..(r + 1) * DIM], &x[r * 4 * DIM..(r + 1) * 4 * DIM], DIM, &recs[r * 24 + 4..r * 24 + 8], &recs[r * 24 + 8..r * 24 + 24]);
        for i in 0..4 * DIM { if got[r * 4 * DIM + i].to_bits() != want[i].to_bits() { bad += 1; } }
    }
    println!("hc_post: {bad} / {} elements differ from the fp32 chain", rows * 4 * DIM);
    assert_eq!(bad, 0);
    let mut b = a; b.out = p(&xd); b.resid = p(&xd);
    hc_post(&d, &b).unwrap();
    assert_eq!(down_bf(&d, &xd).iter().map(|v| v.to_bits()).collect::<Vec<_>>(), got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "in-place hc_post differs");
    println!("hc_post in place == out of place (bitwise)");
}

#[test]
#[ignore = "GPU"]
fn dsv41_km_rmsnorm_all_dims_and_weight_formats() {
    let d = dev();
    let mut rng = Rng(5);
    for dim in [512usize, 1536, 5120, 6144, 128] {
        let rows = 7;
        let x: Vec<f32> = (0..rows * dim).map(|_| bf(rng.normal() * 4.0)).collect();
        let w: Vec<f32> = (0..dim).map(|_| bf(0.5 + rng.uni() as f32)).collect();
        let (xd, wd, wf) = (up_bf(&d, &x), up_bf(&d, &w), d.htod_sync_copy(&w).unwrap());
        for (fmt, wp) in [(1u32, p(&wd)), (3, p(&wf)), (3, 0)] {
            let y = d.alloc_zeros::<u16>(rows * dim).unwrap();
            rmsnorm(&d, &RmsNormArgs { x: p(&xd), x_ld: dim as i64, w: wp, y: p(&y), y_ld: dim as i64, w_fmt: fmt, dim: dim as i32, rows: rows as i32, eps: 1e-20, inv_dim: 1.0 / dim as f32 }).unwrap();
            let g = down_bf(&d, &y);
            let mut worst = 0f64;
            for r in 0..rows {
                let n = rmsnorm_ref(&x[r * dim..(r + 1) * dim], if wp == 0 { None } else { Some(&w) }, 1e-20);
                for i in 0..dim { worst = worst.max((g[r * dim + i] as f64 - n[i]).abs() / n[i].abs().max(1e-3)); }
            }
            assert!(worst <= 0.0078125, "dim {dim} fmt {fmt} w {wp:#x}: {worst}");
        }
    }
    println!("rmsnorm: dims 128 / 512 / 1536 / 5120 / 6144 x (bf16 w, f32 w, no w) within 1 bf16 ulp of f64");
}

/// Router: scores vs f64, top-k vs f64 (ids identical unless the margin is below 1e-6), weights vs f64.
fn router_case(d: &Arc<CudaDevice>, name: &str, wv: &[f32], e: usize, topk: usize, w_fmt: u32, bias: &[f32], bias_fmt: u32, rows: usize, rng: &mut Rng, vl: Option<&[f32]>) {
    router_case_mode(d, name, wv, e, topk, w_fmt, bias, bias_fmt, rows, rng, vl, false)
}
/// `tf32`: the opt-in tensor-core router (`--dsv41-router-tf32 on`, leg PK1), same criteria.
#[allow(clippy::too_many_arguments)]
fn router_case_mode(d: &Arc<CudaDevice>, name: &str, wv: &[f32], e: usize, topk: usize, w_fmt: u32, bias: &[f32], bias_fmt: u32, rows: usize, rng: &mut Rng, vl: Option<&[f32]>, tf32: bool) {
    let x: Vec<f32> = (0..rows * DIM).map(|_| bf(rng.normal())).collect();
    let xd = up_bf(d, &x);
    let wd_u16 = up_bf_or_f16(d, wv, w_fmt);
    let wd_f32 = d.htod_sync_copy(wv).unwrap();
    let wptr = if w_fmt == 3 { p(&wd_f32) } else { p(&wd_u16) };
    let (bd32, bd16) = (d.htod_sync_copy(bias).unwrap(), d.htod_sync_copy(&bias.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect::<Vec<_>>()).unwrap());
    let bptr = if bias_fmt == 3 { p(&bd32) } else { p(&bd16) };
    let vld = vl.map(|v| d.htod_sync_copy(v).unwrap());
    let mask: Vec<u8> = (0..rows).map(|r| (r % 3 == 1) as u8).collect();
    let maskd = d.htod_sync_copy(&mask).unwrap();
    let sc = d.alloc_zeros::<f32>(rows * e).unwrap();
    router_scores_mode(d, &RouterScoresArgs { x: p(&xd), x_ld: DIM as i64, w: wptr, w_ld: DIM as i64, scores: p(&sc), s_ld: e as i64, w_fmt, dim: DIM as i32, e: e as i32, rows: rows as i32 }, tf32).unwrap();
    let (ow, oi) = (d.alloc_zeros::<f32>(rows * topk).unwrap(), d.alloc_zeros::<i32>(rows * topk).unwrap());
    router_select(d, &RouterSelectArgs { scores: p(&sc), s_ld: e as i64, bias: bptr, bias_vl: vld.as_ref().map_or(0, p), image_mask: if vl.is_some() { p(&maskd) } else { 0 },
        out_w: p(&ow), out_i: p(&oi), o_ld: topk as i64, bias_fmt, e: e as i32, topk: topk as i32, rows: rows as i32, route_scale: 1.5, temp: 1.0, norm_topk: 1 }).unwrap();
    let (gs, gw, gi) = (d.dtoh_sync_copy(&sc).unwrap(), d.dtoh_sync_copy(&ow).unwrap(), d.dtoh_sync_copy(&oi).unwrap());
    let (mut sworst, mut wworst, mut flips, mut ties, mut minmargin) = (0f64, 0f64, 0, 0, f64::MAX);
    for r in 0..rows {
        let s64: Vec<f64> = (0..e).map(|j| (0..DIM).map(|k| wv[j * DIM + k] as f64 * x[r * DIM + k] as f64).sum()).collect();
        for j in 0..e { sworst = sworst.max((gs[r * e + j] as f64 - s64[j]).abs() / s64[j].abs().max(1.0)); }
        let b: Vec<f64> = (0..e).map(|j| { let bb = if vl.is_some() && mask[r] == 1 { vl.unwrap()[j] } else { bias[j] }; if bias_fmt == 2 { half::f16::from_f32(bb).to_f32() as f64 } else { bb as f64 } }).collect();
        let (pick, w, margin) = route_ref(&s64, &b, topk, 1.5);
        minmargin = minmargin.min(margin);
        let got: Vec<usize> = (0..topk).map(|i| gi[r * topk + i] as usize).collect();
        if got != pick { if margin < 1e-5 { ties += 1; } else { flips += 1; } }
        else { for i in 0..topk { wworst = wworst.max((gw[r * topk + i] as f64 - w[i]).abs() / w[i]); } }
    }
    println!("router {name}: E {e} top-{topk} rows {rows}: score worst rel {sworst:.2e}, weight worst rel {wworst:.2e}, id flips {flips}, near-tie diffs {ties} (min margin {minmargin:.2e})");
    assert_eq!(flips, 0, "{name}: top-k ids differ from the f64 reference away from a tie");
    assert!(sworst < 2e-5 && wworst < 2e-5);
}
fn up_bf_or_f16(d: &Arc<CudaDevice>, v: &[f32], fmt: u32) -> CudaSlice<u16> {
    if fmt == 2 { d.htod_sync_copy(&v.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect::<Vec<_>>()).unwrap() }
    else { d.htod_sync_copy(&v.iter().map(|&x| bf_bits(x)).collect::<Vec<_>>()).unwrap() }
}

#[test]
#[ignore = "GPU"]
fn dsv41_km_router_top6_and_dspark_top3() {
    let d = dev();
    let mut rng = Rng(2024);
    for (layer, rows) in [(0usize, 24usize), (1, 1), (14, 6), (40, 70)] {
        let g = real(&format!("layers.{layer}.ffn.gate.weight"), layer);
        let b = real(&format!("layers.{layer}.ffn.gate.bias"), layer);
        let bv = real(&format!("layers.{layer}.ffn.gate.bias_vl"), layer);
        let (Some((gw, _)), Some((bias, _)), Some((bias_vl, _))) = (g, b, bv) else { println!("layer {layer}: not in the slice, skipped"); continue };
        // the slice stores bf16 gates: exact in bf16; re-encode as f16 too (the EXL3 pack format) after rounding to f16
        router_case(&d, &format!("L{layer} bf16 gate, f32 bias"), &gw, 384, 6, 1, &bias, 3, rows, &mut rng, Some(&bias_vl));
        let gw16: Vec<f32> = gw.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect();
        router_case(&d, &format!("L{layer} f16 gate, f32 bias"), &gw16, 384, 6, 2, &bias, 3, rows, &mut rng, None);
    }
    // synthetic: DSpark shape (128 experts, top-3, f16 bias) and the f32-weights path
    let gw: Vec<f32> = (0..128 * DIM).map(|_| bf(rng.normal() * 0.02)).collect();
    let bias: Vec<f32> = (0..128).map(|_| rng.normal() * 0.1).collect();
    router_case(&d, "synthetic DSpark 128 top-3 f16 bias", &gw, 128, 3, 1, &bias, 2, 18, &mut rng, None);
    let gw32: Vec<f32> = (0..384 * DIM).map(|_| rng.normal() * 0.02).collect();
    let bias: Vec<f32> = (0..384).map(|_| rng.normal() * 0.1).collect();
    router_case(&d, "synthetic 384 top-6 f32 weights", &gw32, 384, 6, 3, &bias, 3, 33, &mut rng, None);
}

/// The opt-in tf32 tensor-core router (`--dsv41-router-tf32 on`, leg PK1) on the same cases as the fp32 test (real gates when
/// the native slice is on this box, the DSpark shape, f16 re-encoded gates): no top-k id flip away from a near-tie.
#[test]
#[ignore = "GPU"]
fn dsv41_pk1_router_tf32_top6() {
    let d = dev();
    let mut rng = Rng(2024);
    for (layer, rows) in [(0usize, 24usize), (1, 1), (14, 6), (40, 70)] {
        let g = real(&format!("layers.{layer}.ffn.gate.weight"), layer);
        let b = real(&format!("layers.{layer}.ffn.gate.bias"), layer);
        // (no slice on this box: skipped quietly; dsv41_km_router_top6_and_dspark_top3 already reports it)
        let (Some((gw, _)), Some((bias, _))) = (g, b) else { continue };
        router_case_mode(&d, &format!("tf32 L{layer} bf16 gate, f32 bias"), &gw, 384, 6, 1, &bias, 3, rows, &mut rng, None, true);
        let gw16: Vec<f32> = gw.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect();
        router_case_mode(&d, &format!("tf32 L{layer} f16 gate, f32 bias"), &gw16, 384, 6, 2, &bias, 3, rows, &mut rng, None, true);
    }
    for rows in [1usize, 5, 6, 12, 24, 32] {
        let gw: Vec<f32> = (0..128 * DIM).map(|_| bf(rng.normal() * 0.02)).collect();
        let bias: Vec<f32> = (0..128).map(|_| rng.normal() * 0.1).collect();
        router_case_mode(&d, "tf32 synthetic DSpark 128 top-3 f16 bias", &gw, 128, 3, 1, &bias, 2, rows, &mut rng, None, true);
        let gw: Vec<f32> = (0..384 * DIM).map(|_| { let v = bf(rng.normal() * 0.02); half::f16::from_f32(v).to_f32() }).collect();
        let bias: Vec<f32> = (0..384).map(|_| rng.normal() * 0.1).collect();
        router_case_mode(&d, "tf32 synthetic 384 top-6 f16 gate", &gw, 384, 6, 2, &bias, 3, rows, &mut rng, None, true);
    }
}

#[test]
#[ignore = "GPU"]
fn dsv41_km_engram_gate() {
    let d = dev();
    let c = MixConsts::v41();
    let mut rng = Rng(31337);
    let rows = 9;
    let (wq, wk) = match (real("layers.1.engram.q_weight", 1), real("layers.1.engram.k_weight", 1)) {
        (Some((q, _)), Some((k, _))) => (q, k),
        _ => ((0..4 * DIM).map(|_| 1.0 + 0.1 * rng.normal()).collect(), (0..4 * DIM).map(|_| 1.0 + 0.1 * rng.normal()).collect()),
    };
    assert_eq!(wq.len(), 4 * DIM);
    let h = rand_stream(&mut rng, rows, 2.0);
    // key correlated with h on some rows so the gate leaves the clamp region
    let mut kv = vec![0f32; rows * 5 * DIM];
    for r in 0..rows { for i in 0..5 * DIM {
        let base = if i < 4 * DIM && r % 2 == 0 { h[r * 4 * DIM + i] * 0.5 } else { 0.0 };
        kv[r * 5 * DIM + i] = bf(base + rng.normal());
    } }
    let mask: Vec<u8> = (0..rows).map(|r| (r != 4) as u8).collect();
    let (hd, kd, maskd) = (up_bf(&d, &h), up_bf(&d, &kv), d.htod_sync_copy(&mask).unwrap());
    let (qd, kkd) = (d.htod_sync_copy(&wq).unwrap(), d.htod_sync_copy(&wk).unwrap());
    let out = d.alloc_zeros::<u16>(rows * 4 * DIM).unwrap();
    let a = EngramGateArgs { h: p(&hd), h_ld: (4 * DIM) as i64, kv: p(&kd), kv_ld: (5 * DIM) as i64, wq: p(&qd), wk: p(&kkd), gate_on: p(&maskd), out: p(&out), o_ld: (4 * DIM) as i64,
        wq_fmt: 3, wk_fmt: 3, dim: DIM as i32, rows: rows as i32, eps: c.norm_eps, inv_dim: c.inv_dim(), inv_sqrt_dim: c.inv_sqrt_dim(), clamp_min: c.engram_clamp };
    engram_gate(&d, &a).unwrap();
    let got = down_bf(&d, &out);
    let (mut bad, mut worst, mut gmin, mut gmax) = (0usize, 0f64, 1f64, 0f64);
    for r in 0..rows { for cc in 0..4 {
        let hs = &h[(r * 4 + cc) * DIM..(r * 4 + cc + 1) * DIM];
        let ks = &kv[r * 5 * DIM + cc * DIM..r * 5 * DIM + (cc + 1) * DIM];
        let g = if mask[r] == 0 { 0.0 } else { engram_gate_ref(hs, ks, &wq[cc * DIM..(cc + 1) * DIM], &wk[cc * DIM..(cc + 1) * DIM], 1e-20) };
        gmin = gmin.min(g); gmax = gmax.max(g);
        for i in 0..DIM {
            let want = hs[i] as f64 + g * kv[r * 5 * DIM + 4 * DIM + i] as f64;
            let e = (got[(r * 4 + cc) * DIM + i] as f64 - want).abs() / want.abs().max(1e-2);
            worst = worst.max(e);
            if e > 0.0078125 * 1.5 { bad += 1; }
        }
    } }
    println!("engram gate: gate range [{gmin:.4}, {gmax:.4}], worst rel err {worst:.3e}, beyond ~1 ulp {bad}");
    assert_eq!(bad, 0);
    // masked row is untouched
    assert!(got[(4 * 4) * DIM..(4 * 4 + 4) * DIM].iter().zip(&h[16 * DIM..20 * DIM]).all(|(a, b)| a.to_bits() == b.to_bits()), "masked row changed");
    // in place == out of place
    let mut b = a; b.out = p(&hd);
    engram_gate(&d, &b).unwrap();
    assert_eq!(down_bf(&d, &hd).iter().map(|v| v.to_bits()).collect::<Vec<_>>(), got.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    println!("engram gate in place == out of place (bitwise), masked row unchanged");
}

/// Byte-floor table: GB/s of the bytes each kernel must read against the 238 GB/s roofline. Weights rotate over enough copies
/// to defeat the L2 (a layer's weights are cold in a real round).
#[test]
#[ignore = "GPU"]
fn dsv41_km_bench_byte_floor() {
    let d = dev();
    let c = MixConsts::v41();
    let k = HC * DIM;
    const COPIES: usize = 24;
    let mut rng = Rng(9);
    let fn_all = d.alloc_zeros::<f32>(COPIES * 24 * k).unwrap();   // content irrelevant for timing (zeros are fine: no data-dependent paths except Sinkhorn)
    let gate_all = d.alloc_zeros::<u16>(COPIES * 384 * DIM).unwrap();
    let sc = d.htod_sync_copy(&[0.1f32, 0.1, 0.1]).unwrap();
    let bs = d.htod_sync_copy(&[0.0f32; 24]).unwrap();
    let bias = d.htod_sync_copy(&vec![0f32; 384]).unwrap();
    let wn = d.htod_sync_copy(&vec![1f32; DIM]).unwrap();
    let wq = d.htod_sync_copy(&vec![1f32; 4 * DIM]).unwrap();
    println!("| kernel | rows | us/launch | bytes read | GB/s | % of 238 |");
    println!("|---|---:|---:|---:|---:|---:|");
    for rows in [1usize, 2, 6, 12, 24] {
        let x = up_bf(&d, &rand_stream(&mut rng, rows, 1.0));
        let kvd = up_bf(&d, &(0..rows * 5 * DIM).map(|_| bf(rng.normal())).collect::<Vec<_>>());
        let nchunk = k / PART_CHUNK;
        let (rs, ss) = (32usize, rows * 32);
        let part = d.alloc_zeros::<f32>(nchunk * ss).unwrap();
        let rp = d.alloc_zeros::<f32>(ss).unwrap();
        let cnt = d.htod_sync_copy(&vec![0u32; 64]).unwrap();
        let rec = d.alloc_zeros::<f32>(rows * 24).unwrap();
        let ynorm = d.alloc_zeros::<u16>(rows * DIM).unwrap();
        let strm = d.alloc_zeros::<u16>(rows * 4 * DIM).unwrap();
        let scores = d.alloc_zeros::<f32>(rows * 384).unwrap();
        let (ow, oi) = (d.alloc_zeros::<f32>(rows * 6).unwrap(), d.alloc_zeros::<i32>(rows * 6).unwrap());
        let mut time = |name: &str, bytes: f64, n: usize, f: &mut dyn FnMut(usize)| {
            for i in 0..COPIES { f(i); }
            d.synchronize().unwrap();
            let t0 = std::time::Instant::now();
            for i in 0..n { f(i % COPIES); }
            d.synchronize().unwrap();
            let us = t0.elapsed().as_secs_f64() * 1e6 / n as f64;
            let gbs = bytes / us / 1e3;
            println!("| {name} | {rows} | {us:.2} | {:.2} MB | {gbs:.0} | {:.0}% |", bytes / 1e6, gbs / 2.38);
        };
        let n = 400;
        time("mix_partial (hc_fn 1.97 MB)", (24 * k * 4) as f64, n, &mut |i| mix_partial(&d, &MixPartialArgs { x: p(&x), x_ld: k as i64, fn_: p(&fn_all) + (i * 24 * k * 4) as u64, fn_ld: k as i64,
            part: p(&part), chunk_stride: ss as i64, row_stride: rs as i64, rank_part: p(&rp), rp_row_stride: rs as i64, counters: p(&cnt), k0: 0, klen: k as i32, rows: rows as i32, rg: pick_rg(rows) as i32 }).unwrap());
        time("mix_partial TP4 slice (0.49 MB)", (24 * k * 4 / 4) as f64, n, &mut |i| mix_partial(&d, &MixPartialArgs { x: p(&x), x_ld: k as i64, fn_: p(&fn_all) + (i * 24 * k * 4) as u64, fn_ld: k as i64,
            part: p(&part), chunk_stride: ss as i64, row_stride: rs as i64, rank_part: p(&rp), rp_row_stride: rs as i64, counters: p(&cnt), k0: 0, klen: (k / 4) as i32, rows: rows as i32, rg: pick_rg(rows) as i32 }).unwrap());
        time("mix_finish (1 rank slot + Sinkhorn)", (rows * 25 * 4) as f64, n, &mut |_| mix_finish(&d, &MixFinishArgs { part: p(&part), slot_stride: ss as i64, row_stride: rs as i64, scale: p(&sc), base: p(&bs),
            out: p(&rec), out_ld: 24, nslots: 1, rows: rows as i32, inv_k: c.inv_stream(), eps_norm: c.norm_eps, eps_hc: c.hc_eps, sinkhorn_iters: 20 }).unwrap());
        time("hc_pre_norm (stream rows)", (rows * 4 * DIM * 2) as f64, n, &mut |_| hc_pre_norm(&d, &HcPreNormArgs { x: p(&x), x_ld: k as i64, pre: p(&rec), pre_stride: 24, w: p(&wn), y_norm: p(&ynorm), yn_ld: DIM as i64,
            y_raw: 0, yr_ld: 0, w_fmt: 3, dim: DIM as i32, rows: rows as i32, eps: c.norm_eps, inv_dim: c.inv_dim() }).unwrap());
        time("hc_post (stream r+w)", (rows * (4 + 4 + 1) * DIM * 2) as f64, n, &mut |_| hc_post(&d, &HcPostArgs { xs: p(&ynorm), xs_ld: DIM as i64, resid: p(&x), r_ld: k as i64, post: p(&rec) + 16, post_stride: 24,
            comb: p(&rec) + 32, comb_stride: 24, out: p(&strm), o_ld: k as i64, dim: DIM as i32, rows: rows as i32 }).unwrap());
        time("rmsnorm 1536", (rows * 1536 * 2) as f64, n, &mut |_| rmsnorm(&d, &RmsNormArgs { x: p(&x), x_ld: k as i64, w: p(&wn), y: p(&ynorm), y_ld: DIM as i64, w_fmt: 3, dim: 1536, rows: rows as i32, eps: 1e-20, inv_dim: 1.0 / 1536.0 }).unwrap());
        time("router_scores (gate 3.93 MB)", (384 * DIM * 2) as f64, n, &mut |i| router_scores(&d, &RouterScoresArgs { x: p(&x), x_ld: k as i64, w: p(&gate_all) + (i * 384 * DIM * 2) as u64, w_ld: DIM as i64,
            scores: p(&scores), s_ld: 384, w_fmt: 1, dim: DIM as i32, e: 384, rows: rows as i32 }).unwrap());
        time("router_select (top-6 of 384)", (rows * 384 * 4) as f64, n, &mut |_| router_select(&d, &RouterSelectArgs { scores: p(&scores), s_ld: 384, bias: p(&bias), bias_vl: 0, image_mask: 0, out_w: p(&ow), out_i: p(&oi),
            o_ld: 6, bias_fmt: 3, e: 384, topk: 6, rows: rows as i32, route_scale: 1.5, temp: 1.0, norm_topk: 1 }).unwrap());
        time("engram_gate (h + key + value)", (rows * (4 + 4 + 1) * DIM * 2) as f64, n, &mut |_| engram_gate(&d, &EngramGateArgs { h: p(&strm), h_ld: k as i64, kv: p(&kvd), kv_ld: (5 * DIM) as i64, wq: p(&wq), wk: p(&wq), gate_on: 0,
            out: p(&strm), o_ld: k as i64, wq_fmt: 3, wk_fmt: 3, dim: DIM as i32, rows: rows as i32, eps: 1e-20, inv_dim: c.inv_dim(), inv_sqrt_dim: c.inv_sqrt_dim(), clamp_min: 1e-6 }).unwrap());
    }
}
