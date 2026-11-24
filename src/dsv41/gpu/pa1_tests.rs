//! Leg PA1 kernel tests (GPU, `#[ignore]`; run under `flock /tmp/dsv41_gpu.lock`):
//! `cargo test --release --lib dsv41_pa1_gpu -- --ignored --test-threads 1 --nocapture`.
//! Each new launch shape against the launches it replaces, bit for bit:
//!   * the hc side mix at TP2: `d41_mix_partial2` (both ranks' K slices) + `d41_mix_finish` nslots 2 == rank 0's and rank 1's
//!     `d41_mix_partial` riders added once in f32 (the world-2 all-reduce) + `d41_mix_finish` nslots 1;
//!   * `d41_hc_pre_norm_pw` == `d41_hc_pre_norm`;
//!   * `d41_q_kv_norm` == `d41x_dd_normrot` + `d41fw_swa_rows` (y, both rotated targets, the ring's q / s planes).

use crate::dsv41::gpu::attn::probe::{launch_of, sync, Rng};
use super::*;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use std::sync::Arc;

const DIM: usize = 5120;

fn dev() -> Arc<CudaDevice> {
    let d = CudaDevice::new(0).expect("GPU");
    crate::dsv41::gpu::load_module(&d).expect("gpu_dsv41 module");
    d
}
fn p<T>(s: &CudaSlice<T>) -> u64 { *s.device_ptr() as u64 }
fn bf(x: f32) -> u16 { half::bf16::from_f32(x).to_bits() }
fn go<A>(name: &str, grid: (u32, u32, u32), block: u32, a: &A) {
    crate::dsv41::gpu::launch_args(name, grid, (block, 1, 1), 0, launch_of(&dev_once()).stream, a).unwrap();
}
fn dev_once() -> Arc<CudaDevice> {
    static D: std::sync::OnceLock<Arc<CudaDevice>> = std::sync::OnceLock::new();
    D.get_or_init(dev).clone()
}

#[test]
#[ignore = "GPU"]
fn dsv41_pa1_gpu_mix_side_equals_the_reduce_path() {
    let d = dev_once();
    let k = 4 * DIM;
    let klen = k / 2;
    let nch = partial_chunks(klen);
    let mut rng = Rng(0x0A11_5EED);
    let fnw: Vec<f32> = (0..24 * k).map(|_| rng.gauss() * 0.01).collect();
    let fd = d.htod_sync_copy(&fnw).unwrap();
    let sc = d.htod_sync_copy(&[0.11f32, 0.07, 0.13]).unwrap();
    let base: Vec<f32> = (0..24).map(|_| rng.gauss() * 0.3).collect();
    let bs = d.htod_sync_copy(&base).unwrap();
    for rows in [1usize, 2, 3, 6, 8, 12, 24, 32] {
        let x: Vec<u16> = (0..rows * k).map(|_| bf(rng.gauss() * 2.0)).collect();
        let xd = d.htod_sync_copy(&x).unwrap();
        let rg = pick_rg(rows);
        let ng = rows.div_ceil(rg);
        let fin = |rider: u64, nslots: usize, out: u64| MixFinishArgs { part: rider, slot_stride: (rows * 32) as i64, row_stride: 32, scale: p(&sc), base: p(&bs), out, out_ld: 24,
                                                                         nslots: nslots as i32, rows: rows as i32, inv_k: 1.0 / k as f32, eps_norm: 1e-20, eps_hc: 1e-6, sinkhorn_iters: 20 };
        let args = |z: usize, part: u64, rider: u64, cnt: u64| MixPartialArgs { x: p(&xd), x_ld: k as i64, fn_: p(&fd) + (4 * z * klen) as u64, fn_ld: k as i64,
            part: part + (z * nch * rows * 32 * 4) as u64, chunk_stride: (rows * 32) as i64, row_stride: 32, rank_part: rider + (z * rows * 32 * 4) as u64, rp_row_stride: 32,
            counters: cnt + (z * ng * 4) as u64, k0: (z * klen) as i32, klen: klen as i32, rows: rows as i32, rg: rg as i32 };
        // the reduce path: each rank's launch, the f32 add of the two riders, finish over one slot
        let part = d.alloc_zeros::<f32>(2 * nch * rows * 32).unwrap();
        let rider = d.alloc_zeros::<f32>(2 * rows * 32).unwrap();
        let cnt = d.htod_sync_copy(&vec![0u32; 2 * ng]).unwrap();
        for z in 0..2 { go("d41_mix_partial", (nch as u32, ng as u32, 1), 128, &args(z, p(&part), p(&rider), p(&cnt))); }
        sync(&d).unwrap();
        let rv = d.dtoh_sync_copy(&rider).unwrap();
        let red: Vec<f32> = (0..rows * 32).map(|i| rv[i] + rv[rows * 32 + i]).collect();
        let redd = d.htod_sync_copy(&red).unwrap();
        let out_a = d.alloc_zeros::<f32>(rows * 24).unwrap();
        go("d41_mix_finish", ((rows as u32).div_ceil(8), 1, 1), 128, &fin(p(&redd), 1, p(&out_a)));
        // the side path: one launch for both halves (twice: the counters re-arm), finish over two slots
        let part2 = d.alloc_zeros::<f32>(2 * nch * rows * 32).unwrap();
        let rider2 = d.alloc_zeros::<f32>(2 * rows * 32).unwrap();
        let cnt2 = d.htod_sync_copy(&vec![0u32; 2 * ng]).unwrap();
        let out_b = d.alloc_zeros::<f32>(rows * 24).unwrap();
        for _ in 0..2 {
            let h = [args(0, p(&part2), p(&rider2), p(&cnt2)), args(1, p(&part2), p(&rider2), p(&cnt2))];
            go("d41_mix_partial2", (nch as u32, ng as u32, 2), 128, &h);
            go("d41_mix_finish", ((rows as u32).div_ceil(8), 1, 1), 128, &fin(p(&rider2), 2, p(&out_b)));
        }
        sync(&d).unwrap();
        let (a, b) = (d.dtoh_sync_copy(&out_a).unwrap(), d.dtoh_sync_copy(&out_b).unwrap());
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "rows {rows}: the side mix record differs from the reduce path's");
        assert!(d.dtoh_sync_copy(&cnt2).unwrap().iter().all(|&c| c == 0), "rows {rows}: counters not re-armed");
        // leg P6-5: world-4 side mix (`d41_mix_partial4` + `d41_mix_finish` nslots 4) == world-4 reduce `(p0 + p2) + (p1 + p3)` + finish nslots 1
        let klen4 = k / 4;
        let nch4 = partial_chunks(klen4);
        let args4 = |z: usize, part: u64, rider: u64, cnt: u64| MixPartialArgs { x: p(&xd), x_ld: k as i64, fn_: p(&fd) + (4 * z * klen4) as u64, fn_ld: k as i64,
            part: part + (z * nch4 * rows * 32 * 4) as u64, chunk_stride: (rows * 32) as i64, row_stride: 32, rank_part: rider + (z * rows * 32 * 4) as u64, rp_row_stride: 32,
            counters: cnt + (z * ng * 4) as u64, k0: (z * klen4) as i32, klen: klen4 as i32, rows: rows as i32, rg: rg as i32 };
        let part_r4 = d.alloc_zeros::<f32>(4 * nch4 * rows * 32).unwrap();
        let rider_r4 = d.alloc_zeros::<f32>(4 * rows * 32).unwrap();
        let cnt_r4 = d.htod_sync_copy(&vec![0u32; 4 * ng]).unwrap();
        for z in 0..4 { go("d41_mix_partial", (nch4 as u32, ng as u32, 1), 128, &args4(z, p(&part_r4), p(&rider_r4), p(&cnt_r4))); }
        sync(&d).unwrap();
        let rv4 = d.dtoh_sync_copy(&rider_r4).unwrap();
        let s = rows * 32;
        let red4: Vec<f32> = (0..s).map(|i| (rv4[i] + rv4[2 * s + i]) + (rv4[s + i] + rv4[3 * s + i])).collect();
        let red4d = d.htod_sync_copy(&red4).unwrap();
        let out_a4 = d.alloc_zeros::<f32>(rows * 24).unwrap();
        go("d41_mix_finish", ((rows as u32).div_ceil(8), 1, 1), 128, &fin(p(&red4d), 1, p(&out_a4)));
        let part_s4 = d.alloc_zeros::<f32>(4 * nch4 * rows * 32).unwrap();
        let rider_s4 = d.alloc_zeros::<f32>(4 * rows * 32).unwrap();
        let cnt_s4 = d.htod_sync_copy(&vec![0u32; 4 * ng]).unwrap();
        let out_b4 = d.alloc_zeros::<f32>(rows * 24).unwrap();
        for _ in 0..2 {
            let h4 = [args4(0, p(&part_s4), p(&rider_s4), p(&cnt_s4)), args4(1, p(&part_s4), p(&rider_s4), p(&cnt_s4)),
                      args4(2, p(&part_s4), p(&rider_s4), p(&cnt_s4)), args4(3, p(&part_s4), p(&rider_s4), p(&cnt_s4))];
            go("d41_mix_partial4", (nch4 as u32, ng as u32, 4), 128, &h4);
            go("d41_mix_finish", ((rows as u32).div_ceil(8), 1, 1), 128, &fin(p(&rider_s4), 4, p(&out_b4)));
        }
        sync(&d).unwrap();
        let (a4, b4) = (d.dtoh_sync_copy(&out_a4).unwrap(), d.dtoh_sync_copy(&out_b4).unwrap());
        assert!(a4.iter().zip(&b4).all(|(x, y)| x.to_bits() == y.to_bits()), "rows {rows}: TP4 side mix record differs from world-4 reduce path");
        assert!(d.dtoh_sync_copy(&cnt_s4).unwrap().iter().all(|&c| c == 0), "rows {rows}: TP4 counters not re-armed");
        println!("rows {rows:2}: TP2 & TP4 side mix records == reduce path records, bitwise ({} values)", rows * 24);
    }
}

#[test]
#[ignore = "GPU"]
fn dsv41_pa1_gpu_pre_norm_pw_equals() {
    let d = dev_once();
    let mut rng = Rng(0x0A11_0002);
    for rows in [1usize, 6, 24] {
        let x: Vec<u16> = (0..rows * 4 * DIM).map(|_| bf(rng.gauss() * 3.0)).collect();
        let pre: Vec<f32> = (0..rows * 24).map(|_| rng.gauss()).collect();
        let w: Vec<f32> = (0..DIM).map(|_| 1.0 + 0.1 * rng.gauss()).collect();
        let (xd, pd, wd) = (d.htod_sync_copy(&x).unwrap(), d.htod_sync_copy(&pre).unwrap(), d.htod_sync_copy(&w).unwrap());
        let (ya, yb) = (d.alloc_zeros::<u16>(rows * DIM).unwrap(), d.alloc_zeros::<u16>(rows * DIM).unwrap());
        let a = |y: u64| HcPreNormArgs { x: p(&xd), x_ld: (4 * DIM) as i64, pre: p(&pd), pre_stride: 24, w: p(&wd), y_norm: y, yn_ld: DIM as i64, y_raw: 0, yr_ld: 0,
                                         w_fmt: crate::dsv41::gpu::abi::fmt::F32, dim: DIM as i32, rows: rows as i32, eps: 1e-20, inv_dim: 1.0 / DIM as f32 };
        go("d41_hc_pre_norm", (rows as u32, 1, 1), 256, &a(p(&ya)));
        go("d41_hc_pre_norm_pw", (rows as u32, 1, 1), 256, &a(p(&yb)));
        sync(&d).unwrap();
        assert_eq!(d.dtoh_sync_copy(&ya).unwrap(), d.dtoh_sync_copy(&yb).unwrap(), "rows {rows}: d41_hc_pre_norm_pw differs");
        println!("rows {rows:2}: d41_hc_pre_norm_pw == d41_hc_pre_norm, bitwise");
    }
}

#[test]
#[ignore = "GPU"]
fn dsv41_pa1_gpu_q_kv_norm_equals() {
    use crate::dsv41::tp::tpk::SwaRowsArgs;
    use crate::dsv41::weights::exl3::dense_dec::normrot_args;
    let d = dev_once();
    let mut rng = Rng(0x0A11_0003);
    let ql = 1536usize;
    let slots = 134usize;
    for rows in [1usize, 2, 6, 12, 24] {
        let qa: Vec<u16> = (0..rows * ql).map(|_| bf(rng.gauss() * 4.0)).collect();
        let w: Vec<f32> = (0..ql).map(|_| 1.0 + 0.1 * rng.gauss()).collect();
        let suh: Vec<Vec<u16>> = (0..2).map(|_| (0..ql).map(|_| half::f16::from_f32(if rng.next() & 1 == 0 { 0.9 } else { -1.1 }).to_bits()).collect()).collect();
        let kv: Vec<u16> = (0..rows * 512).map(|_| bf(rng.gauss() * 3.0)).collect();
        let kw: Vec<f32> = (0..512).map(|_| 1.0 + 0.1 * rng.gauss()).collect();
        let freq = crate::dsv41::gpu::attn::rope_inv_freq(160000.0, Some((16.0, 65536.0, 32.0, 1.0))).unwrap();
        let pos: Vec<i32> = (0..rows as i32).map(|r| 4000 + 37 * r).collect();
        let (qad, wd, kvd, kwd, fd, posd) = (d.htod_sync_copy(&qa).unwrap(), d.htod_sync_copy(&w).unwrap(), d.htod_sync_copy(&kv).unwrap(), d.htod_sync_copy(&kw).unwrap(),
                                             d.htod_sync_copy(&freq).unwrap(), d.htod_sync_copy(&pos).unwrap());
        let sd: Vec<CudaSlice<u16>> = suh.iter().map(|v| d.htod_sync_copy(v).unwrap()).collect();
        let mut outs = Vec::new();
        for fused in [false, true] {
            let y = d.alloc_zeros::<u16>(rows * ql).unwrap();
            let xh: Vec<CudaSlice<u16>> = (0..2).map(|_| d.htod_sync_copy(&vec![0xFFFFu16; rows * ql]).unwrap()).collect();
            let ring_q = d.htod_sync_copy(&vec![0xA5u8; slots * 512]).unwrap();
            let ring_s = d.htod_sync_copy(&vec![0x5Au8; slots * 16]).unwrap();
            let tq = d.htod_sync_copy(&vec![p(&ring_q); rows]).unwrap();
            let ts = d.htod_sync_copy(&vec![p(&ring_s); rows]).unwrap();
            let nr = normrot_args(p(&qad), ql, p(&wd), crate::dsv41::gpu::abi::fmt::F32, p(&y), ql, ql, rows, 1e-20,
                                  &[(p(&sd[0]), p(&xh[0]), ql), (p(&sd[1]), p(&xh[1]), ql)]).unwrap();
            let sw = SwaRowsArgs { kv_raw: p(&kvd), norm_w: p(&kwd), inv_freq: p(&fd), pos: p(&posd), ring_q: p(&tq), ring_s: p(&ts), rows: rows as i32,
                                   norm_fmt: crate::dsv41::gpu::abi::fmt::F32, eps: 1e-20, pad: 0 };
            if fused {
                #[repr(C)]
                struct Q { nr: crate::dsv41::weights::exl3::dense_dec::NormRotArgs, sw: SwaRowsArgs }
                go("d41_q_kv_norm", (rows as u32, 2, 1), 256, &Q { nr, sw });
            } else {
                go("d41x_dd_normrot", (rows as u32, 1, 1), 256, &nr);
                go("d41fw_swa_rows", (rows as u32, 1, 1), 128, &sw);
            }
            sync(&d).unwrap();
            outs.push((d.dtoh_sync_copy(&y).unwrap(), d.dtoh_sync_copy(&xh[0]).unwrap(), d.dtoh_sync_copy(&xh[1]).unwrap(),
                       d.dtoh_sync_copy(&ring_q).unwrap(), d.dtoh_sync_copy(&ring_s).unwrap()));
        }
        assert!(outs[0] == outs[1], "rows {rows}: d41_q_kv_norm differs from normrot + swa_rows");
        println!("rows {rows:2}: d41_q_kv_norm == d41x_dd_normrot + d41fw_swa_rows (y, 2 rotated targets, ring q / s planes), bitwise");
    }
}
