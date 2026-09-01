use std::sync::Arc;

use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use gb10_inference::quant;
use half::bf16;

fn load() -> Arc<CudaDevice> {
    let dev = CudaDevice::new(0).expect("CUDA device 0");
    let ptx =
        Ptx::from_src(std::fs::read_to_string("src/ptx/gpu_batch.ptx").expect("gpu_batch.ptx"));
    let names = [
        "gptq_scale_stats_f32_b",
        "gptq_scale_stats_bf16_b",
        "gptq_static_scales_b",
        "gptq_static_scales_hessian_b",
        "gptq_permute_w_b",
        "gptq_permute_h_b",
        "gptq_sweep_static_b",
    ];
    dev.load_ptx(ptx, "gptq_kernels_test", &names)
        .expect("load GPTQ kernels");
    dev
}

fn best_scale(x: &[f32], s_tensor: f32, nclip: usize) -> u8 {
    const RATIOS: [f32; 7] = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7];
    let amax = x.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    let mut best = (0u8, f32::INFINITY);
    for &ratio in RATIOS.iter().take(nclip) {
        let raw = if amax > 0.0 {
            amax * ratio / quant::E2M1_MAX / s_tensor
        } else {
            0.0
        };
        let code = quant::f32_to_e4m3(raw);
        let scale = quant::e4m3_to_f32(code) * s_tensor;
        let err = x
            .iter()
            .map(|&v| {
                let q = if scale > 0.0 {
                    quant::e2m1_to_f32(quant::f32_to_e2m1(v / scale)) * scale
                } else {
                    0.0
                };
                (q - v) * (q - v)
            })
            .sum();
        if err < best.1 {
            best = (code, err);
        }
    }
    best.0
}

#[test]
fn gptq_scale_stats_and_static_sweep_match_cpu() {
    let dev = load();
    let (m, k, nclip) = (2usize, 16usize, 7usize);
    let s_tensor = 0.00137f32;
    let w: Vec<f32> = (0..m * k)
        .map(|i| {
            let x = i as f32 - 15.5;
            (x * 0.43).sin() * (0.3 + (i % 11) as f32 * 0.17)
        })
        .collect();
    let wd = dev.htod_sync_copy(&w).unwrap();

    let stats = dev.alloc_zeros::<f64>(3).unwrap();
    let ngroups = (m * k / 16) as i64;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let f = dev
        .get_func("gptq_kernels_test", "gptq_scale_stats_f32_b")
        .unwrap();
    unsafe {
        f.launch(
            cfg,
            (&stats, &wd, ngroups, s_tensor.to_bits(), nclip as i32),
        )
        .unwrap();
    }
    dev.synchronize().unwrap();
    let got_stats = dev.dtoh_sync_copy(&stats).unwrap();

    let mut expected_stats = [0.0f64; 3];
    for x in w.chunks_exact(16) {
        let scode = best_scale(x, s_tensor, nclip);
        let es = quant::e4m3_to_f32(scode);
        let scale = es * s_tensor;
        for &v in x {
            let code = quant::f32_to_e2m1(v / scale);
            let z = es * quant::e2m1_to_f32(code);
            expected_stats[0] += v as f64 * z as f64;
            expected_stats[1] += z as f64 * z as f64;
            let d = quant::e2m1_to_f32(code) * scale - v;
            expected_stats[2] += (d * d) as f64;
        }
    }
    for i in 0..3 {
        let tol = 2e-5 * expected_stats[i].abs().max(1.0);
        assert!(
            (got_stats[i] - expected_stats[i]).abs() <= tol,
            "stats[{i}] gpu={} cpu={} tol={tol}",
            got_stats[i],
            expected_stats[i]
        );
    }

    let scales = dev.alloc_zeros::<u8>(m * k / 16).unwrap();
    let f = dev
        .get_func("gptq_kernels_test", "gptq_static_scales_b")
        .unwrap();
    unsafe {
        f.launch(
            cfg,
            (&scales, &wd, ngroups, s_tensor.to_bits(), nclip as i32),
        )
        .unwrap();
    }

    let mut perm: Vec<i32> = (0..k as i32).collect();
    perm.rotate_left(5);
    let pd = dev.htod_sync_copy(&perm).unwrap();
    let wp = dev.alloc_zeros::<f32>(m * k).unwrap();
    let f = dev
        .get_func("gptq_kernels_test", "gptq_permute_w_b")
        .unwrap();
    unsafe {
        f.launch(cfg, (&wp, &wd, &pd, m as i32, k as i32)).unwrap();
    }

    let mut u = vec![0.0f32; k * k];
    for i in 0..k {
        u[i * k + i] = 1.0;
    }
    let ud = dev.htod_sync_copy(&u).unwrap();
    let qw = dev.alloc_zeros::<u8>(m * k / 2).unwrap();
    let err = dev.alloc_zeros::<f32>(m * k).unwrap();
    let f = dev
        .get_func("gptq_kernels_test", "gptq_sweep_static_b")
        .unwrap();
    unsafe {
        f.launch(
            cfg,
            (
                &wp,
                &qw,
                &scales,
                &err,
                &ud,
                &pd,
                m as i32,
                k as i32,
                0i32,
                k as i32,
                s_tensor.to_bits(),
            ),
        )
        .unwrap();
    }
    dev.synchronize().unwrap();

    let got_scales = dev.dtoh_sync_copy(&scales).unwrap();
    let got_qw = dev.dtoh_sync_copy(&qw).unwrap();
    for r in 0..m {
        let scode = best_scale(&w[r * k..(r + 1) * k], s_tensor, nclip);
        assert_eq!(got_scales[r], scode);
        let scale = quant::e4m3_to_f32(scode) * s_tensor;
        for c in 0..k {
            let expected = quant::f32_to_e2m1(w[r * k + c] / scale);
            let byte = got_qw[r * k / 2 + c / 2];
            let got = if c & 1 == 0 { byte & 0x0f } else { byte >> 4 };
            assert_eq!(got, expected, "row {r} col {c}");
        }
    }
}
#[test]
fn gptq_bf16_rotated_stats_and_hessian_permutation_match_cpu() {
    let dev = load();
    let (k, nclip) = (16usize, 7usize);
    let s_tensor = 0.00111f32;
    let wb: Vec<bf16> = (0..32)
        .map(|i| {
            let x = i as f32 - 15.5;
            bf16::from_f32((x * 0.37).cos() * (0.4 + (i % 9) as f32 * 0.13))
        })
        .collect();
    let mut rotated: Vec<f32> = wb.iter().map(|v| v.to_f32()).collect();
    for x in rotated.chunks_exact_mut(16) {
        for len in [1usize, 2, 4, 8] {
            for i in (0..16).step_by(2 * len) {
                for j in i..i + len {
                    let (a, b) = (x[j], x[j + len]);
                    x[j] = a + b;
                    x[j + len] = a - b;
                }
            }
        }
        for v in x {
            *v *= 0.25;
        }
    }
    let wbd = dev.htod_sync_copy(&wb).unwrap();
    let wrd = dev.htod_sync_copy(&rotated).unwrap();
    let bs = dev.alloc_zeros::<f64>(3).unwrap();
    let fs = dev.alloc_zeros::<f64>(3).unwrap();
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let fb = dev
        .get_func("gptq_kernels_test", "gptq_scale_stats_bf16_b")
        .unwrap();
    unsafe {
        fb.launch(
            cfg,
            (&bs, &wbd, 2i64, 1i32, s_tensor.to_bits(), nclip as i32),
        )
        .unwrap();
    }
    let ff = dev
        .get_func("gptq_kernels_test", "gptq_scale_stats_f32_b")
        .unwrap();
    unsafe {
        ff.launch(cfg, (&fs, &wrd, 2i64, s_tensor.to_bits(), nclip as i32))
            .unwrap();
    }
    dev.synchronize().unwrap();
    let (got, expected) = (
        dev.dtoh_sync_copy(&bs).unwrap(),
        dev.dtoh_sync_copy(&fs).unwrap(),
    );
    for i in 0..3 {
        let tol = 2e-5 * expected[i].abs().max(1.0);
        assert!(
            (got[i] - expected[i]).abs() <= tol,
            "rotated bf16 stats[{i}] gpu={} reference={} tol={tol}",
            got[i],
            expected[i]
        );
    }

    let mut perm: Vec<i32> = (0..k as i32).collect();
    perm.rotate_left(5);
    let pd = dev.htod_sync_copy(&perm).unwrap();
    let h: Vec<f32> = (0..k * k).map(|i| i as f32 * 0.03125).collect();
    let hd = dev.htod_sync_copy(&h).unwrap();
    let hp = dev.alloc_zeros::<f32>(k * k).unwrap();
    let f = dev
        .get_func("gptq_kernels_test", "gptq_permute_h_b")
        .unwrap();
    unsafe {
        f.launch(cfg, (&hp, &hd, &pd, k as i32)).unwrap();
    }
    dev.synchronize().unwrap();
    let got_h = dev.dtoh_sync_copy(&hp).unwrap();
    for r in 0..k {
        for c in 0..k {
            assert_eq!(got_h[r * k + c], h[perm[r] as usize * k + perm[c] as usize]);
        }
    }
}

fn hessian_loss(w: &[f32], h: &[f32], s_tensor: f32, code: u8) -> f32 {
    let scale = quant::e4m3_to_f32(code) * s_tensor;
    let dw: Vec<f32> = w
        .iter()
        .map(|&v| v - quant::e2m1_to_f32(quant::f32_to_e2m1(v / scale)) * scale)
        .collect();
    let mut loss = 0.0f32;
    for i in 0..16 {
        for j in 0..16 {
            loss += dw[i] * h[i * 16 + j] * dw[j];
        }
    }
    loss
}

#[test]
fn gptq_local_hessian_scale_sweep_matches_cpu() {
    let dev = load();
    let (m, k) = (2usize, 16usize);
    let mut w = vec![0.0f32; m * k];
    for r in 0..m {
        w[r * k] = 0.37 + r as f32 * 0.08;
        w[r * k + 1] = -0.43 - r as f32 * 0.06;
        for i in 2..k {
            w[r * k + i] = if i & 1 == 0 {
                4.0 + i as f32 * 0.14 + r as f32 * 0.03
            } else {
                -4.3 - i as f32 * 0.11 - r as f32 * 0.02
            };
        }
    }
    let mut h = vec![0.0f32; k * k];
    for i in 0..k {
        h[i * k + i] = if i < 2 { 1.0e6 } else { 1.0 };
    }
    let amax = w.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    let s_tensor = amax / (quant::E2M1_MAX * quant::E4M3_MAX);
    let wd = dev.htod_sync_copy(&w).unwrap();
    let hd = dev.htod_sync_copy(&h).unwrap();
    let scales = dev.alloc_zeros::<u8>(m).unwrap();
    let fallbacks = dev.alloc_zeros::<u64>(1).unwrap();
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let f = dev
        .get_func("gptq_kernels_test", "gptq_static_scales_hessian_b")
        .unwrap();
    unsafe {
        f.launch(
            cfg,
            (
                &scales,
                &wd,
                &hd,
                &fallbacks,
                m as i32,
                k as i32,
                s_tensor.to_bits(),
            ),
        )
        .unwrap();
    }
    dev.synchronize().unwrap();
    assert_eq!(dev.dtoh_sync_copy(&fallbacks).unwrap(), vec![0]);
    let got = dev.dtoh_sync_copy(&scales).unwrap();
    for r in 0..m {
        let row = &w[r * k..(r + 1) * k];
        let expected = (1u8..=126)
            .min_by(|&a, &b| {
                hessian_loss(row, &h, s_tensor, a)
                    .total_cmp(&hessian_loss(row, &h, s_tensor, b))
                    .then_with(|| a.cmp(&b))
            })
            .unwrap();
        assert_eq!(got[r], expected, "row {r}");
    }
}
