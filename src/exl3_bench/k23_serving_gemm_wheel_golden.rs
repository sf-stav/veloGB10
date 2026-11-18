//! Host-side proof tests (module `k23_serving_gemm_wheel_golden`), moved out of `exl3_bench.rs`; no GPU unless a test says so.
//! Part of the `exl3_bench` probe module (see `mod.rs` for the map); test bodies are unchanged.
    use super::*;

    #[ignore = "INVALID HARNESS 2026-10-30: a1b3_fh<3> is the k2=6-class INTEGER ring body ((t+257)*BITS b1); never reachable for k2=3 serving (fh path requires the a1b3 name; grouped_fn returns xh at bits=1). Superseded by k23_grouped_xh_bits1_vs_wheel."]
    #[test]
    fn k23_a1b3_fh_gemm_vs_wheel() {
        let Ok(j) = std::fs::read_to_string("tests/k23_golden_l4e7up.json")
            else { eprintln!("[skip] lab fixture tests/k23_golden_l4e7up.json not present"); return; };
        let v: serde_json::Value = serde_json::from_str(&j).unwrap();
        let kb = v["kb"].as_u64().unwrap() as usize; // 160
        let nb = v["nb"].as_u64().unwrap() as usize; // 40
        let k = kb * 16; // 2560
        let n = nb * 16; // 640
        let m = 8usize; // the bridge's row-block width
        let u16s = |key: &str| -> Vec<u16> {
            v[key].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as u16).collect()
        };
        let trellis: Vec<i16> = u16s("trellis_i16").iter().map(|b| *b as i16).collect();
        let suh = u16s("suh_bits");
        let svh = u16s("svh_bits");
        let full = u16s("full_f16_bits"); // W_full row-major [k][n]
        assert_eq!(trellis.len(), kb * nb * 24);
        assert_eq!(suh.len(), k);
        assert_eq!(svh.len(), n);
        assert_eq!(full.len(), k * n);

        // deterministic x pattern, small magnitudes (f16-exact values)
        let x: Vec<half::f16> = (0..m * k)
            .map(|i| half::f16::from_f32((((i * 37) % 251) as f32 - 125.0) / 128.0))
            .collect();
        // host reference: f64 accumulate over the wheel's full weight
        let xf: Vec<f32> = x.iter().map(|h| h.to_f32()).collect();
        let wf: Vec<f32> = full.iter().map(|b| half::f16::from_bits(*b).to_f32()).collect();
        let mut want = vec![0f64; m * n];
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = 0f64;
                for ki in 0..k {
                    acc += xf[mi * k + ki] as f64 * wf[ki * n + ni] as f64;
                }
                want[mi * n + ni] = acc;
            }
        }

        let dev = dev0().unwrap();
        load_module(&dev).unwrap();
        let stream = fork_blocking_stream(&dev);
        let raw: Vec<u8> = trellis.iter().flat_map(|x| x.to_le_bytes()).collect();
        let d_tr = dev.htod_sync_copy(&raw).unwrap();
        let d_offs = dev.htod_sync_copy(&[0u64]).unwrap();
        let d_x = dev.htod_sync_copy(&x).unwrap();
        let d_suh = dev.htod_sync_copy(&half_bits_vec(&suh)).unwrap();
        let d_svh = dev.htod_sync_copy(&half_bits_vec(&svh)).unwrap();
        let d_idx = dev.htod_sync_copy(&[0i32]).unwrap();
        let d_esel = dev.htod_sync_copy(&[1i32]).unwrap();
        let mut d_y = unsafe { dev.alloc::<half::f16>(m * n) }.unwrap();
        let f = dev.get_func(MODULE, "xq_gemm_grouped_a1b3_fh").unwrap();
        let tiles = (n / 128) as u32;
        let smem = (m * (k + 8) * 2) as u32; // a1_smem
        unsafe {
            f.clone()
                .launch_on_stream(
                    &stream,
                    LaunchConfig { grid_dim: (tiles, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: smem },
                    (&d_tr, &d_offs, &d_x, 0i32, &d_suh, &d_svh, &d_idx, &mut d_y,
                     m as i32, k as i32, n as i32, &d_esel),
                )
                .unwrap();
        }
        dev.synchronize().unwrap();
        let got = dev.dtoh_sync_copy(&d_y).unwrap();
        // rel-L2 (f16 out vs f64 ref): correct decode ~<1e-2; wrong quad/decode ~0.5+
        let mut num = 0f64;
        let mut den = 0f64;
        let mut first = Vec::new();
        for i in 0..got.len() {
            let g = got[i].to_f32() as f64;
            num += (g - want[i]) * (g - want[i]);
            den += want[i] * want[i];
            if (g - want[i]).abs() > 0.02 * want[i].abs().max(1.0) && first.len() < 8 {
                first.push((i, g, want[i]));
            }
        }
        let rel = (num / den).sqrt();
        eprintln!("k23 fh GEMM rel-L2 = {:.6} ({}x{}x{}), first big diffs {:?}", rel, m, k, n, first);
        assert!(rel < 0.02, "k2=3 SERVING GEMM (a1b3_fh) differs from wheel full-weight: rel-L2 {rel:.5}");
    }

    fn half_bits_vec(bits: &[u16]) -> Vec<half::f16> {
        bits.iter().map(|b| half::f16::from_bits(*b)).collect()
    }
