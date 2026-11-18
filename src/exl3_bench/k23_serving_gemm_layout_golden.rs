//! Host-side proof tests (module `k23_serving_gemm_layout_golden`), moved out of `exl3_bench.rs`; no GPU unless a test says so.
//! Part of the `exl3_bench` probe module (see `mod.rs` for the map); test bodies are unchanged.
    use super::*;

    #[ignore = "INVALID HARNESS 2026-10-30: same wrong-body harness as k23_a1b3_fh_gemm_vs_wheel, interleaved layout variant. Superseded by k23_grouped_xh_bits1_vs_wheel."]
    #[test]
    fn k23_a1b3_fh_interleaved_vs_wheel() {
        let Ok(j) = std::fs::read_to_string("tests/k23_golden_l4e7up.json")
            else { eprintln!("[skip] lab fixture tests/k23_golden_l4e7up.json not present"); return; };
        let v: serde_json::Value = serde_json::from_str(&j).unwrap();
        let kb = v["kb"].as_u64().unwrap() as usize; // 160
        let gnb = v["nb"].as_u64().unwrap() as usize; // 40 (mi=640)
        let k = kb * 16; // 2560
        let mi = gnb * 16; // 640
        let m = 8usize;
        let u16s = |key: &str| -> Vec<u16> {
            v[key].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as u16).collect()
        };
        let quad: Vec<u16> = u16s("trellis_i16").iter().map(|b| *b as u16).collect();
        let suh = u16s("suh_bits");
        let svh = u16s("svh_bits");
        let full = u16s("full_f16_bits");
        assert_eq!(quad.len(), kb * gnb * 24);
        let gw = 24usize; // 16 * k2 / 2, k2 = 3

        // gu_tr_h exactly as the loader builds it: per k16 row r: [strip][strip]
        let mut gu: Vec<u16> = Vec::with_capacity(kb * gnb * 2 * gw);
        for r in 0..kb {
            let strip = &quad[r * gnb * gw..(r + 1) * gnb * gw];
            gu.extend_from_slice(strip);
            gu.extend_from_slice(strip);
        }
        // svh_all for N = 2*mi: gate half then up half (both = up's svh)
        let mut svh2: Vec<u16> = Vec::with_capacity(2 * mi);
        svh2.extend_from_slice(&svh);
        svh2.extend_from_slice(&svh);

        let x: Vec<half::f16> = (0..m * k)
            .map(|i| half::f16::from_f32((((i * 37) % 251) as f32 - 125.0) / 128.0))
            .collect();
        let xf: Vec<f32> = x.iter().map(|h| h.to_f32()).collect();
        let wf: Vec<f32> = full.iter().map(|b| half::f16::from_bits(*b).to_f32()).collect();
        // reference for the UP half: columns mi..2*mi of x @ W (W = up full weight [k][mi])
        let mut want = vec![0f64; m * mi];
        for r in 0..m {
            for c in 0..mi {
                let mut acc = 0f64;
                for ki in 0..k {
                    acc += xf[r * k + ki] as f64 * wf[ki * mi + c] as f64;
                }
                want[r * mi + c] = acc;
            }
        }

        let dev = dev0().unwrap();
        load_module(&dev).unwrap();
        let stream = fork_blocking_stream(&dev);
        let d_tr = dev.htod_sync_copy(&gu).unwrap();
        let d_offs = dev.htod_sync_copy(&[0u64]).unwrap();
        let d_x = dev.htod_sync_copy(&x).unwrap();
        let hb = |bits: &[u16]| -> Vec<half::f16> { bits.iter().map(|b| half::f16::from_bits(*b)).collect() };
        let d_suh = dev.htod_sync_copy(&hb(&suh)).unwrap();
        let d_svh = dev.htod_sync_copy(&hb(&svh2)).unwrap();
        let d_idx = dev.htod_sync_copy(&[0i32]).unwrap();
        let d_esel = dev.htod_sync_copy(&[1i32]).unwrap();
        let n2 = 2 * mi;
        let mut d_y = unsafe { dev.alloc::<half::f16>(m * n2) }.unwrap();
        let f = dev.get_func(MODULE, "xq_gemm_grouped_a1b3_fh").unwrap();
        let tiles = (n2 / 128) as u32;
        let smem = (m * (k + 8) * 2) as u32;
        unsafe {
            f.clone()
                .launch_on_stream(
                    &stream,
                    LaunchConfig { grid_dim: (tiles, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: smem },
                    (&d_tr, &d_offs, &d_x, 0i32, &d_suh, &d_svh, &d_idx, &mut d_y,
                     m as i32, k as i32, n2 as i32, &d_esel),
                )
                .unwrap();
        }
        dev.synchronize().unwrap();
        let got = dev.dtoh_sync_copy(&d_y).unwrap();
        let mut num = 0f64;
        let mut den = 0f64;
        let mut first = Vec::new();
        for r in 0..m {
            for c in 0..mi {
                let g = got[r * n2 + mi + c].to_f32() as f64;
                let w = want[r * mi + c];
                num += (g - w) * (g - w);
                den += w * w;
                if (g - w).abs() > 0.05 * w.abs().max(1.0) && first.len() < 6 {
                    first.push((r, c, g, w));
                }
            }
        }
        let rel = (num / den).sqrt();
        eprintln!("k23 INTERLEAVED up-half rel-L2 = {:.6} (M={} K={} N={})", rel, m, k, n2);
        // also report the gate half vs the same reference (both strips are up bytes, so the
        // gate half should ALSO match want if the body is symmetric-correct)
        let mut numg = 0f64;
        let mut deng = 0f64;
        for r in 0..m {
            for c in 0..mi {
                let g = got[r * n2 + c].to_f32() as f64;
                let w = want[r * mi + c];
                numg += (g - w) * (g - w);
                deng += w * w;
            }
        }
        eprintln!("k23 INTERLEAVED gate-half rel-L2 = {:.6} (expected to match: strips identical)", (numg / deng).sqrt());
        assert!(rel < 0.05, "k2=3 interleaved serving GEMM up-half differs: rel-L2 {rel:.5}, first {first:?}");
    }
