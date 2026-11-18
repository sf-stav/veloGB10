//! Host-side proof tests (module `k23_serving_real_call`), moved out of `exl3_bench.rs`; no GPU unless a test says so.
//! Part of the `exl3_bench` probe module (see `mod.rs` for the map); test bodies are unchanged.
    use super::*;

    #[test]
    fn k23_grouped_xh_bits1_vs_wheel() {
        let Ok(j) = std::fs::read_to_string("tests/k23_golden_l4e7up.json")
            else { eprintln!("[skip] lab fixture tests/k23_golden_l4e7up.json not present"); return; };
        let v: serde_json::Value = serde_json::from_str(&j).unwrap();
        let kb = v["kb"].as_u64().unwrap() as usize; // 160
        let nb = v["nb"].as_u64().unwrap() as usize; // 40
        let k = kb * 16; // 2560
        let mi = nb * 16; // 640
        let m = 8usize;
        let u16s = |key: &str| -> Vec<u16> {
            v[key].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as u16).collect()
        };
        let quad: Vec<u16> = u16s("trellis_i16");
        let suh = u16s("suh_bits");
        let svh = u16s("svh_bits");
        let full = u16s("full_f16_bits");
        assert_eq!(quad.len(), kb * nb * 24);

        let x: Vec<half::f16> = (0..m * k)
            .map(|i| half::f16::from_f32((((i * 37) % 251) as f32 - 125.0) / 128.0))
            .collect();
        let xf: Vec<f32> = x.iter().map(|h| h.to_f32()).collect();
        let wf: Vec<f32> = full.iter().map(|b| half::f16::from_bits(*b).to_f32()).collect();
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
        let d_tr = dev.htod_sync_copy(&quad).unwrap();
        let d_offs = dev.htod_sync_copy(&[0u64]).unwrap();
        // xh_e = f16(H128(x * suh)) per the serving sequence; the epilogue svh applied after.
        // For an exact reference we fold suh/svh into the reference instead (W_full already
        // includes hadamards+scales), so feed the kernel H(x*suh) and compare after svh:
        // simpler: feed x directly is NOT the kernel contract. We compute xh on host with the
        // same butterfly as xq_had_suh_multi? Out of scope here: instead compare RAW kernel
        // output y vs reference built from W_inner (inner = pre-hadamard domain):
        // y = xh @ W_inner^T where xh = H(x*suh). Reference wheel tensors are full-domain;
        // inner bits are in the fixture (inner_f16_bits).
        let inner: Vec<f32> = v["inner_f16_bits"].as_array().unwrap().iter()
            .map(|b| half::f16::from_bits(b.as_i64().unwrap() as u16).to_f32()).collect();
        // host hadamard H128 (sylar order 0..127 xor pattern) — replicate xq_had_suh_multi:
        // H128[x] = sum over the butterfly; use the simple FWHT with the standard EXL3 order.
        fn fwht128(a: &mut [f32]) {
            let n = 128usize;
            let mut h = 1usize;
            while h < n {
                let mut i = 0;
                while i < n {
                    for j in i..i + h {
                        let x = a[j];
                        let y = a[j + h];
                        a[j] = x + y;
                        a[j + h] = x - y;
                    }
                    i += h * 2;
                }
                h *= 2;
            }
        }
        let mut xh = vec![0f32; m * k];
        for r in 0..m {
            for b in 0..k / 128 {
                let mut blk = [0f32; 128];
                for (t, slot) in blk.iter_mut().enumerate() {
                    let su = half::f16::from_bits(suh[b * 128 + t]).to_f32();
                    *slot = xf[r * k + b * 128 + t] * su;
                }
                fwht128(&mut blk);
                xh[r * k + b * 128..r * k + b * 128 + 128].copy_from_slice(&blk);
            }
        }
        let xh16: Vec<half::f16> = xh.iter().map(|v| half::f16::from_f32(*v)).collect();
        // reference: y[r][c] = sum_k xh[r][k] * inner[k][c]  (inner row-major [k][mi])
        let mut want_inner = vec![0f64; m * mi];
        for r in 0..m {
            for c in 0..mi {
                let mut acc = 0f64;
                for ki in 0..k {
                    acc += xh[r * k + ki] as f64 * inner[ki * mi + c] as f64;
                }
                want_inner[r * mi + c] = acc;
            }
        }
        for r in 0..m {
            for c in 0..mi {
                let mut acc = 0f64;
                for ki in 0..k {
                    acc += xh[r * k + ki] as f64 * inner[ki * mi + c] as f64;
                }
                want_inner[r * mi + c] = acc;
            }
        }

        let d_x = dev.htod_sync_copy(&xh16).unwrap();
        let d_idx = dev.htod_sync_copy(&[0i32]).unwrap();
        let d_esel = dev.htod_sync_copy(&[1i32]).unwrap();
        // PRE-INIT y to a sentinel so a no-op is VISIBLE (alloc does not zero)
        let sentinel = half::f16::from_f32(-1234.5f32);
        let mut d_y = unsafe { dev.alloc::<half::f16>(m * mi) }.unwrap();
        let init = vec![sentinel; m * mi];
        unsafe {
            let mut tmp = dev.htod_sync_copy(&init).unwrap();
            std::mem::swap(&mut tmp, &mut d_y);
        }
        let f = dev.get_func(MODULE, "xq_gemm_grouped_xh").unwrap();
        let smem = a1_smem_pub_for_test(m, k);
        unsafe {
            f.clone()
                .launch_on_stream(
                    &stream,
                    LaunchConfig { grid_dim: ((mi / 128) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: smem },
                    (&d_tr, &d_offs, &d_x, &mut d_y, m as i32, k as i32, mi as i32, crate::exl3_forward::exl3_kernel_bits(3) as i32, &d_esel), // K35: the EXL3_FRAC_BITS(1) code (was bits = 1)
                )
                .unwrap();
        }
        dev.synchronize().unwrap();
        let got = dev.dtoh_sync_copy(&d_y).unwrap();
        let unchanged = got.iter().filter(|v| **v == sentinel).count();
        let mut num = 0f64;
        let mut den = 0f64;
        for r in 0..m {
            for c in 0..mi {
                let g = got[r * mi + c].to_f32() as f64;
                let w = want_inner[r * mi + c];
                num += (g - w) * (g - w);
                den += w * w;
            }
        }
        let rel = (num / den).sqrt();
        eprintln!("k23 xh bits=1: sentinel-unchanged {unchanged}/{} rel-L2 {rel:.6}", m * mi);
        if unchanged == m * mi {
            panic!("NO-OP CONFIRMED: xq_gemm_grouped_xh with bits=1 wrote NOTHING (all sentinel) — the k2=3 routed experts are silently skipped");
        }
        assert!(rel < 0.05, "k2=3 real-call GEMM differs: rel-L2 {rel:.5}");
    }
