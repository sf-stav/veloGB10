//! Host-side proof tests (module `k35_kernel_wheel_golden`), moved out of `exl3_bench.rs`; no GPU unless a test says so.
//! Part of the `exl3_bench` probe module (see `mod.rs` for the map); test bodies are unchanged.
    use super::*;
    use crate::exl3::k35_wheel_golden::{load, Golden};

    fn fwht128(a: &mut [f32]) {
        let n = 128usize;
        let mut h = 1usize;
        while h < n {
            let mut i = 0;
            while i < n {
                for j in i..i + h { let (x, y) = (a[j], a[j + h]); a[j] = x + y; a[j + h] = x - y; }
                i += h * 2;
            }
            h *= 2;
        }
    }

    /// xh = H128(x * suh) per 128-block, the serving kernel's A operand (synthetic x, deterministic).
    fn make_xh(g: &Golden, m: usize, seed: usize) -> (Vec<half::f16>, Vec<f32>) {
        let k = g.kb * 16;
        let x: Vec<f32> = (0..m * k).map(|i| ((((i * 37 + seed * 101) % 251) as f32) - 125.0) / 128.0).collect();
        let mut xh = vec![0f32; m * k];
        for r in 0..m {
            for b in 0..k / 128 {
                let mut blk = [0f32; 128];
                for (t, slot) in blk.iter_mut().enumerate() {
                    *slot = x[r * k + b * 128 + t] * half::f16::from_bits(g.suh[b * 128 + t]).to_f32();
                }
                fwht128(&mut blk);
                xh[r * k + b * 128..r * k + b * 128 + 128].copy_from_slice(&blk);
            }
        }
        let xh16: Vec<half::f16> = xh.iter().map(|v| half::f16::from_f32(*v)).collect();
        let xhf: Vec<f32> = xh16.iter().map(|v| v.to_f32()).collect(); // what the kernel actually reads
        (xh16, xhf)
    }

    fn dump_vs_wheel(stem: &str) {
        let Some(g) = load(stem) else { return; };
        let dev = dev0().unwrap();
        load_module(&dev).unwrap();
        let d_rc = dev.htod_sync_copy(&tmap_inv_table()).unwrap();
        let f = dev.get_func(MODULE, "exl3_decode_dump").unwrap();
        let stream = fork_blocking_stream(&dev);
        let d_tr = dev.htod_sync_copy(&g.trellis).unwrap();
        let mut d_out = dev.htod_sync_copy(&vec![0xFFFFu16; g.inner.len()]).unwrap();
        unsafe {
            f.clone().launch_on_stream(&stream,
                LaunchConfig { grid_dim: ((g.kb * g.nb) as u32, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 },
                (&d_tr, &d_rc, &mut d_out, g.nb as i32, g.k2 as i32)).unwrap();
        }
        dev.synchronize().unwrap();
        let got = dev.dtoh_sync_copy(&d_out).unwrap();
        let nbad = got.iter().zip(g.inner.iter()).filter(|(a, b)| a != b).count();
        assert!(nbad == 0, "{}: exl3_decode_dump k2={} differs from the wheel golden: {nbad}/{} values, first {:?}",
                g.base, g.k2, got.len(),
                got.iter().zip(g.inner.iter()).enumerate().filter_map(|(i, (a, b))| if a != b { Some((i, *a, *b)) } else { None }).take(8).collect::<Vec<_>>());
    }

    fn grouped_and_wide_vs_wheel(stem: &str) {
        let Some(g) = load(stem) else { return; };
        let (k, n) = (g.kb * 16, g.nb * 16);
        let code = crate::exl3_forward::exl3_kernel_bits(g.k2);
        assert!(crate::exl3_forward::exl3_is_frac_bits(code), "fixture k2 {} is not fractional", g.k2);
        let dev = dev0().unwrap();
        load_module(&dev).unwrap();
        let stream = fork_blocking_stream(&dev);
        let f_ref = dev.get_func(MODULE, "exl3_hmma_gemm").unwrap();
        let f_grp = dev.get_func(MODULE, "xq_gemm_grouped_xh").unwrap();
        let f_wt = dev.get_func(MODULE, "exl3_hmma_gemm_wide_tiles").unwrap();
        let d_tr = dev.htod_sync_copy(&g.trellis).unwrap();
        let d_offs = dev.htod_sync_copy(&[0u64]).unwrap();
        let d_esel = dev.htod_sync_copy(&[1i32]).unwrap();
        let inner: Vec<f64> = g.inner.iter().map(|b| half::f16::from_bits(*b).to_f32() as f64).collect();
        let mut a1_widths = Vec::new();
        let mut barrier_widths = Vec::new();
        for m in [1usize, 2, 4, 8, 9, 10, 12, 16] {
            let (xh16, xhf) = make_xh(&g, m, m);
            let d_xh = dev.htod_sync_copy(&xh16).unwrap();
            let mut y_ref = dev.htod_sync_copy(&vec![half::f16::from_bits(0xFFFF); m * n]).unwrap();
            let mut y_grp = dev.htod_sync_copy(&vec![half::f16::from_bits(0xFFFF); m * n]).unwrap();
            let bytes = m * (k + 8) * 2;
            let smem = if bytes <= 48 * 1024 { a1_widths.push(m); bytes as u32 } else { barrier_widths.push(m); 0 };
            let cfg = |sm: u32| LaunchConfig { grid_dim: ((n / 128) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: sm };
            unsafe {
                f_ref.clone().launch_on_stream(&stream, cfg(0), (&d_tr, &d_xh, &mut y_ref, m as i32, k as i32, n as i32, g.k2 as i32)).unwrap();
                f_grp.clone().launch_on_stream(&stream, cfg(smem), (&d_tr, &d_offs, &d_xh, &mut y_grp, m as i32, k as i32, n as i32, code as i32, &d_esel)).unwrap();
            }
            dev.synchronize().unwrap();
            let a = dev.dtoh_sync_copy(&y_ref).unwrap();
            let b = dev.dtoh_sync_copy(&y_grp).unwrap();
            let poison = b.iter().filter(|v| v.to_bits() == 0xFFFF).count();
            assert_eq!(poison, 0, "{}: m={m} grouped output left {poison} poisoned elements (unwritten tile = the silent no-op class)", g.base);
            let mism = a.iter().zip(b.iter()).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            assert_eq!(mism, 0, "{}: m={m} xq_gemm_grouped_xh (bits code {code}) != exl3_hmma_gemm (k2 {}) on {mism} elements", g.base, g.k2);
            // against the wheel's weights: y[r][c] = sum_k xh[r][k] * inner[k][c] in f64
            let (mut num, mut den) = (0f64, 0f64);
            for r in 0..m {
                for c in 0..n {
                    let mut acc = 0f64;
                    for ki in 0..k { acc += xhf[r * k + ki] as f64 * inner[ki * n + c]; }
                    let got = b[r * n + c].to_f32() as f64;
                    num += (got - acc) * (got - acc);
                    den += acc * acc;
                }
            }
            let rel = (num / den).sqrt();
            assert!(rel < 2e-3, "{}: m={m} grouped GEMM vs f64 reference on the wheel's weights: rel-L2 {rel:.3e} (fp16 output rounding is ~1e-4-class; a wrong window measures O(1))", g.base);
            if m == 4 { eprintln!("{}: m=4 grouped vs f64 wheel reference rel-L2 {rel:.3e}", g.base); }
        }
        assert!(!a1_widths.is_empty() && (k > 640 || barrier_widths.is_empty()), "widths: a1 {a1_widths:?} barrier {barrier_widths:?}");
        eprintln!("{}: grouped == m16 reference at widths a1 {a1_widths:?}, barrier {barrier_widths:?}", g.base);

        // 3. the prefill wide-tiles body: one expert, ce = 20 rows, every MT class, == the m16 reference
        // (an m16 kernel: launched once per 16-row block — rows 0..16 at M = 16, rows 16..20 at M = 4)
        let ce = 20usize;
        let (xh16, _) = make_xh(&g, ce, 77);
        let d_xh = dev.htod_sync_copy(&xh16).unwrap();
        let mut y_ref = dev.htod_sync_copy(&vec![half::f16::from_bits(0xFFFF); ce * n]).unwrap();
        for (r0, mr) in [(0usize, 16usize), (16, 4)] {
            let xp = *d_xh.device_ptr() as u64 + (r0 * k * 2) as u64;
            let yp = *y_ref.device_ptr() as u64 + (r0 * n * 2) as u64;
            unsafe {
                f_ref.clone().launch_on_stream(&stream,
                    LaunchConfig { grid_dim: ((n / 128) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                    (&d_tr, xp, yp, mr as i32, k as i32, n as i32, g.k2 as i32)).unwrap();
            }
        }
        dev.synchronize().unwrap();
        let a = dev.dtoh_sync_copy(&y_ref).unwrap();
        let d_offs_row = dev.htod_sync_copy(&[0i32, ce as i32]).unwrap();
        let d_cnt = dev.htod_sync_copy(&[ce as i32]).unwrap();
        for mt in [1usize, 2, 4, 8] {
            let nsup = (ce + 16 * mt - 1) / (16 * mt);
            let tiles: Vec<i32> = (0..nsup).flat_map(|s| [0i32, s as i32]).collect();
            let d_tiles = dev.htod_sync_copy(&tiles).unwrap();
            let d_nt = dev.htod_sync_copy(&[nsup as i32]).unwrap();
            let mut y_wt = dev.htod_sync_copy(&vec![half::f16::from_bits(0xFFFF); ce * n]).unwrap();
            unsafe {
                f_wt.clone().launch_on_stream(&stream,
                    LaunchConfig { grid_dim: ((n / 128) as u32, nsup as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                    (&d_tr, &d_offs, &d_xh, &mut y_wt, &d_tiles, &d_nt, &d_offs_row, &d_cnt, k as i32, n as i32, code as i32, mt as i32)).unwrap();
            }
            dev.synchronize().unwrap();
            let b = dev.dtoh_sync_copy(&y_wt).unwrap();
            let poison = b.iter().filter(|v| v.to_bits() == 0xFFFF).count();
            assert_eq!(poison, 0, "{}: wide tiles mt={mt} left {poison} poisoned elements", g.base);
            let mism = a.iter().zip(b.iter()).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            assert_eq!(mism, 0, "{}: exl3_hmma_gemm_wide_tiles (bits code {code}, mt {mt}) != m16 reference on {mism} of {} elements", g.base, ce * n);
        }
        eprintln!("{}: wide tiles (bits code {code}) == m16 reference at mt 1/2/4/8, {ce} rows", g.base);
    }

    #[test]
    fn k35_318_up_kernel_dump_vs_wheel() { dump_vs_wheel("k35_golden_318_l0e7up_proj"); }
    #[test]
    fn k35_318_down_kernel_dump_vs_wheel() { dump_vs_wheel("k35_golden_318_l0e7down_proj"); }
    #[test]
    fn k35_318_up_grouped_and_wide_vs_wheel() { grouped_and_wide_vs_wheel("k35_golden_318_l0e7up_proj"); }
    #[test]
    fn k35_318_down_grouped_and_wide_vs_wheel() { grouped_and_wide_vs_wheel("k35_golden_318_l0e7down_proj"); }

    // K = 2.5 (k2 5; the 2.68 pack's layer-4 expert 7): the same three receipts
    #[test]
    fn k25_268_up_kernel_dump_vs_wheel() { dump_vs_wheel("k35_golden_268_l4e7up_proj"); }
    #[test]
    fn k25_268_down_kernel_dump_vs_wheel() { dump_vs_wheel("k35_golden_268_l4e7down_proj"); }
    #[test]
    fn k25_268_up_grouped_and_wide_vs_wheel() { grouped_and_wide_vs_wheel("k35_golden_268_l4e7up_proj"); }
    #[test]
    fn k25_268_down_grouped_and_wide_vs_wheel() { grouped_and_wide_vs_wheel("k35_golden_268_l4e7down_proj"); }
