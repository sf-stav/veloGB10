//! `impl GpuModel`: GEMM probes (`probe_gemm*`, `probe_splitk`, `probe_moe_gemm`, `probe_dispatch`, `probe_binv*`, sweeps).
//! `probe_binv` is the batch-invariance gate (run after touching any GEMM). Map: `mod.rs`.

use super::*;

impl GpuModel {
    /// GEMM batch-invariance probe: measures whether `gemm(W^T, X)` at N=1 and N=2 produce
    /// bit-identical results for the first column. cuBLAS picks different algos (tiling/Split-K) for
    /// different N → different fp32 accumulation order → nonzero diff. This isolates the GEMM (no
    /// model) so we can test fixes (cuBLASLt algo pinning) against the exact shapes that matter.
    /// Uses random bf16 W [m,k] (OP_T) and X [k,2]; reports max|out_N1[i] - out_N2[i,0]|.
    pub fn probe_gemm(&self, m: usize, k: usize) {
        use cudarc::cublas::{sys::cublasOperation_t as OP, Gemm, GemmConfig};
        let w_host: Vec<half::bf16> = (0..m * k)
            .map(|_| half::bf16::from_f32((rand::random::<f32>() * 2.0) - 1.0)).collect();
        let x_host: Vec<half::bf16> = (0..k * 2)
            .map(|_| half::bf16::from_f32((rand::random::<f32>() * 2.0) - 1.0)).collect();
        let w = self.dev.htod_sync_copy(&w_host).unwrap();
        let x = self.dev.htod_sync_copy(&x_host).unwrap();
        self.dev.synchronize().unwrap();

        // N=1: out1 = W^T @ X[:,0]  -> [m]
        let mut out1 = unsafe { self.dev.alloc::<half::bf16>(m) }.unwrap();
        let cfg1 = GemmConfig::<half::bf16> {
            transa: OP::CUBLAS_OP_T, transb: OP::CUBLAS_OP_N,
            m: m as i32, n: 1, k: k as i32,
            alpha: half::bf16::from_f32(1.0), lda: k as i32,
            ldb: k as i32, beta: half::bf16::from_f32(0.0), ldc: m as i32,
        };
        unsafe { self.blas.gemm(cfg1, &w, &x, &mut out1).expect("gemm n1"); }

        // N=2: out2 = W^T @ X[:,0:2] -> [m,2] column-major
        let mut out2 = unsafe { self.dev.alloc::<half::bf16>(m * 2) }.unwrap();
        let cfg2 = GemmConfig::<half::bf16> {
            transa: OP::CUBLAS_OP_T, transb: OP::CUBLAS_OP_N,
            m: m as i32, n: 2, k: k as i32,
            alpha: half::bf16::from_f32(1.0), lda: k as i32,
            ldb: k as i32, beta: half::bf16::from_f32(0.0), ldc: m as i32,
        };
        unsafe { self.blas.gemm(cfg2, &w, &x, &mut out2).expect("gemm n2"); }
        self.sync_stream();

        let o1: Vec<half::bf16> = self.dev.dtoh_sync_copy(&out1).unwrap();
        let o2: Vec<half::bf16> = self.dev.dtoh_sync_copy(&out2).unwrap();
        let mut mx = 0.0f32;
        let mut nz = 0usize;
        for i in 0..m {
            let d = (half::bf16::to_f32(o1[i]) - half::bf16::to_f32(o2[i])).abs();
            if d > 0.0 { nz += 1; }
            if d > mx { mx = d; }
        }
        println!("GEMM(cuBLAS) M={} K={}: N=1 vs N=2 col-0  max-diff={:.6}  ({}/{} elements differ)",
                 m, k, mx, nz, m);
    }

    /// True if this model has a loaded MTP (multi-token prediction) head.
    pub fn mtp_present(&self) -> bool { self.mtp.is_some() }

    /// cuBLASLt batch-invariance probe: tests whether cuBLASLt (default heuristic, AND with Split-K
    /// forced off) gives a batch-invariant GEMM (N=1 == N=2). If either zeroes the diff, it's the
    /// fix for the 9B/27B MTP cuBLAS divergence. Runs on the compute stream (no cross-stream race).
    pub fn probe_gemm_lt(&self, m: usize, k: usize) {
        use cudarc::cublaslt::{result as lt, sys};
        let handle = lt::create_handle().expect("cublasLtCreate");
        let ws_size: usize = 32 * 1024 * 1024;
        let workspace = unsafe { self.dev.alloc::<u8>(ws_size) }.expect("ws alloc");
        let w_host: Vec<half::bf16> = (0..m * k)
            .map(|_| half::bf16::from_f32((rand::random::<f32>() * 2.0) - 1.0)).collect();
        let x_host: Vec<half::bf16> = (0..k * 2)
            .map(|_| half::bf16::from_f32((rand::random::<f32>() * 2.0) - 1.0)).collect();
        let w = self.dev.htod_sync_copy(&w_host).unwrap();
        let x = self.dev.htod_sync_copy(&x_host).unwrap();
        let mut out1 = unsafe { self.dev.alloc::<half::bf16>(m) }.unwrap();
        let mut out2 = unsafe { self.dev.alloc::<half::bf16>(m * 2) }.unwrap();
        self.dev.synchronize().unwrap();

        let bf = sys::cudaDataType_t::CUDA_R_16BF;
        let desc = lt::create_matmul_desc(
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F, sys::cudaDataType_t::CUDA_R_32F).unwrap();
        let ta: u32 = 1; let tb: u32 = 0;
        unsafe {
            lt::set_matmul_desc_attribute(desc,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                &ta as *const _ as *const _, 4).unwrap();
            lt::set_matmul_desc_attribute(desc,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                &tb as *const _ as *const _, 4).unwrap();
        }
        let pref = lt::create_matmul_pref().unwrap();
        unsafe {
            lt::set_matmul_pref_attribute(pref,
                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &ws_size as *const _ as *const _, 8).unwrap();
        }
        let alpha: f32 = 1.0; let beta: f32 = 0.0;
        let ws_ptr = *workspace.device_ptr() as *mut core::ffi::c_void;
        let stream = self.stream.stream as sys::cudaStream_t;

        let mut run = |force_splitk: bool| -> Option<f32> {
            for &n in &[1usize, 2] {
                let a_layout = lt::create_matrix_layout(bf, k as u64, m as u64, k as i64).ok()?;
                let b_layout = lt::create_matrix_layout(bf, k as u64, n as u64, k as i64).ok()?;
                let c_layout = lt::create_matrix_layout(bf, m as u64, n as u64, m as i64).ok()?;
                let mut heur = lt::get_matmul_algo_heuristic(handle, desc,
                    a_layout, b_layout, c_layout, c_layout, pref).ok()?;
                if force_splitk {
                    let one: u32 = 1;
                    unsafe {
                        sys::cublasLtMatmulAlgoConfigSetAttribute(&mut heur.algo,
                            sys::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                            &one as *const _ as *const _, 4);
                    }
                }
                let out = if n == 1 { &mut out1 } else { &mut out2 };
                let r = unsafe {
                    lt::matmul(handle, desc,
                        &alpha as *const _ as *const core::ffi::c_void,
                        &beta as *const _ as *const core::ffi::c_void,
                        *w.device_ptr() as *const core::ffi::c_void, a_layout,
                        *x.device_ptr() as *const core::ffi::c_void, b_layout,
                        *out.device_ptr() as *const core::ffi::c_void, c_layout,
                        *out.device_ptr() as *mut core::ffi::c_void, c_layout,
                        &heur.algo, ws_ptr, ws_size, stream)
                };
                if r.is_err() { return None; }
            }
            self.sync_stream();
            let o1: Vec<half::bf16> = self.dev.dtoh_sync_copy(&out1).unwrap();
            let o2: Vec<half::bf16> = self.dev.dtoh_sync_copy(&out2).unwrap();
            let mut mx = 0.0f32;
            for i in 0..m {
                let d = (half::bf16::to_f32(o1[i]) - half::bf16::to_f32(o2[i])).abs();
                if d > mx { mx = d; }
            }
            Some(mx)
        };

        match run(false) {
            Some(mx) => println!("GEMM(cuBLASLt default) M={} K={}: N=1 vs N=2 diff={:.6}{}", m, k, mx, if mx==0.0{" << BATCH-INVARIANT"}else{""}),
            None => println!("GEMM(cuBLASLt default) M={} K={}: FAILED", m, k),
        }
        match run(true) {
            Some(mx) => println!("GEMM(cuBLASLt splitK=OFF) M={} K={}: N=1 vs N=2 diff={:.6}{}", m, k, mx, if mx==0.0{" << BATCH-INVARIANT!"}else{""}),
            None => println!("GEMM(cuBLASLt splitK=OFF) M={} K={}: FAILED (algo invalid w/ splitk=1)", m, k),
        }
    }

    /// Custom batch-invariant GEMM probe: runs the `gemm_binv_b` kernel at N=1 and N=2, checks the
    /// column-0 is BIT-IDENTICAL (batch-invariant), and compares vs cuBLAS N=1 (correctness — should
    /// be within bf16 rounding). If the custom kernel is batch-invariant, it's the fix for 9B/27B.
    pub fn probe_gemm_binv(&self, m: usize, k: usize) {
        use cudarc::cublas::{sys::cublasOperation_t as OP, Gemm, GemmConfig};
        let w_host: Vec<half::bf16> = (0..m * k)
            .map(|_| half::bf16::from_f32((rand::random::<f32>() * 2.0) - 1.0)).collect();
        let x_host: Vec<half::bf16> = (0..k * 2)
            .map(|_| half::bf16::from_f32((rand::random::<f32>() * 2.0) - 1.0)).collect();
        let w = self.dev.htod_sync_copy(&w_host).unwrap();
        let x = self.dev.htod_sync_copy(&x_host).unwrap();
        self.dev.synchronize().unwrap();
        let t = 256u32;
        let smem = (16 * 256 * 4) as u32; // Nmax(16) * T(256) * f32

        // custom batch-invariant kernel, N=1 and N=2
        let binv1 = unsafe { self.dev.alloc::<half::bf16>(m) }.unwrap();
        let binv2 = unsafe { self.dev.alloc::<half::bf16>(m * 2) }.unwrap();
        blaunch!(self, "gemm_binv_b", (m as u32,1,1), (t,1,1), smem,
            (d(&binv1), d(&w), d(&x), m as i32, k as i32, 1i32));
        blaunch!(self, "gemm_binv_b", (m as u32,1,1), (t,1,1), smem,
            (d(&binv2), d(&w), d(&x), m as i32, k as i32, 2i32));
        // cuBLAS reference, N=1
        let mut cublas1 = unsafe { self.dev.alloc::<half::bf16>(m) }.unwrap();
        let cfg = GemmConfig::<half::bf16> {
            transa: OP::CUBLAS_OP_T, transb: OP::CUBLAS_OP_N,
            m: m as i32, n: 1, k: k as i32,
            alpha: half::bf16::from_f32(1.0), lda: k as i32, ldb: k as i32,
            beta: half::bf16::from_f32(0.0), ldc: m as i32,
        };
        unsafe { self.blas.gemm(cfg, &w, &x, &mut cublas1).expect("cublas ref"); }
        self.sync_stream();

        let b1: Vec<half::bf16> = self.dev.dtoh_sync_copy(&binv1).unwrap();
        let b2: Vec<half::bf16> = self.dev.dtoh_sync_copy(&binv2).unwrap();
        let c1: Vec<half::bf16> = self.dev.dtoh_sync_copy(&cublas1).unwrap();
        let mut inv_diff = 0.0f32;   // binv N=1 vs N=2 col0 (must be 0)
        let mut ref_diff = 0.0f32;   // binv N=1 vs cuBLAS N=1 (correctness, ~bf16 rounding)
        for i in 0..m {
            let di = (half::bf16::to_f32(b1[i]) - half::bf16::to_f32(b2[i])).abs();
            let dr = (half::bf16::to_f32(b1[i]) - half::bf16::to_f32(c1[i])).abs();
            if di > inv_diff { inv_diff = di; }
            if dr > ref_diff { ref_diff = dr; }
        }
        println!("GEMM(gemm_binv custom) M={} K={}: N=1 vs N=2 col-0 diff={:.6}{}   | vs cuBLAS N=1 diff={:.6}",
                 m, k, inv_diff, if inv_diff == 0.0 { " << BATCH-INVARIANT!" } else { "" }, ref_diff);
    }

    /// THE GATE: bitwise batch-invariance of the serving GEMM, on real weights, across the whole
    /// verify width. Column 0 of an N-wide verify must be BIT-IDENTICAL to a single-token decode —
    /// that is the entire reason greedy MTP is lossless rather than merely "usually the same".
    ///
    /// This drives `gemm_act`, the real dispatcher, so it tests whatever the engine would actually
    /// run. It also compares against the PREFILL path (dequantize-to-bf16 + cuBLAS), which reaches
    /// the same weights through the inverse permutation and a completely different kernel — so a
    /// mis-mapped mma fragment cannot hide: the permutation and its inverse would have to be wrong
    /// in exactly compensating ways, and the inverse is separately unit-tested against the forward.
    /// TP=2 half-width GEMV probe. Runs the REAL production dispatch for the decode GEMV
    /// (`launch_gemm_fp4`: `gemm_mma_fp4_b` persistent-grid, or the split-K partial+reduce where
    /// `gemm_fp4_nsplit` fires — printed as S=) at N=1 on
    /// synthetic (correctly-SIZED) NVFP4 buffers for a linear of shape (outn=M, inn=K), times it, and
    /// reports achieved weight-bandwidth vs the ~245 GB/s sustained roofline. Contents are irrelevant
    /// (timing only): the kernel does identical memory traffic regardless of the bytes it reads.
    ///
    /// Why: TP=2 shards every 27B linear to HALF — column-parallel (gate/up/qkv → half M) or
    /// row-parallel (down/o → half K, i.e. half the reduction depth). The whole ~1.85x projection
    /// hinges on the per-node half-width GEMV still hitting ~80% of roofline. If the smaller shape
    /// drops efficiency (fewer blocks to fill 48 SMs at half-M; a shorter K reduction hiding less
    /// latency at half-K), the win erodes toward ~1.5x. This measures it with ZERO comm code.
    pub fn probe_tp_gemv(&self, m: usize, k: usize, label: &str) {
        assert!(m % 16 == 0 && k % 32 == 0, "shape ({m},{k}) needs M%16==0, K%32==0");
        let qweight = self.dev.alloc_zeros::<u8>(m * k / 2).unwrap();
        let scales  = self.dev.alloc_zeros::<u8>(m * k / 16).unwrap();
        let gs      = self.dev.alloc_zeros::<f32>(m / 16).unwrap();
        let x       = self.dev.alloc_zeros::<half::bf16>(k + 64).unwrap();
        let out     = unsafe { self.dev.alloc::<half::bf16>(m) }.unwrap();
        self.dev.synchronize().unwrap();
        let bytes = (m * k / 2 + m * k / 16) as f64;   // fp4 weights + E4M3 per-tile scales
        // The PRODUCTION dispatch (persistent grid; split-K partial+reduce where gemm_fp4_nsplit
        // says so) — a probe of anything else would measure a kernel the model never runs.
        let ns = gemm_fp4_nsplit(m, k);
        let run = || {
            self.launch_gemm_fp4(d(&out), *qweight.device_ptr() as u64, *scales.device_ptr() as u64,
                                 d(&gs), d(&x), m, k, 1, 0);   // Cf = null: bf16 store
        };
        for _ in 0..5 { run(); }        // warm
        self.sync_stream();
        let iters = 400u32;
        let t0 = std::time::Instant::now();
        for _ in 0..iters { run(); }
        self.sync_stream();
        let s = t0.elapsed().as_secs_f64() / iters as f64;
        let gbs = bytes / s / 1e9;
        println!("  {:<24} M={:>6} K={:>6} S={}   {:>7.1} GB/s   {:>5.1}% roof   ({:>6.1} us, {:>6.2} MB)",
                 label, m, k, ns, gbs, 100.0 * gbs / 245.0, s * 1e6, bytes / 1e6);
    }

    /// Split-K A/B probe: times the NVFP4 serving GEMM on synthetic (correctly-SIZED) buffers at
    /// N=1 for a list of (M, K, label) shapes, sweeping the split count S = {1 (unsplit), auto,
    /// 2, 4, 6, 8} INTERLEAVED in-process over ROUNDS rounds (best-of-R). S=auto prints the exact
    /// value `gemm_fp4_nsplit` would pick in production (subject to --splitk-gemm). Reports
    /// GB/s of packed-weight bytes vs the ~245 GB/s roofline. This is the honest A/B: same process,
    /// same buffers, same clocks, split geometry the only variable.
    pub fn probe_splitk(&self, shapes: &[(usize, usize, String)], rounds: u32) {
        for &(m, k, ref label) in shapes {
            assert!(m % 16 == 0 && k % 32 == 0, "shape ({m},{k}) needs M%16==0, K%32==0");
            let qweight = self.dev.alloc_zeros::<u8>(m * k / 2).unwrap();
            let scales  = self.dev.alloc_zeros::<u8>(m * k / 16).unwrap();
            let gs      = self.dev.alloc_zeros::<f32>(m / 16).unwrap();
            let x       = self.dev.alloc_zeros::<half::bf16>(k + 64).unwrap();
            let out     = unsafe { self.dev.alloc::<half::bf16>(m) }.unwrap();
            self.dev.synchronize().unwrap();
            let bytes = (m * k / 2 + m * k / 16) as f64;   // fp4 weights + E4M3 per-tile scales
            // shape_s: the split the AUTO rule would pick at this shape if --splitk-gemm were on
            // (shape-only, never N); auto: what production dispatches right now (1 unless on/forced).
            let shape_s = if m / 16 <= SPLITK_MAX_TILES && k >= 8192 { (k / 4096).clamp(1, SPLITK_MAX_S) } else { 1 };
            let auto = gemm_fp4_nsplit(m, k);
            let mut svals: Vec<usize> = vec![1, shape_s];
            for s in [2usize, 3, 4, 6, 8] { if !svals.contains(&s) { svals.push(s); } }
            // (S, best GB/s, best us)
            let mut best: Vec<(usize, f64, f64)> = svals.iter().map(|s| (*s, 0.0f64, 0.0f64)).collect();
            let mut round_out = Vec::new();
            for _ in 0..rounds {
                for (si, s) in svals.iter().enumerate() {
                    let s = *s;
                    for _ in 0..5 { self.launch_gemm_fp4_s(d(&out), *qweight.device_ptr() as u64,
                        *scales.device_ptr() as u64, d(&gs), d(&x), m, k, 1, 0, s, false); }   // warm
                    self.sync_stream();
                    let iters = 300u32;
                    let t0 = std::time::Instant::now();
                    for _ in 0..iters { self.launch_gemm_fp4_s(d(&out), *qweight.device_ptr() as u64,
                        *scales.device_ptr() as u64, d(&gs), d(&x), m, k, 1, 0, s, false); }
                    self.sync_stream();
                    let us = t0.elapsed().as_secs_f64() / iters as f64 * 1e6;
                    let gbs = bytes / us / 1e3;
                    round_out.push((s, gbs, us));
                    if gbs > best[si].1 { best[si] = (s, gbs, us); }
                }
            }
            // Compact per-round view: one line per round.
            for (s, g, u) in &round_out { print!("  S={s}:{g:6.1}GB/s  "); }
            println!();
            println!("  {:<22} M={:>6} K={:>6}  prod-S={} shape-S={}   {}",
                     label, m, k, auto, shape_s,
                     best.iter().map(|(s, g, u)| format!("S={s}: {g:6.1} GB/s ({u:6.1} us)"))
                         .collect::<Vec<_>>().join(" | "));
        }
    }

    /// E11 — MoE expert-GEMM A/B probe (`--probe-moe-gemm`). SYNTHETIC weights at the caller's
    /// shapes (default Hy3's slot-fold decode shapes), NO model load: allocates the packed NVFP4
    /// layout the fold kernels read, picks 8 distinct routed experts + the shared slot (sdesc =
    /// the same buffers — desc[3] kk=K/sk_off=0, desc[8] m_local=M, desc[7] rank_off=0), and times
    /// each E11 fold variant INTERLEAVED, best-of-`rounds`. Reports per-variant µs + GB/s of packed
    /// weight bytes (codes + scales). Plus the PDL overlap mechanism test: a synthetic predecessor
    /// (e11_fill_x_b over a 2 MB scratch, ~the silu's duration) stands in for the silu, and the
    /// GEMM runs with/without the programmatic-stream-secondary launch — the overlap shows as the
    /// pdl pair beating (fill + GEMM) end to end.
    pub fn probe_moe_gemm(&self, shapes: &[(usize, usize, String)], rounds: u32) {
        for &(m, k, ref label) in shapes {
            assert!(m % 32 == 0 && k % 32 == 0, "shape ({m},{k}) needs M%32==0, K%32==0");
            let nslots = 9usize;                 // k=8 routed + 1 shared (Hy3's fold decode)
            let espan = 9usize;                  // ids touch experts 0..7; the shared slot re-reads e0
            let kx = ((nslots as i64) << 32);    // Kslots<<32 | x_by_slot=0 (gate/up mapping)
            let qw = self.dev.alloc_zeros::<u8>(espan * m * k / 2).unwrap();
            let sc  = self.dev.alloc_zeros::<u8>(espan * m * k / 16).unwrap();
            let gs  = self.dev.alloc_zeros::<f32>(espan * m / 16).unwrap();
            let x   = self.dev.alloc_zeros::<half::bf16>(k + 64).unwrap();
            let out = unsafe { self.dev.alloc::<half::bf16>(nslots * m) }.unwrap();
            let ids: Vec<i32> = (0..nslots).map(|i| (i % 8) as i32).collect();
            let ids_d = self.dev.htod_sync_copy(&ids).unwrap();
            let sdesc_v: Vec<i64> = vec![
                *qw.device_ptr() as i64, *sc.device_ptr() as i64, *gs.device_ptr() as i64,
                (k as i64) << 32,      // desc[3]: kk << 32 | sk_off(0)
                0, 0, 0,               // desc[4..6] (down region — unused here)
                0,                     // desc[7]: rank_off
                m as i64,              // desc[8]: m_local = M
            ];
            let sdesc = self.dev.htod_sync_copy(&sdesc_v).unwrap();
            self.dev.synchronize().unwrap();
            let bytes = (nslots as f64) * (m as f64) * (k as f64) * 0.5625;  // codes + e4m3 scales
            let qwp = *qw.device_ptr() as u64; let scp = *sc.device_ptr() as u64;
            let gsp = *gs.device_ptr() as u64; let xp = *x.device_ptr() as u64;
            let op = *out.device_ptr() as u64; let ip = *ids_d.device_ptr() as u64;
            let sdp = *sdesc.device_ptr() as u64;
            let (mo, kk_i, eb, es) = (m as i32, k as i32, 0i32, espan as i32);

            // (variant label, grid, kernel name, pdl flag)
            let mut variants: Vec<(&str, (u32, u32, u32), &str, bool)> = vec![
                ("plain", ((m / 16) as u32, nslots as u32, 1), "gemm_moe_mma_fp4_fold", false),
                ("u4",    ((m / 16) as u32, nslots as u32, 1), "gemm_moe_mma_fp4_fold_u4", false),
                ("x2",    ((m / 32) as u32, nslots as u32, 1), "gemm_moe_mma_fp4_fold_x2", false),
                ("rast",  (nslots as u32, (m / 16) as u32, 1), "gemm_moe_mma_fp4_fold_rast", false),
                ("lb5",   ((m / 16) as u32, nslots as u32, 1), "gemm_moe_mma_fp4_fold_lb5", false),
                ("lb4",   ((m / 16) as u32, nslots as u32, 1), "gemm_moe_mma_fp4_fold_lb4", false),
                ("pdl",   ((m / 16) as u32, nslots as u32, 1), "gemm_moe_mma_fp4_fold_pdl", true),
            ];
            // Respect --moe-variant when set: probe only that variant vs plain (iteration mode).
            if let Ok(v) = crate::opts::var(crate::opt!("moe-variant")) {
                variants.retain(|(l, _, _, _)| *l == "plain" || *l == v);
            }
            let mut best: Vec<f64> = variants.iter().map(|_| 0.0f64).collect();
            // Bit-identity first: every variant's output must equal the plain fold's byte-for-byte
            // (the kernel-level oracle check; the full e2e gates are binv + LOSSLESS + acceptance).
            {
                let mut ref_out: Option<Vec<half::bf16>> = None;
                for (label, grid, name, pdl) in &variants {
                    self.launch_moe_probe_variant(*grid, name, *pdl, op, qwp, scp, gsp, xp, ip,
                                                  mo, kk_i, kx, eb, es, sdp);
                    self.sync_stream();
                    let c: Vec<half::bf16> = self.dev.dtoh_sync_copy(&out).unwrap();
                    match &ref_out {
                        None => { ref_out = Some(c); }
                        Some(r) => {
                            let eq = r.len() == c.len() && r.iter().zip(c.iter()).all(|(a, b)| a == b);
                            println!("  bit-identity {}: {}", label, if eq { "OK" } else { "MISMATCH" });
                        }
                    }
                }
            }
            for _ in 0..rounds {
                for (vi, (_, grid, name, pdl)) in variants.iter().enumerate() {
                    for _ in 0..5 {
                        self.launch_moe_probe_variant(*grid, name, *pdl, op, qwp, scp, gsp, xp, ip,
                                                      mo, kk_i, kx, eb, es, sdp);
                    }
                    self.sync_stream();
                    let iters = 300u32;
                    let t0 = std::time::Instant::now();
                    for _ in 0..iters {
                        self.launch_moe_probe_variant(*grid, name, *pdl, op, qwp, scp, gsp, xp, ip,
                                                      mo, kk_i, kx, eb, es, sdp);
                    }
                    self.sync_stream();
                    let us = t0.elapsed().as_secs_f64() / iters as f64 * 1e6;
                    let gbs = bytes / us / 1e3;
                    if gbs > best[vi] { best[vi] = gbs; }
                }
            }
            println!("  {:<6} M={:>5} K={:>5}  {}",
                     label, m, k,
                     variants.iter().zip(best.iter())
                         .map(|((l, _, _, _), g)| format!("{l}: {g:6.1} GB/s"))
                         .collect::<Vec<_>>().join(" | "));
            let _ = &bytes;

            // PDL overlap mechanism: (fill 2MB + GEMM) plain vs pdl, interleaved best-of-rounds.
            let fill_n = 2 * 1024 * 1024 / 4;   // uint32s → 2 MB scratch ≈ the silu's duration
            let fill = self.dev.alloc_zeros::<u32>(fill_n).unwrap();
            let fp = *fill.device_ptr() as u64;
            let mut bp = 0.0f64; let mut bd = 0.0f64;
            for _ in 0..rounds {
                for pdl in [false, true] {
                    for _ in 0..5 {
                        blaunch!(self, "e11_fill_x_b", grid(fill_n), (256,1,1), 0, (fp, fill_n as i32));
                        self.launch_moe_probe_variant(((m/16) as u32, nslots as u32, 1),
                                                      if pdl { "gemm_moe_mma_fp4_fold_pdl" } else { "gemm_moe_mma_fp4_fold" },
                                                      pdl, op, qwp, scp, gsp, xp, ip, mo, kk_i, kx, eb, es, sdp);
                    }
                    self.sync_stream();
                    let iters = 200u32;
                    let t0 = std::time::Instant::now();
                    for _ in 0..iters {
                        blaunch!(self, "e11_fill_x_b", grid(fill_n), (256,1,1), 0, (fp, fill_n as i32));
                        self.launch_moe_probe_variant(((m/16) as u32, nslots as u32, 1),
                                                      if pdl { "gemm_moe_mma_fp4_fold_pdl" } else { "gemm_moe_mma_fp4_fold" },
                                                      pdl, op, qwp, scp, gsp, xp, ip, mo, kk_i, kx, eb, es, sdp);
                    }
                    self.sync_stream();
                    let us = t0.elapsed().as_secs_f64() / iters as f64 * 1e6;
                    if pdl { if us > bd { bd = us; } } else { if us > bp { bp = us; } }
                }
            }
            println!("  PDL overlap (fill+GEMM): plain {bp:6.1} us | pdl {bd:6.1} us  (saved {:.1} us)",
                     bp - bd);
        }
    }

    /// E11 probe launch helper: the fold-variant 12-arg tuple, or the pdl kernel's 13-arg tuple
    /// (the extra pdl flag), plain stream or programmatic-stream-secondary.
    #[allow(clippy::too_many_arguments)]
    fn launch_moe_probe_variant(&self, grid: (u32, u32, u32), name: &str, pdl: bool,
                                c: u64, qw: u64, sc: u64, gs: u64, x: u64, ids: u64,
                                m: i32, k: i32, kx: i64, eb: i32, es: i32, sd: u64) {
        if pdl {
            if let Some(f) = e11_raw_kernels().and_then(|ks| ks.by_name.get(name).copied()) {
                let pdl_one = 1i32;
                let mut params = [
                    c.as_kernel_param(), qw.as_kernel_param(), sc.as_kernel_param(),
                    gs.as_kernel_param(), x.as_kernel_param(), ids.as_kernel_param(),
                    m.as_kernel_param(), k.as_kernel_param(), kx.as_kernel_param(),
                    eb.as_kernel_param(), es.as_kernel_param(), sd.as_kernel_param(),
                    pdl_one.as_kernel_param(),
                ];
                unsafe { self.launch_pss(f, grid, (256, 1, 1), 0, &mut params); }
                return;
            }
        }
        if name.ends_with("_pdl") {
            let pdl_zero = 0i32;
            let mut params = [
                c.as_kernel_param(), qw.as_kernel_param(), sc.as_kernel_param(),
                gs.as_kernel_param(), x.as_kernel_param(), ids.as_kernel_param(),
                m.as_kernel_param(), k.as_kernel_param(), kx.as_kernel_param(),
                eb.as_kernel_param(), es.as_kernel_param(), sd.as_kernel_param(),
                pdl_zero.as_kernel_param(),
            ];
            unsafe { self.launch_raw(name, grid, (256, 1, 1), 0, &mut params); }
        } else {
            let (a6, a7, a8, a9, a10, a11) = (m, k, kx, eb, es, sd);
            let mut params = [
                c.as_kernel_param(), qw.as_kernel_param(), sc.as_kernel_param(),
                gs.as_kernel_param(), x.as_kernel_param(), ids.as_kernel_param(),
                a6.as_kernel_param(), a7.as_kernel_param(), a8.as_kernel_param(),
                a9.as_kernel_param(), a10.as_kernel_param(), a11.as_kernel_param(),
            ];
            unsafe { self.launch_raw(name, grid, (256, 1, 1), 0, &mut params); }
        }
    }
}

impl GpuModel {
    /// DISPATCH_ASSERT (`PLAN/DISPATCH_ASSERT_SPEC.md` §3.3) — the two-legged dispatch probe.
    ///
    /// Leg A (the contract): for every N = 1..=MAX_VERIFY on real quantized weights the arm taken
    /// must be the fixed-shape MMA arm. Two independent signals, because "no error" is not a pass
    /// (AGENTS §3): the assert posture panics on any record that violates the invariant, AND this
    /// leg requires the counter to *move* — if a width records nothing, the probe is blind and the
    /// PASS is vacuous.
    /// Leg B (the negative control, the part that makes Leg A mean something): ONE call at
    /// `batch = MAX_VERIFY + 1` must observe the PREFILL arm. If it does not, the instrument
    /// cannot see the cliff it exists to guard (spec §4 P0) and the probe FAILS.
    pub fn probe_dispatch(&self) -> bool {
        use crate::dispatch_assert as da;
        let h = self.cfg.hidden_size;
        let l0 = &self.layers[0];
        let (mlp_ref, im) = match &l0.mlp {
            Ffn::Dense(m) => (m, self.cfg.intermediate_size),
            Ffn::Moe(moe) => (&moe.shared, self.cfg.shared_expert_intermediate_size),
        };
        let mut cases: Vec<(&str, &W, usize, usize)> = vec![
            ("mlp.gate", &mlp_ref.gate, h, im),
            ("mlp.down", &mlp_ref.down, im, h),
        ];
        if let Some(la) = &l0.la {
            cases.push(("gdn.out_proj", &la.out_proj, self.cfg.value_dim(), h));
        }
        if let Some(fa) = self.layers.iter().find(|ly| ly.fa.is_some()).and_then(|ly| ly.fa.as_ref()) {
            let nh = self.cfg.num_heads.max(1);
            let hd = self.cfg.head_dim.max(1);
            cases.push(("attn.o_proj", &fa.o_proj, nh * hd, h));
        }
        if let Some(lh) = &self.lm_head {
            let rows = if self.lm_head_sharded { self.cfg.vocab_size / self.shard_factor() } else { self.cfg.vocab_size };
            cases.push(("lm_head", lh, h, rows));
        }

        da::set_posture(da::POSTURE_ASSERT);
        println!("=== dispatch assert: a <=MAX_VERIFY verify must never take the prefill dequant arm ===");
        println!("  posture={} MAX_VERIFY={}", da::posture_name(), MAX_VERIFY);
        let mut all_ok = true;
        for (name, w, inn, outn) in cases {
            let fmt = match w {
                W::Nvfp4 { .. } => da::FMT_NVFP4,
                W::Fp8 { .. } => da::FMT_FP8,
                W::Fp8Blk { .. } => da::FMT_FP8_BLK,
                W::Bf16(_) => da::FMT_BF16,
                _ => da::FMT_NVFP4,
            };
            let nref = MAX_VERIFY + 1;
            let x_host: Vec<half::bf16> = (0..inn * nref)
                .map(|_| half::bf16::from_f32(rand::random::<f32>() * 2.0 - 1.0)).collect();
            let x = self.dev.htod_sync_copy(&x_host).unwrap();
            self.dev.synchronize().unwrap();

            // ---- Leg A: the contract, N = 1..=MAX_VERIFY
            let mut mma_moves = 0u64;
            let mut prefill_in_a = 0u64;
            for n in 1..=MAX_VERIFY {
                let mut out = unsafe { self.dev.alloc::<half::bf16>(outn * n) }.unwrap();
                let m0 = da::count_of(fmt, da::ARM_MMA);
                let p0 = da::count_of(fmt, da::ARM_PREFILL);
                self.gemm_act(w, &x, &mut out, inn, outn, n);
                self.sync_stream();
                let dm = da::count_of(fmt, da::ARM_MMA) - m0;
                let dp = da::count_of(fmt, da::ARM_PREFILL) - p0;
                mma_moves += dm;
                prefill_in_a += dp;
                drop(out);
            }
            let vacuous = mma_moves != MAX_VERIFY as u64;
            let leg_a_ok = !vacuous && prefill_in_a == 0;

            // ---- Leg B: the negative control (one call past the bound)
            let p0 = da::count_of(fmt, da::ARM_PREFILL);
            let mut r = unsafe { self.dev.alloc::<half::bf16>(outn * nref) }.unwrap();
            self.gemm_act(w, &x, &mut r, inn, outn, nref);
            self.sync_stream();
            let leg_b_prefill = da::count_of(fmt, da::ARM_PREFILL) - p0;
            let leg_b_ok = leg_b_prefill > 0;
            drop(r);

            all_ok &= leg_a_ok && leg_b_ok;
            println!("  {:<14} fmt={:<6} LegA(N=1..{}) mma_moves={} prefill={} {}   LegB(N={}) prefill_seen={} {}",
                     name, da::FMT_NAMES[fmt as usize], MAX_VERIFY, mma_moves, prefill_in_a,
                     if leg_a_ok { "PASS" } else if vacuous { "FAIL(vacuous: arm not observed)" }
                        else { "FAIL(prefill at <=MAX_VERIFY)" },
                     nref, leg_b_prefill,
                     if leg_b_ok { "PASS" } else { "FAIL(blind: instrument cannot see the cliff)" });
        }
        println!("{}", da::stats_line());
        if all_ok {
            println!("DISPATCH_ASSERT OK (leg A contract + leg B negative control both observed)");
        } else {
            println!("DISPATCH_ASSERT FAILED ({} violations recorded)", da::violations());
        }
        all_ok
    }

    pub fn probe_binv(&self) -> bool {
        let h = self.cfg.hidden_size;
        let l0 = &self.layers[0];
        // On MoE models probe the shared expert (a standard MLP); the stacked grouped-expert GEMM gets
        // its own batch-invariance check when that kernel lands.
        let (mlp_ref, im) = match &l0.mlp {
            Ffn::Dense(m) => (m, self.cfg.intermediate_size),
            Ffn::Moe(moe) => (&moe.shared, self.cfg.shared_expert_intermediate_size),
        };
        let mut cases: Vec<(&str, &W, usize, usize)> = vec![
            ("mlp.gate", &mlp_ref.gate, h, im),
            ("mlp.down", &mlp_ref.down, im, h),
        ];
        if let Some(la) = &l0.la {
            // The FUSED in_proj is the one that most needs this: it stacks four source tensors along M,
            // each with its own NVFP4 tensor scale, resolved by a per-tile lookup. A boundary error
            // there would corrupt one segment's magnitudes and nothing else would notice.
            match &la.in_proj {
                GdnIn::Fused(w) => cases.push(("gdn.in_proj FUSED", w, h, GdnIn::fused_m(&self.cfg))),
                GdnIn::Split { qkv, .. } =>
                    cases.push(("gdn.in_proj_qkv", qkv, h, self.cfg.key_dim() * 2 + self.cfg.value_dim())),
            }
            cases.push(("gdn.out_proj", &la.out_proj, self.cfg.value_dim(), h));
        }
        let fa_ref = self.layers.iter().find(|ly| ly.fa.is_some()).and_then(|ly| ly.fa.as_ref());
        if let Some(fa) = fa_ref {
            if let AttnIn::Fused(w) = &fa.qkv { cases.push(("attn.qkv FUSED", w, h, AttnIn::fused_m(&self.cfg))); }
            // Phase-8: o_proj was never probed (and Split qkv neither) — the depth-2 losslessness
            // break implicates a verify col>=1; cover every FA projection.
            let nh = self.cfg.num_heads.max(1);
            let hd = self.cfg.head_dim.max(1);
            match &fa.qkv {
                AttnIn::Split { q, k, v } => {
                    cases.push(("attn.q (split)", q, h, nh * hd));
                    cases.push(("attn.k (split)", k, h, (self.cfg.num_kv_heads.max(1)) * hd));
                    cases.push(("attn.v (split)", v, h, (self.cfg.num_kv_heads.max(1)) * hd));
                }
                _ => {}
            }
            cases.push(("attn.o_proj", &fa.o_proj, nh * hd, h));
        }
        // A TP vocab-sharded head has v/2 rows — probe the shard's true shape, not cfg.vocab_size.
        if let Some(lh) = &self.lm_head {
            let rows = if self.lm_head_sharded { self.cfg.vocab_size / self.shard_factor() } else { self.cfg.vocab_size };
            cases.push(("lm_head", lh, h, rows));
        }

        println!("=== bitwise batch-invariance: col 0 of an N-wide verify vs a N=1 decode ===");
        let mut all_ok = true;
        for (name, w, inn, outn) in cases {
            // One X, MAX_VERIFY+1 columns. Column 0 is shared by every call, so any difference in the
            // result is the kernel reacting to N — which is exactly what must not happen.
            let nref = MAX_VERIFY + 1;
            let x_host: Vec<half::bf16> = (0..inn * nref)
                .map(|_| half::bf16::from_f32(rand::random::<f32>() * 2.0 - 1.0)).collect();
            let x = self.dev.htod_sync_copy(&x_host).unwrap();
            self.dev.synchronize().unwrap();

            // Phase-8 (Part A): ALL-column invariance — each column c of an N-wide call must
            // match an N=1 call on the SAME x column (col-0-only checked every earlier run; the
            // depth-2 losslessness break at token 442 implicates verify cols >= 1).
            let mut refs: Vec<Vec<half::bf16>> = Vec::with_capacity(nref);
            for c in 0..nref {
                let xc = self.dev.htod_sync_copy(&x_host[c * inn..(c + 1) * inn]).unwrap();
                let mut o1 = unsafe { self.dev.alloc::<half::bf16>(outn) }.unwrap();
                self.gemm_act(w, &xc, &mut o1, inn, outn, 1);
                self.sync_stream();
                refs.push(self.dev.dtoh_sync_copy(&o1).unwrap());
                drop(o1);
            }
            let mut col0: Option<Vec<half::bf16>> = None;
            let mut worst = 0u16;
            let mut first_bad_n = 0usize;
            let mut first_bad_col = 0usize;
            for n in 1..=MAX_VERIFY {
                let mut out = unsafe { self.dev.alloc::<half::bf16>(outn * n) }.unwrap();
                self.gemm_act(w, &x, &mut out, inn, outn, n);
                self.sync_stream();
                let got: Vec<half::bf16> = self.dev.dtoh_sync_copy(&out).unwrap();
                if n == 1 { col0 = Some(got[..outn].to_vec()); }
                for c in 0..n {
                    for i in 0..outn {
                        // BITWISE. Not "close": a 1-ulp drift in ANY column re-argmaxes a token
                        // somewhere in a long generation, and MTP silently stops being lossless.
                        let d = refs[c][i].to_bits() ^ got[c * outn + i].to_bits();
                        if d != 0 && d > worst { worst = d; }
                        if d != 0 && first_bad_n == 0 { first_bad_n = n; first_bad_col = c; }
                    }
                }
            }
            if first_bad_n > 0 {
                println!("  [diag] {name}: first divergent N = {first_bad_n} col = {first_bad_col} (worst xor 0x{worst:04x})");
            }
            // Independent reference: batch > MAX_VERIFY takes the prefill path (dequant + cuBLAS).
            let mut r = unsafe { self.dev.alloc::<half::bf16>(outn * nref) }.unwrap();
            self.gemm_act(w, &x, &mut r, inn, outn, nref);
            self.sync_stream();
            let rv: Vec<half::bf16> = self.dev.dtoh_sync_copy(&r).unwrap();
            let c0 = col0.unwrap();
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for i in 0..outn {
                let (a, b) = (c0[i].to_f32() as f64, rv[i].to_f32() as f64);
                num += (a - b) * (a - b);
                den += b * b;
            }
            let rel = (num / den.max(1e-30)).sqrt();
            let inv_ok = worst == 0;
            all_ok &= inv_ok;
            println!("  {:<16} [{}x{}]  N=1..{} col-0 {}   | vs dequant+cuBLAS rel-L2 {:.5}",
                     name, outn, inn, MAX_VERIFY,
                     if inv_ok { "BIT-IDENTICAL".to_string() }
                     else { format!("DIVERGED (worst xor 0x{:04x})", worst) }, rel);
        }
        println!("{}", if all_ok { "PASS — greedy MTP is lossless at every depth up to 16." }
                       else { "FAIL — batch-invariance is broken; greedy MTP is NOT lossless." });
        all_ok
    }

    /// Phase-7 F9 gate — TP-mode batch-invariance probe. BOTH ranks run this in SPMD lockstep
    /// (the sharded forward's all-reduces pair up); each prints its own verdict.
    ///
    /// At ONE position (the end of `prompt`) it sweeps the verify width N=1..MAX_VERIFY through
    /// the REAL verify path and asserts column 0 is BIT-IDENTICAL to the N=1 forward:
    ///   * the column-0 hidden state (bf16 bits) — the LM-head input;
    ///   * the full column-0 FP32 logits (bitwise, every vocab row);
    ///   * the greedy argmax.
    /// N=1 is the engine's decode-shaped forward, and `--bench-mtp` separately gates N=1 against
    /// a sequential decode, so (this sweep) + (bench-mtp) = "col 0 of an N-wide verify == decode".
    ///
    /// The single-process `--probe-binv` cannot see this class: it probes GEMM entry points only,
    /// at world=1. F9 lived in the attention key partition of the sharded `_e` lane
    /// (kernels/gpu_batch.cu gve_*_impl), which no GEMM probe touches.
    pub fn probe_binv_tp(&self, prompt: &[u32], kv_stride: usize, max_n: usize) -> bool {
        let h = self.cfg.hidden_size;
        let v = self.cfg.vocab_size;
        // Gate hygiene (boundary-targeted): the pre-fix defect only bit when the batch-max
        // partition crossed a seg_size boundary, so a single arbitrary position can MISS it. Pick
        // the prompt length so (plen+1) sits 2 keys below a boundary of the _e lane's partition
        // (seg = the host's SEG formula, same inputs the dispatch uses): every width N >= 3 then
        // straddles it at this position, deterministically. A short prompt keeps 16 resets cheap.
        let nkv_eff = if self.attn_shard_factor() > 1 { self.eff_num_kv_heads() } else { self.cfg.num_kv_heads };
        let base = ((192 + nkv_eff.max(1) - 1) / nkv_eff.max(1)).max(4);
        let seg = base.max((kv_stride + 8191) / 8192).min(63);
        let l0 = prompt.len().min(2048);
        let mut plen = l0.max(64);
        if seg >= 3 {
            let want = (seg - 2) % seg;
            while (plen + 1) % seg != want { plen += 1; }
        }
        let mut ptoks: Vec<u32> = prompt[..l0].to_vec();
        while ptoks.len() < plen { ptoks.push(*ptoks.last().unwrap()); }
        let prompt: &[u32] = &ptoks[..];
        let plen = prompt.len();
        println!("  [binv-tp] probe position plen={plen} seg={seg} (plen+1) % seg = {}", (plen + 1) % seg);
        let mut pool = Pool::new(self.dev.clone());
        println!("=== TP batch-invariance probe (world={} rank={}): col 0 of an N-wide verify vs N=1 ===",
                 self.tp_world, self.tp_rank);
        let mut reference: Option<(Vec<half::bf16>, Vec<f32>, u32)> = None;
        let mut all_ok = true;
        for n in 1..=max_n {
            let mut state = self.new_batch_state(n.max(2), n.max(2), kv_stride);
            self.zero_slot_state(&mut state, 0, kv_stride);
            let (a0, hout) = self.prefill_batch(&mut pool, prompt, &mut state, 0, kv_stride, 0);
            pool.release_bf16(hout, h * plen);
            // Column 0 is the decode-shaped column; the extra columns are deterministic filler
            // (causality means they cannot affect column 0 — that is exactly what is asserted).
            let toks: Vec<u32> = (0..n)
                .map(|i| if i == 0 { a0 } else { (a0 + 1 + i as u32) % (v as u32) })
                .collect();
            let (lg, vout) = self.verify_forward_keep_logits_f32(&mut pool, &toks, &mut state,
                                                                 0, kv_stride, plen, None);
            self.sync_stream();
            // The engine's fp32-logits path requires cfg.lm_head_fp32; this model keeps the head
            // bf16, so the probe compares bf16 logits bitwise — exactly what the argmax consumes.
            let (lrow, vcol, tok) = match &lg {
                VerifyLogits::F32(l) => {
                    let host: Vec<f32> = self.dev.dtoh_sync_copy(l).unwrap();
                    let row = host[..v].to_vec();
                    let (mut bi, mut bv) = (0usize, f32::NEG_INFINITY);
                    for (i, &x) in row.iter().enumerate() {
                        if x > bv { bv = x; bi = i; }
                    }
                    let vh: Vec<half::bf16> = self.dev.dtoh_sync_copy(&vout).unwrap();
                    (row, vh[..h].to_vec(), bi as u32)
                }
                VerifyLogits::B16(l) => {
                    let host: Vec<half::bf16> = self.dev.dtoh_sync_copy(l).unwrap();
                    let row: Vec<f32> = host[..v].iter().map(|x| x.to_f32()).collect();
                    let (mut bi, mut bv) = (0usize, f32::NEG_INFINITY);
                    for (i, &x) in row.iter().enumerate() {
                        if x > bv { bv = x; bi = i; }
                    }
                    let vh: Vec<half::bf16> = self.dev.dtoh_sync_copy(&vout).unwrap();
                    (row, vh[..h].to_vec(), bi as u32)
                }
            };
            match lg {
                VerifyLogits::F32(l) => pool.release(l, v * n),
                VerifyLogits::B16(l) => pool.release_bf16(l, v * n),
            }
            pool.release_bf16(vout, h * n);
            match &reference {
                None => {
                    println!("  N=1 reference: tok={tok}");
                    reference = Some((vcol, lrow, tok));
                }
                Some((rv, rl, rt)) => {
                    let hdiff = rv.iter().zip(vcol.iter())
                        .filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                    let ldiff = rl.iter().zip(lrow.iter())
                        .filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                    let ok = hdiff == 0 && ldiff == 0 && tok == *rt;
                    all_ok &= ok;
                    println!("  N={n:<2} tok={tok} {} (hidden diffs {hdiff}/{h}, logits diffs {ldiff}/{v})",
                             if ok { "BIT-IDENTICAL" } else { "DIVERGED" });
                }
            }
        }
        println!("{}", if all_ok {
            "PROBE-BINV-TP: PASS — col 0 bit-identical at every verify width 1..=N."
        } else {
            "PROBE-BINV-TP: FAIL — column 0 depends on the verify width."
        });
        all_ok
    }
}

impl GpuModel {
    /// Brute-force sweep of all cuBLAS GEMM algos for shape (M=outn, K=inn), measuring the N=1 vs
    /// N=2 column-0 divergence for each. The goal: find an algo whose result is batch-invariant
    /// (diff == 0) so the MTP verify (N=K) matches the decode (N=1) numerically. Prints every algo's
    /// status + diff; algos with diff 0 are the fix for the 9B/27B cuBLAS divergence.
    pub fn probe_gemm_sweep(&self, m: usize, k: usize) {
        use cudarc::cublas::sys;
        use std::mem::transmute;
        let w_host: Vec<half::bf16> = (0..m * k)
            .map(|_| half::bf16::from_f32((rand::random::<f32>() * 2.0) - 1.0)).collect();
        let x_host: Vec<half::bf16> = (0..k * 2)
            .map(|_| half::bf16::from_f32((rand::random::<f32>() * 2.0) - 1.0)).collect();
        let w = self.dev.htod_sync_copy(&w_host).unwrap();
        let x = self.dev.htod_sync_copy(&x_host).unwrap();
        let out1 = unsafe { self.dev.alloc::<half::bf16>(m) }.unwrap();
        let out2 = unsafe { self.dev.alloc::<half::bf16>(m * 2) }.unwrap();
        self.dev.synchronize().unwrap();

        let handle = *self.blas.handle();
        let bf = sys::cudaDataType_t::CUDA_R_16BF;
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        // Algo values: -1=DEFAULT, 0..24=ALGO0..23 (non-tensor-op), 100..124=TENSOR_OP ALGO0..23.
        let mut algos: Vec<i32> = vec![-1];
        algos.extend(0..24);
        algos.extend(100..124);
        println!("=== cuBLAS algo sweep  M={} K={} (N=1 vs N=2 col-0 diff) ===", m, k);
        let mut best: Vec<(i32, f32)> = vec![];
        for av in algos {
            let algo: sys::cublasGemmAlgo_t = unsafe { transmute(av) };
            // N=1
            let r1 = unsafe {
                sys::cublasGemmEx(handle, sys::cublasOperation_t::CUBLAS_OP_T,
                    sys::cublasOperation_t::CUBLAS_OP_N, m as i32, 1, k as i32,
                    (&alpha) as *const _ as *const _, *w.device_ptr() as *const _, bf, k as i32,
                    *x.device_ptr() as *const _, bf, k as i32, (&beta) as *const _ as *const _,
                    *out1.device_ptr() as *mut _, bf, m as i32,
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32F, algo).result()
            };
            // N=2
            let r2 = unsafe {
                sys::cublasGemmEx(handle, sys::cublasOperation_t::CUBLAS_OP_T,
                    sys::cublasOperation_t::CUBLAS_OP_N, m as i32, 2, k as i32,
                    (&alpha) as *const _ as *const _, *w.device_ptr() as *const _, bf, k as i32,
                    *x.device_ptr() as *const _, bf, k as i32, (&beta) as *const _ as *const _,
                    *out2.device_ptr() as *mut _, bf, m as i32,
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32F, algo).result()
            };
            if r1.is_err() || r2.is_err() {
                continue; // algo not valid for this shape
            }
            self.sync_stream();
            let o1: Vec<half::bf16> = self.dev.dtoh_sync_copy(&out1).unwrap();
            let o2: Vec<half::bf16> = self.dev.dtoh_sync_copy(&out2).unwrap();
            let mut mx = 0.0f32;
            for i in 0..m {
                let d = (half::bf16::to_f32(o1[i]) - half::bf16::to_f32(o2[i])).abs();
                if d > mx { mx = d; }
            }
            let tag = if av < 0 { "DEFAULT".to_string() }
                else if av >= 100 { format!("ALGO{}_TENSOR_OP", av - 100) }
                else { format!("ALGO{}", av) };
            println!("  {:>20} ({}): diff={:.6}", tag, av, mx);
            if mx == 0.0 { best.push((av, mx)); }
        }
        if best.is_empty() {
            println!(">>> NO batch-invariant cuBLAS algo found for M={} K={} — need cuBLASLt split-K disable.", m, k);
        } else {
            println!(">>> BATCH-INVARIANT algos (diff=0): {:?}", best.iter().map(|&(a,_)| a).collect::<Vec<_>>());
        }
    }
}
