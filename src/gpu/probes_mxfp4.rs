//! `impl GpuModel`: MXFP4 probes (`probe_mxfp4`, `probe_mxfp4_fused`, cross-chain probes against the reference chain).
//! Probes only; not on the serving path. Map: `mod.rs`.

use super::*;

impl GpuModel {
    // ===================== MXFP4-NATIVE PHASE 0 PROBE (--probe-mxfp4) =====================
    //
    // Standalone OMMA microbenchmark on the REAL model's fp4 tensors: the sm_121a native
    // bf16-chain dispatch (gemm_mma_fp4_b + auto split-K, exactly what serving runs), both fed
    // identical weights — the OMMA side via the lossless load-time repack (src/mxfp4.rs).
    // GO/NO-GO at ~215 GB/s sustained on the byte-weighted aggregate
    // (docs/QWEN_MXFP4_NATIVE_DESIGN.md §7). Probe-only: loads src/ptx/mxfp4_bench.ptx lazily;
    // the serving manifest and the default path are untouched. Sanity: rel-L2 of column 0
    // (OMMA vs bf16 chain) — the activation-quant tolerance fork, loose bound here.
    pub fn probe_mxfp4(&self) {
        use cudarc::nvrtc::Ptx;

        struct Case {
            label: String,
            m: usize,
            k: usize,
            qw: cudarc::driver::CudaSlice<u8>,
            sc: cudarc::driver::CudaSlice<u8>,
            gs: cudarc::driver::CudaSlice<f32>,
        }

        let ptx = Ptx::from_src(crate::ptx_embed::MXFP4_BENCH);
        self.dev.load_ptx(ptx, "mxfp4_bench", &["mxfp4_gemv_bench"]).expect("load mxfp4_bench");
        let f_omma = self.dev.get_func("mxfp4_bench", "mxfp4_gemv_bench")
            .expect("mxfp4_gemv_bench not in mxfp4_bench module");

        let mut cases: Vec<Case> = Vec::new();
        {
            let mut push = |w: &W, label: &str| {
                if let W::Nvfp4 { qweight, scales, gs, m, k } = w {
                    // INTENTIONAL CudaSlice::clone (deep copy) — S5F4 class-audit convention:
                    // the probe snapshots the weights ONCE into its Case table; the weights are
                    // static for the probe's lifetime, so the copy can't go stale (R14's class).
                    // Probe-only — never on a serving path (a CudaSlice::clone is alloc + dtod).
                    cases.push(Case { label: label.to_string(), m: *m, k: *k,
                                      qw: qweight.clone(), sc: scales.clone(), gs: gs.clone() });
                }
            };
            for (li, l) in self.layers.iter().enumerate() {
                let p = format!("L{li}");
                if let Some(la) = &l.la {
                    match &la.in_proj {
                        GdnIn::Fused(w) => push(w, &format!("{p}.in_proj")),
                        GdnIn::Split { qkv, z, .. } => {
                            push(qkv, &format!("{p}.in_proj_qkv"));
                            push(z, &format!("{p}.in_proj_z"));
                        }
                    }
                    push(&la.out_proj, &format!("{p}.out_proj"));
                }
                if let Some(fa) = &l.fa {
                    if let AttnIn::Fused(w) = &fa.qkv { push(w, &format!("{p}.qkv")); }
                    push(&fa.o_proj, &format!("{p}.o_proj"));
                }
                match &l.mlp {
                    Ffn::Dense(mp) => {
                        push(&mp.gate, &format!("{p}.gate"));
                        push(&mp.up, &format!("{p}.up"));
                        push(&mp.down, &format!("{p}.down"));
                    }
                    Ffn::Moe(_) => {} // 122B grouped MoE is Phase 2 scope
                }
            }
            if let Some(mp) = &self.mtp {
                push(&mp.fc, "MTP.fc");
                if let Ffn::Dense(m) = &mp.mlp {
                    push(&m.gate, "MTP.gate");
                    push(&m.up, "MTP.up");
                    push(&m.down, "MTP.down");
                }
            }
        }
        if cases.is_empty() {
            println!("probe-mxfp4: no W::Nvfp4 tensors found (is --model-dir an NVFP4 artifact?)");
            return;
        }

        let rounds = 3u32;
        let iters = 40u32;
        let mut agg_p = 0.0f64; // best seconds
        let mut agg_o = 0.0f64;
        let mut agg_bytes = 0.0f64;
        let mut max_rel = 0.0f32;
        println!("=== MXFP4-native Phase 0: OMMA (sm_121a) vs production bf16-chain GEMM — real fp4 tensors ===");
        println!("  GO/NO-GO: OMMA sustained >= ~215 GB/s on the byte-weighted aggregate (fp8-parity)");
        println!("  bytes counted: packed weights + E4M3 scales (the J3 audit convention)");
        for c in &cases {
            assert!(c.m % 16 == 0 && c.k % 64 == 0, "{}: M%16==0 K%64==0 required (got {}x{})",
                    c.label, c.m, c.k);
            let (ntm, nks) = (c.m / 16, c.k / 64);
            let bytes = (c.m * c.k / 2 + c.m * c.k / 16) as f64;

            // ---- lossless repack (host, threaded) + upload ----
            let t0 = std::time::Instant::now();
            let qw_h = self.dev.dtoh_sync_copy(&c.qw).unwrap();
            let sc_h = self.dev.dtoh_sync_copy(&c.sc).unwrap();
            let gs_h = self.dev.dtoh_sync_copy(&c.gs).unwrap();
            let (aimg, sfa) = crate::mxfp4::repack_nvfp4_omma(&qw_h, &sc_h, c.m, c.k);
            let d_aimg = self.dev.htod_sync_copy(&aimg).unwrap();
            let d_sfa = self.dev.htod_sync_copy(&sfa).unwrap();
            let d_gs = self.dev.htod_sync_copy(&gs_h).unwrap();
            let trepack = t0.elapsed();

            // ---- activations: deterministic 8 tokens, packed B side ----
            let mut xh: Vec<half::bf16> = Vec::with_capacity(8 * c.k);
            let mut s = 12345u32.wrapping_add((c.m as u32).wrapping_mul(7919))
                                       .wrapping_add((c.k as u32).wrapping_mul(104729));
            for _ in 0..8 * c.k {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                xh.push(half::bf16::from_f32((((s >> 8) % 2001) as f32 - 1000.0) / 128.0));
            }
            let (bp, sfb) = crate::mxfp4::pack_b_host(&xh, c.k);
            let d_x = self.dev.htod_sync_copy(&xh).unwrap();
            let d_bp = self.dev.htod_sync_copy(&bp).unwrap();
            let d_sfb = self.dev.htod_sync_copy(&sfb).unwrap();
            let d_out_p = unsafe { self.dev.alloc::<half::bf16>(c.m) }.unwrap();
            let d_out_o = unsafe { self.dev.alloc::<half::bf16>(8 * c.m) }.unwrap();
            self.dev.synchronize().unwrap();

            let grid = (ntm.min(GB10_SMS * MMA_FP4_BLOCKS_PER_SM)) as u32;
            let launch_o = |out: &cudarc::driver::CudaSlice<half::bf16>| unsafe {
                f_omma.clone().launch_on_stream(
                    &self.stream,
                    LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                    (d(out), d(&d_aimg), d(&d_sfa), d(&d_gs), d(&d_bp), d(&d_sfb),
                     ntm as i32, nks as i32, c.m as i32))
                    .unwrap_or_else(|e| panic!("mxfp4_gemv_bench launch: {e:?}"));
            };
            let launch_p = || {
                self.launch_gemm_fp4(d(&d_out_p), d(&c.qw), d(&c.sc), d(&c.gs), d(&d_x),
                                     c.m, c.k, 1, 0);
            };

            // ---- interleaved best-of-RR: production vs OMMA ----
            let mut best_p = 0.0f64;
            let mut best_o = 0.0f64;
            for _ in 0..rounds {
                for _ in 0..5 { launch_p(); }
                self.sync_stream();
                let t0 = std::time::Instant::now();
                for _ in 0..iters { launch_p(); }
                self.sync_stream();
                let dt = t0.elapsed().as_secs_f64() / iters as f64;
                best_p = best_p.max(bytes / dt / 1e9);

                for _ in 0..5 { launch_o(&d_out_o); }
                self.sync_stream();
                let t0 = std::time::Instant::now();
                for _ in 0..iters { launch_o(&d_out_o); }
                self.sync_stream();
                let dt = t0.elapsed().as_secs_f64() / iters as f64;
                best_o = best_o.max(bytes / dt / 1e9);
            }
            agg_p += bytes / best_p / 1e9;
            agg_o += bytes / best_o / 1e9;
            agg_bytes += bytes;

            // ---- numerics sanity: column 0, OMMA vs bf16 chain ----
            launch_p();
            launch_o(&d_out_o);
            self.sync_stream();
            let p0 = self.dev.dtoh_sync_copy(&d_out_p).unwrap();
            let o0 = self.dev.dtoh_sync_copy(&d_out_o).unwrap();
            let mut num = 0.0f64;
            let mut den = 0.0f64;
            for i in 0..c.m {
                let a = half::bf16::to_f32(p0[i]) as f64;
                let b = half::bf16::to_f32(o0[i]) as f64;
                num += (a - b) * (a - b);
                den += a * a;
            }
            let rel = (num.sqrt() / den.sqrt().max(1e-9)) as f32;
            max_rel = max_rel.max(rel);
            let db = 100.0 * (best_o / best_p - 1.0);
            println!("  {:<16} M={:>6} K={:>6}  prod {:>6.1}  omma {:>6.1}  {:+5.1}%   rel-L2 {:.1e}   (repack {:?})",
                     c.label, c.m, c.k, best_p, best_o, db, rel, trepack);
        }
        let t_p = agg_p;
        let t_o = agg_o;
        println!("  ---- aggregate ({:.1} MB fp4 weights + scales) ----", agg_bytes / 1e6);
        println!("  production bf16 chain: {:6.1} GB/s", agg_bytes / t_p / 1e9);
        println!("  OMMA native (sm_121a): {:6.1} GB/s   ({:+.1}%)", agg_bytes / t_o / 1e9,
                 100.0 * ((agg_bytes / t_o) / (agg_bytes / t_p) - 1.0));
        println!("  max |rel-L2| (col 0, vs bf16 chain): {:.1e}", max_rel);
        let omma = agg_bytes / t_o / 1e9;
        if omma >= 215.0 {
            println!("  >>> GO: sustained OMMA >= 215 GB/s — Phase 0 PASSES, proceed to Phase 1");
        } else {
            println!("  >>> NO-GO: sustained OMMA {omma:.1} GB/s < 215 GB/s — the mode cannot pay \
                      for itself; write the THINGS_TRIED row and stop");
        }
    }

    /// Fused activation-quant byte-identity probe (EXPERT_FUSED_QUANT_RESPONSE.md §11 P1,
    /// `--probe-mxfp4-fused`). Synthetic weights — for the Cartesian product
    /// {dense, slot, grouped} × {XSILU 0,1} × N ∈ {1,2,4,8,16} × K ∈ {1536,4096,13312}
    /// byte-compares the FUSED kernels' C against the current quant(+silu)+GEMM chain's C
    /// (bf16 bit equality, no tolerance) AND the Bp/SFB fragment streams against the
    /// standalone quant kernels' output (host reference). Edge sweep: all-zero rows,
    /// subnormal-scale amax (amax/6 < 2⁻⁹), saturation (amax ≥ 2688), mixed ±0, and
    /// near-tie e4m3_ceil boundaries. Prints `FUSED_BIT_IDENTITY OK` when everything passes.
    /// Requires the mxfp4 modules resident (--mxfp4=1 at load).
    pub fn probe_mxfp4_fused(&self) {
        use crate::mxfp4::{e2m1_rn, e4m3_ceil, ue4m3_f};
        use half::bf16;
        let st = match &self.mxfp4 {
            Some(s) => s,
            None => {
                println!("probe-mxfp4-fused: no Mxfp4State (run with --mxfp4=1 at load)");
                return;
            }
        };
        const E: usize = 8;          // synthetic local experts
        const M: usize = 512;        // per-expert output rows (ntm = 32)
        const TOPK: usize = 8;
        // Probe-owned Bp/SFB scratch for the synthetic MoE launches. The module's moe_bp/moe_sfb
        // are sized from the LOADED MODEL's geometry and are NULL on dense models (num_experts=0
        // -> moe_groups_max=0 -> zero-length slice) — writing through them OOB'd at 0x0 on the
        // 9B probe (2026-08-12). The synthetic worst case: slot dn at N=16 -> bk = 16*TOPK = 128
        // groups x nks <= 512 (K=13312 -> 208).
        const SCR_GROUPS: usize = 16 * TOPK;
        let scr_bp = self.dev.alloc_zeros::<u32>(SCR_GROUPS * 512 * 64).unwrap();
        let scr_sfb = self.dev.alloc_zeros::<u32>(SCR_GROUPS * 512 * 32).unwrap();
        let mut n_checks = 0u64;
        let mut n_fail = 0u64;
        let mut fail_notes: Vec<String> = Vec::new();

        // Host mirror of mxfp4_quant_pack_ng_b: ngroups 8-token groups of X, N real rows per
        // group, row_stride between group rows; Bp/SFB in the group-strided OMMA layout.
        // The fragment oracle: the fused kernels' internal packs are byte-identical to the
        // standalone quant's by construction, so validating the standalone pack against this
        // reference plus the random-weight C-compares covers both.
        fn pack_ng_host(x: &[bf16], k: usize, ngroups: usize, nks: usize, n: usize,
                        row_stride: usize) -> (Vec<u32>, Vec<u32>) {
            assert!(x.len() >= ngroups * row_stride * k, "pack_ng_host: x too small");
            let mut bp = vec![0u32; ngroups * nks * 64];
            let mut sfb = vec![0u32; ngroups * nks * 32];
            let mut scales = [[0u8; 4]; 8];
            let mut codes = [[0u8; 64]; 8];
            for grp in 0..ngroups {
                for ks in 0..nks {
                    for rr in 0..8usize {
                        let real = rr < n;
                        let row = grp * row_stride + rr;
                        for v in 0..4 {
                            let amax = if real {
                                (0..16).map(|i| x[row * k + ks * 64 + v * 16 + i].to_f32().abs())
                                    .fold(0.0f32, f32::max)
                            } else { 0.0f32 };
                            scales[rr][v] = e4m3_ceil(amax / 6.0);
                            let inv = if scales[rr][v] == 0 { 0.0 } else { 1.0 / ue4m3_f(scales[rr][v]) };
                            for i in 0..16usize {
                                codes[rr][v * 16 + i] = if real {
                                    e2m1_rn(x[row * k + ks * 64 + v * 16 + i].to_f32() * inv)
                                } else { 0 };
                            }
                        }
                    }
                    for lane in 0..32usize {
                        let (g, t) = (lane >> 2, lane & 3);
                        let mut b0 = 0u32;
                        let mut b1 = 0u32;
                        for j in 0..8usize {
                            b0 |= (codes[g][8 * t + j] as u32) << (4 * j);
                            b1 |= (codes[g][32 + 8 * t + j] as u32) << (4 * j);
                        }
                        let bb = ((grp * nks + ks) * 32 + lane) * 2;
                        bp[bb] = b0;
                        bp[bb + 1] = b1;
                        if t == 0 { sfb[(grp * nks + ks) * 32 + lane] = u32::from_le_bytes(scales[g]); }
                    }
                }
            }
            (bp, sfb)
        }

        struct Tally {
            n_checks: u64,
            n_fail: u64,
            fail_notes: Vec<String>,
        }
        impl Tally {
            fn byte(&mut self, label: String, a: &[bf16], b: &[bf16]) {
                self.n_checks += 1;
                if a.len() != b.len() || !a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()) {
                    self.n_fail += 1;
                    self.fail_notes.push(label);
                }
            }
            fn frag(&mut self, label: String, a: &[u32], b: &[u32]) {
                self.n_checks += 1;
                // The device scratch is module-sized (all groups/ksteps); the reference defines
                // the ACTIVE extent (contiguous from group 0) — compare that prefix only.
                let n = b.len();
                if a.len() < n || a[..n] != b[..] {
                    self.n_fail += 1;
                    let m = a.len().min(n);
                    let mut shown = 0;
                    for i in 0..m {
                        if a[i] != b[i] {
                            if shown < 3 {
                                eprintln!("  [diff] {label}: [{i}] dev={:#010x} host={:#010x}", a[i], b[i]);
                            }
                            shown += 1;
                        }
                    }
                    if shown == 0 {
                        eprintln!("  [diff] {label}: length {} vs {}", a.len(), n);
                    } else {
                        eprintln!("  [diff] {label}: {shown} differing u32s");
                    }
                    self.fail_notes.push(label);
                }
            }
        }
        let mut tally = Tally { n_checks: 0, n_fail: 0, fail_notes: Vec::new() };

        // ---- deterministic random helpers (LCG, like probe_mxfp4) ----
        let mut lcg = 0x5EED_2026u32;
        let mut rnd = move || {
            lcg = lcg.wrapping_mul(1664525).wrapping_add(1013904223);
            lcg
        };
        let rnd_row = |rnd: &mut dyn FnMut() -> u32, k: usize| -> Vec<bf16> {
            (0..k).map(|_| bf16::from_f32((((rnd() >> 8) % 2001) as f32 - 1000.0) / 128.0)).collect()
        };
        let rnd_row_wide = |rnd: &mut dyn FnMut() -> u32, k: usize| -> Vec<bf16> {
            let mut v = Vec::with_capacity(2 * k);
            for _ in 0..k {
                v.push(bf16::from_f32((((rnd() >> 8) % 2001) as f32 - 1000.0) / 128.0));
                v.push(bf16::from_f32((((rnd() >> 8) % 2001) as f32 - 1000.0) / 128.0));
            }
            v
        };

        // ---- synthetic OMMA weight upload (E experts x M x K, raw -> omma repack) ----
        let upload_weights = |rnd: &mut dyn FnMut() -> u32, k: usize| {
            let total_m = E * M;
            let mut qw = vec![0u8; total_m * k / 2];
            let mut sc = vec![0u8; total_m * k / 16];
            for b in qw.iter_mut() { *b = (rnd() >> 16) as u8; }
            for b in sc.iter_mut() { *b = (rnd() >> 24) as u8; }
            let gs: Vec<f32> = (0..total_m / 16).map(|_| 1.0f32).collect();
            let (aimg, sfa) = crate::mxfp4::repack_nvfp4_omma_raw(&qw, &sc, total_m, k);
            let d_aimg = self.dev.htod_sync_copy(&aimg).unwrap();
            let d_sfa = self.dev.htod_sync_copy(&sfa).unwrap();
            let d_gs = self.dev.htod_sync_copy(&gs).unwrap();
            (d_aimg, d_sfa, d_gs)
        };

        let zero_buf = |n: usize| -> B {
            let mut b = unsafe { self.dev.alloc::<bf16>(n) }.unwrap();
            self.dev.memset_zeros(&mut b).unwrap();
            b
        };

        println!("=== probe-mxfp4-fused: fused quant+GEMM vs today's quant(+silu)+GEMM chain ===");
        println!("  product: {{dense, slot, grouped}} x {{XSILU 0,1}} x N x K; byte-compare C + Bp/SFB fragments");

        for &k in &[1536usize, 4096, 13312] {
            let nks = k / 64;
            let ntm = M / 16;
            let (d_aimg, d_sfa, d_gs) = upload_weights(&mut rnd, k);
            let (ap, sp, gp) = (*d_aimg.device_ptr() as u64, *d_sfa.device_ptr() as u64,
                                *d_gs.device_ptr() as u64);

            for &n in &[1usize, 2, 4, 8, 16] {
                // ================= DENSE arm (XSILU n/a) =================
                {
                    let x: Vec<bf16> = {
                        let mut v = Vec::with_capacity(16 * k);
                        for _ in 0..16 { v.extend(rnd_row(&mut rnd, k)); }
                        v
                    };
                    let d_x = self.dev.htod_sync_copy(&x).unwrap();
                    let xp = *d_x.device_ptr() as u64;
                    let c_chain = zero_buf(n * M);
                    let c_fused = zero_buf(n * M);
                    let (cp, fp) = (*c_chain.device_ptr() as u64, *c_fused.device_ptr() as u64);
                    let nr = if n <= 1 { 1 } else if n <= 8 { 8 } else { 16 };
                    if n <= 8 {
                        // chain: quant_pack_b + gemv_native_b (the production pair)
                        unsafe {
                            st.quant.clone().launch_on_stream(&self.stream,
                                LaunchConfig { grid_dim: (nks as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                                (xp, k as i32, n as i32, d(&st.bp), d(&st.sfb))).unwrap();
                            st.gemv.clone().launch_on_stream(&self.stream,
                                LaunchConfig { grid_dim: (ntm as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                                (cp, ap, sp, gp, d(&st.bp), d(&st.sfb),
                                 ntm as i32, nks as i32, M as i32, n as i32)).unwrap();
                        }
                    } else {
                        // N=16: the production quant_pack_b never writes Bp group 1 (a latent
                        // N>8 gap whose columns are never consumed), so the deterministic
                        // reference is the two-group pack (quant_ng) the fused NR=16 kernel
                        // implements — the fused C is compared against that.
                        unsafe {
                            st.quant_ng.clone().launch_on_stream(&self.stream,
                                LaunchConfig { grid_dim: (2 * nks as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                                (xp, k as i32, 2i32, nks as i32, 8i32, 8i32,
                                 d(&scr_bp), d(&scr_sfb), 0u64, 0i32)).unwrap();
                            st.gemv.clone().launch_on_stream(&self.stream,
                                LaunchConfig { grid_dim: (ntm as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                                (cp, ap, sp, gp, d(&scr_bp), d(&scr_sfb),
                                 ntm as i32, nks as i32, M as i32, n as i32)).unwrap();
                        }
                    }
                    let f = match nr { 1 => &st.gemv_fused1, 8 => &st.gemv_fused8, _ => &st.gemv_fused16 };
                    unsafe {
                        f.clone().launch_on_stream(&self.stream,
                            LaunchConfig { grid_dim: (ntm as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                            (fp, ap, sp, gp, xp, ntm as i32, nks as i32, M as i32, n as i32)).unwrap();
                    }
                    self.sync_stream();
                    tally.byte(format!("dense N={n} K={k} C"),
                              &self.dev.dtoh_sync_copy(&c_chain).unwrap(),
                              &self.dev.dtoh_sync_copy(&c_fused).unwrap());
                    // fragments: standalone quant (same params) vs host reference
                    if n <= 8 {
                        unsafe {
                            st.quant.clone().launch_on_stream(&self.stream,
                                LaunchConfig { grid_dim: (nks as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                                (xp, k as i32, n as i32, d(&st.bp), d(&st.sfb))).unwrap();
                        }
                        self.sync_stream();
                        let (ref_bp, ref_sfb) = pack_ng_host(&x, k, 1, nks, n, 1);
                        tally.frag(format!("dense N={n} K={k} Bp"),
                                 &self.dev.dtoh_sync_copy(&st.bp).unwrap(), &ref_bp);
                        tally.frag(format!("dense N={n} K={k} SFB"),
                                 &self.dev.dtoh_sync_copy(&st.sfb).unwrap(), &ref_sfb);
                    } else {
                        unsafe {
                            st.quant_ng.clone().launch_on_stream(&self.stream,
                                LaunchConfig { grid_dim: (2 * nks as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                                (xp, k as i32, 2i32, nks as i32, 8i32, 8i32,
                                 d(&scr_bp), d(&scr_sfb), 0u64, 0i32)).unwrap();
                        }
                        self.sync_stream();
                        let (ref_bp, ref_sfb) = pack_ng_host(&x, k, 2, nks, 8, 8);
                        tally.frag(format!("dense N={n} K={k} Bp"),
                                 &self.dev.dtoh_sync_copy(&scr_bp).unwrap(), &ref_bp);
                        tally.frag(format!("dense N={n} K={k} SFB"),
                                 &self.dev.dtoh_sync_copy(&scr_sfb).unwrap(), &ref_sfb);
                    }
                }

                // ================= SLOT arm (XSILU 0 = gu, 1 = dn) =================
                {
                    let bk = n * TOPK;
                    let x: Vec<bf16> = {
                        let mut v = Vec::with_capacity(n * k);
                        for _ in 0..n { v.extend(rnd_row(&mut rnd, k)); }
                        v
                    };
                    let d_x = self.dev.htod_sync_copy(&x).unwrap();
                    let xp = *d_x.device_ptr() as u64;
                    let gu_s: Vec<bf16> = {
                        let mut v = Vec::with_capacity(bk * 2 * k);
                        for _ in 0..bk { v.extend(rnd_row_wide(&mut rnd, k)); }
                        v
                    };
                    let d_gu_s = self.dev.htod_sync_copy(&gu_s).unwrap();
                    let gup = *d_gu_s.device_ptr() as u64;
                    let mut ids: Vec<i32> = (0..bk).map(|_| (rnd() % E as u32) as i32).collect();
                    for (i, v) in ids.iter_mut().enumerate() {
                        if i % 7 == 3 { *v = -1; }          // remote-expert slots -> exact-zero path
                        if i % 11 == 5 { *v = E as i32 + 1; }
                    }
                    let d_ids = self.dev.htod_sync_copy(&ids).unwrap();
                    let ip = *d_ids.device_ptr() as u64;

                    // --- gu side (XSILU=0) ---
                    let c_chain = zero_buf(bk * M);
                    let c_fused = zero_buf(bk * M);
                    let (cp, fp) = (*c_chain.device_ptr() as u64, *c_fused.device_ptr() as u64);
                    unsafe {
                        st.quant_ng.clone().launch_on_stream(&self.stream,
                            LaunchConfig { grid_dim: (n as u32 * nks as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                            (xp, k as i32, n as i32, nks as i32, 1i32, 1i32,
                             d(&scr_bp), d(&scr_sfb), 0u64, 0i32)).unwrap();
                        st.moe_slot.clone().launch_on_stream(&self.stream,
                            LaunchConfig { grid_dim: (ntm as u32, bk as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                            (cp, ap, sp, gp, d(&scr_bp), d(&scr_sfb), ip,
                             nks as i32, TOPK as i32, 0i32, E as i32, ntm as i32)).unwrap();
                        st.moe_slot_fused0.clone().launch_on_stream(&self.stream,
                            LaunchConfig { grid_dim: (ntm as u32, bk as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                            (fp, ap, sp, gp, xp, ip, nks as i32, k as i32, TOPK as i32, 0i32, E as i32,
                             ntm as i32)).unwrap();
                    }
                    self.sync_stream();
                    tally.byte(format!("slot gu N={n} K={k} C"),
                              &self.dev.dtoh_sync_copy(&c_chain).unwrap(),
                              &self.dev.dtoh_sync_copy(&c_fused).unwrap());
                    unsafe {
                        st.quant_ng.clone().launch_on_stream(&self.stream,
                            LaunchConfig { grid_dim: (n as u32 * nks as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                            (xp, k as i32, n as i32, nks as i32, 1i32, 1i32,
                             d(&scr_bp), d(&scr_sfb), 0u64, 0i32)).unwrap();
                    }
                    self.sync_stream();
                    let (ref_bp, ref_sfb) = pack_ng_host(&x, k, n, nks, 1, 1);
                    tally.frag(format!("slot gu N={n} K={k} Bp"),
                             &self.dev.dtoh_sync_copy(&scr_bp).unwrap(), &ref_bp);
                    tally.frag(format!("slot gu N={n} K={k} SFB"),
                             &self.dev.dtoh_sync_copy(&scr_sfb).unwrap(), &ref_sfb);

                    // --- dn side (XSILU=1): chain = silu(gu_s) -> quant(h_s) -> slot ---
                    let mut h_s = zero_buf(bk * k);
                    let hp = *h_s.device_ptr() as u64;
                    let c_chain = zero_buf(bk * M);
                    let c_fused = zero_buf(bk * M);
                    let (cp, fp) = (*c_chain.device_ptr() as u64, *c_fused.device_ptr() as u64);
                    blaunch!(self, "moe_silu_bf16_b", grid(bk * k), (256, 1, 1), 0,
                        (hp, gup, k as i32, bk as i32, 0u64, 0i32));
                    unsafe {
                        st.quant_ng.clone().launch_on_stream(&self.stream,
                            LaunchConfig { grid_dim: (bk as u32 * nks as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                            (hp, k as i32, bk as i32, nks as i32, 1i32, 1i32,
                             d(&scr_bp), d(&scr_sfb), 0u64, 0i32)).unwrap();
                        st.moe_slot.clone().launch_on_stream(&self.stream,
                            LaunchConfig { grid_dim: (ntm as u32, bk as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                            (cp, ap, sp, gp, d(&scr_bp), d(&scr_sfb), ip,
                             nks as i32, 0i32, 0i32, E as i32, ntm as i32)).unwrap();
                        st.moe_slot_fused1.clone().launch_on_stream(&self.stream,
                            LaunchConfig { grid_dim: (ntm as u32, bk as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                            (fp, ap, sp, gp, gup, ip, nks as i32, k as i32, 0i32, 0i32, E as i32,
                             ntm as i32)).unwrap();
                    }
                    self.sync_stream();
                    tally.byte(format!("slot dn N={n} K={k} C"),
                              &self.dev.dtoh_sync_copy(&c_chain).unwrap(),
                              &self.dev.dtoh_sync_copy(&c_fused).unwrap());
                    unsafe {
                        st.quant_ng.clone().launch_on_stream(&self.stream,
                            LaunchConfig { grid_dim: (bk as u32 * nks as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                            (hp, k as i32, bk as i32, nks as i32, 1i32, 1i32,
                             d(&scr_bp), d(&scr_sfb), 0u64, 0i32)).unwrap();
                    }
                    self.sync_stream();
                    let h_s_v = self.dev.dtoh_sync_copy(&h_s).unwrap();
                    let (ref_bp, ref_sfb) = pack_ng_host(&h_s_v, k, bk, nks, 1, 1);
                    tally.frag(format!("slot dn N={n} K={k} Bp"),
                             &self.dev.dtoh_sync_copy(&scr_bp).unwrap(), &ref_bp);
                    tally.frag(format!("slot dn N={n} K={k} SFB"),
                             &self.dev.dtoh_sync_copy(&scr_sfb).unwrap(), &ref_sfb);
                }

                // ================= GROUPED arm (XSILU 0 = gu, 1 = dn) =================
                {
                    let p = n * TOPK;
                    let mut ids: Vec<i32> = (0..p).map(|_| (rnd() % E as u32) as i32).collect();
                    for (i, v) in ids.iter_mut().enumerate() {
                        if i % 9 == 4 { *v = -1; }           // remote pairs
                        if i % 13 == 7 { *v = E as i32 + 2; }
                    }
                    // host-side counting-sort metadata (mirrors moe_count/offsets/tilemap):
                    // per-expert counts padded to 16; poff prefix; tile_e per 8-token group.
                    let mut count = vec![0i32; E];
                    for &e in &ids {
                        if e >= 0 && (e as usize) < E { count[e as usize] += 1; }
                    }
                    let mut poff = vec![0i32; E + 1];
                    for e in 0..E {
                        poff[e + 1] = poff[e] + ((count[e] + 15) / 16) * 16;
                    }
                    let ppad = poff[E] as usize;
                    assert!(ppad > 0, "grouped probe: empty expert bands");
                    let ngroups_max = ppad / 16;
                    let mut tile_e = vec![0i32; ppad / 8 + 1];
                    for g in 0..ppad / 8 {
                        let r0 = g * 8;
                        let mut e = 0usize;
                        while e < E && poff[e + 1] <= r0 as i32 { e += 1; }
                        tile_e[g] = e as i32;
                    }
                    // x_perm [ppad][k]: rows < p random, rows >= p exact +0 (the gather's -1 rule)
                    let mut xp_rows: Vec<bf16> = Vec::with_capacity(ppad * k);
                    for r in 0..ppad {
                        if r < p {
                            xp_rows.extend(rnd_row(&mut rnd, k));
                        } else {
                            xp_rows.extend(std::iter::repeat(bf16::from_f32(0.0)).take(k));
                        }
                    }
                    let d_xp = self.dev.htod_sync_copy(&xp_rows).unwrap();
                    let xpp = *d_xp.device_ptr() as u64;
                    let gu_p: Vec<bf16> = {
                        let mut v = Vec::with_capacity(ppad * 2 * k);
                        for _ in 0..ppad { v.extend(rnd_row_wide(&mut rnd, k)); }
                        v
                    };
                    let d_gu_p = self.dev.htod_sync_copy(&gu_p).unwrap();
                    let gupp = *d_gu_p.device_ptr() as u64;
                    let d_poff = self.dev.htod_sync_copy(&poff).unwrap();
                    let d_tile = self.dev.htod_sync_copy(&tile_e).unwrap();
                    let (pop, tp) = (*d_poff.device_ptr() as u64, *d_tile.device_ptr() as u64);
                    let ne = E as i32;

                    // --- gu side (XSILU=0) ---
                    let c_chain = zero_buf(ppad * M);
                    let c_fused = zero_buf(ppad * M);
                    let (cp, fp) = (*c_chain.device_ptr() as u64, *c_fused.device_ptr() as u64);
                    unsafe {
                        st.quant_ng.clone().launch_on_stream(&self.stream,
                            LaunchConfig { grid_dim: ((ppad / 8) as u32 * nks as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                            (xpp, k as i32, (ppad / 8) as i32, nks as i32, 8i32, 8i32,
                             d(&scr_bp), d(&scr_sfb), 0u64, 0i32)).unwrap();
                        st.moe_gemv.clone().launch_on_stream(&self.stream,
                            LaunchConfig { grid_dim: (ntm as u32, ngroups_max as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                            (cp, ap, sp, gp, d(&scr_bp), d(&scr_sfb),
                             ntm as i32, nks as i32, 0i32, tp, pop, ne)).unwrap();
                        st.moe_grouped_fused0.clone().launch_on_stream(&self.stream,
                            LaunchConfig { grid_dim: (ntm as u32, ngroups_max as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                            (fp, ap, sp, gp, xpp, ntm as i32, nks as i32, k as i32, 0i32, tp, pop, ne)).unwrap();
                    }
                    self.sync_stream();
                    tally.byte(format!("grouped gu N={n} K={k} C"),
                              &self.dev.dtoh_sync_copy(&c_chain).unwrap(),
                              &self.dev.dtoh_sync_copy(&c_fused).unwrap());
                    unsafe {
                        st.quant_ng.clone().launch_on_stream(&self.stream,
                            LaunchConfig { grid_dim: ((ppad / 8) as u32 * nks as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                            (xpp, k as i32, (ppad / 8) as i32, nks as i32, 8i32, 8i32,
                             d(&scr_bp), d(&scr_sfb), 0u64, 0i32)).unwrap();
                    }
                    self.sync_stream();
                    let (ref_bp, ref_sfb) = pack_ng_host(&xp_rows, k, ppad / 8, nks, 8, 8);
                    tally.frag(format!("grouped gu N={n} K={k} Bp"),
                             &self.dev.dtoh_sync_copy(&scr_bp).unwrap(), &ref_bp);
                    tally.frag(format!("grouped gu N={n} K={k} SFB"),
                             &self.dev.dtoh_sync_copy(&scr_sfb).unwrap(), &ref_sfb);

                    // --- dn side (XSILU=1): chain = silu(gu_p) -> quant(h_p) -> grouped gemv ---
                    let mut h_p = zero_buf(ppad * k);
                    let hpp = *h_p.device_ptr() as u64;
                    let c_chain = zero_buf(ppad * M);
                    let c_fused = zero_buf(ppad * M);
                    let (cp, fp) = (*c_chain.device_ptr() as u64, *c_fused.device_ptr() as u64);
                    blaunch!(self, "moe_silu_bf16_b", grid(ppad * k), (256, 1, 1), 0,
                        (hpp, gupp, k as i32, ppad as i32, 0u64, 0i32));
                    unsafe {
                        st.quant_ng.clone().launch_on_stream(&self.stream,
                            LaunchConfig { grid_dim: ((ppad / 8) as u32 * nks as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                            (hpp, k as i32, (ppad / 8) as i32, nks as i32, 8i32, 8i32,
                             d(&scr_bp), d(&scr_sfb), 0u64, 0i32)).unwrap();
                        st.moe_gemv.clone().launch_on_stream(&self.stream,
                            LaunchConfig { grid_dim: (ntm as u32, ngroups_max as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                            (cp, ap, sp, gp, d(&scr_bp), d(&scr_sfb),
                             ntm as i32, nks as i32, 0i32, tp, pop, ne)).unwrap();
                        st.moe_grouped_fused1.clone().launch_on_stream(&self.stream,
                            LaunchConfig { grid_dim: (ntm as u32, ngroups_max as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                            (fp, ap, sp, gp, gupp, ntm as i32, nks as i32, k as i32, 0i32, tp, pop, ne)).unwrap();
                    }
                    self.sync_stream();
                    tally.byte(format!("grouped dn N={n} K={k} C"),
                              &self.dev.dtoh_sync_copy(&c_chain).unwrap(),
                              &self.dev.dtoh_sync_copy(&c_fused).unwrap());
                    unsafe {
                        st.quant_ng.clone().launch_on_stream(&self.stream,
                            LaunchConfig { grid_dim: ((ppad / 8) as u32 * nks as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                            (hpp, k as i32, (ppad / 8) as i32, nks as i32, 8i32, 8i32,
                             d(&scr_bp), d(&scr_sfb), 0u64, 0i32)).unwrap();
                    }
                    self.sync_stream();
                    let h_p_v = self.dev.dtoh_sync_copy(&h_p).unwrap();
                    let (ref_bp, ref_sfb) = pack_ng_host(&h_p_v, k, ppad / 8, nks, 8, 8);
                    tally.frag(format!("grouped dn N={n} K={k} Bp"),
                             &self.dev.dtoh_sync_copy(&scr_bp).unwrap(), &ref_bp);
                    tally.frag(format!("grouped dn N={n} K={k} SFB"),
                             &self.dev.dtoh_sync_copy(&scr_sfb).unwrap(), &ref_sfb);
                }
            }

            // ---- grouped early-exit case: poff[ne] < ppad (device-side bound vs static grid) ----
            {
                let n = 4usize;
                let p = n * TOPK;
                let mut ids: Vec<i32> = (0..p).map(|_| (rnd() % E as u32) as i32).collect();
                let mut count = vec![0i32; E];
                for &e in &ids { if e >= 0 && (e as usize) < E { count[e as usize] += 1; } }
                let mut poff = vec![0i32; E + 1];
                for e in 0..E { poff[e + 1] = poff[e] + ((count[e] + 15) / 16) * 16; }
                let ppad = poff[E] as usize;
                let ngroups_max = ppad / 16;
                let mut tile_e = vec![0i32; ppad / 8 + 1];
                for g in 0..ppad / 8 {
                    let r0 = g * 8;
                    let mut e = 0usize;
                    while e < E && poff[e + 1] <= r0 as i32 { e += 1; }
                    tile_e[g] = e as i32;
                }
                let mut xp_rows: Vec<bf16> = Vec::with_capacity(ppad * k);
                for r in 0..ppad {
                    if r < p { xp_rows.extend(rnd_row(&mut rnd, k)); }
                    else { xp_rows.extend(std::iter::repeat(bf16::from_f32(0.0)).take(k)); }
                }
                let d_xp = self.dev.htod_sync_copy(&xp_rows).unwrap();
                let xpp = *d_xp.device_ptr() as u64;
                // Shrink poff[ne] so the top 3 16-token blocks early-exit in BOTH kernels.
                let live = ((ppad / 16).saturating_sub(3).max(1)) * 16;
                let mut poff_ee = poff.clone();
                poff_ee[E] = live as i32;
                let d_poff = self.dev.htod_sync_copy(&poff_ee).unwrap();
                let d_tile = self.dev.htod_sync_copy(&tile_e).unwrap();
                let (pop, tp) = (*d_poff.device_ptr() as u64, *d_tile.device_ptr() as u64);
                let c_chain = zero_buf(ppad * M);
                let c_fused = zero_buf(ppad * M);
                let (cp, fp) = (*c_chain.device_ptr() as u64, *c_fused.device_ptr() as u64);
                let ntm = M / 16;
                unsafe {
                    st.quant_ng.clone().launch_on_stream(&self.stream,
                        LaunchConfig { grid_dim: ((ppad / 8) as u32 * nks as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                        (xpp, k as i32, (ppad / 8) as i32, nks as i32, 8i32, 8i32,
                         d(&scr_bp), d(&scr_sfb), 0u64, 0i32)).unwrap();
                    st.moe_gemv.clone().launch_on_stream(&self.stream,
                        LaunchConfig { grid_dim: (ntm as u32, ngroups_max as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                        (cp, ap, sp, gp, d(&scr_bp), d(&scr_sfb),
                         ntm as i32, nks as i32, 0i32, tp, pop, E as i32)).unwrap();
                    st.moe_grouped_fused0.clone().launch_on_stream(&self.stream,
                        LaunchConfig { grid_dim: (ntm as u32, ngroups_max as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                        (fp, ap, sp, gp, xpp, ntm as i32, nks as i32, k as i32, 0i32, tp, pop, E as i32)).unwrap();
                }
                self.sync_stream();
                tally.byte(format!("grouped early-exit N={n} K={k} C"),
                          &self.dev.dtoh_sync_copy(&c_chain).unwrap(),
                          &self.dev.dtoh_sync_copy(&c_fused).unwrap());
            }
        }

        // ---- edge sweep: dense (N=8, K=4096), slot gu (N=4, K=1536) ----
        {
            let k = 4096usize;
            let nks = k / 64;
            let ntm = M / 16;
            let (d_aimg, d_sfa, d_gs) = upload_weights(&mut rnd, k);
            let (ap, sp, gp) = (*d_aimg.device_ptr() as u64, *d_sfa.device_ptr() as u64,
                                *d_gs.device_ptr() as u64);
            // edge value sets: (label, per-16-block fill value) — each hits a distinct
            // e4m3_ceil branch: all-zero, subnormal scale (amax/6 < 2^-9), near-tie ceil
            // boundaries around 1.0/1.125, and saturation (amax/6 >= 448).
            let edges: Vec<(&str, f32)> = vec![
                ("all-zero", 0.0f32),
                ("subnormal-scale", 6.0f32 * 0.0005),      // amax/6 = 5e-4 < 2^-9 (subnormal e4m3)
                ("near-tie-below-1", 5.999f32),            // amax/6 just below 1.0
                ("near-tie-exact-1", 6.0f32),              // amax/6 == 1.0 exactly
                ("near-tie-above-1", 6.001f32),            // amax/6 just above 1.0
                ("near-tie-1p125", 6.75f32),               // amax/6 == 1.125 (next byte up)
                ("saturation", 3000.0f32),                 // amax/6 >= 448 -> 0x7F
            ];
            for (label, v) in &edges {
                // dense: row 0 = constant v, rows 1..7 random (N=8)
                let mut x: Vec<bf16> = Vec::with_capacity(8 * k);
                x.extend(std::iter::repeat(bf16::from_f32(*v)).take(k));
                for _ in 1..8 { x.extend(rnd_row(&mut rnd, k)); }
                let d_x = self.dev.htod_sync_copy(&x).unwrap();
                let xp = *d_x.device_ptr() as u64;
                let c_chain = zero_buf(8 * M);
                let c_fused = zero_buf(8 * M);
                let (cp, fp) = (*c_chain.device_ptr() as u64, *c_fused.device_ptr() as u64);
                unsafe {
                    st.quant.clone().launch_on_stream(&self.stream,
                        LaunchConfig { grid_dim: (nks as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                        (xp, k as i32, 8i32, d(&st.bp), d(&st.sfb))).unwrap();
                    st.gemv.clone().launch_on_stream(&self.stream,
                        LaunchConfig { grid_dim: (ntm as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                        (cp, ap, sp, gp, d(&st.bp), d(&st.sfb),
                         ntm as i32, nks as i32, M as i32, 8i32)).unwrap();
                    st.gemv_fused8.clone().launch_on_stream(&self.stream,
                        LaunchConfig { grid_dim: (ntm as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                        (fp, ap, sp, gp, xp, ntm as i32, nks as i32, M as i32, 8i32)).unwrap();
                }
                self.sync_stream();
                tally.byte(format!("edge dense {label} C"),
                          &self.dev.dtoh_sync_copy(&c_chain).unwrap(),
                          &self.dev.dtoh_sync_copy(&c_fused).unwrap());
                unsafe {
                    st.quant.clone().launch_on_stream(&self.stream,
                        LaunchConfig { grid_dim: (nks as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                        (xp, k as i32, 8i32, d(&st.bp), d(&st.sfb))).unwrap();
                }
                self.sync_stream();
                let (ref_bp, ref_sfb) = pack_ng_host(&x, k, 1, nks, 8, 1);
                tally.frag(format!("edge dense {label} Bp"),
                         &self.dev.dtoh_sync_copy(&st.bp).unwrap(), &ref_bp);
                tally.frag(format!("edge dense {label} SFB"),
                         &self.dev.dtoh_sync_copy(&st.sfb).unwrap(), &ref_sfb);
            }
            // slot gu edge (N=4, K=1536): token 0's row = constant v — exercises the N=1
            // single-row path's scale with each branch.
            {
                let k = 1536usize;
                let nks = k / 64;
                let ntm = M / 16;
                let (d_aimg, d_sfa, d_gs) = upload_weights(&mut rnd, k);
                let (ap, sp, gp) = (*d_aimg.device_ptr() as u64, *d_sfa.device_ptr() as u64,
                                    *d_gs.device_ptr() as u64);
                let n = 4usize;
                let bk = n * TOPK;
                let mut ids: Vec<i32> = (0..bk).map(|_| (rnd() % E as u32) as i32).collect();
                let d_ids = self.dev.htod_sync_copy(&ids).unwrap();
                let ip = *d_ids.device_ptr() as u64;
                for (label, v) in &edges {
                    let mut x: Vec<bf16> = Vec::with_capacity(n * k);
                    x.extend(std::iter::repeat(bf16::from_f32(*v)).take(k));
                    for _ in 1..n { x.extend(rnd_row(&mut rnd, k)); }
                    let d_x = self.dev.htod_sync_copy(&x).unwrap();
                    let xp = *d_x.device_ptr() as u64;
                    let c_chain = zero_buf(bk * M);
                    let c_fused = zero_buf(bk * M);
                    let (cp, fp) = (*c_chain.device_ptr() as u64, *c_fused.device_ptr() as u64);
                    unsafe {
                        st.quant_ng.clone().launch_on_stream(&self.stream,
                            LaunchConfig { grid_dim: (n as u32 * nks as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                            (xp, k as i32, n as i32, nks as i32, 1i32, 1i32,
                             d(&scr_bp), d(&scr_sfb), 0u64, 0i32)).unwrap();
                        st.moe_slot.clone().launch_on_stream(&self.stream,
                            LaunchConfig { grid_dim: (ntm as u32, bk as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                            (cp, ap, sp, gp, d(&scr_bp), d(&scr_sfb), ip,
                             nks as i32, TOPK as i32, 0i32, E as i32, ntm as i32)).unwrap();
                        st.moe_slot_fused0.clone().launch_on_stream(&self.stream,
                            LaunchConfig { grid_dim: (ntm as u32, bk as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                            (fp, ap, sp, gp, xp, ip, nks as i32, k as i32, TOPK as i32, 0i32, E as i32,
                             ntm as i32)).unwrap();
                    }
                    self.sync_stream();
                    tally.byte(format!("edge slot-gu {label} C"),
                              &self.dev.dtoh_sync_copy(&c_chain).unwrap(),
                              &self.dev.dtoh_sync_copy(&c_fused).unwrap());
                }
            }
        }

        println!("  checks: {}, mismatches: {}", tally.n_checks, tally.n_fail);
        for f in &tally.fail_notes { println!("    MISMATCH: {f}"); }
        if tally.n_fail == 0 {
            println!("FUSED_BIT_IDENTITY OK");
        } else {
            println!("FUSED_BIT_IDENTITY FAILED ({} mismatches)", tally.n_fail);
        }
    }
}

impl GpuModel {
    /// Phase A (quality-fix session) — CROSS-CHAIN quality instrument. Runs the SAME real prompt
    /// through the native (OMMA) chain and the bf16 chain in ONE process (both layouts resident;
    /// the allowlist is flipped between passes), capturing per-layer pre-GEMM normed inputs +
    /// residuals + final logits at every decode step. Reports:
    ///   (1) the per-layer hidden rel-L2 curve (compounding: smooth ~sqrt(layer) growth = e2m1
    ///       codebook floor; a layer jumping 10x its neighbors = bug site);
    ///   (2) the per-GEMM-input activation-quant ROUNDTRIP error on REAL hiddens (host mirror of
    ///       the exact device recipe: e4m3_ceil(amax/6) + cvt.rn.satfinite.e2m1x2);
    ///   (3) final logit rel-L2 + top-1 agreement;
    ///   (4) the two token streams (returned for text decoding in main).
    pub fn probe_mxfp4_xchain(&self, pool: &mut Pool, prompt: &[u32], max_new: usize,
                              max_seq_len: usize) -> (Vec<u32>, Vec<u32>) {
        let st = match &self.mxfp4 {
            Some(s) => s,
            None => { println!("probe-mxfp4-xchain: --mxfp4=on required (the native chain needs the OMMA state)"); std::process::exit(1); }
        };
        XCHAIN_UNBOUNDED.store(true, std::sync::atomic::Ordering::Relaxed);
        if crate::opts::var(crate::opt!("mxfp4-economy")).is_ok() {
            println!("probe-mxfp4-xchain: --mxfp4-economy is set — the bf16 chain's standard layout is NOT resident; refusing (27B/0.8B are non-economy)");
            std::process::exit(1);
        }
        let h = self.cfg.hidden_size;
        let plen = prompt.len();
        let kv_stride = max_seq_len;
        crate::opts::set(crate::opt!("mxfp4-xchain-capture"), "1");   // arm the forward hooks
        println!("=== probe-mxfp4-xchain: native (OMMA) vs bf16 chain, same real prompt ===");
        println!("  prompt {} tokens, decode {} greedy steps, {} layers, h={}", plen, max_new, self.layers.len(), h);

        let mut state = self.new_batch_state(2, 2, max_seq_len);
        let mut bufs = self.new_decode_buffers(2);
        self.dev.htod_sync_copy_into(&[0i32, 0], &mut bufs.slot_ids_dev).unwrap();
        self.dev.synchronize().unwrap();

        let mut run_pass = |gpu: &GpuModel, allow_all: bool| -> (Vec<u32>, Vec<XChainCapture>, Vec<f32>) {
            if allow_all { let n = st.allow_all_bf16(); println!("  [pass 2] allowlisted {n} tensors -> bf16 chain"); }
            xchain_capture_reset();
            gpu.zero_slot_state(&mut state, 0, kv_stride);
            let (a0, hout) = gpu.prefill_batch(pool, prompt, &mut state, 0, kv_stride, 0);
            println!("  [prefill] first token = {a0}");
            // Chain binv spot-check: verify N=2 col-0 vs the first decode step at the SAME
            // position — the MTP losslessness contract within the chain. The verify advances the
            // GDN state past position plen, so the slot is zeroed + re-prefilled afterwards
            // (cheap: one extra prefill per pass, probe only).
            let vt = [a0, if (a0 as usize) + 1 < gpu.cfg.vocab_size { a0 + 1 } else { 0 }];
            let (_, vout) = gpu.verify_forward(pool, &vt, &mut state, 0, kv_stride, plen, None, None);
            pool.release_bf16(vout, h * 2);
            gpu.zero_slot_state(&mut state, 0, kv_stride);
            let (a0b, hout2) = gpu.prefill_batch(pool, prompt, &mut state, 0, kv_stride, 0);
            assert_eq!(a0, a0b, "re-prefill after the verify spot-check diverged");
            pool.release_bf16(hout2, h * plen);
            // Cross-chain probe: keep ONE column of the prefill hiddens (token plen/2) to compare
            // passes directly — a prefill divergence would explain a step-1 divergence.
            let hcol: Vec<half::bf16> = gpu.dev.dtoh_sync_copy(&hout)
                .unwrap()[h * (plen / 2)..h * (plen / 2 + 1)].to_vec();
            pool.release_bf16(hout, h * plen);
            let hcol_f: Vec<f32> = hcol.iter().map(|b| half::bf16::to_f32(*b)).collect();
            let mut tokens = vec![a0];
            let mut spos = plen;
            for _ in 0..max_new {
                let tok = *tokens.last().unwrap() as i32;
                gpu.dev.htod_sync_copy_into(&vec![tok, 0], &mut bufs.tokens_dev).unwrap();
                gpu.dev.htod_sync_copy_into(&vec![spos as i32, 0], &mut bufs.pos_dev).unwrap();
                gpu.dev.synchronize().unwrap();
                let next = gpu.forward_decode(pool, &mut bufs, &mut state, kv_stride, spos + 1, 1);
                tokens.push(next[0]);
                spos += 1;
            }
            (tokens, xchain_capture_take(), hcol_f)
        };

        println!("  ---- pass 1: NATIVE chain ----");
        let (tok_nat, caps_nat, hcol_nat) = run_pass(self, false);
        let steps_nat: Vec<&XChainCapture> = caps_nat.iter().filter(|c| c.batch == 1).collect();
        println!("  ---- pass 2: BF16 chain (allowlist = all) ----");
        let (tok_bf16, caps_bf16, hcol_bf16) = run_pass(self, true);
        st.clear_allow_all_bf16();
        self.sync_stream();   // all capture copies are stream-ordered; sync before any read
        let steps_bf16: Vec<&XChainCapture> = caps_bf16.iter().filter(|c| c.batch == 1).collect();
        assert_eq!(steps_nat.len(), max_new, "native capture count");
        assert_eq!(steps_bf16.len(), max_new, "bf16 capture count");
        let nlayers = self.layers.len();
        let rel = |a: &[f32], b: &[f32]| -> f64 {
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for i in 0..a.len() {
                let d = a[i] as f64 - b[i] as f64;
                num += d * d; den += b[i] as f64 * b[i] as f64;
            }
            (num / den.max(1e-30)).sqrt()
        };
        let pfill_rel = rel(&hcol_nat, &hcol_bf16);
        println!("  [prefill] hidden col {}/{} rel-L2 native vs bf16: {:.4}", plen / 2, plen, pfill_rel);
        // Chain binv spot-check: verify N=2 col-0 vs the first decode step, per layer (the MTP
        // losslessness contract within ONE chain). Any layer > ~1e-6 is a binv break in that
        // chain — the MoE native grouped (N=2) vs slot (N=1) kernels are the prime suspect.
        for (tag, caps, steps) in [("native", &caps_nat, &steps_nat), ("bf16", &caps_bf16, &steps_bf16)] {
            if let Some(v) = caps.iter().find(|c| c.batch == 2) {
                let mut worst = 0.0f64;
                let mut worst_li = 0usize;
                let mut first_bad = nlayers;
                for li in 0..nlayers.min(v.layer_inputs.len()) {
                    let r = rel(&v.layer_inputs[li][..h], &steps[0].layer_inputs[li][..h]);
                    if r > worst { worst = r; worst_li = li; }
                    if r > 1e-6 && first_bad == nlayers { first_bad = li; }
                }
                println!("  [verify-vs-decode {tag}] worst col-0 rel-L2 {worst:.2e} at layer {worst_li} \
                         (bar 1e-6; > 0.1 = binv break), first layer > 1e-6: {first_bad}");
                if tag == "native" {
                    for li in (0..nlayers.min(v.layer_inputs.len())).step_by(2) {
                        let r = rel(&v.layer_inputs[li][..h], &steps[0].layer_inputs[li][..h]);
                        println!("    layer {li:>2}: {r:.3e}");
                    }
                }
            }
        }

        let rel = |a: &[f32], b: &[f32]| -> f64 {
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for i in 0..a.len() {
                let d = a[i] as f64 - b[i] as f64;
                num += d * d; den += b[i] as f64 * b[i] as f64;
            }
            (num / den.max(1e-30)).sqrt()
        };
        let argmax = |l: &[f32]| -> u32 {
            let mut b = 0usize;
            for (i, v) in l.iter().enumerate() { if *v > l[b] { b = i; } }
            b as u32
        };

        // (1) per-layer rel-L2 curve at step 1 and the LAST step.
        println!("\n  per-layer hidden rel-L2 (native vs bf16), layer 0 = first layer:");
        println!("  {:>4}  {:>10}  {:>10}    {:>12}", "layer", "step1", "stepLast", "growth");
        let s1_n = &steps_nat[0].layer_outputs;
        let s1_b = &steps_bf16[0].layer_outputs;
        let sl_n = &steps_nat[max_new - 1].layer_outputs;
        let sl_b = &steps_bf16[max_new - 1].layer_outputs;
        for li in 0..nlayers {
            let r1 = rel(&s1_n[li], &s1_b[li]);
            let rl = rel(&sl_n[li], &sl_b[li]);
            if li % 4 == 0 || li + 1 == nlayers {
                println!("  {:>4}  {:>10.4}  {:>10.4}    {:>12.1}", li, r1, rl, rl / r1.max(1e-9));
            }
        }
        let r1_last = rel(&s1_n[nlayers - 1], &s1_b[nlayers - 1]);
        let rl_last = rel(&sl_n[nlayers - 1], &sl_b[nlayers - 1]);
        println!("  LAST layer: step1 rel-L2 {:.4}   step{} rel-L2 {:.4}", r1_last, max_new, rl_last);

        // (2) per-GEMM-input activation-quant ROUNDTRIP error on REAL hiddens (native chain's
        // inputs; host mirror of the exact device recipe).
        println!("\n  per-GEMM-input activation-quant roundtrip (REAL hiddens, native step 1):");
        println!("  {:>4}  {:>10}  {:>12}  {:>10}  {:>12}", "layer", "rel-L2", "max-abs", "rms-in", "amax");
        let mut qrel_worst = 0.0f64;
        for li in [0usize, 1, 3, 7, 15, 31, 47, nlayers - 1] {
            if li >= nlayers { continue; }
            let x = &s1_n[li];
            let mut num = 0.0f64; let mut den = 0.0f64; let mut mx = 0.0f32;
            let mut rms = 0.0f64; let mut amax = 0.0f32;
            for b in x.chunks(16) {
                let am = b.iter().fold(0.0f32, |a, v| a.max(v.abs()));
                amax = amax.max(am);
                let sc = crate::mxfp4::e4m3_ceil(am / 6.0);
                let sv = crate::mxfp4::ue4m3_f(sc);
                let inv = if sv == 0.0 { 0.0 } else { 1.0 / sv };
                for v in b.iter() {
                    let c = crate::mxfp4::e2m1_rn(*v * inv);
                    let q = crate::mxfp4::e2m1_f(c) * sv;
                    let d = *v as f64 - q as f64;
                    num += d * d; den += *v as f64 * *v as f64;
                    mx = mx.max((*v - q).abs());
                    rms += *v as f64 * *v as f64;
                }
            }
            let qrel = (num / den.max(1e-30)).sqrt();
            qrel_worst = qrel_worst.max(qrel);
            println!("  {:>4}  {:>10.4}  {:>12.4}  {:>10.4}  {:>12.3}", li, qrel, mx, (rms / x.len() as f64).sqrt(), amax);
        }
        println!("  worst roundtrip rel-L2 over sampled layers: {:.4}   (uniform-input estimate was ~0.1113)", qrel_worst);

        // (3) final logits: rel-L2 + top-1 agreement (last step).
        let ln = &steps_nat[max_new - 1].logits;
        let lb = &steps_bf16[max_new - 1].logits;
        let lrel = rel(ln, lb);
        let (an, ab) = (argmax(ln), argmax(lb));
        println!("\n  final logits (step {max_new}): rel-L2 {:.4}   top-1 native {} bf16 {}   agree {}",
                 lrel, an, ab, an == ab);
        // top-10 agreement
        let top10 = |l: &[f32]| -> Vec<u32> {
            let mut idx: Vec<u32> = (0..l.len() as u32).collect();
            idx.sort_by(|&a, &b| l[b as usize].partial_cmp(&l[a as usize]).unwrap());
            idx.truncate(10);
            idx
        };
        let (t10n, t10b) = (top10(ln), top10(lb));
        println!("  top-10 agreement: {} (native {:?} vs bf16 {:?})",
                 t10n.iter().zip(&t10b).filter(|(a, b)| a == b).count(), &t10n[..5], &t10b[..5]);

        // (4) token streams.
        let div = tok_nat.iter().zip(&tok_bf16).position(|(a, b)| a != b);
        println!("\n  token stream: native {:?}", &tok_nat[..tok_nat.len().min(24)]);
        println!("  token stream: bf16   {:?}", &tok_bf16[..tok_bf16.len().min(24)]);
        match div {
            Some(d) => println!("  first token divergence at step {d} (native {} vs bf16 {})",
                                tok_nat[d], tok_bf16[d]),
            None => println!("  token streams IDENTICAL over {} tokens", max_new),
        }
        (tok_nat, tok_bf16)
    }

    /// One-load cross-chain dump (--probe-mxfp4-xchain2): ONE pass of the xchain machinery,
    /// written to disk. Run twice — once with `--mxfp4=on` (native OMMA chain; economy mode is
    /// fine — no second chain is needed in-process) and once with `--mxfp4=off` (standard
    /// chain — the reference) — then compare the dumps with `--probe-mxfp4-xchain2-compare`.
    /// This is the economy-mode escape hatch the in-process probe refuses: hy3 runs economy, so
    /// its cross-chain check must span two loads. The dump keeps only what the comparison needs
    /// (prefill col, step-1/last per-layer outputs, final logits, tokens) — ~5 MB for a 295B.
    pub fn probe_mxfp4_xchain2_dump(&self, pool: &mut Pool, prompt: &[u32], max_new: usize,
                                    max_seq_len: usize) -> XChain2Dump {
        let h = self.cfg.hidden_size;
        let plen = prompt.len();
        let kv_stride = max_seq_len;
        crate::opts::set(crate::opt!("mxfp4-xchain-capture"), "1");
        let mut state = self.new_batch_state(2, 2, max_seq_len);
        let mut bufs = self.new_decode_buffers(2);
        self.dev.htod_sync_copy_into(&[0i32, 0], &mut bufs.slot_ids_dev).unwrap();
        self.dev.synchronize().unwrap();
        xchain_capture_reset();
        self.zero_slot_state(&mut state, 0, kv_stride);
        let (a0, hout) = self.prefill_batch(pool, prompt, &mut state, 0, kv_stride, 0);
        let hcol: Vec<half::bf16> = self.dev.dtoh_sync_copy(&hout).unwrap()
            [h * (plen / 2)..h * (plen / 2 + 1)].to_vec();
        pool.release_bf16(hout, h * plen);
        let hcol: Vec<f32> = hcol.iter().map(|b| half::bf16::to_f32(*b)).collect();
        let mut tokens = vec![a0];
        let mut spos = plen;
        for _ in 0..max_new {
            let tok = *tokens.last().unwrap() as i32;
            self.dev.htod_sync_copy_into(&vec![tok, 0], &mut bufs.tokens_dev).unwrap();
            self.dev.htod_sync_copy_into(&vec![spos as i32, 0], &mut bufs.pos_dev).unwrap();
            self.dev.synchronize().unwrap();
            let next = self.forward_decode(pool, &mut bufs, &mut state, kv_stride, spos + 1, 1);
            tokens.push(next[0]);
            spos += 1;
        }
        self.sync_stream();
        let caps = xchain_capture_take();
        let steps: Vec<&XChainCapture> = caps.iter().filter(|c| c.batch == 1).collect();
        assert_eq!(steps.len(), max_new, "xchain2 capture count != max_new");
        XChain2Dump {
            plen, max_new, nlayers: self.layers.len(), h, vocab: self.cfg.vocab_size,
            a0, hcol, tokens,
            step1_outputs: steps[0].layer_outputs.clone(),
            step_last_outputs: steps[max_new - 1].layer_outputs.clone(),
            final_logits: steps[max_new - 1].logits.clone(),
        }
    }
}
