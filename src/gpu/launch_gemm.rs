//! `impl GpuModel`: kernel launch plumbing (E9 / PDL raw launches, `launch_gemm_fp4*`, `launch_gemm_mxfp4`) and `gemm_act*`,
//! the activation-GEMM dispatcher. Batch-invariant: do not change reduction orders (gate: `--probe-binv`). Map: `mod.rs`.

use super::*;

impl GpuModel {
    /// E9 raw handle for `name`, or None when the programmatic path is off: single-node, the
    /// --e9-no-fold escape, or the E9 module failed to load (graceful fallback to `blaunch!`).
    pub(super) fn e9_handle(&self, name: &str) -> Option<cudarc::driver::sys::CUfunction> {
        if self.tp_world != 2 || !e9_fold_enabled() { return None; }
        let ks = E9_KERNELS.get_or_init(e9_load_kernels);
        ks.as_ref().and_then(|k| k.get(name))
    }

    /// E9: launch `f` on the compute stream with the programmatic stream-serialization attribute —
    /// the grid may start its preamble while the PREVIOUS kernel still runs, and MUST call
    /// `griddepcontrol.wait` before consuming the previous kernel's output (every E9-marked kernel
    /// does, in its prologue). `params` is the `as_kernel_param()` array, same convention as
    /// `LaunchAsync` (temporaries live through the synchronous driver marshaling).
    pub(super) unsafe fn launch_pss(&self, f: cudarc::driver::sys::CUfunction,
                         grid: (u32, u32, u32), block: (u32, u32, u32), smem: u32,
                         params: &mut [*mut std::ffi::c_void]) {        use cudarc::driver::sys::*;
        let mut attr = CUlaunchAttribute {
            id: CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
            pad: [0; 4],
            value: CUlaunchAttributeValue { programmaticStreamSerializationAllowed: 1 },
        };
        let mut cfg = CUlaunchConfig {
            gridDimX: grid.0, gridDimY: grid.1, gridDimZ: grid.2,
            blockDimX: block.0, blockDimY: block.1, blockDimZ: block.2,
            sharedMemBytes: smem,
            hStream: self.stream.stream,
            attrs: &mut attr,
            numAttrs: 1,
        };
        let r = cuLaunchKernelEx(&mut cfg, f, params.as_mut_ptr(), std::ptr::null_mut());
        assert!(r == CUresult::CUDA_SUCCESS, "E9 PSS launch failed: {r:?}");
    }

    /// P4 B3: plain raw-handle launch (no launch attributes) for kernels whose dynamic smem
    /// needs the opt-in (gdn_chunk_prefill_b). Same convention as launch_pss minus PSS.
    pub(super) unsafe fn launch_plain_raw(&self, f: cudarc::driver::sys::CUfunction,
                               grid: (u32, u32, u32), block: (u32, u32, u32), smem: u32,
                               params: &mut [*mut std::ffi::c_void]) {
        use cudarc::driver::sys;
        let mut attr = sys::CUlaunchAttribute {
            id: sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_IGNORE,
            pad: [0; 4],
            value: sys::CUlaunchAttributeValue_union { pad: [0; 64] },
        };
        let mut cfg = sys::CUlaunchConfig {
            gridDimX: grid.0, gridDimY: grid.1, gridDimZ: grid.2,
            blockDimX: block.0, blockDimY: block.1, blockDimZ: block.2,
            sharedMemBytes: smem,
            hStream: self.stream.stream,
            attrs: &mut attr,
            numAttrs: 0,
        };
        let r = sys::cuLaunchKernelEx(&mut cfg, f, params.as_mut_ptr(), std::ptr::null_mut());
        assert!(r == sys::CUresult::CUDA_SUCCESS, "launch_plain_raw failed: {r:?}");
    }

    /// E11: raw launch of a batch-module kernel by name with an explicit param array (for kernels
    /// whose arg count exceeds the `launch_on_stream` tuple cap or that carry an extra runtime
    /// flag — the fold-pdl kernel's 13-arg launches). Same launch config convention as `blaunch!`.
    pub(super) unsafe fn launch_raw(&self, name: &str, grid: (u32, u32, u32), block: (u32, u32, u32),
                         smem: u32, params: &mut [*mut std::ffi::c_void]) {
        use cudarc::driver::sys;
        let f = e11_raw_kernels().and_then(|k| k.by_name.get(name).copied())
            .unwrap_or_else(|| panic!("e11 raw kernel {name} unavailable"));
        let mut attr = sys::CUlaunchAttribute {
            id: sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_IGNORE,
            pad: [0; 4],
            value: sys::CUlaunchAttributeValue_union { pad: [0; 64] },
        };
        let mut cfg = sys::CUlaunchConfig {
            gridDimX: grid.0, gridDimY: grid.1, gridDimZ: grid.2,
            blockDimX: block.0, blockDimY: block.1, blockDimZ: block.2,
            sharedMemBytes: smem,
            hStream: self.stream.stream,
            attrs: &mut attr,
            numAttrs: 0,
        };
        let r = sys::cuLaunchKernelEx(&mut cfg, f, params.as_mut_ptr(), std::ptr::null_mut());
        assert!(r == sys::CUresult::CUDA_SUCCESS, "launch_raw {name} failed: {r:?}");
    }

    /// The one launch site for the NVFP4 mma serving GEMM: the plain persistent-grid kernel, or —
    /// when `gemm_fp4_nsplit` says the shape pays (long K, few tiles — the FFN down-proj) — the
    /// split-K partial kernel + the fixed-order reduce over its fp32 partials. `c` and `cf` are the
    /// bf16 output and the optional FP32-preserving TP=2 partial; exactly one is nonzero (see
    /// mma_epilogue). The split/reduce pair is back-to-back on this stream, so the shared partials
    /// scratch is never aliased. Graph-safe: no alloc, no sync, fixed buffer addresses.
    pub(super) fn launch_gemm_fp4(&self, c: u64, qw: u64, sc: u64, gs: u64, x: u64,
                       outn: usize, inn: usize, batch: usize, cf: u64) {
        self.launch_gemm_fp4_s(c, qw, sc, gs, x, outn, inn, batch, cf, gemm_fp4_nsplit(outn, inn), false);
    }

    /// E9: `launch_gemm_fp4` as the programmatic-stream-serialization SECONDARY of the TP barrier
    /// chain (the qkv/GDN projection site) — `pdl=1` streams the weights before griddepcontrol.wait.
    /// Falls back to the plain launch when E9 is off (single-node / escape / load failure).
    fn launch_gemm_fp4_pdl(&self, c: u64, qw: u64, sc: u64, gs: u64, x: u64,
                           outn: usize, inn: usize, batch: usize, cf: u64) {
        self.launch_gemm_fp4_s(c, qw, sc, gs, x, outn, inn, batch, cf, gemm_fp4_nsplit(outn, inn), true);
    }

    /// `launch_gemm_fp4` with an EXPLICIT split count (probe/diagnostic use: the S sweep of
    /// `--probe-splitk` must time the same dispatch geometry production would run).
    pub(super) fn launch_gemm_fp4_s(&self, c: u64, qw: u64, sc: u64, gs: u64, x: u64,
                         outn: usize, inn: usize, batch: usize, cf: u64, ns: usize, pdl: bool) {
        let ntm = outn / 16;
        if ns > 1 {
            assert!(!pdl,
                "E9 PDL launch on a split-K shape (outn={outn} inn={inn} ns={ns}) — the qkv/GDN \
                 consumer never splits (K < 8192); a shape that does needs the E9 preamble in \
                 gemm_mma_splitk_reduce_b before it can be a programmatic secondary");
            let p = d(&self.splitk_partials);
            let grid = (ntm * ns).min(GB10_SMS * MMA_FP4_BLOCKS_PER_SM) as u32;
            blaunch!(self, "gemm_mma_fp4_splitk_b", (grid,1,1), (256,1,1), 0,
                (p, qw, sc, x, outn as i32, inn as i32, batch as i32, ns as i32));
            blaunch!(self, "gemm_mma_splitk_reduce_b", (ntm as u32,1,1), (256,1,1), 0,
                (c, p, gs, outn as i32, batch as i32, ns as i32, cf));
            return;
        }
        // Persistent grid: one block per resident slot, each looping over 16-row tiles.
        // The tile->block map depends on the weight shape only, never on N (probe-binv).
        let grid = ntm.min(GB10_SMS * MMA_FP4_BLOCKS_PER_SM) as u32;
        if pdl {
            // E9: programmatic launch — the kernel's `pdl=1` prologue streams the block's weight
            // bytes into L2, then griddepcontrol.wait gates the (barrier-reduced) activation.
            if let Some(f) = self.e9_handle("gemm_mma_fp4_b") {
                // Named locals for every expression as_kernel_param() points into — the driver
                // marshals the params during the call, so the pointees must outlive the statement.
                let (mo, ki, ba, pdl_one) = (outn as i32, inn as i32, batch as i32, 1i32);
                let mut params = [c.as_kernel_param(), qw.as_kernel_param(), sc.as_kernel_param(),
                                  gs.as_kernel_param(), x.as_kernel_param(), mo.as_kernel_param(),
                                  ki.as_kernel_param(), ba.as_kernel_param(),
                                  cf.as_kernel_param(), pdl_one.as_kernel_param()];
                unsafe { self.launch_pss(f, (grid, 1, 1), (256, 1, 1), 0, &mut params); }
                return;
            }
            // No handle (single-node / escape / load failure) — fall through to the plain launch
            // with pdl=0, which is exactly the pre-E9 kernel.
        }
        blaunch!(self, "gemm_mma_fp4_b", (grid,1,1), (256,1,1), 0,
            (c, qw, sc, gs, x, outn as i32, inn as i32, batch as i32, cf, 0i32));
    }

    /// The MXFP4-native GEMM (`--mxfp4=on`): activation quant + the sm_121a OMMA GEMV.
    /// X is quantized once per (tensor, token batch) to E2M1 nibbles + UE4M3 per-16-block
    /// scales (per-token, per-kstep), then every 16-row weight tile consumes the SAME packed
    /// fragments — the fixed instruction sequence the batch-invariance contract requires.
    /// `st` carries the lossless weight repacks (keyed by qweight ptr) + kernel handles.
    fn launch_gemm_mxfp4(&self, c: u64, qw: &cudarc::driver::CudaSlice<u8>,
                         sc: &cudarc::driver::CudaSlice<u8>, gs: &S, x: u64,
                         outn: usize, inn: usize, batch: usize, st: &crate::mxfp4::Mxfp4State) {
        let ptr = *qw.device_ptr() as u64;
        let (aimg, sfa) = st.w.get(&ptr).unwrap_or_else(|| {
            panic!("mxfp4 mode: tensor @{ptr:x} has no OMMA repack — the loader built the repacks \
                    only for tensors it assembled; a repack gap is a loader bug, refusing to run mixed")
        }).ptrs();
        let (ntm, nks) = (outn / 16, inn / 64);
        assert!(inn <= crate::mxfp4::MXFP4_MAX_K,
                "mxfp4: K={inn} exceeds MXFP4_MAX_K={} (scratch sized at load)",
                crate::mxfp4::MXFP4_MAX_K);
        assert!(batch <= crate::mxfp4::MXFP4_GROUPS * 8,
                "mxfp4: batch={batch} exceeds the {} 8-token groups the scratch allows",
                crate::mxfp4::MXFP4_GROUPS);
        if Self::mxfp4_fused_on() {
            // FUSED (F1 — EXPERT_FUSED_QUANT_RESPONSE.md §4.2): ONE kernel — each warp
            // quantizes its own 64-K kstep window in-block and feeds the same OMMA; byte-
            // identical C to the quant+gemv pair below (the §7 construction), grid/block/
            // k-visit order unchanged. NR by batch: 1 / 8 / 16 (9..=16 needs the second
            // 8-token group, hence the separate NR=16 instantiation).
            let (f, nr_label) = if batch <= 1 {
                (&st.gemv_fused1, "1")
            } else if batch <= 8 {
                (&st.gemv_fused8, "8")
            } else {
                (&st.gemv_fused16, "16")
            };
            let grid = (ntm.min(GB10_SMS * MMA_FP4_BLOCKS_PER_SM)) as u32;
            unsafe {
                f.clone().launch_on_stream(
                    &self.stream,
                    LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                    (c, aimg, sfa, d(gs), x,
                     ntm as i32, nks as i32, outn as i32, batch as i32))
                    .unwrap_or_else(|e| panic!("mxfp4_gemv_native_fused_b{nr_label} launch: {e:?}"));
            }
            return;
        }
        // 1) Quantize the activation batch: X [batch, inn] -> Bp/SFB (one 8-token group).
        unsafe {
            st.quant.clone().launch_on_stream(
                &self.stream,
                LaunchConfig { grid_dim: (nks as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                (x, inn as i32, batch as i32, d(&st.bp), d(&st.sfb)))
                .unwrap_or_else(|e| panic!("mxfp4_quant_pack_b launch: {e:?}"));
        }
        // 2) The persistent-grid OMMA GEMV (same schedule family as gemm_mma_fp4_b).
        let grid = (ntm.min(GB10_SMS * MMA_FP4_BLOCKS_PER_SM)) as u32;
        unsafe {
            st.gemv.clone().launch_on_stream(
                &self.stream,
                LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                (c, aimg, sfa, d(gs), d(&st.bp), d(&st.sfb),
                 ntm as i32, nks as i32, outn as i32, batch as i32))
                .unwrap_or_else(|e| panic!("mxfp4_gemv_native_b launch: {e:?}"));
        }
    }

    /// The single dispatcher every big matmul funnels through.
    ///
    /// For QUANTIZED weights every decode width (1..=MAX_VERIFY) runs the SAME fixed-shape tensor-core
    /// kernel with N padded to 16: it is bound by the packed-weight bytes, which do not change with N,
    /// so verify costs what decode costs and speculation finally pays. Above 16 — i.e. prefill —
    /// weights are read once per many tokens, bandwidth stops being the constraint, and it is cheaper
    /// to dequantize once into scratch and let cuBLAS's tensor cores do the work.
    pub(super) fn gemm_act(&self, w: &W, x: &B, out: &mut B, inn: usize, outn: usize, batch: usize) {
        // BF16 batch-invariance boundary: gemm_binv_b at EVERY verify width. The old cutoff
        // (batch <= 2) silently fell to cuBLAS at widths 3..MAX_VERIFY — cuBLAS picks a different
        // kernel per (M,N,K), so col-0 of a wide verify diverged from decode (probe-binv: bf16
        // lm_head first divergent N = 3, xor 0x8070). Latent until now because every resident
        // artifact quantizes its head; Qwen official FP8 keeps lm_head + in_proj_b/a bf16, so
        // MTP verify ran non-invariant logits. Above MAX_VERIFY (prefill) invariance is not
        // contractual and cuBLAS wins — that boundary is unchanged.
        const BINV_BATCH: usize = MAX_VERIFY;
        // P10 batch-invariance fix pinned: the binv kernel owns EVERY verify width, not
        // just N<=2. Const-asserted so a future merge cannot silently revert it (TRAP 5).
        const _: () = assert!(BINV_BATCH == MAX_VERIFY, "BINV_BATCH must stay MAX_VERIFY (P10 binv fix)");
        // MR-GPTQ: a rotated weight reads the micro-rotated activation (scratch, x untouched).
        let _rot_guard; let x: &B = if self.is_rotated(w) { _rot_guard = self.rot_x(x, inn, batch); &*_rot_guard } else { x };

        // Explicit EXPERIMENTAL narrow W4A4 fork. --w4a4-verify selects groups for both N=1
        // and N<=MAX_VERIFY, but the full lossless gate still fails (other batched operations
        // amplify A4 rounding). Production leaves it unset. --w4a4-lmhead-narrow remains the
        // historical head-only A/B.
        if let (Some(w4), W::Nvfp4 { qweight, m, k, .. }) = (&self.w4a4, w) {
            let ptr = *qweight.device_ptr() as u64;
            if batch <= MAX_VERIFY && w4.narrow_on(ptr) && *k == inn && *m == outn && inn % 64 == 0
                && x.len() >= batch * inn && out.len() >= batch * outn {
                self.w4a4_dense(w4, w, x, out, inn, outn, batch, w4.xgs(ptr));
                return;
            }
        }

        match w {
            W::Nvfp4 { qweight, scales, gs, .. } if batch <= MAX_VERIFY => {
                // MXFP4-native mode (--mxfp4=on): quantize X to E2M1 + UE4M3 scales, then the
                // sm_121a OMMA GEMV. Batch-invariance within the mode is by construction (N
                // padded to 8, fixed k-visit + fixed-order cross-warp reduce, no atomics).
                // Sensitive-tensor allowlist (design §8.1): allowlisted ptrs run the bf16 chain.
                if let Some(st) = &self.mxfp4 {
                    let ptr = *qweight.device_ptr() as u64;
                    if !st.allow_bf16.read().unwrap().contains(&ptr) {
                        self.launch_gemm_mxfp4(d(out), qweight, scales, gs, d(x), outn, inn, batch, st);
                        crate::dispatch_assert::record(crate::dispatch_assert::FMT_MXFP4, batch,
                            crate::dispatch_assert::ARM_MXFP4_NATIVE, MAX_VERIFY, "gemm_act/mxfp4");
                        return;
                    }
                }
                self.launch_gemm_fp4(d(out), *qweight.device_ptr() as u64, *scales.device_ptr() as u64,
                                     d(gs), d(x), outn, inn, batch, 0);
                crate::dispatch_assert::record(crate::dispatch_assert::FMT_NVFP4, batch,
                    crate::dispatch_assert::ARM_MMA, MAX_VERIFY, "gemm_act/nvfp4");
                return;
            }
            W::Fp8 { data, row_scale, .. } if batch <= MAX_VERIFY => {
                blaunch!(self, "gemm_mma_fp8_b", ((outn / 16) as u32,1,1), (256,1,1), 0,
                    (d(out), *data.device_ptr() as u64, d(row_scale),
                     d(x), outn as i32, inn as i32, batch as i32, 0u64));
                crate::dispatch_assert::record(crate::dispatch_assert::FMT_FP8, batch,
                    crate::dispatch_assert::ARM_MMA, MAX_VERIFY, "gemm_act/fp8");
                return;
            }
            W::Fp8Blk { data, blk_scale, .. } if batch <= MAX_VERIFY => {
                // Qwen block-128 FP8: fixed-order per-128-K spans folded ×scale_inv inside the
                // kernel — same N-independent column arithmetic contract as gemm_mma_fp8_b.
                blaunch!(self, "gemm_mma_fp8_blk_b", ((outn / 16) as u32,1,1), (256,1,1), 0,
                    (d(out), *data.device_ptr() as u64, d(blk_scale),
                     d(x), outn as i32, inn as i32, batch as i32, 0u64));
                crate::dispatch_assert::record(crate::dispatch_assert::FMT_FP8_BLK, batch,
                    crate::dispatch_assert::ARM_MMA, MAX_VERIFY, "gemm_act/fp8blk");
                return;
            }
            W::Nvfp4 { .. } | W::Fp8 { .. } | W::Fp8Blk { .. } => {
                // F8 diagnostic (--trace-wide): a quantized GEMM reached the PREFILL branch.
                // Prints shape+caller for the first calls — catches any decode/verify-width leak.
                static WIDE_N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                if crate::opts::var(crate::opt!("trace-wide")).is_ok() {
                    use std::sync::atomic::Ordering::Relaxed;
                    let c = WIDE_N.fetch_add(1, Relaxed);
                    if c < 5 || c % 200 == 0 {
                        let bt = std::backtrace::Backtrace::force_capture();
                        let frames: Vec<String> = format!("{bt}").lines()
                            .filter(|l| l.contains("gb10_inference")).map(|l| l.trim().to_string())
                            .take(6).collect();
                        eprintln!("[wide #{c}] outn={outn} inn={inn} batch={batch}\n   {}",
                                  frames.join("\n   "));
                    }
                }
                let dfmt = match w {
                    W::Nvfp4 { .. } => crate::dispatch_assert::FMT_NVFP4,
                    W::Fp8 { .. } => crate::dispatch_assert::FMT_FP8,
                    _ => crate::dispatch_assert::FMT_FP8_BLK,
                };
                crate::dispatch_assert::record(dfmt, batch,
                    crate::dispatch_assert::ARM_PREFILL, MAX_VERIFY, "gemm_act/prefill");
                self.gemm_quant_prefill(w, x, out, inn, outn, batch);
                return;
            }
            W::Nvfp4Raw { .. } => unreachable!("Nvfp4Raw is MoE-experts-only, not for gemm_act"),
            W::Bf16(b) => { self.gptq_tap_gemm(d(b), d(x), inn, batch); }
        }
        let w = match w { W::Bf16(b) => b, _ => unreachable!() };

        if batch <= BINV_BATCH {
            let smem = (batch * 256 * 4) as u32;
            blaunch!(self, "gemm_binv_b", (outn as u32,1,1), (256,1,1), smem,
                (d(out), d(w), d(x), outn as i32, inn as i32, batch as i32));
            crate::dispatch_assert::record(crate::dispatch_assert::FMT_BF16, batch,
                crate::dispatch_assert::ARM_BF16_BINV, MAX_VERIFY, "gemm_act/bf16-binv");
            return;
        }
        // Phase-8 (F9 root cause): wide bf16 GEMMs run the deterministic per-16-column slices of
        // gemm_binv_b BY DEFAULT. The old cuBLASLt call here fed the MoE ROUTER logits at prefill
        // widths: cuBLASLt's algo heuristic takes the operand ADDRESSES into account, so the same
        // (M,N,K) rounded differently depending on which pool buffer the window landed in — and the
        // router is REPLICATED under TP, computed independently per rank with no all-reduce, so a
        // near-tie top-k flip on one rank only made the expert-parallel combine sum MIXED experts
        // (observed: warm prefix-reuse requests diverged from cold AND from each other,
        // deterministically per request ordinal, while cold requests were bit-identical; layer-0
        // post-mixer residual identical but post-FFN residual wildly different — the router flip
        // signature). The sliced path is address-invariant, rank-invariant (same kernel + same
        // replicated weights + same all-reduced bf16 input), and numerically equal to the decode
        // contract. --prefill-bf16-binv=0 restores the old cuBLASLt behavior (diagnostics A/B).
        if crate::opts::var(crate::opt!("prefill-bf16-binv")).map_or(true, |v| v != "0") {
            static SLICE_MARK: std::sync::Once = std::sync::Once::new();
            SLICE_MARK.call_once(|| eprintln!("[bf16-slice] deterministic sliced gemm_binv_b ENGAGED for wide bf16 GEMMs (batch>{BINV_BATCH})"));
            for c0 in (0..batch).step_by(BINV_BATCH) {
                let cb = BINV_BATCH.min(batch - c0);
                let smem = (cb * 256 * 4) as u32;
                blaunch!(self, "gemm_binv_b", (outn as u32,1,1), (256,1,1), smem,
                    (d(out) + (c0 * outn) as u64 * 2, d(w),
                     d(x) + (c0 * inn) as u64 * 2, outn as i32, inn as i32, cb as i32));
            }
            crate::dispatch_assert::record(crate::dispatch_assert::FMT_BF16, batch,
                crate::dispatch_assert::ARM_BF16_SLICE, MAX_VERIFY, "gemm_act/bf16-slice");
            return;
        }
        let cfg = GemmConfig::<half::bf16> {
            transa: OP::CUBLAS_OP_T, transb: OP::CUBLAS_OP_N,
            m: outn as i32, n: batch as i32, k: inn as i32,
            alpha: half::bf16::from_f32(1.0), lda: inn as i32,
            ldb: inn as i32, beta: half::bf16::from_f32(0.0), ldc: outn as i32,
        };
        unsafe { self.blas.gemm(cfg, w, x, out).expect("gemm bf16 act"); }
    }

    /// E9: `gemm_act` for the FIRST consumer GEMM of a TP barrier — the attention/GDN input
    /// projections, which read the barrier-reduced hidden but whose weights do not depend on it.
    /// Only the NVFP4 serving path (batch <= MAX_VERIFY, mxfp4 off) carries the programmatic
    /// launch + `pdl=1` preamble; every other format/branch falls back to the plain `gemm_act`
    /// (byte-identical — the overlap is an fp4-serving optimization, and `launch_gemm_fp4_pdl`
    /// itself falls back when E9 is off).
    pub(super) fn gemm_act_pdl(&self, w: &W, x: &B, out: &mut B, inn: usize, outn: usize, batch: usize) {
        if let W::Nvfp4 { qweight, scales, gs, .. } = w {
            if batch <= MAX_VERIFY && self.mxfp4.is_none() {
                let _rot_guard; let x: &B = if self.is_rotated(w) { _rot_guard = self.rot_x(x, inn, batch); &*_rot_guard } else { x };
                self.launch_gemm_fp4_pdl(d(out), *qweight.device_ptr() as u64, *scales.device_ptr() as u64,
                                         d(gs), d(x), outn, inn, batch, 0);
                return;
            }
        }
        self.gemm_act(w, x, out, inn, outn, batch);
    }

    /// Batch-invariant bf16 matmul at EVERY serving width: forces `gemm_binv_b` (strided-k +
    /// fixed tree-reduce, N-independent column arithmetic — the same kernel decode uses) instead
    /// of `gemm_act`'s batch>2 cuBLAS arm. cuBLAS picks a different kernel per shape, so a
    /// depth≥3 verify computed ulp-different logits than a decode — which flipped top-8 expert
    /// selections on near-ties and silently broke MoE-MTP losslessness (found 2026-07-20 on the
    /// 122B: single-node d4 MISMATCH at token 25). Quantized weights already route to the
    /// batch-invariant mma kernels via `gemm_act`, so delegate there; only bf16 needs the force.
    /// Used by the MoE router and shared_gate — the bf16 tensors on the decode/verify path.
    pub(super) fn gemm_act_binv(&self, w: &W, x: &B, out: &mut B, inn: usize, outn: usize, batch: usize) {
        let w = match w {
            W::Bf16(b) if batch <= MAX_VERIFY => b,
            // prefill (batch > MAX_VERIFY) is outside the invariance contract — cuBLAS is fine
            // there; quantized weights are already batch-invariant via the mma path.
            _ => { self.gemm_act(w, x, out, inn, outn, batch); return; }
        };
        let smem = (batch * 256 * 4) as u32;
        blaunch!(self, "gemm_binv_b", (outn as u32,1,1), (256,1,1), smem,
            (d(out), d(w), d(x), outn as i32, inn as i32, batch as i32));
    }
}
