//! GPTQ -> NVFP4, device side (driver: `src/gptq.rs`): Hessian / tap types and the `gptq_*` methods. Map: `mod.rs`.

use super::*;

// ===================== GPTQ → NVFP4, device side (driver: src/gptq.rs) =====================

/// One Hessian accumulator: H [k, k] f32 (row-major == column-major, symmetric) += X·Xᵀ over every
/// calibration token whose activation reached the tapped GEMM; `n` counts the tokens.
pub struct GptqHess { pub k: usize, pub h: S, pub n: usize,
    /// |x| max over every calibration activation seen (float bits; `gptq_amax`) — the W4A4 `input_global_scale`.
    pub amax: cudarc::driver::CudaSlice<u32>,
    /// Persistent MR amax scratch (capacity grows to the largest external activation batch).
    pub(super) rot_scratch: Option<B> }

#[derive(Clone, serde::Serialize)]
pub struct CalibLayerProfile {
    pub layer: usize,
    pub mean: f64,
    pub std: f64,
    pub rms: f64,
    pub sketch: Vec<f32>,
}

/// The armed taps of one calibration pass: bf16 GEMM weights by device pointer (`gemm_act`'s
/// `W::Bf16` arm), plus the per-expert accumulators of the layer's bf16 routed experts
/// (`moe_batch`, gate_up inputs at K = hidden, down inputs at K = moe_intermediate).
#[derive(Default)]
pub struct GptqTap {
    pub by_ptr: std::collections::HashMap<u64, GptqHess>,
    pub moe_gu: Vec<GptqHess>,
    pub moe_dn: Vec<GptqHess>,
    /// Layer-wide fallbacks for under-calibrated experts (llm-compressor's `calibrate_all_experts`
    /// idea): the gate_up-input Hessian over EVERY token of the layer, and a fixed subsample of
    /// those tokens (`moe_all_x`, `[h, moe_all_n]`) from which a down-input Hessian can be built
    /// for any expert after the pass.
    pub moe_all: Option<GptqHess>,
    pub moe_all_x: Option<B>,
    pub moe_all_n: usize,
    pub moe_all_cap: usize,
    /// Contiguous ranges copied into `moe_all_x`, with the MaCa weight of the source sequence.
    pub moe_all_segments: Vec<(usize, usize, f32)>,
    /// Hessian weight for the current sequence (1 for legacy GPTQ, 1/L for MaCa).
    pub sample_weight: f32,
    /// Profile-only fields used by `--calib-profile`; inert during GPTQ.
    pub profile_enabled: bool,
    pub profile_layers: Vec<usize>,
    pub profile_sketch_dim: usize,
    pub profile_activations: Vec<CalibLayerProfile>,
    pub profile_expert_counts: std::collections::BTreeMap<usize, Vec<u64>>,
    pub profile_current_layer: Option<usize>,
    /// Calibration-only pass: the layer's output is discarded, so the (slow, scalar) bf16 routed
    /// experts kernel is skipped once the taps have what they need — the MoE output is zeroed.
    pub skip_experts: bool,
}

impl GpuModel {
    pub fn gptq_hess_new(&self, k: usize) -> GptqHess {
        let mut h = self.dev.alloc_zeros::<f32>(k * k).unwrap();
        self.dev.memset_zeros(&mut h).unwrap();
        GptqHess {
            k,
            h,
            n: 0,
            amax: self.dev.htod_sync_copy(&[0u32]).unwrap(),
            rot_scratch: None,
        }
    }
    pub fn gptq_arm(&self, tap: GptqTap) {
        *self.gptq_tap.lock().unwrap() = Some(tap);
    }
    pub fn gptq_disarm(&self) -> Option<GptqTap> {
        self.gptq_tap.lock().unwrap().take()
    }
    pub fn gptq_set_sample_weight(&self, weight: f32) {
        assert!(
            weight.is_finite() && weight > 0.0,
            "invalid GPTQ sample weight {weight}"
        );
        if let Some(tap) = self.gptq_tap.lock().unwrap().as_mut() {
            tap.sample_weight = weight;
        }
    }
    pub fn gptq_profile_layer(&self, layer: usize, x: &B, k: usize, n: usize) {
        let (enabled, selected, dim) = {
            let mut guard = self.gptq_tap.lock().unwrap();
            let Some(tap) = guard.as_mut() else { return };
            if !tap.profile_enabled {
                return;
            }
            tap.profile_current_layer = Some(layer);
            (
                true,
                tap.profile_layers.contains(&layer),
                tap.profile_sketch_dim,
            )
        };
        if !enabled || !selected {
            return;
        }
        assert!(
            dim > 0 && dim <= 256,
            "calibration sketch dimension must be 1..256"
        );
        let stats = self.dev.alloc_zeros::<f64>(2).unwrap();
        let sketch = self.dev.alloc_zeros::<f32>(dim).unwrap();
        blaunch!(
            self,
            "calib_profile_b",
            (k as u32, 1, 1),
            (256, 1, 1),
            (2 * 256 * 8) as u32,
            (
                d(&stats),
                d(&sketch),
                d(x),
                k as i32,
                n as i32,
                dim as i32,
                (0x6a09e667u32 ^ layer as u32)
            )
        );
        self.sync_stream();
        let st = self.dev.dtoh_sync_copy(&stats).unwrap();
        let mut sk = self.dev.dtoh_sync_copy(&sketch).unwrap();
        let count = (k * n) as f64;
        let mean = st[0] / count;
        let variance = (st[1] / count - mean * mean).max(0.0);
        let norm = (k as f32).sqrt().max(1.0);
        for value in &mut sk {
            *value /= norm;
        }
        self.gptq_tap
            .lock()
            .unwrap()
            .as_mut()
            .unwrap()
            .profile_activations
            .push(CalibLayerProfile {
                layer,
                mean,
                std: variance.sqrt(),
                rms: (st[1] / count).max(0.0).sqrt(),
                sketch: sk,
            });
    }

    pub(super) fn gptq_profile_moe(&self, ids_ptr: u64, batch: usize, topk: usize, ne: usize) {
        let layer = {
            let guard = self.gptq_tap.lock().unwrap();
            let Some(tap) = guard.as_ref() else { return };
            if !tap.profile_enabled {
                return;
            }
            let Some(layer) = tap.profile_current_layer else {
                return;
            };
            layer
        };
        self.sync_stream();
        let mut ids = vec![0i32; topk * batch];
        unsafe {
            cudarc::driver::result::memcpy_dtoh_sync(&mut ids, ids_ptr)
                .expect("calib profile ids dtoh");
        }
        let mut counts = vec![0u64; ne];
        for id in ids {
            if id >= 0 && (id as usize) < ne {
                counts[id as usize] += 1;
            }
        }
        self.gptq_tap
            .lock()
            .unwrap()
            .as_mut()
            .unwrap()
            .profile_expert_counts
            .insert(layer, counts);
    }
    pub fn gptq_layer_mut(&mut self, li: usize) -> &mut GpuLayer {
        &mut self.layers[li]
    }
    pub fn gptq_layer(&self, li: usize) -> &GpuLayer {
        &self.layers[li]
    }
    pub fn gptq_dev(&self) -> &Arc<CudaDevice> {
        &self.dev
    }
    /// Accumulate the LM-head Hessian from the complete final-normalized hidden stream. Serving
    /// computes logits only for the last prompt token, so a normal GEMM tap would see one vector
    /// per sample instead of `seqlen` vectors. `normed` is reusable [h, n] scratch.
    ///
    /// The Hessian is accumulated in the original basis; when MR is requested only the amax path
    /// is rotated in-place, because `gptq_h32` rotates H later while W4A4 needs its activation scale
    /// measured in the actually served (Hadamard) basis.
    pub fn gptq_lmhead_hess_accum(
        &self,
        acc: &mut GptqHess,
        residual: &B,
        normed: &B,
        n: usize,
        rotate_amax: bool,
        sample_weight: f32,
    ) {
        let h = self.cfg.hidden_size;
        assert_eq!(acc.k, h);
        assert!(residual.len() >= h * n && normed.len() >= h * n);
        blaunch!(
            self,
            "rmsnorm_b",
            (n as u32, 1, 1),
            (1024, 1, 1),
            (4096) as u32,
            (
                d(normed),
                d(residual),
                d(&self.final_norm),
                h as i32,
                n as i32,
                fbits(self.cfg.rms_eps)
            )
        );
        self.gptq_hess_accum(d(&acc.h), h, d(normed), n, sample_weight);
        if rotate_amax {
            self.rot_inplace(normed, h * n);
        }
        blaunch!(
            self,
            "gptq_absmax_b",
            grid(h * n),
            (256, 1, 1),
            0,
            (d(&acc.amax), d(normed), (h * n) as i64)
        );
        acc.n += n;
    }

    /// H += X·Xᵀ for X = bf16 [k, n] (feature-contiguous columns), f32 accumulate.
    pub(super) fn gptq_hess_accum(&self, h_ptr: u64, k: usize, x_ptr: u64, n: usize, scale: f32) {
        use cudarc::cublas::sys::{cublasComputeType_t, cublasGemmAlgo_t, cudaDataType};
        let onef = 1.0f32;
        unsafe {
            cudarc::cublas::result::gemm_ex(
                *self.blas.handle(),
                OP::CUBLAS_OP_N,
                OP::CUBLAS_OP_T,
                k as i32,
                k as i32,
                n as i32,
                &scale as *const f32 as *const _,
                x_ptr as *const _,
                cudaDataType::CUDA_R_16BF,
                k as i32,
                x_ptr as *const _,
                cudaDataType::CUDA_R_16BF,
                k as i32,
                &onef as *const f32 as *const _,
                h_ptr as *mut _, cudaDataType::CUDA_R_32F, k as i32,
                cublasComputeType_t::CUBLAS_COMPUTE_32F, cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT).expect("gptq hessian gemm");
        }
    }

    /// Accumulate an externally staged bf16 activation matrix `[k,n]`. DFlash2 owns a
    /// separate execution stream, so its calibration driver stages host snapshots onto the
    /// trunk device before calling this method. H stays in the original basis; only the amax
    /// copy is Hadamard-rotated, matching `gptq_h32` and the served MR activation path.
    pub fn gptq_hess_accum_external(&self, acc: &mut GptqHess, x: &B, n: usize,
                                    rotate_amax: bool) {
        assert_eq!(x.len(), acc.k * n, "external GPTQ activation shape");
        self.gptq_hess_accum(d(&acc.h), acc.k, d(x), n, 1.0);
        let ap = d(&acc.amax);
        if rotate_amax {
            let need = acc.k * n;
            if acc.rot_scratch.as_ref().map_or(true, |b| b.len() < need) {
                self.sync_stream();
                acc.rot_scratch = Some(self.dev.alloc_zeros::<half::bf16>(need).expect("external GPTQ MR scratch"));
            }
            let xr = acc.rot_scratch.as_ref().unwrap();
            unsafe { cudarc::driver::result::memcpy_dtod_async(d(xr), d(x), need * 2, self.stream.stream)
                .expect("external GPTQ MR copy"); }
            self.rot_inplace(xr, need);
            blaunch!(self, "gptq_absmax_b", grid(acc.k * n), (256,1,1), 0,
                (ap, d(xr), need as i64));
        } else {
            blaunch!(self, "gptq_absmax_b", grid(acc.k * n), (256,1,1), 0,
                (ap, d(x), (acc.k * n) as i64));
        }
        acc.n += n;
    }

    /// `gemm_act` tap: a registered bf16 weight (by pointer) accumulates its input's Hessian.
    pub(super) fn gptq_tap_gemm(&self, w_ptr: u64, x_ptr: u64, k: usize, n: usize) {
        let mut g = self.gptq_tap.lock().unwrap();
        let Some(tap) = g.as_mut() else { return; };
        let Some(acc) = tap.by_ptr.get_mut(&w_ptr) else { return; };
        assert_eq!(acc.k, k, "gptq tap: K mismatch on weight {w_ptr:#x}");
        let (hp, kk, ap, weight) = (
            d(&acc.h),
            acc.k,
            d(&acc.amax),
            tap.sample_weight.max(f32::MIN_POSITIVE),
        );
        acc.n += n;
        drop(g);
        self.gptq_hess_accum(hp, kk, x_ptr, n, weight);
        blaunch!(
            self,
            "gptq_absmax_b",
            grid(kk * n),
            (256, 1, 1),
            0,
            (ap, x_ptr, (kk * n) as i64)
        );
    }

    /// `moe_batch` tap: gather each expert's routed tokens, accumulate the gate_up-input Hessian
    /// (K = h) and, through the expert's own bf16 gate_up + silu·up, the down-input Hessian (K = mi).
    pub(super) fn gptq_tap_moe(&self, x: &B, ids_ptr: u64, batch: usize, gu: &B, h: usize, mi: usize, k: usize) {
        let armed = { let g = self.gptq_tap.lock().unwrap(); g.as_ref().map_or(false, |t| !t.moe_gu.is_empty()) };
        if !armed { return; }
        use cudarc::cublas::sys::{cudaDataType, cublasComputeType_t, cublasGemmAlgo_t};
        {
            // layer-wide all-token Hessian + the token subsample for the fallbacks
            let (all_ptr, sub, weight) = {
                let g = self.gptq_tap.lock().unwrap();
                let t = g.as_ref().unwrap();
                (
                    t.moe_all.as_ref().map(|a| d(&a.h)),
                    t.moe_all_x
                        .as_ref()
                        .map(|b| (d(b), t.moe_all_n, t.moe_all_cap)),
                    t.sample_weight.max(f32::MIN_POSITIVE),
                )
            };
            if let Some(hp) = all_ptr {
                self.gptq_hess_accum(hp, h, d(x), batch, weight);
                let mut g = self.gptq_tap.lock().unwrap();
                g.as_mut().unwrap().moe_all.as_mut().unwrap().n += batch;
            }
            if let Some((xp, n0, cap)) = sub {
                if n0 < cap {
                    let take = (cap - n0).min(batch);
                    unsafe {
                        cudarc::driver::result::memcpy_dtod_async(
                            xp + (n0 * h * 2) as u64,
                            d(x),
                            take * h * 2,
                            self.stream.stream,
                        )
                        .unwrap();
                    }
                    let mut g = self.gptq_tap.lock().unwrap();
                    let t = g.as_mut().unwrap();
                    t.moe_all_segments.push((n0, take, weight));
                    t.moe_all_n += take;
                }
            }
        }
        self.sync_stream();
        let mut ids = vec![0i32; k * batch];
        unsafe { cudarc::driver::result::memcpy_dtoh_sync(&mut ids, ids_ptr).expect("gptq ids dtoh"); }
        let ne = { let g = self.gptq_tap.lock().unwrap(); g.as_ref().unwrap().moe_gu.len() };
        let mut lists: Vec<Vec<i32>> = vec![Vec::new(); ne];
        for b in 0..batch { for j in 0..k { let e = ids[j + b * k]; if e >= 0 && (e as usize) < ne { lists[e as usize].push(b as i32); } } }
        let maxn = lists.iter().map(|l| l.len()).max().unwrap_or(0);
        if maxn == 0 { return; }
        let xg = self.dev.alloc_zeros::<half::bf16>(h * maxn).unwrap();
        let gu32 = self.dev.alloc_zeros::<f32>(2 * mi * maxn).unwrap();
        let act = self.dev.alloc_zeros::<half::bf16>(mi * maxn).unwrap();
        let (one, zero) = (1.0f32, 0.0f32);
        for (e, list) in lists.iter().enumerate() {
            let n = list.len();
            if n == 0 { continue; }
            let idx = self.dev.htod_sync_copy(list).unwrap();
            blaunch!(
                self,
                "gptq_gather_rows_b",
                grid(n * h),
                (256, 1, 1),
                0,
                (d(&xg), d(x), d(&idx), n as i32, h as i32)
            );
            let (hgu, hdn, agu, adn, weight) = {
                let g = self.gptq_tap.lock().unwrap();
                let t = g.as_ref().unwrap();
                (
                    d(&t.moe_gu[e].h),
                    d(&t.moe_dn[e].h),
                    d(&t.moe_gu[e].amax),
                    d(&t.moe_dn[e].amax),
                    t.sample_weight.max(f32::MIN_POSITIVE),
                )
            };
            self.gptq_hess_accum(hgu, h, d(&xg), n, weight);
            blaunch!(
                self,
                "gptq_absmax_b",
                grid(h * n),
                (256, 1, 1),
                0,
                (agu, d(&xg), (h * n) as i64)
            );
            // gate|up = Wgu_e · x : Wgu_e row-major [2mi, h] == column-major [h, 2mi] → opT
            unsafe {
                cudarc::cublas::result::gemm_ex(*self.blas.handle(),
                    OP::CUBLAS_OP_T, OP::CUBLAS_OP_N,
                    (2 * mi) as i32, n as i32, h as i32,
                    &one as *const f32 as *const _,
                    (d(gu) + (e * 2 * mi * h * 2) as u64) as *const _, cudaDataType::CUDA_R_16BF, h as i32,
                    d(&xg) as *const _, cudaDataType::CUDA_R_16BF, h as i32,
                    &zero as *const f32 as *const _,
                    d(&gu32) as *mut _,
                    cudaDataType::CUDA_R_32F,
                    (2 * mi) as i32,
                    cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
                .expect("gptq expert gate_up gemm");
            }
            blaunch!(
                self,
                "gptq_silu_mul_gu_b",
                grid(mi * n),
                (256, 1, 1),
                0,
                (d(&act), d(&gu32), mi as i32, n as i32)
            );
            self.gptq_hess_accum(hdn, mi, d(&act), n, weight);
            blaunch!(
                self,
                "gptq_absmax_b",
                grid(mi * n),
                (256, 1, 1),
                0,
                (adn, d(&act), (mi * n) as i64)
            );
            {
                let mut g = self.gptq_tap.lock().unwrap();
                let t = g.as_mut().unwrap();
                t.moe_gu[e].n += n;
                t.moe_dn[e].n += n;
            }
            self.sync_stream();   // idx/xg reuse across iterations
        }
    }

    /// f32 working copy of a bf16 [m, k] row-major weight (optionally micro-rotated along K).
    pub fn gptq_w32(&self, w_ptr: u64, m: usize, k: usize, rotate: bool) -> S {
        let out = self.dev.alloc_zeros::<f32>(m * k).unwrap();
        blaunch!(self, "gptq_bf16_to_f32_b", grid(m * k), (256,1,1), 0, (d(&out), w_ptr, (m * k) as i64));
        if rotate { blaunch!(self, "gptq_hadamard16_b", grid(m * (k / 16)), (256,1,1), 0, (d(&out), m as i32, k as i32, 0i32)); }
        out
    }
    /// Activation order for act-order GPTQ: column indices sorted by DESCENDING Hessian diagonal,
    /// stable on ties, non-finite diagonals last (finite first). Verbatim from the PR's
    /// `gptq.rs:static_activation_order` — see the port note in `gptq_h32`.
    fn static_activation_order(diag: &[f32]) -> Vec<i32> {
        let mut order: Vec<usize> = (0..diag.len()).collect();
        order.sort_by(|&a, &b| match (diag[a].is_finite(), diag[b].is_finite()) {
            (true, true) => diag[b].total_cmp(&diag[a]).then_with(|| a.cmp(&b)),
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            (false, false) => a.cmp(&b),
        });
        order.into_iter().map(|i| i as i32).collect()
    }

    /// The Hessian, copied (optionally rotated on both sides: H' = R·H·R) and damped:
    /// H += damp·mean(diag)·I, dead columns (diag 0) pinned to 1.
    pub fn gptq_h32(&self, hess: &S, k: usize, rotate: bool, damp: f32, act_order: bool) -> (S, Option<cudarc::driver::CudaSlice<i32>>) {
        let h = self.dev.alloc_zeros::<f32>(k * k).unwrap();
        unsafe { cudarc::driver::result::memcpy_dtod_async(d(&h), d(hess), k * k * 4, self.stream.stream).unwrap(); }
        if rotate {
            blaunch!(self, "gptq_hadamard16_b", grid(k * (k / 16)), (256,1,1), 0, (d(&h), k as i32, k as i32, 0i32));
            blaunch!(self, "gptq_hadamard16_b", grid((k / 16) * k), (256,1,1), 0, (d(&h), k as i32, k as i32, 1i32));
        }
        self.sync_stream();
        let mut hh = vec![0f32; k * k];
        unsafe { cudarc::driver::result::memcpy_dtoh_sync(&mut hh, d(&h)).unwrap(); }
        let mut mean = 0f64; for i in 0..k { mean += hh[i * k + i] as f64; } mean /= k as f64;
        assert!(mean > 0.0, "gptq_h32: empty Hessian (no calibration token reached this GEMM)");
        let diag: Vec<f32> = (0..k).map(|i| hh[i * k + i]).collect();
        let lam = (damp as f64 * mean) as f32;
        for i in 0..k { let di = &mut hh[i * k + i]; if *di == 0.0 { *di = 1.0; } *di += lam; }
        let h = self.dev.htod_sync_copy(&hh).unwrap();
        if act_order {
            // H0.3a port note: the PR calls `crate::gptq::static_activation_order` here, but
            // `src/gptq.rs` stays staged-but-undeclared this leg (the quantiser line is H0.4).
            // Verbatim copy of the pure function so the merged gpu.rs is self-contained.
            // DELETE this copy and re-point the call the moment `pub mod gptq` lands.
            let order = Self::static_activation_order(&diag);
            let perm = self.dev.htod_sync_copy(&order).unwrap();
            let hp = self.dev.alloc_zeros::<f32>(k * k).unwrap();
            blaunch!(self, "gptq_permute_h_b", grid(k * k), (256,1,1), 0,
                (d(&hp), d(&h), d(&perm), k as i32));
            (hp, Some(perm))
        } else {
            (h, None)
        }
    }
    /// The activation |x| max a tap accumulated (0 if it never fired).
    pub fn gptq_amax(&self, hess: &GptqHess) -> f32 {
        self.sync_stream();
        let mut out = [0u32];
        unsafe { cudarc::driver::result::memcpy_dtoh_sync(&mut out, d(&hess.amax)).unwrap(); }
        f32::from_bits(out[0])
    }
    pub fn gptq_absmax_f32(&self, w32: &S, n: usize) -> f32 {
        let acc = self.dev.htod_sync_copy(&[0u32]).unwrap();
        blaunch!(self, "gptq_absmax_f32_b", grid(n), (256,1,1), 0, (d(&acc), d(w32), n as i64));
        self.sync_stream();
        let mut out = [0u32];
        unsafe { cudarc::driver::result::memcpy_dtoh_sync(&mut out, d(&acc)).unwrap(); }
        f32::from_bits(out[0])
    }
    /// Sufficient statistics for one alternating tensor-scale step on an f32 working weight.
    pub fn gptq_scale_stats_f32(&self, w32: &S, n: usize, s_tensor: f32, nclip: usize) -> (f64, f64, f64) {
        assert!(n % 16 == 0);
        let stats = self.dev.alloc_zeros::<f64>(3).unwrap();
        let blocks = (n / 16).div_ceil(256).clamp(1, 4096) as u32;
        blaunch!(self, "gptq_scale_stats_f32_b", (blocks,1,1), (256,1,1), 0,
            (d(&stats), d(w32), (n / 16) as i64, fbits(s_tensor), nclip as i32));
        self.sync_stream();
        let v = self.dev.dtoh_sync_copy(&stats).unwrap();
        (v[0], v[1], v[2])
    }
    /// Same scale statistics directly from a contiguous bf16 tensor, optionally Hadamard-rotated.
    pub fn gptq_scale_stats_bf16(&self, w_ptr: u64, n: usize, rotate: bool, s_tensor: f32, nclip: usize) -> (f64, f64, f64) {
        assert!(n % 16 == 0);
        let stats = self.dev.alloc_zeros::<f64>(3).unwrap();
        let blocks = (n / 16).div_ceil(256).clamp(1, 4096) as u32;
        blaunch!(self, "gptq_scale_stats_bf16_b", (blocks,1,1), (256,1,1), 0,
            (d(&stats), w_ptr, (n / 16) as i64, rotate as i32, fbits(s_tensor), nclip as i32));
        self.sync_stream();
        let v = self.dev.dtoh_sync_copy(&stats).unwrap();
        (v[0], v[1], v[2])
    }
    /// Freeze the original block-16 E4M3 scales used by static activation-order GPTQ.
    pub fn gptq_static_scales(&self, w32: &S, m: usize, k: usize, s_tensor: f32, nclip: usize) -> cudarc::driver::CudaSlice<u8> {
        assert!(k % 16 == 0);
        let qs = self.dev.alloc_zeros::<u8>(m * k / 16).unwrap();
        blaunch!(self, "gptq_static_scales_b", grid(m * k / 16), (256,1,1), 0,
            (d(&qs), d(w32), (m * k / 16) as i64, fbits(s_tensor), nclip as i32));
        qs
    }
    /// Freeze block-16 scales by minimizing dw^T H_block dw across every positive finite E4M3
    /// scale. h32 must be rotated/damped but unpermuted, matching w32's original block layout.
    pub fn gptq_static_scales_hessian(
        &self, w32: &S, h32: &S, m: usize, k: usize, s_tensor: f32,
    ) -> cudarc::driver::CudaSlice<u8> {
        assert!(k % 16 == 0);
        let qs = self.dev.alloc_zeros::<u8>(m * k / 16).unwrap();
        let fallbacks = self.dev.alloc_zeros::<u64>(1).unwrap();
        let launch_grid = ((k / 16) as u32, m.div_ceil(8) as u32, 1u32);
        blaunch!(self, "gptq_static_scales_hessian_b", launch_grid, (256,1,1), 0,
            (d(&qs), d(w32), d(h32), d(&fallbacks), m as i32, k as i32, fbits(s_tensor)));
        self.sync_stream();
        let n = self.dev.dtoh_sync_copy(&fallbacks).unwrap()[0];
        if n != 0 {
            eprintln!("[gptq] local-Hessian scale search: {n} non-finite groups fell back to amax");
        }
        qs
    }
    /// Permute weight columns into descending activation-importance order.
    pub fn gptq_permute_w(&self, w32: &S, m: usize, k: usize, perm: &cudarc::driver::CudaSlice<i32>) -> S {
        let out = self.dev.alloc_zeros::<f32>(m * k).unwrap();
        blaunch!(self, "gptq_permute_w_b", grid(m * k), (256,1,1), 0,
            (d(&out), d(w32), d(perm), m as i32, k as i32));
        out
    }
    /// The GPTQ block sweep over all K columns. `u` = upper Cholesky factor of H⁻¹ (row-major).
    /// Returns (nibble codes [m, k/2], E4M3 block scales [m, k/16]) in the artifact layout.
    pub fn gptq_sweep(&self, w32: &S, u: &S, m: usize, k: usize, s_tensor: f32, nclip: usize) -> (Vec<u8>, Vec<u8>) {
        use cudarc::cublas::sys::{cudaDataType, cublasComputeType_t, cublasGemmAlgo_t};
        const BS: usize = 128;
        let bs = BS.min(k);
        assert!(k % 16 == 0 && bs % 16 == 0);
        let mut qw = self.dev.alloc_zeros::<u8>(m * k / 2).unwrap(); self.dev.memset_zeros(&mut qw).unwrap();
        let qs = self.dev.alloc_zeros::<u8>(m * k / 16).unwrap();
        let err = self.dev.alloc_zeros::<f32>(m * bs).unwrap();
        let (neg1, one) = (-1.0f32, 1.0f32);
        let mut c0 = 0usize;
        while c0 < k {
            let b = bs.min(k - c0);
            blaunch!(self, "gptq_sweep_b", grid(m), (256,1,1), 0,
                (d(w32), d(&qw), d(&qs), d(&err), d(u), m as i32, k as i32, c0 as i32, b as i32, fbits(s_tensor), nclip as i32));
            let c1 = c0 + b;
            if c1 < k {
                // W[:, c1:] -= Err[:, 0:b] · U[c0:c1, c1:]   (row-major ⇔ column-major transposes)
                unsafe {
                    cudarc::cublas::result::gemm_ex(*self.blas.handle(),
                        OP::CUBLAS_OP_N, OP::CUBLAS_OP_N,
                        (k - c1) as i32, m as i32, b as i32,
                        &neg1 as *const f32 as *const _,
                        (d(u) + ((c0 * k + c1) * 4) as u64) as *const _, cudaDataType::CUDA_R_32F, k as i32,
                        d(&err) as *const _, cudaDataType::CUDA_R_32F, b as i32,
                        &one as *const f32 as *const _,
                        (d(w32) + (c1 * 4) as u64) as *mut _, cudaDataType::CUDA_R_32F, k as i32,
                        cublasComputeType_t::CUBLAS_COMPUTE_32F, cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT).expect("gptq propagation gemm");
                }
            }
            c0 = c1;
        }
        self.sync_stream();
        let mut hq = vec![0u8; m * k / 2]; let mut hs = vec![0u8; m * k / 16];
        unsafe {
            cudarc::driver::result::memcpy_dtoh_sync(&mut hq, d(&qw)).unwrap();
            cudarc::driver::result::memcpy_dtoh_sync(&mut hs, d(&qs)).unwrap();
        }
        (hq, hs)
    }
    /// GPTQ sweep in static activation order. Scales stay in the original block-16 layout and
    /// packed codes are written back to original columns, so serving needs no permutation.
    pub fn gptq_sweep_static(&self, w32: &S, u: &S, perm: &cudarc::driver::CudaSlice<i32>,
                             qs: &cudarc::driver::CudaSlice<u8>, m: usize, k: usize,
                             s_tensor: f32) -> (Vec<u8>, Vec<u8>) {
        use cudarc::cublas::sys::{cudaDataType, cublasComputeType_t, cublasGemmAlgo_t};
        const BS: usize = 128;
        let bs = BS.min(k);
        assert!(k % 16 == 0 && bs % 16 == 0);
        let mut qw = self.dev.alloc_zeros::<u8>(m * k / 2).unwrap();
        self.dev.memset_zeros(&mut qw).unwrap();
        let err = self.dev.alloc_zeros::<f32>(m * bs).unwrap();
        let (neg1, one) = (-1.0f32, 1.0f32);
        let mut c0 = 0usize;
        while c0 < k {
            let b = bs.min(k - c0);
            blaunch!(self, "gptq_sweep_static_b", grid(m), (256,1,1), 0,
                (d(w32), d(&qw), d(qs), d(&err), d(u), d(perm), m as i32, k as i32,
                 c0 as i32, b as i32, fbits(s_tensor)));
            let c1 = c0 + b;
            if c1 < k {
                unsafe {
                    cudarc::cublas::result::gemm_ex(*self.blas.handle(),
                        OP::CUBLAS_OP_N, OP::CUBLAS_OP_N,
                        (k - c1) as i32, m as i32, b as i32,
                        &neg1 as *const f32 as *const _,
                        (d(u) + ((c0 * k + c1) * 4) as u64) as *const _, cudaDataType::CUDA_R_32F, k as i32,
                        d(&err) as *const _, cudaDataType::CUDA_R_32F, b as i32,
                        &one as *const f32 as *const _,
                        (d(w32) + (c1 * 4) as u64) as *mut _, cudaDataType::CUDA_R_32F, k as i32,
                        cublasComputeType_t::CUBLAS_COMPUTE_32F, cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT)
                        .expect("gptq static propagation gemm");
                }
            }
            c0 = c1;
        }
        self.sync_stream();
        let mut hq = vec![0u8; m * k / 2];
        let mut hs = vec![0u8; m * k / 16];
        unsafe {
            cudarc::driver::result::memcpy_dtoh_sync(&mut hq, d(&qw)).unwrap();
            cudarc::driver::result::memcpy_dtoh_sync(&mut hs, d(qs)).unwrap();
        }
        (hq, hs)
    }
    /// Upload packed NVFP4 records as a GEMM weight (MMA-repacked, per-tile reciprocal scale).
    pub fn gptq_w_nvfp4(&self, qw: &[u8], sc: &[u8], m: usize, k: usize, global_scale: f32) -> W {
        let (wt, st2) = crate::quant::repack_nvfp4_mma(qw, sc, m, k);
        let gsv = vec![1.0f32 / global_scale; m / 16];
        W::Nvfp4 { qweight: self.dev.htod_sync_copy(&wt).unwrap(), scales: self.dev.htod_sync_copy(&st2).unwrap(),
                   gs: self.dev.htod_sync_copy(&gsv).unwrap(), m, k }
    }
    /// Upload packed NVFP4 records as stacked MoE experts (raw layout).
    pub fn gptq_w_nvfp4_raw(&self, qw: &[u8], sc: &[u8], m: usize, k: usize, global_scale: f32) -> W {
        W::Nvfp4Raw { qweight: self.dev.htod_sync_copy(qw).unwrap(), scales: self.dev.htod_sync_copy(sc).unwrap(),
                      gs: 1.0 / global_scale, m, k }
    }
    pub fn gptq_w_bf16(&self, data: &[half::bf16]) -> W { W::Bf16(self.dev.htod_sync_copy(data).unwrap()) }
    pub fn gptq_w_fp8(&self, q: crate::quant::Fp8Tensor) -> W {
        let data = crate::quant::repack_fp8_mma(&q.qweight, q.m, q.k);
        W::Fp8 { data: self.dev.htod_sync_copy(&data).unwrap(), row_scale: self.dev.htod_sync_copy(&q.row_scale).unwrap(), m: q.m, k: q.k }
    }
    pub fn gptq_upload_bf16(&self, data: &[half::bf16]) -> B { self.dev.htod_sync_copy(data).unwrap() }
    pub fn gptq_sync(&self) { self.sync_stream(); }
}
