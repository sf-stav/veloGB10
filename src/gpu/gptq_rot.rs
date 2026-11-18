//! `impl GpuModel`: MR-GPTQ activation micro-rotation at serve time, GPTQ down-projection Hessians and lm-head helpers. Map: `mod.rs`.

use super::*;

impl GpuModel {
    /// Device copy of a bf16 buffer on the compute stream (the GPTQ driver keeps a layer's input
    /// hidden states across the two calibration passes).
    pub fn gptq_clone_b(&self, b: &B) -> B {
        let n = b.len();
        let out = self.dev.alloc_zeros::<half::bf16>(n).unwrap();
        unsafe { cudarc::driver::result::memcpy_dtod_async(d(&out), d(b), n * 2, self.stream.stream).unwrap(); }
        self.sync_stream();
        out
    }
}


// ===================== MR-GPTQ: activation micro-rotation at serve time =====================
impl GpuModel {
    pub(super) fn is_rotated(&self, w: &W) -> bool {
        if self.rotated.is_empty() { return false; }
        match w { W::Nvfp4 { qweight, .. } | W::Nvfp4Raw { qweight, .. } => self.rotated.contains(&(*qweight.device_ptr() as u64)), _ => false }
    }
    pub(super) fn take_rotation_marker(&mut self, w: &W) -> bool {
        match w {
            W::Nvfp4 { qweight, .. } | W::Nvfp4Raw { qweight, .. } => {
                self.rotated.remove(&(*qweight.device_ptr() as u64))
            }
            _ => false,
        }
    }
    pub(super) fn restore_rotation_marker(&mut self, rotated: bool, w: &W) {
        if rotated {
            if let W::Nvfp4 { qweight, .. } | W::Nvfp4Raw { qweight, .. } = w {
                self.rotated.insert(*qweight.device_ptr() as u64);
            }
        }
    }
    /// x' = (H16/4)·x per 16-block of the feature dim, into the rotation scratch (grown lazily —
    /// the eager warmup covers every decode shape before a graph capture; prefill is not captured).
    pub(super) fn rot_x(&self, x: &B, inn: usize, batch: usize) -> parking_lot::MappedMutexGuard<'_, B> {
        assert!(inn % 16 == 0, "MR-GPTQ rotation needs K % 16 == 0 (K = {inn})");
        let n = inn * batch;
        let mut g = self.rot_small.lock();
        if g.as_ref().map_or(true, |b| b.len() < n) { *g = Some(self.dev.alloc_zeros::<half::bf16>(n.max(1 << 18)).unwrap()); }
        {
            let buf = g.as_ref().unwrap();
            blaunch!(self, "gptq_rotate_act_b", grid(n / 16), (256,1,1), 0, (d(buf), d(x), (n / 16) as i64));
        }
        parking_lot::MutexGuard::map(g, |o| o.as_mut().unwrap())
    }
    pub(super) fn rot_inplace(&self, x: &B, n: usize) {
        assert!(n % 16 == 0);
        blaunch!(self, "gptq_rotate_act_b", grid(n / 16), (256,1,1), 0, (d(x), d(x), (n / 16) as i64));
    }
    /// Read `quantization_config.transform` from the artifact's config.json and mark the NVFP4
    /// weights of the listed groups as rotated (their inputs get the H16/4 micro-rotation).
    pub(super) fn apply_transform_config(&mut self, model_dir: &str) -> anyhow::Result<()> {
        let p = std::path::Path::new(model_dir).join("config.json");
        let Ok(raw) = std::fs::read_to_string(&p) else { return Ok(()); };
        let j: serde_json::Value = serde_json::from_str(&raw)?;
        let t = &j["quantization_config"]["transform"];
        if t.is_null() { return Ok(()); }
        anyhow::ensure!(t["type"].as_str() == Some("hadamard16"), "unknown quantization transform {:?}", t["type"]);
        anyhow::ensure!(self.mxfp4.is_none(), "MR-GPTQ artifacts are not supported in --mxfp4 mode");
        let groups: Vec<String> = t["groups"].as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect()).unwrap_or_default();
        let has = |g: &str| groups.iter().any(|x| x == g);
        let mut set = std::collections::HashSet::new();
        let mut mark = |w: &W| { if let W::Nvfp4 { qweight, .. } | W::Nvfp4Raw { qweight, .. } = w { set.insert(*qweight.device_ptr() as u64); } };
        for l in &self.layers {
            if has("attn") { if let Some(fa) = &l.fa { match &fa.qkv { AttnIn::Fused(w) => mark(w), AttnIn::Split { q, k, v } => { mark(q); mark(k); mark(v); } } mark(&fa.o_proj); if let Some(ix) = &fa.indexer { mark(&ix.qk_proj); } } }
            if has("gdn") { if let Some(la) = &l.la { match &la.in_proj { GdnIn::Fused(w) => mark(w), GdnIn::Split { qkv, z, b, a } => { mark(qkv); mark(z); mark(b); mark(a); } } mark(&la.out_proj); } }
            match &l.mlp {
                Ffn::Moe(m) => { if has("expert") { mark(&m.gate_up); mark(&m.down); } if has("mlp") { mark(&m.shared.gate); mark(&m.shared.up); mark(&m.shared.down); if let Some(w) = &m.shared_gate_up { mark(w); } } }
                Ffn::Dense(m) => { if has("mlp") { mark(&m.gate); mark(&m.up); mark(&m.down); } }
            }
            if has("hc") { if let Some((a, b)) = &l.hc { mark(&a.down); mark(&a.up); mark(&b.down); mark(&b.up); } }
            if has("ple") { if let Some(p) = &l.ple { mark(&p.key_proj); mark(&p.value_proj); } }
            // the MoE router is GPTQ-able (`--gptq-groups router`): quantized in the rotated basis, so
            // its input must be rotated at serving too (missing before 2026-08-30: a router-GPTQ
            // artifact routed on unrotated logits and emitted EOS on every prompt).
            if has("router") { if let Ffn::Moe(m) = &l.mlp { mark(&m.router); } }
        }
        if has("lmhead") {
            if let Some(w) = &self.lm_head { mark(w); }
            if let Some(w) = &self.df2_full_head { mark(w); }
        }
        println!("MR-GPTQ transform: hadamard16 on groups {:?} — {} rotated weights.", groups, set.len());
        self.rotated = set;
        Ok(())
    }
}


impl GpuModel {
    /// Down-projection input Hessian of expert `e` over the token subsample `xs` [h, n]:
    /// act = silu(gate)·up through the expert's bf16 gate_up, H = act·actᵀ (K = mi).
    pub fn gptq_down_hess_from(
        &self,
        gu: &B,
        e: usize,
        xs: &B,
        n: usize,
        h: usize,
        mi: usize,
        segments: &[(usize, usize, f32)],
    ) -> GptqHess {
        use cudarc::cublas::sys::{cublasComputeType_t, cublasGemmAlgo_t, cudaDataType};
        let hess = self.gptq_hess_new(mi);
        let (one, zero) = (1.0f32, 0.0f32);
        const CH: usize = 2048;
        let gu32 = self.dev.alloc_zeros::<f32>(2 * mi * CH).unwrap();
        let act = self.dev.alloc_zeros::<half::bf16>(mi * CH).unwrap();
        let owned_segments;
        let segments = if segments.is_empty() {
            owned_segments = vec![(0, n, 1.0)];
            owned_segments.as_slice()
        } else {
            segments
        };
        for &(seg_start, seg_len, weight) in segments {
            let mut rel = 0;
            while rel < seg_len {
                let t0 = seg_start + rel;
                let nn = (seg_len - rel).min(CH);
                unsafe {
                    cudarc::cublas::result::gemm_ex(
                        *self.blas.handle(),
                        OP::CUBLAS_OP_T,
                        OP::CUBLAS_OP_N,
                        (2 * mi) as i32,
                        nn as i32,
                        h as i32,
                        &one as *const f32 as *const _,
                        (d(gu) + (e * 2 * mi * h * 2) as u64) as *const _,
                        cudaDataType::CUDA_R_16BF,
                        h as i32,
                        (d(xs) + (t0 * h * 2) as u64) as *const _,
                        cudaDataType::CUDA_R_16BF,
                        h as i32,
                        &zero as *const f32 as *const _,
                        d(&gu32) as *mut _,
                        cudaDataType::CUDA_R_32F,
                        (2 * mi) as i32,
                        cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                    )
                    .expect("gptq fallback gate_up gemm");
                }
                blaunch!(
                    self,
                    "gptq_silu_mul_gu_b",
                    grid(mi * nn),
                    (256, 1, 1),
                    0,
                    (d(&act), d(&gu32), mi as i32, nn as i32)
                );
                self.gptq_hess_accum(d(&hess.h), mi, d(&act), nn, weight);
                rel += nn;
            }
        }
        self.sync_stream();
        GptqHess { k: mi, h: hess.h, n, amax: hess.amax, rot_scratch: hess.rot_scratch }
    }
    /// Replace a layer's big linear weights by 1-element dummies (the `--gptq` driver loads the
    /// base artifact only for its embeddings / norms / PLE / MTP: every layer is rebuilt from the
    /// source, so the base's 68 GB of experts would otherwise sit idle in memory).
    pub fn gptq_drop_layer_weights(&mut self, li: usize) {
        let dummy = || W::Bf16(self.dev.alloc_zeros::<half::bf16>(16).unwrap());
        let layer = &mut self.layers[li];
        if let Some(fa) = layer.fa.as_mut() {
            fa.qkv = AttnIn::Split { q: dummy(), k: dummy(), v: dummy() }; fa.o_proj = dummy();
            if let Some(ix) = fa.indexer.as_mut() { ix.qk_proj = dummy(); }
        }
        if let Some(la) = layer.la.as_mut() { la.in_proj = GdnIn::Split { qkv: dummy(), z: dummy(), b: dummy(), a: dummy() }; la.out_proj = dummy(); }
        match &mut layer.mlp {
            Ffn::Moe(m) => { m.gate_up = dummy(); m.down = dummy(); m.shared.gate = dummy(); m.shared.up = dummy(); m.shared.down = dummy(); m.router = dummy(); }
            Ffn::Dense(m) => { m.gate = dummy(); m.up = dummy(); m.down = dummy(); }
        }
        if let Some((a, b)) = layer.hc.as_mut() { a.down = dummy(); a.up = dummy(); b.down = dummy(); b.up = dummy(); }
        if let Some(p) = layer.ple.as_mut() { p.key_proj = dummy(); p.value_proj = dummy(); }
    }
    pub fn gptq_num_layers(&self) -> usize { self.layers.len() }
    /// `--gptq lmhead`: fold one sample's LM-head input (the final mixer applied to the residual
    /// stream `resid` [n, rw]) into `hess` (K = hidden) and its |x| max.
    pub fn gptq_lm_head_hess_accum(
        &self,
        pool: &mut Pool,
        resid: &B,
        n: usize,
        hess: &GptqHess,
        sample_weight: f32,
    ) {
        let h = self.cfg.hidden_size;
        let out = pool.get_bf16(h * n);
        if let Some(m) = self.hc_mixer.as_ref() {
            self.hc_pre(pool, &out, None, d(resid), m, n);
        } else {
            blaunch!(
                self,
                "rmsnorm_b",
                (n as u32, 1, 1),
                (1024, 1, 1),
                4096u32,
                (
                    d(&out),
                    d(resid),
                    d(&self.final_norm),
                    h as i32,
                    n as i32,
                    fbits(self.cfg.rms_eps)
                )
            );
        }
        self.gptq_hess_accum(d(&hess.h), h, d(&out), n, sample_weight);
        blaunch!(
            self,
            "gptq_absmax_b",
            grid(h * n),
            (256, 1, 1),
            0,
            (d(&hess.amax), d(&out), (h * n) as i64)
        );
        self.sync_stream();
        pool.release_bf16(out, h * n);
    }
    /// The installed bf16 LM head (device pointer, rows) — Some only while the calibration serves it in bf16.
    pub fn gptq_lm_head_bf16(&self) -> Option<(u64, usize)> {
        match &self.lm_head { Some(W::Bf16(b)) => Some((*b.device_ptr() as u64, b.len() / self.cfg.hidden_size)), _ => None }
    }
    /// `--gptq`: serve the non-layer tensors during calibration exactly as the artifact will carry
    /// them (source bf16 / the quantizer's own RTN or FP8), instead of the base artifact's copies.
    pub fn gptq_install_nonlayer(&mut self, embed: Option<W>, lm_head: Option<W>, mixer: Option<(W, W)>) {
        if let Some(e) = embed { self.embed = e; }
        if let Some(l) = lm_head { self.lm_head = Some(l); }
        if let Some((dn, up)) = mixer { if let Some(m) = self.hc_mixer.as_mut() { m.down = dn; m.up = up; } }
        self.sync_stream();
    }
    /// `--rotate`: the sequential pass must serve the freshly installed rotated weights with the
    /// activation micro-rotation, exactly like a loaded MR-GPTQ artifact.
    pub fn gptq_mark_rotated(&mut self, w: &W) {
        if let W::Nvfp4 { qweight, .. } | W::Nvfp4Raw { qweight, .. } = w { self.rotated.insert(*qweight.device_ptr() as u64); }
    }
    /// Fresh bf16 layers must not inherit device-pointer markers from the base artifact.
    pub fn gptq_reset_rotation(&mut self) {
        self.rotated.clear();
    }

}
