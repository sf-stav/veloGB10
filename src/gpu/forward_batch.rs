//! `impl GpuModel`: the batched forward driver: `mlp_batch`, `linear_attn_batch`, `forward_batch*`, `forward_batch_dev`
//! and the batched logits heads. Prefill, decode and verify all run through here. Map: `mod.rs`.

use super::*;

impl GpuModel {
    pub(super) fn mlp_batch(&self, pool: &mut Pool, x: &B, mlp: &GpuMlp, batch: usize, sharded: bool, fuse: bool) -> ReduceOut {
        let h = self.cfg.hidden_size;
        // TP shards the FFN (weights split at attach_tp): gate/up column-parallel (each rank owns
        // im/factor output rows → each streams 1/factor the gate/up bytes), down row-parallel (each rank
        // owns im/factor of K → its down output is a PARTIAL). So the effective intermediate here is
        // im/factor, and the row-parallel partials are summed by the all-reduce. world==1 = full,
        // unsharded. `sharded` is the CALLER's authoritative per-tensor signal (the MTP head passes
        // false — its weights are replicated), so the divisor is per-call, never a global read.
        let div = if sharded { self.ffn_shard_factor() } else { 1 };
        let im = self.cfg.intermediate_size / div;
        // Phase-8 F9 FIX: the W8A8 prefill lane pads tokens to a 64 multiple (tp64) and its GEMM
        // WRITES pad rows — gate/up must be sized to tp64*im or the kernel writes past the pool
        // bucket (batch=468: tp64=512, 512*8704 > next_pow2(468*8704) by 512KB — silent
        // neighbor corruption, deterministic per pool layout: the warm-window stream divergence).
        // Only prefill widths (batch>=256, the lane's own gate) round up; decode/verify buffers
        // keep their exact sizes.
        let bn = self.pf8_pad(batch);
        let mut gate = pool.get_bf16(im*bn);
        let mut upb = pool.get_bf16(im*bn);
        self.gemm_act(&mlp.gate, x, &mut gate, h, im, batch);
        self.gemm_act(&mlp.up, x, &mut upb, h, im, batch);
        blaunch!(self, "silu_mul_b", grid(im*batch), (256,1,1), 0, (d(&gate), d(&gate), d(&upb), (im*batch) as i32));
        if crate::opts::var(crate::opt!("layer-fullhash")).is_ok() && batch > 16 {
            self.sync_stream();
            let host = self.dev.dtoh_sync_copy(&gate).unwrap_or_default();
            let n_real = (im * batch).min(host.len());
            let mut x: u64 = 1469598103934665603;
            for v in host.iter().take(n_real) { for b in v.to_bits().to_le_bytes() { x ^= b as u64; x = x.wrapping_mul(1099511628211); } }
            eprintln!("[fh4] L0 gate {im}x{batch} {:012x}", x & 0xFFFFFFFFFFFF);
        }
        // Phase-8 F9 FIX #2 (found by the pf8 tripwire in the TP2 quality leg): the DOWN output
        // is the pf8 lane's GEMM sink too (both the direct gemm_act and tp_reduce_site's K1
        // partial land in it) — it must carry the same tp64 padding as gate/up or the lane
        // writes past the bucket exactly like the warm-window bug (batch~410: tp64=448).
        let obn = self.pf8_pad(batch);
        let mut out = pool.get_bf16(h*obn);
        // AR landing 2: with `fuse` the row-parallel reduce may stop after K1 and hand the LOCAL
        // partial up for the fused tp_reduce_resnorm_b epilogue (single-barrier only).
        let fused = if sharded {
            self.tp_reduce_site(pool, &mlp.down, &gate, &mut out, im, h, batch, fuse)
        } else {
            self.gemm_act(&mlp.down, &gate, &mut out, im, h, batch);
            None
        };
        pool.release_bf16(gate, im*bn); pool.release_bf16(upb, im*bn);
        match fused {
            Some(FusedPartial::F32(p)) => ReduceOut::Fused(FusedReduce { sink: out, partial: Some(p), mode: 1 }),
            Some(FusedPartial::F32Scratch) => ReduceOut::Fused(FusedReduce { sink: out, partial: None, mode: 1 }),
            Some(FusedPartial::Bf16) => ReduceOut::Fused(FusedReduce { sink: out, partial: None, mode: 0 }),
            None => ReduceOut::Full(out),
        }
    }
}

impl GpuModel {
    fn linear_attn_batch(&self, pool: &mut Pool, hidden: &B, la: &GpuLinearAttn,
                         conv_ptr: u64, s_ptr: u64, slot_ids_ptr: u64, batch: usize, fuse: bool,
                         s_tap: Option<&cudarc::driver::CudaSlice<f32>>) -> ReduceOut {
        let cfg = &self.cfg;
        // TP-local GDN geometry (see eff_lin_v_heads). Every dim below is rank-local when the mixers are
        // sharded, so the conv/state/gate/norm kernels touch only this box's heads.
        let (h, kd, vd) = (cfg.hidden_size, cfg.lin_k_dim, cfg.lin_v_dim);
        let nh = self.eff_lin_v_heads();
        let key_dim = self.eff_key_dim(); let value_dim = self.eff_value_dim();
        let conv_dim = self.eff_conv_dim(); let ck = cfg.conv_kernel;
        // F0: the fused GDN in_proj output IS [token, [qkv|z|b|a]] — under the default qkv/z/b/a are
        // offset views (base + b·mtot) and the split is deleted; --no-qkv-view keeps the packed
        // split for the A/B byte gate. b/a are carried WHOLE (nh_src rows) and sliced to this rank's
        // head range [h0, h0+nh) — under the views that is a pointer offset into the fused buffer.
        let old_pipeline = crate::opts::var(crate::opt!("no-qkv-view")).is_ok();
        let mtot = self.eff_gdn_fused_m();
        let nh_src = self.cfg.lin_num_v_heads;
        let h0 = self.gdn_head0();
        let mut qkv: Option<B> = None;
        let mut z: Option<B> = None;
        let mut b: Option<B> = None;
        let mut a: Option<B> = None;
        let mut fused_qkv: Option<B> = None;     // F0: the fused GDN buffer, views handed to consumers
        match &la.in_proj {
            GdnIn::Fused(w) => {
                // ONE GEMM instead of four. in_proj_b/in_proj_a are M=nh (32 on 9B), i.e. grid=2 on a
                // 48-SM GPU: 26 us to move 74 KB, and across 24 GDN layers that was 4.7% of ALL GEMM
                // time for 0.03% of the bytes. Fused, they are just rows of a big efficient kernel.
                let mut fb = pool.get_bf16(mtot * batch);
                // E9: the GDN input projection is the first consumer GEMM of the TP barrier chain —
                // its weight prefetch runs under the K2 cpu_done wait (plain `gemm_act` elsewhere).
                self.gemm_act_pdl(w, hidden, &mut fb, h, mtot, batch);
                if old_pipeline {
                    let qkvb = pool.get_bf16(conv_dim*batch);
                    let zb = pool.get_bf16(value_dim*batch);
                    let bb = pool.get_bf16(nh*batch);
                    let ab = pool.get_bf16(nh*batch);
                    blaunch!(self, "split_gdn_b", grid(mtot*batch), (256,1,1), 0,
                        (d(&qkvb), d(&zb), d(&bb), d(&ab), d(&fb),
                         conv_dim as i32, value_dim as i32, nh as i32, batch as i32,
                         nh_src as i32, h0 as i32));
                    pool.release_bf16(fb, mtot * batch);
                    qkv = Some(qkvb); z = Some(zb); b = Some(bb); a = Some(ab);
                } else {
                    fused_qkv = Some(fb);
                }
            }
            GdnIn::Split { qkv: pq, z: pz, b: pb, a: pa } => {
                // E9: the FIRST (qkv) GEMM carries the programmatic launch; z/b/a follow it plainly.
                // Geometry contract (asserted once): under TP the quantized parts arrive SHARDED
                // (qkv/z = rank-local head bands) while b/a stay replicated-full — every rank must
                // read rows [0..local) of qkv/z or the all-reduce sums one head band twice (the
                // TP=2 mojibake of 2026-09-04: both ranks silently GEMM'd the first half of
                // replicated full-width weights).
                static SPLIT_GEO: std::sync::Once = std::sync::Once::new();
                SPLIT_GEO.call_once(|| {
                    let rows = |w: &W| -> usize { match w {
                        W::Bf16(v) => v.len() / h.max(1),
                        W::Fp8 { m, .. } | W::Fp8Blk { m, .. } | W::Nvfp4 { m, .. }
                        | W::Nvfp4Raw { m, .. } => *m, } };
                    let (rq, rz, rb, ra) = (rows(pq), rows(pz), rows(pb), rows(pa));
                    assert_eq!(rq, conv_dim, "split-GDN geometry: in_proj_qkv rows {rq} != local conv_dim {conv_dim} (world {})", self.tp_world);
                    assert_eq!(rz, value_dim, "split-GDN geometry: in_proj_z rows {rz} != local value_dim {value_dim} (world {})", self.tp_world);
                    assert_eq!(rb, nh_src, "split-GDN geometry: in_proj_b rows {rb} != full head count {nh_src}");
                    assert_eq!(ra, nh_src, "split-GDN geometry: in_proj_a rows {ra} != full head count {nh_src}");
                });
                let mut qkvb = pool.get_bf16(conv_dim*batch);
                let mut zb = pool.get_bf16(value_dim*batch);
                // b/a are replicated-full (bf16 rows have no shard pipeline): GEMM them at the
                // FULL head count; the kernel reads the rank's band via pointer offset below.
                // At world==1 nh_src == nh, so this is byte-identical to the old local call.
                let mut bb = pool.get_bf16(nh_src*batch);
                let mut ab = pool.get_bf16(nh_src*batch);
                self.gemm_act_pdl(pq, hidden, &mut qkvb, h, conv_dim, batch);
                self.gemm_act(pz, hidden, &mut zb, h, value_dim, batch);
                self.gemm_act(pb, hidden, &mut bb, h, nh_src, batch);
                self.gemm_act(pa, hidden, &mut ab, h, nh_src, batch);
                qkv = Some(qkvb); z = Some(zb); b = Some(bb); a = Some(ab);
            }
        }
        // qkv/z/b/a row geometry: packed (conv_dim / nh*vd / nh) or fused views (mtot + segment offs).
        let (qkv_stride, ba_stride): (usize, usize) = if fused_qkv.is_some() { (mtot, mtot) } else { (conv_dim, nh) };
        let (qkv_ptr, z_ptr): (u64, u64) = match &fused_qkv {
            Some(f) => (d(f), d(f)),
            None => (d(qkv.as_ref().expect("qkv")), d(z.as_ref().expect("z"))),
        };
        let (z_off, z_stride): (usize, usize) = if fused_qkv.is_some() { (conv_dim, mtot) } else { (0, nh*vd) };
        // b/a pointers: packed buffers, or the fused views pre-offset past [qkv|z] plus h0.
        let (b_ptr, a_ptr): (u64, u64) = match &fused_qkv {
            Some(f) => (d(f) + ((conv_dim + value_dim + h0) as u64) * 2,
                        d(f) + ((conv_dim + value_dim + nh_src + h0) as u64) * 2),
            // Split path: b/a are replicated-full — offset to THIS rank's head band (one row
            // per head, bf16). world==1 → rank 0 → offset 0, byte-identical to before.
            None => (d(b.as_ref().expect("b")) + (self.tp_rank as usize * nh * 2) as u64,
                     d(a.as_ref().expect("a")) + (self.tp_rank as usize * nh * 2) as u64),
        };
        blaunch!(self, "conv1d_b", grid(batch*conv_dim), (256,1,1), 0, (qkv_ptr, conv_ptr, d(&la.conv1d), conv_dim as i32, ck as i32, batch as i32, slot_ids_ptr, qkv_stride as i32));
        // NEGATIVE CONTROL (--tp-head-proof-fault=1): deliberately launch the state kernel at the
        // FULL head count — precisely the bug the expert warned about. The output stays CORRECT (out_proj
        // reads only the local value_dim), so the token gate still passes and timing barely moves; only
        // the state red zone can catch it. If this fault does NOT trip the detector, the detector is
        // worthless and the PASS above means nothing.
        // Proof D: per-head visit counter (see delta_step_b). Null outside the proof build.
        let visits_ptr = if self.tp_head_proof() {
            self.head_visits.as_ref().map(|v| *v.device_ptr() as u64).unwrap_or(0)
        } else { 0u64 };
        let fault = self.tp_head_proof() && crate::opts::var(crate::opt!("tp-head-proof-fault")).is_ok();
        let nh_launch = if fault { self.cfg.lin_num_v_heads } else { nh };
        let nk_launch = if fault { self.cfg.lin_num_k_heads } else { self.eff_lin_k_heads() };
        let core = pool.get_bf16(nh_launch*vd*batch);
        let (nchunk, smem) = gdn_launch(kd, vd);
        if self.tp_head_proof() {
            // Proof C. NOTE what this asserts on: `nh_launch`/`nk_launch`, the values ACTUALLY handed to
            // the kernel — not the eff_* accessors. The first version of this check read the accessors,
            // and the injected-fault control sailed straight past it while the kernel launched at full
            // width. A geometry check that does not read the launch argument is theatre.
            static ONCE: std::sync::Once = std::sync::Once::new();
            let (gnh, gnk) = (self.cfg.lin_num_v_heads, self.cfg.lin_num_k_heads);
            ONCE.call_once(|| {
                eprintln!("[head-proof] launch geometry: v_heads {nh_launch} (global {gnh}), k_heads \
                           {nk_launch} (global {gnk}), conv_dim {conv_dim}, value_dim {value_dim}");
            });
            if !fault {
                assert_eq!(nh_launch, gnh / self.gdn_shard_factor(), "delta_step_b launched {nh_launch} value heads, expected local {}", gnh / self.gdn_shard_factor());
                assert_eq!(nk_launch, gnk / self.gdn_shard_factor(), "delta_step_b launched {nk_launch} key heads, expected local {}", gnk / self.gdn_shard_factor());
            }
            assert_eq!(conv_dim, self.eff_conv_dim(), "conv1d_b grid not local");
            assert_eq!(value_dim, self.eff_value_dim(), "rmsnorm_gated_b extent not local");
        }
        let kd_vd = ((kd as u32) << 16) | (vd as u32);
        let stride_pack = ((qkv_stride as u32) << 15) | (ba_stride as u32);
        blaunch!(self, "delta_step_b", ((batch*nh_launch*nchunk) as u32,1,1), (kd as u32,1,1), smem, (d(&core), qkv_ptr, s_ptr, b_ptr, a_ptr, (nh_launch as i32) | ((nk_launch as i32) << 16), kd_vd as i32, d(&la.a_log), d(&la.dt_bias), slot_ids_ptr, visits_ptr, stride_pack as i32));
        if state_tap_active() {
            // GDN internals: the conv-output rows (qkv post-conv1d_b, pitched qkv_stride) and core.
            let cv_src = fused_qkv.as_ref().or(qkv.as_ref());
            if let Some(cvb) = cv_src {
                let cv = self.dev.dtoh_sync_copy(cvb).unwrap_or_default();
                let co = self.dev.dtoh_sync_copy(&core).unwrap_or_default();
                state_tap_gdn(cv, co);
                if batch <= 4 {
                    if let Some(sv) = s_tap {
                        state_tap_s(self.dev.dtoh_sync_copy(sv).unwrap_or_default());
                    }
                }
            }
        }
        let normed = pool.get_bf16(value_dim*batch);
        let z_off_z_stride = ((z_stride as u32) << 15) | (z_off as u32);
        blaunch!(self, self.gdn_gate_kernel(), ((batch*nh) as u32,1,1), (vd as u32,1,1), (vd*4) as u32, (d(&normed), d(&core), z_ptr, d(&la.norm), vd as i32, nh as i32, batch as i32, fbits(cfg.rms_eps), z_off_z_stride as i32));
        let mut out = pool.get_bf16(h*self.pf8_pad(batch));
        // AR landing 2: with `fuse` the row-parallel out_proj reduce may stop after K1 and hand the
        // LOCAL partial up for the fused mixer epilogue (single-barrier only).
        let fused = if self.gdn_shard_factor() > 1 {
            self.tp_reduce_site(pool, &la.out_proj, &normed, &mut out, value_dim, h, batch, fuse)
        } else {
            self.gemm_act(&la.out_proj, &normed, &mut out, value_dim, h, batch);
            None
        };
        // F0: under the default qkv/z/b/a are views into `fused` — one release; packed otherwise.
        if let Some(f) = fused_qkv { pool.release_bf16(f, mtot * batch); }
        if let Some(x) = qkv { pool.release_bf16(x, conv_dim*batch); }
        if let Some(x) = z { pool.release_bf16(x, value_dim*batch); }
        if let Some(x) = b { pool.release_bf16(x, nh*batch); }
        if let Some(x) = a { pool.release_bf16(x, nh*batch); }
        pool.release_bf16(core, nh_launch*vd*batch); pool.release_bf16(normed, value_dim*batch);
        match fused {
            Some(FusedPartial::F32(p)) => ReduceOut::Fused(FusedReduce { sink: out, partial: Some(p), mode: 1 }),
            Some(FusedPartial::F32Scratch) => ReduceOut::Fused(FusedReduce { sink: out, partial: None, mode: 1 }),
            Some(FusedPartial::Bf16) => ReduceOut::Fused(FusedReduce { sink: out, partial: None, mode: 0 }),
            None => ReduceOut::Full(out),
        }
    }

    /// One batched decode step. hidden[hidden,B] bf16. pos_host: per-seq positions.
    pub fn forward_batch(&self, pool: &mut Pool, hidden: B, pos_host: &[usize],
                         state: &mut BatchGpuState, kv_stride: usize, batch: usize) -> B {
        self.forward_batch_core(pool, hidden, pos_host, state, 0, kv_stride, batch)
    }
    /// Prefill/decode a single token into a specific slot (B=1, state offset by `slot`).
    pub fn forward_into_slot(&self, pool: &mut Pool, hidden: B, pos: usize,
                             state: &mut BatchGpuState, slot: usize, kv_stride: usize) -> B {
        self.forward_batch_core(pool, hidden, &[pos], state, slot, kv_stride, 1)
    }
    fn forward_batch_core(&self, pool: &mut Pool, hidden: B, pos_host: &[usize],
                          state: &mut BatchGpuState, slot: usize, kv_stride: usize, batch: usize) -> B {
        let cfg = &self.cfg;
        let rdim = cfg.rotary_dim;
        let max_pc = pos_host.iter().map(|&p| p + 1).max().unwrap_or(1);

        let pos_i32: Vec<i32> = pos_host.iter().map(|&p| p as i32).collect();
        let pos_dev = self.dev.htod_sync_copy(&pos_i32).expect("htod pos");
        let slot_ids: Vec<i32> = (0..batch).map(|i| slot as i32 + i as i32).collect();
        let slot_ids_dev = self.dev.htod_sync_copy(&slot_ids).expect("htod slot_ids");
        let cos = pool.get(batch * rdim);
        let sin = pool.get(batch * rdim);
        self.dev.synchronize().unwrap(); // htod + allocs on NULL stream must complete before non-blocking stream reads
        blaunch!(self, "gather_rope_b", grid(batch*rdim), (256,1,1), 0,
            (d(&cos), d(&sin), d(&self.cos_table), d(&self.sin_table),
             *pos_dev.device_ptr() as u64, rdim as i32, batch as i32));

        let out = self.forward_batch_dev(pool, hidden, &pos_dev, &cos, &sin, max_pc,
                               state, *slot_ids_dev.device_ptr(), kv_stride, batch, 0);
        self.sync_stream(); // ensure compute stream done before pos_dev (non-pool CudaSlice) drops
        pool.release(cos, batch * rdim);
        pool.release(sin, batch * rdim);
        out
    }

    /// Core forward pass using device-side pos/cos/sin (no host syncs). All activations bf16.
    pub(super) fn forward_batch_dev(&self, pool: &mut Pool, hidden: B, pos_dev: &cudarc::driver::CudaSlice<i32>,
                         cos: &S, sin: &S, max_pc: usize,
                         state: &mut BatchGpuState, slot_ids_ptr: u64, kv_stride: usize, batch: usize,
                         tokens_ptr: u64) -> B {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        // qwen4_exp: the residual is the hc-stream stack [rw, batch]; sublayers see the mixed [h, batch].
        let q4 = cfg.is_q4();
        let rw = cfg.resid_width();
        let hcn = cfg.hc_count.max(1);
        let inj: Option<S> = if q4 { Some(pool.get(hcn * batch)) } else { None };

        // Cross-chain probe capture: one sink entry per forward call (batch marks prefill vs
        // decode steps). Inert unless --mxfp4-xchain-capture is set.
        if xchain_active() { xchain_new(batch); }
        // Phase-8 F9 tap: one record per decode call (probe-only; see StateTapCall).
        state_tap_call("decode", batch);
        // Sparse mode (--xchain-ctx-layers="1,20,39,58,77"): record only the listed layers'
        // hiddens — the DFlash context-chain recorder needs 5 of 81 layers; the full prefill
        // capture is ~8 GB of host f32 on a 6K prompt (OOM-thrash), sparse is ~0.5 GB.
        let ctx_layers: Option<Vec<usize>> = if xchain_active() {
            crate::opts::var(crate::opt!("xchain-ctx-layers")).ok()
                .map(|s| s.split(',').filter_map(|x| x.parse::<usize>().ok()).collect())
        } else { None };

        let residual = hidden;
        let normed = pool.get_bf16(h*batch);
        // FFN epilogue fusion (--fuse-residual=1). The MIXER's residual add is already fused into
        // `fused_res_rmsnorm_b`; the FFN's is not — `add_residual_b` then the next layer's `rmsnorm_b`
        // are two kernels doing exactly what that one fused kernel does. Folding them removes ONE kernel
        // per layer (64/token) and hoists the very first input norm out of the loop.
        //
        // It is NOT numerics-neutral, which is why it is a flag: the two-kernel path computes sum_sq from
        // the bf16-ROUNDED residual, while the fused kernel uses the unrounded FP32 sum. That is one
        // fewer rounding — strictly better, and the same reassociation class as the FP32 partials — but
        // it does change output bytes, so the operator makes the call.
        let fuse = self.fuse_residual_norm() && !q4;
        let nlayers = self.layers.len();
        if fuse {
            blaunch_e9!(self, "rmsnorm_b", (batch as u32,1,1), (1024,1,1), (4096) as u32, (d(&normed), d(&residual), d(&self.layers[0].input_ln), h as i32, batch as i32, fbits(cfg.rms_eps)));
        }
        for (li, layer) in self.layers.iter().enumerate() {
            if q4 {
                if let Some(ple) = &layer.ple {
                    self.ple_forward(pool, d(&residual), ple, state, tokens_ptr, slot_ids_ptr, 0, 0, false, batch, None, true);
                }
                self.hc_pre(pool, &normed, inj.as_ref(), d(&residual), &layer.hc.as_ref().unwrap().0, batch);
            } else if !fuse {
                blaunch_e9!(self, "rmsnorm_b", (batch as u32,1,1), (1024,1,1), (4096) as u32, (d(&normed), d(&residual), d(&layer.input_ln), h as i32, batch as i32, fbits(cfg.rms_eps)));
            }
            // Cross-chain probe capture: the normed input feeding this layer's GEMMs.
            if xchain_active() {
                if ctx_layers.as_ref().map_or(true, |l| l.contains(&li)) {
                    xchain_capture(|c| c.layer_inputs.push(xchain_dtoh_f32(self, &normed, h * batch)));
                } else {
                    xchain_capture(|c| c.layer_inputs.push(Vec::new()));
                }
            }
            // AR landing 2: --tp-reduce-fuse — the mixer/FFN epilogue reduces fuse into the norm
            // (K1-only + tp_reduce_resnorm_b) when they are single-barrier; the FFN site needs the
            // fused-residual config too (§4 — its unfused add_residual_b + rmsnorm_b chain has a
            // different rounding placement, so a fused run with the unfused config is not bit-equal).
            let reduce_fuse = Self::tp_reduce_fuse_on();
            let mixer = match layer.layer_type {
                LayerType::LinearAttention => {
                    let conv_ptr = *state.conv_state[li].as_ref().unwrap().device_ptr();
                    let s_ptr = *state.s_state[li].as_ref().unwrap().device_ptr();
                    self.linear_attn_batch(pool, &normed, layer.la.as_ref().unwrap(), conv_ptr, s_ptr, slot_ids_ptr, batch, reduce_fuse,
                        state_tap_active().then(|| state.s_state[li].as_ref().unwrap()))
                }
                LayerType::FullAttention => {
                    let kc_ptr = *state.k_cache[li].as_ref().unwrap().device_ptr();
                    let vc_ptr = *state.v_cache[li].as_ref().unwrap().device_ptr();
                    let qsa = self.qsa_in(state, layer.fa.as_ref().unwrap(), li, kv_stride, *pos_dev.device_ptr() as u64, 0, 0);
                    self.full_attn_batch(pool, &normed, layer.fa.as_ref().unwrap(),
                        &pos_dev, max_pc, kv_stride, kc_ptr, vc_ptr, &cos, &sin, slot_ids_ptr, batch, None,
                        self.attn_shard_factor() > 1, self.kv_mode(), reduce_fuse, qsa)
                }
            };
            if q4 {
                let ReduceOut::Full(m) = mixer else { unreachable!("qwen4_exp: fused TP reduce is not supported") };
                self.hc_post(d(&residual), &m, inj.as_ref().unwrap(), batch);
                pool.release_bf16(m, h * batch);
                self.hc_pre(pool, &normed, inj.as_ref(), d(&residual), &layer.hc.as_ref().unwrap().1, batch);
            } else {
                self.mixer_epilogue(pool, d(&normed), d(&residual), mixer, &layer.post_ln, h, batch, true);
            }
            let mlp_out = self.ffn_batch(pool, &normed, &layer.mlp, batch, self.ffn_shard_factor() > 1, reduce_fuse && fuse);
            if q4 {
                let ReduceOut::Full(m) = mlp_out else { unreachable!("qwen4_exp: fused TP reduce is not supported") };
                self.hc_post(d(&residual), &m, inj.as_ref().unwrap(), batch);
                pool.release_bf16(m, h * batch);
            } else {
            let next_w = if li + 1 < nlayers { &self.layers[li + 1].input_ln } else { &self.final_norm };
            // F5 (EXPERT_FUSION_PASSES_RESPONSE §3.5): the FFN epilogue's flavor-Q kernel is
            // BIT-EXACT to add_residual_b + rmsnorm_b (sum_sq from the ROUNDED residual), so
            // flag-on == flag-off byte-for-byte on the decode path. The mixer-side epilogue stays
            // flavor S (the shipped default there).
            self.ffn_epilogue(pool, d(&normed), d(&residual), mlp_out, fuse, next_w, h, batch, true);
            }
            // Cross-chain probe capture: the residual after this layer's FFN add.
            if xchain_active() {
                if ctx_layers.as_ref().map_or(true, |l| l.contains(&li)) {
                    xchain_capture(|c| c.layer_outputs.push(xchain_dtoh_f32(self, &residual, h * batch)));
                } else {
                    xchain_capture(|c| c.layer_outputs.push(Vec::new()));
                }
            }
            // --layer-csum (diagnostic): prefill-loop twin only — the decode loop's pooled
            // residual rotates between layers and the dtoh here would race its release.
            let _ = li;
            // E29-B3 DFlash tap: the post-FFN-add residual (post-TP-reduce, full-rank) at the
            // tapped layers → rank-0 staging scratch. Device-side 2D D2D on the compute stream,
            // no dtoh; the draft loop copies the columns it needs into the ctx feature buffer.
            // (Only the DFLASH loop calls forward_batch with batch <= BLOCK — wider batches skip.)
            if let Some(tap) = &self.dflash_tap {
                if let Some(k) = tap.tap_index(li) {
                    // P14: the WIDE (absolute-position) ctx surface is a PREFILL-only target from
                    // here on — it is filled by the capture site in `prefill_batch_range` and handed
                    // to the lane by `dflash_prime`. Every other forward (decode here, verify below)
                    // writes the position-INDEPENDENT ring staging, because a decode/verify forward
                    // can run inside a captured CUDA graph whose nodes bake absolute addresses: a
                    // host `wide_pos` read at capture time would freeze the destination column for
                    // every replay. The wide branch is kept only for a capture wider than the ring.
                    if batch <= crate::dflash::MAX_BLOCK {
                        crate::dflash::tap_capture(&self.dev, self.stream.stream, tap,
                                                   *residual.device_ptr() as u64, h, batch, k);
                    } else if let Some(w) = &tap.wide {
                        let p0 = tap.wide_pos.load(std::sync::atomic::Ordering::Relaxed);
                        if p0 + batch <= tap.wide_stride {
                            crate::dflash::tap_capture_to(&self.dev, self.stream.stream,
                                *w.device_ptr() as u64, tap.nctx * h, p0,
                                *residual.device_ptr() as u64, h, batch, k);
                        } else {
                            crate::dflash::warn_wide_skip(p0, batch, tap.wide_stride);
                        }
                    }
                }
            }
            // S4F DFlash2 tap: hidden_states[l+1] for the DFlash2 tap layers [5,19,33,47,61]
            // → the drafter's staging columns [0, n). Same residual pointer, same stream —
            // the D2D is bit-identical to the source by construction (R1 proves it). DEFAULT
            // OFF (df2_capture == None): this branch reads one Option and exits — free.
            if batch <= crate::dflash2::block() {
                if let Some(sink) = &self.df2_capture {
                    if let Some(k) = crate::dflash2::tap_layers().iter().position(|&l| l == li) {
                        crate::dflash2::capture::capture_cols(&self.dev, &self.stream, sink,
                            *residual.device_ptr() as u64, h, hcn, batch, k);
                    }
                }
            }
            // WI1 DSpark tap: identical source (hidden_states[l+1]) at the DSpark tap layers
            // [4,16,28,40,52] → the DSpark round's staging. Independent of the DF2 sink — both
            // arms can be live at once (the acceptance probe's back-to-back arms).
            if batch <= crate::dspark::BLOCK {
                if let Some(sink) = &self.dspark_capture {
                    if let Some(k) = crate::dspark::TAP_LAYERS.iter().position(|&l| (l as i32 + Self::dspark_tap_shift()) == li as i32) {
                        crate::dflash2::capture::capture_cols(&self.dev, &self.stream, sink,
                            *residual.device_ptr() as u64, h, hcn, batch, k);
                    }
                }
            }
            // Phase-8 F9 tap: post-layer residual snapshot (probe-only; syncs the stream).
            if state_tap_active() {
                state_tap_layer(self.dev.dtoh_sync_copy(&residual).unwrap_or_default());
            }
            if crate::opts::var(crate::opt!("f9-xhash")).as_deref() == Ok("2") && batch <= 8 {
                let host = self.dev.dtoh_sync_copy(&residual).unwrap_or_default();
                let mut hs = Vec::with_capacity(batch);
                for c in 0..batch {
                    let mut hv: u64 = 0xcbf29ce484222325;
                    for i in 0..h { let b = host.get(c * h + i).map(|v| v.to_bits()).unwrap_or(0); hv ^= b as u64; hv = hv.wrapping_mul(0x100000001b3); }
                    hs.push(hv);
                }
                F9XHASH.lock().unwrap().push((10u8, F9X_CUR.with(|c| c.get()), li, hs));
            }
        }
        let out = if q4 {
            // The final hyper-connection mixer: streams → one hidden (no final norm on this family).
            let o = pool.get_bf16(h*batch);
            self.hc_pre(pool, &o, None, d(&residual), self.hc_mixer.as_ref().unwrap(), batch);
            pool.release_bf16(normed, h*batch);
            o
        } else if fuse {
            normed          // the last iteration already wrote rmsnorm(residual, final_norm) here
        } else {
            let o = pool.get_bf16(h*batch);
            blaunch!(self, "rmsnorm_b", (batch as u32,1,1), (1024,1,1), (4096) as u32, (d(&o), d(&residual), d(&self.final_norm), h as i32, batch as i32, fbits(cfg.rms_eps)));
            pool.release_bf16(normed, h*batch);
            o
        };
        // Cross-chain probe capture: the post-final-norm hidden.
        if xchain_active() { xchain_capture(|c| c.final_hidden = xchain_dtoh_f32(self, &out, h * batch)); }
        pool.release_bf16(residual, rw*batch);
        if let Some(i) = inj { pool.release(i, hcn * batch); }
        out
    }

    /// FFN residual+norm fusion (see forward_batch_dev). DEFAULT OFF (E14 was flipped ON and
    /// REVERTED 2026-08-09: the verify path does not consult the flag, so decode fused while the
    /// verify didn't — bench_mtp LOSSLESS broke (MTP vs SEQ diverge at the first rounding flip).
    /// The flag remains for the A/B once the verify path gains the same fusion. --fuse-residual=1
    /// enables; the TpConfig tri-state keeps both ranks consistent either way.
    pub(super) fn fuse_residual_norm(&self) -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| crate::opts::var(crate::opt!("fuse-residual")).is_ok())
    }

    pub fn logits_batch(&self, pool: &mut Pool, hidden: &B, batch: usize) -> B {
        let (v, h) = (self.cfg.vocab_size, self.cfg.hidden_size);
        // TP=2 vocab-parallel head: compute THIS rank's half of the vocab rows, place it at
        // [rank*v/2) of every batch column of a ZEROED full buffer (logits layout C[n*v + m]), then
        // sum all-reduce. The peer's rows here are exact +0.0 (and vice versa), so the elementwise
        // sum IS the gather: bitwise-identical to the replicated head's full logits, never a
        // reassociation. The memset is explicit — pool reuse does NOT re-zero (see Pool), and a
        // stale value where a zero is required would corrupt the sum. One extra all-reduce per
        // main-head forward; its barrier count is a pure function of (v, batch, payload), identical
        // on both ranks by construction (SPMD). Non-TP / tied / unsharded paths are untouched.
        if self.tp_world > 1 && self.lm_head_sharded {
            if let Some(w) = &self.lm_head {
                let hv = v / self.shard_factor();
                let mut half = pool.get_bf16(hv * batch);
                self.gemm_act(w, hidden, &mut half, h, hv, batch);
                let mut logits = pool.get_bf16(v * batch);
                let full = *logits.device_ptr() as u64;
                self.memset_compute_stream(full, v * batch * 2);
                let rank = self.tp_rank.max(0) as usize;
                // Strided D2D: per batch column n, copy half[n*hv .. +hv] to
                // logits[n*v + rank*hv .. +hv]. One 2D copy, queued on the compute stream.
                let cp = cudarc::driver::sys::CUDA_MEMCPY2D {
                    srcXInBytes: 0, srcY: 0,
                    srcMemoryType: cudarc::driver::sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                    srcHost: std::ptr::null(), srcDevice: *half.device_ptr() as u64,
                    srcArray: std::ptr::null_mut(), srcPitch: hv * 2,
                    dstXInBytes: rank * hv * 2, dstY: 0,
                    dstMemoryType: cudarc::driver::sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                    dstHost: std::ptr::null_mut(), dstDevice: full,
                    dstArray: std::ptr::null_mut(), dstPitch: v * 2,
                    WidthInBytes: hv * 2, Height: batch,
                };
                unsafe {
                    let r = cudarc::driver::sys::cuMemcpy2DAsync_v2(&cp, self.stream.stream);
                    assert!(r == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                            "lm_head gather 2D copy failed: {r:?}");
                }
                pool.release_bf16(half, hv * batch);
                self.tp_all_reduce_bf16(&mut logits, v * batch);
                return logits;
            }
        }
        let mut logits = pool.get_bf16(v*batch);
        let w = self.lm_head.as_ref().unwrap_or(&self.embed);
        self.gemm_act(w, hidden, &mut logits, h, v, batch);
        logits
    }

    /// fp32-logits variant for `cfg.lm_head_fp32` (hy_v3). The logits GEMM accumulates fp32 on every
    /// path already; `enable_lm_head_fp32` is exactly "do not round the OUTPUT to bf16" — so this runs
    /// `gemm_binv_f32_b`, the f32-output twin of `gemm_binv_b` (identical strided-k + fixed tree-reduce,
    /// so it is batch-invariant for the same reason). Consumers are the f32 kernel twins
    /// (`rep_penalty_f32_b`, `argmax_f32_b`, `sample_f32_b`, `spec_verify_f32_b`) or a host read —
    /// greedy, stochastic decode AND stochastic MTP verify are all f32 end-to-end (E21).
    ///
    /// Remaining bf16 consumers by design: the NLL/PPL probe (`nll_b` — eval-only). Wide batches
    /// (> MAX_VERIFY) also round-trip through bf16 below — the cuBLAS arm of `gemm_act` has no
    /// f32-output twin; decode/verify logits never get that wide.
    pub fn logits_batch_f32(&self, pool: &mut Pool, hidden: &B, batch: usize) -> S {
        let (v, h) = (self.cfg.vocab_size, self.cfg.hidden_size);
        // TP=2 vocab-parallel BF16 head (E7), fp32-logits contract preserved end to end: half-rows
        // gemm into a ZEROED full f32 buffer, then the no-rounding f32 all-reduce (tp_wait_add mode
        // 2). The peer's rows here are exact +0.0 (and vice versa), so the sum IS the gather — the
        // values the replicated head produced, never a bf16 round. The memset is explicit (pool
        // reuse does not re-zero); the 2D copy places half[n*hv .. +hv] at logits[n*v + rank*hv].
        // Barrier count is a pure function of (v, batch, chunk) — SPMD-safe; capture-safe (the
        // mode is a constant per call site). Wide batches keep the documented bf16 fallback below.
        if self.tp_world > 1 && self.lm_head_sharded && batch <= MAX_VERIFY {
            if let Some(W::Bf16(wb)) = &self.lm_head {
                let hv = v / self.shard_factor();
                let mut half = pool.get(hv * batch);
                let smem = (batch * 256 * 4) as u32;
                blaunch!(self, "gemm_binv_f32_b", (hv as u32,1,1), (256,1,1), smem,
                    (d(&half), d(wb), d(hidden), hv as i32, h as i32, batch as i32));
                let mut logits = pool.get(v * batch);
                let full = *logits.device_ptr() as u64;
                self.memset_compute_stream(full, v * batch * 4);
                let rank = self.tp_rank.max(0) as usize;
                let cp = cudarc::driver::sys::CUDA_MEMCPY2D {
                    srcXInBytes: 0, srcY: 0,
                    srcMemoryType: cudarc::driver::sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                    srcHost: std::ptr::null(), srcDevice: *half.device_ptr() as u64,
                    srcArray: std::ptr::null_mut(), srcPitch: hv * 4,
                    dstXInBytes: rank * hv * 4, dstY: 0,
                    dstMemoryType: cudarc::driver::sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                    dstHost: std::ptr::null_mut(), dstDevice: full,
                    dstArray: std::ptr::null_mut(), dstPitch: v * 4,
                    WidthInBytes: hv * 4, Height: batch,
                };
                unsafe {
                    let r = cudarc::driver::sys::cuMemcpy2DAsync_v2(&cp, self.stream.stream);
                    assert!(r == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                            "lm_head f32 gather 2D copy failed: {r:?}");
                }
                pool.release(half, hv * batch);
                self.tp_all_reduce_f32(&mut logits, v * batch);
                return logits;
            }
        }
        let mut logits = pool.get(v*batch);
        let w = self.lm_head.as_ref().unwrap_or(&self.embed);
        match w {
            W::Bf16(wb) if batch <= MAX_VERIFY => {
                let smem = (batch * 256 * 4) as u32;
                blaunch!(self, "gemm_binv_f32_b", (v as u32,1,1), (256,1,1), smem,
                    (d(&logits), d(wb), d(hidden), v as i32, h as i32, batch as i32));
            }
            _ => {
                let tmp = self.logits_batch(pool, hidden, batch);
                unsafe { self.k.bf16tof32.clone().unwrap().launch_on_stream(&self.stream,
                    LaunchConfig { grid_dim: grid(v*batch), block_dim: (256,1,1), shared_mem_bytes: 0 },
                    (d(&logits), d(&tmp), (v*batch) as i32)).unwrap(); }
                pool.release_bf16(tmp, v*batch);
            }
        }
        logits
    }

    // ===================== OPTIMIZED DECODE (pre-computed everything) =====================

    // ===================== LOOPFIX: whole-output penalty state =====================
}
