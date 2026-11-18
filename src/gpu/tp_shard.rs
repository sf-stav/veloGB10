//! `impl GpuModel`: TP attach (`attach_tp`), post-load weight sharding (`tp_shard_weights`, MTP head, per-format `shard_*`
//! helpers), rank weight dump and the NaN redzone checks. Host twins for shard-at-load are in `shard_ops.rs`. Map: `mod.rs`.

use super::*;

impl GpuModel {
    /// Attach the TP=2 data-plane link + this process's rank/world after the cluster sync + RDMA
    /// bring-up, then SHARD the weights so each box streams ~half the bytes/token (the decode speedup).
    /// Decode is weight-bandwidth-bound, so halving bytes-read is the win; the per-reduction all-reduce
    /// (`tp_all_reduce_bf16`) stitches the row-parallel partials back together.
    pub fn attach_tp(&mut self, rank: i32, world: i32, mut link: crate::net::TpLink) {
        self.tp_rank = rank;
        self.tp_world = world;
        if world > 1 {
            let t0 = std::time::Instant::now();
            if self.tp_sharded_at_load {
                // hy_v3: weights were sliced host-side in the loader (the full model never fit one
                // node for the in-place shard). The eff_* head counts key off tp_shard_mixers() —
                // it MUST agree with what the loader did, or every attention/KV shape is wrong.
                assert!(self.tp_shard_mixers(),
                        "weights were TP-sharded at load but mixer sharding is OFF \
                         ([tp-shard-mixers] / TpConfig.shard_mixers) — head counts would disagree \
                         with the sharded tensors. Refusing to attach.");
                eprintln!("[tp] rank {rank}/{world} — weights PRE-SHARDED at load (hy_v3); in-place shard skipped");
            } else {
                assert!(self.cfg.family != crate::qwen::Family::HyV3,
                        "hy_v3 under TP=2 must be loaded with `load_from_dir_tp` (shard-at-load): \
                         the full 169.6 GB cannot be resident on one node for the in-place shard. \
                         Refusing to attach.");
                self.tp_shard_weights(rank as usize, world as usize);
                eprintln!("[tp] rank {rank}/{world} — weights sharded (FFN col/row-parallel) in {:.1}s", t0.elapsed().as_secs_f32());
            }
            // §4.1: the MTP block shards at attach on BOTH load paths (it always arrives whole —
            // the load op maps keep mtp.* replicated). Independent of the backbone branches above.
            if self.tp_shard_mtp_on() {
                self.tp_shard_mtp_weights(rank as usize, world as usize);
            }
            // The DEFAULT payload is the batch=1 (or probe-batch) hidden vector — what decode/barrier
            // benches ship; chunked prefill barriers pass a per-call length that nearly fills a slot.
            // Set it BEFORE the proxy starts — both the proxy and K1/K2 read it, and I8 forbids mutating
            // protocol state underneath a running system.
            let fp32 = self.tp_fp32_partials();
            // --tp-batch-probe=N widens the payload so a batch-N forward can all-reduce
            // hidden*N. Set once, before the proxy starts (I8).
            let nbytes = self.cfg.hidden_size * self.tp_probe_batch() * if fp32 { 4 } else { 2 };
            // Serving mode sets batch_probe = max_batch, so this is also the batched-decode reduce
            // payload. The slot size is baked into captured CUDA graphs — an oversize payload cannot
            // be patched later, so refuse LOUDLY at attach rather than corrupt a ring at runtime.
            // (bf16 hidden=5120 fits max_batch <= 12; fp32 partials halve that.)
            assert!(nbytes <= crate::tp::TP_SLOT_BYTES,
                    "FATAL: TP all-reduce payload {nbytes} B exceeds TP_SLOT_BYTES ({} B) — lower \
                     --max-batch{} (hidden {}, probe batch {})",
                    crate::tp::TP_SLOT_BYTES,
                    if fp32 { " or disable FP32 partials" } else { "" },
                    self.cfg.hidden_size, self.tp_probe_batch());
            link.set_payload(nbytes, fp32).expect("net_set_payload");
            if fp32 { eprintln!("[tp] FP32-PRESERVING partials ({nbytes} B/reduction, one bf16 round in K2)"); }
            // v2 receive mode (EXPERT_GPU_ALLREDUCE §8): GPU-direct tail validation instead of the
            // CPU bounce. Resolved once per process (env-first, then the shipped config, default OFF
            // until the §12 battery passes); both ranks resolve identically (SPMD). Set before the
            // proxy starts (I8).
            if self.tp_gpu_recv_on() {
                link.set_recv_mode(true).expect("net_set_recv_mode");
                eprintln!("[tp] v2 GPU-direct receive ON (K2' tail gate; --tp-gpu-recv)");
            }
            self.tp_ctx_dptr = link.ctx_device_ptr();
            // Hand the RDMA ctx to the persistent proxy thread. The proxy OWNS the transport from here,
            // so we `mem::forget` the TpLink (its Drop would net_shutdown the ctx out from under the
            // proxy); the OS reclaims at exit. Core 19 is a big X925 and pairs with the launch thread on
            // core 9 — pinning was worth 9.0→15.1 tok/s on 27B, so a failure to pin is loud.
            // --tp-trace=1 turns on the same per-barrier timestamping the microbench uses, so a slow
            // model run can be decomposed against the bench's floor instead of guessed at.
            if crate::opts::var(crate::opt!("tp-trace")).is_ok()
                || crate::tp::tp_config().map(|c| c.trace).unwrap_or(false) {
                crate::net::trace_enable(&mut link);
                eprintln!("[tp] per-barrier tracing ON (--tp-trace)");
            }
            if self.tp_head_proof() {
                self.head_visits = Some(self.dev.alloc_zeros::<u64>(self.cfg.lin_num_v_heads).unwrap());
            }
            let ctx_addr = link.ctx_addr();
            std::mem::forget(link);
            crate::net::spawn_proxy(ctx_addr, 19);
            crate::deprintln!("[tp] rank {rank}/{world} — RDMA proxy thread up (doorbell ring R={}, {nbytes} B/reduction)", 8);
            self.tp_align_rounds();
        }
    }

    /// R10: align the barrier epoch to a multiple of `rounds` (world>2). Every logical all-reduce
    /// consumes an exact multiple of `rounds` epochs, so the doubling order (round(e) = e % rounds)
    /// is otherwise FROZEN at whatever parity bring-up left — and different bring-up paths (MTP
    /// calibration on/off, session history) leave different parities. The parity picks the
    /// summation order ((p0+p1)+(p2+p3) vs (p0+p2)+(p1+p3)), which changes the fp rounding, which
    /// flips near-tie argmaxes: the MTP lossless gate (on/off) and cross-session determinism both
    /// need ONE fixed order. One pad at attach makes every process start at parity 0; the multiple-
    /// of-rounds invariant then holds for the process life (graph captures advance nothing, replays
    /// advance by the captured count). The pad is a tiny zero-effect exchange on the f32 scratch.
    fn tp_align_rounds(&self) {
        if self.tp_world <= 2 { return; }
        self.sync_stream();
        let rounds = self.tp_world.trailing_zeros() as usize;
        let e = crate::net::traced_device_epoch() as usize;
        let pad = (rounds - (e % rounds)) % rounds;
        if pad == 0 {
            eprintln!("[tp] R10 round order: epoch {e} already % {rounds} == 0");
            return;
        }
        let sp = self.tp_f32_scratch_ptr();
        let k2 = if self.tp_gpu_recv_on() { "tp_wait_add_g" } else { "tp_wait_add" };
        for _ in 0..pad {
            blaunch!(self, "tp_gate_copy_signal", (1u32,1,1), (1024,1,1), 0, (self.tp_ctx_dptr, sp, 16u32));
            blaunch!(self, k2, (1u32,1,1), (1024,1,1), 0, (self.tp_ctx_dptr, sp, sp, 4i32, 2i32));
        }
        self.sync_stream();
        eprintln!("[tp] R10 round order aligned: epoch {e} + {pad} pad barrier(s) (world={})", self.tp_world);
    }

    /// Split the weights for TP (called once, at attach), per tensor class. Dense FFN gate/up are
    /// COLUMN-parallel (each rank owns `1/factor` of the output rows), down is ROW-parallel (each rank
    /// owns `1/factor` of K). Attention and GDN mixers shard only when `tp_shard_mixers()` AND the
    /// head counts divide `world`; MoE experts shard when `num_experts % world == 0`. Anything
    /// non-divisible stays REPLICATED (factor 1) — the class is simply skipped here, leaving its
    /// weights whole and its `*_sharded` flags false. No kernel change — a sharded weight is just
    /// smaller, so the proven `gemm_mma_fp4_b` reads fewer bytes.
    fn tp_shard_weights(&mut self, rank: usize, world: usize) {
        let ffn_factor = self.ffn_shard_factor();
        let attn_factor = self.attn_shard_factor();
        let gdn_factor = self.gdn_shard_factor();
        let expert_factor = self.experts_shard_factor();
        let vocab_factor = self.vocab_shard_factor();
        let (nh, nkv, hd) = (self.cfg.num_heads, self.cfg.num_kv_heads, self.cfg.head_dim);
        // Fused attention qkv output segments (per split_qkv_b): [qg | k | v]. Each is head-split.
        // qg is nh*hd*2 for qwen3_5 (q|gate) and nh*hd for hy_v3 (no output gate).
        let qmul = if self.cfg.attn_out_gate() { 2 } else { 1 };
        let qkv_segs = [(0, nh * hd * qmul, true), (nh * hd * qmul, nkv * hd, true), (nh * hd * qmul + nkv * hd, nkv * hd, true)];
        // Fused GDN in_proj output segments (per split_gdn_b / GdnIn::fused_m):
        //   [ qkv (= q key_dim | k key_dim | v value_dim) | z value_dim | b n_v | a n_v ]
        // Six independently head-split segments. q/k split on KEY heads, v/z/b/a on VALUE heads — the
        // 48/16 = 3 value-heads-per-key-head grouping survives because both splits are contiguous.
        let (lin_nk, lin_nv) = (self.cfg.lin_num_k_heads, self.cfg.lin_num_v_heads);
        let (kdim, vdim) = (self.cfg.key_dim(), self.cfg.value_dim());
        let (cdim, ck) = (kdim * 2 + vdim, self.cfg.conv_kernel);
        // b/a (one row per value head) are carried WHOLE (split=false): 48 rows will not split at
        // NVFP4's 16-row tile granularity, so every rank keeps all of them and split_gdn_b slices them
        // to the local head range. They must still be PRESENT in the weight or the fused GEMM's M and
        // the tensor disagree — which is exactly the bug this flag was added to fix.
        let gdn_segs = [(0, kdim, true), (kdim, kdim, true), (2 * kdim, vdim, true), (cdim, vdim, true),
                        (cdim + vdim, lin_nv, false), (cdim + vdim + lin_nv, lin_nv, false)];
        // conv1d is [conv_dim][conv_kernel] — slice the same q/k/v channel segments.
        let conv_segs = [(0, kdim), (kdim, kdim), (2 * kdim, vdim)];
        let _ = lin_nk;
        let mut layers = std::mem::take(&mut self.layers);
        for layer in &mut layers {
            // FFN: gate/up column-parallel, down row-parallel (dense only; NOT gated on mixers).
            if ffn_factor > 1 {
                if let Ffn::Dense(mlp) = &mut layer.mlp {
                    let rg = self.take_rotation_marker(&mlp.gate);
                    let gate = self.shard_mxfp4_col(&mlp.gate, rank, world);
                    self.restore_rotation_marker(rg, &gate);
                    mlp.gate = gate;
                    let ru = self.take_rotation_marker(&mlp.up);
                    let up = self.shard_mxfp4_col(&mlp.up, rank, world);
                    self.restore_rotation_marker(ru, &up);
                    mlp.up = up;
                    let rd = self.take_rotation_marker(&mlp.down);
                    let down = self.shard_mxfp4_row(&mlp.down, rank, world);
                    self.restore_rotation_marker(rd, &down);
                    mlp.down = down;
                }
            }
            // MoE EXPERT sharding: the stacked experts are expert-major along M — expert e's 16-row
            // NVFP4 tiles sit at e*(M/16) (gpu_batch.cu gemm_moe_mma_fp4) — and each expert's M
            // (2·mi gate_up rows, h down rows) is a multiple of 16, so a plain COLUMN-parallel shard
            // of the flattened [ne·M_e, K] weight IS the contiguous expert band [rank·ne/world, +ne/world).
            // Router/shared/shared_gate stay replicated. The MTP layer's MoE is deliberately NOT sharded
            // (tp_shard_weights touches only self.layers): the draft head must stay barrier-free.
            if expert_factor > 1 {
                if let Ffn::Moe(moe) = &mut layer.mlp {
                    let rg = self.take_rotation_marker(&moe.gate_up);
                    let gate_up = self.shard_mxfp4_col(&moe.gate_up, rank, world);
                    self.restore_rotation_marker(rg, &gate_up);
                    moe.gate_up = gate_up;
                    let rd = self.take_rotation_marker(&moe.down);
                    let down = self.shard_mxfp4_col(&moe.down, rank, world);
                    self.restore_rotation_marker(rd, &down);
                    moe.down = down;
                    moe.experts_sharded = true;
                    // E8 (revised): shared expert — DOWN row-parallel (each rank computes 1/world of
                    // the shared output; moe_batch folds the partial into the routed all-reduce).
                    // The FUSED quantized shared gate+up stays REPLICATED on THIS path (qwen
                    // attach): a PLAIN col-shard of [2·si, h] would split the gate/up pairing the
                    // SiLU needs (gate[i]·up[i] must pair full rows). hy_v3's shard-at-load DOES
                    // split it via a PAIRED ColSegs split — the pairings stay on one rank; the qwen
                    // attach leaves it replicated because qwen's shared gate/up are separate tensors.
                    if self.shared_expert_shard_factor() > 1
                        && matches!(moe.shared_gate_up, Some(W::Nvfp4 { .. }) | Some(W::Fp8 { .. })) {
                        moe.shared.down = self.shard_mxfp4_row(&moe.shared.down, rank, world);
                        moe.shared_sharded = true;
                    }
                }
            }
            // Attention (mixer-sharding + divisibility): fused qkv column-parallel (per-head, 3 segments),
            // o_proj row-parallel. q_norm/k_norm are shared [head_dim] (rmsnorm_perhead reads w[tid]) → NOT sharded.
            if attn_factor > 1 {
                if let Some(fa) = &mut layer.fa {
                    if let AttnIn::Fused(w) = &fa.qkv {
                        let rotated = self.take_rotation_marker(w);
                        let sharded = self.shard_mxfp4_col_segs(w, &qkv_segs, rank, world);
                        self.restore_rotation_marker(rotated, &sharded);
                        fa.qkv = AttnIn::Fused(sharded);
                    }
                    let rotated = self.take_rotation_marker(&fa.o_proj);
                    let sharded = self.shard_mxfp4_row(&fa.o_proj, rank, world);
                    self.restore_rotation_marker(rotated, &sharded);
                    fa.o_proj = sharded;
                }
            }
            // GDN (head-parallel + divisibility): in_proj column-parallel per segment, conv/a_log/
            // dt_bias sliced to the local heads, out_proj row-parallel → partial → all-reduce.
            // `norm` is [v_head_dim], shared across heads (rmsnorm_gated_b indexes it by lane) → NOT sharded.
            if gdn_factor > 1 {
                if let Some(la) = &mut layer.la {
                    if let GdnIn::Fused(w) = &la.in_proj {
                        let rotated = self.take_rotation_marker(w);
                        let sharded = self.shard_mxfp4_col_segs(w, &gdn_segs, rank, world);
                        self.restore_rotation_marker(rotated, &sharded);
                        la.in_proj = GdnIn::Fused(sharded);
                    } else {
                        panic!("tp GDN shard: expected a fused (quantized) in_proj; the Split path is unsharded");
                    }
                    la.conv1d  = self.shard_f32_segs(&la.conv1d, &conv_segs, ck, rank, world);
                    la.a_log   = self.shard_f32_segs(&la.a_log, &[(0, lin_nv)], 1, rank, world);
                    la.dt_bias = self.shard_f32_segs(&la.dt_bias, &[(0, lin_nv)], 1, rank, world);
                    if self.tp_head_proof() {
                        // Proof A: our convention is slice+local, so a GLOBAL index (head + h0) would
                        // read past the local range. Pad to full width with NaN so that read is loud.
                        la.a_log   = self.pad_nan_redzone(&la.a_log, lin_nv);
                        la.dt_bias = self.pad_nan_redzone(&la.dt_bias, lin_nv);
                    }
                    let rotated = self.take_rotation_marker(&la.out_proj);
                    let sharded = self.shard_mxfp4_row(&la.out_proj, rank, world);
                    self.restore_rotation_marker(rotated, &sharded);
                    la.out_proj = sharded;
                }
            }
        }
        self.layers = layers;

        // LM-head vocab-parallel (quantized heads only): the head is the largest REPLICATED tensor
        // left on the flagships. Shard it exactly like the FFN gate/up col-shard — rank r keeps vocab
        // rows [r*v/world, +v/world), a contiguous 16-row tile band, byte-exact — and `logits_batch`
        // gathers the slices with one all-reduce. Guards: tied embeddings keep lm_head = None, and a
        // head that is neither NVFP4 nor FP8 stays replicated (bf16 is not the serving path).
        if vocab_factor > 1 {
            let m = match &self.lm_head {
                Some(W::Nvfp4 { m, .. }) | Some(W::Fp8 { m, .. }) => *m,
                _ => 0,
            };
            if m > 0 {
                // S9F (TP-DF2 leg): keep the FULL head for the DFlash2 round BEFORE the
                // vocab-parallel shard slices it. [df2-tp]=1 is set by the head when its
                // spec-source is a DF2 variant and installed by the node from the shipped
                // config — the capture is a kept handle (CudaSlice clone shares the device
                // allocation; the shard allocates its own sliced tensor), so it costs no
                // extra bytes, and the round's draft head GEMM needs all v rows.
                if crate::opts::var(crate::opt!("df2-tp")).is_ok() && self.df2_full_head.is_none() {
                    let full = match self.lm_head.as_ref().unwrap() {
                        W::Nvfp4 { qweight, scales, gs, m, k } => W::Nvfp4 {
                            qweight: qweight.clone(), scales: scales.clone(), gs: gs.clone(),
                            m: *m, k: *k },
                        W::Fp8 { data, row_scale, m, k } => W::Fp8 {
                            data: data.clone(), row_scale: row_scale.clone(), m: *m, k: *k },
                        other => panic!("df2 full-head capture: lm_head must be quantized (got a non-quantized weight)"),
                    };
                    crate::deprintln!("[tp] rank {rank} — DFlash2 full lm_head kept for the round ({} x {} rows)",
                              m, self.cfg.hidden_size);
                    self.df2_full_head = Some(full);
                }
                let lh = self.lm_head.take().unwrap();
                let rotated = self.take_rotation_marker(&lh);
                let full_head_ptr = self.df2_full_head.as_ref().and_then(|w| match w {
                    W::Nvfp4 { qweight, .. } | W::Nvfp4Raw { qweight, .. } => Some(*qweight.device_ptr() as u64),
                    _ => None,
                });
                if rotated {
                    if let Some(ptr) = full_head_ptr { self.rotated.insert(ptr); }
                }
                let sharded = self.shard_mxfp4_col_segs(&lh, &[(0, m, true)], rank, world);
                self.restore_rotation_marker(rotated, &sharded);
                self.lm_head = Some(sharded);
                self.lm_head_sharded = true;
                eprintln!("[tp] rank {rank} — lm_head vocab-sharded ({m} -> {} rows/rank)", m / world);
            }
        }
    }

    /// TP4-campaign §4.1: shard the MTP block. The draft head gives up its "barrier-free" property
    /// — each draft forward now contributes the same reduce-site pattern a main layer has — and in
    /// exchange stops reading the full ~0.23 GB (27B dense) / ~0.45 GB (35B MoE) block per rank per
    /// draft token. Runs at attach (AFTER the backbone's shard-at-load or in-place shard) on every
    /// rank identically; the layout is a pure function of (rank, world, format), so both load paths
    /// agree. Per-tensor guards mirror tp_shard_weights: only quantized tensors of divisible shapes
    /// shard, anything else stays replicated with its flag false.
    fn tp_shard_mtp_weights(&mut self, rank: usize, world: usize) {
        // REGRESSION FIX (2026-09-11): F6 (3cadc14) enabled Fp8Blk sharding of the MTP head.
        // Under TP the sharded draft path is numerically broken: draft-1 acceptance fell
        // ~85%->~52% and the chained drafts are ALWAYS rejected (@2: 0% over 500+ steps),
        // collapsing TP2/TP4 MTP serving (owner-measured 41.5->19.8 tok/s at TP2, 56.2->30.4
        // at TP4; policy correctly settles at depth 2). Every correctness gate stayed green
        // because verify rejects the garbage drafts (greedy stays lossless) and step-time
        // instruments cannot see emission — bisected 918d7de(clean)/9705dc2(clean)/
        // 3cadc14(broken), and --tp-shard-mtp=0 on 74a4611 restores full acceptance.
        // The MTP head stays REPLICATED (the pre-F6 shipping behavior, 0.444 GB/rank) until
        // the sharded draft path is fixed AND gated by a TP acceptance probe, not LOSSLESS
        // alone.
        let quant = |w: &W| matches!(w, W::Nvfp4 { .. } | W::Fp8 { .. });
        // TEMP bisect knob (diagnostics only): enable only the named shard part.
        let only = crate::opts::var(crate::opt!("dbg-mtp-only")).ok();
        let want = |p: &str| match only.as_deref() {
            None => true,
            Some("attn_qkv") | Some("attn_oproj") => p == "attn",   // attn sub-knobs
            Some(o) => o == p,
        };
        let attn_f = self.attn_shard_factor();
        let ffn_f = self.ffn_shard_factor();
        let expert_f = self.experts_shard_factor();
        let h = self.cfg.hidden_size;
        let mut mtp = match self.mtp.take() { Some(m) => m, None => return };
        // fc [h, 2h] — COLUMN-parallel m-slice (NOT row-parallel: the fc input concat is full-width
        // and replicated, and a k-slice would need a strided input view the GEMMs don't take; the
        // m-slice instead computes each output element with the IDENTICAL full-k reduction, so the
        // zero+2D-copy+all-reduce gather in mtp_forward is bitwise-equal to the replicated GEMM).
        if want("fc") && quant(&mtp.fc) && h % (16 * world) == 0 {
            mtp.fc = self.shard_mxfp4_col(&mtp.fc, rank, world);
            mtp.fc_sharded = true;
        }
        // Attention: exactly the main layers' qkv col-segs + o_proj row split. Divisibility is
        // attn_shard_factor's (mixers-gated + head counts divide) — the 35B's kv_heads=2 at world=4
        // keeps its MTP attention replicated here, same as its main attention.
        if want("attn") && attn_f > 1 {
            let (nh, nkv, hd) = (self.cfg.num_heads, self.cfg.num_kv_heads, self.cfg.head_dim);
            let qmul = if self.cfg.attn_out_gate() { 2 } else { 1 };
            let qkv_segs = [(0, nh * hd * qmul, true), (nh * hd * qmul, nkv * hd, true),
                            (nh * hd * qmul + nkv * hd, nkv * hd, true)];
            // sub-knobs (diagnostics): "attn_qkv" shards only the fused qkv (full-width KV heads,
            // o_proj replicated); "attn_oproj" only o_proj + head-parallel attention path.
            let sub = crate::opts::var(crate::opt!("dbg-mtp-only")).ok();
            let do_qkv = sub.as_deref() != Some("attn_oproj");
            let do_oproj = sub.as_deref() != Some("attn_qkv");
            if let AttnIn::Fused(w) = &mtp.fa.qkv {
                if quant(w) {
                    if do_qkv {
                        mtp.fa.qkv = AttnIn::Fused(self.shard_mxfp4_col_segs(w, &qkv_segs, rank, world));
                    }
                    if do_oproj {
                        mtp.fa.o_proj = self.shard_mxfp4_row(&mtp.fa.o_proj, rank, world);
                    }
                    mtp.attn_sharded = true;
                }
            }
        }
        // FFN: dense = the main layers' col/col/row split (ffn_batch(sharded=true) reduces the
        // down partial); MoE = the expert band + experts_sharded (moe_batch rebases the router's
        // global ids and all-reduces). The MoE router/shared stay replicated (§4.3/§4.4 of the
        // campaign — measured too small / separately targeted).
        match &mut mtp.mlp {
            Ffn::Dense(mlp) if want("mlp") && ffn_f > 1 && quant(&mlp.gate) && quant(&mlp.up) && quant(&mlp.down) => {
                mlp.gate = self.shard_mxfp4_col(&mlp.gate, rank, world);
                mlp.up   = self.shard_mxfp4_col(&mlp.up, rank, world);
                mlp.down = self.shard_mxfp4_row(&mlp.down, rank, world);
                mtp.mlp_sharded = true;
            }
            Ffn::Moe(moe) if want("mlp") && expert_f > 1 && quant(&moe.gate_up) && quant(&moe.down) => {
                moe.gate_up = self.shard_mxfp4_col(&moe.gate_up, rank, world);
                moe.down    = self.shard_mxfp4_col(&moe.down, rank, world);
                moe.experts_sharded = true;
            }
            _ => {}
        }
        let fc_sharded = mtp.fc_sharded; let attn_sharded = mtp.attn_sharded;
        let mlp_sharded = mtp.mlp_sharded || matches!(&mtp.mlp, Ffn::Moe(m) if m.experts_sharded);
        self.mtp = Some(mtp);
        if fc_sharded || attn_sharded || mlp_sharded {
            eprintln!("[tp] rank {rank}/{world} — MTP block sharded (fc={fc_sharded} attn={attn_sharded} mlp={mlp_sharded}; \
                       the draft path now carries reduce sites)");
        }
    }
    /// Whether the MTP block sharding is on (head's --tp-shard-mtp; --tp-shard-mtp alias;
    /// TpConfig to the nodes — a one-sided shard is a weight-layout + barrier-sequence mismatch).
    /// Phase-8 fix: the env term used `is_ok()`, so `--tp-shard-mtp=0` read as ON — the head
    /// sharded its MTP weights while TpConfig told the nodes `false`, the exact one-sided shard
    /// this comment warns about (observed: boot-time graph-capture tripwire, device epoch 152
    /// ahead of gpu_ready, 3 failed boots). The config field (tp.rs parses `0` correctly) is now
    /// authoritative; the env remains a symmetric override that must EXPLICITLY enable.
    fn tp_shard_mtp_on(&self) -> bool {
        self.tp_world > 1
            && (crate::tp::tp_config().map(|c| c.shard_mtp).unwrap_or(false)
                || crate::opts::var(crate::opt!("tp-shard-mtp")).map_or(false, |v| v != "0"))
    }
    /// MTP attention head-sharded (drives the MTP KV-cache width — every alloc site must use this).
    fn mtp_attn_sharded(&self) -> bool {
        self.mtp.as_ref().map_or(false, |m| m.attn_sharded)
    }
    /// KV heads the MTP draft cache holds on THIS rank.
    pub fn mtp_kv_heads(&self) -> usize {
        if self.mtp_attn_sharded() { self.eff_num_kv_heads() } else { self.cfg.num_kv_heads }
    }

    /// Weight-layout twin of `attach_tp` for offline auditing (`--dump-rank-weights`): applies
    /// exactly the sharding a real TP rank of this world holds — the shard-at-load ops ran inside
    /// `load_from_dir_tp`, the in-place `tp_shard_weights` runs here — WITHOUT the RDMA link/proxy
    /// (the transport never touches weights, so the layout is byte-identical to a live run).
    /// Set [tp-shard-mixers] (etc.) in the env to match the live configuration being audited.
    pub fn prepare_tp_weight_layout(&mut self, rank: i32, world: i32) {
        self.tp_rank = rank;
        self.tp_world = world;
        if world > 1 {
            if !self.tp_sharded_at_load {
                self.tp_shard_weights(rank as usize, world as usize);
            }
            if self.tp_shard_mtp_on() {
                self.tp_shard_mtp_weights(rank as usize, world as usize);
            }
        }
    }

    /// Per-rank weight accounting (TP4 campaign, Step 0): walk every weight tensor resident on
    /// THIS rank and print recipe / dims / bytes / sharded-or-replicated, aggregated by site.
    /// The tail tables are the numbers the memory-design rule targets: (a) whole-model tensors
    /// still replicated on this rank, ranked by bytes, and (b) total vs per-forward streamed
    /// weight bytes. Read-only — no kernel, no comm. The embed is counted as resident but
    /// excluded from the per-forward stream (a gather reads O(batch) rows, not the tensor).
    pub fn dump_rank_weights(&self) {
        type WRow = (String, &'static str, usize, usize, usize, &'static str);
        fn push_w(rows: &mut Vec<WRow>, site: &str, w: &W, status: &'static str) {
            let (recipe, m, k, bytes) = match w {
                W::Bf16(b) => ("bf16", 0, 0, b.len() * 2),
                W::Fp8 { m, k, .. } => ("fp8", *m, *k, w.bytes()),
                W::Fp8Blk { m, k, .. } => ("fp8blk", *m, *k, w.bytes()),
                W::Nvfp4 { m, k, .. } => ("nvfp4", *m, *k, w.bytes()),
                W::Nvfp4Raw { m, k, .. } => ("nvfp4raw", *m, *k, w.bytes()),
            };
            rows.push((site.to_string(), recipe, m, k, bytes, status));
        }
        fn push_s(rows: &mut Vec<WRow>, site: &str, s: &S) {
            rows.push((site.to_string(), "f32", s.len(), 1, s.len() * 4, "lane-local"));
        }
        let mut rows: Vec<WRow> = Vec::new();

        let world = self.tp_world.max(1) as usize;
        let (ffn_f, attn_f, gdn_f) = (self.ffn_shard_factor(), self.attn_shard_factor(), self.gdn_shard_factor());
        let verdict = |dim: usize, full: usize, local: usize| -> &'static str {
            if local > 0 && local < full && dim == local { "sharded" }
            else if dim == full { "REPLICATED" }
            else { "replicated?" }
        };

        // Model-level tensors.
        push_w(&mut rows, "embed", &self.embed, "REPLICATED (gather; O(batch) rows read)");
        if let Some(lh) = &self.lm_head {
            push_w(&mut rows, "lm_head", lh, if self.lm_head_sharded { "sharded (vocab-parallel)" } else { "REPLICATED" });
        }
        if let Some(dh) = &self.draft_head {
            push_w(&mut rows, "draft_head", dh, "REPLICATED (FR draft subset)");
        }
        if let Some(dfl) = &self.dflash_lm_head {
            rows.push(("dflash_lm_head".into(), "bf16", 0, 0, dfl.len() * 2, "rank-0 only (DFlash)"));
        }

        // Backbone layers (shapes from cfg are the FULL-model dims; a sharded tensor's local dim
        // is the comparison that classifies it — per-tensor flags where they exist are authoritative).
        let cfg = &self.cfg;
        let (nh, hd) = (cfg.num_heads, cfg.head_dim);
        let qkv_full = AttnIn::fused_m(cfg);
        let o_full = nh * hd;
        let gdn_full = GdnIn::fused_m(cfg);
        let (o_local, gdn_local, vdim_local) = (
            if attn_f > 1 { o_full / attn_f } else { 0 },
            if gdn_f > 1 { self.eff_gdn_fused_m() } else { 0 },
            if gdn_f > 1 { self.eff_value_dim() } else { 0 },
        );
        let si = cfg.shared_expert_intermediate_size;
        for (i, layer) in self.layers.iter().enumerate() {
            let p = format!("layers[{i}]");
            if let Some(la) = &layer.la {
                match &la.in_proj {
                    GdnIn::Fused(w) => push_w(&mut rows, &format!("{p}.la.in_proj"), w, verdict(w_m(w), gdn_full, gdn_local)),
                    GdnIn::Split { qkv, z, b, a } => {
                        for (n, w) in [("qkv", qkv), ("z", z), ("b", b), ("a", a)] {
                            push_w(&mut rows, &format!("{p}.la.in_proj.{n}"), w, "split (legacy bf16)");
                        }
                    }
                }
                push_s(&mut rows, &format!("{p}.la.conv1d"), &la.conv1d);
                push_s(&mut rows, &format!("{p}.la.a_log"), &la.a_log);
                push_s(&mut rows, &format!("{p}.la.dt_bias"), &la.dt_bias);
                push_s(&mut rows, &format!("{p}.la.norm"), &la.norm);
                push_w(&mut rows, &format!("{p}.la.out_proj"), &la.out_proj, verdict(w_k(&la.out_proj), cfg.value_dim(), vdim_local));
            }
            if let Some(fa) = &layer.fa {
                match &fa.qkv {
                    AttnIn::Fused(w) => push_w(&mut rows, &format!("{p}.fa.qkv"), w, verdict(w_m(w), qkv_full, if attn_f > 1 { qkv_full / attn_f } else { 0 })),
                    AttnIn::Split { q, k, v } => {
                        for (n, w) in [("q", q), ("k", k), ("v", v)] {
                            push_w(&mut rows, &format!("{p}.fa.qkv.{n}"), w, "split (legacy bf16)");
                        }
                    }
                }
                push_w(&mut rows, &format!("{p}.fa.o_proj"), &fa.o_proj, verdict(w_k(&fa.o_proj), o_full, o_local));
                push_s(&mut rows, &format!("{p}.fa.q_norm"), &fa.q_norm);
                push_s(&mut rows, &format!("{p}.fa.k_norm"), &fa.k_norm);
            }
            match &layer.mlp {
                Ffn::Dense(mlp) => {
                    let inter = cfg.intermediate_size;
                    let ilocal = if ffn_f > 1 { inter / ffn_f } else { 0 };
                    push_w(&mut rows, &format!("{p}.mlp.gate"), &mlp.gate, verdict(w_m(&mlp.gate), inter, ilocal));
                    push_w(&mut rows, &format!("{p}.mlp.up"), &mlp.up, verdict(w_m(&mlp.up), inter, ilocal));
                    push_w(&mut rows, &format!("{p}.mlp.down"), &mlp.down, verdict(w_k(&mlp.down), inter, ilocal));
                }
                Ffn::Moe(moe) => {
                    let est = if moe.experts_sharded { "sharded (expert-band)" } else { "REPLICATED" };
                    push_w(&mut rows, &format!("{p}.mlp.experts.gate_up"), &moe.gate_up, est);
                    push_w(&mut rows, &format!("{p}.mlp.experts.down"), &moe.down, est);
                    push_w(&mut rows, &format!("{p}.mlp.router"), &moe.router, "REPLICATED (global ids)");
                    push_w(&mut rows, &format!("{p}.mlp.shared.gate"), &moe.shared.gate, verdict(w_m(&moe.shared.gate), si, if si > 0 { si / world } else { 0 }));
                    push_w(&mut rows, &format!("{p}.mlp.shared.up"), &moe.shared.up, verdict(w_m(&moe.shared.up), si, if si > 0 { si / world } else { 0 }));
                    if let Some(sgu) = &moe.shared_gate_up {
                        push_w(&mut rows, &format!("{p}.mlp.shared.gate_up"), sgu, verdict(w_m(sgu), 2 * si, 2 * si / world));
                    }
                    push_w(&mut rows, &format!("{p}.mlp.shared.down"), &moe.shared.down,
                           if moe.shared_sharded { "sharded (row)" } else { "REPLICATED" });
                    if let Some(sg) = &moe.shared_gate { push_w(&mut rows, &format!("{p}.mlp.shared_gate"), sg, "REPLICATED (tiny)"); }
                }
            }
            push_s(&mut rows, &format!("{p}.input_ln"), &layer.input_ln);
            push_s(&mut rows, &format!("{p}.post_ln"), &layer.post_ln);
        }

        // The MTP block: replicated under TP by default (barrier-free drafting); §4.1 sharding
        // (--tp-shard-mtp) flips the per-tensor flags. Report each tensor so the diff is exact.
        if let Some(mtp) = &self.mtp {
            let st = |on: bool| if on { "sharded (MTP §4.1)" } else { "REPLICATED (MTP by design)" };
            push_w(&mut rows, "mtp.fc", &mtp.fc, st(mtp.fc_sharded));
            match &mtp.fa.qkv {
                AttnIn::Fused(w) => push_w(&mut rows, "mtp.fa.qkv", w, st(mtp.attn_sharded)),
                AttnIn::Split { q, k, v } => {
                    for (n, w) in [("q", q), ("k", k), ("v", v)] {
                        push_w(&mut rows, &format!("mtp.fa.qkv.{n}"), w, st(mtp.attn_sharded));
                    }
                }
            }
            push_w(&mut rows, "mtp.fa.o_proj", &mtp.fa.o_proj, st(mtp.attn_sharded));
            let mst = st(mtp.mlp_sharded || matches!(&mtp.mlp, Ffn::Moe(m) if m.experts_sharded));
            match &mtp.mlp {
                Ffn::Dense(mlp) => {
                    push_w(&mut rows, "mtp.mlp.gate", &mlp.gate, mst);
                    push_w(&mut rows, "mtp.mlp.up", &mlp.up, mst);
                    push_w(&mut rows, "mtp.mlp.down", &mlp.down, mst);
                }
                Ffn::Moe(moe) => {
                    push_w(&mut rows, "mtp.mlp.experts.gate_up", &moe.gate_up, mst);
                    push_w(&mut rows, "mtp.mlp.experts.down", &moe.down, mst);
                    push_w(&mut rows, "mtp.mlp.router", &moe.router, "REPLICATED (global ids)");
                    push_w(&mut rows, "mtp.mlp.shared.gate", &moe.shared.gate, "REPLICATED (shared expert)");
                    push_w(&mut rows, "mtp.mlp.shared.up", &moe.shared.up, "REPLICATED (shared expert)");
                    if let Some(sgu) = &moe.shared_gate_up { push_w(&mut rows, "mtp.mlp.shared.gate_up", sgu, "REPLICATED (shared expert)"); }
                    push_w(&mut rows, "mtp.mlp.shared.down", &moe.shared.down, "REPLICATED (shared expert)");
                    if let Some(sg) = &moe.shared_gate { push_w(&mut rows, "mtp.mlp.shared_gate", sg, "REPLICATED (tiny)"); }
                }
            }
            for (n, s) in [("pre_fc_norm_hidden", &mtp.pre_fc_norm_hidden), ("pre_fc_norm_embedding", &mtp.pre_fc_norm_embedding),
                           ("input_ln", &mtp.input_ln), ("post_ln", &mtp.post_ln), ("final_norm", &mtp.final_norm)] {
                push_s(&mut rows, &format!("mtp.{n}"), s);
            }
        }
        push_s(&mut rows, "final_norm", &self.final_norm);

        // Aggregate: collapse the layer index so 64 identical layers become one ×64 line.
        let fmt_b = |b: usize| if b >= 1 << 30 { format!("{:.2} GB", b as f64 / (1 << 30) as f64) }
                              else if b >= 1 << 20 { format!("{:.1} MB", b as f64 / (1 << 20) as f64) }
                              else if b >= 1 << 10 { format!("{:.1} KB", b as f64 / (1 << 10) as f64) }
                              else { format!("{b} B") };
        let mut groups: std::collections::BTreeMap<(String, &'static str, usize, usize, &'static str), (usize, usize)> = Default::default();
        for (site, recipe, m, k, bytes, status) in &rows {
            let key_site = if let Some(rest) = site.clone().strip_prefix("layers[") {
                let rest = rest.splitn(2, ']').nth(1).unwrap_or("").to_string();
                format!("layers[*]{rest}")
            } else { site.clone() };
            let e = groups.entry((key_site, recipe, *m, *k, status)).or_insert((0, 0));
            e.0 += 1; e.1 += bytes;
        }
        let mut table: Vec<_> = groups.into_iter().collect();
        table.sort_by(|a, b| b.1 .1.cmp(&a.1 .1));

        println!("=== per-rank weight dump — rank {}/{} · family {:?} · h={} layers={} vocab={} ===",
                 self.tp_rank, world, cfg.family, cfg.hidden_size, cfg.num_layers, cfg.vocab_size);
        println!("{:<44} {:<8} {:>14} {:>10} {:>4}  {}", "site", "recipe", "dims", "bytes×n", "n", "status");
        let (mut total, mut stream) = (0usize, 0usize);
        for ((site, recipe, m, k, status), (n, bytes)) in &table {
            total += bytes;
            if *recipe != "f32" && site != "embed" { stream += bytes; }
            let dims = if *m > 0 { format!("{m}x{k}") } else { "-".into() };
            println!("{:<44} {:<8} {:>14} {:>10} {:>4}  {}", site, recipe, dims, fmt_b(*bytes), n, status);
        }
        println!("\nresident weights total: {} (incl. embed, norms, scratch tensors)", fmt_b(total));
        println!("streamed per forward (excl. embed/norms; incl. lm_head + draft_head): {}", fmt_b(stream));
        println!("\n--- REPLICATION TARGETS (whole-model tensors resident on this rank, >16 MB/group) ---");
        let mut any = false;
        for ((site, recipe, _m, _k, status), (n, bytes)) in &table {
            if status.contains("REPLICATED") && *bytes > (1 << 20) * 16 {
                println!("  {:<44} {:<8} {:>10} ×{:>3}  {}", site, recipe, fmt_b(*bytes), n, status);
                any = true;
            }
        }
        if !any { println!("  (none >16 MB — memory rule satisfied on this rank)"); }
    }

    /// Column-parallel split of a packed NVFP4 weight [M,K] → rank-local [M/world, K]. The kernel packs
    /// output tiles of 16 rows contiguously (`Wt32[(mt*nblk+kb)*32+lane]`, `gs[mt]`), so rank r owns the
    /// contiguous tile band [r·M/world/16, …) — a straight byte-slice of qweight/scales/gs. Byte-exact.
    fn shard_nvfp4_col(&self, w: &W, rank: usize, world: usize) -> W {
        let m = match w { W::Nvfp4 { m, .. } => *m, _ => panic!("tp col-shard: expected NVFP4") };
        self.shard_nvfp4_col_segs(w, &[(0, m, true)], rank, world)   // whole-M single segment (FFN gate/up)
    }

    /// MXFP4-native twin of `shard_nvfp4_col` (whole-M col shard + repack re-key).
    fn shard_mxfp4_col(&mut self, w: &W, rank: usize, world: usize) -> W {
        let m = match w { W::Nvfp4 { m, .. } | W::Fp8 { m, .. } | W::Fp8Blk { m, .. } => *m,
                          _ => panic!("tp col-shard: expected NVFP4 or FP8, got another format") };
        self.shard_mxfp4_col_segs(w, &[(0, m, true)], rank, world)
    }

    /// MXFP4-native twins of the sharders: run the base shard, then re-key + slice the tensor's
    /// OMMA repack (the base sharders allocate NEW slices, so the old ptr would dangle into a
    /// stale map entry — the dispatch keys repacks by qweight ptr). The OMMA layout is per-16-row
    /// tile contiguous ((mt*nks+ks) indexing), so BOTH shard directions are byte-exact gathers of
    /// the same tile structure. Allowlisted tensors have no repack entry → no-op.
    /// Non-NVFP4 (the -mixed recipes keep mixer/FFN tensors in FP8) delegate to the format
    /// dispatchers — there is no OMMA repack to re-key there.
    fn shard_mxfp4_col_segs(&mut self, w: &W, segs: &[(usize, usize, bool)], rank: usize, world: usize) -> W {
        let (old_ptr, m, k) = match w {
            W::Nvfp4 { qweight, m, k, .. } => (*qweight.device_ptr() as u64, *m, *k),
            _ => return self.shard_col_segs(w, segs, rank, world),
        };
        let nw = self.shard_nvfp4_col_segs(w, segs, rank, world);
        let entry = self.mxfp4.as_mut().and_then(|st| st.w.remove(&old_ptr));
        // Stay-list (allow_bf16) re-key: the base sharder allocates a NEW qweight slice, so a
        // stay-listed ptr (lm_head/embed) would silently fall out of the allowlist and the
        // native route would take it back over.
        if let Some(st) = self.mxfp4.as_mut() {
            if st.allow_bf16.write().unwrap().remove(&old_ptr) {
                let new_ptr = match &nw { W::Nvfp4 { qweight, .. } => *qweight.device_ptr() as u64,
                                          _ => unreachable!() };
                st.allow_bf16.write().unwrap().insert(new_ptr);
            }
        }
        if let Some(crate::mxfp4::OmmaEntry::Owned(a, s)) = entry {
            let (a_new, s_new) = self.shard_omma_host(&a, &s, m, k, Some(segs), rank, world);
            let new_ptr = match &nw { W::Nvfp4 { qweight, .. } => *qweight.device_ptr() as u64,
                                      _ => unreachable!() };
            self.mxfp4.as_mut().unwrap().w.insert(new_ptr, crate::mxfp4::OmmaEntry::Owned(a_new, s_new));
        }
        nw
    }
    fn shard_mxfp4_row(&mut self, w: &W, rank: usize, world: usize) -> W {
        let (old_ptr, m, k) = match w {
            W::Nvfp4 { qweight, m, k, .. } => (*qweight.device_ptr() as u64, *m, *k),
            _ => return self.shard_row(w, rank, world),
        };
        let nw = self.shard_nvfp4_row(w, rank, world);
        let entry = self.mxfp4.as_mut().and_then(|st| st.w.remove(&old_ptr));
        if let Some(st) = self.mxfp4.as_mut() {
            if st.allow_bf16.write().unwrap().remove(&old_ptr) {
                let new_ptr = match &nw { W::Nvfp4 { qweight, .. } => *qweight.device_ptr() as u64,
                                          _ => unreachable!() };
                st.allow_bf16.write().unwrap().insert(new_ptr);
            }
        }
        if let Some(crate::mxfp4::OmmaEntry::Owned(a, s)) = entry {
            let (a_new, s_new) = self.shard_omma_host(&a, &s, m, k, None, rank, world);
            let new_ptr = match &nw { W::Nvfp4 { qweight, .. } => *qweight.device_ptr() as u64,
                                      _ => unreachable!() };
            self.mxfp4.as_mut().unwrap().w.insert(new_ptr, crate::mxfp4::OmmaEntry::Owned(a_new, s_new));
        }
        nw
    }

    /// Slice the OMMA repack host-side with the SAME tile math as the base sharders.
    /// Col (`segs`): contiguous per-tile band slices. Row: per-tile kstep slices.
    fn shard_omma_host(&self, a: &cudarc::driver::CudaSlice<u8>, s: &cudarc::driver::CudaSlice<u8>,
                       m: usize, k: usize, segs: Option<&[(usize, usize, bool)]>, rank: usize, world: usize)
                       -> (cudarc::driver::CudaSlice<u8>, cudarc::driver::CudaSlice<u8>) {
        let ah = self.dev.dtoh_sync_copy(a).unwrap();
        let sh = self.dev.dtoh_sync_copy(s).unwrap();
        let (nks, nks_local) = (k / 64, k / world / 64);
        let mut a_new = Vec::new();
        let mut s_new = Vec::new();
        if let Some(segs) = segs {
            for &(off, cnt, split) in segs {
                let take = if split { cnt / world } else { cnt };
                let tile_lo = (off + if split { rank * take } else { 0 }) / 16;
                let ntl = take / 16;
                a_new.extend_from_slice(&ah[tile_lo * nks * 512..(tile_lo + ntl) * nks * 512]);
                s_new.extend_from_slice(&sh[tile_lo * nks * 64..(tile_lo + ntl) * nks * 64]);
            }
        } else {
            let tiles = m / 16;
            let ks_lo = rank * nks_local;
            for mt in 0..tiles {
                let a_src = (mt * nks + ks_lo) * 512;
                a_new.extend_from_slice(&ah[a_src..a_src + nks_local * 512]);
                let s_src = (mt * nks + ks_lo) * 64;
                s_new.extend_from_slice(&sh[s_src..s_src + nks_local * 64]);
            }
        }
        (self.dev.htod_sync_copy(&a_new).unwrap(), self.dev.htod_sync_copy(&s_new).unwrap())
    }

    /// Column-parallel split over one or more output-row SEGMENTS. Each `(row_off, row_count)` segment is
    /// independently halved by rank and the node's halves are concatenated → rank-local weight. One
    /// segment = a plain col-shard (FFN gate/up). Three segments = the fused attention qkv, whose output
    /// is `[qg (nh·hd·2) | k (nkv·hd) | v (nkv·hd)]` and must be split per head WITHIN each segment (so
    /// the node gets its q/gate heads AND its kv heads). Every segment offset is 16-tile-aligned and each
    /// half is 16-tile-aligned (row_count % 32 == 0), so this is an exact byte gather. gs is per output
    /// tile → sliced with the tiles.
    /// Each segment is `(row_off, row_count, split)`. `split=false` copies the segment WHOLE to both
    /// ranks — needed for tensors whose rows are one-per-head but too few to halve at NVFP4's 16-row
    /// tile granularity (GDN `b`/`a`: 48 rows, 3 tiles, and 24 is not a multiple of 16). Such a segment
    /// only has to be tile-aligned, not halvable.
    fn shard_nvfp4_col_segs(&self, w: &W, segs: &[(usize, usize, bool)], rank: usize, world: usize) -> W {
        let (qw, sc, gs, _m, k) = match w {
            W::Nvfp4 { qweight, scales, gs, m, k } => (qweight, scales, gs, *m, *k),
            _ => panic!("tp col-shard: expected NVFP4, got another format"),
        };
        let nblk = k / 16;
        let qpt = nblk * 128;            // qweight bytes per output tile (nblk k-blocks × 32 u32)
        let spt = nblk * 16;             // scale bytes per output tile
        let qh = self.dev.dtoh_sync_copy(qw).unwrap();
        let sh = self.dev.dtoh_sync_copy(sc).unwrap();
        let gh = self.dev.dtoh_sync_copy(gs).unwrap();
        let (mut q_new, mut s_new, mut g_new) = (Vec::new(), Vec::new(), Vec::new());
        let mut m_local = 0usize;
        for &(off, cnt, split) in segs {
            assert!(off % 16 == 0, "tp col-shard seg (off {off}) not tile-aligned");
            let take = if split { cnt / world } else { cnt };
            if split { assert!(cnt % (16 * world) == 0, "tp col-shard seg (off {off}, cnt {cnt}) won't split at 16-row tiles across world {world}"); }
            else     { assert!(cnt % 16 == 0, "tp col-shard unsplit seg (off {off}, cnt {cnt}) not tile-aligned"); }
            let tile_lo = (off + if split { rank * take } else { 0 }) / 16;
            let ntl = take / 16;
            q_new.extend_from_slice(&qh[tile_lo * qpt..(tile_lo + ntl) * qpt]);
            s_new.extend_from_slice(&sh[tile_lo * spt..(tile_lo + ntl) * spt]);
            g_new.extend_from_slice(&gh[tile_lo..tile_lo + ntl]);
            m_local += take;
        }
        W::Nvfp4 {
            qweight: self.dev.htod_sync_copy(&q_new).unwrap(),
            scales: self.dev.htod_sync_copy(&s_new).unwrap(),
            gs: self.dev.htod_sync_copy(&g_new).unwrap(),
            m: m_local, k,
        }
    }

    /// Row-parallel split of a packed NVFP4 weight [M,K] → rank-local [M, K/world]. K is chunked into
    /// 16-wide k-blocks per output tile; rank r owns k-blocks [r·nblk/world, …) of EVERY output tile, so the
    /// packed bytes are gathered per tile (strided, one contiguous chunk per tile). gs is per output
    /// tile (K-independent) → copied unchanged: with the same gs on both ranks, (v0·gs)+(v1·gs) =
    /// (v0+v1)·gs, so applying gs to each partial then summing is exact. Byte-exact per tile.
    /// Column-parallel split of an FP32 device tensor laid out as `[rows][row_stride]`, over one or more
    /// row SEGMENTS. Same contiguous per-segment halving as `shard_nvfp4_col_segs`, for the GDN tensors
    /// that are not quantized: `conv1d` ([conv_dim][k], sliced into its q/k/v channel segments) and
    /// `a_log` / `dt_bias` ([n_v_heads], row_stride 1).
    fn shard_f32_segs(&self, s: &S, segs: &[(usize, usize)], row_stride: usize, rank: usize, world: usize) -> S {
        let host = self.dev.dtoh_sync_copy(s).expect("dtoh gdn f32");
        let mut out: Vec<f32> = Vec::new();
        for &(off, rows) in segs {
            assert!(rows % world == 0, "gdn f32 shard: segment of {rows} rows won't split across world {world}");
            let take = rows / world;
            let base = (off + rank * take) * row_stride;
            out.extend_from_slice(&host[base..base + take * row_stride]);
        }
        self.dev.htod_sync_copy(&out).expect("htod gdn f32")
    }

    /// Grow an FP32 vector to `full` entries, filling everything above its current length with NaN.
    /// Used only by the head-execution proof (see `tp_head_proof`).
    fn pad_nan_redzone(&self, s: &S, full: usize) -> S {
        let mut h = self.dev.dtoh_sync_copy(s).expect("dtoh redzone");
        assert!(h.len() <= full);
        h.resize(full, f32::NAN);
        self.dev.htod_sync_copy(&h).expect("htod redzone")
    }

    /// Proof B verification: every GDN state red zone must still be entirely NaN. Any real number in
    /// there is a kernel that ran at the FULL head count and wrote outside this rank's heads — the
    /// gate-invisible, timing-invisible failure mode. Returns (checked, violations).
    pub fn verify_head_redzones(&self, state: &BatchGpuState) -> (usize, usize) {
        if !self.tp_head_proof() { return (0, 0); }
        let cfg = &self.cfg;
        let (mut checked, mut bad) = (0usize, 0usize);
        // Proof D: every LOCAL head must have been visited, and no head above the local range at all.
        // A `blk % nh` aliasing fold shows up as unequal counts across local heads.
        if let Some(v) = &self.head_visits {
            let h = self.dev.dtoh_sync_copy(v).expect("dtoh visits");
            let nloc = self.eff_lin_v_heads();
            let local: Vec<u64> = h[..nloc].to_vec();
            let beyond: u64 = h[nloc..].iter().sum();
            let uniform = local.iter().all(|c| *c == local[0]) && local[0] > 0;
            checked += 1;
            if !uniform || beyond > 0 {
                bad += 1;
                eprintln!("[head-proof] VISIT VIOLATION: local head counts {:?} (want all equal, >0), \
                           visits to heads >= {nloc}: {beyond}", &local[..local.len().min(8)]);
            } else {
                eprintln!("[head-proof] visits: all {nloc} local heads visited {} times, 0 beyond — \
                           no aliasing fold", local[0]);
            }
        }
        for (li, lt) in cfg.layer_types.iter().enumerate() {
            if !matches!(lt, crate::qwen::LayerType::LinearAttention) { continue; }
            let conv_n = self.eff_conv_dim() * cfg.conv_kernel;
            let s_n = self.eff_lin_v_heads() * cfg.lin_k_dim * cfg.lin_v_dim;
            for (buf, local_n, what) in [(&state.conv_state[li], conv_n, "conv"),
                                         (&state.s_state[li], s_n, "recurrent")] {
                let Some(b) = buf else { continue };
                let h = self.dev.dtoh_sync_copy(b).expect("dtoh redzone check");
                if h.len() < local_n * 2 { continue; }
                let violations = h[local_n..].iter().filter(|x| x.to_bits() != TP_REDZONE_SENTINEL).count();
                checked += 1;
                if violations > 0 {
                    bad += 1;
                    eprintln!("[head-proof] VIOLATION layer {li} {what} state: {violations} of {} \
                               red-zone floats were WRITTEN — a kernel ran at full head count",
                              h.len() - local_n);
                }
            }
        }
        (checked, bad)
    }

    fn shard_nvfp4_row(&self, w: &W, rank: usize, world: usize) -> W {
        let (qw, sc, gs, m, k) = match w {
            W::Nvfp4 { qweight, scales, gs, m, k } => (qweight, scales, gs, *m, *k),
            _ => panic!("tp row-shard: expected NVFP4 (dense FFN down), got another format"),
        };
        assert!((k / 16) % world == 0 && (k / world) % 32 == 0, "tp row-shard: k={k} won't split into 32-aligned slices across world {world}");
        let k_local = k / world;
        let nblk_full = k / 16;
        let nblk_local = k_local / 16;
        let tiles = m / 16;
        let kb_lo = rank * nblk_local;
        let qh = self.dev.dtoh_sync_copy(qw).unwrap();
        let sh = self.dev.dtoh_sync_copy(sc).unwrap();
        let gh = self.dev.dtoh_sync_copy(gs).unwrap();
        let mut q_new = Vec::with_capacity(tiles * nblk_local * 128);
        let mut s_new = Vec::with_capacity(tiles * nblk_local * 16);
        for mt in 0..tiles {
            let q_src = (mt * nblk_full + kb_lo) * 128;
            q_new.extend_from_slice(&qh[q_src..q_src + nblk_local * 128]);
            let s_src = (mt * nblk_full + kb_lo) * 16;
            s_new.extend_from_slice(&sh[s_src..s_src + nblk_local * 16]);
        }
        W::Nvfp4 {
            qweight: self.dev.htod_sync_copy(&q_new).unwrap(),
            scales: self.dev.htod_sync_copy(&s_new).unwrap(),
            gs: self.dev.htod_sync_copy(&gh).unwrap(),
            m, k: k_local,
        }
    }

    /// Mixer-shard dispatch by weight format. The -mixed recipes (0.8b/2b/4b/122B-mixed) keep the GDN
    /// (and sometimes attn) mixer tensors in FP8 while the FFN is NVFP4, so the four mixer shard sites
    /// must accept both. The geometry (segments, rank, 16-row tiles) is format-independent.
    fn shard_col_segs(&self, w: &W, segs: &[(usize, usize, bool)], rank: usize, world: usize) -> W {
        match w {
            W::Nvfp4 { .. } => self.shard_nvfp4_col_segs(w, segs, rank, world),
            W::Fp8 { .. }   => self.shard_fp8_col_segs(w, segs, rank, world),
            W::Fp8Blk { .. } => self.shard_fp8blk_col_segs(w, segs, rank, world),
            _ => panic!("tp col-shard: expected NVFP4 or FP8, got another format"),
        }
    }
    fn shard_row(&self, w: &W, rank: usize, world: usize) -> W {
        match w {
            W::Nvfp4 { .. } => self.shard_nvfp4_row(w, rank, world),
            W::Fp8 { .. }   => self.shard_fp8_row(w, rank, world),
            W::Fp8Blk { .. } => self.shard_fp8blk_row(w, rank, world),
            _ => panic!("tp row-shard: expected NVFP4 or FP8, got another format"),
        }
    }

    /// In-place (post-load) twin of `shard_raw_fp8_blk` for block-128 FP8 (`W::Fp8Blk`).
    /// A block-scaled FP8 weight keeps its RAW [M,K] codes and the flat [(M/128)·nb_k] scale grid
    /// resident (the MMA/e4m3 repacks live in `fp8e_map`, keyed by the data pointer and built
    /// lazily at the first prefill — so a shard that runs at attach, before any forward, leaves no
    /// stale entry behind), which is exactly the layout `shard_raw_fp8_blk` was written for. So the
    /// in-place path DELEGATES to it rather than re-deriving the geometry: one source of truth, and
    /// the two load paths are byte-identical by construction (the property the shard_raw_tests
    /// module asserts). Before this, `--dump-rank-weights --world N` on an FP8 trunk (mixers off =>
    /// in-place path) panicked with "tp col-shard: expected NVFP4 or FP8, got another format".
    fn shard_fp8blk_col_segs(&self, w: &W, segs: &[(usize, usize, bool)], rank: usize, world: usize) -> W {
        let (data, sc, m, k, nb_k) = match w {
            W::Fp8Blk { data, blk_scale, nb_k, m, k } => (data, blk_scale, *m, *k, *nb_k),
            _ => panic!("tp fp8blk col-shard: expected Fp8Blk, got another format"),
        };
        let qh = self.dev.dtoh_sync_copy(data).unwrap();
        let sh = self.dev.dtoh_sync_copy(sc).unwrap();
        let (q, s, mo, ko, nb) = shard_raw_fp8_blk(&qh, &sh, m, k, nb_k,
                                                   &LoadShardOp::ColSegs(segs.to_vec()), rank, world);
        W::Fp8Blk {
            data: self.dev.htod_sync_copy(&q).unwrap(),
            blk_scale: self.dev.htod_sync_copy(&s).unwrap(),
            nb_k: nb, m: mo, k: ko,
        }
    }

    /// Row-parallel twin of `shard_fp8blk_col_segs` (K-split): rank r keeps k-blocks
    /// [r·nb_k/world, …) of every 128-row scale band. Requires nb_k % world == 0 — asserted in
    /// `shard_raw_fp8_blk`, not silently replicated, because a wrong k-slice is a silent
    /// numerical error rather than a crash.
    fn shard_fp8blk_row(&self, w: &W, rank: usize, world: usize) -> W {
        let (data, sc, m, k, nb_k) = match w {
            W::Fp8Blk { data, blk_scale, nb_k, m, k } => (data, blk_scale, *m, *k, *nb_k),
            _ => panic!("tp fp8blk row-shard: expected Fp8Blk, got another format"),
        };
        let qh = self.dev.dtoh_sync_copy(data).unwrap();
        let sh = self.dev.dtoh_sync_copy(sc).unwrap();
        let (q, s, mo, ko, nb) = shard_raw_fp8_blk(&qh, &sh, m, k, nb_k,
                                                   &LoadShardOp::Row, rank, world);
        W::Fp8Blk {
            data: self.dev.htod_sync_copy(&q).unwrap(),
            blk_scale: self.dev.htod_sync_copy(&s).unwrap(),
            nb_k: nb, m: mo, k: ko,
        }
    }

    /// Column-parallel split over output-row SEGMENTS of a packed FP8 weight — the FP8 twin of
    /// `shard_nvfp4_col_segs`. The data is MMA-repacked (`repack_fp8_mma`: [m/16 tiles][nblk
    /// k-blocks][256 B]), so the same contiguous 16-row tile-band halving applies — the 16-row
    /// alignment is a PACKED-layout requirement, hence the same segment asserts as NVFP4 even though
    /// FP8 itself needs no row alignment. `row_scale` is one f32 per OUTPUT ROW (constant over K,
    /// folded into the f32 accumulator at the epilogue) → sliced with the rows, where NVFP4's gs is
    /// sliced per tile.
    fn shard_fp8_col_segs(&self, w: &W, segs: &[(usize, usize, bool)], rank: usize, world: usize) -> W {
        let (qw, rs, _m, k) = match w {
            W::Fp8 { data, row_scale, m, k } => (data, row_scale, *m, *k),
            _ => panic!("tp col-shard: expected FP8, got another format"),
        };
        let nblk = k / 16;
        let qpt = nblk * 256;            // fp8 bytes per output tile (nblk k-blocks × 16×16 B)
        let qh = self.dev.dtoh_sync_copy(qw).unwrap();
        let rh = self.dev.dtoh_sync_copy(rs).unwrap();
        let (mut q_new, mut r_new) = (Vec::new(), Vec::new());
        let mut m_local = 0usize;
        for &(off, cnt, split) in segs {
            assert!(off % 16 == 0, "tp fp8 col-shard seg (off {off}) not tile-aligned");
            let take = if split { cnt / world } else { cnt };
            if split { assert!(cnt % (16 * world) == 0, "tp fp8 col-shard seg (off {off}, cnt {cnt}) won't split at 16-row tiles across world {world}"); }
            else     { assert!(cnt % 16 == 0, "tp fp8 col-shard unsplit seg (off {off}, cnt {cnt}) not tile-aligned"); }
            let row_lo = off + if split { rank * take } else { 0 };
            let tile_lo = row_lo / 16;
            let ntl = take / 16;
            q_new.extend_from_slice(&qh[tile_lo * qpt..(tile_lo + ntl) * qpt]);
            r_new.extend_from_slice(&rh[row_lo..row_lo + take]);
            m_local += take;
        }
        W::Fp8 {
            data: self.dev.htod_sync_copy(&q_new).unwrap(),
            row_scale: self.dev.htod_sync_copy(&r_new).unwrap(),
            m: m_local, k,
        }
    }

    /// Row-parallel split of a packed FP8 weight [M,K] → rank-local [M, K/2]. Same per-output-tile
    /// k-block gather as `shard_nvfp4_row`, with 256 B k-blocks. `gemm_mma_fp8_b` reads k-blocks
    /// SINGLY (no adjacent-pair loop like fp4's), so halves only need 16-alignment, not 32.
    /// `row_scale` is per output row (K-independent) → copied WHOLE on both ranks: with the same rs
    /// on both, (v0·rs)+(v1·rs) = (v0+v1)·rs, so applying rs to each partial then all-reducing is
    /// exact under the accepted fp-reassociation class — the NVFP4 row shard's gs argument verbatim.
    fn shard_fp8_row(&self, w: &W, rank: usize, world: usize) -> W {
        let (qw, rs, m, k) = match w {
            W::Fp8 { data, row_scale, m, k } => (data, row_scale, *m, *k),
            _ => panic!("tp row-shard: expected FP8, got another format"),
        };
        assert!((k / 16) % world == 0, "tp fp8 row-shard: k={k} won't split into 16-aligned slices across world {world}");
        let k_local = k / world;
        let nblk_full = k / 16;
        let nblk_local = k_local / 16;
        let tiles = m / 16;
        let kb_lo = rank * nblk_local;
        let qh = self.dev.dtoh_sync_copy(qw).unwrap();
        let rh = self.dev.dtoh_sync_copy(rs).unwrap();
        let mut q_new = Vec::with_capacity(tiles * nblk_local * 256);
        for mt in 0..tiles {
            let q_src = (mt * nblk_full + kb_lo) * 256;
            q_new.extend_from_slice(&qh[q_src..q_src + nblk_local * 256]);
        }
        W::Fp8 {
            data: self.dev.htod_sync_copy(&q_new).unwrap(),
            row_scale: self.dev.htod_sync_copy(&rh).unwrap(),   // whole: per-row, K-independent
            m, k: k_local,
        }
    }
}
