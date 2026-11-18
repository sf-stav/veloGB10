//! CUDA graph support: the `CudaGraph` handle and the graph capture / replay methods for decode, sampled decode,
//! verify and prefill. The compute stream is a BLOCKING stream (AGENTS.md section 2.1). Map: `mod.rs`.

use super::*;

/// A captured CUDA graph that replays the full decode forward pass.
pub struct CudaGraph {
    exec: cudarc::driver::sys::CUgraphExec,
    stream: cudarc::driver::sys::CUstream,
    /// K35: MoE rounds on a fractional expert width (by k2) this graph holds — counted on every
    /// replay (velogb10_exl3_frac_expert_rounds_total), so the receipt covers graphed serving too.
    frac_rounds: [u32; crate::metrics::EXL3_WIDTH_SLOTS],
}

// Safe: the graph handle and stream are only ever used from the single scheduler task.
unsafe impl Send for CudaGraph {}
unsafe impl Sync for CudaGraph {}

impl CudaGraph {
    /// S5F: wrap a raw instantiated graph + stream (the DFlash2 round's graph capture).
    pub fn from_exec(exec: cudarc::driver::sys::CUgraphExec, stream: cudarc::driver::sys::CUstream) -> Self {
        CudaGraph { exec, stream, frac_rounds: [0; crate::metrics::EXL3_WIDTH_SLOTS] }
    }

    /// K35: attach the fractional-expert round tally the capture recorded (metrics::exl3_frac_capture_take).
    pub fn with_frac_rounds(mut self, n: [u32; crate::metrics::EXL3_WIDTH_SLOTS]) -> Self { self.frac_rounds = n; self }

    /// Replay the captured graph on its stream. Does NOT sync — caller must sync before reading results.
    pub fn launch(&self) {
        use cudarc::driver::sys;
        unsafe {
            let r = sys::cuGraphLaunch(self.exec, self.stream);
            assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "cuGraphLaunch failed");
        }
        if self.frac_rounds.iter().any(|v| *v > 0) { crate::metrics::exl3_frac_expert_rounds_replayed(&self.frac_rounds); }
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        use cudarc::driver::sys;
        unsafe { sys::cuGraphExecDestroy(self.exec); }
    }
}

impl GpuModel {
    /// Capture a decode+sample graph for a given batch size. Caller must have valid params in bufs.
    pub fn capture_decode_sample_graph(&self, pool: &mut Pool, bufs: &mut DecodeBuffers,
                                       state: &mut BatchGpuState, kv_stride: usize, max_pc: usize,
                                       batch: usize) -> Option<CudaGraph> {
        let _xchain_guard = XchainCaptureGuard::new();
        use cudarc::driver::sys;
        let stream = self.stream.stream;
        let mut core = |pool: &mut Pool, bufs: &mut DecodeBuffers| {
            if Self::device_loop_on() {
                // NOTE: the resident warmup also executes ids_advance/penalty_*/seed_advance once
                // with dummy contents — harmless, and the seed keys are only uploaded at admission
                // (after capture), so no draw is burned.
                self.decode_step_resident(pool, bufs, state, kv_stride, max_pc, batch, true);
            } else {
                self.decode_sample_gpu_core(pool, bufs, state, kv_stride, max_pc, batch);
            }
        };
        core(pool, bufs);
        self.sync_stream();
        unsafe {
            let r = sys::cuStreamBeginCapture_v2(
                stream, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
            if r != sys::CUresult::CUDA_SUCCESS {
                crate::metrics::graph_capture_lost("nvfp4-decode-sample", "decode sample graph", &format!("BeginCapture {r:?}")); // S-B9-REL-INTEG C1
                return None;
            }
        }
        core(pool, bufs);
        let mut graph: sys::CUgraph = std::ptr::null_mut();
        unsafe {
            let e = sys::cuStreamEndCapture(stream, &mut graph);
            if e != sys::CUresult::CUDA_SUCCESS {
                crate::metrics::graph_capture_lost("nvfp4-decode-sample", "decode sample graph", &format!("{e:?}")); // S-B9-REL-VISIONFIX2 V2b
                return None;
            }
        }
        let mut exec: sys::CUgraphExec = std::ptr::null_mut();
        unsafe {
            let r = sys::cuGraphInstantiate_v2(&mut exec, graph, std::ptr::null_mut(), std::ptr::null_mut(), 0);
            sys::cuGraphDestroy(graph);
            if r != sys::CUresult::CUDA_SUCCESS {
                crate::metrics::graph_capture_lost("nvfp4-decode-sample", "decode sample graph", &format!("instantiate {r:?}")); // S-B9-REL-INTEG C1
                return None;
            }
        }
        Some(CudaGraph { exec, stream, frac_rounds: [0; crate::metrics::EXL3_WIDTH_SLOTS] })
    }

    /// Replay a captured decode+sample graph and read back token ids. Caller must htod fresh
    /// seeds (and any changed params) into bufs before calling.
    pub fn replay_decode_sample(&self, bufs: &DecodeBuffers, graph: &CudaGraph, batch: usize) -> Vec<u32> {
        graph.launch();
        self.sync_stream();
        self.dev.dtoh_sync_copy(&bufs.token_ids_dev).unwrap()
            .into_iter().take(batch).map(|x| x as u32).collect()
    }

    /// Capture a CUDA graph for the decode forward pass at a specific batch size and max_pc.
    /// After capture, replay via `graph.launch()` + `dev.synchronize()` + dtoh readback.
    /// The pool's buffer addresses are frozen at capture time — do not change forward path without re-capturing.
    pub fn capture_decode_graph(&self, pool: &mut Pool, bufs: &mut DecodeBuffers,
                                state: &mut BatchGpuState, kv_stride: usize, max_pc: usize,
                                batch: usize) -> Option<CudaGraph> {
        let _xchain_guard = XchainCaptureGuard::new();
        use cudarc::driver::sys;
        let stream = self.stream.stream;  // forked non-blocking stream (NOT the NULL stream)

        // Warmup: run once to populate pool + warm cuBLAS algorithm cache
        let mut step = |pool: &mut Pool, bufs: &mut DecodeBuffers| {
            if Self::device_loop_on() {
                self.decode_step_resident(pool, bufs, state, kv_stride, max_pc, batch, false);
            } else {
                self.forward_decode_gpu(pool, bufs, state, kv_stride, max_pc, batch);
            }
        };
        step(pool, bufs);
        self.sync_stream();

        // I8/Q4 tripwire (TP=2 only): the barrier protocol must be QUIESCED at capture time. Capture
        // records launches without executing them, so the device epoch must not move here; if it and the
        // published watermark disagree, a barrier is in flight and the graph would inherit a torn epoch.
        // Cheap, permanent, and the only thing standing between us and a very hard bug.
        if self.tp_world == 2 {
            // I8/Q4 tripwire (TP=2 only): the barrier protocol must be QUIESCED at capture time.
            // The ready watermark is published ASYNCHRONOUSLY (K1 posts the epoch, the gate/ready
            // release follows) — at mxfp4=on the warmup's last all-reduce can leave the watermark
            // briefly behind the posted epoch, tripping a false assert. Wait the publication out
            // (bounded) before the assert — a genuinely in-flight barrier either catches up or
            // trips the timeout with the same loud message.
            let mut epoch = crate::net::traced_device_epoch();
            let mut ready = crate::net::traced_gpu_ready();
            for _ in 0..10_000 {
                if epoch == ready { break; }
                std::thread::sleep(std::time::Duration::from_micros(50));
                epoch = crate::net::traced_device_epoch();
                ready = crate::net::traced_gpu_ready();
            }
            assert_eq!(epoch, ready,
                "graph capture with the TP barrier protocol in flight (device epoch {epoch} != gpu_ready {ready})");
            // v2 receive (EXPERT_GPU_ALLREDUCE §6.6): the sibling tripwire — the GPU's receive
            // watermark must equal the device epoch at quiesce, or a K2' skipped a barrier (the
            // rx_done publish happens after consumption; a capture racing an in-flight receive
            // would freeze a torn state into the graph).
            if self.tp_gpu_recv_on() {
                let mut rx = crate::net::traced_rx_done();
                for _ in 0..10_000 {
                    if rx == epoch { break; }
                    std::thread::sleep(std::time::Duration::from_micros(50));
                    rx = crate::net::traced_rx_done();
                }
                assert_eq!(rx, epoch,
                    "graph capture with a v2 receive in flight (rx_done {rx} != device epoch {epoch})");
            }
        }

        // Begin capture. cudarc 0.9.15 launches on the legacy NULL stream, which is NOT capturable
        // (CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED). Detect and fall back to non-graph decode.
        unsafe {
            let r = sys::cuStreamBeginCapture_v2(
                stream, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
            if r != sys::CUresult::CUDA_SUCCESS {
                crate::metrics::graph_capture_lost("nvfp4-decode", "decode graph", &format!("BeginCapture {r:?}")); // S-B9-REL-INTEG C1
                return None;
            }
        }

        step(pool, bufs);

        let mut graph: sys::CUgraph = std::ptr::null_mut();
        unsafe {
            let r = sys::cuStreamEndCapture(stream, &mut graph);
            if r != sys::CUresult::CUDA_SUCCESS {
                crate::metrics::graph_capture_lost("nvfp4-decode", "decode graph", &format!("{r:?}")); // S-B9-REL-VISIONFIX2 V2b
                return None;
            }
        }

        let mut exec: sys::CUgraphExec = std::ptr::null_mut();
        unsafe {
            let r = sys::cuGraphInstantiate_v2(&mut exec, graph, std::ptr::null_mut(), std::ptr::null_mut(), 0);
            sys::cuGraphDestroy(graph);
            if r != sys::CUresult::CUDA_SUCCESS {
                crate::metrics::graph_capture_lost("nvfp4-decode", "decode graph", &format!("instantiate {r:?}")); // S-B9-REL-INTEG C1
                return None;
            }
        }

        Some(CudaGraph { exec, stream, frac_rounds: [0; crate::metrics::EXL3_WIDTH_SLOTS] })
    }

    /// Replay a captured graph and read back token IDs.
    pub fn replay_decode(&self, bufs: &DecodeBuffers, graph: &CudaGraph, batch: usize) -> Vec<u32> {
        graph.launch();
        self.sync_stream();
        self.dev.dtoh_sync_copy(&bufs.token_ids_dev).unwrap()
            .into_iter().take(batch).map(|x| x as u32).collect()
    }
}

impl GpuModel {
    /// E13: capture the chain-verify GPU sequence as a CUDA graph. The eager call the caller just
    /// made is the warmup: it requests EXACTLY the capture pass's pool buckets (attn max_pc is
    /// kv_stride on BOTH passes — see the verify_forward_core_topo callsite — and the capture-only
    /// placeholder bucket is pre-warmed below), so no device alloc happens during capture; the
    /// barrier protocol is quiesced after it — the I8/Q4 tripwire from capture_decode_graph.
    /// The captured region is kernel-only: params expansion, embed, layers, logits, penalty,
    /// argmax into sc_tok. Returns None to stay eager if capture is unsupported.
    pub(super) fn capture_verify_graph(&self, pool: &mut Pool, state: &mut BatchGpuState, slot: usize,
                            kv_stride: usize, pos_start: usize, n: usize, ckpt_slot: Option<usize>,
                            penalty: Option<VerifyPenalty>) -> Option<CudaGraph> {
        let _xchain_guard = XchainCaptureGuard::new();
        use cudarc::driver::sys;
        let stream = self.stream.stream;

        // I8/Q4 tripwire (TP=2): the barrier protocol must be QUIESCED at capture time.
        if self.tp_world == 2 {
            let mut epoch = crate::net::traced_device_epoch();
            let mut ready = crate::net::traced_gpu_ready();
            for _ in 0..10_000 {
                if epoch == ready { break; }
                std::thread::sleep(std::time::Duration::from_micros(50));
                epoch = crate::net::traced_device_epoch();
                ready = crate::net::traced_gpu_ready();
            }
            assert_eq!(epoch, ready,
                "verify graph capture with the TP barrier protocol in flight (device epoch {epoch} != gpu_ready {ready})");
            // v2 receive sibling tripwire (EXPERT_GPU_ALLREDUCE §6.6): rx_done must equal the
            // device epoch at quiesce — a K2' that skipped a barrier would freeze a torn receive.
            if self.tp_gpu_recv_on() {
                let mut rx = crate::net::traced_rx_done();
                for _ in 0..10_000 {
                    if rx == epoch { break; }
                    std::thread::sleep(std::time::Duration::from_micros(50));
                    rx = crate::net::traced_rx_done();
                }
                assert_eq!(rx, epoch,
                    "verify graph capture with a v2 receive in flight (rx_done {rx} != device epoch {epoch})");
            }
        }

        // Pre-warm the ONE bucket the eager warmup cannot cover: verify_forward_core_topo ends its
        // capture pass with a capture-only placeholder get_bf16(1) (the eager pass holds a real
        // residual instead, so that bucket is otherwise requested for the first time INSIDE capture —
        // a NULL-stream alloc there is capture-illegal and aborted the capture at short context).
        let warm = pool.get_bf16(1);
        pool.release_bf16(warm, 1);

        let r = unsafe { sys::cuStreamBeginCapture_v2(stream, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) };
        if r != sys::CUresult::CUDA_SUCCESS {
            crate::metrics::graph_capture_lost("nvfp4-verify", &format!("verify graph, slot {slot}"), &format!("BeginCapture {r:?}")); // S-B9-REL-INTEG C1
            return None;
        }
        pool.capturing.set(true);

        // The capture pass: kernels are RECORDED, nothing executes (state untouched by design).
        let (logits, discard) = self.verify_forward_core_topo(
            pool, &vec![0u32; n], state, slot, kv_stride, pos_start, ckpt_slot, None, true, true);
        pool.release_bf16(discard, 1);
        let vocab = self.cfg.vocab_size;
        match logits {
            VerifyLogits::B16(lg) => {
                if let Some(p) = penalty {
                    self.pen_apply_verify(&p, d(&lg), false, n);
                }
                let block = 1024u32;
                blaunch!(self, "argmax_b", (n as u32,1,1), (block,1,1), (block*8),
                    (*self.sc_tok.device_ptr() as u64, d(&lg), vocab as i32, n as i32));
                pool.release_bf16(lg, vocab*n);
            }
            VerifyLogits::F32(lg) => {
                if let Some(p) = penalty {
                    self.pen_apply_verify(&p, d(&lg), true, n);
                }
                let block = 1024u32;
                blaunch!(self, "argmax_f32_b", (n as u32,1,1), (block,1,1), (block*8),
                    (*self.sc_tok.device_ptr() as u64, d(&lg), vocab as i32, n as i32));
                pool.release(lg, vocab*n);
            }
        }

        let mut graph: sys::CUgraph = std::ptr::null_mut();
        let r = unsafe { sys::cuStreamEndCapture(stream, &mut graph) };
        pool.capturing.set(false);
        if r != sys::CUresult::CUDA_SUCCESS {
            crate::metrics::graph_capture_lost("nvfp4-verify", &format!("verify graph, slot {slot}"), &format!("{r:?}")); // S-B9-REL-VISIONFIX2 V2b
            return None;
        }
        let mut exec: sys::CUgraphExec = std::ptr::null_mut();
        let r = unsafe { sys::cuGraphInstantiate_v2(&mut exec, graph, std::ptr::null_mut(), std::ptr::null_mut(), 0) };
        unsafe { sys::cuGraphDestroy(graph); }
        if r != sys::CUresult::CUDA_SUCCESS {
            crate::metrics::graph_capture_lost("nvfp4-verify", &format!("verify graph, slot {slot}"), &format!("instantiate {r:?}")); // S-B9-REL-INTEG C1
            return None;
        }
        Some(CudaGraph { exec, stream, frac_rounds: [0; crate::metrics::EXL3_WIDTH_SLOTS] })
    }
}

impl GpuModel {
    /// DEBUG CAPTURE (Hy3 oracle validation). Runs the EXACT `prefill_batch` kernel sequence
    /// (single-shot, pos_start=0, slot=0 — the same kernels, in the same order, on the same
    /// stream) and dtoh-copies the residual stream at every layer boundary, in the reference
    /// oracle's convention:
    ///
    ///   dumps[0]      = embed output                  (oracle `layer.00.in`)
    ///   dumps[li + 1] = residual after decoder layer li (oracle `layer.{li}.out` == `layer.{li+1}.in`)
    ///   dumps[L + 1]  = rmsnorm(residual, final_norm) (oracle `final_norm`)
    ///
    /// Each dump is [h * n] bf16 with token t's vector at offset t*h — the same byte layout as the
    /// oracle's [seq, hidden] row-major tensors. FullAttention layers only (hy_v3 is pure GQA).
    pub fn capture_prefill(&self, pool: &mut Pool, prompt: &[u32],
                           state: &mut BatchGpuState, kv_stride: usize) -> Vec<Vec<half::bf16>> {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let n = prompt.len();
        let rdim = cfg.rotary_dim;

        // Same rope-gather prologue as prefill_batch (positions 0..n, slot 0).
        let pos_i32: Vec<i32> = (0..n).map(|i| i as i32).collect();
        let pos_dev = self.dev.htod_sync_copy(&pos_i32).expect("htod pos");
        let slot_ids_dev = self.dev.htod_sync_copy(&vec![0i32; n]).expect("htod slot_ids");
        let cos = pool.get(n * rdim);
        let sin = pool.get(n * rdim);
        self.dev.synchronize().unwrap(); // NULL-stream htod must complete before the compute stream reads
        blaunch!(self, "gather_rope_b", grid(n*rdim), (256,1,1), 0,
            (d(&cos), d(&sin), d(&self.cos_table), d(&self.sin_table),
             *pos_dev.device_ptr() as u64, rdim as i32, n as i32));

        let mut dumps: Vec<Vec<half::bf16>> = Vec::with_capacity(self.layers.len() + 2);
        let residual = self.embed_batch(prompt);
        self.sync_stream();
        // Pool buffers are bucketed UP (power-of-two capacity); truncate every dump to h*n.
        let mut e = self.dev.dtoh_sync_copy(&residual).expect("dtoh embed");
        e.truncate(h * n);
        dumps.push(e);

        let normed = pool.get_bf16(h * n);
        for (li, layer) in self.layers.iter().enumerate() {
            assert!(matches!(layer.layer_type, LayerType::FullAttention),
                "capture_prefill is FullAttention-only (hy_v3 oracle validation); layer {li} is not");
            let fa = layer.fa.as_ref().unwrap();
            blaunch!(self, "rmsnorm_b", (n as u32,1,1), (1024,1,1), (4096) as u32,
                (d(&normed), d(&residual), d(&layer.input_ln), h as i32, n as i32, fbits(cfg.rms_eps)));
            if crate::opts::var(crate::opt!("cap-debug")).is_ok() {
                self.sync_stream();
                let mut nd = self.dev.dtoh_sync_copy(&normed).expect("dtoh normed");
                nd.truncate(h * n);
                let am = nd.iter().fold(0f32, |a, x| a.max(x.to_f32().abs()));
                let ss = nd.iter().map(|x| { let v = x.to_f32(); v * v }).sum::<f32>();
                eprintln!("  capdbg layer {li}: |normed| absmax={am:.4} rms={:.6}", (ss / nd.len() as f32).sqrt());
            }
            let kc_ptr = *state.k_cache[li].as_ref().unwrap().device_ptr();
            let vc_ptr = *state.v_cache[li].as_ref().unwrap().device_ptr();
            // full_attn_batch with prefill_pos_start=Some(0) launches the identical sequence the
            // prefill_batch inline mixer does: (fused|split) qkv GEMM, per-head qk norm, rope,
            // write_kv_prefill, tiled-or-scalar causal prefill attention, o_proj GEMM.
            // TP=2: sharded flags come from the model (rank-local heads + all-reduces fire here —
            // both ranks MUST run this capture in lockstep, and their dumps are bit-identical).
            let qsa = self.qsa_in(state, fa, li, kv_stride, 0, 0, 0);
            let ReduceOut::Full(mixer) = self.full_attn_batch(pool, &normed, fa, &pos_dev, n, kv_stride,
                kc_ptr, vc_ptr, &cos, &sin, *slot_ids_dev.device_ptr(), n, Some(0),
                self.attn_shard_factor() > 1, self.kv_mode(), false, qsa) else {
                unreachable!("AR landing 2: the capture probe never fuses")
            };
            blaunch!(self, "fused_res_rmsnorm_b", (n as u32,1,1), (1024,1,1), (4096) as u32,
                (d(&normed), d(&residual), d(&mixer), d(&layer.post_ln), h as i32, n as i32, fbits(cfg.rms_eps)));
            // --layer-csum: post-MIXER residual (pre-FFN) — splits GDN/attn corruption from MLP.
            if crate::opts::var(crate::opt!("layer-csum")).is_ok() {
                let base = *residual.device_ptr() as cudarc::driver::sys::CUdeviceptr;
                let mut acc = String::new();
                for r in [0usize, n / 2, n - 1] {
                    let mut tmp = vec![0u8; h * 2];
                    unsafe {
                        cudarc::driver::result::memcpy_dtoh_sync(&mut tmp, base + (r * h * 2) as u64)
                    }.expect("csum dtoh");
                    let (mut s, mut mx) = (0.0f64, 0.0f32);
                    for ch in tmp.chunks_exact(2) {
                        let f = half::bf16::from_bits(u16::from_le_bytes([ch[0], ch[1]])).to_f32();
                        s += f as f64; if f.abs() > mx { mx = f.abs(); }
                    }
                    acc.push_str(&format!(" MX{s}:{s:.4}/{mx:.4}"));
                }
                eprintln!("[csum] L{li} n{n}{acc}");
            }
            let ReduceOut::Full(mlp_out) = self.ffn_batch(pool, &normed, &layer.mlp, n, self.ffn_shard_factor() > 1, false) else {
                unreachable!("AR landing 2: the capture probe never fuses")
            };
            if crate::opts::var(crate::opt!("cap-debug")).is_ok() {
                // Decompose a blow-up: dump the mixer (attention out) and FFN out too.
                self.sync_stream();
                let mut mx = self.dev.dtoh_sync_copy(&mixer).expect("dtoh mixer");
                mx.truncate(h * n);
                dumps.push(mx);
                let mut mo = self.dev.dtoh_sync_copy(&mlp_out).expect("dtoh mlp");
                mo.truncate(h * n);
                dumps.push(mo);
            }
            let tot = h * n;
            blaunch!(self, "add_residual_b", grid(tot), (256,1,1), 0,
                (d(&residual), d(&residual), d(&mlp_out), tot as i32));
            pool.release_bf16(mixer, h*n); pool.release_bf16(mlp_out, h*n);
            self.sync_stream();
            let mut ly = self.dev.dtoh_sync_copy(&residual).expect("dtoh layer");
            ly.truncate(h * n);
            dumps.push(ly);
        }
        let out = pool.get_bf16(h * n);
        blaunch!(self, "rmsnorm_b", (n as u32,1,1), (1024,1,1), (4096) as u32,
            (d(&out), d(&residual), d(&self.final_norm), h as i32, n as i32, fbits(cfg.rms_eps)));
        self.sync_stream();
        let mut fnm = self.dev.dtoh_sync_copy(&out).expect("dtoh final_norm");
        fnm.truncate(h * n);
        dumps.push(fnm);

        pool.release_bf16(out, h*n);
        pool.release_bf16(normed, h*n);
        pool.release_bf16(residual, h*n);
        pool.release(cos, n*rdim); pool.release(sin, n*rdim);
        self.sync_stream(); // compute stream done before the non-pool pos/slot CudaSlices drop
        dumps
    }
}
