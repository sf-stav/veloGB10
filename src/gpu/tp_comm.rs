//! `impl GpuModel`: TP all-reduce plumbing (chunking, bf16 / fp32 reductions, the fused reduce+residual+norm epilogue sites),
//! TP probes and the mixer / FFN epilogues. The per-step `agree()` guard and watchdog are load-bearing (AGENTS.md section 2.10). Map: `mod.rs`.

use super::*;

impl GpuModel {
    /// TP=2 sum-all-reduce of a bf16 device buffer across the two boxes — SPIN-WAIT comm (no main-thread
    /// sync). TWO kernels queued on the decode stream, then the main thread races on:
    ///   K1 `tp_gate_copy_signal`: reuse gate → copy local partial into send[e%R] → publish gpu_ready=e
    ///   K2 `tp_wait_add`        : wait cpu_done ≥ e → buf = rank0_partial + rank1_partial
    /// The persistent proxy thread watches gpu_ready, ships length-tag + payload + inline epoch to the
    /// peer, and bounces the peer's arrival into cpu_done to release K2. So the GPU only stalls for the
    /// actual (µs) comm, not the (ms) per-layer launch/sync bubble — the CPU stays arbitrarily far ahead.
    ///
    /// The epoch is NOT a kernel argument: it lives in the device-side ctx counter and K1 increments it.
    /// That is the round-3 capture-hygiene rule, and it is the whole reason graph capture is a no-op
    /// wrap rather than a protocol rewrite. Both ranks call this symmetrically at every reduction →
    /// residual streams identical → greedy SPMD, no token broadcast.
    /// Elements per barrier for a chunked reduction — the chunk size for a reduction too large for one
    /// ring slot. Derived from the SLOT CAPACITY (not the attach payload): the wire protocol carries a
    /// per-epoch length, so prefill chunks nearly fill a slot (up to 16x fewer barriers) while decode
    /// reductions stay one small barrier per site. `TP_SLOT_BYTES - 64` keeps align8(len)+tail inside
    /// the slot; the byte cap is rounded down to an 8 B multiple so chunk*elem_bytes is one too.
    /// --tp-prefill-payload (bytes) caps it lower for A/B escapes. Pure function of values both
    /// ranks share (compile-time slot size + the shipped TpConfig) — lockstep-safe.
    fn tp_chunk_elems(&self, elem_bytes: usize) -> usize {
        static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        let cap = *CAP.get_or_init(|| {
            let max = crate::tp::TP_SLOT_BYTES - 64;
            crate::opts::var(crate::opt!("tp-prefill-payload")).ok().and_then(|v| v.parse().ok())
                .or_else(|| crate::tp::tp_config().and_then(|c| c.prefill_payload))
                .map(|k: usize| k.min(max))
                .unwrap_or(max) & !7
        });
        cap / elem_bytes
    }

    /// Device pointer to the persistent fp32 all-reduce scratch (R9, world>2 mode-3 path). Allocated
    /// at init (before any graph capture), so the mode-3 path is capture-safe.
    pub(super) fn tp_f32_scratch_ptr(&self) -> u64 { *self.tp_f32_scratch.device_ptr() }

    /// P3-1 one-shot all-peers push (world==4, --tp-oneshot, DEFAULT OFF). Resolved from the
    /// transport ctx itself (the same field K1/the proxy read — one source of truth, set at init
    /// from the env on BOTH ranks; a one-sided env mismatches the ring layout and dies loudly at
    /// the first barrier). When on, decode/verify-width all-reduces run ONE K1 (all-peers fanout)
    /// + tp_wait_add_4way (3-tail wait + canonical ((p0+p1)+p2)+p3) instead of the 2-round tree.
    pub(super) fn tp_oneshot_on(&self) -> bool {
        static O: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *O.get_or_init(|| {
            // Same resolution the transport makes at init (env + world==4): deterministic on both
            // ranks; a one-sided env mismatches the ring LAYOUT and dies loudly at the first barrier.
            crate::opts::var(crate::opt!("tp-oneshot")).map_or(false, |v| v != "0") && self.tp_world == 4
        })
    }

    /// P3-1: the transport's DEFAULT payload width (bf16 h*2 — what set_payload installed for the
    /// decode lane). Mirrors c->payload_bytes in net_shim; the per-epoch one-shot predicate on both
    /// sides is "wire_len <= default payload".
    pub(super) fn tp_default_payload_bytes(&self) -> usize {
        self.cfg.hidden_size * 2
    }

    pub(super) fn tp_all_reduce_bf16(&self, buf: &mut B, n: usize) {
        let buf_ptr = *buf.device_ptr();
        // One block on purpose: the flag spin must have a single spinner, and the copy/add is
        // bandwidth-trivial — keeping the wait kernel low-occupancy reserves ~no SM capacity (R3d).
        // CHUNKED: prefill reduces hidden * prompt_len, which is unbounded and can exceed a ring slot by
        // orders of magnitude. Split into slot-sized barriers. Both ranks compute the same n, so the
        // barrier COUNT matches by construction — which is what the protocol requires.
        if self.tp_world > 2 {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| eprintln!("[tp] R10 rounds-complete fp32-accumulate reduce ACTIVE (world={})", self.tp_world));
            // P3-1 one-shot fast path (decode/verify width): a single K1 (the proxy fans the same
            // slot to all 3 peers) + tp_wait_add_4way (3-tail wait, canonical 4-way reduce, one
            // bf16 rounding). The predicate MUST be byte-identical to the proxy's
            // net_use_oneshot (wire_len <= payload_bytes, the DEFAULT decode payload h*2) — the
            // proxy picks one-shot vs tree from len_local on ITS side, and if the two disagree the
            // K2 waits for a one-shot tail the proxy never posted (the mixed-width stage-B wedge).
            // tp_chunk_elems(2) (~slot width) is WRONG here: it classifies wide verify/prefill
            // chunks as one-shot while the proxy (payload_bytes) routes them to the tree.
            if self.tp_oneshot_on() && n * 2 <= self.tp_default_payload_bytes().max(1) {
                let bp = buf_ptr;
                blaunch!(self, "tp_gate_copy_signal", (1u32,1,1), (1024,1,1), 0, (self.tp_ctx_dptr, bp, (n * 2) as u32));
                blaunch!(self, "tp_wait_add_4way", (1u32,1,1), (1024,1,1), 0, (self.tp_ctx_dptr, bp, bp, n as i32, 0i32));
                return;
            }
            // R10 ROOT-CAUSE FIX (world>2): a logical all-reduce must consume `rounds` = log2(world)
            // CONSECUTIVE epochs (one exchange per round; partner = rank ^ 1<<round, derived on-device
            // from the epoch). The pre-R10 driver launched ONE K1+K2 pair per chunk — a single pairwise
            // exchange whose partner alternated with the epoch parity — so at world=4 every "reduced"
            // buffer held p_rank + p_partner and each rank held a different half-sum. That was the
            // deterministic world=4 divergence R9 chased into the GDN state (the transport was
            // byte-clean because each individual exchange was correct — the TREE was incomplete).
            //
            // The loop below is rounds-inner / chunks-outer: chunk c's element range completes ALL
            // rounds (in the persistent fp32 scratch) before the next chunk starts. Round 0 ships the
            // bf16 partial and mode-3 K2 upcasts both sides into fp32; rounds 1.. ship the fp32
            // accumulator and mode-2 K2 adds in place (fp32 end to end, no intermediate bf16 round);
            // the caller's buffer is rounded back ONCE per chunk. Consecutive epochs e0..e0+rounds-1
            // hit every round index exactly once regardless of e0's parity (a step of consecutive
            // integers mod rounds is a permutation), and the canonical lower+upper order in
            // tp_reduce_body makes the folded fp32 sum bitwise rank-identical — so lockstep holds for
            // ANY starting parity, including across CUDA-graph replays.
            //
            // Chunks are sized at tp_chunk_elems(4) — the FP32 wire width — so every round's payload
            // fits one ring slot (round 0's bf16 payload is half that). SPMD-safe: chunking is a pure
            // function of (n, slot bytes) and the launch sequence is identical on every rank.
            let sp = self.tp_f32_scratch_ptr();
            let rounds = self.tp_world.trailing_zeros() as usize;   // log2(world); world>2 ⇒ ≥2
            let chunk = self.tp_chunk_elems(4).max(1);
            let k2 = if self.tp_gpu_recv_on() { "tp_wait_add_g" } else { "tp_wait_add" };
            let mut off = 0usize;
            while off < n {
                let c = (n - off).min(chunk);
                assert!(c <= self.tp_f32_scratch.len(),
                        "world>2 reduce chunk {c} exceeds fp32 scratch {} (raise TP_SLOT_BYTES or shrink prefill chunk)",
                        self.tp_f32_scratch.len());
                let bp = buf_ptr + (off * 2) as u64;   // bf16 partial source (round-0 K1 input)
                blaunch!(self, "tp_gate_copy_signal", (1u32,1,1), (1024,1,1), 0, (self.tp_ctx_dptr, bp, (c * 2) as u32));
                blaunch!(self, k2, (1u32,1,1), (1024,1,1), 0, (self.tp_ctx_dptr, sp, bp, c as i32, 3i32));
                for _ in 1..rounds {
                    blaunch!(self, "tp_gate_copy_signal", (1u32,1,1), (1024,1,1), 0, (self.tp_ctx_dptr, sp, (c * 4) as u32));
                    blaunch!(self, k2, (1u32,1,1), (1024,1,1), 0, (self.tp_ctx_dptr, sp, sp, c as i32, 2i32));
                }
                // Round this chunk's fp32 sum to bf16 ONCE (the single rounding boundary).
                unsafe {
                    self.k.f32tobf16.clone().unwrap().launch_on_stream(&self.stream,
                        cudarc::driver::LaunchConfig { grid_dim: grid(c), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                        (bp, sp, c as i32)).unwrap();
                }
                off += c;
            }
            return;
        }
        let chunk = self.tp_chunk_elems(2).max(1);
        let k2 = if self.tp_gpu_recv_on() { "tp_wait_add_g" } else { "tp_wait_add" };
        let mut off = 0usize;
        while off < n {
            let c = (n - off).min(chunk);
            let p = buf_ptr + (off * 2) as u64;
            blaunch!(self, "tp_gate_copy_signal", (1u32,1,1), (1024,1,1), 0, (self.tp_ctx_dptr, p, (c * 2) as u32));
            blaunch!(self, k2, (1u32,1,1), (1024,1,1), 0, (self.tp_ctx_dptr, p, p, c as i32, 0i32));
            off += c;
        }
    }

    /// AR landing 2 K1-only mode of `tp_all_reduce_bf16` (mode 0): run the gate+copy+signal for a
    /// reduction the caller will finish with the fused `tp_reduce_resnorm_b` (which subsumes K2 +
    /// the residual-add + norm). Returns Some(local_partial_ptr) when the reduction is a SINGLE
    /// barrier (the fused launch must follow); when it CHUNKS, the full K1+K2 chain runs here and
    /// None is returned — `buf` holds the full-rank result and the caller keeps the unfused norm
    /// launch (chunk boundaries are not even row-aligned, so per-row fusion cannot span barriers).
    pub(super) fn tp_all_reduce_bf16_k1(&self, buf: &mut B, n: usize) -> Option<FusedLocal> {
        if self.tp_world > 2 {
            // R10: the fused epilogue kernel performs the LAST round's exchange (it reads the peer
            // slot for its epoch), so this runs the other rounds-1 as full prelim exchanges into the
            // persistent fp32 scratch, then K1s the accumulator for tp_reduce_resnorm_b (mode 1).
            // The fp32 wire width caps the single-barrier size at tp_chunk_elems(4); wider falls
            // back to the unfused chain above.
            if n > self.tp_chunk_elems(4).max(1) {
                self.tp_all_reduce_bf16(buf, n);
                return None;
            }
            assert!(n <= self.tp_f32_scratch.len(), "world>2 fused reduce {n} exceeds fp32 scratch");
            let (bp, sp) = (*buf.device_ptr(), self.tp_f32_scratch_ptr());
            let k2 = if self.tp_gpu_recv_on() { "tp_wait_add_g" } else { "tp_wait_add" };
            let rounds = self.tp_world.trailing_zeros() as usize;
            blaunch!(self, "tp_gate_copy_signal", (1u32,1,1), (1024,1,1), 0, (self.tp_ctx_dptr, bp, (n * 2) as u32));
            blaunch!(self, k2, (1u32,1,1), (1024,1,1), 0, (self.tp_ctx_dptr, sp, bp, n as i32, 3i32));
            for _ in 1..rounds - 1 {
                blaunch!(self, "tp_gate_copy_signal", (1u32,1,1), (1024,1,1), 0, (self.tp_ctx_dptr, sp, (n * 4) as u32));
                blaunch!(self, k2, (1u32,1,1), (1024,1,1), 0, (self.tp_ctx_dptr, sp, sp, n as i32, 2i32));
            }
            blaunch!(self, "tp_gate_copy_signal", (1u32,1,1), (1024,1,1), 0, (self.tp_ctx_dptr, sp, (n * 4) as u32));
            return Some(FusedLocal::F32Scratch);
        }
        let chunk = self.tp_chunk_elems(2).max(1);
        if n <= chunk {
            let p = *buf.device_ptr();
            blaunch!(self, "tp_gate_copy_signal", (1u32,1,1), (1024,1,1), 0, (self.tp_ctx_dptr, p, (n * 2) as u32));
            return Some(FusedLocal::Bf16Buf);
        }
        self.tp_all_reduce_bf16(buf, n);
        None
    }

    /// AR landing 2 K1-only mode of `tp_all_reduce_fp32` (mode 1): see `tp_all_reduce_bf16_k1`.
    /// `out` is the bf16 sink K2 would have written (unused by the fused kernel, kept alive by the
    /// caller); the local partial is the FP32 `partial`.
    pub(super) fn tp_all_reduce_fp32_k1(&self, out: &mut B, partial: &S, n: usize) -> Option<FusedLocal> {
        if self.tp_world > 2 {
            // R10: see tp_all_reduce_bf16_k1 — prelim rounds accumulate the fp32 partial into the
            // persistent scratch, the final K1 ships it, tp_reduce_resnorm_b finishes in mode 1.
            if n > self.tp_chunk_elems(4).max(1) {
                self.tp_all_reduce_fp32(out, partial, n);
                return None;
            }
            assert!(n <= self.tp_f32_scratch.len(), "world>2 fused reduce {n} exceeds fp32 scratch");
            let (pc, sp) = (*partial.device_ptr(), self.tp_f32_scratch_ptr());
            let k2 = if self.tp_gpu_recv_on() { "tp_wait_add_g" } else { "tp_wait_add" };
            let rounds = self.tp_world.trailing_zeros() as usize;
            blaunch!(self, "tp_gate_copy_signal", (1u32,1,1), (1024,1,1), 0, (self.tp_ctx_dptr, pc, (n * 4) as u32));
            blaunch!(self, k2, (1u32,1,1), (1024,1,1), 0, (self.tp_ctx_dptr, sp, pc, n as i32, 2i32));
            for _ in 1..rounds - 1 {
                blaunch!(self, "tp_gate_copy_signal", (1u32,1,1), (1024,1,1), 0, (self.tp_ctx_dptr, sp, (n * 4) as u32));
                blaunch!(self, k2, (1u32,1,1), (1024,1,1), 0, (self.tp_ctx_dptr, sp, sp, n as i32, 2i32));
            }
            blaunch!(self, "tp_gate_copy_signal", (1u32,1,1), (1024,1,1), 0, (self.tp_ctx_dptr, sp, (n * 4) as u32));
            return Some(FusedLocal::F32Scratch);
        }
        let chunk = self.tp_chunk_elems(4).max(1);
        if n <= chunk {
            let p = *partial.device_ptr();
            blaunch!(self, "tp_gate_copy_signal", (1u32,1,1), (1024,1,1), 0, (self.tp_ctx_dptr, p, (n * 4) as u32));
            return Some(FusedLocal::F32Buf);
        }
        self.tp_all_reduce_fp32(out, partial, n);
        None
    }

    /// Row-parallel GEMM that leaves its output as the UNROUNDED FP32 accumulator. Only the NVFP4 MMA
    /// path supports it — that is the production 27B weight format for every row-parallel site — and any
    /// other format is refused rather than silently falling back to a bf16 round, which would defeat the
    /// entire point of the FP32-preserving reduction.
    pub(super) fn gemm_act_f32(&self, w: &W, x: &B, out: &mut S, inn: usize, outn: usize, batch: usize) {
        let _rot_guard; let x: &B = if self.is_rotated(w) { _rot_guard = self.rot_x(x, inn, batch); &*_rot_guard } else { x };
        match w {
            W::Nvfp4 { qweight, scales, gs, .. } if batch <= MAX_VERIFY => {
                // Same dispatch as gemm_act: the split-K reduce writes the FP32 partial to Cf with
                // exactly mma_epilogue's semantics (gs applied, no rounding), so the FP32-preserving
                // reduction keeps working — and the TP=2 down-proj half (K=8704) gets S=2 as well.
                // NB: the TP Cf split path is NOT two-box-verified this session (default OFF); a
                // TP=2 run with --splitk-gemm needs the two-box binv gate before any claim.
                self.launch_gemm_fp4(0, *qweight.device_ptr() as u64, *scales.device_ptr() as u64,
                                     d(gs), d(x), outn, inn, batch, d(out));
            }
            // The -mixed recipes hold the GDN out_proj in FP8: the row-parallel partial must be
            // FP32-preserving there too, or --tp-fp32-partials is silently a no-op on exactly
            // the layers that dominate a hybrid model. Same epilogue Cf path as fp4.
            W::Fp8 { data, row_scale, .. } if batch <= MAX_VERIFY => {
                blaunch!(self, "gemm_mma_fp8_b", ((outn / 16) as u32,1,1), (256,1,1), 0,
                    (0u64, *data.device_ptr() as u64, d(row_scale),
                     d(x), outn as i32, inn as i32, batch as i32, d(out)));
            }
            _ => panic!("FP32-preserving TP partials require NVFP4 or FP8 row-parallel weights at batch<={};                        got another format. Disable --tp-fp32-partials for this model.", MAX_VERIFY),
        }
    }

    /// FP32-preserving all-reduce: ship the UNROUNDED partial, sum in FP32 on both ranks, round to bf16
    /// exactly once. `partial` is the GEMM's FP32 accumulator; `out` is the bf16 buffer the residual
    /// consumes. Same two kernels and the same ring as the bf16 path — only the payload width changes,
    /// which is why the slots were sized for FP32 from day one.
    pub(super) fn tp_all_reduce_fp32(&self, out: &mut B, partial: &S, n: usize) {
        let (o, p) = (*out.device_ptr(), *partial.device_ptr());
        let k2 = if self.tp_gpu_recv_on() { "tp_wait_add_g" } else { "tp_wait_add" };
        // R10 (world>2): a logical all-reduce needs `rounds` exchanges (see tp_all_reduce_bf16).
        // Round 0 ships the fp32 partial, mode-2 K2 accumulates into the persistent fp32 scratch;
        // rounds 1.. ship the scratch in place; the single bf16 rounding happens once per chunk.
        if self.tp_world > 2 {
            let sp = self.tp_f32_scratch_ptr();
            let rounds = self.tp_world.trailing_zeros() as usize;
            let chunk = self.tp_chunk_elems(4).max(1);
            let mut off = 0usize;
            while off < n {
                let c = (n - off).min(chunk);
                let (oc, pc) = (o + (off * 2) as u64, p + (off * 4) as u64);
                blaunch!(self, "tp_gate_copy_signal", (1u32,1,1), (1024,1,1), 0, (self.tp_ctx_dptr, pc, (c * 4) as u32));
                blaunch!(self, k2, (1u32,1,1), (1024,1,1), 0, (self.tp_ctx_dptr, sp, pc, c as i32, 2i32));
                for _ in 1..rounds {
                    blaunch!(self, "tp_gate_copy_signal", (1u32,1,1), (1024,1,1), 0, (self.tp_ctx_dptr, sp, (c * 4) as u32));
                    blaunch!(self, k2, (1u32,1,1), (1024,1,1), 0, (self.tp_ctx_dptr, sp, sp, c as i32, 2i32));
                }
                unsafe {
                    self.k.f32tobf16.clone().unwrap().launch_on_stream(&self.stream,
                        cudarc::driver::LaunchConfig { grid_dim: grid(c), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                        (oc, sp, c as i32)).unwrap();
                }
                off += c;
            }
            return;
        }
        let chunk = self.tp_chunk_elems(4).max(1);
        let mut off = 0usize;
        while off < n {
            let c = (n - off).min(chunk);
            let (oc, pc) = (o + (off * 2) as u64, p + (off * 4) as u64);
            blaunch!(self, "tp_gate_copy_signal", (1u32,1,1), (1024,1,1), 0, (self.tp_ctx_dptr, pc, (c * 4) as u32));
            blaunch!(self, k2, (1u32,1,1), (1024,1,1), 0, (self.tp_ctx_dptr, oc, pc, c as i32, 1i32));
            off += c;
        }
    }

    /// FP32 in-place all-reduce (tp_wait_add mode 2): ship the local f32 buffer, sum in f32 on both
    /// ranks, NO rounding anywhere — the vocab-parallel LM-head gather for `logits_batch_f32`
    /// (hy_v3's enable_lm_head_fp32 contract). Each rank's half-buffer is exact +0.0 outside its own
    /// rows, so the sum IS the gather and reproduces the replicated head's logits value-for-value.
    /// Chunked like the bf16 path: the barrier count is a pure function of (n, chunk) — SPMD-safe.
    pub(super) fn tp_all_reduce_f32(&self, buf: &mut S, n: usize) {
        self.tp_all_reduce_f32_ptr(*buf.device_ptr(), n);
    }

    /// Raw-pointer variant (same mode-2 no-rounding f32 gather), for callers holding a persistent
    /// scratch behind &self rather than a CudaSlice they can borrow.
    fn tp_all_reduce_f32_ptr(&self, p: u64, n: usize) {
        let chunk = self.tp_chunk_elems(4).max(1);
        let k2 = if self.tp_gpu_recv_on() { "tp_wait_add_g" } else { "tp_wait_add" };
        let rounds = if self.tp_world > 2 { self.tp_world.trailing_zeros() as usize } else { 1 };
        let mut off = 0usize;
        while off < n {
            let c = (n - off).min(chunk);
            let pc = p + (off * 4) as u64;
            // R10: `rounds` exchanges per chunk (world==2 keeps its single exchange verbatim —
            // rounds==1 there, so the loop is the legacy sequence exactly).
            for _ in 0..rounds {
                blaunch!(self, "tp_gate_copy_signal", (1u32,1,1), (1024,1,1), 0, (self.tp_ctx_dptr, pc, (c * 4) as u32));
                blaunch!(self, k2, (1u32,1,1), (1024,1,1), 0, (self.tp_ctx_dptr, pc, pc, c as i32, 2i32));
            }
            off += c;
        }
    }

    /// Batch the TP link is sized for. 1 in production; `--tp-batch-probe=N` widens it so we can
    /// measure how a batch-N forward behaves under sharding (the MTP verify cost shape).
    pub(super) fn tp_probe_batch(&self) -> usize {
        static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *N.get_or_init(|| crate::opts::var(crate::opt!("tp-batch-probe")).ok()
            .and_then(|v| v.parse().ok())
            .or_else(|| crate::tp::tp_config().and_then(|c| c.batch_probe))
            .filter(|n| *n >= 1 && *n <= MAX_VERIFY).unwrap_or(1))
    }

    /// Q6 PROBE — does "a batch-N forward costs ≈ a batch-1 forward" survive TP?
    ///
    /// This is the whole mechanism of speculative decoding: a batch-N verify reads the weights ONCE and
    /// produces N drafted positions. Single-node it holds tightly (N=8 costs +4.4 % over N=1). If it does
    /// NOT hold under TP — because the all-reduce payload widens with N, or the sharded GEMM shapes
    /// degrade at N columns — then MTP-under-TP cannot pay and Stage 3A stops.
    ///
    /// It measures the COST SHAPE only: identical kernel shapes and a hidden*N all-reduce per site. It is
    /// not verify semantics, and the tokens it produces are meaningless.
    pub fn tp_batch_probe(&self, batch: usize, iters: usize, max_seq_len: usize) {
        let mut pool = Pool::new(self.dev.clone());
        let mut state = self.new_batch_state(batch.max(1), batch.max(1), max_seq_len);
        let mut bufs = self.new_decode_buffers(batch);
        let toks: Vec<i32> = (0..batch).map(|i| (1000 + i) as i32).collect();
        let pos: Vec<i32> = (0..batch).map(|_| 8i32).collect();
        let slots: Vec<i32> = (0..batch).map(|i| i as i32).collect();
        self.dev.htod_sync_copy_into(&toks, &mut bufs.tokens_dev).unwrap();
        self.dev.htod_sync_copy_into(&pos, &mut bufs.pos_dev).unwrap();
        self.dev.htod_sync_copy_into(&slots, &mut bufs.slot_ids_dev).unwrap();
        self.dev.synchronize().unwrap();

        for _ in 0..8 {   // warm the pool, cuBLAS, and both proxies
            self.forward_decode_gpu(&mut pool, &mut bufs, &mut state, max_seq_len, max_seq_len, batch);
            self.sync_stream();
        }
        let mut ms: Vec<f64> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = std::time::Instant::now();
            self.forward_decode_gpu(&mut pool, &mut bufs, &mut state, max_seq_len, max_seq_len, batch);
            self.sync_stream();
            ms.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = ms[ms.len() / 2];
        eprintln!("[tp{}] BATCH PROBE N={batch}: p50 {p50:.2} ms/forward  ({:.2} ms per drafted position)",
                  self.tp_rank, p50 / batch as f64);
    }

    /// PROBE 2 — measure a synthetic MTP step under TP directly, instead of assembling it from five
    /// separate estimates. Runs the real sequence with scripted (ignored) acceptance:
    ///   (depth-1) × draft  →  verify forward at batch=depth  →  GDN rollback
    /// Timing only; the tokens it produces are meaningless. `mtp_reprime` is excluded and added as its
    /// measured constant, because it needs a real accepted-prefix to reprime from.
    pub fn tp_synthetic_step_probe(&self, depth: usize, iters: usize, max_seq_len: usize) {
        let h = self.cfg.hidden_size;
        let mut pool = Pool::new(self.dev.clone());
        let nslots = depth.max(2);
        let mut state = self.new_batch_state(nslots, nslots, max_seq_len);
        let mut bufs = self.new_decode_buffers(depth);
        let (_nkv, hd) = (self.cfg.num_kv_heads, self.cfg.head_dim);
        let mut mtp_kc = self.dev.alloc_zeros::<half::bf16>(self.mtp_kc_elems(max_seq_len)).unwrap();
        let mut mtp_vc = self.dev.alloc_zeros::<half::bf16>(self.mtp_kv_heads() * max_seq_len * hd).unwrap();
        self.dev.memset_zeros(&mut mtp_kc).unwrap();
        self.dev.memset_zeros(&mut mtp_vc).unwrap();
        let hidden = self.dev.alloc_zeros::<half::bf16>(h).unwrap();
        let toks: Vec<i32> = (0..depth).map(|i| (1000 + i) as i32).collect();
        let pos: Vec<i32> = (0..depth).map(|_| 16i32).collect();
        let slots: Vec<i32> = (0..depth).map(|i| i as i32).collect();
        self.dev.htod_sync_copy_into(&toks, &mut bufs.tokens_dev).unwrap();
        self.dev.htod_sync_copy_into(&pos, &mut bufs.pos_dev).unwrap();
        self.dev.htod_sync_copy_into(&slots, &mut bufs.slot_ids_dev).unwrap();
        self.dev.synchronize().unwrap();
        let (kc, vc) = (*mtp_kc.device_ptr(), *mtp_vc.device_ptr());

        let mut step = |p: &mut Pool, st: &mut BatchGpuState, b: &mut DecodeBuffers| {
            for d in 0..depth.saturating_sub(1) {
                let o = self.mtp_draft_step(p, &hidden, 1000 + d as i32, 16 + d, kc, vc, max_seq_len);
                p.release_bf16(o, h);
            }
            self.forward_decode_gpu(p, b, st, max_seq_len, max_seq_len, depth);
            self.copy_gdn_slot(st, 0, 1);
            self.sync_stream();
        };
        for _ in 0..4 { step(&mut pool, &mut state, &mut bufs); }
        let mut ms: Vec<f64> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = std::time::Instant::now();
            step(&mut pool, &mut state, &mut bufs);
            ms.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = ms[ms.len() / 2];
        eprintln!("[probe2] rank {} depth {depth}: synthetic step p50 {p50:.2} ms \
                   (+1.46 reprime = {:.2} ms) — drafts {} + verify N={depth} + rollback",
                  self.tp_rank, p50 + 1.46, depth.saturating_sub(1));
    }

    /// Is the FP32-preserving reduction active? bf16 partials round EACH rank's partial before summing,
    /// which is a real (if small) precision hole the single-node path does not have; FP32 partials close
    /// it, leaving only reassociation. Opt-in until it has run the long divergence gate.
    pub(super) fn tp_fp32_partials(&self) -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        self.tp_world > 1 && *ON.get_or_init(|| crate::opts::var(crate::opt!("tp-fp32-partials")).is_ok()
            || crate::tp::tp_config().map(|c| c.fp32_partials).unwrap_or(false))
    }

    /// One row-parallel reduction site: FP32-preserving when enabled, bf16 otherwise. AR landing 2:
    /// with `fuse` set (the caller runs the fused `tp_reduce_resnorm_b` epilogue), a SINGLE-barrier
    /// reduction stops after K1 and returns the LOCAL partial (Some); chunked reductions and
    /// unfused callers run today's K1+K2 chain and return None (full-rank value in `out`).
    pub(super) fn tp_reduce_site(&self, pool: &mut Pool, w: &W, x: &B, out: &mut B,
                      inn: usize, outn: usize, batch: usize, fuse: bool) -> Option<FusedPartial> {
        // FP32-preserving partials where the width allows them (decode/verify, batch <= MAX_VERIFY):
        // one rounding boundary for the whole reduction. bf16 partials round each rank's partial
        // BEFORE the cross-rank sum (measured 1868/5120 mismatched elements @2 ULP vs 7/5120 @1 ULP
        // for fp32), and that error compounds in the residual stream across tokens — the MTP draft
        // head is scale-sensitive, so bf16 partials cost measurable acceptance at long context.
        // Wide prefill reductions stay on the chunked bf16 path: gemm_act_f32 only supports
        // batch <= MAX_VERIFY, and prefill reassociation is accepted anyway. (Previously the flag
        // panicked on ANY batch > 16, which made FP32 partials unusable end-to-end.)
        if self.tp_fp32_partials() && batch <= MAX_VERIFY {
            let mut p = pool.get(outn * batch);
            self.gemm_act_f32(w, x, &mut p, inn, outn, batch);
            if fuse {
                if let Some(fl) = self.tp_all_reduce_fp32_k1(out, &p, outn * batch) {
                    match fl {
                        FusedLocal::F32Scratch => { pool.release(p, outn * batch); return Some(FusedPartial::F32Scratch); }
                        _ => return Some(FusedPartial::F32(p)),   // p stays alive for the fused launch
                    }
                }
                pool.release(p, outn * batch);           // chunked — the full chain ran
                return None;
            }
            self.tp_all_reduce_fp32(out, &p, outn * batch);
            pool.release(p, outn * batch);
        } else {
            self.gemm_act(w, x, out, inn, outn, batch);
            // Phase-8 [fh2]: pre- vs post-reduce hash of the FFN's local partial — if pre matches
            // cold-vs-warm but post differs, the cross-rank reduce is the divergence source.
            if crate::opts::var(crate::opt!("layer-fullhash")).is_ok() && batch > 16 {
                self.sync_stream();
                let host = self.dev.dtoh_sync_copy(out).unwrap_or_default();
                let n_real = (outn * batch).min(host.len());
                let mut xx: u64 = 1469598103934665603;
                for v in host.iter().take(n_real) { for b in v.to_bits().to_le_bytes() { xx ^= b as u64; xx = xx.wrapping_mul(1099511628211); } }
                eprintln!("[fh2] pre {outn}x{batch} {:012x}", xx & 0xFFFFFFFFFFFF);
            }
            if fuse {
                if let Some(fl) = self.tp_all_reduce_bf16_k1(out, outn * batch) {
                    return Some(match fl {
                        FusedLocal::F32Scratch => FusedPartial::F32Scratch,
                        _ => FusedPartial::Bf16,         // the local partial IS `out`
                    });
                }
                return None;                             // chunked — the full chain ran
            }
            self.tp_all_reduce_bf16(out, outn * batch);
            if crate::opts::var(crate::opt!("layer-fullhash")).is_ok() && batch > 16 {
                self.sync_stream();
                let host = self.dev.dtoh_sync_copy(out).unwrap_or_default();
                let n_real = (outn * batch).min(host.len());
                let mut xx: u64 = 1469598103934665603;
                for v in host.iter().take(n_real) { for b in v.to_bits().to_le_bytes() { xx ^= b as u64; xx = xx.wrapping_mul(1099511628211); } }
                eprintln!("[fh2] post {outn}x{batch} {:012x}", xx & 0xFFFFFFFFFFFF);
            }
        }
        None
    }

    /// AR landing 2: launch `tp_reduce_resnorm_b` with a K1-only local partial (mode 0 = bf16 in
    /// `sink`, mode 1 = fp32 in `partial`), in place of the K2 + norm two-launch chain, then
    /// release the reduce's buffers. PLAIN launch (§5 rule 2 — K1 → this kernel is a hard stream
    /// edge: it reads c->epoch; its own launch_dependents releases the downstream PSS dependents).
    /// `flavor` selects the norm body: 0 = S (mixer site, sum_sq from the UNROUNDED fp32 add),
    /// 1 = Q (FFN site, sum_sq from the ROUNDED residual — bit-exact to add_residual_b + rmsnorm_b).
    fn launch_tp_reduce_resnorm(&self, pool: &mut Pool, normed_ptr: u64, res_ptr: u64, fr: FusedReduce,
                                w: &S, h: usize, batch: usize, flavor: i32) {
        // The kernel DERIVES the wire length from (n, fp32_mode) — the K1 at these sites shipped
        // exactly n*elem_bytes, so no nbytes arg is needed (tp_wait_add_g's SPMD derivation).
        // R10 world>2 mode 1: the fp32 local is the PERSISTENT fp32 scratch (the prelim rounds'
        // accumulator, already K1'd by the k1 helper) — partial is None there by construction.
        let local = match &fr.partial {
            Some(p) => d(p),
            None if fr.mode == 1 && self.tp_world > 2 => self.tp_f32_scratch_ptr(),
            None => d(&fr.sink),
        };
        blaunch!(self, "tp_reduce_resnorm_b", (batch as u32,1,1), (1024,1,1), (4096) as u32,
            (self.tp_ctx_dptr, normed_ptr, res_ptr, local, d(w), h as i32, batch as i32,
             fbits(self.cfg.rms_eps), fr.mode, flavor));
        pool.release_bf16(fr.sink, h * batch);
        if let Some(p) = fr.partial { pool.release(p, h * batch); }
    }

    /// AR landing 2: the MIXER epilogue — the flavor-S fused_res_rmsnorm_b for a full-rank mixer,
    /// or tp_reduce_resnorm_b (flavor S) consuming a K1-only local partial. The mixer site is
    /// fused whenever the reduce is (its norm kernel is always the fused flavor-S one — there is
    /// no unfused mixer epilogue), so only --tp-reduce-fuse gates it, not fuse_residual_norm.
    /// `e9` selects the unfused launch form: true = the decode loop's blaunch_e9! (PSS), false =
    /// the verify loop's plain blaunch! — each site keeps its current launch form byte-for-byte.
    pub(super) fn mixer_epilogue(&self, pool: &mut Pool, normed_ptr: u64, res_ptr: u64, mixer: ReduceOut,
                      w: &S, h: usize, batch: usize, e9: bool) {
        match mixer {
            ReduceOut::Full(m) => {
                if e9 {
                    blaunch_e9!(self, "fused_res_rmsnorm_b", (batch as u32,1,1), (1024,1,1), (4096) as u32,
                        (normed_ptr, res_ptr, d(&m), d(w), h as i32, batch as i32, fbits(self.cfg.rms_eps)));
                } else {
                    blaunch!(self, "fused_res_rmsnorm_b", (batch as u32,1,1), (1024,1,1), (4096) as u32,
                        (normed_ptr, res_ptr, d(&m), d(w), h as i32, batch as i32, fbits(self.cfg.rms_eps)));
                }
                pool.release_bf16(m, h * batch);
            }
            ReduceOut::Fused(fr) => self.launch_tp_reduce_resnorm(pool, normed_ptr, res_ptr, fr, w, h, batch, 0),
        }
    }

    /// AR landing 2: the FFN epilogue — flavor-Q fused_res_rmsnorm_q_b on a full-rank FFN output,
    /// or tp_reduce_resnorm_b (flavor Q) on a K1-only local partial; the unfused config keeps
    /// add_residual_b. Fused only when `fuse` (the residual-fusion config — the FFN site's norm
    /// kernel IS fused_res_rmsnorm_q_b only then; the unfused chain's rounding placement differs).
    /// `e9` selects the unfused launch form per site (see mixer_epilogue).
    pub(super) fn ffn_epilogue(&self, pool: &mut Pool, normed_ptr: u64, res_ptr: u64, mlp_out: ReduceOut,
                    fuse: bool, w: &S, h: usize, batch: usize, e9: bool) {
        if fuse {
            match mlp_out {
                ReduceOut::Full(m) => {
                    if e9 {
                        blaunch_e9!(self, "fused_res_rmsnorm_q_b", (batch as u32,1,1), (1024,1,1), (4096) as u32,
                            (normed_ptr, res_ptr, d(&m), d(w), h as i32, batch as i32, fbits(self.cfg.rms_eps)));
                    } else {
                        blaunch!(self, "fused_res_rmsnorm_q_b", (batch as u32,1,1), (1024,1,1), (4096) as u32,
                            (normed_ptr, res_ptr, d(&m), d(w), h as i32, batch as i32, fbits(self.cfg.rms_eps)));
                    }
                    pool.release_bf16(m, h * batch);
                }
                ReduceOut::Fused(fr) => self.launch_tp_reduce_resnorm(pool, normed_ptr, res_ptr, fr, w, h, batch, 1),
            }
        } else {
            let ReduceOut::Full(m) = mlp_out else {
                unreachable!("AR landing 2: fused reduce with the unfused FFN epilogue (--fuse-residual off)");
            };
            let tot = h * batch;
            if e9 {
                blaunch_e9!(self, "add_residual_b", grid(tot), (256,1,1), 0, (res_ptr, res_ptr, d(&m), tot as i32));
            } else {
                blaunch!(self, "add_residual_b", grid(tot), (256,1,1), 0, (res_ptr, res_ptr, d(&m), tot as i32));
            }
            pool.release_bf16(m, h * batch);
        }
    }

    /// Prefill-width output padding for the W8A8 lane (writes tp64 rows — see mlp_batch).
    /// TRIPWIRE_SPEC §2: ONE definition. The hand-inlined copy in `mlp_batch` and the bare
    /// `div_ceil(64)*64` in the pf8 sink assert are the same numeric rule written three ways, and
    /// a numeric rule duplicated across sites is how the next divergence happens.
    pub(super) fn pf8_pad(&self, n: usize) -> usize { pf8_pad_pub(n) }
}
