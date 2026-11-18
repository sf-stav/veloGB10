//! `impl GpuModel`: the prefill GEMM entry points (`gemm_fp8_prefill_e`, `gemm_quant_prefill*`). Serving GEMM invariant:
//! the fp4 mma path is batch-invariant (fixed instruction sequence, fixed-order reductions); never fall back silently to dequant. Map: `mod.rs`.

use super::*;

impl GpuModel {
    /// batched bf16 GEMM: out[outn,B] = W_bf16[outn,inn] @ X_f32[inn,B]
    /// Pure bf16→bf16 GEMM for the activation path: zero conversion kernels.
    /// out[outn,B] = W_bf16[outn,inn] @ X_bf16[inn,B]
    /// Batch-invariant custom kernel for batch <= BINV_BATCH (the MTP verify runs at N=2, per-lane
    /// decode at N=1) so the verify's column 0 is bit-identical to N=1 → no cuBLAS N=1-vs-N=2 drift
    /// → 9B/27B MTP is coherent. cuBLAS for larger batch (multi-lane decode) where tensor cores win.
    /// The custom kernel is also slightly faster than cuBLAS at batch 1 (more coalesced, less overhead).
    /// Dequantize a weight into persistent bf16 scratch and immediately GEMM with it (PREFILL only).
    ///
    /// The scratch is sized once to the largest tensor and reused, so there is no hot-path alloc. The
    /// dequant and the GEMM happen together while the buffer is held, so the scratch is never aliased.
    ///
    /// This is emphatically NOT the decode path: dequantizing to bf16 there would stream bf16 bytes
    /// again and throw away the whole point. It exists because prefill reads each weight once for many
    /// tokens, so bandwidth stops being the constraint and cuBLAS's tensor cores win.
    /// F8 parity lane: W8A8 e4m3 prefill GEMM (gemm_fp8_prefill_e_b). Caller guarantees
    /// W::Fp8Blk, inn%128==0, outn%128==0, batch>=256. One-time per-tensor repack into
    /// k32 e4m3 A-frag panels (fp8e_map); per-call X scales (per token x 128-K) + packed
    /// B panels; the GEMM folds w_s(block 128x128) x x_s(token,128K) every 128-K span.
    #[allow(clippy::too_many_arguments)]
    fn gemm_fp8_prefill_e(&self, w: &W, x: &B, out: &mut B, inn: usize, outn: usize, batch: usize, trace: bool) {
        let (data, blk_scale) = match w { W::Fp8Blk { data, blk_scale, .. } => (data, blk_scale), _ => unreachable!() };
        let dev = &self.dev;

        // (a) lazy one-time repack (the decode kernel keeps the ORIGINAL k16 layout —
        // this side buffer is read only by gemm_fp8_prefill_e_b). The repack kernel
        // writes every byte, so the fresh (unzeroed) async alloc is fully initialized.
        let wptr = *data.device_ptr() as u64;
        {
            let mut map = self.fp8e_map.lock().unwrap();
            if !map.contains_key(&wptr) {
                let buf = dev.alloc_zeros::<u8>(outn * inn).unwrap();
                blaunch!(self, "fp8_repack_e_b", (2048, 1, 1), (256, 1, 1), 0,
                    (wptr, *buf.device_ptr() as u64, outn as i32, inn as i32));
                if trace { eprintln!("[pf8] repacked {outn}x{inn} -> k32 panels"); }
                map.insert(wptr, buf);
            }
        }
        let panels = *self.fp8e_map.lock().unwrap()[&wptr].device_ptr() as u64;

        // (b) scratch: Bq panels (tp64 x inn bytes) + Xs scales (tp64 x inn/128 f32).
        // Precondition (documented above): batch >= 256, where pf8_pad_pub is exactly
        // div_ceil(64)*64 — the consolidation cannot weaken the bound or shrink `bq_bytes`/
        // `xs_elems` below what the kernel writes. Below 256 this function is never called.
        let tp64 = pf8_pad_pub(batch);
        assert!(out.len() >= tp64 * outn,
                "pf8 lane OOB: out has {} elems < tp64*outn = {}*{} — caller must size the buffer to the padded token count (see mlp_batch bn)",
                out.len(), tp64, outn);
        let bq_bytes = tp64 * inn;
        let xs_elems = tp64 * (inn / 128);
        {
            let mut sc = self.fp8e_scratch.lock().unwrap();
            if sc.as_ref().map_or(true, |(b, s)| b.len() < bq_bytes || s.len() < xs_elems) {
                *sc = None; // drop old on the blocking stream (ordered after in-flight use)
                *sc = Some((dev.alloc_zeros::<u8>(bq_bytes).unwrap(),
                            dev.alloc_zeros::<f32>(xs_elems).unwrap()));
            }
        }
        let (bq, xs) = self.fp8e_scratch.lock().unwrap().as_ref().map(|(b, s)| (*b.device_ptr() as u64, *s.device_ptr() as u64)).unwrap();

        // (c) fused single-pass quant over the padded token count (pad rows: identity
        // scale, zero codes — never touch X beyond the real rows). One X read.
        let kspans = (inn / 128) as i32;
        let jobs = (tp64 / 8) * (inn / 128) * 32;
        blaunch!(self, "fp8_quant_pack_b", ((jobs as u32 + 255) / 256, 1, 1), (256, 1, 1), 0,
            (d(x), bq, xs, batch as i32, inn as i32, kspans));

        // (d) the GEMM: grid = (outn/128) x (tp64/64) rasterized 1-D in-kernel.
        let blocks = ((outn / 128) * (tp64 / 64)) as u32;
        blaunch!(self, "gemm_fp8_prefill_e_b", (blocks, 1, 1), (256, 1, 1), 4 * 6144,
            (panels, bq, xs, d(blk_scale), d(out), outn as i32, tp64 as i32, inn as i32));
    }

    pub(super) fn gemm_quant_prefill(&self, w: &W, x: &B, out: &mut B, inn: usize, outn: usize, batch: usize) {
        self.gemm_quant_prefill_inner(w, x, out, inn, outn, batch, true)
    }
    /// TRUE reference chain (dequant + cuBLAS): pf4 disabled. The old CHECK arm routed
    /// through `_inner`, which re-dispatched the pf4 path — the "reference" was the fast
    /// path itself (relL2 0.0 exactly, comparing a thing to itself). Found 2026-08-25.
    pub(super) fn gemm_quant_prefill_slow(&self, w: &W, x: &B, out: &mut B, inn: usize, outn: usize, batch: usize) {
        self.gemm_quant_prefill_inner(w, x, out, inn, outn, batch, false)
    }
    /// P4 B2 dispatch + original dequant/cuBLAS path.
    fn gemm_quant_prefill_inner(&self, w: &W, x: &B, out: &mut B, inn: usize, outn: usize, batch: usize, allow_pf4: bool) {

        // F8 parity lane (2026-09-07): W8A8 e4m3 prefill GEMM for the Fp8Blk trunk (the
        // vLLM-recipe equivalence item). Env-gated while experimental: --fp8-prefill=1.
        // Shapes: M%128==0 (every sharded projection), K%128==0, batch>=PF8_MIN. Falls
        // through to dequant+cuBLAS otherwise. --pf8-trace=1 logs the gate + first
        // repacks; --pf8-check=1 runs the slow path too and prints per-site rel-L2
        // (the S3 cross-chain rule: self-consistency proves nothing).
        static PF8_ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        static PF8_TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        static PF8_CHECK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        static PF8_N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        const PF8_MIN: usize = 256;
        // Phase-8 (F9): DEFAULT ON. The dequant+cuBLASLt fallback's algo heuristic reads the
        // operand ADDRESSES, so the same (M,N,K) rounds differently per pool state — on the TP
        // NODE rank that made warm prefix-reuse prefills diverge from cold (deterministically per
        // request ordinal). The W8A8 lane is a fixed-order custom kernel: address-invariant and
        // measured bit-identical to the cuBLAS path's outputs on the head rank (detP==detS
        // streams), so the flip is a determinism fix, not a numerics change. --fp8-prefill=0
        // restores the cuBLAS arm (diagnostics A/B); =1 remains accepted for symmetry.
        let pf8_on = allow_pf4
            && *PF8_ON.get_or_init(|| crate::opts::var(crate::opt!("fp8-prefill")).map_or(true, |v| v != "0"))
            && { static M: std::sync::Once = std::sync::Once::new();
                 M.call_once(|| eprintln!("[pf8-on] W8A8 prefill GEMM lane ENGAGED (default)")); true }
            && matches!(w, W::Fp8Blk { .. })
            && inn % 128 == 0
            && outn % 128 == 0
            && batch >= PF8_MIN;
        if pf8_on {
            let trace = *PF8_TRACE.get_or_init(|| crate::opts::var(crate::opt!("pf8-trace")).is_ok());
            let check = *PF8_CHECK.get_or_init(|| crate::opts::var(crate::opt!("pf8-check")).is_ok());
            let c = PF8_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if trace && (c < 5 || c % 200 == 0) {
                eprintln!("[pf8 #{c}] outn={outn} inn={inn} batch={batch}");
            }
            if check && c < 8 {
                // cross-check: slow reference into a clone, then the fast path into `out`
                let mut ref_out = out.clone();
                self.gemm_quant_prefill_slow(w, x, &mut ref_out, inn, outn, batch);
                self.gemm_fp8_prefill_e(w, x, out, inn, outn, batch, trace);
                self.dev.synchronize().unwrap();
            if crate::opts::var(crate::opt!("pf8-dump")).is_ok() && PF8_N.load(std::sync::atomic::Ordering::Relaxed) == 1 {
                // one-shot evidence dump: exact inputs + fast-path output for offline replay
                if let W::Fp8Blk { data, blk_scale, nb_k, m, k } = w {
                    let (xh, wh, sh) = (self.dev.dtoh_sync_copy(x).unwrap(),
                                        self.dev.dtoh_sync_copy(data).unwrap(),
                                        self.dev.dtoh_sync_copy(blk_scale).unwrap());
                    let oh = self.dev.dtoh_sync_copy(out).unwrap();
                    let hdr = format!("{outn} {inn} {batch} {} {} {}\n", *m, *k, *nb_k);
                    std::fs::write("/tmp/pf8_dump.hdr", hdr).unwrap();
                    let mut xb = Vec::with_capacity(xh.len() * 2); for v in &xh { xb.extend_from_slice(&v.to_le_bytes()); }
                    std::fs::write("/tmp/pf8_dump_x.bin", xb).unwrap();
                    std::fs::write("/tmp/pf8_dump_wt.bin", &wh).unwrap();
                    let mut sb = Vec::with_capacity(sh.len() * 4); for v in &sh { sb.extend_from_slice(&v.to_le_bytes()); }
                    std::fs::write("/tmp/pf8_dump_ws.bin", sb).unwrap();
                    let mut ob = Vec::with_capacity(oh.len() * 2); for v in &oh { ob.extend_from_slice(&v.to_le_bytes()); }
                    std::fs::write("/tmp/pf8_dump_out.bin", ob).unwrap();
                    eprintln!("[pf8-dump] wrote the x, wt, ws and out dumps for outn={outn} inn={inn} batch={batch}");
                }
            }

                let a = self.dev.dtoh_sync_copy(out).unwrap();
                let b = self.dev.dtoh_sync_copy(&ref_out).unwrap();
                let mut num = 0f64;
                let mut den = 0f64;
                for r in 0..batch {
                    for q in 0..outn {
                        let i = r * outn + q;
                        let av = half::bf16::to_f32(a[i]);
                        let bv = half::bf16::to_f32(b[i]);
                        num += ((av - bv) * (av - bv)) as f64;
                        den += (bv * bv) as f64;
                    }
                }
                eprintln!("[pf8-check #{c}] outn={outn} inn={inn} batch={batch} rel-L2 = {:.3e}",
                          (num / den).sqrt());
                return;
            }
            self.gemm_fp8_prefill_e(w, x, out, inn, outn, batch, trace);
            return;
        }
        // NVFP4 W4A4 prefill (src/w4a4.rs): enabled tensors above the verify width run the
        // block-scaled FP4 tensor-core GEMM on the standard tiled weights (no repack, no copy).
        if let W::Nvfp4 { qweight, .. } = w { self.igs_tap_x(*qweight.device_ptr() as u64, x, inn * batch); }
        if let (Some(w4), W::Nvfp4 { qweight, m, k, .. }) = (&self.w4a4, w) {
            let ptr = *qweight.device_ptr() as u64;
            if allow_pf4 && w4.on(ptr) && batch > MAX_VERIFY && *k == inn && *m == outn && inn % 64 == 0
                && x.len() >= batch * inn && out.len() >= batch * outn {
                self.w4a4_dense(w4, w, x, out, inn, outn, batch, w4.xgs(ptr));
                return;
            }
        }
        // P4 B2 (2026-08-18): W4A4 OMMA prefill GEMM (kernels/gpu_mxfp4.cu). Env-gated
        // (--mxfp4-prefill) — ships dark until the B2 gate (prefill >= 1000 tok/s) passes.
        // Constraints mirror the kernel: K % 256 == 0, outn % 128 == 0, batch % 128 == 0
        // (the prefill caller pads batch; outn is the sharded out-dim, a multiple of 128 for
        // every projection in this family). Falls through to dequant+cuBLAS otherwise.
        let pf4_dual_layout = match (&self.mxfp4, w) {
            (Some(st), W::Nvfp4 { qweight, .. }) => {
                let p = *qweight.device_ptr() as u64;
                st.w.get(&p).map(|e| !matches!(e, crate::mxfp4::OmmaEntry::Ptr(..))).unwrap_or(true)
            }
            _ => false,
        };
        // 2026-08-17 B2-gate fix: serving prompts are arbitrary lengths — `batch % 128 == 0`
        // essentially NEVER held in serving (the [pf4] log showed only full 8192-token windows),
        // so the fast path silently fell back to dequant+cuBLAS for every real request while the
        // flag-on arm still paid --mxfp4 native-mode costs (the TP=4 A/B regression).
        // Fix: round batch UP to a 128 multiple (npad); the quantizer covers the real `batch`
        // rows, the GEMM runs at npad, and the padded C rows are garbage every consumer ignores
        // (they sit in the pool bucket's power-of-two slack). Env reads cached in statics —
        // this runs per GEMM launch; the per-launch eprintln! is gone (use --pf4-trace).
        use std::sync::OnceLock;
        static PF4_ON: OnceLock<bool> = OnceLock::new();
        static PF4_TRACE: OnceLock<bool> = OnceLock::new();
        let pf4_on = allow_pf4 && *PF4_ON.get_or_init(|| crate::opts::var(crate::opt!("mxfp4-prefill")).is_ok());
        let pf4_trace = *PF4_TRACE.get_or_init(|| crate::opts::var(crate::opt!("pf4-trace")).is_ok());
        let npad = batch.div_ceil(128) * 128;
        if pf4_on
            && inn % 256 == 0
            && npad >= crate::mxfp4::MXFP4_PREFILL_MIN_BATCH
            && out.len() >= npad * outn && x.len() >= batch * inn   // slack exists — assert, never trust
            && pf4_dual_layout
            // Sensitive-tensor allowlist (design §8.1 + stay-list: the LM head / embed / MTP
            // head keep the bf16 chain — a W4A4 LM head flips near-tie tokens and the effect
            // GROWS with prompt length; loader comment 2026-08-07, expert checklist item 1).
            // 2026-08-26: allowlisted MIXER tensors take the v2 prefill GEMM only when their
            // ptr is in the load-time mixer4 set (--pf-mixer4 — 'safe' keeps the GDN
            // in_proj on bf16); everything else allowlisted stays on the bf16 chain.
            && match (w, &self.mxfp4) {
                (W::Nvfp4 { qweight, .. }, Some(st)) => {
                    let ptr = *qweight.device_ptr() as u64;
                    !st.allow_bf16.read().unwrap().contains(&ptr)
                        || st.mixer4.read().unwrap().contains(&ptr)
                }
                _ => true,
            }
        {
            if let (W::Nvfp4 { qweight, scales, gs, .. }, Some(st)) = (w, &self.mxfp4) {
                // v2 fast path (2026-08-25, PLAN/13): OMMA-native fragment-order GEMM at
                // 213-253 TF/s (old repack path: 94.6 @2048). Reads the RESIDENT Aimg/SFAw
                // straight from the loader's repack — zero per-tensor work. Same numerics
                // class as the old W4A4 arm (e2m1 acts x e2m1 weights + per-16 ue4m3 scales,
                // gs applied once in the epilogue), so the B2-era allowlist/gates carry over.
                let qptr0 = *qweight.device_ptr() as u64;
                let omma = st.w.get(&qptr0).map(|e| e.ptrs());
                if let Some((aimg, sfaw)) = omma {
                    if pf4_trace {
                        eprintln!("[pf4-v2] outn={outn} inn={inn} batch={batch} npad={npad}");
                    }
                    let bq = *st.pf_bq.device_ptr() as u64;
                    let sb = *st.pf_sb.device_ptr() as u64;
                    let xq = *x.device_ptr() as u64;
                    let oq = *out.device_ptr() as u64;
                    let gsq = *gs.device_ptr() as u64;
                    use std::sync::atomic::{AtomicBool, Ordering};
                    static PF_AB_N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                    let check_ab = crate::opts::var(crate::opt!("mxfp4-prefill-check")).is_ok()
                        && PF_AB_N.fetch_add(1, Ordering::SeqCst) < 3;   // first 3 calls only
                    let ref_out: Option<B> = if check_ab {
                        let mut r = unsafe { self.dev.alloc::<half::bf16>(out.len()).unwrap() };
                        std::mem::swap(out, &mut r);
                        self.gemm_quant_prefill_slow(w, x, out, inn, outn, batch);
                        std::mem::swap(out, &mut r);
                        Some(r)
                    } else { None };
                    unsafe {
                        // --v2-det (2026-08-26 flake hunt): the fuzz showed IN-PROCESS
                        // nondeterminism with --mxfp4-prefill on (two identical prefill
                        // calls disagreeing on tokens; E-config 6/6 clean, v2-configs ~50%
                        // fail). This probe replays the FIRST sub-batch's quant+GEMM 30x on
                        // identical device inputs and diffs outputs bitwise: quant-loop (fresh
                        // pack each iter) splits the packer from the GEMM (frozen bq/sb).
                        use std::sync::atomic::AtomicBool as AB1;
                        static V2_DET_DONE: AB1 = AB1::new(false);
                        if crate::opts::var(crate::opt!("v2-det")).is_ok()
                            && !V2_DET_DONE.swap(true, std::sync::atomic::Ordering::SeqCst) {
                            let rows0 = SUB.min(npad);
                            let real0 = batch.min(rows0);
                            let pad80 = real0.div_ceil(8) * 8;
                            let nbytes = rows0 * outn * 2;
                            let det_out = self.dev.alloc_zeros::<u8>(nbytes).unwrap();
                            let det_q = self.dev.alloc_zeros::<u8>(nbytes).unwrap();
                            let det_bq = self.dev.alloc_zeros::<u8>(st.pf_bq.len()).unwrap();
                            let det_sb = self.dev.alloc_zeros::<u8>(st.pf_sb.len()).unwrap();
                            let tn0 = rows0 / 128;
                            let tmg0 = outn.div_ceil(128);
                            let gx0 = (((tn0 + 7) / 8) * (tmg0 * 8)) as u32;
                            let run_pair = |bq: u64, sb: u64, oq: u64| {
                                st.pf_quant2.clone().launch_on_stream(
                                    &self.stream,
                                    LaunchConfig { grid_dim: ((inn / 64) as u32, (pad80 / 8) as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                                    (xq, inn as i32, real0 as i32, pad80 as i32, bq, sb))
                                    .expect("v2det quant launch");
                                st.pf_gemm2.clone().launch_on_stream(
                                    &self.stream,
                                    LaunchConfig { grid_dim: (gx0, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 4 * 10752 },
                                    (aimg, sfaw, bq, sb, gsq, oq, outn as i32, rows0 as i32, inn as i32))
                                    .expect("v2det gemm launch");
                            };
                            let run_gemm = |bq: u64, sb: u64, oq: u64| {
                                st.pf_gemm2.clone().launch_on_stream(
                                    &self.stream,
                                    LaunchConfig { grid_dim: (gx0, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 4 * 10752 },
                                    (aimg, sfaw, bq, sb, gsq, oq, outn as i32, rows0 as i32, inn as i32))
                                    .expect("v2det gemm-only launch");
                            };
                            // phase 1: full pair, fresh pack each iter
                            run_pair(*det_bq.device_ptr() as u64, *det_sb.device_ptr() as u64, *det_out.device_ptr() as u64);
                            self.dev.synchronize().unwrap();
                            let base = self.dev.dtoh_sync_copy(&det_out).unwrap();
                            let base_bq = self.dev.dtoh_sync_copy(&det_bq).unwrap();
                            let base_sb = self.dev.dtoh_sync_copy(&det_sb).unwrap();
                            let mut bad_pair = 0usize;
                            for _ in 0..30 {
                                run_pair(*det_bq.device_ptr() as u64, *det_sb.device_ptr() as u64, *det_out.device_ptr() as u64);
                                self.dev.synchronize().unwrap();
                                let o = self.dev.dtoh_sync_copy(&det_out).unwrap();
                                let q = self.dev.dtoh_sync_copy(&det_bq).unwrap();
                                let sbh = self.dev.dtoh_sync_copy(&det_sb).unwrap();
                                if o != base || q != base_bq || sbh != base_sb { bad_pair += 1; }
                            }
                            eprintln!("[v2-det] pair loop: {bad_pair}/30 differ (bq {}B sb {}B out {}B)",
                                      base_bq.len(), base_sb.len(), base.len());
                            // phase 2: gemm only from the frozen pack
                            let mut bad_gemm = 0usize;
                            run_gemm(*det_bq.device_ptr() as u64, *det_sb.device_ptr() as u64, *det_q.device_ptr() as u64);
                            self.dev.synchronize().unwrap();
                            let g0 = self.dev.dtoh_sync_copy(&det_q).unwrap();
                            for _ in 0..30 {
                                run_gemm(*det_bq.device_ptr() as u64, *det_sb.device_ptr() as u64, *det_q.device_ptr() as u64);
                                self.dev.synchronize().unwrap();
                                let g = self.dev.dtoh_sync_copy(&det_q).unwrap();
                                if g != g0 { bad_gemm += 1; }
                            }
                            eprintln!("[v2-det] gemm-only loop: {bad_gemm}/30 differ");
                        }
                        // Sub-batch through the PREFILL_CHUNK-row pf_bq/pf_sb scratch:
                        // prefill_batch runs the WHOLE prompt as one GEMM (n up to max_seq_len),
                        // the old `batch <= PREFILL_CHUNK` gate silently skipped everything
                        // above 8K (found 2026-08-26: 4K/8K fired, 16K+ never did). Each
                        // sub-batch: quant rows [off, off+rows) then GEMM them; padded tail
                        // rows are zero-filled by the packer (exact 0) and their Out rows are
                        // pool slack every consumer ignores (same contract as npad).
                        const SUB: usize = crate::batch::PREFILL_CHUNK;
                        let nsub = npad.div_ceil(SUB);
                        for si in 0..nsub {
                            let off = si * SUB;
                            let rows = SUB.min(npad - off);
                            let real = batch.saturating_sub(off).min(rows);
                            let pad8 = real.div_ceil(8) * 8;
                            st.pf_quant2.clone().launch_on_stream(
                                &self.stream,
                                LaunchConfig { grid_dim: ((inn / 64) as u32, (pad8 / 8) as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                                (xq + (off * inn * 2) as u64, inn as i32, real as i32, pad8 as i32, bq, sb))
                                .expect("mxfp4_quant_pack_prefill_b launch");
                            let tn = rows / 128;
                            let tmg = outn.div_ceil(128);
                            let grid_x = (((tn + 7) / 8) * (tmg * 8)) as u32;
                            st.pf_gemm2.clone().launch_on_stream(
                                &self.stream,
                                LaunchConfig { grid_dim: (grid_x, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 4 * 10752 },
                                (aimg, sfaw, bq, sb, gsq, oq + (off * outn * 2) as u64, outn as i32, rows as i32, inn as i32))
                                .expect("mxfp4_gemm_prefill_v2_b launch");
                        }
                    }
                    if let Some(r) = ref_out {
                        // A/B the first 3 calls against the dequant+cuBLAS chain (row-major).
                        let got: Vec<half::bf16> = unsafe { self.dev.dtoh_sync_copy(out).unwrap() };
                        let want: Vec<half::bf16> = unsafe { self.dev.dtoh_sync_copy(&r).unwrap() };
                        let mut num = 0f64; let mut den = 0f64; let mut bad = 0usize;
                        for i in 0..batch * outn {
                            let a = got[i].to_f32() as f64; let b = want[i].to_f32() as f64;
                            num += (a - b) * (a - b); den += b * b;
                            if (a - b).abs() > 0.05 * b.abs().max(1e-3) { bad += 1; }
                        }
                        eprintln!("[pf4-v2-check] outn={outn} inn={inn} batch={batch} relL2={:.4e} bad={bad}/{}",
                                  (num / den.max(1e-300)).sqrt(), batch * outn);
                        let _ = &r;
                    }
                    return;
                }
                // old-path + reference-chain boundary: the stay-list still rules everything
                // except the v2 arm above (mixers on v2 = prefill-only, env-gated). The
                // allow_pf4 guard is what breaks the slow() recursion: slow() re-enters
                // _inner with allow_pf4=false and must fall through to the dequant path.
                if allow_pf4 {
                    if let (W::Nvfp4 { qweight, .. }, Some(st)) = (w, &self.mxfp4) {
                        if st.allow_bf16.read().unwrap().contains(&(*qweight.device_ptr() as u64)) {
                            self.gemm_quant_prefill_slow(w, x, out, inn, outn, batch);
                            return;
                        }
                    }
                }
                if npad > crate::batch::PREFILL_CHUNK {
                    // old W4A4 path has no sub-batching; oversized batches fall to the
                    // reference chain (v2 handles them when OMMA layouts are resident)
                    self.gemm_quant_prefill_slow(w, x, out, inn, outn, batch);
                    return;
                }
                // NOTE: the OMMA kernel reads the STANDARD row-major qweight/scales layout
                // (same bytes as dequant_fp4_tiled_b), NOT the OMMA repack — allowlisted
                // tensors included. The gs per-16-row scale applies (kernel epilogue).
                // qweight/scales are the TILED MMA layout (fp4_tiled_at), not row-major;
                // the W4A4 GEMM reads row-major — repack ONCE per tensor (cached in pf_rm).
                let qptr = *qweight.device_ptr() as u64;
                let _always_repack = pf4_rm_nocache();
                if _always_repack || !st.pf_rm.read().unwrap().contains_key(&qptr) {
                    let _g = st.pf_rm_lock.lock().unwrap();
                    if _always_repack || !st.pf_rm.read().unwrap().contains_key(&qptr) {
                        let wrm = self.dev.alloc_zeros::<u8>(outn * (inn / 2)).unwrap();
                        let srm = self.dev.alloc_zeros::<u8>(outn * (inn / 16)).unwrap();
                        // grid must cover BOTH phases: nibbles (M*K/2) + scales (M*K/16)
                        let total = (outn * (inn / 2) + outn * (inn / 16)) as u32;
                        unsafe {
                            st.pf_repack.clone().launch_on_stream(
                                &self.stream,
                                LaunchConfig { grid_dim: ((total + 255) / 256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                                (*wrm.device_ptr() as u64, *srm.device_ptr() as u64,
                                 qptr, *scales.device_ptr() as u64,
                                 outn as i32, inn as i32))
                                .expect("mxfp4_repack_rm_b launch");
                        }
                        self.dev.synchronize().unwrap();
                        if crate::opts::var(crate::opt!("pf-dump")).is_ok() {
                            let w_h = self.dev.dtoh_sync_copy(&wrm).unwrap();
                            let tag = format!("{qptr:x}_{outn}x{inn}");
                            std::fs::write(format!("/tmp/pfdump_{tag}_wq.bin"), &w_h).unwrap();
                            let s_h = self.dev.dtoh_sync_copy(&srm).unwrap();
                            std::fs::write(format!("/tmp/pfdump_{tag}_sq.bin"), &s_h).unwrap();
                            let x_h = self.dev.dtoh_sync_copy(x).unwrap();
                            std::fs::write(format!("/tmp/pfdump_{tag}_x.bin"), unsafe { std::slice::from_raw_parts(x_h.as_ptr() as *const u8, x_h.len()*2) }).unwrap();
                            let gs_h = self.dev.dtoh_sync_copy(gs).unwrap();
                            std::fs::write(format!("/tmp/pfdump_{tag}_gs.bin"), unsafe { std::slice::from_raw_parts(gs_h.as_ptr() as *const u8, gs_h.len()*4) }).unwrap();
                            std::fs::write(format!("/tmp/pfdump_{tag}_meta.txt"), format!("{outn} {inn} {batch}\n")).unwrap();
                            eprintln!("[pf-dump] {tag}");
                        }
                        st.pf_rm.write().unwrap().insert(qptr, (wrm, srm, outn, inn));
                    }
                }
                let (wq, sq) = {
                    let rm = st.pf_rm.read().unwrap();
                    let e = rm.get(&qptr).unwrap();
                    (*e.0.device_ptr() as u64, *e.1.device_ptr() as u64)
                };
                let bq = *st.pf_bq.device_ptr() as u64;
                let sb = *st.pf_sb.device_ptr() as u64;
                let xq = *x.device_ptr() as u64;
                let oq = *out.device_ptr() as u64;
                if pf4_trace {
                    eprintln!("[pf4] outn={outn} inn={inn} batch={batch} npad={npad}");
                }
                let ng = (batch * (inn / 16)) as u32;
                use std::sync::atomic::{AtomicBool, Ordering};
                static PF_AB_N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                let check_ab = crate::opts::var(crate::opt!("mxfp4-prefill-check")).is_ok()
                    && PF_AB_N.fetch_add(1, Ordering::SeqCst) < 3;   // first 3 calls only
                let ref_out: Option<B> = if check_ab {
                    let mut r = unsafe { self.dev.alloc::<half::bf16>(out.len()).unwrap() };
                    std::mem::swap(out, &mut r);
                    self.gemm_quant_prefill_slow(w, x, out, inn, outn, batch);
                    std::mem::swap(out, &mut r);
                    Some(r)
                } else { None };
                unsafe {
                    st.pf_quant.clone().launch_on_stream(
                        &self.stream,
                        LaunchConfig { grid_dim: ((ng + 255) / 256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                        (bq, sb, xq, batch as i32, inn as i32))
                        .expect("mxfp4_quant_prefill_b launch");
                    let grid = ((npad / 128) as u32, (outn / 128) as u32, 1);   // kernel: x=N/128 (tokens), y=M/128 (features)
                    st.pf_gemm.clone().launch_on_stream(
                        &self.stream,
                        LaunchConfig { grid_dim: grid, block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                        (wq, bq, sq, sb, d(gs), oq,
                         outn as i32, npad as i32, inn as i32))
                        .expect("mxfp4_gemm_prefill_b launch");
                }
                // expert Q4.2: dump bq/sb (post-quant) + out (post-GEMM, after full sync)
                if crate::opts::var(crate::opt!("pf-dump")).is_ok() {
                    use std::sync::atomic::{AtomicUsize as AU, Ordering as O3};
                    static PF_FD_N: AU = AU::new(0);
                    let idx = PF_FD_N.fetch_add(1, O3::SeqCst);
                    if idx < 2 {
                        self.dev.synchronize().unwrap();
                        let bq_h = self.dev.dtoh_sync_copy(&st.pf_bq).unwrap();
                        std::fs::write(format!("/tmp/pfdump_full{idx}_bq.bin"), &bq_h).unwrap();
                        let sb_h = self.dev.dtoh_sync_copy(&st.pf_sb).unwrap();
                        std::fs::write(format!("/tmp/pfdump_full{idx}_sb.bin"), &sb_h).unwrap();
                        let o_h = self.dev.dtoh_sync_copy(out).unwrap();
                        let obytes = unsafe { std::slice::from_raw_parts(o_h.as_ptr() as *const u8, o_h.len()*2) };
                        std::fs::write(format!("/tmp/pfdump_full{idx}_out.bin"), obytes).unwrap();
                        eprintln!("[pf-full] idx={idx} outn={outn} inn={inn} batch={batch} dumped bq/sb/out");
                    }
                }
                if let Some(refr) = ref_out {
                    self.dev.synchronize().unwrap();
                    let a = self.dev.dtoh_sync_copy(out).unwrap();
                    let b = self.dev.dtoh_sync_copy(&refr).unwrap();
                    let mut maxd = 0.0f32; let mut maxi = 0usize;
                    for i in 0..a.len() { let d = (a[i].to_f32() - b[i].to_f32()).abs(); if d > maxd { maxd = d; maxi = i; } }
                    let gs_h = self.dev.dtoh_sync_copy(gs).unwrap();
                    let g0 = gs_h.get((maxi / outn.max(1)) / 16).copied().unwrap_or(0.0);
                    eprintln!("[pf4-ab] outn={outn} inn={inn} batch={batch} maxdiff={maxd} at {maxi} (ref {}) fast {} gs0={g0} ratio={}", b[maxi].to_f32(), a[maxi].to_f32(), b[maxi].to_f32() / a[maxi].to_f32().max(1e-9));
                    drop(refr);
                }
                return;
            }
        }
        let n = inn * outn;
        let mut guard = self.deq_scratch.lock().unwrap();
        if guard.as_ref().map_or(true, |(_, cap)| *cap < n) {
            let buf = self.dev.alloc_zeros::<half::bf16>(n).unwrap();
            self.dev.synchronize().unwrap();
            *guard = Some((buf, n));
        }
        let (buf, _) = guard.as_mut().unwrap();
        let ptr = *buf.device_ptr() as u64;
        let grid = ((n + 255) / 256) as u32;
        match w {
            W::Nvfp4 { qweight, scales, gs, .. } => {
                // MXFP4-native mode: the prefill dequant must read the OMMA repack — in dual-layout
                // mode the tensor's qweight/scales are the STANDARD MMA repack and the OMMA repack
                // lives in st.w (reading standard bytes with the OMMA accessor permutes the weight);
                // in economy mode qweight/scales ARE the repack and st.w resolves to the same
                // buffers. Allowlisted tensors keep the standard dequant.
                let native = match &self.mxfp4 {
                    Some(st) => !st.allow_bf16.read().unwrap().contains(&(*qweight.device_ptr() as u64)),
                    None => false,
                };
                if native {
                    let st = self.mxfp4.as_ref().unwrap();
                    let p = *qweight.device_ptr() as u64;
                    let (aimg, sfa) = st.w.get(&p).unwrap_or_else(|| {
                        panic!("mxfp4: tensor @{p:x} has no OMMA repack — the prefill dequant needs it \
                                (loader bug; the repack is built for every non-allowlisted fp4 tensor)")
                    }).ptrs();
                    unsafe {
                        st.dequant.clone().launch_on_stream(
                            &self.stream,
                            LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                            (ptr, aimg, sfa, d(gs), outn as i32, inn as i32))
                            .unwrap_or_else(|e| panic!("mxfp4_dequant_tiled_b launch: {e:?}"));
                    }
                } else {
                    blaunch!(self, "dequant_fp4_tiled_b", (grid,1,1), (256,1,1), 0,
                        (ptr, *qweight.device_ptr() as u64, *scales.device_ptr() as u64,
                         d(gs), outn as i32, inn as i32));
                }
            }
            W::Fp8 { data, row_scale, .. } => {
                blaunch!(self, "dequant_fp8_tiled_b", (grid,1,1), (256,1,1), 0,
                    (ptr, *data.device_ptr() as u64, d(row_scale), outn as i32, inn as i32));
            }
            W::Fp8Blk { data, blk_scale, nb_k, .. } => {
                // F8 diagnostic (--trace-dequant): who sends a GEMM down the dequant path?
                // Backtrace on the first calls + every 500th. Answer-only instrumentation.
                static DQ_N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                if crate::opts::var(crate::opt!("trace-dequant")).is_ok() {
                    use std::sync::atomic::Ordering::Relaxed;
                    let n = DQ_N.fetch_add(1, Relaxed);
                    if n < 4 || n % 500 == 0 {
                        let bt = std::backtrace::Backtrace::force_capture();
                        let frames: Vec<String> = format!("{bt}").lines()
                            .filter(|l| l.contains("gpu") || l.contains("batch"))
                            .map(|l| l.trim().to_string()).take(8).collect();
                        eprintln!("[dq-trace #{n}] outn={outn} inn={inn} batch={batch}\n   {}",
                                  frames.join("\n   "));
                    } else if n % 100 == 0 {
                        // compact cadence probe: is the dequant path still live at this count?
                        eprintln!("[dq-alive #{n}] batch={batch} outn={outn}");
                    }
                }
                blaunch!(self, "dequant_fp8_blk_tiled_b", (grid,1,1), (256,1,1), 0,
                    (ptr, *data.device_ptr() as u64, d(blk_scale), *nb_k as i32, outn as i32, inn as i32));
            }
            W::Nvfp4Raw { .. } => unreachable!("Nvfp4Raw is MoE-experts-only"),
            W::Bf16(_) => unreachable!(),
        }
        let cfg = GemmConfig::<half::bf16> {
            transa: OP::CUBLAS_OP_T, transb: OP::CUBLAS_OP_N,
            m: outn as i32, n: batch as i32, k: inn as i32,
            alpha: half::bf16::from_f32(1.0), lda: inn as i32,
            ldb: inn as i32, beta: half::bf16::from_f32(0.0), ldc: outn as i32,
        };
        unsafe { self.blas.gemm(cfg, &*buf, x, out).expect("gemm dequant prefill"); }
    }
}
