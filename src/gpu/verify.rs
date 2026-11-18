//! `impl GpuModel`: speculative verify forward (`verify_forward*`, topology / eager variants, `verify_forward_core_topo`).
//! Column 0 of an N-wide verify must be bit-identical to a decode (batch invariance, AGENTS.md section 2.4). Map: `mod.rs`.

use super::*;

impl GpuModel {
    /// K-token causal-append forward — the MTP verify primitive. Processes `tokens` (K of them) at
    /// absolute positions pos_start..pos_start+K-1, APPENDING to the slot's existing state (KV cache
    /// + GDN recurrent state already populated for 0..pos_start-1). Does NOT zero state. The GDN
    /// prefill kernels are reused as-is: they advance conv/s_state from wherever it currently is.
    /// The full-attention kernels use pos_start so token i writes KV at pos_start+i and attends to
    /// the existing prefix [0..pos_start-1] plus the new causal block [0..i].
    /// Returns K per-position greedy predictions: pred[i] = argmax(logits at pos_start+i), i.e. the
    /// token each verify position predicts for the NEXT position (the MTP acceptance/rollback signal).
    pub fn verify_forward(&self, pool: &mut Pool, tokens: &[u32],
                          state: &mut BatchGpuState, slot: usize, kv_stride: usize,
                          pos_start: usize, ckpt_slot: Option<usize>,
                          penalty: Option<VerifyPenalty>) -> (Vec<u32>, B) {
        let _f9x_vcall = if f9_xhash_active() { f9_x_new_call() } else { usize::MAX };
        let r = self.verify_forward_topo(pool, tokens, state, slot, kv_stride, pos_start, ckpt_slot, penalty, None);
        if f9_xhash_active() && tokens.len() <= 8 {
            let h = self.cfg.hidden_size;
            let n = tokens.len();
            // SAME STAGE as forward_batch_dev's return: apply the final norm (identical launch
            // config to the decode path's non-fused arm) before hashing — vout is the PRE-norm
            // residual by contract (MTP head feed).
            let normed = pool.get_bf16(h * n);
            blaunch!(self, "rmsnorm_b", (n as u32,1,1), (1024,1,1), (4096) as u32,
                (d(&normed), d(&r.1), d(&self.final_norm), h as i32, n as i32, fbits(self.cfg.rms_eps)));
            self.sync_stream();
            let host = self.dev.dtoh_sync_copy(&normed).unwrap_or_default();
            pool.release_bf16(normed, h * n);
            let mut hs = Vec::with_capacity(n);
            for c in 0..n {
                let mut hv: u64 = 0xcbf29ce484222325;
                for i in 0..h { let b = host.get(c * h + i).map(|v| v.to_bits()).unwrap_or(0); hv ^= b as u64; hv = hv.wrapping_mul(0x100000001b3); }
                hs.push(hv);
            }
            F9XHASH.lock().unwrap().push((1u8, _f9x_vcall, usize::MAX, hs));
        }
        r
    }

    /// PLAN/25 Phase 0 (coverage capture, dump-only): `verify_forward` that hands back the raw
    /// verify LOGITS for the host-side top-k/margin readback (the `bench_accept` methodology,
    /// moved into the scheduler's dump path). Same kernels in the same order as
    /// `verify_forward`'s eager path — the E13 graph is a capture of exactly this sequence, so
    /// the values are bit-identical; only the argmax kernel's readback is replaced by the
    /// caller's host scan over the returned logits (preds are the caller's to derive). The
    /// caller owns the returned logits and must release them (`pool.release[_bf16](lg, vocab*n)`).
    /// No penalty (the coverage jobs run penalty-free, like the spec bench).
    pub fn verify_forward_keep_logits(&self, pool: &mut Pool, tokens: &[u32],
                          state: &mut BatchGpuState, slot: usize, kv_stride: usize,
                          pos_start: usize, ckpt_slot: Option<usize>) -> (VerifyLogits, B) {
        self.verify_forward_core(pool, tokens, state, slot, kv_stride, pos_start, ckpt_slot, false)
    }

    /// `verify_forward_keep_logits` with FP32 logits — the Phase-7 F9 TP probe compares
    /// column-0 logits bitwise, and bf16 would hide a 1-ulp fp32 drift.
    pub fn verify_forward_keep_logits_f32(&self, pool: &mut Pool, tokens: &[u32],
                          state: &mut BatchGpuState, slot: usize, kv_stride: usize,
                          pos_start: usize, ckpt_slot: Option<usize>) -> (VerifyLogits, B) {
        self.verify_forward_core(pool, tokens, state, slot, kv_stride, pos_start, ckpt_slot, true)
    }

    /// `verify_forward` with an optional planted tree topology (fork-then-chain in serving).
    pub fn verify_forward_topo(&self, pool: &mut Pool, tokens: &[u32],
                          state: &mut BatchGpuState, slot: usize, kv_stride: usize,
                          pos_start: usize, ckpt_slot: Option<usize>,
                          penalty: Option<VerifyPenalty>, topo: Option<&TreeTopo>) -> (Vec<u32>, B) {
        // E13: the CHAIN verify (topo == None) runs as a captured CUDA graph — the whole GPU side
        // (params expansion, embed, layers, logits, penalty, argmax) is ONE launch. The graph is
        // keyed by (width, ckpt_slot, has_penalty, state graph_epoch): all host-varying inputs
        // (tokens, pos_start, slot) live in persistent device buffers rewritten per replay; max_pc
        // is pinned to kv_stride (a launch BOUND, always correct over-estimated). The graph_epoch
        // term ties the capture to the exact state it captured (its KV/GDN pointers are baked into
        // the graph) — without it a fresh state per τ rep replays into freed memory (B8 blocker B).
        // Trees, the stochastic verify, and --no-verify-graph keep the eager path. Escape hatch on
        // purpose: a captured-verify bug would be invisible to every gate that only ever runs it —
        // the first-line repro is --no-verify-graph=1.
        // W2 (Phase 13): a captured verify graph encodes the kernel sequence from CAPTURE time —
        // it has no JSON-schema mask launch in it, so replaying it would emit UNCONSTRAINED
        // tokens while the FSM advanced underneath (observed in the first W2 boot). Whenever a
        // mask is armed, take the eager path (which launches the mask before argmax/spec_verify).
        if topo.is_none() && self.verify_mask_ptrs().is_some() {
            return self.verify_forward_topo_eager(pool, tokens, state, slot, kv_stride, pos_start,
                                                  ckpt_slot, penalty, topo);
        }
        if topo.is_none() && crate::opts::var(crate::opt!("no-verify-graph")).is_err() && self.decode_graphs_supported() {
            let n = tokens.len();
            let key = (n, ckpt_slot.unwrap_or(usize::MAX), penalty.is_some(), state.graph_epoch);
            let have = self.verify_graphs.lock().unwrap().contains_key(&key);
            if have {
                // host: params + tokens into the persistent buffers, replay, read preds, copy resid.
                let params = [pos_start as i32, slot as i32, n as i32, self.cfg.conv_kernel as i32];
                let toks_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
                unsafe {
                    cudarc::driver::result::memcpy_htod_async(*self.verify_params.device_ptr() as cudarc::driver::sys::CUdeviceptr, &params, self.stream.stream).expect("htod verify params");
                    cudarc::driver::result::memcpy_htod_async(*self.sc_tok.device_ptr() as cudarc::driver::sys::CUdeviceptr, &toks_i32[..], self.stream.stream).expect("htod verify tokens");
                }
                {
                    let graphs = self.verify_graphs.lock().unwrap();
                    graphs.get(&key).unwrap().launch();
                    VERIFY_GRAPH_REPLAYS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                self.sync_stream();
                let tids = self.dev.dtoh_sync_copy(&self.sc_tok).unwrap();
                let preds: Vec<u32> = tids.into_iter().take(n).map(|x| x as u32).collect();
                // vout contract: the PRE-final-norm residual STREAM STACK [mtp_hidden_width() x n] —
                // the hc-stream stack (rw = 4x hidden) on qwen4_exp, hidden_size elsewhere. The
                // captured backbone built it rw-wide in verify_resid (both constructors allocate
                // resid_width() * MAX_VERIFY); sizing this copy by hidden_size truncated it to
                // stream 0 — the MTP lanes then read rw-wide columns out of the allocation
                // (slab neighbors) and every REPLAYED verify (2nd+ per key) drafted from garbage.
                let mw = self.mtp_hidden_width();
                let vout = pool.get_bf16(mw * n);
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(*vout.device_ptr() as cudarc::driver::sys::CUdeviceptr,
                        *self.verify_resid.device_ptr() as cudarc::driver::sys::CUdeviceptr, mw * n * 2, self.stream.stream).expect("dtod verify resid");
                }
                return (preds, vout);
            }
            // First verify at this key: eager answer (warms the pool), then capture for next time.
            let out = self.verify_forward_topo_eager(pool, tokens, state, slot, kv_stride, pos_start, ckpt_slot, penalty, topo);
            if let Some(g) = self.capture_verify_graph(pool, state, slot, kv_stride, pos_start, n, ckpt_slot, penalty) {
                {
                    let mut graphs = self.verify_graphs.lock().unwrap();
                    graphs.insert(key, g);
                    crate::metrics::set_graph_entries(graphs.len() as u64); // v0.7.3 gauge (LR-6 visibility)
                }
            }
            return out;
        }
        self.verify_forward_topo_eager(pool, tokens, state, slot, kv_stride, pos_start, ckpt_slot, penalty, topo)
    }

    /// The eager (uncaptured) verify path — every tree verify, the first chain verify of each
    /// (width, ckpt, penalty) key, and the --no-verify-graph escape.
    fn verify_forward_topo_eager(&self, pool: &mut Pool, tokens: &[u32],
                          state: &mut BatchGpuState, slot: usize, kv_stride: usize,
                          pos_start: usize, ckpt_slot: Option<usize>,
                          penalty: Option<VerifyPenalty>, topo: Option<&TreeTopo>) -> (Vec<u32>, B) {
        let cfg = &self.cfg;
        let vocab = cfg.vocab_size;
        let _h = cfg.hidden_size;
        let n = tokens.len();

        let (logits, residual) = self.verify_forward_core_topo(pool, tokens, state, slot, kv_stride, pos_start, ckpt_slot, topo, true, false);

        // Optional repetition/presence/frequency penalty (so MTP greedy lanes keep their penalty →
        // no repetition). All n verify positions share the lane's committed-history penalty.
        // hy_v3: the greedy verify's argmax must come from unrounded FP32 logits — the decode side
        // uses them, and a bf16 round here would flip near-ties against it (silent divergence).
        match logits {
            VerifyLogits::B16(lg) => {
                if let Some(p) = penalty {
                    self.pen_apply_verify(&p, d(&lg), false, n);
                }
                if let Some((mw, mf)) = self.verify_mask_ptrs() {
                    let gy = ((mask_words_for(vocab) as u32) + 255) / 256;
                    blaunch!(self, "logit_mask_b", (n as u32, gy, 1), (256,1,1), 0,
                        (d(&lg), mw, mf, vocab as i32, n as i32));
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
                if let Some((mw, mf)) = self.verify_mask_ptrs() {
                    let gy = ((mask_words_for(vocab) as u32) + 255) / 256;
                    blaunch!(self, "logit_mask_f32_b", (n as u32, gy, 1), (256,1,1), 0,
                        (d(&lg), mw, mf, vocab as i32, n as i32));
                }
                let block = 1024u32;
                blaunch!(self, "argmax_f32_b", (n as u32,1,1), (block,1,1), (block*8),
                    (*self.sc_tok.device_ptr() as u64, d(&lg), vocab as i32, n as i32));
                pool.release(lg, vocab*n);
            }
        }
        self.sync_stream();
        let tids = self.dev.dtoh_sync_copy(&self.sc_tok).unwrap();
        // Return predictions AND the pre-final-norm [h, n] backbone hidden (MTP consumes PRE-norm).
        (tids.into_iter().take(n).map(|x| x as u32).collect(), residual)
    }
}

impl GpuModel {
    /// Shared backbone + logits for both greedy and stochastic verify paths. Returns
    /// (unpenalized logits [vocab, n], pre-final-norm residual [h, n]). `f32_logits` requests the
    /// unrounded fp32 buffer when the family contract is `enable_lm_head_fp32` (hy_v3); it is a
    /// no-op on qwen (always bf16).
    pub(super) fn verify_forward_core(&self, pool: &mut Pool, tokens: &[u32],
                           state: &mut BatchGpuState, slot: usize, kv_stride: usize,
                           pos_start: usize, ckpt_slot: Option<usize>, f32_logits: bool) -> (VerifyLogits, B) {
        self.verify_forward_core_topo(pool, tokens, state, slot, kv_stride, pos_start, ckpt_slot, None, f32_logits, false)
    }

    /// Like `verify_forward_core`, but an optional PLANTED tree topology (rope/kv/parent/path/winsrc).
    /// `None` builds the chain-identity topology (the normal path). A planted tree is how the Step-2.9
    /// byte gates drive the verify with arbitrary tree shapes before any fork drafter exists.
    pub(super) fn verify_forward_core_topo(&self, pool: &mut Pool, tokens: &[u32],
                           state: &mut BatchGpuState, slot: usize, kv_stride: usize,
                           pos_start: usize, ckpt_slot: Option<usize>, topo: Option<&TreeTopo>, f32_logits: bool,
                           capture: bool) -> (VerifyLogits, B) {
        // A verify wider than MAX_VERIFY does not fail -- it QUIETLY FALLS OFF A CLIFF. `gemm_act`
        // dispatches batch > MAX_VERIFY to the PREFILL path, which (a) dequantizes every weight into
        // scratch on every single step, and (b) finishes on cuBLAS, which is NOT batch-invariant. So a
        // too-wide verify would be catastrophically slow AND would silently stop being lossless, with
        // every correctness gate still green because the gates never run that wide.
        //
        // This matters now: tree drafting makes wide verifies routine, and the width is a function of
        // depth x branching. Assert, do not hope.
        assert!(tokens.len() <= MAX_VERIFY,
                "verify width {} exceeds MAX_VERIFY {} -- it would fall to the prefill dequant path \
                 and silently stop being batch-invariant", tokens.len(), MAX_VERIFY);
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        // TP-local attention heads when the mixers are sharded (matches the sharded qkv/o_proj and the
        // head-sharded KV cache these paths write into).
        let nh = self.eff_num_heads(); let nkv = self.eff_num_kv_heads(); let hd = cfg.head_dim; let rdim = cfg.rotary_dim;
        // TP-local GDN geometry (see eff_lin_v_heads). Same rule as the decode path: every dim on a
        // sharded path comes from an eff_* accessor, never from cfg directly.
        let lin_nh = self.eff_lin_v_heads(); let kd = cfg.lin_k_dim; let vd = cfg.lin_v_dim;
        let conv_dim = self.eff_conv_dim(); let ck = cfg.conv_kernel;
        let value_dim = self.eff_value_dim();
        let n = tokens.len();
        // Per-column GDN checkpoint band. `delta_step_prefill` / `conv1d_prefill` write one
        // checkpoint per column `t < n-1` at slot `ckpt_slot + t` (`gpu_batch.cu:1299` and `:4120`,
        // both unguarded), so the widest write lands at `ckpt_slot + n - 2`. There is NO bounds
        // check in either kernel: an undersized band silently writes OUT OF the conv/s_state
        // allocation and into whatever the allocator placed next, which is how the P14 DFlash v1
        // lane produced all-zero argmax (`!` = token 0) while every gate stayed green. Assert the
        // band here — this is the eager call that every captured verify also makes first, so a
        // sizing mistake fails loudly instead of corrupting device memory.
        if let Some(cs) = ckpt_slot {
            assert!(cs + n <= state.state_slots + 1,
                    "verify ckpt band overflow: ckpt_slot {cs} + width {n} needs {} GDN state slots, \
                     state has {} (slots: lane band {cs}, checkpoints {}..={}). The GDN scan writes \
                     ckpt_slot+t for t < n-1 with no bounds check — size n_ckpt in the scheduler's \
                     BatchGpuState construction for the WIDEST verify this source can issue \
                     (MAX_VERIFY for the 16-column block lanes).",
                    cs + n - 1, state.state_slots, cs, cs + n - 2);
        }
        // A chain: KV position and RoPE position coincide (both pos_start+j). A tree splits them —
        // siblings share a RoPE position (their tree depth) but occupy distinct KV positions. Kept as
        // two arrays now, populated identically, so the split is proven byte-identical before the tree
        // ever makes them diverge. `sc_pos` is the KV position (write offset + causal bound); `sc_rope`
        // is the RoPE angle.
        // Chain-identity defaults; a planted `topo` overrides pos/rope/parent/path/winsrc.
        // E13: capture mode (chain only) — NO htod, NO host sync: the six chain-identity arrays are
        // expanded ON DEVICE by verify_chain_params_b from `verify_params` (the caller wrote
        // [pos_start, slot, n, ck] there pre-replay); tokens are read from sc_tok; the backbone
        // hidden builds in the persistent verify_resid. Everything below is kernel-only.
        let capture = capture && topo.is_none();
        let chain_pos: Vec<i32> = (pos_start as i32..(pos_start + n) as i32).collect();
        let pos_i32: Vec<i32> = topo.map(|t| t.kv_pos.clone()).unwrap_or_else(|| chain_pos.clone());
        let rope_i32: Vec<i32> = topo.map(|t| t.rope.clone()).unwrap_or_else(|| chain_pos.clone());
        // FOREST: per-column lane slot from the topo; a single tree/chain uses the uniform `slot`.
        let slot_ids: Vec<i32> = topo.and_then(|t| t.slot.clone()).unwrap_or_else(|| vec![slot as i32; n]);
        // FOREST: per-column prefix boundary (lane committed length). None => attention uses pos[0]
        // (single lane / tree), which is byte-identical to the pre-forest rank-space attention.
        let col_pos_start: Option<Vec<i32>> = topo.and_then(|t| t.col_pos_start.clone());
        if capture {
            blaunch!(self, "verify_chain_params_b", (1u32,1,1), (32u32,1,1), 0,
                (*self.verify_params.device_ptr() as u64, *self.sc_pos.device_ptr() as u64,
                 *self.sc_rope.device_ptr() as u64, *self.sc_slot.device_ptr() as u64,
                 *self.sc_winsrc.device_ptr() as u64, *self.sc_parent.device_ptr() as u64,
                 *self.sc_path.device_ptr() as u64, MAX_VERIFY as i32));
        } else {
        unsafe {
            cudarc::driver::result::memcpy_htod_async(*self.sc_pos.device_ptr() as cudarc::driver::sys::CUdeviceptr, &pos_i32[..n], self.stream.stream).expect("htod pos");
            cudarc::driver::result::memcpy_htod_async(*self.sc_rope.device_ptr() as cudarc::driver::sys::CUdeviceptr, &rope_i32[..n], self.stream.stream).expect("htod rope");
            cudarc::driver::result::memcpy_htod_async(*self.sc_slot.device_ptr() as cudarc::driver::sys::CUdeviceptr, &slot_ids[..n], self.stream.stream).expect("htod slot");
            if let Some(cps) = col_pos_start.as_ref() {
                cudarc::driver::result::memcpy_htod_async(*self.sc_pstart.device_ptr() as cudarc::driver::sys::CUdeviceptr, &cps[..n], self.stream.stream).expect("htod pstart");
            }
        }
        }
        let col_pos_start_ptr: u64 = if col_pos_start.is_some() { *self.sc_pstart.device_ptr() as u64 } else { 0 };
        // FOREST lane segments (start, len): a new lane begins at each scan ROOT (raw parent == -1). The
        // per-lane conv-state carry (conv1d_prefill_state) must run ONCE PER LANE — committing each lane's
        // final conv window to its own slot — else a non-last lane's conv state goes stale after a full
        // accept (the s-state is handled per-lane by delta_step_prefill's commit; conv was the gap). A
        // single tree/chain is one segment (0, n): byte-identical to the old single call.
        let lane_segs: Vec<(usize, usize)> = {
            let raw_par: Vec<i32> = topo.map(|t| t.parent.clone())
                .unwrap_or_else(|| (0..n as i32).map(|t| t - 1).collect());
            let mut segs = Vec::new();
            let mut c = 0usize;
            while c < n {
                let start = c;
                c += 1;
                while c < n && raw_par[c] != -1 { c += 1; }
                segs.push((start, c - start));
            }
            segs
        };
        if !capture { self.sync_stream(); }
        // Conv window sources, CHAIN-identity: node t's width-k window is positions t..t+k-1, i.e.
        // win_src[t*k + j] = t + j - (k-1). Exactly reproduces the pre-tree stencil (a tree overwrites
        // this with the ancestor-path window). Built once per verify, reused across all GDN layers.
        // (capture: all three arrays come from verify_chain_params_b instead.)
        if !capture {
            // winsrc: chain node t's window is positions t-(k-1)..t; a tree gives the ancestor-path window.
            let ws: Vec<i32> = topo.map(|t| t.winsrc.clone()).unwrap_or_else(|| {
                let mut ws = vec![0i32; n * ck];
                for t in 0..n { for j in 0..ck { ws[t*ck + j] = (t as i32) + (j as i32) - (ck as i32 - 1); } }
                ws
            });
            unsafe {
                cudarc::driver::result::memcpy_htod_async(
                    *self.sc_winsrc.device_ptr() as cudarc::driver::sys::CUdeviceptr,
                    &ws[..n*ck], self.stream.stream).expect("htod winsrc");
            }
            // parent: chain parent[t]=t-1 (parent[0]=-1); a tree gives the DFS parent. PACKED per column:
            // low 16 bits = DFS parent, high 16 bits = the column's lane slot (delta_step_prefill reads
            // both). Single lane: slot is uniform, so the low bits are the exact old parent and the high
            // bits just re-encode `slot`, keeping the GDN scan byte-identical.
            let par_raw: Vec<i32> = topo.map(|t| t.parent.clone()).unwrap_or_else(|| (0..n as i32).map(|t| t - 1).collect());
            let par: Vec<i32> = (0..n)
                .map(|t| (((slot_ids[t] as u32 & 0xFFFF) << 16) | (par_raw[t] as u32 & 0xFFFF)) as i32)
                .collect();
            unsafe {
                cudarc::driver::result::memcpy_htod_async(
                    *self.sc_parent.device_ptr() as cudarc::driver::sys::CUdeviceptr,
                    &par[..n], self.stream.stream).expect("htod parent");
            }
            // path: chain path[b][d]=d; a tree gives the ancestor KV-slot offsets (slot-pos_start).
            let pth: Vec<u8> = topo.map(|t| t.path.clone()).unwrap_or_else(|| {
                let mut pth = vec![0u8; n * MAX_VERIFY];
                for b in 0..n { for dd in 0..MAX_VERIFY { pth[b*MAX_VERIFY + dd] = dd as u8; } }
                pth
            });
            unsafe {
                cudarc::driver::result::memcpy_htod_async(
                    *self.sc_path.device_ptr() as cudarc::driver::sys::CUdeviceptr,
                    &pth[..n*MAX_VERIFY], self.stream.stream).expect("htod path");
            }
        }
        let pos_dev_ptr = *self.sc_pos.device_ptr() as u64;
        let rope_dev_ptr = *self.sc_rope.device_ptr() as u64;
        let slot_ids_ptr = *self.sc_slot.device_ptr() as u64;
        let winsrc_ptr = *self.sc_winsrc.device_ptr() as u64;
        let parent_ptr = *self.sc_parent.device_ptr() as u64;
        let path_ptr = *self.sc_path.device_ptr() as u64;
        // Cross-chain probe capture: verify forwards are their own per-layer loop (not
        // forward_batch_dev), so they get their own sink entry here.
        if xchain_active() { xchain_new(tokens.len()); }
        // Phase-8 F9 tap: the verify record (eager only — capture mode has no live residual).
        if !capture { state_tap_call("verify", n); }
        let cos = pool.get(n * rdim);
        let sin = pool.get(n * rdim);
        blaunch!(self, "gather_rope_b", grid(n*rdim), (256,1,1), 0,
            (d(&cos), d(&sin), d(&self.cos_table), d(&self.sin_table),
             rope_dev_ptr, rdim as i32, n as i32));

        // E13 capture: embed into the persistent verify_resid from sc_tok (tokens were written
        // pre-replay); the backbone hidden builds there. Eager: a fresh pool buffer, as before.
        let q4 = cfg.is_q4();
        let hcn = cfg.hc_count.max(1);
        if q4 && !capture {
            // The PLE hash reads the column tokens from sc_tok (the capture arm's convention).
            let toks_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
            unsafe {
                cudarc::driver::result::memcpy_htod_async(*self.sc_tok.device_ptr() as cudarc::driver::sys::CUdeviceptr, &toks_i32[..n], self.stream.stream).expect("htod q4 tokens");
            }
            self.sync_stream();
        }
        let inj: Option<S> = if q4 { Some(pool.get(hcn * n)) } else { None };
        let residual = if capture { None } else { Some(self.embed_batch_resid(tokens)) };
        let res_ptr = if capture {
            let rp = *self.verify_resid.device_ptr() as u64;
            self.embed_gather_resid(pool, rp, *self.sc_tok.device_ptr() as u64, n);
            rp
        } else { d(residual.as_ref().unwrap()) };
        let normed = pool.get_bf16(h * n);
        // F5 (EXPERT_FUSION_PASSES_RESPONSE §3.5): the E14 re-land on the VERIFY path. The decode
        // loop has the FFN-side fusion flag; the verify never consulted it (cae5ffd reverted the
        // default for exactly this asymmetry). Same structure as forward_batch_dev: hoist the first
        // input norm, fuse each layer's add_residual_b + next input norm into the flavor-Q kernel
        // (BIT-EXACT to the two-kernel chain), and let the last layer's call write the final norm.
        let fuse = self.fuse_residual_norm() && !q4;
        if fuse {
            blaunch!(self, "rmsnorm_b", (n as u32,1,1), (1024,1,1), (4096) as u32,
                (d(&normed), res_ptr, d(&self.layers[0].input_ln), h as i32, n as i32, fbits(cfg.rms_eps)));
        }
        for (li, layer) in self.layers.iter().enumerate() {
            if q4 {
                if let Some(ple) = &layer.ple {
                    self.ple_forward(pool, res_ptr, ple, state, *self.sc_tok.device_ptr() as u64,
                                     slot_ids_ptr, slot, parent_ptr, true, n, ckpt_slot, false);
                }
                self.hc_pre(pool, &normed, inj.as_ref(), res_ptr, &layer.hc.as_ref().unwrap().0, n);
            } else if !fuse {
                blaunch!(self, "rmsnorm_b", (n as u32,1,1), (1024,1,1), (4096) as u32,
                    (d(&normed), res_ptr, d(&layer.input_ln), h as i32, n as i32, fbits(cfg.rms_eps)));
            }
            // Cross-chain probe capture: this layer's pre-GEMM input (col 0 == the N=1 decode).
            if xchain_active() { xchain_capture(|c| c.layer_inputs.push(xchain_dtoh_f32(self, &normed, h * n))); }
            // AR landing 2: --tp-reduce-fuse — the mixer/FFN epilogue reduces fuse into the norm
            // (K1-only + tp_reduce_resnorm_b) when they are single-barrier; the FFN site needs the
            // fused-residual config too (§4 — its unfused add_residual_b + rmsnorm_b chain has a
            // different rounding placement, so a fused run with the unfused config is not bit-equal).
            let reduce_fuse = Self::tp_reduce_fuse_on();
            let mixer = match layer.layer_type {
                LayerType::LinearAttention => {
                    let la = layer.la.as_ref().unwrap();
                    // Forest verify: pass the layer BASE pointers; the GDN prefill kernels apply the
                    // per-column lane offset from `slot_ids` (sc_slot). Single-lane: sc_slot is uniform,
                    // so `slot_ids[t]*stride` reproduces the old pre-offset `slot*stride` byte-for-byte.
                    let conv_ptr = *state.conv_state[li].as_ref().unwrap().device_ptr();
                    let s_ptr = *state.s_state[li].as_ref().unwrap().device_ptr();
                    let mid_conv_ptr: u64 = match ckpt_slot {
                        Some(ss) => *state.conv_state[li].as_ref().unwrap().device_ptr()
                            + (ss as u64 * conv_dim as u64 * ck as u64 * 4),
                        None => 0,
                    };
                    let mid_s_ptr: u64 = match ckpt_slot {
                        Some(ss) => *state.s_state[li].as_ref().unwrap().device_ptr()
                            + (ss as u64 * lin_nh as u64 * kd as u64 * vd as u64 * 4),
                        None => 0,
                    };
                    let mut qkv: Option<B> = None;
                    let mut z: Option<B> = None;
                    let mut b: Option<B> = None;
                    let mut a: Option<B> = None;
                    let mut fused_qkv: Option<B> = None;   // F0: views into the fused GDN output
                    // F0: the fused GDN in_proj output IS [token, [qkv|z|b|a]] — views under the
                    // default; --no-qkv-view keeps the packed split for the A/B byte gate.
                    let old_pipeline = crate::opts::var(crate::opt!("no-qkv-view")).is_ok();
                    let mtot = self.eff_gdn_fused_m();
                    let nh_src = self.cfg.lin_num_v_heads;
                    let h0 = self.gdn_head0();
                    match &la.in_proj {
                        GdnIn::Fused(w) => {
                            let mut fb = pool.get_bf16(mtot * n);
                            self.gemm_act(w, &normed, &mut fb, h, mtot, n);
                            if old_pipeline {
                                let qkvb = pool.get_bf16(conv_dim * n);
                                let zb = pool.get_bf16(value_dim * n);
                                let bb = pool.get_bf16(lin_nh * n);
                                let ab = pool.get_bf16(lin_nh * n);
                                blaunch!(self, "split_gdn_b", grid(mtot*n), (256,1,1), 0,
                                    (d(&qkvb), d(&zb), d(&bb), d(&ab), d(&fb),
                                     conv_dim as i32, value_dim as i32, lin_nh as i32, n as i32,
                                     nh_src as i32, h0 as i32));
                                pool.release_bf16(fb, mtot * n);
                                qkv = Some(qkvb); z = Some(zb); b = Some(bb); a = Some(ab);
                            } else {
                                fused_qkv = Some(fb);
                            }
                        }
                        GdnIn::Split { qkv: pq, z: pz, b: pb, a: pa } => {
                            // TP: qkv/z sharded (local dims); b/a replicated-full — GEMM at the
                            // FULL head count, rank band selected by pointer offset below.
                            let mut qkvb = pool.get_bf16(conv_dim * n);
                            let mut zb = pool.get_bf16(value_dim * n);
                            let mut bb = pool.get_bf16(nh_src * n);
                            let mut ab = pool.get_bf16(nh_src * n);
                            self.gemm_act(pq, &normed, &mut qkvb, h, conv_dim, n);
                            self.gemm_act(pz, &normed, &mut zb, h, value_dim, n);
                            self.gemm_act(pb, &normed, &mut bb, h, nh_src, n);
                            self.gemm_act(pa, &normed, &mut ab, h, nh_src, n);
                            qkv = Some(qkvb); z = Some(zb); b = Some(bb); a = Some(ab);
                        }
                    }
                    // conv1d is a causal depthwise STENCIL, not a recurrence: out[t] reads in[t-3..t].
                    // It is now fully parallel over (channel, position), which forced it out of place --
                    // a thread reads inputs its neighbours are writing. So: convolve into a fresh buffer,
                    // swap, and carry the final window back into `state` in a separate launch (the main
                    // kernel reads `state` from every thread; writing it there would race).
                    let mut convo = pool.get_bf16(conv_dim * n);
                    // F0: the conv reads the qkv VIEW (in_stride = mtot) under the default, else the
                    // packed qkv; the convolved rows always land in the fresh packed `convo`.
                    let (conv_in_ptr, conv_in_stride): (u64, usize) = match &fused_qkv {
                        Some(f) => (d(f), mtot),
                        None => (d(qkv.as_ref().expect("qkv")), conv_dim),
                    };
                    blaunch!(self, "conv1d_prefill", grid(conv_dim * n), (256,1,1), 0,
                        (d(&convo), conv_in_ptr, conv_ptr, d(&la.conv1d),
                         conv_dim as i32, ck as i32, n as i32, mid_conv_ptr, winsrc_ptr, slot_ids_ptr, conv_in_stride as i32));
                    // Per-lane conv-state carry: commit each lane's final window to ITS slot (see lane_segs).
                    for &(seg_start, seg_len) in &lane_segs {
                        blaunch!(self, "conv1d_prefill_state", grid(conv_dim), (256,1,1), 0,
                            (conv_ptr, conv_in_ptr, conv_dim as i32, ck as i32,
                             (seg_start + seg_len - 1) as i32, seg_len as i32, slot_ids_ptr, conv_in_stride as i32));
                    }
                    let core = pool.get_bf16(lin_nh * vd * n);
                    let (nchunk, smem) = gdn_launch(kd, vd);
                    // b/a pointers: packed buffers, or the fused views pre-offset past [qkv|z] + h0.
                    // Split path: replicated-full b/a — offset to THIS rank's head band (bf16 rows).
                    let (b_ptr, a_ptr): (u64, u64) = match &fused_qkv {
                        Some(f) => (d(f) + ((conv_dim + value_dim + h0) as u64) * 2,
                                    d(f) + ((conv_dim + value_dim + nh_src + h0) as u64) * 2),
                        None => (d(b.as_ref().expect("b")) + (self.tp_rank as usize * lin_nh * 2) as u64,
                                 d(a.as_ref().expect("a")) + (self.tp_rank as usize * lin_nh * 2) as u64),
                    };
                    let ba_stride = if fused_qkv.is_some() { mtot } else { nh_src };
                    let stride_pack = ((conv_dim as u32) << 15) | (ba_stride as u32);
                    blaunch!(self, "delta_step_prefill", ((lin_nh*nchunk) as u32,1,1), (kd as u32,1,1), smem,
                        (d(&core), d(&convo), s_ptr, b_ptr, a_ptr, stride_pack as i32, (kd as i32) | ((vd as i32) << 16),
                         d(&la.a_log), d(&la.dt_bias), (n as i32 & 0xFFFFFF) | ((self.eff_lin_k_heads() as i32 & 0xFF) << 24),
                         mid_s_ptr, parent_ptr));
                    if state_tap_active() && !capture {
                        let cv = self.dev.dtoh_sync_copy(&convo).unwrap_or_default();
                        let co = self.dev.dtoh_sync_copy(&core).unwrap_or_default();
                        state_tap_gdn(cv, co);
                        if n <= 4 {
                            state_tap_s(self.dev.dtoh_sync_copy(state.s_state[li].as_ref().unwrap()).unwrap_or_default());
                        }
                    }
                    let gnormed = pool.get_bf16(value_dim * n);
                    let (z_ptr, z_off, z_stride): (u64, usize, usize) = match &fused_qkv {
                        Some(f) => (d(f), conv_dim, mtot),
                        None => (d(z.as_ref().expect("z")), 0, lin_nh * vd),
                    };
                    let z_off_z_stride = ((z_stride as u32) << 15) | (z_off as u32);
                    blaunch!(self, self.gdn_gate_kernel(), ((n*lin_nh) as u32,1,1), (vd as u32,1,1), (vd*4) as u32,
                        (d(&gnormed), d(&core), z_ptr, d(&la.norm), vd as i32, lin_nh as i32, n as i32, fbits(cfg.rms_eps), z_off_z_stride as i32));
                    let mut out = pool.get_bf16(h * n);
                    let fused = if self.gdn_shard_factor() > 1 {
                        self.tp_reduce_site(pool, &la.out_proj, &gnormed, &mut out, value_dim, h, n, reduce_fuse)
                    } else {
                        self.gemm_act(&la.out_proj, &gnormed, &mut out, value_dim, h, n);
                        None
                    };
                    // F0: under the default qkv/z/b/a are views into `fused_qkv` — one release; packed otherwise.
                    if let Some(f) = fused_qkv { pool.release_bf16(f, mtot * n); }
                    if let Some(x) = qkv { pool.release_bf16(x, conv_dim*n); }
                    if let Some(x) = z { pool.release_bf16(x, value_dim*n); }
                    if let Some(x) = b { pool.release_bf16(x, lin_nh*n); }
                    if let Some(x) = a { pool.release_bf16(x, lin_nh*n); }
                    pool.release_bf16(convo, conv_dim * n);
                    pool.release_bf16(core, lin_nh*vd*n); pool.release_bf16(gnormed, value_dim*n);
                    match fused {
                        Some(FusedPartial::F32(p)) => ReduceOut::Fused(FusedReduce { sink: out, partial: Some(p), mode: 1 }),
                        Some(FusedPartial::F32Scratch) => ReduceOut::Fused(FusedReduce { sink: out, partial: None, mode: 1 }),
                        Some(FusedPartial::Bf16) => ReduceOut::Fused(FusedReduce { sink: out, partial: None, mode: 0 }),
                        None => ReduceOut::Full(out),
                    }
                }
                LayerType::FullAttention => {
                    let fa = layer.fa.as_ref().unwrap();
                    let kc_ptr = *state.k_cache[li].as_ref().unwrap().device_ptr();
                    let vc_ptr = *state.v_cache[li].as_ref().unwrap().device_ptr();
                    // qwen3_5 fuses a sigmoid output-gate into q_proj ([q|gate]); hy_v3's q_proj is bare.
                    let qmul = if cfg.attn_out_gate() { 2 } else { 1 };
                    let qg_dim = nh*hd*qmul;
                    let kv_dim = nkv*hd;
                    let mtot = if self.attn_shard_factor() > 1 { qg_dim + 2*kv_dim } else { AttnIn::fused_m(cfg) };
                    // F0/F4: see full_attn_batch — the fused qkv output IS [token, [q|k|v]] (views
                    // under the default); the escapes + TQ/K8v4 keep the packed split.
                    let old_pipeline = crate::opts::var(crate::opt!("no-qkv-view")).is_ok()
                        || crate::opts::var(crate::opt!("no-fused-perhead-rope")).is_ok()
                        || matches!(self.kv_mode(), KVCacheMode::Tq | KVCacheMode::K8v4 | KVCacheMode::K8v8);
                    let mut qg: Option<B> = None;
                    let mut k: Option<B> = None;
                    let mut vbuf: Option<B> = None;
                    let mut fused_qkv: Option<B> = None;
                    match &fa.qkv {
                        AttnIn::Fused(w) => {
                            let mut fb = pool.get_bf16(mtot * n);
                            self.gemm_act(w, &normed, &mut fb, h, mtot, n);
                            if old_pipeline {
                                let qgb = pool.get_bf16(qg_dim*self.pf8_pad(n));
                                let kb = pool.get_bf16(kv_dim*self.pf8_pad(n));
                                let vb = pool.get_bf16(kv_dim*self.pf8_pad(n));
                                blaunch!(self, "split_qkv_b", grid(mtot*n), (256,1,1), 0,
                                    (d(&qgb), d(&kb), d(&vb), d(&fb), qg_dim as i32, kv_dim as i32, n as i32));
                                pool.release_bf16(fb, mtot * n);
                                qg = Some(qgb); k = Some(kb); vbuf = Some(vb);
                            } else {
                                fused_qkv = Some(fb);
                            }
                        }
                        AttnIn::Split { q: qp, k: kp, v: vp } => {
                            let mut qgb = pool.get_bf16(qg_dim*self.pf8_pad(n));
                            let mut kb = pool.get_bf16(kv_dim*self.pf8_pad(n));
                            let mut vb = pool.get_bf16(kv_dim*self.pf8_pad(n));
                            self.gemm_act(qp, &normed, &mut qgb, h, qg_dim, n);
                            self.gemm_act(kp, &normed, &mut kb, h, kv_dim, n);
                            self.gemm_act(vp, &normed, &mut vb, h, kv_dim, n);
                            qg = Some(qgb); k = Some(kb); vbuf = Some(vb);
                        }
                    }
                    let (q, gate): (Option<B>, Option<B>) = if qmul == 2 {
                        // qg source: the packed qg (old pipeline / split weights) or the fused qkv view (F0).
                        let (qgs, qg_stride) = match (&qg, &fused_qkv) {
                            (Some(qgs), _) => (d(qgs), qg_dim),
                            (None, Some(f)) => (d(f), mtot),
                            _ => unreachable!("qg source"),
                        };
                        let qb = pool.get_bf16(nh*hd*n);
                        let gb = pool.get_bf16(nh*hd*n);
                        blaunch!(self, "split_qgate_b", grid(nh*hd*n), (256,1,1), 0, (d(&qb), d(&gb), qgs, nh as i32, hd as i32, n as i32, qg_stride as i32));
                        if let Some(qgs) = qg.take() { pool.release_bf16(qgs, qg_dim*n); }
                        (Some(qb), Some(gb))
                    } else if fused_qkv.is_some() {
                        (None, None)                       // q is the fused slice itself (view, pitch mtot)
                    } else {
                        (qg.take(), None)
                    };
                    let (q_ptr, q_pitch): (u64, usize) = match (&q, &fused_qkv) {
                        (Some(qb), _) => (d(qb), nh*hd),
                        (None, Some(f)) => (d(f), mtot),
                        (None, None) => unreachable!("q source"),
                    };
                    let eps = fbits(cfg.rms_eps);
                    // qwen4_exp QSA: per-column selection in rank space (path / col_pos_start aware).
                    let qsa_sel: Option<(S, S)> = self.qsa_in(state, fa, li, kv_stride, rope_dev_ptr, path_ptr, col_pos_start_ptr)
                        .map(|qs| self.qsa_select_cols(pool, qs.idx, &normed, qs.keys_ptr, kv_stride, pos_dev_ptr,
                                                       qs.logical_ptr, slot_ids_ptr, qs.path_ptr, qs.cps_ptr, &cos, &sin, n, kv_stride));
                    let qsa_sel_ptrs: Option<(u64, u64)> = qsa_sel.as_ref().map(|(a, b)| (d(a), d(b)));
                    let attn = if !old_pipeline {
                        if let Some(fb) = &fused_qkv {
                            // F0+F4: q is a view at 0 (hy3) or the packed split_qgate output (qwen); k/v are views.
                            blaunch!(self, "rmsnorm_rope_b", ((n*nh) as u32,1,1), (hd as u32,1,1), (hd*4) as u32,
                                (q_ptr, d(&fa.q_norm), d(&cos), d(&sin), nh as i32, hd as i32, rdim as i32, n as i32, eps, 0i32, q_pitch as i32));
                            let kvwrite = match self.kv_mode() {
                                KVCacheMode::Q4 => "rmsnorm_rope_kvwrite_q4_b",
                                KVCacheMode::Bf16 => "rmsnorm_rope_kvwrite_b",
                                _ => unreachable!("F4 fused kvwrite: TQ/K8v4 use the old pipeline"),
                            };
                            let stride_nkv = pack_stride_nkv(nkv, kv_stride);
                            let hd_rdim = ((hd as u32) << 16) | (rdim as u32);
                            blaunch!(self, kvwrite, ((n*nkv) as u32,1,1), (256,1,1), (hd*4) as u32,
                                (kc_ptr, vc_ptr, d(fb), d(&fa.k_norm), d(&cos), d(&sin), *self.sc_pos.device_ptr() as u64, slot_ids_ptr,
                                 stride_nkv as i32, hd_rdim as i32, qg_dim as i32, eps));
                            // F0 q-dispatch fix (Class B): qmul==2 (qwen) — dispatch the normed+roped
                            // packed q (`qb`) instead of the raw fused view; k/v stay fused views
                            // (kv_prewritten=true: the fused kvwrite already cached them).
                            let (q_ref, q_pitch_d) = match &q {
                                Some(qb) => (qb, nh * hd),
                                None => (fb, mtot),
                            };
                            {
                                let at = self.attn_dispatch(pool, q_ref, fb, fb, q_pitch_d, true, &self.sc_pos, kc_ptr, vc_ptr,
                                slot_ids_ptr, kv_stride, kv_stride, n, None, path_ptr, rope_dev_ptr, col_pos_start_ptr, self.attn_shard_factor() > 1, self.kv_mode(), topo.is_none(), qsa_sel_ptrs);
                                if crate::opts::var(crate::opt!("f9-xhash")).as_deref() == Ok("3") && li == 3 && !capture {
                                    let tag = |buf: &B, wdt: usize, nm: &str| {
                                        let host = self.dev.dtoh_sync_copy(buf).unwrap_or_default();
                                        let mut hv: u64 = 0xcbf29ce484222325;
                                        for i in 0..wdt { let b = host.get(i).map(|v| v.to_bits()).unwrap_or(0); hv ^= b as u64; hv = hv.wrapping_mul(0x100000001b3); }
                                        eprintln!("  [f9x-L3-V] {nm} 0x{hv:016x} (n={n})");
                                    };
                                    // per-column hashes so col 1 can be compared to a decode
                                    for c in 0..n {
                                        let tagc = |buf: &B, pitch: usize, nm: &str| {
                                            let host = self.dev.dtoh_sync_copy(buf).unwrap_or_default();
                                            let mut hv: u64 = 0xcbf29ce484222325;
                                            for i in 0..pitch { let b = host.get(c * pitch + i).map(|v| v.to_bits()).unwrap_or(0); hv ^= b as u64; hv = hv.wrapping_mul(0x100000001b3); }
                                            eprintln!("  [f9x-L3-V] {nm} col{c} 0x{hv:016x}");
                                        };
                                        tagc(q_ref, q_pitch_d, "q");
                                        tagc(fb, mtot, "qkv");
                                        tagc(&at, nh*hd, "attn");
                                    }
                                }
                                at
                            }
                        } else {
                            // AttnIn::Split weights — no fused buffer to view; packed fused rope + dispatch write.
                            let qb = q.as_ref().expect("q buffer");
                            let kb = k.as_ref().expect("k buffer");
                            let vbn = vbuf.as_ref().expect("v buffer");
                            blaunch!(self, "rmsnorm_rope_b", ((n*nh) as u32,1,1), (hd as u32,1,1), (hd*4) as u32, (q_ptr, d(&fa.q_norm), d(&cos), d(&sin), nh as i32, hd as i32, rdim as i32, n as i32, eps, 0i32, q_pitch as i32));
                            blaunch!(self, "rmsnorm_rope_b", ((n*nkv) as u32,1,1), (hd as u32,1,1), (hd*4) as u32, (d(kb), d(&fa.k_norm), d(&cos), d(&sin), nkv as i32, hd as i32, rdim as i32, n as i32, eps, 0i32, kv_dim as i32));
                            self.attn_dispatch(pool, qb, kb, vbn, nh*hd, false, &self.sc_pos, kc_ptr, vc_ptr,
                                slot_ids_ptr, kv_stride, kv_stride, n, None, path_ptr, rope_dev_ptr, col_pos_start_ptr, self.attn_shard_factor() > 1, self.kv_mode(), topo.is_none(), qsa_sel_ptrs)
                        }
                    } else {
                        // Old pipeline: packed q/k/v; the rope is fused unless the F4 escape is set.
                        let qb = q.as_ref().expect("q buffer");
                        let kb = k.as_ref().expect("k buffer");
                        let vbn = vbuf.as_ref().expect("v buffer");
                        if crate::opts::var(crate::opt!("no-fused-perhead-rope")).is_ok() {
                            blaunch!(self, "rmsnorm_perhead_b", ((n*nh) as u32,1,1), (hd as u32,1,1), (hd*4) as u32, (d(qb), d(qb), d(&fa.q_norm), nh as i32, hd as i32, n as i32, eps));
                            blaunch!(self, "rmsnorm_perhead_b", ((n*nkv) as u32,1,1), (hd as u32,1,1), (hd*4) as u32, (d(kb), d(kb), d(&fa.k_norm), nkv as i32, hd as i32, n as i32, eps));
                            blaunch!(self, "rope_b", grid(n*nh*(rdim/2)), (256,1,1), 0, (d(qb), d(&cos), d(&sin), nh as i32, hd as i32, rdim as i32, n as i32));
                            blaunch!(self, "rope_b", grid(n*nkv*(rdim/2)), (256,1,1), 0, (d(kb), d(&cos), d(&sin), nkv as i32, hd as i32, rdim as i32, n as i32));
                        } else {
                            blaunch!(self, "rmsnorm_rope_b", ((n*nh) as u32,1,1), (hd as u32,1,1), (hd*4) as u32, (q_ptr, d(&fa.q_norm), d(&cos), d(&sin), nh as i32, hd as i32, rdim as i32, n as i32, eps, 0i32, q_pitch as i32));
                            blaunch!(self, "rmsnorm_rope_b", ((n*nkv) as u32,1,1), (hd as u32,1,1), (hd*4) as u32, (d(kb), d(&fa.k_norm), d(&cos), d(&sin), nkv as i32, hd as i32, rdim as i32, n as i32, eps, 0i32, kv_dim as i32));
                        }
                        let at = self.attn_dispatch(pool, qb, kb, vbn, nh*hd, false, &self.sc_pos, kc_ptr, vc_ptr,
                            slot_ids_ptr, kv_stride, kv_stride, n, None, path_ptr, rope_dev_ptr, col_pos_start_ptr, self.attn_shard_factor() > 1, self.kv_mode(), topo.is_none(), qsa_sel_ptrs);
                        if crate::opts::var(crate::opt!("f9-xhash")).as_deref() == Ok("3") && li == 3 && !capture {
                            let tag = |buf: &B, wdt: usize, nm: &str| {
                                let host = self.dev.dtoh_sync_copy(buf).unwrap_or_default();
                                let mut hv: u64 = 0xcbf29ce484222325;
                                for i in 0..wdt { let b = host.get(i).map(|v| v.to_bits()).unwrap_or(0); hv ^= b as u64; hv = hv.wrapping_mul(0x100000001b3); }
                                eprintln!("  [f9x-L3-V] {nm} 0x{hv:016x} (n={n})", );
                            };
                            tag(qb, nh*hd*n, "q");
                            tag(kb, kv_dim*n, "k");
                            tag(&at, nh*hd*n, "attn");
                        }
                        at
                    };
                    if let Some((a, b)) = qsa_sel { self.qsa_release(pool, a, b, n); }
                    if let Some(g) = &gate {
                        blaunch!(self, "sigmoid_gate_b", grid(nh*hd*n), (256,1,1), 0, (d(&attn), d(g), (nh*hd*n) as i32));
                    }
                    // FIX #3 (2026-09-11, fused_qkv branch): same pf8 tp64 pad as the write_kv
                    // o_proj site above — unpadded h*n panics on pool-capacity gaps (n=408 class).
                    let mut out = pool.get_bf16(h * self.pf8_pad(n));
                    let fused = if self.attn_shard_factor() > 1 {
                        self.tp_reduce_site(pool, &fa.o_proj, &attn, &mut out, nh*hd, h, n, reduce_fuse)
                    } else {
                        self.gemm_act(&fa.o_proj, &attn, &mut out, nh*hd, h, n);
                        None
                    };
                    // F0: under the default q/k/v are views into `fused_qkv` — one release; packed otherwise.
                    if let Some(f) = fused_qkv { pool.release_bf16(f, mtot * n); }
                    if let Some(kb) = k { pool.release_bf16(kb, kv_dim*n); }
                    if let Some(vb) = vbuf { pool.release_bf16(vb, kv_dim*n); }
                    if let Some(qb) = q { pool.release_bf16(qb, nh*hd*n); }
                    pool.release_bf16(attn, nh*hd*n);
                    if let Some(g) = gate { pool.release_bf16(g, nh*hd*n); }
                    match fused {
                        Some(FusedPartial::F32(p)) => ReduceOut::Fused(FusedReduce { sink: out, partial: Some(p), mode: 1 }),
                        Some(FusedPartial::F32Scratch) => ReduceOut::Fused(FusedReduce { sink: out, partial: None, mode: 1 }),
                        Some(FusedPartial::Bf16) => ReduceOut::Fused(FusedReduce { sink: out, partial: None, mode: 0 }),
                        None => ReduceOut::Full(out),
                    }
                }
            };
            if q4 {
                let ReduceOut::Full(m) = mixer else { unreachable!("qwen4_exp: fused TP reduce is not supported") };
                self.hc_post(res_ptr, &m, inj.as_ref().unwrap(), n);
                pool.release_bf16(m, h * n);
                self.hc_pre(pool, &normed, inj.as_ref(), res_ptr, &layer.hc.as_ref().unwrap().1, n);
            } else {
                self.mixer_epilogue(pool, d(&normed), res_ptr, mixer, &layer.post_ln, h, n, false);
            }
            let mlp_out = self.ffn_batch(pool, &normed, &layer.mlp, n, self.ffn_shard_factor() > 1, reduce_fuse && fuse);
            if q4 {
                let ReduceOut::Full(m) = mlp_out else { unreachable!("qwen4_exp: fused TP reduce is not supported") };
                self.hc_post(res_ptr, &m, inj.as_ref().unwrap(), n);
                pool.release_bf16(m, h * n);
            } else {
            let next_w = if li + 1 < self.layers.len() { &self.layers[li + 1].input_ln } else { &self.final_norm };
            // F5: the FFN epilogue's flavor-Q kernel is BIT-EXACT to add_residual_b + rmsnorm_b
            // (sum_sq from the ROUNDED residual); the mixer epilogue stays flavor S.
            self.ffn_epilogue(pool, d(&normed), res_ptr, mlp_out, fuse, next_w, h, n, false);
            if crate::opts::var(crate::opt!("f9-xhash")).as_deref() == Ok("2") && n <= 8 && !capture {
                let rv = residual.as_ref().expect("eager residual");
                let host = self.dev.dtoh_sync_copy(rv).unwrap_or_default();
                let mut hs = Vec::with_capacity(n);
                for c in 0..n {
                    let mut hv: u64 = 0xcbf29ce484222325;
                    for i in 0..h { let b = host.get(c * h + i).map(|v| v.to_bits()).unwrap_or(0); hv ^= b as u64; hv = hv.wrapping_mul(0x100000001b3); }
                    hs.push(hv);
                }
                F9XHASH.lock().unwrap().push((11u8, F9X_CUR.with(|c| c.get()), li, hs));
            }
            // --layer-csum (diagnostic, prefill loop): see the block_forward twin.
            if crate::opts::var(crate::opt!("layer-csum")).is_ok() {
                let rv = residual.as_ref().expect("prefill residual");
                let v: Vec<half::bf16> = self.dev.dtoh_sync_copy(rv).unwrap();
                let (mut s, mut mx) = (0.0f64, 0.0f32);
                for x in v.iter().take(h * n) { let f = half::bf16::to_f32(*x); s += f as f64; if f.abs() > mx { mx = f.abs(); } }
                eprintln!("[csum] L{li} pf{n} sum {:.6e} maxabs {mx:.6e}", s);
            }
            }
            // E29-B3 DFlash tap: post-FFN-add residual at the tapped layers → rank-0 staging
            // scratch (the verify is where the draft loop's accepted-span features come from).
            // Capturable: the D2D becomes a memcpy node inside the chain-verify CUDA graph.
            //
            // P14 GRAPH SAFETY: the chain verify runs as a CAPTURED CUDA graph whose nodes bake
            // absolute addresses, and `wide_pos` is HOST state read at CAPTURE time — writing the
            // wide (position-addressed) surface here would freeze every replay of that graph at the
            // first step's column, silently mis-placing every later span. So the verify writes the
            // POSITION-INDEPENDENT ring staging at columns [0, n) (exactly the discipline the DF2
            // sink's `capture_cols` follows) and the lane D2Ds the ACCEPTED span to its absolute ctx
            // column afterwards (`DflashLane::feat`, the S5F3 `sync_staging_from_sink` analogue).
            // Without that host-side copy the lane consumed ctx columns no forward had written.
            // The wide branch survives only for a capture wider than the ring — never the verify
            // (n <= MAX_VERIFY == MAX_BLOCK) but kept for a caller that outgrows it.
            if let Some(tap) = &self.dflash_tap {
                if let Some(k) = tap.tap_index(li) {
                    if n <= crate::dflash::MAX_BLOCK {
                        crate::dflash::tap_capture(&self.dev, self.stream.stream, tap, res_ptr, h, n, k);
                    } else if let Some(w) = &tap.wide {
                        let p0 = tap.wide_pos.load(std::sync::atomic::Ordering::Relaxed);
                        if p0 + n <= tap.wide_stride {
                            crate::dflash::tap_capture_to(&self.dev, self.stream.stream,
                                *w.device_ptr() as u64, tap.nctx * h, p0, res_ptr, h, n, k);
                        } else {
                            crate::dflash::warn_wide_skip(p0, n, tap.wide_stride);
                        }
                    }
                }
            }
            // S4F DFlash2 tap (verify path — the round loop's committed positions): same D2D,
            // same stream; capturable identically. DEFAULT OFF = dead branch.
            // PLAN/25 Phase 1: n > BLOCK is the TREE verify (up to MAX_VERIFY nodes) — its
            // per-column taps land in the WIDE sink; the accepted path is gathered out of it.
            if n > crate::dflash2::block() && n <= MAX_VERIFY {
                if let Some(sink) = &self.df2_capture_tree {
                    if let Some(k) = crate::dflash2::tap_layers().iter().position(|&l| l == li) {
                        crate::dflash2::capture::capture_cols(&self.dev, &self.stream, sink,
                            res_ptr, h, hcn, n, k);
                    }
                }
            }
            if n <= crate::dflash2::block() {
                if let Some(sink) = &self.df2_capture {
                    if let Some(k) = crate::dflash2::tap_layers().iter().position(|&l| l == li) {
                        crate::dflash2::capture::capture_cols(&self.dev, &self.stream, sink,
                            res_ptr, h, hcn, n, k);
                    }
                }
            }
            // WI1 DSpark verify taps: the DSpark lane never runs the wide tree verify (v1 is
            // the width-8 chain only), so only the chain-width arm exists here. The verify pack
            // is committed + BLOCK drafts = BLOCK+1 = 8 wide — the gate MUST be <= BLOCK+1
            // (BLOCK=7 alone never fired: every inject pushed stale staging, acceptance 2%).
            if n <= crate::dspark::BLOCK + 1 {
                if let Some(sink) = &self.dspark_capture {
                    if let Some(k) = crate::dspark::TAP_LAYERS.iter().position(|&l| (l as i32 + Self::dspark_tap_shift()) == li as i32) {
                        crate::dflash2::capture::capture_cols(&self.dev, &self.stream, sink,
                            res_ptr, h, hcn, n, k);
                    }
                }
            }
            // Phase-8 F9 tap: post-layer residual snapshot (eager only; syncs the stream).
            if state_tap_active() && !capture {
                state_tap_layer(self.dev.dtoh_sync_copy(residual.as_ref().unwrap()).unwrap_or_default());
            }
        }
        let out = if q4 {
            let o = pool.get_bf16(h * n);
            self.hc_pre(pool, &o, None, res_ptr, self.hc_mixer.as_ref().unwrap(), n);
            pool.release_bf16(normed, h*n);
            o
        } else if fuse {
            normed          // the last iteration's fused FFN epilogue wrote rmsnorm(residual, final_norm)
        } else {
            let o = pool.get_bf16(h * n);
            blaunch!(self, "rmsnorm_b", (n as u32,1,1), (1024,1,1), (4096) as u32,
                (d(&o), res_ptr, d(&self.final_norm), h as i32, n as i32, fbits(cfg.rms_eps)));
            pool.release_bf16(normed, h*n);
            o
        };
        if let Some(i) = inj { pool.release(i, hcn * n); }
        pool.release(cos, n*rdim);
        pool.release(sin, n*rdim);

        let logits = if f32_logits && cfg.lm_head_fp32 {
            VerifyLogits::F32(self.logits_batch_f32(pool, &out, n))
        } else {
            VerifyLogits::B16(self.logits_batch(pool, &out, n))
        };
        pool.release_bf16(out, h*n);
        // capture: the backbone hidden is in verify_resid (the caller's replay arm reads it from
        // there); the returned B is a placeholder the capture caller discards.
        (logits, residual.unwrap_or_else(|| pool.get_bf16(1)))
    }
}
