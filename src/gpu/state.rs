//! `impl GpuModel`: KV-cache layout accessors, batch / slot state allocation and zeroing, KV mirrors, GDN slot copy and KV
//! compaction, hidden-column copies and memsets. GDN recurrent state lives in shared memory for the token loop (AGENTS.md section 2). Map: `mod.rs`.

use super::*;

impl GpuModel {
    /// Packed V-row size in bytes for the 4-bit (and k8v4-V) cache: hd/16 blocks × 9 B (8 B codes
    /// + 1 B scale + pad). Matches the kernels' KVQ_ROW_BYTES. The k8v4 mode reuses this q4 V
    /// layout byte-for-byte, so the assert accepts both.
    pub fn kvq_row_bytes(&self, hd: usize) -> usize {
        assert!((self.kv_quant || self.kv_k8v4) && hd % 16 == 0, "kvq layout needs hd % 16 == 0");
        // The packed readers (gqa_attn_splitk_q4 / _k8v4) dequantize DPL = hd/32 dims per lane in
        // u16 groups of 4 nibbles: hd < 128 makes that loop empty and K/V read as ZEROS silently.
        assert!(hd % 128 == 0 && hd <= 512, "the q4/k8v4 KV readers need head_dim % 128 == 0 and <= 512 (got {hd}); use --kv-cache bf16");
        (hd / 16) * 12
    }

    /// k8v4 K-row size in bytes: hd/16 blocks × 20 B (16 B int8 codes + 2 B fp16 scale + 2 B pad).
    /// Matches the kernels' KV8_ROW_BYTES. Meaningful only under [kv-k8v4].
    pub fn kv_k_row_bytes(&self, hd: usize) -> usize {
        assert!((self.kv_k8v4 || self.kv_k8v8) && hd % 16 == 0,
                "int8-block layout needs [kv-k8v4]/[kv-k8v8] and hd % 16 == 0");
        (hd / 16) * 20
    }

    /// TurboQuant row stride: 52 B (b=2 K, [kv-tq]=1) or 68 B (b=3 K, [kv-tq]=3) for both
    /// K and V rows; the d=128 layout — see TQ_ROW_BYTES / TQ_ROW_BYTES_B3.
    pub fn tq_row_bytes(&self) -> usize {
        assert!(self.kv_tq, "TQ row size is only meaningful under [kv-tq]");
        assert_eq!(self.cfg.head_dim, 128, "the TurboQuant layout is d=128 (Hy3) only");
        if self.kv_tq_b3 { TQ_ROW_BYTES_B3 } else { TQ_ROW_BYTES }
    }

    /// The KV cache format this GpuModel uses.
    pub fn kv_mode(&self) -> KVCacheMode {
        if self.kv_tq { KVCacheMode::Tq }
        else if self.kv_k8v8 { KVCacheMode::K8v8 }
        else if self.kv_k8v4 { KVCacheMode::K8v4 }
        else if self.kv_quant { KVCacheMode::Q4 }
        else { KVCacheMode::Bf16 }
    }

    /// bf16 elements per K-cache position (one KV head): hd normally, or the packed byte count / 2
    /// under the packed caches. Under k8v4 the K cache diverges from the V cache for the first
    /// time (K rows 20 B/16, V rows 12 B/16) — size sites must use the per-channel halves.
    pub fn kv_k_elems_per_pos(&self) -> usize {
        if self.kv_tq { self.tq_row_bytes() / 2 }
        else if self.kv_k8v4 || self.kv_k8v8 { self.kv_k_row_bytes(self.cfg.head_dim) / 2 }
        else if self.kv_quant { self.kvq_row_bytes(self.cfg.head_dim) / 2 }
        else { self.cfg.head_dim }
    }

    /// bf16 elements per V-cache position (one KV head). Under k8v4 the V cache is the q4 layout;
    /// under k8v8 both channels are the int8-block layout; otherwise K and V share one size.
    pub fn kv_v_elems_per_pos(&self) -> usize {
        if self.kv_tq { self.tq_row_bytes() / 2 }
        else if self.kv_k8v8 { self.kv_k_row_bytes(self.cfg.head_dim) / 2 }
        else if self.kv_k8v4 || self.kv_quant { self.kvq_row_bytes(self.cfg.head_dim) / 2 }
        else { self.cfg.head_dim }
    }

    /// Whether the 4-bit KV cache is active ([kv-quant]=1 at load).
    pub fn kv_quant(&self) -> bool { self.kv_quant }
}

impl GpuModel {
    pub fn new_batch_state(&self, kv_slots: usize, state_slots: usize, max_seq_len: usize) -> BatchGpuState {        let cfg = &self.cfg;
        let stride = max_seq_len;
        let mut k_cache = vec![];
        let mut v_cache = vec![];
        let mut conv_state = vec![];
        let mut s_state = vec![];
        let mut kv_mirror = vec![];
        for lt in &cfg.layer_types {
            match lt {
                LayerType::FullAttention => {
                    // TP=2: each box caches only its half of the KV heads (2 of 4).
                    // 4-bit KV (kv_quant): the cache holds packed per-16 blocks (9 B/16 elems
                    // = 3.56x fewer bytes), addressed as bf16-element pairs.
                    // TurboQuant (kv_tq): 52-B rows (3.2 bpw, the E4 context enabler).
                    // k8v4 (kv_k8v4): K rows diverge from V for the first time — K is 20 B/16
                    // (int8 + fp16 scale), V reuses the q4 12 B/16 — so the two caches are sized
                    // per channel.
                    let k_per_pos = self.kv_k_elems_per_pos();
                    let v_per_pos = self.kv_v_elems_per_pos();
                    let k_bytes = kv_slots * self.eff_num_kv_heads() * stride * k_per_pos;
                    let v_bytes = kv_slots * self.eff_num_kv_heads() * stride * v_per_pos;
                    k_cache.push(Some(self.dev.alloc_zeros::<half::bf16>(k_bytes).unwrap()));
                    v_cache.push(Some(self.dev.alloc_zeros::<half::bf16>(v_bytes).unwrap()));
                    conv_state.push(None); s_state.push(None);
                    kv_mirror.push((0..kv_slots).map(|_| None).collect());   // E2 Fix 2: allocated per slot, lazily
                }
                LayerType::LinearAttention => {
                    // TP=2 mixer-sharded: each box holds only its half of the GDN heads, so the conv and
                    // recurrent state halve too and never cross the wire.
                    let conv_dim = self.eff_conv_dim();
                    let conv_n = state_slots * conv_dim * cfg.conv_kernel;
                    let s_n = state_slots * self.eff_lin_v_heads() * cfg.lin_k_dim * cfg.lin_v_dim;
                    if self.tp_head_proof() {
                        // Proof B: over-allocate to the FULL head shape and poison everything above the
                        // local range with NaN. The kernels index by local head/channel, so correct code
                        // never touches the red zone; a full-head-count launch writes straight into it.
                        assert_eq!(state_slots, 1, "head-proof red zone assumes a single state slot");
                        // A distinctive BIT PATTERN, not plain NaN: the injected-fault control writes
                        // NaN into these heads (their a_log is NaN-padded by proof A), and NaN-over-NaN
                        // would be undetectable. Comparing bits catches any write at all.
                        let (cf, sf) = (conv_n * 2, s_n * 2);
                        let sent = f32::from_bits(TP_REDZONE_SENTINEL);
                        let mut cv = vec![0f32; cf]; for x in &mut cv[conv_n..] { *x = sent; }
                        let mut sv = vec![0f32; sf]; for x in &mut sv[s_n..] { *x = sent; }
                        conv_state.push(Some(self.dev.htod_sync_copy(&cv).unwrap()));
                        s_state.push(Some(self.dev.htod_sync_copy(&sv).unwrap()));
                    } else {
                    conv_state.push(Some(self.dev.alloc_zeros::<f32>(conv_n).unwrap()));
                    s_state.push(Some(self.dev.alloc_zeros::<f32>(s_n).unwrap()));
                    }
                    k_cache.push(None); v_cache.push(None);
                    kv_mirror.push(vec![]);
                }
            }
        }
        // qwen4_exp PLE state: conv window + token ring per slot, plus one scratch slot (main-slot
        // commits are computed into it and copied back — the state kernel reads the old state).
        let (ple_conv_state, ple_ring) = if cfg.is_q4() && cfg.ple_layer.is_some() {
            let rw = cfg.resid_width();
            let l = cfg.ple_conv_state_len();
            let nslot = state_slots + 1;
            let ring: Vec<i32> = vec![cfg.eos_token_id as i32; nslot * (cfg.ple_ngram_size - 1)];
            let mut cs = self.dev.alloc_zeros::<f32>(nslot * l * rw).unwrap();
            self.dev.memset_zeros(&mut cs).unwrap();   // alloc_zeros does NOT zero
            (Some(cs), Some(self.dev.htod_sync_copy(&ring).unwrap()))
        } else { (None, None) };
        // qwen4_exp QSA: one raw-key cache per full-attention layer when the indexer is live at this
        // context length (kv_stride > budget+ratio-1; below it every block is selected — dense).
        let qsa_keys: Vec<Option<B>> = cfg.layer_types.iter().map(|lt| {
            if matches!(lt, LayerType::FullAttention) && self.qsa_enabled(stride) {
                Some(self.dev.alloc_zeros::<half::bf16>(kv_slots * stride * cfg.indexer_head_dim).unwrap())
            } else { None }
        }).collect();
        BatchGpuState {
            k_cache, v_cache, conv_state, s_state, kv_mirror,
            graph_epoch: GRAPH_EPOCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            vision_embeds: None,
            vision_spans: Vec::new(),
            state_slots,
            ple_conv_state, ple_ring, ple_scratch_slot: state_slots,
            qsa_keys,
        }
    }

    /// Zero all GDN recurrent + conv state and KV cache. Needed after graph capture
    /// (capture advances stateful GDN state during warmup+capture passes).
    pub fn zero_state(&self, state: &mut BatchGpuState) {
        for opt in state.conv_state.iter_mut() {
            if let Some(s) = opt { self.dev.memset_zeros(s).unwrap(); }
        }
        for opt in state.s_state.iter_mut() {
            if let Some(s) = opt { self.dev.memset_zeros(s).unwrap(); }
        }
        for opt in state.k_cache.iter_mut() {
            if let Some(s) = opt { self.dev.memset_zeros(s).unwrap(); }
        }
        for opt in state.v_cache.iter_mut() {
            if let Some(s) = opt { self.dev.memset_zeros(s).unwrap(); }
        }
        // E2 Fix 2: zeroed caches invalidate every dequant mirror (defensive — today zero_state
        // runs at startup before any prefill, so there is never a mirror yet).
        for ms in state.kv_mirror.iter_mut() { for m in ms.iter_mut() { *m = None; } }
        if let Some(cs) = state.ple_conv_state.as_mut() { self.dev.memset_zeros(cs).unwrap(); }
        if let Some(r) = state.ple_ring.as_mut() {
            let n = r.len();
            self.dev.htod_sync_copy_into(&vec![self.cfg.eos_token_id as i32; n], r).unwrap();
        }
    }

    /// Zero the GDN recurrent + conv state for a specific slot (when reusing a slot for a new request).
    /// Uses the compute stream (ordered with subsequent prefill kernels).
    pub fn zero_slot_state(&self, state: &mut BatchGpuState, slot: usize, kv_stride: usize) {
        let cfg = &self.cfg;
        // must match new_batch_state's (possibly TP-local) allocation, or this zeroes the wrong extent
        let conv_dim = self.eff_conv_dim();
        let ck = cfg.conv_kernel;
        let lin_nh = self.eff_lin_v_heads();
        let kd = cfg.lin_k_dim;
        let vd = cfg.lin_v_dim;
        let nkv = self.eff_num_kv_heads();
        let hd = cfg.head_dim;
        let stream = self.stream.stream;
        self.ple_zero_slot(state, slot);
        for (li, lt) in cfg.layer_types.iter().enumerate() {
            match lt {
                LayerType::LinearAttention => {
                    let conv_bytes = conv_dim * ck * 4;
                    let s_bytes = lin_nh * kd * vd * 4;
                    unsafe {
                        if let Some(s) = state.conv_state[li].as_ref() {
                            let ptr = *s.device_ptr() as u64 + (slot * conv_bytes) as u64;
                            cudarc::driver::sys::cuMemsetD8Async(
                                ptr as cudarc::driver::sys::CUdeviceptr, 0, conv_bytes, stream);
                        }
                        if let Some(s) = state.s_state[li].as_ref() {
                            let ptr = *s.device_ptr() as u64 + (slot * s_bytes) as u64;
                            cudarc::driver::sys::cuMemsetD8Async(
                                ptr as cudarc::driver::sys::CUdeviceptr, 0, s_bytes, stream);
                        }
                    }
                }
                LayerType::FullAttention => {
                    // E2 Fix 2: a slot reuse invalidates the q4 dequant mirror (contents are
                    // request-scoped) — free it; the next long prompt re-allocates lazily.
                    // The mirror is allocated 0..kv_slots (lane + prefix-cache slots), NOT the
                    // full state-slot band: checkpoint/audit slots beyond it have no mirror and
                    // nothing to invalidate (pp.rs:140 documents the same coupling). Bound-guard
                    // so a state slot >= kv_slots (S-B16's shadow slot) does not panic here.
                    if slot < state.kv_mirror[li].len() {
                        state.kv_mirror[li][slot] = None;
                    }
                    // KV does NOT need zeroing on a cold admit: every attention consumer is
                    // read-bounded at pos/pc — gqa_attn_splitk caps keys at min(split, pc),
                    // gqa_attn_prefill uses pc = pos_start + t + 1, attn_prefill_tiled is
                    // kmax-bounded, write_kv only writes — and prefill writes 0..plen before
                    // anything reads. This memset was ~64 KB x kv_stride per layer of dead work on
                    // EVERY cold admit (~2.4 ms at 8K, ~9.4 ms at 32K of pure TTFT on 9B; ~2x on
                    // 27B), and `new_batch_state` uses alloc_zeros so this was the only zeroing.
                    // --zero-kv=1 restores it (A/B escape hatch). The GDN conv/recurrent
                    // state above DOES need zeroing — the recurrence reads it every step.
                    if zero_kv_enabled() {
                        let k_bytes = if self.kv_tq { nkv * kv_stride * self.tq_row_bytes() }
                                      else if self.kv_k8v4 { nkv * kv_stride * self.kv_k_row_bytes(hd) }
                                      else if self.kv_quant { nkv * kv_stride * self.kvq_row_bytes(hd) }
                                      else { nkv * kv_stride * hd * 2 };
                        let v_bytes = if self.kv_tq { nkv * kv_stride * self.tq_row_bytes() }
                                      else if self.kv_k8v4 || self.kv_quant { nkv * kv_stride * self.kvq_row_bytes(hd) }
                                      else { nkv * kv_stride * hd * 2 };
                        unsafe {
                            if let Some(s) = state.k_cache[li].as_ref() {
                                let ptr = *s.device_ptr() as u64 + (slot * k_bytes) as u64;
                                cudarc::driver::sys::cuMemsetD8Async(
                                    ptr as cudarc::driver::sys::CUdeviceptr, 0, k_bytes, stream);
                            }
                            if let Some(s) = state.v_cache[li].as_ref() {
                                let ptr = *s.device_ptr() as u64 + (slot * v_bytes) as u64;
                                cudarc::driver::sys::cuMemsetD8Async(
                                    ptr as cudarc::driver::sys::CUdeviceptr, 0, v_bytes, stream);
                            }
                        }
                    }
                }
            }
        }
    }

    /// S-B16 (lossless auditor): zero the FULL KV extent of one slot — the unconditional mirror
    /// of zero_slot_state's `zero_kv_enabled()` branch. The auditor's shadow slot has no prefill:
    /// its first decode attends over rows [0, pos) that nothing wrote, so they must start zeroed.
    /// One memset pair per request (~ms of dead work); never on the serving path.
    pub fn zero_slot_kv(&self, state: &mut BatchGpuState, slot: usize, kv_stride: usize) {
        let nkv = self.eff_num_kv_heads();
        let hd = self.cfg.head_dim;
        let stream = self.stream.stream;
        for (li, lt) in self.cfg.layer_types.iter().enumerate() {
            if !matches!(lt, LayerType::FullAttention) { continue; }
            let k_bytes = if self.kv_tq { nkv * kv_stride * self.tq_row_bytes() }
                          else if self.kv_k8v4 { nkv * kv_stride * self.kv_k_row_bytes(hd) }
                          else if self.kv_quant { nkv * kv_stride * self.kvq_row_bytes(hd) }
                          else { nkv * kv_stride * hd * 2 };
            let v_bytes = if self.kv_tq { nkv * kv_stride * self.tq_row_bytes() }
                          else if self.kv_k8v4 || self.kv_quant { nkv * kv_stride * self.kvq_row_bytes(hd) }
                          else { nkv * kv_stride * hd * 2 };
            unsafe {
                if let Some(s) = state.k_cache[li].as_ref() {
                    let ptr = *s.device_ptr() as u64 + (slot * k_bytes) as u64;
                    cudarc::driver::sys::cuMemsetD8Async(
                        ptr as cudarc::driver::sys::CUdeviceptr, 0, k_bytes, stream);
                }
                if let Some(s) = state.v_cache[li].as_ref() {
                    let ptr = *s.device_ptr() as u64 + (slot * v_bytes) as u64;
                    cudarc::driver::sys::cuMemsetD8Async(
                        ptr as cudarc::driver::sys::CUdeviceptr, 0, v_bytes, stream);
                }
            }
        }
    }

    /// E2 Fix 2 — position budget of the per-(layer, slot) q4-KV dequant mirror: min(kv_stride,
    /// --kv-mirror-budget) positions (default 32768; the A/B escape). Purely LOCAL memory
    /// layout — the mirror is a rank-private cache of dequants, it never crosses the wire and no
    /// barrier count depends on it, so SPMD is unaffected by the value (both ranks share it anyway).
    pub(super) fn kv_mirror_budget(&self, kv_stride: usize) -> usize {
        static B: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        kv_stride.min(*B.get_or_init(|| crate::opts::var(crate::opt!("kv-mirror-budget")).ok()
            .and_then(|v| v.parse().ok()).filter(|n| *n >= 1).unwrap_or(32768)))
    }

    /// E2 Fix 2: clamp the slot's q4-KV dequant mirror watermarks to `pos` — mirror rows past `pos`
    /// are stale (prefix-cache reuse, KV compaction), rows [0, pos) stay valid. The next prefill
    /// window re-dequants only the tail past the clamp. No-op when the slot has no mirrors.
    pub fn clamp_kv_mirrors(&self, state: &mut BatchGpuState, slot: usize, pos: usize) {
        for (li, lt) in self.cfg.layer_types.iter().enumerate() {
            if !matches!(lt, LayerType::FullAttention) { continue; }
            if let Some(m) = state.kv_mirror[li][slot].as_mut() { m.up_to = m.up_to.min(pos); }
        }
    }
}

impl GpuModel {
    /// Device-copy all GDN-layer recurrent state (conv_state + s_state) for one slot from `src` to
    /// `dst` within the BatchGpuState buffers. Used for MTP GDN-rollback snapshots (the snapshot
    /// lives in a dedicated slot). KV cache is NOT copied: verify overwrites stale KV positions, so
    /// no KV rollback is needed.
    pub fn copy_gdn_slot(&self, state: &BatchGpuState, src: usize, dst: usize) {
        // Slot bounds: this is a raw pointer/offset copy into the slot band, so an out-of-range slot
        // silently reads/writes a neighbouring allocation. The block lanes roll back to
        // `mtp_snapshot_slot + nacc`, whose upper bound is set by the verify WIDTH (P14: block 16 ⇒
        // nacc ≤ 14 ⇒ slot ≤ 15), so the band must be sized for the widest source too.
        assert!(src < state.state_slots && dst < state.state_slots,
                "copy_gdn_slot out of range: src {src} dst {dst}, state has {} GDN state slots",
                state.state_slots);
        // Route through the COMPUTE stream (async) so the snapshot/restore is stream-ordered with
        // the verify/reverify GDN writes (all on self.stream) — no cross-stream race with the
        // device's default stream.
        //
        // E0: ONE fused kernel (`gdn_rollback_b`) replaces the per-layer host loop below — the
        // loop issued 2*n_gdn cudaMemcpyDtoDAsync driver calls per rollback (96 on the 27B: 48
        // layers x {conv_state 40,960 B, s_state 786,432 B rank-local}), and nsys showed ~25% of
        // the rollback span idle in issue gaps between them. The kernel is a PURE BYTE COPY over
        // the same rank-local strides — bit-identical to the dtod path by construction. The
        // pointer table is built once per BatchGpuState (graph_epoch keyed — same lifetime
        // discipline as `verify_graphs`). --gdn-rollback-d2d=1 restores the legacy per-layer
        // dtod path (the A/B escape and first-line repro).
        let cfg = &self.cfg;
        // TP-aware: the state slots were allocated at the LOCAL head count, so the per-slot stride must
        // match or this copies the wrong extent. Under sharding it also halves — a free win for MTP,
        // since rollback is per-rank-local (each rank owns whole heads, nothing crosses the wire).
        let lin_nh = self.eff_lin_v_heads(); let kd = cfg.lin_k_dim; let vd = cfg.lin_v_dim;
        let conv_dim = self.eff_conv_dim(); let ck = cfg.conv_kernel;
        let cb = (conv_dim * ck) as u64 * 4;      // bytes per slot: conv_state
        let sb = (lin_nh * kd * vd) as u64 * 4;   // bytes per slot: s_state
        let stream = self.stream.stream;
        self.ple_copy_slot(state, src, dst);
        if crate::opts::var(crate::opt!("gdn-rollback-d2d")).is_ok() {
            for (li, lt) in cfg.layer_types.iter().enumerate() {
                if matches!(lt, LayerType::LinearAttention) {
                    unsafe {
                        let cs = *state.conv_state[li].as_ref().unwrap().device_ptr();
                        let ss = *state.s_state[li].as_ref().unwrap().device_ptr();
                        cudarc::driver::result::memcpy_dtod_async(cs + dst as u64 * cb, cs + src as u64 * cb, cb as usize, stream).unwrap();
                        cudarc::driver::result::memcpy_dtod_async(ss + dst as u64 * sb, ss + src as u64 * sb, sb as usize, stream).unwrap();
                    }
                }
            }
            return;
        }
        {
            let mut tables = self.gdn_slot_tables.lock().unwrap();
            if !tables.contains_key(&state.graph_epoch) {
                let mut conv_ptrs: Vec<u64> = Vec::new();
                let mut s_ptrs: Vec<u64> = Vec::new();
                for (li, lt) in cfg.layer_types.iter().enumerate() {
                    if matches!(lt, LayerType::LinearAttention) {
                        conv_ptrs.push(*state.conv_state[li].as_ref().unwrap().device_ptr() as u64);
                        s_ptrs.push(*state.s_state[li].as_ref().unwrap().device_ptr() as u64);
                    }
                }
                let n_gdn = conv_ptrs.len() as i32;
                conv_ptrs.extend_from_slice(&s_ptrs);
                let tbl = self.dev.htod_sync_copy(&conv_ptrs)
                    .expect("gdn rollback pointer table upload");
                tables.insert(state.graph_epoch, (tbl, n_gdn));
            }
            let (tbl, n_gdn) = tables.get(&state.graph_epoch).unwrap();
            // grid (n_gdn, 4): each layer's conv and s_state regions are independently partitioned
            // over 4 y-blocks (192 blocks at n_gdn=48 — one block per layer cannot hold the
            // bandwidth roofline at ~17% occupancy).
            blaunch!(self, "gdn_rollback_b", (*n_gdn as u32, 4, 1), (256, 1, 1), 0,
                (*tbl.device_ptr() as u64, *n_gdn, cb, sb, src as i32, dst as i32));
        }
    }

    /// Compact an accepted tree path's KV into contiguous positions: for k in 0..len, move the K/V at
    /// KV position `pos_start + src_pos[k]` down to `pos_start + k`, for every full-attention layer.
    /// Two-pass gather/scatter through pooled scratch (no in-place aliasing). `src_pos` is host-side
    /// (the accepted DFS indices). A no-op when the accepted path is already contiguous (src_pos[k]==k).
    pub fn compact_kv(&self, pool: &mut Pool, state: &mut BatchGpuState, slot: usize, pos_start: usize,
                      src_pos: &[i32], kv_stride: usize) {
        let len = src_pos.len();
        if len == 0 || src_pos.iter().enumerate().all(|(k, &s)| s as usize == k) { return; }  // already contiguous
        // E2 Fix 2: the compact MOVES KV content within [pos_start, pos_start+len) — the slot's
        // dequant mirrors are stale past pos_start (rows [0, pos_start) are untouched and stay
        // valid). Clamp the watermarks; the next prefill window re-dequants only the tail.
        self.clamp_kv_mirrors(state, slot, pos_start);
        // eff, not full: the KV cache is head-sharded per rank under TP (per-slot stride is
        // eff_nkv*kv_stride*hd). Reading cfg.num_kv_heads here doubled every slot's base and every
        // head extent — the same "sharding is a property of the tensor" bug class as the tiled
        // prefill fix. bench_tree_accept's host check below matches.
        let (nkv, hd) = (self.eff_num_kv_heads(), self.cfg.head_dim);
        // upload src_pos to sc_pos scratch (reused; sized MAX_VERIFY)
        unsafe {
            cudarc::driver::result::memcpy_htod_async(
                *self.sc_pos.device_ptr() as cudarc::driver::sys::CUdeviceptr,
                &src_pos[..len], self.stream.stream).expect("htod compact src");
        }
        let sp_ptr = *self.sc_pos.device_ptr() as u64;
        // qwen4_exp QSA: the raw-key caches move with the KV (same columns, same rule).
        if self.qsa_enabled(kv_stride) {
            let hdx = self.cfg.indexer_head_dim;
            let scratch = pool.get_bf16(len * hdx);
            for keys in state.qsa_keys.iter().flatten() {
                for dir in 0..2i32 {
                    blaunch!(self, "qsa_compact_b", grid(len * hdx), (256,1,1), 0,
                        (d(keys), d(&scratch), sp_ptr, len as i32, pos_start as i32, slot as i32, kv_stride as i32, hdx as i32, dir));
                }
            }
            pool.release_bf16(scratch, len * hdx);
        }
        if self.kv_quant || self.kv_tq || self.kv_k8v4 || self.kv_k8v8 {
            // Per-channel row sizes: k8v4 is the first mode where K and V diverge (K 20 B/16,
            // V 12 B/16) — the scratch buffers are sized per channel and the k8v4 kernel's grid
            // spans K+V bytes (the shared-rb q4/TQ kernels copy both caches per element over
            // len*nkv*rb).
            let (k_rb, v_rb) = if self.kv_tq { (self.tq_row_bytes(), self.tq_row_bytes()) }
                               else if self.kv_k8v4 { (self.kv_k_row_bytes(hd), self.kvq_row_bytes(hd)) }
                               else if self.kv_k8v8 { (self.kv_k_row_bytes(hd), self.kv_k_row_bytes(hd)) } // both channels 20 B/16
                               else { (self.kvq_row_bytes(hd), self.kvq_row_bytes(hd)) };
            let (total, k_scratch, v_scratch) = if self.kv_k8v4 || self.kv_k8v8 {
                (len * nkv * (k_rb + v_rb), len * nkv * k_rb / 2, len * nkv * v_rb / 2)
            } else {
                (len * nkv * k_rb, len * nkv * k_rb / 2, len * nkv * v_rb / 2)
            };
            let ks = pool.get_bf16(k_scratch);
            let vs = pool.get_bf16(v_scratch);
            let total = total as u32;
            for (li, lt) in self.cfg.layer_types.iter().enumerate() {
                if !matches!(lt, LayerType::FullAttention) { continue; }
                let kc = *state.k_cache[li].as_ref().unwrap().device_ptr();
                let vc = *state.v_cache[li].as_ref().unwrap().device_ptr();
                for dir in 0..2i32 {   // 0 = gather to scratch, 1 = scatter to contiguous
                    if self.kv_tq {
                        blaunch!(self, "compact_kv_tq", grid(total as usize), (256,1,1), 0,
                            (kc, vc, d(&ks), d(&vs), sp_ptr, len as i32, pos_start as i32,
                             slot as i32, nkv as i32, kv_stride as i32, dir));
                    } else if self.kv_k8v4 {
                        blaunch!(self, "compact_kv_k8v4", grid(total as usize), (256,1,1), 0,
                            (kc, vc, d(&ks), d(&vs), sp_ptr, len as i32, pos_start as i32,
                             slot as i32, nkv as i32, kv_stride as i32, hd as i32, dir));
                    } else if self.kv_k8v8 {
                        blaunch!(self, "compact_kv_k8v8", grid(total as usize), (256,1,1), 0,
                            (kc, vc, d(&ks), d(&vs), sp_ptr, len as i32, pos_start as i32,
                             slot as i32, nkv as i32, kv_stride as i32, hd as i32, dir));
                    } else {
                        blaunch!(self, "compact_kv_q4", grid(total as usize), (256,1,1), 0,
                            (kc, vc, d(&ks), d(&vs), sp_ptr, len as i32, pos_start as i32,
                             slot as i32, nkv as i32, kv_stride as i32, hd as i32, dir));
                    }
                }
            }
            pool.release_bf16(ks, k_scratch);
            pool.release_bf16(vs, v_scratch);
            return;
        }
        let ks = pool.get_bf16(len * nkv * hd);
        let vs = pool.get_bf16(len * nkv * hd);
        let total = (len * nkv * hd) as u32;
        for (li, lt) in self.cfg.layer_types.iter().enumerate() {
            if !matches!(lt, LayerType::FullAttention) { continue; }
            let kc = *state.k_cache[li].as_ref().unwrap().device_ptr();
            let vc = *state.v_cache[li].as_ref().unwrap().device_ptr();
            for dir in 0..2i32 {   // 0 = gather to scratch, 1 = scatter to contiguous
                blaunch!(self, "compact_kv_b", grid(total as usize), (256,1,1), 0,
                    (kc, vc, d(&ks), d(&vs), sp_ptr, len as i32, pos_start as i32,
                     slot as i32, nkv as i32, kv_stride as i32, hd as i32, dir));
            }
        }
        pool.release_bf16(ks, len * nkv * hd);
        pool.release_bf16(vs, len * nkv * hd);
    }

    /// Copy one hidden column (`col`, of `h` bf16 elems) out of an `[h, n]` column-major buffer
    /// `src` into the device pointer `dst`. Routed through the COMPUTE stream so it is stream-ordered
    /// with the kernels that produced/consume these hiddens (no cross-stream race with stream 0).
    pub fn copy_hidden_col(&self, dst_ptr: u64, src: &B, col: usize) {
        let h = self.mtp_hidden_width();
        let stream = self.stream.stream;
        unsafe {
            let src_ptr = *src.device_ptr() as u64 + (col as u64) * (h as u64) * 2;
            cudarc::driver::result::memcpy_dtod_async(dst_ptr, src_ptr, h * 2, stream).unwrap();
        }
    }

    /// Copy `ncols` CONTIGUOUS hidden columns starting at `col` out of an `[h, n]` column-major
    /// buffer into `dst` — ONE dtod for what used to be `ncols` separate `copy_hidden_col` calls
    /// (the re-prime path assembles accepted-prefix hiddens, whose verify columns are contiguous).
    pub fn copy_hidden_cols(&self, dst_ptr: u64, src: &B, col: usize, ncols: usize) {
        let h = self.mtp_hidden_width();
        let stream = self.stream.stream;
        unsafe {
            let src_ptr = *src.device_ptr() as u64 + (col as u64) * (h as u64) * 2;
            cudarc::driver::result::memcpy_dtod_async(dst_ptr, src_ptr, ncols * h * 2, stream).unwrap();
        }
    }

    /// Memset (zero) `bytes` bytes at `ptr` on the COMPUTE stream — ordered with subsequent compute
    /// kernels. Use this (NOT the device-default-stream memset) for any buffer the compute stream
    /// reads, e.g. per-lane MTP KV that must be zeroed before priming.
    pub fn memset_compute_stream(&self, ptr: u64, bytes: usize) {
        unsafe {
            cudarc::driver::sys::cuMemsetD8Async(
                ptr as cudarc::driver::sys::CUdeviceptr, 0, bytes, self.stream.stream);
        }
    }

    /// 32-bit variant (the E16 padding guard fills perm_tok with -1 = 0xFFFF_FFFF per dispatch).
    /// Same stream-ordering discipline as memset_compute_stream.
    pub(super) fn memset_compute_stream_u32(&self, ptr: u64, val: u32, count: usize) {
        unsafe {
            cudarc::driver::sys::cuMemsetD32Async(
                ptr as cudarc::driver::sys::CUdeviceptr, val, count, self.stream.stream);
        }
    }
}

impl GpuModel {
    pub fn kv_stride(&self) -> usize { self.cfg.max_position_embeddings }
}
