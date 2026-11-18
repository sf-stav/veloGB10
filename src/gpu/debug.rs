//! `impl GpuModel`: debug dumps and hashes (GDN state checksums, prefill hash, KV / bf16 hashes), per-token latency
//! report and TP trace dump. Map: `mod.rs`.

use super::*;

impl GpuModel {
    /// `kv_slots` = slots that can hold a SEQUENCE (one per concurrent lane).
    /// `state_slots` = slots that can hold GDN recurrent state — always >= kv_slots, because the extra
    /// ones are pure state: the MTP GDN-rollback snapshots and the prompt checkpoints.
    ///
    /// These used to be the same number, so every snapshot slot also got a full KV cache it can never
    /// use — nothing indexes k_cache above a real lane. At 27B/depth-8 that was SEVEN unused KV slots,
    /// ~3.8 GB of VRAM. KV is position-addressable and the snapshots never roll it back (verify simply
    /// overwrites stale positions), which is exactly why they need none.
    /// R9 DECISIVE: per-layer FNV-1a checksum of the GDN recurrent (s_state) + conv state for slot 0,
    /// printed rank-tagged. A cross-rank diff shows whether the state prefill built is already
    /// divergent before decode — the layer-0-input-identical-but-output-divergent signature means the
    /// state (which never crosses the wire) is the divergence source.
    pub fn dump_gdn_state_checksums(&self, state: &BatchGpuState, rank: i32) {
        let cfg = &self.cfg;
        let world = self.tp_world.max(1) as usize;
        let sharded = self.tp_shard_mixers();
        let nlv = cfg.lin_num_v_heads;
        let nh = if sharded { nlv / world } else { nlv };
        let h0 = if sharded { (self.tp_rank.max(0) as usize) * nh } else { 0 };
        let (kd, vd) = (cfg.lin_k_dim, cfg.lin_v_dim);
        let head_elems = kd * vd;
        for (li, lt) in cfg.layer_types.iter().enumerate() {
            if !matches!(lt, LayerType::LinearAttention) { continue; }
            if li > 1 { break; }   // first two GDN layers are enough to localize
            if let Some(s) = state.s_state[li].as_ref() {
                self.sync_stream();
                if let Ok(host) = self.dev.dtoh_sync_copy(s) {
                    // Per-LOCAL-head L2 norm + checksum: head h (0..nh) is the GLOBAL head h0+h. The
                    // NORM shows which rank's state is systematically wrong (zero / scaled / wrong heads).
                    let mut line = String::new();
                    for hh in 0..nh {
                        let base = hh * head_elems;
                        let end = (base + head_elems).min(host.len());
                        let mut n2 = 0.0f64;
                        let mut h: u64 = 1469598103934665603;
                        for x in &host[base..end] {
                            n2 += (*x as f64) * (*x as f64);
                            for b in x.to_bits().to_le_bytes() { h ^= b as u64; h = h.wrapping_mul(1099511628211); }
                        }
                        line.push_str(&format!(" g{}({:.3},{:06x})", h0 + hh, n2.sqrt(), h as u32 & 0xFFFFFF));
                    }
                    eprintln!("[gdn-heads] rank {rank} layer {li} h0={h0} nh={nh}:{line}");
                }
            }
        }
    }

    /// Phase-8 [pfhash] (diagnostics): FNV-1a of the last prefill hidden column + the phys
    /// slot's GDN conv/s rows, sampled across the layer stack. Warm-vs-cold prefill equality
    /// probe: if cold and prefix-reuse prefills of the same prompt disagree anywhere, these
    /// hashes localize whether it is the trunk hidden or the GDN slot state that moved.
    pub fn pf_hash(&self, state: &BatchGpuState, phys: usize, hw: Option<(&B, usize, usize)>) -> String {
        self.sync_stream();
        let fnv = |b: &[half::bf16]| -> u64 {
            let mut x: u64 = 1469598103934665603;
            for v in b.iter() { for by in v.to_bits().to_le_bytes() { x ^= by as u64; x = x.wrapping_mul(1099511628211); } }
            x
        };
        let mut s = String::new();
        if let Some((hw, h, win)) = hw {
            let host = self.dev.dtoh_sync_copy(hw).unwrap_or_default();
            let b = win.saturating_sub(1) * h;
            if b + h <= host.len() { s.push_str(&format!(" hid={:012x}", fnv(&host[b..b + h]))); }
        }
        let (kd, vd) = (self.cfg.lin_k_dim, self.cfg.lin_v_dim);
        let s_ext = self.eff_lin_v_heads() * kd * vd;
        let conv_ext = self.eff_conv_dim() * self.cfg.conv_kernel;
        let gdn: Vec<usize> = self.cfg.layer_types.iter().enumerate()
            .filter(|(_, lt)| matches!(lt, LayerType::LinearAttention)).map(|(i, _)| i).collect();
        // Phase-8 localization: hash the FIRST 14 GDN layers densely (not sampled) — the first
        // one whose conv/s hash moves between cold and warm pins the culprit stack region.
        let mut shown = 0;
        for &li in gdn.iter().take(14) {
            if shown >= 14 { break; }
            shown += 1;
            let cs = state.conv_state[li].as_ref().and_then(|c| self.dev.dtoh_sync_copy(c).ok());
            let ss = state.s_state[li].as_ref().and_then(|c| self.dev.dtoh_sync_copy(c).ok());
            let fnv32 = |b: &[f32]| -> u64 {
                let mut x: u64 = 1469598103934665603;
                for v in b.iter() { for by in v.to_bits().to_le_bytes() { x ^= by as u64; x = x.wrapping_mul(1099511628211); } }
                x
            };
            let (mut hc, mut hs) = (0u64, 0u64);
            if let Some(v) = cs { let b = phys * conv_ext; if b + conv_ext <= v.len() { hc = fnv32(&v[b..b + conv_ext]); } }
            if let Some(v) = ss { let b = phys * s_ext; if b + s_ext <= v.len() { hs = fnv32(&v[b..b + s_ext]); } }
            s.push_str(&format!(" l{li}c={:012x} l{li}s={:012x}", hc & 0xFFFFFFFFFFFF, hs & 0xFFFFFFFFFFFF));
        }
        s
    }
}

impl GpuModel {
    // --- --debug-hash=1 (TEMPORARY instrumentation: TP bench_accept vs bench_mtp divergence hunt).
    // Host-side only (sync + dtoh + eprintln); no collectives, SPMD-safe. All sites are env-gated so
    // the deployed binary is behavior-identical with the gate off.
    fn dbg_fnv_update(mut h: u64, v: &[half::bf16]) -> u64 {
        for x in v {
            let b = x.to_bits().to_le_bytes();
            h = (h ^ b[0] as u64).wrapping_mul(0x100000001b3);
            h = (h ^ b[1] as u64).wrapping_mul(0x100000001b3);
        }
        h
    }

    /// FNV-1a over the raw bf16 bits of the first `n` elements + the first few values.
    pub fn dbg_hash_bf16(&self, tag: &str, buf: &B, n: usize) {
        if crate::opts::var(crate::opt!("debug-hash")).is_err() { return; }
        self.sync_stream();
        let v: Vec<half::bf16> = self.dev.dtoh_sync_copy(buf).unwrap();
        let n = n.min(v.len());
        let h = Self::dbg_fnv_update(0xcbf29ce484222325, &v[..n]);
        let first: Vec<f32> = v[..n.min(4)].iter().map(|x| x.to_f32()).collect();
        eprintln!("  [dbghash] {tag} n={n} fnv={h:016x} first={first:?}");
    }

    /// Hash only the WRITTEN region (positions 0..npos) of a main-model KV cache slot:
    /// layout [head][pos][hd] with pos-stride kv_stride (unwritten positions hold alloc_zeros
    /// garbage, which legitimately differs between runs and would be pure noise).
    pub fn dbg_hash_kv(&self, tag: &str, cache: &B, nkv: usize, kv_stride: usize, hd: usize, npos: usize) {
        if crate::opts::var(crate::opt!("debug-hash")).is_err() { return; }
        self.sync_stream();
        let v: Vec<half::bf16> = self.dev.dtoh_sync_copy(cache).unwrap();
        let mut h: u64 = 0xcbf29ce484222325;
        for head in 0..nkv {
            let base = head * kv_stride * hd;
            h = Self::dbg_fnv_update(h, &v[base..base + npos * hd]);
        }
        eprintln!("  [dbghash] {tag} nkv={nkv} npos={npos} fnv={h:016x}");
    }

    /// hout0 + slot-0 main KV of the first FullAttention layer, right after prefill.
    pub fn dbg_probe_prefill(&self, tag: &str, a0: u32, hout0: &B, h: usize, plen: usize,
                             state: &BatchGpuState, kv_stride: usize) {
        if crate::opts::var(crate::opt!("debug-hash")).is_err() { return; }
        eprintln!("  [dbghash] {tag}.prefill a0={a0} plen={plen} kv_stride={kv_stride}");
        self.dbg_hash_bf16(&format!("{tag}.hout0"), hout0, h * plen);
        if let Some(fa_li) = self.cfg.layer_types.iter().position(|t| matches!(t, LayerType::FullAttention)) {
            let (nkv, hd) = (self.eff_num_kv_heads(), self.cfg.head_dim);
            self.dbg_hash_kv(&format!("{tag}.maink0.L{fa_li}"), state.k_cache[fa_li].as_ref().unwrap(), nkv, kv_stride, hd, plen);
            self.dbg_hash_kv(&format!("{tag}.mainv0.L{fa_li}"), state.v_cache[fa_li].as_ref().unwrap(), nkv, kv_stride, hd, plen);
        }
    }
    // --- end --debug-hash instrumentation
}

impl GpuModel {
    /// Per-token latency percentiles with the warmup discarded. The first tokens carry one-time costs
    /// (allocator first-touch, cuBLAS workspace, page faults on the mapped comm rings) that have nothing
    /// to do with steady-state decode, and folding them into an aggregate tok/s silently understates the
    /// rate — which is exactly how a run can look like a regression when the steady state is fine.
    pub(super) fn report_token_latency(tok_ns: &[u64], rank: i32) {
        const WARMUP: usize = 16;
        if tok_ns.len() <= WARMUP + 4 { return; }
        let warm = &tok_ns[..WARMUP];
        let mut v: Vec<u64> = tok_ns[WARMUP..].to_vec();
        v.sort_unstable();
        let p = |q: f64| v[(((v.len() - 1) as f64) * q).round() as usize] as f64 / 1e6;
        let warm_ms: f64 = warm.iter().sum::<u64>() as f64 / 1e6;
        let steady: f64 = v.iter().sum::<u64>() as f64 / 1e6;
        eprintln!("[tp{rank}] per-token latency over {} tokens after {WARMUP} warmup: \
                   p50 {:.2} ms  p95 {:.2} ms  p99 {:.2} ms  max {:.2} ms  => steady-state {:.2} tok/s",
                  v.len(), p(0.50), p(0.95), p(0.99), p(1.0), 1000.0 * v.len() as f64 / steady);
        eprintln!("[tp{rank}] warmup: first {WARMUP} tokens took {warm_ms:.1} ms total \
                   (first token {:.1} ms)", warm[0] as f64 / 1e6);
    }

    /// Dump the barrier trace plus the per-layer-type cost split (needs cfg, hence a method here).
    pub fn tp_trace_dump(&self, label: &str) {
        crate::tp_bench::trace_dump(label);
        eprintln!("[tp{}] verify-graph replays: {}", self.tp_rank,
                  VERIFY_GRAPH_REPLAYS.load(std::sync::atomic::Ordering::Relaxed));
        let is_gdn: Vec<bool> = self.cfg.layer_types.iter()
            .map(|lt| matches!(lt, crate::qwen::LayerType::LinearAttention)).collect();
        crate::tp_bench::trace_layer_split(label, &is_gdn, self.tp_shard_mixers());
    }

    pub fn dev_for_state(&self) -> &Arc<CudaDevice> { &self.dev }
}
