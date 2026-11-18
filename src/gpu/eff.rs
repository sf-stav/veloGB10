//! `impl GpuModel`: TP world / rank, shard factors per tensor family and the effective (per-rank) head and dimension counts.
//! Sharding is a tensor property, not a process flag. Map: `mod.rs`.

use super::*;

impl GpuModel {
    pub fn tp_world(&self) -> i32 { self.tp_world }
    pub fn tp_rank(&self) -> i32 { self.tp_rank }

    /// Whether to shard the attn/GDN MIXERS (not just the FFN). Correct but MEASURED NET-NEGATIVE on 27B
    /// (attn-sharded 15.1→4.4 tok/s): the extra per-layer all-reduce barriers hit a comm-throughput wall
    /// in the eager single-proxy design — at 64 FFN barriers comm is free, but denser barriers (2/attn
    /// layer) stall the GPU on the pinned proxy. So it's OFF by default (the fast 1.29× FFN-only path);
    /// the code is kept, flag-gated, for the graph-captured-barrier rework that would make it pay.
    /// `[tp-shard-mixers]=1` to enable. See TP_M4_NOTES.md.
    pub(super) fn tp_shard_mixers(&self) -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        self.tp_world > 1 && *ON.get_or_init(|| crate::opts::var(crate::opt!("tp-shard-mixers")).is_ok()
            || crate::tp::tp_config().map(|c| c.shard_mixers).unwrap_or(false))
    }

    /// Per-class shard factors (thin delegates over the free predicates; the free forms are the
    /// testable single source of truth shared with the load-time shard ops).
    pub(super) fn shard_factor(&self) -> usize { self.tp_world.max(1) as usize }
    pub(super) fn ffn_shard_factor(&self) -> usize {
        ffn_shard_factor_for(&self.cfg, self.tp_world.max(1) as usize, self.tp_shard_mixers())
    }
    pub(super) fn attn_shard_factor(&self) -> usize {
        attn_shard_factor_for(&self.cfg, self.tp_world.max(1) as usize, self.tp_shard_mixers())
    }
    pub(super) fn gdn_shard_factor(&self) -> usize {
        gdn_shard_factor_for(&self.cfg, self.tp_world.max(1) as usize, self.tp_shard_mixers())
    }
    pub(super) fn vocab_shard_factor(&self) -> usize {
        vocab_shard_factor_for(&self.cfg, self.tp_world.max(1) as usize, self.tp_shard_mixers())
    }
    pub(super) fn experts_shard_factor(&self) -> usize {
        experts_shard_factor_for(&self.cfg, self.tp_world.max(1) as usize, self.tp_shard_mixers())
    }
    pub(super) fn shared_expert_shard_factor(&self) -> usize {
        shared_expert_shard_factor_for(&self.cfg, self.tp_world.max(1) as usize, self.tp_shard_mixers())
    }

    /// Effective (TP-local) head counts: each box owns `1/factor` of the heads when mixer-sharding is
    /// on AND the head counts divide `world`. Used by the sharded attn mixer + KV-cache sizing so
    /// weights, state, and kernel launches all agree.
    pub(super) fn eff_num_heads(&self) -> usize { self.cfg.num_heads / self.attn_shard_factor() }
    pub(super) fn eff_num_kv_heads(&self) -> usize { self.cfg.num_kv_heads / self.attn_shard_factor() }

    /// TP-local GDN geometry. The 48 GDN layers are 21 % of decode weight bytes — the single largest
    /// block that was still replicated on both boxes — and head-parallel is the correct split for a
    /// linear-attention layer: the delta rule is independent per value head, conv is per channel, and the
    /// recurrent state is per head, so the state halves with the heads and never crosses the wire. Only
    /// `out_proj` becomes a partial and needs the all-reduce.
    ///
    /// The value→key head map inside `delta_step_b` is `head * n_k_heads / nh`, so a CONTIGUOUS split
    /// keeps each key head's group of value heads intact (48/16 = 3 per group → 24/8 = 3 per group).
    ///
    /// DANGER, and it is why these accessors exist rather than open-coded halving: if any per-head kernel
    /// is left launched with the FULL head count, the output is still CORRECT — each rank's `out_proj`
    /// only consumes its own heads — so the token-identical gate cannot see it. It just silently does 2×
    /// the work. Every GDN dimension on a sharded path must come from here.
    fn gdn_split(&self) -> usize { self.gdn_shard_factor() }
    pub(super) fn eff_lin_v_heads(&self) -> usize { self.cfg.lin_num_v_heads / self.gdn_split() }
    pub(super) fn eff_lin_k_heads(&self) -> usize { self.cfg.lin_num_k_heads / self.gdn_split() }
    pub(super) fn eff_key_dim(&self) -> usize { self.cfg.lin_k_dim * self.eff_lin_k_heads() }
    pub(super) fn eff_value_dim(&self) -> usize { self.cfg.lin_v_dim * self.eff_lin_v_heads() }
    pub(super) fn eff_conv_dim(&self) -> usize { self.eff_key_dim() * 2 + self.eff_value_dim() }
    /// M of the (possibly sharded) fused GDN in_proj. NOTE the b/a segments keep the FULL head count:
    /// they are one row per value head and NVFP4 tiles output rows in 16s, so 48 rows will not halve at
    /// tile granularity. They are 0.6 % of in_proj, so they stay replicated and `split_gdn_b` slices
    /// them to the local head range instead.
    pub(super) fn eff_gdn_fused_m(&self) -> usize {
        self.eff_conv_dim() + self.eff_value_dim() + 2 * self.cfg.lin_num_v_heads
    }
    /// First value head owned by this rank (contiguous split).
    pub(super) fn gdn_head0(&self) -> usize {
        if self.gdn_shard_factor() > 1 { self.tp_rank.max(0) as usize * self.eff_lin_v_heads() } else { 0 }
    }
    /// The GDN paths that are NOT wired for sharding (MTP draft/verify, checkpoint probes). Reaching one
    /// while mixer-sharded would read full-width weights with local dims — wrong results, not slow ones.
    /// Fail loudly instead.
    /// `--tp-head-proof=1` — positive proof that the GDN kernels execute at LOCAL head count.
    ///
    /// This exists because the failure mode is invisible to everything else we have. A per-head kernel
    /// left launched at the full head count stays CORRECT (each rank's `out_proj` only consumes its own
    /// heads), so the token-identical gate cannot see it — it just silently does 2× the work. And it
    /// cannot be caught by TIMING either: on a ~48-SM part a 24-head and a 48-head state kernel both run
    /// in a single wave, so the wasted work does not even show up as latency.
    ///
    /// So we prove it positively, by making the wrong answer LOUD:
    ///   A. parameter red-zone — `a_log`/`dt_bias` are padded to FULL width with NaN above the local
    ///      range. Our convention is slice+local, so a global index (`head + h0`) lands in the NaN.
    ///   B. state red-zone — conv and recurrent state are allocated at FULL head shape with NaN above
    ///      the local range. A full-head-count launch writes into it; we check it is pristine afterwards.
    ///   C. launch telemetry — grid geometry of the per-head kernels asserted once against local dims.
    pub(super) fn tp_head_proof(&self) -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        self.tp_shard_mixers() && *ON.get_or_init(|| crate::opts::var(crate::opt!("tp-head-proof")).is_ok())
    }

    fn assert_gdn_unsharded(&self, who: &str) {
        assert!(self.gdn_shard_factor() <= 1,
                "{who} uses full-width GDN geometry but the mixers are sharded ([tp-shard-mixers]). \
                 This path is not TP-aware; wire it through the eff_* accessors before using it.");
    }
}
