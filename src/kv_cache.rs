use super::model::ModelConfig;
use half::f16;

/// KV cache structure for efficient attention computation.
/// Stores keys and values for all transformer layers and all sequence positions.
pub struct KVCache {
    pub max_seq_len: usize,
    pub current_len: usize,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub ptrs: crate::memory::KVCachePtrs,
}

impl KVCache {
    pub fn new(
        config: &ModelConfig,
        _pool: &crate::memory::UnifiedMemoryPool,
    ) -> Self {
        let ptrs = _pool.allocate_kv_cache(
            config.num_layers,
            config.num_kv_heads,
            config.head_dim,
            config.max_seq_len,
        ).expect("Failed to allocate KV cache");

        Self {
            max_seq_len: config.max_seq_len,
            current_len: 0,
            num_layers: config.num_layers,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            ptrs,
        }
    }

    /// Write K and V for a single layer and token position
    pub unsafe fn write_kv(&mut self, layer: usize, pos: usize, k: *const f16, v: *const f16) {
        let k_dst = self.ptrs.k_ptr(layer, pos);
        let v_dst = self.ptrs.v_ptr(layer, pos);
        let n = self.num_kv_heads * self.head_dim;

        std::ptr::copy_nonoverlapping(k, k_dst, n);
        std::ptr::copy_nonoverlapping(v, v_dst, n);
    }

    /// Read K for a specific layer and token position
    pub unsafe fn read_k(&self, layer: usize, pos: usize, out: *mut f16) {
        let k_src = self.ptrs.k_ptr(layer, pos);
        let n = self.num_kv_heads * self.head_dim;
        std::ptr::copy_nonoverlapping(k_src, out, n);
    }

    /// Read V for a specific layer and token position
    pub unsafe fn read_v(&self, layer: usize, pos: usize, out: *mut f16) {
        let v_src = self.ptrs.v_ptr(layer, pos);
        let n = self.num_kv_heads * self.head_dim;
        std::ptr::copy_nonoverlapping(v_src, out, n);
    }

    /// Increment sequence length
    pub fn advance(&mut self) {
        self.current_len += 1;
    }

    /// Reset the cache (for new sequence)
    pub fn reset(&mut self) {
        self.current_len = 0;
    }
}


// =================================================================================================
// W3 (Phase 13): kv-mode x lane-width compatibility, evaluated at ARG PARSE time.
// =================================================================================================
//
// The owner's TP4 repro (`--kv-cache k8v8 ... --tp 4 --max-batch 4 --spec-source dflash2`) used to
// load the whole model and then die inside the attention dispatch: k8v8 reads its int8 K/V rows
// ONLY on the `_e` attention lane (`gqa_attn_verify_e_k8v8`), and that lane's contract is
// `batch <= MAX_VERIFY (16)` with `chain_ok` (gpu.rs:9261). A multi-lane serve whose packed
// verify/step width leaves that envelope hits
//     assert!(use_e, "k8v8 requires the _e attention lane (batch<=8, gqa<=48, no --no-attn-e)")
// — a runtime panic AFTER a full model load, with no hint of the remedy.
//
// The check below is the host-side predicate, evaluated BEFORE the model load and BEFORE any node
// contact, so the same mistake fails in milliseconds with the conflict and the remedy named.
// `--kv-cache bf16` (or `--max-batch 1`) are the two supported escapes.

/// The verdict for one (kv-cache, max-batch, tp) configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvLaneVerdict {
    Ok,
    /// The configuration is invalid; the string is the user-facing remedy text.
    Reject(String),
}

/// W3 (Phase 13): does this kv-cache mode support this many serving lanes?
///
/// k8v8 is a SINGLE-LANE cache by construction (one kernel, the `_e` attention lane, whose
/// contract is `batch <= MAX_VERIFY`): multi-lane serving packs more columns than the kernel's
/// row-group grid covers, and the dispatch panics instead of degrading (deliberately — a silent
/// bf16 read of an int8 buffer is the mojibake hazard). Every other mode is unconstrained.
pub fn kv_lane_check(kv_cache: Option<&str>, max_batch: usize, tp: usize) -> KvLaneVerdict {
    let k8v8 = matches!(kv_cache, Some("k8v8")) || crate::opts::var(crate::opt!("kv-k8v8")).ok().as_deref() == Some("1");
    if k8v8 && max_batch > 1 {
        return KvLaneVerdict::Reject(format!(
            "k8v8 KV does not support multi-lane serving (max-batch {max_batch} > 1, tp {tp}): the \
             int8 K/V rows are read only by the single-lane `_e` attention kernel. Use \
             --kv-cache bf16 (or k8v4) or --max-batch 1."
        ));
    }
    KvLaneVerdict::Ok
}

/// The predicate as a TABLE (W3): the unit test walks exactly these cells and asserts the verdict,
/// so the rule can never drift from its documentation.
#[cfg(test)]
fn verdict_of(kv: &str, mb: usize, tp: usize) -> bool {
    matches!(kv_lane_check(Some(kv), mb, tp), KvLaneVerdict::Ok)
}

#[cfg(test)]
mod kv_lane_tests {
    use super::*;

    /// W3's table: k8v8 × {mb1, mb4} and bf16 × {mb1, mb4}, each at tp {1, 2, 4}.
    #[test]
    fn kv_lane_table() {
        for tp in [1usize, 2, 4] {
            // bf16 is unconstrained at every lane width and topology.
            assert!(verdict_of("bf16", 1, tp), "bf16 mb1 tp{tp}");
            assert!(verdict_of("bf16", 4, tp), "bf16 mb4 tp{tp}");
            // k8v8 is single-lane: mb1 passes, mb4 is rejected BEFORE the load.
            assert!(verdict_of("k8v8", 1, tp), "k8v8 mb1 tp{tp}");
            assert!(!verdict_of("k8v8", 4, tp), "k8v8 mb4 tp{tp} must fail fast");
            // The other modes keep their standing behaviour (not part of the W3 conflict).
            assert!(verdict_of("q4", 4, tp), "q4 mb4 tp{tp}");
            assert!(verdict_of("tq", 4, tp), "tq mb4 tp{tp}");
        }
    }

    /// The rejection text names the conflict AND both remedies (the F6 lesson: an error a user
    /// cannot act on is a silent failure with extra steps).
    #[test]
    fn kv_lane_reject_message_is_actionable() {
        match kv_lane_check(Some("k8v8"), 4, 4) {
            KvLaneVerdict::Reject(m) => {
                assert!(m.contains("k8v8"), "{m}");
                assert!(m.contains("max-batch 4 > 1"), "{m}");
                assert!(m.contains("--kv-cache bf16"), "{m}");
                assert!(m.contains("--max-batch 1"), "{m}");
            }
            KvLaneVerdict::Ok => panic!("k8v8 x mb4 must be rejected"),
        }
    }

    /// No kv-cache mode is selected by default (None) — nothing to check, never a false positive.
    #[test]
    fn kv_lane_default_is_ok() {
        assert_eq!(kv_lane_check(None, 8, 4), KvLaneVerdict::Ok);
    }
}
