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
// History: k8v8 used to be a SINGLE-LANE cache. It reads its int8 K/V rows on the `_e` tensor-core attention
// lane (`gqa_attn_verify_e_k8v8`), which addresses ONE slot for every column, and no other reader existed, so a
// multi-lane serve (`--kv-cache k8v8 --max-batch 4`) loaded the whole model and then died in the dispatch
// (`k8v8 requires the _e attention lane`). The check below rejected it BEFORE the load.
//
// Since the community report (GitHub, "k8v8 with --max-batch > 1") the dispatch has an int8 split-K reader pair
// (`gqa_attn_splitk_k8v8` / `_gq`: per-column slot and position, the k8v4 pair's structure) for every attention
// call the `_e` lane is not host-gated to — plain batched decode of several lanes, FOREST / tree verify — so
// k8v8 serves any lane count. The predicate stays (one place that answers "does this kv mode support this lane
// width?" at parse time, and the table test pins the answer) and today it accepts every cell.

/// The verdict for one (kv-cache, max-batch, tp) configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvLaneVerdict {
    Ok,
    /// The configuration is invalid; the string is the user-facing remedy text.
    Reject(String),
}

/// W3 (Phase 13): does this kv-cache mode support this many serving lanes? Every NVFP4 mode does (k8v8 since
/// the int8 split-K readers landed); kept as the single parse-time gate for any future mode that does not.
pub fn kv_lane_check(_kv_cache: Option<&str>, _max_batch: usize, _tp: usize) -> KvLaneVerdict {
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

    /// Every NVFP4 KV mode accepts every lane width at every topology (k8v8 included since its multi-lane readers).
    #[test]
    fn kv_lane_table() {
        for tp in [1usize, 2, 4] {
            for kv in ["bf16", "q4", "tq", "tq3", "k8v4", "k8v8"] {
                for mb in [1usize, 2, 4, 8, 16] {
                    assert!(verdict_of(kv, mb, tp), "{kv} mb{mb} tp{tp}");
                }
            }
        }
    }

    /// No kv-cache mode is selected by default (None) — nothing to check, never a false positive.
    #[test]
    fn kv_lane_default_is_ok() {
        assert_eq!(kv_lane_check(None, 8, 4), KvLaneVerdict::Ok);
    }
}
