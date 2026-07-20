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
