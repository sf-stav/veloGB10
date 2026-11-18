//! Free helpers of the GPU model: the chunk partition (`chunk_ranges`), geometry packing, device-pointer
//! helpers, the `Ws` bump workspace and its size formula. Siblings: `rows.rs` / `step.rs` use them; `mod.rs` has the map.

use super::*;

/// The row partition a chunked prefill walks: contiguous `[0, cap)` blocks, the last one short.
/// Pure, and the ONE place the chunking arithmetic lives -- the serving loop and the unit test both
/// go through it, so a drift between them fails the test instead of becoming a silent gap in the
/// served prompt.
pub fn chunk_ranges(total: usize, cap: usize) -> Vec<(usize, usize)> {
    let cap = cap.max(1);
    let mut v = Vec::new();
    let mut c0 = 0usize;
    while c0 < total {
        let mc = (total - c0).min(cap);
        v.push((c0, mc));
        c0 += mc;
    }
    v
}

/// Pack the DSA split kernels' `geom` word. Layout (bit ranges, low to high):
/// `K [0:16) | nsplits [16:24) | split_len [24:40) | S [40:64)`.
///
/// `S` is the layer's KV row count (`max_pos` for a DSA layer) and is the per-head stride
/// `g*S+t`, so it must survive 262144. It is given 24 bits FOR THAT REASON: it used to be
/// packed at bit 48 and unpacked with `& 0xFFFF`, which silently truncated it to 0 at the
/// shipped `--max-seq-len 262144`, aliasing every KV head onto head 0 on all DSA layers.
/// The kernel side is `geom >> 40 & 0xFFFFFF` in `nv_attn_dsa_split`/`_grp`; `geom_fields_match_cuda`
/// ties the two together, because a hand-written mask in CUDA drifting from a hand-written pack
/// here is exactly the bug class, and neither side is type-checked against the other.
pub fn pack_geom(k: usize, nsplits: usize, split_len: usize, s_rows: usize) -> i64 {
    ((k as i64) & 0xFFFF) | (((nsplits as i64) & 0xFF) << 16) | (((split_len as i64) & 0xFFFF) << 24) | (((s_rows as i64) & 0xFFFFFF) << 40)
}

/// The `geom` field layout as (shift, mask_width) per field, in pack order.
pub const GEOM_FIELDS: [(&str, u32, u32); 4] =
    [("K", 0, 16), ("nsplits", 16, 8), ("split_len", 24, 16), ("S", 40, 24)];

/// Fixed grid of the indexer score kernel (per row): strides over the keys <= pos.
pub const IDX_SCORE_BLOCKS: u32 = 512;

pub(super) fn b1(n: usize) -> LaunchConfig { LaunchConfig { grid_dim: (((n + 255) / 256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 } }
pub(super) fn b32(g: usize) -> LaunchConfig { LaunchConfig { grid_dim: (g as u32, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 } }
/// Device address of a slice.
pub fn dptr<T>(s: &CudaSlice<T>) -> u64 { *s.device_ptr() as u64 }
/// Synchronous device -> host copy of `n` elements at a device address (the caller synced the stream).
pub fn dtoh<T: Default + Clone + cudarc::driver::DeviceRepr>(p: u64, n: usize) -> Result<Vec<T>> {
    let mut v = vec![T::default(); n];
    unsafe { cudarc::driver::result::memcpy_dtoh_sync(&mut v, p) }.context("dtoh")?;
    Ok(v)
}

/// Workspace arena: one device slab, bump-allocated (256-B aligned) and reset per step; `mark` /
/// `release` free a layer's scratch. Nothing in a step allocates device memory, so the step is
/// graph-capturable and address-stable across replays (the same allocation sequence every time).
pub struct Ws {
    _slab: CudaSlice<u8>,
    base: u64,
    cap: usize,
    off: usize,
    pub peak: usize,
}

impl Ws {
    pub fn new(dev: &Arc<CudaDevice>, bytes: usize) -> Result<Ws> {
        let slab = dev.alloc_zeros::<u8>(bytes)?;
        dev.synchronize()?; // the zeroing rides the default stream; the model's stream is non-blocking
        let base = *slab.device_ptr() as u64;
        Ok(Ws { _slab: slab, base, cap: bytes, off: 0, peak: 0 })
    }
    pub fn reset(&mut self) { self.off = 0; }
    pub fn mark(&self) -> usize { self.off }
    pub fn release(&mut self, mark: usize) { self.off = mark; }
    pub fn alloc<T>(&mut self, n: usize) -> Result<u64> {
        let bytes = (n * std::mem::size_of::<T>() + 255) / 256 * 256;
        ensure!(self.off + bytes <= self.cap, "workspace exhausted: {} + {bytes} > {} bytes", self.off, self.cap);
        let p = self.base + self.off as u64;
        self.off += bytes;
        self.peak = self.peak.max(self.off);
        Ok(p)
    }
}

/// Workspace bytes for a step of `m` rows: the per-layer peak (MoE at ecap experts, the indexer
/// scores over max_pos) + the residual/taps + the logits, with headroom.
pub fn ws_bytes(c: &NaiveConfig, m: usize, max_pos: usize, taps_layers: usize) -> usize {
    let (h, mi, v) = (c.hidden_size, c.moe_intermediate_size, c.vocab_size);
    let ecap = c.num_experts.min(m * c.num_experts_per_tok);
    let k = c.num_experts_per_tok;
    let moe = if m > MAX_ROWS {
        // gathered rows R = m*k: xh_g, xh_u, yd_raw, yd [R][h]; yg_raw, yg, yu_raw, yu, xhd [R][mi]; maps
        m * k * (4 * h + 5 * mi) * 2 + m * k * 8 + m * c.num_experts * 4 + (1 << 20)
    } else {
        ecap * m * (3 * h + 6 * mi) * 2 + ecap * 8 * m * h * 4 + (1 << 20)
    };
    let attn = m * (c.num_heads * c.head_dim * 2 * 3 + c.num_heads * 8 * (2 + c.v_head_dim) * 4 + c.index_top_k * 4) + m * max_pos * 4;
    let dense = m * c.intermediate_size * 2 * 6;
    let chains = m * 16384 * 2 * 6 + 64 * m * 4096 * 4; // xh/yraw/y of the widest chain + split-K partials
    let head = m * v * 2 * 3 + v * 4 * 2;
    let fixed = m * h * 4 + m * h * 2 + taps_layers * m * h * 4;
    (fixed + head + moe.max(attn + dense) + chains + 4096 * 256) * 5 / 4
}

