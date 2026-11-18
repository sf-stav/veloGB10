//! `impl GpuModel`: embedding gather, vision splice and the `prefill_batch` entry point (the chunk loop is `prefill.rs`).
//! Map: `mod.rs`.

use super::*;

impl GpuModel {
    /// Gather `batch` embedding rows into `out_ptr`, dequantizing if the table is quantized.
    /// The embedding is a weight like any other — and on a tied-embedding model it IS the LM head, so
    /// leaving it bf16 would keep ~23% of the model's bytes unquantized.
    ///
    /// When `mma` is set the table is stored permuted into mma-fragment order, so the gather has to
    /// invert that permutation per element. This is the ONLY place a weight is read element-wise on
    /// the hot path; it costs a few index ops on one row per token.
    pub(super) fn embed_gather(&self, out_ptr: u64, toks_ptr: u64, h: usize, batch: usize) {
        match &self.embed {
            W::Bf16(e) => {
                blaunch!(self, "embed_gather_b", grid(h*batch), (256,1,1), 0,
                    (out_ptr, *e.device_ptr() as u64, toks_ptr, h as i32, batch as i32));
            }
            W::Nvfp4 { qweight, scales, gs, .. } => {
                // MXFP4-native mode: gather with the OMMA element accessor FROM THE OMMA REPACK
                // (st.w — in dual-layout mode the embed's qweight/scales are the STANDARD MMA
                // repack and reading them with the OMMA accessor permutes every embedding, which
                // is garbage from token 0; in economy mode qweight IS the repack and st.w resolves
                // to the same buffers). Allowlisted tensors (and mode off) keep the standard gather.
                let native = match &self.mxfp4 {
                    Some(st) => !st.allow_bf16.read().unwrap().contains(&(*qweight.device_ptr() as u64)),
                    None => false,
                };
                if native {
                    let st = self.mxfp4.as_ref().unwrap();
                    let p = *qweight.device_ptr() as u64;
                    let (aimg, sfa) = st.w.get(&p).unwrap_or_else(|| {
                        panic!("mxfp4: embed @{p:x} has no OMMA repack (loader bug)")
                    }).ptrs();
                    unsafe {
                        // One element per thread: grid covers h*batch THREADS (h*batch/256 blocks).
                        // The kernel is `int i = blockIdx*blockDim + tid` with `i < h*batch` — a
                        // grid of h*batch BLOCKS makes i overflow i32 at h*batch*256 > 2^31
                        // (batch >= 1639 at h=5120), the wild-read crash of 2026-08-07 (TP=2 and
                        // single-node, long prompts; found via compute-sanitizer).
                        st.embed.clone().launch_on_stream(&self.stream,
                            LaunchConfig { grid_dim: (((h*batch + 255) / 256) as u32,1,1), block_dim: (256,1,1), shared_mem_bytes: 0 },
                            (out_ptr, aimg, sfa, d(gs), toks_ptr, h as i32, batch as i32))
                            .unwrap_or_else(|e| panic!("mxfp4 embed gather: {e:?}"));
                    }
                } else {
                    blaunch!(self, "embed_gather_fp4_tiled_b", grid(h*batch), (256,1,1), 0,
                        (out_ptr, *qweight.device_ptr() as u64, *scales.device_ptr() as u64,
                         d(gs), toks_ptr, h as i32, batch as i32));
                }
            }
            W::Fp8 { data, row_scale, .. } => {
                blaunch!(self, "embed_gather_fp8_tiled_b", grid(h*batch), (256,1,1), 0,
                    (out_ptr, *data.device_ptr() as u64, d(row_scale),
                     toks_ptr, h as i32, batch as i32));
            }
            W::Fp8Blk { .. } => unreachable!(
                "embed as block-128 FP8 is not a supported artifact layout (Qwen keeps embed bf16)"),
            W::Nvfp4Raw { .. } => unreachable!("Nvfp4Raw is MoE-experts-only"),
        }
    }

    pub fn embed_batch(&self, tokens: &[u32]) -> B {
        let h = self.cfg.hidden_size;
        let b = tokens.len();
        let toks_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        // htod_sync_copy is already host-blocking, and the blocking compute stream (I1) orders the
        // gather after the NULL-stream copy — the old dev.synchronize() here was a full pipeline
        // drain per verify, and the trailing sync_stream had no host consumer (callers stay on the
        // same stream). alloc_zeros is fine without a memset: the gather overwrites all h*b elems.
        let toks_dev = self.dev.htod_sync_copy(&toks_i32).expect("htod tokens");
        let hidden = self.dev.alloc_zeros::<half::bf16>(h * b).unwrap();
        self.embed_gather(*hidden.device_ptr() as u64, *toks_dev.device_ptr() as u64, h, b);
        hidden
    }
}

impl GpuModel {
    /// Batched prefill: process all N prompt tokens in one pass.
    /// GDN layers use sequential prefill kernels; full-attn uses batched causal attention.
    /// Returns the first generated token.
    /// Prefill `prompt` into `slot`, starting at absolute position `pos_start`.
    ///
    /// `pos_start > 0` is PREFIX REUSE: the slot's KV cache already holds positions [0, pos_start), and
    /// its GDN recurrent + conv1d states are already the states AFTER token pos_start-1. The caller has
    /// verified the cached tokens are an exact prefix of this prompt, and must NOT have zeroed the slot.
    ///
    /// The recurrent state is what makes this different from a pure-attention engine. KV is
    /// position-addressable — any prefix of it is meaningful on its own. The GDN state is NOT: it exists
    /// only at the single point in the sequence where it was left. So a prefix can only be reused from a
    /// slot that was carried to exactly that token and no further, which is why the cache is per-slot and
    /// keyed on the whole previous sequence rather than a shared radix tree.
    /// V3: overwrite the image span rows of the embedded hidden (`residual`, [window_len, hidden])
    /// with the merged image embeddings (`embeds`, bf16, [total_emb, hidden]). Columns match (vision
    /// out_hidden 5120 == text hidden 5120), so each image row is a contiguous D2D copy. Spans are
    /// ABSOLUTE positions in the expanded prompt; this writes the rows intersecting [pos_start, +len).
    pub(super) fn splice_vision(
        &self,
        embeds: &B,
        spans: &[crate::vision_encoder::ImageSpan],
        residual: &B,
        pos_start: usize,
        window_len: usize,
    ) {
        let h = self.cfg.hidden_size;
        if spans.is_empty() { return; }
        let w_lo = pos_start;
        let w_hi = pos_start + window_len;
        // `embeds` is the CONCATENATION of every image's merged embeddings, in `spans` order. The
        // within-span row offset `emb_row0` is relative to THIS image's own block, so the source
        // address must be `emb_off + emb_row0`, not `emb_row0` (which is only correct for the FIRST
        // image — image 2+ would splice image 1's embeddings at its own span, the cross-image
        // contamination). `emb_off` tracks each image's global row offset across spans.
        let mut emb_off = 0usize;
        for span in spans {
            let a = span.start.max(w_lo);
            let b = (span.start + span.num_tokens).min(w_hi);
            if b > a {
                let emb_row0 = a - span.start;
                let res_row0 = a - pos_start;
                let nrows = b - a;
                let dst = *residual.device_ptr() as cudarc::driver::sys::CUdeviceptr
                    + (res_row0 * h * 2) as u64;
                let src = *embeds.device_ptr() as cudarc::driver::sys::CUdeviceptr
                    + ((emb_off + emb_row0) * h * 2) as u64;
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(
                        dst, src, nrows * h * 2, self.stream.stream).unwrap();
                }
            }
            emb_off += span.num_tokens;
        }
    }

    pub fn prefill_batch(&self, pool: &mut Pool, prompt: &[u32],
                         state: &mut BatchGpuState, slot: usize, kv_stride: usize,
                         pos_start: usize) -> (u32, B) {
        // Full prefill: every layer, from the embedding. Thin wrapper over the layer-range
        // variant (PP-prefill, PLAN/14 §5.2) — identical code path, full range.
        self.prefill_batch_range(pool, prompt, state, slot, kv_stride, pos_start,
                                 0, self.layers.len(), None)
    }
}
