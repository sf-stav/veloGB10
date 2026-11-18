//! `Df2Round` block-position refresh and the `dump_*` / `ctl_*` diagnostic readbacks used by the probes.
//! Siblings: `draft.rs` produces what these read back; `mod.rs` has the map.

use super::*;

impl Df2Round {
    /// Refresh the per-round block position arrays (call when nprev changes, before the
    /// round): block RoPE positions = ABSOLUTE `[nprev, nprev+8)`; the linear-control rows
    /// = `[nprev, nprev+8)`.
    pub fn refresh_block_pos(&mut self) -> Result<()> {
        self.pos_at = None;
        let n = self.nprev;
        let pos: Vec<i32> = (0..block()).map(|b| (n + b) as i32).collect();
        self.dev.htod_sync_copy_into(&pos, &mut self.pos_blk)?;
        let ctl: Vec<i32> = (0..block()).map(|b| (n + b) as i32).collect();
        self.dev.htod_sync_copy_into(&ctl, &mut self.wrow_ctl)?;
        self.gather_rope(&self.blk.cos8, &self.blk.sin8, d(&self.pos_blk), block());
        self.dev.synchronize()?;
        self.pos_at = Some(n);
        Ok(())
    }

    /// PERFFIX1: `refresh_block_pos` unless the arrays already hold this frontier's values (the round's draft
    /// refreshed them and nothing moved the ring since): saves two blocking uploads, a rope gather and a device sync
    /// per DF2 round on the drafting rank. A rank that did not draft (a TP node adopting the head's draft) refreshes.
    pub fn refresh_block_pos_if_stale(&mut self) -> Result<()> {
        if self.pos_at == Some(self.nprev) { return Ok(()); }
        self.refresh_block_pos()
    }

    /// The ring-vs-linear attention control (probe-only): re-run LAYER 0's attention with the
    /// S3F `gqa_attn_band_b` kernel over the SAME ring cache (block rows must have been
    /// dual-written) and return the output for a bit-for-bit diff vs the ring kernel's.
    pub fn ctl_linear_attn(&mut self, window: usize) -> Result<Vec<f32>> {
        let ntot = self.nprev + block();
        assert!(ntot <= ring(), "the linear control needs ntot {} <= ring {}", ntot, ring());
        let scale = 1.0f32 / (head_dim() as f32).sqrt();
        let smem = crate::dflash2::band_smem(window, ntot);
        let (nq, nkv) = (self.nq_l, self.nkv_l);
        let nh_packed = ((nq << 20) | (head_dim() << 10) | nkv) as i32;
        let ntot_stride = ((ntot as u64) << 16) | ring_stride() as u64;
        let window_b = ((window << crate::dflash2::DF2_WB_SHIFT) | block()) as i32;
        klaunch!(self, "gqa_attn_band_b", ((block() * nq) as u32, 1, 1), (head_dim() as u32, 1, 1), smem as u32,
            (d(&self.blk.attn), d(&self.blk.q), d(&self.k_ring[0]), d(&self.v_ring[0]),
             d(&self.pos_blk), ntot_stride, nh_packed, window_b, fbits(scale)));
        let a: Vec<bf16> = self.dev.dtoh_sync_copy(&self.blk.attn)?;
        Ok(a.iter().map(|x| x.to_f32()).collect())
    }

    /// Read back the ring k/v rows for one layer (probe diff surfaces), as
    /// `[RING_STRIDE * nkv * hd]` bf16-as-f32.
    /// True per-head ring regions: block rows [C_ring, C_ring+8) of EVERY kv head, plus a
    /// committed-row sample (rows 0..4 and C-4..C) — the write_kv targets and their neighbors.
    pub fn dump_ring_regions(&self, li: usize, c: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let stride_r: usize = ring_stride(); // 2056, runtime (block-dependent)
        let ring_r: usize = ring();            // 2048 for both shipped artifacts
        let nkv = self.nkv_l;
        let mut grab = |buf: &CudaSlice<bf16>| -> Vec<f32> {
            let full: Vec<bf16> = self.dev.dtoh_sync_copy(buf).unwrap();
            let mut out = Vec::with_capacity(nkv * 16 * 128);
            for h in 0..nkv {
                let base = h * stride_r * 128;
                for r in 0..4usize { out.extend(full[base + r * 128..base + r * 128 + 128].iter().map(|x| x.to_f32())); }
                for r in (c - 4)..c { out.extend(full[base + r * 128..base + r * 128 + 128].iter().map(|x| x.to_f32())); }
                for r in ring_r..(ring_r + 8) { out.extend(full[base + r * 128..base + r * 128 + 128].iter().map(|x| x.to_f32())); }
            }
            out
        };
        Ok((grab(&self.k_ring[li]), grab(&self.v_ring[li])))
    }

    /// Re-run ONLY the ring attention (layer 0's exact launch) into ctl_attn_ref — the
    /// determinism discriminator: if this reproduces the dumped attn, the INPUTS changed;
    /// if it reproduces round-0's value, blk.attn was overwritten after attention.
    pub fn ctl_ring_attn(&mut self, window: usize) -> Result<Vec<f32>> {
        let ntot = self.nprev + block();
        let scale = 1.0f32 / (head_dim() as f32).sqrt();
        let smem = crate::dflash2::band_smem(window, ntot);
        let (nq, nkv) = (self.nq_l, self.nkv_l);
        let nh_packed = ((nq << 20) | (head_dim() << 10) | nkv) as i32;
        let packed = (((ntot as u64) << 32) | ((ring() as u64) << 16) | ring_stride() as u64);
        let window_b = ((window << crate::dflash2::DF2_WB_SHIFT) | block()) as i32;
        klaunch!(self, "gqa_attn_band_ring_b", ((block() * nq) as u32, 1, 1), (head_dim() as u32, 1, 1), smem as u32,
            (d(&self.ctl_attn_ref), d(&self.blk.q), d(&self.k_ring[0]), d(&self.v_ring[0]),
             d(&self.pos_blk), packed, self.ntot_dev, nh_packed, window_b, fbits(scale)));
        let v: Vec<bf16> = self.dev.dtoh_sync_copy(&self.ctl_attn_ref)?;
        Ok(v.iter().map(|x| x.to_f32()).collect())
    }

    /// The dual-write control's attention snapshot (recomputed right after attention ran,
    /// BEFORE o_proj/MLP — the pre-corruption reference for the determinism hunt).
    pub fn dump_ctl_attn_ref(&self) -> Result<Vec<f32>> {
        let v: Vec<bf16> = self.dev.dtoh_sync_copy(&self.ctl_attn_ref)?;
        Ok(v.iter().map(|x| x.to_f32()).collect())
    }

    /// S5F3 dump: read back selected ring() rows (k+v, bf16-as-f32) for one layer — the ctx
    /// rows `[c_lo, c_hi)` + the block rows `[ring(), ring()+8)`. The S1 ring-bisect surface.
    pub fn dump_ring_rows(&self, li: usize, c_lo: usize, c_hi: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let nkv = self.nkv_l;
        let hd = crate::dflash2::head_dim();
        let rows: Vec<usize> = (c_lo..c_hi.min(ring())).chain(ring()..ring() + 8).collect();
        // The ring's writers run on the ROUND's stream — the dtoh (NULL stream) must not race
        // them: a device-wide sync first (dump-only path).
        self.dev.synchronize()?;
        let mut grab = |buf: &CudaSlice<bf16>| -> Result<Vec<f32>> {
            let full: Vec<bf16> = self.dev.dtoh_sync_copy(buf)?;
            let mut out = Vec::with_capacity(rows.len() * nkv * hd);
            for h in 0..nkv {
                let base = h * ring_stride() * hd;
                for &r in &rows {
                    out.extend(full[base + r * hd..base + r * hd + hd].iter().map(|x| x.to_f32()));
                }
            }
            Ok(out)
        };
        Ok((grab(&self.k_ring[li])?, grab(&self.v_ring[li])?))
    }

    /// S5F3: the round's fc projection output `th` [5120, 8] bf16-as-f32 (the inject's fc
    /// result — the S1 GEMM-bisect surface vs the oracle's tap_project).
        pub fn dump_ring_kv(&self, li: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let k: Vec<bf16> = self.dev.dtoh_sync_copy(&self.k_ring[li])?;
        let v: Vec<bf16> = self.dev.dtoh_sync_copy(&self.v_ring[li])?;
        Ok((k.iter().map(|x| x.to_f32()).collect(), v.iter().map(|x| x.to_f32()).collect()))
    }

    /// The staging buffer read back as f32 (probe).
    pub fn dump_staging(&self) -> Result<Vec<f32>> {
        let s: Vec<bf16> = self.dev.dtoh_sync_copy(&self.staging)?;
        Ok(s.iter().map(|x| x.to_f32()).collect())
    }

    /// S5F probe debug: the post-final-norm block hidden read back (row-major [8][HIDDEN]).
    pub fn dump_h_final(&self) -> Result<Vec<f32>> {
        let v: Vec<bf16> = self.dev.dtoh_sync_copy(&self.blk.h_final)?;
        Ok(v[..hidden() * block()].iter().map(|x| x.to_f32()).collect())
    }

    /// The block hidden (pre-final-norm) read back (probe).
    pub fn dump_block_h(&self) -> Result<Vec<f32>> {
        let h: Vec<bf16> = self.dev.dtoh_sync_copy(&self.blk.h)?;
        Ok(h.iter().map(|x| x.to_f32()).collect())
    }

    /// The block's last-layer k/v scratch BEFORE write_kv (probe determinism localization).
    pub fn dump_block_kv(&self) -> Result<(Vec<f32>, Vec<f32>)> {
        let k: Vec<bf16> = self.dev.dtoh_sync_copy(&self.blk.k)?;
        let v: Vec<bf16> = self.dev.dtoh_sync_copy(&self.blk.v)?;
        Ok((k.iter().map(|x| x.to_f32()).collect(), v.iter().map(|x| x.to_f32()).collect()))
    }

    /// The block's last-layer k/v scratch BEFORE write_kv (probe determinism localization).
    /// The last-run layer's post-input_ln normed buffer (probe determinism localization).
    pub fn dump_block_normed(&self) -> Result<Vec<f32>> {
        let v: Vec<bf16> = self.dev.dtoh_sync_copy(&self.blk.normed)?;
        Ok(v.iter().map(|x| x.to_f32()).collect())
    }

    /// The last-run layer's dyn conv coefficients (probe determinism localization).
    pub fn dump_block_dyn(&self) -> Result<Vec<f32>> {
        let v: Vec<bf16> = self.dev.dtoh_sync_copy(&self.blk.dyn_attn)?;
        Ok(v.iter().map(|x| x.to_f32()).collect())
    }

    pub fn dump_block_q(&self) -> Result<Vec<f32>> {
        let v: Vec<bf16> = self.dev.dtoh_sync_copy(&self.blk.q)?;
        Ok(v.iter().map(|x| x.to_f32()).collect())
    }

    /// The block's last-layer conv input scratch (probe determinism localization).
    pub fn dump_block_x(&self) -> Result<Vec<f32>> {
        let v: Vec<bf16> = self.dev.dtoh_sync_copy(&self.blk.x_conv)?;
        Ok(v.iter().map(|x| x.to_f32()).collect())
    }

    /// Access the block scratch attention buffer (control diff source).
    pub fn dump_attn_scratch(&self) -> Result<Vec<f32>> {
        let a: Vec<bf16> = self.dev.dtoh_sync_copy(&self.blk.attn)?;
        Ok(a.iter().map(|x| x.to_f32()).collect())
    }
}
