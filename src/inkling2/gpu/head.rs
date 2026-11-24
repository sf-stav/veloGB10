//! Inkling-2 LM head on the GPU (kernels/gpu_inkling2/65_lm_head.cuh): the stored bf16 `unembed` rows
//! `[row_base, row_base + n)` of the UNPADDED vocabulary (TP vocab-parallel: each rank loads its own row shard;
//! TP=1 loads `[0, 200058)`), fp32 final hidden in, fp32 softcapped logits out, plus the fused argmax partials.

use super::{dptr, layout};
use crate::ink2_launch;
use crate::inkling2::pack::{Inkling2Pack, P};
use anyhow::{ensure, Result};
use cudarc::driver::{CudaDevice, CudaSlice};
use std::sync::Arc;

pub struct HeadDev {
    /// B32 [ceil(n/16)*16][H] bf16
    pub w: CudaSlice<u16>,
    pub n: usize,
    pub k: usize,
    pub row_base: usize,
    pub cap: f32,
}

impl HeadDev {
    /// Rows `[row_base, row_base + n)` of `unembed` (n = 0: the whole unpadded vocabulary from row_base).
    pub fn load(dev: &Arc<CudaDevice>, pack: &Inkling2Pack, row_base: usize, n: usize) -> Result<HeadDev> {
        let c = &pack.cfg;
        let k = c.hidden;
        let n = if n == 0 { c.unpadded_vocab - row_base } else { n };
        ensure!(row_base + n <= c.unpadded_vocab, "LM head rows [{row_base}, {}) beyond the unpadded vocab {}", row_base + n, c.unpadded_vocab);
        let name = format!("{P}unembed.weight");
        let b = pack.st.raw_range(&name, (row_base * k * 2) as u64, ((row_base + n) * k * 2) as u64)?;
        let bits: Vec<u16> = b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let w = dev.htod_copy(layout::bf16_frag(&bits, n, k))?;
        Ok(HeadDev { w, n, k, row_base, cap: c.logit_softcap })
    }
    pub fn grid(&self) -> u32 { self.n.div_ceil(128) as u32 }
    pub fn bytes(&self) -> usize { self.n * self.k * 2 }
}

/// Argmax partial buffers for up to 16 rows.
pub struct HeadScratch {
    pub pmax: CudaSlice<f32>,
    pub pidx: CudaSlice<i32>,
    pub out_idx: CudaSlice<i32>,
    pub out_val: CudaSlice<f32>,
}
impl HeadScratch {
    pub fn new(dev: &Arc<CudaDevice>, h: &HeadDev) -> Result<HeadScratch> {
        let g = h.grid() as usize;
        Ok(HeadScratch { pmax: dev.alloc_zeros(16 * g)?, pidx: dev.alloc_zeros(16 * g)?, out_idx: dev.alloc_zeros(16)?, out_val: dev.alloc_zeros(16)? })
    }
}

/// x [m][K] fp32 (m <= 16) -> logits [m][ldo] (columns 0..n of this shard), argmax partials; `argmax` also reduces
/// them to (out_idx, out_val) [m] (global row ids).
pub fn forward(dev: &Arc<CudaDevice>, h: &HeadDev, hs: &HeadScratch, x: u64, logits: u64, ldo: usize, m: usize, argmax: bool) -> Result<()> {
    ensure!((1..=16).contains(&m) && ldo >= h.n, "inkling2 LM head: m {m} (1..=16), ldo {ldo} >= {}", h.n);
    let g = h.grid();
    ink2_launch!(dev, "ink2_lm_head", (g, 1, 1), (256, 1, 1), 0,
                 (dptr(&h.w, 0), x, logits, dptr(&hs.pmax, 0), dptr(&hs.pidx, 0), m as i32, h.k as i32, h.n as i32,
                  ldo as i32, h.row_base as i32, h.cap))?;
    if argmax {
        ink2_launch!(dev, "ink2_head_argmax", (m as u32, 1, 1), (256, 1, 1), 0,
                     (dptr(&hs.pmax, 0), dptr(&hs.pidx, 0), g as i32, dptr(&hs.out_idx, 0), dptr(&hs.out_val, 0)))?;
    }
    Ok(())
}

/// x [m][K] fp32, any m >= 1 -> logits [m][ldo], no argmax: m <= 16 `forward`; 17..=48 `ink2_lm_head_g` (the weight
/// read once for all rows); more: 48-row groups. Row r's logits are bit-identical whatever path / group it is in
/// (rows are independent MMA rows with one arithmetic).
pub fn forward_rows(dev: &Arc<CudaDevice>, h: &HeadDev, hs: &HeadScratch, x: u64, logits: u64, ldo: usize, m: usize) -> Result<()> {
    if m <= 16 { return forward(dev, h, hs, x, logits, ldo, m, false); }
    for r0 in (0..m).step_by(48) {
        let mm = (m - r0).min(48);
        if mm <= 16 { forward(dev, h, hs, x + (r0 * h.k * 4) as u64, logits + (r0 * ldo * 4) as u64, ldo, mm, false)?; continue; }
        ink2_launch!(dev, "ink2_lm_head_g", (h.grid(), 1, 1), (256, 1, 1), 0,
                     (dptr(&h.w, 0), x + (r0 * h.k * 4) as u64, logits + (r0 * ldo * 4) as u64, mm as i32, h.k as i32, h.n as i32,
                      ldo as i32, h.cap))?;
    }
    Ok(())
}
