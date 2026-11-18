//! RoPE frequency tables (RopeFreqs, prefill cos/sin), the VIS-2 per-slot mrope gather (SlotSrc, cos_gather,
//! mrope_cos_row), the vision row upload and qsa_key_rope_full. See `mod.rs` for the file map.

use super::*;

/// YARN1: the ONE place the EXL3 trunk's rope rescaling lives — the per-dim inverse
/// frequencies and the attention mscale of every table this model serves (decode cos_tab,
/// the prefill host loop, the QSA indexer tables, the vision mrope rows). Native, or YaRN at
/// factor > 1 via the dspark oracle's validated ramp (`yaarn_freqs`) + `yaarn_mscale`.
/// Factor 1.0 reproduces the pre-YaRN tables bit-for-bit: the same
/// `theta.powf(-(2i)/rdim)` formula and op order, mscale exactly 1.0 (an exact f32 identity,
/// so `x * mscale` cannot perturb a value). The factor is resolved once per process by
/// `qwen::resolved_yarn` (--rope-yarn-factor > shipped TpConfig > the pack's `rope_scaling`);
/// the boot line names the factor and its source.
pub(crate) struct RopeFreqs {
    mscale: f32,
    inv: Vec<f32>,
}

impl RopeFreqs {
    /// Native RoPE — the load-time loop's formula and op order, verbatim.
    pub fn native(rdim: usize, theta: f32) -> RopeFreqs {
        RopeFreqs { mscale: 1.0, inv: (0..rdim / 2).map(|i| theta.powf(-(2.0 * i as f32) / rdim as f32)).collect() }
    }

    /// YaRN at an explicit factor (no resolution, no boot line): the oracle ramp + mscale.
    pub fn yarn(rdim: usize, theta: f32, factor: f32, native_max: usize) -> RopeFreqs {
        RopeFreqs { mscale: crate::dspark::oracle::yaarn_mscale(factor),
                    inv: crate::dspark::oracle::yaarn_freqs(rdim, theta, factor, native_max, 32, 1) }
    }

    /// The rope this process serves for `cfg`: native, or YaRN at the resolved factor
    /// (vLLM per-dim ramp, beta 32/1, over `native_max`). Prints the boot line when active.
    pub fn for_opts(rdim: usize, theta: f32, native_max: usize, cfg: &crate::qwen::Config) -> RopeFreqs {
        let (factor, src) = crate::qwen::resolved_yarn(cfg);
        if factor <= 1.0 {
            return RopeFreqs::native(rdim, theta);
        }
        let src = match src {
            Some(crate::qwen::YarnSource::Flag) => "--rope-yarn-factor",
            Some(crate::qwen::YarnSource::Tp) => "shipped TpConfig",
            Some(crate::qwen::YarnSource::Config) | None => "config rope_scaling",
        };
        crate::drprintln!("[rope] EXL3 trunk YaRN factor {factor} (from {src}): mscale {:.4}, native {native_max} (vLLM per-dim ramp, beta 32/1; shared oracle yaarn_freqs)",
                 crate::dspark::oracle::yaarn_mscale(factor));
        RopeFreqs::yarn(rdim, theta, factor, native_max)
    }

    /// (cos(p·inv[i]), sin(p·inv[i])) scaled by mscale. At factor 1.0 mscale is exactly 1.0,
    /// so the multiply is the f32 identity and legacy tables stay byte-identical.
    pub fn cos_sin(&self, p: usize, i: usize) -> (f32, f32) {
        let f = p as f32 * self.inv[i];
        (f.cos() * self.mscale, f.sin() * self.mscale)
    }

    pub fn inv(&self, i: usize) -> f32 { self.inv[i] }
    pub fn mscale(&self) -> f32 { self.mscale }
    pub fn half(&self) -> usize { self.inv.len() }
}

/// The prefill prologue's pre-WP05 host RoPE rows for positions pos0..pos0+c (per row:
/// cos[rdim/2] | 0[rdim/2] | sin[rdim/2] | 0[rdim/2]) — the old per-chunk loop verbatim; the
/// --wp05-off=pro path and the load-time cos_tab comparison. YARN1: takes the model's
/// `RopeFreqs` — the tables' own source — so the host loop and the tables can never disagree
/// about a YaRN factor.
pub(super) fn prefill_cossin_host(fr: &RopeFreqs, rdim: usize, pos0: usize, c: usize) -> Vec<f32> {
    let mut cossin: Vec<f32> = Vec::with_capacity(c * 2 * rdim);
    for p in pos0..pos0 + c {
        for i in 0..rdim / 2 { cossin.push(fr.cos_sin(p, i).0); }
        for _ in 0..rdim / 2 { cossin.push(0.0); }
        for i in 0..rdim / 2 { cossin.push(fr.cos_sin(p, i).1); }
        for _ in 0..rdim / 2 { cossin.push(0.0); }
    }
    cossin
}



/// VIS-2: where `xq_cos_gather_v` finds each row's slot id.
#[derive(Clone, Copy)]
pub(crate) enum SlotSrc {
    /// the per-row [slot, pos] pairs buffer (sc.slots / psc.slotpos_rows): row b's slot at [2b]
    Pairs(u64),
    /// one device int (a re-prime meta[0]) for every row
    Ptr(u64),
    /// one host slot for every row (prefill chunks, the host re-prime)
    Const(usize),
}

impl FwdModel {
    /// VIS-2: the main-attention RoPE row gather through the per-slot mrope map (`xq_cos_gather_v`):
    /// `m` rows of [2 * rotary_dim] into `dst` for the KV positions at `pos` (device i32 [m]).
    /// A text-only slot (rope_plen = rope_delta = 0) reads cos_tab[pos] — the old xq_cos_gather.
    pub(crate) fn cos_gather(&self, l: &Launcher, dst: &mut CudaSlice<f32>, pos: u64, slot: SlotSrc, m: usize) -> Result<()> {
        let stride = 2 * self.cfg.rotary_dim;
        let (sp, ss, sk) = match slot {
            SlotSrc::Pairs(p) => (p, 2i32, -1i32),
            SlotSrc::Ptr(p) => (p, 0, -1),
            SlotSrc::Const(s) => (0u64, 0, s as i32),
        };
        let dp = |x: &CudaSlice<i32>| *x.device_ptr() as u64;
        let (tab, mtab, d) = (*self.cos_tab.device_ptr() as u64, *self.mrope_tab.device_ptr() as u64, *dst.device_ptr() as u64);
        xqlaunch_raw!(l, "xq_cos_gather_v", ((((m * stride) as u32) + 255) / 256, 1, 1), (256, 1, 1), 0,
                      (tab, mtab, d, pos, sp, ss, sk, dp(&self.rope_map), dp(&self.rope_plen),
                       dp(&self.rope_delta), self.width as i32, self.max_pos as i32, stride as i32, m as i32))?;
        Ok(())
    }
}

impl FwdModel {
    /// VIS-2: arm `slot` for an image request — its prompt's 3-axis RoPE positions `pos3`
    /// (`vision_encoder::mrope_positions`), the post-prompt `delta`, and the interleaved
    /// `mrope_section` ([11, 11, 10]: frequency i takes the H position when i % 3 == 1 and
    /// i < 3 * sec[1], W when i % 3 == 2 and i < 3 * sec[2], else T — HF apply_interleaved_mrope).
    /// Text positions (t == h == w) map to cos_tab[t]; every other row is computed here with
    /// cos_tab's own f32 formula into the slot's mrope_tab block.
    pub fn set_slot_vision(&self, slot: usize, pos3: &[[i64; 3]], delta: i64, sec: [usize; 3]) -> Result<()> {
        anyhow::ensure!(slot < self.width, "VIS-2: slot {slot} out of range");
        let len = pos3.len();
        anyhow::ensure!(len <= self.max_pos, "VIS-2: {len}-token prompt past max_pos {}", self.max_pos);
        let rdim = self.cfg.rotary_dim;
        let stride = 2 * rdim;
        let mut map = Vec::with_capacity(len);
        let mut rows: Vec<f32> = Vec::new();
        let (mut qrows_c, mut qrows_s): (Vec<f32>, Vec<f32>) = (Vec::new(), Vec::new());
        for p in pos3 {
            anyhow::ensure!(p.iter().all(|&v| v >= 0 && (v as usize) < self.max_pos), "VIS-2: position {p:?} out of the RoPE table");
            if p[0] == p[1] && p[1] == p[2] {
                map.push(p[0] as i32);
                continue;
            }
            let k = rows.len() / stride;
            anyhow::ensure!(k < VIS_MROPE_ROWS, "VIS-2: more than {VIS_MROPE_ROWS} image tokens in one request");
            map.push((self.max_pos + slot * VIS_MROPE_ROWS + k) as i32);
            let row = mrope_cos_row(&self.rope, *p, sec);
            // VIS-3: the indexer's row for the same position, in qsa_cos/qsa_sin's layout
            let h = rdim / 2;
            let mut qc = vec![0f32; rdim];
            let mut qs = vec![0f32; rdim];
            qc[..h].copy_from_slice(&row[..h]);
            qs[..h].copy_from_slice(&row[rdim..rdim + h]);
            if self.qsa_full {
                qc[h..].copy_from_slice(&row[..h]);
                qs[h..].copy_from_slice(&row[rdim..rdim + h]);
            }
            qrows_c.extend(qc);
            qrows_s.extend(qs);
            rows.extend(row);
        }
        anyhow::ensure!(delta <= 0 && delta > -(self.max_pos as i64), "VIS-2: rope delta {delta} out of range");
        vis_upload(&self.dev, &self.rope_map, slot * self.max_pos, &map)?;
        if !rows.is_empty() {
            vis_upload(&self.dev, &self.mrope_tab, slot * VIS_MROPE_ROWS * stride, &rows)?;
            vis_upload(&self.dev, &self.qm_cos, slot * VIS_MROPE_ROWS * rdim, &qrows_c)?;
            vis_upload(&self.dev, &self.qm_sin, slot * VIS_MROPE_ROWS * rdim, &qrows_s)?;
        }
        vis_upload(&self.dev, &self.rope_plen, slot, &[len as i32])?;
        vis_upload(&self.dev, &self.rope_delta, slot, &[delta as i32])?;
        Ok(())
    }

    /// VIS-2: upload one request's image rows for the prefill splice.
    pub fn vis_rows_upload(&self, rows: &[f32]) -> Result<CudaSlice<f32>> {
        Ok(self.dev.htod_sync_copy(rows)?)
    }

    /// VIS-2: back to plain text RoPE for `slot` (rope_plen = rope_delta = 0: cos_tab[pos]).
    pub fn clear_slot_vision(&self, slot: usize) -> Result<()> {
        anyhow::ensure!(slot < self.width, "VIS-2: slot {slot} out of range");
        vis_upload(&self.dev, &self.rope_plen, slot, &[0i32])?;
        vis_upload(&self.dev, &self.rope_delta, slot, &[0i32])?;
        Ok(())
    }
}

/// VIS-2: one RoPE row [cos(rdim/2) | 0 (rdim/2) | sin(rdim/2) | 0 (rdim/2)] for a 3-axis position,
/// in cos_tab's layout and from the model's own `RopeFreqs` (cos_tab's source; YARN1), each
/// frequency i taking the axis HF apply_interleaved_mrope gives it: H when i % 3 == 1 and
/// i < 3 * sec[1], W when i % 3 == 2 and i < 3 * sec[2], else T. For t == h == w this is
/// cos_tab's row t, bit for bit.
pub(crate) fn mrope_cos_row(fr: &RopeFreqs, p: [i64; 3], sec: [usize; 3]) -> Vec<f32> {
    let rdim = fr.half() * 2;
    let axis = |i: usize| -> usize {
        if i % 3 == 1 && i < 3 * sec[1] { 1 } else if i % 3 == 2 && i < 3 * sec[2] { 2 } else { 0 }
    };
    let mut row = Vec::with_capacity(2 * rdim);
    for i in 0..rdim / 2 {
        row.push(fr.cos_sin(p[axis(i)] as usize, i).0);
    }
    row.extend(std::iter::repeat(0.0).take(rdim / 2));
    for i in 0..rdim / 2 {
        row.push(fr.cos_sin(p[axis(i)] as usize, i).1);
    }
    row.extend(std::iter::repeat(0.0).take(rdim / 2));
    row
}

/// VIS-2: synchronous host->device copy of `src` into `buf` at element offset `off`, through a
/// shared (&) buffer. A NULL-stream copy: it orders after all work already queued on the blocking
/// compute stream and completes before returning (AGENTS §2.1 — the same ordering every htod relies on).
fn vis_upload<T: cudarc::driver::DeviceRepr>(dev: &Arc<CudaDevice>, buf: &CudaSlice<T>, off: usize, src: &[T]) -> Result<()> {
    dev.bind_to_thread()?; // a raw driver call: the calling thread needs the context current
    anyhow::ensure!(off + src.len() <= buf.len(), "VIS-2: upload {}..{} past a {}-element buffer", off, off + src.len(), buf.len());
    let dst = *buf.device_ptr() + (off * std::mem::size_of::<T>()) as u64;
    unsafe { cudarc::driver::result::memcpy_htod_sync(dst, src) }.map_err(|e| anyhow::anyhow!("VIS-2 upload: {e:?}"))?;
    Ok(())
}
