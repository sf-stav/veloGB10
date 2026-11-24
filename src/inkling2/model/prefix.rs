//! S4 prefix cache: a snapshot of one lane's position-dependent state at a prompt end, so the next request whose
//! prompt starts with that whole prompt resumes there instead of prefilling it again (multi-turn chat: the Inkling-2
//! template appends messages and adds no generation prompt, so turn N's prompt is a prefix of turn N + 1's).
//!
//! What a lane needs at position p (`Lane`, mod.rs):
//!   * full-attention K / V rows [0, p): linear in the slot, never overwritten by later positions -> left in place
//!     (the entry is valid only while the lane has not started another prompt);
//!   * sliding K / V rows [p - 1024, p): a 4,096-row ring that the reply overwrites -> copied out;
//!   * the conv input rings (32 rows, rows p - 3 .. p - 1 are read): overwritten by the reply -> copied out whole.
//! Restore copies them back and sets pos = p; the rest of the prompt prefills from p. Prefill rows are independent
//! of how the prompt is chunked (the S3 / S4 path checks: chunk 512 == chunk 2,048 bitwise), so a resumed prompt
//! gives the same bytes as a cold one.

use super::{GpuModel, Lane};
use crate::inkling2::gpu::dp;
use anyhow::{ensure, Result};
use cudarc::driver::{sys, CudaSlice, DeviceSlice};

/// Rows of the sliding window kept per sliding slot.
const WIN: usize = 1024;
/// Bytes of one K (or V) cache row of slot `slot` (8 heads x 128 x f16 at TP1; the rank's KV heads under TP).
fn rowb(lane: &Lane, slot: usize) -> usize { lane.kc[slot].len() / lane.cap[slot] * 2 }

pub struct Snap {
    /// per slot: [K rows WIN][V rows WIN] (sliding slots only; full slots: None)
    kv: Vec<Option<CudaSlice<u16>>>,
    /// per slot: the 4 conv rings, concatenated
    rings: Vec<CudaSlice<f32>>,
}

/// A saved prompt end on one lane.
/// `mtp`: the drafter's lane state at the same prompt end (speculation on), so a resumed prompt speculates too.
/// `img`: (first row, rows, image hash) of the images inside `toks` (a hit needs the same images at the same rows).
pub struct Entry { pub toks: Vec<u32>, pub snap: Snap, pub mtp: Option<crate::inkling2::mtp::gpu::MtpSnap>, pub img: Vec<(usize, usize, u64)> }

fn d2d(dst: u64, src: u64, bytes: usize) -> Result<()> {
    if bytes == 0 { return Ok(()); }
    let r = unsafe { sys::cuMemcpyDtoDAsync_v2(dst, src, bytes, std::ptr::null_mut()) };
    ensure!(r == sys::CUresult::CUDA_SUCCESS, "prefix cache copy: {r:?}");
    Ok(())
}

/// Copy positions [a, b) between a cache of `cap_x` rows (row = q & (cap_x - 1)) and one of `cap_y` rows, in
/// contiguous pieces. `to_snap`: cache (x) -> snapshot (y), else snapshot -> cache.
fn copy_rows(cache: u64, cap: usize, snap: u64, a: usize, b: usize, to_snap: bool, rb: usize) -> Result<()> {
    let mut q = a;
    while q < b {
        let (rc, rs) = (q & (cap - 1), q & (WIN - 1));
        let n = (b - q).min(cap - rc).min(WIN - rs);
        let (c, s) = (cache + (rc * rb) as u64, snap + (rs * rb) as u64);
        if to_snap { d2d(s, c, n * rb)?; } else { d2d(c, s, n * rb)?; }
        q += n;
    }
    Ok(())
}

impl GpuModel {
    pub fn new_snap(&self, lane: &Lane) -> Result<Snap> {
        let n = self.cfg.visits();
        let mut kv = Vec::with_capacity(n);
        let mut rings = Vec::with_capacity(n);
        for slot in 0..n {
            let l = if slot < self.cfg.layers { slot } else { self.cfg.loop_start + (slot - self.cfg.layers) % self.cfg.loop_len() };
            kv.push(if self.cfg.is_sliding[l] { Some(self.dev.alloc_zeros::<u16>(2 * WIN * rowb(lane, slot) / 2)?) } else { None });
            let tot: usize = lane.ring[slot].iter().map(|r| r.len()).sum();
            rings.push(self.dev.alloc_zeros::<f32>(tot)?);
        }
        Ok(Snap { kv, rings })
    }

    /// Save `lane`'s state at its current position (a prompt end) into `snap`. Stream-ordered (NULL stream).
    pub fn snap_save(&self, lane: &Lane, snap: &mut Snap) -> Result<()> {
        self.mirror(|| crate::inkling2::tp::Op::SnapSave { lane: lane.id, pos: lane.pos })?;
        self.snap_copy(lane, snap, true)
    }
    /// Restore `snap` (saved at position p) into `lane`; the caller sets `lane.pos = p`.
    pub fn snap_restore(&self, lane: &Lane, snap: &Snap, p: usize) -> Result<()> {
        self.mirror(|| crate::inkling2::tp::Op::SnapRestore { lane: lane.id, p })?;
        self.snap_copy_at(lane, snap, false, p)
    }

    fn snap_copy(&self, lane: &Lane, snap: &Snap, to_snap: bool) -> Result<()> { self.snap_copy_at(lane, snap, to_snap, lane.pos) }

    fn snap_copy_at(&self, lane: &Lane, snap: &Snap, to_snap: bool, p: usize) -> Result<()> {
        for slot in 0..lane.kc.len() {
            if let Some(buf) = &snap.kv[slot] {
                let cap = lane.cap[slot];
                ensure!(cap >= WIN && cap.is_power_of_two(), "sliding cache rows {cap}");
                let a = p.saturating_sub(WIN);
                let rb = rowb(lane, slot);
                copy_rows(dp(&lane.kc[slot]), cap, dp(buf), a, p, to_snap, rb)?;
                copy_rows(dp(&lane.vc[slot]), cap, dp(buf) + (WIN * rb) as u64, a, p, to_snap, rb)?;
            }
            let mut off = 0u64;
            for r in &lane.ring[slot] {
                let bytes = r.len() * 4;
                if to_snap { d2d(dp(&snap.rings[slot]) + off, dp(r), bytes)?; } else { d2d(dp(r), dp(&snap.rings[slot]) + off, bytes)?; }
                off += bytes as u64;
            }
        }
        Ok(())
    }
}
