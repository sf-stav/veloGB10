//! `Df2Round` state: tap-sink attach and staging sync, TP all-reduce helper, ring bookkeeping, reset / rewind and probes.
//! Siblings: `inject.rs` writes the ring this tracks; `mod.rs` has the map.

use super::*;

impl Df2Round {
    /// Attach the TRUNK's tap sink as this round's fc input (the capture staging is then read
    /// directly by [`inject_dev`] — no intermediate copy).
    pub fn attach_sink(&mut self, sink: &Arc<Df2TapSink>) {
        // S5F3 fix: cudarc's CudaSlice::clone() is a DEEP copy (try_clone = fresh alloc +
        // dtod), NOT a shared handle. Cloning here snapshot the sink's then-zeroed buffer and
        // the trunk's live captures never reached the round: every inject consumed ZERO taps.
        // Keep the Arc; the serving lanes call `sync_staging_from_sink` before each inject.
        self.sink = Some(sink.clone());
    }

    /// S5F3: copy the attached sink's LIVE staging (the trunk's capture columns [0..8)) into
    /// the round's own staging, stream-ordered on the round's stream. The caller must have
    /// synced the trunk's capture first (the lane calls this after the verify returns).
    /// No-op when no sink is attached (the probe's upload_chunk path writes self.staging).
    pub fn sync_staging_from_sink(&mut self) -> Result<()> {
        if let Some(sink) = &self.sink {
            use cudarc::driver::sys;
            let cp = sys::CUDA_MEMCPY2D {
                srcXInBytes: 0, srcY: 0,
                srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                srcHost: std::ptr::null(), srcDevice: *sink.staging.device_ptr() as u64,
                srcArray: std::ptr::null_mut(), srcPitch: tap_concat_dim() * 2,
                dstXInBytes: 0, dstY: 0,
                dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                dstHost: std::ptr::null_mut(), dstDevice: *self.staging.device_ptr() as u64,
                dstArray: std::ptr::null_mut(), dstPitch: tap_concat_dim() * 2,
                WidthInBytes: tap_concat_dim() * 2, Height: block(),
            };
            unsafe {
                let r = sys::cuMemcpy2DAsync_v2(&cp, self.stream.stream);
                if r != sys::CUresult::CUDA_SUCCESS {
                    anyhow::bail!("df2 sync staging D2D failed: {r:?}");
                }
            }
            self.dsync()?;
        }
        Ok(())
    }

    pub fn nprev(&self) -> usize { self.nprev }

    /// P2: the rank-local kv-head count (NUM_KV_HEADS / world when sharded).
    pub fn nkv(&self) -> usize { self.nkv_l }

    /// P2 — world>2 all-reduce of a bf16 buffer ON THE ROUND'S STREAM: the faithful mirror of
    /// `GpuModel::tp_all_reduce_bf16`'s world>2 branch — one-shot (P3-1 predicate permitting)
    /// or the recursive-doubling tree (K1 `tp_gate_copy_signal` + K2 `tp_wait_add{,_g}`, fp32
    /// accumulate in the persistent scratch, ONE bf16 rounding per chunk via `f32tobf16`, the
    /// canonical fold that keeps the sum bitwise rank-identical). The round's payloads
    /// ([5120×8] bf16 = 80 KB) always fit a single chunk (slot ~1 MB). Launched inside
    /// `layer_forward`, so the sites capture into the round's CUDA graph exactly like the
    /// trunk's verify-graph ARs; the device-side epoch counter (not a kernel arg) advances
    /// identically on every rank — SPMD by construction.
    pub(super) fn tp_ar_bf16(&self, ptr: u64, n: usize) {
        let Some(ar) = &self.ar else { panic!("df2 tp_ar_bf16: no AR ctx (unsharded round)") };
        debug_assert!(ar.world > 2, "df2 round shard engages only at world > 2");
        if ar.oneshot && n * 2 <= ar.payload_bytes.max(1) {
            klaunch!(self, "tp_gate_copy_signal", (1u32, 1, 1), (1024, 1, 1), 0,
                (ar.ctx_dptr, ptr, (n * 2) as u32));
            klaunch!(self, "tp_wait_add_4way", (1u32, 1, 1), (1024, 1, 1), 0,
                (ar.ctx_dptr, ptr, ptr, n as i32, 0i32));
            return;
        }
        let sp = ar.f32_scratch;
        let rounds = ar.world.trailing_zeros() as usize;   // log2(world); world>2 ⇒ ≥2
        let chunk = ((crate::tp::TP_SLOT_BYTES - 64) & !7) / 4;   // fp32 elems per ring slot
        let k2: &str = if ar.k2_gpu_recv { "tp_wait_add_g" } else { "tp_wait_add" };
        let mut off = 0usize;
        while off < n {
            let c = (n - off).min(chunk);
            let bp = ptr + (off * 2) as u64;   // bf16 partial source (round-0 K1 input)
            klaunch!(self, "tp_gate_copy_signal", (1u32, 1, 1), (1024, 1, 1), 0,
                (ar.ctx_dptr, bp, (c * 2) as u32));
            klaunch!(self, k2, (1u32, 1, 1), (1024, 1, 1), 0,
                (ar.ctx_dptr, sp, bp, c as i32, 3i32));
            for _ in 1..rounds {
                klaunch!(self, "tp_gate_copy_signal", (1u32, 1, 1), (1024, 1, 1), 0,
                    (ar.ctx_dptr, sp, (c * 4) as u32));
                klaunch!(self, k2, (1u32, 1, 1), (1024, 1, 1), 0,
                    (ar.ctx_dptr, sp, sp, c as i32, 2i32));
            }
            // Round this chunk's fp32 sum to bf16 ONCE (the single rounding boundary).
            klaunch!(self, "f32tobf16", grid(c), (256, 1, 1), 0, (bp, sp, c as i32));
            off += c;
        }
    }

    /// The drafter's ring-KV allocation ranges (ptr, bytes) per layer — the S5F no-alias probe
    /// asserts these never intersect the trunk's KV-cache ranges (the drafter's ring is
    /// drafter-private; an overlap would corrupt trunk KV on write).
    pub fn ring_kv_ptr_ranges(&self) -> Vec<(u64, u64)> {
        let bytes = self.nkv_l * ring_stride() * head_dim() * 2;
        (0..n_layers()).flat_map(|li| {
            [(d(&self.k_ring[li]), bytes as u64), (d(&self.v_ring[li]), bytes as u64)]
        }).collect()
    }

    /// Gather ONLY the block input (anchor + 7×MASK) from the trunk's embed — the probe's
    /// embed gate, isolated from the layer stack.
    pub fn embed_probe(&mut self, anchor: u32) -> Result<Vec<f32>> {
        let toks: Vec<i32> = [anchor as i32].iter().copied()
            .chain(std::iter::repeat(crate::dflash2::mask_token_id() as i32).take(block() - 1)).collect();
        self.dev.htod_sync_copy_into(&toks, &mut self.toks_blk)?;
        self.embed_gather();
        self.dev.synchronize()?;
        let h: Vec<bf16> = self.dev.dtoh_sync_copy(&self.blk.h)?;
        Ok(h.iter().map(|x| x.to_f32()).collect())
    }

    /// DFLASH2-GEN (mask sub-item): one-boot measurement of the trunk's QUANTIZED mask row
    /// vs the artifact's shipped BF16 copy (`mask_embedding.safetensors` — an exact copy of
    /// the target's pre-quant row). The incoai card documents the failure class: the row's
    /// values are ~1e-6-scale and a format that zeroes tiny block scales destroys it, halving
    /// acceptance. Our own NVFP4 pack has two-level scales and *should* survive, but that is
    /// a measurement, not an assumption: this runs the SERVED gather path (the borrowed
    /// embed's dequant) once for the mask token and reports rel-L2 / max-abs. Verdict bands:
    /// rel-L2 <= 0.05 healthy (normal NVFP4 quant noise); 0.05..0.25 elevated (logged, no
    /// action); > 0.25 DAMAGED — surfaced loudly as an owner decision with this receipt
    /// (the fix is reading this override file into the round, per the findings doc).
    pub(super) fn mask_row_check(&mut self, dir: &str) {
        use safetensors::SafeTensors;
        if self.embed.is_none() {
            return; // no borrowed embed (e.g. the GPTQ calibration load) — nothing to measure.
        }
        let p = std::path::Path::new(dir).join("mask_embedding.safetensors");
        let buf = match std::fs::read(&p) {
            Ok(b) => b,
            Err(_) => return, // artifact ships no override copy — nothing to compare against.
        };
        let st = match SafeTensors::deserialize(&buf) {
            Ok(s) => s,
            Err(e) => {
                crate::reprintln!("[df2] mask row: unreadable {} — {e}", p.display());
                return;
            }
        };
        let Some(name) = st.names().first().map(|s| s.to_string()) else { return };
        let Ok(view) = st.tensor(&name) else { return };
        let h = hidden();
        if view.dtype() != safetensors::Dtype::BF16 || view.shape() != [h] {
            crate::reprintln!(
                "[df2] mask row: override {name} dtype {:?} shape {:?} != BF16 [{h},] — ignored",
                view.dtype(), view.shape());
            return;
        }
        let ref_row: Vec<f32> = view.data()
            .chunks_exact(2)
            .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect();
        let gathered = match self.embed_probe(crate::dflash2::mask_token_id()) {
            Ok(v) => v,
            Err(e) => {
                crate::reprintln!("[df2] mask row: gather probe failed — {e:#}");
                return;
            }
        };
        // Row 0 of the probe block IS the mask row exactly as serving gathers it.
        let g = &gathered[..h];
        let mut num = 0f64;
        let mut den = 0f64;
        let mut mx = 0f64;
        for i in 0..h {
            let d = (g[i] - ref_row[i]) as f64;
            num += d * d;
            den += (ref_row[i] as f64) * (ref_row[i] as f64);
            mx = mx.max(d.abs());
        }
        let rel = if den > 0.0 { (num / den).sqrt() } else { f64::INFINITY };
        let verdict = if rel > 0.25 {
            "DAMAGED — see the override line below for whether the artifact's BF16 copy serves"
        } else if rel > 0.05 {
            "elevated (logged; within quant-noise-plus; no action)"
        } else {
            "healthy (normal quant noise)"
        };
        crate::reprintln!(
            "[df2] mask row {}: rel-L2 {rel:.4} max-abs {mx:.5} vs {name} in mask_embedding.safetensors — {verdict}",
            crate::dflash2::mask_token_id());
        if rel > 0.25 && !crate::dflash2::mask_override() {
            crate::reprintln!(
                "[df2] mask row OVERRIDE suppressed by --df2-mask-override off — the DAMAGED \
                 quantized row serves MASK positions (the measured A/B arm)");
            return;
        }
        if rel > 0.25 {
            // DFLASH2-GEN: the row is destroyed by quantization — read the artifact's exact
            // BF16 copy into the round (the sanctioned fix): every block's MASK rows embed
            // the copy instead of the trunk's quantized row. One-time staging; the broadcast
            // rides embed_gather (graph-stable: staged at load, before any capture).
            let ovr_bf: Vec<bf16> = ref_row.iter().map(|&x| bf16::from_f32(x)).collect();
            let toks: Vec<i32> = vec![0i32; block() - 1];
            if let (Ok(t), Ok(r)) = (self.dev.htod_sync_copy(&toks),
                                     self.dev.htod_sync_copy(&ovr_bf)) {
                self.mask_ovr_toks = Some(t);
                self.mask_ovr_row = Some(r);
                crate::metrics::df2_mask_override(true);
                crate::reprintln!(
                    "[df2] mask row OVERRIDE engaged: the block's MASK rows now embed the artifact's exact BF16 copy");
            } else {
                crate::reprintln!(
                    "[df2] mask row OVERRIDE staging FAILED — serving continues with the damaged quantized row");
            }
        }
    }

    /// Embed the block input `toks_blk` from the TRUNK's embed, dispatching on the borrowed
    /// dtype: NVFP4 (the MMA-repacked serving embed) or BF16 (the plain-BF16 trunk class).
    pub(super) fn embed_gather(&self) {
        match self.embed.expect("embed ptrs") {
            BorrowedW::TwoStage { .. } => {
                // df2_borrow only ever wraps the HEAD in TwoStage; the embed is never TwoStage.
                panic!("TwoStage borrow on the embed (contract violation)");
            }
            BorrowedW::Nvfp4(p) => {
                klaunch!(self, "embed_gather_fp4_tiled_b", grid(hidden() * block()), (256, 1, 1), 0,
                    (d(&self.blk.h), p.qweight, p.scales, p.gs, d(&self.toks_blk), hidden() as i32, block() as i32));
            }
            BorrowedW::Bf16 { ptr } => {
                klaunch!(self, "embed_gather_b", grid(hidden() * block()), (256, 1, 1), 0,
                    (d(&self.blk.h), ptr, d(&self.toks_blk), hidden() as i32, block() as i32));
            }
        }
        // DFLASH2-GEN mask override: with a DAMAGED quantized row staged, the MASK rows
        // (1..block — [anchor, MASK..] by construction at every gather site) are overwritten
        // with the artifact's exact BF16 copy: one broadcast gather over a 1-row table.
        if let (Some(row), Some(toks)) = (&self.mask_ovr_row, &self.mask_ovr_toks) {
            let rows = block() - 1;
            // Row 1 of blk.h starts at hidden()*2 BYTES (bf16 elements) past the base —
            // the out pointer is a byte address; a bare `+ hidden()` lands half a row in.
            let row1 = std::mem::size_of::<bf16>() as u64 * hidden() as u64;
            klaunch!(self, "embed_gather_b", grid(hidden() * rows), (256, 1, 1), 0,
                (d(&self.blk.h) + row1, d(row), d(toks), hidden() as i32, rows as i32));
        }
    }

    /// The borrowed head at N=7 on h_final cols 1..7 (`[5120,7]` col-major at +5120), dispatching
    /// on the borrowed dtype: NVFP4 (the trunk's own persistent `gemm_mma_fp4_b`; N<=16 legal via
    /// the N-clamped X reads) or BF16 (`gemm_binv_b`, NC=7 specialization, the batch-invariant
    /// fixed-order reduction — the same kernel the bf16 serving chain's logits use).
    pub(super) fn head_logits(&self) {
        match self.head.expect("head ptrs") {
            BorrowedW::TwoStage { q, bf16_ptr } => {
                // Stage 1: coarse NVFP4 pass over the full vocab (reads 0.68 GB).
                let persistent = (crate::gpu::GB10_SMS * 6).min(vocab() / 16) as u32;
                klaunch!(self, "gemm_mma_fp4_b", (persistent, 1, 1), (256, 1, 1), 0,
                    (d(&self.logits), q.qweight, q.scales, q.gs,
                     d(&self.blk.h_final) + (hidden() * 2) as u64,
                     vocab() as i32, hidden() as i32, levels() as i32, 0u64, 0i32));
                // Stage 1b: per-column top-256 shortlist (deterministic; recall@256 is the only
                // quality requirement — stage 2 supplies the exact ranking).
                klaunch!(self, "df2_top256_b", (levels() as u32, 1, 1), (256, 1, 1), 0,
                    (d(&self.coarse_idx), d(&self.logits), vocab() as i32, levels() as i32));
                // Stage 2: exact bf16 re-rank of the shortlist against the ORIGINAL hiddens
                // (2.6 MB read; fp32 accumulation, deterministic order).
                klaunch!(self, "df2_head_rerank_b", (levels() as u32 * 256, 1, 1), (256, 1, 1), 0,
                    (d(&self.logits), d(&self.coarse_idx), bf16_ptr,
                     d(&self.blk.h_final) + (hidden() * 2) as u64,
                     vocab() as i32, hidden() as i32, levels() as i32));
            }
            BorrowedW::Nvfp4(p) => {
                let persistent = (crate::gpu::GB10_SMS * 6).min(vocab() / 16) as u32;
                klaunch!(self, "gemm_mma_fp4_b", (persistent, 1, 1), (256, 1, 1), 0,
                    (d(&self.logits), p.qweight, p.scales, p.gs,
                     d(&self.blk.h_final) + (hidden() * 2) as u64,
                     vocab() as i32, hidden() as i32, levels() as i32, 0u64, 0i32));
            }
            BorrowedW::Bf16 { ptr } => {
                let smem = (levels() * 256 * 4) as u32;
                klaunch!(self, "gemm_binv_b", (vocab() as u32, 1, 1), (256, 1, 1), smem,
                    (d(&self.logits), ptr,
                     d(&self.blk.h_final) + (hidden() * 2) as u64,
                     vocab() as i32, hidden() as i32, levels() as i32));
            }
        }
    }

    /// Restart the round at position 0 (the ring contents are simply re-overwritten in
    /// position order; un-committed rows are never read).
    pub fn reset(&mut self) {
        self.nprev = 0;
        // DF2_CARRY: a reset drops the ring's identity claim as well as its frontier. Any
        // rows still physically present must not be carried by a later request.
        self.ring_slot = None;
        self.ring_len = 0;
    }

    /// DF2_CARRY: the slot whose committed prefix the ring currently reflects (None = unproven).
    pub fn ring_slot(&self) -> Option<usize> { self.ring_slot }
    /// DF2_CARRY: the committed frontier `nprev` the ring is valid up to.
    pub fn ring_len(&self) -> usize { self.ring_len }

    /// DF2_CARRY: bind the ring to `slot` at the CURRENT committed frontier. Called after the
    /// lane has primed and stepped, so `nprev` is the whole committed sequence this ring holds.
    pub fn note_ring(&mut self, slot: usize) {
        self.ring_slot = Some(slot);
        self.ring_len = self.nprev;
    }

    /// DF2_CARRY: drop the identity claim — the ring's contents can no longer be attributed to
    /// any slot's prefix (a failed prime can leave it half-overwritten).
    pub fn invalidate_ring(&mut self) {
        self.ring_slot = None;
        self.ring_len = 0;
    }

    /// DF2_CARRY: move the committed frontier back to `nprev` keeping rows `[0, nprev)` — the
    /// exact twin of `DsparkRound::rewind`. NOTE this only moves the frontier: it does not make
    /// the rows valid. The caller must have PROVEN them still resident (a modulo ring recycles
    /// rows, so `ring_len >= nprev` alone is not sufficient — see the scheduler's guard).
    /// The first `prime_window` overwrites `nprev` absolutely, so this is bookkeeping, not data.
    pub fn rewind(&mut self, nprev: usize) {
        self.nprev = nprev;
    }

    /// Probe-only: decode tile-row `mt` of an NVFP4 tensor via the trunk's OWN
    /// `dequant_fp4_tiled_b` (fp4_tiled_at over the repacked tiles), rows [0,16) x K.
    pub fn dequant_probe(&self, mt: usize, w: u64, sc: u64, gs: u64, k: usize) -> Result<Vec<f32>> {
        let out = self.dev.alloc_zeros::<bf16>(16 * k).context("dequant probe alloc")?;
        let base = (mt * (k / 16) * 128) as u64;
        let sbase = (mt * (k / 16) * 16) as u64;
        klaunch!(self, "dequant_fp4_tiled_b", grid((16 * k + 255) / 256), (256, 1, 1), 0,
            (d(&out), w + base, sc + sbase, gs, 16i32, k as i32));
        self.dev.synchronize()?;
        let o: Vec<bf16> = self.dev.dtoh_sync_copy(&out)?;
        Ok(o.iter().map(|x| x.to_f32()).collect())
    }

    /// The ring-vs-linear control pair (layer 0): (ring reference, S3F linear) attn outputs.
    /// Requires the last round to have run with `ctl_dual_write`.
    pub fn dump_ctl_pair(&self) -> Result<(Vec<f32>, Vec<f32>)> {
        let r: Vec<bf16> = self.dev.dtoh_sync_copy(&self.ctl_attn_ref)?;
        let l: Vec<bf16> = self.dev.dtoh_sync_copy(&self.ctl_attn)?;
        Ok((r.iter().map(|x| x.to_f32()).collect(), l.iter().map(|x| x.to_f32()).collect()))
    }

    /// The last chunk's k/v scratch AFTER inject (layer N-1's k post-rope / raw v),
    /// `[8, nkv*hd]` row-major — the probe's k-path gate.
    pub fn dump_kvc(&self) -> Result<(Vec<f32>, Vec<f32>)> {
        let k: Vec<bf16> = self.dev.dtoh_sync_copy(&self.kc)?;
        let v: Vec<bf16> = self.dev.dtoh_sync_copy(&self.vc)?;
        Ok((k.iter().map(|x| x.to_f32()).collect(), v.iter().map(|x| x.to_f32()).collect()))
    }

    /// Probe-only: rewind the committed position (the perf loop's steady-state cycle).
    pub fn rollback_nprev(&mut self, n: usize) {
        assert!(n <= self.nprev);
        self.nprev = n;
    }

    /// The committed th rows `[nprev, 5120]` (bf16 values as f32; probe gate).
    pub fn dump_th(&self) -> Result<Vec<f32>> {
        let t: Vec<bf16> = self.dev.dtoh_sync_copy(&self.th)?;
        Ok(t.iter().map(|x| x.to_f32()).collect())
    }
    pub fn max_c(&self) -> usize { self.max_c }

}
