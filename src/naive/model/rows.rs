//! `GpuModel::forward_rows*`: host-side entry points that run row batches (chunked prefill / verify) through the step.
//! Siblings: `step.rs` (the step they drive), `helpers.rs` (`chunk_ranges`); `mod.rs` has the map.

use super::*;

impl GpuModel {
    /// M rows at consecutive positions: logits f32 [M][vl] (this rank's vocabulary slice) and the residual after every layer.
    /// T2-d: chunked prefill. The single-shot path is unchanged for `m <= MAX_PREFILL_ROWS` (one
    /// chunk == one call to the old code path), so the 2048-row behaviour is bit-for-bit what it
    /// was; a WIDER prompt now walks `MAX_PREFILL_ROWS` rows at a time instead of being refused.
    ///
    /// Why chunking is CORRECT here, not just legal:
    ///  * Naive-N0.5 is pure attention (`AttnKind::Swa` | `Dsa`) -- there is NO recurrent/GDN state
    ///    to carry, so nothing is lost at a chunk boundary (AGENTS.md 2.8's GDN rule is about a
    ///    different model family).
    ///  * `nv_rope_kv` writes the chunk's K/V into `l.kc`/`l.vc` BEFORE the chunk attends, and the
    ///    causal mask is the per-row ABSOLUTE `pos[]`, so chunk `c0` sees every earlier position.
    ///  * the taps land in ONE [layers][m][h] buffer indexed by absolute row, so the head's input
    ///    (the last layer's block) and the drafter's per-layer taps are exactly what a single-shot
    ///    step would have produced.
    ///
    /// The workspace is sized by the CHUNK, not the prompt: that is the point of chunking, and it
    /// keeps the DSA `m * max_pos` score buffer and the MoE gathered-rows buffer bounded.
    pub fn forward_rows(&self, ids: &[u32], pos: &[usize]) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        self.forward_rows_cap(ids, pos, MAX_PREFILL_ROWS)
    }

    /// `cap` FORCES the prefill chunk width (1c). The serving path chunks at `MAX_PREFILL_ROWS` and
    /// a prompt at or under that width is ONE pass — so on the default arm the chunked path never
    /// runs, and comparing it against the unchunked arm is comparing a call with ITSELF. A cap
    /// below the row count is the only way to exercise chunking on a prompt that also fits one
    /// pass. `cap == 0` asks for one pass. Clamped to `MAX_PREFILL_ROWS`: the workspace is sized for
    /// a chunk that wide and no wider allocation has ever been budgeted for.
    pub fn forward_rows_cap(&self, ids: &[u32], pos: &[usize], cap: usize) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        let m = ids.len();
        ensure!(m >= 1 && pos.len() == m, "rows {m}");
        ensure!(pos.iter().all(|&p| p < self.max_pos), "position beyond max_pos");
        let (h, v) = (self.cfg.hidden_size, self.vl);
        let ids_d = self.dev.htod_sync_copy(&ids.iter().map(|x| *x as i32).collect::<Vec<i32>>())?;
        let pos_d = self.dev.htod_sync_copy(&pos.iter().map(|x| *x as i32).collect::<Vec<i32>>())?;
        let nl = self.layers.len();

        // one chunk == the old code path exactly
        let cap = if cap == 0 { m.min(MAX_PREFILL_ROWS) } else { cap.min(MAX_PREFILL_ROWS) }.min(m);
        let ranges = chunk_ranges(m, cap);
        debug_assert!(ranges.iter().map(|&(_, n)| n).sum::<usize>() == m, "chunks must tile the prompt");
        let ws_m = ranges.iter().map(|&(_, n)| n).max().unwrap_or(1);

        // the per-layer taps: one [nl][m][h] f32 set, indexed by ABSOLUTE row so chunks interleave
        // correctly. This replaces the old per-call workspace taps (same shape, one allocation).
        let taps_d = self.dev.alloc_zeros::<f32>(nl * m * h)?;
        let mut ws = Ws::new(&self.dev, ws_bytes(&self.cfg, ws_m, self.max_pos, 0))?;
        let taps_base = dptr(&taps_d);
        for (c0, mc) in ranges {
            self.step_kernels_taps_ext(&mut ws,
                                       dptr(&ids_d) + (c0 * 4) as u64,
                                       dptr(&pos_d) + (c0 * 4) as u64,
                                       mc, taps_base, m, c0,
                                       // the slot ids are per ROW, so the chunk's rows are offset
                                       // exactly like ids/pos are. Passing the base for every chunk
                                       // makes chunk 1+ read `sids[0..]` -- i.e. chunk 1's slots --
                                       // and silently cross-contaminate the requests.
                                       self.sids + (c0 * 4) as u64)?;
            self.sync()?;
        }

        // the head on every row, in MAX_ROWS groups, off the LAST layer's block
        let x_last = taps_base + ((nl - 1) * m * h * 4) as u64;
        let mut all = Vec::with_capacity(m * v);
        for g in (0..m).step_by(MAX_ROWS) {
            let mm = (m - g).min(MAX_ROWS);
            let mark = ws.mark();
            let hb = ws.alloc::<u16>(mm * h)?;
            let lgg = self.head_logits_ws(x_last + (g * h * 4) as u64, hb, mm, &mut ws)?;
            self.sync()?;
            all.extend(dtoh::<u16>(lgg, mm * v)?.iter().map(|b| half::f16::from_bits(*b).to_f32()));
            ws.release(mark);
        }
        let taps = (0..nl).map(|li| dtoh::<f32>(taps_base + (li * m * h * 4) as u64, m * h)).collect::<Result<Vec<_>>>()?;
        Ok((all, taps))
    }

    /// M rows; the LAST row's logits when asked (None = a rank that does not need them).
    pub fn forward_rows_last(&self, ids: &[u32], pos: &[usize], want_logits: bool) -> Result<Option<Vec<f32>>> {
        let m = ids.len();
        ensure!(m >= 1 && m <= MAX_ROWS && pos.len() == m, "rows {m}");
        ensure!(pos.iter().all(|&p| p < self.max_pos), "position beyond max_pos");
        let v = self.vl;
        let ids_d = self.dev.htod_sync_copy(&ids.iter().map(|x| *x as i32).collect::<Vec<i32>>())?;
        let pos_d = self.dev.htod_sync_copy(&pos.iter().map(|x| *x as i32).collect::<Vec<i32>>())?;
        let mut ws = Ws::new(&self.dev, ws_bytes(&self.cfg, m, self.max_pos, 0))?;
        let lg = self.step_kernels(&mut ws, dptr(&ids_d), dptr(&pos_d), m, want_logits, None)?;
        self.sync()?;
        match lg {
            None => Ok(None),
            Some(p) => Ok(Some(dtoh::<u16>(p + ((m - 1) * v * 2) as u64, v)?.iter().map(|b| half::f16::from_bits(*b).to_f32()).collect())),
        }
    }
}
