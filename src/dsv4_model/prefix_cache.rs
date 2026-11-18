//! Prefix-cached forwards of `Dsv4GpuModel` (128-aligned checkpoints in a small LRU) and the `PrefixCache` itself.
//! Siblings: `forward.rs` (the plain forwards these wrap); `mod.rs` has the map.

use super::*;

impl Dsv4GpuModel {
    /// R2.3 prefix-cache forward: carry model state across chat turns via snapshot/restore at the
    /// conversation prefix (before the generation priming). Turn 2+ forwards only the delta (the
    /// new tokens), not the full conversation — bitwise-identical to a full re-prefill (the
    /// §12.B.5 chunked-prefill proof extends across request boundaries: the recurrent state
    /// carries in self.states, and a request boundary IS a chunk boundary).
    ///
    /// The split point is the conversation prefix length (prompt_len - priming_len), padded DOWN
    /// to a 128-aligned position (compressor block alignment). The "prefix" is prefill+snapshot'd;
    /// the "tail" (non-aligned remainder + priming, ≤ 129 tokens) is forwarded after the snapshot.
    /// On turn 2+, the longest matching snapshot is restored, the delta (prefix growth + new
    /// tail) is forwarded. Item 2.3 (session 9): the single snapshot became a small LRU of
    /// 128-aligned checkpoints — see [`PrefixCache`] and docs/DSV4_PREFIX_CACHE_DESIGN.md.
    ///
    /// Caller owns a `cache` (per server session). `priming_len` = the generation priming token
    /// count (ASSISTANT_SP + THINKING_START/END).
    /// Returns the last-row fp32 logits [vocab] for the first decode token.
    pub fn forward_prefix_cached(
        &mut self,
        ids: &[i32],
        priming_len: usize,
        cache: &mut PrefixCache,
    ) -> Result<S> {
        let (x, tail_len, _mh) = self.forward_prefix_cached_core(ids, priming_len, cache, false)?;
        let (_c, logits) = self.forward_head(&x, tail_len)?;
        Ok(logits)
    }

    /// R3.2 (L2): prefix-cached prefill + next token via the vocab-parallel maxloc head
    /// (`None` when TP is not attached — the caller uses `forward_prefix_cached` + argmax).
    pub fn forward_prefix_cached_next(
        &mut self,
        ids: &[i32],
        priming_len: usize,
        cache: &mut PrefixCache,
    ) -> Result<Option<u32>> {
        let (x, tail_len, _mh) = self.forward_prefix_cached_core(ids, priming_len, cache, false)?;
        self.forward_head_next(&x, tail_len)
    }

    /// Item 3.4 (DSpark-in-server): prefix-cached prefill that ALSO captures the per-position
    /// main_hidden (h.mean(streams) @ layers 40/41/42) for the DSpark draft ring warm — the
    /// cache-entry prefix's hidden means are stored in the entry (see [`PrefixCacheEntry`]) and
    /// restored on a hit, so the returned buffer covers ALL `s` positions either way. Bitwise
    /// identical to a full `forward_capture_main` re-prefill (the R2.3/§12.B.5 proof — the
    /// chunked forward sequence is the same; only the checkpoint/restore is added). Greedy
    /// callers use `forward_prefix_cached` (no capture cost). Returns (last-row logits, main_hidden).
    pub fn forward_prefix_cached_capture_main(
        &mut self,
        ids: &[i32],
        priming_len: usize,
        cache: &mut PrefixCache,
    ) -> Result<(S, Option<B>)> {
        let (x, tail_len, main_hidden) = self.forward_prefix_cached_core(ids, priming_len, cache, true)?;
        let (_c, logits) = self.forward_head(&x, tail_len)?;
        Ok((logits, main_hidden))
    }

    /// The shared prefix-cache prefill body (R2.3 + item 2.3): restore-or-recompute the 128-aligned
    /// conversation prefix, then forward the tail. Returns the tail streams [tail, hc*dim] bf16
    /// + the tail length; the caller runs a trunk-top variant (`forward_head` for full logits,
    /// `forward_head_next` for the R3.2 maxloc head) on it. `capture_main` additionally captures
    /// the full-prompt main_hidden into the returned `Option` (DSpark server path).
    ///
    /// Item 2.3 (session 9): longest-prefix match over the LRU at ANY 128-aligned boundary.
    /// Checkpoint policy: cold prefill snapshots at each PREFILL_CHUNK boundary (4096, itself
    /// 128-aligned) + the final aligned boundary; growth/delta forwards snapshot at EVERY
    /// 128-boundary (1–3 per turn, ~0.4 ms each). All forward/snapshot sequences are the same
    /// chunked-prefill code path the R2.3 proof covers → bitwise-identical to a full re-prefill.
    fn forward_prefix_cached_core(
        &mut self,
        ids: &[i32],
        priming_len: usize,
        cache: &mut PrefixCache,
        capture_main: bool,
    ) -> Result<(B, usize, Option<B>)> {
        let s = ids.len();
        anyhow::ensure!(s > priming_len, "forward_prefix_cached: prompt {s} ≤ priming {priming_len}");
        let conv_prefix_len = s - priming_len;
        let aligned_len = (conv_prefix_len / 128) * 128;

        // Item 3.4: a full-prompt main_hidden capture buffer [s, 3*dim] — filled per chunk by
        // forward_chunk_capture (greedy callers pass capture_main=false → None, zero overhead).
        let three_d = 3 * self.cfg.dim;
        let targets = &self.cfg.dspark_target_layer_ids;
        let have_all = targets.iter().all(|&t| t < self.layers.len());
        let mut full_mh: Option<B> = if capture_main && have_all {
            Some(self.dev.alloc_zeros::<bf16>(s * three_d)?)
        } else {
            None
        };
        // A MISS with aligned_len==0 (very short prompt < 128 tokens) cannot take a snapshot —
        // the lookup itself excludes it (entries require len > 0, len <= aligned_len == 0), so
        // no stale entry can falsely match a future request. A HIT entry WITHOUT stored
        // main_hidden (a greedy-process entry — can't happen in a DSpark process, but be
        // defensive) is treated as a miss: the warm needs the prefix's hidden means.
        let hit = cache.lookup(ids, aligned_len)
            .filter(|e| !capture_main || e.main_hidden.is_some());

        if hit.is_none() {
            // Full re-prefill: reset, clear stale cache, prefill the 128-aligned prefix with
            // checkpoints, forward the tail. (aligned_len==0 leaves the cache untouched — the
            // lookup can never match it.)
            self.reset_states()?;
            if aligned_len > 0 {
                // R2.3 fix (RUN-4 crash): the aligned prefix can EXCEED s_max (any prompt > 4226
                // tok — the harness's 6148 row SIGSEGV'd both ranks). forward_streams processes
                // ≤ s_max rows per call — the prefix MUST go through the §12.B.5 chunked path
                // (bitwise-identical to one-shot, so the snapshot is unchanged). Checkpoint at
                // each PREFILL_CHUNK boundary (128-aligned) + the final boundary — the coarse
                // cross-conversation reuse points.
                let mut c0 = 0usize;
                while c0 < aligned_len {
                    let cs = PREFILL_CHUNK.min(aligned_len - c0);
                    let _x = self.forward_chunk_capture(&ids[c0..c0 + cs], c0, full_mh.as_mut(), c0)?;
                    c0 += cs;
                    let mh = prefix_mh_clone(&self.dev, &self.rt.stream, full_mh.as_ref(), c0, three_d)?;
                    cache.insert(c0, &ids[..c0], self.snapshot_verify_state()?, mh);
                }
            }
            // Forward the tail (non-aligned remainder + priming) and get logits.
            let tail = &ids[aligned_len..];
            let x = self.forward_chunk_capture(tail, aligned_len, full_mh.as_mut(), aligned_len)?;
            Ok((x, tail.len(), full_mh))
        } else {
            // Prefix cache HIT: restore the longest match, forward the delta (prefix growth) with
            // a checkpoint at EVERY 128-boundary, forward tail.
            let entry = hit.expect("checked above");
            self.restore_verify_state(&entry.snap)?;
            // Item 3.4: restore the hit prefix's main_hidden rows into the capture buffer (the
            // warm needs ALL positions; the delta/tail rows are freshly captured below).
            if let (Some(full), Some(emh)) = (full_mh.as_mut(), entry.main_hidden.as_ref()) {
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(
                        *full.device_ptr(), *emh.device_ptr(), entry.len * three_d * 2, self.rt.stream.stream,
                    ).map_err(|e| anyhow!("prefix main_hidden restore d2d: {e}"))?;
                }
            }
            let start = entry.len;
            if aligned_len > start {
                // Forward the prefix growth (128-aligned start and end → 128-token slices land
                // on compressor-block boundaries; the checkpoint after each slice makes every
                // aligned position a future hit point). A >4K-token turn is legal — the slices
                // ride the same start_pos chain as the chunked-prefill path.
                let mut g0 = start;
                while g0 < aligned_len {
                    let cs = 128.min(aligned_len - g0);
                    let _x = self.forward_chunk_capture(&ids[g0..g0 + cs], g0, full_mh.as_mut(), g0)?;
                    g0 += cs;
                    let mh = prefix_mh_clone(&self.dev, &self.rt.stream, full_mh.as_ref(), g0, three_d)?;
                    cache.insert(g0, &ids[..g0], self.snapshot_verify_state()?, mh);
                }
            }
            // Forward the tail + priming and get logits.
            let tail = &ids[aligned_len..];
            let x = self.forward_chunk_capture(tail, aligned_len, full_mh.as_mut(), aligned_len)?;
            Ok((x, tail.len(), full_mh))
        }
    }

    /// One prefill chunk forward with OPTIONAL main_hidden capture (item 3.4): rows
    /// `[row0..row0+cs]` of the caller's `[s, 3*dim]` buffer. With `full_mh: None` this is the
    /// exact `forward_streams` sequence (bitwise); with capture it mirrors
    /// `forward_capture_main_chunked`'s per-chunk loop (h.mean(streams) @ 40/41/42 per chunk,
    /// d2d'd into the caller's buffer).
    fn forward_chunk_capture(
        &mut self,
        ids_chunk: &[i32],
        start_pos: usize,
        full_mh: Option<&mut B>,
        row0: usize,
    ) -> Result<B> {
        let s = ids_chunk.len();
        let ids_dev = self.dev.htod_sync_copy(ids_chunk)?;
        let mut x = self.embed_tokens(&ids_dev, s)?;
        if let Some(full) = full_mh {
            let targets = &self.cfg.dspark_target_layer_ids;
            let have_all = targets.iter().all(|&t| t < self.layers.len());
            let three_d = 3 * self.cfg.dim;
            let mut caps: [Option<B>; 3] = [None, None, None];
            for i in 0..self.layers.len() {
                let o = self
                    .rt
                    .block_forward::<B, S, CudaSlice<i32>, CudaSlice<u8>, CudaSlice<u32>>(&self.layers[i], &mut self.states[i], &mut self.scratch, &x, s, start_pos, &ids_dev, &self.cfg)
                    .with_context(|| format!("block_forward layer {i} (prefix capture @{start_pos})"))?;
                x = o.y;
                if have_all {
                    for (k, &t) in targets.iter().enumerate() {
                        if i == t { caps[k] = Some(x.clone()); }
                    }
                }
            }
            if have_all {
                let cmh = self.rt.compute_main_hidden(
                    caps[0].as_ref().unwrap(), caps[1].as_ref().unwrap(), caps[2].as_ref().unwrap(),
                    s, &self.cfg,
                )?;
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(
                        (*full.device_ptr()) + (row0 * three_d * 2) as u64,
                        *cmh.device_ptr(), s * three_d * 2, self.rt.stream.stream,
                    ).map_err(|e| anyhow!("prefix main_hidden dtod @{row0}: {e}"))?;
                }
            }
        } else {
            for i in 0..self.layers.len() {
                let o = self
                    .rt
                    .block_forward::<B, S, CudaSlice<i32>, CudaSlice<u8>, CudaSlice<u32>>(&self.layers[i], &mut self.states[i], &mut self.scratch, &x, s, start_pos, &ids_dev, &self.cfg)
                    .with_context(|| format!("block_forward layer {i} (prefix @{start_pos})"))?;
                x = o.y;
            }
        }
        Ok(x)
    }
}

/// Item 3.4: d2d clone of the captured main_hidden rows [0..len] for a prefix-cache entry (a
/// later HIT restores them for the DSpark draft-ring warm). `None` when the capture is off
/// (greedy-process entries store no hidden means). A free helper so the caller's borrows of
/// `self` (forward/snapshot) stay disjoint from the entry write.
fn prefix_mh_clone(
    dev: &Arc<CudaDevice>,
    stream: &cudarc::driver::CudaStream,
    full_mh: Option<&B>,
    len: usize,
    three_d: usize,
) -> Result<Option<B>> {
    match full_mh {
        None => Ok(None),
        Some(fm) => {
            let c = dev.alloc_zeros::<bf16>(len * three_d)?;
            unsafe {
                cudarc::driver::result::memcpy_dtod_async(
                    *c.device_ptr(), *fm.device_ptr(), len * three_d * 2, stream.stream,
                ).map_err(|e| anyhow!("prefix main_hidden entry d2d: {e}"))?;
            }
            Ok(Some(c))
        }
    }
}

/// Item 2.3 prefix cache: a small LRU of 128-aligned exact-prefix checkpoints (docs/
/// DSV4_PREFIX_CACHE_DESIGN.md). Each entry is a full per-layer attention-state snapshot
/// ([`LayerVerifySnap`]) at an aligned prefix length, keyed by the EXACT prefix tokens
/// (the u64 hash is a fast-rejection prefilter only — the authoritative key is `ids`).
/// Eviction: LRU at capacity (K=8 → ~840 MB @ SEQ 8192, ~0.8% of the working set).
pub struct PrefixCacheEntry {
    pub len: usize,        // aligned prefix length (multiple of 128, > 0)
    pub hash: u64,         // FNV-1a over ids[..len] — rejection prefilter, never the authority
    pub ids: Vec<i32>,     // exact prefix tokens — the authoritative key
    pub snap: Vec<LayerVerifySnap>,
    /// Item 3.4 (DSpark-in-server): the prefix's main_hidden rows [len, 3*dim] bf16 — captured
    /// during a DSpark-process prefill so a later HIT can re-warm the draft ring without a
    /// re-forward. Greedy-process entries store None (zero capture cost). ~12 KB/token.
    pub main_hidden: Option<B>,
    pub last_use: u64,
}

pub struct PrefixCache {
    pub entries: Vec<PrefixCacheEntry>,
    pub cap: usize,
    tick: u64,
}

impl PrefixCache {
    pub fn new(cap: usize) -> Self {
        PrefixCache { entries: Vec::new(), cap: cap.max(1), tick: 0 }
    }

    fn fnv1a(ids: &[i32]) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for &t in ids {
            for b in t.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        h
    }

    /// Longest-prefix match: the entry with the greatest `len <= aligned_len` whose exact
    /// token prefix equals `ids[..len]`. O(cap × len) memcmp-class — a few µs on a miss.
    pub fn lookup(&self, ids: &[i32], aligned_len: usize) -> Option<&PrefixCacheEntry> {
        let mut best: Option<&PrefixCacheEntry> = None;
        for e in &self.entries {
            if e.len == 0 || e.len > aligned_len {
                continue;
            }
            if e.hash != Self::fnv1a(&ids[..e.len]) {
                continue;
            }
            if ids[..e.len] == e.ids[..] && best.map_or(true, |b| e.len > b.len) {
                best = Some(e);
            }
        }
        best
    }

    /// Insert (or refresh, replacing any same-`len` entry — a re-checkpoint of the same
    /// boundary is value-identical, the LRU tick is what changes) + LRU-evict at capacity.
    /// `main_hidden` = the entry's captured prefix hidden means (None on the greedy path).
    pub fn insert(&mut self, len: usize, ids: &[i32], snap: Vec<LayerVerifySnap>, main_hidden: Option<B>) {
        self.tick = self.tick.wrapping_add(1);
        if let Some(existing) = self.entries.iter_mut().find(|e| e.len == len) {
            existing.hash = Self::fnv1a(ids);
            existing.ids.clear();
            existing.ids.extend_from_slice(ids);
            existing.snap = snap;
            existing.main_hidden = main_hidden;
            existing.last_use = self.tick;
            return;
        }
        self.entries.push(PrefixCacheEntry {
            len,
            hash: Self::fnv1a(ids),
            ids: ids.to_vec(),
            snap,
            main_hidden,
            last_use: self.tick,
        });
        while self.entries.len() > self.cap {
            let oldest = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_use)
                .map(|(i, _)| i)
                .expect("non-empty");
            self.entries.remove(oldest);
        }
    }

    /// Touch an entry after a successful hit (LRU recency).
    pub fn touch(&mut self, len: usize) {
        self.tick = self.tick.wrapping_add(1);
        if let Some(e) = self.entries.iter_mut().find(|e| e.len == len) {
            e.last_use = self.tick;
        }
    }
}

