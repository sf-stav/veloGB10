//! `impl FwdModel` methods that act on per-slot state: sampling and penalty setters, reset / snapshot /
//! restore, the prefix pool (pool_*), KV / state spill and `copy_slot_state`, plus small accessors.
//! See `mod.rs` for the file map.

use super::*;

impl FwdModel {
    /// Host copy of the full logits scratch [W rows x vocab] - sampled serving lanes read
    /// their row from here (the device argmax path skips it).
    /// API-parity G2: stage per-row sampler params for the NEXT forward_step / mtp_round
    /// (row r = scratch row r). `None` rows are greedy. Must be called before every step
    /// whose rows differ from the last staged set; an all-greedy call after an all-greedy
    /// state is free (no upload).
    pub fn set_sampling(&self, sc: &mut Scratch, rows: &[Option<SampParams>]) -> Result<()> {
        let live = rows.iter().any(|r| r.is_some());
        if !live && !sc.samp_live {
            return Ok(());
        }
        let n = sc.samp.len();
        anyhow::ensure!(rows.len() * 8 <= n, "set_sampling: {} rows > scratch width {}", rows.len(), n / 8);
        let mut buf = vec![0u32; n];
        for (r, p) in rows.iter().enumerate() {
            if let Some(p) = p {
                buf[r * 8..r * 8 + 8].copy_from_slice(&p.pack());
            }
        }
        self.dev.htod_copy_into(buf, &mut sc.samp)?;
        sc.samp_live = live;
        Ok(())
    }

    /// WP24: `set_sampling` for an MTP round plus the real-q draft sampler. `rq = None` is
    /// exactly `set_sampling` (draft mode 0, match-rule verify: today's graphs) and clears every
    /// WP24 staging bit. Mode 2 stages the sampled rows with flag 2 (xq_sample_rows_rq: rows
    /// 0..m-2 judge their draft under min(1, p/q'), the bonus row samples p) and the draft params;
    /// mode 1 (--wp24-dump) keeps the match-rule rows and only adds the top-32 lists. The modes
    /// ride the draft / verify graph keys (rq_draft_key / rq_verify_key).
    pub fn set_spec_sampling(&self, sc: &mut Scratch, rows: &[Option<SampParams>],
                             rq: Option<RqStage>) -> Result<()> {
        let Some(rq) = rq else {
            sc.rq_mode = 0;
            sc.rq_live = false;
            sc.rq_dctr = 0;
            return self.set_sampling(sc, rows);
        };
        anyhow::ensure!(rq.mode == 1 || rq.mode == 2, "set_spec_sampling: mode {} (1 dump, 2 real-q)", rq.mode);
        anyhow::ensure!(rq.tau.is_finite() && rq.tau > 0.0, "set_spec_sampling: draft temperature {}", rq.tau);
        if rq.mode == 2 {
            let n = sc.samp.len();
            anyhow::ensure!(rows.len() * 8 <= n, "set_spec_sampling: {} rows > scratch width {}", rows.len(), n / 8);
            let mut buf = vec![0u32; n];
            for (r, p) in rows.iter().enumerate() {
                if let Some(p) = p {
                    // flag 2 = a real-q row (xq_sample_rows treats any non-zero flag as sampled)
                    buf[r * 8..r * 8 + 8].copy_from_slice(&p.pack_flag(2));
                }
            }
            self.dev.htod_copy_into(buf, &mut sc.samp)?;
            sc.samp_live = rows.iter().any(|r| r.is_some());
        } else {
            self.set_sampling(sc, rows)?;
        }
        let d = rq.pack();
        if d != sc.rq_dsamp_host {
            self.dev.htod_copy_into(d.to_vec(), &mut sc.rq_dsamp)?;
            sc.rq_dsamp_host = d;
        }
        sc.rq_mode = rq.mode;
        sc.rq_live = rq.mode == 2;
        sc.rq_dctr = rq.dctr as i32;
        Ok(())
    }

    /// WP15: stage per-row penalty params for the NEXT forward_step / mtp_round (row r = scratch
    /// row r, same contract as set_sampling). All-None after an all-None state is free.
    pub fn set_penalties(&self, sc: &mut Scratch, rows: &[Option<PenParams>]) -> Result<()> {
        let live = rows.iter().any(|r| r.is_some());
        if !live && !sc.pen_live {
            return Ok(());
        }
        let n = sc.pen.len();
        anyhow::ensure!(rows.len() * 8 <= n, "set_penalties: {} rows > scratch width {}", rows.len(), n / 8);
        let mut buf = vec![0u32; n];
        for (r, p) in rows.iter().enumerate() {
            if let Some(p) = p {
                buf[r * 8..r * 8 + 8].copy_from_slice(&p.pack());
            }
        }
        self.dev.htod_copy_into(buf, &mut sc.pen)?;
        sc.pen_live = live;
        sc.pen_mono = rows.iter().flatten().all(|p| p.monotone());
        Ok(())
    }

    /// W2-EXL3: stage the JSON-schema masks for the NEXT forward_step: row r of the step keeps
    /// only the tokens set in `rows[r]` (a `vmask_words(V)`-word little-endian bitset; shorter
    /// bitsets are zero-padded = the missing ids are disallowed); `None` rows are untouched.
    /// All-`None` stages nothing. One-shot: forward_step consumes the staging.
    pub fn set_vmask(&self, sc: &mut Scratch, rows: &[Option<&[u32]>]) -> Result<()> {
        let live = rows.iter().any(|r| r.is_some());
        if !live {
            sc.vmask_live = false;
            return Ok(());
        }
        let nw = vmask_words(self.cfg.vocab_size);
        let w = sc.vmask_flags.len();
        anyhow::ensure!(rows.len() <= w, "set_vmask: {} rows > scratch width {}", rows.len(), w);
        let mut words = vec![0u32; sc.vmask.len()];
        let mut flags = vec![0i32; w];
        for (r, bits) in rows.iter().enumerate() {
            if let Some(bits) = bits {
                let n = bits.len().min(nw);
                words[r * nw..r * nw + n].copy_from_slice(&bits[..n]);
                flags[r] = 1;
            }
        }
        self.dev.htod_copy_into(words, &mut sc.vmask)?;
        self.dev.htod_copy_into(flags, &mut sc.vmask_flags)?;
        sc.vmask_live = true;
        Ok(())
    }

    /// WP15: seed `slot`'s penalty ring with the committed prompt positions 0..toks.len() (the
    /// chunk-prefilled prompt[..plen-1]; the seam step commits the last prompt token itself).
    /// Only the tail the largest window can reach is written.
    pub fn pen_seed(&self, sc: &mut Scratch, slot: usize, toks: &[u32]) -> Result<()> {
        anyhow::ensure!((slot + 1) * PEN_RING <= sc.pen_hist.len(),
                        "pen_seed: slot {slot} outside the scratch penalty ring");
        let mut ring = vec![0i32; PEN_RING];
        let from = toks.len().saturating_sub(PEN_RING - 64);
        for (p, &t) in toks.iter().enumerate().skip(from) {
            ring[p & (PEN_RING - 1)] = t as i32;
        }
        let mut dst = sc.pen_hist.slice_mut(slot * PEN_RING..(slot + 1) * PEN_RING);
        self.dev.htod_sync_copy_into(&ring, &mut dst)?;
        Ok(())
    }

    pub fn logits_host(&self, sc: &Scratch) -> Result<Vec<u16>> {
        Ok(self.dtoh(&sc.logits)?)
    }

    /// TP-C (G-T1-b graphed): FNV-64 digests of the replicated outputs the last step / round left
    /// in the scratch — each of the first `rows` logits rows, the whole residual (the tap class)
    /// and the whole taps window — to be compared across ranks. Syncs + dtoh: harness only.
    pub fn tp_ident_digests(&self, sc: &Scratch, rows: usize) -> Result<Vec<u64>> {
        self.dsync()?;
        let fnv = |it: &mut dyn Iterator<Item = u32>| -> u64 {
            let mut h = 0xcbf2_9ce4_8422_2325u64;
            for w in it { h ^= w as u64; h = h.wrapping_mul(0x0000_0100_0000_01b3); }
            h
        };
        let v = self.cfg.vocab_size;
        // TP-H #4: under the vocab-parallel greedy tail the logits are per-rank SHARDS (not
        // replicated); the replicated output of the tail is the merged key row (value + global id)
        // and the ids — digest those instead.
        let mut out: Vec<u64> = if self.vp_greedy(sc, rows) {
            let ks = self.dtoh(&sc.vp_keys)?;
            let ids = self.dtoh(&sc.argmax)?;
            (0..rows.min(16)).map(|r| fnv(&mut [ks[r] as u32, (ks[r] >> 32) as u32, ids[r] as u32].into_iter())).collect()
        } else {
            let lg = self.dtoh(&sc.logits)?;
            (0..rows.min(lg.len() / v))
                .map(|r| fnv(&mut lg[r * v..(r + 1) * v].iter().map(|&x| x as u32))).collect()
        };
        let rs = self.dtoh(&sc.resid)?;
        out.push(fnv(&mut rs.iter().map(|x| x.to_bits())));
        let tk = self.dtoh(&sc.taps_keep)?;
        out.push(fnv(&mut tk.iter().map(|x| x.to_bits())));
        // MTPAGREE1 (diagnostic): the MTP head's own state next to the trunk's — its collapsed hidden
        // (sc.x, the lm-head input of the last draft pass), the DHEAD-rotated copy (sc.chain_xh) and
        // the head's KV rows of slot 0, positions 0..512 of every K|V plane (where the head attends).
        // Labels in exl3_serve::tp_round: m = resid, m+1 = taps, m+2 = x, m+3 = chain_xh, m+4 = head_kv.
        let xs = self.dtoh(&sc.x)?;
        out.push(fnv(&mut xs.iter().map(|&x| x as u32)));
        let cx = self.dtoh(&sc.chain_xh)?;
        out.push(fnv(&mut cx.iter().map(|&x| x as u32)));
        if let Some(head) = self.mtp.as_ref() {
            let nkv = self.cfg.num_kv_heads;
            let rb = kv_rowbytes(self.kv_fmt, self.cfg.head_dim);
            let npos = self.max_pos.min(512);
            let mut h = 0xcbf2_9ce4_8422_2325u64;
            for plane in 0..2 * nkv {
                let start = (plane * self.max_pos) * rb; // slot 0
                let mut bytes = vec![0u8; npos * rb];
                self.dtoh_into(&head.kv.slice(start..start + npos * rb), &mut bytes)?;
                for b in bytes { h ^= b as u64; h = h.wrapping_mul(0x0000_0100_0000_01b3); }
            }
            out.push(h);
        } else { out.push(0); }
        Ok(out)
    }

    pub fn dev(&self) -> &Arc<CudaDevice> { &self.dev }
    /// S-A3-e serve fix: zero this slot's conv + GDN recurrent state rows and the host
    /// PLE n-gram ring. Finished lanes left state behind and the next request on the slot
    /// inherited it (request 1 exact, later requests drift). k_cache needs no reset
    /// (write-before-read per position).
    pub fn reset_slot(&self, slot: usize) -> Result<()> {
        self.reset_slot_ex(slot, false)
    }

    /// PFX1 (b): `reset_slot` for a slot whose next use is a CHUNKED prefill from position 0
    /// (the serve's fresh-prompt admit). With the head fill on (--exl3-head-fill != 0) the
    /// slot's head-KV memset (1.07 GB f32 / 0.28 GB q8 at a 262K window) is skipped: every
    /// head-KV row a draft pass can READ is written by this request before it is read —
    ///   rows 0..plen-2   head_fill_rows (every prefill chunk, the full row incl. q8 scales),
    ///   row  plen-1      the seam prime (mtp_seam_prime: head_kv_rows, or head_prime),
    ///   row  p (round)   draft pass 0 writes (xq_attn_prep) before its attention, or the
    ///                    control round's mtp_plain_prime; rows p+1.. the chain's own passes;
    ///   accepted rows    rewritten by the re-prime (device path: rows < a only);
    /// and every head-attention kernel reads positions 0..=p only (xq_attn_dense_splitk4 t < p+1,
    /// xq_attn_softmax / xq_attn_scan t <= p, QSA selections <= p). Stale rows > p are never read.
    /// The one-time zero at load stays. Bench/probe callers keep `reset_slot` (their token-by-
    /// token prefill has no head fill), as does the boot warm-up.
    pub fn reset_slot_for_prefill(&self, slot: usize) -> Result<()> {
        self.reset_slot_ex(slot, pfx1_on(PFX1_HEADKV) && head_fill_on())
    }

    fn reset_slot_ex(&self, slot: usize, skip_head_kv: bool) -> Result<()> {
        let cfg = &self.tc; // TP-B: the trunk's per-rank geometry
        let per_conv = (cfg.key_dim() * 2 + cfg.value_dim()) * cfg.conv_kernel;
        let per_s = cfg.lin_num_v_heads * cfg.lin_k_dim * cfg.lin_v_dim;
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        let grid = |n: usize| (((n + 255) / 256) as u32, 1, 1);
        for cs in &self.conv_state {
            let n = per_conv as i64;
            xqlaunch!(l, "xq_memset_f32", grid(per_conv), (256, 1, 1), 0,
                      (cs, (slot as i64) * n, n))?;
        }
        for ss in &self.s_state {
            let n = per_s as i64;
            xqlaunch!(l, "xq_memset_f32", grid(per_s), (256, 1, 1), 0,
                      (ss, (slot as i64) * n, n))?;
        }
        // S-A3-f-d: the M1 head's own KV must reset with the slot — the chain-verify
        // draft steps attend the head window [0..pos], and stale entries from the
        // previous request on this slot would poison the draft history (harmless to
        // losslessness — verify rejects — but they drift acceptance round to round).
        if let Some(head) = &self.mtp {
            let per = (head.kv.len() / self.width / 4) as i64; // f32 words (row bytes are % 16)
            if !skip_head_kv {
                xqlaunch!(l, "xq_memset_f32", grid(per as usize), (256, 1, 1), 0,
                          (&head.kv, (slot as i64) * per, per))?;
            } else if pfx1_headkv_poison() {
                // PFX1 (b) gate: NaN-poison the slot's head KV instead of zeroing it — 0x7FC0 is a
                // NaN as f16 (q8 scales), as f32 (0x7FC07FC0) and in the fp8 value/scale bytes, so a
                // draft pass that ever READ a row this request did not write would go NaN and move
                // drafted/accepted. Hashes + drafted/accepted identical to --pfx1-off=headkv
                // prove the invariant on the served path.
                let n = per * 2; // u16 elements
                xqlaunch!(l, "xq_memset_u16", grid(n as usize), (256, 1, 1), 0,
                          (*head.kv.device_ptr() as u64 + (slot as u64) * (per as u64) * 4, n, 0x7FC0i32))?;
            }
        }
        // S-A3-e FIX: reset ONLY the admitted slot's ring pair - the whole-ring reset wiped
        // the state of lanes already decoding in other slots.
        {
            let mut h = self.ple_hist.lock().unwrap();
            h[slot * 2] = self.ple_eos;
            h[slot * 2 + 1] = self.ple_eos;
        }
        // PLE device-side conv ring ([W*10240*9] f16, rolled per step, SLOT-indexed as of
        // this build): zero only this slot's row. Zero f16 == 0x0000; row is even-sized.
        {
            let n = (10240 * 9 / 2) as i64;
            let start = (slot as i64) * n;
            xqlaunch!(l, "xq_memset_f32", grid(46080), (256, 1, 1), 0,
                      (*self.ple_state.device_ptr() as u64, start, n))?;
        }
        Ok(())
    }

    /// Prefix cache: copy slot's recurrent state (+ the head-fill carry in `psc`) into its
    /// snapshot. Device copies on the compute stream: bit-exact.
    pub fn snapshot_slot(&self, slot: usize, psc: &PrefillScratch) -> Result<()> {
        self.copy_slot_state(slot, psc, true)
    }
    /// Prefix cache: restore slot's recurrent state (+ carry into `psc`) from its snapshot.
    pub fn restore_slot(&self, slot: usize, psc: &mut PrefillScratch) -> Result<()> {
        self.copy_slot_state(slot, psc, false)
    }
    pub fn prefix_snapshots(&self) -> bool { self.snap.is_some() }
    // ---------------------------------------------------------- LANERES2: --kv-spill
    /// The per-slot spill regions in capture order: (device ptr, byte len). Every plane here is
    /// [width × rest], so the slot stride is len()/width — no layout knowledge beyond that. The
    /// position planes (trunk KV, QSA keys/pool) are read from the LIVE buffers (rows 0..tokens
    /// of this slot are the conversation's own); the recurrent state is read from the snapshot
    /// buffers snapshot_slot filled at the end of the last prefill — and restored the same way,
    /// so the existing restore_slot path keeps working unchanged.
    fn spill_regions(&self, slot: usize, tokens: usize) -> Result<Vec<(u64, usize)>> {
        let snap = self.snap.as_ref().context("prefix snapshots not allocated (--prefix-cache off)")?;
        let w = self.width;
        anyhow::ensure!(slot < w, "spill: slot {slot} >= width {w}");
        let mut v: Vec<(u64, usize)> = Vec::new();
        // ISSUES 122: BYTE offsets and lengths (u8 KV, u16 QSA keys/pool + PLE ring, f32 GDN conv/S + head-fill carry).
        fn push<T>(v: &mut Vec<(u64, usize)>, p: &CudaSlice<T>, w: usize, slot: usize) -> usize {
            let (at, bytes) = spill_slot_region(*p.device_ptr() as u64, p.len(), std::mem::size_of::<T>(), w, slot);
            v.push((at, bytes));
            bytes
        }
        // ISSUES 123: the POSITION planes carry only the rows the conversation uses (`spill_rows`): a 4K-token snapshot
        // was the whole 262K-position lane (2.75 GB per rank at TP=4). The recurrent state planes stay whole (per slot).
        let ratio = if self.cfg.has_indexer() { self.cfg.indexer_compress_ratio } else { 1 };
        let rows = spill_rows(tokens, ratio, self.max_pos);
        let kv_rb = kv_rowbytes(self.kv_fmt, self.tc.head_dim);
        for l in &self.k_cache {
            let r = spill_row_regions(*l.device_ptr() as u64, l.len(), w, slot, self.max_pos, kv_rb, rows)?;
            anyhow::ensure!(r.len() == 2 * self.tc.num_kv_heads, "spill: KV plane has {} segments, want 2 x {}", r.len(), self.tc.num_kv_heads);
            v.extend(r);
        }
        let hdx_b = self.cfg.indexer_head_dim * 2; // bf16 rows
        for l in &self.qsa_keys {
            if l.len() == 0 { continue; }
            v.extend(spill_row_regions(*l.device_ptr() as u64, l.len() * 2, w, slot, self.max_pos, hdx_b, rows)?);
        }
        let nblk = qsa_nblk_stride(self.max_pos, ratio); // issue #41: the kernels' lane stride (was ceil)
        for l in &self.qsa_pool {
            if l.len() == 0 { continue; }
            v.extend(spill_row_regions(*l.device_ptr() as u64, l.len() * 2, w, slot, nblk, hdx_b, rows.div_ceil(ratio).min(nblk))?);
        }
        // PFXPOOL: the draft head's own positional planes (KV + raw / pooled QSA keys). A lane's own prefix hit keeps
        // them (the head fill wrote rows 0..plen-2 during the prefill); a snapshot restored into ANOTHER lane must bring
        // them, or the draft passes attend the previous conversation's rows: lossless (verify rejects) but acceptance
        // drops for that request. Payload version 4. The head is replicated under TP (full cfg geometry).
        if let Some(head) = self.mtp.as_ref() {
            if head.kv.len() > 0 {
                let hrb = kv_rowbytes(self.kv_fmt, self.cfg.head_dim);
                let r = spill_row_regions(*head.kv.device_ptr() as u64, head.kv.len(), w, slot, self.max_pos, hrb, rows)?;
                anyhow::ensure!(r.len() == 2 * self.cfg.num_kv_heads, "spill: head KV plane has {} segments, want 2 x {}", r.len(), self.cfg.num_kv_heads);
                v.extend(r);
            }
            if let Some(k) = head.qsa_keys.as_ref() {
                v.extend(spill_row_regions(*k.device_ptr() as u64, k.len() * 2, w, slot, self.max_pos, hdx_b, rows)?);
            }
            if let Some(pl) = head.qsa_pool.as_ref() {
                v.extend(spill_row_regions(*pl.device_ptr() as u64, pl.len() * 2, w, slot, nblk, hdx_b, rows.div_ceil(ratio).min(nblk))?);
            }
        }
        for sn in &snap.conv { push(&mut v, sn, w, slot); }
        for sn in &snap.s { push(&mut v, sn, w, slot); }
        push(&mut v, &snap.ple, w, slot);
        push(&mut v, &snap.carry, w, slot);
        Ok(v)
    }

    // ---------------------------------------------------------- PFXPOOL: the prefix pool
    /// Bytes one pool entry of `tokens` cached positions takes on THIS rank (the spill regions, the draft head's planes
    /// included). A pure function of the boot geometry — identical on every rank of a symmetric TP split, which the
    /// boot agree hash checks — so the replicated pool policy can account with it.
    pub fn pool_entry_bytes(&self, tokens: usize) -> Result<u64> {
        Ok(self.spill_regions(0, tokens)?.iter().map(|(_, b)| *b as u64).sum())
    }

    /// PFXPOOL: an uninitialised device buffer of `bytes` (the pool's entry slab).
    pub fn pool_alloc(&self, bytes: usize) -> Result<CudaSlice<u8>> {
        Ok(unsafe { self.dev.alloc::<u8>(bytes.max(1)) }?)
    }

    /// PFXPOOL2: how many of `spill_regions`' entries (at its tail) are the recurrent snapshot planes.
    fn pool_n_snap(&self) -> Result<usize> {
        let snap = self.snap.as_ref().context("prefix snapshots not allocated")?;
        Ok(snap.conv.len() + snap.s.len() + 2)
    }

    /// PFXPOOL2: bytes of the recurrent snapshot planes of one slot (the part of an entry that is not positional rows;
    /// a child entry holds only this). Deterministic boot geometry, like `pool_entry_bytes`.
    pub fn pool_snap_bytes(&self) -> Result<u64> {
        let r = self.spill_regions(0, 1)?;
        let k = self.pool_n_snap()?;
        Ok(r[r.len().saturating_sub(k)..].iter().map(|(_, b)| *b as u64).sum())
    }

    /// PFXPOOL2: copy `slot`'s recurrent snapshot planes to device address `dst` (back to back, `pool_snap_bytes`
    /// long). Compute stream, no sync. Returns the host PLE ring pair.
    pub fn pool_capture_snap_to(&self, slot: usize, dst: u64) -> Result<[i32; 2]> {
        let snap = self.snap.as_ref().context("prefix snapshots not allocated")?;
        let r = self.spill_regions(slot, 1)?;
        let k = self.pool_n_snap()?;
        let mut off = 0u64;
        for (src, n) in &r[r.len().saturating_sub(k)..] {
            if *n > 0 { unsafe { cudarc::driver::result::memcpy_dtod_async(dst + off, *src, *n, self.stream.stream) }?; }
            off += *n as u64;
        }
        let h = snap.hist.lock().unwrap();
        Ok([h[slot * 2], h[slot * 2 + 1]])
    }

    /// PFXPOOL2: write recurrent snapshot planes from device address `src` (as `pool_capture_snap_to` laid them out)
    /// into `slot`'s snapshot buffers, and the PLE pair into its hist (the caller then runs `restore_slot`, as an
    /// own-lane prefix hit does). Compute stream, no sync.
    pub fn pool_restore_snap_from(&self, slot: usize, src: u64, hist: [i32; 2]) -> Result<()> {
        let snap = self.snap.as_ref().context("prefix snapshots not allocated")?;
        let r = self.spill_regions(slot, 1)?;
        let k = self.pool_n_snap()?;
        let mut off = 0u64;
        for (dst, n) in &r[r.len().saturating_sub(k)..] {
            if *n > 0 { unsafe { cudarc::driver::result::memcpy_dtod_async(*dst, src + off, *n, self.stream.stream) }?; }
            off += *n as u64;
        }
        let mut h = snap.hist.lock().unwrap();
        h[slot * 2] = hist[0];
        h[slot * 2 + 1] = hist[1];
        Ok(())
    }

    /// PFXPOOL2: device-to-device copy on the compute stream (pool slab reshaping).
    pub fn pool_copy(&self, dst: u64, src: u64, n: usize) -> Result<()> {
        if n > 0 { unsafe { cudarc::driver::result::memcpy_dtod_async(dst, src, n, self.stream.stream) }?; }
        Ok(())
    }

    /// PFXPOOL2: the positional planes of `slot`, in `spill_regions` order: (address of the slot's unit 0, bytes per
    /// unit, rows per unit, units in the plane). A unit is a row, or a pooled QSA block of `indexer_compress_ratio` rows.
    fn pool_planes(&self, slot: usize) -> Result<Vec<(u64, usize, usize, usize)>> {
        let w = self.width;
        anyhow::ensure!(slot < w, "pool: slot {slot} >= width {w}");
        let mut v = Vec::new();
        let ratio = if self.cfg.has_indexer() { self.cfg.indexer_compress_ratio.max(1) } else { 1 };
        let nblk = qsa_nblk_stride(self.max_pos, ratio); // issue #41: the kernels' lane stride (was ceil)
        let kv_rb = kv_rowbytes(self.kv_fmt, self.tc.head_dim);
        let hdx_b = self.cfg.indexer_head_dim * 2;
        let mut add = |base: u64, bytes: usize, cap: usize, rb: usize, div: usize| -> Result<()> {
            for (at, _) in spill_row_regions(base, bytes, w, slot, cap, rb, 0)? { v.push((at, rb, div, cap)); }
            Ok(())
        };
        for l in &self.k_cache { add(*l.device_ptr() as u64, l.len(), self.max_pos, kv_rb, 1)?; }
        for l in &self.qsa_keys { if l.len() > 0 { add(*l.device_ptr() as u64, l.len() * 2, self.max_pos, hdx_b, 1)?; } }
        for l in &self.qsa_pool { if l.len() > 0 { add(*l.device_ptr() as u64, l.len() * 2, nblk, hdx_b, ratio)?; } }
        if let Some(head) = self.mtp.as_ref() {
            if head.kv.len() > 0 {
                add(*head.kv.device_ptr() as u64, head.kv.len(), self.max_pos, kv_rowbytes(self.kv_fmt, self.cfg.head_dim), 1)?;
            }
            if let Some(k) = head.qsa_keys.as_ref() { add(*k.device_ptr() as u64, k.len() * 2, self.max_pos, hdx_b, 1)?; }
            if let Some(pl) = head.qsa_pool.as_ref() { add(*pl.device_ptr() as u64, pl.len() * 2, nblk, hdx_b, ratio)?; }
        }
        Ok(v)
    }

    /// PFXPOOL2: rows a pool entry of `tokens` cached positions holds (the spill tier's row count) and the row grid a
    /// shared prefix is cut on (a pooled QSA block is never split).
    pub fn pool_rows_for(&self, tokens: usize) -> usize {
        let ratio = if self.cfg.has_indexer() { self.cfg.indexer_compress_ratio } else { 1 };
        spill_rows(tokens, ratio, self.max_pos)
    }
    pub fn pool_row_grid(&self) -> usize {
        if self.cfg.has_indexer() { self.cfg.indexer_compress_ratio.max(1) } else { 1 }
    }

    /// PFXPOOL2: (first, end) unit of rows r0..r1 in a plane (r0 on the row grid).
    fn pool_units(r0: usize, r1: usize, div: usize, cap: usize) -> (usize, usize) {
        ((r0 / div).min(cap), r1.div_ceil(div).min(cap))
    }

    /// PFXPOOL2: bytes of positional rows r0..r1 over every plane (deterministic boot geometry).
    pub fn pool_rows_bytes(&self, r0: usize, r1: usize) -> Result<u64> {
        Ok(self.pool_planes(0)?.iter().map(|&(_, rb, div, cap)| {
            let (u0, u1) = Self::pool_units(r0, r1, div, cap);
            (u1.saturating_sub(u0) * rb) as u64
        }).sum())
    }

    /// PFXPOOL2: copy rows r0..r1 of every positional plane of `slot` to `dst`, plane after plane. Compute stream.
    /// Returns the bytes written.
    pub fn pool_capture_rows(&self, slot: usize, r0: usize, r1: usize, dst: u64) -> Result<u64> {
        let mut off = 0u64;
        for (at, rb, div, cap) in self.pool_planes(slot)? {
            let (u0, u1) = Self::pool_units(r0, r1, div, cap);
            let n = u1.saturating_sub(u0) * rb;
            if n > 0 {
                unsafe { cudarc::driver::result::memcpy_dtod_async(dst + off, at + (u0 * rb) as u64, n, self.stream.stream) }?;
            }
            off += n as u64;
        }
        Ok(off)
    }

    /// PFXPOOL2: write rows a..b of every positional plane of `slot` from an entry slab at `src` that holds rows e0..e1
    /// (laid out by `pool_capture_rows`). a and e0 are on the row grid; e0 <= a <= b <= e1. Compute stream, no sync.
    pub fn pool_restore_rows_part(&self, slot: usize, a: usize, b: usize, e0: usize, e1: usize, src: u64) -> Result<()> {
        anyhow::ensure!(e0 <= a && a <= b && b <= e1, "pool: rows {a}..{b} outside the entry's {e0}..{e1}");
        let mut off = 0u64;
        for (at, rb, div, cap) in self.pool_planes(slot)? {
            let (eu0, eu1) = Self::pool_units(e0, e1, div, cap);
            let (u0, u1) = Self::pool_units(a, b, div, cap);
            let n = u1.saturating_sub(u0) * rb;
            if n > 0 {
                anyhow::ensure!(u0 >= eu0 && u1 <= eu1, "pool: plane units {u0}..{u1} outside {eu0}..{eu1}");
                let from = src + off + ((u0 - eu0) * rb) as u64;
                unsafe { cudarc::driver::result::memcpy_dtod_async(at + (u0 * rb) as u64, from, n, self.stream.stream) }?;
            }
            off += (eu1.saturating_sub(eu0) * rb) as u64;
        }
        Ok(())
    }

    /// PFXPOOL: bytes of one WP16 checkpoint on this rank (pool entries carry copies).
    pub fn ckpt_bytes(&self) -> u64 { self.ckpt_layout().bytes() as u64 }

    /// LANERES2 (`--kv-spill`): serialize slot's prefix snapshot into `out` (GB10SPIL, payload version SPILL_PAYLOAD_VERSION):
    /// 8B magic, u32 version, u64 tokens, u64 region count, u64 payload bytes, the raw regions
    /// in spill_regions order, then the 2 host PLE ring ids. Bit-exact by construction: every
    /// region is a memcpy of the exact device words the serving path reads. TEXT-ONLY: the
    /// per-slot VIS rope rows are not spilled — an image-slot restore re-prefills (counted on
    /// /metrics); agent traces are text. D2H runs on the NULL stream through a reused pinned
    /// staging chunk (the PinnedHost pattern, WP05): the compute stream is synchronized first
    /// so the planes' writers retire, then chunked copies run (~5 GB/s on .13, ~0.1 s per
    /// 0.5-GiB snapshot). Evictions are rare and the admission round is dominated by the next
    /// conversation's prefill; the DISK write is the async part (kv_spill::Store writer thread).
    pub fn spill_capture(&self, slot: usize, tokens: usize, out: &mut Vec<u8>) -> Result<()> {
        let snap = self.snap.as_ref().context("prefix snapshots not allocated")?;
        anyhow::ensure!(tokens > 0 && tokens <= self.max_pos, "spill: tokens {tokens} out of range");
        let regions = self.spill_regions(slot, tokens)?;
        let total: usize = regions.iter().map(|(_, b)| *b).sum();
        out.clear();
        out.reserve(72 + total + 8);
        out.resize(72, 0); // LANERES3: outer 36B (kv_spill::finalize_header) + inner 36B
        // body header (GB10SPIL) that spill_restore parses — LANERES2 shipped the capture
        // without it (restores never engaged on hardware, so every restore would have
        // died "bad magic/length"; first exercised by LANERES3's TP tier).
        // order: retire ALL device work (the planes' writers run on the compute stream) before
        // the NULL-stream DtoH reads them; device-wide is stronger than stream sync and this
        // path is rare + dominated by the next prefill.
        self.dsync()?;
        const CHUNK: usize = 64 << 20;
        let mut pin = PinnedHost::new(&self.dev, CHUNK.min(total.max(16)))?;
        for (src, bytes) in &regions {
            let mut left = *bytes;
            let mut at = *src;
            while left > 0 {
                let n = left.min(CHUNK);
                pin.dtoh(0, at, n)?;
                self.dsync()?;
                out.extend_from_slice(pin.as_bytes(0, n));
                at += n as u64;
                left -= n;
            }
        }
        // the inner body header the restore parses (magic, ver=1, tokens, nreg, total):
        out[36..44].copy_from_slice(b"GB10SPIL");
        out[44..48].copy_from_slice(&SPILL_PAYLOAD_VERSION.to_le_bytes());
        out[48..56].copy_from_slice(&(tokens as u64).to_le_bytes());
        out[56..64].copy_from_slice(&(regions.len() as u64).to_le_bytes());
        out[64..72].copy_from_slice(&(total as u64).to_le_bytes());
        let h = snap.hist.lock().unwrap();
        out.extend_from_slice(&h[slot * 2].to_le_bytes());
        out.extend_from_slice(&h[slot * 2 + 1].to_le_bytes());
        Ok(())
    }

    /// LANERES2: restore a spill_capture payload into `slot`. The position planes go to the
    /// LIVE planes (rows 0..tokens), the recurrent snapshot into the snapshot buffers (the
    /// caller then runs restore_slot, exactly like a prefix-cache hit), the PLE ring ids into
    /// the snapshot's hist. Any mismatch (magic/version/shape — a different model or flag set
    /// captured the file) is an Err: the caller falls back to re-prefill and counts it; the
    /// server never refuses a request over a spilled snapshot.
    /// LANERES3: the parse-only half of `spill_restore` — validates a payload (magic,
    /// version, shape against THIS model's regions) without touching GPU state, so the
    /// TP tier can agree the PARSE outcome across ranks before either side restores: a
    /// payload one rank cannot parse is dropped everywhere (graceful re-prefill), never
    /// a post-agree fatal. Returns the payload's token count.
    pub fn spill_restore_check(&self, slot: usize, payload: &[u8]) -> Result<usize> {
        // payload = Store::get's payload (the OUTER file header is already stripped):
        // inner GB10SPIL header (36B), the regions, then the 8B hist tail. Both TP callers
        // pass exactly this; treating it as the whole file double-strips 36B and rejects
        // every capture (lr3dbg receipt, 2026-10-22).
        anyhow::ensure!(payload.len() >= 44, "spill: file too short");
        let bytes = payload;
        anyhow::ensure!(bytes.len() >= 44 && &bytes[0..8] == b"GB10SPIL", "spill: bad magic/length");
        let ver = u32::from_le_bytes(bytes[8..12].try_into()?);
        anyhow::ensure!(ver == SPILL_PAYLOAD_VERSION, "spill: payload version {ver}, this build reads {SPILL_PAYLOAD_VERSION} \
                        (older files: dropped, re-prefill)");
        let tokens = u64::from_le_bytes(bytes[12..20].try_into()?) as usize;
        let nreg = u64::from_le_bytes(bytes[20..28].try_into()?) as usize;
        let total = u64::from_le_bytes(bytes[28..36].try_into()?) as usize;
        anyhow::ensure!(bytes.len() == 36 + total + 8, "spill: payload length mismatch");
        anyhow::ensure!(tokens > 0 && tokens <= self.max_pos, "spill: tokens {tokens} out of range");
        let regions = self.spill_regions(slot, tokens)?;
        anyhow::ensure!(regions.len() == nreg, "spill: region count {} != {nreg}", regions.len());
        anyhow::ensure!(total == regions.iter().map(|(_, b)| *b).sum::<usize>(),
                        "spill: payload bytes {total} != this model's regions (flags/model changed since capture)");
        Ok(tokens)
    }

    pub fn spill_restore(&self, slot: usize, payload: &[u8]) -> Result<usize> {
        // payload = Store::get's payload: the capture body (its own 36B header: MAGIC, ver,
        // tokens, nreg, total), then the regions, then the 8B hist tail.
        let tokens = self.spill_restore_check(slot, payload)?;
        let bytes = payload;
        let total = u64::from_le_bytes(bytes[28..36].try_into()?) as usize;
        self.dsync()?;
        const CHUNK: usize = 64 << 20;
        let mut pin = PinnedHost::new(&self.dev, CHUNK.min(total.max(16)))?;
        let regions = self.spill_regions(slot, tokens)?; // LANERES3: re-derive (the check validated them)
        let mut off = 36usize;
        for (dst, dbytes) in &regions {
            let mut left = *dbytes;
            let mut at = *dst;
            while left > 0 {
                let n = left.min(CHUNK);
                pin.copy_from_slice(0, &bytes[off..off + n]);
                pin.htod(0, at, n)?;
                self.dsync()?;
                at += n as u64;
                left -= n;
                off += n;
            }
        }
        let snap = self.snap.as_ref().unwrap();
        let h0 = i32::from_le_bytes(bytes[off..off + 4].try_into()?);
        let h1 = i32::from_le_bytes(bytes[off + 4..off + 8].try_into()?);
        let mut h = snap.hist.lock().unwrap();
        h[slot * 2] = h0;
        h[slot * 2 + 1] = h1;
        Ok(tokens)
    }

    fn copy_slot_state(&self, slot: usize, psc: &PrefillScratch, save: bool) -> Result<()> {
        let snap = self.snap.as_ref().context("prefix snapshots not allocated")?;
        let cfg = &self.tc; // TP-B: the trunk's per-rank geometry
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        let grid = |n: usize| ((((n as u64) + 255) / 256) as u32, 1, 1);
        let per_conv = (cfg.key_dim() * 2 + cfg.value_dim()) * cfg.conv_kernel;
        let per_s = cfg.lin_num_v_heads * cfg.lin_k_dim * cfg.lin_v_dim;
        let rw = cfg.hc_count.max(1) * cfg.hidden_size;
        let mut cp = |live: u64, sn: u64, n: usize| -> Result<()> {
            let off = (slot * n) as i64;
            let (dst, src) = if save { (sn, live) } else { (live, sn) };
            xqlaunch!(l, "xq_copy_f32", grid(n), (256, 1, 1), 0, (dst, src, n as i64, off, off))?;
            Ok(())
        };
        for (lv, sv) in self.conv_state.iter().zip(snap.conv.iter()) {
            cp(*lv.device_ptr() as u64, *sv.device_ptr() as u64, per_conv)?;
        }
        for (lv, sv) in self.s_state.iter().zip(snap.s.iter()) {
            cp(*lv.device_ptr() as u64, *sv.device_ptr() as u64, per_s)?;
        }
        // PLE f16 ring row = 10240*9 halves = 46080 f32 words
        cp(*self.ple_state.device_ptr() as u64, *snap.ple.device_ptr() as u64, 10240 * 9 / 2)?;
        // head-fill carry: psc holds ONE carry (the slot being prefilled); the snapshot keeps one per slot
        {
            let (dst, src, doff, soff) = if save {
                (*snap.carry.device_ptr() as u64, *psc.hfill_carry.device_ptr() as u64, (slot * rw) as i64, 0i64)
            } else {
                (*psc.hfill_carry.device_ptr() as u64, *snap.carry.device_ptr() as u64, 0i64, (slot * rw) as i64)
            };
            if psc.hfill_carry.len() >= rw {
                xqlaunch!(l, "xq_copy_f32", grid(rw), (256, 1, 1), 0, (dst, src, rw as i64, doff, soff))?;
            }
        }
        let mut live = self.ple_hist.lock().unwrap();
        let mut sh = snap.hist.lock().unwrap();
        if save { sh[slot * 2] = live[slot * 2]; sh[slot * 2 + 1] = live[slot * 2 + 1]; }
        else { live[slot * 2] = sh[slot * 2]; live[slot * 2 + 1] = sh[slot * 2 + 1]; }
        Ok(())
    }

    /// Argmax ids of the last forward_step, read back to host (serve prefill uses this
    /// to capture the first generated token).
    pub fn argmax_host(&self, sc: &Scratch) -> Result<Vec<u32>> {
        let v = self.dtoh(&sc.argmax)?;
        Ok(v.iter().map(|&x| x as u32).collect())
    }
    pub fn max_pos(&self) -> usize { self.max_pos }
    /// Host bytes held by the RAM-resident PLE n-gram table ([ple-ram], see ple_ram_load);
    /// 0 when the table is read through the page cache (pread) or the model has no PLE. These are
    /// ordinary host Vec pages — invisible to nvidia-smi — but on GB10 they come out of the same
    /// unified pool cuMemGetInfo reports, so they are part of a load's measured footprint.
    pub fn ple_ram_bytes(&self) -> u64 {
        self.ple.as_ref().and_then(|p| p.ram.get())
            .map(|r| r.iter().map(|v| v.len() as u64).sum()).unwrap_or(0)
    }
    /// CF-P1d `--ple-ram auto`: called once the boot has allocated every lane's KV, scratch, graphs and the vision
    /// tower. Moves the n-gram table into RAM iff it fits with headroom to spare (16 GiB base +
    /// `--ple-headroom-per-lane-gb` per additional lane; default per-lane 0 = the old fixed 16 GiB
    /// — PR #17 ported as opt-in), measured now; otherwise it stays on SSD (identical output,
    /// ~1-2% decode) and the log says why. No-op for ram / ssd or a model without PLE.
    pub fn ple_promote_auto(&self) -> Result<()> {
        let Some(p) = self.ple.as_ref() else { return Ok(()) };
        if ple_mode() != "auto" || p.ram.get().is_some() { return Ok(()); }
        let per = p.rows_per_shard as usize * p.row_bytes;
        let need = per as u64 * p.shards.len() as u64;
        let avail = mem_available_bytes();
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        let headroom = ple_auto_headroom(self.width);
        if avail < need + headroom {
            crate::boot_report::notice(&format!("PLE n-gram table: stays on SSD (--ple-ram auto: after the boot {:.1} GiB are free, the table needs {:.1} GiB + {:.1} GiB headroom; \
                      output is identical, decode ~1-2% slower; fewer lanes / a shorter --max-seq-len would let it move to RAM)",
                     gib(avail), gib(need), headroom as f64 / (1u64 << 30) as f64));
            return Ok(());
        }
        let t0 = std::time::Instant::now();
        let t = match p.fx.as_ref() {
            Some(fx) => fp4l_fwd::ple_fx_ram(&p.shards, fx, p.rows_per_shard)?, // FP4L: interleaved 90-B rows
            None => ple_read_table(&p.shards, &p.shard_data_off, per)?,
        };
        let _ = p.ram.set(t);
        crate::boot_report::notice(&format!("PLE n-gram table moved to RAM: {:.1} GiB in {:.1}s (--ple-ram auto; {:.1} GiB were free after the boot)",
                 gib(need), t0.elapsed().as_secs_f32(), gib(avail)));
        Ok(())
    }

    /// PLE n-gram table residency label for receipts: "on" (RAM-resident), "pread" (page cache),
    /// or "none" (the model has no PLE).
    pub fn ple_residency(&self) -> &'static str {
        match self.ple.as_ref() {
            None => "none",
            Some(p) if p.ram.get().is_some() => "on",
            Some(_) => "pread",
        }
    }

    /// S-A3-f-g: staging size (u16 elements) for the prefill reconstruct path —
    /// the largest trunk dense matrix the wide-M arm will dequant per call.
    /// lm_head is excluded (it only ever runs at m=1 in prefill and stays trellis).
    pub fn recon_staging_elems(&self) -> usize {
        let mut mx = 0usize;
        let mut scan = |q: &Quad| mx = mx.max(q.k as usize * q.n as usize);
        for l in &self.layers {
            match &l.mixer {
                Mixer::Gdn(g) => { scan(&g.in_qkv); scan(&g.in_z); scan(&g.out_proj); }
                Mixer::Attn(a) => { scan(&a.q_proj); scan(&a.k_proj); scan(&a.v_proj); scan(&a.o_proj); }
            }
            scan(&l.moe.sh_gate);
            scan(&l.moe.sh_up);
            scan(&l.moe.sh_down);
        }
        // TP4 boot fix (2026-10-17): the MTP head is REPLICATED at every world, so its
        // full-width quads (dense-MLP sh_* at h x mi, fc_*, draft attn) exceed the trunk
        // scan's max once the trunk slices shrink at world >= 4 — the node's boot recon
        // dry-run then trips "recon staging too small: need 16777216 have 12582912"
        // (9B pack, world=4). Scan the draft head's chain quads with the same rule.
        if let Some(mtp) = &self.mtp {
            scan(&mtp.moe.sh_gate);
            scan(&mtp.moe.sh_up);
            scan(&mtp.moe.sh_down);
            scan(&mtp.fc_hidden);
            scan(&mtp.fc_embed);
            scan(&mtp.attn.q_proj); scan(&mtp.attn.k_proj); scan(&mtp.attn.v_proj); scan(&mtp.attn.o_proj);
        }
        mx
    }

}
