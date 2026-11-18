//! `impl GpuModel`: rejection-gate, tree and lane benches (`bench_rejection_gate*`, `bench_tree_*`, `bench_lanes*`).
//! Bench harnesses, not on the serving path. Map: `mod.rs`.

use super::*;

impl GpuModel {
    /// S5F: the rejection-sampling distribution gate for a FIXED (committed, greedy draft) pair
    /// (extracted verbatim from `bench_mtp_sample` — the draft source is irrelevant to the sampler
    /// claim, so the DFlash2 gate drives this with the round's draft). See `bench_mtp_sample`'s doc
    /// for the method description and the A/B/C arm semantics.
    pub fn bench_rejection_gate(&self, pool: &mut Pool, state: &mut BatchGpuState, prompt: &[u32],
                                kv_stride: usize, a0: u32, x_draft: u32,
                                temp: f32, top_k: usize, top_p: f32, trials: usize,
                                rng_base: u64) -> MtpSampleStats {
        use crate::batch::{rng_u32, rng_uniform, splitmix64, RNG_DOM_VERIFY, RNG_DOM_ACCEPT};
        let h = self.cfg.hidden_size;
        let vocab = self.cfg.vocab_size;
        let plen = prompt.len();
        let copy_stream = self.stream.stream;

        // Verify [committed, draft] → target logits. Column 0 is the distribution the drafted
        // position is judged against; that column is what both A and B sample from.
        let (logits, residual) = self.verify_forward_core(pool, &[a0, x_draft], state, 0, kv_stride, plen, None, false);
        let logits = logits.into_b16();
        pool.release_bf16(residual, h * 2);
        let mut pristine = self.dev.alloc_zeros::<half::bf16>(vocab).unwrap();
        self.dev.memset_zeros(&mut pristine).unwrap();
        self.dev.synchronize().unwrap();
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(
                *pristine.device_ptr() as u64, *logits.device_ptr() as u64, vocab * 2, copy_stream).unwrap();
        }
        self.sync_stream();
        pool.release_bf16(logits, vocab * 2);

        // Ground truth: the nucleus distribution, computed on the host from the same logits.
        let lg: Vec<half::bf16> = self.dev.dtoh_sync_copy(&pristine).unwrap();
        let lgf: Vec<f32> = lg.iter().map(|x| x.to_f32()).collect();
        let p_analytic = nucleus_dist(&lgf, temp, top_k, top_p);
        let mut pmap = vec![0.0f64; vocab];
        for &(t, pr) in &p_analytic { pmap[t as usize] = pr as f64; }
        let p_draft_analytic = pmap[x_draft as usize] as f32;

        // One launch handles CH columns. `sample_b` samples all CH; `spec_verify_b` reserves its
        // last column for the bonus, so it yields CH-1 draft-column trials and exactly ONE bonus
        // sample. The bonus column is the emission on every all-accepted step (~77% of them in
        // production), so it needs enough samples to be testable — hence a modest CH: the draft-trial
        // work is unchanged, but the launch count (and so the bonus sample count) goes up ~8x.
        const CH: usize = 32;
        let per_launch = CH - 1;
        let kmax = 64usize; let nthr = 256u32;
        let smem = ((2 * kmax + 2 * nthr as usize) * 4) as u32;

        let work = pool.get_bf16(vocab * CH);
        let mut tok_out = self.dev.alloc_zeros::<i32>(CH).unwrap();
        let p_out = pool.get(CH);
        let mut r_out = self.dev.alloc_zeros::<i32>(CH).unwrap();
        let mut b_out = self.dev.alloc_zeros::<i32>(1).unwrap();
        let mut dt_dev = self.dev.alloc_zeros::<i32>(CH).unwrap();
        let mut dq_dev = self.dev.alloc_zeros::<f32>(CH).unwrap();
        let mut t_dev = self.dev.alloc_zeros::<f32>(CH).unwrap();
        let mut k_dev = self.dev.alloc_zeros::<i32>(CH).unwrap();
        let mut p_dev = self.dev.alloc_zeros::<f32>(CH).unwrap();
        let mut sd_dev = self.dev.alloc_zeros::<u32>(CH).unwrap();
        self.dev.memset_zeros(&mut tok_out).unwrap();
        self.dev.memset_zeros(&mut r_out).unwrap();
        self.dev.memset_zeros(&mut b_out).unwrap();
        self.dev.synchronize().unwrap();
        self.dev.htod_sync_copy_into(&vec![x_draft as i32; CH], &mut dt_dev).unwrap();
        self.dev.htod_sync_copy_into(&vec![1.0f32; CH], &mut dq_dev).unwrap();   // greedy draft: q = 1
        self.dev.htod_sync_copy_into(&vec![temp; CH], &mut t_dev).unwrap();
        self.dev.htod_sync_copy_into(&vec![top_k as i32; CH], &mut k_dev).unwrap();
        self.dev.htod_sync_copy_into(&vec![top_p; CH], &mut p_dev).unwrap();
        self.dev.synchronize().unwrap();

        // `sample_b` masks its logits row in place as it selects top-k, so the working copy has to be
        // re-seeded from the pristine column before every launch.
        let refill = |w: &B| unsafe {
            for c in 0..CH {
                cudarc::driver::result::memcpy_dtod_async(
                    *w.device_ptr() as u64 + (c * vocab * 2) as u64,
                    *pristine.device_ptr() as u64, vocab * 2, copy_stream).unwrap();
            }
        };

        let mut hist_a = vec![0u64; vocab];
        let mut hist_b = vec![0u64; vocab];
        let mut hist_c = vec![0u64; vocab];
        let (mut n_a, mut n_b, mut n_c, mut n_accept) = (0u64, 0u64, 0u64, 0u64);
        let mut p_draft_device = 0.0f32;
        let mut p_agree_acc = 0.0f32;
        let mut done = 0usize;
        let mut li = 0usize;

        while done < trials {
            // Each column is an independent trial, keyed like one production decode step. `rng_base`
            // is varied by the caller across temperatures — otherwise every temperature would reuse
            // the SAME uniform stream and merely apply a different threshold to it, so one unlucky
            // draw would show up as a "systematic" bias in every row.
            let keys: Vec<u64> = (0..CH)
                .map(|c| splitmix64(rng_base ^ ((li * CH + c) as u64)))
                .collect();

            // ---- A: the plain sampler on the same column (control / noise floor) ----
            let seeds_a: Vec<u32> = keys.iter().map(|&k| rng_u32(k, RNG_DOM_VERIFY, 7)).collect();
            refill(&work);
            self.dev.htod_sync_copy_into(&seeds_a, &mut sd_dev).unwrap();
            self.dev.synchronize().unwrap();
            blaunch!(self, "sample_b", (CH as u32,1,1), (nthr,1,1), smem,
                (d(&tok_out), d(&work), d(&t_dev), d(&k_dev), d(&p_dev), d(&sd_dev),
                 vocab as i32, CH as i32));
            self.sync_stream();
            let ta = self.dev.dtoh_sync_copy(&tok_out).unwrap();
            for c in 0..CH { hist_a[ta[c] as usize] += 1; n_a += 1; }

            // ---- B: stochastic MTP — the real kernel + the real host accept rule ----
            let seeds_b: Vec<u32> = keys.iter().map(|&k| rng_u32(k, RNG_DOM_VERIFY, 0)).collect();
            refill(&work);
            self.dev.htod_sync_copy_into(&seeds_b, &mut sd_dev).unwrap();
            self.dev.synchronize().unwrap();
            blaunch!(self, "spec_verify_b", (CH as u32,1,1), (nthr,1,1), smem,
                (d(&p_out), d(&r_out), d(&work), d(&dt_dev), d(&dq_dev),
                 d(&t_dev), d(&k_dev), d(&p_dev), d(&sd_dev), vocab as i32, CH as i32));
            self.sync_stream();
            let pv = self.dev.dtoh_sync_copy(&p_out).unwrap();
            let rv = self.dev.dtoh_sync_copy(&r_out).unwrap();
            let bv = [rv[CH - 1]];   // the bonus column now rides in resid_tok[depth-1]
            p_draft_device = pv[0];

            for c in 0..per_launch {
                if done >= trials { break; }
                // q = 1 for a greedy draft, so the accept ratio is exactly p(x_draft).
                let ru = rng_uniform(keys[c], RNG_DOM_ACCEPT, 0);
                let emit = if ru < pv[c] { n_accept += 1; x_draft } else { rv[c] as u32 };
                hist_b[emit as usize] += 1;
                n_b += 1;
                done += 1;
            }
            hist_c[bv[0] as usize] += 1;
            n_c += 1;
            li += 1;
        }

        // The draft is accepted with probability exactly p(x_draft), so the accept rate is itself a
        // binomial check on the accept rule (independent of what the residual then emits).
        let accept_rate = n_accept as f32 / n_b.max(1) as f32;
        let pd = p_draft_analytic as f64;
        let accept_z = if pd <= 0.0 || pd >= 1.0 {
            0.0
        } else {
            ((accept_rate as f64 - pd) / (pd * (1.0 - pd) / n_b as f64).sqrt()) as f32
        };

        MtpSampleStats {
            x_draft,
            nucleus_size: p_analytic.len(),
            p_draft_analytic,
            p_draft_device,
            accept_rate,
            accept_z,
            p_agree_mean: 0.0,
            accept_e: pd as f32,
            trials: n_b as usize,
            bonus_trials: n_c as usize,
            sampler: gof(&hist_a, n_a, &p_analytic),
            mtp:     gof(&hist_b, n_b, &p_analytic),
            bonus:   gof(&hist_c, n_c, &p_analytic),
            mtp_vs_sampler:   gof2(&hist_b, n_b, &hist_a, n_a),
            bonus_vs_sampler: gof2(&hist_c, n_c, &hist_a, n_a),
        }
    }

    /// S5F2 L2 — the REAL-q rejection gate (`--bench-df2-sample-realq`): `bench_rejection_gate`
    /// with the sampled-selector draft under the true `u·q < p` accept and the EXACT relu(p−q)
    /// residual (`spec_verify_realq_b`). The gate's contract: with a valid proposal q and the
    /// exact residual, the emitted distribution is EXACTLY the target p — so the emitted
    /// histogram must match the plain sampler's. `x_draft` = the DRAWN draft, `q_draft` = its
    /// selector probability, `cand_tok`/`cand_q` = the full candidate table `[16]` (the gate
    /// fixes ONE committed+draft pair; all CH columns share the candidate table).
    #[allow(clippy::too_many_arguments)]
    pub fn bench_rejection_gate_rq(&self, pool: &mut Pool, state: &mut BatchGpuState, prompt: &[u32],
                                   kv_stride: usize, a0: u32, x_draft: u32, q_draft: f32,
                                   cand_tok: &[u32], cand_q: &[f32],
                                   temp: f32, top_k: usize, top_p: f32, trials: usize,
                                   rng_base: u64) -> MtpSampleStats {
        use crate::batch::{rng_u32, rng_uniform, splitmix64, RNG_DOM_VERIFY, RNG_DOM_ACCEPT};
        let h = self.cfg.hidden_size;
        let vocab = self.cfg.vocab_size;
        let plen = prompt.len();
        let copy_stream = self.stream.stream;

        // Verify [committed, draft] → target logits (column 0 is the judged distribution).
        let (logits, residual) = self.verify_forward_core(pool, &[a0, x_draft], state, 0, kv_stride, plen, None, false);
        let logits = logits.into_b16();
        pool.release_bf16(residual, h * 2);
        let mut pristine = self.dev.alloc_zeros::<half::bf16>(vocab).unwrap();
        self.dev.memset_zeros(&mut pristine).unwrap();
        self.dev.synchronize().unwrap();
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(
                *pristine.device_ptr() as u64, *logits.device_ptr() as u64, vocab * 2, copy_stream).unwrap();
        }
        self.sync_stream();
        pool.release_bf16(logits, vocab * 2);

        // Ground truth: the nucleus distribution, computed on the host from the same logits.
        let lg: Vec<half::bf16> = self.dev.dtoh_sync_copy(&pristine).unwrap();
        let lgf: Vec<f32> = lg.iter().map(|x| x.to_f32()).collect();
        let p_analytic = nucleus_dist(&lgf, temp, top_k, top_p);
        let mut pmap = vec![0.0f64; vocab];
        for &(t, pr) in &p_analytic { pmap[t as usize] = pr as f64; }
        let p_draft_analytic = pmap[x_draft as usize] as f32;

        const CH: usize = 32;
        let per_launch = CH - 1;
        let kmax = 64usize; let nthr = 256u32;
        let smem = ((2 * kmax + 2 * nthr as usize) * 4) as u32;

        let work = pool.get_bf16(vocab * CH);
        let mut tok_out = self.dev.alloc_zeros::<i32>(CH).unwrap();
        let p_out = pool.get(CH);
        let mut r_out = self.dev.alloc_zeros::<i32>(CH).unwrap();
        let mut b_out = self.dev.alloc_zeros::<i32>(1).unwrap();
        let mut dt_dev = self.dev.alloc_zeros::<i32>(CH).unwrap();
        let mut dq_dev = self.dev.alloc_zeros::<f32>(CH).unwrap();
        let mut t_dev = self.dev.alloc_zeros::<f32>(CH).unwrap();
        let mut k_dev = self.dev.alloc_zeros::<i32>(CH).unwrap();
        let mut p_dev = self.dev.alloc_zeros::<f32>(CH).unwrap();
        let mut sd_dev = self.dev.alloc_zeros::<u32>(CH).unwrap();
        let mut cand_dev = self.dev.alloc_zeros::<u64>(CH * 16).unwrap();
        self.dev.memset_zeros(&mut tok_out).unwrap();
        self.dev.memset_zeros(&mut r_out).unwrap();
        self.dev.memset_zeros(&mut b_out).unwrap();
        self.dev.synchronize().unwrap();
        self.dev.htod_sync_copy_into(&vec![temp; CH], &mut t_dev).unwrap();
        self.dev.htod_sync_copy_into(&vec![top_k as i32; CH], &mut k_dev).unwrap();
        self.dev.htod_sync_copy_into(&vec![top_p; CH], &mut p_dev).unwrap();
        // Packed candidate table: (q bits << 32) | tok, all CH columns identical.
        let mut cand_packed: Vec<u64> = Vec::with_capacity(CH * 16);
        for _c in 0..CH {
            for k in 0..16 {
                cand_packed.push((cand_q[k].to_bits() as u64) << 32 | cand_tok[k] as u64);
            }
        }
        self.dev.htod_sync_copy_into(&cand_packed, &mut cand_dev).unwrap();
        self.dev.synchronize().unwrap();

        // `sample_b` masks its logits row in place as it selects top-k, so the working copy has to be
        // re-seeded from the pristine column before every launch.
        let refill = |w: &B| unsafe {
            for c in 0..CH {
                cudarc::driver::result::memcpy_dtod_async(
                    *w.device_ptr() as u64 + (c * vocab * 2) as u64,
                    *pristine.device_ptr() as u64, vocab * 2, copy_stream).unwrap();
            }
        };

        let mut hist_a = vec![0u64; vocab];
        let mut hist_b = vec![0u64; vocab];
        let mut hist_c = vec![0u64; vocab];
        let (mut n_a, mut n_b, mut n_c, mut n_accept) = (0u64, 0u64, 0u64, 0u64);
        let mut p_draft_device = 0.0f32;
        let mut p_agree_acc = 0.0f32;
        let mut done = 0usize;
        let mut li = 0usize;

        while done < trials {
            let keys: Vec<u64> = (0..CH)
                .map(|c| splitmix64(rng_base ^ ((li * CH + c) as u64)))
                .collect();

            // ---- A: the plain sampler on the same column (control / noise floor) ----
            let seeds_a: Vec<u32> = keys.iter().map(|&k| rng_u32(k, RNG_DOM_VERIFY, 7)).collect();
            refill(&work);
            self.dev.htod_sync_copy_into(&seeds_a, &mut sd_dev).unwrap();
            self.dev.synchronize().unwrap();
            blaunch!(self, "sample_b", (CH as u32,1,1), (nthr,1,1), smem,
                (d(&tok_out), d(&work), d(&t_dev), d(&k_dev), d(&p_dev), d(&sd_dev),
                 vocab as i32, CH as i32));
            self.sync_stream();
            let ta = self.dev.dtoh_sync_copy(&tok_out).unwrap();
            for c in 0..CH { hist_a[ta[c] as usize] += 1; n_a += 1; }

            // ---- B: the real-q verify — the real kernel + the real host accept rule (u*q < p).
            // The draft is DRAWN PER TRIAL from the selector's candidate multinomial (the classic
            // RS exactness is an average over the draft distribution — a fixed draft + q<1 does
            // NOT emit p, because the accept term q_y*min(1,p_y/q_y) needs the per-trial draw).
            // Per column: draw x ~ q (host LCG from the column key, RNG_DOM_DF2_SEL), then the
            // kernel computes p(x); accept iff u < min(1, p(x)/q(x)); else emit the kernel's
            // exact relu(p-q) residual sample. ----
            let seeds_b: Vec<u32> = keys.iter().map(|&k| rng_u32(k, RNG_DOM_VERIFY, 0)).collect();
            // Per-column drawn drafts + their q's.
            let mut drafts_c: Vec<i32> = Vec::with_capacity(CH);
            let mut qs_c: Vec<f32> = Vec::with_capacity(CH);
            for c in 0..CH {
                // multinomial draw over the 16 candidates (the walk's draw semantics).
                let sd = rng_u32(keys[c], crate::batch::RNG_DOM_DF2_SEL, 0);
                let mut sr = sd;
                sr = sr.wrapping_mul(1664525).wrapping_add(1013904223);
                let ru = (sr >> 8) as f32 * (1.0f32 / 16777216.0f32);
                let mut cum = 0.0f32;
                let mut chosen = 15usize;
                for k in 0..16 {
                    cum += cand_q[k];
                    if ru < cum { chosen = k; break; }
                }
                drafts_c.push(cand_tok[chosen] as i32);
                qs_c.push(cand_q[chosen]);
            }
            self.dev.htod_sync_copy_into(&drafts_c, &mut dt_dev).unwrap();
            self.dev.htod_sync_copy_into(&qs_c, &mut dq_dev).unwrap();
            refill(&work);
            self.dev.htod_sync_copy_into(&seeds_b, &mut sd_dev).unwrap();
            self.dev.synchronize().unwrap();
            blaunch!(self, "spec_verify_realq_b", (CH as u32,1,1), (nthr,1,1), smem,
                (d(&p_out), d(&r_out), d(&work), d(&dt_dev), d(&dq_dev), d(&cand_dev),
                 d(&t_dev), d(&k_dev), d(&p_dev), d(&sd_dev), vocab as i32, CH as i32));
            self.sync_stream();
            let pv = self.dev.dtoh_sync_copy(&p_out).unwrap();
            let rv = self.dev.dtoh_sync_copy(&r_out).unwrap();
            let bv = [rv[CH - 1]];
            p_draft_device = pv[0];

            for c in 0..per_launch {
                if done >= trials { break; }
                // real-q accept: u*q < p  ⇔  ru < min(1, p/q) with the DRAWN draft's q.
                // p-consistency: the device p(draft) vs the host nucleus p of the SAME draft.
                let qc = qs_c[c];
                let p_host_c = pmap[drafts_c[c] as usize] as f32;
                p_agree_acc += (pv[c] - p_host_c).abs();
                let ratio = if qc < 1e-12f32 { 1.0 } else { (pv[c] / qc).min(1.0) };
                let ru = rng_uniform(keys[c], RNG_DOM_ACCEPT, 0);
                let emit = if ru < ratio { n_accept += 1; drafts_c[c] as u32 } else { rv[c] as u32 };
                hist_b[emit as usize] += 1;
                n_b += 1;
                done += 1;
            }
            hist_c[bv[0] as usize] += 1;
            n_c += 1;
            li += 1;
        }

        // The accept-rate expectation over the draft distribution: E[min(1, p(x)/q(x))] =
        // sum_x q(x)*min(1, p(x)/q(x)) = sum_x min(p(x), q(x)) (the RS acceptance probability).
        let p_accept_analytic: f64 = (0..16)
            .map(|k| (pmap[cand_tok[k] as usize].min(cand_q[k] as f64)))
            .sum();
        let accept_rate = n_accept as f32 / n_b.max(1) as f32;
        let pd = p_accept_analytic;
        let accept_z = if pd <= 0.0 || pd >= 1.0 {
            0.0
        } else {
            ((accept_rate as f64 - pd) / (pd * (1.0 - pd) / n_b as f64).sqrt()) as f32
        };
        // Per-trial p-consistency: mean |device p(draft) - host p(draft)| over the DRAWN drafts.
        let p_agree_mean = p_agree_acc / n_b.max(1) as f32;

        if crate::opts::var(crate::opt!("df2-gate-dump")).is_ok() {
            println!("  [gate-dump] x_draft={x_draft} q_draft={q_draft:.4} p_analytic_len={} accept_rate={accept_rate:.4} (E={pd:.4})",
                     p_analytic.len());
            let top = |h: &Vec<u64>, n: u64, name: &str| {
                let mut idx: Vec<usize> = (0..h.len()).collect();
                idx.sort_by(|&a, &b| h[b].cmp(&h[a]));
                let mut s = format!("  [gate-dump] {name}: ");
                for &i in idx.iter().take(12) {
                    let pr = h[i] as f64 / n.max(1) as f64;
                    s.push_str(&format!("{i}={pr:.4} "));
                }
                println!("{s}");
            };
            top(&hist_a, n_a, "hist_a (sampler)");
            top(&hist_b, n_b, "hist_b (realq)");
            top(&hist_c, n_c, "hist_c (bonus)");
            let mut s = String::from("  [gate-dump] p_analytic: ");
            for &(t, pr) in p_analytic.iter().take(12) { s.push_str(&format!("{t}={pr:.4} ")); }
            println!("{s}");
            let mut s = String::from("  [gate-dump] cand: ");
            for k in 0..16 { s.push_str(&format!("{}:{:.3} ", cand_tok[k], cand_q[k])); }
            println!("{s}");
        }

        MtpSampleStats {
            x_draft,
            nucleus_size: p_analytic.len(),
            p_draft_analytic,
            p_draft_device,
            accept_rate,
            accept_z,
            p_agree_mean,
            accept_e: pd as f32,
            trials: n_b as usize,
            bonus_trials: n_c as usize,
            sampler: gof(&hist_a, n_a, &p_analytic),
            mtp:     gof(&hist_b, n_b, &p_analytic),
            bonus:   gof(&hist_c, n_c, &p_analytic),
            mtp_vs_sampler:   gof2(&hist_b, n_b, &hist_a, n_a),
            bonus_vs_sampler: gof2(&hist_c, n_c, &hist_a, n_a),
        }
    }

    /// DIAGNOSTIC: why is MTP acceptance 39.5% on tool-calling traffic and ~80% on prose?
    ///
    /// Two causes look identical from the outside and have OPPOSITE fixes:
    ///   * the DRAFT HEAD IS WEAK  — the target is confident and predictable, the head still misses.
    ///                               Fixable: distil a better head, FR-Spec, more draft capacity.
    ///   * the TEXT IS HARD        — the target itself is near-tied between tokens, so its argmax is
    ///                               close to a coin flip. NO draft head can track that. Irreducible:
    ///                               stop spending columns on draft depth, spend them on lanes.
    ///
    /// The discriminator is the TARGET's own confidence at the positions where the draft was rejected.
    /// So: run the real greedy MTP loop and, at every verify column, record the target's top-1
    /// probability and its top1-top2 margin alongside whether the draft matched. Then bucket
    /// acceptance by target confidence.
    ///
    /// (Note: the `1 - TV` acceptance ceiling from the speculative-decoding literature is for
    /// REJECTION SAMPLING. Greedy MTP accepts on ARGMAX AGREEMENT, which is not bounded by 1 - TV —
    /// two distributions can differ wildly and still share an argmax. Hence measuring directly.)
    /// `ngram`: if > 0, override each draft with a prompt-lookup n-gram proposal when the last `ngram`
    /// tokens of the running context appear earlier in it. The MTP forward still runs on the PROPOSED
    /// token, so the draft chain stays coherent; only WHICH token is proposed changes. Verify checks
    /// every draft, so this can never affect correctness — only acceptance.
    /// Build a full `TreeTopo` from a `parent[]` array (parent[0] = -1). Derives rope (depth), kv_pos
    /// (DFS slot), path (ancestor KV-slot offsets) and winsrc (ancestor-path conv window) generically,
    /// so any tree shape is expressible by its parent array alone.
    pub fn topo_from_parent(&self, parent: &[i32], pos_start: usize) -> TreeTopo {
        let n = parent.len();
        let ck = self.cfg.conv_kernel;
        let mut rope = vec![0i32; n];
        let mut kv_pos = vec![0i32; n];
        let mut path = vec![0u8; n * MAX_VERIFY];
        let mut winsrc = vec![0i32; n * ck];
        for c in 0..n {
            // ancestor DFS indices, root-first: walk[depth] = the column at that depth on c's path.
            let mut walk = vec![c as i32];
            let mut x = parent[c];
            while x >= 0 { walk.push(x); x = parent[x as usize]; }
            walk.reverse();
            let depth = walk.len() - 1;
            rope[c] = (pos_start + depth) as i32;
            kv_pos[c] = (pos_start + c) as i32;
            for (dd, &idx) in walk.iter().enumerate() { path[c * MAX_VERIFY + dd] = idx as u8; }
            for j in 0..ck {
                let wd = depth as i32 - (ck as i32 - 1) + j as i32;   // window depth
                winsrc[c * ck + j] = if wd < 0 { wd } else { walk[wd as usize] };
            }
        }
        TreeTopo { rope, kv_pos, parent: parent.to_vec(), path, winsrc, slot: None, col_pos_start: None }
    }

    /// Step-2.9 TWIN-CHAIN byte gate. Plant a fork whose two branches carry IDENTICAL tokens; the two
    /// branches' columns at equal depth must produce BIT-EQUAL logits (equal logical pc, equal ancestor
    /// tokens => identical rank-space computation). This is the direct detector for the §1 slot-derived-pc
    /// flaw: under that design the twins drift by ulps. Returns (n_columns_compared, n_bit_mismatches).
    pub fn bench_tree_twin(&self, pool: &mut Pool, state: &mut BatchGpuState, prompt: &[u32],
                           kv_stride: usize, depth: usize) -> (usize, usize) {
        let vocab = self.cfg.vocab_size;
        let plen = prompt.len();
        // Prefill the prompt into slot 0 (sets prefix KV + GDN/conv state).
        self.zero_slot_state(state, 0, kv_stride);
        let (committed, hout) = self.prefill_batch(pool, prompt, state, 0, kv_stride, 0);
        pool.release_bf16(hout, self.cfg.hidden_size * plen);

        // Twin fork: col 0 = committed; a-branch cols 1..d; b-branch cols d..2d-1 with IDENTICAL tokens.
        let n = 2 * depth - 1;
        assert!(n <= MAX_VERIFY, "twin tree width {n} exceeds MAX_VERIFY");
        let mut parent = vec![-1i32; n];
        for i in 1..depth { parent[i] = (i - 1) as i32; }         // a-branch chains from root
        parent[depth] = 0;                                        // b1 from root
        for j in 1..depth - 1 { parent[depth + j] = (depth + j - 1) as i32; }

        let mut tokens = vec![committed];
        let draft: Vec<u32> = (0..depth - 1).map(|i| ((i * 131 + 7) % (vocab - 1) + 1) as u32).collect();
        tokens.extend_from_slice(&draft);   // a-branch
        tokens.extend_from_slice(&draft);   // b-branch (identical)

        let topo = self.topo_from_parent(&parent, plen);
        // ckpt_slot=Some(2): the GDN scan writes each column's post-state into checkpoint slot (2+t), and
        // a branch reloads its parent's from there. REQUIRED for a tree (a branch's parent != t-1).
        // The caller must allocate >= 2 + n GDN state slots. Slot 0 is the lane; 1 is the MTP snapshot.
        let (logits, resid) = self.verify_forward_core_topo(pool, &tokens, state, 0, kv_stride, plen, Some(2), Some(&topo), false, false);
        let logits = logits.into_b16();
        self.sync_stream();
        let lg: Vec<half::bf16> = self.dev.dtoh_sync_copy(&logits).unwrap();
        pool.release_bf16(logits, vocab * n);
        pool.release_bf16(resid, self.cfg.hidden_size * n);

        // Compare a-branch col i vs b-branch col i, bitwise over the whole logit row.
        let mut mism = 0usize;
        let mut cmp = 0usize;
        for i in 1..depth {
            let a = &lg[i * vocab..(i + 1) * vocab];
            let bcol = depth - 1 + i;                              // b-branch col at depth i
            let b = &lg[bcol * vocab..(bcol + 1) * vocab];
            cmp += 1;
            if a.iter().map(|x| x.to_bits()).ne(b.iter().map(|x| x.to_bits())) { mism += 1; }
        }
        (cmp, mism)
    }

    /// Step-2.9 PATH-ORACLE byte gate — the strongest: for a RANDOM tree, every column's logits must be
    /// bit-equal to running that column's root-to-leaf token path as an independent PLAIN CHAIN. This is
    /// absolute ground truth (the twin test is relative — it would miss a bug that shifts both twins the
    /// same way). GDN state is snapshotted after the prefill (slot 1) and restored before every run, so
    /// each chain and the tree see the identical prefix state; the prefix KV [0,plen) is never
    /// overwritten (runs write at plen..). Returns (columns_checked, bit_mismatches).
    pub fn bench_tree_oracle(&self, pool: &mut Pool, state: &mut BatchGpuState, prompt: &[u32],
                             kv_stride: usize, n: usize, seed: u64) -> (usize, usize) {
        let vocab = self.cfg.vocab_size;
        let h = self.cfg.hidden_size;
        let plen = prompt.len();
        self.zero_slot_state(state, 0, kv_stride);
        let (committed, hout) = self.prefill_batch(pool, prompt, state, 0, kv_stride, 0);
        pool.release_bf16(hout, h * plen);
        self.copy_gdn_slot(state, 0, 1);   // snapshot post-prefill GDN state into slot 1

        // Random tree: parent[c] is a random earlier column (< c), giving a valid DFS tree. Random tokens.
        let mut rng = seed | 1;
        let mut next = |m: u64| { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng % m };
        let mut parent = vec![-1i32; n];
        for c in 1..n { parent[c] = next(c as u64) as i32; }
        let tokens: Vec<u32> = std::iter::once(committed)
            .chain((1..n).map(|_| (next((vocab - 2) as u64) + 1) as u32)).collect();

        // Tree run: restore state, verify the whole tree, capture per-column logits.
        self.copy_gdn_slot(state, 1, 0);
        let topo = self.topo_from_parent(&parent, plen);
        let (tl, tr) = self.verify_forward_core_topo(pool, &tokens, state, 0, kv_stride, plen, Some(2), Some(&topo), false, false);
        let tl = tl.into_b16();
        self.sync_stream();
        let tree_lg: Vec<half::bf16> = self.dev.dtoh_sync_copy(&tl).unwrap();
        pool.release_bf16(tl, vocab * n); pool.release_bf16(tr, h * n);

        // For each column, run its root-to-leaf path as a plain chain and compare the leaf's logits.
        let mut mism = 0usize;
        for c in 0..n {
            let mut walk = vec![c]; let mut x = parent[c];
            while x >= 0 { walk.push(x as usize); x = parent[x as usize]; }
            walk.reverse();
            let path_toks: Vec<u32> = walk.iter().map(|&i| tokens[i]).collect();
            self.copy_gdn_slot(state, 1, 0);   // restore prefix state
            let (cl, cr) = self.verify_forward_core(pool, &path_toks, state, 0, kv_stride, plen, Some(2), false);
            let cl = cl.into_b16();
            self.sync_stream();
            let chain_lg: Vec<half::bf16> = self.dev.dtoh_sync_copy(&cl).unwrap();
            let leaf = path_toks.len() - 1;
            pool.release_bf16(cl, vocab * path_toks.len()); pool.release_bf16(cr, h * path_toks.len());
            let a = &tree_lg[c * vocab..(c + 1) * vocab];
            let b = &chain_lg[leaf * vocab..(leaf + 1) * vocab];
            if a.iter().map(|x| x.to_bits()).ne(b.iter().map(|x| x.to_bits())) { mism += 1; }
        }
        (n, mism)
    }

    /// FOREST oracle (batched verify across lanes — LANES design Step 3a/3b). Pack TWO independent draft
    /// chains — one per lane, each rooted in its OWN committed slot state — into ONE verify, and assert
    /// every lane's per-column logits are BIT-EQUAL to running that lane's chain ALONE. This is the
    /// batch-invariance / GDN-forest-independence gate (`LANES_OK`): a lane's emitted logits must not
    /// depend on which other lanes share the forward. Lanes may have DIFFERENT committed lengths — each
    /// column carries its lane's prefix boundary via the topo's `col_pos_start` (Step 3b), so this also
    /// gates the per-column pos_start in the rank-space attention. GDN state of each lane is snapshotted
    /// (slots 2,3) and restored before every run; committed prefix KV is never overwritten (runs write
    /// at plen_lane..). Returns (columns, bit_mismatches).
    pub fn bench_lanes(&self, pool: &mut Pool, state: &mut BatchGpuState,
                       prompt_a: &[u32], prompt_b: &[u32], kv_stride: usize, depth: usize) -> (usize, usize) {
        let vocab = self.cfg.vocab_size;
        let h = self.cfg.hidden_size;
        let ck = self.cfg.conv_kernel;
        let plen_l = [prompt_a.len(), prompt_b.len()];
        assert!(2 * depth <= MAX_VERIFY, "forest width {} exceeds MAX_VERIFY {}", 2 * depth, MAX_VERIFY);

        // Prefill lane 0 -> slot 0, lane 1 -> slot 1; snapshot each GDN state to scratch slots 2, 3.
        self.zero_slot_state(state, 0, kv_stride);
        let (comm_a, ha) = self.prefill_batch(pool, prompt_a, state, 0, kv_stride, 0);
        pool.release_bf16(ha, h * plen_l[0]);
        self.zero_slot_state(state, 1, kv_stride);
        let (comm_b, hb) = self.prefill_batch(pool, prompt_b, state, 1, kv_stride, 0);
        pool.release_bf16(hb, h * plen_l[1]);
        self.copy_gdn_slot(state, 0, 2);
        self.copy_gdn_slot(state, 1, 3);

        // Distinct draft tokens per lane (verify checks them; the oracle only needs them equal across
        // the packed and alone runs). Each lane's chain = [committed, drafts...], `depth` columns.
        let draft_a: Vec<u32> = (0..depth - 1).map(|i| ((i * 131 + 7) % (vocab - 1) + 1) as u32).collect();
        let draft_b: Vec<u32> = (0..depth - 1).map(|i| ((i * 197 + 23) % (vocab - 1) + 1) as u32).collect();
        let toks_a: Vec<u32> = std::iter::once(comm_a).chain(draft_a.iter().copied()).collect();
        let toks_b: Vec<u32> = std::iter::once(comm_b).chain(draft_b.iter().copied()).collect();

        // FOREST topo: columns [lane0 chain (depth), lane1 chain (depth)]. Each lane's first column is a
        // ROOT (parent -1); positions/path are LANE-LOCAL (rank within the lane relative to its OWN
        // committed length), the per-column `slot` routes KV/GDN/conv to the lane, and `col_pos_start`
        // gives the lane's prefix boundary so the attention splits prefix/ancestor at the right point.
        let n = 2 * depth;
        let mut tokens = toks_a.clone(); tokens.extend_from_slice(&toks_b);
        let mut parent = vec![-1i32; n];
        let mut slotv = vec![0i32; n];
        let mut rope = vec![0i32; n];
        let mut kv_pos = vec![0i32; n];
        let mut cps = vec![0i32; n];
        let mut path = vec![0u8; n * MAX_VERIFY];
        let mut winsrc = vec![0i32; n * ck];
        for lane in 0..2usize {
            let base = lane * depth;
            let plen = plen_l[lane];
            for r in 0..depth {                          // r = rank within the lane
                let c = base + r;
                parent[c] = if r == 0 { -1 } else { (c - 1) as i32 };
                slotv[c] = lane as i32;
                rope[c] = (plen + r) as i32;
                kv_pos[c] = (plen + r) as i32;
                cps[c] = plen as i32;                    // this lane's prefix boundary
                for dd in 0..=r { path[c * MAX_VERIFY + dd] = dd as u8; }  // ancestor at lane-KV offset dd
                for j in 0..ck {
                    // conv window: `in[wd*conv_dim]` indexes the GLOBAL packed qkv buffer, so a positive
                    // window depth maps to the GLOBAL column base+wd; a negative one is the committed tail.
                    let wd = r as i32 - (ck as i32 - 1) + j as i32;
                    winsrc[c * ck + j] = if wd < 0 { wd } else { base as i32 + wd };
                }
            }
        }
        let topo = TreeTopo { rope, kv_pos, parent, path, winsrc, slot: Some(slotv), col_pos_start: Some(cps) };
        // max_pc upper bound for the packed verify grid = max lane length + n.
        let pos_start_max = *plen_l.iter().max().unwrap();

        // Packed forest verify (restore both lanes). Chains never backtrack, so no mid_s (ckpt_slot=None).
        self.copy_gdn_slot(state, 2, 0);
        self.copy_gdn_slot(state, 3, 1);
        let (pl, pr) = self.verify_forward_core_topo(pool, &tokens, state, 0, kv_stride, pos_start_max, None, Some(&topo), false, false);
        let pl = pl.into_b16();
        self.sync_stream();
        let packed: Vec<half::bf16> = self.dev.dtoh_sync_copy(&pl).unwrap();
        pool.release_bf16(pl, vocab * n); pool.release_bf16(pr, h * n);

        // Each lane ALONE (single-lane path), from its own restored committed state and pos_start.
        self.copy_gdn_slot(state, 2, 0);
        let (al, ar) = self.verify_forward_core(pool, &toks_a, state, 0, kv_stride, plen_l[0], None, false);
        let al = al.into_b16();
        self.sync_stream();
        let alone_a: Vec<half::bf16> = self.dev.dtoh_sync_copy(&al).unwrap();
        pool.release_bf16(al, vocab * depth); pool.release_bf16(ar, h * depth);

        self.copy_gdn_slot(state, 3, 1);
        let (bl, br) = self.verify_forward_core(pool, &toks_b, state, 1, kv_stride, plen_l[1], None, false);
        let bl = bl.into_b16();
        self.sync_stream();
        let alone_b: Vec<half::bf16> = self.dev.dtoh_sync_copy(&bl).unwrap();
        pool.release_bf16(bl, vocab * depth); pool.release_bf16(br, h * depth);

        // Bitwise compare each lane's columns: packed vs alone.
        let mut mism = 0usize;
        for r in 0..depth {
            let pa = &packed[r * vocab..(r + 1) * vocab];
            let aa = &alone_a[r * vocab..(r + 1) * vocab];
            if pa.iter().map(|x| x.to_bits()).ne(aa.iter().map(|x| x.to_bits())) { mism += 1; }
            let pb = &packed[(depth + r) * vocab..(depth + r + 1) * vocab];
            let bb = &alone_b[r * vocab..(r + 1) * vocab];
            if pb.iter().map(|x| x.to_bits()).ne(bb.iter().map(|x| x.to_bits())) { mism += 1; }
        }
        (n, mism)
    }

    /// P11 (BATCHED_MTP P1): the PACK gate — same byte oracle as `bench_lanes`, at the production
    /// pack widths p = 2..=5. The 2-lane gate cannot see take=3/4/5 (AGENTS §3's self-consistency
    /// blind spot: a path can be internally consistent and still wrong for the shape that ships).
    /// Per-lane depth uses the PRODUCTION allocator (MAX_VERIFY/p − 1, capped by the policy depth
    /// cap), and every lane's per-column logits must stay bit-equal to running that lane alone.
    /// Returns (columns, bit_mismatches).
    pub fn bench_lanes_pack(&self, pool: &mut Pool, state: &mut BatchGpuState,
                            prompts: &[&[u32]], kv_stride: usize, depth_cap: usize) -> (usize, usize) {
        let vocab = self.cfg.vocab_size;
        let h = self.cfg.hidden_size;
        let ck = self.cfg.conv_kernel;
        let p = prompts.len();
        assert!((2..=5).contains(&p), "pack gate covers the production take range 2..=5 (got {p})");
        let depth = ((MAX_VERIFY / p).saturating_sub(1)).clamp(1, depth_cap.max(1));
        assert!(p * depth <= MAX_VERIFY, "forest pack p={p} d={depth} exceeds MAX_VERIFY {MAX_VERIFY}");

        // Prefill lane i -> slot i; snapshot each GDN state to scratch slot p + i.
        let plen_l: Vec<usize> = prompts.iter().map(|s| s.len()).collect();
        let mut comms = Vec::with_capacity(p);
        for (i, pr) in prompts.iter().enumerate() {
            self.zero_slot_state(state, i, kv_stride);
            let (c, hb) = self.prefill_batch(pool, pr, state, i, kv_stride, 0);
            pool.release_bf16(hb, h * plen_l[i]);
            comms.push(c);
        }
        for i in 0..p { self.copy_gdn_slot(state, i, p + i); }

        // Distinct synthetic drafts per lane (the oracle only needs them equal across runs).
        let toks: Vec<Vec<u32>> = (0..p).map(|lane| {
            let drafts = (0..depth - 1)
                .map(|i| (((lane + 1) * (i * 131 + 7) + 23) % (vocab - 1) + 1) as u32)
                .collect::<Vec<_>>();
            std::iter::once(comms[lane]).chain(drafts.iter().copied()).collect()
        }).collect();

        // FOREST topo: p chains of `depth` columns each — exactly the production mtp_forest_step
        // layout (lane-local positions, per-column slot routing, per-lane prefix boundary).
        let n = p * depth;
        let mut tokens: Vec<u32> = Vec::with_capacity(n);
        let mut parent = vec![-1i32; n];
        let mut slotv = vec![0i32; n];
        let mut rope = vec![0i32; n];
        let mut kv_pos = vec![0i32; n];
        let mut cps = vec![0i32; n];
        let mut path = vec![0u8; n * MAX_VERIFY];
        let mut winsrc = vec![0i32; n * ck];
        for lane in 0..p {
            let base = lane * depth;
            tokens.extend_from_slice(&toks[lane]);
            let plen = plen_l[lane];
            for r in 0..depth {
                let c = base + r;
                parent[c] = if r == 0 { -1 } else { (c - 1) as i32 };
                slotv[c] = lane as i32;
                rope[c] = (plen + r) as i32;
                kv_pos[c] = (plen + r) as i32;
                cps[c] = plen as i32;
                for dd in 0..=r { path[c * MAX_VERIFY + dd] = dd as u8; }
                for j in 0..ck {
                    let wd = r as i32 - (ck as i32 - 1) + j as i32;
                    winsrc[c * ck + j] = if wd < 0 { wd } else { base as i32 + wd };
                }
            }
        }
        let topo = TreeTopo { rope, kv_pos, parent, path, winsrc, slot: Some(slotv), col_pos_start: Some(cps) };
        let pos_start_max = *plen_l.iter().max().unwrap();

        // Packed forest verify with every lane restored from its snapshot.
        for i in 0..p { self.copy_gdn_slot(state, p + i, i); }
        let (pl, pr) = self.verify_forward_core_topo(pool, &tokens, state, 0, kv_stride, pos_start_max, None, Some(&topo), false, false);
        let pl = pl.into_b16();
        self.sync_stream();
        let packed: Vec<half::bf16> = self.dev.dtoh_sync_copy(&pl).unwrap();
        pool.release_bf16(pl, vocab * n); pool.release_bf16(pr, h * n);

        // Each lane ALONE, from its own restored committed state and pos_start.
        let mut mism = 0usize;
        for lane in 0..p {
            self.copy_gdn_slot(state, p + lane, lane);
            let (al, ar) = self.verify_forward_core(pool, &toks[lane], state, lane, kv_stride, plen_l[lane], None, false);
            let al = al.into_b16();
            self.sync_stream();
            let alone: Vec<half::bf16> = self.dev.dtoh_sync_copy(&al).unwrap();
            pool.release_bf16(al, vocab * depth); pool.release_bf16(ar, h * depth);
            for r in 0..depth {
                let pp = &packed[(lane * depth + r) * vocab..(lane * depth + r + 1) * vocab];
                let aa = &alone[r * vocab..(r + 1) * vocab];
                if pp.iter().map(|x| x.to_bits()).ne(aa.iter().map(|x| x.to_bits())) { mism += 1; }
            }
        }
        (n, mism)
    }

    /// Step-3a end-to-end gate: verify + accept-walk + KV compaction on a planted fork whose SECOND
    /// branch carries the target's own greedy continuation (so it is fully accepted) and whose FIRST
    /// branch is a decoy (rejected at depth 1). Checks: (a) the accepted path is the decoy's sibling
    /// chain, (b) compaction moves each accepted node's KV to its contiguous slot (bitwise), (c) the GDN
    /// leaf checkpoint adoption lands the right state. Returns (emitted_matches_greedy, kv_compacted_ok).
    pub fn bench_tree_accept(&self, pool: &mut Pool, state: &mut BatchGpuState, prompt: &[u32],
                             kv_stride: usize, depth: usize) -> (bool, bool) {
        let vocab = self.cfg.vocab_size;
        let (nkv, hd) = (self.eff_num_kv_heads(), self.cfg.head_dim);   // eff: matches compact_kv (KV is head-sharded under TP)
        let h = self.cfg.hidden_size;
        let plen = prompt.len();
        self.zero_slot_state(state, 0, kv_stride);
        let (committed, hout) = self.prefill_batch(pool, prompt, state, 0, kv_stride, 0);
        pool.release_bf16(hout, h * plen);
        self.copy_gdn_slot(state, 0, 1);   // snapshot post-prefill GDN

        // The target's own greedy continuation, built one token at a time via chain verify.
        let mut greedy = vec![committed];
        for _ in 0..depth {
            self.copy_gdn_slot(state, 1, 0);
            let (l, r) = self.verify_forward_core(pool, &greedy, state, 0, kv_stride, plen, Some(2), false);
            let l = l.into_b16();
            self.sync_stream();
            let lg: Vec<half::bf16> = self.dev.dtoh_sync_copy(&l).unwrap();
            pool.release_bf16(l, vocab * greedy.len()); pool.release_bf16(r, h * greedy.len());
            let last = greedy.len() - 1;
            let col = &lg[last * vocab..(last + 1) * vocab];
            let (mut bi, mut bv) = (0usize, f32::NEG_INFINITY);
            for (i, &x) in col.iter().enumerate() { let v = x.to_f32(); if v > bv { bv = v; bi = i; } }
            greedy.push(bi as u32);
        }
        // greedy = [committed, g1, g2, ..., g_depth]. Accepted branch tokens = g1..g_{depth-1}; bonus g_depth.

        // Plant fork-then-chain: col0=committed; col1=decoy (child of root); cols 2..depth = B chain (g1..).
        let n = depth + 1;   // root + decoy + (depth-1) B nodes
        assert!(n <= MAX_VERIFY);
        let mut parent = vec![-1i32; n];
        parent[1] = 0;                                   // decoy, child of root
        parent[2] = 0;                                   // B1, child of root
        for j in 3..n { parent[j] = (j - 1) as i32; }    // B chain
        let decoy = if greedy[1] == 1 { 2 } else { 1 };  // any token != g1
        let mut tokens = vec![committed, decoy];
        tokens.extend_from_slice(&greedy[1..depth]);     // B branch = g1..g_{depth-1}
        assert_eq!(tokens.len(), n);

        self.copy_gdn_slot(state, 1, 0);
        let topo = self.topo_from_parent(&parent, plen);
        let (tl, tr) = self.verify_forward_core_topo(pool, &tokens, state, 0, kv_stride, plen, Some(2), Some(&topo), false, false);
        let tl = tl.into_b16();
        self.sync_stream();
        let tlg: Vec<half::bf16> = self.dev.dtoh_sync_copy(&tl).unwrap();
        pool.release_bf16(tl, vocab * n); pool.release_bf16(tr, h * n);
        // preds per column
        let preds: Vec<u32> = (0..n).map(|c| {
            let col = &tlg[c*vocab..(c+1)*vocab];
            let (mut bi, mut bv) = (0usize, f32::NEG_INFINITY);
            for (i,&x) in col.iter().enumerate() { let v=x.to_f32(); if v>bv { bv=v; bi=i; } }
            bi as u32
        }).collect();

        // Accept walk (host).
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
        for c in 1..n { children[parent[c] as usize].push(c); }
        let mut path = vec![0usize]; let mut emitted = vec![]; let mut cur = 0usize;
        loop {
            let want = preds[cur]; emitted.push(want);
            match children[cur].iter().copied().find(|&c| tokens[c] == want) {
                Some(c) => { path.push(c); cur = c; } None => break,
            }
        }
        // Expected emitted = greedy[1..=depth].
        let emit_ok = emitted == greedy[1..=depth].to_vec();

        // Compaction: snapshot the scattered KV of the accepted path, compact, compare.
        let src_pos: Vec<i32> = path.iter().map(|&p| p as i32).collect();
        // read the pre-compact KV at the accepted slots (layer 0's first full-attn layer) for the check
        let fa0 = self.cfg.layer_types.iter().position(|t| matches!(t, LayerType::FullAttention)).unwrap();
        let before: Vec<half::bf16> = self.dev.dtoh_sync_copy(state.k_cache[fa0].as_ref().unwrap()).unwrap();
        self.compact_kv(pool, state, 0, plen, &src_pos, kv_stride);
        self.sync_stream();
        let after: Vec<half::bf16> = self.dev.dtoh_sync_copy(state.k_cache[fa0].as_ref().unwrap()).unwrap();
        // after compaction, position (plen+k) must equal before's position (plen + src_pos[k]) for every head.
        let mut kv_ok = true;
        for k in 0..path.len() {
            for hh in 0..nkv {
                let dst = (((0*nkv + hh) * kv_stride) + plen + k) * hd;
                let src = (((0*nkv + hh) * kv_stride) + plen + src_pos[k] as usize) * hd;
                if before[src..src+hd].iter().map(|x|x.to_bits()).ne(after[dst..dst+hd].iter().map(|x|x.to_bits())) {
                    kv_ok = false;
                }
            }
        }
        (emit_ok, kv_ok)
    }
}
