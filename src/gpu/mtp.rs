//! `impl GpuModel`: MTP head: hidden argmax / top-k, `mtp_reprime`, `mtp_prime_prompt`, `mtp_fork_draft`, draft steps and
//! `mtp_forward`. MTP semantics: pre-final-norm hidden, single-slot head KV (slot 0 always), AGENTS.md section 2.5-2.6. Map: `mod.rs`.

use super::*;

impl GpuModel {
    /// Greedy argmax of lm_head(hidden) for a single (batch=1) hidden vector. Returns the token id.
    /// Argmax over the DRAFT head (or the full head if no draft vocab is configured). This is only
    /// ever called to pick a draft token -- never to emit one -- so restricting its vocabulary cannot
    /// affect the output, only the acceptance rate.
    pub fn argmax_hidden(&self, pool: &mut Pool, hidden: &B) -> u32 {
        // qwen4_exp: an MTP output is the hc-stream stack — mix it to one hidden for the head.
        let mixed = self.mtp_mix_for_head(pool, hidden, 1);
        let hidden = mixed.as_ref().unwrap_or(hidden);
        let r = self.argmax_hidden_inner(pool, hidden);
        if let Some(m) = mixed { pool.release_bf16(m, self.cfg.hidden_size); }
        if self.draft_head.is_some() { crate::metrics::draft_vocab_step(); }
        r
    }
    fn argmax_hidden_inner(&self, pool: &mut Pool, hidden: &B) -> u32 {
        // The no-draft-head arm goes through `logits_batch` so a TP vocab-sharded head is GATHERED
        // before the argmax — a direct gemm on the shard would read out of bounds and argmax only
        // half the vocab. Unsharded, logits_batch is the same gemm as before, byte for byte.
        // hy_v3: the head is contractually FP32 (enable_lm_head_fp32) — bf16-rounded draft logits
        // flip near-tie proposals and cost real acceptance. The draft head IS that same head, so
        // its argmax follows the same contract (the verify side already does).
        if self.draft_head.is_none() && self.cfg.lm_head_fp32 {
            let vocab = self.cfg.vocab_size;
            let logits = self.logits_batch_f32(pool, hidden, 1);
            let block = 1024u32;
            blaunch!(self, "argmax_f32_b", (1,1,1), (block,1,1), (block*8),
                (*self.sc_tok.device_ptr() as u64, d(&logits), vocab as i32, 1));
            pool.release(logits, vocab);
            self.sync_stream();
            return self.dev.dtoh_sync_copy(&self.sc_tok).unwrap()[0] as u32;
        }
        let (mut logits, vocab) = match &self.draft_head {
            Some(dh) => {
                let vocab = self.draft_ids.len();
                if let W::Bf16(wb) = dh {
                    // E19: fused argmax — the draft logits never materialize. The per-row dots are
                    // the same fixed-order instruction sequence as gemm_binv_b at N=1, and the final
                    // reduce mirrors argmax_b's comparison order, so the chosen row is bit-identical
                    // to the materialize-then-argmax path. (bf16 head only — the hy_v3 FR-Spec
                    // shape; NVFP4/FP8 draft heads keep the old path.)
                    let part = pool.get(2 * vocab);
                    blaunch!(self, "gemm_binv_argmax_b", (vocab as u32,1,1), (256,1,1), (256*4) as u32,
                        (d(&part), d(wb), d(hidden), vocab as i32, self.cfg.hidden_size as i32));
                    blaunch!(self, "argmax_part_b", (1u32,1,1), (1024u32,1,1), (1024*8) as u32,
                        (*self.sc_tok.device_ptr() as u64, d(&part), vocab as i32));
                    pool.release(part, 2 * vocab);
                    self.sync_stream();
                    let row = self.dev.dtoh_sync_copy(&self.sc_tok).unwrap()[0] as usize;
                    return self.draft_ids[row.min(self.draft_ids.len() - 1)];
                }
                let mut lg = pool.get_bf16(vocab);
                self.gemm_act(dh, hidden, &mut lg, self.cfg.hidden_size, vocab, 1);
                (lg, vocab)
            }
            None => (self.logits_batch(pool, hidden, 1), self.cfg.vocab_size),
        };
        let block = 1024u32;
        blaunch!(self, "argmax_b", (1,1,1), (block,1,1), (block*8),
            (*self.sc_tok.device_ptr() as u64, d(&logits), vocab as i32, 1));
        pool.release_bf16(logits, vocab);
        self.sync_stream();
        let row = self.dev.dtoh_sync_copy(&self.sc_tok).unwrap()[0] as usize;
        match &self.draft_head {
            Some(_) => self.draft_ids[row.min(self.draft_ids.len() - 1)],
            None => row as u32,
        }
    }

    /// Vocabulary the draft head proposes from (== full vocab when FR-Spec is off).
    pub fn draft_vocab(&self) -> usize {
        if self.draft_head.is_some() { self.draft_ids.len() } else { self.cfg.vocab_size }
    }

    /// The draft head's top-`k` candidate token ids (diagnostic only — pulls the full draft-vocab logit
    /// vector to the host and partial-sorts). Used by `bench_accept` to measure how often the target's
    /// argmax lands in the head's top-2/top-3 — i.e. what a fork at that position could rescue. NOT on
    /// any serving path.
    pub fn topk_hidden(&self, pool: &mut Pool, hidden: &B, k: usize) -> Vec<u32> {
        // Same gather rule as argmax_hidden: the no-draft-head arm must not gemm a TP-sharded head.
        let (logits, vocab) = match &self.draft_head {
            Some(dh) => {
                let vocab = self.draft_ids.len();
                let mut lg = pool.get_bf16(vocab);
                self.gemm_act(dh, hidden, &mut lg, self.cfg.hidden_size, vocab, 1);
                (lg, vocab)
            }
            None => (self.logits_batch(pool, hidden, 1), self.cfg.vocab_size),
        };
        self.sync_stream();
        let lg: Vec<half::bf16> = self.dev.dtoh_sync_copy(&logits).unwrap();
        pool.release_bf16(logits, vocab);
        if crate::opts::var(crate::opt!("accept-debug")).is_ok() {
            let nans = lg.iter().filter(|x| x.is_nan()).count();
            let mut by_val: Vec<usize> = (0..vocab).collect();
            by_val.select_nth_unstable_by(4, |&a, &b| lg[b].to_f32().total_cmp(&lg[a].to_f32()));
            eprintln!("  [accept-dbg] draft logits: {nans} NaN / {vocab}; top5 by total_cmp: {:?}",
                      by_val[vocab-5..].iter().rev().map(|&i| (i, lg[i].to_f32())).collect::<Vec<_>>());
        }
        let mut idx: Vec<usize> = (0..vocab).collect();
        let k = k.min(vocab);
        idx.select_nth_unstable_by(k - 1, |&a, &b| lg[b].to_f32().total_cmp(&lg[a].to_f32()));
        idx.truncate(k);
        idx.sort_by(|&a, &b| lg[b].to_f32().total_cmp(&lg[a].to_f32()));
        idx.into_iter().map(|row| match &self.draft_head {
            Some(_) => self.draft_ids[row.min(self.draft_ids.len() - 1)],
            None => row as u32,
        }).collect()
    }

    /// PLAN/25 Phase 0 — `topk_hidden` returning the raw logits alongside the ids (the
    /// coverage curves need the drafter's score shape, not just the ranking). Same rules as
    /// `topk_hidden`: diagnostic only, NOT on any serving path; the draft itself is still
    /// chosen by `argmax_hidden` (this function never picks the draft).
    pub fn topk_hidden_kv(&self, pool: &mut Pool, hidden: &B, k: usize) -> Vec<(u32, f32)> {
        let (logits, vocab) = match &self.draft_head {
            Some(dh) => {
                let vocab = self.draft_ids.len();
                let mut lg = pool.get_bf16(vocab);
                self.gemm_act(dh, hidden, &mut lg, self.cfg.hidden_size, vocab, 1);
                (lg, vocab)
            }
            None => (self.logits_batch(pool, hidden, 1), self.cfg.vocab_size),
        };
        self.sync_stream();
        let lg: Vec<half::bf16> = self.dev.dtoh_sync_copy(&logits).unwrap();
        pool.release_bf16(logits, vocab);
        let mut idx: Vec<usize> = (0..vocab).collect();
        let k = k.min(vocab);
        idx.select_nth_unstable_by(k - 1, |&a, &b| lg[b].to_f32().total_cmp(&lg[a].to_f32()));
        idx.truncate(k);
        idx.sort_by(|&a, &b| lg[b].to_f32().total_cmp(&lg[a].to_f32()));
        idx.into_iter().map(|row| (match &self.draft_head {
            Some(_) => self.draft_ids[row.min(self.draft_ids.len() - 1)],
            None => row as u32,
        }, lg[row].to_f32())).collect()
    }

    /// One MTP draft step: given the previous hidden (main or prior MTP output) and a token, run the
    /// MTP layer forward at the given absolute position, returning the next MTP hidden. The caller
    /// argmaxes lm_head(next_hidden) to get the predicted token. batch=1.
    /// Re-prime the MTP KV over the accepted prefix in ONE batched MTP-layer forward.
    ///
    /// After a verify, the MTP's KV for the accepted positions has to be rewritten using the REAL
    /// main-model hidden (from the verify output) instead of the MTP's own speculative draft hidden,
    /// so the head's KV stays consistent with the sequence the main model actually committed to.
    /// That used to be one `mtp_draft_step` per accepted position, each re-reading the entire MTP
    /// layer's weights — at depth 2, two full weight reads where one would do.
    ///
    /// Batching is safe for exactly the reason the multi-token verify is: `full_attn_batch` writes
    /// ALL N KV entries before the attention kernel runs, and query column k attends to
    /// KV[0..=pos[k]], so column k sees the KV written by columns < k. Causal-append in one pass.
    ///
    /// `hidden_cols` is [h, n] (column k = the real hidden feeding position pos_start+k). The MTP
    /// output is discarded — only the KV writes matter here.
    pub fn mtp_reprime(&self, pool: &mut Pool, hidden_cols: &B, tokens: &[u32], pos_start: usize,
                       mtp_kc_ptr: u64, mtp_vc_ptr: u64, kv_stride: usize) {
        let n = tokens.len();
        if n == 0 { return; }
        assert!(n <= MAX_VERIFY, "re-prime width {} exceeds MAX_VERIFY {}", n, MAX_VERIFY);
        assert!(pos_start + n <= kv_stride,
                "mtp_reprime positions {}..{} exceed the KV stride {} — rows >= stride are OOB writes",
                pos_start, pos_start + n - 1, kv_stride);
        let h = self.mtp_hidden_width();
        let rdim = self.cfg.rotary_dim;
        let tk: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let ps: Vec<i32> = (0..n).map(|k| (pos_start + k) as i32).collect();
        unsafe {
            use cudarc::driver::sys::CUdeviceptr;
            cudarc::driver::result::memcpy_htod_async(
                *self.mr_tok.device_ptr() as CUdeviceptr, &tk[..], self.stream.stream).unwrap();
            cudarc::driver::result::memcpy_htod_async(
                *self.mr_pos.device_ptr() as CUdeviceptr, &ps[..], self.stream.stream).unwrap();
        }
        self.sync_stream();
        let cos = pool.get(rdim * n);
        let sin = pool.get(rdim * n);
        blaunch!(self, "gather_rope_b", grid(rdim * n), (256,1,1), 0,
            (d(&cos), d(&sin), d(&self.cos_table), d(&self.sin_table),
             *self.mr_pos.device_ptr() as u64, rdim as i32, n as i32));
        let out = self.mtp_forward(pool, hidden_cols, *self.mr_tok.device_ptr() as u64, &self.mr_pos,
                                   mtp_kc_ptr, mtp_vc_ptr, pos_start + n, &cos, &sin,
                                   kv_stride, n, None);
        // No trailing sync_stream: nothing host-side consumes the output, and pool.release never
        // frees (I5) — the sync was a full pipeline drain serialized into every MTP step. (The
        // sync BEFORE the forward stays: mr_tok/mr_pos are async-copied from stack Vecs whose
        // lifetime must outlive the copy.)
        pool.release(cos, rdim * n);
        pool.release(sin, rdim * n);
        pool.release_bf16(out, h * n);
    }

    /// Prime the MTP head's KV over the WHOLE PROMPT, in batched causal-append passes.
    ///
    /// This used to be one `mtp_draft_step` per prompt token: 8012 sequential single-column forwards
    /// for an 8K prompt, each re-reading the entire MTP layer's weights and launching ~6 kernels.
    /// Measured on an 8K prompt it was 42,240 `gemm_mma_fp4_b` launches and roughly 11 of the 16
    /// seconds it took to produce the first token. The cost is exactly proportional to prompt length,
    /// so every short-prompt benchmark we ran reported it as zero -- it only ever hurt real clients,
    /// which is precisely how it survived this long.
    ///
    /// The head is a single layer, so the whole prompt primes in one causal-append pass. Column k
    /// consumes (hidden[k], embed(prompt[k+1])) and lands at MTP position k; `write_kv_prefill` writes
    /// all N KV entries before the attention kernel runs, and query column k attends to KV[0..=k], so
    /// a column sees the KV of every column before it. Same argument as the multi-token verify.
    ///
    /// Chunked only to bound transient memory (a 32K prompt would otherwise want multi-GB of activation
    /// scratch); the chunks are causally chained via `pos_start`, so the result is independent of CHUNK.
    ///
    /// This CANNOT affect losslessness. The MTP head only ever PROPOSES; every draft is verified by
    /// argmax against the main model. Changing how the head is primed changes the acceptance rate, and
    /// nothing else.
    /// `pos_start` is the absolute MTP position of `tokens[0]`'s slot. Non-zero on a prefix-cache hit:
    /// the head's KV for [0, pos_start) was written by the request that ended there, and only the new
    /// suffix needs priming.
    pub fn mtp_prime_prompt(&self, pool: &mut Pool, hidden_cols: &B, tokens: &[u32],
                            mtp_kc_ptr: u64, mtp_vc_ptr: u64, kv_stride: usize, pos_start: usize) {
        let n = tokens.len();
        if n == 0 { return; }
        assert!(pos_start + n <= kv_stride,
                "mtp_prime positions {}..{} exceed the KV stride {} (OOB KV write)",
                pos_start, pos_start + n - 1, kv_stride);
        let h = self.mtp_hidden_width();   // backbone hidden width (hc streams on qwen4_exp)
        let rdim = self.cfg.rotary_dim;
        const CHUNK: usize = 2048;

        let mut c0 = 0usize;
        while c0 < n {
            let cn = CHUNK.min(n - c0);
            let tk: Vec<i32> = tokens[c0..c0 + cn].iter().map(|&t| t as i32).collect();
            let ps: Vec<i32> = (c0..c0 + cn).map(|p| (pos_start + p) as i32).collect();
            let tok_dev = self.dev.htod_sync_copy(&tk).expect("htod mtp prime tokens");
            let pos_dev = self.dev.htod_sync_copy(&ps).expect("htod mtp prime pos");
            self.dev.synchronize().unwrap();   // NULL-stream htod must land before the compute stream reads

            // hidden_cols is [h, plen] COLUMN-major, so this chunk's columns are contiguous.
            let hchunk = pool.get_bf16(h * cn);
            unsafe {
                let dst = *hchunk.device_ptr() as u64;
                let src = *hidden_cols.device_ptr() as u64 + (c0 as u64) * (h as u64) * 2;
                cudarc::driver::result::memcpy_dtod_async(dst, src, cn * h * 2, self.stream.stream)
                    .expect("mtp_prime: copy hidden chunk");
            }
            let cos = pool.get(rdim * cn);
            let sin = pool.get(rdim * cn);
            blaunch!(self, "gather_rope_b", grid(rdim * cn), (256,1,1), 0,
                (d(&cos), d(&sin), d(&self.cos_table), d(&self.sin_table),
                 *pos_dev.device_ptr() as u64, rdim as i32, cn as i32));
            let out = self.mtp_forward(pool, &hchunk, *tok_dev.device_ptr() as u64, &pos_dev,
                                       mtp_kc_ptr, mtp_vc_ptr, pos_start + c0 + cn, &cos, &sin,
                                       kv_stride, cn, Some(pos_start + c0));
            self.sync_stream();
            pool.release(cos, rdim * cn);
            pool.release(sin, rdim * cn);
            pool.release_bf16(hchunk, h * cn);
            pool.release_bf16(out, h * cn);
            c0 += cn;
        }
    }

    /// Draft a FORK-THEN-CHAIN tree (k=2 at position 1, plus n-gram extra leaves). Returns
    /// (parent, tokens) with tokens[0] = committed; the a-branch has depth-1 nodes. Branch A is the
    /// head's greedy chain (bit-identical to the plain-chain drafter). Everything else is ADD-ONLY
    /// candidates (E18 + E20):
    ///   1. N-GRAM FORK (free, host-side): an n-gram match in `work` whose position-1 token DIVERGES
    ///      from A's — the whole tree is a + n-gram branch (rooted at 0), the proven exact-copy
    ///      rescue; the head top-2 fork is skipped in this case (mode 1, unchanged).
    ///   2. HEAD TOP-2 (E18): no diverging n-gram — fork with the head's second-choice token at
    ///      position 1. Measured: fork@1 yield 1.95 → 2.10 at d3 (+7.3%); one extra head forward
    ///      per step (~2.5%) plus the top-2 readback.
    ///   3. N-GRAM EXTRA LEAVES (E20): when the n-gram AGREES with A at position 1, its value is the
    ///      continuation — tool-output copies run long past the head's depth. The n-gram chain is
    ///      walked against the a-branch and the part PAST the first divergence is appended as a
    ///      sub-branch rooted at the a-branch's divergence column (own parent chain). The shared
    ///      prefix is NOT duplicated: identical tokens at the same tree position create accept-walk
    ///      ambiguity and waste columns (review §5.3 dedup rule). N-gram branches are always
    ///      host-side lookups — they never cost GPU, and may extend past the head's depth.
    /// Column cap (MAX_VERIFY total, root included): allocate the a-branch whole, then the head-fork
    /// branch whole (≤ depth-1 each, so at depth ≤ 8 they always fit: 1+7+7=15), then the n-gram
    /// continuation gets what is left — its tail is the only part ever trimmed. Dedup: no fork when
    /// the head-fork sibling equals A's first token. MTP KV is written transiently (branch chains
    /// overwrite shared positions; write-then-attend keeps each chain on its own path; the
    /// post-accept re-prime restores the accepted path — verified LOSSLESS at e53ec6d). Drafting is
    /// speculative: a bad branch only costs acceptance, never correctness.
    pub fn mtp_fork_draft(&self, pool: &mut Pool, h_prev: &B, committed: i32, mtp_pos: usize, depth: usize,
                          mtp_kc_ptr: u64, mtp_vc_ptr: u64, kv_stride: usize,
                          work: &[u32], ngram: usize) -> (Vec<i32>, Vec<u32>) {
        let hs = self.mtp_hidden_width();
        // Chain depth-1 tokens from the shared hidden m0: `first` is the branch's position-1 token;
        // each further token is the argmax of one more MTP step. Nothing is stepped at depth == 2
        // (nothing to chain — byte- and cost-identical to the old chain), and the LAST token is not
        // stepped — its hidden would be unused and its KV is covered by the post-accept re-prime.
        let chain_branch = |pool: &mut Pool, m0: &B, first: u32| -> Vec<u32> {
            let mut toks = vec![first];
            if depth > 2 {
                let mut src = self.mtp_draft_step(pool, m0, first as i32, mtp_pos + 1, mtp_kc_ptr, mtp_vc_ptr, kv_stride);
                for step in 0..depth - 2 {
                    let tok = self.argmax_hidden(pool, &src) as i32;
                    toks.push(tok as u32);
                    if step + 1 < depth - 2 {
                        let nx = self.mtp_draft_step(pool, &src, tok, mtp_pos + 2 + step, mtp_kc_ptr, mtp_vc_ptr, kv_stride);
                        pool.release_bf16(src, hs);
                        src = nx;
                    }
                }
                pool.release_bf16(src, hs);
            }
            toks
        };

        // Shared first step: m0 is branch A's first hidden AND the head-fork source (top-2 + chain).
        let m0 = self.mtp_draft_step(pool, h_prev, committed, mtp_pos, mtp_kc_ptr, mtp_vc_ptr, kv_stride);
        let tok_a = self.argmax_hidden(pool, &m0);

        // The n-gram continuation, host-side in every case (E20: it is add-only, so it is always
        // worth computing). Runs past the head's depth when the corpus keeps matching; the COLUMN
        // budget is applied at assembly, not here.
        let mut ng: Vec<u32> = Vec::new();
        if ngram > 0 && work.len() >= ngram {
            let mut ctx: Vec<u32> = work.to_vec();
            for _ in 0..MAX_VERIFY - 1 {
                let ts = ctx.len() - ngram;
                let mut found = None;
                for j in (0..ts).rev() {
                    if ctx[j..j + ngram] == ctx[ts..] { if j + ngram < ctx.len() { found = Some(ctx[j + ngram]); } break; }
                }
                match found { Some(t) => { ng.push(t); ctx.push(t); } None => break }
            }
        }

        // Branch A first (its chain is the unchanged 1x-cost draft).
        let a = chain_branch(pool, &m0, tok_a);
        // First n-gram divergence from the a-branch: ng[0..j] == a[0..j], ng[j] != a[j] (or ng ran out).
        let j = (0..ng.len().min(a.len())).find(|&k| ng[k] != a[k]).unwrap_or(ng.len().min(a.len()));

        // MODE 1 — the n-gram diverges at position 1: exact-copy rescue, highest value. Tree is
        // a + n-gram branch (rooted at 0), NO head fork (today's behavior, kept).
        if j == 0 && !ng.is_empty() {
            let nb = ng.len().min(MAX_VERIFY - 1 - a.len());   // a whole, n-gram tail trimmed at the cap
            let na = a.len();
            let n = 1 + na + nb;
            let mut parent = vec![-1i32; n];
            for i in 1..=na { parent[i] = (i - 1) as i32; }
            parent[1 + na] = 0;
            for k in 1..nb { parent[1 + na + k] = (na + k) as i32; }
            let mut tokens = vec![committed as u32];
            tokens.extend_from_slice(&a);
            tokens.extend_from_slice(&ng[..nb]);
            pool.release_bf16(m0, hs);
            return (parent, tokens);
        }

        // HEAD TOP-2 fork (E18) — the strongest position-1 token that ISN'T A's. (find rather than
        // top2[1]: under hy_v3's fp32 head contract the argmax and a bf16 top-2 can disagree on a
        // near-tie — acceptance-rate noise, never correctness.)
        let b: Vec<u32> = if depth >= 3 {
            let top2 = self.topk_hidden(pool, &m0, 2);
            match top2.into_iter().find(|&t| t != tok_a) {
                Some(tok_b) => chain_branch(pool, &m0, tok_b),
                None => vec![],
            }
        } else {
            vec![]   // depth < 3: no room to fork (each branch is one token)
        };
        pool.release_bf16(m0, hs);

        // E20 EXTRA LEAVES: the n-gram's continuation PAST the shared prefix (j >= 1 here — the
        // j == 0 case returned above; a strict prefix of a adds nothing). Rooted at the a-branch's
        // divergence column, never duplicating the shared prefix (§5.3).
        let cont: &[u32] = if j >= 1 && j < ng.len() { &ng[j..] } else { &[] };

        if b.is_empty() && cont.is_empty() {
            // Degenerate: no head-fork sibling, no n-gram coverage — a plain chain (root + a-branch).
            let n = a.len() + 1;
            let parent: Vec<i32> = (0..n as i32).map(|c| c - 1).collect();
            let mut tokens = vec![committed as u32]; tokens.extend_from_slice(&a);
            return (parent, tokens);
        }
        // Assemble: root, a-branch, head-fork branch (rooted at 0), n-gram continuation (rooted at
        // a-branch column j). Cap priority: a whole, then the head-fork branch whole, then the
        // continuation gets the remaining columns (at depth <= 8 the first two always fit, so only
        // the continuation's tail is ever trimmed).
        let (na, nb) = (a.len(), b.len());
        let nc = cont.len().min(MAX_VERIFY - 1 - na - nb);
        let n = 1 + na + nb + nc;
        let mut parent = vec![-1i32; n];
        for i in 1..=na { parent[i] = (i - 1) as i32; }
        if nb > 0 {
            parent[1 + na] = 0;
            for k in 1..nb { parent[1 + na + k] = (na + k) as i32; }
        }
        if nc > 0 {
            parent[1 + na + nb] = j as i32;              // a-branch column j (depth j)
            for k in 1..nc { parent[1 + na + nb + k] = (na + nb + k) as i32; }
        }
        let mut tokens = vec![committed as u32];
        tokens.extend_from_slice(&a);
        tokens.extend_from_slice(&b);
        tokens.extend_from_slice(&cont[..nc]);
        (parent, tokens)
    }

    pub fn mtp_draft_step(&self, pool: &mut Pool, hidden: &B, token: i32, pos: usize,
                      mtp_kc_ptr: u64, mtp_vc_ptr: u64, kv_stride: usize) -> B {
        assert!(pos < kv_stride, "mtp_draft_step pos {pos} >= KV stride {kv_stride} (OOB KV write)");
        let rdim = self.cfg.rotary_dim;
        // Htod token/pos into persistent scratch on the COMPUTE stream (ordered with the kernels).
        let tk = [token]; let ps = [pos as i32];
        unsafe {
            cudarc::driver::result::memcpy_htod_async(*self.sc_i1a.device_ptr() as cudarc::driver::sys::CUdeviceptr, &tk[..], self.stream.stream).unwrap();
            cudarc::driver::result::memcpy_htod_async(*self.sc_i1b.device_ptr() as cudarc::driver::sys::CUdeviceptr, &ps[..], self.stream.stream).unwrap();
        }
        self.sync_stream(); // ensure htod consumed (SoC source-lifetime; expert §5.5)
        let token_ptr = *self.sc_i1a.device_ptr() as u64;
        let cos = pool.get(rdim);
        let sin = pool.get(rdim);
        blaunch!(self, "gather_rope_b", grid(rdim), (256,1,1), 0,
            (d(&cos), d(&sin), d(&self.cos_table), d(&self.sin_table),
             *self.sc_i1b.device_ptr() as u64, rdim as i32, 1));
        let out = self.mtp_forward(pool, hidden, token_ptr, &self.sc_i1b,
                                   mtp_kc_ptr, mtp_vc_ptr, pos + 1, &cos, &sin, kv_stride, 1, None);
        self.sync_stream();
        pool.release(cos, rdim);
        pool.release(sin, rdim);
        out
    }

    /// Stochastic MTP draft step: like mtp_draft_step but samples from the MTP head (instead of
    /// argmax) and returns the chosen token's normalized probability q(x) under the
    /// temperature→topk→softmax→topp nucleus. The caller uses q(x) in the speculative
    /// rejection-sampling accept loop: accept draft with prob min(1, p_target(x)/q_draft(x)).
    pub fn mtp_draft_step_sample(&self, pool: &mut Pool, hidden: &B, token: i32, pos: usize,
                                 mtp_kc_ptr: u64, mtp_vc_ptr: u64, kv_stride: usize,
                                 temp: f32, top_k: usize, top_p: f32, seed: u32,
                                 draft_penalty: Option<&DraftPenalty>) -> (B, u32, f32) {
        let rdim = self.cfg.rotary_dim;
        let _h = self.cfg.hidden_size;
        let vocab = self.cfg.vocab_size;
        let tk = [token]; let ps = [pos as i32];
        unsafe {
            cudarc::driver::result::memcpy_htod_async(*self.sc_i1a.device_ptr() as cudarc::driver::sys::CUdeviceptr, &tk[..], self.stream.stream).unwrap();
            cudarc::driver::result::memcpy_htod_async(*self.sc_i1b.device_ptr() as cudarc::driver::sys::CUdeviceptr, &ps[..], self.stream.stream).unwrap();
        }
        self.sync_stream();
        let token_ptr = *self.sc_i1a.device_ptr() as u64;
        let cos = pool.get(rdim);
        let sin = pool.get(rdim);
        blaunch!(self, "gather_rope_b", grid(rdim), (256,1,1), 0,
            (d(&cos), d(&sin), d(&self.cos_table), d(&self.sin_table),
             *self.sc_i1b.device_ptr() as u64, rdim as i32, 1));
        let mhidden = self.mtp_forward(pool, hidden, token_ptr, &self.sc_i1b,
                                       mtp_kc_ptr, mtp_vc_ptr, pos + 1, &cos, &sin, kv_stride, 1, None);
        pool.release(cos, rdim);
        pool.release(sin, rdim);

        // Logits for the DRAFT, from the draft head if FR-Spec is on.
        //
        // Distribution-exactness survives the restriction, and this is the subtle part. `sample_prob_b`
        // samples from the softmax of whatever logits it is given and reports THAT probability as `q`.
        // Feed it the restricted logits and it samples from -- and reports -- the RENORMALIZED
        // restricted distribution. That restricted distribution *is* the proposal `q`, so the verify's
        // `min(1, p/q)` is the standard Leviathan/Chen scheme with a perfectly valid proposal. Tokens
        // outside the subset have q = 0 and can only enter via the residual `(p-q)+` resample, which
        // runs over the FULL vocabulary on the verify side. No approximation anywhere.
        //
        // What would be WRONG is to restrict the draft but report `q` from the full softmax: then the
        // accept ratio uses a `q` the drafter never sampled from, and the output distribution is
        // quietly skewed. The two must be the same distribution.
        let mixed = self.mtp_mix_for_head(pool, &mhidden, 1);
        let head_in: &B = mixed.as_ref().unwrap_or(&mhidden);
        let (mut logits, vocab) = match &self.draft_head {
            Some(dh) => {
                let vocab = self.draft_ids.len();
                let mut lg = pool.get_bf16(vocab);
                self.gemm_act(dh, head_in, &mut lg, self.cfg.hidden_size, vocab, 1);
                (lg, vocab)
            }
            // TP vocab-sharded head: gather via logits_batch (see argmax_hidden).
            None => (self.logits_batch(pool, head_in, 1), self.cfg.vocab_size),
        };
        if let Some(m) = mixed { pool.release_bf16(m, self.cfg.hidden_size); }
        if let Some(p) = draft_penalty {
            blaunch!(self, "rep_penalty_b", (1,1,1), (256,1,1), 0,
                (d(&logits), p.tokens_ptr, p.counts_ptr, MAX_PEN_TOKENS as i32,
                 p.rep_pen_ptr, p.presence_ptr, p.freq_ptr, vocab as i32, 1));
        }
        let tk_dev = self.dev.alloc_zeros::<i32>(1).unwrap();  // [1] i32
        let qp_dev = pool.get(1);    // [1] f32
        let mut t_dev = pool.get(1);     // [1] f32
        let mut k_dev = self.dev.alloc_zeros::<i32>(1).unwrap();   // [1] i32
        let mut p_dev = pool.get(1);     // [1] f32
        let mut sd_dev = self.dev.alloc_zeros::<u32>(1).unwrap();  // [1] u32
        let t = [temp]; let k = [crate::sampler::gpu_top_k(top_k)]; let p = [top_p]; let sd = [seed];
        self.dev.htod_sync_copy_into(&t, &mut t_dev).unwrap();
        self.dev.htod_sync_copy_into(&k, &mut k_dev).unwrap();
        self.dev.htod_sync_copy_into(&p, &mut p_dev).unwrap();
        self.dev.htod_sync_copy_into(&sd, &mut sd_dev).unwrap();
        self.dev.synchronize().unwrap();
        let smem = ((2 * 64usize + 2 * 256usize) * 4) as u32;
        blaunch!(self, "sample_prob_b", (1,1,1), (256,1,1), smem,
            (d(&tk_dev), d(&qp_dev), d(&logits), d(&t_dev), d(&k_dev), d(&p_dev), d(&sd_dev),
             vocab as i32, 1));
        pool.release_bf16(logits, vocab);
        drop(k_dev); drop(sd_dev);
        pool.release(t_dev, 1); pool.release(p_dev, 1);
        self.sync_stream();
        let row = self.dev.dtoh_sync_copy(&tk_dev).unwrap()[0] as usize;
        let token = match &self.draft_head {
            Some(_) => self.draft_ids[row.min(self.draft_ids.len() - 1)],
            None => row as u32,
        };
        let qprob = self.dev.dtoh_sync_copy(&qp_dev).unwrap()[0];
        drop(tk_dev);
        pool.release(qp_dev, 1);
        (mhidden, token, qprob)
    }
}

impl GpuModel {
    /// MTP forward pass: given the main model's hidden state h_t and the predicted token t+1,
    /// compute the next hidden state h' and predict the next token.
    /// Uses the MTP layer's own KV cache (mtp_k_cache, mtp_v_cache).
    /// Returns the predicted token id.
    pub fn mtp_forward(&self, pool: &mut Pool, hidden: &B, token_ptr: u64,
                       pos_dev: &cudarc::driver::CudaSlice<i32>,
                       mtp_kc_ptr: u64, mtp_vc_ptr: u64, max_pc: usize,
                       cos: &S, sin: &S, kv_stride: usize, batch: usize,
                       prefill_pos_start: Option<usize>) -> B {
        let slot_ids_ptr = *self.mtp_sids.device_ptr() as u64;   // always slot 0 — see `mtp_sids`
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let _nh = cfg.num_heads; let _nkv = cfg.num_kv_heads; let _hd = cfg.head_dim; let _rdim = cfg.rotary_dim;
        let mtp = self.mtp.as_ref().expect("MTP layer not loaded");
        if mtp.q4.is_some() {
            return self.mtp_forward_q4(pool, hidden, token_ptr, pos_dev, mtp_kc_ptr, mtp_vc_ptr, max_pc,
                                       cos, sin, kv_stride, batch, prefill_pos_start);
        }

        // 1. RMSNorm hidden and embedding, then concat and FC.
        if crate::opts::var(crate::opt!("accept-debug")).is_ok() {
            let v: Vec<half::bf16> = self.dev.dtoh_sync_copy(hidden).unwrap();
            let nans = v.iter().filter(|x| x.is_nan()).count();
            eprintln!("  [mtp-dbg] input hidden[{}]: {nans} NaN", v.len());
        }
        // hy_v3's MTP consumes the POST-final-norm target hidden (verified against the bf16
        // reference: pre-norm input reproduces our broken drafts, post-norm predicts correctly).
        // qwen's MTP takes the PRE-final-norm residual. Callers always hand us the pre-norm
        // residual (verify vout / prefill hout), so apply the main final_norm here for hy_v3.
        let mut post_hidden = None;
        let hidden_in: &B = if cfg.family == crate::qwen::Family::HyV3 {
            let post = pool.get_bf16(h * batch);
            blaunch!(self, "rmsnorm_b", (batch as u32,1,1), (1024,1,1), (4096) as u32,
                (d(&post), d(hidden), d(&self.final_norm), h as i32, batch as i32, fbits(cfg.rms_eps)));
            post_hidden = Some(post);
            post_hidden.as_ref().unwrap()
        } else { hidden };
        let norm_h = pool.get_bf16(h * batch);
        blaunch!(self, "rmsnorm_b", (batch as u32,1,1), (1024,1,1), (4096) as u32,
            (d(&norm_h), d(hidden_in), d(&mtp.pre_fc_norm_hidden), h as i32, batch as i32, fbits(cfg.rms_eps)));
        if let Some(post) = post_hidden { pool.release_bf16(post, h * batch); }

        // Gather embedding for the predicted token (token_ptr is a device-resident i32).
        let norm_e = pool.get_bf16(h * batch);
        self.embed_gather(*norm_e.device_ptr() as u64, token_ptr, h, batch);
        blaunch!(self, "rmsnorm_b", (batch as u32,1,1), (1024,1,1), (4096) as u32,
            (d(&norm_e), d(&norm_e), d(&mtp.pre_fc_norm_embedding), h as i32, batch as i32, fbits(cfg.rms_eps)));

        // Concat [norm_e, norm_h] -> concat [2h, batch] on-device (no host round-trip),
        // then FC: fc_out = fc @ concat, where fc is [h, 2h].
        let concat = pool.get_bf16(2 * h * batch);
        blaunch!(self, "concat_b", grid(2*h*batch), (256,1,1), 0,
            (d(&concat), d(&norm_e), d(&norm_h), h as i32, batch as i32));
        pool.release_bf16(norm_h, h * batch);
        pool.release_bf16(norm_e, h * batch);

        let mut fc_out = pool.get_bf16(h * batch);
        if mtp.fc_sharded {
            // Vocab-parallel-style gather (the logits_batch construction): rank r computes its
            // m-slice of fc's output rows, places it at [rank·h/world) of every batch column of a
            // ZEROED full buffer, and the all-reduce sum IS the gather — the peer's rows are exact
            // +0.0, and an m-slice never changes the k reduction order, so this is BITWISE-identical
            // to the replicated fc GEMM. The memset is explicit: pool reuse does not re-zero.
            let hl = h / self.shard_factor();
            let mut local = pool.get_bf16(hl * batch);
            self.gemm_act(&mtp.fc, &concat, &mut local, 2 * h, hl, batch);
            let full = *fc_out.device_ptr() as u64;
            self.memset_compute_stream(full, h * batch * 2);
            let rank = self.tp_rank.max(0) as usize;
            let cp = cudarc::driver::sys::CUDA_MEMCPY2D {
                srcXInBytes: 0, srcY: 0,
                srcMemoryType: cudarc::driver::sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                srcHost: std::ptr::null(), srcDevice: *local.device_ptr() as u64,
                srcArray: std::ptr::null_mut(), srcPitch: hl * 2,
                dstXInBytes: rank * hl * 2, dstY: 0,
                dstMemoryType: cudarc::driver::sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                dstHost: std::ptr::null_mut(), dstDevice: full,
                dstArray: std::ptr::null_mut(), dstPitch: h * 2,
                WidthInBytes: hl * 2, Height: batch,
            };
            unsafe {
                let r = cudarc::driver::sys::cuMemcpy2DAsync_v2(&cp, self.stream.stream);
                assert!(r == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                        "mtp fc gather 2D copy failed: {r:?}");
            }
            pool.release_bf16(local, hl * batch);
            self.tp_all_reduce_bf16(&mut fc_out, h * batch);
        } else {
            self.gemm_act(&mtp.fc, &concat, &mut fc_out, 2 * h, h, batch);
        }
        if crate::opts::var(crate::opt!("accept-debug")).is_ok() {
            let v: Vec<half::bf16> = self.dev.dtoh_sync_copy(&fc_out).unwrap();
            let nans = v.iter().filter(|x| x.is_nan()).count();
            let fin = v.iter().filter(|x| x.is_finite()).count();
            eprintln!("  [mtp-dbg] fc_out[{}]: {nans} NaN, {fin} finite", v.len());
        }
        pool.release_bf16(concat, 2 * h * batch);

        // 2. Standard PRE-NORM decoder layer — mirror the main full-attn layer EXACTLY. The MTP layer is
        // a real Qwen3.5 decoder layer (confirmed vs vLLM `qwen3_5_mtp.py`): `input_layernorm` is applied
        // BEFORE attention, and the RAW residual (fc_out + attn) — not its post-norm — feeds the MLP add.
        // (Previously this fed un-normed fc_out to attention AND aliased the fused-rmsnorm out/in, so the
        // MLP residual added onto the POST-normed value. Harmless-ish for the dense MLP but it miscalibrated
        // the MoE top-8 ROUTER — the 122B/35B MTP heads drafted at ~40% instead of the dense heads' ~73%.)
        let residual = fc_out;
        let normed = pool.get_bf16(h * batch);
        blaunch!(self, "rmsnorm_b", (batch as u32,1,1), (1024,1,1), (4096) as u32,
            (d(&normed), d(&residual), d(&mtp.input_ln), h as i32, batch as i32, fbits(cfg.rms_eps)));
        // §4.1: with --tp-shard-mtp the head runs the SAME sharded mixer/FFN paths a main layer
        // takes (o_proj/down partials restored by the reduce machinery; per-tensor flags, never
        // tp_world). Unsharded (default, or a non-divisible/bf16 tensor) keeps topology (a): the
        // barrier-free replicated draft.
        let ReduceOut::Full(mixer) = self.full_attn_batch(pool, &normed, &mtp.fa, pos_dev, max_pc, kv_stride,
                                         mtp_kc_ptr, mtp_vc_ptr, cos, sin, slot_ids_ptr, batch,
                                         prefill_pos_start, mtp.attn_sharded,
                                         KVCacheMode::Bf16, false, None) else {   // the DRAFT cache stays bf16 — quantized draft KV
                                                                             // costs real acceptance (the verify can't rescue it)
            unreachable!("AR landing 2: fuse=false always returns a full-rank mixer")
        };
        // residual += mixer (raw kept in `residual`); normed = rmsnorm(residual, post_ln) for the MLP input.
        if crate::opts::var(crate::opt!("accept-debug")).is_ok() {
            let v: Vec<half::bf16> = self.dev.dtoh_sync_copy(&mixer).unwrap();
            let nans = v.iter().filter(|x| x.is_nan()).count();
            let fin = v.iter().filter(|x| x.is_finite()).count();
            eprintln!("  [mtp-dbg] mixer[{}]: {nans} NaN, {fin} finite", v.len());
        }
        blaunch!(self, "fused_res_rmsnorm_b", (batch as u32,1,1), (1024,1,1), (4096) as u32,
            (d(&normed), d(&residual), d(&mixer), d(&mtp.post_ln), h as i32, batch as i32, fbits(cfg.rms_eps)));
        // E13: capture the MTP head's MLP input + output (the draft hiddens — the ONLY site where
        // the E12 fold fires at decode widths: the head's MoE is replicated, so fp32_shared is
        // false there and the fold conditions hold). Tagged mtp_head so the MTPX dump can pick them.
        if xchain_active() {
            xchain_new(batch);
            xchain_capture(|c| {
                c.mtp_head = true;
                c.layer_inputs.push(xchain_dtoh_f32(self, &normed, h * batch));
            });
        }
        let ReduceOut::Full(mlp_out) = self.ffn_batch(pool, &normed, &mtp.mlp, batch, mtp.mlp_sharded, false) else {
            unreachable!("AR landing 2: fuse=false always returns a full-rank FFN output")
        };
        let tot = h * batch;
        blaunch!(self, "add_residual_b", grid(tot), (256,1,1), 0, (d(&residual), d(&residual), d(&mlp_out), tot as i32));
        if xchain_active() {
            xchain_capture(|c| c.layer_outputs.push(xchain_dtoh_f32(self, &residual, h * batch)));
        }
        pool.release_bf16(mixer, h*batch); pool.release_bf16(mlp_out, h*batch); pool.release_bf16(normed, h*batch);

        // 4. Final norm
        let out = pool.get_bf16(h * batch);
        blaunch!(self, "rmsnorm_b", (batch as u32,1,1), (1024,1,1), (4096) as u32,
            (d(&out), d(&residual), d(&mtp.final_norm), h as i32, batch as i32, fbits(cfg.rms_eps)));
        if crate::opts::var(crate::opt!("accept-debug")).is_ok() {
            let v: Vec<half::bf16> = self.dev.dtoh_sync_copy(&out).unwrap();
            let nans = v.iter().filter(|x| x.is_nan()).count();
            let fin = v.iter().filter(|x| x.is_finite()).count();
            let mx = v.iter().map(|x| x.to_f32().abs()).fold(0.0f32, f32::max);
            eprintln!("  [mtp-dbg] final out[{}]: {nans} NaN, {fin} finite, max|.|={mx}", v.len());
        }
        pool.release_bf16(residual, h*batch);

        out
    }
}
