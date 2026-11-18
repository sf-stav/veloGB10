//! Unit tests of `copy_span` (the former inline `mod tests`; same module name). See `mod.rs` for the code under test.

    use super::*;

    /// Deterministic generator (no external RNG state).
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    fn cfg(min_n: usize, max_n: usize) -> CopySpanCfg {
        CopySpanCfg { min_n, max_n, diag_n: min_n, yield_w: 0, start_w: usize::MAX, yield_ratio: 0 }
    }

    /// Search only (no cursor): `(tokens, match_len)`.
    fn search(cs: &mut CopySpan, k: usize) -> (Vec<u32>, usize) {
        cs.drop_cursor();
        match cs.propose(k) {
            Some(Proposal { tokens, source: Source::Search { match_len }, .. }) => (tokens, match_len),
            Some(p) => panic!("cursor proposal after drop_cursor: {p:?}"),
            None => (Vec::new(), 0),
        }
    }

    /// The parked naive scanner: longest suffix, most recent strictly earlier start.
    fn naive(ctx: &[u32], min_n: usize, max_n: usize, k: usize) -> (Vec<u32>, usize) {
        let n = ctx.len();
        if k == 0 || n < min_n + 1 {
            return (Vec::new(), 0);
        }
        for l in (min_n..=max_n).rev() {
            if n < l + 1 {
                continue;
            }
            let suf = &ctx[n - l..n];
            for pos in (0..n - l).rev() {
                if &ctx[pos..pos + l] == suf {
                    return (ctx[pos + l..(pos + l + k).min(n)].to_vec(), l);
                }
            }
        }
        (Vec::new(), 0)
    }

    #[test]
    fn copy_span_longest_match_wins() {
        // tail 1,2,3 occurred twice; the most recent earlier occurrence (pos 4) continues with 7
        let ctx = vec![1u32, 2, 3, 9, 1, 2, 3, 7, 1, 2, 3];
        let mut d = CopySpan::new(&ctx, cfg(2, 8));
        assert_eq!(search(&mut d, 4), (vec![7, 1, 2, 3], 3));
        // a 3-gram match beats a 2-gram one
        let ctx = vec![4u32, 5, 6, 8, 0, 0, 4, 5, 6];
        let mut d = CopySpan::new(&ctx, cfg(2, 8));
        assert_eq!(search(&mut d, 2), (vec![8, 0], 3));
    }

    #[test]
    fn copy_span_self_match_impossible() {
        let ctx: Vec<u32> = (0..24).map(|i| i * 7 + 1).collect();
        let mut d = CopySpan::new(&ctx, cfg(3, 8));
        assert_eq!(search(&mut d, 4), (vec![], 0));
        assert_eq!(d.last_miss(), Some(MissReason::NoMatch));
        let ctx = vec![5u32, 6, 7, 5, 6, 7];
        let mut d = CopySpan::new(&ctx, cfg(3, 8));
        assert_eq!(search(&mut d, 4), (vec![5, 6, 7], 3), "continuation of the occurrence at pos 0");
    }

    #[test]
    fn copy_span_matches_naive_scanner_and_clamps_w() {
        let mut rng = Lcg(7);
        let mut checks = 0;
        for trial in 0..300 {
            let vocab = [2u64, 3, 5, 20, 1000][rng.below(5) as usize];
            let len = 1 + rng.below(120) as usize;
            let mut seq: Vec<u32> = Vec::new();
            if rng.below(2) == 0 {
                seq.extend((0..len).map(|_| rng.below(vocab) as u32));
            } else {
                let start = 3 + rng.below(18) as usize;
                seq.extend((0..start).map(|_| rng.below(vocab) as u32));
                while seq.len() < len {
                    if rng.below(10) < 6 && seq.len() > 4 {
                        let a = rng.below(seq.len() as u64 - 2) as usize;
                        let b = (a + 2 + rng.below(14) as usize).min(seq.len());
                        let chunk = seq[a..b].to_vec();
                        seq.extend(chunk);
                    } else {
                        seq.push(rng.below(vocab) as u32);
                    }
                }
            }
            let min_n = [1usize, 2, 3, 4, 6, 16][rng.below(6) as usize];
            let max_n = min_n.max([3usize, 8, 8, 20][rng.below(4) as usize]);
            let k = [1usize, 4, 10, 15][rng.below(4) as usize];
            let np = rng.below(seq.len() as u64 + 1) as usize;
            let mut d = CopySpan::new(&seq[..np], cfg(min_n, max_n));
            for i in np..=seq.len() {
                let got = search(&mut d, k);
                let want = naive(d.ctx(), min_n, max_n, k);
                assert_eq!(got, want, "trial {trial} at {i}");
                assert!(got.0.len() <= k, "W clamp");
                checks += 1;
                if i < seq.len() {
                    d.commit(None, 0, &seq[i..i + 1]);
                }
            }
        }
        assert!(checks > 1000);
    }

    /// Bulk build == incremental build at every split: same context, same index, same proposals.
    #[test]
    fn copy_span_incremental_equals_bulk_at_every_split() {
        let seq: Vec<u32> = vec![7, 8, 9, 4, 7, 8, 9, 5, 1, 1, 7, 8, 9, 4, 2, 2, 7, 8, 9, 4, 7, 8, 9, 5, 3, 7, 8, 9, 4, 2];
        for c in [cfg(3, 6), CopySpanCfg::new(4), CopySpanCfg::new(5)] {
            for split in 0..=seq.len() {
                let mut bulk = CopySpan::new(&seq, c);
                let mut incr = CopySpan::new(&seq[..split], c);
                incr.commit(None, 0, &seq[split..]);
                assert_eq!(bulk.ctx(), incr.ctx());
                assert_eq!(bulk.index, incr.index, "split {split}: index differs");
                assert_eq!(bulk.diag, incr.diag, "split {split}: diag index differs");
                assert_eq!(search(&mut bulk, 6), search(&mut incr, 6), "split {split}");
            }
        }
    }

    #[test]
    fn copy_span_deterministic_two_instances() {
        let c = CopySpanCfg::new(3);
        let mut a = CopySpan::new(&[3u32, 1, 4, 1, 5, 9, 2, 6], c);
        let mut b = CopySpan::new(&[3u32, 1, 4, 1, 5, 9, 2, 6], c);
        let mut rng = Lcg(99);
        for step in 0..400 {
            let pa = a.propose(10);
            let pb = b.propose(10);
            assert_eq!(pa, pb, "step {step}");
            let t = rng.below(7) as u32;
            match &pa {
                Some(p) => {
                    let m = rng.below(p.tokens.len() as u64 + 1) as usize;
                    let mut em = p.tokens[..m].to_vec();
                    em.push(t);
                    a.commit(pa.as_ref(), m, &em);
                    b.commit(pb.as_ref(), m, &em);
                }
                None => {
                    a.commit(None, 0, &[t]);
                    b.commit(None, 0, &[t]);
                }
            }
            assert_eq!(a.cursor(), b.cursor());
            assert_eq!(a.counters(), b.counters());
        }
    }

    /// Unique tokens: base..base+n.
    fn uniq(base: u32, n: usize) -> Vec<u32> {
        (0..n as u32).map(|i| base + i).collect()
    }

    /// COPYPF1: the stretch proposal reads the live cursor only, ignores the start ramp, and its commit
    /// keeps the cursor exactly when every copied token was accepted and the model's token continues
    /// the source.
    #[test]
    fn copy_span_propose_span_cursor_only_and_commit_rule() {
        let src = uniq(1000, 300);
        let mut prompt = uniq(1, 10);
        prompt.extend(&src);
        prompt.extend(uniq(50, 10));
        let mut cs = CopySpan::new(&prompt, CopySpanCfg::new(16).with_start_w(4));
        assert!(cs.propose_span(64).is_none(), "no cursor: no stretch, no search");
        assert_eq!(cs.counters().searches, 0);
        assert_eq!(cs.cursor_left(), 0);
        // the answer starts re-writing the source: 16 tokens, then a narrow (ramped) copy round
        cs.commit(None, 0, &src[..16]);
        let p = cs.propose(10).expect("a 16-token match starts a copy");
        assert_eq!(p.tokens.len(), 4, "the start ramp");
        let mut em = p.tokens.clone();
        em.push(src[20]);
        cs.commit(Some(&p), 4, &em);
        assert!(cs.cursor().is_some());
        assert_eq!(cs.cursor_run(), 5, "a new copy's run: 4 accepted + the continuing model token");
        assert_eq!(cs.cursor_left(), prompt.len() + 21 - cs.cursor().unwrap());
        // the stretch: 200 tokens at once, the ramp ignored
        let s = cs.propose_span(200).expect("a live cursor");
        assert_eq!(s.tokens, src[21..221].to_vec());
        assert_eq!(s.source, Source::Cursor);
        // all accepted + the model continues the source: the cursor survives
        let mut em = s.tokens.clone();
        em.push(src[221]);
        cs.commit(Some(&s), 200, &em);
        assert!(cs.cursor().is_some());
        assert_eq!(cs.cursor_run(), 5 + 201, "a cursor proposal adds to the run");
        // a stretch never runs past the end of the context
        let left = cs.cursor_left();
        let s2 = cs.propose_span(10_000).unwrap();
        assert_eq!(s2.tokens.len(), left);
        assert_eq!(&s2.tokens[..78], &src[222..]);
        // a disagreement at row 5: the model's token differs from the source; the cursor drops
        let mut em = s2.tokens[..5].to_vec();
        em.push(999_999);
        cs.commit(Some(&s2), 5, &em);
        assert!(cs.cursor().is_none());
        assert_eq!(cs.cursor_run(), 0, "an ended copy has no run");
        assert_eq!(cs.last_end(), Some(CopyEnd::Mismatch));
        assert!(cs.propose_span(64).is_none());
    }

    fn test_record() -> crate::copy_record::CopyRecord {
        use crate::copy_record as cr;
        cr::CopyRecord::memory(cr::Identity::from_tokenizer_bytes(b"{}", 100_000_000), 8 * cr::CHUNK_FILE_BYTES, false)
    }

    /// COPYREC1: a record copy is offered only when the context has none, starts at the ramp width, then follows its
    /// cursor through the record's continuation (copy rounds and stretches), and ends by the same commit rule; its
    /// rounds reach the record's landing rates; `propose_with(None)` is `propose`.
    #[test]
    fn copy_span_record_copy_follows_cursor_and_commit_rule() {
        let mut rec = test_record();
        let prompt = uniq(1, 30);
        let reply = uniq(5000, 300);
        rec.add_reply(&prompt, &reply);
        let ctx0 = [prompt.clone(), vec![reply[0]]].concat();
        let mut plain = CopySpan::new(&ctx0, CopySpanCfg::new(16).with_start_w(4));
        assert!(plain.propose_with(10, None).is_none(), "no record: nothing to copy in the context");
        let mut cs = CopySpan::new(&ctx0, CopySpanCfg::new(16).with_start_w(4));
        let p = cs.propose_with(10, Some(&mut rec)).expect("the record continues the 31-token tail");
        assert!(p.ext && p.source == Source::Record { match_len: 31 });
        assert_eq!(p.tokens, reply[1..5].to_vec(), "a copy's first round: the start ramp");
        assert_eq!(rec.proposals, 1);
        let mut em = p.tokens.clone();
        em.push(reply[5]);
        cs.commit(Some(&p), 4, &em);
        assert!(cs.cursor_in_record() && cs.cursor().unwrap() & EXT_TAG != 0);
        let p2 = cs.propose_with(10, Some(&mut rec)).unwrap();
        assert!(p2.ext && p2.source == Source::Cursor);
        assert_eq!(p2.tokens, reply[6..14].to_vec(), "the ramp doubles");
        let mut em = p2.tokens.clone();
        em.push(reply[14]);
        cs.commit(Some(&p2), 8, &em);
        assert_eq!(cs.cursor_left(), 300 - 15);
        let s = cs.propose_span(100).expect("a stretch over the record copy");
        assert!(s.ext);
        assert_eq!(s.tokens, reply[15..115].to_vec());
        // the model departs at row 3: the copy ends
        let mut em = s.tokens[..3].to_vec();
        em.push(7);
        cs.commit(Some(&s), 3, &em);
        assert!(cs.cursor().is_none() && !cs.cursor_in_record());
        assert_eq!(cs.last_end(), Some(CopyEnd::Mismatch));
        assert_eq!(rec.accepted, 4, "the first round's outcome reached the rates at the next ask");
        cs.flush_record(&mut rec);
        assert_eq!(rec.accepted, 4 + 8 + 3, "every round of the record copy reached the landing rates");
        // the context's own copy wins when it has one: a context with an internal repeat never asks the record
        let rep: Vec<u32> = [uniq(40_000, 20), uniq(40_000, 20)].concat();
        let mut own = CopySpan::new(&[reply[..40].to_vec(), rep[..36].to_vec()].concat(), CopySpanCfg::new(16));
        let looks = rec.lookups;
        let p = own.propose_with(10, Some(&mut rec)).unwrap();
        assert!(!p.ext && matches!(p.source, Source::Search { .. }));
        assert_eq!(rec.lookups, looks);
    }

    /// COPYREC1: two (proposer, record) pairs fed the same stream make the same proposals and hold the same state
    /// (the TP requirement: every rank proposes alone, from identical inputs).
    #[test]
    fn copy_span_record_deterministic_two_ranks() {
        let mut ra = test_record();
        let mut rb = test_record();
        let mut rng = Lcg(77);
        let texts: Vec<Vec<u32>> = (0..4).map(|i| (0..200).map(|_| 1000 * (i + 1) + rng.below(400) as u32).collect()).collect();
        for t in &texts { ra.add_reply(&[], t); rb.add_reply(&[], t); }
        for req in 0..6 {
            let start = texts[req % 4][..20].to_vec();
            let mut a = CopySpan::new(&start, CopySpanCfg::new(16).with_start_w(4));
            let mut b = CopySpan::new(&start, CopySpanCfg::new(16).with_start_w(4));
            let mut emitted_all = Vec::new();
            for step in 0..120 {
                let pa = a.propose_with(10, Some(&mut ra));
                let pb = b.propose_with(10, Some(&mut rb));
                assert_eq!(pa, pb, "request {req} step {step}");
                let t = if rng.below(5) == 0 { rng.below(5000) as u32 } else { texts[(req + step / 50) % 4][(20 + step) % 200] };
                let em = match &pa {
                    Some(p) => {
                        let m = rng.below(p.tokens.len() as u64 + 1) as usize;
                        let mut em = p.tokens[..m].to_vec();
                        em.push(t);
                        a.commit(pa.as_ref(), m, &em);
                        b.commit(pb.as_ref(), m, &em);
                        em
                    }
                    None => {
                        if rng.below(3) == 0 { a.note_neural_round(3); b.note_neural_round(3); }
                        a.commit(None, 0, &[t]);
                        b.commit(None, 0, &[t]);
                        vec![t]
                    }
                };
                emitted_all.extend(em);
                assert_eq!((a.cursor(), a.counters()), (b.cursor(), b.counters()));
            }
            a.flush_record(&mut ra);
            b.flush_record(&mut rb);
            ra.add_reply(&start, &emitted_all);
            rb.add_reply(&start, &emitted_all);
            assert_eq!(ra.ids.digest(), rb.ids.digest());
            assert_eq!(ra.rates, rb.rates);
        }
        assert!(ra.proposals > 0, "the stream exercised record copies");
    }

    #[test]
    fn copy_span_cursor_long_copy_no_search_after_first_hit() {
        let src = uniq(1000, 400);
        let mut prompt = uniq(1, 30);
        prompt.extend(&src);
        prompt.extend(uniq(50, 20));
        let w = 10;
        let (st, c) = replay(&prompt, &src, CopySpanCfg::new(16), w, true);
        // 16 plain rounds until a 16-token match exists, then copy rounds of W+1
        assert_eq!(c.searches, 17, "16 misses + the one search that found the copy");
        assert_eq!(c.proposals_search, 1);
        assert_eq!(c.proposals_cursor, st.hits - 1);
        assert_eq!(st.tok_hit, 400 - 16);
        assert_eq!(st.m0, 0);
        assert_eq!(st.hits, (384 + 10) / 11, "every copy round but the last emits W+1");
        assert_eq!(c.miss_below_min, 12, "the 4..15-token matches are counted below the minimum");
        assert_eq!(c.miss_no_match, 4);
    }

    #[test]
    fn copy_span_cursor_edit_drops_then_search_reestablishes() {
        let src = uniq(1000, 200);
        let mut prompt = uniq(1, 10);
        prompt.extend(&src);
        prompt.extend(uniq(50, 10));
        let mut cs = CopySpan::new(&prompt, CopySpanCfg::new(16));
        // generate the first 16 source tokens as plain rounds, then the copy starts
        cs.commit(None, 0, &src[..16]);
        let p = cs.propose(10).expect("16-token match");
        assert_eq!(p.source, Source::Search { match_len: 16 });
        assert_eq!(p.tokens, src[16..26]);
        cs.commit(Some(&p), 10, &src[16..27]);
        assert_eq!(cs.cursor(), Some(10 + 27));
        // one-token edit inside the copy: accept 3, the model writes 7777 instead of src[30]
        let p = cs.propose(10).unwrap();
        assert_eq!(p.source, Source::Cursor);
        assert_eq!(p.tokens, src[27..37]);
        cs.commit(Some(&p), 3, &[src[27], src[28], src[29], 7777]);
        assert_eq!(cs.cursor(), None);
        assert_eq!(cs.last_end(), Some(CopyEnd::Mismatch));
        // the suffix now contains the edit: nothing to copy until 16 tokens past it match again
        for i in 31..31 + 16 {
            assert!(cs.propose(10).is_none(), "no 16-token match yet at {i}");
            cs.commit(None, 0, &[src[i]]);
        }
        let p = cs.propose(10).expect("re-established");
        assert_eq!(p.source, Source::Search { match_len: 16 });
        assert_eq!(p.tokens, src[47..57]);
        cs.commit(Some(&p), 10, &src[47..58]);
        assert_eq!(cs.cursor(), Some(10 + 58));
        assert_eq!(cs.last_end(), None);
        assert_eq!(cs.counters().end_mismatch, 1);
    }

    #[test]
    fn copy_span_runs_off_the_end_shorter_then_none() {
        // period-6 text: the most recent earlier occurrence sits 6 tokens back, so the proposal is
        // the 6 tokens up to the end of the context (shorter than W) and the copy then has no next
        // source token; after a new token nothing matches.
        let period = uniq(500, 6);
        let prompt: Vec<u32> = period.iter().cycle().take(24).copied().collect();
        let mut cs = CopySpan::new(&prompt, CopySpanCfg::new(16));
        let p = cs.propose(10).expect("periodic match");
        assert_eq!(p.tokens.len(), 6, "shorter than W: runs to the end of the context");
        assert_eq!(p.src + p.tokens.len(), cs.ctx_len());
        assert_eq!(p.tokens, period);
        let mut em = p.tokens.clone();
        em.push(9999);
        cs.commit(Some(&p), 6, &em);
        assert_eq!(cs.cursor(), None);
        assert_eq!(cs.last_end(), Some(CopyEnd::EndOfContext));
        assert!(cs.propose(10).is_none());
        assert_eq!(cs.last_miss(), Some(MissReason::NoMatch));
        // a cursor proposal never runs past the end either, and is never empty
        let mut cs = CopySpan::new(&uniq(1, 40), CopySpanCfg::new(4));
        cs.commit(None, 0, &uniq(30, 4)); // tail 30..33 occurred at 29.. -> source 33
        let p = cs.propose(15).unwrap();
        let mut want = uniq(34, 7);
        want.extend(uniq(30, 4));
        assert_eq!(p.tokens, want, "11 < W tokens: the copy reaches the end of the context");
        assert_eq!(p.src + p.tokens.len(), cs.ctx_len());
        assert!(cs.propose(0).is_none());
    }

    /// 100 records that share field names and differ in values; the output is an exact copy.
    /// Every record has one unique id; its other values come from small sets, so most n-grams
    /// recur across records with a different continuation (the JSONL failure of the parked rule).
    fn records() -> (Vec<u32>, Vec<u32>) {
        let mut recs = Vec::new();
        for r in 0..100u32 {
            recs.push(1); // {
            recs.extend([10, 11, 12]); // "id":
            recs.push(100_000 + r);
            recs.extend([20, 21, 22]); // , "kind": "
            recs.push(200 + r % 3);
            recs.extend([30, 31]); // ", "state":
            recs.push(300 + (r / 3) % 4);
            recs.extend([40, 41, 42]); // , "level":
            recs.push(400 + (r * 7) % 5);
            recs.extend([2, 3]); // }\n
        }
        let mut prompt = uniq(900, 20);
        prompt.extend(&recs);
        prompt.extend(uniq(950, 12));
        (prompt, recs)
    }

    #[test]
    fn copy_span_repetitive_records_cursor_beats_parked_rule() {
        let (prompt, gen) = records();
        let w = 10;
        let (parked, _) = replay(&prompt, &gen, CopySpanCfg::new(4), w, false);
        let (cur4, _) = replay(&prompt, &gen, CopySpanCfg::new(4), w, true);
        let (cur16, c16) = replay(&prompt, &gen, CopySpanCfg::new(16), w, true);
        println!("records: parked {:.2} tok/copy-round ({} rounds, {} m0); cursor min_n 4 {:.2} ({} rounds); cursor min_n 16 {:.2} ({} rounds)",
                 parked.per_hit(), parked.hits, parked.m0, cur4.per_hit(), cur4.hits, cur16.per_hit(), cur16.hits);
        assert!(cur4.per_hit() >= 10.0, "cursor min_n 4: {:.2} tokens per copy round", cur4.per_hit());
        assert!(cur16.per_hit() >= 10.0, "cursor min_n 16: {:.2} tokens per copy round", cur16.per_hit());
        assert!(parked.per_hit() < 9.0, "the parked rule should lose on these records: {:.2}", parked.per_hit());
        assert_eq!(c16.proposals_search, 1, "one search, then the cursor for the whole copy");
    }

    #[test]
    fn copy_span_gate_min_n_16() {
        // a 10-token coincidence: no copy starts at min_n 16, a counted below-minimum miss
        let coincidence = uniq(700, 10);
        let mut prompt = uniq(1, 40);
        prompt.extend(&coincidence);
        prompt.extend(uniq(100, 40));
        let mut cs = CopySpan::new(&prompt, CopySpanCfg::new(16));
        cs.commit(None, 0, &uniq(800, 5));
        cs.commit(None, 0, &coincidence);
        assert!(cs.propose(10).is_none());
        assert_eq!(cs.last_miss(), Some(MissReason::BelowMin));
        assert_eq!(cs.counters().miss_below_min, 1);
        // the same context at min_n 10 would start a copy (the gate is what held it back)
        let mut lo = CopySpan::new(cs.ctx(), CopySpanCfg::new(10));
        assert_eq!(lo.propose(10).map(|p| p.tokens), Some(uniq(100, 10)));
        // with a live cursor no minimum applies: plant a cursor on the 10-token match via a
        // proposal from the min_n-10 instance and continue it at min_n 16
        let src = uniq(100, 40);
        let mut cs = CopySpan::new(&prompt, CopySpanCfg::new(16));
        cs.commit(None, 0, &uniq(800, 5));
        cs.commit(None, 0, &coincidence);
        let planted = Proposal { tokens: src[..10].to_vec(), src: 50, source: Source::Search { match_len: 10 }, max_w: 10, ctx_len: cs.ctx_len(), ramp: usize::MAX, ext: false };
        let mut em = src[..10].to_vec();
        em.push(src[10]);
        cs.commit(Some(&planted), 10, &em);
        let p = cs.propose(10).expect("cursor continues regardless of min_n");
        assert_eq!(p.source, Source::Cursor);
        assert_eq!(p.tokens, src[11..21]);
        assert_eq!(cs.counters().searches, 0, "the cursor proposal ran no search");
    }

    #[test]
    fn copy_span_reset_equals_fresh() {
        let (prompt, gen) = records();
        let c = CopySpanCfg::new(16);
        let mut used = CopySpan::new(&uniq(5, 300), c);
        let _ = replay_on(&mut used, &gen[..500], 10, true);
        used.reset(&prompt);
        let mut fresh = CopySpan::new(&prompt, c);
        assert_eq!(used.ctx(), fresh.ctx());
        assert_eq!(used.index, fresh.index);
        assert_eq!(used.diag, fresh.diag);
        assert_eq!(used.cursor(), fresh.cursor());
        assert_eq!(used.counters(), fresh.counters());
        assert_eq!(used.last_miss(), fresh.last_miss());
        assert_eq!(used.last_end(), fresh.last_end());
        let a = replay_on(&mut used, &gen, 10, true);
        let b = replay_on(&mut fresh, &gen, 10, true);
        assert_eq!(a, b);
        assert_eq!(used.counters(), fresh.counters());
    }

    #[test]
    fn copy_span_commit_clamps_inconsistent_accept_and_stale_proposals() {
        let mut cs = CopySpan::new(&uniq(1, 40), CopySpanCfg::new(4));
        cs.commit(None, 0, &uniq(10, 4));
        let p = cs.propose(10).unwrap();
        assert_eq!(p.tokens, uniq(14, 10));
        // caller claims 5 accepted, but emitted differs at index 2: m is clamped to 2
        cs.commit(Some(&p), 5, &[14, 15, 99]);
        assert_eq!(cs.counters().inconsistent_accept, 1);
        assert_eq!(cs.counters().accepted_tokens, 2);
        assert_eq!(cs.cursor(), None);
        // a proposal for an older context length is treated as unused
        let mut cs = CopySpan::new(&uniq(1, 40), CopySpanCfg::new(4));
        cs.commit(None, 0, &uniq(10, 4));
        let p = cs.propose(10).unwrap();
        cs.commit(Some(&p), 10, &uniq(14, 11));
        let stale = p.clone();
        cs.commit(Some(&stale), 3, &uniq(25, 1));
        assert_eq!(cs.cursor(), None);
        assert_eq!(cs.last_end(), Some(CopyEnd::Unused));
    }

    // ---- the yield gate ----------------------------------------------------------------------

    /// x256 fixed point of a decimal with two places (test helper).
    fn fx(v100: i64) -> i64 {
        v100 * YG_ONE / 100
    }

    fn gate_with(ec: i64, en: i64) -> YieldGate {
        YieldGate { ec, en, nc: 2, nn: 1, denied_run: 0 }
    }

    #[test]
    fn yield_gate_ratio_rule() {
        // COPY27B: the cost-ratio rule (trunk): copies must emit ratio/256 x the tokens of a neural round
        let mut g = YieldGate::default();
        assert!(g.ratio_allows(256), "no samples: allow");
        g.note_neural(6);
        g.note_copy(5);
        assert!(g.ratio_allows(256), "one copy sample: allow");
        g.note_copy(5);
        assert!(!g.ratio_allows(256), "5 per copy round < 6 per neural round");
        assert!(g.ratio_allows(200), "a lower bar (0.78) lets 5 vs 6 through");
        let mut d = 0;
        for _ in 0..YG_PROBE_EVERY { if g.decide_ratio(256) == GateDecision::Probe { d += 1; } }
        assert_eq!(d, 1, "one probe per 16 denied starts");
        for _ in 0..8 { g.note_copy(16); }
        assert!(g.ratio_allows(256) && g.decide_ratio(256) == GateDecision::Allow && !g.suspended());
        // the proposer uses the ratio rule when it is set, the EXL3 rule otherwise
        let prompt: Vec<u32> = (0..40u32).chain(0..20u32).collect();
        let mut cs = CopySpan::new(&prompt, CopySpanCfg::new(16).with_yield_gate(15).with_yield_ratio(256));
        cs.gate.note_neural(6); cs.gate.note_copy(2); cs.gate.note_copy(2);
        assert!(cs.propose(15).is_none() && cs.last_miss() == Some(MissReason::YieldGate));
    }

    #[test]
    fn yield_gate_rule_worked_points() {
        // en = 7.5: the bar is ec >= 8.48 (7.5 * 13 / 11.5)
        assert!(gate_with(fx(850), fx(750)).rule_allows(10));
        assert!(!gate_with(fx(845), fx(750)).rule_allows(10));
        assert!(!gate_with(fx(680), fx(750)).rule_allows(10), "agent echo: 6.8 per copy vs 7.5 neural");
        // en = 3: the bar is ec >= 5.57 (3 * 13 / 7)
        assert!(gate_with(fx(560), fx(300)).rule_allows(10));
        assert!(!gate_with(fx(550), fx(300)).rule_allows(10));
        // full copies (11 per round) clear every neural yield the chain can produce
        for en in [fx(300), fx(750), fx(950), fx(1100)] {
            assert!(gate_with(fx(1100), en).rule_allows(10), "en {en}");
        }
        // too few samples: allow
        assert!(YieldGate { ec: fx(100), en: fx(900), nc: 1, nn: 5, denied_run: 0 }.rule_allows(10));
        assert!(YieldGate { ec: fx(100), en: 0, nc: 5, nn: 0, denied_run: 0 }.rule_allows(10));
        // the averages: first sample initialises, then weight 1/4 (copy) and 1/8 (neural)
        let mut g = YieldGate::default();
        g.note_copy(11);
        assert_eq!(g.ec, 11 * YG_ONE);
        g.note_copy(3);
        assert_eq!(g.ec, 9 * YG_ONE); // 11 + (3 - 11) / 4
        g.note_neural(8);
        g.note_neural(0);
        assert_eq!(g.en, 7 * YG_ONE); // 8 + (0 - 8) / 8
    }

    #[test]
    fn yield_gate_suspension_and_probe_every_16th() {
        let mut g = gate_with(fx(500), fx(750));
        let mut seq = Vec::new();
        for _ in 0..48 {
            seq.push(g.decide(10));
        }
        for (i, d) in seq.iter().enumerate() {
            let want = if (i + 1) % 16 == 0 { GateDecision::Probe } else { GateDecision::Deny };
            assert_eq!(*d, want, "start {i}");
        }
        assert!(g.suspended());
        // the copy turns good: the next allowed start ends the suspension
        g.ec = fx(1100);
        assert_eq!(g.decide(10), GateDecision::Allow);
        assert!(!g.suspended());
    }

    /// Drive a gate with a round stream: `neural` rounds of the given yields between copies, each
    /// copy start followed by its rounds (only the start is gated; continuations are counted).
    /// Returns (allowed starts, denied, probes).
    fn drive(g: &mut YieldGate, episodes: usize, neural: &[usize], copy_rounds: &[usize]) -> (u64, u64, u64) {
        let (mut al, mut de, mut pr) = (0, 0, 0);
        for e in 0..episodes {
            for &n in neural {
                g.note_neural(n);
            }
            match g.decide(10) {
                GateDecision::Deny => { de += 1; continue; }
                GateDecision::Probe => pr += 1,
                GateDecision::Allow => al += 1,
            }
            let _ = e;
            for &c in copy_rounds {
                g.note_copy(c);
            }
        }
        (al, de, pr)
    }

    #[test]
    fn yield_gate_agent_echo_suspends_full_copy_never_does() {
        // agent echo (measured): copies yield ~6-7 tokens per round, neural rounds 7.5
        let mut g = YieldGate::default();
        let (al, de, pr) = drive(&mut g, 64, &[7, 8, 7, 8], &[7, 6]);
        assert!(g.suspended(), "agent echo ends suspended: {g:?}");
        assert!(de > 40 && pr >= 2 && al <= 4, "allowed {al} denied {de} probes {pr}");
        // full copies at 11 per round, neural high (9.5) and stale-low (3): never suspended
        for neural in [&[9usize, 10][..], &[3][..]] {
            let mut g = YieldGate::default();
            let (al, de, pr) = drive(&mut g, 64, neural, &[11, 11, 11]);
            assert_eq!((de, pr), (0, 0), "neural {neural:?}: {g:?}");
            assert_eq!(al, 64);
            assert!(!g.suspended());
        }
        // a full-copy request whose neural rounds are the highest possible (11): still allowed
        let mut g = YieldGate::default();
        let (_, de, _) = drive(&mut g, 16, &[11], &[11, 11]);
        assert_eq!(de, 0);
    }

    #[test]
    fn yield_gate_recovers_after_probes_when_copies_turn_good() {
        let mut g = YieldGate::default();
        drive(&mut g, 40, &[7, 8], &[6, 6]);
        assert!(g.suspended());
        // the text turns copy-heavy: the next probe's copy runs many full rounds
        let (al, de, pr) = drive(&mut g, 40, &[8], &[11, 11, 11, 11, 11, 11]);
        assert!(pr >= 1, "a probe let a copy through");
        assert!(!g.suspended(), "recovered: {g:?}");
        assert!(al >= 20, "allowed {al} denied {de}");
    }

    /// The gate inside the proposer: a denied search returns no proposal (`YieldGate` miss), a
    /// cursor continuation is never gated, the 16th denied start is proposed, reset clears it.
    #[test]
    fn yield_gate_in_proposer_and_reset() {
        let src = uniq(1000, 200);
        let mut prompt = uniq(1, 10);
        prompt.extend(&src);
        prompt.extend(uniq(50, 10));
        let c = CopySpanCfg::new(16).with_yield_gate(10);
        let mut cs = CopySpan::new(&prompt, c);
        cs.commit(None, 0, &src[..16]);
        // two poor copies (6 emitted each) against neural rounds of 8
        for _ in 0..4 {
            cs.note_neural_round(8);
        }
        let mut pos = 16;
        for _ in 0..2 {
            let p = cs.propose(10).expect("search hit, gate has too few copy samples");
            assert!(matches!(p.source, Source::Search { .. }));
            let mut em = src[pos..pos + 5].to_vec();
            em.push(4242 + pos as u32); // departs from the source
            cs.commit(Some(&p), 5, &em);
            assert_eq!(cs.cursor(), None);
            // re-sync with the source: 16 tokens of plain text that match it again
            cs.commit(None, 0, &src[pos + 6..pos + 22]);
            pos += 22;
        }
        assert_eq!(cs.gate().nc, 2);
        assert_eq!(cs.gate().ec, 6 * YG_ONE);
        // now the search finds a 16-token match, but the gate denies it 15 times, then probes
        for i in 0..15 {
            assert!(cs.propose(10).is_none(), "denied start {i}");
            assert_eq!(cs.last_miss(), Some(MissReason::YieldGate));
        }
        assert_eq!(cs.counters().gate_denied, 15);
        let p = cs.propose(10).expect("the 16th denied start is a probe");
        assert_eq!(cs.counters().gate_probes, 1);
        assert_eq!(p.tokens, src[pos..pos + 10]);
        // a full round: the cursor continues without asking the gate
        cs.commit(Some(&p), 10, &src[pos..pos + 11]);
        let q = cs.propose(10).expect("cursor");
        assert_eq!(q.source, Source::Cursor);
        assert_eq!(cs.counters().gate_denied, 15);
        // reset == fresh, gate included
        cs.reset(&prompt);
        let fresh = CopySpan::new(&prompt, c);
        assert_eq!(cs.gate(), fresh.gate());
        assert_eq!(cs.counters(), fresh.counters());
        // the gate off: the same stream never denies
        let mut off = CopySpan::new(&prompt, CopySpanCfg::new(16));
        off.commit(None, 0, &src[..16]);
        for _ in 0..4 { off.note_neural_round(8); }
        for _ in 0..3 {
            let p = off.propose(10).unwrap();
            let pos = p.src - 10;
            let mut em = src[pos..pos + 5].to_vec();
            em.push(4242);
            off.commit(Some(&p), 5, &em);
            off.commit(None, 0, &src[pos + 6..pos + 22]);
        }
        assert_eq!(off.counters().gate_denied, 0);
    }

    // ---- the start ramp (WIDEVERIFY1 refinement 1) ---------------------------------------------

    /// Replay that records every proposal as (tokens, src, source) — the ramp field aside.
    fn replay_props(prompt: &[u32], gen: &[u32], cfg: CopySpanCfg, max_w: usize) -> (Vec<Option<(Vec<u32>, usize, Source)>>, ReplayStats) {
        let mut cs = CopySpan::new(prompt, cfg);
        let mut st = ReplayStats { n: gen.len() as u64, ..Default::default() };
        let mut log = Vec::new();
        let mut pos = 0;
        while pos < gen.len() {
            match cs.propose(max_w) {
                Some(p) => {
                    let m = p.tokens.iter().zip(&gen[pos..]).take_while(|(a, b)| a == b).count();
                    let e = (m + 1).min(gen.len() - pos);
                    st.hits += 1;
                    st.tok_hit += e as u64;
                    st.m0 += (m == 0) as u64;
                    st.full += (m == p.tokens.len() && m == max_w) as u64;
                    cs.commit(Some(&p), m, &gen[pos..pos + e]);
                    log.push(Some((p.tokens, p.src, p.source)));
                    pos += e;
                }
                None => {
                    cs.commit(None, 0, &gen[pos..pos + 1]);
                    log.push(None);
                    pos += 1;
                }
            }
        }
        (log, st)
    }

    /// A long exact copy: the copy's rounds carry 4, 8, then the cap (10; with max_w 15: 4, 8, 15).
    #[test]
    fn copy_span_ramp_width_sequence_on_a_long_copy() {
        let src = uniq(1000, 400);
        let mut prompt = uniq(1, 30);
        prompt.extend(&src);
        prompt.extend(uniq(50, 20));
        for (max_w, want) in [(10usize, vec![4usize, 8, 10, 10, 10]), (15, vec![4, 8, 15, 15, 15]), (4, vec![4, 4, 4, 4]),
                              (3, vec![3, 3, 3])] {
            let (log, st) = replay_props(&prompt, &src, CopySpanCfg::new(16).with_start_w(4), max_w);
            let widths: Vec<usize> = log.iter().flatten().map(|(t, _, _)| t.len()).collect();
            assert_eq!(&widths[..want.len()], &want[..], "max_w {max_w}: {widths:?}");
            assert!(widths[want.len()..widths.len() - 1].iter().all(|&w| w == max_w), "the cap holds after the ramp: {widths:?}");
            assert_eq!(st.m0, 0);
            assert_eq!(st.tok_hit, 400 - 16, "the ramp changes the rounds, never the tokens");
            // the first proposal came from a search, every later one from the cursor
            assert!(matches!(log.iter().flatten().next().unwrap().2, Source::Search { .. }));
            assert!(log.iter().flatten().skip(1).all(|(_, _, s)| *s == Source::Cursor));
        }
        // counters: one narrow search, then narrow cursor rounds until the cap is reached (4 -> 8 at max_w 10)
        let (_, c) = replay(&prompt, &src, CopySpanCfg::new(16).with_start_w(4), 10, true);
        assert_eq!((c.narrow_search, c.narrow_cursor), (1, 1), "{c:?}");
        let (_, c) = replay(&prompt, &src, CopySpanCfg::new(16).with_start_w(1), 10, true);
        assert_eq!((c.narrow_search, c.narrow_cursor), (1, 3), "1, 2, 4, 8 are narrow below 10: {c:?}");
    }

    /// A mismatch drops the cursor; the next copy (a new search) starts at start_w again, and a
    /// clamped round (a small caller max_w) does not shrink the ramp stage.
    #[test]
    fn copy_span_ramp_resets_after_a_mismatch() {
        let src = uniq(1000, 200);
        let mut prompt = uniq(1, 10);
        prompt.extend(&src);
        prompt.extend(uniq(50, 10));
        let mut cs = CopySpan::new(&prompt, CopySpanCfg::new(16).with_start_w(4));
        cs.commit(None, 0, &src[..16]);
        let p = cs.propose(10).expect("search");
        assert_eq!((p.tokens.len(), p.ramp), (4, 4));
        assert_eq!(p.tokens, src[16..20]);
        cs.commit(Some(&p), 4, &src[16..21]);
        let p = cs.propose(10).expect("cursor");
        assert_eq!((p.source, p.tokens.len(), p.ramp), (Source::Cursor, 8, 8));
        cs.commit(Some(&p), 8, &src[21..30]);
        // a round clamped by the scheduler (depth 3) proposes 3, but the ramp keeps its stage (16)
        let p = cs.propose(3).expect("cursor");
        assert_eq!((p.tokens.len(), p.ramp), (3, 16));
        cs.commit(Some(&p), 3, &src[30..34]);
        let p = cs.propose(10).expect("cursor");
        assert_eq!((p.tokens.len(), p.ramp), (10, 32));
        // the model departs from the source after 2 tokens: the copy ends
        cs.commit(Some(&p), 2, &[src[34], src[35], 9999]);
        assert_eq!(cs.cursor(), None);
        assert_eq!(cs.last_end(), Some(CopyEnd::Mismatch));
        // re-sync 16 tokens past the edit: the new copy starts narrow again
        cs.commit(None, 0, &src[37..53]);
        let p = cs.propose(10).expect("re-established");
        assert!(matches!(p.source, Source::Search { .. }));
        assert_eq!((p.tokens.len(), p.ramp), (4, 4));
        assert_eq!(p.tokens, src[53..57]);
        // a zero-accept round also ends the copy: the next start is narrow
        cs.commit(Some(&p), 0, &[4242]);
        assert_eq!(cs.cursor(), None);
        // reset == fresh (ramp included)
        cs.reset(&prompt);
        cs.commit(None, 0, &src[..16]);
        assert_eq!(cs.propose(10).unwrap().tokens.len(), 4);
    }

    /// start_w >= max_w reproduces the pre-ramp proposals exactly (tokens, source position, source)
    /// on every existing test stream: the long copy, the repetitive records, the edit stream, random
    /// streams with copies; and start_w = usize::MAX (the `new` default) is the same thing.
    #[test]
    fn copy_span_ramp_full_width_reproduces_the_previous_proposals() {
        let mut streams: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
        {
            let src = uniq(1000, 400);
            let mut prompt = uniq(1, 30);
            prompt.extend(&src);
            prompt.extend(uniq(50, 20));
            streams.push((prompt, src));
        }
        streams.push(records());
        {
            let src = uniq(1000, 200);
            let mut prompt = uniq(1, 10);
            prompt.extend(&src);
            let mut gen = src[..30].to_vec();
            gen.push(7777);
            gen.extend(&src[31..200]);
            streams.push((prompt, gen));
        }
        let mut rng = Lcg(31);
        for _ in 0..40 {
            let vocab = [5u64, 50, 1000][rng.below(3) as usize];
            let mut seq: Vec<u32> = (0..20).map(|_| rng.below(vocab) as u32).collect();
            while seq.len() < 400 {
                if rng.below(10) < 7 {
                    let a = rng.below(seq.len() as u64 - 2) as usize;
                    let b = (a + 2 + rng.below(40) as usize).min(seq.len());
                    let chunk = seq[a..b].to_vec();
                    seq.extend(chunk);
                } else {
                    seq.push(rng.below(vocab) as u32);
                }
            }
            let np = 100 + rng.below(200) as usize;
            streams.push((seq[..np].to_vec(), seq[np..].to_vec()));
        }
        for (si, (prompt, gen)) in streams.iter().enumerate() {
            for min_n in [4usize, 16] {
                for max_w in [3usize, 10, 15] {
                    let base = replay_props(prompt, gen, CopySpanCfg::new(min_n), max_w);
                    for sw in [max_w, max_w + 1, 64, usize::MAX] {
                        let got = replay_props(prompt, gen, CopySpanCfg::new(min_n).with_start_w(sw), max_w);
                        assert_eq!(got.0, base.0, "stream {si} min_n {min_n} max_w {max_w} start_w {sw}: proposals differ");
                        assert_eq!(got.1, base.1);
                    }
                }
            }
        }
    }

    /// The replay probe honours --copy-span-start-w: at full width it prints exactly the pre-ramp
    /// totals (`replay` without a ramp), at 4 the ramp's.
    #[test]
    fn copy_span_probe_replay_honours_start_w() {
        let (prompt, gen) = records();
        let rec = serde_json::json!([{ "name": "records", "prompt_ids": prompt, "gen_ids": gen }]);
        let path = std::env::temp_dir().join(format!("copy_span_probe_{}.json", std::process::id()));
        std::fs::write(&path, rec.to_string()).unwrap();
        let run = |sw: usize| {
            let mut out = Vec::new();
            probe_replay(path.to_str().unwrap(), 16, 10, sw, &mut out).unwrap();
            let t = String::from_utf8(out).unwrap();
            t.lines().find(|l| l.starts_with("COPY_SPAN_REPLAY_TOTAL")).unwrap().to_string()
        };
        let (base, _) = replay(&prompt, &gen, CopySpanCfg::new(16), 10, true);
        let want = format!("COPY_SPAN_REPLAY_TOTAL N={} hits={} tok_hit={} m0={} full={}", base.n, base.hits, base.tok_hit, base.m0, base.full);
        assert_eq!(run(10), want);
        assert_eq!(run(64), want);
        let (r4, _) = replay(&prompt, &gen, CopySpanCfg::new(16).with_start_w(4), 10, true);
        assert_eq!(run(4), format!("COPY_SPAN_REPLAY_TOTAL N={} hits={} tok_hit={} m0={} full={}", r4.n, r4.hits, r4.tok_hit, r4.m0, r4.full));
        assert_ne!(run(4), want, "the ramp splits the copy's first rounds");
        let _ = std::fs::remove_file(&path);
    }

    /// Cost on one core: an 8K-token prompt (a code-like file with repeated lines), a 4K-token
    /// output that copies it with an edit every ~400 tokens, then 1K tokens of unrelated text.
    /// Run: cargo test --release copy_span_cost -- --nocapture
    #[test]
    fn copy_span_cost_8k_prompt() {
        let mut rng = Lcg(2026);
        let mut prompt: Vec<u32> = Vec::with_capacity(8192);
        let lines: Vec<Vec<u32>> = (0..200).map(|_| (0..(4 + rng.below(20))).map(|_| rng.below(50_000) as u32).collect()).collect();
        while prompt.len() < 8192 {
            if rng.below(4) == 0 {
                let l = &lines[rng.below(lines.len() as u64) as usize];
                prompt.extend(l);
            } else {
                prompt.push(rng.below(150_000) as u32);
            }
        }
        prompt.truncate(8192);
        let mut gen: Vec<u32> = Vec::new();
        let mut i = 100;
        while gen.len() < 4096 && i < prompt.len() {
            gen.push(prompt[i]);
            i += 1;
            if rng.below(400) == 0 {
                gen.push(200_000 + rng.below(1000) as u32);
            }
        }
        gen.extend((0..1024).map(|_| 160_000 + rng.below(30_000) as u32));
        for &min_n in &[16usize, 4] {
            let c = CopySpanCfg::new(min_n);
            let t0 = std::time::Instant::now();
            let mut cs = CopySpan::new(&prompt, c);
            let build = t0.elapsed();
            let (mut tp, mut tc) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
            let (mut np, mut ns, mut nt) = (0u64, 0u64, 0u64);
            let mut worst_p = std::time::Duration::ZERO;
            let mut pos = 0;
            while pos < gen.len() {
                let a = std::time::Instant::now();
                let p = cs.propose(10);
                let d = a.elapsed();
                tp += d;
                worst_p = worst_p.max(d);
                np += 1;
                let (m, e) = match &p {
                    Some(p) => {
                        let m = p.tokens.iter().zip(&gen[pos..]).take_while(|(a, b)| a == b).count();
                        (m, (m + 1).min(gen.len() - pos))
                    }
                    None => (0, 1),
                };
                ns += (p.is_none() || matches!(p.as_ref().unwrap().source, Source::Search { .. })) as u64;
                let b = std::time::Instant::now();
                cs.commit(p.as_ref(), m, &gen[pos..pos + e]);
                tc += b.elapsed();
                nt += e as u64;
                pos += e;
            }
            println!("COPY_SPAN_COST min_n={min_n} prompt=8192 build={:.3} ms ({:.3} us/token); propose {:.3} us/call over {np} calls ({ns} searched, worst {:.1} us); commit {:.3} us/committed token over {nt} tokens",
                     build.as_secs_f64() * 1e3, build.as_secs_f64() * 1e6 / 8192.0,
                     tp.as_secs_f64() * 1e6 / np as f64, worst_p.as_secs_f64() * 1e6, tc.as_secs_f64() * 1e6 / nt as f64);
            assert!(tp.as_secs_f64() / np as f64 <= 200e-6, "propose over the 0.2 ms round budget");
        }
    }

    /// ULTRA2 merge: `peek` applies the same yield rule as `propose` when COPY27B's cost-ratio rule is set (a gate
    /// state the EXL3 rule allows and the ratio rule denies: both say no copy).
    #[test]
    fn copy_span_peek_follows_the_ratio_rule() {
        let prompt: Vec<u32> = (0..40u32).chain(0..20u32).collect();
        let mut cs = CopySpan::new(&prompt, CopySpanCfg::new(16).with_yield_gate(1).with_yield_ratio(256));
        cs.gate.note_neural(6);
        cs.gate.note_copy(5);
        cs.gate.note_copy(5);
        assert!(cs.gate.rule_allows(1) && !cs.gate.ratio_allows(256), "the case under test");
        assert_eq!(cs.peek(15), None);
        assert!(cs.propose(15).is_none() && cs.last_miss() == Some(MissReason::YieldGate));
        // the EXL3 configuration (ratio 0) on the same gate state: both copy
        let mut ex = CopySpan::new(&prompt, CopySpanCfg::new(16).with_yield_gate(1));
        ex.gate.note_neural(6);
        ex.gate.note_copy(5);
        ex.gate.note_copy(5);
        let pk = ex.peek(15);
        assert!(pk.is_some());
        assert_eq!(pk, ex.propose(15).map(|p| (p.tokens.len(), p.source == Source::Cursor)));
    }

    /// COPYLANES: `peek` agrees with the next `propose` (length and source) at every step of a copy with a ramp, a
    /// mismatch and a re-search, and changes nothing (counters, cursor, gate).
    #[test]
    fn copy_span_peek_agrees_with_propose_and_mutates_nothing() {
        let src = uniq(1000, 300);
        let mut prompt = uniq(1, 10);
        prompt.extend(&src);
        prompt.extend(uniq(50, 10));
        for cfg in [CopySpanCfg::new(16), CopySpanCfg::new(16).with_start_w(4), CopySpanCfg::new(8).with_start_w(2).with_yield_gate(10)] {
            let mut cs = CopySpan::new(&prompt, cfg);
            let mut i = 0usize;
            let mut edits = 0;
            while i + 20 < src.len() {
                for w in [1usize, 5, 10, 15] {
                    let before = (*cs.counters(), cs.cursor(), *cs.gate());
                    let pk = cs.peek(w);
                    assert_eq!(before, (*cs.counters(), cs.cursor(), *cs.gate()), "peek changed the proposer");
                    if w == 15 {
                        let p = cs.propose(w);
                        assert_eq!(pk, p.as_ref().map(|p| (p.tokens.len(), p.source == Source::Cursor)), "peek vs propose at {i}");
                        match p {
                            Some(p) if edits < 3 && i % 37 == 5 => {
                                // the model writes something else after 2 tokens
                                let a = 2.min(p.tokens.len());
                                let mut e: Vec<u32> = p.tokens[..a].to_vec();
                                e.push(9_000_000 + i as u32);
                                cs.commit(Some(&p), a, &e);
                                edits += 1;
                                i += a + 1;
                            }
                            Some(p) => {
                                let a = p.tokens.len();
                                let take = (i + a + 1).min(src.len());
                                cs.commit(Some(&p), a, &src[i..take]);
                                i = take;
                            }
                            None => {
                                cs.commit(None, 0, &src[i..i + 1]);
                                i += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    /// COPYLANES: `commit_follow` keeps a live copy alive across rounds that used no proposal when their tokens are the
    /// source's next tokens; anything else is exactly `commit(None, ..)`.
    #[test]
    fn copy_span_commit_follow_keeps_a_copy_alive() {
        let src = uniq(1000, 200);
        let mut prompt = uniq(1, 10);
        prompt.extend(&src);
        prompt.extend(uniq(50, 10));
        let mut cs = CopySpan::new(&prompt, CopySpanCfg::new(16).with_start_w(4));
        cs.commit(None, 0, &src[..16]);
        let p = cs.propose(15).expect("16-token match");
        assert_eq!(p.tokens.len(), 4, "start ramp");
        cs.commit(Some(&p), 4, &src[16..21]);
        assert_eq!(cs.cursor(), Some(10 + 21));
        assert_eq!(cs.cursor_run(), 5);
        // two neural rounds (packed / shared) wrote the next 7 source tokens: the copy follows them
        cs.commit_follow(&src[21..28]);
        assert_eq!(cs.cursor(), Some(10 + 28));
        assert_eq!(cs.cursor_run(), 12);
        assert_eq!(cs.counters().end_unused, 0);
        let p = cs.propose(15).unwrap();
        assert_eq!(p.source, Source::Cursor);
        assert_eq!(p.tokens, src[28..36], "the ramp continues where it was (8)");
        cs.commit(Some(&p), 8, &src[28..37]);
        // a neural round that left the source: exactly commit(None, ..)
        let mut a = CopySpan::new(&prompt, CopySpanCfg::new(16).with_start_w(4));
        let mut b = CopySpan::new(&prompt, CopySpanCfg::new(16).with_start_w(4));
        for cs in [&mut a, &mut b] {
            cs.commit(None, 0, &src[..16]);
            let p = cs.propose(15).unwrap();
            cs.commit(Some(&p), 4, &src[16..21]);
        }
        let off = [src[21], src[22], 4242];
        a.commit_follow(&off);
        b.commit(None, 0, &off);
        assert_eq!(a.cursor(), None);
        assert_eq!(a.counters(), b.counters());
        assert_eq!(a.ctx(), b.ctx());
        assert_eq!(a.peek(15), b.peek(15));
        // the tokens reach the end of the source: no next source token, exactly commit(None, ..)
        let mut c = CopySpan::new(&prompt, CopySpanCfg::new(16));
        c.commit(None, 0, &src[..16]);
        let p = c.propose(15).unwrap();
        c.commit(Some(&p), 15, &src[16..32]);
        let mut rest: Vec<u32> = prompt[10 + 32..].to_vec();
        rest.extend(&src[..32]); // (the context holds the generated tokens too: run to its very end)
        c.commit_follow(&rest);
        assert_eq!(c.cursor(), None);
        assert_eq!(c.counters().end_unused, 1);
        // nothing emitted: no change
        let before = *c.counters();
        c.commit_follow(&[]);
        assert_eq!(before, *c.counters());
    }
