//! Unit tests (module `stop_info_tests`), moved out of the old inline block in `exl3_serve.rs`; the test text is unchanged.
//! The module is declared in `mod.rs`; `use super::*` there resolves against the parent module as before.

    use super::*;
    use crate::exl3_forward::{stop_token_stats, SampParams};

    fn row(logits: &[f32]) -> Vec<u16> { logits.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect() }
    fn near(a: f64, b: f64) -> bool { (a - b).abs() < 2e-3 } // f16 logits: three decimals
    fn samp_opt(t: f32, top_p: f32, top_k: usize, min_p: f32) -> Option<SampParams> {
        SampParams::sampled(t, top_p, top_k, min_p, 0, 0)
    }

    /// The S-B9-REL-SMALLFIXES computation, kept verbatim as the reference `stop_token_stats` must
    /// reproduce BIT FOR BIT (SURFACE2 item 2 replaced its full sort and its per-token exp calls).
    fn stop_token_stats_reference(row: &[u16], tok: usize, samp: Option<&SampParams>) -> Option<(f64, u32)> {
        let lg = |b: u16| half::f16::from_bits(b).to_f32() as f64;
        let xt = lg(*row.get(tok)?);
        let (mut lmax, mut rank) = (f64::NEG_INFINITY, 1u32);
        for &b in row {
            let x = lg(b);
            if x > lmax { lmax = x; }
            if x > xt { rank += 1; }
        }
        if !xt.is_finite() || !lmax.is_finite() { return None; }
        let Some(s) = samp else {
            let z: f64 = row.iter().map(|&b| (lg(b) - lmax).exp()).filter(|w| w.is_finite()).sum();
            return Some(((xt - lmax).exp() / z, rank));
        };
        let t = (s.temperature as f64).max(1e-6);
        let w = |x: f64| ((x - lmax) / t).exp();
        let tau = if s.top_k > 0 && (s.top_k as usize) < row.len() {
            let mut xs: Vec<f64> = row.iter().map(|&b| lg(b)).filter(|x| !x.is_nan()).collect();
            let k = (s.top_k as usize - 1).min(xs.len() - 1);
            *xs.select_nth_unstable_by(k, |a, b| b.total_cmp(a)).1
        } else {
            f64::NEG_INFINITY
        };
        let mfloor = if s.min_p > 0.0 { lmax + t * (s.min_p as f64).ln() } else { f64::NEG_INFINITY };
        let mut cand: Vec<f64> = row.iter().map(|&b| lg(b)).filter(|&x| x >= tau && x >= mfloor).collect();
        let mass: f64 = cand.iter().map(|&x| w(x)).sum();
        let mut theta = f64::NEG_INFINITY;
        if s.top_p < 1.0 && mass > 0.0 {
            cand.sort_unstable_by(|a, b| b.total_cmp(a));
            let (target, mut cum, mut i) = (s.top_p as f64 * mass, 0.0f64, 0usize);
            while i < cand.len() {
                let x = cand[i];
                while i < cand.len() && cand[i] == x { cum += w(x); i += 1; }
                theta = x;
                if cum >= target { break; }
            }
        }
        if !(xt >= tau && xt >= mfloor && xt >= theta) { return Some((0.0, rank)); }
        let z: f64 = cand.iter().filter(|&&x| x >= theta).map(|&x| w(x)).sum();
        Some((w(xt) / z, rank))
    }

    #[test]
    fn a_greedy_row_reports_the_models_probability_and_rank() {
        let r = row(&[1.0, 2.0, 3.0, 0.0]);
        let z = (1f64.exp() + 2f64.exp() + 3f64.exp() + 1.0);
        let (p, rank) = stop_token_stats(&r, 2, None).unwrap();
        assert!(near(p, 3f64.exp() / z) && rank == 1, "{p} {rank}");
        let (p, rank) = stop_token_stats(&r, 3, None).unwrap();
        assert!(near(p, 1.0 / z) && rank == 4, "{p} {rank}");
        // ties share the better rank (rank = 1 + the number of STRICTLY greater logits)
        let t = row(&[5.0, 5.0, 1.0]);
        assert_eq!(stop_token_stats(&t, 0, None).unwrap().1, 1);
        assert_eq!(stop_token_stats(&t, 1, None).unwrap().1, 1);
        assert_eq!(stop_token_stats(&t, 2, None).unwrap().1, 3);
        // out of range, or a masked token
        assert_eq!(stop_token_stats(&r, 4, None), None);
        assert_eq!(stop_token_stats(&row(&[1.0, f32::NEG_INFINITY]), 1, None), None);
    }

    #[test]
    fn a_sampled_row_reports_the_probability_in_the_law_the_sampler_draws_from() {
        let samp = |t: f32, top_p: f32, top_k: usize, min_p: f32| SampParams::sampled(t, top_p, top_k, min_p, 7, 0).unwrap();
        // temperature only: softmax(l / T)
        let r = row(&[0.0, 1.0, 2.0]);
        let s = samp(0.5, 1.0, 0, 0.0);
        let z = 0f64.exp() + 2f64.exp() + 4f64.exp();
        let (p, rank) = stop_token_stats(&r, 1, Some(&s)).unwrap();
        assert!(near(p, 2f64.exp() / z) && rank == 2, "{p} {rank}");
        // top-k = 2 over [3, 2, 1, 0]: the third token is truncated away (p = 0), its rank stays 3
        let r = row(&[3.0, 2.0, 1.0, 0.0]);
        let s = samp(1.0, 1.0, 2, 0.0);
        let (p, rank) = stop_token_stats(&r, 1, Some(&s)).unwrap();
        assert!(near(p, 2f64.exp() / (3f64.exp() + 2f64.exp())) && rank == 2, "{p} {rank}");
        assert_eq!(stop_token_stats(&r, 2, Some(&s)).unwrap(), (0.0, 3));
        // top-k keeps every tie of the k-th largest (the kernel's rule)
        let r = row(&[3.0, 2.0, 2.0, 0.0]);
        let (p, _) = stop_token_stats(&r, 2, Some(&s)).unwrap();
        assert!(near(p, 2f64.exp() / (3f64.exp() + 2.0 * 2f64.exp())), "{p}");
        // top-p 0.7 over probabilities .5 / .3 / .2: the prefix .5 + .3 reaches it; .2 is cut
        let r = row(&[0.5f32.ln(), 0.3f32.ln(), 0.2f32.ln()]);
        let s = samp(1.0, 0.7, 0, 0.0);
        let (p, rank) = stop_token_stats(&r, 1, Some(&s)).unwrap();
        assert!(near(p, 0.3 / 0.8) && rank == 2, "{p} {rank}");
        assert_eq!(stop_token_stats(&r, 2, Some(&s)).unwrap(), (0.0, 3));
        // min_p 0.2 over .6 / .3 / .1: the floor is 0.2 * 0.6 = 0.12, so .1 is cut
        let r = row(&[0.6f32.ln(), 0.3f32.ln(), 0.1f32.ln()]);
        let s = samp(1.0, 1.0, 0, 0.2);
        let (p, _) = stop_token_stats(&r, 1, Some(&s)).unwrap();
        assert!(near(p, 0.3 / 0.9), "{p}");
        assert_eq!(stop_token_stats(&r, 2, Some(&s)).unwrap().0, 0.0);
        // every kept probability sums to 1 (top-k 3 + top-p 0.9 + min_p together)
        let r = row(&[2.0, 1.5, 1.0, 0.5, 0.0, -1.0]);
        let s = samp(0.8, 0.9, 4, 0.05);
        let total: f64 = (0..6).map(|t| stop_token_stats(&r, t, Some(&s)).unwrap().0).sum();
        assert!((total - 1.0).abs() < 1e-9, "{total}");
    }

    /// A greedy token is its row's argmax: any other rank means the wrong row was read back.
    #[test]
    fn a_greedy_row_that_does_not_put_its_token_first_is_not_reported() {
        assert_eq!(stop_info_checked(Some((0.9, 1)), true), Some((0.9, 1)));
        assert_eq!(stop_info_checked(Some((0.01, 7)), true), None, "greedy + rank 7 = not the row that chose the token");
        assert_eq!(stop_info_checked(Some((0.01, 7)), false), Some((0.01, 7)), "a sample can have any rank");
        assert_eq!(stop_info_checked(Some((0.0, 40)), false), Some((0.0, 40)));
        assert_eq!(stop_info_checked(None, true), None);
        assert_eq!(stop_info_checked(Some((f64::NAN, 1)), false), None);
    }

    /// The readback is on the finishing step only: the two call sites sit behind `reason == "stop"`
    /// inside the finish branch, and nothing else in the scheduler calls it (source pin: the unit
    /// suite cannot run a step).
    #[test]
    fn the_readback_runs_only_when_a_request_finishes_on_a_stop_token() {
        let src = src_text!("exl3_serve");
        // S-B9-CF-PACK4: the call gained a `packed` argument and step_mtp's emit loop moved into round_emit
        let call = format!("self.send_{}(s, ", "stop_info");
        assert_eq!(src.matches(call.as_str()).count(), 2, "round_emit (step_mtp's emit loop, packed or not) and step_plain");
        // PACK4-REVIEW B4: the ROW argument is pinned again (the pin had been loosened to the call's prefix): a
        // packed round reads the stop token's row at the lane's row offset + its index, a serial round row j
        let (packed_call, plain_call) = (format!("self.send_{}(s, packed_row0.map_or(j, |r0| r0 + j), t, packed_row0.is_some())", "stop_info"),
                                         format!("self.send_{}(s, j, t, false)", "stop_info"));
        assert_eq!(src.matches(packed_call.as_str()).count(), 1, "round_emit reads the packed row r0 + j");
        assert_eq!(src.matches(plain_call.as_str()).count(), 1, "step_plain reads row j of the shared step");
        for (i, _) in src.match_indices(call.as_str()) {
            let line_start = src[..i].rfind('\n').unwrap() + 1;
            let line = &src[line_start..i];
            assert!(line.contains("reason == \"stop\""), "a call not guarded by the stop reason: `{}`", line.trim());
        }
        let rb = format!(".logits_{}(", "row");
        assert_eq!(src.matches(rb.as_str()).count(), 2, "two readback sites in the scheduler (send_stop_info, send_logprob)");
        // send_logprob runs only for a request that asked for log probabilities, and never under TP
        let lp = format!("self.send_{}(", "logprob");
        assert_eq!(src.matches(lp.as_str()).count(), 3, "the plain step, the prefill's first token, a forced token of a speculative round");
        for (i, _) in src.match_indices(lp.as_str()) {
            let before = &src[src[..i].rfind("\n        ").unwrap_or(0)..i];
            let ctx = &src[i.saturating_sub(400)..i];
            assert!(ctx.contains("logprobs") && ctx.contains("self.lp_reader()"),
                    "a send_logprob call not gated on the request's logprobs and the reading rank: `{}`", before.trim());
        }
        // logprobs with speculation: ONE batched readback per round, only for a lane that asked for them, only at TP=1
        let rbs = format!(".logits_{}(", "rows");
        assert_eq!(src.matches(rbs.as_str()).count(), 1, "one batched readback site (round_logprobs)");
        let rl = format!("self.round_{}(", "logprobs");
        assert_eq!(src.matches(rl.as_str()).count(), 1, "round_emit calls round_logprobs once per round");
        let i = src.find(rl.as_str()).unwrap();
        let ctx = &src[i.saturating_sub(300)..i];
        assert!(ctx.contains("l.logprobs") && ctx.contains("self.lp_reader()"), "round_logprobs gated on the lane's logprobs and the reading rank");
        let f = &src[src.find(format!("fn send_{}(&self", "stop_info").as_str()).expect("send_stop_info")..];
        assert!(f.find("if self.tp.is_some() { return; }").unwrap() < f.find(rb.as_str()).unwrap(), "no readback under TP");
    }

    /// A vocabulary-sized logits row shaped like a served one: a bulk around `mean` with spread
    /// `sd` (f16-rounded, so many exact ties), a few dozen plausible tokens above it and one clear
    /// winner at index 7. `peaked` false = a flat row (a wide nucleus: the top-p loop runs long).
    fn synth_row(seed: u64, n: usize, mean: f32, sd: f32, peaked: bool) -> Vec<u16> {
        let mut x = seed | 1;
        let mut u = move || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x >> 11) as f64 / (1u64 << 53) as f64 };
        let mut v: Vec<f32> = (0..n).map(|_| {
            let g = (-2.0 * u().max(1e-300).ln()).sqrt() * (std::f64::consts::TAU * u()).cos(); // Box-Muller
            mean + sd * g as f32
        }).collect();
        if peaked {
            for j in 0..48 { v[(j * 5003 + 11) % n] = 6.0 + 9.0 * (u() as f32); }
            v[7] = 21.5;
            v[9] = 19.0;
        }
        v.iter().map(|&f| half::f16::from_f32(f).to_bits()).collect()
    }

    /// SURFACE2 item 2: `stop_token_stats` no longer sorts the vocabulary nor takes one exp per
    /// token; (p, rank) must equal the first implementation's BIT FOR BIT, on served-shape rows
    /// and on the awkward ones (heavy ties, both zeros, infinities, NaN, masked rows, tiny rows).
    #[test]
    fn stop_token_stats_is_bit_identical_to_the_reference() {
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let mut rnd = move || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x };
        let bits = |r: Option<(f64, u32)>| r.map(|(p, k)| (p.to_bits(), k));
        let mut rows: Vec<Vec<u16>> = vec![
            synth_row(3, 248_320, -1.5, 2.6, true),
            synth_row(4, 248_320, 0.0, 3.0, false),
            synth_row(5, 40_000, 0.0, 0.02, false),                   // nearly every logit a tie
            row(&[0.0, -0.0, 0.0, -0.0, -1.0, -0.0]),                 // the maximum is a zero of either sign
            row(&[-0.0, 0.0, -3.0]),
            row(&[2.0, f32::NEG_INFINITY, 2.0, 1.0, f32::NEG_INFINITY, f32::NAN, 0.5]),
            row(&[f32::NAN, f32::NAN]),
            row(&[f32::NEG_INFINITY, f32::NEG_INFINITY]),
            row(&[1.0, f32::INFINITY]),
            row(&[3.25]),
            row(&[]),
        ];
        for i in 0..40 { // small rows from a small alphabet: ties, zeros and specials everywhere
            let n = 1 + (rnd() % 60) as usize;
            let vals = [0.0f32, -0.0, 1.0, 1.5, -2.0, 7.25, -7.25, 1e-3, f32::NEG_INFINITY, f32::NAN, 30.0, 0.5];
            let lim = if i % 4 == 0 { vals.len() } else { 8 };
            rows.push((0..n).map(|_| half::f16::from_f32(vals[(rnd() % lim as u64) as usize]).to_bits()).collect());
        }
        for _ in 0..3 { rows.push((0..3000).map(|_| (rnd() >> 13) as u16).collect()); } // raw f16 patterns
        let samplers: Vec<Option<SampParams>> = {
            let mut v = vec![None];
            // (an infinite temperature is not refused by the API: with a -inf logit the weight is NaN)
            for &t in &[1.0f32, 0.7, 0.05, 2.5, 1e-9, f32::INFINITY, 3e38] {
                for &top_p in &[1.0f32, 0.95, 0.8, 0.5, 1e-6, 0.999_999] {
                    for &top_k in &[0usize, 1, 2, 20, 64, 5000, usize::MAX] {
                        for &min_p in &[0.0f32, 0.05, 0.5, 1.0, 1.5] {
                            v.push(Some(SampParams { temperature: t, top_p, top_k: top_k.min(u32::MAX as usize) as u32,
                                                     min_p, seed: 0, counter: 0 }));
                        }
                    }
                }
            }
            v
        };
        let mut checked = 0usize;
        for (ri, r) in rows.iter().enumerate() {
            // the top token, a few random ones, the last one, and one past the end
            let top = (0..r.len()).max_by(|&a, &b| half::f16::from_bits(r[a]).to_f32().total_cmp(&half::f16::from_bits(r[b]).to_f32())).unwrap_or(0);
            let mut toks = vec![top, 0, r.len().saturating_sub(1), r.len()];
            for _ in 0..3 { toks.push((rnd() % r.len().max(1) as u64) as usize); }
            // every row takes a slice of the sampler grid, offset by the row index so the rows
            // cover all of it between them (the reference sorts 248K values per call on the big
            // rows; the new function's histogram is a fixed ~70 us per call on the small ones)
            let stride = if r.len() > 10_000 { 11 } else { 9 };
            for (si, sp) in samplers.iter().enumerate().filter(|(si, _)| (si + ri) % stride == 0) {
                for &tok in if r.len() > 10_000 { &toks[..2] } else { &toks[..] } {
                    let (got, want) = (stop_token_stats(r, tok, sp.as_ref()), stop_token_stats_reference(r, tok, sp.as_ref()));
                    assert_eq!(bits(got), bits(want), "row {ri} (len {}), sampler {si} {sp:?}, tok {tok}: {got:?} vs {want:?}", r.len());
                    checked += 1;
                }
            }
        }
        assert!(checked > 40_000, "{checked} comparisons");
    }

    /// SURFACE2 item 2 instrument (run by hand: `cargo test --release --lib stop_token_stats_cost --
    /// --ignored --nocapture`): the host pass of a finishing step, per sampler setting, on a
    /// vocabulary-sized row. Prints min / median of 15 runs in ms.
    #[test]
    #[ignore]
    fn stop_token_stats_cost() {
        const V: usize = 248_320;
        let time = |f: &dyn Fn() -> Option<(f64, u32)>| -> (f64, f64, Option<(f64, u32)>) {
            let mut ts: Vec<f64> = (0..15).map(|_| { let t = std::time::Instant::now(); std::hint::black_box(f()); t.elapsed().as_secs_f64() * 1e3 }).collect();
            ts.sort_by(|a, b| a.total_cmp(b));
            (ts[0], ts[7], f())
        };
        for (shape, row) in [("peaked", synth_row(1, V, -1.5, 2.6, true)), ("flat", synth_row(2, V, 0.0, 3.0, false))] {
            let distinct = row.iter().collect::<std::collections::HashSet<_>>().len();
            crate::dprintln!("row {shape}: {V} logits, {distinct} distinct f16 values");
            let cases: [(&str, Option<SampParams>, usize); 7] = [
                ("greedy (softmax at T=1)", None, 7),
                ("T=0.7 top_k=20 top_p=0.8", samp_opt(0.7, 0.8, 20, 0.0), 7),
                ("T=0.7 top_k=20 top_p=1.0", samp_opt(0.7, 1.0, 20, 0.0), 7),
                ("T=1.0 top_k=0  top_p=1.0", samp_opt(1.0, 1.0, 0, 0.0), 7),
                ("T=0.7 top_k=0  top_p=0.95", samp_opt(0.7, 0.95, 0, 0.0), 7),
                ("T=1.0 top_k=0  top_p=0.95", samp_opt(1.0, 0.95, 0, 0.0), 9),
                ("T=1.0 top_k=0  top_p=0.95 min_p=0.05", samp_opt(1.0, 0.95, 0, 0.05), 7),
            ];
            for (name, sp, tok) in cases {
                let (new_min, new_med, got) = time(&|| stop_token_stats(&row, tok, sp.as_ref()));
                let (old_min, old_med, want) = time(&|| stop_token_stats_reference(&row, tok, sp.as_ref()));
                assert_eq!(got.map(|(p, r)| (p.to_bits(), r)), want.map(|(p, r)| (p.to_bits(), r)), "{name}: not bit-exact");
                crate::dprintln!("  {name:<38} reference min {old_min:7.3} med {old_med:7.3} ms | now min {new_min:7.3} med {new_med:7.3} ms | (p, rank) = {got:?}");
            }
        }
    }
