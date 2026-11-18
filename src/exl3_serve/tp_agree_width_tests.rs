//! Unit tests (module `tp_agree_width_tests`), moved out of the old inline block in `exl3_serve.rs`; the test text is unchanged.
//! The module is declared in `mod.rs`; `use super::*` there resolves against the parent module as before.

    use super::*;

    type Said = (usize, usize, u32); // (accept, verify width, digest) as one rank passes them to tp_agree_eq

    /// What the two ranks of a TP=2 pair decide at one lockstep point: (this rank, the peer).
    fn pair(step: u64, a: Said, b: Said) -> (std::result::Result<(), String>, std::result::Result<(), String>) {
        let token = |s: Said| tp_agree_fields(s.0, s.1).map(|(a8, k8)| (a8, k8, crate::net::agree_token(step, a8, k8, s.2)));
        let side = |me: Said, other: Said| -> std::result::Result<(), String> {
            let (a8, k8, _) = token(me)?; // this rank refused its own fields: it aborts the link
            // a peer that refused never publishes a token: this rank's net_agree ends in an aborted link (None)
            let reply = token(other).ok().map(|(_, _, t)| crate::net::agree_token_reply(t));
            tp_agree_decide(a8, k8, me.2, reply)
        };
        (side(a, b), side(b, a))
    }

    /// Digests of the kind the lockstep points pass: FNVs of short id lists, plus the corner words.
    fn digests() -> Vec<u32> {
        let mut v = vec![0u32, u32::MAX, 0x8000_0000, 0x7800_0000, 0x0800_0000, 0xF800_0000, 0x07FF_FFFF, 1];
        v.extend((0..32).map(|b| 1u32 << b));
        for n in 0..24u32 {
            v.push(fnv32(0, (0..n).map(|i| 1000 + 7 * i)));
            v.push(fnv32(0x5EED, (0..n).map(|i| 31 * i + n)));
        }
        v
    }

    #[test]
    fn the_digest_can_set_and_clear_every_bit_of_its_word() {
        // the bits `h` can set: ALL 32. fnv32 is a full-width hash, so no bit of the 32-bit word is free for
        // a width unless the word reserves it (which is what the next test demands).
        let (mut any, mut all) = (0u32, u32::MAX);
        for n in 0..200u32 {
            let h = fnv32(0, (0..n % 17 + 1).map(|i| n.wrapping_mul(2654435761).wrapping_add(i)));
            any |= h;
            all &= h;
        }
        assert_eq!(any, u32::MAX, "the digest never set bits {:#010x}", !any);
        assert_eq!(all, 0, "the digest never cleared bits {all:#010x}");
    }

    #[test]
    fn the_width_owns_the_top_five_bits_whatever_the_digest_holds() {
        const DIGEST_MASK: u32 = (1 << 27) - 1;
        for h in digests() {
            for k in 0..=31u8 {
                let w = crate::net::agree_word(k, h);
                assert_eq!(w >> 27, k as u32, "width {k}, digest {h:#010x}: the word's top five bits are {:#04x}", w >> 27);
                assert_eq!(w & DIGEST_MASK, crate::net::agree_word(0, h) & DIGEST_MASK,
                           "width {k} changed the digest part of the word (digest {h:#010x})");
            }
            // the 27-bit digest part still hears every one of the 32 digest bits
            for b in 0..32 {
                assert_ne!(crate::net::agree_word(0, h ^ (1 << b)), crate::net::agree_word(0, h),
                           "digest bit {b} of {h:#010x} does not reach the word");
            }
        }
    }

    #[test]
    fn equal_inputs_agree_and_every_width_or_accept_difference_is_refused_0_to_17() {
        let step = agree_field(41, AGREE_A);
        let mut silent: Vec<String> = Vec::new(); // differing inputs that BOTH ranks accepted
        let mut loud: Vec<String> = Vec::new(); // equal inputs a rank refused
        for h in [0u32, 0xDEAD_BEEF, u32::MAX, 0x7800_0000, fnv32(0, 0..16), fnv32(0, 0..15)] {
            for (wa, wb) in (0..=17usize).flat_map(|a| (0..=17usize).map(move |b| (a, b))) {
                for (aa, ab) in (0..=17usize).flat_map(|a| (0..=17usize).map(move |b| (a, b))) {
                    let (ra, rb) = pair(step, (aa, wa, h), (ab, wb, h));
                    let same = wa == wb && aa == ab;
                    match (same, ra.is_ok() && rb.is_ok()) {
                        (true, false) => loud.push(format!("width {wa} accept {aa}: {ra:?} / {rb:?}")),
                        (false, true) => silent.push(format!("(width {wa}, accept {aa}) vs (width {wb}, accept {ab})")),
                        _ => {}
                    }
                }
            }
        }
        assert!(loud.is_empty(), "equal inputs must agree; refused: {:?}", &loud[..loud.len().min(6)]);
        let widths: std::collections::BTreeSet<(usize, usize)> = silent.iter()
            .filter_map(|s| { let n: Vec<usize> = s.split(|c: char| !c.is_ascii_digit()).filter_map(|t| t.parse().ok()).collect();
                              (n.len() == 4 && n[0] < n[2]).then(|| (n[0], n[2])) }).collect();
        assert!(silent.is_empty(), "{} differing (width, accept) pairs agreed in silence; width pairs that alias: {widths:?}; first: {:?}",
                silent.len(), &silent[..silent.len().min(4)]);
    }

    #[test]
    fn no_digest_difference_can_cancel_a_width_difference() {
        let step = agree_field(7, AGREE_PV);
        let mut silent: Vec<String> = Vec::new();
        let mut x = 0x9E37_79B9u32;
        for (ka, kb) in (0..=17usize).flat_map(|a| (0..=17usize).map(move |b| (a, b))).filter(|(a, b)| a != b) {
            for ha in digests() {
                // the exact differences that cancel a width XOR-folded at bit 27 (4-bit and 5-bit folds), every
                // other top-five-bit difference, and a spread of unrelated digests
                let mut others: Vec<u32> = vec![ha ^ ((((ka ^ kb) & 0xF) as u32) << 27), ha ^ ((((ka ^ kb) & 0x1F) as u32) << 27),
                                                ha ^ ((((ka.min(15) ^ kb.min(15)) & 0xF) as u32) << 27)];
                others.extend((1..32u32).map(|d| ha ^ (d << 27)));
                for _ in 0..8 { x ^= x << 13; x ^= x >> 17; x ^= x << 5; others.push(x); }
                for hb in others {
                    let (ra, rb) = pair(step, (3, ka, ha), (3, kb, hb));
                    if ra.is_ok() || rb.is_ok() {
                        silent.push(format!("width {ka} digest {ha:#010x} vs width {kb} digest {hb:#010x}"));
                    }
                }
            }
        }
        assert!(silent.is_empty(), "{} (width, digest) pairs with DIFFERENT widths produced equal words; first: {:?}",
                silent.len(), &silent[..silent.len().min(4)]);
    }

    #[test]
    fn a_value_the_token_cannot_carry_is_refused_not_clamped() {
        for a in 0..=255usize {
            for k in 0..=31usize {
                assert_eq!(tp_agree_fields(a, k), Ok((a as u8, k as u8)), "accept {a}, width {k} must ride the token exactly");
            }
        }
        for (a, k) in [(0usize, 32usize), (0, 33), (0, 255), (0, 256), (256, 0), (257, 16), (usize::MAX, 16), (3, usize::MAX)] {
            let e = tp_agree_fields(a, k).expect_err(&format!("accept {a}, width {k} cannot ride the token: it must be refused"));
            assert!(e.contains("REFUSED"), "{e}");
        }
        // a rank that refuses takes its peer down with it (the peer sees the aborted link), never one-sided
        let (ra, rb) = pair(agree_field(9, AGREE_A), (0, 32, 5), (0, 16, 5));
        assert!(ra.is_err() && rb.is_err(), "{ra:?} / {rb:?}");
        let (ra, rb) = pair(agree_field(9, AGREE_A), (256, 4, 5), (0, 4, 5));
        assert!(ra.is_err() && rb.is_err(), "{ra:?} / {rb:?}");
    }

    #[test]
    fn the_token_fields_do_not_overlap_and_the_reply_reads_them_back() {
        for (counter, kind) in [(0u64, AGREE_A), (1, AGREE_PV), (0x3F_FFFF, AGREE_B), (0x40_0000, AGREE_GO), (12345, AGREE_A)] {
            let step = agree_field(counter, kind);
            assert!(step <= 0xFF_FFFF, "the step field is 24 bits");
            for a in [0u8, 1, 15, 16, 17, 255] {
                for k in [0u8, 1, 15, 16, 17, 31] {
                    for h in [0u32, u32::MAX, 0xDEAD_BEEF] {
                        let t = crate::net::agree_token(step, a, k, h);
                        assert_eq!(t >> 40, step, "step field");
                        assert_eq!(crate::net::agree_token_reply(t), (a, crate::net::agree_word(k, h)));
                    }
                }
            }
        }
        // the boot agree's raw step word keeps its 24 bits too
        assert_eq!(crate::net::agree_token(0xB0_0700, 0, 0, 1) >> 40, 0xB0_0700);
    }

    #[test]
    fn the_step_go_fence_word_tells_step_and_event_count_apart() {
        // the pairs master's XOR word could not tell apart, and the one a folded XOR word could not
        let step = agree_field(3, AGREE_GO);
        let word = |hs: u64, hn: usize| crate::net::agree_token(step, 0, 0, go_fence_word(hs, hn));
        for ((hs_a, hn_a), (hs_b, hn_b)) in [((5u64, 0usize), (4u64, 8usize)), ((7, 0), (7 ^ (1 << 24), 1)), ((9, 0), (9, 256)),
                                             ((9, 1), (9, 257)), ((1, 0), (1 + (1 << 32), 0)), ((0, 1), (1, 0))] {
            assert_ne!(word(hs_a, hn_a), word(hs_b, hn_b), "(step {hs_a}, {hn_a} events) vs (step {hs_b}, {hn_b} events)");
        }
        // and a dense block: 64 steps x 64 event counts, all different
        let mut seen = std::collections::HashSet::new();
        for hs in 0..64u64 { for hn in 0..64usize { assert!(seen.insert(word(1000 + hs, hn)), "step {hs}, events {hn} repeats a word"); } }
    }

    #[test]
    fn a_lane_count_the_pairwise_token_cannot_name_is_refused_at_boot() {
        for w in [1usize, 4, 16, 31] { assert_eq!(tp_lane_width_fits(2, w), Ok(()), "TP=2 --max-batch {w}"); }
        for w in [32usize, 33, 64] {
            let e = tp_lane_width_fits(2, w).expect_err("TP=2 with more than 31 lanes must not boot");
            assert!(e.contains("--max-batch") && e.contains("31"), "{e}");
        }
        // TP=1 has no lockstep and the world > 2 frames carry whole u32 words: no bound from the token
        for world in [1u32, 4, 8] { assert_eq!(tp_lane_width_fits(world, 64), Ok(())); }
    }

    /// The pairwise lockstep points hand `tp_agree_eq` the real accept count and the real width: no
    /// saturation may come back between the scheduler and the token (source pin; the call sites need a
    /// live link, so they cannot be driven from a unit test).
    #[test]
    fn the_lockstep_call_sites_pass_the_real_width() {
        let src = src_text!("exl3_serve");
        let body = |from: &str, to: &str| -> &str {
            let s = &src[src.find(from).unwrap_or_else(|| panic!("{from} not found"))..];
            &s[..s.find(to).unwrap_or_else(|| panic!("{to} not found after {from}"))]
        };
        let sat15 = ["min", "(15)"].concat();
        let sat255 = ["min", "(255)"].concat();
        for (name, text) in [("pre_verify", body("    fn pre_verify(&mut self, w: usize", "impl TpSync {")),
                             ("round_hub", body("    pub(crate) fn round_hub<X", "/// TP-E: the 24-bit agree step field")),
                             ("tp_agree_eq", body("pub(crate) fn tp_agree_eq(", "fn fnv32(")),
                             ("tp_round", body("    fn tp_round(&mut self, what: &str", "        let digests = if ident"))] {
            assert!(!text.contains(&sat15), "{name} saturates a verify width at 15");
            assert!(!text.contains(&sat255), "{name} saturates an accept count at 255");
        }
        assert!(body("    fn pre_verify(&mut self, w: usize", "impl TpSync {").contains("tp_agree_eq(step, w, w + 1, h)"));
        assert!(body("    fn tp_round(&mut self, what: &str", "        let digests = if ident")
                    .contains("tp_agree_eq(agree_field(step, AGREE_A), a, t.width_said(m), h)"));
    }
