//! Host-side proof tests (module `mpk_tests`), moved out of `exl3_bench.rs`; no GPU unless a test says so.
//! Part of the `exl3_bench` probe module (see `mod.rs` for the map); test bodies are unchanged.
    const KSTEP: usize = 4;
    const KROW: usize = 768;
    const STB: usize = KSTEP * KROW;
    const BB: usize = 96;

    /// mpk_items, transcribed.
    fn items(b: usize, g: usize, n: usize, il: bool) -> (usize, usize, usize, usize) {
        let (beg, step, end) = if il { (b, g, n) } else { (b * n / g, 1, (b + 1) * n / g) };
        let cnt = if beg < end { (end - beg + step - 1) / step } else { 0 };
        (beg, step, end, cnt)
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Grp { Slot { slot: usize, item: usize, ks: usize }, A { e: usize }, Empty }

    /// One thread's cp.async group queue (groups retire FIFO; a group's smem writes land when a
    /// wait retires it). The kernel has two thread roles: ring issuers (tid < MPK_NCH: slot groups,
    /// never an A group) and A issuers (tid >= MPK_NCH: EMPTY slot groups + the A groups).
    struct Q {
        groups: Vec<Grp>,
        lo: usize,
    }
    impl Q {
        /// cp.async.wait_group keep: every group but the newest `keep` retires; returns what landed.
        fn retire(&mut self, keep: usize) -> Vec<Grp> {
            let mut out = Vec::new();
            while self.lo + keep < self.groups.len() {
                out.push(self.groups[self.lo]);
                self.lo += 1;
            }
            out
        }
        fn pending(&self) -> &[Grp] { &self.groups[self.lo..] }
    }

    /// One CTA of the persistent loop (both roles). Asserts (a) every consumed ring slot holds
    /// exactly the (item, ks) the consumer expects, (b) no copy is pending on a slot being refilled
    /// or on A while the MMAs read it, (c) A holds the consumed item's expert, (d) the A wait never
    /// retires a ring group (the ring keeps its depth across expert changes), (e) nothing real is in
    /// flight at exit. Returns the consumed (item, ks) sequence.
    fn sim_cta(b: usize, g: usize, n: usize, il: bool, nst: usize, nks: usize, t: usize) -> Vec<(usize, usize)> {
        let (beg, step, end, cnt) = items(b, g, n, il);
        let total = cnt * nks;
        let mut out = Vec::new();
        if cnt == 0 { return out; }
        let mut rq = Q { groups: Vec::new(), lo: 0 };   // a ring-issuing thread
        let mut aq = Q { groups: Vec::new(), lo: 0 };   // an A-issuing thread
        let mut ring: Vec<Option<(usize, usize)>> = vec![None; nst];
        let mut a_state: Option<usize> = None;
        let land = |v: Vec<Grp>, ring: &mut Vec<Option<(usize, usize)>>, a_state: &mut Option<usize>| {
            for gr in v {
                match gr {
                    Grp::Slot { slot, item, ks } => ring[slot] = Some((item, ks)),
                    Grp::A { e } => *a_state = Some(e),
                    Grp::Empty => {}
                }
            }
        };
        let (mut p_item, mut p_ks, mut p_slot, mut p_i) = (beg, 0usize, 0usize, 0usize);
        macro_rules! issue {
            () => {
                if p_i < total {
                    assert!(!rq.pending().iter().any(|gr| matches!(gr, Grp::Slot { slot, .. } if *slot == p_slot)),
                            "b{b}: refill of slot {p_slot} while a copy into it is pending");
                    rq.groups.push(Grp::Slot { slot: p_slot, item: p_item, ks: p_ks });
                    p_i += 1;
                    p_ks += 1;
                    if p_ks == nks { p_ks = 0; p_item += step; }
                    p_slot += 1;
                    if p_slot == nst { p_slot = 0; }
                } else {
                    rq.groups.push(Grp::Empty);
                }
                aq.groups.push(Grp::Empty);              // the A threads commit the ring's group empty
            };
        }
        for _ in 0..nst - 1 { issue!(); }
        aq.groups.push(Grp::A { e: beg / t });            // mpk_issue_a: A threads only
        let mut a_pend = true;
        let (mut c_item, mut c_ks, mut c_slot) = (beg, 0usize, 0usize);
        let mut consumed: Vec<bool> = vec![false; nst];
        for it in 0..total {
            if a_pend {
                // mpk_wait_a: the A threads wait_all (their non-empty groups are A groups only);
                // ring threads do not wait; then the barrier.
                let landed = aq.retire(0);
                assert!(landed.iter().all(|gr| !matches!(gr, Grp::Slot { .. })), "b{b}: the A wait retired a ring group");
                land(landed, &mut ring, &mut a_state);
                a_pend = false;
            }
            let l1 = rq.retire(nst - 2);                   // cp.async.wait_group NST-2, both roles
            land(l1, &mut ring, &mut a_state);
            let l2 = aq.retire(nst - 2);
            land(l2, &mut ring, &mut a_state);
            // barrier: every warp is past slot it-1 — the slot refilled now was consumed there
            if p_i < total {
                assert_eq!(p_slot, (it + nst - 1) % nst);
                if it > 0 { assert!(consumed[p_slot], "b{b} it{it}: refilling an unconsumed slot"); }
            }
            issue!();
            assert_eq!(ring[c_slot], Some((c_item, c_ks)), "b{b} it{it}: slot {c_slot} content");
            assert!(!rq.pending().iter().any(|gr| matches!(gr, Grp::Slot { slot, .. } if *slot == c_slot)),
                    "b{b} it{it}: consuming a slot with a pending copy");
            assert!(!aq.pending().iter().any(|gr| matches!(gr, Grp::A { .. })), "b{b} it{it}: A being written while the MMAs read it");
            assert_eq!(a_state, Some(c_item / t), "b{b} it{it}: A holds the wrong expert");
            out.push((c_item, c_ks));
            consumed[c_slot] = true;
            c_slot += 1;
            if c_slot == nst { c_slot = 0; }
            c_ks += 1;
            if c_ks == nks {
                // epilogue: the stage-tile barrier (A dead), then the last-reader refill
                let e = c_item / t;
                let nx = c_item + step;
                if nx < end && nx / t != e {
                    aq.groups.push(Grp::A { e: nx / t });
                    a_pend = true;
                }
                c_ks = 0;
                c_item = nx;
            }
        }
        assert!(rq.pending().iter().chain(aq.pending().iter()).all(|gr| matches!(gr, Grp::Empty)), "b{b}: real copies pending at exit");
        out
    }

    #[test]
    fn mpk_schedule_covers_every_item_once_in_order() {
        let mut cases = 0usize;
        for &(t, nks) in &[(10usize, 40usize), (20, 10)] {
            for &live in &[1usize, 2, 7, 10, 26, 47, 90, 160, 256] {
                let n = live * t;
                for &g in &[1usize, 3, 7, 48, 96, 144, n, n + 5] {
                    for &il in &[false, true] {
                        for &nst in &[4usize, 8] {
                            let mut seen = vec![0usize; n];
                            for b in 0..g {
                                let seq = sim_cta(b, g, n, il, nst, nks, t);
                                for (k, &(item, ks)) in seq.iter().enumerate() {
                                    assert_eq!(ks, k % nks, "per-item k order");
                                    if ks == 0 { seen[item] += 1; }
                                }
                            }
                            assert!(seen.iter().all(|&c| c == 1), "t{t} live{live} g{g} il{il} nst{nst}: coverage");
                            cases += 1;
                        }
                    }
                }
            }
        }
        assert!(cases > 500);
    }

    /// The ring's byte mapping: producer chunk tid -> (row g, byte r*16) of the slot, from the
    /// expert's trellis at k16 row ks*KSTEP + g, tile t; consumer warp w, step g reads the 96-B
    /// block at slot + g*KROW + w*BB. Must equal the a1b3 body's block (kb, nb_col = 8t + w).
    #[test]
    fn mpk_ring_bytes_are_the_a1b3_blocks() {
        for &nb in &[80usize, 160] {                    // gate/up N = 1280, down N = 2560
            let kb_total = 16usize;
            let row_b = nb * BB;
            let mut st = 0x1234_5678u32 ^ nb as u32;
            let trellis: Vec<u8> = (0..row_b * kb_total).map(|_| { st ^= st << 13; st ^= st >> 17; st ^= st << 5; st as u8 }).collect();
            for t in 0..nb / 8 {
                for ks in 0..kb_total / KSTEP {
                    let mut slot = vec![0u8; STB];
                    for tid in 0..256usize {
                        if tid >= STB / 16 { continue; }                 // c_live
                        let cg = tid / (KROW / 16);
                        let cr = tid - cg * (KROW / 16);
                        let src = ks * KSTEP * row_b + t * KROW + cg * row_b + cr * 16;
                        let dst = cg * KROW + cr * 16;
                        slot[dst..dst + 16].copy_from_slice(&trellis[src..src + 16]);
                    }
                    for g in 0..KSTEP {
                        for w in 0..8 {
                            let kb = ks * KSTEP + g;
                            let nb_col = t * 8 + w;
                            let blk = &trellis[kb * row_b + nb_col * BB..kb * row_b + nb_col * BB + BB];
                            let got = &slot[g * KROW + w * BB..g * KROW + w * BB + BB];
                            assert_eq!(got, blk, "nb{nb} t{t} ks{ks} g{g} w{w}");
                        }
                    }
                }
            }
        }
    }

    /// mpk_diet_consts + mpk_consume's window cut == exl3_dq_from's funnel form for every lane
    /// and B-fragment slot at 3 bits (the WP20 rung-1 diet, transcribed independently).
    #[test]
    fn mpk_diet_windows_equal_dq_from() {
        let bits = 3usize;
        let w = 8 * bits;
        let funnel = |lo: u32, hi: u32, s: u32| ((((hi as u64) << 32) | lo as u64) >> (s & 31)) as u32;
        let mut st = 0xC0FF_EE11u32;
        for rep in 0..(1 + w * 32 + 512) {
            // rep 0: zeros; then one-hot over every ring bit; then random rings
            let ring: Vec<u32> = (0..w).map(|i| {
                if rep == 0 { 0 }
                else if rep <= w * 32 { if (rep - 1) / 32 == i { 1u32 << ((rep - 1) % 32) } else { 0 } }
                else { st ^= st << 13; st ^= st >> 17; st ^= st << 5; st }
            }).collect();
            for lane in 0..32usize {
                let r = 2 * (lane & 3);
                let c = lane >> 2;
                let l = 8 * ((c >> 1) & 3) + ((r & 7) >> 1);
                let t0 = 8 * l + (r & 1) + 2 * ((r >> 3) & 1) + 4 * (c >> 3) + 32 * (c & 1);
                let b10 = (t0 + 257) * bits;
                let a = (b10 - 16) >> 5;
                let (off_a, off_b) = (a % w, (a + 1) % w);
                let sh7 = 32 * (a + 2) - (b10 + 7 * bits);
                assert!(sh7 < 32, "sh7 {sh7}");
                let p = (((ring[off_a] as u64) << 32) | ring[off_b] as u64) >> sh7;
                for h in 0..2 {
                    for q in 0..4 {
                        let j = h * 4 + q;
                        let got = ((p >> (3 * (7 - j))) & 0xFFFF) as u32;
                        let rr = 2 * (lane & 3) + (q & 1) + 8 * (q >> 1);
                        let cc = 8 * h + (lane >> 2);
                        let ll = 8 * ((cc >> 1) & 3) + ((rr & 7) >> 1);
                        let tt = 8 * ll + (rr & 1) + 2 * ((rr >> 3) & 1) + 4 * (cc >> 3) + 32 * (cc & 1);
                        let b1 = (tt + 257) * bits;
                        let i0 = (b1 - 16) >> 5;
                        let i1 = (b1 - 1) >> 5;
                        let sf = (((i1 + 1) << 5) - b1) as u32;
                        let want = funnel(ring[i1 % w], ring[i0 % w], sf) & 0xFFFF;
                        assert_eq!(got, want, "rep {rep} lane {lane} h {h} q {q}");
                    }
                }
            }
        }
    }

    /// Host mirror of the smem layout (exl3_forward::mpk_smem_bytes) vs the kernel's pointer
    /// arithmetic: ring | A [M][K+8] | suh [K] (gate/up fold) | epi, every region 16-B aligned.
    #[test]
    fn mpk_smem_layout_matches_the_kernel() {
        for &nst in &[4usize, 8] {
            for m in 1..=16usize {
                for &(k, gu) in &[(2560usize, true), (640, false)] {
                    for &fold in &[false, true] {
                        let fold = fold && gu;
                        let sa = nst * STB;
                        let su = sa + m * (k + 8) * 2;
                        let epi = su + if fold { k * 2 } else { 0 };
                        let epi_b = if gu { (m * 256).max(4096) } else { m * 256 };
                        assert!(sa % 16 == 0 && su % 16 == 0 && epi % 16 == 0);
                        assert_eq!(epi + epi_b, crate::exl3_forward::mpk_smem_bytes(nst, m, k, gu, fold));
                    }
                }
            }
        }
    }
