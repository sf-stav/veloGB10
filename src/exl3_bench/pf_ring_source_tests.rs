//! Host-side proof tests (module `pf_ring_source_tests`), moved out of `exl3_bench.rs`; no GPU unless a test says so.
//! Part of the `exl3_bench` probe module (see `mod.rs` for the map); test bodies are unchanged.
    #[test]
    fn xq_pf_ring_sources() {
        for bits in 2i32..=7 {
            let ring = 8 * bits;
            let nsrc = if bits <= 4 { 2 } else { 3 };
            let mut used_max = 0;
            for lane in 0i32..32 {
                let (mut i0m, mut i1m) = ([0i32; 8], [0i32; 8]);
                for h in 0..2 {
                    for s in 0..4 {
                        let r = 2 * (lane & 3) + (s & 1) + 8 * (s >> 1);
                        let c = 8 * h + (lane >> 2);
                        let l = 8 * ((c >> 1) & 3) + ((r & 7) >> 1);
                        let t = 8 * l + (r & 1) + 2 * ((r >> 3) & 1) + 4 * (c >> 3) + 32 * (c & 1);
                        let b1 = (t + 257) * bits;
                        i0m[(h * 4 + s) as usize] = ((b1 - 16) >> 5) % ring;
                        i1m[(h * 4 + s) as usize] = ((b1 - 1) >> 5) % ring;
                    }
                }
                // == xq_pf_sel
                let base = i0m[0];
                let src: Vec<i32> = (0..nsrc).map(|k| (base + k) % ring).collect();
                for j in 0..8 {
                    for (idx, what) in [(i1m[j], "lo"), (i0m[j], "hi")] {
                        let off = (idx - base + ring) % ring;
                        assert!(off < nsrc, "bits {bits} lane {lane} window {j}: {what} word offset {off} >= {nsrc}");
                        used_max = used_max.max(off);
                        // == xq_pf_fetch + xq_pf_pick: the picked source is the window's own word, read from the same
                        // lane / ring load the per-window shuffles used (lane idx & 31, the second load iff idx >= 32)
                        assert_eq!(src[off as usize], idx, "bits {bits} lane {lane} window {j}: {what} picks the wrong word");
                        assert!(idx < ring && (ring > 32 || idx < 32));
                    }
                }
            }
            assert_eq!(used_max, nsrc - 1, "bits {bits}: N = {nsrc} is the tight source count");
        }
    }
