//! Host-side proof tests (module `lmh_tests`), moved out of `exl3_bench.rs`; no GPU unless a test says so.
//! Part of the `exl3_bench` probe module (see `mod.rs` for the map); test bodies are unchanged.
    fn funnel_r(lo: u32, hi: u32, s: u32) -> u32 {
        ((((hi as u64) << 32) | lo as u64) >> (s & 31)) as u32
    }
    /// exl3_gemm_body / exl3_dq_from, transcribed.
    fn old_idx(ring: &[u32], bits: usize, lane: usize, h: usize, s: usize) -> (usize, u32) {
        let r = 2 * (lane & 3) + (s & 1) + 8 * (s >> 1);
        let c = 8 * h + (lane >> 2);
        let l = 8 * ((c >> 1) & 3) + ((r & 7) >> 1);
        let t = 8 * l + (r & 1) + 2 * ((r >> 3) & 1) + 4 * (c >> 3) + 32 * (c & 1);
        assert_eq!(t, crate::exl3::tmap(r, c), "kernel tmap transcription vs crate::exl3::tmap");
        let w = 8 * bits;
        let b1 = (t + 257) * bits;
        let i0 = (b1 - 16) >> 5;
        let i1 = (b1 - 1) >> 5;
        let sf = (((i1 + 1) << 5) - b1) as u32;
        (t, funnel_r(ring[i1 % w], ring[i0 % w], sf) & 0xFFFF)
    }
    /// exl3_lmh_body, transcribed.
    fn new_idx(ring: &[u32], bits: usize, lane: usize, q: usize, s: usize) -> u32 {
        let w = 8 * bits;
        let t0 = 8 * lane + 4 * q;
        let b0 = (t0 + 257) * bits - 16;
        let b2 = (t0 + 3 + 257) * bits;
        let i0 = b0 >> 5;
        let i2 = (b2 - 1) >> 5;
        let s2 = ((i2 + 1) << 5) - b2;
        let (wa, wb) = (ring[i0 % w], ring[i2 % w]);
        let sh = s2 + (3 - s) * bits;
        assert!(sh < 48, "shift {sh} out of the single-word-select envelope");
        let d = ((sh & 31) as u32) | if sh >= 32 { 32 } else { 0 };
        let lo = if d & 32 != 0 { wa } else { wb };
        funnel_r(lo, wa, d & 31) & 0xFFFF
    }
    fn check_ring(ring: &[u32], bits: usize) {
        for lane in 0..32 {
            for h in 0..2 {
                for s in 0..4 {
                    let (t, a) = old_idx(ring, bits, lane, h, s);
                    assert_eq!(t, 8 * lane + 4 * h + s, "slot order lane {lane} h {h} s {s}");
                    let b = new_idx(ring, bits, lane, h, s);
                    assert_eq!(a, b, "bits {bits} lane {lane} h {h} s {s}: old {a:#06x} new {b:#06x}");
                }
            }
        }
    }
    #[test]
    fn lmh_quad_extract_equals_dq_from() {
        for bits in 3..=5usize {
            let w = 8 * bits;
            check_ring(&vec![0u32; w], bits);
            for p in 0..w * 32 {
                let mut ring = vec![0u32; w];
                ring[p / 32] = 1u32 << (p % 32);
                check_ring(&ring, bits);
            }
            let mut st = 0x9E37_79B9u32 ^ bits as u32;
            for _ in 0..256 {
                let ring: Vec<u32> = (0..w).map(|_| { st ^= st << 13; st ^= st >> 17; st ^= st << 5; st }).collect();
                check_ring(&ring, bits);
            }
        }
    }
