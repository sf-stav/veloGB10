//! Host-side proof tests (module `wp20_suh_tests`), moved out of `exl3_bench.rs`; no GPU unless a test says so.
//! Part of the `exl3_bench` probe module (see `mod.rs` for the map); test bodies are unchanged.
    use super::{wp20_synth, wp20_synth_edge, WP20_SUH_SEQ};
    use half::f16;

    const HNORM: f32 = 0.08838834764831845f32;

    fn f(b: u16) -> f32 { f16::from_bits(b).to_f32() }

    /// x0*s0 (+|-) x1*s1 under contraction c: 0 = both products rounded, 1 = first fused,
    /// 2 = second fused.
    fn pm(x0: f32, s0: f32, x1: f32, s1: f32, c: u32, sub: bool) -> f32 {
        match c {
            1 => { let q = x1 * s1; x0.mul_add(s0, if sub { -q } else { q }) }
            2 => (if sub { -x1 } else { x1 }).mul_add(s1, x0 * s0),
            _ => { let (p, q) = (x0 * s0, x1 * s1); if sub { p - q } else { p + q } }
        }
    }

    fn conv_of(ci: u32) -> u32 { (ci % 3) | ((ci / 3 % 3) << 2) | ((ci / 9 % 3) << 4) | ((ci / 27 % 3) << 6) }

    /// xq_had_suh_multi's source semantics (exl3_had128 as written) under contraction `conv`.
    fn src_block(x: &[u16], s: &[u16], conv: u32) -> [u16; 128] {
        let mut v = [[0f32; 4]; 32];
        for (lane, vl) in v.iter_mut().enumerate() {
            let xs = [0, 1, 2, 3].map(|j| f(x[lane * 4 + j]));
            let ss = [0, 1, 2, 3].map(|j| f(s[lane * 4 + j]));
            let c = |i: u32| (conv >> (2 * i)) & 3;
            let a = [pm(xs[0], ss[0], xs[1], ss[1], c(0), false), pm(xs[0], ss[0], xs[1], ss[1], c(1), true),
                     pm(xs[2], ss[2], xs[3], ss[3], c(2), false), pm(xs[2], ss[2], xs[3], ss[3], c(3), true)];
            *vl = [a[0] + a[2], a[1] + a[3], a[0] - a[2], a[1] - a[3]];     // stage len 2 (r, r+2)
        }
        for st in 2..7 {
            let mask = 1usize << (st - 2);
            let prev = v;
            for lane in 0..32 {
                for r in 0..4 {
                    let t = prev[lane ^ mask][r];
                    v[lane][r] = if lane & mask != 0 { t - prev[lane][r] } else { prev[lane][r] + t };
                }
            }
        }
        let mut o = [0u16; 128];
        for lane in 0..32 { for r in 0..4 { o[lane * 4 + r] = f16::from_f32(v[lane][r] * HNORM).to_bits(); } }
        o
    }

    /// The epilogue's wp20_suh_had, op for op (the served SASS form: FSEL +-v, FADD t + w).
    fn pinned_block(x: &[u16], s: &[u16]) -> [u16; 128] {
        let mut v = [[0f32; 4]; 32];
        for (lane, vl) in v.iter_mut().enumerate() {
            let xs = [0, 1, 2, 3].map(|j| f(x[lane * 4 + j]));
            let ss = [0, 1, 2, 3].map(|j| f(s[lane * 4 + j]));
            let q1 = xs[1] * ss[1];
            let q3 = xs[3] * ss[3];
            let (a0, a1) = (xs[0].mul_add(ss[0], q1), xs[0].mul_add(ss[0], -q1));
            let (a2, a3) = (xs[2].mul_add(ss[2], q3), xs[2].mul_add(ss[2], -q3));
            *vl = [a0 + a2, a1 + a3, a0 - a2, a1 - a3];
        }
        for st in 2..7 {
            let mask = 1usize << (st - 2);
            let prev = v;
            for lane in 0..32 {
                let hi = lane & mask != 0;
                for r in 0..4 {
                    let t = prev[lane ^ mask][r];
                    v[lane][r] = t + if hi { -prev[lane][r] } else { prev[lane][r] };
                }
            }
        }
        let mut o = [0u16; 128];
        for lane in 0..32 { for r in 0..4 { o[lane * 4 + r] = f16::from_f32(v[lane][r] * HNORM).to_bits(); } }
        o
    }

    fn xorshift(st: &mut u64) -> u64 { *st ^= *st << 13; *st ^= *st >> 7; *st ^= *st << 17; *st }

    /// Rows of x (row-major, width k) against one suh row: every 128-block, all 81 contractions.
    /// Returns (blocks, f16-inf outputs).
    fn check_rows(x: &[u16], s: &[u16], k: usize) -> (usize, usize) {
        let (mut blocks, mut inf) = (0, 0);
        for row in 0..x.len() / k {
            for b in 0..k / 128 {
                let (xb, sb) = (&x[row * k + b * 128..][..128], &s[b * 128..][..128]);
                let p = pinned_block(xb, sb);
                inf += p.iter().filter(|v| **v & 0x7FFF == 0x7C00).count();
                for ci in 0..81 {
                    assert_eq!(src_block(xb, sb, conv_of(ci)), p, "row {row} block {b} contraction {ci}");
                }
                blocks += 1;
            }
        }
        (blocks, inf)
    }

    #[test]
    fn wp20_f16_products_are_exact_in_f32() {
        let mut st = 0x1234_5678_9ABC_DEF1u64;
        let fin = |b: u16| b & 0x7C00 != 0x7C00;
        let mut n = 0;
        for _ in 0..2_000_000 {
            let w = xorshift(&mut st);
            let (a, b) = (w as u16, (w >> 16) as u16);
            if !fin(a) || !fin(b) { continue; }
            assert_eq!((f(a) * f(b)) as f64, f(a) as f64 * f(b) as f64, "{a:04x} * {b:04x}");
            n += 1;
        }
        for a in 0..=0xFFFFu16 {
            if !fin(a) { continue; }
            for b in [0x0001u16, 0x8001, 0x03FF, 0x0400, 0x7BFF, 0xFBFF, 0x3C01, 0x3555] {
                assert_eq!((f(a) * f(b)) as f64, f(a) as f64 * f(b) as f64, "{a:04x} * {b:04x}");
            }
        }
        assert!(n > 1_500_000);
    }

    #[test]
    fn wp20_every_contraction_equals_the_pinned_sequence() {
        let k = 640;
        // v1's calibration rows, then the v1c verification's edge expert
        check_rows(&wp20_synth(16 * k, 0x20C0, 1.0), &wp20_synth(k, 0x5CA1, 0.5), k);
        let (blocks, inf) = check_rows(&wp20_synth_edge(16 * k, 0x20C1_5EED_0001), &wp20_synth_edge(k, 0x5CA1_5EED_0002), k);
        eprintln!("edge expert: {inf} f16-inf outputs of {}", blocks * 128);
        assert!(inf > 0 && inf < blocks * 128 / 2, "edge expert coverage: {inf} inf of {}", blocks * 128);
        // random full-domain rows: x regimes 0..3 against suh regimes 1,2,3,0 (fresh per row)
        let mut st = 0xC0FF_EE00_D15E_A5E5u64;
        let (mut tb, mut ti) = (0, 0);
        for row in 0..500u64 {
            let x = wp20_synth_edge(512, xorshift(&mut st) ^ row);
            let s = wp20_synth_edge(640, xorshift(&mut st) ^ (row << 20));
            let (b, i) = check_rows(&x, &s[128..], 512);
            tb += b; ti += i;
        }
        eprintln!("full-domain rows: {ti} f16-inf outputs of {}", tb * 128);
        assert_eq!(WP20_SUH_SEQ, conv_of(1 + 3 + 9 + 27), "the pinned id is 'first fused' on all four ops");
    }

    #[test]
    fn wp20_v1_calibration_could_never_find_a_unique_match() {
        let (r, k) = (16usize, 640usize);
        let (x, s) = (wp20_synth(r * k, 0x20C0, 1.0), wp20_synth(k, 0x5CA1, 0.5));
        let run = |conv: u32| -> Vec<u16> {
            let mut out = Vec::with_capacity(r * k);
            for row in 0..r {
                for b in 0..k / 128 {
                    out.extend(src_block(&x[row * k + b * 128..][..128], &s[b * 128..][..128], conv));
                }
            }
            out
        };
        let served = run(WP20_SUH_SEQ);
        let hits = (0..81u32).filter(|&ci| run(conv_of(ci)) == served).count();
        assert_eq!(hits, 81, "all 81 conventions are bitwise identical on f16 inputs");
    }
