//! Unit tests (module `wp05_tests`) moved out of the old inline block in `exl3_forward.rs`; text unchanged.
//! Declared from `mod.rs`; the body keeps its original indentation and `use super::*` imports.

    use super::*;

    /// WP05: the group-decode rewrite equals the per-bit reference on every non-NaN output
    /// (random rows incl. subnormal/inf/huge scales and biases; the real mul1 codebook).
    #[test]
    fn ple_decode_fast_matches_ref() {
        let cb = PleDev::mul1_codebook();
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        let isnan = |b: u16| (b & 0x7c00) == 0x7c00 && (b & 0x03ff) != 0;
        let (mut raw, mut bb, mut bf) = ([0u8; 102], [0u16; 160], [0f32; 160]);
        let (mut o1, mut o2) = ([0u16; 160], [0u16; 160]);
        let mut bad = 0usize;
        for r in 0..20_000usize {
            for b in raw.iter_mut() { *b = next() as u8; }
            let sc: u16 = match r % 8 { 0 => next() as u16, 1 => next() as u16 & 0x03ff, _ => 0x2000 | (next() as u16 & 0x0fff) };
            raw[0] = sc as u8;
            raw[1] = (sc >> 8) as u8;
            for i in 0..160 {
                bb[i] = if r % 5 == 0 { next() as u16 } else { 0x1000 | (next() as u16 & 0x8fff) };
                bf[i] = half::f16::from_bits(bb[i]).to_f32();
            }
            ple_decode_row_ref(&raw, &bb, &cb, &mut o1);
            ple_decode_row_fast(&raw, &bf, &cb, &mut o2);
            for i in 0..160 {
                if o1[i] != o2[i] && !(isnan(o1[i]) && isnan(o2[i])) { bad += 1; }
            }
        }
        assert_eq!(bad, 0, "ple_decode_row_fast differs from the reference on {bad} non-NaN outputs");
    }
