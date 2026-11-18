//! Unit tests (module `ple_k_tests`) moved out of the old inline block in `exl3_forward.rs`; text unchanged.
//! Declared from `mod.rs`; the body keeps its original indentation and `use super::*` imports.

    use super::*;

    /// CF-P1g: the K-generic decoders. K = 5 must reproduce the existing 102-byte reference/fast decoders exactly, and
    /// the K = 6 group decode must equal the K = 6 per-bit reference on every non-NaN output.
    #[test]
    fn ple_decode_k_generic_matches() {
        let cb = PleDev::mul1_codebook();
        let mut s = 0xD1B5_4A32_D192_ED03u64;
        let mut next = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        let isnan = |b: u16| (b & 0x7c00) == 0x7c00 && (b & 0x03ff) != 0;
        let (mut bb, mut bf) = ([0u16; 160], [0f32; 160]);
        for k in 1..=8usize {
            let rb = ple_row_bytes(k);
            assert!(rb <= PLE_ROW_MAX, "K={k}: row {rb} B exceeds PLE_ROW_MAX");
            let mut raw = [0u8; PLE_ROW_MAX];
            let (mut o1, mut o2, mut o3) = ([0u16; 160], [0u16; 160], [0u16; 160]);
            let mut bad = 0usize;
            for r in 0..6_000usize {
                for b in raw[..rb].iter_mut() { *b = next() as u8; }
                let sc: u16 = match r % 8 { 0 => next() as u16, 1 => next() as u16 & 0x03ff, _ => 0x2000 | (next() as u16 & 0x0fff) };
                raw[0] = sc as u8; raw[1] = (sc >> 8) as u8;
                for i in 0..160 {
                    bb[i] = if r % 5 == 0 { next() as u16 } else { 0x1000 | (next() as u16 & 0x8fff) };
                    bf[i] = half::f16::from_bits(bb[i]).to_f32();
                }
                ple_decode_row_ref_k(&raw, k, &bb, &cb, &mut o1);
                ple_decode_row_fast_k(&raw, k, &bf, &cb, &mut o2);
                if k == 5 {
                    ple_decode_row_ref(<&[u8; 102]>::try_from(&raw[..102]).unwrap(), &bb, &cb, &mut o3);
                    for i in 0..160 { if o1[i] != o3[i] && !(isnan(o1[i]) && isnan(o3[i])) { bad += 1; } }
                }
                for i in 0..160 { if o1[i] != o2[i] && !(isnan(o1[i]) && isnan(o2[i])) { bad += 1; } }
            }
            assert_eq!(bad, 0, "K={k}: generic ple decoders disagree on {bad} non-NaN outputs");
        }
    }
