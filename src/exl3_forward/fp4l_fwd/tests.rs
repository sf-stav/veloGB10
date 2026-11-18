//! Unit tests of the FP4L fragment layouts, kernel entry lists and host emulations (the former inline `mod tests`; same module name).

    use super::*;

    /// Every entry this file launches is in XF_FNS (the module load list): a launched name missing there fails at
    /// the first dispatch on the GPU (`kernel ... missing from gpu_fp4l`), which no host test would otherwise see.
    #[test]
    fn every_launched_entry_is_loaded() {
        let src = src_text!("exl3_forward/fp4l_fwd");
        let mut missing = Vec::new();
        let mut n = 0;
        for (i, pat) in src.match_indices("xflaunch!(") {
            // the second macro argument, when it is a string literal (a name expression such as nv4_tiles_fn's
            // result is a row of XF_FNS by construction)
            let rest = &src[i + pat.len()..];
            let Some(c) = rest.find(',') else { continue };
            let arg = rest[c + 1..].trim_start();
            let Some(lit) = arg.strip_prefix('"') else { continue };
            let Some(e) = lit.find('"') else { continue };
            let name = &lit[..e];
            n += 1;
            if !XF_FNS.contains(&name) && !missing.contains(&name) { missing.push(name); }
        }
        assert!(n > 20, "found only {n} literal launches: the scan is broken");
        assert!(missing.is_empty(), "launched but not in XF_FNS: {missing:?}");
    }

    /// xf_mx8_gemm_x2 takes `struct XfPair` by value: five 2-pointer arrays, 80 bytes, member z at index z.
    #[test]
    fn xf_pair_matches_the_kernel_struct() {
        assert_eq!(std::mem::size_of::<XfPair>(), 80);
        assert_eq!(std::mem::align_of::<XfPair>(), 8);
        let p = XfPair { w: [1, 2], s: [3, 4], y: [5, 6], ws: [7, 8], cnt: [9, 10] };
        let words: [u64; 10] = unsafe { std::mem::transmute(p) };
        assert_eq!(words, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "field order = the kernel's w, s, y, ws, cnt");
    }

    /// Host emulation of the kernel's integer E2M1 path (PRMT LUT + sign bits) against the table.
    fn prmt(a: u32, b: u32, sel: u32) -> u32 {
        let bytes = [(a & 0xFF) as u8, (a >> 8) as u8, (a >> 16) as u8, (a >> 24) as u8,
                     (b & 0xFF) as u8, (b >> 8) as u8, (b >> 16) as u8, (b >> 24) as u8];
        let mut r = 0u32;
        for i in 0..4 {
            let s = (sel >> (4 * i)) & 0xF;
            let mut v = bytes[(s & 7) as usize];
            if s & 8 != 0 { v = if v & 0x80 != 0 { 0xFF } else { 0 }; }
            r |= (v as u32) << (8 * i);
        }
        r
    }
    fn e2m1x8_int(w: u32) -> [u32; 4] {
        let (lut_lo, lut_hi) = (0x3E3C3800u32, 0x46444240u32);
        let mlo = prmt(lut_lo, lut_hi, w & 0x7777);
        let mhi = prmt(lut_lo, lut_hi, (w >> 16) & 0x7777);
        let s = w & 0x88888888;
        let (r0, r1, r2, r3) = (prmt(mlo, 0, 0x1404), prmt(mlo, 0, 0x3424), prmt(mhi, 0, 0x1404), prmt(mhi, 0, 0x3424));
        [r0 | ((s << 12) & 0x8000) | ((s << 24) & 0x8000_0000),
         r1 | ((s << 4) & 0x8000) | ((s << 16) & 0x8000_0000),
         r2 | ((s >> 4) & 0x8000) | ((s << 8) & 0x8000_0000),
         r3 | ((s >> 12) & 0x8000) | (s & 0x8000_0000)]
    }

    #[test]
    fn e2m1_int_path_matches_the_table_for_every_nibble_pair() {
        use crate::fp4l::pack::E2M1;
        for lo in 0..16u32 {
            for hi in 0..16u32 {
                for pos in 0..4 {
                    let w = (lo | (hi << 4)) << (8 * pos);
                    let o = e2m1x8_int(w);
                    let h = o[pos];
                    let (a, b) = (half::f16::from_bits((h & 0xFFFF) as u16).to_f32(), half::f16::from_bits((h >> 16) as u16).to_f32());
                    assert_eq!(a.to_bits(), E2M1[lo as usize].to_bits(), "lo {lo} pos {pos}");
                    assert_eq!(b.to_bits(), E2M1[hi as usize].to_bits(), "hi {hi} pos {pos}");
                }
            }
        }
    }

    #[test]
    fn ks_is_a_function_of_shape_and_divides_k() {
        for &(k, n) in &[(2560usize, 10240usize), (2560, 6144), (6144, 2560), (2560, 12288), (2560, 512), (6144, 2560),
                         (2560, 640), (640, 2560), (1536, 2560), (2560, 3072), (2560, 248320), (2560, 65536), (384, 2560), (128, 2560)] {
            let ks = fx_ks(k, n) as usize;
            assert!(ks >= 1 && (k / 128) % ks == 0, "k {k} n {n} ks {ks}");
            assert_eq!(fx_ks(k, n), fx_ks(k, n));
        }
        assert_eq!(fx_ks(2560, 248320), 1);
        assert_eq!(fx_ks(2560, 10240), 4);
    }

    /// The trunk combine's gate tree, on the host: per-lane partials over c = t, t + bs, ... (`part += sg * x`, an
    /// fma), then the halving tree of `bs` leaves (xq_moe_combine / _rows / _ep / _rows_ep: bs = 1024).
    fn tree_halving(sgc: &[u16], xg: &[u16], bs: usize) -> f32 {
        let h = sgc.len();
        let f = |b: u16| half::f16::from_bits(b).to_f32();
        let mut sm = vec![0f32; bs];
        for (t, v) in sm.iter_mut().enumerate() {
            let mut part = 0f32;
            let mut c = t;
            while c < h { part = f(sgc[c]).mul_add(f(xg[c]), part); c += bs; }
            *v = part;
        }
        let mut s2 = bs / 2;
        while s2 > 0 { for t in 0..s2 { sm[t] += sm[t + s2]; } s2 /= 2; }
        sm[0]
    }
    /// xq_moe_combine_rows_ep2's tree: 128 lanes x 8 columns (v = 8 tid + u over c0 = 8 tid + 1024 j), leaves
    /// re-paired in one warp (levels 512..32 in registers, 16..1 by shuffle) — the same pairs as the halving tree.
    fn tree_ep2(sgc: &[u16], xg: &[u16]) -> f32 {
        let h = sgc.len();
        let f = |b: u16| half::f16::from_bits(b).to_f32();
        let mut sm = vec![0f32; 1024];
        for tid in 0..128 {
            let mut part = [0f32; 8];
            let mut c0 = tid * 8;
            while c0 < h { for u in 0..8 { part[u] = f(sgc[c0 + u]).mul_add(f(xg[c0 + u]), part[u]); } c0 += 1024; }
            for u in 0..8 { sm[tid * 8 + u] = part[u]; }
        }
        let mut lanes = vec![0f32; 32];
        for (l, lv) in lanes.iter_mut().enumerate() {
            let mut a: Vec<f32> = (0..32).map(|j| sm[l + 32 * j]).collect();
            for (n, _) in [(16, 512), (8, 256), (4, 128), (2, 64)] { for j in 0..n { a[j] = a[j] + a[j + n]; } }
            *lv = a[0] + a[1];
        }
        let mut s2 = 16;
        while s2 > 0 { for t in 0..s2 { lanes[t] += lanes[t + s2]; } s2 /= 2; }
        lanes[0]
    }

    #[test]
    fn shared_gate_channels_rebuild_the_logit_in_every_combine_tree() {
        let h = 2560;
        let sgc = sg_channel_constants(h);
        // the constants are what the kernel's up-scales assume
        let want = [1.0f32, 2f32.powi(-11), 2f32.powi(-22), 2f32.powi(-24), 2f32.powi(-24), 2f32.powi(-24)];
        for i in 0..6 { assert_eq!(half::f16::from_bits(sgc[i]).to_f32(), want[i], "sgc[{i}]"); }
        assert!(sgc[6..].iter().all(|&b| b == 0));
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed };
        let mut checked = 0;
        for i in 0..200_000 {
            // log-uniform magnitudes 2^-25 .. 2^16, random sign and mantissa bits, plus edge cases
            let r = next();
            let mut l = f32::from_bits(((((r >> 32) as u32) % 41 + 102) << 23) | ((r as u32) & 0x7F_FFFF));
            if r >> 63 == 1 { l = -l; }
            if i < 8 { l = [0.0f32, -0.0, 65504.0, -65504.0, 2f32.powi(-25), 1.0, -0.5, 65503.0][i]; }
            if l.abs() > 65504.0 { continue; }
            let ch = sg_encode(l);
            let mut xg = vec![0u16; h];
            xg[..6].copy_from_slice(&ch);
            for c in &ch { assert!(half::f16::from_bits(*c).to_f32().is_finite()); }
            // bit for bit, except that -0 rebuilds as +0 (sigmoid(-0) == sigmoid(+0) == 0.5)
            let want = if l == 0.0 { 0u32 } else { l.to_bits() };
            assert_eq!(tree_halving(&sgc, &xg, 1024).to_bits(), want, "halving tree, L = {l:e}");
            assert_eq!(tree_ep2(&sgc, &xg).to_bits(), want, "ep2 tree, L = {l:e}");
            checked += 1;
        }
        assert!(checked > 190_000);
        // |L| > 65504 clamps to +-65504 (sigmoid is exactly 0 or 1 there either way)
        let ch = sg_encode(65519.0);
        let mut xg = vec![0u16; h];
        xg[..6].copy_from_slice(&ch);
        assert_eq!(tree_halving(&sgc, &xg, 1024), 65504.0);
        let ch = sg_encode(1e9);
        let mut xg = vec![0u16; h];
        xg[..6].copy_from_slice(&ch);
        assert_eq!(tree_halving(&sgc, &xg, 1024), 65504.0);
        // below 2^-25 only bits under 2^-48 are dropped: the rebuilt value is within 2^-48
        let tiny = 1.2345e-9f32;
        let ch = sg_encode(tiny);
        let mut xg = vec![0u16; h];
        xg[..6].copy_from_slice(&ch);
        assert!((tree_halving(&sgc, &xg, 1024) - tiny).abs() <= 2f32.powi(-48));
    }

    #[test]
    fn bf_ks_is_a_shape_function_that_divides_k_into_32_groups() {
        for &(k, n) in &[(2560usize, 512usize), (2560, 10240), (2560, 2560), (10240, 320), (320, 10240), (2560, 248320),
                         (6144, 2560), (640, 2560), (2560, 640), (2560, 12288), (32, 64), (64, 2)] {
            let ks = bf_ks(k, n) as usize;
            let g = k / 32;
            assert!(ks >= 1 && g % ks == 0, "k {k} n {n} ks {ks}");
            assert!(ks == 1 || g / ks >= 2, "k {k} n {n}: a slab holds >= 2 groups");
            assert_eq!(bf_ks(k, n), bf_ks(k, n));
        }
        assert_eq!(bf_ks(2560, 512), 40);     // the router: 160 CTAs
        assert_eq!(bf_ks(2560, 10240), 4);    // the n-gram key projection: 320 CTAs
        assert_eq!(bf_ks(320, 10240), 5);     // the hc up projection (K = 320 is not a multiple of 128)
        assert_eq!(bf_ks(2560, 248320), 1);   // lm_head class
    }

    #[test]
    fn fx_scratch_sizes_cover_every_decode_stored_bf16_gemm() {
        // the Flash-Next geometry at the widest Scratch (16 rows): ws holds ks * 16 * n and the counters every N tile
        let (h, hcn, lr, ne, w) = (2560usize, 4usize, 320usize, 512usize, 16usize);
        let ws = |k: usize, n: usize| bf_ks(k, n) as usize * w * n;
        let rw = h * hcn;
        let need_r = ws(h, ne);
        let need_p = ws(2560, 10240).max(ws(2560, 2560));
        let need_h = ws(rw, lr).max(ws(lr, rw));
        // FxScratch::with_dims's own arithmetic (kept in step with it by this test)
        assert_eq!(need_r, 40 * 16 * 512);
        assert_eq!(need_p, 4 * 16 * 10240);
        assert!(need_h >= 5 * 16 * 10240 && need_h >= bf_ks(rw, lr) as usize * 16 * lr);
        assert!(10240usize.div_ceil(128) <= rw.max(lr).div_ceil(128));
    }

    #[test]
    fn e4m3_rne_round_trips_every_code_and_rounds_midpoints_to_even() {
        for c in 0u8..=0x7E {
            let v = crate::fp4l::pack::e4m3(c);
            assert_eq!(f32_to_e4m3_rne(v), c);
            assert_eq!(f32_to_e4m3_rne(-v) & 0x7F, c);
        }
        assert_eq!(f32_to_e4m3_rne(1000.0), 0x7E);
        // midpoint between 1.0 (0x38) and 1.125 (0x39) -> even 0x38
        assert_eq!(f32_to_e4m3_rne(1.0625), 0x38);
    }

    #[test]
    fn mx8_frag_is_the_kernels_index_map_and_a_permutation() {
        // n = 37 (two full tiles + a padded one), k = 192 (three 64-k blocks): every element and scale lands at the
        // offset the kernels' xf_mx8_f64_at computes, and nothing else is written
        let (n, k) = (37usize, 192usize);
        let codes: Vec<u8> = (0..n * k).map(|i| (i.wrapping_mul(2654435761) >> 7) as u8 | 1).collect();
        let scales: Vec<u8> = (0..n * k / 32).map(|i| (i % 250) as u8).collect();
        let (w, sc) = mx8_frag(&codes, &scales, n, k);
        assert_eq!((w.len(), sc.len()), (48 * k, 48 * k / 32));
        let kb = k / 64;
        let mut hit = vec![false; w.len()];
        for row in 0..n {
            for kk in 0..k {
                let (t, rr) = (row / 16, row % 16);
                let (h, gid) = (rr >> 3, rr & 7);
                let (b, gs, tig, j) = (kk >> 6, (kk >> 5) & 1, (kk >> 3) & 3, kk & 7);
                let o = ((t * kb + b) * 2 + h) * 512 + (gid * 4 + tig) * 16 + gs * 8 + j;
                assert_eq!(w[o], codes[row * k + kk], "code ({row}, {kk})");
                hit[o] = true;
                let so = (t * kb + b) * 32 + (gid >> 1) * 8 + h * 4 + (gid & 1) * 2 + gs;
                assert_eq!(sc[so], scales[row * (k / 32) + kk / 32], "scale ({row}, {kk})");
            }
        }
        assert!(w.iter().zip(&hit).all(|(&v, &h)| h || v == 0), "padding rows are zero codes");
    }

    #[test]
    fn nv4_frag_is_the_kernels_index_map_and_a_permutation() {
        // n = 32 (two tiles), k = 384 (three 128-k blocks): a permutation of distinct-position markers lands every
        // nibble byte and scale where the xf_nv4_* kernels read it
        let (n, k) = (32usize, 384usize);
        let codes: Vec<u8> = (0..n * k / 2).map(|i| (i.wrapping_mul(2654435761) >> 9) as u8).collect();
        let scales: Vec<u8> = (0..n * k / 16).map(|i| (i * 7 % 251) as u8).collect();
        let (mut w, mut sc) = (vec![0u8; codes.len()], vec![0u8; scales.len()]);
        nv4_frag_into(&codes, &scales, n, k, &mut w, &mut sc);
        let mut ws = w.clone(); ws.sort_unstable();
        let mut cs = codes.clone(); cs.sort_unstable();
        assert_eq!(ws, cs, "codes are a permutation");
        let kb = k / 128;
        for row in 0..n {
            let (t, h, gid) = (row / 16, (row % 16) / 8, row % 8);
            for kk in (0..k).step_by(2) {
                // kernel view: block b, step s2 (64 k), lane tig owns k 16tig..16tig+15 of the step
                let (b, s2, tig, jb) = (kk / 128, (kk / 64) % 2, (kk % 64) / 16, (kk % 16) / 2);
                let o = ((t * kb + b) * 2 + h) * 512 + (gid * 4 + tig) * 16 + s2 * 8 + jb;
                assert_eq!(w[o], codes[row * k / 2 + kk / 2], "nibble byte ({row}, {kk})");
                let so = (t * kb + b) * 128 + (gid * 4 + tig) * 4 + 2 * h + s2;
                assert_eq!(sc[so], scales[row * k / 16 + kk / 16], "scale ({row}, {kk})");
            }
        }
    }

    #[test]
    fn bf16_frag_is_the_kernels_index_map_and_unfrag_inverts_it() {
        // n = 37 (padded to 48), k = 96 (three 32-k groups): every value lands where xf_bf16w_gemm / xf_b32_to_rm read it
        let (n, k) = (37usize, 96usize);
        let bits: Vec<u16> = (0..n * k).map(|i| (i as u32).wrapping_mul(2654435761).rotate_right(7) as u16 | 1).collect();
        let w = bf16_frag(&bits, n, k);
        assert_eq!(w.len(), 48 * k);
        let g = k / 32;
        for row in 0..n {
            for kk in 0..k {
                let (t, h, gid) = (row / 16, (row % 16) / 8, row % 8);
                let (gg, tig, j) = (kk / 32, (kk % 32) / 8, kk % 8);
                assert_eq!(w[((t * g + gg) * 2 + h) * 256 + (gid * 4 + tig) * 8 + j], bits[row * k + kk], "({row}, {kk})");
            }
        }
        assert_eq!(w.iter().filter(|&&v| v == 0).count(), 11 * k, "only the 11 padding rows are zero");
        assert_eq!(bf16_unfrag(&w, n, k), bits);
    }

    #[test]
    fn mx8_group_quant_is_within_half_an_ulp() {
        let v: Vec<f32> = (0..32).map(|i| ((i as f32) - 15.5) * 0.013).collect();
        let mut c = [0u8; 32];
        let s = mx8_quant_group(&v, &mut c);
        let scale = crate::fp4l::pack::ue8m0(s);
        for (i, &x) in v.iter().enumerate() {
            let d = crate::fp4l::pack::e4m3(c[i]) * scale;
            assert!((d - x).abs() <= x.abs() * 0.0625 + scale * 2f32.powi(-9), "{x} -> {d}");
        }
    }

    #[test]
    fn ple_row_decode_matches_the_pack_reference() {
        let mut raw = [0u8; 90];
        for (i, b) in raw.iter_mut().enumerate() { *b = if i < 80 { (i * 37 % 256) as u8 } else { 0x30 + (i as u8 % 8) }; }
        let gs = 3.3242e-5f32;
        let mut out = [0u16; 160];
        ple_fx_decode(&raw, gs, &mut out);
        let h = crate::fp4l::pack::Nv4Host { packed: raw[..80].to_vec(), scales: raw[80..].to_vec(), gs, n: 1, k: 160 };
        let mut r = vec![0f32; 160];
        h.row_f32(0, &mut r);
        for i in 0..160 { assert_eq!(out[i], half::f16::from_f32(r[i]).to_bits(), "dim {i}"); }
    }
