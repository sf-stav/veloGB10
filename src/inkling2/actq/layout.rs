//! Native-MMA weight layouts of the expert tensors (kernels/gpu_inkling2/c0 header): pure byte permutations of the
//! checkpoint, made once at load. The w13 gate|up tile interleave (`gpu::layout::w13_tile_rows`) is applied first, as
//! for K2's layouts, so a warp's 16-row tile holds gate (rows 0-7) and up (rows 8-15) of the same 8 channels.

/// NVFP4 "NA": codes `[n][k/2]` + e4m3 scales `[n][k/16]` -> per 16-row tile T and 64-k step s:
/// codes `[T][s][lane][16]` (A fragment a0..a3 of lane (g, t)), scales `[T][s][16][4]`.
pub fn nv4_native_into(codes: &[u8], scales: &[u8], n: usize, k: usize, w: &mut [u8], sc: &mut [u8]) {
    assert!(n % 16 == 0 && k % 64 == 0 && codes.len() == n * k / 2 && scales.len() == n * k / 16
            && w.len() == codes.len() && sc.len() == scales.len(), "nv4_native: [{n}][{k}]");
    let (ns, k2, k16) = (k / 64, k / 2, k / 16);
    for t16 in 0..n / 16 {
        for s in 0..ns {
            let base = (t16 * ns + s) * 512;
            for lane in 0..32 {
                let (g, t) = (lane / 4, lane % 4);
                for r in 0..4 {
                    let row = 16 * t16 + g + 8 * (r & 1);
                    let byte = 32 * s + 16 * (r >> 1) + 4 * t;
                    let o = base + lane * 16 + r * 4;
                    w[o..o + 4].copy_from_slice(&codes[row * k2 + byte..row * k2 + byte + 4]);
                }
            }
            for rr in 0..16 {
                let row = 16 * t16 + rr;
                let o = ((t16 * ns + s) * 16 + rr) * 4;
                sc[o..o + 4].copy_from_slice(&scales[row * k16 + 4 * s..row * k16 + 4 * s + 4]);
            }
        }
    }
}

/// MXFP8 "MA": e4m3 codes `[n][k]` + ue8m0 exponents `[n][k/32]` -> per tile T and 32-k step s: codes
/// `[T][s][lane][16]`; exponents per 4-step quad q: `[T][q][16][4]` (byte v = step 4q + v; last quad zero-padded
/// when `k % 128 == 64`). `k % 64 == 0`.
pub fn mx8_native_into(codes: &[u8], exps: &[u8], n: usize, k: usize, w: &mut [u8], sc: &mut [u8]) {
    let (ns, nq, k32) = (k / 32, k.div_ceil(128), k / 32);
    assert!(n % 16 == 0 && k % 64 == 0 && codes.len() == n * k && exps.len() == n * k32
            && w.len() == codes.len() && sc.len() == (n / 16) * nq * 64, "mx8_native: [{n}][{k}]");
    for t16 in 0..n / 16 {
        for s in 0..ns {
            let base = (t16 * ns + s) * 512;
            for lane in 0..32 {
                let (g, t) = (lane / 4, lane % 4);
                for r in 0..4 {
                    let row = 16 * t16 + g + 8 * (r & 1);
                    let byte = 32 * s + 16 * (r >> 1) + 4 * t;
                    let o = base + lane * 16 + r * 4;
                    w[o..o + 4].copy_from_slice(&codes[row * k + byte..row * k + byte + 4]);
                }
            }
        }
        for q in 0..nq {
            let take = (k32 - 4 * q).min(4);
            for rr in 0..16 {
                let row = 16 * t16 + rr;
                let o = ((t16 * nq + q) * 16 + rr) * 4;
                sc[o..o + take].copy_from_slice(&exps[row * k32 + 4 * q..row * k32 + 4 * q + take]);
                for z in take..4 { sc[o + z] = 0; }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_layouts_are_permutations() {
        let (n, k) = (32usize, 256usize);
        let codes: Vec<u8> = (0..n * k / 2).map(|i| (i * 7 + i / 300) as u8).collect();
        let scales: Vec<u8> = (0..n * k / 16).map(|i| (i * 13) as u8).collect();
        let (mut w, mut s) = (vec![0u8; codes.len()], vec![0u8; scales.len()]);
        nv4_native_into(&codes, &scales, n, k, &mut w, &mut s);
        let (mut a, mut b) = (codes.clone(), w.clone());
        a.sort_unstable(); b.sort_unstable();
        assert_eq!(a, b);
        // lane (g=1, t=2), tile 1, step 1, a3 = row 16 + 9, bytes 32 + 16 + 8
        let o = ((1 * (k / 64) + 1) * 512) + (1 * 4 + 2) * 16 + 12;
        assert_eq!(&w[o..o + 4], &codes[25 * k / 2 + 56..25 * k / 2 + 60]);
        let mc: Vec<u8> = (0..n * k).map(|i| (i * 31 + i / 77) as u8).collect();
        let me: Vec<u8> = (0..n * k / 32).map(|i| (i * 3) as u8).collect();
        let (mut w, mut s) = (vec![0u8; mc.len()], vec![0u8; me.len()]);
        mx8_native_into(&mc, &me, n, k, &mut w, &mut s);
        let (mut a, mut b) = (mc.clone(), w.clone());
        a.sort_unstable(); b.sort_unstable();
        assert_eq!(a, b);
        assert_eq!(s[((1 * (k / 128) + 1) * 16 + 3) * 4 + 2], me[19 * (k / 32) + 6]);
        // Drop 3 (#3b): 64-channel odd widths (192, 320, 576) preserve codes and pack trailing 2 scales + 2 zeros.
        for &kk in &[192usize, 320, 576] {
            let mc: Vec<u8> = (0..n * kk).map(|i| (i * 17 + 3) as u8).collect();
            let me: Vec<u8> = (0..n * kk / 32).map(|i| ((i * 5 + 1) & 0x7F) as u8).collect();
            let nq = kk.div_ceil(128);
            let (mut w, mut s) = (vec![0u8; mc.len()], vec![0xAAu8; (n / 16) * nq * 64]);
            mx8_native_into(&mc, &me, n, kk, &mut w, &mut s);
            let (mut a, mut b) = (mc.clone(), w.clone());
            a.sort_unstable(); b.sort_unstable();
            assert_eq!(a, b, "mx8 codes k={kk}");
            let last_q = nq - 1;
            let o = ((1 * nq + last_q) * 16 + 3) * 4;
            assert_eq!(s[o], me[19 * (kk / 32) + 4 * last_q]);
            assert_eq!(s[o + 1], me[19 * (kk / 32) + 4 * last_q + 1]);
            assert_eq!((s[o + 2], s[o + 3]), (0, 0));
        }
    }
}
