//! Host-side weight layouts for the Inkling-2 MoE / LM-head kernels (kernels/gpu_inkling2/60-65): pure byte
//! permutations made at load, no value changes. The three fragment-native layouts are COPIES of the fp4l loaders
//! (`exl3_forward::fp4l_fwd::weights::{nv4_frag_into, mx8_frag, bf16_frag}`, never edited there) so a change to the
//! other family cannot move this family's bytes; the gate|up tile interleave is Inkling-2's own.

/// The checkpoint's interleaved w13 rows (row `2i` = gate `i`, row `2i+1` = up `i`, `n = 2 * inter`) -> the
/// kernels' tile-interleaved order: in 16-row tile `t`, rows 0..7 = gate `8t..8t+7`, rows 8..15 = up `8t..8t+7`.
/// `row_bytes` = bytes per row of the array being permuted (codes or scales). `inter % 8 == 0`.
pub fn w13_tile_rows(src: &[u8], n: usize, row_bytes: usize) -> Vec<u8> {
    assert!(n % 16 == 0 && src.len() == n * row_bytes, "w13_tile_rows: [{n}] x {row_bytes} B");
    let mut o = vec![0u8; src.len()];
    for p in 0..n {
        let (t, rr) = (p / 16, p % 16);
        let (h, c) = (rr / 8, rr % 8);
        let i = 8 * t + c;
        let s = 2 * i + h;
        o[p * row_bytes..(p + 1) * row_bytes].copy_from_slice(&src[s * row_bytes..(s + 1) * row_bytes]);
    }
    o
}

/// The inverse map: kernel row `p` of a tile-interleaved w13 -> (is_up, intermediate channel i).
pub fn w13_row_of(p: usize) -> (bool, usize) {
    let (t, rr) = (p / 16, p % 16);
    (rr >= 8, 8 * t + rr % 8)
}

/// NVFP4 "F128" + optional trailing 64-k half-block (Drop 3 #3b): codes `[n][k/2]` + E4M3 scales `[n][k/16]` ->
/// per 16-row tile `t`, `kb = k / 128` full F128 blocks (`W'[b][h][lane][16]`, `S'[b][lane][4]`) followed when
/// `k % 128 == 64` by one 64-k half-block (`W'[h][lane][8]` = 512 B, `S'[lane][2]` = 64 B).
/// `n % 16 == 0`, `k % 64 == 0`. Byte-identical to pure F128 whenever `k % 128 == 0`.
pub fn nv4_frag_into(codes: &[u8], scales: &[u8], n: usize, k: usize, w: &mut [u8], sc: &mut [u8]) {
    assert!(n % 16 == 0 && k % 64 == 0 && codes.len() == n * k / 2 && scales.len() == n * k / 16
            && w.len() == codes.len() && sc.len() == scales.len(), "nv4_frag: [{n}][{k}] shape");
    let (kb, has_half, k2, k16) = (k / 128, (k % 128 == 64) as usize, k / 2, k / 16);
    let w_stride = kb * 1024 + has_half * 512;
    let s_stride = kb * 128 + has_half * 64;
    for t in 0..n / 16 {
        let tw = t * w_stride;
        let ts = t * s_stride;
        for h in 0..2 {
            for gid in 0..8 {
                let row = 16 * t + 8 * h + gid;
                for b in 0..kb {
                    for tig in 0..4 {
                        let lane = gid * 4 + tig;
                        for s2 in 0..2 {
                            let o = tw + b * 1024 + h * 512 + lane * 16 + s2 * 8;
                            let i = row * k2 + 64 * b + 32 * s2 + 8 * tig;
                            w[o..o + 8].copy_from_slice(&codes[i..i + 8]);
                            sc[ts + b * 128 + lane * 4 + 2 * h + s2] = scales[row * k16 + 8 * b + 4 * s2 + tig];
                        }
                    }
                }
                if has_half != 0 {
                    for tig in 0..4 {
                        let lane = gid * 4 + tig;
                        let o = tw + kb * 1024 + h * 256 + lane * 8;
                        let i = row * k2 + 64 * kb + 8 * tig;
                        w[o..o + 8].copy_from_slice(&codes[i..i + 8]);
                        sc[ts + kb * 128 + lane * 2 + h] = scales[row * k16 + 8 * kb + tig];
                    }
                }
            }
        }
    }
}

/// MXFP8 "F64" (copy of fp4l `mx8_frag`, serial form): E4M3 codes `[n][k]` + UE8M0 scales `[n][k/32]` ->
/// W'[t][b][h][lane][16], S'[t][b][tig][8]. `n % 16 == 0`, `k % 64 == 0`.
pub fn mx8_frag_into(codes: &[u8], scales: &[u8], n: usize, k: usize, w: &mut [u8], sc: &mut [u8]) {
    assert!(n % 16 == 0 && k % 64 == 0 && codes.len() == n * k && scales.len() == n * k / 32
            && w.len() == codes.len() && sc.len() == scales.len(), "mx8_frag: [{n}][{k}] shape");
    let kb = k / 64;
    for t in 0..n / 16 {
        for rr in 0..16 {
            let row = 16 * t + rr;
            let (h, gid) = (rr >> 3, rr & 7);
            let src = &codes[row * k..(row + 1) * k];
            let ssrc = &scales[row * (k / 32)..(row + 1) * (k / 32)];
            for b in 0..kb {
                for gs in 0..2 {
                    for tig in 0..4 {
                        let o = ((t * kb + b) * 2 + h) * 512 + (gid * 4 + tig) * 16 + gs * 8;
                        let i = 64 * b + 32 * gs + 8 * tig;
                        w[o..o + 8].copy_from_slice(&src[i..i + 8]);
                    }
                    sc[(t * kb + b) * 32 + (gid >> 1) * 8 + h * 4 + (gid & 1) * 2 + gs] = ssrc[2 * b + gs];
                }
            }
        }
    }
}

/// Stored bf16 "B32" (copy of fp4l `bf16_frag`): `[n][k]` -> W'[t][g][h][lane][8], rows padded to a multiple of 16
/// with zeros. Parallel over tiles.
pub fn bf16_frag(bits: &[u16], n: usize, k: usize) -> Vec<u16> {
    assert!(k % 32 == 0 && bits.len() == n * k, "bf16_frag: [{n}][{k}] shape ({} values)", bits.len());
    let (nt, g) = (n.div_ceil(16), k / 32);
    let mut w = vec![0u16; nt * 16 * k];
    let th = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(8).clamp(1, 16);
    let per = nt.div_ceil(th).max(1);
    std::thread::scope(|sco| {
        for (ci, wc) in w.chunks_mut(per * 16 * k).enumerate() {
            sco.spawn(move || {
                for tl in 0..wc.len() / (16 * k) {
                    for rr in 0..16 {
                        let row = 16 * (ci * per + tl) + rr;
                        if row >= n { break; }
                        let (h, gid) = (rr >> 3, rr & 7);
                        let src = &bits[row * k..(row + 1) * k];
                        for gg in 0..g {
                            for tig in 0..4 {
                                let o = ((tl * g + gg) * 2 + h) * 256 + (gid * 4 + tig) * 8;
                                let i = 32 * gg + 8 * tig;
                                wc[o..o + 8].copy_from_slice(&src[i..i + 8]);
                            }
                        }
                    }
                }
            });
        }
    });
    w
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn w13_tile_rows_roundtrip() {
        let (n, rb) = (32usize, 3usize);
        let src: Vec<u8> = (0..n * rb).map(|i| (i / rb) as u8).collect();
        let o = w13_tile_rows(&src, n, rb);
        for p in 0..n {
            let (up, i) = w13_row_of(p);
            assert_eq!(o[p * rb] as usize, 2 * i + up as usize);
        }
    }

    #[test]
    fn nv4_frag_64_channel_half_block_is_permutation() {
        for &k in &[128usize, 192, 320, 576] {
            let n = 32usize;
            let codes: Vec<u8> = (0..n * k / 2).map(|i| ((i * 13 + i / 17) & 0xFF) as u8).collect();
            let scales: Vec<u8> = (0..n * k / 16).map(|i| ((i * 7 + 1) & 0xFF) as u8).collect();
            let (mut w, mut sc) = (vec![0u8; codes.len()], vec![0u8; scales.len()]);
            nv4_frag_into(&codes, &scales, n, k, &mut w, &mut sc);
            let (mut a, mut b) = (codes.clone(), w.clone());
            a.sort_unstable(); b.sort_unstable();
            assert_eq!(a, b, "codes k={k}");
            let (mut sa, mut sb) = (scales.clone(), sc.clone());
            sa.sort_unstable(); sb.sort_unstable();
            assert_eq!(sa, sb, "scales k={k}");
        }
    }
}
