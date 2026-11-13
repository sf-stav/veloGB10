//! exl3 1.6.3 "ngram" token-embedding decode (1.66bpw-class packs).
//!
//! 1.66 packs carry no plain `embed_tokens.weight`; instead `embed_tokens.trellis`
//! (I16 rings) + `embed_tokens.signs` (F16). Mirrors
//! `exllamav3/modules/quant/exl3_lib/ngram_codec.py` (TableEmbedding):
//!
//! * a row is `groups` consecutive rings; ring = 1 fp16 scale word +
//!   `dim*K/16` little-endian u16 stream words;
//! * stream bits `[i*K, (i+1)*K)` are the low K bits of position i's 16-bit
//!   tail-biting trellis state; the state's higher bits are the symbols of the
//!   preceding positions, K bits each (mod dim);
//! * `codebook[state]` comes from the mul1 hash: byte-sum of `(s*0x83DCD12D)
//!   & 0xFFFFFFFF`, mapped `(1024+bsum)*k_inv + k_bias` rounded to fp16
//!   (k_inv = f16 0x1eee, k_bias = f16 0xc931);
//! * group value = hadamard(codebook[state] * scale) * signs[g], all f32,
//!   rounded once to f16.
//!
//! Bit-exactness vs the reference is pinned by `tests/embed_ngram_golden.json`
//! (4 real rows of Qwen3.8-Flash-Next-EXL3-1.66bpw decoded with the 1.6.3 wheel).

use anyhow::{bail, Result};
use half::f16;

/// The full 65536-entry mul1 codebook as f16 bit patterns (fp16, bit-exact with
/// `decode_3inst<2>` per ngram_codec.mul1_codebook).
pub fn mul1_codebook() -> &'static [u16; 65536] {
    static CB: std::sync::OnceLock<Box<[u16; 65536]>> = std::sync::OnceLock::new();
    CB.get_or_init(|| {
        let k_inv = f16::from_bits(0x1eee).to_f32();
        let k_bias = f16::from_bits(0xc931).to_f32();
        let mut cb = Box::new([0u16; 65536]);
        for (s, v) in cb.iter_mut().enumerate() {
            let prod = (s as u64 * 0x83DCD12Du64) & 0xFFFF_FFFF;
            let bsum = (prod & 255) + ((prod >> 8) & 255) + ((prod >> 16) & 255) + ((prod >> 24) & 255);
            // h*k_inv + k_bias evaluated as mul-then-add in f32 (no FMA): mirrors torch.
            *v = f16::from_f32((1024.0 + bsum as f32) * k_inv + k_bias).to_bits();
        }
        cb
    })
}

/// Decode one ring (`words` = 1 scale word + stream words) into `out` (dim f32 values):
/// states -> codebook*scale -> in-place orthonormal Hadamard -> * dim^-0.5.
fn decode_ring(words: &[u16], k: usize, cb: &[u16; 65536], out: &mut [f32]) {
    let stream_words = words.len() - 1;
    let dim = stream_words * 16 / k;
    assert_eq!(dim, out.len(), "ring dim must equal group dim");
    assert!(dim.is_power_of_two(), "hadamard needs a power-of-two dim");
    let scale = f16::from_bits(words[0]).to_f32();

    // Symbols of every position through a two-word window (tail-biting wrap).
    let mut sym = vec![0u16; dim];
    for (i, s) in sym.iter_mut().enumerate() {
        let bit = i * k;
        let w0 = words[1 + (bit >> 4)] as u32;
        let w1 = words[1 + ((bit >> 4) + 1) % stream_words] as u32;
        let window = w0 | (w1 << 16);
        *s = ((window >> (bit & 15)) & ((1 << k) - 1)) as u16;
    }
    // State i stacks the symbols of positions i, i-1, ... from bit 0 up.
    for i in 0..dim {
        let mut s = sym[i] as u32;
        for j in 1..((15 + k) / k) {
            let prev = (i + dim - (j % dim)) % dim;
            s |= (sym[prev] as u32) << (j * k);
        }
        out[i] = f16::from_bits(cb[(s & 0xFFFF) as usize]).to_f32() * scale;
    }
    // Butterfly Hadamard in the reference's stage order, then orthonormal scale.
    let mut h = 1;
    while h < dim {
        let mut a = 0;
        while a < dim {
            for b in 0..h {
                let (x, y) = (out[a + b], out[a + b + h]);
                out[a + b] = x + y;
                out[a + b + h] = x - y;
            }
            a += 2 * h;
        }
        h *= 2;
    }
    let inv = (dim as f64).powf(-0.5) as f32;
    for v in out.iter_mut() {
        *v *= inv;
    }
}

/// Dequantize the whole token-embedding table: `trellis` (`rows * row_words` u16),
/// `sign_bits` (`groups * group_dim` f16 bits) -> `rows * groups * group_dim` f16 bits
/// (row-major), bit-identical to the reference `dequant_rows(...).to(float16)`.
pub fn dequant_embed(
    trellis: &[u16],
    sign_bits: &[u16],
    rows: usize,
    row_words: usize,
    groups: usize,
    group_dim: usize,
) -> Result<Vec<u16>> {
    if groups == 0 || group_dim == 0 || row_words == 0 || rows == 0 {
        bail!("ngram embed: empty tensor (rows={rows} words={row_words} groups={groups} dim={group_dim})");
    }
    if trellis.len() != rows * row_words {
        bail!("ngram embed: trellis has {} words, expected {rows}x{row_words}", trellis.len());
    }
    if sign_bits.len() != groups * group_dim {
        bail!("ngram embed: signs has {} words, expected {groups}x{group_dim}", sign_bits.len());
    }
    if row_words % groups != 0 {
        bail!("ngram embed: {row_words} row words not divisible into {groups} groups");
    }
    let per = row_words / groups;
    if (per - 1) * 16 % group_dim != 0 {
        bail!("ngram embed: ring of {per} words does not cover dim {group_dim} at integer K");
    }
    let k = (per - 1) * 16 / group_dim;
    if !(1..=16).contains(&k) {
        bail!("ngram embed: implausible K={k} (per={per} dim={group_dim})");
    }
    let cb = mul1_codebook();
    let hidden = groups * group_dim;
    let mut out = vec![0u16; rows * hidden];
    let nth = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(8).clamp(1, 16);
    let rows_per = rows.div_ceil(nth);
    std::thread::scope(|scope| {
        for (t, chunk) in out.chunks_mut(rows_per * hidden).enumerate() {
            let cb = &*cb;
            scope.spawn(move || {
                let mut buf = vec![0f32; group_dim];
                let base = t * rows_per;
                for (ri, row) in chunk.chunks_mut(hidden).enumerate() {
                    let rings = &trellis[(base + ri) * row_words..(base + ri + 1) * row_words];
                    for g in 0..groups {
                        decode_ring(&rings[g * per..(g + 1) * per], k, cb, &mut buf);
                        let sg = &sign_bits[g * group_dim..(g + 1) * group_dim];
                        for i in 0..group_dim {
                            let s = f16::from_bits(sg[i]).to_f32();
                            let v = buf[i] * s;
                            // torch .to(float16) canonicalizes NaN payloads to 0x7FFF; the pack's
                            // signs table carries 19 NaN entries (2 per group), so mirror it bit-exactly.
                            row[g * group_dim + i] = if v.is_nan() { 0x7FFF } else { f16::from_f32(v).to_bits() };
                        }
                    }
                }
            });
        }
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u16s(v: &serde_json::Value, key: &str) -> Vec<u16> {
        // bits are dumped as signed i16 JSON ints; u16 view is the bit pattern
        v[key].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as u16).collect()
    }

    /// Real-pack golden: decoded by the 1.6.3 wheel's own ngram_codec on .13
    /// (tool_probe/embed_golden.py, 2026-10-30). The port must be bit-identical.
    #[test]
    fn golden_rows_bit_exact() { run_golden("tests/embed_ngram_golden.json"); }

    /// Extended golden: 16 SCATTERED rows (0..248319) — guards against a decode that
    /// only matches the head rows (found via the 1.66 ppl-29 bug hunt, 2026-10-30).
    #[test]
    fn golden_rows_ext_bit_exact() { run_golden("tests/embed_ngram_golden_ext.json"); }

    fn run_golden(path: &str) {
        let j = std::fs::read_to_string(path).unwrap();
        let v: serde_json::Value = serde_json::from_str(&j).unwrap();
        let g = |k: &str| v[k].as_u64().unwrap() as usize;
        let (k, groups, row_words, hidden, nrows) =
            (g("K"), g("groups"), g("row_words"), g("hidden"), g("nrows"));

        let cb8 = u16s(&v, "codebook_first8_bits");
        let cb = mul1_codebook();
        for i in 0..cb8.len() {
            assert_eq!(cb[i], cb8[i], "codebook[{i}] mismatch vs reference");
        }

        let out = dequant_embed(
            &u16s(&v, "trellis_bits"),
            &u16s(&v, "signs_bits"),
            nrows,
            row_words,
            groups,
            hidden / groups,
        )
        .unwrap();
        assert_eq!(k, 4);
        let gold = u16s(&v, "out_f16_bits");
        assert_eq!(out.len(), gold.len());
        let mut bad = Vec::new();
        for (i, (a, b)) in out.iter().zip(gold.iter()).enumerate() {
            if a != b && bad.len() < 8 {
                bad.push((i, *a, *b));
            }
        }
        assert!(bad.is_empty(), "ngram embed decode differs from reference: {bad:?}");
    }
}
