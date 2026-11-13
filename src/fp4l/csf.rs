//! FP4L — the "csf" compressed scale format of NVIDIA ModelOpt NVFP4 checkpoints
//! (`weight_scale_encoding: "csf"`, codec `byte-window4-fixed-stream-u24-exceptions/1`).
//!
//! A `[rows, cols]` E4M3 block-scale matrix (one byte per 16 weights along K) is stored as two
//! safetensors tensors next to the weight:
//!
//! * `<name>.weight_scale.nvfp4_csf_fixed` — U8 `[rows/16, 16 + 8*cols]`. Per 16-row slab: 16 base
//!   bytes (one per row, each <= 240), then the 16 rows' 4-bit codes, `cols/2` bytes per row, LOW
//!   nibble = even column. `scale = base + code` (integer add on the raw byte; no float math).
//! * `<name>.weight_scale.nvfp4_csf_exceptions` — U32 LE words, strictly increasing by position:
//!   low 24 bits = flat position `row*cols + col`, high 8 bits = the scale byte stored there
//!   (a value outside its row's window `[base, base + 15]`). The nibble at an exception is 0.
//!
//! Sources (2026-10-28): vLLM draft PR #60646 `csf_scales.py` (Apache-2.0) and TensorFold PR #904
//! `nvfp4_csf.zig` agree on the layout; the ENCODER rule (`encode`) was recovered from the
//! checkpoint itself (FP4L-0): `base` = the lowest value in `0..=240` whose 16-wide window holds the
//! most of the row's scales; out-of-window scales become exceptions with nibble 0. `encode(decode(s))
//! == s` byte-for-byte over every csf matrix of `nvidia/Qwen3.8-Flash-Next-FP4-Local` @ b9ce18a is
//! the losslessness receipt (`--probe-fp4l`), alongside a cross-check against the public decoder.
//!
//! Decoding is load-time host work: integer adds over ~7.5 GB of scales for this pack.

use anyhow::{bail, ensure, Result};

/// The codec id this module implements (the only csf variant this pack uses; GLM packs use
/// `csf_32x2_v1`, five tensors, which is refused by name at the pack reader).
pub const CODEC: &str = "byte-window4-fixed-stream-u24-exceptions/1";
/// Largest legal row base (base + 15 must fit a byte).
pub const BASE_MAX: u8 = 240;

/// Geometry of a fixed stream: `(rows, cols)` from its `[slabs, 16 + 8*cols]` shape.
pub fn geometry(fixed_shape: &[usize]) -> Result<(usize, usize)> {
    ensure!(fixed_shape.len() == 2, "csf fixed stream must be 2-D, got {:?}", fixed_shape);
    let (slabs, w) = (fixed_shape[0], fixed_shape[1]);
    ensure!(slabs > 0 && w > 16 && w % 16 == 0, "csf fixed stream shape {:?} is not [slabs, 16 + 8*cols]", fixed_shape);
    let cols = (w / 16 - 1) * 2;
    let rows = slabs * 16;
    ensure!(rows.checked_mul(cols).map_or(false, |n| n <= 1 << 24),
            "csf plane {rows}x{cols} exceeds the 2^24 u24 position space");
    Ok((rows, cols))
}

/// Decode one scale plane. `fixed` is the raw U8 stream, `exceptions` the raw LE bytes of the U32
/// stream. Returns `rows*cols` scale bytes (row-major) — the uncompressed checkpoint's E4M3 bytes.
pub fn decode(fixed: &[u8], fixed_shape: &[usize], exceptions: &[u8]) -> Result<Vec<u8>> {
    let (rows, cols) = geometry(fixed_shape)?;
    let stride = 16 + 8 * cols;
    ensure!(fixed.len() == (rows / 16) * stride, "csf fixed stream has {} bytes, want {}", fixed.len(), (rows / 16) * stride);
    ensure!(exceptions.len() % 4 == 0, "csf exception stream length {} is not a multiple of 4", exceptions.len());
    let mut out = vec![0u8; rows * cols];
    let half = cols / 2;
    for slab in 0..rows / 16 {
        let s = &fixed[slab * stride..(slab + 1) * stride];
        for r16 in 0..16 {
            let base = s[r16];
            if base > BASE_MAX {
                bail!("csf row base {base} > {BASE_MAX} (row {})", slab * 16 + r16);
            }
            let codes = &s[16 + r16 * half..16 + (r16 + 1) * half];
            let o = &mut out[(slab * 16 + r16) * cols..(slab * 16 + r16 + 1) * cols];
            for (j, &b) in codes.iter().enumerate() {
                o[2 * j] = base + (b & 0x0F);
                o[2 * j + 1] = base + (b >> 4);
            }
        }
    }
    let mut last: i64 = -1;
    for w in exceptions.chunks_exact(4) {
        let word = u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
        let pos = (word & 0x00FF_FFFF) as usize;
        ensure!((pos as i64) > last, "csf exceptions not strictly increasing at position {pos}");
        ensure!(pos < rows * cols, "csf exception position {pos} outside the {rows}x{cols} plane");
        out[pos] = (word >> 24) as u8;
        last = pos as i64;
    }
    Ok(out)
}

/// Encode a `[rows, cols]` scale plane with the producer's rule (see the module doc). Returns
/// `(fixed, exceptions_le_bytes)`. Used by the round-trip receipt and the unit tests.
pub fn encode(scales: &[u8], rows: usize, cols: usize) -> Result<(Vec<u8>, Vec<u8>)> {
    ensure!(rows % 16 == 0 && rows > 0 && cols % 2 == 0 && cols > 0, "csf encode wants rows % 16 == 0 and even cols, got {rows}x{cols}");
    ensure!(scales.len() == rows * cols, "csf encode: {} bytes for {rows}x{cols}", scales.len());
    ensure!(rows * cols <= 1 << 24, "csf encode: plane exceeds the u24 position space");
    let stride = 16 + 8 * cols;
    let half = cols / 2;
    let mut fixed = vec![0u8; (rows / 16) * stride];
    let mut exc: Vec<u8> = Vec::new();
    for r in 0..rows {
        let row = &scales[r * cols..(r + 1) * cols];
        let mut hist = [0u32; 256];
        for &v in row { hist[v as usize] += 1; }
        // window count for base b = sum hist[b..=b+15]; the lowest b with the maximum wins
        let mut win: u32 = hist[0..16].iter().sum();
        let (mut best_b, mut best_n) = (0usize, win);
        for b in 1..=BASE_MAX as usize {
            win = win - hist[b - 1] + hist[b + 15];
            if win > best_n { best_n = win; best_b = b; }
        }
        let base = best_b as u8;
        let slab = &mut fixed[(r / 16) * stride..(r / 16 + 1) * stride];
        slab[r % 16] = base;
        let codes = &mut slab[16 + (r % 16) * half..16 + (r % 16 + 1) * half];
        for (c, &v) in row.iter().enumerate() {
            let nib = if v >= base && v - base <= 15 { v - base } else {
                let word = ((v as u32) << 24) | ((r * cols + c) as u32);
                exc.extend_from_slice(&word.to_le_bytes());
                0
            };
            if c % 2 == 0 { codes[c / 2] |= nib; } else { codes[c / 2] |= nib << 4; }
        }
    }
    Ok((fixed, exc))
}

/// Rows `[r0, r1)` of a plane, decoded without materialising the rest (TP row slices and
/// expert-parallel loads read only what they keep). `r0`, `r1` need not be slab-aligned.
pub fn decode_rows(fixed: &[u8], fixed_shape: &[usize], exceptions: &[u8], r0: usize, r1: usize) -> Result<Vec<u8>> {
    let (rows, cols) = geometry(fixed_shape)?;
    ensure!(r0 <= r1 && r1 <= rows, "csf decode_rows [{r0}, {r1}) outside {rows} rows");
    let stride = 16 + 8 * cols;
    ensure!(fixed.len() == (rows / 16) * stride, "csf fixed stream has {} bytes, want {}", fixed.len(), (rows / 16) * stride);
    ensure!(exceptions.len() % 4 == 0, "csf exception stream length {} is not a multiple of 4", exceptions.len());
    let half = cols / 2;
    let mut out = vec![0u8; (r1 - r0) * cols];
    for r in r0..r1 {
        let s = &fixed[(r / 16) * stride..(r / 16 + 1) * stride];
        let base = s[r % 16];
        ensure!(base <= BASE_MAX, "csf row base {base} > {BASE_MAX} (row {r})");
        let codes = &s[16 + (r % 16) * half..16 + (r % 16 + 1) * half];
        let o = &mut out[(r - r0) * cols..(r - r0 + 1) * cols];
        for (j, &b) in codes.iter().enumerate() {
            o[2 * j] = base + (b & 0x0F);
            o[2 * j + 1] = base + (b >> 4);
        }
    }
    let (lo, hi) = (r0 * cols, r1 * cols);
    let mut last: i64 = -1;
    for w in exceptions.chunks_exact(4) {
        let word = u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
        let pos = (word & 0x00FF_FFFF) as usize;
        ensure!((pos as i64) > last && pos < rows * cols, "csf exception stream malformed at position {pos}");
        last = pos as i64;
        if pos >= lo && pos < hi { out[pos - lo] = (word >> 24) as u8; }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> u64 { *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); *seed >> 33 }

    #[test]
    fn geometry_matches_the_pack_shapes() {
        // down_proj scales [2560, 40] -> fixed [160, 336]; gate/up [640, 160] -> [40, 1296]
        assert_eq!(geometry(&[160, 336]).unwrap(), (2560, 40));
        assert_eq!(geometry(&[40, 1296]).unwrap(), (640, 160));
        assert!(geometry(&[40, 16]).is_err());
        assert!(geometry(&[40, 1297]).is_err());
    }

    #[test]
    fn round_trip_random_planes_every_byte_value() {
        let mut s = 7u64;
        for &(rows, cols) in &[(16usize, 2usize), (32, 40), (640, 160), (64, 2)] {
            let mut plane = vec![0u8; rows * cols];
            for (i, v) in plane.iter_mut().enumerate() {
                // mostly clustered (like real scales), plus outliers covering all 256 values
                *v = if i % 37 == 0 { (lcg(&mut s) % 256) as u8 } else { 90 + (lcg(&mut s) % 20) as u8 };
            }
            let (f, e) = encode(&plane, rows, cols).unwrap();
            let shape = [rows / 16, 16 + 8 * cols];
            assert_eq!(decode(&f, &shape, &e).unwrap(), plane);
            let (f2, e2) = encode(&decode(&f, &shape, &e).unwrap(), rows, cols).unwrap();
            assert_eq!((f2, e2), (f.clone(), e.clone()));
            let mid = rows / 2 + 3;
            assert_eq!(decode_rows(&f, &shape, &e, 5, mid).unwrap(), plane[5 * cols..mid * cols].to_vec());
        }
    }

    #[test]
    fn handwritten_slab_and_exception() {
        // 16 rows x 2 cols: row r has bases r, codes (1, 2); row 3 col 1 is an exception 0xFF.
        let (rows, cols) = (16usize, 2usize);
        let mut fixed = vec![0u8; 16 + 8 * cols];
        for r in 0..16 { fixed[r] = r as u8; fixed[16 + r] = 0x21; }
        fixed[16 + 3] = 0x01; // the exception slot's nibble is 0
        let exc = ((0xFFu32 << 24) | (3 * 2 + 1) as u32).to_le_bytes().to_vec();
        let d = decode(&fixed, &[1, 16 + 8 * cols], &exc).unwrap();
        for r in 0..16 {
            assert_eq!(d[r * cols], r as u8 + 1);
            assert_eq!(d[r * cols + 1], if r == 3 { 0xFF } else { r as u8 + 2 });
        }
        let _ = rows;
    }

    #[test]
    fn malformed_streams_are_refused() {
        let cols = 2usize;
        let mut fixed = vec![0u8; 16 + 8 * cols];
        fixed[0] = 241;
        assert!(decode(&fixed, &[1, 32], &[]).is_err(), "base > 240");
        fixed[0] = 0;
        let w = |p: u32| ((7u32 << 24) | p).to_le_bytes();
        let mut e = Vec::new(); e.extend_from_slice(&w(5)); e.extend_from_slice(&w(5));
        assert!(decode(&fixed, &[1, 32], &e).is_err(), "duplicate position");
        assert!(decode(&fixed, &[1, 32], &w(32)).is_err(), "position out of range");
        assert!(decode(&fixed, &[1, 32], &[1, 2, 3]).is_err(), "ragged exception stream");
    }
}
