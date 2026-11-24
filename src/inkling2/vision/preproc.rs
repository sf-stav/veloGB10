//! Image bytes -> the tower's input, exactly as `Inkling2ImageProcessor` (transformers, torchvision backend):
//! decode (RGBA kept when the image has transparency), checkerboard composite, pad bottom / right to multiples of
//! 40 with -1.0 on the 0..255 scale, `(x - 127.5) / 127.5` in f32 (the processor fuses rescale 1/255 and
//! normalize 0.5 / 0.5 into mean = std = 127.5), cut into one 40x40x3 tile per LM token (tile raster order,
//! inside a tile (y, x, c)).
//!
//! The reference has NO resize and no size limit. This build keeps a per-image token cap (`--inkling2-image-max-tokens`,
//! shown at boot): an image above it is downscaled (Lanczos3, aspect kept) to the largest size whose tile count fits,
//! with one log line and a counter; nothing is refused (AGENTS 1b-bis 8). Below the cap the input is the reference's
//! bit for bit (PNG); a JPEG decodes through the `image` crate, not libjpeg-turbo (PIL), so its pixels may differ by
//! the decoders' IDCT rounding (measured in the S5 report).
//!
//! Inline images only: `data:<mime>;base64,<payload>` (and bare base64). Remote URLs are not fetched.

use anyhow::{anyhow, bail, ensure, Result};
use base64::Engine;

use super::Grid;

/// Decoded-pixel ceiling for one input image (checked from the header before decoding), as the other families.
pub const MAX_INPUT_PIXELS: u64 = 64_000_000;
/// Default per-image LM token cap (= tiles of 40x40): 4,096 tokens is ~2,560 x 2,560 pixels.
pub const DEFAULT_MAX_TOKENS: usize = 4096;

/// One preprocessed image.
#[derive(Clone)]
pub struct PreImage {
    pub grid: Grid,
    /// `[tokens][40][40][3]` f32 in [-1.0078, 1]
    pub pixels: Vec<f32>,
    /// content key: FNV-1a over the grid and the pixel bits (what a prompt / tower cache must key on)
    pub hash: u64,
    /// decoded size (w, h)
    pub src: (usize, usize),
    /// Some((w, h)) when the token cap downscaled the image
    pub resized: Option<(usize, usize)>,
    pub had_alpha: bool,
}

/// The base64 payload of a data URL (or bare base64) to bytes.
pub fn url_bytes(url: &str) -> Result<Vec<u8>> {
    let u = url.trim();
    if u.starts_with("http://") || u.starts_with("https://") || u.starts_with("file:") {
        bail!("remote image URLs are not fetched by this server; send the image inline as a base64 data: URL");
    }
    let b64 = if u.starts_with("data:") {
        let (head, payload) = u.split_once(',').ok_or_else(|| anyhow!("malformed data URL (no ',')"))?;
        ensure!(head.contains(";base64"), "data URL is not base64-encoded ({head})");
        payload
    } else { u };
    let clean: String = b64.chars().filter(|c| !c.is_whitespace()).collect();
    base64::engine::general_purpose::STANDARD.decode(clean.as_bytes())
        .or_else(|_| base64::engine::general_purpose::STANDARD_NO_PAD.decode(clean.trim_end_matches('=').as_bytes()))
        .map_err(|e| anyhow!("image base64: {e}"))
}

/// Decode to RGB8 the way the reference does: an image with transparency is composited onto the 8x8 checkerboard
/// (204 / 153) with `(rgb * a + bg * (255 - a) + 127) // 255`; otherwise plain RGB conversion. Returns (rgb, w, h, alpha?).
pub fn decode_rgb(bytes: &[u8]) -> Result<(Vec<u8>, usize, usize, bool)> {
    let rd = image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format()?;
    let (w0, h0) = rd.into_dimensions()?;
    ensure!((w0 as u64) * (h0 as u64) <= MAX_INPUT_PIXELS, "image {w0}x{h0} exceeds the decode limit of {} MP", MAX_INPUT_PIXELS / 1_000_000);
    ensure!(w0 > 0 && h0 > 0, "empty image");
    let img = image::load_from_memory(bytes)?;
    let (w, h) = (img.width() as usize, img.height() as usize);
    if img.color().has_alpha() {
        let rgba = img.to_rgba8();
        let mut out = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let p = &rgba.as_raw()[(y * w + x) * 4..(y * w + x) * 4 + 4];
                let bg: u32 = if (y / 8 + x / 8) % 2 == 0 { 204 } else { 153 };
                let a = p[3] as u32;
                for c in 0..3 { out[(y * w + x) * 3 + c] = ((p[c] as u32 * a + bg * (255 - a) + 127) / 255) as u8; }
            }
        }
        Ok((out, w, h, true))
    } else {
        Ok((img.to_rgb8().into_raw(), w, h, false))
    }
}

/// Tiles of a (w, h) image.
pub fn grid_of(w: usize, h: usize, tile: usize) -> Grid { Grid { rows: h.div_ceil(tile), cols: w.div_ceil(tile) } }

/// The largest (w', h') <= (w, h), aspect kept, whose tile count is <= `max_tokens`.
pub fn capped_dims(w: usize, h: usize, tile: usize, max_tokens: usize) -> (usize, usize) {
    if grid_of(w, h, tile).tokens() <= max_tokens { return (w, h); }
    let mut s = ((max_tokens * tile * tile) as f64 / (w * h) as f64).sqrt();
    loop {
        let (nw, nh) = (((w as f64 * s).floor() as usize).max(1), ((h as f64 * s).floor() as usize).max(1));
        if grid_of(nw, nh, tile).tokens() <= max_tokens { return (nw, nh); }
        s *= 0.995;
    }
}

/// RGB8 -> padded, normalized tiles.
pub fn tiles(rgb: &[u8], w: usize, h: usize, tile: usize) -> (Vec<f32>, Grid) {
    let g = grid_of(w, h, tile);
    let mut px = vec![0f32; g.tokens() * tile * tile * 3];
    let pad = (-1.0f32 - 127.5f32) / 127.5f32;
    for tr in 0..g.rows {
        for tc in 0..g.cols {
            let base = (tr * g.cols + tc) * tile * tile * 3;
            for yy in 0..tile {
                let y = tr * tile + yy;
                for xx in 0..tile {
                    let x = tc * tile + xx;
                    let o = base + (yy * tile + xx) * 3;
                    if y < h && x < w {
                        for c in 0..3 { px[o + c] = (rgb[(y * w + x) * 3 + c] as f32 - 127.5f32) / 127.5f32; }
                    } else {
                        for c in 0..3 { px[o + c] = pad; }
                    }
                }
            }
        }
    }
    (px, g)
}

pub fn fnv(h: &mut u64, bytes: &[u8]) {
    for &b in bytes { *h ^= b as u64; *h = h.wrapping_mul(0x100000001b3); }
}

/// Bytes -> PreImage (decode, composite, cap, pad, normalize, tile).
pub fn preprocess(bytes: &[u8], tile: usize, max_tokens: usize) -> Result<PreImage> {
    let (rgb, w, h, alpha) = decode_rgb(bytes)?;
    let (nw, nh) = capped_dims(w, h, tile, max_tokens.max(1));
    let (rgb2, resized) = if (nw, nh) != (w, h) {
        let im = image::RgbImage::from_raw(w as u32, h as u32, rgb).ok_or_else(|| anyhow!("rgb buffer"))?;
        let r = image::imageops::resize(&im, nw as u32, nh as u32, image::imageops::FilterType::Lanczos3);
        (r.into_raw(), Some((nw, nh)))
    } else { (rgb, None) };
    let (pixels, grid) = tiles(&rgb2, nw, nh, tile);
    let mut hs = 0xcbf29ce484222325u64;
    fnv(&mut hs, &(grid.rows as u64).to_le_bytes());
    fnv(&mut hs, &(grid.cols as u64).to_le_bytes());
    for p in &pixels { fnv(&mut hs, &p.to_bits().to_le_bytes()); }
    Ok(PreImage { grid, pixels, hash: hs, src: (w, h), resized, had_alpha: alpha })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cap_keeps_aspect_and_fits() {
        let (w, h) = capped_dims(4000, 3000, 40, 4096);
        assert!(grid_of(w, h, 40).tokens() <= 4096);
        assert!((w as f64 / h as f64 - 4.0 / 3.0).abs() < 0.01);
        assert_eq!(capped_dims(640, 480, 40, 4096), (640, 480));
    }
    #[test]
    fn rgba_checkerboard_and_data_url() {
        // 16x9 RGBA: pixel (x, y) = (200, 100, 50, alpha 0 left half / 255 right half)
        let mut im = image::RgbaImage::new(16, 9);
        for (x, _y, p) in im.enumerate_pixels_mut() { *p = image::Rgba([200, 100, 50, if x < 8 { 0 } else { 255 }]); }
        let mut png = Vec::new();
        image::DynamicImage::ImageRgba8(im).write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).unwrap();
        let url = format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(&png));
        let pre = preprocess(&url_bytes(&url).unwrap(), 40, 4096).unwrap();
        assert!(pre.had_alpha);
        assert_eq!((pre.grid.rows, pre.grid.cols), (1, 1));
        let px = |x: usize, y: usize, c: usize| pre.pixels[(y * 40 + x) * 3 + c];
        let n = |v: f32| (v - 127.5) / 127.5;
        assert_eq!(px(0, 0, 0), n(204.0));          // transparent, checker square (0, 0) -> 204
        assert_eq!(px(3, 8, 1), n(153.0));          // transparent, square (0, 1) -> 153
        assert_eq!(px(9, 2, 0), n(200.0));          // opaque keeps its colour
        assert_eq!(px(20, 0, 2), n(-1.0));          // padding
        assert!(url_bytes("https://example.com/a.png").is_err());
    }
    #[test]
    fn pad_value_and_tile_order() {
        // 41x1 image: 2 tiles in a row, the second holds pixel x=40 at (0,0) and padding elsewhere
        let rgb: Vec<u8> = (0..41 * 3).map(|i| (i % 256) as u8).collect();
        let (px, g) = tiles(&rgb, 41, 1, 40);
        assert_eq!((g.rows, g.cols), (1, 2));
        let t1 = 40 * 40 * 3;
        assert_eq!(px[t1], (120.0f32 - 127.5) / 127.5);
        assert_eq!(px[t1 + 3], (-1.0f32 - 127.5) / 127.5);
        assert_eq!(px[40 * 3], (-1.0f32 - 127.5) / 127.5); // row 1 of tile 0 is padding
    }
}
