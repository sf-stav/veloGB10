//! DeepSeek-V4.1-Flash image preprocessing and placeholder expansion (`inference/image_processor.py`).
//!
//! Security & API contract:
//! * Accepts inline base64 (`data`, Messages API `source.data`, OpenAI `data:image/...;base64,...` URLs).
//! * Rejects `http://`, `https://`, non-base64 `data:` URLs, and local filesystem paths with 400
//!   `ApiError` (`code: Some("invalid_image_url")`).
//! * Enforces `--dsv41-max-images` (default 16, 0 = unlimited) and decoded byte / pixel bounds.
//! * Exact `plan_image_grid` / `safe_resize` / `solve_resize_ratio`, exact Pillow-compatible
//!   separable bicubic resampler + `ImageOps.pad(..., color=(127, 127, 127))`, `((rgb/255)-0.5)/0.5`
//!   rounded to BF16, reshaped to `[n_vit_h * n_vit_w, 3 * patch * patch]`.
//! * Expands each `image_token_id` (`129264`) into `n_llm_h * (n_llm_w + 1) + 2` copies of
//!   `image_token_id` with `token_types` (`TEXT = -1`, `IMAGE_START = 0`, `IMAGE = 1`,
//!   `IMAGE_NEW_LINE = 2`, `IMAGE_END = 3`) and computes a SHA-256 `ImageSpanKey` per image span.

use std::io::Cursor;

use base64::Engine as _;
use half::bf16;
use image::{ImageReader, Limits};
use sha2::{Digest, Sha256};

use crate::dsv41::chat::jx;
use crate::dsv41::config::Vision;
use crate::dsv41::serve::api::ApiError;
use crate::dsv4_chat::Json;

pub const TOK_TEXT: i8 = -1;
pub const TOK_IMAGE_START: i8 = 0;
pub const TOK_IMAGE: i8 = 1;
pub const TOK_IMAGE_NEW_LINE: i8 = 2;
pub const TOK_IMAGE_END: i8 = 3;

/// Maximum raw encoded/decoded image file payload (64 MiB).
pub const MAX_IMAGE_BYTES: usize = 64 * 1024 * 1024;
/// Maximum decoded image dimension on either axis.
pub const MAX_IMAGE_DIM: u32 = 16_384;
/// Maximum decoded pixels per image (64 MP).
pub const MAX_IMAGE_PIXELS: u64 = 64 * 1024 * 1024;

/// Prefix-cache identity of one expanded image span `[start, end)` in prompt token coordinates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageSpanKey {
    pub start: usize,
    pub end: usize,
    pub digest: [u8; 32],
}

/// One preprocessed image ready for Rank-0 GPU ViT + Aligner encoding.
#[derive(Clone, Debug)]
pub struct PreparedImage {
    pub start: usize,
    pub end: usize,
    pub n_vit_h: usize,
    pub n_vit_w: usize,
    pub n_llm_h: usize,
    pub n_llm_w: usize,
    /// Flattened `[n_vit_h * n_vit_w, 3 * patch * patch]` BF16 bit patterns (`u16`).
    pub patches: Vec<u16>,
    /// Token types over `start..end` (`0`=START, `1`=IMAGE, `2`=NEW_LINE, `3`=END).
    pub types: Vec<i8>,
    pub digest: [u8; 32],
}

impl PreparedImage {
    pub fn n_patches(&self) -> usize {
        self.n_vit_h * self.n_vit_w
    }
    pub fn n_aligner_tokens(&self) -> usize {
        self.n_llm_h * self.n_llm_w
    }
}

/// Expanded multimodal prompt tokens, per-token VL types, and per-image patch inputs.
#[derive(Clone, Debug, Default)]
pub struct VlPrepared {
    pub prompt_ids: Vec<u32>,
    pub token_types: Vec<i8>,
    pub images: Vec<PreparedImage>,
    pub spans: Vec<ImageSpanKey>,
}

fn bad_img(msg: impl Into<String>) -> ApiError {
    ApiError::bad(msg).code("invalid_image_url")
}

/// `num_image_tokens(n_llm_h, n_llm_w)` (`image_processor.py:35-36`).
pub fn num_image_tokens(n_llm_h: usize, n_llm_w: usize) -> usize {
    n_llm_h * (n_llm_w + 1) + 2
}

/// `llm_grid(best_height, best_width, patch_size, downsample_ratio)` (`image_processor.py:39-43`).
pub fn llm_grid(best_height: usize, best_width: usize, patch_size: usize, downsample_ratio: usize) -> (usize, usize) {
    let vh = best_height / patch_size;
    let vw = best_width / patch_size;
    (vh.div_ceil(downsample_ratio), vw.div_ceil(downsample_ratio))
}

/// `solve_resize_ratio` (`image_processor.py:46-57`).
pub fn solve_resize_ratio(
    height: f64,
    width: f64,
    patch_size: usize,
    downsample_ratio: usize,
    max_n_token: usize,
) -> (usize, usize) {
    let r = height / width;
    let max_w_float = ((max_n_token - 2) as f64 / r + 0.25).sqrt() - 0.5;
    let max_h_float = max_w_float * r;
    let cell = patch_size * downsample_ratio;
    if max_w_float < 1.0 {
        return (((max_n_token - 2) / 2) * cell, cell);
    }
    if max_h_float < 1.0 {
        return (cell, (max_n_token - 3) * cell);
    }
    let beta = (max_w_float.floor() * cell as f64 / width).min(max_h_float.floor() * cell as f64 / height);
    let bh = (height * beta / patch_size as f64).floor() as usize * patch_size;
    let bw = (width * beta / patch_size as f64).floor() as usize * patch_size;
    (bh.max(patch_size), bw.max(patch_size))
}

/// `safe_resize` (`image_processor.py:60-67`).
pub fn safe_resize(
    height: f64,
    width: f64,
    mut best_height: usize,
    mut best_width: usize,
    patch_size: usize,
    downsample_ratio: usize,
    max_n_token: usize,
) -> (usize, usize, usize, usize) {
    let (mut n_llm_h, mut n_llm_w) = llm_grid(best_height, best_width, patch_size, downsample_ratio);
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token {
        (best_height, best_width) = solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token);
        (n_llm_h, n_llm_w) = llm_grid(best_height, best_width, patch_size, downsample_ratio);
        debug_assert!(num_image_tokens(n_llm_h, n_llm_w) <= max_n_token);
    }
    (n_llm_h, n_llm_w, best_height, best_width)
}

/// `plan_image_grid` (`image_processor.py:101-112`): returns `(n_llm_h, n_llm_w, best_height, best_width)`.
pub fn plan_image_grid(width: usize, height: usize, vis: &Vision) -> (usize, usize, usize, usize) {
    let p = vis.patch;
    let mut w = width as f64;
    let mut h = height as f64;
    if let Some(mwr) = vis.max_wh_ratio {
        if w > h * mwr {
            w = h * mwr;
        }
    }
    if w * h > 0.0 && w * h < vis.min_pixels as f64 {
        let ratio = (vis.min_pixels as f64 / (w * h)).sqrt();
        w = (w * ratio).trunc();
        h = (h * ratio).trunc();
    }
    let best_width = ((w / p as f64).ceil() as usize * p).max(p);
    let best_height = ((h / p as f64).ceil() as usize * p).max(p);
    safe_resize(h, w, best_height, best_width, p, vis.downsample, vis.max_image_tokens)
}

/// `image_token_types` (`image_processor.py:132-137`).
pub fn image_token_types(n_llm_h: usize, n_llm_w: usize) -> Vec<i8> {
    let total = num_image_tokens(n_llm_h, n_llm_w);
    let mut out = Vec::with_capacity(total);
    out.push(TOK_IMAGE_START);
    for _ in 0..n_llm_h {
        out.extend(std::iter::repeat_n(TOK_IMAGE, n_llm_w));
        out.push(TOK_IMAGE_NEW_LINE);
    }
    out.push(TOK_IMAGE_END);
    out
}

fn decode_b64(payload: &str) -> Result<Vec<u8>, ApiError> {
    let trimmed = payload.trim();
    if trimmed.is_empty() {
        return Err(bad_img("empty base64 image payload"));
    }
    // Strip embedded ASCII whitespace if MIME-wrapped, otherwise decode directly.
    let clean: std::borrow::Cow<'_, str> = if trimmed.bytes().any(|b| b.is_ascii_whitespace()) {
        std::borrow::Cow::Owned(trimmed.chars().filter(|c| !c.is_ascii_whitespace()).collect())
    } else {
        std::borrow::Cow::Borrowed(trimmed)
    };
    if clean.len() > MAX_IMAGE_BYTES * 4 / 3 + 16 {
        return Err(bad_img(format!("base64 image exceeds maximum allowed size ({MAX_IMAGE_BYTES} bytes)")));
    }
    let b = clean.as_bytes();
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b)
        .or_else(|_| base64::engine::general_purpose::STANDARD_NO_PAD.decode(b))
        .or_else(|_| base64::engine::general_purpose::URL_SAFE.decode(b))
        .or_else(|_| base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(b))
        .map_err(|e| bad_img(format!("invalid base64 image payload: {e}")))?;
    if bytes.is_empty() || bytes.len() > MAX_IMAGE_BYTES {
        return Err(bad_img(format!("decoded image byte length {} out of bounds (1..={MAX_IMAGE_BYTES})", bytes.len())));
    }
    Ok(bytes)
}

fn is_data_base64_header(header: &str) -> bool {
    let Some(rest) = header.get(5..) else { return false };
    rest.split(';').any(|seg| seg.trim().eq_ignore_ascii_case("base64"))
}

fn sniff_known_raster_magic(bytes: &[u8]) -> bool {
    image::guess_format(bytes).is_ok() || (bytes.len() >= 54 && &bytes[0..2] == b"BM")
}

fn decode_url_field(url: &str) -> Result<Vec<u8>, ApiError> {
    let u = url.trim();
    if u.is_empty() {
        return Err(bad_img("image url is empty"));
    }
    if u.get(..5).is_some_and(|p| p.eq_ignore_ascii_case("data:")) {
        let Some((header, payload)) = u.split_once(',') else {
            return Err(bad_img("malformed data: image URL (missing comma separator)"));
        };
        if !is_data_base64_header(header) {
            return Err(bad_img(format!("unsupported data URL encoding '{header}'; expected data:image/...;base64,...")));
        }
        return decode_b64(payload);
    }
    let lc_prefix = u.get(..10).unwrap_or(u).to_ascii_lowercase();
    if lc_prefix.starts_with("http://") || lc_prefix.starts_with("https://") {
        return Err(bad_img(
            "remote http/https image URLs are rejected; provide inline base64 or a data:image/...;base64,... URL",
        ));
    }
    if lc_prefix.starts_with("file://") || lc_prefix.starts_with("ftp://") {
        return Err(bad_img(
            "local filesystem image paths and non-base64 schemes are rejected; provide inline base64 or a data:image/...;base64,... URL",
        ));
    }
    // Accept bare inline base64 payloads (e.g. JPEG '/9j/4AAQ...', PNG 'iVBORw0KGgo...') only when
    // in-memory base64 decoding succeeds AND the decoded header matches a valid raster image magic.
    // Local paths ('/etc/passwd', 'relative/image.png') never touch disk and fail with invalid_image_url.
    if u.len() >= 32 {
        if let Ok(bytes) = decode_b64(u) {
            if sniff_known_raster_magic(&bytes) {
                return Ok(bytes);
            }
        }
    }
    Err(bad_img(
        "local filesystem image paths and non-base64 schemes are rejected; provide inline base64 or a data:image/...;base64,... URL",
    ))
}

/// Extract and validate inline image bytes from a `chat::encode_messages` image record.
pub fn load_image_bytes(record: &Json) -> Result<Vec<u8>, ApiError> {
    if let Some(Json::Str(d)) = jx::get(record, "data") {
        return decode_b64(d);
    }
    if let Some(src @ Json::Object(_)) = jx::get(record, "source") {
        if let Some(Json::Str(d)) = jx::get(src, "data") {
            return decode_b64(d);
        }
        if let Some(Json::Str(u)) = jx::get(src, "url") {
            if !u.is_empty() {
                return decode_url_field(u);
            }
        }
    }
    if let Some(iu) = jx::get(record, "image_url") {
        match iu {
            Json::Str(u) if !u.is_empty() => return decode_url_field(u),
            Json::Object(_) => {
                if let Some(Json::Str(u)) = jx::get(iu, "url") {
                    if !u.is_empty() {
                        return decode_url_field(u);
                    }
                }
            }
            _ => {}
        }
    }
    if let Some(Json::Str(u)) = jx::get(record, "url") {
        if !u.is_empty() {
            return decode_url_field(u);
        }
    }
    Err(bad_img("cannot load image from record: expected inline base64 'data', 'source.data', or 'data:image/...;base64,...' 'url'"))
}

// ---- Exact Pillow-compatible separable bicubic uint8 resampler (`libImaging/Resample.c`) ----

const PRECISION_BITS: i32 = 22; // 32 - 8 - 2

#[inline]
fn cubic_filter(mut x: f64) -> f64 {
    if x < 0.0 {
        x = -x;
    }
    if x < 1.0 {
        ((1.5 * x - 2.5) * x) * x + 1.0
    } else if x < 2.0 {
        ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
    } else {
        0.0
    }
}

struct ResampleTap {
    xmin: usize,
    ksize: usize,
}

fn precompute_bicubic_coeffs(in_size: usize, out_size: usize) -> (usize, Vec<ResampleTap>, Vec<i32>) {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = 2.0 * filterscale;
    let max_ksize = support.ceil() as usize * 2 + 1;
    let ss = 1.0 / filterscale;

    let mut taps = Vec::with_capacity(out_size);
    let mut kk = vec![0i32; out_size * max_ksize];
    let mut tmp = vec![0.0f64; max_ksize];

    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let xmin = (center - support + 0.5).max(0.0) as usize;
        let xmax = ((center + support + 0.5).max(0.0) as usize).min(in_size);
        let ksize = xmax.saturating_sub(xmin).min(max_ksize);
        let mut ww = 0.0f64;
        for x in 0..ksize {
            let w = cubic_filter(((x + xmin) as f64 - center + 0.5) * ss);
            tmp[x] = w;
            ww += w;
        }
        let base = xx * max_ksize;
        let scale_fp = (1i64 << PRECISION_BITS) as f64;
        for x in 0..ksize {
            let w = if ww == 0.0 { 0.0 } else { tmp[x] / ww };
            kk[base + x] = (w * scale_fp).round() as i32;
        }
        taps.push(ResampleTap { xmin, ksize });
    }
    (max_ksize, taps, kk)
}

#[inline]
fn clip8(v: i32) -> u8 {
    (v >> PRECISION_BITS).clamp(0, 255) as u8
}

/// Exact Pillow `Image.resize((out_w, out_h), Resampling.BICUBIC)` on RGB8 pixels.
pub fn pil_resize_rgb(src: &[u8], in_w: usize, in_h: usize, out_w: usize, out_h: usize) -> Vec<u8> {
    if in_w == out_w && in_h == out_h {
        return src.to_vec();
    }
    let bias = 1i32 << (PRECISION_BITS - 1);
    let horiz: Vec<u8> = if in_w == out_w {
        src.to_vec()
    } else {
        let (kmax, taps, kk) = precompute_bicubic_coeffs(in_w, out_w);
        let mut out = vec![0u8; in_h * out_w * 3];
        for y in 0..in_h {
            let row_in = &src[y * in_w * 3..(y + 1) * in_w * 3];
            let row_out = &mut out[y * out_w * 3..(y + 1) * out_w * 3];
            for (xx, tap) in taps.iter().enumerate() {
                let krow = &kk[xx * kmax..xx * kmax + tap.ksize];
                let mut sr = bias;
                let mut sg = bias;
                let mut sb = bias;
                for (x, &c) in krow.iter().enumerate() {
                    let p = (tap.xmin + x) * 3;
                    sr += row_in[p] as i32 * c;
                    sg += row_in[p + 1] as i32 * c;
                    sb += row_in[p + 2] as i32 * c;
                }
                let q = xx * 3;
                row_out[q] = clip8(sr);
                row_out[q + 1] = clip8(sg);
                row_out[q + 2] = clip8(sb);
            }
        }
        out
    };
    if in_h == out_h {
        return horiz;
    }
    let (kmax, taps, kk) = precompute_bicubic_coeffs(in_h, out_h);
    let stride = out_w * 3;
    let mut out = vec![0u8; out_h * stride];
    for (yy, tap) in taps.iter().enumerate() {
        let krow = &kk[yy * kmax..yy * kmax + tap.ksize];
        let row_out = &mut out[yy * stride..(yy + 1) * stride];
        for xx in 0..out_w {
            let col = xx * 3;
            let mut sr = bias;
            let mut sg = bias;
            let mut sb = bias;
            for (y, &c) in krow.iter().enumerate() {
                let p = (tap.xmin + y) * stride + col;
                sr += horiz[p] as i32 * c;
                sg += horiz[p + 1] as i32 * c;
                sb += horiz[p + 2] as i32 * c;
            }
            row_out[col] = clip8(sr);
            row_out[col + 1] = clip8(sg);
            row_out[col + 2] = clip8(sb);
        }
    }
    out
}

/// Exact Pillow `ImageOps.pad(image, (best_w, best_h), color=(127, 127, 127))` (or direct resize when
/// `orig_w >= max_wh_ratio * orig_h`, matching `image_processor.py:122-125`).
pub fn resize_and_pad_rgb(
    rgb: &[u8],
    orig_w: usize,
    orig_h: usize,
    best_w: usize,
    best_h: usize,
    max_wh_ratio: Option<f64>,
) -> Vec<u8> {
    if max_wh_ratio.is_some_and(|mwr| orig_w as f64 >= mwr * orig_h as f64) {
        return pil_resize_rgb(rgb, orig_w, orig_h, best_w, best_h);
    }
    let im_ratio = orig_w as f64 / orig_h as f64;
    let dest_ratio = best_w as f64 / best_h as f64;
    let (mut rw, mut rh) = (best_w, best_h);
    if im_ratio != dest_ratio {
        if im_ratio > dest_ratio {
            let new_h = ((orig_h as f64 / orig_w as f64 * best_w as f64).round_ties_even() as usize).max(1);
            if new_h != best_h {
                rh = new_h;
            }
        } else {
            let new_w = ((orig_w as f64 / orig_h as f64 * best_h as f64).round_ties_even() as usize).max(1);
            if new_w != best_w {
                rw = new_w;
            }
        }
    }
    let resized = pil_resize_rgb(rgb, orig_w, orig_h, rw, rh);
    if rw == best_w && rh == best_h {
        return resized;
    }
    let mut out = vec![127u8; best_h * best_w * 3];
    let (x0, y0) = if rw != best_w {
        (((best_w - rw) as f64 * 0.5).round_ties_even() as usize, 0usize)
    } else {
        (0usize, ((best_h - rh) as f64 * 0.5).round_ties_even() as usize)
    };
    for y in 0..rh {
        let src_row = &resized[y * rw * 3..(y + 1) * rw * 3];
        let dst_off = ((y0 + y) * best_w + x0) * 3;
        out[dst_off..dst_off + rw * 3].copy_from_slice(src_row);
    }
    out
}

fn byte_to_bf16_lut() -> [u16; 256] {
    let mut lut = [0u16; 256];
    for (i, slot) in lut.iter_mut().enumerate() {
        let x = i as f32 / 255.0;
        let norm = (x - 0.5) / 0.5;
        *slot = bf16::from_f32(norm).to_bits();
    }
    lut
}

fn detect_unsupported_container(raw: &[u8]) -> Option<&'static str> {
    if raw.len() >= 12 && &raw[4..8] == b"ftyp" {
        return match &raw[8..12] {
            b"avif" | b"avis" | b"avio" => Some("AVIF"),
            b"heic" | b"heix" | b"hevc" | b"hevx" | b"mif1" | b"msf1" => Some("HEIC/HEIF"),
            _ => None,
        };
    }
    if raw.len() >= 4 && (&raw[0..4] == b"II*\0" || &raw[0..4] == b"MM\0*") {
        return Some("TIFF");
    }
    let trim = raw.iter().copied().skip_while(|b| b.is_ascii_whitespace()).take(64).collect::<Vec<u8>>();
    if trim.starts_with(b"<svg") || (trim.starts_with(b"<?xml") && raw.windows(4).take(512).any(|w| w.eq_ignore_ascii_case(b"<svg"))) {
        return Some("SVG");
    }
    None
}

/// Pure-Rust bounds-checked 24-bit / 32-bit uncompressed BMP (`BM`) decoder so clipboard BMP pastes
/// work without adding extra Cargo features on shared crates.
fn decode_bmp_rgb(raw: &[u8]) -> Result<(usize, usize, Vec<u8>), ApiError> {
    if raw.len() < 54 || &raw[0..2] != b"BM" {
        return Err(bad_img("malformed BMP header"));
    }
    let pixel_offset = u32::from_le_bytes(raw[10..14].try_into().unwrap()) as usize;
    let dib_size = u32::from_le_bytes(raw[14..18].try_into().unwrap()) as usize;
    if dib_size < 40 || raw.len() < 14 + dib_size || pixel_offset > raw.len() {
        return Err(bad_img("unsupported or truncated BMP DIB header"));
    }
    let w_i32 = i32::from_le_bytes(raw[18..22].try_into().unwrap());
    let h_i32 = i32::from_le_bytes(raw[22..26].try_into().unwrap());
    if w_i32 <= 0 || h_i32 == 0 || h_i32 == i32::MIN {
        return Err(bad_img("invalid BMP dimensions"));
    }
    let top_down = h_i32 < 0;
    let (w32, h32) = (w_i32 as u32, h_i32.unsigned_abs());
    if w32 > MAX_IMAGE_DIM || h32 > MAX_IMAGE_DIM || (w32 as u64) * (h32 as u64) > MAX_IMAGE_PIXELS {
        return Err(bad_img(format!("image dimensions {w32}x{h32} exceed limits (max {MAX_IMAGE_DIM}x{MAX_IMAGE_DIM}, {MAX_IMAGE_PIXELS} pixels)")));
    }
    let bpp = u16::from_le_bytes(raw[28..30].try_into().unwrap());
    let comp = u32::from_le_bytes(raw[30..34].try_into().unwrap());
    if !matches!(bpp, 24 | 32) || !matches!(comp, 0 | 3) {
        return Err(bad_img(format!("unsupported BMP encoding (bpp={bpp}, compression={comp}); convert to PNG or JPEG")));
    }
    let (w, h) = (w32 as usize, h32 as usize);
    let bytes_pp = (bpp / 8) as usize;
    let row_stride = (w * bytes_pp).div_ceil(4) * 4;
    let needed = pixel_offset.checked_add(row_stride.checked_mul(h).ok_or_else(|| bad_img("BMP stride overflow"))?)
        .ok_or_else(|| bad_img("BMP offset overflow"))?;
    if raw.len() < needed {
        return Err(bad_img("truncated BMP pixel data"));
    }
    let mut rgb = vec![0u8; w * h * 3];
    for dst_y in 0..h {
        let src_y = if top_down { dst_y } else { h - 1 - dst_y };
        let src_row = &raw[pixel_offset + src_y * row_stride..pixel_offset + (src_y + 1) * row_stride];
        let dst_row = &mut rgb[dst_y * w * 3..(dst_y + 1) * w * 3];
        for x in 0..w {
            let p = x * bytes_pp;
            // Standard Windows BGR / BGRA byte order.
            dst_row[x * 3] = src_row[p + 2];
            dst_row[x * 3 + 1] = src_row[p + 1];
            dst_row[x * 3 + 2] = src_row[p];
        }
    }
    Ok((w, h, rgb))
}

/// Decode, resize/pad, normalize to BF16 patches `[n_vit_h * n_vit_w, 3 * p * p]`, and hash one image.
pub fn decode_and_patchify(raw_bytes: &[u8], vis: &Vision) -> Result<(PreparedImage, (usize, usize)), ApiError> {
    if let Some(fmt_name) = detect_unsupported_container(raw_bytes) {
        return Err(bad_img(format!(
            "unsupported image format ({fmt_name}); please convert the image to PNG, JPEG, WebP, GIF, or BMP"
        )));
    }
    let (orig_w, orig_h, rgb) = if raw_bytes.len() >= 54 && &raw_bytes[0..2] == b"BM" {
        decode_bmp_rgb(raw_bytes)?
    } else {
        let mut limits = Limits::default();
        limits.max_image_width = Some(MAX_IMAGE_DIM);
        limits.max_image_height = Some(MAX_IMAGE_DIM);
        limits.max_alloc = Some(512 * 1024 * 1024);

        let mut reader = ImageReader::new(Cursor::new(raw_bytes))
            .with_guessed_format()
            .map_err(|e| bad_img(format!("failed to read image header: {e}")))?;
        reader.limits(limits);
        let (w32, h32) = reader.into_dimensions().map_err(|e| bad_img(format!("invalid or unsupported image: {e}")))?;
        if w32 == 0 || h32 == 0 || w32 > MAX_IMAGE_DIM || h32 > MAX_IMAGE_DIM || (w32 as u64) * (h32 as u64) > MAX_IMAGE_PIXELS {
            return Err(bad_img(format!("image dimensions {w32}x{h32} exceed limits (max {MAX_IMAGE_DIM}x{MAX_IMAGE_DIM}, {MAX_IMAGE_PIXELS} pixels)")));
        }

        let mut reader = ImageReader::new(Cursor::new(raw_bytes))
            .with_guessed_format()
            .map_err(|e| bad_img(format!("failed to read image: {e}")))?;
        let mut limits2 = Limits::default();
        limits2.max_image_width = Some(MAX_IMAGE_DIM);
        limits2.max_image_height = Some(MAX_IMAGE_DIM);
        limits2.max_alloc = Some(512 * 1024 * 1024);
        reader.limits(limits2);

        let dyn_img = reader.decode().map_err(|e| bad_img(format!("failed to decode image: {e}")))?;
        let rgb_img = dyn_img.into_rgb8();
        let (w, h) = (rgb_img.width() as usize, rgb_img.height() as usize);
        (w, h, rgb_img.into_raw())
    };

    let p = vis.patch;
    let (n_llm_h, n_llm_w, best_h, best_w) = plan_image_grid(orig_w, orig_h, vis);
    let (n_vit_h, n_vit_w) = (best_h / p, best_w / p);
    let padded = resize_and_pad_rgb(&rgb, orig_w, orig_h, best_w, best_h, vis.max_wh_ratio);

    let lut = byte_to_bf16_lut();
    let patch_dim = 3 * p * p;
    let mut patches = vec![0u16; n_vit_h * n_vit_w * patch_dim];
    for vh in 0..n_vit_h {
        for vw in 0..n_vit_w {
            let dst = &mut patches[(vh * n_vit_w + vw) * patch_dim..(vh * n_vit_w + vw + 1) * patch_dim];
            for c in 0..3 {
                for py in 0..p {
                    let row_off = ((vh * p + py) * best_w + vw * p) * 3 + c;
                    let col_off = (c * p + py) * p;
                    for px in 0..p {
                        dst[col_off + px] = lut[padded[row_off + px * 3] as usize];
                    }
                }
            }
        }
    }

    let mut hasher = Sha256::new();
    hasher.update(b"dsv41-vis1-span\0");
    for v in [orig_w as u64, orig_h as u64, n_vit_h as u64, n_vit_w as u64, n_llm_h as u64, n_llm_w as u64] {
        hasher.update(v.to_le_bytes());
    }
    hasher.update(&rgb);
    let digest: [u8; 32] = hasher.finalize().into();

    let types = image_token_types(n_llm_h, n_llm_w);
    Ok((
        PreparedImage { start: 0, end: types.len(), n_vit_h, n_vit_w, n_llm_h, n_llm_w, patches, types, digest },
        (orig_w, orig_h),
    ))
}

/// Expand all `image_token_id` placeholders in `raw_prompt_ids` using `records` (`image_processor.prepare_vl_inputs`).
pub fn prepare_vl_inputs(
    raw_prompt_ids: &[u32],
    records: &[Json],
    image_token_id: u32,
    vis: &Vision,
    max_images: usize,
) -> Result<VlPrepared, ApiError> {
    let num_placeholders = raw_prompt_ids.iter().filter(|&&t| t == image_token_id).count();
    if num_placeholders != records.len() {
        return Err(ApiError::bad(format!(
            "found {num_placeholders} image placeholder tokens in prompt but got {} image records",
            records.len()
        )));
    }
    if records.is_empty() {
        return Ok(VlPrepared {
            prompt_ids: raw_prompt_ids.to_vec(),
            token_types: Vec::new(),
            images: Vec::new(),
            spans: Vec::new(),
        });
    }
    if max_images > 0 && records.len() > max_images {
        return Err(bad_img(format!(
            "request contains {} images, exceeding --dsv41-max-images {max_images}",
            records.len()
        )));
    }

    let mut decoded = Vec::with_capacity(records.len());
    for rec in records {
        let bytes = load_image_bytes(rec)?;
        let (img, _) = decode_and_patchify(&bytes, vis)?;
        decoded.push(img);
    }

    let extra: usize = decoded.iter().map(|im| im.types.len().saturating_sub(1)).sum();
    let total_len = raw_prompt_ids.len() + extra;
    let mut prompt_ids = Vec::with_capacity(total_len);
    let mut token_types = Vec::with_capacity(total_len);
    let mut spans = Vec::with_capacity(decoded.len());
    let mut img_iter = decoded.into_iter();
    let mut images = Vec::with_capacity(records.len());

    for &tok in raw_prompt_ids {
        if tok != image_token_id {
            prompt_ids.push(tok);
            token_types.push(TOK_TEXT);
            continue;
        }
        let mut img = img_iter.next().expect("checked placeholder count");
        let start = prompt_ids.len();
        let span_len = img.types.len();
        let end = start + span_len;
        img.start = start;
        img.end = end;
        prompt_ids.extend(std::iter::repeat_n(image_token_id, span_len));
        token_types.extend_from_slice(&img.types);
        spans.push(ImageSpanKey { start, end, digest: img.digest });
        images.push(img);
    }

    Ok(VlPrepared { prompt_ids, token_types, images, spans })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsv4_chat::parse_json;

    fn test_vis() -> Vision {
        Vision {
            layers: 32,
            hidden: 1024,
            heads: 16,
            inter: 2816,
            patch: 14,
            rope_theta: 10000.0,
            downsample: 3,
            max_image_tokens: 1024,
            min_pixels: 295_936,
            max_wh_ratio: None,
        }
    }

    #[test]
    fn grid_plans_match_python_reference() {
        let mut v = test_vis();
        assert_eq!(plan_image_grid(300, 200, &v), (11, 16, 448, 672));
        assert_eq!(num_image_tokens(11, 16), 189);
        assert_eq!(plan_image_grid(1200, 800, &v), (20, 29, 812, 1204));
        assert_eq!(num_image_tokens(20, 29), 602);
        assert_eq!(plan_image_grid(1800, 1200, &v), (25, 38, 1050, 1568));
        assert_eq!(num_image_tokens(25, 38), 977);
        assert_eq!(plan_image_grid(1600, 200, &v), (5, 39, 210, 1610));
        assert_eq!(num_image_tokens(5, 39), 202);
        // Extreme tall and wide aspect ratios hit single-column / single-row branches cleanly
        let (th, tw, _, _) = plan_image_grid(50, 4000, &v);
        assert!(num_image_tokens(th, tw) <= 1024 && tw >= 1);
        let (wh, ww, _, _) = plan_image_grid(4000, 50, &v);
        assert!(num_image_tokens(wh, ww) <= 1024 && wh >= 1);
        // max_wh_ratio cap
        v.max_wh_ratio = Some(4.0);
        let (mh, mw, _, _) = plan_image_grid(1600, 200, &v);
        assert!(num_image_tokens(mh, mw) <= 1024);
    }

    #[test]
    fn url_and_path_inputs_are_rejected_with_invalid_image_url() {
        for bad in [
            r#"{"url": "http://example.com/a.png"}"#,
            r#"{"url": "https://example.com/a.png"}"#,
            r#"{"url": "/etc/passwd"}"#,
            r#"{"url": "relative/image.png"}"#,
            r#"{"url": "data:image/png,rawbytes"}"#,
            r#"{"source": {"type": "url", "url": "https://evil.test/x.png"}}"#,
        ] {
            let j = parse_json(bad).unwrap();
            let err = load_image_bytes(&j).unwrap_err();
            assert_eq!(err.status, 400);
            assert_eq!(err.code, Some("invalid_image_url"), "{bad} -> {err:?}");
        }
    }

    #[test]
    fn inline_base64_expands_spans_and_separates_digests() {
        let mut img_a = image::RgbImage::from_pixel(64, 48, image::Rgb([220, 30, 40]));
        img_a.put_pixel(10, 10, image::Rgb([10, 240, 50]));
        let mut buf_a = Vec::new();
        image::DynamicImage::ImageRgb8(img_a).write_to(&mut Cursor::new(&mut buf_a), image::ImageFormat::Png).unwrap();

        let img_b = image::RgbImage::from_pixel(64, 48, image::Rgb([30, 40, 220]));
        let mut buf_b = Vec::new();
        image::DynamicImage::ImageRgb8(img_b).write_to(&mut Cursor::new(&mut buf_b), image::ImageFormat::Png).unwrap();

        let b64_a = base64::engine::general_purpose::STANDARD.encode(&buf_a);
        let b64_b = base64::engine::general_purpose::STANDARD.encode(&buf_b);
        let rec_a = parse_json(&format!(r#"{{"url": "data:image/png;base64,{b64_a}"}}"#)).unwrap();
        let rec_b = parse_json(&format!(r#"{{"source": {{"type": "base64", "data": "{b64_b}"}}}}"#)).unwrap();

        let vis = test_vis();
        let raw = vec![0u32, 10, 129264, 20, 129264, 30];
        let out = prepare_vl_inputs(&raw, &[rec_a.clone(), rec_b], 129264, &vis, 16).unwrap();
        assert_eq!(out.spans.len(), 2);
        assert_ne!(out.spans[0].digest, out.spans[1].digest, "same-size distinct images must have distinct digests");
        let s0 = &out.spans[0];
        assert_eq!(s0.start, 2);
        assert_eq!(out.token_types[0], TOK_TEXT);
        assert_eq!(out.token_types[1], TOK_TEXT);
        assert_eq!(out.token_types[s0.start], TOK_IMAGE_START);
        assert_eq!(out.token_types[s0.end - 1], TOK_IMAGE_END);
        assert!(out.prompt_ids[s0.start..s0.end].iter().all(|&t| t == 129264));

        // Max images limit enforced
        let lim_err = prepare_vl_inputs(&[129264, 129264], &[rec_a.clone(), rec_a], 129264, &vis, 1).unwrap_err();
        assert_eq!(lim_err.status, 400);
        assert_eq!(lim_err.code, Some("invalid_image_url"));
    }

    #[test]
    fn jpeg_webp_gif_bmp_bare_b64_and_data_base64_all_decode() {
        let vis = test_vis();
        let mut img = image::RgbImage::from_pixel(32, 24, image::Rgb([40, 120, 210]));
        img.put_pixel(5, 5, image::Rgb([250, 10, 20]));
        let dyn_img = image::DynamicImage::ImageRgb8(img);

        for fmt in [image::ImageFormat::Jpeg, image::ImageFormat::WebP, image::ImageFormat::Gif, image::ImageFormat::Png] {
            let mut buf = Vec::new();
            dyn_img.write_to(&mut Cursor::new(&mut buf), fmt).unwrap();
            // Bare base64 inside `url` (e.g. JPEG `/9j/...` without `data:` prefix)
            let b64 = base64::engine::general_purpose::STANDARD.encode(&buf);
            let rec_bare = parse_json(&format!(r#"{{"url": "{b64}"}}"#)).unwrap();
            let loaded = load_image_bytes(&rec_bare).unwrap();
            let (prep, dims) = decode_and_patchify(&loaded, &vis).unwrap();
            assert_eq!(dims, (32, 24), "{fmt:?}");
            assert!(!prep.patches.is_empty());

            // URL-safe no-pad base64 + `data:base64,...` (no media-type semicolon)
            let b64_url = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&buf);
            let rec_d = parse_json(&format!(r#"{{"url": "data:base64,{b64_url}"}}"#)).unwrap();
            let loaded_d = load_image_bytes(&rec_d).unwrap();
            assert_eq!(loaded_d, buf);
        }

        // 24-bit uncompressed BMP (`BM`)
        let (w, h) = (16usize, 12usize);
        let row_stride = (w * 3).div_ceil(4) * 4;
        let mut bmp = vec![0u8; 54 + row_stride * h];
        bmp[0..2].copy_from_slice(b"BM");
        let total_sz = bmp.len() as u32;
        bmp[2..6].copy_from_slice(&total_sz.to_le_bytes());
        bmp[10..14].copy_from_slice(&54u32.to_le_bytes());
        bmp[14..18].copy_from_slice(&40u32.to_le_bytes());
        bmp[18..22].copy_from_slice(&(w as i32).to_le_bytes());
        bmp[22..26].copy_from_slice(&(h as i32).to_le_bytes());
        bmp[26..28].copy_from_slice(&1u16.to_le_bytes());
        bmp[28..30].copy_from_slice(&24u16.to_le_bytes());
        let (prep_bmp, dims_bmp) = decode_and_patchify(&bmp, &vis).unwrap();
        assert_eq!(dims_bmp, (w, h));
        assert!(!prep_bmp.patches.is_empty());

        // Unsupported containers report explicit format names
        let avif_hdr = b"\x00\x00\x00\x1cftypavif\x00\x00\x00\x00avifmif1";
        let err_avif = decode_and_patchify(avif_hdr, &vis).unwrap_err();
        assert!(err_avif.message.contains("AVIF"), "{err_avif:?}");
    }
}
