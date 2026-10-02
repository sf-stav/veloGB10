//! Vision preprocessing for the Qwen3.5-27B visual tower.
//!
//! Reproduces the HF `Qwen2VLImageProcessor` (torchvision backend) preprocessing that the
//! model's own `preprocessor_config.json`/`config.json` select. Every parameter is pinned in
//! `PLAN/W2_PREPROC_SPEC.md`; this module is the Rust implementation of sections 2 and 3.
//!
//! Stages (given a decoded RGB `image` of `H x W` uint8):
//!   1. smart_resize (area-based; factor = patch*merge = 32, min_pixels = 65536, max = 16777216)
//!   2. antialiased bicubic resize to `(hb, wb)` (mirrors torch `interpolate(bicubic, antialias)`)
//!   3. normalize `(x - 127.5)/127.5`
//!   4. flatten to `[gh*gw, C*t*p*p = 1536]` per image, in the exact order the Conv3d
//!      `patch_embed` consumes (per-patch `[c, t, ph, pw]`, cell-blocked 2x2 merge).
//!
//! Output dtype is f32; the caller feeds the rows to the Conv3d patch-embed.

/// Patch config for the vision tower (from the model's `vision_config` / `preprocessor_config`).
#[derive(Clone, Copy, Debug)]
pub struct VisionPreprocConfig {
    pub patch_size: usize,            // 16
    pub merge_size: usize,            // 2
    pub temporal_patch_size: usize,   // 2
    pub in_channels: usize,           // 3
    pub min_pixels: usize,            // size.shortest_edge = 65536
    pub max_pixels: usize,            // size.longest_edge = 16777216
    /// D-VIS-RESIZE (owner 2026-09-30): longer side capped at this many pixels (aspect kept) BEFORE
    /// smart_resize's grid snap; 0 = no cap. Server flag `--image-max-edge` (default 1024).
    pub max_edge: usize,
}

impl VisionPreprocConfig {
    pub fn factor(&self) -> usize {
        self.patch_size * self.merge_size // 32
    }
    pub fn hidden_patch_rows(&self, gh: usize, gw: usize) -> usize {
        gh * gw * self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }
}

/// The default config taken from the model's own preprocessor_config.json.
pub const QWEN27B_PREPROC: VisionPreprocConfig = VisionPreprocConfig {
    patch_size: 16,
    merge_size: 2,
    temporal_patch_size: 2,
    in_channels: 3,
    min_pixels: 65536,
    max_pixels: 16777216,
    max_edge: 1024,
};

/// Area-based smart_resize, mirroring `image_processing_qwen2_vl.py::smart_resize`.
///
/// Returns `(hb, wb)`, each a multiple of `factor`, with the aspect ratio preserved as closely as
/// possible and `min_pixels <= hb*wb <= max_pixels`.
pub fn smart_resize(h: usize, w: usize, cfg: &VisionPreprocConfig) -> (usize, usize) {
    let factor = cfg.factor() as f64;
    let (hf, wf) = (h as f64, w as f64);
    let min_px = cfg.min_pixels as f64;
    let max_px = cfg.max_pixels as f64;

    // Python `round()` = banker's rounding (ties to even); replicate via round_ties_even.
    let h_bar = (round_ties_even(hf / factor) * factor) as i64;
    let w_bar = (round_ties_even(wf / factor) * factor) as i64;
    let mut hb = h_bar as f64;
    let mut wb = w_bar as f64;

    if hb * wb > max_px {
        let beta = (hf * wf / max_px).sqrt();
        hb = factor.max((hf / beta / factor).floor() * factor);
        wb = factor.max((wf / beta / factor).floor() * factor);
    } else if hb * wb < min_px {
        let beta = (min_px / (hf * wf)).sqrt();
        hb = (hf * beta / factor).ceil() * factor;
        wb = (wf * beta / factor).ceil() * factor;
    }
    (hb as usize, wb as usize)
}

/// D-VIS-RESIZE: the final `(hb, wb)` for an `h x w` input. When the longer side exceeds
/// `cfg.max_edge`, the image is first scaled so that side is exactly `max_edge` (the other side by
/// the same factor, rounded half-to-even), then `smart_resize` snaps both sides to the 32-px grid.
/// The pixels are resampled ONCE, straight to these dims (one bicubic pass instead of two).
pub fn capped_resize_dims(h: usize, w: usize, cfg: &VisionPreprocConfig) -> (usize, usize) {
    let long = h.max(w);
    if cfg.max_edge == 0 || long <= cfg.max_edge {
        return smart_resize(h, w, cfg);
    }
    let s = cfg.max_edge as f64 / long as f64;
    let scale = |v: usize| -> usize {
        if v == long { cfg.max_edge } else { (round_ties_even(v as f64 * s) as usize).max(1) }
    };
    smart_resize(scale(h), scale(w), cfg)
}

/// `round()` with ties-to-even (Python `round` semantics used by smart_resize).
pub fn round_ties_even(x: f64) -> f64 {
    let r = x.round();
    let frac = x - r;
    if frac.abs() == 0.5 {
        // tie: round to even
        if (r as i64) % 2 == 0 {
            r
        } else {
            r + x.signum()
        }
    } else {
        r
    }
}

/// Keys bicubic kernel with `a = -0.5`, matching torch's `BicubicFilterFunctor`.
#[inline]
fn bicubic(x: f64) -> f64 {
    let x = x.abs();
    let a = -0.5;
    if x < 1.0 {
        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * a
    } else {
        0.0
    }
}

/// 1-D antialiased bicubic resampler, mirroring ATen's `upsample_antialias` math.
fn resize_1d(inp: &[f32], out_len: usize, scale: f64) -> Vec<f32> {
    let support = 2.0;
    let n = inp.len();
    let mut out = vec![0.0f32; out_len];
    for i in 0..out_len {
        let center = scale * (i as f64 + 0.5);
        let xmin = ((center - support + 0.5).floor() as i64).max(0) as usize;
        let xmax = ((center + support + 0.5).floor() as i64).min(n as i64) as usize;
        let xsize = if xmax > xmin { xmax - xmin } else { 0 };
        let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
        let mut total = 0.0f64;
        let mut weights = vec![0.0f64; xsize];
        for j in 0..xsize {
            let w = bicubic((xmin as f64 + j as f64 + 0.5 - center) * invscale);
            weights[j] = w;
            total += w;
        }
        if total != 0.0 {
            for w in weights.iter_mut() {
                *w /= total;
            }
        }
        let mut acc = 0.0f64;
        for j in 0..xsize {
            acc += inp[xmin + j] as f64 * weights[j];
        }
        out[i] = acc as f32;
    }
    out
}

/// Separable 2-D antialiased bicubic resize of a `[C, H, W]` channel-first f32 image (values 0..255).
/// Returns `[C, out_h, out_w]` f32. Horizontal pass then vertical pass.
pub fn resize2d(img: &[f32], in_w: usize, in_h: usize, out_w: usize, out_h: usize) -> Vec<f32> {
    let c = 3usize;
    // horizontal pass
    let mut hp = vec![0.0f32; c * in_h * out_w];
    let sw = in_w as f64 / out_w as f64;
    for ch in 0..c {
        for y in 0..in_h {
            let row = &img[(ch * in_h + y) * in_w..(ch * in_h + y) * in_w + in_w];
            let r = resize_1d(row, out_w, sw);
            for x in 0..out_w {
                hp[(ch * in_h + y) * out_w + x] = r[x];
            }
        }
    }
    // vertical pass
    let mut out = vec![0.0f32; c * out_h * out_w];
    let sh = in_h as f64 / out_h as f64;
    for ch in 0..c {
        for x in 0..out_w {
            let mut col = vec![0.0f32; in_h];
            for y in 0..in_h {
                col[y] = hp[(ch * in_h + y) * out_w + x];
            }
            let r = resize_1d(&col, out_h, sh);
            for y in 0..out_h {
                out[(ch * out_h + y) * out_w + x] = r[y];
            }
        }
    }
    out
}

/// One image preprocessed to the `pixel_values` (flatten_patches) the vision model consumes.
pub struct PreprocessedImage {
    /// Flattened patch rows: `[gh*gw, C*t*p*p]`, f32, matching the HF processor's ordering.
    pub pixel_values: Vec<f32>,
    /// `(gh, gw)` patch grid.
    pub grid_h: usize,
    pub grid_w: usize,
}

/// Full pipeline: resize + normalize + flatten for one decoded RGB image.
pub fn preprocess_image(
    h: usize,
    w: usize,
    rgb: &[u8], // len == h*w*3
    cfg: &VisionPreprocConfig,
) -> PreprocessedImage {
    let (hb, wb) = capped_resize_dims(h, w, cfg);
    // build [C, H, W] f32 in 0..255
    let mut img = vec![0.0f32; 3 * h * w];
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                img[(c * h + y) * w + x] = rgb[(y * w + x) * 3 + c] as f32;
            }
        }
    }
    let mut resized = resize2d(&img, w, h, wb, hb);
    // normalize (fused rescale-into-mean: (x - 127.5)/127.5); resize on [0,255] then normalize.
    for v in resized.iter_mut() {
        *v = (*v - 127.5) / 127.5;
    }
    let gh = hb / cfg.patch_size;
    let gw = wb / cfg.patch_size;
    let rows = flatten(&resized, gh, gw, cfg);
    PreprocessedImage {
        pixel_values: rows,
        grid_h: gh,
        grid_w: gw,
    }
}

/// Flatten a normalized `[C, H, W]` (H=gh*patch, W=gw*patch) into `[gh*gw, C*t*p*p]` patch rows
/// in the exact order the Conv3d patch_embed consumes. Cell-blocked 2x2 merge; per-patch
/// within-token order is `[c, t, ph, pw]` with the temporal dim broadcast (silent replicate for
/// still images). Mirrors the HF `reshape/permute/expand/reshape` chain.
pub fn flatten(resized: &[f32], gh: usize, gw: usize, cfg: &VisionPreprocConfig) -> Vec<f32> {
    let c = cfg.in_channels;          // 3
    let p = cfg.patch_size;           // 16
    let t = cfg.temporal_patch_size;  // 2
    let m = cfg.merge_size;           // 2
    let h = gh * p;
    let w = gw * p;
    let mut rows = vec![0.0f32; gh * gw * c * t * p * p];
    let mut ri = 0usize;
    for gh_m in 0..gh / m {
        for gw_m in 0..gw / m {
            for mh in 0..m {
                for mw in 0..m {
                    for cch in 0..c {
                        for _ti in 0..t {
                            for ph in 0..p {
                                for pw in 0..p {
                                    let y = (gh_m * m + mh) * p + ph;
                                    let x = (gw_m * m + mw) * p + pw;
                                    rows[ri] = resized[(cch * h + y) * w + x];
                                    ri += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_grids() {
        // Verified against the HF processor on the three committed test images.
        assert_eq!(smart_resize(580, 800, &QWEN27B_PREPROC), (576, 800));
        assert_eq!(smart_resize(812, 587, &QWEN27B_PREPROC), (800, 576));
        assert_eq!(smart_resize(200, 300, &QWEN27B_PREPROC), (224, 320));
    }

    #[test]
    fn smart_resize_banker_rounding() {
        // 200/32 = 6.25 (tie? no), 300/32=9.375 -> round 9 -> 288. Then min-pixels floor.
        assert_eq!(smart_resize(200, 300, &QWEN27B_PREPROC), (224, 320));
    }
}
