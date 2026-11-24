//! DSV4.1 vision support (leg VIS1 / M4): DeepSeek-V4.1-Flash ViT (32 layers, 1024 dim, 16 heads x 64,
//! 2D RoPE, bidirectional SDPA) + 3x3 unfold MLP aligner (9216 -> 5120 -> 5120, GELU), exact PIL-compatible
//! image preprocessing and placeholder expansion (`preproc`), Rank-0 tensor-core GPU tower (`tower`),
//! and `VisionMode` policy helpers.

pub mod preproc;
pub mod probe;
pub mod tower;

use crate::dsv41::census::classify::TensorClass as C;
use crate::dsv41::config::Dsv41Config;

/// `--dsv41-vision auto|on|off`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisionMode {
    Auto,
    On,
    Off,
}

impl VisionMode {
    pub fn from_registry() -> Self {
        match crate::opts::lookup("dsv41-vision").and_then(crate::opts::get).as_deref().map(str::trim) {
            Some("off" | "0" | "false") => VisionMode::Off,
            Some("on" | "1" | "true") => VisionMode::On,
            _ => VisionMode::Auto,
        }
    }
}

/// Whether vision weights should be loaded on Rank 0 and served for `cfg`.
pub fn vision_enabled(cfg: &Dsv41Config) -> bool {
    match VisionMode::from_registry() {
        VisionMode::Off => false,
        VisionMode::Auto | VisionMode::On => cfg.vision.as_ref().is_some_and(|v| v.layers > 0),
    }
}

/// `--dsv41-max-images` (default 16; 0 = unlimited).
pub fn max_images_from_registry() -> usize {
    crate::opts::lookup("dsv41-max-images")
        .and_then(|o| crate::opts::ranged(o).ok())
        .unwrap_or(16)
        .max(0) as usize
}

/// Is tensor class `c` one of the non-layer Rank-0 vision / aligner / delimiter classes?
pub fn is_vision_global_class(c: C) -> bool {
    matches!(
        c,
        C::VisPatchEmbed | C::VisNorm | C::VisQkv | C::VisWo | C::VisMlpW1 | C::VisMlpW2
        | C::AlignerW1 | C::AlignerW2 | C::ImageSep
    )
}
