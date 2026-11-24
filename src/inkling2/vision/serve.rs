//! The request side of image input and the GPU worker's tower cache.
//!
//! Request (HTTP thread, CPU): every image part's data URL is decoded and preprocessed ([`super::preproc`]); each
//! `<|unused_200054|>` of the rendered prompt is expanded into `rows x cols` copies, as the reference processor does
//! (`Inkling2Processor.replace_image_token`); the expanded prompt is what the model prefills and what `usage` counts.
//!
//! Worker (GPU thread): the tower runs once per distinct image content (the [`TowerCache`] keys on the preprocessed
//! pixels' hash, so a multi-turn chat that re-sends the same image re-uses its features); the features are spliced over
//! the placeholder rows right after `embed_norm` in every prefill chunk the span touches (`model::Work::img`).
//!
//! Prompt-cache keys: the placeholder ids of two different images are identical, so any prefix cache keyed on token ids
//! must mix in [`VisionReq::key`] (the images' content hashes in prompt order) up to each image's span. This build has no
//! Inkling-2 prefix cache (`--prefix-cache` is accepted and not wired); the tower cache keys on content only.

use super::preproc::{self, PreImage};
use super::IMAGE_TOKEN;
use crate::inkling2::chat::request::{ApiError, Rendered};
use crate::inkling2::serve::metrics as mt;
use cudarc::driver::DeviceSlice;
use std::sync::Arc;

/// The images of one request, after preprocessing and placeholder expansion.
pub struct VisionReq {
    pub images: Vec<Arc<PreImage>>,
    /// (first prompt row, rows) of each image's span in the expanded prompt
    pub spans: Vec<(usize, usize)>,
}

impl VisionReq {
    /// Content key of the request's images, in prompt order (for a prefix cache keyed on ids).
    pub fn key(&self) -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        for (im, sp) in self.images.iter().zip(&self.spans) {
            preproc::fnv(&mut h, &im.hash.to_le_bytes());
            preproc::fnv(&mut h, &(sp.0 as u64).to_le_bytes());
        }
        h
    }
}

/// Decode + preprocess the images of `r` and expand its placeholders. `max_tokens` = the per-image token cap,
/// `max_seq_len` the served context (the expanded prompt must leave room to generate).
pub fn prepare(r: &mut Rendered, max_tokens: usize, max_seq_len: usize) -> Result<(), ApiError> {
    if r.image_urls.is_empty() { return Ok(()); }
    let mut images = Vec::with_capacity(r.image_urls.len());
    for (i, u) in r.image_urls.iter().enumerate() {
        let url = u.as_deref().ok_or_else(|| ApiError::bad(format!("image part {i} has no url (send {{\"type\":\"image_url\",\"image_url\":{{\"url\":\"data:image/png;base64,...\"}}}})")).code("invalid_image"))?;
        let bytes = preproc::url_bytes(url).map_err(|e| { mt::inc(&mt::VIS_BAD); ApiError::bad(format!("image part {i}: {e:#}")).code("invalid_image") })?;
        let pre = preproc::preprocess(&bytes, 40, max_tokens).map_err(|e| { mt::inc(&mt::VIS_BAD); ApiError::bad(format!("image part {i}: {e:#}")).code("invalid_image") })?;
        if let Some((w, h)) = pre.resized {
            mt::inc(&mt::VIS_DOWNSCALED);
            crate::rprintln!("[inkling2] notice: image part {i}: {}x{} would take {} tokens, above --image-max-tokens {max_tokens}; downscaled to {w}x{h} ({} tokens)",
                pre.src.0, pre.src.1, preproc::grid_of(pre.src.0, pre.src.1, 40).tokens(), pre.grid.tokens());
        }
        images.push(Arc::new(pre));
    }
    let n_img: usize = images.iter().map(|p| p.grid.tokens()).sum();
    let mut ids = Vec::with_capacity(r.prompt_ids.len() + n_img);
    let mut spans = Vec::with_capacity(images.len());
    let mut k = 0;
    for &t in &r.prompt_ids {
        if t == IMAGE_TOKEN && k < images.len() {
            let n = images[k].grid.tokens();
            spans.push((ids.len(), n));
            ids.extend(std::iter::repeat(IMAGE_TOKEN).take(n));
            k += 1;
        } else { ids.push(t); }
    }
    if k != images.len() {
        return Err(ApiError::bad(format!("the prompt holds {k} image placeholders for {} image parts", images.len())));
    }
    if max_seq_len > 0 && ids.len() + 1 > max_seq_len {
        return Err(ApiError::bad(format!("this model's context is {max_seq_len} tokens; the prompt has {} with its images ({n_img} image tokens). Send smaller images, start the server with a lower --image-max-tokens, or a larger --max-seq-len",
                                         ids.len())).code("context_length_exceeded"));
    }
    r.image_splices = spans.iter().map(|s| s.0).collect();
    r.prompt_ids = ids;
    mt::add(&mt::VIS_IMAGE_TOKENS, n_img as u64);
    r.vision = Some(Arc::new(VisionReq { images, spans }));
    Ok(())
}

/// Device copies of recent towers' outputs, keyed on the image content (LRU by bytes; entries of the request being
/// prefilled are pinned, so the cache may exceed its budget for the length of one request).
pub struct TowerCache {
    cap_bytes: usize,
    used: usize,
    tick: u64,
    ent: Vec<(u64, u64, cudarc::driver::CudaSlice<f32>)>, // (hash, last use, features [tokens][2304])
}

impl TowerCache {
    pub fn new(cap_bytes: usize) -> Self { TowerCache { cap_bytes, used: 0, tick: 0, ent: Vec::new() } }
    pub fn get(&mut self, hash: u64) -> Option<u64> {
        self.tick += 1;
        let t = self.tick;
        self.ent.iter_mut().find(|e| e.0 == hash).map(|e| { e.1 = t; crate::inkling2::gpu::dp(&e.2) })
    }
    /// Insert (evicting least-recently-used entries not in `pinned`) and return the device address of the features.
    pub fn put(&mut self, hash: u64, buf: cudarc::driver::CudaSlice<f32>, pinned: &[u64]) -> u64 {
        let bytes = buf.len() * 4;
        while self.used + bytes > self.cap_bytes {
            let Some(i) = self.ent.iter().enumerate().filter(|(_, e)| !pinned.contains(&e.0)).min_by_key(|(_, e)| e.1).map(|(i, _)| i) else { break };
            let e = self.ent.swap_remove(i);
            self.used -= e.2.len() * 4;
        }
        self.tick += 1;
        self.used += bytes;
        let p = crate::inkling2::gpu::dp(&buf);
        self.ent.push((hash, self.tick, buf));
        p
    }
    pub fn bytes(&self) -> usize { self.used }
}
