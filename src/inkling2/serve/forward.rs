//! The seam between the serve shell and the model. The integration leg implements `Forward` on the GPU model (prefill with the
//! prompt's image splice points, decode steps, the sampler); this leg ships `NotBuilt`, which answers every request with a clear
//! error, so the whole HTTP, template, parser and metrics surface is exercised without a model.

use anyhow::Result;

use super::super::chat::request::Rendered;

/// A generation request as the forward sees it.
pub struct Job<'a> {
    pub prompt: &'a [u32],
    /// Where each `<|unused_200054|>` sits in `prompt` (the vision stage expands each into rows x cols embeddings).
    pub image_splices: &'a [usize],
    /// The preprocessed images and their spans in `prompt` (None: text only, or a forward without vision).
    pub vision: Option<std::sync::Arc<crate::inkling2::vision::serve::VisionReq>>,
    pub max_new: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub seed: Option<u64>,
}

impl<'a> Job<'a> {
    pub fn from_rendered(r: &'a Rendered, max_new: usize, temperature: f32, top_p: f32, top_k: usize, seed: Option<u64>) -> Self {
        Job { prompt: &r.prompt_ids, image_splices: &r.image_splices, vision: r.vision.clone(), max_new, temperature, top_p, top_k, seed }
    }
}

/// How a generation ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum End { Stop, Length }

/// An error the client should see as such: `not_built` is the skeleton's answer.
#[derive(Debug)]
pub struct ForwardError { pub code: &'static str, pub message: String }
impl std::fmt::Display for ForwardError { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(&self.message) } }
impl std::error::Error for ForwardError {}

pub trait Forward: Send + Sync {
    fn name(&self) -> &'static str;
    /// True when this forward runs the vision tower (the request side then expands each image placeholder).
    fn vision(&self) -> bool { false }
    /// Run `job`, calling `emit` with every committed token id (EOS included) in order; `emit` returns false to stop (client gone,
    /// stop string). The reply is the sequence of ids; the HTTP side parses it.
    fn generate(&self, job: &Job<'_>, emit: &mut dyn FnMut(u32) -> bool) -> Result<End>;
}

/// The skeleton's forward: no model behind it.
pub struct NotBuilt;
impl Forward for NotBuilt {
    fn name(&self) -> &'static str { "not-built" }
    fn generate(&self, _job: &Job<'_>, _emit: &mut dyn FnMut(u32) -> bool) -> Result<End> {
        Err(ForwardError { code: "model_not_built", message:
            "the Inkling-2 forward is not built yet: this server answers the chat surface (template, reasoning and tool-call parsing, /metrics) but has no model behind it. \
             The integration leg plugs the model in (src/inkling2/serve/forward.rs)".into() }.into())
    }
}
