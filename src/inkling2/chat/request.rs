//! One OpenAI chat body to what the template and the reply parser need: the messages, the offered tools and `tool_choice`, the
//! `reasoning_effort` (a name or a number), the rendered prompt and its ids, and where each image placeholder sits in them (the
//! vision stage expands each `<|unused_200054|>` into rows x cols copies, as the reference processor does).
//! No I/O: a pure function of the body, the server's defaults and the tokenizer.

use serde_json::Value;

use super::catalogue::Catalogue;
use super::effort::{self, Effort};
use super::parse::ParseOptions;
use super::special::IMAGE;
use crate::tokenizer::{ChatMessage, QwenTokenizer, ThinkingMode};

/// A client-visible error: HTTP status, OpenAI `type`, `code` and message.
#[derive(Debug, Clone)]
pub struct ApiError { pub status: u16, pub kind: &'static str, pub code: Option<&'static str>, pub message: String }

impl ApiError {
    pub fn bad(message: impl Into<String>) -> Self { Self { status: 400, kind: "invalid_request_error", code: None, message: message.into() } }
    pub fn code(mut self, c: &'static str) -> Self { self.code = Some(c); self }
    pub fn json(&self) -> String {
        serde_json::json!({"error": {"message": self.message, "type": self.kind, "param": null, "code": self.code}}).to_string()
    }
}

/// What the server decided at boot that a request can override.
#[derive(Debug, Clone, Default)]
pub struct Defaults {
    /// `--reasoning-effort`
    pub effort: Option<String>,
    pub max_seq_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolChoice { Auto, None, Required, Named }

/// A request ready for the engine and the output side.
pub struct Rendered {
    pub prompt_text: String,
    pub prompt_ids: Vec<u32>,
    /// Index in `prompt_ids` of every `<|unused_200054|>`, in order (one per image part, in message order).
    pub image_splices: Vec<usize>,
    pub n_images: usize,
    /// The `url` of every image part, in prompt order (the vision stage decodes them; `None` = a part without a url).
    pub image_urls: Vec<Option<String>>,
    /// Filled by the vision stage (`vision::serve::prepare`): the preprocessed images and their spans in the EXPANDED `prompt_ids`.
    pub vision: Option<std::sync::Arc<crate::inkling2::vision::serve::VisionReq>>,
    pub effort: Effort,
    pub catalogue: Catalogue,
    pub tools_offered: bool,
    pub tool_choice: ToolChoice,
    /// `tool_choice: "required"` or a named function cannot be forced by this build; the call may still come, or not: said, counted.
    pub choice_unenforced: bool,
    pub parse: ParseOptions,
    /// What the template received as extra kwargs (for the log line).
    pub kwargs_desc: String,
}

fn tool_choice(v: Option<&Value>) -> Result<(ToolChoice, Option<String>), ApiError> {
    match v {
        None | Some(Value::Null) => Ok((ToolChoice::Auto, None)),
        Some(Value::String(s)) => match s.as_str() {
            "auto" => Ok((ToolChoice::Auto, None)),
            "none" => Ok((ToolChoice::None, None)),
            "required" => Ok((ToolChoice::Required, None)),
            o => Err(ApiError::bad(format!("'tool_choice' {o:?} is not one of auto, none, required or a function"))),
        },
        Some(o) => match o.pointer("/function/name").and_then(|n| n.as_str()) {
            Some(n) => Ok((ToolChoice::Named, Some(n.to_string()))),
            None => Err(ApiError::bad("'tool_choice' must be a string or {\"type\":\"function\",\"function\":{\"name\":...}}")),
        },
    }
}

/// Build the prompt of a chat request. `body` is the parsed JSON.
pub fn render(body: &Value, d: &Defaults, tok: &QwenTokenizer) -> Result<Rendered, ApiError> {
    let msgs_json = body.get("messages").and_then(|m| m.as_array()).filter(|a| !a.is_empty())
        .ok_or_else(|| ApiError::bad("'messages' must be a non-empty list"))?;
    let messages: Vec<ChatMessage> = msgs_json.iter().enumerate().map(|(i, m)| serde_json::from_value::<ChatMessage>(m.clone())
        .map_err(|e| ApiError::bad(format!("messages[{i}]: {e}")))).collect::<Result<_, _>>()?;
    if let Some(m) = messages.iter().find(|m| !m.unsupported_parts.is_empty()) {
        return Err(ApiError::bad(format!("content part type {:?} is not served (text and images only)", m.unsupported_parts[0])).code("unsupported_content"));
    }
    let n_images: usize = messages.iter().map(|m| m.images.len()).sum();
    let image_urls: Vec<Option<String>> = messages.iter().flat_map(|m| m.images.iter().map(|i| i.url.clone())).collect();
    // tools
    let tools_in: Vec<Value> = match body.get("tools") { None | Some(Value::Null) => Vec::new(), Some(Value::Array(a)) => a.clone(),
        Some(_) => return Err(ApiError::bad("'tools' must be a list")) };
    let (choice, named) = tool_choice(body.get("tool_choice"))?;
    let offered = !tools_in.is_empty() && choice != ToolChoice::None;
    let prompt_tools: Option<&[Value]> = if offered { Some(&tools_in) } else { None };
    let mut catalogue = if offered { Catalogue::from_openai_tools(&tools_in) } else { Catalogue::default() };
    if let Some(n) = named.as_deref() { catalogue = catalogue.only(n); }
    // reasoning effort: a name or a NUMBER, through the template kwargs (the shared render path sends an effort only as a string)
    let kw = body.get("chat_template_kwargs").filter(|v| !v.is_null());
    if let Some(k) = kw { if !k.is_object() { return Err(ApiError::bad("'chat_template_kwargs' must be an object")); } }
    let eff = effort::resolve(kw, body.get("reasoning_effort"), d.effort.as_deref()).map_err(ApiError::bad)?;
    let mut kwargs = kw.cloned().unwrap_or_else(|| serde_json::json!({}));
    if let Some(v) = &eff.value { kwargs["reasoning_effort"] = v.clone(); }
    let kwargs_desc = kwargs.to_string();
    let prompt_text = tok.apply_chat_template(&messages, prompt_tools, None, Some(&kwargs), ThinkingMode::Auto)
        .map_err(|e| ApiError::bad(format!("{e:#}")))?;
    let prompt_ids = tok.encode(&prompt_text, false).map_err(|e| ApiError::bad(format!("tokenizer: {e:#}")))?;
    if prompt_ids.is_empty() { return Err(ApiError::bad("the prompt is empty")); }
    if d.max_seq_len > 0 && prompt_ids.len() + 1 > d.max_seq_len {
        return Err(ApiError::bad(format!("this model's context is {} tokens; the prompt has {} (no room to generate). Shorten the messages or start the server with a larger --max-seq-len",
                                         d.max_seq_len, prompt_ids.len())).code("context_length_exceeded"));
    }
    let image_splices: Vec<usize> = prompt_ids.iter().enumerate().filter(|(_, &t)| t == IMAGE).map(|(i, _)| i).collect();
    if image_splices.len() != n_images {
        return Err(ApiError::bad(format!("the prompt holds {} image placeholders for {} image parts (a literal image token in the text?)", image_splices.len(), n_images)));
    }
    let max_calls = match body.get("parallel_tool_calls") { Some(Value::Bool(false)) => Some(1), _ => None };
    Ok(Rendered {
        prompt_text, prompt_ids, image_splices, n_images, image_urls, vision: None, effort: eff, catalogue, tools_offered: offered, tool_choice: choice,
        choice_unenforced: offered && matches!(choice, ToolChoice::Required | ToolChoice::Named),
        parse: ParseOptions { separate_reasoning: true, max_calls }, kwargs_desc,
    })
}
