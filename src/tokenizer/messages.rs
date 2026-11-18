//! OpenAI chat message types: `ToolCall` / `FunctionCall`, `ImageInput`, `ChatMessage` with its lenient content
//! deserialization and the template-JSON conversion. Map of the directory in `mod.rs`.

use super::*;

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct FunctionCall {
    pub name: String,
    /// OpenAI sends this as a JSON **string**, e.g. "{\"city\":\"Paris\"}".
    #[serde(default)]
    pub arguments: String,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ToolCall {
    #[serde(default)]
    pub id: String,
    #[serde(default = "default_tool_type", rename = "type")]
    pub kind: String,
    pub function: FunctionCall,
}

fn default_tool_type() -> String { "function".to_string() }

/// Deserialize an OpenAI chat-message `content`, which the spec allows to be either a plain
/// STRING or an ARRAY of content parts (`{"type":"text",...}`, `{"type":"image_url",...}`).
/// Agent clients (the OpenAI agent SDK and the Pi harness) send the array form, which a bare
/// `Option<String>` rejected with a confusing 422 `invalid type: sequence, expected a string`.
///
/// Resolution (documented in the hotfix commit `serve-content-parts`):
/// - string -> used verbatim (unchanged from before);
/// - null / absent -> `None` (agents send `"content": null` on the assistant `tool_calls` turn);
/// - array -> only `{"type":"text","text":...}` parts are kept. A single text part is used
///   VERBATIM; multiple are JOINED with "\n" (a code comment in the commit records that choice).
///   Any other part shape is handled below.
/// - non-text part (`image_url`, `input_image`, ...) -> a CLEAR, actionable 422 rather than a
///   serde type error. **This is the vision entry point**: when the image tower lands, this
///   branch becomes the dispatch into image preprocessing instead of rejecting the request.
fn deserialize_content<'de, D>(d: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    let v: serde_json::Value = serde_json::Value::deserialize(d)?;
    Ok(parse_content_text(&v))
}

/// Parse a `content` value (string|array|null) into the TEXT portion, preserving the hotfix's
/// behavior exactly: string verbatim; null -> None; array -> text parts (verbatim single, joined
/// "\n" multiple, lenient bare-string); non-text (image) parts are now CAPTURED, not rejected.
fn parse_content_text(v: &serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::Null => None,
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Array(parts) => {
            let mut texts: Vec<String> = Vec::new();
            for part in parts {
                match part {
                    serde_json::Value::String(s) => texts.push(s.clone()),
                    serde_json::Value::Object(o) => match o.get("type").and_then(|t| t.as_str()) {
                        Some("text") => {
                            if let Some(t) = o.get("text").and_then(|t| t.as_str()) {
                                texts.push(t.to_string());
                            }
                        }
                        // image_url / input_image / ... : captured (not rejected); no text.
                        Some(_) => {}
                        None => {
                            if let Some(t) = o.get("text").and_then(|t| t.as_str()) {
                                texts.push(t.to_string());
                            }
                        }
                    },
                    _ => {}
                }
            }
            if texts.is_empty() { None } else { Some(texts.join("\n")) }
        }
        _ => None,
    }
}

/// VIS-5: the content parts in their ORIGINAL order — Some(text) for a text part, None for an image —
/// so the template renders images where the client put them (not all images before all text).
fn parse_content_layout(v: &serde_json::Value) -> Vec<Option<String>> {
    let mut out = Vec::new();
    if let serde_json::Value::Array(parts) = v {
        for part in parts {
            match part {
                serde_json::Value::String(s) => out.push(Some(s.clone())),
                serde_json::Value::Object(o) => match o.get("type").and_then(|t| t.as_str()) {
                    Some("image_url") | Some("image") | Some("input_image") => out.push(None),
                    Some("text") | None => {
                        if let Some(t) = o.get("text").and_then(|t| t.as_str()) { out.push(Some(t.to_string())); }
                    }
                    Some(_) => {}
                },
                _ => {}
            }
        }
    }
    out
}

/// VID-0 (VIS-5): content part types this server does not take (video, audio, files, ...). A
/// request carrying one is refused with a 400 naming the type, never silently dropped.
fn parse_content_unsupported(v: &serde_json::Value) -> Vec<String> {
    let mut out = Vec::new();
    if let serde_json::Value::Array(parts) = v {
        for part in parts {
            if let serde_json::Value::Object(o) = part {
                if let Some(t) = o.get("type").and_then(|t| t.as_str()) {
                    if !matches!(t, "text" | "image_url" | "image" | "input_image") { out.push(t.to_string()); }
                }
            }
        }
    }
    out
}

/// Extract image parts from a `content` array into `Vec<ImageInput>`.
fn parse_content_images(v: &serde_json::Value) -> Vec<ImageInput> {
    let mut imgs = Vec::new();
    if let serde_json::Value::Array(parts) = v {
        for part in parts {
            if let serde_json::Value::Object(o) = part {
                let is_img = o.get("type").and_then(|t| t.as_str())
                    .map(|t| t == "image_url" || t == "image" || t == "input_image").unwrap_or(false);
                if is_img {
                    let url = o.get("image_url")
                        .and_then(|u| u.get("url").or(Some(u)))
                        .and_then(|u| u.as_str());
                    imgs.push(ImageInput { url: url.map(String::from) });
                }
            }
        }
    }
    imgs
}

/// An image content part captured from a vision request (`{"type":"image_url",...}`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct ImageInput {
    /// `image_url.url` — a `data:` URL (base64 image) or a remote URL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

/// One message of the conversation, in the shape agent harnesses actually send.
///
/// `content` MUST be optional: every OpenAI client sends `"content": null` on the assistant turn that
/// carries `tool_calls`. It used to be a bare `String`, so that request failed to deserialize and the
/// server answered **HTTP 422** the moment any harness tried to return a tool result. That single line
/// is most of why tool calling appeared broken.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ChatMessage {
    pub role: String,
    /// Spec allows a STRING or an ARRAY of content parts. `content: Option<String>` alone
    /// rejected the array form → 422. See `deserialize_content` for the resolution steps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Captured image parts (`image_url`) from a vision request. Empty for text-only traffic.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<ImageInput>,
    /// Assistant turn: the calls the model previously made.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// `role: "tool"` turn: which call this is the result of.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// VIS-5: the content parts in client order (Some(text) / None = image); empty = string content.
    #[serde(skip)]
    pub layout: Vec<Option<String>>,
    /// VID-0: content part types the server does not take (the handler answers 400).
    #[serde(skip)]
    pub unsupported_parts: Vec<String>,
}

impl<'de> serde::Deserialize<'de> for ChatMessage {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Raw {
            role: String,
            #[serde(default)]
            content: Option<serde_json::Value>,
            #[serde(default, skip_serializing_if = "Option::is_none")]
            tool_calls: Option<Vec<ToolCall>>,
            #[serde(default, skip_serializing_if = "Option::is_none")]
            tool_call_id: Option<String>,
            #[serde(default, skip_serializing_if = "Option::is_none")]
            name: Option<String>,
            #[serde(default, skip_serializing_if = "Option::is_none")]
            reasoning_content: Option<String>,
        }
        let raw = Raw::deserialize(d)?;
        let content_owned = raw.content.clone();
        let content = content_owned.as_ref().and_then(|v| parse_content_text(v));
        let images = content_owned.as_ref().map(parse_content_images).unwrap_or_default();
        let layout = content_owned.as_ref().map(parse_content_layout).unwrap_or_default();
        let unsupported_parts = content_owned.as_ref().map(parse_content_unsupported).unwrap_or_default();
        // OpenAI's Chat Completions spec replaced `system` with `developer` for
        // structured-output/agent contexts, and vLLM-class backends accept it by mapping it to
        // `system` at ingest. Forwarded verbatim it reaches the pack's chat template, which only
        // knows `system|user|assistant|tool`, and the template's own `else` arm turns that into
        // "Unexpected message role" → HTTP 500 for spec-compliant clients (issue #37). Normalize
        // HERE — one mapping at ingest, every downstream renderer (Jinja template and the
        // fallback ChatML) sees `system`, templates stay strict. The dsv4/NVIDIA-template path
        // works on raw JSON and already maps `developer` itself, so it is unaffected.
        let role = if raw.role == "developer" { "system".to_string() } else { raw.role };
        Ok(ChatMessage {
            role,
            content,
            images,
            layout,
            unsupported_parts,
            tool_calls: raw.tool_calls,
            tool_call_id: raw.tool_call_id,
            name: raw.name,
            reasoning_content: raw.reasoning_content,
        })
    }
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".into(), content: Some(content.into()), images: vec![], layout: vec![], unsupported_parts: vec![],
               tool_calls: None, tool_call_id: None, name: None, reasoning_content: None }
    }

    /// Render into the JSON shape the model's Jinja template expects.
    ///
    /// The one trap: the template iterates `tool_call.arguments | items`, i.e. it expects a MAPPING.
    /// OpenAI hands us `arguments` as a JSON **string**. Passing the string straight through makes the
    /// template blow up (or worse, silently emit nonsense), so parse it back into an object here. If it
    /// is not valid JSON we pass an empty object rather than failing the whole request.
    pub(super) fn to_template_json(&self, raw_tool_args: bool) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        m.insert("role".into(), serde_json::Value::String(self.role.clone()));
        let content = match &self.content {
            Some(c) => c.clone(),
            None => String::new(),
        };
        if !self.images.is_empty() {
            // Vision: render as a content ARRAY of parts so the model's Jinja template turns each
            // image into `<|vision_start|><|image_pad|><|vision_end|>` and keeps the text.
            // VIS-5: in the client's order (the layout); a message built without one (internal
            // constructors) keeps the old images-then-text shape.
            let mut parts = Vec::new();
            if self.layout.iter().filter(|p| p.is_none()).count() == self.images.len() {
                for p in &self.layout {
                    match p {
                        None => parts.push(serde_json::json!({"type": "image"})),
                        Some(t) => parts.push(serde_json::json!({"type": "text", "text": t})),
                    }
                }
            } else {
                for _ in &self.images {
                    parts.push(serde_json::json!({"type": "image"}));
                }
                if !content.is_empty() {
                    parts.push(serde_json::json!({"type": "text", "text": content}));
                }
            }
            m.insert("content".into(), serde_json::Value::Array(parts));
        } else {
            // Text-only: keep the string (or null) — byte-identical to the hotfix/text path.
            m.insert("content".into(), match &self.content {
                Some(c) => serde_json::Value::String(c.clone()),
                None => serde_json::Value::Null,
            });
        }
        if let Some(r) = &self.reasoning_content {
            m.insert("reasoning_content".into(), serde_json::Value::String(r.clone()));
        }
        if let Some(tcs) = &self.tool_calls {
            let arr: Vec<serde_json::Value> = tcs.iter().map(|tc| {
                // OpenAI hands us `arguments` as a JSON STRING. The reference stacks (vLLM's chat preprocessing,
                // dgpp) parse it into an object BEFORE rendering, so the template's `is mapping` branch expands
                // `<parameter=…>` blocks — the format its system prompt asks for and the format the model itself
                // generated (GitHub issue #6: passing the raw string rendered `{"command": …}` inside
                // `<function=…>`, a mixed XML/JSON history the model copied in long agent sessions and the
                // tool-call parser did not match). So: a JSON object is always parsed. Only when a template has a
                // string branch (`raw_tool_args`) AND the arguments are not a JSON object (unparseable, array,
                // scalar) is the raw string kept, because that is all the branch can show; a template without
                // the string branch iterates `arguments | items`, so it gets `{}` there.
                let parsed = serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
                    .ok().filter(|v| v.is_object());
                let args: serde_json::Value = match parsed {
                    Some(obj) => obj,
                    None if raw_tool_args => serde_json::Value::String(tc.function.arguments.clone()),
                    None => serde_json::json!({}),
                };
                serde_json::json!({
                    "id": tc.id,
                    "type": tc.kind,
                    "function": { "name": tc.function.name, "arguments": args },
                })
            }).collect();
            m.insert("tool_calls".into(), serde_json::Value::Array(arr));
        }
        if let Some(id) = &self.tool_call_id {
            m.insert("tool_call_id".into(), serde_json::Value::String(id.clone()));
        }
        if let Some(n) = &self.name {
            m.insert("name".into(), serde_json::Value::String(n.clone()));
        }
        serde_json::Value::Object(m)
    }
}
