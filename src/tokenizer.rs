use tokenizers::Tokenizer;
use anyhow::Result;
use std::path::Path;

pub struct QwenTokenizer {
    tokenizer: Tokenizer,
    /// Directory the tokenizer was loaded from — where generation_config.json / tokenizer_config.json
    /// live. Stop tokens are read from the model's own files, never hardcoded.
    model_dir: Option<std::path::PathBuf>,
    /// Rendered from the model's own `chat_template.jinja` (or the `chat_template`
    /// field of `tokenizer_config.json`). `None` only when no template file is found
    /// next to the tokenizer, in which case the legacy hand-rolled template is used.
    chat_env: Option<minijinja::Environment<'static>>,
}

impl QwenTokenizer {
    pub fn from_file(path: &str) -> Result<Self> {
        let tokenizer = load_tokenizer(path)?;
        let chat_env = load_chat_env(path);
        let model_dir = Path::new(path).parent().map(|p| p.to_path_buf());
        Ok(Self { tokenizer, chat_env, model_dir })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self.tokenizer.encode(text, add_special_tokens)
            .map_err(|e| anyhow::anyhow!("Encoding failed: {}", e))?;
        Ok(encoding.get_ids().iter().map(|&x| x as u32).collect())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let ids_u32: Vec<u32> = ids.iter().copied().collect();
        self.tokenizer.decode(&ids_u32, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))
    }

    /// An incremental byte-level stream decoder for THIS tokenizer (all qwen byte-level BPE
    /// models). See `StreamByteDecoder` — the per-token `decode(&[t])` path it replaces
    /// mangles every multi-byte char split across tokens (all emoji) into "�".
    pub fn stream_decoder(&self) -> StreamByteDecoder {
        let path = self.model_dir.as_deref()
            .map(|d| d.join("tokenizer.json").to_string_lossy().into_owned())
            .unwrap_or_default();
        StreamByteDecoder::new(&self.tokenizer, &path)
    }

    pub fn eos_token_id(&self) -> u32 {
        self.tokenizer.get_vocab(true).get("<|endoftext|>").copied().unwrap_or(151643) as u32
    }

    /// Vocab size INCLUDING added tokens — the exclusive upper bound of a valid token id.
    /// Used by the /v1/tokenize endpoint to reject id-list prompts that carry garbage ids.
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    /// The think-block markers for this model's chat format, and whether a fresh generation starts
    /// INSIDE a think block. Qwen templates prime `<think>\n` (starts in reasoning, closes with
    /// `</think>`). hy_v3's no_think prompt renders `<think:opensource></think:opensource>` INTO the
    /// prompt (the empty block is already closed), so generation starts as CONTENT; its markers are
    /// the `:opensource`-suffixed forms. Resolved from the vocab, never hardcoded per family.
    pub fn think_tags(&self) -> (&'static str, &'static str, bool) {
        if self.tokenizer.get_vocab(true).contains_key("</think:opensource>") {
            ("<think:opensource>", "</think:opensource>", false)
        } else {
            ("<think>", "</think>", true)
        }
    }

    /// EVERY token that ends an assistant turn — read from the MODEL'S OWN FILES, not hardcoded.
    ///
    /// THIS WAS A REAL, USER-VISIBLE BUG. Qwen3.5's `config.json` declares
    /// `eos_token_id = <|endoftext|>` (248044), but a CHAT turn actually ends with `<|im_end|>`
    /// (248046) — which is what `tokenizer_config.json` names as the eos_token, and what the model
    /// emits. We stopped only on `<|endoftext|>`, so the assistant sailed straight past the end of its
    /// own turn and kept generating: it invented the next `user` message, then `assistant`, then a
    /// fresh `<think>` block, until it happened to emit `<|endoftext|>` or hit max_tokens.
    ///
    /// Symptoms in a real agent session: a fabricated conversation leaking into the UI after the
    /// answer; 238 tokens spent on a one-line file write; 1500–2500-token replies to trivial prompts;
    /// and — worst — the model wandering far enough to emit a SECOND, conflicting tool call.
    ///
    /// Sources, in the order every serving stack consults them (HF's `generation_config.eos_token_id`
    /// is a LIST for exactly this reason — a turn can end more than one way):
    ///   1. `generation_config.json` → `eos_token_id`  (int OR list — the canonical answer)
    ///   2. `tokenizer_config.json`  → `eos_token`     (a NAME; resolve it against the vocab)
    ///   3. `config.json`            → `eos_token_id`  (what we were using, and it is not enough)
    ///
    /// Nothing here is Qwen-specific: it is whatever the model ships. The name fallback at the end
    /// only fires if the model declares nothing at all.
    pub fn stop_token_ids(&self, config_eos: u32) -> Vec<u32> {
        let vocab = self.tokenizer.get_vocab(true);
        let mut ids: Vec<u32> = vec![config_eos];
        fn push(ids: &mut Vec<u32>, id: u32) { if !ids.contains(&id) { ids.push(id); } }

        let dir = self.model_dir.as_deref().unwrap_or(Path::new("."));

        // 1. generation_config.json — the canonical source, and it may be a list.
        if let Ok(raw) = std::fs::read_to_string(dir.join("generation_config.json")) {
            if let Ok(gc) = serde_json::from_str::<serde_json::Value>(&raw) {
                match &gc["eos_token_id"] {
                    serde_json::Value::Number(n) => { if let Some(i) = n.as_u64() { push(&mut ids, i as u32); } }
                    serde_json::Value::Array(a) => {
                        for v in a { if let Some(i) = v.as_u64() { push(&mut ids, i as u32); } }
                    }
                    _ => {}
                }
            }
        }

        // 2. tokenizer_config.json — names the CHAT terminator. This is the one we were missing.
        if let Ok(raw) = std::fs::read_to_string(dir.join("tokenizer_config.json")) {
            if let Ok(tc) = serde_json::from_str::<serde_json::Value>(&raw) {
                let name = tc["eos_token"].as_str()
                    .or_else(|| tc["eos_token"]["content"].as_str());
                if let Some(n) = name {
                    if let Some(&id) = vocab.get(n) { push(&mut ids, id as u32); }
                }
            }
        }

        // 2b. config.json fields beyond eos_token_id (already covered by the caller): hy_v3
        //     declares eod_token_id (120026) as a second advertised terminator.
        if let Ok(raw) = std::fs::read_to_string(dir.join("config.json")) {
            if let Ok(cj) = serde_json::from_str::<serde_json::Value>(&raw) {
                if let Some(i) = cj["eod_token_id"].as_u64() { push(&mut ids, i as u32); }
            }
        }

        // 2c. Turn terminators no config field advertises — hy_v3's `<｜hy_EOT｜>` (120008) ends a
        //     chat turn but lives only in the vocab. Name-resolved: fires only on models that have it.
        for n in ["<｜hy_EOT｜>"] {
            if let Some(&id) = vocab.get(n) { push(&mut ids, id); }
        }

        // 3. Last resort: the model declared nothing usable beyond config.json. Fall back to the
        //    conventional ChatML terminators by name so we at least do not run off the end of a turn.
        if ids.len() == 1 {
            for n in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>"] {
                if let Some(&id) = vocab.get(n) { push(&mut ids, id as u32); }
            }
        }
        ids
    }

    /// Render the conversation with the model's official chat template.
    ///
    /// This fixes two correctness issues the previous hand-rolled template had on
    /// Qwen3.5 thinking models:
    ///   1. Prior assistant turns have their `<think>…</think>` block stripped and
    ///      re-normalized (history turns must not carry raw think content).
    ///   2. The generation prompt ends with `<|im_start|>assistant\n<think>\n`,
    ///      priming the model to reason — instead of a bare `assistant\n`.
    /// `tools` is the OpenAI `tools` array, passed straight through to the template. The model's own
    /// template renders it into a `# Tools` system block; WITHOUT it the model is never told the tools
    /// exist and simply answers in prose. We used to drop it on the floor (the field did not exist on
    /// the request struct, so serde discarded it silently), which is why every agent harness failed.
    pub fn apply_chat_template(&self, messages: &[ChatMessage],
                               tools: Option<&[serde_json::Value]>,
                               reasoning_effort: Option<&str>) -> Result<String> {
        self.render_chat(messages, tools, true, reasoning_effort)
    }

    /// The same prompt WITHOUT the trailing `<|im_start|>assistant\n<think>\n`.
    ///
    /// This is the message boundary, and it is the longest prefix of this prompt that the NEXT turn is
    /// guaranteed to reproduce byte-for-byte — the template renders each past message independently, so
    /// everything up to here comes back unchanged, while the generation prompt does not (the next turn
    /// re-renders our assistant reply, and `<think>\n` vs `<think>\n\n</think>` are different tokens).
    ///
    /// Checkpointing the GDN state one token later than this made the prefix cache miss by exactly one
    /// token: `879 of 880 matched`. That single token cost a full re-prefill of the entire conversation.
    pub fn apply_chat_template_no_gen(&self, messages: &[ChatMessage],
                                      tools: Option<&[serde_json::Value]>,
                                      reasoning_effort: Option<&str>) -> Result<String> {
        self.render_chat(messages, tools, false, reasoning_effort)
    }

    fn render_chat(&self, messages: &[ChatMessage], tools: Option<&[serde_json::Value]>,
                   add_generation_prompt: bool, reasoning_effort: Option<&str>) -> Result<String> {
        if let Some(env) = &self.chat_env {
            let msgs: Vec<serde_json::Value> = messages.iter().map(|m| m.to_template_json()).collect();
            let mut ctx = serde_json::json!({
                "messages": msgs,
                "tools": tools,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": true,
            });
            // hy_v3 optional reasoning: the template's `reasoning_effort` knob ('no_think'|'low'|
            // 'high'; undefined => 'no_think'). Passed as a STRING only — a JSON null raises in the
            // template. Other families' templates ignore the variable.
            // S5F2 L3: `--thinking off` renders the template's enable_thinking=false branch (the
            // S3R2 A2 mirror — a normal chat turn WITHOUT the <think> block; the engine's
            // non-template raw-text path is a different regime).
            // S9F: "no_think" is the server's normalization of the OpenAI no-think conventions
            // (off/none/minimal) — the Qwen template RAISES on it as a reasoning_effort, so it
            // must take the enable_thinking=false branch ("off"'s sibling). hy_v3's own serving
            // path (dsv4_chat) handles its "no_think" knob separately; this branch only ever
            // sees the Qwen "chat" template.
            if let Some(e) = reasoning_effort {
                if e == "off" || e == "no_think" {
                    ctx["enable_thinking"] = serde_json::Value::Bool(false);
                } else {
                    ctx["reasoning_effort"] = serde_json::Value::String(e.to_string());
                }
            }
            let rendered = env.get_template("chat")
                .map_err(|e| anyhow::anyhow!("minijinja get_template: {}", e))?
                .render(&ctx)
                .map_err(|e| anyhow::anyhow!("minijinja render chat template: {}", e))?;
            return Ok(rendered);
        }
        // Legacy fallback: only hit when no chat_template.jinja sits next to the tokenizer.
        // It cannot render tools -- say so rather than silently producing a tool-less prompt, which is
        // exactly the failure mode that made tool calling look broken in the first place.
        if tools.map_or(false, |t| !t.is_empty()) {
            anyhow::bail!("this model has no chat_template.jinja, so tool definitions cannot be \
                           rendered; tool calling requires the model's own template");
        }
        let mut result = String::new();
        for msg in messages {
            let c = msg.content.as_deref().unwrap_or("");
            match msg.role.as_str() {
                "system" => result.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", c)),
                "user" => result.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", c)),
                "assistant" => result.push_str(&format!("<|im_start|>assistant\n{}<|im_end|>\n", c)),
                _ => {}
            }
        }
        result.push_str("<|im_start|>assistant\n");
        Ok(result)
    }
}

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
        Ok(ChatMessage {
            role: raw.role,
            content,
            images,
            tool_calls: raw.tool_calls,
            tool_call_id: raw.tool_call_id,
            name: raw.name,
            reasoning_content: raw.reasoning_content,
        })
    }
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".into(), content: Some(content.into()), images: vec![],
               tool_calls: None, tool_call_id: None, name: None, reasoning_content: None }
    }

    /// Render into the JSON shape the model's Jinja template expects.
    ///
    /// The one trap: the template iterates `tool_call.arguments | items`, i.e. it expects a MAPPING.
    /// OpenAI hands us `arguments` as a JSON **string**. Passing the string straight through makes the
    /// template blow up (or worse, silently emit nonsense), so parse it back into an object here. If it
    /// is not valid JSON we pass an empty object rather than failing the whole request.
    fn to_template_json(&self) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        m.insert("role".into(), serde_json::Value::String(self.role.clone()));
        let content = match &self.content {
            Some(c) => c.clone(),
            None => String::new(),
        };
        if !self.images.is_empty() {
            // Vision: render as a content ARRAY of parts so the model's Jinja template turns each
            // image into `<|vision_start|><|image_pad|><|vision_end|>` and keeps the text.
            let mut parts = Vec::new();
            for _ in &self.images {
                parts.push(serde_json::json!({"type": "image"}));
            }
            if !content.is_empty() {
                parts.push(serde_json::json!({"type": "text", "text": content}));
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
                let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                    .unwrap_or_else(|_| serde_json::json!({}));
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression: a prior assistant tool_call whose arguments are NON-STRING (numbers,
    /// booleans, objects) takes the template's `tojson(ensure_ascii=False)` path — the filter
    /// must tolerate the kwarg (was: "too many arguments" → HTTP 500 on the tool-result turn).
    #[test]
    fn render_tool_call_with_numeric_arguments() {
        let tok = match QwenTokenizer::from_file("/mnt/models/hy3-nvfp4/tokenizer.json") {
            Ok(t) => t,
            Err(e) => {
                eprintln!("skip: hy3 tokenizer not present ({})", e);
                return;
            }
        };
        assert!(tok.chat_env.is_some(), "hy3 chat template should have loaded");

        let messages = vec![
            ChatMessage::user("Read the first 60 lines of the file."),
            ChatMessage {
                role: "assistant".into(),
                content: None,
                images: vec![],
                tool_calls: Some(vec![ToolCall {
                    id: "call_0".into(),
                    kind: "function".into(),
                    function: FunctionCall {
                        name: "read".into(),
                        arguments: "{\"filePath\":\"/tmp/x\",\"limit\":60,\"offset\":1}".into(),
                    },
                }]),
                tool_call_id: None, name: None, reasoning_content: None,
            },
            ChatMessage {
                role: "tool".into(),
                content: Some("line one\nline two".into()),
                images: vec![], tool_calls: None, tool_call_id: Some("call_0".into()), name: Some("read".into()),
                reasoning_content: None,
            },
        ];
        let rendered = tok.apply_chat_template(&messages, None, None)
            .expect("render must not raise 'too many arguments' on non-string tool args");
        assert!(rendered.contains("read"), "tool name should appear in the prompt");
        assert!(rendered.contains("60"), "numeric argument should render unescaped");
    }

    /// HOTFIX `serve-content-parts`: the OpenAI spec allows `content` as a STRING or an ARRAY of
    /// content parts. A bare `Option<String>` rejected the array form with a confusing 422
    /// `invalid type: sequence, expected a string`. Headless unit test on the deserializer.
    #[test]
    fn content_accepts_string_array_and_null() {
        // (a) plain string (regression): unchanged.
        let m: ChatMessage = serde_json::from_str(r#"{"role":"user","content":"hello"}"#).unwrap();
        assert_eq!(m.content.as_deref(), Some("hello"));

        // (b) single-text-part array — the Pi payload shape. Used verbatim.
        let m: ChatMessage = serde_json::from_str(
            r#"{"role":"user","content":[{"type":"text","text":"hello world"}]}"#).unwrap();
        assert_eq!(m.content.as_deref(), Some("hello world"));

        // (c) multi-text-part array — joined with "\n" (documented join choice).
        let m: ChatMessage = serde_json::from_str(
            r#"{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}"#).unwrap();
        assert_eq!(m.content.as_deref(), Some("a\nb"));

        // (d) null content on an assistant tool_calls turn -> None (agents send this).
        let m: ChatMessage = serde_json::from_str(
            r#"{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]}"#).unwrap();
        assert_eq!(m.content, None);
        assert!(m.tool_calls.is_some(), "tool_calls must still deserialize");

        // (e) an array containing an image_url part -> CAPTURED into `images` (vision entry point);
        //     no 422, content is None (no text part).
        let m: ChatMessage = serde_json::from_str(
            r#"{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}"#).unwrap();
        assert_eq!(m.content, None);
        assert_eq!(m.images.len(), 1);
        assert_eq!(m.images[0].url.as_deref(), Some("data:image/png;base64,AAAA"));

        // (e2) mixed: an image part + a text part -> text captured in content, image in images.
        let m: ChatMessage = serde_json::from_str(
            r#"{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,BBBB"}},{"type":"text","text":"what is this?"}]}"#).unwrap();
        assert_eq!(m.content.as_deref(), Some("what is this?"));
        assert_eq!(m.images.len(), 1);
        assert_eq!(m.images[0].url.as_deref(), Some("data:image/png;base64,BBBB"));

        // (f) realistic agent conversation: system + user + assistant(tool_calls,null) + tool result.
        let msgs: Vec<ChatMessage> = serde_json::from_str(
            r#"[
                {"role":"system","content":"You are a helpful assistant."},
                {"role":"user","content":[{"type":"text","text":"What is 2+2?"}]},
                {"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"calc","arguments":"{\"expr\":\"2+2\"}"}}]},
                {"role":"tool","tool_call_id":"call_1","name":"calc","content":[{"type":"text","text":"4"}]}
            ]"#).unwrap();
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[1].content.as_deref(), Some("What is 2+2?"));
        assert_eq!(msgs[2].content, None);
        assert_eq!(msgs[3].content.as_deref(), Some("4"));
        assert_eq!(msgs[3].tool_call_id.as_deref(), Some("call_1"));
    }

    /// Regression: the SSE streaming path decodes one token at a time; a multi-byte UTF-8 char
    /// split across byte-level-BPE tokens (every emoji) must round-trip without "�" — the
    /// crate's per-token decode is String::from_utf8_lossy and mangles each fragment.
    #[test]
    fn stream_decoder_reassembles_emoji() {
        let tpath = match std::env::var("GB10_TEST_TOKENIZER") {
            Ok(t) if !t.is_empty() => t,
            _ => { eprintln!("skip: GB10_TEST_TOKENIZER not set"); return; }
        };
        let tok = match QwenTokenizer::from_file(&tpath) {
            Ok(t) => t,
            Err(e) => { eprintln!("skip: model tokenizer not present ({})", e); return; }
        };
        let text = "Hello \u{1F600} world! \u{1F680}\u{1F4A5}";   // 😀 🚀 💥
        let ids = tok.encode(text, false).expect("encode");
        let mut dec = tok.stream_decoder();
        let mut out = String::new();
        for &id in &ids {
            out.push_str(&dec.push(id));
        }
        out.push_str(&dec.finish());
        assert_eq!(out, text, "incremental decode must reassemble multi-byte chars");
        assert!(!out.contains('\u{FFFD}'), "no replacement chars allowed: {out:?}");
        // the whole-list decode (non-streaming path) must agree
        let whole = tok.decode(&ids, false).expect("decode");
        assert_eq!(whole, text, "whole-list decode should match too");
    }

    /// Smoke-test that the model's real Jinja template is loaded and renders a
    /// multi-turn conversation the way Qwen3.5 expects. Verifies the two fixes the
    /// legacy hand-rolled template got wrong:
    ///   1. The generation prompt ends with `<|im_start|>assistant\n<think>\n`.
    ///   2. A prior assistant turn that carried a `<think>…</think>` block has it
    ///      stripped in the rendered history (history turns are answer-only).
    #[test]
    fn render_4b_multiturn_thinking_template() {
        let tok = match QwenTokenizer::from_file("4b/tokenizer.json") {
            Ok(t) => t,
            Err(e) => {
                eprintln!("skip: 4b tokenizer not present ({})", e);
                return;
            }
        };
        assert!(tok.chat_env.is_some(), "chat template should have loaded from 4b/");

        let messages = vec![
            ChatMessage::user("Write a one-sentence sci-fi story."),
            ChatMessage {
                role: "assistant".into(),
                // Simulate what round-trips from a client: reasoning + answer.
                content: Some("<think>\nI should keep it short.\n</think>\n\nThe beacon awoke, and so did something else.\n".into()),
                images: vec![], tool_calls: None, tool_call_id: None, name: None, reasoning_content: None,
            },
            ChatMessage::user("Continue it for another 1000 words."),
        ];
        let rendered = tok.apply_chat_template(&messages, None, None).expect("render");

        eprintln!("===== RENDERED PROMPT START =====\n{}\n===== RENDERED PROMPT END =====", rendered);

        // Fix 1: generation prompt primes thinking.
        assert!(
            rendered.ends_with("<|im_start|>assistant\n<think>\n"),
            "generation prompt must end with `<|im_start|>assistant\\n<think>\\n`; got tail: {:?}",
            rendered.chars().rev().take(40).collect::<String>().chars().rev().collect::<String>()
        );

        // The history (first) assistant turn must appear, but its raw `<think>` body
        // should NOT be present — Qwen3.5 strips reasoning from history turns.
        assert!(rendered.contains("The beacon awoke"), "prior answer text should be present");
        assert!(
            !rendered.contains("I should keep it short"),
            "prior `<think>` body leaked into history; the template should strip it"
        );

        // Sanity: there should be exactly one `<think>` open tag — the one priming
        // the *current* generation — and no dangling `</think>` from history.
        assert_eq!(
            rendered.matches("<think>").count(),
            1,
            "expected exactly one priming <think> tag, got:\n{}",
            rendered
        );
    }
}

/// Load a tokenizer.json, transparently upgrading the pair-array merges form (`[["a","b"], ...]`)
/// to the space-joined string form (`["a b", ...]`) that tokenizers 0.19's BPE deserializer
/// expects (`merges: Vec<String>` in its BPE visitor). hy_v3's tokenizer.json ships the
/// pair-array form (HF's newer serialization default); qwen's ships strings. Detection is by
/// shape, not family — the fast path (no upgrade needed) costs nothing extra.
fn load_tokenizer(path: &str) -> Result<Tokenizer> {
    match Tokenizer::from_file(path) {
        Ok(t) => Ok(t),
        Err(e0) => {
            let raw = std::fs::read(path)?;
            let mut v: serde_json::Value = serde_json::from_slice(&raw)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e0} (file is not valid JSON either: {e})"))?;
            let ok = v.get_mut("model").and_then(|m| m.get_mut("merges")).and_then(|m| m.as_array_mut())
                .filter(|merges| merges.first().map_or(false, |m| m.is_array()))
                .map(|merges| {
                    for m in merges.iter_mut() {
                        let pair: Vec<String> = m.as_array().unwrap().iter()
                            .map(|x| x.as_str().unwrap_or("").to_string()).collect();
                        *m = serde_json::Value::String(pair.join(" "));
                    }
                });
            if ok.is_none() {
                return Err(anyhow::anyhow!("Failed to load tokenizer: {e0}"));
            }
            let bytes = serde_json::to_vec(&v)?;
            Tokenizer::from_bytes(bytes)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer (after pair-merge upgrade): {e}"))
        }
    }
}

/// Load the chat template that sits beside the tokenizer file and compile it into
/// a minijinja environment. Looks for `chat_template.jinja` first, then falls back
/// to the `chat_template` string inside `tokenizer_config.json`. Returns `None`
/// (so the legacy template is used) only if neither exists or the template fails
/// to compile.
fn load_chat_env(tokenizer_path: &str) -> Option<minijinja::Environment<'static>> {
    let dir = Path::new(tokenizer_path).parent()?;
    let jinja_path = dir.join("chat_template.jinja");
    let (source, origin) = if jinja_path.exists() {
        (std::fs::read_to_string(&jinja_path).ok()?, jinja_path.display().to_string())
    } else {
        let tc_path = dir.join("tokenizer_config.json");
        let raw = std::fs::read_to_string(&tc_path).ok()?;
        let tc: serde_json::Value = serde_json::from_str(&raw).ok()?;
        let s = tc.get("chat_template")?.as_str()?.to_string();
        (s, tc_path.display().to_string())
    };
    // The template source must outlive the environment. The server is a long-running
    // process that loads each model exactly once, so a one-time leak of a few KB is
    // acceptable and avoids per-request recompilation.
    let static_src: &'static str = Box::leak(source.into_boxed_str());
    let mut env = minijinja::Environment::new();
    register_pycompat(&mut env);
    match env.add_template("chat", static_src) {
        Ok(_) => {
            eprintln!("[tokenizer] loaded chat template from {}", origin);
            Some(env)
        }
        Err(e) => {
            eprintln!("[tokenizer] WARNING: chat template failed to compile ({}); using legacy manual template", e);
            None
        }
    }
}

/// Register an `unknown_method_callback` that bridges Jinja2/Python string methods —
/// which HuggingFace chat templates call freely but minijinja does not ship — so the
/// model's official template renders unchanged.
///
/// Handles the Python str methods this template family uses (`startswith`, `endswith`,
/// `lstrip`, `rstrip`, `strip`) directly, then falls back to minijinja's built-in
/// filters for anything else (`split`, `replace`, `trim`, `lower`, …). Python's
/// `lstrip`/`rstrip`/`strip` treat the argument as a *set of characters*, matching
/// `trim_*_matches` semantics below.
fn register_pycompat(env: &mut minijinja::Environment<'static>) {
    // HF chat templates are written against Jinja2 (Flask flavour). minijinja does not ship these, and
    // they are reachable ONLY on the tools path -- so the template compiled fine, rendered fine for
    // every ordinary chat, and blew up with a 500 the first time a tool definition was passed.
    //
    // `tojson`         - serialises the tool schema into the `# Tools` block, and any object/array
    //                    argument when a prior assistant tool_call is replayed.
    // `raise_exception`- the template calls it on malformed input; without it, a bad message would
    //                    fail with "unknown function" instead of the template's own diagnostic.
    env.add_filter("tojson", |v: minijinja::value::Value, kwargs: minijinja::value::Kwargs|
                   -> Result<String, minijinja::Error> {
        // The hy_v3 template calls `value | tojson(ensure_ascii=False)` when replaying non-string
        // tool_call arguments. serde_json already emits literal UTF-8 (== ensure_ascii=False
        // semantics — the only value any known template passes), so the kwarg is consumed and
        // ignored; WITHOUT this a kwarg'd call raises "too many arguments" and the whole request 500s.
        let _ = kwargs.get::<Option<bool>>("ensure_ascii").ok().flatten();
        serde_json::to_string(&v).map_err(|e| minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation, format!("tojson: {e}")))
    });
    env.add_function("raise_exception", |msg: String| -> Result<minijinja::value::Value, minijinja::Error> {
        Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
    });

    env.set_unknown_method_callback(|state, value, method, args| {
        let s = value.as_str();
        let res: Option<minijinja::value::Value> = match (method, s) {
            // `split` must return a *subscriptable* list — minijinja's built-in split
            // filter yields a one-shot iterable that the template then indexes with
            // `[0]` / `[-1]`. Build a real Vec so indexing works.
            ("split", Some(s)) => match arg_str(args, 0) {
                Ok(sep) => {
                    let parts: Vec<minijinja::value::Value> =
                        s.split(sep).map(minijinja::value::Value::from).collect();
                    Some(minijinja::value::Value::from(parts))
                }
                Err(()) => {
                    let parts: Vec<minijinja::value::Value> =
                        s.split_whitespace().map(minijinja::value::Value::from).collect();
                    Some(minijinja::value::Value::from(parts))
                }
            },
            ("startswith", Some(s)) => match arg_str(args, 0) {
                Ok(prefix) => Some(minijinja::value::Value::from(s.starts_with(prefix))),
                Err(()) => None,
            },
            ("endswith", Some(s)) => match arg_str(args, 0) {
                Ok(suffix) => Some(minijinja::value::Value::from(s.ends_with(suffix))),
                Err(()) => None,
            },
            ("lstrip", Some(s)) => Some(match arg_str(args, 0) {
                Ok(chars) => minijinja::value::Value::from(s.trim_start_matches(|c| chars.contains(c))),
                Err(()) => minijinja::value::Value::from(s.trim_start()),
            }),
            ("rstrip", Some(s)) => Some(match arg_str(args, 0) {
                Ok(chars) => minijinja::value::Value::from(s.trim_end_matches(|c| chars.contains(c))),
                Err(()) => minijinja::value::Value::from(s.trim_end()),
            }),
            ("strip", Some(s)) => Some(match arg_str(args, 0) {
                Ok(chars) => minijinja::value::Value::from(s.trim_matches(|c| chars.contains(c))),
                Err(()) => minijinja::value::Value::from(s.trim()),
            }),
            // Python str.format — NOT minijinja's printf-style `format` filter (%s). The hy_v3
            // template builds EVERY special token this way ('<｜hy_eos{}｜>'.format(HYTK)); letting
            // the call fall through to the built-in renders the string unchanged and plants
            // literal `{}` in the prompt. `{}` takes the next positional arg, `{N}` the Nth;
            // anything richer (format specs) is out of scope and delegates to the built-in.
            ("format", Some(s)) => {
                let render = |v: &minijinja::value::Value| match v.as_str() {
                    Some(x) => x.to_string(),
                    None => v.to_string(),
                };
                let mut out = String::with_capacity(s.len() + 8);
                let mut rest = s;
                let mut next = 0usize;
                let mut py_ok = true;
                while let Some(p) = rest.find('{') {
                    out.push_str(&rest[..p]);
                    match rest[p..].find('}') {
                        Some(q) => {
                            let spec = &rest[p + 1..p + q];
                            let idx = if spec.is_empty() {
                                let i = next; next += 1; i
                            } else if let Ok(i) = spec.parse::<usize>() {
                                next = i + 1; i
                            } else { py_ok = false; break; };   // a format spec — not Python-simple
                            match args.get(idx) {
                                Some(v) => out.push_str(&render(v)),
                                None => { py_ok = false; break; }
                            }
                            rest = &rest[p + q + 1..];
                        }
                        None => { py_ok = false; break; }
                    }
                }
                if py_ok {
                    out.push_str(rest);
                    Some(minijinja::value::Value::from(out))
                } else { None }
            },
            _ => None,
        };
        match res {
            Some(v) => Ok(v),
            None => {
                // Delegate to any built-in filter of the same name (e.g. `split`, `replace`).
                let mut all = vec![value.clone()];
                all.extend_from_slice(args);
                state.apply_filter(method, &all)
            }
        }
    });
}

fn arg_str(args: &[minijinja::value::Value], i: usize) -> Result<&str, ()> {
    args.get(i).and_then(|v| v.as_str()).ok_or(())
}

/// GPT-2-style `bytes_to_unicode` INVERSE table (mapped char -> raw byte), mirroring the
/// `tokenizers` crate's CHAR_BYTES. Qwen byte-level BPE vocabs store raw bytes as single
/// mapped chars (e.g. 'Ā' -> 0x00, 'ð' -> 0xF0). The crate's ByteLevel decoder maps them
/// back but decodes the byte stream with `String::from_utf8_lossy` — so a multi-byte UTF-8
/// char split across tokens (every emoji: 2-4 byte-mapped tokens) becomes one U+FFFD "�"
/// per fragment in any per-token decode. `StreamByteDecoder` uses this table to reassemble
/// the raw bytes and decode only complete sequences.
fn bytes_to_unicode_inverse() -> std::collections::HashMap<char, u8> {
    let mut inv = std::collections::HashMap::with_capacity(256);
    let mut bs: Vec<u32> = Vec::with_capacity(256);
    let mut cs: Vec<u32> = Vec::with_capacity(256);
    bs.extend(33..=126); bs.extend(161..=172); bs.extend(174..=255);
    cs.extend(33..=126); cs.extend(161..=172); cs.extend(174..=255);
    let mut n = 0u32;
    for b in 0u32..256 {
        if !bs.contains(&b) { bs.push(b); cs.push(256 + n); n += 1; }
    }
    for (b, c) in bs.iter().zip(cs.iter()) {
        if let Some(c) = char::from_u32(*c) { inv.insert(c, *b as u8); }
    }
    inv
}

/// Raw bytes a vocab piece decodes to under byte-level BPE: byte-mapped chars -> their byte
/// (any non-mapped char in the piece -> the piece's UTF-8 bytes, the crate's fallback), plus
/// the `<0xXX>` byte-fallback form for tokenizers that ship it.
fn piece_bytes(piece: &str, inv: &std::collections::HashMap<char, u8>) -> Vec<u8> {
    if piece.len() == 6 && piece.starts_with("<0x") && piece.ends_with('>') {
        if let Ok(b) = u8::from_str_radix(&piece[3..5], 16) { return vec![b]; }
    }
    let mut bytes = Vec::with_capacity(piece.len());
    for c in piece.chars() {
        match inv.get(&c) {
            Some(b) => bytes.push(*b),
            None => return piece.as_bytes().to_vec(),
        }
    }
    bytes
}

/// Special-token ids from tokenizer.json's `added_tokens` (honoring the `special` flag, the
/// crate's skip_special_tokens semantics). Empty on any parse failure (never fires in
/// practice — the crate parsed the same file).
fn special_ids(tokenizer_path: &str) -> std::collections::HashSet<u32> {
    let mut set = std::collections::HashSet::new();
    if let Ok(raw) = std::fs::read(tokenizer_path) {
        if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&raw) {
            if let Some(arr) = v["added_tokens"].as_array() {
                for tok in arr {
                    if tok["special"].as_bool().unwrap_or(false) {
                        if let Some(id) = tok["id"].as_u64() { set.insert(id as u32); }
                    }
                }
            }
        }
    }
    set
}

/// Incremental byte-level stream decoder: reassembles multi-byte UTF-8 chars split across
/// tokens by accumulating RAW bytes and emitting only complete UTF-8 sequences, holding back
/// the ≤3-byte tail. Replaces the lossy per-token `decode(&[t], true)` in the SSE streaming
/// path (server.rs) — without it, "That's wonderful to hear! 😀" arrives as "�" per byte.
/// Applies to ALL qwen byte-level BPE models via the shared server path.
pub struct StreamByteDecoder {
    inv: std::collections::HashMap<char, u8>,
    vocab: std::collections::HashMap<u32, String>,
    specials: std::collections::HashSet<u32>,
    pending: Vec<u8>,
}

impl StreamByteDecoder {
    pub fn new(tokenizer: &Tokenizer, tokenizer_path: &str) -> Self {
        let vocab = tokenizer.get_vocab(true).into_iter()
            .map(|(s, id)| (id as u32, s)).collect();
        Self {
            inv: bytes_to_unicode_inverse(),
            vocab,
            specials: special_ids(tokenizer_path),
            pending: Vec::with_capacity(16),
        }
    }

    /// Feed one generated token id; returns the decodable text (an incomplete trailing
    /// UTF-8 char is held back for the next token). Special tokens are skipped, matching
    /// `decode(ids, true)`.
    pub fn push(&mut self, id: u32) -> String {
        if self.specials.contains(&id) { return String::new(); }
        let piece = match self.vocab.get(&id) { Some(p) => p.clone(), None => return String::new() };
        self.pending.extend(piece_bytes(&piece, &self.inv));
        // Emit the longest complete UTF-8 prefix; keep the trailing ≤3 bytes that could be
        // a truncated char.
        for cut in 0..=3.min(self.pending.len()) {
            let end = self.pending.len() - cut;
            if let Ok(s) = std::str::from_utf8(&self.pending[..end]) {
                let out = s.to_string();
                self.pending.drain(..end);
                return out;
            }
        }
        String::new()
    }

    /// Flush the held-back tail at stream end; invalid bytes become U+FFFD (the crate's
    /// lossy semantics for a genuinely truncated sequence).
    pub fn finish(&mut self) -> String {
        let out = String::from_utf8_lossy(&self.pending).to_string();
        self.pending.clear();
        out
    }
}
