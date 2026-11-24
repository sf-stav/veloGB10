//! Request-level resolution for the serving layer: thinking toggles, reasoning effort and tool choice.
//!
//! The release ships two spellings of the same format: `encoding.py` (strict, the authority) and a
//! `chat_template.jinja` for the Hugging Face API, which also takes `thinking` / `enable_thinking`,
//! the OpenAI effort names (`none`, `minimal`, `medium`, `xhigh`) and `chat_template_kwargs`. A server
//! flag, a request field and a kwargs entry can all name the same thing, so the order is fixed here
//! and tested; the encoder itself stays strict.

use crate::dsv4_chat::Json;

use super::catalogue::Catalogue;
use super::error::ChatError;
use super::jx::get;
use super::render::ThinkingMode;

/// What a request said about thinking, from the most to the least specific source.
#[derive(Debug, Clone, Default)]
pub struct ThinkingRequest<'a> {
    /// `chat_template_kwargs` (a JSON object), if sent.
    pub chat_template_kwargs: Option<&'a Json>,
    /// The top-level `reasoning_effort` field.
    pub reasoning_effort: Option<&'a Json>,
    /// A request-level boolean (`thinking` / `enable_thinking` as a top-level field).
    pub thinking: Option<bool>,
    /// The server's `--reasoning-effort` default.
    pub server_effort: Option<&'a Json>,
    /// The server default for thinking (`--thinking on|off`); `None` = on, the model's trained default.
    pub server_thinking: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct ResolvedThinking {
    pub mode: ThinkingMode,
    /// The effort for `EncodeOptions::reasoning_effort`, aliases already mapped to the strict forms.
    pub effort: Option<Json>,
}

fn bool_of(v: Option<&Json>, what: &str) -> Result<Option<bool>, ChatError> {
    match v {
        None | Some(Json::Null) => Ok(None),
        Some(Json::Bool(b)) => Ok(Some(*b)),
        Some(_) => Err(ChatError::value(format!("{what} must be a boolean"))),
    }
}

/// OpenAI effort names onto the strict ones, as the release template maps them. `none`, `off`, and
/// `no_think` are "75, and thinking off unless set explicitly".
fn lenient_effort(e: &Json) -> (Json, bool /* none */) {
    if let Json::Str(s) = e {
        // INT12: `medium` is 62, the pack's own chat template (`effort_map` in chat_template.jinja: low 50, medium 62, high 75,
        // xhigh 75, max 100) and the external reference agree; it was mapped to high (75). `xhigh` / `minimal`: owner decisions
        if s == "medium" { return (Json::Num("62".to_string()), false); }
        let (mapped, none) = match s.as_str() {
            "none" | "off" | "no_think" => ("high", true),
            "minimal" => ("low", false),
            "xhigh" => ("high", false),
            other => (other, false),
        };
        return (Json::Str(mapped.to_string()), none);
    }
    (e.clone(), false)
}

/// Order of precedence (first present wins):
/// thinking: kwargs `thinking_mode` ("thinking"|"chat") > kwargs `thinking` > kwargs `enable_thinking` >
///   request `thinking` > effort `none` (kwargs, request or server) > server `--thinking` > on.
/// effort: kwargs `reasoning_effort` > request `reasoning_effort` > server `--reasoning-effort` > default.
pub fn resolve_thinking(req: &ThinkingRequest<'_>) -> Result<ResolvedThinking, ChatError> {
    let kw = req.chat_template_kwargs;
    if let Some(k) = kw {
        if !matches!(k, Json::Object(_)) {
            return Err(ChatError::value("'chat_template_kwargs' must be a JSON object"));
        }
    }
    let kget = |name: &str| kw.and_then(|k| get(k, name));
    let effort_src = [kget("reasoning_effort"), req.reasoning_effort, req.server_effort]
        .into_iter()
        .flatten()
        .find(|v| !matches!(v, Json::Null) && !matches!(v, Json::Str(s) if s.is_empty()));
    let (effort, effort_none) = match effort_src {
        Some(e) => {
            let (m, n) = lenient_effort(e);
            (Some(m), n)
        }
        None => (None, false),
    };
    let explicit = match kget("thinking_mode") {
        Some(Json::Str(s)) => Some(ThinkingMode::parse(s)?),
        Some(Json::Null) | None => None,
        Some(_) => return Err(ChatError::assertion("Invalid thinking_mode")),
    };
    let from_bool = |b: bool| if b { ThinkingMode::Thinking } else { ThinkingMode::Chat };
    let mode = if let Some(m) = explicit {
        m
    } else if let Some(b) = bool_of(kget("thinking"), "chat_template_kwargs.thinking")? {
        from_bool(b)
    } else if let Some(b) = bool_of(kget("enable_thinking"), "chat_template_kwargs.enable_thinking")? {
        from_bool(b)
    } else if let Some(b) = req.thinking {
        from_bool(b)
    } else if effort_none {
        ThinkingMode::Chat
    } else if let Some(b) = req.server_thinking {
        from_bool(b)
    } else {
        ThinkingMode::Thinking
    };
    Ok(ResolvedThinking { mode, effort })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Named(String),
}

impl ToolChoice {
    pub fn from_json(v: Option<&Json>) -> Result<Self, ChatError> {
        match v {
            None | Some(Json::Null) => Ok(Self::Auto),
            Some(Json::Str(s)) => match s.as_str() {
                "auto" => Ok(Self::Auto),
                "none" => Ok(Self::None),
                "required" => Ok(Self::Required),
                other => Err(ChatError::value(format!("unknown tool_choice '{other}'"))),
            },
            Some(o @ Json::Object(_)) => match get(o, "function").and_then(|f| get(f, "name")) {
                Some(Json::Str(n)) => Ok(Self::Named(n.clone())),
                _ => match get(o, "type").and_then(|t| t.as_str()) {
                    Some("auto") => Ok(Self::Auto),
                    Some("none") => Ok(Self::None),
                    Some("required" | "any") => Ok(Self::Required),
                    _ => Err(ChatError::value("tool_choice object needs function.name")),
                },
            },
            Some(_) => Err(ChatError::value("tool_choice must be a string or an object")),
        }
    }
}

/// What a request's tools mean for the prompt and for the parser.
#[derive(Debug, Clone)]
pub struct PreparedTools {
    /// Attach to the first message for `encode_messages` (`None` = no tool block in the prompt).
    pub prompt_tools: Option<Json>,
    /// The parser's catalogue (`None` = never read calls out of the reply).
    pub catalogue: Option<Catalogue>,
}

/// `tool_choice: "none"` hides the tools from the prompt unless `--keep-tools-when-tool-choice-none`
/// is set; even then no call is ever read out of the reply ("none" means none). A named choice
/// restricts the parser to that tool.
pub fn prepare_tools(tools: Option<&Json>, choice: &ToolChoice, keep_tools_when_none: bool) -> Result<PreparedTools, ChatError> {
    let Some(tools) = tools.filter(|t| matches!(t, Json::Array(a) if !a.is_empty())) else {
        return Ok(PreparedTools { prompt_tools: None, catalogue: None });
    };
    let cat = Catalogue::from_openai_tools(tools)?;
    Ok(match choice {
        ToolChoice::None => PreparedTools { prompt_tools: if keep_tools_when_none { Some(tools.clone()) } else { None }, catalogue: None },
        ToolChoice::Named(n) => PreparedTools { prompt_tools: Some(tools.clone()), catalogue: Some(cat.only(n)) },
        ToolChoice::Auto | ToolChoice::Required => PreparedTools { prompt_tools: Some(tools.clone()), catalogue: Some(cat) },
    })
}

/// `--reasoning-config`: the cap on reasoning tokens (`thinking_token_budget`). The engine loop forces
/// `</think>` (id `special::THINK_END_ID`) when the count is reached; the request value wins over the
/// server default.
pub fn thinking_token_budget(request: Option<u64>, server_default: Option<u64>) -> Option<u64> {
    request.or(server_default).filter(|n| *n > 0)
}
