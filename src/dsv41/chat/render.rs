//! `render_message` and `encode_messages` of `encoding.py` (DeepSeek-V4.1), byte for byte.

use crate::dsv4_chat::Json;

use super::effort;
use super::error::ChatError;
use super::jx::{self, get, get_str, truthy};
use super::preprocess::{
    merge_tool_messages, process_image_messages, sort_tool_results_by_call_order, validate_no_image_sp_tokens,
};
use super::pyjson::{py_str, to_json};
use super::tools::{
    fmt_named, render_tool_calls, render_tools, tool_calls_from_openai_format, tools_from_openai_format,
    RESPONSE_FORMAT_TEMPLATE, THINKING_END, THINKING_START,
};

pub const BOS_TOKEN: &str = "<｜begin▁of▁sentence｜>";
pub const EOS_TOKEN: &str = "<｜end▁of▁sentence｜>";
pub const USER_SP_TOKEN: &str = "<｜User｜>";
pub const ASSISTANT_SP_TOKEN: &str = "<｜Assistant｜>";
pub const SYSTEM_SP_TOKEN: &str = "<｜System｜>";
pub const LATEST_REMINDER_SP_TOKEN: &str = "<｜latest_reminder｜>";

pub fn task_sp_token(task: &str) -> Option<&'static str> {
    Some(match task {
        "action" => "<｜action｜>",
        "query" => "<｜query｜>",
        "authority" => "<｜authority｜>",
        "domain" => "<｜domain｜>",
        "title" => "<｜title｜>",
        "read_url" => "<｜read_url｜>",
        _ => return None,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingMode {
    Chat,
    Thinking,
}

impl ThinkingMode {
    pub fn parse(s: &str) -> Result<Self, ChatError> {
        match s {
            "chat" => Ok(Self::Chat),
            "thinking" => Ok(Self::Thinking),
            other => Err(ChatError::assertion(format!("Invalid thinking_mode `{other}`"))),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EncodeOptions {
    pub thinking: ThinkingMode,
    /// Earlier turns that are already part of the prompt (multi-chunk serving); BOS is added only
    /// when this is empty.
    pub context: Vec<Json>,
    pub drop_thinking: bool,
    pub add_default_bos_token: bool,
    /// `None` = the default (`high`); an integer 1..=100 or `low` / `high` / `max` as JSON.
    pub reasoning_effort: Option<Json>,
}

impl EncodeOptions {
    pub fn new(thinking: ThinkingMode) -> Self {
        Self { thinking, context: Vec::new(), drop_thinking: true, add_default_bos_token: true, reasoning_effort: None }
    }
}

/// The prompt string and the image records its placeholders stand for, in prompt order.
#[derive(Debug, Clone, PartialEq)]
pub struct Encoded {
    pub prompt: String,
    pub images: Vec<Json>,
}

/// `find_last_user_index`: user messages, and mid-conversation system messages; a user message that
/// holds only tool results does not start a new turn.
pub fn find_last_user_index(messages: &[Json]) -> isize {
    for idx in (0..messages.len()).rev() {
        let role = get_str(&messages[idx], "role");
        if role == Some("user") || (role == Some("system") && idx > 0) {
            if let Some(Json::Array(blocks)) = get(&messages[idx], "content_blocks") {
                if !blocks.is_empty()
                    && !blocks.iter().any(|b| matches!(get_str(b, "type"), Some("text") | Some("image") | Some("image_url")))
                {
                    continue;
                }
            }
            return idx as isize;
        }
    }
    -1
}

/// `x or ""` for a field the template formats as text.
fn text_or_empty(v: Option<&Json>) -> Result<String, ChatError> {
    match v {
        None | Some(Json::Null) => Ok(String::new()),
        Some(Json::Str(s)) => Ok(s.clone()),
        Some(other) if !super::pyjson::truthy(other) => Ok(String::new()),
        Some(_) => Err(ChatError::input("message text must be a string")),
    }
}

fn block_text(b: &Json) -> Result<String, ChatError> {
    match get(b, "text") {
        None => Ok(String::new()),
        Some(Json::Str(s)) => Ok(s.clone()),
        Some(_) => Err(ChatError::input("text block text must be a string")),
    }
}

fn user_body(msg: &Json) -> Result<String, ChatError> {
    if truthy(get(msg, "content_blocks")) {
        let Some(Json::Array(blocks)) = get(msg, "content_blocks") else {
            return Err(ChatError::input("content_blocks must be a list"));
        };
        let mut parts: Vec<String> = Vec::with_capacity(blocks.len());
        for block in blocks {
            if !matches!(block, Json::Object(_)) {
                return Err(ChatError::input("content block must be an object"));
            }
            match get_str(block, "type") {
                Some("text") => parts.push(block_text(block)?),
                Some("tool_result") => {
                    let content = match get(block, "content") {
                        None => String::new(),
                        Some(Json::Array(items)) => {
                            let mut texts = Vec::with_capacity(items.len());
                            for b in items {
                                if !matches!(b, Json::Object(_)) {
                                    return Err(ChatError::input("tool_result content item must be an object"));
                                }
                                if get_str(b, "type") == Some("text") {
                                    texts.push(block_text(b)?);
                                } else {
                                    texts.push(format!("[Unsupported {}]", type_name(b)));
                                }
                            }
                            texts.join("\n\n")
                        }
                        Some(other) => py_str(other)?,
                    };
                    parts.push(format!("<tool_result>{content}</tool_result>"));
                }
                _ => parts.push(format!("[Unsupported {}]", type_name(block))),
            }
        }
        Ok(parts.join("\n\n"))
    } else {
        text_or_empty(get(msg, "content"))
    }
}

/// `{b.get('type')}` as Python prints it.
fn type_name(b: &Json) -> String {
    match get(b, "type") {
        None | Some(Json::Null) => "None".to_string(),
        Some(v) => py_str(v).unwrap_or_else(|_| "?".to_string()),
    }
}

/// `render_message`: one message at `index`, with the generation header when it is last (or followed
/// by something that is not an assistant turn or a reminder it should precede).
pub fn render_message(
    index: usize,
    messages: &[Json],
    thinking: ThinkingMode,
    drop_thinking: bool,
    reasoning_effort: Option<&Json>,
) -> Result<String, ChatError> {
    if index >= messages.len() {
        return Err(ChatError::assertion("message index out of range"));
    }
    let msg = &messages[index];
    let last_user_idx = find_last_user_index(messages);
    let role = get_str(msg, "role").map(str::to_string);
    let role_s = role.as_deref().unwrap_or("");

    let tools = match get(msg, "tools") {
        Some(t) if truthy(Some(t)) => tools_from_openai_format(t)?,
        _ => Vec::new(),
    };
    let tool_calls = match get(msg, "tool_calls") {
        Some(t) if truthy(Some(t)) => tool_calls_from_openai_format(t)?,
        _ => Vec::new(),
    };
    let wo_eos = jx::get(msg, "wo_eos").map_or(false, |v| super::pyjson::truthy(v));

    // reasoning effort is validated on every call, as in the reference
    let budget = effort::budget(reasoning_effort)?;
    let effort_prompt = if index == 0 && thinking == ThinkingMode::Thinking { effort::effort_prefix(budget) } else { String::new() };
    let mut prompt = String::new();
    if index == 0 && (!effort_prompt.is_empty() || role_s == "system") {
        prompt.push_str(SYSTEM_SP_TOKEN);
    }
    prompt.push_str(&effort_prompt);

    match role_s {
        "system" => {
            if index > 0 {
                prompt.push_str(SYSTEM_SP_TOKEN);
            }
            prompt.push_str(&text_or_empty(get(msg, "content"))?);
            if !tools.is_empty() {
                prompt.push_str("\n\n");
                prompt.push_str(&render_tools(&tools));
            }
            if let Some(rf) = get(msg, "response_format") {
                if truthy(Some(rf)) {
                    prompt.push_str("\n\n");
                    prompt.push_str(&fmt_named(RESPONSE_FORMAT_TEMPLATE, &[("schema", to_json(rf).as_str())]));
                }
            }
        }
        "user" => {
            prompt.push_str(USER_SP_TOKEN);
            prompt.push_str(&user_body(msg)?);
        }
        "latest_reminder" => {
            prompt.push_str(LATEST_REMINDER_SP_TOKEN);
            prompt.push_str(&match get(msg, "content") {
                None => "None".to_string(),
                Some(v) => py_str(v)?,
            });
        }
        "tool" => {
            return Err(ChatError::not_implemented(
                "deepseek_v41 merges tool messages into user; please preprocess with merge_tool_messages()",
            ));
        }
        "assistant" => {
            let tc_content = render_tool_calls(&tool_calls)?;
            let summary = text_or_empty(get(msg, "content"))?;
            let rc = text_or_empty(get(msg, "reasoning_content"))?;
            let prev_has_task = index >= 1 && jx::present(&messages[index - 1], "task");
            let mut thinking_part = String::new();
            if thinking == ThinkingMode::Thinking && !prev_has_task && (!drop_thinking || (index as isize) > last_user_idx) {
                thinking_part = format!("{rc}{THINKING_END}");
            }
            prompt.push_str(&thinking_part);
            prompt.push_str(&summary);
            prompt.push_str(&tc_content);
            if !wo_eos {
                prompt.push_str(EOS_TOKEN);
            }
        }
        other => return Err(ChatError::not_implemented(format!("Unknown role: {other}"))),
    }

    // transition tokens based on what follows
    if index + 1 < messages.len() {
        let next = get_str(&messages[index + 1], "role");
        if next != Some("assistant") && next != Some("latest_reminder") {
            return Ok(prompt);
        }
    }

    if jx::present(msg, "task") {
        let task = get_str(msg, "task").unwrap_or("");
        let sp = task_sp_token(task).ok_or_else(|| {
            ChatError::assertion(format!("Invalid task: '{task}'. Valid tasks are: action, query, authority, domain, title, read_url"))
        })?;
        if task != "action" {
            prompt.push_str(sp);
        } else {
            prompt.push_str(ASSISTANT_SP_TOKEN);
            prompt.push_str(if thinking != ThinkingMode::Thinking { THINKING_END } else { THINKING_START });
            prompt.push_str(sp);
        }
    } else if role_s == "user" || (role_s == "system" && index > 0) {
        prompt.push_str(ASSISTANT_SP_TOKEN);
        if !drop_thinking && thinking == ThinkingMode::Thinking {
            prompt.push_str(THINKING_START);
        } else if drop_thinking && thinking == ThinkingMode::Thinking && (index as isize) >= last_user_idx {
            prompt.push_str(THINKING_START);
        } else {
            prompt.push_str(THINKING_END);
        }
    }
    Ok(prompt)
}

/// `_drop_thinking_messages`
fn drop_thinking_messages(messages: &[Json]) -> Vec<Json> {
    let last_user = find_last_user_index(messages);
    let mut out = Vec::new();
    for (idx, msg) in messages.iter().enumerate() {
        let role = get_str(msg, "role");
        let keep = matches!(role, Some("user") | Some("system") | Some("tool") | Some("latest_reminder") | Some("direct_search_results"));
        if keep || (idx as isize) >= last_user {
            out.push(msg.clone());
        } else if role == Some("assistant") {
            let mut m = msg.clone();
            jx::remove(&mut m, "reasoning_content");
            out.push(m);
        }
    }
    out
}

/// `_encode_messages_text`
fn encode_messages_text(
    messages: &[Json],
    thinking: ThinkingMode,
    context: &[Json],
    drop_thinking: bool,
    add_bos: bool,
    effort: Option<&Json>,
) -> Result<String, ChatError> {
    let mut full: Vec<Json> = context.to_vec();
    full.extend_from_slice(messages);
    let mut prompt = if add_bos && context.is_empty() { BOS_TOKEN.to_string() } else { String::new() };

    let mut effective_drop = drop_thinking;
    if full.iter().any(|m| truthy(get(m, "tools"))) {
        effective_drop = false;
    }

    let (num_to_render, context_len);
    if thinking == ThinkingMode::Thinking && effective_drop {
        full = drop_thinking_messages(&full);
        num_to_render = full.len() - drop_thinking_messages(context).len().min(full.len());
        context_len = full.len() - num_to_render;
    } else {
        num_to_render = messages.len();
        context_len = context.len();
    }
    for idx in 0..num_to_render {
        prompt.push_str(&render_message(idx + context_len, &full, thinking, effective_drop, effort)?);
    }
    Ok(prompt)
}

/// Encode OpenAI-style messages into the DeepSeek-V4.1 prompt string and the image records.
/// `tools` and `response_format` ride on the first message, as `encoding.py` expects.
pub fn encode_messages(messages: &[Json], opts: &EncodeOptions) -> Result<Encoded, ChatError> {
    for m in opts.context.iter().chain(messages.iter()) {
        validate_no_image_sp_tokens(m)?;
    }
    let mut context = merge_tool_messages(&opts.context)?;
    let merged = merge_tool_messages(messages)?;
    let ctx_len = context.len();
    context.extend(merged);
    sort_tool_results_by_call_order(&mut context);
    let messages_sorted = context.split_off(ctx_len);
    let (processed_context, _) = if ctx_len > 0 { process_image_messages(&context)? } else { (Vec::new(), Vec::new()) };
    let (processed_messages, images) = process_image_messages(&messages_sorted)?;
    let prompt = encode_messages_text(
        &processed_messages,
        opts.thinking,
        &processed_context,
        opts.drop_thinking,
        opts.add_default_bos_token,
        opts.reasoning_effort.as_ref(),
    )?;
    Ok(Encoded { prompt, images })
}

/// Attach a request's `tools` to its first message (`load_cases` of the reference).
pub fn attach_tools(messages: &mut Vec<Json>, tools: &Json) -> Result<(), ChatError> {
    if messages.is_empty() {
        return Err(ChatError::value("A case with tools must contain at least one message"));
    }
    jx::set(&mut messages[0], "tools", tools.clone());
    Ok(())
}
