//! Structural token ids of the Inkling-2 vocabulary (`tokenizer.json` added tokens of the stock pack; `--probe-inkling2-tok` and the
//! `tokens_match_the_pack` test check them against the file). A reply is a sequence of messages:
//! `<|message_model|> [header text] <|content_KIND|> body <|end_message|>`, ending with `<|content_model_end_sampling|>` (EOS).

pub const MESSAGE_USER: u32 = 200000;
pub const MESSAGE_MODEL: u32 = 200001;
pub const MESSAGE_SYSTEM: u32 = 200002;
pub const MESSAGE_TOOL: u32 = 200003;
pub const CONTENT_TEXT: u32 = 200004;
pub const CONTENT_IMAGE: u32 = 200005;
/// End of sampling: the one stop token (`generation_config.eos_token_id`).
pub const EOS: u32 = 200006;
pub const CONTENT_THINKING: u32 = 200008;
pub const END_MESSAGE: u32 = 200010;
pub const CONTENT_TOOL_ERROR: u32 = 200022;
pub const CONTENT_XML: u32 = 200024;
pub const BEGIN_OF_TEXT: u32 = 200028;
pub const CONTENT_INVOKE_TOOL_JSON: u32 = 200049;
/// The image placeholder: one in the prompt per image; the vision stage expands each into rows x cols copies.
pub const IMAGE: u32 = 200054;
pub const CONTENT_INVOKE_TOOL_TEXT: u32 = 200057;
/// Reserved by the checkpoint (native audio ids kept as unused tokens): the model must never sample one.
pub const CONTENT_AUDIO_INPUT: u32 = 200020;
pub const AUDIO_END: u32 = 200043;

/// The (name, id) table the tests compare against `tokenizer.json`.
pub const TABLE: [(&str, u32); 14] = [
    ("<|message_user|>", MESSAGE_USER), ("<|message_model|>", MESSAGE_MODEL), ("<|message_system|>", MESSAGE_SYSTEM),
    ("<|message_tool|>", MESSAGE_TOOL), ("<|content_text|>", CONTENT_TEXT), ("<|content_image|>", CONTENT_IMAGE),
    ("<|content_model_end_sampling|>", EOS), ("<|content_thinking|>", CONTENT_THINKING), ("<|end_message|>", END_MESSAGE),
    ("<|content_tool_error|>", CONTENT_TOOL_ERROR), ("<|content_xml|>", CONTENT_XML), ("<|begin_of_text|>", BEGIN_OF_TEXT),
    ("<|content_invoke_tool_json|>", CONTENT_INVOKE_TOOL_JSON), ("<|content_invoke_tool_text|>", CONTENT_INVOKE_TOOL_TEXT),
];

/// The stop tokens of a chat turn: EOS only (the pack's `generation_config.json` and `config.json` agree).
pub fn stop_token_ids() -> Vec<u32> { vec![EOS] }

/// Ids the checkpoint keeps as unused tokens for audio: sampling one is an engine error, never text.
pub fn is_reserved(id: u32) -> bool { matches!(id, CONTENT_AUDIO_INPUT | AUDIO_END) }

/// Every content-kind token: the token after a header that opens the body of a message.
pub fn is_content_kind(id: u32) -> bool {
    matches!(id, CONTENT_TEXT | CONTENT_IMAGE | CONTENT_THINKING | CONTENT_XML | CONTENT_TOOL_ERROR
                 | CONTENT_INVOKE_TOOL_JSON | CONTENT_INVOKE_TOOL_TEXT | CONTENT_AUDIO_INPUT)
}
