//! `impl QwenTokenizer`: construction from `tokenizer.json`, encode/decode, stop-token union, think tags and chat-template
//! rendering. The struct itself is in `mod.rs`; the template environment comes from `jinja.rs` (map in `mod.rs`).

use super::*;

impl QwenTokenizer {
    pub fn from_file(path: &str) -> Result<Self> {
        let tokenizer = load_tokenizer(path)?;
        let (chat_env, template_meta, raw_tool_args_history) = match load_chat_env(path)? {
            Some((env, meta, raw)) => (Some(env), Some(meta), raw),
            None => (None, None, false),
        };
        let model_dir = Path::new(path).parent().map(|p| p.to_path_buf());
        let cache = if wp01_env(crate::opt!("wp01-off")) {
            eprintln!("[tokenizer] --wp01-off=1: legacy per-request vocab/special-table rebuilds");
            None
        } else {
            Some(TokCache::build(&tokenizer, &stream_json_path(model_dir.as_deref())))
        };
        let me = Self {
            tokenizer, chat_env, model_dir, tool_call_format: None, auto_disable_think_tools: false,
            template_meta, raw_tool_args_history, cache,
        };
        if me.cache.is_some() && wp01_env(crate::opt!("wp01-xcheck")) { me.wp01_xcheck(); }
        Ok(me)
    }

    /// --wp01-xcheck=1 (diagnostic): exhaustive load-time diff of every cached table against
    /// the legacy per-call expressions — think/eos/vocab_size, and for EVERY id (plus a margin past
    /// the table) the special flag and the piece bytes the legacy decoder would append. Prints
    /// `[wp01-xcheck] ... mismatches=0` on success, the first mismatches otherwise. Per-request
    /// decoders additionally shadow-run the legacy decoder and compare every emitted string.
    fn wp01_xcheck(&self) {
        let Some(c) = &self.cache else { return };
        let t0 = std::time::Instant::now();
        let vocab = self.tokenizer.get_vocab(true);
        let legacy_think = vocab.contains_key("</think:opensource>");
        let legacy_eos = vocab.get("<|endoftext|>").copied().unwrap_or(151643) as u32;
        let legacy_vsz = self.tokenizer.get_vocab_size(true);
        let mut bad = 0usize;
        if legacy_think != c.think_hy || legacy_eos != c.eos_id || legacy_vsz != c.vocab_size {
            bad += 1;
            eprintln!("[wp01-xcheck] MISMATCH scalars: think {}/{} eos {}/{} vocab_size {}/{} (new/old)",
                      c.think_hy, legacy_think, c.eos_id, legacy_eos, c.vocab_size, legacy_vsz);
        }
        let legacy = LegacyStreamDecoder::new(&self.tokenizer, &stream_json_path(self.model_dir.as_deref()));
        let n = c.stream.kind.len() as u32;
        for id in 0..n.saturating_add(64) {
            let old: Option<Vec<u8>> = if legacy.specials.contains(&id) { None }
                else { legacy.vocab.get(&id).map(|p| piece_bytes(p, &legacy.inv)) };
            let new: Option<&[u8]> = c.stream.piece(id);
            if old.as_deref() != new {
                bad += 1;
                if bad <= 8 {
                    eprintln!("[wp01-xcheck] MISMATCH id={id}: new={new:?} old={old:?}");
                }
            }
        }
        eprintln!("[wp01-xcheck] load tables: ids={} specials={} dup_ids={} mismatches={bad} ({:.1} ms)",
                  n, legacy.specials.len(), c.stream.dup_ids, t0.elapsed().as_secs_f64() * 1e3);
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
        let path = stream_json_path(self.model_dir.as_deref());
        // WP01: the id->bytes/special tables were built once at load; a request only allocates
        // its ≤3-byte pending buffer. --wp01-off=1 = the legacy per-request rebuild.
        match &self.cache {
            Some(c) => {
                let shadow = wp01_env(crate::opt!("wp01-xcheck"))
                    .then(|| Box::new(LegacyStreamDecoder::new(&self.tokenizer, &path)));
                StreamByteDecoder::from_tables(std::sync::Arc::clone(&c.stream), shadow)
            }
            None => StreamByteDecoder::new(&self.tokenizer, &path),
        }
    }

    pub fn eos_token_id(&self) -> u32 {
        if let Some(c) = &self.cache { return c.eos_id; }
        self.tokenizer.get_vocab(true).get("<|endoftext|>").copied().unwrap_or(151643) as u32
    }

    /// Vocab size INCLUDING added tokens — the exclusive upper bound of a valid token id.
    /// Used by the /v1/tokenize endpoint to reject id-list prompts that carry garbage ids.
    pub fn vocab_size(&self) -> usize {
        if let Some(c) = &self.cache { return c.vocab_size; }
        self.tokenizer.get_vocab_size(true)
    }

    /// The think-block markers for this model's chat format, and whether a fresh generation starts
    /// INSIDE a think block. Qwen templates prime `<think>\n` (starts in reasoning, closes with
    /// `</think>`). hy_v3's no_think prompt renders `<think:opensource></think:opensource>` INTO the
    /// prompt (the empty block is already closed), so generation starts as CONTENT; its markers are
    /// the `:opensource`-suffixed forms. Resolved from the vocab, never hardcoded per family.
    pub fn think_tags(&self) -> (&'static str, &'static str, bool) {
        // WP01: resolved once at load (was a 248K-entry HashMap build per call, 3x per request).
        let hy = match &self.cache {
            Some(c) => c.think_hy,
            None => self.tokenizer.get_vocab(true).contains_key("</think:opensource>"),
        };
        if hy {
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
    /// Every (token id, piece BYTES) pair of the model vocabulary. This is the token layer of
    /// the JSON-schema FSM walk (W2): the trie is built over bytes because a piece is not always
    /// valid UTF-8 on its own, and the schema machine consumes bytes.
    /// (token id, the exact bytes the token adds to the OUTPUT text) for every ordinary token —
    /// the table the JSON-schema mask walks. Special tokens are left out (they are never
    /// allowed inside a constrained answer, and the stream decoder prints nothing for them).
    ///
    /// This used to return `get_vocab(true)`'s piece STRINGS as bytes. For a byte-level BPE
    /// vocabulary (Qwen) those strings are the GPT-2 byte-to-unicode encoding, not the output:
    /// a leading space is `Ġ` (bytes C4 A0) and a newline `Ċ`, so the schema machine saw bytes
    /// the client never receives — no space could ever be produced (an enum value containing a
    /// space was unreachable) and special tokens such as `<|im_end|>` looked like legal
    /// string characters. The bytes now come from the same tables the stream decoder uses
    /// (`piece_bytes` over the inverse byte map), so mask and output agree byte for byte.
    pub fn vocab_pieces(&self) -> Vec<(u32, Vec<u8>)> {
        let built;
        let tables: &StreamTables = match &self.cache {
            Some(c) => &c.stream,
            None => {
                let json = stream_json_path(self.model_dir.as_deref());
                built = StreamTables::build(&self.tokenizer, &self.tokenizer.get_vocab(true), &special_ids(&json));
                &built
            }
        };
        (0..tables.kind.len() as u32)
            .filter_map(|id| tables.piece(id).map(|b| (id, b.to_vec())))
            .collect()
    }

    /// The exact output bytes of ONE token: the ordinary-token piece bytes (the same tables the
    /// stream decoder and the JSON-schema mask use), else what `decode(&[id], skip_special)`
    /// prints for it (an added token's literal text; nothing for a skipped special token). A
    /// piece is not always valid UTF-8 on its own (a multi-byte character can span tokens), so
    /// the bytes of a token sequence concatenate to its decoded text.
    pub fn token_bytes(&self, id: u32, skip_special: bool) -> Vec<u8> {
        match &self.cache {
            Some(c) => {
                if let Some(b) = c.stream.piece(id) { return b.to_vec(); }
            }
            // without the load-time tables: the same raw piece bytes, derived from the vocab
            // piece directly (a single-token `decode` would replace a partial UTF-8 character)
            None => {
                if !self.tokenizer.get_added_tokens_decoder().contains_key(&id) {
                    if let Some(p) = self.tokenizer.id_to_token(id) {
                        return piece_bytes(&p, &bytes_to_unicode_inverse());
                    }
                }
            }
        }
        self.decode(&[id], skip_special).unwrap_or_default().into_bytes()
    }

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
                               reasoning_effort: Option<&str>,
                               tpl_kwargs: Option<&serde_json::Value>,
                               thinking: ThinkingMode) -> Result<String> {
        self.render_chat(messages, tools, true, reasoning_effort, tpl_kwargs, thinking)
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
                                      reasoning_effort: Option<&str>,
                                      tpl_kwargs: Option<&serde_json::Value>,
                                      thinking: ThinkingMode) -> Result<String> {
        self.render_chat(messages, tools, false, reasoning_effort, tpl_kwargs, thinking)
    }

    fn render_chat(&self, messages: &[ChatMessage], tools: Option<&[serde_json::Value]>,
                   add_generation_prompt: bool, reasoning_effort: Option<&str>,
                   tpl_kwargs: Option<&serde_json::Value>, thinking: ThinkingMode) -> Result<String> {
        if let Some(env) = &self.chat_env {
            let msgs: Vec<serde_json::Value> =
                messages.iter().map(|m| m.to_template_json(self.raw_tool_args_history)).collect();
            // NAIVE S5: a toolless request renders `tools` as an EMPTY LIST, not null. minijinja answers
            // `none is iterable` with true (Jinja2: false), so a template guarded by
            // `tools is iterable and tools | length > 0` (Naive-N0.5) raised "cannot calculate length of
            // value of type none"; `[]` reads the same as null under every guard in use (`if tools`,
            // `tools and tools is iterable and tools is not mapping`: Qwen3.8).
            let mut ctx = serde_json::json!({
                "messages": msgs,
                "tools": tools.map(|t| serde_json::Value::Array(t.to_vec())).unwrap_or_else(|| serde_json::Value::Array(Vec::new())),
                "add_generation_prompt": add_generation_prompt,
            });
            // W1: `chat_template_kwargs` (the vLLM/SGLang field OpenAI clients send) is an
            // ARBITRARY dict forwarded to the model's own template context. Managed keys are
            // handled with explicit precedence below; every other key passes through verbatim so
            // a template can consume whatever variable it declares. The engine-owned context keys
            // (messages/tools/add_generation_prompt) are never clobberable by a request.
            let mut kw_enable: Option<bool> = None;
            if let Some(v) = tpl_kwargs {
                let map = v.as_object().ok_or_else(|| anyhow::anyhow!(
                    "'chat_template_kwargs' must be a JSON object"))?;
                for (k, val) in map {
                    match k.as_str() {
                        "enable_thinking" => {
                            kw_enable = Some(val.as_bool().ok_or_else(|| anyhow::anyhow!(
                                "'chat_template_kwargs.enable_thinking' must be a boolean (got {val})"))?);
                        }
                        "messages" | "tools" | "add_generation_prompt" => {
                            anyhow::bail!("'chat_template_kwargs' may not override engine key '{k}'");
                        }
                        _ => { ctx[k.as_str()] = val.clone(); }
                    }
                }
            }
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
                if e != "off" && e != "no_think" {
                    ctx["reasoning_effort"] = serde_json::Value::String(e.to_string());
                }
            }
            // W1 fix: `enable_thinking` is passed ONLY when somebody actually asked for a
            // direction (request kwarg > no-think effort > --thinking on/off). In Auto with no
            // request directive we pass NOTHING, so the model's own template decides — a
            // user-edited template with thinking defaulted OFF now renders its own default
            // instead of being overridden by a hardcoded engine value.
            if let Some(et) = resolve_enable_thinking(kw_enable, reasoning_effort, thinking) {
                ctx["enable_thinking"] = serde_json::Value::Bool(et);
            }
            if let Some(f) = &self.tool_call_format {
                ctx["tool_call_format"] = serde_json::Value::String(f.clone());
            }
            if self.auto_disable_think_tools {
                ctx["auto_disable_thinking_with_tools"] = serde_json::Value::Bool(true);
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
