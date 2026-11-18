//! `POST /v1/chat/completions`: the `chat_completions` handler (template render, prefix cache, vision, admission,
//! streaming and non-streaming responses).
//! Siblings: `chat_params.rs`, `chat_stream.rs`, `engine.rs`, `logprobs.rs`, `reasoning.rs`; `mod.rs` has the map.

use super::*;

pub(super) async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    payload: Result<Json<ChatCompletionRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(mut req) = match payload { Ok(j) => j, Err(e) => return bad_json(e) };
    // Log the request parameters the client sent (useful for debugging OpenWebUI behavior)
    crate::dreprintln!(
        "[req] params  temp={:?} top_p={:?} top_k={:?} max_tok={:?} max_completion_tok={:?} rep_pen={:?} presence={:?} freq={:?} stream={} effort={:?} think_budget={:?} ctk={}",
        req.temperature, req.top_p, req.top_k,
        req.max_tokens, req.max_completion_tokens, req.repetition_penalty, req.presence_penalty, req.frequency_penalty,
        req.stream, req.reasoning_effort, req.thinking_token_budget,
        req.chat_template_kwargs.as_ref().map(|v| v.to_string()).unwrap_or_else(|| "-".into())
    );
    if !req.unused_fields.is_empty() {
        let names: Vec<&str> = req.unused_fields.keys().map(|k| k.as_str()).collect();
        crate::dreprintln!("[req] fields not used by this server: {names:?}");
    }
    if let Err(msg) = validate_penalties(req.repetition_penalty, req.presence_penalty,
                                         req.frequency_penalty, req.min_p) {
        return bad_param(msg);
    }
    if let Some(m) = req.max_completion_tokens {
        req.max_tokens = Some(m);
    }
    if let Some(t) = &req.tools {
        let names: Vec<&str> = t.iter()
            .filter_map(|x| x.pointer("/function/name").and_then(|v| v.as_str())).collect();
        crate::reprintln!("[req] tools   {} offered: {:?} tool_choice={:?}", t.len(), names, req.tool_choice);
    }
    // Resolve the effective reasoning effort. Two families, two vocabularies:
    //   - hy_v3's template accepts `no_think|low|high` (default low), and its Rust dsv4 path treats
    //     None/"" as low.
    //   - Qwen3.5's template accepts `xhigh|medium|low` (default xhigh) and RAISES on anything else.
    // Forward the client's value verbatim when it is valid for THIS family's template; convert
    // the OpenAI API convention onto the nearest native level. When NEITHER the request nor
    // --reasoning-effort specifies one, pass None so the model's own template default wins
    // (xhigh for Qwen 3.8, low for hy_v3) — never a hardcoded guess.
    //
    // Qwen 3.8 native (the ONLY values its template accepts; anything else raises => 500):
    //   xhigh (default) | medium | low          ["no_think"/"off" => enable_thinking=false]
    // OpenAI API -> Qwen 3.8 (owner spec 2026-08-30):
    //   none    -> thinking off      (latency-critical; no reasoning)
    //   low     -> low               (efficient reasoning)
    //   medium  -> medium            (balanced; OpenAI's default)
    //   high    -> xhigh             (hard reasoning)
    //   xhigh   -> xhigh             (deep research)
    //   max     -> xhigh             (maximum)
    // hy_v3 native: no_think | low | high  =>  none->no_think, low->low, medium/high/xhigh/max->high
    //
    // REGRESSION FIX (2026-08-30): the 289e1a1 refactor lumped "high" into the no_think arm,
    // so every OpenAI-convention client sending reasoning_effort=high silently LOST thinking.
    // W2 (Phase 13): `response_format` — compiled HERE, before any work. A supported schema arms
    // the token-level mask when enforcement is on; when it is off the request is still SERVED and
    // the response is marked `x-json-schema-enforced: none` (see warn_unenforced_schema_once).
    // NEVER a 400 for a schema this build can parse: refusing was `e737d08`, and it broke the
    // public quality benchmark (88 → 85) by hard-failing requests that used to succeed.
    // Flip JSON_SCHEMA_ENFORCEMENT_ENABLED to true only when the emission path is proven
    // end-to-end: `cargo test --lib json_schema::tests` green AND the W2 probe 6/6 enforced
    // across temperature × TP × spec source. Until then, serving + advertising beats refusing.
    // W2-EXL3: the constant became the opt-in `--json-schema-enforce on`; the EXL3 serve
    // path now arms the mask too (exl3_serve.rs), so the flag is meaningful on every server.
    let json_schema_enforcement_enabled = crate::opts::on(crate::opt!("json-schema-enforce"));
    // The mask is finalised (think deferral) after the prompt is rendered, below.
    let mut schema_pending: Option<crate::json_schema::SchemaMask> = None;
    let mut schema_unenforced: Option<String> = None;
    if let Some(rf) = req.response_format.as_ref() {
        match crate::json_schema::compile_response_format(rf) {
            Ok(None) => {}
            Ok(Some(m)) if json_schema_enforcement_enabled && !state.schema_ok => {
                // PR #31 port, owner rule (fall back, never refuse): an engine that applies no schema mask (the EXL3
                // TP head, the Naive server) serves the request unconstrained and says so in the header
                schema_unenforced = Some(format!(
                    "schema compiled ({}) but this serving engine applies no schema mask (the EXL3 server at TP=1 \
and the NVFP4 server do)", m.summary()));
            }
            Ok(Some(mut m)) if json_schema_enforcement_enabled => {
                m.set_vocab_table(&schema_vocab_table(&state.tokenizer));
                crate::reprintln!("[req] response_format: constrained decoding armed ({})", m.summary());
                schema_pending = Some(m);
            }
            Ok(Some(m)) => {
                schema_unenforced = Some(format!(
                    "schema compiled ({}) but this build does not enforce constrained decoding yet",
                    m.summary()));
            }
            Err(e) => {
                schema_unenforced = Some(format!("schema not compiled: {e}"));
            }
        }
        if let Some(why) = schema_unenforced.as_deref() { warn_unenforced_schema_once(why); }
    }
    // W1: `chat_template_kwargs` (arbitrary dict → the model's own template). Validate LOUDLY
    // before any render: a malformed value used to be invisible (serde dropped the field), and
    // the customer was left with a model that ignored the request. 400, never accept-and-ignore.
    let logprobs_n = match chat_logprobs_param(req.logprobs, req.top_logprobs) {
        Ok(n) => n,
        Err(msg) => return bad_param(msg),
    };
    // Per-token entries carry the generated bytes; a secret the response scrubber removes from the text can
    // span several tokens, so with the scrubber on (and on an engine without them) the request is served with
    // logprobs null instead (logprobs_gate: said once, counted).
    let logprobs_n = logprobs_gate(logprobs_n, &state, req.stream);
    if let Err(e) = crate::tokenizer::enable_thinking_kwarg(req.chat_template_kwargs.as_ref()) {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": {
            "message": e.to_string(), "type": "invalid_request_error", "code": "invalid_chat_template_kwargs",
        }}))).into_response();
    }
    // An unknown effort is the CLIENT's error: a 400 naming the accepted values, never the
    // template's raise surfacing as a plain-text 500.
    if let Some(e) = req.reasoning_effort.as_deref() {
        if !matches!(e, "" | "none" | "off" | "minimal" | "no_think" | "low" | "medium" | "high" | "xhigh" | "max") {
            return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": {
                "message": format!("reasoning_effort must be one of none|minimal|low|medium|high|xhigh|max (got '{e}')"),
                "type": "invalid_request_error", "code": "invalid_reasoning_effort",
            }}))).into_response();
        }
    }
    let effort_owned = resolve_reasoning_effort(&state.tokenizer, req.reasoning_effort.as_deref(),
                                                state.reasoning_effort.as_deref());
    let effort: Option<&str> = effort_owned.as_deref();
    // OpenAI/vLLM tool_choice contract. The engine has no guided-decode forcing, so the
    // semantics are implemented at the prompt level: "required" / a specific function append an
    // explicit forcing instruction naming the constraint; the "none" render/parse DECISION lives
    // in tool_choice_decision() below. Scoring harnesses (tool-eval-bench) drive scenarios
    // through these modes.
    //
    // S-B9-REL-NONETOOLS: with --keep-tools-when-tool-choice-none (default off) a "none"
    // request keeps the tools in the prompt (vLLM's default contract; our legacy default is the
    // llama.cpp approach — tools removed, "the model cannot call what it cannot see"). The
    // prompt then shares its long prefix with the conversation's normal requests, so the prefix
    // cache / WP16 checkpoints survive a compaction/summary turn (the Qwen template renders the
    // tools inside the system block, so removing them invalidates the cache from token 0).
    let (tools_for_template, force_line, parse_as_tool_calls) = tool_choice_decision(
        req.tool_choice.as_ref(), req.tools.as_deref(),
        crate::opts::var(crate::opt!("keep-tools-when-tool-choice-none")).is_ok(),
    );
    let mut messages = req.messages.clone();
    if let Some(line) = force_line {
        apply_force_line(&mut messages, &line); // S4: the ONE append (unit-tested)
    }
    let t_render = std::time::Instant::now();
    let prompt = match state.tokenizer.apply_chat_template(&messages, tools_for_template, effort,
                                                          req.chat_template_kwargs.as_ref(),
                                                          state.thinking) {
        Ok(p) => p,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    };
    let render_ms = t_render.elapsed().as_secs_f64() * 1000.0;

    // Optional diagnostic: dump the exact rendered prompt string so the bytes a model
    // actually sees can be inspected/diffed across models or turns. Enable with
    // --dump-prompt=1. Writes /tmp/rust_infer_prompt_<n>.txt per request.
    if crate::opts::var(crate::opt!("dump-prompt")).is_ok() {
        static DUMP_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = DUMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dump_path = format!("/tmp/rust_infer_prompt_{}.txt", n);
        if std::fs::write(&dump_path, &prompt).is_ok() {
            crate::reprintln!("[req] dumped prompt ({} chars) -> {}", prompt.chars().count(), dump_path);
        }
    }

    let t_encode = std::time::Instant::now();
    let mut prompt_tokens = match encode_off_thread(&state.tokenizer, &prompt, true).await {
        Ok(t) => t,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    };
    let encode_ms = t_encode.elapsed().as_secs_f64() * 1000.0;
    // TTFT fix 0 attribution (--prefill-trace): the request-path pre-model costs.
    if crate::opts::var(crate::opt!("prefill-trace")).ok().is_some() {
        crate::reprintln!("[pf] server render={render_ms:.3}ms encode={encode_ms:.3}ms");
    }

    // V3 vision dispatch: if any message carries images (and the server has a vision tower),
    // decode+preprocess+run the tower, expand the image_pad span in the token stream, and carry the
    // merged embeddings + spans for the model prefill splice. Text-only traffic is unchanged.
    let mut image_embeds: Option<Vec<f32>> = None;
    let mut image_spans: Vec<crate::vision_encoder::ImageSpan> = Vec::new();
    // VID-0 (VIS-5): a content part this server cannot take (video, audio, file, ...) is a 400 naming the
    // type — it used to be dropped silently (the model then answered about content it never saw).
    if let Some(t) = req.messages.iter().flat_map(|m| m.unsupported_parts.iter()).next() {
        return (StatusCode::BAD_REQUEST,
                format!("content part type '{t}' is not supported (text and images only; video input is not available yet)"))
            .into_response();
    }
    let urls: Vec<String> = req.messages.iter()
        .flat_map(|m| m.images.iter().filter_map(|i| i.url.clone()))
        .collect();
    if !urls.is_empty() {
        let vt0 = std::time::Instant::now();
        // Prefer the GPU tower (fast path) unless --vision-cpu forces the CPU reference.
        // VIS-2: decode + preprocess + tower run on a blocking worker, not inline on the async
        // handler (a long tower used to pin a tokio worker under the std Mutex).
        let (gpu, cpu_tower, force_cpu) = (state.vision_gpu.clone(), state.vision_tower.clone(), state.vision_cpu);
        let toks = prompt_tokens.clone();
        let urls_w = urls.clone();
        // PR #16: keep a second CPU-tower handle + request copies for the worker-PANIC arm
        // (the moved values are gone once the spawn_blocking closure owns them).
        let (cpu_fb, urls_fb, toks_fb) = (state.vision_tower.clone(), urls.clone(), prompt_tokens.clone());
        let prep = match tokio::task::spawn_blocking(move || {
            let cpu = || match &cpu_tower {
                Some(tower) => crate::vision_encoder::prepare_vision_request(tower, &urls_w, &toks),
                None => Err(anyhow::anyhow!("no vision tower loaded")),
            };
            match gpu.filter(|_| !force_cpu) {
                Some(g) => vision_dispatch_with_recovery(&g, |gvt| {
                    crate::vision_encoder::prepare_vision_request_gpu(gvt, &urls_w, &toks)
                }, cpu),
                None => cpu(),
            }
        }).await {
            Ok(r) => r,
            // PR #16: a panic inside the GPU worker (kernel launch, decode of the payload) used
            // to fail the request outright; the CPU tower serves the image instead — the request
            // is never silently served without its image. A lock poisoned by that panic is
            // recovered by vision_dispatch_with_recovery on the NEXT request.
            // S-B9-REL-PRPORT3 (review R4): the CPU fallback itself runs on a blocking worker —
            // the CPU tower is multi-second and must never run on the async handler thread.
            Err(e) => vision_panic_fallback(cpu_fb, urls_fb, toks_fb, e).await,
        };
        crate::reprintln!("[vision] dispatch {} images, prepare took {} ms, len={} cache={}/{}",
            urls.len(), vt0.elapsed().as_millis(), prep.as_ref().map(|p| p.image_embeds.len()).unwrap_or(0),
            prep.as_ref().map(|p| p.cache_hits).unwrap_or(0), urls.len());
        match prep {
            Ok(prep) => {
                prompt_tokens = prep.expanded_tokens;
                image_embeds = Some(prep.image_embeds);
                image_spans = prep.spans;
            }
            Err(e) => return (StatusCode::BAD_REQUEST,
                format!("vision preprocessing failed: {e}")).into_response(),
        }
    }

    // truncate_prompt_tokens (vLLM's field): keep the LAST n tokens — the same left-truncation
    // convention the /v1/tokenize handler implements. Runs BEFORE the max_seq_len check so any n
    // that fits turns the over-length 400 into a served request; the boundary snapshot (ckpt_at)
    // below re-renders the FULL template, so a truncated stream simply fails its `n < prompt_len`
    // filter and skips the checkpoint this turn (truncated history is not a prefix of the next
    // full-history turn anyway — the cache can't be trusted across the cut).
    if let Some(n) = req.truncate_prompt_tokens {
        if n == 0 {
            return (StatusCode::BAD_REQUEST,
                    "'truncate_prompt_tokens' must be an integer >= 1").into_response();
        }
        if prompt_tokens.len() > n && !image_spans.is_empty() {
            // VIS-2: dropping the front of an image prompt would leave the image spans pointing at the
            // wrong rows (the spans index the expanded stream) — refuse instead of splicing wrongly.
            return (StatusCode::BAD_REQUEST,
                    "'truncate_prompt_tokens' cannot cut a prompt that carries images").into_response();
        }
        if prompt_tokens.len() > n {
            let dropped = prompt_tokens.len() - n;
            prompt_tokens.drain(..dropped); // keep the LAST n — vLLM's left-truncation convention
            crate::reprintln!("[req] truncate_prompt_tokens: dropped the OLDEST {dropped} of {} prompt \
                       tokens (kept the last {n})", dropped + n);
        }
    }

    let prompt_len = prompt_tokens.len();

    if state.output_prompts > 0 {
        let mname = req.model.clone().unwrap_or_else(|| state.model_name.clone());
        log_request_human(&req, effort, &prompt, prompt_len, state.output_prompts, &mname, render_ms);
    }

    // Where to snapshot the GDN state: the message boundary, i.e. this prompt without its trailing
    // generation prompt. Everything up to here is what the NEXT turn replays verbatim. Rendering the
    // template a second time costs microseconds and saves a whole re-prefill per turn — but only
    // when the scheduler's prefix cache is actually on (batch.rs filters ckpt_at again); with the
    // cache off this second render+encode is pure TTFT cost (fix (e), EXPERT_TTFT_PREFILL_RESPONSE).
    let ckpt_at = if state.prefix_cache {
        state.tokenizer
            .apply_chat_template_no_gen(&req.messages, req.tools.as_deref(), effort,
                                        req.chat_template_kwargs.as_ref(), state.thinking).ok()
            .and_then(|s| state.tokenizer.encode(&s, true).ok())
            .map(|t| t.len())
            .filter(|&n| n > 0 && n < prompt_len)
    } else { None };

    // The KV cache holds exactly `max_seq_len` positions. A prompt past that end used to be written
    // out of bounds — silently, corrupting whatever allocation followed, which showed up as two
    // identical prefills disagreeing. Reject what cannot fit, and cap generation at the room left:
    // running short is a `finish_reason: "length"`, which is in the contract. Corruption is not.
    if generation_room(state.max_seq_len, prompt_len, state.decode_headroom).is_none() {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": {
            "message": format!("This model's maximum context length is {} tokens, but your messages \
                                came to {} tokens and decoding reserves {} more positions. Shorten the input or \
                                restart the server with a larger --max-seq-len.",
                                state.max_seq_len, prompt_len, state.decode_headroom),
            "type": "invalid_request_error", "code": "context_length_exceeded",
        }}))).into_response();
    }
    let room = generation_room(state.max_seq_len, prompt_len, state.decode_headroom)
        .expect("context room checked above");
    let asked = req.max_tokens.unwrap_or(state.default_max_tokens);
    let req_max = asked.min(room);
    // If the KV cache forced generation shorter than asked, SAY SO. A thinking model spends a big fixed
    // chunk on its <think> block, so a silently-shrunk budget looks like "truncated output / only
    // reasoning" as a conversation grows — which is exactly how this surfaced in the wild. Raise
    // --max-seq-len (graphs cost ~nothing here; KV is ~64 KB/token) to give multi-turn room.
    if req_max < asked && req.max_tokens.is_some() {
        crate::reprintln!("[req] max_tokens clamped {} -> {} (KV cache: {}-token prompt + {} reserved decode positions of {}; \
                   raise --max-seq-len)", asked, req_max, prompt_len, state.decode_headroom, state.max_seq_len);
    }
    // Sampling: explicit request values win; anything left out takes the server's default for
    // THIS request's mode (the rendered prompt opens a <think> block or it does not).
    let thinking_on = prompt.trim_end().ends_with(state.tokenizer.think_tags().0);
    // W2-EXL3: a prompt that opens a think block reasons first; the schema applies to the answer
    // after the think-close token (a single token in the supported tokenizers; if it is not, the
    // constraint starts at the first token, the previous behaviour).
    // A schema that cannot be deferred past the reasoning block is not enforced at all (applied
    // from the first token it would constrain the reasoning instead of the answer).
    let schema_mask: Option<std::sync::Arc<crate::json_schema::SchemaMask>> = schema_pending.take().and_then(|mut m| {
        if thinking_on {
            let close = state.tokenizer.think_tags().1;
            let why = if !state.schema_defer {
                Some("this engine cannot start the constraint after the reasoning block (send enable_thinking=false)".to_string())
            } else {
                match state.tokenizer.encode(close, false).as_deref() {
                    Ok([id]) => { m.set_defer_until(Some(*id)); None }
                    _ => Some(format!("the think-close tag {close:?} is not one token")),
                }
            };
            if let Some(why) = why {
                crate::reprintln!("[req] response_format: not enforced: {why}");
                warn_unenforced_schema_once(&why);
                schema_unenforced = Some(why);
                return None;
            }
        }
        Some(std::sync::Arc::new(m))
    });
    // A constrained answer is a JSON document taken literally: no tool-call parsing, and think
    // markers inside its strings are text (see `split_constrained`).
    let literal_content = schema_mask.is_some();
    // PR #31 port: every json_schema / json_object request counted, enforced or served unconstrained
    if schema_mask.is_some() || schema_unenforced.is_some() { crate::metrics::schema_request(schema_mask.is_some()); }
    let parse_as_tool_calls = parse_as_tool_calls && !literal_content;
    let (d_temp, d_top_p, d_top_k) = if thinking_on { state.sampling_defaults.think }
                                     else { state.sampling_defaults.no_think };
    let temperature = req.temperature.unwrap_or(d_temp);
    let top_p = req.top_p.unwrap_or(d_top_p).max(0.01);
    let top_k = req.top_k.unwrap_or(d_top_k);
    crate::dreprintln!("[req] sampling temp={temperature} top_p={top_p} top_k={top_k} (thinking={thinking_on}; \
               sent: temp={:?} top_p={:?} top_k={:?})", req.temperature, req.top_p, req.top_k);

    // Submit to the batching scheduler and receive tokens on a channel.
    // Use request's penalties if explicitly set, else fall back to server defaults.
    let rep_penalty = req.repetition_penalty.unwrap_or(state.default_rep_penalty);
    let presence_penalty = req.presence_penalty.unwrap_or(state.default_presence_penalty);
    let pp_source = if req.presence_penalty.is_some() { "request" } else { "server-default" };
    let frequency_penalty = req.frequency_penalty.unwrap_or(state.default_frequency_penalty);
    // TOOLEVAL (P18): arm the per-turn tool trace for requests that involve tools - the
    // request offers tools, or its history carries a tool result. None (and no further cost)
    // when the flag is off. The trace only READS the prompt, the raw output and the parse the
    // server already computed: no served byte changes with it on or off.
    let tool_trace = {
        let tool_turn = req.tools.is_some() || req.messages.iter().any(|m| m.role == "tool");
        if !tool_turn { None } else {
            crate::tool_trace::capture(&prompt, serde_json::json!({
                "id": format!("tt-{}", Uuid::new_v4()),
                "ts": chrono::Utc::now().to_rfc3339(),
                "stream": req.stream,
                "seed": req.seed,
                "sampling": {
                    "temperature": temperature, "top_p": top_p, "top_k": top_k,
                    "presence_penalty": presence_penalty, "frequency_penalty": frequency_penalty,
                    "sent": { "temperature": req.temperature, "top_p": req.top_p, "top_k": req.top_k },
                },
                "thinking_on": thinking_on,
                "n_tools": req.tools.as_ref().map_or(0, |t| t.len()),
                "tool_choice": req.tool_choice.clone(),
                "parse_as_tool_calls": parse_as_tool_calls,
                "prompt_tokens": prompt_len,
                "roles": req.messages.iter().map(|m| m.role.as_str()).collect::<Vec<_>>(),
            }))
        }
    };

    let (tx, mut rx, bl) = tok_channel(); // LR-5: `bl` counts consumed events in the loops below
    let request = BatchRequest {
        prompt: prompt_tokens.clone(),
        max_new: req_max,
        temperature,
        received_at: std::time::Instant::now(),
        top_p,
        top_k,
        rep_penalty,
        presence_penalty,
        frequency_penalty,
        min_p: req.min_p.unwrap_or(0.0),
        min_new: req.min_tokens.unwrap_or(0),
        ignore_eos: req.ignore_eos.unwrap_or(false),
        tx,
        seed: req.seed,
        ckpt_at,
        domain: crate::batch::classify_domain(&prompt),
        image_embeds,
        image_spans,
        schema: schema_mask.clone(),
        thinking_budget: req.thinking_token_budget,
        logprobs: logprobs_n,
    };
    if admit_gate_busy() { return server_busy(); } // LR-4: --max-waiting
    // WP02: a DEAD engine or a gone scheduler is a 503, never a request that hangs or reads "length"
    if !engine_ok() || state.scheduler.send(request).is_err() {
        return engine_unavailable();
    }
    let mut mt = crate::metrics::Req::start(prompt_len); // /metrics (recorded when the response ends)

    // SESSION identity for the OTel generation-telemetry (crate::otel::SessionRegistry). One
    // resolution per REQUEST, only when the emitter is on (off = no session work at all):
    // explicit client key first (X-Session-Id header, then metadata.session_id / .conversation_id),
    // else the engine infers the continuous conversation from the messages-prefix rule — turn
    // N+1's messages array contains turn N's as an exact prefix, so a follow-up lands in the
    // SAME session id and the client sees one continuous session, not one per response.
    let otel_session: Option<String> = state.otel.as_ref().map(|s| {
        let explicit = headers.get("x-session-id").and_then(|v| v.to_str().ok())
            .map(str::to_string)
            .or_else(|| req.metadata.as_ref().and_then(|m| m.get("session_id"))
                .and_then(|v| v.as_str()).map(str::to_string))
            .or_else(|| req.metadata.as_ref().and_then(|m| m.get("conversation_id"))
                .and_then(|v| v.as_str()).map(str::to_string));
        // Canonical per-message JSON, in order — the fingerprint material (serialized once).
        let per_msg: Vec<String> = req.messages.iter()
            .map(|m| serde_json::to_string(m).unwrap_or_default()).collect();
        let sess = s.resolve_session(explicit, &per_msg);
        crate::reprintln!("[req] session {sess} (turn of {} message(s))", req.messages.len());
        sess
    });

    // TOOLSCRUB1: the per-request emission-side scrubber (one registry read +, when
    // provenance is on, candidate extraction from this conversation's tool results).
    let scrub = crate::tool_scrub::Scrubber::new(crate::tool_scrub::Mode::from_opt(), &req.messages);
    if req.stream {
        crate::dreprintln!("[req] stream  prompt_tokens={} max_tokens={} stop={:?}", prompt_len, req_max, req.stop);
        let tokenizer = Arc::clone(&state.tokenizer);
        let model_name = req.model.clone().unwrap_or_else(|| state.model_name.clone());
        let stops = req.stop.clone();
        let completion_id = format!("chatcmpl-{}", Uuid::new_v4());
        let created = chrono::Utc::now().timestamp();
        let t0 = std::time::Instant::now();
        let req_tools = req.tools.clone();
        // S-B9-REL-NONETOOLS: the parse decision for this request. `tool_hold` is the
        // <tool_call hold-back needle — None for a keep-tools "none" request, whose stray
        // call streams through as plain content instead of being buffered for a tool_calls delta.
        let tool_hold: Option<&'static str> = if parse_as_tool_calls { Some(TOOL_OPEN) } else { None };
        let include_usage = req.stream_options.as_ref().map(|o| o.include_usage).unwrap_or(true);
        // Owned copy: the SSE generator is 'static, so it cannot capture the `&str` that borrows
        // effort_owned (A3 log line only; the render/ckpt calls above still use `effort`).
        let effort_for_log: Option<String> = effort.map(|s| s.to_string());
        // Think markers + the initial reasoning/content state. Derive the start mode from the
        // RENDERED PROMPT TAIL, not a family constant: qwen's template primes an OPEN think block
        // when thinking (prompt ends with `<think>`), but its no-think branch (enable_thinking=
        // false — effort none/no_think/off) emits a CLOSED empty block `<think>\n\n</think>\n\n`,
        // so that stream starts in CONTENT. A family-constant `primed` mislabeled the direct
        // answer as reasoning_content forever (content stayed empty; the model card's non-thinking
        // mode is real and respected by the model — verified sync+bf16 2026-08-30). hy_v3 keeps
        // the effort arm: its low|high template primes `…assistant<think:opensource>` even when
        // the tail check can't see it (see `stream_starts_in_reasoning`).
        let (think_open, think_close, family_primed) = tokenizer.think_tags();
        // trim_end: the primed form is `<think>\n` — the trailing newline must not defeat the
        // tail check (a plain ends_with(think_open) sent thinking streams to content: exactly
        // the 35b4b15 follow-up bug). The no-think tail `<think>\n\n</think>\n\n` trims to
        // `</think>` and stays content.
        let starts_in_reasoning = stream_starts_in_reasoning(&prompt, think_open, family_primed, effort);

        // OTel generation telemetry (--otel-endpoint). Handle built ONCE per request; the hooks
        // below forward the SAME chunk strings the SSE path yields (single source of truth —
        // the telemetry path never re-derives or re-encodes a delta). `None` (endpoint absent)
        // = the default = every hook below compiles away. The hooks are pure observers: they
        // cannot alter the SSE bytes (see crate::otel — lock-free ring, drop-on-full, the
        // sender runs on its own timer off the compute stream). request.id = the SESSION key
        // (stable across the conversation's turns); generation.id = this POST's execution id.
        let otel_req = match (&state.otel, &otel_session) {
            (Some(s), Some(sess)) =>
                Some(s.open_request(sess, &format!("gen-{}", Uuid::new_v4()))),
            _ => None,
        };

        // S-B9-REL-BENCHFIX2: the stream's generator is `chat_sse_stream` (the unit tests drive that
        // same function); it sends the role chunk when the first token has been generated.
        let cx = ChatSse {
            rx, bl, mt, otel_req, otel_session,
            role_chunk: role_chunk(&completion_id, created, &model_name), // S8: esc(model)
            completion_id, created, model_name, stops,
            think_open, think_close, starts_in_reasoning, tool_hold, parse_as_tool_calls,
            literal_content, req_tools, include_usage, prompt_len, t0,
            tool_trace, scrub,
            next_turn: NextTurn::arm(&state, &req.messages, tools_for_template, effort, req.chat_template_kwargs.as_ref()),
            gen_log: Some(GenLog {
                tokenizer: Arc::clone(&state.tokenizer), stop_ids: state.stop_ids.clone(),
                effort: effort_for_log, presence_penalty, pp_source,
            }),
        };
        let ka_frame = sse_keepalive_frame(&cx.completion_id, cx.created, &cx.model_name, "chat.completion.chunk");
        let stream = chat_sse_stream(cx, move || {
            // Byte-level stream decoder: the per-token decode path above would mangle every
            // multi-byte char split across tokens (all emoji) into "�" — the crate's ByteLevel
            // decode is String::from_utf8_lossy per call. Reassembles raw bytes across tokens.
            let mut dec = tokenizer.stream_decoder();
            move |t| dec.push(t)
        });
        let mut resp = sse_with_keepalive(stream, sse_keepalive(), ka_frame);
        attach_schema_unenforced_header(&mut resp, schema_unenforced.is_some(), schema_mask.is_some());
        resp
    } else {
        crate::reprintln!("[req] sync   prompt_tokens={} max_tokens={} stop={:?}", prompt_len, req_max, req.stop);
        let t0 = std::time::Instant::now();
        let mut tokens = Vec::new();
        // log probabilities (logprobs requested): one entry per token, from the Logprob event
        // the engine sends right before the token
        let mut lps: Vec<TokLp> = Vec::new();
        let mut pending_lp: TokLp = None;
        // WP02: a channel closed without a Finish is an engine failure, not "length"
        let mut finish = "error: engine stopped".to_string();
        let mut first_tok: Option<std::time::Instant> = None;
        let mut cached = 0usize; // v0.7.3 (#8.2): prefix-cache-served prompt tokens
        // [gen] line (item 5): the scheduler's own finish reason, a stop-string hit, the stop-token statistics
        let mut sched_reason: Option<String> = None;
        let mut stop_string_hit = false;
        let mut stop_info: Option<(f32, u32, bool)> = None;
        while let Some(ev) = recv_counted(&mut rx, &bl).await { // LR-5: the one decrement
            match ev {
                TokEvent::Admitted { cached_tokens } => {
                    cached = cached_tokens as usize;
                    mt.cached(cached_tokens as u64); // /metrics counter
                }
                TokEvent::Tok(t) => {
                    tokens.push(t);
                    if logprobs_n.is_some() { lps.push(pending_lp.take()); }
                    mt.tok();
                    if first_tok.is_none() { first_tok = Some(std::time::Instant::now()); }
                    // Apply stop strings LIVE, not just post-hoc: on a hit, break AND let rx drop —
                    // the scheduler sees the closed channel and cancels the lane instead of decoding
                    // to EOS/max_new. Only the tail is searched (a stop string spans a few tokens;
                    // one longer than the window is still honoured post-hoc below, just not early).
                    if !req.stop.is_empty() && tokens.len() % 4 == 0 {
                        let tail = &tokens[tokens.len().saturating_sub(96)..];
                        let s = state.tokenizer.decode(tail, true).unwrap_or_default();
                        if req.stop.iter().any(|x| !x.is_empty() && s.contains(x.as_str())) {
                            finish = "stop".to_string();
                            stop_string_hit = true;
                            break;
                        }
                    }
                }
                TokEvent::StopInfo { p, rank, sampled } => { stop_info = Some((p, rank, sampled)); }
                TokEvent::Logprob { logprob, top } => { pending_lp = Some((logprob, top)); }
                TokEvent::Finish { reason } => { sched_reason = Some(reason.clone()); finish = reason; break; }
            }
        }
        drop(rx); // WP02: release the lane now
        let dt = t0.elapsed().as_secs_f32();
        let mut text = state.tokenizer.decode(&tokens, true).unwrap_or_default();
        if !req.stop.is_empty() {
            if let Some(p) = req.stop.iter().filter(|s| !s.is_empty()).filter_map(|s| text.find(s.as_str())).min() {
                text.truncate(p); finish = "stop".to_string();
                stop_string_hit = true;
            }
        }
        crate::reprintln!("[req] done   tok={} ({:.1} tok/s wall) finish={}", tokens.len(), if dt>1e-6 {tokens.len() as f32/dt} else {0.0}, finish);
        // An engine error is an HTTP error, never a 200 that a client (or a benchmark) would read
        // as an answer. WP02: also when some tokens came first (the text is incomplete).
        if finish.starts_with("error") {
            mt.finish(&finish);
            return (engine_error_status(&finish), Json(serde_json::json!({"error": {
                "message": finish, "type": "server_error", "completion_tokens": tokens.len(),
            }}))).into_response();
        }
        let completion_id = format!("chatcmpl-{}", Uuid::new_v4());
        let (think_open, think_close, family_primed) = state.tokenizer.think_tags();
        // [gen] line (item 5b): did the request end inside the think block its prompt opened?
        // (the same rule as the streaming branch's `starts_in_reasoning`)
        let think_closed = text.contains(think_close);
        let starts_in_reasoning = stream_starts_in_reasoning(&prompt, think_open, family_primed, effort);
        let (reasoning, content) = if literal_content {
            split_constrained(&text, think_close, starts_in_reasoning)
        } else {
            split_think_in(&text, think_open, think_close, starts_in_reasoning)
        };
        // SURFACE2 item 8: the generation ended inside the block (budget exhausted while reasoning,
        // or a stop string hit there) — the text is reasoning, as the streaming branch sends it.
        // The RENDERED PROMPT decides, never the effort arm of `starts_in_reasoning`: with
        // reasoning_effort low|high and thinking switched off by something that outranks it
        // (enable_thinking false, --thinking off), the prompt ends in a CLOSED block and the model
        // answers directly; that answer has no close tag either, and it is content (review finding).
        let prompt_opened_think = prompt.trim_end().ends_with(think_open);
        let (reasoning, content) = if prompt_opened_think && !think_closed && !unclosed_think_as_content() {
            unclosed_think_split(&text, parse_as_tool_calls)
        } else {
            (reasoning, content)
        };

        // The model emits calls as <tool_call><function=..><parameter=..>..  -- NOT as JSON. Turn them
        // into OpenAI tool_calls, or the harness just sees XML in the content and never invokes
        // anything. finish_reason MUST become "tool_calls": that is the flag every harness branches on.
        // The (content, tool_calls, finish) DECISION is crate::tools::finalize_parsed — the one
        // canonical serializer shared with the streaming mode, so the two can never diverge again
        // (2026-08-27 user report: a malformed call block vanished in streaming and leaked in JSON).
        // With tools offered, the model's LITERAL output is the only artifact that settles a "the tool
        // ran but nothing happened" report. Log it when asked (--dump-tools=1), and ALWAYS log
        // it when tools were offered and we parsed nothing — that combination means either the model
        // declined, or it emitted a call we failed to understand, and those need very different fixes.
        // parse_as_tool_calls == false (keep-tools "none"): never parse — the reply is plain
        // content; finalize_parsed then returns the raw text with finish stop/length.
        let parsed = if parse_as_tool_calls {
            crate::tools::parse_served(&content, req.tools.as_deref(), &finish)
        } else {
            crate::tools::ParsedOutput { content: content.trim().to_string(), tool_calls: Vec::new() }
        };
        if req.tools.is_some() {
            let dump = crate::opts::var(crate::opt!("dump-tools")).is_ok();
            if dump || parsed.tool_calls.is_empty() {
                crate::dreprintln!("[req] raw model output ({} chars): {:?}", content.chars().count(),
                          content.chars().take(1200).collect::<String>());
            }
            if dump { crate::reprintln!("{}", dump_tools_line(&content)); }
        }
        // Phase-4 A2: a call to a tool whose schema declares parameters that arrives with EMPTY
        // arguments while the model emitted a NON-EMPTY body is an argument drop, not a quiet
        // success. Log the raw body so the class can never regress unseen again.
        for a in crate::tools::empty_arg_alarms(&content, req.tools.as_deref(), &parsed.tool_calls) {
            crate::reprintln!("[tool-args-alarm] {a}");
        }
        let parser_input = if tool_trace.is_some() { content.clone() } else { String::new() };
        let (content, tool_calls, finish) = crate::tools::finalize_parsed(&content, parsed, &finish);
        // TOOLEVAL: the same per-turn trace as the streaming branch - raw output (thinking
        // included), the parser input, and the finalized verdict the JSON response carries.
        if let Some(t) = &tool_trace {
            crate::tool_trace::log(t, &text, &parser_input, content.as_deref(), &tool_calls, &finish);
        }
        mt.finish(&finish);
        if !tool_calls.is_empty() {
            // Log the ARGUMENTS, not just the names. When opencode reported a write as successful and
            // no file appeared, the log said `tool_calls 1: ["write"]` — which is exactly enough to
            // know a tool was called and not nearly enough to know what it was told to do. The path the
            // model chose is the whole question.
            for t in &tool_calls {
                crate::reprintln!("[req] tool_call  {} {}({})", t.id, t.function.name, t.function.arguments);
            }
        }
        let tool_calls = if tool_calls.is_empty() { None } else { Some(tool_calls) };
        // the decoded-text span `content` came from (reasoning, tool-call markup and trimmed
        // whitespace carry no content logprobs)
        let content_range = logprobs_n
            .and_then(|_| content_byte_range(&text, content.as_deref().unwrap_or(""), think_close));

        // TOOLSCRUB1: emission-side scrub - AFTER the raw-args log above (the server log
        // stays raw, same class as --tool-trace / --dump-tools) and after the trace; only
        // the response body text changes. Off mode = borrow passthrough, no allocation.
        let content = content.map(|c| scrub.scrub_full(&c).into_owned());
        let reasoning = reasoning.map(|r| scrub.scrub_full(&r).into_owned());
        let tool_calls = tool_calls.map(|v| v.into_iter().map(|mut t| {
            t.function.arguments = scrub.scrub_full(&t.function.arguments).into_owned();
            t }).collect::<Vec<_>>());
        // TURNRESUME: the reply as the client will send it back -> the scheduler's background next-turn prefill
        if !finish.starts_with("error") && !stop_string_hit {
            if let Some(nt) = NextTurn::arm(&state, &req.messages, tools_for_template, effort, req.chat_template_kwargs.as_ref()) {
                nt.post(reasoning.clone(), content.clone().unwrap_or_default(), tool_calls.clone().unwrap_or_default());
            }
        }

        let (r_log, c_log) = gen_log_spans(reasoning.as_deref(), content.as_deref().unwrap_or(""),
                                           think_closed, starts_in_reasoning);
        log_generation(&state.tokenizer, &state.stop_ids, &completion_id, &finish,
                       tokens.last().copied(), prompt_len, tokens.len(), r_log, c_log,
                       tool_calls.as_ref().map(|v| v.len()).unwrap_or(0),
                       req.tools.as_ref().map(|t| t.len()).unwrap_or(0),
                       effort, presence_penalty, pp_source,
                       stop_kind(sched_reason.as_deref(), stop_string_hit), stop_info);
        dump_tokens(&completion_id, &tokens);
        let logprobs_json = logprobs_n.map(|_| {
            let Some(range) = content_range.clone() else {
                crate::reprintln!("[req] logprobs: the content is not one span of the generated text; returning null");
                return serde_json::Value::Null;
            };
            chat_logprobs_json(&state.tokenizer, &state.stop_ids, &tokens, &lps, range).unwrap_or_else(|| {
                crate::reprintln!("[req] logprobs: the engine sent no entry for some output token(s); returning null");
                serde_json::Value::Null
            })
        });
        let response = ChatCompletionResponse {
            id: completion_id,
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: req.model.clone().unwrap_or_else(|| state.model_name.clone()),
            choices: vec![ChatChoice {
                index: 0,
                message: ResponseMessage {
                    role: "assistant".to_string(), content,
                    reasoning_content: reasoning, tool_calls,
                },
                finish_reason: spec_finish_reason(&finish).to_string(),
                stop_reason: stop_reason_of(&finish),
                logprobs: logprobs_json,
            }],
            usage: Usage {
                prompt_tokens: prompt_len,
                completion_tokens: tokens.len(),
                total_tokens: prompt_len + tokens.len(),
                prompt_tokens_details: PromptTokensDetails { cached_tokens: cached },
            },
            timings: make_timings(t0, first_tok, prompt_len, tokens.len()),
            session_id: otel_session,
        };
        let mut resp = Json(response).into_response();
        attach_schema_unenforced_header(&mut resp, schema_unenforced.is_some(), schema_mask.is_some());
        resp
    }
}
