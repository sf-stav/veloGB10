//! `POST /v1/completions`: the `completions` handler. The `CompletionRequest` type lives in `mod.rs`.
//! Siblings: `engine.rs`, `logprobs.rs`, `sse.rs` provide the shared pieces; `mod.rs` has the map.

use super::*;

pub(super) async fn completions(
    State(state): State<AppState>,
    payload: Result<Json<CompletionRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(req) = match payload { Ok(j) => j, Err(e) => return bad_json(e) };
    if let Err(msg) = validate_penalties(req.repetition_penalty, req.presence_penalty,
                                         req.frequency_penalty, req.min_p) {
        return bad_param(msg);
    }
    let logprobs_n = match completion_logprobs_param(req.logprobs) {
        Ok(n) => logprobs_gate(n, &state, req.stream),
        Err(msg) => return bad_param(msg),
    };
    let prompt_tokens: Vec<u32> = match req.prompt.as_ref() {
        Some(serde_json::Value::String(t)) => match encode_off_thread(&state.tokenizer, t, true).await {
            Ok(v) => v,
            Err(e) => return (StatusCode::BAD_REQUEST, format!("tokenize failed: {e}")).into_response(),
        },
        // ISSUES 129: every element must be an integer in [0, vocab) — no element dropped, none truncated to u32
        // (an out-of-range id used to reach the GPU embedding gather unchecked)
        Some(serde_json::Value::Array(a)) if !a.is_empty() => match prompt_token_ids(a, state.tokenizer.vocab_size()) {
            Ok(v) => v,
            Err(e) => return (StatusCode::BAD_REQUEST, e).into_response(),
        },
        _ => return (StatusCode::BAD_REQUEST,
                     "prompt must be a string or a non-empty token-id array".to_string()).into_response(),
    };
    let prompt_len = prompt_tokens.len();
    // WP02: the same context budget as the chat path — the decode headroom is reserved (this cap
    // ignored it and ran 3 rows past the KV window at the end of an uncapped request)
    let room = generation_room(state.max_seq_len, prompt_len, state.decode_headroom);
    if prompt_len + 8 >= state.max_seq_len || room.is_none() {
        return (StatusCode::BAD_REQUEST, format!(
            "prompt {} tokens leaves no room within max_seq_len {}", prompt_len, state.max_seq_len)).into_response();
    }
    let req_max = req.max_tokens.unwrap_or(16).min(room.unwrap_or(0));
    let (tx, mut rx, bl) = tok_channel(); // LR-5: `bl` counts consumed events in the loops below
    let request = BatchRequest {
        prompt: prompt_tokens.clone(),
        max_new: req_max,
        temperature: req.temperature.unwrap_or_else(default_temperature),
        top_p: req.top_p.unwrap_or_else(default_top_p),
        top_k: req.top_k.unwrap_or_else(default_top_k),
        rep_penalty: req.repetition_penalty.unwrap_or(1.0),
        presence_penalty: req.presence_penalty.unwrap_or(0.0),
        frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
        min_p: req.min_p.unwrap_or(0.0),
        min_new: req.min_tokens.unwrap_or(0).min(req_max),
        ignore_eos: req.ignore_eos.unwrap_or(false),
        tx,
        seed: req.seed,
        ckpt_at: None,
        domain: crate::batch::Domain::General,
        received_at: std::time::Instant::now(),
        image_embeds: None,
        image_spans: Vec::new(),
        schema: None,
        thinking_budget: req.thinking_token_budget,
        logprobs: logprobs_n,
    };
    if admit_gate_busy() { return server_busy(); } // LR-4: --max-waiting
    let (_mn, _ie) = (request.min_new, request.ignore_eos);
    if !engine_ok() || state.scheduler.send(request).is_err() {
        return engine_unavailable(); // WP02
    }
    let mut mt = crate::metrics::Req::start(prompt_len); // /metrics
    crate::reprintln!("[req] completions prompt_tokens={} max_tokens={} min_tokens={} ignore_eos={} stream={}",
              prompt_len, req_max, _mn, _ie, req.stream);
    let model_name = req.model.clone().unwrap_or_else(|| state.model_name.clone());
    let cid = format!("cmpl-{}", uuid::Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();

    if req.stream {
        let t0 = std::time::Instant::now();
        let include_usage = completions_include_usage(req.stream_options.as_ref()); // item 6
        // LANERES1: the keep-alive frame shape (comment or empty-choices data frame) is decided
        // before the generator below moves cid/created/model_name into the stream.
        let ka_frame = sse_keepalive_frame(&cid, created, &model_name, "text_completion.chunk");
        let stream = async_stream::stream! {
            let mut ntok: usize = 0;
            let mut cached = 0usize; // #8.2: prefix-cache-served prompt tokens (the usage chunk)
            let mut first_tok: Option<std::time::Instant> = None;
            // WP02: closed without a Finish = engine failure; errors are an SSE error event + "stop"
            let mut finish = "error: engine stopped".to_string();
            while let Some(ev) = recv_counted(&mut rx, &bl).await { // LR-5: the one decrement (H0)
                match ev {
                    // v0.7.3 (#8.2): counted for /metrics, and reported in the usage chunk
                    // when the request asked for one (stream_options.include_usage).
                    TokEvent::Admitted { cached_tokens } => {
                        cached = cached_tokens as usize;
                        mt.cached(cached_tokens as u64);
                    }
                    TokEvent::StopInfo { .. } => {} // [gen]-line data; this path prints no [gen] line
                    TokEvent::Logprob { .. } => {} // refused with stream=true (completion_logprobs_param)
                    TokEvent::Tok(t) => {
                        if first_tok.is_none() { first_tok = Some(std::time::Instant::now()); }
                        ntok += 1;
                        mt.tok();
                        let text = state.tokenizer.decode(&[t], true).unwrap_or_default();
                        let chunk = serde_json::json!({
                            "id": cid, "object": "text_completion.chunk", "created": created,
                            "model": model_name,
                            "choices": [{"index": 0, "text": text, "finish_reason": null}],
                        });
                        yield Ok::<_, std::convert::Infallible>(Event::default().data(chunk.to_string()));
                    }
                    TokEvent::Finish { reason } => { finish = reason; break; }
                }
            }
            drop(rx);
            if finish.starts_with("error") {
                crate::reprintln!("[req] completions stream ended in an engine error after {ntok} tokens: {finish}");
                yield Ok::<_, std::convert::Infallible>(Event::default().data(sse_error_event(&finish)));
            }
            mt.finish(&finish);
            let fr = if finish == "length" { "length" } else { "stop" };
            let mut chunk = serde_json::json!({
                "id": cid, "object": "text_completion.chunk", "created": created,
                "model": model_name,
                "choices": [{"index": 0, "text": "", "finish_reason": fr}],
            });
            if let Some(r) = stop_reason_of(&finish) {
                chunk["choices"][0]["stop_reason"] = serde_json::json!(r); // WP08
            }
            yield Ok::<_, std::convert::Infallible>(Event::default().data(chunk.to_string()));
            if include_usage {
                let u = completions_usage_chunk(&cid, created, &model_name, prompt_len, ntok, cached,
                                                make_timings(t0, first_tok, prompt_len, ntok));
                yield Ok::<_, std::convert::Infallible>(Event::default().data(u.to_string()));
            }
            yield Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"));
            let dt = t0.elapsed().as_secs_f32();
            crate::reprintln!("[req] done   completions tok={} ({:.1} tok/s wall) finish={}", ntok, if dt>1e-6 {ntok as f32/dt} else {0.0}, fr);
        };
        // S-B9-REL-BENCHFIX: this stream never sent anything before its first token; it gets the
        // same idle comment frames as the chat stream (queue wait / long prefill).
        return sse_with_keepalive(stream, sse_keepalive(), ka_frame);
    }
    // non-streaming: collect everything, detokenize once, single response.
    let mut toks: Vec<u32> = Vec::with_capacity(req_max);
    let mut lps: Vec<TokLp> = Vec::new();
    let mut pending_lp: TokLp = None;
    // WP02: an engine error (or a channel closed without a Finish) is an HTTP error, not 200 "stop"
    let mut finish = "error: engine stopped".to_string();
    let mut stop_reason: Option<String> = None;
    let mut cached = 0usize; // v0.7.3 (#8.2): prefix-cache-served prompt tokens
    while let Some(ev) = recv_counted(&mut rx, &bl).await { // LR-5: the one decrement (H2: this loop never decremented)
        match ev {
            TokEvent::Admitted { cached_tokens } => {
                cached = cached_tokens as usize;
                mt.cached(cached_tokens as u64); // /metrics counter
            }
            TokEvent::StopInfo { .. } => {} // [gen]-line data; this path prints no [gen] line
            TokEvent::Logprob { logprob, top } => { pending_lp = Some((logprob, top)); }
            TokEvent::Tok(t) => { toks.push(t); lps.push(pending_lp.take()); mt.tok(); }
            TokEvent::Finish { reason } => {
                stop_reason = stop_reason_of(&reason); // WP08
                finish = reason;
                break;
            }
        }
    }
    drop(rx);
    if finish.starts_with("error") {
        mt.finish(&finish);
        return (engine_error_status(&finish), Json(serde_json::json!({"error": {
            "message": finish, "type": "server_error", "completion_tokens": toks.len(),
        }}))).into_response();
    }
    mt.finish(&finish);
    let finish = if finish == "length" { "length".to_string() } else { "stop".to_string() };
    let text = state.tokenizer.decode(&toks, true).unwrap_or_default();
    dump_tokens(&cid, &toks);
    let mut json = serde_json::json!({
        "id": cid, "object": "text_completion", "created": created, "model": model_name,
        "choices": [{"index": 0, "text": text, "finish_reason": finish, "logprobs": null}],
        "usage": {"prompt_tokens": prompt_len, "completion_tokens": toks.len(),
                  "total_tokens": prompt_len + toks.len(),
                  "prompt_tokens_details": {"cached_tokens": cached}},
    });
    if let Some(r) = stop_reason {
        json["choices"][0]["stop_reason"] = serde_json::json!(r); // WP08
    }
    if logprobs_n.is_some() {
        json["choices"][0]["logprobs"] = completion_logprobs_json(&state.tokenizer, &state.stop_ids, &toks, &lps, &text)
            .unwrap_or_else(|| {
                crate::reprintln!("[req] logprobs: the engine sent no entry for some output token(s); returning null");
                serde_json::Value::Null
            });
    }
    (StatusCode::OK, axum::Json(json)).into_response()
}

