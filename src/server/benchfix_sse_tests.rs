//! Inline unit tests of the server module, moved out of `server.rs` verbatim (module `benchfix_sse_tests`).
//! Declared from `mod.rs`; the body keeps its original indentation and its `use super::*` imports.
    use super::*;
    use crate::metrics::SchedState;
    use std::time::Duration;
    use tokio::time::Instant;

    fn idle() -> SchedState { SchedState { busy: false, work_age: None } }

    const CID: &str = "chatcmpl-benchfix";
    const MODEL: &str = "Qwen/Qwen3.8-Flash-Next";
    const KA: &str = ": keep-alive\n\n";

    /// The test vocabulary: token id = index.
    const PIECES: &[&str] = &[
        "Hello",                 // 0
        " world",                // 1
        "",                      // 2  a token that decodes to no text (a special token, half a UTF-8 char)
        "plan",                  // 3
        "</think>",              // 4
        "\n\n",                  // 5
        "<tool_call>",           // 6
        "\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n", // 7
        "</tool_call>",          // 8
        " oops",                 // 9
        "<",                     // 10 could become "<tool_call" or "<think>": held back
        "</thi",                 // 11 could become "</think>": held back inside a think block
        "<tool_call_x y",        // 12 holds the marker "<tool_call" but is not a call tag
        " and <think>why</think>\n\nso", // 13 a think block that opens and closes inside one piece
    ];

    fn secs(n: u64) -> Duration { Duration::from_secs(n) }
    fn tok(t: u32) -> TokEvent { TokEvent::Tok(t) }
    fn fin(reason: &str) -> TokEvent { TokEvent::Finish { reason: reason.into() } }

    #[derive(Clone, Copy)]
    struct Shape { thinking: bool, tools: bool, usage: bool }
    const PLAIN: Shape = Shape { thinking: false, tools: false, usage: false };

    /// The generator's inputs, as the handler builds them (no tokenizer: no `[gen]` log line).
    fn cx(rx: mpsc::UnboundedReceiver<TokEvent>, bl: TokBacklog, sh: Shape, otel_req: Option<crate::otel::OtelRequest>) -> ChatSse {
        let weather = serde_json::json!({"type": "function", "function": {"name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}});
        ChatSse {
            rx, bl, mt: crate::metrics::Req::start(7), otel_req, otel_session: None,
            role_chunk: role_json(), completion_id: CID.into(), created: 1, model_name: MODEL.into(),
            stops: Vec::new(), think_open: "<think>", think_close: "</think>",
            starts_in_reasoning: sh.thinking,
            tool_hold: sh.tools.then_some(TOOL_OPEN), parse_as_tool_calls: sh.tools,
            literal_content: false,
            req_tools: sh.tools.then(|| vec![weather]),
            include_usage: sh.usage, prompt_len: 7, t0: std::time::Instant::now(), gen_log: None,
            tool_trace: None,
            scrub: crate::tool_scrub::Scrubber::new(crate::tool_scrub::Mode::Off, &[]),
            next_turn: None,
        }
    }

    /// The real generator over a scripted channel.
    fn chat(rx: mpsc::UnboundedReceiver<TokEvent>, bl: TokBacklog, sh: Shape)
        -> impl Stream<Item = Result<Event, axum::Error>> + Send + 'static {
        chat_sse_stream(cx(rx, bl, sh, None), || |t: u32| PIECES[t as usize].to_string())
    }

    /// The real generator with an OTel request handle (the telemetry test in otel.rs).
    pub(crate) fn chat_with_otel(rx: mpsc::UnboundedReceiver<TokEvent>, bl: TokBacklog, req: crate::otel::OtelRequest)
        -> impl Stream<Item = Result<Event, axum::Error>> + Send + 'static {
        chat_sse_stream(cx(rx, bl, PLAIN, Some(req)), || |t: u32| PIECES[t as usize].to_string())
    }

    /// The server's response builder with explicit scheduler gauges (`sse_with_keepalive` passes
    /// `metrics::sched_state`). `idle` = the scheduler has completed nothing and is not busy, so
    /// only the stream's own age counts.
    fn respond<S, E, F>(stream: S, every: Option<Duration>, sched: F) -> Response
    where
        S: Stream<Item = Result<Event, E>> + Send + 'static,
        E: Into<axum::BoxError> + Send + 'static,
        F: Fn() -> SchedState + Send + 'static,
    {
        match every {
            Some(d) => Sse::new(keepalive_frames(stream, d, SseKeepAliveFrame::Comment, SSE_KEEPALIVE_STALL, sched)).into_response(),
            None => Sse::new(stream).into_response(),
        }
    }

    /// Send each event at its time (seconds on the paused clock, from now; an Admitted event
    /// goes out at once, like the scheduler's). The sender is dropped after the last event.
    fn feed(tx: TokTx, script: Vec<(u64, TokEvent)>) {
        let t0 = Instant::now();
        tokio::spawn(async move {
            let _ = tx.send(TokEvent::Admitted { cached_tokens: 0 });
            for (at, ev) in script {
                tokio::time::sleep_until(t0 + secs(at)).await;
                let _ = tx.send(ev);
            }
        });
    }

    /// One wire frame: when it arrived (paused clock) and its raw bytes (one frame per event).
    struct Frame { at: Duration, raw: String }

    impl Frame {
        fn is_comment(&self) -> bool { self.raw.starts_with(':') }
        /// The `data:` payload of a data frame (None for a comment frame).
        fn data(&self) -> Option<&str> {
            self.raw.strip_prefix("data: ").map(|s| s.trim_end_matches('\n'))
        }
        fn json(&self) -> Option<serde_json::Value> {
            self.data().filter(|d| *d != "[DONE]").map(|d| serde_json::from_str(d)
                .unwrap_or_else(|e| panic!("data frame is not JSON ({e}): {d}")))
        }
        fn has_choices(&self) -> bool {
            self.json().map_or(false, |j| j["choices"].as_array().map_or(false, |a| !a.is_empty()))
        }
    }

    /// Every frame of the response body until it ends (or, for a stream that never ends, until
    /// `until` on the paused clock).
    async fn read_frames_until(resp: Response, until: Option<Duration>) -> Vec<Frame> {
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(resp.headers()["content-type"], "text/event-stream");
        let t0 = Instant::now();
        let mut body = resp.into_body().into_data_stream();
        let mut out = Vec::new();
        loop {
            let next = match until {
                Some(u) => match tokio::time::timeout_at(t0 + u, body.next()).await { Ok(n) => n, Err(_) => break },
                None => body.next().await,
            };
            let Some(chunk) = next else { break };
            let raw = String::from_utf8(chunk.expect("body frame").to_vec()).expect("utf-8");
            assert!(raw.ends_with("\n\n"), "every SSE frame ends with a blank line: {raw:?}");
            out.push(Frame { at: t0.elapsed(), raw });
        }
        out
    }

    async fn read_frames(resp: Response) -> Vec<Frame> { read_frames_until(resp, None).await }

    fn data_frames(frames: &[Frame]) -> Vec<String> {
        frames.iter().filter_map(|f| f.data().map(str::to_string)).collect()
    }

    /// (seconds, "ka" | "role" | "data") per frame: the timeline of a response.
    fn timeline(frames: &[Frame]) -> Vec<(u64, &'static str)> {
        frames.iter().map(|f| {
            assert_eq!(f.at.subsec_nanos(), 0, "frames land on whole seconds of the paused clock: {:?}", f.at);
            (f.at.as_secs(), if f.is_comment() { "ka" } else if f.data() == Some(role_json().as_str()) { "role" } else { "data" })
        }).collect()
    }

    pub(crate) fn role_json() -> String { role_chunk(CID, 1, MODEL) }
    fn content(t: &str) -> String { content_chunk(CID, 1, MODEL, t) }
    fn reasoning(t: &str) -> String { reasoning_chunk(CID, 1, MODEL, t) }
    fn finish(reason: &str) -> String { final_chunk(CID, 1, MODEL, reason, "") }

    /// THE defect of post #46: with a first token after 35 s (queue wait + prefill) nothing with
    /// a `choices` entry is on the wire before it; the wait carries comment frames only, one per
    /// 10 s; the role chunk arrives with the first token and before its content.
    #[tokio::test(start_paused = true)]
    async fn no_choices_chunk_before_the_first_generated_token() {
        let (tx, rx, bl) = tok_channel();
        feed(tx, vec![(35, tok(0)), (36, tok(1)), (37, fin("stop"))]);
        let sh = Shape { usage: true, ..PLAIN };
        let frames = read_frames(respond(chat(rx, bl, sh), Some(secs(10)), idle)).await;

        assert_eq!(timeline(&frames), [(10, "ka"), (20, "ka"), (30, "ka"), (35, "role"), (35, "data"), (36, "data"),
                                       (37, "data"), (37, "data"), (37, "data")]);
        for f in &frames[..3] {
            assert_eq!(f.raw, KA, "a waiting frame is the comment frame, nothing else");
            assert!(!f.raw.contains('{') && !f.raw.contains("data:") && !f.raw.contains("choices"),
                    "a comment frame carries no JSON and no choices");
        }
        // the first frame with a `choices` entry is the role chunk
        let first_choices = frames.iter().position(|f| f.has_choices()).unwrap();
        assert_eq!(first_choices, 3);
        assert_eq!(frames[3].json().unwrap()["choices"][0]["delta"], serde_json::json!({"role": "assistant"}));
        let d = data_frames(&frames);
        assert_eq!(d[..4], [role_json(), content("Hello"), content(" world"), finish("stop")]);
        let usage = frames[7].json().unwrap();
        assert!(usage["choices"].as_array().unwrap().is_empty(), "the usage chunk has empty choices");
        assert_eq!(usage["usage"]["prompt_tokens"], 7);
        assert_eq!(usage["usage"]["completion_tokens"], 2);
        assert_eq!(d[5], "[DONE]");
        // every data frame is one line of valid JSON (or the terminator)
        for f in frames.iter().filter(|f| !f.is_comment()) {
            let p = f.data().unwrap();
            assert!(!p.contains('\n') && (p == "[DONE]" || serde_json::from_str::<serde_json::Value>(p).is_ok()));
        }
    }

    /// W2 (review m3): thinking off, the reply is ONLY a tool call. The call is buffered, so
    /// the generator yields no delta until the end; the role chunk still leaves at the first
    /// generated token (35 s), not with the tool_calls delta (51 s). The silence while the call
    /// is buffered carries a comment frame.
    #[tokio::test(start_paused = true)]
    async fn a_held_tool_call_sends_the_role_chunk_at_the_first_generated_token() {
        let (tx, rx, bl) = tok_channel();
        feed(tx, vec![(35, tok(6)), (40, tok(7)), (50, tok(8)), (51, fin("stop"))]);
        let sh = Shape { tools: true, ..PLAIN };
        let frames = read_frames(respond(chat(rx, bl, sh), Some(secs(10)), idle)).await;
        assert_eq!(timeline(&frames), [(10, "ka"), (20, "ka"), (30, "ka"), (35, "role"), (45, "ka"),
                                       (51, "data"), (51, "data"), (51, "data")]);
        let j: Vec<serde_json::Value> = frames.iter().filter_map(|f| f.json()).collect();
        assert_eq!(j.len(), 3);
        assert_eq!(j[0]["choices"][0]["delta"], serde_json::json!({"role": "assistant"}));
        let call = &j[1]["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(call["function"]["name"], "get_weather");
        assert_eq!(serde_json::from_str::<serde_json::Value>(call["function"]["arguments"].as_str().unwrap()).unwrap(),
                   serde_json::json!({"city": "Paris"}));
        assert_eq!(j[2]["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(frames.last().unwrap().data(), Some("[DONE]"));
    }

    /// The same rule for every way a first token can produce no delta: a token without text,
    /// and a held-back marker prefix. The role chunk is at the first token; the first delta is
    /// later.
    #[tokio::test(start_paused = true)]
    async fn the_role_chunk_does_not_wait_for_the_first_delta() {
        // a first token that decodes to nothing
        let (tx, rx, bl) = tok_channel();
        feed(tx, vec![(5, tok(2)), (9, tok(0)), (10, fin("stop"))]);
        let frames = read_frames(respond(chat(rx, bl, PLAIN), None, idle)).await;
        assert_eq!(timeline(&frames), [(5, "role"), (9, "data"), (10, "data"), (10, "data")]);
        assert_eq!(data_frames(&frames), [role_json(), content("Hello"), finish("stop"), "[DONE]".to_string()]);
        // a first token that is held back as a possible marker prefix, in reasoning mode too
        for thinking in [false, true] {
            let (tx, rx, bl) = tok_channel();
            feed(tx, vec![(5, tok(10)), (9, tok(9)), (10, fin("length"))]);
            let sh = Shape { thinking, tools: true, usage: false };
            let frames = read_frames(respond(chat(rx, bl, sh), None, idle)).await;
            assert_eq!(timeline(&frames), [(5, "role"), (9, "data"), (10, "data"), (10, "data")], "thinking {thinking}");
            let text = if thinking { reasoning("< oops") } else { content("< oops") };
            assert_eq!(data_frames(&frames), [role_json(), text, finish("length"), "[DONE]".to_string()]);
        }
    }

    /// The data events are the old stream's: role chunk first, then the deltas, the finish
    /// chunk and the terminator, each with the bytes of its chunk builder.
    #[tokio::test(start_paused = true)]
    async fn the_data_events_keep_their_bytes_and_order() {
        // reasoning, the close tag, then content
        let (tx, rx, bl) = tok_channel();
        feed(tx, vec![(3, tok(3)), (4, tok(4)), (5, tok(5)), (6, tok(0)), (7, tok(1)), (8, fin("length"))]);
        let sh = Shape { thinking: true, ..PLAIN };
        let frames = read_frames(respond(chat(rx, bl, sh), None, idle)).await;
        // (white space after the close tag is dropped only when it arrives in the same decode
        // chunk as the tag: the "\n\n" token here is a content delta, as it always was)
        assert_eq!(data_frames(&frames), [role_json(), reasoning("plan"), content("\n\n"), content("Hello"),
                                          content(" world"), finish("length"), "[DONE]".to_string()]);
        assert_eq!(timeline(&frames)[..2], [(3, "role"), (3, "data")], "role and the first reasoning delta together");
        // reasoning, then a tool call (the shape of the banked GPU gate `tool_call`)
        let (tx, rx, bl) = tok_channel();
        feed(tx, vec![(3, tok(3)), (4, tok(4)), (5, tok(6)), (6, tok(7)), (7, tok(8)), (8, fin("stop"))]);
        let sh = Shape { thinking: true, tools: true, usage: false };
        let frames = read_frames(respond(chat(rx, bl, sh), None, idle)).await;
        let d = data_frames(&frames);
        assert_eq!(d.len(), 5, "{d:?}");
        assert_eq!(d[..2], [role_json(), reasoning("plan")]);
        let tc: serde_json::Value = serde_json::from_str(&d[2]).unwrap();
        assert_eq!(tc["choices"][0]["delta"]["tool_calls"][0]["function"]["name"], "get_weather");
        assert_eq!(d[3..], [finish("tool_calls"), "[DONE]".to_string()]);
        // a block that looked like a call but did not parse is surfaced as content at the end
        let (tx, rx, bl) = tok_channel();
        feed(tx, vec![(3, tok(6)), (4, tok(9)), (5, fin("stop"))]);
        let sh = Shape { tools: true, ..PLAIN };
        let frames = read_frames(respond(chat(rx, bl, sh), None, idle)).await;
        assert_eq!(data_frames(&frames), [role_json(), content("<tool_call> oops"), finish("stop"), "[DONE]".to_string()]);
        assert_eq!(timeline(&frames)[..2], [(3, "role"), (5, "data")]);
    }

    /// A request that generates NO token sends the role chunk immediately before its terminal
    /// events and nothing while it waits: a finish without a token, an engine error before the
    /// first token (the error event has no `choices`), and an engine that went away.
    #[tokio::test(start_paused = true)]
    async fn a_request_without_a_token_sends_the_role_chunk_before_its_terminal_events() {
        let (tx, rx, bl) = tok_channel();
        feed(tx, vec![(25, fin("stop"))]);
        let frames = read_frames(respond(chat(rx, bl, PLAIN), Some(secs(10)), idle)).await;
        assert_eq!(timeline(&frames), [(10, "ka"), (20, "ka"), (25, "role"), (25, "data"), (25, "data")]);
        assert_eq!(data_frames(&frames), [role_json(), finish("stop"), "[DONE]".to_string()]);

        let (tx, rx, bl) = tok_channel();
        feed(tx, vec![(8, fin("error: prefill failed: boom"))]);
        let frames = read_frames(respond(chat(rx, bl, PLAIN), None, idle)).await;
        assert_eq!(timeline(&frames), [(8, "role"), (8, "data"), (8, "data"), (8, "data")]);
        let d = data_frames(&frames);
        let err: serde_json::Value = serde_json::from_str(&d[1]).unwrap();
        assert_eq!(err["error"]["message"], "error: prefill failed: boom");
        assert_eq!(err["error"]["code"], 500);
        assert!(err.get("choices").is_none());
        assert_eq!(d[2..], [finish("stop"), "[DONE]".to_string()]);

        // the scheduler went away without a Finish (the sender is dropped at 4 s): never a hang
        let (tx, rx, bl) = tok_channel();
        feed(tx, vec![(4, TokEvent::Admitted { cached_tokens: 0 })]);
        let frames = read_frames(respond(chat(rx, bl, PLAIN), None, idle)).await;
        let d = data_frames(&frames);
        assert_eq!(d[0], role_json());
        assert!(d[1].contains("engine stopped") && frames[0].at == secs(4));
    }

    /// S-B9-REL-INTEG B5: after an in-band engine error event, nothing but the finish chunk follows, unless the
    /// held span holds a COMPLETE call (the rule before SURFACE2B's held-tail flush; the flush now runs only on a
    /// finish that is not an error).
    #[tokio::test(start_paused = true)]
    async fn nothing_follows_an_engine_error_but_the_finish_chunk() {
        // text with a held tail ("<" could become a marker), then the engine fails
        let (tx, rx, bl) = tok_channel();
        feed(tx, vec![(1, tok(0)), (2, tok(10)), (3, fin("error: engine broke"))]);
        let d = data_frames(&read_frames(respond(chat(rx, bl, Shape { tools: true, ..PLAIN }), None, idle)).await);
        assert_eq!(d.len(), 5, "{d:?}");
        assert_eq!(d[..2], [role_json(), content("Hello")]);
        let err: serde_json::Value = serde_json::from_str(&d[2]).unwrap();
        assert_eq!(err["error"]["message"], "error: engine broke");
        assert_eq!(d[3..], [finish("stop"), "[DONE]".to_string()], "the held `<` must not follow the error");
        // a COMPLETE call in the held span still becomes its tool_calls delta (master's rule)
        let (tx, rx, bl) = tok_channel();
        feed(tx, vec![(1, tok(6)), (2, tok(7)), (3, tok(8)), (4, fin("error: engine broke"))]);
        let d = data_frames(&read_frames(respond(chat(rx, bl, Shape { tools: true, ..PLAIN }), None, idle)).await);
        assert_eq!(d.len(), 5, "{d:?}");
        assert!(d[1].contains("\"error\""), "{}", d[1]);
        assert!(d[2].contains("\"tool_calls\"") && d[2].contains("get_weather"), "{}", d[2]);
        assert_eq!(d[3..], [finish("tool_calls"), "[DONE]".to_string()]);
    }

    /// S-B9-REL-SURFACE2B R5: a reply that ENDS in text the hold-back was keeping. The emit sites
    /// hold the tail of the buffer while it could still become a marker; when the generation
    /// ended there, that tail was never sent: a streamed reply ending in `<` (or cut by
    /// max_tokens right after one) lost it, where the non-streaming response keeps it. Every
    /// case: the deltas of the stream add up to the generated text.
    #[tokio::test(start_paused = true)]
    async fn a_held_tail_leaves_at_the_finish() {
        async fn run(sh: Shape, toks: &[u32], reason: &str) -> Vec<String> {
            let (tx, rx, bl) = tok_channel();
            let mut script: Vec<(u64, TokEvent)> = toks.iter().enumerate().map(|(i, t)| (i as u64 + 1, tok(*t))).collect();
            script.push((toks.len() as u64 + 1, fin(reason)));
            feed(tx, script);
            let d = data_frames(&read_frames(respond(chat(rx, bl, sh), None, idle)).await);
            // nothing is lost and nothing is sent twice: reasoning + content deltas = the pieces
            // (a closed think block's tag is the one thing a stream never sends)
            let sent: String = d.iter().filter_map(|x| serde_json::from_str::<serde_json::Value>(x).ok()).map(|j| {
                let delta = &j["choices"][0]["delta"];
                format!("{}{}", delta["reasoning_content"].as_str().unwrap_or(""), delta["content"].as_str().unwrap_or(""))
            }).collect();
            let made: String = toks.iter().map(|t| PIECES[*t as usize]).collect();
            assert_eq!(sent, made.replace("</think>", ""), "{toks:?}");
            d
        }
        let end = |reason: &str| [finish(reason), "[DONE]".to_string()];
        let sh = |thinking, tools| Shape { thinking, tools, usage: false };
        for tools in [false, true] {
            // content: "<" may start "<think>" (always watched) or "<tool_call" (with tools)
            let d = run(sh(false, tools), &[0, 10], "length").await;
            assert_eq!(d, [vec![role_json(), content("Hello"), content("<")], end("length").to_vec()].concat(), "tools {tools}");
            let d = run(sh(false, tools), &[10], "stop").await;
            assert_eq!(d, [vec![role_json(), content("<")], end("stop").to_vec()].concat(), "a reply that is only the held character");
            // inside a think block that never closes, the tail is reasoning: "<" and "</thi"
            for held in [10u32, 11] {
                let d = run(sh(true, tools), &[3, held], "length").await;
                assert_eq!(d, [vec![role_json(), reasoning("plan"), reasoning(PIECES[held as usize])], end("length").to_vec()].concat(), "tools {tools}");
            }
            // after the block closed it is content again
            let d = run(sh(true, tools), &[3, 4, 0, 10], "length").await;
            assert_eq!(d, [vec![role_json(), reasoning("plan"), content("Hello"), content("<")], end("length").to_vec()].concat());
        }
        // text that holds the marker "<tool_call" but is not a call: held as a possible call,
        // not parsed, so it is content (inside an unclosed think block too: a call-looking block
        // the parser cannot read stays visible as content)
        let d = run(sh(false, true), &[0, 12], "stop").await;
        assert_eq!(d, [vec![role_json(), content("Hello"), content("<tool_call_x y")], end("stop").to_vec()].concat());
        let d = run(sh(true, true), &[3, 12, 9], "length").await;
        assert_eq!(d, [vec![role_json(), reasoning("plan"), content("<tool_call_x y oops")], end("length").to_vec()].concat());
        // without tools nothing watches "<tool_call": it streams as it comes
        let d = run(sh(false, false), &[0, 12], "stop").await;
        assert_eq!(d, [vec![role_json(), content("Hello"), content("<tool_call_x y")], end("stop").to_vec()].concat());
        // a think block that opens and closes inside the reply's LAST piece: reasoning, then
        // content, no raw tag (review 2026-10-10: the flush sent `why</think>\n\nso` as reasoning)
        let (tx, rx, bl) = tok_channel();
        feed(tx, vec![(1, tok(0)), (2, tok(13)), (3, fin("stop"))]);
        let d = data_frames(&read_frames(respond(chat(rx, bl, sh(false, false)), None, idle)).await);
        assert_eq!(d, [vec![role_json(), content("Hello"), content(" and "), reasoning("why"), content("so")], end("stop").to_vec()].concat());
        // a parsed call is still the tool_calls delta alone: its text is not sent as content too
        let (tx, rx, bl) = tok_channel();
        feed(tx, vec![(1, tok(0)), (2, tok(6)), (3, tok(7)), (4, tok(8)), (5, fin("stop"))]);
        let d = data_frames(&read_frames(respond(chat(rx, bl, sh(false, true)), None, idle)).await);
        assert_eq!(d.len(), 5, "{d:?}");
        assert_eq!(d[..2], [role_json(), content("Hello")]);
        assert!(d[2].contains("\"tool_calls\"") && !d[2].contains("\"content\""), "{}", d[2]);
        assert_eq!(d[3..], [finish("tool_calls"), "[DONE]".to_string()]);
    }

    /// A stop string ends the reply BEFORE itself, in a stream as in the non-streaming response
    /// (S-B9-REL-SURFACE2B, found while fixing R5). The stream used to check for a stop string
    /// after it had sent the token's text: a streamed reply carried its stop string, the rest of
    /// the token that completed it, and the start of a stop string that spanned tokens. Every
    /// case states the text the stream delivers; it is the text up to the stop string.
    #[tokio::test(start_paused = true)]
    async fn a_stop_string_is_not_part_of_the_stream() {
        async fn run(sh: Shape, stop: &str, toks: &[u32], reason: &str) -> Vec<String> {
            let (tx, rx, bl) = tok_channel();
            let mut script: Vec<(u64, TokEvent)> = toks.iter().enumerate().map(|(i, t)| (i as u64 + 1, tok(*t))).collect();
            script.push((toks.len() as u64 + 1, fin(reason)));
            feed(tx, script);
            let mut c = cx(rx, bl, sh, None);
            c.stops = vec![stop.to_string()];
            let d = data_frames(&read_frames(respond(chat_sse_stream(c, || |t: u32| PIECES[t as usize].to_string()), None, idle)).await);
            // the delivered text is the generated text up to the stop string (or all of it)
            let sent: String = d.iter().filter_map(|x| serde_json::from_str::<serde_json::Value>(x).ok()).map(|j| {
                let delta = &j["choices"][0]["delta"];
                format!("{}{}", delta["reasoning_content"].as_str().unwrap_or(""), delta["content"].as_str().unwrap_or(""))
            }).collect();
            let made: String = toks.iter().map(|t| PIECES[*t as usize]).collect();
            let want = made.find(stop).map_or(made.as_str(), |p| &made[..p]);
            // (a parsed tool call is its tool_calls delta, not text)
            if !d.iter().any(|x| x.contains("\"tool_calls\"")) {
                assert_eq!(sent, want.replace("</think>", ""), "stop {stop:?}, tokens {toks:?}");
            }
            d
        }
        let end = |reason: &str| [finish(reason), "[DONE]".to_string()];
        // the stop string is a whole token: it is not sent, and nothing after it is
        let d = run(PLAIN, "\n\n", &[0, 5, 1], "length").await;
        assert_eq!(d, [vec![role_json(), content("Hello")], end("stop").to_vec()].concat());
        // ...inside a token: the token's text before it is sent, the rest is not
        let d = run(PLAIN, "wor", &[0, 1, 9], "length").await;
        assert_eq!(d, [vec![role_json(), content("Hello"), content(" ")], end("stop").to_vec()].concat());
        // ...spanning two tokens: its first half was never sent
        let d = run(PLAIN, "lo wor", &[0, 1, 9], "length").await;
        assert_eq!(d, [vec![role_json(), content("Hel")], end("stop").to_vec()].concat());
        // a tail that only LOOKED like the start of the stop string leaves with the next token...
        let d = run(PLAIN, "lo!", &[0, 1], "stop").await;
        assert_eq!(d, [vec![role_json(), content("Hel"), content("lo world")], end("stop").to_vec()].concat());
        // ...or at the finish, when the reply ends there
        let d = run(PLAIN, "lo!", &[0], "length").await;
        assert_eq!(d, [vec![role_json(), content("Hel"), content("lo")], end("length").to_vec()].concat());
        // inside a think block, and after it closed
        let think = Shape { thinking: true, ..PLAIN };
        let d = run(think, " oo", &[3, 9, 4, 0], "length").await;
        assert_eq!(d, [vec![role_json(), reasoning("plan")], end("stop").to_vec()].concat());
        let d = run(think, "lo wor", &[3, 4, 0, 1], "length").await;
        assert_eq!(d, [vec![role_json(), reasoning("plan"), content("Hel")], end("stop").to_vec()].concat());
        // a stop string after a tool call: the call is parsed from the text before it
        let d = run(Shape { tools: true, ..PLAIN }, " oops", &[6, 7, 8, 9, 0], "length").await;
        assert_eq!(d.len(), 4, "{d:?}");
        assert!(d[1].contains("\"tool_calls\"") && d[1].contains("get_weather"), "{}", d[1]);
        assert_eq!(d[2..], [finish("tool_calls"), "[DONE]".to_string()]);
        // a stop string that never comes changes nothing
        let d = run(PLAIN, "zzz", &[0, 1], "stop").await;
        assert_eq!(d, [vec![role_json(), content("Hello"), content(" world")], end("stop").to_vec()].concat());
    }

    /// A stop string may hold any characters: only whole characters of it count as a prefix.
    #[test]
    fn a_partial_marker_is_whole_characters() {
        assert_eq!(partial_overlap("abc<to", "<tool_call"), 3);
        assert_eq!(partial_overlap("abc", "<tool_call"), 0);
        assert_eq!(partial_overlap("x\u{e9}", "\u{e9}!"), 2, "the two bytes of the first character");
        assert_eq!(partial_overlap("x\u{c3}\u{a0}", "\u{e9}!"), 0, "a different character with the same lead byte");
        assert_eq!(partial_overlap("\u{4f60}", "\u{4f60}\u{597d}"), 3);
        assert_eq!(partial_overlap("a\u{1f600}", "\u{1f600}\u{1f600}"), 4);
    }

    /// S-B9-REL-INTEG B1: `partial_overlap` is the old rule, in linear time. The stop string is the client's,
    /// and the old loop (every length from the top, `k` bytes compared for each) was quadratic in it: 1.7 ms
    /// per call at 10 KB, 308 ms at 200 KB, per token and per stop string. The assertion counts steps, not time.
    #[test]
    fn partial_overlap_is_the_old_rule_in_linear_time() {
        let naive = |s: &str, m: &str| (1..m.len()).rev().find(|&k| m.is_char_boundary(k) && s.ends_with(&m[..k])).unwrap_or(0);
        // every (text, marker) pair of short strings over a small alphabet that includes 2-, 3- and 4-byte characters
        let alpha = ["a", "b", "\u{e9}", "\u{4f60}", "\u{1f600}", "<"];
        let mut rng = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
        let mut pairs = 0;
        for _ in 0..30_000 {
            let mut mk = |max: u64| -> String { (0..(next() % max)).map(|_| alpha[(next() % alpha.len() as u64) as usize]).collect() };
            let (s, m) = (mk(14), mk(9));
            assert_eq!(partial_overlap(&s, &m), naive(&s, &m), "text {s:?} marker {m:?}");
            pairs += 1;
        }
        assert_eq!(pairs, 30_000);
        // the hostile shapes: stop = a^(L-1) b against text a^L c (no overlap) and against a^L (overlap L-1)
        let mut last = 0u64;
        for l in [1_000usize, 10_000, 100_000] {
            let marker = format!("{}b", "a".repeat(l - 1));
            let (none, steps_none) = partial_overlap_steps(&format!("{}c", "a".repeat(l)), &marker);
            let (all, steps_all) = partial_overlap_steps(&"a".repeat(l), &marker);
            assert_eq!((none, all), (0, l - 1));
            for st in [steps_none, steps_all] {
                assert!(st <= 6 * l as u64, "L={l}: {st} steps is not linear in the stop string");
            }
            assert!(last == 0 || steps_none <= last * 11, "growth between sizes is not linear");
            last = steps_none;
        }
    }

    /// S-B9-REL-INTEG B2: an empty stop string stops nothing. It matches at offset 0 of every text, so the reply
    /// came back EMPTY (non-streaming before the stream fix, and the stream with it).
    #[tokio::test(start_paused = true)]
    async fn an_empty_stop_string_stops_nothing() {
        let r: ChatCompletionRequest = serde_json::from_str(r#"{"messages":[{"role":"user","content":"hi"}],"stop":""}"#).unwrap();
        assert!(r.stop.is_empty(), "{:?}", r.stop);
        let r: ChatCompletionRequest = serde_json::from_str(r#"{"messages":[{"role":"user","content":"hi"}],"stop":["","x",""]}"#).unwrap();
        assert_eq!(r.stop, vec!["x".to_string()]);
        // the stream itself, handed an empty stop string by a caller that did not go through the request
        let (tx, rx, bl) = tok_channel();
        feed(tx, vec![(1, tok(0)), (2, tok(1)), (3, fin("stop"))]);
        let mut c = cx(rx, bl, PLAIN, None);
        c.stops = vec![String::new(), "zzz".to_string()];
        let d = data_frames(&read_frames(respond(chat_sse_stream(c, || |t: u32| PIECES[t as usize].to_string()), None, idle)).await);
        assert_eq!(d, [vec![role_json(), content("Hello"), content(" world")], vec![finish("stop"), "[DONE]".to_string()]].concat());
        // the non-streaming handler's post-hoc search skips empty strings too
        let src = src_text!("server");
        let code = &src[..src.find("mod benchfix_sse_tests").unwrap()];
        assert!(code.contains("req.stop.iter().filter(|s| !s.is_empty()).filter_map(|s| text.find(s.as_str())).min()"));
    }

    /// `--dump-tools`: one line, the whole text, and it reads back exactly.
    #[test]
    fn the_tool_dump_line_carries_the_whole_text() {
        let raw = format!("<tool_call>\n<function=write_file>\n<parameter=content>\n{}\n\"q\" \u{e9}\t\u{1f600}\n</parameter>\n</function>\n</tool_call>", "x".repeat(5000));
        let line = dump_tools_line(&raw);
        assert!(!line.contains('\n') && line.starts_with("[dump-tools] {"), "{}", &line[..40]);
        let back: serde_json::Value = serde_json::from_str(line.strip_prefix("[dump-tools] ").unwrap()).unwrap();
        assert_eq!(back["raw"].as_str(), Some(raw.as_str()));
        // both response modes write it, next to the older 1200-character line
        let src = src_text!("server");
        let code = &src[..src.find("mod benchfix_sse_tests").unwrap()];
        assert_eq!(code.matches(&["if dump { crate::reprintln!(\"{}\", ", "dump_tools_line(&"].concat()).count(), 2);
        assert!(crate::opt!("dump-tools").def().help.contains("[dump-tools]"));
    }

    /// `held_tail`, the split itself.
    #[test]
    fn the_held_tail_split() {
        let t = Some(TOOL_OPEN);
        let ht = |acc: &'static str, emitted, in_reasoning, tool| held_tail(acc, emitted, in_reasoning, tool, "</think>");
        assert_eq!(ht("a <", 2, false, t), ("", "<"));
        assert_eq!(ht("a <", 3, false, t), ("", ""), "everything was sent");
        assert_eq!(ht("a", 5, false, t), ("", ""), "a stop string cut the buffer below the cursor");
        assert_eq!(ht("plan</thi", 4, true, t), ("</thi", ""));
        assert_eq!(ht("plan<tool", 4, true, t), ("<tool", ""), "a marker PREFIX is reasoning text");
        assert_eq!(ht("plan<tool_call_x y", 4, true, t), ("", "<tool_call_x y"), "from a marker on: content");
        assert_eq!(ht("pl</<tool_call> x", 2, true, t), ("</", "<tool_call> x"));
        assert_eq!(ht("plan<tool_call_x", 4, true, None), ("<tool_call_x", ""), "tool parsing off: no marker");
        assert_eq!(ht("é<", 2, false, t), ("", "<"), "cursors are char boundaries");
        // a think block that closed inside the last piece: the tag is not sent
        assert_eq!(ht("a<think>b</think>\n\nc", 8, true, t), ("b", "c"));
        assert_eq!(ht("a<think>b</think>", 8, true, t), ("b", ""));
        assert_eq!(ht("x<tool_call> </think> y", 1, true, t), ("", "<tool_call> </think> y"), "a close tag after a held marker is the marker's text");
        assert_eq!(ht("a</think>b", 1, false, t), ("", "</think>b"), "in content mode a close tag is text");
    }

    /// --sse-keepalive 0: no comment frames; the data events are unchanged.
    #[tokio::test(start_paused = true)]
    async fn keepalive_off_sends_no_comment_frames() {
        let (tx, rx, bl) = tok_channel();
        feed(tx, vec![(150, tok(0)), (151, fin("stop"))]);
        let frames = read_frames(respond(chat(rx, bl, PLAIN), None, idle)).await;
        assert_eq!(timeline(&frames), [(150, "role"), (150, "data"), (151, "data"), (151, "data")]);
    }

    /// The legacy /v1/completions stream has no role chunk; it gets the same idle comment frames.
    #[tokio::test(start_paused = true)]
    async fn a_stream_without_a_role_chunk_gets_the_same_comment_frames() {
        let s = async_stream::stream! {
            tokio::time::sleep(secs(25)).await;
            yield Ok::<_, std::convert::Infallible>(Event::default().data("{\"choices\":[{\"index\":0,\"text\":\"x\"}]}"));
            yield Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"));
        };
        let frames = read_frames(respond(s, Some(secs(10)), idle)).await;
        assert_eq!(timeline(&frames), [(10, "ka"), (20, "ka"), (25, "data"), (25, "data")]);
        assert!(frames[..2].iter().all(|f| f.raw == KA));
    }

    /// A client that leaves during the wait: dropping the response drops the generator and its
    /// receiver, so the scheduler's `tx.is_closed()` sweep sees it (the lane is freed) without
    /// any data having been written, before and after a keep-alive frame.
    #[tokio::test(start_paused = true)]
    async fn dropping_the_response_before_the_first_token_closes_the_event_channel() {
        for after_a_frame in [false, true] {
            let (tx, rx, bl) = tok_channel();
            let _ = tx.send(TokEvent::Admitted { cached_tokens: 0 });
            let mut body = respond(chat(rx, bl, PLAIN), Some(secs(10)), idle).into_body().into_data_stream();
            if after_a_frame {
                let first = String::from_utf8(body.next().await.unwrap().unwrap().to_vec()).unwrap();
                assert_eq!(first, KA, "the request is still waiting: only a comment frame so far");
            } else {
                // polled once (the generator has started and holds the receiver), nothing sent
                assert!(tokio::time::timeout(secs(3), body.next()).await.is_err());
            }
            assert!(!tx.is_closed());
            drop(body);
            assert!(tx.is_closed(), "the scheduler's cancel signal: the receiver is gone");
        }
    }

    /// W4 (review m4): keep-alive frames do not hide a hung engine. A stream that never
    /// produces an event behind a scheduler that never completes a unit of work gets frames for
    /// SSE_KEEPALIVE_STALL (60 s) and is then silent, exactly like a stream without keep-alive,
    /// so its client's idle timeout can fire. A stream that arrives while the scheduler is
    /// already wedged (busy, last completed work older than the bound) gets no frame at all.
    #[tokio::test(start_paused = true)]
    async fn keepalive_stops_when_the_scheduler_completes_no_work() {
        assert_eq!(SSE_KEEPALIVE_STALL, secs(60));
        let never = || tokio_stream::pending::<Result<Event, std::convert::Infallible>>();
        let frames = read_frames_until(respond(never(), Some(secs(10)), idle), Some(secs(600))).await;
        assert_eq!(timeline(&frames), [(10, "ka"), (20, "ka"), (30, "ka"), (40, "ka"), (50, "ka"), (60, "ka")]);
        // an interval longer than the bound: the first tick is already past it
        assert!(read_frames_until(respond(never(), Some(secs(61)), idle), Some(secs(600))).await.is_empty());
        // already wedged when the stream starts: busy, nothing completed for 10 minutes
        let wedged = || SchedState { busy: true, work_age: Some(secs(600)) };
        assert!(read_frames_until(respond(never(), Some(secs(10)), wedged), Some(secs(600))).await.is_empty());
        // busy and nothing ever completed (a first request whose first unit of work hangs)
        let stuck_first = || SchedState { busy: true, work_age: None };
        assert!(read_frames_until(respond(never(), Some(secs(10)), stuck_first), Some(secs(600))).await.is_empty());
        // the largest accepted interval and an interval no clock can add: no frame, no panic
        // (axum's own KeepAlive computes `Instant::now() + interval` unchecked at every event)
        for every in [secs(3600), Duration::MAX] {
            let (tx, rx, bl) = tok_channel();
            feed(tx, vec![(5, tok(0)), (6, fin("stop"))]);
            let alive = || SchedState { busy: true, work_age: Some(secs(0)) };
            let frames = read_frames(respond(chat(rx, bl, PLAIN), Some(every), alive)).await;
            assert_eq!(timeline(&frames), [(5, "role"), (5, "data"), (6, "data"), (6, "data")]);
        }
    }

    /// The rule itself, as a table.
    #[test]
    fn the_keepalive_rule() {
        let st = |busy, age: Option<u64>| SchedState { busy, work_age: age.map(secs) };
        let stall = SSE_KEEPALIVE_STALL;
        // the scheduler completed work within the bound: alive, whatever else
        for busy in [false, true] {
            assert!(keepalive_live(st(busy, Some(0)), secs(100_000), stall));
            assert!(keepalive_live(st(busy, Some(60)), secs(100_000), stall), "60 s is inside the bound");
        }
        // idle for hours, a request has just arrived: the stream's own age covers the pick-up
        assert!(keepalive_live(st(false, Some(7200)), secs(10), stall));
        assert!(keepalive_live(st(false, None), secs(60), stall));
        assert!(!keepalive_live(st(false, Some(7200)), secs(61), stall), "nothing completed for this stream in 61 s");
        assert!(!keepalive_live(st(false, None), secs(61), stall));
        // busy and nothing completed for longer than the bound: wedged, for old and new streams
        assert!(!keepalive_live(st(true, Some(61)), secs(10), stall));
        assert!(!keepalive_live(st(true, Some(61)), secs(100_000), stall));
        assert!(!keepalive_live(st(true, None), secs(10), stall));
    }

    /// W4: the frames follow the scheduler's progress. Work completes every 3 s (prefill
    /// chunks, the decode rounds of other requests: a long legitimate wait) from 1 s to 118 s,
    /// then nothing (a hang) until 301 s, then again every 3 s until 331 s. Frames: every 10 s
    /// while the last completed work is at most 60 s old (the last one at 170 s; 180 s is 62 s
    /// after 118 s), silence during the hang, frames again from 310 s to 390 s.
    #[tokio::test(start_paused = true)]
    async fn keepalive_follows_the_scheduler_progress() {
        let last_work: Arc<std::sync::Mutex<Option<Instant>>> = Arc::new(std::sync::Mutex::new(None));
        let t0 = Instant::now();
        let w = Arc::clone(&last_work);
        tokio::spawn(async move {
            let ticks = (1..=118).step_by(3).chain((301..=331).step_by(3));
            for at in ticks {
                tokio::time::sleep_until(t0 + secs(at)).await;
                *w.lock().unwrap() = Some(Instant::now());
            }
        });
        let w = Arc::clone(&last_work);
        let sched = move || SchedState { busy: true, work_age: w.lock().unwrap().map(|t| t.elapsed()) };
        let never = tokio_stream::pending::<Result<Event, std::convert::Infallible>>();
        let frames = read_frames_until(respond(never, Some(secs(10)), sched), Some(secs(600))).await;
        let at: Vec<u64> = timeline(&frames).iter().map(|(t, k)| { assert_eq!(*k, "ka"); *t }).collect();
        let want: Vec<u64> = (10..=170).step_by(10).chain((310..=390).step_by(10)).collect();
        assert_eq!(at, want);
    }

    /// The real scheduler gauges (`metrics::sched_state`) through `sse_with_keepalive`, the
    /// function both endpoints call. The gauges are process globals that the metrics tests
    /// drive, so this test takes their lock and sets the state it asserts on.
    #[tokio::test(start_paused = true)]
    async fn sse_with_keepalive_reads_the_scheduler_gauges() {
        let _g = crate::metrics::tests::SCHED_GAUGE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        crate::metrics::set_max_batch(2); // the value the metrics tests assert; initialises the gauge clock
        // idle scheduler, young stream: frames
        crate::metrics::sched_idle();
        let (tx, rx, bl) = tok_channel();
        feed(tx, vec![(25, tok(0)), (26, fin("stop"))]);
        let frames = read_frames(sse_with_keepalive(chat(rx, bl, PLAIN), Some(secs(10)), SseKeepAliveFrame::Comment)).await;
        assert_eq!(timeline(&frames), [(10, "ka"), (20, "ka"), (25, "role"), (25, "data"), (26, "data"), (26, "data")]);
        // work completed just now (real clock: the whole test takes milliseconds): alive although busy
        crate::metrics::sched_step();
        crate::metrics::sched_touch();
        std::thread::sleep(Duration::from_millis(2)); // the gauge has millisecond resolution; 0 ms reads as "never"
        crate::metrics::sched_touch();
        assert!(keepalive_live(crate::metrics::sched_state(), secs(100_000), SSE_KEEPALIVE_STALL));
        crate::metrics::sched_idle();
    }

    /// LANERES1 `--sse-keepalive-frame data`: the pure payload is a chunk with an EMPTY choices
    /// array (the first real choices entry is still the request's first token — the keep-alive
    /// adds none), carrying the stream's own id/created/model; "data" mode builds that frame and
    /// the default mode stays the comment line.
    #[test]
    fn the_keepalive_data_frame_is_an_empty_choices_chunk() {
        let p = sse_keepalive_data_payload("chatcmpl-x", 1_700_000_000, "test-model", "chat.completion.chunk");
        let v: serde_json::Value = serde_json::from_str(&p).expect("valid JSON");
        assert_eq!(v["id"], "chatcmpl-x");
        assert_eq!(v["object"], "chat.completion.chunk");
        assert_eq!(v["created"], 1_700_000_000);
        assert_eq!(v["model"], "test-model");
        assert_eq!(v["choices"], serde_json::json!([]), "no choices entry before the first real token");
        assert!(matches!(sse_keepalive_frame_of_mode("data", "id", 1, "m", "text_completion.chunk"), SseKeepAliveFrame::Data(_)));
        assert!(matches!(sse_keepalive_frame_of_mode("comment", "id", 1, "m", "chat.completion.chunk"), SseKeepAliveFrame::Comment));
        assert!(matches!(sse_keepalive_frame_of_mode("", "id", 1, "m", "chat.completion.chunk"), SseKeepAliveFrame::Comment));
    }

    /// `--inject-stall ROUND:SECONDS`, the W4 drill flag: parsed strictly (the store is not
    /// written here; unset in a unit-test process, so `sched_step` never sleeps).
    #[test]
    fn the_stall_drill_flag() {
        assert_eq!(parse_stall_drill(None), Ok(None));
        assert_eq!(parse_stall_drill(Some("40:100")), Ok(Some((40, 100))));
        assert_eq!(parse_stall_drill(Some(" 1 : 600 ")), Ok(Some((1, 600))));
        for bad in ["", "40", "40:", ":100", "0:100", "40:0", "40:601", "-1:5", "a:b", "40:100:1"] {
            let e = parse_stall_drill(Some(bad)).unwrap_err();
            assert!(e.starts_with("--inject-stall takes ROUND:SECONDS") && e.ends_with(&format!("(got '{bad}')")), "{e}");
        }
        let o = crate::opt!("inject-stall").def();
        assert_eq!((o.env, o.def, o.cat, o.scope), ("", "unset", crate::opts::Cat::Diag, crate::opts::Scope::Local));
        assert!(crate::opts::get(crate::opt!("inject-stall")).is_none(), "never set in the unit suite");
        // refused at the start of `main`, with the command-line errors (source pin: main.rs is the binary)
        let main_src = src_text!("main");
        assert_eq!(main_src.matches(&format!("parse_cli(&args).and_then(|_| gb10_inference::server::{}(&args))", "check_serve_options")).count(), 1);
        assert_eq!(check_serve_options(&[]), Ok(()), "nothing is set in the unit suite: the defaults resolve");
        // R6(b): a one-box drill. Refused with any TP role on the command line; fine without one,
        // and no refusal at all when the drill is not set.
        let argv = |a: &[&str]| a.iter().map(|x| x.to_string()).collect::<Vec<String>>();
        for role in [&["--server", "--tp", "2"][..], &["--server", "--tp=4"], &["--tp"], &["--head", "--model-dir", "m"], &["--node", "--port", "29500"]] {
            let e = stall_drill_topology(true, &argv(role)).unwrap_err();
            assert!(e.starts_with("--inject-stall is a one-box drill and is refused with --"), "{e}");
            assert!(e.contains(role.iter().find(|a| a.starts_with("--tp") || **a == "--head" || **a == "--node").unwrap().split('=').next().unwrap()), "{e}");
            assert_eq!(stall_drill_topology(false, &argv(role)), Ok(()));
        }
        assert_eq!(stall_drill_topology(true, &argv(&["--server", "--model-dir", "m", "--tp-like-name", "--port", "9000"])), Ok(()));
        assert!(o.help.contains("REFUSED at startup together with --tp, --head or --node") && o.help.contains("nothing on the DSV4 server"), "{}", o.help);
    }

    /// W3: the boot line, and the values behind it (the option store is not written here: other
    /// tests latch `stream_backlog_events()` from it).
    #[test]
    fn the_keepalive_boot_line_and_its_default() {
        assert_eq!(sse_keepalive_boot_line(None), "[serve] sse keep-alive: off (--sse-keepalive 0)");
        let on = sse_keepalive_boot_line(Some(secs(10)));
        assert!(on.starts_with("[serve] sse keep-alive: every 10 s (--sse-keepalive;"), "{on}");
        assert!(on.contains("in the last 60 s"), "{on}");
        assert!(on.contains("the whole scheduler") && on.contains("for its first 60 s"), "{on}");
        // nothing is set in a unit-test process: the default is the registry row's (an owner
        // ruling, 2026-10-09: on, every 10 s), and it boots
        let id = crate::opt!("sse-keepalive");
        if crate::opts::get(id).is_none() {
            assert_eq!(resolve_sse_keepalive(), Ok(Some(secs(10))));
            let line = serve_options_boot_line().unwrap();
            assert!(line.starts_with(&on), "{line}");
            // S-B9-TOOLPARSE3: the parser rules and the counters are named on the next boot line
            assert!(line.contains("[serve] tool-call parser: values end by --tool-quote-pairing off, blocks end by --tool-block-end after-function, unclosed blocks by --tool-unclosed drop"), "{line}");
        }
        assert_eq!(crate::opts::ranged_from(id, Some("0")), Ok(0), "0 = off");
        assert_eq!(crate::opts::ranged_from(id, Some("10")), Ok(10));
        // the help text states the bound that is in the code, and which servers honour the flag
        let help = id.def().help;
        assert!(help.contains(&format!("for {} s they stop", SSE_KEEPALIVE_STALL.as_secs())), "{help}");
        assert!(help.contains("the DSV4 server sends no keep-alive frames"), "{help}");
        // ...and what the bound is NOT (review of BENCHFIX2): it is server-wide, not per request
        assert!(help.contains("the WHOLE scheduler") && help.contains("of any request"), "{help}");
        assert!(help.contains("keeps receiving frames") && help.contains("already hung"), "{help}");
    }

    /// Real wire bytes: the response is served by axum on a TCP socket and read back as bytes.
    /// Real time, so only ORDER is asserted (no upper bound on any duration): the test waits
    /// until two comment frames are on the wire, checks that nothing else is, and only then lets
    /// the first token out.
    #[tokio::test]
    async fn wire_bytes_over_a_real_socket() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let (tx, rx, bl) = tok_channel();
        let slot = Arc::new(std::sync::Mutex::new(Some((rx, bl))));
        let app = axum::Router::new().route("/s", get(move || {
            let slot = Arc::clone(&slot);
            async move {
                let (rx, bl) = slot.lock().unwrap().take().expect("one request");
                respond(chat(rx, bl, Shape { usage: false, thinking: false, tools: false }),
                        Some(Duration::from_millis(20)), idle)
            }
        }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); });
        let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
        // HTTP/1.0: the body is the SSE bytes themselves (no chunked transfer framing)
        s.write_all(b"GET /s HTTP/1.0\r\nhost: t\r\n\r\n").await.unwrap();
        let mut wire = Vec::new();
        let mut buf = [0u8; 4096];
        let body_of = |wire: &[u8]| -> Option<String> {
            let text = String::from_utf8(wire.to_vec()).expect("utf-8");
            text.find("\r\n\r\n").map(|i| text[i + 4..].to_string())
        };
        // the wait: read until two comment frames have arrived
        loop {
            let n = s.read(&mut buf).await.unwrap();
            assert!(n > 0, "the stream ended during the wait");
            wire.extend_from_slice(&buf[..n]);
            if body_of(&wire).map_or(false, |b| b.matches(KA).count() >= 2) { break; }
        }
        let head = String::from_utf8(wire.clone()).unwrap();
        assert!(head.starts_with("HTTP/1.0 200"), "{head}");
        assert!(head.to_ascii_lowercase().contains("content-type: text/event-stream"), "{head}");
        assert!(!head.to_ascii_lowercase().contains("transfer-encoding"), "{head}");
        let waited = body_of(&wire).unwrap();
        assert!(waited.len() >= 2 * KA.len() && waited.replace(KA, "").is_empty(),
                "before the first token the body is comment frames only: {waited:?}");
        // now the first token, a second one, the end
        tx.send(tok(0)).unwrap();
        tx.send(tok(1)).unwrap();
        tx.send(fin("stop")).unwrap();
        s.read_to_end(&mut wire).await.unwrap();
        let body = body_of(&wire).unwrap();
        let tail = &body[waited.len()..];
        let data_part = tail.replace(KA, ""); // a frame may still land between two reads
        assert_eq!(data_part, format!("data: {}\n\ndata: {}\n\ndata: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
                                      role_json(), content("Hello"), content(" world"), finish("stop")));
        assert_eq!(body.find("\"choices\""), body.find(&role_json()).map(|i| i + role_json().find("\"choices\"").unwrap()),
                   "the first `choices` on the wire is the role chunk's");
    }

    /// What the keep-alive rule ASSUMES of the two schedulers (source pins: the unit suite cannot
    /// build a scheduler). The rule reads one signal, `metrics::sched_touch`, and takes it to mean
    /// "a decode round or a prefill chunk completed". S-B9-REL-SURFACE2B R0: on the EXL3
    /// scheduler that signal used to be written at the end of ONE arm of the step (round-robin)
    /// and per prefill chunk; the shared plain step and the `--lane-order fcfs` arm returned
    /// before it, so a request waiting behind lanes that decode on those arms stopped receiving
    /// frames after 60 s and velogb10_scheduler_last_step_age_seconds read a false wedge.
    #[test]
    fn every_arm_of_both_scheduler_steps_reaches_the_touch() {
        // a function's text: from its signature to the next method of the impl
        fn body<'a>(src: &'a str, sig: &str) -> &'a str {
            let at = src.find(sig).unwrap_or_else(|| panic!("{sig} not found"));
            let end = src[at + sig.len()..].find("\n    fn ").map_or(src.len(), |e| at + sig.len() + e);
            &src[at..end]
        }
        // code only: a `//` comment may say "return"
        fn code(text: &str) -> String {
            text.lines().map(|l| l.split("//").next().unwrap_or("")).collect::<Vec<_>>().join("\n")
        }
        let (touch, round_start, idle) = (["crate::metrics::", "sched_touch()"].concat(), ["crate::metrics::", "sched_step()"].concat(),
                                          ["crate::metrics::", "sched_idle()"].concat());

        // --- EXL3 (src/exl3_serve.rs)
        let exl3 = src_text!("exl3_serve");
        let step = body(exl3, "fn step(&mut self) -> Result<()> {");
        // LANEWIDE: step_rounds decides the wide arm, then today's arms run in step_rounds_rest (one body for the checks)
        // COPYLANES: cl_rest (the rest of a busy step with copy work) is entered from step_rounds only: one body with it
        let rounds_s = [body(exl3, "fn step_rounds(&mut self) -> Result<()> {"), body(exl3, "fn step_rounds_rest(&mut self,"),
                        body(exl3, "fn cl_rest(&mut self,")].concat();
        let rounds = rounds_s.as_str();
        // `step` is the wrapper: it runs the rounds once and touches after ANY arm that ran one
        assert_eq!(step.matches("let r = self.step_rounds();").count(), 1);
        assert!(step.contains(&["if r.is_ok() && self.steps != before { ", touch.as_str(), "; }"].concat()), "{step}");
        assert_eq!(exl3.matches(&["self.", "step_rounds()"].concat()).count(), 1, "nothing but `step` runs the rounds");
        // the rounds: the round counter moves exactly once, before any arm, and the only return
        // ahead of it is the one that ran nothing (no lane holds a request)
        assert_eq!(code(rounds).matches("self.steps += 1;").count(), 1);
        assert_eq!(exl3.matches("self.steps += 1;").count(), 1, "the round counter moves in step_rounds only");
        let before_count = code(&rounds[..rounds.find("self.steps += 1;").unwrap()]);
        assert_eq!(before_count.matches("return").count(), 1, "one return before the round is counted");
        assert!(before_count.contains("if slots.is_empty() {\n            return Ok(());"), "...and it is the empty one");
        assert!(code(rounds).trim_start_matches(|c| c != '{').contains(&round_start), "a round starts with sched_step");
        assert!(!code(rounds).contains(&touch), "no arm writes the signal itself: the arms return early");
        // the arms (speculative lane, plain batch) are called from the rounds, and from the packed
        // arm (`--lane-pack`: `step_packed` and its `step_pair`, which run the lone round / the plain
        // batch for lanes outside a pair and for the control-cadence fallback), and from nowhere else.
        // The packed arm itself is entered from the rounds only, so it ends in `step`'s touch too.
        let packed = body(exl3, "fn step_packed(&mut self,");
        let pair = body(exl3, "fn step_pair(&mut self,");
        let wide = body(exl3, "fn step_wide(&mut self,");
        assert_eq!(exl3.matches(&["self.", "step_packed("].concat()).count(), 2, "two calls of the packed arm (rounds, cl_rest)");
        assert_eq!(code(rounds).matches(&["self.", "step_packed("].concat()).count(), 2, "...and they are in step_rounds / cl_rest");
        assert_eq!(exl3.matches(&["self.", "cl_rest("].concat()).count(), 1, "cl_rest is entered from step_rounds only");
        assert_eq!(code(body(exl3, "fn step_rounds(&mut self) -> Result<()> {")).matches(&["self.", "cl_rest("].concat()).count(), 1);
        assert_eq!(exl3.matches(&["self.", "step_pair("].concat()).count(), 1, "one call of the packed round");
        assert_eq!(code(packed).matches(&["self.", "step_pair("].concat()).count(), 1, "...and it is in step_packed");
        for arm in [["self.", "step_mtp("].concat(), ["self.", "step_plain("].concat()] {
            assert!(code(rounds).matches(arm.as_str()).count() >= 3, "{arm}: rr, fcfs and shared-plain arms");
            let inside = [rounds, packed, pair, wide].iter().map(|f| code(f).matches(arm.as_str()).count()).sum::<usize>();
            assert_eq!(exl3.matches(arm.as_str()).count(), inside, "{arm} outside step_rounds / step_packed / step_pair / step_wide");
        }
        // a prefill chunk completes -> the signal; both scheduler loops mark idle before they block
        assert_eq!(body(exl3, "fn prefill_range(&mut self,").matches(&touch).count(), 1);
        // PACKPF: a packed prefill pass completes -> the signal too (pack_run, the --prefill-pack admission)
        assert_eq!(body(exl3, "fn pack_run(&mut self,").matches(&touch).count(), 1);
        // TURNRESUME: a background next-turn prefill chunk completes -> the signal too (a long pass stays live)
        assert_eq!(body(exl3, "fn turn_pass(&mut self,").matches(&touch).count(), 1);
        assert_eq!(exl3.matches(&touch).count(), 4, "step, prefill_range, pack_run and turn_pass, nowhere else");
        assert!(exl3.matches(&idle).count() >= 2, "run and run_tp_head");

        // --- NVFP4 (src/batch.rs): one arm, no early return
        let nv = src_text!("batch");
        let dstep = body(nv, "fn decode_step(&mut self, b: usize) {");
        let dcode = code(dstep);
        let first = dcode.lines().skip(1).map(str::trim).find(|l| !l.is_empty()).unwrap();
        assert_eq!(first, format!("{round_start};"), "the round starts with sched_step");
        assert_eq!(dcode.matches(&touch).count(), 1);
        assert!(!dcode[..dcode.find(&touch).unwrap()].contains("return"), "decode_step returns nowhere before the signal");
        assert_eq!(nv.matches(&["self.", "decode_step("].concat()).count(), 4, "the scheduler loops (one box, TP head, TP node, inline) all go through it");
        // a prefill window completes -> the signal; the two returns ahead of it did no prefill
        // work (no cursor; a full prefix hit, which goes straight to pf_finish)
        let pf = code(body(nv, "fn pf_advance(&mut self) {"));
        assert_eq!(pf.matches(&touch).count(), 1);
        assert_eq!(pf[..pf.find(&touch).unwrap()].matches("return").count(), 2);
        assert_eq!((nv.matches(&touch).count(), nv.matches(&round_start).count()), (2, 1));
        assert!(nv.matches(&idle).count() >= 2);
    }

    /// The link from the handlers to the tested code (source pins; the generator and the
    /// response builder themselves are exercised above). The streaming branch of
    /// /v1/chat/completions returns exactly `sse_with_keepalive(chat_sse_stream(..))`, the
    /// /v1/completions stream goes through `sse_with_keepalive`, and nothing in this file builds
    /// an SSE response any other way.
    #[test]
    fn the_handlers_build_their_streams_from_the_tested_functions() {
        let src = src_text!("server");
        let code = &src[..src.find("mod benchfix_sse_tests").unwrap()];
        let n = |needle: String| code.matches(&needle).count();
        assert_eq!(n(format!("let stream = {}(cx, move || {{", "chat_sse_stream")), 1, "the chat handler's stream");
        assert_eq!(n(format!("let mut resp = {}(stream, sse_keepalive(), ka_frame);", "sse_with_keepalive")), 1);
        assert_eq!(n(format!("return {}(stream, sse_keepalive(), ka_frame);", "sse_with_keepalive")), 1, "/v1/completions");
        // the only Sse responses are the two arms of sse_with_keepalive, and the only generator
        // that writes a role chunk is chat_sse_stream (once at the first token, once before the
        // terminal events)
        assert_eq!(n(format!("{}::new(", "Sse")), 2);
        assert_eq!(n(format!("role_pending.{}()", "take")), 2);
        assert_eq!(n(format!("{}!", "async_stream::stream")), 3, "keepalive_frames, chat_sse_stream, /v1/completions");
        // the OTel stream_start hook: one call site, the generator's first statement
        assert_eq!(n(format!("r.start(&{});", "role_chunk")), 1);
        let gen = &code[code.find("pub(crate) fn chat_sse_stream").unwrap()..];
        let body = &gen[gen.find("async_stream::stream! {").unwrap() + "async_stream::stream! {".len()..];
        let first = body.lines().map(str::trim).find(|l| !l.is_empty() && !l.starts_with("//")).unwrap();
        assert_eq!(first, "if let Some(r) = &otel_req { r.start(&role_chunk); }");
    }

    // ===================================================================================
    // S-B9-TOOLPARSE3 T8 (b)(c)(d): the serial fixtures, the fuzz class D reproduction and
    // two stated behaviours, all on the REAL generator (`chat_sse_stream`).
    // ===================================================================================

    /// What a client assembles from one chat response, streamed or not.
    #[derive(Debug, Default, PartialEq)]
    struct Wire {
        /// every `reasoning_content` delta, concatenated
        reasoning: String,
        /// every `content` delta, concatenated
        content: String,
        /// (name, arguments as JSON) of every tool call, in order
        calls: Vec<(String, serde_json::Value)>,
        /// the finish chunk's `finish_reason`
        finish: String,
        /// the delta kinds in wire order: 'r' reasoning, 'c' content, 't' tool_calls
        order: String,
    }

    fn wire_of(data: &[String]) -> Wire {
        let mut w = Wire::default();
        for f in data {
            if f == "[DONE]" { continue; }
            let j: serde_json::Value = serde_json::from_str(f).expect("a data frame is JSON");
            let Some(ch) = j["choices"].as_array().and_then(|a| a.first()) else { continue };
            let d = &ch["delta"];
            if let Some(s) = d["reasoning_content"].as_str() { w.reasoning.push_str(s); w.order.push('r'); }
            if let Some(s) = d["content"].as_str() { w.content.push_str(s); w.order.push('c'); }
            if let Some(a) = d["tool_calls"].as_array() {
                w.order.push('t');
                for c in a {
                    let args = c["function"]["arguments"].as_str().expect("arguments is a string");
                    w.calls.push((c["function"]["name"].as_str().unwrap().to_string(),
                                  serde_json::from_str(args).expect("arguments is JSON")));
                }
            }
            if let Some(r) = ch["finish_reason"].as_str() { w.finish = r.to_string(); }
        }
        w
    }

    /// The request's `tools` array as `cx` builds it.
    fn t8_tools() -> Vec<serde_json::Value> {
        vec![serde_json::json!({"type": "function", "function": {"name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}})]
    }

    /// The REAL generator over a pre-queued event channel: one token per piece (the decoder
    /// returns the piece), then `Finish`. `thinking` = the prompt opened the think block.
    async fn t8_stream(thinking: bool, tools: bool, pieces: &[String]) -> Wire {
        let (tx, rx, bl) = tok_channel();
        let _ = tx.send(TokEvent::Admitted { cached_tokens: 0 });
        for i in 0..pieces.len() { let _ = tx.send(tok(i as u32)); }
        let _ = tx.send(fin("stop"));
        let table = pieces.to_vec();
        let sh = Shape { thinking, tools, usage: false };
        let s = chat_sse_stream(cx(rx, bl, sh, None), move || move |t: u32| table[t as usize].clone());
        wire_of(&data_frames(&read_frames(respond(s, None, idle)).await))
    }

    /// The non-streaming handler's decision for a finished generation, call for call (the
    /// non-streaming branch of `chat_completions`: `split_think`, the SURFACE2 unclosed-think
    /// split, `tools::parse`, `finalize_parsed`). The handler itself needs an AppState (a
    /// tokenizer and a scheduler), so this is the one hand-written half of the comparison.
    fn t8_non_stream(thinking: bool, tools: bool, text: &str) -> Wire {
        let (open, close) = ("<think>", "</think>");
        let (reasoning, content) = split_think_in(text, open, close, thinking);
        let closed = text.contains(close);
        let (reasoning, content) = if thinking && !closed && !unclosed_think_as_content() {
            unclosed_think_split(text, tools)
        } else {
            (reasoning, content)
        };
        let t = t8_tools();
        let parsed = if tools {
            // F8: the real non-streaming branch calls parse_served with the engine finish (a
            // wrapper the model never closed is completed on `stop`): the reference must too
            crate::tools::parse_served(&content, Some(&t), "stop")
        } else {
            crate::tools::ParsedOutput { content: content.trim().to_string(), tool_calls: Vec::new() }
        };
        let (content, calls, finish) = crate::tools::finalize_parsed(&content, parsed, "stop");
        Wire {
            reasoning: reasoning.unwrap_or_default(),
            content: content.unwrap_or_default(),
            calls: calls.iter().map(|c| (c.function.name.clone(),
                                         serde_json::from_str(&c.function.arguments).unwrap())).collect(),
            finish: spec_finish_reason(&finish).to_string(),
            order: String::new(),
        }
    }

    /// `text` as pieces of the given sizes in CHARACTERS (cycled; the last piece takes the rest).
    fn t8_cut(text: &str, sizes: &[usize]) -> Vec<String> {
        let mut out = Vec::new();
        let mut it = text.chars().peekable();
        let mut k = 0;
        while it.peek().is_some() {
            let n = sizes[k % sizes.len()].max(1);
            out.push(it.by_ref().take(n).collect::<String>());
            k += 1;
        }
        out
    }

    /// Every way to cut `text` at ONE character boundary, plus: one piece, one character per
    /// piece, and eight pseudo-random cuts into pieces of 1 to 9 characters (a fixed LCG: the
    /// same cuts on every run). A marker that spans a cut is held at that cut.
    fn t8_chunkings(text: &str) -> Vec<Vec<String>> {
        let mut all: Vec<Vec<String>> = vec![vec![text.to_string()], t8_cut(text, &[1])];
        for (p, _) in text.char_indices().skip(1) {
            all.push(vec![text[..p].to_string(), text[p..].to_string()]);
        }
        let mut s: u64 = 0x2545_f491_4f6c_dd1d;
        for _ in 0..8 {
            let sizes: Vec<usize> = (0..16).map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                1 + ((s >> 33) % 9) as usize
            }).collect();
            all.push(t8_cut(text, &sizes));
        }
        all
    }

    /// The `tests/toolcall_serial_test.rs` fixtures (the same strings: that file asserts the
    /// finish-time DECISION on them, this test asserts what the stream SENDS). Each entry:
    /// (name, generated text, the calls the response must carry, the content it must carry).
    fn t8_fixtures() -> Vec<(&'static str, &'static str, Vec<(&'static str, serde_json::Value)>, &'static str)> {
        let w = |c: &str| serde_json::json!({"city": c});
        vec![
            ("malformed captured",
             "<tool_call>\nfunction=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=units>\nc\n</parameter>\n</function>\n</tool_call>",
             vec![("get_weather", serde_json::json!({"city": "Paris", "units": "c"}))], ""),
            ("well-formed",
             "<tool_call>\n<function=get_weather>\n<parameter=city>\nBerlin\n</parameter>\n</function>\n</tool_call>",
             vec![("get_weather", w("Berlin"))], ""),
            ("f8: wrapper never closed, element closed",
             "Writing it down.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>",
             vec![("get_weather", w("Paris"))], "Writing it down."),
            ("f8: two closed elements then a cut one",
             "Both.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n<function=get_weather>\n<parameter=city>\nOs",
             vec![("get_weather", w("Paris")), ("get_weather", w("Tokyo"))], "Both."),
            ("two calls, prose before",
             "I'll check both cities.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>",
             vec![("get_weather", w("Paris")), ("get_weather", w("Tokyo"))], "I'll check both cities."),
            ("no arguments",
             "<tool_call>\n<function=list_cities>\n</function>\n</tool_call>",
             vec![("list_cities", serde_json::json!({}))], ""),
            ("unicode",
             "<tool_call>\n<function=get_weather>\n<parameter=city>\nZürich 🇨🇭 北çå\n</parameter>\n<parameter=note>\ntempérature — très élevée\n</parameter>\n</function>\n</tool_call>",
             vec![("get_weather", serde_json::json!({"city": "Zürich 🇨🇭 北çå", "note": "température — très élevée"}))], ""),
            ("hy_v3 opensource",
             "Checking.<tool_calls:opensource>\n<tool_call:opensource>get_weather<tool_sep:opensource>\n<arg_key:opensource>city</arg_key:opensource>\n<arg_value:opensource>\nParis\n</arg_value:opensource>\n</tool_call:opensource>\n</tool_calls:opensource>",
             vec![("get_weather", w("Paris"))], "Checking."),
            ("truncated block",
             "Let me check.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis",
             vec![], "Let me check.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis"),
            ("plain prose",
             "The sky is blue because Rayleigh scattering.",
             vec![], "The sky is blue because Rayleigh scattering."),
            // a `<` the model never completed, and a marker-looking word that is not a call
            ("ends in a marker prefix", "Done <", vec![], "Done <"),
            ("marker-looking text", "see <tool_call_x y> and <thinking>", vec![], "see <tool_call_x y> and <thinking>"),
        ]
    }

    /// (b) The serial fixtures through the REAL stream, in every chunking `t8_chunkings` makes,
    /// content mode and think mode (`plan</think>\n\n` first, the close tag cut anywhere too):
    /// the client's reasoning, content, calls and finish_reason equal the non-streaming
    /// response's AND the fixture's stated outcome. The success signal is asserted per
    /// fixture: the calls it must carry are present, the content it must carry is non-empty
    /// where it should be, and a parsed call's text is never also sent as content.
    #[tokio::test(start_paused = true)]
    async fn the_serial_fixtures_stream_what_the_non_streaming_response_returns() {
        let _complete = crate::tools::test_unclosed(crate::tools::Unclosed::Complete); // F11: the default is drop; the F8 fixtures are about `complete`
        let mut streams = 0usize;
        for (name, raw, calls, content) in t8_fixtures() {
            for thinking in [false, true] {
                let text = if thinking { format!("plan</think>\n\n{raw}") } else { raw.to_string() };
                let want_reasoning = if thinking { "plan" } else { "" };
                let reference = t8_non_stream(thinking, true, &text);
                // the reference itself says what the fixture must produce (not only "agrees")
                let names: Vec<&str> = reference.calls.iter().map(|c| c.0.as_str()).collect();
                assert_eq!(names, calls.iter().map(|c| c.0).collect::<Vec<_>>(), "{name}: reference calls");
                for ((_, got), (_, want)) in reference.calls.iter().zip(&calls) {
                    assert_eq!(got, want, "{name}: reference arguments");
                }
                assert_eq!(reference.finish, if calls.is_empty() { "stop" } else { "tool_calls" }, "{name}");
                assert_eq!(reference.content.trim(), content, "{name}: reference content");
                assert_eq!(reference.reasoning.trim(), want_reasoning, "{name}: reference reasoning");
                for pieces in t8_chunkings(&text) {
                    streams += 1;
                    let got = t8_stream(thinking, true, &pieces).await;
                    let at = format!("{name} / thinking {thinking} / {pieces:?}");
                    assert_eq!(got.calls, reference.calls, "calls: {at}");
                    assert_eq!(got.finish, reference.finish, "finish: {at}");
                    assert_eq!(got.reasoning.trim(), reference.reasoning.trim(), "reasoning: {at}");
                    assert_eq!(got.content.trim(), reference.content.trim(), "content: {at}");
                    // the success signals, stated again on the stream's own bytes
                    assert_eq!(got.content.trim(), content, "stream content: {at}");
                    if !calls.is_empty() {
                        assert!(!got.calls.is_empty() && got.order.ends_with('t'), "the call is the last delta: {at}");
                        assert!(!got.content.contains("<tool_call") && !got.reasoning.contains("<tool_call"),
                                "a parsed call is not also sent as text: {at}");
                    } else {
                        assert!(got.calls.is_empty() && !got.order.contains('t'), "no call: {at}");
                        assert!(!got.content.trim().is_empty(), "the text is not dropped: {at}");
                    }
                }
            }
        }
        assert!(streams > 2000, "the fixtures ran through {streams} streams");
    }

    /// F8: a response cut by the token limit inside a wrapper it never closed returns NO call in
    /// the stream (the finish chunk carries `length`, never `tool_calls`) and none in the
    /// non-streaming response — under every chunking. The same text stopped by itself DOES return
    /// the closed element (the fixtures above, every chunking).
    #[tokio::test(start_paused = true)]
    async fn f8_unclosed_cut_by_length_returns_no_call_in_stream_or_json() {
        let _complete = crate::tools::test_unclosed(crate::tools::Unclosed::Complete); // F11: the default is drop
        let text = "Prose.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>";
        for pieces in t8_chunkings(text) {
            let (tx, rx, bl) = tok_channel();
            let _ = tx.send(TokEvent::Admitted { cached_tokens: 0 });
            for i in 0..pieces.len() { let _ = tx.send(tok(i as u32)); }
            let _ = tx.send(fin("length"));
            let table = pieces.to_vec();
            let sh = Shape { thinking: false, tools: true, usage: false };
            let s = chat_sse_stream(cx(rx, bl, sh, None), move || move |t: u32| table[t as usize].clone());
            let got = wire_of(&data_frames(&read_frames(respond(s, None, idle)).await));
            assert!(got.calls.is_empty(), "cut by length: no call: {pieces:?}");
            assert_eq!(got.finish, "length", "cut by length: finish carried as is: {pieces:?}");
        }
        assert!(crate::tools::parse_served(text, None, "length").tool_calls.is_empty());
        assert_eq!(crate::tools::parse_served(text, None, "stop").tool_calls.len(), 1);
    }

    /// (c) FUZZ CLASS D (the independent review of SURFACE2B, B12: 4 of 300,000 fuzzed streams;
    /// pre-existing on master). REPRODUCTION, PINNED AS IT IS TODAY.
    ///
    /// Input (thinking off, tools offered, the pieces below, each one decode chunk):
    ///   `Hello` | `<tool_call>` | `\n<function=get_weather>…Paris…</function>\n` | `</tool_call>`
    ///   | `<think>` | `hmm`
    /// What the client receives: content `Hello`; then content with the RAW call text
    /// (`<tool_call>…</tool_call>`, sent when the `<think>` arrives); then reasoning `hmm`;
    /// then the `tool_calls` delta carrying the same call; finish `tool_calls`. The call is
    /// delivered twice: as text and as a structured call.
    ///
    /// Verdict: a defect of the stream's hold-back in `chat_sse_stream`, not of `tools::parse`
    /// or `split_think`. In content mode the `<think>` hand-off (`f_open.find(&acc, cs,
    /// think_open)` -> `Some(upto)`) emits `acc[content_emitted..upto]` WITHOUT clamping at a
    /// held tool marker, where every other emit site goes through `hold_back_end` (which
    /// stops at the marker). `content_emitted` is parked AT the marker by the hold-back, so
    /// the whole held call text is inside `acc[content_emitted..upto]` and leaves as content.
    /// The parser then reads the whole `acc` at the finish (no close tag: `split_think` returns
    /// it as content) and finds the call, so the structured call follows. The non-streaming
    /// response for the same text is content `Hello<think>hmm` + the call (`<think>` is
    /// returned as text there: no close tag); it never sends the call as text.
    ///
    /// FIXED (S-B9-TOOLPARSE3 part B item 5, independent review M2): a `<think>` that lies after a
    /// held tool marker is text of the held call, so the hand-off to reasoning takes only a hit
    /// BEFORE the marker. What the client receives now: content `Hello`, no raw call text, the
    /// structured call last. (`hmm`, which follows the stray `<think>` after the call, is held text and
    /// is dropped because a call parsed: as in the non-streaming response.)
    #[tokio::test(start_paused = true)]
    async fn fuzz_class_d_a_think_open_after_a_held_tool_marker_no_longer_sends_the_call_as_content_too() {
        let body = "\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n";
        let raw_call = format!("<tool_call>{body}</tool_call>");
        let pieces: Vec<String> = ["Hello", "<tool_call>", body, "</tool_call>", "<think>", "hmm"]
            .iter().map(|s| s.to_string()).collect();
        let text = pieces.concat();
        let w = t8_stream(false, true, &pieces).await;

        assert_eq!(w.calls, vec![("get_weather".to_string(), serde_json::json!({"city": "Paris"}))]);
        assert_eq!(w.finish, "tool_calls");
        let n = t8_non_stream(false, true, &text);
        assert_eq!(n.calls, w.calls, "non-streaming: the same single call");
        assert_eq!(n.content, "Hello<think>hmm", "non-streaming: the prose, `<think>` left in (no close tag)");
        assert!(!n.content.contains("<tool_call"), "non-streaming never sends the call as text");

        // the raw call text does NOT leave as content any more
        assert!(!w.content.contains(&raw_call) && !w.content.contains("<tool_call"), "the raw call is not content: {:?}", w.content);
        assert_eq!(w.content, "Hello");
        assert_eq!(w.reasoning, "");
        assert_eq!(w.order, "ct", "content `Hello`, then the tool_calls delta");

        // inside ONE decode chunk (the whole reply is one piece)
        let one = t8_stream(false, true, &[text.clone()]).await;
        assert_eq!(one.calls, w.calls);
        assert_eq!((one.content.as_str(), one.reasoning.as_str(), one.order.as_str()), ("Hello", "", "ct"));

        // and after a think block that closed: thinking on, `plan</think>\n\nHello<call><think>hmm`
        let think_pieces: Vec<String> = std::iter::once("plan</think>\n\n".to_string()).chain(pieces.iter().cloned()).collect();
        let t = t8_stream(true, true, &think_pieces).await;
        assert_eq!(t.calls, w.calls);
        assert_eq!((t.reasoning.as_str(), t.content.as_str(), t.order.as_str()), ("plan", "Hello", "rct"));

        // CONTROL: the same reply WITHOUT the `<think>` is held to the end and sent as the call only
        let ctl: Vec<String> = pieces[..4].to_vec();
        let c = t8_stream(false, true, &ctl).await;
        assert_eq!((c.content.as_str(), c.order.as_str()), ("Hello", "ct"), "no `<think>`: no duplicate");
        assert_eq!(c.calls, w.calls);
    }

    /// Item 5, the natural trigger: an ARGUMENT that contains the text `<think>` (a chat template, a
    /// document about reasoning tags, this repository's own server.rs), in EVERY chunking, thinking off
    /// and on, streamed and not: one call whose argument is whole, the prose before it, nothing of the
    /// call's markup in content or reasoning. And an argument that contains `</think>`, with thinking
    /// OFF: the call used to be thrown away (the first close tag was taken for the end of reasoning).
    #[tokio::test(start_paused = true)]
    async fn a_think_tag_inside_an_argument_is_text_of_the_call_in_every_chunking() {
        for (arg, label) in [
            ("The model starts its reasoning with <think> and then answers.", "open tag"),
            ("A close tag </think> inside an argument.", "close tag"),
            ("<think>draft</think> both, as a template shows", "both"),
        ] {
            let call = format!("<tool_call>\n<function=get_weather>\n<parameter=city>\n{arg}\n</parameter>\n</function>\n</tool_call>");
            let want = vec![("get_weather".to_string(), serde_json::json!({"city": arg}))];
            // thinking OFF
            let text = format!("I will write it.\n{call}");
            let n = t8_non_stream(false, true, &text);
            assert_eq!(n.calls, want, "{label}: non-streaming, thinking off");
            assert_eq!(n.content, "I will write it.", "{label}");
            for pieces in t8_chunkings(&text) {
                let w = t8_stream(false, true, &pieces).await;
                assert_eq!(w.calls, want, "{label}: streamed {pieces:?}");
                // (the stream sends the prose before a held call as it arrives, newline included;
                // clients trim it, the non-streaming response does)
                assert_eq!(w.content.trim(), "I will write it.", "{label}: {pieces:?}");
                assert_eq!(w.reasoning, "", "{label}: {pieces:?}");
                assert_eq!(w.finish, "tool_calls");
                assert!(w.order.ends_with('t') && !w.order.contains('r'), "{label}: {pieces:?} order {}", w.order);
            }
            // thinking ON, the block closed first
            let text = format!("plan</think>\n\nI will write it.\n{call}");
            let n = t8_non_stream(true, true, &text);
            assert_eq!(n.calls, want, "{label}: non-streaming, thinking on");
            assert_eq!((n.reasoning.as_str(), n.content.as_str()), ("plan", "I will write it."), "{label}");
            for pieces in t8_chunkings(&text) {
                let w = t8_stream(true, true, &pieces).await;
                assert_eq!(w.calls, want, "{label}: streamed thinking on {pieces:?}");
                assert_eq!((w.reasoning.as_str(), w.content.trim()), ("plan", "I will write it."), "{label}: {pieces:?}");
                assert_eq!(w.finish, "tool_calls");
            }
        }
    }

    /// (d) 1 of 2, STATED AS IT IS TODAY. A tool call drafted INSIDE a think block that is never
    /// closed (the prompt opened it; the model writes its thoughts and jumps to the call without
    /// `</think>`) is RETURNED AS A CALL, in both response modes.
    ///  * non-streaming: `unclosed_think_split(&text, parse_as_tool_calls)` (the narrow guard
    ///    `prompt_opened_think && !think_closed && !unclosed_think_as_content()`): the text up to
    ///    the first `<tool_call` marker is `reasoning_content`, the text from the marker on is
    ///    what `tools::parse` reads: the call becomes `tool_calls`, finish `tool_calls`.
    ///    (`--unclosed-think content` restores the v0.7.3 arm: `split_think` returns the whole
    ///    text as content, the parser reads all of it, the SAME call comes out and the thoughts
    ///    are `content`.)
    ///  * streaming: the reasoning branch holds back from the marker (`f_tool_r`), the thoughts
    ///    leave as `reasoning_content`; at the finish `split_think(&acc)` has no close tag, so
    ///    `tools::parse` reads the whole buffer: the same call, finish `tool_calls`.
    /// Minimal input: the prompt opened the block; the model wrote
    /// `I need the weather.\n<tool_call>\n<function=get_weather>…Athens…</function>\n</tool_call>`
    /// and nothing else.
    #[tokio::test(start_paused = true)]
    async fn a_call_drafted_in_an_unclosed_think_block_is_returned_as_a_call_is_pinned() {
        let text = "I need the weather.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nAthens\n</parameter>\n</function>\n</tool_call>";
        let want = vec![("get_weather".to_string(), serde_json::json!({"city": "Athens"}))];
        let n = t8_non_stream(true, true, text);
        assert_eq!(n.calls, want);
        assert_eq!(n.finish, "tool_calls");
        assert_eq!(n.reasoning, "I need the weather.", "the thoughts are reasoning_content");
        assert_eq!(n.content, "", "and content is empty");
        for pieces in t8_chunkings(text) {
            let s = t8_stream(true, true, &pieces).await;
            assert_eq!(s.calls, want, "{pieces:?}");
            assert_eq!(s.finish, "tool_calls", "{pieces:?}");
            assert_eq!(s.reasoning.trim(), "I need the weather.", "{pieces:?}");
            assert_eq!(s.content, "", "{pieces:?}");
        }
        // the v0.7.3 arm (`--unclosed-think content`): the same call, the thoughts are content
        let (r, c) = split_think(text, "<think>", "</think>");
        assert_eq!((r, c.as_str()), (None, text));
        let p = crate::tools::parse(&c, Some(&t8_tools()));
        assert_eq!((p.tool_calls.len(), p.content.as_str()), (1, "I need the weather."));
        // tool parsing off for the request: nothing is a call; the marker is ordinary reasoning text
        let (r, c) = unclosed_think_split(text, false);
        assert_eq!((r.as_deref(), c.as_str()), (Some(text), ""));
    }

    /// (d) 2 of 2, STATED AS IT IS TODAY. A stream DROPS the prose that follows a parsed call;
    /// the non-streaming response returns it. `tools::parse` (Qwen markup) keeps the prose
    /// before, between and after the calls as `content`; the stream stops sending content at the
    /// first tool marker (the hold-back), and when a call parsed the finish sends the
    /// `tool_calls` delta ONLY (`held_tail` runs only when nothing parsed), so everything after
    /// the marker that is not a call is never sent. The hy_v3 markup is the same in both modes
    /// (`parse_hy3` returns only the prose BEFORE the wrapper).
    /// Minimal input: `<tool_call>…Paris…</tool_call>\nI will report back.`
    /// non-streaming: content `I will report back.`, one call, finish `tool_calls`;
    /// streaming: no content delta at all, one call, finish `tool_calls`.
    #[tokio::test(start_paused = true)]
    async fn a_stream_drops_the_prose_that_follows_a_parsed_call_is_pinned() {
        let call = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";
        let want = vec![("get_weather".to_string(), serde_json::json!({"city": "Paris"}))];
        // prose only after the call
        let text = format!("{call}\nI will report back.");
        let n = t8_non_stream(false, true, &text);
        assert_eq!((n.calls.clone(), n.content.as_str(), n.finish.as_str()), (want.clone(), "I will report back.", "tool_calls"));
        for pieces in t8_chunkings(&text) {
            let s = t8_stream(false, true, &pieces).await;
            assert_eq!(s.calls, want, "{pieces:?}");
            assert_eq!(s.finish, "tool_calls");
            // PINNED: the prose after the call never leaves
            assert_eq!(s.content, "", "dropped: {pieces:?}");
            assert_eq!(s.order, "t", "{pieces:?}");
        }
        // prose before AND after: the stream keeps only the part before
        let text = format!("Checking.\n{call}\nDone.");
        let n = t8_non_stream(false, true, &text);
        assert_eq!(n.content, "Checking.\n\nDone.", "non-streaming: before and after, joined");
        for pieces in t8_chunkings(&text) {
            let s = t8_stream(false, true, &pieces).await;
            assert_eq!(s.calls, want, "{pieces:?}");
            assert_eq!(s.content.trim(), "Checking.", "streaming: only the prose before the call: {pieces:?}");
        }
        // hy_v3 markup: both modes drop the prose after the wrapper, so they agree
        let hy = "Checking.<tool_calls:opensource>\n<tool_call:opensource>get_weather<tool_sep:opensource>\n<arg_key:opensource>city</arg_key:opensource>\n<arg_value:opensource>\nParis\n</arg_value:opensource>\n</tool_call:opensource>\n</tool_calls:opensource>\nAfter.";
        let n = t8_non_stream(false, true, hy);
        assert_eq!((n.calls.len(), n.content.as_str()), (1, "Checking."));
        let s = t8_stream(false, true, &t8_cut(hy, &[3])).await;
        assert_eq!((s.calls.len(), s.content.as_str()), (1, "Checking."));
    }
