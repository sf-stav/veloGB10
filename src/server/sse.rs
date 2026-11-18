//! SSE wire framing: hand-built chunk strings (role / content / reasoning / final), the JSON escape `esc`,
//! the keep-alive machinery (frames, stall guard, `--stall-drill`) and the serve-option boot lines.
//! Siblings: `chat_stream.rs` (the chat generator that emits these frames), `completions.rs` (the same for /v1/completions).

use super::*;

/// S8 (REL_V0_7_3 review): the hand-built SSE chunk builders, module-level so the escaping
/// is unit-testable. The MODEL ID is escaped exactly like the text: --model-name is honoured
/// verbatim now, and a quote or backslash in it would break every chunk.
pub(super) fn role_chunk(cid: &str, created: i64, model: &str) -> String {
    format!("{{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\"}},\"finish_reason\":null}}]}}",
        esc(cid), created, esc(model))
}

pub(super) fn content_chunk(cid: &str, created: i64, model: &str, text: &str) -> String {
    format!("{{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{}\"}},\"finish_reason\":null}}]}}",
        esc(cid), created, esc(model), esc(text))
}

pub(super) fn reasoning_chunk(cid: &str, created: i64, model: &str, text: &str) -> String {
    format!("{{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"reasoning_content\":\"{}\"}},\"finish_reason\":null}}]}}",
        esc(cid), created, esc(model), esc(text))
}

/// finish is the spec finish-reason; stop_field is the pre-rendered stop_reason JSON field
/// suffix (empty when the loop detector did not end the stream) — WP08.
pub(super) fn final_chunk(cid: &str, created: i64, model: &str, finish: &str, stop_field: &str) -> String {
    format!("{{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"{}\"{}}}]}}",
        esc(cid), created, esc(model), esc(finish), stop_field)
}

/// The comment text of the keep-alive frame: the wire bytes are `: keep-alive\n\n`. An SSE
/// comment is a line that starts with ':'; it has no `data:` field, so no JSON and no `choices`.
pub(super) const SSE_KEEPALIVE_TEXT: &str = "keep-alive";

/// S-B9-REL-BENCHFIX2 (W4): keep-alive frames must not hide a hung engine. With a comment frame
/// every N seconds a client's idle-read timeout never fires, so a request stuck behind a
/// scheduler that no longer runs would hold its client open for ever (before the keep-alive
/// existed, that timeout ended it). The rule (`keepalive_live`): a frame is sent only while
///   * the scheduler COMPLETED a unit of work (a decode round or a prefill chunk:
///     `metrics::sched_touch`, called by both schedulers) within the last `SSE_KEEPALIVE_STALL`, or
///   * the stream itself is younger than that AND the scheduler is not marked busy (it was idle
///     and is picking this request up: its last completed work may be hours old).
/// Why progress and not a ceiling on total silence: legitimate silence has no useful upper bound
/// (a 262K-token prefill is minutes; `--max-waiting` allows 256 requests queued behind it), but
/// legitimate silence always shows the scheduler completing work every few seconds (a 4095-row
/// prefill chunk takes ~3 s on one GB10, a decode round ~50 ms), and a hung scheduler completes
/// nothing. 60 s is 20 times the measured chunk time. A unit of work that really takes longer
/// only pauses the frames (the stream is then silent exactly as it was before keep-alives
/// existed) and they resume when it completes.
pub(super) const SSE_KEEPALIVE_STALL: std::time::Duration = std::time::Duration::from_secs(60);

/// `--sse-keepalive N` (seconds; 1..=3600, 0 = off; the default is the registry row's): the
/// longest silence on an open stream of a live engine. Nothing but these comment frames is on
/// the wire during a queue wait or a prefill (a 30 s prefill carries two of them at 10 s).
/// Resolved once; `serve_options_boot_line` resolves it before the server listens.
pub(crate) fn sse_keepalive() -> Option<std::time::Duration> {
    static V: std::sync::OnceLock<Option<std::time::Duration>> = std::sync::OnceLock::new();
    *V.get_or_init(|| resolve_sse_keepalive().unwrap_or_else(|e| fatal_option(&e)))
}

/// Pure over the option store (unit-tested): Ok(None) = off.
pub(super) fn resolve_sse_keepalive() -> Result<Option<std::time::Duration>, String> {
    let secs = crate::opts::ranged(crate::opt!("sse-keepalive"))? as u64;
    Ok((secs > 0).then(|| std::time::Duration::from_secs(secs)))
}

/// An option value the registry should have refused at startup reached a server reader: stop
/// with the message instead of serving on a silently substituted default.
pub(super) fn fatal_option(e: &str) -> ! {
    eprintln!("FATAL: {e}");
    crate::logq::flush_and_exit(std::time::Duration::from_millis(300), 2)
}

/// "[serve] sse keep-alive: every 10 s" / "... off" (the boot line; unit-tested).
pub(super) fn sse_keepalive_boot_line(every: Option<std::time::Duration>) -> String {
    match every {
        Some(d) => format!("[serve] sse keep-alive: every {} s (--sse-keepalive; a comment frame on a silent stream while \
                            the whole scheduler completed work, any request's decode round or prefill chunk, in the last {} s; \
                            a stream opened on a hung engine can still get frames for its first {} s)",
                           d.as_secs(), SSE_KEEPALIVE_STALL.as_secs(), SSE_KEEPALIVE_STALL.as_secs()),
        None => "[serve] sse keep-alive: off (--sse-keepalive 0)".to_string(),
    }
}

/// Resolve every HTTP-layer option that has a range BEFORE the server listens (a bad value is a
/// refusal at boot, never a silent default at the first request) and return the boot line.
pub(super) fn serve_options_boot_line() -> Result<String, String> {
    let every = resolve_sse_keepalive()?;
    crate::opts::ranged(crate::opt!("max-waiting"))?;
    crate::opts::ranged(crate::opt!("stream-backlog-events"))?;
    let mut line = sse_keepalive_boot_line(every);
    // S-B9-TOOLPARSE3: which parser rules this server serves, and that the others are only counted
    // (the liveness receipt of the shadow counters: `velogb10_toolcall_rules_info` says the same)
    line += &format!("\n[serve] tool-call parser: values end by --tool-quote-pairing {}, blocks end by --tool-block-end {}, unclosed blocks by --tool-unclosed {}; the other cheap rules \
                      (quote pairing, the block rule) are also read on each response that holds a <tool_call> block (text up to {} KiB, \
                      up to {} blocks, one more linear pass per rule) and the disagreements are COUNTED (velogb10_toolcall_*); the cost \
                      model is {} the shadow; wherever it runs it spends at most {} work units per response ({} per block; about {} ms of one \
                      core), and a block over that is read by the v0.7.3 rule and counted; nothing of the shadow is served",
                     crate::opts::choice(crate::opt!("tool-quote-pairing")), crate::opts::choice(crate::opt!("tool-block-end")),
                     crate::opts::choice(crate::opt!("tool-unclosed")),
                     crate::tools::SHADOW_MAX_BYTES / 1024, crate::tools::SHADOW_MAX_BLOCKS,
                     if crate::opts::choice(crate::opt!("tool-shadow-cost")) == "on" { "ALSO read in" } else { "NOT read in (--tool-shadow-cost off)" },
                     crate::tools::cost::RESPONSE_UNITS, crate::tools::cost::BLOCK_UNITS, crate::tools::cost::response_ms_nominal());
    // TOOLSCRUB1: one line naming the scrub mode (off = the default = no served byte changes).
    line += &format!("\n{}", crate::tool_scrub::boot_line(crate::tool_scrub::Mode::from_opt()));
    if let Some((round, secs)) = parse_stall_drill(crate::opts::get(crate::opt!("inject-stall")).as_deref())? {
        line += &format!("\n[serve] DIAGNOSTIC --inject-stall {round}:{secs}: the scheduler thread will sleep {secs} s at round {round} \
                          (a drill: never on a server that serves users)");
    }
    Ok(line)
}

/// The same resolution without the line: `main` calls it right after the command line is parsed,
/// so a value the registry cannot refuse by type (`--inject-stall`) is refused within
/// milliseconds of the start, before anything is loaded. (`http_listen` resolves again, for the
/// line, when the server binds.)
pub fn check_serve_options(args: &[String]) -> Result<(), String> {
    serve_options_boot_line()?;
    stall_drill_topology(crate::opts::get(crate::opt!("inject-stall")).is_some(), args)
}

/// `--inject-stall` is a one-box drill: it puts ONE process's scheduler thread to sleep. On a TP
/// rank (head or node) the other ranks' watchdogs would declare the sleeping one dead, and the
/// option is per box (`Scope::Local`), so a node started with it would keep it whatever the head
/// says. Refused at startup with any TP role on the command line.
pub(super) fn stall_drill_topology(drill_set: bool, args: &[String]) -> Result<(), String> {
    if !drill_set { return Ok(()); }
    match args.iter().find(|a| matches!(a.as_str(), "--tp" | "--head" | "--node") || a.starts_with("--tp=")) {
        Some(role) => Err(format!("--inject-stall is a one-box drill and is refused with {} (the sleeping rank would stall \
                                   the others and their watchdogs would declare it dead)", role.split('=').next().unwrap_or(role))),
        None => Ok(()),
    }
}

/// The SSE response of both streaming endpoints: `stream` plus, when `every` is set, the
/// keep-alive frames of `keepalive_frames` (liveness = the scheduler's work age). The frame shape
/// is `--sse-keepalive-frame`: the comment line (default) or a data frame (LANERES1).
pub(super) fn sse_with_keepalive<S, E>(stream: S, every: Option<std::time::Duration>, frame: SseKeepAliveFrame) -> Response
where
    S: Stream<Item = Result<Event, E>> + Send + 'static,
    E: Into<axum::BoxError> + Send + 'static,
{
    match every {
        Some(d) => Sse::new(keepalive_frames(stream, d, frame, SSE_KEEPALIVE_STALL, crate::metrics::sched_state)).into_response(),
        None => Sse::new(stream).into_response(),
    }
}

/// The keep-alive rule (pure; `SSE_KEEPALIVE_STALL` has the reason): may a comment frame be sent
/// on a stream of age `stream_age`, given the scheduler's gauges?
pub(super) fn keepalive_live(sched: crate::metrics::SchedState, stream_age: std::time::Duration, stall: std::time::Duration) -> bool {
    sched.work_age.map_or(false, |w| w <= stall) || (!sched.busy && stream_age <= stall)
}

/// `inner` plus one comment frame after each `every` of silence (the timer restarts at every
/// event, so a stream that is producing tokens carries none), as long as `keepalive_live` says
/// the engine is alive (`sched()` = the scheduler's gauges at that moment). The data events pass
/// through untouched, in order. This replaces axum's `KeepAlive`, which cannot be conditional
/// and computes `Instant::now() + interval` unchecked at every event.
pub(super) fn keepalive_frames<S, E, F>(inner: S, every: std::time::Duration, frame: SseKeepAliveFrame, stall: std::time::Duration, sched: F)
    -> impl Stream<Item = Result<Event, E>> + Send + 'static
where
    S: Stream<Item = Result<Event, E>> + Send + 'static,
    E: Send + 'static,
    F: Fn() -> crate::metrics::SchedState + Send + 'static,
{
    async_stream::stream! {
        tokio::pin!(inner);
        let born = tokio::time::Instant::now();
        // an interval too large to add to a clock means "no frame", never a panic
        let after = |t: tokio::time::Instant| t.checked_add(every);
        let mut deadline = after(born);
        loop {
            let next = match deadline {
                Some(d) => tokio::time::timeout_at(d, inner.next()).await.ok(),
                None => Some(inner.next().await),
            };
            match next {
                Some(Some(item)) => { yield item; }
                Some(None) => break,
                None => {
                    if keepalive_live(sched(), born.elapsed(), stall) {
                        yield Ok(match &frame {
                            SseKeepAliveFrame::Comment => Event::default().comment(SSE_KEEPALIVE_TEXT),
                            SseKeepAliveFrame::Data(payload) => Event::default().data(payload.as_ref()),
                        });
                    }
                }
            }
            deadline = after(tokio::time::Instant::now());
        }
    }
}

/// LANERES1 `--sse-keepalive-frame`: the shape of the idle keep-alive frames. `Comment` = today:
/// the SSE comment line `: keep-alive` (no data field; SSE clients ignore it). `Data` = an SSE
/// data frame carrying a chat-completions chunk with an EMPTY `choices` array: SDKs whose stream
/// guard counts only chunks with a choices entry never yield it to the caller, but client idle
/// timers fed by received SSE data frames reset (andrei.gnezdilov: qwen-code is such a client).
/// The house rule stands: no choices ENTRY is ever sent before the request's first real token —
/// the keep-alive chunk's `choices` array is empty.
#[derive(Clone, Debug)]
pub(crate) enum SseKeepAliveFrame {
    Comment,
    /// The prebuilt JSON payload (this stream's id/created/model; `choices: []`).
    Data(std::sync::Arc<str>),
}

/// The configured mode (--sse-keepalive-frame; a registry-validated Choice, default comment).
pub(super) fn sse_keepalive_frame_mode() -> &'static str {
    static V: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
    V.get_or_init(|| match crate::opts::choice(crate::opt!("sse-keepalive-frame")).as_str() {
        "data" => "data",
        _ => "comment",
    })
}

/// The pure payload of a data-mode keep-alive frame (unit-tested): a chunk with an EMPTY choices
/// array and the stream's own id/created/model.
pub(super) fn sse_keepalive_data_payload(id: &str, created: i64, model: &str, object: &str) -> String {
    serde_json::json!({
        "id": id, "object": object, "created": created, "model": model, "choices": [],
    }).to_string()
}

/// Pure over the mode string (unit-tested): "data" → the data frame, anything else → comment.
pub(super) fn sse_keepalive_frame_of_mode(mode: &str, id: &str, created: i64, model: &str, object: &str) -> SseKeepAliveFrame {
    match mode {
        "data" => SseKeepAliveFrame::Data(sse_keepalive_data_payload(id, created, model, object).into()),
        _ => SseKeepAliveFrame::Comment,
    }
}

/// The frame argument of `sse_with_keepalive` for THIS stream (id/created/model at the call site).
pub(super) fn sse_keepalive_frame(id: &str, created: i64, model: &str, object: &str) -> SseKeepAliveFrame {
    sse_keepalive_frame_of_mode(sse_keepalive_frame_mode(), id, created, model, object)
}

/// `--inject-stall ROUND:SECONDS` (diagnostic; unset = no drill): Ok(None) when unset. ROUND is
/// the scheduler round counted from boot (velogb10_scheduler_steps_total + 1 is the next one),
/// SECONDS 1..=600.
pub(super) fn parse_stall_drill(v: Option<&str>) -> Result<Option<(u64, u64)>, String> {
    let Some(v) = v else { return Ok(None) };
    let bad = || format!("--inject-stall takes ROUND:SECONDS, ROUND >= 1 and SECONDS from 1 to 600 (got '{v}')");
    let (r, s) = v.split_once(':').ok_or_else(bad)?;
    match (r.trim().parse::<u64>(), s.trim().parse::<u64>()) {
        (Ok(r), Ok(s)) if r >= 1 && (1..=600).contains(&s) => Ok(Some((r, s))),
        _ => Err(bad()),
    }
}

pub(super) fn stall_drill_cfg() -> Option<(u64, u64)> {
    static V: std::sync::OnceLock<Option<(u64, u64)>> = std::sync::OnceLock::new();
    *V.get_or_init(|| parse_stall_drill(crate::opts::get(crate::opt!("inject-stall")).as_deref())
        .unwrap_or_else(|e| fatal_option(&e)))
}

/// The W4 drill, called by `metrics::sched_step` at the start of every scheduler round, on the
/// scheduler thread: with `--inject-stall ROUND:SECONDS` that thread sleeps SECONDS once, at
/// round ROUND. Nothing completes meanwhile (no decode round, no prefill chunk, no admission),
/// which is what a hung scheduler looks like to the HTTP layer; the HTTP task itself keeps
/// running (/health, /metrics, the keep-alive timers). Unset: one OnceLock read per round.
pub(crate) fn stall_drill(round: u64) {
    if let Some((at, secs)) = stall_drill_cfg() {
        if round == at {
            crate::reprintln!("[drill] --inject-stall: the scheduler thread sleeps {secs} s at round {at}: no work completes; \
                               keep-alive frames must stop once the last completed work is {} s old", SSE_KEEPALIVE_STALL.as_secs());
            std::thread::sleep(std::time::Duration::from_secs(secs));
            crate::reprintln!("[drill] --inject-stall: the scheduler thread resumes after {secs} s");
        }
    }
}

pub(super) fn esc(t: &str) -> String {
    // SSE chunks interpolate this inside a JSON string. Hand-escaping only backslashes,
    // quotes and newlines left literal tabs (common in Go), carriage returns and other
    // control characters in the payload. Clients discard that invalid JSON, which looks
    // like the first characters of indented lines were truncated. Use the complete JSON
    // escaping rules (PR #4), then remove the surrounding quotes supplied by the serializer.
    let quoted = serde_json::to_string(t).expect("serializing a Rust string cannot fail");
    quoted[1..quoted.len() - 1].to_string()
}

