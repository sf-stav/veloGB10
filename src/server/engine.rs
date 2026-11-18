//! Engine liveness and the scheduler-to-HTTP channel: `ENGINE_STATE`, the waiting-queue cap, `server_busy`,
//! `TokTx` / `tok_channel` (counted backlog of token events) and the engine-error responses.
//! Siblings: `chat.rs` and `completions.rs` submit work through these; `mod.rs` has the map.

use super::*;

/// WP02 liveness: the serving engine's state, read by /health and admission. One engine per
/// process; a backend marks it DEAD when its scheduler can no longer serve (sticky CUDA error,
/// scheduler thread panic). Backends that never set it stay OK.
pub static ENGINE_STATE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(ENGINE_OK);
pub const ENGINE_OK: u8 = 0;
pub const ENGINE_DEAD: u8 = 1;

pub fn engine_ok() -> bool {
    ENGINE_STATE.load(std::sync::atomic::Ordering::Acquire) == ENGINE_OK
}

pub fn set_engine_dead(why: &str) {
    ENGINE_STATE.store(ENGINE_DEAD, std::sync::atomic::Ordering::Release);
    eprintln!("[engine] state DEAD: {why} (/health and new requests answer 503)");
}

/// LR-4 (REL_LONGRUN): `--max-waiting N` (default 256, 0 = unlimited; a negative or non-numeric
/// value is refused at startup, it used to become 256 silently). The HTTP handlers push
/// every accepted request into an UNBOUNDED scheduler channel; once N or more requests wait
/// beyond the lanes, refuse instead of queueing. H8 (REL_V0_7_3): the counter is the Req
/// guard's IN-FLIGHT count — a request counts for its WHOLE handler lifetime (a stalled
/// reader's request stays counted, not just its queue wait), and a burst can overshoot by the
/// requests admitted between the check and the send. Each queued request pins its prompt and
/// messages until a lane frees; the only other guard is the memwatch kill-switch.
pub fn max_waiting() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| crate::opts::ranged(crate::opt!("max-waiting")).unwrap_or_else(|e| fatal_option(&e)) as usize)
}

/// Pure bound arithmetic (unit-tested): `inflight` counts running + waiting (the exact Req-guard
/// counter); with all lanes busy, waiting = inflight - cap.
pub(super) fn over_wait_limit(inflight: u64, cap: u64, max_waiting: usize) -> bool {
    max_waiting > 0 && inflight >= cap + max_waiting as u64
}

pub(super) fn admit_gate_busy() -> bool {
    over_wait_limit(crate::metrics::inflight(), crate::metrics::max_batch(), max_waiting())
}

/// The LR-4 refusal: 503 in the engine's existing error shape plus `Retry-After: 5`.
pub(super) fn server_busy() -> Response {
    crate::metrics::reject_request();
    // MAXBATCH1: the refused arm of the over-cap arrival class (every lane busy AND the waiting
    // queue at --max-waiting) — paired with `velogb10_over_cap_arrivals_total{disposition="queued"}`.
    crate::metrics::over_cap_rejected();
    (
        StatusCode::SERVICE_UNAVAILABLE,
        [(axum::http::header::RETRY_AFTER, "5")],
        Json(serde_json::json!({"error": {
            "message": "server busy: too many requests are waiting for a lane; retry shortly",
            "type": "server_error", "code": "server_busy",
        }})),
    ).into_response()
}

/// LR-5 mechanism: the per-request event sender with an UNCONSUMED-EVENT counter. Every `send`
/// increments, the HTTP handler's receive loops decrement, and the scheduler's cancel sweeps
/// read `len()`. (tokio 1.52 exposes `len()` only on the RECEIVER, which lives inside the
/// handler task — exactly the task that is blocked when a stream reader stalls — so the count
/// rides with the sender.) Drop-in for `UnboundedSender<TokEvent>` at every existing call site.
#[derive(Clone)]
pub struct TokTx {
    tx: tokio::sync::mpsc::UnboundedSender<TokEvent>,
    backlog: std::sync::Arc<std::sync::atomic::AtomicIsize>,
}

impl TokTx {
    pub fn send(&self, ev: TokEvent) -> Result<(), tokio::sync::mpsc::error::SendError<TokEvent>> {
        let r = self.tx.send(ev);
        if r.is_ok() { self.backlog.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
        r
    }
    pub fn is_closed(&self) -> bool { self.tx.is_closed() }
    /// Unconsumed events (events sent minus events received by the handler).
    pub fn len(&self) -> usize {
        self.backlog.load(std::sync::atomic::Ordering::Relaxed).max(0) as usize
    }
}

/// The handler-side counter handle (decrement once per received event).
pub type TokBacklog = std::sync::Arc<std::sync::atomic::AtomicIsize>;

pub fn tok_channel()
    -> (TokTx, tokio::sync::mpsc::UnboundedReceiver<TokEvent>, TokBacklog) {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<TokEvent>();
    let backlog: TokBacklog = std::sync::Arc::new(std::sync::atomic::AtomicIsize::new(0));
    (TokTx { tx, backlog: std::sync::Arc::clone(&backlog) }, rx, backlog)
}

/// THE single LR-5 receive mechanism (REL_V0_7_3 review H0/H2/H6): receive the next event and
/// decrement the per-request UNCONSUMED-event counter exactly once. Every handler receive loop
/// goes through this helper (the sync bench/gate drains use `try_recv_counted`) — a loop that
/// forgets the decrement makes a HEALTHY long generation look stalled (the scheduler sweep
/// cancels it at `--stream-backlog-events`), and a loop that decrements twice cancels it at
/// half the limit. Exactly one `fetch_sub` on the backlog may exist in the tree: here.
pub async fn recv_counted(
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<TokEvent>,
    bl: &TokBacklog,
) -> Option<TokEvent> {
    counted_event(rx.recv().await, bl)
}

/// Sync twin of `recv_counted` for the bench/gate drains (`run_spec_bench`, `wp24_gate_arm`,
/// `spec_arm`): the same single decrement, `try_recv`-based, so a healthy drain cannot be
/// cancelled by the backlog sweep either.
pub fn try_recv_counted(
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<TokEvent>,
    bl: &TokBacklog,
) -> Option<TokEvent> {
    counted_event(rx.try_recv().ok(), bl)
}

pub(super) fn counted_event(ev: Option<TokEvent>, bl: &TokBacklog) -> Option<TokEvent> {
    if ev.is_some() {
        bl.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
    ev
}

/// LR-5 (REL_LONGRUN): `--stream-backlog-events N` (default 65536, 0 = unlimited; a negative or
/// non-numeric value is refused at startup, it used to become 65536 silently). A stream
/// client that stops reading makes its per-request channel grow for the whole generation; past
/// this many UNCONSUMED events the scheduler cancels the request exactly like a client
/// disconnect (the lane sweep / wire Cancel), freeing the GPU lane.
pub fn stream_backlog_events() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| crate::opts::ranged(crate::opt!("stream-backlog-events")).unwrap_or_else(|e| fatal_option(&e)) as usize)
}

/// Pure predicate (unit-tested); `len` is the TokTx send-minus-receive counter (unconsumed
/// events) — tokio's UnboundedSender exposes no len(); ours rides with the sender.
pub(super) fn over_backlog(len: usize, limit: usize) -> bool {
    limit > 0 && len >= limit
}

pub fn stream_backlog_exceeded(len: usize) -> bool {
    over_backlog(len, stream_backlog_events())
}

/// WP02: 503 for a request the engine cannot take (DEAD, or its scheduler channel is gone).
pub(super) fn engine_unavailable() -> Response {
    (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"error": {
        "message": "engine stopped: the inference engine hit a fatal error and cannot serve; restart the server",
        "type": "server_error", "code": "engine_unavailable",
    }}))).into_response()
}

/// WP02: HTTP status for an engine error reason (busy / stopped = 503, anything else = 500).
pub(super) fn engine_error_status(reason: &str) -> StatusCode {
    // VIS-2: an engine refusal the CLIENT caused carries its class in the reason
    if reason.starts_with("error: bad request:") { StatusCode::BAD_REQUEST }
    else if reason.starts_with("error: unprocessable:") { StatusCode::UNPROCESSABLE_ENTITY }
    else if reason.contains("busy") || reason.contains("engine stopped") { StatusCode::SERVICE_UNAVAILABLE }
    else { StatusCode::INTERNAL_SERVER_ERROR }
}

/// WP02: the SSE error event for a generation that ended in an engine error (OpenAI SDKs raise
/// on a chunk carrying `error`); the final chunk after it carries a spec finish_reason.
pub(super) fn sse_error_event(reason: &str) -> String {
    serde_json::json!({"error": {"message": reason, "type": "server_error",
                                 "code": engine_error_status(reason).as_u16()}}).to_string()
}

