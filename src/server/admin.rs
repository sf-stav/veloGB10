//! Operational routes: `/metrics`, the copy-record routes and `/health` (503 once the engine is dead).
//! Siblings: `tokenize.rs` holds the tokenizer routes; `engine.rs` owns the liveness state read here.

use super::*;

/// Phase-9 A.2 — the status route now carries the engine's OWN per-window decode telemetry
/// (`crate::tel`): mode (mtp|dflash2), tp width, df2 block, chosen depth, accept@k, yield
/// (tokens per verify forward), step p50/p90. Lock-free read; it never touches a decode step.
/// Clients (owner harness, accept_gate.py) use it to see TRUE alpha instead of inferring it
/// from wall-clock tokens. `status` stays "ok" so existing liveness probes are unaffected.
/// Prometheus text exposition (CF-P1b). Cheap to scrape: atomics read, one string built.
pub(super) async fn metrics() -> Response {
    ([(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
     crate::metrics::render(engine_ok())).into_response()
}

/// COPYREC1: `GET /v1/copy-record` — the record's state, size and landing rates (atomics; never touches a step).
pub(super) async fn copy_record_get() -> Response {
    Json(crate::copy_record::state_json()).into_response()
}

/// COPYREC1: `DELETE /v1/copy-record` — the scheduler empties the record (memory and its chunk files; under TP on every
/// rank) before it admits the next request: 202 with the state; 409 when the record is off or read-only.
pub(super) async fn copy_record_delete() -> Response {
    match crate::copy_record::request_clear() {
        Ok(()) => (StatusCode::ACCEPTED, Json(serde_json::json!({
            "emptied": "before the next request is admitted (every TP rank)", "state": crate::copy_record::state_json()}))).into_response(),
        Err(why) => (StatusCode::CONFLICT, Json(serde_json::json!({"error": {"message": why, "type": "invalid_request_error"}}))).into_response(),
    }
}

pub(super) async fn health() -> Response {
    // WP02: a DEAD engine answers 503 (a supervisor / load balancer acts on it)
    if !engine_ok() {
        return (StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"status": "dead", "telemetry": crate::tel::snapshot_json()}))).into_response();
    }
    Json(serde_json::json!({"status": "ok", "telemetry": crate::tel::snapshot_json()})).into_response()
}

