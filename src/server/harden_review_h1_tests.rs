//! Inline unit tests of the server module, moved out of `server.rs` verbatim (module `harden_review_h1_tests`).
//! Declared from `mod.rs`; the body keeps its original indentation and its `use super::*` imports.
    use super::*;

    /// LR-4 admit-bound arithmetic: unlimited at 0, refusal starts exactly at cap + max_waiting.
    #[test]
    fn wait_limit_arithmetic_is_exact() {
        assert!(!over_wait_limit(4, 4, 256), "all lanes busy, nobody waiting");
        assert!(!over_wait_limit(4 + 255, 4, 256), "max_waiting-1 waiting is still admissible");
        assert!(over_wait_limit(4 + 256, 4, 256), "the boundary fires: inflight == cap + max_waiting");
        assert!(over_wait_limit(10_000, 4, 256), "far beyond the bound still fires");
        assert!(!over_wait_limit(u64::MAX, 4, 0), "0 = unlimited: never over");
        assert!(!over_wait_limit(0, 0, 0), "0 = unlimited holds at zero inflight too");
    }

    /// LR-4 refusal shape: 503 + `Retry-After: 5` + the engine's error JSON with code
    /// `server_busy`, and the rejection is COUNTED (velogb10_requests_rejected_total).
    #[tokio::test]
    async fn server_busy_response_shape() {
        let before = crate::metrics::render(true);
        let resp = server_busy();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(resp.headers().get(axum::http::header::RETRY_AFTER).unwrap(), "5");
        let body = axum::body::to_bytes(resp.into_body(), 1 << 16).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["code"], "server_busy", "body: {v}");
        assert_eq!(v["error"]["type"], "server_error");
        assert!(v["error"]["message"].as_str().unwrap().contains("server busy"));
        let after = crate::metrics::render(true);
        let cnt = |s: &str| s.lines().find(|l| l.starts_with("velogb10_requests_rejected_total"))
            .unwrap().split_whitespace().last().unwrap().parse::<u64>().unwrap();
        assert!(cnt(&after) > cnt(&before), "server_busy must count the rejection");
        // MAXBATCH1: the 503 is the refused arm of the over-cap arrival class.
        let oc = |s: &str| s.lines().find(|l| l.starts_with("velogb10_over_cap_arrivals_total{disposition=\"rejected\"}"))
            .unwrap().split_whitespace().last().unwrap().parse::<u64>().unwrap();
        assert!(oc(&after) > oc(&before), "server_busy must count the over-cap rejection");
        assert!(after.contains("velogb10_over_cap_arrivals_total{disposition=\"queued\"}"), "the queued series is always present");
    }

    /// LR-5 backlog predicate: fires AT the limit (>=), never below it; 0 = unlimited.
    #[test]
    fn backlog_predicate_fires_at_limit_only() {
        assert!(!over_backlog(0, 8192));
        assert!(!over_backlog(8191, 8192), "one below the limit is admissible");
        assert!(over_backlog(8192, 8192), "AT the limit the sweep fires");
        assert!(over_backlog(99_999, 8192));
        assert!(!over_backlog(99_999, 0), "0 = unlimited");
        assert!(!over_backlog(0, 0));
    }

    /// Handler-level (a): the REAL helper moves the unconsumed counter by exactly ONE per
    /// received event (H0: the double decrement left N-2K).
    #[tokio::test]
    async fn recv_counted_moves_backlog_exactly_once() {
        let (tx, mut rx, bl) = tok_channel();
        for i in 0..10u32 { tx.send(TokEvent::Tok(i)).unwrap(); }
        assert_eq!(tx.len(), 10, "N sent, none consumed");
        for k in 0..3u32 {
            assert!(matches!(recv_counted(&mut rx, &bl).await, Some(TokEvent::Tok(t)) if t == k));
        }
        assert_eq!(tx.len(), 7, "N-K after the handler read K via recv_counted");
        // the sync twin agrees (the bench/gate drains' path)
        assert!(matches!(try_recv_counted(&mut rx, &bl), Some(TokEvent::Tok(_))));
        assert_eq!(tx.len(), 6, "try_recv_counted decrements exactly once too");
    }

    /// Handler-level (b): a reader that consumed 20,000 events and THEN stalls trips the
    /// predicate only when UNCONSUMED events reach the limit — the test that would have
    /// caught both the H0 double decrement and the H2 missing decrement.
    #[tokio::test]
    async fn stalled_reader_trips_only_on_unconsumed_events() {
        let (tx, mut rx, bl) = tok_channel();
        let limit = 8192usize;
        for i in 0..20_000u32 { tx.send(TokEvent::Tok(i)).unwrap(); }
        for _ in 0..20_000 { recv_counted(&mut rx, &bl).await.unwrap(); }
        assert_eq!(tx.len(), 0, "a healthy reader leaves zero unconsumed events");
        assert!(!over_backlog(tx.len(), limit), "consumed history must not count");
        // ...then the reader stalls: 8191 unread are admissible, the 8192nd trips.
        for i in 0..8191u32 { tx.send(TokEvent::Tok(i)).unwrap(); }
        assert!(!over_backlog(tx.len(), limit), "8191 unconsumed < 8192");
        tx.send(TokEvent::Tok(0)).unwrap();
        assert!(over_backlog(tx.len(), limit), "8192 unconsumed trips the sweep");
    }

    /// Handler-level (c): a 70,000-event generation consumed by a healthy reader never trips
    /// even the default limit (65536) — the H2 bug cancelled exactly this non-streaming run.
    #[tokio::test]
    async fn healthy_70k_event_generation_never_trips_default_limit() {
        let (tx, mut rx, bl) = tok_channel();
        for i in 0..70_000u32 { tx.send(TokEvent::Tok(i)).unwrap(); }
        for _ in 0..70_000 { recv_counted(&mut rx, &bl).await.unwrap(); }
        assert_eq!(tx.len(), 0, "70,000 events, all consumed by the helper");
        assert!(!stream_backlog_exceeded(tx.len()),
            "a healthy 70k-token completion must never be cancelled by the sweep");
    }
