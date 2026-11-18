//! Inline unit tests of the server module, moved out of `server.rs` verbatim (module `s10_models_route_tests`).
//! Declared from `mod.rs`; the body keeps its original indentation and its `use super::*` imports.
    #[tokio::test]
    async fn models_route_accepts_raw_slash_and_encoded_ids() {
        use axum::routing::get;
        async fn echo_id(axum::extract::Path(id): axum::extract::Path<String>) -> String { id }
        let app = axum::Router::new()
            .route("/v1/models/*id", get(echo_id));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); });
        let fetch = |path: String| async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
            let req = format!("GET {path} HTTP/1.1\r\nhost: t\r\nconnection: close\r\n\r\n");
            s.write_all(req.as_bytes()).await.unwrap();
            let mut buf = String::new();
            s.read_to_string(&mut buf).await.unwrap();
            buf
        };
        // raw slash (the EXL3 default id is slashed now): 200 + the full id extracts
        let r = fetch("/v1/models/Qwen/Qwen3.8-Flash-Next".into()).await;
        assert!(r.starts_with("HTTP/1.1 200"), "raw slash must route (S10): {r}");
        assert!(r.ends_with("Qwen/Qwen3.8-Flash-Next"), "body: {r}");
        // the percent-encoded form keeps working exactly as before
        let r = fetch("/v1/models/Qwen%2FQwen3.8-Flash-Next".into()).await;
        assert!(r.starts_with("HTTP/1.1 200"), "encoded form must keep routing: {r}");
        assert!(r.ends_with("Qwen/Qwen3.8-Flash-Next"), "body: {r}");
    }
