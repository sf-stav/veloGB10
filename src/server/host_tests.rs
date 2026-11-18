//! Inline unit tests of the server module, moved out of `server.rs` verbatim (module `host_tests`).
//! Declared from `mod.rs`; the body keeps its original indentation and its `use super::*` imports.
    use super::http_bind_addr;

    /// `--host`: default every interface; IPv4, bracketed/unbracketed IPv6 and `localhost` resolve; junk is refused
    /// with a message naming the flag (one test, so the shared option store is not raced).
    #[test]
    fn host_flag_resolves_and_refuses() {
        let o = crate::opt!("host");
        crate::opts::unset(o);
        assert_eq!(http_bind_addr(9000).unwrap().to_string(), "0.0.0.0:9000");
        crate::opts::set(o, "127.0.0.1");
        assert_eq!(http_bind_addr(9000).unwrap().to_string(), "127.0.0.1:9000");
        crate::opts::set(o, "[::1]");
        assert_eq!(http_bind_addr(9000).unwrap().to_string(), "[::1]:9000");
        crate::opts::set(o, "localhost");
        assert!(http_bind_addr(9000).unwrap().ip().is_loopback());
        crate::opts::set(o, "not a host!");
        assert!(http_bind_addr(9000).unwrap_err().to_string().contains("--host"));
        crate::opts::unset(o);
    }
