//! Source-text test of the bare CLI path (`run_cli` honours the model-dir option); the former inline
//! `prehard1_cli_model_dir_tests` module of `main.rs`. It reads the binary sources through `src_text!`.

    #[test]
    fn bare_cli_honors_model_dir_with_one_clear_failure_line() {
        let src = src_text!("main");
        let start = src.find("fn run_cli").expect("run_cli exists");
        let end = src[start + 1..].find("\nfn ").map(|e| start + 1 + e).unwrap_or(src.len());
        let body = &src[start..end];
        // --model-dir stands in for --model.
        assert!(body.contains(".or_else(|| parse_arg(args, \"--model-dir\"))"),
            "run_cli: --model falls back to --model-dir");
        // The tokenizer defaults INSIDE the resolved dir.
        assert!(body.contains("tokenizer.json\", parse_arg(args, \"--model-dir\")"),
            "run_cli: the tokenizer defaults into --model-dir");
        // ONE clear failure line and exit 2, before any load is attempted.
        assert!(body.contains("model path not found"), "the clear one-line failure");
        assert!(body.contains("std::process::exit(2)"), "exit code 2");
        // The old CWD-relative tokenizer default no longer shadows the derived one.
        assert!(!body.contains("unwrap_or(\"model/tokenizer.json\")"),
            "run_cli: no CWD-relative tokenizer default left");
    }
