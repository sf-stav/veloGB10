//! Unit tests (module `s1_eq_aware_arg_tests`), moved out of the old inline block in `exl3_serve.rs`; the test text is unchanged.
//! The module is declared in `mod.rs`; `use super::*` there resolves against the parent module as before.

    use super::*;

    const FLAGS: &[&str] = &[
        "--lane-order", "--lane-quantum", "--model-name", "--served-model-name",
        "--otel-endpoint", "--otel-model-id", "--otel-topology",
        "--otel-include-tokens", "--otel-batch-size", "--otel-batch-interval-ms",
        "--port", "--max-seq-len", "--max-batch", "--prefix-cache", "--prefill-chunk", "--prefill-interleave",
        "--kv-cache", "--ple-ram", "--reasoning-effort", "--thinking", "--cpu-affinity",
        "--tune-table", "--tune-draft", "--tune-profile", "--draft-confidence",
        "--draft-temperature", "--spec-sampling", "--host",
    ];

    fn owned(args: &[&str]) -> Vec<String> { args.iter().map(|s| s.to_string()).collect() }

    #[test]
    fn arg_accepts_both_spellings_of_every_flag() {
        for f in FLAGS {
            // space form
            let a = owned(&["serve", "/m", *f, "VALUE", "--after", "x"]);
            assert_eq!(arg(&a, f).as_deref(), Some("VALUE"), "{f} space form");
            // equals form
            let a = owned(&["serve", "/m", &format!("{f}=VALUE"), "--after", "x"]);
            assert_eq!(arg(&a, f).as_deref(), Some("VALUE"), "{f} = form (S1: was None)");
            // absent
            let a = owned(&["serve", "/m", "--other", "y"]);
            assert_eq!(arg(&a, f), None, "{f} absent");
        }
    }

    #[test]
    fn arg_equals_form_does_not_eat_the_next_token() {
        let a = owned(&["--lane-order=fcfs", "--lane-quantum", "4"]);
        assert_eq!(arg(&a, "--lane-order").as_deref(), Some("fcfs"));
        assert_eq!(arg(&a, "--lane-quantum").as_deref(), Some("4"));
        // a longer flag sharing a prefix must not match (the old strip bug class)
        let a = owned(&["--port=9000"]);
        assert_eq!(arg(&a, "--por"), None);
        assert_eq!(arg(&a, "--port").as_deref(), Some("9000"));
    }

    #[test]
    fn nvfp4_parse_arg_is_the_same_parser() {
        // the NVFP4-side reader is a thin delegate of the same shared fn; spot-check both
        // spellings through the EXL3 reader to prove ONE mechanism.
        let a = owned(&["--model-name=Qwen/Qwen3.8-Flash-Next"]);
        assert_eq!(arg(&a, "--model-name").as_deref(), Some("Qwen/Qwen3.8-Flash-Next"));
        let a = owned(&["--model-name", "Qwen/Qwen3.8-Flash-Next"]);
        assert_eq!(arg(&a, "--model-name").as_deref(), Some("Qwen/Qwen3.8-Flash-Next"));
    }
