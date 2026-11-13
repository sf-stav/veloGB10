//! BASICS (owner 2026-11-02): what every run prints first. Line 1 is the app name and version (also
//! what `--version` prints); then the configuration the run was started with: the command line,
//! the mode, the model and its parameters, the topology and nodes, the serving settings, the prefix
//! cache and speculative decoding. Values typed on the command line are shown as typed; the rest
//! carry "(default)". Pure string builders (unit-tested) plus `print_*`. These are boot banners and
//! stay DIRECT prints: the logq rule.

use serde_json::Value;

pub const APP: &str = "gb10_inference";

/// `gb10_inference v0.7.7 (veloGB10) — commit 32652817, kernel build …, source build …`
pub fn version_line() -> String {
    let commit = option_env!("TUNE_GIT_COMMIT").unwrap_or("unknown");
    let short = commit.get(..8).unwrap_or(commit);
    let tree = match option_env!("TUNE_GIT_DIRTY") {
        Some("0") => "",
        Some("1") => " + uncommitted changes",
        _ => " (tree state unknown)",
    };
    format!("{APP} v{} (veloGB10) — commit {short}{tree}, kernel build {}, source build {}",
            env!("CARGO_PKG_VERSION"), env!("KERNEL_BUILD_ID"), env!("SOURCE_BUILD_ID"))
}

/// The runs that print text and exit before any model work: no configuration block for them.
pub fn is_info_only(args: &[String]) -> bool {
    args.iter().skip(1).any(|a| matches!(a.as_str(), "--help" | "-h" | "--help-advanced" | "--help-diag" | "--print-config"))
}

/// Line 1 of every run. `--print-config` keeps its stdout byte-identical: the line goes to stderr there;
/// `--help` / `--help-advanced` print it as the first line of the help text instead.
pub fn print_banner(args: &[String]) {
    if args.iter().skip(1).any(|a| a == "--print-config") {
        eprintln!("{}", version_line());
    } else if !args.iter().skip(1).any(|a| matches!(a.as_str(), "--help" | "-h" | "--help-advanced")) {
        println!("{}", version_line());
    }
}

/// The configuration block, after the banner (call once the option store is filled).
pub fn print_config(args: &[String]) {
    if is_info_only(args) { return; }
    println!("{}", config_lines(args).join("\n"));
}

fn has(args: &[String], f: &str) -> bool { args.iter().skip(1).any(|a| a == f) }

fn val<'a>(args: &'a [String], f: &str) -> Option<&'a str> { crate::arg_value(args, f) }

/// A typed value as typed, else `<default> (default)`.
fn or_default(typed: Option<&str>, default: &str) -> String {
    match typed { Some(v) => v.to_string(), None => format!("{default} (default)") }
}

/// Which mode the command line selects (the same precedence as `main`'s dispatch for the common ones).
pub fn mode(args: &[String]) -> String {
    if has(args, "--node") { return "TP node (waits for a head; the model and the run options come from the head)".into(); }
    if has(args, "--server") || has(args, "server") { return "server (OpenAI-compatible HTTP API)".into(); }
    if has(args, "--quantize") { return "quantizer (offline)".into(); }
    if has(args, "--perplexity") { return "perplexity tool".into(); }
    let tool = args.iter().skip(1).find(|a| {
        ["--bench", "--probe", "--autotune", "--cached-models", "--dump-", "--capture-", "--sweep-", "--net-test", "--gen-", "--exl3-", "--pp-"]
            .iter().any(|p| a.starts_with(p))
    });
    match tool {
        Some(t) => format!("{} (tool)", t.split('=').next().unwrap_or(t)),
        None => "interactive generation (load the model, generate from a prompt)".into(),
    }
}

/// `--tp` as the dispatch reads it (bare = 2), without exiting on a bad value: the dispatch refuses that.
fn tp_world(args: &[String]) -> Option<String> {
    for (i, a) in args.iter().enumerate() {
        if let Some(v) = a.strip_prefix("--tp=") { return Some(v.to_string()); }
        if a == "--tp" {
            return Some(match args.get(i + 1) { Some(n) if !n.starts_with('-') => n.clone(), _ => "2".into() });
        }
    }
    None
}

fn read_json(path: &std::path::Path) -> Option<Value> {
    serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()
}

/// `qwen4_exp, 48 layers, hidden 2560, 512 experts (10 per token), vocab 248320, trained context 262144`
pub fn model_params(cfg: &Value) -> String {
    let t = cfg.get("text_config").filter(|t| t.is_object()).unwrap_or(cfg);
    let num = |k: &str| t.get(k).and_then(Value::as_u64).or_else(|| cfg.get(k).and_then(Value::as_u64));
    let mut parts = Vec::new();
    if let Some(m) = cfg.get("model_type").and_then(Value::as_str) { parts.push(m.to_string()); }
    if let Some(n) = num("num_hidden_layers") { parts.push(format!("{n} layers")); }
    if let Some(n) = num("hidden_size") { parts.push(format!("hidden {n}")); }
    if let Some(e) = num("num_experts").or_else(|| num("n_routed_experts")) {
        match num("num_experts_per_tok") {
            Some(k) => parts.push(format!("{e} experts ({k} per token)")),
            None => parts.push(format!("{e} experts")),
        }
    }
    if let Some(n) = num("vocab_size") { parts.push(format!("vocab {n}")); }
    if let Some(n) = num("max_position_embeddings") { parts.push(format!("trained context {n}")); }
    if parts.is_empty() { "config.json has none of the usual fields".into() } else { parts.join(", ") }
}

/// The weight format: `quantization_config.json` (EXL3 writes its exact bpw there), else config.json's block.
pub fn weights_desc(qc_file: Option<&Value>, cfg: &Value) -> String {
    let qc = qc_file.or_else(|| cfg.get("quantization_config"));
    let Some(qc) = qc else { return "unquantized (no quantization_config)".into() };
    let s = |k: &str| qc.get(k).and_then(|v| v.as_str().map(str::to_string).or_else(|| v.as_f64().map(|f| f.to_string())));
    match s("quant_method").as_deref() {
        Some("exl3") => {
            let mut d = format!("EXL3 {} bpw", s("bits").unwrap_or_else(|| "?".into()));
            let extra: Vec<String> = [("head_bits", "lm_head"), ("mtp_bits", "MTP head")].iter()
                .filter_map(|(k, name)| s(k).map(|b| format!("{name} {b} bits"))).collect();
            if !extra.is_empty() { d.push_str(&format!(" ({})", extra.join(", "))); }
            d
        }
        Some(m) => {
            let mut d = m.to_string();
            for k in ["format", "quant_algo"] { if let Some(v) = s(k) { d.push_str(&format!(" {v}")); } }
            if let Some(r) = s("recipe") { d.push_str(&format!(" (recipe {r})")); }
            d
        }
        None => "quantized (quant_method not named)".into(),
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Family { Exl3, Trunk, Other }

fn family(dir: &str) -> Family {
    if crate::exl3_serve::is_exl3_pack(dir) { return Family::Exl3; }
    // a single-file EXL3 pack has no index until the boot synthesizes one (cli_opts_bridge, after this block)
    let qc = read_json(&std::path::Path::new(dir).join("quantization_config.json"));
    if qc.and_then(|q| q.get("quant_method").and_then(Value::as_str).map(|m| m == "exl3")).unwrap_or(false) { return Family::Exl3; }
    if crate::naive::config::NaiveConfig::is_naive_dir(dir) { return Family::Other; }
    let p = std::path::Path::new(dir).join("config.json");
    let mt = read_json(&p).and_then(|c| c.get("model_type").and_then(Value::as_str).map(str::to_string));
    match mt.as_deref() { Some(m) if m.starts_with("deepseek") => Family::Other, _ => Family::Trunk }
}

/// The labelled configuration lines (no trailing newline).
pub fn config_lines(args: &[String]) -> Vec<String> {
    let mut out = vec!["configuration:".to_string()];
    let mut row = |k: &str, v: String| out.push(format!("  {k:<14} {v}"));
    let cmd: Vec<String> = args.iter().map(|a| if a.is_empty() || a.contains(' ') { format!("{a:?}") } else { a.clone() }).collect();
    row("command line", cmd.join(" "));
    row("mode", mode(args));
    if has(args, "--node") {
        row("listening", format!("port {}", or_default(val(args, "--port"), "29500")));
        row("log level", crate::opts::get(crate::opt!("log-level")).unwrap_or_else(|| "normal (default)".into()));
        return out;
    }
    let dir = val(args, "--model-dir").map(|d| d.trim_end_matches('/').to_string());
    let fam = dir.as_deref().map(family);
    match dir.as_deref() {
        None => row("model", match val(args, "--model") {
            Some(m) => format!("{m} (legacy --model file)"),
            None => "no --model-dir given".into(),
        }),
        Some(d) => {
            let p = std::path::Path::new(d);
            if !p.is_dir() {
                row("model", format!("{d} (NOT FOUND on this box)"));
            } else {
                row("model", d.to_string());
                match read_json(&p.join("config.json")) {
                    Some(cfg) => {
                        row("model params", model_params(&cfg));
                        row("weights", weights_desc(read_json(&p.join("quantization_config.json")).as_ref(), &cfg));
                    }
                    None => row("model params", "no readable config.json".into()),
                }
            }
        }
    }
    if has(args, "--quantize") {
        row("recipe", or_default(val(args, "--recipe"), "all,gdn:fp8"));
        row("output", val(args, "--out").unwrap_or("no --out given").to_string());
        return out;
    }
    let topo = match tp_world(args) {
        None => "single box (TP=1)".to_string(),
        Some(w) => format!("TP={w}, this box is the head; nodes: {}", match val(args, "--nodes") {
            Some(n) => n.split(',').collect::<Vec<_>>().join(", "),
            None => "found by discovery on the local network (no --nodes)".into(),
        }),
    };
    row("topology", topo);
    let exl3 = fam == Some(Family::Exl3);
    let other = fam == Some(Family::Other);
    let def = |trunk: &str, exl3_v: &str| if other { "per model family".to_string() } else if exl3 { exl3_v.into() } else { trunk.into() };
    let host = crate::opts::get(crate::opt!("host"));
    let max_seq = crate::opts::get(crate::opt!("max-seq-len"));
    let kv = crate::opts::get(crate::opt!("kv-cache"));
    row("serving", format!("{}:{}, max-batch {}, max-seq-len {}, max-tokens {}, KV cache {}",
        host.as_deref().unwrap_or("0.0.0.0"), or_default(val(args, "--port"), "8000"),
        or_default(val(args, "--max-batch"), "8"), or_default(max_seq.as_deref(), "4096"),
        or_default(val(args, "--max-tokens"), &def("8192", "no cap, bounded by the context")),
        or_default(kv.as_deref(), "per model")));
    row("prefix cache", or_default(val(args, "--prefix-cache"), &def("off", "on")));
    let mtp = crate::opts::get(crate::opt!("mtp"));
    let draft = crate::opts::get(crate::opt!("draft-dir"));
    row("speculation", format!("mtp {}, depth {}, source {}, draft dir {}",
        or_default(mtp.as_deref(), "auto"), or_default(val(args, "--mtp-depth"), "auto"),
        or_default(val(args, "--spec-source"), "auto: a DFlash2 drafter when --draft-dir names one, else the model's MTP head"),
        draft.as_deref().unwrap_or("none")));
    row("log level", crate::opts::get(crate::opt!("log-level")).unwrap_or_else(|| "normal (default)".into()));
    row("options set", crate::opts::set_summary());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_line_names_the_app_and_the_crate_version() {
        let v = version_line();
        assert!(v.starts_with(&format!("{APP} v{} (veloGB10) — commit ", env!("CARGO_PKG_VERSION"))), "{v}");
        assert!(v.contains(env!("KERNEL_BUILD_ID")) && v.contains(env!("SOURCE_BUILD_ID")), "{v}");
        assert!(!v.contains('\n'));
    }

    fn a(v: &[&str]) -> Vec<String> { v.iter().map(|s| s.to_string()).collect() }

    #[test]
    fn info_only_runs_get_no_configuration_block() {
        for f in ["--help", "-h", "--help-advanced", "--help-diag", "--print-config"] {
            assert!(is_info_only(&a(&["bin", f])), "{f}");
        }
        assert!(!is_info_only(&a(&["bin", "--server", "--model-dir", "/m"])));
        // a value that happens to read like a flag is still a flag here (argv[0] is skipped)
        assert!(!is_info_only(&a(&["--help"])));
    }

    #[test]
    fn modes_follow_the_dispatch() {
        assert!(mode(&a(&["bin", "--server", "--model-dir", "/m"])).starts_with("server"));
        assert!(mode(&a(&["bin", "server"])).starts_with("server"));
        assert!(mode(&a(&["bin", "--node", "--port", "29500"])).starts_with("TP node"));
        assert!(mode(&a(&["bin", "--quantize", "--model-dir", "/m"])).starts_with("quantizer"));
        assert_eq!(mode(&a(&["bin", "--probe-binv", "--model-dir", "/m"])), "--probe-binv (tool)");
        assert!(mode(&a(&["bin", "--model-dir", "/m"])).starts_with("interactive"));
        assert_eq!(tp_world(&a(&["bin", "--tp"])), Some("2".into()));
        assert_eq!(tp_world(&a(&["bin", "--tp", "4", "--nodes", "x"])), Some("4".into()));
        assert_eq!(tp_world(&a(&["bin", "--tp=4"])), Some("4".into()));
        assert_eq!(tp_world(&a(&["bin", "--tp", "--nodes", "x"])), Some("2".into()));
    }

    #[test]
    fn model_params_and_weights_read_nested_configs() {
        let cfg: Value = serde_json::json!({
            "model_type": "qwen4_exp",
            "text_config": {"num_hidden_layers": 48, "hidden_size": 2560, "num_experts": 512, "num_experts_per_tok": 10,
                            "vocab_size": 248320, "max_position_embeddings": 262144},
            "quantization_config": {"quant_method": "exl3", "bits": 3, "head_bits": 5}
        });
        assert_eq!(model_params(&cfg),
                   "qwen4_exp, 48 layers, hidden 2560, 512 experts (10 per token), vocab 248320, trained context 262144");
        // quantization_config.json carries the exact bpw and wins over config.json's rounded copy
        let qc: Value = serde_json::json!({"quant_method": "exl3", "bits": 3.05, "head_bits": 5, "mtp_bits": 3});
        assert_eq!(weights_desc(Some(&qc), &cfg), "EXL3 3.05 bpw (lm_head 5 bits, MTP head 3 bits)");
        assert_eq!(weights_desc(None, &cfg), "EXL3 3 bpw (lm_head 5 bits)");
        let nv: Value = serde_json::json!({"model_type": "qwen3_5", "num_hidden_layers": 32,
            "quantization_config": {"quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized", "recipe": "all,gdn:fp8"}});
        assert_eq!(model_params(&nv), "qwen3_5, 32 layers");
        assert_eq!(weights_desc(None, &nv), "compressed-tensors nvfp4-pack-quantized (recipe all,gdn:fp8)");
        assert_eq!(weights_desc(None, &serde_json::json!({})), "unquantized (no quantization_config)");
    }

    #[test]
    fn config_block_shows_typed_values_and_marks_defaults() {
        let l = config_lines(&a(&["bin", "--server", "--model-dir", "/nonexistent-basics-test", "--port", "9000",
                                  "--tp", "2", "--nodes", "10.0.0.2:29500,10.0.0.3:29500", "--prefix-cache", "on"]));
        let all = l.join("\n");
        assert_eq!(l[0], "configuration:");
        assert!(all.contains("  command line   bin --server --model-dir /nonexistent-basics-test --port 9000"), "{all}");
        assert!(all.contains("/nonexistent-basics-test (NOT FOUND on this box)"), "{all}");
        assert!(all.contains("TP=2, this box is the head; nodes: 10.0.0.2:29500, 10.0.0.3:29500"), "{all}");
        assert!(all.contains(":9000, max-batch 8 (default)"), "{all}");
        assert!(all.contains("  prefix cache   on\n"), "{all}");
        assert!(all.contains("  speculation    mtp "), "{all}");
        let node = config_lines(&a(&["bin", "--node", "--port", "29500"])).join("\n");
        assert!(node.contains("listening      port 29500") && !node.contains("prefix cache"), "{node}");
        let q = config_lines(&a(&["bin", "--quantize", "--model-dir", "/x", "--out", "/y"])).join("\n");
        assert!(q.contains("recipe         all,gdn:fp8 (default)") && q.contains("output         /y"), "{q}");
    }
}
