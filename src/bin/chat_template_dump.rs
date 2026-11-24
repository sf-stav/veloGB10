//! Offline chat-template render dump (no GPU, no model weights): renders a fixed prompt set through
//! the engine's shared tokenizer + jinja path for every template source named on the command line and
//! prints one JSON line per (template, case, mode). Used to prove that a change to
//! `src/tokenizer/jinja.rs` leaves every other family's rendered prompt byte-identical:
//! build once before the change, once after, diff the two outputs.
//!
//! usage: chat_template_dump --fallback-tokenizer <tokenizer.json> <pack dir | file.jinja>...
use gb10_inference::tokenizer::{ChatMessage, QwenTokenizer, ThinkingMode};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

fn template_of(src: &Path) -> Option<(String, Option<PathBuf>)> {
    if src.is_file() {
        return Some((std::fs::read_to_string(src).ok()?, None));
    }
    let tk = src.join("tokenizer.json");
    let tk = if tk.exists() { Some(tk) } else { None };
    let j = src.join("chat_template.jinja");
    if j.exists() { return Some((std::fs::read_to_string(j).ok()?, tk)); }
    let raw = std::fs::read_to_string(src.join("tokenizer_config.json")).ok()?;
    let v: Value = serde_json::from_str(&raw).ok()?;
    v.get("chat_template")?.as_str().map(|s| (s.to_string(), tk))
}

fn cases() -> Vec<(&'static str, Value, Option<Value>)> {
    let tools = json!([
        {"type":"function","function":{"name":"get_weather","description":"Get the weather for a city",
          "parameters":{"type":"object","properties":{"city":{"type":"string","description":"City name"},
          "unit":{"type":"string","enum":["c","f"]}},"required":["city"]}}},
        {"type":"function","function":{"name":"search","description":"Web search \u{e9}\u{4e2d}",
          "parameters":{"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer"}}}}}
    ]);
    let calls = json!([{"id":"call_1","type":"function","function":{"name":"get_weather",
        "arguments":"{\"city\":\"Z\u{fc}rich\",\"unit\":\"c\",\"opts\":{\"b\":[1,2.5,{\"z\":null,\"a\":true}],\"a\":\"x\"}}"}},
        {"id":"call_2","type":"function","function":{"name":"search","arguments":"{\"query\":\"q\",\"limit\":3}"}}]);
    vec![
        ("user_only", json!([{"role":"user","content":"Hello"}]), None),
        ("sys_user", json!([{"role":"system","content":"You are terse."},{"role":"user","content":"Hi there"}]), None),
        ("multi_turn", json!([{"role":"system","content":"S"},{"role":"user","content":"a"},
            {"role":"assistant","content":"b"},{"role":"user","content":"c"}]), None),
        ("assistant_reasoning", json!([{"role":"user","content":"a"},
            {"role":"assistant","content":"ans","reasoning_content":"thinking about a"},{"role":"user","content":"c"}]), None),
        ("tools_user", json!([{"role":"user","content":"weather in Paris?"}]), Some(tools.clone())),
        ("tool_roundtrip", json!([{"role":"user","content":"weather in Zurich and search?"},
            {"role":"assistant","content":null,"tool_calls":calls},
            {"role":"tool","tool_call_id":"call_1","content":"{\"t\":12}"},
            {"role":"tool","tool_call_id":"call_2","content":"results"},
            {"role":"user","content":"thanks"}]), Some(tools.clone())),
        ("tool_calls_text_and_reasoning", json!([{"role":"user","content":"go"},
            {"role":"assistant","content":"calling","reasoning_content":"need tool","tool_calls":calls},
            {"role":"tool","tool_call_id":"call_1","name":"get_weather","content":"ok"}]), Some(tools)),
    ]
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut fb: Option<String> = None;
    let mut srcs = vec![];
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--fallback-tokenizer" { fb = Some(args[i + 1].clone()); i += 2; } else { srcs.push(args[i].clone()); i += 1; }
    }
    let fb = fb.expect("--fallback-tokenizer");
    let tmp = std::env::temp_dir().join(format!("ctd_{}", std::process::id()));
    for (n, s) in srcs.iter().enumerate() {
        let Some((text, tk)) = template_of(Path::new(s)) else {
            println!("{}", json!({"source": s, "status": "no_chat_template"}));
            continue;
        };
        use sha2::Digest;
        let sha = format!("{:x}", sha2::Sha256::digest(text.as_bytes()));
        let d = tmp.join(n.to_string());
        std::fs::create_dir_all(&d)?;
        std::fs::write(d.join("chat_template.jinja"), &text)?;
        let tkp = tk.unwrap_or_else(|| PathBuf::from(&fb));
        std::os::unix::fs::symlink(&tkp, d.join("tokenizer.json"))?;
        let tok = match QwenTokenizer::from_file(d.join("tokenizer.json").to_str().unwrap()) {
            Ok(t) => t, Err(e) => { println!("{}", json!({"source": s, "sha": sha, "status": format!("tokenizer error {e:#}")})); continue; }
        };
        if tok.template_meta.is_none() {
            println!("{}", json!({"source": s, "sha": sha, "status": "template_not_compiled"}));
            continue;
        }
        for (name, msgs, tools) in cases() {
            let msgs: Vec<ChatMessage> = serde_json::from_value(msgs)?;
            let tools: Option<Vec<Value>> = tools.map(|t| t.as_array().unwrap().clone());
            for (mode, effort, kw) in [
                (ThinkingMode::Auto, None, None), (ThinkingMode::On, None, None), (ThinkingMode::Off, None, None),
                (ThinkingMode::Auto, Some("low"), None), (ThinkingMode::Auto, Some("high"), None),
                (ThinkingMode::Auto, None, Some(json!({"enable_thinking": false}))),
            ] {
                for gen in [true, false] {
                    let r = if gen { tok.apply_chat_template(&msgs, tools.as_deref(), effort, kw.as_ref(), mode) }
                            else { tok.apply_chat_template_no_gen(&msgs, tools.as_deref(), effort, kw.as_ref(), mode) };
                    let (ok, out) = match r { Ok(t) => (true, t), Err(e) => (false, format!("{e:#}")) };
                    println!("{}", json!({"sha": sha, "case": name, "mode": mode.as_str(), "effort": effort, "kw": kw, "gen": gen, "ok": ok, "out": out}));
                }
            }
        }
    }
    let _ = std::fs::remove_dir_all(&tmp);
    Ok(())
}
