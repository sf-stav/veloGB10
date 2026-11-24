//! S1 side check: does the engine's shared tokenizer + chat-template environment (used read-only)
//! reproduce transformers' `apply_chat_template` ids for the Inkling-2 template? Cases come from
//! `tools/inkling2/ref_tokens.py` (`<ref>/tok_cases.json`). Report-only: the jinja gaps it finds
//! are listed in PLAN/INKLING2_S01_REPORT.md; `src/tokenizer/jinja.rs` is not edited here.

use crate::tokenizer::{ChatMessage, QwenTokenizer, ThinkingMode};
use anyhow::{Context, Result};
use std::path::Path;

/// Python `json.dumps` with explicit separators (the reference `tojson` filter).
struct SepFmt { item: Vec<u8>, key: Vec<u8> }
impl serde_json::ser::Formatter for SepFmt {
    fn begin_array_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W, first: bool) -> std::io::Result<()> {
        if first { Ok(()) } else { w.write_all(&self.item) }
    }
    fn begin_object_key<W: ?Sized + std::io::Write>(&mut self, w: &mut W, first: bool) -> std::io::Result<()> {
        if first { Ok(()) } else { w.write_all(&self.item) }
    }
    fn begin_object_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> { w.write_all(&self.key) }
}
fn sort_rec(v: serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::Object(m) => {
            let mut e: Vec<(String, serde_json::Value)> = m.into_iter().collect();
            e.sort_by(|a, b| a.0.cmp(&b.0));
            serde_json::Value::Object(e.into_iter().map(|(k, v)| (k, sort_rec(v))).collect())
        }
        serde_json::Value::Array(a) => serde_json::Value::Array(a.into_iter().map(sort_rec).collect()),
        o => o,
    }
}
fn dumps(v: &minijinja::value::Value, item: &str, key: &str, sort: bool) -> Result<String, minijinja::Error> {
    use serde::Serialize;
    let err = |e: String| minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, e);
    let mut j = serde_json::to_value(v).map_err(|e| err(e.to_string()))?;
    if sort { j = sort_rec(j); }
    let mut buf = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, SepFmt { item: item.into(), key: key.into() });
    j.serialize(&mut ser).map_err(|e| err(e.to_string()))?;
    String::from_utf8(buf).map_err(|e| err(e.to_string()))
}

/// A what-if environment: the stock template with every `generation` statement stripped (all four
/// whitespace-control spellings), `raise_exception`, and either the engine's CURRENT `tojson`
/// (`ensure_ascii` consumed, any other kwarg left unconsumed, Python default separators) or a
/// reference-complete `tojson` (`sort_keys`, `separators`, `ensure_ascii`).
fn whatif_env(src: &str, full_tojson: bool) -> Result<minijinja::Environment<'static>> {
    let mut s = src.to_string();
    for t in ["{%- generation -%}", "{%- generation %}", "{% generation -%}", "{% generation %}",
              "{%- endgeneration -%}", "{%- endgeneration %}", "{% endgeneration -%}", "{% endgeneration %}"] {
        s = s.replace(t, "");
    }
    let src: &'static str = Box::leak(s.into_boxed_str());
    let mut env = minijinja::Environment::new();
    if full_tojson {
        env.add_filter("tojson", |v: minijinja::value::Value, kw: minijinja::value::Kwargs| -> Result<String, minijinja::Error> {
            let _ = kw.get::<Option<bool>>("ensure_ascii")?;
            let sort = kw.get::<Option<bool>>("sort_keys")?.unwrap_or(false);
            let seps: Option<Vec<String>> = kw.get("separators")?;
            kw.assert_all_used()?;
            let (i, k) = match seps { Some(p) if p.len() == 2 => (p[0].clone(), p[1].clone()), _ => (", ".into(), ": ".into()) };
            dumps(&v, &i, &k, sort)
        });
    } else {
        env.add_filter("tojson", |v: minijinja::value::Value, kw: minijinja::value::Kwargs| -> Result<String, minijinja::Error> {
            let _ = kw.get::<Option<bool>>("ensure_ascii").ok().flatten();
            dumps(&v, ", ", ": ", false)
        });
    }
    env.add_function("raise_exception", |msg: String| -> Result<minijinja::value::Value, minijinja::Error> {
        Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
    });
    env.add_template("chat", src)?;
    Ok(env)
}

pub fn probe_tok(dir: &Path, ref_dir: &str) -> Result<()> {
    let tp = dir.join("tokenizer.json");
    let tok = QwenTokenizer::from_file(tp.to_str().context("path")?)?;
    let cases: serde_json::Value = serde_json::from_slice(&std::fs::read(Path::new(ref_dir).join("tok_cases.json"))?)?;
    let (mut ok, mut n) = (0, 0);
    for c in cases.as_array().context("tok_cases.json")? {
        n += 1;
        let name = c["name"].as_str().unwrap_or("?");
        let mut wire = c["messages"].clone();
        for m in wire.as_array_mut().context("messages")? {
            if let Some(tcs) = m.get_mut("tool_calls").and_then(|t| t.as_array_mut()) {
                for tc in tcs {
                    if let Some(a) = tc.pointer_mut("/function/arguments") { if !a.is_string() { *a = serde_json::Value::String(a.to_string()); } }
                }
            }
        }
        let msgs: Vec<ChatMessage> = serde_json::from_value(wire).context("messages")?;
        let tools: Option<Vec<serde_json::Value>> = c.get("tools").and_then(|t| t.as_array()).cloned();
        let effort = c.get("reasoning_effort").and_then(|e| e.as_str());
        let kwargs = c.get("chat_template_kwargs").cloned();
        let want_text = c["text"].as_str().unwrap_or("");
        let want: Vec<u32> = c["ids"].as_array().context("ids")?.iter().map(|x| x.as_u64().unwrap() as u32).collect();
        let r = tok.apply_chat_template(&msgs, tools.as_deref(), effort, kwargs.as_ref(), ThinkingMode::Auto);
        match r {
            Err(e) => println!("INK2_TOK {name}: RENDER ERROR {e:#}"),
            Ok(text) => {
                let ids = tok.encode(&text, false)?;
                let ids_ref_text = tok.encode(want_text, false)?;
                if ids == want {
                    ok += 1;
                    println!("INK2_TOK {name}: MATCH ({} ids)", ids.len());
                } else {
                    let p = text.char_indices().zip(want_text.chars()).find(|((_, a), b)| a != b).map(|((i, _), _)| i)
                        .unwrap_or(text.len().min(want_text.len()));
                    let lo = p.saturating_sub(60);
                    let snip = |s: &str| { let a = s.char_indices().map(|x| x.0).filter(|&i| i >= lo).next().unwrap_or(s.len()); s[a..].chars().take(140).collect::<String>() };
                    println!("INK2_TOK {name}: MISMATCH ours {} ids vs ref {} ids (ref TEXT encoded by us: {}); first text diff at byte {p}\n    ours: {:?}\n    ref : {:?}",
                             ids.len(), want.len(), if ids_ref_text == want { "same ids as ref" } else { "DIFFERENT ids" }, snip(&text), snip(want_text));
                }
            }
        }
    }
    println!("INK2_TOK {ok}/{n} cases reproduce the reference ids through the engine's tokenizer + template path");
    // The tokenizer alone: our encode of the reference TEXT vs the reference ids.
    let mut enc_ok = 0;
    for c in cases.as_array().unwrap() {
        let want: Vec<u32> = c["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        if tok.encode(c["text"].as_str().unwrap_or(""), false)? == want { enc_ok += 1; }
        else { println!("INK2_TOK_ENCODE {}: our encode of the reference text differs", c["name"]); }
    }
    println!("INK2_TOK_ENCODE {enc_ok}/{n} reference texts encode to the reference ids");
    // What-if renders (template context = the reference's own message JSON), to list the jinja gaps.
    let src = std::fs::read_to_string(dir.join("chat_template.jinja"))?;
    for (label, full) in [("strip-generation + CURRENT tojson", false), ("strip-generation + reference tojson(sort_keys, separators)", true)] {
        let env = match whatif_env(&src, full) { Ok(e) => e, Err(e) => { println!("INK2_TOK_WHATIF [{label}] compile error: {e:#}"); continue; } };
        let mut good = 0;
        for c in cases.as_array().unwrap() {
            let name = c["name"].as_str().unwrap_or("?");
            let mut ctx = serde_json::json!({ "messages": c["messages"], "tools": c.get("tools").cloned().unwrap_or(serde_json::json!([])), "add_generation_prompt": true });
            if let Some(e) = c.get("reasoning_effort") { ctx["reasoning_effort"] = e.clone(); }
            if let Some(kw) = c.get("chat_template_kwargs").and_then(|k| k.as_object()) { for (k, v) in kw { ctx[k.as_str()] = v.clone(); } }
            let want = c["text"].as_str().unwrap_or("");
            match env.get_template("chat").and_then(|t| t.render(&ctx)) {
                Err(e) => println!("INK2_TOK_WHATIF [{label}] {name}: RENDER ERROR {e:#}"),
                Ok(t) if t == want => good += 1,
                Ok(t) => {
                    let p = t.bytes().zip(want.bytes()).position(|(a, b)| a != b).unwrap_or(t.len().min(want.len()));
                    let lo = p.saturating_sub(50);
                    let cut = |x: &str| x.get(lo..).map(|y| y.chars().take(120).collect::<String>()).unwrap_or_default();
                    println!("INK2_TOK_WHATIF [{label}] {name}: TEXT DIFF at byte {p}\n    ours: {:?}\n    ref : {:?}", cut(&t), cut(want));
                }
            }
        }
        println!("INK2_TOK_WHATIF [{label}] {good}/{n} texts identical to the reference");
    }
    Ok(())
}
