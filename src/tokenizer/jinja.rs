//! The chat-template environment: `load_chat_env` (chat_template.jinja or tokenizer_config.json, `--chat-template` wins)
//! and the Python-compatible helpers registered on minijinja. Map of the directory in `mod.rs`.

use super::*;

/// Load the chat template that sits beside the tokenizer file and compile it into
/// a minijinja environment. Looks for `chat_template.jinja` first, then falls back
/// to the `chat_template` string inside `tokenizer_config.json`. Returns `None`
/// (so the legacy template is used) only if neither exists or the template fails
/// to compile.
pub(super) fn load_chat_env(tokenizer_path: &str) -> Result<Option<(minijinja::Environment<'static>, (String, String, u64), bool)>> {
    let dir = match Path::new(tokenizer_path).parent() { Some(d) => d, None => return Ok(None) };
    // Issue #34: --chat-template <PATH> wins over everything; a named-but-broken file must
    // fail LOUDLY at boot (bail), never silently fall back to the pack's own template —
    // the user asked for THAT template; serving different chat formatting would be wrong output.
    let argv: Vec<String> = std::env::args().collect();
    let override_path = crate::opt_or_arg(&argv, "--chat-template");
    let (source, origin) = if let Some(op) = override_path.as_deref() {
        match std::fs::read_to_string(op) {
            Ok(s) => (s, op.to_string()),
            Err(e) => anyhow::bail!("--chat-template {}: cannot read the file ({e}); refusing to fall back to the model's own template - name a readable file or drop the flag", op),
        }
    } else {
        let jinja_path = dir.join("chat_template.jinja");
        if jinja_path.exists() {
            (match std::fs::read_to_string(&jinja_path) { Ok(t) => t, Err(_) => return Ok(None) }, jinja_path.display().to_string())
        } else {
            let tc_path = dir.join("tokenizer_config.json");
            let raw = match std::fs::read_to_string(&tc_path) { Ok(r) => r, Err(_) => return Ok(None) };
            let tc: serde_json::Value = match serde_json::from_str(&raw) { Ok(v) => v, Err(_) => return Ok(None) };
            let s = match tc.get("chat_template").and_then(|v| v.as_str()) { Some(v) => v.to_string(), None => return Ok(None) };
            (s, tc_path.display().to_string())
        }
    };
    // Provenance for the boot line (Phase-2 A3): hash the template TEXT that is compiled, so a
    // leg can never silently run against a different template than the one it claims.
    let nbytes = source.len() as u64;
    let digest = {
        use sha2::Digest;
        let mut h = sha2::Sha256::new();
        h.update(source.as_bytes());
        format!("{:x}", h.finalize())
    };
    // Froggeric-class templates have a string-arguments branch. Valid JSON-object arguments are still parsed
    // into an object (issue #6); the flag only keeps the raw string for arguments that are not an object.
    let raw_tool_args = source.contains("arguments is string");
    // HF TRAINING TAGS (v0.7.10 prep): the Qwen3.6 family templates carry `{% generation %}` /
    // `{% endgeneration %}` statements (loss-mask spans for TRAINING only). minijinja has no such
    // statement, so the template fails to compile and the model silently falls back to the legacy
    // manual template — losing thinking-kwarg control on the flagship family. For inference the
    // spans are inert: strip the two statements (with any whitespace controls) before compiling.
    let source = strip_generation_tags(&source);
    // The template source must outlive the environment. The server is a long-running
    // process that loads each model exactly once, so a one-time leak of a few KB is
    // acceptable and avoids per-request recompilation.
    let static_src: &'static str = Box::leak(source.into_boxed_str());
    let mut env = minijinja::Environment::new();
    register_pycompat(&mut env);
    match env.add_template("chat", static_src) {
        Ok(_) => {
            if crate::boot_report::debug() {
                crate::deprintln!("[tokenizer] loaded chat template from {} (sha256 {} {} bytes, raw_tool_args_history={})",
                          origin, &digest[..12], nbytes, raw_tool_args);
            }
            Ok(Some((env, (origin, digest, nbytes), raw_tool_args)))
        }
        Err(e) => {
            if let Some(op) = override_path.as_deref() {
                anyhow::bail!("--chat-template {}: the template failed to compile ({}); refusing to fall back to the model's own template - fix the file or drop the flag", op, e);
            }
            eprintln!("[tokenizer] WARNING: chat template failed to compile ({}); using legacy manual template", e);
            Ok(None)
        }
    }
}

/// Python `json.dumps(..., ensure_ascii=False)` separators: `", "` between items and `": "`
/// between key and value. serde_json's default formatter writes compact separators, which made
/// our tool-definition block 345-1,346 tokens shorter than the reference render on the harness
/// catalogs (RENDER_AUDIT.md 2.1). Combined with the `preserve_order` features (serde_json +
/// minijinja) this makes the template's `tojson` byte-identical to transformers' `tojson`
/// filter for the same catalog. Non-ASCII passes through unescaped, as in Python with
/// ensure_ascii=False.
struct PyJsonFormatter { item: String, key: String, ascii: bool }
impl serde_json::ser::Formatter for PyJsonFormatter {
    fn begin_array_value<W: ?Sized + std::io::Write>(&mut self, writer: &mut W, first: bool)
        -> std::io::Result<()> {
        if first { Ok(()) } else { writer.write_all(self.item.as_bytes()) }
    }
    fn begin_object_key<W: ?Sized + std::io::Write>(&mut self, writer: &mut W, first: bool)
        -> std::io::Result<()> {
        if first { Ok(()) } else { writer.write_all(self.item.as_bytes()) }
    }
    fn begin_object_value<W: ?Sized + std::io::Write>(&mut self, writer: &mut W)
        -> std::io::Result<()> {
        writer.write_all(self.key.as_bytes())
    }
    /// `ensure_ascii=True`: every non-ASCII character leaves as `\uXXXX` (lower-case hex, UTF-16
    /// surrogate pairs above the BMP), exactly as Python's `json.dumps` does.
    fn write_string_fragment<W: ?Sized + std::io::Write>(&mut self, writer: &mut W, fragment: &str)
        -> std::io::Result<()> {
        if !self.ascii || fragment.is_ascii() { return writer.write_all(fragment.as_bytes()); }
        let mut buf = [0u16; 2];
        for c in fragment.chars() {
            if c.is_ascii() { writer.write_all(&[c as u8])?; }
            else { for u in c.encode_utf16(&mut buf) { write!(writer, "\\u{:04x}", u)?; } }
        }
        Ok(())
    }
}

/// Recursively sort every object's keys (Python `sort_keys=True`; code-point order == UTF-8 byte order).
fn sort_json_keys(v: serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::Object(m) => {
            let mut e: Vec<(String, serde_json::Value)> = m.into_iter().collect();
            e.sort_by(|a, b| a.0.cmp(&b.0));
            serde_json::Value::Object(e.into_iter().map(|(k, v)| (k, sort_json_keys(v))).collect())
        }
        serde_json::Value::Array(a) => serde_json::Value::Array(a.into_iter().map(sort_json_keys).collect()),
        o => o,
    }
}

/// Serialize a minijinja value the way Python's `json.dumps(x, ensure_ascii=..., sort_keys=...,
/// separators=...)` does. Defaults (no kwargs) are the HF `tojson` defaults: `ensure_ascii=False`,
/// insertion order, `", "` / `": "`. Used by the `tojson` filter, the only JSON templates emit.
fn to_python_json(v: &minijinja::value::Value, sort_keys: bool, ascii: bool, seps: Option<(String, String)>)
    -> Result<String, minijinja::Error> {
    use serde::Serialize;
    let err = |e: String| minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, format!("tojson: {e}"));
    let mut buf: Vec<u8> = Vec::with_capacity(256);
    let (item, key) = seps.unwrap_or_else(|| (", ".to_string(), ": ".to_string()));
    let fmt = PyJsonFormatter { item, key, ascii };
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, fmt);
    if sort_keys {
        let j = serde_json::to_value(v).map_err(|e| err(e.to_string()))?;
        sort_json_keys(j).serialize(&mut ser).map_err(|e| err(e.to_string()))?;
    } else {
        v.serialize(&mut ser).map_err(|e| err(e.to_string()))?;
    }
    String::from_utf8(buf).map_err(|e| err(e.to_string()))
}

/// Statement-level HF training tags `{% generation %}` / `{% endgeneration %}` (loss-mask spans; inert for
/// inference) become `{% if true %}` / `{% endif %}`: the tag keeps its own whitespace controls (`-` / `+`
/// on either side), so the surrounding whitespace is trimmed exactly as Jinja does for any block tag.
/// Handles every spelling (`{%- generation -%}`, `{% generation %}`, ...).
pub(super) fn strip_generation_tags(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    while let Some(p) = rest.find("{%") {
        out.push_str(&rest[..p]);
        let tail = &rest[p..];
        let Some(q) = tail.find("%}") else { out.push_str(tail); return out; };
        let inner = &tail[2..q];
        let l = if inner.starts_with(['-', '+']) { &inner[..1] } else { "" };
        let r = if inner.ends_with(['-', '+']) && inner.len() > l.len() { &inner[inner.len() - 1..] } else { "" };
        let body = inner[l.len()..inner.len() - r.len()].trim();
        match body {
            "generation" => out.push_str(&format!("{{%{l} if true {r}%}}")),
            "endgeneration" => out.push_str(&format!("{{%{l} endif {r}%}}")),
            _ => out.push_str(&tail[..q + 2]),
        }
        rest = &tail[q + 2..];
    }
    out.push_str(rest);
    out
}

/// Register an `unknown_method_callback` that bridges Jinja2/Python string methods —
/// which HuggingFace chat templates call freely but minijinja does not ship — so the
/// model's official template renders unchanged.
///
/// Handles the Python str methods this template family uses (`startswith`, `endswith`,
/// `lstrip`, `rstrip`, `strip`) directly, then falls back to minijinja's built-in
/// filters for anything else (`split`, `replace`, `trim`, `lower`, …). Python's
/// `lstrip`/`rstrip`/`strip` treat the argument as a *set of characters*, matching
/// `trim_*_matches` semantics below.
pub(super) fn register_pycompat(env: &mut minijinja::Environment<'static>) {
    // HF chat templates are written against Jinja2 (Flask flavour). minijinja does not ship these, and
    // they are reachable ONLY on the tools path -- so the template compiled fine, rendered fine for
    // every ordinary chat, and blew up with a 500 the first time a tool definition was passed.
    //
    // `tojson`         - serialises the tool schema into the `# Tools` block, and any object/array
    //                    argument when a prior assistant tool_call is replayed.
    // `raise_exception`- the template calls it on malformed input; without it, a bad message would
    //                    fail with "unknown function" instead of the template's own diagnostic.
    env.add_filter("tojson", |v: minijinja::value::Value, kwargs: minijinja::value::Kwargs|
                   -> Result<String, minijinja::Error> {
        // Python `json.dumps` kwargs templates use: `ensure_ascii` (HF default False: literal UTF-8; True
        // escapes non-ASCII as \uXXXX), `sort_keys` (recursive), `separators=(item, key)`. All three are
        // honoured; before 2026-11 only `ensure_ascii` was consumed and the other two were silently ignored
        // (the Inkling-2 tool declaration rendered with the wrong separators and key order).
        let ascii = kwargs.get::<Option<bool>>("ensure_ascii").ok().flatten().unwrap_or(false);
        let sort_keys = kwargs.get::<Option<bool>>("sort_keys").ok().flatten().unwrap_or(false);
        let seps: Option<Vec<String>> = kwargs.get::<Option<Vec<String>>>("separators").ok().flatten();
        let seps = match seps { Some(mut p) if p.len() == 2 => { let k = p.pop().unwrap(); Some((p.pop().unwrap(), k)) } _ => None };
        to_python_json(&v, sort_keys, ascii, seps)
    });
    env.add_function("raise_exception", |msg: String| -> Result<minijinja::value::Value, minijinja::Error> {
        Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
    });

    env.set_unknown_method_callback(|state, value, method, args| {
        let s = value.as_str();
        let res: Option<minijinja::value::Value> = match (method, s) {
            // `split` must return a *subscriptable* list — minijinja's built-in split
            // filter yields a one-shot iterable that the template then indexes with
            // `[0]` / `[-1]`. Build a real Vec so indexing works.
            ("split", Some(s)) => match arg_str(args, 0) {
                Ok(sep) => {
                    // Python `str.split(sep, maxsplit)`: the second argument bounds the number of splits
                    let parts: Vec<minijinja::value::Value> = match args.get(1).and_then(|v| i64::try_from(v.clone()).ok()) {
                        Some(n) if n >= 0 => s.splitn(n as usize + 1, sep).map(minijinja::value::Value::from).collect(),
                        _ => s.split(sep).map(minijinja::value::Value::from).collect(),
                    };
                    Some(minijinja::value::Value::from(parts))
                }
                Err(()) => {
                    let parts: Vec<minijinja::value::Value> =
                        s.split_whitespace().map(minijinja::value::Value::from).collect();
                    Some(minijinja::value::Value::from(parts))
                }
            },
            // dict.get(key[, default]): the DeepSeek-V4.1 template reads optional message fields this way.
            // minijinja maps have no `get`, so this used to fall through to a filter and fail with
            // "unknown filter"; a template that did not call it renders exactly as before.
            ("get", None) if value.kind() == minijinja::value::ValueKind::Map && !args.is_empty() => {
                let hit = value.get_item(&args[0]).ok().filter(|v| !v.is_undefined());
                Some(hit.unwrap_or_else(|| args.get(1).cloned().unwrap_or_else(|| minijinja::value::Value::from(()))))
            }
            ("startswith", Some(s)) => match arg_str(args, 0) {
                Ok(prefix) => Some(minijinja::value::Value::from(s.starts_with(prefix))),
                Err(()) => None,
            },
            ("endswith", Some(s)) => match arg_str(args, 0) {
                Ok(suffix) => Some(minijinja::value::Value::from(s.ends_with(suffix))),
                Err(()) => None,
            },
            ("lstrip", Some(s)) => Some(match arg_str(args, 0) {
                Ok(chars) => minijinja::value::Value::from(s.trim_start_matches(|c| chars.contains(c))),
                Err(()) => minijinja::value::Value::from(s.trim_start()),
            }),
            ("rstrip", Some(s)) => Some(match arg_str(args, 0) {
                Ok(chars) => minijinja::value::Value::from(s.trim_end_matches(|c| chars.contains(c))),
                Err(()) => minijinja::value::Value::from(s.trim_end()),
            }),
            ("strip", Some(s)) => Some(match arg_str(args, 0) {
                Ok(chars) => minijinja::value::Value::from(s.trim_matches(|c| chars.contains(c))),
                Err(()) => minijinja::value::Value::from(s.trim()),
            }),
            // Python str.format — NOT minijinja's printf-style `format` filter (%s). The hy_v3
            // template builds EVERY special token this way ('<｜hy_eos{}｜>'.format(HYTK)); letting
            // the call fall through to the built-in renders the string unchanged and plants
            // literal `{}` in the prompt. `{}` takes the next positional arg, `{N}` the Nth;
            // anything richer (format specs) is out of scope and delegates to the built-in.
            ("format", Some(s)) => {
                let render = |v: &minijinja::value::Value| match v.as_str() {
                    Some(x) => x.to_string(),
                    None => v.to_string(),
                };
                let mut out = String::with_capacity(s.len() + 8);
                let mut rest = s;
                let mut next = 0usize;
                let mut py_ok = true;
                while let Some(p) = rest.find('{') {
                    out.push_str(&rest[..p]);
                    match rest[p..].find('}') {
                        Some(q) => {
                            let spec = &rest[p + 1..p + q];
                            let idx = if spec.is_empty() {
                                let i = next; next += 1; i
                            } else if let Ok(i) = spec.parse::<usize>() {
                                next = i + 1; i
                            } else { py_ok = false; break; };   // a format spec — not Python-simple
                            match args.get(idx) {
                                Some(v) => out.push_str(&render(v)),
                                None => { py_ok = false; break; }
                            }
                            rest = &rest[p + q + 1..];
                        }
                        None => { py_ok = false; break; }
                    }
                }
                if py_ok {
                    out.push_str(rest);
                    Some(minijinja::value::Value::from(out))
                } else { None }
            },
            _ => None,
        };
        match res {
            Some(v) => Ok(v),
            None => {
                // Delegate to any built-in filter of the same name (e.g. `split`, `replace`).
                let mut all = vec![value.clone()];
                all.extend_from_slice(args);
                state.apply_filter(method, &all)
            }
        }
    });
}

fn arg_str(args: &[minijinja::value::Value], i: usize) -> Result<&str, ()> {
    args.get(i).and_then(|v| v.as_str()).ok_or(())
}
