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
    static STRIP: [(&str, &str); 4] = [
        ("{%- generation %}", ""), ("{% generation %}", ""),
        ("{%- endgeneration %}", ""), ("{% endgeneration %}", ""),
    ];
    let mut source = source.to_string();
    for (tag, repl) in STRIP { source = source.replace(tag, repl); }
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
struct PyJsonFormatter;
impl serde_json::ser::Formatter for PyJsonFormatter {
    fn begin_array_value<W: ?Sized + std::io::Write>(&mut self, writer: &mut W, first: bool)
        -> std::io::Result<()> {
        if first { Ok(()) } else { writer.write_all(b", ") }
    }
    fn begin_object_key<W: ?Sized + std::io::Write>(&mut self, writer: &mut W, first: bool)
        -> std::io::Result<()> {
        if first { Ok(()) } else { writer.write_all(b", ") }
    }
    fn begin_object_value<W: ?Sized + std::io::Write>(&mut self, writer: &mut W)
        -> std::io::Result<()> {
        writer.write_all(b": ")
    }
}

/// Serialize a minijinja value the way Python's `json.dumps(x, ensure_ascii=False)` does
/// (default separators, insertion order). Used by the `tojson` filter — the only JSON the
/// templates emit into the prompt.
fn to_python_json(v: &minijinja::value::Value) -> Result<String, minijinja::Error> {
    use serde::Serialize;
    let mut buf: Vec<u8> = Vec::with_capacity(256);
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, PyJsonFormatter);
    v.serialize(&mut ser).map_err(|e| minijinja::Error::new(
        minijinja::ErrorKind::InvalidOperation, format!("tojson: {e}")))?;
    String::from_utf8(buf).map_err(|e| minijinja::Error::new(
        minijinja::ErrorKind::InvalidOperation, format!("tojson: {e}")))
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
        // The hy_v3 template calls `value | tojson(ensure_ascii=False)` when replaying non-string
        // tool_call arguments. Python's json.dumps(ensure_ascii=False) emits literal UTF-8, which
        // is what to_python_json does, so the kwarg is consumed and ignored; WITHOUT this a
        // kwarg'd call raises "too many arguments" and the whole request 500s.
        let _ = kwargs.get::<Option<bool>>("ensure_ascii").ok().flatten();
        to_python_json(&v)
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
                    let parts: Vec<minijinja::value::Value> =
                        s.split(sep).map(minijinja::value::Value::from).collect();
                    Some(minijinja::value::Value::from(parts))
                }
                Err(()) => {
                    let parts: Vec<minijinja::value::Value> =
                        s.split_whitespace().map(minijinja::value::Value::from).collect();
                    Some(minijinja::value::Value::from(parts))
                }
            },
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
