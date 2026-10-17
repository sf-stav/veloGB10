//! CLI-1 (owner rule 2026-09-29, AGENTS §7): THE options registry. Every engine option is a
//! command-line flag; the engine reads NO environment variables for features, levers, diagnostics
//! or test drills.
//!
//! One typed table (`REG`, generated into `opts_table.rs`) lists every option: its flag, the
//! environment variable it replaced (only for the startup rejection message and the old -> new
//! mapping), its type, its default, a one-line help, a category (the `--help` grouping; `Diag` goes
//! to `--help-diag`) and a scope:
//!   * `Spmd`  — shipped to every TP node and hashed into the boot agree (both ranks must match),
//!   * `Head`  — shipped (so a node resolves exactly what the head resolved) but never hashed,
//!   * `Local` — per box (RDMA device, blob cache, memory watchdog), never shipped.
//!
//! Life cycle: `main` calls `reject_env()` and `parse_cli()` before anything else; the values then
//! live in one process-global store. Call sites read `opts::var(opt!("flag"))`, which returns the
//! same `Result<String, VarError>` shape the old `std::env::var("GB10_…")` read did, so every
//! reader's parse is unchanged and "unset" means exactly what an unset env var meant (defaults are
//! bitwise the old no-env defaults by construction). A TP head ships `snapshot()` in its TpConfig;
//! the node `install_head()`s it (replacing every non-Local value) before it loads a byte.
//!
//! Booleans: `Flag` options are presence switches (`--x`, `--x on` store "1"; `--x off` stores
//! nothing — identical to an unset var, so presence readers stay exact); `Bool` options store "1"
//! or "0" (their readers compare the value; the old `=0` / `=1` spellings map 1:1).
//! `Internal` options are never parsed from the command line: another flag (named in the variant)
//! or the binary itself writes them (`set` / `unset`) — the replacement of the old `set_var` plumbing.

use std::sync::RwLock;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ty {
    /// presence switch: on = "1", off = unset
    Flag,
    /// value switch: on = "1", off = "0"
    Bool,
    /// integer, or bare = "1" (stored raw)
    IntFlag,
    Int,
    /// integer in lo..=hi, refused at startup outside it; stored CANONICAL (the parsed number as
    /// text), so "-0", "+5" or " 5 " cannot reach a reader that parses an unsigned type
    IntRange(i64, i64),
    Float,
    Text,
    Path,
    /// one of the listed spellings (stored raw)
    Choice(&'static [&'static str]),
    /// not a command-line option: written by the named flag / by the binary itself
    Internal(&'static str),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cat { Serve, Tp, Exl3, Spec, Kv, Nvfp4, Dsv4, Load, Ppl, Diag }

impl Cat {
    pub const ALL: [Cat; 10] = [Cat::Serve, Cat::Tp, Cat::Kv, Cat::Spec, Cat::Exl3, Cat::Nvfp4, Cat::Dsv4, Cat::Load, Cat::Ppl, Cat::Diag];
    pub fn title(self) -> &'static str {
        match self {
            Cat::Serve => "SERVING", Cat::Tp => "TENSOR PARALLEL", Cat::Exl3 => "EXL3 ENGINE LEVERS",
            Cat::Spec => "SPECULATION", Cat::Kv => "KV / PREFIX CACHE", Cat::Nvfp4 => "NVFP4 / MXFP4 ENGINE LEVERS",
            Cat::Dsv4 => "DSV4 / DSPARK", Cat::Load => "LOADING / MEMORY", Cat::Ppl => "PERPLEXITY TOOL (--perplexity)",
            Cat::Diag => "DIAGNOSTICS / TEST DRILLS",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Scope { Spmd, Head, Local }

pub struct Opt {
    /// the command-line flag, without the leading "--"
    pub flag: &'static str,
    /// the environment variable this option replaced ("" = none; the startup rejection names it)
    pub env: &'static str,
    pub ty: Ty,
    /// the default, as text (what an unset option means)
    pub def: &'static str,
    pub cat: Cat,
    pub scope: Scope,
    pub help: &'static str,
}

/// A registry slot, resolved at compile time by `opt!("flag")` (an unknown name does not compile).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct OptId(pub u16);

impl OptId {
    pub fn def(self) -> &'static Opt { &REG[self.0 as usize] }
    /// "--flag"
    pub fn flag(self) -> String { format!("--{}", self.def().flag) }
}

/// An option prints as its flag ("--exl3-round-prof") in messages.
impl std::fmt::Display for OptId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "--{}", self.def().flag) }
}

include!("opts_table.rs");

const fn str_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() { return false; }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] { return false; }
        i += 1;
    }
    true
}

/// Registry index of `flag` (compile-time in `opt!`; panics on an unknown name).
pub const fn idx(flag: &str) -> usize {
    let mut i = 0;
    while i < REG.len() {
        if str_eq(REG[i].flag, flag) { return i; }
        i += 1;
    }
    panic!("opt!(): not a registered option name (src/opts_table.rs)")
}

/// `opt!("exl3-round-prof")` — the registry slot of a flag, checked at compile time.
#[macro_export]
macro_rules! opt {
    ($f:literal) => { $crate::opts::OptId({ const I: usize = $crate::opts::idx($f); I } as u16) };
}

/// Runtime lookup by flag name (TpConfig maps, tests); None = unknown.
pub fn lookup(flag: &str) -> Option<OptId> {
    let f = flag.trim_start_matches("--");
    REG.iter().position(|o| o.flag == f).map(|i| OptId(i as u16))
}

// ---------------------------------------------------------------------------------------------
// The store
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Src { Cli, Head, Internal }

impl Src {
    fn name(self) -> &'static str { match self { Src::Cli => "cli", Src::Head => "head", Src::Internal => "internal" } }
}

static STORE: RwLock<Vec<Option<(String, Src)>>> = RwLock::new(Vec::new());

fn with_store<R>(f: impl FnOnce(&mut Vec<Option<(String, Src)>>) -> R) -> R {
    let mut g = STORE.write().unwrap_or_else(|e| e.into_inner());
    if g.len() != REG.len() { g.resize(REG.len(), None); }
    f(&mut g)
}

/// The value of an option, shaped like the `std::env::var` read it replaced (Err = unset).
#[inline]
pub fn var(o: OptId) -> Result<String, std::env::VarError> {
    let g = STORE.read().unwrap_or_else(|e| e.into_inner());
    match g.get(o.0 as usize) {
        Some(Some((v, _))) => Ok(v.clone()),
        _ => Err(std::env::VarError::NotPresent),
    }
}

/// `std::env::var_os` shape.
pub fn var_os(o: OptId) -> Option<std::ffi::OsString> { var(o).ok().map(Into::into) }

/// The value, if set.
pub fn get(o: OptId) -> Option<String> { var(o).ok() }

/// Set and not "0" / "off" / "false" / "no" (any case) / empty.
pub fn on(o: OptId) -> bool {
    var(o).map_or(false, |v| is_on_value(&v))
}

/// The truth of one stored raw value. The command-line parse normalizes Flag / Bool / IntFlag
/// values ("1" / "0" / absent), so those never reach here in another spelling; a value that reaches
/// the store raw (`set`, the head's options snapshot, a Choice option whose list has `no`) must not
/// read as ON because of its spelling (S-B9-REL-VISIONFIX2 V5: `no`, `No`, `False`, `Off` read as ON
/// before). Pure, so it is unit-tested without the process-global store.
pub(crate) fn is_on_value(v: &str) -> bool {
    !matches!(v.trim().to_ascii_lowercase().as_str(), "" | "0" | "off" | "false" | "no")
}

/// Internal write (the replacement of the old `set_var` plumbing): stored raw.
pub fn set(o: OptId, v: impl AsRef<str>) {
    let v = v.as_ref().to_string();
    with_store(|s| s[o.0 as usize] = Some((v, Src::Internal)));
}

/// Internal clear (the replacement of the old `remove_var`).
pub fn unset(o: OptId) {
    with_store(|s| s[o.0 as usize] = None);
}

// ---------------------------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------------------------

fn bool_word(v: &str) -> Option<bool> {
    match v.trim().to_ascii_lowercase().as_str() {
        "1" | "on" | "true" | "yes" => Some(true),
        "0" | "off" | "false" | "no" => Some(false),
        _ => None,
    }
}

/// Validate + canonicalize one value. Ok(None) = "store nothing" (a Flag turned off).
fn canon(o: &Opt, v: Option<&str>) -> Result<Option<String>, String> {
    let f = o.flag;
    let need = |v: Option<&str>| v.map(str::to_string).ok_or_else(|| format!("--{f} needs a value"));
    match o.ty {
        Ty::Flag => match v {
            None => Ok(Some("1".into())),
            Some(v) => match bool_word(v) {
                Some(true) => Ok(Some("1".into())),
                Some(false) => Ok(None),
                None => Err(format!("--{f} takes on|off (got '{v}')")),
            },
        },
        Ty::Bool => match v {
            None => Ok(Some("1".into())),
            Some(v) => match bool_word(v) {
                Some(b) => Ok(Some(if b { "1" } else { "0" }.into())),
                None => Err(format!("--{f} takes on|off (got '{v}')")),
            },
        },
        Ty::IntFlag => match v {
            None => Ok(Some("1".into())),
            Some(v) => {
                if let Some(b) = bool_word(v) { return Ok(Some(if b { "1" } else { "0" }.into())); }
                v.trim().parse::<i64>().map(|_| Some(v.to_string())).map_err(|_| format!("--{f} takes an integer (got '{v}')"))
            }
        },
        Ty::Int => {
            let v = need(v)?;
            v.trim().parse::<i64>().map(|_| Some(v.clone())).map_err(|_| format!("--{f} takes an integer (got '{v}')"))
        }
        Ty::IntRange(lo, hi) => {
            let v = need(v)?;
            match v.trim().parse::<i64>() {
                Ok(n) if (lo..=hi).contains(&n) => Ok(Some(n.to_string())),
                _ => Err(format!("--{f} takes {} (got '{v}')", range_words(lo, hi))),
            }
        }
        Ty::Float => {
            let v = need(v)?;
            v.trim().parse::<f64>().map(|_| Some(v.clone())).map_err(|_| format!("--{f} takes a number (got '{v}')"))
        }
        Ty::Text | Ty::Path => Ok(Some(need(v)?)),
        Ty::Choice(list) => {
            let v = need(v)?;
            if list.iter().any(|c| *c == v) { Ok(Some(v)) } else { Err(format!("--{f} takes {} (got '{v}')", list.join("|"))) }
        }
        Ty::Internal(via) => Err(format!("--{f} is not a command-line option (set it with {via})")),
    }
}

/// "an integer from 0 to 3600" / "an integer >= 0" (an open upper end is `i64::MAX`).
fn range_words(lo: i64, hi: i64) -> String {
    if hi == i64::MAX { format!("an integer >= {lo}") } else { format!("an integer from {lo} to {hi}") }
}

/// The value of an `IntRange` option: the set value, else the row's default. A set value was
/// range-checked and canonicalized by `canon` when the command line was parsed (the process
/// exits 2 there on a bad one), so the Err arms are a value that reached the store another way
/// (`set`, a head built from another source) or a row whose default is not a number in its own
/// range: the servers resolve these before they listen and refuse to start on an Err. There is
/// no silent fallback to the default.
pub fn ranged(o: OptId) -> Result<i64, String> {
    ranged_from(o, get(o).as_deref())
}

/// `ranged` over an explicit stored value (pure: the unit tests never write the shared store).
pub fn ranged_from(o: OptId, set: Option<&str>) -> Result<i64, String> {
    let d = o.def();
    let Ty::IntRange(lo, hi) = d.ty else { return Err(format!("--{} is not a ranged integer option", d.flag)) };
    let (text, what) = match set { Some(v) => (v, "value"), None => (d.def, "default") };
    match text.trim().parse::<i64>() {
        Ok(n) if (lo..=hi).contains(&n) => Ok(n),
        _ => Err(format!("--{} takes {} ({what} '{text}')", d.flag, range_words(lo, hi))),
    }
}

/// The value of a `Choice` option: the set value, else the row's default. One place states the
/// default (the registry row), so flipping it is one edit (S-B9-REL-SURFACE2B: the defaults of
/// `--unclosed-think` and `--tool-quote-pairing` are owner rulings).
pub fn choice(o: OptId) -> String {
    get(o).unwrap_or_else(|| o.def().def.to_string())
}

/// Does a Flag / Bool / IntFlag take the next argument as its value?
fn takes_next(ty: Ty, next: Option<&String>) -> bool {
    let Some(n) = next else { return false };
    match ty {
        Ty::Flag | Ty::Bool => bool_word(n).is_some(),
        Ty::IntFlag => bool_word(n).is_some() || n.trim().parse::<i64>().is_ok(),
        _ => true,
    }
}

/// Fill the store from argv (every registered, non-Internal flag; `--x v` and `--x=v`; last wins).
/// Other arguments are left to their own parsers.
pub fn parse_cli(args: &[String]) -> Result<(), String> {
    let mut i = 0;
    while i < args.len() {
        let Some((id, c, next)) = parse_one(args, i)? else { i += 1; continue };
        i = next;
        with_store(|s| s[id.0 as usize] = c.map(|v| (v, Src::Cli)));
    }
    Ok(())
}

/// The pure half of `parse_cli`: the same walk, the same `canon` validation, the same last-wins
/// rule, but the values are RETURNED (flag order, one entry per occurrence; `None` = a switch turned
/// off) instead of written to the process-global store. A tool that must read its own typed options
/// without depending on, or disturbing, the store (the `--perplexity` tool, and its unit tests) uses
/// this; it cannot drift from `parse_cli` because `parse_cli` is built on the same `parse_one`.
pub fn parse_args(args: &[String]) -> Result<Vec<(OptId, Option<String>)>, String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match parse_one(args, i)? {
            Some((id, c, next)) => { out.push((id, c)); i = next; }
            None => i += 1,
        }
    }
    Ok(out)
}

/// One argument at `i`: Ok(None) = not a registered, settable option (left to its own parser);
/// Ok(Some((slot, canonical value, index after the value)))`.
fn parse_one(args: &[String], i: usize) -> Result<Option<(OptId, Option<String>, usize)>, String> {
    let Some(body) = args[i].strip_prefix("--") else { return Ok(None) };
    let (name, inline) = match body.split_once('=') { Some((n, v)) => (n, Some(v)), None => (body, None) };
    let Some(id) = lookup(name) else { return Ok(None) };
    let o = id.def();
    if matches!(o.ty, Ty::Internal(_)) { return Ok(None); }
    let mut next = i + 1;
    // S-B9-REL-FINALFIX F7 (owner 2026-10-11): a free-form value (Text/Path) must never EAT the
    // next `--flag`. Before F7 the space form silently swallowed it -- `--spec-pass --mtp off`
    // stored "--mtp" as the value and dropped the real option (HTTP 200, wrong config). Now the
    // boot refuses at once (main.rs exits 2 before any model load, network or TP shipping), names
    // BOTH flags and says the `=` escape. A value that merely starts with ONE dash (`-x`) is
    // still a value; a value that really starts with "--" is given as `--x=--value`.
    if inline.is_none() && matches!(o.ty, Ty::Text | Ty::Path) {
        match args.get(next) {
            Some(n) if n.starts_with("--") => {
                return Err(format!(
                    "--{} needs a value, but the next argument is {n}; to give a value that really starts with \"--\" write --{}={n}",
                    o.flag, o.flag));
            }
            None => {
                return Err(format!(
                    "--{} needs a value, but it is the last argument; to give a value that starts with \"--\" write --{}=<value>",
                    o.flag, o.flag));
            }
            _ => {}
        }
    }
    let v: Option<&str> = match inline {
        Some(v) => Some(v),
        None if takes_next(o.ty, args.get(next)) => { next += 1; Some(args[next - 1].as_str()) }
        None => None,
    };
    Ok(Some((id, canon(o, v)?, next)))
}

/// The startup guard: an environment variable the engine used to read is an ERROR naming its flag
/// (an old launcher must fail loudly — a silently ignored perf knob is the AGENTS §1b hazard).
/// Variables the engine never read (a script's own GB10_* names, third-party CUDA_* / NCCL_* /
/// RUST_LOG / RUST_BACKTRACE / …) are not ours and are ignored.
pub fn env_violations() -> Vec<String> {
    let mut out = Vec::new();
    for (k, _) in std::env::vars_os() {
        let Some(k) = k.to_str() else { continue };
        if let Some(o) = REG.iter().find(|o| !o.env.is_empty() && o.env == k) {
            let with = match o.ty {
                Ty::Internal(via) => via.to_string(),
                Ty::Flag => format!("--{}", o.flag),
                Ty::Bool => format!("--{} on|off", o.flag),
                _ => format!("--{} <value>", o.flag),
            };
            out.push(format!("{k} -> {with}"));
        } else if let Some((_, with)) = REMOVED.iter().find(|(e, _)| *e == k) {
            out.push(format!("{k} -> {with}"));
        }
    }
    out.sort();
    out
}

pub fn reject_env() {
    let v = env_violations();
    if v.is_empty() { return; }
    eprintln!("FATAL: the engine no longer reads environment variables — every option is a command-line flag \
               (AGENTS §7). Unset these and pass the flag instead: {}. See --help / --help-diag / --print-config.",
              v.join("; "));
    std::process::exit(2);
}

// ---------------------------------------------------------------------------------------------
// TP: the head's resolved options ride TpConfig
// ---------------------------------------------------------------------------------------------

/// Every set, non-Local option as (flag, value), sorted — what a TP head ships to its nodes.
pub fn snapshot() -> Vec<(String, String)> {
    let g = STORE.read().unwrap_or_else(|e| e.into_inner());
    let mut v: Vec<(String, String)> = REG.iter().enumerate()
        .filter(|(_, o)| o.scope != Scope::Local)
        .filter_map(|(i, o)| g.get(i).cloned().flatten().map(|(val, _)| (o.flag.to_string(), val)))
        .collect();
    v.sort();
    v
}

/// The node side: install exactly the head's non-Local values (an option the head did not set is
/// cleared here, whatever this process had — SPMD). Unknown names are a version skew: refused.
pub fn install_head(map: &[(String, String)]) -> Result<(), String> {
    for (k, _) in map {
        let id = lookup(k).ok_or_else(|| format!("the head shipped an option this build does not know: --{k}"))?;
        if id.def().scope == Scope::Local { return Err(format!("the head shipped a node-local option --{k}")); }
    }
    with_store(|s| {
        for (i, o) in REG.iter().enumerate() {
            if o.scope == Scope::Local { continue; }
            s[i] = map.iter().find(|(k, _)| k == o.flag).map(|(_, v)| (v.clone(), Src::Head));
        }
    });
    Ok(())
}

/// F7b (owner ruling 2026-10-11): flags the mode dispatch reads by hand — not registry rows — with
/// whether the hand reader takes the NEXT argument as their value. `unknown_warnings` treats these
/// as known (no "unknown option" line, and the token after one that takes a value is its value, not
/// a flag); the source-scan test `f7b_hand_flags_cover_every_argv_literal` fails the build when a
/// "--x" literal compared against argv in main.rs / exl3_ppl.rs / exl3_serve.rs / lib.rs is neither
/// registered nor on this list. GENERATED from those four files on 2026-10-11 (363 names, 173 of
/// them value-taking); keep it sorted when a hand flag is added.
pub const HAND_FLAGS: &[(&str, bool)] = &[
    ("after", false),
    ("arm", true),
    ("auto-disable-thinking-with-tools", false),
    ("autotune", false),
    ("autotune-report", false),
    ("barriers", false),
    ("batch", true),
    ("bench-accept", false),
    ("bench-batch", false),
    ("bench-decode-ctx", false),
    ("bench-df2-matrix", false),
    ("bench-df2-sample", false),
    ("bench-df2-sample-realq", false),
    ("bench-dspark", false),
    ("bench-exl3-forward", false),
    ("bench-exl3-gemm", false),
    ("bench-exl3-mtp", false),
    ("bench-exl3-wide", false),
    ("bench-file", true),
    ("bench-lanes", false),
    ("bench-mtp", false),
    ("bench-mtp-sample", false),
    ("bench-prefill", false),
    ("bench-tau", false),
    ("bench-tree", false),
    ("bench-verify", false),
    ("binv", false),
    ("binv-trace", false),
    ("blocks", false),
    ("bundle", true),
    ("cached-models-list", false),
    ("cached-models-remove", false),
    ("cached-models-remove-all", false),
    ("capture-layers", false),
    ("chat", false),
    ("check", false),
    ("chunk-sizes", true),
    ("chunked", false),
    ("classes", true),
    ("clear-model-blobs", false),
    ("converted-load", true),
    ("cpu-affinity", false),
    ("cq-hold", false),
    ("cq-hold-us", false),
    ("ctx", true),
    ("ctxs", true),
    ("default-frequency-penalty", true),
    ("default-presence-penalty", true),
    ("default-repetition-penalty", true),
    ("depth", true),
    ("dev", true),
    ("dev2", true),
    ("df2-bake-fp8", false),
    ("df2-bake-nvfp4", false),
    ("df2-block", true),
    ("df2-carry", true),
    ("df2-prose-lane", true),
    ("df2-quant-fidelity", false),
    ("df2-round-shard", true),
    ("df2-step-dump", true),
    ("dflash-dir", true),
    ("discover-wait", true),
    ("domains", true),
    ("draft-confidence", false),
    ("draft-temperature", false),
    ("draws", true),
    ("dry-run", false),
    ("dspark", false),
    ("dspark-depth", true),
    ("dspark-fp8-head", true),
    ("dsv4-bench-decode", false),
    ("dsv4-bench-prefill", false),
    ("dump-argmax", false),
    ("dump-logits", true),
    ("dump-rank-weights", false),
    ("dump-weights", true),
    ("embed", true),
    ("embed-from", true),
    ("exl3-bench-steps", true),
    ("exl3-grouped-e", true),
    ("exl3-iters", true),
    ("exl3-l2-alias", true),
    ("exl3-pack-manifest", false),
    ("exl3-slab-cols", true),
    ("exl3-wide-ms", true),
    ("exl3-wide-mts", true),
    ("exl3-widths", true),
    ("extract-dspark", false),
    ("floats", false),
    ("from", true),
    ("gen-dflash2-synth", true),
    ("gen-dspark-synth", true),
    ("gid", true),
    ("golden", true),
    ("gpu", false),
    ("gpu-only", false),
    ("graph-precapture", false),
    ("grid", true),
    ("gt-reprime", false),
    ("head", true),
    ("help", false),
    ("help-diag", false),
    ("ids", true),
    ("imatrix", true),
    ("in-x", true),
    ("inject-delay-us-max", false),
    ("io", true),
    ("iters", true),
    ("job", true),
    ("jsonl", true),
    ("kvnll-accept", false),
    ("kvnll-accept-ctx", false),
    ("kvnll-csv", true),
    ("kvnll-dump-ref", true),
    ("kvnll-ids", true),
    ("kvnll-label", true),
    ("kvnll-needles", true),
    ("kvnll-ref", true),
    ("kvnll-text", true),
    ("kvnll-tf", false),
    ("kvnll-windows", true),
    ("label", true),
    ("ladder-dir", true),
    ("ladder-layers", true),
    ("ladder-step", true),
    ("layers", true),
    ("legacy-oracle", false),
    ("limit", true),
    ("list-model-blobs", false),
    ("lookahead", false),
    ("loop-detect", false),
    ("loop-min-reps", false),
    ("loop-window", false),
    ("ltrows-c", true),
    ("ltrows-shapes", true),
    ("main-core", false),
    ("max-batch", true),
    ("max-n", true),
    ("max-new-tokens", true),
    ("max-tokens", true),
    ("mix", false),
    ("mode", true),
    ("model", true),
    ("model-dir", true),
    ("model-dir-b", true),
    ("model-name", true),
    ("module", true),
    ("mtp-depth", true),
    ("net-test", false),
    ("ngram", true),
    ("ngram-draft", true),
    ("no-chat", false),
    ("no-shard-mixers", false),
    ("node", false),
    ("nodes", true),
    ("offset", true),
    ("once", false),
    ("oracle", true),
    ("oracle-dir", true),
    ("otel-endpoint", false),
    ("otel-model-id", false),
    ("otel-topology", false),
    ("other", false),
    ("out", true),
    ("out-dir", true),
    ("out-x", true),
    ("output-prompts", false),
    ("payload-bytes", false),
    ("peer", true),
    ("peer-ips", true),
    ("penalty-range", false),
    ("pixels", true),
    ("plan-deals", true),
    ("plan-worlds", true),
    ("poison", false),
    ("pool-census", false),
    ("por", false),
    ("port", true),
    ("positions", false),
    ("pp-bench-prefill", false),
    ("pp-chunk-list", true),
    ("pp-node", false),
    ("ppl", false),
    ("ppl-ref-windowz", false),
    ("ppl-zzzz-qqqq", false),
    ("prefill-chunk", true),
    ("prefix", false),
    ("prefix-cache", true),
    ("prefix-ckpt-mem-gb", false),
    ("print-config", false),
    ("probe-bandwidth", false),
    ("probe-bandwidth-sustained", false),
    ("probe-binv", false),
    ("probe-df2-draft", false),
    ("probe-df2-graph", false),
    ("probe-df2-lossless", false),
    ("probe-df2-prime", false),
    ("probe-df2-round", true),
    ("probe-df2-tapcap", false),
    ("probe-dflash", true),
    ("probe-dflash-local", false),
    ("probe-dflash2", false),
    ("probe-dispatch", false),
    ("probe-dspark", false),
    ("probe-dspark-bind", true),
    ("probe-dspark-rollback", false),
    ("probe-dspark-round", true),
    ("probe-dspark-synth", true),
    ("probe-dsv4", false),
    ("probe-exl3", false),
    ("probe-exl3-binv", false),
    ("probe-exl3-bitwidths", false),
    ("probe-exl3-flashpf", false),
    ("probe-exl3-hcw4", false),
    ("probe-exl3-kernel", false),
    ("probe-exl3-kvnll", false),
    ("probe-exl3-kvq", false),
    ("probe-exl3-ltrows", false),
    ("probe-exl3-moeglue", false),
    ("probe-exl3-oracle", false),
    ("probe-exl3-pack2", false),
    ("probe-exl3-pack3", false),
    ("probe-exl3-packstate", false),
    ("probe-exl3-pfxtrace", false),
    ("probe-exl3-prefillx", false),
    ("probe-exl3-regime", false),
    ("probe-exl3-routergemm", false),
    ("probe-exl3-slice", false),
    ("probe-exl3-sppin", false),
    ("probe-exl3-tailgemm", false),
    ("probe-exl3-tpspec", false),
    ("probe-exl3-wide", false),
    ("probe-exl3-wp16", false),
    ("probe-exl3-xchain", false),
    ("probe-exl3-xtp", false),
    ("probe-fold-xchain-compare", true),
    ("probe-gemm", false),
    ("probe-moe", false),
    ("probe-moe-gemm", false),
    ("probe-mxfp4", false),
    ("probe-mxfp4-fused", false),
    ("probe-mxfp4-xchain", false),
    ("probe-mxfp4-xchain2", true),
    ("probe-mxfp4-xchain2-compare", true),
    ("probe-penalties", false),
    ("probe-ppsplit", false),
    ("probe-q4", false),
    ("probe-reject", false),
    ("probe-sampler", false),
    ("probe-serve-pinned", false),
    ("probe-shard-plan", false),
    ("probe-spec-sampling", false),
    ("probe-splitk", false),
    ("probe-state", false),
    ("probe-tp-gemv", false),
    ("probe-tq", true),
    ("probe-verify-m8", false),
    ("probe-vision-tower", false),
    ("profile-mtp", false),
    ("prompt", true),
    ("prompt-file", true),
    ("prompt-ids", true),
    ("prompt-len", true),
    ("proxy-core", false),
    ("proxy-core2", false),
    ("quantize", false),
    ("quick", false),
    ("rank", true),
    ("real-taps", true),
    ("reasoning-effort", true),
    ("recipe", true),
    ("reduces", false),
    ("remove-model-blob", false),
    ("replay-df2-bisect", false),
    ("replay-df2-round", false),
    ("reps", true),
    ("requant-gdn", false),
    ("requant-q2fake", false),
    ("requant-sim", false),
    ("rounds", true),
    ("runs", true),
    ("scope", true),
    ("seconds", true),
    ("seed", true),
    ("seeds", true),
    ("seq-len", true),
    ("served-model-name", true),
    ("server", false),
    ("server-dspark", true),
    ("server-port", true),
    ("sha256", true),
    ("shapes", true),
    ("shard", true),
    ("shard-end", true),
    ("shard-rank", true),
    ("shard-start", true),
    ("shard-world", true),
    ("sizes-mb", true),
    ("slice-experts", true),
    ("spacing-us", false),
    ("spec-dds", true),
    ("spec-depths", true),
    ("spec-gen", false),
    ("spec-ident", true),
    ("spec-prefix", true),
    ("spec-prompts", true),
    ("spec-reps", false),
    ("spec-samp", true),
    ("spec-sampling", false),
    ("spec-source", true),
    ("spec-trials", false),
    ("split", true),
    ("sppin-c", true),
    ("sppin-world", true),
    ("stall-consumer-every", false),
    ("stall-us", false),
    ("step", true),
    ("steps", true),
    ("stq-bake", false),
    ("streaming", false),
    ("sweep-gemm", false),
    ("taps", true),
    ("temp", true),
    ("temperature", true),
    ("thinking", true),
    ("thinking-on", false),
    ("time-widths", true),
    ("tmp", true),
    ("tokenizer", true),
    ("tokens", true),
    ("tool-call-format", true),
    ("top-k", true),
    ("top-p", true),
    ("tp", false),
    ("tp-barrier-bench", false),
    ("tp-reduce-bench", false),
    ("tp-sim", false),
    ("tp-sim-full", false),
    ("tp-sim-layer", true),
    ("tree-draft", true),
    ("trials", true),
    ("tripwire", true),
    ("tripwire-selftest", false),
    ("tune-draft", false),
    ("tune-profile", false),
    ("tune-table", false),
    ("verify", false),
    ("vision-fail-gemm-x", false),
    ("warmup", false),
    ("with-trunk", false),
    ("world", true),
    ("wp16-b-ids", true),
    ("wp16-split", true),
    ("x-in", true),
    ("xport", false),
    ("xtp-gates", true),
    ("xtp-gen", true),
    ("xtp-ident-only", false),
    ("xtp-out", true),
    ("xtp-pf-chunk", true),
    ("xtp-ref", true),
    ("xtp-sp-fault", true),
    ("xtp-time", true),
    ("y-out", true),
];

/// F9 (owner decision A, 2026-10-11): `HAND_FLAGS` value-taking rows get the F7 refusal too.
///
/// ONE startup pass over argv — the single choke point covering every `parse_arg` /
/// `crate::arg_value` site (main.rs, exl3_serve.rs, exl3_ppl.rs, exl3_autotune.rs) at once,
/// before any mode dispatch, model load, network or TP shipping. For every hand flag marked
/// value-taking: if the token after it starts with `--`, or it is the last argument, the line
/// refuses with the same sentence shape as the registry rule (F7; main.rs prints FATAL, exit 2).
/// The `=` form always works (`--flag=--value` included). Registry rows are skipped here —
/// their own rule already refused the line in `parse_cli`. `--tp` is value-OPTIONAL by design
/// (bare `--tp` = 2), so it refuses only when the next token starts with `--` (the swallowed
/// flag), never when bare.
pub fn refuse_hand_flag_swallows(args: &[String]) -> Option<String> {
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        let Some(body) = a.strip_prefix("--") else { i += 1; continue };
        if body.is_empty() { i += 1; continue; } // a bare `--` token is not a flag
        let (name, inline) = match body.split_once('=') {
            Some((n, v)) => (n, Some(v)),
            None => (body, None),
        };
        let Some(&(flag, takes_value)) = HAND_FLAGS.iter().find(|(f, _)| *f == name) else { i += 1; continue };
        if lookup(name).is_some() { i += 1; continue; } // registry rows: F7 owns them
        if inline.is_some() { i += 1; continue; }        // the `=` form is always fine
        // F9b: optional-value flags (bare is a documented form) never refuse, in either direction
        let optional = OPTIONAL_VALUE_FLAGS.iter().any(|(vf, _)| *vf == name);
        if optional { i += 1; continue; }
        if !takes_value { i += 1; continue; }
        match args.get(i + 1) {
            Some(n) if n.starts_with("--") => {
                return Some(format!(
                    "--{flag} needs a value, but the next argument is {n}; to give a value that really starts with \"--\" write --{flag}={n}"));
            }
            None => {
                return Some(format!(
                    "--{flag} needs a value, but it is the last argument; to give a value that starts with \"--\" write --{flag}=<value>"));
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// F9b (2026-10-11): hand flags whose BARE form is documented or used. Exempt from the F9 refusal
/// in BOTH directions (a following --token and last-on-the-line): bare is a valid form. Built
/// systematically from (a) every --flag [x] entry in --help and --help-diag, (b) hand readers that
/// treat an absent value as valid, (c) every bare-flag-then--flag usage in the repo scripts, the
/// READMEs, the model cards and the PLAN launch files. The test
/// f9b_help_optionals_cover_every_help_entry fails if a --help entry with a [...] hint is missing.
pub const OPTIONAL_VALUE_FLAGS: &[(&str, &str)] = &[
    ("tp", "help --tp [N]; bare = 2 (parse_tp_world); README, 3 model cards, 36 script lines use it bare before --nodes"),
    ("head", "bare = TP-head mode (presence read in main); the value is read only by --replay-df2-round; scripts and docs use --head --model-dir bare"),
    ("probe-tq", "help --probe-tq [goldens-dir]; the dir is optional (main parse_arg with a default)"),
    ("output-prompts", "help --output-prompts [n]; n optional; marked no-value in HAND_FLAGS; listed for the help test"),
    ("probe-bandwidth-sustained", "help --probe-bandwidth-sustained [--seconds N]; the flag itself is no-value (--seconds carries the option)"),
    ("splitk-gemm", "help --splitk-gemm [N]; a REGISTRY IntFlag (bare = on, =N selects); listed for the help test"),
];

/// F9b: the ONE info line for a bare --tp (printed on stderr by main, once, at startup; never on
/// stdout, so --print-config output stays byte-identical). Mirrors the parse_tp_world bare rule:
/// an exact "--tp" token, no --tp= form, followed by no token or by a "-"-prefixed token.
pub fn tp_bare_hint(args: &[String]) -> Option<String> {
    for (i, a) in args.iter().enumerate() {
        if a == "--tp" {
            let next_flagless = match args.get(i + 1) { Some(n) => n.starts_with("-"), None => true };
            if next_flagless && !args.iter().any(|x| x.starts_with("--tp=")) {
                return Some("[opts] --tp without a number: 2 machines; write --tp 4 for four".to_string());
            }
        }
    }
    None
}



/// F7 item 4 (TP): the F7 refusal and this warning both live in parse_cli / main of the HEAD, which
/// parses its own argv BEFORE any model load, network or TP shipping — a bad head line never ships.
/// A TP node parses only its own `--node --port ...` line (same rule, same exit 2) and receives
/// every other option resolved on TpConfig, so no node-side parse path can ever see a swallowed
/// value. F7b warnings print on the head only (see main): a node does not repeat them.

/// F7b: warn, never refuse. One line per DISTINCT "--"-token that is not a registered option, not
/// a hand-parsed flag, not the value slot of the previous option, and not the bare "--" separator.
/// Until now such a token silently disappeared from the line (the --ppl-ref-windowz defect class:
/// a misspelt option name drops whatever it was setting). Up to three near registered names by
/// edit distance; nothing when nothing is close. Pure function of argv; main() prints each line
/// once, head only.
pub fn unknown_warnings(args: &[String]) -> Vec<String> {
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    let mut prev_takes_value = false;
    let mut after_sep = false;
    for a in args.iter().skip(1) {
        if after_sep { continue; } // everything after a bare "--" is positional, not an option
        if prev_takes_value { prev_takes_value = false; continue; }
        if a == "--" { after_sep = true; continue; }
        let Some(body) = a.strip_prefix("--") else { continue };
        if body.is_empty() || body.contains('=') { continue; }
        let known = lookup(body).is_some() || HAND_FLAGS.iter().any(|(f, _)| *f == body);
        if !known {
            if seen.insert(body.to_string()) {
                let near = suggest(body);
                out.push(if near.is_empty() {
                    format!("unknown option --{body}: ignored")
                } else {
                    format!("unknown option --{body}: ignored; did you mean {}?", join_flags(&near))
                });
            }
            // an unknown token might itself read a value; we cannot know, so the NEXT token is
            // judged on its own (worst case both warn — every line names a real token).
            continue;
        }
        prev_takes_value = if let Some(id) = lookup(body) {
            !matches!(id.def().ty, Ty::Flag | Ty::Bool)
        } else {
            HAND_FLAGS.iter().any(|(f, v)| *f == body && *v)
        };
    }
    out
}

/// "--a", "--a or --b", "--a, --b or --c" (at most three, nearest first).
fn join_flags(v: &[String]) -> String {
    let s: Vec<String> = v.iter().map(|n| format!("--{n}")).collect();
    match s.len() { 1 => s[0].clone(), 2 => format!("{} or {}", s[0], s[1]), _ => format!("{}, {} or {}", s[0], s[1], s[2]) }
}

/// Up to three known names close to `name`: distance at most 3 and at most half the name's
/// length, nearest first (then alphabetical, so the line is deterministic). The pool is the
/// registry PLUS the hand-parsed flags — the addendum's own example suggests --max-batch for
/// --max-bach, and --max-batch is a hand flag, not a registry row.
fn suggest(name: &str) -> Vec<String> {
    let mut c: Vec<(usize, &str)> = REG.iter()
        .filter(|o| !matches!(o.ty, Ty::Internal(_)))
        .map(|o| (edit_distance(name, o.flag), o.flag))
        .chain(HAND_FLAGS.iter().map(|(f, _)| (edit_distance(name, f), *f)))
        .filter(|(d, _)| *d <= 3 && d * 2 <= name.len())
        .collect();
    c.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(b.1)));
    c.iter().take(3).map(|(_, n)| n.to_string()).collect()
}

/// Plain Levenshtein over bytes (flag names are ASCII lowercase).
fn edit_distance(a: &str, b: &str) -> usize {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// FNV-1a over the set Spmd options ("flag=value\n", registry order); None when none is set (a
/// default boot's agree hash stays the pre-registry one).
pub fn spmd_digest() -> Option<u32> {
    let g = STORE.read().unwrap_or_else(|e| e.into_inner());
    let mut h: u32 = 0x811c_9dc5;
    let mut any = false;
    for (i, o) in REG.iter().enumerate() {
        if o.scope != Scope::Spmd { continue; }
        if let Some(Some((v, _))) = g.get(i) {
            any = true;
            for b in o.flag.bytes().chain(std::iter::once(b'=')).chain(v.bytes()).chain(std::iter::once(b'\n')) {
                h ^= b as u32;
                h = h.wrapping_mul(0x0100_0193);
            }
        }
    }
    any.then_some(h)
}

/// "--a=1 --b=x" of every set option (boot lines).
pub fn set_summary() -> String {
    let g = STORE.read().unwrap_or_else(|e| e.into_inner());
    let v: Vec<String> = REG.iter().enumerate()
        .filter_map(|(i, o)| g.get(i).cloned().flatten().map(|(val, s)| format!("--{}={val}({})", o.flag, s.name())))
        .collect();
    if v.is_empty() { "(none set; every option at its default)".into() } else { v.join(" ") }
}

// ---------------------------------------------------------------------------------------------
// --help / --help-diag / --print-config
// ---------------------------------------------------------------------------------------------

fn value_hint(o: &Opt) -> String {
    match o.ty {
        Ty::Flag => String::new(),
        Ty::Bool => " <on|off>".into(),
        Ty::IntFlag => " [N]".into(),
        Ty::Int | Ty::IntRange(..) => " <N>".into(),
        Ty::Float => " <X>".into(),
        Ty::Text => " <TEXT>".into(),
        Ty::Path => " <PATH>".into(),
        Ty::Choice(l) => format!(" <{}>", choice_shown(o, l).join("|")),
        Ty::Internal(_) => String::new(),
    }
}

/// The values of a Choice that `--help` lists. A list that has both `on` and `off` often also
/// accepts the older boolean spellings (1 / 0 / true / false / yes / no) so that existing command
/// lines keep working; those stay ACCEPTED and are not listed (`--mtp <auto|on|off>`, not
/// `<auto|on|off|1|0|true|false>`). Every other list is shown whole, and so is the list of an
/// option whose help text explains its values by such a spelling ("0 = flash256, 1 = bitwise
/// twin"): the list and the text must name the same values (S-B9-REL-SURFACE2B R8).
fn choice_shown(o: &Opt, l: &'static [&'static str]) -> Vec<&'static str> {
    const ALIASES: [&str; 6] = ["1", "0", "true", "false", "yes", "no"];
    if l.contains(&"on") && l.contains(&"off") && !ALIASES.iter().any(|v| help_explains(o.help, v)) {
        l.iter().copied().filter(|v| !ALIASES.contains(v)).collect()
    } else {
        l.to_vec()
    }
}

/// Does a help text explain the value `v` as an item of a value list, "(`v` = meaning" or
/// ", `v` = meaning"? (`--mtp`'s "1 / true = on" names an older spelling, it does not list one.)
fn help_explains(help: &str, v: &str) -> bool {
    ["(", ", "].iter().any(|pre| help.contains(&format!("{pre}{v} = ")))
}

fn help_line(o: &Opt) -> String {
    let lhs = format!("--{}{}", o.flag, value_hint(o));
    let mut s = if lhs.len() <= 34 { format!("    {lhs:<34} ") } else { format!("    {lhs}\n    {:<34} ", "") };
    s += o.help;
    s += &format!("  [{}]", o.def);
    if o.scope == Scope::Local { s += " (per box)"; }
    s
}

/// The generated option sections of `--help` (diag = false) or `--help-diag` (diag = true).
pub fn help_text(diag: bool) -> String {
    let mut s = String::new();
    let n_diag = REG.iter().filter(|o| o.cat == Cat::Diag && !matches!(o.ty, Ty::Internal(_))).count();
    s += if diag { "\nDIAGNOSTIC / TEST-DRILL OPTIONS (every engine option is a flag; the engine reads no env vars)\n" }
         else { "\nENGINE OPTIONS (generated from the option registry; every engine option is a flag — the engine\n\
                 reads no environment variables; `--x` / `--x on|off` / `--x=V` / `--x V`)\n" };
    for c in Cat::ALL {
        if (c == Cat::Diag) != diag { continue; }
        let rows: Vec<&Opt> = REG.iter().filter(|o| o.cat == c && !matches!(o.ty, Ty::Internal(_))).collect();
        if rows.is_empty() { continue; }
        s += &format!("\n  {}\n", c.title());
        let mut rows = rows;
        rows.sort_by_key(|o| o.flag);
        for o in rows { s += &help_line(o); s.push('\n'); }
    }
    if !diag {
        s += &format!("\n  {n_diag} diagnostic / test-drill options: --help-diag.   Resolved values: --print-config.\n");
    } else {
        s += "\n  Set internally (not command-line options; shown by --print-config):\n";
        for o in REG.iter().filter(|o| matches!(o.ty, Ty::Internal(_))) {
            if let Ty::Internal(via) = o.ty { s += &format!("    {:<34} {} (via {via})\n", o.flag, o.help); }
        }
    }
    s += "\n  A value that really starts with \"--\" must be written --option=value; a value-taking option is never taken from the next --flag - no value-taking option, registry or hand-read, ever eats one (F7/F9).\n";
    s
}

/// `--print-config`: every option, its value (or its default) and where the value came from.
pub fn print_config() -> String {
    let g = STORE.read().unwrap_or_else(|e| e.into_inner());
    let mut s = String::from("# gb10_inference --print-config: every registry option (flag = value  [source]  scope)\n");
    for (i, o) in REG.iter().enumerate() {
        let scope = match o.scope { Scope::Spmd => "spmd", Scope::Head => "head", Scope::Local => "local" };
        match g.get(i).cloned().flatten() {
            Some((v, src)) => s += &format!("--{} = {v}  [{}]  {scope}\n", o.flag, src.name()),
            None => s += &format!("--{} = {}  [default]  {scope}\n", o.flag, o.def),
        }
    }
    s
}

/// `--print-config`, part 2: what the resolvers make of the registry values — the owner-approved
/// output-changing defaults (`prefix.tail_ckpt`, `spec.ratio_dhead`), the TP=2 serving posture flags
/// and every EXL3 tunable's override state. Pure: parses only, touches no GPU.
pub fn print_resolved() -> String {
    use crate::exl3_forward::xtp;
    let mut s = String::from("# resolved (the resolvers' view of the values above)\n");
    let show = |r: anyhow::Result<u32>, f: &dyn Fn(u32) -> String| match r {
        Ok(c) => f(c),
        Err(e) => format!("INVALID ({e:#})"),
    };
    s += &format!("tp-prefill-overlap -> {}\n",
                  show(xtp::parse_pf_overlap(get(crate::opt!("tp-prefill-overlap")).as_deref()), &|c| xtp::pf_overlap_desc(c)));
    s += &format!("tp-vp-sampled -> {}\n",
                  show(xtp::parse_vp_sampled(get(crate::opt!("tp-vp-sampled")).as_deref()), &|c| xtp::vp_sampled_desc(c).to_string()));
    s += &format!("tp-seq-parallel -> {}\n",
                  show(xtp::parse_seq_parallel(get(crate::opt!("tp-seq-parallel")).as_deref()), &|c| xtp::seq_parallel_desc(c).to_string()));
    s += &format!("ep-deal -> {}\n", xtp::ep_deal_kind());
    s += &format!("tp-dds-prior (TP=2 WP23 cost-guard prior, ms/draft) -> {}\n", crate::exl3_serve::dds_prior_tp());
    s += &format!("tp-dds-prior (TP=4 WP23 cost-guard prior, ms/draft) -> {}\n", crate::exl3_serve::dds_prior_tp4());
    s += "# EXL3 tunables (id = default | override flag -> value; unset = the tuned table / the default)\n";
    for t in crate::exl3_tune::REGISTRY {
        let ov = (t.env_parse)();
        s += &format!("tune {} = {} | {} -> {}\n", t.id, t.default, if t.env.is_empty() { "-" } else { t.env },
                      ov.map_or("unset".to_string(), |v| v.to_string()));
    }
    s
}

/// Test support: set / clear an option's raw value (as the old tests set / removed env vars).
#[doc(hidden)]
pub fn test_set(o: OptId, v: Option<&str>) {
    match v { Some(v) => set(o, v), None => unset(o) }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- F7 item 2: the = form is accepted for EVERY row (a value the row itself accepts) ----
    #[test]
    fn f7_equals_form_accepted_for_every_row() {
        for o in REG.iter() {
            if matches!(o.ty, Ty::Internal(_)) { continue; }
            // the row's default is NOT always a parseable value (a Choice may default to a
            // meta-word like "all" outside the list; an Int may default to "unset"): pick the
            // first value canon accepts for the row's type
            let valid = match o.ty {
                Ty::Flag | Ty::Bool | Ty::IntFlag => "on".to_string(),
                Ty::Text | Ty::Path => if o.def.is_empty() { "x".to_string() } else { o.def.to_string() },
                Ty::Choice(list) => if list.contains(&o.def) { o.def.to_string() } else { list[0].to_string() },
                _ => if o.def.parse::<f64>().is_ok() { o.def.to_string() } else { "1".to_string() },
            };
            let arg = format!("--{}={valid}", o.flag);
            assert!(parse_args(&[arg.clone()]).is_ok(), "{arg}");
        }
    }

    // ---- F7b: unknown flags warn, never refuse -----------------------------------------------
    #[test]
    fn f7b_unknown_flags_warn_with_near_names() {
        let a = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(unknown_warnings(&a(&["bin", "--max-bach", "4"])),
                   vec!["unknown option --max-bach: ignored; did you mean --max-batch?"]);
        // nothing close: no suggestion at all
        assert_eq!(unknown_warnings(&a(&["bin", "--qqqqqqqqqq"])),
                   vec!["unknown option --qqqqqqqqqq: ignored"]);
        // registered rows, hand-parsed flags, value slots and the separator never warn
        assert!(unknown_warnings(&a(&["bin", "--max-batch", "4", "--bench-mtp", "--model-dir", "/m"])).is_empty());
        // the bare separator ends the options: a "--x" after it is positional, not a warning
        assert!(unknown_warnings(&a(&["bin", "--", "--x"])).is_empty());
        // each distinct name exactly once
        assert_eq!(unknown_warnings(&a(&["bin", "--max-bach", "--max-bach"])).len(), 1);
        // argv[0] (the program path) is never judged
        assert!(unknown_warnings(&a(&["/home/x/--weird-path"])).is_empty());
    }

    #[test]
    fn f7b_hand_flags_cover_every_argv_literal() {
        // zero false positives, by construction: every "--x" literal in the hand-dispatch files is
        // a registered option or on HAND_FLAGS, so unknown_warnings can never flag one.
        fn flag_literals(src: &str) -> Vec<String> {
            let b = src.as_bytes();
            let mut out = Vec::new();
            let mut i = 0;
            while i + 2 < b.len() {
                if b[i] == b'"' && b[i + 1] == b'-' && b[i + 2] == b'-' {
                    let mut j = i + 3;
                    while j < b.len() && (b[j].is_ascii_lowercase() || b[j].is_ascii_digit() || b[j] == b'-') { j += 1; }
                    if j > i + 3 && j < b.len() && b[j] == b'"' {
                        out.push(src[i + 3..j].to_string());
                        i = j;
                        continue;
                    }
                }
                i += 1;
            }
            out
        }
        let mut hand: Vec<&str> = HAND_FLAGS.iter().map(|(f, _)| *f).collect();
        let sorted = hand.is_sorted();
        hand.sort();
        assert!(sorted, "HAND_FLAGS stays sorted");
        for (file, src) in [("src/main.rs", include_str!("main.rs")),
                            ("src/exl3_ppl.rs", include_str!("exl3_ppl.rs")),
                            ("src/exl3_serve.rs", include_str!("exl3_serve.rs")),
                            ("src/lib.rs", include_str!("lib.rs"))] {
            for lit in flag_literals(src) {
                let known = lookup(&lit).is_some() || hand.binary_search(&lit.as_str()).is_ok();
                assert!(known, "{file}: the argv literal --{lit} is neither registered nor on HAND_FLAGS (a warning would fire on our own launch lines)");
            }
        }
    }

    /// S-B9-REL-MTPREPROBE2 W1.4: every decision-affecting spec flag must be Spmd so it
    /// rides TpConfig into the nodes and the boot digest (a Head-scoped spec flag would
    /// silently not ship and the head/nodes would disagree on the schedule). Pins the
    /// whole Cat::Spec class, plus the schedule-affecting Diag drills by name.
    #[test]
    fn spec_decision_flags_are_spmd() {
        for o in REG {
            if o.cat == Cat::Spec {
                assert_eq!(o.scope, Scope::Spmd, "--{} must be Spmd (rides TpConfig + boot digest)", o.flag);
            }
        }
        for f in ["mtp-reprobe", "mtp-reprobe-drill"] {
            let o = REG.iter().find(|o| o.flag == f).unwrap_or_else(|| panic!("missing --{f}"));
            assert_eq!(o.scope, Scope::Spmd, "--{f} must be Spmd (rides TpConfig + boot digest)");
        }
    }

    #[test]
    fn flags_unique_and_well_formed() {
        let mut seen = std::collections::HashSet::new();
        for o in REG {
            assert!(seen.insert(o.flag), "duplicate flag --{}", o.flag);
            assert!(!o.flag.is_empty() && o.flag.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-'),
                    "bad flag name --{}", o.flag);
            assert!(!o.help.is_empty(), "--{} has no help", o.flag);
        }
        let mut envs = std::collections::HashSet::new();
        for o in REG.iter().filter(|o| !o.env.is_empty()) { assert!(envs.insert(o.env), "env {} mapped twice", o.env); }
        for (e, _) in REMOVED { assert!(!envs.contains(e), "{e} is both registered and removed"); }
    }

    /// CLI-1's standing proof (AGENTS §7): the engine source reads no environment variable of its own.
    /// Every `std::env` read / write in src/ must name a third-party variable (HOME, CARGO_MANIFEST_DIR)
    /// or a `#[test]`-only fixture path; native/*.c must not call getenv at all. The one reader of the
    /// whole environment is `opts::env_violations` (the startup refusal).
    #[test]
    fn no_engine_env_reads_remain() {
        const ALLOWED: [&str; 4] = ["HOME", "CARGO_MANIFEST_DIR", "GB10_TEST_MODEL_DIR", "GB10_TEST_TOKENIZER"];
        const CALLS: [&str; 6] = ["env::var(", "env::var_os(", "env::set_var(", "env::remove_var(", "env::vars(", "env::vars_os("];
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut bad = Vec::new();
        let mut stack = vec![root.join("src")];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).unwrap().flatten() {
                let p = e.path();
                if p.is_dir() { stack.push(p); continue; }
                if p.extension().map_or(true, |x| x != "rs") || p.ends_with("opts.rs") { continue; }
                let text = std::fs::read_to_string(&p).unwrap();
                for (i, line) in text.lines().enumerate() {
                    let code = line.split("//").next().unwrap_or("");
                    for c in CALLS {
                        let mut from = 0;
                        while let Some(k) = code[from..].find(c) {
                            let at = from + k + c.len();
                            let rest = code[at..].trim_start();
                            let lit = rest.strip_prefix('"').and_then(|r| r.split('"').next());
                            if !lit.map_or(false, |n| ALLOWED.contains(&n)) {
                                bad.push(format!("{}:{}: {}", p.display(), i + 1, line.trim()));
                            }
                            from = at;
                        }
                    }
                }
            }
        }
        for f in ["native/net_shim.c"] {
            let text = std::fs::read_to_string(root.join(f)).unwrap();
            for (i, line) in text.lines().enumerate() {
                if line.split("//").next().unwrap_or("").contains("getenv(") { bad.push(format!("{f}:{}: {}", i + 1, line.trim())); }
            }
        }
        assert!(bad.is_empty(), "environment reads outside the registry (AGENTS §7 / CLI-1):\n{}", bad.join("\n"));
    }

    /// rel/cutb (v0.7.3): --exit-on-fatal became a default-ON registered Bool (owner decision
    /// 2026-10-06). Pins the registry row (env slot empty, def "on") and the parse semantics the
    /// serve path relies on: bare flag = on, `on|1` = on, `off|0` = off; the reader
    /// `var(opt).map_or(true, |v| v != "0")` makes an unset value ON (the flipped default).
    #[test]
    fn exit_on_fatal_default_on() {
        let o = &REG[idx("exit-on-fatal")];
        assert_eq!(o.env, "");
        assert_eq!(o.def, "on");
        assert_eq!(canon(o, None).unwrap().as_deref(), Some("1")); // bare --exit-on-fatal
        assert_eq!(canon(o, Some("on")).unwrap().as_deref(), Some("1"));
        assert_eq!(canon(o, Some("1")).unwrap().as_deref(), Some("1"));
        assert_eq!(canon(o, Some("off")).unwrap().as_deref(), Some("0"));
        assert_eq!(canon(o, Some("0")).unwrap().as_deref(), Some("0"));
        assert!(canon(o, Some("bogus")).is_err());
        // the exact read exl3_serve.rs performs, through the real store + parser:
        let id = crate::opt!("exit-on-fatal");
        let read = || crate::opts::var(id).map_or(true, |v| v != "0");
        crate::opts::test_set(id, None);        assert!(read(),  "unset must mean ON (the flipped default)");
        crate::opts::test_set(id, Some("0"));   assert!(!read(), "--exit-on-fatal off must disable");
        crate::opts::test_set(id, Some("1"));   assert!(read(),  "--exit-on-fatal on must exit");
        crate::opts::test_set(id, None);        // leave the store as found
        // the shipped CLI surface end-to-end: bare flag and value form both parse
        let a = "--exit-on-fatal".to_string();
        assert!(parse_cli(&[a.clone()]).is_ok());
        assert!(read(), "bare --exit-on-fatal must mean ON");
        let off = "off".to_string();
        assert!(parse_cli(&[a, off]).is_ok());
        assert!(!read(), "--exit-on-fatal off must disable");
        crate::opts::test_set(id, None);        // restore
    }

    /// S-B9-REL-SURFACE2B R8: the value list `--help` shows and the values the help text explains
    /// agree. `--pq8-flash` and `--exl3-score-mr` explain `0` and `1`, so their lists show them;
    /// an option that explains only on / off keeps the short list.
    #[test]
    fn a_choice_hint_lists_the_values_its_text_explains() {
        assert_eq!(value_hint(&REG[idx("pq8-flash")]), " <0|off|1|on|causal>");
        assert_eq!(value_hint(&REG[idx("exl3-score-mr")]), " <0|off|1|on|generic>");
        assert_eq!(value_hint(&REG[idx("mtp")]), " <auto|on|off>", "the older spellings stay accepted and unlisted");
        assert_eq!(value_hint(&REG[idx("tool-quote-pairing")]), " <off|on|cost>");
        let mut choices = 0usize;
        for o in REG.iter() {
            let Ty::Choice(l) = o.ty else { continue };
            choices += 1;
            let hint = value_hint(o);
            let shown: Vec<&str> = hint.trim().trim_matches(|c| c == '<' || c == '>').split('|').collect();
            for v in l.iter().filter(|v| help_explains(o.help, v)) {
                assert!(shown.contains(v), "--{}: the help text explains '{v} = ...' but the list shows {hint}", o.flag);
            }
        }
        assert!(choices > 20, "{choices}");
    }

    /// The value of a Choice option is the set value or the row's default (`choice`): the
    /// defaults that are owner rulings (2026-10-09: `reasoning`, `off`) are stated by their rows
    /// and nowhere else.
    #[test]
    fn choice_reads_the_row_default() {
        for (flag, def) in [("unclosed-think", "reasoning"), ("tool-quote-pairing", "off"), ("tool-block-end", "after-function"), ("tool-value-trim", "one")] {
            let id = lookup(flag).unwrap();
            assert_eq!((id.def().def, id.def().env, id.def().scope), (def, "", Scope::Head), "--{flag}");
            if get(id).is_none() { assert_eq!(choice(id), def, "--{flag}"); }
            let Ty::Choice(l) = id.def().ty else { panic!("--{flag} is a Choice") };
            assert!(l.contains(&def) && l.len() >= 2, "--{flag}: {l:?}");
        }
    }

    /// S-B9-REL-BENCHFIX2 (W3): a ranged integer option is refused at startup outside its range
    /// (it used to pass `Ty::Int` and become the default silently in the reader), and what is
    /// stored is the canonical number.
    #[test]
    fn ranged_integer_options_refuse_at_startup() {
        let o = &REG[idx("sse-keepalive")];
        // the default is an owner ruling (2026-10-09: on, 10 s): this row's `def` is the one place
        // that states it, and this line and `the_keepalive_boot_line_and_its_default` test it
        assert_eq!((o.env, o.def, o.ty, o.scope), ("", "10", Ty::IntRange(0, 3600), Scope::Head));
        for bad in ["-1", "3601", "99999999999999999999", "18446744073709551615", "ten", "1.5", ""] {
            let e = canon(o, Some(bad)).unwrap_err();
            assert_eq!(e, format!("--sse-keepalive takes an integer from 0 to 3600 (got '{bad}')"));
        }
        assert_eq!(canon(o, None).unwrap_err(), "--sse-keepalive needs a value");
        for (given, stored) in [("0", "0"), ("-0", "0"), ("+5", "5"), (" 5 ", "5"), ("1", "1"), ("3600", "3600"), ("010", "10")] {
            assert_eq!(canon(o, Some(given)).unwrap().as_deref(), Some(stored), "given {given:?}");
        }
        // the whole command line, as `main` parses it: FATAL + exit 2 there on an Err
        let args = |v: &str| vec!["gb10_inference".to_string(), "--sse-keepalive".to_string(), v.to_string()];
        assert_eq!(parse_cli(&args("-1")).unwrap_err(), "--sse-keepalive takes an integer from 0 to 3600 (got '-1')");
        assert!(parse_cli(&["x".to_string(), "--sse-keepalive=86400".to_string()]).is_err());
        // the reader: the default is the row's (one source), and a value that reached the store
        // without `canon` is an Err, never the default
        let id = crate::opt!("sse-keepalive");
        assert_eq!(ranged_from(id, None), Ok(10));
        assert_eq!(ranged_from(id, Some("25")), Ok(25));
        assert_eq!(ranged_from(id, Some("-1")).unwrap_err(), "--sse-keepalive takes an integer from 0 to 3600 (value '-1')");
        assert!(ranged_from(id, Some("")).is_err() && ranged_from(id, Some("3601")).is_err());
        assert!(ranged_from(crate::opt!("mtp-reprobe"), None).is_err(), "not a ranged row");
        // the two sibling options with the same silent fallback (LR-4, LR-5): 0 = unlimited
        for (flag, def) in [("max-waiting", 256), ("stream-backlog-events", 65536)] {
            let o = &REG[idx(flag)];
            assert_eq!((o.env, o.ty), ("", Ty::IntRange(0, i64::MAX)));
            assert_eq!(canon(o, Some("-1")).unwrap_err(), format!("--{flag} takes an integer >= 0 (got '-1')"));
            assert!(canon(o, Some("many")).is_err());
            assert_eq!(canon(o, Some("0")).unwrap().as_deref(), Some("0"));
            assert_eq!(canon(o, Some("-0")).unwrap().as_deref(), Some("0"));
            let id = lookup(flag).unwrap();
            assert_eq!(ranged_from(id, None), Ok(def));
            assert_eq!(ranged_from(id, Some("7")), Ok(7));
            assert!(ranged_from(id, Some("-1")).is_err());
        }
        // every ranged row's default is a number inside its own range, or the row is off unless given
        // ("unset": the test drills; their readers take `get`, an Option, never `ranged`)
        for (i, o) in REG.iter().enumerate() {
            if let Ty::IntRange(lo, hi) = o.ty {
                assert!(lo <= hi, "--{}", o.flag);
                assert!(o.def == "unset" || ranged_from(OptId(i as u16), None).is_ok(),
                        "--{}: default '{}' outside {lo}..={hi}", o.flag, o.def);
                assert_eq!(value_hint(o), " <N>");
            }
        }
        // S-B9-CF-AGREE16B (W3): the two lockstep drills are validated at startup. `--tp-width-drill 0`
        // or a negative value used to pass the parse and arm nothing, in silence.
        let w = &REG[idx("tp-width-drill")];
        assert_eq!((w.env, w.def, w.ty, w.scope, w.cat), ("", "unset", Ty::IntRange(1, 31), Scope::Local, Cat::Diag));
        for bad in ["0", "-1", "32", "sixteen", "16.0", ""] {
            assert_eq!(canon(w, Some(bad)).unwrap_err(), format!("--tp-width-drill takes an integer from 1 to 31 (got '{bad}')"));
        }
        assert_eq!(canon(w, None).unwrap_err(), "--tp-width-drill needs a value");
        for (given, stored) in [("1", "1"), ("16", "16"), (" 16 ", "16"), ("+31", "31")] {
            assert_eq!(canon(w, Some(given)).unwrap().as_deref(), Some(stored));
        }
        assert!(parse_cli(&["x".to_string(), "--tp-width-drill".to_string(), "0".to_string()]).is_err());
        let a = &REG[idx("tp-agree-drill")];
        assert_eq!((a.def, a.ty, a.scope), ("unset", Ty::IntRange(0, i64::MAX), Scope::Local));
        assert_eq!(canon(a, Some("-1")).unwrap_err(), "--tp-agree-drill takes an integer >= 0 (got '-1')");
        assert_eq!(canon(a, Some("0")).unwrap().as_deref(), Some("0"));
    }

    /// S-B9-REL-SMALLFIXES item 3: what `--help` says. The generated sections come from the registry
    /// (`help_text`); the hand-written part is `print_help` in src/main.rs (a binary: its text is
    /// read from the source here, as the env-read test above reads the tree).
    #[test]
    fn help_says_what_the_servers_do() {
        let (gen, diag) = (help_text(false), help_text(true));
        // every one of these is a serving option: listed in --help (not --help-diag) with its default
        for (flag, def) in [("qsa-row-regime", "[on]"), ("prefill-interleave", "[0]"), ("mtp-lanes", "[off]"),
                            ("prefill-sched", "[cursor]"), ("mtp", "[auto]"), ("lane-order", "[rr]"), ("lane-pack", "[on]")] {
            let o = REG.iter().find(|o| o.flag == flag).unwrap_or_else(|| panic!("--{flag} is not registered"));
            assert_ne!(o.cat, Cat::Diag, "--{flag} is a serving option");
            assert_eq!(o.env, "", "--{flag}: no environment variable");
            let line = help_line(o);
            assert!(line.trim_start().starts_with(&format!("--{flag} ")), "{line}");
            assert!(line.ends_with(def), "--{flag} must show its default {def}: ...{}", &line[line.len().saturating_sub(40)..]);
            assert!(gen.contains(&line), "--{flag} missing from --help");
            assert!(!diag.contains(&line), "--{flag} must not be a --help-diag option");
        }
        // the two flags main.rs parsed but --help never listed: both ride to the TP nodes
        for flag in ["mtp-lanes", "prefill-sched"] {
            let o = &REG[idx(flag)];
            assert_eq!(o.scope, Scope::Spmd, "--{flag}");
            assert!(o.help.starts_with("NVFP4 server"), "--{flag} names the server that reads it: {}", o.help);
            assert!(canon(o, None).is_err(), "--{flag} needs a value (main.rs reads the next argument)");
            assert!(canon(o, Some("bogus")).is_err());
        }
        assert_eq!(canon(&REG[idx("mtp-lanes")], Some("on")).unwrap().as_deref(), Some("on"));
        assert_eq!(canon(&REG[idx("prefill-sched")], Some("inline")).unwrap().as_deref(), Some("inline"));
        // --lane-order: the measured numbers (PLAN/CF_FCFS_REPORT.md section 4), not the design arithmetic
        let lo = REG[idx("lane-order")].help;
        assert!(lo.contains("23% lower with 2") && lo.contains("30% lower with 3"), "{lo}");
        assert!(!lo.contains("about 25%") && !lo.contains("about 33%"), "{lo}");
        // --prefill-interleave: no claim beyond what was gated (interleave > 0 was not, on the fixed tree)
        let il = REG[idx("prefill-interleave")].help;
        assert!(!il.contains("is removed by the per-row attention regime"), "{il}");
        assert!(il.contains("expected, not proven"), "{il}");
        // --mtp: says that off is honoured and names the banner
        assert!(REG[idx("mtp")].help.contains("MTP OFF (--mtp off)"));

        // the hand-written part
        let main = std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/main.rs")).unwrap();
        let a = main.find("fn print_help() {").expect("print_help");
        let body = &main[a..a + main[a..].find("\n}\n").expect("end of print_help")];
        // --max-tokens: a default for requests that omit it, per server; never "[8192]" for every server
        assert!(!body.contains("Default generation cap when a request omits max_tokens [8192]"));
        assert!(body.contains("--max-tokens <N>           Output length for a CHAT request that omits max_tokens. A default,"));
        assert!(body.contains("not a cap") && body.contains("[EXL3: unlimited; NVFP4: 8192]"));
        // --default-presence-penalty: the EXL3 server's default is 0.0
        assert!(!body.contains("Presence penalty       [2.0 on 2B, else 1.5]"));
        assert!(body.contains("[EXL3: 0.0; NVFP4: 2.0 on 2B, else 1.5]"));
        // unsent sampler values: the thinking-mode values of the EXL3 chat path are named
        assert!(!body.contains("defaults 0.7 / 0.8 / 20)"));
        assert!(body.contains("Unsent: 0.7 / 0.8 / 20") && body.contains("1.0 / 0.95 / 20"));
        // --mtp: off is honoured by every server; the option is described ONCE, by its registry line
        // (SURFACE2 item 5: the hand-written block repeated it, and the two texts had drifted apart)
        assert!(!body.contains("on/off force it (benchmarking)."));
        let mtp = help_line(&REG[idx("mtp")]);
        assert!(mtp.trim_start().starts_with("--mtp <auto|on|off> "), "a three-value choice in --help: {}", &mtp[..60]);
        assert!(mtp.contains("--exl3-no-mtp") && mtp.contains("1 / true = on"), "the older spellings are named");
        for v in ["1", "0", "true", "false"] { assert!(canon(&REG[idx("mtp")], Some(v)).is_ok(), "--mtp {v} is still accepted"); }
        assert!(REG[idx("exl3-no-mtp")].help.contains("older, EXL3-only spelling of --mtp off"));
        // no double-encoded characters in the help (they printed as garbage: U+00C2..U+00F4 + U+0080..U+00BF)
        assert_eq!(body.lines().filter(|l| double_encoded(l)).count(), 0, "print_help has double-encoded (mojibake) lines");
    }

    /// UTF-8 text that was decoded as Latin-1 and encoded again: a lead byte U+00C2..U+00F4
    /// followed by a continuation byte U+0080..U+00BF, both as characters (an em dash reads as
    /// three junk characters).
    fn double_encoded(l: &str) -> bool {
        let c: Vec<char> = l.chars().collect();
        c.windows(2).any(|w| ('\u{c2}'..='\u{f4}').contains(&w[0]) && ('\u{80}'..='\u{bf}').contains(&w[1]))
    }

    /// SURFACE2 item 7: no double-encoded line anywhere in the sources (48 lines of src/main.rs
    /// outside --help held them: 31 comments and 17 printed strings, RESULT lines and boot lines
    /// among them). Walks src/, kernels/, scripts/ and the build files.
    #[test]
    fn no_double_encoded_text_in_the_sources() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut files: Vec<std::path::PathBuf> = vec![root.join("build.rs"), root.join("Cargo.toml")];
        let mut dirs = vec![root.join("src"), root.join("kernels"), root.join("scripts")];
        while let Some(d) = dirs.pop() {
            let Ok(rd) = std::fs::read_dir(&d) else { continue };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() { dirs.push(p); continue; }
                if matches!(p.extension().and_then(|x| x.to_str()), Some("rs" | "cu" | "h" | "sh" | "py")) { files.push(p); }
            }
        }
        assert!(files.len() > 60, "the walk found the tree ({} files)", files.len());
        // Lines that are ALLOWED to match. The scan flags a character in U+00C2..U+00F4 followed
        // by one in U+0080..U+00BF, which is what double-encoded text looks like, but a line may
        // hold such a pair on purpose (two real Latin-1 characters side by side, a test fixture
        // of mojibake). List it here as (path from the repository root, the line trimmed) and it
        // passes; an entry that no longer matches a line fails, so the list cannot rot. Empty:
        // no source line needs it today.
        const ALLOWED: &[(&str, &str)] = &[];
        let mut used = vec![false; ALLOWED.len()];
        let mut hits: Vec<String> = Vec::new();
        for f in &files {
            let Ok(text) = std::fs::read_to_string(f) else { continue }; // not UTF-8: not this defect
            let rel = f.strip_prefix(root).unwrap_or(f).display().to_string();
            for (i, l) in text.lines().enumerate() {
                if !double_encoded(l) { continue; }
                match ALLOWED.iter().position(|(p, line)| *p == rel && *line == l.trim()) {
                    Some(k) => used[k] = true,
                    None => hits.push(format!("{rel}:{}", i + 1)),
                }
            }
        }
        assert!(hits.is_empty(), "{} double-encoded line(s), first: {:?}", hits.len(), &hits[..hits.len().min(8)]);
        let stale: Vec<&(&str, &str)> = ALLOWED.iter().zip(&used).filter(|(_, u)| !**u).map(|(a, _)| a).collect();
        assert!(stale.is_empty(), "allow-list entries that match no line any more: {stale:?}");
        // the scan itself: mojibake is flagged, the same characters in proper text are not
        assert!(double_encoded("caf\u{c3}\u{a9}") && !double_encoded("caf\u{e9} \u{2014} na\u{ef}ve"));
    }

    /// The option entries of the hand-written part of `--help`: a line that starts, after 2 to 8
    /// spaces, with `--flag` (print_help writes them as `the exact 4-space-indented --flag help format` or inside a
    /// multi-line string whose lines begin with `\x20`).
    fn hand_written_entries(body: &str) -> std::collections::BTreeSet<String> {
        let mut out = std::collections::BTreeSet::new();
        for line in body.lines() {
            let l = line.trim_start();
            let l = l.strip_prefix("print").and_then(|r| r.strip_prefix("ln!(\u{22}")).unwrap_or(l);
            let (l, lead) = match l.strip_prefix("\\x20") { Some(r) => (r, 1), None => (l, 0) };
            let spaces = lead + l.len() - l.trim_start_matches(' ').len();
            let Some(rest) = l.trim_start_matches(' ').strip_prefix("--") else { continue };
            if !(2..=8).contains(&spaces) { continue; }
            let flag: String = rest.chars().take_while(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || *c == '-').collect();
            if !flag.is_empty() { out.insert(flag); }
        }
        out
    }

    /// SURFACE2 item 5: an option is described ONCE in `--help`. `--mtp` and `--prefill-interleave`
    /// were listed by hand AND by their registry line, and the two `--mtp` texts had drifted apart
    /// (one cited a version the code does not have). They are registry-only now. The older options
    /// still listed twice are pinned here, by name: the set may shrink (delete the hand-written
    /// entry, then its name below), it must never grow.
    #[test]
    fn help_lists_an_option_once() {
        const STILL_TWICE: [&str; 22] = [
            "cpu-sample", "draft-vocab", "exact-gemm", "fp8-prefill", "fuse-residual", "hc-fp16", "max-seq-len", "moe-variant",
            "mxfp4", "no-decode-graphs", "no-gqpack", "ple-offload", "prefill-scalar", "rdma-dev", "splitk-gemm", "tp-cache",
            "tp-fp32-partials", "tp-graph", "tp-prefill-overlap", "tp-seq-parallel", "tp-shard-mtp", "tp-vp-sampled",
        ];
        let main = std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/main.rs")).unwrap();
        let a = main.find("fn print_help() {").expect("print_help");
        let body = &main[a..a + main[a..].find("\n}\n").expect("end of print_help")];
        let hand = hand_written_entries(body);
        assert!(hand.len() > 80 && hand.contains("max-seq-len") && hand.contains("max-tokens"), "the parse found the entries ({})", hand.len());
        let twice: Vec<&str> = REG.iter().filter(|o| o.cat != Cat::Diag && !matches!(o.ty, Ty::Internal(_)))
            .map(|o| o.flag).filter(|f| hand.contains(*f)).collect();
        let mut want: Vec<&str> = STILL_TWICE.to_vec();
        let mut got = twice.clone();
        want.sort();
        got.sort();
        assert_eq!(got, want, "options listed by hand and by the registry");
        for flag in ["mtp", "prefill-interleave"] {
            assert!(!hand.contains(flag), "--{flag} is described by its registry line only");
            let line = help_line(&REG[idx(flag)]);
            assert_eq!(help_text(false).matches(line.as_str()).count(), 1, "--{flag}: one generated line");
        }
        // no help or registry text cites a release the crate is not (the cut bumps the version, not the text)
        let ver = env!("CARGO_PKG_VERSION");
        let parts: Vec<u32> = ver.split('.').map(|x| x.parse().unwrap()).collect();
        let future = |text: &str| -> Option<String> {
            let b = text.as_bytes();
            (0..b.len()).filter(|&i| b[i] == b'v' && (i == 0 || !b[i - 1].is_ascii_alphanumeric())).find_map(|i| {
                let v: String = text[i + 1..].chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
                let v = v.trim_end_matches('.');
                let n: Vec<u32> = v.split('.').filter_map(|x| x.parse().ok()).collect();
                (n.len() == 3 && n > parts).then(|| format!("v{v}"))
            })
        };
        for o in REG.iter() {
            assert_eq!(future(o.help), None, "--{} cites a release newer than {ver}", o.flag);
        }
        assert_eq!(future(body), None, "print_help cites a release newer than {ver}");
        assert_eq!(future("since v0.7.3 and v99.0.1"), Some("v99.0.1".to_string()), "the check can fail");
    }

    #[test]
    fn canon_rules() {
        let f = &REG[idx("exl3-round-prof")];
        assert_eq!(canon(f, None).unwrap().as_deref(), Some("1"));
        assert_eq!(canon(f, Some("off")).unwrap(), None);
        assert!(canon(f, Some("bogus")).is_err());
        let b = &REG[idx("exl3-head-fill")];
        assert_eq!(canon(b, Some("off")).unwrap().as_deref(), Some("0"));
        assert_eq!(canon(b, Some("on")).unwrap().as_deref(), Some("1"));
        let i = &REG[idx("load-workers")];
        assert!(canon(i, Some("x")).is_err());
        assert_eq!(canon(i, Some("12")).unwrap().as_deref(), Some("12"));
    }

    /// V5: a raw stored value's truth does not depend on its spelling.
    #[test]
    fn on_reads_every_spelling_of_off_as_off() {
        for off in ["", " ", "0", "off", "OFF", "Off", "false", "False", "FALSE", "no", "No", "NO", " no "] {
            assert!(!is_on_value(off), "{off:?} must read as off");
        }
        for on in ["1", "on", "On", "true", "True", "yes", "2", "ssd", "attn,mlp"] {
            assert!(is_on_value(on), "{on:?} must read as on");
        }
    }

    /// V5: a Choice option whose list carries `no` stores it raw (`install_head` copies the head's stored
    /// values as they are); that value must read as off. (`install_head` itself is not called here: it
    /// rewrites the whole process-global store.)
    #[test]
    fn a_value_installed_raw_as_no_reads_off() {
        let id = lookup("mtp-lanes").expect("registered");
        assert!(matches!(id.def().ty, Ty::Choice(l) if l.contains(&"no")), "the option's list carries `no`");
        for raw in ["no", "False", "No"] {
            assert!(!is_on_value(raw));
        }
        // the pure parse of the same command line: `--mtp-lanes no` is accepted and stored as given
        let o = id.def();
        assert_eq!(canon(o, Some("no")).unwrap().as_deref(), Some("no"));
    }


    // ---- S-B9-REL-FINALFIX F7 (addendum 2026-10-11): the option-swallow fix ----

    /// The refusal sentence, one builder so the pinned text stays identical everywhere.
    fn f7_msg(flag: &str, next: &str) -> String {
        format!("--{flag} needs a value, but the next argument is {next}; to give a value that really starts with \"--\" write --{flag}={next}")
    }
    fn f7_last(flag: &str) -> String {
        format!("--{flag} needs a value, but it is the last argument; to give a value that starts with \"--\" write --{flag}=<value>")
    }

    /// Every Text/Path row (39 Text + 26 Path = 65): the four cases of the addendum.
    #[test]
    fn f7_text_path_rows_never_eat_a_flag_shaped_token() {
        let rows: Vec<&Opt> = REG.iter().filter(|o| matches!(o.ty, Ty::Text | Ty::Path)).collect();
        assert_eq!(rows.len(), 65, "39 Text + 26 Path");
        let args = |v: &[&str]| -> Vec<String> { v.iter().map(|s| s.to_string()).collect() };
        for o in rows {
            let id = lookup(o.flag).unwrap();
            let f = format!("--{}", o.flag);
            let e = parse_cli(&args(&[&f, "--tp", "2"])).unwrap_err();
            assert_eq!(e, f7_msg(o.flag, "--tp"), "{f}");
            let e = parse_cli(&args(&[&f, "--mtp", "off"])).unwrap_err();
            assert_eq!(e, f7_msg(o.flag, "--mtp"), "{f}");
            assert!(parse_cli(&args(&[&f, "-x"])).is_ok(), "{f}");
            crate::opts::test_set(id, None);
            assert!(parse_cli(&args(&[&f, "v", "--tp", "2"])).is_ok(), "{f}");
            crate::opts::test_set(id, None);
            assert_eq!(parse_cli(&args(&[&f])).unwrap_err(), f7_last(o.flag), "{f}");
            assert!(parse_cli(&args(&[&format!("{f}=--tp")])).is_ok(), "{f}");
            assert_eq!(crate::opts::var(id).as_deref(), Ok("--tp"), "{f}");
            crate::opts::test_set(id, None);
        }
    }

    /// Value types already fail loudly on a flag-shaped value; switch types never eat one.
    #[test]
    fn f7_other_value_types_fail_or_never_eat_a_flag_shaped_token() {
        let args = |v: &[&str]| -> Vec<String> { v.iter().map(|s| s.to_string()).collect() };
        let iv = REG.iter().find(|o| o.ty == Ty::Int).expect("an Int row");
        let e = parse_cli(&args(&[&format!("--{}", iv.flag), "--tp"])).unwrap_err();
        assert!(e.starts_with(&format!("--{} takes", iv.flag)) && e.contains("'--tp'"), "{e}");
        let e = parse_cli(&args(&["--sse-keepalive", "--tp"])).unwrap_err();
        assert!(e.starts_with("--sse-keepalive takes an integer from") && e.contains("'--tp'"), "{e}");
        let fl = REG.iter().find(|o| matches!(o.ty, Ty::Float)).expect("a Float row");
        let e = parse_cli(&args(&[&format!("--{}", fl.flag), "--tp"])).unwrap_err();
        assert!(e.starts_with(&format!("--{} takes", fl.flag)) && e.contains("'--tp'"), "{e}");
        let e = parse_cli(&args(&["--mtp", "--tp"])).unwrap_err();
        assert!(e.starts_with("--mtp takes ") && e.contains("(got '--tp')"), "{e}");
        let id = crate::opt!("exit-on-fatal");
        assert!(parse_cli(&args(&["--exit-on-fatal", "--tp", "2"])).is_ok());
        assert_eq!(crate::opts::var(id).as_deref(), Ok("1"));
        crate::opts::test_set(id, None);
        for want in [Ty::IntFlag, Ty::Flag] {
            let o = REG.iter().find(|o| o.ty == want).expect("a row of the type");
            let oid = lookup(o.flag).unwrap();
            assert!(parse_cli(&args(&[&format!("--{}", o.flag), "--tp", "2"])).is_ok(), "{}", o.flag);
            crate::opts::test_set(oid, None);
            }
    }

    /// F7: the --help footer carries the same sentence as the refusal.
    #[test]
    fn f7_help_footer_says_the_escape() {
        for diag in [false, true] {
            assert!(help_text(diag).contains("must be written --option=value"), "diag={diag}");
        }
    }

    #[test]
    fn f7_differential_vs_frozen_old_parser() { f7_differential(25_000); }
    // ---- F9 (owner decision A, 2026-10-11): hand-read value flags refuse the swallow ----
    fn f9_args(v: &[&str]) -> Vec<String> { v.iter().map(|s| s.to_string()).collect() }
    // the frozen pre-F9 reader (F9 changes no read; the guard is the startup pass)
    fn frozen_read<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
        args.iter().position(|a| a == flag || a.starts_with(&format!("{flag}=")))
            .and_then(|i| {
                let arg = &args[i];
                if let Some(val) = arg.strip_prefix(&format!("{flag}=")) { Some(val) }
                else { args.get(i + 1).map(|s| s.as_str()) }
            })
    }
    #[test]
    fn f9_generated_hand_flag_rules() {
        for &(f, takes) in HAND_FLAGS {
            let registry_owned = lookup(f).is_some();
            let fflag = format!("--{f}");
            // (a) a swallowed flag refuses, naming BOTH flags
            let v = f9_args(&["--max-batch", "4", &fflag, "--probe-binv"]);
            let r = refuse_hand_flag_swallows(&v);
            let optional = OPTIONAL_VALUE_FLAGS.iter().any(|(vf, _)| *vf == f);
            if takes && !optional && !registry_owned {
                let m = r.expect(f);
                let w1 = format!("--{f} needs a value, but the next argument is --probe-binv");
                let w2 = format!("write --{f}=--probe-binv");
                assert!(m.contains(&w1), "{f}: {m}");
                assert!(m.contains(&w2), "{f}: {m}");
            } else {
                assert!(r.is_none(), "{f}: unexpected refusal");
            }
            // (b) the = form always works, even a value starting with --
            let v = f9_args(&[&format!("--{f}=--value"), "x"]);
            assert!(refuse_hand_flag_swallows(&v).is_none());
            if !registry_owned {
                assert_eq!(crate::arg_value(&v, &fflag), Some("--value"));
            }
            // (c) values that start with ONE dash or merely contain -- are values
            if !registry_owned {
                for val in ["-1", "a--b", "0.0.0.0"] {
                    let v = f9_args(&[&fflag, val]);
                    assert!(refuse_hand_flag_swallows(&v).is_none());
                    assert_eq!(crate::arg_value(&v, &fflag), Some(val));
                }
            }
            // (d) last on the line refuses (except optional --tp; registry rows own themselves)
            let v = f9_args(&["--probe-binv", &fflag]);
            let r = refuse_hand_flag_swallows(&v);
            if takes && !optional && !registry_owned {
                let m = r.expect(f);
                let w1 = format!("--{f} needs a value, but it is the last argument");
                assert!(m.contains(&w1), "{f}: {m}");
            } else {
                assert!(r.is_none(), "{f}: unexpected refusal");
            }
        }
    }
    #[test]
    fn f9_tp_optional_value_keeps_working() {
        assert!(refuse_hand_flag_swallows(&f9_args(&["--tp"])).is_none());
        assert!(refuse_hand_flag_swallows(&f9_args(&["--tp", "4"])).is_none());
        assert!(refuse_hand_flag_swallows(&f9_args(&["--tp=4"])).is_none());
        // F9b: bare --tp before --nodes is the DOCUMENTED form (README, cards, 36 scripts)
        assert!(refuse_hand_flag_swallows(&f9_args(&["--tp", "--nodes", "a,b"])).is_none());
        assert!(tp_bare_hint(&f9_args(&["--tp", "--nodes", "a,b"])).is_some());
        assert!(tp_bare_hint(&f9_args(&["--tp"])).is_some());
        assert!(tp_bare_hint(&f9_args(&["--tp", "4"])).is_none());
        assert!(tp_bare_hint(&f9_args(&["--tp=4"])).is_none());
        assert!(tp_bare_hint(&f9_args(&["--server"])).is_none());
    }
    #[test]
    fn f9_real_lines_parse_as_before() {
        let node = f9_args(&["--node", "--host", "0.0.0.0", "--port", "29500"]);
        assert!(refuse_hand_flag_swallows(&node).is_none());
        let owner = f9_args(&["--server", "--host", "0.0.0.0", "--port", "9000", "--model-dir", "/m",
            "--tp", "4", "--nodes", "a:1,b:2,c:3", "--max-seq-len", "262144",
            "--max-batch", "4", "--max-tokens", "65536", "--prefix-cache", "on", "--mtp=auto"]);
        assert!(refuse_hand_flag_swallows(&owner).is_none());
        assert!(refuse_hand_flag_swallows(&f9_args(&["--zzz-unknown", "--port", "9000"])).is_none());
        assert!(refuse_hand_flag_swallows(&f9_args(&["--host", "--max-batch", "4"])).is_none());
    }
    #[test]
    fn f9b_help_optionals_cover_every_help_entry() {
        for diag in [false, true] {
            for line in help_text(diag).lines() {
                let t = line.trim_start();
                let Some(body) = t.strip_prefix("--") else { continue };
                let Some((name, rest)) = body.split_once(" ") else { continue };
                if !rest.starts_with("[") { continue }
                assert!(
                    OPTIONAL_VALUE_FLAGS.iter().any(|(vf, _)| *vf == name) || lookup(name).is_some(),
                    "help shows --{name} with an optional [x] hint but it is not on OPTIONAL_VALUE_FLAGS");
            }
        }
    }
    #[test]
    fn f9b_named_lines_parse() {
        // the public README zero-configuration line and the three model-card lines (a real peer IP
        // stands in for the placeholder): bare --tp before --nodes is the documented form
        let readme = f9_args(&["--server", "--tp", "--nodes", "10.0.0.2:29500", "--port", "9000"]);
        assert!(refuse_hand_flag_swallows(&readme).is_none());
        for _ in 0..3 {
            let card = f9_args(&["--tp", "--nodes", "10.0.0.2:29500", "--port", "9000"]);
            assert!(refuse_hand_flag_swallows(&card).is_none());
        }
        assert!(refuse_hand_flag_swallows(&f9_args(&["--output-prompts", "--max-batch", "4"])).is_none());
        assert!(refuse_hand_flag_swallows(&f9_args(&["--splitk-gemm", "--x"])).is_none());
        assert!(refuse_hand_flag_swallows(&f9_args(&["--head", "--model-dir", "/m"])).is_none());
        // the refusal itself stays for every other value-taking hand flag
        let m = refuse_hand_flag_swallows(&f9_args(&["--max-batch", "--probe-binv"])).expect("refuses");
        assert!(m.contains("--max-batch needs a value, but the next argument is --probe-binv"));
    }

    fn f9_differential(n: u64) {
        let mut s = 0x9E3779B97F4A7C15u64;
        let mut next = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        let mut pool = Vec::new();
        for &(f, _) in HAND_FLAGS { pool.push(format!("--{f}")); }
        for v in ["4", "29500", "on", "auto", "-1", "a--b", "0.0.0.0", "262144", "--",
                  "--max-batch", "--tp", "--probe-binv", "--unknown-flag", "--nodes=a,b"] {
            pool.push(v.to_string());
        }
        for f in ["model-dir", "port", "nodes", "max-batch", "tp"] {
            pool.push(format!("--{f}=--x"));
            pool.push(format!("--{f}=4"));
        }
        let mut value_taking = Vec::new();
        for &(f, t) in HAND_FLAGS {
            if t && lookup(f).is_none() && !OPTIONAL_VALUE_FLAGS.iter().any(|(vf, _)| *vf == f) {
                value_taking.push(format!("--{f}"));
            }
        }
        // the intended-refusal predicate: a value-taking hand flag (not a registry row, and not on
        // the OPTIONAL_VALUE_FLAGS list: --tp, --head, --probe-tq, ...) met a --token next, or sat
        // last on the line
        let intended = |v: &[String]| {
            for j in 0..v.len() {
                let Some(body) = v[j].strip_prefix("--") else { continue };
                if body.is_empty() { continue; }
                let (name, inline) = match body.split_once('=') {
                    Some((a, b)) => (a, Some(b)),
                    None => (body, None),
                };
                if inline.is_some() || lookup(name).is_some() { continue; }
                let mut hit = false;
                for &(hf, takes) in HAND_FLAGS {
                    if hf == name && !OPTIONAL_VALUE_FLAGS.iter().any(|(vf, _)| *vf == hf) {
                        if takes && v.get(j + 1).is_some_and(|x| x.starts_with("--")) { hit = true; }
                        if takes && j + 1 == v.len() { hit = true; }
                    }
                }
                if hit { return true; }
            }
            false
        };
        let mut identical = 0u64;
        let mut refused = 0u64;
        for _ in 0..n {
            let len = 1 + (next() % 8) as usize;
            let mut v = Vec::new();
            for _ in 0..len { v.push(pool[(next() % pool.len() as u64) as usize].clone()); }
            let r = refuse_hand_flag_swallows(&v);
            if let Some(m) = r {
                assert!(intended(&v), "unintended refusal {m} on {v:?}");
                refused += 1;
            } else {
                assert!(!intended(&v), "missed intended refusal on {v:?}");
                for f in &value_taking {
                    assert_eq!(frozen_read(&v, f), crate::arg_value(&v, f));
                }
                identical += 1;
            }
        }
        assert!(refused > n / 20, "the pool must exercise the refusal");
        eprintln!("f9_differential: {n} vectors: identical {identical}, intended refusals {refused}");
    }
    #[test] fn f9_differential_fast() { f9_differential(25_000); }
    #[test] #[ignore] fn f9_differential_long() { f9_differential(1_000_000); }

    #[test]
    #[ignore]
    fn f7_differential_long() { f7_differential(1_000_000); }

    /// F7 differential vs a FROZEN COPY of the pre-F7 walk: identical everywhere except a
    /// Text/Path row whose next token starts with "--" (now refused) and the last-argument
    /// sentence. lookup/canon are shared; F7 touches neither.
    fn f7_differential(n: u64) {
        fn old_bool_word(v: &str) -> Option<bool> {
            match v.trim().to_ascii_lowercase().as_str() {
                "1" | "on" | "true" | "yes" => Some(true),
                "0" | "off" | "false" | "no" => Some(false),
                _ => None,
            }
        }
        fn old_takes_next(ty: Ty, next: Option<&String>) -> bool {
            let Some(n) = next else { return false };
            match ty {
                Ty::Flag | Ty::Bool => old_bool_word(n).is_some(),
                Ty::IntFlag => old_bool_word(n).is_some() || n.trim().parse::<i64>().is_ok(),
                _ => true,
            }
        }
        fn old_parse_one(args: &[String], i: usize) -> Result<Option<(OptId, Option<String>, usize)>, String> {
            let Some(body) = args[i].strip_prefix("--") else { return Ok(None) };
            let (name, inline) = match body.split_once('=') { Some((n, v)) => (n, Some(v)), None => (body, None) };
            let Some(id) = lookup(name) else { return Ok(None) };
            let o = id.def();
            if matches!(o.ty, Ty::Internal(_)) { return Ok(None); }
            let mut next = i + 1;
            let v: Option<&str> = match inline {
                Some(v) => Some(v),
                None if old_takes_next(o.ty, args.get(next)) => { next += 1; Some(args[next - 1].as_str()) }
                None => None,
            };
            Ok(Some((id, canon(o, v)?, next)))
        }
        fn old_walk(args: &[String]) -> Result<Vec<(usize, Option<String>)>, String> {
            let mut out = Vec::new();
            let mut i = 0;
            while i < args.len() {
                match old_parse_one(args, i)? {
                    Some((id, c, next)) => { out.push((id.0 as usize, c)); i = next; }
                    None => i += 1,
                }
            }
            Ok(out)
        }
        let mut pool: Vec<String> = Vec::new();
        for o in REG.iter() {
            if matches!(o.ty, Ty::Internal(_)) { continue; }
            pool.push(format!("--{}", o.flag));
        }
        pool.push("--unknown-flag".into());
        for extra in ["--tp=2", "--rdma-dev=--tp", "--spec-pass=off", "on", "off", "1", "0",
                      "2", "262144", "-1", "3.5", "v", "-x", "--tp", "--", "auto", "x=y"] {
            pool.push(extra.into());
        }
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || { s ^= s >> 12; s ^= s << 25; s ^= s >> 27; s.wrapping_mul(0x2545_F491_4F6C_DD1D) };
        let mut identical = 0u64;
        let mut refused = 0u64;
        for _ in 0..n {
            let len = 1 + (next() % 8) as usize;
            let args: Vec<String> = (0..len).map(|_| pool[(next() % pool.len() as u64) as usize].clone()).collect();
            let old = old_walk(&args);
            let new = parse_args(&args);
            match (old, new) {
                (Ok(a), Ok(b)) => {
                    let b: Vec<(usize, Option<String>)> = b.into_iter().map(|(i, v)| (i.0 as usize, v)).collect();
                    assert_eq!(a, b, "parsed values differ: {args:?}");
                    identical += 1;
                }
                (Err(e1), Err(e2)) => {
                    if e1 == e2 { identical += 1; }
                    else {
                        assert!(e2.contains("needs a value, but the next argument is")
                                || e2.contains("it is the last argument"),
                            "an error F7 did not touch changed: {args:?} old={e1} new={e2}");
                        refused += 1;
                    }
                }
                (Ok(_), Err(e)) => {
                    assert!(e.contains("needs a value, but the next argument is"), "{args:?} -> {e}");
                    refused += 1;
                }
                (Err(e), Ok(_)) => panic!("F7 made a refusing line parse: {args:?} old={e}"),
            }
        }
        assert!(refused > 0, "the pool must exercise the new refusal");
        eprintln!("f7_differential: {n} vectors: identical {identical}, differing {refused}");
    }

}
