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
//!
//! Map of this directory: this file holds the registry machinery (types, store, parser, help renderers,
//! HAND_FLAGS); `tests.rs` holds its unit tests (declared at the end of this file). The option rows
//! themselves stay in `src/opts_table.rs`, `include!`d below.

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

/// CLIPRUNE1: where an option shows up. `Day` = the day-to-day launch set (`--help`);
/// `Adv` = everything else (diagnostics, drills, bench/probe knobs): `--help-advanced`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Vis { Day, Adv }

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
    pub vis: Vis,
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

include!("../opts_table.rs");

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
        // --draft-vocab is free-form (off|on|N|P%|0.F) but the loader reads it long after the boot:
        // a bad spelling is refused HERE, never silently read as "off" at load time.
        Ty::Text if f == "draft-vocab" => crate::quant::DraftVocab::canonical_text(&need(v)?).map(Some),
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
pub const HAND_FLAGS: &[(&str, bool, Vis)] = &[
    ("after", false, Vis::Adv),
    ("arm", true, Vis::Adv),
    ("auto-disable-thinking-with-tools", false, Vis::Adv),
    ("autotune", false, Vis::Adv),
    ("autotune-report", false, Vis::Adv),
    ("barriers", false, Vis::Adv),
    ("batch", true, Vis::Adv),
    ("bench-accept", false, Vis::Adv),
    ("bench-batch", false, Vis::Adv),
    ("bench-decode-ctx", false, Vis::Adv),
    ("bench-df2-matrix", false, Vis::Adv),
    ("bench-df2-sample", false, Vis::Adv),
    ("bench-df2-sample-realq", false, Vis::Adv),
    ("bench-dspark", false, Vis::Adv),
    ("bench-exl3-forward", false, Vis::Adv),
    ("bench-exl3-gemm", false, Vis::Adv),
    ("bench-exl3-ladder", false, Vis::Adv),
    ("bench-exl3-mtp", false, Vis::Adv),
    ("bench-exl3-wide", false, Vis::Adv),
    ("bench-file", true, Vis::Adv),
    ("bench-lanes", false, Vis::Adv),
    ("bench-mtp", false, Vis::Adv),
    ("bench-mtp-sample", false, Vis::Adv),
    ("bench-only", false, Vis::Adv),
    ("bench-prefill", false, Vis::Adv),
    ("bench-tau", false, Vis::Adv),
    ("bench-tree", false, Vis::Adv),
    ("bench-verify", false, Vis::Adv),
    ("binv", false, Vis::Adv),
    ("binv-trace", false, Vis::Adv),
    ("blocks", false, Vis::Adv),
    ("bundle", true, Vis::Adv),
    ("cached-models-list", false, Vis::Adv),
    ("cached-models-remove", false, Vis::Adv),
    ("cached-models-remove-all", false, Vis::Adv),
    ("capture-layers", false, Vis::Adv),
    ("chat", false, Vis::Adv),
    ("chat-template", true, Vis::Adv),
    ("check", false, Vis::Adv),
    ("chunk-sizes", true, Vis::Adv),
    ("chunk-sizes", true, Vis::Adv),
    ("chunked", false, Vis::Adv),
    ("classes", true, Vis::Adv),
    ("clear-model-blobs", false, Vis::Adv),
    ("converted-load", true, Vis::Adv),
    ("cpu-affinity", false, Vis::Adv),
    ("cq-hold", false, Vis::Adv),
    ("cq-hold-us", false, Vis::Adv),
    ("ctx", true, Vis::Adv),
    ("ctxs", true, Vis::Adv),
    ("default-frequency-penalty", true, Vis::Adv),
    ("default-presence-penalty", true, Vis::Day),
    ("default-repetition-penalty", true, Vis::Adv),
    ("depth", true, Vis::Adv),
    ("dev", true, Vis::Adv),
    ("dev2", true, Vis::Adv),
    ("df2-bake-fp8", false, Vis::Adv),
    ("df2-bake-nvfp4", false, Vis::Adv),
    ("df2-block", true, Vis::Adv),
    ("df2-carry", true, Vis::Adv),
    ("df2-k", true, Vis::Adv),
    ("df2-mask-override", true, Vis::Adv),
    ("df2-prose-lane", true, Vis::Adv),
    ("df2-quant-fidelity", false, Vis::Adv),
    ("df2-round-shard", true, Vis::Adv),
    ("df2-step-dump", true, Vis::Adv),
    ("df2-trace", false, Vis::Adv),
    ("dflash-dir", true, Vis::Adv),
    ("discover-wait", true, Vis::Adv),
    ("discovery-port", true, Vis::Adv), // OUT2: node UDP bind (default 29499); a 2nd node can coexist on a busy box
    ("domains", true, Vis::Adv),
    ("draft-confidence", false, Vis::Adv),
    ("draft-temperature", false, Vis::Adv),
    ("draws", true, Vis::Adv),
    ("dry-run", false, Vis::Adv),
    ("dspark", false, Vis::Adv),
    ("dspark-depth", true, Vis::Adv),
    ("dspark-fp8-head", true, Vis::Adv),
    ("dsv4-bench-decode", false, Vis::Adv),
    ("dsv4-bench-prefill", false, Vis::Adv),
    ("dump-argmax", false, Vis::Adv),
    ("dump-logits", true, Vis::Adv),
    ("dump-rank-weights", false, Vis::Adv),
    ("dump-states", false, Vis::Adv),
    ("dump-weights", true, Vis::Adv),
    ("embed", true, Vis::Adv),
    ("embed-from", true, Vis::Adv),
    ("exl3-bench-steps", true, Vis::Adv),
    ("exl3-grouped-e", true, Vis::Adv),
    ("exl3-iters", true, Vis::Adv),
    ("exl3-ks", true, Vis::Adv),
    ("exl3-l2-alias", true, Vis::Adv),
    ("exl3-ladder-out", true, Vis::Adv),
    ("exl3-ladder-skip-n", true, Vis::Adv),
    ("exl3-ladder-window", true, Vis::Adv),
    ("exl3-pack-manifest", false, Vis::Adv),
    ("exl3-prime-batch", true, Vis::Adv),
    ("exl3-prime-batch-xcheck", true, Vis::Adv),
    ("exl3-slab-cols", true, Vis::Adv),
    ("exl3-wide-ms", true, Vis::Adv),
    ("exl3-wide-mts", true, Vis::Adv),
    ("exl3-widths", true, Vis::Adv),
    ("extract-dspark", false, Vis::Adv),
    ("floats", false, Vis::Adv),
    ("fp4l-ref", true, Vis::Adv),
    ("fp4l-threads", true, Vis::Adv),
    ("from", true, Vis::Adv),
    ("gemm-m", true, Vis::Adv),
    ("gen", true, Vis::Adv),
    ("gen-dflash2-synth", true, Vis::Adv),
    ("gen-dspark-synth", true, Vis::Adv),
    ("gid", true, Vis::Adv),
    ("golden", true, Vis::Adv),
    ("gpu", false, Vis::Adv),
    ("gpu-only", false, Vis::Adv),
    ("graph-precapture", false, Vis::Adv),
    ("grid", true, Vis::Adv),
    ("gt-reprime", false, Vis::Adv),
    ("head", true, Vis::Adv),
    ("help", false, Vis::Adv),
    ("help-advanced", false, Vis::Adv),
    ("help-diag", false, Vis::Adv),
    ("hf-cache", true, Vis::Adv),
    ("ids", true, Vis::Adv),
    ("imatrix", true, Vis::Adv),
    ("in-x", true, Vis::Adv),
    ("inject-delay-us-max", false, Vis::Adv),
    ("io", true, Vis::Adv),
    ("iters", true, Vis::Adv),
    ("job", true, Vis::Adv),
    ("jsonl", true, Vis::Adv),
    ("kvnll-accept", false, Vis::Adv),
    ("kvnll-accept-ctx", false, Vis::Adv),
    ("kvnll-csv", true, Vis::Adv),
    ("kvnll-dump-ref", true, Vis::Adv),
    ("kvnll-ids", true, Vis::Adv),
    ("kvnll-label", true, Vis::Adv),
    ("kvnll-needles", true, Vis::Adv),
    ("kvnll-ref", true, Vis::Adv),
    ("kvnll-text", true, Vis::Adv),
    ("kvnll-tf", false, Vis::Adv),
    ("kvnll-windows", true, Vis::Adv),
    ("label", true, Vis::Adv),
    ("ladder-dir", true, Vis::Adv),
    ("ladder-layers", true, Vis::Adv),
    ("ladder-step", true, Vis::Adv),
    ("layer", true, Vis::Adv),
    ("layers", true, Vis::Adv),
    ("legacy-oracle", false, Vis::Adv),
    ("limit", true, Vis::Adv),
    ("list-model-blobs", false, Vis::Adv),
    ("lookahead", false, Vis::Adv),
    ("loop-detect", false, Vis::Adv),
    ("loop-min-reps", false, Vis::Adv),
    ("loop-window", false, Vis::Adv),
    ("ltrows-c", true, Vis::Adv),
    ("ltrows-shapes", true, Vis::Adv),
    ("main-core", false, Vis::Adv),
    ("max-batch", true, Vis::Day),
    ("max-n", true, Vis::Adv),
    ("max-new-tokens", true, Vis::Adv),
    ("max-tokens", true, Vis::Day),
    ("mix", false, Vis::Adv),
    ("mode", true, Vis::Adv),
    ("model", true, Vis::Adv),
    ("model-dir", true, Vis::Day),
    ("model-dir-b", true, Vis::Adv),
    ("model-name", true, Vis::Adv),
    ("module", true, Vis::Adv),
    ("mtp-depth", true, Vis::Adv),
    ("net-test", false, Vis::Adv),
    ("ngram", true, Vis::Adv),
    ("ngram-draft", true, Vis::Adv),
    ("nll", false, Vis::Adv),
    ("nll-file", true, Vis::Adv),
    ("nll-out", true, Vis::Adv),
    ("nll-rows", true, Vis::Adv),
    ("nll-tokens", true, Vis::Adv),
    ("no-chat", false, Vis::Adv),
    ("no-shard-mixers", false, Vis::Adv),
    ("no-update-check", false, Vis::Adv),
    ("node", false, Vis::Day),
    ("nodes", true, Vis::Adv),
    ("offset", true, Vis::Adv),
    ("once", false, Vis::Adv),
    ("oracle", true, Vis::Adv),
    ("oracle-dir", true, Vis::Adv),
    // MTPAGREE1 (public v0.7.6 report, Q3): the six --otel-* options are read by otel::config_from_args
    // through a closure, never as "--otel-…" literals in the scanned files, so three of them were
    // missing here and F7b printed "unknown option --otel-include-tokens: ignored" for a value the
    // server DID apply (v0.7.4 and v0.7.6 alike). All six take a value.
    ("otel-batch-interval-ms", true, Vis::Adv),
    ("otel-batch-size", true, Vis::Adv),
    ("otel-endpoint", true, Vis::Adv),
    ("otel-include-tokens", true, Vis::Adv),
    ("otel-model-id", true, Vis::Adv),
    ("otel-topology", true, Vis::Adv),
    ("other", false, Vis::Adv),
    ("out", true, Vis::Adv),
    ("out-dir", true, Vis::Adv),
    ("out-x", true, Vis::Adv),
    ("output-prompts", false, Vis::Adv),
    ("payload-bytes", false, Vis::Adv),
    ("peer", true, Vis::Adv),
    ("peer-ips", true, Vis::Adv),
    ("penalty-range", false, Vis::Adv),
    ("pixels", true, Vis::Adv),
    ("plan-deals", true, Vis::Adv),
    ("plan-worlds", true, Vis::Adv),
    ("poison", false, Vis::Adv),
    ("pool-census", false, Vis::Adv),
    ("por", false, Vis::Adv),
    ("port", true, Vis::Day),
    ("positions", false, Vis::Adv),
    ("pp-bench-prefill", false, Vis::Adv),
    ("pp-chunk-list", true, Vis::Adv),
    ("pp-node", false, Vis::Adv),
    ("ppl", false, Vis::Adv),
    ("ppl-ref-windowz", false, Vis::Adv),
    ("ppl-zzzz-qqqq", false, Vis::Adv),
    ("prefill-chunk", true, Vis::Adv),
    ("prefix", false, Vis::Adv),
    ("prefix-cache", true, Vis::Day),
    ("prefix-ckpt-mem-gb", false, Vis::Adv),
    ("print-config", false, Vis::Adv),
    ("probe-bandwidth", false, Vis::Adv),
    ("probe-bandwidth-sustained", false, Vis::Adv),
    ("probe-binv", false, Vis::Adv),
    ("probe-df2-draft", false, Vis::Adv),
    ("probe-df2-graph", false, Vis::Adv),
    ("probe-df2-lossless", false, Vis::Adv),
    ("probe-df2-prime", false, Vis::Adv),
    ("probe-df2-round", true, Vis::Adv),
    ("probe-df2-tapcap", false, Vis::Adv),
    ("probe-dflash", true, Vis::Adv),
    ("probe-dflash-local", false, Vis::Adv),
    ("probe-dflash2", false, Vis::Adv),
    ("probe-dispatch", false, Vis::Adv),
    ("probe-dspark", false, Vis::Adv),
    ("probe-dspark-bind", true, Vis::Adv),
    ("probe-dspark-rollback", false, Vis::Adv),
    ("probe-dspark-round", true, Vis::Adv),
    ("probe-dspark-synth", true, Vis::Adv),
    ("probe-dsv4", false, Vis::Adv),
    ("probe-exl3", false, Vis::Adv),
    ("probe-exl3-binv", false, Vis::Adv),
    ("probe-exl3-bitwidths", false, Vis::Adv), // PR#27-D3: synthetic width-matrix gate (no model dir)
    ("probe-exl3-flashpf", false, Vis::Adv),
    ("probe-exl3-hcw4", false, Vis::Adv),
    ("probe-exl3-kernel", false, Vis::Adv),
    ("probe-exl3-kvnll", false, Vis::Adv),
    ("probe-exl3-kvq", false, Vis::Adv),
    ("probe-exl3-ltrows", false, Vis::Adv),
    ("probe-exl3-moeglue", false, Vis::Adv),
    ("probe-exl3-oracle", false, Vis::Adv),
    ("probe-exl3-pack2", false, Vis::Adv),
    ("probe-exl3-pack3", false, Vis::Adv),
    ("probe-exl3-packstate", false, Vis::Adv),
    ("probe-exl3-pfxtrace", false, Vis::Adv),
    ("probe-exl3-prefillx", false, Vis::Adv),
    ("probe-exl3-regime", false, Vis::Adv),
    ("probe-exl3-routergemm", false, Vis::Adv),
    ("probe-exl3-slice", false, Vis::Adv),
    ("probe-exl3-sppin", false, Vis::Adv),
    ("probe-exl3-tailgemm", false, Vis::Adv),
    ("probe-exl3-tpspec", false, Vis::Adv),
    ("probe-exl3-wide", false, Vis::Adv),
    ("probe-exl3-wp16", false, Vis::Adv),
    ("probe-exl3-xchain", false, Vis::Adv),
    ("probe-exl3-xtp", false, Vis::Adv),
    ("probe-fold-xchain-compare", true, Vis::Adv),
    ("probe-fp4l", false, Vis::Adv),
    ("probe-fp4l-gpu", false, Vis::Adv),
    ("probe-gemm", false, Vis::Adv),
    ("probe-moe", false, Vis::Adv),
    ("probe-moe-gemm", false, Vis::Adv),
    ("probe-mxfp4", false, Vis::Adv),
    ("probe-mxfp4-fused", false, Vis::Adv),
    ("probe-mxfp4-xchain", false, Vis::Adv),
    ("probe-mxfp4-xchain2", true, Vis::Adv),
    ("probe-mxfp4-xchain2-compare", true, Vis::Adv),
    ("probe-naive-a1-ab", false, Vis::Adv),
    ("probe-naive-attn-prefill", false, Vis::Adv),
    ("probe-naive-chunkxcpu", false, Vis::Adv),
    ("probe-naive-cpu", false, Vis::Adv),
    ("probe-naive-dequant", false, Vis::Adv),
    ("probe-naive-drafter", false, Vis::Adv),
    ("probe-naive-full", false, Vis::Adv),
    ("probe-naive-fuse", false, Vis::Adv),
    ("probe-naive-gemm", false, Vis::Adv),
    ("probe-naive-gemm-bench", false, Vis::Adv),
    ("probe-naive-graph", false, Vis::Adv),
    ("probe-naive-idx-prefill", false, Vis::Adv),
    ("probe-naive-lanes", false, Vis::Adv),
    ("probe-naive-loop", false, Vis::Adv),
    ("probe-naive-oracle", false, Vis::Adv),
    ("probe-naive-pack", false, Vis::Adv),
    ("probe-naive-prefill", false, Vis::Adv),
    ("probe-naive-sampler", false, Vis::Adv),
    ("probe-naive-shard", false, Vis::Adv),
    ("probe-naive-shard-dir", false, Vis::Adv),
    ("probe-naive-slice", false, Vis::Adv),
    ("probe-naive-slots", false, Vis::Adv),
    ("probe-naive-spec", false, Vis::Adv),
    ("probe-naive-topk", false, Vis::Adv),
    ("probe-naive-tp", false, Vis::Adv),
    ("probe-naive-tp-engine", false, Vis::Adv),
    ("probe-penalties", false, Vis::Adv),
    ("probe-ppsplit", false, Vis::Adv),
    ("probe-q4", false, Vis::Adv),
    ("probe-reject", false, Vis::Adv),
    ("probe-sampler", false, Vis::Adv),
    ("probe-serve-pinned", false, Vis::Adv),
    ("probe-shard-plan", false, Vis::Adv),
    ("probe-spec-sampling", false, Vis::Adv),
    ("probe-splitk", false, Vis::Adv),
    ("probe-state", false, Vis::Adv),
    ("probe-tp-gemv", false, Vis::Adv),
    ("probe-tq", true, Vis::Adv),
    ("probe-verify-m8", false, Vis::Adv),
    ("probe-vision-tower", false, Vis::Adv),
    ("profile-mtp", false, Vis::Adv),
    ("prompt", true, Vis::Adv),
    ("prompt-file", true, Vis::Adv),
    ("prompt-ids", true, Vis::Adv),
    ("prompt-len", true, Vis::Adv),
    ("proxy-core", false, Vis::Adv),
    ("proxy-core2", false, Vis::Adv),
    ("quantize", false, Vis::Adv),
    ("quick", false, Vis::Adv),
    ("rank", true, Vis::Adv),
    ("real-taps", true, Vis::Adv),
    ("reasoning-effort", true, Vis::Adv),
    ("recipe", true, Vis::Adv),
    ("reduces", false, Vis::Adv),
    ("remove-model-blob", false, Vis::Adv),
    ("replay-df2-bisect", false, Vis::Adv),
    ("replay-df2-round", false, Vis::Adv),
    ("reps", true, Vis::Adv),
    ("requant-gdn", false, Vis::Adv),
    ("requant-q2fake", false, Vis::Adv),
    ("requant-sim", false, Vis::Adv),
    ("rounds", true, Vis::Adv),
    ("rows-chunk", true, Vis::Adv),
    ("runs", true, Vis::Adv),
    ("scope", true, Vis::Adv),
    ("seconds", true, Vis::Adv),
    ("seed", true, Vis::Adv),
    ("seeds", true, Vis::Adv),
    ("seq-len", true, Vis::Adv),
    ("served-model-name", true, Vis::Adv),
    ("server", false, Vis::Day),
    ("server-dspark", true, Vis::Adv),
    ("server-port", true, Vis::Adv),
    ("sha256", true, Vis::Adv),
    ("shapes", true, Vis::Adv),
    ("shard", true, Vis::Adv),
    ("shard-dir", true, Vis::Adv),
    ("shard-end", true, Vis::Adv),
    ("shard-rank", true, Vis::Adv),
    ("shard-start", true, Vis::Adv),
    ("shard-world", true, Vis::Adv),
    ("sizes-mb", true, Vis::Adv),
    ("slice-experts", true, Vis::Adv),
    ("spacing-us", false, Vis::Adv),
    ("spec-block", true, Vis::Adv),
    ("spec-dds", true, Vis::Adv),
    ("spec-depths", true, Vis::Adv),
    ("spec-gen", false, Vis::Adv),
    ("spec-ident", true, Vis::Adv),
    ("spec-mode", true, Vis::Adv),
    ("spec-prefix", true, Vis::Adv),
    ("spec-prompts", true, Vis::Adv),
    ("spec-reps", false, Vis::Adv),
    ("spec-samp", true, Vis::Adv),
    ("spec-sampling", false, Vis::Adv),
    ("spec-source", true, Vis::Adv),
    ("spec-trials", false, Vis::Adv),
    ("split", true, Vis::Adv),
    ("sppin-c", true, Vis::Adv),
    ("sppin-world", true, Vis::Adv),
    ("stall-consumer-every", false, Vis::Adv),
    ("stall-us", false, Vis::Adv),
    ("step", true, Vis::Adv),
    ("steps", true, Vis::Adv),
    ("stq-bake", false, Vis::Adv),
    ("streaming", false, Vis::Adv),
    ("sweep-gemm", false, Vis::Adv),
    ("taps", true, Vis::Adv),
    ("temp", true, Vis::Adv),
    ("temperature", true, Vis::Adv),
    ("thinking", true, Vis::Day),
    ("thinking-on", false, Vis::Adv),
    ("time-widths", true, Vis::Adv),
    ("tmp", true, Vis::Adv),
    ("tokenizer", true, Vis::Adv),
    ("tokens", true, Vis::Adv),
    ("tool-call-format", true, Vis::Adv),
    ("top-k", true, Vis::Adv),
    ("top-p", true, Vis::Adv),
    ("tp", false, Vis::Day),
    ("tp-barrier-bench", false, Vis::Adv),
    ("tp-reduce-bench", false, Vis::Adv),
    ("tp-sim", false, Vis::Adv),
    ("tp-sim-full", false, Vis::Adv),
    ("tp-sim-layer", true, Vis::Adv),
    ("tree-draft", true, Vis::Adv),
    ("trials", true, Vis::Adv),
    ("tripwire", true, Vis::Adv),
    ("tripwire-selftest", false, Vis::Adv),
    ("tune-draft", false, Vis::Adv),
    ("tune-profile", false, Vis::Adv),
    ("tune-table", false, Vis::Adv),
    ("update", false, Vis::Adv),
    ("verify", false, Vis::Adv),
    ("version", false, Vis::Adv),
    ("vision-fail-gemm-x", false, Vis::Adv),
    ("warmup", false, Vis::Adv),
    ("with-trunk", false, Vis::Adv),
    ("world", true, Vis::Adv),
    ("wp16-b-ids", true, Vis::Adv),
    ("wp16-split", true, Vis::Adv),
    ("x-in", true, Vis::Adv),
    ("xport", false, Vis::Adv),
    ("xtp-gates", true, Vis::Adv),
    ("xtp-gen", true, Vis::Adv),
    ("xtp-ident-only", false, Vis::Adv),
    ("xtp-out", true, Vis::Adv),
    ("xtp-pf-chunk", true, Vis::Adv),
    ("xtp-ref", true, Vis::Adv),
    ("xtp-sp-fault", true, Vis::Adv),
    ("xtp-time", true, Vis::Adv),
    ("y-out", true, Vis::Adv),
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
        let Some(&(flag, takes_value, _vis)) = HAND_FLAGS.iter().find(|(f, _, _)| *f == name) else { i += 1; continue };
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
        let known = lookup(body).is_some() || HAND_FLAGS.iter().any(|(f, _, _)| *f == body);
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
            HAND_FLAGS.iter().any(|(f, v, _)| *f == body && *v)
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
        .chain(HAND_FLAGS.iter().map(|(f, _, _)| (edit_distance(name, f), *f)))
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

/// CLIPRUNE1: the three option views. `--help` = day-to-day launch set only (registry rows with
/// `vis: Vis::Day` + the day-to-day hand-read launch flags). `--help-diag` = the diagnostic /
/// test-drill group (unchanged view). `--help-advanced` = EVERYTHING the binary accepts: every
/// registry row (diagnostics included, grouped by category), the internally-set rows, and every
/// hand-read mode/bench flag. Drift in the two owner-facing views is pinned by snapshot fixtures
/// (tests/fixtures/help_{day,advanced}.txt; refresh with `cargo test -- --bless`).
pub fn help_day() -> String {
    let mut s = String::from("\nDAY-TO-DAY LAUNCH OPTIONS (everything the binary accepts: --help-advanced; resolved values: --print-config)\n");
    let mut hand: Vec<&str> = HAND_FLAGS.iter().filter(|(_, _, v)| *v == Vis::Day).map(|(f, _, _)| *f).collect();
    hand.sort();
    if !hand.is_empty() {
        s += "\n  launch flags (read at boot)\n";
        for f in hand {
            let takes = HAND_FLAGS.iter().any(|(hf, t, _)| *hf == f && *t);
            s += &format!("    --{}{}\n", f, if takes { " <value>" } else { "" });
        }
    }
    for c in Cat::ALL {
        let mut rows: Vec<&Opt> = REG.iter()
            .filter(|o| o.cat == c && o.vis == Vis::Day && !matches!(o.ty, Ty::Internal(_))).collect();
        rows.sort_by_key(|o| o.flag);
        if rows.is_empty() { continue; }
        s += &format!("\n  {}\n", c.title());
        for o in rows { s += &help_line(o); s.push('\n'); }
    }
    let n_adv = REG.iter().filter(|o| o.vis == Vis::Adv).count()
        + HAND_FLAGS.iter().filter(|(_, _, v)| *v == Vis::Adv).count();
    s += &format!("\n  {n_adv} further options (diagnostics, test drills, probes, tunables): --help-advanced.   Resolved values: --print-config.\n");
    s += F7_NOTE;
    s
}

/// `--help-diag` (unchanged): the diagnostic / test-drill group only.
pub fn help_text(diag: bool) -> String {
    debug_assert!(diag, "CLIPRUNE1: --help is help_day(); help_text(false) is retired");
    let mut s = String::from("\nDIAGNOSTIC / TEST-DRILL OPTIONS (every engine option is a flag; the engine reads no env vars)\n");
    for c in Cat::ALL {
        if c != Cat::Diag { continue; }
        let mut rows: Vec<&Opt> = REG.iter().filter(|o| o.cat == c && !matches!(o.ty, Ty::Internal(_))).collect();
        rows.sort_by_key(|o| o.flag);
        if rows.is_empty() { continue; }
        s += &format!("\n  {}\n", c.title());
        for o in rows { s += &help_line(o); s.push('\n'); }
    }
    s += INTERNAL_NOTE;
    for o in REG.iter().filter(|o| matches!(o.ty, Ty::Internal(_))) {
        if let Ty::Internal(via) = o.ty { s += &format!("    {:<34} {} (via {via})\n", o.flag, o.help); }
    }
    s += F7_NOTE;
    s
}

/// `--help-advanced`: every registry row (grouped, diagnostics included), the internally-set
/// rows, and every hand-read mode/bench flag — the complete accepted option set.
pub fn help_advanced() -> String {
    let mut s = String::from("\nALL OPTIONS (the complete accepted set; the day-to-day launch subset: --help)\n");
    for c in Cat::ALL {
        let mut rows: Vec<&Opt> = REG.iter().filter(|o| o.cat == c && !matches!(o.ty, Ty::Internal(_))).collect();
        rows.sort_by_key(|o| o.flag);
        if rows.is_empty() { continue; }
        s += &format!("\n  {}\n", c.title());
        for o in rows { s += &help_line(o); s.push('\n'); }
    }
    s += INTERNAL_NOTE;
    let mut all_hand: Vec<(&str, bool, bool)> = HAND_FLAGS.iter()
        .map(|(f, t, v)| (*f, *t, *v == Vis::Day)).collect();
    all_hand.sort();
    s += &format!("\n  MODE / BENCH / PROBE FLAGS (hand-read by the mode dispatch; [day-to-day] ones are the launch set in --help)\n");
    for (f, takes, day) in all_hand {
        s += &format!("    --{}{}{}\n", f, if takes { " <value>" } else { "" }, if day { "   [day-to-day]" } else { "" });
    }
    s += F7_NOTE;
    s
}

const F7_NOTE: &str = "\n  A value that really starts with \"--\" must be written --option=value; a value-taking option is never taken from the next --flag - no value-taking option, registry or hand-read, ever eats one (F7/F9).\n";

const INTERNAL_NOTE: &str = "\n  Set internally (not command-line options; shown by --print-config):\n";

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
mod tests;
