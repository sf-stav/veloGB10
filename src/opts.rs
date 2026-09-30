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
    Float,
    Text,
    Path,
    /// one of the listed spellings (stored raw)
    Choice(&'static [&'static str]),
    /// not a command-line option: written by the named flag / by the binary itself
    Internal(&'static str),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cat { Serve, Tp, Exl3, Spec, Kv, Nvfp4, Dsv4, Load, Diag }

impl Cat {
    pub const ALL: [Cat; 9] = [Cat::Serve, Cat::Tp, Cat::Kv, Cat::Spec, Cat::Exl3, Cat::Nvfp4, Cat::Dsv4, Cat::Load, Cat::Diag];
    pub fn title(self) -> &'static str {
        match self {
            Cat::Serve => "SERVING", Cat::Tp => "TENSOR PARALLEL", Cat::Exl3 => "EXL3 ENGINE LEVERS",
            Cat::Spec => "SPECULATION", Cat::Kv => "KV / PREFIX CACHE", Cat::Nvfp4 => "NVFP4 / MXFP4 ENGINE LEVERS",
            Cat::Dsv4 => "DSV4 / DSPARK", Cat::Load => "LOADING / MEMORY", Cat::Diag => "DIAGNOSTICS / TEST DRILLS",
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

/// Set and not "0" / "off" / "false" / empty.
pub fn on(o: OptId) -> bool {
    var(o).map_or(false, |v| !matches!(v.trim(), "" | "0" | "off" | "false" | "OFF"))
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
        let a = &args[i];
        i += 1;
        let Some(body) = a.strip_prefix("--") else { continue };
        let (name, inline) = match body.split_once('=') { Some((n, v)) => (n, Some(v)), None => (body, None) };
        let Some(id) = lookup(name) else { continue };
        let o = id.def();
        if matches!(o.ty, Ty::Internal(_)) { continue; }
        let v: Option<&str> = match inline {
            Some(v) => Some(v),
            None if takes_next(o.ty, args.get(i)) => { i += 1; Some(args[i - 1].as_str()) }
            None => None,
        };
        let c = canon(o, v)?;
        with_store(|s| s[id.0 as usize] = c.map(|v| (v, Src::Cli)));
    }
    Ok(())
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
        Ty::Int => " <N>".into(),
        Ty::Float => " <X>".into(),
        Ty::Text => " <TEXT>".into(),
        Ty::Path => " <PATH>".into(),
        Ty::Choice(l) => format!(" <{}>", l.join("|")),
        Ty::Internal(_) => String::new(),
    }
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
}
