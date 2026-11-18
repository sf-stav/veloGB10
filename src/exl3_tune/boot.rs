//! Boot path: select a table among candidates, validate it strictly against the current fingerprint, freeze it and print the boot line.
//! A missing or mismatched table falls back to the defaults, which are bitwise today's behaviour. See `mod.rs` for the file map.

use super::*;

// ---------------------------------------------------------------------------------------------
// Boot: select, validate, freeze, print
// ---------------------------------------------------------------------------------------------

/// `--tune-table auto|off|<path>`.
#[derive(Clone, Debug, PartialEq)]
pub enum TableSel { Auto, Off, Path(PathBuf) }

impl TableSel {
    pub fn parse(v: Option<&str>) -> anyhow::Result<TableSel> {
        Ok(match v {
            None | Some("auto") => TableSel::Auto,
            Some("off") => TableSel::Off,
            Some(p) if !p.is_empty() && !p.starts_with("--") => TableSel::Path(PathBuf::from(p)),
            Some(other) => anyhow::bail!("--tune-table must be auto, off or a table path (got '{other}')"),
        })
    }
}

pub struct BootReq<'a> {
    pub sel: TableSel,
    pub model_dir: &'a str,
    pub posture: Posture,
    /// `--tune-profile <MHz>` (None = detect, else the stock profile)
    pub profile_mhz: Option<u32>,
    /// `--tune-draft on`: also apply section D (draft-only; seeded sampled bytes may shift)
    pub draft_on: bool,
    pub tp: u32,
}

fn round100(x: f64) -> u32 { ((x / 100.0).round() * 100.0).max(0.0) as u32 }

/// Best-effort SM-clock profile of this box: the application-clock target from nvidia-smi (2 s cap),
/// rounded to 100 MHz; None when unavailable ("GPU counters are optional, never required").
pub fn detect_clock_profile() -> Option<u32> {
    let out = run_timeout("nvidia-smi", &["--query-gpu=clocks.applications.graphics", "--format=csv,noheader,nounits"],
                          std::time::Duration::from_secs(2))?;
    let v: f64 = out.lines().next()?.trim().parse().ok()?;
    let r = round100(v);
    if (1000..=3200).contains(&r) { Some(r) } else { None }
}

/// Run a command with a wall-clock cap; stdout on success, None on error / timeout (killed).
pub fn run_timeout(cmd: &str, args: &[&str], cap: std::time::Duration) -> Option<String> {
    use std::io::Read;
    let mut child = std::process::Command::new(cmd).args(args)
        .stdin(std::process::Stdio::null()).stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null()).spawn().ok()?;
    let t0 = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(st)) => {
                let mut s = String::new();
                child.stdout.take()?.read_to_string(&mut s).ok()?;
                return if st.success() { Some(s) } else { None };
            }
            Ok(None) if t0.elapsed() < cap => std::thread::sleep(std::time::Duration::from_millis(20)),
            _ => { let _ = child.kill(); let _ = child.wait(); return None; }
        }
    }
}

/// The table boot picked (select_table).
pub struct Pick {
    pub path: PathBuf,
    pub bytes: Vec<u8>,
    pub table: Table,
}

/// Pure table selection (§7.2): parse every candidate file, drop the ones whose format / seal /
/// strict fingerprint + posture does not match `have`, then prefer a user-forced table over a
/// shipped one, then the nearest clock profile, then file order. Err = the FALLBACK reason (the
/// last mismatch seen, or "no table matched").
pub fn select_table(files: Vec<(PathBuf, Vec<u8>)>, have: &Fingerprint, posture: &Posture, profile: u32)
    -> Result<Pick, String> {
    select_table_x(files, have, posture, profile, false)
}

/// `select_table` with `explicit` = the table was named by `--tune-table <path>`: only then may a
/// table whose final gate did not pass (confirm.pass != true: ungated, rejected or hand-written)
/// load — `auto` never picks one (a rejected table must never serve by accident).
pub fn select_table_x(files: Vec<(PathBuf, Vec<u8>)>, have: &Fingerprint, posture: &Posture, profile: u32,
                      explicit: bool) -> Result<Pick, String> {
    if files.is_empty() { return Err("no table in ./tune".into()); }
    let mut best_reason = String::from("no table matched");
    let mut picks: Vec<Pick> = Vec::new();
    for (path, bytes) in files {
        let table: Table = match serde_json::from_slice(&bytes) {
            Ok(t) => t,
            Err(e) => { best_reason = format!("parse error in {}: {e}", path.display()); continue; }
        };
        if table.format != TABLE_FORMAT {
            best_reason = format!("format {} in {}", table.format, path.display());
            continue;
        }
        if !table.seal_ok() {
            best_reason = format!("table_sha256 ({}: content edited after sealing)", path.display());
            continue;
        }
        if !explicit && !confirm_passed(&table) {
            best_reason = format!("unconfirmed ({}: its final gate did not pass — load it only by explicit --tune-table <path>)",
                                  path.display());
            continue;
        }
        if let Err(r) = strict_match(&table.fingerprint, have, &table.posture, posture) {
            best_reason = format!("{r} ({})", path.display());
            continue;
        }
        picks.push(Pick { path, bytes, table });
    }
    picks.sort_by_key(|p| ((p.table.origin != "user") as u8,
                           (p.table.fingerprint.clock_mhz as i64 - profile as i64).abs()));
    picks.into_iter().next().ok_or(best_reason)
}

/// Did the table's final gate pass? `confirm.pass == true` (every writer since L10, including a
/// clean 0-decision run). Back-compat: a table written before L10 by a 0-decision run carries
/// `{"skipped": "no decision adopted"}` with no `pass` field — it is the built-in defaults and
/// passes too, but ONLY with zero decisions and no explicit `pass` (an explicit `pass: false`,
/// e.g. `--autotune-skip-gate` or a rejected gate, is never overridden).
pub fn confirm_passed(t: &Table) -> bool {
    match t.confirm.get("pass") {
        Some(v) => v.as_bool() == Some(true),
        None => t.decisions.is_empty()
            && t.confirm.get("skipped").and_then(|v| v.as_str()) == Some("no decision adopted"),
    }
}

fn scan_tables(sel: &TableSel) -> Result<Vec<(PathBuf, Vec<u8>)>, String> {
    match sel {
        TableSel::Off => Ok(Vec::new()),
        TableSel::Path(p) => std::fs::read(p).map(|b| vec![(p.clone(), b)])
            .map_err(|e| format!("cannot read {}: {e}", p.display())),
        TableSel::Auto => {
            let mut v: Vec<PathBuf> = match std::fs::read_dir("tune") {
                Ok(rd) => rd.filter_map(|e| e.ok()).map(|e| e.path())
                    .filter(|p| p.is_file() && p.extension().map_or(false, |x| x == "json"))
                    .collect(),
                Err(_) => Vec::new(),
            };
            v.sort();
            Ok(v.into_iter().filter_map(|p| std::fs::read(&p).ok().map(|b| (p, b))).collect())
        }
    }
}

/// Validate a table's decisions against the registry; returns (per-slot decision maps, #S, #D, ignored).
pub(super) fn build_decisions(t: &Table, draft_on: bool) -> (Vec<HashMap<Key, i32>>, usize, usize, Vec<String>) {
    let mut dec: Vec<HashMap<Key, i32>> = (0..REGISTRY.len()).map(|_| HashMap::new()).collect();
    let (mut ns, mut nd) = (0usize, 0usize);
    let mut ignored = Vec::new();
    for d in &t.decisions {
        let Some(def) = REGISTRY.iter().find(|r| r.id == d.id) else {
            ignored.push(format!("{} (unknown id)", d.id));
            continue;
        };
        if def.class.name() != d.class {
            ignored.push(format!("{} (class {} != registry {})", d.id, d.class, def.class.name()));
            continue;
        }
        if d.rev != def.rev {
            ignored.push(format!("{} (rev {} != {})", d.id, d.rev, def.rev));
            continue;
        }
        if def.xcheck.is_empty() {
            // §4.7.3: no bit-diff harness proves this domain — never tuned, never loaded
            ignored.push(format!("{} (DOMAIN UNPROVEN: no xcheck harness)", d.id));
            continue;
        }
        match def.class {
            Class::S => {}
            Class::D if draft_on => {}
            Class::D => { ignored.push(format!("{} (section D: needs --tune-draft on)", d.id)); continue; }
            Class::N => { ignored.push(format!("{} (section N: numerics profiles are never auto-loaded)", d.id)); continue; }
            Class::P => { ignored.push(format!("{} (class P: policy is not a table value)", d.id)); continue; }
        }
        if !domain_has(def.domain, d.value) {
            ignored.push(format!("{}={} (outside domain {:?})", d.id, d.value, def.domain));
            continue;
        }
        let keys = match decision_keys(def, &d.scope) {
            Ok(k) => k,
            Err(e) => { ignored.push(format!("{} ({e})", d.id)); continue; }
        };
        let mut ok = true;
        for k in &keys {
            let cx = ctx_of_key(k);
            if !(def.valid)(&cx, d.value) { ok = false; }
        }
        if !ok { ignored.push(format!("{}={} (fails valid())", d.id, d.value)); continue; }
        for k in keys { dec[def.slot as usize].insert(k, d.value); }
        if matches!(def.class, Class::D) { nd += 1; } else { ns += 1; }
    }
    (dec, ns, nd, ignored)
}

/// Expand a decision's scope into registry keys (lists of m / chunk classes fan out).
pub fn decision_keys(def: &TunableDef, s: &DecScope) -> Result<Vec<Key>, String> {
    let regime = |r: &Option<String>| -> Result<u8, String> {
        match r.as_deref() { None | Some("dense") => Ok(0), Some("sparse") => Ok(1), Some(o) => Err(format!("regime '{o}'")) }
    };
    Ok(match def.scope {
        Scope::Global | Scope::Load => vec![Key::G],
        Scope::Fam => {
            let f = s.fam.as_deref().and_then(Fam::parse).ok_or("Fam scope needs fam")?;
            vec![Key::F(f as u8)]
        }
        Scope::Shape => {
            let (k, n, b) = (s.k.ok_or("Shape scope needs k")?, s.n.ok_or("Shape scope needs n")?, s.bits.ok_or("Shape scope needs bits")?);
            vec![Key::S(k, n, b)]
        }
        Scope::Width => {
            let f = s.fam.as_deref().and_then(Fam::parse).ok_or("Width scope needs fam")?;
            let r = regime(&s.regime)?;
            let ms = s.m.as_ref().filter(|v| !v.is_empty()).ok_or("Width scope needs m")?;
            ms.iter().map(|&m| Key::W(f as u8, m, r)).collect()
        }
        Scope::Chunk => {
            let r = regime(&s.regime)?;
            let cs = s.chunk.as_ref().filter(|v| !v.is_empty()).ok_or("Chunk scope needs chunk")?;
            cs.iter().map(|&c| Key::C(c, r)).collect()
        }
    })
}

/// Would boot accept `value` for `def` at `key`? (the valid() predicate at the key's context)
pub fn key_valid(def: &TunableDef, key: &Key, value: i32) -> bool {
    (def.valid)(&ctx_of_key(key), value)
}

/// The overlay a table puts in force at boot: exactly `build_decisions` (class / rev / domain /
/// unproven / valid() filtering), flattened to keyed assignments. The tuner's final gate runs
/// this, so it measures the table boot will load, not the search's view of it.
pub fn table_assign(t: &Table, draft_on: bool) -> (Vec<(&'static TunableDef, Option<Key>, i32)>, Vec<String>) {
    let (dec, _, _, ignored) = build_decisions(t, draft_on);
    let mut v = Vec::new();
    for (i, m) in dec.iter().enumerate() {
        let mut keys: Vec<(&Key, &i32)> = m.iter().collect();
        keys.sort_by_key(|(k, _)| format!("{k:?}"));
        for (k, val) in keys { v.push((REGISTRY[i], Some(*k), *val)); }
    }
    (v, ignored)
}

fn ctx_of_key(k: &Key) -> Ctx {
    let f = |i: u8| FAMS.get(i as usize).copied().unwrap_or(Fam::Util);
    match *k {
        Key::G => Ctx::UTIL,
        Key::F(i) => Ctx { fam: f(i), ..Ctx::UTIL },
        Key::S(k, n, b) => Ctx { k, n, bits: b, ..Ctx::UTIL },
        Key::W(i, m, r) => Ctx { fam: f(i), m, regime: r, ..Ctx::UTIL },
        Key::C(c, r) => Ctx { fam: Fam::Prefill, chunk_class: c, regime: r, ..Ctx::UTIL },
    }
}

fn freeze_with(dec: Vec<HashMap<Key, i32>>, status: BootStatus) -> bool {
    let env = env_cache();
    let entries: Vec<Entry> = REGISTRY.iter().enumerate().map(|(i, t)| {
        let d = dec.get(i).cloned().unwrap_or_default();
        let e = env.get(i).copied().flatten();
        Entry { env: e, fast: if d.is_empty() { Some(e.unwrap_or(t.default)) } else { None }, dec: d }
    }).collect();
    FROZEN.set(Frozen { entries, status }).is_ok()
}

/// Print the one-line tune status (+ the env overrides) — the first log line of a tuned process.
fn print_boot_line(st: &BootStatus) {
    let ov = env_overrides();
    let head = match st.state {
        "OK" => format!("TUNE: {} OK ({}, profile {} MHz tp{}, S decisions {} in force, D {}; early_lookups {})",
                        st.table_sha8.as_deref().unwrap_or("?"), st.table_path.as_deref().unwrap_or("?"),
                        st.profile_mhz, st.tp, st.decisions_s,
                        if st.decisions_d > 0 { format!("{} in force (--tune-draft on)", st.decisions_d) } else { "off".into() },
                        early_lookups()),
        "OFF" => format!("TUNE: OFF ({}) — built-in defaults; early_lookups {}", st.reason, early_lookups()),
        "DEFAULTS" => format!("TUNE: DEFAULTS ({}) — built-in defaults; early_lookups {}", st.reason, early_lookups()),
        _ => format!("TUNE: FALLBACK({}) — built-in defaults for everything; early_lookups {}", st.reason, early_lookups()),
    };
    if crate::boot_report::debug() {
    crate::dprintln!("{head}");
    if !st.ignored.is_empty() {
        crate::dprintln!("TUNE: {} table decision(s) ignored: {}", st.ignored.len(), st.ignored.join("; "));
    }
    if !ov.is_empty() {
        let s: Vec<String> = ov.iter().map(|(id, env, v)| format!("{env} -> {id}={v}")).collect();
        crate::dprintln!("TUNE: OVERRIDDEN(flag: {}) — override flags are diagnostics; they win over the table", s.join(", "));
    }
    crate::dprintln!("TUNE: registry {} entries, schema {}, tune_build_id {}",
             REGISTRY.len(), &registry_schema()[..12], env!("TUNE_BUILD_ID"));
    }
}

/// Resolve the table for this process, freeze it, print the boot line. Call BEFORE FwdModel::load.
/// Never fails the boot: every problem is a loud FALLBACK to the built-in defaults (bitwise).
pub fn boot(req: &BootReq) {
    if FROZEN.get().is_some() {
        crate::dprintln!("TUNE: boot requested after the table was frozen — keeping the frozen table");
        return;
    }
    let tp = req.tp.max(1);
    let base = BootStatus {
        state: "FALLBACK", reason: String::new(), table_path: None, table_sha8: None,
        decisions_s: 0, decisions_d: 0, ignored: Vec::new(), profile_mhz: STOCK_PROFILE_MHZ, tp,
    };
    if req.sel == TableSel::Off {
        let st = BootStatus { state: "OFF", reason: "--tune-table off".into(), ..base };
        freeze_with(Vec::new(), st.clone());
        print_boot_line(&st);
        return;
    }
    let files = match scan_tables(&req.sel) {
        Ok(f) => f,
        Err(e) => {
            let st = BootStatus { reason: e, ..base };
            freeze_with(Vec::new(), st.clone());
            print_boot_line(&st);
            return;
        }
    };
    if files.is_empty() {
        let st = BootStatus { reason: "no table in ./tune".into(), ..base };
        freeze_with(Vec::new(), st.clone());
        print_boot_line(&st);
        return;
    }
    let (profile, psrc) = match req.profile_mhz {
        Some(p) => (p, "--tune-profile"),
        None => match detect_clock_profile() {
            Some(p) => (p, "nvidia-smi application clock"),
            None => (STOCK_PROFILE_MHZ, "stock default"),
        },
    };
    let have = current_fingerprint(req.model_dir, tp, profile);
    // prefer a user-forced table over a shipped one, then the nearest clock profile, then file order
    let explicit = matches!(req.sel, TableSel::Path(_));
    let pick = match select_table_x(files, &have, &req.posture, profile, explicit) {
        Ok(p) => p,
        Err(best_reason) => {
            let st = BootStatus { reason: best_reason, profile_mhz: profile, ..base };
            freeze_with(Vec::new(), st.clone());
            print_boot_line(&st);
            return;
        }
    };
    if pick.table.fingerprint.clock_mhz != profile {
        crate::dprintln!("TUNE: no {profile} MHz profile table ({psrc}); nearest is {} MHz — loading it",
                 pick.table.fingerprint.clock_mhz);
    }
    if pick.table.fingerprint.driver != have.driver || pick.table.fingerprint.cuda != have.cuda {
        println!("TUNE: WARNING driver/CUDA differ (table {} / {}, box {} / {}) — loading (S values are \
                  output-neutral by construction; re-tune recommended: a new driver JIT can change codegen)",
                 pick.table.fingerprint.driver, pick.table.fingerprint.cuda, have.driver, have.cuda);
    }
    let (dec, ns, nd, ignored) = build_decisions(&pick.table, req.draft_on);
    let st = BootStatus {
        state: "OK", reason: String::new(),
        table_path: Some(pick.path.display().to_string()),
        table_sha8: Some(sha256_hex(&pick.bytes)[..8].to_string()),
        decisions_s: ns, decisions_d: nd, ignored, profile_mhz: pick.table.fingerprint.clock_mhz, tp,
    };
    freeze_with(dec, st.clone());
    print_boot_line(&st);
}

/// §4.1 `freeze()`: make the resolved table immutable for the life of the process. `boot` freezes
/// what it resolved; a bare `freeze()` (nothing booted) freezes the built-in defaults. Idempotent.
pub fn freeze() { ensure_frozen("tune::freeze"); }

/// Freeze the built-in defaults if no entry point booted a table (probe / bench paths). Idempotent.
pub fn ensure_frozen(who: &str) {
    if FROZEN.get().is_some() { return; }
    let st = BootStatus {
        state: "DEFAULTS", reason: format!("{who}: this entry point loads no table"), table_path: None,
        table_sha8: None, decisions_s: 0, decisions_d: 0, ignored: Vec::new(),
        profile_mhz: STOCK_PROFILE_MHZ, tp: 1,
    };
    if freeze_with(Vec::new(), st.clone()) { print_boot_line(&st); }
}
