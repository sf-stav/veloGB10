//! Table output (`table_file_name`, `write_table`, `report`) and the EXL3 stale-kernel handshake (`ptx_build_id`, `assert_gpu_batch_build_id`).
//! See `mod.rs` for the file map.

use super::*;

/// The table's canonical file name: tune/<model8>.<posture8>.tp<N>.<MHz>.json (the posture hash
/// keeps a 4K-context table from shadowing a 256K one; boot scans ./tune/*.json either way).
pub fn table_file_name(t: &Table) -> String {
    let m8 = t.fingerprint.model.config_sha256.get(..8).unwrap_or("nomodel").to_string();
    let p8 = sha256_hex(serde_json::to_string(&t.posture).unwrap_or_default().as_bytes())[..8].to_string();
    format!("tune/{m8}.{p8}.tp{}.{}.json", t.fingerprint.tp, t.fingerprint.clock_mhz)
}

/// Seal and write a table (pretty JSON); returns (path, file sha256).
pub fn write_table(t: &mut Table, path: &str) -> anyhow::Result<String> {
    t.seal();
    let s = serde_json::to_string_pretty(t)?;
    if let Some(dir) = Path::new(path).parent() {
        if !dir.as_os_str().is_empty() { std::fs::create_dir_all(dir)?; }
    }
    std::fs::write(path, s.as_bytes())?;
    Ok(sha256_hex(s.as_bytes()))
}

/// `--autotune-report <table>`: the human view — provenance, decisions with Δ / CI / runner-up,
/// the family statuses (INERT, UNPROVEN, UNTUNED (budget), OVERRIDDEN, ...), the final LI gate.
pub fn report(t: &Table, raw: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let fp = &t.fingerprint;
    let _ = writeln!(s, "TUNE TABLE {} (file sha256 {}, seal {})", TABLE_FORMAT, sha256_hex(raw),
                     if t.table_sha256.is_empty() { "none".to_string() }
                     else if t.seal_ok() { format!("{} OK", &t.table_sha256[..12.min(t.table_sha256.len())]) }
                     else { "MISMATCH (edited after sealing: boot refuses it)".to_string() });
    let _ = writeln!(s, "  origin {} | tune_build_id {} | registry {} | tp {} | clock profile {} MHz",
                     t.origin, fp.tune_build_id, fp.registry_schema.get(..12).unwrap_or(""), fp.tp, fp.clock_mhz);
    let _ = writeln!(s, "  gpu {} cc {} sms {} | driver {} cuda {} | model config {}",
                     fp.gpu.name, fp.gpu.cc, fp.gpu.sms, fp.driver, fp.cuda, fp.model.config_sha256.get(..12).unwrap_or(""));
    let _ = writeln!(s, "  posture {}", serde_json::to_string(&t.posture).unwrap_or_default());
    let p = &t.provenance;
    let g = |k: &str| p.get(k).map(|v| v.to_string()).unwrap_or_else(|| "-".into());
    let _ = writeln!(s, "  provenance: commit {} dirty {} binary {} box {} date {} elapsed_min {} budget_min {}",
                     g("commit"), g("dirty"), g("binary_sha256"), g("box"), g("date"), g("elapsed_min"), g("budget_min"));
    if let Some(nf) = p.get("noise_floor_ms") { let _ = writeln!(s, "  noise floor (ms): {nf}"); }
    if let Some(pr) = p.get("protocol") { let _ = writeln!(s, "  measurement protocol: {pr}"); }
    // §5.2 v2: the replay-position effect the run measured (per unit x width x regime x positions)
    if let Some(pe) = p.get("position_effect").and_then(|v| v.as_array()) {
        let _ = writeln!(s, "POSITION EFFECT ({} cell(s); ms; position 0 = the first timed replay after the warm-up):", pe.len());
        let arr = |v: Option<&serde_json::Value>| v.and_then(|x| x.as_array())
            .map(|a| a.iter().map(|x| x.as_f64().map_or("-".into(), |x| format!("{x:+.3}"))).collect::<Vec<_>>().join(", "))
            .unwrap_or_default();
        let arr_abs = |v: Option<&serde_json::Value>| v.and_then(|x| x.as_array())
            .map(|a| a.iter().map(|x| x.as_f64().map_or("-".into(), |x| format!("{x:.3}"))).collect::<Vec<_>>().join(", "))
            .unwrap_or_default();
        for c in pe {
            let g = |k: &str| c.get(k).map(|v| v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string())).unwrap_or_else(|| "-".into());
            let _ = writeln!(s, "  {:<10} m{:<2} {:<6} n_pos {:<2} rounds {:<5} warm-up {} | median by position [{}] | fitted offsets [{}] | spread {}",
                             g("unit"), g("m"), g("regime"), g("n_pos"), g("rounds"),
                             c.get("warmup_median_ms").and_then(|v| v.as_f64()).map_or("-".into(), |x| format!("{x:.3}")),
                             arr_abs(c.get("median_ms_by_pos")), arr(c.get("offsets_ms")),
                             c.get("spread_ms").and_then(|v| v.as_f64()).map_or("-".into(), |x| format!("{x:.3}")));
        }
    }
    let _ = writeln!(s, "DECISIONS ({}):", t.decisions.len());
    for d in &t.decisions {
        let ev = &d.evidence;
        let f = |k: &str| ev.get(k).and_then(|v| v.as_f64());
        let ci = ev.get("ci95_ms").and_then(|v| v.as_array())
            .map(|a| a.iter().map(|x| x.as_f64().map_or("?".into(), |x| format!("{x:+.4}"))).collect::<Vec<_>>().join(", "))
            .unwrap_or_default();
        let conf = ev.get("confirm").map(|c| c.to_string()).unwrap_or_default();
        let _ = writeln!(s, "  {:<22} = {:<6} (default {}) scope {} | unit {} n {} HL {} CI95 [{}] base {} ms | runner-up {} | digests ok {} | confirm {}",
                         d.id, d.value, d.default.map_or("-".into(), |v| v.to_string()), serde_json::to_string(&d.scope).unwrap_or_default(),
                         ev.get("unit").and_then(|v| v.as_str()).unwrap_or("?"),
                         ev.get("n").map(|v| v.to_string()).unwrap_or_default(),
                         f("hl_delta_ms").map_or("?".into(), |x| format!("{x:+.4}")), ci,
                         f("base_ms").map_or("?".into(), |x| format!("{x:.3}")),
                         ev.get("runner_up").map(|v| v.to_string()).unwrap_or_else(|| "-".into()),
                         ev.get("digest_ok").map(|v| v.to_string()).unwrap_or_default(), conf);
    }
    if let Some(fams) = t.families.as_object() {
        let _ = writeln!(s, "FAMILIES ({}):", fams.len());
        for (k, v) in fams {
            let st = v.get("status").and_then(|x| x.as_str()).unwrap_or("?");
            let _ = writeln!(s, "  {:<28} {}", k, st);
        }
    }
    if !t.confirm.is_null() {
        let _ = writeln!(s, "FINAL GATE: {}", t.confirm);
    }
    s
}

// ---------------------------------------------------------------------------------------------
// Stale-kernel handshake for the EXL3 modules (finding §14.1)
// ---------------------------------------------------------------------------------------------

/// The KERNEL_BUILD_ID baked into a PTX text's `kernel_build_id` entry (every integer literal of
/// the entry body is a candidate; the one equal to `expect` wins, else the first non-trivial one).
pub fn ptx_build_id(ptx: &str, expect: u64) -> Option<u64> {
    let at = ptx.find(".entry kernel_build_id(")?;
    let body_start = at + ptx[at..].find('{')?;
    let body_end = body_start + ptx[body_start..].find('}')?;
    let body = &ptx[body_start..body_end];
    let mut found: Vec<u64> = Vec::new();
    for tok in body.split(|c: char| c.is_whitespace() || c == ',' || c == ';' || c == '[' || c == ']') {
        // nvcc prints the 64-bit immediate as a SIGNED decimal (`mov.b64 %rd3, -7963248470597812263;`)
        let v = if let Some(h) = tok.strip_prefix("0x").or_else(|| tok.strip_prefix("0X")) {
            u64::from_str_radix(h.trim_end_matches(['U', 'L', 'u', 'l']), 16).ok()
        } else if let Some(d) = tok.strip_prefix('-').filter(|d| !d.is_empty() && d.chars().all(|c| c.is_ascii_digit())) {
            d.parse::<u64>().ok().filter(|&x| x <= 1u64 << 63).map(|x| (x as i64).wrapping_neg() as u64)
        } else if !tok.is_empty() && tok.chars().all(|c| c.is_ascii_digit()) {
            tok.parse::<u64>().ok()
        } else { None };
        if let Some(v) = v { found.push(v); }
    }
    if found.contains(&expect) { return Some(expect); }
    found.into_iter().find(|&v| v > 0xffff)
}

/// gpu_batch half of the EXL3 handshake. When the loaded `gpu_batch` module exposes
/// `kernel_build_id` (another loader registered it), the check LAUNCHES it (machine code actually
/// loaded — the GpuModel::assert_kernel_build_id contract). Otherwise the EXL3 qsa-subset load did
/// not register it, and the check reads the stamp from `src/ptx/gpu_batch.ptx` — the exact file that
/// load read (a CUBIN deploy's fatbin is pinned by its own hash check; this covers the PTX path).
pub fn assert_gpu_batch_build_id(dev: &std::sync::Arc<cudarc::driver::CudaDevice>) -> anyhow::Result<()> {
    if dev.get_func("gpu_batch", "kernel_build_id").is_some() {
        return crate::gpu::GpuModel::assert_kernel_build_id(dev, "gpu_batch");
    }
    let expect = u64::from_str_radix(env!("KERNEL_BUILD_ID"), 16).unwrap_or(0);
    let ptx = crate::ptx_embed::GPU_BATCH;
    match ptx_build_id(&ptx, expect) {
        Some(got) if got == expect => Ok(()),
        Some(got) => anyhow::bail!(
            "STALE KERNELS: src/ptx/gpu_batch.ptx was built from different kernel sources than this binary \
             (ptx={got:016x}, binary={expect:016x}). A deploy is the binary + src/ptx/*.ptx from ONE build. \
             Run `cargo build --release && ./scripts/build_stable.sh`."),
        None => anyhow::bail!(
            "STALE KERNELS: src/ptx/gpu_batch.ptx has no kernel_build_id entry — it predates the build-ID \
             stamp and cannot be verified against this binary. Run `cargo build --release`."),
    }
}
