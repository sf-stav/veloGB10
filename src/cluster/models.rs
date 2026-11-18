//! Blob cache management (ops CLI): list / remove cached models and blobs. See `mod.rs` for the map.

use super::*;

// ---------------------------------------------------------------------------------------------------
// Blob cache management (ops CLI: --cached-models-list / --cached-models-remove /
// --cached-models-remove-all — MODEL-centric; the old blob-centric names are deprecated aliases)
// ---------------------------------------------------------------------------------------------------

/// Walk every symlink under `cache/models/<model_id>/`, calling `f(model_id, link_path, target)`.
fn walk_model_links(mut f: impl FnMut(&str, &Path, &Path)) {
    let mdir = cache_root().join("models");
    let Ok(models) = std::fs::read_dir(&mdir) else { return };
    for e in models.flatten() {
        let mid = e.file_name().to_string_lossy().to_string();
        let mut stack = vec![e.path()];
        while let Some(d) = stack.pop() {
            let Ok(rd) = std::fs::read_dir(&d) else { continue };
            for ent in rd.flatten() {
                let p = ent.path();
                let Ok(ft) = ent.file_type() else { continue };
                if ft.is_dir() { stack.push(p); continue; }
                if ft.is_symlink() {
                    if let Ok(t) = std::fs::read_link(&p) { f(&mid, &p, &t); }
                }
            }
        }
    }
}

fn fmt_gib(b: u64) -> String { format!("{:.2} GiB", b as f64 / (1u64 << 30) as f64) }

/// The blobs (by hash) referenced by the assembled model dir `model_id` (empty if absent).
pub(super) fn model_blob_hashes(model_id: &str) -> Vec<String> {
    let mut v = Vec::new();
    walk_model_links(|mid, _p, t| {
        if mid == model_id {
            if let Some(h) = t.file_name() { v.push(h.to_string_lossy().to_string()); }
        }
    });
    v.sort(); v.dedup();
    v
}

/// Blob file size in `blobs/<hash>` (0 if missing).
fn blob_size(hash: &str) -> u64 {
    std::fs::metadata(blob_path(hash)).map(|m| m.len()).unwrap_or(0)
}

/// `--cached-models-list`: ONE line per assembled MODEL — name, total size (sum of its blob
/// files), blob count — plus a cache summary and any orphan-blob / interrupted-fetch tail.
pub fn list_cached_models() -> Result<()> {
    let mdir = cache_root().join("models");
    let mut names: Vec<String> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&mdir) {
        for e in rd.flatten() {
            let mid = e.file_name().to_string_lossy().to_string();
            if e.file_type().map(|ft| ft.is_dir()).unwrap_or(false) { names.push(mid); }
        }
    }
    names.sort();
    // B32: per model — its total (every blob it links), its blob count, and how much of it is SHARED with another
    // model (those blobs stay when this model is removed).
    let per_model: Vec<(String, Vec<String>)> = names.iter().map(|m| (m.clone(), model_blob_hashes(m))).collect();
    let mut refcount: HashMap<String, usize> = HashMap::new();
    for (_, hs) in &per_model { for h in hs { *refcount.entry(h.clone()).or_default() += 1; } }
    let mut rows: Vec<(String, u64, usize, usize, u64)> = Vec::new();
    let mut all_blobs: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (mid, hashes) in &per_model {
        let total: u64 = hashes.iter().map(|h| blob_size(h)).sum();
        let shared: Vec<&String> = hashes.iter().filter(|h| refcount[*h] > 1).collect();
        let shared_b: u64 = shared.iter().map(|h| blob_size(h)).sum();
        all_blobs.extend(hashes.iter().cloned());
        rows.push((mid.clone(), total, hashes.len(), shared.len(), shared_b));
    }
    let blob_dir = cache_root().join("blobs");
    let mut blob_total = 0u64;
    let mut n_blobs = 0usize;
    let mut tmp: Vec<(String, u64)> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&blob_dir) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            let size = e.metadata().map(|m| m.len()).unwrap_or(0);
            if name.starts_with("tmp.") { tmp.push((name, size)); }
            else { blob_total += size; n_blobs += 1; }
        }
    }
    let orphans: Vec<String> = std::fs::read_dir(&blob_dir).into_iter().flatten()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| !n.starts_with("tmp.") && !all_blobs.contains(n))
        .collect();
    println!("cached models ({}):", rows.len());
    for (mid, total, n, ns, sb) in &rows {
        if *n == 0 {
            println!("  {mid}  (empty: nothing cached — a placeholder left by the legacy EXL3 sync; --cached-models-remove clears it)");
        } else if *ns > 0 {
            println!("  {mid}  {:>12}  {n} blob(s)  ({ns} blob(s), {} shared with other models)", fmt_gib(*total), fmt_gib(*sb));
        } else {
            println!("  {mid}  {:>12}  {n} blob(s)", fmt_gib(*total));
        }
    }
    println!("cache {} — {} blob(s), {} total", blob_dir.display(), n_blobs, fmt_gib(blob_total));
    if !orphans.is_empty() {
        let osz: u64 = orphans.iter().map(|h| blob_size(h)).sum();
        println!("-- {} orphan blob(s), {} (referenced by no model; --cached-models-remove-all reclaims)", orphans.len(), fmt_gib(osz));
    }
    if !tmp.is_empty() {
        let t: u64 = tmp.iter().map(|x| x.1).sum();
        println!("-- {} interrupted-fetch partial(s), {} reclaimable (tmp.*)", tmp.len(), fmt_gib(t));
    }
    Ok(())
}

/// Match a model id by exact name or unique prefix (a 4-char floor, like the old blob op).
fn resolve_model(id: &str) -> Result<String> {
    let mdir = cache_root().join("models");
    let mut matches: Vec<String> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&mdir) {
        for e in rd.flatten() {
            let mid = e.file_name().to_string_lossy().to_string();
            if e.file_type().map(|ft| ft.is_dir()).unwrap_or(false)
                && (mid == id || mid.starts_with(id)) { matches.push(mid); }
        }
    }
    match matches.len() {
        1 => Ok(matches.pop().unwrap()),
        0 => bail!("no cached model matching '{id}' in {}", mdir.display()),
        n => bail!("'{id}' matches {n} models — give a longer prefix"),
    }
}

/// `--cached-models-remove <id>`: remove ONE cached MODEL — its assembled dir plus the blobs
/// referenced by NO other model (shared blobs stay). Interrupted-fetch partials are never
/// referenced and are reclaimed here too. Other models are untouched.
pub fn remove_cached_model(id: &str) -> Result<()> {
    if id.len() < 4 { bail!("refusing to match an id shorter than 4 characters"); }
    let mid = resolve_model(id)?;
    let hashes = model_blob_hashes(&mid);
    let mdir = cache_root().join("models").join(&mid);
    std::fs::remove_dir_all(&mdir).with_context(|| format!("remove model dir {mdir:?}"))?;
    // Delete THIS model's blobs that no remaining model references (B32: only its own — other models'
    // orphans and in-flight tmp.* partials of a running fetch are left to --cached-models-remove-all).
    let mut refs: std::collections::HashSet<String> = std::collections::HashSet::new();
    walk_model_links(|_m, _p, t| {
        if let Some(h) = t.file_name() { refs.insert(h.to_string_lossy().to_string()); }
    });
    let mut removed = 0u64;
    let mut n_removed = 0usize;
    for h in &hashes {
        if refs.contains(h) { continue; }
        let p = blob_path(h);
        if let Ok(md) = std::fs::metadata(&p) {
            let _ = std::fs::remove_file(&p);
            removed += md.len(); n_removed += 1;
        }
    }
    let kept = hashes.iter().filter(|h| blob_path(h).exists()).count();
    println!("removed model {mid} — {n_removed} unreferenced blob(s) reclaimed, {}",
             fmt_gib(removed));
    println!("  {kept} of its blob(s) still referenced by other models were kept");
    Ok(())
}

/// `--cached-models-remove-all`: clear the whole cache (blobs incl. tmp.* partials + assembled
/// model dirs). The next head run re-syncs from scratch.
pub fn remove_all_cached_models() -> Result<()> {
    let dir = cache_root().join("blobs");
    let mut total = 0u64;
    let mut n = 0usize;
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for e in rd.flatten() {
            total += e.metadata().map(|m| m.len()).unwrap_or(0);
            let _ = std::fs::remove_file(e.path());
            n += 1;
        }
    }
    let mdir = cache_root().join("models");
    if mdir.exists() { std::fs::remove_dir_all(&mdir).context("remove assembled model dirs")?; }
    println!("cleared {n} blob(s), {} — assembled model dirs removed; next head run re-syncs from scratch",
             fmt_gib(total));
    Ok(())
}

// -- deprecated aliases (the old blob-centric names; warn once, route to the model-centric ops) --

/// DEPRECATED alias for `--cached-models-list` (was: one line per BLOB). Warns once.
pub fn list_model_blobs() -> Result<()> {
    eprintln!("warning: --list-model-blobs is deprecated — use --cached-models-list (lists MODELS, not blobs)");
    list_cached_models()
}

/// DEPRECATED alias for `--cached-models-remove <model-id>` (was: remove one BLOB by hash).
/// The argument is now a MODEL name/prefix.
pub fn remove_model_blob(id: &str) -> Result<()> {
    eprintln!("warning: --remove-model-blob is deprecated — use --cached-models-remove <model> (removes a MODEL, not a blob)");
    remove_cached_model(id)
}

/// DEPRECATED alias for `--cached-models-remove-all`. Warns once.
pub fn clear_model_blobs() -> Result<()> {
    eprintln!("warning: --clear-model-blobs is deprecated — use --cached-models-remove-all");
    remove_all_cached_models()
}
