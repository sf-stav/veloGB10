//! SL: engagement receipts of the per-layer forward (AGENTS §4: identical hashes are not a liveness
//! receipt). Every call the forward makes is counted by (path, backend, kernel entry, row bucket, served
//! class); the forward prints ONE block per process naming which entries served each class and width
//! (`boot_lines`), and `/metrics` renders the counters as `velogb10_dsv41_fwd_calls_total` (nothing at all
//! unless a dsv41 forward ran in this process, so every other family's `/metrics` body stays byte-identical).

use crate::dsv41::weights::{BackendKind, RowClass};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

type Key = (&'static str, &'static str, &'static str, &'static str, &'static str);
static CALLS: Mutex<BTreeMap<Key, u64>> = Mutex::new(BTreeMap::new());
static PRINTED: AtomicBool = AtomicBool::new(false);

/// The width bucket of a call (the verify-width classes of the contract, with 17-32 split out: the
/// serving verify widths of 3-4 lanes live there).
pub fn bucket(rows: usize) -> &'static str {
    match rows { 0 => "0", 1 => "1", 2 => "2", 3..=16 => "3-16", 17..=32 => "17-32", _ => ">32" }
}

/// A call through a weight backend (the class it returned).
pub fn count(path: &'static str, b: BackendKind, entry: &'static str, rows: usize, cls: RowClass) {
    if let Ok(mut m) = CALLS.lock() { *m.entry((path, b.name(), entry, bucket(rows), cls.name())).or_insert(0) += 1; }
}
/// A weight-free kernel (mixers, norms, caches, indexer, attention): no backend, no row class.
pub fn kernel(path: &'static str, entry: &'static str, rows: usize) {
    if let Ok(mut m) = CALLS.lock() { *m.entry((path, "-", entry, bucket(rows), "-")).or_insert(0) += 1; }
}

/// The receipts so far: one line per (path, backend, entry, rows, class).
pub fn lines() -> Vec<String> {
    let m = match CALLS.lock() { Ok(m) => m.clone(), Err(_) => return vec![] };
    m.iter().map(|((p, b, e, r, c), n)| format!("[dsv41 fwd] {p:<18} rows {r:>5}: {b:<6} {e} (class {c}) x{n}")).collect()
}

/// The once-per-process block (empty after the first call).
pub fn boot_lines() -> Vec<String> {
    if PRINTED.swap(true, Ordering::Relaxed) { return vec![]; }
    let mut v = vec!["[dsv41 fwd] kernel entries that served each path and width (once per process):".to_string()];
    v.extend(lines());
    v
}

/// `/metrics`: velogb10_dsv41_fwd_calls_total{path,backend,entry,rows,class}.
pub fn render(o: &mut String) {
    let m = match CALLS.lock() { Ok(m) => m.clone(), Err(_) => return };
    if m.is_empty() { return; }
    o.push_str("# HELP velogb10_dsv41_fwd_calls_total dsv41 per-layer forward calls by path, backend, kernel entry, row bucket and served class\n");
    o.push_str("# TYPE velogb10_dsv41_fwd_calls_total counter\n");
    for ((p, b, e, r, c), n) in m {
        o.push_str(&format!("velogb10_dsv41_fwd_calls_total{{path=\"{p}\",backend=\"{b}\",entry=\"{e}\",rows=\"{r}\",class=\"{c}\"}} {n}\n"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dsv41_sl_receipts_count_and_render() {
        count("test.path", BackendKind::Exl3, "d41x_test", 6, RowClass::Coop);
        count("test.path", BackendKind::Exl3, "d41x_test", 6, RowClass::Coop);
        let mut o = String::new();
        render(&mut o);
        assert!(o.contains("velogb10_dsv41_fwd_calls_total{path=\"test.path\",backend=\"exl3\",entry=\"d41x_test\",rows=\"3-16\",class=\"coop\"} 2"), "{o}");
        assert!(lines().iter().any(|l| l.contains("test.path") && l.contains("x2")));
        assert_eq!(bucket(24), "17-32");
    }
}
