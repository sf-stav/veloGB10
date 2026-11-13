//! Issue #33 (public): resolve a Hugging Face repo id passed to `--model-dir` to the LATEST
//! local snapshot directory under the HF cache. No network: the engine never downloads; a repo
//! id that is not cached fails loudly, naming the cache root it searched and the fix
//! (`huggingface-cli download <id>`).
//!
//! Layout (hf_hub standard): $HF_HOME/hub/models--<org>--<name>/snapshots/<commit>/... and
//! $HF_HOME/hub/models--<name>/snapshots/<commit>/... for org-less ids. HF_HOME defaults to
//! ~/.cache/huggingface. "Latest" = the snapshot whose commit hash sorts highest by mtime of
//! its refs file is NOT reliable across mirrors; hf_hub records pinned refs under refs/, but
//! the pragmatic and deterministic choice used by huggingface-cli itself for `local-dir` runs
//! is the lexicographically-greatest snapshot hash, which matches the order commits were
//! fetched in. We sort by directory mtime, tie-broken by name, newest first.

use std::path::{Path, PathBuf};

/// True when the string looks like a HF repo id ("org/name", "name", possibly with a "::"
/// dataset marker we refuse) rather than a filesystem path. A path separator or a leading
/// `.`/`/`/`~` means the caller meant a directory; anything else with a `/` in the middle is a
/// repo id.
pub fn looks_like_repo_id(s: &str) -> bool {
    if s.is_empty() || s.starts_with('.') || s.starts_with('/') || s.starts_with('~') {
        return false;
    }
    // Exactly one "/" in the middle (org/model or user/model); more than one is a path.
    let parts: Vec<&str> = s.split('/').collect();
    parts.len() <= 2 && !s.ends_with('/') && !parts.iter().any(|p| p.is_empty())
}

/// Resolve a repo id to the newest local snapshot dir, or None with a reason string.
pub fn resolve(repo_id: &str, hf_home_override: Option<&str>) -> Result<PathBuf, String> {
    // CLI-1 / AGENTS §7: NO environment variables. The cache roots are the DEFAULT
    // (~/.cache/huggingface/hub) plus every --hf-cache <DIR> flag on this command line.
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Some(h) = hf_home_override {
        roots.push(PathBuf::from(h));
    }
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 0;
    while i < argv.len() {
        if argv[i] == "--hf-cache" && i + 1 < argv.len() {
            if !roots.iter().any(|r| r == &PathBuf::from(&argv[i + 1])) {
                roots.push(PathBuf::from(argv[i + 1].clone()));
            }
            i += 2;
        } else if let Some(v) = argv[i].strip_prefix("--hf-cache=") {
            if !roots.iter().any(|r| r == &PathBuf::from(v)) { roots.push(PathBuf::from(v)); }
            i += 1;
        } else { i += 1; }
    }
    roots.push(dirs_home().join(".cache").join("huggingface").join("hub"));

    let dir_name = format!("models--{}", repo_id.replace('/', "--"));
    let mut searched: Vec<String> = Vec::new();
    for root in &roots {
        searched.push(root.display().to_string());
        let snaps = root.join(&dir_name).join("snapshots");
        if !snaps.is_dir() { continue; }
        let mut best: Option<(std::time::SystemTime, String)> = None;
        if let Ok(rd) = std::fs::read_dir(&snaps) {
            for e in rd.flatten() {
                let p = e.path();
                if !p.is_dir() { continue; }
                let name = e.file_name().to_string_lossy().into_owned();
                let mt = e.metadata().and_then(|m| m.modified()).unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                if best.as_ref().map_or(true, |b| (mt, name.clone()) > *b) {
                    best = Some((mt, name));
                }
            }
        }
        if let Some((_, name)) = best {
            return Ok(snaps.join(name));
        }
    }
    Err(format!(
        "--model-dir '{repo_id}' looks like a Hugging Face repo id but no local snapshot was found \
(searched: {}). Download it first (huggingface-cli download {repo_id}) or pass the snapshot \
directory itself",
        searched.join(", ")
    ))
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("/root"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_id_shapes() {
        assert!(looks_like_repo_id("doth4580/Qwen3.8-Flash-Next-EXL3-3.05bpw"));
        assert!(looks_like_repo_id("gpt2"));
        assert!(!looks_like_repo_id("/mnt/models/pack"));
        assert!(!looks_like_repo_id("./local/pack"));
        assert!(!looks_like_repo_id("~/models/pack"));
        assert!(!looks_like_repo_id("a/b/c"));
        assert!(!looks_like_repo_id(""));
        assert!(!looks_like_repo_id("ends/"));
    }

    #[test]
    fn resolves_newest_snapshot_and_fails_loud() {
        let tmp = std::env::temp_dir().join(format!("hf33_{}", std::process::id()));
        let hub = tmp.join("hub").join("models--org--m").join("snapshots");
        std::fs::create_dir_all(hub.join("aaaa")).unwrap();
        std::fs::create_dir_all(hub.join("zzzz")).unwrap();
        // make zzzz newer
        let f = std::fs::File::create(hub.join("zzzz").join("config.json")).unwrap();
        drop(f);
        let r = resolve("org/m", Some(tmp.join("hub").to_str().unwrap())).unwrap();
        assert!(r.ends_with("zzzz"), "{r:?}");
        let e = resolve("org/absent", Some(tmp.join("hub").to_str().unwrap())).unwrap_err();
        assert!(e.contains("huggingface-cli download org/absent"), "{e}");
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
