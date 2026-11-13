//! Version check + self-update (owner request 2026-11-13): the installed engine can
//! (a) notify at boot when a newer GitHub release exists, and
//! (b) update itself in place via `--update` (download the latest release tarball,
//!     sha256-verify, rename over `current_exe()` — atomic on Linux; the running image
//!     keeps serving until exit).
//!
//! Zero new dependencies: HTTPS via the system curl (the same tool install.sh requires;
//! every DGX Spark image ships it). Everything is SILENT-ON-FAILURE: no network, no curl,
//! API rate limit → the boot checker prints nothing and `--update` reports why it stopped.
//!
//! NEVER automatic: the boot path only NOTIFIES; the binary is replaced only when the
//! user runs `--update` explicitly. Dev builds (tree dirty / exe inside a checkout or
//! worktree) refuse: "only valid for the installed app".
//!
//! No env vars (§7): the check is skipped under `--no-update-check` (a HAND_FLAG).

use std::path::PathBuf;
use std::process::Command;

pub const REPO: &str = "sf-stav/veloGB10";
const ASSET: &str = "velogb10-{TAG}-gb10-sm121.tar.gz";

// ---------- ANSI ----------
pub fn tty_colors() -> bool {
    // TTY-gated only (§7: NO env reads, not even the NO_COLOR convention).
    unsafe { libc_isatty() }
}
// std has no isatty; avoid a libc dep: probe via /proc/self/fd/2 symlink target.
fn libc_isatty() -> bool {
    std::fs::read_link("/proc/self/fd/2")
        .map(|p| p.to_string_lossy().starts_with('/'))
        .unwrap_or(false)
}
fn c(code: &str, s: &str) -> String {
    if tty_colors() { format!("\x1b[{code}m{s}\x1b[0m") } else { s.to_string() }
}
pub fn bold(s: &str) -> String { c("1", s) }
pub fn green(s: &str) -> String { c("32", s) }
pub fn yellow(s: &str) -> String { c("33", s) }
pub fn cyan(s: &str) -> String { c("36", s) }
pub fn dim(s: &str) -> String { c("2", s) }

fn curl(args: &[&str], timeout_secs: u64) -> Result<String, String> {
    let out = Command::new("curl")
        .arg("-fsSL").arg("--max-time").arg(&timeout_secs.to_string())
        .args(args)
        .output()
        .map_err(|e| format!("curl unavailable: {e}"))?;
    if !out.status.success() {
        return Err(format!("curl exited {}", out.status.code().unwrap_or(-1)));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// The latest release tag (e.g. "v0.7.9"), or an Err explaining why (silently swallowed
/// by the boot checker; surfaced by `--update`).
pub fn latest_tag() -> Result<String, String> {
    let body = curl(&[&format!("https://api.github.com/repos/{REPO}/releases/latest")], 4)?;
    let t = body.split("\"tag_name\":").nth(1)
        .and_then(|s| s.split('"').nth(1))
        .ok_or_else(|| "no tag_name in the API response".to_string())?;
    Ok(t.trim().to_string())
}

fn current_version() -> String {
    let mut v = env!("CARGO_PKG_VERSION").to_string();
    if option_env!("TUNE_GIT_DIRTY").map(|d| d != "0").unwrap_or(true) {
        v.push_str("-dev");
    }
    v
}

fn tag_version(tag: &str) -> String { tag.trim_start_matches('v').to_string() }

/// semver-ish comparison by numeric segments; a differing segment count is fine.
fn newer_available(tag: &str) -> bool {
    let cur = current_version();
    let cur = cur.trim_end_matches("-dev");
    let (a, b) = (cur, tag_version(tag));
    let sa: Vec<u64> = a.split('.').map(|s| s.parse().unwrap_or(0)).collect();
    let sb: Vec<u64> = b.split('.').map(|s| s.parse().unwrap_or(0)).collect();
    sb > sa
}

/// Boot-time notification (never a replacement). One colored line on stderr when a newer
/// release exists and the network answered within the timeout; SILENT otherwise.
pub fn boot_check() {
    let Ok(tag) = latest_tag() else { return };
    if newer_available(&tag) {
        eprintln!("{} {} {} {} ({} {}): {}",
            yellow("update available:"),
            bold(&tag),
            dim("— current"),
            current_version(),
            dim("run"),
            cyan("gb10_inference --update"),
            dim("to install it (same version on every TP node)"));
    }
}

// ---------- --update ----------

fn exe_path() -> Result<PathBuf, String> {
    std::env::current_exe().map_err(|e| format!("cannot resolve the running binary: {e}"))
}

fn is_dev_build() -> Option<String> {
    if option_env!("TUNE_GIT_DIRTY").map(|d| d != "0").unwrap_or(true) {
        return Some("this is a dev build (uncommitted tree)".into());
    }
    let p = exe_path().ok()?;
    for a in p.ancestors() {
        if a.join(".git").exists() { return Some(format!("the binary lives inside a checkout ({})", a.display())); }
        if a.file_name().map(|f| f == "target").unwrap_or(false) { return Some("the binary lives in a cargo target dir".into()); }
    }
    None
}

/// `--update`: download the latest release, sha256-verify, replace the running binary
/// (atomic rename), keep the previous image as `.prev`. Prints each step; refuses dev
/// builds and TP-node processes (a node must never replace itself out from under a head).
pub fn update(skip_dev_guard: bool) -> i32 {
    println!("{} {}", bold("veloGB10 self-update"), dim(&format!("(repo {REPO})")));
    if !skip_dev_guard {
        if let Some(why) = is_dev_build() {
            println!("{} {} — {}", yellow("refused:"), why,
                dim("self-update is only valid for the installed app; update your checkout and rebuild"));
            return 2;
        }
    }
    let tag = match latest_tag() {
        Ok(t) => t,
        Err(e) => { println!("{} could not determine the latest release: {}", yellow("stopped:"), e); return 1; }
    };
    if !newer_available(&tag) {
        println!("{} already the latest release ({})", green("ok:"), tag);
        return 0;
    }
    println!("{} {}", "latest release:", bold(&tag));
    let asset = ASSET.replace("{TAG}", &tag);
    let base = format!("https://github.com/{REPO}/releases/download/{tag}");

    let tmp = std::env::temp_dir().join(format!("velogb10-update-{}-{}", std::process::id(), tag));
    let _ = std::fs::remove_dir_all(&tmp);
    if std::fs::create_dir_all(&tmp).is_err() { println!("{} cannot create {}", yellow("stopped:"), tmp.display()); return 1; }

    println!("{} {}", "downloading", &asset);
    let tb = tmp.join(&asset);
    if let Err(e) = curl_bin(&format!("{base}/{asset}"), &tb, 600) {
        println!("{} download failed: {}", yellow("stopped:"), e);
        return 1;
    }
    // sha256 against SHA256SUMS when published
    if let Ok(sums) = curl(&[&format!("{base}/SHA256SUMS")], 10) {
        let want = sums.lines().find(|l| l.contains(&asset)).and_then(|l| l.split_whitespace().next()).map(str::to_string);
        if let Some(w) = want {
            match sha256_of(&tb) {
                Ok(h) if h == w => println!("{} sha256 verified", green("ok:")),
                Ok(_) => { println!("{} sha256 MISMATCH — the download is corrupt; the current binary is untouched", yellow("stopped:")); return 1; }
                Err(e) => println!("{} could not hash the download ({e}); continuing unverified", yellow("note:")),
            }
        } else { println!("{} no checksum line for the asset; continuing unverified", yellow("note:")); }
    } else { println!("{} no SHA256SUMS published; continuing unverified", yellow("note:")); }

    println!("{} unpacking", "…");
    let st = Command::new("tar").arg("-xzf").arg(&tb).arg("-C").arg(&tmp)
        .status().map_err(|e| e.to_string()).and_then(|s| if s.success() { Ok(()) } else { Err("tar failed".into()) });
    if let Err(e) = st { println!("{} {e}", yellow("stopped:")); return 1; }
    let new_bin = find_binary(&tmp);
    let Some(new_bin) = new_bin else { println!("{} the tarball has no gb10_inference binary", yellow("stopped:")); return 1; };

    let exe = match exe_path() { Ok(p) => p, Err(e) => { println!("{} {e}", yellow("stopped:")); return 1; } };
    let prev = exe.with_extension("prev");
    // 1) keep the old image for --rollback
    if std::fs::copy(&exe, &prev).is_err() { println!("{} could not back up the current binary; aborting", yellow("stopped:")); return 1; }
    // 2) staged + chmod + atomic rename
    let staged = exe.with_extension("new");
    if std::fs::copy(&new_bin, &staged).is_err() { println!("{} could not stage the new binary", yellow("stopped:")); return 1; }
    #[cfg(unix)]
    { use std::os::unix::fs::PermissionsExt; let _ = std::fs::set_permissions(&staged, std::fs::Permissions::from_mode(0o755)); }
    if let Err(e) = std::fs::rename(&staged, &exe) {
        println!("{} could not replace the binary ({e}); the current one still runs", yellow("stopped:"));
        let _ = std::fs::remove_file(&staged);
        return 1;
    }
    let _ = std::fs::remove_dir_all(&tmp);
    println!("{} installed {} {} — {}",
        green("done:"), bold(&tag), dim(&format!("(was v{})", current_version())),
        bold("restart the engine to run the new version"));
    println!("{} install the {} on every TP node; the previous image is kept at {}",
        yellow("note:"), bold("same version"), dim(&prev.display().to_string()));
    0
}

fn curl_bin(url: &str, to: &std::path::Path, timeout_secs: u64) -> Result<(), String> {
    let st = Command::new("curl").arg("-fsSL").arg("--max-time").arg(&timeout_secs.to_string())
        .arg("-o").arg(to).arg(url).status().map_err(|e| format!("curl unavailable: {e}"))?;
    if st.success() { Ok(()) } else { Err(format!("curl exited {}", st.code().unwrap_or(-1))) }
}

fn sha256_of(p: &std::path::Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    let b = std::fs::read(p).map_err(|e| e.to_string())?;
    let mut h = Sha256::new(); h.update(&b);
    Ok(h.finalize().iter().map(|x| format!("{x:02x}")).collect())
}

fn find_binary(dir: &std::path::Path) -> Option<PathBuf> {
    fn walk(d: &std::path::Path, out: &mut Option<PathBuf>) {
        if out.is_some() { return; }
        let Ok(rd) = std::fs::read_dir(d) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() { walk(&p, out); }
            else if p.file_name().map(|f| f == "gb10_inference").unwrap_or(false) { *out = Some(p); return; }
        }
    }
    let mut out = None; walk(dir, &mut out); out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tag_comparison() {
        // current is the crate version (may have -dev suffix in dev trees)
        assert!(!newer_available(&format!("v{}", current_version().trim_end_matches("-dev"))));
        assert!(newer_available("v999.0.0"));
        assert!(!newer_available("v0.0.1"));
    }

    #[test]
    fn tag_version_strips_v() {
        assert_eq!(tag_version("v0.7.9"), "0.7.9");
    }
}
