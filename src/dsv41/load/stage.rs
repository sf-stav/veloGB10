//! DSV4.1 loader (leg LS): the FAST-REBOOT LOCAL STAGE (UX row 15, the `--naive-load-cache` pattern).
//!
//! A rank's shard is a set of strided gathers out of the pack's shards (an expert's gate / up trellis cut
//! along its output is one 144-byte-ish run per tile row). Read cold off the pack that is slow and it is
//! repeated at every boot; a NODE never pays it twice (it holds the shipped blobs), but the HEAD's own rank 0
//! would. The stage materialises the head's shard dir once, under the tp cache root, as plain contiguous
//! files in exactly the layout a node's assembled dir has, and every later boot reads that:
//!
//! * `<tp-cache>/dsv41-load/<source>-w<W>-r<R>-<key>/` — `<key>` = the sha256 of every planned file's recipe
//!   (the source files' path / mtime / size, every piece), so a changed pack, plan version, split flag or
//!   world picks another dir and the old one is simply unused;
//! * built in `<dir>.part`, published by rename, with `.complete` (name + size of every file) written last: an
//!   interrupted build leaves a `.part` the next boot resumes (finished files are kept);
//! * a warm boot checks the sizes in `.complete` and nothing else (a read of ~100 GB to re-hash it would cost
//!   what the stage saves; `--dsv41-load-cache on` rebuilds the dir from scratch);
//! * never a refusal: no room, a write error or `--dsv41-load-cache off` -> `Outcome::Skipped` with one line and
//!   the caller loads straight from the pack through the same plan.

use crate::boot_report::{Progress, Unit};
use crate::shard_plan::{segment_recipe, write_segment, PlanFile, PlanSrc, RankPlan};
use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome { Warm, Cold, Skipped }

#[derive(Clone, Debug)]
pub struct Staged { pub dir: Option<PathBuf>, pub outcome: Outcome, pub bytes: u64, pub why: Option<String> }

/// `--dsv41-load-cache auto|on|off`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode { Auto, On, Off }

impl Mode {
    pub fn parse(s: Option<&str>) -> Mode { match s { Some("off") => Mode::Off, Some("on") => Mode::On, _ => Mode::Auto } }
}

const MARK: &str = ".complete";
/// Room kept free on the drive after the stage.
const MARGIN: u64 = 8 << 30;

/// The stage key of a plan: every planned file's name and content recipe.
pub fn key_of(files: &[PlanFile]) -> Result<String> {
    let mut h = Sha256::new();
    for f in files {
        h.update(f.logical.as_bytes());
        match &f.src {
            PlanSrc::Segment(ps) => h.update(segment_recipe(ps)?.as_bytes()),
            PlanSrc::Whole(p) => { let md = std::fs::metadata(p).with_context(|| p.display().to_string())?; h.update(format!("whole {} {}", p.display(), md.len()).as_bytes()) }
            PlanSrc::Bytes(b) => h.update(b),
            PlanSrc::Generated(g) => h.update(g.key().as_bytes()),
        }
    }
    Ok(h.finalize().iter().take(8).map(|x| format!("{x:02x}")).collect())
}

fn root() -> PathBuf {
    crate::opts::lookup("tp-cache").and_then(crate::opts::get).map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").unwrap_or_else(|| "/tmp".into())).join(".cache/gb10_tp")).join("dsv41-load")
}

fn free_bytes(p: &Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let mut q = p.to_path_buf();
    while !q.exists() { if !q.pop() { return None; } }
    let c = std::ffi::CString::new(q.as_os_str().as_bytes()).ok()?;
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c.as_ptr(), &mut st) } != 0 { return None; }
    Some(st.f_bavail as u64 * st.f_frsize as u64)
}

struct Counted<'a, W: Write> { inner: W, prog: &'a Progress }
impl<W: Write> Write for Counted<'_, W> {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> { let n = self.inner.write(b)?; self.prog.add(n as u64); Ok(n) }
    fn flush(&mut self) -> std::io::Result<()> { self.inner.flush() }
}

fn marker_ok(dir: &Path, files: &[PlanFile]) -> bool {
    let Ok(txt) = std::fs::read_to_string(dir.join(MARK)) else { return false };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) else { return false };
    let Some(m) = v.as_object() else { return false };
    m.len() == files.len() && files.iter().all(|f| {
        let want = m.get(&f.logical).and_then(|x| x.as_u64());
        want.is_some() && std::fs::metadata(dir.join(&f.logical)).map(|md| md.len()).ok() == want
    })
}

/// Stage `files` (the head's own rank) for `source`/`world`/`rank`; `mode` from the flag.
pub fn stage(root: &Path, files: &[PlanFile], source: &str, world: usize, rank: usize, mode: Mode, rank_label: (usize, usize)) -> Result<Staged> {
    if mode == Mode::Off { return Ok(Staged { dir: None, outcome: Outcome::Skipped, bytes: 0, why: Some("--dsv41-load-cache off".into()) }); }
    let key = key_of(files)?;
    let dir = root.join(format!("{source}-w{world}-r{rank}-{key}"));
    let total: u64 = files.iter().map(|f| RankPlan::file_size(f)).sum::<Result<u64>>()?;
    if mode != Mode::On && marker_ok(&dir, files) {
        return Ok(Staged { dir: Some(dir), outcome: Outcome::Warm, bytes: total, why: None });
    }
    let part = PathBuf::from(format!("{}.part", dir.display()));
    if mode == Mode::On { let _ = std::fs::remove_dir_all(&dir); let _ = std::fs::remove_dir_all(&part); }
    // room: what is still to write + margin (a resumed .part already holds some)
    let have: u64 = files.iter().filter_map(|f| std::fs::metadata(part.join(&f.logical)).ok().map(|m| m.len())).sum();
    let need = total.saturating_sub(have) + MARGIN;
    if let Some(free) = free_bytes(root) {
        if free < need {
            return Ok(Staged { dir: None, outcome: Outcome::Skipped, bytes: 0, why: Some(format!(
                "no room for the local stage: {:.1} GB to write + {:.0} GB kept free, {:.1} GB free under {} (loading straight from the pack; --tp-cache <a bigger drive> to stage)",
                (total - have) as f64 / 1e9, MARGIN as f64 / 1e9, free as f64 / 1e9, root.display())) });
        }
    }
    let prog = super::progress(super::Phase::Stage, Some(rank_label), total);
    prog.add(have);
    let mut sizes = serde_json::Map::new();
    for f in files {
        let dst = part.join(&f.logical);
        let size = RankPlan::file_size(f)?;
        sizes.insert(f.logical.clone(), size.into());
        if std::fs::metadata(&dst).map(|m| m.len()).ok() == Some(size) { continue; } // finished before an interruption
        std::fs::create_dir_all(dst.parent().unwrap())?;
        let tmp = PathBuf::from(format!("{}.tmp", dst.display()));
        {
            let out = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(&tmp).with_context(|| tmp.display().to_string())?);
            let mut w = Counted { inner: out, prog: &prog };
            match &f.src {
                PlanSrc::Segment(ps) => write_segment(ps, &mut w)?,
                PlanSrc::Whole(p) => { let mut r = std::fs::File::open(p)?; std::io::copy(&mut r, &mut w)?; }
                PlanSrc::Bytes(b) => w.write_all(b)?,
                PlanSrc::Generated(g) => g.write_to(&mut w)?,
            }
            w.flush()?;
        }
        std::fs::rename(&tmp, &dst)?;
    }
    std::fs::write(part.join(MARK), serde_json::to_vec_pretty(&serde_json::Value::Object(sizes))?)?;
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::rename(&part, &dir).with_context(|| format!("publish {}", dir.display()))?;
    prog.finish();
    Ok(Staged { dir: Some(dir), outcome: Outcome::Cold, bytes: total, why: None })
}

/// `stage` that never fails a boot: an error is one line and `Skipped`.
pub fn stage_or_skip(files: &[PlanFile], source: &str, world: usize, rank: usize, mode: Mode, rank_label: (usize, usize)) -> Staged {
    let s = stage(&root(), files, source, world, rank, mode, rank_label).unwrap_or_else(|e| Staged { dir: None, outcome: Outcome::Skipped, bytes: 0, why: Some(format!("the local stage failed ({e:#}); loading straight from the pack")) });
    super::metrics_rows::stage(s.outcome);
    if let (Outcome::Skipped, Some(w)) = (s.outcome, &s.why) { if mode != Mode::Off { crate::boot_report::notice(&format!("fast-reboot stage skipped: {w}")); } }
    s
}

#[allow(dead_code)]
fn _unit() -> Unit { Unit::Bytes }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shard_plan::Piece;

    fn fixture(tag: &str) -> (PathBuf, Vec<PlanFile>) {
        let d = std::env::temp_dir().join(format!("ls_stage_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let src = d.join("src.bin");
        std::fs::write(&src, (0..4096u32).map(|i| (i % 251) as u8).collect::<Vec<u8>>()).unwrap();
        let seg = PlanSrc::Segment(vec![
            Piece { name: "a".into(), dtype: "U8".into(), shape: vec![64], src: src.clone(), off: 100, len: 64, rows: 1, stride: 0 },
            Piece { name: "b".into(), dtype: "U8".into(), shape: vec![4, 8], src: src.clone(), off: 200, len: 32, rows: 4, stride: 16 },
        ]);
        (d, vec![PlanFile { logical: "seg/L00.safetensors".into(), src: seg }, PlanFile { logical: "x.json".into(), src: PlanSrc::Bytes(b"{}".to_vec()) }])
    }

    #[test]
    fn a_stage_builds_cold_then_reads_warm_and_the_key_follows_the_recipe() {
        let (d, files) = fixture("a");
        // the tp-cache root: a unique dir via the real option store is process-global, so test `stage` against its own root
        let k1 = key_of(&files).unwrap();
        assert_eq!(k1, key_of(&files).unwrap());
        let mut f2 = files.clone();
        if let PlanSrc::Segment(ps) = &mut f2[0].src { ps[0].off += 1; }
        assert_ne!(k1, key_of(&f2).unwrap(), "a moved piece is another stage");
        let root = d.join("root");
        let a = stage(&root, &files, "m", 2, 0, Mode::Auto, (0, 2)).unwrap();
        assert_eq!(a.outcome, Outcome::Cold);
        let dir = a.dir.clone().unwrap();
        let b = stage(&root, &files, "m", 2, 0, Mode::Auto, (0, 2)).unwrap();
        assert_eq!((b.outcome, b.dir.as_ref()), (Outcome::Warm, Some(&dir)));
        assert!(marker_ok(&dir, &files));
        assert_eq!(stage(&root, &files, "m", 2, 0, Mode::Off, (0, 2)).unwrap().outcome, Outcome::Skipped);
        // the staged segment is byte-identical to what a shipped one would be
        let mut want = Vec::new();
        if let PlanSrc::Segment(ps) = &files[0].src { write_segment(ps, &mut want).unwrap(); }
        assert_eq!(std::fs::read(dir.join("seg/L00.safetensors")).unwrap(), want);
        // a truncated file voids the marker
        let f = std::fs::OpenOptions::new().write(true).open(dir.join("x.json")).unwrap();
        f.set_len(1).unwrap();
        assert!(!marker_ok(&dir, &files));
        // ... so the next boot rebuilds it
        assert_eq!(stage(&root, &files, "m", 2, 0, Mode::Auto, (0, 2)).unwrap().outcome, Outcome::Cold);
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn modes_parse() {
        assert_eq!(Mode::parse(None), Mode::Auto);
        assert_eq!(Mode::parse(Some("off")), Mode::Off);
        assert_eq!(Mode::parse(Some("on")), Mode::On);
    }
}
