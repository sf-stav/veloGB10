//! DSV4.1 loader (leg LS2): an Engram head file as a STREAMED source. A rank's head file the head does not hold
//! (it is 25.3 GB; `.11` cannot keep all eight) is packed from the native checkpoint's source shards and sent to
//! the node while it is packed, never written on the head's disk. The head needs only the head's manifest
//! (a few KB: the sha256 and size of the file it would have): the manifest a previous pack or a pre-staged box left
//! in `--engram-dir` (a manifest without its `.bin` is a "known hash" record), else one pack pass into a hasher
//! (about two minutes) that is then recorded as such a manifest, so it is paid once.

use crate::dsv41::engram::manifest::{file_name, manifest_name, HeadManifest, RecordLayout, FORMAT_VERSION};
use crate::dsv41::engram::pack::{self, WriterOut};
use crate::dsv41::engram::source::{self, SourceConfig};
use crate::shard_plan::GenSource;
use anyhow::{bail, Context, Result};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// One head file, generated when sent.
pub struct GenHead {
    source: PathBuf,
    cfg: Arc<SourceConfig>,
    head: usize,
    layout: RecordLayout,
    size: u64,
    tag: String,
    hint: Option<String>,
}

impl std::fmt::Debug for GenHead {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GenHead(head {} of {}, {} B, hint {})", self.head, self.source.display(), self.size, self.hint.is_some())
    }
}

/// The manifest of head `h` found in `dir` (complete head or a manifest-only record), if it parses and fits.
pub fn recorded_manifest(dir: &Path, h: usize, expect_bytes: u64) -> Option<HeadManifest> {
    let m: HeadManifest = serde_json::from_slice(&std::fs::read(dir.join(manifest_name(h))).ok()?).ok()?;
    (m.head == h && m.version == FORMAT_VERSION && m.file_bytes == expect_bytes && m.sha256.len() == 64).then_some(m)
}

impl GenHead {
    pub fn head(&self) -> usize { self.head }
    pub fn new(source: &Path, cfg: Arc<SourceConfig>, head: usize, edir: &Path) -> Result<GenHead> {
        let layout = RecordLayout::Packed;
        let size = pack::head_bytes(&cfg.geometry, head, layout);
        let tag = super::engram::index_tag(source).unwrap_or_else(|_| "unknown".into());
        let hint = recorded_manifest(edir, head, size)
            .or_else(|| recorded_manifest(source, head, size))
            .map(|m| m.sha256);
        Ok(GenHead { source: source.to_path_buf(), cfg, head, layout, size, tag, hint })
    }

    /// The head's manifest, hashing one pack pass first when no record exists. The record is written to `edir`
    /// (a few KB; the head file itself is never written), so the next plan finds it.
    pub fn manifest(&mut self, edir: &Path) -> Result<HeadManifest> {
        let sha = match &self.hint { Some(h) => h.clone(), None => {
            crate::boot_report::notice(&format!("Engram head {} has no recorded sha256 here: packing it once into a hasher (no file is written, about 2 min; recorded in {} so later starts skip this)",
                                                self.head, edir.display()));
            let h = GenSource::hash(self)?;
            self.hint = Some(h.clone());
            h
        } };
        let m = pack::manifest_of(&self.source, &self.cfg, self.head, self.layout, self.size, sha);
        if !edir.join(manifest_name(self.head)).exists() {
            std::fs::create_dir_all(edir).with_context(|| edir.display().to_string())?;
            std::fs::write(edir.join(manifest_name(self.head)), serde_json::to_vec_pretty(&m)?)?;
        }
        Ok(m)
    }

    fn pass(&self, w: &mut dyn Write, phase: super::Phase) -> Result<String> {
        let prog = super::progress(phase, None, self.size);
        let (bytes, sha) = pack::pack_head_into(&self.source, &self.cfg, self.head, self.layout, WriterOut(w), &prog)
            .with_context(|| format!("packing Engram head {} from {}", self.head, self.source.display()))?;
        prog.finish();
        if bytes != self.size { bail!("Engram head {}: the pack is {bytes} B, expected {}", self.head, self.size); }
        Ok(sha)
    }
}

impl GenSource for GenHead {
    fn size(&self) -> u64 { self.size }
    fn key(&self) -> String { format!("gen|dsv41-engram|{}|h{}|{}|v{}", self.tag, self.head, self.layout.name(), FORMAT_VERSION) }
    fn known_hash(&self) -> Option<String> { self.hint.clone() }
    fn hash(&self) -> Result<String> { self.pass(&mut std::io::sink(), super::Phase::EngramPack) }
    fn write_to(&self, w: &mut dyn Write) -> Result<()> {
        let sha = self.pass(w, super::Phase::EngramShip)?;
        if let Some(want) = &self.hint {
            if &sha != want { bail!("Engram head {}: the streamed pack hashes to {sha}, the recorded manifest says {want} (the node would reject the blob); delete {} from the Engram dir and retry", self.head, manifest_name(self.head)); }
        }
        super::metrics_rows::engram_heads_streamed(1);
        Ok(())
    }
}

/// The native checkpoint's Engram source for `source` (headers only).
pub fn load_source(source: &Path) -> Result<Arc<SourceConfig>> { Ok(Arc::new(source::load(source)?)) }

/// Logical names of a head's two files.
pub fn names(h: usize) -> (String, String) { (format!("{}/{}", super::engram::SUBDIR, file_name(h)), format!("{}/{}", super::engram::SUBDIR, manifest_name(h))) }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn logical_names_are_the_ones_the_runtime_opens() {
        let (b, m) = names(5);
        assert_eq!(b, "engram/engram_head5.bin");
        assert_eq!(m, "engram/engram_head5.manifest.json");
    }
}
