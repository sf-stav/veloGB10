//! `--dsv41-engram-pack`: stream the Engram tables of a native checkpoint into one packed file per head.

use super::manifest::*;
use super::source;
use anyhow::{anyhow, bail, Context, Result};
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use crate::boot_report::{Progress, Unit};

/// Record layout chosen by the read benchmark (see PLAN/DSV41_ENGB_REPORT.md).
pub const DEFAULT_LAYOUT: &str = "packed";
const CHUNK_ROWS: usize = 32768;
const SINK_BYTES: usize = 4 << 20;

/// Free bytes on the filesystem holding `p`.
pub fn free_bytes(p: &Path) -> Result<u64> {
    let c = std::ffi::CString::new(p.to_str().ok_or_else(|| anyhow!("non-utf8 path"))?)?;
    let mut s: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c.as_ptr(), &mut s) } != 0 {
        bail!("statvfs {}: {}", p.display(), std::io::Error::last_os_error());
    }
    Ok(s.f_bavail as u64 * s.f_frsize as u64)
}

pub fn parse_heads(s: &str, n_heads: usize) -> Result<Vec<usize>> {
    let mut v = vec![];
    for part in s.split(',').filter(|p| !p.is_empty()) {
        if let Some((a, b)) = part.split_once('-') {
            let (a, b): (usize, usize) = (a.parse()?, b.parse()?);
            v.extend(a..=b);
        } else {
            v.push(part.parse()?);
        }
    }
    v.sort();
    v.dedup();
    if v.is_empty() || v.iter().any(|h| *h >= n_heads) {
        bail!("--engram-heads '{s}': need heads in 0..{n_heads}");
    }
    Ok(v)
}

/// Where a [`Sink`] writes: a file (page-cache friendly write-back) or any stream (`WriterOut`, LS2's streamed ship).
pub trait SinkOut: Write {
    /// Bytes `[from, from+len)` were written and are not needed again (file: write back and drop them from the page cache).
    fn behind(&mut self, _from: u64, _len: u64) {}
    /// The data is complete (file: fsync).
    fn done(&mut self) -> Result<()> { Ok(()) }
}
impl SinkOut for File {
    fn behind(&mut self, from: u64, len: u64) {
        let fd = self.as_raw_fd();
        unsafe {
            libc::sync_file_range(fd, from as i64, len as i64, libc::SYNC_FILE_RANGE_WAIT_BEFORE | libc::SYNC_FILE_RANGE_WRITE | libc::SYNC_FILE_RANGE_WAIT_AFTER);
            libc::posix_fadvise(fd, from as i64, len as i64, libc::POSIX_FADV_DONTNEED);
        }
    }
    fn done(&mut self) -> Result<()> { self.sync_all()?; Ok(()) }
}
/// A stream (a socket writer, a hasher): nothing to write back.
pub struct WriterOut<W: Write>(pub W);
impl<W: Write> Write for WriterOut<W> {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> { self.0.write(b) }
    fn flush(&mut self) -> std::io::Result<()> { self.0.flush() }
}
impl<W: Write> SinkOut for WriterOut<W> {}

/// Writes the data area of one head file: lays records out per `RecordLayout`, hashes everything.
pub struct Sink<W: SinkOut = File> {
    f: W,
    layout: RecordLayout,
    buf: Vec<u8>,
    in_page: u64,
    pub written: u64,
    sha: Sha256,
    synced: u64,
}
impl<W: SinkOut> Sink<W> {
    pub fn new(f: W, layout: RecordLayout, header: &[u8]) -> Result<Self> {
        let mut s = Sink { f, layout, buf: Vec::with_capacity(SINK_BYTES + 8192), in_page: 0, written: 0, sha: Sha256::new(), synced: 0 };
        s.buf.extend_from_slice(header);
        Ok(s)
    }
    fn flush(&mut self) -> Result<()> {
        if self.buf.is_empty() {
            return Ok(());
        }
        self.sha.update(&self.buf);
        self.f.write_all(&self.buf)?;
        self.written += self.buf.len() as u64;
        self.buf.clear();
        // keep the page cache from piling up on a shared box: write back + drop what is behind us
        if self.written - self.synced >= (1 << 30) {
            self.f.behind(self.synced, self.written - self.synced);
            self.synced = self.written;
        }
        Ok(())
    }
    pub fn push_records(&mut self, recs: &[u8]) -> Result<()> {
        debug_assert_eq!(recs.len() % RECORD_BYTES, 0);
        match self.layout {
            RecordLayout::Packed => self.buf.extend_from_slice(recs),
            RecordLayout::Page15 => {
                for r in recs.chunks_exact(RECORD_BYTES) {
                    self.buf.extend_from_slice(r);
                    self.in_page += 1;
                    if self.in_page == RECORDS_PER_PAGE {
                        self.buf.resize(self.buf.len() + (PAGE_BYTES as usize - RECORDS_PER_PAGE as usize * RECORD_BYTES), 0);
                        self.in_page = 0;
                    }
                }
            }
        }
        if self.buf.len() >= SINK_BYTES {
            self.flush()?;
        }
        Ok(())
    }
    /// Pad to a whole page, flush, fsync; returns (file bytes, sha256 hex).
    pub fn finish(mut self) -> Result<(u64, String)> {
        if self.layout == RecordLayout::Page15 && self.in_page > 0 {
            let used = self.in_page as usize * RECORD_BYTES;
            self.buf.resize(self.buf.len() + (PAGE_BYTES as usize - used), 0);
        }
        // total file length is a multiple of 4096 (header is one page)
        let tail = (self.written + self.buf.len() as u64) % PAGE_BYTES;
        if tail != 0 {
            self.buf.resize(self.buf.len() + (PAGE_BYTES - tail) as usize, 0);
        }
        self.flush()?;
        self.f.flush()?;
        self.f.done()?;
        let hex: String = self.sha.finalize().iter().map(|b| format!("{b:02x}")).collect();
        Ok((self.written, hex))
    }
}

/// Pack one head's file bytes into `out` (a file, or a stream: LS2 sends them without writing them on this disk).
/// Returns (file bytes, sha256 hex).
pub fn pack_head_into<W: SinkOut>(model_dir: &Path, src: &source::SourceConfig, head: usize, layout: RecordLayout, out: W, progress: &Progress) -> Result<(u64, String)> {
    let geo = &src.geometry;
    let (sections, total_records) = sections_for(geo, head);
    let files = source::SourceFiles::open(model_dir, &src.layers)?;
    let mut sink = Sink::new(out, layout, &header_page(head, layout, total_records))?;
    let (mut wbuf, mut sbuf, mut rec) = (vec![], vec![], vec![]);
    let mut n_written = 0u64;
    for s in &sections {
        let mut row = 0u64;
        while row < s.rows {
            let n = ((s.rows - row) as usize).min(CHUNK_ROWS);
            source::read_records(&files, &src.layers, s.layer_idx, s.source_row_start + row, n, &mut wbuf, &mut sbuf, &mut rec)?;
            sink.push_records(&rec)?;
            // source pages are never needed again
            source::fadvise_dontneed(&files.weight[s.layer_idx], src.layers[s.layer_idx].weight.abs_offset + (s.source_row_start + row) * VALUE_BYTES as u64, (n * VALUE_BYTES) as u64);
            source::fadvise_dontneed(&files.scale[s.layer_idx], src.layers[s.layer_idx].scale.abs_offset + (s.source_row_start + row) * SCALE_BYTES as u64, (n * SCALE_BYTES) as u64);
            row += n as u64;
            n_written += n as u64;
            progress.add((n * RECORD_BYTES) as u64);
        }
    }
    if n_written != total_records {
        bail!("head {head}: wrote {n_written} records, expected {total_records}");
    }
    let (file_bytes, sha256) = sink.finish()?;
    let expect = HEADER_BYTES + layout.data_bytes(total_records);
    if file_bytes != expect {
        bail!("head {head}: file is {file_bytes} B, expected {expect}");
    }
    Ok((file_bytes, sha256))
}

/// Pack one head. `progress` is a shared byte counter.
pub fn pack_head(model_dir: &Path, src: &source::SourceConfig, head: usize, out_dir: &Path, layout: RecordLayout, progress: &Progress) -> Result<HeadManifest> {
    let final_path = out_dir.join(file_name(head));
    let part_path = out_dir.join(format!("{}.part", file_name(head)));
    let f = OpenOptions::new().write(true).create(true).truncate(true).open(&part_path).with_context(|| format!("create {}", part_path.display()))?;
    let (file_bytes, sha256) = pack_head_into(model_dir, src, head, layout, f, progress)?;
    std::fs::rename(&part_path, &final_path)?;
    let m = manifest_of(model_dir, src, head, layout, file_bytes, sha256);
    let mp = out_dir.join(manifest_name(head));
    std::fs::write(&mp, serde_json::to_vec_pretty(&m)?)?;
    Ok(m)
}

/// The manifest of a packed head (a pure function of the source, the head, the layout and the file's size and sha256:
/// LS2 builds it for a head it streams without ever holding the file).
pub fn manifest_of(model_dir: &Path, src: &source::SourceConfig, head: usize, layout: RecordLayout, file_bytes: u64, sha256: String) -> HeadManifest {
    let geo = &src.geometry;
    let (sections, total_records) = sections_for(geo, head);
    HeadManifest {
        format: "dsv41-engram-pack".into(),
        version: FORMAT_VERSION,
        head,
        file: file_name(head),
        record_bytes: RECORD_BYTES,
        value_bytes: VALUE_BYTES,
        scale_bytes: SCALE_BYTES,
        layout,
        records_per_page: RECORDS_PER_PAGE,
        page_bytes: PAGE_BYTES,
        header_bytes: HEADER_BYTES,
        total_records,
        file_bytes,
        sha256,
        source_model: model_dir.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
        source_layers: src.layers.clone(),
        geometry: geo.clone(),
        sections,
    }
}

/// Bytes a head file needs for `layout`.
pub fn head_bytes(geo: &Geometry, head: usize, layout: RecordLayout) -> u64 {
    HEADER_BYTES + layout.data_bytes(sections_for(geo, head).1)
}

/// Library API for the boot path (leg LS): pack `heads` of the native checkpoint into `out_dir`.
///
/// Silent except for the `[boot] engram pack: NN%` progress lines (boot_report::Progress). Heads whose
/// manifest already exists in `out_dir` are skipped (idempotent: a re-run packs nothing). Checks free
/// space first. Each head ends up as `engram_head{h}.bin` + `engram_head{h}.manifest.json`; the manifest
/// carries the file's sha256, so the pair is content-addressable for the TP blob cache and a rank is shipped
/// only its own heads. Returns the manifests of ALL requested heads (packed now or found).
pub fn pack_heads(model_dir: &Path, heads: &[usize], out_dir: &Path, layout: RecordLayout, threads: usize, force: bool) -> Result<Vec<HeadManifest>> {
    let src = source::load(model_dir)?;
    for h in heads {
        if *h >= src.geometry.n_heads {
            bail!("engram head {h} out of range 0..{}", src.geometry.n_heads);
        }
    }
    std::fs::create_dir_all(out_dir)?;
    let todo: Vec<usize> = heads.iter().copied().filter(|h| force || !(out_dir.join(manifest_name(*h)).exists() && out_dir.join(file_name(*h)).exists())).collect();
    let need: u64 = todo.iter().map(|h| head_bytes(&src.geometry, *h, layout)).sum();
    if !todo.is_empty() {
        let free = free_bytes(out_dir)?;
        let margin = 8u64 << 30;
        if need + margin > free {
            bail!("not enough free space in {} for the Engram pack: need {:.1} GB + {:.0} GB margin, have {:.1} GB", out_dir.display(), need as f64 / 1e9, margin as f64 / 1e9, free as f64 / 1e9);
        }
    }
    let progress = Progress::new("engram pack", need, Unit::Bytes);
    let queue = parking_lot::Mutex::new(todo.clone());
    let errors = parking_lot::Mutex::new(Vec::<String>::new());
    std::thread::scope(|sc| {
        for _ in 0..threads.max(1).min(todo.len().max(1)) {
            sc.spawn(|| loop {
                let h = match queue.lock().pop() {
                    Some(h) => h,
                    None => break,
                };
                if let Err(e) = pack_head(model_dir, &src, h, out_dir, layout, &progress) {
                    errors.lock().push(format!("head {h}: {e:#}"));
                }
            });
        }
    });
    if let Some(e) = errors.lock().first() {
        bail!("engram pack failed: {e}");
    }
    progress.finish();
    heads.iter().map(|h| Ok(PackedHead::open(out_dir, *h).map_err(|e| anyhow!("{e}"))?.manifest)).collect()
}

/// `--dsv41-engram-pack` for operators who pre-stage the files.
pub fn run_cli(args: &[String]) -> Result<()> {
    let arg = |k: &str| crate::arg_value(args, k).map(|s| s.to_string());
    let model_dir = PathBuf::from(arg("--engram-source").ok_or_else(|| anyhow!("--engram-source <native model dir> required"))?);
    let out = PathBuf::from(arg("--out").ok_or_else(|| anyhow!("--out <dir> required"))?);
    let layout = RecordLayout::parse(&arg("--engram-record-layout").unwrap_or_else(|| DEFAULT_LAYOUT.into())).ok_or_else(|| anyhow!("--engram-record-layout packed|page15"))?;
    let n_heads = source::load(&model_dir)?.geometry.n_heads;
    let heads = parse_heads(&arg("--engram-heads").unwrap_or_else(|| "0-7".into()), n_heads)?;
    let threads = arg("--engram-pack-threads").and_then(|s| s.parse().ok()).unwrap_or(4usize);
    let t0 = std::time::Instant::now();
    let ms = pack_heads(&model_dir, &heads, &out, layout, threads, args.iter().any(|a| a == "--engram-force"))?;
    for m in &ms {
        println!("head {}: {} records, {} B, layout {}, sha256 {}", m.head, m.total_records, m.file_bytes, m.layout.name(), m.sha256);
    }
    println!("pack done in {:.0}s", t0.elapsed().as_secs_f64());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_records(n: usize) -> Vec<u8> {
        (0..n * RECORD_BYTES).map(|i| ((i * 31 + i / 264) % 251) as u8).collect()
    }

    #[test]
    fn sink_layouts_place_records_where_the_locator_says() {
        for layout in [RecordLayout::Packed, RecordLayout::Page15] {
            let dir = std::env::temp_dir().join(format!("engb_sink_{}_{}", layout.name(), std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let p = dir.join("t.bin");
            let n = 100usize;
            let recs = synth_records(n);
            let mut sink = Sink::new(File::create(&p).unwrap(), layout, &header_page(0, layout, n as u64)).unwrap();
            for c in recs.chunks(RECORD_BYTES * 7) {
                sink.push_records(c).unwrap();
            }
            let (bytes, _) = sink.finish().unwrap();
            let data = std::fs::read(&p).unwrap();
            assert_eq!(data.len() as u64, bytes);
            assert_eq!(bytes % PAGE_BYTES, 0);
            assert_eq!(bytes, HEADER_BYTES + layout.data_bytes(n as u64));
            assert_eq!(&data[..8], MAGIC);
            for i in 0..n {
                let o = (HEADER_BYTES + layout.data_offset(i as u64)) as usize;
                assert_eq!(&data[o..o + RECORD_BYTES], &recs[i * RECORD_BYTES..(i + 1) * RECORD_BYTES], "{layout:?} rec {i}");
            }
            std::fs::remove_dir_all(&dir).ok();
        }
    }

    #[test]
    fn parse_heads_ranges() {
        assert_eq!(parse_heads("0-3,6", 8).unwrap(), vec![0, 1, 2, 3, 6]);
        assert!(parse_heads("8", 8).is_err());
    }
}
