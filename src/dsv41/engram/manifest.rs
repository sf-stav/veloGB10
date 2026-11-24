//! Engram pack geometry, record format, manifest and the record locator (the read API for EIO).
//!
//! One packed file per Engram head h (0..n_heads). It holds that head's `ngram_cols` columns
//! (column index in the source table = g * n_heads + h, g = n-gram order - 2) of every Engram layer, one
//! record per table row: 256 FP8 value bytes then 8 E8M0 scale bytes (264 B). Sections are laid out
//! layer-major, g-minor; `Section::first_record` is the record index of that column's row 0.
//!
//! File = 4096-byte header page, then the data area. The record of record-index `i` lives at
//! `HEADER_BYTES + data_offset(layout, i)`; see [`RecordLayout`].

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const VALUE_BYTES: usize = 256;
pub const SCALE_BYTES: usize = 8;
pub const RECORD_BYTES: usize = VALUE_BYTES + SCALE_BYTES; // 264
pub const PAGE_BYTES: u64 = 4096;
pub const HEADER_BYTES: u64 = 4096;
pub const RECORDS_PER_PAGE: u64 = PAGE_BYTES / RECORD_BYTES as u64; // 15
pub const MAGIC: &[u8; 8] = b"DSV41ENG";
pub const FORMAT_VERSION: u32 = 1;

/// How records are placed inside the data area.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RecordLayout {
    /// Back to back, stride 264 B. A 4 KiB-aligned read of one record straddles two pages 6.4% of the time.
    Packed,
    /// 15 records per 4 KiB page (3960 B used, 136 B zero pad): a record never crosses a page. +3.4% on disk.
    Page15,
}

impl RecordLayout {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "packed" => Some(Self::Packed),
            "page15" => Some(Self::Page15),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Packed => "packed",
            Self::Page15 => "page15",
        }
    }
    /// Byte offset of record `i` inside the data area.
    #[inline]
    pub fn data_offset(self, i: u64) -> u64 {
        match self {
            Self::Packed => i * RECORD_BYTES as u64,
            Self::Page15 => (i / RECORDS_PER_PAGE) * PAGE_BYTES + (i % RECORDS_PER_PAGE) * RECORD_BYTES as u64,
        }
    }
    /// Data-area bytes needed for `n` records, rounded up to a whole page.
    pub fn data_bytes(self, n: u64) -> u64 {
        let raw = match self {
            Self::Packed => n * RECORD_BYTES as u64,
            Self::Page15 => n.div_ceil(RECORDS_PER_PAGE) * PAGE_BYTES,
        };
        raw.div_ceil(PAGE_BYTES) * PAGE_BYTES
    }
}

// ---------------------------------------------------------------- geometry

/// The one primality test of the Engram layout (ENGa's `layout::is_prime`, deterministic Miller-Rabin).
pub use super::layout::is_prime;

/// Bucket layout of the Engram hash tables: the same arithmetic as `EngramLayout.from_args` in the
/// checkpoint's `inference/engram.py` (checked against its dump in the unit tests).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Geometry {
    pub layer_ids: Vec<u32>,
    pub n_heads: usize,
    pub ngram_cols: usize,
    pub vocab_size: u64,
    pub num_embeddings: Vec<u64>,
    /// primes[layer_idx][g][head]: bucket modulus of column (g, head)
    pub primes: Vec<Vec<Vec<u64>>>,
    /// offsets[layer_idx][g * n_heads + head]: first table row of the column
    pub offsets: Vec<Vec<u64>>,
}

impl Geometry {
    /// The geometry from config values: the prime search and offsets are `layout::column_primes` /
    /// `layout::column_offsets`, the same code `EngramLayout::new` runs (one layout, never two).
    pub fn compute(layer_ids: &[u32], max_ngram_size: usize, n_heads: usize, vocab_size: u64, num_embeddings: &[u64]) -> Self {
        let ngram_cols = max_ngram_size - 1;
        let primes = super::layout::column_primes(layer_ids.len(), ngram_cols, n_heads, vocab_size);
        let offsets = super::layout::column_offsets(&primes);
        Geometry { layer_ids: layer_ids.to_vec(), n_heads, ngram_cols, vocab_size, num_embeddings: num_embeddings.to_vec(), primes, offsets }
    }

    /// The geometry of a resolved hash layout (what a pack for this model must carry).
    pub fn from_layout(lay: &super::layout::EngramLayout) -> Self {
        let c = &lay.cfg;
        Geometry {
            layer_ids: c.layer_ids.clone(),
            n_heads: c.n_heads,
            ngram_cols: c.max_ngram_size - 1,
            vocab_size: c.vocab_size,
            num_embeddings: c.num_embeddings.clone(),
            primes: lay.primes.clone(),
            offsets: lay.offsets.clone(),
        }
    }

    pub fn column_rows(&self, layer_idx: usize, g: usize, head: usize) -> u64 {
        self.primes[layer_idx][g][head]
    }
    pub fn column_offset(&self, layer_idx: usize, g: usize, head: usize) -> u64 {
        self.offsets[layer_idx][g * self.n_heads + head]
    }
    /// Check that the columns tile each layer table exactly.
    pub fn check_tiles(&self) -> Result<(), String> {
        for (l, ne) in self.num_embeddings.iter().enumerate() {
            let sum: u64 = self.primes[l].iter().flatten().sum();
            if sum != *ne {
                return Err(format!("layer idx {l}: prime sum {sum} != num_embeddings {ne}"));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------- manifest

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorRef {
    pub name: String,
    pub shard: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    /// absolute byte offset of the tensor data inside the shard file
    pub abs_offset: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SourceLayer {
    pub layer_id: u32,
    pub weight: TensorRef,
    pub scale: TensorRef,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Section {
    pub layer_idx: usize,
    pub layer_id: u32,
    pub g: usize,
    /// first row of this column in the source table
    pub source_row_start: u64,
    pub rows: u64,
    /// record index (inside the file) of the column's row 0
    pub first_record: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HeadManifest {
    pub format: String,
    pub version: u32,
    pub head: usize,
    pub file: String,
    pub record_bytes: usize,
    pub value_bytes: usize,
    pub scale_bytes: usize,
    pub layout: RecordLayout,
    pub records_per_page: u64,
    pub page_bytes: u64,
    pub header_bytes: u64,
    pub total_records: u64,
    pub file_bytes: u64,
    pub sha256: String,
    pub source_model: String,
    pub source_layers: Vec<SourceLayer>,
    pub geometry: Geometry,
    pub sections: Vec<Section>,
}

pub fn manifest_name(head: usize) -> String {
    format!("engram_head{head}.manifest.json")
}
pub fn file_name(head: usize) -> String {
    format!("engram_head{head}.bin")
}

/// Build the section table for one head (also the record-count source of truth). The order and the record
/// arithmetic are `layout::head_rows`, the function behind `EngramLayout::rows_for_head` (one layout).
pub fn sections_for(geo: &Geometry, head: usize) -> (Vec<Section>, u64) {
    let rows = super::layout::head_rows(&geo.layer_ids, &geo.primes, &geo.offsets, geo.n_heads, head);
    let secs = rows
        .sections
        .iter()
        .map(|h| Section { layer_idx: h.layer_idx, layer_id: h.layer_id, g: h.g, source_row_start: h.source_row_start, rows: h.rows, first_record: h.first_record })
        .collect();
    (secs, rows.total_records)
}

/// 4096-byte file header (sanity check at open; the manifest carries the full description).
pub fn header_page(head: usize, layout: RecordLayout, total_records: u64) -> Vec<u8> {
    let mut h = vec![0u8; HEADER_BYTES as usize];
    h[..8].copy_from_slice(MAGIC);
    h[8..12].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    h[12..16].copy_from_slice(&(head as u32).to_le_bytes());
    h[16..20].copy_from_slice(&(if layout == RecordLayout::Packed { 0u32 } else { 1u32 }).to_le_bytes());
    h[20..24].copy_from_slice(&(RECORD_BYTES as u32).to_le_bytes());
    h[24..32].copy_from_slice(&total_records.to_le_bytes());
    h
}

// ---------------------------------------------------------------- reader API (for EIO)

/// I/O errors with enough context for the reader to decide: retry, fail the request, or stop.
#[derive(Debug)]
pub enum EngramIoError {
    /// An OS error from open/pread/io_uring (`errno` is the raw value, e.g. libc::EIO).
    Os { op: &'static str, errno: i32, file: String, offset: u64 },
    /// Fewer bytes than the record needed came back (truncated file / bad offset).
    Short { file: String, offset: u64, got: usize, want: usize },
    /// (layer_idx, g, row) outside the head's section table.
    OutOfRange { layer_idx: usize, g: usize, row: u64 },
    /// Manifest / header / file size disagree.
    Corrupt(String),
}
impl std::fmt::Display for EngramIoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Os { op, errno, file, offset } => write!(f, "engram {op} failed on {file} @{offset}: errno {errno}"),
            Self::Short { file, offset, got, want } => write!(f, "engram short read on {file} @{offset}: {got}/{want}"),
            Self::OutOfRange { layer_idx, g, row } => write!(f, "engram row out of range: layer_idx {layer_idx} g {g} row {row}"),
            Self::Corrupt(s) => write!(f, "engram pack corrupt: {s}"),
        }
    }
}
impl std::error::Error for EngramIoError {}

/// Where one record lives and the aligned read that covers it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadSpan {
    /// absolute file offset of the record
    pub record_offset: u64,
    /// aligned read start (multiple of `align`)
    pub aligned_offset: u64,
    /// aligned read length (multiple of `align`)
    pub aligned_len: usize,
    /// offset of the record inside the aligned buffer
    pub intra: usize,
}

impl ReadSpan {
    pub fn for_record(record_offset: u64, align: u64) -> Self {
        let a = record_offset / align * align;
        let end = (record_offset + RECORD_BYTES as u64).div_ceil(align) * align;
        ReadSpan { record_offset, aligned_offset: a, aligned_len: (end - a) as usize, intra: (record_offset - a) as usize }
    }
}

/// One opened head pack: manifest + locator. Opening validates the header page and the file size.
pub struct PackedHead {
    pub manifest: HeadManifest,
    pub path: PathBuf,
}

impl PackedHead {
    pub fn open(dir: &Path, head: usize) -> Result<Self, EngramIoError> {
        let mp = dir.join(manifest_name(head));
        let txt = std::fs::read_to_string(&mp).map_err(|e| EngramIoError::Os { op: "read manifest", errno: e.raw_os_error().unwrap_or(0), file: mp.display().to_string(), offset: 0 })?;
        let manifest: HeadManifest = serde_json::from_str(&txt).map_err(|e| EngramIoError::Corrupt(format!("{}: {e}", mp.display())))?;
        if manifest.head != head || manifest.version != FORMAT_VERSION || manifest.record_bytes != RECORD_BYTES {
            return Err(EngramIoError::Corrupt(format!("{}: head/version/record size mismatch", mp.display())));
        }
        let path = dir.join(&manifest.file);
        let md = std::fs::metadata(&path).map_err(|e| EngramIoError::Os { op: "stat", errno: e.raw_os_error().unwrap_or(0), file: path.display().to_string(), offset: 0 })?;
        if md.len() != manifest.file_bytes {
            return Err(EngramIoError::Corrupt(format!("{}: size {} != manifest {}", path.display(), md.len(), manifest.file_bytes)));
        }
        Ok(PackedHead { manifest, path })
    }

    /// Absolute file offset of the record for (layer_idx, g, row-within-column). `row` is the hash value
    /// (`NgramHashState` output minus the column offset).
    #[inline]
    pub fn record_offset(&self, layer_idx: usize, g: usize, row: u64) -> Result<u64, EngramIoError> {
        let n_g = self.manifest.geometry.ngram_cols;
        let s = self.manifest.sections.get(layer_idx * n_g + g).ok_or(EngramIoError::OutOfRange { layer_idx, g, row })?;
        if row >= s.rows {
            return Err(EngramIoError::OutOfRange { layer_idx, g, row });
        }
        Ok(self.manifest.header_bytes + self.manifest.layout.data_offset(s.first_record + row))
    }

    /// The aligned read (`align` = 512 or 4096) that covers the record.
    #[inline]
    pub fn span(&self, layer_idx: usize, g: usize, row: u64, align: u64) -> Result<ReadSpan, EngramIoError> {
        Ok(ReadSpan::for_record(self.record_offset(layer_idx, g, row)?, align))
    }
}

/// Split a 264-byte record into (256 FP8 value bytes, 8 E8M0 scale bytes).
#[inline]
pub fn split_record(rec: &[u8]) -> (&[u8], &[u8]) {
    (&rec[..VALUE_BYTES], &rec[VALUE_BYTES..RECORD_BYTES])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_offsets() {
        assert_eq!(RECORDS_PER_PAGE, 15);
        assert_eq!(RecordLayout::Packed.data_offset(3), 792);
        assert_eq!(RecordLayout::Page15.data_offset(14), 14 * 264);
        assert_eq!(RecordLayout::Page15.data_offset(15), 4096);
        assert_eq!(RecordLayout::Page15.data_bytes(16), 8192);
        // page15 never crosses a 4 KiB page
        for i in 0..1000u64 {
            let o = RecordLayout::Page15.data_offset(i);
            assert_eq!(o / 4096, (o + 263) / 4096);
        }
        let s = ReadSpan::for_record(4096 + 3900, 4096);
        assert_eq!((s.aligned_offset, s.aligned_len, s.intra), (4096, 8192, 3900));
        let s = ReadSpan::for_record(1000, 512);
        assert_eq!((s.aligned_offset, s.aligned_len, s.intra), (512, 1024, 488));
    }

    #[test]
    fn primes_known() {
        assert!(is_prime(16000057) && is_prime(2) && !is_prime(16000058) && !is_prime(1));
        assert!(is_prime(1_000_000_007) && !is_prime(1_000_000_007u64 * 3));
    }

    /// Geometry must equal the dump of the real `inference/engram.py` EngramLayout.
    #[test]
    fn geometry_matches_reference_dump() {
        let j: serde_json::Value = serde_json::from_str(include_str!("../../../tools/dsv41_ref/engram_layout_1017.json")).unwrap();
        let layer_ids: Vec<u32> = j["layer_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let ne: Vec<u64> = j["num_embeddings"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        let geo = Geometry::compute(&layer_ids, 4, 8, j["vocab_size"].as_u64().unwrap(), &ne);
        geo.check_tiles().unwrap();
        for (l, layer) in j["primes"].as_array().unwrap().iter().enumerate() {
            for (g, row) in layer.as_array().unwrap().iter().enumerate() {
                for (h, p) in row.as_array().unwrap().iter().enumerate() {
                    assert_eq!(geo.primes[l][g][h], p.as_u64().unwrap(), "prime l{l} g{g} h{h}");
                    assert_eq!(geo.column_offset(l, g, h), j["offsets"][l][g * 8 + h].as_u64().unwrap());
                }
            }
        }
    }

    /// One layout, never two (leg EIO): the packer's geometry from config values equals the geometry of the
    /// hash side's resolved layout (both now call `layout::column_primes` / `column_offsets`; the section
    /// order is `layout::head_rows` on both sides, cross-checked head by head in `cross_tests.rs`), and
    /// `sections_for` reproduces `rows_for_head` for every head of the real V4.1 config.
    #[test]
    fn geometry_from_config_equals_geometry_from_layout() {
        use super::super::layout::{EngramConfig, EngramLayout};
        let j: serde_json::Value = serde_json::from_str(include_str!("../../../tools/dsv41_ref/engram_layout_1017.json")).unwrap();
        let layer_ids: Vec<u32> = j["layer_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let ne: Vec<u64> = j["num_embeddings"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        let vocab = j["vocab_size"].as_u64().unwrap();
        let cfg = EngramConfig { layer_ids: layer_ids.clone(), max_ngram_size: 4, vocab_size: vocab, n_heads: 8, head_dim: 256,
                                 num_embeddings: ne.clone(), pad_id: 2, compressed_vocab_size: 99_092 };
        let lay = EngramLayout::new(cfg).unwrap();
        let geo = Geometry::compute(&layer_ids, 4, 8, vocab, &ne);
        assert_eq!(geo, Geometry::from_layout(&lay));
        for h in 0..8 {
            let rows = lay.rows_for_head(h);
            let (secs, total) = sections_for(&geo, h);
            assert_eq!(total, rows.total_records, "head {h}");
            assert!(secs.iter().zip(&rows.sections).all(|(a, b)| (a.layer_idx, a.g, a.source_row_start, a.rows, a.first_record)
                == (b.layer_idx, b.g, b.source_row_start, b.rows, b.first_record)), "head {h}");
        }
    }

    #[test]
    fn sections_tile_the_file() {
        let geo = Geometry::compute(&[1, 14], 4, 8, 16_000_000, &[384006168, 384016682]);
        let (secs, total) = sections_for(&geo, 5);
        assert_eq!(secs.len(), 6);
        assert_eq!(secs[1].first_record, secs[0].rows);
        assert_eq!(total, secs.iter().map(|s| s.rows).sum::<u64>());
        assert!(total > 95_000_000 && total < 97_000_000);
    }
}
