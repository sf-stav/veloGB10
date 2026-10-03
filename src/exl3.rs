//! EXL3 (ExLlamaV3) pack loader — track A3, session S-A3-b.
//!
//! Scope discipline (session brief): loader + census + the pure decode ONLY.
//! The weight-path GEMM is S-A3-c; no serving integration, no MTP wiring (S-A3-e).
//! Correctness contract: the decode here reproduces `ext.reconstruct` BIT-EXACTLY
//! (proven 2026-09-19 on the 3.05bpw Flash-Next pack: 3.8M blocks across bits
//! {3,4,5}, 0 mismatched fp16 patterns — see PLAN/S_A3_B_LOADER_ORACLE.md).
//!
//! Format (PLAN/A3_EXL3_FORMAT_SPEC.md, derived from reference @ 523ecd3):
//! one quantized matrix = quadruple `.trellis` int16 [K/16, N/16, 16*bits] +
//! `.suh` fp16 [K] + `.svh` fp16 [N] + `.mul1` int32. Decode: 16-bit windows over
//! each block's little-endian uint32 ring; window for temporal index t ENDS at bit
//! (t+257)*bits; extraction is a funnel shift on the REVERSED word pair (later
//! word LOW). Values come from the mul1 codebook (64K fp16 entries). Output
//! position (r,c) reads temporal index t(r,c) — see `tmap`.

use anyhow::{bail, Context};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// Tensor suffixes that make up a quantized-matrix quadruple.
pub const EXL3_SUFFIXES: [&str; 4] = ["trellis", "suh", "svh", "mul1"];

/// Suffixes the census tracks explicitly (everything else lands in "other").
const CENSUS_SUFFIXES: [&str; 12] = [
    "trellis", "suh", "svh", "mul1", "weight", "bias", "A_log", "dt_bias", "head_bias",
    "head_offsets", "head_vocab_sizes", "layer_multipliers",
];

#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord)]
pub enum TensorClass {
    Expert,
    Dense,
    LmHead,
    Mtp,
    Ngram,
}

impl TensorClass {
    pub fn name(self) -> &'static str {
        match self {
            TensorClass::Expert => "expert",
            TensorClass::Dense => "dense",
            TensorClass::LmHead => "lm_head",
            TensorClass::Mtp => "mtp",
            TensorClass::Ngram => "ngram",
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorMeta {
    pub name: String,
    pub shard: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Absolute file offset of the tensor data (safetensors header skipped).
    pub offset: usize,
    pub n_bytes: usize,
}

impl TensorMeta {
    /// Read the raw tensor bytes from its shard (little-endian on disk as stored).
    pub fn read_bytes(&self, dir: &str) -> anyhow::Result<Vec<u8>> {
        use std::io::{Read, Seek, SeekFrom};
        let path = format!("{}/{}", dir.trim_end_matches('/'), self.shard);
        let mut f = std::fs::File::open(&path)
            .with_context(|| format!("open {} for tensor {}", path, self.name))?;
        f.seek(SeekFrom::Start(self.offset as u64))
            .with_context(|| format!("seek {} @ {} for {}", path, self.offset, self.name))?;
        let mut buf = vec![0u8; self.n_bytes];
        f.read_exact(&mut buf)
            .with_context(|| format!("read {} B of {} from {}", self.n_bytes, self.name, path))?;
        Ok(buf)
    }
}

#[derive(Clone, Debug)]
pub struct ModuleMeta {
    /// Module key without the suffix, e.g. "model.language_model.layers.1.mlp.experts.0.down_proj".
    pub name: String,
    /// bits per weight (= trellis shape[-1] / 16). The reference calls this K.
    pub bits: usize,
    /// Matrix rows (input dim) = trellis shape[0] * 16.
    pub k: usize,
    /// Matrix cols (output dim) = trellis shape[1] * 16.
    pub n: usize,
    pub trellis: TensorMeta,
    pub suh: TensorMeta,
    pub svh: TensorMeta,
    pub mul1: TensorMeta,
    pub class: TensorClass,
}

#[derive(Clone, Debug, Default)]
pub struct SuffixStat {
    pub count: usize,
    pub bytes: usize,
    pub dtypes: BTreeSet<String>,
}

#[derive(Clone, Debug, Default)]
pub struct Census {
    pub per_suffix: BTreeMap<String, SuffixStat>,
    pub total_tensors: usize,
    pub total_bytes: usize,
    pub incomplete_quadruples: usize,
}

impl Census {
    pub fn stat(&self, suffix: &str) -> SuffixStat {
        self.per_suffix.get(suffix).cloned().unwrap_or_default()
    }

    /// The measured manifest for the known pack (spec §1.3, index-wide counts).
    /// Other packs are not judged by it (see `probe_census`).
    fn known_pack_manifest() -> BTreeMap<&'static str, usize> {
        BTreeMap::from([
            ("trellis", 75_879),
            ("suh", 75_751),
            ("svh", 75_751),
            ("mul1", 75_751),
            ("weight", 785),
            ("bias", 247),
            ("A_log", 36),
            ("dt_bias", 36),
            ("head_bias", 1),
            ("head_offsets", 1),
            ("head_vocab_sizes", 1),
            ("layer_multipliers", 1),
        ])
    }

    /// Compare against the known-pack manifest. Returns the mismatch lines (empty = pass).
    pub fn manifest_mismatches(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (suffix, want) in Self::known_pack_manifest() {
            let got = self.stat(suffix).count;
            if got != want {
                out.push(format!(
                    "  MANIFEST MISMATCH suffix={suffix}: got {got}, spec says {want}"
                ));
            }
        }
        out
    }
}

#[derive(Clone, Debug)]
pub struct Sidecar {
    pub file: String,
    pub tensors: usize,
    pub bytes: usize,
    pub note: &'static str,
}

#[derive(Clone, Debug)]
pub struct QuantGlobals {
    pub quant_method: String,
    pub version: String,
    /// Average bits per weight across the pack (float, e.g. 3.05) — informational.
    pub bits_avg: f64,
    pub head_bits: f64,
    pub mtp_bits: f64,
    /// Only "mul1" is supported (the pack uses it exclusively; "mcg" is the other
    /// codebook family and is NOT implemented — refuse loudly rather than mis-decode).
    pub codebook: String,
}

pub struct Exl3Pack {
    pub dir: String,
    /// Parsed with the COMMON qwen family gate — Family::Qwen4Exp for Flash-Next.
    /// An EXL3 dir whose config says anything else is refused by `open` (wrong-model hazard).
    pub cfg: crate::qwen::Config,
    pub quant: QuantGlobals,
    pub census: Census,
    pub modules: Vec<ModuleMeta>,
    /// Quantized-module counts per class.
    pub class_counts: BTreeMap<TensorClass, usize>,
    pub sidecars: Vec<Sidecar>,
    pub shards: Vec<String>,
}

/// Directory looks like an EXL3 pack: a quantization_config.json declaring exl3.
pub fn is_exl3_dir(dir: &str) -> bool {
    let path = format!("{}/quantization_config.json", dir.trim_end_matches('/'));
    std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| v["quant_method"].as_str().map(|s| s == "exl3"))
        .unwrap_or(false)
}

/// One tensor's entry from a safetensors JSON header.
#[derive(Clone, Debug)]
pub struct HdrEntry {
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Absolute file offset of the tensor data (8-byte length prefix + JSON header skipped).
    pub offset: usize,
    pub n_bytes: usize,
}

/// Header-only safetensors parse: 8-byte LE length prefix + JSON header. NO tensor
/// bytes are touched — a census must not page 12 GB shards through memory.
pub(crate) fn shard_header(path: &str) -> anyhow::Result<BTreeMap<String, HdrEntry>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).with_context(|| format!("open shard {path}"))?;
    let mut len8 = [0u8; 8];
    f.read_exact(&mut len8)
        .with_context(|| format!("read header length {path}"))?;
    let n = u64::from_le_bytes(len8) as usize;
    let mut buf = vec![0u8; n];
    f.read_exact(&mut buf)
        .with_context(|| format!("read header json ({n} B) {path}"))?;
    let v: serde_json::Value = serde_json::from_slice(&buf)
        .with_context(|| format!("parse safetensors header json {path}"))?;
    let obj = v
        .as_object()
        .with_context(|| format!("safetensors header not an object: {path}"))?;
    let mut out = BTreeMap::new();
    for (name, e) in obj {
        if name == "__metadata__" {
            continue;
        }
        let shape = e["shape"]
            .as_array()
            .map(|a| a.iter().filter_map(|x| x.as_u64().map(|x| x as usize)).collect())
            .unwrap_or_default();
        let offs: Vec<u64> = e["data_offsets"]
            .as_array()
            .map(|a| a.iter().filter_map(|x| x.as_u64()).collect())
            .unwrap_or_default();
        let (s, t) = match offs.as_slice() {
            [a, b] => (*a as usize, *b as usize),
            _ => (0usize, 0usize),
        };
        out.insert(
            name.clone(),
            HdrEntry {
                dtype: e["dtype"].as_str().unwrap_or("?").to_string(),
                shape,
                offset: 8 + n + s,
                n_bytes: t.saturating_sub(s),
            },
        );
    }
    Ok(out)
}

/// Classify a module/tensor name. Order matters: lm_head before the generic
/// suffix walk; ngram shards carry "ngram_embedding".
fn classify(name: &str) -> TensorClass {
    if name.starts_with("lm_head") {
        TensorClass::LmHead
    } else if name.starts_with("mtp.") {
        TensorClass::Mtp
    } else if name.contains("ngram_embedding") {
        TensorClass::Ngram
    } else if name.contains(".mlp.experts.") {
        TensorClass::Expert
    } else {
        TensorClass::Dense
    }
}

impl Exl3Pack {
    /// Open + fully parse an EXL3 pack directory. Loud failures everywhere (the
    /// wrong-model hazard): wrong family, wrong quant_method, mcg codebook,
    /// incomplete quadruples — all refuse before anything downstream can inherit.
    pub fn open(dir: &str) -> anyhow::Result<Exl3Pack> {
        let dir = dir.trim_end_matches('/').to_string();

        // 1. Config + family gate (COMMON family resolution — no parallel family).
        let cfg = crate::qwen::Config::from_config_json(&format!("{dir}/config.json"))
            .with_context(|| format!("EXL3 pack {dir}: config.json missing or unparsable"))?;
        if cfg.family != crate::qwen::Family::Qwen4Exp {
            bail!(
                "EXL3 pack {dir}: config.json declares family {:?} — this EXL3 loader only \
                 serves the qwen4_exp family (the wrong-model hazard). Refusing.",
                cfg.family
            );
        }

        // 2. quantization_config.json — quant_method + codebook gate.
        let qraw = std::fs::read_to_string(format!("{dir}/quantization_config.json"))
            .with_context(|| format!("EXL3 pack {dir}: quantization_config.json missing"))?;
        let q: serde_json::Value = serde_json::from_str(&qraw)
            .with_context(|| format!("EXL3 pack {dir}: quantization_config.json unparsable"))?;
        let quant = QuantGlobals {
            quant_method: q["quant_method"].as_str().unwrap_or("").to_string(),
            version: q["version"].as_str().unwrap_or("").to_string(),
            bits_avg: q["bits"].as_f64().unwrap_or(0.0),
            head_bits: q["head_bits"].as_f64().unwrap_or(0.0),
            mtp_bits: q["mtp_bits"].as_f64().unwrap_or(0.0),
            codebook: q["codebook"].as_str().unwrap_or("").to_string(),
        };
        // CF-P1g: a pack with a K = 6 module needs the K = 6 kernel module (decided here, before any module loads).
        let max_k = q["tensor_storage"].as_object()
            .map(|o| o.values().filter_map(|v| v["bits_per_weight"].as_u64()).max().unwrap_or(0)).unwrap_or(0);
        crate::exl3_bench::set_k6_module(max_k >= 6);
        if quant.quant_method != "exl3" {
            bail!(
                "EXL3 pack {dir}: quant_method={:?} (wanted \"exl3\") — refusing to guess",
                quant.quant_method
            );
        }
        if quant.codebook != "mul1" {
            bail!(
                "EXL3 pack {dir}: codebook={:?} — only \"mul1\" is implemented (spec §0.1). \
                 An mcg pack would decode to garbage. Refusing.",
                quant.codebook
            );
        }

        // 3. Index.
        let iraw = std::fs::read_to_string(format!("{dir}/model.safetensors.index.json"))
            .with_context(|| format!("EXL3 pack {dir}: model.safetensors.index.json missing"))?;
        let idx: serde_json::Value = serde_json::from_str(&iraw)
            .with_context(|| format!("EXL3 pack {dir}: index unparsable"))?;
        let weight_map = idx["weight_map"]
            .as_object()
            .with_context(|| format!("EXL3 pack {dir}: index has no weight_map"))?
            .clone();

        // 4. Shard headers (metadata only — no tensor bytes read).
        let mut shard_files: BTreeSet<String> = BTreeSet::new();
        for shard in weight_map.values() {
            if let Some(s) = shard.as_str() {
                shard_files.insert(s.to_string());
            }
        }
        // name -> (shard, shape, dtype, n_bytes, abs_offset)
        let mut shapes: BTreeMap<String, (String, Vec<usize>, String, usize, usize)> =
            BTreeMap::new();
        for shard in &shard_files {
            for (name, e) in shard_header(&format!("{dir}/{shard}"))? {
                shapes.insert(name, (shard.clone(), e.shape, e.dtype, e.n_bytes, e.offset));
            }
        }

        // 5. Census (suffix -> counts/bytes/dtypes).
        let mut census = Census::default();
        for (name, (_shard, _shape, dtype, bytes, _off)) in &shapes {
            let suffix = name.rsplit('.').next().unwrap_or("");
            let e = census.per_suffix.entry(suffix.to_string()).or_default();
            e.count += 1;
            e.bytes += bytes;
            e.dtypes.insert(dtype.clone());
            census.total_tensors += 1;
            census.total_bytes += bytes;
        }

        // 6. Module catalog from .trellis entries (quadruple completeness enforced).
        let mut modules = Vec::new();
        let mut class_counts: BTreeMap<TensorClass, usize> = BTreeMap::new();
        let mut incomplete = 0usize;
        for (name, (shard, shape, dtype, bytes, off)) in &shapes {
            if !name.ends_with(".trellis") || shape.len() != 3 {
                continue;
            }
            let module = name.trim_end_matches(".trellis").to_string();
            let pick = |suffix: &str| -> anyhow::Result<TensorMeta> {
                let n = format!("{module}.{suffix}");
                match shapes.get(&n) {
                    Some((sh, sshape, dt, by, soff)) => Ok(TensorMeta {
                        name: n,
                        shard: sh.clone(),
                        dtype: dt.clone(),
                        shape: sshape.clone(),
                        offset: *soff,
                        n_bytes: *by,
                    }),
                    None => bail!("EXL3 pack {dir}: module {module} is missing .{suffix}"),
                }
            };
            let suh = pick("suh");
            let svh = pick("svh");
            let mul1 = pick("mul1");
            let (suh, svh, mul1) = match (suh, svh, mul1) {
                (Ok(a), Ok(b), Ok(c)) => (a, b, c),
                (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => {
                    incomplete += 1;
                    eprintln!("WARNING: {e}");
                    continue;
                }
            };
            let bits = shape[2] / 16;
            if shape[2] % 16 != 0 || bits == 0 || bits > 8 {
                bail!(
                    "EXL3 pack {dir}: {name}: trellis last dim {} is not 16*bits (bits 1..8)",
                    shape[2]
                );
            }
            let class = classify(&module);
            *class_counts.entry(class).or_insert(0) += 1;
            modules.push(ModuleMeta {
                name: module,
                bits,
                k: shape[0] * 16,
                n: shape[1] * 16,
                trellis: TensorMeta {
                    name: name.clone(),
                    shard: shard.clone(),
                    dtype: dtype.clone(),
                    shape: shape.clone(),
                    offset: *off,
                    n_bytes: *bytes,
                },
                suh,
                svh,
                mul1,
                class,
            });
        }
        census.incomplete_quadruples = incomplete;
        modules.sort_by(|a, b| a.name.cmp(&b.name));

        // 7. Sidecars (catalogued; the MTP mixer patch is redundant in-pack data —
        //    spec §6.3 says prefer the main-shard copies and record the patch as ignored).
        let mut sidecars = Vec::new();
        for (file, note) in [
            (
                "ngram_embedding.safetensors",
                "PLE n-gram tables, exl3_ngram_trellis format (spec §8) — decoded at load in B3, catalogued only here",
            ),
            (
                "mtp_hyper_connection_mixer_patch.safetensors",
                "redundant in-pack MTP mixers — main-shard copies preferred (spec §6.3); IGNORED",
            ),
        ] {
            let p = format!("{dir}/{file}");
            if !Path::new(&p).exists() {
                continue;
            }
            let hdr =
                shard_header(&p).with_context(|| format!("EXL3 pack {dir}: sidecar {file}"))?;
            let tensors = hdr.len();
            let bytes: usize = hdr.values().map(|e| e.n_bytes).sum();
            sidecars.push(Sidecar { file: file.to_string(), tensors, bytes, note });
        }

        Ok(Exl3Pack {
            dir,
            cfg,
            quant,
            census,
            modules,
            class_counts,
            sidecars,
            shards: shard_files.into_iter().collect(),
        })
    }
}

/// `--probe-exl3` entry: full census print + manifest compare. No model load.
pub fn probe_census(dir: &str) -> anyhow::Result<()> {
    let pack = Exl3Pack::open(dir)?;
    println!("EXL3 CENSUS {}", pack.dir);
    println!(
        "  family={:?} model_type gate OK (config.json -> qwen::Config)",
        pack.cfg.family
    );
    println!(
        "  quant_method={} version={} codebook={} bits_avg={:.2} head_bits={:.0} mtp_bits={:.0}",
        pack.quant.quant_method,
        pack.quant.version,
        pack.quant.codebook,
        pack.quant.bits_avg,
        pack.quant.head_bits,
        pack.quant.mtp_bits
    );
    println!(
        "  index: {} tensors across {} shards",
        pack.census.total_tensors,
        pack.shards.len()
    );
    for s in CENSUS_SUFFIXES {
        let st = pack.census.stat(s);
        if st.count == 0 {
            continue;
        }
        println!(
            "  suffix {:<18} count={:<7} bytes={:<14} dtypes={}",
            s,
            st.count,
            st.bytes,
            st.dtypes.iter().cloned().collect::<Vec<_>>().join(",")
        );
    }
    let other: usize = pack
        .census
        .per_suffix
        .iter()
        .filter(|(k, _)| !CENSUS_SUFFIXES.contains(&k.as_str()))
        .map(|(_, v)| v.count)
        .sum();
    if other > 0 {
        println!("  suffix other              count={other:<7} (untracked suffixes)");
    }
    println!(
        "  quadruples: {} complete, {} incomplete",
        pack.modules.len(),
        pack.census.incomplete_quadruples
    );
    let mut classes: Vec<_> = pack.class_counts.iter().collect();
    classes.sort_by_key(|(c, _)| **c);
    for (c, n) in classes {
        println!("  class {:<8} modules={n}", c.name());
    }
    for sc in &pack.sidecars {
        println!(
            "  sidecar {}: {} tensors, {} B — {}",
            sc.file, sc.tensors, sc.bytes, sc.note
        );
    }
    if pack.census.total_tensors == 304_240 {
        let mm = pack.census.manifest_mismatches();
        if mm.is_empty() {
            println!("  MANIFEST: PASS (matches spec §1.3 counts for the known 3.05bpw pack)");
        } else {
            for l in &mm {
                println!("{l}");
            }
            bail!("EXL3 pack {}: census does not match the spec manifest", dir);
        }
    } else {
        println!(
            "  MANIFEST: skipped ({} tensors ≠ the known pack's 304240; manifest is pack-specific)",
            pack.census.total_tensors
        );
    }
    println!(
        "EXL3_CENSUS_OK modules={} bytes={}",
        pack.modules.len(),
        pack.census.total_bytes
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// The decode (G-A3-1 core). Bit-exact vs ext.reconstruct @ 523ecd3.
// ---------------------------------------------------------------------------

/// k_inv = fp16 bits 0x1EEE, k_bias = fp16 bits 0xC931 (codebook.cuh decode_mul1_product_2).
pub const MUL1: u32 = 0x83DCD12D;
pub const MUL1_K_INV_BITS: u16 = 0x1EEE;
pub const MUL1_K_BIAS_BITS: u16 = 0xC931;

/// The 65,536-entry mul1 codebook as raw fp16 bit patterns. Matches
/// `ngram_codec.mul1_codebook` and `decode_3inst<2>` (verified: fused vs
/// double-rounded arithmetic agree on all 65536 entries).
pub fn mul1_codebook() -> Vec<u16> {
    let k_inv = half::f16::from_bits(MUL1_K_INV_BITS).to_f32();
    let k_bias = half::f16::from_bits(MUL1_K_BIAS_BITS).to_f32();
    let mut cb = Vec::with_capacity(65_536);
    for s in 0u64..65_536 {
        let prod = (s.wrapping_mul(MUL1 as u64)) & 0xFFFF_FFFF;
        let bsum = (prod & 255) + ((prod >> 8) & 255) + ((prod >> 16) & 255) + ((prod >> 24) & 255);
        cb.push(half::f16::from_f32((1024.0 + bsum as f32) * k_inv + k_bias).to_bits());
    }
    cb
}

/// Output position (r,c) in a 16x16 block -> temporal index t (0..256).
/// Bijection (unit-tested). Derived from reconstruct_tile's shuffle; the
/// `(r//8)%2` term is the m2/m3 rows (r0+8/r0+9 read frag0[1]).
pub fn tmap(r: usize, c: usize) -> usize {
    let cp = c / 2;
    let rr = (r % 8) / 2;
    let l = 8 * (cp % 4) + rr;
    8 * l + (r % 2) + 2 * ((r / 8) % 2) + 4 * (c / 8) + 32 * (c % 2)
}

/// Full 256-entry tmap row-major over (r, c).
pub fn tmap_table() -> [[usize; 16]; 16] {
    let mut t = [[0usize; 16]; 16];
    for r in 0..16 {
        for c in 0..16 {
            t[r][c] = tmap(r, c);
        }
    }
    t
}

/// Decode one 16x16 block. `ring` = the block's 16*bits int16 words viewed as
/// 8*bits little-endian uint32s (out[t] = fp16 bit pattern of value at t).
pub fn decode_block(ring: &[u32], bits: usize, cb: &[u16], out: &mut [u16; 256]) {
    let w_words = bits * 8;
    debug_assert_eq!(ring.len(), w_words);
    for t in 0..256 {
        let b1 = (t + 257) * bits; // window END (unwrapped bits)
        let b0 = b1 - 16; // window START
        let i0 = b0 / 32;
        let i1 = (b1 - 1) / 32;
        let s = ((i1 + 1) * 32) - b1; // funnel-shift amount, 0..=31
        debug_assert!(s <= 31);
        let a = ring[i0 % w_words] as u64;
        let b = ring[i1 % w_words] as u64;
        let idx = if s == 0 {
            b & 0xFFFF // funnelshift clamp: low word, low half
        } else {
            ((b >> s) | ((a << (32 - s)) & 0xFFFF_FFFF)) & 0xFFFF
        };
        out[t] = cb[idx as usize];
    }
}

/// Decode a whole trellis tensor [kb, nb, 16*bits] int16 into W [kb*16, nb*16]
/// fp16 bit patterns (row-major). This is `ext.reconstruct` pre-suh/svh.
pub fn decode_trellis_host(trellis: &[i16], kb: usize, nb: usize, bits: usize) -> Vec<u16> {
    let cb = mul1_codebook();
    let ring = trellis_to_ring(trellis, bits);
    let mut out = vec![0u16; kb * 16 * nb * 16];
    let mut block = [0u16; 256];
    let tmap = tmap_table();
    for k in 0..kb {
        for n in 0..nb {
            let b = k * nb + n;
            decode_block(&ring[b * bits * 8..(b + 1) * bits * 8], bits, &cb, &mut block);
            for r in 0..16 {
                let orow = (k * 16 + r) * (nb * 16);
                for c in 0..16 {
                    out[orow + n * 16 + c] = block[tmap[r][c]];
                }
            }
        }
    }
    out
}

/// reinterpret [.., 16*bits] int16 as [.., 8*bits] LE uint32 per block.
fn trellis_to_ring(trellis: &[i16], bits: usize) -> Vec<u32> {
    let words = bits * 8;
    let nblocks = trellis.len() / (16 * bits);
    let mut ring = Vec::with_capacity(nblocks * words);
    for b in 0..nblocks {
        let base = b * 16 * bits;
        for j in 0..words {
            let lo = trellis[base + 2 * j] as u16 as u32;
            let hi = trellis[base + 2 * j + 1] as u16 as u32;
            ring.push(lo | (hi << 16));
        }
    }
    ring
}

// ---------------------------------------------------------------------------
// G-A3-1 oracle: our Rust decode vs pinned ext.reconstruct dumps, bitwise.
// ---------------------------------------------------------------------------

/// `--probe-exl3-oracle`: decode every module in the reference manifest with OUR
/// decode and compare BITWISE against pinned `ext.reconstruct` outputs (generated
/// on .13 by scripts/a3b/a3b_gen_ref.py from reference @ 523ecd3). Whole matrices
/// only — every block of every dumped matrix is checked. Acceptance: 0 mismatched
/// fp16 patterns anywhere, ≥1e6 blocks per sampled bits class (bits=4 is a full
/// census of the pack's class and may fall short of 1e6 — reported as such).
pub fn probe_oracle(dir: &str, ref_dir: &str) -> anyhow::Result<()> {
    #[derive(serde::Deserialize)]
    struct RefEntry {
        module: String,
        bits: usize,
        k: usize,
        n: usize,
        blocks: usize,
        file: String,
    }
    let mpath = format!("{}/manifest.json", ref_dir.trim_end_matches('/'));
    let mraw = std::fs::read_to_string(&mpath).with_context(|| {
        format!(
            "oracle: {mpath} missing — generate it on .13 with \
             scripts/a3b/a3b_gen_ref.py (reference @ 523ecd3)"
        )
    })?;
    let entries: Vec<RefEntry> =
        serde_json::from_str(&mraw).with_context(|| "oracle: manifest unparsable")?;
    let pack = Exl3Pack::open(dir)?;
    let by_module: BTreeMap<&str, &ModuleMeta> =
        pack.modules.iter().map(|m| (m.name.as_str(), m)).collect();

    // bits -> [blocks, bad_values, bad_blocks, matrices, gbps_sum, trellis_bytes]
    let mut per_bits: BTreeMap<usize, [f64; 6]> = BTreeMap::new();
    let t_all = std::time::Instant::now();
    for (i, e) in entries.iter().enumerate() {
        let meta = by_module.get(e.module.as_str())
            .with_context(|| format!("oracle: module {} not in pack", e.module))?;
        if meta.bits != e.bits {
            bail!("oracle: bits mismatch on {}: pack={}, ref={}", e.module, meta.bits, e.bits);
        }
        if meta.k != e.k || meta.n != e.n {
            bail!("oracle: shape mismatch on {}: pack={}x{} ref={}x{}", e.module, meta.k, meta.n, e.k, e.n);
        }
        let tr_bytes = meta.trellis.read_bytes(dir)?;
        let mut trellis = vec![0i16; tr_bytes.len() / 2];
        for (j, c) in tr_bytes.chunks_exact(2).enumerate() {
            trellis[j] = i16::from_le_bytes([c[0], c[1]]);
        }
        let t0 = std::time::Instant::now();
        let ours = decode_trellis_host(&trellis, meta.k / 16, meta.n / 16, meta.bits);
        let dt = t0.elapsed().as_secs_f64();
        let ref_bytes = std::fs::read(format!("{}/{}", ref_dir.trim_end_matches('/'), e.file))
            .with_context(|| format!("oracle: ref dump {}", e.file))?;
        if ref_bytes.len() != ours.len() * 2 {
            bail!(
                "oracle: {} ref dump {} B != our decode {} B",
                e.module,
                ref_bytes.len(),
                ours.len() * 2
            );
        }
        let mut bad_values = 0usize;
        let mut bad_blocks = 0usize;
        for (bi, chunk) in ref_bytes.chunks_exact(512).enumerate() {
            let mut block_bad = 0usize;
            for (j, pair) in chunk.chunks_exact(2).enumerate() {
                let r = u16::from_le_bytes([pair[0], pair[1]]);
                if ours[bi * 256 + j] != r {
                    bad_values += 1;
                    block_bad += 1;
                }
            }
            if block_bad > 0 {
                bad_blocks += 1;
            }
        }
        let gbps = tr_bytes.len() as f64 / dt / 1e9;
        let s = per_bits.entry(e.bits).or_insert([0.0; 6]);
        s[0] += e.blocks as f64;
        s[1] += bad_values as f64;
        s[2] += bad_blocks as f64;
        s[3] += 1.0;
        s[4] += gbps;
        s[5] += tr_bytes.len() as f64;
        if bad_values != 0 || i % 25 == 0 || e.blocks > 100_000 {
            println!(
                "ORACLE {} {} bits={} blocks={} bad_values={bad_values} bad_blocks={bad_blocks} decode={dt:.3}s ({gbps:.1} GB/s trellis)",
                if bad_values == 0 { "OK " } else { "BAD" },
                e.module,
                e.bits,
                e.blocks
            );
        }
    }
    let mut total_blocks = 0f64;
    let mut total_bad = 0f64;
    for (bits, s) in &per_bits {
        let kind = if *bits == 4 { "CENSUS(pack class)" } else { "SAMPLED" };
        let target_met = s[0] >= 1_000_000.0;
        println!(
            "CLASS bits={bits}: {kind} blocks={:.0} bad_values={:.0} bad_blocks={:.0} matrices={:.0} avg_gb_s={:.1} -> {}",
            s[0],
            s[1],
            s[2],
            s[3],
            s[4] / s[3].max(1.0),
            if s[1] == 0.0 && (target_met || *bits == 4) { "OK" } else if s[1] == 0.0 { "OK_SHORTFALL" } else { "FAIL" }
        );
        total_blocks += s[0];
        total_bad += s[1];
    }
    let ok = total_bad == 0.0 && total_blocks > 0.0;
    println!(
        "ORACLE_VERDICT {} blocks={:.0} bad_values={:.0} wall={:.1}s",
        if ok { "ORACLE_OK" } else { "ORACLE_FAIL" },
        total_blocks,
        total_bad,
        t_all.elapsed().as_secs_f64()
    );
    if !ok {
        bail!("oracle: decode is NOT bit-exact — STOP (a near-exact decode is the most dangerous artifact this programme can produce)");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tmap_is_a_bijection() {
        let t = tmap_table();
        let mut seen = vec![false; 256];
        for r in t.iter() {
            for v in r.iter() {
                assert!(*v < 256, "tmap value {v} out of range");
                assert!(!seen[*v], "tmap value {v} repeated — not a bijection");
                seen[*v] = true;
            }
        }
        assert!(seen.iter().all(|s| *s));
    }

    #[test]
    fn window_math_invariants() {
        for bits in 1..=8usize {
            for t in 0..256usize {
                let b1 = (t + 257) * bits;
                let b0 = b1 - 16;
                let i0 = b0 / 32;
                let i1 = (b1 - 1) / 32;
                let s = ((i1 + 1) * 32) - b1;
                assert!(s <= 31, "bits={bits} t={t}: shift {s} out of range");
                assert!(i1 >= i0);
                // window spans at most two words
                assert!(i1 - i0 <= 1, "bits={bits} t={t}: span {} words", i1 - i0);
            }
        }
    }

    #[test]
    fn codebook_shape_and_finiteness() {
        let cb = mul1_codebook();
        assert_eq!(cb.len(), 65_536);
        for (i, bits) in cb.iter().enumerate() {
            let v = half::f16::from_bits(*bits);
            assert!(v.is_finite(), "LUT[{i}] not finite");
            assert!(v.to_f32().abs() < 4.0, "LUT[{i}] = {v} out of expected range");
        }
        // idx 0: prod = 0, bsum = 0 -> fp16(1024*k_inv + k_bias)
        let k_inv = half::f16::from_bits(MUL1_K_INV_BITS).to_f32();
        let k_bias = half::f16::from_bits(MUL1_K_BIAS_BITS).to_f32();
        assert_eq!(cb[0], half::f16::from_f32(1024.0 * k_inv + k_bias).to_bits());
    }

    #[test]
    fn decode_block_all_zero_ring_decodes_to_cb0() {
        let cb = mul1_codebook();
        let ring = vec![0u32; 24]; // bits=3
        let mut out = [0u16; 256];
        decode_block(&ring, 3, &cb, &mut out);
        assert!(out.iter().all(|v| *v == cb[0]));
    }
}
