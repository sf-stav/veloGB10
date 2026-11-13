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
    /// Half-bit units per weight (= trellis shape[-1] / 8 = 2*K, or 2*K+1 for the
    /// fractional widths K = 1.5/2.5/3.5). The reference's own switch index (k2_from_K,
    /// exllamav3 v1.6.3 bits_k.cuh). Integer K = k2/2; k2 & 1 is the half flag.
    pub k2: usize,
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
/// K35: an expert module (routed `layers.N.mlp.experts.E.*` or the MTP head's `mtp.layers.0.mlp.experts.E.*`) is
/// served by the EXPERT kernel family whatever its TensorClass says (the head's experts are classed Mtp).
pub fn is_expert_module(name: &str) -> bool { name.contains(".mlp.experts.") }

/// K35: can this build serve a module of half-bit width `k2` under `name`? Expert modules against the expert
/// bodies, everything else against the dense bodies — the one predicate the MTP pre-scan and the catalog audit use.
pub fn width_supported(name: &str, k2: usize) -> bool {
    if is_expert_module(name) { crate::exl3_forward::expert_k2_supported(k2) } else { crate::exl3_forward::quad_k2_supported(k2) }
}

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
        // EXL27 (V074): Qwen35 (Qwen3.8-27B dense hybrid) is served too — the EXL3-side
        // plumbing is family-generic: dense MLP rides the 1-expert MoE fiction, the plain
        // residual recipe rides hc_count=1/hc_lowrank=0 (qwen.rs already sets both).
        if !matches!(cfg.family, crate::qwen::Family::Qwen4Exp | crate::qwen::Family::Qwen35) {
            bail!(
                "EXL3 pack {dir}: config.json declares family {:?} — this EXL3 loader only \
                 serves the qwen4_exp / qwen3_5 families (the wrong-model hazard). Refusing.",
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
        // 4b. MTP draft-sidecar shadow (HermiHg mtp-*.safetensors): a sidecar dropped into
        // the pack dir carries REPLACEMENT tensors for the pack's MTP head (same tensor
        // names, its own bit widths). Redirect those reads to the sidecar before any
        // module loads; only names already present in the pack are shadowed (a name
        // mismatch shows up as redirected < sidecar tensors). Several sidecars: the
        // lexicographically LAST wins (mtp- sorts after model-*, vendor shadow order).
        // Note: "mtp_hyper_connection_mixer_patch.safetensors" starts "mtp_" (underscore)
        // and is deliberately NOT matched.
        let mut sc_files: Vec<String> = std::fs::read_dir(&dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|f| f.starts_with("mtp-") && f.ends_with(".safetensors"))
            .collect();
        sc_files.sort();
        if !sc_files.is_empty() {
            let fname = sc_files[sc_files.len() - 1].clone();
            let hdr = shard_header(&format!("{dir}/{fname}"))
                .with_context(|| format!("EXL3 pack {dir}: MTP sidecar {fname}"))?;
            let mut n = 0usize;
            for (name, e) in &hdr {
                if shapes.contains_key(name) {
                    shapes.insert(name.clone(), (fname.clone(), e.shape.clone(), e.dtype.clone(), e.n_bytes, e.offset));
                    n += 1;
                }
            }
            println!("[exl3] MTP sidecar shadow: {fname} — {n} of {} sidecar tensor(s) redirected", hdr.len());
        }
        // 4c. MTP mixer-patch fallback: the h4/h5-branch packs ship the MTP head's mixer
        // tensors ONLY in mtp_hyper_connection_mixer_patch.safetensors (no main-shard
        // copies). Spec 6.3 keeps main-shard copies preferred, so insert-if-absent only.
        let mp = format!("{dir}/mtp_hyper_connection_mixer_patch.safetensors");
        if std::path::Path::new(&mp).exists() {
            let hdr = shard_header(&mp)
                .with_context(|| format!("EXL3 pack {dir}: mixer patch sidecar"))?;
            let mut n = 0usize;
            for (name, e) in &hdr {
                if !shapes.contains_key(name) {
                    shapes.insert(name.clone(), ("mtp_hyper_connection_mixer_patch.safetensors".to_string(), e.shape.clone(), e.dtype.clone(), e.n_bytes, e.offset));
                    n += 1;
                }
            }
            if n > 0 { println!("[exl3] MTP mixer patch: {n} missing tensor(s) filled from sidecar"); }
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
        // TP2BIT5 #26 Fix 4 pre-scan: if ANY mtp.* trellis module has an unsupported k2
        // (Pack A: mtp.fc_embedding K=8/k2 16), skip the WHOLE draft head and serve without
        // speculation — one line at load, a counter on /metrics, never a boot abort.
        // K35 (2026-10-31): the width a module can serve is decided per KERNEL FAMILY — routed and
        // MTP-head experts (`.mlp.experts.`) by the expert bodies, everything else by the dense
        // bodies (width_supported below; the pre-scan used the dense set for the head's experts,
        // which would have skipped the 3.68 / 2.68 packs' K = 3.5 / 2.5 draft experts and admitted
        // a K = 7 draft expert without a body).
        let skip_mtp_head = shapes
            .iter()
            .filter(|(n, (_, shape, _, _, _))| {
                n.starts_with("mtp.") && n.ends_with(".trellis") && shape.len() == 3
            })
            .filter_map(|(n, (_, shape, _, _, _))| {
                k2_from_words(shape[2]).ok().map(|k2| (n, k2))
            })
            .find(|(n, k2)| !width_supported(n, *k2))
            .map(|(n, k2)| {
                crate::metrics::exl3_mtp_skipped();
                eprintln!(
                    "[exl3] #26 Fix 4: MTP draft head SKIPPED — {n}: K={} (k2 {k2}) has no {} kernel \
in this build ({}); serving WITHOUT speculation (plain decode; this pack cannot use MTP until that width lands).",
                    k2 as f32 / 2.0,
                    if is_expert_module(n) { "expert" } else { "dense" },
                    crate::exl3_forward::width_support_text()
                );
                true
            })
            .unwrap_or(false);
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
            let k2 = k2_from_words(shape[2]).map_err(|e| anyhow::anyhow!("EXL3 pack {dir}: {name}: {e}"))?;
            let class = classify(&module);
            // TP2BIT5 #26 Fix 4: a draft head wider than the served set is SKIPPED WHOLE (the
            // loader addresses mtp.layers.0.* directly, so a partial head would fail the boot
            // later); the model then serves WITHOUT speculation. §1b-bis.8: no refusal.
            if skip_mtp_head && matches!(class, TensorClass::Mtp) {
                continue;
            }
            // Capability audit (S-B9-REL-INTEG D1 class: a width whose kernel body does not exist must
            // REFUSE here, never reach a `default: break` switch that writes nothing = garbage with a
            // 200). Per kernel family (K35): expert modules — routed and MTP-head — against the expert
            // bodies (expert_k2_supported), every other module against the dense bodies
            // (quad_k2_supported). The text names what IS served, from the predicates themselves
            // (roadmap ISSUES 183: the old hand-typed line said "experts serve K=3..=6" after K=1.5 and
            // K=2 experts were admitted). This is a would-serve-wrong-output case: the §1b-bis.8
            // honest-refusal carve-out.
            if !width_supported(name, k2) {
                let expert = is_expert_module(name);
                bail!(
                    "EXL3 pack {dir}: {name}: K={} (k2 {k2}) {} modules have no decode kernel in this build. {}. \
Serve a pack whose {} modules are in that set.",
                    k2 as f32 / 2.0,
                    if expert { "expert" } else { "dense" },
                    crate::exl3_forward::width_support_text(),
                    if expert { "expert" } else { "dense" },
                );
            }
            *class_counts.entry(class).or_insert(0) += 1;
            modules.push(ModuleMeta {
                name: module,
                k2,
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
/// Half-bit units from a trellis last dim: 16*K uint16 words = 8*k2. Mirrors the reference
/// `bits_from_K` (exllamav3 v1.6.3 bits_k.cuh): integer K 1..=8, or the fractional widths
/// 1.5 / 2.5 / 3.5 (mul1 codebook only). K=0.5 and fractional K above 3.5 do not exist.
pub fn k2_from_words(words: usize) -> Result<usize, String> {
    if words % 8 != 0 {
        return Err(format!("trellis last dim {} is not 8*k2 (16*K words, K integer or .5)", words));
    }
    let k2 = words / 8;
    let half = k2 % 2 == 1;
    if k2 < 2 || k2 > 16 || (half && k2 > 7) {
        return Err(format!("k2 {} is outside the format (K 1..=8 integer, or 1.5/2.5/3.5)", k2));
    }
    Ok(k2)
}

/// k2 from the pack per-module bits_per_weight float.
pub fn k2_from_bpw(k: f64) -> f64 {
    2.0 * k
}

pub fn decode_block(ring: &[u32], k2: usize, cb: &[u16], out: &mut [u16; 256]) {
    if k2 % 2 == 1 {
        decode_block_half(ring, k2 / 2, cb, out);
        return;
    }
    let bits = k2 / 2;
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

/// Fractional K = KA + 0.5 host mirror of the reference `dq8_half` (exllamav3 v1.6.3
/// exl3_dq.cuh), transcribed literally: weight PAIRS consume bits2 = 2*KA+1 bits;
/// the eight 16-bit windows of a group are fetched from two 64-bit merges (e7/e3),
/// windows alternating (KA+1, KA) spacing; each window decodes through the same mul1
/// 3-instruction codebook as integer K (the full 16-bit pattern is the codeword).
/// Weight t0+k of the group maps to window w_k. Validated against the reference
/// reconstruct oracle (W6); the integer path above is the G-A3-1-validated original.
fn decode_block_half(ring: &[u32], ka: usize, cb: &[u16], out: &mut [u16; 256]) {
    let bits2 = 2 * ka + 1;
    let words = 4 * bits2;
    let gspan = 18 + 3 * ka; // bits spanned by four windows (incl. the 16-bit fetch of the top one)
    debug_assert_eq!(ring.len(), words);
    let mut w = [0u32; 8];
    for t0 in (0..256).step_by(8) {
        let e7 = ((t0 / 2) + 4) * bits2 + 128 * bits2;
        let e3 = e7 - 2 * bits2;
        for &(e, top) in &[(e7, 7usize), (e3, 3usize)] {
            let lo = (e - gspan) / 32;
            let hi = (e - 1) / 32;
            let s = (hi + 1) * 32 - e;
            debug_assert!(s <= 31);
            let a = ring[lo % words] as u64;
            let b = ring[hi % words] as u64;
            let merged: u64 = if s == 0 {
                b
            } else {
                (b >> s) | ((a << (32 - s)) & 0xFFFF_FFFF)
            };
            w[top] = merged as u32;
            w[top - 1] = w[top] >> (ka + 1);
            w[top - 2] = w[top - 1] >> ka;
            w[top - 3] = w[top - 2] >> (ka + 1);
        }
        for k in 0..8 {
            out[t0 + k] = cb[(w[k] & 0xFFFF) as usize];
        }
    }
}

/// Decode a whole trellis tensor [kb, nb, 8*k2] int16 into W [kb*16, nb*16]
/// fp16 bit patterns (row-major). This is `ext.reconstruct` pre-suh/svh.
pub fn decode_trellis_host(trellis: &[i16], kb: usize, nb: usize, k2: usize) -> Vec<u16> {
    let cb = mul1_codebook();
    let ring = trellis_to_ring(trellis, k2);
    let mut out = vec![0u16; kb * 16 * nb * 16];
    let mut block = [0u16; 256];
    let tmap = tmap_table();
    for k in 0..kb {
        for n in 0..nb {
            let b = k * nb + n;
            decode_block(&ring[b * 4 * k2..(b + 1) * 4 * k2], k2, &cb, &mut block);
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
fn trellis_to_ring(trellis: &[i16], k2: usize) -> Vec<u32> {
    let words = 4 * k2; // uint32 per 16x16 tile: 8*k2 uint16
    let nblocks = trellis.len() / (8 * k2);
    let mut ring = Vec::with_capacity(nblocks * words);
    for b in 0..nblocks {
        let base = b * 8 * k2;
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
        if meta.k2 != 2 * e.bits {
            bail!("oracle: bits mismatch on {}: pack={}, ref={}", e.module, meta.k2, 2 * e.bits);
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
        let ours = decode_trellis_host(&trellis, meta.k / 16, meta.n / 16, meta.k2);
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
        let s = per_bits.entry(e.bits).or_insert([0.0; 6]); // report keyed on integer bits (ref manifest)
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
    fn k2_from_words_matches_reference_rates() {
        // integer K: 16*K words -> k2 = 2*K
        for (words, want) in [(16usize, 2usize), (32, 4), (48, 6), (64, 8), (80, 10), (96, 12), (112, 14), (128, 16)] {
            assert_eq!(k2_from_words(words).unwrap(), want, "words={words}");
        }
        // fractional K: 24 -> 1.5, 40 -> 2.5, 56 -> 3.5
        for (words, want) in [(24usize, 3usize), (40, 5), (56, 7)] {
            assert_eq!(k2_from_words(words).unwrap(), want, "frac words={words}");
        }
        // refusals: not 8-aligned, K=0.5, fractional above 3.5, above K=8
        assert!(k2_from_words(12).is_err());
        assert!(k2_from_words(20).is_err());
        assert!(k2_from_words(8).is_err()); // k2=1 -> K=0.5
        assert!(k2_from_words(88).is_err()); // k2=11 -> K=5.5
        assert!(k2_from_words(136).is_err()); // k2=17 -> K=8.5
        assert!(k2_from_words(144).is_err()); // k2=18 -> K=9
    }

    /// PR#27-D4 (2026-10-30): loader trust boundary. Width capability is decided from the
    /// ACTUAL TRELLIS HEADER (k2_from_words on the tensor's ring-word shape), never from the
    /// pack's advisory tensor_storage.bits_per_weight float (k2_from_bpw has no callers —
    /// advisory only). The two derivations must AGREE for every legal rate, so an advisory
    /// lie can never silently select a different kernel than the header says.
    #[test]
    fn advisory_bpw_agrees_with_header_k2() {
        for k2 in [2usize, 3, 4, 6, 8, 10, 12, 14, 16] {
            let header = k2_from_words(8 * k2).unwrap(); // 16*K ring words
            let advisory = k2_from_bpw(k2 as f64 / 2.0) as usize; // bits_per_weight float
            assert_eq!(header, advisory, "k2={k2}: header {header} != advisory {advisory}");
        }
    }

    #[test]
    fn decode_block_half_zero_ring_decodes_to_cb0_and_wraps() {
        // K=1.5 (ka=1): 12 uint32 per tile. All-zero ring: every window fetch is 0 -> cb[0]
        // everywhere, and the e7/e3 window math + ring modulo must not panic for any group.
        let cb = mul1_codebook();
        for k2 in [3usize, 5, 7] {
            let ring = vec![0u32; 4 * k2];
            let mut out = [0u16; 256];
            decode_block(&ring, k2, &cb, &mut out);
            assert!(out.iter().all(|&v| v == cb[0]), "k2={k2}");
        }
    }

    #[test]
    fn decode_block_k2_even_routes_to_integer_path() {
        // Regression: the k2-routed integer path is the G-A3-1-validated original.
        let cb = mul1_codebook();
        let ring = vec![0x12345678u32; 8 * 5]; // K=5 tile
        let mut a = [0u16; 256];
        decode_block(&ring, 10, &cb, &mut a);
        // old integer body, inlined for comparison
        let bits = 5usize;
        let w_words = bits * 8;
        let mut b = [0u16; 256];
        for t in 0..256 {
            let b1 = (t + 257) * bits;
            let b0 = b1 - 16;
            let i0 = b0 / 32;
            let i1 = (b1 - 1) / 32;
            let sh = ((i1 + 1) * 32) - b1;
            let x = ring[i0 % w_words] as u64;
            let y = ring[i1 % w_words] as u64;
            let idx = if sh == 0 {
                y & 0xFFFF
            } else {
                ((y >> sh) | ((x << (32 - sh)) & 0xFFFF_FFFF)) & 0xFFFF
            };
            b[t] = cb[idx as usize];
        }
        assert_eq!(a, b);
    }

    fn decode_block_all_zero_ring_decodes_to_cb0() {
        let cb = mul1_codebook();
        let ring = vec![0u32; 24]; // bits=3
        let mut out = [0u16; 256];
        decode_block(&ring, 3, &cb, &mut out);
        assert!(out.iter().all(|v| *v == cb[0]));
    }
}


// -------------------------------------------------------------------------------------------
// LEG-B 166 (2026-10-30): wheel-golden cross-check for the k2=3 (K = 1.5 fractional) decode.
// The 1.66bpw pack's damage was localized to the FIRST k2=3 expert layer (L4) by per-layer
// drift vs the exllamav3 1.6.3 wheel's own taps (tests/fixtures note in PLAN report). This
// test decodes a REAL L4 expert quad (layers.4.mlp.experts.7.up_proj, trellis (160, 40, 24))
// with our host oracle and requires bit-identity with the wheel's own CUDA reconstruct
// (LinearEXL3::get_inner_weight_tensor). Cross-implementation per the standing rule: our
// kernel-vs-oracle checks (probe_kernel_decode) are self-consistency and cannot catch this.
// Fixture: tests/k23_golden_l4e7up.json (wheel-built on .13, /tmp/k23_golden_l4e7up.json).
// -------------------------------------------------------------------------------------------
#[cfg(test)]
mod k23_wheel_golden {
    use super::*;

    #[test]
    fn k23_expert_wheel_golden_inner() {
        let j = std::fs::read_to_string("tests/k23_golden_l4e7up.json").unwrap();
        let v: serde_json::Value = serde_json::from_str(&j).unwrap();
        let kb = v["kb"].as_u64().unwrap() as usize;
        let nb = v["nb"].as_u64().unwrap() as usize;
        let trellis: Vec<i16> = v["trellis_i16"].as_array().unwrap()
            .iter().map(|x| x.as_i64().unwrap() as i16).collect();
        let want: Vec<u16> = v["inner_f16_bits"].as_array().unwrap()
            .iter().map(|x| x.as_i64().unwrap() as u16).collect();
        let got = decode_trellis_host(&trellis, kb, nb, 3);
        assert_eq!(got.len(), want.len(), "decoded size mismatch");
        let nbad = got.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
        assert!(
            nbad == 0,
            "k2=3 inner decode differs from wheel golden: {nbad}/{} values, first diffs {:?}",
            got.len(),
            got.iter().zip(want.iter()).enumerate()
                .filter_map(|(i, (a, b))| if a != b { Some((i, *a, *b)) } else { None })
                .take(8).collect::<Vec<_>>()
        );
    }
}

// -------------------------------------------------------------------------------------------
// K35 (2026-10-31): wheel-golden cross-check for the FRACTIONAL expert widths beyond K = 1.5 —
// K = 3.5 (k2 7; the 3.18 / 3.68 packs) and, when its fixture lands, K = 2.5 (k2 5; 2.68 / 2.18).
// A REAL expert quad of turboderp's self-calibrated pack, decoded by the exllamav3 1.6.3 wheel's
// own CUDA reconstruct (LinearEXL3.get_inner_weight_tensor = ext.reconstruct; generator
// k35_golden.py, see PLAN/K35_EXPERT_REPORT_2026-10-31.md), must equal our host oracle bit for
// bit. Cross-implementation per the standing rule: our kernel-vs-oracle checks are
// self-consistency and cannot catch a window formula both sides share.
// Fixture: tests/k35_golden_<pack>_<quad>.json (meta) + .u16le (trellis words, inner f16 bits
// [k][n], suh f16 bits [k], svh f16 bits [n]; little-endian u16).
// -------------------------------------------------------------------------------------------
#[cfg(test)]
pub(crate) mod k35_wheel_golden {
    use super::*;

    pub struct Golden {
        pub base: String,
        pub k2: usize,
        pub kb: usize,
        pub nb: usize,
        pub trellis: Vec<u16>,
        pub inner: Vec<u16>,
        pub suh: Vec<u16>,
        pub svh: Vec<u16>,
    }

    pub fn load(stem: &str) -> Golden {
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("tests/{stem}.json")).unwrap()).unwrap();
        let raw = std::fs::read(format!("tests/{stem}.u16le")).unwrap();
        let words: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let u = |k: &str| meta[k].as_u64().unwrap() as usize;
        let (k2, kb, nb) = (u("k2"), u("kb"), u("nb"));
        let (nt, ni, ns, nv) = (u("n_trellis"), u("n_inner"), u("n_suh"), u("n_svh"));
        assert_eq!(nt, kb * nb * 8 * k2, "trellis words vs (kb, nb, k2)");
        assert_eq!(ni, kb * 16 * nb * 16);
        assert_eq!(words.len(), nt + ni + ns + nv, "fixture size");
        Golden {
            base: meta["base"].as_str().unwrap().to_string(),
            k2, kb, nb,
            trellis: words[..nt].to_vec(),
            inner: words[nt..nt + ni].to_vec(),
            suh: words[nt + ni..nt + ni + ns].to_vec(),
            svh: words[nt + ni + ns..].to_vec(),
        }
    }

    fn host_decode_matches(stem: &str, want_k2: usize) {
        let g = load(stem);
        assert_eq!(g.k2, want_k2, "{stem}: fixture k2");
        let tr: Vec<i16> = g.trellis.iter().map(|w| *w as i16).collect();
        let got = decode_trellis_host(&tr, g.kb, g.nb, g.k2);
        assert_eq!(got.len(), g.inner.len(), "decoded size mismatch");
        let nbad = got.iter().zip(g.inner.iter()).filter(|(a, b)| a != b).count();
        assert!(
            nbad == 0,
            "{}: k2={} host decode differs from the wheel golden: {nbad}/{} values, first diffs {:?}",
            g.base, g.k2, got.len(),
            got.iter().zip(g.inner.iter()).enumerate()
                .filter_map(|(i, (a, b))| if a != b { Some((i, *a, *b)) } else { None })
                .take(8).collect::<Vec<_>>()
        );
        // the fixture is not degenerate: a real quad has many distinct codes
        // (the mul1 codebook has at most 1021 distinct outputs: (1024 + byte-sum) * k_inv + k_bias)
        let distinct = g.inner.iter().collect::<std::collections::HashSet<_>>().len();
        assert!(distinct > 256, "{stem}: only {distinct} distinct inner values");
    }

    #[test]
    fn k35_318_up_host_decode_vs_wheel() { host_decode_matches("k35_golden_318_l0e7up_proj", 7); }

    #[test]
    fn k35_318_down_host_decode_vs_wheel() { host_decode_matches("k35_golden_318_l0e7down_proj", 7); }

    // K = 2.5 (k2 5): a layer-4 expert quad of the 2.68 pack (layers 4..43 and the MTP head are K = 2.5 there)
    #[test]
    fn k25_268_up_host_decode_vs_wheel() { host_decode_matches("k35_golden_268_l4e7up_proj", 5); }

    #[test]
    fn k25_268_down_host_decode_vs_wheel() { host_decode_matches("k35_golden_268_l4e7down_proj", 5); }
}
