//! The provenance-stamped table file format (`Table`, `Decision`, `Fingerprint`, `Posture`) and the fingerprint inputs (model, GPU, driver, PTX).
//! See `mod.rs` for the file map.

use super::*;

// ---------------------------------------------------------------------------------------------
// Table file format (§7.1) + fingerprint (§7.2)
// ---------------------------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct ModelFp {
    pub config_sha256: String,
    pub quant_sha256: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct GpuFp {
    pub name: String,
    pub cc: String,
    pub sms: i32,
    pub l2_bytes: i64,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct Fingerprint {
    pub tune_build_id: String,
    /// sha256 of src/ptx/<stem>.ptx as read from CWD (the PTX this process loads, or the PTX the
    /// shipped fatbin was compiled from)
    pub ptx_sha256: BTreeMap<String, String>,
    /// sha256 of src/ptx/<stem>.fatbin when present (CUBIN deploys); provenance + match when both sides have it
    #[serde(default)]
    pub fatbin_sha256: BTreeMap<String, String>,
    pub registry_schema: String,
    pub model: ModelFp,
    pub gpu: GpuFp,
    pub driver: String,
    pub cuda: String,
    /// tensor-parallel degree the table was tuned at (per-rank shapes change with TP)
    pub tp: u32,
    /// SM-clock profile (MHz, rounded to 100) the table was tuned at
    pub clock_mhz: u32,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct Posture {
    pub lanes: usize,
    pub mtp_max_k: usize,
    pub dds: bool,
    pub kv_fmt: u32,
    pub max_pos: usize,
    pub prefill_chunk: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct DecScope {
    #[serde(default, skip_serializing_if = "Option::is_none")] pub fam: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")] pub m: Option<Vec<u16>>,
    #[serde(default, skip_serializing_if = "Option::is_none")] pub regime: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")] pub k: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")] pub n: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")] pub bits: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")] pub chunk: Option<Vec<u8>>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Decision {
    pub id: String,
    pub rev: u32,
    pub class: String,
    #[serde(default)] pub scope: DecScope,
    pub value: i32,
    #[serde(default)] pub default: Option<i32>,
    #[serde(default)] pub evidence: serde_json::Value,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Table {
    pub format: String,
    /// "shipped" (build_stable.sh) | "user" (a local `--autotune`; boot prefers it)
    #[serde(default = "origin_default")] pub origin: String,
    pub fingerprint: Fingerprint,
    pub posture: Posture,
    #[serde(default)] pub provenance: serde_json::Value,
    #[serde(default)] pub decisions: Vec<Decision>,
    #[serde(default)] pub families: serde_json::Value,
    #[serde(default)] pub confirm: serde_json::Value,
    /// sha256 of the table serialized with this field empty (`seal`); "" = unsealed (hand-written).
    /// A sealed table whose content no longer matches is refused at boot (tamper / hand edit).
    #[serde(default)] pub table_sha256: String,
}
fn origin_default() -> String { "shipped".into() }

impl Table {
    /// The sha256 a sealed table must carry: over its pretty JSON with `table_sha256` = "".
    pub fn content_sha256(&self) -> String {
        let mut t = self.clone();
        t.table_sha256 = String::new();
        sha256_hex(serde_json::to_string_pretty(&t).unwrap_or_default().as_bytes())
    }
    /// Stamp `table_sha256` (the writer's last step).
    pub fn seal(&mut self) { self.table_sha256 = self.content_sha256(); }
    /// Ok for an unsealed table or a seal that matches the content.
    pub fn seal_ok(&self) -> bool { self.table_sha256.is_empty() || self.table_sha256 == self.content_sha256() }
}

pub fn sha256_hex(b: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(b);
    h.finalize().iter().map(|x| format!("{x:02x}")).collect()
}

fn sha256_file(p: &Path) -> Option<String> { std::fs::read(p).ok().map(|b| sha256_hex(&b)) }

/// sha256 over the canonical registry description (id|class|scope|fams|domain|default|rev per entry).
pub fn registry_schema() -> String {
    let mut s = String::from(TABLE_FORMAT);
    for t in REGISTRY {
        s += &format!("\n{}|{}|{}|{:?}|{:?}|{}|{}", t.id, t.class.name(), t.scope.name(),
                      t.fams.iter().map(|f| f.name()).collect::<Vec<_>>(), t.domain, t.default, t.rev);
    }
    sha256_hex(s.as_bytes())
}

/// The kernel modules the EXL3 engine loads (the fingerprint's PTX set).
pub const PTX_STEMS: [&str; 2] = ["exl3_bench", "gpu_batch"];

/// GPU identity via the driver API (cuInit is idempotent). Unqueryable fields stay empty / -1.
pub fn gpu_identity() -> (GpuFp, String) {
    use cudarc::driver::sys;
    let mut g = GpuFp { sms: -1, l2_bytes: -1, ..Default::default() };
    let mut cuda = String::new();
    unsafe {
        if sys::cuInit(0) != sys::CUresult::CUDA_SUCCESS { return (g, cuda); }
        let mut d: sys::CUdevice = 0;
        if sys::cuDeviceGet(&mut d, 0) != sys::CUresult::CUDA_SUCCESS { return (g, cuda); }
        let mut buf = [0 as std::ffi::c_char; 256];
        if sys::cuDeviceGetName(buf.as_mut_ptr(), 255, d) == sys::CUresult::CUDA_SUCCESS {
            g.name = std::ffi::CStr::from_ptr(buf.as_ptr()).to_string_lossy().trim().to_string();
        }
        let attr = |a: sys::CUdevice_attribute| -> i64 {
            let mut v: std::ffi::c_int = -1;
            if sys::cuDeviceGetAttribute(&mut v, a, d) == sys::CUresult::CUDA_SUCCESS { v as i64 } else { -1 }
        };
        let (maj, min) = (attr(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
                          attr(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR));
        g.cc = format!("{maj}.{min}");
        g.sms = attr(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) as i32;
        g.l2_bytes = attr(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE);
        let mut v: std::ffi::c_int = 0;
        if sys::cuDriverGetVersion(&mut v) == sys::CUresult::CUDA_SUCCESS {
            cuda = format!("{}.{}", v / 1000, (v % 1000) / 10);
        }
    }
    (g, cuda)
}

/// The NVIDIA kernel-module version ("580.173.02") from /proc (no process spawn), or "".
pub fn driver_version() -> String {
    std::fs::read_to_string("/proc/driver/nvidia/version").ok()
        .and_then(|s| s.lines().next().map(|l| l.to_string()))
        .and_then(|l| l.split_whitespace()
            .find(|w| w.chars().filter(|c| *c == '.').count() >= 1
                && w.chars().all(|c| c.is_ascii_digit() || c == '.')
                && w.chars().next().map_or(false, |c| c.is_ascii_digit()))
            .map(|w| w.to_string()))
        .unwrap_or_default()
}

/// This process's fingerprint (the EMBEDDED PTX modules, the on-disk fatbin when one is present,
/// and the pack's config files). FIX1 #78: the PTX hashes come from `ptx_embed` — the bytes the
/// engine actually loads — never from src/ptx on disk, so a binary-alone EMB1 deploy still
/// computes them and the tune-table fast path still hits.
pub fn current_fingerprint(model_dir: &str, tp: u32, clock_mhz: u32) -> Fingerprint {
    let d = model_dir.trim_end_matches('/');
    let mut ptx = BTreeMap::new();
    let mut fatbin = BTreeMap::new();
    // FIX1 #78 (EMB1): hash the PTX the binary EMBEDS, not src/ptx files on disk. A deploy is
    // ONE file (EMB1): on a binary-alone box src/ptx/ does not exist, the disk read came back
    // empty, and strict_match then rejected every tune table on ptx_sha256 — the boot silently
    // fell back to the built-in defaults (the slow path) on hardware that HAD a valid table.
    // The embedded bytes are the modules the engine loads, so this is the fingerprint that was
    // always meant; it is CWD-independent and, on a source tree of the same build, byte-identical
    // to the old disk hash (include_str! reads the very file build.rs emitted).
    let embedded: [(&str, &str); 2] = [("exl3_bench", crate::ptx_embed::EXL3_BENCH),
                                       ("gpu_batch", crate::ptx_embed::GPU_BATCH)];
    for (stem, src) in embedded {
        ptx.insert(stem.to_string(), sha256_hex(src.as_bytes()));
    }
    for stem in PTX_STEMS {
        if let Some(h) = sha256_file(Path::new(&format!("src/ptx/{stem}.fatbin"))) { fatbin.insert(stem.to_string(), h); }
    }
    let (gpu, cuda) = gpu_identity();
    Fingerprint {
        tune_build_id: env!("TUNE_BUILD_ID").to_string(),
        ptx_sha256: ptx,
        fatbin_sha256: fatbin,
        registry_schema: registry_schema(),
        model: ModelFp {
            config_sha256: sha256_file(Path::new(&format!("{d}/config.json"))).unwrap_or_default(),
            quant_sha256: sha256_file(Path::new(&format!("{d}/quantization_config.json"))).unwrap_or_default(),
        },
        gpu,
        driver: driver_version(),
        cuda,
        tp,
        clock_mhz,
    }
}

/// Strict fingerprint + posture match (clock and driver excluded — handled by the caller).
/// Err = the first mismatching field (the FALLBACK reason).
pub fn strict_match(want: &Fingerprint, have: &Fingerprint, want_p: &Posture, have_p: &Posture)
    -> Result<(), String> {
    if want.tune_build_id != have.tune_build_id { return Err("tune_build_id".into()); }
    if want.ptx_sha256 != have.ptx_sha256 { return Err("ptx_sha256".into()); }
    // a fatbin present on both sides must match (a PTX-only table still loads on a fatbin deploy)
    for (k, v) in &want.fatbin_sha256 {
        if let Some(h) = have.fatbin_sha256.get(k) { if h != v { return Err(format!("fatbin_sha256[{k}]")); } }
    }
    if want.registry_schema != have.registry_schema { return Err("registry_schema".into()); }
    if want.model != have.model { return Err("model".into()); }
    if want.gpu.name != have.gpu.name || want.gpu.cc != have.gpu.cc || want.gpu.sms != have.gpu.sms {
        return Err("gpu".into());
    }
    if want.tp != have.tp { return Err("tp".into()); }
    if want_p != have_p { return Err("posture".into()); }
    Ok(())
}
