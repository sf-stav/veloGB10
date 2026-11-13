//! FP4L — serving NVIDIA's ModelOpt mixed-precision Qwen3.8-Flash-Next checkpoint
//! (`nvidia/Qwen3.8-Flash-Next-FP4-Local`) through the EXL3 trunk (`exl3_forward::FwdModel`).
//!
//! Design: `PLAN/FP4L_DESIGN.md`. The EXL3 trunk is the stack that serves Qwen3.8-Flash-Next at
//! TP=1/2/4 with lane packing, multi-lane MTP and the prefix cache; FP4L adds three weight kinds
//! to it (MXFP8 dense, NVFP4 routed experts, f16 plain linears) and an NVFP4 PLE row codec. Every
//! piece of this module is host-side (reader, codecs, references, probes); the kernels live in
//! `kernels/gpu_fp4l.cu` (module `gpu_fp4l`) and the dispatch in `exl3_forward/fp4l_fwd.rs`.
pub mod csf;
pub mod pack;
pub mod probe;

/// FP4L stored-precision default for an FP4-Local pack, applied once at a head / single-process entry
/// point BEFORE anything reads the options (it then rides TpConfig to the nodes as a set option, like any
/// flag the user passed): no int8 copies of the hyper-connection mixers (--hc-fp16; the mixers are then served at
/// the precision --fp4l-plain chooses: the stored bf16 bits by default, or f16 copies). The trunk's default for
/// the EXL3 packs is int8 mixers — a re-quantization of stored weights, which this pack is used to avoid; owner
/// 2026-10-29: "no conversions". Only an option the user did NOT set is touched; one boot line says so. The KV
/// cache keeps the trunk's default (q8): an owner decision (D-FP4L-4); f16 KV keeps packed speculative verify only
/// under the ungated opt-in --lane-pack-kv any.
pub fn apply_pack_defaults(dir: &str) {
    if !pack::is_fp4l_dir(std::path::Path::new(dir)) { return; }
    if crate::opts::var(crate::opt!("hc-int8")).is_err() && crate::opts::var(crate::opt!("hc-fp16")).is_err() {
        crate::opts::set(crate::opt!("hc-fp16"), "1");
        crate::rprintln!("[fp4l] FP4-Local pack: hyper-connection mixers at their stored precision, no int8 copies (--hc-fp16 \
                          applied; --hc-int8 re-quantizes them to int8 for ~0.6 GB/step less). KV cache: {} (--kv-cache f16 \
                          keeps the cache unquantized; packed speculative verify then needs --lane-pack-kv any, ungated)",
                         crate::opts::var(crate::opt!("kv-cache")).unwrap_or_else(|_| "q8 (default)".into()));
    }
}

/// FP4L: (device weight bytes, n-gram table bytes, vision tower f32 bytes) of an FP4-Local dir (a whole
/// pack or a B32 shard dir), from the tensor headers — for the serve's pre-load fit check. Device bytes
/// count the csf scale planes DECODED (one byte per 16 weights; the streams are ~52% of that), the tower
/// is excluded (it runs on the head, as f32), the n-gram table is reported apart (RAM or SSD, never device).
pub fn fit_bytes(dir: &std::path::Path) -> Option<(u64, u64, u64)> {
    if !pack::is_fp4l_dir(dir) { return None; }
    let st = pack::StDir::open(dir).ok()?;
    let (mut dev, mut table, mut tower) = (0u64, 0u64, 0u64);
    for (name, r) in &st.t {
        if name.contains("ngram_embedding.shard_") { table += r.len; continue; }
        if name.starts_with("model.visual.") {
            let elems: u64 = r.shape.iter().product::<usize>() as u64;
            tower += 4 * if r.dtype == pack::Dt::U8 { 2 * elems } else { elems };
            continue;
        }
        if name.ends_with(".nvfp4_csf_exceptions") { continue; }
        if name.ends_with(".nvfp4_csf_fixed") {
            if let Ok((rows, cols)) = csf::geometry(&r.shape) { dev += (rows * cols) as u64; }
            continue;
        }
        // bf16 linears are served as f16 (same bytes); everything else as stored
        dev += r.len;
    }
    Some((dev, table, tower))
}

/// The `mtp-*.safetensors` file directly in `dir` — the HermiHg MTP-head sidecar convention (`--fp4l-mtp-sidecar
/// auto`; the shard plan ships it to every TP rank). With several, the lexicographically LAST one (the pick of the
/// EXL3 loader's draft-sidecar shadow, so one directory selects the same file for either format).
pub fn mtp_sidecar_in_dir(dir: &std::path::Path) -> Option<std::path::PathBuf> {
    let mut v: Vec<std::path::PathBuf> = std::fs::read_dir(dir).ok()?.filter_map(|e| e.ok()).map(|e| e.path())
        .filter(|p| p.is_file() && p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("mtp-") && n.ends_with(".safetensors")))
        .collect();
    v.sort();
    v.pop()
}
