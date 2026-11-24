//! Plain (non-trellis) safetensors reader and f16/bf16 conversions used at load, plus the K-split / chain-grid
//! pickers, the GDN tensor-core function lookup and the KV spill-region arithmetic. Pure helpers, no model state.
//! Part of the former single `exl3_forward.rs`; see `mod.rs` for the file map.

use super::*;

// ---------------------------------------------------------------------------
// safetensors plain-tensor reader (non-quad weights)
// ---------------------------------------------------------------------------

pub(super) fn read_hdr(path: &str) -> Result<BTreeMap<String, PlainTensor>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path).with_context(|| format!("open {path}"))?;
    let mut len8 = [0u8; 8];
    f.read_exact(&mut len8)?;
    let n = u64::from_le_bytes(len8) as usize;
    let mut buf = vec![0u8; n];
    f.read_exact(&mut buf)?;
    let v: serde_json::Value = serde_json::from_slice(&buf)?;
    let obj = v.as_object().context("hdr not object")?;
    let mut out = BTreeMap::new();
    for (name, e) in obj {
        if name == "__metadata__" {
            continue;
        }
        let offs = e["data_offsets"].as_array().context("offsets")?;
        out.insert(
            name.clone(),
            PlainTensor {
                dtype: e["dtype"].as_str().unwrap_or("?").to_string(),
                offset: 8 + n + offs[0].as_u64().unwrap() as usize,
                n_bytes: (offs[1].as_u64().unwrap() - offs[0].as_u64().unwrap()) as usize,
                shard: std::path::Path::new(path)
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .to_string(),
                shape: e["shape"].as_array()
                    .map(|a| a.iter().filter_map(|v| v.as_u64().map(|x| x as usize)).collect())
                    .unwrap_or_default(),
            },
        );
    }
    Ok(out)
}

impl PlainTensor {
    pub(super) fn read(&self, dir: &str) -> Result<Vec<u8>> {
        use std::io::{Read, Seek, SeekFrom};
        let path = format!("{}/{}", dir.trim_end_matches('/'), self.shard);
        let mut f = std::fs::File::open(&path).with_context(|| format!("open {path}"))?;
        f.seek(SeekFrom::Start(self.offset as u64))?;
        let mut buf = vec![0u8; self.n_bytes];
        f.read_exact(&mut buf).with_context(|| format!("read {} B @{} of {path}", self.n_bytes, self.offset))?;
        Ok(buf)
    }
}

/// S-A3-i: exact f16 bits -> f32 (indexer layernorms load as f32 for QsaParams).
pub(super) fn h16_to_f32(b: u16) -> f32 {
    let sign = ((b >> 15) as u32) << 31;
    let exp = ((b >> 10) & 0x1F) as u32;
    let man = (b & 0x3FF) as u32;
    let bits = if exp == 0 {
        if man == 0 { sign } else {
            let mut e: i32 = -1; let mut m = man;
            while m & 0x400 == 0 { m <<= 1; e -= 1; }
            sign | (((127 - 15 + e) as u32) << 23) | ((m & 0x3FF) << 13)
        }
    } else if exp == 0x1F { sign | 0x7F80_0000 | (man << 13) }
    else { sign | ((exp + 112) << 23) | (man << 13) };
    f32::from_bits(bits)
}

pub(super) fn to_f16_bits(raw: &[u8], dtype: &str) -> Result<Vec<u16>> {
    match dtype {
        "F16" => Ok(raw
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()),
        "BF16" => {
            // bf16 -> f16: exact for |v| >= fp16 min-normal mantissa class; the only
            // lossy cases are subnormal-class magnitudes (< ~6e-5) where f16 has fewer
            // effective mantissa bits. Those are counted (reported at load), not fatal:
            // they sit 4+ orders under the tensors' operating scale.
            let mut out = Vec::with_capacity(raw.len() / 2);
            for c in raw.chunks_exact(2) {
                let b = u16::from_le_bytes([c[0], c[1]]);
                let val = half::bf16::from_bits(b).to_f32();
                let h = f16::from_f32(val);
                if h.to_f32() != val && val.is_finite() && val.abs() >= 6.0e-5 {
                    bail!("bf16->f16 value loss ({b} -> {val} -> {})", h.to_f32());
                }
                out.push(h.to_bits());
            }
            Ok(out)
        }
        "F32" => {
            // EXL27 (owner scope add 2026-10-16): some packs (e.g. Qwen3.5-9B-exl3-3.5bpw) store
            // their plain (non-trellis) tensors as F32. Convert to the internal f16 with
            // round-to-nearest; every rounded value is counted and reported at load (never silent,
            // §1b-bis 8: support + show, no refusal).
            let mut out = Vec::with_capacity(raw.len() / 4);
            let mut n_round = 0u64;
            for c in raw.chunks_exact(4) {
                let b = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                let val = f32::from_bits(b);
                let h = f16::from_f32(val);
                if h.to_f32() != val && val.is_finite() {
                    n_round += 1;
                }
                out.push(h.to_bits());
            }
            if n_round > 0 {
                crate::rprintln!("[exl3] plain F32 tensor: {n_round} values rounded f32->f16 at load");
            }
            Ok(out)
        }
        other => bail!("unsupported plain dtype {other}"),
    }
}

pub(super) fn le_f16_bits(raw: &[u8]) -> Vec<u16> {
    raw.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

pub(super) fn bf16_to_f32(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(2)
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}

/// EXL27B: a NON-BF16 plain tensor as f32 by its stored dtype (the f32 loaders keep bf16_to_f32 for
/// BF16 — every FN tensor). The Qwen3.5-9B pack stores `linear_attn.A_log` as F32: read as BF16, each
/// float became two values (its low half — noise — then its high half) and every GDN head read the
/// wrong decay (9B perplexity 28.2 vs 9.3 for the NVFP4 9B).
pub(super) fn plain_to_f32(raw: &[u8], dtype: &str) -> Result<Vec<f32>> {
    match dtype {
        "F16" => Ok(raw.chunks_exact(2).map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect()),
        "F32" => Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
        other => bail!("unsupported plain dtype {other} for an f32 tensor"),
    }
}

// S-A3-f-f: K-split picker. Target ~200 resident blocks (the f-a ladder / lm_head
// evidence: 1940 blocks @ 122 GB/s vs 20-96 blocks @ 11-22 GB/s on the SAME kernel).
// Pure function of (k, n): deterministic per model, m-independent (G-A3-5).
pub(crate) fn chain_ks(k: i32, n: i32) -> u32 { chain_ks_target(k, n, 200) }

/// EXL27B: chain_ks for a dense pack's MLP quads (a copy; chain_ks itself is untouched). chain_ks
/// treats >= 128 column tiles as lm_head-class and picks ks = 1 — the per-k16-step barrier body —
/// which put the 27B's gate/up (N 17408 = 136 tiles, K 5120) 27% behind the split-K path.
pub(crate) fn dense_chain_ks(k: i32, n: i32) -> u32 {
    let tiles = (n.max(0) as u32) / 128;
    let kb = (k.max(0) as u32) / 16;
    if tiles == 0 || kb == 0 { return 1; }
    if tiles >= 512 { return 1; }
    let target = ((200 + tiles - 1) / tiles).min(kb).max(1);
    let mut ks = target;
    while ks > 1 && kb % ks != 0 { ks -= 1; }
    ks
}

/// chain_ks with an explicit resident-block target (`chain_ks` = 200). TP-I #8: the TP-sliced
/// quads may re-derive theirs from --tp-ks-target (xtp::tp_chain_ks).
pub(crate) fn chain_ks_target(k: i32, n: i32, blocks: u32) -> u32 {
    let tiles = (n.max(0) as u32) / 128;
    let kb = (k.max(0) as u32) / 16;
    if tiles == 0 || kb == 0 { return 1; }
    if tiles >= 128 { return 1; } // lm_head-class: already saturated (1940t @ 122 GB/s)
    let target = ((blocks.max(1) + tiles - 1) / tiles).min(kb).max(1);
    let mut ks = target;
    while ks > 1 && kb % ks != 0 { ks -= 1; }
    ks
}

// S-A3-f-d step 0 lever 1: router K-split picker. The monolithic router GEMV runs
// ceil(ne/256) blocks (2 for ne=512) — latency-bound; split K so grid.x * ks
// reaches ~128 resident blocks. Constraint: K % ks == 0 AND (K/ks) % 8 == 0
// (float4-clean slices). Pure function of (K, ne): batch-invariant (G-A3-5).
// --exl3-router-ks overrides (0/1 = the old monolithic path, diagnostics/A-B).
pub(super) fn router_ks(k: i32, ne: i32) -> u32 {
    if let Ok(v) = crate::opts::var(crate::opt!("exl3-router-ks")) {
        let ks: i32 = v.parse().unwrap_or(1);
        return if ks > 1 && k % ks == 0 && (k / ks) % 8 == 0 { ks as u32 } else { 1 };
    }
    let gx = ((ne.max(0) + 255) / 256).max(1);
    let mut ks = ((128i32 / gx).max(1)).min(k.max(0));
    while ks > 1 && (k % ks != 0 || (k / ks) % 8 != 0) { ks -= 1; }
    ks.max(1) as u32
}

// S-A3-f-f ITEM 1: measure xq_hc_fuse co-residency (max co-resident blocks for the
// whole device at block=256). Cudarc keeps CudaFunction's raw handle private, so the
// PTX is loaded a second time through the raw driver API purely for this query
// (gdn_chunk_load_raw_fn precedent, gpu.rs; process-lifetime leak by design).
// ANY failure returns 0 => the fused mixer is disabled for ALL m and the old
// 3-kernel path runs consistently (batch invariance never splits across paths).
pub(super) fn hc_fuse_co_residency(_dev: &Arc<CudaDevice>) -> u32 {
    use cudarc::driver::sys;
    let ptx = crate::exl3_bench::bench_ptx_src();
    let Ok(ptx_c) = std::ffi::CString::new(ptx) else { return 0; };
    let mut module: sys::CUmodule = std::ptr::null_mut();
    let r = unsafe {
        sys::cuModuleLoadDataEx(&mut module, ptx_c.as_ptr() as *const std::ffi::c_void,
                                0, std::ptr::null_mut(), std::ptr::null_mut())
    };
    if r != sys::CUresult::CUDA_SUCCESS { return 0; }
    let Ok(cname) = std::ffi::CString::new("xq_hc_fuse") else { return 0; };
    let mut f: sys::CUfunction = std::ptr::null_mut();
    let r = unsafe { sys::cuModuleGetFunction(&mut f, module, cname.as_ptr()) };
    if r != sys::CUresult::CUDA_SUCCESS { return 0; }
    let mut blocks_per_sm: std::ffi::c_int = 0;
    let r = unsafe {
        sys::cuOccupancyMaxActiveBlocksPerMultiprocessor(&mut blocks_per_sm, f, 256, 0)
    };
    if r != sys::CUresult::CUDA_SUCCESS { return 0; }
    let mut dev_h: sys::CUdevice = 0;
    if unsafe { sys::cuDeviceGet(&mut dev_h, 0) } != sys::CUresult::CUDA_SUCCESS { return 0; }
    let mut sms: std::ffi::c_int = 0;
    let r = unsafe {
        sys::cuDeviceGetAttribute(&mut sms,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev_h)
    };
    if r != sys::CUresult::CUDA_SUCCESS { return 0; }
    (blocks_per_sm.max(0) as u32).saturating_mul(sms.max(0) as u32)
}

/// S-A3-v: dynamic smem of `xq_gdn_chunk_tc` — mirrors XQ_GTC_SMEM in exl3_bench.cu
/// (C=32, VC=32, kd=vd=128): halves (4*32*136 + 32*40 + 2*32*40 + 4*32*40) * 2
/// + f32 (128*36 + 32*32 + 32*32 + 32 + 33) * 4.
pub(crate) const GTC_SMEM: i32 = (4 * 32 * 136 + 32 * 40 + 2 * 32 * 40 + 4 * 32 * 40) * 2
    + (128 * 36 + 32 * 32 + 32 * 32 + 32 + 33) * 4;
/// S-A3-v: value columns per block of `xq_gdn_chunk_tc` (XQ_GTC_VC).
pub(super) const GTC_VC: usize = 32;

/// S-A3-v: raw handle of the chunkwise-parallel GDN prefill scan with the >48 KB dynamic-smem
/// opt-in (cudarc 0.9 keeps CudaFunction's handle private — hc_fuse_co_residency precedent:
/// the PTX is loaded a second time through the driver API; process-lifetime leak by design).
/// --exl3-gdn-tc-prec selects the split-operand variant (7 = all split, default; 0/1/3 =
/// the ablation twins). None (stale PTX, opt-in refused) => the sequential xq_gdn_step_chunk.
/// A5-L1: `v2` (knob prefill.gdn_scan) selects the re-scheduled bitwise twin `xq_gdn_chunk_tc2`
/// (wavefront solve, register prefetch) for the default split-7 variant; the ablation twins
/// (--exl3-gdn-tc-prec=0/1/3) keep their kernels.
pub(super) fn gdn_chunk_tc_fn(v2: bool) -> Option<cudarc::driver::sys::CUfunction> {
    #[derive(Clone, Copy)]
    struct FnH(cudarc::driver::sys::CUfunction);
    unsafe impl Send for FnH {}
    unsafe impl Sync for FnH {}
    static V: std::sync::OnceLock<Option<FnH>> = std::sync::OnceLock::new();
    static V2: std::sync::OnceLock<Option<FnH>> = std::sync::OnceLock::new();
    (if v2 { &V2 } else { &V }).get_or_init(|| {
        use cudarc::driver::sys;
        let kname = match crate::opts::var(crate::opt!("exl3-gdn-tc-prec")).as_deref() {
            Ok("0") => "xq_gdn_chunk_tc_s0", Ok("1") => "xq_gdn_chunk_tc_s1",
            Ok("3") => "xq_gdn_chunk_tc_s3", _ => if v2 { "xq_gdn_chunk_tc2" } else { "xq_gdn_chunk_tc" },
        };
        let ptx = crate::exl3_bench::bench_ptx_src();
        let ptx_c = std::ffi::CString::new(ptx).ok()?;
        let mut module: sys::CUmodule = std::ptr::null_mut();
        let r = unsafe {
            sys::cuModuleLoadDataEx(&mut module, ptx_c.as_ptr() as *const std::ffi::c_void,
                                    0, std::ptr::null_mut(), std::ptr::null_mut())
        };
        if r != sys::CUresult::CUDA_SUCCESS { crate::deprintln!("[gtc-x] module load failed ({r:?})"); return None; }
        let cname = std::ffi::CString::new(kname).ok()?;
        let mut f: sys::CUfunction = std::ptr::null_mut();
        let r = unsafe { sys::cuModuleGetFunction(&mut f, module, cname.as_ptr()) };
        if r != sys::CUresult::CUDA_SUCCESS { crate::deprintln!("[gtc-x] get {kname} failed ({r:?})"); return None; }
        const _: () = assert!(GTC_SMEM <= 99 * 1024, "xq_gdn_chunk_tc smem over the GB10 opt-in cap");
        let r = unsafe {
            sys::cuFuncSetAttribute(f, sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, GTC_SMEM)
        };
        if r != sys::CUresult::CUDA_SUCCESS { crate::deprintln!("[gtc-x] smem opt-in failed ({r:?})"); return None; }
        crate::deprintln!("[gtc-x] chunkwise GDN prefill scan loaded: {kname} (dynamic smem {GTC_SMEM} B)");
        Some(FnH(f))
    }).map(|h| h.0)
}

/// S-A3-z: generic raw CUfunction from exl3_bench.ptx for kernels past the typed launcher's 12-arg
/// cap (one extra module load per process, the hc_fuse_co_residency precedent). None = stale PTX.
pub(crate) fn xq_raw_fn(name: &str) -> Option<cudarc::driver::sys::CUfunction> {
    use cudarc::driver::sys;
    #[derive(Clone, Copy)]
    struct H(sys::CUmodule);
    unsafe impl Send for H {}
    unsafe impl Sync for H {}
    #[derive(Clone, Copy)]
    struct FnH(sys::CUfunction);
    unsafe impl Send for FnH {}
    unsafe impl Sync for FnH {}
    static MODULE: std::sync::OnceLock<Option<H>> = std::sync::OnceLock::new();
    static FNS: std::sync::Mutex<Vec<(String, FnH)>> = std::sync::Mutex::new(Vec::new());
    if let Some((_, f)) = FNS.lock().ok()?.iter().find(|(n, _)| n == name) { return Some(f.0); }
    let module = MODULE.get_or_init(|| {
        let ptx = crate::exl3_bench::bench_ptx_src();
        let ptx_c = std::ffi::CString::new(ptx).ok()?;
        let mut m: sys::CUmodule = std::ptr::null_mut();
        let r = unsafe { sys::cuModuleLoadDataEx(&mut m, ptx_c.as_ptr() as *const std::ffi::c_void,
                                                  0, std::ptr::null_mut(), std::ptr::null_mut()) };
        if r != sys::CUresult::CUDA_SUCCESS { crate::deprintln!("[xq-raw] module load failed ({r:?})"); return None; }
        Some(H(m))
    }).as_ref()?.0;
    let cname = std::ffi::CString::new(name).ok()?;
    let mut f: sys::CUfunction = std::ptr::null_mut();
    let r = unsafe { sys::cuModuleGetFunction(&mut f, module, cname.as_ptr()) };
    if r != sys::CUresult::CUDA_SUCCESS { crate::deprintln!("[xq-raw] get {name} failed ({r:?})"); return None; }
    FNS.lock().ok()?.push((name.to_string(), FnH(f)));
    Some(f)
}

/// S-A3-v: minimum chunk rows for the chunkwise GDN prefill scan (--exl3-gdn-tc-min; 0 = off,
/// i.e. every chunk keeps the sequential xq_gdn_step_chunk). Short chunks stay sequential: a
/// short prompt's only chunk then keeps its exact pre-v bits (the SAU-11 lesson — short-prompt
/// text moves are measured), and a sub-64-row tail has little recurrence to parallelize.
pub(super) fn gdn_tc_min_rows() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| crate::opts::var(crate::opt!("exl3-gdn-tc-min")).ok()
        .and_then(|v| v.parse().ok()).unwrap_or(64))
}

/// LANERES2 spill payload version. 2 = ISSUES 122 (every plane captured whole, byte strides); 3 = ISSUES 123 (the
/// position planes carry only the rows the conversation uses, `spill_rows`); 4 = PFXPOOL (the draft head's KV + QSA key
/// planes are carried too); 5 = issue #41 (the pooled QSA key planes are captured at the kernels' lane stride: a
/// version-4 file written by a lane other than 0 with max_pos % ratio != 0 holds that lane's pooled blocks shifted).
/// Files of another version are dropped at the restore check (graceful re-prefill); version-1 files held 1/4 of each
/// f32 and 1/2 of each u16 per-slot plane. SPILLTRUNC bumped 5 -> 6: the payload gained the checkpoint section
/// after the hist tail (nck, grid C, fixed-bytes, the (pos, prefix-key, hist) table, and the ckpt words themselves)
/// so a prompt that is a TRUNCATED prefix of a snapshot can resume at a carried boundary. The section format and
/// version live in `kv_spill` (the boot scanner validates exactly what the restore accepts).
pub(crate) const SPILL_PAYLOAD_VERSION: u32 = crate::kv_spill::SPILL_PAYLOAD_VERSION;

/// KVPOOL (fork): rows mapped AHEAD of the highest row a caller names (kv_vmm.rs; > the deepest draft chain, the
/// dense-v3 tile prefetch and one pooled-key block), and the rows every lane keeps mapped while empty.
pub(crate) const VMM_MARGIN_ROWS: usize = 1024;
pub(crate) const VMM_FLOOR_ROWS: usize = 4096;

/// One slot's region of a `[width x per_slot]` device plane: (byte address, byte length). `elems` = the plane's
/// element count, `elem_bytes` = size_of the element type (ISSUES 122: the stride must be in BYTES).
pub(crate) fn spill_slot_region(base: u64, elems: usize, elem_bytes: usize, width: usize, slot: usize) -> (u64, usize) {
    let bytes = (elems / width) * elem_bytes;
    (base + (slot * bytes) as u64, bytes)
}

/// ISSUES 123: the rows a snapshot of `tokens` cached positions carries on every POSITION plane (trunk KV, raw indexer
/// keys): through the end of the `ratio`-row block that holds position `tokens` — the restore resumes at `tokens`, rows
/// at or past it are rewritten before they are read, and a pooled block is recomputed from all `ratio` raw rows of its
/// block, so that whole block is carried bit for bit as an own-lane hit has it. Capped at `max_pos`.
/// KVPOOL / TPPOOL (design v3 step 4): units of one position plane a lane holds for `rows` rows: a row per unit, or a
/// pooled QSA block (`div` rows) plus one block of read-ahead, clamped to the plane (`cap`). The one formula
/// `vmm_ensure`, `vmm_truncate` and their dry runs share.
pub(crate) fn plane_units(rows: usize, div: usize, cap: usize) -> usize {
    (if div > 1 { rows.div_ceil(div) + 1 } else { rows }).min(cap)
}

pub(crate) fn spill_rows(tokens: usize, ratio: usize, max_pos: usize) -> usize {
    let r = ratio.max(1);
    ((tokens + 1).div_ceil(r) * r).min(max_pos)
}

/// ISSUES 123: one slot's first `rows` rows of a `[width][segs][rows_cap][row_bytes]` device plane (`plane_bytes` total):
/// one (byte address, byte length) region per segment. Err if the plane is not that shape.
pub(crate) fn spill_row_regions(base: u64, plane_bytes: usize, width: usize, slot: usize, rows_cap: usize,
                                row_bytes: usize, rows: usize) -> Result<Vec<(u64, usize)>> {
    anyhow::ensure!(width > 0 && slot < width && rows_cap > 0 && row_bytes > 0 && rows <= rows_cap,
                    "spill: bad region request (width {width}, slot {slot}, rows {rows} of {rows_cap}, row {row_bytes} B)");
    let slot_bytes = plane_bytes / width;
    let seg_bytes = rows_cap * row_bytes;
    anyhow::ensure!(plane_bytes % width == 0 && slot_bytes % seg_bytes == 0 && slot_bytes > 0,
                    "spill: a {plane_bytes}-byte plane is not [{width}][segs][{rows_cap}][{row_bytes}]");
    Ok((0..slot_bytes / seg_bytes).map(|g| (base + (slot * slot_bytes + g * seg_bytes) as u64, rows * row_bytes)).collect())
}
