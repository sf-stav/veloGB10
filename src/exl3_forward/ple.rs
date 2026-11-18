//! PLE (per-layer embedding input): WP05 row decoders and pinned host staging, the RAM / SSD residency and
//! headroom rules, the sidecar shard sources, `impl PleDev` and the `impl FwdModel` ple_* methods.
//! See `mod.rs` for the file map.

use super::*;

/// WP05 escape (diagnostics / A-B / bisect): --wp05-off=1 (or `all`) restores every pre-WP05
/// host path; a comma list disables single parts: `row` (embed_row group decode -> per-bit
/// decode), `par` (prefill PLE: parallel build + pinned async upload -> sequential +
/// htod_sync_copy), `pro` (prefill prologue: cos_tab gather + pinned async tables -> host
/// RoPE loop + htod_sync_copy), `dds` (DDS round: one-sync pinned per-pass readback + PLE row
/// interleave + packed pinned verify uploads -> the old per-pass sync + pageable copies).
pub(super) fn wp05_on(part: &str) -> bool {
    static V: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let v = V.get_or_init(|| crate::opts::var(crate::opt!("wp05-off")).unwrap_or_default());
    if v.is_empty() || v == "0" { return true; }
    if v == "1" || v == "all" { return false; }
    !v.split(',').any(|p| p.trim() == part)
}

/// WP05 --ple-xcheck (alias --wp05-xcheck): eager new-vs-old diff of every WP05 host
/// product — PLE embedding bits (decode, verify, prefill), prefill RoPE rows, verify staging —
/// printing mismatch counts and the first mismatch. Diagnostic only (adds host work + syncs).
pub(super) fn ple_xcheck_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::opts::var(crate::opt!("ple-xcheck")).is_ok() || crate::opts::var(crate::opt!("wp05-xcheck")).is_ok())
}

/// First-mismatch report for the WP05 XCHECK lines: (count, first index, new, old).
pub(super) fn wp05_diff_u16(new: &[u16], old: &[u16]) -> (usize, Option<(usize, u16, u16)>) {
    let mut n = 0usize;
    let mut first = None;
    for (i, (a, b)) in new.iter().zip(old.iter()).enumerate() {
        if a != b {
            if first.is_none() { first = Some((i, *a, *b)); }
            n += 1;
        }
    }
    (n + new.len().abs_diff(old.len()), first)
}

/// Pre-WP05 per-bit decode of one 102-byte trellis row (f16 scale | 800-bit stream) into 160 f16
/// bits — kept verbatim as the reference (--wp05-off=row, XCHECK, unit test).
pub(super) fn ple_decode_row_ref(raw: &[u8; 102], bias: &[u16], codebook: &[f32], out: &mut [u16]) {
    let mut words = [0u16; 51];
    for (w_i, c) in raw.chunks_exact(2).enumerate() {
        words[w_i] = u16::from_le_bytes([c[0], c[1]]);
    }
    let scale = half::f16::from_bits(words[0]).to_f32();
    for i in 0..160usize {
        let mut st: u32 = 0;
        for mm in 0..16usize {
            let src = ((i as i64 - (mm as i64) / 5).rem_euclid(160) as usize) * 5 + mm % 5;
            let bit = (words[1 + src / 16] >> (src % 16)) & 1;
            st |= (bit as u32) << mm;
        }
        let bias = half::f16::from_bits(bias[i]).to_f32();
        let v = codebook[st as usize] * scale + bias;
        out[i] = half::f16::from_f32(v).to_bits();
    }
}

/// WP05 exact rewrite of `ple_decode_row_ref`. Output i's 16-bit codebook index gathers stream
/// bits ((i - mm/5) mod 160)*5 + mm%5 for mm = 0..16, i.e. the 5-bit groups g(i), g(i-1),
/// g(i-2) and bit 0 of g(i-3) (group j = stream bits 5j..5j+5, LE bit order of raw[2..]):
///   st = g(i) | g(i-1) << 5 | g(i-2) << 10 | (g(i-3) & 1) << 15.
/// The 160 groups are cut once (8 per 40-bit window), then each output is one lookup — the SAME
/// f32 `codebook[st] * scale + bias` (Rust never contracts to FMA) and the same f16 rounding
/// (half::f16::from_f32), so every non-NaN output is bit-identical (NaN payloads are unspecified
/// in Rust for either version; the pack's scales/biases are finite). `bias` = the f32 head_bias
/// row (the exact widening of the f16 bits the reference converts per element).
pub(super) fn ple_decode_row_fast(raw: &[u8; 102], bias: &[f32], codebook: &[f32], out: &mut [u16]) {
    let scale = half::f16::from_bits(u16::from_le_bytes([raw[0], raw[1]])).to_f32();
    // g[3 + j] = group j; g[0..3] = groups 157..159 (the mod-160 wrap for i < 3)
    let mut g = [0u32; 163];
    for k in 0..20usize {
        let o = 2 + 5 * k;
        let x = raw[o] as u64 | (raw[o + 1] as u64) << 8 | (raw[o + 2] as u64) << 16
            | (raw[o + 3] as u64) << 24 | (raw[o + 4] as u64) << 32;
        for t in 0..8usize {
            g[3 + 8 * k + t] = ((x >> (5 * t)) & 31) as u32;
        }
    }
    g[0] = g[160];
    g[1] = g[161];
    g[2] = g[162];
    let out = &mut out[..160];
    let bias = &bias[..160];
    for i in 0..160usize {
        let st = g[i + 3] | (g[i + 2] << 5) | (g[i + 1] << 10) | ((g[i] & 1) << 15);
        let v = codebook[st as usize] * scale + bias[i];
        out[i] = half::f16::from_f32(v).to_bits();
    }
}

/// CF-P1g (4.05 bpw): n-gram row geometry. A row is `10*K + 1` u16 words (f16 scale | 160 groups of K bits): K = 5 -> 102 B
/// (the 3.05 pack), K = 6 -> 122 B (the 4.05 pack). `PLE_ROW_MAX` bounds every row buffer.
pub(crate) const PLE_ROW_MAX: usize = 164;
pub(crate) fn ple_row_bytes(k: usize) -> usize { (10 * k + 1) * 2 }

/// K-generic per-bit reference decode (the formula of `ple_decode_row_ref` with the group width K): output i's 16-bit
/// codebook index takes bit mm from group (i - mm / K) mod 160, bit mm % K. For K = 5 this IS `ple_decode_row_ref`.
pub(super) fn ple_decode_row_ref_k(raw: &[u8], k: usize, bias: &[u16], codebook: &[f32], out: &mut [u16]) {
    let nw = 10 * k + 1;
    // LATENT (K=7/8): nw*2 up to 162 B for K=8; the scratch must cover the whole
    // row (82 u16 = 164 B), not just the K=6 max (64 u16) — else a heap overrun.
    let mut words = [0u16; 82];
    for (w_i, c) in raw[..nw * 2].chunks_exact(2).enumerate() {
        words[w_i] = u16::from_le_bytes([c[0], c[1]]);
    }
    let scale = half::f16::from_bits(words[0]).to_f32();
    for i in 0..160usize {
        let mut st: u32 = 0;
        for mm in 0..16usize {
            let src = ((i as i64 - (mm / k) as i64).rem_euclid(160) as usize) * k + mm % k;
            let bit = (words[1 + src / 16] >> (src % 16)) & 1;
            st |= (bit as u32) << mm;
        }
        let b = half::f16::from_bits(bias[i]).to_f32();
        out[i] = half::f16::from_f32(codebook[st as usize] * scale + b).to_bits();
    }
}

/// K-generic group decode (the structure of `ple_decode_row_fast`): cut the 160 K-bit groups once (8 per K-byte window),
/// then each output is one codebook lookup over a 16-bit window of groups i, i-1, ... — bit-identical to the reference.
pub(super) fn ple_decode_row_fast_k(raw: &[u8], k: usize, bias: &[f32], codebook: &[f32], out: &mut [u16]) {
    debug_assert!((1..=8).contains(&k));
    let scale = half::f16::from_bits(u16::from_le_bytes([raw[0], raw[1]])).to_f32();
    let mask = (1u64 << k) - 1;
    // g[16 + j] = group j; g[0..16] = groups 144..159 (the mod-160 wrap). Pad = 16 because the
    // back-walk needs ceil(16/k)-1 groups and k can be 1 (15 steps); the old 3-group pad
    // underflowed for k<=3 (#26 Pack B: PLE K=3). Bit-identical to ple_decode_row_ref_k's
    // (i - mm/k).rem_euclid(160) per-bit reference.
    let mut g = [0u32; 176];
    for blk in 0..20usize {
        let o = 2 + k * blk;
        let mut x = 0u64;
        for b in 0..k { x |= (raw[o + b] as u64) << (8 * b); }
        for t in 0..8usize { g[16 + 8 * blk + t] = ((x >> (k * t)) & mask) as u32; }
    }
    for p in 0..16usize { g[p] = g[160 + p]; }
    for i in 0..160usize {
        let mut st = g[i + 16];
        let (mut shift, mut j) = (k, 1usize);
        while shift < 16 {
            let take = k.min(16 - shift);
            st |= (g[i + 16 - j] & ((1u32 << take) - 1)) << shift;
            shift += take;
            j += 1;
        }
        out[i] = half::f16::from_f32(codebook[st as usize] * scale + bias[i]).to_bits();
    }
}

// The buffer is plain host memory; all mutation goes through &mut self.
unsafe impl Send for PinnedHost {}
unsafe impl Sync for PinnedHost {}

impl PinnedHost {
    pub(super) fn new(dev: &Arc<CudaDevice>, len: usize) -> Result<Self> {
        dev.bind_to_thread()?;
        let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
        let r = unsafe { cudarc::driver::sys::cuMemHostAlloc(&mut p, len.max(16), 0) };
        anyhow::ensure!(r == cudarc::driver::sys::CUresult::CUDA_SUCCESS && !p.is_null(),
                        "cuMemHostAlloc({len}) failed ({r:?})");
        // real zeros (padded rows/lanes are uploaded as zero, like the old vec![0; n] staging)
        unsafe { std::ptr::write_bytes(p as *mut u8, 0, len.max(16)) };
        Ok(Self { ptr: p as *mut u8, len, dev: dev.clone() })
    }
    pub(super) fn i32s(&mut self, off: usize, n: usize) -> &mut [i32] {
        assert!(off % 4 == 0 && off + n * 4 <= self.len);
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(off) as *mut i32, n) }
    }
    pub(super) fn u16s(&mut self, off: usize, n: usize) -> &mut [u16] {
        assert!(off % 2 == 0 && off + n * 2 <= self.len);
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(off) as *mut u16, n) }
    }
    /// Async HtoD of `bytes` at `off` into device address `dst` on the NULL stream.
    pub(super) fn htod(&self, off: usize, dst: u64, bytes: usize) -> Result<()> {
        assert!(off + bytes <= self.len);
        if bytes == 0 { return Ok(()); }
        let r = unsafe {
            cudarc::driver::sys::cuMemcpyHtoDAsync_v2(dst, self.ptr.add(off) as *const std::ffi::c_void,
                                                       bytes, *self.dev.cu_stream())
        };
        anyhow::ensure!(r == cudarc::driver::sys::CUresult::CUDA_SUCCESS, "pinned htod ({r:?})");
        Ok(())
    }
    /// Async DtoH of `bytes` from device address `src` into `off` on the NULL stream.
    pub(super) fn dtoh(&mut self, off: usize, src: u64, bytes: usize) -> Result<()> {
        assert!(off + bytes <= self.len);
        let r = unsafe {
            cudarc::driver::sys::cuMemcpyDtoHAsync_v2(self.ptr.add(off) as *mut std::ffi::c_void, src,
                                                       bytes, *self.dev.cu_stream())
        };
        anyhow::ensure!(r == cudarc::driver::sys::CUresult::CUDA_SUCCESS, "pinned dtoh ({r:?})");
        Ok(())
    }
    /// LANERES2: read a copied region back as bytes (after the stream synchronize).
    pub(super) fn as_bytes(&self, off: usize, n: usize) -> &[u8] {
        assert!(off + n <= self.len);
        unsafe { std::slice::from_raw_parts(self.ptr.add(off), n) }
    }
    /// LANERES2: stage bytes for an HtoD copy.
    pub(super) fn copy_from_slice(&self, off: usize, src: &[u8]) {
        assert!(off + src.len() <= self.len);
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr.add(off), src.len()) }
    }
    /// Read a 4-byte word the DMA engine wrote (call only after the stream synchronize).
    pub(super) fn read_u32(&self, off: usize) -> u32 {
        assert!(off % 4 == 0 && off + 4 <= self.len);
        unsafe { std::ptr::read_volatile(self.ptr.add(off) as *const u32) }
    }
}

impl Drop for PinnedHost {
    fn drop(&mut self) {
        // no copy may still read/write the buffer once it is freed
        let _ = self.dev.synchronize();
        let _ = self.dev.bind_to_thread();
        unsafe { cudarc::driver::sys::cuMemFreeHost(self.ptr as *mut std::ffi::c_void); }
    }
}

impl Wp05Stage {
    pub(super) fn new(dev: &Arc<CudaDevice>, w: usize) -> Result<Self> {
        let bytes = (w * 5 + 8) * 4 + w * 2560 * 2;
        Ok(Self { pin: PinnedHost::new(dev, bytes)?, w, inflight: false, ple_pre: None, spec: None })
    }
    pub(super) fn off_toks(&self) -> usize { 0 }
    pub(super) fn off_pos(&self) -> usize { self.w * 4 }
    pub(super) fn off_slots(&self) -> usize { self.w * 8 }
    pub(super) fn off_sids(&self) -> usize { self.w * 16 }
    pub(super) fn off_meta(&self) -> usize { self.w * 20 }
    pub(super) fn off_rb(&self) -> usize { self.w * 20 + 16 }
    pub(super) fn off_ple(&self) -> usize { self.w * 20 + 32 }
    pub(super) fn ple_row(&mut self, q: usize) -> &mut [u16] {
        let o = self.off_ple() + q * 2560 * 2;
        self.pin.u16s(o, 2560)
    }
}

/// The n-gram table location: `--ple-ram ram|ssd|auto` (`on`/`off` are the older spellings of ram/ssd).
pub(super) fn ple_mode() -> &'static str {
    match crate::opts::var(crate::opt!("ple-ram")).as_deref().map(str::trim) {
        Ok("on") | Ok("ram") => "ram",
        Ok("off") | Ok("ssd") => "ssd",
        _ => "auto",
    }
}

pub(super) fn mem_available_bytes() -> u64 {
    std::fs::read_to_string("/proc/meminfo").ok()
        .and_then(|m| m.lines().find(|l| l.starts_with("MemAvailable:"))
            .and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<u64>().ok()))
        .unwrap_or(0) * 1024
}

/// Headroom `auto` keeps free after a RAM-resident table (the AGENTS §5 rule): a 16 GiB base.
/// The pre-port fixed headroom base (GiB) — one definition, used by the arithmetic below.
const PLE_BASE_HEADROOM_GIB: f64 = 16.0;

/// PR #17 (ported, opt-in): `--ple-ram auto` headroom in GiB = 16 + `--ple-headroom-per-lane-gb`
/// per ADDITIONAL decode lane (width-1). Default per-lane 0 reproduces the pre-port fixed
/// 16 GiB exactly (bit-identical residency decision). The PR's hardcoded 2 GiB/lane default is
/// deliberately NOT taken — changing the default residency decision is the owner's call
/// (context: the public memory-spike report, S-B9-REL-MEMSPIKE).
pub(crate) fn ple_auto_headroom_gib(per_lane_gib: f64, width: usize) -> u64 {
    let lanes = width.saturating_sub(1) as f64;
    // inf * 0 = NaN would zero the headroom at width 1 (review nit): no lanes -> no per-lane term.
    let per_lane_total = if lanes == 0.0 { 0.0 } else { per_lane_gib.max(0.0) * lanes };
    let total = PLE_BASE_HEADROOM_GIB + per_lane_total;
    (total.max(0.0) as u64).saturating_mul(1 << 30)
}

/// The resolved `--ple-ram auto` headroom for THIS boot's lane width (flag -> default 0).
pub(crate) fn ple_auto_headroom(width: usize) -> u64 {
    let per_lane: f64 = crate::opts::var(crate::opt!("ple-headroom-per-lane-gb"))
        .ok().and_then(|s| s.trim().parse::<f64>().ok()).unwrap_or(0.0);
    ple_auto_headroom_gib(per_lane, width)
}

/// Read every shard's data section into host memory (8 threads).
pub(super) fn ple_read_table(shards: &[std::fs::File], data_off: &[u64], per: usize) -> Result<Vec<Vec<u8>>> {
    use std::os::unix::fs::FileExt;
    let n = shards.len();
    let mut out: Vec<Vec<u8>> = (0..n).map(|_| Vec::new()).collect();
    let chunks: Vec<(usize, &mut Vec<u8>)> = out.iter_mut().enumerate().collect();
    let per_thread = n.div_ceil(8);
    let mut groups: Vec<Vec<(usize, &mut Vec<u8>)>> = (0..8).map(|_| Vec::new()).collect();
    for (i, c) in chunks { groups[i / per_thread].push((i, c)); }
    // BASICS (owner 2026-11-02): the table read as a percentage at the normal level
    let prog = crate::boot_report::Progress::new("loading the n-gram table into RAM", (n * per) as u64,
                                                 crate::boot_report::Unit::Bytes);
    let prog = &prog;
    let res: Vec<Result<()>> = std::thread::scope(|sc| {
        let hs: Vec<_> = groups.into_iter().map(|g| sc.spawn(move || -> Result<()> {
            for (i, buf) in g {
                let mut v = vec![0u8; per];
                shards[i].read_exact_at(&mut v, data_off[i]).with_context(|| format!("ple ram shard {i}"))?;
                *buf = v;
                prog.add(per as u64);
            }
            Ok(())
        })).collect();
        hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(anyhow::anyhow!("ple ram thread panicked")))).collect()
    });
    for r in res { r?; }
    prog.finish();
    Ok(out)
}

// ---------------------------------------------------------------------------
// REL-NGRAM (v0.7.3 issue #9): the 128-shard n-gram sidecar layout of exllamav3 1.5.x.
//
// Packs written by exllamav3 1.5.x keep the PLE n-gram table as 128 tensors
// `<PLE_KEY>.ple_embedding.ngram_embedding.shard_{0..127}.trellis` inside the sidecar
// `ngram_embedding.safetensors` — and the model's model.safetensors.index.json does NOT list
// them. The pre-fix loader resolved every shard only through the index (`ple shard N not in
// index`) and insisted on row width 51 (K=5). The fix: when a shard is not in the index
// weight_map, resolve it from the sidecar file's own safetensors header, with the same
// 10*K+1 / K in {5,6} width rule the single-tensor path (CF-P1g) already uses, plus full
// layout validation (dtype I16, uniform rows/width, shards contiguous and in order).
// Shards that ARE in the index keep today's code path byte-for-byte.
// ---------------------------------------------------------------------------

/// Where one of the 128 n-gram trellis shards comes from.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PleShardSrc {
    /// Resolved through model.safetensors.index.json's weight_map (today's only path; the
    /// named file's own safetensors header supplies shape + data_offsets, as before).
    Index(String),
    /// Not in the index: resolved from `ngram_embedding.safetensors`'s own header
    /// (exllamav3 1.5.x layout). `abs_off` is the shard's absolute byte offset in the file.
    Sidecar { fname: String, abs_off: u64 },
}

/// Read + parse the n-gram sidecar file's safetensors header once (None if there is no
/// sidecar file). Returns (fname, header_json_len, parsed_header).
pub(crate) fn ple_sidecar_header(pack_dir: &str) -> Result<Option<(String, u64, serde_json::Value)>> {
    use std::io::Read;
    let fname = "ngram_embedding.safetensors".to_string();
    let path = format!("{}/{}", pack_dir.trim_end_matches('/'), fname);
    if !std::path::Path::new(&path).is_file() { return Ok(None); }
    let mut f = std::fs::File::open(&path).with_context(|| format!("ple open {path}"))?;
    let mut lb = [0u8; 8];
    f.read_exact(&mut lb)?;
    let hlen = u64::from_le_bytes(lb);
    let mut hb = vec![0u8; hlen as usize];
    f.read_exact(&mut hb)?;
    let hdr: serde_json::Value = serde_json::from_slice(&hb).with_context(|| format!("ple sidecar header parse {path}"))?;
    Ok(Some((fname, hlen, hdr)))
}

/// Validate the 128-shard layout described by a sidecar safetensors header and compute each
/// shard's absolute data offset (8 + header_len + data_offsets[0]). Pure — no I/O — so it is
/// unit-testable with synthetic headers. Rules (mirroring the single-tensor path's width rule
/// at CF-P1g and the reporter's verified layout for issue #9):
///   * every `names[i]` must be present (error names the shard);
///   * dtype must be I16;
///   * shape must be [rows, 10*K+1] with K in {5,6}, uniform across shards;
///   * rows must be uniform across shards;
///   * the byte span of every shard must equal rows*width*2 and shards must be contiguous
///     and in order (shard i starts exactly where shard i-1 ends);
///   * `__metadata__.shard_rows`, when present, is cross-checked against the shapes and only
///     warned about on mismatch (the single-tensor path derives everything from shapes and
///     ignores metadata — we keep that semantic and never trust metadata over shapes).
/// Returns (rows_per_shard, row_words, absolute offsets in `names` order).
pub(crate) fn ple_sidecar_shard_plan(
    hdr: &serde_json::Value,
    hlen: u64,
    names: &[String],
) -> Result<(i64, usize, Vec<u64>)> {
    let mut rows_per_shard: i64 = 0;
    let mut row_words: usize = 0;
    let mut offs = Vec::with_capacity(names.len());
    let mut prev_end: Option<u64> = None;
    for (i, name) in names.iter().enumerate() {
        let ent = hdr.get(name.as_str())
            .with_context(|| format!("ple shard {i} missing from ngram_embedding.safetensors header (model index does not list it either)"))?;
        let dtype = ent.get("dtype").and_then(|v| v.as_str()).unwrap_or("?");
        anyhow::ensure!(dtype == "I16", "ple sidecar shard {i} dtype {dtype:?}, expected I16");
        let shape = ent.get("shape").and_then(|v| v.as_array()).context("ple sidecar shard shape")?;
        anyhow::ensure!(shape.len() == 2, "ple sidecar shard {i} shape {shape:?}: expected 2 dims");
        let (srows, swidth) = (shape[0].as_i64().unwrap_or(-1), shape[1].as_i64().unwrap_or(-1));
        anyhow::ensure!(swidth > 0 && (swidth - 1) % 10 == 0 && matches!((swidth - 1) / 10, 1..=8),
                        "ple sidecar shard {i} row width {swidth}: rows must be 10*K+1 words with K = 1..=8");
        if i == 0 {
            rows_per_shard = srows;
            row_words = swidth as usize;
        } else {
            anyhow::ensure!(srows == rows_per_shard,
                            "ple sidecar shard {i} has {srows} rows, shard 0 has {rows_per_shard}: rows must be uniform");
            anyhow::ensure!(swidth as usize == row_words,
                            "ple sidecar shard {i} row width {swidth} differs from shard 0 ({row_words}): width must be uniform");
        }
        let doff = ent.get("data_offsets").and_then(|v| v.as_array()).context("ple sidecar shard offsets")?;
        let (d0, d1) = (doff[0].as_u64().context("ple sidecar shard offset start")?,
                        doff[1].as_u64().context("ple sidecar shard offset end")?);
        anyhow::ensure!(d1 > d0 && (d1 - d0) == srows as u64 * swidth as u64 * 2,
                        "ple sidecar shard {i} byte span {} != rows*width*2 = {}", d1 - d0, srows as u64 * swidth as u64 * 2);
        if let Some(pe) = prev_end {
            anyhow::ensure!(d0 == pe,
                            "ple sidecar shards not contiguous/in order: shard {i} starts at {d0} but shard {} ends at {pe}", i - 1);
        }
        prev_end = Some(d1);
        offs.push(8 + hlen + d0);
    }
    if let Some(m) = hdr.get("__metadata__") {
        if let Some(sr) = m.get("shard_rows").and_then(|v| v.as_i64()) {
            if sr != rows_per_shard {
                crate::dprintln!("  [ple] note: sidecar metadata shard_rows={sr} but shard shapes say {rows_per_shard}; trusting the shapes");
            }
        }
    }
    Ok((rows_per_shard, row_words, offs))
}

/// Decide, for each of the 128 shard names, whether it resolves through the index weight_map
/// (today's path, untouched) or through the sidecar header (issue #9 packs), and validate the
/// sidecar-sourced subset. Returns the per-shard sources plus the sidecar layout
/// (rows_per_shard, row_words) when at least one shard is sidecar-sourced.
/// K8 (REL v0.7.3 review): plan ONLY the sidecar-sourced subset of shards. Same
/// dtype/width/rows/span validation as `ple_sidecar_shard_plan`, but the subset's spans
/// must merely not OVERLAP — numeric order / global contiguity is not required when every
/// offset is explicit (a sidecar may hold just the shards the model index does not list).
/// Returns (rows_per_shard, row_words, absolute offsets in `idxs` order). Pure.
pub(crate) fn ple_sidecar_shard_plan_subset(
    hdr: &serde_json::Value,
    hlen: u64,
    idxs: &[usize],
    names: &[String],
) -> Result<(i64, usize, Vec<u64>)> {
    let mut rows_per_shard: i64 = 0;
    let mut row_words: usize = 0;
    let mut spans: Vec<(u64, u64, usize)> = Vec::with_capacity(idxs.len()); // (d0, d1, shard)
    let mut offs = Vec::with_capacity(idxs.len());
    for (k, &i) in idxs.iter().enumerate() {
        let name = names.get(i).context("ple sidecar subset index out of range")?;
        let ent = hdr.get(name.as_str())
            .with_context(|| format!("ple shard {i} missing from ngram_embedding.safetensors header (model index does not list it either)"))?;
        let dtype = ent.get("dtype").and_then(|v| v.as_str()).unwrap_or("?");
        anyhow::ensure!(dtype == "I16", "ple sidecar shard {i} dtype {dtype:?}, expected I16");
        let shape = ent.get("shape").and_then(|v| v.as_array()).context("ple sidecar shard shape")?;
        anyhow::ensure!(shape.len() == 2, "ple sidecar shard {i} shape {shape:?}: expected 2 dims");
        let (srows, swidth) = (shape[0].as_i64().unwrap_or(-1), shape[1].as_i64().unwrap_or(-1));
        anyhow::ensure!(swidth > 0 && (swidth - 1) % 10 == 0 && matches!((swidth - 1) / 10, 1..=8),
                        "ple sidecar shard {i} row width {swidth}: rows must be 10*K+1 words with K = 1..=8");
        if k == 0 {
            rows_per_shard = srows;
            row_words = swidth as usize;
        } else {
            anyhow::ensure!(srows == rows_per_shard,
                            "ple sidecar shard {i} has {srows} rows, shard {} has {rows_per_shard}: rows must be uniform",
                            idxs[0]);
            anyhow::ensure!(swidth as usize == row_words,
                            "ple sidecar shard {i} row width {swidth} differs from shard {} ({row_words}): width must be uniform",
                            idxs[0]);
        }
        let doff = ent.get("data_offsets").and_then(|v| v.as_array()).context("ple sidecar shard offsets")?;
        let (d0, d1) = (doff[0].as_u64().context("ple sidecar shard offset start")?,
                        doff[1].as_u64().context("ple sidecar shard offset end")?);
        anyhow::ensure!(d1 > d0 && (d1 - d0) == srows as u64 * swidth as u64 * 2,
                        "ple sidecar shard {i} byte span {} != rows*width*2 = {}", d1 - d0, srows as u64 * swidth as u64 * 2);
        spans.push((d0, d1, i));
        offs.push(8 + hlen + d0);
    }
    spans.sort_by_key(|s| s.0);
    for w in spans.windows(2) {
        anyhow::ensure!(w[1].0 >= w[0].1,
                        "ple sidecar shards {} and {} overlap in the sidecar data region", w[0].2, w[1].2);
    }
    Ok((rows_per_shard, row_words, offs))
}

pub(crate) fn ple_shard_sources(
    wmap: Option<&serde_json::Map<String, serde_json::Value>>,
    sidecar: Option<(&str, u64, &serde_json::Value)>,
    names: &[String],
) -> Result<(Vec<PleShardSrc>, Option<(i64, usize)>)> {
    let mut srcs = Vec::with_capacity(names.len());
    let mut sidecar_idx: Vec<usize> = Vec::new();
    for (i, name) in names.iter().enumerate() {
        let via_index = wmap.and_then(|w| w.get(name)).and_then(|v| v.as_str());
        match via_index {
            Some(f) => srcs.push(PleShardSrc::Index(f.to_string())),
            None => {
                if sidecar.is_none() {
                    anyhow::bail!("ple shard {i} not in index");
                }
                sidecar_idx.push(i);
                srcs.push(PleShardSrc::Sidecar { fname: String::new(), abs_off: 0 });
            }
        }
    }
    let mut layout = None;
    if !sidecar_idx.is_empty() {
        let (fname, hlen, hdr) = sidecar.unwrap();
        // K8: a sidecar holding ONLY the shards the index does not list plans just that
        // subset (non-overlapping spans; numeric order not required). The full-sidecar
        // layout keeps today's stricter contiguous-and-in-order planner.
        let (rows, words, offs) = if sidecar_idx.len() == names.len() {
            ple_sidecar_shard_plan(hdr, hlen, names)?
        } else {
            ple_sidecar_shard_plan_subset(hdr, hlen, &sidecar_idx, names)?
        };
        for (k, &i) in sidecar_idx.iter().enumerate() {
            srcs[i] = PleShardSrc::Sidecar { fname: fname.to_string(), abs_off: offs[k] };
        }
        layout = Some((rows, words));
    }
    Ok((srcs, layout))
}


/// PLE n-gram table location at weight-load time. `ram`: read into host memory now. `ssd`: per-token pread
/// through the page cache (identical bytes; ~1-2% decode, a few % prefill). `auto` (default): start on ssd and let
/// `FwdModel::ple_promote_auto` decide once the whole boot has allocated (CF-P1d).
pub(super) fn ple_ram_load(shards: &[std::fs::File], data_off: &[u64], rows: i64, row_bytes: usize, width: usize) -> Result<Option<Vec<Vec<u8>>>> {
    let mode = ple_mode();
    let per = rows as usize * row_bytes;
    let need = per as u64 * shards.len() as u64;
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    match mode {
        "ssd" => {
            crate::dprintln!("  PLE n-gram table: on SSD, read through the page cache (--ple-ram ssd; table {:.1} GiB stays out of RAM)", gib(need));
            Ok(None)
        }
        "auto" => {
            // PR #17: print the headroom the post-boot promote gate will apply, either way
            // (with the default per-lane 0 this reads the same 16.0 GiB as before the port).
            crate::dprintln!("  PLE n-gram table: on SSD until the boot has allocated everything else; then RAM iff table {:.1} GiB + {:.1} GiB headroom stay free (--ple-ram auto, {} lane(s); --ple-headroom-per-lane-gb)",
                     gib(need), ple_auto_headroom(width) as f64 / (1u64 << 30) as f64, width);
            Ok(None)
        }
        _ => {
            let avail = mem_available_bytes();
            let t0 = std::time::Instant::now();
            let out = ple_read_table(shards, data_off, per)?;
            crate::dprintln!("  PLE n-gram table RAM-resident: {:.1} GiB in {:.1}s (--ple-ram ram; MemAvailable was {:.1} GiB)",
                     gib(need), t0.elapsed().as_secs_f32(), gib(avail));
            Ok(Some(out))
        }
    }
}

impl PleDev {
    pub(super) fn mul1_codebook() -> Vec<f32> {
        const MUL1: u64 = 0x83DCD12D;
        let k_inv = half::f16::from_bits(0x1eee).to_f32();
        let k_bias = half::f16::from_bits(0xc931).to_f32();
        (0..65536u64)
            .map(|s| {
                let prod = s.wrapping_mul(MUL1) & 0xFFFF_FFFF;
                let bsum = (prod & 255) + ((prod >> 8) & 255) + ((prod >> 16) & 255) + ((prod >> 24) & 255);
                let h = 1024f32 + bsum as f32;
                half::f16::from_f32(h * k_inv + k_bias).to_f32()
            })
            .collect()
    }

    fn floor_mod(a: i64, b: i64) -> i64 { ((a % b) + b) % b }

    /// h = [prev2, prev1, cur] (eos-padded). 16 global row ids: bigram heads 0..8, trigram 8..16.
    pub(super) fn ngram_uids(&self, h: &[i32; 3]) -> [i64; 16] {
        let mut uids = [0i64; 16];
        let (x0, x1, x2) = (h[0] as i64, h[1] as i64, h[2] as i64);
        // shifted[1] = prev1 (reading eos as eos is identity); shifted[2] = prev2 unless an eos
        // sits between (n-grams never span eos boundaries).
        let s1 = x1;
        let s2 = if x1 == self.eos as i64 || x0 == self.eos as i64 { self.eos as i64 } else { x0 };
        let (m0, m1, m2) = (self.layer_multipliers[0], self.layer_multipliers[1], self.layer_multipliers[2]);
        let mix2 = x2.wrapping_mul(m0) ^ s1.wrapping_mul(m1);
        let mix3 = mix2 ^ s2.wrapping_mul(m2);
        for j in 0..8usize {
            uids[j] = Self::floor_mod(mix2, self.head_vocab_sizes[j]) + self.head_offsets[j];
            uids[8 + j] = Self::floor_mod(mix3, self.head_vocab_sizes[8 + j]) + self.head_offsets[8 + j];
        }
        uids
    }

    /// Decode the 16 rows and concat into 2560 f16 bits (head h -> dims [h*160, (h+1)*160)).
    /// WP05: the fetch is unchanged; the decode is the exact group rewrite (ple_decode_row_fast)
    /// unless --wp05-off disables `row`.
    pub(super) fn embed_row(&self, uids: &[i64; 16], out: &mut [u16]) -> Result<()> {
        if !wp05_on("row") {
            return self.embed_row_ref(uids, out);
        }
        assert_eq!(out.len(), 2560);
        let mut raw = [0u8; PLE_ROW_MAX];
        if let Some(fx) = self.fx.as_ref() {
            // FP4L: NVFP4 rows (e2m1 * e4m3 * gs, one RN to f16)
            for (hh, &uid) in uids.iter().enumerate() {
                self.fetch_row(uid, &mut raw)?;
                fp4l_fwd::ple_fx_decode(&raw, fx.gs, &mut out[hh * 160..(hh + 1) * 160]);
            }
            return Ok(());
        }
        for (hh, &uid) in uids.iter().enumerate() {
            self.fetch_row(uid, &mut raw)?;
            let (bias, o) = (&self.head_bias_f32[hh * 160..(hh + 1) * 160], &mut out[hh * 160..(hh + 1) * 160]);
            if self.row_k == 5 {
                ple_decode_row_fast(<&[u8; 102]>::try_from(&raw[..102]).unwrap(), bias, &self.codebook, o);
            } else {
                // LATENT (K=7/8): fast_k indexes into raw[2 + k*blk + (k-1)],
                // up to byte 161 at K=8. The fetch budget is row_bytes (<= 162 B
                // for K=8), while raw is PLE_ROW_MAX (128 B) — slice to the row
                // so the index stays within the actual row span.
                ple_decode_row_fast_k(&raw[..self.row_bytes], self.row_k, bias, &self.codebook, o);
            }
        }
        Ok(())
    }

    /// CF-P1g oracle (--exl3-ple-dump): the production-path embeddings (ngram_uids + the threaded `rows_par` fetch/decode)
    /// of 1,536 deterministic token positions, 1/29 of the tokens eos (segmentation), for scripts/ple_oracle.py.
    /// File: "PLEO", n, row_k, eos (u32 each); tokens (n+2) i32; row ids n*16 i64; values n*2560 u16 (fp16 bits).
    pub(super) fn dump_oracle(&self, path: &str) -> Result<()> {
        use std::io::Write;
        let n = 1536usize;
        let mut s = 0x1234_5678_9ABC_DEF1u64;
        let mut next = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        let toks: Vec<i32> = (0..n + 2).map(|_| { let r = next(); if r % 29 == 0 { self.eos as i32 } else { ((r >> 8) % 248_320) as i32 } }).collect();
        let h3s: Vec<[i32; 3]> = (2..n + 2).map(|t| [toks[t - 2], toks[t - 1], toks[t]]).collect();
        let mut out = vec![0u16; n * 2560];
        self.rows_par(&h3s, &mut out, false)?;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path).with_context(|| format!("--exl3-ple-dump {path}"))?);
        f.write_all(b"PLEO")?;
        for v in [n as u32, self.row_k as u32, self.eos as u32] { f.write_all(&v.to_le_bytes())?; }
        for t in &toks { f.write_all(&t.to_le_bytes())?; }
        for h in &h3s { for u in self.ngram_uids(h) { f.write_all(&u.to_le_bytes())?; } }
        for v in &out { f.write_all(&v.to_le_bytes())?; }
        f.flush()?;
        Ok(())
    }

    /// PERFFIX1: start the SSD reads of these rows NOW (posix_fadvise WILLNEED: the kernel queues them all at once)
    /// so the sequential `fetch_row` preads that follow find them in the page cache instead of each waiting for its
    /// own drive round trip (TP=1 posture, table on SSD: ~128 preads per verify round were ~2.5 ms of host time
    /// with the GPU idle). Advisory only: errors are ignored, nothing is read here, the bytes `fetch_row` returns are
    /// unchanged. No-op for the RAM-resident table.
    pub(super) fn prefetch_rows(&self, uids: &[i64]) {
        if self.ram.get().is_some() { return; }
        use std::os::unix::io::AsRawFd;
        let rb = self.row_bytes as i64;
        for &uid in uids {
            let shard = (uid / self.rows_per_shard) as usize;
            if shard >= self.shards.len() { continue; }
            let local = uid - shard as i64 * self.rows_per_shard;
            if let Some(fx) = self.fx.as_ref() {
                // FP4L: the two planes of the row
                let fd = self.shards[shard].as_raw_fd();
                // SAFETY: a valid fd and advisory ranges; the calls read or write no memory of ours.
                unsafe {
                    libc::posix_fadvise(fd, fx.w_off[shard] as i64 + local * 80, 80, libc::POSIX_FADV_WILLNEED);
                    libc::posix_fadvise(fd, fx.s_off[shard] as i64 + local * 10, 10, libc::POSIX_FADV_WILLNEED);
                }
                continue;
            }
            let off = self.shard_data_off[shard] as i64 + local * rb;
            // SAFETY: a valid fd and an advisory range; the call reads or writes no memory of ours.
            unsafe { libc::posix_fadvise(self.shards[shard].as_raw_fd(), off, rb, libc::POSIX_FADV_WILLNEED); }
        }
    }

    /// FP4L-SPEED1: prefill lookahead — read the n-gram rows of prompt positions [from, to) into the page cache on
    /// detached threads (16 positioned reads in flight per thread, up to 16 threads) while the GPU runs the current
    /// chunk, so the next chunk's gather (`rows_par`) finds them cached instead of waiting on the drive (a 4K-row
    /// chunk's ~131K reads at 8-deep took ~0.7 s of GPU idle per chunk). Reads exactly the bytes the gather will
    /// read (the same ngram_uids on the prompt's own history); the results are discarded; errors are ignored. No-op
    /// for the RAM-resident table.
    pub(super) fn warm_ahead(&self, prompt: &[u32], from: usize, to: usize) {
        if self.ram.get().is_some() || from >= to || to > prompt.len() { return; }
        let tok = |i: usize| -> i32 { prompt[i] as i32 };
        let mut reqs: Vec<(u32, u64, u8)> = Vec::with_capacity((to - from) * 32);
        for i in from..to {
            let h = [if i >= 2 { tok(i - 2) } else { self.eos }, if i >= 1 { tok(i - 1) } else { self.eos }, tok(i)];
            for uid in self.ngram_uids(&h) {
                let shard = (uid / self.rows_per_shard) as usize;
                if shard >= self.shards.len() { continue; }
                let local = (uid - shard as i64 * self.rows_per_shard) as u64;
                if let Some(fx) = self.fx.as_ref() {
                    reqs.push((shard as u32, fx.w_off[shard] + local * 80, 80));
                    reqs.push((shard as u32, fx.s_off[shard] + local * 10, 10));
                } else {
                    reqs.push((shard as u32, self.shard_data_off[shard] + local * self.row_bytes as u64, self.row_bytes as u8));
                }
            }
        }
        // one read per 4 KB page is enough to cache it: dedupe by (shard, page)
        reqs.sort_unstable_by_key(|r| (r.0, r.1));
        reqs.dedup_by_key(|r| (r.0, r.1 >> 12, (r.1 + r.2 as u64 - 1) >> 12));
        crate::metrics::ple_warm_ahead(((to - from) * 16) as u64, reqs.len() as u64); // engagement receipt (owner_check)
        let files: Vec<std::fs::File> = match self.shards.iter().map(|f| f.try_clone()).collect::<std::io::Result<_>>() {
            Ok(v) => v,
            Err(_) => return,
        };
        let nt = (reqs.len() / 256).clamp(1, 16);
        std::thread::spawn(move || {
            use std::os::unix::fs::FileExt;
            let per = reqs.len().div_ceil(nt).max(1);
            std::thread::scope(|sc| {
                for part in reqs.chunks(per) {
                    let files = &files;
                    sc.spawn(move || {
                        let mut buf = [0u8; 256];
                        for &(sh, off, len) in part {
                            let _ = files[sh as usize].read_at(&mut buf[..len as usize], off);
                        }
                    });
                }
            });
        });
    }

    /// PERFFIX1: `prefetch_rows` for the 16 rows of each n-gram history.
    pub(super) fn prefetch_h3s(&self, h3s: &[[i32; 3]]) {
        if self.ram.get().is_some() { return; }
        let uids: Vec<i64> = h3s.iter().flat_map(|h| self.ngram_uids(h)).collect();
        self.prefetch_rows(&uids);
    }

    /// `prefetch_h3s` split over up to `nt` scoped threads (each advise call is independent; at most one
    /// thread per 64 histories so short chunks stay on the calling thread). Workers re-pin to the worker
    /// CPU mask, as in `rows_par`.
    pub(super) fn prefetch_h3s_par(&self, h3s: &[[i32; 3]], nt: usize) {
        if self.ram.get().is_some() || h3s.is_empty() { return; }
        let nt = nt.min(h3s.len().div_ceil(64)).max(1);
        if nt <= 1 { return self.prefetch_h3s(h3s); }
        let per = h3s.len().div_ceil(nt);
        // Re-pin to the worker mask like rows_par: a pinned scheduler thread (TP serve) would otherwise
        // confine every spawned prefetch thread to its own core.
        std::thread::scope(|s| {
            for c in h3s.chunks(per) {
                s.spawn(move || { crate::cpu_affinity::pin_worker(); self.prefetch_h3s(c) });
            }
        });
    }

    /// One 102-byte trellis row (RAM table or pread) — the pre-WP05 fetch, unchanged.
    fn fetch_row(&self, uid: i64, raw: &mut [u8; PLE_ROW_MAX]) -> Result<()> {
        use std::os::unix::fs::FileExt;
        let rb = self.row_bytes;
        let shard = (uid / self.rows_per_shard) as usize;
        let local = (uid - shard as i64 * self.rows_per_shard) as u64;
        if let Some(ram) = self.ram.get() {
            let o = (local * rb as u64) as usize;
            raw[..rb].copy_from_slice(&ram[shard][o..o + rb]);
        } else if let Some(fx) = self.fx.as_ref() {
            // FP4L: the nibbles and the scales are separate tensors — two positioned reads
            self.shards[shard].read_exact_at(&mut raw[..80], fx.w_off[shard] + local * 80)
                .with_context(|| format!("ple read shard {shard} row {uid} (nibbles)"))?;
            self.shards[shard].read_exact_at(&mut raw[80..90], fx.s_off[shard] + local * 10)
                .with_context(|| format!("ple read shard {shard} row {uid} (scales)"))?;
        } else {
            let off = self.shard_data_off[shard] + local * rb as u64;
            self.shards[shard].read_exact_at(&mut raw[..rb], off)
                .with_context(|| format!("ple read shard {shard} row {uid}"))?;
        }
        Ok(())
    }

    /// WP05: PLE rows for a list of n-gram histories (h3 = [prev2, prev1, cur]) into
    /// out[q*2560..], rows split over up to 8 scoped threads (each row depends only on its own
    /// history, so the split cannot change any bit). `reference` = the pre-WP05 decode (XCHECK).
    pub(super) fn rows_par(&self, h3s: &[[i32; 3]], out: &mut [u16], reference: bool) -> Result<()> {
        let n = h3s.len();
        anyhow::ensure!(out.len() >= n * 2560, "ple rows_par: out {} < {} rows", out.len(), n);
        let run = |hh: &[[i32; 3]], o: &mut [u16]| -> Result<()> {
            for (q, h) in hh.iter().enumerate() {
                let uids = self.ngram_uids(h);
                let row = &mut o[q * 2560..(q + 1) * 2560];
                if reference { self.embed_row_ref(&uids, row)?; } else { self.embed_row(&uids, row)?; }
            }
            Ok(())
        };
        // SSD-resident table: the gather is drive-latency-bound (two positioned reads per FP4L row), so more readers
        // in flight; the RAM table is decode-bound (8 threads). Rows are independent: the split changes no bit.
        let nt = if reference { 1 } else if self.ram.get().is_some() { (n / 32).clamp(1, 8) } else { (n / 16).clamp(1, 64) };
        if nt <= 1 {
            return run(h3s, &mut out[..n * 2560]);
        }
        let per = n.div_ceil(nt);
        let res: Vec<Result<()>> = std::thread::scope(|s| {
            let hs: Vec<_> = h3s.chunks(per).zip(out[..n * 2560].chunks_mut(per * 2560))
                .map(|(hh, o)| s.spawn(move || { crate::cpu_affinity::pin_worker(); run(hh, o) }))
                .collect();
            hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(anyhow::anyhow!("ple row thread panicked")))).collect()
        });
        for r in res { r?; }
        Ok(())
    }

    /// The pre-WP05 embed_row (per-bit decode) — --wp05-off=row, XCHECK reference. Same fetch,
    /// and ple_decode_row_ref is the old inner loop verbatim.
    fn embed_row_ref(&self, uids: &[i64; 16], out: &mut [u16]) -> Result<()> {
        assert_eq!(out.len(), 2560);
        if self.fx.is_some() { return self.embed_row(uids, out); } // FP4L: one decoder (exact by construction)
        let mut raw = [0u8; PLE_ROW_MAX];
        for (hh, &uid) in uids.iter().enumerate() {
            self.fetch_row(uid, &mut raw)?;
            let (bias, o) = (&self.head_bias_bits[hh * 160..(hh + 1) * 160], &mut out[hh * 160..(hh + 1) * 160]);
            if self.row_k == 5 {
                ple_decode_row_ref(<&[u8; 102]>::try_from(&raw[..102]).unwrap(), bias, &self.codebook, o);
            } else {
                ple_decode_row_ref_k(&raw, self.row_k, bias, &self.codebook, o);
            }
        }
        Ok(())
    }
}



impl FwdModel {
    /// PLE n-gram injection (S-A3-d Phase 0b): emb gather (host) + proj/norm/gate/conv (device),
    /// delta added to the stream stack BEFORE block 1's hc_pre reads it.
    /// S-A3-f Item 1b: HOST half of PLE (n-gram ring update + embedding bits build).
    /// Pure host work on per-slot history + this step's tokens — runs in the step prefix,
    /// OUTSIDE the graphed kernel region.
    pub(super) fn ple_build_embed(&self, m: usize, toks: &[i32], slots: &[usize], emb_bits: &mut [u16]) -> Result<()> {
        let ple = match self.ple.as_ref() {
            Some(p) => p,
            None => return Ok(()),
        };
        let mut hist = self.ple_hist.lock().unwrap();
        // PERFFIX1: queue every SSD row of this step at once (the ring advanced on a copy exactly as the loop below
        // advances it); no-op for the RAM table
        if ple.ram.get().is_none() {
            let mut sim = hist.clone();
            let h3s: Vec<[i32; 3]> = (0..m).map(|q| {
                let s = slots[q];
                let h = [sim[s * 2], sim[s * 2 + 1], toks[q]];
                sim[s * 2] = sim[s * 2 + 1];
                sim[s * 2 + 1] = toks[q];
                h
            }).collect();
            ple.prefetch_h3s(&h3s);
        }
        for q in 0..m {
            // S-A3-e FIX: the n-gram ring is PER-SLOT (batch row q is the launch position).
            let s = slots[q];
            let h = [hist[s * 2], hist[s * 2 + 1], toks[q]];
            let uids = ple.ngram_uids(&h);
            ple.embed_row(&uids, &mut emb_bits[q * 2560..(q + 1) * 2560])?;
            if ple_xcheck_on() {
                let mut refr = vec![0u16; 2560];
                ple.embed_row_ref(&uids, &mut refr)?;
                let (n, first) = wp05_diff_u16(&emb_bits[q * 2560..(q + 1) * 2560], &refr);
                if n > 0 || q == 0 {
                    crate::dprintln!("PLE_XCHECK step row {q}/{m} slot {s}: {n} mismatching f16 of 2560 (first (idx, new, old) {first:?})");
                }
            }
            hist[s * 2] = hist[s * 2 + 1];
            hist[s * 2 + 1] = toks[q];
        }
        Ok(())
    }

    /// S-A3-f Item 1b: DEVICE half of PLE — launches only (capture-safe). Reads the
    /// embedding bits uploaded into the persistent sc.ple_emb by the step prefix.
    pub(super) fn ple_kernels(&self, l: &Launcher, sc: &mut Scratch, m: usize, prefix_done: bool) -> Result<()> {
        let ple = match self.ple.as_ref() {
            Some(p) => p,
            None => return Ok(()),
        };
        let eps = self.cfg.rms_eps as f32;
        let scale = 1.0f32 / (2560f32).sqrt();
        // S-A3-m: row-batched bit-exact twin (weights read once for all m rows).
        anyhow::ensure!(m <= 16, "ple_kernels: m {m} > 16");
        // WP27: `prefix_done` = ple_prefix already ran early (layer 0's MoE window).
        if !prefix_done {
            self.ple_prefix(l, sc, m)?;
        }
        let _ = eps;
        let dp = |p: &cudarc::driver::sys::CUdeviceptr| *p as u64;
        self.ple_norm(l, ple, 1, dp(sc.resid.device_ptr()), dp(sc.ple_qn.device_ptr()), m)?;
        xqlaunch!(l, "xq_ple_gate", (m as u32 * 4, 1, 1), (256, 1, 1), 0,
                 (&sc.ple_qn, &sc.ple_kn, &sc.ple_val, &mut sc.ple_gated, scale))?;
        self.ple_norm(l, ple, 2, dp(sc.ple_gated.device_ptr()), dp(sc.ple_normed.device_ptr()), m)?;
        if let Some(cf) = ple.conv_f.as_ref() {
            fp4l_fwd::ple_conv_f(l, dp(sc.resid.device_ptr()), dp(sc.ple_gated.device_ptr()), dp(sc.ple_normed.device_ptr()),
                                 dp(self.ple_state.device_ptr()), cf, m, dp(sc.slots.device_ptr()))?;
        } else {
        xqlaunch!(l, "xq_ple_conv", (((m * 10240) as u32 + 255) / 256, 1, 1), (256, 1, 1), 0,
                 (&mut sc.resid, &sc.ple_gated, &sc.ple_normed, &self.ple_state, &ple.conv_w, m as i32, &sc.slots))?;
        }
        xqlaunch!(l, "xq_ple_state_shift", (((m * 10240) as u32 + 255) / 256, 1, 1), (256, 1, 1), 0,
                 (&sc.ple_normed, *self.ple_state.device_ptr() as u64, m as i32, &sc.slots))?;
        Ok(())
    }

    /// PLE norm `which` (0 = key: f16 in / f32 out, 1 = query: f32 in / f32 out, 2 = conv: f16 in / f16 out) over
    /// `rows` rows: the trunk kernel on the f16 norm, or its twin on the stored fp32 norm (--fp4l-norms f32).
    pub(super) fn ple_norm(&self, l: &Launcher, ple: &PleDev, which: u8, x: u64, out: u64, rows: usize) -> Result<()> {
        let eps = self.cfg.rms_eps as f32;
        if let Some(nf) = ple.norm_f.as_ref() {
            return fp4l_fwd::ple_norm_f(l, which, x, &nf[which.min(2) as usize], out, rows, eps);
        }
        let (k, w) = match which {
            0 => ("xq_ple_norm_f16", &ple.norm_key_w),
            1 => ("xq_ple_norm_f32", &ple.norm_query_w),
            _ => ("xq_ple_norm_hh", &ple.norm_conv_w),
        };
        xqlaunch!(l, k, ((rows * 4) as u32, 1, 1), (256, 1, 1), 0, (x, w, out, eps))?;
        Ok(())
    }

    /// The head of the PLE device chain that reads nothing but the staged embedding bits
    /// (sc.ple_emb): key/value projections (WP09 k-major launch, else the row-major pair) and
    /// the key norm, in the original kernel order. Writes ple_key16 / ple_val / ple_kn only.
    /// Decode (ple_kernels) and verify run it at the layer-1 boundary, or — WP27 PLE fork —
    /// early, beside layer 0's MoE window.
    pub(super) fn ple_prefix(&self, l: &Launcher, sc: &mut Scratch, m: usize) -> Result<()> {
        let ple = match self.ple.as_ref() {
            Some(p) => p,
            None => return Ok(()),
        };
        let eps = self.cfg.rms_eps as f32;
        let _ = eps;
        let dp = |p: &cudarc::driver::sys::CUdeviceptr| *p as u64;
        if ple.bf {
            // FP4L --fp4l-plain exact: key / value from the stored bf16 weights (no k-major copies exist), in the
            // original order (key, key norm, value); the Scratch's PLE slabs / counters (the WP27 PLE fork may run
            // this beside layer 0's routed branch, which uses the router's own set)
            let fxs = sc.fxs.as_ref().context("FP4L stored-bf16 PLE: the Scratch has no FP4L buffers")?;
            let emb = dp(sc.ple_emb.device_ptr());
            fp4l_fwd::bf_gemm(l, &ple.key_w, emb, dp(sc.ple_key16.device_ptr()), 0, m, 2560, 10240, &fxs.ws_p, &fxs.cnt_p)?;
            self.ple_norm(l, ple, 0, dp(sc.ple_key16.device_ptr()), dp(sc.ple_kn.device_ptr()), m)?;
            let fxs = sc.fxs.as_ref().unwrap();
            return fp4l_fwd::bf_gemm(l, &ple.value_w, emb, dp(sc.ple_val.device_ptr()), 0, m, 2560, 2560, &fxs.ws_p, &fxs.cnt_p);
        }
        // WP09: key+value in ONE coalesced k-major launch (bitwise); else the row-major pair.
        let km = self.ple_proj_km(l, sc, m)?;
        if !km {
        xqlaunch!(l, "xq_gemm_f16_rows", (80u32, 1, 1), (128, 1, 1), 0,
                 (&mut sc.ple_key16, &ple.key_w, &sc.ple_emb, m as i32, 10240i32, 2560i32))?;
        }
        self.ple_norm(l, ple, 0, dp(sc.ple_key16.device_ptr()), dp(sc.ple_kn.device_ptr()), m)?;
        if !km {
        xqlaunch!(l, "xq_gemm_f16_rows", (20u32, 1, 1), (128, 1, 1), 0,
                 (&mut sc.ple_val, &ple.value_w, &sc.ple_emb, m as i32, 2560i32, 2560i32))?;
        }
        Ok(())
    }

    /// WP27 PLE fork: launch ple_prefix beside layer 0's MoE window (decode step and verify
    /// alike) instead of at the layer-1 boundary. Same conditions as the layer-1 PLE block.
    pub(super) fn wp27_ple_early(&self) -> bool {
        crate::exl3_wp27::fork_on(crate::exl3_wp27::FORK_PLE) && self.ple.is_some() && self.layers.len() > 1
            && crate::opts::var(crate::opt!("exl3-no-ple")).is_err()
    }

    /// WP09: the PLE key (N=10240) + value (N=2560) projections of sc.ple_emb through the k-major
    /// twin xq_gemm_f16_rows_km — ONE 100-block launch (80 key + 20 value blocks), coalesced
    /// 1-KB weight bursts, per-output arithmetic identical to xq_gemm_f16_rows (bitwise).
    /// Ok(false) = no k-major copies (--wp09-off=1 / --wp09-ple=0 at load): the caller runs
    /// the row-major pair at its original positions. Launch-only (capture-safe) unless
    /// --wp09-xcheck=1 on an eager step.
    fn ple_proj_km(&self, l: &Launcher, sc: &mut Scratch, m: usize) -> Result<bool> {
        let ple = match self.ple.as_ref() {
            Some(p) => p,
            None => return Ok(false),
        };
        let (kk, vk) = match (ple.key_km.as_ref(), ple.value_km.as_ref()) {
            (Some(a), Some(b)) => (a, b),
            _ => return Ok(false),
        };
        anyhow::ensure!(m >= 1 && m <= 16, "ple_proj_km: m {m} out of 1..=16");
        // W4S (ple): x staged in smem, exact-M body (bitwise; --w4s-off=ple = the launch below)
        if !self.w4s_ple_km(l, sc, kk, vk, m)? {
        xqlaunch!(l, "xq_gemm_f16_rows_km", (100u32, 1, 1), (128, 1, 1), 0,
                 (&mut sc.ple_key16, kk, 10240i32, 80i32, &mut sc.ple_val, vk, 2560i32,
                  &sc.ple_emb, m as i32, 2560i32))?;
        }
        if wp09_xcheck() && !self.stream_capturing() {
            let mut ko = self.dev.alloc_zeros::<u16>(m * 10240)?;
            let mut vo = self.dev.alloc_zeros::<u16>(m * 2560)?;
            xqlaunch!(l, "xq_gemm_f16_rows", (80u32, 1, 1), (128, 1, 1), 0,
                     (&mut ko, &ple.key_w, &sc.ple_emb, m as i32, 10240i32, 2560i32))?;
            xqlaunch!(l, "xq_gemm_f16_rows", (20u32, 1, 1), (128, 1, 1), 0,
                     (&mut vo, &ple.value_w, &sc.ple_emb, m as i32, 2560i32, 2560i32))?;
            self.dsync()?;
            let kn = self.dtoh(&sc.ple_key16)?;
            let vn = self.dtoh(&sc.ple_val)?;
            let ko = self.dtoh(&ko)?;
            let vo = self.dtoh(&vo)?;
            wp09_diff_u16("ple_key", m, &kn[..m * 10240], &ko);
            wp09_diff_u16("ple_val", m, &vn[..m * 2560], &vo);
        }
        Ok(true)
    }

}
