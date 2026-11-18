//! FP4L launch helpers for the trunk-kernel twins (hc fuse / norm / mix, router top-k, PLE, shared-expert gate), the `FxMoe` / `PleFx` carriers
//! and the split-K shape rules. Batch-invariant: ks is a pure function of (K, N). See `mod.rs` for the file map.

use super::*;

// ---- trunk-kernel twins (launch helpers; raw device pointers, the trunk sites own the buffers) ----

/// The exact mixer on the B32 down / up (xf_hc_fuse_b32): ONE launch of `g` co-resident CTAs (the caller sized g
/// within the kernel's measured co-residency); `ws` >= HCF_WS_PER_LR * lr fp32; `bar` TWO zeroed ints (count,
/// generation).
#[allow(clippy::too_many_arguments)]
pub(crate) fn hc_fuse_b32(l: &Launcher, dd: u64, uu: u64, hn: u64, down: &CudaSlice<u16>, up: &CudaSlice<u16>, ws: &CudaSlice<f32>,
                          bar: u64, hcn: f32, lr: usize, rw: usize, g: u32, m: usize) -> Result<()> {
    anyhow::ensure!((1..=16).contains(&m) && rw % 1024 == 0 && lr % 32 == 0 && lr <= 480 && ws.len() >= HCF_WS_PER_LR * lr && g >= 1,
                    "FP4L xf_hc_fuse_b32: m {m}, rw {rw}, lr {lr}, ws {} outside the contract", ws.len());
    crate::metrics::fp4l_twin("hc_mixer");
    xflaunch!(l, "xf_hc_fuse_b32", (g, 1, 1), (256, 1, 1), 0,
              (dd, uu, hn, down, up, ws, bar, hcn, lr as i32, rw as i32, m as i32))
}
/// fp32 scratch per mixer rank of xf_hc_fuse_b32: KS = 32 slabs x 16 rows of partials, then 16 rows of s.
pub(crate) const HCF_WS_PER_LR: usize = 33 * 16;
/// The CTA count of xf_hc_fuse_b32: 80 (one item per warp in both phases at rw 10240 / lr 320) within the margined
/// co-residency `cap` (0 = the kernel is not co-resident enough: the caller runs the unfused GEMMs).
pub(crate) fn hc_fuse_grid(cap: u32) -> u32 { if cap < 16 { 0 } else { cap.min(80) } }

/// The FP4L router's softmax / top-k / renorm (xf_router_topk: xq_router_topk's math, a faster exact selection) over
/// `b` rows of fp32 logits at `logits` -> ids / wts (device pointers).
pub(crate) fn router_topk(l: &Launcher, ids: u64, wts: u64, logits: u64, ne: usize, k: usize, b: usize) -> Result<()> {
    anyhow::ensure!(k * 8 <= 256 && ne >= k, "FP4L router top-k: k {k} / ne {ne} outside the kernel's contract");
    xflaunch!(l, "xf_router_topk", (b as u32, 1, 1), (256, 1, 1), ((256 + ne + 512) * 4) as u32,
              (ids, wts, logits, ne as i32, k as i32, b as i32))
}

/// xq_hc_norm with fp32 norm weights. grid hc*b, block 1024, 4 KB smem.
#[allow(clippy::too_many_arguments)]
pub(crate) fn hc_norm_f(l: &Launcher, out: u64, x: u64, w: &CudaSlice<f32>, h: usize, hcn: usize, b: usize, eps: f32) -> Result<()> {
    crate::metrics::fp4l_twin("hc_norm");
    xflaunch!(l, "xf_hc_norm_f", ((hcn * b) as u32, 1, 1), (1024, 1, 1), 4096,
              (out, x, w, h as i32, hcn as i32, b as i32, eps))
}
/// xq_hc_inj_norm (decode fused inject + norm) with fp32 norm weights. grid hc*b, block 1024.
#[allow(clippy::too_many_arguments)]
pub(crate) fn hc_inj_norm_f(l: &Launcher, out: u64, x: u64, y: u64, inj: u64, w: &CudaSlice<f32>, h: usize, hcn: usize,
                            b: usize, eps: f32) -> Result<()> {
    crate::metrics::fp4l_twin("hc_norm");
    xflaunch!(l, "xf_hc_inj_norm_f", ((hcn * b) as u32, 1, 1), (1024, 1, 1), 0,
              (out, x, y, inj, w, h as i32, hcn as i32, b as i32, eps))
}
/// xq_pf_hc_inj_norm (prefill fused inject + norm; y == 0 skips the inject) with fp32 norm weights. block 128.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pf_hc_inj_norm_f(l: &Launcher, out: u64, x: u64, y: u64, inj: u64, w: u64, h: usize, hcn: usize,
                               b: usize, eps: f32) -> Result<()> {
    crate::metrics::fp4l_twin("hc_norm");
    xflaunch!(l, "xf_pf_hc_inj_norm_f", ((hcn * b) as u32, 1, 1), (128, 1, 1), 0,
              (out, x, y, inj, w, h as i32, hcn as i32, b as i32, eps))
}
/// xq_hc_fuse with the stored bf16 mixers: grid (bx, m) — the caller sized bx within this kernel's co-residency.
#[allow(clippy::too_many_arguments)]
pub(crate) fn hc_fuse_bf(l: &Launcher, dd: u64, uu: u64, hn: u64, down: &CudaSlice<u16>, up: &CudaSlice<u16>, bar: u64,
                         hcn: f32, lr: usize, rw: usize, bx: u32, m: usize) -> Result<()> {
    crate::metrics::fp4l_twin("hc_mixer");
    xflaunch!(l, "xf_hc_fuse_bf", (bx, m as u32, 1), (256, 1, 1), 0,
              (dd, uu, hn, down, up, bar, hcn, lr as i32, rw as i32))
}
/// xq_hc_mix_rg with the stored bf16 inject weights (any b): grid (ninj + ceil(h/1024), b), block 1024.
#[allow(clippy::too_many_arguments)]
pub(crate) fn hc_mix_bf(l: &Launcher, x: u64, inj: u64, hn: u64, uu: u64, winj: u64, h: usize, hcn: usize, b: usize) -> Result<()> {
    crate::metrics::fp4l_twin("hc_mix");
    let ninj = if winj != 0 { hcn } else { 0 };
    let gx = (ninj + h.div_ceil(1024)) as u32;
    xflaunch!(l, "xf_hc_mix_bf", (gx, b as u32, 1), (1024, 1, 1), 0,
              (x, inj, hn, uu, winj, h as i32, hcn as i32, b as i32))
}
/// xq_pf_hc_mix_2560x4 with the stored bf16 inject weights (h == 2560, hc == 4 only). grid b, block 128.
pub(crate) fn pf_hc_mix_bf(l: &Launcher, x: u64, inj: u64, hn: u64, uu: u64, winj: u64, b: usize) -> Result<()> {
    crate::metrics::fp4l_twin("hc_mix");
    xflaunch!(l, "xf_pf_hc_mix_bf", (b as u32, 1, 1), (128, 1, 1), 0, (x, inj, hn, uu, winj, b as i32))
}
/// xq_embed_resid with the stored bf16 table.
pub(crate) fn embed_resid_bf(l: &Launcher, resid: u64, emb: &CudaSlice<u16>, toks: u64, h: usize, hcn: usize, b: usize) -> Result<()> {
    crate::metrics::fp4l_twin("embed");
    let n = (h * hcn * b) as u32;
    xflaunch!(l, "xf_embed_resid_bf", (n / 256 + 1, 1, 1), (256, 1, 1), 0, (resid, emb, toks, h as i32, hcn as i32, b as i32))
}
/// The shared-expert gate channels of m rows of x (see xf_sg_gate); `xg` [m][h], columns >= 6 zero.
pub(crate) fn sg_gate(l: &Launcher, xg: u64, sg_bf: &CudaSlice<u16>, x: u64, h: usize, m: usize) -> Result<()> {
    crate::metrics::fp4l_twin("shared_gate");
    xflaunch!(l, "xf_sg_gate", (m as u32, 1, 1), (256, 1, 1), 0, (xg, sg_bf, x, h as i32, m as i32))
}

/// The n-gram norm twins with the stored fp32 weights: `which` 0 = key (xq_ple_norm_f16: f16 in, f32 out),
/// 1 = query (xq_ple_norm_f32: f32 in, f32 out), 2 = conv (xq_ple_norm_hh: f16 in, f16 out). grid rows*4, block 256.
pub(crate) fn ple_norm_f(l: &Launcher, which: u8, x: u64, w: &CudaSlice<f32>, out: u64, rows: usize, eps: f32) -> Result<()> {
    crate::metrics::fp4l_twin("ple");
    let f = match which { 0 => "xf_ple_norm_f16_f", 1 => "xf_ple_norm_f32_f", _ => "xf_ple_norm_hh_f" };
    xflaunch!(l, f, ((rows * 4) as u32, 1, 1), (256, 1, 1), 0, (x, w, out, eps))
}
/// xq_ple_conv (decode / verify step) with the conv taps in fp32 (the stored bf16, widened exactly).
#[allow(clippy::too_many_arguments)]
pub(crate) fn ple_conv_f(l: &Launcher, resid: u64, gated: u64, normed: u64, state: u64, conv: &CudaSlice<f32>, m: usize,
                         slots: u64) -> Result<()> {
    crate::metrics::fp4l_twin("ple");
    xflaunch!(l, "xf_ple_conv_f", (((m * 10240) as u32 + 255) / 256, 1, 1), (256, 1, 1), 0,
              (resid, gated, normed, state, conv, m as i32, slots))
}
/// xq_ple_conv_chunk (prefill chunk / verify ring) with fp32 conv taps.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ple_conv_chunk_f(l: &Launcher, resid: u64, gated: u64, normed: u64, state: u64, conv: &CudaSlice<f32>,
                               slot: usize, c: usize) -> Result<()> {
    crate::metrics::fp4l_twin("ple");
    xflaunch!(l, "xf_ple_conv_chunk_f", ((10240u32 + 255) / 256, 1, 1), (256, 1, 1), 0,
              (resid, gated, normed, state, conv, slot as i32, c as i32))
}
/// xq_ple_conv_chunk_seg (lane-pack segments) with fp32 conv taps.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ple_conv_chunk_seg_f(l: &Launcher, resid: u64, gated: u64, normed: u64, state: u64, conv: &CudaSlice<f32>,
                                   seg: u64, seg_k: usize) -> Result<()> {
    crate::metrics::fp4l_twin("ple");
    xflaunch!(l, "xf_ple_conv_chunk_seg_f", ((10240u32 + 255) / 256, seg_k as u32, 1), (256, 1, 1), 0,
              (resid, gated, normed, state, conv, seg))
}

/// The f16 channel constants xf_sg_gate's channels are paired with: (1, 2^-11, 2^-22, 2^-24, 2^-24, 2^-24, 0, ...) [h].
pub(crate) fn sg_channel_constants(h: usize) -> Vec<u16> {
    let mut v = vec![0u16; h];
    for (i, b) in [0x3C00u16, 0x1000, 0x0004, 0x0001, 0x0001, 0x0001].into_iter().enumerate() { v[i] = b; }
    v
}

/// Host model of xf_sg_gate's encoding + a combine's fp32 sum over the channels (unit tests): returns the
/// six f16 channels of `l` and the value a combine rebuilds from them in the given pairing order.
pub(crate) fn sg_encode(l: f32) -> [u16; 6] {
    let mut a = l.abs().min(65504.0);
    let lo = [5i32, -6, -17, -28, -39, -48];
    let up = [0i32, 11, 22, 24, 24, 24];
    let mut out = [0u16; 6];
    for i in 0..6 {
        let v = (a * 2f32.powi(-lo[i])).floor() * 2f32.powi(lo[i]);
        a -= v;
        let c = v * 2f32.powi(up[i]);
        out[i] = half::f16::from_f32(if l < 0.0 { -c } else { c }).to_bits();
    }
    out
}

/// NVFP4 routed experts of one layer (or the MTP head), stacked in LOCAL order, each expert matrix in the
/// fragment-native F128 layout (`nv4_frag`; same byte counts as the checkpoint's row-major arrays).
pub(crate) struct FxMoe {
    /// gate|up fused per expert: F128 of nibbles [2*mi][h/2] + E4M3 scales [2*mi][h/16], [gate; up] rows
    pub gu_w: CudaSlice<u8>,
    pub gu_s: CudaSlice<u8>,
    /// down: F128 of nibbles [h][mi/2] + scales [h][mi/16]
    pub dn_w: CudaSlice<u8>,
    pub dn_s: CudaSlice<u8>,
    /// [E][3] fp32 multipliers (gate, up, down) = the checkpoint's weight_scale_2
    pub gs: CudaSlice<f32>,
    pub n_local: usize,
}

/// The NVFP4 PLE table (`ngram_embedding.shard_{s}.weight` [rows, 80] + `.weight_scale` [rows, 10]).
pub(crate) struct PleFx {
    /// per shard: absolute offset of the nibble data and of the scale data in `PleDev::shards[s]`
    pub w_off: Vec<u64>,
    pub s_off: Vec<u64>,
    /// the table's single weight_scale_2 (dequant multiplies by it)
    pub gs: f32,
}

/// FP4L PLE row: 80 B nibbles then 10 B E4M3 scales (the RAM table interleaves them this way too).
pub(crate) const PLE_FX_ROW: usize = 90;

// ------------------------------------------------------------------------------------------------
// Shapes / split-K
// ------------------------------------------------------------------------------------------------

/// Split-K factor of an m <= 16 dense FP4L GEMM: a pure function of (K, N) (batch invariance).
/// The smallest divisor of K/128 that puts >= 192 CTAs (4 waves of 48 SMs) on the device, else the
/// largest one; lm_head-class N (>= 192 tiles) runs ks = 1. (FP4L-SPEED1 tried a one-wave rule from a cold-L2
/// microbench that favoured it per shape: in the real decode step it was not faster — plain 26.9 -> 26.7 tok/s —
/// so this rule stands.)
pub(crate) fn fx_ks(k: usize, n: usize) -> u32 {
    let tiles = n.div_ceil(128).max(1);
    let kc = (k / 128).max(1);
    if tiles >= 192 { return 1; }
    let divs: Vec<usize> = (1..=kc).filter(|d| kc % d == 0).collect();
    for &d in &divs { if tiles * d >= 192 { return d as u32; } }
    *divs.last().unwrap() as u32
}

pub(super) fn check_dense_shape(k: usize, n: usize, what: &str) -> Result<()> {
    anyhow::ensure!(k % 128 == 0 && n % 2 == 0 && k > 0 && n > 0,
                    "FP4L {what}: K {k} must be a multiple of 128 and N {n} even (the xf_mx8 / xf_f16w contract)");
    Ok(())
}
