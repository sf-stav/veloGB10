//! DSV4.1 GPU host side — leg KI: the indexer (kernels `kernels/gpu_dsv41/50_indexer.cuh` + 51-54).
//!
//! What one index layer runs (PLAN/DSV41_REF_MAP.md §3.5; layers 2, 8, 14, 20, 24, 28, 32, 36):
//! 1. `score` (`d41_idx_score_r{1,2,4,8}`): RoPE + FP4 of every row's 32 x 128 query, FP4 tensor-core
//!    scores against the lane's index-K cache, the reference's bf16 chain, and a running top-512 filter
//!    per row. Scores are never stored. Grid = (chunks G, row blocks); a row block is <= R rows of ONE
//!    lane (they share the key pass: verify rows, prefill rows). Mode 1 visits only the candidate
//!    blocks (layers 24-36); the candidate source (20) also writes per-8-block maxima (`write_bmax`).
//! 2. `merge` (`d41_idx_merge`): only when a row block used more than one chunk; merges the sorted
//!    per-chunk lists. The same kernel merges per-rank key stripes under TP (T0 §3.8): give it the
//!    gathered lists as `part` / `pcnt` with G = ranks.
//! 3. layer 20 only: `cand` (`d41_idx_cand`) = top-2048 blocks per row (pinned newest block, lowest
//!    id on ties), then `union` (`d41_idx_union`) = the row block's union with per-row masks, which
//!    mode 1 of layers 24-36 visits.
//! Output: `sel` [rows][512] ascending compressed positions (-1 padded) and `sel_cnt` [rows]
//! (= min(512, visible)); the attention adds its own offset (the reference's `idxs + offset`).
//! Ties: (score desc, position asc), one total order, so chunking, stripes and row blocking never
//! change a selection (gated bit for bit in `probe`).
//!
//! Every length the kernels read (positions -> visible counts, union counts) comes from DEVICE
//! buffers, so a captured graph does not depend on the context (T0 §4.4).

#[path = "idx_ref.rs"]
pub mod reference;
#[path = "idx_test.rs"]
pub mod probe;

use crate::dsv41::config::Dsv41Config;
use crate::dsv41::weights::Launch;
use anyhow::{ensure, Context, Result};

/// Entry kernels of this band; the gpu_dsv41 loader (`super::load_module`) collects them.
pub const FNS: &[&str] = &[
    "d41_idx_score_r1", "d41_idx_score_r2", "d41_idx_score_r4", "d41_idx_score_r8",
    "d41_idx_merge", "d41_idx_cand", "d41_idx_union", "d41_idx_abi",
];

pub const HEADS: usize = 32;
pub const HEAD_DIM: usize = 128;
pub const TOPK: usize = 512;
pub const CAND_BLOCKS: usize = 2048;
pub const CAND_BLOCK: usize = 8;
pub const TILE: usize = 128;
/// Threads of every score CTA (16 warps: 4 key warps x row groups x tile groups).
pub const THREADS: u32 = 512;
/// Bytes of one prepared query row in the `dbg_q` dump: 32 x 64 codes + 32 x 4 scales.
pub const DBG_Q_ROW: usize = HEADS * 68;

/// One row block (Rust mirror of `d41_idx_rb`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct IdxRb { pub kq: u64, pub ks: u64, pub row0: i32, pub nrows: i32, pub pad0: i32, pub pad1: i32 }

/// The argument block of every indexer entry (Rust mirror of `d41_idx_args`; the ABI probe checks
/// the size on the GPU).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct IdxArgs {
    pub iq: u64, pub w: u64, pub pos: u64, pub rbs: u64,
    pub sel: u64, pub sel_cnt: u64, pub part: u64, pub pcnt: u64,
    pub bmax: u64, pub cand: u64, pub cand_cnt: u64, pub bm: u64,
    pub uni: u64, pub ucnt: u64, pub dbg: u64, pub err: u64, pub dbg_q: u64,
    /// i32 [n_rb], zeroed once: the score kernel's last chunk CTA merges the row block itself and re-zeroes
    /// it (no merge launch); 0 = the separate `merge` kernel (one CTA per row, rows in parallel)
    pub tick: u64,
    pub n_rb: i32, pub rows_total: i32, pub iq_ld: i32, pub w_ld: i32,
    pub in_f32: i32, pub ratio: i32, pub mode: i32, pub write_bmax: i32,
    pub n_keys_cap: i32, pub nb_cap: i32, pub nbw_cap: i32, pub g: i32,
    pub min_tiles: i32, pub dbg_cap: i32, pub ucap: i32, pub r: i32,
    /// key-stripe mode (TP): only keys [stripe_lo, stripe_hi) (multiples of 128), partial lists; 0 = off
    pub stripe_lo: i32, pub stripe_hi: i32,
    /// partial-list placement: row (b, r) lists at (b*R + r) * lists + list_off + c; lists = 0 means G
    pub list_off: i32, pub lists: i32,
    pub inv_freq: [f32; 32],
}

impl Default for IdxArgs {
    fn default() -> Self {
        IdxArgs { iq: 0, w: 0, pos: 0, rbs: 0, sel: 0, sel_cnt: 0, part: 0, pcnt: 0, bmax: 0, cand: 0, cand_cnt: 0, bm: 0,
                  uni: 0, ucnt: 0, dbg: 0, err: 0, dbg_q: 0, tick: 0, n_rb: 0, rows_total: 0, iq_ld: 0, w_ld: 0, in_f32: 0, ratio: 1,
                  mode: 0, write_bmax: 0, n_keys_cap: 0, nb_cap: 0, nbw_cap: 0, g: 1, min_tiles: 8, dbg_cap: 0, ucap: 0, r: 1,
                  stripe_lo: 0, stripe_hi: 0, list_off: 0, lists: 0, inv_freq: [0.0; 32] }
    }
}

/// Error bits a kernel may set in `err` (mirror of 51_idx_common.cuh).
pub const ERR_POS: i32 = 1;
pub const ERR_UCAP: i32 = 2;

/// The indexer shapes are compile-time constants of the kernels: refuse a config that differs.
pub fn check_config(cfg: &Dsv41Config) -> Result<()> {
    ensure!(cfg.index_heads == HEADS, "indexer kernels are built for 32 index heads, config has {}", cfg.index_heads);
    ensure!(cfg.index_head_dim == HEAD_DIM, "indexer kernels are built for index_head_dim 128, config has {}", cfg.index_head_dim);
    ensure!(cfg.rope_dim == 64, "indexer kernels rotate the last 64 dims, config rope_head_dim {}", cfg.rope_dim);
    ensure!(cfg.index_topk == TOPK, "indexer kernels select 512, config index_topk {}", cfg.index_topk);
    ensure!(cfg.candidate_topk_blocks == CAND_BLOCKS && cfg.candidate_block == CAND_BLOCK,
            "candidate pool kernels are built for 2048 x 8, config {} x {}", cfg.candidate_topk_blocks, cfg.candidate_block);
    Ok(())
}

/// The RoPE table of the compressed layers (every index source has compress_ratio > 0: YaRN with
/// compress_rope_theta): the ONE dsv41 builder (`crate::dsv41::rope`, torch on CUDA, GPU-built).
pub fn rope_inv_freq(cfg: &Dsv41Config) -> Result<[f32; 32]> {
    crate::dsv41::rope::table_for(cfg, crate::dsv41::rope::Class::Compressed)
}

/// The score entry for row blocks of up to `rows` rows: (kernel, R).
pub fn score_entry(rows: usize) -> (&'static str, usize) {
    match rows { 0..=1 => ("d41_idx_score_r1", 1), 2 => ("d41_idx_score_r2", 2), 3..=4 => ("d41_idx_score_r4", 4), _ => ("d41_idx_score_r8", 8) }
}
/// (stages, tile groups, candidate buffer per row) of the score entry with R rows (52_idx_score.cuh).
fn shape(r: usize) -> (usize, usize, usize) {
    match r { 1 => (2, 2, 1536), 2 => (3, 2, 1024), 4 => (3, 1, 1024), _ => (2, 1, 1024) }
}

/// Threads per CTA of the score kernel with R rows (`r1` = 256 at 2 CTAs/SM; `r2`/`r4`/`r8` = 512).
pub fn score_threads(r: usize) -> u32 {
    if r == 1 { 256 } else { THREADS }
}

/// Dynamic shared memory of the score kernel with R rows (mirror of `d41_idx_smem<R, S, TG, CAP>::TOTAL`).
pub fn score_smem(r: usize) -> usize {
    const MSM_BYTES: usize = 38_240;
    let (s, tg, cap) = shape(r);
    let kt = tg * TILE;
    let nv_end = r * cap * 8 + s * kt * 64 + s * kt * 4 + r * 2048 + r * 128 + r * 64 + s * tg * 64 + r * 8 + r * 4 + r * 4;
    let raw = (nv_end + 15) / 16 * 16;
    raw.max(MSM_BYTES)
}

/// Scratch the caller provides, in elements, for `n_rb` row blocks of R rows, G chunks, `rows` rows,
/// a key capacity `n_keys_cap` and `ucap` union entries per row block.
#[derive(Clone, Copy, Debug)]
pub struct IdxSizes { pub part_u64: usize, pub pcnt_i32: usize, pub sel_i32: usize, pub bmax_u16: usize, pub cand_i32: usize, pub bm_u32: usize, pub uni_u32: usize }
pub fn sizes(n_rb: usize, r: usize, g: usize, rows: usize, n_keys_cap: usize, ucap: usize) -> IdxSizes {
    let nb_cap = n_keys_cap.div_ceil(CAND_BLOCK);
    IdxSizes { part_u64: n_rb * r * g * TOPK, pcnt_i32: n_rb * r * g, sel_i32: rows * TOPK, bmax_u16: rows * nb_cap,
               cand_i32: rows * CAND_BLOCKS, bm_u32: rows * nb_cap.div_ceil(32), uni_u32: n_rb * ucap }
}

/// `cuLaunchKernelEx` on the raw handle (the > 48 KB smem opt-in lives there), PDL when asked.
fn launch(name: &str, grid: (u32, u32, u32), block: u32, smem: u32, l: &Launch, a: &IdxArgs) -> Result<()> {
    use cudarc::driver::sys::*;
    let f = super::raw_fn(name).with_context(|| format!("gpu_dsv41 entry {name} missing"))?;
    unsafe {
        if smem > 48 * 1024 {
            let r = cuFuncSetAttribute(f, CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32);
            ensure!(r == CUresult::CUDA_SUCCESS, "cuFuncSetAttribute({name}, smem {smem}): {r:?}");
        }
        let mut attr = CUlaunchAttribute {
            id: CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
            pad: [0; 4],
            value: CUlaunchAttributeValue { programmaticStreamSerializationAllowed: 1 },
        };
        let mut cfg = CUlaunchConfig {
            gridDimX: grid.0, gridDimY: grid.1, gridDimZ: grid.2, blockDimX: block, blockDimY: 1, blockDimZ: 1,
            sharedMemBytes: smem, hStream: l.stream as CUstream, attrs: &mut attr, numAttrs: if l.pdl { 1 } else { 0 },
        };
        let mut p = [a as *const IdxArgs as *mut std::ffi::c_void];
        let r = cuLaunchKernelEx(&mut cfg, f, p.as_mut_ptr(), std::ptr::null_mut());
        ensure!(r == CUresult::CUDA_SUCCESS, "launch {name}: {r:?}");
    }
    Ok(())
}

/// The fused score + selection over `a.n_rb` row blocks of up to `a.r` rows, `a.g` chunks each.
pub fn score(l: &Launch, a: &IdxArgs) -> Result<()> {
    let (name, r) = score_entry(a.r as usize);
    ensure!(r == a.r as usize, "IdxArgs.r must be 1, 2, 4 or 8 (got {})", a.r);
    launch(name, (a.g as u32, a.n_rb as u32, 1), score_threads(r), score_smem(r) as u32, l, a)
}

/// Merge the per-chunk lists (a no-op for row blocks the score kernel finished in one chunk or merged
/// itself through `tick`).
pub fn merge(l: &Launch, a: &IdxArgs) -> Result<()> {
    launch("d41_idx_merge", ((a.n_rb * a.r) as u32, 1, 1), 512, 0, l, a)
}

/// Layer 20: the candidate blocks of every row (needs `bmax` from a `write_bmax` score call).
pub fn cand(l: &Launch, a: &IdxArgs) -> Result<()> {
    launch("d41_idx_cand", ((a.n_rb * a.r) as u32, 1, 1), 512, 0, l, a)
}

/// Layer 20: each row block's union of its rows' candidate blocks (mode 1 of layers 24-36 visits it).
pub fn union(l: &Launch, a: &IdxArgs) -> Result<()> {
    launch("d41_idx_union", (a.n_rb as u32, 1, 1), 512, 0, l, a)
}

/// TP key stripes (T0 §3.8): rank k of N scores keys [k*S, (k+1)*S) with `stripe_lo/hi` set,
/// `lists = N * G` and `list_off = k * G` into a gathered `part` / `pcnt`; after the gather, `merge`
/// with the same `lists` gives every row its global selection, bit-identical to the unsplit one
/// (gated in `probe`). The stripe width S must be a multiple of `TILE`.
pub fn stripe_args(a: &IdxArgs, k: usize, n: usize, stripe: usize) -> IdxArgs {
    let mut s = *a;
    s.stripe_lo = (k * stripe) as i32;
    s.stripe_hi = ((k + 1) * stripe) as i32;
    s.lists = (n * a.g as usize) as i32;
    s.list_off = (k * a.g as usize) as i32;
    s
}

/// One index layer: score (+ merge when chunked); for the candidate source also cand + union.
pub fn select(l: &Launch, a: &IdxArgs, candidate_source: bool) -> Result<()> {
    score(l, a)?;
    if a.g > 1 && a.tick == 0 { merge(l, a)?; }
    if candidate_source { cand(l, a)?; union(l, a)?; }
    Ok(())
}

/// The decode / verify shape of an index layer (leg PK1; for the forward to adopt, see PLAN/DSV41_PK1_REPORT.md):
/// which rows share a key pass (row blocks), how many key chunks a row block is cut into, and the merge path. The
/// selection is the same bits in every shape (KI's gate: chunking, row blocking and stripes never change it), so this
/// is purely a speed choice. Measured (PK1, `.12`): the prefill shape the forward uses today (8-row blocks, ONE chunk)
/// costs 104 us per ratio-2 layer for 6 rows at 4K and 1.59 ms at 128K, against 53 / 129 us for this plan.
#[derive(Clone, Debug, PartialEq)]
pub struct IdxPlan {
    /// rows per row block (the score entry's R: 1, 2, 4 or 8)
    pub r: usize,
    /// key chunks per row block (`IdxArgs::g`)
    pub g: usize,
    /// `IdxArgs::min_tiles`
    pub min_tiles: usize,
    /// row blocks as (row0, nrows): every block holds rows of ONE lane
    pub rbs: Vec<(i32, i32)>,
    /// 1-row blocks merge in the score kernel's last chunk CTA (`IdxArgs::tick`, i32 [n_rb] zeroed once); otherwise the
    /// separate merge kernel runs when g > 1
    pub fused_merge: bool,
}

/// Visible keys up to which every row gets its own row block (no shared key pass): at short and medium contexts the
/// call is latency-bound and one-row blocks run in parallel (4 x 1 rows at 4K: 22 us per layer against 59 us for one
/// lane's 6 rows in an 8-row block; 16K: 45 against 67); above it the shared key pass of a lane's rows saves DRAM bytes.
pub const PLAN_ROW_KEYS: usize = 16384;

/// The plan for `lanes` = (row0, nrows) per lane (rows of a lane consecutive) and `n_keys` = the most visible keys of
/// any row (positions / ratio). Mode-1 layers (24-36) must reuse layer 20's plan (its union is per row block).
pub fn decode_plan(lanes: &[(usize, usize)], n_keys: usize) -> IdxPlan {
    let per_lane = lanes.iter().map(|l| l.1).max().unwrap_or(1).max(1);
    let r = if n_keys <= PLAN_ROW_KEYS { 1 } else { score_entry(per_lane).1 };
    let mut rbs = Vec::new();
    for &(row0, n) in lanes {
        let mut i = 0;
        while i < n { let k = (n - i).min(r); rbs.push(((row0 + i) as i32, k as i32)); i += k; }
    }
    let n_rb = rbs.len().max(1);
    IdxPlan { r, g: (96 / n_rb).max(1), min_tiles: 8, rbs, fused_merge: r == 1 }
}

#[cfg(test)]
mod plan_tests {
    use super::*;
    #[test]
    fn dsv41_pk1_idx_plan_shapes() {
        let p = decode_plan(&[(0, 6)], 2048);
        assert_eq!((p.r, p.g, p.rbs.len(), p.fused_merge), (1, 16, 6, true));
        let p = decode_plan(&[(0, 6)], 65536);
        assert_eq!((p.r, p.g, p.rbs.clone(), p.fused_merge), (8, 96, vec![(0, 6)], false));
        let p = decode_plan(&[(0, 6), (6, 6), (12, 6), (18, 6)], 524288);
        assert_eq!((p.r, p.g, p.rbs.len()), (8, 24, 4));
        let p = decode_plan(&[(0, 1), (1, 1), (2, 1), (3, 1)], 1 << 20);
        assert_eq!((p.r, p.g, p.rbs.len(), p.fused_merge), (1, 24, 4, true));
        let p = decode_plan(&[(0, 2)], 1 << 20);
        assert_eq!((p.r, p.rbs.clone()), (2, vec![(0, 2)]));
        // every row in exactly one block, blocks never cross a lane
        let lanes = [(0usize, 5usize), (5, 3), (8, 6)];
        for keys in [1000usize, 100_000] {
            let p = decode_plan(&lanes, keys);
            let mut seen = vec![0; 14];
            for &(r0, n) in &p.rbs {
                assert!(n as usize <= p.r);
                let lane = lanes.iter().position(|&(a, b)| (r0 as usize) >= a && (r0 as usize) < a + b).unwrap();
                assert!((r0 + n) as usize <= lanes[lane].0 + lanes[lane].1);
                for i in r0..r0 + n { seen[i as usize] += 1; }
            }
            assert!(seen.iter().all(|&c| c == 1));
        }
    }
}
