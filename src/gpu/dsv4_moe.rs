//! DSV4 routed-expert GPU helpers: `Dsv4MoeGpu`, workspaces, `dsv4_moe_experts_*` launch functions and the `dsv4_launch!`
//! macros (textually scoped to this file). N=1 and grouped paths are bitwise-identical per output element by construction. Map: `mod.rs`.

use super::*;

// -----------------------------------------------------------------------------------------------
// DSV4 routed-expert MoE (Phase 2 lane 2B, DEEPSEEK_V4_PORT.md §12.A.6 + §B.9).
//
// DSV4 semantics differ from Hy3's in exactly two places, both handled here:
//   1. The expert-GEMM inputs are FP8-QAT activations (§C.1 group-128 UE8M0): x, and the
//      SwiGLU output h, are round-tripped IN PLACE by `dsv4_act_quant_sim_g128` (bit-exact
//      vs the reference's act_quant), so the bf16-input NVFP4 expert GEMMs see value-identical
//      inputs to the reference fp4_gemm's fp8 codes (§C.4 via §C.1 — no new GEMM needed).
//   2. The routing weight multiplies h BEFORE w2 (`h *= routing_weight`, then w2(bf16(h)),
//      §B.9), behind the asymmetric clamps (up ±10, gate ≤ +10 — NO lower bound). That is the
//      new `dsv4_swiglu_clamp` kernel; the combine kernels then sum with an all-ones weight
//      (multiplying by exactly 1.0f is exact) instead of re-applying wts.
// Everything else reuses the proven machinery unchanged: `gemm_moe_mma_fp4` (N=1),
// `gemm_moe_grouped_mma_fp4` + the 6-kernel metadata pipeline (N=2..16 verify),
// `moe_combine_experts_b` / `moe_combine_grouped_b` (slot-order fp32).
//
// Batch-invariance (AGENTS.md §2.4): the N=1 and grouped paths are bitwise-identical per
// output element by construction — same MMA k-order per element, per-column independent
// epilogues, zero-filled padding rows, slot-order fp32 combines; the scatter permutation
// moves rows, never values. tests/dsv4_moe_dsv4_test.rs asserts to_bits equality for N=2..16.
//
// Phase-2 test wiring: these helpers launch on the default stream with dispatch-local
// scratch. Phase 3 serving integration moves scratch to the pool/compute stream, and runs
// the router (sqrtsoftplus/hash) BEFORE the in-place x act-sim (the router reads un-simmed x).
// -----------------------------------------------------------------------------------------------

/// Device-resident stacked routed experts for one DSV4 layer (host twin: dsv4_moe::Dsv4MoeHost).
pub struct Dsv4MoeGpu {
    pub gu_wt: cudarc::driver::CudaSlice<u8>,
    pub gu_st: cudarc::driver::CudaSlice<u8>,
    pub gu_gs: S,
    pub dn_wt: cudarc::driver::CudaSlice<u8>,
    pub dn_st: cudarc::driver::CudaSlice<u8>,
    pub dn_gs: S,
    pub ne: usize,
    pub h: usize,
    pub inter: usize,
    /// TP=2 expert-parallel band this rank holds (§5). `[e_base, e_base+e_span)` are the GLOBAL
    /// expert ids whose weights are stacked in `gu_*`/`dn_*` (LOCAL row 0 = global `e_base`).
    /// Unsharded: `e_base=0, e_span=ne`. The kernels rebase the router's global id by `e_base`
    /// and contribute EXACT ZEROS for remote experts (`gemm_moe_mma_fp4` gpu_batch.cu:3841;
    /// `moe_count_b`/`moe_scatter_b` skip remote pairs), so a rank's routed-expert output is a
    /// PARTIAL — one `tp_all_reduce_bf16` after the combine, BEFORE the shared-expert add,
    /// restores the full Σ (shared is replicated — reducing after the add would double it; see
    /// `gpu.rs:2577` for the proven Hy3 reference).
    pub e_base: usize,
    pub e_span: usize,
}

impl Dsv4MoeGpu {
    /// Unsharded upload (all `host.ne` experts). `e_base=0, e_span=ne`.
    pub fn upload(dev: &Arc<CudaDevice>, host: &crate::dsv4_moe::Dsv4MoeHost) -> anyhow::Result<Self> {
        Ok(Self {
            gu_wt: dev.htod_sync_copy(&host.gu_wt)?,
            gu_st: dev.htod_sync_copy(&host.gu_st)?,
            gu_gs: dev.htod_sync_copy(&host.gu_gs)?,
            dn_wt: dev.htod_sync_copy(&host.dn_wt)?,
            dn_st: dev.htod_sync_copy(&host.dn_st)?,
            dn_gs: dev.htod_sync_copy(&host.dn_gs)?,
            ne: host.ne,
            h: host.h,
            inter: host.inter,
            e_base: 0,
            e_span: host.ne,
        })
    }

    /// TP=2 sharded upload: stack only global experts `[e_base, e_base+e_span)` from `host` (the
    /// host is the full-256 `pack_moe_layer` output; each expert is a contiguous per-expert block
    /// in every buffer — §F.3, `dsv4_moe.rs`). `ne` stays the GLOBAL count (the grouped kernel's
    /// `poff[ne]` bound + the scratch sizing are global); only the weight buffers are sliced.
    pub fn upload_sharded(
        dev: &Arc<CudaDevice>,
        host: &crate::dsv4_moe::Dsv4MoeHost,
        e_base: usize,
        e_span: usize,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(e_base + e_span <= host.ne, "dsv4 moe shard [{e_base},{e_base}+{e_span}) > ne {}", host.ne);
        let (ne, h, inter) = (host.ne, host.h, host.inter);
        let gu_tiles = (2 * inter / 16) * (h / 16); // per-expert tiles
        let dn_tiles = (h / 16) * (inter / 16);
        let slice_u8 = |v: &[u8], per: usize| -> Vec<u8> {
            v[e_base * per..(e_base + e_span) * per].to_vec()
        };
        let slice_f = |v: &[f32], per: usize| -> Vec<f32> {
            v[e_base * per..(e_base + e_span) * per].to_vec()
        };
        let gu_wt = slice_u8(&host.gu_wt, gu_tiles * 128);
        let gu_st = slice_u8(&host.gu_st, gu_tiles * 16);
        let gu_gs = slice_f(&host.gu_gs, 2 * inter / 16);
        let dn_wt = slice_u8(&host.dn_wt, dn_tiles * 128);
        let dn_st = slice_u8(&host.dn_st, dn_tiles * 16);
        let dn_gs = slice_f(&host.dn_gs, h / 16);
        Ok(Self {
            gu_wt: dev.htod_sync_copy(&gu_wt)?,
            gu_st: dev.htod_sync_copy(&gu_st)?,
            gu_gs: dev.htod_sync_copy(&gu_gs)?,
            dn_wt: dev.htod_sync_copy(&dn_wt)?,
            dn_st: dev.htod_sync_copy(&dn_st)?,
            dn_gs: dev.htod_sync_copy(&dn_gs)?,
            ne,
            h,
            inter,
            e_base,
            e_span,
        })
    }
}

/// DSV4 R3.1 (L1) persistent MoE workspace: every buffer the FFN sublayer used to allocate
/// PER DISPATCH (the R1 nsys audit measured ~2528 cuMemAllocAsync + ~2334 memset PER TOKEN —
/// ~11 allocs + 3 host syncs + one htod per layer per token in the MoE path alone, 27% of CPU
/// API time). Allocated ONCE per model at load; each dispatch writes every byte before reading
/// it (the exact discipline the per-call allocs documented — sim scale outputs `sx`/`sh` are
/// written-but-never-consumed, padding rows are zero-filled by their producers, `count` is
/// refilled to 0 and `perm_tok` to -1 per dispatch). Fixed sizes: chunk ≤16 rows (the
/// G2-proven regime), `ppad_max` grouped rows (covers the N=1 pairs bound), `s_cap` rows of
/// router/routed traffic. ~160 MB per model at s_cap=4096 — nothing in the hot path allocs.
///
/// R3A.4 (wide dispatch): the stage buffers (xc/idc/wtc/outc/sx) are sized by
/// `s_cap.min(MOE_D_CAP)` — prefill routes whole 4096-row chunks in ONE dispatch instead of
/// 128 sequential 16-row dispatches (each of which re-read every touched expert's weights;
/// nsys: 9.5 s of a 21 s 2048-token prefill was the grouped GEMM's amplified re-reads).
pub struct Dsv4MoeWorkspace {
    pub s_cap: usize,
    pub scores: S,                 // [s_cap * ne] router scores (fp32)
    pub biased: S,                 // [s_cap * ne] bias-added scores (gate_bias arm only)
    pub routed_out: B,             // [s_cap * h] routed-expert partial (all-reduced in place under TP)
    pub sh_out: B,                 // [s_cap * h] shared-expert output
    pub sh_gu: B,                  // [s_cap * 2*inter] shared gate|up
    pub sh_h: B,                   // [s_cap * inter] shared swiglu output
    pub sh_q: cudarc::driver::CudaSlice<u8>,       // [s_cap * h] shared act-quant codes (x, then h)
    pub sh_qs: cudarc::driver::CudaSlice<u8>,      // [s_cap * h/128] shared act-quant scales
    pub sh_q2: cudarc::driver::CudaSlice<u8>,      // [s_cap * inter] shared h codes
    pub sh_qs2: cudarc::driver::CudaSlice<u8>,     // [s_cap * inter/128] shared h scales
    pub xc: B,                     // [d_cap * h] chunk x copy (the in-place sim guard)
    pub idc: cudarc::driver::CudaSlice<i32>,       // [d_cap * topk] chunk expert ids
    pub wtc: S,                    // [d_cap * topk] chunk routing weights
    pub outc: B,                   // [d_cap * h] chunk routed output
    pub gu: B,                     // [ppad_max * 2*inter] gate|up (grouped bound covers N=1 pairs)
    pub hh: B,                     // [ppad_max * inter] swiglu output
    pub dn: B,                     // [ppad_max * h] down output
    pub xperm: B,                  // [ppad_max * h] gathered activations (grouped path)
    pub sx: cudarc::driver::CudaSlice<u8>,         // [d_cap * (h/128)] sim scales x (written, never consumed)
    pub sh: cudarc::driver::CudaSlice<u8>,         // [ppad_max * (inter/128)] sim scales h (written, never consumed)
    pub ones: S,                   // [ppad_max] constant 1.0 (combine weight — was a per-call htod)
    pub zeros_host: Vec<i32>,      // [ne] count refill source (constant)
    pub neg1_host: Vec<i32>,       // [ppad_max] perm_tok refill source (constant)
}

/// R3A.4: widest routed-MoE dispatch (rows per `moe_routed` call). Prefill chunks at 4096
/// rows, so one dispatch per chunk per layer; decode/verify (≤16) is unaffected.
pub const MOE_D_CAP: usize = 4096;

/// `MoeGroupedScratch` without a qwen `Config` (DSV4 Phase-2 wiring; serving keeps building
/// it per-model via `new_moe_grouped_scratch`). `p_max` bounds batch·topk pairs per dispatch.
/// Buffers follow the struct's written-before-read discipline: `count`/`perm_tok` are refilled
/// by the caller every dispatch; the rest are written by the metadata kernels before any read.
/// R3.1: also builds the persistent `Dsv4MoeWorkspace` (`s_cap` = the largest FFN batch this
/// model will route — prefill chunk at serving, 16 for probes; `topk` = n_activated_experts).
pub fn new_moe_grouped_scratch_raw(
    dev: &Arc<CudaDevice>,
    ne: usize,
    h: usize,
    inter: usize,
    topk: usize,
    s_cap: usize,
    p_max: usize,
) -> MoeGroupedScratch {
    // E23: 32-padded experts (x4 grouped prefill tiles). E12's shared-expert region
    // (moe_offsets_fold_b appends it after the routed band; poff grows to ne+2) holds pad32(ns)
    // rows for the whole dispatch's tokens — pad32(p_max/topk), the worst case.
    let ppad_max = p_max + ne * 32 + ((p_max / topk + 31) / 32) * 32;
    let d_cap = s_cap.min(MOE_D_CAP);      // stage buffers: widest single dispatch (R3A.4)
    let ws = Dsv4MoeWorkspace {
        s_cap,
        scores: dev.alloc_zeros::<f32>(s_cap * ne).unwrap(),
        biased: dev.alloc_zeros::<f32>(s_cap * ne).unwrap(),
        routed_out: dev.alloc_zeros::<half::bf16>(s_cap * h).unwrap(),
        sh_out: dev.alloc_zeros::<half::bf16>(s_cap * h).unwrap(),
        sh_gu: dev.alloc_zeros::<half::bf16>(s_cap * 2 * inter).unwrap(),
        sh_h: dev.alloc_zeros::<half::bf16>(s_cap * inter).unwrap(),
        sh_q: dev.alloc_zeros::<u8>(s_cap * h).unwrap(),
        sh_qs: dev.alloc_zeros::<u8>(s_cap * h / 128).unwrap(),
        sh_q2: dev.alloc_zeros::<u8>(s_cap * inter).unwrap(),
        sh_qs2: dev.alloc_zeros::<u8>(s_cap * inter / 128).unwrap(),
        xc: dev.alloc_zeros::<half::bf16>(d_cap * h).unwrap(),
        idc: dev.alloc_zeros::<i32>(d_cap * topk).unwrap(),
        wtc: dev.alloc_zeros::<f32>(d_cap * topk).unwrap(),
        outc: dev.alloc_zeros::<half::bf16>(d_cap * h).unwrap(),
        gu: dev.alloc_zeros::<half::bf16>(ppad_max * 2 * inter).unwrap(),
        hh: dev.alloc_zeros::<half::bf16>(ppad_max * inter).unwrap(),
        dn: dev.alloc_zeros::<half::bf16>(ppad_max * h).unwrap(),
        xperm: dev.alloc_zeros::<half::bf16>(ppad_max * h).unwrap(),
        sx: dev.alloc_zeros::<u8>(d_cap * (h / 128)).unwrap(),
        sh: dev.alloc_zeros::<u8>(ppad_max * (inter / 128)).unwrap(),
        ones: {
            let o = dev.htod_sync_copy(&vec![1.0f32; ppad_max]).unwrap();
            o
        },
        zeros_host: vec![0i32; ne],
        neg1_host: vec![-1i32; ppad_max],
    };
    MoeGroupedScratch {
        count: dev.alloc_zeros::<i32>(ne).unwrap(),
        poff: dev.alloc_zeros::<i32>(ne + 2).unwrap(),
        cursor: dev.alloc_zeros::<i32>(ne).unwrap(),
        perm_tok: dev.alloc_zeros::<i32>(ppad_max).unwrap(),
        perm_wt: dev.alloc_zeros::<f32>(ppad_max).unwrap(),
        inv_pos: dev.alloc_zeros::<i32>(p_max).unwrap(),
        tile_e: dev.alloc_zeros::<i32>(ppad_max / 8 + 1).unwrap(),
        p_max,
        ppad_max,
        dsv4_ws: Some(Box::new(ws)),
    }
}

/// Launch a function from a name map on the default stream (DSV4 test-facing helpers).
macro_rules! dsv4_launch {
    ($funcs:expr, $name:expr, $g:expr, $b:expr, ($($a:expr),+ $(,)?)) => {{
        let f = $funcs.get($name).unwrap_or_else(|| panic!("dsv4 moe: kernel {}", $name)).clone();
        let (g0, g1, g2) = $g;
        let (b0, b1, b2) = $b;
        unsafe {
            f.launch(
                LaunchConfig { grid_dim: (g0, g1, g2), block_dim: (b0, b1, b2), shared_mem_bytes: 0 },
                ($($a),+),
            ).unwrap_or_else(|e| panic!("dsv4 moe: launch {}: {:?}", $name, e));
        }
    }};
}

/// Block count for `dsv4_act_quant_sim_g128`: one warp per (row, 128-wide group).
fn dsv4_sim_blocks(rows: usize, n: usize) -> u32 {
    ((rows * (n / 128) * 32 + 255) / 256) as u32
}

/// DSV4 routed-expert MoE, N=1 (decode): act-sim x → per-(token,slot) gate_up GEMV →
/// `dsv4_swiglu_clamp` (clamps + routing weight) → act-sim h → per-slot down GEMV →
/// slot-order fp32 combine. `x` is act-simmed IN PLACE (callers needing the original
/// must keep a copy — the Phase-3 router runs on the un-simmed x first).
#[allow(clippy::too_many_arguments)]
pub fn dsv4_moe_experts_n1(
    dev: &Arc<CudaDevice>,
    bk: &HashMap<String, CudaFunction>,
    df: &HashMap<String, CudaFunction>,
    moe: &Dsv4MoeGpu,
    batch: usize,
    topk: usize,
    swiglu_limit: f32,
    x: &B,                                // [batch, h]
    ids: &cudarc::driver::CudaSlice<i32>, // [batch*topk] global expert ids
    wts: &S,                              // [batch*topk] routing weights, fp32
    out: &mut B,                          // [batch, h]
) -> anyhow::Result<()> {
    let (ne, h, inter) = (moe.ne, moe.h, moe.inter);
    let (e_base, e_span) = (moe.e_base as i32, moe.e_span as i32);
    let pairs = batch * topk;
    // Scratch is alloc_zeros (NOT zeroed — AGENTS.md §2.2): every byte is written by its
    // producer kernel before any consumer reads it. The sim scale outputs are written
    // but never consumed (the simmed bf16 values carry the fp8 codes' content exactly).
    let s_x = dev.alloc_zeros::<u8>(batch * (h / 128))?;
    dsv4_launch!(df, "dsv4_act_quant_sim_g128", (dsv4_sim_blocks(batch, h), 1, 1), (256, 1, 1),
        (d(x), d(&s_x), batch as i32, h as i32));
    let gu = dev.alloc_zeros::<half::bf16>(pairs * 2 * inter)?;
    dsv4_launch!(bk, "gemm_moe_mma_fp4", (((2 * inter / 16) as u32), pairs as u32, 1), (256, 1, 1),
        (d(&gu), d(&moe.gu_wt), d(&moe.gu_st), d(&moe.gu_gs), d(x), d(ids),
         (2 * inter) as i32, h as i32, topk as i32, 0i32, e_base, e_span));
    let hh = dev.alloc_zeros::<half::bf16>(pairs * inter)?;
    dsv4_launch!(df, "dsv4_swiglu_clamp", (grid(pairs * inter)), (256, 1, 1),
        (d(&hh), d(&gu), d(wts), swiglu_limit, inter as i32, pairs as i32));
    let s_h = dev.alloc_zeros::<u8>(pairs * (inter / 128))?;
    dsv4_launch!(df, "dsv4_act_quant_sim_g128", (dsv4_sim_blocks(pairs, inter), 1, 1), (256, 1, 1),
        (d(&hh), d(&s_h), pairs as i32, inter as i32));
    let dn = dev.alloc_zeros::<half::bf16>(pairs * h)?;
    dsv4_launch!(bk, "gemm_moe_mma_fp4", (((h / 16) as u32), pairs as u32, 1), (256, 1, 1),
        (d(&dn), d(&moe.dn_wt), d(&moe.dn_st), d(&moe.dn_gs), d(&hh), d(ids),
         h as i32, inter as i32, topk as i32, 1i32, e_base, e_span));
    // Routing weight already folded into h by the swiglu kernel — combine with exact 1.0f.
    // Remote experts (outside [e_base,e_base+e_span)) contributed exact 0 above, so this is the
    // rank's routed PARTIAL (tp_all_reduce_bf16 restores the full Σ at the TP boundary).
    let ones: S = dev.htod_sync_copy(&vec![1.0f32; pairs])?;
    dsv4_launch!(bk, "moe_combine_experts_b", (grid(batch * h)), (256, 1, 1),
        (d(out), d(&dn), d(&ones), h as i32, topk as i32, batch as i32));
    Ok(())
}

/// DSV4 routed-expert MoE, grouped (verify N=2..16 / prefill): act-sim x → E16 device-only
/// metadata (count/offsets/scatter/tilemap/gather, static grids, zero readback) → grouped
/// gate_up GEMM → `dsv4_swiglu_clamp` (permuted routing weights) → act-sim h → grouped down
/// GEMM → slot-order fp32 combine. Bitwise-identical to `dsv4_moe_experts_n1` per output
/// element (see the section comment). `x` is act-simmed IN PLACE before the gather, so the
/// permuted rows carry the same simmed values the N=1 path reads.
#[allow(clippy::too_many_arguments)]
pub fn dsv4_moe_experts_grouped(
    dev: &Arc<CudaDevice>,
    bk: &HashMap<String, CudaFunction>,
    df: &HashMap<String, CudaFunction>,
    moe: &Dsv4MoeGpu,
    g: &mut MoeGroupedScratch,
    batch: usize,
    topk: usize,
    swiglu_limit: f32,
    x: &B,                                // [batch, h]
    ids: &cudarc::driver::CudaSlice<i32>, // [batch*topk]
    wts: &S,                              // [batch*topk]
    out: &mut B,                          // [batch, h]
) -> anyhow::Result<()> {
    let (ne, h, inter) = (moe.ne, moe.h, moe.inter);
    let (e_base, e_span) = (moe.e_base as i32, moe.e_span as i32);
    let (e_lo, e_hi) = (e_base, e_base + e_span);
    let p = batch * topk;
    assert!(p <= g.p_max, "dsv4 grouped MoE: {p} pairs > scratch p_max {}", g.p_max);
    let ppad = g.ppad_max;
    let s_x = dev.alloc_zeros::<u8>(batch * (h / 128))?;
    dsv4_launch!(df, "dsv4_act_quant_sim_g128", (dsv4_sim_blocks(batch, h), 1, 1), (256, 1, 1),
        (d(x), d(&s_x), batch as i32, h as i32));
    // E16 metadata (device-only; count refilled to 0, perm_tok to -1 = padding). TP=2: count/scatter
    // take this rank's band [e_lo,e_hi); remote pairs are not enqueued (inv_pos=-1, combine skips).
    dev.htod_sync_copy_into(&vec![0i32; ne], &mut g.count)?;
    dsv4_launch!(bk, "moe_count_b", (grid(p)), (256, 1, 1),
        (d(&g.count), d(ids), p as i32, e_lo, e_hi));
    dsv4_launch!(bk, "moe_offsets_b", (1u32, 1, 1), (32, 1, 1),
        (d(&g.poff), d(&g.cursor), d(&g.count), ne as i32));
    dev.htod_sync_copy_into(&vec![-1i32; ppad], &mut g.perm_tok)?;
    dsv4_launch!(bk, "moe_scatter_b", (grid(p)), (256, 1, 1),
        (d(&g.perm_tok), d(&g.perm_wt), d(&g.inv_pos), d(&g.cursor), d(ids), d(wts),
         p as i32, topk as i32, e_lo, e_hi));
    dsv4_launch!(bk, "moe_tilemap_b", (grid(ne)), (256, 1, 1),
        (d(&g.tile_e), d(&g.poff), ne as i32));
    // Gathered activations: every row written by moe_gather_x_b (0 for padding rows).
    let x_perm = dev.alloc_zeros::<half::bf16>(ppad * h)?;
    dsv4_launch!(bk, "moe_gather_x_b", (grid(ppad * h)), (256, 1, 1),
        (d(&x_perm), d(x), d(&g.perm_tok), h as i32, ppad as i32, d(&g.poff), ne as i32));
    // Grouped GEMMs write rows < poff[ne] only (static-grid early-exit); rows past it are
    // dead — swiglu/sim read them but their outputs are never consumed (combine reads only
    // real inv_pos rows), the exact pattern moe_silu_bf16_b already uses on this path.
    // TP=2: expert_base rebases tile_e's GLOBAL id to LOCAL; ne stays the global count (poff[ne] bound).
    let gu = dev.alloc_zeros::<half::bf16>(ppad * 2 * inter)?;
    dsv4_launch!(bk, "gemm_moe_grouped_mma_fp4",
        (((2 * inter / 16) as u32), (ppad / 16) as u32, 1), (256, 1, 1),
        (d(&gu), d(&moe.gu_wt), d(&moe.gu_st), d(&moe.gu_gs), d(&x_perm), d(&g.tile_e),
         (2 * inter) as i32, h as i32, e_base, d(&g.poff), ne as i32));
    let hh = dev.alloc_zeros::<half::bf16>(ppad * inter)?;
    dsv4_launch!(df, "dsv4_swiglu_clamp", (grid(ppad * inter)), (256, 1, 1),
        (d(&hh), d(&gu), d(&g.perm_wt), swiglu_limit, inter as i32, ppad as i32));
    let s_h = dev.alloc_zeros::<u8>(ppad * (inter / 128))?;
    dsv4_launch!(df, "dsv4_act_quant_sim_g128", (dsv4_sim_blocks(ppad, inter), 1, 1), (256, 1, 1),
        (d(&hh), d(&s_h), ppad as i32, inter as i32));
    let dn = dev.alloc_zeros::<half::bf16>(ppad * h)?;
    dsv4_launch!(bk, "gemm_moe_grouped_mma_fp4",
        (((h / 16) as u32), (ppad / 16) as u32, 1), (256, 1, 1),
        (d(&dn), d(&moe.dn_wt), d(&moe.dn_st), d(&moe.dn_gs), d(&hh), d(&g.tile_e),
         h as i32, inter as i32, e_base, d(&g.poff), ne as i32));
    // Remote pairs carry inv_pos = -1 (combine skips them); this is the rank's routed PARTIAL.
    let ones: S = dev.htod_sync_copy(&vec![1.0f32; ppad])?;
    dsv4_launch!(bk, "moe_combine_grouped_b", (grid(batch * h)), (256, 1, 1),
        (d(out), d(&dn), d(&ones), d(&g.inv_pos), h as i32, topk as i32, batch as i32));
    Ok(())
}

// -----------------------------------------------------------------------------------------------
// R3.1 (L1) — the serving MoE path on the compute stream with the persistent workspace.
// These are the SAME kernels with the SAME launch configs as the Phase-2 helpers above
// (bitwise-identical per element — only the stream and the buffer provenance changed): the
// expert path leaves the legacy default stream (which serializes against the blocking compute
// stream — the R3.0 ncu stream evidence: MoE kernels on stream 7, everything else on 14) and
// stops allocating per dispatch. All launches go through `launch_on_stream`; the two metadata
// refills are `memcpy_htod_async` on the same stream from CONSTANT host vectors.
// -----------------------------------------------------------------------------------------------

/// Launch from a name map on an EXPLICIT stream (the serving MoE path — contrast `dsv4_launch!`).
macro_rules! dsv4_launch_ws {
    ($funcs:expr, $name:expr, $stream:expr, $g:expr, $b:expr, ($($a:expr),+ $(,)?)) => {{
        let f = $funcs.get($name).unwrap_or_else(|| panic!("dsv4 moe: kernel {}", $name)).clone();
        let (g0, g1, g2) = $g;
        let (b0, b1, b2) = $b;
        unsafe {
            f.launch_on_stream(
                $stream,
                LaunchConfig { grid_dim: (g0, g1, g2), block_dim: (b0, b1, b2), shared_mem_bytes: 0 },
                ($($a),+),
            ).unwrap_or_else(|e| panic!("dsv4 moe: launch {}: {:?}", $name, e));
        }
    }};
}

/// DSV4 routed-expert MoE, N=1 (decode), workspace+stream variant of [`dsv4_moe_experts_n1`].
/// Reads `ws.xc`/`ws.idc`/`ws.wtc` (pre-staged by the caller — the helper sims xc IN PLACE),
/// writes `ws.outc`. Identical kernels/launches to the reference helper, so per-element bitwise.
#[allow(clippy::too_many_arguments)]
pub fn dsv4_moe_experts_n1_ws(
    dev: &Arc<CudaDevice>,
    stream: &cudarc::driver::CudaStream,
    bk: &HashMap<String, CudaFunction>,
    df: &HashMap<String, CudaFunction>,
    moe: &Dsv4MoeGpu,
    ws: &mut Dsv4MoeWorkspace,
    batch: usize,
    topk: usize,
    swiglu_limit: f32,
) -> anyhow::Result<()> {
    let _ = dev;
    let (ne, h, inter) = (moe.ne, moe.h, moe.inter);
    let (e_base, e_span) = (moe.e_base as i32, moe.e_span as i32);
    let pairs = batch * topk;
    assert!(batch <= 16 && pairs <= ws.ones.len(), "dsv4 n1 ws: batch {batch} pairs {pairs} over workspace");
    dsv4_launch_ws!(df, "dsv4_act_quant_sim_g128", stream, (dsv4_sim_blocks(batch, h), 1, 1), (256, 1, 1),
        (d(&ws.xc), d(&ws.sx), batch as i32, h as i32));
    dsv4_launch_ws!(bk, "gemm_moe_mma_fp4", stream, (((2 * inter / 16) as u32), pairs as u32, 1), (256, 1, 1),
        (d(&ws.gu), d(&moe.gu_wt), d(&moe.gu_st), d(&moe.gu_gs), d(&ws.xc), d(&ws.idc),
         (2 * inter) as i32, h as i32, topk as i32, 0i32, e_base, e_span));
    dsv4_launch_ws!(df, "dsv4_swiglu_clamp", stream, (grid(pairs * inter)), (256, 1, 1),
        (d(&ws.hh), d(&ws.gu), d(&ws.wtc), swiglu_limit, inter as i32, pairs as i32));
    dsv4_launch_ws!(df, "dsv4_act_quant_sim_g128", stream, (dsv4_sim_blocks(pairs, inter), 1, 1), (256, 1, 1),
        (d(&ws.hh), d(&ws.sh), pairs as i32, inter as i32));
    dsv4_launch_ws!(bk, "gemm_moe_mma_fp4", stream, (((h / 16) as u32), pairs as u32, 1), (256, 1, 1),
        (d(&ws.dn), d(&moe.dn_wt), d(&moe.dn_st), d(&moe.dn_gs), d(&ws.hh), d(&ws.idc),
         h as i32, inter as i32, topk as i32, 1i32, e_base, e_span));
    // Routing weight already folded into h — combine with the persistent exact-1.0f weights.
    let _ = ne;
    dsv4_launch_ws!(bk, "moe_combine_experts_b", stream, (grid(batch * h)), (256, 1, 1),
        (d(&ws.outc), d(&ws.dn), d(&ws.ones), h as i32, topk as i32, batch as i32));
    Ok(())
}

/// DSV4 routed-expert MoE, grouped (verify N=2..16 / prefill chunks), workspace+stream variant
/// of [`dsv4_moe_experts_grouped`]. Reads `ws.xc`/`ws.idc`/`ws.wtc`, writes `ws.outc` (first
/// `batch * h` rows — the combine reads only real inv_pos rows, same as the reference).
#[allow(clippy::too_many_arguments)]
pub fn dsv4_moe_experts_grouped_ws(
    dev: &Arc<CudaDevice>,
    stream: &cudarc::driver::CudaStream,
    bk: &HashMap<String, CudaFunction>,
    df: &HashMap<String, CudaFunction>,
    moe: &Dsv4MoeGpu,
    g: &mut MoeGroupedScratch,
    batch: usize,
    topk: usize,
    swiglu_limit: f32,
) -> anyhow::Result<()> {
    let (ne, h, inter) = (moe.ne, moe.h, moe.inter);
    let (e_base, e_span) = (moe.e_base as i32, moe.e_span as i32);
    let (e_lo, e_hi) = (e_base, e_base + e_span);
    let p = batch * topk;
    assert!(p <= g.p_max, "dsv4 grouped MoE: {p} pairs > scratch p_max {}", g.p_max);
    // R3A.4: per-call padded bound (each touched expert pads to 16; at most min(e_span, p)
    // experts are touched) — the launch grid shrinks to the real work instead of the static
    // ppad_max (the buffers stay ppad_max-sized; the GEMM early-exits past poff[ne] on device).
    let ppad = p + (e_span as usize).min(p) * 16;
    // Split the field borrows ONCE: the metadata buffers (count/poff/...) and the workspace
    // are disjoint fields, usable side by side in the same launch tuples.
    let MoeGroupedScratch { count, poff, cursor, perm_tok, perm_wt, inv_pos, tile_e, dsv4_ws, .. } = &mut *g;
    let ws = dsv4_ws.as_mut().expect("dsv4 moe workspace").as_mut();
    dsv4_launch_ws!(df, "dsv4_act_quant_sim_g128", stream, (dsv4_sim_blocks(batch, h), 1, 1), (256, 1, 1),
        (d(&ws.xc), d(&ws.sx), batch as i32, h as i32));
    // E16 metadata refills — CONSTANT host sources, async on the compute stream (no host sync).
    unsafe {
        cudarc::driver::result::memcpy_htod_async(*count.device_ptr(), &ws.zeros_host, stream.stream)
            .map_err(|e| anyhow::anyhow!("count refill: {e}"))?;
        cudarc::driver::result::memcpy_htod_async(*perm_tok.device_ptr(), &ws.neg1_host, stream.stream)
            .map_err(|e| anyhow::anyhow!("perm_tok refill: {e}"))?;
    }
    dsv4_launch_ws!(bk, "moe_count_b", stream, (grid(p)), (256, 1, 1),
        (d(count), d(&ws.idc), p as i32, e_lo, e_hi));
    dsv4_launch_ws!(bk, "moe_offsets_b", stream, (1u32, 1, 1), (32, 1, 1),
        (d(poff), d(cursor), d(count), ne as i32));
    dsv4_launch_ws!(bk, "moe_scatter_b", stream, (grid(p)), (256, 1, 1),
        (d(perm_tok), d(perm_wt), d(inv_pos), d(cursor), d(&ws.idc), d(&ws.wtc),
         p as i32, topk as i32, e_lo, e_hi));
    dsv4_launch_ws!(bk, "moe_tilemap_b", stream, (grid(ne)), (256, 1, 1),
        (d(tile_e), d(poff), ne as i32));
    dsv4_launch_ws!(bk, "moe_gather_x_b", stream, (grid(ppad * h)), (256, 1, 1),
        (d(&ws.xperm), d(&ws.xc), d(perm_tok), h as i32, ppad as i32, d(poff), ne as i32));
    dsv4_launch_ws!(bk, "gemm_moe_grouped_mma_fp4", stream,
        (((2 * inter / 16) as u32), (ppad / 16) as u32, 1), (256, 1, 1),
        (d(&ws.gu), d(&moe.gu_wt), d(&moe.gu_st), d(&moe.gu_gs), d(&ws.xperm), d(tile_e),
         (2 * inter) as i32, h as i32, e_base, d(poff), ne as i32));
    dsv4_launch_ws!(df, "dsv4_swiglu_clamp", stream, (grid(ppad * inter)), (256, 1, 1),
        (d(&ws.hh), d(&ws.gu), d(perm_wt), swiglu_limit, inter as i32, ppad as i32));
    dsv4_launch_ws!(df, "dsv4_act_quant_sim_g128", stream, (dsv4_sim_blocks(ppad, inter), 1, 1), (256, 1, 1),
        (d(&ws.hh), d(&ws.sh), ppad as i32, inter as i32));
    dsv4_launch_ws!(bk, "gemm_moe_grouped_mma_fp4", stream,
        (((h / 16) as u32), (ppad / 16) as u32, 1), (256, 1, 1),
        (d(&ws.dn), d(&moe.dn_wt), d(&moe.dn_st), d(&moe.dn_gs), d(&ws.hh), d(tile_e),
         h as i32, inter as i32, e_base, d(poff), ne as i32));
    dsv4_launch_ws!(bk, "moe_combine_grouped_b", stream, (grid(batch * h)), (256, 1, 1),
        (d(&ws.outc), d(&ws.dn), d(&ws.ones), d(inv_pos), h as i32, topk as i32, batch as i32));
    let _ = dev;
    Ok(())
}

// -----------------------------------------------------------------------------------------------
// x2-tile streaming variants (queue #3, 2026-07-31): SAME metadata pipeline and launch sequence as
// the single-tile helpers above, but the two GEMM kernels are `*_x2` (two 16-row weight tiles per
// CTA, grid.x = M/32) — per-element K chains untouched, so every bitwise gate (N=1 vs grouped at
// every N in 2..=16, wide-vs-chunked, serving hashes) stays the contract. The single-tile helpers
// remain registered for the bitwise equivalence gates.
// -----------------------------------------------------------------------------------------------

/// x2-tile variant of [`dsv4_moe_experts_n1_ws`]: `gemm_moe_mma_fp4_x2`, grid.x = (2*inter)/32.
#[allow(clippy::too_many_arguments)]
pub fn dsv4_moe_experts_n1_ws_x2(
    dev: &Arc<CudaDevice>,
    stream: &cudarc::driver::CudaStream,
    bk: &HashMap<String, CudaFunction>,
    df: &HashMap<String, CudaFunction>,
    moe: &Dsv4MoeGpu,
    ws: &mut Dsv4MoeWorkspace,
    batch: usize,
    topk: usize,
    swiglu_limit: f32,
) -> anyhow::Result<()> {
    let _ = dev;
    let (ne, h, inter) = (moe.ne, moe.h, moe.inter);
    let (e_base, e_span) = (moe.e_base as i32, moe.e_span as i32);
    let pairs = batch * topk;
    assert!(batch <= 16 && pairs <= ws.ones.len(), "dsv4 n1 ws: batch {batch} pairs {pairs} over workspace");
    dsv4_launch_ws!(df, "dsv4_act_quant_sim_g128", stream, (dsv4_sim_blocks(batch, h), 1, 1), (256, 1, 1),
        (d(&ws.xc), d(&ws.sx), batch as i32, h as i32));
    dsv4_launch_ws!(bk, "gemm_moe_mma_fp4_x2", stream, (((2 * inter / 32) as u32), pairs as u32, 1), (256, 1, 1),
        (d(&ws.gu), d(&moe.gu_wt), d(&moe.gu_st), d(&moe.gu_gs), d(&ws.xc), d(&ws.idc),
         (2 * inter) as i32, h as i32, topk as i32, 0i32, e_base, e_span));
    dsv4_launch_ws!(df, "dsv4_swiglu_clamp", stream, (grid(pairs * inter)), (256, 1, 1),
        (d(&ws.hh), d(&ws.gu), d(&ws.wtc), swiglu_limit, inter as i32, pairs as i32));
    dsv4_launch_ws!(df, "dsv4_act_quant_sim_g128", stream, (dsv4_sim_blocks(pairs, inter), 1, 1), (256, 1, 1),
        (d(&ws.hh), d(&ws.sh), pairs as i32, inter as i32));
    dsv4_launch_ws!(bk, "gemm_moe_mma_fp4_x2", stream, (((h / 32) as u32), pairs as u32, 1), (256, 1, 1),
        (d(&ws.dn), d(&moe.dn_wt), d(&moe.dn_st), d(&moe.dn_gs), d(&ws.hh), d(&ws.idc),
         h as i32, inter as i32, topk as i32, 1i32, e_base, e_span));
    dsv4_launch_ws!(bk, "moe_combine_experts_b", stream, (grid(batch * h)), (256, 1, 1),
        (d(&ws.outc), d(&ws.dn), d(&ws.ones), h as i32, topk as i32, batch as i32));
    let _ = ne;
    Ok(())
}

/// x2-tile variant of [`dsv4_moe_experts_grouped_ws`]: `gemm_moe_grouped_mma_fp4_x2`, grid.x = M/32.
#[allow(clippy::too_many_arguments)]
pub fn dsv4_moe_experts_grouped_ws_x2(
    dev: &Arc<CudaDevice>,
    stream: &cudarc::driver::CudaStream,
    bk: &HashMap<String, CudaFunction>,
    df: &HashMap<String, CudaFunction>,
    moe: &Dsv4MoeGpu,
    g: &mut MoeGroupedScratch,
    batch: usize,
    topk: usize,
    swiglu_limit: f32,
) -> anyhow::Result<()> {
    let (ne, h, inter) = (moe.ne, moe.h, moe.inter);
    let (e_base, e_span) = (moe.e_base as i32, moe.e_span as i32);
    let (e_lo, e_hi) = (e_base, e_base + e_span);
    let p = batch * topk;
    assert!(p <= g.p_max, "dsv4 grouped MoE: {p} pairs > scratch p_max {}", g.p_max);
    let ppad = p + (e_span as usize).min(p) * 16;
    let MoeGroupedScratch { count, poff, cursor, perm_tok, perm_wt, inv_pos, tile_e, dsv4_ws, .. } = &mut *g;
    let ws = dsv4_ws.as_mut().expect("dsv4 moe workspace").as_mut();
    dsv4_launch_ws!(df, "dsv4_act_quant_sim_g128", stream, (dsv4_sim_blocks(batch, h), 1, 1), (256, 1, 1),
        (d(&ws.xc), d(&ws.sx), batch as i32, h as i32));
    unsafe {
        cudarc::driver::result::memcpy_htod_async(*count.device_ptr(), &ws.zeros_host, stream.stream)
            .map_err(|e| anyhow::anyhow!("count refill: {e}"))?;
        cudarc::driver::result::memcpy_htod_async(*perm_tok.device_ptr(), &ws.neg1_host, stream.stream)
            .map_err(|e| anyhow::anyhow!("perm_tok refill: {e}"))?;
    }
    dsv4_launch_ws!(bk, "moe_count_b", stream, (grid(p)), (256, 1, 1),
        (d(count), d(&ws.idc), p as i32, e_lo, e_hi));
    dsv4_launch_ws!(bk, "moe_offsets_b", stream, (1u32, 1, 1), (32, 1, 1),
        (d(poff), d(cursor), d(count), ne as i32));
    dsv4_launch_ws!(bk, "moe_scatter_b", stream, (grid(p)), (256, 1, 1),
        (d(perm_tok), d(perm_wt), d(inv_pos), d(cursor), d(&ws.idc), d(&ws.wtc),
         p as i32, topk as i32, e_lo, e_hi));
    dsv4_launch_ws!(bk, "moe_tilemap_b", stream, (grid(ne)), (256, 1, 1),
        (d(tile_e), d(poff), ne as i32));
    dsv4_launch_ws!(bk, "moe_gather_x_b", stream, (grid(ppad * h)), (256, 1, 1),
        (d(&ws.xperm), d(&ws.xc), d(perm_tok), h as i32, ppad as i32, d(poff), ne as i32));
    dsv4_launch_ws!(bk, "gemm_moe_grouped_mma_fp4_x2", stream,
        (((2 * inter / 32) as u32), (ppad / 16) as u32, 1), (256, 1, 1),
        (d(&ws.gu), d(&moe.gu_wt), d(&moe.gu_st), d(&moe.gu_gs), d(&ws.xperm), d(tile_e),
         (2 * inter) as i32, h as i32, e_base, d(poff), ne as i32));
    dsv4_launch_ws!(df, "dsv4_swiglu_clamp", stream, (grid(ppad * inter)), (256, 1, 1),
        (d(&ws.hh), d(&ws.gu), d(perm_wt), swiglu_limit, inter as i32, ppad as i32));
    dsv4_launch_ws!(df, "dsv4_act_quant_sim_g128", stream, (dsv4_sim_blocks(ppad, inter), 1, 1), (256, 1, 1),
        (d(&ws.hh), d(&ws.sh), ppad as i32, inter as i32));
    dsv4_launch_ws!(bk, "gemm_moe_grouped_mma_fp4_x2", stream,
        (((h / 32) as u32), (ppad / 16) as u32, 1), (256, 1, 1),
        (d(&ws.dn), d(&moe.dn_wt), d(&moe.dn_st), d(&moe.dn_gs), d(&ws.hh), d(tile_e),
         h as i32, inter as i32, e_base, d(poff), ne as i32));
    dsv4_launch_ws!(bk, "moe_combine_grouped_b", stream, (grid(batch * h)), (256, 1, 1),
        (d(&ws.outc), d(&ws.dn), d(&ws.ones), d(inv_pos), h as i32, topk as i32, batch as i32));
    let _ = dev;
    Ok(())
}

// -----------------------------------------------------------------------------------------------
// u4 burst-load variants (queue #3 probe): SAME launch sequence as the single-tile helpers, but the
// two GEMM kernels are `*_u4` (four pair-iterations' loads hoisted per loop step; per-element K
// chains untouched). Same bitwise contract as the x2 twins.
// -----------------------------------------------------------------------------------------------

/// u4 variant of [`dsv4_moe_experts_n1_ws`]: `gemm_moe_mma_fp4_u4` (same grid as the single-tile).
#[allow(clippy::too_many_arguments)]
pub fn dsv4_moe_experts_n1_ws_u4(
    dev: &Arc<CudaDevice>,
    stream: &cudarc::driver::CudaStream,
    bk: &HashMap<String, CudaFunction>,
    df: &HashMap<String, CudaFunction>,
    moe: &Dsv4MoeGpu,
    ws: &mut Dsv4MoeWorkspace,
    batch: usize,
    topk: usize,
    swiglu_limit: f32,
) -> anyhow::Result<()> {
    let _ = dev;
    let (ne, h, inter) = (moe.ne, moe.h, moe.inter);
    let (e_base, e_span) = (moe.e_base as i32, moe.e_span as i32);
    let pairs = batch * topk;
    assert!(batch <= 16 && pairs <= ws.ones.len(), "dsv4 n1 ws: batch {batch} pairs {pairs} over workspace");
    dsv4_launch_ws!(df, "dsv4_act_quant_sim_g128", stream, (dsv4_sim_blocks(batch, h), 1, 1), (256, 1, 1),
        (d(&ws.xc), d(&ws.sx), batch as i32, h as i32));
    dsv4_launch_ws!(bk, "gemm_moe_mma_fp4_u4", stream, (((2 * inter / 16) as u32), pairs as u32, 1), (256, 1, 1),
        (d(&ws.gu), d(&moe.gu_wt), d(&moe.gu_st), d(&moe.gu_gs), d(&ws.xc), d(&ws.idc),
         (2 * inter) as i32, h as i32, topk as i32, 0i32, e_base, e_span));
    dsv4_launch_ws!(df, "dsv4_swiglu_clamp", stream, (grid(pairs * inter)), (256, 1, 1),
        (d(&ws.hh), d(&ws.gu), d(&ws.wtc), swiglu_limit, inter as i32, pairs as i32));
    dsv4_launch_ws!(df, "dsv4_act_quant_sim_g128", stream, (dsv4_sim_blocks(pairs, inter), 1, 1), (256, 1, 1),
        (d(&ws.hh), d(&ws.sh), pairs as i32, inter as i32));
    dsv4_launch_ws!(bk, "gemm_moe_mma_fp4_u4", stream, (((h / 16) as u32), pairs as u32, 1), (256, 1, 1),
        (d(&ws.dn), d(&moe.dn_wt), d(&moe.dn_st), d(&moe.dn_gs), d(&ws.hh), d(&ws.idc),
         h as i32, inter as i32, topk as i32, 1i32, e_base, e_span));
    dsv4_launch_ws!(bk, "moe_combine_experts_b", stream, (grid(batch * h)), (256, 1, 1),
        (d(&ws.outc), d(&ws.dn), d(&ws.ones), h as i32, topk as i32, batch as i32));
    let _ = ne;
    Ok(())
}

/// u4 variant of [`dsv4_moe_experts_grouped_ws`]: `gemm_moe_grouped_mma_fp4_u4`.
#[allow(clippy::too_many_arguments)]
pub fn dsv4_moe_experts_grouped_ws_u4(
    dev: &Arc<CudaDevice>,
    stream: &cudarc::driver::CudaStream,
    bk: &HashMap<String, CudaFunction>,
    df: &HashMap<String, CudaFunction>,
    moe: &Dsv4MoeGpu,
    g: &mut MoeGroupedScratch,
    batch: usize,
    topk: usize,
    swiglu_limit: f32,
) -> anyhow::Result<()> {
    let (ne, h, inter) = (moe.ne, moe.h, moe.inter);
    let (e_base, e_span) = (moe.e_base as i32, moe.e_span as i32);
    let (e_lo, e_hi) = (e_base, e_base + e_span);
    let p = batch * topk;
    assert!(p <= g.p_max, "dsv4 grouped MoE: {p} pairs > scratch p_max {}", g.p_max);
    let ppad = p + (e_span as usize).min(p) * 16;
    let MoeGroupedScratch { count, poff, cursor, perm_tok, perm_wt, inv_pos, tile_e, dsv4_ws, .. } = &mut *g;
    let ws = dsv4_ws.as_mut().expect("dsv4 moe workspace").as_mut();
    dsv4_launch_ws!(df, "dsv4_act_quant_sim_g128", stream, (dsv4_sim_blocks(batch, h), 1, 1), (256, 1, 1),
        (d(&ws.xc), d(&ws.sx), batch as i32, h as i32));
    unsafe {
        cudarc::driver::result::memcpy_htod_async(*count.device_ptr(), &ws.zeros_host, stream.stream)
            .map_err(|e| anyhow::anyhow!("count refill: {e}"))?;
        cudarc::driver::result::memcpy_htod_async(*perm_tok.device_ptr(), &ws.neg1_host, stream.stream)
            .map_err(|e| anyhow::anyhow!("perm_tok refill: {e}"))?;
    }
    dsv4_launch_ws!(bk, "moe_count_b", stream, (grid(p)), (256, 1, 1),
        (d(count), d(&ws.idc), p as i32, e_lo, e_hi));
    dsv4_launch_ws!(bk, "moe_offsets_b", stream, (1u32, 1, 1), (32, 1, 1),
        (d(poff), d(cursor), d(count), ne as i32));
    dsv4_launch_ws!(bk, "moe_scatter_b", stream, (grid(p)), (256, 1, 1),
        (d(perm_tok), d(perm_wt), d(inv_pos), d(cursor), d(&ws.idc), d(&ws.wtc),
         p as i32, topk as i32, e_lo, e_hi));
    dsv4_launch_ws!(bk, "moe_tilemap_b", stream, (grid(ne)), (256, 1, 1),
        (d(tile_e), d(poff), ne as i32));
    dsv4_launch_ws!(bk, "moe_gather_x_b", stream, (grid(ppad * h)), (256, 1, 1),
        (d(&ws.xperm), d(&ws.xc), d(perm_tok), h as i32, ppad as i32, d(poff), ne as i32));
    dsv4_launch_ws!(bk, "gemm_moe_grouped_mma_fp4_u4", stream,
        (((2 * inter / 16) as u32), (ppad / 16) as u32, 1), (256, 1, 1),
        (d(&ws.gu), d(&moe.gu_wt), d(&moe.gu_st), d(&moe.gu_gs), d(&ws.xperm), d(tile_e),
         (2 * inter) as i32, h as i32, e_base, d(poff), ne as i32));
    dsv4_launch_ws!(df, "dsv4_swiglu_clamp", stream, (grid(ppad * inter)), (256, 1, 1),
        (d(&ws.hh), d(&ws.gu), d(perm_wt), swiglu_limit, inter as i32, ppad as i32));
    dsv4_launch_ws!(df, "dsv4_act_quant_sim_g128", stream, (dsv4_sim_blocks(ppad, inter), 1, 1), (256, 1, 1),
        (d(&ws.hh), d(&ws.sh), ppad as i32, inter as i32));
    dsv4_launch_ws!(bk, "gemm_moe_grouped_mma_fp4_u4", stream,
        (((h / 16) as u32), (ppad / 16) as u32, 1), (256, 1, 1),
        (d(&ws.dn), d(&moe.dn_wt), d(&moe.dn_st), d(&moe.dn_gs), d(&ws.hh), d(tile_e),
         h as i32, inter as i32, e_base, d(poff), ne as i32));
    dsv4_launch_ws!(bk, "moe_combine_grouped_b", stream, (grid(batch * h)), (256, 1, 1),
        (d(&ws.outc), d(&ws.dn), d(&ws.ones), d(inv_pos), h as i32, topk as i32, batch as i32));
    let _ = dev;
    Ok(())
}
