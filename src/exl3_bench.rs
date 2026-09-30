//! S-A3-c — EXL3 weight-path GEMM: kernel decode gate (G-A3-1 through the kernel),
//! batch-invariance probe (G-A3-5), and the standalone bench.
//!
//! Probe-only: loads `src/ptx/exl3_bench.ptx` lazily on a bare `CudaDevice`
//! (mxfp4_bench precedent, gpu.rs). No model load, no serving path, no TP.
//! The pack is read tensor-wise via `crate::exl3` (loader semantics untouched).
//!
//! Flags (wired in main.rs):
//!   --probe-exl3-kernel --model-dir <pack>   decode dump vs Rust oracle, bitwise
//!   --probe-exl3-binv   --model-dir <pack>   y_raw/y row 0 bit-identical, widths {1,2,4,8}
//!   --bench-exl3-gemm   --model-dir <pack>   per-class GB/s at widths, vs 238 roofline
//!
//! Numerics contract: the kernel's decode is bitwise vs `decode_trellis_host`
//! (which is G-A3-1-green vs ext.reconstruct). The GEMM host reference mirrors the
//! kernel exactly: same Walsh butterfly order (fp32 pairwise, ascending len) for the
//! Hadamards; the GEMM itself is compared after rounding the f32 reference to fp16
//! (residual = mma association order only, threshold 1e-5 rel-L2; raw f32 diff also
//! reported). Batch invariance: M zero-pads a fixed m16 tile; the K loop and the
//! mma's per-element reduction order are width-independent — verified empirically.

use crate::exl3::{decode_trellis_host, tmap, Exl3Pack, ModuleMeta, TensorClass};
use crate::gpu::fork_blocking_stream;
use anyhow::{bail, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use half::f16;

const MODULE: &str = "exl3_bench";
const MODULE_FNS: &[&str] = &[
    "exl3_decode_dump",
    // S-A3-u G: coalesced reconstruct (same values; nb % 8 == 0)
    "exl3_decode_dump8",
    // TUNE T0: stale-kernel handshake (FwdModel::load launches it; §14.1)
    "kernel_build_id",
    // TUNE T1: --autotune harness only (L2 conditioning, write-set digest, known-positive spin)
    "xq_l2_touch",
    "xq_digest64",
    "xq_spin_ns",
    "exl3_hmma_gemm",
    "exl3_hmma_gemm_grouped",
    // S-A3-f-b: wide-M GEMM (chunked prefill); gated by --probe-exl3-wide
    "exl3_hmma_gemm_wide",
    "exl3_had_suh",
    "exl3_had_svh",
    // S-A3-d forward kernels (--bench-exl3-forward)
    "xq_gemm_f16",
    // S-A3-m: row-batched bit-exact twin (PLE projections)
    "xq_gemm_f16_rows",
    "xq_gemm_f16_f32",
    "xq_hc_norm",
    "xq_silu_div",
    "xq_hc_mix",
    "xq_hc_inject",
    "xq_conv1d",
    "xq_gdn_step",
    "xq_gdn_gate",
    "xq_router_topk",
    "xq_moe_gate_mul",
    "xq_silu_mul",
    "xq_moe_combine",
    "xq_argmax",
    // S-A3-e: MTP draft-head glue
    "xq_add_f16_f32",
    // S-A3-e: per-slot state reset
    "xq_memset_f32",
    // S-A3-e: NaN-poison probe (--exl3-poison)
    "xq_memset_u16",
    // S-A3-f Item 1: per-step RoPE gather from the precomputed host-exact table
    "xq_cos_gather",
    // S-A3-f Item 2: device-side MoE routing (route kernel + padded consumer grids)
    "xq_moe_route",
    // TP-A (EXL3 TP=2 rung 3): expert-parallel route filter + FP32 partial combine + the
    // single f16 rounding after the all-reduce
    "xq_moe_route_ep",
    "xq_moe_combine_ep",
    "xq_cvt_f32_f16",
    // TP-B (rung 4): row-parallel FP32 split-K combine / widening + EP prefill routing/combine
    "xq_ks_combine_f32",
    "xq_cvt_f16_f32",
    // TP-G: decode all-reduce with K1 folded into the producers + GPU-side receive; EP route fold
    "xq_ks_combine_f32_k1l",
    "xq_moe_combine_ep_k1l",
    "xq_cvt_f16_f32_k1l",
    "xq_tp_wait_add_dec",
    // TP-I3 (c1): K2m folded into the prefill consumers (--tp-prefill-overlap fold)
    "xq_pf_hc_inj_norm_k2",
    "exl3_had_svh_k2",
    // TP-H #4: vocab-parallel greedy tail (argmax keys with K1 folded + key-merge K2)
    "xq_argmax_rows_vp",
    "xq_tp_wait_keys",
    // TP-H2: vocab-parallel sampled rows (row all-gather K1 / K2 of the shard logits; --tp-vp-sampled)
    "xq_vp_gather_k1",
    "xq_vp_gather_k2",
    // TP-H #3: sharded draft-head screen (K1 folded into the last block; K2 = xq_tp_wait_add_dec)
    "xq_dh_screen_k1",
    "xq_router_fold_c_ep",
    "xq_moe_ids_ep",
    "xq_moe_pf_ep_fill",
    "xq_moe_combine_rows_ep",
    // S-A3-o: live-expert histogram diagnostic (--exl3-esel-hist)
    "xq_esel_hist",
    // TP-I #6: per-(layer, expert) routing histogram diagnostic (--tp-ep-hist)
    "xq_expert_hist",
    // S-A3-s: per-position QSA selection scatter (--qsa-rowdump-layer diagnostic)
    "xq_dbg_scatter_sel",
    "xq_embed_resid",
    "xq_had_suh_multi",
    "xq_had_svh_multi",
    "xq_gemm_grouped_xh",
    // S-A3-o O3: dedicated A-once 3-bit expert entry (3 CTAs/SM)
    "xq_gemm_grouped_a1b3",
    // A5 WP20: word-diet twin of the a1b3 entry + the suh-convention calibration kernel (the
    // MoE epilogue kernels xq_moe_{gu,dn}_epi[_sh] take > 12 params: raw launches, unlisted).
    "xq_gemm_grouped_a1b3_wd",
    "xq_wp20_suh_cal",
    "xq_attn_decode",
    // S-A3-f-b: chunked prefill (wide-M + in-kernel token loops)
    "xq_conv1d_chunk",
    "xq_gdn_step_chunk",
    "xq_attn_kv_prefill",
    // PQ8 item 1: row-parallel twin of xq_attn_kv_prefill (grid c*nkv, bit-identical bytes)
    "xq_attn_kv_prefill_rows",
    "xq_attn_prefill_q",
    // S-A3-f-g: flash-style prefill attention (q-tile blocks + SMEM K/V staging)
    "xq_attn_prefill_flash",
    "xq_attn_prefill_flash256",
    // PQ8 item 2: bit-identical flash256 twin (vector staging + float4 scores)
    "xq_attn_prefill_flash256v",
    // PQ8-v2 opt-in (--pq8-flash=causal): flash256v with causal (block-wide) K/V staging
    "xq_attn_prefill_flash256c",
    // w5/PFIX opt-in (same --pq8-flash=causal): the hd=128 causal-staging twin
    "xq_attn_prefill_flash128c",
    "xq_ple_conv_chunk",
    "xq_had_suh_rows",
    "xq_had_svh_rows",
    "xq_gemm_grouped_rows",
    "xq_moe_combine_rows",
    "xq_copy_row",
    // S-A3-f-f ITEM 1/2: fused hc-mixer (down+silu+up, grid barrier), vectorized a/b,
    // bit-exact fp32-out router, split-K chain GEMM + combine (xq_grid_barrier is a
    // __device__ helper — no entry, not registered).
    "xq_hc_fuse",
    // S-A3-q P1: grouped prefill MoE wide GEMM (one launch per side)
    "exl3_hmma_gemm_wide_grouped",
    // S-A3-u D: device-side prefill MoE routing + tile-list grouped GEMM
    "xq_moe_pf_count",
    "xq_moe_pf_place",
    "exl3_hmma_gemm_wide_tiles",
    // A5-P1: pipelined per-MT prefill MoE expert twins (+ svh / gate|up glue epilogues), WP19 lists
    "xq_moe_pf_t1", "xq_moe_pf_t2", "xq_moe_pf_t4", "xq_moe_pf_t8",
    "xq_moe_pf_s1", "xq_moe_pf_s2", "xq_moe_pf_s4", "xq_moe_pf_s8",
    "xq_moe_pf_g1", "xq_moe_pf_g2", "xq_moe_pf_g4", "xq_moe_pf_g8",
    "xq_moe_pf_count4", "xq_moe_pf_smem_query",
    // A5-L2: smem trellis-ring twins (prefill.moe_pf2) + the `_prof` clock64 twins (harness)
    "xq_moe_pf2_t1", "xq_moe_pf2_t2", "xq_moe_pf2_t4", "xq_moe_pf2_t8",
    "xq_moe_pf2_s1", "xq_moe_pf2_s2", "xq_moe_pf2_s4", "xq_moe_pf2_s8",
    "xq_moe_pf2_g1", "xq_moe_pf2_g2", "xq_moe_pf2_g4", "xq_moe_pf2_g8",
    "xq_moe_pf2_h1", "xq_moe_pf2_h2", "xq_moe_pf2_prof_h1", "xq_moe_pf2_prof_h2",
    "xq_moe_pf2_smem_query", "xq_moe_pf_prof_take",
    "xq_moe_pf_prof_s1", "xq_moe_pf_prof_s2", "xq_moe_pf_prof_s4", "xq_moe_pf_prof_s8",
    "xq_moe_pf_prof_g1", "xq_moe_pf_prof_g2", "xq_moe_pf_prof_g4", "xq_moe_pf_prof_g8",
    "xq_moe_pf2_prof_s1", "xq_moe_pf2_prof_s2", "xq_moe_pf2_prof_s4", "xq_moe_pf2_prof_s8",
    "xq_moe_pf2_prof_g1", "xq_moe_pf2_prof_g2", "xq_moe_pf2_prof_g4", "xq_moe_pf2_prof_g8",
    // A5-P2: prefill hc inject+norm / mix twins (128-thread CTAs, bitwise)
    "xq_pf_hc_inj_norm", "xq_pf_hc_mix_2560x4",
    // S-A3-q P3: row-batched bit-exact prefill router
    "xq_gemm_f16_f32_rows",
    // S-A3-u B: 32-row smem-staged bit-exact router twin
    "xq_gemm_f16_f32_rows32",
    // PFX1 (d): 64-col x 8-row bit-exact router twin for short prefill chunks
    "xq_gemm_f16_f32_rows8",
    // S-A3-p: int8-weight hc mixer twin (--hc-int8, opt-in)
    "xq_hc_fuse_i8",
    "xq_gemm_f16_v",
    "xq_gemm_f16_f32_v",
    "exl3_hmma_gemm_ks",
    "exl3_ks_combine",
    // S-A3-o C1: paired same-input chain stages (grid.z = member)
    "exl3_had_suh_x2",
    "exl3_hmma_gemm_ks_x2",
    "exl3_ks_combine_x2",
    "exl3_had_svh_x2",
    // w3-PSK (T4e): persistent split-K chain GEMM + pair twin (bitwise == the two above, any G)
    "exl3_hmma_gemm_ks_p",
    "exl3_hmma_gemm_ks_p_x2",
    // WP21: split-K fixup epilogue (ks GEMM + ascending combine + H128 svh in one launch)
    "exl3_hmma_gemm_ks_fx",
    "exl3_hmma_gemm_ks_fx_x2",
    // S-A3-d Phase 0b: PLE injection
    "xq_ple_norm_f16",
    "xq_ple_norm_f32",
    "xq_ple_norm_hh",
    "xq_ple_gate",
    "xq_ple_conv",
    "xq_ple_state_shift",
    // S-A3-f-d: router K-split + chain-verify (chained attn/group, commit replay,
    // on-device accept, copy/slice helpers, pruned-head trellis slice)
    "xq_router_ks",
    "xq_router_combine",
    "xq_attn_decode_group",
    // S-A3-m: probe-only per-row-loop reference (xq_attn_softmax class)
    "xq_attn_decode_group_ref",
    // WP13: dense-regime attention v3 (prep -> GQA-shared dots -> sequential accumulate),
    // bitwise to xq_attn_decode_group / xq_attn_decode; + the XCHECK diff kernel.
    "xq_attn_dense_prep",
    "xq_attn_dense_dots",
    "xq_attn_dense_acc4",
    "xq_attn_dense_acc1",
    "xq_dv3_xcheck",
    "xq_gdn_commit",
    // S-A3-o G1: rank-1 ring commit
    "xq_gdn_commit_ring",
    // WP12: register-resident GDN verify chunk + decode step (one device body), nostore
    // verify conv on the live ring, one-launch a|b GEMV (all bitwise; --wp12-off)
    "xq_gdn_step_chunk_r",
    "xq_gdn_step_r",
    "xq_conv1d_chunk_ns",
    "xq_gemm_f16_v_ab",
    "xq_conv_commit",
    "xq_ple_ring_commit",
    "xq_accept",
    // API-parity G2: device sampler (sampled lanes: plain steps + MTP verify rows)
    "xq_sample_rows",
    // WP15: device penalties (decode/verify rows + draft mirror)
    "xq_pen_rows",
    "xq_pen_draft",
    // WP24: real-q speculative sampling (draft top-32 + sample; ratio verify rows)
    "xq_rq_draft",
    "xq_sample_rows_rq",
    // A5-L4: real-q draft on DHEAD's candidate set (spec.ratio_dhead)
    "xq_rq_draft_dh",
    // API-parity DDS: draft-pass confidence (max logit)
    "xq_rowmax_f16",
    // DHEAD (w2): draft lm_head diet — screen build, prep, B-bit screen, exact rescore
    "xq_dh_build",
    "xq_dh_prep",
    "xq_dh_screen",
    "xq_dh_rescore",
    // DHEADP (w3): penalized draft passes — pass A (C1 exact + penalized, the bound) / pass B (expansion)
    "xq_dh_rescore_pa",
    "xq_dh_rescore_pb",
    // A5 D5: split-K dense draft-head attention (GQA-shared K/V reads)
    "xq_attn_dense_splitk4",
    // A5 D17: parallel prefill conv (bit-identical) + ring advance
    "xq_conv1d_chunk_par",
    "xq_conv1d_state_upd",
    // 2026-09-26: row-batched int8 hc mixer (bitwise per row)
    "xq_hc_fuse_i8_rb",
    "xq_hc_mix4",
    // WP11: bitwise hc mixer rungs (k-major twins + regridded mix)
    "xq_hc_fuse_i8k",
    "xq_hc_fuse_i8_rbk",
    "xq_hc_mix_rg",
    // HC (w4): regridded mix-fused hc mixer (15 params: raw launch, exl3_forward::w4hc_launch) +
    // fused inject+norm (both bitwise twins of the p4c path)
    "xq_hc_w4",
    "xq_hc_inj_norm",
    "xq_gemm_grouped_a1b3_fh",
    "xq_copy1_i32",
    // HOST / RO-5(a): head-row save/restore around a speculative draft pass
    "xq_spec_rows",
    // S-A3-p: graphed draft chain setup
    "xq_draft_setup",
    // REPRIME (w2): fixed-width graphed head re-prime (device stage, masked key write, device-meta KV)
    "xq_reprime_stage",
    "xq_qsa_key_write_dn",
    "xq_attn_kv_rows_dn",
    "xq_copy_f32",
    "xq_copy_u16",
    "xq_slice_trellis_u16",
    "xq_slice_trellis_u16_off",
    // S-A3-h item 3 rung 1: cooperative MoE expert-stream GEMM pair (schedule
    // port of the rival's exl3_moe_coop_a/b; math identical to
    // xq_gemm_grouped_xh — see kernels/exl3_bench.cu). A = gate/up, B = down;
    // launched from fn moe() behind --moe-coop.
    "xq_moe_coop_a",
    "xq_moe_coop_b",
    // S-A3-i: QSA selection + sparse attention chain (kernels/exl3_bench.cu).
    "xq_qsa_h2b",
    "xq_qsa_qnorm_rope",
    "xq_qsa_topk_asc",
    "xq_attn_prep",
    "xq_attn_sel_splitk",
    "xq_attn_sel_combine",
    // S-A3-v: float4 prefill twin of xq_attn_sel_splitk (bit-identical).
    "xq_attn_sel_splitk4",
    // PQ8 item 3: 8-bit-cache prefill gather (bit-identical partials to splitk4)
    "xq_attn_sel_pq8",
    // WP25: KV cache format round-trip through the served helpers (--probe-exl3-kvq).
    "xq_kv_selftest",
    // S-A3-z: fused router GEMV (bit-identical to router_ks + combine).
    "xq_router_fused",
    // WP09: k-major PLE key+value twin (bitwise == xq_gemm_f16_rows) + one-launch multi-block
    // argmax (same ids as xq_argmax).
    "xq_gemm_f16_rows_km",
    "xq_argmax_rows",
    // WP14 (SURPASS_PLAN_2026-09-26): bit-identical top-k twin + head-grouped gathers.
    "xq_qsa_topk_asc2",
    // A5-K3 (attn.qsa_select): latency-lean bit-identical top-k twin of asc2.
    "xq_qsa_topk_asc3",
    "xq_attn_sel_hg1",
    "xq_attn_sel_hg3",
    // W4/SMALL (wave 4): bitwise small-kernel twins (xq_router_fold_w4 is a raw launch, unlisted)
    "xq_gemm_f16_rows_km_w4m1",
    "xq_gemm_f16_rows_km_w4m2",
    "xq_gemm_f16_rows_km_w4m3",
    "xq_gemm_f16_rows_km_w4m4",
    "xq_gemm_f16_rows_km_w4m5",
    "xq_gemm_f16_rows_km_w4m6",
    "xq_gemm_f16_rows_km_w4m7",
    "xq_gemm_f16_rows_km_w4m8",
    "xq_gdn_commit_all_w4",
    "xq_attn_dense_acc4_w4",
    "xq_attn_dense_acc1_w4",
    "xq_attn_dense_splitk4_w4",
    "xq_attn_sel_combine_w4",
    // W4/QSA: row-shared union gather for multi-row verifies (bit-identical partials to the above)
    // + the binv sparse section's KV synthesizer (probe-only).
    "xq_attn_sel_union",
    "xq_qu_synth_kv",
    // WP18 rung A: bounded prefill selection (bitwise tiled scorer + carried top-K merge) and
    // its XCHECK score diff.
    "xq_wp18_score",
    "xq_wp18_merge",
    "xq_wp18_cmp",
];

fn dev0() -> Result<std::sync::Arc<CudaDevice>> {
    CudaDevice::new(0).context("CudaDevice::new(0)")
}

fn load_module(dev: &std::sync::Arc<CudaDevice>) -> Result<()> {
    let ptx = Ptx::from_src(
        std::fs::read_to_string("src/ptx/exl3_bench.ptx")
            .context("src/ptx/exl3_bench.ptx missing — run cargo build --release")?,
    );
    dev.load_ptx(ptx, MODULE, MODULE_FNS)
        .context("load exl3_bench module")?;
    Ok(())
}

/// Inverse tmap LUT: rc_of_t[t] = r*16 + c (from the unit-tested Rust bijection).
pub(crate) fn tmap_inv_table() -> Vec<u8> {
    let mut inv = vec![0u8; 256];
    for r in 0..16usize {
        for c in 0..16usize {
            inv[tmap(r, c)] = (r * 16 + c) as u8;
        }
    }
    inv
}

fn le_i16(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn le_f16_bits(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// The bench/gate matrix set: one expert gate (b3), one expert down (b3), one dense
/// (b5), lm_head (b5), one mtp fc (b4) — spec §2.2's representative shapes.
fn pick_modules(pack: &Exl3Pack) -> Vec<ModuleMeta> {
    let find = |pred: &dyn Fn(&ModuleMeta) -> bool| {
        pack.modules.iter().find(|m| pred(m)).cloned()
    };
    let mut out = Vec::new();
    if let Some(m) = find(&|m: &ModuleMeta| {
        m.class == TensorClass::Expert && m.bits == 3 && m.name.ends_with("gate_proj")
    }) {
        out.push(m);
    }
    if let Some(m) = find(&|m: &ModuleMeta| {
        m.class == TensorClass::Expert && m.bits == 3 && m.name.ends_with("down_proj")
    }) {
        out.push(m);
    }
    if let Some(m) = find(&|m: &ModuleMeta| {
        m.class == TensorClass::Dense && m.bits == 5 && m.n == 10240
    }) {
        out.push(m);
    }
    if let Some(m) = find(&|m: &ModuleMeta| m.class == TensorClass::LmHead) {
        out.push(m);
    }
    if let Some(m) = find(&|m: &ModuleMeta| {
        m.class == TensorClass::Mtp && m.bits == 4 && m.k == 2560 && m.n == 2560
    }) {
        out.push(m);
    }
    out
}

/// Deterministic fp16 activations, row-major [rows, k]. Row 0 is INDEPENDENT of
/// `rows` (same seed) — the batch-invariance requirement.
fn synth_x(rows: usize, k: usize, seed: u32) -> Vec<u16> {
    let mut s = seed;
    let mut out = Vec::with_capacity(rows * k);
    for r in 0..rows {
        let mut sr = seed.wrapping_add(7919).wrapping_mul(r as u32 + 1); // per-row, row-0 stable
        let _ = s;
        for _ in 0..k {
            sr = sr.wrapping_mul(1664525).wrapping_add(1013904223);
            let v = (((sr >> 8) % 4001) as f32 - 2000.0) / 512.0; // ~±3.9
            out.push(f16::from_f32(v).to_bits());
        }
    }
    out
}

/// Host Walsh-Hadamard butterfly on one 128-block — same op order as the kernel's
/// `exl3_had128` (ascending len; commutative adds make lane split irrelevant).
fn had128_ref(v: &mut [f32]) {
    debug_assert_eq!(v.len(), 128);
    let mut len = 1usize;
    while len < 128 {
        for i in 0..128 {
            if i & len == 0 {
                let a = v[i];
                let b = v[i + len];
                v[i] = a + b;
                v[i + len] = a - b;
            }
        }
        len <<= 1;
    }
}

fn had_scale_ref(x: &[u16], scale: &[u16], rows: usize, dim: usize, pre_scale: bool) -> Vec<u16> {
    // pre_scale=true: y = H(x ⊙ scale)  — the INPUT side (suh before the butterfly,
    // their had kernel's pre_scale order). pre_scale=false: y = H(x) ⊙ scale — the
    // OUTPUT side (svh after the butterfly, their post_scale order).
    let blocks = dim / 128;
    let mut out = vec![0u16; rows * dim];
    for r in 0..rows {
        for blk in 0..blocks {
            let mut v = [0f32; 128];
            for j in 0..128 {
                let xv = f16::from_bits(x[r * dim + blk * 128 + j]).to_f32();
                v[j] = if pre_scale {
                    xv * f16::from_bits(scale[blk * 128 + j]).to_f32()
                } else {
                    xv
                };
            }
            had128_ref(&mut v);
            // Kernel mirror: the shared exl3_had128 now normalizes orthonormally
            // (their convention, probed); the host butterfly must match.
            for j in 0..128 {
                v[j] *= 0.08838834764831845;
            }
            for j in 0..128 {
                let val = if pre_scale {
                    v[j]
                } else {
                    v[j] * f16::from_bits(scale[blk * 128 + j]).to_f32()
                };
                out[r * dim + blk * 128 + j] = f16::from_f32(val).to_bits();
            }
        }
    }
    out
}

/// Column-slab a trellis tensor [kb][nb][16*bits] to [kb][nc][16*bits] (nc blocks
/// starting at block column 0) and the matching svh slice. Full tensor when None.
fn slab_trellis(tr: &[u8], kb: usize, nb: usize, bits: usize, cols: Option<usize>) -> (Vec<u8>, usize) {
    match cols {
        None => (tr.to_vec(), nb * 16),
        Some(c) => {
            let nc = (c / 16).min(nb);
            let blw = 16 * bits * 2; // block bytes
            let mut out = Vec::with_capacity(kb * nc * blw);
            for k in 0..kb {
                out.extend_from_slice(&tr[k * nb * blw..k * nb * blw + nc * blw]);
            }
            (out, nc * 16) // COLUMNS, matching the full-tensor branch's contract
        }
    }
}

struct Chain {
    d_tr: CudaSlice<u16>,
    d_suh: CudaSlice<u16>,
    d_svh: CudaSlice<u16>,
    d_x: CudaSlice<u16>,
    d_xh: CudaSlice<u16>,
    d_yraw: CudaSlice<u16>,
    d_y: CudaSlice<u16>,
    m_rows: usize,
    k: usize,
    n_eff: usize,
    bits: usize,
}

/// Upload one case and run the full chain (had_suh -> gemm -> had_svh).
fn run_chain(
    dev: &std::sync::Arc<CudaDevice>,
    tr_bits: &[u16],
    suh: &[u16],
    svh: &[u16],
    x: &[u16],
    m_rows: usize,
    k: usize,
    n_eff: usize,
    bits: usize,
) -> Result<Chain> {
    assert!(k % 128 == 0 && n_eff % 128 == 0 && n_eff % 16 == 0);
    let d_tr = dev.htod_sync_copy(tr_bits).context("htod trellis")?;
    let d_suh = dev.htod_sync_copy(suh).context("htod suh")?;
    let d_svh = dev.htod_sync_copy(svh).context("htod svh")?;
    let d_x = dev.htod_sync_copy(x).context("htod x")?;
    let d_xh = unsafe { dev.alloc::<u16>(m_rows * k) }.context("alloc xh")?;
    let d_yraw = unsafe { dev.alloc::<u16>(m_rows * n_eff) }.context("alloc yraw")?;
    let d_y = unsafe { dev.alloc::<u16>(m_rows * n_eff) }.context("alloc y")?;
    let mut c = Chain {
        d_tr,
        d_suh,
        d_svh,
        d_x,
        d_xh,
        d_yraw,
        d_y,
        m_rows,
        k,
        n_eff,
        bits,
    };
    launch_chain(dev, &mut c)?;
    Ok(c)
}

fn launch_chain(dev: &std::sync::Arc<CudaDevice>, c: &mut Chain) -> Result<()> {
    let stream = fork_blocking_stream(dev);
    let r = launch_chain_on(dev, &stream, c);
    dev.synchronize().context("sync")?;
    r
}

fn launch_chain_on(dev: &std::sync::Arc<CudaDevice>, stream: &CudaStream, c: &mut Chain) -> Result<()> {
    let _ = dev;
    let f_had_suh = dev
        .get_func(MODULE, "exl3_had_suh")
        .ok_or_else(|| anyhow::anyhow!("fn exl3_had_suh missing"))?;
    let f_gemm = dev
        .get_func(MODULE, "exl3_hmma_gemm")
        .ok_or_else(|| anyhow::anyhow!("fn exl3_hmma_gemm missing"))?;
    let f_had_svh = dev
        .get_func(MODULE, "exl3_had_svh")
        .ok_or_else(|| anyhow::anyhow!("fn exl3_had_svh missing"))?;
    let cfg = |grid: u32| LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        f_had_suh
            .clone()
            .launch_on_stream(
                stream,
                cfg((c.m_rows * c.k / 128) as u32),
                (&c.d_x, &c.d_suh, &mut c.d_xh, c.k as i32),
            )
            .context("launch had_suh")?;
    }
    unsafe {
        f_gemm
            .clone()
            .launch_on_stream(
                stream,
                LaunchConfig {
                    grid_dim: ((c.n_eff / 128) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &c.d_tr,
                    &c.d_xh,
                    &mut c.d_yraw,
                    c.m_rows as i32,
                    c.k as i32,
                    c.n_eff as i32,
                    c.bits as i32,
                ),
            )
            .context("launch hmma_gemm")?;
    }
    unsafe {
        f_had_svh
            .clone()
            .launch_on_stream(
                stream,
                cfg((c.m_rows * c.n_eff / 128) as u32),
                (&c.d_yraw, &c.d_svh, &mut c.d_y, c.n_eff as i32),
            )
            .context("launch had_svh")?;
    }
    Ok(())
}

fn rel_l2(a: &[u16], b_f32: &[f32]) -> (f64, f64) {
    // (rel-L2 vs f32 reference, rel-L2 vs fp16-rounded reference)
    let mut num = 0f64;
    let den = b_f32
        .iter()
        .map(|v| {
            let x = f64::from(*v);
            x * x
        })
        .sum::<f64>()
        .max(1e-30);
    let mut num16 = 0f64;
    for (i, av) in a.iter().enumerate() {
        let av = f16::from_bits(*av).to_f32() as f64;
        let bv = b_f32[i] as f64;
        num += (av - bv) * (av - bv);
        let br = f16::from_f32(b_f32[i]).to_f32() as f64;
        num16 += (av - br) * (av - br);
    }
    (num.sqrt() / den.sqrt(), num16.sqrt() / den.sqrt())
}

fn gemm_ref(w: &[u16], xh: &[u16], rows: usize, k: usize, n: usize) -> Vec<f32> {
    // w = decoded trellis, ROW-MAJOR [K, N] (decode_trellis_host contract:
    // "W [kb*16, nb*16]"). y[r, n] = sum_k xh[r, k] * w[k*n + n].
    let mut y = vec![0f32; rows * n];
    for r in 0..rows {
        for kk in 0..k {
            let xv = f16::from_bits(xh[r * k + kk]).to_f32();
            if xv == 0.0 {
                continue;
            }
            let wrow = &w[kk * n..kk * n + n];
            for ni in 0..n {
                y[r * n + ni] += xv * f16::from_bits(wrow[ni]).to_f32();
            }
        }
    }
    y
}

// ---------------------------------------------------------------------------
// 1. G-A3-1 through the kernel: CUDA decode vs Rust oracle, bitwise.
// ---------------------------------------------------------------------------
pub fn probe_kernel_decode(dir: &str) -> Result<()> {
    let pack = Exl3Pack::open(dir)?;
    let dev = dev0()?;
    load_module(&dev)?;
    let rc = tmap_inv_table();
    let d_rc = dev.htod_sync_copy(&rc)?;
    let f_dump = dev
        .get_func(MODULE, "exl3_decode_dump")
        .ok_or_else(|| anyhow::anyhow!("fn exl3_decode_dump missing"))?;
    let stream = fork_blocking_stream(&dev);

    let modules = pick_modules(&pack);
    if modules.is_empty() {
        bail!("no gate matrices found in pack {dir}");
    }
    let mut total_blocks = 0usize;
    for m in &modules {
        let (kb, nb) = (m.k / 16, m.n / 16);
        let t0 = std::time::Instant::now();
        let raw = m.trellis.read_bytes(&pack.dir)?;
        let trellis = le_i16(&raw);
        let ref_w = decode_trellis_host(&trellis, kb, nb, m.bits);
        let t_ref = t0.elapsed();

        let d_tr = dev.htod_sync_copy(&raw)?;
        let mut d_out = unsafe { dev.alloc::<u16>(ref_w.len()) }?;
        unsafe {
            f_dump
                .clone()
                .launch_on_stream(
                    &stream,
                    LaunchConfig {
                        grid_dim: ((kb * nb) as u32, 1, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&d_tr, &d_rc, &mut d_out, nb as i32, m.bits as i32),
                )
                .context("launch exl3_decode_dump")?;
        }
        dev.synchronize()?;
        let mut got = dev.dtoh_sync_copy(&d_out)?;
        // S-A3-u G: the coalesced prefill reconstruct must produce the same bytes.
        if nb % 8 == 0 {
            let f8 = dev.get_func(MODULE, "exl3_decode_dump8")
                .ok_or_else(|| anyhow::anyhow!("fn exl3_decode_dump8 missing"))?;
            let mut d_out8 = unsafe { dev.alloc::<u16>(ref_w.len()) }?;
            unsafe {
                f8.launch_on_stream(&stream,
                    LaunchConfig { grid_dim: ((kb * nb / 8) as u32, 1, 1), block_dim: (256, 1, 1),
                                   shared_mem_bytes: 0 },
                    (&d_tr, &d_rc, &mut d_out8, nb as i32, m.bits as i32))
                    .context("launch exl3_decode_dump8")?;
            }
            dev.synchronize()?;
            let got8 = dev.dtoh_sync_copy(&d_out8)?;
            let m8 = got8.iter().zip(got.iter()).filter(|(a, b)| a != b).count();
            if m8 > 0 {
                bail!("EXL3-KERNEL-DECODE FAIL: exl3_decode_dump8 differs from exl3_decode_dump in {m8} of {} on {}", got.len(), m.name);
            }
            got = got8; // and it is what gets compared to the oracle below
        }

        let mut mism = 0usize;
        let mut first: Vec<(usize, u16, u16)> = Vec::new();
        for (i, (g, r)) in got.iter().zip(ref_w.iter()).enumerate() {
            if g != r {
                mism += 1;
                if first.len() < 4 {
                    first.push((i, *g, *r));
                }
            }
        }
        total_blocks += kb * nb;
        println!(
            "  {:<28} b={} {:>5}x{:<6} {:>9} blocks  mism {}  (ref {:?}, gpu {:?})",
            m.name.rsplit('.').take(3).collect::<Vec<_>>().join("."),
            m.bits,
            m.k,
            m.n,
            kb * nb,
            mism,
            t_ref,
            t0.elapsed() - t_ref
        );
        for (i, g, r) in &first {
            println!("    first mismatch @{}: kernel {:04x} vs ref {:04x}", i, g, r);
        }
        if mism > 0 {
            bail!("EXL3-KERNEL-DECODE FAIL on {}", m.name);
        }
    }
    println!(
        "EXL3-KERNEL-DECODE: PASS (bitwise vs decode_trellis_host, {} blocks, {} matrices)",
        total_blocks,
        modules.len()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 2. G-A3-5: batch invariance + GEMM correctness vs host reference.
// ---------------------------------------------------------------------------
pub fn probe_gemm(dir: &str, widths: &[usize], slab_cols: usize) -> Result<()> {
    let pack = Exl3Pack::open(dir)?;
    let dev = dev0()?;
    load_module(&dev)?;
    // --binv-section=wp18: only the WP18 prefill-selection section (synthetic; no pack reads).
    if crate::opts::var(crate::opt!("binv-section")).as_deref() == Ok("wp18") {
        return probe_wp18_select(&dev);
    }
    // --binv-section=moe_pf: only the A5-P1 prefill MoE section (synthetic; no pack reads).
    if crate::opts::var(crate::opt!("binv-section")).as_deref() == Ok("moe_pf") {
        return probe_moe_pf(&dev);
    }
    // --binv-section=pf_hc: only the A5-P2 prefill hc section (synthetic; no pack reads).
    if crate::opts::var(crate::opt!("binv-section")).as_deref() == Ok("pf_hc") {
        return probe_pf_hc(&dev);
    }
    // --binv-section=gdn_tc: only the A5-L1 chunkwise GDN scan section (synthetic; no pack reads).
    if crate::opts::var(crate::opt!("binv-section")).as_deref() == Ok("gdn_tc") {
        return probe_gdn_tc2(&dev);
    }

    for m in pick_modules(&pack) {
        // lm_head: gate on a column slab (the host reference for the full 248320-col
        // matrix would decode 1.27 GB and GEMM it — the slab carries the same per-tile
        // arithmetic; the full matrix is the BENCH's job).
        let raw = m.trellis.read_bytes(&pack.dir)?;
        let (trb, n_eff) = if m.class == TensorClass::LmHead {
            slab_trellis(&raw, m.k / 16, m.n / 16, m.bits, Some(slab_cols))
        } else {
            (raw.clone(), m.n)
        };
        let n_eff16 = n_eff;
        let suh = le_f16_bits(&m.suh.read_bytes(&pack.dir)?);
        let svh_full = le_f16_bits(&m.svh.read_bytes(&pack.dir)?);
        let svh: Vec<u16> = if n_eff16 == m.n {
            svh_full
        } else {
            svh_full[..n_eff16].to_vec()
        };

        // correctness once at width 4 (covers padded rows), then binv across widths.
        let k = m.k;
        let bits = m.bits;
        let tr_bits: Vec<u16> = trb
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();

        println!(
            "  {:<24} b={} K={:<5} N={:<6} (slab {} cols)",
            m.name.rsplit('.').take(2).collect::<Vec<_>>().join("."),
            bits,
            k,
            m.n,
            n_eff16
        );

        // ---- correctness at width 4 ----
        let w = 4usize;
        let x = synth_x(w, k, 0xC0FFEE);
        let mut chain = run_chain(&dev, &tr_bits, &suh, &svh, &x, w, k, n_eff16, bits)?;
        let yraw = dev.dtoh_sync_copy(&chain.d_yraw)?;
        let y = dev.dtoh_sync_copy(&chain.d_y)?;

        let xh_ref = had_scale_ref(&x, &suh, w, k, true);
        let w_dec = decode_trellis_host(
            &tr_bits.iter().map(|v| *v as i16).collect::<Vec<_>>(),
            k / 16,
            n_eff16 / 16,
            bits,
        );
        let yraw_ref = gemm_ref(&w_dec, &xh_ref, w, k, n_eff16);
        let (rel_raw_f32, rel_raw_16) = rel_l2(&yraw[..w * n_eff16], &yraw_ref);
        let y_ref = had_scale_ref(
            &yraw_ref.iter().map(|v| f16::from_f32(*v).to_bits()).collect::<Vec<_>>(),
            &svh,
            w,
            n_eff16,
            false,
        );
        let (rel_y_f32, rel_y_16) = rel_l2(
            &y[..w * n_eff16],
            &y_ref.iter().map(|b| f16::from_bits(*b).to_f32()).collect::<Vec<_>>(),
        );
        println!(
            "    correctness W={w}: y_raw rel-L2 (vs f32 {:.2e} | vs fp16(ref) {:.2e})   y rel-L2 (vs f32 {:.2e} | vs fp16(ref) {:.2e})   [residual = mma-vs-sequential association + fp16 output rounding; decode is bitwise-green]",
            rel_raw_f32, rel_raw_16, rel_y_f32, rel_y_16
        );
        // Thresholds calibrated 2026-09-19 on .13: fp16 OUTPUT rounding alone is ~1e-4-class
        // (ULP 9.8e-4 rel; ~5% of elements straddle a rounding boundary at the measured
        // association residual), so 1e-5 was unachievable by construction. Structural bugs
        // (transposed ref, wrong placement) measure O(1) — the gate's real target.
        if rel_raw_16 > 1e-4 || rel_y_16 > 5e-4 {
            bail!("EXL3-GEMM correctness FAIL on {}", m.name);
        }

        // ---- batch invariance: row 0 bitwise across widths ----
        let mut row0_w: Vec<Vec<u16>> = Vec::new();
        for wd in widths {
            let x = synth_x(*wd, k, 0xC0FFEE);
            // reuse buffers: relaunch with new x
            let d_x = dev.htod_sync_copy(&x)?;
            chain.d_x = d_x;
            launch_chain(&dev, &mut chain)?;
            let yraw = dev.dtoh_sync_copy(&chain.d_yraw)?;
            let y = dev.dtoh_sync_copy(&chain.d_y)?;
            row0_w.push(yraw[..n_eff16].to_vec());
            row0_w.push(y[..n_eff16].to_vec());
            let _ = y;
        }
        let base_raw = &row0_w[0];
        let mut mism = 0usize;
        for (wi, chunk) in row0_w.chunks(2).enumerate().skip(1) {
            for (i, (a, b)) in base_raw.iter().zip(chunk[0].iter()).enumerate() {
                if a != b {
                    mism += 1;
                    if mism < 4 {
                        println!("    binv mismatch y_raw W={}[{}]: {:04x} vs W=1 {:04x}", widths[wi], i, b, a);
                    }
                }
            }
            let _ = &chunk[1];
        }
        if mism > 0 {
            bail!("EXL3-BINV FAIL on {} ({} mismatches)", m.name, mism);
        }
        println!("    binv: PASS (y_raw row 0 bitwise identical across widths {widths:?})");
        probe_grouped_a1(&dev, &chain.d_tr, k, n_eff16, bits)?;
        // W3/LMH: the persistent lm_head GEMM on this matrix (the head: its column slab), every
        // m in 1..16 — every class exercises the 3/4/5-bit decode extraction.
        let all_m: Vec<usize> = (1..=16).collect();
        let nsts: &[usize] = if bits == 5 { &[4, 6, 8] } else { &[6] };
        probe_lmh(&dev, &chain.d_tr, &chain.d_svh, k, n_eff16, bits, &all_m, nsts, "matrix/slab")?;
        if m.class == TensorClass::LmHead {
            // The served shapes: the FULL head (N = vocab: 1,940 tiles over the SM grid, uneven
            // tiles per CTA) and the 65,536-id draft slice (512 tiles).
            let full: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            let svh_all = le_f16_bits(&m.svh.read_bytes(&pack.dir)?);
            let d_full = dev.htod_sync_copy(&full)?;
            let d_svh_all = dev.htod_sync_copy(&svh_all)?;
            drop(full);
            probe_lmh(&dev, &d_full, &d_svh_all, k, m.n, bits, &all_m, &[6], "full head")?;
            probe_lmh(&dev, &d_full, &d_svh_all, k, m.n, bits, &[1, 6, 9, 16], &[4, 8], "full head, ring sweep")?;
            drop(d_full);
            let (trd, nd) = slab_trellis(&raw, k / 16, m.n / 16, bits, Some(65536));
            let trd16: Vec<u16> = trd.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            let d_draft = dev.htod_sync_copy(&trd16)?;
            let d_svh_draft = dev.htod_sync_copy(&svh_all[..nd])?;
            probe_lmh(&dev, &d_draft, &d_svh_draft, k, nd, bits, &[1, 2, 6, 16], &[4, 6, 8], "draft slice")?;
        }
    }
    probe_ks_p(&dev, &pack)?;
    probe_wp21(&dev, &pack)?;
    probe_dense(&dev, &pack)?;
    println!("EXL3-GEMM: PASS (correctness + batch invariance)");
    probe_attn(&dev)?;
    probe_attn_sparse(&dev)?;
    probe_qsa_select(&dev)?;
    probe_wp18_select(&dev)?;
    probe_gemm_f16_rows(&dev)?;
    probe_moe_epi(&dev)?;
    probe_hc_w4(&dev)?;
    probe_moe_pf(&dev)?;
    probe_pf_hc(&dev)?;
    probe_gdn_tc2(&dev)?;
    // TP-SP1: the sequence-parallel prefill's pinned-plan gate over the whole split range (world 2)
    crate::exl3_forward::probe_sp_pinned(&(crate::exl3_forward::xtp::SP_MIN_ROWS..=crate::exl3_forward::xtp::SP_MAX_ROWS)
                                           .collect::<Vec<usize>>(), 2)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// A5-L1: xq_gdn_chunk_tc2 (wavefront solve + register prefetch) must equal the served chunkwise GDN
// prefill scan xq_gdn_chunk_tc (split 7) BITWISE — core rows (f16) and the f32 recurrent state —
// at the served shape (nh 48 value heads, 16 key heads, kd = vd = 128; every head, slot 1 of 2) over
// N in {2048, 775, 96, 16} from a fresh state each, then the same four chained on one carried state.
// Core outputs NaN-poisoned first; the reference must leave no poison, must move the state and
// must write non-zero core rows. Inputs: q/k rows at spread magnitudes (incl. an all-zero q row and
// an all-zero k row: the eps path), v, beta/decay logits spanning strong decay .. g ~ 1.
// Harness-only: per-call timing at N = 2048 (x36 GDN layers = per 2048-row chunk) and the _prof
// twins' per-phase clock64 anatomy (cycles per 32-token sub-chunk, thread 0 between barriers).
fn probe_gdn_tc2(dev: &std::sync::Arc<CudaDevice>) -> Result<()> {
    use cudarc::driver::sys;
    let (nh, nk, kd, vd) = (48usize, 16usize, 128usize, 128usize);
    let stride = 2 * nk * kd + nh * vd;
    let nmax = 2048usize;
    let nslot = 2usize;
    let smem = crate::exl3_forward::GTC_SMEM;
    let get = |nm: &str| -> Result<sys::CUfunction> {
        let f = crate::exl3_forward::xq_raw_fn(nm).ok_or_else(|| anyhow::anyhow!("{nm} raw fn (stale PTX?)"))?;
        let r = unsafe { sys::cuFuncSetAttribute(f, sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem) };
        if r != sys::CUresult::CUDA_SUCCESS { bail!("EXL3-GDN-TC2 FAIL: smem opt-in {nm} ({r:?})"); }
        Ok(f)
    };
    let (f_old, f_new, f_oldp, f_newp) = (get("xq_gdn_chunk_tc")?, get("xq_gdn_chunk_tc2")?,
                                          get("xq_gdn_chunk_tc_prof")?, get("xq_gdn_chunk_tc2_prof")?);
    // inputs (f16 bits)
    let mut qkv = wp20_synth(nmax * stride, 0x6D11, 0.6);
    for t in 0..nmax {
        let sc = [0.05f32, 0.3, 1.0, 4.0][t % 4];
        for i in 0..2 * nk * kd {
            let v = f16::from_bits(qkv[t * stride + i]).to_f32() * sc;
            qkv[t * stride + i] = f16::from_f32(v).to_bits();
        }
    }
    for i in 0..nk * kd { qkv[7 * stride + i] = 0; }                 // token 7: q all zero
    for i in 0..nk * kd { qkv[40 * stride + nk * kd + i] = 0; }      // token 40: k all zero
    let b_in = wp20_synth(nmax * nh, 0x6D12, 1.0);                   // beta = sigmoid(-3.9..3.9)
    let a_in: Vec<u16> = wp20_synth(nmax * nh, 0x6D13, 1.0).iter()
        .map(|&b| f16::from_f32(f16::from_bits(b).to_f32() - 1.5).to_bits()).collect();
    let a_log: Vec<f32> = (0..nh).map(|i| -2.5 + 3.5 * (i as f32) / (nh as f32)).collect();
    let dt_bias: Vec<f32> = (0..nh).map(|i| ((i * 7 % 11) as f32 - 5.0) / 5.0).collect();
    let st0: Vec<f32> = wp20_synth(nslot * nh * kd * vd, 0x6D14, 0.1).iter().map(|&b| f16::from_bits(b).to_f32()).collect();
    let d_qkv = dev.htod_sync_copy(&qkv)?;
    let d_b = dev.htod_sync_copy(&b_in)?;
    let d_a = dev.htod_sync_copy(&a_in)?;
    let d_alog = dev.htod_sync_copy(&a_log)?;
    let d_dtb = dev.htod_sync_copy(&dt_bias)?;
    let mut d_sa = dev.htod_sync_copy(&st0)?;
    let mut d_sb = dev.htod_sync_copy(&st0)?;
    let mut d_ca = dev.alloc_zeros::<u16>(nmax * nh * vd)?;
    let mut d_cb = dev.alloc_zeros::<u16>(nmax * nh * vd)?;
    let mut d_prof = dev.alloc_zeros::<u64>(nh * (vd / 32) * 10)?;
    let grid = (nh * (vd / 32)) as u32;
    let launch = |f: sys::CUfunction, core: u64, st: u64, qoff: usize, n: usize, prof: Option<u64>| -> Result<()> {
        let mut a = [core, *d_qkv.device_ptr() as u64 + (qoff * stride * 2) as u64, st,
                     *d_b.device_ptr() as u64 + (qoff * nh * 2) as u64, *d_a.device_ptr() as u64 + (qoff * nh * 2) as u64,
                     *d_alog.device_ptr() as u64, *d_dtb.device_ptr() as u64];
        let mut d = [(nh | (nk << 16)) as i32, 1i32, n as i32];
        let mut pp = prof.unwrap_or(0);
        let mut p: [*mut std::ffi::c_void; 11] = [
            &mut a[0] as *mut u64 as *mut _, &mut a[1] as *mut u64 as *mut _, &mut a[2] as *mut u64 as *mut _,
            &mut a[3] as *mut u64 as *mut _, &mut a[4] as *mut u64 as *mut _, &mut a[5] as *mut u64 as *mut _,
            &mut a[6] as *mut u64 as *mut _, &mut d[0] as *mut i32 as *mut _, &mut d[1] as *mut i32 as *mut _,
            &mut d[2] as *mut i32 as *mut _, &mut pp as *mut u64 as *mut _,
        ];
        let r = unsafe { sys::cuLaunchKernel(f, grid, 1, 1, 256, 1, 1, smem as u32, std::ptr::null_mut(),
                                             p.as_mut_ptr(), std::ptr::null_mut()) };
        if r != sys::CUresult::CUDA_SUCCESS { bail!("EXL3-GDN-TC2 FAIL: launch ({r:?})"); }
        Ok(())
    };
    let (pca, pcb, psa, psb) = (*d_ca.device_ptr() as u64, *d_cb.device_ptr() as u64,
                                *d_sa.device_ptr() as u64, *d_sb.device_ptr() as u64);
    let mut nchk = 0usize;
    let mut check = |tag: &str, n: usize, prev: &[f32], d_ca: &CudaSlice<u16>, d_cb: &CudaSlice<u16>,
                     d_sa: &CudaSlice<f32>, d_sb: &CudaSlice<f32>| -> Result<Vec<f32>> {
        let (ca, cb) = (dev.dtoh_sync_copy(d_ca)?, dev.dtoh_sync_copy(d_cb)?);
        let (sa, sb) = (dev.dtoh_sync_copy(d_sa)?, dev.dtoh_sync_copy(d_sb)?);
        let m = n * nh * vd;
        let bad_c = ca[..m].iter().zip(&cb[..m]).filter(|(p, q)| p != q).count();
        let bad_s = sa.iter().zip(&sb).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
        if bad_c > 0 || bad_s > 0 {
            bail!("EXL3-GDN-TC2 FAIL {tag} N={n}: core {bad_c} of {m} f16 differ (first {:?}), state {bad_s} of {} f32 differ (first {:?})",
                  ca[..m].iter().zip(&cb[..m]).position(|(p, q)| p != q), sa.len(),
                  sa.iter().zip(&sb).position(|(p, q)| p.to_bits() != q.to_bits()));
        }
        if ca[..m].iter().any(|&v| v == 0x7E00) { bail!("EXL3-GDN-TC2 FAIL {tag} N={n}: poison left in the reference core"); }
        if sa.iter().any(|v| !v.is_finite()) { bail!("EXL3-GDN-TC2 FAIL {tag} N={n}: non-finite reference state"); }
        let moved = sa[nh * kd * vd..].iter().zip(&prev[nh * kd * vd..]).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
        let slot0 = sa[..nh * kd * vd].iter().zip(&prev[..nh * kd * vd]).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
        let nz = ca[..m].iter().filter(|&&v| v & 0x7FFF != 0).count();
        if moved < nh * kd * vd / 2 || slot0 != 0 || nz < m / 2 {
            bail!("EXL3-GDN-TC2 FAIL {tag} N={n}: implausible reference (state moved {moved}, slot 0 touched {slot0}, non-zero core {nz} of {m})");
        }
        nchk += 2;
        println!("  EXL3-GDN-TC2 {tag} N={n}: core ({m} f16) + state ({} f32) bitwise == xq_gdn_chunk_tc", sa.len());
        Ok(sa)
    };
    let poison = |d: &mut CudaSlice<u16>| -> Result<()> { let k = d.len(); dev.htod_sync_copy_into(&vec![0x7E00u16; k], d)?; Ok(()) };
    // fresh state per N (qkv window offsets differ so each N sees other rows)
    for (i, &n) in [2048usize, 775, 96, 16].iter().enumerate() {
        let qoff = [0usize, 1000, 1900, 2020][i].min(nmax - n);
        dev.htod_sync_copy_into(&st0, &mut d_sa)?; dev.htod_sync_copy_into(&st0, &mut d_sb)?;
        poison(&mut d_ca)?; poison(&mut d_cb)?;
        launch(f_old, pca, psa, qoff, n, None)?;
        launch(f_new, pcb, psb, qoff, n, None)?;
        dev.synchronize()?;
        check("fresh", n, &st0, &d_ca, &d_cb, &d_sa, &d_sb)?;
    }
    // chained: the four lengths back to back on one carried state (a 2048-row chunk then partial tails)
    dev.htod_sync_copy_into(&st0, &mut d_sa)?; dev.htod_sync_copy_into(&st0, &mut d_sb)?;
    let mut prev = st0.clone();
    for (i, &n) in [2048usize, 775, 96, 16].iter().enumerate() {
        let qoff = [0usize, 1273, 1952, 2032][i];
        poison(&mut d_ca)?; poison(&mut d_cb)?;
        launch(f_old, pca, psa, qoff, n, None)?;
        launch(f_new, pcb, psb, qoff, n, None)?;
        dev.synchronize()?;
        prev = check("chained", n, &prev, &d_ca, &d_cb, &d_sa, &d_sb)?;
    }
    // prof twins must compute the same bits too (they are the anatomy's instrument)
    dev.htod_sync_copy_into(&st0, &mut d_sa)?; dev.htod_sync_copy_into(&st0, &mut d_sb)?;
    poison(&mut d_ca)?; poison(&mut d_cb)?;
    launch(f_oldp, pca, psa, 0, nmax, Some(*d_prof.device_ptr() as u64))?;
    launch(f_newp, pcb, psb, 0, nmax, Some(*d_prof.device_ptr() as u64))?;
    dev.synchronize()?;
    check("prof-twins", nmax, &st0, &d_ca, &d_cb, &d_sa, &d_sb)?;
    println!("EXL3-GDN-TC2: PASS (A5-L1: xq_gdn_chunk_tc2 (wavefront solve, fused U, in-warp scan, register prefetch) bitwise == \
              xq_gdn_chunk_tc (split 7) at nh 48 / nk 16 / kd = vd = 128, slot 1 of 2; {nchk} comparisons over N in {{2048, 775, 96, 16}} \
              fresh + chained, NaN-poisoned core)");
    // ---- harness-only timing + anatomy (ranks only; the served A/B prices it)
    let reps = 20usize;
    let mut tm = |f: sys::CUfunction, core: u64, st: u64| -> Result<f64> {
        launch(f, core, st, 0, nmax, None)?; dev.synchronize()?;
        let t0 = std::time::Instant::now();
        for _ in 0..reps { launch(f, core, st, 0, nmax, None)?; }
        dev.synchronize()?;
        Ok(t0.elapsed().as_secs_f64() * 1e3 / reps as f64)
    };
    let (t_old, t_new) = (tm(f_old, pca, psa)?, tm(f_new, pcb, psb)?);
    let (t_old2, t_new2) = (tm(f_old, pca, psa)?, tm(f_new, pcb, psb)?);
    println!("    EXL3-GDN-TC2 timing N=2048 (harness-only, ms/call; x36 = per chunk): chunk_tc {t_old:.3}/{t_old2:.3} -> tc2 {t_new:.3}/{t_new2:.3} \
              | per chunk {:.1} -> {:.1} ms", 36.0 * t_old.min(t_old2), 36.0 * t_new.min(t_new2));
    for (nm, f, core, st) in [("chunk_tc", f_oldp, pca, psa), ("tc2", f_newp, pcb, psb)] {
        launch(f, core, st, 0, nmax, Some(*d_prof.device_ptr() as u64))?;
        dev.synchronize()?;
        let pr = dev.dtoh_sync_copy(&d_prof)?;
        let nb = grid as usize;
        let nsub = (nmax / 32) as f64;
        let mut ph = [0f64; 8];
        let (mut cyc, mut ns, mut g0, mut g1) = (0f64, 0f64, u64::MAX, 0u64);
        for b in 0..nb {
            for i in 0..8 { ph[i] += pr[b * 10 + i] as f64 / nb as f64 / nsub; }
            cyc += (0..7).map(|i| pr[b * 10 + i] as f64).sum::<f64>();
            ns += (pr[b * 10 + 9] - pr[b * 10 + 8]) as f64;
            g0 = g0.min(pr[b * 10 + 8]); g1 = g1.max(pr[b * 10 + 9]);
        }
        let ghz = cyc / ns;
        println!("    EXL3-GDN-TC2 anatomy {nm} (cycles per 32-token sub-chunk, block-avg; {ghz:.2} GHz; block {:.1} us; span {:.3} ms): \
                  stage {:.0} | scan {:.0} | grams+W {:.0} | solve {:.0} | U {:.0} | O {:.0} | S {:.0} | total {:.0}",
                 ns / nb as f64 / 1e3, (g1 - g0) as f64 / 1e6, ph[0], ph[1], ph[2], ph[3], ph[4], ph[5], ph[6],
                 ph[..7].iter().sum::<f64>());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// A5-P2: the prefill hc elementwise twins must equal the served chain BITWISE at the served shape
// (h 2560, hc 4) over c in {2048, 775, 96, 16}, outputs NaN-poisoned first:
//   xq_pf_hc_inj_norm(y)    == xq_hc_inject + xq_hc_norm   (resid f32 AND hn f16)
//   xq_pf_hc_inj_norm(null) == xq_hc_norm                  (hn; resid untouched)
//   xq_pf_hc_mix_2560x4     == xq_hc_mix4                  (x f16 AND inj f32; with and without winj)
// Inputs: resid rows with per-row magnitudes spanning 1e-3..1e3 (+ exact zeros / subnormal-scale
// rows), the mix fed the REAL norm output. Also a harness-only timing at c = 2048 (x96 = per chunk).
fn probe_pf_hc(dev: &std::sync::Arc<CudaDevice>) -> Result<()> {
    let (h, hc) = (2560usize, 4usize);
    let n = h * hc;
    let eps = 1e-6f32;
    let f = |nm: &str| dev.get_func(MODULE, nm).with_context(|| nm.to_string());
    let lc = |g: u32, b: u32, sm: u32| LaunchConfig { grid_dim: (g, 1, 1), block_dim: (b, 1, 1), shared_mem_bytes: sm };
    let mut st = 0x9E37_79B9u32;
    let mut unif = move || { st = st.wrapping_mul(1664525).wrapping_add(1013904223); ((st >> 8) as f32 + 0.5) / 16777216.0 };
    let cmax = 2048usize;
    // resid: row r scale 10^((r % 7) - 3); every 13th row exactly zero in its first stream
    let mut resid0 = vec![0f32; cmax * n];
    for r in 0..cmax {
        let sc = 10f32.powi((r % 7) as i32 - 3);
        for i in 0..n {
            let v = (unif() * 2.0 - 1.0) * sc * (1.0 + 3.0 * (i % 97 == 0) as i32 as f32);
            resid0[r * n + i] = if r % 13 == 5 && i < h { 0.0 } else { v };
        }
    }
    let y = wp20_synth(cmax * h, 0xB0A1, 0.5);
    let injv: Vec<f32> = (0..cmax * hc).map(|_| unif() * 2.0).collect();
    let w = wp20_synth(n, 0xB0A2, 0.05);
    let uu = wp20_synth(cmax * n, 0xB0A3, 1.5);
    let winj = wp20_synth(hc * n, 0xB0A4, 0.01);
    let d_r0 = dev.htod_sync_copy(&resid0)?;
    let d_y = dev.htod_sync_copy(&y)?;
    let d_inj0 = dev.htod_sync_copy(&injv)?;
    let d_w = dev.htod_sync_copy(&w)?;
    let d_uu = dev.htod_sync_copy(&uu)?;
    let d_winj = dev.htod_sync_copy(&winj)?;
    let mut d_ra = dev.alloc_zeros::<f32>(cmax * n)?;
    let mut d_rb = dev.alloc_zeros::<f32>(cmax * n)?;
    let mut d_hna = dev.alloc_zeros::<u16>(cmax * n)?;
    let mut d_hnb = dev.alloc_zeros::<u16>(cmax * n)?;
    let mut d_xa = dev.alloc_zeros::<u16>(cmax * h)?;
    let mut d_xb = dev.alloc_zeros::<u16>(cmax * h)?;
    let mut d_ia = dev.alloc_zeros::<f32>(cmax * hc)?;
    let mut d_ib = dev.alloc_zeros::<f32>(cmax * hc)?;
    let (f_inject, f_norm, f_mix4, f_injn, f_mix) =
        (f("xq_hc_inject")?, f("xq_hc_norm")?, f("xq_hc_mix4")?, f("xq_pf_hc_inj_norm")?, f("xq_pf_hc_mix_2560x4")?);
    let yp = *d_y.device_ptr() as u64;
    let winjp = *d_winj.device_ptr() as u64;
    let pz16 = |d: &mut CudaSlice<u16>| -> Result<()> { let k = d.len(); dev.htod_sync_copy_into(&vec![0x7E00u16; k], d)?; Ok(()) };
    let pz32 = |d: &mut CudaSlice<f32>| -> Result<()> { let k = d.len(); dev.htod_sync_copy_into(&vec![f32::NAN; k], d)?; Ok(()) };
    let cmp16 = |c: usize, what: &str, a: &[u16], b: &[u16]| -> Result<()> {
        let bad = a.iter().zip(b.iter()).filter(|(p, q)| p != q).count();
        if bad > 0 || a.len() != b.len() {
            bail!("EXL3-PF-HC FAIL c={c} {what}: {bad} of {} f16 differ (first {:?})", a.len(), a.iter().zip(b.iter()).position(|(p, q)| p != q));
        }
        if a.iter().any(|&v| v == 0x7E00) { bail!("EXL3-PF-HC FAIL c={c} {what}: poisoned value left in the reference"); }
        Ok(())
    };
    let cmp32 = |c: usize, what: &str, a: &[f32], b: &[f32]| -> Result<()> {
        let bad = a.iter().zip(b.iter()).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
        if bad > 0 || a.len() != b.len() {
            bail!("EXL3-PF-HC FAIL c={c} {what}: {bad} of {} f32 differ (first {:?})", a.len(),
                  a.iter().zip(b.iter()).position(|(p, q)| p.to_bits() != q.to_bits()));
        }
        if a.iter().any(|v| v.is_nan()) { bail!("EXL3-PF-HC FAIL c={c} {what}: NaN in the reference"); }
        Ok(())
    };
    let mut nchk = 0usize;
    for &c in &[2048usize, 775, 96, 16] {
        let (ci, hci, hi) = (c as i32, hc as i32, h as i32);
        // ---- inject + norm
        dev.dtod_copy(&d_r0, &mut d_ra)?; dev.dtod_copy(&d_r0, &mut d_rb)?;
        pz16(&mut d_hna)?; pz16(&mut d_hnb)?;
        unsafe {
            f_inject.clone().launch(lc(((n * c) as u32) / 256 + 1, 256, 0), (&mut d_ra, yp, &d_inj0, hi, hci, ci))?;
            f_norm.clone().launch(lc((hc * c) as u32, 1024, 4096), (&mut d_hna, &d_ra, &d_w, hi, hci, ci, eps))?;
            f_injn.clone().launch(lc((hc * c) as u32, 128, 0), (&mut d_hnb, &mut d_rb, yp, &d_inj0, &d_w, hi, hci, ci, eps))?;
        }
        dev.synchronize()?;
        cmp32(c, "inj_norm resid", &dev.dtoh_sync_copy(&d_ra)?[..c * n], &dev.dtoh_sync_copy(&d_rb)?[..c * n])?;
        let hn_ref = dev.dtoh_sync_copy(&d_hna)?;
        cmp16(c, "inj_norm hn", &hn_ref[..c * n], &dev.dtoh_sync_copy(&d_hnb)?[..c * n])?;
        nchk += 2;
        // ---- plain norm (y = null): the first hc_pre of a chunk / after the PLE flush
        dev.dtod_copy(&d_r0, &mut d_rb)?;
        pz16(&mut d_hna)?; pz16(&mut d_hnb)?;
        unsafe {
            f_norm.clone().launch(lc((hc * c) as u32, 1024, 4096), (&mut d_hna, &d_r0, &d_w, hi, hci, ci, eps))?;
            f_injn.clone().launch(lc((hc * c) as u32, 128, 0), (&mut d_hnb, &mut d_rb, 0u64, &d_inj0, &d_w, hi, hci, ci, eps))?;
        }
        dev.synchronize()?;
        cmp16(c, "norm hn", &dev.dtoh_sync_copy(&d_hna)?[..c * n], &dev.dtoh_sync_copy(&d_hnb)?[..c * n])?;
        cmp32(c, "norm resid untouched", &resid0[..c * n], &dev.dtoh_sync_copy(&d_rb)?[..c * n])?;
        nchk += 2;
        // ---- mix (on the real inject+norm output), with and without winj
        let d_hn = dev.htod_sync_copy(&hn_ref)?;
        for with_w in [true, false] {
            let wp = if with_w { winjp } else { 0u64 };
            pz16(&mut d_xa)?; pz16(&mut d_xb)?; pz32(&mut d_ia)?; pz32(&mut d_ib)?;
            unsafe {
                f_mix4.clone().launch(lc(c as u32, 1024, 4 * 1024 * 4), (&mut d_xa, &mut d_ia, &d_hn, &d_uu, wp, hi, hci, ci))?;
                f_mix.clone().launch(lc(c as u32, 128, 0), (&mut d_xb, &mut d_ib, &d_hn, &d_uu, wp, ci))?;
            }
            dev.synchronize()?;
            cmp16(c, &format!("mix x (winj {})", with_w as u8), &dev.dtoh_sync_copy(&d_xa)?[..c * h], &dev.dtoh_sync_copy(&d_xb)?[..c * h])?;
            nchk += 1;
            if with_w {
                cmp32(c, "mix inj", &dev.dtoh_sync_copy(&d_ia)?[..c * hc], &dev.dtoh_sync_copy(&d_ib)?[..c * hc])?;
                nchk += 1;
            }
        }
        println!("  EXL3-PF-HC c={c}: inject+norm, norm, mix (winj 1|0) bitwise == the served chain");
        // ---- timing (harness-only; 84 MB f32 state per call at c = 2048)
        if c == 2048 {
            let reps = 20usize;
            let mut t = |run: &mut dyn FnMut() -> Result<()>| -> Result<f64> {
                run()?; dev.synchronize()?;
                let t0 = std::time::Instant::now();
                for _ in 0..reps { run()?; }
                dev.synchronize()?;
                Ok(t0.elapsed().as_secs_f64() * 1e3 / reps as f64)
            };
            let t_old_in = t(&mut || { unsafe {
                f_inject.clone().launch(lc(((n * c) as u32) / 256 + 1, 256, 0), (&mut d_ra, yp, &d_inj0, hi, hci, ci))?;
                f_norm.clone().launch(lc((hc * c) as u32, 1024, 4096), (&mut d_hna, &d_ra, &d_w, hi, hci, ci, eps))?; } Ok(()) })?;
            let t_new_in = t(&mut || { unsafe {
                f_injn.clone().launch(lc((hc * c) as u32, 128, 0), (&mut d_hnb, &mut d_rb, yp, &d_inj0, &d_w, hi, hci, ci, eps))?; } Ok(()) })?;
            let t_old_mix = t(&mut || { unsafe {
                f_mix4.clone().launch(lc(c as u32, 1024, 4 * 1024 * 4), (&mut d_xa, &mut d_ia, &d_hn, &d_uu, winjp, hi, hci, ci))?; } Ok(()) })?;
            let t_new_mix = t(&mut || { unsafe {
                f_mix.clone().launch(lc(c as u32, 128, 0), (&mut d_xb, &mut d_ib, &d_hn, &d_uu, winjp, ci))?; } Ok(()) })?;
            let gbs = |mb: f64, ms: f64| mb / ms;
            println!("    EXL3-PF-HC timing c=2048 (harness-only, ms/call; x96 = per chunk): inject+norm {t_old_in:.3} -> inj_norm {t_new_in:.3} \
                      ({:.0} GB/s on 220 MB) | mix4 {t_old_mix:.3} -> pf_mix {t_new_mix:.3} ({:.0} GB/s on 94 MB) | per chunk {:.1} -> {:.1} ms",
                     gbs(220.0, t_new_in), gbs(94.4, t_new_mix), 96.0 * (t_old_in + t_old_mix), 96.0 * (t_new_in + t_new_mix));
        }
    }
    println!("EXL3-PF-HC: PASS (A5-P2: xq_pf_hc_inj_norm (inject+norm and norm-only) and xq_pf_hc_mix_2560x4 (winj and final-mixer) \
              bitwise == xq_hc_inject + xq_hc_norm / xq_hc_mix4 at h 2560 hc 4; {nchk} comparisons over c in {{2048, 775, 96, 16}}, NaN-poisoned outputs)");
    Ok(())
}

// ---------------------------------------------------------------------------
// HC (decode wave 4): xq_hc_inj_norm + xq_hc_w4 must equal the p4c hc path (xq_hc_inject +
// xq_hc_norm + xq_hc_fuse_i8k (m = 1) / xq_hc_fuse_i8_rbk (m >= 2) + xq_hc_mix_rg) BITWISE on every
// output — resid, hn, dd, uu, x, inj — for m = 1..8, over grids {default, 8 (every wrap path: several
// units / gates / tiles per CTA, gates on unit CTAs), max} and flag sets (mix fused / separate, uu
// written or not, prefetch 0/1/2, look-ahead), with and without the inject gates (the final mixer),
// each launched twice (grid-barrier re-arm); outputs poisoned before every run. Batch invariance:
// row r of every width == row r of width 8. Then a harness-only, DRAM-cold (8 weight copies) timing.
// ---------------------------------------------------------------------------
fn probe_hc_w4(dev: &std::sync::Arc<CudaDevice>) -> Result<()> {
    let (h, hc, lr, mmax) = (2560usize, 4usize, 320usize, 8usize);
    let rw = h * hc;
    let eps = 1e-6f32;
    let cap = crate::exl3_forward::w4hc_capacity();
    if cap == 0 { bail!("EXL3-HC-W4 FAIL: xq_hc_w4 co-residency probe returned 0 (stale PTX / carveout pin refused)"); }
    let gdef = crate::exl3_forward::w4hc_default_grid(cap, h);
    let (gmin, gmax) = (h.div_ceil(32) as u32, (cap * 5) / 6);
    if gdef == 0 { bail!("EXL3-HC-W4 FAIL: co-residency {cap} x 5/6 below the phase-B minimum grid {gmin}"); }
    println!("  HC-W4: xq_hc_w4 co-residency {cap} blocks (256t, carveout max) -> default grid {gdef}, min {gmin}, max {gmax}");
    // ---- synthetic mixer: int8 over the FULL code range (incl. -128), fp32 per-row scales
    let mut st = 0x9E37_79B9u32;
    let mut rnd = move || { st ^= st << 13; st ^= st >> 17; st ^= st << 5; st };
    let mut mixer = |seed: u32| -> (Vec<i8>, Vec<f32>, Vec<i8>, Vec<f32>, Vec<u16>) {
        let qd: Vec<i8> = (0..lr * rw).map(|_| (rnd() >> 24) as u8 as i8).collect();
        let sd: Vec<f32> = (0..lr).map(|_| 4e-4 + (rnd() % 1000) as f32 * 1e-6).collect();
        let qu: Vec<i8> = (0..rw * lr).map(|_| (rnd() >> 24) as u8 as i8).collect();
        let su: Vec<f32> = (0..rw).map(|_| 6e-4 + (rnd() % 1000) as f32 * 1e-6).collect();
        let mut uk = vec![0i8; rw * lr];         // WP11 R3 k-major relayout (hc_quantize_i8)
        for n in 0..rw { for k in 0..lr { uk[((k / 16) * rw + n) * 16 + (k % 16)] = qu[n * lr + k]; } }
        (qd, sd, uk, su, wp20_synth(hc * rw, 0x51 ^ seed, 0.01))
    };
    let (qd, sd, qu, su, winj) = mixer(1);
    let (d_qd, d_sd, d_qu, d_su, d_winj) = (dev.htod_sync_copy(&qd)?, dev.htod_sync_copy(&sd)?,
        dev.htod_sync_copy(&qu)?, dev.htod_sync_copy(&su)?, dev.htod_sync_copy(&winj)?);
    let d_nw = dev.htod_sync_copy(&wp20_synth(hc * h, 0x77, 0.05))?;
    // per-row inputs (row r identical at every width: synth_x is row-seeded)
    let resid0: Vec<f32> = synth_x(mmax, rw, 0x4E51).iter().map(|&b| f16::from_bits(b).to_f32() * 0.5).collect();
    let y0 = synth_x(mmax, h, 0x7A11);
    let inj0: Vec<f32> = (0..mmax * hc).map(|i| 0.25 + 0.07 * i as f32).collect();
    let d_y = dev.htod_sync_copy(&y0)?;
    let mut d_resid = dev.htod_sync_copy(&resid0)?;
    let mut d_inj = dev.htod_sync_copy(&inj0)?;
    let mut d_hn = dev.htod_sync_copy(&vec![0u16; mmax * rw])?;
    let mut d_dd = dev.htod_sync_copy(&vec![0u16; mmax * lr])?;
    let mut d_uu = dev.htod_sync_copy(&vec![0u16; mmax * rw])?;
    let mut d_x = dev.htod_sync_copy(&vec![0u16; mmax * h])?;
    let d_bar = dev.htod_sync_copy(&[0i32])?;
    let f = |n: &str| dev.get_func(MODULE, n).with_context(|| format!("{n} missing"));
    let (f_inject, f_norm, f_i8k, f_rbk, f_mix, f_injn) = (f("xq_hc_inject")?, f("xq_hc_norm")?, f("xq_hc_fuse_i8k")?,
        f("xq_hc_fuse_i8_rbk")?, f("xq_hc_mix_rg")?, f("xq_hc_inj_norm")?);
    let lc = |g: (u32, u32, u32), b: u32, sm: u32| LaunchConfig { grid_dim: g, block_dim: (b, 1, 1), shared_mem_bytes: sm };
    let dp = |p: &cudarc::driver::sys::CUdeviceptr| *p as u64;
    struct Out { resid: Vec<f32>, hn: Vec<u16>, dd: Vec<u16>, uu: Vec<u16>, x: Vec<u16>, inj: Vec<f32> }
    // weights of one mixer copy, as device pointers: (qd, sd, qu, su, winj)
    type W = (u64, u64, u64, u64, u64);
    let w0: W = (dp(d_qd.device_ptr()), dp(d_sd.device_ptr()), dp(d_qu.device_ptr()), dp(d_su.device_ptr()),
                 dp(d_winj.device_ptr()));
    let hcn = hc as f32;
    // p4c fuse (+ separate mix) on hn -> dd/uu/x/inj; `gated` = inject gates present
    let old_fuse_mix = |d_dd: &mut CudaSlice<u16>, d_uu: &mut CudaSlice<u16>, d_x: &mut CudaSlice<u16>,
                        d_inj: &mut CudaSlice<f32>, d_hn: &CudaSlice<u16>, w: W, m: usize, gated: bool,
                        fuse: bool| -> Result<()> {
        if fuse {
            if m == 1 {
                unsafe { f_i8k.clone().launch(lc((40, 1, 1), 256, 0), (&mut *d_dd, &mut *d_uu, d_hn, w.0, w.1, w.2, w.3,
                    &d_bar, hcn, lr as i32, rw as i32, 1i32)) }?;
            } else {
                unsafe { f_rbk.clone().launch(lc((40, 1, 1), 256, 0), (&mut *d_dd, &mut *d_uu, d_hn, w.0, w.1, w.2, w.3,
                    &d_bar, hcn, (lr as i32) | ((m as i32) << 16), rw as i32, 17i32)) }?;
            }
        }
        let gx = (if gated { hc } else { 0 } + (h + 1023) / 1024) as u32;
        let wi = if gated { w.4 } else { 0u64 };
        unsafe { f_mix.clone().launch(lc((gx, m as u32, 1), 1024, 0), (&mut *d_x, &mut *d_inj, d_hn, &*d_uu, wi,
            h as i32, hc as i32, m as i32)) }?;
        Ok(())
    };
    let new_fuse = |w: W, m: usize, gated: bool, g: u32, flags: i32, ptrs: [u64; 5]| -> Result<()> {
        // ptrs: dd, uu, x, inj, hn
        crate::exl3_forward::w4hc_launch(std::ptr::null_mut(), g,
            [ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], w.0, w.1, w.2, w.3, if gated { w.4 } else { 0 }, dp(d_bar.device_ptr())],
            hcn, lr, m, h, hc, flags)
    };
    let poison = |d_hn: &mut CudaSlice<u16>, d_dd: &mut CudaSlice<u16>, d_uu: &mut CudaSlice<u16>, d_x: &mut CudaSlice<u16>,
                  d_resid: &mut CudaSlice<f32>, d_inj: &mut CudaSlice<f32>| -> Result<()> {
        dev.htod_copy_into(vec![0x7E00u16; mmax * rw], d_hn)?;
        dev.htod_copy_into(vec![0x7E01u16; mmax * lr], d_dd)?;
        dev.htod_copy_into(vec![0x7E02u16; mmax * rw], d_uu)?;
        dev.htod_copy_into(vec![0x7E03u16; mmax * h], d_x)?;
        dev.htod_copy_into(resid0.clone(), d_resid)?;
        dev.htod_copy_into(inj0.clone(), d_inj)?;
        Ok(())
    };
    let read = |d_resid: &CudaSlice<f32>, d_hn: &CudaSlice<u16>, d_dd: &CudaSlice<u16>, d_uu: &CudaSlice<u16>,
                d_x: &CudaSlice<u16>, d_inj: &CudaSlice<f32>| -> Result<Out> {
        dev.synchronize()?;
        Ok(Out { resid: dev.dtoh_sync_copy(d_resid)?, hn: dev.dtoh_sync_copy(d_hn)?, dd: dev.dtoh_sync_copy(d_dd)?,
                 uu: dev.dtoh_sync_copy(d_uu)?, x: dev.dtoh_sync_copy(d_x)?, inj: dev.dtoh_sync_copy(d_inj)? })
    };
    // (mismatches over the m live rows, first index) per buffer; uu skipped when `uu` is false
    let cmp = |a: &Out, b: &Out, m: usize, uu: bool| -> ([usize; 6], Option<(usize, usize)>) {
        fn d<T: PartialEq + Copy>(x: &[T], y: &[T], neq: impl Fn(T, T) -> bool) -> (usize, Option<usize>) {
            let mut k = (0usize, None);
            for (i, (&p, &q)) in x.iter().zip(y.iter()).enumerate() { if neq(p, q) { k.0 += 1; k.1.get_or_insert(i); } }
            k
        }
        let f32n = |p: f32, q: f32| p.to_bits() != q.to_bits();
        let u16n = |p: u16, q: u16| p != q;
        let r = [d(&a.resid[..m * rw], &b.resid[..m * rw], f32n), d(&a.hn[..m * rw], &b.hn[..m * rw], u16n),
                 d(&a.dd[..m * lr], &b.dd[..m * lr], u16n),
                 if uu { d(&a.uu[..m * rw], &b.uu[..m * rw], u16n) } else { (0, None) },
                 d(&a.x[..m * h], &b.x[..m * h], u16n), d(&a.inj[..m * hc], &b.inj[..m * hc], f32n)];
        let first = r.iter().enumerate().find_map(|(bi, k)| k.1.map(|i| (bi, i)));
        ([r[0].0, r[1].0, r[2].0, r[3].0, r[4].0, r[5].0], first)
    };
    const MIX: i32 = 1;
    const UU: i32 = 2;
    let cfgs: Vec<(&str, u32, i32)> = vec![
        ("default", gdef, MIX | UU | (1 << 2)),
        ("grid min (gates on unit CTAs at m >= 2), bulk pf, look-ahead 2", gmin, MIX | UU | (2 << 2) | (2 << 4)),
        ("grid max, no pf, look-ahead 1", gmax, MIX | UU | (1 << 4)),
        ("mix separate (uu only)", gdef, 1 << 2),
        ("default, uu not written", gdef, MIX | (1 << 2)),
    ];
    let (mut fails, mut checked) = (0usize, 0usize);
    let mut rows_by_m: Vec<(usize, bool, Out)> = Vec::new();
    for gated in [true, false] {
        for m in 1..=mmax {
            // ---- p4c reference
            poison(&mut d_hn, &mut d_dd, &mut d_uu, &mut d_x, &mut d_resid, &mut d_inj)?;
            unsafe { f_inject.clone().launch(lc((((h * hc * m) as u32) / 256 + 1, 1, 1), 256, 0),
                (&mut d_resid, &d_y, &d_inj, h as i32, hc as i32, m as i32)) }?;
            unsafe { f_norm.clone().launch(lc(((hc * m) as u32, 1, 1), 1024, 4096),
                (&mut d_hn, &d_resid, &d_nw, h as i32, hc as i32, m as i32, eps)) }?;
            old_fuse_mix(&mut d_dd, &mut d_uu, &mut d_x, &mut d_inj, &d_hn, w0, m, gated, true)?;
            let refo = read(&d_resid, &d_hn, &d_dd, &d_uu, &d_x, &d_inj)?;
            let nzx = refo.x[..m * h].iter().filter(|&&b| b & 0x7fff != 0 && b & 0x7c00 != 0x7c00).count();
            if nzx < m * h / 2 { bail!("EXL3-HC-W4 FAIL: implausible reference x (m={m}: {nzx} finite nonzero of {})", m * h); }
            // ---- the new path, every config, twice
            for (name, g, flags) in &cfgs {
                if *g < gmin || *g > gmax { continue; }
                for rep in 0..2 {
                    poison(&mut d_hn, &mut d_dd, &mut d_uu, &mut d_x, &mut d_resid, &mut d_inj)?;
                    unsafe { f_injn.clone().launch(lc(((hc * m) as u32, 1, 1), 1024, 0),
                        (&mut d_hn, &mut d_resid, &d_y, &d_inj, &d_nw, h as i32, hc as i32, m as i32, eps)) }?;
                    let ptrs = [dp(d_dd.device_ptr()), dp(d_uu.device_ptr()), dp(d_x.device_ptr()), dp(d_inj.device_ptr()),
                                dp(d_hn.device_ptr())];
                    new_fuse(w0, m, gated, *g, *flags, ptrs)?;
                    if flags & MIX == 0 {
                        old_fuse_mix(&mut d_dd, &mut d_uu, &mut d_x, &mut d_inj, &d_hn, w0, m, gated, false)?;
                    }
                    let o = read(&d_resid, &d_hn, &d_dd, &d_uu, &d_x, &d_inj)?;
                    let (k, first) = cmp(&refo, &o, m, flags & MIX == 0 || flags & UU != 0);
                    checked += 1;
                    if k.iter().sum::<usize>() > 0 {
                        fails += 1;
                        println!("    HC-W4 FAIL gates={gated} m={m} [{name}] G={g} rep={rep}: mismatches resid/hn/dd/uu/x/inj {k:?} \
                                  (first (buffer, idx) {first:?})");
                    }
                    if *name == "default" && rep == 0 { rows_by_m.push((m, gated, o)); }
                }
            }
        }
    }
    // ---- batch invariance: row r at width m == row r at width 8 (same gating)
    let mut binv_bad = 0usize;
    for gated in [true, false] {
        let full = rows_by_m.iter().find(|(m, g, _)| *m == mmax && *g == gated).map(|t| &t.2)
            .context("HC-W4: missing width-8 run")?;
        for (m, _, o) in rows_by_m.iter().filter(|(_, g, _)| *g == gated) {
            let (k, first) = cmp(full, o, *m, true);
            if k.iter().sum::<usize>() > 0 {
                binv_bad += 1;
                println!("    HC-W4 BINV FAIL gates={gated} width {m} vs 8: {k:?} (first {first:?})");
            }
        }
    }
    if fails + binv_bad > 0 {
        bail!("EXL3-HC-W4 FAIL ({fails} of {checked} runs differ from the p4c path, {binv_bad} width rows differ)");
    }
    println!("EXL3-HC-W4: PASS (xq_hc_inj_norm + xq_hc_w4 bitwise == xq_hc_inject + xq_hc_norm + i8k/rbk + mix_rg on \
              resid/hn/dd/uu/x/inj, M 1..=8, gated + final-mixer, {checked} runs over grids {{{gdef}, {gmin}, {gmax}}} x mix \
              fused/separate x pf 0/1/2 x look-ahead; batch-invariant rows across widths 1..8)");

    // ---- standalone timing (harness-only): 8 weight copies (52 MB > L2) rotated => DRAM-cold weights
    let nw = 8usize;
    let mut copies: Vec<(CudaSlice<i8>, CudaSlice<f32>, CudaSlice<i8>, CudaSlice<f32>, CudaSlice<u16>)> = Vec::new();
    for i in 0..nw {
        let (a, b, c, d, e) = mixer(100 + i as u32);
        copies.push((dev.htod_sync_copy(&a)?, dev.htod_sync_copy(&b)?, dev.htod_sync_copy(&c)?, dev.htod_sync_copy(&d)?,
                     dev.htod_sync_copy(&e)?));
    }
    let ws: Vec<W> = copies.iter().map(|c| (dp(c.0.device_ptr()), dp(c.1.device_ptr()), dp(c.2.device_ptr()),
                                            dp(c.3.device_ptr()), dp(c.4.device_ptr()))).collect();
    let iters = 30 * nw;
    let time = |run: &mut dyn FnMut(usize) -> Result<()>| -> Result<f64> {
        for it in 0..nw { run(it)?; }
        dev.synchronize()?;
        let t = std::time::Instant::now();
        for it in 0..iters { run(it)?; }
        dev.synchronize()?;
        Ok(t.elapsed().as_secs_f64() * 1e6 / iters as f64)
    };
    for m in [1usize, 2, 4, 6, 8] {
        let t_old = time(&mut |it| old_fuse_mix(&mut d_dd, &mut d_uu, &mut d_x, &mut d_inj, &d_hn, ws[it % nw], m, true, true))?;
        let ptrs = [dp(d_dd.device_ptr()), dp(d_uu.device_ptr()), dp(d_x.device_ptr()), dp(d_inj.device_ptr()),
                    dp(d_hn.device_ptr())];
        let t_new = time(&mut |it| new_fuse(ws[it % nw], m, true, gdef, MIX | (1 << 2), ptrs))?;
        let mut sweep = String::new();
        for g in [80u32, 88, 96, 112, 120, 144] {
            if g >= gmin && g <= gmax {
                let t = time(&mut |it| new_fuse(ws[it % nw], m, true, g, MIX | (1 << 2), ptrs))?;
                sweep += &format!(" G{g} {t:.1}");
            }
        }
        for (tag, fl) in [("pf0", MIX), ("pf2", MIX | (2 << 2)), ("la1", MIX | (1 << 2) | (1 << 4)),
                          ("la2", MIX | (1 << 2) | (2 << 4)), ("la3", MIX | (1 << 2) | (3 << 4))] {
            let t = time(&mut |it| new_fuse(ws[it % nw], m, true, gdef, fl, ptrs))?;
            sweep += &format!(" {tag} {t:.1}");
        }
        let t_in_old = time(&mut |_| {
            unsafe { f_inject.clone().launch(lc((((h * hc * m) as u32) / 256 + 1, 1, 1), 256, 0),
                (&mut d_resid, &d_y, &d_inj, h as i32, hc as i32, m as i32)) }?;
            unsafe { f_norm.clone().launch(lc(((hc * m) as u32, 1, 1), 1024, 4096),
                (&mut d_hn, &d_resid, &d_nw, h as i32, hc as i32, m as i32, eps)) }?;
            Ok(())
        })?;
        let t_in_new = time(&mut |_| {
            unsafe { f_injn.clone().launch(lc(((hc * m) as u32, 1, 1), 1024, 0),
                (&mut d_hn, &mut d_resid, &d_y, &d_inj, &d_nw, h as i32, hc as i32, m as i32, eps)) }?;
            Ok(())
        })?;
        println!("    EXL3-HC-W4 timing (standalone, harness-only, DRAM-cold weights) m={m}: p4c fuse+mix {t_old:.1} us | \
                  w4 G{gdef} {t_new:.1} us ({:+.1}) | sweep:{sweep} | inject+norm {t_in_old:.1} -> inj_norm {t_in_new:.1} us",
                 t_new - t_old);
        // TP-I3 (b): per-phase anatomy from the timestamp twin (same body), plain back-to-back launches;
        // cold = the 8 rotated weight copies (DRAM), warm = copy 0 only (L2-resident: the compute floor)
        for (arm, fl, warm) in [("cold pf1", MIX | (1 << 2), false), ("cold pf0", MIX, false), ("warm pf1", MIX | (1 << 2), true)] {
        let nl = 64usize;
        let gu = gdef as usize;
        const PS: usize = 16;
        let d_prof = dev.alloc_zeros::<u64>(nl * gu * PS)?;
        for it in 0..nl {
            let w = if warm { ws[0] } else { ws[it % nw] };
            crate::exl3_forward::w4hc_prof_launch(std::ptr::null_mut(), gdef,
                [ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], w.0, w.1, w.2, w.3, w.4, dp(d_bar.device_ptr())],
                dp(d_prof.device_ptr()) + (it * gu * PS * 8) as u64, hcn, lr, m, h, hc, fl)?;
        }
        dev.synchronize()?;
        let pr = dev.dtoh_sync_copy(&d_prof)?;
        // per launch, relative to the first CTA entry (us): [entry spread, wait end, A end (max), gates end (max),
        // barrier release (min t4), barrier last (max t4), up end (max), end (max), unit mean, silu staged (max)],
        // medians over launches nw..nl
        let mut acc: Vec<Vec<f64>> = vec![Vec::new(); 11];
        let nunits = if m >= 2 { 2 * lr.div_ceil(8) } else { lr.div_ceil(8) };
        for it in nw..nl {
            let p = &pr[it * gu * PS..(it + 1) * gu * PS];
            let col = |i: usize| (0..gu).map(move |c| p[c * PS + i]);
            let t0 = col(0).min().unwrap_or(0);
            let us = |t: u64| (t.saturating_sub(t0)) as f64 / 1e3;
            acc[0].push(us(col(0).max().unwrap()));
            acc[1].push(us(col(1).max().unwrap()));
            acc[2].push(us((0..gu.min(nunits)).map(|c| p[c * PS + 2]).max().unwrap()));
            acc[3].push(us(col(3).max().unwrap()));
            acc[4].push(us(col(4).min().unwrap()));
            acc[5].push(us(col(4).max().unwrap()));
            acc[6].push(us(col(5).max().unwrap()));
            acc[7].push(us(col(6).max().unwrap()));
            // phase-A unit duration (mean over unit CTAs, t2 - t1)
            let ua: f64 = (0..gu.min(nunits)).map(|c| (p[c * PS + 2] - p[c * PS + 1]) as f64 / 1e3).sum::<f64>()
                / gu.min(nunits) as f64;
            acc[8].push(ua);
            acc[9].push(us(col(8).max().unwrap()));
            // SM clock (MHz): per CTA clock64 cycles / globaltimer ns over its whole life, mean over CTAs
            let mhz: f64 = (0..gu).map(|c| (p[c * PS + 10].wrapping_sub(p[c * PS + 9])) as f64
                / ((p[c * PS + 6] - p[c * PS]).max(1) as f64) * 1e3).sum::<f64>() / gu as f64;
            acc[10].push(mhz);
        }
        let med = |v: &mut Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };
        let r: Vec<f64> = acc.iter_mut().map(med).collect();
        println!("    EXL3-HC-W4 anatomy m={m} G{gdef} {arm} (prof twin, median of {} launches, us from the first CTA entry): \
                  entry spread {:.1} | PDL wait end {:.1} | phase A end {:.1} (unit mean {:.1}) | gates end {:.1} | \
                  barrier release {:.1} (last {:.1}) | silu staged {:.1} | up end {:.1} | end {:.1} | SM clock {:.0} MHz",
                 nl - nw, r[0], r[1], r[2], r[8], r[3], r[4], r[5], r[9], r[6], r[7], r[10]);
        }
    }
    Ok(())
}

/// Deterministic f16 vector with values in about [-4, 4] * scale (LCG; synth_x's generator).
fn wp20_synth(n: usize, seed: u32, scale: f32) -> Vec<u16> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(12345);
    (0..n).map(|_| {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        let v = ((((s >> 8) % 4001) as f32 - 2000.0) / 512.0) * scale;
        f16::from_f32(v).to_bits()
    }).collect()
}

/// A5 WP20-v1c: the id of the epilogue's pinned suh sequence (kernels/exl3_bench.cu
/// wp20_suh_had), in v1's convention-word encoding (2 bits per op: add01 sub01 add23 sub23;
/// 1 = "first product fused": FMUL q = x1*s1, FFMA x0*s0 +- q) — exactly the op sequence of
/// xq_had_suh_multi's served SASS. Logged only; the kernels hard-code the sequence.
pub(crate) const WP20_SUH_SEQ: u32 = 0x55;

/// Deterministic f16 vector spanning the f16 domain's edge cases (xorshift; never inf/NaN).
/// A Hadamard output depends on all 128 inputs of its block, so the regime is chosen PER
/// 128-BLOCK (block b: b % 4): 0 = |v| in [2^-10, 32) with +-0 / subnormals sprinkled,
/// 1 = subnormals and tiny normals (|v| < 2^-8), 2 = large (|v| < 2^10: f16 outputs can
/// overflow to inf), 3 = any finite pattern incl. +-65504 (mostly saturating outputs).
fn wp20_synth_edge(n: usize, seed: u64) -> Vec<u16> {
    let mut s = seed | 1;
    (0..n).map(|i| {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        let (b, sm) = (s as u16, (s as u16) & 0x83FF);       // random sign + mantissa
        let e = |lo: u64, hi: u64| (((s >> 32) % (hi - lo + 1) + lo) as u16) << 10;
        match ((i / 128) % 4, (s >> 20) % 16) {
            (0, 0) => sm & 0x8000,                               // +-0
            (0, 1) => sm,                                        // +-subnormal (or 0)
            (0, _) => sm | e(5, 19),
            (1, _) => sm | e(0, 6),
            (2, _) => sm | e(10, 24),
            (_, 0) => (b & 0x8000) | 0x7BFF,                     // +-65504
            _ => if b & 0x7C00 == 0x7C00 { b & 0xBBFF } else { b },
        }
    }).collect()
}

/// A5 WP20-v1c: boot VERIFICATION of the epilogue's pinned suh sequence against the JIT'd
/// xq_had_suh_multi itself. v1/v1b searched 81 contraction conventions for a unique bitwise
/// match; x and s are f16, so x*s is exact in f32 and every convention yields the same bits —
/// all 81 matched and the search always reported "no unique match" (rung 2 OFF at boot on
/// preview p2). The epilogue now pins xq_had_suh_multi's served SASS op sequence with .rn
/// intrinsics (wp20_suh_had); this runs xq_wp20_suh_cal (xq_had_suh_multi's signature and
/// indexing, the pinned arithmetic) and the real kernel on the same inputs — per-expert input
/// form (the down transform's), 2 experts x 16 rows x 640: expert 0 = realistic activations and
/// suh (v1's synth rows), expert 1 = f16 edge cases (+-0, subnormals, +-65504, any finite;
/// f16-overflowing outputs included) — and requires every output byte equal.
/// Some(WP20_SUH_SEQ) = verified; None = a mismatch (rung 2 then stays off, loudly).
pub(crate) fn wp20_calibrate(dev: &std::sync::Arc<CudaDevice>) -> Result<Option<u32>> {
    let (e, r, k) = (2usize, 16usize, 640usize);
    let mut xs = wp20_synth(r * k, 0x20C0, 1.0);
    xs.extend(wp20_synth_edge(r * k, 0x20C1_5EED_0001));
    let mut ss = wp20_synth(k, 0x5CA1, 0.5);
    ss.extend(wp20_synth_edge(k, 0x5CA1_5EED_0002));
    let d_x = dev.htod_sync_copy(&xs)?;
    let d_s = dev.htod_sync_copy(&ss)?;
    let d_ix = dev.htod_sync_copy(&[0i32, 1])?;
    let d_es = dev.htod_sync_copy(&[e as i32])?;
    // poisoned outputs: a launch that skips a write can never compare equal by accident
    let mut d_ref = dev.htod_sync_copy(&vec![0x7E00u16; e * r * k])?;
    let mut d_out = dev.htod_sync_copy(&vec![0x7E01u16; e * r * k])?;
    let f_old = dev.get_func(MODULE, "xq_had_suh_multi").context("xq_had_suh_multi")?;
    let f_cal = dev.get_func(MODULE, "xq_wp20_suh_cal").context("xq_wp20_suh_cal (stale PTX?)")?;
    let g = (e * r * (k / 128)) as u32;
    let lc = LaunchConfig { grid_dim: (g, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
    dev.synchronize()?;
    unsafe {
        f_old.launch(lc, (&d_x, &d_s, &d_ix, &mut d_ref, r as i32, k as i32, &d_es, 1i32))?;
        f_cal.launch(lc, (&d_x, &d_s, &d_ix, &mut d_out, r as i32, k as i32, &d_es, 1i32))?;
    }
    dev.synchronize()?;
    let rf = dev.dtoh_sync_copy(&d_ref)?;
    let out = dev.dtoh_sync_copy(&d_out)?;
    // plausibility: the realistic expert is finite and mostly nonzero; nothing is a poison NaN
    let real = &rf[..r * k];
    if real.iter().filter(|v| **v & 0x7FFF != 0).count() < r * k / 2
        || real.iter().any(|v| *v & 0x7C00 == 0x7C00)
        || rf.iter().any(|v| *v & 0x7FFF > 0x7C00) {
        bail!("WP20 suh verification: implausible xq_had_suh_multi reference");
    }
    let diff = rf.iter().zip(&out).filter(|(p, q)| p != q).count();
    if diff != 0 {
        let i = rf.iter().zip(&out).position(|(p, q)| p != q).unwrap_or(0);
        println!("WP20 suh verification: pinned sequence differs from xq_had_suh_multi in {diff}/{} f16 \
                  (first [{i}] old {:04x} new {:04x})", e * r * k, rf[i], out[i]);
        return Ok(None);
    }
    Ok(Some(WP20_SUH_SEQ))
}

/// A5 WP20 gate (binv grouped section): the MoE epilogue kernels vs the OLD launch chain
/// (had_suh_multi -> a1b3 -> had_svh_multi -> gate_mul -> had_suh_multi -> a1b3 -> had_svh_multi
/// -> moe_combine) on synthetic 3-bit experts (h 2560, mi 640 — the served shapes) for EVERY
/// compiled down-epilogue top-k (WP20-v1b): top-8 over 12 experts (exactly the v1 case) and
/// top-10 over 64 experts (Qwen3.8-Flash-Next's top-k; up to ~50 live experts per launch):
/// ygu, xhd, yd and moe_out BITWISE, both decode bodies (diet / SHFL), widths {1,2,3,5,6,8,9}
/// (8 = the widened verify, 9 = the gate/up A-once cap), each run twice (the second launch rides
/// the re-armed counters), every counter back at 0. Grids padded to w*topk with the device live
/// count, as served. The per-top-k entry names come from the launcher's own table
/// (exl3_forward::wp20_dn_fn), so the gate exercises exactly what serving launches.
fn probe_moe_epi(dev: &std::sync::Arc<CudaDevice>) -> Result<()> {
    const NE_MAX: usize = 64;      // physical trellis sets (expert id e uses set e % NE_MAX)
    const NE_TAB: usize = 256;     // suh/svh rows (the 256-expert routing case: every id distinct)
    let (h, mi) = (2560usize, 640usize);
    let conv = match wp20_calibrate(dev)? {
        Some(c) => c,
        None => bail!("EXL3-MOE-EPI FAIL: the pinned suh sequence does not match xq_had_suh_multi (see the line above)"),
    };
    let gu_words = (h / 16) * (2 * mi / 16) * 48;
    let d_words = (mi / 16) * (h / 16) * 48;
    // NE_MAX experts' worth; the LCG makes the first 12 experts' bytes identical to v1's 12-expert set.
    let d_gu = dev.htod_sync_copy(&wp20_synth(NE_MAX * gu_words, 0x6001, 1.0))?;   // any u16 decodes finite
    let d_dn = dev.htod_sync_copy(&wp20_synth(NE_MAX * d_words, 0x6002, 1.0))?;
    // (NE_TAB rows: the LCG only appends, so the first NE_MAX rows are the earlier tables' bytes)
    let d_suh_gu = dev.htod_sync_copy(&wp20_synth(NE_TAB * h, 0x6003, 0.25))?;
    let d_svh_gu = dev.htod_sync_copy(&wp20_synth(NE_TAB * 2 * mi, 0x6004, 0.005))?;
    let d_suh_d = dev.htod_sync_copy(&wp20_synth(NE_TAB * mi, 0x6005, 0.02))?;
    let d_svh_d = dev.htod_sync_copy(&wp20_synth(NE_TAB * h, 0x6006, 0.01))?;
    let d_sg = dev.htod_sync_copy(&wp20_synth(h, 0x6007, 0.01))?;
    let f = |n: &str| dev.get_func(MODULE, n).with_context(|| n.to_string());
    let (f_suh, f_gemm, f_svh, f_gm, f_comb) =
        (f("xq_had_suh_multi")?, f("xq_gemm_grouped_a1b3")?, f("xq_had_svh_multi")?, f("xq_moe_gate_mul")?, f("xq_moe_combine")?);
    let lc = |g: u32, b: u32, s: u32| LaunchConfig { grid_dim: (g, 1, 1), block_dim: (b, 1, 1), shared_mem_bytes: s };
    let a1 = |m: usize, k: usize| (m * (k + 8) * 2) as u32;
    let mut done = Vec::new();
    let mut w4_done = 0usize;
    let mut w4_cfgs: Vec<String> = Vec::new();
    // (top-k, routed experts): v1's top-8 over 12; Qwen3.8-Flash-Next's top-10 over 64 (served-like
    // live counts: ~46 at m = 8); top-10 over 256 (ids all distinct: live = m * 10, the cap).
    for (topk, ne) in [(8usize, 12usize), (10, NE_MAX), (10, NE_TAB)] {
    let (dn_diet, dn_sh) = match (crate::exl3_forward::wp20_dn_fn(topk, true), crate::exl3_forward::wp20_dn_fn(topk, false)) {
        (Some(a), Some(b)) => (a, b),
        _ => bail!("EXL3-MOE-EPI FAIL: no down-epilogue instance for top-k {topk} in the launcher table"),
    };
    for m in 1usize..=9 {
        // the launcher's smem contract at this (m, top-k): tile + [m][top-k] tables inside a1(m, mi)
        if (a1(m, mi) as usize) < m * 256 + m * topk * 8 {
            bail!("EXL3-MOE-EPI FAIL: down epilogue smem {} B < tile + tables {} B at m={m} top-k {topk}",
                  a1(m, mi), m * 256 + m * topk * 8);
        }
        let emax = m * topk;
        // routing: row r picks topk distinct experts (LCG), positive weights summing to ~1
        let mut s = 0x9E37_79B9u32.wrapping_mul(m as u32 + 1);
        let mut ids = Vec::with_capacity(m * topk);
        let mut wts = Vec::with_capacity(m * topk);
        for _ in 0..m {
            let mut picked: Vec<i32> = Vec::new();
            while picked.len() < topk {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let e = ((s >> 8) % ne as u32) as i32;
                if !picked.contains(&e) { picked.push(e); }
            }
            let raw: Vec<f32> = (0..topk).map(|_| { s = s.wrapping_mul(1664525).wrapping_add(1013904223); 0.05 + ((s >> 8) % 1000) as f32 / 1000.0 }).collect();
            let sum: f32 = raw.iter().sum();
            ids.extend(picked);
            wts.extend(raw.iter().map(|v| v / sum));
        }
        // xq_moe_route's tables (first-seen over ids row-major)
        let mut slotmap = vec![-1i32; ne];
        let mut idxmap = vec![0i32; emax];
        let mut es = 0usize;
        for &e in &ids { if slotmap[e as usize] < 0 { slotmap[e as usize] = es as i32; idxmap[es] = e; es += 1; } }
        let offs_gu: Vec<u64> = (0..emax).map(|i| (idxmap[i] as usize % NE_MAX) as u64 * gu_words as u64).collect();
        let offs_d: Vec<u64> = (0..emax).map(|i| (idxmap[i] as usize % NE_MAX) as u64 * d_words as u64).collect();
        let (d_ids, d_wts, d_sm, d_ix) = (dev.htod_sync_copy(&ids)?, dev.htod_sync_copy(&wts)?,
                                         dev.htod_sync_copy(&slotmap)?, dev.htod_sync_copy(&idxmap)?);
        let (d_ogu, d_od, d_es) = (dev.htod_sync_copy(&offs_gu)?, dev.htod_sync_copy(&offs_d)?, dev.htod_sync_copy(&[es as i32])?);
        let d_x = dev.htod_sync_copy(&wp20_synth(m * h, 0x7000 + m as u32, 0.5))?;
        let d_ysh = dev.htod_sync_copy(&wp20_synth(m * h, 0x7100 + m as u32, 0.25))?;
        let z = |n: usize| dev.htod_sync_copy(&vec![0u16; n]);
        let (mut xh_e, mut ygu_raw, mut ygu, mut din) = (z(emax * m * h)?, z(emax * m * 2 * mi)?, z(emax * m * 2 * mi)?, z(emax * m * mi)?);
        let (mut xhd0, mut ydr, mut yd0, mut out0) = (z(emax * m * mi)?, z(emax * m * h)?, z(emax * m * h)?, z(m * h)?);
        // ---- the old chain
        unsafe {
            f_suh.clone().launch(lc((emax * m * (h / 128)) as u32, 32, 0),
                (&d_x, &d_suh_gu, &d_ix, &mut xh_e, m as i32, h as i32, &d_es, 0i32))?;
            f_gemm.clone().launch(lc((emax * (2 * mi / 128)) as u32, 256, a1(m, h)),
                (&d_gu, &d_ogu, &xh_e, &mut ygu_raw, m as i32, h as i32, (2 * mi) as i32, 3i32, &d_es))?;
            f_svh.clone().launch(lc((emax * m * (2 * mi / 128)) as u32, 32, 0),
                (&ygu_raw, &d_svh_gu, &d_ix, &mut ygu, m as i32, (2 * mi) as i32, &d_es))?;
            f_gm.clone().launch(lc(((emax * m * mi) / 256 + 1) as u32, 256, 0), (&mut din, &ygu, (m * mi) as i64, &d_es))?;
            f_suh.clone().launch(lc((emax * m * (mi / 128)) as u32, 32, 0),
                (&din, &d_suh_d, &d_ix, &mut xhd0, m as i32, mi as i32, &d_es, 1i32))?;
            f_gemm.clone().launch(lc((emax * (h / 128)) as u32, 256, a1(m, mi)),
                (&d_dn, &d_od, &xhd0, &mut ydr, m as i32, mi as i32, h as i32, 3i32, &d_es))?;
            f_svh.clone().launch(lc((emax * m * (h / 128)) as u32, 32, 0),
                (&ydr, &d_svh_d, &d_ix, &mut yd0, m as i32, h as i32, &d_es))?;
            f_comb.clone().launch(lc(m as u32, 1024, 4096),
                (&mut out0, &yd0, &d_ysh, &d_ids, &d_wts, &d_sm, &d_sg, &d_x, topk as i32, h as i32, m as i32))?;
        }
        dev.synchronize()?;
        let x_old = dev.dtoh_sync_copy(&xhd0)?;
        let g_old = dev.dtoh_sync_copy(&ygu)?;
        let y_old = dev.dtoh_sync_copy(&yd0)?;
        let o_old = dev.dtoh_sync_copy(&out0)?;
        let finite = |v: &[u16]| v.iter().all(|b| b & 0x7C00 != 0x7C00);
        if !finite(&o_old[..m * h]) || !finite(&x_old[..es * m * mi])
            || o_old[..m * h].iter().filter(|b| **b & 0x7FFF != 0).count() < m * h / 2 {
            bail!("EXL3-MOE-EPI FAIL: implausible reference (m={m}: non-finite or mostly-zero old-chain output)");
        }
        // A5-K7 fold row: xq_moe_gu_epi_f (xq_had_suh_multi folded into the gate/up prologue: reads x +
        // the gate/up suh table, never xh_e) + the diet down epilogue, vs the same old chain.
        for (diet, fold, gu, dn) in [(true, false, crate::exl3_forward::wp20_gu_fn(true), dn_diet),
                                     (false, false, crate::exl3_forward::wp20_gu_fn(false), dn_sh),
                                     (true, true, crate::exl3_forward::WP20_GU_FOLD_FN, dn_diet)] {
            let (mut ygu1, mut xhd1, mut yd1, mut out1) = (z(emax * m * 2 * mi)?, z(emax * m * mi)?, z(emax * m * h)?, z(m * h)?);
            let d_cnt = dev.htod_sync_copy(&vec![0u32; emax * (mi / 128) + h / 128 + 1])?;
            let mut d_sgv = dev.htod_sync_copy(&vec![0f32; m])?;
            let cnt = *d_cnt.device_ptr() as u64;
            let cnt_dn = cnt + (emax * (mi / 128) * 4) as u64;
            let fg = crate::exl3_forward::xq_raw_fn(gu).ok_or_else(|| anyhow::anyhow!("{gu} raw fn (stale PTX?)"))?;
            let fd = crate::exl3_forward::xq_raw_fn(dn).ok_or_else(|| anyhow::anyhow!("{dn} raw fn (stale PTX?)"))?;
            for rep in 0..2 {
                // NaN-poison every output first: a launch that skips a write can never pass on
                // the previous rep's bytes (the counters must really re-arm for rep 1).
                for (buf, n) in [(&mut ygu1, emax * m * 2 * mi), (&mut xhd1, emax * m * mi), (&mut yd1, emax * m * h),
                                 (&mut out1, m * h)] {
                    dev.htod_sync_copy_into(&vec![0x7E00u16; n], buf)?;
                }
                dev.htod_sync_copy_into(&vec![f32::NAN; m], &mut d_sgv)?;
                let mut a = [*d_gu.device_ptr() as u64, *d_ogu.device_ptr() as u64,
                             if fold { *d_x.device_ptr() as u64 } else { *xh_e.device_ptr() as u64 },
                             *ygu1.device_ptr() as u64];
                let mut d = [m as i32, h as i32, (2 * mi) as i32];
                let mut b = [*d_es.device_ptr() as u64, *d_ix.device_ptr() as u64, *d_svh_gu.device_ptr() as u64,
                             *d_suh_d.device_ptr() as u64, *xhd1.device_ptr() as u64, cnt, *d_x.device_ptr() as u64,
                             *d_sg.device_ptr() as u64, *d_sgv.device_ptr() as u64, *d_suh_gu.device_ptr() as u64];
                let mut p: [*mut std::ffi::c_void; 17] = [
                    &mut a[0] as *mut u64 as *mut _, &mut a[1] as *mut u64 as *mut _, &mut a[2] as *mut u64 as *mut _,
                    &mut a[3] as *mut u64 as *mut _, &mut d[0] as *mut i32 as *mut _, &mut d[1] as *mut i32 as *mut _,
                    &mut d[2] as *mut i32 as *mut _, &mut b[0] as *mut u64 as *mut _, &mut b[1] as *mut u64 as *mut _,
                    &mut b[2] as *mut u64 as *mut _, &mut b[3] as *mut u64 as *mut _, &mut b[4] as *mut u64 as *mut _,
                    &mut b[5] as *mut u64 as *mut _, &mut b[6] as *mut u64 as *mut _, &mut b[7] as *mut u64 as *mut _,
                    &mut b[8] as *mut u64 as *mut _, &mut b[9] as *mut u64 as *mut _,
                ];
                // (cuLaunchKernel reads exactly the kernel's own parameter count: 16 unfolded, 17 folded)
                let r1 = unsafe { cudarc::driver::sys::cuLaunchKernel(fg, (emax * (2 * mi / 128)) as u32, 1, 1, 256, 1, 1,
                                    a1(m, h), std::ptr::null_mut(), p.as_mut_ptr(), std::ptr::null_mut()) };
                let mut a2 = [*d_dn.device_ptr() as u64, *d_od.device_ptr() as u64, *xhd1.device_ptr() as u64,
                              *yd1.device_ptr() as u64];
                let mut d2 = [m as i32, mi as i32, h as i32];
                let mut b2 = [*d_es.device_ptr() as u64, *d_ix.device_ptr() as u64, *d_svh_d.device_ptr() as u64, cnt_dn,
                              *d_ysh.device_ptr() as u64, *d_ids.device_ptr() as u64, *d_wts.device_ptr() as u64,
                              *d_sm.device_ptr() as u64, *d_sgv.device_ptr() as u64, *out1.device_ptr() as u64];
                let mut k2 = topk as i32;
                let mut p2: [*mut std::ffi::c_void; 18] = [
                    &mut a2[0] as *mut u64 as *mut _, &mut a2[1] as *mut u64 as *mut _, &mut a2[2] as *mut u64 as *mut _,
                    &mut a2[3] as *mut u64 as *mut _, &mut d2[0] as *mut i32 as *mut _, &mut d2[1] as *mut i32 as *mut _,
                    &mut d2[2] as *mut i32 as *mut _, &mut b2[0] as *mut u64 as *mut _, &mut b2[1] as *mut u64 as *mut _,
                    &mut b2[2] as *mut u64 as *mut _, &mut b2[3] as *mut u64 as *mut _, &mut b2[4] as *mut u64 as *mut _,
                    &mut b2[5] as *mut u64 as *mut _, &mut b2[6] as *mut u64 as *mut _, &mut b2[7] as *mut u64 as *mut _,
                    &mut b2[8] as *mut u64 as *mut _, &mut b2[9] as *mut u64 as *mut _, &mut k2 as *mut i32 as *mut _,
                ];
                let r2 = unsafe { cudarc::driver::sys::cuLaunchKernel(fd, (emax * (h / 128)) as u32, 1, 1, 256, 1, 1,
                                    a1(m, mi), std::ptr::null_mut(), p2.as_mut_ptr(), std::ptr::null_mut()) };
                if r1 != cudarc::driver::sys::CUresult::CUDA_SUCCESS || r2 != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    bail!("EXL3-MOE-EPI FAIL: launch {gu} {r1:?} / {dn} {r2:?}");
                }
                dev.synchronize()?;
                let x_new = dev.dtoh_sync_copy(&xhd1)?;
                let g_new = dev.dtoh_sync_copy(&ygu1)?;
                let y_new = dev.dtoh_sync_copy(&yd1)?;
                let o_new = dev.dtoh_sync_copy(&out1)?;
                let cn = dev.dtoh_sync_copy(&d_cnt)?;
                let mx = x_old[..es * m * mi].iter().zip(&x_new[..es * m * mi]).filter(|(p, q)| p != q).count();
                let mg = g_old[..es * m * 2 * mi].iter().zip(&g_new[..es * m * 2 * mi]).filter(|(p, q)| p != q).count();
                let my = y_old[..es * m * h].iter().zip(&y_new[..es * m * h]).filter(|(p, q)| p != q).count();
                let mo = o_old[..m * h].iter().zip(&o_new[..m * h]).filter(|(p, q)| p != q).count();
                let armed = cn.iter().all(|v| *v == 0);
                if mg + mx + my + mo > 0 || !armed {
                    bail!("EXL3-MOE-EPI FAIL: top-k {topk} ({gu} + {dn}) m={m} esel={es} diet={diet} fold={fold} rep={rep}: bit-diff ygu {mg}, \
                           xhd {mx}, yd {my}, moe_out {mo}; counters {}", if armed { "re-armed" } else { "NOT re-armed" });
                }
            }
        }
        // ---- W4/MOE: the persistent kernels through the SERVED launchers (exl3_forward::mpk_*):
        // every schedule (ring depth, grid, item order, A-fold) and the mixed WP20/W4 pairings vs
        // the same old chain; NaN-poisoned outputs, 2 launches per schedule (re-armed counters).
        {
            use crate::exl3_forward as fw;
            let (sgu, sdn) = (a1(m, h), a1(m, mi));
            let cap_gu = (es * (2 * mi / 128)) as u32;
            let cap_dn = (es * (h / 128)) as u32;
            // (label, gate/up side, down side, interleaved order, A-fold); a side is Some((nst, grid))
            // with nst 0 = the served plan (mpk_plan), or None = the WP20 rung-2 kernel.
            type Side = Option<(usize, u32)>;
            let scheds: [(&str, Side, Side, bool, bool); 8] = [
                ("served", Some((0, 0)), Some((0, 0)), false, true),
                ("served-il", Some((0, 0)), Some((0, 0)), true, true),
                ("served-nofold", Some((0, 0)), Some((0, 0)), false, false),
                ("s4-G1", Some((4, 1)), Some((4, 1)), false, true),
                ("s8-G3-il", Some((8, 3)), Some((8, 3)), true, true),
                ("s4-Gitems-il-nofold", Some((4, cap_gu)), Some((4, cap_dn)), true, false),
                ("gu-pk+dn-epi", Some((0, 0)), None, false, true),
                ("gu-epi+dn-pk", None, Some((0, 0)), false, true),
            ];
            let plan = |gu: bool, side: (usize, u32), fold: bool| -> Result<fw::MpkPlan> {
                let (k, n) = if gu { (h, 2 * mi) } else { (mi, h) };
                if side.0 == 0 {
                    return fw::mpk_plan(dev, gu, topk, m, k, n, emax, fold)
                        .ok_or_else(|| anyhow::anyhow!("EXL3-MOE-EPI FAIL: W4/MOE has no served plan (gu {gu}) at m={m} top-k {topk}"));
                }
                let name = fw::mpk_entry(gu, topk, side.0)
                    .ok_or_else(|| anyhow::anyhow!("EXL3-MOE-EPI FAIL: no W4/MOE entry gu {gu} top-k {topk} nst {}", side.0))?;
                let smem = fw::mpk_smem_bytes(side.0, m, k, gu, fold) as u32;
                let (_, cps) = fw::mpk_fn(name, smem)
                    .ok_or_else(|| anyhow::anyhow!("EXL3-MOE-EPI FAIL: {name} not launchable at {smem} B smem"))?;
                Ok(fw::MpkPlan { name, grid: side.1.max(1), smem, cps, nst: side.0 })
            };
            let fg_epi = fw::xq_raw_fn(fw::wp20_gu_fn(true)).ok_or_else(|| anyhow::anyhow!("gu epi raw fn (stale PTX?)"))?;
            let fd_epi = fw::xq_raw_fn(dn_diet).ok_or_else(|| anyhow::anyhow!("{dn_diet} raw fn (stale PTX?)"))?;
            for (label, gs, ds, il, fold) in scheds.iter().copied() {
                let gplan = match gs { Some(sd) => Some(plan(true, sd, fold)?), None => None };
                let dplan = match ds { Some(sd) => Some(plan(false, sd, false)?), None => None };
                let fold = fold && gplan.is_some();
                let flags = if il { fw::MPK_FLAG_IL } else { 0 };
                let (mut ygu1, mut xhd1, mut yd1, mut out1) = (z(emax * m * 2 * mi)?, z(emax * m * mi)?, z(emax * m * h)?, z(m * h)?);
                let d_cnt = dev.htod_sync_copy(&vec![0u32; emax * (mi / 128) + h / 128 + 1])?;
                let mut d_sgv = dev.htod_sync_copy(&vec![0f32; m])?;
                let cnt = *d_cnt.device_ptr() as u64;
                let cnt_dn = cnt + (emax * (mi / 128) * 4) as u64;
                for rep in 0..2 {
                    for (buf, n) in [(&mut ygu1, emax * m * 2 * mi), (&mut xhd1, emax * m * mi), (&mut yd1, emax * m * h),
                                     (&mut out1, m * h)] {
                        dev.htod_sync_copy_into(&vec![0x7E00u16; n], buf)?;
                    }
                    dev.htod_sync_copy_into(&vec![f32::NAN; m], &mut d_sgv)?;
                    let sgvp = *d_sgv.device_ptr() as u64;
                    let (pygu, pxhd, pyd, pout) = (*ygu1.device_ptr() as u64, *xhd1.device_ptr() as u64,
                                                   *yd1.device_ptr() as u64, *out1.device_ptr() as u64);
                    let (px, pxh, pes, pix) = (*d_x.device_ptr() as u64, *xh_e.device_ptr() as u64,
                                               *d_es.device_ptr() as u64, *d_ix.device_ptr() as u64);
                    match gplan {
                        Some(gp) => fw::mpk_launch_gu(dev, std::ptr::null_mut(), &gp, *d_gu.device_ptr() as u64,
                                        *d_ogu.device_ptr() as u64, if fold { px } else { pxh },
                                        *d_suh_gu.device_ptr() as u64, pygu, m, h, 2 * mi, pes, pix,
                                        *d_svh_gu.device_ptr() as u64, *d_suh_d.device_ptr() as u64, pxhd, cnt, px,
                                        *d_sg.device_ptr() as u64, sgvp, flags | if fold { fw::MPK_FLAG_FOLD } else { 0 })?,
                        None => {
                            let mut a = [*d_gu.device_ptr() as u64, *d_ogu.device_ptr() as u64, pxh, pygu];
                            let mut d = [m as i32, h as i32, (2 * mi) as i32];
                            let mut b = [pes, pix, *d_svh_gu.device_ptr() as u64, *d_suh_d.device_ptr() as u64,
                                         pxhd, cnt, px, *d_sg.device_ptr() as u64, sgvp];
                            let mut pp: [*mut std::ffi::c_void; 16] = [
                                &mut a[0] as *mut u64 as *mut _, &mut a[1] as *mut u64 as *mut _, &mut a[2] as *mut u64 as *mut _,
                                &mut a[3] as *mut u64 as *mut _, &mut d[0] as *mut i32 as *mut _, &mut d[1] as *mut i32 as *mut _,
                                &mut d[2] as *mut i32 as *mut _, &mut b[0] as *mut u64 as *mut _, &mut b[1] as *mut u64 as *mut _,
                                &mut b[2] as *mut u64 as *mut _, &mut b[3] as *mut u64 as *mut _, &mut b[4] as *mut u64 as *mut _,
                                &mut b[5] as *mut u64 as *mut _, &mut b[6] as *mut u64 as *mut _, &mut b[7] as *mut u64 as *mut _,
                                &mut b[8] as *mut u64 as *mut _,
                            ];
                            let r = unsafe { cudarc::driver::sys::cuLaunchKernel(fg_epi, (emax * (2 * mi / 128)) as u32, 1, 1,
                                        256, 1, 1, sgu, std::ptr::null_mut(), pp.as_mut_ptr(), std::ptr::null_mut()) };
                            if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS { bail!("EXL3-MOE-EPI FAIL: launch gu epi {r:?}"); }
                        }
                    }
                    match dplan {
                        Some(dp) => fw::mpk_launch_dn(dev, std::ptr::null_mut(), &dp, *d_dn.device_ptr() as u64,
                                        *d_od.device_ptr() as u64, pxhd, pyd, m, mi, h, pes, pix,
                                        *d_svh_d.device_ptr() as u64, cnt_dn, *d_ysh.device_ptr() as u64,
                                        *d_ids.device_ptr() as u64, *d_wts.device_ptr() as u64, *d_sm.device_ptr() as u64,
                                        sgvp, pout, topk, flags)?,
                        None => {
                            let mut a2 = [*d_dn.device_ptr() as u64, *d_od.device_ptr() as u64, pxhd, pyd];
                            let mut d2 = [m as i32, mi as i32, h as i32];
                            let mut b2 = [pes, pix, *d_svh_d.device_ptr() as u64, cnt_dn, *d_ysh.device_ptr() as u64,
                                          *d_ids.device_ptr() as u64, *d_wts.device_ptr() as u64, *d_sm.device_ptr() as u64,
                                          sgvp, pout];
                            let mut k2 = topk as i32;
                            let mut p2: [*mut std::ffi::c_void; 18] = [
                                &mut a2[0] as *mut u64 as *mut _, &mut a2[1] as *mut u64 as *mut _, &mut a2[2] as *mut u64 as *mut _,
                                &mut a2[3] as *mut u64 as *mut _, &mut d2[0] as *mut i32 as *mut _, &mut d2[1] as *mut i32 as *mut _,
                                &mut d2[2] as *mut i32 as *mut _, &mut b2[0] as *mut u64 as *mut _, &mut b2[1] as *mut u64 as *mut _,
                                &mut b2[2] as *mut u64 as *mut _, &mut b2[3] as *mut u64 as *mut _, &mut b2[4] as *mut u64 as *mut _,
                                &mut b2[5] as *mut u64 as *mut _, &mut b2[6] as *mut u64 as *mut _, &mut b2[7] as *mut u64 as *mut _,
                                &mut b2[8] as *mut u64 as *mut _, &mut b2[9] as *mut u64 as *mut _, &mut k2 as *mut i32 as *mut _,
                            ];
                            let r = unsafe { cudarc::driver::sys::cuLaunchKernel(fd_epi, (emax * (h / 128)) as u32, 1, 1, 256, 1, 1,
                                        sdn, std::ptr::null_mut(), p2.as_mut_ptr(), std::ptr::null_mut()) };
                            if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS { bail!("EXL3-MOE-EPI FAIL: launch dn epi {r:?}"); }
                        }
                    }
                    dev.synchronize().with_context(|| format!("W4/MOE [{label}] m={m} top-k {topk} (kernel fault?)"))?;
                    let x_new = dev.dtoh_sync_copy(&xhd1)?;
                    let g_new = dev.dtoh_sync_copy(&ygu1)?;
                    let y_new = dev.dtoh_sync_copy(&yd1)?;
                    let o_new = dev.dtoh_sync_copy(&out1)?;
                    let cn = dev.dtoh_sync_copy(&d_cnt)?;
                    let mx = x_old[..es * m * mi].iter().zip(&x_new[..es * m * mi]).filter(|(p, q)| p != q).count();
                    let mg = g_old[..es * m * 2 * mi].iter().zip(&g_new[..es * m * 2 * mi]).filter(|(p, q)| p != q).count();
                    let my = y_old[..es * m * h].iter().zip(&y_new[..es * m * h]).filter(|(p, q)| p != q).count();
                    let mo = o_old[..m * h].iter().zip(&o_new[..m * h]).filter(|(p, q)| p != q).count();
                    let armed = cn.iter().all(|v| *v == 0);
                    if mg + mx + my + mo > 0 || !armed {
                        let gname = gplan.map_or("xq_moe_gu_epi".to_string(), |g| format!("{} G{}", g.name, g.grid));
                        let dname = dplan.map_or(dn_diet.to_string(), |d| format!("{} G{}", d.name, d.grid));
                        bail!("EXL3-MOE-EPI FAIL: W4/MOE [{label}] top-k {topk} ({gname} + {dname}, order {}, fold {fold}) m={m} \
                               esel={es} rep={rep}: bit-diff ygu {mg}, xhd {mx}, yd {my}, moe_out {mo}; counters {}",
                              if il { "il" } else { "contiguous" }, if armed { "re-armed" } else { "NOT re-armed" });
                    }
                }
                w4_done += 1;
                if m == 8 && topk == 10 && ne == NE_MAX {
                    let f = |pl: Option<fw::MpkPlan>| pl.map_or("epi".to_string(), |p| format!("s{}G{}", p.nst, p.grid));
                    w4_cfgs.push(format!("{label}: gu {} dn {}", f(gplan), f(dplan)));
                }
            }
        }
        done.push((topk, m, es));
    }
    }
    println!("EXL3-MOE-EPI: PASS (WP20 gate/up + down epilogues bitwise == the old MoE chain, ygu/xhd/yd/moe_out, \
              top-k 8 + 10 instances, diet + SHFL bodies, 2 launches each, counters re-armed; suh seq 0x{conv:02x} (pinned, verified); \
              (top-k, m, esel) {done:?})");
    println!("EXL3-MOE-EPI: A5-K7 GU_FOLD PASS ({} (xq_had_suh_multi folded into the gate/up prologue) + the diet down \
              epilogue bitwise == the old MoE chain (hence == the unfolded xq_moe_gu_epi rows above) on ygu/xhd/yd/moe_out, \
              NaN-poisoned outputs, 2 launches each, counters re-armed; {} (top-k, m) cases, m 1..9 x top-k/experts \
              8/12, 10/64, 10/256)", crate::exl3_forward::WP20_GU_FOLD_FN, done.len());
    println!("EXL3-MOE-EPI: W4/MOE PASS (persistent xq_moe_gu_pk / xq_moe_dn_pk_k<topk> through the served launchers, \
              bitwise == the old chain on ygu/xhd/yd/moe_out, NaN-poisoned outputs, 2 launches each, counters re-armed; \
              {w4_done} (top-k, m, schedule) cases over widths 1..9 x (top-k/experts) 8/12, 10/64, 10/256; schedules at \
              m=8 top-10/64: [{}])", w4_cfgs.join("; "));
    Ok(())
}

// ---------------------------------------------------------------------------
// S-A3-n F3: the grouped expert kernel's A-once body must be BITWISE equal to the
// barrier body (exl3_hmma_gemm) on the same trellis + activations, every row,
// every width that takes it. One expert (offs = [0], esel = [1]).
// ---------------------------------------------------------------------------
fn probe_grouped_a1(dev: &std::sync::Arc<CudaDevice>, d_tr: &CudaSlice<u16>, k: usize, n: usize,
                    bits: usize) -> Result<()> {
    if k % 8 != 0 || n % 128 != 0 { return Ok(()); }
    let f_old = dev.get_func(MODULE, "exl3_hmma_gemm").context("exl3_hmma_gemm")?;
    let f_grp = dev.get_func(MODULE, "xq_gemm_grouped_xh").context("xq_gemm_grouped_xh")?;
    let d_offs = dev.htod_sync_copy(&[0u64])?;
    let d_esel = dev.htod_sync_copy(&[1i32])?;
    let mut done = Vec::new();
    for m in [1usize, 2, 3, 4, 6, 8, 12, 16] {
        let smem = m * (k + 8) * 2;
        if smem > 48 * 1024 { continue; }
        let xh = synth_x(m, k, 0xBEEF ^ m as u32);
        let d_xh = dev.htod_sync_copy(&xh)?;
        let mut y_old = dev.htod_sync_copy(&vec![0u16; m * n])?;
        let mut y_grp = dev.htod_sync_copy(&vec![0u16; m * n])?;
        let cfg = |sm: u32| LaunchConfig { grid_dim: ((n / 128) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: sm };
        unsafe {
            f_old.clone().launch(cfg(0), (d_tr, &d_xh, &mut y_old, m as i32, k as i32, n as i32, bits as i32))?;
            f_grp.clone().launch(cfg(smem as u32), (d_tr, &d_offs, &d_xh, &mut y_grp, m as i32, k as i32, n as i32, bits as i32, &d_esel))?;
        }
        dev.synchronize()?;
        let a = dev.dtoh_sync_copy(&y_old)?;
        let b = dev.dtoh_sync_copy(&y_grp)?;
        if a.iter().filter(|v| **v != 0).count() < m * n / 2 { bail!("EXL3-GROUPED-A1 FAIL: implausible reference (m={m})"); }
        let mism = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
        if mism > 0 { bail!("EXL3-GROUPED-A1 FAIL: m={m} K={k} N={n} b={bits}: {mism} mismatches vs the barrier body"); }
        if bits == 3 {
            // S-A3-o O3: the dedicated 3-bit entry must equal the barrier body too.
            let f_b3 = dev.get_func(MODULE, "xq_gemm_grouped_a1b3").context("xq_gemm_grouped_a1b3")?;
            let mut y_b3 = dev.htod_sync_copy(&vec![0u16; m * n])?;
            unsafe { f_b3.clone().launch(cfg(smem as u32), (d_tr, &d_offs, &d_xh, &mut y_b3, m as i32, k as i32, n as i32, bits as i32, &d_esel))?; }
            dev.synchronize()?;
            let c = dev.dtoh_sync_copy(&y_b3)?;
            let mism = a.iter().zip(c.iter()).filter(|(x, y)| x != y).count();
            if mism > 0 { bail!("EXL3-GROUPED-A1B3 FAIL: m={m} K={k} N={n}: {mism} mismatches vs the barrier body"); }
            // A5 WP20 rung 1: the word-diet twin (2 ring words per lane per k16 step, no SHFL).
            let f_wd = dev.get_func(MODULE, "xq_gemm_grouped_a1b3_wd").context("xq_gemm_grouped_a1b3_wd")?;
            let mut y_wd = dev.htod_sync_copy(&vec![0u16; m * n])?;
            unsafe { f_wd.clone().launch(cfg(smem as u32), (d_tr, &d_offs, &d_xh, &mut y_wd, m as i32, k as i32, n as i32, bits as i32, &d_esel))?; }
            dev.synchronize()?;
            let c = dev.dtoh_sync_copy(&y_wd)?;
            let mism = a.iter().zip(c.iter()).filter(|(x, y)| x != y).count();
            if mism > 0 { bail!("EXL3-GROUPED-A1B3-WD FAIL: m={m} K={k} N={n}: {mism} mismatches vs the barrier body"); }
        }
        done.push(m);
    }
    println!("    grouped A-once body{}: PASS (bitwise == barrier body, widths {done:?})",
             if bits == 3 { " + a1b3 entry + a1b3_wd (WP20 diet)" } else { "" });
    Ok(())
}

// ---------------------------------------------------------------------------
// W3/LMH gate: the persistent lm_head GEMM (exl3_lmh_*, launched through the SERVED launcher
// crate::exl3_forward::lmh_gemm_launch) must be BITWISE equal to exl3_hmma_gemm (y_raw) and to
// exl3_hmma_gemm + exl3_had_svh (fused epilogue -> logits) at every m in `ms`, for every ring
// depth in `nsts`. Output buffers are poisoned (0xFFFF) so an unwritten element is a mismatch.
// ---------------------------------------------------------------------------
#[allow(clippy::too_many_arguments)]
fn probe_lmh(dev: &std::sync::Arc<CudaDevice>, d_tr: &CudaSlice<u16>, d_svh: &CudaSlice<u16>,
             k: usize, n: usize, bits: usize, ms: &[usize], nsts: &[usize], tag: &str) -> Result<()> {
    // The launcher's static eligibility (lmh_gemm_launch): shapes outside it keep the old kernel
    // by design — skip them here, so a refusal below is always a real (resource/PTX) failure.
    if k % (16 * crate::exl3_forward::LMH_KSTEP) != 0 || k % 128 != 0 || n % 128 != 0 || !(3..=5).contains(&bits) {
        println!("    W3/LMH [{tag}]: skipped (K={k} N={n} b={bits} outside the persistent kernel's shape class)");
        return Ok(());
    }
    let f_old = dev.get_func(MODULE, "exl3_hmma_gemm").context("exl3_hmma_gemm")?;
    let f_svh = dev.get_func(MODULE, "exl3_had_svh").context("exl3_had_svh")?;
    let stream = fork_blocking_stream(dev);
    let mut checked = 0usize;
    for &m in ms {
        let xh = synth_x(m, k, 0x1A4D ^ (m as u32).wrapping_mul(2654435761));
        let d_xh = dev.htod_sync_copy(&xh)?;
        let mut yr_old = dev.htod_sync_copy(&vec![0u16; m * n])?;
        let mut y_old = dev.htod_sync_copy(&vec![0u16; m * n])?;
        unsafe {
            f_old.clone().launch_on_stream(&stream,
                LaunchConfig { grid_dim: ((n / 128) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                (d_tr, &d_xh, &mut yr_old, m as i32, k as i32, n as i32, bits as i32))?;
            f_svh.clone().launch_on_stream(&stream,
                LaunchConfig { grid_dim: ((m * n / 128) as u32, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 },
                (&yr_old, d_svh, &mut y_old, n as i32))?;
        }
        dev.synchronize()?;
        let r_old = dev.dtoh_sync_copy(&yr_old)?;
        let v_old = dev.dtoh_sync_copy(&y_old)?;
        if r_old.iter().filter(|v| **v != 0).count() < m * n / 2 {
            bail!("W3/LMH FAIL [{tag}]: implausible reference (m={m}: y_raw mostly zero)");
        }
        for &nst in nsts {
            for fuse in [false, true] {
                let out = dev.htod_sync_copy(&vec![0xFFFFu16; m * n])?;
                let ok = crate::exl3_forward::lmh_gemm_launch(dev, stream.stream, *d_tr.device_ptr(),
                    *d_xh.device_ptr(), *out.device_ptr(), *d_svh.device_ptr(), m, k, n, bits, nst, fuse)?;
                if !ok {
                    bail!("W3/LMH FAIL [{tag}]: launcher refused m={m} K={k} N={n} b={bits} nst={nst}");
                }
                dev.synchronize()?;
                let got = dev.dtoh_sync_copy(&out)?;
                let refv = if fuse { &v_old } else { &r_old };
                let mut mism = 0usize;
                let mut first = None;
                for i in 0..m * n {
                    if got[i] != refv[i] {
                        mism += 1;
                        if first.is_none() { first = Some((i / n, i % n, got[i], refv[i])); }
                    }
                }
                if mism > 0 {
                    bail!("W3/LMH FAIL [{tag}]: m={m} K={k} N={n} b={bits} nst={nst} fused={fuse}: \
                           {mism} of {} mismatches, first (row,col,new,old) {first:?}", m * n);
                }
                checked += 1;
            }
        }
    }
    println!("    W3/LMH [{tag}]: PASS (bitwise == exl3_hmma_gemm / +exl3_had_svh; K={k} N={n} b={bits}, \
              m {ms:?}, nst {nsts:?}, both epilogues, {checked} launches)");
    Ok(())
}

// ---------------------------------------------------------------------------
// w3-PSK (T4e): the persistent split-K chain GEMM must be BITWISE equal to the one-CTA-per-item
// grid it replaces — every fp32 partial ws[slab][row][col] — for EVERY chain shape this pack
// splits (non-expert, non-lm_head modules with chain_ks > 1), every width M = 1..16, a spread of
// grid sizes G (1, odd, 48-multiples, the built-in default, items-1, items, > items) and trellis
// prefetch depths 0/1/2: exl3_hmma_gemm_ks_p vs exl3_hmma_gemm_ks, and exl3_hmma_gemm_ks_p_x2 vs
// exl3_hmma_gemm_ks_x2 (two distinct same-shape modules when the pack has them). PSK workspaces
// are NaN-poisoned before every launch (an unwritten partial cannot pass), the reference must
// be plausible (finite, mostly nonzero), and y_raw through the UNCHANGED exl3_ks_combine keeps
// row 0 bitwise across widths (batch invariance).
// ---------------------------------------------------------------------------
fn probe_ks_p(dev: &std::sync::Arc<CudaDevice>, pack: &Exl3Pack) -> Result<()> {
    use cudarc::driver::sys;
    use std::collections::BTreeMap;
    let f_old = dev.get_func(MODULE, "exl3_hmma_gemm_ks").context("exl3_hmma_gemm_ks")?;
    let f_old2 = dev.get_func(MODULE, "exl3_hmma_gemm_ks_x2").context("exl3_hmma_gemm_ks_x2")?;
    let f_new = dev.get_func(MODULE, "exl3_hmma_gemm_ks_p").context("exl3_hmma_gemm_ks_p (stale PTX?)")?;
    let f_new2 = dev.get_func(MODULE, "exl3_hmma_gemm_ks_p_x2").context("exl3_hmma_gemm_ks_p_x2 (stale PTX?)")?;
    let f_comb = dev.get_func(MODULE, "exl3_ks_combine").context("exl3_ks_combine")?;
    let sms = dev.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?.max(1) as u32;
    let slots1 = f_new.occupancy_max_active_blocks_per_multiprocessor(256, 0, None)? * sms;
    let slots2 = f_new2.occupancy_max_active_blocks_per_multiprocessor(256, 0, None)? * sms;
    if slots1 == 0 || slots2 == 0 {
        bail!("EXL3-KS-P FAIL: zero occupancy for the PSK kernels (single {slots1}, pair {slots2})");
    }
    // distinct (K, N, bits) chain shapes -> up to two modules (pair members)
    let mut shapes: BTreeMap<(usize, usize, usize), Vec<&ModuleMeta>> = BTreeMap::new();
    for md in &pack.modules {
        if matches!(md.class, TensorClass::LmHead | TensorClass::Ngram) || md.name.contains(".mlp.experts.") {
            continue;
        }
        if md.k % 128 != 0 || md.n % 128 != 0 { continue; }
        if crate::exl3_forward::chain_ks(md.k as i32, md.n as i32) <= 1 { continue; }
        let e = shapes.entry((md.k, md.n, md.bits)).or_default();
        if e.len() < 2 { e.push(md); }
    }
    if shapes.is_empty() {
        bail!("EXL3-KS-P FAIL: no split-K chain shapes in the pack (harness failure, nothing was tested)");
    }
    println!("    PSK persistent split-K: {} chain shape(s); slots single {slots1} / pair {slots2}", shapes.len());
    let word = |bits: usize, ks: u32, pf: i32| -> i32 { (bits as i32) | ((ks as i32) << 8) | (pf << 24) };
    let poison = f32::from_bits(0x7FC0_DEAD);
    let rd = |md: &ModuleMeta| -> Result<Vec<u16>> {
        Ok(md.trellis.read_bytes(&pack.dir)?.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
    };
    let glist = |items: u32, slots: u32| -> Vec<u32> {
        let mut v = vec![1, 5, 47, 48, 96, items.min(slots), items.saturating_sub(1).max(1), items, items + 7];
        v.sort_unstable();
        v.dedup();
        v
    };
    let mut cases = 0usize;
    for (&(k, n, bits), mods) in &shapes {
        let ks = crate::exl3_forward::chain_ks(k as i32, n as i32);
        let tiles = (n / 128) as u32;
        let items = tiles * ks;
        let d_tr0 = dev.htod_sync_copy(&rd(mods[0])?)?;
        let d_tr1 = dev.htod_sync_copy(&rd(mods[mods.len() - 1])?)?;
        let g1 = glist(items, slots1);
        let g2 = glist(2 * items, slots2);
        // (G, pf) combos: every G at the default depth, plus the default G at depths 0 and 2
        let combos = |gs: &[u32], gdef: u32| -> Vec<(u32, i32)> {
            let mut v: Vec<(u32, i32)> = gs.iter().map(|&g| (g, 1)).collect();
            v.push((gdef, 0));
            v.push((gdef, 2));
            v
        };
        let c1 = combos(&g1, items.min(slots1));
        let c2 = combos(&g2, (2 * items).min(slots2));
        let mut row0: Option<Vec<u16>> = None;
        for m in 1..=16usize {
            let need = ks as usize * m * n;
            let pz = vec![poison; need];
            let d_x0 = dev.htod_sync_copy(&synth_x(m, k, 0x5EED))?;           // row 0 shared by every m
            let d_x1 = dev.htod_sync_copy(&synth_x(m, k, 0xF00D ^ m as u32))?;
            let mut r0 = dev.htod_sync_copy(&pz)?;
            let mut p0 = dev.htod_sync_copy(&pz)?;
            let mut p1 = dev.htod_sync_copy(&pz)?;
            let mut w0 = dev.htod_sync_copy(&pz)?;
            let mut w1 = dev.htod_sync_copy(&pz)?;
            let old_cfg = |z: u32| LaunchConfig { grid_dim: (tiles, ks, z), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            let new_cfg = |g: u32| LaunchConfig { grid_dim: (g, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            unsafe {
                f_old.clone().launch(old_cfg(1), (&d_tr0, &d_x0, &mut r0, m as i32, k as i32, n as i32, bits as i32))?;
                f_old2.clone().launch(old_cfg(2), (&d_tr0, &d_tr1, &d_x0, &d_x1, &mut p0, &mut p1,
                                                   m as i32, k as i32, n as i32, bits as i32))?;
            }
            dev.synchronize()?;
            let ref0 = dev.dtoh_sync_copy(&r0)?;
            let (refp0, refp1) = (dev.dtoh_sync_copy(&p0)?, dev.dtoh_sync_copy(&p1)?);
            // the success signal: a finite, mostly-nonzero reference (and the old pair's member 0
            // IS the old single on the same inputs)
            for (tag, v) in [("single", &ref0), ("pair[0]", &refp0), ("pair[1]", &refp1)] {
                let live = v.iter().filter(|x| x.is_finite() && **x != 0.0).count();
                if live < need / 2 {
                    bail!("EXL3-KS-P FAIL: implausible {tag} reference K={k} N={n} b{bits} m={m} ({live}/{need} finite nonzero)");
                }
            }
            if ref0.iter().zip(refp0.iter()).any(|(a, b)| a.to_bits() != b.to_bits()) {
                bail!("EXL3-KS-P FAIL: old pair member 0 != old single (K={k} N={n} m={m}) — reference inconsistent");
            }
            for &(g, pf) in &c1 {
                dev.htod_sync_copy_into(&pz, &mut w0)?;
                unsafe {
                    f_new.clone().launch(new_cfg(g), (&d_tr0, &d_x0, &mut w0, m as i32, k as i32, n as i32,
                                                      word(bits, ks, pf), &d_tr0))?;
                }
                dev.synchronize()?;
                let got = dev.dtoh_sync_copy(&w0)?;
                let mism = got.iter().zip(ref0.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                if mism > 0 {
                    let i = got.iter().zip(ref0.iter()).position(|(a, b)| a.to_bits() != b.to_bits()).unwrap_or(0);
                    bail!("EXL3-KS-P FAIL: single K={k} N={n} b{bits} ks={ks} m={m} G={g} pf={pf}: {mism} of {need} partials differ \
                           (first [{i}] psk {:e} vs old {:e})", got[i], ref0[i]);
                }
                cases += 1;
            }
            for &(g, pf) in &c2 {
                dev.htod_sync_copy_into(&pz, &mut w0)?;
                dev.htod_sync_copy_into(&pz, &mut w1)?;
                unsafe {
                    f_new2.clone().launch(new_cfg(g), (&d_tr0, &d_tr1, &d_x0, &d_x1, &mut w0, &mut w1,
                                                       m as i32, k as i32, n as i32, word(bits, ks, pf), &d_tr0, &d_tr1))?;
                }
                dev.synchronize()?;
                let (a, b) = (dev.dtoh_sync_copy(&w0)?, dev.dtoh_sync_copy(&w1)?);
                let mism = a.iter().zip(refp0.iter()).chain(b.iter().zip(refp1.iter()))
                    .filter(|(x, y)| x.to_bits() != y.to_bits()).count();
                if mism > 0 {
                    bail!("EXL3-KS-P FAIL: pair K={k} N={n} b{bits} ks={ks} m={m} G={g} pf={pf}: {mism} of {} partials differ",
                          2 * need);
                }
                cases += 1;
            }
            // y_raw through the unchanged combine on the PSK partials (w0 = single, default G):
            // row 0 bitwise across widths.
            dev.htod_sync_copy_into(&pz, &mut w0)?;
            let mut yr = dev.htod_sync_copy(&vec![0u16; m * n])?;
            unsafe {
                f_new.clone().launch(new_cfg(items.min(slots1)), (&d_tr0, &d_x0, &mut w0, m as i32, k as i32, n as i32,
                                                                  word(bits, ks, 1), &d_tr0))?;
                f_comb.clone().launch(LaunchConfig { grid_dim: (((m * n + 255) / 256) as u32, 1, 1), block_dim: (256, 1, 1),
                                                     shared_mem_bytes: 0 },
                                      (&w0, &mut yr, m as i32, n as i32, ks as i32))?;
            }
            dev.synchronize()?;
            let y = dev.dtoh_sync_copy(&yr)?;
            match &row0 {
                None => row0 = Some(y[..n].to_vec()),
                Some(r) => {
                    let mism = r.iter().zip(y[..n].iter()).filter(|(a, b)| a != b).count();
                    if mism > 0 { bail!("EXL3-KS-P BINV FAIL: K={k} N={n} b{bits}: y_raw row 0 at m={m} differs from m=1 in {mism} of {n}"); }
                }
            }
        }
        println!("      K={k:<5} N={n:<6} b{bits} ks={ks:<3} items {items:>4}: bitwise == old grid, M 1..16, \
                  single G {g1:?} + pair G {g2:?}, pf 0/1/2; y_raw row-0 binv PASS ({})",
                 mods.iter().map(|md| md.name.as_str()).collect::<Vec<_>>().join(" | "));
    }
    println!("    EXL3-KS-P: PASS ({cases} (shape, M, G, pf) launches bitwise == exl3_hmma_gemm_ks(_x2))");
    Ok(())
}

// ---------------------------------------------------------------------------
// WP21 (split-K fixup epilogue, the served default): exl3_hmma_gemm_ks_fx and exl3_hmma_gemm_ks_fx_x2
// (the x2 through the SERVED raw launcher exl3_forward::wp21_fx_x2_raw) must be BITWISE equal to the
// 3-launch tail they replace — exl3_had_suh -> exl3_hmma_gemm_ks -> exl3_ks_combine -> exl3_had_svh
// (dense_ref) — on every y element AND every fp32 partial ws[slab][row][col] (the fused body stores
// the same partials), for every split chain shape of the pack (non-expert, non-head, chain_ks > 1),
// M = 1..16, single and same-shape pair (two distinct modules when the pack has them; each member
// with its own counter slice). Every fused launch runs TWICE on the same counters with partials
// NaN-poisoned and y 0xFFFF-poisoned before each run: the second run only writes y if the first left
// every tile counter re-armed at 0 (graph-replay safety); counters are read back at 0 after each
// run. y row 0 is bitwise across M (batch invariance). REPRIME's wp22r1 form (M = 17 / 28 / 32 as
// <= 16-row slabs sharing ws and the counters, each fused launch writing ITS rows of y) must equal
// the old slab loop (ks GEMM + combine per slab, one svh over all rows), and its row 0 must equal the
// M <= 16 chains' row 0. The reference must be plausible (finite, mostly nonzero).
// ---------------------------------------------------------------------------
fn probe_wp21(dev: &std::sync::Arc<CudaDevice>, pack: &Exl3Pack) -> Result<()> {
    use std::collections::BTreeMap;
    let st = fork_blocking_stream(dev);
    let gf = |n: &str| dev.get_func(MODULE, n).with_context(|| format!("{n} (stale PTX?)"));
    let f_fx = gf("exl3_hmma_gemm_ks_fx")?;
    let rd16 = |b: Vec<u8>| -> Vec<u16> { b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect() };
    let mut shapes: BTreeMap<(usize, usize, usize), Vec<&ModuleMeta>> = BTreeMap::new();
    for md in &pack.modules {
        if matches!(md.class, TensorClass::LmHead | TensorClass::Ngram) || md.name.contains(".mlp.experts.") { continue; }
        if md.k % 128 != 0 || md.n % 128 != 0 || !(3..=5).contains(&md.bits) { continue; }
        let ks = crate::exl3_forward::chain_ks(md.k as i32, md.n as i32) as usize;
        if ks <= 1 || (md.k / 16) % ks != 0 { continue; }
        let e = shapes.entry((md.k, md.n, md.bits)).or_default();
        if e.len() < 2 { e.push(md); }
    }
    if shapes.is_empty() {
        bail!("EXL3-WP21 FAIL: no split-K chain shapes in the pack (harness failure, nothing was tested)");
    }
    let load = |md: &ModuleMeta| -> Result<DenseMod> {
        Ok(DenseMod {
            name: md.name.clone(),
            tr: dev.htod_sync_copy(&rd16(md.trellis.read_bytes(&pack.dir)?))?,
            suh: dev.htod_sync_copy(&le_f16_bits(&md.suh.read_bytes(&pack.dir)?))?,
            svh: dev.htod_sync_copy(&le_f16_bits(&md.svh.read_bytes(&pack.dir)?))?,
            k: md.k, n: md.n, bits: md.bits,
            ks: crate::exl3_forward::chain_ks(md.k as i32, md.n as i32) as usize,
        })
    };
    let cfg = |g: (u32, u32, u32), b: u32| LaunchConfig { grid_dim: g, block_dim: (b, 1, 1), shared_mem_bytes: 0 };
    let poison = f32::from_bits(0x7FC0_DEAD);
    let nz = |c: &CudaSlice<u32>| -> Result<usize> { Ok(dev.dtoh_sync_copy(c)?.iter().filter(|&&v| v != 0).count()) };
    let pmism = |a: &[f32], b: &[f32]| a.iter().zip(b.iter()).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    let ymism = |a: &[u16], b: &[u16]| a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
    println!("    WP21 split-K fixup epilogue: {} chain shape(s), fused vs 3-launch tail (single + pair + wp22r1 slabs)",
             shapes.len());
    let (mut launches, mut slab_cells) = (0usize, 0usize);
    for (&(k, n, bits), mods) in &shapes {
        let a = load(mods[0])?;
        let b = load(mods[mods.len() - 1])?;
        let ks = a.ks;
        let tiles = n / 128;
        let cnt0 = dev.htod_sync_copy(&vec![0u32; tiles.max(1)])?;
        let cnt1 = dev.htod_sync_copy(&vec![0u32; tiles.max(1)])?;
        let mut row0: Option<Vec<u16>> = None;
        for m in 1..=16usize {
            let x = dev.htod_sync_copy(&synth_x(m, k, 0x5EED))?; // row 0 shared by every m
            let (rp0, ry0) = dense_ref(dev, &st, &a, &x, None, m)?;
            let (rp1, ry1) = dense_ref(dev, &st, &b, &x, None, m)?;
            for (tag, p, y) in [("member 0", &rp0, &ry0), ("member 1", &rp1, &ry1)] {
                let pl = p.iter().filter(|v| v.is_finite() && **v != 0.0).count();
                let yl = y.iter().filter(|&&v| v != 0 && f16::from_bits(v).to_f32().is_finite()).count();
                if pl < p.len() / 2 || yl < y.len() / 2 {
                    bail!("EXL3-WP21 FAIL: implausible {tag} reference K={k} N={n} b{bits} m={m} ({pl}/{} partials, {yl}/{} y live)",
                          p.len(), y.len());
                }
            }
            let mut xh0 = dev.htod_sync_copy(&vec![0u16; m * k])?;
            let mut xh1 = dev.htod_sync_copy(&vec![0u16; m * k])?;
            unsafe {
                gf("exl3_had_suh_x2")?.launch_on_stream(&st, cfg(((m * k / 128) as u32, 1, 2), 32),
                                                        (&x, &a.suh, &b.suh, &mut xh0, &mut xh1, k as i32))?;
            }
            let mut ws0 = dev.htod_sync_copy(&vec![poison; ks * m * n])?;
            let mut ws1 = dev.htod_sync_copy(&vec![poison; ks * m * n])?;
            let mut y0 = dev.htod_sync_copy(&vec![0xFFFFu16; m * n])?;
            let mut y1 = dev.htod_sync_copy(&vec![0xFFFFu16; m * n])?;
            for rep in 0..2 {
                // single (member 0)
                dev.htod_sync_copy_into(&vec![poison; ks * m * n], &mut ws0)?;
                dev.htod_sync_copy_into(&vec![0xFFFFu16; m * n], &mut y0)?;
                unsafe {
                    f_fx.clone().launch_on_stream(&st, cfg((tiles as u32, ks as u32, 1), 256),
                        (&a.tr, &xh0, &mut ws0, &a.svh, &mut y0, &cnt0, m as i32, k as i32, n as i32, bits as i32))?;
                }
                dev.synchronize()?;
                launches += 1;
                let (p, y, c) = (dev.dtoh_sync_copy(&ws0)?, dev.dtoh_sync_copy(&y0)?, nz(&cnt0)?);
                let (pm, ym) = (pmism(&p, &rp0), ymism(&y, &ry0));
                if pm + ym + c > 0 {
                    bail!("EXL3-WP21 FAIL: single K={k} N={n} b{bits} ks={ks} m={m} run {rep}: partials mism {pm}, y mism {ym} \
                           of {}, counters not re-armed {c} ({})", m * n, a.name);
                }
                if rep == 0 {
                    match &row0 {
                        None => row0 = Some(y[..n].to_vec()),
                        Some(r) => {
                            let d = ymism(r, &y[..n]);
                            if d > 0 { bail!("EXL3-WP21 BINV FAIL: K={k} N={n} b{bits}: y row 0 at m={m} differs from m=1 in {d} of {n}"); }
                        }
                    }
                }
                // pair (members 0 and 1, same x, own counter slices) — the served raw launcher
                dev.htod_sync_copy_into(&vec![poison; ks * m * n], &mut ws0)?;
                dev.htod_sync_copy_into(&vec![poison; ks * m * n], &mut ws1)?;
                dev.htod_sync_copy_into(&vec![0xFFFFu16; m * n], &mut y0)?;
                dev.htod_sync_copy_into(&vec![0xFFFFu16; m * n], &mut y1)?;
                let pp: [u64; 12] = [
                    *a.tr.device_ptr(), *b.tr.device_ptr(), *xh0.device_ptr(), *xh1.device_ptr(),
                    *ws0.device_ptr(), *ws1.device_ptr(), *a.svh.device_ptr(), *b.svh.device_ptr(),
                    *y0.device_ptr(), *y1.device_ptr(), *cnt0.device_ptr(), *cnt1.device_ptr(),
                ];
                crate::exl3_forward::wp21_fx_x2_raw(st.stream, pp, m, k as i32, n as i32, bits as i32, ks as u32)?;
                dev.synchronize()?;
                launches += 1;
                let (p0, p1) = (dev.dtoh_sync_copy(&ws0)?, dev.dtoh_sync_copy(&ws1)?);
                let (g0, g1) = (dev.dtoh_sync_copy(&y0)?, dev.dtoh_sync_copy(&y1)?);
                let c = nz(&cnt0)? + nz(&cnt1)?;
                let (pm, ym) = (pmism(&p0, &rp0) + pmism(&p1, &rp1), ymism(&g0, &ry0) + ymism(&g1, &ry1));
                if pm + ym + c > 0 {
                    bail!("EXL3-WP21 FAIL: pair K={k} N={n} b{bits} ks={ks} m={m} run {rep}: partials mism {pm}, y mism {ym} \
                           of {}, counters not re-armed {c} ({} | {})", 2 * m * n, a.name, b.name);
                }
            }
        }
        // REPRIME wp22r1 row slabs (m > 16): fused per slab vs the old slab loop, sharing ws + counters.
        let r0_ref = row0.clone().unwrap_or_default();
        for m in [17usize, 28, 32] {
            let x = dev.htod_sync_copy(&synth_x(m, k, 0x5EED))?;
            let mut xh = dev.htod_sync_copy(&vec![0u16; m * k])?;
            let mut ws = dev.htod_sync_copy(&vec![poison; ks * 16 * n])?;
            let yr = dev.htod_sync_copy(&vec![0u16; m * n])?; // written through yr_p (slab row offsets)
            let mut yref = dev.htod_sync_copy(&vec![0xFFFFu16; m * n])?;
            let mut yfx = dev.htod_sync_copy(&vec![0xFFFFu16; m * n])?;
            let (xh_p, yr_p, yf_p) = (*xh.device_ptr(), *yr.device_ptr(), *yfx.device_ptr());
            let slabs: Vec<(usize, usize)> = (0..m).step_by(16).map(|r0| (r0, (m - r0).min(16))).collect();
            unsafe {
                gf("exl3_had_suh")?.launch_on_stream(&st, cfg(((m * k / 128) as u32, 1, 1), 32), (&x, &a.suh, &mut xh, k as i32))?;
                for &(r0, ms) in &slabs {
                    gf("exl3_hmma_gemm_ks")?.launch_on_stream(&st, cfg((tiles as u32, ks as u32, 1), 256),
                        (&a.tr, xh_p + (r0 * k * 2) as u64, &mut ws, ms as i32, k as i32, n as i32, bits as i32))?;
                    gf("exl3_ks_combine")?.launch_on_stream(&st, cfg((((ms * n + 255) / 256) as u32, 1, 1), 256),
                        (&ws, yr_p + (r0 * n * 2) as u64, ms as i32, n as i32, ks as i32))?;
                }
                gf("exl3_had_svh")?.launch_on_stream(&st, cfg(((m * n / 128) as u32, 1, 1), 32), (&yr, &a.svh, &mut yref, n as i32))?;
            }
            dev.synchronize()?;
            let want = dev.dtoh_sync_copy(&yref)?;
            let live = want.iter().filter(|&&v| v != 0 && v != 0xFFFF).count();
            if live < want.len() / 2 {
                bail!("EXL3-WP21 FAIL: implausible slab reference K={k} N={n} b{bits} m={m} ({live}/{} live)", want.len());
            }
            for rep in 0..2 {
                dev.htod_sync_copy_into(&vec![poison; ks * 16 * n], &mut ws)?;
                dev.htod_sync_copy_into(&vec![0xFFFFu16; m * n], &mut yfx)?;
                for &(r0, ms) in &slabs {
                    unsafe {
                        f_fx.clone().launch_on_stream(&st, cfg((tiles as u32, ks as u32, 1), 256),
                            (&a.tr, xh_p + (r0 * k * 2) as u64, &mut ws, &a.svh, yf_p + (r0 * n * 2) as u64, &cnt0,
                             ms as i32, k as i32, n as i32, bits as i32))?;
                    }
                    launches += 1;
                }
                dev.synchronize()?;
                let got = dev.dtoh_sync_copy(&yfx)?;
                let (ym, c) = (ymism(&got, &want), nz(&cnt0)?);
                let r0d = if r0_ref.len() == n { ymism(&got[..n], &r0_ref) } else { n };
                if ym + c + r0d > 0 {
                    bail!("EXL3-WP21 FAIL: wp22r1 slabs K={k} N={n} b{bits} ks={ks} m={m} ({} slabs) run {rep}: y mism {ym} of {}, \
                           counters not re-armed {c}, row 0 vs the m<=16 chains {r0d} ({})", slabs.len(), m * n, a.name);
                }
            }
            slab_cells += 1;
        }
        println!("      K={k:<5} N={n:<6} b{bits} ks={ks:<3} tiles {tiles:>3}: fused == 3-launch (partials + y), M 1..16 single + pair, \
                  2 runs/cell (counters re-armed), y row-0 binv; wp22r1 slabs M 17/28/32 PASS ({})",
                 mods.iter().map(|md| md.name.as_str()).collect::<Vec<_>>().join(" | "));
    }
    println!("    EXL3-WP21: PASS ({launches} fused launches bitwise == exl3_hmma_gemm_ks(_x2) + exl3_ks_combine(_x2) + \
              exl3_had_svh(_x2); {} shape(s), {slab_cells} wp22r1 slab cell(s))", shapes.len());
    Ok(())
}

// ---------------------------------------------------------------------------
// W4/DENSE gate: exl3_dks_* — launched through the SERVED launcher (exl3_forward::dense::
// probe_launch -> dks_launch) — must be BITWISE equal to the old chain it replaces,
//   [xq_silu_mul ->] exl3_had_suh -> exl3_hmma_gemm_ks -> exl3_ks_combine -> exl3_had_svh,
// member by member: every fp32 partial ws[slab][row][col] and every y element, for every split
// chain shape of the pack (singles), same-shape pairs, one same-K multi-member group (big + small
// members in one item list), the silu prologue (shared-expert down), M = 1..16, grid sizes G (the
// served rule, 1, odd, 40, 48, more CTAs than items), ring depths, and the escape flag sets
// (partials-only tail, staged xh). Partials are NaN-poisoned and y 0xFFFF-poisoned before every
// launch (an unwritten element cannot pass); every fixup launch must leave its tile counters at 0
// (self re-arming); the reference must be plausible; y row 0 is bitwise across M (batch invariance).
// A (job, M) cell the served grid rule cannot place (dks_plan: no G fits the A-entry / smem budget,
// e.g. the K2560 x N10240 / N12288 same-shape pairs at M 9..16) is SKIPPED with a printed line — the
// served path falls back to the old chain there (dense_chains -> Ok(false), nothing launched). For
// pairs that fallback is exl3_chain_pair's x2 chain, which is run here and bit-checked against the
// old chains; for singles / multi groups / the silu down it is the reference chain itself. A skip
// never ends the probe (the later binv sections must still run); zero DENSE launches is a FAIL.
// ---------------------------------------------------------------------------
struct DenseMod {
    name: String,
    tr: CudaSlice<u16>,
    suh: CudaSlice<u16>,
    svh: CudaSlice<u16>,
    k: usize,
    n: usize,
    bits: usize,
    ks: usize,
}

/// The old chain on `x` (m rows) for one member -> (fp32 partials, y).
fn dense_ref(dev: &std::sync::Arc<CudaDevice>, st: &CudaStream, md: &DenseMod, x: &CudaSlice<u16>,
             x2: Option<&CudaSlice<u16>>, m: usize) -> Result<(Vec<f32>, Vec<u16>)> {
    let gf = |n: &str| dev.get_func(MODULE, n).with_context(|| n.to_string());
    let (k, n, ks) = (md.k, md.n, md.ks);
    let mut xh = dev.htod_sync_copy(&vec![0u16; m * k])?;
    let mut ws = dev.htod_sync_copy(&vec![0f32; ks * m * n])?;
    let mut yr = dev.htod_sync_copy(&vec![0u16; m * n])?;
    let mut y = dev.htod_sync_copy(&vec![0u16; m * n])?;
    let cfg = |g: (u32, u32, u32), b: u32| LaunchConfig { grid_dim: g, block_dim: (b, 1, 1), shared_mem_bytes: 0 };
    unsafe {
        let xin = match x2 {
            Some(u) => {
                let mut s = dev.htod_sync_copy(&vec![0u16; m * k])?;
                gf("xq_silu_mul")?.launch_on_stream(st, cfg((((m * k) as u32) / 256 + 1, 1, 1), 256),
                                                    (&mut s, x, u, (m * k) as i64))?;
                Some(s)
            }
            None => None,
        };
        let xr = xin.as_ref().unwrap_or(x);
        gf("exl3_had_suh")?.launch_on_stream(st, cfg(((m * k / 128) as u32, 1, 1), 32), (xr, &md.suh, &mut xh, k as i32))?;
        gf("exl3_hmma_gemm_ks")?.launch_on_stream(st, cfg(((n / 128) as u32, ks as u32, 1), 256),
                                                  (&md.tr, &xh, &mut ws, m as i32, k as i32, n as i32, md.bits as i32))?;
        gf("exl3_ks_combine")?.launch_on_stream(st, cfg((((m * n + 255) / 256) as u32, 1, 1), 256),
                                                (&ws, &mut yr, m as i32, n as i32, ks as i32))?;
        gf("exl3_had_svh")?.launch_on_stream(st, cfg(((m * n / 128) as u32, 1, 1), 32), (&yr, &md.svh, &mut y, n as i32))?;
        dev.synchronize()?;
        drop(xin);
    }
    Ok((dev.dtoh_sync_copy(&ws)?, dev.dtoh_sync_copy(&y)?))
}

/// The SERVED fallback of a same-shape pair the DENSE grid rule cannot place: exl3_chain_pair past
/// its DENSE block with PSK at its default (off) — exl3_had_suh_x2 -> exl3_hmma_gemm_ks_x2 ->
/// exl3_ks_combine_x2 -> exl3_had_svh_x2 on x (m rows) -> per member (fp32 partials, y). Partials
/// NaN-poisoned, y 0xFFFF-poisoned first (an unwritten element cannot compare equal).
fn dense_pair_fallback(dev: &std::sync::Arc<CudaDevice>, st: &CudaStream, a: &DenseMod, b: &DenseMod,
                       x: &CudaSlice<u16>, m: usize) -> Result<[(Vec<f32>, Vec<u16>); 2]> {
    let gf = |n: &str| dev.get_func(MODULE, n).with_context(|| n.to_string());
    let (k, n, ks) = (a.k, a.n, a.ks);
    anyhow::ensure!(b.k == k && b.n == n && b.ks == ks && b.bits == a.bits,
                    "pair fallback: members differ (K{k} N{n} ks{ks} b{} vs K{} N{} ks{} b{})", a.bits, b.k, b.n, b.ks, b.bits);
    let poison = f32::from_bits(0x7FC0_DEAD);
    let mut xh0 = dev.htod_sync_copy(&vec![0u16; m * k])?;
    let mut xh1 = dev.htod_sync_copy(&vec![0u16; m * k])?;
    let mut ws0 = dev.htod_sync_copy(&vec![poison; ks * m * n])?;
    let mut ws1 = dev.htod_sync_copy(&vec![poison; ks * m * n])?;
    let mut yr0 = dev.htod_sync_copy(&vec![0u16; m * n])?;
    let mut yr1 = dev.htod_sync_copy(&vec![0u16; m * n])?;
    let mut y0 = dev.htod_sync_copy(&vec![0xFFFFu16; m * n])?;
    let mut y1 = dev.htod_sync_copy(&vec![0xFFFFu16; m * n])?;
    let cfg = |g: (u32, u32, u32), b: u32| LaunchConfig { grid_dim: g, block_dim: (b, 1, 1), shared_mem_bytes: 0 };
    unsafe {
        gf("exl3_had_suh_x2")?.launch_on_stream(st, cfg(((m * k / 128) as u32, 1, 2), 32),
                                                (x, &a.suh, &b.suh, &mut xh0, &mut xh1, k as i32))?;
        gf("exl3_hmma_gemm_ks_x2")?.launch_on_stream(st, cfg(((n / 128) as u32, ks as u32, 2), 256),
                                                     (&a.tr, &b.tr, &xh0, &xh1, &mut ws0, &mut ws1,
                                                      m as i32, k as i32, n as i32, a.bits as i32))?;
        gf("exl3_ks_combine_x2")?.launch_on_stream(st, cfg((((m * n + 255) / 256) as u32, 1, 2), 256),
                                                   (&ws0, &ws1, &mut yr0, &mut yr1, m as i32, n as i32, ks as i32))?;
        gf("exl3_had_svh_x2")?.launch_on_stream(st, cfg(((m * n / 128) as u32, 1, 2), 32),
                                                (&yr0, &yr1, &a.svh, &b.svh, &mut y0, &mut y1, n as i32))?;
    }
    dev.synchronize()?;
    Ok([(dev.dtoh_sync_copy(&ws0)?, dev.dtoh_sync_copy(&y0)?), (dev.dtoh_sync_copy(&ws1)?, dev.dtoh_sync_copy(&y1)?)])
}

/// Widths as compact ranges ("1..8, 11, 14..16"; "none" when empty).
fn m_ranges(ms: &[usize]) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < ms.len() {
        let mut j = i;
        while j + 1 < ms.len() && ms[j + 1] == ms[j] + 1 { j += 1; }
        out.push(if j > i { format!("{}..{}", ms[i], ms[j]) } else { ms[i].to_string() });
        i = j + 1;
    }
    if out.is_empty() { "none".to_string() } else { out.join(", ") }
}

fn probe_dense(dev: &std::sync::Arc<CudaDevice>, pack: &Exl3Pack) -> Result<()> {
    use crate::exl3_forward::dense::{self as dn, ProbeMem, DKS_F_FIX, DKS_F_SILU, DKS_F_SUH};
    use std::collections::BTreeMap;
    let st = fork_blocking_stream(dev);
    let rd16 = |b: Vec<u8>| -> Vec<u16> { b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect() };
    // every split chain module (non-expert, non-head), grouped by (K, N, bits) -> up to 2 modules
    let mut shapes: BTreeMap<(usize, usize, usize), Vec<&ModuleMeta>> = BTreeMap::new();
    for md in &pack.modules {
        if matches!(md.class, TensorClass::LmHead | TensorClass::Ngram) || md.name.contains(".mlp.experts.") { continue; }
        let ks = crate::exl3_forward::chain_ks(md.k as i32, md.n as i32) as usize;
        if !dn::dks_member_ok(md.k, md.n, ks) || !(3..=5).contains(&md.bits) { continue; }
        let e = shapes.entry((md.k, md.n, md.bits)).or_default();
        if e.len() < 2 { e.push(md); }
    }
    if shapes.is_empty() {
        bail!("EXL3-DENSE FAIL: no split chain shapes in the pack (harness failure, nothing was tested)");
    }
    let load = |md: &ModuleMeta| -> Result<DenseMod> {
        Ok(DenseMod {
            name: md.name.clone(),
            tr: dev.htod_sync_copy(&rd16(md.trellis.read_bytes(&pack.dir)?))?,
            suh: dev.htod_sync_copy(&le_f16_bits(&md.suh.read_bytes(&pack.dir)?))?,
            svh: dev.htod_sync_copy(&le_f16_bits(&md.svh.read_bytes(&pack.dir)?))?,
            k: md.k, n: md.n, bits: md.bits,
            ks: crate::exl3_forward::chain_ks(md.k as i32, md.n as i32) as usize,
        })
    };
    let poison = f32::from_bits(0x7FC0_DEAD);
    let served_nst = dn::dense_nst();
    let mut launches = 0usize;

    // One job: members `mods` (same K / bits) on x (+x2) at width m, grid g, ring nst, flags; checks
    // every member's partials (+ y with FIX) against `refs`, counters re-armed. Returns member 0's y.
    #[allow(clippy::too_many_arguments)]
    let mut run = |mods: &[&DenseMod], refs: &[(Vec<f32>, Vec<u16>)], x: &CudaSlice<u16>, x2: Option<&CudaSlice<u16>>,
                   m: usize, g: usize, nst: usize, flags: i32, tag: &str| -> Result<Vec<u16>> {
        let k = mods[0].k;
        let mut wss = Vec::new();
        let mut ys = Vec::new();
        let mut cnts = Vec::new();
        let mut xhs = Vec::new();
        let mut pm = Vec::new();
        for md in mods {
            wss.push(dev.htod_sync_copy(&vec![poison; md.ks * m * md.n])?);
            ys.push(dev.htod_sync_copy(&vec![0xFFFFu16; m * md.n])?);
            cnts.push(dev.htod_sync_copy(&vec![0u32; (md.n / 128).max(1)])?);
            // staged xh (flags without SUH): the old had_suh on the same x
            let mut xh = dev.htod_sync_copy(&vec![0u16; m * k])?;
            if flags & DKS_F_SUH == 0 {
                unsafe {
                    dev.get_func(MODULE, "exl3_had_suh").context("exl3_had_suh")?.launch_on_stream(&st,
                        LaunchConfig { grid_dim: ((m * k / 128) as u32, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 },
                        (x, &md.suh, &mut xh, k as i32))?;
                }
            }
            xhs.push(xh);
        }
        for (j, md) in mods.iter().enumerate() {
            pm.push(ProbeMem { tr: *md.tr.device_ptr(), suh: *md.suh.device_ptr(), svh: *md.svh.device_ptr(),
                               xh: *xhs[j].device_ptr(), ws: *wss[j].device_ptr(), y: *ys[j].device_ptr(),
                               cnt: *cnts[j].device_ptr(), n: md.n, ks: md.ks });
        }
        dev.synchronize()?;
        let x2p = x2.map_or(0u64, |u| *u.device_ptr());
        if !dn::probe_launch(st.stream, &pm, *x.device_ptr(), x2p, m, k, mods[0].bits, g, nst, flags)? {
            bail!("EXL3-DENSE FAIL [{tag}]: launcher refused m={m} G={g} nst={nst} flags={flags}");
        }
        dev.synchronize()?;
        launches += 1;
        let mut y0 = Vec::new();
        for (j, md) in mods.iter().enumerate() {
            let p = dev.dtoh_sync_copy(&wss[j])?;
            let mism = p.iter().zip(refs[j].0.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            if mism > 0 {
                let i = p.iter().zip(refs[j].0.iter()).position(|(a, b)| a.to_bits() != b.to_bits()).unwrap_or(0);
                bail!("EXL3-DENSE FAIL [{tag}] member {j} ({}) K={k} N={} ks={} m={m} G={g} nst={nst} flags={flags}: \
                       {mism} of {} partials differ (first [{i}] new {:e} old {:e})",
                      md.name, md.n, md.ks, p.len(), p[i], refs[j].0[i]);
            }
            if flags & DKS_F_FIX != 0 {
                let y = dev.dtoh_sync_copy(&ys[j])?;
                let mism = y.iter().zip(refs[j].1.iter()).filter(|(a, b)| a != b).count();
                if mism > 0 {
                    let i = y.iter().zip(refs[j].1.iter()).position(|(a, b)| a != b).unwrap_or(0);
                    bail!("EXL3-DENSE FAIL [{tag}] member {j} ({}) K={k} N={} m={m} G={g} nst={nst} flags={flags}: \
                           {mism} of {} y differ (first (row {}, col {}) new {:04x} old {:04x})",
                          md.name, md.n, y.len(), i / md.n, i % md.n, y[i], refs[j].1[i]);
                }
                let nz = dev.dtoh_sync_copy(&cnts[j])?.iter().filter(|&&c| c != 0).count();
                if nz > 0 {
                    bail!("EXL3-DENSE FAIL [{tag}] member {j} K={k} N={} m={m} G={g}: {nz} tile counters not re-armed", md.n);
                }
                if j == 0 { y0 = y; }
            }
        }
        Ok(y0)
    };
    let plausible = |r: &(Vec<f32>, Vec<u16>), tag: &str, m: usize| -> Result<()> {
        let live = r.0.iter().filter(|v| v.is_finite() && **v != 0.0).count();
        let ylive = r.1.iter().filter(|v| **v != 0 && **v != 0xFFFF).count();
        if live < r.0.len() / 2 || ylive < r.1.len() / 2 {
            bail!("EXL3-DENSE FAIL [{tag}]: implausible reference at m={m} ({live}/{} partials, {ylive}/{} y live)",
                  r.0.len(), r.1.len());
        }
        Ok(())
    };
    let full = DKS_F_SUH | DKS_F_FIX;
    println!("    W4/DENSE: {} split chain shape(s); served ring nst {served_nst}", shapes.len());
    // (job, m) cells with no feasible served grid (skipped), and pair fallbacks bit-checked there
    let (mut nskip, mut nfb) = (0usize, 0usize);
    const NOGRID: &str = "no feasible grid (served path falls back to the old chain) — skipped";

    // ---- singles: every shape, M 1..16, a G sweep; ring depths + escape flag sets at M 1/8/16
    let mut mods_all: Vec<DenseMod> = Vec::new();
    let mut idx_of: BTreeMap<(usize, usize, usize), Vec<usize>> = BTreeMap::new();
    for (&key, mds) in &shapes {
        for md in mds {
            idx_of.entry(key).or_default().push(mods_all.len());
            mods_all.push(load(md)?);
        }
    }
    for (&(k, n, bits), ids) in &idx_of {
        let md = &mods_all[ids[0]];
        let items = (n / 128) * md.ks;
        let mut row0: Option<(usize, Vec<u16>)> = None;
        let mut gl_all = Vec::new();
        let (mut ran, mut skipped) = (Vec::new(), Vec::new());
        for m in 1..=16usize {
            let x = dev.htod_sync_copy(&synth_x(m, k, 0x5EED))?;
            let r = dense_ref(dev, &st, md, &x, None, m)?;
            plausible(&r, "single", m)?;
            let Some(gs) = dn::probe_grid(dev, k, bits, &[(n, md.ks)], m) else {
                // served: exl3_chain_impl's own path — the reference chain above
                println!("      EXL3-DENSE single K={k} N={n} b{bits} m={m}: {NOGRID} (the fallback is this section's reference chain)");
                skipped.push(m);
                nskip += 1;
                continue;
            };
            let mut gl = vec![gs, 1, 7, 13, 40, 48, items + 3];
            gl.sort_unstable();
            gl.dedup();
            // only grids the launcher accepts (A entries / smem budget) — the served picker and the
            // --w4dense-g override never choose another
            gl.retain(|&g| dn::dks_feasible(&[(n, md.ks)], k, bits, served_nst, g, m));
            for &g in &gl {
                let y = run(&[md], std::slice::from_ref(&r), &x, None, m, g, served_nst, full, "single")?;
                if g == gs {
                    match &row0 {
                        None => row0 = Some((m, y[..n].to_vec())),
                        Some((m0, r0)) => if r0[..] != y[..n] {
                            bail!("EXL3-DENSE BINV FAIL: K={k} N={n} b{bits}: y row 0 at m={m} differs from m={m0}");
                        },
                    }
                }
            }
            if matches!(m, 1 | 8 | 16) {
                let nsts: &[usize] = if bits == 3 { &[6] } else { &[4, 6, 8] };
                for &ns in nsts {
                    if dn::dks_feasible(&[(n, md.ks)], k, bits, ns, gs, m) {
                        run(&[md], std::slice::from_ref(&r), &x, None, m, gs, ns, full, "single ring")?;
                    }
                }
                for fl in [DKS_F_FIX, DKS_F_SUH, 0] {
                    run(&[md], std::slice::from_ref(&r), &x, None, m, gs, served_nst, fl, "single flags")?;
                }
            }
            gl_all = gl;
            ran.push(m);
        }
        let sk = if skipped.is_empty() { String::new() } else { format!(" [M {} skipped: no feasible grid]", m_ranges(&skipped)) };
        if ran.is_empty() {
            println!("      K={k:<5} N={n:<6} b{bits} ks={:<3} items {items:>4}: single — every width skipped (no feasible grid; \
                      the served path runs the old chain) ({})", md.ks, md.name);
        } else {
            println!("      K={k:<5} N={n:<6} b{bits} ks={:<3} items {items:>4}: single bitwise == old chain, M {}, G {gl_all:?}, \
                      nst sweep + flag sets at M 1/8/16, counters re-armed, y row-0 binv PASS{sk} ({})",
                     md.ks, m_ranges(&ran), md.name);
        }
    }

    // ---- pairs: two modules of one shape on the same x (the k|v / gate|up launch shape)
    let mut pairs = 0usize;
    for (&(k, n, bits), ids) in &idx_of {
        if ids.len() < 2 { continue; }
        let (a, b) = (&mods_all[ids[0]], &mods_all[ids[1]]);
        let items = 2 * (n / 128) * a.ks;
        let (mut ran, mut fb) = (Vec::new(), Vec::new());
        for m in 1..=16usize {
            let x = dev.htod_sync_copy(&synth_x(m, k, 0xA11C))?;
            let refs = [dense_ref(dev, &st, a, &x, None, m)?, dense_ref(dev, &st, b, &x, None, m)?];
            let Some(gs) = dn::probe_grid(dev, k, bits, &[(n, a.ks), (n, b.ks)], m) else {
                // served: exl3_chain_pair -> dense_chains declines (nothing launched) -> the x2 chain;
                // run it and bit-check both members against the old chains
                plausible(&refs[0], "pair fallback", m)?;
                plausible(&refs[1], "pair fallback", m)?;
                let got = dense_pair_fallback(dev, &st, a, b, &x, m)?;
                for (j, (g, r)) in got.iter().zip(refs.iter()).enumerate() {
                    let pm = g.0.iter().zip(r.0.iter()).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
                    let ym = g.1.iter().zip(r.1.iter()).filter(|(p, q)| p != q).count();
                    if pm + ym > 0 {
                        bail!("EXL3-DENSE FAIL [pair fallback] member {j} ({}) K={k} N={n} b{bits} m={m}: the served x2 fallback \
                               chain differs from the old chain ({pm} of {} partials, {ym} of {} y)",
                              if j == 0 { &a.name } else { &b.name }, g.0.len(), g.1.len());
                    }
                }
                println!("      EXL3-DENSE pair K={k} N={n} b{bits} m={m}: {NOGRID}; fallback x2 chain (exl3_had_suh_x2 + \
                          exl3_hmma_gemm_ks_x2 + exl3_ks_combine_x2 + exl3_had_svh_x2) bitwise == old chains (partials + y, \
                          both members)");
                fb.push(m);
                nskip += 1;
                nfb += 1;
                continue;
            };
            let mut gl = vec![gs, 1, 13, items + 3];
            gl.sort_unstable();
            gl.dedup();
            gl.retain(|&g| dn::dks_feasible(&[(n, a.ks), (n, b.ks)], k, bits, served_nst, g, m));
            for &g in &gl { run(&[a, b], &refs, &x, None, m, g, served_nst, full, "pair")?; }
            if matches!(m, 1 | 8 | 16) { run(&[a, b], &refs, &x, None, m, gs, served_nst, DKS_F_FIX, "pair staged xh")?; }
            ran.push(m);
        }
        pairs += 1;
        println!("      pair K={k} N={n} b{bits}: DENSE bitwise == old chains at M {}{} ({} | {})", m_ranges(&ran),
                 if fb.is_empty() { String::new() }
                 else { format!("; M {}: no feasible grid, served x2 fallback bitwise == old chains", m_ranges(&fb)) },
                 a.name, b.name);
    }

    // ---- multi-member group: per (K, bits), the largest-N module first, then the smallest shapes'
    // modules (the attention q + k|v + indexer composition when the pack has it), up to 4 members.
    let mut groups = 0usize;
    let mut by_kb: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
    for (&(k, _n, bits), ids) in &idx_of { by_kb.entry((k, bits)).or_default().extend(ids.iter().copied()); }
    for (&(k, bits), ids) in &by_kb {
        if ids.len() < 3 { continue; }
        let mut sorted = ids.clone();
        sorted.sort_by_key(|&i| std::cmp::Reverse(mods_all[i].n));
        let mut pick = vec![sorted[0]];
        for &i in sorted.iter().rev() { if pick.len() < 4 && !pick.contains(&i) { pick.push(i); } }
        let mods: Vec<&DenseMod> = pick.iter().map(|&i| &mods_all[i]).collect();
        let mem: Vec<(usize, usize)> = mods.iter().map(|d| (d.n, d.ks)).collect();
        let (mut ran, mut skipped) = (Vec::new(), Vec::new());
        for m in 1..=16usize {
            let Some(gs) = dn::probe_grid(dev, k, bits, &mem, m) else {
                // served: the group splits back to its own chains (each the reference chain here)
                println!("      EXL3-DENSE multi K={k} b{bits} {} m={m}: {NOGRID} (the fallback is this section's reference chains)",
                         mods.iter().map(|d| format!("N{}", d.n)).collect::<Vec<_>>().join("+"));
                skipped.push(m);
                nskip += 1;
                continue;
            };
            let x = dev.htod_sync_copy(&synth_x(m, k, 0x6A0F))?;
            let refs: Vec<(Vec<f32>, Vec<u16>)> = mods.iter().map(|d| dense_ref(dev, &st, d, &x, None, m)).collect::<Result<_>>()?;
            let mut gl = vec![gs, 1, 13, 40, 48];
            gl.sort_unstable();
            gl.dedup();
            gl.retain(|&g| dn::dks_feasible(&mem, k, bits, served_nst, g, m));
            for &g in &gl { run(&mods, &refs, &x, None, m, g, served_nst, full, "multi")?; }
            ran.push(m);
        }
        groups += 1;
        println!("      multi K={k} b{bits} [{}]: bitwise == the old chains, M {}{}",
                 mods.iter().map(|d| format!("N{} {}", d.n, d.name)).collect::<Vec<_>>().join(" + "), m_ranges(&ran),
                 if skipped.is_empty() { String::new() } else { format!(" [M {} skipped: no feasible grid]", m_ranges(&skipped)) });
    }

    // ---- silu prologue (the shared-expert down): x := f16(silu(x) * x2) before the suh
    let silu_shape = idx_of.keys().find(|&&(k, _, _)| k == 640).or_else(|| idx_of.keys().next()).copied();
    if let Some(key) = silu_shape {
        let md = &mods_all[idx_of[&key][0]];
        let (k, n, bits) = key;
        let (mut ran, mut skipped) = (Vec::new(), Vec::new());
        for m in 1..=16usize {
            let x = dev.htod_sync_copy(&synth_x(m, k, 0x51A0))?;
            let u = dev.htod_sync_copy(&synth_x(m, k, 0x0B0E))?;
            let r = dense_ref(dev, &st, md, &x, Some(&u), m)?;
            plausible(&r, "silu", m)?;
            let Some(gs) = dn::probe_grid(dev, k, bits, &[(n, md.ks)], m) else {
                // served: shared_down's xq_silu_mul + exl3_chain — this section's reference chain
                println!("      EXL3-DENSE silu K={k} N={n} b{bits} m={m}: {NOGRID} (the fallback is this section's reference chain)");
                skipped.push(m);
                nskip += 1;
                continue;
            };
            for g in [gs, 1, 13] {
                if g == gs || dn::dks_feasible(&[(n, md.ks)], k, bits, served_nst, g, m) {
                    run(&[md], std::slice::from_ref(&r), &x, Some(&u), m, g, served_nst, full | DKS_F_SILU, "silu")?;
                }
            }
            run(&[md], std::slice::from_ref(&r), &x, Some(&u), m, gs, served_nst, DKS_F_SUH | DKS_F_SILU, "silu partials")?;
            ran.push(m);
        }
        println!("      silu prologue K={k} N={n} b{bits}: bitwise == xq_silu_mul + old chain, M {}{} ({})", m_ranges(&ran),
                 if skipped.is_empty() { String::new() } else { format!(" [M {} skipped: no feasible grid]", m_ranges(&skipped)) },
                 md.name);
    }
    if launches == 0 {
        bail!("EXL3-DENSE FAIL: no DENSE launch ran ({nskip} cell(s) skipped: no feasible grid) — harness failure, nothing was tested");
    }
    println!("    EXL3-DENSE: PASS ({launches} launches bitwise == the old chains: {} single shape(s), {pairs} pair(s), \
              {groups} multi group(s), silu; partials + y + counters; {nskip} (job, M) cell(s) skipped: no feasible grid, \
              the served path falls back to the old chain — {nfb} pair fallback(s) run and bitwise == the old chains)",
             idx_of.len());
    Ok(())
}

// ---------------------------------------------------------------------------
// S-A3-m: xq_gemm_f16_rows (row-batched) must be BITWISE equal per row to
// xq_gemm_f16 (grid-y per row) — the PLE projection shapes, M in {1,3,6,16}.
// ---------------------------------------------------------------------------
fn probe_gemm_f16_rows(dev: &std::sync::Arc<CudaDevice>) -> Result<()> {
    let f_old = dev.get_func(MODULE, "xq_gemm_f16").context("xq_gemm_f16")?;
    let f_new = dev.get_func(MODULE, "xq_gemm_f16_rows").context("xq_gemm_f16_rows")?;
    for &(n, k) in &[(10240usize, 2560usize), (2560, 2560)] {
        let w = synth_x(n, k, 0xF16);
        let d_w = dev.htod_sync_copy(&w)?;
        for m in [1usize, 3, 6, 16] {
            let x = synth_x(m, k, 0xA11);
            let d_x = dev.htod_sync_copy(&x)?;
            let mut o1 = dev.htod_sync_copy(&vec![0u16; m * n])?;
            let mut o2 = dev.htod_sync_copy(&vec![0u16; m * n])?;
            let g = ((n + 127) / 128) as u32;
            unsafe {
                f_old.clone().launch(LaunchConfig { grid_dim: (g, m as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 },
                                     (&mut o1, &d_w, &d_x, m as i32, n as i32, k as i32))?;
                f_new.clone().launch(LaunchConfig { grid_dim: (g, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 },
                                     (&mut o2, &d_w, &d_x, m as i32, n as i32, k as i32))?;
            }
            dev.synchronize()?;
            let a = dev.dtoh_sync_copy(&o1)?;
            let b = dev.dtoh_sync_copy(&o2)?;
            if a.iter().filter(|v| **v != 0).count() < m * n / 2 {
                bail!("EXL3-F16ROWS FAIL: implausible reference output (m={m} n={n})");
            }
            let mism = a.iter().zip(b.iter()).filter(|(p, q)| p != q).count();
            if mism > 0 { bail!("EXL3-F16ROWS FAIL: m={m} N={n} K={k}: {mism} mismatches"); }
        }
    }
    println!("EXL3-F16ROWS: PASS (row-batched PLE GEMV bitwise == xq_gemm_f16, M in {{1,3,6,16}})");
    // WP09: the k-major twin ([K/16][N][16] weights, key+value in ONE launch — the served shape)
    // must be BITWISE equal to xq_gemm_f16 per row; plus a tail-path shape (K/16 % 4 != 0,
    // N % 128 != 0, single matrix).
    {
        let f_ref = dev.get_func(MODULE, "xq_gemm_f16").context("xq_gemm_f16")?;
        let f_km = dev.get_func(MODULE, "xq_gemm_f16_rows_km").context("xq_gemm_f16_rows_km")?;
        let km = crate::exl3_forward::relayout_km16;
        let mut done = Vec::new();
        for &(na, nb2, k) in &[(10240usize, 2560usize, 2560usize), (300, 0, 1040)] {
            let wa = synth_x(na, k, 0xF16);
            let wb = synth_x(nb2.max(1), k, 0xF17);
            let d_wa = dev.htod_sync_copy(&wa)?;
            let d_wb = dev.htod_sync_copy(&wb)?;
            let d_wa_km = dev.htod_sync_copy(&km(&wa, na, k))?;
            let d_wb_km = dev.htod_sync_copy(&km(&wb, nb2.max(1), k))?;
            let (ga, gb) = (((na + 127) / 128) as u32, ((nb2 + 127) / 128) as u32);
            for m in [1usize, 2, 3, 5, 6, 8, 9, 16] {
                let d_x = dev.htod_sync_copy(&synth_x(m, k, 0xA11 ^ m as u32))?;
                let mut ra = dev.htod_sync_copy(&vec![0u16; m * na])?;
                let mut rb = dev.htod_sync_copy(&vec![0u16; m * nb2.max(1)])?;
                let mut ka = dev.htod_sync_copy(&vec![0u16; m * na])?;
                let mut kb = dev.htod_sync_copy(&vec![0u16; m * nb2.max(1)])?;
                unsafe {
                    f_ref.clone().launch(LaunchConfig { grid_dim: (ga, m as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 },
                                         (&mut ra, &d_wa, &d_x, m as i32, na as i32, k as i32))?;
                    if nb2 > 0 {
                        f_ref.clone().launch(LaunchConfig { grid_dim: (gb, m as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 },
                                             (&mut rb, &d_wb, &d_x, m as i32, nb2 as i32, k as i32))?;
                    }
                    f_km.clone().launch(LaunchConfig { grid_dim: (ga + gb, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 },
                                        (&mut ka, &d_wa_km, na as i32, ga as i32, &mut kb, &d_wb_km, nb2 as i32,
                                         &d_x, m as i32, k as i32))?;
                }
                // W4/SMALL: the x-in-smem exact-M twin (m <= 8), 160-thread blocks, poisoned outputs
                let mut wa4 = dev.htod_sync_copy(&vec![0x7e01u16; m * na])?;
                let mut wb4 = dev.htod_sync_copy(&vec![0x7e01u16; m * nb2.max(1)])?;
                if m <= 8 {
                    let f_w4 = dev.get_func(MODULE, &format!("xq_gemm_f16_rows_km_w4m{m}"))
                        .ok_or_else(|| anyhow::anyhow!("xq_gemm_f16_rows_km_w4m{m} missing"))?;
                    let (ga4, gb4) = (na.div_ceil(160) as u32, nb2.div_ceil(160) as u32);
                    unsafe {
                        f_w4.launch(LaunchConfig { grid_dim: (ga4 + gb4, 1, 1), block_dim: (160, 1, 1),
                                                   shared_mem_bytes: (m * k * 2) as u32 },
                                    (&mut wa4, &d_wa_km, na as i32, ga4 as i32, &mut wb4, &d_wb_km, nb2 as i32,
                                     &d_x, m as i32, k as i32))?;
                    }
                }
                dev.synchronize()?;
                for (tag, r, t, t4, n) in [("a", &ra, &ka, &wa4, na), ("b", &rb, &kb, &wb4, nb2)] {
                    if n == 0 { continue; }
                    let a = dev.dtoh_sync_copy(r)?;
                    let b = dev.dtoh_sync_copy(t)?;
                    if a.iter().filter(|v| **v != 0).count() < m * n / 2 {
                        bail!("EXL3-F16ROWS-KM FAIL: implausible reference output (m={m} n={n} {tag})");
                    }
                    let mism = a.iter().zip(b.iter()).filter(|(p, q)| p != q).count();
                    if mism > 0 { bail!("EXL3-F16ROWS-KM FAIL: m={m} N={n} K={k} ({tag}): {mism} mismatches vs xq_gemm_f16"); }
                    if m <= 8 {
                        let c = dev.dtoh_sync_copy(t4)?;
                        let mism4 = a.iter().zip(c.iter()).take(m * n).filter(|(p, q)| p != q).count();
                        if mism4 > 0 {
                            bail!("EXL3-F16ROWS-KM FAIL: W4S xq_gemm_f16_rows_km_w4m{m} N={n} K={k} ({tag}): {mism4} mismatches vs xq_gemm_f16");
                        }
                    }
                }
                done.push((na, nb2, k, m));
            }
        }
        println!("EXL3-F16ROWS-KM: PASS (k-major key+value twin bitwise == xq_gemm_f16; {} (Na,Nb,K,M) cases incl. tail K/N; \
                  W4S _w4m<M> twin bitwise too at M <= 8)", done.len());
    }
    probe_argmax_rows(dev)?;
    // S-A3-u B: the prefill router twins must be BITWISE equal to xq_gemm_f16_f32 (router bits
    // decide top-k selections, AGENTS §7).
    {
        let f_ref = dev.get_func(MODULE, "xq_gemm_f16_f32").context("xq_gemm_f16_f32")?;
        let f_r32 = dev.get_func(MODULE, "xq_gemm_f16_f32_rows32").context("xq_gemm_f16_f32_rows32")?;
        let (n, k) = (512usize, 2560usize);
        let d_w = dev.htod_sync_copy(&synth_x(n, k, 0x5A3))?;
        for m in [1usize, 17, 33, 100, 2048] {
            let d_x = dev.htod_sync_copy(&synth_x(m, k, 0x77 ^ m as u32))?;
            let mut o1 = dev.htod_sync_copy(&vec![0f32; m * n])?;
            let mut o2 = dev.htod_sync_copy(&vec![0f32; m * n])?;
            unsafe {
                f_ref.clone().launch(LaunchConfig { grid_dim: ((n / 128) as u32, m as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 },
                                     (&mut o1, &d_w, &d_x, m as i32, n as i32, k as i32))?;
                f_r32.clone().launch(LaunchConfig { grid_dim: ((n / 128) as u32, ((m + 31) / 32) as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 },
                                     (&mut o2, &d_w, &d_x, m as i32, n as i32, k as i32))?;
            }
            dev.synchronize()?;
            let a = dev.dtoh_sync_copy(&o1)?; let b = dev.dtoh_sync_copy(&o2)?;
            if a.iter().filter(|v| **v != 0.0).count() < m * n / 2 { bail!("EXL3-ROUTER32 FAIL: implausible reference (m={m})"); }
            let mism = a.iter().zip(b.iter()).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
            if mism > 0 { bail!("EXL3-ROUTER32 FAIL: m={m}: {mism} mismatches vs xq_gemm_f16_f32"); }
        }
        println!("EXL3-ROUTER32: PASS (32-row smem router twin bitwise == xq_gemm_f16_f32, M in {{1,17,33,100,2048}})");
        // PFX1 (d): the 64-col x 8-row twin must equal BOTH the reference and the current router
        // (rows32) bit for bit at the served chunk shapes (and M = 2048, past its dispatch range).
        let f_r8 = dev.get_func(MODULE, "xq_gemm_f16_f32_rows8").context("xq_gemm_f16_f32_rows8")?;
        let ms8 = [1usize, 7, 22, 26, 72, 144, 384, 2048];
        for &m in &ms8 {
            let d_x = dev.htod_sync_copy(&synth_x(m, k, 0x88 ^ m as u32))?;
            let mut o1 = dev.htod_sync_copy(&vec![0f32; m * n])?;
            let mut o2 = dev.htod_sync_copy(&vec![0f32; m * n])?;
            // poisoned output: a row the kernel fails to write cannot pass as 0 == 0
            let mut o3 = dev.htod_sync_copy(&vec![f32::from_bits(0x7fc0_dead); m * n])?;
            unsafe {
                f_ref.clone().launch(LaunchConfig { grid_dim: ((n / 128) as u32, m as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 },
                                     (&mut o1, &d_w, &d_x, m as i32, n as i32, k as i32))?;
                f_r32.clone().launch(LaunchConfig { grid_dim: ((n / 128) as u32, ((m + 31) / 32) as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 },
                                     (&mut o2, &d_w, &d_x, m as i32, n as i32, k as i32))?;
                f_r8.clone().launch(LaunchConfig { grid_dim: ((n / 64) as u32, m.div_ceil(8) as u32, 1), block_dim: (64, 1, 1), shared_mem_bytes: 0 },
                                    (&mut o3, &d_w, &d_x, m as i32, n as i32, k as i32))?;
            }
            dev.synchronize()?;
            let a = dev.dtoh_sync_copy(&o1)?; let b = dev.dtoh_sync_copy(&o2)?; let c8 = dev.dtoh_sync_copy(&o3)?;
            if a.iter().filter(|v| **v != 0.0).count() < m * n / 2 { bail!("EXL3-ROUTER8 FAIL: implausible reference (m={m})"); }
            let mr = a.iter().zip(c8.iter()).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
            let m32 = b.iter().zip(c8.iter()).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
            if mr > 0 || m32 > 0 {
                bail!("EXL3-ROUTER8 FAIL: m={m}: {mr} mismatches vs xq_gemm_f16_f32, {m32} vs rows32 (of {})", m * n);
            }
            println!("EXL3-ROUTER8: m={m:>4}: 0 of {} bits differ (vs xq_gemm_f16_f32 and rows32)", m * n);
        }
        println!("EXL3-ROUTER8: PASS (64x8 register-prefetch router twin bitwise == xq_gemm_f16_f32 == rows32, M in {ms8:?})");
        // Standalone per-layer timing (harness-only: it RANKS the bodies, it does not price the
        // in-round cost — AGENTS §4). Weights rotated over 16 copies (42 MB > L2) so every
        // launch reads its router from DRAM, as the 48 per-layer routers of a prefill do.
        let nw = 16usize;
        let wall: Vec<u16> = (0..nw).flat_map(|i| synth_x(n, k, 0x5A3 ^ (i as u32 * 131))).collect();
        let d_wall = dev.htod_sync_copy(&wall)?;
        let wb0 = *d_wall.device_ptr() as u64;
        let wstride = (n * k * 2) as u64;
        // 256..448 bracket the rows8/rows32 crossover (PFX1_ROWS8_MAX, exl3_forward.rs); "served"
        // is what the prefill router dispatch picks at this m on this device (env overrides apply).
        for m in [26usize, 72, 160, 256, 320, 384, 448, 512, 736] {
            let d_x = dev.htod_sync_copy(&synth_x(m, k, 0x99 ^ m as u32))?;
            let mut o = dev.htod_sync_copy(&vec![0f32; m * n])?;
            let mut us = [0f64; 2];
            for (which, slot) in us.iter_mut().enumerate() {
                let mut run = |it: usize| -> Result<()> {
                    let wp = wb0 + (it % nw) as u64 * wstride;
                    unsafe {
                        if which == 0 {
                            f_r32.clone().launch(LaunchConfig { grid_dim: ((n / 128) as u32, ((m + 31) / 32) as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 },
                                                 (&mut o, wp, &d_x, m as i32, n as i32, k as i32))?;
                        } else {
                            f_r8.clone().launch(LaunchConfig { grid_dim: ((n / 64) as u32, m.div_ceil(8) as u32, 1), block_dim: (64, 1, 1), shared_mem_bytes: 0 },
                                                (&mut o, wp, &d_x, m as i32, n as i32, k as i32))?;
                        }
                    }
                    Ok(())
                };
                for it in 0..nw { run(it)?; }
                dev.synchronize()?;
                let iters = 20 * nw;
                let t = std::time::Instant::now();
                for it in 0..iters { run(it)?; }
                dev.synchronize()?;
                *slot = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
            }
            println!("    EXL3-ROUTER8 timing (standalone, harness-only, DRAM-cold weights) m={m}: rows32 {:.1} us/layer, \
                      rows8 {:.1} us/layer ({:.2}x); x48 layers = {:.2} -> {:.2} ms per chunk; served: {}",
                     us[0], us[1], us[0] / us[1].max(1e-9), us[0] * 48e-3, us[1] * 48e-3,
                     if crate::exl3_forward::pfx1_router_rows8(m, n, k) { "rows8" } else { "rows32" });
        }
    }
    probe_router_fold(dev)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// WP10: xq_router_fold (router GEMV + softmax / top-k / renorm / first-seen route in ONE launch)
// must be BITWISE equal to the S-A3-z pair xq_router_fused + xq_router_topk_route — logits, ids,
// wts, slotmap, esel, idxmap / offs_gu / offs_d [..esel] — for M in 1..=8 (ne 512, topk 10,
// K 2560) on random, tie-heavy (duplicated expert rows: equal logits -> equal probs -> id-asc
// order) and NaN-row data, each fold launched TWICE back-to-back (the self re-arming counter must
// read 0 after). Both GEMV lane maps (xq_router_fold, xq_router_fold_m0). Then a STANDALONE
// per-layer timing (weights rotated over 16 copies > L2): it ranks the bodies, it does not price
// the in-round cost (AGENTS §4) — harness-only numbers.
// ---------------------------------------------------------------------------
fn probe_router_fold(dev: &std::sync::Arc<CudaDevice>) -> Result<()> {
    use cudarc::driver::sys;
    use std::ffi::c_void;
    let raw = |n: &str| crate::exl3_forward::xq_raw_fn(n)
        .ok_or_else(|| anyhow::anyhow!("{n} raw fn unavailable (stale PTX?)"));
    let (f_fused, f_route) = (raw("xq_router_fused")?, raw("xq_router_topk_route")?);
    let (f_fold, f_fold0) = (raw("xq_router_fold")?, raw("xq_router_fold_m0")?);
    // W4/SMALL: the persistent twin (grid #SMs x 768, x staged in smem)
    let f_w4 = raw("xq_router_fold_w4")?;
    // A5-K2: the coalesced-weight twin (grid ne/4 x 256, dynamic smem RFC_SMEM)
    let f_c = raw("xq_router_fold_c")?;
    let (ne, k, kd, mmax) = (512usize, 10usize, 2560usize, 8usize);
    let (gw, dw) = (0x0012_3457u64, 0x9abcu64);
    let g_w4 = crate::exl3_forward::w4s_router_grid(dev, ne);
    unsafe fn go(f: sys::CUfunction, grid: u32, smem: u32, args: &mut [*mut c_void]) -> Result<()> {
        go_b(f, grid, 256, smem, args)
    }
    unsafe fn go_b(f: sys::CUfunction, grid: u32, block: u32, smem: u32, args: &mut [*mut c_void]) -> Result<()> {
        let r = sys::cuLaunchKernel(f, grid, 1, 1, block, 1, 1, smem, std::ptr::null_mut(),
                                    args.as_mut_ptr(), std::ptr::null_mut());
        anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "router fold probe launch ({r:?})");
        Ok(())
    }
    struct Out { lg: CudaSlice<f32>, ids: CudaSlice<i32>, wts: CudaSlice<f32>, sm: CudaSlice<i32>,
                 ix: CudaSlice<i32>, es: CudaSlice<i32>, og: CudaSlice<u64>, od: CudaSlice<u64>,
                 cnt: CudaSlice<i32> }
    let mk = || -> Result<Out> {
        Ok(Out { lg: dev.htod_sync_copy(&vec![0f32; mmax * ne])?, ids: dev.htod_sync_copy(&vec![-7i32; mmax * k])?,
                 wts: dev.htod_sync_copy(&vec![0f32; mmax * k])?, sm: dev.htod_sync_copy(&vec![-9i32; ne])?,
                 ix: dev.htod_sync_copy(&vec![-5i32; mmax * k])?, es: dev.htod_sync_copy(&[-1i32])?,
                 og: dev.htod_sync_copy(&vec![0u64; mmax * k])?, od: dev.htod_sync_copy(&vec![0u64; mmax * k])?,
                 cnt: dev.htod_sync_copy(&[0i32])? })
    };
    let dp = |p: &cudarc::driver::sys::CUdeviceptr| *p as u64;
    let pair = |o: &Out, w: u64, x: u64, m: usize| -> Result<()> {
        let (mut a_lg, mut a_w, mut a_x) = (dp(o.lg.device_ptr()), w, x);
        let (mut a_m, mut a_n, mut a_h) = (m as i32, ne as i32, kd as i32);
        let mut p1: [*mut c_void; 6] = [&mut a_lg as *mut u64 as *mut _, &mut a_w as *mut u64 as *mut _,
            &mut a_x as *mut u64 as *mut _, &mut a_m as *mut i32 as *mut _, &mut a_n as *mut i32 as *mut _,
            &mut a_h as *mut i32 as *mut _];
        unsafe { go(f_fused, (ne / 4) as u32, 0, &mut p1)? };
        let (mut a_ids, mut a_wts, mut a_lg2) = (dp(o.ids.device_ptr()), dp(o.wts.device_ptr()), dp(o.lg.device_ptr()));
        let (mut a_ne, mut a_k, mut a_b) = (ne as i32, k as i32, m as i32);
        let (mut a_cnt, mut a_sm, mut a_ix) = (dp(o.cnt.device_ptr()), dp(o.sm.device_ptr()), dp(o.ix.device_ptr()));
        let (mut a_es, mut a_og, mut a_od) = (dp(o.es.device_ptr()), dp(o.og.device_ptr()), dp(o.od.device_ptr()));
        let (mut a_gw, mut a_dw) = (gw, dw);
        let mut p2: [*mut c_void; 14] = [&mut a_ids as *mut u64 as *mut _, &mut a_wts as *mut u64 as *mut _,
            &mut a_lg2 as *mut u64 as *mut _, &mut a_ne as *mut i32 as *mut _, &mut a_k as *mut i32 as *mut _,
            &mut a_b as *mut i32 as *mut _, &mut a_cnt as *mut u64 as *mut _, &mut a_sm as *mut u64 as *mut _,
            &mut a_ix as *mut u64 as *mut _, &mut a_es as *mut u64 as *mut _, &mut a_og as *mut u64 as *mut _,
            &mut a_od as *mut u64 as *mut _, &mut a_gw as *mut u64 as *mut _, &mut a_dw as *mut u64 as *mut _];
        unsafe { go(f_route, m as u32, ((256 + ne + 8 * 16 * 2) * 4) as u32, &mut p2) }
    };
    let fold = |f: sys::CUfunction, o: &Out, w: u64, x: u64, m: usize| -> Result<()> {
        let (mut a_lg, mut a_w, mut a_x) = (dp(o.lg.device_ptr()), w, x);
        let (mut a_m, mut a_n, mut a_h) = (m as i32, ne as i32, kd as i32);
        let (mut a_cnt, mut a_ids, mut a_wts) = (dp(o.cnt.device_ptr()), dp(o.ids.device_ptr()), dp(o.wts.device_ptr()));
        let mut a_k = k as i32;
        let (mut a_sm, mut a_ix, mut a_es) = (dp(o.sm.device_ptr()), dp(o.ix.device_ptr()), dp(o.es.device_ptr()));
        let (mut a_og, mut a_od, mut a_gw, mut a_dw) = (dp(o.og.device_ptr()), dp(o.od.device_ptr()), gw, dw);
        let mut p: [*mut c_void; 17] = [&mut a_lg as *mut u64 as *mut _, &mut a_w as *mut u64 as *mut _,
            &mut a_x as *mut u64 as *mut _, &mut a_m as *mut i32 as *mut _, &mut a_n as *mut i32 as *mut _,
            &mut a_h as *mut i32 as *mut _, &mut a_cnt as *mut u64 as *mut _, &mut a_ids as *mut u64 as *mut _,
            &mut a_wts as *mut u64 as *mut _, &mut a_k as *mut i32 as *mut _, &mut a_sm as *mut u64 as *mut _,
            &mut a_ix as *mut u64 as *mut _, &mut a_es as *mut u64 as *mut _, &mut a_og as *mut u64 as *mut _,
            &mut a_od as *mut u64 as *mut _, &mut a_gw as *mut u64 as *mut _, &mut a_dw as *mut u64 as *mut _];
        if f == f_w4 {
            let smem = crate::exl3_forward::w4s_router_smem(m, kd) as u32;
            unsafe { go_b(f, g_w4, crate::exl3_forward::W4S_ROUTER_NT, smem, &mut p) }
        } else if f == f_c {
            unsafe { go(f, (ne / 4) as u32, crate::exl3_forward::RFC_SMEM, &mut p) }
        } else {
            unsafe { go(f, (ne / 4) as u32, 0, &mut p) }
        }
    };
    // bitwise diff count of b vs the reference a (rows < m); also returns the reference esel.
    let cmp = |a: &Out, b: &Out, m: usize| -> Result<(usize, i32, i32)> {
        dev.synchronize()?;
        let (la, lb) = (dev.dtoh_sync_copy(&a.lg)?, dev.dtoh_sync_copy(&b.lg)?);
        let (ia, ib) = (dev.dtoh_sync_copy(&a.ids)?, dev.dtoh_sync_copy(&b.ids)?);
        let (wa, wb) = (dev.dtoh_sync_copy(&a.wts)?, dev.dtoh_sync_copy(&b.wts)?);
        let (sa, sb) = (dev.dtoh_sync_copy(&a.sm)?, dev.dtoh_sync_copy(&b.sm)?);
        let (xa, xb) = (dev.dtoh_sync_copy(&a.ix)?, dev.dtoh_sync_copy(&b.ix)?);
        let (ea, eb) = (dev.dtoh_sync_copy(&a.es)?[0], dev.dtoh_sync_copy(&b.es)?[0]);
        let (ga, gb) = (dev.dtoh_sync_copy(&a.og)?, dev.dtoh_sync_copy(&b.og)?);
        let (da, db) = (dev.dtoh_sync_copy(&a.od)?, dev.dtoh_sync_copy(&b.od)?);
        let cnt_b = dev.dtoh_sync_copy(&b.cnt)?[0];
        let ns = (ea.max(0) as usize).min(mmax * k);
        let mut d = (0..m * ne).filter(|&i| la[i].to_bits() != lb[i].to_bits()).count();
        d += (0..m * k).filter(|&i| ia[i] != ib[i] || wa[i].to_bits() != wb[i].to_bits()).count();
        d += (sa != sb) as usize + (ea != eb) as usize + (xa[..ns] != xb[..ns]) as usize
            + (ga[..ns] != gb[..ns]) as usize + (da[..ns] != db[..ns]) as usize;
        Ok((d, ea, cnt_b))
    };
    let w_rand = synth_x(ne, kd, 0x5A3);
    let mut w_tie = w_rand.clone();          // expert e = expert e % 37: 13-14 bit-identical logits each
    for e in 37..ne { let (src, dst) = ((e % 37) * kd, e * kd); w_tie.copy_within(src..src + kd, dst); }
    let (a, b, c, e4) = (mk()?, mk()?, mk()?, mk()?);
    let mut fails = 0usize;
    for (case, wv) in [("rand", &w_rand), ("ties", &w_tie), ("nan-row", &w_rand)] {
        let d_w = dev.htod_sync_copy(wv)?;
        for m in 1..=mmax {
            let mut xv = synth_x(m, kd, 0xB0 ^ m as u32);
            if case == "nan-row" { for v in &mut xv[(m - 1) * kd..] { *v = 0x7E00; } }
            let d_x = dev.htod_sync_copy(&xv)?;
            let (wp, xp) = (dp(d_w.device_ptr()), dp(d_x.device_ptr()));
            pair(&a, wp, xp, m)?;
            for rep in 0..2 {
                fold(f_fold, &b, wp, xp, m)?;
                fold(f_fold0, &c, wp, xp, m)?;
                fold(f_w4, &e4, wp, xp, m)?;
                let (d1, es, c1) = cmp(&a, &b, m)?;
                let (d2, _, c2) = cmp(&a, &c, m)?;
                let (d3, _, c3) = cmp(&a, &e4, m)?;
                if es <= 0 || d1 + d2 + d3 > 0 || c1 != 0 || c2 != 0 || c3 != 0 {
                    fails += 1;
                    println!("    ROUTERFOLD FAIL {case} m={m} rep={rep}: fold {d1} diffs, fold_m0 {d2} diffs, \
                              fold_w4 {d3} diffs, esel {es}, counters {c1}/{c2}/{c3} (must be 0)");
                }
            }
        }
    }
    if fails > 0 { bail!("EXL3-ROUTERFOLD FAIL ({fails} cases)"); }
    println!("EXL3-ROUTERFOLD: PASS (xq_router_fold / _m0 / W4S _w4 (grid {g_w4} x {}) bitwise == xq_router_fused + \
              xq_router_topk_route: logits/ids/wts/route tables, M 1..=8, rand/ties/nan-row, x2 launches, counters re-armed)",
             crate::exl3_forward::W4S_ROUTER_NT);
    // ---- A5-K2: xq_router_fold_c (moe.router_coal) bitwise == xq_router_fold_w4 (the served fold) AND the
    // S-A3-z pair, every output NaN/garbage-poisoned before each launch, M 1..=8 (every width the fold
    // serves: m*topk <= 128, m <= 8; wider verifies take the unfused path), five data classes:
    // rand, ties37 (cross-lane ties), nan-row, ties32 (expert e = e % 32: every lane's 16 probs tie —
    // the lane-tree tie order), peaked (x * 300: most probs underflow to +0, top-k picks +0 by id asc).
    {
        let poison = |o: &Out| -> Result<()> {
            use cudarc::driver::sys;
            let fill = |p: u64, n: usize, v: u32| -> Result<()> {
                let r = unsafe { sys::cuMemsetD32_v2(p, v, n) };
                anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "poison memset ({r:?})");
                Ok(())
            };
            fill(dp(o.lg.device_ptr()), mmax * ne, 0x7FC0_0001)?;
            fill(dp(o.ids.device_ptr()), mmax * k, 0xA5A5_A5A5)?;
            fill(dp(o.wts.device_ptr()), mmax * k, 0x7FC0_0002)?;
            fill(dp(o.sm.device_ptr()), ne, 0xA5A5_A5A5)?;
            fill(dp(o.ix.device_ptr()), mmax * k, 0xA5A5_A5A5)?;
            fill(dp(o.es.device_ptr()), 1, 0xA5A5_A5A5)?;
            fill(dp(o.og.device_ptr()), mmax * k * 2, 0xA5A5_A5A5)?;
            fill(dp(o.od.device_ptr()), mmax * k * 2, 0xA5A5_A5A5)?;
            dev.synchronize()?;
            Ok(())
        };
        let mut w_t32 = w_rand.clone();
        for e in 32..ne { let (src, dst) = ((e % 32) * kd, e * kd); w_t32.copy_within(src..src + kd, dst); }
        let (r4, cc) = (mk()?, mk()?);
        let mut fails = 0usize;
        let mut ncase = 0usize;
        for (case, wv) in [("rand", &w_rand), ("ties37", &w_tie), ("nan-row", &w_rand), ("ties32", &w_t32), ("peaked", &w_rand)] {
            let d_w = dev.htod_sync_copy(wv)?;
            for m in 1..=mmax {
                let mut xv = synth_x(m, kd, 0xD0 ^ m as u32);
                if case == "nan-row" { for v in &mut xv[(m - 1) * kd..] { *v = 0x7E00; } }
                if case == "peaked" {
                    for v in &mut xv { *v = f16::from_f32(f16::from_bits(*v).to_f32() * 300.0).to_bits(); }
                }
                let d_x = dev.htod_sync_copy(&xv)?;
                let (wp, xp) = (dp(d_w.device_ptr()), dp(d_x.device_ptr()));
                pair(&a, wp, xp, m)?;
                poison(&r4)?;
                fold(f_w4, &r4, wp, xp, m)?;
                for rep in 0..2 {
                    poison(&cc)?;
                    fold(f_c, &cc, wp, xp, m)?;
                    let (d1, es, c1) = cmp(&r4, &cc, m)?;
                    let (d2, _, _) = cmp(&a, &cc, m)?;
                    ncase += 1;
                    if es <= 0 || d1 + d2 > 0 || c1 != 0 {
                        fails += 1;
                        println!("    ROUTERFOLD-C FAIL {case} m={m} rep={rep}: vs fold_w4 {d1} diffs, vs pair {d2} diffs, \
                                  esel {es}, counter {c1} (must be 0)");
                    }
                }
            }
        }
        if fails > 0 { bail!("EXL3-ROUTERFOLD-C FAIL ({fails}/{ncase} cases)"); }
        println!("EXL3-ROUTERFOLD-C: PASS (A5-K2 xq_router_fold_c (grid {} x 256, smem {}) bitwise == xq_router_fold_w4 == \
                  xq_router_fused + xq_router_topk_route: logits/ids/wts/slotmap/esel/idxmap/offs, M 1..=8, \
                  rand/ties37/nan-row/ties32/peaked, NaN-poisoned outputs, x2 launches, counter re-armed; {ncase} cases)",
                 ne / 4, crate::exl3_forward::RFC_SMEM);
    }
    // ---- standalone timing: weights rotated over 16 copies (42 MB > L2) so they stream from DRAM ----
    let nw = 16usize;
    let wbig: Vec<u16> = (0..nw).flat_map(|i| synth_x(ne, kd, 0x100 + i as u32)).collect();
    let d_wb = dev.htod_sync_copy(&wbig)?;
    let wb0 = dp(d_wb.device_ptr());
    let wstride = (ne * kd * 2) as u64;
    for m in [1usize, 2, 4, 6, 8] {
        let d_x = dev.htod_sync_copy(&synth_x(m, kd, 0xC0 ^ m as u32))?;
        let xp = dp(d_x.device_ptr());
        let mut us = [0f64; 5];
        for (which, slot) in us.iter_mut().enumerate() {
            let run = |it: usize| -> Result<()> {
                let wp = wb0 + (it % nw) as u64 * wstride;
                match which { 0 => pair(&a, wp, xp, m), 1 => fold(f_fold, &b, wp, xp, m), 2 => fold(f_fold0, &c, wp, xp, m),
                              3 => fold(f_w4, &e4, wp, xp, m), _ => fold(f_c, &e4, wp, xp, m) }
            };
            for it in 0..nw { run(it)?; }
            dev.synchronize()?;
            let iters = 30 * nw;
            let t = std::time::Instant::now();
            for it in 0..iters { run(it)?; }
            dev.synchronize()?;
            *slot = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        }
        println!("    EXL3-ROUTERFOLD timing (standalone, harness-only) m={m}: pair {:.2} us/layer, fold {:.2} ({:+.2}), \
                  fold_m0 {:.2} ({:+.2}), W4S fold_w4 {:.2} ({:+.2} vs fold), K2 fold_c {:.2} ({:+.2} vs fold_w4)", us[0], us[1],
                 us[1] - us[0], us[2], us[2] - us[0], us[3], us[3] - us[1], us[4], us[4] - us[3]);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// WP09: xq_argmax_rows (one launch, nb x m blocks, last-block reduce) must return the SAME ids
// as m single-block xq_argmax launches AND the host fold (fp32 compare, lowest id on ties,
// init (-inf, 0), NaN skipped) — adversarial rows: unique max, cross-block ties, all -inf,
// all NaN, NaN-sprinkled, +-0 max ties, +inf ties, constant rows; aligned (V % 8 == 0) and
// misaligned (odd V -> scalar path) row strides; empty slices (tiny V); nb in {1,7,16,32};
// a start0 offset; two back-to-back launches on the same counters (the re-arm).
// ---------------------------------------------------------------------------
fn probe_argmax_rows(dev: &std::sync::Arc<CudaDevice>) -> Result<()> {
    let f_old = dev.get_func(MODULE, "xq_argmax").context("xq_argmax")?;
    let f_new = dev.get_func(MODULE, "xq_argmax_rows").context("xq_argmax_rows")?;
    let h = |x: f32| f16::from_f32(x).to_bits();
    let (ninf, pinf, nan) = (0xFC00u16, 0x7C00u16, 0x7E00u16);
    let host = |row: &[u16]| -> i32 {
        let (mut best, mut bi) = (f32::NEG_INFINITY, 0usize);
        for (i, &b) in row.iter().enumerate() {
            let v = f16::from_bits(b).to_f32();
            if v > best || (v == best && i < bi) { best = v; bi = i; }
        }
        bi as i32
    };
    let mut cases = 0usize;
    for &v in &[248_320usize, 248_323, 1000, 37] {
        let m = 16usize;
        let mut lg = synth_x(m, v, 0xA46 ^ v as u32);
        for r in 0..m {
            let row = &mut lg[r * v..(r + 1) * v];
            match r % 8 {
                0 => { row[(v * 7) / 11] = h(60000.0); }                        // unique max
                1 => { for &i in &[v - 1, v / 2, 3usize.min(v - 1), v / 3] { row[i] = h(100.0); } } // cross-block ties
                2 => { row.iter_mut().for_each(|b| *b = ninf); }                 // all -inf -> 0
                3 => { row.iter_mut().for_each(|b| *b = nan); }                  // all NaN -> 0
                4 => { for i in (0..v).step_by(3) { row[i] = nan; } row[v - 2] = h(50.0); row[v / 7] = nan; }
                // max = 0 with -0 at the LOWEST id: the canonical-+0 key must keep the id order
                5 => { row.iter_mut().for_each(|b| *b = h(-1.0)); row[v - 1] = 0x0000; row[(v / 2) | 1] = 0x0000; row[(v / 3) | 1] = 0x8000; }
                6 => { row[v / 4] = pinf; row[v - 3] = pinf; row[1usize.min(v - 1)] = nan; }
                _ => { row.iter_mut().for_each(|b| *b = if r == 7 { h(1.0) } else { ninf }); row[v - 1] = h(-65504.0); }
            }
        }
        let d_lg = dev.htod_sync_copy(&lg)?;
        // reference: m single-block xq_argmax launches
        let mut ids_old = dev.htod_sync_copy(&vec![-7i32; m])?;
        for i in 0..m {
            unsafe {
                f_old.clone().launch(LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 1024 * 12 },
                                     (&mut ids_old, i as i32, &d_lg, (i * v) as i64, v as i32))?;
            }
        }
        dev.synchronize()?;
        let old = dev.dtoh_sync_copy(&ids_old)?;
        for r in 0..m {
            let want = host(&lg[r * v..(r + 1) * v]);
            if old[r] != want { bail!("EXL3-ARGMAX-ROWS FAIL: reference xq_argmax row {r} V={v}: {} vs host {want}", old[r]); }
        }
        let mut part = dev.alloc_zeros::<u64>(m * 32)?;
        let mut cnt = dev.htod_sync_copy(&vec![0u32; m])?;
        for &nb in &[16usize, 32, 7, 1] {
            for rep in 0..2 {
                let mut ids = dev.htod_sync_copy(&vec![-9i32; m])?;
                unsafe {
                    f_new.clone().launch(LaunchConfig { grid_dim: (nb as u32, m as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                                         (&mut ids, 0i32, &d_lg, 0i64, v as i64, v as i32, &mut part, &mut cnt))?;
                }
                dev.synchronize()?;
                let got = dev.dtoh_sync_copy(&ids)?;
                if got != old {
                    let r = (0..m).find(|&r| got[r] != old[r]).unwrap_or(0);
                    bail!("EXL3-ARGMAX-ROWS FAIL: V={v} nb={nb} rep={rep} row {r}: {} vs xq_argmax {}", got[r], old[r]);
                }
                cases += 1;
            }
            let c = dev.dtoh_sync_copy(&cnt)?;
            if c.iter().any(|&x| x != 0) { bail!("EXL3-ARGMAX-ROWS FAIL: counters not re-armed (V={v} nb={nb}): {c:?}"); }
        }
        // start0 offset + lane0: rows 1..m into ids[1..m]
        {
            let mut ids = dev.htod_sync_copy(&vec![-9i32; m])?;
            unsafe {
                f_new.clone().launch(LaunchConfig { grid_dim: (16, (m - 1) as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                                     (&mut ids, 1i32, &d_lg, v as i64, v as i64, v as i32, &mut part, &mut cnt))?;
            }
            dev.synchronize()?;
            let got = dev.dtoh_sync_copy(&ids)?;
            if got[1..] != old[1..] { bail!("EXL3-ARGMAX-ROWS FAIL: V={v} start0/lane0 offset: {:?} vs {:?}", &got[1..], &old[1..]); }
            cases += 1;
        }
    }
    println!("EXL3-ARGMAX-ROWS: PASS (one-launch argmax ids == xq_argmax == host fold; {cases} launches x 16 adversarial rows, \
              V in {{248320, 248323, 1000, 37}}, nb in {{1,7,16,32}}, counters re-armed)");
    Ok(())
}

// ---------------------------------------------------------------------------
// S-A3-m item 5: the attention bit-contract (S-A3-l gate-coverage lesson — binv
// covered GEMM shapes only). For each verify width m and prefix length p0, the
// verify kernel `xq_attn_decode_group` (v2 warp-tree scan) must be BITWISE equal,
// in every row's output AND the KV cache it writes, to (a) the pre-S-A3-l
// per-row loop over xq_attn_softmax (`xq_attn_decode_group_ref`, the smem
// sync-tree class) and (b) m sequential single-row `xq_attn_decode` launches
// (what plain decode computes for those tokens). Synthetic Qwen3.8-Flash-Next
// shapes: 24 q-heads / 2 kv-heads, hd 256, partial rope 64.
// WP13: (c) the dense v3 chain (xq_attn_dense_prep -> _dots -> _acc4 AND _acc1, both
// accumulate variants at every width) must equal v2 bitwise too — attn rows + KV cache,
// in all four KV formats (f32 / f16 / fp8 / q8 — q8 = the p2 default: rotated staged q,
// in-kernel output un-rotation), prefixes up to the dense cap (p0 2036 + m 16 -> positions
// 0..=2051). (a)/(b) stay fatal for f32 (their original contract) and are reported for
// f16/fp8/q8. Then a small v2-vs-v3 timing table (synthetic, f32 and q8).
// ---------------------------------------------------------------------------
fn probe_attn(dev: &std::sync::Arc<CudaDevice>) -> Result<()> {
    const NH: usize = 24;
    const NKV: usize = 2;
    const HD: usize = 256;
    const RDIM: usize = 64;
    const MAXP: usize = 2304;   // holds the whole dense regime (positions 0..=2051)
    const TS: usize = 2304;     // XQ_DV3_TS
    const NCH: usize = 36;      // XQ_DV3_NCH
    const MMAX: usize = 16;
    let f_v2 = dev.get_func(MODULE, "xq_attn_decode_group").context("xq_attn_decode_group")?;
    let f_ref = dev.get_func(MODULE, "xq_attn_decode_group_ref").context("xq_attn_decode_group_ref")?;
    let f_dec = dev.get_func(MODULE, "xq_attn_decode").context("xq_attn_decode")?;
    let f_prep = dev.get_func(MODULE, "xq_attn_dense_prep").context("xq_attn_dense_prep")?;
    let f_dots = dev.get_func(MODULE, "xq_attn_dense_dots").context("xq_attn_dense_dots")?;
    let f_acc4 = dev.get_func(MODULE, "xq_attn_dense_acc4").context("xq_attn_dense_acc4")?;
    let f_acc1 = dev.get_func(MODULE, "xq_attn_dense_acc1").context("xq_attn_dense_acc1")?;
    let mut rng = 0x5eed_1234u32;
    let mut rnd = move |amp: f32| -> f32 {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        (((rng >> 8) % 20001) as f32 / 10000.0 - 1.0) * amp
    };
    let nh_p = (NH | (NKV << 16)) as i32;
    let hd_p = (HD | (RDIM << 16)) as i32;
    // v3 scratch (qstage, score plane, chunk maxima) sized for the widest width
    let d_qs = dev.htod_sync_copy(&vec![0f32; MMAX * NH * HD])?;
    let d_s = dev.htod_sync_copy(&vec![0f32; MMAX * NH * TS])?;
    let d_pm = dev.htod_sync_copy(&vec![0f32; MMAX * NH * NCH])?;
    let (pqs, ps, ppm) = (*d_qs.device_ptr() as u64, *d_s.device_ptr() as u64, *d_pm.device_ptr() as u64);
    // the host's launch sequence (FwdModel::dense_attn_v3); acc4 = true -> 64-dim slices
    // W4/SMALL: the 5-stage-ring accumulate twins (bitwise to acc4 / acc1) + the draft-pass chain
    let f_acc4w = dev.get_func(MODULE, "xq_attn_dense_acc4_w4").context("xq_attn_dense_acc4_w4")?;
    let f_acc1w = dev.get_func(MODULE, "xq_attn_dense_acc1_w4").context("xq_attn_dense_acc1_w4")?;
    let f_aprep = dev.get_func(MODULE, "xq_attn_prep").context("xq_attn_prep")?;
    let f_sk4 = dev.get_func(MODULE, "xq_attn_dense_splitk4").context("xq_attn_dense_splitk4")?;
    let f_comb = dev.get_func(MODULE, "xq_attn_sel_combine").context("xq_attn_sel_combine")?;
    let f_sk4w = dev.get_func(MODULE, "xq_attn_dense_splitk4_w4").context("xq_attn_dense_splitk4_w4")?;
    let f_combw = dev.get_func(MODULE, "xq_attn_sel_combine_w4").context("xq_attn_sel_combine_w4")?;
    const NSPL: usize = 64;   // DRAFT_ATTN_SPLITS
    let d_dpm = dev.htod_sync_copy(&vec![0f32; MMAX * NH * NSPL])?;
    let d_dpl = dev.htod_sync_copy(&vec![0f32; MMAX * NH * NSPL])?;
    let d_dpa = dev.htod_sync_copy(&vec![0f32; MMAX * NH * NSPL * HD])?;
    let (pdm, pdl, pda) = (*d_dpm.device_ptr() as u64, *d_dpl.device_ptr() as u64, *d_dpa.device_ptr() as u64);
    // the draft head's chain (FwdModel::head_launches): w4 = dense_prep + splitk4_w4 + combine_w4
    let run_draft = |w4: bool, m: usize, pa: u64, pq: u64, pk: u64, pv: u64, pc: u64, pn: u64,
                     pcs: u64, psp: u64, mpf: i32| -> Result<()> {
        let lc = |g: u32, b: u32| LaunchConfig { grid_dim: (g, 1, 1), block_dim: (b, 1, 1), shared_mem_bytes: 0 };
        let geom: i64 = (NSPL as i64) | (((mpf as u32) as i64) << 32);
        let rows_kvf = (m as i32) | (((mpf as u32 >> 28) as i32) << 28);
        unsafe {
            if w4 {
                f_prep.clone().launch(lc((m * NH) as u32, HD as u32),
                    (pqs, pc, pq, pk, pv, pn, pcs, psp, nh_p, hd_p, mpf, 1e-6f32))?;
                f_sk4w.clone().launch(lc((m * NKV * NSPL) as u32, 256), (pdm, pdl, pda, pqs, pc, psp, nh_p, geom))?;
                f_combw.clone().launch(lc((m * NH) as u32, HD as u32), (pa, pdm, pdl, pda, pq, nh_p, NSPL as i32, rows_kvf))?;
            } else {
                f_aprep.clone().launch(lc((m * NKV) as u32, HD as u32),
                    (pqs, pc, pq, pk, pv, pn, pcs, psp, nh_p, hd_p, mpf, 1e-6f32))?;
                f_sk4.clone().launch(lc((m * NKV * NSPL) as u32, 256), (pdm, pdl, pda, pqs, pc, psp, nh_p, geom))?;
                f_comb.clone().launch(lc((m * NH) as u32, HD as u32), (pa, pdm, pdl, pda, pq, nh_p, NSPL as i32, rows_kvf))?;
            }
        }
        Ok(())
    };
    let mut draft_checked = 0usize;
    let run_v3 = |w4: bool, acc4: bool, m: usize, pa: u64, pq: u64, pk: u64, pv: u64, pc: u64, pn: u64,
                  pcs: u64, psp: u64, mpf: i32| -> Result<()> {
        let lc = |g: (u32, u32, u32), b: u32| LaunchConfig { grid_dim: g, block_dim: (b, 1, 1), shared_mem_bytes: 0 };
        let nchg = (96 / (NKV * m)).clamp(1, NCH) as u32;
        let (fa, sd) = match (w4, acc4) {
            (false, true) => (&f_acc4, 64usize), (false, false) => (&f_acc1, 16usize),
            (true, true) => (&f_acc4w, 64usize), (true, false) => (&f_acc1w, 16usize),
        };
        unsafe {
            f_prep.clone().launch(lc(((m * NH) as u32, 1, 1), HD as u32),
                (pqs, pc, pq, pk, pv, pn, pcs, psp, nh_p, hd_p, mpf, 1e-6f32))?;
            f_dots.clone().launch(lc((nchg, NKV as u32, m as u32), 256),
                (ps, ppm, pqs, pc, psp, nh_p, hd_p, mpf))?;
            fa.clone().launch(lc(((HD / sd) as u32, NKV as u32, m as u32), 192),
                (pa, ps, ppm, pc, pq, psp, nh_p, hd_p, mpf))?;
        }
        Ok(())
    };
    let mut checked = 0usize;
    for fmt in 0..4usize {
        let fname = ["f32", "f16", "fp8", "q8"][fmt];
        // q8 (WP25): hd codes + hd/32 f16 group scales padded to 16 B (= HD + 16 at hd 256)
        let rb = match fmt { 0 => 4 * HD, 1 => 2 * HD, _ => HD + 16 };
        let mpf = (MAXP as i32) | ((fmt as i32) << 28);
        let cache_bytes = 2 * NKV * MAXP * rb;
        for &p0 in &[0usize, 5, 63, 300, 1100, 2036] {
            // one prefix cache shared by every width at this p0 (positions < p0)
            let mut cache0 = vec![0u8; cache_bytes];
            for kv in 0..2 {
                for h in 0..NKV {
                    for t in 0..p0 {
                        let base = ((kv * NKV + h) * MAXP + t) * rb;
                        let amp = if kv == 0 { 1.5 } else { 1.0 };
                        match fmt {
                            0 => for d in 0..HD {
                                cache0[base + 4 * d..base + 4 * d + 4].copy_from_slice(&rnd(amp).to_le_bytes());
                            },
                            1 => for d in 0..HD {
                                cache0[base + 2 * d..base + 2 * d + 2]
                                    .copy_from_slice(&f16::from_f32(rnd(amp)).to_bits().to_le_bytes());
                            },
                            2 => {
                                for d in 0..HD {
                                    let mut b = ((rnd(1.0) + 1.0) * 127.9) as u8;
                                    if b & 0x7F == 0x7F { b ^= 1; }   // no e4m3 NaN codes
                                    cache0[base + d] = b;
                                }
                                let scl = 0.004f32 + (rnd(1.0) + 1.0) * 0.008;
                                cache0[base + HD..base + HD + 4].copy_from_slice(&scl.to_le_bytes());
                            }
                            _ => {
                                // q8: any code byte is valid; one positive f16 scale per 32-group
                                for d in 0..HD {
                                    cache0[base + d] = ((rnd(1.0) + 1.0) * 127.9) as u8;
                                }
                                for g in 0..HD / 32 {
                                    let scl = f16::from_f32(amp * (0.25 + (rnd(1.0) + 1.0) * 0.5));
                                    cache0[base + HD + 2 * g..base + HD + 2 * g + 2].copy_from_slice(&scl.to_bits().to_le_bytes());
                                }
                            }
                        }
                    }
                }
            }
            for &m in &[1usize, 2, 3, 4, 5, 6, 7, 8, 12, 16] { // WP23: + 7 (depth-6 verify width)
                let qg: Vec<u16> = (0..m * NH * HD * 2).map(|_| f16::from_f32(rnd(2.0)).to_bits()).collect();
                let kp: Vec<u16> = (0..m * NKV * HD).map(|_| f16::from_f32(rnd(2.0)).to_bits()).collect();
                let vp: Vec<u16> = (0..m * NKV * HD).map(|_| f16::from_f32(rnd(1.0)).to_bits()).collect();
                let qnkn: Vec<u16> = (0..2 * HD).map(|_| f16::from_f32(rnd(0.3)).to_bits()).collect();
                let mut cs = vec![0f32; m * 2 * RDIM];
                for r in 0..m {
                    for i in 0..RDIM {
                        let a = (p0 + r) as f32 * 0.01 * (i as f32 + 1.0);
                        cs[r * 2 * RDIM + i] = a.cos();
                        cs[r * 2 * RDIM + RDIM + i] = a.sin();
                    }
                }
                let sp: Vec<i32> = (0..m).flat_map(|r| [0i32, (p0 + r) as i32]).collect();
                let d_qg = dev.htod_sync_copy(&qg)?;
                let d_kp = dev.htod_sync_copy(&kp)?;
                let d_vp = dev.htod_sync_copy(&vp)?;
                let d_qn = dev.htod_sync_copy(&qnkn)?;
                let d_cs = dev.htod_sync_copy(&cs)?;
                let d_sp = dev.htod_sync_copy(&sp)?;
                let mut outs: Vec<(Vec<u16>, Vec<u8>)> = Vec::new();
                for which in 0..9 {
                    let d_cache = dev.htod_sync_copy(&cache0)?;
                    let d_attn = dev.htod_sync_copy(&vec![0u16; m * NH * HD])?;
                    let pa = *d_attn.device_ptr() as u64;
                    let pq = *d_qg.device_ptr() as u64;
                    let pk = *d_kp.device_ptr() as u64;
                    let pv = *d_vp.device_ptr() as u64;
                    let pc = *d_cache.device_ptr() as u64;
                    let pn = *d_qn.device_ptr() as u64;
                    let pcs = *d_cs.device_ptr() as u64;
                    let psp = *d_sp.device_ptr() as u64;
                    let cfg = |gx: u32, gy: u32| LaunchConfig {
                        grid_dim: (gx, gy, 1),
                        block_dim: (HD as u32, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe {
                        match which {
                            0 => f_v2.clone().launch(cfg(NH as u32, m as u32),
                                (pa, pq, pk, pv, pc, pn, pcs, psp, nh_p, hd_p, mpf, 1e-6f32))?,
                            1 => f_ref.clone().launch(cfg(NH as u32, m as u32),
                                (pa, pq, pk, pv, pc, pn, pcs, psp, nh_p, hd_p, mpf, 1e-6f32))?,
                            2 => {
                                // plain decode, one token at a time (row r sees rows < r written)
                                for r in 0..m {
                                    f_dec.clone().launch(cfg(NH as u32, 1),
                                        (pa + (r * NH * HD * 2) as u64,
                                         pq + (r * NH * HD * 2 * 2) as u64,
                                         pk + (r * NKV * HD * 2) as u64,
                                         pv + (r * NKV * HD * 2) as u64,
                                         pc, pn,
                                         pcs + (r * 2 * RDIM * 4) as u64,
                                         psp + (r * 2 * 4) as u64,
                                         nh_p, hd_p, mpf, 1e-6f32))?;
                                }
                            }
                            // WP13: the host's rule (m == 1 -> acc1, else acc4), then the other variant
                            3 => run_v3(false, m != 1, m, pa, pq, pk, pv, pc, pn, pcs, psp, mpf)?,
                            4 => run_v3(false, m == 1, m, pa, pq, pk, pv, pc, pn, pcs, psp, mpf)?,
                            // W4/SMALL: the ring twins, host rule then the other variant
                            5 => run_v3(true, m != 1, m, pa, pq, pk, pv, pc, pn, pcs, psp, mpf)?,
                            6 => run_v3(true, m == 1, m, pa, pq, pk, pv, pc, pn, pcs, psp, mpf)?,
                            // W4/SMALL draft chain: old (7) vs new (8), compared with each other
                            7 => run_draft(false, m, pa, pq, pk, pv, pc, pn, pcs, psp, mpf)?,
                            _ => run_draft(true, m, pa, pq, pk, pv, pc, pn, pcs, psp, mpf)?,
                        }
                    }
                    dev.synchronize()?;
                    outs.push((dev.dtoh_sync_copy(&d_attn)?, dev.dtoh_sync_copy(&d_cache)?));
                }
                // success signal: finite, nonzero outputs
                let nz = outs[0].0.iter().filter(|b| { let v = f16::from_bits(**b).to_f32(); v != 0.0 && v.is_finite() }).count();
                if nz < m * NH * HD / 2 {
                    bail!("EXL3-ATTN FAIL: implausible v2 output at fmt={fname} m={m} p0={p0} ({nz} finite nonzero of {})", m * NH * HD);
                }
                for (name, i) in [("ref-loop", 1usize), ("seq-decode", 2), ("v3-hostrule", 3), ("v3-other-acc", 4),
                                  ("W4S-v3-hostrule", 5), ("W4S-v3-other-acc", 6)] {
                    let am = outs[0].0.iter().zip(outs[i].0.iter()).filter(|(a, b)| a != b).count();
                    let cm = outs[0].1.iter().zip(outs[i].1.iter()).filter(|(a, b)| a != b).count();
                    if am + cm > 0 {
                        let first = outs[0].0.iter().zip(outs[i].0.iter()).position(|(a, b)| a != b);
                        let msg = format!("fmt={fname} m={m} p0={p0} vs {name}: {am} attn / {cm} cache-byte mismatches (first attn idx {first:?} = row {:?})",
                                          first.map(|f| f / (NH * HD)));
                        if i >= 3 || fmt == 0 { bail!("EXL3-ATTN FAIL {msg}"); }
                        println!("    EXL3-ATTN note (not gated for {fname}): {msg}");
                    }
                }
                // W4/SMALL draft chain: new (8) vs old (7) — attn rows + KV cache (its own class: the
                // split-K online softmax is not v2's two-pass scan, so it is gated against itself)
                {
                    let nzd = outs[7].0.iter().filter(|b| { let v = f16::from_bits(**b).to_f32(); v != 0.0 && v.is_finite() }).count();
                    if nzd < m * NH * HD / 2 {
                        bail!("EXL3-ATTN FAIL: implausible draft-chain output at fmt={fname} m={m} p0={p0} ({nzd} finite nonzero)");
                    }
                    let am = outs[7].0.iter().zip(outs[8].0.iter()).filter(|(a, b)| a != b).count();
                    let cm = outs[7].1.iter().zip(outs[8].1.iter()).filter(|(a, b)| a != b).count();
                    if am + cm > 0 {
                        bail!("EXL3-ATTN FAIL fmt={fname} m={m} p0={p0}: W4S draft chain (dense_prep + splitk4_w4 + combine_w4) \
                               vs (xq_attn_prep + splitk4 + combine): {am} attn / {cm} cache-byte mismatches");
                    }
                    draft_checked += 1;
                }
                checked += 1;
            }
            println!("    attn {fname} p0={p0}: widths {{1..8,12,16}} v2 == v3 (acc4 + acc1) == W4S acc4_w4/acc1_w4{} \
                      (attn rows + KV cache); W4S draft chain == old draft chain",
                     if fmt == 0 { " == ref-loop == seq-decode" } else { "" });
        }
    }
    println!("EXL3-ATTN: PASS ({checked} cells: v2 warp-tree scan bitwise vs xq_attn_softmax class; \
              WP13 dense v3 bitwise vs v2 in f32/f16/fp8/q8; W4S dense acc4_w4/acc1_w4 bitwise vs v2; \
              W4S draft chain bitwise vs the old draft chain in {draft_checked} cells)");

    // ---- WP13 timing (synthetic, f32 + q8, warm, 50 launches each; K/V rewrites are identical words)
    for &(fmt, m, p0) in &[(0usize, 1usize, 300usize), (0, 1, 2036), (0, 6, 300), (0, 6, 1100), (0, 6, 2036),
                           (3, 1, 2036), (3, 6, 2036)] {
        let rb = if fmt == 0 { 4 * HD } else { HD + 16 };
        let d_cache = dev.htod_sync_copy(&vec![0u8; 2 * NKV * MAXP * rb])?;
        let d_attn = dev.htod_sync_copy(&vec![0u16; m * NH * HD])?;
        let qg: Vec<u16> = (0..m * NH * HD * 2).map(|_| f16::from_f32(rnd(2.0)).to_bits()).collect();
        let kvp: Vec<u16> = (0..m * NKV * HD).map(|_| f16::from_f32(rnd(1.0)).to_bits()).collect();
        let qnkn: Vec<u16> = (0..2 * HD).map(|_| f16::from_f32(rnd(0.3)).to_bits()).collect();
        let cs = vec![0.5f32; m * 2 * RDIM];
        let sp: Vec<i32> = (0..m).flat_map(|r| [0i32, (p0 + r) as i32]).collect();
        let (d_qg, d_kv, d_qn, d_cs, d_sp) = (dev.htod_sync_copy(&qg)?, dev.htod_sync_copy(&kvp)?,
            dev.htod_sync_copy(&qnkn)?, dev.htod_sync_copy(&cs)?, dev.htod_sync_copy(&sp)?);
        let (pa, pq, pkv, pc, pn, pcs, psp) = (*d_attn.device_ptr() as u64, *d_qg.device_ptr() as u64,
            *d_kv.device_ptr() as u64, *d_cache.device_ptr() as u64, *d_qn.device_ptr() as u64,
            *d_cs.device_ptr() as u64, *d_sp.device_ptr() as u64);
        let mpf = (MAXP as i32) | ((fmt as i32) << 28);
        let cfg = |gx: u32, gy: u32| LaunchConfig { grid_dim: (gx, gy, 1), block_dim: (HD as u32, 1, 1), shared_mem_bytes: 0 };
        let mut t_old = 0f64;
        let mut t_new = 0f64;
        let mut t_w4 = 0f64;
        for pass in 0..2 {
            for which in 0..3 {
                dev.synchronize()?;
                let t0 = std::time::Instant::now();
                for _ in 0..50 {
                    if which == 0 {
                        unsafe {
                            if m == 1 {
                                f_dec.clone().launch(cfg((m * NH) as u32, 1),
                                    (pa, pq, pkv, pkv, pc, pn, pcs, psp, nh_p, hd_p, mpf, 1e-6f32))?;
                            } else {
                                f_v2.clone().launch(cfg(NH as u32, m as u32),
                                    (pa, pq, pkv, pkv, pc, pn, pcs, psp, nh_p, hd_p, mpf, 1e-6f32))?;
                            }
                        }
                    } else {
                        run_v3(which == 2, m != 1, m, pa, pq, pkv, pkv, pc, pn, pcs, psp, mpf)?;
                    }
                }
                dev.synchronize()?;
                let us = t0.elapsed().as_secs_f64() * 1e6 / 50.0;
                if pass == 1 { match which { 0 => t_old = us, 1 => t_new = us, _ => t_w4 = us } }
            }
        }
        println!("    EXL3-ATTN-V3 timing {} m={m} p0={p0}: old {} {t_old:.1} us, v3 {t_new:.1} us, W4S v3+acc_w4 {t_w4:.1} us \
                  per layer call ({:.1}x / {:.1}x; synthetic, launch-inclusive, harness-only)",
                 ["f32", "f16", "fp8", "q8"][fmt], if m == 1 { "xq_attn_decode" } else { "xq_attn_decode_group" },
                 t_old / t_new.max(1e-9), t_old / t_w4.max(1e-9));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// W4/QSA: the SPARSE-regime attention bit-contract (probe_attn above covers the dense regime).
// For each KV format, context (~8K / 32K / 128K), selection mode and width m in {1..8, 12, 16},
// the row-shared union gather `xq_attn_sel_union` must write pm / pl / pacc BITWISE equal to the
// old served gathers — hg3 (the m >= 2 kernel it replaces), hg1 (m = 1) and splitk4 (the class
// the hg family was proven against) — and the unchanged xq_attn_sel_combine must produce the same
// attention rows from either. Synthetic Qwen3.8-Flash-Next shapes: 24 q-heads / 2 kv-heads, hd
// 256, 32 decode splits over the 2,051-entry selection pitch (split_len 65), ratio 4, top-512
// blocks; KV planes written on device through the served writer (xq_qu_synth_kv) at EVERY
// position (distinct content, so an off-by-position read cannot pass). Modes: overlap (the served
// shape: consecutive positions = one base top-512 incl. the local window + per-row block swaps +
// each row's newer blocks and tail), identical, disjoint (independent top-512 per row: the union
// approaches m x the row), two-slot (rows alternate between 2 slots: entries never shared across
// slots), fallback (a swapped pair, a duplicate and a -1: the kernel's exact verbatim path).
// Then a timing table (synthetic, isolated: RANKS the kernels; in-round cost is the ledger's job).
// ---------------------------------------------------------------------------
fn probe_attn_sparse(dev: &std::sync::Arc<CudaDevice>) -> Result<()> {
    use crate::exl3_forward::{qsa_union_fits, kv_rowbytes};
    const NH: usize = 24;
    const NKV: usize = 2;
    const HD: usize = 256;
    const NSD: usize = 32;
    const RATIO: usize = 4;
    const TOPK: usize = 512;
    const SEL_MAX: usize = 2048 + RATIO - 1;
    const LOCAL: usize = 64;                     // recent blocks every row keeps (the local window)
    let l = SEL_MAX.div_ceil(NSD);
    let fget = |n: &str| dev.get_func(MODULE, n).with_context(|| format!("{n} (stale PTX?)"));
    let (f_hg3, f_hg1, f_sk4, f_un, f_comb, f_syn) = (fget("xq_attn_sel_hg3")?, fget("xq_attn_sel_hg1")?,
        fget("xq_attn_sel_splitk4")?, fget("xq_attn_sel_union")?, fget("xq_attn_sel_combine")?, fget("xq_qu_synth_kv")?);
    let nh_p = (NH | (NKV << 16)) as i32;
    let split_hd = (l | (HD << 16)) as i32;
    let mut rs = 0x51A5_0001_2345_6789u64;
    let mut rnd = move || -> u64 { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; rs };
    // top-k block sets -> ascending token lists (block j = tokens 4j..4j+3, then the row's tail)
    let emit = |blocks: &mut Vec<usize>, pos: usize| -> Vec<i32> {
        blocks.sort_unstable();
        let nb = (pos + 1) / RATIO;
        let mut v: Vec<i32> = blocks.iter().flat_map(|&j| (0..RATIO).map(move |i| (j * RATIO + i) as i32)).collect();
        v.extend((nb * RATIO..=pos).map(|t| t as i32));
        v
    };
    let (mut cells, mut fb_cells) = (0usize, 0usize);
    let mut union_stat: Vec<String> = Vec::new();
    let widths = [1usize, 2, 3, 4, 5, 6, 7, 8, 12, 16];
    for &(ctx, fmts) in &[(8192usize, &[0usize, 1, 2, 3][..]), (32768, &[0, 1, 2, 3][..]), (131072, &[0, 3][..])] {
        let modes: &[&str] = if ctx == 8192 { &["overlap", "identical", "disjoint", "two-slot", "fallback"] }
                             else { &["overlap", "identical", "disjoint"] };
        let nslot = if ctx == 8192 { 2usize } else { 1 };
        let maxp = ctx + 64;
        let p0 = ctx - 16;                       // rows at p0..p0+m-1 (QSA regime: > 2,051)
        for &fmt in fmts {
            let fname = ["f32", "f16", "fp8", "q8"][fmt];
            let rb = kv_rowbytes(fmt as u32, HD);
            let mpf = (maxp as i32) | ((fmt as i32) << 28);
            let d_cache = dev.alloc_zeros::<u8>(nslot * 2 * NKV * maxp * rb)?;
            let pc = *d_cache.device_ptr() as u64;
            unsafe {
                f_syn.clone().launch(LaunchConfig { grid_dim: (maxp as u32, (nslot * 2 * NKV) as u32, 1),
                                                    block_dim: (HD as u32, 1, 1), shared_mem_bytes: 0 },
                                     (pc, mpf, HD as i32, NKV as i32, 0i32, 0xC0DE_0000u32 ^ (ctx as u32) ^ ((fmt as u32) << 20)))?;
            }
            dev.synchronize()?;
            for &mode in modes {
                for &m in &widths {
                    // ---- selection lists
                    let nb0 = (p0 + 1) / RATIO;
                    let mut base: Vec<usize> = (0..4).chain(nb0 - LOCAL..nb0).collect();
                    let mut inb = vec![false; nb0 + 16];
                    for &j in &base { inb[j] = true; }
                    while base.len() < TOPK {
                        let j = 4 + (rnd() as usize) % (nb0 - LOCAL - 4);
                        if !inb[j] { inb[j] = true; base.push(j); }
                    }
                    let mut sel = vec![-1i32; m * SEL_MAX];
                    let mut psel = vec![0i32; m];
                    let mut sp = vec![0i32; 2 * m];
                    let mut rows: Vec<Vec<i32>> = Vec::new();
                    for r in 0..m {
                        let pos = p0 + r;
                        let nb = (pos + 1) / RATIO;
                        let mut blocks: Vec<usize> = if mode == "disjoint" {
                            let mut s = vec![false; nb];
                            let mut b = Vec::new();
                            while b.len() < TOPK { let j = (rnd() as usize) % nb; if !s[j] { s[j] = true; b.push(j); } }
                            b
                        } else {
                            let mut b = base.clone();
                            let mut isb = inb.clone();
                            // this row's newer complete blocks join its local window
                            for j in nb0..nb { b.push(j); isb[j] = true; }
                            let swaps = if mode == "identical" { 0 } else { (rnd() % 9) as usize };
                            let mut extra = b.len() - TOPK + swaps;
                            if mode == "identical" {
                                // deterministic: every row drops the same (lowest droppable) blocks
                                b.sort_unstable();
                                b.retain(|&j| if extra > 0 && j >= 4 && j < nb0 - LOCAL { extra -= 1; false } else { true });
                            }
                            while extra > 0 {                                // drop non-local blocks
                                let i = (rnd() as usize) % b.len();
                                if b[i] >= 4 && b[i] < nb0 - LOCAL { isb[b[i]] = false; b.swap_remove(i); extra -= 1; }
                            }
                            while b.len() < TOPK {                           // add fresh ones
                                let j = 4 + (rnd() as usize) % (nb0 - LOCAL - 4);
                                if !isb[j] { isb[j] = true; b.push(j); }
                            }
                            b
                        };
                        let mut lst = emit(&mut blocks, pos);
                        if mode == "fallback" {
                            match r % 3 {
                                0 => lst.swap(100, 101),                     // unsorted pair
                                1 => lst[300] = lst[299],                    // duplicate
                                _ => lst[500] = -1,                          // a skipped entry
                            }
                        }
                        anyhow::ensure!(lst.len() <= SEL_MAX, "synthetic list {} > {SEL_MAX}", lst.len());
                        sel[r * SEL_MAX..r * SEL_MAX + lst.len()].copy_from_slice(&lst);
                        psel[r] = lst.len() as i32 - 1;
                        sp[2 * r] = if mode == "two-slot" { (r % 2) as i32 } else { 0 };
                        sp[2 * r + 1] = pos as i32;
                        rows.push(lst);
                    }
                    if mode == "overlap" && m == 8 {
                        // union entries per split vs one row's entries (what the kernel streams)
                        let (mut un, mut one) = (0usize, 0usize);
                        for s in 0..NSD {
                            let mut u: Vec<i32> = Vec::new();
                            for lst in &rows { u.extend(lst.iter().skip(s * l).take(l)); }
                            u.sort_unstable(); u.dedup();
                            un += u.len();
                            one += rows[0].iter().skip(s * l).take(l).count();
                        }
                        union_stat.push(format!("{fname}@{ctx}: {un} union entries / {one} per row ({:.2}x)", un as f64 / one as f64));
                    }
                    let qs: Vec<f32> = (0..m * NH * HD).map(|_| (rnd() % 40001) as f32 / 10000.0 - 2.0).collect();
                    let qg: Vec<u16> = (0..m * NH * HD * 2).map(|_| f16::from_f32((rnd() % 40001) as f32 / 10000.0 - 2.0).to_bits()).collect();
                    let (d_sel, d_ps, d_sp, d_qs, d_qg) = (dev.htod_sync_copy(&sel)?, dev.htod_sync_copy(&psel)?,
                        dev.htod_sync_copy(&sp)?, dev.htod_sync_copy(&qs)?, dev.htod_sync_copy(&qg)?);
                    let (psl, pps, psp, pqs, pqg) = (*d_sel.device_ptr() as u64, *d_ps.device_ptr() as u64,
                        *d_sp.device_ptr() as u64, *d_qs.device_ptr() as u64, *d_qg.device_ptr() as u64);
                    let np = m * NH * NSD;
                    let g2: u64 = (NSD as u64) | ((SEL_MAX as u64) << 20) | ((mpf as u32 as u64) << 32);
                    let poison = f32::from_bits(0x7FC0_DEAD);
                    // 0 union, 1 hg3, 2 hg1, 3 splitk4
                    let mut parts: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::new();
                    let mut bufs = Vec::new();
                    for k in 0..4 {
                        let (bm, bl, ba) = (dev.htod_sync_copy(&vec![poison; np])?, dev.htod_sync_copy(&vec![poison; np])?,
                                            dev.htod_sync_copy(&vec![poison; np * HD])?);
                        let (pm, pl, pa) = (*bm.device_ptr() as u64, *bl.device_ptr() as u64, *ba.device_ptr() as u64);
                        let grid = ((m * NKV * NSD) as u32, 1, 1);
                        unsafe {
                            match k {
                                0 => {
                                    let smem = qsa_union_fits(fmt as u32, NH, NKV, HD, m, l)
                                        .context("union gather contract rejects the served shape")?;
                                    f_un.clone().launch(LaunchConfig { grid_dim: ((NKV * NSD * (NH / NKV / 4)) as u32, 1, 1),
                                                                       block_dim: ((32 * m) as u32, 1, 1), shared_mem_bytes: smem as u32 },
                                        (pm, pl, pa, pqs, pc, psl, pps, psp, nh_p, g2 as i64, split_hd, m as i32))?
                                }
                                1 => f_hg3.clone().launch(LaunchConfig { grid_dim: grid, block_dim: (128, 1, 1), shared_mem_bytes: 0 },
                                        (pm, pl, pa, pqs, pc, psl, pps, psp, nh_p, g2 as i64, split_hd))?,
                                2 => f_hg1.clone().launch(LaunchConfig { grid_dim: grid, block_dim: (384, 1, 1), shared_mem_bytes: 0 },
                                        (pm, pl, pa, pqs, pc, psl, pps, psp, nh_p, g2 as i64, split_hd))?,
                                _ => f_sk4.clone().launch(LaunchConfig { grid_dim: grid, block_dim: (HD as u32, 1, 1), shared_mem_bytes: 0 },
                                        (pm, pl, pa, pqs, pc, psl, pps, psp, nh_p, g2 as i64, l as i32))?,
                            }
                        }
                        dev.synchronize()?;
                        parts.push((dev.dtoh_sync_copy(&bm)?, dev.dtoh_sync_copy(&bl)?, dev.dtoh_sync_copy(&ba)?));
                        bufs.push((bm, bl, ba));
                    }
                    // success signal: the reference is real — no poison left, live splits in every row
                    let r0 = &parts[3];
                    let left = r0.0.iter().chain(&r0.1).chain(&r0.2).filter(|v| v.to_bits() == 0x7FC0_DEAD).count();
                    let live_rows = (0..m).filter(|&r| (0..NH * NSD).any(|i| r0.0[r * NH * NSD + i] > f32::NEG_INFINITY)).count();
                    let nzacc = r0.2.iter().filter(|v| **v != 0.0 && v.is_finite()).count();
                    if left > 0 || live_rows != m || nzacc < np * HD / 4 {
                        bail!("EXL3-ATTN-SPARSE FAIL: implausible splitk4 reference at {fname} ctx={ctx} {mode} m={m} \
                               (poison {left}, live rows {live_rows}/{m}, nonzero acc {nzacc}/{})", np * HD);
                    }
                    let bits = |x: &[f32], y: &[f32]| x.iter().zip(y).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
                    for (name, k) in [("union vs hg3", 1usize), ("union vs hg1", 2), ("union vs splitk4", 3)] {
                        let (dm, dl, da) = (bits(&parts[0].0, &parts[k].0), bits(&parts[0].1, &parts[k].1), bits(&parts[0].2, &parts[k].2));
                        if dm + dl + da > 0 {
                            let first = (0..np).find(|&i| parts[0].0[i].to_bits() != parts[k].0[i].to_bits()
                                                        || parts[0].1[i].to_bits() != parts[k].1[i].to_bits());
                            bail!("EXL3-ATTN-SPARSE FAIL {fname} ctx={ctx} {mode} m={m} {name}: pm {dm} pl {dl} pacc {da} bit-differ \
                                   of {np}/{} (first partial {first:?} = row {:?})", np * HD, first.map(|f| f / (NH * NSD)));
                        }
                    }
                    // the unchanged combine on the union partials == on hg3's (attention rows, f16 bits)
                    let mut outs = Vec::new();
                    for k in [0usize, 1] {
                        let d_attn = dev.htod_sync_copy(&vec![0x7E01u16; m * NH * HD])?;
                        let (pm, pl, pa) = (*bufs[k].0.device_ptr() as u64, *bufs[k].1.device_ptr() as u64, *bufs[k].2.device_ptr() as u64);
                        unsafe {
                            f_comb.clone().launch(LaunchConfig { grid_dim: ((m * NH) as u32, 1, 1), block_dim: (HD as u32, 1, 1), shared_mem_bytes: 0 },
                                (*d_attn.device_ptr() as u64, pm, pl, pa, pqg, nh_p, NSD as i32, (m as i32) | ((fmt as i32) << 28)))?;
                        }
                        dev.synchronize()?;
                        outs.push(dev.dtoh_sync_copy(&d_attn)?);
                    }
                    let fin = outs[1].iter().filter(|b| { let v = f16::from_bits(**b).to_f32(); v.is_finite() && v != 0.0 }).count();
                    let am = outs[0].iter().zip(&outs[1]).filter(|(a, b)| a != b).count();
                    if am > 0 || fin < m * NH * HD / 2 {
                        bail!("EXL3-ATTN-SPARSE FAIL {fname} ctx={ctx} {mode} m={m}: combine rows differ in {am} (finite nonzero {fin}/{})",
                              m * NH * HD);
                    }
                    cells += 1;
                    if mode == "fallback" { fb_cells += 1; }
                }
                println!("    attn-sparse {fname} ctx={ctx} {mode:<9}: widths {widths:?} union == hg3 == hg1 == splitk4 (pm/pl/pacc) \
                          and combine rows equal");
            }
        }
    }
    println!("EXL3-ATTN-SPARSE: PASS ({cells} cells incl. {fb_cells} fallback-path cells: W4/QSA union gather bitwise vs \
              hg3 / hg1 / splitk4 in f32/f16/fp8/q8 at 8K/32K and f32/q8 at 128K)");
    println!("    union stats (overlap mode, m=8): {}", union_stat.join("; "));

    // ---- timing (synthetic, isolated, warm; 20 launches each after 3 warm-ups)
    for &(ctx, fmt, m) in &[(32768usize, 0usize, 8usize), (32768, 0, 4), (32768, 3, 8), (131072, 0, 8), (131072, 3, 8)] {
        let maxp = ctx + 64;
        let p0 = ctx - 16;
        let rb = kv_rowbytes(fmt as u32, HD);
        let mpf = (maxp as i32) | ((fmt as i32) << 28);
        let d_cache = dev.alloc_zeros::<u8>(2 * NKV * maxp * rb)?;
        let pc = *d_cache.device_ptr() as u64;
        unsafe {
            f_syn.clone().launch(LaunchConfig { grid_dim: (maxp as u32, (2 * NKV) as u32, 1), block_dim: (HD as u32, 1, 1), shared_mem_bytes: 0 },
                                 (pc, mpf, HD as i32, NKV as i32, 0i32, 0x7157u32))?;
        }
        let nb0 = (p0 + 1) / RATIO;
        let mut base: Vec<usize> = (0..4).chain(nb0 - LOCAL..nb0).collect();
        let mut inb = vec![false; nb0 + 16];
        for &j in &base { inb[j] = true; }
        while base.len() < TOPK { let j = 4 + (rnd() as usize) % (nb0 - LOCAL - 4); if !inb[j] { inb[j] = true; base.push(j); } }
        let mut sel = vec![-1i32; m * SEL_MAX];
        let mut psel = vec![0i32; m];
        let mut sp = vec![0i32; 2 * m];
        for r in 0..m {
            let pos = p0 + r;
            let mut b = base.clone();
            let mut isb = inb.clone();
            for _ in 0..4 {   // 4 swaps per row
                loop { let i = (rnd() as usize) % b.len(); if b[i] >= 4 && b[i] < nb0 - LOCAL { isb[b[i]] = false; b.swap_remove(i); break; } }
                loop { let j = 4 + (rnd() as usize) % (nb0 - LOCAL - 4); if !isb[j] { isb[j] = true; b.push(j); break; } }
            }
            let nb = (pos + 1) / RATIO;
            for j in nb0..nb { b.push(j); }
            while b.len() > TOPK { let i = (rnd() as usize) % b.len(); if b[i] >= 4 && b[i] < nb0 - LOCAL { b.swap_remove(i); } }
            let lst = emit(&mut b, pos);
            sel[r * SEL_MAX..r * SEL_MAX + lst.len()].copy_from_slice(&lst);
            psel[r] = lst.len() as i32 - 1;
            sp[2 * r + 1] = pos as i32;
        }
        let qs: Vec<f32> = (0..m * NH * HD).map(|_| (rnd() % 40001) as f32 / 10000.0 - 2.0).collect();
        let (d_sel, d_ps, d_sp, d_qs) = (dev.htod_sync_copy(&sel)?, dev.htod_sync_copy(&psel)?, dev.htod_sync_copy(&sp)?, dev.htod_sync_copy(&qs)?);
        let (psl, pps, psp, pqs) = (*d_sel.device_ptr() as u64, *d_ps.device_ptr() as u64, *d_sp.device_ptr() as u64, *d_qs.device_ptr() as u64);
        let np = m * NH * NSD;
        let (d_pm, d_pl, d_pa) = (dev.alloc_zeros::<f32>(np)?, dev.alloc_zeros::<f32>(np)?, dev.alloc_zeros::<f32>(np * HD)?);
        let (pm, pl, pa) = (*d_pm.device_ptr() as u64, *d_pl.device_ptr() as u64, *d_pa.device_ptr() as u64);
        let g2: u64 = (NSD as u64) | ((SEL_MAX as u64) << 20) | ((mpf as u32 as u64) << 32);
        let smem = qsa_union_fits(fmt as u32, NH, NKV, HD, m, l).context("union contract")? as u32;
        let mut t = [0f64; 2];
        for (k, tk) in t.iter_mut().enumerate() {
            let run = || -> Result<()> {
                unsafe {
                    if k == 0 {
                        f_hg3.clone().launch(LaunchConfig { grid_dim: ((m * NKV * NSD) as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 },
                            (pm, pl, pa, pqs, pc, psl, pps, psp, nh_p, g2 as i64, split_hd))?;
                    } else {
                        f_un.clone().launch(LaunchConfig { grid_dim: ((NKV * NSD * (NH / NKV / 4)) as u32, 1, 1),
                                                           block_dim: ((32 * m) as u32, 1, 1), shared_mem_bytes: smem },
                            (pm, pl, pa, pqs, pc, psl, pps, psp, nh_p, g2 as i64, split_hd, m as i32))?;
                    }
                }
                Ok(())
            };
            for _ in 0..3 { run()?; }
            dev.synchronize()?;
            let t0 = std::time::Instant::now();
            for _ in 0..20 { run()?; }
            dev.synchronize()?;
            *tk = t0.elapsed().as_secs_f64() * 1e6 / 20.0;
        }
        println!("    EXL3-ATTN-SPARSE timing {} ctx={ctx} m={m}: hg3 {:.1} us, union {:.1} us per layer call ({:.2}x; synthetic \
                  back-to-back launches of one layer's KV — L2-warm, launch-inclusive; ranks the kernels, does not price in-round)",
                 ["f32", "f16", "fp8", "q8"][fmt], t[0], t[1], t[0] / t[1].max(1e-9));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// A5-K3 (attn.qsa_select): xq_qsa_topk_asc3 must write BITWISE the same selection rows as the served
// xq_qsa_topk_asc2 — the WHOLE sel row (pre-filled -1: entries past pos_sel must stay untouched) and
// pos_sel — at 8K / 32K / 128K / 256K-sized score rows (served pitch max_pos/ratio = 65,536) plus the
// dense/sparse boundary (nb 511..513 around topk 512) and an unaligned pitch, widths m = 1..8 (rows at
// consecutive positions, as a verify), eleven key classes incl. adversarial ties: real-like, peaked,
// 64-level ties, 97% +0 (ties at T = +0), all-equal, two-valued (kp = K/2 of the lower value), raw u32
// bits (negatives / NaNs / -0 rank by u32 value in both), narrow (keys within 16 ulps: short digit),
// capflood (90% of keys in one 4,096-ulp cluster: first-digit bin over the list capacity -> a second
// global digit before the list), rising, falling; "two" at >= 32K has > 4,096 keys == T, i.e. asc3's
// global-row fallback path (d); all-equal and the boundary rows take its identity path (0). Bytes past each row's nb are poisoned with the max key (an over-read would win).
// Success signal: the asc2 reference itself == a CPU (key desc, index asc) top-K + ascending emission
// on every row of every m = 8 cell. Then a harness-only warm timing (ranks, does not price in-round).
// ---------------------------------------------------------------------------
fn probe_qsa_select(dev: &std::sync::Arc<CudaDevice>) -> Result<()> {
    const RATIO: usize = 4;
    const TOPK: usize = 512;
    const SEL_MAX: usize = 2048 + RATIO - 1;
    const MMAX: usize = 8;
    let fget = |n: &str| dev.get_func(MODULE, n).with_context(|| format!("{n} (stale PTX?)"));
    let (f2, f3) = (fget("xq_qsa_topk_asc2")?, fget("xq_qsa_topk_asc3")?);
    // QsaParams (64 B): 4 pointers (unused here) | eps | hd heads rdim ratio topk sel_max pad
    let mut pb: Vec<u8> = vec![0u8; 32];
    pb.extend_from_slice(&1e-6f32.to_le_bytes());
    for v in [128usize, 4, 64, RATIO, TOPK, SEL_MAX, 0] { pb.extend_from_slice(&(v as i32).to_le_bytes()); }
    let d_params = dev.htod_sync_copy(&pb)?;
    let pp = *d_params.device_ptr() as u64;
    let mut rs = 0x4B33_5E1E_2026_0928u64;
    let mut rnd = move || -> u64 { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; rs };
    let modes = ["real", "peaked", "ties64", "zeros", "allequal", "two", "bits", "narrow", "capflood", "rising", "falling"];
    // (name, p0 = row 0 position, pitch)
    let ctxs: [(&str, usize, usize); 6] = [("boundary", 2044, 65536), ("8K", 8192 - 16, 65536), ("8K-odd", 8192 - 16, 2053),
                                           ("32K", 32768 - 16, 65536), ("128K", 131072 - 16, 65536), ("256K", 262144 - 16, 65536)];
    let launch = |f: &cudarc::driver::CudaFunction, sel: u64, ps: u64, scp: u64, posp: u64, m: usize, pitch: usize| -> Result<()> {
        unsafe {
            f.clone().launch(LaunchConfig { grid_dim: (m as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 },
                             (sel, ps, scp, posp, pp, pitch as i32, 0i32))?;
        }
        Ok(())
    };
    // CPU reference: (key desc, index asc) top-K blocks -> ascending token list + tail
    let cpu_ref = |keys: &[u32], pos: usize| -> Vec<i32> {
        let nb = (pos + 1) / RATIO;
        let mut idx: Vec<usize> = (0..nb).collect();
        if nb > TOPK {
            idx.sort_by(|&a, &b| keys[b].cmp(&keys[a]).then(a.cmp(&b)));
            idx.truncate(TOPK);
            idx.sort_unstable();
        }
        let mut v: Vec<i32> = idx.iter().flat_map(|&j| (0..RATIO).map(move |i| (j * RATIO + i) as i32)).collect();
        v.extend((nb * RATIO..=pos).map(|t| t as i32));
        v
    };
    let (mut cells, mut cpu_rows, mut tie_rows, mut flood_rows) = (0usize, 0usize, 0usize, 0usize);
    for &(cname, p0, pitch) in &ctxs {
        for &mode in &modes {
            let mut h: Vec<u32> = vec![0x7F7F_FFFFu32; MMAX * pitch];   // poison past nb: the max finite key
            for r in 0..MMAX {
                let nb = (p0 + r + 1) / RATIO;
                anyhow::ensure!(nb <= pitch, "row {r} nb {nb} > pitch {pitch}");
                let row = &mut h[r * pitch..r * pitch + nb];
                let u = |x: u64| (x >> 11) as f64 / (1u64 << 53) as f64;
                match mode {
                    "real" => for v in row.iter_mut() {
                        let s: f64 = (0..4).map(|_| (u(rnd()) * 2.0 - 1.3).max(0.0)).sum();
                        *v = ((s * 4.0 / 11.3137) as f32).to_bits();
                    },
                    "peaked" => for v in row.iter_mut() {
                        *v = if rnd() % 100 == 0 { (50.0 + u(rnd()) as f32).to_bits() } else { (1.0 + (rnd() % 1000) as f32 * 1e-6).to_bits() };
                    },
                    "ties64" => for v in row.iter_mut() { *v = ((rnd() % 64) as f32 * 0.125).to_bits(); },
                    "zeros" => for v in row.iter_mut() { *v = if rnd() % 100 < 97 { 0 } else { (u(rnd()) as f32 + 0.01).to_bits() }; },
                    "allequal" => row.fill(0.75f32.to_bits()),
                    "two" => {
                        row.fill(1.0f32.to_bits());
                        let mut c = 0;
                        while c < TOPK / 2 { let j = (rnd() as usize) % nb; if row[j] != 2.0f32.to_bits() { row[j] = 2.0f32.to_bits(); c += 1; } }
                    }
                    "bits" => for v in row.iter_mut() { *v = rnd() as u32; },
                    "narrow" => for v in row.iter_mut() { *v = 1.0f32.to_bits() + (rnd() % 16) as u32; },
                    "capflood" => for v in row.iter_mut() {
                        *v = if rnd() % 10 == 0 { ((u(rnd()) * 0.5) as f32).to_bits() } else { (1.0f32.to_bits() & !0xFFF) | (rnd() as u32 & 0xFFF) };
                    },
                    "rising" => for (j, v) in row.iter_mut().enumerate() { *v = (j as f32 * 0.01).to_bits(); },
                    _ => for (j, v) in row.iter_mut().enumerate() { *v = ((nb - j) as f32 * 0.01).to_bits(); },
                }
            }
            let d_sc = dev.htod_sync_copy(&h)?;
            let pos: Vec<i32> = (0..MMAX).map(|r| (p0 + r) as i32).collect();
            let d_pos = dev.htod_sync_copy(&pos)?;
            let (scp, posp) = (*d_sc.device_ptr() as u64, *d_pos.device_ptr() as u64);
            for m in 1..=MMAX {
                let d_s2 = dev.htod_sync_copy(&vec![-1i32; m * SEL_MAX])?;
                let d_s3 = dev.htod_sync_copy(&vec![-1i32; m * SEL_MAX])?;
                let d_p2 = dev.htod_sync_copy(&vec![-77i32; m])?;
                let d_p3 = dev.htod_sync_copy(&vec![-99i32; m])?;
                launch(&f2, *d_s2.device_ptr() as u64, *d_p2.device_ptr() as u64, scp, posp, m, pitch)?;
                launch(&f3, *d_s3.device_ptr() as u64, *d_p3.device_ptr() as u64, scp, posp, m, pitch)?;
                dev.synchronize()?;
                let (s2, s3) = (dev.dtoh_sync_copy(&d_s2)?, dev.dtoh_sync_copy(&d_s3)?);
                let (q2, q3) = (dev.dtoh_sync_copy(&d_p2)?, dev.dtoh_sync_copy(&d_p3)?);
                let nd = s2.iter().zip(&s3).filter(|(a, b)| a != b).count() + q2.iter().zip(&q3).filter(|(a, b)| a != b).count();
                if nd > 0 {
                    let first = (0..m * SEL_MAX).find(|&i| s2[i] != s3[i]);
                    bail!("EXL3-QSA-SELECT FAIL {cname} {mode} m={m}: asc3 vs asc2 differ in {nd} words (pos_sel asc2 {q2:?} asc3 {q3:?}; \
                           first entry {first:?} = row {:?}: asc2 {:?} asc3 {:?})", first.map(|f| f / SEL_MAX),
                          first.map(|f| s2[f]), first.map(|f| s3[f]));
                }
                if m == MMAX {
                    for r in 0..m {
                        let p = p0 + r;
                        let want = cpu_ref(&h[r * pitch..(r + 1) * pitch], p);
                        let got = &s2[r * SEL_MAX..r * SEL_MAX + want.len()];
                        if q2[r] != want.len() as i32 - 1 || got != &want[..] || s2[r * SEL_MAX + want.len()..(r + 1) * SEL_MAX].iter().any(|&x| x != -1) {
                            bail!("EXL3-QSA-SELECT FAIL {cname} {mode} row {r}: the asc2 reference != the CPU top-K (pos_sel {} vs {})",
                                  q2[r], want.len() as i32 - 1);
                        }
                        cpu_rows += 1;
                        // a tie row: the K-th largest key is shared with an unselected block
                        let nb = (p + 1) / RATIO;
                        if nb > TOPK {
                            let row = &h[r * pitch..r * pitch + nb];
                            let mut ks: Vec<u32> = row.to_vec();
                            ks.sort_unstable_by(|a, b| b.cmp(a));
                            let t = ks[TOPK - 1];
                            if ks[TOPK] == t { tie_rows += 1; }
                            // asc3's global-row fallback (d): > XQ_TK3_CAP keys equal to T (no digit's bin ever fits)
                            if ks.iter().filter(|&&k| k == t).count() > 4096 { flood_rows += 1; }
                        }
                    }
                }
                cells += 1;
            }
        }
        println!("    qsa-select {cname:<8} (p0 {p0}, pitch {pitch}): asc3 == asc2 (whole rows + pos_sel), m 1..=8 x {} key classes", modes.len());
    }
    anyhow::ensure!(tie_rows > 0 && flood_rows > 0, "EXL3-QSA-SELECT: the adversarial classes did not produce ties at T ({tie_rows}) \
                                                     or a fallback-path row ({flood_rows})");
    println!("EXL3-QSA-SELECT: PASS ({cells} cells: A5-K3 xq_qsa_topk_asc3 bitwise == xq_qsa_topk_asc2 on whole -1-prefilled sel rows + \
              pos_sel; asc2 == CPU (key desc, index asc) top-K on {cpu_rows} rows, {tie_rows} with ties at T, {flood_rows} on asc3's \
              global-row fallback (> {} keys == T))", 4096);
    // ---- harness-only warm timing (rows L2-resident as in-round after the scorer; 50 back-to-back launches)
    for &(ctx, m) in &[(32768usize, 8usize), (32768, 1), (131072, 8), (131072, 1), (262144, 8), (262144, 1)] {
        let pitch = 65536usize;
        let p0 = ctx - 16;
        let mut h = vec![0u32; m * pitch];
        for r in 0..m {
            for v in &mut h[r * pitch..r * pitch + (p0 + r + 1) / RATIO] {
                let s: f64 = (0..4).map(|_| (((rnd() >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.3).max(0.0)).sum();
                *v = ((s * 4.0 / 11.3137) as f32).to_bits();
            }
        }
        let d_sc = dev.htod_sync_copy(&h)?;
        let d_pos = dev.htod_sync_copy(&(0..m).map(|r| (p0 + r) as i32).collect::<Vec<i32>>())?;
        let d_s = dev.htod_sync_copy(&vec![-1i32; m * SEL_MAX])?;
        let d_p = dev.htod_sync_copy(&vec![0i32; m])?;
        let (scp, posp, sp, psp) = (*d_sc.device_ptr() as u64, *d_pos.device_ptr() as u64, *d_s.device_ptr() as u64, *d_p.device_ptr() as u64);
        let mut t = [0f64; 2];
        for (k, f) in [&f2, &f3].into_iter().enumerate() {
            for _ in 0..3 { launch(f, sp, psp, scp, posp, m, pitch)?; }
            dev.synchronize()?;
            let t0 = std::time::Instant::now();
            for _ in 0..50 { launch(f, sp, psp, scp, posp, m, pitch)?; }
            dev.synchronize()?;
            t[k] = t0.elapsed().as_secs_f64() * 1e6 / 50.0;
        }
        println!("    EXL3-QSA-SELECT timing ctx={ctx} m={m}: asc2 {:.1} us, asc3 {:.1} us per call ({:.2}x; synthetic real-like \
                  scores, L2-warm, launch-inclusive — ranks the kernels, does not price in-round)", t[0], t[1], t[0] / t[1].max(1e-9));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// WP18 rung A: the PREFILL selection bit-contract (EXL3-WP18). Synthetic Qwen3.8-Flash-Next
// indexer (hd 128, 4 heads, ratio 4, top-512 blocks, sel_max 2,051); chunks of the served width
// C = 2048 ending at ~8.5K / 32K / 128K (1 / 1 / 4 slab tiles), a QSA-straddle chunk (sparse rows
// start mid-chunk, t0 > 0) and a short odd chunk (77 rows, partial CTA tiles). For every sparse
// row, the WP18 walk — xq_wp18_score + xq_wp18_merge over exl3_forward::wp18_plan, the served
// loop, at slab row groups {32, 256, 2048} — must write the SAME list entries [0, pos_sel] and
// pos_sel as the old pair (qsa_score_prefill_b + xq_qsa_topk_asc), touch nothing else (poisoned
// lists), and every scored slab tile must equal the old matrix bitwise (xq_wp18_cmp; the compared
// count must be every visible (row, block) pair). Key modes: random; ties (a third of the blocks
// copy the block one tile earlier, a sixth a nearby earlier block — exact score ties across and
// within tiles, resolved by the lower index); zeros (97% zero keys: rows with < 512 positive
// blocks fill up with the LOWEST-index zero-score blocks); rising (key scale grows with the block
// index: every tile floods the carried list — the merge's worst case). Then a harness-only timing.
// ---------------------------------------------------------------------------
fn probe_wp18_select(dev: &std::sync::Arc<CudaDevice>) -> Result<()> {
    use crate::exl3_forward::{wp18_plan, WP18_MT, WP18_T};
    const HD: usize = 128;
    const NHX: usize = 4;
    const QK: usize = (NHX + 1) * HD;
    const RATIO: usize = 4;
    const BUDGET: usize = 2048;
    const TOPK: usize = BUDGET / RATIO;
    const SEL_MAX: usize = BUDGET + RATIO - 1;
    const PF_QT: usize = 8;                      // gpu_batch.cu QSA_PF_QT
    if dev.get_func("wp18_gb", "qsa_score_prefill_b").is_none() {
        let ptx = Ptx::from_src(std::fs::read_to_string("src/ptx/gpu_batch.ptx")
            .context("src/ptx/gpu_batch.ptx missing — run cargo build --release")?);
        dev.load_ptx(ptx, "wp18_gb", &["qsa_score_prefill_b"]).context("load gpu_batch (qsa_score_prefill_b)")?;
    }
    let f_old = dev.get_func("wp18_gb", "qsa_score_prefill_b").context("qsa_score_prefill_b")?;
    let fget = |n: &str| dev.get_func(MODULE, n).with_context(|| format!("{n} (stale PTX?)"));
    let (f_topk, f_sc, f_mg, f_cmp) = (fget("xq_qsa_topk_asc")?, fget("xq_wp18_score")?, fget("xq_wp18_merge")?,
                                       fget("xq_wp18_cmp")?);
    // QsaParams (64 B): 4 pointers (unused by these kernels) | eps | hd heads rdim ratio topk sel_max pad
    let mut pb: Vec<u8> = vec![0u8; 32];
    pb.extend_from_slice(&1e-6f32.to_le_bytes());
    for v in [HD, NHX, 64, RATIO, TOPK, SEL_MAX, 0] { pb.extend_from_slice(&(v as i32).to_le_bytes()); }
    let d_params = dev.htod_sync_copy(&pb)?;
    let pp = *d_params.device_ptr() as u64;
    let mut rs = 0x5EED_0018_2026_0927u64;
    let mut rnd = move || -> u64 { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; rs };
    let bf = |x: f32| half::bf16::from_f32(x).to_bits();
    let lim = SEL_MAX;                           // qsa_limit(): rows at pos > lim are sparse
    let cases: [(&str, usize, usize); 5] = [("8.5K", 8704 - 2048, 2048), ("32K", 32768 - 2048, 2048),
                                            ("128K", 131072 - 2048, 2048), ("straddle", 1024, 2048),
                                            ("odd", 40000, 77)];
    let modes = ["random", "ties", "zeros", "rising"];
    let rgs = [32usize, 256, 2048];
    let (mut cells, mut rows_tot, mut pairs_tot, mut ties_tot, mut ztie_tot) = (0usize, 0usize, 0usize, 0usize, 0usize);
    // timing inputs kept from the random mode at 32K / 128K
    struct Keep { name: &'static str, pos0: usize, c: usize, nbs: usize, q: CudaSlice<u16>, plane: CudaSlice<u16>, pos: CudaSlice<i32> }
    let mut keep: Vec<Keep> = Vec::new();
    for &(cname, pos0, c) in &cases {
        let t0 = if pos0 > lim { 0 } else { (lim + 1 - pos0).min(c) };
        let rows = c - t0;
        anyhow::ensure!(rows > 0, "case {cname}: no sparse rows");
        let nblk = (pos0 + c) / RATIO;
        let nbs = nblk + 8;                      // matrix pitch (served: max_pos / ratio; any pitch >= nblk)
        let ntiles = nblk.div_ceil(WP18_T);
        for &mode in &modes {
            // ---- synthetic q rows (4 heads + the unused k head) and pooled keys
            let q: Vec<u16> = (0..c * QK).map(|_| bf((rnd() % 30001) as f32 / 10000.0 - 1.5)).collect();
            let mut keys: Vec<u16> = (0..nbs * HD).map(|_| bf((rnd() % 20001) as f32 / 10000.0 - 1.0)).collect();
            match mode {
                "ties" => for j in 1..nblk {
                    let r = rnd() % 6;
                    let src = if r < 2 && j >= WP18_T { Some(j - WP18_T) }
                              else if r == 2 { Some(j - 1 - (rnd() as usize) % j.min(700)) } else { None };
                    if let Some(s) = src { let row: Vec<u16> = keys[s * HD..(s + 1) * HD].to_vec(); keys[j * HD..(j + 1) * HD].copy_from_slice(&row); }
                },
                "zeros" => for j in 0..nblk { if rnd() % 100 < 97 { keys[j * HD..(j + 1) * HD].fill(0); } },
                "rising" => for j in 0..nblk {
                    let s = 0.25 + 3.0 * j as f32 / nblk as f32;
                    for v in &mut keys[j * HD..(j + 1) * HD] { *v = bf(half::bf16::from_bits(*v).to_f32() * s); }
                },
                _ => {}
            }
            let pos: Vec<i32> = (0..c).map(|i| (pos0 + i) as i32).collect();
            let (d_q, d_plane, d_pos) = (dev.htod_sync_copy(&q)?, dev.htod_sync_copy(&keys)?, dev.htod_sync_copy(&pos)?);
            let (qp, kp, posp) = (*d_q.device_ptr() as u64, *d_plane.device_ptr() as u64, *d_pos.device_ptr() as u64);
            // ---- reference: the old pair (all c rows scored, as served)
            let d_scores = dev.alloc_zeros::<f32>(c * nbs)?;
            let scp = *d_scores.device_ptr() as u64;
            let d_rsel = dev.htod_sync_copy(&vec![-9i32; c * SEL_MAX])?;
            let d_rps = dev.htod_sync_copy(&vec![-9i32; c])?;
            unsafe {
                f_old.clone().launch(LaunchConfig { grid_dim: (nblk.max(1).div_ceil(256) as u32, c.div_ceil(PF_QT) as u32, 1),
                                                    block_dim: (256, 1, 1), shared_mem_bytes: (PF_QT * NHX * HD * 4) as u32 },
                    (scp, qp, kp, pp, pos0 as i32, c as i32, nbs as i32, QK as i32))?;
                f_topk.clone().launch(LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 },
                    (*d_rsel.device_ptr() as u64 + (t0 * SEL_MAX * 4) as u64, *d_rps.device_ptr() as u64 + (t0 * 4) as u64,
                     scp + (t0 * nbs * 4) as u64, posp + (t0 * 4) as u64, pp, nbs as i32, 0i32))?;
            }
            dev.synchronize()?;
            let (rsel, rps) = (dev.dtoh_sync_copy(&d_rsel)?, dev.dtoh_sync_copy(&d_rps)?);
            let full = dev.dtoh_sync_copy(&d_scores)?;
            // reference plausibility + tie statistics (sampled rows)
            let mut vis = 0usize;
            let (mut ties, mut zties) = (0usize, 0usize);
            for t in t0..c {
                let p = pos0 + t;
                let nb = (p + 1) / RATIO;
                vis += nb;
                let want = nb.min(TOPK) * RATIO + (p + 1 - nb * RATIO);
                if rps[t] != want as i32 - 1 {
                    bail!("EXL3-WP18 FAIL {cname} {mode}: implausible reference pos_sel {} at row {t} (want {})", rps[t], want - 1);
                }
                let l = &rsel[t * SEL_MAX..t * SEL_MAX + want];
                if l.windows(2).any(|w| w[0] >= w[1]) || l[0] < 0 {
                    bail!("EXL3-WP18 FAIL {cname} {mode}: reference list of row {t} not strictly ascending");
                }
                if (t - t0) % 8 == 0 && nb > TOPK {
                    let mut v: Vec<u32> = full[t * nbs..t * nbs + nb].iter().map(|x| x.to_bits()).collect();
                    v.select_nth_unstable_by(TOPK, |a, b| b.cmp(a));   // v[TOPK] = the (K+1)-th largest
                    let k1 = v[TOPK];
                    let kth = *v[..TOPK].iter().min().unwrap();
                    if kth == k1 { ties += 1; }
                    if kth == 0 { zties += 1; }
                }
            }
            ties_tot += ties;
            ztie_tot += zties;
            if mode == "zeros" && (cname == "8.5K" || cname == "32K") && zties == 0 {
                bail!("EXL3-WP18 FAIL {cname} zeros: no sampled row has its top-{TOPK} boundary at score 0 (tie path unexercised)");
            }
            for &rg in &rgs {
                let gr = rg.min(c);
                let d_slab = dev.alloc_zeros::<f32>(gr * WP18_T)?;
                let d_ck = dev.alloc_zeros::<u32>(gr * TOPK)?;
                let d_ci = dev.alloc_zeros::<i32>(gr * TOPK)?;
                let d_sel = dev.htod_sync_copy(&vec![-7i32; c * SEL_MAX])?;
                let d_ps = dev.htod_sync_copy(&vec![-7i32; c])?;
                let d_cnt = dev.htod_sync_copy(&[0i32, i32::MAX, 0, 0])?;
                let (slp, ckp, cip, selp, psp, cntp) = (*d_slab.device_ptr() as u64, *d_ck.device_ptr() as u64,
                    *d_ci.device_ptr() as u64, *d_sel.device_ptr() as u64, *d_ps.device_ptr() as u64, *d_cnt.device_ptr() as u64);
                let plan = wp18_plan(pos0 + t0, rows, RATIO, gr);
                for st in &plan {
                    let row = t0 + st.g0;
                    unsafe {
                        if st.score_grid.0 > 0 {
                            f_sc.clone().launch(LaunchConfig { grid_dim: (st.score_grid.0, st.score_grid.1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                                (slp, qp + (row * QK * 2) as u64, kp, pp, st.pos_first as i32, st.gn as i32, (st.kt * WP18_T) as i32, QK as i32))?;
                            f_cmp.clone().launch(LaunchConfig { grid_dim: (st.gn as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                                (cntp, slp, scp, pp, st.pos_first as i32, row as i32, st.kt as i32, nbs as i32))?;
                        }
                        f_mg.clone().launch(LaunchConfig { grid_dim: (st.gn as u32, 1, 1), block_dim: (WP18_MT as u32, 1, 1), shared_mem_bytes: 0 },
                            (selp + (row * SEL_MAX * 4) as u64, psp + (row * 4) as u64, slp, ckp, cip, pp, st.pos_first as i32, st.kt as i32))?;
                    }
                }
                dev.synchronize()?;
                let (sel, ps, cnt) = (dev.dtoh_sync_copy(&d_sel)?, dev.dtoh_sync_copy(&d_ps)?, dev.dtoh_sync_copy(&d_cnt)?);
                if cnt[0] != 0 {
                    bail!("EXL3-WP18 FAIL {cname} {mode} rows/group {gr}: {} slab scores bit-differ from qsa_score_prefill_b \
                           (first at chunk row {} block {})", cnt[0], cnt[1] >> 16, cnt[1] & 0xFFFF);
                }
                if cnt[2] as usize != vis {
                    bail!("EXL3-WP18 FAIL {cname} {mode} rows/group {gr}: compared {} scores, want every visible pair {vis}", cnt[2]);
                }
                for t in 0..c {
                    let r = &sel[t * SEL_MAX..(t + 1) * SEL_MAX];
                    if t < t0 {
                        if ps[t] != -7 || r.iter().any(|&x| x != -7) {
                            bail!("EXL3-WP18 FAIL {cname} {mode} rows/group {gr}: dense row {t} was written");
                        }
                        continue;
                    }
                    let n = (rps[t] + 1) as usize;
                    if ps[t] != rps[t] || r[..n] != rsel[t * SEL_MAX..t * SEL_MAX + n] || r[n..].iter().any(|&x| x != -7) {
                        let i = (0..n).find(|&i| r[i] != rsel[t * SEL_MAX + i]);
                        bail!("EXL3-WP18 FAIL {cname} {mode} rows/group {gr}: row {t} (pos {}) pos_sel {} vs {}; first entry diff {:?}; \
                               writes past pos_sel {}", pos0 + t, ps[t], rps[t],
                              i.map(|i| (i, r[i], rsel[t * SEL_MAX + i])), r[n..].iter().filter(|&&x| x != -7).count());
                    }
                }
                cells += 1;
                rows_tot += rows;
                pairs_tot += vis;
            }
            println!("    wp18 {cname:<8} {mode:<6}: rows {rows} (t0 {t0}), {nblk} blocks = {ntiles} tile(s); lists + pos_sel == \
                      old at row groups {rgs:?}; {vis} scores bitwise; sampled boundary ties {ties}, zero-score boundaries {zties}");
            if mode == "random" && (cname == "32K" || cname == "128K") {
                keep.push(Keep { name: cname, pos0, c, nbs, q: d_q, plane: d_plane, pos: d_pos });
            }
        }
    }
    if ties_tot == 0 {
        bail!("EXL3-WP18 FAIL: no sampled row had a tie at its top-{TOPK} boundary (tie-break path unexercised)");
    }
    println!("EXL3-WP18: PASS ({cells} cells: {rows_tot} sparse rows, {pairs_tot} scores — WP18 bounded walk == \
              qsa_score_prefill_b + xq_qsa_topk_asc bitwise at ~8.5K/32K/128K, straddle and odd chunks; \
              {ties_tot} sampled boundary ties, {ztie_tot} zero-score boundaries)");

    // ---- timing (harness-only: one synthetic chunk of C = 2048 rows at the window end, one layer,
    // L2-warm, launch-inclusive; ranks the paths — the served number is the prefill tok/s A/B)
    for k in &keep {
        let (pos0, c, nbs) = (k.pos0, k.c, k.nbs);
        let t0 = if pos0 > lim { 0 } else { (lim + 1 - pos0).min(c) };
        let rows = c - t0;
        let nblk = (pos0 + c) / RATIO;
        let (qp, kp, posp) = (*k.q.device_ptr() as u64, *k.plane.device_ptr() as u64, *k.pos.device_ptr() as u64);
        let d_scores = dev.alloc_zeros::<f32>(c * nbs)?;
        let d_sel = dev.alloc_zeros::<i32>(c * SEL_MAX)?;
        let d_ps = dev.alloc_zeros::<i32>(c)?;
        let gr = crate::exl3_forward::wp18_rows().min(c);
        let d_slab = dev.alloc_zeros::<f32>(gr * WP18_T)?;
        let d_ck = dev.alloc_zeros::<u32>(gr * TOPK)?;
        let d_ci = dev.alloc_zeros::<i32>(gr * TOPK)?;
        let (scp, selp, psp) = (*d_scores.device_ptr() as u64, *d_sel.device_ptr() as u64, *d_ps.device_ptr() as u64);
        let (slp, ckp, cip) = (*d_slab.device_ptr() as u64, *d_ck.device_ptr() as u64, *d_ci.device_ptr() as u64);
        let plan = wp18_plan(pos0 + t0, rows, RATIO, gr);
        let old_score = || -> Result<()> { unsafe {
            f_old.clone().launch(LaunchConfig { grid_dim: (nblk.max(1).div_ceil(256) as u32, c.div_ceil(PF_QT) as u32, 1),
                                                block_dim: (256, 1, 1), shared_mem_bytes: (PF_QT * NHX * HD * 4) as u32 },
                (scp, qp, kp, pp, pos0 as i32, c as i32, nbs as i32, QK as i32))?; } Ok(()) };
        let old_topk = || -> Result<()> { unsafe {
            f_topk.clone().launch(LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 },
                (selp + (t0 * SEL_MAX * 4) as u64, psp + (t0 * 4) as u64, scp + (t0 * nbs * 4) as u64, posp + (t0 * 4) as u64,
                 pp, nbs as i32, 0i32))?; } Ok(()) };
        let walk = || -> Result<()> {
            for st in &plan {
                let row = t0 + st.g0;
                unsafe {
                    if st.score_grid.0 > 0 {
                        f_sc.clone().launch(LaunchConfig { grid_dim: (st.score_grid.0, st.score_grid.1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                            (slp, qp + (row * QK * 2) as u64, kp, pp, st.pos_first as i32, st.gn as i32, (st.kt * WP18_T) as i32, QK as i32))?;
                    }
                    f_mg.clone().launch(LaunchConfig { grid_dim: (st.gn as u32, 1, 1), block_dim: (WP18_MT as u32, 1, 1), shared_mem_bytes: 0 },
                        (selp + (row * SEL_MAX * 4) as u64, psp + (row * 4) as u64, slp, ckp, cip, pp, st.pos_first as i32, st.kt as i32))?;
                }
            }
            Ok(())
        };
        let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
            for _ in 0..2 { f()?; }
            dev.synchronize()?;
            let t = std::time::Instant::now();
            for _ in 0..5 { f()?; }
            dev.synchronize()?;
            Ok(t.elapsed().as_secs_f64() * 1e3 / 5.0)
        };
        let (ts, tk, tw) = (time(&old_score)?, time(&old_topk)?, time(&walk)?);
        println!("    EXL3-WP18 timing {} (pos0 {pos0}, {rows} rows, {nblk} blocks): old {:.2} ms (qsa_score_prefill_b {ts:.2} + \
                  xq_qsa_topk_asc {tk:.2}) vs WP18 walk {tw:.2} ms at {gr} rows/group ({:.2}x) — harness-only, one layer-chunk, \
                  L2-warm; ranks the paths, the served number is the prefill A/B vs --wp18-off=1",
                 k.name, ts + tk, (ts + tk) / tw.max(1e-9));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// S-A3-f-b: WIDE-M GEMM gate (chunked prefill enables the M > 16 path).
// For every module class and a ladder of M:
//   (a) y_raw rel-L2 vs fp16(gemm_ref) on the first min(M,16) rows at the
//       calibrated 1e-4 threshold (full-M host refs cost O(M·K·N) scalar FLOPs);
//   (b) y_raw BITWISE identical across MT in {1,2,4,8} at EVERY M (structural
//       exactness: any tile-boundary/indexing/padding bug is a bit diff);
//   (c) at M=16, y_raw BITWISE identical to the ORIGINAL m16 kernel
//       (row-wise equivalence of the wide body, decode included).
//   ./gb10_inference --probe-exl3-wide --model-dir <pack> \
//     [--exl3-wide-ms 1,4,16,17,32,64,256,512] [--exl3-wide-mts 1,2,4,8]
// ---------------------------------------------------------------------------
pub fn probe_wide(dir: &str, ms: &[usize], mts: &[usize], slab_cols: usize) -> Result<()> {
    let pack = Exl3Pack::open(dir)?;
    let dev = dev0()?;
    load_module(&dev)?;
    let stream = fork_blocking_stream(&dev);

    for m in pick_modules(&pack) {
        let raw = m.trellis.read_bytes(&pack.dir)?;
        let (trb, n_eff) = if m.class == TensorClass::LmHead {
            slab_trellis(&raw, m.k / 16, m.n / 16, m.bits, Some(slab_cols))
        } else {
            (raw.clone(), m.n)
        };
        let suh = le_f16_bits(&m.suh.read_bytes(&pack.dir)?);
        let k = m.k;
        let bits = m.bits;
        let tr_bits: Vec<u16> = trb
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        // one host decode per module; the wide kernel must reproduce it through
        // the mma path (rel-L2) and the m16 kernel bitwise (structural).
        let w_dec = decode_trellis_host(
            &tr_bits.iter().map(|v| *v as i16).collect::<Vec<_>>(),
            k / 16,
            n_eff / 16,
            bits,
        );
        println!(
            "  {:<24} b={} K={:<5} N={:<6}",
            m.name.rsplit('.').take(2).collect::<Vec<_>>().join("."),
            bits,
            k,
            n_eff
        );
        let f_wide = dev
            .get_func(MODULE, "exl3_hmma_gemm_wide")
            .ok_or_else(|| anyhow::anyhow!("fn exl3_hmma_gemm_wide missing"))?;
        let f_m16 = dev
            .get_func(MODULE, "exl3_hmma_gemm")
            .ok_or_else(|| anyhow::anyhow!("fn exl3_hmma_gemm missing"))?;
        let d_tr = dev.htod_sync_copy(&tr_bits)?;
        for &mm in ms {
            let x = synth_x(mm, k, 0xC0FFEE);
            let xh = had_scale_ref(&x, &suh, mm, k, true);
            let d_xh = dev.htod_sync_copy(&xh)?;
            let mut d_y = unsafe { dev.alloc::<u16>(mm * n_eff) }?;
            let mut first: Vec<u16> = Vec::new();
            let mut rel_report = 0.0f64;
            // (c-first) BITWISE vs the ORIGINAL m16 kernel at M=16 — run before any
            // tolerance check so a decode/indexing defect is diagnosed directly.
            if mm == 16 {
                // S-A3-f-b param-echo bisect: dump the args the kernel RECEIVES.
                {
                    let echo_in: Vec<u16> = vec![0x7777u16; 64];
                    let mut d_echo = dev.htod_sync_copy(&echo_in)?;
                    unsafe {
                        f_wide
                            .clone()
                            .launch_on_stream(
                                &stream,
                                LaunchConfig {
                                    grid_dim: (1, 1, 1),
                                    block_dim: (32, 1, 1),
                                    shared_mem_bytes: 0,
                                },
                                (&d_tr, &d_xh, &mut d_echo, 16i32, k as i32, n_eff as i32, bits as i32, -1i32),
                            )
                            .unwrap();
                    }
                    dev.synchronize().unwrap();
                    let eh: Vec<u16> = dev.dtoh_sync_copy(&d_echo)?;
                    let wd = |i: usize| eh[i] as u64;
                    let ptr = |o: usize| wd(o) | (wd(o + 1) << 16) | (wd(o + 2) << 32) | (wd(o + 3) << 48);
                    let sc = |i: usize| i32::from(eh[12 + i]) | (i32::from(eh[13 + i]) << 16);
                    println!(
                        "    echo: trellis={:#x} xh={:#x} y={:#x} M={} K={} N={} bits={} mt={}",
                        ptr(0), ptr(4), ptr(8), sc(0), sc(2), sc(4), sc(6), sc(8)
                    );
                    println!(
                        "    host: trellis={:#x} xh={:#x}",
                        *d_tr.device_ptr(),
                        *d_xh.device_ptr()
                    );
                }
                let mut d_y16 = unsafe { dev.alloc::<u16>(16 * n_eff) }?;
                unsafe {
                    f_m16
                        .clone()
                        .launch_on_stream(
                            &stream,
                            LaunchConfig {
                                grid_dim: ((n_eff / 128) as u32, 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (&d_tr, &d_xh, &mut d_y16, 16i32, k as i32, n_eff as i32, bits as i32),
                        )
                        .unwrap();
                }
                dev.synchronize().unwrap();
                let y16 = dev.dtoh_sync_copy(&d_y16)?;
                // POISON: the wide target starts as 0xDEAD — if the launch writes
                // nothing, every 0xDEAD survives (allocator-stable garbage would
                // otherwise mimic a "wrong but real" kernel).
                let poison = vec![0xDEADu16; 16 * n_eff];
                let d_poison = dev.htod_sync_copy(&poison)?;
                drop(d_y);
                d_y = d_poison;
                unsafe {
                    f_wide
                        .clone()
                        .launch_on_stream(
                            &stream,
                            LaunchConfig {
                                grid_dim: (1u32, (n_eff / 128) as u32, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (&d_tr, &d_xh, &mut d_y, 16i32, k as i32, n_eff as i32, bits as i32, 1i32),
                        )
                        .unwrap();
                }
                dev.synchronize().unwrap();
                let yw = dev.dtoh_sync_copy(&d_y)?;
                let survivors = yw.iter().filter(|v| **v == 0xDEAD).count();
                let mut bad = 0usize;
                for (i, (a, b)) in y16.iter().zip(yw.iter()).enumerate() {
                    if a != b {
                        bad += 1;
                        if bad <= 6 {
                            println!(
                                "    wide-vs-m16 [{i}] (row {}, col {}): m16 {:04x} wide {:04x}",
                                i / n_eff,
                                i % n_eff,
                                a,
                                b
                            );
                        }
                    }
                }
                println!(
                    "    M=16 m16-vs-wide checksum: m16 {:016x} wide {:016x} mismatches {bad}/{}  poison_survivors {survivors}",
                    y16.iter().fold(0u64, |s, v| s.wrapping_mul(31).wrapping_add(*v as u64)),
                    yw.iter().fold(0u64, |s, v| s.wrapping_mul(31).wrapping_add(*v as u64)),
                    y16.len()
                );
                if bad > 0 {
                    bail!("EXL3-WIDE FAIL on {} M=16: wide != m16 kernel bitwise ({bad})", m.name);
                }
            }
            for &mt in mts {
                let msup = (mm + 16 * mt - 1) / (16 * mt);
                unsafe {
                    f_wide
                        .clone()
                        .launch_on_stream(
                            &stream,
                            LaunchConfig {
                                grid_dim: (msup as u32, (n_eff / 128) as u32, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                &d_tr,
                                &d_xh,
                                &mut d_y,
                                mm as i32,
                                k as i32,
                                n_eff as i32,
                                bits as i32,
                                mt as i32,
                            ),
                        )
                        .unwrap();
                }
                dev.synchronize().unwrap();
                let y = dev.dtoh_sync_copy(&d_y)?;
                // (b) bitwise across MT — the structural exactness check
                if first.is_empty() {
                    first = y.clone();
                } else if first[..] != y[..] {
                    let mut bad = 0usize;
                    for (i, (a, b)) in first.iter().zip(y.iter()).enumerate() {
                        if a != b {
                            bad += 1;
                            if bad < 4 {
                                println!(
                                    "    mt={mt} mismatch [{i}]: {:04x} vs mt={} {:04x}",
                                    b, mts[0], a
                                );
                            }
                        }
                    }
                    bail!(
                        "EXL3-WIDE FAIL on {} M={} mt={mt}: {bad} bitwise mismatches vs mt={}",
                        m.name,
                        mm,
                        mts[0]
                    );
                }
                // (a) rel-L2 vs fp16(host ref) on the first min(M,16) rows
                let rref = mm.min(16);
                let yref = gemm_ref(&w_dec, &xh, rref, k, n_eff);
                let yref16: Vec<f32> = yref.iter().map(|v| f16::from_f32(*v).to_f32()).collect();
                let (_, rel16) = rel_l2(&y[..rref * n_eff], &yref16);
                rel_report = rel16;
                // Sanity threshold only (the REAL gate is the bitwise checks above):
                // single-row draws straddle the fp16(ref) rounding boundary — measured
                // 1.028e-4 on the lm_head slab b5 at M=1 with the m16 kernel itself
                // bitwise-green on the same data (S-A3-c: structural bugs are O(1)).
                let rel_cap = if rref < 4 { 2.5e-4 } else { 1e-4 };
                if rel16 > rel_cap {
                    bail!(
                        "EXL3-WIDE FAIL on {} M={} mt={mt}: y_raw rel-L2 {:.3e} > {}",
                        m.name,
                        mm,
                        rel16,
                        rel_cap
                    );
                }
            }
            // (c) M=16: bitwise vs the ORIGINAL m16 kernel (same x, same xh)
            if mm == 16 {
                let mut d_y16 = unsafe { dev.alloc::<u16>(16 * n_eff) }?;
                unsafe {
                    f_m16
                        .clone()
                        .launch_on_stream(
                            &stream,
                            LaunchConfig {
                                grid_dim: ((n_eff / 128) as u32, 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (&d_tr, &d_xh, &mut d_y16, 16i32, k as i32, n_eff as i32, bits as i32),
                        )
                        .unwrap();
                }
                dev.synchronize().unwrap();
                let y16 = dev.dtoh_sync_copy(&d_y16)?;
                if y16[..] != first[..] {
                    let mut bad = 0usize;
                    for (i, (a, b)) in y16.iter().zip(first.iter()).enumerate() {
                        if a != b {
                            bad += 1;
                            if bad < 4 {
                                println!("    wide-vs-m16 mismatch [{i}]: {:04x} vs {:04x}", b, a);
                            }
                        }
                    }
                    bail!(
                        "EXL3-WIDE FAIL on {} M=16: wide != m16 kernel bitwise ({bad} mismatches)",
                        m.name
                    );
                }
            }
            println!(
                "    M={:<4} mts {:?}: PASS (bitwise across MT{}, rel-L2 vs fp16 ref {:.2e})",
                mm,
                mts,
                if mm == 16 { " + vs m16" } else { "" },
                rel_report
            );
        }
    }
    println!("EXL3-WIDE: PASS");
    Ok(())
}

// ---------------------------------------------------------------------------
// S-A3-f-b: wide-M GEMM bench (the chunked-prefill weight sweep). GEMM-only
// timing (Hadamards excluded, same convention as bench_gemm's gemm-only rows).
// GB/s = compressed-input (trellis bytes / time, one-way reads, roofline 238);
// rows/s = M / time is the prefill-throughput metric the TTFT table cares about.
//   ./gb10_inference --bench-exl3-wide --model-dir <pack> \
//     [--exl3-wide-ms 32,64,128,256,512,1024] [--exl3-wide-mt 4] [--exl3-iters 20]
// ---------------------------------------------------------------------------
pub fn bench_wide(dir: &str, ms: &[usize], mt: usize, iters: usize) -> Result<()> {
    let pack = Exl3Pack::open(dir)?;
    let dev = dev0()?;
    load_module(&dev)?;
    let stream = fork_blocking_stream(&dev);
    println!(
        "=== EXL3 wide-M GEMM bench (mt={mt}, {iters} iters/rep, best of 3; GB/s = compressed-input trellis bytes/time, roofline 238) ==="
    );
    for m in pick_modules(&pack) {
        let raw = m.trellis.read_bytes(&pack.dir)?;
        let tr_bits: Vec<u16> = raw
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        let suh = le_f16_bits(&m.suh.read_bytes(&pack.dir)?);
        let label = m
            .name
            .rsplit('.')
            .take(2)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join(".");
        let d_tr = dev.htod_sync_copy(&tr_bits)?;
        let d_suh = dev.htod_sync_copy(&suh)?;
        let f_wide = dev
            .get_func(MODULE, "exl3_hmma_gemm_wide")
            .ok_or_else(|| anyhow::anyhow!("fn exl3_hmma_gemm_wide missing"))?;
        let f_suh = dev
            .get_func(MODULE, "exl3_had_suh")
            .ok_or_else(|| anyhow::anyhow!("fn exl3_had_suh missing"))?;
        for &mm in ms {
            let x = synth_x(mm, m.k, 0xBEEF);
            let d_x = dev.htod_sync_copy(&x)?;
            let mut d_xh = unsafe { dev.alloc::<u16>(mm * m.k) }?;
            let mut d_y = unsafe { dev.alloc::<u16>(mm * m.n) }?;
            // stage xh once (excluded from timing, same as the chain benches)
            unsafe {
                f_suh
                    .clone()
                    .launch_on_stream(
                        &stream,
                        LaunchConfig {
                            grid_dim: ((mm * m.k / 128) as u32, 1, 1),
                            block_dim: (32, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (&d_x, &d_suh, &mut d_xh, m.k as i32),
                    )
                    .unwrap();
            }
            dev.synchronize().unwrap();
            let mut launch = || unsafe {
                f_wide
                    .clone()
                    .launch_on_stream(
                        &stream,
                        LaunchConfig {
                            grid_dim: (
                                ((mm + 16 * mt - 1) / (16 * mt)) as u32,
                                (m.n / 128) as u32,
                                1,
                            ),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            &d_tr,
                            &d_xh,
                            &mut d_y,
                            mm as i32,
                            m.k as i32,
                            m.n as i32,
                            m.bits as i32,
                            mt as i32,
                        ),
                    )
                    .unwrap();
            };
            for _ in 0..3 {
                launch();
            }
            dev.synchronize().unwrap();
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    launch();
                }
                dev.synchronize().unwrap();
                best = best.min(t0.elapsed().as_secs_f64() / iters as f64);
            }
            let gbs = m.trellis.n_bytes as f64 / best / 1e9;
            println!(
                "  {:<10} b={} K={:<5} N={:<6}  M={:<5} {:>7.1} GB/s  {:>9.0} rows/s  {:>7.2} ms",
                label,
                m.bits,
                m.k,
                m.n,
                mm,
                gbs,
                mm as f64 / best,
                best * 1e3
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 2b. Cross-chain mode: read x [M,K] fp16-u16 from --x-in, run the full chain
// (had_suh -> fused GEMM -> had_svh) on ONE module, write y [M,N] u16 to --y-out.
// The python driver (a3c_xchain.py) builds x from real embeddings on their side
// and compares rel-L2 < 1e-3 (G-A3-2 shape).
// ---------------------------------------------------------------------------
pub fn probe_xchain(dir: &str, x_path: &str, y_path: &str, module: &str) -> Result<()> {
    let pack = Exl3Pack::open(dir)?;
    let dev = dev0()?;
    load_module(&dev)?;
    let m = pack
        .modules
        .iter()
        .find(|mm| mm.name == module)
        .ok_or_else(|| anyhow::anyhow!("module {module} not in pack"))?
        .clone();
    let xbits = std::fs::read(x_path).with_context(|| format!("read {x_path}"))?;
    let x: Vec<u16> = xbits
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    if x.len() % m.k != 0 {
        bail!("x-in is {} u16, not a multiple of K={}", x.len(), m.k);
    }
    let rows = x.len() / m.k;
    let suh = le_f16_bits(&m.suh.read_bytes(&pack.dir)?);
    let svh = le_f16_bits(&m.svh.read_bytes(&pack.dir)?);
    let raw = m.trellis.read_bytes(&pack.dir)?;
    let tr_bits: Vec<u16> = raw
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let mut chain = run_chain(&dev, &tr_bits, &suh, &svh, &x, rows, m.k, m.n, m.bits)?;
    let y = dev.dtoh_sync_copy(&chain.d_y)?;
    let mut out = Vec::with_capacity(y.len() * 2);
    for v in &y {
        out.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(y_path, &out).with_context(|| format!("write {y_path}"))?;
    println!(
        "XCHAIN_WRITE module={module} bits={} K={} N={} rows={} y={}",
        m.bits, m.k, m.n, rows, y_path
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 3. Standalone bench: per-class GB/s at widths, vs 238 roofline / 79-83 naive floor.
// ---------------------------------------------------------------------------
pub fn bench_gemm(
    dir: &str,
    widths: &[usize],
    iters: usize,
    grouped_e: usize,
    l2_alias_mb: usize,
) -> Result<()> {
    let pack = Exl3Pack::open(dir)?;
    let dev = dev0()?;
    load_module(&dev)?;

    println!(
        "=== EXL3 weight-path GEMM bench ({} iters/rep, best of 3; roofline 238 GB/s, naive floor 79-83 GB/s) ===",
        iters
    );
    for m in pick_modules(&pack) {
        let raw = m.trellis.read_bytes(&pack.dir)?;
        let tr_bits: Vec<u16> = raw
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        let suh = le_f16_bits(&m.suh.read_bytes(&pack.dir)?);
        let svh = le_f16_bits(&m.svh.read_bytes(&pack.dir)?);
        let label = m
            .name
            .rsplit('.')
            .take(2)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join(".");
        for wd in widths {
            let x = synth_x(*wd, m.k, 0xBEEF);
            let mut chain =
                run_chain(&dev, &tr_bits, &suh, &svh, &x, *wd, m.k, m.n, m.bits)?;
            let ref stream = fork_blocking_stream(&dev);
            let gf = |n: &str| {
                dev.get_func(MODULE, n)
                    .ok_or_else(|| anyhow::anyhow!("fn {n} missing"))
            };
            let f_gemm = gf("exl3_hmma_gemm")?;
            let f_suh = gf("exl3_had_suh")?;
            let f_svh = gf("exl3_had_svh")?;
            let cfg_g = LaunchConfig {
                grid_dim: ((m.n / 128) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let cfg_v = LaunchConfig {
                grid_dim: ((*wd * m.k / 128) as u32, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            let cfg_o = LaunchConfig {
                grid_dim: ((*wd * m.n / 128) as u32, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            // warmup + best-of-3 rounds of `iters`
            for _ in 0..3 {
                bench_launch(stream, &f_suh, &f_gemm, &f_svh, &mut chain, m.k, m.n, *wd, false, cfg_g, cfg_v, cfg_o);
            }
            dev.synchronize().unwrap();
            let mut best_g = f64::INFINITY;
            let mut best_c = f64::INFINITY;
            for _ in 0..3 {
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    bench_launch(stream, &f_suh, &f_gemm, &f_svh, &mut chain, m.k, m.n, *wd, false, cfg_g, cfg_v, cfg_o);
                }
                dev.synchronize().unwrap();
                best_g = best_g.min(t0.elapsed().as_secs_f64() / iters as f64);

                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    bench_launch(stream, &f_suh, &f_gemm, &f_svh, &mut chain, m.k, m.n, *wd, true, cfg_g, cfg_v, cfg_o);
                }
                dev.synchronize().unwrap();
                best_c = best_c.min(t0.elapsed().as_secs_f64() / iters as f64);
            }
            let tr_bytes = (m.trellis.n_bytes) as f64;
            let aux_bytes = (2 * *wd * m.k + 2 * *wd * m.n) as f64; // x + xh read, yraw w + y w/r
            println!(
                "  {:<20} b={} W={:<2}  gemm {:>7.3} ms  {:>7.1} GB/s (trellis) | chain {:>7.3} ms  {:>7.1} GB/s",
                label,
                m.bits,
                wd,
                best_g * 1e3,
                tr_bytes / best_g / 1e9,
                best_c * 1e3,
                tr_bytes / best_c / 1e9
            );
            let _ = aux_bytes;
            if m.class == TensorClass::LmHead {
                // W3/LMH: the persistent kernel on the full head at this width
                bench_lmh(&dev, &chain.d_tr, &chain.d_svh, m.k, m.n, m.bits, *wd, iters, tr_bytes, "lm_head")?;
            }
        }
        if m.class == TensorClass::LmHead {
            // W3/LMH: the 65,536-id draft slice (the MTP draft pass's head)
            let (trd, nd) = slab_trellis(&raw, m.k / 16, m.n / 16, m.bits, Some(65536));
            let trd16: Vec<u16> = trd.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            let d_draft = dev.htod_sync_copy(&trd16)?;
            let d_svh_draft = dev.htod_sync_copy(&svh[..nd])?;
            for wd in widths {
                bench_lmh(&dev, &d_draft, &d_svh_draft, m.k, nd, m.bits, *wd, iters,
                          (trd.len()) as f64, "draft 65536")?;
            }
        }
    }
    // ---- grouped-expert bench (the MoE serving form): E experts, one launch ----
    // All experts of a group share the shape (gate/up: 2560->640 b3, down: 640->2560 b3).
    // xh here is synthetic activation content (per-expert suh handling is S-A3-d scope —
    // noted in the bank); the bench measures the weight-streaming path: bytes / time.
    let ge = if grouped_e > 0 { grouped_e } else { 512 };
    for suffix in ["gate_proj", "down_proj"] {
        let mods: Vec<ModuleMeta> = pack
            .modules
            .iter()
            .filter(|m| m.class == TensorClass::Expert && m.bits == 3 && m.name.ends_with(suffix))
            .take(ge)
            .cloned()
            .collect();
        if mods.len() < 2 {
            continue;
        }
        let m0 = &mods[0];
        // Single concatenated trellis buffer + per-expert ELEMENT offsets (no raw
        // pointer table; one allocation, contiguous streaming).
        let mut concat: Vec<u16> = Vec::new();
        let mut offs: Vec<u64> = Vec::with_capacity(mods.len());
        for m in &mods {
            let raw = m.trellis.read_bytes(&pack.dir)?;
            offs.push(concat.len() as u64);
            concat.extend(raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])));
        }
        let d_base = dev.htod_sync_copy(&concat)?;
        // S-A3-f-a X3 (L2-resident discriminator): --exl3-l2-alias [MB] folds the expert
        // offset table so every CTA reads one of the first alias_e trellis experts — the
        // working set (~16 MB default) fits L2, so the trellis loads stop paying DRAM
        // latency. Kernel, grid and launch config are UNCHANGED; only the offs CONTENT
        // differs. Under alias the GB/s figure is by construction NOT DRAM traffic; the
        // decision metric is the 16x16 block rate (printed on the [L2-ALIAS] line).
        let d_offs = if l2_alias_mb > 0 {
            let alias_e = ((l2_alias_mb * 1024 * 1024) / m0.trellis.n_bytes).clamp(1, mods.len());
            let offs_alias: Vec<u64> = (0..mods.len()).map(|e| offs[e % alias_e]).collect();
            println!(
                "  [L2-ALIAS] {} experts -> first {} (footprint {:.1} MB); GB/s below is NOT DRAM traffic",
                mods.len(),
                alias_e,
                alias_e as f64 * m0.trellis.n_bytes as f64 / 1e6
            );
            dev.htod_sync_copy(&offs_alias)?
        } else {
            dev.htod_sync_copy(&offs)?
        };
        let tr_bytes_total = (mods.len() * m0.trellis.n_bytes) as f64;
        println!(
            "  grouped {:>9} x{}  b={} K={:<5} N={:<6}  ({:.1} MB/launch)",
            suffix,
            mods.len(),
            m0.bits,
            m0.k,
            m0.n,
            tr_bytes_total / 1e6
        );
        for wd in widths {
            let x = synth_x(*wd, m0.k, 0xBEEF);
            let d_xh = dev.htod_sync_copy(&x)?;
            let mut d_y = unsafe { dev.alloc::<u16>(mods.len() * wd * m0.n) }?;
            let f_g = dev
                .get_func(MODULE, "exl3_hmma_gemm_grouped")
                .ok_or_else(|| anyhow::anyhow!("fn exl3_hmma_gemm_grouped missing"))?;
            let stream = fork_blocking_stream(&dev);
            let mut launch = || unsafe {
                f_g.clone()
                    .launch_on_stream(
                        &stream,
                        LaunchConfig {
                            grid_dim: ((mods.len() * m0.n / 128) as u32, 1, 1),
                            block_dim: (256, 1, 1),
                            // S-A3-n F3: the A-once body's smem (mirrors exl3_a1_fits)
                            shared_mem_bytes: {
                                let b = *wd * (m0.k + 8) * 2;
                                if *wd <= 16 && m0.k % 8 == 0 && b <= 48 * 1024 { b as u32 } else { 0 }
                            },
                        },
                        (
                            &d_base,
                            &d_offs,
                            &d_xh,
                            &mut d_y,
                            *wd as i32,
                            m0.k as i32,
                            m0.n as i32,
                            m0.bits as i32,
                        ),
                    )
                    .unwrap();
            };
            for _ in 0..3 {
                launch();
            }
            dev.synchronize().unwrap();
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    launch();
                }
                dev.synchronize().unwrap();
                best = best.min(t0.elapsed().as_secs_f64() / iters as f64);
            }
            // mapping sanity: expert 0 and expert 1 row 0 must differ (different weights).
            // Skipped under L2-ALIAS: experts alias to shared weights by construction.
            let y0 = dev.dtoh_sync_copy(&d_y)?;
            let n0 = m0.n;
            let same = y0[..n0] == y0[n0 * *wd..n0 * *wd + n0];
            if l2_alias_mb > 0 {
                // The X3 decision metric: 16x16 trellis blocks decoded per second.
                let blocks = (mods.len() * m0.n * m0.k / 256) as f64;
                println!(
                    "    [L2-ALIAS] W={:<2} {:>7.3} ms  block rate {:>6.3} G blocks/s  (mapping-check skipped)",
                    wd,
                    best * 1e3,
                    blocks / best / 1e9
                );
            } else {
                println!(
                    "    W={:<2} gemm {:>7.3} ms  {:>7.1} GB/s (aggregate trellis, {} experts)  mapping-check: {}",
                    wd,
                    best * 1e3,
                    tr_bytes_total / best / 1e9,
                    mods.len(),
                    if same { "FAIL (expert rows identical!)" } else { "ok (expert rows differ)" }
                );
                if same {
                    bail!("grouped mapping check FAILED on {suffix}");
                }
            }
        }
    }
    println!("EXL3-GEMM-BENCH: done");
    Ok(())
}

/// W3/LMH bench: old exl3_hmma_gemm (and + exl3_had_svh) vs the persistent kernel (y_raw, and
/// fused -> logits, every ring depth) at one m. Back-to-back launches on one stream, 3 warmups,
/// best of 3 x `iters`. GB/s = trellis bytes / time (the 238 GB/s floor is the target).
#[allow(clippy::too_many_arguments)]
fn bench_lmh(dev: &std::sync::Arc<CudaDevice>, d_tr: &CudaSlice<u16>, d_svh: &CudaSlice<u16>,
             k: usize, n: usize, bits: usize, m: usize, iters: usize, tr_bytes: f64, label: &str)
             -> Result<()> {
    let stream = fork_blocking_stream(dev);
    let x = synth_x(m, k, 0xBEEF);
    let d_xh = dev.htod_sync_copy(&x)?;
    let mut d_yr = dev.htod_sync_copy(&vec![0u16; m * n])?;
    let mut d_y = dev.htod_sync_copy(&vec![0u16; m * n])?;
    let f_old = dev.get_func(MODULE, "exl3_hmma_gemm").context("exl3_hmma_gemm")?;
    let f_svh = dev.get_func(MODULE, "exl3_had_svh").context("exl3_had_svh")?;
    let cfg_g = LaunchConfig { grid_dim: ((n / 128) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    let cfg_o = LaunchConfig { grid_dim: ((m * n / 128) as u32, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
    let (p_tr, p_xh, p_svh) = (*d_tr.device_ptr(), *d_xh.device_ptr(), *d_svh.device_ptr());
    let (p_yr, p_y) = (*d_yr.device_ptr(), *d_y.device_ptr());
    // variant: 0 = old gemm, 1 = old gemm + had_svh, 2.. = LMH (nst, fuse)
    let mut variants: Vec<(String, usize, bool)> = vec![("old gemm".into(), 0, false), ("old gemm+svh".into(), 0, true)];
    let nsts: &[usize] = if bits == 5 { &[4, 6, 8] } else { &[6] };
    for &nst in nsts {
        variants.push((format!("LMH s{nst}"), nst, false));
        variants.push((format!("LMH s{nst} fused"), nst, true));
    }
    let mut line = format!("  {label:<12} b={bits} W={m:<2}");
    for (name, nst, flag) in &variants {
        let mut run = |_: ()| -> Result<()> {
            if *nst == 0 {
                unsafe {
                    f_old.clone().launch_on_stream(&stream, cfg_g,
                        (d_tr, &d_xh, &mut d_yr, m as i32, k as i32, n as i32, bits as i32))?;
                    if *flag {
                        f_svh.clone().launch_on_stream(&stream, cfg_o, (&d_yr, d_svh, &mut d_y, n as i32))?;
                    }
                }
            } else {
                let out = if *flag { p_y } else { p_yr };
                if !crate::exl3_forward::lmh_gemm_launch(dev, stream.stream, p_tr, p_xh, out, p_svh,
                                                          m, k, n, bits, *nst, *flag)? {
                    bail!("W3/LMH bench: launcher refused m={m} K={k} N={n} b={bits} nst={nst}");
                }
            }
            Ok(())
        };
        for _ in 0..3 { run(())?; }
        dev.synchronize()?;
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let t0 = std::time::Instant::now();
            for _ in 0..iters { run(())?; }
            dev.synchronize()?;
            best = best.min(t0.elapsed().as_secs_f64() / iters as f64);
        }
        let gbs = tr_bytes / best / 1e9;
        line.push_str(&format!(" | {name} {:.3} ms {:.1} GB/s ({:.1}%)", best * 1e3, gbs, gbs / 238.0 * 100.0));
    }
    let _ = p_y;
    println!("{line}");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn bench_launch(
    stream: &CudaStream,
    f_suh: &cudarc::driver::CudaFunction,
    f_gemm: &cudarc::driver::CudaFunction,
    f_svh: &cudarc::driver::CudaFunction,
    c: &mut Chain,
    k: usize,
    n: usize,
    w: usize,
    full_chain: bool,
    cfg_g: LaunchConfig,
    cfg_v: LaunchConfig,
    cfg_o: LaunchConfig,
) {
    if full_chain {
        unsafe {
            f_suh
                .clone()
                .launch_on_stream(stream, cfg_v, (&c.d_x, &c.d_suh, &mut c.d_xh, k as i32))
                .unwrap();
        }
    }
    unsafe {
        f_gemm
            .clone()
            .launch_on_stream(stream, cfg_g, (&c.d_tr, &c.d_xh, &mut c.d_yraw, w as i32, k as i32, n as i32, c.bits as i32))
            .unwrap();
    }
    if full_chain {
        unsafe {
            f_svh
                .clone()
                .launch_on_stream(stream, cfg_o, (&c.d_yraw, &c.d_svh, &mut c.d_y, n as i32))
                .unwrap();
        }
    }
}

/// S-A3-d: module loader for the offline forward probe (same .ptx + registry).
pub(crate) fn load_module_pub(dev: &std::sync::Arc<CudaDevice>) -> Result<()> {
    load_module(dev)
}

/// TP-I3 (b) (`--probe-exl3-hcw4`): the EXL3-HC-W4 section of --probe-exl3-binv alone (synthetic,
/// no pack): xq_hc_w4 bitwise vs the p4c hc path over m = 1..8 x grids x flags + the standalone
/// DRAM-cold timing table.
pub fn probe_hcw4() -> Result<()> {
    let dev = dev0()?;
    load_module(&dev)?;
    probe_hc_w4(&dev)
}

// ---------------------------------------------------------------------------
// API-parity G2 gate: `--probe-sampler`. Distribution-exactness of xq_sample_rows vs the
// host-computed target distribution (same semantics: keys -> top-k with ties -> min_p floor ->
// top-p with boundary ties -> softmax/T), on a fixed synthetic fp16 row at the real vocab.
// Chi-square over (tokens with expected >= 20) + 64 index-range buckets for the rest; hard
// fail on any sample outside the support; a second seed as the control; determinism check.
fn skey(b: u16) -> u32 { if b & 0x8000 != 0 { (!(b as u32)) & 0xFFFF } else { (b as u32) | 0x8000 } }

fn sampler_target(lg: &[u16], t: f32, top_k: u32, top_p: f32, min_p: f32) -> Vec<f64> {
    let v = lg.len();
    let keys: Vec<u32> = lg.iter().map(|&b| skey(b)).collect();
    let val = |i: usize| f16::from_bits(lg[i]).to_f32();
    let kmax = *keys.iter().max().unwrap();
    let lmax = (0..v).find(|&i| keys[i] == kmax).map(val).unwrap();
    let tau = if top_k > 0 && (top_k as usize) < v {
        let mut ks = keys.clone();
        ks.sort_unstable_by(|a, b| b.cmp(a));
        ks[top_k as usize - 1]
    } else { 0 };
    let t = t.max(1e-6);
    let mfloor = if min_p > 0.0 { lmax + t * min_p.ln() } else { f32::NEG_INFINITY };
    let cand = |i: usize| keys[i] >= tau && val(i) >= mfloor;
    let w = |i: usize| (((val(i) - lmax) / t) as f64).exp();
    let s: f64 = (0..v).filter(|&i| cand(i)).map(w).sum();
    let mut thr = tau;
    if top_p < 1.0 {
        // mass by key, descending; theta = largest key with cum mass >= top_p * S
        let mut by: std::collections::BTreeMap<u32, f64> = Default::default();
        for i in 0..v { if cand(i) { *by.entry(keys[i]).or_insert(0.0) += w(i); } }
        let target = top_p as f64 * s;
        let mut cum = 0.0;
        for (&k, &m) in by.iter().rev() {
            cum += m;
            if cum >= target { thr = thr.max(k); break; }
        }
    }
    let mut p: Vec<f64> = (0..v).map(|i| if cand(i) && keys[i] >= thr { w(i) } else { 0.0 }).collect();
    let z: f64 = p.iter().sum();
    for x in p.iter_mut() { *x /= z; }
    p
}

// ---------------------------------------------------------------------------
// w5/PFIX: the chunked-prefill attention kernels against the SWEEP xq_attn_prefill_q (one block
// per head, token loop over the decode softmax — a different kernel class, causally correct) on
// synthetic data: no model, f32 KV, 4 q heads / 2 kv heads, partial rope 64, chunk starts that put
// block rows on, across and deep past tile boundaries. Both head dims:
//   hd=128: xq_attn_prefill_flash (the scalar BT=16 shape; no host caller since 2463afc)
//           xq_attn_prefill_flash128c (--pq8-flash=causal's hd=128 twin)
//   hd=256: xq_attn_prefill_flash256 / flash256v (the default; must equal flash256 BITWISE)
//           xq_attn_prefill_flash256c (--pq8-flash=causal)
// PASS = every CAUSAL twin within f16 output rounding of the sweep (max |d| <= 4e-3 on |out| < 1)
// with every row written (outputs NaN-poisoned before each launch), and flash256v == flash256 bit
// for bit. The scalar kernels' diagonal-tile staging defect is REPORTED (max |d| vs the sweep):
// the host model (exl3_forward.rs flash256_stage_tests) predicts it far above the causal twins'.
//   ./gb10_inference --probe-exl3-flashpf
// ---------------------------------------------------------------------------
pub fn probe_flash_prefill() -> Result<()> {
    const NH: usize = 4;
    const NKV: usize = 2;
    const RDIM: usize = 64;
    const MAXP: usize = 512;
    const TOL: f32 = 4e-3;
    let dev = dev0()?;
    load_module(&dev)?;
    let mut st: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut rnd = move |amp: f32| -> f32 {
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        (((st >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0) * amp
    };
    let h16 = |x: f32| f16::from_f32(x).to_bits();
    let mut fail = 0usize;
    let mut worst_causal = 0f32;
    let mut worst_scalar = 0f32;
    // (hd, rows per block, scalar kernel, causal twin)
    for &(hd, bt, scalar, causal) in &[(128usize, 16usize, "xq_attn_prefill_flash", "xq_attn_prefill_flash128c"),
                                       (256, 8, "xq_attn_prefill_flash256", "xq_attn_prefill_flash256c")] {
        // slot 0 of the cache: [k|v][kvh][pos][hd] f32 (fmt bits of max_pos = 0 = f32)
        let kv: Vec<f32> = (0..2 * NKV * MAXP * hd).map(|_| rnd(1.0)).collect();
        let d_kv = dev.htod_sync_copy(&kv)?;
        let qnkn: Vec<u16> = (0..2 * hd).map(|_| h16(rnd(0.2))).collect();
        let d_qnkn = dev.htod_sync_copy(&qnkn)?;
        for &(pos0, c) in &[(0usize, 48usize), (8, 40), (13, 37), (16, 33), (21, 64), (40, 100), (256, 33), (400, 64)] {
            assert!(pos0 + c.div_ceil(bt) * bt + 32 <= MAXP);
            let qg: Vec<u16> = (0..c * NH * 2 * hd).map(|_| h16(rnd(1.0))).collect();
            let cs: Vec<f32> = (0..c).flat_map(|_| {
                let ang: Vec<f32> = (0..RDIM).map(|_| rnd(3.1)).collect();
                ang.iter().map(|a| a.cos()).chain(ang.iter().map(|a| a.sin())).collect::<Vec<f32>>()
            }).collect();
            let d_qg = dev.htod_sync_copy(&qg)?;
            let d_cs = dev.htod_sync_copy(&cs)?;
            let n = c * NH * hd;
            let nh_nkv = (NKV | (NH << 16)) as i32;
            let hd_rdim = (hd | (RDIM << 16)) as i32;
            let slot_pos0 = (pos0 << 12) as i32;
            let run = |name: &str, grid: (u32, u32, u32), block: u32| -> Result<Vec<f32>> {
                let f = dev.get_func(MODULE, name).with_context(|| format!("{name} missing"))?;
                let mut out = dev.htod_sync_copy(&vec![0xFFFFu16; n])?;   // f16 NaN poison
                unsafe {
                    f.launch(LaunchConfig { grid_dim: grid, block_dim: (block, 1, 1), shared_mem_bytes: 0 },
                             (&mut out, &d_qg, &d_kv, &d_qnkn, &d_cs, nh_nkv, hd_rdim, MAXP as i32,
                              slot_pos0, c as i32, 1e-6f32))?;
                }
                dev.synchronize()?;
                Ok(dev.dtoh_sync_copy(&out)?.iter().map(|&b| f16::from_bits(b).to_f32()).collect())
            };
            let fgrid = ((c.div_ceil(bt)) as u32, NH as u32, 1);
            let reference = run("xq_attn_prefill_q", (NH as u32, 1, 1), hd as u32)?;
            let sc = run(scalar, fgrid, (bt * 32) as u32)?;
            let ca = run(causal, fgrid, (bt * 32) as u32)?;
            let unwritten = |x: &[f32]| x.iter().filter(|v| v.is_nan()).count();
            let maxd = |x: &[f32]| x.iter().zip(&reference).map(|(a, b)| (a - b).abs()).fold(0f32, |m, d| if d.is_nan() { f32::INFINITY } else { m.max(d) });
            let (nr, nc) = (unwritten(&reference), unwritten(&ca));
            if nr > 0 || reference.iter().all(|&v| v == 0.0) {
                bail!("EXL3-FLASHPF FAIL: implausible sweep reference (hd {hd} pos0 {pos0} c {c}: {nr} NaN)");
            }
            let (dsc, dca) = (maxd(&sc), maxd(&ca));
            worst_causal = worst_causal.max(dca);
            worst_scalar = worst_scalar.max(dsc);
            let ok = nc == 0 && dca <= TOL;
            let mut line = format!("EXL3-FLASHPF: hd {hd} pos0 {pos0:>3} c {c:>3}: vs sweep max|d| {} {dca:.2e}{} | {} {dsc:.2e} (scalar staging)",
                                   causal.trim_start_matches("xq_attn_prefill_"),
                                   if nc > 0 { format!(" ({nc} rows' halves unwritten)") } else { String::new() },
                                   scalar.trim_start_matches("xq_attn_prefill_"));
            let mut vok = true;
            if hd == 256 {
                let vv = run("xq_attn_prefill_flash256v", fgrid, 256)?;
                let nd = vv.iter().zip(&sc).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                vok = nd == 0;
                line += &format!(" | flash256v vs flash256 differ {nd} of {n}");
            }
            println!("{line}{}", if ok && vok { "" } else { "  <-- FAIL" });
            if !(ok && vok) { fail += 1; }
        }
    }
    println!("EXL3-FLASHPF: worst causal-twin max|d| vs sweep {worst_causal:.2e} (tol {TOL:.0e}); worst scalar-staging \
              max|d| {worst_scalar:.2e}{}", if worst_scalar > 10.0 * worst_causal.max(1e-4) { " (defect visible)" }
              else { " (defect NOT visible — the host staging model disagrees with the GPU)" });
    if fail > 0 { bail!("EXL3-FLASHPF FAIL: {fail} case(s)"); }
    println!("EXL3-FLASHPF: PASS (flash128c + flash256c match the sweep within f16 rounding; flash256v == flash256 bitwise)");
    Ok(())
}

pub fn probe_sampler() -> Result<()> {
    let dev = dev0()?;
    let ptx = Ptx::from_src(std::fs::read_to_string("src/ptx/exl3_bench.ptx")
        .context("src/ptx/exl3_bench.ptx missing")?);
    dev.load_ptx(ptx, "exl3_sampler_probe", &["xq_argmax", "xq_sample_rows"])?;
    let f = dev.get_func("exl3_sampler_probe", "xq_sample_rows").context("xq_sample_rows missing")?;
    let v: usize = 248_320;
    // synthetic row: a Gaussian bulk (sd 2) + a 40-token head spread over [6, 11] so top-k/top-p
    // cut inside a populated region; fp16 rounding makes real ties.
    let mut st: u64 = 0x9E3779B97F4A7C15;
    let mut rnd = || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; (st >> 11) as f64 / (1u64 << 53) as f64 };
    let mut row: Vec<u16> = (0..v).map(|_| {
        let (u1, u2) = (rnd().max(1e-12), rnd());
        let g = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        f16::from_f64(g * 2.0).to_bits()
    }).collect();
    for j in 0..40usize {
        let idx = (j * 6151 + 17) % v;
        row[idx] = f16::from_f64(6.0 + 5.0 * rnd()).to_bits();
    }
    let rows_per_launch = 512usize;
    let launches = 400usize;
    let mut lg_dev = dev.htod_sync_copy(&row.iter().cycle().take(v * rows_per_launch).cloned().collect::<Vec<u16>>())?;
    let _ = &mut lg_dev;
    let mut ids_dev = dev.alloc_zeros::<i32>(rows_per_launch)?;
    // (T, top_k, top_p, min_p)
    let cfgs: [(f32, u32, f32, f32); 5] = [
        (0.7, 20, 0.8, 0.0),   // our server default
        (0.8, 0, 1.0, 0.08),   // the rival shim default (min_p)
        (1.0, 0, 1.0, 0.0),    // pure temperature, full vocab
        (0.6, 0, 0.95, 0.0),   // top-p over the full vocab
        (1.3, 50, 1.0, 0.0),   // top-k only, hot
    ];
    let mut all_ok = true;
    for &(t, k, p, mp) in cfgs.iter() {
        let target = sampler_target(&row, t, k, p, mp);
        let support = target.iter().filter(|&&x| x > 0.0).count();
        for seed in [0x1234_5678_9ABCu64, 0x0DDB_A11u64] {
            let mut counts = vec![0u64; v];
            let mut first: Vec<i32> = Vec::new();
            for l in 0..launches {
                let mut samp = vec![0u32; rows_per_launch * 8];
                for r in 0..rows_per_launch {
                    let c = (l * rows_per_launch + r) as u32;
                    samp[r * 8..r * 8 + 8].copy_from_slice(&[1, t.to_bits(), p.to_bits(), k,
                        seed as u32, (seed >> 32) as u32, c, mp.to_bits()]);
                }
                let samp_dev = dev.htod_sync_copy(&samp)?;
                unsafe {
                    f.clone().launch(LaunchConfig { grid_dim: (rows_per_launch as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 },
                        (&mut ids_dev, &lg_dev, v as i32, &samp_dev))?;
                }
                let ids = dev.dtoh_sync_copy(&ids_dev)?;
                if l == 0 { first = ids.clone(); }
                for &id in &ids { counts[id as usize] += 1; }
            }
            // determinism: relaunch l=0 and compare
            {
                let mut samp = vec![0u32; rows_per_launch * 8];
                for r in 0..rows_per_launch {
                    samp[r * 8..r * 8 + 8].copy_from_slice(&[1, t.to_bits(), p.to_bits(), k,
                        seed as u32, (seed >> 32) as u32, r as u32, mp.to_bits()]);
                }
                let samp_dev = dev.htod_sync_copy(&samp)?;
                unsafe {
                    f.clone().launch(LaunchConfig { grid_dim: (rows_per_launch as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 },
                        (&mut ids_dev, &lg_dev, v as i32, &samp_dev))?;
                }
                let again = dev.dtoh_sync_copy(&ids_dev)?;
                if again != first { println!("  DETERMINISM FAIL"); all_ok = false; }
            }
            let n: u64 = counts.iter().sum();
            let outside: u64 = (0..v).filter(|&i| target[i] == 0.0).map(|i| counts[i]).sum();
            // chi-square bins
            let (mut chi2, mut df) = (0.0f64, 0i64);
            let mut bucket_e = vec![0.0f64; 64];
            let mut bucket_o = vec![0u64; 64];
            for i in 0..v {
                if target[i] == 0.0 { continue; }
                let e = target[i] * n as f64;
                if e >= 20.0 {
                    let d = counts[i] as f64 - e;
                    chi2 += d * d / e; df += 1;
                } else {
                    let b = i * 64 / v;
                    bucket_e[b] += e; bucket_o[b] += counts[i];
                }
            }
            for b in 0..64 {
                if bucket_e[b] >= 20.0 {
                    let d = bucket_o[b] as f64 - bucket_e[b];
                    chi2 += d * d / bucket_e[b]; df += 1;
                }
            }
            df -= 1;
            // Wilson-Hilferty: z = ((chi2/df)^(1/3) - (1 - 2/(9df))) / sqrt(2/(9df))
            let dff = df.max(1) as f64;
            let z = ((chi2 / dff).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * dff))) / (2.0 / (9.0 * dff)).sqrt();
            let tvd: f64 = 0.5 * (0..v).map(|i| (counts[i] as f64 / n as f64 - target[i]).abs()).sum::<f64>();
            let ok = outside == 0 && z < 3.5;
            all_ok &= ok;
            println!("  T={t} top_k={k} top_p={p} min_p={mp} seed={seed:#x}: n={n} support={support} chi2={chi2:.1} df={df} z={z:.2} tvd={tvd:.4} outside={outside} -> {}",
                     if ok { "PASS" } else { "FAIL" });
        }
    }
    println!("{}", if all_ok { "PROBE_SAMPLER_OK" } else { "PROBE_SAMPLER_FAIL" });
    if !all_ok { bail!("sampler probe failed"); }
    Ok(())
}

// ---------------------------------------------------------------------------
// WP15 gate: `--probe-penalties` (no model, no weights: synthetic fp16 rows at the real vocab).
// xq_pen_rows / xq_pen_draft vs a DENSE host transcription of the rival's two kernels
// (exllamav3_ext/generator/rep_pen.cu: every logit rewritten, float factors, float frequency
// sum in ascending position order). Asserts:
//   (a) every logit the window did not touch is bit-identical (the sparse rewrite is exact);
//   (b) touched logits equal the reference within 1 fp16 ulp (only the frequency sum's float
//       association and FMA contraction differ) — the exact / 1-ulp split is reported;
//   (c) SPECULATION EXACTNESS: verify row r (history + b, d0..d_{r-1} in flight) is bit-identical
//       to a plain single-row launch whose committed history already holds those tokens;
//   (d) the ring commit: every penalized row wrote its input token at its position; flag-0 rows
//       touch neither their logits nor the ring;
//   (e) the draft mirror (pruned 65,536-id slice, in-chain drafts as past) against the same
//       reference.
fn pen_factors(past: &[u32], vlim: usize, s: usize, d: usize, freq: f32)
               -> (std::collections::HashMap<u32, f32>, std::collections::HashMap<u32, f32>,
                   std::collections::HashMap<u32, f32>) {
    // (rep factor max, presence factor max, frequency sum) per token — rep_pen.cu verbatim
    let (mut rf, mut pf, mut fs) = (std::collections::HashMap::new(), std::collections::HashMap::new(),
                                    std::collections::HashMap::new());
    let pl = past.len() as i64;
    for (i, &t) in past.iter().enumerate() {
        if (i as i64) <= pl - s as i64 - d as i64 { continue; }
        if t as usize >= vlim { continue; }
        let dist = (pl - i as i64) as f32;
        let (sf, df) = (s as f32, d as f32);
        let r = if dist <= sf { 1.0f32 } else {
            (if d > 0 { 1.0 - (dist - sf) / df } else { 1.0 }).clamp(0.0, 1.0)
        };
        let e = rf.entry(t).or_insert(0.0f32); if r > *e { *e = r; }
        let p = (if d > 0 { 1.0 - (dist - sf) / df } else { 1.0f32 }).clamp(0.0, 1.0);
        *fs.entry(t).or_insert(0.0f32) += p * freq;
        let e = pf.entry(t).or_insert(0.0f32); if p > *e { *e = p; }
    }
    (rf, pf, fs)
}

fn pen_ref_row(row: &mut [u16], past: &[u32], vlim: usize, rep: f32, pres: f32, freq: f32, s: usize, d: usize) {
    let (rf, pf, fs) = pen_factors(past, vlim, s, d, freq);
    for j in 0..vlim {
        let mut v = f16::from_bits(row[j]).to_f32();
        if rep != 1.0 {
            let f = rf.get(&(j as u32)).copied().unwrap_or(0.0);
            let w = if v > 0.0 { v / rep } else { v * rep };
            let fr = f + 1e-30f32;
            let f1 = (1.0f32 - fr) + 1e-30f32;
            v = v * f1 + w * fr;
        }
        if pres != 0.0 || freq != 0.0 {
            v -= fs.get(&(j as u32)).copied().unwrap_or(0.0);
            v -= pf.get(&(j as u32)).copied().unwrap_or(0.0) * pres;
        }
        if v > 65504.0 { v = 65504.0 } else if v < -65504.0 { v = -65504.0 }
        row[j] = f16::from_f32(v).to_bits();
    }
}

fn f16_ulps(a: u16, b: u16) -> u32 {
    let o = |x: u16| -> i32 { if x & 0x8000 != 0 { -((x & 0x7FFF) as i32) } else { x as i32 } };
    (o(a) - o(b)).unsigned_abs()
}

pub fn probe_penalties() -> Result<()> {
    use crate::exl3_forward::{PEN_RING, PEN_RING_LOG2};
    let dev = dev0()?;
    let ptx = Ptx::from_src(std::fs::read_to_string("src/ptx/exl3_bench.ptx")
        .context("src/ptx/exl3_bench.ptx missing")?);
    dev.load_ptx(ptx, "exl3_pen_probe", &["xq_pen_rows", "xq_pen_draft"])?;
    let f_rows = dev.get_func("exl3_pen_probe", "xq_pen_rows").context("xq_pen_rows missing")?;
    let f_draft = dev.get_func("exl3_pen_probe", "xq_pen_draft").context("xq_pen_draft missing")?;
    let v: usize = 248_320;
    let vd: usize = 65_536;
    let nslot = 8usize;
    let mut st: u64 = 0xD1B5_4A32_D192_ED03;
    let mut rnd = move || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; st };
    // a skewed token stream (heavy repeats, both spans' edges, ids >= the draft slice)
    let tok = |r: &mut dyn FnMut() -> u64| -> u32 {
        let x = r();
        match x % 10 {
            0..=4 => (x >> 8) as u32 % 300,                         // hot, repeats a lot
            5 | 6 => 4090 + ((x >> 8) as u32 % 12),                 // straddles a 4096 span edge
            7 => (v as u32 - 1) - ((x >> 8) as u32 % 5),            // the last span's tail
            8 => 65_530 + ((x >> 8) as u32 % 12),                   // the draft slice edge
            _ => (x >> 8) as u32 % v as u32,
        }
    };
    let gauss_row = |r: &mut dyn FnMut() -> u64| -> Vec<u16> {
        (0..v).map(|_| {
            let u1 = ((r() >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
            let u2 = (r() >> 11) as f64 / (1u64 << 53) as f64;
            f16::from_f64(3.0 * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()).to_bits()
        }).collect()
    };
    // (rep, pres, freq, sustain, decay)
    let cfgs: [(f32, f32, f32, usize, usize); 6] = [
        (1.1, 0.0, 0.0, 1024, 1024),
        (1.0, 0.5, 0.3, 1024, 1024),
        (1.05, 1.5, 0.0, 1024, 1024),
        (0.9, 0.0, 0.2, 1024, 1024),  // rep < 1: accepted (owner decision), rewards repeats
        (1.3, -0.5, 0.25, 64, 0),     // no decay tier
        (1.2, 0.4, 0.4, 16, 8),       // short windows: the decay factors are fractional
    ];
    let pack = |c: &(f32, f32, f32, usize, usize)| -> [u32; 8] {
        let flag = (c.0 != 1.0) as u32 | (((c.1 != 0.0 || c.2 != 0.0) as u32) << 1);
        [flag, c.0.to_bits(), c.1.to_bits(), c.2.to_bits(), c.3 as u32, c.4 as u32, 0, 0]
    };
    let launch_rows = |logits: &mut CudaSlice<u16>, pen: &CudaSlice<u32>, toks: &CudaSlice<i32>,
                       slots: &CudaSlice<i32>, hist: &mut CudaSlice<i32>, m: usize| -> Result<()> {
        unsafe {
            f_rows.clone().launch(LaunchConfig { grid_dim: (v.div_ceil(4096) as u32, m as u32, 1),
                                                 block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                (logits, v as i32, pen, toks, slots, hist, PEN_RING_LOG2 as i32))?;
        }
        Ok(())
    };
    let mut all_ok = true;
    let (mut n_exact, mut n_ulp1) = (0usize, 0usize);
    for (ci, c) in cfgs.iter().enumerate() {
        // ---- (a)(b)(c)(d) verify layout: slot 3, positions p..p+5 (the ring has wrapped)
        let m = 6usize;
        let slot = 3usize;
        let p = 9_000 + ci * 37;
        let seq: Vec<u32> = (0..p + m).map(|_| tok(&mut rnd)).collect(); // committed [0,p) + in-flight
        let mut ring = vec![0i32; nslot * PEN_RING];
        for (q, &t) in seq[..p].iter().enumerate() { ring[slot * PEN_RING + (q & (PEN_RING - 1))] = t as i32; }
        let rows: Vec<u16> = (0..m).flat_map(|_| gauss_row(&mut rnd)).collect();
        let mut pen = vec![0u32; m * 8];
        for r in 0..m { pen[r * 8..r * 8 + 8].copy_from_slice(&pack(c)); }
        let toks: Vec<i32> = seq[p..p + m].iter().map(|&t| t as i32).collect();
        let slots: Vec<i32> = (0..m).flat_map(|r| [slot as i32, (p + r) as i32]).collect();
        let mut lg = dev.htod_sync_copy(&rows)?;
        let pen_d = dev.htod_sync_copy(&pen)?;
        let toks_d = dev.htod_sync_copy(&toks)?;
        let slots_d = dev.htod_sync_copy(&slots)?;
        let mut hist_d = dev.htod_sync_copy(&ring)?;
        launch_rows(&mut lg, &pen_d, &toks_d, &slots_d, &mut hist_d, m)?;
        let out = dev.dtoh_sync_copy(&lg)?;
        let hist_out = dev.dtoh_sync_copy(&hist_d)?;
        let (mut untouched_bad, mut big) = (0usize, 0usize);
        for r in 0..m {
            let mut refrow = rows[r * v..(r + 1) * v].to_vec();
            pen_ref_row(&mut refrow, &seq[..=p + r], v, c.0, c.1, c.2, c.3, c.4);
            let (rf, _, _) = pen_factors(&seq[..=p + r], v, c.3, c.4, c.2);
            for j in 0..v {
                let (o, e, i0) = (out[r * v + j], refrow[j], rows[r * v + j]);
                if !rf.contains_key(&(j as u32)) {
                    if o != i0 { untouched_bad += 1; }
                } else {
                    match f16_ulps(o, e) { 0 => n_exact += 1, 1 => n_ulp1 += 1, _ => big += 1 }
                }
            }
        }
        // (d) commits: row r wrote toks[r] at p + r; nothing else in the ring moved
        let mut ring_want = ring.clone();
        for r in 0..m { ring_want[slot * PEN_RING + ((p + r) & (PEN_RING - 1))] = toks[r]; }
        let ring_ok = hist_out == ring_want;
        // (c) speculation exactness: row r == a plain single-row launch with b, d0..d_{r-1} committed
        let mut spec_bad = 0usize;
        for r in 0..m {
            let mut ring2 = ring.clone();
            for q in p..p + r { ring2[slot * PEN_RING + (q & (PEN_RING - 1))] = seq[q] as i32; }
            let mut lg1 = dev.htod_sync_copy(&rows[r * v..(r + 1) * v].to_vec())?;
            let pen1 = dev.htod_sync_copy(&pack(c).to_vec())?;
            let toks1 = dev.htod_sync_copy(&vec![toks[r]])?;
            let slots1 = dev.htod_sync_copy(&vec![slot as i32, (p + r) as i32])?;
            let mut hist1 = dev.htod_sync_copy(&ring2)?;
            launch_rows(&mut lg1, &pen1, &toks1, &slots1, &mut hist1, 1)?;
            let o1 = dev.dtoh_sync_copy(&lg1)?;
            spec_bad += o1.iter().zip(&out[r * v..(r + 1) * v]).filter(|(a, b)| a != b).count();
        }
        // flag-0 rows (a plain batch with an unpenalized lane next to a penalized one)
        let mut pen_mix = vec![0u32; 2 * 8];
        pen_mix[8..16].copy_from_slice(&pack(c));
        let rows2: Vec<u16> = (0..2).flat_map(|_| gauss_row(&mut rnd)).collect();
        let mut lg2 = dev.htod_sync_copy(&rows2)?;
        let pen2 = dev.htod_sync_copy(&pen_mix)?;
        let toks2 = dev.htod_sync_copy(&vec![seq[0] as i32, seq[1] as i32])?;
        // lane A = slot 1 at pos 5 (lo clamps to 0), lane B = slot 6 at pos 12
        let slots2 = dev.htod_sync_copy(&vec![1i32, 5, 6, 12])?;
        let mut ring3 = vec![0i32; nslot * PEN_RING];
        for q in 0..12 { ring3[6 * PEN_RING + q] = seq[q] as i32; }
        for q in 0..5 { ring3[PEN_RING + q] = seq[20 + q] as i32; }
        let mut hist2 = dev.htod_sync_copy(&ring3)?;
        launch_rows(&mut lg2, &pen2, &toks2, &slots2, &mut hist2, 2)?;
        let o2 = dev.dtoh_sync_copy(&lg2)?;
        let h2 = dev.dtoh_sync_copy(&hist2)?;
        let flag0_ok = o2[..v] == rows2[..v] && h2[PEN_RING + 5] == ring3[PEN_RING + 5];
        let mut past_b: Vec<u32> = seq[..12].to_vec();
        past_b.push(seq[1]);
        let mut ref_b = rows2[v..].to_vec();
        pen_ref_row(&mut ref_b, &past_b, v, c.0, c.1, c.2, c.3, c.4);
        let short_big = o2[v..].iter().zip(&ref_b).filter(|(a, b)| f16_ulps(**a, **b) > 1).count();
        // ---- (e) draft mirror: meta {b, p, slot}, passes 0, 2, 4 over the 65,536-id slice
        let mut draft_big = 0usize;
        for &i in &[0usize, 2, 4] {
            let row = gauss_row(&mut rnd)[..vd].to_vec();
            let mut lgd = dev.htod_sync_copy(&row)?;
            let pen_d0 = dev.htod_sync_copy(&pack(c).to_vec())?;
            let meta = dev.htod_sync_copy(&vec![seq[p] as i32, p as i32, slot as i32, 0])?;
            let drafts: Vec<i32> = seq[p + 1..p + 6].iter().map(|&t| t as i32).collect();
            let dd = dev.htod_sync_copy(&drafts)?;
            let hd = dev.htod_sync_copy(&ring)?;
            unsafe {
                f_draft.clone().launch(LaunchConfig { grid_dim: (vd.div_ceil(4096) as u32, 1, 1),
                                                      block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                    (&mut lgd, vd as i32, &pen_d0, &meta, &dd, i as i32, &hd, PEN_RING_LOG2 as i32))?;
            }
            let od = dev.dtoh_sync_copy(&lgd)?;
            let mut refd = row.clone();
            pen_ref_row(&mut refd, &seq[..=p + i], vd, c.0, c.1, c.2, c.3, c.4);
            draft_big += od.iter().zip(&refd).filter(|(a, b)| f16_ulps(**a, **b) > 1).count();
        }
        let ok = untouched_bad == 0 && big == 0 && ring_ok && spec_bad == 0 && flag0_ok
            && short_big == 0 && draft_big == 0;
        all_ok &= ok;
        println!("  rep={} pres={} freq={} range={}+{}: untouched_changed={untouched_bad} >1ulp={big} \
                  spec_row_diffs={spec_bad} ring_commit={} flag0_untouched={} short_ctx>1ulp={short_big} \
                  draft>1ulp={draft_big} -> {}",
                 c.0, c.1, c.2, c.3, c.4, if ring_ok { "ok" } else { "BAD" }, if flag0_ok { "ok" } else { "BAD" },
                 if ok { "PASS" } else { "FAIL" });
    }
    println!("  penalized logits vs the rival transcription: exact={n_exact} 1-ulp={n_ulp1}");
    println!("{}", if all_ok { "PROBE_PENALTIES_OK" } else { "PROBE_PENALTIES_FAIL" });
    if !all_ok { bail!("penalty probe failed"); }
    Ok(())
}

/// A5 WP20-v1c: host emulation of the suh stage, bitwise (IEEE f32: Rust never contracts,
/// f32::mul_add is the fused, singly-rounded FFMA; half's f16::from_f32 is RN-even == F2FP).
///  - `pinned_block` transcribes the epilogue's wp20_suh_had (kernels/exl3_bench.cu).
///  - `src_block(conv)` transcribes xq_had_suh_multi's SOURCE (x*s, then exl3_had128 as
///    written) under each of the 81 first-stage contractions ptxas may pick (v1's convention
///    word); conv 0x55 is the one its served SASS executes (FMUL q = x1*s1; FFMA x0*s0 +- q).
/// Proves: (1) every contraction == the pinned sequence, bit for bit, on realistic, edge-case
/// and full-domain f16 inputs; (2) the root cause of p2's boot fallback: on v1's calibration
/// inputs all 81 conventions match the served kernel, so "exactly one match" never held.
#[cfg(test)]
mod wp20_suh_tests {
    use super::{wp20_synth, wp20_synth_edge, WP20_SUH_SEQ};
    use half::f16;

    const HNORM: f32 = 0.08838834764831845f32;

    fn f(b: u16) -> f32 { f16::from_bits(b).to_f32() }

    /// x0*s0 (+|-) x1*s1 under contraction c: 0 = both products rounded, 1 = first fused,
    /// 2 = second fused.
    fn pm(x0: f32, s0: f32, x1: f32, s1: f32, c: u32, sub: bool) -> f32 {
        match c {
            1 => { let q = x1 * s1; x0.mul_add(s0, if sub { -q } else { q }) }
            2 => (if sub { -x1 } else { x1 }).mul_add(s1, x0 * s0),
            _ => { let (p, q) = (x0 * s0, x1 * s1); if sub { p - q } else { p + q } }
        }
    }

    fn conv_of(ci: u32) -> u32 { (ci % 3) | ((ci / 3 % 3) << 2) | ((ci / 9 % 3) << 4) | ((ci / 27 % 3) << 6) }

    /// xq_had_suh_multi's source semantics (exl3_had128 as written) under contraction `conv`.
    fn src_block(x: &[u16], s: &[u16], conv: u32) -> [u16; 128] {
        let mut v = [[0f32; 4]; 32];
        for (lane, vl) in v.iter_mut().enumerate() {
            let xs = [0, 1, 2, 3].map(|j| f(x[lane * 4 + j]));
            let ss = [0, 1, 2, 3].map(|j| f(s[lane * 4 + j]));
            let c = |i: u32| (conv >> (2 * i)) & 3;
            let a = [pm(xs[0], ss[0], xs[1], ss[1], c(0), false), pm(xs[0], ss[0], xs[1], ss[1], c(1), true),
                     pm(xs[2], ss[2], xs[3], ss[3], c(2), false), pm(xs[2], ss[2], xs[3], ss[3], c(3), true)];
            *vl = [a[0] + a[2], a[1] + a[3], a[0] - a[2], a[1] - a[3]];     // stage len 2 (r, r+2)
        }
        for st in 2..7 {
            let mask = 1usize << (st - 2);
            let prev = v;
            for lane in 0..32 {
                for r in 0..4 {
                    let t = prev[lane ^ mask][r];
                    v[lane][r] = if lane & mask != 0 { t - prev[lane][r] } else { prev[lane][r] + t };
                }
            }
        }
        let mut o = [0u16; 128];
        for lane in 0..32 { for r in 0..4 { o[lane * 4 + r] = f16::from_f32(v[lane][r] * HNORM).to_bits(); } }
        o
    }

    /// The epilogue's wp20_suh_had, op for op (the served SASS form: FSEL +-v, FADD t + w).
    fn pinned_block(x: &[u16], s: &[u16]) -> [u16; 128] {
        let mut v = [[0f32; 4]; 32];
        for (lane, vl) in v.iter_mut().enumerate() {
            let xs = [0, 1, 2, 3].map(|j| f(x[lane * 4 + j]));
            let ss = [0, 1, 2, 3].map(|j| f(s[lane * 4 + j]));
            let q1 = xs[1] * ss[1];
            let q3 = xs[3] * ss[3];
            let (a0, a1) = (xs[0].mul_add(ss[0], q1), xs[0].mul_add(ss[0], -q1));
            let (a2, a3) = (xs[2].mul_add(ss[2], q3), xs[2].mul_add(ss[2], -q3));
            *vl = [a0 + a2, a1 + a3, a0 - a2, a1 - a3];
        }
        for st in 2..7 {
            let mask = 1usize << (st - 2);
            let prev = v;
            for lane in 0..32 {
                let hi = lane & mask != 0;
                for r in 0..4 {
                    let t = prev[lane ^ mask][r];
                    v[lane][r] = t + if hi { -prev[lane][r] } else { prev[lane][r] };
                }
            }
        }
        let mut o = [0u16; 128];
        for lane in 0..32 { for r in 0..4 { o[lane * 4 + r] = f16::from_f32(v[lane][r] * HNORM).to_bits(); } }
        o
    }

    fn xorshift(st: &mut u64) -> u64 { *st ^= *st << 13; *st ^= *st >> 7; *st ^= *st << 17; *st }

    /// Rows of x (row-major, width k) against one suh row: every 128-block, all 81 contractions.
    /// Returns (blocks, f16-inf outputs).
    fn check_rows(x: &[u16], s: &[u16], k: usize) -> (usize, usize) {
        let (mut blocks, mut inf) = (0, 0);
        for row in 0..x.len() / k {
            for b in 0..k / 128 {
                let (xb, sb) = (&x[row * k + b * 128..][..128], &s[b * 128..][..128]);
                let p = pinned_block(xb, sb);
                inf += p.iter().filter(|v| **v & 0x7FFF == 0x7C00).count();
                for ci in 0..81 {
                    assert_eq!(src_block(xb, sb, conv_of(ci)), p, "row {row} block {b} contraction {ci}");
                }
                blocks += 1;
            }
        }
        (blocks, inf)
    }

    #[test]
    fn wp20_f16_products_are_exact_in_f32() {
        let mut st = 0x1234_5678_9ABC_DEF1u64;
        let fin = |b: u16| b & 0x7C00 != 0x7C00;
        let mut n = 0;
        for _ in 0..2_000_000 {
            let w = xorshift(&mut st);
            let (a, b) = (w as u16, (w >> 16) as u16);
            if !fin(a) || !fin(b) { continue; }
            assert_eq!((f(a) * f(b)) as f64, f(a) as f64 * f(b) as f64, "{a:04x} * {b:04x}");
            n += 1;
        }
        for a in 0..=0xFFFFu16 {
            if !fin(a) { continue; }
            for b in [0x0001u16, 0x8001, 0x03FF, 0x0400, 0x7BFF, 0xFBFF, 0x3C01, 0x3555] {
                assert_eq!((f(a) * f(b)) as f64, f(a) as f64 * f(b) as f64, "{a:04x} * {b:04x}");
            }
        }
        assert!(n > 1_500_000);
    }

    #[test]
    fn wp20_every_contraction_equals_the_pinned_sequence() {
        let k = 640;
        // v1's calibration rows, then the v1c verification's edge expert
        check_rows(&wp20_synth(16 * k, 0x20C0, 1.0), &wp20_synth(k, 0x5CA1, 0.5), k);
        let (blocks, inf) = check_rows(&wp20_synth_edge(16 * k, 0x20C1_5EED_0001), &wp20_synth_edge(k, 0x5CA1_5EED_0002), k);
        eprintln!("edge expert: {inf} f16-inf outputs of {}", blocks * 128);
        assert!(inf > 0 && inf < blocks * 128 / 2, "edge expert coverage: {inf} inf of {}", blocks * 128);
        // random full-domain rows: x regimes 0..3 against suh regimes 1,2,3,0 (fresh per row)
        let mut st = 0xC0FF_EE00_D15E_A5E5u64;
        let (mut tb, mut ti) = (0, 0);
        for row in 0..500u64 {
            let x = wp20_synth_edge(512, xorshift(&mut st) ^ row);
            let s = wp20_synth_edge(640, xorshift(&mut st) ^ (row << 20));
            let (b, i) = check_rows(&x, &s[128..], 512);
            tb += b; ti += i;
        }
        eprintln!("full-domain rows: {ti} f16-inf outputs of {}", tb * 128);
        assert_eq!(WP20_SUH_SEQ, conv_of(1 + 3 + 9 + 27), "the pinned id is 'first fused' on all four ops");
    }

    #[test]
    fn wp20_v1_calibration_could_never_find_a_unique_match() {
        let (r, k) = (16usize, 640usize);
        let (x, s) = (wp20_synth(r * k, 0x20C0, 1.0), wp20_synth(k, 0x5CA1, 0.5));
        let run = |conv: u32| -> Vec<u16> {
            let mut out = Vec::with_capacity(r * k);
            for row in 0..r {
                for b in 0..k / 128 {
                    out.extend(src_block(&x[row * k + b * 128..][..128], &s[b * 128..][..128], conv));
                }
            }
            out
        };
        let served = run(WP20_SUH_SEQ);
        let hits = (0..81u32).filter(|&ci| run(conv_of(ci)) == served).count();
        assert_eq!(hits, 81, "all 81 conventions are bitwise identical on f16 inputs");
    }
}

// ---------------------------------------------------------------------------
// W3/LMH bit-identity proof, part 1 (host, no GPU): the persistent kernel's quad extraction
// (two ring words per 4-slot quad, per-slot shift + earlier-word select) yields the SAME 16-bit
// trellis index as exl3_dq_from's funnelshift(ring[i1m], ring[i0m], sf) for every lane, slot and
// bits in {3,4,5}. Both sides are pure bit SELECTIONS of the ring, so agreement on the zero ring
// and on every one-hot ring (each of the 8*bits*32 bit positions) proves equality for ALL ring
// contents; random rings are an extra belt. Also pins exl3_gemm_body's tmap-derived slot t to
// 8*lane + 4*h + s (the kernel's decode order) and to crate::exl3::tmap.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod lmh_tests {
    fn funnel_r(lo: u32, hi: u32, s: u32) -> u32 {
        ((((hi as u64) << 32) | lo as u64) >> (s & 31)) as u32
    }
    /// exl3_gemm_body / exl3_dq_from, transcribed.
    fn old_idx(ring: &[u32], bits: usize, lane: usize, h: usize, s: usize) -> (usize, u32) {
        let r = 2 * (lane & 3) + (s & 1) + 8 * (s >> 1);
        let c = 8 * h + (lane >> 2);
        let l = 8 * ((c >> 1) & 3) + ((r & 7) >> 1);
        let t = 8 * l + (r & 1) + 2 * ((r >> 3) & 1) + 4 * (c >> 3) + 32 * (c & 1);
        assert_eq!(t, crate::exl3::tmap(r, c), "kernel tmap transcription vs crate::exl3::tmap");
        let w = 8 * bits;
        let b1 = (t + 257) * bits;
        let i0 = (b1 - 16) >> 5;
        let i1 = (b1 - 1) >> 5;
        let sf = (((i1 + 1) << 5) - b1) as u32;
        (t, funnel_r(ring[i1 % w], ring[i0 % w], sf) & 0xFFFF)
    }
    /// exl3_lmh_body, transcribed.
    fn new_idx(ring: &[u32], bits: usize, lane: usize, q: usize, s: usize) -> u32 {
        let w = 8 * bits;
        let t0 = 8 * lane + 4 * q;
        let b0 = (t0 + 257) * bits - 16;
        let b2 = (t0 + 3 + 257) * bits;
        let i0 = b0 >> 5;
        let i2 = (b2 - 1) >> 5;
        let s2 = ((i2 + 1) << 5) - b2;
        let (wa, wb) = (ring[i0 % w], ring[i2 % w]);
        let sh = s2 + (3 - s) * bits;
        assert!(sh < 48, "shift {sh} out of the single-word-select envelope");
        let d = ((sh & 31) as u32) | if sh >= 32 { 32 } else { 0 };
        let lo = if d & 32 != 0 { wa } else { wb };
        funnel_r(lo, wa, d & 31) & 0xFFFF
    }
    fn check_ring(ring: &[u32], bits: usize) {
        for lane in 0..32 {
            for h in 0..2 {
                for s in 0..4 {
                    let (t, a) = old_idx(ring, bits, lane, h, s);
                    assert_eq!(t, 8 * lane + 4 * h + s, "slot order lane {lane} h {h} s {s}");
                    let b = new_idx(ring, bits, lane, h, s);
                    assert_eq!(a, b, "bits {bits} lane {lane} h {h} s {s}: old {a:#06x} new {b:#06x}");
                }
            }
        }
    }
    #[test]
    fn lmh_quad_extract_equals_dq_from() {
        for bits in 3..=5usize {
            let w = 8 * bits;
            check_ring(&vec![0u32; w], bits);
            for p in 0..w * 32 {
                let mut ring = vec![0u32; w];
                ring[p / 32] = 1u32 << (p % 32);
                check_ring(&ring, bits);
            }
            let mut st = 0x9E37_79B9u32 ^ bits as u32;
            for _ in 0..256 {
                let ring: Vec<u32> = (0..w).map(|_| { st ^= st << 13; st ^= st >> 17; st ^= st << 5; st }).collect();
                check_ring(&ring, bits);
            }
        }
    }
}

// W4/MOE host proofs (no GPU): the persistent expert kernels' schedule arithmetic and ring
// addressing, transcribed from kernels/exl3_bench.cu (mpk_items / mpk_gu_body / mpk_dn_body /
// mpk_issue_slot / mpk_consume / mpk_diet_consts). The GPU gate is --probe-exl3-binv EXL3-MOE-EPI.
#[cfg(test)]
mod mpk_tests {
    const KSTEP: usize = 4;
    const KROW: usize = 768;
    const STB: usize = KSTEP * KROW;
    const BB: usize = 96;

    /// mpk_items, transcribed.
    fn items(b: usize, g: usize, n: usize, il: bool) -> (usize, usize, usize, usize) {
        let (beg, step, end) = if il { (b, g, n) } else { (b * n / g, 1, (b + 1) * n / g) };
        let cnt = if beg < end { (end - beg + step - 1) / step } else { 0 };
        (beg, step, end, cnt)
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Grp { Slot { slot: usize, item: usize, ks: usize }, A { e: usize }, Empty }

    /// One thread's cp.async group queue (groups retire FIFO; a group's smem writes land when a
    /// wait retires it). The kernel has two thread roles: ring issuers (tid < MPK_NCH: slot groups,
    /// never an A group) and A issuers (tid >= MPK_NCH: EMPTY slot groups + the A groups).
    struct Q {
        groups: Vec<Grp>,
        lo: usize,
    }
    impl Q {
        /// cp.async.wait_group keep: every group but the newest `keep` retires; returns what landed.
        fn retire(&mut self, keep: usize) -> Vec<Grp> {
            let mut out = Vec::new();
            while self.lo + keep < self.groups.len() {
                out.push(self.groups[self.lo]);
                self.lo += 1;
            }
            out
        }
        fn pending(&self) -> &[Grp] { &self.groups[self.lo..] }
    }

    /// One CTA of the persistent loop (both roles). Asserts (a) every consumed ring slot holds
    /// exactly the (item, ks) the consumer expects, (b) no copy is pending on a slot being refilled
    /// or on A while the MMAs read it, (c) A holds the consumed item's expert, (d) the A wait never
    /// retires a ring group (the ring keeps its depth across expert changes), (e) nothing real is in
    /// flight at exit. Returns the consumed (item, ks) sequence.
    fn sim_cta(b: usize, g: usize, n: usize, il: bool, nst: usize, nks: usize, t: usize) -> Vec<(usize, usize)> {
        let (beg, step, end, cnt) = items(b, g, n, il);
        let total = cnt * nks;
        let mut out = Vec::new();
        if cnt == 0 { return out; }
        let mut rq = Q { groups: Vec::new(), lo: 0 };   // a ring-issuing thread
        let mut aq = Q { groups: Vec::new(), lo: 0 };   // an A-issuing thread
        let mut ring: Vec<Option<(usize, usize)>> = vec![None; nst];
        let mut a_state: Option<usize> = None;
        let land = |v: Vec<Grp>, ring: &mut Vec<Option<(usize, usize)>>, a_state: &mut Option<usize>| {
            for gr in v {
                match gr {
                    Grp::Slot { slot, item, ks } => ring[slot] = Some((item, ks)),
                    Grp::A { e } => *a_state = Some(e),
                    Grp::Empty => {}
                }
            }
        };
        let (mut p_item, mut p_ks, mut p_slot, mut p_i) = (beg, 0usize, 0usize, 0usize);
        macro_rules! issue {
            () => {
                if p_i < total {
                    assert!(!rq.pending().iter().any(|gr| matches!(gr, Grp::Slot { slot, .. } if *slot == p_slot)),
                            "b{b}: refill of slot {p_slot} while a copy into it is pending");
                    rq.groups.push(Grp::Slot { slot: p_slot, item: p_item, ks: p_ks });
                    p_i += 1;
                    p_ks += 1;
                    if p_ks == nks { p_ks = 0; p_item += step; }
                    p_slot += 1;
                    if p_slot == nst { p_slot = 0; }
                } else {
                    rq.groups.push(Grp::Empty);
                }
                aq.groups.push(Grp::Empty);              // the A threads commit the ring's group empty
            };
        }
        for _ in 0..nst - 1 { issue!(); }
        aq.groups.push(Grp::A { e: beg / t });            // mpk_issue_a: A threads only
        let mut a_pend = true;
        let (mut c_item, mut c_ks, mut c_slot) = (beg, 0usize, 0usize);
        let mut consumed: Vec<bool> = vec![false; nst];
        for it in 0..total {
            if a_pend {
                // mpk_wait_a: the A threads wait_all (their non-empty groups are A groups only);
                // ring threads do not wait; then the barrier.
                let landed = aq.retire(0);
                assert!(landed.iter().all(|gr| !matches!(gr, Grp::Slot { .. })), "b{b}: the A wait retired a ring group");
                land(landed, &mut ring, &mut a_state);
                a_pend = false;
            }
            let l1 = rq.retire(nst - 2);                   // cp.async.wait_group NST-2, both roles
            land(l1, &mut ring, &mut a_state);
            let l2 = aq.retire(nst - 2);
            land(l2, &mut ring, &mut a_state);
            // barrier: every warp is past slot it-1 — the slot refilled now was consumed there
            if p_i < total {
                assert_eq!(p_slot, (it + nst - 1) % nst);
                if it > 0 { assert!(consumed[p_slot], "b{b} it{it}: refilling an unconsumed slot"); }
            }
            issue!();
            assert_eq!(ring[c_slot], Some((c_item, c_ks)), "b{b} it{it}: slot {c_slot} content");
            assert!(!rq.pending().iter().any(|gr| matches!(gr, Grp::Slot { slot, .. } if *slot == c_slot)),
                    "b{b} it{it}: consuming a slot with a pending copy");
            assert!(!aq.pending().iter().any(|gr| matches!(gr, Grp::A { .. })), "b{b} it{it}: A being written while the MMAs read it");
            assert_eq!(a_state, Some(c_item / t), "b{b} it{it}: A holds the wrong expert");
            out.push((c_item, c_ks));
            consumed[c_slot] = true;
            c_slot += 1;
            if c_slot == nst { c_slot = 0; }
            c_ks += 1;
            if c_ks == nks {
                // epilogue: the stage-tile barrier (A dead), then the last-reader refill
                let e = c_item / t;
                let nx = c_item + step;
                if nx < end && nx / t != e {
                    aq.groups.push(Grp::A { e: nx / t });
                    a_pend = true;
                }
                c_ks = 0;
                c_item = nx;
            }
        }
        assert!(rq.pending().iter().chain(aq.pending().iter()).all(|gr| matches!(gr, Grp::Empty)), "b{b}: real copies pending at exit");
        out
    }

    #[test]
    fn mpk_schedule_covers_every_item_once_in_order() {
        let mut cases = 0usize;
        for &(t, nks) in &[(10usize, 40usize), (20, 10)] {
            for &live in &[1usize, 2, 7, 10, 26, 47, 90, 160, 256] {
                let n = live * t;
                for &g in &[1usize, 3, 7, 48, 96, 144, n, n + 5] {
                    for &il in &[false, true] {
                        for &nst in &[4usize, 8] {
                            let mut seen = vec![0usize; n];
                            for b in 0..g {
                                let seq = sim_cta(b, g, n, il, nst, nks, t);
                                for (k, &(item, ks)) in seq.iter().enumerate() {
                                    assert_eq!(ks, k % nks, "per-item k order");
                                    if ks == 0 { seen[item] += 1; }
                                }
                            }
                            assert!(seen.iter().all(|&c| c == 1), "t{t} live{live} g{g} il{il} nst{nst}: coverage");
                            cases += 1;
                        }
                    }
                }
            }
        }
        assert!(cases > 500);
    }

    /// The ring's byte mapping: producer chunk tid -> (row g, byte r*16) of the slot, from the
    /// expert's trellis at k16 row ks*KSTEP + g, tile t; consumer warp w, step g reads the 96-B
    /// block at slot + g*KROW + w*BB. Must equal the a1b3 body's block (kb, nb_col = 8t + w).
    #[test]
    fn mpk_ring_bytes_are_the_a1b3_blocks() {
        for &nb in &[80usize, 160] {                    // gate/up N = 1280, down N = 2560
            let kb_total = 16usize;
            let row_b = nb * BB;
            let mut st = 0x1234_5678u32 ^ nb as u32;
            let trellis: Vec<u8> = (0..row_b * kb_total).map(|_| { st ^= st << 13; st ^= st >> 17; st ^= st << 5; st as u8 }).collect();
            for t in 0..nb / 8 {
                for ks in 0..kb_total / KSTEP {
                    let mut slot = vec![0u8; STB];
                    for tid in 0..256usize {
                        if tid >= STB / 16 { continue; }                 // c_live
                        let cg = tid / (KROW / 16);
                        let cr = tid - cg * (KROW / 16);
                        let src = ks * KSTEP * row_b + t * KROW + cg * row_b + cr * 16;
                        let dst = cg * KROW + cr * 16;
                        slot[dst..dst + 16].copy_from_slice(&trellis[src..src + 16]);
                    }
                    for g in 0..KSTEP {
                        for w in 0..8 {
                            let kb = ks * KSTEP + g;
                            let nb_col = t * 8 + w;
                            let blk = &trellis[kb * row_b + nb_col * BB..kb * row_b + nb_col * BB + BB];
                            let got = &slot[g * KROW + w * BB..g * KROW + w * BB + BB];
                            assert_eq!(got, blk, "nb{nb} t{t} ks{ks} g{g} w{w}");
                        }
                    }
                }
            }
        }
    }

    /// mpk_diet_consts + mpk_consume's window cut == exl3_dq_from's funnel form for every lane
    /// and B-fragment slot at 3 bits (the WP20 rung-1 diet, transcribed independently).
    #[test]
    fn mpk_diet_windows_equal_dq_from() {
        let bits = 3usize;
        let w = 8 * bits;
        let funnel = |lo: u32, hi: u32, s: u32| ((((hi as u64) << 32) | lo as u64) >> (s & 31)) as u32;
        let mut st = 0xC0FF_EE11u32;
        for rep in 0..(1 + w * 32 + 512) {
            // rep 0: zeros; then one-hot over every ring bit; then random rings
            let ring: Vec<u32> = (0..w).map(|i| {
                if rep == 0 { 0 }
                else if rep <= w * 32 { if (rep - 1) / 32 == i { 1u32 << ((rep - 1) % 32) } else { 0 } }
                else { st ^= st << 13; st ^= st >> 17; st ^= st << 5; st }
            }).collect();
            for lane in 0..32usize {
                let r = 2 * (lane & 3);
                let c = lane >> 2;
                let l = 8 * ((c >> 1) & 3) + ((r & 7) >> 1);
                let t0 = 8 * l + (r & 1) + 2 * ((r >> 3) & 1) + 4 * (c >> 3) + 32 * (c & 1);
                let b10 = (t0 + 257) * bits;
                let a = (b10 - 16) >> 5;
                let (off_a, off_b) = (a % w, (a + 1) % w);
                let sh7 = 32 * (a + 2) - (b10 + 7 * bits);
                assert!(sh7 < 32, "sh7 {sh7}");
                let p = (((ring[off_a] as u64) << 32) | ring[off_b] as u64) >> sh7;
                for h in 0..2 {
                    for q in 0..4 {
                        let j = h * 4 + q;
                        let got = ((p >> (3 * (7 - j))) & 0xFFFF) as u32;
                        let rr = 2 * (lane & 3) + (q & 1) + 8 * (q >> 1);
                        let cc = 8 * h + (lane >> 2);
                        let ll = 8 * ((cc >> 1) & 3) + ((rr & 7) >> 1);
                        let tt = 8 * ll + (rr & 1) + 2 * ((rr >> 3) & 1) + 4 * (cc >> 3) + 32 * (cc & 1);
                        let b1 = (tt + 257) * bits;
                        let i0 = (b1 - 16) >> 5;
                        let i1 = (b1 - 1) >> 5;
                        let sf = (((i1 + 1) << 5) - b1) as u32;
                        let want = funnel(ring[i1 % w], ring[i0 % w], sf) & 0xFFFF;
                        assert_eq!(got, want, "rep {rep} lane {lane} h {h} q {q}");
                    }
                }
            }
        }
    }

    /// Host mirror of the smem layout (exl3_forward::mpk_smem_bytes) vs the kernel's pointer
    /// arithmetic: ring | A [M][K+8] | suh [K] (gate/up fold) | epi, every region 16-B aligned.
    #[test]
    fn mpk_smem_layout_matches_the_kernel() {
        for &nst in &[4usize, 8] {
            for m in 1..=16usize {
                for &(k, gu) in &[(2560usize, true), (640, false)] {
                    for &fold in &[false, true] {
                        let fold = fold && gu;
                        let sa = nst * STB;
                        let su = sa + m * (k + 8) * 2;
                        let epi = su + if fold { k * 2 } else { 0 };
                        let epi_b = if gu { (m * 256).max(4096) } else { m * 256 };
                        assert!(sa % 16 == 0 && su % 16 == 0 && epi % 16 == 0);
                        assert_eq!(epi + epi_b, crate::exl3_forward::mpk_smem_bytes(nst, m, k, gu, fold));
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// A5-P1 (PLAN/notes_2026-09-27/P1.md): the prefill MoE routed-expert chain.
// Reference = the served device-routed chain (xq_moe_pf_count at the PFX1 rule MT ->
// xq_moe_pf_place -> xq_had_suh_rows -> exl3_hmma_gemm_wide_tiles -> xq_had_svh_rows ->
// xq_moe_gate_mul -> xq_had_suh_rows -> exl3_hmma_gemm_wide_tiles -> xq_had_svh_rows) over 512
// synthetic experts (distinct trellis sets, 942 MB: DRAM-cold like serving) and skewed top-10
// routing (log-normal expert popularity: a hot tail with >= 128-row experts) at the served chunk
// classes c = 2048, 775 (a partial last chunk), 96 (MT 1 rule). Every candidate is compared
// BITWISE (outputs NaN-poisoned first) on every compact row:
//   * prefill.moe_kernel: xq_moe_pf_t{1,2,4,8} on both sides (y_raw) vs the reference y_raw;
//   * prefill.moe_mt_per_expert: xq_moe_pf_count4's lists (cnt / offs_row == xq_moe_pf_count's),
//     per-class launches with the served kernel AND the pipelined one;
//   * prefill.moe_glue_fold: xq_moe_pf_g{MT} (gate|up + svh + gate_mul + down suh) -> xhd, and
//     xq_moe_pf_s{MT} (down + svh) -> yd, uniform MT and per-class.
// Then a harness-only timing of the chains (ranks; the served number is the prefill A/B).
// ---------------------------------------------------------------------------
fn probe_moe_pf(dev: &std::sync::Arc<CudaDevice>) -> Result<()> {
    use crate::exl3_forward::{moe_pf_class_caps, moe_pf_fn, moe_pf_smem_bytes, pfx1_moe_mt, moe_pf2_fn, moe_pf2_smem_bytes, moe_pf2h_smem_bytes};
    let (h, mi, ne, topk) = (2560usize, 640usize, 512usize, 10usize);
    let f = |n: &str| dev.get_func(MODULE, n).with_context(|| n.to_string());
    // smem mirror
    {
        let mut d_q = dev.alloc_zeros::<i32>(8)?;
        unsafe { f("xq_moe_pf_smem_query")?.launch(LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 }, (&mut d_q,))? };
        let q = dev.dtoh_sync_copy(&d_q)?;
        for (i, (mt, nw)) in [(1, 8), (2, 8), (4, 8), (8, 8), (1, 16), (2, 16), (4, 16), (8, 16)].iter().enumerate() {
            if q[i] as u32 != moe_pf_smem_bytes(*mt, *nw) {
                bail!("EXL3-MOE-PF FAIL: smem mirror mt {mt} nw {nw}: kernel {} host {}", q[i], moe_pf_smem_bytes(*mt, *nw));
            }
            if q[i] > 48 * 1024 { bail!("EXL3-MOE-PF FAIL: mt {mt} nw {nw} needs {} B > 48 KB (no opt-in)", q[i]); }
        }
        // A5-L2: the pf2 instances, bits 3 and 4 (bits 3 = the served packs must fit 48 KB)
        let mut d_q2 = dev.alloc_zeros::<i32>(20)?;
        unsafe { f("xq_moe_pf2_smem_query")?.launch(LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 }, (&mut d_q2,))? };
        let q2 = dev.dtoh_sync_copy(&d_q2)?;
        for (i, (mt, nw)) in [(1, 8), (2, 8), (4, 8), (8, 8), (1, 16), (2, 16), (4, 16), (8, 16)].iter().enumerate() {
            for (b, bits) in [(0usize, 3usize), (1, 4)] {
                if q2[b * 8 + i] as u32 != moe_pf2_smem_bytes(*mt, *nw, bits) {
                    bail!("EXL3-MOE-PF FAIL: pf2 smem mirror mt {mt} nw {nw} bits {bits}: kernel {} host {}",
                          q2[b * 8 + i], moe_pf2_smem_bytes(*mt, *nw, bits));
                }
            }
            if q2[i] > 48 * 1024 { bail!("EXL3-MOE-PF FAIL: pf2 mt {mt} nw {nw} bits 3 needs {} B > 48 KB", q2[i]); }
        }
        for (i, (mt, bits)) in [(1usize, 3usize), (2, 3), (1, 4), (2, 4)].iter().enumerate() {
            if q2[16 + i] as u32 != moe_pf2h_smem_bytes(*mt, *bits) {
                bail!("EXL3-MOE-PF FAIL: pf2 paired smem mirror mt {mt} bits {bits}: kernel {} host {}", q2[16 + i], moe_pf2h_smem_bytes(*mt, *bits));
            }
        }
        if q2[16] > 48 * 1024 || q2[17] > 48 * 1024 { bail!("EXL3-MOE-PF FAIL: pf2 paired bits 3 over 48 KB"); }
        println!("  EXL3-MOE-PF pf2 smem (bits 3) t/s {:?} g {:?} h {:?} B", &q2[0..4], &q2[4..8], &q2[16..18]);
    }
    let gu_words = (h / 16) * (2 * mi / 16) * 48;
    let d_words = (mi / 16) * (h / 16) * 48;
    let d_gu = dev.htod_sync_copy(&wp20_synth(ne * gu_words, 0x7001, 1.0))?;
    let d_dn = dev.htod_sync_copy(&wp20_synth(ne * d_words, 0x7002, 1.0))?;
    let d_suh_gu = dev.htod_sync_copy(&wp20_synth(ne * h, 0x7003, 0.25))?;
    let d_svh_gu = dev.htod_sync_copy(&wp20_synth(ne * 2 * mi, 0x7004, 0.005))?;
    let d_suh_d = dev.htod_sync_copy(&wp20_synth(ne * mi, 0x7005, 0.02))?;
    let d_svh_d = dev.htod_sync_copy(&wp20_synth(ne * h, 0x7006, 0.01))?;
    let d_offs_gu = dev.htod_sync_copy(&(0..ne as u64).map(|e| e * gu_words as u64).collect::<Vec<u64>>())?;
    let d_offs_d = dev.htod_sync_copy(&(0..ne as u64).map(|e| e * d_words as u64).collect::<Vec<u64>>())?;
    let d_ident = dev.htod_sync_copy(&(0..ne as i32).collect::<Vec<i32>>())?;
    let (svh_gu, suh_d, svh_d) = (*d_svh_gu.device_ptr() as u64, *d_suh_d.device_ptr() as u64, *d_svh_d.device_ptr() as u64);
    // log-normal popularity (sigma 1.1), cumulative table
    let mut s = 0x1234_5678u32;
    let mut unif = || { s = s.wrapping_mul(1664525).wrapping_add(1013904223); ((s >> 8) as f64 + 0.5) / 16777216.0 };
    let mut cum = Vec::with_capacity(ne);
    let mut tot = 0.0f64;
    for _ in 0..ne {
        let z = (-2.0 * unif().ln()).sqrt() * (2.0 * std::f64::consts::PI * unif()).cos();
        tot += (1.1 * z).exp();
        cum.push(tot);
    }
    let lc = |gx: u32, gy: u32, b: u32, sm: u32| LaunchConfig { grid_dim: (gx, gy, 1), block_dim: (b, 1, 1), shared_mem_bytes: sm };
    let mut timing: Vec<String> = Vec::new();
    let mut nchk = 0usize;
    for &c in &[2048usize, 775, 96] {
        let r = c * topk;
        let max_tiles = r / 16 + ne;
        let caps = moe_pf_class_caps(r, ne);
        let mt_rule = pfx1_moe_mt(c);
        let mut ids: Vec<i32> = Vec::with_capacity(r);
        for _ in 0..c {
            let mut picked: Vec<i32> = Vec::new();
            while picked.len() < topk {
                let u = unif() * tot;
                let e = cum.partition_point(|&v| v < u).min(ne - 1) as i32;
                if !picked.contains(&e) { picked.push(e); }
            }
            ids.extend(picked);
        }
        let mut hist = vec![0usize; ne];
        for &e in &ids { hist[e as usize] += 1; }
        let cls_n: Vec<usize> = [(1usize, 16usize), (17, 32), (33, 127), (128, usize::MAX)].iter()
            .map(|&(a, b)| hist.iter().filter(|&&n| n >= a && n <= b).count()).collect();
        let d_ids = dev.htod_sync_copy(&ids)?;
        let d_x = dev.htod_sync_copy(&wp20_synth(c * h, 0x7010 + c as u32, 1.0))?;
        let d_dident = dev.htod_sync_copy(&(0..r as i32).collect::<Vec<i32>>())?;
        let d_moerows = dev.htod_sync_copy(&[r as i32])?;
        let mut d_cnt = dev.alloc_zeros::<i32>(ne)?;
        let mut d_offs_row = dev.alloc_zeros::<i32>(ne + 1)?;
        let mut d_tiles = dev.alloc_zeros::<i32>(2 * max_tiles)?;
        let mut d_nt = dev.alloc_zeros::<i32>(1)?;
        let mut d_tiles4 = dev.alloc_zeros::<i32>(8 * max_tiles)?;
        let mut d_nt4 = dev.alloc_zeros::<i32>(4)?;
        let mut d_row_tok = dev.alloc_zeros::<i32>(r)?;
        let mut d_row_eidx = dev.alloc_zeros::<i32>(r)?;
        let mut d_cand = dev.alloc_zeros::<i32>(r)?;
        let mut d_xh = dev.alloc_zeros::<u16>(r * h)?;
        let mut d_ygu_raw = dev.alloc_zeros::<u16>(r * 2 * mi)?;
        let mut d_ygu = dev.alloc_zeros::<u16>(r * 2 * mi)?;
        let mut d_din = dev.alloc_zeros::<u16>(r * mi)?;
        let mut d_xhd = dev.alloc_zeros::<u16>(r * mi)?;
        let mut d_xhd2 = dev.alloc_zeros::<u16>(r * mi)?;
        let mut d_ydr = dev.alloc_zeros::<u16>(r * h)?;
        let mut d_yd = dev.alloc_zeros::<u16>(r * h)?;
        let kn_gu = (h | ((2 * mi) << 16)) as i32;
        let kn_d = (mi | (h << 16)) as i32;
        let tl4 = *d_tiles4.device_ptr() as u64;
        let nt4 = *d_nt4.device_ptr() as u64;
        let tl1 = *d_tiles.device_ptr() as u64;
        let nt1 = *d_nt.device_ptr() as u64;
        let groups_pe: Vec<(usize, u64, u64, u32)> = (0..4).filter(|&k| caps[k] > 0)
            .map(|k| (1usize << k, tl4 + (k * max_tiles * 8) as u64, nt4 + (k * 4) as u64, caps[k] as u32)).collect();
        let groups_1 = |mt: usize| vec![(mt, tl1, nt1, max_tiles as u32)];
        // ---- launch helpers ----
        macro_rules! count1 { ($mt:expr) => { unsafe { f("xq_moe_pf_count")?.launch(lc(1, 1, 1024, 0),
            (&d_ids, r as i32, ne as i32, &mut d_cnt, &mut d_offs_row, &mut d_tiles, &mut d_nt, (16 * $mt) as i32))? } } }
        macro_rules! count4 { () => { unsafe { f("xq_moe_pf_count4")?.launch(lc(1, 1, 1024, 0),
            (&d_ids, r as i32, ne as i32, &mut d_cnt, &mut d_offs_row, &mut d_tiles4, &mut d_nt4, max_tiles as i32))? } } }
        // gate/up side: kind 'w' = served wide_tiles, 't' pipelined, 'g' fold (writes xhd2);
        // A5-L2: 'T' / 'G' = the pf2 twins of 't' / 'g', 'P' / 'Q' = the pf / pf2 fold `_prof` twins.
        let prof_fn = |v: usize, kind: char, mt: usize| -> String {
            format!("xq_moe_pf{}_prof_{kind}{mt}", if v == 0 { "" } else { "2" })
        };
        macro_rules! gu { ($kind:expr, $groups:expr) => { for &(gmt, tp, np, cap) in $groups.iter() {
            match $kind {
                'T' => unsafe { f(moe_pf2_fn('t', gmt))?.launch(lc((2 * mi / 128) as u32, cap, 256, moe_pf2_smem_bytes(gmt, 8, 3)),
                    (&d_gu, &d_offs_gu, &d_xh, &mut d_ygu_raw, tp, np, &d_offs_row, &d_cnt, kn_gu, 3i32, 0u64, 0u64))? },
                'G' | 'Q' => unsafe { f(&if $kind == 'G' { moe_pf2_fn('g', gmt).to_string() } else { prof_fn(1, 'g', gmt) })?
                    .launch(lc((2 * mi / 256) as u32, cap, 512, moe_pf2_smem_bytes(gmt, 16, 3)),
                    (&d_gu, &d_offs_gu, &d_xh, &mut d_xhd2, tp, np, &d_offs_row, &d_cnt, kn_gu, 3i32, svh_gu, suh_d))? },
                'P' => unsafe { f(&prof_fn(0, 'g', gmt))?.launch(lc((2 * mi / 256) as u32, cap, 512, moe_pf_smem_bytes(gmt, 16)),
                    (&d_gu, &d_offs_gu, &d_xh, &mut d_xhd2, tp, np, &d_offs_row, &d_cnt, kn_gu, 3i32, svh_gu, suh_d))? },
                'H' | 'R' if gmt <= 2 => unsafe { f(&format!("xq_moe_pf2_{}h{gmt}", if $kind == 'R' { "prof_" } else { "" }))?
                    .launch(lc((2 * mi / 256) as u32, cap, 256, moe_pf2h_smem_bytes(gmt, 3)),
                    (&d_gu, &d_offs_gu, &d_xh, &mut d_xhd2, tp, np, &d_offs_row, &d_cnt, kn_gu, 3i32, svh_gu, suh_d))? },
                'H' | 'R' => unsafe { f(&if $kind == 'H' { moe_pf2_fn('g', gmt).to_string() } else { prof_fn(1, 'g', gmt) })?
                    .launch(lc((2 * mi / 256) as u32, cap, 512, moe_pf2_smem_bytes(gmt, 16, 3)),
                    (&d_gu, &d_offs_gu, &d_xh, &mut d_xhd2, tp, np, &d_offs_row, &d_cnt, kn_gu, 3i32, svh_gu, suh_d))? },
                'w' => unsafe { f("exl3_hmma_gemm_wide_tiles")?.launch(lc((2 * mi / 128) as u32, cap, 256, 0),
                    (&d_gu, &d_offs_gu, &d_xh, &mut d_ygu_raw, tp, np, &d_offs_row, &d_cnt, h as i32, (2 * mi) as i32, 3i32, gmt as i32))? },
                't' => unsafe { f(moe_pf_fn('t', gmt))?.launch(lc((2 * mi / 128) as u32, cap, 256, moe_pf_smem_bytes(gmt, 8)),
                    (&d_gu, &d_offs_gu, &d_xh, &mut d_ygu_raw, tp, np, &d_offs_row, &d_cnt, kn_gu, 3i32, 0u64, 0u64))? },
                _ => unsafe { f(moe_pf_fn('g', gmt))?.launch(lc((2 * mi / 256) as u32, cap, 512, moe_pf_smem_bytes(gmt, 16)),
                    (&d_gu, &d_offs_gu, &d_xh, &mut d_xhd2, tp, np, &d_offs_row, &d_cnt, kn_gu, 3i32, svh_gu, suh_d))? },
            } } } }
        // down side from `$xin`: 'w' / 't' -> yd_raw, 's' -> yd
        macro_rules! dn { ($kind:expr, $groups:expr, $xin:expr) => { for &(gmt, tp, np, cap) in $groups.iter() {
            match $kind {
                'T' => unsafe { f(moe_pf2_fn('t', gmt))?.launch(lc((h / 128) as u32, cap, 256, moe_pf2_smem_bytes(gmt, 8, 3)),
                    (&d_dn, &d_offs_d, $xin, &mut d_ydr, tp, np, &d_offs_row, &d_cnt, kn_d, 3i32, 0u64, 0u64))? },
                'S' | 'Q' => unsafe { f(&if $kind == 'S' { moe_pf2_fn('s', gmt).to_string() } else { prof_fn(1, 's', gmt) })?
                    .launch(lc((h / 128) as u32, cap, 256, moe_pf2_smem_bytes(gmt, 8, 3)),
                    (&d_dn, &d_offs_d, $xin, &mut d_yd, tp, np, &d_offs_row, &d_cnt, kn_d, 3i32, svh_d, 0u64))? },
                'P' => unsafe { f(&prof_fn(0, 's', gmt))?.launch(lc((h / 128) as u32, cap, 256, moe_pf_smem_bytes(gmt, 8)),
                    (&d_dn, &d_offs_d, $xin, &mut d_yd, tp, np, &d_offs_row, &d_cnt, kn_d, 3i32, svh_d, 0u64))? },
                'w' => unsafe { f("exl3_hmma_gemm_wide_tiles")?.launch(lc((h / 128) as u32, cap, 256, 0),
                    (&d_dn, &d_offs_d, $xin, &mut d_ydr, tp, np, &d_offs_row, &d_cnt, mi as i32, h as i32, 3i32, gmt as i32))? },
                't' => unsafe { f(moe_pf_fn('t', gmt))?.launch(lc((h / 128) as u32, cap, 256, moe_pf_smem_bytes(gmt, 8)),
                    (&d_dn, &d_offs_d, $xin, &mut d_ydr, tp, np, &d_offs_row, &d_cnt, kn_d, 3i32, 0u64, 0u64))? },
                _ => unsafe { f(moe_pf_fn('s', gmt))?.launch(lc((h / 128) as u32, cap, 256, moe_pf_smem_bytes(gmt, 8)),
                    (&d_dn, &d_offs_d, $xin, &mut d_yd, tp, np, &d_offs_row, &d_cnt, kn_d, 3i32, svh_d, 0u64))? },
            } } } }
        macro_rules! place_suh { () => { unsafe {
            f("xq_moe_pf_place")?.launch(lc(ne as u32, 1, 256, 0),
                (&d_ids, r as i32, topk as i32, &d_offs_row, &mut d_row_tok, &mut d_row_eidx, &mut d_cand))?;
            f("xq_had_suh_rows")?.launch(lc((r * 20) as u32, 1, 32, 0),
                (&d_x, &d_suh_gu, &d_ident, &d_row_tok, &d_row_eidx, &mut d_xh, r as i32, h as i32))?;
        } } }
        macro_rules! glue_gu { () => { unsafe {
            f("xq_had_svh_rows")?.launch(lc((r * 10) as u32, 1, 32, 0),
                (&d_ygu_raw, &d_svh_gu, &d_ident, &d_row_eidx, &mut d_ygu, r as i32, (2 * mi) as i32))?;
            f("xq_moe_gate_mul")?.launch(lc(((r * mi) as u32 + 255) / 256, 1, 256, 0), (&mut d_din, &d_ygu, mi as i64, &d_moerows))?;
            f("xq_had_suh_rows")?.launch(lc((r * (mi / 128)) as u32, 1, 32, 0),
                (&d_din, &d_suh_d, &d_ident, &d_dident, &d_row_eidx, &mut d_xhd, r as i32, mi as i32))?;
        } } }
        macro_rules! glue_d { () => { unsafe {
            f("xq_had_svh_rows")?.launch(lc((r * (h / 128)) as u32, 1, 32, 0),
                (&d_ydr, &d_svh_d, &d_ident, &d_row_eidx, &mut d_yd, r as i32, h as i32))?;
        } } }
        let poison = |d: &mut CudaSlice<u16>| -> Result<()> { let n = d.len(); dev.htod_sync_copy_into(&vec![0xFFFFu16; n], d)?; Ok(()) };
        let cmp = |what: &str, a: &[u16], b: &[u16]| -> Result<()> {
            let bad = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
            if bad > 0 || a.len() != b.len() {
                let first = a.iter().zip(b.iter()).position(|(x, y)| x != y);
                bail!("EXL3-MOE-PF FAIL c={c} {what}: {bad} of {} f16 differ (first {:?})", a.len(), first);
            }
            Ok(())
        };
        // ---- reference: the served chain ----
        count1!(mt_rule);
        place_suh!();
        let cnt_ref = dev.dtoh_sync_copy(&d_cnt)?;
        let offs_ref = dev.dtoh_sync_copy(&d_offs_row)?;
        poison(&mut d_ygu_raw)?; poison(&mut d_xhd)?; poison(&mut d_ydr)?; poison(&mut d_yd)?;
        gu!('w', groups_1(mt_rule));
        glue_gu!();
        dn!('w', groups_1(mt_rule), &d_xhd);
        glue_d!();
        dev.synchronize()?;
        let ygu_ref = dev.dtoh_sync_copy(&d_ygu_raw)?;
        let xhd_ref = dev.dtoh_sync_copy(&d_xhd)?;
        let ydr_ref = dev.dtoh_sync_copy(&d_ydr)?;
        let yd_ref = dev.dtoh_sync_copy(&d_yd)?;
        if ygu_ref.iter().any(|&v| v == 0xFFFF) || yd_ref.iter().any(|&v| v == 0xFFFF) || xhd_ref.iter().any(|&v| v == 0xFFFF) {
            bail!("EXL3-MOE-PF FAIL c={c}: the reference chain left poisoned rows");
        }
        // every candidate, uniform MT lists (the ref's routing tables are MT-independent)
        for mt in [1usize, 2, 4, 8] {
            count1!(mt);
            for kind in ['w', 't', 'T'] {
                poison(&mut d_ygu_raw)?; poison(&mut d_ydr)?;
                gu!(kind, groups_1(mt));
                dn!(kind, groups_1(mt), &d_xhd);   // the reference xhd as input
                dev.synchronize()?;
                cmp(&format!("{kind} mt {mt} ygu_raw"), &dev.dtoh_sync_copy(&d_ygu_raw)?, &ygu_ref)?;
                cmp(&format!("{kind} mt {mt} yd_raw"), &dev.dtoh_sync_copy(&d_ydr)?, &ydr_ref)?;
                nchk += 2;
            }
            for (gk, dk, nm) in [('g', 's', "fold"), ('G', 'S', "pf2 fold"), ('H', 'S', "pf2 paired fold"), ('P', 'P', "pf prof fold"),
                                 ('Q', 'Q', "pf2 prof fold"), ('R', 'Q', "pf2 paired prof fold")] {
                poison(&mut d_xhd2)?; poison(&mut d_yd)?;
                gu!(gk, groups_1(mt));
                dn!(dk, groups_1(mt), &d_xhd);
                dev.synchronize()?;
                cmp(&format!("{nm} mt {mt} xhd"), &dev.dtoh_sync_copy(&d_xhd2)?, &xhd_ref)?;
                cmp(&format!("{nm} mt {mt} yd"), &dev.dtoh_sync_copy(&d_yd)?, &yd_ref)?;
                nchk += 2;
            }
        }
        // WP19 per-expert lists
        count4!();
        dev.synchronize()?;
        cmp("count4 cnt", &dev.dtoh_sync_copy(&d_cnt)?.iter().map(|v| *v as u16).collect::<Vec<_>>(),
            &cnt_ref.iter().map(|v| *v as u16).collect::<Vec<_>>())?;
        if dev.dtoh_sync_copy(&d_offs_row)? != offs_ref { bail!("EXL3-MOE-PF FAIL c={c}: count4 offs_row != count's"); }
        let nt4h = dev.dtoh_sync_copy(&d_nt4)?;
        for k in 0..4 { if nt4h[k] as usize > caps[k] { bail!("EXL3-MOE-PF FAIL c={c}: class {k} has {} tiles > cap {}", nt4h[k], caps[k]); } }
        for kind in ['w', 't', 'T'] {
            poison(&mut d_ygu_raw)?; poison(&mut d_ydr)?;
            gu!(kind, groups_pe);
            dn!(kind, groups_pe, &d_xhd);
            dev.synchronize()?;
            cmp(&format!("{kind} per-expert ygu_raw"), &dev.dtoh_sync_copy(&d_ygu_raw)?, &ygu_ref)?;
            cmp(&format!("{kind} per-expert yd_raw"), &dev.dtoh_sync_copy(&d_ydr)?, &ydr_ref)?;
            nchk += 2;
        }
        for (gk, dk, nm) in [('g', 's', "fold"), ('G', 'S', "pf2 fold"), ('H', 'S', "pf2 paired fold"), ('P', 'P', "pf prof fold"),
                             ('Q', 'Q', "pf2 prof fold"), ('R', 'Q', "pf2 paired prof fold")] {
            poison(&mut d_xhd2)?; poison(&mut d_yd)?;
            gu!(gk, groups_pe);
            dn!(dk, groups_pe, &d_xhd2);      // chained on the FOLDED xhd: the served fold chain end to end
            dev.synchronize()?;
            cmp(&format!("{nm} per-expert xhd"), &dev.dtoh_sync_copy(&d_xhd2)?, &xhd_ref)?;
            cmp(&format!("{nm} per-expert yd (chained)"), &dev.dtoh_sync_copy(&d_yd)?, &yd_ref)?;
            nchk += 2;
        }
        // A5-L2: every pf2 mask (mixed pf / pf2 classes in one chain) == the reference
        for mask in [1i32, 2, 4, 8, 3, 7, 15] {
            poison(&mut d_xhd2)?; poison(&mut d_yd)?;
            for &g in groups_pe.iter() { let k = g.0.trailing_zeros() as i32; gu!(if mask & (1 << k) != 0 { 'H' } else { 'g' }, [g]); }
            for &g in groups_pe.iter() { let k = g.0.trailing_zeros() as i32; dn!(if mask & (1 << k) != 0 { 'S' } else { 's' }, [g], &d_xhd2); }
            dev.synchronize()?;
            cmp(&format!("pf2 mask {mask} xhd"), &dev.dtoh_sync_copy(&d_xhd2)?, &xhd_ref)?;
            cmp(&format!("pf2 mask {mask} yd (chained)"), &dev.dtoh_sync_copy(&d_yd)?, &yd_ref)?;
            nchk += 2;
        }
        println!("  EXL3-MOE-PF c={c}: rows {r}, experts by count [1-16 {} | 17-32 {} | 33-127 {} | >=128 {}], \
                  max {}, rule MT {mt_rule}, per-expert tiles {:?}: every candidate bitwise == the served chain",
                 cls_n[0], cls_n[1], cls_n[2], cls_n[3], hist.iter().max().unwrap(), nt4h);
        // ---- timing (harness-only; 512 distinct experts = DRAM-cold trellis) ----
        if c >= 512 || crate::opts::var(crate::opt!("p1-time-all")).is_ok() {
            let reps = 8usize;
            let mut t = |name: &str, run: &mut dyn FnMut() -> Result<()>| -> Result<f64> {
                run()?; dev.synchronize()?;
                let t0 = std::time::Instant::now();
                for _ in 0..reps { run()?; }
                dev.synchronize()?;
                let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
                timing.push(format!("c={c} {name}: {ms:.3} ms/layer"));
                Ok(ms)
            };
            let base = t("OFF   served (wide_tiles rule MT + glue)", &mut || { count1!(mt_rule); gu!('w', groups_1(mt_rule)); glue_gu!(); dn!('w', groups_1(mt_rule), &d_xhd); glue_d!(); Ok(()) })?;
            let g0 = t("  gemms only: wide_tiles rule MT", &mut || { gu!('w', groups_1(mt_rule)); dn!('w', groups_1(mt_rule), &d_xhd); Ok(()) })?;
            let g1 = t("  gemms only: pf_t rule MT", &mut || { gu!('t', groups_1(mt_rule)); dn!('t', groups_1(mt_rule), &d_xhd); Ok(()) })?;
            for mt in [2usize, 4, 8] {
                count1!(mt);
                t(&format!("  gemms only: pf_t MT {mt}"), &mut || { gu!('t', groups_1(mt)); dn!('t', groups_1(mt), &d_xhd); Ok(()) })?;
            }
            count4!();
            let g2 = t("  gemms only: wide_tiles per-expert MT", &mut || { gu!('w', groups_pe); dn!('w', groups_pe, &d_xhd); Ok(()) })?;
            let g3 = t("  gemms only: pf_t per-expert MT", &mut || { gu!('t', groups_pe); dn!('t', groups_pe, &d_xhd); Ok(()) })?;
            let l1 = t("L1    pf_t rule MT + glue", &mut || { count1!(mt_rule); gu!('t', groups_1(mt_rule)); glue_gu!(); dn!('t', groups_1(mt_rule), &d_xhd); glue_d!(); Ok(()) })?;
            let l12 = t("L1+2  pf_t per-expert + glue", &mut || { count4!(); gu!('t', groups_pe); glue_gu!(); dn!('t', groups_pe, &d_xhd); glue_d!(); Ok(()) })?;
            let l2 = t("L2    wide_tiles per-expert + glue", &mut || { count4!(); gu!('w', groups_pe); glue_gu!(); dn!('w', groups_pe, &d_xhd); glue_d!(); Ok(()) })?;
            let all = t("ALL   fold per-expert", &mut || { count4!(); gu!('g', groups_pe); dn!('s', groups_pe, &d_xhd2); Ok(()) })?;
            let l13 = t("L1+3  fold rule MT", &mut || { count1!(mt_rule); gu!('g', groups_1(mt_rule)); dn!('s', groups_1(mt_rule), &d_xhd2); Ok(()) })?;
            let mut l2x: Vec<String> = Vec::new();
            // A5-L2: the served ALL chain with pf2 per MT-class mask, then each class alone (pf vs pf2)
            let mut mrow = Vec::new();
            for mask in [0i32, 1, 2, 4, 8, 3, 7, 15] {
                let ms = t(&format!("L2    ALL + pf2 mask {mask}"), &mut || {
                    count4!();
                    for &g in groups_pe.iter() { let k = g.0.trailing_zeros() as i32; gu!(if mask & (1 << k) != 0 { 'H' } else { 'g' }, [g]); }
                    for &g in groups_pe.iter() { let k = g.0.trailing_zeros() as i32; dn!(if mask & (1 << k) != 0 { 'S' } else { 's' }, [g], &d_xhd2); }
                    Ok(()) })?;
                mrow.push(format!("{mask}:{:.1}", ms * 48.0));
            }
            l2x.push(format!("c={c} L2 SUMMARY x48 (ms/chunk) ALL by pf2 mask {}", mrow.join(" | ")));
            let (gu_b, d_b) = ((gu_words * 2) as f64, (d_words * 2) as f64);
            for &g in groups_pe.iter() {
                let k = g.0.trailing_zeros() as usize;
                let ntl = nt4h[k] as f64;
                if ntl == 0.0 { continue; }
                let a = t(&format!("  class MT {} g  pf ", g.0), &mut || { gu!('g', [g]); Ok(()) })?;
                let b = t(&format!("  class MT {} g  pf2", g.0), &mut || { gu!('G', [g]); Ok(()) })?;
                let bh = if g.0 <= 2 { t(&format!("  class MT {} g  pf2 paired", g.0), &mut || { gu!('H', [g]); Ok(()) })? } else { b };
                let cc = t(&format!("  class MT {} s  pf ", g.0), &mut || { dn!('s', [g], &d_xhd2); Ok(()) })?;
                let d = t(&format!("  class MT {} s  pf2", g.0), &mut || { dn!('S', [g], &d_xhd2); Ok(()) })?;
                l2x.push(format!("c={c} L2 CLASS MT {} ({} tiles): g pf {:.3} ms ({:.0} GB/s) -> pf2 {:.3} ms ({:.0} GB/s), paired {:.3} ms ({:.0} GB/s) | s pf {:.3} ms ({:.0} GB/s) -> pf2 {:.3} ms ({:.0} GB/s)  [trellis GB/s = tiles x expert bytes / time]",
                    g.0, nt4h[k], a, ntl * gu_b / a / 1e6, b, ntl * gu_b / b / 1e6, bh, ntl * gu_b / bh / 1e6, cc, ntl * d_b / cc / 1e6, d, ntl * d_b / d / 1e6));
            }
            // `_prof` anatomy (thread 0 of each live CTA: stage-top wait / decode+mma / epilogue / span)
            {
                let mut d_pr = dev.alloc_zeros::<u64>(120)?;
                unsafe { f("xq_moe_pf_prof_take")?.launch(lc(1, 1, 128, 0), (&mut d_pr,))? };
                gu!('P', groups_pe); dn!('P', groups_pe, &d_xhd2);
                gu!('Q', groups_pe); dn!('Q', groups_pe, &d_xhd2);
                gu!('R', groups_pe);
                unsafe { f("xq_moe_pf_prof_take")?.launch(lc(1, 1, 128, 0), (&mut d_pr,))? };
                dev.synchronize()?;
                let pr = dev.dtoh_sync_copy(&d_pr)?;
                let mut slots: Vec<(usize, &str, char, usize)> = Vec::new();
                for v in 0..2usize {
                    for (ki, kind) in ['s', 'g'].iter().enumerate() {
                        for (mi2, mt) in [1usize, 2, 4, 8].iter().enumerate() {
                            slots.push((v * 8 + ki * 4 + mi2, if v == 0 { "pf " } else { "pf2" }, *kind, *mt));
                        }
                    }
                }
                slots.push((16, "pf2", 'h', 1)); slots.push((17, "pf2", 'h', 2));
                {
                    {
                        for &(sl, vn, kind, mt) in slots.iter() {
                            let x = &pr[sl * 5..sl * 5 + 5];
                            if x[4] == 0 { continue; }
                            let n = x[4] as f64;
                            let span = x[3] as f64 / n;
                            l2x.push(format!("c={c} L2 PROF {} {kind}{mt}: {} CTAs, span {:.0} cyc/CTA = wait {:.1}% + decode/mma {:.1}% + epilogue {:.1}% (rest {:.1}%)",
                                vn, x[4], span, 100.0 * x[0] as f64 / n / span,
                                100.0 * x[1] as f64 / n / span, 100.0 * x[2] as f64 / n / span,
                                100.0 * (1.0 - (x[0] + x[1] + x[2]) as f64 / x[3] as f64)));
                        }
                    }
                }
            }
            timing.push(format!("c={c} SUMMARY x48 layers (ms/chunk): OFF {:.1} | L1 {:.1} | L2 {:.1} | L1+2 {:.1} | L1+3 {:.1} | ALL {:.1} \
                                 | gemms wide {:.1} -> pf {:.1} (wide pe {:.1}, pf pe {:.1})",
                                base * 48.0, l1 * 48.0, l2 * 48.0, l12 * 48.0, l13 * 48.0, all * 48.0, g0 * 48.0, g1 * 48.0, g2 * 48.0, g3 * 48.0));
            timing.extend(l2x);
        }
    }
    for l in &timing { println!("    EXL3-MOE-PF timing {l} (harness-only, serial launches, DRAM-cold trellis; ranks the paths)"); }
    println!("EXL3-MOE-PF: PASS (A5-P1: xq_moe_pf_t{{1,2,4,8}} (prefill.moe_kernel), per-expert MT lists (prefill.moe_mt_per_expert, \
              both kernels) and the folded xq_moe_pf_g/s chain (prefill.moe_glue_fold); A5-L2: the xq_moe_pf2 smem-ring twins \
              (prefill.moe_pf2: t/g/s every MT, per-expert, every class mask) and both `_prof` twins; all bitwise == the served wide_tiles + glue chain \
              on every compact row; {nchk} comparisons over c in {{2048, 775, 96}}, 512 experts, skewed top-10 routing)");
    Ok(())
}
