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
//!
//! Module map (this file keeps the shared helpers: device/module loading, host references, the `Chain` launcher):
//!   gemm.rs        kernel-decode gate, multi-width GEMM probe, fixed-cost ladder / GEMM / LMH benches
//!   wide.rs        wide-GEMM probe + bench, cross-chain probe
//!   hc.rs          GDN tensor-core, prefill HC and W4 HC probes, `probe_hcw4`
//!   moe_epi.rs     WP20 synth/calibration helpers, MoE epilogue probes (all bits widths, bwd)
//!   moe_epi_k4.rs  the K = 4 twin of the MoE epilogue probe
//!   moe_pf.rs      prefill MoE routed-expert chain probe
//!   dense.rs       grouped A1, LM-head, split-K, WP21 and dense-fusion probes
//!   router.rs      f16-rows GEMM, router fold, R16, argmax-rows probes
//!   attn.rs        dense and sparse attention probes
//!   select.rs      QSA select and WP18 select probes
//!   bitwidths.rs   the all-bitwidths gate
//!   tp4.rs         TP=4 W=4 probes (tail GEMM, router GEMM, MoE glue)
//!   sampler.rs     sampler, flash-prefill and penalty probes
//!   <test files>   the former inline #[cfg(test)] modules, one file each

use crate::exl3::{decode_trellis_host, tmap, Exl3Pack, ModuleMeta, TensorClass};
use crate::gpu::fork_blocking_stream;
use anyhow::{bail, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use half::f16;

mod attn;
mod bitwidths;
mod dense;
mod gemm;
mod hc;
mod moe_epi;
mod moe_epi_k4;
mod moe_pf;
mod router;
mod sampler;
mod select;
mod tp4;
mod wide;

pub use self::bitwidths::probe_bitwidths;
pub use self::gemm::{bench_gemm, bench_ladder, probe_gemm, probe_kernel_decode};
pub use self::hc::probe_hcw4;
pub(crate) use self::moe_epi::wp20_calibrate;
#[cfg(test)]
use self::moe_epi::WP20_SUH_SEQ;
pub use self::sampler::{probe_flash_prefill, probe_penalties, probe_sampler};
pub use self::tp4::{probe_moeglue, probe_routergemm, probe_tailgemm};
pub use self::wide::{bench_wide, probe_wide, probe_xchain};

use self::attn::{probe_attn, probe_attn_sparse};
use self::dense::{probe_dense, probe_grouped_a1, probe_ks_p, probe_lmh, probe_wp21};
use self::hc::{probe_gdn_tc2, probe_hc_w4, probe_pf_hc};
use self::router::{probe_gemm_f16_rows};
use self::moe_epi::{probe_moe_epi, probe_moe_epi_bwd, wp20_synth};
#[cfg(test)]
use self::moe_epi::wp20_synth_edge;
use self::moe_epi_k4::{probe_moe_epi_k4};
use self::moe_pf::{probe_moe_pf};
use self::select::{probe_qsa_select, probe_wp18_select};

const MODULE: &str = "exl3_bench";
pub(crate) const MODULE_FNS: &[&str] = &[
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
    // TP-4T1: bit-exact small-M twin for the prefill tail chunks (hc down/up, a/b, PLE)
    "xq_gemm_f16_tail",
    "xq_gemm_f16_f32",
    "xq_hc_norm",
    "xq_silu_div",
    "xq_hc_mix",
    "xq_hc_inject",
    "xq_conv1d",
    "xq_gdn_step",
    "xq_gdn_gate",
    "xq_gdn_gate_silu", // EXL27B: the Qwen3.5-class gated norm (silu), a copy of xq_gdn_gate
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
    "xq_cos_gather_v",
    "xq_splice_rows",
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
    // TP-4F2: single-stage 4-way decode reduce K2 (`--tp-reduce single`; sums the three peers + own, one f16 rounding)
    "xq_tp_wait_add_dec_single",
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
    // CF-P1e: gate/up epilogue with K-chunked A staging (9..16 rows)
    "xq_moe_gu_epi_c2",
    "xq_moe_gu_epi_c4",
    // CF-P1g step 2: K = 4 expert twins
    "xq_moe_gu_epi_b4",
    "xq_moe_gu_epi_b4_c2",
    "xq_moe_gu_epi_b4_c4",
    "xq_moe_dn_epi_b4_k10",
    // P7 B3 (K = 5 experts): the WP20 b5 SHFL epilogues + the K-chunked b5 gate/up + a1b5.
    "xq_moe_gu_epi_b5",
    "xq_moe_gu_epi_b5_c2",
    "xq_moe_gu_epi_b5_c4",
    "xq_moe_dn_epi_b5_k10",
    "xq_gemm_grouped_a1b5",
    // P7 B3 leg 2 (K = 2/6/7 experts): the WP20 SHFL twins + a1 entries.
    "xq_moe_gu_epi_b2",
    "xq_moe_gu_epi_b2_c2",
    "xq_moe_gu_epi_b2_c4",
    "xq_moe_gu_epi_b6",
    "xq_moe_gu_epi_b6_c2",
    "xq_moe_gu_epi_b6_c4",
    "xq_moe_gu_epi_b7",
    "xq_moe_gu_epi_b7_c2",
    "xq_moe_gu_epi_b7_c4",
    "xq_moe_dn_epi_b2_k10",
    "xq_moe_dn_epi_b6_k10",
    "xq_moe_dn_epi_b7_k10",
    "xq_gemm_grouped_a1b2",
    "xq_gemm_grouped_a1b6",
    "xq_gemm_grouped_a1b7",
    "xq_gemm_grouped_b8", // K8EXP: K = 8 grouped expert entry (a1 / barrier)
    "xq_attn_dense_splitk4_vr", // DSPLITK round 2: shared-read verify
    "xq_attn_dense_splitk4_vr_w4", // DSPLITK round 2b: the ring-staged shared-read verify
    "xq_attn_dense_splitk4_t4", // DSPLITK round 3: the trunk split-K with q8/fp8 rows decoded once per block
    "xq_attn_pf_qstage", // PFTC: tensor-core dense prefill (q staging, attention, rotate-back + gate)
    "xq_attn_pf_tc256",
    "xq_attn_pf_tc256p", // PFTC round 3: the cp.async-pipelined twin
    "xq_attn_pf_tc_epi",
    "xq_gemm_grouped_a1b4",
    "xq_gemm_grouped_a1b4_wd",
    // CF-P1e step 0: per-call expert ids ring (--exl3-route-log)
    "xq_route_log",
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
    "xq_attn_kv_prefill_pairs",
    "xq_gather_tap_rows",
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
    // TP-4M1: live-row vectorised twin (EP prefill up-input rotation) + the vectorised EP combine
    "xq_had_suh_rows_live_u1",
    "xq_had_suh_rows_live_u2",
    "xq_had_suh_rows_live_u4",
    "xq_moe_combine_rows_ep2_k10",
    "xq_moe_combine_rows_ep2_k16",
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
    // TP-4R1: register-tiled order-preserving prefill router (128x64 / 64x64 CTA tiles)
    "xq_gemm_f16_f32_tile128", "xq_gemm_f16_f32_tile64",
    // S-A3-p: int8-weight hc mixer twin (--hc-int8, opt-in)
    "xq_hc_fuse_i8",
    "xq_gemm_f16_v",
    "xq_gemm_f16_f32_v",
    "exl3_hmma_gemm_ks",
    // LANEWIDE-MID: the split-K chain body for 17..128 rows (--exl3-mid-gemm)
    "exl3_hmma_gemm_ks_mid",
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
    "exl3_hmma_gemm_ks_fx_probe",
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
    // TP-4G: G=6 (W=4 / TP=4) instantiations
    "xq_attn_dense_dots_g6",
    "xq_attn_dense_acc4_g6",
    "xq_attn_dense_acc1_g6",
    "xq_dv3_xcheck",
    "xq_gdn_commit",
    // S-A3-o G1: rank-1 ring commit
    "xq_gdn_commit_ring",
    // WP12: register-resident GDN verify chunk + decode step (one device body), nostore
    // verify conv on the live ring, one-launch a|b GEMV (all bitwise; --wp12-off)
    "xq_gdn_step_chunk_r",
    "xq_gdn_step_r",
    "xq_conv1d_chunk_ns",
    // PACK1 LEG 1: the segmented packed-verify entries (probe-only; see exl3_forward probe_pack2)
    "xq_gdn_step_chunk_seg",
    "xq_conv1d_chunk_seg_ns",
    "xq_ple_conv_chunk_seg",
    "xq_accept_seg",
    "xq_pad_route",
    // PACK2 LEG 2: per-segment device-keyed commit entries (probe-only; see exl3_forward probe_packstate)
    "xq_conv_commit_seg",
    "xq_gdn_commit_ring_seg",
    "xq_ple_ring_commit_seg",
    "xq_gemm_f16_v_ab",
    "xq_conv_commit",
    "xq_ple_ring_commit",
    "xq_accept",
    // API-parity G2: device sampler (sampled lanes: plain steps + MTP verify rows)
    "xq_sample_rows",
    // WP15: device penalties (decode/verify rows + draft mirror)
    "xq_pen_rows",
    "xq_pen_draft",
    // W2-EXL3: JSON-schema token mask (decode / seam rows)
    "xq_vmask_rows",
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
    // port of the reference implementation's exl3_moe_coop_a/b; math identical to
    // xq_gemm_grouped_xh — see kernels/exl3_bench.cu). A = gate/up, B = down;
    // launched from fn moe() behind --moe-coop.
    "xq_moe_coop_a",
    "xq_moe_coop_b",
    // S-A3-i: QSA selection + sparse attention chain (kernels/exl3_bench.cu).
    "xq_qsa_h2b",
    // EXL27C-DF2 (fwd): the DFlash2 EXL3 tap capture + the boot-time head transpose.
    "xq_df2_tap_cvt",
    "xq_df2_transpose_f16",
    "xq_qsa_qnorm_rope",
    "xq_qsa_topk_asc",
    "xq_attn_prep",
    "xq_attn_sel_splitk",
    "xq_attn_sel_combine",
    // S-A3-v: float4 prefill twin of xq_attn_sel_splitk (bit-identical).
    "xq_attn_sel_splitk4",
    // PQ8 item 3: 8-bit-cache prefill gather (bit-identical partials to splitk4)
    "xq_attn_sel_pq8",
    // POSTURE1 --exl3-sel-tc: the prefill sparse gather on tensor cores (opt-in, not bitwise pq8)
    "xq_attn_sel_tc",
    "xq_attn_sel_tcu",
    "xq_attn_sel_tcu8",
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
    "xq_attn_dense_acc4_w4_g6",
    "xq_attn_dense_acc1_w4_g6",
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

/** The EXL3 kernel module this process loads, from the bytes embedded at compile time.
 * P7 B7: ONE module for every width — the K=6 arms are always compiled in; the
 * `exl3_bench_k6.ptx` per-process swap (4.05-bpw packs) is gone. */
pub(crate) fn bench_ptx_src() -> &'static str { crate::ptx_embed::EXL3_BENCH }

fn load_module(dev: &std::sync::Arc<CudaDevice>) -> Result<()> {
    let ptx = Ptx::from_src(bench_ptx_src());
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

/// The bench/gate matrix set: one module of every stored (class, width, shape) the pack carries. PR #27 (lesserevil):
/// a hand-written list named one historical pack's widths and shapes, so the probes of any other pack ran on a subset
/// of what it serves (the 3.05 pack: 8 of its 24 geometries; vision, most MTP shapes and the K=6 dense never ran).
/// Repeated experts / layers add no new kernel geometry. The n-gram table is not a GEMM operand (its own probes).
fn pick_modules(pack: &Exl3Pack) -> Vec<ModuleMeta> {
    let mut seen = std::collections::BTreeSet::new();
    pack.modules.iter()
        .filter(|m| m.class != TensorClass::Ngram && seen.insert((m.class, m.k2, m.k, m.n)))
        .cloned().collect()
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
fn slab_trellis(tr: &[u8], kb: usize, nb: usize, k2: usize, cols: Option<usize>) -> (Vec<u8>, usize) {
    match cols {
        None => (tr.to_vec(), nb * 16),
        Some(c) => {
            let nc = (c / 16).min(nb);
            let blw = 16 * k2; // block bytes (16x16 tile, k2 half-bits /8; == 16*bits*2 only for integer widths — P7 B8)
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
    k2: usize, // k2-keyed switch index (2*K + half) — P7 B8
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
    k2: usize,
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
        k2,
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
                    c.k2 as i32,
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

/// S-A3-d: module loader for the offline forward probe (same .ptx + registry).
pub(crate) fn load_module_pub(dev: &std::sync::Arc<CudaDevice>) -> Result<()> {
    load_module(dev)
}

// helper the test above needs (a1_smem is private in exl3_forward; recompute locally)
fn a1_smem_pub_for_test(m: usize, k: usize) -> u32 {
    let bytes = m * (k + 8) * 2;
    if bytes <= 48 * 1024 { bytes as u32 } else { 0 }
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
mod wp20_suh_tests;

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
mod lmh_tests;

// W4/MOE host proofs (no GPU): the persistent expert kernels' schedule arithmetic and ring
// addressing, transcribed from kernels/exl3_bench.cu (mpk_items / mpk_gu_body / mpk_dn_body /
// mpk_issue_slot / mpk_consume / mpk_diet_consts). The GPU gate is --probe-exl3-binv EXL3-MOE-EPI.
#[cfg(test)]
mod mpk_tests;

// S-B9-REL-FIXB K2: pure-Rust model of the accept rule (xq_accept / xq_accept_seg).
// The kernels are the ground truth on device; this mirrors their loops exactly so the
// bound (drafts, not rows) is pinned by a host test — including the zero-tail case that
// read the next segment before the fix.
#[cfg(test)]
mod rel_fixb_accept_rule;

// -------------------------------------------------------------------------------------------
// LEG-B 166 (2026-10-30): GPU-kernel-vs-WHEEL golden for the k2=3 expert decode.
// probe_kernel_decode's pick_modules only selects k2/2==3 experts (k2 6/7) — the k2=3
// fractional expert path was never kernel-vs-oracle checked, and the host oracle is now
// wheel-exact (src/exl3.rs k23_wheel_golden). This test launches the REAL decode kernel
// (exl3_decode_dump, same function the MoE path feeds) on the wheel-built L4 golden quad
// and requires bit-identity with LinearEXL3::get_inner_weight_tensor.
// -------------------------------------------------------------------------------------------
#[cfg(test)]
mod k23_kernel_wheel_golden;

// -------------------------------------------------------------------------------------------
// LEG-B 166 (2026-10-30): SERVING-GEMM-vs-WHEEL golden for k2=3 experts.
// The decode kernel is wheel-exact (k23_kernel_wheel_golden) yet the 1.66 pack's hiddens
// diverge from the wheel from the FIRST k2=3 expert layer (L4) on. This test drives the
// SERVED body (xq_gemm_grouped_a1b3_fh -> exl3_gemm_body_a1_fh<3>, the fused
// suh+GEMM+svh entry the moe() path launches for bits=3) standalone on the wheel-built
// L4 golden quad and compares against the wheel's full-weight matmul. If this passes,
// the bug is NOT the kernel body but the serving-side binding (loader offsets / concat /
// idxmap) — those feed this kernel with base+offs pointers the dump golden never used.
// -------------------------------------------------------------------------------------------
#[cfg(test)]
mod k23_serving_gemm_wheel_golden;

// -------------------------------------------------------------------------------------------
// LEG-B 166 (2026-10-30), test 2: LAYOUT-FAITHFUL serving golden. The TP1 loader feeds
// gu_tr INTERLEAVED (per k16 row: gate strip then up strip; ring window = 8*BITS u32),
// while the first test (k23_a1b3_fh_gemm_vs_wheel) fed the RAW up quad — its 2.78 may be
// a layout artifact. Here the buffer is built EXACTLY as exl3_forward.rs builds gu_tr_h
// for one expert (up quad bytes in BOTH strips), N = 2*mi; the UP half of the output must
// match the wheel's full-weight matmul. Pass => body+layout sound, bug elsewhere
// (down/raw path or bridge); fail => the body's fractional window math is the bug.
// -------------------------------------------------------------------------------------------
#[cfg(test)]
mod k23_serving_gemm_layout_golden;

// -------------------------------------------------------------------------------------------
// LEG-B 166 (2026-10-30), test 3: the REAL serving call. moe_inner launches grouped_fn(m,h,
// moe.bits) with bits = k2/2 = 1 on the 1.66 pack -> xq_gemm_grouped_xh, whose switch has
// cases 2..7 ONLY -> default: break = silent no-op (the comment there says this exact class
// bit them for bits 2/7 before). This test launches the REAL kernel with the REAL args
// (raw up quad, N = mi, bits = the fractional width code EXL3_FRAC_BITS(1) — K35; LEG-B passed bits = 1) and checks whether y is written at all.
// EXPECT (pre-fix): y unchanged (no-op proven, root cause confirmed).
// EXPECT (post-fix, case 1 -> fractional body): matches the wheel full-weight matmul.
// -------------------------------------------------------------------------------------------
#[cfg(test)]
mod k23_serving_real_call;

// -------------------------------------------------------------------------------------------
// K35 (2026-10-31): KERNEL-vs-WHEEL goldens for the fractional expert widths (K = 3.5 first; K = 2.5
// when its fixture lands) on REAL quads of turboderp's self-calibrated packs (the fixtures of
// src/exl3.rs k35_wheel_golden). Three receipts per quad, every one against the wheel's own
// reconstruct (inner_f16_bits) rather than our host oracle:
//   1. exl3_decode_dump (k2-keyed; the reconstruct arm's decode) == inner, bit for bit;
//   2. the SERVED grouped kernel xq_gemm_grouped_xh at the EXL3_FRAC_BITS code — A-once body where
//      M*(K+8)*2 <= 48 KiB, the barrier body above — == the k2-keyed m16 reference bit for bit at every
//      width 1..=16, fully overwrites a poisoned output, and lands within fp16 rounding of a f64 GEMM
//      on the wheel's weights (a wrong window formula measures O(1));
//   3. the device-routed PREFILL body exl3_hmma_gemm_wide_tiles at the same code, every MT class
//      (1/2/4/8) on a 20-row expert block, == the m16 reference bit for bit.
// -------------------------------------------------------------------------------------------
#[cfg(test)]
mod k35_kernel_wheel_golden;

/// PR #32 (andersc), generalized: `xq_pf_sel` / `xq_pf_fetch` / `xq_pf_pick` in kernels/exl3_bench.cu fetch each
/// lane's N distinct source ring words once (N = 2 for BITS <= 4, 3 for BITS 5..7) instead of one shuffle per
/// (window, half). Host mirror of the table arithmetic (`xq_pf_tables`): for every lane and every served width
/// the 16 (lo, hi) words are N consecutive ring words from window 0's hi word, and the selection reproduces the
/// original per-window word, so the kernels decode the same bits (bitwise).
#[cfg(test)]
mod pf_ring_source_tests;

