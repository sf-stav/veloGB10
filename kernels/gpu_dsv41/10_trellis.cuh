// =============================================================================
// kernels/gpu_dsv41/10_trellis.cuh — band 10-19 head (leg KE): EXL3 trellis kernels for DSV4.1.
// Copied and re-parameterised from kernels/exl3_bench (never edited there). Files:
//   11_exl3_decode.cuh  decode helpers, Hadamard-128, slot constants, the tile loop
//   12_exl3_dense.cuh   prep / GEMM (GEMV + coop + wide) / post, with (kb, nb) slices on either side
//   13_exl3_moe.cuh     routing compaction + grouped decode MoE over a per-expert inner-width slice table
//   14_exl3_plain.cuh   plain BF16 / F16 GEMV and the vocab-parallel embedding gather
//   15_exl3_moe_dec.cuh KE2: decode MoE (<= 32 rows) in 3 launches, routed + shared fused, cp.async ring, PDL-aware
//   16_exl3_moe_wide.cuh KE2: prefill MoE (> 32 rows): plan, weight-stationary tensor-core gate/up + down, combine
//   17_exl3_dense_wide.cuh KE2: dense wide (> 32 rows): tensor-core tile GEMM, trellis or reconstructed-f16 B source
//   18_exl3_dense_dec.cuh PK4: dense decode / verify (<= 32 rows, integer K): stream-K persistent GEMM, fused output transform
//   19_exl3_moe_cp.cuh  PM1: decode MoE (<= 32 rows) in the external reference's short-CTA structure (per-warp rings, ready flags)
// =============================================================================
#pragma once
#include "11_exl3_decode.cuh"
#include "12_exl3_dense.cuh"
#include "13_exl3_moe.cuh"
#include "14_exl3_plain.cuh"
#include "15_exl3_moe_dec.cuh"
#include "16_exl3_moe_wide.cuh"
#include "17_exl3_dense_wide.cuh"
#include "18_exl3_dense_dec.cuh"
#include "19_exl3_moe_cp.cuh"
