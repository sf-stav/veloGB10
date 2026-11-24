// =============================================================================
// kernels/gpu_dsv41/50_indexer.cuh — band 50 head (leg KI): the DSV4.1 indexer.
//   51_idx_common.cuh  constants, the argument block, MMA / FP4 / ordering helpers, sorts
//   52_idx_score.cuh   d41_idx_score_r{1,2,4,8}: fused RoPE + FP4 q, FP4 MMA scores, top-512 filter
//   53_idx_merge.cuh   the row merge (fused into the score kernel's last chunk CTA) + d41_idx_merge (TP)
//   54_idx_cand.cuh    d41_idx_cand / d41_idx_union: layer-20 candidate blocks and their union
// Host side: src/dsv41/gpu/indexer.rs (FNS, launches), idx_ref.rs (CPU reference), idx_test.rs.
// =============================================================================
#pragma once
#include "51_idx_common.cuh"
#include "53_idx_merge.cuh"
#include "52_idx_score.cuh"
#include "54_idx_cand.cuh"
