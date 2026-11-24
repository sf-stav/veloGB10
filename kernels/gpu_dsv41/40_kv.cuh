// =============================================================================
// kernels/gpu_dsv41/40_kv.cuh — band 40 head file (leg KC): the compressor and the cache writes.
//   41_kv_quant.cuh     value helpers: RNE e2m1 / e4m3 casts, the two FP4 scale rules, the FP8 rule,
//                       RMSNorm of one row, interleaved RoPE of the tail (bit-twins of model.py / kernel.py)
//   42_kv_compress.cuh  d41_kc_compress (pool + norm + main-KV record), d41_kc_stash (frontier ring)
//   43_kv_index_swa.cuh d41_kc_index_k (norm + RoPE + FP4/E8M0 plane write), d41_kc_swa_write
//                       (norm + RoPE + FP8/E8M0 ring + chunk write)
// Inputs are the outputs of the projection GEMMs (wkv / wgate / wk, other legs); these kernels own
// everything after them. See src/dsv41/gpu/kv.rs for the launch contract and PLAN/DSV41_KC_REPORT.md.
// =============================================================================
#include "41_kv_quant.cuh"
#include "42_kv_compress.cuh"
#include "43_kv_index_swa.cuh"
// leg FW: graph-safe twins (device positions) of the kernels above for the TP decode / verify calls
#include "44_kv_graph.cuh"
