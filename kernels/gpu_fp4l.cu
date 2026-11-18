// gpu_fp4l.cu — FP4L weight kernels: nvidia/Qwen3.8-Flash-Next-FP4-Local served on the EXL3 trunk.
//
// Module `gpu_fp4l` (sm_121a: the E2M1x2 -> F16x2 cvt is an SM12x family-specific form; every
// other instruction here is baseline sm_121). Loaded only when an FP4-Local pack is opened
// (exl3_forward/fp4l_fwd.rs). Activations are the EXL3 trunk's f16; accumulation is fp32.
// Design + contracts: PLAN/FP4L_DESIGN.md §4. Every entry opens with XF_PDL_ENTRY (the EXL3
// module's PDL contract) so a PDL-launched neighbour may overlap it.
//
// Weight formats (all row-major [N output rows][K]; K contiguous — the checkpoint layout, no repack):
//   MXFP8  : E4M3 codes + UE8M0 scales per 32, w = e4m3(c) * 2^(s-127), stored on the device in the FRAGMENT-NATIVE
//            layout "F64" (a pure byte permutation of the checkpoint's [N][K] / [N][K/32] arrays made at load,
//            fp4l_fwd::mx8_frag; N padded to a multiple of 16 with zero rows): codes W'[t][b][h][lane][16]
//            (t = N/16 tiles, b = K/64 blocks, h = 0/1 the two 8-row halves, lane = gid*4 + tig) hold row
//            16t+8h+gid, k = 64b + 32gs + 8tig + j at byte gs*8 + j; scales S'[t][b][tig][8] hold, at byte
//            h*4 + r*2 + gs, the scale of row 16t+8h+2tig+r for group 2b+gs. A warp's 16-B load is then 512
//            contiguous bytes that ARE its MMA fragments (the row-major layout's 8-B loads touched 8 rows per
//            instruction and streamed at ~130 GB/s; F64 streams at the roofline, microbench 2026-10-29).
//   NVFP4  : E2M1 nibbles [N][K/2] (low nibble = even k) + E4M3 scales [N][K/16], one fp32 global
//            multiplier per (expert, projection): w = e2m1(q) * e4m3(s) * gs
//   F16    : [N][K] f16 = f16(w * 2^k), one power-of-two shift per tensor (--fp4l-lm-head f16 / --fp4l-plain f16)
//   BF16   : [N][K] the checkpoint's bf16 bits AS STORED (--fp4l-lm-head bf16 / --fp4l-plain exact). The f16
//            activation x is split into two bf16 halves x = hi + lo (hi = bf16 RN of x, lo = x - hi: both exact in
//            bf16, since x has 11 significant bits), so W*x = W*hi + W*lo with every product exact in fp32
//            (8 x 8 significant bits): two bf16 MMAs per f16 one, the same weight bytes.
//
// Exactness. e4m3 codes and e2m1*e4m3 products are exact in f16 (|v| <= 2688, >= 2^-10). The
// MXFP8 power-of-two group scale is NOT folded into f16 (56,179 values of this pack would round):
// each 32-k group is reduced by its own pair of MMAs into a zeroed fp32 tile which is then
// multiplied by 2^(s-127) (exact) and added to the accumulator. The NVFP4 global multiplier is
// applied once to the fp32 accumulator in the epilogue.
//
// The k-permutation. Inside one m16n8k16 MMA each lane owns the k slots {2t, 2t+1, 2t+8, 2t+9}
// (t = lane & 3) in BOTH the A and the B fragment. A dot product does not care which actual k sits
// in a slot as long as A and B agree, so lane t is given CONTIGUOUS actual k (8 per 32-k group for
// MXFP8 / f16; 16 = one NVFP4 block per 64-k step for NVFP4): one wide load per row instead of four
// scattered ones, and for NVFP4 the lane's 16 values share ONE block scale. The mapping is fixed,
// so every output's reduction order is the same at every M (batch invariance, AGENTS §2.4).
//
// Batch invariance. M (tokens) is the MMA M dimension; each output row depends only on its own A
// row and the fixed k order; rows >= M read a clamped valid row and are never stored. The split-K
// factor ks is a pure function of (K, N) (host: fp4l_fwd::mx8_ks); slabs are summed in ascending
// order (the WP21 fixup contract: the last-arriving CTA of an N-tile sums slabs 0..ks-1 and re-arms
// the counter). No atomics touch data.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda/atomic>
#include <stdint.h>

#ifdef KERNEL_BUILD_ID
extern "C" __global__ void kernel_build_id(unsigned long long* out) { *out = KERNEL_BUILD_ID; }
#else
extern "C" __global__ void kernel_build_id(unsigned long long* out) { *out = 0; }
#endif

#define XF_PDL_ENTRY() do { asm volatile("griddepcontrol.wait;" ::: "memory"); \
                            asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); } while (0)

#include "gpu_fp4l/01_mx8_nv4_gemms_dequant.cuh"
#include "gpu_fp4l/02_precision_levers_router_hc.cuh"
#include "gpu_fp4l/03_embed_gate_ple_heads.cuh"
