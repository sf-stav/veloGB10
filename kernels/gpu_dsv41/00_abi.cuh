// =============================================================================
// kernels/gpu_dsv41/00_abi.cuh — the DSV4.1 kernel ABI (leg I0). Included FIRST by
// kernels/gpu_dsv41.cu. Every dsv41 kernel file follows these conventions; the Rust mirror of every
// constant and struct size is src/dsv41/gpu/abi.rs, and `--probe-dsv41-abi` checks both agree on
// the GPU (sizes, tags, and the decode of every E4M3 / E8M0 / E2M1 code).
//
// 1. MODULE. One PTX module, `gpu_dsv41` (sm_121a: the FP4 / FP8 block-scaled MMA and the e2m1
//    conversions are sm_121a family features). Section files live in kernels/gpu_dsv41/ and are
//    pulled in by gpu_dsv41.cu through the reserved band head files (`NN_<name>.cuh`, see there):
//    a leg adds kernels by creating / editing ITS band's files only. build.rs hashes every
//    kernels/gpu_dsv41/*.cuh into KERNEL_BUILD_ID (a stale PTX cannot boot).
// 2. NAMES. Every extern "C" entry is `d41_<area>_<what>`; device helpers are `d41_*` and static
//    inline. Each leg lists its entries in its Rust host file's `FNS` (src/dsv41/gpu/*.rs).
// 3. STREAMS. Every launch goes on the engine's BLOCKING compute stream (AGENTS §2.1). Never the
//    legacy default stream, never a NonBlocking fork; D2D copies on the compute stream (§2.3).
// 4. PDL. Every entry kernel's first statement is D41_PDL_ENTRY() (griddepcontrol.wait, then
//    launch_dependents). Without the launch attribute both are no-ops; with it the prologue
//    overlaps the predecessor's tail. A kernel that must not let dependents start early calls
//    D41_PDL_WAIT_ONLY() instead and signals at its end.
// 5. DETERMINISM / BATCH INVARIANCE (AGENTS §2.4). Fixed reduction orders; no float atomics on any
//    value that feeds an output; split-K combines in a fixed order. Row i of an N-row call equals
//    a 1-row call within a row class. Row classes (Rust `RowClass`): GEMV <= D41_GEMV_MAX_ROWS,
//    cooperative <= D41_COOP_MAX_ROWS, wide above.
// 6. LOCAL MEMORY (AGENTS §4). No runtime-indexed per-thread arrays: template on sizes. Check
//    `nvcc -Xptxas -v` for stack frames in hot loops. Dynamic smem above 48 KB needs the opt-in on
//    the handle that launches (src/dsv41/gpu/mod.rs `raw_fn`); the GB10 cap is 101376 B per block.
// 7. ERRORS. No printf in served kernels. A kernel that detects a contract violation writes a code
//    into a caller-provided `int* err` (0 = fine) instead of trapping.
// 8. FORMATS. The value codes and the cache layouts below are THE storage formats of dsv41 state.
// =============================================================================
#pragma once

#define D41_ABI_VERSION 1u

#define D41_SWA_WINDOW 128
#define D41_BACKTRACK 6
#define D41_SWA_SLOTS (D41_SWA_WINDOW + D41_BACKTRACK)

#define D41_GEMV_MAX_ROWS 2
#define D41_COOP_MAX_ROWS 16

#define D41_PDL_ENTRY() do { asm volatile("griddepcontrol.wait;" ::: "memory"); \
                             asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); } while (0)
#define D41_PDL_WAIT_ONLY() do { asm volatile("griddepcontrol.wait;" ::: "memory"); } while (0)
#define D41_PDL_RELEASE() do { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); } while (0)

// ---- weight format tags (Rust: abi::FmtTag) --------------------------------------------------
#define D41_FMT_BF16 1u
#define D41_FMT_F16 2u
#define D41_FMT_F32 3u
#define D41_FMT_FP8_B32X32_UE8M0 10u   // F8_E4M3 values, one ue8m0 scale per 32 x 32 block (DeepSeek dense)
#define D41_FMT_FP8_B1X32_UE8M0 11u    // F8_E4M3 values, one ue8m0 scale per 32 inputs of a row (Engram tables)
#define D41_FMT_MXFP4 20u              // E2M1 x2 packed, one ue8m0 scale per 32 inputs (DeepSeek experts)
#define D41_FMT_NVFP4 21u              // E2M1 x2 packed, one E4M3 scale per 16 + an f32 tensor scale
#define D41_FMT_EXL3_BASE 32u          // EXL3 trellis with K bits per weight: tag = 32 + K (K = 1..8)
#define D41_FMT_EXL3(k) (D41_FMT_EXL3_BASE + (unsigned)(k))

// ---- value codes -----------------------------------------------------------------------------
// E2M1 (FP4): sign | 2-bit exponent (bias 1) | 1-bit mantissa. Packed two per byte, element 2i in
// the LOW nibble, 2i+1 in the high nibble (the reference's convert.py FP4_TABLE order).
// Decoded arithmetically (a runtime-indexed table would land in local memory, rule 6):
// exponent 0 -> m * 0.5; else (1 + m/2) * 2^(e-1). Magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6.
__device__ __forceinline__ float d41_e2m1(unsigned code) {
    unsigned e = (code >> 1) & 3u, m = code & 1u;
    float v = e ? __int_as_float((int)(((e + 126u) << 23) | (m << 22))) : 0.5f * (float)m;
    return (code & 8u) ? -v : v;
}
// E8M0: 2^(e - 127); e = 255 is NaN; e = 0 is 2^-127 (a float denormal).
__device__ __forceinline__ float d41_e8m0(unsigned e) {
    e &= 0xFFu;
    if (e == 0xFFu) return __int_as_float(0x7FC00000);
    if (e == 0u) return __int_as_float(0x00400000);
    return __int_as_float((int)(e << 23));
}
// E4M3 (fn, no infinities): sign | 4-bit exponent (bias 7) | 3-bit mantissa; 0x7F / 0xFF are NaN;
// exponent 0 is subnormal (m / 8 * 2^-6). Max finite 448.
__device__ __forceinline__ float d41_e4m3(unsigned c) {
    c &= 0xFFu;
    unsigned s = c >> 7, e = (c >> 3) & 15u, m = c & 7u;
    float v;
    if (e == 15u && m == 7u) v = __int_as_float(0x7FC00000);
    else if (e == 0u) v = (float)m * 0.001953125f;            // m / 8 * 2^-6 = m * 2^-9
    else v = __int_as_float((int)(((e + 120u) << 23) | (m << 20)));  // 2^(e-7) * (1 + m/8)
    return s ? -v : v;
}

// ---- state cache layouts (Rust: src/dsv41/state.rs, abi.rs) ----------------------------------
// Main compressed KV: one RECORD per compressed position (sparse gathers read whole rows):
//   512 values as E2M1 (256 B, element 2i low nibble) then 32 E4M3 scales, one per 16 values.
struct d41_kv_main_row { unsigned char q[256]; unsigned char s[32]; };      // 288 B
// Index-K: two PLANES per cache (the indexer scans every compressed position densely):
//   plane q [rows][64 B] = 128 values E2M1; plane s [rows][4 B] = one E8M0 per 32 values.
struct d41_index_k_q { unsigned char q[64]; };                               // 64 B
struct d41_index_k_s { unsigned char s[4]; };                                 // 4 B
// SWA ring (per layer slot, trunk and DSpark): D41_SWA_SLOTS = window 128 + speculative backtracking 6
// = 134 slots, slot = position % 134 (DeepSeek's engine sizes it so: a rejected draft rolls back at
// most block + 1 = 6 rows). Two PLANES: plane q [134][512 B] E4M3; plane s [134][16 B] E8M0 per 32 values.
// The compressor frontier ring of a ratio-r KV source keeps r + D41_BACKTRACK - 1 f32 rows (7 at r = 2).
struct d41_swa_q { unsigned char q[512]; };                                   // 512 B
struct d41_swa_s { unsigned char s[16]; };                                    // 16 B
// Engram row record (ENGb packs these per owned head): 256 E4M3 values then 8 E8M0 scales.
struct d41_engram_row { unsigned char q[256]; unsigned char s[8]; };         // 264 B

// ---- launch geometry hints --------------------------------------------------------------------
#define D41_WARP 32
#define D41_SMEM_OPTIN_MAX 101376

// cp.async group primitives shared by every band (were defined twice, in band 20 / KN1 and band 50 / KI;
// moved here at the 2026-11-18 integration merge so each kernel file can use them).
__device__ __forceinline__ void d41_cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void d41_cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

// ---- RoPE: ONE arithmetic for every band (leg SL; src/dsv41/rope.rs has the table builder) ----------------
// REFb (and DeepSeek's engine) rotate with torch on CUDA. Measured on GB10 against torch (1M samples per RoPE
// class, positions 0..1M; PLAN/DSV41_SL_REPORT.md): sin / cos = sincosf(fp32(pos) * freq) == torch.polar on all
// of them, and so is d41_sincos below; the rotation must be the complex multiply as torch's CUDA build
// contracts it (two FMAs): separately rounded products differ from torch on ~3e-5 of the bf16 results.
// d41_sincos: sin / cos of an fp32 angle without local memory (was KA's d41a_sincos). |x| < 105615: the
// libdevice sincosf fast path operation for operation (3-term Cody-Waite reduction with FMA, its two minimax
// polynomials); larger |x|: the quadrant and remainder from a double-precision reduction, then the same
// polynomials (libdevice's Payne-Hanek path keeps a runtime-indexed array in local memory).
__device__ __forceinline__ void d41_sincos(float x, float& s, float& c) {
    float r;
    int k;
    if (fabsf(x) < 105615.0f) {
        const float kf = rintf(__fmul_rn(x, __uint_as_float(0x3F22F983u)));
        k = (int)kf;
        r = __fmaf_rn(kf, __uint_as_float(0xBFC90FDAu), x);
        r = __fmaf_rn(kf, __uint_as_float(0xB3A22168u), r);
        r = __fmaf_rn(kf, __uint_as_float(0xA7C234C5u), r);
    } else {
        const double xd = (double)x;
        const double kd = rint(xd * 0.63661977236758138);
        k = (int)(long long)kd;
        double rd = fma(-kd, 1.5707963267948966, xd);
        rd = fma(-kd, 6.123233995736766e-17, rd);
        r = (float)rd;
    }
    const float r2 = __fmul_rn(r, r);
    float pc = __fmaf_rn(r2, __uint_as_float(0x37CBAC00u), __uint_as_float(0xBAB607EDu));
    pc = __fmaf_rn(pc, r2, __uint_as_float(0x3D2AAABBu));
    pc = __fmaf_rn(pc, r2, __uint_as_float(0xBEFFFFFFu));
    pc = __fmaf_rn(pc, r2, 1.0f);
    const float r3 = __fmaf_rn(r2, r, 0.0f);
    float ps = __fmaf_rn(r2, __uint_as_float(0xB94D4153u), __uint_as_float(0x3C0885E4u));
    ps = __fmaf_rn(ps, r2, __uint_as_float(0xBE2AAAA8u));
    ps = __fmaf_rn(ps, r3, r);
    const float sv = (k & 1) ? pc : ps;
    const float cv = (k & 1) ? ps : pc;
    s = (k & 2) ? -sv : sv;
    c = ((k + 1) & 2) ? -cv : cv;
}
// cos / sin of the rotation of position `pos` by inverse frequency `freq` (torch.outer then torch.polar).
__device__ __forceinline__ void d41_rope_cs(long long pos, float freq, float& c, float& s) { d41_sincos(__fmul_rn((float)pos, freq), s, c); }
// (a + ib)(c + is): re = fma(a, c, -(b s)), im = fma(b, c, a s)
__device__ __forceinline__ void d41_rope_fwd(float& a, float& b, float c, float s) {
    const float re = __fmaf_rn(a, c, -__fmul_rn(b, s)), im = __fmaf_rn(b, c, __fmul_rn(a, s));
    a = re; b = im;
}
// (a + ib)(c - is) (freqs_cis.conj()): re = fma(a, c, b s), im = fma(b, c, -(a s))
__device__ __forceinline__ void d41_rope_inv(float& a, float& b, float c, float s) {
    const float re = __fmaf_rn(a, c, __fmul_rn(b, s)), im = __fmaf_rn(b, c, -__fmul_rn(a, s));
    a = re; b = im;
}
