// gpu_w4a4.cu — NVFP4 W4A4 PREFILL GEMMs that read the engine's STANDARD MMA-tiled NVFP4 weights.
//
// The serving weights stay exactly as the loader repacks them (quant.rs `repack_nvfp4_mma`: 128-B
// 16x16 tiles + 16-B scale tiles + one f32 `gs` per 16-row tile) — no OMMA repack, no second copy,
// so the 97-GB expert set of qwen4_exp needs zero extra bytes and the decode/verify W4A16 chain is
// untouched. Activations are quantized per token and per 16-block along K to E2M1 codes + a UE4M3
// block scale (the mxf4nvf4 recipe of gpu_mxfp4.cu, statement-identical arithmetic), with a
// per-tensor INPUT GLOBAL SCALE `x_gs` (compressed-tensors `input_global_scale`: x_q = e2m1 * e4m3 /
// x_gs; x_gs = 1 recovers the plain amax/6 recipe). The block-scaled MMA
//   mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X .f32.e2m1.e2m1.f32.ue4m3
// accumulates in f32; the epilogue applies gs[tile] * out_scale (out_scale = 1 / x_gs).
//
// Fragment sourcing from the STANDARD layout (the whole point of this file): the OMMA A fragment
// register a_r of lane (g = lane>>2, t = lane&3) holds 8 nibbles of row g + 8*(r&1), k = 8t +
// 32*(r>>1) .. +7 of the 64-wide kstep. In the standard tiled layout (fp4_tile_slot) that is, in
// tile kb = 2*(r>>1) + (t>>1) of the kstep, byte jb = (r&1) | ((t&1)<<1) of the four u32 words
// g*4 .. g*4+3 — i.e. byte jb of a 16-B chunk. src/mxfp4.rs `repack_nvfp4_omma` is exactly this
// permutation done once on the host; here it is done per fragment from shared memory: the stage
// holds the 4 standard tiles of a kstep (512 B, the same bytes an OMMA image would be), the 16-B
// chunks are placed with an XOR swizzle so the two chunks a lane reads (kb = t>>1 and 2 + t>>1)
// sit in different bank groups, and 3 byte-permutes assemble each register. SFA: the OMMA scale
// word of lane (g, t<=1) is bytes v = 0..3 = the scale of row g + 8t in tile ks*4 + v — 4 byte
// reads from the 64-B scale slice.
//
// Batch invariance: N padded to 8 (packer zero-fills), fixed k-visit order, no atomics, so a
// token's output does not depend on which other tokens share its tile. The kernels are prefill-only
// (batch > MAX_VERIFY): the lossless MTP verify contract (decode ≡ verify) is untouched.
//
// COMPILE: sm_121a (the mma and cvt.e2m1x2 reject plain sm_121) — build.rs.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

#ifndef KERNEL_BUILD_ID
#define KERNEL_BUILD_ID 0ULL
#endif
extern "C" __global__ void kernel_build_id(unsigned long long* out) { *out = KERNEL_BUILD_ID; }

typedef __nv_bfloat16 bf16;
__device__ __forceinline__ float b2f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 f2b(float x) { return __float2bfloat16(x); }

// Two f32 -> one byte of two e2m1 nibbles (low nibble = src[0]), RNE + satfinite (hardware).
__device__ __forceinline__ unsigned char cvt_e2m1x2(float lo, float hi) {
    unsigned tmp;
    asm volatile(
        "{\n.reg .b8 byte0, byte1, byte2, byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n}"
        : "=r"(tmp) : "f"(lo), "f"(hi));
    return (unsigned char)(tmp & 0xff);
}

// ue4m3 encode of |x| rounded UP (sign bit 0). Statement-identical to gpu_mxfp4.cu's.
__device__ __forceinline__ unsigned char e4m3_ceil(float x) {
    // E4M3 has NO infinity and 0x7F is NaN: the largest finite code is 0x7E = 448. Saturate there
    // (the codes then clip through cvt.satfinite) — 0x7F would poison the whole MMA output.
    if (!(x > 0.f)) return 0x00;
    if (x >= 448.0f) return 0x7E;
    int e;
    float m = frexpf(x, &e);
    int e4 = e + 6;
    int mant = (int)ceilf((m - 0.5f) * 16.0f);
    if (mant >= 8) { mant = 0; e4++; }
    if (e4 < 0) {
        int sm = (int)ceilf(x * 512.0f);
        return (unsigned char)(sm > 7 ? 7 : sm);
    }
    if (e4 > 15 || (e4 == 15 && mant >= 7)) return 0x7E;
    return (unsigned char)((e4 << 3) | mant);
}
__device__ __forceinline__ float ue4m3_f(unsigned char s) {
    int e = (s >> 3) & 0xF, m = s & 7;
    if (e == 0) return (float)m * 0.001953125f;
    return (1.0f + m / 8.0f) * exp2f((float)e - 7);
}

__device__ __forceinline__ void mxf4_mma(
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1, unsigned sfa, unsigned sfb,
    float& d0, float& d1, float& d2, float& d3)
{
    const unsigned short z = 0;
    asm volatile(
    "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%0,  %1,  %2,  %3},"
    "{%10},"
    "{%11, %12},"
    "{%13},"
    "{%14, %15};\n"
    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
      "r"(b0), "r"(b1),
      "r"(sfa), "h"(z), "h"(z),
      "r"(sfb), "h"(z), "h"(z));
}
__device__ __forceinline__ void cp16(void* smem, const void* gmem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template<int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

// ---------------------------------------------------------------------------
// Activation quant + pack: bf16 X [rows][K] row-major -> Bp/SFB in the OMMA B layouts
// (8-token groups per 64-K kstep):
//   Bp[((q*nks+ks)*32 + lane)*2 + r]  u32  (r=0: k 8t..8t+7, r=1: k+32 of token g = lane>>2)
//   SFB[(q*nks+ks)*32 + lane]         u32  (byte v = scale(token g, kblock ks*4+v); lanes t==0)
// scale = e4m3_ceil(amax16 * x_gs / 6), codes = rn(x * x_gs / ue4m3(scale)). Rows >= M are
// zero-filled (exact 0 contributions). grid (nks, Mpad/8), block 256 = 8 rows x 32 lanes.
// Port of gpu_mxfp4.cu mxfp4_quant_pack_prefill_b with the input global scale.
// ---------------------------------------------------------------------------
extern "C" __global__ void w4a4_quant_pack_b(
    const bf16* __restrict__ X, int K, int M, int Mpad,
    uint32_t* __restrict__ Bp, uint32_t* __restrict__ SFB, float x_gs)
{
    const int ks = blockIdx.x, q = blockIdx.y;
    const int nks = gridDim.x;
    const int n = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int row = q * 8 + n;
    __shared__ float sh[8][64];
    if (row < M) {
        const bf16* xr = X + (long long)row * K + (long long)ks * 64;
#pragma unroll
        for (int i = 0; i < 2; i++) sh[n][lane * 2 + i] = b2f(xr[lane * 2 + i]) * x_gs;
    } else {
#pragma unroll
        for (int i = 0; i < 2; i++) sh[n][lane * 2 + i] = 0.0f;
    }
    __syncthreads();
    __shared__ float sc[8][4];
    if (lane < 4) {
        float amax = 0.f;
#pragma unroll
        for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(sh[n][lane * 16 + i]));
        sc[n][lane] = e4m3_ceil(amax / 6.0f);
    }
    __syncthreads();
    uint32_t b0 = 0, b1 = 0;
    const float inv0 = sc[g][t >> 1] == 0.f ? 0.f : 1.0f / ue4m3_f((unsigned char)sc[g][t >> 1]);
    const float inv1 = sc[g][2 + (t >> 1)] == 0.f ? 0.f : 1.0f / ue4m3_f((unsigned char)sc[g][2 + (t >> 1)]);
#pragma unroll
    for (int j = 0; j < 4; j++) {
        unsigned cb = (unsigned)cvt_e2m1x2(sh[g][8 * t + 2 * j] * inv0, sh[g][8 * t + 2 * j + 1] * inv0);
        b0 |= cb << (8 * j);
    }
#pragma unroll
    for (int j = 0; j < 4; j++) {
        unsigned cb = (unsigned)cvt_e2m1x2(sh[g][32 + 8 * t + 2 * j] * inv1, sh[g][32 + 8 * t + 2 * j + 1] * inv1);
        b1 |= cb << (8 * j);
    }
    Bp[((size_t)q * nks + ks) * 64 + lane * 2 + 0] = b0;
    Bp[((size_t)q * nks + ks) * 64 + lane * 2 + 1] = b1;
    uint32_t sfb = 0;
    if (t == 0) {
        unsigned char b0c = (unsigned char)sc[g][0], b1c = (unsigned char)sc[g][1];
        unsigned char b2c = (unsigned char)sc[g][2], b3c = (unsigned char)sc[g][3];
        sfb = (uint32_t)b0c | ((uint32_t)b1c << 8) | ((uint32_t)b2c << 16) | ((uint32_t)b3c << 24);
    }
    SFB[((size_t)q * nks + ks) * 32 + lane] = sfb;
}

// ---------------------------------------------------------------------------
// The 128x128 block GEMM core (gpu_mxfp4.cu v2 pipeline: 4-stage cp.async, 8 warps of 32x64,
// smem-transposed token-major epilogue) with A fragments gathered from the standard tiled layout.
//   wt/sct/gs : ONE matrix [Mf rows] in standard tiled form (tile (mt, kb) at (mt*nblk+kb)*128 B,
//               scale tile at (mt*nblk+kb)*16 B, gs[mt]); nblk = K/16, nks = K/64.
//   Bp/SFB    : packed rows (global 8-row groups); rows [bn, bn+128) of this block; n_end bounds
//               the B loads and the stores (rows >= n_end are never read nor written).
//   Out       : token-major [rows][Mf]; features [bm, bm+128).
// ---------------------------------------------------------------------------
#define W4_BM 128
#define W4_BN 128
#define W4_STAGE_A 4096
#define W4_STAGE_B 4096
#define W4_STAGE_SFA 512
#define W4_STAGE_SFB 2048
#define W4_STAGE_BYTES (W4_STAGE_A + W4_STAGE_B + W4_STAGE_SFA + W4_STAGE_SFB)  // 10752
#define W4_NSTAGES 4
#define W4_SMEM (W4_NSTAGES * W4_STAGE_BYTES)                                   // 43008

// Narrow decode/verify specialization: one 8-token group instead of the wide core's 128-token
// tile. The hardware MMA is m16n8k64, so this removes the 15 unused n8 fragments (and the second
// token warp group) rather than computing 128 rows and discarding 120 of them.
#define W4_N8_STAGE_A 4096
#define W4_N8_STAGE_B 256
#define W4_N8_STAGE_SFA 512
#define W4_N8_STAGE_SFB 128
#define W4_N8_STAGE_BYTES (W4_N8_STAGE_A + W4_N8_STAGE_B + W4_N8_STAGE_SFA + W4_N8_STAGE_SFB)
#define W4_N8_SMEM (W4_NSTAGES * W4_N8_STAGE_BYTES)                              // 19968

// Swizzled position of source 16-B chunk `ch` (= tile kb = ch>>3, lane-group g = ch&7) inside the
// 512-B A slice: the two chunks a lane reads (tiles kb and kb+2... no: kb = t>>1 and 2 + (t>>1))
// differ in bit 0 of kb only when t>>1 differs; XOR-ing g with 4*(kb&1) puts them 16 words apart.
__device__ __forceinline__ int w4_swz(int ch) { return (ch & 0x18) | ((ch & 7) ^ (((ch >> 3) & 1) << 2)); }

// byte j of each of the four words of q, packed LSB-first
__device__ __forceinline__ uint32_t w4_gather(uint4 q, int j) {
    const unsigned sel = (unsigned)j | ((unsigned)(4 + j) << 4);
    uint32_t p01 = __byte_perm(q.x, q.y, sel);
    uint32_t p23 = __byte_perm(q.z, q.w, sel);
    return __byte_perm(p01, p23, 0x5410);
}

__device__ __forceinline__ void w4a4_gemm_core(
    const uint8_t* __restrict__ wt, const uint8_t* __restrict__ sct, const float* __restrict__ gs,
    const uint32_t* __restrict__ Bp, const uint32_t* __restrict__ SFB,
    bf16* __restrict__ Out, int Mf, int K, int bm, int bn, int n_end, float out_scale,
    uint8_t* smem)
{
    const int nks = K >> 6;
    const int tid = threadIdx.x;
    const int lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int warp = tid >> 5;
    const int wm = warp >> 1, wn = warp & 1;

    auto load_stage = [&](int s, int ks) {
        uint8_t* base = smem + s * W4_STAGE_BYTES;
        // flat chunk map: [0,256) A | [256,512) B | [512,544) SFA | [544,672) SFB
#pragma unroll
        for (int i = 0; i < 3; i++) {
            int idx = tid + i * 256;
            if (idx >= 672) break;
            if (idx < 256) {                                    // A: 8 slices x 4 tiles x 128 B
                int sl = idx >> 5, ch = idx & 31;
                int mt = (bm >> 4) + sl;
                uint8_t* dst = base + sl * 512 + w4_swz(ch) * 16;
                if (mt * 16 < Mf)
                    cp16(dst, wt + ((size_t)mt * nks + ks) * 512 + ch * 16);
                else
                    *(uint4*)dst = make_uint4(0u, 0u, 0u, 0u);
            } else if (idx < 512) {                             // B: 16 groups x 256 B
                int j = idx - 256, q = j >> 4, ch = j & 15;
                int qg = (bn >> 3) + q;
                if (qg * 8 < n_end)
                    cp16(base + W4_STAGE_A + q * 256 + ch * 16, Bp + ((size_t)qg * nks + ks) * 64 + ch * 4);
                else
                    *(uint4*)(base + W4_STAGE_A + q * 256 + ch * 16) = make_uint4(0u, 0u, 0u, 0u);
            } else if (idx < 544) {                             // SFA: 8 slices x 4 scale tiles x 16 B
                int j = idx - 512, sl = j >> 2, ch = j & 3;
                int mt = (bm >> 4) + sl;
                uint8_t* dst = base + W4_STAGE_A + W4_STAGE_B + sl * 64 + ch * 16;
                if (mt * 16 < Mf)
                    cp16(dst, sct + ((size_t)mt * nks + ks) * 64 + ch * 16);
                else
                    *(uint4*)dst = make_uint4(0u, 0u, 0u, 0u);
            } else {                                            // SFB: 16 groups x 128 B
                int j = idx - 544, q = j >> 3, ch = j & 7;
                int qg = (bn >> 3) + q;
                uint8_t* dst = base + W4_STAGE_A + W4_STAGE_B + W4_STAGE_SFA + q * 128 + ch * 16;
                if (qg * 8 < n_end)
                    cp16(dst, SFB + ((size_t)qg * nks + ks) * 32 + ch * 4);
                else
                    *(uint4*)dst = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        cp_commit();
    };

    load_stage(0, 0);
    if (nks > 1) load_stage(1, 1);
    if (nks > 2) load_stage(2, 2);

    float acc[2][8][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[i][j][c] = 0.f;

    // per-lane constant gather geometry
    const int kb0 = (t >> 1), kb1 = 2 + (t >> 1);
    const int ch0 = w4_swz((kb0 << 3) | g), ch1 = w4_swz((kb1 << 3) | g);
    const int jb = (t & 1) << 1;

    for (int kt = 0; kt < nks; kt++) {
        const int s = kt % W4_NSTAGES;
        const int committed = (kt + 3 < nks) ? (kt + 3) : nks;
        const int need = committed - kt - 1;
        if (need <= 0) cp_wait<0>();
        else if (need == 1) cp_wait<1>();
        else cp_wait<2>();
        __syncthreads();
        if (kt + 3 < nks) load_stage((kt + 3) % W4_NSTAGES, kt + 3);

        const uint8_t* st = smem + s * W4_STAGE_BYTES;
        const uint8_t* sfa_s8 = st + W4_STAGE_A + W4_STAGE_B;
        const uint32_t* sfb_s = (const uint32_t*)(st + W4_STAGE_A + W4_STAGE_B + W4_STAGE_SFA);

        uint32_t a[2][4], b[8][2], sfa[2], sfb[8];
#pragma unroll
        for (int ma = 0; ma < 2; ma++) {
            const int sl = wm * 2 + ma;
            const uint8_t* As = st + sl * 512;
            uint4 q0 = *(const uint4*)(As + ch0 * 16);
            uint4 q1 = *(const uint4*)(As + ch1 * 16);
            a[ma][0] = w4_gather(q0, jb);
            a[ma][1] = w4_gather(q0, jb + 1);
            a[ma][2] = w4_gather(q1, jb);
            a[ma][3] = w4_gather(q1, jb + 1);
            if (t <= 1) {
                const uint8_t* ss = sfa_s8 + sl * 64 + g + 8 * t;
                sfa[ma] = (uint32_t)ss[0] | ((uint32_t)ss[16] << 8) | ((uint32_t)ss[32] << 16) | ((uint32_t)ss[48] << 24);
            } else sfa[ma] = 0u;
        }
#pragma unroll
        for (int na = 0; na < 8; na++) {
            const uint2* p = (const uint2*)(st + W4_STAGE_A + ((wn * 8 + na) * 32 + lane) * 8);
            b[na][0] = p->x; b[na][1] = p->y;
            sfb[na] = sfb_s[(wn * 8 + na) * 32 + lane];
        }
#pragma unroll
        for (int ma = 0; ma < 2; ma++)
#pragma unroll
            for (int na = 0; na < 8; na++)
                mxf4_mma(a[ma][0], a[ma][1], a[ma][2], a[ma][3],
                         b[na][0], b[na][1], sfa[ma], sfb[na],
                         acc[ma][na][0], acc[ma][na][1], acc[ma][na][2], acc[ma][na][3]);
    }

    // smem-transpose epilogue (aliases the dead stages): token-major tile [128 tok][136 feat]
    __syncthreads();
    bf16* tile = (bf16*)smem;
    const int TSTRIDE = 136;
#pragma unroll
    for (int ma = 0; ma < 2; ma++) {
#pragma unroll
        for (int na = 0; na < 8; na++) {
#pragma unroll
            for (int c = 0; c < 4; c++) {
                int fl = wm * 32 + ma * 16 + g + 8 * (c >= 2);
                int tl = wn * 64 + na * 8 + 2 * t + (c & 1);
                const int gi = (bm + fl) < Mf ? ((bm + fl) >> 4) : ((Mf - 1) >> 4);
                tile[tl * TSTRIDE + fl] = f2b(acc[ma][na][c] * gs[gi] * out_scale);
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = tid + i * 256;
        int tl = idx >> 4, f8 = (idx & 15) * 8;
        int tok = bn + tl, feat = bm + f8;
        if (tok >= n_end) continue;
        if (feat + 7 < Mf) {
            *(uint4*)(Out + (size_t)tok * Mf + feat) = *(const uint4*)(tile + tl * TSTRIDE + f8);
        } else {
#pragma unroll
            for (int e = 0; e < 8; e++)
                if (feat + e < Mf)
                    Out[(size_t)tok * Mf + feat + e] = tile[tl * TSTRIDE + f8 + e];
        }
    }
}

// One 128-feature x 8-token tile. Four warps cover the four 32-feature bands; each warp issues
// 2 (m16) x 1 (n8) MMAs per K64 instead of the wide core's 2 x 8. Results already have the
// token-major coordinates in registers, so the 128x136 shared-memory transpose is unnecessary.
__device__ __forceinline__ void w4a4_gemm_n8_core(
    const uint8_t* __restrict__ wt, const uint8_t* __restrict__ sct, const float* __restrict__ gs,
    const uint32_t* __restrict__ Bp, const uint32_t* __restrict__ SFB,
    bf16* __restrict__ Out, int Mf, int K, int bm, int bn, int n_end, float out_scale,
    uint8_t* smem)
{
    const int nks = K >> 6;
    const int tid = threadIdx.x;
    const int lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int wm = tid >> 5;
    const int qg = bn >> 3;

    auto load_stage = [&](int s, int ks) {
        uint8_t* base = smem + s * W4_N8_STAGE_BYTES;
#pragma unroll
        for (int i = 0; i < 3; i++) {
            int idx = tid + i * 128;
            if (idx >= 312) break;
            if (idx < 256) {                                    // A: 8 slices x 4 tiles x 128 B
                int sl = idx >> 5, ch = idx & 31;
                int mt = (bm >> 4) + sl;
                uint8_t* dst = base + sl * 512 + w4_swz(ch) * 16;
                if (mt * 16 < Mf)
                    cp16(dst, wt + ((size_t)mt * nks + ks) * 512 + ch * 16);
                else
                    *(uint4*)dst = make_uint4(0u, 0u, 0u, 0u);
            } else if (idx < 272) {                             // B: one 8-token group, 256 B
                int ch = idx - 256;
                cp16(base + W4_N8_STAGE_A + ch * 16,
                     Bp + ((size_t)qg * nks + ks) * 64 + ch * 4);
            } else if (idx < 304) {                             // SFA: 8 slices x 4 chunks
                int j = idx - 272, sl = j >> 2, ch = j & 3;
                int mt = (bm >> 4) + sl;
                uint8_t* dst = base + W4_N8_STAGE_A + W4_N8_STAGE_B + sl * 64 + ch * 16;
                if (mt * 16 < Mf)
                    cp16(dst, sct + ((size_t)mt * nks + ks) * 64 + ch * 16);
                else
                    *(uint4*)dst = make_uint4(0u, 0u, 0u, 0u);
            } else {                                            // SFB: one group, 128 B
                int ch = idx - 304;
                cp16(base + W4_N8_STAGE_A + W4_N8_STAGE_B + W4_N8_STAGE_SFA + ch * 16,
                     SFB + ((size_t)qg * nks + ks) * 32 + ch * 4);
            }
        }
        cp_commit();
    };

    load_stage(0, 0);
    if (nks > 1) load_stage(1, 1);
    if (nks > 2) load_stage(2, 2);

    float acc[2][4];
#pragma unroll
    for (int ma = 0; ma < 2; ma++)
#pragma unroll
        for (int c = 0; c < 4; c++) acc[ma][c] = 0.f;

    const int kb0 = (t >> 1), kb1 = 2 + (t >> 1);
    const int ch0 = w4_swz((kb0 << 3) | g), ch1 = w4_swz((kb1 << 3) | g);
    const int jb = (t & 1) << 1;

    for (int kt = 0; kt < nks; kt++) {
        const int s = kt % W4_NSTAGES;
        const int committed = (kt + 3 < nks) ? (kt + 3) : nks;
        const int need = committed - kt - 1;
        if (need <= 0) cp_wait<0>();
        else if (need == 1) cp_wait<1>();
        else cp_wait<2>();
        __syncthreads();
        if (kt + 3 < nks) load_stage((kt + 3) % W4_NSTAGES, kt + 3);

        const uint8_t* st = smem + s * W4_N8_STAGE_BYTES;
        const uint8_t* sfa_s8 = st + W4_N8_STAGE_A + W4_N8_STAGE_B;
        const uint32_t* sfb_s = (const uint32_t*)(sfa_s8 + W4_N8_STAGE_SFA);
        const uint2* bp = (const uint2*)(st + W4_N8_STAGE_A + lane * 8);
        const uint32_t b0 = bp->x, b1 = bp->y, sfb = sfb_s[lane];

#pragma unroll
        for (int ma = 0; ma < 2; ma++) {
            const int sl = wm * 2 + ma;
            const uint8_t* As = st + sl * 512;
            uint4 q0 = *(const uint4*)(As + ch0 * 16);
            uint4 q1 = *(const uint4*)(As + ch1 * 16);
            uint32_t a0 = w4_gather(q0, jb), a1 = w4_gather(q0, jb + 1);
            uint32_t a2 = w4_gather(q1, jb), a3 = w4_gather(q1, jb + 1);
            uint32_t sfa = 0u;
            if (t <= 1) {
                const uint8_t* ss = sfa_s8 + sl * 64 + g + 8 * t;
                sfa = (uint32_t)ss[0] | ((uint32_t)ss[16] << 8)
                    | ((uint32_t)ss[32] << 16) | ((uint32_t)ss[48] << 24);
            }
            mxf4_mma(a0, a1, a2, a3, b0, b1, sfa, sfb,
                     acc[ma][0], acc[ma][1], acc[ma][2], acc[ma][3]);
        }
    }

#pragma unroll
    for (int ma = 0; ma < 2; ma++) {
#pragma unroll
        for (int c = 0; c < 4; c++) {
            int fl = wm * 32 + ma * 16 + g + 8 * (c >= 2);
            int tok = bn + 2 * t + (c & 1);
            int feat = bm + fl;
            if (tok < n_end && feat < Mf) {
                const int gi = feat >> 4;
                Out[(size_t)tok * Mf + feat] = f2b(acc[ma][c] * gs[gi] * out_scale);
            }
        }
    }
}

// Grid (ceil(Mf/128), ceil(Nt/8)); Nt is restricted to 1..16 by the runtime narrow path.
extern "C" __global__ __launch_bounds__(128, 4) void w4a4_gemm_n8_b(
    const uint8_t* __restrict__ wt, const uint8_t* __restrict__ sct,
    const uint32_t* __restrict__ Bp, const uint32_t* __restrict__ SFB,
    const float* __restrict__ gs, bf16* __restrict__ Out,
    int Mf, int Nt, int K, float out_scale)
{
    const int bm = blockIdx.x * W4_BM;
    const int bn = blockIdx.y * 8;
    if (bm >= Mf || bn >= Nt) return;
    extern __shared__ uint8_t smem[];
    w4a4_gemm_n8_core(wt, sct, gs, Bp, SFB, Out, Mf, K, bm, bn, Nt, out_scale, smem);
}

// Dense: one weight matrix [Mf][K], Nt packed rows. Token-fastest raster (group width 8) so
// co-resident blocks share each weight tile in L2. grid = ceil(tn/8)*8 * tm, dynamic smem W4_SMEM.
extern "C" __global__ __launch_bounds__(256, 2) void w4a4_gemm_b(
    const uint8_t* __restrict__ wt, const uint8_t* __restrict__ sct,
    const uint32_t* __restrict__ Bp, const uint32_t* __restrict__ SFB,
    const float* __restrict__ gs, bf16* __restrict__ Out,
    int Mf, int Nt, int K, float out_scale)
{
    const int tm = (Mf + W4_BM - 1) / W4_BM;
    const int gw = 8;
    const int per_ng = tm * gw;
    const int ng = blockIdx.x / per_ng, rem = blockIdx.x % per_ng;
    const int bm = (rem / gw) * W4_BM;
    const int bn = (ng * gw + rem % gw) * W4_BN;
    if (bn >= Nt) return;
    extern __shared__ uint8_t smem[];
    w4a4_gemm_core(wt, sct, gs, Bp, SFB, Out, Mf, K, bm, bn, Nt, out_scale, smem);
}

// MoE (grouped prefill): 128-row tiles per expert region [poff[e], poff[e+1]) of the permuted
// rows. tmap_e/tmap_row/tcount are built on device by w4a4_moe_tilemap_b; the launch grid is the
// host bound (tiles_max, M/128) and blocks past tcount exit. Expert e's weights are the stacked
// tensor's rows [e*M, (e+1)*M) — a contiguous run of (M/16)*nblk tiles and M/16 gs entries.
// tmap layout: [0] = tile count, [1 .. 1+tiles_max) = expert id, [1+tiles_max ..) = first row.
extern "C" __global__ void w4a4_moe_tilemap_b(int* tmap, const int* __restrict__ poff, int ne, int tiles_max) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int* tmap_e = tmap + 1; int* tmap_row = tmap + 1 + tiles_max;
    int t = 0;
    for (int e = 0; e < ne; e++) {
        const int s = poff[e], end = poff[e + 1];
        for (int r = s; r < end; r += W4_BN) {
            if (t < tiles_max) { tmap_e[t] = e; tmap_row[t] = r; }
            t++;
        }
    }
    tmap[0] = t < tiles_max ? t : tiles_max;
}

extern "C" __global__ __launch_bounds__(256, 2) void w4a4_gemm_moe_b(
    bf16* __restrict__ Out, const uint8_t* __restrict__ wt, const uint8_t* __restrict__ sct,
    const float* __restrict__ gs, const uint32_t* __restrict__ Bp, const uint32_t* __restrict__ SFB,
    const int* __restrict__ tmap, const int* __restrict__ poff, int M, int K, int expert_base, float out_scale)
{
    const int tile = blockIdx.x;
    if (tile >= tmap[0]) return;
    const int tiles_max = (gridDim.x);
    const int e = tmap[1 + tile];
    const int row0 = tmap[1 + tiles_max + tile];
    const int n_end = poff[e + 1];
    const int el = e - expert_base;
    const int nks = K >> 6, ntm = M >> 4;
    const uint8_t* wt_e = wt + (size_t)el * ntm * nks * 512;
    const uint8_t* sct_e = sct + (size_t)el * ntm * nks * 64;
    const float* gs_e = gs + (size_t)el * ntm;
    const int bm = blockIdx.y * W4_BM;
    extern __shared__ uint8_t smem[];
    w4a4_gemm_core(wt_e, sct_e, gs_e, Bp, SFB, Out, M, K, bm, row0, n_end, out_scale, smem);
}

// ---------------------------------------------------------------------------
// Debug (GB10_W4A4_CHECK): fake-quant of X with EXACTLY the packer's recipe — Y = dequant(quant(X))
// in bf16, so the bf16 reference chain run on Y differs from the W4A4 GEMM only by the MMA's own
// f32 accumulation order (relL2 ~1e-3), which isolates a kernel bug from the A4 rounding itself.
// ---------------------------------------------------------------------------
extern "C" __global__ void w4a4_fakequant_b(const bf16* __restrict__ X, bf16* __restrict__ Y, int K, int M, float x_gs) {
    const int ks = blockIdx.x, q = blockIdx.y;
    const int n = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = q * 8 + n;
    __shared__ float sh[8][64];
    __shared__ float sc[8][4];
    if (row < M) {
        const bf16* xr = X + (long long)row * K + (long long)ks * 64;
#pragma unroll
        for (int i = 0; i < 2; i++) sh[n][lane * 2 + i] = b2f(xr[lane * 2 + i]) * x_gs;
    } else {
#pragma unroll
        for (int i = 0; i < 2; i++) sh[n][lane * 2 + i] = 0.0f;
    }
    __syncthreads();
    if (lane < 4) {
        float amax = 0.f;
#pragma unroll
        for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(sh[n][lane * 16 + i]));
        sc[n][lane] = e4m3_ceil(amax / 6.0f);
    }
    __syncthreads();
    if (row >= M) return;
    const float lut[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    bf16* yr = Y + (long long)row * K + (long long)ks * 64;
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int c = lane * 2 + i;
        const unsigned char s = (unsigned char)sc[n][c >> 4];
        float y = 0.f;
        if (s != 0) {
            const float scale = ue4m3_f(s);
            const float inv = 1.0f / scale;   // x * inv, exactly as the packer (not x / scale)
            const unsigned char cb = cvt_e2m1x2(sh[n][c & ~1] * inv, sh[n][c | 1] * inv);
            const unsigned char code = (c & 1) ? (cb >> 4) : (cb & 15);
            y = lut[code] * scale / x_gs;
        }
        yr[c] = f2b(y);
    }
}
