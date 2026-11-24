// 17_exl3_dense_wide.cuh (leg KE2 / P7PF): dense EXL3 linears at prefill widths (> 32 rows), tensor cores.
//   d41x_dw_prep8           xh = fp16(H128(x * suh)), 8 warps x 4 rows/CTA (suh reused across 4 rows) [rows][kloc]
//   d41x_dw_recon           (reconstruct arm) W16T = the decoded slice, n-major [nblk * 128][kloc] f16, any width code
//                           (integer K = 2..8: SMEM-staged D41xDiet + 128-B uint4 stores; fractional K: per-tile warp)
//   d41x_dw_tr / d41x_dw_f16  2 CTAs/SM (256 threads = 16 warps/SM) 64x128 tile GEMMs with 2D L2 CTA swizzling and
//                           L2-cached cp.async, 2-stage ring over 4 k16 steps, one MMA chain per 8-step k-block added in
//                           f32; epilogue H128 + svh, un-aligned slice trim, bf16 / f16 / f32 out, overwrite or add.
//   d41x_dw_post            (cuBLASLt arm) fused 8-warp epilogue H128 + svh + trim + dtype + accumulate on FP32 raw GEMM.
// B source: _tr decodes the trellis per warp from the ring (integer K 2..8); _f16 reads the reconstructed slice with
// ldmatrix (every width code: fractional K too). Both give the exact same bits (same fp16 weights, same MMA order).
// This is the WIDE class's own numerics (one pass over K): not bit-equal to the decode class (12_*/18_*: K split by shape).
#pragma once

#define D41X_DW_STG 3
#define D41X_DW_ASTG (128 * 128)          // geometry handshake constant (128 rows x 64 k x 2 B)
#define D41X_DW_STAGE_EPI (128 * 132 * 4)
#define D41X_DW_ASTG64 (64 * 128)         // 64 rows x 64 k x 2 B (2 CTAs/SM tile)
#define D41X_DW_EPI64 (64 * 132 * 4)

struct D41xDwArgs {
    const uint16_t* tr;          // trellis, tile (0, 0) of the resident tensor (tr); the W16T slice (f16)
    const __half* xh;            // prep output, row 0 / column 0 of the slice, [rows][lda]
    const __half* svh;           // svh offset to the first output block
    void* out; int ocode, ldo, accum;
    int rows, lda, kb0, nkb, nb_stride, nb0, nblk, skip, ncols, bits;
    int w16_ld;                  // f16 source: halves per W16T row (= kloc)
};

// Stage row offset: [N rows][8 chunks of 16 B] (64 k), chunk c of row r at c ^ (r & 7)
__device__ __forceinline__ unsigned d41x_dw_aoff(int row, int chunk) { return (unsigned)(row * 128 + ((chunk ^ (row & 7)) << 4)); }

__device__ __forceinline__ void d41x_dw_cp16_z(unsigned dst, const void* src, bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(ok ? 16 : 0));
}

// 2D L2 CTA swizzle: groups GROUP_M row-tiles across nblk column blocks so L2 reuses both A and B tiles.
template <int GROUP_M>
__device__ __forceinline__ void d41x_dw_swizzle(int nblk, int nrt, int& blk, int& rt) {
    const int pid = blockIdx.y * nblk + blockIdx.x;
    const int num_pid_in_group = GROUP_M * nblk;
    const int group_id = pid / num_pid_in_group;
    const int first_rt = group_id * GROUP_M;
    const int group_size_m = min(nrt - first_rt, GROUP_M);
    const int in_grp = pid % num_pid_in_group;
    rt = first_rt + (in_grp % group_size_m);
    blk = in_grp / group_size_m;
}

template <int SRC_K>
__device__ __forceinline__ void d41x_dw_tr_body(const D41xDwArgs& a, unsigned char* smem) {
    constexpr int TB = 32 * SRC_K;
    constexpr int BSTG = 4 * 8 * TB;
    constexpr int STG = D41X_DW_ASTG64 + BSTG;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    int blk, rt;
    d41x_dw_swizzle<16>(a.nblk, (a.rows + 63) >> 6, blk, rt);
    const int rows = min(64, a.rows - rt * 64);
    if (rows <= 0) return;
    const unsigned s_u = (unsigned)__cvta_generic_to_shared(smem);
    const int NS = a.nkb / 4;                                    // stages (kloc is whole 128-blocks: nkb % 8 == 0)
    const __half* xh0 = a.xh + (size_t)rt * 64 * a.lda;
    auto issue = [&](int st) {
        const int buf = st & 1;
        const unsigned au = s_u + (unsigned)(buf * STG), bu = au + D41X_DW_ASTG64;
        for (int c = t; c < 64 * 8; c += 256) {
            const int r = c >> 3, ch = c & 7;
            d41x_dw_cp16_z(au + d41x_dw_aoff(r, ch), xh0 + (size_t)(r < rows ? r : 0) * a.lda + st * 64 + ch * 8, r < rows);
        }
        const unsigned char* base = (const unsigned char*)a.tr + ((size_t)(a.kb0 + st * 4) * a.nb_stride + a.nb0 + blk * 8) * TB;
        for (int c = t; c < BSTG / 16; c += 256) {
            const int kk = c / (8 * TB / 16), cc = c - kk * (8 * TB / 16);
            d41x_cp16(bu + (unsigned)(kk * 8 * TB + cc * 16), base + (size_t)kk * a.nb_stride * TB + cc * 16);
        }
        d41_cp_commit();
    };
    float sum[4][2][4], blkacc[4][2][4];
    #pragma unroll
    for (int m = 0; m < 4; ++m)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int r = 0; r < 4; ++r) { sum[m][h][r] = 0.f; blkacc[m][h][r] = 0.f; }
    issue(0);
    for (int st = 0; st < NS; ++st) {
        d41_cp_wait<0>();
        __syncthreads();
        if (st + 1 < NS) issue(st + 1); else d41_cp_commit();
        const int buf = st & 1;
        const unsigned au = s_u + (unsigned)(buf * STG);
        const unsigned char* bp = smem + buf * STG + D41X_DW_ASTG64;
        for (int kk = 0; kk < 4; ++kk) {
            unsigned bf[2][2];
            D41xDiet<SRC_K> D; D.init(lane);
            D.decode((const uint32_t*)(bp + (kk * 8 + warp) * TB), bf);
            #pragma unroll
            for (int m = 0; m < 4; ++m) {
                unsigned a0, a1, a2, a3;
                d41x_ldsm4(au + d41x_dw_aoff(16 * m + (lane & 15), 2 * kk + (lane >> 4)), a0, a1, a2, a3);
                d41x_mma(blkacc[m][0], a0, a1, a2, a3, bf[0][0], bf[0][1]);
                d41x_mma(blkacc[m][1], a0, a1, a2, a3, bf[1][0], bf[1][1]);
            }
        }
        if (st & 1) {
            #pragma unroll
            for (int m = 0; m < 4; ++m)
                #pragma unroll
                for (int h = 0; h < 2; ++h)
                    #pragma unroll
                    for (int r = 0; r < 4; ++r) { sum[m][h][r] = __fadd_rn(sum[m][h][r], blkacc[m][h][r]); blkacc[m][h][r] = 0.f; }
        }
    }
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();
    float* stg = (float*)smem;
    const int gid = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int m = 0; m < 4; ++m)
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r0 = 16 * m + gid, col = warp * 16 + 8 * h + 2 * tig;
            *(float2*)(stg + r0 * 132 + col) = make_float2(sum[m][h][0], sum[m][h][1]);
            *(float2*)(stg + (r0 + 8) * 132 + col) = make_float2(sum[m][h][2], sum[m][h][3]);
        }
    __syncthreads();
    for (int r = warp; r < rows; r += 8) {
        float v[4];
        #pragma unroll
        for (int q = 0; q < 4; ++q) v[q] = stg[r * 132 + lane * 4 + q];
        d41x_had128(v, lane);
        const size_t orow = (size_t)(rt * 64 + r) * a.ldo;
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int c = blk * 128 + lane * 4 + q, oc = c - a.skip;
            if (oc < 0 || oc >= a.ncols) continue;
            float o = __fmul_rn(v[q], __half2float(a.svh[c]));
            if (a.accum) o = __fadd_rn(o, d41x_ld(a.out, a.ocode, orow + oc));
            d41x_st(a.out, a.ocode, orow + oc, o);
        }
    }
}

extern "C" __global__ void __launch_bounds__(256, 2) d41x_dw_tr(D41xDwArgs a) {
    D41_PDL_ENTRY();
    extern __shared__ __align__(128) unsigned char d41x_dw_smem[];
    switch (a.bits) {
        case 2: d41x_dw_tr_body<2>(a, d41x_dw_smem); break;
        case 3: d41x_dw_tr_body<3>(a, d41x_dw_smem); break;
        case 4: d41x_dw_tr_body<4>(a, d41x_dw_smem); break;
        case 5: d41x_dw_tr_body<5>(a, d41x_dw_smem); break;
        case 6: d41x_dw_tr_body<6>(a, d41x_dw_smem); break;
        case 7: d41x_dw_tr_body<7>(a, d41x_dw_smem); break;
        case 8: d41x_dw_tr_body<8>(a, d41x_dw_smem); break;
        default: break;
    }
}

extern "C" __global__ void __launch_bounds__(256, 2) d41x_dw_f16(D41xDwArgs a) {
    D41_PDL_ENTRY();
    extern __shared__ __align__(128) unsigned char d41x_dw_smem[];
    constexpr int BSTG128 = 128 * 128;
    constexpr int STG = D41X_DW_ASTG64 + BSTG128;                // 24 KB/stage * 2 stages = 48 KB -> 2 CTAs/SM
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int wm = warp >> 2, wn = warp & 3;                     // 2 M-groups (32 rows) x 4 N-groups (32 cols)
    int blk, rt;
    d41x_dw_swizzle<16>(a.nblk, (a.rows + 63) >> 6, blk, rt);
    const int rows = min(64, a.rows - rt * 64);
    if (rows <= 0) return;
    const unsigned s_u = (unsigned)__cvta_generic_to_shared(d41x_dw_smem);
    const int NS = a.nkb / 4;
    const __half* xh0 = a.xh + (size_t)rt * 64 * a.lda;
    const __half* w0 = (const __half*)a.tr + (size_t)blk * 128 * a.w16_ld;
    auto issue = [&](int st) {
        const int buf = st & 1;
        const unsigned au = s_u + (unsigned)(buf * STG), bu = au + D41X_DW_ASTG64;
        if (rows == 64) {
            #pragma unroll
            for (int c = t; c < 64 * 8; c += 256) {
                const int r = c >> 3, ch = c & 7;
                d41x_cp16(au + d41x_dw_aoff(r, ch), xh0 + (size_t)r * a.lda + st * 64 + ch * 8);
            }
        } else {
            #pragma unroll
            for (int c = t; c < 64 * 8; c += 256) {
                const int r = c >> 3, ch = c & 7;
                d41x_dw_cp16_z(au + d41x_dw_aoff(r, ch), xh0 + (size_t)(r < rows ? r : 0) * a.lda + st * 64 + ch * 8, r < rows);
            }
        }
        #pragma unroll
        for (int c = t; c < 128 * 8; c += 256) {
            const int n = c >> 3, ch = c & 7;
            d41x_cp16(bu + d41x_dw_aoff(n, ch), w0 + (size_t)n * a.w16_ld + st * 64 + ch * 8);
        }
        d41_cp_commit();
    };
    float sum[2][4][4], blkacc[2][4][4];
    #pragma unroll
    for (int m = 0; m < 2; ++m)
        #pragma unroll
        for (int h = 0; h < 4; ++h)
            #pragma unroll
            for (int r = 0; r < 4; ++r) { sum[m][h][r] = 0.f; blkacc[m][h][r] = 0.f; }
    issue(0);
    for (int st = 0; st < NS; ++st) {
        d41_cp_wait<0>();
        __syncthreads();
        if (st + 1 < NS) issue(st + 1); else d41_cp_commit();
        const int buf = st & 1;
        const unsigned au = s_u + (unsigned)(buf * STG), bu = au + D41X_DW_ASTG64;
        #pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            unsigned bf[4][2];
            d41x_ldsm4(bu + d41x_dw_aoff(32 * wn + (lane & 7) + 8 * (lane >> 4), 2 * kk + ((lane >> 3) & 1)), bf[0][0], bf[0][1], bf[1][0], bf[1][1]);
            d41x_ldsm4(bu + d41x_dw_aoff(32 * wn + 16 + (lane & 7) + 8 * (lane >> 4), 2 * kk + ((lane >> 3) & 1)), bf[2][0], bf[2][1], bf[3][0], bf[3][1]);
            #pragma unroll
            for (int m = 0; m < 2; ++m) {
                unsigned a0, a1, a2, a3;
                d41x_ldsm4(au + d41x_dw_aoff(32 * wm + 16 * m + (lane & 15), 2 * kk + (lane >> 4)), a0, a1, a2, a3);
                #pragma unroll
                for (int h = 0; h < 4; ++h) d41x_mma(blkacc[m][h], a0, a1, a2, a3, bf[h][0], bf[h][1]);
            }
        }
        if (st & 1) {
            #pragma unroll
            for (int m = 0; m < 2; ++m)
                #pragma unroll
                for (int h = 0; h < 4; ++h)
                    #pragma unroll
                    for (int r = 0; r < 4; ++r) { sum[m][h][r] = __fadd_rn(sum[m][h][r], blkacc[m][h][r]); blkacc[m][h][r] = 0.f; }
        }
    }
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();
    float* stg = (float*)d41x_dw_smem;
    const int gid = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int m = 0; m < 2; ++m)
        #pragma unroll
        for (int h = 0; h < 4; ++h) {
            const int r0 = 32 * wm + 16 * m + gid, col = wn * 32 + 8 * h + 2 * tig;
            *(float2*)(stg + r0 * 132 + col) = make_float2(sum[m][h][0], sum[m][h][1]);
            *(float2*)(stg + (r0 + 8) * 132 + col) = make_float2(sum[m][h][2], sum[m][h][3]);
        }
    __syncthreads();
    for (int r = warp; r < rows; r += 8) {
        float v[4];
        #pragma unroll
        for (int q = 0; q < 4; ++q) v[q] = stg[r * 132 + lane * 4 + q];
        d41x_had128(v, lane);
        const size_t orow = (size_t)(rt * 64 + r) * a.ldo;
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int c = blk * 128 + lane * 4 + q, oc = c - a.skip;
            if (oc < 0 || oc >= a.ncols) continue;
            float o = __fmul_rn(v[q], __half2float(a.svh[c]));
            if (a.accum) o = __fadd_rn(o, d41x_ld(a.out, a.ocode, orow + oc));
            d41x_st(a.out, a.ocode, orow + oc, o);
        }
    }
}

// Reconstruct (fractional K): one warp per 16x16 tile of the slice (k-steps [kb0, kb0 + nkb), n-tiles [nb0, nb0 + nbn));
// W16T[n][k] (n relative to the slice's first block, k relative to kb0). Values = the GEMM's B fragments exactly.
template <int KA, bool H>
__device__ __forceinline__ void d41x_dw_recon_tile(const D41xDwArgs& a, int kt, int nt, int lane) {
    constexpr int WR = D41xW<KA, H>::WR;
    D41xSlots S; d41x_slots<KA, H>(lane, S);
    const uint32_t* ring = (const uint32_t*)a.tr + ((size_t)(a.kb0 + kt) * a.nb_stride + a.nb0 + nt) * WR;
    unsigned bf[1][2][2];
    d41x_dec_group<KA, H, 1>(ring, 0, S, bf);
    __half* w = (__half*)a.out;
    const int gid = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const size_t n = (size_t)(nt * 16 + 8 * h + gid);
        *(unsigned*)(w + n * a.w16_ld + kt * 16 + 2 * tig) = bf[0][h][0];
        *(unsigned*)(w + n * a.w16_ld + kt * 16 + 2 * tig + 8) = bf[0][h][1];
    }
}

// Reconstruct (integer K = 2..8): CTA (blockIdx.x = nb_blk, blockIdx.y = kb_blk) decodes [128 n][64 k] in SMEM with
// D41xDiet<K> and writes 128-B coalesced uint4 rows to W16T[n][k]. Bit-identical to d41x_dw_recon_tile.
template <int K>
__device__ __forceinline__ void d41x_dw_recon_blk(const D41xDwArgs& a, unsigned char* smem) {
    constexpr int WR = 8 * K;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int gid = lane >> 2, tig = lane & 3;
    const int nb_blk = blockIdx.x, kb_blk = blockIdx.y;
    const int nt = nb_blk * 8 + warp, kt0 = kb_blk * 4;
    D41xDiet<K> D; D.init(lane);
    const uint32_t* ring_base = (const uint32_t*)a.tr + ((size_t)(a.kb0 + kt0) * a.nb_stride + a.nb0 + nt) * WR;
    const size_t kstep = (size_t)a.nb_stride * WR;
    #pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
        if (kt0 + kk < a.nkb) {
            unsigned bf[2][2];
            D.decode(ring_base + kk * kstep, bf);
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = warp * 16 + 8 * h + gid;
                *(unsigned*)(smem + d41x_dw_aoff(r, 2 * kk) + 4 * tig) = bf[h][0];
                *(unsigned*)(smem + d41x_dw_aoff(r, 2 * kk + 1) + 4 * tig) = bf[h][1];
            }
        }
    }
    __syncthreads();
    __half* w_out = (__half*)a.out + (size_t)nb_blk * 128 * a.w16_ld + kt0 * 16;
    const int rem_k = min(64, (a.nkb - kt0) * 16);
    if (rem_k == 64) {
        #pragma unroll
        for (int c = t; c < 128 * 8; c += 256) {
            const int r = c >> 3, ch = c & 7;
            *(uint4*)(w_out + (size_t)r * a.w16_ld + ch * 8) = *(const uint4*)(smem + d41x_dw_aoff(r, ch));
        }
    } else {
        for (int c = t; c < 128 * (rem_k / 8); c += 256) {
            const int nch = rem_k / 8, r = c / nch, ch = c % nch;
            *(uint4*)(w_out + (size_t)r * a.w16_ld + ch * 8) = *(const uint4*)(smem + d41x_dw_aoff(r, ch));
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) d41x_dw_recon(D41xDwArgs a) {
    D41_PDL_ENTRY();
    __shared__ __align__(128) unsigned char smem[128 * 128];
    if (a.bits >= 17) {
        const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        const int nbn = a.nblk * 8;
        const long long tile = ((long long)blockIdx.y * gridDim.x + blockIdx.x) * 8 + warp;
        if (tile >= (long long)a.nkb * nbn) return;
        const int kt = (int)(tile / nbn), nt = (int)(tile % nbn);
        switch (a.bits) {
            case D41X_FRAC(1): d41x_dw_recon_tile<1, true>(a, kt, nt, lane); break;
            case D41X_FRAC(2): d41x_dw_recon_tile<2, true>(a, kt, nt, lane); break;
            case D41X_FRAC(3): d41x_dw_recon_tile<3, true>(a, kt, nt, lane); break;
            default: break;
        }
        return;
    }
    switch (a.bits) {
        case 2: d41x_dw_recon_blk<2>(a, smem); break;
        case 3: d41x_dw_recon_blk<3>(a, smem); break;
        case 4: d41x_dw_recon_blk<4>(a, smem); break;
        case 5: d41x_dw_recon_blk<5>(a, smem); break;
        case 6: d41x_dw_recon_blk<6>(a, smem); break;
        case 7: d41x_dw_recon_blk<7>(a, smem); break;
        case 8: d41x_dw_recon_blk<8>(a, smem); break;
        default: break;
    }
}

// Multi-warp wide prep (8 warps x 4 rows = 32 rows/CTA, suh loaded once per warp across 4 rows; bit-identical to d41x_prep).
extern "C" __global__ void __launch_bounds__(256) d41x_dw_prep8(const void* __restrict__ x, int xcode, int ldx,
                                                                const __half* __restrict__ suh, int rows,
                                                                __half* __restrict__ xh, int lda) {
    D41_PDL_ENTRY();
    const int blk = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = (blockIdx.y * 8 + warp) * 4;
    if (r0 >= rows) return;
    float sv[4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) sv[r] = __half2float(suh[blk * 128 + lane * 4 + r]);
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        if (r0 + q >= rows) break;
        float v[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r) v[r] = d41x_ld(x, xcode, (size_t)(r0 + q) * ldx + blk * 128 + lane * 4 + r);
        d41x_prep4(v, sv, lane);
        __half o[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r) o[r] = __float2half_rn(v[r]);
        *(uint2*)(xh + (size_t)(r0 + q) * lda + blk * 128 + lane * 4) = *(const uint2*)o;
    }
}

// Fused 8-warp epilogue for the cuBLASLt arm: FP32 raw [rows][nblk*128] -> H128 + svh + trim + dtype + accumulate.
extern "C" __global__ void __launch_bounds__(256) d41x_dw_post(const float* __restrict__ yraw, D41xDwArgs a) {
    D41_PDL_ENTRY();
    const int blk = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = (blockIdx.y * 8 + warp) * 4;
    if (r0 >= a.rows) return;
    const int ldy = a.nblk * 128;
    float sv[4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) sv[r] = __half2float(a.svh[blk * 128 + lane * 4 + r]);
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int row = r0 + q;
        if (row >= a.rows) break;
        const float4 f4 = *(const float4*)(yraw + (size_t)row * ldy + blk * 128 + lane * 4);
        float v[4] = {f4.x, f4.y, f4.z, f4.w};
        d41x_had128(v, lane);
        const size_t orow = (size_t)row * a.ldo;
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int c = blk * 128 + lane * 4 + r, oc = c - a.skip;
            if (oc < 0 || oc >= a.ncols) continue;
            float o = __fmul_rn(v[r], sv[r]);
            if (a.accum) o = __fadd_rn(o, d41x_ld(a.out, a.ocode, orow + oc));
            d41x_st(a.out, a.ocode, orow + oc, o);
        }
    }
}

extern "C" __global__ void d41x_dw_sizes(int* o) { if (threadIdx.x == 0 && blockIdx.x == 0) { o[0] = (int)sizeof(D41xDwArgs); o[1] = D41X_DW_ASTG; o[2] = D41X_DW_STAGE_EPI; } }
