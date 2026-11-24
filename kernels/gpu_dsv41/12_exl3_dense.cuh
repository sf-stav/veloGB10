// 12_exl3_dense.cuh (leg KE): dense EXL3 linears: prep (suh + H128) -> trellis GEMM -> post (sum splits + H128 + svh).
// Slices: the GEMM takes (kb0, nkb) on the input side and (nb0, nbn) on the output side of a trellis whose
// resident row length is nb_stride tiles. Out slices need not be 128-aligned (post trims); In slices are whole
// 128-blocks (the caller's activation slice and suh slice start at a block boundary).
// Row classes: rows <= 32 use ONE partition (ks, g) chosen from the shape only, MT = 1 (<= 16 rows) or 2, so
// column 0 of any N-row call is bit-identical to the 1-row call for N = 1..32. rows > 32 (prefill, wide class)
// uses ks = 1 with MT = 4 / 8 super-rows and is its own class.
#pragma once

struct D41xGemmArgs {
    const uint16_t* tr;      // trellis, tile (0,0) of the resident tensor
    const __half* xh;        // activations after prep, [rows][lda], column 0 = first kb of the slice
    float* y;                // raw outputs [g][rows][ldy] (f32)
    long long gstride;       // elements between k-split planes
    int nb_stride, nb0, nbn; // resident row length in n-tiles, first n-tile, n-tile count
    int kb0, nkb;            // first k16 step, step count
    int lda, rows, ldy;
    int ks, g, bits;
};

__shared__ float d41x_red[8 * 2 * 8 * 32];   // one 16 KB block shared by every instantiation (MT <= 2 only)

template <int KA, bool H, int MT>
__device__ __forceinline__ void d41x_gemm_core(const D41xGemmArgs& a, int bx, int by, int bz) {
    constexpr int WR = D41xW<KA, H>::WR;
    float* red = d41x_red;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int ks = a.ks;
    const int nt = warp / ks, kslot = warp % ks;
    const int ntw = 8 / ks;
    const int tile = bx * ntw + nt;
    const int g = by;
    const int mrow0 = bz * 16 * MT;
    const int cg = (a.nkb + a.g - 1) / a.g;
    const int g0 = min(g * cg, a.nkb), g1 = min(g0 + cg, a.nkb);
    const int ck = (g1 - g0 + ks - 1) / ks;
    const int k0 = min(g0 + kslot * ck, g1), k1 = min(k0 + ck, g1);
    float acc[MT][2][4];
    #pragma unroll
    for (int m = 0; m < MT; ++m)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc[m][h][r] = 0.f;
    if (tile < a.nbn && k1 > k0) {
        D41xSlots S; d41x_slots<KA, H>(lane, S);
        const uint32_t* ring0 = (const uint32_t*)a.tr + ((size_t)(a.kb0 + k0) * a.nb_stride + a.nb0 + tile) * WR;
        d41x_tiles<KA, H, MT>(ring0, (size_t)a.nb_stride * WR, k1 - k0, a.xh + (size_t)k0 * 16, a.lda, a.rows, mrow0, S, lane, acc);
    }
    if constexpr (MT <= 2) {
        if (ks > 1) {
            #pragma unroll
            for (int m = 0; m < MT; ++m)
                #pragma unroll
                for (int h = 0; h < 2; ++h)
                    #pragma unroll
                    for (int r = 0; r < 4; ++r) red[((((warp * MT + m) * 2 + h) * 4) + r) * 32 + lane] = acc[m][h][r];
            __syncthreads();
            if (kslot == 0) {
                for (int j = 1; j < ks; ++j) {
                    const int w2 = nt * ks + j;
                    #pragma unroll
                    for (int m = 0; m < MT; ++m)
                        #pragma unroll
                        for (int h = 0; h < 2; ++h)
                            #pragma unroll
                            for (int r = 0; r < 4; ++r) acc[m][h][r] = __fadd_rn(acc[m][h][r], red[((((w2 * MT + m) * 2 + h) * 4) + r) * 32 + lane]);
                }
            }
        }
    }
    if (kslot != 0 || tile >= a.nbn) return;
    const int gid = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int m = 0; m < MT; ++m)
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = tile * 16 + 8 * h + 2 * tig;
            const int r0 = mrow0 + 16 * m + gid, r1 = r0 + 8;
            if (r0 < a.rows) *(float2*)&a.y[g * a.gstride + (size_t)r0 * a.ldy + col] = make_float2(acc[m][h][0], acc[m][h][1]);
            if (r1 < a.rows) *(float2*)&a.y[g * a.gstride + (size_t)r1 * a.ldy + col] = make_float2(acc[m][h][2], acc[m][h][3]);
        }
}

template <int MT>
__device__ __forceinline__ void d41x_gemm_dispatch(const D41xGemmArgs& a, int bx, int by, int bz) {
    switch (a.bits) {
        case 2: d41x_gemm_core<2, false, MT>(a, bx, by, bz); break;
        case 3: d41x_gemm_core<3, false, MT>(a, bx, by, bz); break;
        case 4: d41x_gemm_core<4, false, MT>(a, bx, by, bz); break;
        case 5: d41x_gemm_core<5, false, MT>(a, bx, by, bz); break;
        case 6: d41x_gemm_core<6, false, MT>(a, bx, by, bz); break;
        case 7: d41x_gemm_core<7, false, MT>(a, bx, by, bz); break;
        case 8: d41x_gemm_core<8, false, MT>(a, bx, by, bz); break;
        case D41X_FRAC(1): d41x_gemm_core<1, true, MT>(a, bx, by, bz); break;
        case D41X_FRAC(2): d41x_gemm_core<2, true, MT>(a, bx, by, bz); break;
        case D41X_FRAC(3): d41x_gemm_core<3, true, MT>(a, bx, by, bz); break;
        default: break;
    }
}
extern "C" __global__ void __launch_bounds__(256, 2) d41x_gemm_m1(D41xGemmArgs a) { D41_PDL_ENTRY(); d41x_gemm_dispatch<1>(a, blockIdx.x, blockIdx.y, blockIdx.z); }
extern "C" __global__ void __launch_bounds__(256, 2) d41x_gemm_m2(D41xGemmArgs a) { D41_PDL_ENTRY(); d41x_gemm_dispatch<2>(a, blockIdx.x, blockIdx.y, blockIdx.z); }
extern "C" __global__ void __launch_bounds__(256, 1) d41x_gemm_m4(D41xGemmArgs a) { D41_PDL_ENTRY(); d41x_gemm_dispatch<4>(a, blockIdx.x, blockIdx.y, blockIdx.z); }
extern "C" __global__ void __launch_bounds__(256, 1) d41x_gemm_m8(D41xGemmArgs a) { D41_PDL_ENTRY(); d41x_gemm_dispatch<8>(a, blockIdx.x, blockIdx.y, blockIdx.z); }

// prep: xh[row][blk*128 + j] = fp16( H128( x[row][blk*128 + j] * suh[blk*128 + j] ) ). x / suh already offset to the slice.
extern "C" __global__ void __launch_bounds__(32) d41x_prep(const void* __restrict__ x, int xcode, int ldx,
                                                           const __half* __restrict__ suh, int rows,
                                                           __half* __restrict__ xh, int lda) {
    D41_PDL_ENTRY();
    const int blk = blockIdx.x, row = blockIdx.y, lane = threadIdx.x;
    if (row >= rows) return;
    float v[4], sv[4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) { v[r] = d41x_ld(x, xcode, (size_t)row * ldx + blk * 128 + lane * 4 + r); sv[r] = __half2float(suh[blk * 128 + lane * 4 + r]); }
    d41x_prep4(v, sv, lane);
    #pragma unroll
    for (int r = 0; r < 4; ++r) xh[(size_t)row * lda + blk * 128 + lane * 4 + r] = __float2half_rn(v[r]);
}

// post: out[row][c - skip] (+)= H128( sum_g y[g][row][c] ) * svh[c], c over the nblk*128 computed columns; columns
// outside [skip, skip + ncols) are dropped (an un-aligned output slice). g summed ascending. svh offset to the first block.
extern "C" __global__ void __launch_bounds__(32) d41x_post(const float* __restrict__ y, long long gstride, int g, int ldy,
                                                           const __half* __restrict__ svh, int rows, int skip, int ncols,
                                                           void* __restrict__ out, int ocode, int ldo, int accum) {
    D41_PDL_ENTRY();
    const int blk = blockIdx.x, row = blockIdx.y, lane = threadIdx.x;
    if (row >= rows) return;
    float v[4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) v[r] = y[(size_t)row * ldy + blk * 128 + lane * 4 + r];
    for (int j = 1; j < g; ++j)
        #pragma unroll
        for (int r = 0; r < 4; ++r) v[r] = __fadd_rn(v[r], y[j * gstride + (size_t)row * ldy + blk * 128 + lane * 4 + r]);
    d41x_had128(v, lane);
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int c = blk * 128 + lane * 4 + r;
        const int oc = c - skip;
        if (oc < 0 || oc >= ncols) continue;
        float o = __fmul_rn(v[r], __half2float(svh[c]));
        const size_t oi = (size_t)row * ldo + oc;
        if (accum) o = __fadd_rn(o, d41x_ld(out, ocode, oi));
        d41x_st(out, ocode, oi, o);
    }
}
