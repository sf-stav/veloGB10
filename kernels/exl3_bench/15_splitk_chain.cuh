// exl3_bench/15_splitk_chain.cuh: split-K chain GEMM bodies (LANEWIDE-MID, persistent split-K, combine).
// Identical decode/staging to exl3_gemm_body; FP32 partials in ws[ks_id][m][n].
// Section 15 of 23 of exl3_bench.cu: included in order by exl3_bench.cu, one translation unit; text is moved verbatim.
// ---- split-K chain body: identical decode/staging to exl3_gemm_body, but the
// K loop runs over this block's [kb0, kb1) slice and the epilogue writes FP32
// partials to ws[ks_id][m][n] (layout [ks][M][N]). grid (tiles, KS).
template <int BITS, bool HALF = false>
__device__ __forceinline__ void exl3_gemm_body_ks(const uint16_t* __restrict__ trellis,
                                                  const __half* __restrict__ xh,
                                                  float* __restrict__ ws,
                                                  int M, int K, int N, int cta_tile,
                                                  int kb0, int kb1, int ks_id) {
    constexpr int WRING = HALF ? (4 * (2 * BITS + 1)) : (8 * BITS);
    __shared__ __half sa[16][16];
    __shared__ __half sa2[16][16]; // S-KFIX1: 2-tile pipeline second staging tile (+512 B)
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int cta_n = cta_tile * 128;
    const int nb_col = cta_n / 16 + warp;
    const int nb = N >> 4;

    // hoisted per-lane slot constants (same as exl3_gemm_body)
    int s_i0m[8], s_i1m[8], s_sf[8];
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int r = 2 * (lane & 3) + (s & 1) + 8 * (s >> 1);
            const int c = 8 * h + (lane >> 2);
            const int l = 8 * ((c >> 1) & 3) + ((r & 7) >> 1);
            const int t = 8 * l + (r & 1) + 2 * ((r >> 3) & 1)
                              + 4 * (c >> 3) + 32 * (c & 1);
            const int b1 = HALF ? exl3_half_b1<BITS>(t) : (t + 257) * BITS;
            const int i0 = (b1 - 16) >> 5;
            const int i1 = (b1 - 1) >> 5;
            s_i0m[h * 4 + s] = i0 % WRING;
            s_i1m[h * 4 + s] = i1 % WRING;
            s_sf[h * 4 + s] = ((i1 + 1) << 5) - b1;
        }
    }

    float acc[2][4];
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[h][r] = 0.0f;

    const uint32_t* __restrict__ ring =
        (const uint32_t*)(trellis + ((size_t)kb0 * nb + nb_col) * (HALF ? (8 * (2 * BITS + 1)) : (16 * BITS)));
    const size_t ring_kstride = (size_t)nb * (HALF ? (4u * (2 * BITS + 1)) : (8u * BITS));

    // S-KFIX1: 2-tile software pipeline (2026-10-18, the 1.5bpw per-call deficit):
    // one barrier pair per TWO tiles; tile+1's ring dequant is issued in the same block as
    // tile's, so its latency hides under tile's mma. Tile order and the per-tile mma pair
    // order are unchanged => the accumulation sequence is BITWISE IDENTICAL to the 1-tile
    // loop (probe-binv contract). Odd tail runs the original 1-tile body.
    int kb = kb0;
    for (; kb + 1 < kb1; kb += 2) {
        {
            const int row = tid >> 4, col = tid & 15;
            const bool live0 = (row < M) && (kb * 16 + col < K);
            sa[row][col] = live0 ? xh[(size_t)row * K + kb * 16 + col]
                                 : __ushort_as_half(0);
            const bool live1 = (row < M) && ((kb + 1) * 16 + col < K);
            sa2[row][col] = live1 ? xh[(size_t)row * K + (kb + 1) * 16 + col]
                                  : __ushort_as_half(0);
        }
        __syncthreads();

        const uint32_t* ring1 = ring + ring_kstride;
        uint32_t bf0[2][2], bf1[2][2];
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint16_t r0 = exl3_dq_from(ring,  s_i0m[h*4+0], s_i1m[h*4+0], s_sf[h*4+0]);
            const uint16_t r1 = exl3_dq_from(ring,  s_i0m[h*4+1], s_i1m[h*4+1], s_sf[h*4+1]);
            const uint16_t r2 = exl3_dq_from(ring,  s_i0m[h*4+2], s_i1m[h*4+2], s_sf[h*4+2]);
            const uint16_t r3 = exl3_dq_from(ring,  s_i0m[h*4+3], s_i1m[h*4+3], s_sf[h*4+3]);
            bf0[h][0] = (uint32_t)r0 | ((uint32_t)r1 << 16);
            bf0[h][1] = (uint32_t)r2 | ((uint32_t)r3 << 16);
            const uint16_t u0 = exl3_dq_from(ring1, s_i0m[h*4+0], s_i1m[h*4+0], s_sf[h*4+0]);
            const uint16_t u1 = exl3_dq_from(ring1, s_i0m[h*4+1], s_i1m[h*4+1], s_sf[h*4+1]);
            const uint16_t u2 = exl3_dq_from(ring1, s_i0m[h*4+2], s_i1m[h*4+2], s_sf[h*4+2]);
            const uint16_t u3 = exl3_dq_from(ring1, s_i0m[h*4+3], s_i1m[h*4+3], s_sf[h*4+3]);
            bf1[h][0] = (uint32_t)u0 | ((uint32_t)u1 << 16);
            bf1[h][1] = (uint32_t)u2 | ((uint32_t)u3 << 16);
        }
        ring += 2 * ring_kstride;

        const int gid = lane >> 2, tig = lane & 3;
        const uint32_t a0 = *(const uint32_t*)&sa[gid][2 * tig];
        const uint32_t a1 = *(const uint32_t*)&sa[gid + 8][2 * tig];
        const uint32_t a2 = *(const uint32_t*)&sa[gid][2 * tig + 8];
        const uint32_t a3 = *(const uint32_t*)&sa[gid + 8][2 * tig + 8];
        const uint32_t c0 = *(const uint32_t*)&sa2[gid][2 * tig];
        const uint32_t c1 = *(const uint32_t*)&sa2[gid + 8][2 * tig];
        const uint32_t c2 = *(const uint32_t*)&sa2[gid][2 * tig + 8];
        const uint32_t c3 = *(const uint32_t*)&sa2[gid + 8][2 * tig + 8];

        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[h][0]), "+f"(acc[h][1]), "+f"(acc[h][2]), "+f"(acc[h][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(bf0[h][0]), "r"(bf0[h][1]));
        }
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[h][0]), "+f"(acc[h][1]), "+f"(acc[h][2]), "+f"(acc[h][3])
                : "r"(c0), "r"(c1), "r"(c2), "r"(c3),
                  "r"(bf1[h][0]), "r"(bf1[h][1]));
        }
        __syncthreads();
    }
    for (; kb < kb1; ++kb) {
        {
            const int row = tid >> 4, col = tid & 15;
            const bool live = (row < M) && (kb * 16 + col < K);
            sa[row][col] = live ? xh[(size_t)row * K + kb * 16 + col]
                                : __ushort_as_half(0);
        }
        __syncthreads();

        uint32_t bfrag[2][2];
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint16_t r0 = exl3_dq_from(ring, s_i0m[h*4+0], s_i1m[h*4+0], s_sf[h*4+0]);
            const uint16_t r1 = exl3_dq_from(ring, s_i0m[h*4+1], s_i1m[h*4+1], s_sf[h*4+1]);
            const uint16_t r2 = exl3_dq_from(ring, s_i0m[h*4+2], s_i1m[h*4+2], s_sf[h*4+2]);
            const uint16_t r3 = exl3_dq_from(ring, s_i0m[h*4+3], s_i1m[h*4+3], s_sf[h*4+3]);
            bfrag[h][0] = (uint32_t)r0 | ((uint32_t)r1 << 16);
            bfrag[h][1] = (uint32_t)r2 | ((uint32_t)r3 << 16);
        }
        ring += ring_kstride;

        const int gid = lane >> 2, tig = lane & 3;
        const uint32_t a0 = *(const uint32_t*)&sa[gid][2 * tig];
        const uint32_t a1 = *(const uint32_t*)&sa[gid + 8][2 * tig];
        const uint32_t a2 = *(const uint32_t*)&sa[gid][2 * tig + 8];
        const uint32_t a3 = *(const uint32_t*)&sa[gid + 8][2 * tig + 8];

        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[h][0]), "+f"(acc[h][1]), "+f"(acc[h][2]), "+f"(acc[h][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(bfrag[h][0]), "r"(bfrag[h][1]));
        }
        __syncthreads();
    }

    // ---- fp32 partial epilogue: ws[ks_id][row][col] (rows >= M skipped, as before)
    const int gid = lane >> 2, tig = lane & 3;
    const int row0 = gid, row1 = gid + 8;
    float* base = ws + (size_t)ks_id * M * N;
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int col = cta_n + 16 * warp + 8 * h + 2 * tig;
        if (row0 < M) {
            base[(size_t)row0 * N + col]     = acc[h][0];
            base[(size_t)row0 * N + col + 1] = acc[h][1];
        }
        if (row1 < M) {
            base[(size_t)row1 * N + col]     = acc[h][2];
            base[(size_t)row1 * N + col + 1] = acc[h][3];
        }
    }
}

extern "C" __global__ void exl3_hmma_gemm_ks(const uint16_t* __restrict__ trellis,
                                             const __half* __restrict__ xh,
                                             float* __restrict__ ws,
                                             int M, int K, int N, int bits) {
    // grid.y = KS; KB must be divisible by KS (launcher asserts). Fixed chunking.
    const int ks = gridDim.y;
    const int KB = K >> 4;
    const int chunk = KB / ks;
    const int kb0 = blockIdx.y * chunk;
    const int kb1 = kb0 + chunk;
    XQ_PDL_ENTRY();
    switch (bits) {
        case 6: exl3_gemm_body_ks<3>(trellis, xh, ws, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y); break;
        case 3: exl3_gemm_body_ks<1, true>(trellis, xh, ws, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y); break;
        case 4: exl3_gemm_body_ks<2>(trellis, xh, ws, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y); break;  // K=2.0
        case 8: exl3_gemm_body_ks<4>(trellis, xh, ws, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y); break;
        case 10: exl3_gemm_body_ks<5>(trellis, xh, ws, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y); break;
        case 12: exl3_gemm_body_ks<6>(trellis, xh, ws, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y); break;
        case 14: exl3_gemm_body_ks<7>(trellis, xh, ws, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y); break;  // P7 B6: K=7 (k2=14)
        case 16: exl3_gemm_body_ks<8>(trellis, xh, ws, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y); break; // PR#27-D2: K=8 (ks_p bits-guard 2..=6 falls back here)
        default: break;
    }
}
// ---- LANEWIDE-MID (2026-11-13): the split-K chain body for 17..64 rows (--exl3-mid-gemm).
// exl3_gemm_body_ks with MT m16 tiles per CTA: each warp decodes its 16x16 trellis block ONCE per k16 step and
// feeds it to the MT tiles' mma pairs (the decode, its ALU and the trellis bytes are shared by every row of the
// CTA). Per output element the operation sequence is the m16 body's: the same k-slice [kb0, kb1) of the same
// ks split, the same per-k16 B fragment, the same m16n8k16 accumulation in ascending kb, fp32 partials to
// ws[ks][row][col]; the caller runs the same ascending exl3_ks_combine + svh. A row's result therefore depends on
// nothing but that row (row-independent at every M and every CTA grouping) and equals the decode-class
// (m <= 16) chain's result for that row bit for bit. grid (N/128, KS, ceil(M / (16 * MT))).
#define MID_KS 8
// one file-scope stage shared by every instantiation (switch cases never run together; per-instantiation __shared__
// arrays would add up past the 48 KiB static ceiling)
__shared__ __align__(16) __half exl3_mid_sa[8 * 16][MID_KS * 16 + 8];
template <int BITS, int MT, bool HALF = false>
__device__ __forceinline__ void exl3_gemm_body_ks_mid(const uint16_t* __restrict__ trellis,
                                                      const __half* __restrict__ xh,
                                                      float* __restrict__ ws,
                                                      int M, int K, int N, int cta_tile,
                                                      int kb0, int kb1, int ks_id, int m0) {
    constexpr int WRING = HALF ? (4 * (2 * BITS + 1)) : (8 * BITS);
    __half (&sa)[8 * 16][MID_KS * 16 + 8] = exl3_mid_sa;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int cta_n = cta_tile * 128;
    const int nb_col = cta_n / 16 + warp;
    const int nb = N >> 4;

    int s_i0m[8], s_i1m[8], s_sf[8];
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int r = 2 * (lane & 3) + (s & 1) + 8 * (s >> 1);
            const int c = 8 * h + (lane >> 2);
            const int l = 8 * ((c >> 1) & 3) + ((r & 7) >> 1);
            const int t = 8 * l + (r & 1) + 2 * ((r >> 3) & 1)
                              + 4 * (c >> 3) + 32 * (c & 1);
            const int b1 = HALF ? exl3_half_b1<BITS>(t) : (t + 257) * BITS;
            const int i0 = (b1 - 16) >> 5;
            const int i1 = (b1 - 1) >> 5;
            s_i0m[h * 4 + s] = i0 % WRING;
            s_i1m[h * 4 + s] = i1 % WRING;
            s_sf[h * 4 + s] = ((i1 + 1) << 5) - b1;
        }
    }

    float acc[MT][2][4];
    #pragma unroll
    for (int t = 0; t < MT; ++t)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc[t][h][r] = 0.0f;

    const uint32_t* __restrict__ ring =
        (const uint32_t*)(trellis + ((size_t)kb0 * nb + nb_col) * (HALF ? (8 * (2 * BITS + 1)) : (16 * BITS)));
    const size_t ring_kstride = (size_t)nb * (HALF ? (4u * (2 * BITS + 1)) : (8u * BITS));
    const int gid = lane >> 2, tig = lane & 3;

    // A is staged MID_KS k16 steps at a time with 16-byte loads (one barrier pair per MID_KS steps instead of per
    // step); rows padded by 8 halves so the fragment reads are bank-conflict free. The per-element mma order (kb
    // ascending, the same B fragment per k16) is the m16 body's.
    for (int kbs = kb0; kbs < kb1; kbs += MID_KS) {
        const int nks = min(MID_KS, kb1 - kbs);
        {
            constexpr int VPR = MID_KS * 16 / 8;              // uint4 per row of the stage
            for (int v = tid; v < MT * 16 * VPR; v += 256) {
                const int rr = v / VPR, cv = v % VPR;            // rr = t*16 + row
                const int grow = m0 + rr;
                const int kcol = kbs * 16 + cv * 8;
                uint4 val = make_uint4(0, 0, 0, 0);
                if (grow < M && cv < nks * 2 && kcol + 8 <= K)
                    val = *(const uint4*)(xh + (size_t)grow * K + kcol);
                *(uint4*)&sa[rr][cv * 8] = val;
            }
        }
        __syncthreads();
        int j = 0;
        for (; j < nks; ++j) {
            uint32_t bfrag[2][2];
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint16_t r0 = exl3_dq_from(ring, s_i0m[h*4+0], s_i1m[h*4+0], s_sf[h*4+0]);
                const uint16_t r1 = exl3_dq_from(ring, s_i0m[h*4+1], s_i1m[h*4+1], s_sf[h*4+1]);
                const uint16_t r2 = exl3_dq_from(ring, s_i0m[h*4+2], s_i1m[h*4+2], s_sf[h*4+2]);
                const uint16_t r3 = exl3_dq_from(ring, s_i0m[h*4+3], s_i1m[h*4+3], s_sf[h*4+3]);
                bfrag[h][0] = (uint32_t)r0 | ((uint32_t)r1 << 16);
                bfrag[h][1] = (uint32_t)r2 | ((uint32_t)r3 << 16);
            }
            ring += ring_kstride;
            const int c0 = j * 16;
            #pragma unroll
            for (int t = 0; t < MT; ++t) {
                const uint32_t a0 = *(const uint32_t*)&sa[t * 16 + gid][c0 + 2 * tig];
                const uint32_t a1 = *(const uint32_t*)&sa[t * 16 + gid + 8][c0 + 2 * tig];
                const uint32_t a2 = *(const uint32_t*)&sa[t * 16 + gid][c0 + 2 * tig + 8];
                const uint32_t a3 = *(const uint32_t*)&sa[t * 16 + gid + 8][c0 + 2 * tig + 8];
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(acc[t][h][0]), "+f"(acc[t][h][1]), "+f"(acc[t][h][2]), "+f"(acc[t][h][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(bfrag[h][0]), "r"(bfrag[h][1]));
                }
            }
        }
        __syncthreads();
    }

    float* base = ws + (size_t)ks_id * M * N;
    #pragma unroll
    for (int t = 0; t < MT; ++t) {
        const int row0 = m0 + t * 16 + gid, row1 = row0 + 8;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = cta_n + 16 * warp + 8 * h + 2 * tig;
            if (row0 < M) {
                base[(size_t)row0 * N + col]     = acc[t][h][0];
                base[(size_t)row0 * N + col + 1] = acc[t][h][1];
            }
            if (row1 < M) {
                base[(size_t)row1 * N + col]     = acc[t][h][2];
                base[(size_t)row1 * N + col + 1] = acc[t][h][3];
            }
        }
    }
}

template <int MT>
__device__ __forceinline__ void exl3_ks_mid_dispatch(const uint16_t* trellis, const __half* xh, float* ws,
                                                     int M, int K, int N, int bits, int kb0, int kb1, int m0) {
    const int ct = (int)blockIdx.x, ks = (int)blockIdx.y;
    switch (bits) {
        case 6: exl3_gemm_body_ks_mid<3, MT>(trellis, xh, ws, M, K, N, ct, kb0, kb1, ks, m0); break;
        case 3: exl3_gemm_body_ks_mid<1, MT, true>(trellis, xh, ws, M, K, N, ct, kb0, kb1, ks, m0); break;
        case 4: exl3_gemm_body_ks_mid<2, MT>(trellis, xh, ws, M, K, N, ct, kb0, kb1, ks, m0); break;
        case 5: exl3_gemm_body_ks_mid<2, MT, true>(trellis, xh, ws, M, K, N, ct, kb0, kb1, ks, m0); break;
        case 7: exl3_gemm_body_ks_mid<3, MT, true>(trellis, xh, ws, M, K, N, ct, kb0, kb1, ks, m0); break;
        case 8: exl3_gemm_body_ks_mid<4, MT>(trellis, xh, ws, M, K, N, ct, kb0, kb1, ks, m0); break;
        case 10: exl3_gemm_body_ks_mid<5, MT>(trellis, xh, ws, M, K, N, ct, kb0, kb1, ks, m0); break;
        case 12: exl3_gemm_body_ks_mid<6, MT>(trellis, xh, ws, M, K, N, ct, kb0, kb1, ks, m0); break;
        case 14: exl3_gemm_body_ks_mid<7, MT>(trellis, xh, ws, M, K, N, ct, kb0, kb1, ks, m0); break;
        case 16: exl3_gemm_body_ks_mid<8, MT>(trellis, xh, ws, M, K, N, ct, kb0, kb1, ks, m0); break;
        default: break;
    }
}

// grid (N/128, KS, ceil(M / (16 * mt))); mt in {1, 2, 4, 8}; `bits` = the module's k2 (as exl3_hmma_gemm_ks).
extern "C" __global__ void exl3_hmma_gemm_ks_mid(const uint16_t* __restrict__ trellis,
                                                 const __half* __restrict__ xh,
                                                 float* __restrict__ ws,
                                                 int M, int K, int N, int bits, int mt) {
    const int ks = gridDim.y;
    const int KB = K >> 4;
    const int chunk = KB / ks;
    const int kb0 = blockIdx.y * chunk;
    const int kb1 = kb0 + chunk;
    const int m0 = (int)blockIdx.z * 16 * mt;
    XQ_PDL_ENTRY();
    switch (mt) {
        case 1: exl3_ks_mid_dispatch<1>(trellis, xh, ws, M, K, N, bits, kb0, kb1, m0); break;
        case 2: exl3_ks_mid_dispatch<2>(trellis, xh, ws, M, K, N, bits, kb0, kb1, m0); break;
        case 4: exl3_ks_mid_dispatch<4>(trellis, xh, ws, M, K, N, bits, kb0, kb1, m0); break;
        case 8: exl3_ks_mid_dispatch<8>(trellis, xh, ws, M, K, N, bits, kb0, kb1, m0); break;
        default: break;
    }
}

extern "C" __global__ void exl3_hmma_gemm_ks_x2(const uint16_t* __restrict__ tr0, const uint16_t* __restrict__ tr1,
                                                const __half* __restrict__ xh0, const __half* __restrict__ xh1,
                                                float* __restrict__ ws0, float* __restrict__ ws1,
                                                int M, int K, int N, int bits) {
    // exl3_hmma_gemm_ks per pair member (grid.z): identical chunking and body.
    const bool z = blockIdx.z != 0;
    const uint16_t* trellis = z ? tr1 : tr0;
    const __half* xh = z ? xh1 : xh0;
    float* ws = z ? ws1 : ws0;
    const int ks = gridDim.y;
    const int KB = K >> 4;
    const int chunk = KB / ks;
    const int kb0 = blockIdx.y * chunk;
    const int kb1 = kb0 + chunk;
    XQ_PDL_ENTRY();
    switch (bits) {
        case 6: exl3_gemm_body_ks<3>(trellis, xh, ws, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y); break;
        case 3: exl3_gemm_body_ks<1, true>(trellis, xh, ws, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y); break;
        case 4: exl3_gemm_body_ks<2>(trellis, xh, ws, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y); break;  // K=2.0
        case 8: exl3_gemm_body_ks<4>(trellis, xh, ws, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y); break;
        case 10: exl3_gemm_body_ks<5>(trellis, xh, ws, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y); break;
        case 12: exl3_gemm_body_ks<6>(trellis, xh, ws, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y); break;
        case 14: exl3_gemm_body_ks<7>(trellis, xh, ws, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y); break;  // P7 B6: K=7 (k2=14)
        case 16: exl3_gemm_body_ks<8>(trellis, xh, ws, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y); break; // PR#27-D2: K=8 (ks_p bits-guard 2..=6 falls back here)
        default: break;
    }
}

// ---- WP21: split-K FIXUP epilogue — exl3_hmma_gemm_ks + exl3_ks_combine + exl3_had_svh in
// ONE launch. Every CTA writes its fp32 partials exactly as exl3_gemm_body_ks does, then
// arrives on its N-tile's counter (cnt[tile], one u32 per 128-column tile, owned by the
// Quad => distinct slices per chain in flight). The LAST-arriving CTA of the tile (no spin:
// no co-residency requirement, PDL-safe) re-arms the counter to 0 and runs the fixup:
//   combine : acc = 0; acc += ws[s][r][c] for s = 0..ks-1 ASCENDING (exl3_ks_combine's loop,
//             same fp32 add sequence incl. the 0.0f seed), yraw = f16_rn(acc)
//   svh     : warp w owns rows w, w+8 (< M); lane holds cols 4*lane..4*lane+3 of the tile =
//             exl3_had_svh_dev's (row, 128-block) warp layout; same exl3_had128 butterfly,
//             same y = f16_rn(v * svh) rounding.
// No multiply feeds an add anywhere (no FMA-contraction freedom), so y is BITWISE the
// 3-launch path's. yraw is never materialized (no caller reads it after a chain).
//
// v2 (WP21 fix/tune 1). The v1 nsys A/B on .13 had the fused GEMM at +4.1 us/call (+8.8 us
// on the ks=2 N=10240 grid) and the excess grew with the K-chunk length, so it was the MAIN
// LOOP, not the fixup tail. SASS (nvcc/ptxas 13.0, sm_121), SASS instructions per 2-iteration
// k-loop body for bits 3/4/5:
//     exl3_hmma_gemm_ks (old, unbounded, 64 regs)         250/164/240   (x2: 253/163/242 @66)
//     v1 fx  (__launch_bounds__(256,4), inline fixup)     295/181/278   (x2: 309/177/288)
//     v2 fx  (__launch_bounds__(256), isolated fixup)     243/162/236   (x2: 247/162/239 @64)
// The (256,4) / __maxnreg__(64) cap makes ptxas schedule the loop under a hard 64 target and
// rematerialize address math / decode constants inside it (the same cap gives 294/179/285 even
// with a small fixup); v1's inline fixup also CSE'd tid/tile/ks into registers live across the
// loop. This GEMM runs at ~80% of roofline with a dependent LDG -> decode -> HMMA chain per
// iteration, so +16% loop instructions was +10% kernel time. v2:
//   * __launch_bounds__(256) only (.maxntid, no min-blocks): ptxas lands at 64 regs = 4 CTAs/SM
//     on its own and knows tid < 256 (shorter address math: fewer PRMT/IMAD.WIDE than the old
//     unbounded kernel);
//   * the fixup shares NO value with the body: tid / ctaid.x / nctaid.y are re-read through
//     asm volatile (un-CSE-able) and every pointer comes from the kernel params, so nothing
//     extra is live across the k-loop;
//   * arrival: every CTA does a RELEASE-only RMW (MEMBAR + ATOMG, no CCTL.IVALL: v1's acq_rel
//     invalidated the SM's whole L1 under the co-resident CTAs once per CTA); only the elected
//     last CTA issues the acquire fence (acquire pattern = the strong RMW read + fence.acq_rel);
//     the partials are then read with ld.global.cg (L2);
//   * the combine issues up to EXL3_FX_B = 7 partial loads per lane before adding (ks 2/5 = one
//     L2 round trip, 8/10 = two, 40 = six; v1: unroll 8 + a serial remainder loop); adds stay
//     strictly ascending in s. B = 8 pushes the kernel to 78 regs (3 CTAs/SM) — do not raise;
//   * the svh load is issued before the partial loads (independent; v1 issued it after had128).
// Ordering proof (PTX memory model): every thread's partial stores -> bar.sync -> thread 0
// atom.release.gpu (release pattern, cumulative over the CTA through bar.sync). The last CTA's
// thread 0 RMW reads the value written by the ks-1 earlier RMWs (single modification order of
// cnt[tile]) -> fence.acq_rel.gpu (acquire pattern) -> bar.sync -> every fixup thread's loads
// are causally after every producer's stores. Reset: the elected CTA's relaxed store of 0 is
// the last write to cnt[tile] in this launch; the next launch is stream-ordered after it.
#define EXL3_FX_B 7

__device__ __forceinline__ unsigned xq_sreg_tid_x() {
    unsigned r; asm volatile("mov.u32 %0, %%tid.x;" : "=r"(r)); return r;
}
__device__ __forceinline__ unsigned xq_sreg_ctaid_x() {
    unsigned r; asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(r)); return r;
}
__device__ __forceinline__ unsigned xq_sreg_nctaid_y() {
    unsigned r; asm volatile("mov.u32 %0, %%nctaid.y;" : "=r"(r)); return r;
}

__device__ __forceinline__ void exl3_ks_fixup(const float* __restrict__ ws,
                                              const __half* __restrict__ svh,
                                              __half* __restrict__ y,
                                              unsigned int* __restrict__ cnt,
                                              int M, int N) {
    __shared__ int s_last;
    const unsigned tid = xq_sreg_tid_x();
    const unsigned tile = xq_sreg_ctaid_x();
    const unsigned ks = xq_sreg_nctaid_y();
    unsigned int* const ctr = cnt + tile;
    __syncthreads();                                   // this CTA's partial stores precede the arrival
    if (tid == 0) {
        unsigned prev;
        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;"
                     : "=r"(prev) : "l"(ctr) : "memory");
        const int last = (prev == ks - 1u) ? 1 : 0;
        if (last) {
            asm volatile("fence.acq_rel.gpu;" ::: "memory");          // acquire (elected CTA only)
            asm volatile("st.relaxed.gpu.global.u32 [%0], %1;"          // self re-arming (graph replays)
                         :: "l"(ctr), "r"(0u) : "memory");
        }
        s_last = last;
    }
    __syncthreads();
    if (!s_last) return;
    const int warp = (int)(tid >> 5), lane = (int)(tid & 31u);
    const int col = (int)tile * 128 + lane * 4;
    const size_t sstride = (size_t)M * N;
    const int nks = (int)ks;
    for (int r = warp; r < M; r += 8) {                // warp-uniform: exl3_had128 shuffles full-mask
        // 8-byte svh load (col % 4 == 0, N % 128 == 0 => aligned), independent of the partials.
        const uint2 sv = __ldg((const uint2*)(svh + col));
        const float* p = ws + (size_t)r * N + col;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int s0 = 0; s0 < nks; s0 += EXL3_FX_B) {
            float4 q[EXL3_FX_B];
            #pragma unroll
            for (int j = 0; j < EXL3_FX_B; ++j)          // issue the batch's loads first ...
                if (s0 + j < nks) q[j] = __ldcg((const float4*)(p + (size_t)(s0 + j) * sstride));
            #pragma unroll
            for (int j = 0; j < EXL3_FX_B; ++j)          // ... then add strictly ascending in s
                if (s0 + j < nks) { a0 += q[j].x; a1 += q[j].y; a2 += q[j].z; a3 += q[j].w; }
        }
        float v[4];
        v[0] = __half2float(__float2half_rn(a0));
        v[1] = __half2float(__float2half_rn(a1));
        v[2] = __half2float(__float2half_rn(a2));
        v[3] = __half2float(__float2half_rn(a3));
        exl3_had128(v, lane);
        // values and roundings are exl3_had_svh_dev's per-element f16_rn(v * svh).
        const __half* sh = (const __half*)&sv;
        __half o[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * __half2float(sh[j]));
        *(uint2*)(y + (size_t)r * N + col) = *(const uint2*)o;
    }
}

// exl3_hmma_gemm_ks's chunking + body verbatim on one member's pointers, then the fixup.
#define EXL3_FX_MEMBER(TR, XH, WS, SVH, Y, CNT)                                                     \
    do {                                                                                            \
        switch (bits) {                                                                             \
            case 6: exl3_gemm_body_ks<3>(TR, XH, WS, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y); break; \
            case 3: exl3_gemm_body_ks<1, true>(TR, XH, WS, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y); break; \
            case 4: exl3_gemm_body_ks<2>(TR, XH, WS, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y); break; \
            case 8: exl3_gemm_body_ks<4>(TR, XH, WS, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y); break; \
            case 10: exl3_gemm_body_ks<5>(TR, XH, WS, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y); break; \
            case 12: exl3_gemm_body_ks<6>(TR, XH, WS, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y); break; \
            case 14: exl3_gemm_body_ks<7>(TR, XH, WS, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y); break; \
            default: break;                                                                         \
        }                                                                                           \
        exl3_ks_fixup(WS, SVH, Y, CNT, M, N);                                                       \
    } while (0)

extern "C" __global__ void __launch_bounds__(256) exl3_hmma_gemm_ks_fx(const uint16_t* __restrict__ trellis,
                                                const __half* __restrict__ xh,
                                                float* __restrict__ ws,
                                                const __half* __restrict__ svh,
                                                __half* __restrict__ y,
                                                unsigned int* __restrict__ cnt,
                                                int M, int K, int N, int bits) {
    const int ks = gridDim.y;
    const int KB = K >> 4;
    const int chunk = KB / ks;
    const int kb0 = blockIdx.y * chunk;
    const int kb1 = kb0 + chunk;
    XQ_PDL_ENTRY();
    EXL3_FX_MEMBER(trellis, xh, ws, svh, y, cnt);
}
extern "C" __global__ void __launch_bounds__(256, 4) exl3_hmma_gemm_ks_fx_x2(const uint16_t* __restrict__ tr0, const uint16_t* __restrict__ tr1,
                                                   const __half* __restrict__ xh0, const __half* __restrict__ xh1,
                                                   float* __restrict__ ws0, float* __restrict__ ws1,
                                                   const __half* __restrict__ svh0, const __half* __restrict__ svh1,
                                                   __half* __restrict__ y0, __half* __restrict__ y1,
                                                   unsigned int* __restrict__ cnt0, unsigned int* __restrict__ cnt1,
                                                   int M, int K, int N, int bits) {
    // exl3_hmma_gemm_ks_x2 per pair member (grid.z), each member with its own counter slice.
    // (A per-member branch around body + fixup compiles to the same loop quality at twice the
    // code size; the selects are kept.)
    const bool z = blockIdx.z != 0;
    const int ks = gridDim.y;
    const int KB = K >> 4;
    const int chunk = KB / ks;
    const int kb0 = blockIdx.y * chunk;
    const int kb1 = kb0 + chunk;
    XQ_PDL_ENTRY();
    EXL3_FX_MEMBER(z ? tr1 : tr0, z ? xh1 : xh0, z ? ws1 : ws0, z ? svh1 : svh0, z ? y1 : y0, z ? cnt1 : cnt0);
}
#undef EXL3_FX_MEMBER

// ---- fixed-order combine of the split-K partials: ascending ks, fp32, one
// fp16 RN store per element (same final rounding as the single-block epilogue).

template <int BITS, int MODE>
__device__ __forceinline__ uint16_t exl3_probe_dq(const uint32_t* __restrict__ ring,
                                                        int i0, int i1, int sf) {
    if (MODE == 2) { const uint32_t w0 = ring[i0]; const uint32_t w1 = ring[i1]; (void)sf; return (uint16_t)(w0 ^ w1); }
    return exl3_dq_from(ring, i0, i1, sf);
}
template <int BITS, int MODE>
__device__ __forceinline__ void exl3_gemm_body_ks_probe(const uint16_t* __restrict__ trellis,
                                                  const __half* __restrict__ xh,
                                                  float* __restrict__ ws,
                                                  int M, int K, int N, int cta_tile,
                                                  int kb0, int kb1, int ks_id, int w_kb) {
    __shared__ __half sa[16][16];
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int cta_n = cta_tile * 128;
    const int nb_col = cta_n / 16 + warp;
    const int nb = N >> 4;

    // hoisted per-lane slot constants (same as exl3_gemm_body)
    int s_i0m[8], s_i1m[8], s_sf[8];
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int r = 2 * (lane & 3) + (s & 1) + 8 * (s >> 1);
            const int c = 8 * h + (lane >> 2);
            const int l = 8 * ((c >> 1) & 3) + ((r & 7) >> 1);
            const int t = 8 * l + (r & 1) + 2 * ((r >> 3) & 1)
                              + 4 * (c >> 3) + 32 * (c & 1);
            const int b1 = (t + 257) * BITS;
            const int i0 = (b1 - 16) >> 5;
            const int i1 = (b1 - 1) >> 5;
            s_i0m[h * 4 + s] = i0 % (BITS * 8);
            s_i1m[h * 4 + s] = i1 % (BITS * 8);
            s_sf[h * 4 + s] = ((i1 + 1) << 5) - b1;
        }
    }

    float acc[2][4];
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[h][r] = 0.0f;

    const uint32_t* __restrict__ ring0 =
        (const uint32_t*)(trellis + ((size_t)kb0 * nb + nb_col) * (16 * BITS));
    const size_t ring_kstride = (size_t)nb * 8u * BITS;
    const int span = kb1 - kb0;
    const int wk = (MODE == 1 && w_kb > 0 && w_kb < span) ? w_kb : span;

    for (int kb = kb0; kb < kb1; ++kb) {
        const uint32_t* __restrict__ ring =
            ring0 + (size_t)((kb - kb0) % wk) * ring_kstride;
        {
            const int row = tid >> 4, col = tid & 15;
            const bool live = (row < M) && (kb * 16 + col < K);
            sa[row][col] = live ? xh[(size_t)row * K + kb * 16 + col]
                                : __ushort_as_half(0);
        }
        __syncthreads();

        uint32_t bfrag[2][2];
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint16_t r0 = exl3_probe_dq<BITS, MODE>(ring, s_i0m[h*4+0], s_i1m[h*4+0], s_sf[h*4+0]);
            const uint16_t r1 = exl3_probe_dq<BITS, MODE>(ring, s_i0m[h*4+1], s_i1m[h*4+1], s_sf[h*4+1]);
            const uint16_t r2 = exl3_probe_dq<BITS, MODE>(ring, s_i0m[h*4+2], s_i1m[h*4+2], s_sf[h*4+2]);
            const uint16_t r3 = exl3_probe_dq<BITS, MODE>(ring, s_i0m[h*4+3], s_i1m[h*4+3], s_sf[h*4+3]);
            bfrag[h][0] = (uint32_t)r0 | ((uint32_t)r1 << 16);
            bfrag[h][1] = (uint32_t)r2 | ((uint32_t)r3 << 16);
        }

        const int gid = lane >> 2, tig = lane & 3;
        const uint32_t a0 = *(const uint32_t*)&sa[gid][2 * tig];
        const uint32_t a1 = *(const uint32_t*)&sa[gid + 8][2 * tig];
        const uint32_t a2 = *(const uint32_t*)&sa[gid][2 * tig + 8];
        const uint32_t a3 = *(const uint32_t*)&sa[gid + 8][2 * tig + 8];

        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[h][0]), "+f"(acc[h][1]), "+f"(acc[h][2]), "+f"(acc[h][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(bfrag[h][0]), "r"(bfrag[h][1]));
        }
        __syncthreads();
    }

    // ---- fp32 partial epilogue: ws[ks_id][row][col] (rows >= M skipped, as before)
    const int gid = lane >> 2, tig = lane & 3;
    const int row0 = gid, row1 = gid + 8;
    float* base = ws + (size_t)ks_id * M * N;
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int col = cta_n + 16 * warp + 8 * h + 2 * tig;
        if (row0 < M) {
            base[(size_t)row0 * N + col]     = acc[h][0];
            base[(size_t)row0 * N + col + 1] = acc[h][1];
        }
        if (row1 < M) {
            base[(size_t)row1 * N + col]     = acc[h][2];
            base[(size_t)row1 * N + col + 1] = acc[h][3];
        }
    }
}

// ----------------------------------------------------------------------------
// S-C4STEP5 ablation-ladder probe (bench-harness ONLY — never launched by the
// served path; timed to decompose the ks_fx dense cost, nothing ships from it):
//   MODE 1 = ring wraps into a w_kb-row window (L2-resident trellis reads).
//   MODE 2 = load-preserving decode stub (both ring loads kept, bit assembly
//            removed) — isolates the decode ALU cost.
// Verbatim body of exl3_gemm_body_ks except the two probe points above.
// ----------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256) exl3_hmma_gemm_ks_fx_probe(const uint16_t* __restrict__ trellis,
                                                const __half* __restrict__ xh,
                                                float* __restrict__ ws,
                                                const __half* __restrict__ svh,
                                                __half* __restrict__ y,
                                                unsigned int* __restrict__ cnt,
                                                int M, int K, int N, int bits, int mode, int w_kb) {
    const int ks = gridDim.y;
    const int KB = K >> 4;
    const int chunk = KB / ks;
    const int kb0 = blockIdx.y * chunk;
    const int kb1 = kb0 + chunk;
    XQ_PDL_ENTRY();
    if (mode == 2) {
        switch (bits) {
            case 3: exl3_gemm_body_ks_probe<3,2>(trellis, xh, ws, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y, w_kb); break;
            case 4: exl3_gemm_body_ks_probe<4,2>(trellis, xh, ws, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y, w_kb); break;
            case 5: exl3_gemm_body_ks_probe<5,2>(trellis, xh, ws, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y, w_kb); break;
            case 6: exl3_gemm_body_ks_probe<6,2>(trellis, xh, ws, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y, w_kb); break;
            default: break;
        }
    } else {
        switch (bits) {
            case 3: exl3_gemm_body_ks_probe<3,1>(trellis, xh, ws, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y, w_kb); break;
            case 4: exl3_gemm_body_ks_probe<4,1>(trellis, xh, ws, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y, w_kb); break;
            case 5: exl3_gemm_body_ks_probe<5,1>(trellis, xh, ws, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y, w_kb); break;
            case 6: exl3_gemm_body_ks_probe<6,1>(trellis, xh, ws, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y, w_kb); break;
            default: break;
        }
    }
    exl3_ks_fixup(ws, svh, y, cnt, M, N);
}

extern "C" __global__ void exl3_ks_combine(const float* __restrict__ ws,
                                           __half* __restrict__ yraw,
                                           int M, int N, int KS) {
    XQ_PDL_ENTRY();
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    const int m = idx / N;
    const int n = idx - (size_t)m * N;
    float acc = 0.0f;
    for (int s = 0; s < KS; ++s) acc += ws[((size_t)s * M + m) * N + n];
    yraw[(size_t)m * N + n] = __float2half_rn(acc);
}
extern "C" __global__ void exl3_ks_combine_x2(const float* __restrict__ ws0, const float* __restrict__ ws1,
                                              __half* __restrict__ yraw0, __half* __restrict__ yraw1,
                                              int M, int N, int KS) {
    XQ_PDL_ENTRY();
    const float* ws = blockIdx.z ? ws1 : ws0;
    __half* yraw = blockIdx.z ? yraw1 : yraw0;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    const int m = idx / N;
    const int n = idx - (size_t)m * N;
    float acc = 0.0f;
    for (int s = 0; s < KS; ++s) acc += ws[((size_t)s * M + m) * N + n];
    yraw[(size_t)m * N + n] = __float2half_rn(acc);
}

// =============================================================================
// T4e / w3-PSK: PERSISTENT split-K chain GEMM — exl3_hmma_gemm_ks_p (+ _x2 pair twin).
//
// CONTRACT: every fp32 partial ws[slab][row][col] (rows < M) is BITWISE equal to what
// exl3_hmma_gemm_ks / exl3_hmma_gemm_ks_x2 write, for EVERY grid size G >= 1 and every M in
// 1..16; exl3_ks_combine(_x2) is unchanged, so y_raw (and everything downstream) is unchanged.
// Why it holds: the logical work ITEM is exactly one CTA of the old grid — (member, N128 tile,
// K slab), slab = k16 steps [s*chunk, (s+1)*chunk), chunk = (K/16)/ks, ks the SAME per-shape
// factor (chain_ks) — and each item runs the same per-element instruction sequence:
//   acc = 0.0f; for kb ascending over the slab: B = exl3_dq_from on the same ring words,
//   A = the same xh halves (rows >= M -> 0), two m16n8k16.f32.f16.f16.f32 mma (h = 0, 1);
//   then one fp32 store per element at the same address.
// G only decides WHICH CTA runs an item and WHEN, never HOW => G is a pure schedule knob
// (the autotuner's T4e axis). The partition (ks) never depends on m => batch invariance as
// before (AGENTS 2.4: fixed-order combine, no atomics).
//
// Schedule changes vs one-CTA-per-item (all value-transparent):
//   * G persistent CTAs stride the items: item = blockIdx.x + i*G, linearized tile-fastest,
//     then slab, then member (the old grid's dispatch order, so G == items reproduces the old
//     CTA->item map exactly).
//   * A: PSK_S = 8 k16 steps staged per smem block by cp.async.cg (16 B, L1 bypass), DOUBLE
//     buffered: the next block — possibly the NEXT ITEM's first — is in flight while this one
//     computes; ONE barrier per 8 steps (the old body: a dependent LDG.U16 -> STS -> BAR -> ...
//     -> BAR on EVERY step's critical path). Only rows < M are staged; fragment reads of rows
//     >= M are predicated to 0 (the old body read staged zeros: same register values).
//   * trellis: a per-warp L2 prefetch cursor pf_d windows ahead (lanes 0..BITS-1, 32 B each =
//     the 16x16 block's 32*BITS-byte window) that runs ACROSS item boundaries, so an item's
//     first windows are already requested while the previous item drains (S-A3-n F4 mechanism,
//     here without the A-once smem that cost the old F3+F4 chain trial an occupancy step).
//     The prologue is issued BEFORE the PDL wait (weights are static). pf_d = 0 disables it.
//   * the decode keeps exl3_dq_from's scalar window loads (the F8/F9 record: the shuffle
//     decode regressed b5 chains and mispredicted in-round).
// Resources: static smem 2 x 16 x (8*16+8) halves = 8704 B, owned by the ENTRY (one copy for
// all BITS instantiations); __launch_bounds__(256, 4) = the old kernel's 64-reg / 4-CTA/SM class.
// =============================================================================
#define PSK_S 8                                  // k16 steps of A per staged block
#define PSK_SROW (PSK_S * 16 + 8)                // halves per staged row (+16 B: conflict-free frags)
#define PSK_SMEM_HALVES (2 * 16 * PSK_SROW)      // two buffers x 16 rows

__device__ __forceinline__ void psk_cp16(void* smem_dst, const void* gsrc) {
    const unsigned d = (unsigned)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(d), "l"(__cvta_generic_to_global(gsrc)) : "memory");
}

// u16-element offset (within its member's trellis) of this warp's 16x16 block at the FIRST k16
// step of `item`; mem_out = the item's member (0/1).
__device__ __forceinline__ size_t psk_off(int item, int per_m, int tiles, int chunk, int nb, int warp,
                                          int blk_elems, bool& mem_out) {
    const bool mem = item >= per_m;
    const int r = mem ? item - per_m : item;
    const int slab = r / tiles;
    const int tile = r - slab * tiles;
    mem_out = mem;
    return ((size_t)slab * chunk * nb + tile * 8 + warp) * (size_t)blk_elems;
}

template <int BITS>
__device__ __forceinline__ void exl3_ks_p_body(const uint16_t* __restrict__ tr0,
                                               const uint16_t* __restrict__ tr1,
                                               const __half* __restrict__ xh0,
                                               const __half* __restrict__ xh1,
                                               float* __restrict__ ws0, float* __restrict__ ws1,
                                               const char* pfb0, const char* pfb1,
                                               int M, int K, int N, int ks, int members, int pf_d,
                                               __half* __restrict__ sa_raw) {
    static_assert(PSK_S == 8, "staging map assumes 16 rows x 16 x 16-B pieces = 256 threads");
    typedef __half SaRow[PSK_SROW];
    SaRow* sa = (SaRow*)sa_raw;                  // [buf*16 + row][col]
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int nb = N >> 4;
    const int tiles = N >> 7;
    const int chunk = (K >> 4) / ks;             // launcher asserts (K/16) % ks == 0
    const int per_m = tiles * ks;                // items per member
    const int n_items = per_m * members;
    const int G = (int)gridDim.x;
    const int nblk = (chunk + PSK_S - 1) / PSK_S;
    constexpr int BLK = 16 * BITS;               // u16 elements per 16x16 trellis block
    const int kstride = nb * 8 * BITS;           // u32 words between consecutive k16 steps

    // ---- trellis L2 prefetch, one burst per staged block, pf_d blocks ahead of the compute
    // block (value-transparent hint). The cursor walks the CTA's (item, block) sequence, so it
    // crosses item boundaries: the next item's first windows are requested while this one runs.
    int pf_it = (int)blockIdx.x, pf_bk = 0;
    auto pf_burst = [&]() {
        if (pf_it < n_items) {
            if (lane < BITS) {
                bool pm;
                const size_t off = psk_off(pf_it, per_m, tiles, chunk, nb, warp, BLK, pm);
                const char* p = (pm ? pfb1 : pfb0) + off * 2u
                              + ((size_t)pf_bk * PSK_S * kstride) * 4u + lane * 32;
                const int ns = min(PSK_S, chunk - pf_bk * PSK_S);
                for (int j = 0; j < ns; ++j, p += (size_t)kstride * 4u)
                    asm volatile("prefetch.global.L2 [%0];" :: "l"(__cvta_generic_to_global(p)));
            }
            if (++pf_bk == nblk) { pf_bk = 0; pf_it += G; }
        }
    };
    for (int d = 0; d < pf_d; ++d) pf_burst();   // prologue: weights are static -> before the wait

    XQ_PDL_ENTRY();                              // xh (had_suh) / ws (previous combine) after this

    int item = (int)blockIdx.x;
    if (item >= n_items) return;                 // idle CTA (launcher clamps G <= items)

    // ---- A staging: block (it, bk) -> buffer b. Thread = (row, 16-B piece); rows < M only.
    auto stage = [&](int it, int bk, int b) {
        const bool mem = it >= per_m;
        const int r = mem ? it - per_m : it;
        const int slab = r / tiles;
        const int s0 = bk * PSK_S;
        const int ns = min(PSK_S, chunk - s0);
        const int row = tid >> 4, pc = tid & 15;
        if (row < M && pc < 2 * ns)
            psk_cp16(&sa[b * 16 + row][pc * 8],
                     (mem ? xh1 : xh0) + (size_t)row * K + ((size_t)slab * chunk + s0) * 16 + pc * 8);
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    };
    stage(item, 0, 0);

    // hoisted per-lane slot constants (identical to exl3_gemm_body_ks)
    int s_i0m[8], s_i1m[8], s_sf[8];
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int r = 2 * (lane & 3) + (s & 1) + 8 * (s >> 1);
            const int c = 8 * h + (lane >> 2);
            const int l = 8 * ((c >> 1) & 3) + ((r & 7) >> 1);
            const int t = 8 * l + (r & 1) + 2 * ((r >> 3) & 1)
                              + 4 * (c >> 3) + 32 * (c & 1);
            const int b1 = (t + 257) * BITS;
            const int i0 = (b1 - 16) >> 5;
            const int i1 = (b1 - 1) >> 5;
            s_i0m[h * 4 + s] = i0 % (BITS * 8);
            s_i1m[h * 4 + s] = i1 % (BITS * 8);
            s_sf[h * 4 + s] = ((i1 + 1) << 5) - b1;
        }
    }
    const int gid = lane >> 2, tig = lane & 3;
    const bool l0 = gid < M, l1 = (gid + 8) < M, hi_rows = M > 8;

    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    bool rm;
    size_t roff = psk_off(item, per_m, tiles, chunk, nb, warp, BLK, rm);
    const uint32_t* __restrict__ ring = (const uint32_t*)((rm ? tr1 : tr0) + roff);
    float acc[2][4];
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[h][r] = 0.0f;

    int blk = 0, buf = 0;
    for (;;) {
        // the block after this one (the next item's first when this is the item's last)
        int n_item = item, n_blk = blk + 1;
        if (n_blk == nblk) { n_item = item + G; n_blk = 0; }
        const bool more = n_item < n_items;
        if (more) stage(n_item, n_blk, buf ^ 1); // buf^1 is free: every warp passed the last barrier
        if (pf_d > 0) pf_burst();                // trellis windows pf_d blocks ahead

        const int ns = min(PSK_S, chunk - blk * PSK_S);
        const __half* ar0 = &sa[buf * 16 + gid][2 * tig];
        const __half* ar1 = ar0 + 8 * PSK_SROW;
        #pragma unroll 1
        for (int j = 0; j < ns; ++j) {
            uint32_t bfrag[2][2];
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint16_t r0 = exl3_dq_from(ring, s_i0m[h*4+0], s_i1m[h*4+0], s_sf[h*4+0]);
                const uint16_t r1 = exl3_dq_from(ring, s_i0m[h*4+1], s_i1m[h*4+1], s_sf[h*4+1]);
                const uint16_t r2 = exl3_dq_from(ring, s_i0m[h*4+2], s_i1m[h*4+2], s_sf[h*4+2]);
                const uint16_t r3 = exl3_dq_from(ring, s_i0m[h*4+3], s_i1m[h*4+3], s_sf[h*4+3]);
                bfrag[h][0] = (uint32_t)r0 | ((uint32_t)r1 << 16);
                bfrag[h][1] = (uint32_t)r2 | ((uint32_t)r3 << 16);
            }
            ring += kstride;

            const uint32_t a0 = l0 ? *(const uint32_t*)(ar0) : 0u;
            const uint32_t a2 = l0 ? *(const uint32_t*)(ar0 + 8) : 0u;
            uint32_t a1 = 0u, a3 = 0u;
            if (hi_rows) {
                a1 = l1 ? *(const uint32_t*)(ar1) : 0u;
                a3 = l1 ? *(const uint32_t*)(ar1 + 8) : 0u;
            }
            ar0 += 16;
            ar1 += 16;
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[h][0]), "+f"(acc[h][1]), "+f"(acc[h][2]), "+f"(acc[h][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                      "r"(bfrag[h][0]), "r"(bfrag[h][1]));
            }
        }

        if (n_blk == 0) {
            // ---- item done: fp32 partial epilogue, ws[slab][row][col] (rows >= M skipped).
            // float2 = the old two scalar stores' exact values at the same addresses.
            const bool mem = item >= per_m;
            const int r = mem ? item - per_m : item;
            const int slab = r / tiles;
            const int tile = r - slab * tiles;
            float* base = (mem ? ws1 : ws0) + (size_t)slab * M * N;
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = tile * 128 + 16 * warp + 8 * h + 2 * tig;
                if (l0) *(float2*)&base[(size_t)gid * N + col] = make_float2(acc[h][0], acc[h][1]);
                if (l1) *(float2*)&base[(size_t)(gid + 8) * N + col] = make_float2(acc[h][2], acc[h][3]);
                #pragma unroll
                for (int q = 0; q < 4; ++q) acc[h][q] = 0.0f;
            }
        }
        if (!more) break;
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();                         // next block visible; this buffer free
        buf ^= 1;
        blk = n_blk;
        if (n_blk == 0) {
            item = n_item;
            roff = psk_off(item, per_m, tiles, chunk, nb, warp, BLK, rm);
            ring = (const uint32_t*)((rm ? tr1 : tr0) + roff);
        }
    }
}

// Launch word: cfg = bits | ks << 8 | pf_d << 24 (bits 3..5, ks < 65536, pf_d < 128).
// pf0/pf1 = the SAME device addresses as the trellis args, passed separately and WITHOUT
// __restrict__: the prefetch asm takes its address as an operand, which makes that pointer
// escape — derived from `trellis` it would cost the ring loads their ld.global.nc (read-only)
// form that exl3_hmma_gemm_ks has. Values are unaffected either way.
#define PSK_CFG_BITS(c) ((c) & 0xFF)
#define PSK_CFG_KS(c)   (((c) >> 8) & 0xFFFF)
#define PSK_CFG_PF(c)   (((c) >> 24) & 0x7F)

// grid (G, 1, 1), block 256.
extern "C" __global__ void __launch_bounds__(256, 4)
exl3_hmma_gemm_ks_p(const uint16_t* __restrict__ trellis, const __half* __restrict__ xh,
                    float* __restrict__ ws, int M, int K, int N, int cfg, const char* pf0) {
    __shared__ __align__(16) __half psk_sa[PSK_SMEM_HALVES];
    const int ks = PSK_CFG_KS(cfg), pf_d = PSK_CFG_PF(cfg);
    switch (PSK_CFG_BITS(cfg)) {
        case 2: exl3_ks_p_body<2>(trellis, trellis, xh, xh, ws, ws, pf0, pf0, M, K, N, ks, 1, pf_d, psk_sa); break;  // P7 B6: K=2.0 (bits=2)
        case 3: exl3_ks_p_body<3>(trellis, trellis, xh, xh, ws, ws, pf0, pf0, M, K, N, ks, 1, pf_d, psk_sa); break;
        case 4: exl3_ks_p_body<4>(trellis, trellis, xh, xh, ws, ws, pf0, pf0, M, K, N, ks, 1, pf_d, psk_sa); break;
        case 5: exl3_ks_p_body<5>(trellis, trellis, xh, xh, ws, ws, pf0, pf0, M, K, N, ks, 1, pf_d, psk_sa); break;
        case 6: exl3_ks_p_body<6>(trellis, trellis, xh, xh, ws, ws, pf0, pf0, M, K, N, ks, 1, pf_d, psk_sa); break;
        default: break;
    }
}
// Pair twin of exl3_hmma_gemm_ks_x2: items [0, per_m) are member 0, [per_m, 2*per_m) member 1.
extern "C" __global__ void __launch_bounds__(256, 4)
exl3_hmma_gemm_ks_p_x2(const uint16_t* __restrict__ tr0, const uint16_t* __restrict__ tr1,
                       const __half* __restrict__ xh0, const __half* __restrict__ xh1,
                       float* __restrict__ ws0, float* __restrict__ ws1,
                       int M, int K, int N, int cfg, const char* pf0, const char* pf1) {
    __shared__ __align__(16) __half psk_sa[PSK_SMEM_HALVES];
    const int ks = PSK_CFG_KS(cfg), pf_d = PSK_CFG_PF(cfg);
    switch (PSK_CFG_BITS(cfg)) {
        case 2: exl3_ks_p_body<2>(tr0, tr1, xh0, xh1, ws0, ws1, pf0, pf1, M, K, N, ks, 2, pf_d, psk_sa); break;  // P7 B6: K=2.0 (bits=2)
        case 3: exl3_ks_p_body<3>(tr0, tr1, xh0, xh1, ws0, ws1, pf0, pf1, M, K, N, ks, 2, pf_d, psk_sa); break;
        case 4: exl3_ks_p_body<4>(tr0, tr1, xh0, xh1, ws0, ws1, pf0, pf1, M, K, N, ks, 2, pf_d, psk_sa); break;
        case 5: exl3_ks_p_body<5>(tr0, tr1, xh0, xh1, ws0, ws1, pf0, pf1, M, K, N, ks, 2, pf_d, psk_sa); break;
        case 6: exl3_ks_p_body<6>(tr0, tr1, xh0, xh1, ws0, ws1, pf0, pf1, M, K, N, ks, 2, pf_d, psk_sa); break;
        default: break;
    }
}

