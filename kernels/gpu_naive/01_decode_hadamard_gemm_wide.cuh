// gpu_naive/01_decode_hadamard_gemm_wide.cuh: BPB-templated trellis decode, Hadamard passes, small-M and WIDE HMMA GEMMs, gathered prefill MoE for the Naive-N0.5-Flash pack.
// Batch-invariant: K-ascending fp32 accumulation and the decode/tmap order are the engine's verbatim (AGENTS section 2.4); do not change reduction orders.
// Section 1 of 3 of gpu_naive.cu: included in order by gpu_naive.cu, one translation unit; text is moved verbatim.
// ---------------------------------------------------------------------------
// Trellis decode (exl3_dq_from, verbatim): the 16-bit window (words i1m LOW, i0m HIGH, funnel
// shift sf) -> mul1 codebook value as fp16 bits, single-rounded fp16 FMA, no I2F.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint16_t nv_dq_from(const uint32_t* __restrict__ ring, int i0m, int i1m, int sf) {
    const unsigned lo = ring[i1m];
    const unsigned hi = ring[i0m];
    const unsigned idx = __funnelshift_r(lo, hi, (unsigned)sf) & 0xFFFFu;
    const unsigned sum = __dp4a(idx * NV_MUL1, 0x01010101u, 0x00006400u);
    const __half h = __ushort_as_half((unsigned short)sum);
    const __half w16 = __hfma(h, __ushort_as_half((unsigned short)0x1EEE), __ushort_as_half((unsigned short)0xC931));
    return (uint16_t)__half_as_ushort(w16);
}

// Window END (unwrapped bit index, >= 16) of temporal index t at BPB bits per 16 weights.
template <int BPB>
__device__ __forceinline__ int nv_win_end(int t) {
    if constexpr (BPB % 16 == 0) {
        return (t + 257) * (BPB / 16);                       // the integer decoder's (t+257)*bits
    } else {
        constexpr int KA = BPB / 16;                          // 3 for 3.5 bpw
        const int j = (t & 15) + 1;
        // mask 0xAAAA: odd ring positions carry the extra bit -> j>>1 of them among the first j
        return (t >> 4) * BPB + KA * j + (j >> 1) + 16 * BPB; // + one ring length keeps START >= 0
    }
}

// (i0m, i1m, sf) of temporal index t: the two ring words holding the window and the funnel shift.
template <int BPB>
__device__ __forceinline__ void nv_win(int t, int& i0m, int& i1m, int& sf) {
    constexpr int WW = BPB / 2;                               // u32 words per tile
    const int b1 = nv_win_end<BPB>(t);
    const int i0 = (b1 - 16) >> 5;
    const int i1 = (b1 - 1) >> 5;
    i0m = i0 % WW;
    i1m = i1 % WW;
    sf = ((i1 + 1) << 5) - b1;
}

// tmap(r, c) -> temporal index (src/exl3.rs tmap, the unit-tested bijection)
__device__ __forceinline__ int nv_tmap(int r, int c) {
    const int l = 8 * ((c >> 1) & 3) + ((r & 7) >> 1);
    return 8 * l + (r & 1) + 2 * ((r >> 3) & 1) + 4 * (c >> 3) + 32 * (c & 1);
}

// ---------------------------------------------------------------------------
// Kernel 1: decode gate. One warp per 16x16 block, lane decodes t in {lane + 32*step}; rc_of_t is
// the inverse tmap LUT. Output: W row-major [kb*16, nb*16] fp16 bits (= ext.reconstruct, rotated
// basis, before suh / svh). Gate: bitwise vs naive::frac::decode_trellis_host.
// ---------------------------------------------------------------------------
template <int BPB>
__device__ __forceinline__ void nv_decode_block(const uint16_t* __restrict__ trellis,
                                                const uint8_t* __restrict__ rc_of_t,
                                                uint16_t* __restrict__ wout, int nb) {
    const int b = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const uint32_t* __restrict__ ring = (const uint32_t*)(trellis + (size_t)b * BPB);
    const int kb = b / nb, nbk = b % nb;
    uint16_t* __restrict__ orow = wout + (size_t)kb * 16 * (nb * 16) + nbk * 16;
    #pragma unroll
    for (int step = 0; step < 8; ++step) {
        const int t = lane + 32 * step;
        int i0m, i1m, sf;
        nv_win<BPB>(t, i0m, i1m, sf);
        const uint16_t v = nv_dq_from(ring, i0m, i1m, sf);
        const int rc = rc_of_t[t];
        orow[(rc >> 4) * (nb * 16) + (rc & 15)] = v;
    }
}
extern "C" __global__ void nv_decode_dump(const uint16_t* __restrict__ trellis,
                                          const uint8_t* __restrict__ rc_of_t,
                                          uint16_t* __restrict__ wout, int nb, int bpb) {
    NV_PDL_ENTRY();
    switch (bpb) {
        case 48: nv_decode_block<48>(trellis, rc_of_t, wout, nb); break;
        case 56: nv_decode_block<56>(trellis, rc_of_t, wout, nb); break;
        case 64: nv_decode_block<64>(trellis, rc_of_t, wout, nb); break;
        case 80: nv_decode_block<80>(trellis, rc_of_t, wout, nb); break;
        case 96: nv_decode_block<96>(trellis, rc_of_t, wout, nb); break;
        default: break;
    }
}

// ---------------------------------------------------------------------------
// Kernels 2/3: the Hadamard passes (exl3_had128 / exl3_had_suh_dev / exl3_had_svh_dev verbatim).
//   xh = H128(x * suh)   per 128-block along K (suh BEFORE the butterfly: the quantizer's order)
//   y  = H128(y_raw) * svh per 128-block along N
// Orthonormal (1/sqrt(128)) — the convention the pack's suh/svh were calibrated under.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void nv_had128(float v[4], int lane) {
    #pragma unroll
    for (int s = 0; s < 2; ++s) {
        const int len = 1 << s;
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            if ((r & len) == 0) {
                const float a = v[r], b = v[r + len];
                v[r] = a + b;
                v[r + len] = a - b;
            }
        }
    }
    #pragma unroll
    for (int s = 2; s < 7; ++s) {
        const unsigned mask = (unsigned)(1u << (s - 2));
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const float t = __shfl_xor_sync(0xFFFFFFFFu, v[r], mask);
            v[r] = (lane & mask) ? (t - v[r]) : (v[r] + t);
        }
    }
    const float hnorm = 0.08838834764831845f;
    #pragma unroll
    for (int r = 0; r < 4; ++r) v[r] *= hnorm;
}

// grid m * (k/128), block 32
extern "C" __global__ void nv_had_suh(const __half* __restrict__ x, const __half* __restrict__ suh,
                                      __half* __restrict__ xh, int k) {
    NV_PDL_ENTRY();
    const int blk = blockIdx.x;
    const int kb = blk % (k >> 7);
    const int m = blk / (k >> 7);
    const int lane = threadIdx.x & 31;
    const size_t base = (size_t)m * k + (size_t)kb * 128 + lane * 4;
    float v[4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) v[r] = nv_h2f(x[base + r]) * nv_h2f(suh[kb * 128 + lane * 4 + r]);
    nv_had128(v, lane);
    #pragma unroll
    for (int r = 0; r < 4; ++r) xh[base + r] = nv_f2h(v[r]);
}

// grid m * (n/128), block 32
extern "C" __global__ void nv_had_svh(const __half* __restrict__ yraw, const __half* __restrict__ svh,
                                      __half* __restrict__ y, int n) {
    NV_PDL_ENTRY();
    const int blk = blockIdx.x;
    const int nb = blk % (n >> 7);
    const int m = blk / (n >> 7);
    const int lane = threadIdx.x & 31;
    const size_t base = (size_t)m * n + (size_t)nb * 128 + lane * 4;
    float v[4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) v[r] = nv_h2f(yraw[base + r]);
    nv_had128(v, lane);
    #pragma unroll
    for (int r = 0; r < 4; ++r) y[base + r] = nv_f2h(v[r] * nv_h2f(svh[nb * 128 + lane * 4 + r]));
}

// ---------------------------------------------------------------------------
// The HMMA GEMM: y_raw[m, n] = sum_k xh[m, k] * W[n, k], W decoded from the trellis per k16 step
// straight into the two m16n8k16 B fragments. CTA = 256 threads = 8 warps, tile m16 x N128; warp w
// owns columns [n0 + 16w, n0 + 16w + 16). fp32 accumulators, K ascending (fixed order).
// B-fragment slot (lane l, n-tile h, slot s): r = 2*(l%4) + (s&1) + 8*(s>>1), c = 8h + (l>>2),
// t = tmap(r, c) — the hoisted per-lane window constants below (exl3_gemm_body).
// ---------------------------------------------------------------------------
template <int BPB>
__device__ __forceinline__ void nv_slot_windows(int lane, int* s_i0m, int* s_i1m, int* s_sf) {
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int r = 2 * (lane & 3) + (s & 1) + 8 * (s >> 1);
            const int c = 8 * h + (lane >> 2);
            nv_win<BPB>(nv_tmap(r, c), s_i0m[h * 4 + s], s_i1m[h * 4 + s], s_sf[h * 4 + s]);
        }
    }
}

// (a) the per-step staged body (exl3_gemm_body / exl3_gemm_body_ks): A tile [16 x 16] re-staged
// every k16 step; works for every M <= 16 and every K. kb range [kb0, kb1); PARTIAL stores fp32
// partials to ws[ks_id][M][N], else fp16 RN to yraw.
template <int BPB, bool PARTIAL>
__device__ __forceinline__ void nv_gemm_body(const uint16_t* __restrict__ trellis,
                                             const __half* __restrict__ xh,
                                             __half* __restrict__ yraw, float* __restrict__ ws,
                                             int M, int K, int N, int cta_tile,
                                             int kb0, int kb1, int ks_id) {
    __shared__ __half sa[16][16];
    constexpr int WW = BPB / 2;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int cta_n = cta_tile * 128;
    const int nb_col = cta_n / 16 + warp;
    const int nb = N >> 4;

    int s_i0m[8], s_i1m[8], s_sf[8];
    nv_slot_windows<BPB>(lane, s_i0m, s_i1m, s_sf);

    float acc[2][4];
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[h][r] = 0.0f;

    const uint32_t* __restrict__ ring = (const uint32_t*)(trellis + ((size_t)kb0 * nb + nb_col) * BPB);
    const size_t ring_kstride = (size_t)nb * WW;

    for (int kb = kb0; kb < kb1; ++kb) {
        {
            const int row = tid >> 4, col = tid & 15;
            const bool live = (row < M) && (kb * 16 + col < K);
            sa[row][col] = live ? xh[(size_t)row * K + kb * 16 + col] : __ushort_as_half(0);
        }
        __syncthreads();
        uint32_t bfrag[2][2];
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint16_t r0 = nv_dq_from(ring, s_i0m[h*4+0], s_i1m[h*4+0], s_sf[h*4+0]);
            const uint16_t r1 = nv_dq_from(ring, s_i0m[h*4+1], s_i1m[h*4+1], s_sf[h*4+1]);
            const uint16_t r2 = nv_dq_from(ring, s_i0m[h*4+2], s_i1m[h*4+2], s_sf[h*4+2]);
            const uint16_t r3 = nv_dq_from(ring, s_i0m[h*4+3], s_i1m[h*4+3], s_sf[h*4+3]);
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
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bfrag[h][0]), "r"(bfrag[h][1]));
        }
        __syncthreads();
    }

    const int gid = lane >> 2, tig = lane & 3;
    const int row0 = gid, row1 = gid + 8;
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int col = cta_n + 16 * warp + 8 * h + 2 * tig;
        if constexpr (PARTIAL) {
            float* base = ws + (size_t)ks_id * M * N;
            if (row0 < M) { base[(size_t)row0 * N + col] = acc[h][0]; base[(size_t)row0 * N + col + 1] = acc[h][1]; }
            if (row1 < M) { base[(size_t)row1 * N + col] = acc[h][2]; base[(size_t)row1 * N + col + 1] = acc[h][3]; }
        } else {
            if (row0 < M) *(__half2*)&yraw[(size_t)row0 * N + col] = __floats2half2_rn(acc[h][0], acc[h][1]);
            if (row1 < M) *(__half2*)&yraw[(size_t)row1 * N + col] = __floats2half2_rn(acc[h][2], acc[h][3]);
        }
    }
}

// ---- F13: the split-K combine folded into the split-K GEMM. `cnt[blockIdx.x]` counts the KS CTAs
// of one n-tile; the LAST one to arrive sums the partials (K ascending, the rounding points of
// nv_ks_combine_had: f16 before the butterfly, f16 after) and writes y / out32, then resets its
// counter (self-cleaning: the host zeroes the array once). Partials are read with __ldcg (L2) after a
// __threadfence on both sides -- the threadFenceReduction pattern. cnt == nullptr: no fusion.
struct NvFuse { const __half* svh; __half* y; float* out32; int* cnt; };
__device__ __forceinline__ void nv_combine_had_warp_cg(const float* __restrict__ ws, const __half* __restrict__ svh,
                                                       __half* __restrict__ y, int M, int N, int KS, long long rb, int lane,
                                                       float* __restrict__ out32) {
    const int nblk = N >> 7;
    const int m = (int)(rb / nblk), blk = (int)(rb - (long long)m * nblk);
    const size_t base = (size_t)m * N + (size_t)blk * 128 + lane * 4;
    float v[4];
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        float acc = 0.f;
        for (int s2 = 0; s2 < KS; ++s2) acc += __ldcg(ws + (size_t)s2 * M * N + base + q);
        v[q] = nv_h2f(nv_f2h(acc));
    }
    nv_had128(v, lane);
    const __half* sr = svh + (size_t)blk * 128 + lane * 4;
    if (out32 != nullptr) {
        #pragma unroll
        for (int q = 0; q < 4; ++q) out32[base + q] = nv_h2f(nv_f2h(v[q] * nv_h2f(sr[q])));
    } else {
        #pragma unroll
        for (int q = 0; q < 4; ++q) y[base + q] = nv_f2h(v[q] * nv_h2f(sr[q]));
    }
}

// (b) the staged-A body (exl3_gemm_body_a1 + the F7 shuffle decode): the M live rows land in
// dynamic smem once (row stride K+8 halves), the K loop runs with no barriers; one coalesced ring
// load per lane per step, decode words by shuffle. Same values as (a) bit for bit. Dynamic smem =
// M*(K+8)*2 bytes (nv_a1_fits).
__host__ __device__ __forceinline__ bool nv_a1_fits(int M, int K) {
    return M >= 1 && M <= 16 && (K & 7) == 0 && (long long)M * (K + 8) * 2 <= 48 * 1024;
}
// Fused input transform (S6 F1): when `suh` is non-null the staged A is xh = H128(x * suh) computed
// in smem (x f16 [M][K]; or, with `x2` non-null, x = silu(x) * x2 first: the MoE down input), the
// SAME ops and rounding points as nv_had_suh / nv_silu_mul (f32 products, the butterfly, one f16
// rounding), so y is bitwise the unfused chain's. `suh == nullptr`: `xh` is already transformed.
template <int BPB, bool PARTIAL>
__device__ __forceinline__ void nv_gemm_body_a1(const uint16_t* __restrict__ trellis,
                                                const __half* __restrict__ xh,
                                                __half* __restrict__ yraw, float* __restrict__ ws,
                                                int M, int K, int N, int cta_tile, int kb0, int kb1, int ks_id,
                                                const __half* __restrict__ suh = nullptr,
                                                const __half* __restrict__ x2 = nullptr,
                                                NvFuse fz = NvFuse{nullptr, nullptr, nullptr, nullptr}) {
    extern __shared__ __half a1_sa[];
    constexpr int WW = BPB / 2;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int cta_n = cta_tile * 128;
    const int nb_col = cta_n / 16 + warp;
    const int KB = K >> 4;
    const int nb = N >> 4;
    // T0-c: stage ONLY this CTA's k-window [kb0, kb1). A split-K CTA walks only that window, so
    // staging all of K read (and held) M*(K+8)*2 halves of A it never touches: ks times the bytes,
    // and the reason a staged split-K launch was gated to M*(K+8)*2 <= 48KB. Chunking it is
    // bitwise-identical (same words, same order, same rounding points) and ks times less smem AND
    // ks times less global traffic on the staging read. Host mirror: naive::a1_smem(m, k, ks);
    // for ks == 1 the window is [0, K) so SK is unchanged from the pre-T0-c K + 8.
    const int SK = ((kb1 - kb0) << 4) + 8;   // window row stride, halves (was K + 8)
    if (suh == nullptr) {
        const int v0 = kb0 << 1;             // window origin in 8-half (uint4) units (kb0 * 16 / 8)
        const int vn = (kb1 - kb0) << 1;     // window width in those units
        for (int i = tid; i < M * vn; i += blockDim.x) {
            const int r = i / vn, c = i - r * vn;
            *(uint4*)&a1_sa[(size_t)r * SK + c * 8] = *(const uint4*)&xh[(size_t)r * K + (v0 + c) * 8];
        }
    } else {
        // one warp per (row, 128-block): lane owns 4 consecutive elements (nv_had_suh's layout)
        // T0-c: the window is 128-aligned (the host gate requires (K/ks) % 128 == 0), so every
        // butterfly group this CTA needs is wholly inside its own window: restricting the loop to
        // [g0, g0+nblk) leaves each nv_had128 on the same 128 elements as before.
        const int nblk = (kb1 - kb0) >> 3;
        const int g0 = kb0 >> 3;
        for (int b = warp; b < M * nblk; b += blockDim.x >> 5) {
            const int r = b / nblk, blk = b - r * nblk;
            const size_t src = (size_t)r * K + (size_t)(blk + g0) * 128 + lane * 4;
            const __half* sr = suh + (size_t)(blk + g0) * 128 + lane * 4;
            float v[4];
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                float xv = nv_h2f(xh[src + q]);
                if (x2 != nullptr) { xv = xv / (1.f + __expf(-xv)) * nv_h2f(x2[src + q]); xv = nv_h2f(nv_f2h(xv)); }
                v[q] = xv * nv_h2f(sr[q]);
            }
            nv_had128(v, lane);
            #pragma unroll
            for (int q = 0; q < 4; ++q) a1_sa[(size_t)r * SK + (size_t)blk * 128 + lane * 4 + q] = nv_f2h(v[q]);
        }
    }
    __syncthreads();

    int s_i0m[8], s_i1m[8], s_sf[8];
    nv_slot_windows<BPB>(lane, s_i0m, s_i1m, s_sf);
    float acc[2][4];
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[h][r] = 0.0f;

    const uint32_t* __restrict__ ring = (const uint32_t*)(trellis + ((size_t)kb0 * nb + nb_col) * BPB);
    const size_t ring_kstride = (size_t)nb * WW;
    const int gid = lane >> 2, tig = lane & 3;
    const bool hi_rows = M > 8;
    const bool l0 = gid < M, l1 = (gid + 8) < M;
    const __half* ar0 = a1_sa + (size_t)gid * SK + 2 * tig;
    const __half* ar1 = a1_sa + (size_t)(gid + 8) * SK + 2 * tig;
    (void)KB;

    constexpr int PF_D = 8;
    const bool pf_lane = lane * 32 < WW * 4;                 // 32-B chunks of the ring window
    const char* pf_ptr = (const char*)(ring + (size_t)PF_D * ring_kstride) + lane * 32;
    const size_t pf_step = ring_kstride * 4;
    if (pf_lane) {
        const char* p0 = (const char*)ring + lane * 32;
        #pragma unroll
        for (int d = 0; d < PF_D; ++d)
            if (kb0 + d < kb1) asm volatile("prefetch.global.L2 [%0];" :: "l"(p0 + (size_t)d * pf_step));
    }
    // S6 F7: software pipeline — the ring words of the NEXT group of NV_PIPE k-steps are loaded (one
    // coalesced u32 per lane per step) while the current group decodes and multiplies; the decode and
    // the mma sequence per step are unchanged (same words, same order => same bits).
    constexpr int NV_PIPE = 4;
    unsigned wc0[NV_PIPE], wc1[NV_PIPE], wn0[NV_PIPE], wn1[NV_PIPE];
    #pragma unroll
    for (int i = 0; i < NV_PIPE; ++i) {
        const bool live = kb0 + i < kb1;
        const uint32_t* rp = ring + (size_t)i * ring_kstride;
        wc0[i] = (live && lane < WW) ? __ldg(rp + lane) : 0u;
        wc1[i] = (live && WW > 32 && lane + 32 < WW) ? __ldg(rp + lane + 32) : 0u;
    }
    for (int kbg = kb0; kbg < kb1; kbg += NV_PIPE) {
        #pragma unroll
        for (int i = 0; i < NV_PIPE; ++i) {
            const int kbn = kbg + NV_PIPE + i;
            const bool live = kbn < kb1;
            const uint32_t* rp = ring + (size_t)(NV_PIPE + i) * ring_kstride;
            wn0[i] = (live && lane < WW) ? __ldg(rp + lane) : 0u;
            wn1[i] = (live && WW > 32 && lane + 32 < WW) ? __ldg(rp + lane + 32) : 0u;
        }
        if (pf_lane) {
            #pragma unroll
            for (int i = 0; i < NV_PIPE; ++i) {
                if (kbg + i + PF_D < kb1) asm volatile("prefetch.global.L2 [%0];" :: "l"(pf_ptr));
                pf_ptr += pf_step;
            }
        }
        #pragma unroll
        for (int i = 0; i < NV_PIPE; ++i) {
            const int kb = kbg + i;
            if (kb >= kb1) break;
            const unsigned w0 = wc0[i], w1 = wc1[i];
            uint32_t bfrag[2][2];
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                uint16_t rv[4];
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int j = h * 4 + q;
                    unsigned lo = __shfl_sync(0xFFFFFFFFu, w0, s_i1m[j] & 31);
                    unsigned hi = __shfl_sync(0xFFFFFFFFu, w0, s_i0m[j] & 31);
                    if (WW > 32) {
                        const unsigned lo2 = __shfl_sync(0xFFFFFFFFu, w1, s_i1m[j] & 31);
                        const unsigned hi2 = __shfl_sync(0xFFFFFFFFu, w1, s_i0m[j] & 31);
                        lo = s_i1m[j] >= 32 ? lo2 : lo;
                        hi = s_i0m[j] >= 32 ? hi2 : hi;
                    }
                    const unsigned idx = __funnelshift_r(lo, hi, (unsigned)s_sf[j]) & 0xFFFFu;
                    const unsigned sum = __dp4a(idx * NV_MUL1, 0x01010101u, 0x00006400u);
                    rv[q] = (uint16_t)__half_as_ushort(__hfma(__ushort_as_half((unsigned short)sum),
                                __ushort_as_half((unsigned short)0x1EEE), __ushort_as_half((unsigned short)0xC931)));
                }
                bfrag[h][0] = (uint32_t)rv[0] | ((uint32_t)rv[1] << 16);
                bfrag[h][1] = (uint32_t)rv[2] | ((uint32_t)rv[3] << 16);
            }
            // T0-c: A is window-local now, so the tile's column is relative to the window origin.
            const int col = (kb - kb0) << 4;
            const uint32_t a0 = l0 ? *(const uint32_t*)(ar0 + col) : 0u;
            const uint32_t a2 = l0 ? *(const uint32_t*)(ar0 + col + 8) : 0u;
            uint32_t a1 = 0u, a3 = 0u;
            if (hi_rows) {
                a1 = l1 ? *(const uint32_t*)(ar1 + col) : 0u;
                a3 = l1 ? *(const uint32_t*)(ar1 + col + 8) : 0u;
            }
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[h][0]), "+f"(acc[h][1]), "+f"(acc[h][2]), "+f"(acc[h][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bfrag[h][0]), "r"(bfrag[h][1]));
            }
        }
        ring += (size_t)NV_PIPE * ring_kstride;
        #pragma unroll
        for (int i = 0; i < NV_PIPE; ++i) { wc0[i] = wn0[i]; wc1[i] = wn1[i]; }
    }
    const int row0 = gid, row1 = gid + 8;
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int col = cta_n + 16 * warp + 8 * h + 2 * tig;
        if constexpr (PARTIAL) {
            float* base = ws + (size_t)ks_id * M * N;
            if (row0 < M) { base[(size_t)row0 * N + col] = acc[h][0]; base[(size_t)row0 * N + col + 1] = acc[h][1]; }
            if (row1 < M) { base[(size_t)row1 * N + col] = acc[h][2]; base[(size_t)row1 * N + col + 1] = acc[h][3]; }
        } else {
            if (row0 < M) *(__half2*)&yraw[(size_t)row0 * N + col] = __floats2half2_rn(acc[h][0], acc[h][1]);
            if (row1 < M) *(__half2*)&yraw[(size_t)row1 * N + col] = __floats2half2_rn(acc[h][2], acc[h][3]);
        }
    }
    if constexpr (PARTIAL) {
        if (fz.cnt != nullptr) {
            __shared__ int s_last;
            __threadfence();
            __syncthreads();
            if (tid == 0) s_last = (atomicAdd(fz.cnt + blockIdx.x, 1) == (int)gridDim.y - 1);
            __syncthreads();
            if (s_last) {
                __threadfence();
                const int nblk = N >> 7;
                for (int r = warp; r < M; r += 8)
                    nv_combine_had_warp_cg(ws, fz.svh, fz.y, M, N, (int)gridDim.y, (long long)r * nblk + cta_tile, lane, fz.out32);
                if (tid == 0) fz.cnt[blockIdx.x] = 0;
            }
        }
    }
}

#define NV_BPB_SWITCH(CALL)                 \
    switch (bpb) {                          \
        case 48: { constexpr int BPB = 48; CALL; break; } \
        case 56: { constexpr int BPB = 56; CALL; break; } \
        case 64: { constexpr int BPB = 64; CALL; break; } \
        case 80: { constexpr int BPB = 80; CALL; break; } \
        case 96: { constexpr int BPB = 96; CALL; break; } \
        default: break;                     \
    }

// Dense quad, M <= 16: grid (N/128), block 256, dyn smem = M*(K+8)*2 when nv_a1_fits else 0.
extern "C" __global__ void nv_gemm(const uint16_t* __restrict__ trellis, const __half* __restrict__ xh,
                                   __half* __restrict__ yraw, int M, int K, int N, int bpb) {
    NV_PDL_ENTRY();
    const int KB = K >> 4;
    if (nv_a1_fits(M, K)) {
        NV_BPB_SWITCH((nv_gemm_body_a1<BPB, false>(trellis, xh, yraw, nullptr, M, K, N, (int)blockIdx.x, 0, KB, 0)));
    } else {
        NV_BPB_SWITCH((nv_gemm_body<BPB, false>(trellis, xh, yraw, nullptr, M, K, N, (int)blockIdx.x, 0, KB, 0)));
    }
}

// Split-K: grid (N/128, ks), KB % ks == 0 (host asserts); fp32 partials ws[ks][M][N].
extern "C" __global__ void nv_gemm_ks(const uint16_t* __restrict__ trellis, const __half* __restrict__ xh,
                                      float* __restrict__ ws, int M, int K, int N, int bpb) {
    NV_PDL_ENTRY();
    const int ks = gridDim.y;
    const int KB = K >> 4;
    const int chunk = KB / ks;
    const int kb0 = blockIdx.y * chunk;
    const int kb1 = kb0 + chunk;
    NV_BPB_SWITCH((nv_gemm_body<BPB, true>(trellis, xh, nullptr, ws, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y)));
}

// Fixed-order ascending combine of the split-K partials (exl3_ks_combine verbatim).
extern "C" __global__ void nv_ks_combine(const float* __restrict__ ws, __half* __restrict__ yraw, int M, int N, int KS) {
    NV_PDL_ENTRY();
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    const int m = idx / N;
    const int n = idx - (size_t)m * N;
    float acc = 0.0f;
    for (int s = 0; s < KS; ++s) acc += ws[((size_t)s * M + m) * N + n];
    yraw[(size_t)m * N + n] = __float2half_rn(acc);
}

// Grouped expert GEMM with per-expert xh ([E][M][K]) (xq_gemm_grouped_xh): grid padded to the
// top-k cap, blocks of absent experts exit on the device-routed count *esel. offs[e] = u16 word
// offset of expert e's trellis in `base`. Dyn smem = M*(K+8)*2 when nv_a1_fits else 0.
extern "C" __global__ void nv_gemm_grouped_xh(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs,
                                              const __half* __restrict__ xh, __half* __restrict__ y,
                                              int M, int K, int N, int bpb, const int* __restrict__ esel) {
    NV_PDL_ENTRY();
    const int tiles = N / 128;
    const int expert = blockIdx.x / tiles;
    const int tile = blockIdx.x - expert * tiles;
    if (expert >= *esel) return;
    const uint16_t* tr = base + (size_t)offs[expert];
    const __half* xe = xh + (long long)expert * M * K;
    __half* ye = y + (long long)expert * M * N;
    const int KB = K >> 4;
    if (nv_a1_fits(M, K)) {
        NV_BPB_SWITCH((nv_gemm_body_a1<BPB, false>(tr, xe, ye, nullptr, M, K, N, tile, 0, KB, 0)));
    } else {
        NV_BPB_SWITCH((nv_gemm_body<BPB, false>(tr, xe, ye, nullptr, M, K, N, tile, 0, KB, 0)));
    }
}

// S6: split-K with the STAGED body (A in dyn smem once, L2 prefetch): grid (N/128, KS) -> fp32
// partials ws[ks][M][N] (K-ascending chunks), then nv_ks_combine. Dyn smem = M*(K+8)*2 (nv_a1_fits).
// suh: the fused input transform (nullptr = xh already transformed); x2: non-null = x is silu(xh) * x2.
extern "C" __global__ void nv_gemm_a1ks(const uint16_t* __restrict__ trellis, const __half* __restrict__ xh,
                                        float* __restrict__ ws, int M, int K, int N, int bpb,
                                        const __half* __restrict__ suh, const __half* __restrict__ x2, NvFuse fz) {
    NV_PDL_ENTRY();
    const int ks = gridDim.y, ks_id = blockIdx.y;
    const int KB = K >> 4;
    const int chunk = KB / ks;
    const int kb0 = ks_id * chunk, kb1 = kb0 + chunk;
    NV_BPB_SWITCH((nv_gemm_body_a1<BPB, true>(trellis, xh, nullptr, ws, M, K, N, (int)blockIdx.x, kb0, kb1, ks_id, suh, x2, fz)));
}
// grouped experts, staged body, split-K: grid (E_cap * N/128, KS) -> ws[e][ks][M][N] -> nv_ks_combine_grouped.
// in_mode: 0 = xh is the SHARED [M][K] input (gate / up: x = hbuf), 1 = per-expert [E][M][K] input,
// 2 = per-expert PAIR (x = silu(xh[e]) * x2[e]: the down input from the gate / up outputs).
// suh_all [E_local][K] + idxmap (local expert index per slot) = the fused per-expert input transform.
// (12-argument launch limit: mb = M | bpb << 8, kn = K | N << 16)
extern "C" __global__ void nv_gemm_grouped_a1ks(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs,
                                                const __half* __restrict__ xh, float* __restrict__ ws,
                                                int mb, int kn, const int* __restrict__ esel,
                                                const __half* __restrict__ suh_all, const int* __restrict__ idxmap,
                                                const __half* __restrict__ x2, int in_mode, NvFuse fz) {
    NV_PDL_ENTRY();
    const int M = mb & 0xFF, bpb = mb >> 8, K = kn & 0xFFFF, N = (int)((unsigned)kn >> 16);
    const int tiles = N / 128;
    const int expert = blockIdx.x / tiles;
    const int tile = blockIdx.x - expert * tiles;
    if (expert >= *esel) return;
    const int ks = gridDim.y, ks_id = blockIdx.y;
    const int KB = K >> 4;
    const int chunk = KB / ks;
    const int kb0 = ks_id * chunk, kb1 = kb0 + chunk;
    const uint16_t* tr = base + (size_t)offs[expert];
    const __half* xe = (in_mode == 0) ? xh : xh + (long long)expert * M * K;
    const __half* x2e = (in_mode == 2) ? x2 + (long long)expert * M * K : nullptr;
    const __half* suh = suh_all ? suh_all + (long long)idxmap[expert] * K : nullptr;   // null = xh already transformed
    float* we = ws + (long long)expert * ks * M * N;
    if (fz.cnt != nullptr) {   // F13: this expert's svh row and y slab (the grouped combine's addressing)
        fz.svh = fz.svh + (long long)idxmap[expert] * N;
        fz.y = fz.y + (long long)expert * M * N;
    }
    NV_BPB_SWITCH((nv_gemm_body_a1<BPB, true>(tr, xe, nullptr, we, M, K, N, tile, kb0, kb1, ks_id, suh, x2e, fz)));
}

// ---- F2: split-K combine fused with the output transform: y[m][:] = H128(sum_ks ws) * svh, the sum
// in fixed K-ascending order, rounded to f16 BEFORE the butterfly (nv_ks_combine's rounding point),
// then one f16 rounding after (nv_had_svh's). One warp per (row, 128-block); block 256 = 8 warps.
__device__ __forceinline__ void nv_combine_had_warp(const float* __restrict__ ws, const __half* __restrict__ svh,
                                                    __half* __restrict__ y, int M, int N, int KS, long long rb, int lane,
                                                    float* __restrict__ out32 = nullptr) {
    const int nblk = N >> 7;
    const int m = (int)(rb / nblk), blk = (int)(rb - (long long)m * nblk);
    const size_t base = (size_t)m * N + (size_t)blk * 128 + lane * 4;
    float v[4];
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        float acc = 0.f;
        for (int s2 = 0; s2 < KS; ++s2) acc += ws[(size_t)s2 * M * N + base + q];
        v[q] = nv_h2f(nv_f2h(acc));
    }
    nv_had128(v, lane);
    const __half* sr = svh + (size_t)blk * 128 + lane * 4;
    if (out32 != nullptr) {
        // the TP partial: the f16-rounded value as fp32 (what nv_cvt_f16_f32 of y would give)
        #pragma unroll
        for (int q = 0; q < 4; ++q) out32[base + q] = nv_h2f(nv_f2h(v[q] * nv_h2f(sr[q])));
    } else {
        #pragma unroll
        for (int q = 0; q < 4; ++q) y[base + q] = nv_f2h(v[q] * nv_h2f(sr[q]));
    }
}
// out32 non-null (TP): the result lands as fp32 in out32 (the all-reduce partial) instead of f16 y.
extern "C" __global__ void nv_ks_combine_had(const float* __restrict__ ws, const __half* __restrict__ svh,
                                             __half* __restrict__ y, int M, int N, int KS, float* __restrict__ out32) {
    NV_PDL_ENTRY();
    const long long rb = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (rb >= (long long)M * (N >> 7)) return;
    nv_combine_had_warp(ws, svh, y, M, N, KS, rb, threadIdx.x & 31, out32);
}
// grouped: ws[e][ks][M][N], svh_all [E_local][N] via idxmap, y [E_cap][M][N]. grid over E_cap*M*N/128 warps.
extern "C" __global__ void nv_ks_combine_grouped_had(const float* __restrict__ ws, const __half* __restrict__ svh_all,
                                                     const int* __restrict__ idxmap, __half* __restrict__ y,
                                                     int M, int N, int KS, const int* __restrict__ esel) {
    NV_PDL_ENTRY();
    const long long rb = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const long long per = (long long)M * (N >> 7);
    const int e = (int)(rb / per);
    if (e >= *esel) return;
    nv_combine_had_warp(ws + (size_t)e * KS * M * N, svh_all + (size_t)idxmap[e] * N, y + (size_t)e * M * N, M, N, KS, rb - (long long)e * per, threadIdx.x & 31);
}

// =============================================================================
// S6b: the WIDE GEMM (prefill, M > 16) — exl3_gemm_body_wide re-parameterized for the half-K decoder:
// each warp decodes its 16x16 trellis tile ONCE per k-step (the a1 body's shuffle decode) and multiplies
// MT m-tiles of 16 rows from the staged A [MT*16][KS*16 + 8] (KS = 8 k-steps per stage). Rows >= M and
// columns >= K are zero. y_raw f16 [M][N], RN. grid (ceil(M / (MT*16)), N/128), block 256.
// K-ascending fp32 accumulation per output element: per-row results are bitwise independent of M and of
// the m-tile a row lands in (batch invariance). Same decode words / order as the small-M bodies.
// =============================================================================
#define NV_WIDE_KS 8
template <int BPB, int MT>
__device__ __forceinline__ void nv_gemm_body_wide(const uint16_t* __restrict__ trellis, const __half* __restrict__ xh,
                                                  __half* __restrict__ yraw, int M, int K, int N, int cta_tile, int m_sup,
                                                  __half* __restrict__ sa_raw) {
    typedef __half SaRow[NV_WIDE_KS * 16 + 8];
    SaRow* sa = (SaRow*)sa_raw;
    constexpr int WW = BPB / 2;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int cta_n = cta_tile * 128;
    const int nb_col = cta_n / 16 + warp;
    const int KB = K >> 4;
    const int nb = N >> 4;
    const int row_base = m_sup * (MT * 16);
    int s_i0m[8], s_i1m[8], s_sf[8];
    nv_slot_windows<BPB>(lane, s_i0m, s_i1m, s_sf);
    float acc[MT][2][4];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc[mt][h][r] = 0.0f;
    const uint32_t* __restrict__ ring = (const uint32_t*)(trellis + ((size_t)nb_col) * BPB);
    const size_t ring_kstride = (size_t)nb * WW;
    const int gid = lane >> 2, tig = lane & 3;
    for (int kb0 = 0; kb0 < KB; kb0 += NV_WIDE_KS) {
        {
            constexpr int C8 = NV_WIDE_KS * 2;
            for (int i = tid; i < MT * 16 * C8; i += blockDim.x) {
                const int row = i / C8, c8 = i - row * C8;
                const int gr = row_base + row;
                const int col = kb0 * 16 + c8 * 8;
                uint4 v = make_uint4(0u, 0u, 0u, 0u);
                if (gr < M && col < K) v = *(const uint4*)&xh[(size_t)gr * K + col];
                *(uint4*)&sa[row][c8 * 8] = v;
            }
        }
        __syncthreads();
        for (int s = 0; s < NV_WIDE_KS; ++s) {
            const int kb = kb0 + s;
            if (kb >= KB) break;
            const uint32_t* rp = ring + (size_t)kb * ring_kstride;
            const unsigned w0 = (lane < WW) ? __ldg(rp + lane) : 0u;
            const unsigned w1 = (WW > 32 && lane + 32 < WW) ? __ldg(rp + lane + 32) : 0u;
            uint32_t bfrag[2][2];
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                uint16_t rv[4];
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int j = h * 4 + q;
                    unsigned lo = __shfl_sync(0xFFFFFFFFu, w0, s_i1m[j] & 31);
                    unsigned hi = __shfl_sync(0xFFFFFFFFu, w0, s_i0m[j] & 31);
                    if (WW > 32) {
                        const unsigned lo2 = __shfl_sync(0xFFFFFFFFu, w1, s_i1m[j] & 31);
                        const unsigned hi2 = __shfl_sync(0xFFFFFFFFu, w1, s_i0m[j] & 31);
                        lo = s_i1m[j] >= 32 ? lo2 : lo;
                        hi = s_i0m[j] >= 32 ? hi2 : hi;
                    }
                    const unsigned idx = __funnelshift_r(lo, hi, (unsigned)s_sf[j]) & 0xFFFFu;
                    const unsigned sum = __dp4a(idx * NV_MUL1, 0x01010101u, 0x00006400u);
                    rv[q] = (uint16_t)__half_as_ushort(__hfma(__ushort_as_half((unsigned short)sum),
                                __ushort_as_half((unsigned short)0x1EEE), __ushort_as_half((unsigned short)0xC931)));
                }
                bfrag[h][0] = (uint32_t)rv[0] | ((uint32_t)rv[1] << 16);
                bfrag[h][1] = (uint32_t)rv[2] | ((uint32_t)rv[3] << 16);
            }
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const SaRow* sm = sa + mt * 16;
                const int cs = s * 16;
                const uint32_t a0 = *(const uint32_t*)&sm[gid][cs + 2 * tig];
                const uint32_t a1 = *(const uint32_t*)&sm[gid + 8][cs + 2 * tig];
                const uint32_t a2 = *(const uint32_t*)&sm[gid][cs + 2 * tig + 8];
                const uint32_t a3 = *(const uint32_t*)&sm[gid + 8][cs + 2 * tig + 8];
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(acc[mt][h][0]), "+f"(acc[mt][h][1]), "+f"(acc[mt][h][2]), "+f"(acc[mt][h][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bfrag[h][0]), "r"(bfrag[h][1]));
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int row0 = row_base + mt * 16 + gid;
        const int row1 = row0 + 8;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = cta_n + 16 * warp + 8 * h + 2 * tig;
            if (row0 < M) *(__half2*)&yraw[(size_t)row0 * N + col] = __floats2half2_rn(acc[mt][h][0], acc[mt][h][1]);
            if (row1 < M) *(__half2*)&yraw[(size_t)row1 * N + col] = __floats2half2_rn(acc[mt][h][2], acc[mt][h][3]);
        }
    }
}
#define NV_WIDE_SMEM(MT) ((MT) * 16 * (NV_WIDE_KS * 16 + 8) * 2)
// mt in {1, 2, 4, 8}; dyn smem = NV_WIDE_SMEM(mt). grid (ceil(M / (mt*16)), N/128).
extern "C" __global__ void nv_gemm_wide(const uint16_t* __restrict__ trellis, const __half* __restrict__ xh,
                                        __half* __restrict__ yraw, int M, int K, int N, int bpb, int mt) {
    NV_PDL_ENTRY();
    extern __shared__ __half wide_sa[];
    const int cta_tile = (int)blockIdx.y, m_sup = (int)blockIdx.x;
#define NV_WIDE_CASE(B) case B: switch (mt) { \
        case 1: nv_gemm_body_wide<B, 1>(trellis, xh, yraw, M, K, N, cta_tile, m_sup, wide_sa); break; \
        case 2: nv_gemm_body_wide<B, 2>(trellis, xh, yraw, M, K, N, cta_tile, m_sup, wide_sa); break; \
        case 4: nv_gemm_body_wide<B, 4>(trellis, xh, yraw, M, K, N, cta_tile, m_sup, wide_sa); break; \
        case 8: nv_gemm_body_wide<B, 8>(trellis, xh, yraw, M, K, N, cta_tile, m_sup, wide_sa); break; \
        default: break; } break;
    switch (bpb) { NV_WIDE_CASE(48) NV_WIDE_CASE(56) NV_WIDE_CASE(64) NV_WIDE_CASE(80) NV_WIDE_CASE(96) default: break; }
#undef NV_WIDE_CASE
}
// grouped (prefill MoE): experts' rows are gathered contiguously: rows [offs_row[z], offs_row[z] + cnt[z]) of xh / y
// belong to compact expert slot z (trellis at base + offs[z]). grid (msup_max, N/128, esel_cap): a block exits
// when its slot is empty or its m-tile is past the slot's rows.
extern "C" __global__ void nv_gemm_wide_grouped(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs,
                                                const __half* __restrict__ xh, __half* __restrict__ y,
                                                const int* __restrict__ offs_row, const int* __restrict__ cnt,
                                                const int* __restrict__ esel, int K, int N, int bpb, int mt) {
    NV_PDL_ENTRY();
    extern __shared__ __half wide_sa[];
    const int z = blockIdx.z;
    if (z >= *esel) return;
    const int ce = cnt[z];
    if (ce == 0) return;
    const int m_sup = (int)blockIdx.x;
    if (m_sup * mt * 16 >= ce) return;
    const int row0 = offs_row[z];
    const uint16_t* tr = base + (size_t)offs[z];
    const __half* xe = xh + (size_t)row0 * K;
    __half* ye = y + (size_t)row0 * N;
    const int cta_tile = (int)blockIdx.y;
#define NV_WG_CASE(B) case B: switch (mt) { \
        case 1: nv_gemm_body_wide<B, 1>(tr, xe, ye, ce, K, N, cta_tile, m_sup, wide_sa); break; \
        case 2: nv_gemm_body_wide<B, 2>(tr, xe, ye, ce, K, N, cta_tile, m_sup, wide_sa); break; \
        case 4: nv_gemm_body_wide<B, 4>(tr, xe, ye, ce, K, N, cta_tile, m_sup, wide_sa); break; \
        case 8: nv_gemm_body_wide<B, 8>(tr, xe, ye, ce, K, N, cta_tile, m_sup, wide_sa); break; \
        default: break; } break;
    switch (bpb) { NV_WG_CASE(48) NV_WG_CASE(56) NV_WG_CASE(64) NV_WG_CASE(80) NV_WG_CASE(96) default: break; }
#undef NV_WG_CASE
}

// =============================================================================
// S6b: prefill MoE on gathered rows. R = M * k (routed (token, slot) pairs); rows of one LOCAL expert are
// contiguous: [offs_row[le], offs_row[le] + cnt[le]). rowmap[m][j] = the gathered row of pair (m, j), or -1
// when the expert belongs to another rank. The combine sums the k slots in j order per token: deterministic.
// =============================================================================
// cnt[le] (zeroed by the host memset... no: zeroed here by block 0 would race; the caller passes a zeroed buffer
// from the workspace via nv_zero_i32) += 1 per routed pair of an owned expert. grid ceil(M*k/256).
extern "C" __global__ void nv_zero_i32(int* __restrict__ p, int n) {
    NV_PDL_ENTRY();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0;
}
extern "C" __global__ void nv_moe_hist(const int* __restrict__ ids, const int* __restrict__ local_of, int* __restrict__ cnt, int MK) {
    NV_PDL_ENTRY();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= MK) return;
    const int le = local_of[ids[i]];
    if (le >= 0) atomicAdd(&cnt[le], 1);
}
// exclusive prefix over the local experts -> offs_row; cursor[le] = offs_row[le] (the gather's running index);
// esel[0] = E_local. One block.
extern "C" __global__ void __launch_bounds__(1024)
nv_moe_offsets(const int* __restrict__ cnt, int* __restrict__ offs_row, int* __restrict__ cursor, int* __restrict__ esel, int EL) {
    NV_PDL_ENTRY();
    __shared__ int sc[1024];
    const int tid = threadIdx.x;
    int v = (tid < EL) ? cnt[tid] : 0;
    sc[tid] = v; __syncthreads();
    for (int o = 1; o < 1024; o <<= 1) { const int x = (tid >= o) ? sc[tid - o] : 0; __syncthreads(); sc[tid] += x; __syncthreads(); }
    if (tid < EL) { const int off = sc[tid] - v; offs_row[tid] = off; cursor[tid] = off; }
    if (tid == 0) esel[0] = EL;
}
// gather: pair (m, j) of an owned expert takes row r = cursor[le]++ (atomic order: every row's results are
// batch-invariant and the combine reads rows through rowmap in j order, so the order is immaterial);
// xh_g[r] = H128(x[m] * suh_g[le]) and xh_u[r] = H128(x[m] * suh_u[le]) (gate and up share the row);
// rowmap[m][j] = r; row_e[r] = le. One block (128 threads) per pair; nv_had_suh's lane layout per 128-block.
extern "C" __global__ void __launch_bounds__(128)
nv_moe_gather_had(const __half* __restrict__ x, const __half* __restrict__ suh_g, const __half* __restrict__ suh_u,
                  const int* __restrict__ ids, const int* __restrict__ local_of, int* __restrict__ cursor,
                  __half* __restrict__ xh_g, __half* __restrict__ xh_u, int* __restrict__ rowmap, int* __restrict__ row_e,
                  int k, int K) {
    NV_PDL_ENTRY();
    __shared__ int s_r;
    const int pair = blockIdx.x;
    const int m = pair / k;
    const int le = local_of[ids[pair]];
    if (threadIdx.x == 0) s_r = (le >= 0) ? atomicAdd(&cursor[le], 1) : -1;
    __syncthreads();
    const int r = s_r;
    if (threadIdx.x == 0) { rowmap[pair] = r; if (r >= 0) row_e[r] = le; }
    if (r < 0) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nblk = K >> 7;
    const __half* xr = x + (size_t)m * K;
    const __half* sg = suh_g + (size_t)le * K;
    const __half* su = suh_u + (size_t)le * K;
    __half* og = xh_g + (size_t)r * K;
    __half* ou = xh_u + (size_t)r * K;
    for (int blk = warp; blk < nblk; blk += 4) {
        float xv[4], v[4];
        #pragma unroll
        for (int q = 0; q < 4; ++q) xv[q] = nv_h2f(xr[blk * 128 + lane * 4 + q]);
        #pragma unroll
        for (int q = 0; q < 4; ++q) v[q] = xv[q] * nv_h2f(sg[blk * 128 + lane * 4 + q]);
        nv_had128(v, lane);
        #pragma unroll
        for (int q = 0; q < 4; ++q) og[blk * 128 + lane * 4 + q] = nv_f2h(v[q]);
        #pragma unroll
        for (int q = 0; q < 4; ++q) v[q] = xv[q] * nv_h2f(su[blk * 128 + lane * 4 + q]);
        nv_had128(v, lane);
        #pragma unroll
        for (int q = 0; q < 4; ++q) ou[blk * 128 + lane * 4 + q] = nv_f2h(v[q]);
    }
}
// per-row output transform with the row's expert: y[r] = H128(yraw[r]) * svh[row_e[r]]. grid R * (N/128) warps (block 256).
extern "C" __global__ void nv_had_svh_rows(const __half* __restrict__ yraw, const __half* __restrict__ svh_all,
                                           const int* __restrict__ row_e, __half* __restrict__ y, int R, int N) {
    NV_PDL_ENTRY();
    const long long wid = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int nblk = N >> 7;
    if (wid >= (long long)R * nblk) return;
    const int r = (int)(wid / nblk), blk = (int)(wid - (long long)r * nblk);
    const int lane = threadIdx.x & 31;
    const size_t base = (size_t)r * N + (size_t)blk * 128 + lane * 4;
    const __half* sr = svh_all + (size_t)row_e[r] * N + (size_t)blk * 128 + lane * 4;
    float v[4];
    #pragma unroll
    for (int q = 0; q < 4; ++q) v[q] = nv_h2f(yraw[base + q]);
    nv_had128(v, lane);
    #pragma unroll
    for (int q = 0; q < 4; ++q) y[base + q] = nv_f2h(v[q] * nv_h2f(sr[q]));
}
// per-row input transform with the row's expert (the down input): xh[r] = H128(silu(g[r]) * u[r] * suh[row_e[r]]),
// the silu product rounded to f16 first (nv_silu_mul's rounding point). grid R * (K/128) warps.
extern "C" __global__ void nv_had_suh_rows_silu(const __half* __restrict__ g, const __half* __restrict__ u,
                                                const __half* __restrict__ suh_all, const int* __restrict__ row_e,
                                                __half* __restrict__ xh, int R, int K) {
    NV_PDL_ENTRY();
    const long long wid = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int nblk = K >> 7;
    if (wid >= (long long)R * nblk) return;
    const int r = (int)(wid / nblk), blk = (int)(wid - (long long)r * nblk);
    const int lane = threadIdx.x & 31;
    const size_t base = (size_t)r * K + (size_t)blk * 128 + lane * 4;
    const __half* sr = suh_all + (size_t)row_e[r] * K + (size_t)blk * 128 + lane * 4;
    float v[4];
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const float gv = nv_h2f(g[base + q]);
        const float d = nv_h2f(nv_f2h(gv / (1.f + __expf(-gv)) * nv_h2f(u[base + q])));
        v[q] = d * nv_h2f(sr[q]);
    }
    nv_had128(v, lane);
    #pragma unroll
    for (int q = 0; q < 4; ++q) xh[base + q] = nv_f2h(v[q]);
}
// combine in j order per token: x[m][:] (+)= sum_j wts[m][j] * y_d[rowmap[m][j]] (skipping -1). grid M, block 1024.
extern "C" __global__ void nv_moe_gather_combine(float* __restrict__ x, const __half* __restrict__ y_d, const int* __restrict__ rowmap,
                                                 const float* __restrict__ wts, int k, int H, int accumulate) {
    NV_PDL_ENTRY();
    const int m = blockIdx.x;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < k; j++) {
            const int r = rowmap[m * k + j];
            if (r >= 0) acc += wts[m * k + j] * nv_h2f(y_d[(size_t)r * H + i]);
        }
        if (accumulate) x[(size_t)m * H + i] += acc; else x[(size_t)m * H + i] = acc;
    }
}

// S6 bench control: the per-step body regardless of nv_a1_fits (same values, no A staging / prefetch).
extern "C" __global__ void nv_gemm_nostage(const uint16_t* __restrict__ trellis, const __half* __restrict__ xh,
                                           __half* __restrict__ yraw, int M, int K, int N, int bpb) {
    NV_PDL_ENTRY();
    const int KB = K >> 4;
    NV_BPB_SWITCH((nv_gemm_body<BPB, false>(trellis, xh, yraw, nullptr, M, K, N, (int)blockIdx.x, 0, KB, 0)));
}

// S6: grouped expert GEMM with split-K: grid (E_cap * N/128, KS); fp32 partials ws[e][ks][M][N]
// (the K-ascending fixed-order chunks of nv_gemm_body PARTIAL), then nv_ks_combine_grouped.
extern "C" __global__ void nv_gemm_grouped_ks(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs,
                                              const __half* __restrict__ xh, float* __restrict__ ws,
                                              int M, int K, int N, int bpb, const int* __restrict__ esel) {
    NV_PDL_ENTRY();
    const int tiles = N / 128;
    const int expert = blockIdx.x / tiles;
    const int tile = blockIdx.x - expert * tiles;
    if (expert >= *esel) return;
    const int ks = gridDim.y, ks_id = blockIdx.y;
    const int KB = K >> 4;
    const int chunk = KB / ks;
    const int kb0 = ks_id * chunk, kb1 = kb0 + chunk;
    const uint16_t* tr = base + (size_t)offs[expert];
    const __half* xe = xh + (long long)expert * M * K;
    float* we = ws + (long long)expert * ks * M * N;
    NV_BPB_SWITCH((nv_gemm_body<BPB, true>(tr, xe, nullptr, we, M, K, N, tile, kb0, kb1, ks_id)));
}
// y[e][m][n] = sum_ks ws[e][ks][m][n] (fixed order), f16 RN. grid ceil(E_cap*M*N/256).
extern "C" __global__ void nv_ks_combine_grouped(const float* __restrict__ ws, __half* __restrict__ y, int M, int N, int KS,
                                                 const int* __restrict__ esel) {
    NV_PDL_ENTRY();
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long per = (long long)M * N;
    const int e = (int)(idx / per);
    if (e >= *esel) return;
    const long long mn = idx - (long long)e * per;
    const float* w = ws + (long long)e * KS * per + mn;
    float acc = 0.f;
    for (int s2 = 0; s2 < KS; s2++) acc += w[(long long)s2 * per];
    y[idx] = __float2half_rn(acc);
}

