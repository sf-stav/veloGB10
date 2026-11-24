// 73_mix_router.cuh — the MoE router (model.py 794-829): fp32 scores = x . W^T, sqrt(softplus), noaux_tc top-k on scores + bias
// (the bias selects, the unbiased score weights), renormalise with + 1e-20, multiply by route_scale.
#pragma once

// ---- d41_router_scores_r{8,16,32}: scores[row][e] = sum_k x[row][k] * W[e][k], fp32 accumulate. W f32 / bf16 / f16 [E][dim],
// dim % 1024 == 0 (5120). One BLOCK = one expert: 4 warps split k in 4 contiguous quarters (more warps in flight per SM: the
// kernel is latency-bound), each lane owns 8 consecutive k per step; a quarter is reduced with the fixed reduce-scatter tree and
// the quarters are added in order 0..3, so a row's score is identical for any row count (RC is only the register budget).
struct d41_router_scores_args {
    const d41_bf16* x; long long x_ld;
    const void* w; long long w_ld;
    float* scores; long long s_ld;           // scores[row * s_ld + e]
    unsigned w_fmt; int dim; int E; int rows;   // rows <= RC (8 / 16 / 32); the host splits larger batches
};
template <int RC, int EP>
__device__ __forceinline__ void d41_router_scores_body(const d41_router_scores_args& a) {
    D41_PDL_ENTRY();
    __shared__ float sq[4][EP][32];
    const int lane = threadIdx.x & 31, wq = threadIdx.x >> 5;
    const int e0 = blockIdx.x * EP;                 // EP experts per block share every x load
    const int wb = a.w_fmt == D41_FMT_F32 ? 4 : 2;
    const int kq = a.dim >> 2;                      // k per warp
    const int k0 = wq * kq;
    float acc[EP][RC];
#pragma unroll
    for (int j = 0; j < EP; ++j)
#pragma unroll
        for (int r = 0; r < RC; ++r) acc[j][r] = 0.f;
    for (int kb = 0; kb < kq; kb += 1024) {         // up to 4 steps of 256 in flight
        uint4 wl[4][EP][2];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int k = k0 + kb + u * 256 + lane * 8;
#pragma unroll
            for (int j = 0; j < EP; ++j) {
                const char* wrow = (const char*)a.w + (long long)min(e0 + j, a.E - 1) * a.w_ld * wb;
                if (kb + u * 256 < kq) {
                    wl[u][j][0] = __ldg(reinterpret_cast<const uint4*>(wrow + (long long)k * wb));
                    wl[u][j][1] = (wb == 4) ? __ldg(reinterpret_cast<const uint4*>(wrow + (long long)k * wb + 16)) : make_uint4(0, 0, 0, 0);
                }
            }
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int k = k0 + kb + u * 256 + lane * 8;
            if (kb + u * 256 < kq) {
                float wv[EP][8];
#pragma unroll
                for (int j = 0; j < EP; ++j) {
                    if (a.w_fmt == D41_FMT_F32) {
                        wv[j][0] = __uint_as_float(wl[u][j][0].x); wv[j][1] = __uint_as_float(wl[u][j][0].y); wv[j][2] = __uint_as_float(wl[u][j][0].z); wv[j][3] = __uint_as_float(wl[u][j][0].w);
                        wv[j][4] = __uint_as_float(wl[u][j][1].x); wv[j][5] = __uint_as_float(wl[u][j][1].y); wv[j][6] = __uint_as_float(wl[u][j][1].z); wv[j][7] = __uint_as_float(wl[u][j][1].w);
                    } else {
                        unsigned q[4] = {wl[u][j][0].x, wl[u][j][0].y, wl[u][j][0].z, wl[u][j][0].w};
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            if (a.w_fmt == D41_FMT_F16) {
                                wv[j][2 * i] = __half2float(__ushort_as_half((unsigned short)(q[i] & 0xFFFFu)));
                                wv[j][2 * i + 1] = __half2float(__ushort_as_half((unsigned short)(q[i] >> 16)));
                            } else {
                                wv[j][2 * i] = d41_bf2f_u16((unsigned short)(q[i] & 0xFFFFu));
                                wv[j][2 * i + 1] = d41_bf2f_u16((unsigned short)(q[i] >> 16));
                            }
                        }
                    }
                }
#pragma unroll
                for (int r = 0; r < RC; ++r) {
                    if (r < a.rows) {
                        uint4 xu = __ldg(reinterpret_cast<const uint4*>(a.x + (long long)r * a.x_ld + k));
                        const unsigned xq[4] = {xu.x, xu.y, xu.z, xu.w};
                        float xf[8];
#pragma unroll
                        for (int i = 0; i < 4; ++i) { xf[2 * i] = d41_bf2f_u16((unsigned short)(xq[i] & 0xFFFFu)); xf[2 * i + 1] = d41_bf2f_u16((unsigned short)(xq[i] >> 16)); }
#pragma unroll
                        for (int j = 0; j < EP; ++j) {
                            float sacc = acc[j][r];
#pragma unroll
                            for (int i = 0; i < 8; ++i) sacc = fmaf(wv[j][i], xf[i], sacc);
                            acc[j][r] = sacc;
                        }
                    }
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < EP; ++j) {
        float v[32];
#pragma unroll
        for (int r = 0; r < 32; ++r) v[r] = r < RC ? acc[j][r] : 0.f;
        sq[wq][j][lane] = d41_rs32(v, lane);
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < EP; ++j)
        if (wq == j && lane < a.rows && e0 + j < a.E) a.scores[(long long)lane * a.s_ld + e0 + j] = ((sq[0][j][lane] + sq[1][j][lane]) + sq[2][j][lane]) + sq[3][j][lane];
}
// ---- d41_router_scores_wide (leg PP1): the prefill-width router, SAME BITS as the r* entries above for every row.
// The r* body re-reads every x row from L2 once per expert (a 2,048-row prefill call: 64 launches of 32 rows, each block one
// expert): ~8 GB of L2 traffic per layer and chunk for 4 G FMAs. This entry keeps the r* arithmetic of a score exactly and
// only shares the loads:
//   * warp = (quarter q, expert group); the quarter's 32 lanes own k = q * kq + s * 256 + lane * 8 + i (steps s ascending,
//     i = 0..7), one fmaf chain per (row, expert) in that order with the same operands (w, x, acc): the r* chain;
//   * the quarter sums of a (row, expert) leave the warp through d41_rs32 (a fixed xor tree whose result for a slot does not
//     depend on the slot nor on the other slots' values: the r* tree), then ((q0 + q1) + q2) + q3;
//   * a block = EG expert groups x 4 quarters = EG * EP experts x RT rows; the x rows of a step are staged in shared memory
//     once per block (bf16, 4 quarters x 256 k x RT rows) and every expert group reads them there.
// Grid (ceil(E / (EG * EP)), ceil(rows / RT)), 128 * EG threads. dim % 1024 == 0, bf16 / f16 / f32 weights.
template <int EP, int RT, int EG>
__device__ __forceinline__ void d41_router_wide_body(const d41_router_scores_args& a) {
    D41_PDL_ENTRY();
    __shared__ __align__(16) unsigned short xs[RT][4 * 256];       // one step: row r, quarter q at [q * 256 ..]
    __shared__ float red[4][EG * EP][RT + 1];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int q = warp & 3, eg = warp >> 2;
    const int e0 = blockIdx.x * (EG * EP) + eg * EP;
    const int r0 = blockIdx.y * RT;
    const int nr = min(RT, a.rows - r0);
    const int wb = a.w_fmt == D41_FMT_F32 ? 4 : 2;
    const int kq = a.dim >> 2;
    float acc[EP][RT];
#pragma unroll
    for (int j = 0; j < EP; ++j)
#pragma unroll
        for (int r = 0; r < RT; ++r) acc[j][r] = 0.f;
    for (int ks = 0; ks < kq; ks += 256) {
        // the weights of this warp's experts for this step (issued before the stage barrier)
        uint4 wl[EP][2];
        const int k = q * kq + ks + lane * 8;
#pragma unroll
        for (int j = 0; j < EP; ++j) {
            const char* wrow = (const char*)a.w + (long long)min(e0 + j, a.E - 1) * a.w_ld * wb;
            wl[j][0] = __ldg(reinterpret_cast<const uint4*>(wrow + (long long)k * wb));
            wl[j][1] = (wb == 4) ? __ldg(reinterpret_cast<const uint4*>(wrow + (long long)k * wb + 16)) : make_uint4(0, 0, 0, 0);
        }
        __syncthreads();                                            // the previous step's readers are done
        for (int o = threadIdx.x; o < RT * 128; o += blockDim.x) {  // RT rows x 4 quarters x 32 uint4
            const int r = o >> 7, qq = (o >> 5) & 3, c = o & 31;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (r < nr) v = __ldg(reinterpret_cast<const uint4*>(a.x + (long long)(r0 + r) * a.x_ld + qq * kq + ks + c * 8));
            *reinterpret_cast<uint4*>(&xs[r][qq * 256 + c * 8]) = v;
        }
        __syncthreads();
        float wv[EP][8];
#pragma unroll
        for (int j = 0; j < EP; ++j) {
            if (a.w_fmt == D41_FMT_F32) {
                wv[j][0] = __uint_as_float(wl[j][0].x); wv[j][1] = __uint_as_float(wl[j][0].y); wv[j][2] = __uint_as_float(wl[j][0].z); wv[j][3] = __uint_as_float(wl[j][0].w);
                wv[j][4] = __uint_as_float(wl[j][1].x); wv[j][5] = __uint_as_float(wl[j][1].y); wv[j][6] = __uint_as_float(wl[j][1].z); wv[j][7] = __uint_as_float(wl[j][1].w);
            } else {
                const unsigned u[4] = {wl[j][0].x, wl[j][0].y, wl[j][0].z, wl[j][0].w};
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    if (a.w_fmt == D41_FMT_F16) {
                        wv[j][2 * i] = __half2float(__ushort_as_half((unsigned short)(u[i] & 0xFFFFu)));
                        wv[j][2 * i + 1] = __half2float(__ushort_as_half((unsigned short)(u[i] >> 16)));
                    } else {
                        wv[j][2 * i] = d41_bf2f_u16((unsigned short)(u[i] & 0xFFFFu));
                        wv[j][2 * i + 1] = d41_bf2f_u16((unsigned short)(u[i] >> 16));
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < RT; ++r) {
            if (r < nr) {
                const uint4 xu = *reinterpret_cast<const uint4*>(&xs[r][q * 256 + lane * 8]);
                const unsigned xq[4] = {xu.x, xu.y, xu.z, xu.w};
                float xf[8];
#pragma unroll
                for (int i = 0; i < 4; ++i) { xf[2 * i] = d41_bf2f_u16((unsigned short)(xq[i] & 0xFFFFu)); xf[2 * i + 1] = d41_bf2f_u16((unsigned short)(xq[i] >> 16)); }
#pragma unroll
                for (int j = 0; j < EP; ++j) {
                    float sacc = acc[j][r];
#pragma unroll
                    for (int i = 0; i < 8; ++i) sacc = fmaf(wv[j][i], xf[i], sacc);
                    acc[j][r] = sacc;
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < EP; ++j) {
        float v[32];
#pragma unroll
        for (int r = 0; r < 32; ++r) v[r] = r < RT ? acc[j][r] : 0.f;
        const float t = d41_rs32(v, lane);
        if (lane < RT) red[q][eg * EP + j][lane] = t;
    }
    __syncthreads();
    for (int o = threadIdx.x; o < EG * EP * RT; o += blockDim.x) {
        const int el = o / RT, r = o - el * RT;
        const int e = blockIdx.x * (EG * EP) + el;
        if (r < nr && e < a.E) a.scores[(long long)(r0 + r) * a.s_ld + e] = ((red[0][el][r] + red[1][el][r]) + red[2][el][r]) + red[3][el][r];
    }
}
extern "C" __global__ void __launch_bounds__(256) d41_router_scores_wide(d41_router_scores_args a) { d41_router_wide_body<4, 16, 2>(a); }
extern "C" __global__ void __launch_bounds__(512) d41_router_scores_wide4(d41_router_scores_args a) { d41_router_wide_body<4, 16, 4>(a); }
extern "C" __global__ void __launch_bounds__(256) d41_router_scores_wide8(d41_router_scores_args a) { d41_router_wide_body<8, 8, 2>(a); }
// ---- the tf32 tensor-core scores (OPT-IN, `--dsv41-router-tf32 on`; bf16 / f16 weights; leg PK1). The arithmetic of a score
// on this path (every launch shape and row count of the tc entries computes these bits):
//   * K in D41_RT_PARTS = 16 contiguous parts of dim / 16 (320 k at dim 5120);
//   * a part = its 64-k groups in ascending order; group k0 .. k0 + 63 = eight tf32 m16n8k8 MMAs with a ZERO accumulator
//     (products exact: bf16 and f16 values are exact in tf32), each MMA's result added in fp32 (__fadd_rn) to the part's sum;
//     inside a group, MMA step s takes the k pairs {k0 + 16t + 2s, k0 + 16t + 2s + 1} (t = 0..3): a fixed permutation of k
//     (the dot is a sum, any fixed assignment is the same math) that lets each lane load 16 contiguous values (the 4 lanes
//     of an expert read one whole 128-byte line per group);
//   * the 16 part sums added in part order 0..15.
// The MMA's A operand is x (16 rows per m-tile), B is W (8 experts per CTA tile). A score depends only on its own row of x and
// its expert's row of W (an MMA output element reads one A row and one B column): batch invariant. The zero accumulator keeps
// the GB10 tensor-core accumulator out of the chain (KN2 / KE2: it is not a full fp32 add).
// Launch shape: CTA b < E / 8 owns experts 8b .. 8b + 7 (CTAs past that exit), its warps take the parts in turn; any block of
// 32 .. 512 threads gives the same bits.
#define D41_RT_ET 8
#define D41_RT_PARTS 16
__device__ __forceinline__ void d41_rt_mma(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.f));
}
// a 16-bit word pair -> two fp32 bit patterns (exact; tf32 reads the top 19 bits)
template <bool F16>
__device__ __forceinline__ void d41_rt_cvt(unsigned w, unsigned& lo, unsigned& hi) {
    if (F16) {
        lo = __float_as_uint(__half2float(__ushort_as_half((unsigned short)(w & 0xFFFFu))));
        hi = __float_as_uint(__half2float(__ushort_as_half((unsigned short)(w >> 16))));
    } else { lo = w << 16; hi = w & 0xFFFF0000u; }
}
__device__ __forceinline__ unsigned d41_rt_w(const uint4& v, int s) { return s == 0 ? v.x : s == 1 ? v.y : s == 2 ? v.z : v.w; }
// MT0 = m-tiles of 16 rows (0: rows <= 8, one m-tile whose rows g + 8 are padding: no loads, no registers). A warp runs NP
// parts together and keeps B groups of each in flight (NP / B only change the load schedule, never the arithmetic).
// red[part][expert][row] (row stride RS).
template <int MT0, int B, int NP, bool F16>
__device__ __forceinline__ void d41_router_tc_parts(const d41_router_scores_args& a, int e0, float* red) {
    constexpr int MT = MT0 == 0 ? 1 : MT0, NA = MT0 == 0 ? 1 : 2;    // m-tiles, A rows loaded per m-tile
    constexpr int RS = MT * 16 + 1;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int kp = a.dim / D41_RT_PARTS;                 // k per part; multiple of 64
    const unsigned short* wr = (const unsigned short*)a.w + (long long)min(e0 + g, a.E - 1) * a.w_ld + 16 * t;
    const unsigned short* xr[NA * MT];
    bool xv[NA * MT];
#pragma unroll
    for (int j = 0; j < NA * MT; ++j) {                  // A rows: 16 m + g (+ 8 for the second of a pair)
        const int r = 16 * (j / NA) + 8 * (j % NA) + g;
        xv[j] = r < a.rows;
        xr[j] = (const unsigned short*)a.x + (long long)(xv[j] ? r : 0) * a.x_ld + 16 * t;
    }
    for (int p0 = warp; p0 < D41_RT_PARTS; p0 += nw * NP) {
        float acc[NP][MT][4];
#pragma unroll
        for (int q = 0; q < NP; ++q)
#pragma unroll
            for (int m = 0; m < MT; ++m) { acc[q][m][0] = 0.f; acc[q][m][1] = 0.f; acc[q][m][2] = 0.f; acc[q][m][3] = 0.f; }
        for (int gb = 0; gb < kp; gb += 64 * B) {
            uint4 wv[NP][B], xb[NP][B][NA * MT], wv2[NP][B], xb2[NP][B][NA * MT];
#pragma unroll
            for (int q = 0; q < NP; ++q) {
                const int p = p0 + q * nw;
#pragma unroll
                for (int u = 0; u < B; ++u) {
                    const int k = p * kp + gb + 64 * u;
                    const bool ok = p < D41_RT_PARTS && gb + 64 * u < kp;
                    wv[q][u] = ok ? __ldg(reinterpret_cast<const uint4*>(wr + k)) : make_uint4(0, 0, 0, 0);
                    wv2[q][u] = ok ? __ldg(reinterpret_cast<const uint4*>(wr + k + 8)) : make_uint4(0, 0, 0, 0);
#pragma unroll
                    for (int j = 0; j < NA * MT; ++j) {
                        xb[q][u][j] = (ok && xv[j]) ? __ldg(reinterpret_cast<const uint4*>(xr[j] + k)) : make_uint4(0, 0, 0, 0);
                        xb2[q][u][j] = (ok && xv[j]) ? __ldg(reinterpret_cast<const uint4*>(xr[j] + k + 8)) : make_uint4(0, 0, 0, 0);
                    }
                }
            }
#pragma unroll
            for (int q = 0; q < NP; ++q) {
#pragma unroll
                for (int u = 0; u < B; ++u) {
                    if (gb + 64 * u < kp) {
#pragma unroll
                        for (int s = 0; s < 8; ++s) {
                            unsigned b0, b1;
                            d41_rt_cvt<F16>(d41_rt_w(s < 4 ? wv[q][u] : wv2[q][u], s & 3), b0, b1);
#pragma unroll
                            for (int m = 0; m < MT; ++m) {
                                const uint4& X0 = s < 4 ? xb[q][u][NA * m] : xb2[q][u][NA * m];
                                const uint4& X1 = s < 4 ? xb[q][u][NA * m + NA - 1] : xb2[q][u][NA * m + NA - 1];
                                const unsigned xlo = d41_rt_w(X0, s & 3), xhi = NA == 2 ? d41_rt_w(X1, s & 3) : 0u;
                                float d[4];
                                d41_rt_mma(d, xlo << 16, xhi << 16, xlo & 0xFFFF0000u, xhi & 0xFFFF0000u, b0, b1);
                                acc[q][m][0] = __fadd_rn(acc[q][m][0], d[0]); acc[q][m][1] = __fadd_rn(acc[q][m][1], d[1]);
                                acc[q][m][2] = __fadd_rn(acc[q][m][2], d[2]); acc[q][m][3] = __fadd_rn(acc[q][m][3], d[3]);
                            }
                        }
                    }
                }
            }
        }
        // C fragment: c0 (row g, expert 2t), c1 (g, 2t + 1), c2 (row g + 8, 2t), c3 (g + 8, 2t + 1)
#pragma unroll
        for (int q = 0; q < NP; ++q) {
            const int p = p0 + q * nw;
            if (p < D41_RT_PARTS) {
                float* rp = red + (long long)p * (D41_RT_ET * RS);
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    const int r = 16 * m + g;
                    rp[(2 * t) * RS + r] = acc[q][m][0]; rp[(2 * t + 1) * RS + r] = acc[q][m][1];
                    rp[(2 * t) * RS + r + 8] = acc[q][m][2]; rp[(2 * t + 1) * RS + r + 8] = acc[q][m][3];
                }
            }
        }
    }
}
template <int MT0, int B, int NP>
__device__ __forceinline__ void d41_router_tc(const d41_router_scores_args& a) {
    D41_PDL_ENTRY();
    constexpr int MT = MT0 == 0 ? 1 : MT0;
    constexpr int RS = MT * 16 + 1;
    __shared__ float red[D41_RT_PARTS * D41_RT_ET * RS];     // one per entry (both formats share it)
    const int e0 = blockIdx.x * D41_RT_ET;
    if (e0 >= a.E) return;
    if (a.w_fmt == D41_FMT_F16) d41_router_tc_parts<MT0, B, NP, true>(a, e0, red); else d41_router_tc_parts<MT0, B, NP, false>(a, e0, red);
    __syncthreads();
    for (int o = threadIdx.x; o < D41_RT_ET * MT * 16; o += blockDim.x) {
        const int e = o / (MT * 16), r = o - e * (MT * 16);
        if (r < a.rows && e0 + e < a.E) {
            float s = red[e * RS + r];
#pragma unroll
            for (int p = 1; p < D41_RT_PARTS; ++p) s = __fadd_rn(s, red[(p * D41_RT_ET + e) * RS + r]);
            a.scores[(long long)r * a.s_ld + e0 + e] = s;
        }
    }
}
// The DEFAULT router is the fp32 CUDA-core body above through the r* entries (grid E, E / 2 for r12; 128 threads): the
// forward's path, bit-identical to the KM kernels at every width. The tf32 tensor-core path is OPT-IN
// (`--dsv41-router-tf32 on`, orchestrator decision 2026-11-19: reduced-precision router logits flip near-tie top-k, AGENTS
// §7; the products here are exact, bf16 / f16 being exact in tf32, but the summation order differs, so the scores change
// bits): entries d41_router_tc_r8 / r16 / r32, grid E / 8, 512 threads (256 for tc_r32), bf16 / f16 gates with E % 8 == 0
// and dim % 1024 == 0; any launch shape of them gives the same bits.
extern "C" __global__ void __launch_bounds__(128) d41_router_scores_r8(d41_router_scores_args a) { d41_router_scores_body<8, 1>(a); }
extern "C" __global__ void __launch_bounds__(128) d41_router_scores_r12(d41_router_scores_args a) { d41_router_scores_body<12, 2>(a); }
extern "C" __global__ void __launch_bounds__(128) d41_router_scores_r24(d41_router_scores_args a) { d41_router_scores_body<24, 1>(a); }
extern "C" __global__ void __launch_bounds__(128) d41_router_scores_r32(d41_router_scores_args a) { d41_router_scores_body<32, 1>(a); }
// tf32 tensor-core entries (opt-in): tc_r8 up to 8 rows, tc_r16 16, tc_r32 32.
extern "C" __global__ void __launch_bounds__(512) d41_router_tc_r8(d41_router_scores_args a) { d41_router_tc<0, 5, 1>(a); }
extern "C" __global__ void __launch_bounds__(512) d41_router_tc_r16(d41_router_scores_args a) { d41_router_tc<1, 3, 1>(a); }
extern "C" __global__ void __launch_bounds__(256) d41_router_tc_r32(d41_router_scores_args a) { d41_router_tc<2, 2, 1>(a); }

// ---- d41_router_select: one BLOCK (128 threads) per row. s = sqrt(softplus(score / temp)); pick the top-k of s + bias (bias_vl for
// image rows), ties to the LOWER expert id; weights = s[pick] / (sum + 1e-20) * route_scale. Output order = descending biased
// score (torch.topk sorted). E <= 512, topk <= 8. bias: f32 or f16 (the EXL3 pack stores the DSpark biases as F16).
// (A warp-per-row version was latency-bound at 12 us: the per-element log1p/exp chains now spread over 128 threads.)
struct d41_router_select_args {
    const float* scores; long long s_ld;
    const void* bias; const void* bias_vl; const unsigned char* image_mask;   // bias_vl / image_mask may be null
    float* out_w; int* out_i; long long o_ld;
    unsigned bias_fmt; int E; int topk; int rows;
    float route_scale; float temp; int norm_topk;
};
extern "C" __global__ void __launch_bounds__(128) d41_router_select(d41_router_select_args a) {
    D41_PDL_ENTRY();
    __shared__ unsigned long long skey[8]; __shared__ float swt[8]; __shared__ int sit[8];
    const int tid = threadIdx.x;
    if (tid < 8) skey[tid] = 0ull;
    const int row = blockIdx.x;
    const bool vl = a.image_mask && a.bias_vl && a.image_mask[row];
    const void* bias = vl ? a.bias_vl : a.bias;
    float s[4], t[4], rawv[4], bv_[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int e = tid + 128 * i;
        rawv[i] = e < a.E ? a.scores[(long long)row * a.s_ld + e] : 0.f;
        bv_[i] = e < a.E ? d41_ldw(bias, a.bias_fmt, e) : 0.f;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int e = tid + 128 * i;
        if (e < a.E) {
            float x = rawv[i];
            if (a.temp != 1.0f) x = x / a.temp;
            const float sp = x > 20.0f ? x : log1pf(expf(x));
            s[i] = sqrtf(sp);
            t[i] = s[i] + bv_[i];
        } else { s[i] = 0.f; t[i] = -INFINITY; }
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
        if (kk < a.topk) {
            // key = (order-preserving f32 bits, ~expert id): the max is the largest score, ties to the LOWER id
            unsigned long long mk = 0ull;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const unsigned u = __float_as_uint(t[i]);
                const unsigned ord = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
                const unsigned long long k = ((unsigned long long)ord << 32) | (unsigned)(0xFFFFFFFFu - (unsigned)(tid + 128 * i));
                if (t[i] > -INFINITY && k > mk) mk = k;
            }
            // warp maximum first (exact), then one shared atomic per warp (PK1: 4 contended atomics per round, not 128)
#pragma unroll
            for (int o = 16; o >= 1; o >>= 1) { const unsigned long long ok = __shfl_xor_sync(0xffffffffu, mk, o); mk = ok > mk ? ok : mk; }
            if ((tid & 31) == 0 && mk) atomicMax(&skey[kk], mk);
            __syncthreads();
            const int bi = (int)(0xFFFFFFFFu - (unsigned)(skey[kk] & 0xFFFFFFFFull));
#pragma unroll
            for (int i = 0; i < 4; ++i) { if (tid + 128 * i == bi) { swt[kk] = s[i]; sit[kk] = bi; t[i] = -INFINITY; } }
        }
    }
    __syncthreads();
    if (tid == 0) {
        float sum = 0.f;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) if (kk < a.topk) sum = kk == 0 ? swt[0] : sum + swt[kk];
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            if (kk < a.topk) {
                float w = swt[kk];
                if (a.norm_topk && a.topk > 1) w = w / (sum + 1e-20f);
                a.out_w[(long long)row * a.o_ld + kk] = w * a.route_scale;
                a.out_i[(long long)row * a.o_ld + kk] = sit[kk];
            }
        }
    }
}
