// 19_exl3_moe_cp.cuh (leg PM1): routed-MoE DECODE (<= 32 rows) in the external reference's TP2 decode structure
// (PLAN/DSV41_RX3_REPORT.md B.3: decode_prep, grouped_cp_kernel<...,1> gate/up, grouped_cp_kernel<...,2> down + combine),
// with OUR arithmetic (bit-identical to 15_* / 13_*):
//   d41x_mc_prep   one CTA computes the call's routing table once (15_*'s compaction, into the scratch) and bumps the
//                  call epoch; the others: a warp per (id slot or shared row, 128-block, matrix): xh1 = fp16(H128(x * suh))
//                  (15_*'s d41x_prep4). It lets the next grid launch at its first instruction.
//   d41x_mc_gu_aR  SHORT CTAs of 128 threads (<= 128 registers, 4 per SM): CTA = (slot, 128-column block, matrix, K part of
//                  sk), slot-major, the shared expert (slot 0) first; each warp owns 2 of the block's 8 n16 tiles over the
//                  part's k-blocks and streams them through ITS OWN 3-stage cp.async ring (stage = 4 k16 steps of its 2 tiles,
//                  2 stages in flight); the activation fragments come one k16 step ahead from global (L2). Waits for prep
//                  first, then lets the down grid launch. The (slot, block)'s last arrival folds the K parts, runs SwiGLU +
//                  w2's input transform (15_*'s act), and the slot's last block publishes ready[s] = epoch.
//   d41x_mc_dn_aR  CTA = (slot, output 128-block), slot-major (shared expert first); reads the routing (prep has completed:
//                  every gate/up CTA waited for it before it let this grid launch), issues its first two ring stages of
//                  w2, THEN waits only for its own slot's ready flag (128 ns sleeps); H128 -> h2; the last slot to finish an
//                  output block combines it (15_*'s combine order: y = (0|y + routed experts in ascending id) + shared).
// NUMERICS (why this equals 15_* bit for bit): every output column of w1 / w3 / w2 is computed by one warp per k-block as one
// m16n8k16 MMA chain of the 8 k16 steps of that 128-k block from a zero accumulator (the same decode, the same A values, the
// same instruction), and the chains are added in f32 from 0 in ASCENDING k-block order (D41X_KBLK): w2 and K part 0 of w1 /
// w3 fold their own chains in registers (sum = 0; sum += chain, as 15_*); K parts > 0 store each k-block's chain (f32,
// [k-block][pair][mat][col], PV1's chain layout) and the (slot, block)'s last arrival continues part 0's fold over those
// chains in k-block order: the same f32 additions in the same order as one pass over K, for any sk and any arrival order.
// A warp never splits a k-block (the reference's warps own 20 k16 steps = 2.5 of our k-blocks: that split is not ours).
// SMEM: [running sums: 4 warps x 4 MT quads x 32 lanes x 16 B][ring: 4 warps x S stages x KPS x 2 tiles x 32 K B]
// (down: the [rows][132] f32 epilogue stage aliases the ring).
#pragma once

#define D41X_MC_THREADS 128
#ifndef D41X_MC_STAGES
#define D41X_MC_STAGES 3                 // gate/up: ring stages per warp (2 in flight)
#endif
#ifndef D41X_MC_KPS
#define D41X_MC_KPS 4                    // gate/up: k16 steps per ring stage (8 / KPS stages = one 128-k accumulation block)
#endif
#ifndef D41X_MC_DSTAGES
#define D41X_MC_DSTAGES 3                // down: ring stages per warp
#endif
#ifndef D41X_MC_DKPS
#define D41X_MC_DKPS 4                   // down: k16 steps per ring stage
#endif

// P7K1: per-(K, AR) ring geometry keeping every (AR = 8 / 16, K = 2..5) instance <= 23.5 KB dynamic smem (4 CTAs/SM on GB10):
// K = 2 uses KPS = 8, S = 3 (12 KB, 16 k16 steps in flight, NCH = 64); K = 3 at AR = 16 uses KPS = 4, S = 5 in gu (15 KB + 8 KB
// sums = 23 KB, 16 k16 steps in flight) and S = 4 in dn (12 KB + 8 KB = 20 KB, 3 pre-wait stages); AR = 8 keeps S = 3 for K >= 3
// (with 4 KB float2 sums) so 1-row dn only pre-issues 2 stages during gu's tail. Arithmetic is 100% unchanged (8-step k-blocks).
__host__ __device__ constexpr int d41x_mc_kps(int k, int ar) { return (k == 2 && ar <= 16) ? 8 : D41X_MC_KPS; }
__host__ __device__ constexpr int d41x_mc_stages(int k, int ar) { return (k == 3 && ar == 16) ? 5 : D41X_MC_STAGES; }
__host__ __device__ constexpr int d41x_mc_dkps(int k, int ar) { return (k == 2 && ar <= 16) ? 8 : D41X_MC_DKPS; }
__host__ __device__ constexpr int d41x_mc_dstages(int k, int ar) { return (k == 3 && ar == 16) ? 4 : D41X_MC_DSTAGES; }
__host__ __device__ constexpr int d41x_mc_sum_bytes(int ar) { return ar <= 8 ? 4 * 4 * 32 * 8 : 4 * (ar > 16 ? 8 : 4) * 32 * 16; }
__host__ __device__ constexpr int d41x_mc_ring_bytes(int k, int ar, bool dn) {
    return 4 * (dn ? d41x_mc_dstages(k, ar) : d41x_mc_stages(k, ar)) * (dn ? d41x_mc_dkps(k, ar) : d41x_mc_kps(k, ar)) * 2 * 32 * k;
}

struct D41xMcArgs {
    D41xMdArgs a;                        // 15_*'s call arguments (chain = the K-part chain scratch when sk > 1)
    D41xMdRoute* route;                  // scratch: the call's routing table (header + mask[E] + slot_of[E]), written by prep
    int sk;                              // gate/up K parts (sk | dim / 128)
    int nsl;                             // slot rows of the grids (>= the call's slots: rows * topk + shared rows, <= MAXS)
    int nby;                             // gate/up column blocks per slot (>= every expert's nblk)
    int nmt;                             // KM1: m tiles of 16 rows a slot may have (1 = a slot is one tile; 2 = 17-32 rows). The grids
                                         // carry nmt copies of every slot (virtual slot = slot * nmt + tile); a tile past the slot's rows exits
};

// harness-only timeline hooks (tools/dsv41_pm1/pm_bench.cu, -DD41X_MC_TRACE): per CTA [start, data/ready, end, tag]
#define D41X_MC_TN 16384
#ifdef D41X_MC_TRACE
__device__ unsigned long long d41x_mc_trace[D41X_MC_TN][4];
__device__ unsigned long long d41x_mc_rtrace[D41X_MD_MAXS];
__device__ __forceinline__ unsigned long long d41x_mc_now() { unsigned long long v; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(v)); return v; }
#define D41X_MC_T(idx, f, v) do { if (threadIdx.x == 0 && (idx) < D41X_MC_TN) d41x_mc_trace[(idx)][(f)] = (v); } while (0)
#else
#define D41X_MC_T(idx, f, v) do { } while (0)
#endif

__device__ __forceinline__ int d41x_mc_ldi(const int* p) { return __ldcg(p); }
__device__ __forceinline__ int d41x_mc_lds(const short* p) {   // a 16-bit field of the routing table, through L2
    const unsigned w = __ldcg((const unsigned*)((size_t)p & ~(size_t)3));
    return (int)(short)(((size_t)p & 2) ? (w >> 16) : (w & 0xFFFFu));
}

// ---- one warp's tiles over a k-block range: its own ring, A one k16 step ahead ----------------------------------------
// T: trellis of the warp's first tile at k16 step 0 of the matrix (tiles of a k-row at 32 K B, k-rows `kstride` B apart);
// X: A rows of the slot ([cnt rows] at `ldx` halves, column k0 = 0); acc / sum per [m tile][tile][n8 half][4].
// PRE: the first D41X_MC_STAGES - 1 stages were issued by the caller (pre-wait issue in down) when `pre` is true.
template <int K, int AR, int S_, int KPS_>
struct D41xMcWarp {
    static constexpr int MT = AR > 16 ? 2 : 1, S = S_, KPS = KPS_;
    static constexpr int TB = 32 * K;                       // bytes of one 16x16 tile
    static constexpr int SB = KPS * 2 * TB;                 // bytes of one stage (KPS k16 steps x 2 tiles)
    static constexpr int NCH = SB / 16;                     // 16-byte chunks of a stage
    static constexpr int CPL = (NCH + 31) / 32;             // chunks per lane
    const unsigned char* T; long long kstride; unsigned ring; int lane;
    __device__ __forceinline__ void issue(int st, int slot) const {    // stage st (k16 steps KPS st ..) into ring slot
        const unsigned dst = ring + (unsigned)(slot * SB);
        const unsigned long long pol = d41x_pol_ef();
        #pragma unroll
        for (int i = 0; i < CPL; ++i) {
            const int c = lane + 32 * i;
            if ((NCH % 32) == 0 || c < NCH) {
                const int ks = c / (2 * TB / 16), cc = c - ks * (2 * TB / 16);
                d41x_cp16_ef(dst + (unsigned)(ks * 2 * TB + cc * 16), T + (long long)(KPS * st + ks) * kstride + cc * 16, pol);
            }
        }
    }
};

// The f32 running sums of a warp's 2 tiles (K part 0 of gate/up; down), kept in SHARED memory, not registers: the loop already
// holds the chain accumulators, and 16 more registers a thread spill at 128 (4 CTAs per SM). One quad (m, t, h) of a lane is
// one float4 at [warp][quad][lane] (conflict-free 128-bit accesses). The adds are the same f32 adds in the same order.
template <int AR>
struct D41xMcSum {
    static constexpr int MT = AR > 16 ? 2 : 1, NQ = 4 * MT;
    static constexpr int BYTES = 4 * NQ * 32 * 16;          // per CTA (4 warps)
    float4* p;                                              // this lane's quad 0
    __device__ __forceinline__ void init(unsigned char* base, int warp, int lane) { p = reinterpret_cast<float4*>(base) + warp * NQ * 32 + lane; }
    __device__ __forceinline__ void zero() const {
        #pragma unroll
        for (int q = 0; q < NQ; ++q) p[q * 32] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    __device__ __forceinline__ void add(const float (&acc)[MT][2][2][4]) const {
        #pragma unroll
        for (int q = 0; q < NQ; ++q) {
            float4 v = p[q * 32];
            const float* c = acc[q >> 2][(q >> 1) & 1][q & 1];
            v.x = __fadd_rn(v.x, c[0]); v.y = __fadd_rn(v.y, c[1]); v.z = __fadd_rn(v.z, c[2]); v.w = __fadd_rn(v.w, c[3]);
            p[q * 32] = v;
        }
    }
    __device__ __forceinline__ float4 get(int m, int t, int h) const { return p[(((m * 2) + t) * 2 + h) * 32]; }
};

template <>
struct D41xMcSum<8> {
    static constexpr int MT = 1, NQ = 4;
    static constexpr int BYTES = 4 * NQ * 32 * 8;           // AR = 8: rows 8..15 do not exist, so float2 halves smem & LDS/STS
    float2* p;
    __device__ __forceinline__ void init(unsigned char* base, int warp, int lane) { p = reinterpret_cast<float2*>(base) + warp * NQ * 32 + lane; }
    __device__ __forceinline__ void zero() const {
        #pragma unroll
        for (int q = 0; q < NQ; ++q) p[q * 32] = make_float2(0.f, 0.f);
    }
    __device__ __forceinline__ void add(const float (&acc)[1][2][2][4]) const {
        #pragma unroll
        for (int q = 0; q < NQ; ++q) {
            float2 v = p[q * 32];
            const float* c = acc[0][(q >> 1) & 1][q & 1];
            v.x = __fadd_rn(v.x, c[0]); v.y = __fadd_rn(v.y, c[1]);
            p[q * 32] = v;
        }
    }
    __device__ __forceinline__ float4 get(int, int t, int h) const {
        const float2 v = p[(t * 2 + h) * 32];
        return make_float4(v.x, v.y, 0.f, 0.f);
    }
};

__device__ __forceinline__ unsigned d41x_mc_lda(const unsigned* p) {
    unsigned v;
    asm volatile("ld.global.ca.b32 %0, [%1];\n" : "=r"(v) : "l"(p));
    return v;
}

// A fragment registers of one k16 step for MT m tiles: rows r0 = 16 m + gid (a0 / a2) and r0 + 8 (a1 / a3, AR > 8 only).
// P7K1: L1-cached `ld.global.ca` shares the 128-B cache line across all 4 warps of the CTA and all 4 k16 steps of the line
// (16x fewer L2 requests vs `__ldcg`), coherent with `griddepcontrol.wait` in gu and `ld.acquire.gpu` on ready[s] in dn.
template <int AR>
__device__ __forceinline__ void d41x_mc_aload(const __half* X, int ldx, int cnt, int k, int lane, unsigned (&an)[AR > 16 ? 2 : 1][4]) {
    constexpr int MT = AR > 16 ? 2 : 1;
    const int gid = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int m = 0; m < MT; ++m) {
        const int r0 = 16 * m + gid, r1 = r0 + 8;
        an[m][0] = an[m][1] = an[m][2] = an[m][3] = 0u;
        if (r0 < cnt) { const __half* p = X + (size_t)r0 * ldx + k + 2 * tig; an[m][0] = d41x_mc_lda((const unsigned*)p); an[m][2] = d41x_mc_lda((const unsigned*)(p + 8)); }
        if (AR > 8 && r1 < cnt) { const __half* p = X + (size_t)r1 * ldx + k + 2 * tig; an[m][1] = d41x_mc_lda((const unsigned*)p); an[m][3] = d41x_mc_lda((const unsigned*)(p + 8)); }
    }
}

// The streaming loop: k-blocks [0, nkb) of the warp's 2 tiles; after each k-block the chain goes to `blk(kb, acc)`.
// Stages 0 .. S-2 must already be issued (and committed one group each).
template <typename WT, int K, int AR, typename BLK>
__device__ __forceinline__ void d41x_mc_stream(const WT& W, const __half* X, int ldx, int cnt, int nkb, BLK&& blk) {
    constexpr int MT = AR > 16 ? 2 : 1, S = WT::S, KPS = WT::KPS, SPB = D41X_KBLK / KPS;   // SPB stages per 128-k block
    static_assert(D41X_KBLK % KPS == 0, "a stage never straddles two accumulation blocks");
    const int lane = W.lane;
#ifndef D41X_MC_Q96
#define D41X_MC_Q96 1
#endif
    D41xDiet<K, (D41X_MC_Q96 && K > 4)> D; D.init(lane);   // PK3's Q96 field extraction for K > 4: the same fields (exhaustive test)
    const int nst = nkb * SPB;
    extern __shared__ __align__(128) unsigned char d41x_mc_smem[];
    const unsigned char* ringp = d41x_mc_smem + (W.ring - (unsigned)__cvta_generic_to_shared(d41x_mc_smem));
    unsigned an[MT][4];
    d41x_mc_aload<AR>(X, ldx, cnt, 0, lane, an);
    float acc[MT][2][2][4];
    #pragma unroll
    for (int m = 0; m < MT; ++m)
        #pragma unroll
        for (int t = 0; t < 2; ++t)
            #pragma unroll
            for (int h = 0; h < 2; ++h)
                #pragma unroll
                for (int r = 0; r < 4; ++r) acc[m][t][h][r] = 0.f;
    int slot = 0;
    for (int st = 0; st < nst; ++st) {
        if (st + S - 1 < nst) { int ws = slot + S - 1; if (ws >= S) ws -= S; W.issue(st + S - 1, ws); }
        d41_cp_commit();
        d41_cp_wait<S - 1>();
        __syncwarp();
        const unsigned char* sb = ringp + slot * WT::SB;
        #pragma unroll 4
        for (int ks = 0; ks < KPS; ++ks) {
            unsigned a[MT][4];
            #pragma unroll
            for (int m = 0; m < MT; ++m)
                #pragma unroll
                for (int r = 0; r < 4; ++r) a[m][r] = an[m][r];
            const int kn = (st * KPS + ks + 1) * 16;
            if (kn < nst * KPS * 16) d41x_mc_aload<AR>(X, ldx, cnt, kn, lane, an);
            #pragma unroll
            for (int t = 0; t < 2; ++t) {
                unsigned bf[2][2];
                D.decode((const uint32_t*)(sb + ks * 2 * WT::TB + t * WT::TB), bf);
                #pragma unroll
                for (int m = 0; m < MT; ++m) {
                    d41x_mma(acc[m][t][0], a[m][0], a[m][1], a[m][2], a[m][3], bf[0][0], bf[0][1]);
                    d41x_mma(acc[m][t][1], a[m][0], a[m][1], a[m][2], a[m][3], bf[1][0], bf[1][1]);
                }
            }
        }
        __syncwarp();                                    // every lane read the stage before a later issue refills it
        if (++slot == S) slot = 0;
        if (st % SPB == SPB - 1) {                       // a 128-k block (8 k16 steps) is complete
            blk(st / SPB, acc);
            #pragma unroll
            for (int m = 0; m < MT; ++m)
                #pragma unroll
                for (int t = 0; t < 2; ++t)
                    #pragma unroll
                    for (int h = 0; h < 2; ++h)
                        #pragma unroll
                        for (int r = 0; r < 4; ++r) acc[m][t][h][r] = 0.f;
        }
    }
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}

// ---- prep -------------------------------------------------------------------------------------------------------------
// CTA 0 = epoch + routing table + compact non-empty (slot, tile) list in R.s_blk[0..R.q); CTAs >= 1: each CTA item covers
// one id-slot y and 8 128-blocks (both w1 and w3 from one x load, warp 0 computing y's pair index once for all 8 warps).
extern "C" __global__ void __launch_bounds__(256) d41x_mc_prep(D41xMcArgs m) {
    D41_PDL_RELEASE();                                   // the gate/up grid may launch now (its CTAs wait for this one)
    D41_PDL_WAIT_ONLY();                                 // ids and x come from the router / the hc kernel
    const D41xMdArgs& a = m.a;
    if (blockIdx.x == 0) {
        extern __shared__ __align__(128) unsigned char d41x_mcp_smem[];
        D41xMdRoute& R = *reinterpret_cast<D41xMdRoute*>(d41x_mcp_smem);
        if (threadIdx.x == 0) atomicAdd(d41x_md_rdy(a) + 2 * D41X_MD_MAXS, 1u);
        d41x_md_route(a, R);                             // 15_*'s compaction (ends with __syncthreads())
        if (threadIdx.x < 32) {
            const int lane = threadIdx.x;
            const int ns = min(R.n_slots, m.nsl);
            const int ps = (ns + 31) >> 5, s0 = lane * ps;
            int nt = 0;
            for (int i = 0; i < ps; ++i) {
                const int s = s0 + i;
                if (s < ns) nt += m.nmt > 1 ? ((R.s_cnt[s] + 15) >> 4) : 1;
            }
            int pfx = nt;
            #pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int x = __shfl_up_sync(0xFFFFFFFFu, pfx, o);
                if (lane >= o) pfx += x;
            }
            int pos = pfx - nt;
            for (int i = 0; i < ps; ++i) {
                const int s = s0 + i;
                if (s < ns) {
                    const int kt = m.nmt > 1 ? ((R.s_cnt[s] + 15) >> 4) : 1;
                    for (int tl = 0; tl < kt; ++tl) R.s_blk[pos++] = (tl << 16) | s;
                }
            }
            if (lane == 31) R.q = pfx;
        }
        __syncthreads();
        const int nw = d41x_md_route_bytes(a.n_experts) / 4;
        const unsigned* src = (const unsigned*)d41x_mcp_smem;
        unsigned* dst = (unsigned*)m.route;
        for (int i = threadIdx.x; i < nw; i += blockDim.x) dst[i] = src[i];
        return;
    }
    if (gridDim.x <= 1) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int KB = a.dim >> 7;
    const int P = a.rows * a.topk, ny = P + (a.sh ? a.rows : 0);
    if (a.rows <= 4) {
        // <= 4 rows: P <= 32 (one ballot/shuffle pass, no loop) and the whole grid fits in one wave on 48 SMs:
        // keep 1 (y, blk, mat) per warp with zero CTA barriers so w1 and w3 transforms run concurrently.
        for (int gw = ((int)blockIdx.x - 1) * 8 + warp; gw < ny * KB * 2; gw += ((int)gridDim.x - 1) * 8) {
            const int y = gw / (KB * 2), rem = gw - y * KB * 2, blk = rem >> 1, mat = rem & 1;
            const D41xExpert* E; int pair, row;
            if (y < P) {
                row = y / a.topk;
                const int e = __ldcg(a.ids + y);
                if (e < 0 || e >= a.n_experts || a.ex[e].nblk <= 0) continue;
                int c = 0;
                for (int p = lane; p < P; p += 32) {
                    const int e2 = __ldcg(a.ids + p);
                    const bool held = e2 >= 0 && e2 < a.n_experts && a.ex[e2].nblk > 0;
                    c += held && (e2 < e || (e2 == e && p / a.topk < row));
                }
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) c += __shfl_xor_sync(0xFFFFFFFFu, c, o);
                pair = (a.sh ? a.rows : 0) + c; E = a.ex + e;
            } else {
                row = y - P; pair = row; E = a.sh;
            }
            float v[4], sv[4];
            #pragma unroll
            for (int r = 0; r < 4; ++r) { v[r] = d41x_ld(a.x, a.xcode, (size_t)row * a.ldx + blk * 128 + lane * 4 + r); sv[r] = __half2float(E->suh[mat][blk * 128 + lane * 4 + r]); }
            d41x_prep4(v, sv, lane);
            __half o[4];
            #pragma unroll
            for (int r = 0; r < 4; ++r) o[r] = __float2half_rn(v[r]);
            *(uint2*)(a.xh1 + ((size_t)pair * 2 + mat) * a.dim + blk * 128 + lane * 4) = *(const uint2*)o;
        }
        return;
    }
    __shared__ int s_pair, s_e, s_row;
    const int BG = (KB + 7) >> 3;
    const int nitems = ny * BG;
    int prev_y = -1;
    for (int idx = (int)blockIdx.x - 1; idx < nitems; idx += (int)gridDim.x - 1) {
        const int y = idx % ny, blk = (idx / ny) * 8 + warp;
        if (y != prev_y) {
            if (prev_y >= 0) __syncthreads();
            if (warp == 0) {
                int pair = -1, e_id = -2, row = 0;
                if (y < P) {
                    row = y / a.topk;
                    const int e = __ldcg(a.ids + y);
                    if (e >= 0 && e < a.n_experts && a.ex[e].nblk > 0) {
                        int c = 0;
                        for (int p = lane; p < P; p += 32) {
                            const int e2 = __ldcg(a.ids + p);
                            const bool held = e2 >= 0 && e2 < a.n_experts && a.ex[e2].nblk > 0;
                            c += held && (e2 < e || (e2 == e && p / a.topk < row));
                        }
                        #pragma unroll
                        for (int o = 16; o > 0; o >>= 1) c += __shfl_xor_sync(0xFFFFFFFFu, c, o);
                        pair = (a.sh ? a.rows : 0) + c; e_id = e;
                    }
                } else {
                    row = y - P; pair = row; e_id = -1;
                }
                if (lane == 0) { s_pair = pair; s_e = e_id; s_row = row; }
            }
            __syncthreads();
            prev_y = y;
        }
        if (s_pair < 0 || blk >= KB) continue;
        const D41xExpert* E = s_e < 0 ? a.sh : (a.ex + s_e);
        const int pair = s_pair, row = s_row;
        float vx[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r) vx[r] = d41x_ld(a.x, a.xcode, (size_t)row * a.ldx + blk * 128 + lane * 4 + r);
        #pragma unroll
        for (int mat = 0; mat < 2; ++mat) {
            float v[4] = {vx[0], vx[1], vx[2], vx[3]}, sv[4];
            #pragma unroll
            for (int r = 0; r < 4; ++r) sv[r] = __half2float(E->suh[mat][blk * 128 + lane * 4 + r]);
            d41x_prep4(v, sv, lane);
            __half o[4];
            #pragma unroll
            for (int r = 0; r < 4; ++r) o[r] = __float2half_rn(v[r]);
            *(uint2*)(a.xh1 + ((size_t)pair * 2 + mat) * a.dim + blk * 128 + lane * 4) = *(const uint2*)o;
        }
    }
}

// ---- the act epilogue of (slot s, block b) with 4 warps: graw (every part folded) -> xh2 (15_*'s d41x_md_act math) ----
__device__ __forceinline__ void d41x_mc_act(const D41xMdArgs& a, const D41xExpert& E, int e, unsigned msk, int cnt, int off, int b) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const bool shx = e < 0;
    const float lim = shx ? a.limit_sh : a.limit;
    if (warp >= cnt) return;
    const int c0 = b * 128 + lane * 4;
    const __half *sv0 = E.svh[0], *sv1 = E.svh[1], *su2 = E.suh[2];
    float s0[4], s1[4], s2[4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) { s0[r] = __half2float(sv0[c0 + r]); s1[r] = __half2float(sv1[c0 + r]); s2[r] = __half2float(su2[c0 + r]); }
    for (int j = warp; j < cnt; j += D41X_MC_THREADS / 32) {
        const int pair = off + j;
        const float4 g4 = __ldcg((const float4*)(a.graw + ((size_t)pair * 2 + 0) * a.inter_ld + c0));
        const float4 u4 = __ldcg((const float4*)(a.graw + ((size_t)pair * 2 + 1) * a.inter_ld + c0));
        float w = 1.f;
        if (!shx) {
            const int row = d41x_nth_bit(msk, j);
            int id = -1; float wk = 0.f;
            if (lane < a.topk) { id = __ldcg(a.ids + row * a.topk + lane); wk = __ldcg(a.wts + row * a.topk + lane); }
            const unsigned hit = __ballot_sync(0xFFFFFFFFu, id == e);
            w = hit ? __shfl_sync(0xFFFFFFFFu, wk, __ffs(hit) - 1) : 0.f;
        }
        float g[4] = {g4.x, g4.y, g4.z, g4.w}, u[4] = {u4.x, u4.y, u4.z, u4.w}, v[4];
        d41x_had128(g, lane); d41x_had128(u, lane);
        #pragma unroll
        for (int r = 0; r < 4; ++r) v[r] = d41x_act1(__fmul_rn(g[r], s0[r]), __fmul_rn(u[r], s1[r]), w, s2[r], lim);
        d41x_had128(v, lane);
        __half o[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r) o[r] = __float2half_rn(v[r]);
        *(uint2*)(a.xh2 + (size_t)pair * a.inter_ld + c0) = *(const uint2*)o;
    }
}

// ---- the combine of output block jb over every row, 4 warps, routing from the scratch table (15_*'s d41x_md_combine) ----
template <int CB>
__device__ __forceinline__ void d41x_mc_combine(const D41xMdArgs& a, D41xMdRoute* G, int jb) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int c0 = jb * 128 + lane * 4;
    const unsigned* mask = d41x_md_mask(*G);
    const short* slot_of = d41x_md_slot_of(*G, a.n_experts);
    for (int r = warp; r < a.rows; r += D41X_MC_THREADS / 32) {
        int e = 0x7FFFFFFF;
        unsigned me = 0u;
        if (lane < a.topk) {
            const int x = __ldcg(a.ids + r * a.topk + lane);
            if (x >= 0 && x < a.n_experts) {
                me = __ldcg(mask + x);
                if (d41x_mc_lds(slot_of + x) >= 0 && ((me >> r) & 1u)) e = x;
            }
        }
        for (int j = 0; j < a.topk; ++j) { const int ej = __shfl_sync(0xFFFFFFFFu, e, j); if (j < lane && ej == e) e = 0x7FFFFFFF; }
        const bool val = e != 0x7FFFFFFF;
        const __half* esv = val ? a.ex[e].svh[2] : nullptr;
        const int poff = val ? d41x_mc_lds(G->s_off + d41x_mc_lds(slot_of + e)) + __popc(me & ((1u << r) - 1u)) : 0;
        int rank = 0;
        for (int j = 0; j < a.topk; ++j) { const int ej = __shfl_sync(0xFFFFFFFFu, e, j); rank += ej < e; }
        const int nv = __popc(__ballot_sync(0xFFFFFFFFu, val));
        float* o = a.out + (size_t)r * a.ldo + c0;
        float yv[4];
        #pragma unroll
        for (int q = 0; q < 4; ++q) yv[q] = a.ow ? 0.f : __ldcg(o + q);
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        for (int i0 = 0; i0 < nv; i0 += CB) {
            float4 v[CB]; uint2 sv[CB];
            #pragma unroll
            for (int i = 0; i < CB; ++i) {
                if (i0 + i >= nv) break;
                const int src = __ffs(__ballot_sync(0xFFFFFFFFu, val && rank == i0 + i)) - 1;
                const int pair = __shfl_sync(0xFFFFFFFFu, poff, src);
                const __half* svp = (const __half*)__shfl_sync(0xFFFFFFFFu, (unsigned long long)esv, src);
                v[i] = __ldcg((const float4*)(a.h2 + (size_t)pair * a.dim + c0));
                sv[i] = *(const uint2*)(svp + c0);
            }
            #pragma unroll
            for (int i = 0; i < CB; ++i) {
                if (i0 + i >= nv) break;
                const float vv[4] = {v[i].x, v[i].y, v[i].z, v[i].w};
                const __half* hh = reinterpret_cast<const __half*>(&sv[i]);
                #pragma unroll
                for (int q = 0; q < 4; ++q) acc[q] = __fmaf_rn(vv[q], __half2float(hh[q]), acc[q]);
            }
        }
        float ov[4];
        #pragma unroll
        for (int q = 0; q < 4; ++q) ov[q] = __fadd_rn(yv[q], acc[q]);
        if (a.sh) {
            const float4 v4 = __ldcg((const float4*)(a.h2 + (size_t)r * a.dim + c0));
            const float v[4] = {v4.x, v4.y, v4.z, v4.w};
            #pragma unroll
            for (int q = 0; q < 4; ++q) ov[q] = __fadd_rn(ov[q], __fmaf_rn(v[q], __half2float(a.sh->svh[2][c0 + q]), 0.f));
        }
        *(float4*)o = make_float4(ov[0], ov[1], ov[2], ov[3]);
    }
}

__device__ __forceinline__ void d41x_mc_discard(const void* p) {
    if (((size_t)p & 127u) == 0u) asm volatile("discard.global.L2 [%0], 128;" :: "l"(p) : "memory");
}

// ---- gate/up ------------------------------------------------------------------------------------------------------------
// a CTA's item, kept in shared memory across the streaming loop (registers are the loop's: 128 a thread, no spill)
struct D41xMcItem { int s, by, e, nb, cnt, off, mat, kb0; };

template <int K, int AR, bool PRE>
__device__ __forceinline__ void d41x_mc_gu_k(const D41xMcArgs& m, const D41xExpert& E, int mat, int by, int kb0, int nkb,
                                             int cnt, int off, int aux) {
    const D41xMdArgs& a = m.a;
    constexpr int MT = AR > 16 ? 2 : 1;
    extern __shared__ __align__(128) unsigned char d41x_mc_smem[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    typedef D41xMcWarp<K, AR, d41x_mc_stages(K, AR), d41x_mc_kps(K, AR)> WT;
    WT W;
    W.lane = lane;
    W.kstride = (long long)E.nbs13 * 32 * K;
    W.T = (const unsigned char*)E.tr[mat] + (long long)(8 * by + 2 * warp) * (32 * K) + (long long)kb0 * D41X_KBLK * W.kstride;
    W.ring = (unsigned)__cvta_generic_to_shared(d41x_mc_smem) + (unsigned)(D41xMcSum<AR>::BYTES + warp * WT::S * WT::SB);
    const int nst = nkb * (D41X_KBLK / WT::KPS);
    #pragma unroll
    for (int d = 0; d < WT::S - 1; ++d) { if (d < nst) W.issue(d, d); d41_cp_commit(); }
    const __half* X = a.xh1 + ((size_t)off * 2 + mat) * a.dim + kb0 * 128;
    const int col0 = (8 * by + 2 * warp) * 16;
    D41xMcSum<AR> sum; sum.init(d41x_mc_smem, warp, lane);
    if (PRE) sum.zero();
    const size_t cs = PRE ? 0 : (size_t)aux * 2 * a.inter_ld;   // chain plane stride (aux = npairs when !PRE, lin when PRE)
    d41x_mc_stream<WT, K, AR>(W, X, 2 * a.dim, cnt, nkb, [&](int kb, float (&acc)[MT][2][2][4]) {
        if (PRE) {                                       // K part 0: the prefix fold (sum = 0; sum += chain, f32)
            sum.add(acc);
        } else {                                         // K parts > 0: each k-block's chain, folded by the last arrival
            float* cb = a.chain + (size_t)(kb0 + kb) * cs;
            #pragma unroll
            for (int i = 0; i < MT; ++i)
                #pragma unroll
                for (int t = 0; t < 2; ++t)
                    #pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int r0 = 16 * i + gid, r1 = r0 + 8, col = col0 + 16 * t + 8 * h + 2 * tig;
                        if (r0 < cnt) *(float2*)(cb + ((size_t)(off + r0) * 2 + mat) * a.inter_ld + col) = make_float2(acc[i][t][h][0], acc[i][t][h][1]);
                        if (AR > 8 && r1 < cnt) *(float2*)(cb + ((size_t)(off + r1) * 2 + mat) * a.inter_ld + col) = make_float2(acc[i][t][h][2], acc[i][t][h][3]);
                    }
        }
    });
    if (PRE) {
        const int mt = ((aux % (m.nby * 2 * m.sk)) / m.nby) / m.sk;   // = mat, recomputed (keeps it out of the loop's registers)
        #pragma unroll
        for (int i = 0; i < MT; ++i)
            #pragma unroll
            for (int t = 0; t < 2; ++t)
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int r0 = 16 * i + gid, r1 = r0 + 8, col = col0 + 16 * t + 8 * h + 2 * tig;
                    const float4 v = sum.get(i, t, h);
                    if (r0 < cnt) *(float2*)(a.graw + ((size_t)(off + r0) * 2 + mt) * a.inter_ld + col) = make_float2(v.x, v.y);
                    if (AR > 8 && r1 < cnt) *(float2*)(a.graw + ((size_t)(off + r1) * 2 + mt) * a.inter_ld + col) = make_float2(v.z, v.w);
                }
    }
}

template <int AR>
__device__ __forceinline__ void d41x_mc_gu_body(const D41xMcArgs& m) {
    const D41xMdArgs& a = m.a;
    __shared__ int last_sh;
    __shared__ D41xMcItem it_sh;
    D41_PDL_WAIT_ONLY();                                 // the routing table and the rotated rows (prep)
    D41_PDL_RELEASE();                                   // down's CTAs may take the SM slots this grid frees
    D41xMdRoute* G = m.route;
    const int nvt = d41x_mc_ldi(&G->q), npairs = d41x_mc_ldi(&G->n_pairs);
    const int total = nvt * (m.nby * 2 * m.sk);
    for (int lin = blockIdx.x; lin < total; lin += gridDim.x) {
#ifdef D41X_MC_TRACE
        const unsigned long long t_start = d41x_mc_now();
#endif
        // slot-major (slot 0 = the shared expert first), compact non-empty (slot, tile) list in G->s_blk
        {
            const int per = m.nby * 2 * m.sk, KB = a.dim >> 7;
            const int vs = lin / per, rem = lin - vs * per, by = rem % m.nby, bz = rem / m.nby, part = bz % m.sk, mat = bz / m.sk;
            const int vpack = d41x_mc_ldi(G->s_blk + vs);
            const int s = vpack & 0xFFFF, tl = vpack >> 16;
            const int e = d41x_mc_lds(G->s_e + s), nb = d41x_mc_lds(G->s_nb + s);
            if (by >= nb) continue;
            const int cnt = d41x_mc_lds(G->s_cnt + s), off = d41x_mc_lds(G->s_off + s);   // the slot's rows (every tile)
            const int tcnt = AR > 16 ? cnt : min(16, cnt - 16 * tl), toff = off + 16 * tl;
            const D41xExpert& E = e < 0 ? *a.sh : a.ex[e];
            const int kb0 = part * KB / m.sk, nkb = (part + 1) * KB / m.sk - kb0;
            if (threadIdx.x == 0) it_sh = D41xMcItem{s, by, e, nb, cnt, off, mat, kb0};   // the SLOT's rows: the last arrival covers every tile
#define D41X_MC_GU_CASE(KK) case KK: if (kb0 == 0) d41x_mc_gu_k<KK, AR, true>(m, E, mat, by, kb0, nkb, tcnt, toff, lin);   \
                                     else d41x_mc_gu_k<KK, AR, false>(m, E, mat, by, kb0, nkb, tcnt, toff, npairs); break;
            switch (mat ? E.k3 : E.k1) {
                D41X_MC_GU_CASE(2) D41X_MC_GU_CASE(3) D41X_MC_GU_CASE(4) D41X_MC_GU_CASE(5) D41X_MC_GU_CASE(6) D41X_MC_GU_CASE(7) D41X_MC_GU_CASE(8)
                default: __trap();
            }
#undef D41X_MC_GU_CASE
        }
        // the (slot, block)'s arrivals: 2 matrices x sk parts; the last folds the parts, runs the act, publishes readiness
        __syncthreads();                                 // every thread's graw / chain stores precede thread 0's release
        const D41xMcItem I = it_sh;
        const int s = I.s, by = I.by, e = I.e, nb = I.nb, cnt = I.cnt, off = I.off;
#ifdef D41X_MC_TRACE
        D41X_MC_T(lin, 0, t_start); D41X_MC_T(lin, 1, d41x_mc_now()); D41X_MC_T(lin, 3, ((unsigned long long)s << 12) | (unsigned)by);
#endif
        int* ctr = a.cnt + s * a.ncb + by;
        const int kt = m.nmt > 1 ? ((cnt + 15) >> 4) : 1;
        if (threadIdx.x == 0) last_sh = (int)(d41x_add_acqrel((unsigned*)ctr, 1u) == (unsigned)(2 * m.sk * kt - 1));
        __syncthreads();
        if (last_sh) {
            const int t = threadIdx.x, KB = a.dim >> 7;
            if (m.sk > 1) {                              // continue K part 0's fold over the stored chains, ascending k-block
                const size_t cs = (size_t)npairs * 2 * a.inter_ld;
                const int e0 = KB / m.sk;
                for (int q = t; q < cnt * 2 * 32; q += D41X_MC_THREADS) {      // (row, mat, 4-column group)
                    const int j = q / 64, mt = (q >> 5) & 1, c4 = (q & 31) * 4;
                    float* g = a.graw + ((size_t)(off + j) * 2 + mt) * a.inter_ld + by * 128 + c4;
                    const float* c = a.chain + ((size_t)(off + j) * 2 + mt) * a.inter_ld + by * 128 + c4;
                    float4 f = __ldcg((const float4*)g);
                    for (int k0 = e0; k0 < KB; k0 += 8) {
                        float4 v[8];
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) if (k0 + i < KB) v[i] = __ldcg((const float4*)(c + (size_t)(k0 + i) * cs));
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) if (k0 + i < KB) { f.x = __fadd_rn(f.x, v[i].x); f.y = __fadd_rn(f.y, v[i].y); f.z = __fadd_rn(f.z, v[i].z); f.w = __fadd_rn(f.w, v[i].w); }
                    }
                    *(float4*)g = f;
                }
                __syncthreads();
                // the chains are dead: drop their L2 lines unwritten (rows x 2 mats x 4 lines per k-block)
                for (int q = t; q < (KB - e0) * cnt * 8; q += D41X_MC_THREADS) {
                    const int kb = e0 + q / (cnt * 8), r2 = q % (cnt * 8), j = r2 >> 3, mt = (r2 >> 2) & 1, l = r2 & 3;
                    d41x_mc_discard(a.chain + (size_t)kb * cs + ((size_t)(off + j) * 2 + mt) * a.inter_ld + by * 128 + l * 32);
                }
            }
            const D41xExpert& E = e < 0 ? *a.sh : a.ex[e];
            const unsigned msk = e < 0 ? 0u : __ldcg(d41x_md_mask(*G) + e);
            d41x_mc_act(a, E, e, msk, cnt, off, by);
            if (t == 0) *ctr = 0;
            __syncthreads();                             // every thread's xh2 stores precede the release + graw discard below
            for (int q = t; q < cnt * 8; q += D41X_MC_THREADS) {
                const int j = q >> 3, mt = (q >> 2) & 1, l = q & 3;
                d41x_mc_discard(a.graw + ((size_t)(off + j) * 2 + mt) * a.inter_ld + by * 128 + l * 32);
            }
            if (t == 0) {
                unsigned* rdy = d41x_md_rdy(a);
                const unsigned old = d41x_add_acqrel(rdy + s, 1u);
                if (old == (unsigned)nb - 1u) {
                    rdy[s] = 0u;
                    d41x_st_rel(rdy + D41X_MD_MAXS + s, __ldcg(rdy + 2 * D41X_MD_MAXS));
#ifdef D41X_MC_TRACE
                    d41x_mc_rtrace[s] = d41x_mc_now();
#endif
                }
            }
        }
#ifdef D41X_MC_TRACE
        D41X_MC_T(lin, 2, d41x_mc_now());
#endif
    }
}

// ---- down + combine -------------------------------------------------------------------------------------------------------
template <int K, int AR>
__device__ __forceinline__ void d41x_mc_dn_k(const D41xMcArgs& m, const D41xExpert& E, int s, int j, int cnt, int off, int nb, unsigned ep,
                                             int trace_idx) {
    const D41xMdArgs& a = m.a;
    constexpr int MT = AR > 16 ? 2 : 1;
    extern __shared__ __align__(128) unsigned char d41x_mc_smem[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    typedef D41xMcWarp<K, AR, d41x_mc_dstages(K, AR), d41x_mc_dkps(K, AR)> WT;
    WT W;
    W.lane = lane;
    W.kstride = (long long)(a.dim / 16) * 32 * K;
    W.T = (const unsigned char*)E.tr[2] + (long long)(8 * j + 2 * warp) * (32 * K);
    W.ring = (unsigned)__cvta_generic_to_shared(d41x_mc_smem) + (unsigned)(D41xMcSum<AR>::BYTES + warp * WT::S * WT::SB);
    const int nst = nb * (D41X_KBLK / WT::KPS);
    // the weights do not depend on gate/up: the first stages go out before the wait
    #pragma unroll
    for (int d = 0; d < WT::S - 1; ++d) { if (d < nst) W.issue(d, d); d41_cp_commit(); }
    {   // this slot's rows only (the rest of the gate/up grid may still run); every lane acquires for itself
        const unsigned* f = d41x_md_rdy(a) + D41X_MD_MAXS + s;
        if (lane == 0) while (d41x_ld_acq(f) != ep) __nanosleep(128);
        __syncwarp();
        (void)d41x_ld_acq(f);
    }
#ifdef D41X_MC_TRACE
    if (warp == 0) D41X_MC_T(trace_idx, 1, d41x_mc_now());
#else
    (void)trace_idx;
#endif
    D41xMcSum<AR> sum; sum.init(d41x_mc_smem, warp, lane);
    sum.zero();
    d41x_mc_stream<WT, K, AR>(W, a.xh2 + (size_t)off * a.inter_ld, a.inter_ld, cnt, nb, [&](int, float (&acc)[MT][2][2][4]) { sum.add(acc); });
    __syncthreads();                                     // every warp is done with its ring: it becomes the epilogue stage
    float* stage = reinterpret_cast<float*>(d41x_mc_smem + D41xMcSum<AR>::BYTES);
    #pragma unroll
    for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int t = 0; t < 2; ++t)
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r0 = 16 * i + gid, r1 = r0 + 8, col = warp * 32 + 16 * t + 8 * h + 2 * tig;
                const float4 v = sum.get(i, t, h);
                if (r0 < cnt) *(float2*)(stage + r0 * 132 + col) = make_float2(v.x, v.y);
                if (AR > 8 && r1 < cnt) *(float2*)(stage + r1 * 132 + col) = make_float2(v.z, v.w);
            }
}

template <int AR>
__device__ __forceinline__ void d41x_mc_dn_body(const D41xMcArgs& m) {
    const D41xMdArgs& a = m.a;
    extern __shared__ __align__(128) unsigned char d41x_mc_smem[];
    __shared__ int last_sh;
    __shared__ D41xMcItem it_sh;
    // prep has completed: every gate/up CTA waited for it before it let this grid launch (no CTA of gate/up exits before
    // its wait), so the routing table and the epoch are final; the slots' rows are not (ready flags below)
    D41xMdRoute* G = m.route;
    const int NJ = a.dim >> 7;
    const int total = d41x_mc_ldi(&G->q) * NJ;
    const unsigned ep = __ldcg(d41x_md_rdy(a) + 2 * D41X_MD_MAXS);
    bool worked = false;
    for (int lin = blockIdx.x; lin < total; lin += gridDim.x) {
#ifdef D41X_MC_TRACE
        const unsigned long long t_start = d41x_mc_now();
        const int tix = D41X_MC_TN / 2 + lin;
#else
        const int tix = 0;
#endif
        {
            const int vs = lin / NJ, j = lin - vs * NJ;
            const int vpack = d41x_mc_ldi(G->s_blk + vs);
            const int s = vpack & 0xFFFF, tl = vpack >> 16;
            const int e = d41x_mc_lds(G->s_e + s), nb = d41x_mc_lds(G->s_nb + s);
            const int scnt = d41x_mc_lds(G->s_cnt + s), soff = d41x_mc_lds(G->s_off + s);
            const int cnt = AR > 16 ? scnt : min(16, scnt - 16 * tl), off = soff + 16 * tl;
            const D41xExpert& E = e < 0 ? *a.sh : a.ex[e];
            if (threadIdx.x == 0) it_sh = D41xMcItem{s, j, e, nb, cnt, off, 0, 0};
            switch (E.k2) {
                case 2: d41x_mc_dn_k<2, AR>(m, E, s, j, cnt, off, nb, ep, tix); break;
                case 3: d41x_mc_dn_k<3, AR>(m, E, s, j, cnt, off, nb, ep, tix); break;
                case 4: d41x_mc_dn_k<4, AR>(m, E, s, j, cnt, off, nb, ep, tix); break;
                case 5: d41x_mc_dn_k<5, AR>(m, E, s, j, cnt, off, nb, ep, tix); break;
                case 6: d41x_mc_dn_k<6, AR>(m, E, s, j, cnt, off, nb, ep, tix); break;
                case 7: d41x_mc_dn_k<7, AR>(m, E, s, j, cnt, off, nb, ep, tix); break;
                case 8: d41x_mc_dn_k<8, AR>(m, E, s, j, cnt, off, nb, ep, tix); break;
                default: __trap();
            }
        }
        worked = true;
        __syncthreads();
        const D41xMcItem I = it_sh;
        const int s = I.s, j = I.by, cnt = I.cnt, off = I.off;
        (void)s;
        const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        const float* stage = reinterpret_cast<const float*>(d41x_mc_smem + D41xMcSum<AR>::BYTES);
        for (int r = warp; r < cnt; r += D41X_MC_THREADS / 32) {
            float v[4];
            #pragma unroll
            for (int q = 0; q < 4; ++q) v[q] = stage[r * 132 + lane * 4 + q];
            d41x_had128(v, lane);
            *(float4*)(a.h2 + (size_t)(off + r) * a.dim + j * 128 + lane * 4) = make_float4(v[0], v[1], v[2], v[3]);
        }
        __syncthreads();                                 // h2 stores precede the release; the stage is free again
        const int npairs = d41x_mc_ldi(&G->n_pairs);
        int* ctr = a.cnt + D41X_MD_MAXS * a.ncb + j;       // KM1: counts ROWS (every tile adds its own): the last tile of the block sees npairs
        if (threadIdx.x == 0) last_sh = (int)(d41x_add_acqrel((unsigned*)ctr, (unsigned)cnt) == (unsigned)(npairs - cnt));
        __syncthreads();
        if (last_sh) {
            d41x_mc_combine<3>(a, G, j);
            __syncthreads();
            for (int q = threadIdx.x; q < npairs * 4; q += D41X_MC_THREADS) {
                const int pair = q >> 2, l = q & 3;
                d41x_mc_discard(a.h2 + (size_t)pair * a.dim + j * 128 + l * 32);
            }
            if (threadIdx.x == 0) *ctr = 0;
        }
#ifdef D41X_MC_TRACE
        D41X_MC_T(tix, 0, t_start); D41X_MC_T(tix, 2, d41x_mc_now()); D41X_MC_T(tix, 3, (1ull << 24) | ((unsigned long long)s << 12) | (unsigned)j);
#endif
    }
    if (worked || blockIdx.x == 0) D41_PDL_WAIT_ONLY();  // the stream's PDL chain stays transitive (down done => gate/up done)
}

#define D41X_MC_ENTRY(AR, MINB)                                                                                                   \
    extern "C" __global__ void __launch_bounds__(D41X_MC_THREADS, MINB) d41x_mc_gu_a##AR(D41xMcArgs m) { d41x_mc_gu_body<AR>(m); } \
    extern "C" __global__ void __launch_bounds__(D41X_MC_THREADS, MINB) d41x_mc_dn_a##AR(D41xMcArgs m) { d41x_mc_dn_body<AR>(m); }
D41X_MC_ENTRY(8, 4)
D41X_MC_ENTRY(16, 4)
D41X_MC_ENTRY(32, 3)

// The host reads the geometry once (prepare time): [0] sizeof(D41xMcArgs), [1] / [2] gate/up / down ring bytes per CTA per
// unit of K, [3] sizeof(D41xMdRoute)
extern "C" __global__ void d41x_mc_sizes(int* o) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    o[0] = (int)sizeof(D41xMcArgs); o[1] = 4 * D41X_MC_STAGES * D41X_MC_KPS * 2 * 32; o[2] = 4 * D41X_MC_DSTAGES * D41X_MC_DKPS * 2 * 32;
    o[3] = (int)sizeof(D41xMdRoute);
}
