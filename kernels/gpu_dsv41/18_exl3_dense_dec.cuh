// 18_exl3_dense_dec.cuh (leg PK4): dense EXL3 linears at decode / verify widths (rows <= 32), two launches per call:
//   d41x_dd_prep        xh = fp16(H128(x * suh)) per (segment, row, 128-block of the K slice) (the d41x_prep math, the same
//                       bits), and zeroes the call's block arrival counters (after its PDL wait: the previous call on the
//                       same scratch has completed by then)
//   d41x_dd_a{8,16,24,32}_k{2..8} a persistent STREAM-K trellis GEMM with the output transform fused: the call's work is a line of
//                       UNITS (one unit = KPS k16 steps of one 128-column output block of one segment); P = min(D41X_DD_P,
//                       units) partitions cut that line into equal non-empty runs (shape only: never the row count, never
//                       the grid); a CTA streams its
//                       partitions' units through one cp.async ring (trellis evict_first, activations evict_last) and runs
//                       8 warps x one n-tile each. A (partition, block) pair that covers its whole block finishes the block
//                       itself (H128 . svh, dtype, accumulate, un-aligned trim); otherwise it writes an f32 partial and the
//                       LAST arrival (block counter) sums the block's partials in ascending partition order and finishes it.
// SEGMENTS: one call may carry up to D41X_DD_MAXSEG linears with the same row count and width K (e.g. the per-group wo_a
//   calls of a rank, or wq_a + wkv on the same input); their unit lines are concatenated, so the partitions balance bytes
//   across all of them. A single linear is one segment.
// NUMERICS (rows <= 32 class): per output column, one MMA chain per k-block (2 ring slots = 8 k16 steps, aligned to the
//   slice's 128-blocks; a partition edge inside a k-block splits it into two shorter chains), each chain added in f32
//   (__fadd_rn) into the pair's running sum, ascending k; the block's pairs added in ascending partition order. All of it
//   is fixed by (shape, P), so row i of an N-row call is bit-identical to its 1-row call for N = 1..32 (the MMA
//   computes rows independently; AR / MT only decide which rows are live). Not bit-equal to KE's split-K dense class
//   (12_*) nor to the wide class (17_*): this class replaces 12_* for integer K at rows <= 32.
// PDL: the GEMM issues the TRELLIS of its first ring slots before griddepcontrol.wait (weights are constant), then waits,
//   triggers its dependents, and stages the activations. Without the launch attribute both are no-ops.
#pragma once

#define D41X_DD_KPS 4            // k16 steps per ring slot (= per unit)
#define D41X_DD_THREADS 256
#define D41X_DD_P 96             // partitions of a call's unit line (shape-only constant; 2 per SM on GB10)
#define D41X_DD_MAXSEG 8
#ifdef D41X_DD_TRACE      // tools/dsv41_pk4/dd_bench.cu --trace only: per-CTA globaltimer stamps
__device__ unsigned long long* d41x_dd_trace_buf;
#define D41X_DD_STAMP(i) do { if (threadIdx.x == 0) { unsigned long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); d41x_dd_trace_buf[blockIdx.x * 8 + (i)] = t_; } } while (0)
#else
#define D41X_DD_STAMP(i) do { } while (0)
#endif

struct D41xDdSeg {
    const unsigned char* tr;     // trellis bytes of tile (first k-step, first n-tile) of the slice in the resident tensor
    const void* x;               // input activations of the slice [rows][ldx] (bf16 / f16 / f32 by xcode)
    const __half* suh;           // suh at the slice's first input column
    const __half* svh;           // svh at the first computed column (block-aligned)
    void* out;                   // [rows][ldo] by ocode (0 bf16, 1 f16, 2 f32); accum = add into it
    __half* xh;                  // the rotated input fp16(H128(x * suh)) [rows][xld]: written by d41x_dd_prep, or (pre = 1) by
                                 // the producer of x in its epilogue (PD1: the norm before wq_b, the attention before wo_a, wo_a
                                 // before wo_b), and then the call has no prep for this segment
    const __half* rsuh;          // PD1 rot_out: the NEXT linear's suh at this segment's output column 0 (null = none): the
                                 // finished block's stored values are rotated into the next linear's input (rxh), the
                                 // d41x_dd_prep math on the same bits (the host asks skip 0, whole blocks, no accumulate)
    __half* rxh;                 // rot_out destination [rows][rld] fp16
    long long kstride;           // bytes between consecutive k16 steps of one n-tile (resident row length * tile bytes)
    int nkb, ncb;                // k16 steps of the slice (multiple of 8), computed 128-column blocks
    int skip, ncols;             // computed column c lands at output column c - skip when 0 <= c - skip < ncols
    int ldx, xcode, ldo, ocode, accum;
    int u0, jb0;                 // first unit, first global block index of this segment (prefix sums over segments)
    int p0, np;                  // PD1: the segment's partitions: global partitions [p0, p0 + np), np = min(D41X_DD_P, units of
                                 // the segment), cut exactly as a 1-segment call cuts them (so a grouped call's bits per
                                 // segment equal the segment's own call)
    int xld, pre, rld, pad_;     // xh row stride (elements); 1 = xh is already written (no prep); rxh row stride
};

struct D41xDdArgs {
    D41xDdSeg seg[D41X_DD_MAXSEG];
    float* part;                 // [P + ncb_all][rows][128] f32 partials (pair (p, block jg) at index p + jg)
    int* cnt;                    // [ncb_all] block arrival counters: a persistent zeroed buffer (PD1), re-armed by the last arrival
    int nseg, rows, U, ncb_all;  // segments, rows, units of the call, blocks of the call
    int bits;                    // K (integer 2..8, the same for every segment)
    int slot, ns, wreg;          // ring slot stride (B), slots (3..16), trellis bytes per slot (A rows follow)
    int maxkb;                   // max over segments of nkb / 8 (prep grid)
    int P;                       // partitions of the call = sum of the segments' np; every partition holds >= 1 unit, so a
                                 // block's contributors are consecutive partitions of its segment
};

// 32-bit math: the host checks U * D41X_DD_P < 2^31 per segment (the largest dense slice, the whole head, is 80,800 units)
__device__ __forceinline__ int d41x_dd_start(int p, int U, int P) { return (int)(((unsigned)p * (unsigned)U) / (unsigned)P); }
// the partition holding unit u: the largest p with start(p) <= u (P <= U: no partition is empty)
__device__ __forceinline__ int d41x_dd_pof(int u, int U, int P) { return (int)((((unsigned)u + 1u) * (unsigned)P - 1u) / (unsigned)U); }

// unit cursor: global unit u, its segment, block j inside the segment, slot s inside the block
struct D41xDdCur { int u, ue, p, seg, j, s; };
__device__ __forceinline__ void d41x_dd_locate(const D41xDdArgs& a, D41xDdCur& c) {
    int sg = 0;
    while (sg + 1 < a.nseg && a.seg[sg + 1].u0 <= c.u) ++sg;
    const int S = a.seg[sg].nkb / D41X_DD_KPS, lu = c.u - a.seg[sg].u0;
    c.seg = sg; c.j = lu / S; c.s = lu - c.j * S;
}
// PD1: global partition p -> its segment's unit range [u0, u1) (the segment's own cut: start(q, U_s, np_s))
__device__ __forceinline__ void d41x_dd_prange(const D41xDdArgs& a, int p, int& u0, int& u1) {
    int sg = 0;
    while (sg + 1 < a.nseg && a.seg[sg + 1].p0 <= p) ++sg;
    const D41xDdSeg& G = a.seg[sg];
    const int Us = G.ncb * (G.nkb / D41X_DD_KPS), q = p - G.p0;
    u0 = G.u0 + d41x_dd_start(q, Us, G.np);
    u1 = G.u0 + d41x_dd_start(q + 1, Us, G.np);
}
// next partition of this CTA with work (u = start, ue = end); false when none is left
__device__ __forceinline__ bool d41x_dd_part(const D41xDdArgs& a, D41xDdCur& c, int p) {
    for (; p < a.P; p += gridDim.x) {
        int u0, u1;
        d41x_dd_prange(a, p, u0, u1);
        if (u1 > u0) { c.p = p; c.u = u0; c.ue = u1; d41x_dd_locate(a, c); return true; }
    }
    c.p = a.P; return false;
}
// advance one unit; returns false when the CTA has no unit left
__device__ __forceinline__ bool d41x_dd_next(const D41xDdArgs& a, D41xDdCur& c) {
    if (++c.u == c.ue) return d41x_dd_part(a, c, c.p + gridDim.x);
    if (++c.s == a.seg[c.seg].nkb / D41X_DD_KPS) { c.s = 0; if (++c.j == a.seg[c.seg].ncb) { c.j = 0; ++c.seg; } }
    return true;
}

// cp.async with an L2 policy. ptxas 13.0 (sm_121a) folds a UNIFORM slot base into the LDGSTS address ([R + UR0]) and then
// encodes registers it never wrote (UR0, and the 64-bit memory descriptor in the ODD uniform register UR1):
// CUDA_ERROR_ILLEGAL_INSTRUCTION (KE2 hit the same in 15_*, in its warm-up copy loops). d41x_dd_issue is one non-inlined
// function, so the warm-up loops and the steady state share one compiled copy of the copies (the steady-state form);
// tools/dsv41_ke2/sass_check.py guards it.
__device__ __forceinline__ void d41x_dd_cp16(unsigned dst, const void* src, unsigned long long pol) {
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "l"(pol) : "memory");
}

// one thread's copy jobs of a slot: chunk ch = t + 256 i; W chunks first (k-step kk, 16-B piece cc of the 8-tile run),
// then A chunks (k-step kk, row jr, half hf). Encoded once per call: bit 31 = A, else W.
template <int K>
__device__ __forceinline__ int d41x_dd_job(int ch, int rows) {
    constexpr int CPR = 8 * 32 * K / 16;                 // 16-B pieces per k-step run (8 tiles)
    constexpr int NW = D41X_DD_KPS * CPR;
    if (ch < NW) { const int kk = ch / CPR, cc = ch - kk * CPR; return (kk << 16) | cc; }
    const int ca = ch - NW;
    if (ca >= D41X_DD_KPS * rows * 2) return -1;
    const int hf = ca & 1, rest = ca >> 1, jr = rest % rows, kk = rest / rows;
    return (int)(0x80000000u | (unsigned)(kk << 8) | (unsigned)(jr << 1) | (unsigned)hf);
}
// Every argument by value: a reference to the kernel's parameter struct would make each field a generic load (an L1 / L2
// round trip per slot on the critical path), and a reference to the cursor / jobs would put them on the stack. The W and A
// copies stay in separate branches with a fixed L2 policy each (a per-thread policy select brought the ptxas encoding back).
template <int K>
__device__ __noinline__ void d41x_dd_issue(const unsigned char* wb, long long kstride, const __half* ab, int kloc, int wreg, int rows,
                                           int job0, int job1, int job2, unsigned dst, bool do_w, bool do_a,
                                           unsigned long long pol, unsigned long long pol_a) {
    constexpr int TB = 32 * K;
    const int job[3] = {job0, job1, job2};
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int jb = job[i];
        if (jb == -1) continue;
        if (jb >= 0) {
            if (!do_w) continue;
            const int kk = jb >> 16, cc = jb & 0xFFFF;
            d41x_dd_cp16(dst + (unsigned)(kk * 8 * TB + cc * 16), wb + (long long)kk * kstride + cc * 16, pol);
        } else {
            if (!do_a) continue;
            const int kk = (jb >> 8) & 0xFF, jr = (jb >> 1) & 0x7F, hf = jb & 1;
            d41x_dd_cp16(dst + (unsigned)(wreg + kk * rows * 32 + d41x_arow(jr, hf)), ab + (size_t)jr * kloc + kk * 16 + hf * 8, pol_a);
        }
    }
}
// issue the unit under cursor c into ring slot `dst` (the segment fields are kernel-parameter (constant bank) loads here)
template <int K>
__device__ __forceinline__ void d41x_dd_go(const D41xDdArgs& a, const D41xDdCur& c, const int (&job)[3], unsigned dst, bool do_w, bool do_a,
                                           unsigned long long pol, unsigned long long pol_a) {
    const D41xDdSeg& G = a.seg[c.seg];
    const unsigned char* wb = G.tr + (long long)(D41X_DD_KPS * c.s) * G.kstride + (long long)(8 * c.j) * (32 * K);
    const __half* ab = G.xh + D41X_DD_KPS * 16 * c.s;
    d41x_dd_issue<K>(wb, G.kstride, ab, G.xld, a.wreg, a.rows, job[0], job[1], job[2], dst, do_w, do_a, pol, pol_a);
}

__device__ __forceinline__ void d41x_dd_mma_lo(float& c0, float& c1, unsigned a0, unsigned a2, unsigned b0, unsigned b1) {
    float d2, d3;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%10,%10};\n"
        : "+f"(c0), "+f"(c1), "=f"(d2), "=f"(d3) : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1), "f"(0.f));
    (void)d2; (void)d3;
}

// one ring slot of MMAs: warp w owns n-tile w of the unit's 8-tile run, KPS k-steps
template <int K, int AR>
__device__ __forceinline__ void d41x_dd_consume(const unsigned char* slot, int wreg, int rows, int warp, int lane, const D41xDiet<K, true>& D,
                                                float (&acc)[AR > 16 ? 2 : 1][2][4]) {
    constexpr int TB = 32 * K, MT = AR > 16 ? 2 : 1;
    const int gid = lane >> 2, tig = lane & 3;
    const unsigned char* tb = slot + warp * TB;
    const unsigned char* ab = slot + wreg;
    #pragma unroll
    for (int kk = 0; kk < D41X_DD_KPS; ++kk) {
        unsigned bf[2][2];
        D.decode((const uint32_t*)(tb + kk * 8 * TB), bf);
        const unsigned char* Ab = ab + kk * rows * 32;
        #pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (16 * m + 8 < AR) {
                unsigned a0, a1, a2, a3;
                d41x_afrag<true>(Ab, 16 * m + gid, rows, tig, a0, a1, a2, a3);
                d41x_mma(acc[m][0], a0, a1, a2, a3, bf[0][0], bf[0][1]);
                d41x_mma(acc[m][1], a0, a1, a2, a3, bf[1][0], bf[1][1]);
            } else {
                unsigned a0, a1, a2, a3;
                d41x_afrag<false>(Ab, 16 * m + gid, rows, tig, a0, a1, a2, a3);
                d41x_dd_mma_lo(acc[m][0][0], acc[m][0][1], a0, a2, bf[0][0], bf[0][1]);
                d41x_dd_mma_lo(acc[m][1][0], acc[m][1][1], a0, a2, bf[1][0], bf[1][1]);
            }
        }
    }
}

// finish rows of block (segment G, block j) from 128 f32 values per row: H128, . svh, trim, dtype, accumulate
// PD1: the value d41x_st stores, as a later d41x_ld reads it back (the rot_out epilogue rotates exactly those bits)
__device__ __forceinline__ float d41x_dd_stored(int code, float v) {
    if (code == 0) return __bfloat162float(__float2bfloat16_rn(v));
    if (code == 1) return __half2float(__float2half_rn(v));
    return v;
}
// finish rows of block (segment G, block j) from 128 f32 values per row: H128, . svh, trim, dtype, accumulate; with rot_out
// (G.rsuh) also the next linear's rotated input of this block: fp16(H128(stored * suh_next)), d41x_dd_prep's math on the
// stored bits (a whole warp calls this for one row: the butterflies are warp-wide)
__device__ __forceinline__ void d41x_dd_finish(const D41xDdSeg& G, int j, int row, float (&v)[4], int lane) {
    d41x_had128(v, lane);
    float st[4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int c = j * 128 + lane * 4 + r;
        const int oc = c - G.skip;
        st[r] = 0.f;
        if (oc < 0 || oc >= G.ncols) continue;
        float o = __fmul_rn(v[r], __half2float(G.svh[c]));
        const size_t oi = (size_t)row * G.ldo + oc;
        if (G.accum) o = __fadd_rn(o, d41x_ld(G.out, G.ocode, oi));
        d41x_st(G.out, G.ocode, oi, o);
        st[r] = d41x_dd_stored(G.ocode, o);
    }
    if (G.rsuh != nullptr) {
        float sv[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r) sv[r] = __half2float(G.rsuh[j * 128 + lane * 4 + r]);
        d41x_prep4(st, sv, lane);
        __half o[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r) o[r] = __float2half_rn(st[r]);
        *(uint2*)(G.rxh + (size_t)row * G.rld + j * 128 + lane * 4) = *(const uint2*)o;
    }
}

template <int AR, int K>
__device__ __forceinline__ void d41x_dd_body(const D41xDdArgs& a) {
    constexpr int MT = AR > 16 ? 2 : 1;
    constexpr int NCH = (D41X_DD_KPS * 8 * 32 * K / 16 + D41X_DD_KPS * AR * 2 + D41X_DD_THREADS - 1) / D41X_DD_THREADS;
    static_assert(NCH <= 3, "a thread carries 3 copy jobs per slot");
    extern __shared__ __align__(128) unsigned char d41x_dd_smem[];
    __shared__ int s_flag;
    unsigned char* ring = d41x_dd_smem;
    float* stage = reinterpret_cast<float*>(ring + a.ns * a.slot);       // [16 * MT][132] f32
    const unsigned ring_u = (unsigned)__cvta_generic_to_shared(ring);
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int rows = a.rows, SLOT = a.slot, NS = a.ns, D = NS - 1;
    const unsigned long long pol = d41x_pol_ef();
    unsigned long long pol_a; asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol_a));

    // this CTA's slot count (CTA-uniform)
    int N = 0;
    for (int p = blockIdx.x; p < a.P; p += gridDim.x) { int u0, u1; d41x_dd_prange(a, p, u0, u1); N += u1 - u0; }
    if (N == 0) return;
    D41X_DD_STAMP(0);
    int job[3];
    #pragma unroll
    for (int i = 0; i < 3; ++i) job[i] = i < NCH ? d41x_dd_job<K>(t + D41X_DD_THREADS * i, rows) : -1;

    // producer cursor; pre-wait: the trellis of slots 0..D-1 (one group each)
    D41xDdCur pc; d41x_dd_part(a, pc, blockIdx.x);
    {
        D41xDdCur tc = pc;
        bool live = true;
        for (int n = 0; n < D; ++n) {
            if (n < N && live) { d41x_dd_go<K>(a, tc, job, ring_u + (unsigned)(n * SLOT), true, false, pol, pol_a); live = d41x_dd_next(a, tc); }
            d41_cp_commit();
        }
    }
    D41X_DD_STAMP(1);
    D41_PDL_WAIT_ONLY();
    D41_PDL_RELEASE();
    for (int n = 0; n < D && n < N; ++n) { d41x_dd_go<K>(a, pc, job, ring_u + (unsigned)(n * SLOT), false, true, pol, pol_a); d41x_dd_next(a, pc); }
    d41_cp_commit();

    D41xDiet<K, true> Dg; Dg.init(lane);
    D41xDdCur cc; d41x_dd_part(a, cc, blockIdx.x);
    int pair_s0 = cc.s;                                  // first slot of the current (partition, block) pair
    // the pair's running sum: registers for <= 24 rows (at AR = 24 tile 1 has 8 rows, so acc + sum is 24 regs); for 25-32 rows
    // it lives in this warp's own columns of the epilogue stage (the same f32 adds, starting from 0 at each pair)
    constexpr bool SMS = AR > 24;
    float acc[MT][2][4], sum[SMS ? 1 : MT][2][4];
    bool first = true;                                   // SMS: the next fold is the pair's first (its old sum is 0)
    #pragma unroll
    for (int m = 0; m < MT; ++m)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int r = 0; r < 4; ++r) { acc[m][h][r] = 0.f; if (!SMS) sum[m][h][r] = 0.f; }
    const int gid = lane >> 2, tig = lane & 3;
    int buf = 0, pbuf = D;
    for (int n = 0; n < N; ++n) {
        if (n == 0) asm volatile("cp.async.wait_all;\n" ::: "memory"); else d41x_md_wait(D);
        __syncthreads();
        if (n == 0) D41X_DD_STAMP(2);
        if (n + D < N) { d41x_dd_go<K>(a, pc, job, ring_u + (unsigned)(pbuf * SLOT), true, true, pol, pol_a); d41x_dd_next(a, pc); }
        d41_cp_commit();
        if (++pbuf == NS) pbuf = 0;
        d41x_dd_consume<K, AR>(ring + buf * SLOT, a.wreg, rows, warp, lane, Dg, acc);
        if (++buf == NS) buf = 0;
        const D41xDdSeg& G = a.seg[cc.seg];
        const int S = G.nkb / D41X_DD_KPS;
        const bool pair_end = (cc.s == S - 1) || (cc.u + 1 == cc.ue);
        if ((cc.s & 1) || pair_end) {      // a k-block chain is complete: fold it into the pair's running sum (f32)
            if constexpr (SMS) {
                // this warp's own stage columns; the previous pair's epilogue reads ended before this slot's barrier
                #pragma unroll
                for (int m = 0; m < MT; ++m)
                    #pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        float* p0 = stage + (16 * m + gid) * 132 + warp * 16 + 8 * h + 2 * tig;
                        const float2 o0 = first ? make_float2(0.f, 0.f) : *(const float2*)p0;
                        *(float2*)p0 = make_float2(__fadd_rn(o0.x, acc[m][h][0]), __fadd_rn(o0.y, acc[m][h][1]));
                        if (16 * m + 8 < AR) {
                            float* p1 = p0 + 8 * 132;
                            const float2 o1 = first ? make_float2(0.f, 0.f) : *(const float2*)p1;
                            *(float2*)p1 = make_float2(__fadd_rn(o1.x, acc[m][h][2]), __fadd_rn(o1.y, acc[m][h][3]));
                        }
                        #pragma unroll
                        for (int r = 0; r < 4; ++r) acc[m][h][r] = 0.f;
                    }
                first = false;
            } else {
                #pragma unroll
                for (int m = 0; m < MT; ++m)
                    #pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        sum[m][h][0] = __fadd_rn(sum[m][h][0], acc[m][h][0]); acc[m][h][0] = 0.f;
                        sum[m][h][1] = __fadd_rn(sum[m][h][1], acc[m][h][1]); acc[m][h][1] = 0.f;
                        if (16 * m + 8 < AR) {
                            sum[m][h][2] = __fadd_rn(sum[m][h][2], acc[m][h][2]); acc[m][h][2] = 0.f;
                            sum[m][h][3] = __fadd_rn(sum[m][h][3], acc[m][h][3]); acc[m][h][3] = 0.f;
                        }
                    }
            }
        }
        if (pair_end) {
            D41X_DD_STAMP(3);
            // ---- pair epilogue: stage [row][col] f32 (SMS: already there)
            if constexpr (!SMS) {
                #pragma unroll
                for (int m = 0; m < MT; ++m)
                    #pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int r0 = 16 * m + gid, col = warp * 16 + 8 * h + 2 * tig;
                        *(float2*)(stage + r0 * 132 + col) = make_float2(sum[m][h][0], sum[m][h][1]);
                        if (16 * m + 8 < AR) *(float2*)(stage + (r0 + 8) * 132 + col) = make_float2(sum[m][h][2], sum[m][h][3]);
                    }
            }
            first = true;
            __syncthreads();
            const int jg = G.jb0 + cc.j;
            if (pair_s0 == 0 && cc.s == S - 1) {
                // the pair is the block's only contributor: finish it here
                for (int r = warp; r < rows; r += D41X_DD_THREADS / 32) {
                    const float4 v4 = *(const float4*)(stage + r * 132 + lane * 4);
                    float v[4] = {v4.x, v4.y, v4.z, v4.w};
                    d41x_dd_finish(G, cc.j, r, v, lane);
                }
            } else {
                float* pp = a.part + (size_t)(cc.p + jg) * rows * 128;
                for (int r = warp; r < rows; r += D41X_DD_THREADS / 32)
                    __stcg((float4*)(pp + (size_t)r * 128 + lane * 4), *(const float4*)(stage + r * 132 + lane * 4));
                __syncthreads();
                const int ub = G.u0 + cc.j * S;                       // the block's first unit
                const int Us = G.ncb * S;                             // the segment's own partition cut (PD1)
                const int pf = G.p0 + d41x_dd_pof(ub - G.u0, Us, G.np), pl = G.p0 + d41x_dd_pof(ub - G.u0 + S - 1, Us, G.np);
                if (t == 0) { __threadfence(); s_flag = (atomicAdd(a.cnt + jg, 1) == pl - pf); }
                __syncthreads();
                if (s_flag) {
                    __threadfence();
                    // the block's partials in ascending partition order; loads issued 8 at a time ahead of the (ordered) adds
                    const size_t qs = (size_t)rows * 128;
                    for (int r = warp; r < rows; r += D41X_DD_THREADS / 32) {
                        const float* base = a.part + ((size_t)jg * rows + r) * 128 + lane * 4;
                        float v[4];
                        for (int q0 = pf; q0 <= pl; q0 += 8) {
                            float4 tq[8];
                            #pragma unroll
                            for (int i = 0; i < 8; ++i) if (q0 + i <= pl) tq[i] = __ldcg((const float4*)(base + (size_t)(q0 + i) * qs));
                            #pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                if (q0 + i > pl) break;
                                if (q0 + i == pf) { v[0] = tq[i].x; v[1] = tq[i].y; v[2] = tq[i].z; v[3] = tq[i].w; }
                                else { v[0] = __fadd_rn(v[0], tq[i].x); v[1] = __fadd_rn(v[1], tq[i].y); v[2] = __fadd_rn(v[2], tq[i].z); v[3] = __fadd_rn(v[3], tq[i].w); }
                            }
                        }
                        d41x_dd_finish(G, cc.j, r, v, lane);
                    }
                    if (t == 0) a.cnt[jg] = 0;
                }
            }
            if constexpr (!SMS) {
                #pragma unroll
                for (int m = 0; m < MT; ++m)
                    #pragma unroll
                    for (int h = 0; h < 2; ++h)
                        #pragma unroll
                        for (int r = 0; r < 4; ++r) sum[m][h][r] = 0.f;
            }
        }
        if (n + 1 < N) { d41x_dd_next(a, cc); if (pair_end) pair_s0 = cc.s; }
    }
    D41X_DD_STAMP(4);
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}

// one entry per (rows class, K): a short call runs cold code, so each entry carries ONE body (ncu: 47 % of the warp stalls
// were instruction fetch with all seven K bodies behind one switch). __maxnreg__(120): two CTAs per SM leave 4,096
// registers free, so the NEXT call's 1-warp prep CTAs become resident while this GEMM still runs (PDL), trigger the next
// GEMM's launch, and that GEMM's CTAs take over each SM the moment ours leave: the weight stream never drains between
// consecutive linears (measured: the prep launch cost 3-9 us per call in graph replay when it could not co-reside).
#define D41X_DD_ENTRY(AR, K) \
    extern "C" __global__ void __maxnreg__(120) d41x_dd_a##AR##_k##K(D41xDdArgs a) { d41x_dd_body<AR, K>(a); }
#define D41X_DD_ENTRIES(K) D41X_DD_ENTRY(8, K) D41X_DD_ENTRY(16, K) D41X_DD_ENTRY(24, K) D41X_DD_ENTRY(32, K)
D41X_DD_ENTRIES(2)
D41X_DD_ENTRIES(3)
D41X_DD_ENTRIES(4)
D41X_DD_ENTRIES(5)
D41X_DD_ENTRIES(6)
D41X_DD_ENTRIES(7)
D41X_DD_ENTRIES(8)

// prep: one 1-warp CTA per (128-block, group of 4 rows, segment): few CTAs (at most ~5 fit next to the running GEMM per
// SM: it leaves 4,096 registers), each with its 4 rows' loads in flight together. It triggers its dependents FIRST (the next
// GEMM then launches while this call's predecessor still runs; that GEMM's own griddepcontrol.wait covers this kernel's
// completion, which follows the predecessor's), then waits for the predecessor, zeroes the block counters and transforms.
#define D41X_DD_PREP_ROWS 4
extern "C" __global__ void __maxnreg__(40) d41x_dd_prep(D41xDdArgs a) {      // 1 warp; 40 registers (PD1: 32 spilled 16 B with the per-segment fields): ~3 fit per SM next to the GEMM
    D41_PDL_RELEASE();
    D41_PDL_WAIT_ONLY();
    if ((int)blockIdx.z >= a.nseg) return;
    const D41xDdSeg& G = a.seg[blockIdx.z];
    if (G.pre) return;                                   // PD1: the producer already wrote this segment's rotated rows
    const int lane = threadIdx.x, blk = blockIdx.x, r0 = blockIdx.y * D41X_DD_PREP_ROWS;
    if (blk >= G.nkb / 8 || r0 >= a.rows) return;
    float sv[4], v[D41X_DD_PREP_ROWS][4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) sv[r] = __half2float(G.suh[blk * 128 + lane * 4 + r]);
    #pragma unroll
    for (int q = 0; q < D41X_DD_PREP_ROWS; ++q)
        #pragma unroll
        for (int r = 0; r < 4; ++r) v[q][r] = r0 + q < a.rows ? d41x_ld(G.x, G.xcode, (size_t)(r0 + q) * G.ldx + blk * 128 + lane * 4 + r) : 0.f;
    #pragma unroll
    for (int q = 0; q < D41X_DD_PREP_ROWS; ++q) {
        if (r0 + q >= a.rows) break;
        d41x_prep4(v[q], sv, lane);
        __half o[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r) o[r] = __float2half_rn(v[q][r]);
        *(uint2*)(G.xh + (size_t)(r0 + q) * G.xld + blk * 128 + lane * 4) = *(const uint2*)o;
    }
}

// PD1: an RMSNorm whose epilogue also writes up to two dense linears' rotated inputs (the producer-side rotation: the
// next linears then run without d41x_dd_prep). y = d41_rmsnorm's output, the same body and bits (71_mix_hc.cuh
// d41_norm_body, declared here: band 70 is compiled after this file); then per target t and 128-block, one warp:
// xh_t[row][blk * 128 ..] = fp16(H128(f32(y) * suh_t)), d41x_dd_prep's math on the stored bf16 bits.
template <bool HC, bool PW>
__device__ __forceinline__ void d41_norm_body(const __nv_bfloat16* x, long long x_ld, const float* pre, long long pre_stride, const void* w, unsigned w_fmt,
                                              __nv_bfloat16* y_norm, long long yn_ld, __nv_bfloat16* y_raw, long long yr_ld, int dim, float eps, float inv_dim);
struct D41xNormRotArgs {
    const __nv_bfloat16* x; long long x_ld;
    const void* w;
    __nv_bfloat16* y; long long y_ld;
    unsigned w_fmt; int dim; int rows;
    float eps; float inv_dim;
    int nt;                                  // rotation targets (0..2)
    const __half* suh[2];                    // each target's suh at its first input column (= y column 0)
    __half* xh[2];                           // each target's rotated rows [rows][xld]
    int xld[2];
};
__device__ __forceinline__ void d41x_dd_normrot_body(const D41xNormRotArgs& a) {
    d41_norm_body<false, false>(a.x, a.x_ld, nullptr, 0, a.w, a.w_fmt, a.y, a.y_ld, nullptr, 0, a.dim, a.eps, a.inv_dim);
    __syncthreads();                         // this block's y stores are visible to its own threads
    const int row = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nb = a.dim / 128;
    const __nv_bfloat16* yr = a.y + (long long)row * a.y_ld;
    for (int t = 0; t < a.nt; ++t) {
        for (int blk = warp; blk < nb; blk += 8) {
            float v[4], sv[4];
            #pragma unroll
            for (int r = 0; r < 4; ++r) { v[r] = __bfloat162float(yr[blk * 128 + lane * 4 + r]); sv[r] = __half2float(a.suh[t][blk * 128 + lane * 4 + r]); }
            d41x_prep4(v, sv, lane);
            __half o[4];
            #pragma unroll
            for (int r = 0; r < 4; ++r) o[r] = __float2half_rn(v[r]);
            *(uint2*)(a.xh[t] + (size_t)row * a.xld[t] + blk * 128 + lane * 4) = *(const uint2*)o;
        }
    }
}
extern "C" __global__ void __launch_bounds__(256) d41x_dd_normrot(const __grid_constant__ D41xNormRotArgs a) {
    D41_PDL_ENTRY();
    d41x_dd_normrot_body(a);
}

// L2 prefetch of a weight range (T0 lever 3), for a forward that knows which weights come next: `count` runs of `run`
// bytes, `stride` bytes apart, from p (a slice of a whole-resident trellis is one run per k-step). Lines go to L2 with
// evict_last. PDL order: trigger the successor FIRST, issue the prefetches (they overlap the predecessor's tail and the
// successor), THEN wait for the predecessor, so the successor's own wait still covers the real producer.
extern "C" __global__ void __launch_bounds__(256) d41x_l2_prefetch(const unsigned char* __restrict__ p, long long run, long long stride, int count) {
    D41_PDL_RELEASE();
    const long long lines = (run + 127) / 128;
    const long long total = lines * count;
    for (long long i = blockIdx.x * 256ll + threadIdx.x; i < total; i += (long long)gridDim.x * 256) {
        const long long r = i / lines, l = i - r * lines;
        asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(p + r * stride + l * 128));
    }
    D41_PDL_WAIT_ONLY();
}

#include "09_l2_paced.cuh"

// the host reads the argument layout once (prepare time, never inside a capture)
extern "C" __global__ void d41x_dd_sizes(int* o) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    o[0] = (int)sizeof(D41xDdArgs); o[1] = (int)sizeof(D41xDdSeg); o[2] = D41X_DD_P; o[3] = D41X_DD_KPS; o[4] = D41X_DD_MAXSEG; o[5] = D41X_DD_PREP_ROWS;
    o[6] = (int)sizeof(D41xNormRotArgs); o[7] = (int)sizeof(D41xL2PacedArgs);
}
