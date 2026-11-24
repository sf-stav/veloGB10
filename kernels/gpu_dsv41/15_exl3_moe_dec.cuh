// 15_exl3_moe_dec.cuh (leg KE2): routed-MoE DECODE (<= 32 rows) in three launches, routed + shared expert fused.
//   d41x_md_prep   xh1[pair][mat] = fp16(H128(x[row] * suh_mat))      (pairs: the shared expert's rows, then routed)
//   d41x_md_gu_*   w1 / w3 over items (slot, 128-block, part of q): persistent CTAs (2 per SM for <= 16 rows) stream
//                  trellis + A through a cp.async ring sized at run time; the last part of a block to finish runs SwiGLU
//                  + w2's input transform for it -> xh2
//   d41x_md_dn_*   w2 over items (slot, output 128-block); H128 -> h2; the last item of an output block combines its
//                  rows: y = (y + sum of the routed experts in ascending id) + shared expert
// NUMERICS == the generic path (13_*) and the prefill kernels (16_*) bit for bit: per output column one MMA chain per
// k-block of 8 k16 steps, the k-blocks added in f32 in ascending order (D41X_KBLK, 11_*: the GB10 f16 accumulator is
// lossy over long chains), the same pinned per-element math (d41x_prep4 / d41x_act1 / d41x_had128), the same combine
// order. The item split (q) only decides which CTA computes a column, never its arithmetic.
// ROUTING: every CTA recomputes the compaction in shared memory from ids (rows <= 32: one u32 row mask per expert):
//   slot 0 = the shared expert (when given), then the held routed experts in ascending id; a slot's pairs are its rows
//   in ascending order; pair index = slot offset + rank of the row in the expert's mask.
// RING: `ns` slots of `slot` bytes (host-computed from the call's widest K and its row count): [trellis: wreg B][A rows].
// PDL: gu / dn read ids and issue the TRELLIS part of their first ring slots before griddepcontrol.wait (the weights
//   are constant and ids were written before the predecessor's own wait: every dsv41 kernel waits before it
//   triggers), then wait, then trigger, then stage the activations. Without the attribute both are no-ops.
// COUNTERS (last-arrival epilogues) live in the layer's MoeTable allocation: zeroed once, re-armed by the last arrival.
// PF1 (the external reference's TP2 decode MoE structure): prep bumps the table's call epoch; gate/up publishes each slot's
//   readiness (ready[s] = epoch, release) when the last of the slot's 128-blocks has written its xh2 rows; down does NOT
//   wait for the whole gate/up grid: a down CTA issues its first ring slots' trellis, then waits only for the slot of the
//   item whose activations it is about to stage (slot 0 = the shared expert, first), and executes griddepcontrol.wait at its
//   end (so its completion still implies gate/up's: the stream's PDL chain stays transitive). `ow`: the combine writes
//   y = 0 + sum (the bits of a zeroed y plus the sum) instead of y + sum.
#pragma once

#define D41X_MD_MAXS 257
#define D41X_MD_KPS 4            // k16 steps per ring slot
#define D41X_MD_THREADS 256
#define D41X_MD_PMAX 8           // PV1: gate/up K parts at most
#define D41X_MD_FOLDB 16         // PV1: chain loads in flight per thread in the K-part fold

struct D41xMdArgs {
    const D41xExpert* ex;        // routed table [n_experts]
    const D41xExpert* sh;        // shared expert table [1], or null
    const int* ids; const float* wts;          // [rows][topk]
    const void* x; int xcode, ldx;
    __half* xh1;                 // [pair][2][dim]
    float* graw;                 // [pair][2][inter_ld]  raw w1 / w3 column sums
    __half* xh2;                 // [pair][inter_ld]
    float* h2;                   // [pair][dim]          H128(w2 raw), before svh
    float* out; int ldo;         // f32 [rows][ldo], ADDED into
    int* cnt;                    // [MAXS * ncb] gate/up arrivals, then [dim / 128] down arrivals
    float limit, limit_sh;
    int n_experts, rows, topk, dim, inter_ld, ncb, kmax, kmin;   // SX1: informational; items read K from their entry
    int slot, ns, wreg, sms;     // ring slot stride (B), ring slots (>= 3), trellis bytes per slot (A rows follow), SMs
    float* chain;                // PV1 (19_*, stream-K): per-k-block chains of split items [KB][pair][2][inter_ld] / [KB][pair][dim]
    int ow, pad;                 // PF1: 1 = the combine overwrites y (0 + sum)
};

// PF1: the readiness block after the down arrivals in the MoeTable counters: [0, MAXS) the gate/up blocks done per slot,
// [MAXS, 2 MAXS) each slot's ready epoch, [2 MAXS] the call epoch (bumped by prep)
__device__ __forceinline__ unsigned* d41x_md_rdy(const D41xMdArgs& a) { return (unsigned*)(a.cnt + D41X_MD_MAXS * a.ncb + (a.dim >> 7)); }
__device__ __forceinline__ unsigned d41x_ld_acq(const unsigned* p) {
    unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ void d41x_st_rel(unsigned* p, unsigned v) { asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ unsigned d41x_add_acqrel(unsigned* p, unsigned v) {
    unsigned o; asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(o) : "l"(p), "r"(v) : "memory"); return o;
}

// routing block: this header, then mask[n_experts] (u32), then slot_of[n_experts] (short)
struct D41xMdRoute {
    short s_e[D41X_MD_MAXS], s_cnt[D41X_MD_MAXS], s_off[D41X_MD_MAXS], s_nb[D41X_MD_MAXS];
    int s_blk[D41X_MD_MAXS + 1];  // prefix of s_nb (128-blocks before slot s)
    int n_slots, n_pairs, flag, q;
    int p;                        // PV1: gate/up K parts per item (1 = the whole K; > 1 only with a chain scratch)
};
__device__ __forceinline__ unsigned* d41x_md_mask(D41xMdRoute& R) { return (unsigned*)((unsigned char*)&R + sizeof(D41xMdRoute)); }
__device__ __forceinline__ short* d41x_md_slot_of(D41xMdRoute& R, int E) { return (short*)(d41x_md_mask(R) + E); }
__host__ __device__ __forceinline__ int d41x_md_route_bytes(int E) { return (int)((sizeof(D41xMdRoute) + 6 * (size_t)E + 127) / 128 * 128); }

// ---- the routing compaction (all threads; ends with a barrier) --------------------------------------------------------
__device__ __forceinline__ void d41x_md_route(const D41xMdArgs& a, D41xMdRoute& R) {
    const int t = threadIdx.x, lane = t & 31;
    unsigned* mask = d41x_md_mask(R);
    short* slot_of = d41x_md_slot_of(R, a.n_experts);
    for (int e = t; e < a.n_experts; e += blockDim.x) mask[e] = 0u;
    __syncthreads();
    const int P = a.rows * a.topk;
    for (int p = t; p < P; p += blockDim.x) {
        const int e = __ldcg(a.ids + p);
        if (e >= 0 && e < a.n_experts && a.ex[e].nblk > 0) atomicOr(&mask[e], 1u << (p / a.topk));
    }
    __syncthreads();
    if (t < 32) {
        const int base = a.sh ? 1 : 0, rows_sh = a.sh ? a.rows : 0;
        const int per = (a.n_experts + 31) >> 5, e0 = lane * per;
        int c = 0, f = 0;
        for (int i = 0; i < per; ++i) { const int e = e0 + i; if (e < a.n_experts) { const unsigned m = mask[e]; c += __popc(m); f += (m != 0u); } }
        int ic = c, iff = f;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int x1 = __shfl_up_sync(0xFFFFFFFFu, ic, o), x2 = __shfl_up_sync(0xFFFFFFFFu, iff, o);
            if (lane >= o) { ic += x1; iff += x2; }
        }
        int bc = rows_sh + ic - c, bf = base + iff - f;
        for (int i = 0; i < per; ++i) {
            const int e = e0 + i;
            if (e >= a.n_experts) break;
            const unsigned m = mask[e];
            if (m) { slot_of[e] = (short)bf; R.s_e[bf] = (short)e; R.s_cnt[bf] = (short)__popc(m); R.s_off[bf] = (short)bc; R.s_nb[bf] = (short)a.ex[e].nblk; bc += __popc(m); ++bf; }
            else slot_of[e] = -1;
        }
        const int ns = base + __shfl_sync(0xFFFFFFFFu, iff, 31);
        const int np = rows_sh + __shfl_sync(0xFFFFFFFFu, ic, 31);
        if (lane == 0) {
            if (a.sh) { R.s_e[0] = -1; R.s_cnt[0] = (short)a.rows; R.s_off[0] = 0; R.s_nb[0] = (short)a.sh->nblk; }
            R.n_slots = ns; R.n_pairs = np;
        }
        __syncwarp();
        const int ps = (ns + 31) >> 5, s0 = lane * ps;
        int it = 0;
        for (int i = 0; i < ps; ++i) { const int s = s0 + i; if (s < ns) it += R.s_nb[s]; }
        int iit = it;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) { const int x1 = __shfl_up_sync(0xFFFFFFFFu, iit, o); if (lane >= o) iit += x1; }
        int b = iit - it;
        for (int i = 0; i < ps; ++i) { const int s = s0 + i; if (s < ns) { R.s_blk[s] = b; b += R.s_nb[s]; } }
        // item split: 4 parts per 128-block when 2 would leave SMs idle (the split never changes a column's arithmetic).
        // PV1: with a chain scratch, the call's 128-block count NB picks column parts q and K parts p (exact: the chains of
        // parts > 0 are folded in k-block order by the block's last arrival). Measured on GB10 (PV1 report, md_bench,
        // graph + PDL, 48 SMs, 96 CTAs): 2 NB < SMs (TP8 one row): q = 2 (no half-tile decode) and ~80 items; CTAs < 2 NB
        // <= 1.2 CTAs (TP4 two rows, TP2 one row routed-only: a second wave of a few items): ~10 items per SM; otherwise one
        // pass over K (p = 1: the split's fold made TP4 one row, TP8 four rows, TP4 three rows and TP2 one row with the
        // shared expert (2 NB = 126) slower).
        if (lane == 31) {
            const int q2 = 2 * iit;
            R.s_blk[ns] = iit; R.q = (q2 < a.sms) ? 4 : 2; R.p = 1;
            if (a.chain != nullptr) {
                if (q2 > 0 && q2 < a.sms) { R.q = 2; R.p = min(D41X_MD_PMAX, (80 + q2 - 1) / q2); }   // KF2: q2 = 0 (no held block) keeps p = 1
                else if (q2 > (int)gridDim.x && 5 * q2 <= 6 * (int)gridDim.x) R.p = min(D41X_MD_PMAX, (10 * a.sms + q2 - 1) / q2);
            }
        }
    }
    __syncthreads();
}

__device__ __forceinline__ const D41xExpert& d41x_md_entry(const D41xMdArgs& a, const D41xMdRoute& R, int s) {
    return R.s_e[s] < 0 ? *a.sh : a.ex[R.s_e[s]];
}

// the j-th set bit of m (j < popc(m))
__device__ __forceinline__ int d41x_nth_bit(unsigned m, int j) { for (int i = 0; i < j; ++i) m &= m - 1u; return __ffs(m) - 1; }
// routing weight of (row, expert e)
__device__ __forceinline__ float d41x_md_weight(const D41xMdArgs& a, int row, int e) {
    for (int k = 0; k < a.topk; ++k) if (__ldcg(a.ids + row * a.topk + k) == e) return __ldcg(a.wts + row * a.topk + k);
    return 0.f;
}

// ---- trellis decode of one 16x16 tile from its ring words: the "word diet" (2 words per lane for K <= 4, 3 above) ----
__device__ __forceinline__ int d41x_lane_t0(int lane) {
    const int c = lane >> 2;
    return 8 * (8 * ((c >> 1) & 3) + (lane & 3)) + 32 * (c & 1);
}
// Q96 (PK3, used by the prefill kernels 16_*): for K > 4 the same 8 fields with fewer live registers (see decode); the decode
// kernels here keep Q96 = false (their code is unchanged).
template <int K, bool Q96 = false>
struct D41xDiet {
    static constexpr int WR = 8 * K;
    int o0, o1, o2, e0;
    __device__ __forceinline__ void init(int lane) {
        const int b10 = (d41x_lane_t0(lane) + 257) * K;
        const int A = (b10 - 16) >> 5;
        o0 = A % WR; o1 = (A + 1) % WR; o2 = (A + 2) % WR;
        e0 = b10 - 32 * A;
    }
    __device__ __forceinline__ void decode(const uint32_t* __restrict__ w, unsigned (&bf)[2][2]) const {
        unsigned idx[8];
        if constexpr (K <= 4) {
            const unsigned long long P = ((((unsigned long long)w[o0]) << 32) | (unsigned long long)w[o1]) >> (64 - (e0 + 7 * K));
            #pragma unroll
            for (int j = 0; j < 8; ++j) idx[j] = (unsigned)(P >> (K * (7 - j))) & 0xFFFFu;
        } else {
          if constexpr (Q96) {
            // PK3: the same fields with fewer live registers: Q = (w0:w1:w2) << (e0 - 16) (one runtime funnel shift per word), then
            // field j = the 16 bits at [jK, jK + 16) from the top of Q, every shift compile-time. Equal to the P01 / P12 form for
            // every e0 in [16, 47] and word values (tools/dsv41_pk3/diet_q96_test.cu, exhaustive over e0, random words).
            const unsigned w0 = w[o0], w1 = w[o1], w2 = w[o2];
            const unsigned sh = (unsigned)(e0 - 16);
            const unsigned q0 = __funnelshift_l(w1, w0, sh), q1 = __funnelshift_l(w2, w1, sh), q2 = w2 << sh;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int r = 80 - j * K;                       // field j = (Q >> r) & 0xFFFF, Q 96 bits (q0 top)
                unsigned f;
                if (r >= 64) f = q0 >> (r - 64);
                else if (r >= 32) f = __funnelshift_r(q1, q0, r - 32);
                else f = __funnelshift_r(q2, q1, r);
                idx[j] = f & 0xFFFFu;
            }
          } else {
            const unsigned w1 = w[o1];
            const unsigned long long P01 = (((unsigned long long)w[o0]) << 32) | (unsigned long long)w1;
            const unsigned long long P12 = (((unsigned long long)w1) << 32) | (unsigned long long)w[o2];
            #pragma unroll
            for (int j = 0; j < 8; ++j) { const int e = e0 + j * K; idx[j] = (unsigned)(e <= 64 ? (P01 >> (64 - e)) : (P12 >> (96 - e))) & 0xFFFFu; }
          }
        }
        const __half2 kinv = __half2half2(__ushort_as_half((unsigned short)0x1EEE));
        const __half2 kbias = __half2half2(__ushort_as_half((unsigned short)0xC931));
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int p = 0; p < 2; ++p) {
                const unsigned s0 = __dp4a(idx[4 * h + 2 * p] * D41X_MUL1, 0x01010101u, 0x00006400u);
                const unsigned s1 = __dp4a(idx[4 * h + 2 * p + 1] * D41X_MUL1, 0x01010101u, 0x00006400u);
                const __half2 v = __hfma2(__halves2half2(__ushort_as_half((unsigned short)s0), __ushort_as_half((unsigned short)s1)), kinv, kbias);
                bf[h][p] = *reinterpret_cast<const unsigned*>(&v);
            }
    }
};

__device__ __forceinline__ void d41x_mma(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void d41x_cp16(unsigned dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void d41x_cp16_ef(unsigned dst, const void* src, unsigned long long pol) {
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "l"(pol) : "memory");
}
__device__ __forceinline__ unsigned long long d41x_pol_ef() {
    unsigned long long p; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(p)); return p;
}
// A rows in the ring: 32 B per row and k16 step; the two 16-B halves of row j are swapped when bit 2 of j is set
// (the m16n8k16 A loads of rows gid and gid + 4 then hit different banks).
__device__ __forceinline__ int d41x_arow(int j, int half) { return j * 32 + ((half ^ ((j >> 2) & 1)) << 4); }

// A fragment (rows r0 / r0 + 8 of one m16 tile, k16 step) from a ring A block [rows][32 B]; rows >= cnt read as 0.
template <bool HI>
__device__ __forceinline__ void d41x_afrag(const unsigned char* Ab, int r0, int cnt, int tig, unsigned& a0, unsigned& a1, unsigned& a2, unsigned& a3) {
    a0 = a2 = a1 = a3 = 0u;
    if (r0 < cnt) { a0 = *(const unsigned*)(Ab + d41x_arow(r0, 0) + 4 * tig); a2 = *(const unsigned*)(Ab + d41x_arow(r0, 1) + 4 * tig); }
    if (HI && r0 + 8 < cnt) { a1 = *(const unsigned*)(Ab + d41x_arow(r0 + 8, 0) + 4 * tig); a3 = *(const unsigned*)(Ab + d41x_arow(r0 + 8, 1) + 4 * tig); }
}


// ---- one ring slot of MMAs (KPS k16 steps) ---------------------------------------------------------------------------
// tb0: this warp's tile at k-step 0 of the slot; tkk: trellis bytes per k-step; ab0 / akk: the A block of k-step 0 and its
// stride; HALF: the warp owns one n8 half (`hsel`) of its tile (q = 4 items), else both halves.
template <int K, int AR, bool HALF>
__device__ __forceinline__ void d41x_md_consume(const unsigned char* tb0, int tkk, const unsigned char* ab0, int akk, int hsel,
                                                int cnt, int lane, const D41xDiet<K>& D, float (&acc)[AR > 16 ? 2 : 1][2][4]) {
    constexpr int MT = AR > 16 ? 2 : 1;
    const int gid = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int kk = 0; kk < D41X_MD_KPS; ++kk) {
        unsigned bf[2][2];
        D.decode((const uint32_t*)(tb0 + kk * tkk), bf);
        if (HALF && hsel) { bf[0][0] = bf[1][0]; bf[0][1] = bf[1][1]; }
        const unsigned char* Ab = ab0 + kk * akk;
        #pragma unroll
        for (int m = 0; m < MT; ++m) {
            unsigned a0, a1, a2, a3;
            d41x_afrag<(AR > 8)>(Ab, 16 * m + gid, cnt, tig, a0, a1, a2, a3);
            d41x_mma(acc[m][0], a0, a1, a2, a3, bf[0][0], bf[0][1]);
            if (!HALF) d41x_mma(acc[m][1], a0, a1, a2, a3, bf[1][0], bf[1][1]);
        }
    }
}

// ---- item descriptors ------------------------------------------------------------------------------------------------
struct D41xMdItem {
    const unsigned char* w[2];   // trellis run bases at k-step 0 (gu: w1 / w3 at the item's first n-tile; dn: w2)
    long long kunit;             // SX1: bytes between k-steps per unit of K (a run's k-step stride = kunit * its K)
    const __half* xa;            // A rows: gu: xh1 + off*2*dim; dn: xh2 + off*inter_ld
    int nq;                      // ring slots in the item
    int K, cnt, off, s, b, h, ntr;   // width of run 0, rows, first pair, slot, 128-block (gu) / out-block (dn), part (gu), tiles per run
    int K3;                      // SX1: width of run 1 (gu: w3, which may differ from w1; dn: = K)
    int p, kb0;                  // PV1: K part of the item (gu) and its first k-block (128 k)
};
// PV1: k-block range [kb0, kb1) of gate/up K part p of P (KB = dim / 128 k-blocks)
__device__ __forceinline__ int d41x_md_pkb(int p, int P, int KB) { return p * KB / P; }
__device__ __forceinline__ int d41x_md_find_slot(const D41xMdRoute& R, int blk) {
    int lo = 0, hi = R.n_slots - 1;      // largest s with s_blk[s] <= blk
    while (lo < hi) { const int mid = (lo + hi + 1) >> 1; if (R.s_blk[mid] <= blk) lo = mid; else hi = mid - 1; }
    return lo;
}
__device__ __forceinline__ void d41x_md_item_gu(const D41xMdArgs& a, const D41xMdRoute& R, int i, D41xMdItem& it) {
    const int q = R.q, P = R.p, gb = i / (q * P), r = i - gb * q * P, KB = a.dim >> 7;
    const int s = d41x_md_find_slot(R, gb);
    const D41xExpert& E = d41x_md_entry(a, R, s);
    // SX1: w1 (run 0) and w3 (run 1) each at their own width code; equal codes give the pre-SX1 addresses exactly
    const int K1 = E.k1, K3 = E.k3, TB1 = 32 * K1, TB3 = 32 * K3;
    it.s = s; it.b = gb - R.s_blk[s]; it.h = r / P; it.p = r - it.h * P; it.K = K1; it.K3 = K3; it.ntr = 8 / q;
    it.cnt = R.s_cnt[s]; it.off = R.s_off[s];
    it.kb0 = d41x_md_pkb(it.p, P, KB);
    it.kunit = (long long)E.nbs13 * 32;
    const long long tile0 = (long long)(8 * it.b + it.ntr * it.h);
    it.w[0] = (const unsigned char*)E.tr[0] + tile0 * TB1 + (long long)it.kb0 * 8 * (it.kunit * K1);
    it.w[1] = (const unsigned char*)E.tr[1] + tile0 * TB3 + (long long)it.kb0 * 8 * (it.kunit * K3);
    it.xa = a.xh1 + (size_t)it.off * 2 * a.dim + it.kb0 * 128;
    it.nq = (d41x_md_pkb(it.p + 1, P, KB) - it.kb0) * 8 / D41X_MD_KPS;
}
__device__ __forceinline__ void d41x_md_item_dn(const D41xMdArgs& a, const D41xMdRoute& R, int i, D41xMdItem& it) {
    const int NJ = a.dim >> 7;
    const int s = i / NJ, j = i - s * NJ;
    const D41xExpert& E = d41x_md_entry(a, R, s);
    const int K = E.k2, TB = 32 * K;    // SX1: w2 at its own width code (may differ from w1 / w3)
    it.s = s; it.b = j; it.h = 0; it.K = K; it.K3 = K; it.ntr = 8; it.p = 0; it.kb0 = 0;
    it.cnt = R.s_cnt[s]; it.off = R.s_off[s];
    it.w[0] = it.w[1] = (const unsigned char*)E.tr[2] + (long long)(8 * j) * TB;
    it.kunit = (long long)(a.dim / 16) * 32;
    it.xa = a.xh2 + (size_t)it.off * a.inter_ld;
    it.nq = E.nblk * 8 / D41X_MD_KPS;
}

// ---- producer: this thread's copy descriptors for an item (computed once per item; a slot then adds q * step) ----------
// chunk c of a slot (c = t + 256 i, i < 4): W chunks first (k-step kk, run, 16-B piece cc), then A chunks (k-step kk,
// matrix, row j, half hf). W lands at [kk][run][tiles][32K B]; A at wreg + [kk][mat][rows][32 B] (d41x_arow swizzle).
// SX1: a k-step of a gu item holds run 0 (w1, tpr tiles of 32 K1 B) then run 1 (w3, tpr tiles of 32 K3 B); with K1 == K3
// every chunk's source, destination and owning thread are the pre-SX1 ones. flags bit 8 + i: chunk i is run 1 (w3); bits
// 16..19 / 20..23: K of run 0 / run 1; wstep = the slot step per unit of K (a chunk steps by wstep * its run's K).
struct D41xMdIss { const unsigned char* src[4]; unsigned dst[4]; unsigned flags; long long wstep; };
template <bool GU>
__device__ __forceinline__ void d41x_md_iss(const D41xMdArgs& a, const D41xMdItem& it, D41xMdIss& I) {
    const int t = threadIdx.x;
    const int TB0 = 32 * it.K, TB1 = GU ? 32 * it.K3 : 0;
    const int tpr = it.ntr, cpr0 = tpr * TB0 / 16, cps = cpr0 + tpr * TB1 / 16;
    const int nw = D41X_MD_KPS * cps;
    const int mats = GU ? 2 : 1;
    const int na = D41X_MD_KPS * mats * it.cnt * 2;
    I.flags = ((unsigned)it.K << 16) | ((unsigned)it.K3 << 20); I.wstep = (long long)D41X_MD_KPS * it.kunit;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c = t + D41X_MD_THREADS * i;
        I.src[i] = nullptr; I.dst[i] = 0u;
        if (c < nw) {
            const int kk = c / cps, rem = c - kk * cps;
            const int run = rem >= cpr0, cc = run ? rem - cpr0 : rem;
            I.src[i] = (run ? it.w[1] : it.w[0]) + (long long)kk * it.kunit * (run ? it.K3 : it.K) + cc * 16;
            I.dst[i] = (unsigned)(kk * tpr * (TB0 + TB1) + (run ? tpr * TB0 : 0) + cc * 16);
            I.flags |= (1u << i) | ((unsigned)run << (8 + i));
        } else if (c < nw + na) {
            const int ca = c - nw, hf = ca & 1, rest = ca >> 1;
            const int j = rest % it.cnt, km = rest / it.cnt;          // km = kk * mats + mat
            const int kk = km / mats, mat = km - kk * mats;
            const __half* src = GU ? it.xa + ((size_t)j * 2 + mat) * a.dim + kk * 16 + hf * 8
                                   : it.xa + (size_t)j * a.inter_ld + kk * 16 + hf * 8;
            I.src[i] = (const unsigned char*)src;
            I.dst[i] = (unsigned)(a.wreg + km * it.cnt * 32 + d41x_arow(j, hf));
            I.flags |= 0x11u << i;
        }
    }
}
// issue slot q of the item described by I into ring slot `dst` (W and / or A parts). Every copy carries an explicit L2
// policy (trellis: evict_first; activations: evict_last, other CTAs re-read them): ptxas 13.0 for sm_121a emitted the
// plain cp.async of this loop with an uninitialised memory descriptor (desc[UR1], CUDA_ERROR_ILLEGAL_INSTRUCTION).
__device__ __forceinline__ void d41x_md_issue(const D41xMdIss& I, int q, unsigned dst, bool do_w, bool do_a, unsigned long long pol, unsigned long long pol_a) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (!((I.flags >> i) & 1u)) continue;
        if ((I.flags >> (4 + i)) & 1u) { if (do_a) d41x_cp16_ef(dst + I.dst[i], I.src[i] + (long long)q * (D41X_MD_KPS * 32), pol_a); }
        else if (do_w) d41x_cp16_ef(dst + I.dst[i], I.src[i] + (long long)q * I.wstep * (int)((I.flags >> (((I.flags >> (8 + i)) & 1u) ? 20 : 16)) & 15u), pol);
    }
}

// The warm-up loops (the first D slots, once per call) compute every chunk address directly, the way the v7 kernels did:
// with the precomputed descriptors in those two loops, ptxas 13.0 (sm_121a) emitted the copies with their 64-bit memory
// descriptor in an ODD uniform register (desc[UR1]) and the device raised CUDA_ERROR_ILLEGAL_INSTRUCTION. The steady
// state keeps the precomputed descriptors (that loop compiles correctly; the probe's SASS scan checks every instance).
template <bool GU>
__device__ __forceinline__ void d41x_md_issue_direct(const D41xMdArgs& a, const D41xMdItem& it, int q, unsigned dst, bool do_w, bool do_a,
                                                     unsigned long long pol, unsigned long long pol_a) {
    const int t = threadIdx.x;
    const int TB0 = 32 * it.K, TB1 = GU ? 32 * it.K3 : 0;
    const int tpr = it.ntr, cpr0 = tpr * TB0 / 16, cps = cpr0 + tpr * TB1 / 16;
    const int nw = do_w ? D41X_MD_KPS * cps : 0;
    const int mats = GU ? 2 : 1;
    const int na = do_a ? D41X_MD_KPS * mats * it.cnt * 2 : 0;
    const int k0 = q * D41X_MD_KPS;
    for (int c = t; c < nw + na; c += D41X_MD_THREADS) {
        if (c < nw) {
            const int kk = c / cps, rem = c - kk * cps;
            const int run = rem >= cpr0, cc = run ? rem - cpr0 : rem;
            d41x_cp16_ef(dst + (unsigned)(kk * tpr * (TB0 + TB1) + (run ? tpr * TB0 : 0) + cc * 16),
                         (run ? it.w[1] : it.w[0]) + (long long)(k0 + kk) * it.kunit * (run ? it.K3 : it.K) + cc * 16, pol);
        } else {
            const int ca = c - nw, hf = ca & 1, rest = ca >> 1;
            const int j = rest % it.cnt, km = rest / it.cnt;
            const int kk = km / mats, mat = km - kk * mats;
            const __half* src = GU ? it.xa + ((size_t)j * 2 + mat) * a.dim + (k0 + kk) * 16 + hf * 8
                                   : it.xa + (size_t)j * a.inter_ld + (k0 + kk) * 16 + hf * 8;
            d41x_cp16_ef(dst + (unsigned)(a.wreg + km * it.cnt * 32 + d41x_arow(j, hf)), src, pol_a);
        }
    }
}

template <int K, int AR, bool GU>
__device__ __forceinline__ void d41x_md_consume_k(const unsigned char* slot, const D41xMdArgs& a, const D41xMdItem& it, int warp, int lane,
                                                  float (&acc)[AR > 16 ? 2 : 1][2][4]) {
    constexpr int TB = 32 * K;   // this warp's matrix width (gu: w1 for warps 0..3, w3 for 4..7; see d41x_md_consume_any)
    D41xDiet<K> D; D.init(lane);
    if (GU) {
        const int mat = warp >> 2;
        const unsigned char* ab0 = slot + a.wreg + mat * it.cnt * 32;
        // SX1: a k-step = run 0 (ntr tiles of 32 K1 B) then run 1 (ntr tiles of 32 K3 B); K1 == K3 gives (mat * ntr + t) * TB
        const int ks = it.ntr * 32 * (it.K + it.K3), base = mat ? it.ntr * 32 * it.K : 0;
        if (it.ntr == 4) d41x_md_consume<K, AR, false>(slot + base + (warp & 3) * TB, ks, ab0, 2 * it.cnt * 32, 0, it.cnt, lane, D, acc);
        else             d41x_md_consume<K, AR, true>(slot + base + ((warp >> 1) & 1) * TB, ks, ab0, 2 * it.cnt * 32, warp & 1, it.cnt, lane, D, acc);
    } else {
        d41x_md_consume<K, AR, false>(slot + warp * TB, 8 * TB, slot + a.wreg, it.cnt * 32, 0, it.cnt, lane, D, acc);
    }
}
template <int AR, int KMAX, bool GU>
__device__ __forceinline__ void d41x_md_consume_any(const unsigned char* slot, const D41xMdArgs& a, const D41xMdItem& it, int warp, int lane,
                                                    float (&acc)[AR > 16 ? 2 : 1][2][4]) {
    // SX1: a gu warp consumes its own matrix (warps 0..3 w1 at K, 4..7 w3 at K3); a dn item has one width. Every K the
    // call's tables hold is <= KMAX (the host picks the instance from the largest code of either table, w2 included).
    switch (GU && (warp >> 2) ? it.K3 : it.K) {
        case 2: d41x_md_consume_k<2, AR, GU>(slot, a, it, warp, lane, acc); break;
        case 3: d41x_md_consume_k<3, AR, GU>(slot, a, it, warp, lane, acc); break;
        case 4: d41x_md_consume_k<4, AR, GU>(slot, a, it, warp, lane, acc); break;
        case 5: if constexpr (KMAX >= 5) d41x_md_consume_k<5, AR, GU>(slot, a, it, warp, lane, acc); break;
        case 6: if constexpr (KMAX >= 6) d41x_md_consume_k<6, AR, GU>(slot, a, it, warp, lane, acc); break;
        case 7: if constexpr (KMAX >= 7) d41x_md_consume_k<7, AR, GU>(slot, a, it, warp, lane, acc); break;
        case 8: if constexpr (KMAX >= 8) d41x_md_consume_k<8, AR, GU>(slot, a, it, warp, lane, acc); break;
        default: break;
    }
}
__device__ __forceinline__ void d41x_md_wait(int D) {    // cp.async.wait_group needs an immediate
    switch (D) {
        case 2: d41_cp_wait<1>(); break;  case 3: d41_cp_wait<2>(); break;  case 4: d41_cp_wait<3>(); break;
        case 5: d41_cp_wait<4>(); break;  case 6: d41_cp_wait<5>(); break;  case 7: d41_cp_wait<6>(); break;
        case 8: d41_cp_wait<7>(); break;  case 9: d41_cp_wait<8>(); break;  case 10: d41_cp_wait<9>(); break;
        case 11: d41_cp_wait<10>(); break; case 12: d41_cp_wait<11>(); break; case 13: d41_cp_wait<12>(); break;
        case 14: d41_cp_wait<13>(); break; case 15: d41_cp_wait<14>(); break;
        default: asm volatile("cp.async.wait_all;\n" ::: "memory"); break;
    }
}

// ---- PV1: the K-part fold of (slot s, block b): graw holds part 0's prefix; the chains of k-blocks [KB / P, KB) are added
// onto it in k-block order (the same f32 left fold a single pass makes); thread t: matrix t / 128, column t % 128 ----------
__device__ __forceinline__ void d41x_md_fold(const D41xMdArgs& a, D41xMdRoute& R, int s, int b) {
    const int t = threadIdx.x, mat = t >> 7, col = b * 128 + (t & 127);
    const int KB = a.dim >> 7, e0 = d41x_md_pkb(1, R.p, KB), cnt = R.s_cnt[s], off = R.s_off[s];
    const size_t cs = (size_t)R.n_pairs * 2 * a.inter_ld;
    for (int j = 0; j < cnt; ++j) {
        float* g = a.graw + ((size_t)(off + j) * 2 + mat) * a.inter_ld + col;
        const float* c = a.chain + ((size_t)(off + j) * 2 + mat) * a.inter_ld + col;
        float f = __ldcg(g);
        for (int k0 = e0; k0 < KB; k0 += D41X_MD_FOLDB) {
            float v[D41X_MD_FOLDB];
            #pragma unroll
            for (int i = 0; i < D41X_MD_FOLDB; ++i) v[i] = k0 + i < KB ? __ldcg(c + (size_t)(k0 + i) * cs) : 0.f;
            #pragma unroll
            for (int i = 0; i < D41X_MD_FOLDB; ++i) if (k0 + i < KB) f = __fadd_rn(f, v[i]);
        }
        *g = f;
    }
    __syncthreads();
}

// ---- the act epilogue of (slot s, block b): graw (every part complete) -> xh2 ------------------------------------------
// PV1: the same math, laid out for latency (it is the tail of the call at 1-2 rows): the expert's scale vectors and the
// rows' graw are loaded before the routing weight, which lanes k < topk read in parallel (the first match in top-k order,
// as d41x_md_weight's scan), so a row costs ~2 dependent memory round trips instead of up to 8.
__device__ __forceinline__ void d41x_md_act(const D41xMdArgs& a, D41xMdRoute& R, int s, int b) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const bool shx = R.s_e[s] < 0;
    const float lim = shx ? a.limit_sh : a.limit;
    const int cnt = R.s_cnt[s], off = R.s_off[s];
    if (warp >= cnt) return;
    const D41xExpert& E = d41x_md_entry(a, R, s);
    const unsigned m = shx ? 0u : d41x_md_mask(R)[R.s_e[s]];
    const int c0 = b * 128 + lane * 4;
    const __half *sv0 = E.svh[0], *sv1 = E.svh[1], *su2 = E.suh[2];
    float s0[4], s1[4], s2[4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) { s0[r] = __half2float(sv0[c0 + r]); s1[r] = __half2float(sv1[c0 + r]); s2[r] = __half2float(su2[c0 + r]); }
    for (int j = warp; j < cnt; j += D41X_MD_THREADS / 32) {
        const int pair = off + j;
        const float4 g4 = __ldcg((const float4*)(a.graw + ((size_t)pair * 2 + 0) * a.inter_ld + c0));
        const float4 u4 = __ldcg((const float4*)(a.graw + ((size_t)pair * 2 + 1) * a.inter_ld + c0));
        float w = 1.f;
        if (!shx) {
            const int row = d41x_nth_bit(m, j);
            int id = -1; float wk = 0.f;
            if (lane < a.topk) { id = __ldcg(a.ids + row * a.topk + lane); wk = __ldcg(a.wts + row * a.topk + lane); }
            const unsigned hit = __ballot_sync(0xFFFFFFFFu, id == R.s_e[s]);
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

// ---- the combine of output block jb over every row -----------------------------------------------------------------------
// y = (y + the row's held routed experts in ascending id, each fma with its svh) + the shared expert. PV1: the same order,
// laid out for latency: lanes k < topk read the row's ids at once, the ascending order comes from warp minima (a repeated
// id counts once, as the sequential scan did), every expert's h2 and svh load is issued before the ordered fmas.
template <int CB>
__device__ __forceinline__ void d41x_md_combine(const D41xMdArgs& a, D41xMdRoute& R, int jb) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int c0 = jb * 128 + lane * 4;
    const unsigned* mask = d41x_md_mask(R);
    const short* slot_of = d41x_md_slot_of(R, a.n_experts);
    for (int r = warp; r < a.rows; r += D41X_MD_THREADS / 32) {
        // lane k < topk: the row's k-th id if this rank holds it (a repeat of an earlier lane's id counts once)
        int e = 0x7FFFFFFF;
        if (lane < a.topk) {
            const int x = __ldcg(a.ids + r * a.topk + lane);
            if (x >= 0 && x < a.n_experts && slot_of[x] >= 0 && ((mask[x] >> r) & 1u)) e = x;
        }
        for (int j = 0; j < a.topk; ++j) { const int ej = __shfl_sync(0xFFFFFFFFu, e, j); if (j < lane && ej == e) e = 0x7FFFFFFF; }
        const bool val = e != 0x7FFFFFFF;
        const __half* esv = val ? a.ex[e].svh[2] : nullptr;
        int rank = 0;                                  // ascending position among the held ids
        for (int j = 0; j < a.topk; ++j) { const int ej = __shfl_sync(0xFFFFFFFFu, e, j); rank += ej < e; }
        const int nv = __popc(__ballot_sync(0xFFFFFFFFu, val));
        float* o = a.out + (size_t)r * a.ldo + c0;
        float yv[4];
        #pragma unroll
        for (int q = 0; q < 4; ++q) yv[q] = a.ow ? 0.f : __ldcg(o + q);   // PF1: overwrite = a zeroed y (0 + sum below)
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        for (int i0 = 0; i0 < nv; i0 += CB) {          // CB experts' loads in flight, then their fmas in order
            float4 v[CB]; uint2 sv[CB];
            #pragma unroll
            for (int i = 0; i < CB; ++i) {
                if (i0 + i >= nv) break;
                const int src = __ffs(__ballot_sync(0xFFFFFFFFu, val && rank == i0 + i)) - 1;
                const int be = __shfl_sync(0xFFFFFFFFu, e, src);
                const __half* svp = (const __half*)__shfl_sync(0xFFFFFFFFu, (unsigned long long)esv, src);
                const int pair = R.s_off[slot_of[be]] + __popc(mask[be] & ((1u << r) - 1u));
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

// harness-only timeline hooks (tools/dsv41_pv1/md_bench.cu, -DD41X_MD_TRACE): per CTA, globaltimer at fixed points
#ifdef D41X_MD_TRACE
__device__ unsigned long long d41x_md_trace[2][128][24];
#define D41X_MD_T(i) do { if (threadIdx.x == 0 && (i) < 24 && blockIdx.x < 128) { unsigned long long v_; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(v_)); d41x_md_trace[GU ? 0 : 1][blockIdx.x][(i)] = v_; } } while (0)
__device__ unsigned long long d41x_md_rtrace[D41X_MD_MAXS];   // PM1: per slot, the time gate/up published ready[s]
#define D41X_MD_RT(s) do { unsigned long long v_; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(v_)); d41x_md_rtrace[(s)] = v_; } while (0)
#else
#define D41X_MD_T(i) do { } while (0)
#define D41X_MD_RT(s) do { } while (0)
#endif

// ---- the persistent bodies -----------------------------------------------------------------------------------------------
// dynamic smem: [route block][ring: ns x slot][dn: epilogue stage [max(AR,16)][132] f32]
template <int AR, int KMAX, bool GU>
__device__ __forceinline__ void d41x_md_body(const D41xMdArgs& a) {
    constexpr int MT = AR > 16 ? 2 : 1;
    extern __shared__ __align__(128) unsigned char d41x_md_smem[];
    D41xMdRoute& R = *reinterpret_cast<D41xMdRoute*>(d41x_md_smem);
    const int route_b = d41x_md_route_bytes(a.n_experts);
    unsigned char* ring = d41x_md_smem + route_b;
    float* stage = reinterpret_cast<float*>(ring + a.ns * a.slot);
    const unsigned ring_u = (unsigned)__cvta_generic_to_shared(ring);
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int SLOT = a.slot, NS = a.ns, D = NS - 1;
    const unsigned long long pol = d41x_pol_ef();
    unsigned long long pol_a; asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol_a));

    D41X_MD_T(0);
    d41x_md_route(a, R);
    D41X_MD_T(1);
    const int NJ = a.dim >> 7;
    const int total = GU ? R.s_blk[R.n_slots] * R.q * R.p : R.n_slots * NJ;
    const int G0 = gridDim.x;
    int N = 0;                                     // this CTA's slots (CTA-uniform)
    for (int i = blockIdx.x; i < total; i += G0) {
        if (GU) { const int pp = i % R.p; N += (d41x_md_pkb(pp + 1, R.p, a.dim >> 7) - d41x_md_pkb(pp, R.p, a.dim >> 7)) * 8 / D41X_MD_KPS; }
        else { const int s = i / NJ; N += R.s_nb[s] * 8 / D41X_MD_KPS; }
    }
    auto item = [&](int i, D41xMdItem& it) { if (GU) d41x_md_item_gu(a, R, i, it); else d41x_md_item_dn(a, R, i, it); };
    // producer cursor
    D41xMdItem pi; D41xMdIss pd; int p_i = blockIdx.x, p_q = 0;
    if (p_i < total) { item(p_i, pi); d41x_md_iss<GU>(a, pi, pd); }
    auto p_adv = [&]() { if (++p_q == pi.nq) { p_q = 0; p_i += G0; if (p_i < total) { item(p_i, pi); d41x_md_iss<GU>(a, pi, pd); } } };
    // pre-wait: the trellis of slots 0..D-1 (one group each)
    {
        D41xMdItem ti = pi; int ti_i = p_i, ti_q = p_q;
        for (int n = 0; n < D; ++n) {
            if (n < N) {
                d41x_md_issue_direct<GU>(a, ti, ti_q, ring_u + (unsigned)(n * SLOT), true, false, pol, pol_a);
                if (++ti_q == ti.nq) { ti_q = 0; ti_i += G0; if (ti_i < total) item(ti_i, ti); }
            }
            d41_cp_commit();
        }
    }
    // PF1: gate/up waits for prep (its whole input); down waits per slot (below) and once more at its end
    if (GU) D41_PDL_WAIT_ONLY();
    D41_PDL_RELEASE();
    // down: the call epoch (prep, gate/up's predecessor, bumped it; down runs after gate/up's trigger, i.e. after prep
    // completed) and the last slot this CTA saw ready (its items' slots never decrease). Gate/up keeps neither live (128
    // registers: it reads the epoch again where it publishes)
    const unsigned ep = GU ? 0u : __ldcg(d41x_md_rdy(a) + 2 * D41X_MD_MAXS);
    int rdy_s = -1;
    auto need = [&](const D41xMdItem& it) {       // CTA-uniform: every thread calls it at the same point
        if (GU || it.s <= rdy_s) return;
        if (t == 0) { const unsigned* f = d41x_md_rdy(a) + D41X_MD_MAXS + it.s; while (d41x_ld_acq(f) != ep) __nanosleep(128); }
        __syncthreads();                           // acquire (thread 0) -> barrier -> every thread's activation copies
        if (rdy_s < 0) D41X_MD_T(20); else D41X_MD_T(21);   // PM1 trace (harness only): first / latest slot wait returned
        rdy_s = it.s;
    };
    for (int n = 0; n < D && n < N; ++n) { need(pi); d41x_md_issue_direct<GU>(a, pi, p_q, ring_u + (unsigned)(n * SLOT), false, true, pol, pol_a); p_adv(); }
    d41_cp_commit();
    // consumer cursor
    D41xMdItem ci; int c_i = blockIdx.x, c_q = 0;
    if (c_i < total) item(c_i, ci);
    static_assert(2 * D41X_MD_KPS == D41X_KBLK, "two ring slots make one accumulation k-block");
    float acc[MT][2][4], sum[MT][2][4];
    #pragma unroll
    for (int m = 0; m < MT; ++m)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int r = 0; r < 4; ++r) { acc[m][h][r] = 0.f; sum[m][h][r] = 0.f; }
    int buf = 0, pbuf = D, nit = 0;
    D41X_MD_T(2);
    for (int n = 0; n < N; ++n) {
        if (n == 0) asm volatile("cp.async.wait_all;\n" ::: "memory"); else d41x_md_wait(D);
        __syncthreads();
        if (n == 0) D41X_MD_T(3);
        if (n + D < N) { need(pi); d41x_md_issue(pd, p_q, ring_u + (unsigned)(pbuf * SLOT), true, true, pol, pol_a); p_adv(); }
        d41_cp_commit();
        if (++pbuf == NS) pbuf = 0;
        d41x_md_consume_any<AR, KMAX, GU>(ring + buf * SLOT, a, ci, warp, lane, acc);
        if (++buf == NS) buf = 0;
        if (c_q & 1) {      // a k-block (2 slots = 8 k16 steps) is complete: fold its chain into the running sum (f32)
            if (!GU || ci.p == 0) {
                #pragma unroll
                for (int m = 0; m < MT; ++m)
                    #pragma unroll
                    for (int h = 0; h < 2; ++h)
                        #pragma unroll
                        for (int r = 0; r < 4; ++r) { sum[m][h][r] = __fadd_rn(sum[m][h][r], acc[m][h][r]); acc[m][h][r] = 0.f; }
            } else {        // PV1: a K part > 0 stores each chain; the block's last arrival folds them in k-block order
                const int gid = lane >> 2, tig = lane & 3, mat = warp >> 2;
                const bool half = ci.ntr == 2;
                const int tl = half ? ((warp >> 1) & 1) : (warp & 3), hs = half ? (warp & 1) : 0;
                const int col0 = (8 * ci.b + ci.ntr * ci.h + tl) * 16;
                float* cb = a.chain + (size_t)(ci.kb0 + (c_q >> 1)) * R.n_pairs * 2 * a.inter_ld;
                #pragma unroll
                for (int m = 0; m < MT; ++m)
                    #pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        if (half && h == 1) continue;
                        const int r0 = 16 * m + gid, r1 = r0 + 8, col = col0 + 8 * (half ? hs : h) + 2 * tig;
                        if (r0 < ci.cnt) *(float2*)(cb + ((size_t)(ci.off + r0) * 2 + mat) * a.inter_ld + col) = make_float2(acc[m][h][0], acc[m][h][1]);
                        if (r1 < ci.cnt) *(float2*)(cb + ((size_t)(ci.off + r1) * 2 + mat) * a.inter_ld + col) = make_float2(acc[m][h][2], acc[m][h][3]);
                    }
                #pragma unroll
                for (int m = 0; m < MT; ++m)
                    #pragma unroll
                    for (int h = 0; h < 2; ++h)
                        #pragma unroll
                        for (int r = 0; r < 4; ++r) acc[m][h][r] = 0.f;
            }
        }
        if (++c_q < ci.nq) continue;
        D41X_MD_T(4 + 2 * nit);
        // ---- item epilogue
        const int gid = lane >> 2, tig = lane & 3;
        if (GU) {
            const int mat = warp >> 2;
            const bool half = ci.ntr == 2;
            const int tl = half ? ((warp >> 1) & 1) : (warp & 3), hs = half ? (warp & 1) : 0;
            const int col0 = (8 * ci.b + ci.ntr * ci.h + tl) * 16;
            #pragma unroll
            for (int m = 0; m < MT; ++m)
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    if (half && h == 1) continue;
                    if (ci.p != 0) continue;          // PV1: K parts > 0 wrote their chains
                    const int r0 = 16 * m + gid, r1 = r0 + 8, col = col0 + 8 * (half ? hs : h) + 2 * tig;
                    if (r0 < ci.cnt) *(float2*)(a.graw + ((size_t)(ci.off + r0) * 2 + mat) * a.inter_ld + col) = make_float2(sum[m][h][0], sum[m][h][1]);
                    if (r1 < ci.cnt) *(float2*)(a.graw + ((size_t)(ci.off + r1) * 2 + mat) * a.inter_ld + col) = make_float2(sum[m][h][2], sum[m][h][3]);
                }
            __syncthreads();
            int* ctr = a.cnt + ci.s * a.ncb + ci.b;
            if (t == 0) { __threadfence(); R.flag = (atomicAdd(ctr, 1) == R.q * R.p - 1); }
            __syncthreads();
            if (R.flag) {
                __threadfence();
                if (R.p > 1) d41x_md_fold(a, R, ci.s, ci.b);
                d41x_md_act(a, R, ci.s, ci.b);
                if (t == 0) *ctr = 0;
                // PF1: this block's xh2 rows are written (barrier: every thread's stores precede thread 0's release); the
                // slot's last block publishes ready[s] = epoch for the down grid and re-arms the slot's count
                __syncthreads();
                if (t == 0) {
                    unsigned* rdy = d41x_md_rdy(a);
                    const unsigned old = d41x_add_acqrel(rdy + ci.s, 1u);
                    if (old == (unsigned)R.s_nb[ci.s] - 1u) { rdy[ci.s] = 0u; d41x_st_rel(rdy + D41X_MD_MAXS + ci.s, __ldcg(rdy + 2 * D41X_MD_MAXS)); D41X_MD_RT(ci.s); }
                }
            }
        } else {
            #pragma unroll
            for (int m = 0; m < MT; ++m)
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int r0 = 16 * m + gid, col = warp * 16 + 8 * h + 2 * tig;
                    *(float2*)(stage + r0 * 132 + col) = make_float2(sum[m][h][0], sum[m][h][1]);
                    *(float2*)(stage + (r0 + 8) * 132 + col) = make_float2(sum[m][h][2], sum[m][h][3]);
                }
            __syncthreads();
            for (int j = warp; j < ci.cnt; j += D41X_MD_THREADS / 32) {
                float v[4];
                #pragma unroll
                for (int q = 0; q < 4; ++q) v[q] = stage[j * 132 + lane * 4 + q];
                d41x_had128(v, lane);
                *(float4*)(a.h2 + (size_t)(ci.off + j) * a.dim + ci.b * 128 + lane * 4) = make_float4(v[0], v[1], v[2], v[3]);
            }
            __syncthreads();
            int* ctr = a.cnt + D41X_MD_MAXS * a.ncb + ci.b;
            if (t == 0) { __threadfence(); R.flag = (atomicAdd(ctr, 1) == R.n_slots - 1); }
            __syncthreads();
            if (R.flag) {
                __threadfence();
                d41x_md_combine<(AR > 16 && KMAX > 5) ? 2 : 3>(a, R, ci.b);   // 2 keeps the K8 32-row instance spill-free
                if (t == 0) *ctr = 0;
            }
        }
        #pragma unroll
        for (int m = 0; m < MT; ++m)
            #pragma unroll
            for (int h = 0; h < 2; ++h)
                #pragma unroll
                for (int r = 0; r < 4; ++r) sum[m][h][r] = 0.f;
        D41X_MD_T(5 + 2 * nit); ++nit;
        c_q = 0; c_i += G0;
        if (c_i < total) item(c_i, ci);
    }
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    if (!GU) D41_PDL_WAIT_ONLY();                  // PF1: down completes only after gate/up has (the chain stays transitive)
    D41X_MD_T(23);
}

// ---- prep: one warp per (pair, 128-block); grid (ceil(dim/128/4), rows*topk + shared rows), 128 threads --------------------
extern "C" __global__ void __launch_bounds__(128) d41x_md_prep(D41xMdArgs a) {
    D41_PDL_ENTRY();
    // PF1: this call's epoch (the previous call of this table completed long before: the stream's PDL chain is transitive)
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) atomicAdd(d41x_md_rdy(a) + 2 * D41X_MD_MAXS, 1u);   // at L2 (never a stale L1 line)
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int blk = blockIdx.x * 4 + warp;
    if (blk >= (a.dim >> 7)) return;
    const int P = a.rows * a.topk, y = blockIdx.y;
    const D41xExpert* E; int pair, row;
    if (y < P) {
        row = y / a.topk;
        const int e = __ldcg(a.ids + y);
        if (e < 0 || e >= a.n_experts || a.ex[e].nblk <= 0) return;
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
    #pragma unroll
    for (int mat = 0; mat < 2; ++mat) {
        float v[4], sv[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r) { v[r] = d41x_ld(a.x, a.xcode, (size_t)row * a.ldx + blk * 128 + lane * 4 + r); sv[r] = __half2float(E->suh[mat][blk * 128 + lane * 4 + r]); }
        d41x_prep4(v, sv, lane);
        __half o[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r) o[r] = __float2half_rn(v[r]);
        *(uint2*)(a.xh1 + ((size_t)pair * 2 + mat) * a.dim + blk * 128 + lane * 4) = *(const uint2*)o;
    }
}

// two CTAs per SM (<= 128 registers) at every row count (PK4: the 17..32-row instances compile spill-free at 128 and ran one
// CTA per SM before; the host falls back to one CTA per SM when fewer than 3 ring slots fit in 48 KB)
#define D41X_MD_ENTRY(AR, KM, OCC)                                                                                          \
    extern "C" __global__ void __launch_bounds__(D41X_MD_THREADS, OCC) d41x_md_gu_a##AR##_k##KM(D41xMdArgs a) { d41x_md_body<AR, KM, true>(a); }  \
    extern "C" __global__ void __launch_bounds__(D41X_MD_THREADS, OCC) d41x_md_dn_a##AR##_k##KM(D41xMdArgs a) { d41x_md_body<AR, KM, false>(a); }
D41X_MD_ENTRY(8, 5, 2)
D41X_MD_ENTRY(16, 5, 2)
D41X_MD_ENTRY(32, 5, 2)
D41X_MD_ENTRY(8, 8, 2)
D41X_MD_ENTRY(16, 8, 2)
D41X_MD_ENTRY(32, 8, 2)

// The host reads the geometry from the device once (prepare time, never inside a capture).
extern "C" __global__ void d41x_md_sizes(int* o) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    o[0] = (int)sizeof(D41xMdArgs); o[1] = (int)sizeof(D41xMdRoute); o[2] = D41X_MD_MAXS; o[3] = D41X_MD_KPS; o[4] = d41x_md_route_bytes(384);
    o[5] = (int)sizeof(D41xExpert);   // SX1: the expert entry now carries k1 / k3 / k2
}
