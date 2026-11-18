// exl3_bench/16_dense_dks_verify.cuh: W4/DENSE dense split-K chains (exl3_dks_*) and the MTP k-chain verify kernels (S-A3-f-d).
// One launch replaces had_suh -> hmma_gemm_ks -> ks_combine -> had_svh.
// Section 16 of 23 of exl3_bench.cu: included in order by exl3_bench.cu, one translation unit; text is moved verbatim.
// =============================================================================
// W4/DENSE (2026-09-27): the dense split-K chains on W3/LMH's streaming design.
// exl3_dks_b{BITS}_s{NST} replaces, per chain (or per GROUP of same-input chains), the four
// launches exl3_had_suh -> exl3_hmma_gemm_ks[_p] -> exl3_ks_combine -> exl3_had_svh with ONE.
//
// Target contract: sm_121 BASELINE — cp.async.cg 16B + L2::cache_hint (createpolicy evict_first),
// mma.sync.m16n8k16 f16->f32, dp4a, shf, fma.rn.f16x2, atom.release.gpu / fence.acq_rel.gpu,
// griddepcontrol (no-ops without PDL), __grid_constant__ params (LDC c[0x0][R+imm]). No f/a
// suffix feature is used or needed.
//
// WHY PSK v1 LOST IN-ROUND, AND WHY THIS WILL NOT (ledger p4c §3/§8/§9 #2): PSK kept
// exl3_dq_from's 16 dependent scalar loads per lane per k16 step, so every warp had exactly ONE
// 32*BITS-byte trellis window in flight: 32 warps/SM x 160 B = ~5 KB/SM. At 238 GB/s / 48 SMs =
// 5 GB/s per SM that sustains the roofline only if a window returns in <= ~1 us — the loaded
// in-round DRAM latency is longer, so the bodies ran latency-bound at 69-79% (qkv 180 GB/s,
// q 189, out/o 165, z 138 on the fork, small shapes 120-145) — within +-2% of the old
// one-CTA-per-item grid (eager-solo ks_p 9.7 ms vs old ks 9.93 at m=8: -0.25 ms delivered of
// -1.5 priced). Its L2 prefetch and cp.async A staging did not change bytes-in-flight per SM,
// the actual limiter; the persistent grid only removed CTA launch cost that was never the gap.
// LMH streams the SAME 5-bit trellis at 244 GB/s IN-ROUND (1.63 ms, 397 MB) with: a cp.async.cg
// 16-B smem RING (L1 bypass, L2 evict_first), NST-1 = 5 slots x 5 KB = 25 KB in flight per SM
// (5x PSK) that never drains across tile boundaries, decode from smem (4 LDS.32 per lane per
// block instead of 16 dependent LDG), 1 CTA/SM x 8 warps. This kernel IS that pipeline, fed with
// the split-K work items. Its in-flight bytes per SM do not depend on the item size, so the
// small-K-slab items (4 k16 steps) stream exactly like the big ones.
//
//   * item = (member, N128 tile, K slab) = exactly one CTA of exl3_hmma_gemm_ks (same chain_ks).
//     Global item g is member-major, then tile, then slab (slab fastest); CTA c runs items
//     g = c + i*G ascending. G is a multiple of ks => a CTA sees ONE slab per member.
//   * the ring walks the CTA's (item, slot) sequence (slot = DKS_KSTEP k16 rows of the tile's 8
//     trellis blocks) with no drain between items; warps 0..6 issue, warp 7 never issues (its
//     lane 0 owns the tile-counter RMW, so that RMW's release fence cannot wait on a cp.async).
//   * A (the activation after the input Hadamard) is staged ONCE per CTA in smem for every
//     (member, slab) entry the CTA touches: computed in the prologue from x with the pinned WP20
//     suh sequence (lever c: no exl3_had_suh launch; SILU: x := f16(silu(x) * x2) first, the
//     shared-expert down's xq_silu_mul folded in) or copied from a staged xh (escape).
//   * split-K fixup (lever b, WP21-v2's math): every item stores its fp32 partials and arrives on
//     its tile counter (release RMW, DEFERRED one slot so its latency hides under the next slot's
//     MMA); the LAST arrival does the ascending combine + H128 * svh -> y one slot later, and
//     re-arms the counter (graph-replay safe, one real zero at allocation). No exl3_ks_combine /
//     exl3_had_svh launch.
//   * up to DKS_MAXMEM same-input chains (same K, bits, M) share one launch (lever d): one ramp,
//     one tail, the small items fill the big ones' last round.
//
// BIT-IDENTITY CONTRACT (== exl3_had_suh [after xq_silu_mul] -> exl3_hmma_gemm_ks ->
// exl3_ks_combine -> exl3_had_svh per member, every M in 1..16, every G, NST, grouping):
//   1. xh = xq_f2h(wp20_suh_had(x, suh)) — the pinned WP20-v1c op sequence, value-exact against
//      exl3_had_suh_dev under ANY contraction ptxas picks there (x*suh of two f16 values is exact
//      in f32, so fma(x0,s0,q) == rn(rn(x0*s0)+q); the WP20 note). SILU: xq_silu_mul's
//      expression verbatim (xq_f2h(xq_silu(xq_h2f(g)) * xq_h2f(u)): no product feeds an add).
//   2. partial[s][r][c]: acc = 0.0f; for kb ascending over slab s: two m16n8k16 (n8 halves h=0,1)
//      with the same A halves (rows >= M -> 0) and B = W3/LMH's quad extract (== exl3_dq_from;
//      host test lmh_quad_extract_equals_dq_from) -> one fp32 store per element, same address.
//   3. y: acc = 0.0f; acc += partial[s] for s = 0..ks-1 ASCENDING; f16_rn; exl3_had128;
//      f16_rn(v * svh) — exl3_ks_combine + exl3_had_svh_dev's exact sequence (WP21 v2). No
//      multiply feeds an add in 3 (no contraction freedom).
//   G, NST, the grouping, the fixup CTA and the arrival order decide WHO/WHEN, never HOW. The
//   partition (ks) is the chain's own m-independent chain_ks => batch invariance as before.
// Ordering (PTX memory model, WP21's proof): partial stores (all threads) -> bar.sync ->
// atom.release.gpu (warp 7 lane 0; cumulative over the CTA through bar.sync). The last arrival's
// RMW reads the value written by the ks-1 earlier RMWs (single modification order) ->
// fence.acq_rel.gpu (acquire pattern) -> s_fix -> bar.sync -> the fixup's ld.global.cg reads.
// =============================================================================
#define DKS_KSTEP      4                          // k16 steps per ring slot (== LMH_KSTEP)
#define DKS_WARPS      8
#define DKS_THREADS    (DKS_WARPS * 32)
#define DKS_PTHREADS   (DKS_THREADS - 32)         // cp.async producers: warps 0..6
#define DKS_ARRIVE_TID (DKS_THREADS - 32)         // warp 7 lane 0: tile-counter RMW + acquire
#define DKS_MAXMEM     4
#define DKS_MAXENT     24                         // A entries per CTA (host plan checks; kernel traps)
#define DKS_FX_B       20                         // fixup partial loads in flight per lane per batch
#define DKS_AB         8                          // A-prologue tasks per warp per load batch
#define DKS_F_SUH      1                          // A from x via the fused suh (else copy staged xh)
#define DKS_F_FIX      2                          // in-launch split-K fixup -> y (else partials only)
#define DKS_F_SILU     4                          // x := f16(silu(x) * x2) before the suh

// Launch parameter (by value, __grid_constant__). Mirrored by DksMemH / DksJobH (#[repr(C)]) in
// src/exl3_forward.rs — the static_asserts below and the Rust const asserts pin the layout.
struct DksMem {
    unsigned long long tr;    // const uint16_t* trellis, [K/16][N/16] blocks of 16*BITS u16
    unsigned long long suh;   // const __half*   input scale [K]                    (DKS_F_SUH)
    unsigned long long svh;   // const __half*   output scale [N]                   (DKS_F_FIX)
    unsigned long long xh;    // const __half*   staged H(x*suh) [M][K]             (!DKS_F_SUH)
    unsigned long long ws;    // float*          partials [ks][M][N]
    unsigned long long y;     // __half*         output [M][N]                      (DKS_F_FIX)
    unsigned long long cnt;   // unsigned*       tile arrival counters [N/128], 0 at rest (DKS_F_FIX)
    int n, ks, off, rsv;      // N, split count, first global item of this member, 0
};
struct DksJob {
    DksMem mb[DKS_MAXMEM];
    unsigned long long x;     // const __half* input [M][K]                         (DKS_F_SUH)
    unsigned long long x2;    // const __half* second input [M][K]                  (DKS_F_SILU)
    int nmem, M, K, n_items, flags, rsv;
};
static_assert(sizeof(DksMem) == 72, "DksMem layout (mirrored by DksMemH in src/exl3_forward.rs)");
static_assert(sizeof(DksJob) == DKS_MAXMEM * 72 + 16 + 24, "DksJob layout (mirrored by DksJobH)");

struct DksMt  { int g0, cnt, per, ebase; };            // per member: first item, #items, A period, entry base
struct DksEnt { int a_off, c0, w, hb, t_end, j; };     // per A entry: halves offset, slab col0, width,
                                                       // hb0 | nhb << 16, cumulative task end, member
struct DksCur { int g, j, tile, slab, q, nq, chunk; }; // (item, slot) cursor

__host__ __device__ __forceinline__ int dks_gcd(int a, int b) {
    while (b != 0) { const int t = a % b; a = b; b = t; }
    return a;
}

// Member of global item g (members are contiguous item ranges, mb[0].off == 0).
__device__ __forceinline__ int dks_member(const DksJob& job, int g) {
    int j = 0;
    #pragma unroll
    for (int i = 1; i < DKS_MAXMEM; ++i)
        if (i < job.nmem && g >= job.mb[i].off) j = i;
    return j;
}

__device__ __forceinline__ void dks_seek(const DksJob& job, int KB, DksCur& c) {
    c.j = dks_member(job, c.g);
    const int ks = job.mb[c.j].ks;
    const int it = c.g - job.mb[c.j].off;
    c.tile = it / ks;
    c.slab = it - c.tile * ks;
    c.q = 0;
    c.chunk = KB / ks;
    c.nq = c.chunk / DKS_KSTEP;
}

__device__ __forceinline__ void dks_advance(const DksJob& job, int KB, int G, int n_items, DksCur& c) {
    if (++c.q == c.nq) {
        c.g += G;
        if (c.g < n_items) dks_seek(job, KB, c);
    }
}

// One ring slot of item p: DKS_KSTEP k16 rows x the tile's 8 trellis blocks (KROW contiguous bytes
// per row), 16-B chunks, L2 evict_first (the weights are streamed once per call).
template <int BITS, int CPT>
__device__ __forceinline__ void dks_issue(const DksJob& job, const DksCur& p, unsigned slot_u,
                                          const int (&c_row)[CPT], const int (&c_byte)[CPT],
                                          const bool (&c_live)[CPT], unsigned long long pol) {
    constexpr int BB = 32 * BITS;
    constexpr int KROW = DKS_WARPS * BB;
    const long long nb = job.mb[p.j].n >> 4;
    const long long rowB = nb * BB;
    const long long kb0 = (long long)p.slab * p.chunk + p.q * DKS_KSTEP;
    const unsigned char* base = (const unsigned char*)job.mb[p.j].tr + (kb0 * nb + p.tile * 8) * BB;
    #pragma unroll
    for (int i = 0; i < CPT; ++i)
        if (c_live[i]) lmh_cp16_ef(slot_u + (unsigned)(c_row[i] * KROW + c_byte[i]),
                                   base + c_row[i] * rowB + c_byte[i], pol);
}

// Tile-counter arrival (release) for item g; returns the pre-add value.
__device__ __forceinline__ unsigned dks_arrive(const DksJob& job, int g) {
    const int j = dks_member(job, g);
    const int tile = (g - job.mb[j].off) / job.mb[j].ks;
    unsigned* ctr = (unsigned*)job.mb[j].cnt + tile;
    unsigned prev;
    asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(prev) : "l"(ctr) : "memory");
    return prev;
}

// Split-K fixup of item g's tile (called by the whole CTA once the last arrival is acquired):
// exl3_ks_combine's ascending fp32 chain from a 0.0f seed + exl3_had_svh_dev's (row, 128-block)
// warp layout, butterfly and f16_rn(v * svh). Warp w owns rows w, w+8 (< M): warp-uniform.
__device__ __forceinline__ void dks_fixup(const DksJob& job, int g, int M, int warp, int lane) {
    const int j = dks_member(job, g);
    const int N = job.mb[j].n, ks = job.mb[j].ks;
    const int tile = (g - job.mb[j].off) / ks;
    const int col = tile * 128 + lane * 4;
    const float* ws = (const float*)job.mb[j].ws;
    const __half* svh = (const __half*)job.mb[j].svh;
    __half* y = (__half*)job.mb[j].y;
    const size_t sstride = (size_t)M * N;
    for (int r = warp; r < M; r += DKS_WARPS) {
        const uint2 sv = __ldg((const uint2*)(svh + col));       // independent of the partials
        const float* p = ws + (size_t)r * N + col;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int s0 = 0; s0 < ks; s0 += DKS_FX_B) {
            // The batch's loads first (UNpredicated: an index past ks re-reads slab ks-1, whose
            // value is then not added — a predicated load + zero-init made ptxas reuse one
            // destination register and serialize the batch) ...
            float4 q[DKS_FX_B];
            #pragma unroll
            for (int b = 0; b < DKS_FX_B; ++b)
                q[b] = __ldcg((const float4*)(p + (size_t)min(s0 + b, ks - 1) * sstride));
            #pragma unroll
            for (int b = 0; b < DKS_FX_B; ++b)                    // ... then the adds, ascending s
                if (s0 + b < ks) { a0 += q[b].x; a1 += q[b].y; a2 += q[b].z; a3 += q[b].w; }
        }
        float v[4];
        v[0] = __half2float(__float2half_rn(a0));
        v[1] = __half2float(__float2half_rn(a1));
        v[2] = __half2float(__float2half_rn(a2));
        v[3] = __half2float(__float2half_rn(a3));
        exl3_had128(v, lane);
        const __half* sh = (const __half*)&sv;
        __half o[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) o[i] = __float2half_rn(v[i] * __half2float(sh[i]));
        *(uint2*)(y + (size_t)r * N + col) = *(const uint2*)o;
    }
}

// A prologue, fused-suh form: task (entry e, row r, 128-block hb) = one warp computes the whole
// Hadamard block of row r (every lane: the butterfly is warp-collective) and keeps the columns
// inside the entry's slab [c0, c0 + w). DKS_AB tasks' loads are issued before any math.
__device__ __forceinline__ void dks_stage_a_suh(const DksJob& job, const DksEnt* ent, int ne, __half* A,
                                                int K, int warp, int lane) {
    const int T = ne > 0 ? ent[ne - 1].t_end : 0;
    const bool silu = (job.flags & DKS_F_SILU) != 0;
    const __half* x = (const __half*)job.x;
    const __half* x2 = (const __half*)job.x2;
    for (int t0 = warp; t0 < T; t0 += DKS_WARPS * DKS_AB) {
        uint2 xv[DKS_AB], x2v[DKS_AB], sv[DKS_AB];
        int ei[DKS_AB], rr[DKS_AB], cc[DKS_AB];
        #pragma unroll
        for (int b = 0; b < DKS_AB; ++b) {
            const int t = t0 + b * DKS_WARPS;                      // warp-uniform
            ei[b] = -1; rr[b] = 0; cc[b] = 0;
            xv[b] = make_uint2(0u, 0u); x2v[b] = make_uint2(0u, 0u); sv[b] = make_uint2(0u, 0u);
            if (t < T) {
                int e = 0;
                while (t >= ent[e].t_end) ++e;
                const int tb = e > 0 ? ent[e - 1].t_end : 0;
                const int nhb = ent[e].hb >> 16, hb0 = ent[e].hb & 0xFFFF;
                const int u = t - tb;
                const int r = u / nhb;
                const int col = (hb0 + (u - r * nhb)) * 128 + lane * 4;
                const __half* suh = (const __half*)job.mb[ent[e].j].suh;
                xv[b] = *(const uint2*)(x + (size_t)r * K + col);
                if (silu) x2v[b] = *(const uint2*)(x2 + (size_t)r * K + col);
                sv[b] = __ldg((const uint2*)(suh + col));
                ei[b] = e; rr[b] = r; cc[b] = col;
            }
        }
        #pragma unroll
        for (int b = 0; b < DKS_AB; ++b) {
            if (ei[b] < 0) continue;                               // warp-uniform
            const __half* xh4 = (const __half*)&xv[b];
            const __half* x24 = (const __half*)&x2v[b];
            const __half* sh4 = (const __half*)&sv[b];
            float xs[4], ss[4], v[4];
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                // SILU: xq_silu_mul's element (stored f16, reloaded by exl3_had_suh) verbatim.
                xs[q] = silu ? xq_h2f(xq_f2h(xq_silu(xq_h2f(xh4[q])) * xq_h2f(x24[q]))) : xq_h2f(xh4[q]);
                ss[q] = xq_h2f(sh4[q]);
            }
            wp20_suh_had(xs, ss, v, lane);
            const DksEnt& en = ent[ei[b]];
            const int cl = cc[b] - en.c0;
            if (cl >= 0 && cl < en.w) {
                __half o[4];
                #pragma unroll
                for (int q = 0; q < 4; ++q) o[q] = xq_f2h(v[q]);
                *(uint2*)(A + en.a_off + rr[b] * (en.w + 8) + cl) = *(const uint2*)o;
            }
        }
    }
}

// A prologue, staged-xh form (GB10_W4DENSE_SUH=0): copy the slab columns of every entry.
__device__ __forceinline__ void dks_stage_a_copy(const DksJob& job, const DksEnt* ent, int ne, __half* A,
                                                 int M, int K, int tid) {
    for (int e = 0; e < ne; ++e) {
        const DksEnt en = ent[e];
        const __half* xh = (const __half*)job.mb[en.j].xh;
        const int v8 = en.w >> 3;
        for (int u = tid; u < M * v8; u += DKS_THREADS) {
            const int r = u / v8, c8 = u - r * v8;
            *(uint4*)(A + en.a_off + r * (en.w + 8) + c8 * 8) =
                *(const uint4*)(xh + (size_t)r * K + en.c0 + c8 * 8);
        }
    }
}

template <int BITS, int NST>
__device__ __forceinline__ void exl3_dks_body(const DksJob& job) {
    extern __shared__ __align__(16) unsigned char dks_sm[];
    __shared__ DksMt s_mt[DKS_MAXMEM];
    __shared__ DksEnt s_ent[DKS_MAXENT];
    __shared__ int s_nent;
    __shared__ int s_fix[2];
    constexpr int BB   = 32 * BITS;                 // bytes per 16x16 trellis block
    constexpr int KROW = DKS_WARPS * BB;            // bytes per k16 row of one 128-col tile
    constexpr int STB  = DKS_KSTEP * KROW;          // bytes per ring slot
    constexpr int CPR  = KROW / 16;                 // 16-B chunks per k16 row
    constexpr int NCH  = STB / 16;                  // 16-B chunks per slot
    constexpr int CPT  = (NCH + DKS_PTHREADS - 1) / DKS_PTHREADS;
    constexpr int WW   = 8 * BITS;                  // ring words per block
    static_assert(NST >= 2, "ring needs >= 2 slots");
    static_assert(BB % 16 == 0 && KROW % 16 == 0, "16-B chunking");

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gid = lane >> 2, tig = lane & 3;
    const int G = (int)gridDim.x, cta = (int)blockIdx.x;
    const int M = job.M, K = job.K, KB = K >> 4, n_items = job.n_items, flags = job.flags;
    const bool fix_on = (flags & DKS_F_FIX) != 0;
    if (cta >= n_items) {                           // CTA-uniform: no item for this CTA
        XQ_PDL_ENTRY();
        return;
    }
    unsigned char* ring = dks_sm;
    __half* A = (__half*)(dks_sm + NST * STB);
    const unsigned ring_u = (unsigned)__cvta_generic_to_shared(ring);

    // Host/device layout tripwires (the TRIPWIRE_SPEC §3.4 class): trap instead of overrunning.
    if (tid == 0) {
        unsigned dsz;
        asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dsz));
        if (dsz < (unsigned)(NST * STB)) __trap();
        if (M < 1 || M > 16 || job.nmem < 1 || job.nmem > DKS_MAXMEM || job.mb[0].off != 0 || (K & 127) != 0) __trap();
    }

    unsigned long long pol;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));

    // ---- per-thread slot chunk geometry: chunk c -> (k16 row, byte in the row span); warps 0..6.
    int c_row[CPT], c_byte[CPT];
    bool c_live[CPT];
    #pragma unroll
    for (int i = 0; i < CPT; ++i) {
        const int c = tid + i * DKS_PTHREADS;
        c_live[i] = tid < DKS_PTHREADS && c < NCH;
        c_row[i] = c / CPR;
        c_byte[i] = (c - (c / CPR) * CPR) * 16;
    }

    // ---- ring prologue: slots 0..NST-2 of this CTA's sequence (static weights: before the PDL wait)
    DksCur P;
    P.g = cta;
    dks_seek(job, KB, P);
    int p_slot = 0;
    #pragma unroll 1
    for (int s = 0; s < NST - 1; ++s) {
        if (P.g < n_items) {
            dks_issue<BITS, CPT>(job, P, ring_u + (unsigned)(p_slot * STB), c_row, c_byte, c_live, pol);
            dks_advance(job, KB, G, n_items, P);
            if (++p_slot == NST) p_slot = 0;
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }

    // ---- this CTA's member table + A entries (host mirror: dks_cta_plan, src/exl3_forward.rs)
    if (tid < job.nmem) {
        const int j = tid;
        const int off = job.mb[j].off, ks = job.mb[j].ks;
        const int end = (j + 1 < job.nmem) ? job.mb[j + 1].off : n_items;
        if (ks < 1 || (KB % ks) != 0 || ((KB / ks) % DKS_KSTEP) != 0 || (job.mb[j].n & 127) != 0 || end < off) __trap();
        const int g0 = off + (((cta - off) % G) + G) % G;      // first item of member j on this CTA
        s_mt[j].g0 = g0;
        s_mt[j].cnt = g0 < end ? (end - 1 - g0) / G + 1 : 0;
        s_mt[j].per = ks / dks_gcd(G, ks);                     // its slabs repeat with this period
    }
    __syncthreads();
    if (tid == 0) {
        int ne = 0, a_off = 0, t = 0;
        for (int j = 0; j < job.nmem; ++j) {
            const int ks = job.mb[j].ks, W = (KB / ks) * 16;
            const int E = min(s_mt[j].per, s_mt[j].cnt);
            s_mt[j].ebase = ne;
            for (int e = 0; e < E; ++e) {
                if (ne >= DKS_MAXENT) __trap();
                const int slab = (s_mt[j].g0 - job.mb[j].off + e * G) % ks;
                const int c0 = slab * W;
                const int hb0 = c0 >> 7, nhb = ((c0 + W - 1) >> 7) - hb0 + 1;
                t += M * nhb;
                s_ent[ne].a_off = a_off; s_ent[ne].c0 = c0; s_ent[ne].w = W;
                s_ent[ne].hb = hb0 | (nhb << 16); s_ent[ne].t_end = t; s_ent[ne].j = j;
                a_off += (M * (W + 8) + 7) & ~7;
                ++ne;
            }
        }
        s_nent = ne;
        unsigned dsz;
        asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dsz));
        if (dsz < (unsigned)(NST * STB + a_off * 2)) __trap();
    }

    // ---- PDL: x / x2 / xh / ws / cnt / y are touched only past this point.
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    __syncthreads();
    int total = 0;                                  // this CTA's ring slots
    for (int j = 0; j < job.nmem; ++j) total += s_mt[j].cnt * ((KB / job.mb[j].ks) / DKS_KSTEP);
    if (flags & DKS_F_SUH) dks_stage_a_suh(job, s_ent, s_nent, A, K, warp, lane);
    else                   dks_stage_a_copy(job, s_ent, s_nent, A, M, K, tid);
    // (A visibility: the main loop's first __syncthreads)

    // ---- per-lane decode constants (W3/LMH): quad q (= n8 half h) holds t = 8*lane + 4q + s.
    int qa[2], qb[2];
    unsigned dsh[8];
    #pragma unroll
    for (int q = 0; q < 2; ++q) {
        const int t0 = 8 * lane + 4 * q;
        const int b0 = (t0 + 257) * BITS - 16;
        const int b2 = (t0 + 3 + 257) * BITS;
        const int I0 = b0 >> 5;
        const int I2 = (b2 - 1) >> 5;
        const int s2 = ((I2 + 1) << 5) - b2;
        qa[q] = I0 % WW;
        qb[q] = I2 % WW;
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int sh = s2 + (3 - s) * BITS;
            dsh[q * 4 + s] = (unsigned)(sh & 31) | (sh >= 32 ? 32u : 0u);
        }
    }
    const __half2 k_inv2  = __half2half2(__ushort_as_half((unsigned short)0x1EEE));
    const __half2 k_bias2 = __half2half2(__ushort_as_half((unsigned short)0xC931));

    float acc[2][4];
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[h][r] = 0.0f;
    const bool l0 = gid < M, l1 = (gid + 8) < M;

    DksCur C;
    C.g = cta;
    dks_seek(job, KB, C);
    int c_slot = 0, srow = 8;
    const __half* ar0 = A;
    int arr_g = -1;                                 // item whose partials were stored last iteration
    int t0_g = -1;                                  // DKS_ARRIVE_TID: item whose RMW is in flight
    unsigned t0_prev = 0u;

    #pragma unroll 1
    for (int k = 0; k < total + 2; ++k) {           // + 2: arrival / fixup flush of the last items
        // slot k landed (<= NST-2 newer groups pending) and is visible CTA-wide; every warp is past
        // slot k-1, so its ring slot may be refilled.
        if (k < total) asm volatile("cp.async.wait_group %0;\n" :: "n"(NST - 2) : "memory");
        __syncthreads();
        const int fix = (fix_on && k > 0) ? s_fix[(k - 1) & 1] : -1;   // decided last iteration
        if (fix_on && tid == DKS_ARRIVE_TID && arr_g >= 0) {           // stores -> bar -> release
            t0_prev = dks_arrive(job, arr_g);
            t0_g = arr_g;
        }
        if (k < total) {
            if (P.g < n_items) {
                dks_issue<BITS, CPT>(job, P, ring_u + (unsigned)(p_slot * STB), c_row, c_byte, c_live, pol);
                dks_advance(job, KB, G, n_items, P);
                if (++p_slot == NST) p_slot = 0;
            }
            asm volatile("cp.async.commit_group;\n" ::: "memory");
        }
        if (fix >= 0) dks_fixup(job, fix, M, warp, lane);             // CTA-uniform
        int end_g = -1;
        if (k < total) {
            if (C.q == 0) {                                            // item start: its A entry
                const DksMt mt = s_mt[C.j];
                const DksEnt en = s_ent[mt.ebase + ((C.g - mt.g0) / G) % mt.per];
                srow = en.w + 8;
                ar0 = A + en.a_off + gid * srow + 2 * tig;
            }
            const unsigned char* slot = ring + c_slot * STB + warp * BB;
            #pragma unroll
            for (int g = 0; g < DKS_KSTEP; ++g) {
                const uint32_t* w = (const uint32_t*)(slot + g * KROW);
                uint32_t bfrag[2][2];
                #pragma unroll
                for (int q = 0; q < 2; ++q) {
                    const uint32_t wa = w[qa[q]];
                    const uint32_t wb = w[qb[q]];
                    uint32_t sum[4];
                    #pragma unroll
                    for (int s = 0; s < 4; ++s) {
                        const unsigned d = dsh[q * 4 + s];
                        const uint32_t lo = (d & 32u) ? wa : wb;
                        const unsigned idx = __funnelshift_r(lo, wa, d & 31u) & 0xFFFFu;
                        sum[s] = __dp4a(idx * EXL3_MUL1, 0x01010101u, 0x00006400u);
                    }
                    #pragma unroll
                    for (int p = 0; p < 2; ++p) {
                        const __half2 hv = __halves2half2(__ushort_as_half((unsigned short)sum[2 * p]),
                                                          __ushort_as_half((unsigned short)sum[2 * p + 1]));
                        const __half2 wv = __hfma2(hv, k_inv2, k_bias2);
                        bfrag[q][p] = *reinterpret_cast<const uint32_t*>(&wv);
                    }
                }
                const int col = (C.q * DKS_KSTEP + g) * 16;
                uint32_t a0 = 0u, a1 = 0u, a2 = 0u, a3 = 0u;
                if (l0) {
                    a0 = *(const uint32_t*)(ar0 + col);
                    a2 = *(const uint32_t*)(ar0 + col + 8);
                }
                if (l1) {
                    a1 = *(const uint32_t*)(ar0 + 8 * srow + col);
                    a3 = *(const uint32_t*)(ar0 + 8 * srow + col + 8);
                }
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
            if (++c_slot == NST) c_slot = 0;
            if (C.q == C.nq - 1) {
                // ---- item done: fp32 partials ws[slab][row][col] (rows >= M skipped) — the old
                // kernel's values at its addresses (float2 = its two scalar stores).
                const int N = job.mb[C.j].n;
                float* base = (float*)job.mb[C.j].ws + (size_t)C.slab * M * N;
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int col = C.tile * 128 + 16 * warp + 8 * h + 2 * tig;
                    if (l0) *(float2*)&base[(size_t)gid * N + col] = make_float2(acc[h][0], acc[h][1]);
                    if (l1) *(float2*)&base[(size_t)(gid + 8) * N + col] = make_float2(acc[h][2], acc[h][3]);
                    #pragma unroll
                    for (int r = 0; r < 4; ++r) acc[h][r] = 0.0f;
                }
                end_g = C.g;
                C.g += G;
                if (C.g < n_items) dks_seek(job, KB, C);
            } else {
                ++C.q;
            }
        }
        if (fix_on && tid == DKS_ARRIVE_TID) {
            int res = -1;
            if (t0_g >= 0) {
                const int j = dks_member(job, t0_g);
                if (t0_prev == (unsigned)(job.mb[j].ks - 1)) {         // the tile's last arrival
                    unsigned* ctr = (unsigned*)job.mb[j].cnt + (t0_g - job.mb[j].off) / job.mb[j].ks;
                    asm volatile("fence.acq_rel.gpu;" ::: "memory");            // acquire
                    asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(ctr), "r"(0u) : "memory");
                    res = t0_g;
                }
                t0_g = -1;
            }
            s_fix[k & 1] = res;                     // read by every thread after the next barrier
        }
        arr_g = end_g;
    }
}

// Entries exl3_dks_b{BITS}_s{NST}: 5-bit (the served dense chains) and 4-bit (MTP fc) at ring
// depth 4/6/8 (GB10_W4DENSE_NST), 3-bit at 6 for coverage. 1 CTA/SM x 8 warps (the LMH class).
#define DKS_ENTRY(B, NS)                                                                        \
extern "C" __global__ void __launch_bounds__(DKS_THREADS, 1)                                    \
exl3_dks_b##B##_s##NS(const __grid_constant__ DksJob job) {                                     \
    exl3_dks_body<B, NS>(job);                                                                  \
}
DKS_ENTRY(5, 4)
DKS_ENTRY(5, 6)
DKS_ENTRY(5, 8)
DKS_ENTRY(4, 4)
DKS_ENTRY(4, 6)
DKS_ENTRY(4, 8)
DKS_ENTRY(3, 6)

// =============================================================================
// S-A3-f-d — MTP k-chain VERIFY (width m = k+1 rows of ONE slot, positions
// p..p+m-1). Transcribed structure from the reference implementation recipe's device-resident
// draft-chain patch (0003) + pruned draft head (0002); every stateful op is a
// CHAIN variant of the decode kernel: the group's rows are processed
// SEQUENTIALLY so each row sees the previous rows' committed state, with the
// SAME device functions / reduction orders as the m=1 decode path => verified
// rows are BIT-IDENTICAL to sequential decode (the losslessness contract).
// Rollback model: the verify runs against SHADOW state buffers (copied from
// the live state before the launch); after the on-device accept computes `a`
// (accepted draft count), the commit kernels re-run rows 0..a from the LIVE
// state (bit-identical replay) so rows a+1..m-1 never commit.
// =============================================================================

// ---- chained full-attention decode: grid (nh, m) — m rides gridDim.y (the
// 12-arg cap); rows loop IN-BLOCK so row r's softmax reads rows <= r's K/V
// already written by THIS block (no cross-block race, no separate pre-pass).
// Same shared fns (xq_k_norm_rope / xq_attn_softmax) and the same per-row
// cache-write + ascending-two-pass order as xq_attn_decode.
// ---- chained verify attention (rows = one sequence's candidate tokens, one slot).
// S-A3-l: the OLD body ran xq_attn_softmax once PER ROW — the per-position
// halving-tree sync chain ran m times over nearly the same prefix. The S-A3-l
// budget priced that at +24.7 ms of the +55.6 ms m=1->m=6 overcost (30.1 ms of
// the 101.2 ms m=6 round). The BATCHED body runs the two passes ONCE with every
// row's score partials segmented in shared memory (thread tid = dim tid of EVERY
// row); the halving tree steps all segments under the same syncs.
// Bit contract: phase 1 is the old per-row body verbatim and each row's scan ops
// are xq_attn_softmax's exact per-row sequence (same tree within the segment,
// same ascending-t passes, same fmaxf/__expf ops) => row 0 stays bit-identical
// to xq_attn_decode (probe-exl3-binv). acc/denom/mx are TEMPLATE arrays (G
// compile-time): a runtime-m per-thread array lands in local memory (AGENTS §4).
__device__ void xq_attn_decode_group_loop(__half* __restrict__ attn,
                                          const __half* __restrict__ qg,
                                          const __half* __restrict__ kp,
                                          const __half* __restrict__ vp,
                                          float* __restrict__ kcvc,
                                          const __half* __restrict__ qnkn,
                                          const float* __restrict__ cossin,
                                          const int* __restrict__ slotpos,
                                          int nh_packed, int hd_packed,
                                          int max_pos, float eps) {
    const int nh = nh_packed & 0xFFFF;
    const int nkv = (nh_packed >> 16) & 0xFFFF;
    const int hd = hd_packed & 0xFFFF;
    const int rdim = (unsigned)hd_packed >> 16;
    const int head = blockIdx.x;          // grid (nh, m)
    const int m = gridDim.y;
    const int tid = threadIdx.x;
    const int kvh = head / (nh / nkv);
    // issue #35: rows of a packed verify can be DIFFERENT lanes — resolve the slot per row.
    // (When every row shares one slot, as a plain verify does, this yields the same value
    // the old single read produced, so decode/verify words are unchanged.)
    __shared__ float red[1024];
    __shared__ float qbuf[1024];
    __shared__ float kbuf[1024];
    for (int r = 0; r < m; r++) {
        const int p = slotpos[r * 2 + 1];
        const int slot = slotpos[r * 2 + 0];
        const float* c = cossin + (long long)r * 2 * rdim;
        const float* s = c + rdim;
        const __half* qn_w = qnkn;
        const __half* kn_w = qnkn + hd;
        // ---- q: rmsnorm (1+w), rope (shared order with xq_attn_decode)
        float qv = xq_h2f(qg[(long long)r * nh * hd * 2 + (long long)head * hd * 2 + tid]);
        red[tid] = qv * qv; __syncthreads();
        for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) { if (tid < s2) red[tid] += red[tid + s2]; __syncthreads(); }
        qv *= rsqrtf(red[0] / (float)hd + eps) * (1.0f + xq_h2f(qn_w[tid]));
        qbuf[tid] = qv; __syncthreads();
        if (tid < rdim / 2) {
            float x1 = qbuf[tid], x2 = qbuf[tid + rdim / 2];
            qbuf[tid] = x1 * c[tid] - x2 * s[tid];
            qbuf[tid + rdim / 2] = x2 * c[tid] + x1 * s[tid];
        }
        __syncthreads();
        qv = xq_kv_rot(qbuf[tid], xq_kv_fmt(max_pos), tid & 31);   // WP25 q8: rotated basis
        // ---- k: rmsnorm (1+w), rope, cache write (bit-identical K words)
        xq_k_norm_rope(kbuf, red, kp + (long long)r * nkv * hd + (long long)kvh * hd,
                       kn_w, c, s, tid, hd, rdim, eps);
        __syncthreads();
        const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
        xq_kv_put(kc.k + (long long)p * kc.rb, kc.fmt, hd, tid, kbuf[tid]);
        xq_kv_put(kc.v + (long long)p * kc.rb, kc.fmt, hd, tid, xq_h2f(vp[(long long)r * nkv * hd + (long long)kvh * hd + tid]));
        __syncthreads();
        // ---- shared two-pass softmax over [0..p] (rows <= r written above)
        float acc, denom;
        xq_attn_softmax(red, qv, kc, p, tid, hd, acc, denom);
        float g = xq_h2f(qg[(long long)r * nh * hd * 2 + (long long)head * hd * 2 + hd + tid]);
        const float o = xq_kv_rot(acc / denom, kc.fmt, tid & 31);
        attn[(long long)r * nh * hd + (long long)head * hd + tid] =
            xq_f2h(o * xq_sig(g));
        __syncthreads();
    }
}

// ---- S-A3-m attention v2: one block per (head, row) — grid (nh, m), blockIdx.y
// = row. (The S-A3-l body ignored blockIdx.y: every y-block recomputed ALL rows,
// m-fold redundant.) Block r writes the K/V of rows 0..r itself (identical words
// to every other block writing them — benign) so its own scan only reads cache
// words it wrote before a __syncthreads; then row r's q and ONE scan over [0..p_r]
// through xq_attn_scan (bit-identical to xq_attn_softmax — row r == the decode of
// that token, probe-exl3-binv attention section). Phase-1 order per row is the
// old body verbatim.
__device__ void xq_attn_decode_group_v2(__half* __restrict__ attn,
                                        const __half* __restrict__ qg,
                                        const __half* __restrict__ kp,
                                        const __half* __restrict__ vp,
                                        float* __restrict__ kcvc,
                                        const __half* __restrict__ qnkn,
                                        const float* __restrict__ cossin,
                                        const int* __restrict__ slotpos,
                                        int nh_packed, int hd_packed,
                                        int max_pos, float eps) {
    const int nh = nh_packed & 0xFFFF;
    const int nkv = (nh_packed >> 16) & 0xFFFF;
    const int hd = hd_packed & 0xFFFF;
    const int rdim = (unsigned)hd_packed >> 16;
    const int head = blockIdx.x;
    const int r = blockIdx.y;
    const int tid = threadIdx.x;
    const int kvh = head / (nh / nkv);
    // issue #35: rows of a packed verify can be DIFFERENT lanes, so phase 1 writes each
    // row K/V into that row own slot and the scan reads row r slot. When every row shares
    // one slot (a plain verify) every kc equals the old single kc: words unchanged.
    const __half* qn_w = qnkn;
    const __half* kn_w = qnkn + hd;
    __shared__ float red[1024];
    __shared__ float qbuf[1024];
    __shared__ float kbuf[1024];
    __shared__ float wbuf[XQ_SCAN_T];
    __shared__ float wm[32];
    // ---- phase 1: K/V writes for rows 0..r (xq_k_norm_rope = the shared K words)
    for (int rr = 0; rr <= r; rr++) {
        const int p = slotpos[rr * 2 + 1];
        const XqKv kcw = xq_kv_make(kcvc, max_pos, hd, nkv, slotpos[rr * 2 + 0], kvh);
        const float* c = cossin + (long long)rr * 2 * rdim;
        const float* s = c + rdim;
        xq_k_norm_rope(kbuf, red, kp + (long long)rr * nkv * hd + (long long)kvh * hd,
                       kn_w, c, s, tid, hd, rdim, eps);
        __syncthreads();
        xq_kv_put(kcw.k + (long long)p * kcw.rb, kcw.fmt, hd, tid, kbuf[tid]);
        xq_kv_put(kcw.v + (long long)p * kcw.rb, kcw.fmt, hd, tid, xq_h2f(vp[(long long)rr * nkv * hd + (long long)kvh * hd + tid]));
        __syncthreads();
    }
    // ---- row r's q: rmsnorm (1+w), rope (shared order with xq_attn_decode)
    const int p = slotpos[r * 2 + 1];
    const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slotpos[r * 2 + 0], kvh);
    const float* c = cossin + (long long)r * 2 * rdim;
    const float* s = c + rdim;
    float qv = xq_h2f(qg[(long long)r * nh * hd * 2 + (long long)head * hd * 2 + tid]);
    red[tid] = qv * qv; __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) { if (tid < s2) red[tid] += red[tid + s2]; __syncthreads(); }
    qv *= rsqrtf(red[0] / (float)hd + eps) * (1.0f + xq_h2f(qn_w[tid]));
    qbuf[tid] = qv; __syncthreads();
    if (tid < rdim / 2) {
        float x1 = qbuf[tid], x2 = qbuf[tid + rdim / 2];
        qbuf[tid] = x1 * c[tid] - x2 * s[tid];
        qbuf[tid + rdim / 2] = x2 * c[tid] + x1 * s[tid];
    }
    __syncthreads();
    if (kc.fmt >= XQ_KV_Q8) {   // WP25 q8: q into the rotated basis (own element; then publish)
        qbuf[tid] = xq_h32r(qbuf[tid], tid & 31);
        __syncthreads();
    }
    float acc, denom;
    xq_attn_scan_hd(qbuf, kc, p, tid, hd, wbuf, wm, acc, denom);
    float g = xq_h2f(qg[(long long)r * nh * hd * 2 + (long long)head * hd * 2 + hd + tid]);
    const float o = xq_kv_rot(acc / denom, kc.fmt, tid & 31);
    attn[(long long)r * nh * hd + (long long)head * hd + tid] =
        xq_f2h(o * xq_sig(g));
}

extern "C" __global__ void xq_attn_decode_group(__half* __restrict__ attn,
                                                const __half* __restrict__ qg,
                                                const __half* __restrict__ kp,
                                                const __half* __restrict__ vp,
                                                float* __restrict__ kcvc,
                                                const __half* __restrict__ qnkn,
                                                const float* __restrict__ cossin,
                                                const int* __restrict__ slotpos,
                                                int nh_packed, int hd_packed,
                                                int max_pos, float eps) {
    XQ_PDL_ENTRY();
    const int hd = hd_packed & 0xFFFF;
    if (hd == 128 || hd == 256 || hd == 512) {
        xq_attn_decode_group_v2(attn, qg, kp, vp, kcvc, qnkn, cossin, slotpos,
                                nh_packed, hd_packed, max_pos, eps);
    } else if (blockIdx.y == 0) {
        // other head dims: the verbatim per-row loop, once per head
        xq_attn_decode_group_loop(attn, qg, kp, vp, kcvc, qnkn, cossin, slotpos,
                                  nh_packed, hd_packed, max_pos, eps);
    }
}

// Probe-only reference (probe-exl3-binv attention section): the pre-S-A3-l
// per-row loop over xq_attn_softmax — a different CLASS (smem sync-tree) from the
// v2 warp-tree scan. Launch grid (nh, m): the loop takes m = gridDim.y and every
// y-block runs all rows (identical writes).
extern "C" __global__ void xq_attn_decode_group_ref(__half* __restrict__ attn,
                                                    const __half* __restrict__ qg,
                                                    const __half* __restrict__ kp,
                                                    const __half* __restrict__ vp,
                                                    float* __restrict__ kcvc,
                                                    const __half* __restrict__ qnkn,
                                                    const float* __restrict__ cossin,
                                                    const int* __restrict__ slotpos,
                                                    int nh_packed, int hd_packed,
                                                    int max_pos, float eps) {
    XQ_PDL_ENTRY();
    xq_attn_decode_group_loop(attn, qg, kp, vp, kcvc, qnkn, cossin, slotpos,
                              nh_packed, hd_packed, max_pos, eps);
}

