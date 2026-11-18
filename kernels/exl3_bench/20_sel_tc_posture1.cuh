// exl3_bench/20_sel_tc_posture1.cuh: POSTURE1 --exl3-sel-tc tensor-core prefill sparse gather, dense splitk4, sel combine.
// OPT-IN; NOT bitwise xq_attn_sel_pq8.
// Section 20 of 23 of exl3_bench.cu: included in order by exl3_bench.cu, one translation unit; text is moved verbatim.
// ---- POSTURE1 --exl3-sel-tc (OPT-IN; NOT bitwise xq_attn_sel_pq8): the PREFILL sparse gather on tensor cores.
// What runs today (SASS of xq_attn_sel_pq8, sm_121): the hot loop is ~171 issue slots per selected entry per warp
// (38 FFMA + 41 FMUL + 21 FADD + 18 SHFL + 6 MUFU for 3 heads x 8 dims: dot product, xor-tree reduction, online
// softmax, acc update), i.e. issue-bound at ~2.7 T MAC/s (38.7 ms per 4,095-row layer call, 18.6% of a sparse chunk).
// Here the 12 query heads of one (row, kv head) ARE the M dimension of an m16n8k16 f16 MMA (rows 12..15 zero): each
// selected K / V row is converted once per CTA to f16 in a swizzled smem tile (xq_kv_ld8: the cache-basis values
// pq8 uses), S = Q.K^T and O += P.V run on the tensor cores with f32 accumulation, the online softmax runs once per
// 32-key tile. Same grid (rows * nkv * nsplits), same split ranges, same selection entries (-1 skipped), same
// partial layout and semantics (pm = max scaled score, pl = sum exp(s - pm), pacc = sum exp(s - pm) v) -> the
// combine is unchanged. Differences from pq8: q, K, V, P rounded to f16 before the products; summation order.
// Block 128 = 4 warps: QK -> warp w scores keys 8w..8w+7 of the tile (full K=256); PV -> warp w owns dims 64w..64w+63.
#define XQ_SELTC_TE 32
#define XQ_SELTC_PLD 40   // Ps row stride (halves): 80 B rows keep the P ldmatrix conflict-free
extern "C" __global__ void __launch_bounds__(128)
xq_attn_sel_tc(float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc,
               const float* __restrict__ qstage, const float* __restrict__ kcvc, const int* __restrict__ sel,
               const int* __restrict__ pos_sel, const int* __restrict__ slotpos,
               int nh_packed, long long geom, int split_hd) {
    XQ_PDL_ENTRY();
    constexpr int HD = 256, TE = XQ_SELTC_TE;
    __shared__ __align__(16) __half Ks[TE * HD];
    __shared__ __align__(16) __half Vs[TE * HD];
    __shared__ __align__(16) __half Ps[16 * XQ_SELTC_PLD];
    __shared__ float Ss[16][TE + 1];
    __shared__ float s_al[16];
    __shared__ int s_ok[TE];
    const int nh = nh_packed & 0xFFFF;
    const int nkv = (nh_packed >> 16) & 0xFFFF;
    const int nsplits = (int)(geom & 0xFFFFF);
    const int sel_max = (int)((geom >> 20) & 0xFFF);
    const int max_pos = (int)(geom >> 32);
    const int split_len = split_hd & 0xFFFF;
    const int sp = blockIdx.x % nsplits;
    const int kvh = (blockIdx.x / nsplits) % nkv;
    const int row = blockIdx.x / (nsplits * nkv);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, tq = lane & 3;
    const int gqa = nh / nkv;
    const int slot = slotpos[row * 2 + 0];
    const XqKv kc = xq_kv_make(kcvc, max_pos, HD, nkv, slot, kvh);
    const int nsel = pos_sel[row] + 1;
    const int begin = sp * split_len;
    const int end = min((sp + 1) * split_len, nsel);
    const float rsq = rsqrtf((float)HD);
    // Q A-fragments (f16) of the 12 heads (rows g, g + 8; rows >= gqa are zero)
    const float* q0 = qstage + ((long long)row * nh + kvh * gqa + g) * HD;
    const bool r1ok = g + 8 < gqa;
    const float* q1 = qstage + ((long long)row * nh + kvh * gqa + (r1ok ? g + 8 : g)) * HD;
    unsigned qf[HD / 16][4];
    #pragma unroll
    for (int ks = 0; ks < HD / 16; ks++) {
        const int d = 16 * ks + 2 * tq;
        const float2 a = *reinterpret_cast<const float2*>(q0 + d), b = *reinterpret_cast<const float2*>(q0 + d + 8);
        const float2 c = *reinterpret_cast<const float2*>(q1 + d), e = *reinterpret_cast<const float2*>(q1 + d + 8);
        qf[ks][0] = xq_pftc_h2(a.x, a.y);
        qf[ks][1] = r1ok ? xq_pftc_h2(c.x, c.y) : 0u;
        qf[ks][2] = xq_pftc_h2(b.x, b.y);
        qf[ks][3] = r1ok ? xq_pftc_h2(e.x, e.y) : 0u;
    }
    float o[8][4];
    #pragma unroll
    for (int j = 0; j < 8; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    // softmax role: head sh = tid >> 3 (0..15), keys 4 * (tid & 7) .. + 3 of the tile
    const int sh = tid >> 3, sub = tid & 7;
    float m_h = -INFINITY, l_h = 0.f;
    // staging role: warp w loads the (w & 1 ? V : K) rows of keys (w >> 1) + 2i, i = 0..15; lane = 8-dim chunk.
    // The NEXT tile's raw bytes (8 codes + the group's f16 scale, pq8's XqQ8Raw) are loaded into registers before
    // the current tile's products, so the L2 latency hides behind them (one tile in flight).
    const int kvw = warp & 1, k0w = warp >> 1;
    const char* kvb = kvw ? kc.v : kc.k;
    const int* selrow = sel + (long long)row * sel_max;
    XqQ8Raw raw[TE / 2];
    unsigned okm = 0u;
    auto fetch = [&](int c0) {
        const int nn = min(TE, end - c0);
        okm = 0u;
        #pragma unroll
        for (int i = 0; i < TE / 2; i++) {
            const int key = k0w + 2 * i;
            const int t = key < nn ? selrow[c0 + key] : -1;
            if (t >= 0) { okm |= 1u << i; xq_q8_raw(raw[i], kvb + (long long)t * kc.rb, HD, lane * 8); }
        }
    };
    if (begin < end) fetch(begin);
    for (int cb = begin; cb < end; cb += TE) {
        const int n = min(TE, end - cb);
        __syncthreads();                                   // the previous tile's readers are done
        #pragma unroll
        for (int i = 0; i < TE / 2; i++) {                 // convert ONCE per CTA into the swizzled f16 tile
            const int key = k0w + 2 * i;
            float f[8];
            if (okm & (1u << i)) {
                xq_q8_deq8(f, raw[i]);
            } else {
                #pragma unroll
                for (int e = 0; e < 8; e++) f[e] = 0.f;
            }
            const int cs = (lane & 24) | ((lane & 7) ^ (key & 7));  // Swizzle<3,3,3> (xq_attn_pf_tc256)
            *reinterpret_cast<uint4*>((kvw ? Vs : Ks) + key * HD + (cs << 3)) =
                make_uint4(xq_pftc_h2(f[0], f[1]), xq_pftc_h2(f[2], f[3]), xq_pftc_h2(f[4], f[5]), xq_pftc_h2(f[6], f[7]));
            if (kvw == 0 && lane == 0) s_ok[key] = (okm >> i) & 1u;
        }
        __syncthreads();
        if (cb + TE < end) fetch(cb + TE);                // next tile's bytes in flight during the products
        // ===== S = Q.K^T for keys 8w..8w+7 (K[key][d] is the .col B operand: ldmatrix, no transpose) =====
        float s[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int ks = 0; ks < HD / 16; ks += 2) {
            const int key = 8 * warp + (lane & 7);
            const int chunk = 2 * ks + (lane >> 3);       // dims 16ks + 8 * (lane >> 3): two k16 steps per x4
            const int cs = (chunk & 24) | ((chunk & 7) ^ (key & 7));
            unsigned r0, r1, r2, r3;
            xq_pftc_ldm_x4(r0, r1, r2, r3, (unsigned)__cvta_generic_to_shared(Ks + key * HD + (cs << 3)));
            const unsigned b0[2] = {r0, r1}, b1[2] = {r2, r3};
            xq_pftc_mma(s, qf[ks], b0);
            xq_pftc_mma(s, qf[ks + 1], b1);
        }
        {
            const int k0 = 8 * warp + 2 * tq;
            const bool v0 = k0 < n && s_ok[k0], v1 = k0 + 1 < n && s_ok[k0 + 1];
            Ss[g][k0] = v0 ? s[0] * rsq : -INFINITY;
            Ss[g][k0 + 1] = v1 ? s[1] * rsq : -INFINITY;
            Ss[g + 8][k0] = v0 ? s[2] * rsq : -INFINITY;
            Ss[g + 8][k0 + 1] = v1 ? s[3] * rsq : -INFINITY;
        }
        __syncthreads();
        // ===== online softmax, once per tile and head =====
        {
            float v[4], tmax = -INFINITY;
            #pragma unroll
            for (int i = 0; i < 4; i++) { v[i] = sh < gqa ? Ss[sh][4 * sub + i] : -INFINITY; tmax = fmaxf(tmax, v[i]); }
            tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 1));
            tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 2));
            tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 4));
            const float m_new = fmaxf(m_h, tmax);
            const float al = (m_new == -INFINITY) ? 1.f : __expf(m_h - m_new);
            float ps = 0.f, p[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) { p[i] = (v[i] == -INFINITY) ? 0.f : __expf(v[i] - m_new); ps += p[i]; }
            ps += __shfl_xor_sync(0xffffffffu, ps, 1);
            ps += __shfl_xor_sync(0xffffffffu, ps, 2);
            ps += __shfl_xor_sync(0xffffffffu, ps, 4);
            l_h = l_h * al + ps;
            m_h = m_new;
            *reinterpret_cast<uint2*>(Ps + sh * XQ_SELTC_PLD + 4 * sub) = make_uint2(xq_pftc_h2(p[0], p[1]), xq_pftc_h2(p[2], p[3]));
            if (sub == 0) s_al[sh] = al;
        }
        __syncthreads();
        // ===== O(dims 64w..64w+63) = O * alpha + P.V (V via ldmatrix .trans) =====
        {
            const float a0 = s_al[g], a1 = s_al[g + 8];
            #pragma unroll
            for (int j = 0; j < 8; j++) { o[j][0] *= a0; o[j][1] *= a0; o[j][2] *= a1; o[j][3] *= a1; }
        }
        #pragma unroll
        for (int kq = 0; kq < TE / 16; kq++) {
            unsigned pa[4];
            {
                const int mi = lane >> 3, rr = (lane & 7) + 8 * (mi & 1), cc = 16 * kq + 8 * (mi >> 1);
                xq_pftc_ldm_x4(pa[0], pa[1], pa[2], pa[3], (unsigned)__cvta_generic_to_shared(Ps + rr * XQ_SELTC_PLD + cc));
            }
            #pragma unroll
            for (int jn = 0; jn < 4; jn++) {
                int lkey, ldim;
                if (lane < 8)       { lkey = 16 * kq + lane;            ldim = 64 * warp + 16 * jn; }
                else if (lane < 16) { lkey = 16 * kq + 8 + lane - 8;    ldim = 64 * warp + 16 * jn; }
                else if (lane < 24) { lkey = 16 * kq + lane - 16;       ldim = 64 * warp + 16 * jn + 8; }
                else                { lkey = 16 * kq + 8 + lane - 24;   ldim = 64 * warp + 16 * jn + 8; }
                const int chunk = ldim >> 3;
                const int cs = (chunk & 24) | ((chunk & 7) ^ (lkey & 7));
                unsigned r0, r1, r2, r3;
                xq_pftc_ldm_x4_trans(r0, r1, r2, r3, (unsigned)__cvta_generic_to_shared(Vs + lkey * HD + (cs << 3)));
                const unsigned ba[2] = {r0, r1}, bb[2] = {r2, r3};
                xq_pftc_mma(o[2 * jn], pa, ba);
                xq_pftc_mma(o[2 * jn + 1], pa, bb);
            }
        }
    }
    // partials (pq8's layout): pm / pl by the softmax role's first lane of each live head, pacc by the PV fragments
    if (sub == 0 && sh < gqa) {
        const long long hg = (long long)row * nh + kvh * gqa + sh;
        pm[hg * nsplits + sp] = m_h;
        pl[hg * nsplits + sp] = l_h;
    }
    #pragma unroll
    for (int hh = 0; hh < 2; hh++) {
        const int head = g + 8 * hh;
        if (head >= gqa) continue;
        const long long base = (((long long)row * nh + kvh * gqa + head) * nsplits + sp) * HD + 64 * warp + 2 * tq;
        #pragma unroll
        for (int j = 0; j < 8; j++)
            *reinterpret_cast<float2*>(pacc + base + 8 * j) = make_float2(o[j][2 * hh], o[j][2 * hh + 1]);
    }
}

// ---- POSTURE1 --exl3-sel-tc, union staging (xq_attn_sel_tcu): neighbouring query rows select almost the same keys
// (measured, SEL_UNION, Flash-Next 3.05: 4 rows select 1.05-1.15x the keys of one row), so ONE block serves R = 4
// consecutive rows of one kv head: it builds the union of their selected key blocks (indexer blocks of `ratio` tokens)
// with a per-block membership byte, stages each union K/V row ONCE (f16, swizzled) and runs the 48 query rows
// (4 tokens x 12 heads, token-major) through xq_attn_pf_tc256's MMA / online-softmax body: 6 warps, warp w = rows
// 16 (w >> 1) .. + 15 and output dims 128 (w & 1) .. + 127 (each warp of a pair scores the same 16 rows, so their
// softmax states are identical; the split halves the accumulator registers). A key
// counts for a row iff the row selected its block AND it is at or before the row's position (the tail of a row's
// list is the open block up to the row itself). Writes split 0's partials and empties in splits 1..nsplits-1, so the
// combine is unchanged. Membership is built per segment of XQ_SELU_SEG blocks (any context length).
//
// seltc6: a second entry, xq_attn_sel_tcu8, runs this body templated on R, the tokens per block, with R * gqa == 48 MMA
// rows (host-checked): R = 4 is xq_attn_sel_tcu's 12 query heads per kv head (TP=1/TP=2; that entry keeps its own
// verbatim code), R = 8 serves 6 (Flash-Next at TP=4). The 48 rows fill the same 3 row tiles x 2 dim halves, so the MMA / softmax / staging work per block
// is the same shape; R = 8 stages the union of 8 rows' selections (the membership byte holds 8 row bits: R <= 8).
#define XQ_SELU_R 4
#define XQ_SELU_SEG 2048
#define XQ_SELU_TB 8                                  // union blocks per tile (32 keys at ratio 4)
#define XQ_SELU_HALF 4                                // staging rows per thread per load batch (3 batches cover 11)
extern "C" __global__ void __launch_bounds__(192)
xq_attn_sel_tcu(float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc,
                const float* __restrict__ qstage, const float* __restrict__ kcvc, const int* __restrict__ sel,
                const int* __restrict__ pos_sel, const int* __restrict__ slotpos,
                int nh_packed, long long geom, int rows_ratio) {
    XQ_PDL_ENTRY();
    constexpr int HD = 256, R = XQ_SELU_R, TK = 32;
    __shared__ __align__(16) __half Ks[TK * HD];
    __shared__ __align__(16) __half Vs[TK * HD];
    __shared__ unsigned memw[XQ_SELU_SEG / 4];        // membership bytes, 4 blocks per word (bit r = row r)
    __shared__ int ublk[XQ_SELU_SEG];
    __shared__ unsigned char umask[XQ_SELU_SEG];
    __shared__ int s_cnt[192];
    __shared__ int s_nu;
    __shared__ unsigned char s_kok[XQ_SELU_R][TK];     // the tile's (row, key) validity
    const int nh = nh_packed & 0xFFFF;
    const int nkv = (nh_packed >> 16) & 0xFFFF;
    const int nsplits = (int)(geom & 0xFFFFF);
    const int sel_max = (int)((geom >> 20) & 0xFFF);
    const int max_pos = (int)(geom >> 32);
    const int nrows_all = rows_ratio & 0xFFFFFF;
    constexpr int ratio = 4;                        // indexer_compress_ratio (host-checked)
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int kvh = blockIdx.y;
    const int g0 = blockIdx.x * R;
    const int nrow = min(R, nrows_all - g0);
    const int gqa = nh / nkv;
    const int slot = slotpos[g0 * 2 + 0];
    const XqKv kc = xq_kv_make(kcvc, max_pos, HD, nkv, slot, kvh);
    __shared__ int s_rpos[R], s_rns[R];
    if (tid < R) {
        s_rpos[tid] = tid < nrow ? slotpos[(g0 + tid) * 2 + 1] : -1;
        s_rns[tid] = tid < nrow ? pos_sel[g0 + tid] + 1 : 0;
    }
    __syncthreads();
    int maxpos = 0;
    #pragma unroll
    for (int r = 0; r < R; r++) maxpos = max(maxpos, s_rpos[r]);
    // this thread's two MMA rows (token-major: rr -> token rr / gqa, head rr % gqa)
    const int rt = warp >> 1, dh = warp & 1;            // row tile, output-dim half
    const int rr0 = 16 * rt + (lane >> 2), rr1 = rr0 + 8;
    const int ti0 = rr0 / gqa, ti1 = rr1 / gqa;
    const int hd0 = rr0 % gqa, hd1 = rr1 % gqa;
    const bool ok0 = ti0 < nrow, ok1 = ti1 < nrow;
    unsigned qf[HD / 16][4];
    {
        const float* q0 = qstage + ((long long)(g0 + (ok0 ? ti0 : 0)) * nh + kvh * gqa + hd0) * HD;
        const float* q1 = qstage + ((long long)(g0 + (ok1 ? ti1 : 0)) * nh + kvh * gqa + hd1) * HD;
        #pragma unroll
        for (int ks = 0; ks < HD / 16; ks++) {
            const int d = 16 * ks + 2 * (lane & 3);
            const float2 a = *reinterpret_cast<const float2*>(q0 + d), b = *reinterpret_cast<const float2*>(q0 + d + 8);
            const float2 c = *reinterpret_cast<const float2*>(q1 + d), e = *reinterpret_cast<const float2*>(q1 + d + 8);
            qf[ks][0] = ok0 ? xq_pftc_h2(a.x, a.y) : 0u;
            qf[ks][1] = ok1 ? xq_pftc_h2(c.x, c.y) : 0u;
            qf[ks][2] = ok0 ? xq_pftc_h2(b.x, b.y) : 0u;
            qf[ks][3] = ok1 ? xq_pftc_h2(e.x, e.y) : 0u;
        }
    }
    float o[HD / 16][4];
    #pragma unroll
    for (int j = 0; j < HD / 16; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
    const float rsqrt_hd = rsqrtf((float)HD);
    const int nblk_all = maxpos / ratio + 1;
    for (int sb = 0; sb < nblk_all; sb += XQ_SELU_SEG) {
        // ---- membership of this segment's blocks
        __syncthreads();                               // the previous segment's readers are done
        for (int i = tid; i < XQ_SELU_SEG / 4; i += 192) memw[i] = 0u;
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < R; r++) {
            const int* srow = sel + (long long)(g0 + r) * sel_max;
            for (int i = tid; i < s_rns[r]; i += 192) {
                const int t = srow[i];
                if (t < 0) continue;
                const int b = t / ratio - sb;
                if (b >= 0 && b < XQ_SELU_SEG) atomicOr(&memw[b >> 2], 1u << (8 * (b & 3) + r));
            }
        }
        __syncthreads();
        // ---- compaction: the segment's selected blocks in ascending order (thread tid owns a contiguous range)
        const int per = (XQ_SELU_SEG + 191) / 192;
        const int lo = tid * per, hi = min(lo + per, XQ_SELU_SEG);
        int cnt = 0;
        for (int b = lo; b < hi; b++) cnt += ((memw[b >> 2] >> (8 * (b & 3))) & 0xFFu) != 0u;
        s_cnt[tid] = cnt;
        __syncthreads();
        if (tid == 0) {
            int run = 0;
            for (int i = 0; i < 192; i++) { const int c = s_cnt[i]; s_cnt[i] = run; run += c; }
            s_nu = run;
        }
        __syncthreads();
        {
            int w = s_cnt[tid];
            for (int b = lo; b < hi; b++) {
                const unsigned mb = (memw[b >> 2] >> (8 * (b & 3))) & 0xFFu;
                if (mb) { ublk[w] = sb + b; umask[w] = (unsigned char)mb; w++; }
            }
        }
        __syncthreads();
        const int nu = s_nu;
        // ---- tiles of XQ_SELU_TB union blocks
        for (int u0 = 0; u0 < nu; u0 += XQ_SELU_TB) {
            __syncthreads();                           // the previous tile's readers are done
            // stage 2 * TK rows (K, V) x 32 chunks; warp w stages rows w, w + 3, ... ; loads batched per half
            #pragma unroll
            for (int half = 0; half < 3; half++) {
                XqQ8Raw raw[XQ_SELU_HALF];
                unsigned okm = 0u;
                #pragma unroll
                for (int i = 0; i < XQ_SELU_HALF; i++) {
                    const int rs = warp + 6 * (i + half * XQ_SELU_HALF);
                    if (rs >= 2 * TK) continue;
                    const int key = rs >> 1, kv = rs & 1;
                    const int u = u0 + key / ratio;
                    const int t = u < nu ? ublk[u] * ratio + key % ratio : -1;
                    if (t >= 0 && t <= maxpos) {
                        okm |= 1u << i;
                        xq_q8_raw(raw[i], (kv ? kc.v : kc.k) + (long long)t * kc.rb, HD, lane * 8);
                    }
                }
                #pragma unroll
                for (int i = 0; i < XQ_SELU_HALF; i++) {
                    const int rs = warp + 6 * (i + half * XQ_SELU_HALF);
                    if (rs >= 2 * TK) continue;
                    const int key = rs >> 1, kv = rs & 1;
                    float f[8];
                    if (okm & (1u << i)) {
                        xq_q8_deq8(f, raw[i]);
                    } else {
                        #pragma unroll
                        for (int e = 0; e < 8; e++) f[e] = 0.f;
                    }
                    const int cs = (lane & 24) | ((lane & 7) ^ (key & 7));
                    *reinterpret_cast<uint4*>((kv ? Vs : Ks) + key * HD + (cs << 3)) =
                        make_uint4(xq_pftc_h2(f[0], f[1]), xq_pftc_h2(f[2], f[3]), xq_pftc_h2(f[4], f[5]), xq_pftc_h2(f[6], f[7]));
                }
            }
            for (int x = tid; x < R * TK; x += 192) {
                const int r = x / TK, key = x % TK;
                const int u = u0 + key / ratio;
                bool okk = false;
                if (u < nu) {
                    const int t = ublk[u] * ratio + key % ratio;
                    okk = ((umask[u] >> r) & 1u) && t <= s_rpos[r];
                }
                s_kok[r][key] = okk;
            }
            __syncthreads();
            // ===== S = Q.K^T, 16 rows x 32 keys per warp (xq_attn_pf_tc256's body) =====
            float s[4][4];
            #pragma unroll
            for (int j = 0; j < 4; j++) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
            #pragma unroll
            for (int ks = 0; ks < HD / 16; ks++) {
                #pragma unroll
                for (int j2 = 0; j2 < 2; j2++) {
                    int lkey, ldim;
                    if (lane < 8)       { lkey = 16 * j2 + lane;            ldim = 16 * ks; }
                    else if (lane < 16) { lkey = 16 * j2 + lane - 8;        ldim = 16 * ks + 8; }
                    else if (lane < 24) { lkey = 16 * j2 + 8 + lane - 16;   ldim = 16 * ks; }
                    else                { lkey = 16 * j2 + 8 + lane - 24;   ldim = 16 * ks + 8; }
                    const int chunk = ldim >> 3;
                    const int cs = (chunk & 24) | ((chunk & 7) ^ (lkey & 7));
                    unsigned r0, r1, r2, r3;
                    xq_pftc_ldm_x4(r0, r1, r2, r3, (unsigned)__cvta_generic_to_shared(Ks + lkey * HD + (cs << 3)));
                    const unsigned ba[2] = {r0, r1}, bb[2] = {r2, r3};
                    xq_pftc_mma(s[2 * j2], qf[ks], ba);
                    xq_pftc_mma(s[2 * j2 + 1], qf[ks], bb);
                }
            }
            // ===== mask (row selected the key's block, key at or before the row) + online softmax =====
            float mx0 = -1e30f, mx1 = -1e30f;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int key = 8 * j + 2 * (lane & 3) + (e & 1);
                    const bool ok = (e < 2 ? ok0 : ok1) && s_kok[e < 2 ? ti0 : ti1][key];
                    const float v = ok ? s[j][e] * rsqrt_hd : -1e30f;
                    s[j][e] = v;
                    if (e < 2) mx0 = fmaxf(mx0, v); else mx1 = fmaxf(mx1, v);
                }
            }
            mx0 = fmaxf(mx0, __shfl_xor_sync(~0u, mx0, 1));
            mx0 = fmaxf(mx0, __shfl_xor_sync(~0u, mx0, 2));
            mx1 = fmaxf(mx1, __shfl_xor_sync(~0u, mx1, 1));
            mx1 = fmaxf(mx1, __shfl_xor_sync(~0u, mx1, 2));
            const bool any0 = mx0 > -1e29f, any1 = mx1 > -1e29f;
            const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
            const float al0 = (any0 && m0 > -1e29f) ? __expf(m0 - mn0) : (any0 ? 0.f : 1.f);
            const float al1 = (any1 && m1 > -1e29f) ? __expf(m1 - mn1) : (any1 ? 0.f : 1.f);
            float su0 = 0.f, su1 = 0.f;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    const float p = (s[j][e] > -1e29f) ? __expf(s[j][e] - (e < 2 ? mn0 : mn1)) : 0.f;
                    s[j][e] = p;
                    if (e < 2) su0 += p; else su1 += p;
                }
            }
            su0 += __shfl_xor_sync(~0u, su0, 1); su0 += __shfl_xor_sync(~0u, su0, 2);
            su1 += __shfl_xor_sync(~0u, su1, 1); su1 += __shfl_xor_sync(~0u, su1, 2);
            l0 = l0 * al0 + su0;  m0 = any0 ? mn0 : m0;
            l1 = l1 * al1 + su1;  m1 = any1 ? mn1 : m1;
            if (al0 != 1.0f) {
                #pragma unroll
                for (int j = 0; j < HD / 16; j++) { o[j][0] *= al0; o[j][1] *= al0; }
            }
            if (al1 != 1.0f) {
                #pragma unroll
                for (int j = 0; j < HD / 16; j++) { o[j][2] *= al1; o[j][3] *= al1; }
            }
            // ===== P (f16, in registers) . V (ldmatrix .trans) =====
            unsigned pa[2][4];
            #pragma unroll
            for (int ks = 0; ks < 2; ks++) {
                pa[ks][0] = xq_pftc_h2(s[2 * ks][0], s[2 * ks][1]);
                pa[ks][1] = xq_pftc_h2(s[2 * ks][2], s[2 * ks][3]);
                pa[ks][2] = xq_pftc_h2(s[2 * ks + 1][0], s[2 * ks + 1][1]);
                pa[ks][3] = xq_pftc_h2(s[2 * ks + 1][2], s[2 * ks + 1][3]);
            }
            #pragma unroll
            for (int ks = 0; ks < 2; ks++) {
                #pragma unroll
                for (int jn = 0; jn < HD / 32; jn++) {
                    int lkey, ldim;
                    const int dq = 128 * dh + 16 * jn;
                    if (lane < 8)       { lkey = 16 * ks + lane;            ldim = dq; }
                    else if (lane < 16) { lkey = 16 * ks + 8 + lane - 8;    ldim = dq; }
                    else if (lane < 24) { lkey = 16 * ks + lane - 16;       ldim = dq + 8; }
                    else                { lkey = 16 * ks + 8 + lane - 24;   ldim = dq + 8; }
                    const int chunk = ldim >> 3;
                    const int cs = (chunk & 24) | ((chunk & 7) ^ (lkey & 7));
                    unsigned r0, r1, r2, r3;
                    xq_pftc_ldm_x4_trans(r0, r1, r2, r3, (unsigned)__cvta_generic_to_shared(Vs + lkey * HD + (cs << 3)));
                    const unsigned ba[2] = {r0, r1}, bb[2] = {r2, r3};
                    xq_pftc_mma(o[2 * jn], pa[ks], ba);
                    xq_pftc_mma(o[2 * jn + 1], pa[ks], bb);
                }
            }
        }
    }
    // ---- partials: split 0 = this block's result (pq8 semantics: max scaled score, sum exp, unnormalized acc);
    // splits 1.. empty
    #pragma unroll
    for (int hh = 0; hh < 2; hh++) {
        const bool ok = hh ? ok1 : ok0;
        if (!ok) continue;
        const int ti = hh ? ti1 : ti0, head = hh ? hd1 : hd0;
        const float mm = hh ? m1 : m0, ll = hh ? l1 : l0;
        const long long hg = (long long)(g0 + ti) * nh + kvh * gqa + head;
        if ((lane & 3) == 0 && dh == 0) {
            pm[hg * nsplits] = mm > -1e29f ? mm : -INFINITY;
            pl[hg * nsplits] = ll;
            for (int sp = 1; sp < nsplits; sp++) { pm[hg * nsplits + sp] = -INFINITY; pl[hg * nsplits + sp] = 0.f; }
        }
        #pragma unroll
        for (int j = 0; j < HD / 16; j++) {
            const int d = 128 * dh + 8 * j + 2 * (lane & 3);
            *reinterpret_cast<float2*>(pacc + hg * nsplits * HD + d) = make_float2(o[j][2 * hh], o[j][2 * hh + 1]);
            for (int sp = 1; sp < nsplits; sp++)
                *reinterpret_cast<float2*>(pacc + (hg * nsplits + sp) * HD + d) = make_float2(0.f, 0.f);
        }
    }
}

// seltc6: the same body as a template on R (tokens per block) for the 6-head entry below. xq_attn_sel_tcu above is
// kept verbatim: compiled through the template its SASS changed (5,880 vs 5,768 instructions, same FP op order) and TP=1
// prompt reading measured ~1.5-2% slower, so the 12-head path stays the original code.
template <int R>
__device__ __forceinline__ void
xq_attn_sel_tcu_body(float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc,
                     const float* __restrict__ qstage, const float* __restrict__ kcvc, const int* __restrict__ sel,
                     const int* __restrict__ pos_sel, const int* __restrict__ slotpos,
                     int nh_packed, long long geom, int rows_ratio) {
    static_assert(R >= 1 && R <= 8, "the membership byte holds one bit per row");
    XQ_PDL_ENTRY();
    constexpr int HD = 256, TK = 32;
    __shared__ __align__(16) __half Ks[TK * HD];
    __shared__ __align__(16) __half Vs[TK * HD];
    __shared__ unsigned memw[XQ_SELU_SEG / 4];        // membership bytes, 4 blocks per word (bit r = row r)
    __shared__ int ublk[XQ_SELU_SEG];
    __shared__ unsigned char umask[XQ_SELU_SEG];
    __shared__ int s_cnt[192];
    __shared__ int s_nu;
    __shared__ unsigned char s_kok[R][TK];             // the tile's (row, key) validity
    const int nh = nh_packed & 0xFFFF;
    const int nkv = (nh_packed >> 16) & 0xFFFF;
    const int nsplits = (int)(geom & 0xFFFFF);
    const int sel_max = (int)((geom >> 20) & 0xFFF);
    const int max_pos = (int)(geom >> 32);
    const int nrows_all = rows_ratio & 0xFFFFFF;
    constexpr int ratio = 4;                        // indexer_compress_ratio (host-checked)
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int kvh = blockIdx.y;
    const int g0 = blockIdx.x * R;
    const int nrow = min(R, nrows_all - g0);
    const int gqa = nh / nkv;
    const int slot = slotpos[g0 * 2 + 0];
    const XqKv kc = xq_kv_make(kcvc, max_pos, HD, nkv, slot, kvh);
    __shared__ int s_rpos[R], s_rns[R];
    if (tid < R) {
        s_rpos[tid] = tid < nrow ? slotpos[(g0 + tid) * 2 + 1] : -1;
        s_rns[tid] = tid < nrow ? pos_sel[g0 + tid] + 1 : 0;
    }
    __syncthreads();
    int maxpos = 0;
    #pragma unroll
    for (int r = 0; r < R; r++) maxpos = max(maxpos, s_rpos[r]);
    // this thread's two MMA rows (token-major: rr -> token rr / gqa, head rr % gqa)
    const int rt = warp >> 1, dh = warp & 1;            // row tile, output-dim half
    const int rr0 = 16 * rt + (lane >> 2), rr1 = rr0 + 8;
    const int ti0 = rr0 / gqa, ti1 = rr1 / gqa;
    const int hd0 = rr0 % gqa, hd1 = rr1 % gqa;
    const bool ok0 = ti0 < nrow, ok1 = ti1 < nrow;
    unsigned qf[HD / 16][4];
    {
        const float* q0 = qstage + ((long long)(g0 + (ok0 ? ti0 : 0)) * nh + kvh * gqa + hd0) * HD;
        const float* q1 = qstage + ((long long)(g0 + (ok1 ? ti1 : 0)) * nh + kvh * gqa + hd1) * HD;
        #pragma unroll
        for (int ks = 0; ks < HD / 16; ks++) {
            const int d = 16 * ks + 2 * (lane & 3);
            const float2 a = *reinterpret_cast<const float2*>(q0 + d), b = *reinterpret_cast<const float2*>(q0 + d + 8);
            const float2 c = *reinterpret_cast<const float2*>(q1 + d), e = *reinterpret_cast<const float2*>(q1 + d + 8);
            qf[ks][0] = ok0 ? xq_pftc_h2(a.x, a.y) : 0u;
            qf[ks][1] = ok1 ? xq_pftc_h2(c.x, c.y) : 0u;
            qf[ks][2] = ok0 ? xq_pftc_h2(b.x, b.y) : 0u;
            qf[ks][3] = ok1 ? xq_pftc_h2(e.x, e.y) : 0u;
        }
    }
    float o[HD / 16][4];
    #pragma unroll
    for (int j = 0; j < HD / 16; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
    const float rsqrt_hd = rsqrtf((float)HD);
    const int nblk_all = maxpos / ratio + 1;
    for (int sb = 0; sb < nblk_all; sb += XQ_SELU_SEG) {
        // ---- membership of this segment's blocks
        __syncthreads();                               // the previous segment's readers are done
        for (int i = tid; i < XQ_SELU_SEG / 4; i += 192) memw[i] = 0u;
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < R; r++) {
            const int* srow = sel + (long long)(g0 + r) * sel_max;
            for (int i = tid; i < s_rns[r]; i += 192) {
                const int t = srow[i];
                if (t < 0) continue;
                const int b = t / ratio - sb;
                if (b >= 0 && b < XQ_SELU_SEG) atomicOr(&memw[b >> 2], 1u << (8 * (b & 3) + r));
            }
        }
        __syncthreads();
        // ---- compaction: the segment's selected blocks in ascending order (thread tid owns a contiguous range)
        const int per = (XQ_SELU_SEG + 191) / 192;
        const int lo = tid * per, hi = min(lo + per, XQ_SELU_SEG);
        int cnt = 0;
        for (int b = lo; b < hi; b++) cnt += ((memw[b >> 2] >> (8 * (b & 3))) & 0xFFu) != 0u;
        s_cnt[tid] = cnt;
        __syncthreads();
        if (tid == 0) {
            int run = 0;
            for (int i = 0; i < 192; i++) { const int c = s_cnt[i]; s_cnt[i] = run; run += c; }
            s_nu = run;
        }
        __syncthreads();
        {
            int w = s_cnt[tid];
            for (int b = lo; b < hi; b++) {
                const unsigned mb = (memw[b >> 2] >> (8 * (b & 3))) & 0xFFu;
                if (mb) { ublk[w] = sb + b; umask[w] = (unsigned char)mb; w++; }
            }
        }
        __syncthreads();
        const int nu = s_nu;
        // ---- tiles of XQ_SELU_TB union blocks
        for (int u0 = 0; u0 < nu; u0 += XQ_SELU_TB) {
            __syncthreads();                           // the previous tile's readers are done
            // stage 2 * TK rows (K, V) x 32 chunks; warp w stages rows w, w + 3, ... ; loads batched per half
            #pragma unroll
            for (int half = 0; half < 3; half++) {
                XqQ8Raw raw[XQ_SELU_HALF];
                unsigned okm = 0u;
                #pragma unroll
                for (int i = 0; i < XQ_SELU_HALF; i++) {
                    const int rs = warp + 6 * (i + half * XQ_SELU_HALF);
                    if (rs >= 2 * TK) continue;
                    const int key = rs >> 1, kv = rs & 1;
                    const int u = u0 + key / ratio;
                    const int t = u < nu ? ublk[u] * ratio + key % ratio : -1;
                    if (t >= 0 && t <= maxpos) {
                        okm |= 1u << i;
                        xq_q8_raw(raw[i], (kv ? kc.v : kc.k) + (long long)t * kc.rb, HD, lane * 8);
                    }
                }
                #pragma unroll
                for (int i = 0; i < XQ_SELU_HALF; i++) {
                    const int rs = warp + 6 * (i + half * XQ_SELU_HALF);
                    if (rs >= 2 * TK) continue;
                    const int key = rs >> 1, kv = rs & 1;
                    float f[8];
                    if (okm & (1u << i)) {
                        xq_q8_deq8(f, raw[i]);
                    } else {
                        #pragma unroll
                        for (int e = 0; e < 8; e++) f[e] = 0.f;
                    }
                    const int cs = (lane & 24) | ((lane & 7) ^ (key & 7));
                    *reinterpret_cast<uint4*>((kv ? Vs : Ks) + key * HD + (cs << 3)) =
                        make_uint4(xq_pftc_h2(f[0], f[1]), xq_pftc_h2(f[2], f[3]), xq_pftc_h2(f[4], f[5]), xq_pftc_h2(f[6], f[7]));
                }
            }
            for (int x = tid; x < R * TK; x += 192) {
                const int r = x / TK, key = x % TK;
                const int u = u0 + key / ratio;
                bool okk = false;
                if (u < nu) {
                    const int t = ublk[u] * ratio + key % ratio;
                    okk = ((umask[u] >> r) & 1u) && t <= s_rpos[r];
                }
                s_kok[r][key] = okk;
            }
            __syncthreads();
            // ===== S = Q.K^T, 16 rows x 32 keys per warp (xq_attn_pf_tc256's body) =====
            float s[4][4];
            #pragma unroll
            for (int j = 0; j < 4; j++) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
            #pragma unroll
            for (int ks = 0; ks < HD / 16; ks++) {
                #pragma unroll
                for (int j2 = 0; j2 < 2; j2++) {
                    int lkey, ldim;
                    if (lane < 8)       { lkey = 16 * j2 + lane;            ldim = 16 * ks; }
                    else if (lane < 16) { lkey = 16 * j2 + lane - 8;        ldim = 16 * ks + 8; }
                    else if (lane < 24) { lkey = 16 * j2 + 8 + lane - 16;   ldim = 16 * ks; }
                    else                { lkey = 16 * j2 + 8 + lane - 24;   ldim = 16 * ks + 8; }
                    const int chunk = ldim >> 3;
                    const int cs = (chunk & 24) | ((chunk & 7) ^ (lkey & 7));
                    unsigned r0, r1, r2, r3;
                    xq_pftc_ldm_x4(r0, r1, r2, r3, (unsigned)__cvta_generic_to_shared(Ks + lkey * HD + (cs << 3)));
                    const unsigned ba[2] = {r0, r1}, bb[2] = {r2, r3};
                    xq_pftc_mma(s[2 * j2], qf[ks], ba);
                    xq_pftc_mma(s[2 * j2 + 1], qf[ks], bb);
                }
            }
            // ===== mask (row selected the key's block, key at or before the row) + online softmax =====
            float mx0 = -1e30f, mx1 = -1e30f;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int key = 8 * j + 2 * (lane & 3) + (e & 1);
                    const bool ok = (e < 2 ? ok0 : ok1) && s_kok[e < 2 ? ti0 : ti1][key];
                    const float v = ok ? s[j][e] * rsqrt_hd : -1e30f;
                    s[j][e] = v;
                    if (e < 2) mx0 = fmaxf(mx0, v); else mx1 = fmaxf(mx1, v);
                }
            }
            mx0 = fmaxf(mx0, __shfl_xor_sync(~0u, mx0, 1));
            mx0 = fmaxf(mx0, __shfl_xor_sync(~0u, mx0, 2));
            mx1 = fmaxf(mx1, __shfl_xor_sync(~0u, mx1, 1));
            mx1 = fmaxf(mx1, __shfl_xor_sync(~0u, mx1, 2));
            const bool any0 = mx0 > -1e29f, any1 = mx1 > -1e29f;
            const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
            const float al0 = (any0 && m0 > -1e29f) ? __expf(m0 - mn0) : (any0 ? 0.f : 1.f);
            const float al1 = (any1 && m1 > -1e29f) ? __expf(m1 - mn1) : (any1 ? 0.f : 1.f);
            float su0 = 0.f, su1 = 0.f;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    const float p = (s[j][e] > -1e29f) ? __expf(s[j][e] - (e < 2 ? mn0 : mn1)) : 0.f;
                    s[j][e] = p;
                    if (e < 2) su0 += p; else su1 += p;
                }
            }
            su0 += __shfl_xor_sync(~0u, su0, 1); su0 += __shfl_xor_sync(~0u, su0, 2);
            su1 += __shfl_xor_sync(~0u, su1, 1); su1 += __shfl_xor_sync(~0u, su1, 2);
            l0 = l0 * al0 + su0;  m0 = any0 ? mn0 : m0;
            l1 = l1 * al1 + su1;  m1 = any1 ? mn1 : m1;
            if (al0 != 1.0f) {
                #pragma unroll
                for (int j = 0; j < HD / 16; j++) { o[j][0] *= al0; o[j][1] *= al0; }
            }
            if (al1 != 1.0f) {
                #pragma unroll
                for (int j = 0; j < HD / 16; j++) { o[j][2] *= al1; o[j][3] *= al1; }
            }
            // ===== P (f16, in registers) . V (ldmatrix .trans) =====
            unsigned pa[2][4];
            #pragma unroll
            for (int ks = 0; ks < 2; ks++) {
                pa[ks][0] = xq_pftc_h2(s[2 * ks][0], s[2 * ks][1]);
                pa[ks][1] = xq_pftc_h2(s[2 * ks][2], s[2 * ks][3]);
                pa[ks][2] = xq_pftc_h2(s[2 * ks + 1][0], s[2 * ks + 1][1]);
                pa[ks][3] = xq_pftc_h2(s[2 * ks + 1][2], s[2 * ks + 1][3]);
            }
            #pragma unroll
            for (int ks = 0; ks < 2; ks++) {
                #pragma unroll
                for (int jn = 0; jn < HD / 32; jn++) {
                    int lkey, ldim;
                    const int dq = 128 * dh + 16 * jn;
                    if (lane < 8)       { lkey = 16 * ks + lane;            ldim = dq; }
                    else if (lane < 16) { lkey = 16 * ks + 8 + lane - 8;    ldim = dq; }
                    else if (lane < 24) { lkey = 16 * ks + lane - 16;       ldim = dq + 8; }
                    else                { lkey = 16 * ks + 8 + lane - 24;   ldim = dq + 8; }
                    const int chunk = ldim >> 3;
                    const int cs = (chunk & 24) | ((chunk & 7) ^ (lkey & 7));
                    unsigned r0, r1, r2, r3;
                    xq_pftc_ldm_x4_trans(r0, r1, r2, r3, (unsigned)__cvta_generic_to_shared(Vs + lkey * HD + (cs << 3)));
                    const unsigned ba[2] = {r0, r1}, bb[2] = {r2, r3};
                    xq_pftc_mma(o[2 * jn], pa[ks], ba);
                    xq_pftc_mma(o[2 * jn + 1], pa[ks], bb);
                }
            }
        }
    }
    // ---- partials: split 0 = this block's result (pq8 semantics: max scaled score, sum exp, unnormalized acc);
    // splits 1.. empty
    #pragma unroll
    for (int hh = 0; hh < 2; hh++) {
        const bool ok = hh ? ok1 : ok0;
        if (!ok) continue;
        const int ti = hh ? ti1 : ti0, head = hh ? hd1 : hd0;
        const float mm = hh ? m1 : m0, ll = hh ? l1 : l0;
        const long long hg = (long long)(g0 + ti) * nh + kvh * gqa + head;
        if ((lane & 3) == 0 && dh == 0) {
            pm[hg * nsplits] = mm > -1e29f ? mm : -INFINITY;
            pl[hg * nsplits] = ll;
            for (int sp = 1; sp < nsplits; sp++) { pm[hg * nsplits + sp] = -INFINITY; pl[hg * nsplits + sp] = 0.f; }
        }
        #pragma unroll
        for (int j = 0; j < HD / 16; j++) {
            const int d = 128 * dh + 8 * j + 2 * (lane & 3);
            *reinterpret_cast<float2*>(pacc + hg * nsplits * HD + d) = make_float2(o[j][2 * hh], o[j][2 * hh + 1]);
            for (int sp = 1; sp < nsplits; sp++)
                *reinterpret_cast<float2*>(pacc + (hg * nsplits + sp) * HD + d) = make_float2(0.f, 0.f);
        }
    }
}
// seltc6: 8 tokens x 6 query heads per block (Flash-Next at TP=4: 6 query heads per replicated kv head)
extern "C" __global__ void __launch_bounds__(192)
xq_attn_sel_tcu8(float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc,
                 const float* __restrict__ qstage, const float* __restrict__ kcvc, const int* __restrict__ sel,
                 const int* __restrict__ pos_sel, const int* __restrict__ slotpos,
                 int nh_packed, long long geom, int rows_ratio) {
    xq_attn_sel_tcu_body<8>(pm, pl, pacc, qstage, kcvc, sel, pos_sel, slotpos, nh_packed, geom, rows_ratio);
}

// ---- A5 D5: dense split-K decode attention for the MTP draft head (was xq_attn_decode:
// one block per QUERY head, each serially streaming its kv head's whole f32 K/V — at 32K
// context 12x redundant reads (GQA 12), 6.6 ms of a 7.65 ms draft pass). One block per
// (row, kv head, split); warps own query heads (w, w+8) and share every K/V row read, so a
// pass reads K/V ONCE. The split range is derived ON DEVICE from the row's position
// (split_len = ceil((p+1)/nsplits)), so a graph captured at any context replays at every
// context. Partials feed xq_attn_sel_combine (non-empty splits, ascending). Draft-only:
// the verify decides every emitted token (acceptance, not losslessness, is the gate).
extern "C" __global__ void xq_attn_dense_splitk4(float* __restrict__ pm, float* __restrict__ pl,
                                                float* __restrict__ pacc, const float* __restrict__ qstage,
                                                const float* __restrict__ kcvc, const int* __restrict__ slotpos,
                                                int nh_packed, long long geom) {
    XQ_PDL_ENTRY();
    const int nh = nh_packed & 0xFFFF;
    const int nkv = (nh_packed >> 16) & 0xFFFF;
    const int nsplits = (int)(geom & 0xFFFFF);
    const int max_pos = (int)(geom >> 32);
    const int sp = blockIdx.x % nsplits;
    const int kvh = (blockIdx.x / nsplits) % nkv;
    const int row = blockIdx.x / (nsplits * nkv);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gqa = nh / nkv;
    const int hd = blockDim.x;                   // 256: 32 lanes x 8 dims per head row
    const int slot = slotpos[row * 2 + 0];
    const int npos = slotpos[row * 2 + 1] + 1;   // keys 0..=p (p's own K/V written by xq_attn_prep)
    const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
    const int split_len = (npos + nsplits - 1) / nsplits;
    const int begin = sp * split_len;
    const int end = min(begin + split_len, npos);
    for (int hh = 0; hh < 2; hh++) {
        const int w = warp + hh * 8;
        if (w >= gqa) break;
        const int h = kvh * gqa + w;
        const float* qrow = qstage + ((long long)row * nh + h) * hd;
        float qv[8];
        *reinterpret_cast<float4*>(qv) = reinterpret_cast<const float4*>(qrow + lane * 8)[0];
        *reinterpret_cast<float4*>(qv + 4) = reinterpret_cast<const float4*>(qrow + lane * 8)[1];
        const float qs = rsqrtf((float)hd);
        float m = -INFINITY, l = 0.0f;
        float acc[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) acc[i] = 0.0f;
        for (int t = begin; t < end; t++) {
            float kv[8], vv[8];
            xq_kv_ld8(kv, kc.k + (long long)t * kc.rb, kc.fmt, hd, lane * 8);
            xq_kv_ld8(vv, kc.v + (long long)t * kc.rb, kc.fmt, hd, lane * 8);
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; i++) dot += qv[i] * kv[i];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
            const float sc = dot * qs;
            const float m_new = fmaxf(m, sc);
            const float scale_f = __expf(m - m_new);
            const float wt = __expf(sc - m_new);
            l = l * scale_f + wt;
            #pragma unroll
            for (int i = 0; i < 8; i++) acc[i] = acc[i] * scale_f + wt * vv[i];
            m = m_new;
        }
        const long long hg = (long long)row * nh + h;
        if (lane == 0) {
            pm[hg * nsplits + sp] = m;
            pl[hg * nsplits + sp] = l;
        }
        #pragma unroll
        for (int i = 0; i < 8; i++) pacc[(hg * nsplits + sp) * hd + lane * 8 + i] = acc[i];
    }
}

// ---- DSPLITK round 2 (owner 2026-11-07): the SHARED-READ twin of xq_attn_dense_splitk4 for the rows of ONE verify
// (one slot, the host guarantees it). xq_attn_dense_splitk4 gives every row its own blocks, so a verify of m rows
// fetches the context m times (35B at 48K: ~+10 ms per draft depth). Here a block takes XQ_VR_ROWS rows of one
// (KV head, slice): each K/V row of the union of the rows' slice ranges is loaded ONCE and folded into every row
// whose OWN range [sp*split_len_r, ...) contains it — per row the same keys, the same ascending order and the same
// expressions as xq_attn_dense_splitk4, so its partials (and the combine) are bitwise those of a decode at that
// position (the --probe-exl3-tpspec LOSSLESS gate is the check). Grid ceil(m / XQ_VR_ROWS) x nkv x nsplits, block 256.
#define XQ_VR_ROWS 4
extern "C" __global__ void xq_attn_dense_splitk4_vr(float* __restrict__ pm, float* __restrict__ pl,
                                                   float* __restrict__ pacc, const float* __restrict__ qstage,
                                                   const float* __restrict__ kcvc, const int* __restrict__ slotpos,
                                                   int nh_packed, long long geom, int m) {
    XQ_PDL_ENTRY();
    const int nh = nh_packed & 0xFFFF;
    const int nkv = (nh_packed >> 16) & 0xFFFF;
    const int nsplits = (int)(geom & 0xFFFFF);
    const int max_pos = (int)(geom >> 32);
    const int sp = blockIdx.x % nsplits;
    const int kvh = (blockIdx.x / nsplits) % nkv;
    const int r0 = (blockIdx.x / (nsplits * nkv)) * XQ_VR_ROWS;
    const int nrows = min(XQ_VR_ROWS, m - r0);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gqa = nh / nkv;
    const int hd = blockDim.x;                   // 256: 32 lanes x 8 dims per head row
    const int slot = slotpos[r0 * 2 + 0];
    const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
    int rb[XQ_VR_ROWS], re[XQ_VR_ROWS];
    int ub = 0x7fffffff, ue = 0;
    #pragma unroll
    for (int r = 0; r < XQ_VR_ROWS; r++) {
        rb[r] = 0; re[r] = 0;
        if (r < nrows) {
            const int npos = slotpos[(r0 + r) * 2 + 1] + 1;    // keys 0..=p (p's own K/V written by xq_attn_prep)
            const int split_len = (npos + nsplits - 1) / nsplits;
            rb[r] = sp * split_len;
            re[r] = min(rb[r] + split_len, npos);
            if (rb[r] < re[r]) { ub = min(ub, rb[r]); ue = max(ue, re[r]); }
        }
    }
    for (int hh = 0; hh < 2; hh++) {
        const int w = warp + hh * 8;
        if (w >= gqa) break;
        const int h = kvh * gqa + w;
        float qv[XQ_VR_ROWS][8];
        #pragma unroll
        for (int r = 0; r < XQ_VR_ROWS; r++) {
            const float* qrow = qstage + ((long long)(r0 + min(r, nrows - 1)) * nh + h) * hd;
            *reinterpret_cast<float4*>(qv[r]) = reinterpret_cast<const float4*>(qrow + lane * 8)[0];
            *reinterpret_cast<float4*>(qv[r] + 4) = reinterpret_cast<const float4*>(qrow + lane * 8)[1];
        }
        const float qs = rsqrtf((float)hd);
        float mr[XQ_VR_ROWS], lr[XQ_VR_ROWS], acc[XQ_VR_ROWS][8];
        #pragma unroll
        for (int r = 0; r < XQ_VR_ROWS; r++) {
            mr[r] = -INFINITY; lr[r] = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; i++) acc[r][i] = 0.0f;
        }
        for (int t = ub; t < ue; t++) {
            float kv[8], vv[8];
            xq_kv_ld8(kv, kc.k + (long long)t * kc.rb, kc.fmt, hd, lane * 8);
            xq_kv_ld8(vv, kc.v + (long long)t * kc.rb, kc.fmt, hd, lane * 8);
            #pragma unroll
            for (int r = 0; r < XQ_VR_ROWS; r++) {
                if (t < rb[r] || t >= re[r]) continue;       // warp-uniform: the row's own slice only
                float dot = 0.0f;
                #pragma unroll
                for (int i = 0; i < 8; i++) dot += qv[r][i] * kv[i];
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
                const float sc = dot * qs;
                const float m_new = fmaxf(mr[r], sc);
                const float scale_f = __expf(mr[r] - m_new);
                const float wt = __expf(sc - m_new);
                lr[r] = lr[r] * scale_f + wt;
                #pragma unroll
                for (int i = 0; i < 8; i++) acc[r][i] = acc[r][i] * scale_f + wt * vv[i];
                mr[r] = m_new;
            }
        }
        #pragma unroll
        for (int r = 0; r < XQ_VR_ROWS; r++) {
            if (r >= nrows) break;
            const long long hg = (long long)(r0 + r) * nh + h;
            if (lane == 0) {
                pm[hg * nsplits + sp] = mr[r];
                pl[hg * nsplits + sp] = lr[r];
            }
            #pragma unroll
            for (int i = 0; i < 8; i++) pacc[(hg * nsplits + sp) * hd + lane * 8 + i] = acc[r][i];
        }
    }
}

// ---- 6. split merge + gate — xq_attn_decode's final expression order.
// Grid (rows*nh), block hd. m_g = max over split maxes; l_g = sum of
// pl*exp(pm-m_g) and acc[tid] = sum of pacc*exp(pm-m_g) over NON-EMPTY splits
// (pm > -INFINITY), s ASCENDING; then the per-head gate at qg's +hd offset
// and attn = f16((acc/l_g) * sig(g)) — exactly xq_attn_decode's closing lines.
// WP25: bits 28..31 of `rows` carry the KV format (q8: the split-K partials accumulate in the
// cache's rotated basis; the output rotates back here, once, before the gate — the reference implementation's merge).
extern "C" __global__ void xq_attn_sel_combine(__half* __restrict__ attn, const float* __restrict__ pm,
                                               const float* __restrict__ pl, const float* __restrict__ pacc,
                                               const __half* __restrict__ qg, int nh_packed,
                                               int nsplits, int rows) {
    XQ_PDL_ENTRY();
    const int kvf = xq_kv_fmt(rows);             // grid is rows*nh; b derives from blockIdx
    const int nh = nh_packed & 0xFFFF;
    const int hd = blockDim.x;
    const int b = blockIdx.x / nh;
    const int h = blockIdx.x % nh;
    const int tid = threadIdx.x;
    const long long hg = (long long)b * nh + h;
    float m_g = -INFINITY;
    for (int s = 0; s < nsplits; s++) m_g = fmaxf(m_g, pm[hg * nsplits + s]);
    float l_g = 0.0f;
    for (int s = 0; s < nsplits; s++)
        if (pm[hg * nsplits + s] > -INFINITY) l_g += pl[hg * nsplits + s] * __expf(pm[hg * nsplits + s] - m_g);
    float acc = 0.0f;
    for (int s = 0; s < nsplits; s++) {
        if (pm[hg * nsplits + s] > -INFINITY) {
            const float w = __expf(pm[hg * nsplits + s] - m_g);
            acc += pacc[(hg * nsplits + s) * hd + tid] * w;
        }
    }
    const float g = xq_h2f(qg[(long long)b * nh * hd * 2 + (long long)h * hd * 2 + hd + tid]);
    const float o = xq_kv_rot(acc / l_g, kvf, tid & 31);
    attn[(long long)b * nh * hd + (long long)h * hd + tid] = xq_f2h(o * xq_sig(g));
}

