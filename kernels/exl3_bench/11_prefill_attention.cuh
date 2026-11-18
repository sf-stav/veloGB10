// exl3_bench/11_prefill_attention.cuh: chunked prefill attention: KV append, dense/flash prefill attention, PFTC tensor-core attention.
// Layout contract = xq_attn_decode.
// Section 11 of 23 of exl3_bench.cu: included in order by exl3_bench.cu, one translation unit; text is moved verbatim.
// ---- full-attention KV append over a chunk: per kv-head, k-norm + partial rope
// + cache write for each chunk position (pos = pos0 + t). V is cached raw.
// Layout contract = xq_attn_decode: kp/vp rows [pos][kvh][hd], cossin rows
// [pos][cos rdim/2 | pad | sin rdim/2 | pad].
// 12-arg cap: nkv_hd = nkv|hd<<16; rdim_maxpos = rdim|max_pos<<8
// (rdim needs 8 bits: Flash-Next head_dim=256 × partial_rotary 0.25 = 64 —
// the old 6-bit field truncated 64→0 and silently disabled K rope);
// slot_pos0 = slot|pos0<<12 (slot < 4096, pos0 < 2^19).
extern "C" __global__ void xq_attn_kv_prefill(__half* __restrict__ kp, __half* __restrict__ vp,
                                              float* __restrict__ kcvc, const __half* __restrict__ qnkn,
                                              const float* __restrict__ cossin, int nkv_hd,
                                              int rdim_maxpos, int slot_pos0, int C, float eps) {
    XQ_PDL_ENTRY();
    const int nkv = nkv_hd & 0xFFFF;
    const int hd = ((unsigned)nkv_hd) >> 16;
    const int rdim = rdim_maxpos & 0xFF;
    // K1: bits 8..27 = max_pos, bits 28..31 = KV format -> repacked as xq_kv_make's mpf
    const int max_pos = (int)(((((unsigned)rdim_maxpos) >> 8) & 0xFFFFFu) | ((((unsigned)rdim_maxpos) >> 28) << 28));
    const int slot = slot_pos0 & 0xFFF;
    const int pos0 = ((unsigned)slot_pos0) >> 12;
    const int kvh = blockIdx.x;
    const int tid = threadIdx.x;
    const __half* knw = qnkn + hd;
    __shared__ float red[1024];
    __shared__ float kraw[1024];
    __shared__ float kbuf[1024];
    const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
    for (int t = 0; t < C; t++) {
        const long long row = (long long)t * nkv * hd + (long long)kvh * hd;
        // shared fn: bit-identical K words vs xq_attn_decode's write
        xq_k_norm_rope(kbuf, red, kp + row, knw, cossin + (long long)t * 2 * rdim,
                       cossin + (long long)t * 2 * rdim + rdim, tid, hd, rdim, eps);
        const long long p = pos0 + t;
        xq_kv_put(kc.k + p * kc.rb, kc.fmt, hd, tid, kbuf[tid]);
        xq_kv_put(kc.v + p * kc.rb, kc.fmt, hd, tid, xq_h2f(vp[row + tid]));
        __syncthreads();
    }
}

// ---- PQ8 item 1 (SURPASS_PLAN WP06 item 1): the chunk KV append with one CTA per (row, kv head).
// xq_attn_kv_prefill runs grid = nkv (2 CTAs on 48 SMs) and walks the C rows serially — a latency
// chain of ~11 block barriers + 3 dependent global loads per row: 2.72 ms/call at C = 2048, 13 calls
// per chunk (12 trunk layers + the MTP head fill). An 8-bit writer lengthens every row's chain (q8:
// two 5-stage H32 butterflies, two 5-stage absmax trees and an IEEE divide per row; PROBES_14
// measured the fp8 writer at 786 vs 578 ms per 32K prefill). Here grid = C*nkv: t = bid / nkv,
// kvh = bid % nkv, blockDim = hd, and the body is xq_attn_kv_prefill's loop body VERBATIM (the
// shared inline xq_k_norm_rope — blockDim.x = hd, so the same halving tree — then xq_kv_put K and
// V), so every cache byte equals the serial kernel's for every format (GB10_PQ8_XCHECK diffs the
// rows against it; GB10_PQ8_KVW=0 / GB10_PQ8_OFF=1 keep the serial kernel). Rows are independent:
// row t reads kp/vp/cossin row t and writes cache row pos0 + t only. Same 12-arg packing.
extern "C" __global__ void xq_attn_kv_prefill_rows(__half* __restrict__ kp, __half* __restrict__ vp,
                                                   float* __restrict__ kcvc, const __half* __restrict__ qnkn,
                                                   const float* __restrict__ cossin, int nkv_hd,
                                                   int rdim_maxpos, int slot_pos0, int C, float eps) {
    XQ_PDL_ENTRY();
    const int nkv = nkv_hd & 0xFFFF;
    const int hd = ((unsigned)nkv_hd) >> 16;
    const int rdim = rdim_maxpos & 0xFF;
    const int max_pos = (int)(((((unsigned)rdim_maxpos) >> 8) & 0xFFFFFu) | ((((unsigned)rdim_maxpos) >> 28) << 28));
    const int slot = slot_pos0 & 0xFFF;
    const int pos0 = ((unsigned)slot_pos0) >> 12;
    const int t = blockIdx.x / nkv;
    const int kvh = blockIdx.x % nkv;
    if (t >= C) return;                           // block-uniform (grid is exactly C*nkv)
    const int tid = threadIdx.x;
    const __half* knw = qnkn + hd;
    __shared__ float red[1024];
    __shared__ float kbuf[1024];
    const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
    const long long row = (long long)t * nkv * hd + (long long)kvh * hd;
    xq_k_norm_rope(kbuf, red, kp + row, knw, cossin + (long long)t * 2 * rdim,
                   cossin + (long long)t * 2 * rdim + rdim, tid, hd, rdim, eps);
    const long long p = pos0 + t;
    xq_kv_put(kc.k + p * kc.rb, kc.fmt, hd, tid, kbuf[tid]);
    xq_kv_put(kc.v + p * kc.rb, kc.fmt, hd, tid, xq_h2f(vp[row + tid]));
}

// ---- S-C4PRIMEA: the multi-slot twin of xq_attn_kv_prefill_rows. Row t writes
// to (slot, pos) taken PER ROW from the pairs buffer (sc.slots: [slot,pos] i32
// pairs) instead of the single (slot, pos0+t). Per-row math is IDENTICAL
// (xq_k_norm_rope / xq_kv_put on the same inputs), so the bytes written for a
// lane's row are bit-identical to the per-lane xq_attn_kv_prefill launch -
// only the addressing differs. Serves the BATCHED shared-step prime; the
// control site keeps xq_attn_kv_prefill.
extern "C" __global__ void xq_attn_kv_prefill_pairs(__half* __restrict__ kp, __half* __restrict__ vp,
                                                    float* __restrict__ kcvc, const __half* __restrict__ qnkn,
                                                    const float* __restrict__ cossin, const int* __restrict__ pairs,
                                                    int nkv_hd, int rdim_maxpos, int C, float eps) {
    XQ_PDL_ENTRY();
    const int nkv = nkv_hd & 0xFFFF;
    const int hd = ((unsigned)nkv_hd) >> 16;
    const int rdim = rdim_maxpos & 0xFF;
    const int max_pos = (int)(((((unsigned)rdim_maxpos) >> 8) & 0xFFFFFu) | ((((unsigned)rdim_maxpos) >> 28) << 28));
    const int t = blockIdx.x / nkv;
    const int kvh = blockIdx.x % nkv;
    if (t >= C) return;                           // block-uniform (grid is exactly C*nkv)
    const int tid = threadIdx.x;
    const __half* knw = qnkn + hd;
    __shared__ float red[1024];
    __shared__ float kbuf[1024];
    const int slot = pairs[2 * t];                // per-row slot (block-uniform)
    const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
    const long long row = (long long)t * nkv * hd + (long long)kvh * hd;
    xq_k_norm_rope(kbuf, red, kp + row, knw, cossin + (long long)t * 2 * rdim,
                   cossin + (long long)t * 2 * rdim + rdim, tid, hd, rdim, eps);
    const long long p = pairs[2 * t + 1];         // per-row pos
    xq_kv_put(kc.k + p * kc.rb, kc.fmt, hd, tid, kbuf[tid]);
    xq_kv_put(kc.v + p * kc.rb, kc.fmt, hd, tid, xq_h2f(vp[row + tid]));
}

// ---- S-C4PRIMEA: ONE-launch gather of the shared step's per-lane tap rows
// into the re-prime's lane-major resid rows (replaces the per-lane xq_copy_f32
// launches). meta[i] = slot | (tap_row << 12); row i copies he floats from
// taps_keep[slot*win_f + tap_row*he] to resid[i*he]. Bytes identical to the
// per-lane copies (plain float copy).
extern "C" __global__ void xq_gather_tap_rows(float* __restrict__ dst, const float* __restrict__ src,
                                              const int* __restrict__ meta, long long win_f, int he, int L) {
    XQ_PDL_ENTRY();
    const int i = blockIdx.x;
    if (i >= L) return;
    const int slot = meta[i] & 0xFFF;
    const int row = meta[i] >> 12;
    const float* s = src + (long long)slot * win_f + (long long)row * he;
    float* d = dst + (long long)i * he;
    for (int j = threadIdx.x; j < he; j += blockDim.x) d[j] = s[j];
}

// ---- full-attention prefill: per (head), loop the chunk's query positions;
// q-norm + rope per position, two-pass softmax over t' <= pos (cache layout as
// decode, ascending t). Output row layout = decode's attn [pos][head][hd].
// 12-arg cap: nh_nkv = nkv|nh<<16; hd_rdim = hd|rdim<<16; slot_pos0 as above.
extern "C" __global__ void xq_attn_prefill_q(__half* __restrict__ attn,
                                             const __half* __restrict__ qg,
                                             const float* __restrict__ kcvc,
                                             const __half* __restrict__ qnkn,
                                             const float* __restrict__ cossin,
                                             int nh_nkv, int hd_rdim, int max_pos,
                                             int slot_pos0, int C, float eps) {
    XQ_PDL_ENTRY();
    const int nkv = nh_nkv & 0xFFFF;
    const int nh = ((unsigned)nh_nkv) >> 16;
    const int hd = hd_rdim & 0xFFFF;
    const int rdim = ((unsigned)hd_rdim) >> 16;
    const int slot = slot_pos0 & 0xFFF;
    const int pos0 = ((unsigned)slot_pos0) >> 12;
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    const int kvh = head / (nh / nkv);
    __shared__ float red[1024];
    __shared__ float qbuf[1024];
    __shared__ float qraw[1024];
    const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
    const float scale = rsqrtf((float)hd);
    for (int t = 0; t < C; t++) {
        const long long qrow = (long long)t * nh * hd * 2 + (long long)head * hd * 2;
        float qv = xq_h2f(qg[qrow + tid]);
        red[tid] = qv * qv;
        __syncthreads();
        for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) { if (tid < s2) red[tid] += red[tid + s2]; __syncthreads(); }
        qv *= rsqrtf(red[0] / (float)hd + eps) * (1.0f + xq_h2f(qnkn[tid]));
        const float* c = cossin + (long long)t * 2 * rdim;
        const float* s = c + rdim;
        qraw[tid] = qv;
        __syncthreads();
        if (tid < rdim / 2) {
            float x1 = qraw[tid], x2 = qraw[tid + rdim / 2];
            qbuf[tid] = x1 * c[tid] - x2 * s[tid];
            qbuf[tid + rdim / 2] = x2 * c[tid] + x1 * s[tid];
        } else if (tid >= rdim) {
            qbuf[tid] = qraw[tid];
        }
        __syncthreads();
        qv = xq_kv_rot(qbuf[tid], kc.fmt, tid & 31);   // WP25 q8: rotated basis
        const int p = pos0 + t;
        float acc, denom;
        xq_attn_softmax(red, qv, kc, p, tid, hd, acc, denom);
        // S-A3-f-b FIX: the sweep write was missing the per-head sigmoid gate that
        // xq_attn_decode applies — sweep rows fed later layers' GDN state ungated,
        // leaving logits rel ~0.8 vs sequential. qg rows are [values|gate] pairs
        // (stride 2*hd per head); the gate is the second half.
        float g = xq_h2f(qg[qrow + hd + tid]);
        const float o = xq_kv_rot(acc / denom, kc.fmt, tid & 31);
        attn[(long long)t * nh * hd + (long long)head * hd + tid] =
            xq_f2h(o * xq_sig(g));
        __syncthreads();
    }
}

// ---- S-A3-f-g: flash-style chunked prefill attention (the speed directive).
// Replaces xq_attn_prefill_q's one-block-per-head serial sweep (143 ms/layer at
// C=512) with one block per (q-tile of BT rows, head). Block 512 threads = 16
// warps, one warp per q row. K/V tiles are staged through shared memory once
// per block (padded rows kill bank conflicts) and reused by every warp; each
// warp keeps the online-softmax state (m, l, acc[4]) in registers with the
// standard rescale on max growth. Math matches the sweep kernel (fp32 accum,
// f16 out, sigmoid gate after averaging); reduction order differs — the drift
// protocol applies. KV layout: kcvc[slot][k|v][kvh][pos][hd] fp32 as written
// by xq_attn_kv_prefill. qg rows are [values|gate] pairs per head.
// S-A3-f-d night FIX: the original body hardcoded the hd=128 layout (qf[.][128],
// 4 dims/lane, KPAD=132). On Flash-Next (head_dim=256) it staged HALF the q/K/V
// dims and wrote SMEM out of bounds for e >= 132 — progressively corrupting
// attention from the second chunk position on (prefillx evidence: C=16 state
// maxd 3.69 vs 1.45 flash-off; C=32/60 first-token MISMATCH; flash off → every
// width matches). The body is now a template on the head-dim layout; hd=256
// instantiates BT=8/BKV=16 so static SMEM stays < 48 KB (qf 8K + Ks/Vs 33.3K
// + P 0.5K ≈ 41 KB). Any other head_dim falls back to the sweep kernel.
//
// PQ8 item 2 (VEC = true, xq_attn_prefill_flash256v): the same kernel with two data-movement
// changes and NO arithmetic change — Ks/Vs/qf hold the same f32 values and every score / PV / gate
// op runs in the same order, so attn is bit-identical to VEC = false (GB10_PQ8_XCHECK diffs it):
//  (a) staging: the scalar loop did, per element, a runtime-hd integer divide + modulo, a runtime
//      format branch and ONE scalar load (q8: a byte load + an f16 scale load + an unpack, 32
//      dependent iterations per thread per tile). Here a thread stages 8-dim groups through
//      xq_kv_ld8 with the format a compile-time constant (--probe-exl3-kvq: ld8 == ld bitwise on
//      every format): q8 = one 8-B code load + one scale load per 8 values, f32 = two 16-B loads;
//      every item's loads issue before any store (masked rows load a valid row and are zeroed by a
//      select: the old `ok ? ld : 0.0f`, under the scalar element owner's mask — see PQ8-v2 below),
//      then two 16-B smem stores per item;
//  (b) scores: d = fma(q[e], k[e], d) over e ascending from +0 — the chain the scalar loop compiles
//      to (fma.rn.f32 from 0f00000000) — read as float4 from qf / Ks: 3 smem wavefronts per 4 dims
//      (qf broadcast + 16 lanes x 16 B at bank 4*lane+e, conflict-free) instead of 12 (scalar Ks is
//      2-way conflicted at KPAD 260 over 16 lanes). This loop was the kernel's smem bottleneck.
//      (PQ8-v2 SASS check: ptxas already emits LDS.128 + an ascending FFMA chain for flash256's
//      scalar score loop, so (b) changes the PTX only; the served-code saving is (a).)
// PQ8-v2 FIX (the GPU XCHECK found ~48% of f16 halves differing): the math above IS identical (SASS
// of the served module: same FFMA chain from RZ, FMNMX/EX2 softmax, FFMA PV, div + sigmoid gate),
// but the v1 stager wrote a DIFFERENT shared-memory image. The scalar loop hands tile element
// (jj, e) to thread e (blockDim == hd == 256: x = tid + k*256 -> jj = k, e = tid), i.e. to warp
// e >> 5, and zeroes it unless jt + jj <= THAT warp's own q-row pmax (its per-warp jn); a warp whose
// row loop has ended stages nothing (the element keeps the previous tile's value). v1 staged rows
// w and w+8 (all dims of group `lane`) under warp w's jn — on a block's diagonal tile that differs
// from the scalar image in every q row but the block's first. v2 reproduces the scalar image
// exactly: the owner of dims grp*8..grp*8+7 is warp grp >> 2; the item is stored iff
// jt <= pbase + (grp >> 2) (owner still in its loop) and zeroed iff jt + jj > pbase + (grp >> 2);
// the body runs every warp to the block's last tile (pbase + BT - 1), each warp computing only
// while jt <= its own pmax — the scalar kernel's loop, with the early-exited warps' barriers made
// explicit. Host proof: exl3_forward.rs flash256_stage_tests (cargo test --lib flash256_stage).
// NOTE — the scalar image is NOT the causal-softmax attention: dims of earlier warps are zeroed in
// the keys/values a later q row reads on its diagonal tile (row t reads its own key with dims
// [0, 32*(t%8)) zeroed when pos0 % 8 == 0). v2 is bitwise flash256 by contract; correcting it is
// a behaviour change left to the owner: CAUSAL (xq_attn_prefill_flash256c, opt-in only via
// GB10_PQ8_FLASH=causal) stages every row <= the block's last pmax in full (pw = pbase + BT - 1
// for every item), so each warp's scores / PV see exactly the causal keys / values.
template <int FMT, int HD, int BT, int BKV, int KPAD, bool CAUSAL = false>
__device__ __forceinline__ void a3fg_stage_v(float (*Ks)[KPAD], float (*Vs)[KPAD], const XqKv& kc,
                                             int jt, int pbase, int tid) {
    constexpr int G = HD / 8;                     // 8-dim groups per row
    constexpr int NTHR = BT * 32;
    constexpr int ITEMS = BKV * 2 * G;            // (row, K|V, group) items per tile
    static_assert(ITEMS % NTHR == 0, "a3fg_stage_v: items must tile the block");
    // The scalar-image owner map (element e -> thread e) needs blockDim == HD. CAUSAL staging has
    // no owner map (every row <= the block's last pmax is staged in full): any block shape works
    // (w5/PFIX: the hd=128 twin runs 512 threads over 128 dims).
    static_assert(CAUSAL || NTHR == HD, "a3fg_stage_v: the scalar owner map (element e -> thread e) needs blockDim == HD");
    constexpr int IPT = ITEMS / NTHR;
    float o[IPT][8];
    #pragma unroll
    for (int k = 0; k < IPT; k++) {
        const int it = tid + k * NTHR;
        const int grp = it % G, jj = (it / G) % BKV, isv = it / (G * BKV);
        // pmax of the scalar owner warp of dims grp*8..+7 (CAUSAL: the block's last row's pmax)
        const int pw = CAUSAL ? pbase + BT - 1 : pbase + (grp >> 2);
        const int p = min(jt + jj, pw);           // a valid row (<= the block's last pmax)
        xq_kv_ld8(o[k], (isv ? kc.v : kc.k) + (long long)p * kc.rb, FMT, HD, grp * 8);
    }
    #pragma unroll
    for (int k = 0; k < IPT; k++) {
        const int it = tid + k * NTHR;
        const int grp = it % G, jj = (it / G) % BKV, isv = it / (G * BKV);
        const int pw = CAUSAL ? pbase + BT - 1 : pbase + (grp >> 2);
        if (jt <= pw) {                           // the owner warp is still in its row loop
            const bool ok = jt + jj <= pw;        // == the owner's `jj < jn`
            float* dst = (isv ? &Vs[jj][0] : &Ks[jj][0]) + grp * 8;
            *reinterpret_cast<float4*>(dst) = make_float4(ok ? o[k][0] : 0.0f, ok ? o[k][1] : 0.0f,
                                                          ok ? o[k][2] : 0.0f, ok ? o[k][3] : 0.0f);
            *reinterpret_cast<float4*>(dst + 4) = make_float4(ok ? o[k][4] : 0.0f, ok ? o[k][5] : 0.0f,
                                                              ok ? o[k][6] : 0.0f, ok ? o[k][7] : 0.0f);
        }
    }
}
// VEC: 0 = scalar staging (flash / flash256), 1 = the flash256v twin, 2 = CAUSAL staging (flash256c,
// flash128c).
template <int HD, int BT, int BKV, int KPAD, int PL, int VEC = 0>
__device__ __forceinline__ void a3fg_flash_body(
        __half* __restrict__ attn, const __half* __restrict__ qg,
        const float* __restrict__ kcvc, const __half* __restrict__ qnkn,
        const float* __restrict__ cossin, int nh_nkv, int hd_rdim, int max_pos,
        int slot_pos0, int C, float eps) {
    const int nkv = nh_nkv & 0xFFFF;
    const int nh = ((unsigned)nh_nkv) >> 16;
    const int hd = hd_rdim & 0xFFFF;
    const int rdim = ((unsigned)hd_rdim) >> 16;
    const int slot = slot_pos0 & 0xFFF;
    const int pos0 = ((unsigned)slot_pos0) >> 12;
    const int head = blockIdx.y;
    const int t0 = blockIdx.x * BT;
    const int warp = threadIdx.x >> 5;   // BT warps (blockDim BT*32)
    const int lane = threadIdx.x & 31;
    const int r = warp;                  // q row within the tile
    const int t = t0 + r;                // chunk token index
    const int kvh = head / (nh / nkv);
    const float scale = rsqrtf((float)hd);

    // PQ8 (VEC): 16-B aligned for the float4 staging stores / score reads (rows are 16-B
    // multiples: HD*4 and KPAD*4 = 1040); VEC = false keeps the pre-PQ8 4-B alignment.
    __shared__ alignas(VEC ? 16 : 4) float qf[BT][HD];        // normalized+roped q rows
    __shared__ alignas(VEC ? 16 : 4) float Ks[BKV][KPAD]; // staged K tile
    __shared__ alignas(VEC ? 16 : 4) float Vs[BKV][KPAD]; // staged V tile
    __shared__ float P[BT][BKV];    // exp scores of current tile

    const long long qrow = (long long)t * nh * hd * 2 + (long long)head * hd * 2;

    // ---- normalize + rope this row's q. HD dims over 32 lanes = PL/lane.
    float qv[PL], qw[PL];
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < PL; i++) {
        int d = lane + 32 * i;
        qv[i] = xq_h2f(qg[qrow + d]);
        ss += qv[i] * qv[i];
        qw[i] = 1.0f + xq_h2f(qnkn[d]);
    }
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xffffffff, ss, off);
    ss = __shfl_sync(0xffffffff, ss, 0);
    const float inv = rsqrtf(ss / (float)hd + eps);
    __syncwarp();
#pragma unroll
    for (int i = 0; i < PL; i++) qf[r][lane + 32 * i] = qv[i] * inv * qw[i];
    __syncwarp();
    // rope: x1 at d (< rdim/2), partner x2 at d + rdim/2 — SMEM read, warp-local.
#pragma unroll
    for (int i = 0; i < PL; i++) {
        int d = lane + 32 * i;
        if (d < rdim / 2) {
            float x1 = qf[r][d], x2 = qf[r][d + rdim / 2];
            const float* c = cossin + (long long)t * 2 * rdim;
            const float* s = c + rdim;
            qf[r][d] = x1 * c[d] - x2 * s[d];
            qf[r][d + rdim / 2] = x2 * c[d] + x1 * s[d];
        }
        // dims >= rdim: already in place (pass-through).
    }
    __syncwarp();

    const XqKv kc = xq_kv_make(kcvc, max_pos, hd, nkv, slot, kvh);
    if (kc.fmt >= XQ_KV_Q8) {   // WP25 q8: this warp's q row into the cache's rotated basis
#pragma unroll
        for (int i = 0; i < PL; i++) qf[r][lane + 32 * i] = xq_h32r(qf[r][lane + 32 * i], lane);
        __syncwarp();
    }
    const int pmax = pos0 + t; // last visible position (causal)

    float m = -INFINITY, l = 0.0f, acc[PL];
#pragma unroll
    for (int i = 0; i < PL; i++) acc[i] = 0.0f;

    const int tid = threadIdx.x;
    // PQ8-v2 (VEC): every warp walks to the block's last tile (its last row's pmax) so each scalar
    // owner's elements get staged (a3fg_stage_v); a warp computes only while jt <= its own pmax.
    const int pbase = pos0 + t0;
    const int pend = VEC ? pbase + BT - 1 : pmax;

    for (int jt = 0; jt <= pend; jt += BKV) {
        const int jn = min(BKV, pmax + 1 - jt);
        // ---- cooperative stage of K/V tile (whole block; padded SMEM rows)
        __syncthreads(); // previous tile's consumers done before overwrite
        if constexpr (VEC != 0) {
            // PQ8 (a): 8-dim groups, format a compile-time constant (hd == HD: host dispatch)
            switch (kc.fmt) {
                case XQ_KV_F32: a3fg_stage_v<XQ_KV_F32, HD, BT, BKV, KPAD, VEC == 2>(Ks, Vs, kc, jt, pbase, tid); break;
                case XQ_KV_F16: a3fg_stage_v<XQ_KV_F16, HD, BT, BKV, KPAD, VEC == 2>(Ks, Vs, kc, jt, pbase, tid); break;
                case XQ_KV_FP8: a3fg_stage_v<XQ_KV_FP8, HD, BT, BKV, KPAD, VEC == 2>(Ks, Vs, kc, jt, pbase, tid); break;
                case XQ_KV_Q4K: a3fg_stage_v<XQ_KV_Q4K, HD, BT, BKV, KPAD, VEC == 2>(Ks, Vs, kc, jt, pbase, tid); break;
                default:        a3fg_stage_v<XQ_KV_Q8, HD, BT, BKV, KPAD, VEC == 2>(Ks, Vs, kc, jt, pbase, tid); break;
            }
        } else {
        for (int x = tid; x < BKV * hd; x += blockDim.x) {
            int jj = x / hd, e = x % hd;
            bool ok = jj < jn;
            Ks[jj][e] = ok ? xq_kv_k(kc, jt + jj, e) : 0.0f;
            Vs[jj][e] = ok ? xq_kv_v(kc, jt + jj, e) : 0.0f;
        }
        }
        __syncthreads();
        if (VEC && jt > pmax) continue;   // warp-uniform: this row's loop is over (scalar: warp left it)
        // ---- scores: lane j owns position jt+j (serial fp32 dot over SMEM)
        float s = -INFINITY;
        if (lane < jn) {
            float d = 0.0f;
            if constexpr (VEC != 0) {
                // PQ8 (b): the same fma chain from +0, e ascending, read 4 dims at a time
                const float4* q4 = reinterpret_cast<const float4*>(&qf[r][0]);
                const float4* k4 = reinterpret_cast<const float4*>(&Ks[lane][0]);
                #pragma unroll 16
                for (int e4 = 0; e4 < HD / 4; e4++) {
                    const float4 a = q4[e4], b = k4[e4];
                    d = __fmaf_rn(a.x, b.x, d);
                    d = __fmaf_rn(a.y, b.y, d);
                    d = __fmaf_rn(a.z, b.z, d);
                    d = __fmaf_rn(a.w, b.w, d);
                }
            } else {
            for (int e = 0; e < hd; e++) d += qf[r][e] * Ks[lane][e];
            }
            s = d * scale;
        }
        float mtile = s;
        for (int off = 16; off > 0; off >>= 1)
            mtile = fmaxf(mtile, __shfl_down_sync(0xffffffff, mtile, off));
        mtile = __shfl_sync(0xffffffff, mtile, 0);
        // ---- online-softmax state update with rescale (registers per warp)
        float mnew = fmaxf(m, mtile);
        float factor = __expf(m - mnew); // first tile: exp(-inf)=0 → clean reset
        m = mnew;
        l *= factor;
#pragma unroll
        for (int i = 0; i < PL; i++) acc[i] *= factor;
        float p = (lane < jn) ? __expf(s - m) : 0.0f;
        if (lane < BKV) P[r][lane] = p; // BKV < 32: keep idle lanes in-bounds
        float lp = p;
        for (int off = 16; off > 0; off >>= 1) lp += __shfl_down_sync(0xffffffff, lp, off);
        l += __shfl_sync(0xffffffff, lp, 0);
        __syncwarp();
        // ---- accumulate V: each lane strides the tile's rows for its PL dims
        for (int jj = 0; jj < jn; jj++) {
            float pj = P[r][jj];
#pragma unroll
            for (int i = 0; i < PL; i++) acc[i] += pj * Vs[jj][lane + 32 * i];
        }
        __syncwarp();
    }
    if (t < C) {
#pragma unroll
        for (int i = 0; i < PL; i++) {
            int e = lane + 32 * i;
            float g = xq_h2f(qg[qrow + hd + e]);
            const float o = xq_kv_rot(acc[i] / l, kc.fmt, lane);
            attn[(long long)t * nh * hd + (long long)head * hd + e] =
                xq_f2h(o * xq_sig(g));
        }
    }
}

extern "C" __global__ void xq_attn_prefill_flash(__half* __restrict__ attn,
                                                 const __half* __restrict__ qg,
                                                 const float* __restrict__ kcvc,
                                                 const __half* __restrict__ qnkn,
                                                 const float* __restrict__ cossin,
                                                 int nh_nkv, int hd_rdim, int max_pos,
                                                 int slot_pos0, int C, float eps) {
    XQ_PDL_ENTRY();
    a3fg_flash_body<128, 16, 32, 132, 4>(attn, qg, kcvc, qnkn, cossin,
                                         nh_nkv, hd_rdim, max_pos, slot_pos0, C, eps);
}

// hd=256 shape (Flash-Next): BT=8 rows/block, BKV=16 kv/tile — SMEM ≈ 41 KB.
extern "C" __global__ void xq_attn_prefill_flash256(__half* __restrict__ attn,
                                                    const __half* __restrict__ qg,
                                                    const float* __restrict__ kcvc,
                                                    const __half* __restrict__ qnkn,
                                                    const float* __restrict__ cossin,
                                                    int nh_nkv, int hd_rdim, int max_pos,
                                                    int slot_pos0, int C, float eps) {
    XQ_PDL_ENTRY();
    a3fg_flash_body<256, 8, 16, 260, 8>(attn, qg, kcvc, qnkn, cossin,
                                        nh_nkv, hd_rdim, max_pos, slot_pos0, C, eps);
}

// PQ8 item 2: the bit-identical twin of xq_attn_prefill_flash256 (vector staging + float4 scores;
// see a3fg_stage_v — PQ8-v2 stages the scalar kernel's exact smem image). Host: hd == 256 and
// blockDim 256 only (the body indexes by HD), GB10_PQ8_FLASH=0 keeps the above.
extern "C" __global__ void xq_attn_prefill_flash256v(__half* __restrict__ attn,
                                                     const __half* __restrict__ qg,
                                                     const float* __restrict__ kcvc,
                                                     const __half* __restrict__ qnkn,
                                                     const float* __restrict__ cossin,
                                                     int nh_nkv, int hd_rdim, int max_pos,
                                                     int slot_pos0, int C, float eps) {
    XQ_PDL_ENTRY();
    a3fg_flash_body<256, 8, 16, 260, 8, 1>(attn, qg, kcvc, qnkn, cossin,
                                           nh_nkv, hd_rdim, max_pos, slot_pos0, C, eps);
}

// PQ8-v2 OPT-IN (GB10_PQ8_FLASH=causal; NOT bitwise flash256 — a behaviour change): the flash256v
// body with CAUSAL staging — every key/value row a q row of the block may read is staged in full,
// so the diagonal tile no longer loses the dims of earlier warps (see the note above a3fg_stage_v).
extern "C" __global__ void xq_attn_prefill_flash256c(__half* __restrict__ attn,
                                                     const __half* __restrict__ qg,
                                                     const float* __restrict__ kcvc,
                                                     const __half* __restrict__ qnkn,
                                                     const float* __restrict__ cossin,
                                                     int nh_nkv, int hd_rdim, int max_pos,
                                                     int slot_pos0, int C, float eps) {
    XQ_PDL_ENTRY();
    a3fg_flash_body<256, 8, 16, 260, 8, 2>(attn, qg, kcvc, qnkn, cossin,
                                           nh_nkv, hd_rdim, max_pos, slot_pos0, C, eps);
}

// =====================================================================================
// PFTC (owner 2026-11-07, #1 priority): tensor-core dense prefill attention for the EXL3 packs WITHOUT
// sparse attention (Qwen 3.5/3.6 35B / 27B / 9B; hd = 256). The scalar flash256 body above does Q.K^T and
// P.V on FP32 cores, 8 q rows x ONE head per block, so every K/V tile is staged once per head of its GQA
// group: 35B TP1 128K TTFT 433 s, 27B TP2 128K 553 s. This is the trunk's attn_prefill_fa_b design
// (gpu_batch.cu: 96 rows = Br_t tokens x g heads of ONE kv head per block, so the g heads share every
// staged K/V tile; m16n8k16 MMA; online softmax in registers) on the EXL3 cache: K/V rows are read
// through xq_kv_ld8 (f32 / f16 / fp8 / q8, cache basis) and staged as f16. Three launches:
//   xq_attn_pf_qstage  q rows normed + roped (+ q8: into the cache's rotated basis) -> f16 [C][nh][hd]
//   xq_attn_pf_tc256   attention -> un-gated f32 o/l [C][nh][hd] (cache basis)
//   xq_attn_pf_tc_epi  q8 rotate back + gate: attn = f16(rot(o) * sig(g)), flash256's closing expression
// Not bitwise to flash256 (f16 operands, P rounded to f16 before P.V, different summation order).
// Flash-Next never takes it (the host gates it on a layer without an indexer).
// =====================================================================================
#define XQ_PFTC_BC 32
#define XQ_PFTC_THREADS 192

// grid (C*nh) 1-D, block hd (256 = 8 warps: warp w holds dims 32w..32w+31 = one H32 group, lane = dim & 31).
extern "C" __global__ void xq_attn_pf_qstage(__half* __restrict__ qs, const __half* __restrict__ qg,
                                             const __half* __restrict__ qnkn, const float* __restrict__ cossin,
                                             int nh_nkv, int hd_rdim, int mpf, float eps) {
    XQ_PDL_ENTRY();
    __shared__ float red[32];
    __shared__ float qn[512];
    const int nh = ((unsigned)nh_nkv) >> 16;
    const int hd = hd_rdim & 0xFFFF;
    const int rdim = ((unsigned)hd_rdim) >> 16;
    const int t = blockIdx.x / nh, head = blockIdx.x % nh;
    const int e = threadIdx.x, lane = e & 31, warp = e >> 5;
    const long long qrow = (long long)t * nh * hd * 2 + (long long)head * hd * 2;
    const float v = xq_h2f(qg[qrow + e]);
    float ss = v * v;
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, off);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float x = lane < (int)(blockDim.x >> 5) ? red[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) x += __shfl_down_sync(0xffffffffu, x, off);
        if (lane == 0) red[0] = x;
    }
    __syncthreads();
    const float inv = rsqrtf(red[0] / (float)hd + eps);
    qn[e] = v * inv * (1.0f + xq_h2f(qnkn[e]));
    __syncthreads();
    // rope: x1 at d (< rdim/2), partner x2 at d + rdim/2 (flash256's rotate-half); dims >= rdim pass through
    const int h2 = rdim / 2;
    const float* c = cossin + (long long)t * 2 * rdim;
    const float* s = c + rdim;
    float q = qn[e];
    if (e < h2)        q = qn[e] * c[e] - qn[e + h2] * s[e];
    else if (e < rdim) { const int d = e - h2; q = qn[e] * c[d] + qn[d] * s[d]; }
    if (xq_kv_fmt(mpf) >= XQ_KV_Q8) q = xq_h32r(q, lane);   // WP25 q8: into the cache's rotated basis
    qs[((long long)t * nh + head) * hd + e] = __float2half_rn(q);
}

__device__ __forceinline__ unsigned xq_pftc_h2(float lo, float hi) {
    __half2 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<unsigned*>(&v);
}
__device__ __forceinline__ void xq_pftc_mma(float* d, const unsigned* a, const unsigned* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void xq_pftc_ldm_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void xq_pftc_ldm_x4_trans(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

// grid (ceil(C / Br_t), nkv), block 192 (6 warps x 16 rows = 96 rows = Br_t tokens x g heads, token-major),
// static smem: one K and one V tile of 32 keys x 256 dims f16 (32 KB), Swizzle<3,3,3> as attn_prefill_fa_b.
// Keys 0..=pos0+t per row (causal; the chunk's own K/V rows are already in the cache). g must divide 96.
extern "C" __global__ void __launch_bounds__(XQ_PFTC_THREADS, 1)
xq_attn_pf_tc256(float* __restrict__ of, const __half* __restrict__ qs, const float* __restrict__ kcvc,
                 int nh_nkv, int mpf, int slot_pos0, int C) {
    XQ_PDL_ENTRY();
    constexpr int HD = 256;
    __shared__ __align__(16) __half Ks[XQ_PFTC_BC * HD];
    __shared__ __align__(16) __half Vs[XQ_PFTC_BC * HD];
    const int nkv = nh_nkv & 0xFFFF;
    const int nh = ((unsigned)nh_nkv) >> 16;
    const int g = nh / nkv;
    const int Br_t = 96 / g;
    const int slot = slot_pos0 & 0xFFF;
    const int pos0 = ((unsigned)slot_pos0) >> 12;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, tid = (int)threadIdx.x;
    const int kvh = blockIdx.y;
    const int t0 = (int)(gridDim.x - 1 - blockIdx.x) * Br_t;      // longest blocks first
    const int pc = pos0 + C;
    const int kmax = min(pos0 + t0 + Br_t - 1, pc - 1);
    const XqKv kc = xq_kv_make(kcvc, mpf, HD, nkv, slot, kvh);

    const int rr0 = 16 * warp + (lane >> 2), rr1 = rr0 + 8;
    const int ttok0 = t0 + rr0 / g, ttok1 = t0 + rr1 / g;
    const int hed0 = kvh * g + rr0 % g, hed1 = kvh * g + rr1 % g;
    const int ctok0 = min(ttok0, C - 1), ctok1 = min(ttok1, C - 1);  // dead rows: safe loads, no stores
    const long long qb0 = ((long long)ctok0 * nh + hed0) * HD;
    const long long qb1 = ((long long)ctok1 * nh + hed1) * HD;
    const int qpos0 = pos0 + ttok0, qpos1 = pos0 + ttok1;

    unsigned qf[HD / 16][4];
    #pragma unroll
    for (int ks = 0; ks < HD / 16; ks++) {
        const int doff = 16 * ks + 2 * (lane & 3);
        qf[ks][0] = *reinterpret_cast<const unsigned*>(qs + qb0 + doff);
        qf[ks][1] = *reinterpret_cast<const unsigned*>(qs + qb1 + doff);
        qf[ks][2] = *reinterpret_cast<const unsigned*>(qs + qb0 + doff + 8);
        qf[ks][3] = *reinterpret_cast<const unsigned*>(qs + qb1 + doff + 8);
    }
    float o[HD / 8][4];
    #pragma unroll
    for (int j = 0; j < HD / 8; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
    const float rsqrt_hd = rsqrtf((float)HD);

    for (int s0 = 0; s0 <= kmax; s0 += XQ_PFTC_BC) {
        __syncthreads();                                   // the previous tile's readers are done
        for (int c = tid; c < XQ_PFTC_BC * (HD >> 3); c += XQ_PFTC_THREADS) {
            const int key = c / (HD >> 3), chunk = c % (HD >> 3);
            const int gkey = s0 + key;
            float kf[8], vf[8];
            if (gkey <= kmax) {
                xq_kv_ld8(kf, kc.k + (long long)gkey * kc.rb, kc.fmt, HD, chunk << 3);
                xq_kv_ld8(vf, kc.v + (long long)gkey * kc.rb, kc.fmt, HD, chunk << 3);
            } else {
                #pragma unroll
                for (int i = 0; i < 8; i++) { kf[i] = 0.f; vf[i] = 0.f; }
            }
            const int cs = (chunk & 24) | ((chunk & 7) ^ (key & 7));  // Swizzle<3,3,3>
            *reinterpret_cast<uint4*>(Ks + key * HD + (cs << 3)) =
                make_uint4(xq_pftc_h2(kf[0], kf[1]), xq_pftc_h2(kf[2], kf[3]), xq_pftc_h2(kf[4], kf[5]), xq_pftc_h2(kf[6], kf[7]));
            *reinterpret_cast<uint4*>(Vs + key * HD + (cs << 3)) =
                make_uint4(xq_pftc_h2(vf[0], vf[1]), xq_pftc_h2(vf[2], vf[3]), xq_pftc_h2(vf[4], vf[5]), xq_pftc_h2(vf[6], vf[7]));
        }
        __syncthreads();

        // ===== Q.K^T (K[key][d] is the B .col operand: ldmatrix without transpose) =====
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
                const unsigned addr = (unsigned)__cvta_generic_to_shared(Ks + lkey * HD + (cs << 3));
                unsigned r0, r1, r2, r3;
                xq_pftc_ldm_x4(r0, r1, r2, r3, addr);
                const unsigned ba[2] = {r0, r1}, bb[2] = {r2, r3};
                xq_pftc_mma(s[2 * j2], qf[ks], ba);
                xq_pftc_mma(s[2 * j2 + 1], qf[ks], bb);
            }
        }

        // ===== online softmax (attn_prefill_fa_b's guards) =====
        const bool masked_tile = (s0 + XQ_PFTC_BC > pos0 + t0);
        float mx0 = -1e30f, mx1 = -1e30f;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                float v = s[j][e] * rsqrt_hd;
                if (masked_tile) {
                    const int key = s0 + 8 * j + 2 * (lane & 3) + (e & 1);
                    if (key > (e < 2 ? qpos0 : qpos1)) v = -1e30f;
                }
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
        l0 = l0 * al0 + su0;  m0 = mn0;
        l1 = l1 * al1 + su1;  m1 = mn1;
        if (al0 != 1.0f) {
            #pragma unroll
            for (int j = 0; j < HD / 8; j++) { o[j][0] *= al0; o[j][1] *= al0; }
        }
        if (al1 != 1.0f) {
            #pragma unroll
            for (int j = 0; j < HD / 8; j++) { o[j][2] *= al1; o[j][3] *= al1; }
        }

        // ===== P -> A in-register repack (f16) + P.V (V via ldmatrix .trans) =====
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
            for (int jn = 0; jn < HD / 16; jn++) {
                int lkey, ldim;
                if (lane < 8)       { lkey = 16 * ks + lane;            ldim = 16 * jn; }
                else if (lane < 16) { lkey = 16 * ks + 8 + lane - 8;    ldim = 16 * jn; }
                else if (lane < 24) { lkey = 16 * ks + lane - 16;       ldim = 16 * jn + 8; }
                else                { lkey = 16 * ks + 8 + lane - 24;   ldim = 16 * jn + 8; }
                const int chunk = ldim >> 3;
                const int cs = (chunk & 24) | ((chunk & 7) ^ (lkey & 7));
                const unsigned addr = (unsigned)__cvta_generic_to_shared(Vs + lkey * HD + (cs << 3));
                unsigned r0, r1, r2, r3;
                xq_pftc_ldm_x4_trans(r0, r1, r2, r3, addr);
                const unsigned ba[2] = {r0, r1}, bb[2] = {r2, r3};
                xq_pftc_mma(o[2 * jn], pa[ks], ba);
                xq_pftc_mma(o[2 * jn + 1], pa[ks], bb);
            }
        }
    }

    const float il0 = (l0 > 0.f) ? 1.f / l0 : 0.f;
    const float il1 = (l1 > 0.f) ? 1.f / l1 : 0.f;
    if (ttok0 < C) {
        const long long ob = ((long long)ttok0 * nh + hed0) * HD;
        #pragma unroll
        for (int j = 0; j < HD / 8; j++)
            *reinterpret_cast<float2*>(of + ob + 8 * j + 2 * (lane & 3)) = make_float2(o[j][0] * il0, o[j][1] * il0);
    }
    if (ttok1 < C) {
        const long long ob = ((long long)ttok1 * nh + hed1) * HD;
        #pragma unroll
        for (int j = 0; j < HD / 8; j++)
            *reinterpret_cast<float2*>(of + ob + 8 * j + 2 * (lane & 3)) = make_float2(o[j][2] * il1, o[j][3] * il1);
    }
}

// ---- PFTC round 3 (owner 2026-11-08: prefill is the next priority after the long-context decode): the twin of
// xq_attn_pf_tc256 whose K/V tile no longer waits on global memory. xq_attn_pf_tc256 loaded each 32-key tile with
// synchronous global reads and only then ran the MMAs, so every tile paid the full DRAM/L2 latency with the tensor
// cores idle (35B q8 at 48K: ~15 TFLOPS of causal attention, ~52% of the prefill). Here the raw cache bytes of the
// NEXT tile are cp.async'ed into a one-tile smem buffer while the current tile computes; the convert to the f16
// swizzled tile reads those bytes through the same xq_kv_ld8 -> the same tile values -> bitwise == xq_attn_pf_tc256.
// Dynamic smem: Ks + Vs (32 KB) + 32 x 2 x rb raw bytes (q8 17 KB, f16 32 KB): the host opts in (> 48 KB).
__device__ __forceinline__ void xq_pftc_stage_raw(char* raw, const XqKv& kc, int s0, int kmax, int rb, int nc16, int tid) {
    const int per = 2 * nc16;                              // K row + V row, 16-B chunks
    for (int i = tid; i < XQ_PFTC_BC * per; i += XQ_PFTC_THREADS) {
        const int key = i / per, r = i - key * per;
        const int gkey = s0 + key;
        if (gkey <= kmax) {
            const int kvsel = r >= nc16 ? 1 : 0;
            const int c = r - kvsel * nc16;
            const char* src = (kvsel ? kc.v : kc.k) + (long long)gkey * kc.rb + c * 16;
            moe_coop_cp16(raw + (2 * key + kvsel) * rb + c * 16, src, 16);
        }
    }
}
extern "C" __global__ void __launch_bounds__(XQ_PFTC_THREADS, 1)
xq_attn_pf_tc256p(float* __restrict__ of, const __half* __restrict__ qs, const float* __restrict__ kcvc,
                  int nh_nkv, int mpf, int slot_pos0, int C) {
    XQ_PDL_ENTRY();
    constexpr int HD = 256;
    extern __shared__ __align__(16) unsigned char xq_pftc_dyn[];
    __half* Ks = reinterpret_cast<__half*>(xq_pftc_dyn);
    __half* Vs = Ks + XQ_PFTC_BC * HD;
    char* raw = reinterpret_cast<char*>(Vs + XQ_PFTC_BC * HD);       // [key][K|V][rb] raw cache bytes, one tile
    const int nkv = nh_nkv & 0xFFFF;
    const int nh = ((unsigned)nh_nkv) >> 16;
    const int g = nh / nkv;
    const int Br_t = 96 / g;
    const int slot = slot_pos0 & 0xFFF;
    const int pos0 = ((unsigned)slot_pos0) >> 12;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, tid = (int)threadIdx.x;
    const int kvh = blockIdx.y;
    const int t0 = (int)(gridDim.x - 1 - blockIdx.x) * Br_t;      // longest blocks first
    const int pc = pos0 + C;
    const int kmax = min(pos0 + t0 + Br_t - 1, pc - 1);
    const XqKv kc = xq_kv_make(kcvc, mpf, HD, nkv, slot, kvh);

    const int rr0 = 16 * warp + (lane >> 2), rr1 = rr0 + 8;
    const int ttok0 = t0 + rr0 / g, ttok1 = t0 + rr1 / g;
    const int hed0 = kvh * g + rr0 % g, hed1 = kvh * g + rr1 % g;
    const int ctok0 = min(ttok0, C - 1), ctok1 = min(ttok1, C - 1);  // dead rows: safe loads, no stores
    const long long qb0 = ((long long)ctok0 * nh + hed0) * HD;
    const long long qb1 = ((long long)ctok1 * nh + hed1) * HD;
    const int qpos0 = pos0 + ttok0, qpos1 = pos0 + ttok1;

    unsigned qf[HD / 16][4];
    #pragma unroll
    for (int ks = 0; ks < HD / 16; ks++) {
        const int doff = 16 * ks + 2 * (lane & 3);
        qf[ks][0] = *reinterpret_cast<const unsigned*>(qs + qb0 + doff);
        qf[ks][1] = *reinterpret_cast<const unsigned*>(qs + qb1 + doff);
        qf[ks][2] = *reinterpret_cast<const unsigned*>(qs + qb0 + doff + 8);
        qf[ks][3] = *reinterpret_cast<const unsigned*>(qs + qb1 + doff + 8);
    }
    float o[HD / 8][4];
    #pragma unroll
    for (int j = 0; j < HD / 8; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
    const float rsqrt_hd = rsqrtf((float)HD);

    const int rb = (int)kc.rb;                             // host: rb % 16 == 0, rb <= 512
    const int nc16 = rb >> 4;
    xq_pftc_stage_raw(raw, kc, 0, kmax, rb, nc16, tid);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    for (int s0 = 0; s0 <= kmax; s0 += XQ_PFTC_BC) {
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();                                   // raw tile s0 landed; the previous tile's readers are done
        for (int c = tid; c < XQ_PFTC_BC * (HD >> 3); c += XQ_PFTC_THREADS) {
            const int key = c / (HD >> 3), chunk = c % (HD >> 3);
            const int gkey = s0 + key;
            float kf[8], vf[8];
            if (gkey <= kmax) {
                xq_kv_ld8(kf, raw + (2 * key + 0) * rb, kc.fmt, HD, chunk << 3);
                xq_kv_ld8(vf, raw + (2 * key + 1) * rb, kc.fmt, HD, chunk << 3);
            } else {
                #pragma unroll
                for (int i = 0; i < 8; i++) { kf[i] = 0.f; vf[i] = 0.f; }
            }
            const int cs = (chunk & 24) | ((chunk & 7) ^ (key & 7));  // Swizzle<3,3,3>
            *reinterpret_cast<uint4*>(Ks + key * HD + (cs << 3)) =
                make_uint4(xq_pftc_h2(kf[0], kf[1]), xq_pftc_h2(kf[2], kf[3]), xq_pftc_h2(kf[4], kf[5]), xq_pftc_h2(kf[6], kf[7]));
            *reinterpret_cast<uint4*>(Vs + key * HD + (cs << 3)) =
                make_uint4(xq_pftc_h2(vf[0], vf[1]), xq_pftc_h2(vf[2], vf[3]), xq_pftc_h2(vf[4], vf[5]), xq_pftc_h2(vf[6], vf[7]));
        }
        __syncthreads();                                   // tiles ready; the raw buffer is free
        if (s0 + XQ_PFTC_BC <= kmax)                       // the next tile's bytes fly while this one computes
            xq_pftc_stage_raw(raw, kc, s0 + XQ_PFTC_BC, kmax, rb, nc16, tid);
        asm volatile("cp.async.commit_group;\n" ::: "memory");

        // ===== Q.K^T (K[key][d] is the B .col operand: ldmatrix without transpose) =====
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
                const unsigned addr = (unsigned)__cvta_generic_to_shared(Ks + lkey * HD + (cs << 3));
                unsigned r0, r1, r2, r3;
                xq_pftc_ldm_x4(r0, r1, r2, r3, addr);
                const unsigned ba[2] = {r0, r1}, bb[2] = {r2, r3};
                xq_pftc_mma(s[2 * j2], qf[ks], ba);
                xq_pftc_mma(s[2 * j2 + 1], qf[ks], bb);
            }
        }

        // ===== online softmax (attn_prefill_fa_b's guards) =====
        const bool masked_tile = (s0 + XQ_PFTC_BC > pos0 + t0);
        float mx0 = -1e30f, mx1 = -1e30f;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                float v = s[j][e] * rsqrt_hd;
                if (masked_tile) {
                    const int key = s0 + 8 * j + 2 * (lane & 3) + (e & 1);
                    if (key > (e < 2 ? qpos0 : qpos1)) v = -1e30f;
                }
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
        l0 = l0 * al0 + su0;  m0 = mn0;
        l1 = l1 * al1 + su1;  m1 = mn1;
        if (al0 != 1.0f) {
            #pragma unroll
            for (int j = 0; j < HD / 8; j++) { o[j][0] *= al0; o[j][1] *= al0; }
        }
        if (al1 != 1.0f) {
            #pragma unroll
            for (int j = 0; j < HD / 8; j++) { o[j][2] *= al1; o[j][3] *= al1; }
        }

        // ===== P -> A in-register repack (f16) + P.V (V via ldmatrix .trans) =====
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
            for (int jn = 0; jn < HD / 16; jn++) {
                int lkey, ldim;
                if (lane < 8)       { lkey = 16 * ks + lane;            ldim = 16 * jn; }
                else if (lane < 16) { lkey = 16 * ks + 8 + lane - 8;    ldim = 16 * jn; }
                else if (lane < 24) { lkey = 16 * ks + lane - 16;       ldim = 16 * jn + 8; }
                else                { lkey = 16 * ks + 8 + lane - 24;   ldim = 16 * jn + 8; }
                const int chunk = ldim >> 3;
                const int cs = (chunk & 24) | ((chunk & 7) ^ (lkey & 7));
                const unsigned addr = (unsigned)__cvta_generic_to_shared(Vs + lkey * HD + (cs << 3));
                unsigned r0, r1, r2, r3;
                xq_pftc_ldm_x4_trans(r0, r1, r2, r3, addr);
                const unsigned ba[2] = {r0, r1}, bb[2] = {r2, r3};
                xq_pftc_mma(o[2 * jn], pa[ks], ba);
                xq_pftc_mma(o[2 * jn + 1], pa[ks], bb);
            }
        }
    }

    const float il0 = (l0 > 0.f) ? 1.f / l0 : 0.f;
    const float il1 = (l1 > 0.f) ? 1.f / l1 : 0.f;
    if (ttok0 < C) {
        const long long ob = ((long long)ttok0 * nh + hed0) * HD;
        #pragma unroll
        for (int j = 0; j < HD / 8; j++)
            *reinterpret_cast<float2*>(of + ob + 8 * j + 2 * (lane & 3)) = make_float2(o[j][0] * il0, o[j][1] * il0);
    }
    if (ttok1 < C) {
        const long long ob = ((long long)ttok1 * nh + hed1) * HD;
        #pragma unroll
        for (int j = 0; j < HD / 8; j++)
            *reinterpret_cast<float2*>(of + ob + 8 * j + 2 * (lane & 3)) = make_float2(o[j][2] * il1, o[j][3] * il1);
    }
}

// grid (C*nh), block hd: q8 rotate back + gate — flash256's closing expression (element e = lane + 32*warp).
extern "C" __global__ void xq_attn_pf_tc_epi(__half* __restrict__ attn, const float* __restrict__ of,
                                             const __half* __restrict__ qg, int nh_nkv, int hd, int mpf) {
    XQ_PDL_ENTRY();
    const int nh = ((unsigned)nh_nkv) >> 16;
    const int t = blockIdx.x / nh, head = blockIdx.x % nh, e = threadIdx.x;
    const long long ob = ((long long)t * nh + head) * hd;
    const float o = xq_kv_rot(of[ob + e], xq_kv_fmt(mpf), e & 31);
    const float g = xq_h2f(qg[(long long)t * nh * hd * 2 + (long long)head * hd * 2 + hd + e]);
    attn[ob + e] = xq_f2h(o * xq_sig(g));
}

// w5/PFIX OPT-IN (the SAME GB10_PQ8_FLASH=causal switch; output-changing): the hd=128 causal twin.
// xq_attn_prefill_flash (the hd=128 shape: BT=16 / BKV=32, blockDim 512) has flash256's diagonal-tile
// defect under a different owner map: blockDim (512) != hd (128), so its scalar stager hands tile
// element (jj, e) to thread (jj % 4) * 128 + e, i.e. warp (jj % 4) * 4 + (e >> 5), and zeroes it
// (or, once that warp's row loop has ended, leaves the previous tile's value) against THAT warp's
// pmax — whole dim slices of the diagonal-tile rows a later q row reads are wrong. Host proof:
// exl3_forward.rs flash256_stage_tests::flash128c_stage_is_causal (cargo test --lib flash256_stage).
// xq_attn_prefill_flash has had NO host caller since 2463afc (S-A3-f-d night): hd=128 dense prefill
// rows run the sweep xq_attn_prefill_q, and that default is unchanged. With GB10_PQ8_FLASH=causal the
// hd=128 dense rows (t >= 16) take this kernel instead: the flash256c body (vector staging with
// block-wide causal pw = pbase + BT - 1, float4 fma score chain) at the hd=128 shape. SMEM: qf 8 KB
// + Ks/Vs 2 x 16.5 KB + P 2 KB = ~43 KB static (< 48 KB).
extern "C" __global__ void xq_attn_prefill_flash128c(__half* __restrict__ attn,
                                                     const __half* __restrict__ qg,
                                                     const float* __restrict__ kcvc,
                                                     const __half* __restrict__ qnkn,
                                                     const float* __restrict__ cossin,
                                                     int nh_nkv, int hd_rdim, int max_pos,
                                                     int slot_pos0, int C, float eps) {
    XQ_PDL_ENTRY();
    a3fg_flash_body<128, 16, 32, 132, 4, 2>(attn, qg, kcvc, qnkn, cossin,
                                           nh_nkv, hd_rdim, max_pos, slot_pos0, C, eps);
}

// ---- PLE conv over a chunk: 9-deep ring in registers (dilated taps at
// offsets 0/3/6 of the ring + cur), conv then ring-shift per token; the ring
// is stored back once at the end. Matches xq_ple_conv + xq_ple_state_shift.
extern "C" __global__ void xq_ple_conv_chunk(float* __restrict__ resid, const __half* __restrict__ gated,
                                             const __half* __restrict__ normed, __half* __restrict__ state,
                                             const __half* __restrict__ conv_w, int slot, int C) {
    XQ_PDL_ENTRY();
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= 10240) return;
    __half st[9];
    __half* sbase = state + ((long long)slot * 10240 + d) * 9;
    for (int a = 0; a < 9; a++) st[a] = sbase[a];
    const float w0 = xq_h2f(conv_w[d * 4 + 0]);
    const float w1 = xq_h2f(conv_w[d * 4 + 1]);
    const float w2 = xq_h2f(conv_w[d * 4 + 2]);
    const float w3 = xq_h2f(conv_w[d * 4 + 3]);
    for (int t = 0; t < C; t++) {
        const long long i = (long long)t * 10240 + d;
        const float cur = xq_h2f(normed[i]);
        const float acc = w0 * xq_h2f(st[0]) + w1 * xq_h2f(st[3])
                        + w2 * xq_h2f(st[6]) + w3 * cur;
        resid[i] += xq_h2f(gated[i]) + xq_silu(acc);
        for (int a = 0; a < 8; a++) st[a] = st[a + 1];
        st[8] = xq_f2h(cur);
    }
    for (int a = 0; a < 9; a++) sbase[a] = st[a];
}

// PACK1 LEG 1: segmented PLE conv chunk — grid (ceil(10240/256), k_seg), block 256.
// Per segment: the SAME 9-deep ring walk with the ring held in `state` plane blockIdx.y
// (the packed PLE shadow: verify_shadow_seg copies each segment's live PLE row to its own
// plane; the write-back lands in the shadow, never live) and rows offset by seg.row_start.
// Per-token op order identical to xq_ple_conv_chunk; per-segment bytes match the lone
// chunk bit for bit.
extern "C" __global__ void xq_ple_conv_chunk_seg(float* __restrict__ resid,
                                                 const __half* __restrict__ gated,
                                                 const __half* __restrict__ normed,
                                                 __half* __restrict__ state,
                                                 const __half* __restrict__ conv_w,
                                                 const int* __restrict__ seg) {
    XQ_PDL_ENTRY();
    const int s = blockIdx.y * 4;
    const int row0 = seg[s + 1], C = seg[s + 2];
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= 10240) return;
    __half st[9];
    __half* sbase = state + ((long long)blockIdx.y * 10240 + d) * 9;
    for (int a = 0; a < 9; a++) st[a] = sbase[a];
    const float w0 = xq_h2f(conv_w[d * 4 + 0]);
    const float w1 = xq_h2f(conv_w[d * 4 + 1]);
    const float w2 = xq_h2f(conv_w[d * 4 + 2]);
    const float w3 = xq_h2f(conv_w[d * 4 + 3]);
    for (int t = 0; t < C; t++) {
        const long long i = (long long)(row0 + t) * 10240 + d;
        const float cur = xq_h2f(normed[i]);
        const float acc = w0 * xq_h2f(st[0]) + w1 * xq_h2f(st[3])
                        + w2 * xq_h2f(st[6]) + w3 * cur;
        resid[i] += xq_h2f(gated[i]) + xq_silu(acc);
        for (int a = 0; a < 8; a++) st[a] = st[a + 1];
        st[8] = xq_f2h(cur);
    }
    for (int a = 0; a < 9; a++) sbase[a] = st[a];
}

