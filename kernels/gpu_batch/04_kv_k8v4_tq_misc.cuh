// gpu_batch/04_kv_k8v4_tq_misc.cuh: k8v4 and TurboQuant KV write/compact kernels, penalty ring and seed-advance helpers.
// Deterministic per position.
// Section 4 of 15 of gpu_batch.cu: included in order by gpu_batch.cu, one translation unit; text is moved verbatim.
// ===================================================================================================
// k8v4 KV cache (GB10_KV_K8V4=1) — int8 K + the q4 V, per-16 affine blocks, deterministic per
// position. K: per-block scale s = amax/127 stored as fp16 (RNE), codes = lrintf(x*127/amax)
// clamped to [-127, 127] (the -128 code unused, mirroring q4's ±7-of-±8), 16 B codes + 2 B fp16
// scale = 18 meaningful B in a 20 B/16 stride (2 B pad — keeps every row 4-B/u32-aligned). The
// V cache reuses the q4 signed-nibble layout byte-for-byte (kvq16_pack, the 12 B/16 stride).
// The int8 grid is 127 steps vs q4's 7 -> ~18x quieter on the score dot; the fp16 scale
// removes q4's ~3% e4m3 common-mode scale rounding. Same determinism contract as q4: the packed
// form of position p depends only on (kvh, p), so decode, verify, and every split dequantize to
// the SAME values — the lossless-MTP contract is preserved by construction.
// ===================================================================================================
#define KV8_ROW_BYTES(hd) (((hd) / KVQ_BLK) * 20)   // [16 B codes][2 B fp16 scale][2 B pad] — u32-aligned

__device__ __forceinline__ void kv8_pack16(const __nv_bfloat16* x, unsigned char* out) {
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < KVQ_BLK; i++) amax = fmaxf(amax, fabsf(b2f(x[i])));
    // Same convention as kvq16_pack: round the scale to its stored precision (fp16) first, then
    // code with the rounded value, so the coder and the readers agree on the scale exactly.
    const float s = amax > 0.f ? __half2float(__float2half(amax / 127.0f)) : 0.f;   // amax==0 -> s=0, codes 0
    const float inv_s = s > 0.f ? 1.0f / s : 0.f;
    signed char c[KVQ_BLK];
    #pragma unroll
    for (int i = 0; i < KVQ_BLK; i++) {
        int q = (int)lrintf(b2f(x[i]) * inv_s);              // RNE, same arithmetic order as kvq16_pack
        q = max(-127, min(127, q));
        c[i] = (signed char)q;
    }
    #pragma unroll
    for (int i = 0; i < KVQ_BLK; i++) out[i] = (unsigned char)c[i];
    *(unsigned short*)(out + KVQ_BLK) = __half_as_ushort(__float2half(s));   // fp16 scale, RNE, LE
    out[KVQ_BLK + 2] = 0;                                     // pad
    out[KVQ_BLK + 3] = 0;
}

// write_kv_b_k8v4 — quantize + write one token's K/V (decode/verify path).
// One block per (b, kvh, block16), the q4 launch shape: K block via kv8_pack16 into the K cache
// at the 20 B/16 stride, V block via the EXISTING kvq16_pack into the V cache at the 12 B/16 stride.
extern "C" __global__ void write_kv_b_k8v4(unsigned char* k_cache, unsigned char* v_cache,
        const __nv_bfloat16* k_new, const __nv_bfloat16* v_new,
        const int* pos, int stride, int nkv, int hd, int B, const int* slot_ids) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int nb = hd / KVQ_BLK;
    const int total = B * nkv * nb;
    if (idx >= total) return;
    const int b = idx / (nkv * nb);
    const int rem = idx % (nkv * nb);
    const int h = rem / nb;
    const int blk = rem % nb;
    const int slot = slot_ids[b];
    const long long crow = ((long long)slot * nkv + h) * (long long)stride + pos[b];
    const long long kcoff = crow * (long long)KV8_ROW_BYTES(hd) + blk * 20;
    const long long vcoff = crow * (long long)KVQ_ROW_BYTES(hd) + blk * 12;
    __nv_bfloat16 tmp[KVQ_BLK];
    #pragma unroll
    for (int i = 0; i < KVQ_BLK; i++)
        tmp[i] = k_new[((long long)b * nkv * hd + h * hd) + blk * KVQ_BLK + i];
    kv8_pack16(tmp, k_cache + kcoff);
    #pragma unroll
    for (int i = 0; i < KVQ_BLK; i++)
        tmp[i] = v_new[((long long)b * nkv * hd + h * hd) + blk * KVQ_BLK + i];
    kvq16_pack(tmp, v_cache + vcoff);
}

extern "C" __global__ void write_kv_b_k8v8(unsigned char* k_cache, unsigned char* v_cache,
        const __nv_bfloat16* k_new, const __nv_bfloat16* v_new,
        const int* pos, int stride, int nkv, int hd, int B, const int* slot_ids) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int nb = hd / KVQ_BLK;
    const int total = B * nkv * nb;
    if (idx >= total) return;
    const int b = idx / (nkv * nb);
    const int rem = idx % (nkv * nb);
    const int h = rem / nb;
    const int blk = rem % nb;
    const int slot = slot_ids[b];
    const long long crow = ((long long)slot * nkv + h) * (long long)stride + pos[b];
    const long long kcoff = crow * (long long)KV8_ROW_BYTES(hd) + blk * 20;
    const long long vcoff = crow * (long long)KV8_ROW_BYTES(hd) + blk * 20;
    __nv_bfloat16 tmp[KVQ_BLK];
    #pragma unroll
    for (int i = 0; i < KVQ_BLK; i++)
        tmp[i] = k_new[((long long)b * nkv * hd + h * hd) + blk * KVQ_BLK + i];
    kv8_pack16(tmp, k_cache + kcoff);
    #pragma unroll
    for (int i = 0; i < KVQ_BLK; i++)
        tmp[i] = v_new[((long long)b * nkv * hd + h * hd) + blk * KVQ_BLK + i];
    kv8_pack16(tmp, v_cache + vcoff);
}

// write_kv_prefill_k8v4 — quantize + append N tokens' K/V for one slot at pos_start..pos_start+N-1.
extern "C" __global__ void write_kv_prefill_k8v4(unsigned char* k_cache, unsigned char* v_cache,
        const __nv_bfloat16* k_new, const __nv_bfloat16* v_new,
        int stride, int nkv, int hd, int N, int pos_start) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int nb = hd / KVQ_BLK;
    const int total = N * nkv * nb;
    if (idx >= total) return;
    const int t = idx / (nkv * nb);
    const int rem = idx % (nkv * nb);
    const int h = rem / nb;
    const int blk = rem % nb;
    const long long crow = ((long long)h * (long long)stride) + (pos_start + t);
    const long long kcoff = crow * (long long)KV8_ROW_BYTES(hd) + blk * 20;
    const long long vcoff = crow * (long long)KVQ_ROW_BYTES(hd) + blk * 12;
    __nv_bfloat16 tmp[KVQ_BLK];
    #pragma unroll
    for (int i = 0; i < KVQ_BLK; i++)
        tmp[i] = k_new[((long long)t * nkv * hd + h * hd) + blk * KVQ_BLK + i];
    kv8_pack16(tmp, k_cache + kcoff);
    #pragma unroll
    for (int i = 0; i < KVQ_BLK; i++)
        tmp[i] = v_new[((long long)t * nkv * hd + h * hd) + blk * KVQ_BLK + i];
    kvq16_pack(tmp, v_cache + vcoff);
}

// compact_kv_k8v4 — verbatim copy of packed rows between the cache and a snapshot buffer
// (rollback / prefix-cache paths; the packed form is position-local, so a byte copy suffices).
// The K and V caches diverge in row size for the first time: K rows are (hd/16)*20 B, V rows
// stay (hd/16)*12 B. Each byte of the combined grid belongs to exactly one cache: the first
// k_rb bytes of a (k, h) row's scratch are the K row, the remaining v_rb the V row. The host
// sizes the two scratch buffers separately (len*nkv*k_rb and len*nkv*v_rb bytes).
extern "C" __global__ void compact_kv_k8v4(unsigned char* k_cache, unsigned char* v_cache,
    unsigned char* ks, unsigned char* vs, const int* src_pos, int len, int pos_start,
    int slot, int nkv, int stride, int hd, int dir) {
    const int k_rb = KV8_ROW_BYTES(hd);
    const int v_rb = KVQ_ROW_BYTES(hd);
    const int row_b = k_rb + v_rb;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = len * nkv * row_b;
    if (idx >= total) return;
    const int k = idx / (nkv * row_b);
    const int rem = idx % (nkv * row_b);
    const int h = rem / row_b;
    const int dv = rem % row_b;
    const long long cache_pos = (dir == 0) ? (pos_start + src_pos[k]) : (pos_start + k);
    const long long cbase = (((long long)slot * nkv + h) * stride + cache_pos);
    const long long sbase = ((long long)k * nkv + h);
    if (dv < k_rb) {
        const long long coff = cbase * k_rb + dv;
        const long long soff = sbase * k_rb + dv;
        if (dir == 0) { ks[soff] = k_cache[coff]; } else { k_cache[coff] = ks[soff]; }
    } else {
        const int vdv = dv - k_rb;
        const long long coff = cbase * v_rb + vdv;
        const long long soff = sbase * v_rb + vdv;
        if (dir == 0) { vs[soff] = v_cache[coff]; } else { v_cache[coff] = vs[soff]; }
    }
}


// k8v8 KV cache — int8 K + int8 V, per-16 affine blocks on BOTH sides (the k8v4 K layout
// applied to V: 16 B codes + 2 B fp16 scale, 20 B/16 stride). ~2.2x KV capacity vs bf16 AND the
// direct-read path for the p8-class verify kernel (block scales tile-align with 16-key tiles).
// Determinism contract identical to k8v4 (packed form depends only on (kvh, p)) — lossless MTP
// preserved by construction. The p8b verify kernel dequantizes in-register at stage time.
extern "C" __global__ void write_kv_prefill_k8v8(unsigned char* k_cache, unsigned char* v_cache,
        const __nv_bfloat16* k_new, const __nv_bfloat16* v_new,
        int stride, int nkv, int hd, int N, int pos_start) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int nb = hd / KVQ_BLK;
    const int total = N * nkv * nb;
    if (idx >= total) return;
    const int t = idx / (nkv * nb);
    const int rem = idx % (nkv * nb);
    const int h = rem / nb;
    const int blk = rem % nb;
    const long long crow = ((long long)h * (long long)stride) + (pos_start + t);
    const long long kcoff = crow * (long long)KV8_ROW_BYTES(hd) + blk * 20;
    const long long vcoff = crow * (long long)KV8_ROW_BYTES(hd) + blk * 20;
    __nv_bfloat16 tmp[KVQ_BLK];
    #pragma unroll
    for (int i = 0; i < KVQ_BLK; i++)
        tmp[i] = k_new[((long long)t * nkv * hd + h * hd) + blk * KVQ_BLK + i];
    kv8_pack16(tmp, k_cache + kcoff);
    #pragma unroll
    for (int i = 0; i < KVQ_BLK; i++)
        tmp[i] = v_new[((long long)t * nkv * hd + h * hd) + blk * KVQ_BLK + i];
    kv8_pack16(tmp, v_cache + vcoff);
}

// compact_kv_k8v8 — verbatim copy of packed rows between the cache and a snapshot buffer
// (tree-path compaction; the packed form is position-local, so a byte copy suffices).
// K AND V rows are both (hd/16)*20 B in k8v8 (the V cache has the K layout). Each byte of the
// combined grid belongs to exactly one cache: the first k_rb bytes of a (k, h) row's scratch
// are the K row, the remaining v_rb the V row. The host sizes the two scratch buffers
// separately (len*nkv*k_rb and len*nkv*v_rb bytes).
extern "C" __global__ void compact_kv_k8v8(unsigned char* k_cache, unsigned char* v_cache,
    unsigned char* ks, unsigned char* vs, const int* src_pos, int len, int pos_start,
    int slot, int nkv, int stride, int hd, int dir) {
    const int k_rb = KV8_ROW_BYTES(hd);
    const int v_rb = KV8_ROW_BYTES(hd);
    const int row_b = k_rb + v_rb;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = len * nkv * row_b;
    if (idx >= total) return;
    const int k = idx / (nkv * row_b);
    const int rem = idx % (nkv * row_b);
    const int h = rem / row_b;
    const int dv = rem % row_b;
    const long long cache_pos = (dir == 0) ? (pos_start + src_pos[k]) : (pos_start + k);
    const long long cbase = (((long long)slot * nkv + h) * stride + cache_pos);
    const long long sbase = ((long long)k * nkv + h);
    if (dv < k_rb) {
        const long long coff = cbase * k_rb + dv;
        const long long soff = sbase * k_rb + dv;
        if (dir == 0) { ks[soff] = k_cache[coff]; } else { k_cache[coff] = ks[soff]; }
    } else {
        const int vdv = dv - k_rb;
        const long long coff = cbase * v_rb + vdv;
        const long long soff = sbase * v_rb + vdv;
        if (dir == 0) { vs[soff] = v_cache[coff]; } else { v_cache[coff] = vs[soff]; }
    }
}

// dequant_kv_k8v4 — expand packed K (k8v4) to bf16 scratch for the prefill paths, in CACHE layout
// ([kvh][pos][hd]). The formula is IDENTICAL to the splitk reader's — bf16(fp32(int8 code) ×
// fp32(fp16 scale)), both upcasts exact, one rounding, position-stateless — so prefill and decode
// see the same values. K CHANNEL ONLY: the V cache reuses dequant_kv_q4 unchanged (the call sites
// already launch the dequant twice, K then V). Same `out_stride` semantics as dequant_kv_q4.
extern "C" __global__ void dequant_kv_k8v4(__nv_bfloat16* out, const unsigned char* cache,
        int nkv, int stride, int hd, int n_pos, int pos_start, int out_stride) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int nb = hd / KVQ_BLK;
    const int total = n_pos * nkv * nb;
    if (idx >= total) return;
    const int t = idx / (nkv * nb);
    const int rem = idx % (nkv * nb);
    const int h = rem / nb;
    const int blk = rem % nb;
    const long long crow = ((long long)h * (long long)stride) + (pos_start + t);
    const unsigned char* in = cache + crow * (long long)KV8_ROW_BYTES(hd) + blk * 20;
    const float s = __half2float(__ushort_as_half(*(const unsigned short*)(in + KVQ_BLK)));
    float x[KVQ_BLK];
    #pragma unroll
    for (int i = 0; i < KVQ_BLK; i++) x[i] = (float)((signed char)in[i]) * s;
    __nv_bfloat16* o = out + ((long long)h * out_stride + t) * hd + blk * KVQ_BLK;
    #pragma unroll
    for (int i = 0; i < KVQ_BLK; i++) o[i] = f2b(x[i]);
}

// ===================================================================================================
// TurboQuant KV (E4, GB10_KV_TQ=1) — 3.5-bit pack-at-write rotated-domain KV cache.
// Reference: /tmp/tq_ref2 (REPORT.md + tq.py + goldens) + HY3_TURBOQUANT_KV_PLAN.md.
//
// Layout per (slot, kvh, position), hd == 128 ONLY (the d=128 reference layout):
//   K row — 52 B: [0,32)   2-bit Lloyd-Max codes (coord j at bits [2j, 2j+2), LSB-first bitstream)
//                 [32,48)  QJL signs (coord j = bit j; 1 -> +1, 0 -> -1)
//                 [48,50)  ||r|| fp16 LE (unit-key residual norm, fp16-rounded at pack)
//                 [50,52)  ||k|| fp16 LE
//   V row — 52 B:  [0,48)  3-bit Lloyd-Max codes (coord j at bits [3j, 3j+3), LSB-first)
//                  [48,50) ||v|| fp16 LE
//                  [50,52) pad (uniform 52-B row stride so K and V share one row size)
// Row index = (slot*nkv + kvh) * stride + pos, row-major; stride in positions.
//
// Math (row vectors; "apply Pi" = x @ Pi^T == Pi·x as a column-vector matvec):
//   K:  kn = ||k||; xhat = k/kn; y = Pi·xhat; idx = argmin |y - cb2| (ties -> higher index, the
//       reference's searchsorted-right pair rule); r = xhat - Pi^T cb2[idx];
//       sign = sign(S·r) with sign(0) := +1; rn = ||r||.  Pack (idx, sign, fp16 rn, fp16 kn).
//       Unbiased reconstruction: k~ = kn·(Pi^T cb2[idx] + sqrt(pi/2)/d · rn · S^T sign).
//   V:  vn = ||v||; y = Pi·(v/vn); idx = argmin |y - cb3|;  v~ = vn·Pi^T cb3[idx] (MSE).
//   Score (the dual dot; the engine's attention): s = kn·( <Pi q, cb2[idx]> + sqrt(pi/2)/d·rn·<S q, sign> )
//       — NO 1/sqrt(hd) factor: this estimator IS <q,k> (REPORT §4.1).
//   PV:  acc in the ROTATED domain (vn·cb3[idx]); the splitk merge applies out = Pi^T·acc once
//       per (token, head).
//
// Determinism: every row is a pure function of its (kvh, pos) inputs computed by ONE block with a
// FIXED thread mapping and fixed-order reductions (no atomics) — batch-invariant by construction.
// The fp16 norms are the __float2half (round-to-nearest-even) values, exactly the reference's
// numpy '<f2' pack.
// ===================================================================================================
#define TQ_HD 128
// TurboQuant K-channel bit width, selected at compile time by -DTQ_B3 (the b=3 variant:
// GB10_KV_TQ=3). The b=2 build (no define) is the golden-anchored E4 layout; the b=3 build
// packs the K codes at 3 bits (48 B) -> K row 68 B = codes[0,48) | signs[48,64) | rn[64,66)
// | kn[66,68); V rows stay 3-bit codes + fp16 norm (50 meaningful B) padded to the shared row
// size so K and V caches keep one uniform stride. Both layouts share the TQ_TAB_* tables (the
// b=3 K codebook IS the existing cb3 block — 8 ascending centroids).
#ifdef TQ_B3
#define TQ_K_BITS     3
#define TQ_ROW_BYTES  68
#define TQ_SIGN_OFF   48
#define TQ_RN_OFF     64
#define TQ_KN_OFF     66
#else
#define TQ_K_BITS     2
#define TQ_ROW_BYTES  52
#define TQ_SIGN_OFF   32
#define TQ_RN_OFF     48
#define TQ_KN_OFF     50
#endif
// Host table buffer (f32, uploaded once — src/gpu.rs build_tq_tables). Offsets in floats:
//   [0, 16384)       Pi    row-major Pi[j][i]
//   [16384, 32768)   PiT   row-major PiT[j][i] = Pi[i][j]
//   [32768, 49152)   S     row-major S[j][i]
//   [49152, 65536)   ST    row-major ST[j][i] = S[i][j]
//   [65536, 65540)   cb2   4 ascending 2-bit centroids
//   [65540, 65548)   cb3   8 ascending 3-bit centroids  <- K codebook for the b=3 build
//   [65548]          qjl_scale  sqrt(pi/2)/128 (== scale.bin)
#define TQ_TAB_PI    0
#define TQ_TAB_PIT   16384
#define TQ_TAB_S     32768
#define TQ_TAB_ST    49152
#define TQ_TAB_CB2   65536
#define TQ_TAB_CB3   65540
#define TQ_TAB_SCALE 65548
#if TQ_K_BITS == 3
#define TQ_ENCODE_SMEM_BYTES 4176   // tq_encode_rows dynamic smem (must match the host launch)
#else
#define TQ_ENCODE_SMEM_BYTES 4160   // tq_encode_rows dynamic smem (must match the host launch)
#endif

__device__ __forceinline__ float tq_half_at(const unsigned char* row, int off) {
    return __half2float(__ushort_as_half(*(const unsigned short*)(row + off)));
}

// Nearest centroid, ties -> HIGHER index (the reference's searchsorted-right + `<=` pair rule).
__device__ __forceinline__ int tq_quant2(const float y, const float* cb2) {
    const float d0 = fabsf(y - cb2[0]), d1 = fabsf(y - cb2[1]),
                d2 = fabsf(y - cb2[2]), d3 = fabsf(y - cb2[3]);
    int i = 0; float bd = d0;
    if (d1 <= bd) { i = 1; bd = d1; }
    if (d2 <= bd) { i = 2; bd = d2; }
    if (d3 <= bd) { i = 3; bd = d3; }
    return i;
}
__device__ __forceinline__ int tq_quant3(const float y, const float* cb3) {
    int i = 0; float bd = fabsf(y - cb3[0]);
    #pragma unroll
    for (int k = 1; k < 8; k++) { const float d = fabsf(y - cb3[k]); if (d <= bd) { i = k; bd = d; } }
    return i;
}

// One block = one (kvh, position); blockDim.x = 128 (one thread per coord). Encodes the K row
// (b-bit Lloyd-Max + QJL signs + fp16 ||r||/||k||, b = TQ_K_BITS) and the V row (3-bit + fp16
// ||v||) from bf16 inputs. Dynamic smem layout (floats):
//   xk[128] | xv[128] | wk[128] | wv[128] | ridx[128] | vidx[128] | sgn[128] | wpk[4] | wpv[4]
//   | bkn[1] | bvn[1] | brn[1]  — then bytes: kcs[128] | kss[128] | vcs[128] | krow[TQ_ROW_BYTES] | vrow[52].
__device__ __forceinline__ void tq_encode_rows(
        const float* tables,
        const __nv_bfloat16* krow_in, const __nv_bfloat16* vrow_in,
        unsigned char* kout, unsigned char* vout, float* sh) {
    const int t = threadIdx.x;
    const int warp = t >> 5, lane = t & 31;
    const float* pi  = tables + TQ_TAB_PI;
    const float* pit = tables + TQ_TAB_PIT;
    const float* s   = tables + TQ_TAB_S;
    const float* cb2 = tables + TQ_TAB_CB2;
    const float* cb3 = tables + TQ_TAB_CB3;
    float* xk = sh;            float* xv = sh + 128;
    float* wk = sh + 256;      float* wv = sh + 384;
    float* ridx = sh + 512;    float* vidx = sh + 640;
    float* sgn = sh + 768;
    float* wpk = sh + 896;     float* wpv = sh + 900;
    float* bkn = sh + 904;     float* bvn = sh + 908;    float* brn = sh + 912;
    unsigned char* kcs = (unsigned char*)(sh + 916);    // 128 B
    unsigned char* kss = kcs + 128;                     // 128 B
    unsigned char* vcs = kss + 128;                     // 128 B
    unsigned char* krow = vcs + 128;                    // TQ_ROW_BYTES
    unsigned char* vrow = krow + TQ_ROW_BYTES;          // 52 B (50 meaningful + 2 pad)

    xk[t] = b2f(krow_in[t]);
    xv[t] = b2f(vrow_in[t]);
    // per-warp sums of squares (deterministic halving tree), then fixed warp-order combine
    float pk = xk[t] * xk[t], pv = xv[t] * xv[t];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) { pk += __shfl_xor_sync(0xffffffffu, pk, off); pv += __shfl_xor_sync(0xffffffffu, pv, off); }
    __syncthreads();
    if (lane == 0) { wpk[warp] = pk; wpv[warp] = pv; }
    __syncthreads();
    if (t == 0) {
        bkn[0] = sqrtf(wpk[0] + wpk[1] + wpk[2] + wpk[3]);
        bvn[0] = sqrtf(wpv[0] + wpv[1] + wpv[2] + wpv[3]);
    }
    __syncthreads();
    const float kn = bkn[0], vn = bvn[0];
    xk[t] *= 1.0f / fmaxf(kn, 1e-30f);                 // unit key/value (clamp like the reference)
    xv[t] *= 1.0f / fmaxf(vn, 1e-30f);
    __syncthreads();
    // rotate: y = Pi·x  (y[t] = sum_i Pi[t][i]·x[i], fp32 ascending i)
    float acc = 0.0f;
    const float* pirow = pi + t * TQ_HD;
    #pragma unroll
    for (int i = 0; i < TQ_HD; i++) acc += pirow[i] * xk[i];
    wk[t] = acc;
    acc = 0.0f;
    #pragma unroll
    for (int i = 0; i < TQ_HD; i++) acc += pirow[i] * xv[i];
    wv[t] = acc;
    __syncthreads();
#if TQ_K_BITS == 3
    ridx[t] = (float)tq_quant3(wk[t], cb3);
#else
    ridx[t] = (float)tq_quant2(wk[t], cb2);
#endif
    vidx[t] = (float)tq_quant3(wv[t], cb3);
    __syncthreads();
    // residual in the ORIGINAL domain: r[t] = xhat[t] - (Pi^T cb)[t], (Pi^T cb)[t] = sum_j Pi[j][t]·cb[j]
    acc = 0.0f;
    const float* pitrow = pit + t * TQ_HD;
#if TQ_K_BITS == 3
    #pragma unroll
    for (int j = 0; j < TQ_HD; j++) acc += pitrow[j] * cb3[(int)ridx[j]];
#else
    #pragma unroll
    for (int j = 0; j < TQ_HD; j++) acc += pitrow[j] * cb2[(int)ridx[j]];
#endif
    const float r = xk[t] - acc;
    sgn[t] = r;                                        // stage r for the S·r matvec
    __syncthreads();
    acc = 0.0f;
    const float* srow = s + t * TQ_HD;
    #pragma unroll
    for (int i = 0; i < TQ_HD; i++) acc += srow[i] * sgn[i];
    const float sv = (acc >= 0.0f) ? 1.0f : -1.0f;    // sign(0) := +1; kept in a register —
    // sgn still holds the staged r and is NEVER overwritten (a cross-thread write here would race
    // the other warps' S·r reads).
    // rn = ||r|| (same fixed reduction)
    pk = r * r;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) pk += __shfl_xor_sync(0xffffffffu, pk, off);
    __syncthreads();
    if (lane == 0) wpk[warp] = pk;
    __syncthreads();
    if (t == 0) brn[0] = sqrtf(wpk[0] + wpk[1] + wpk[2] + wpk[3]);
    __syncthreads();
    const float rn = brn[0];
    // ---- stage + pack K row ----
    kcs[t] = (unsigned char)(int)ridx[t];
    kss[t] = (sv > 0.0f) ? 1u : 0u;
    __syncthreads();
#if TQ_K_BITS == 3
    if (t < 48) {                                      // 3-bit LSB-first bitstream (like the V pack)
        const int g = t;
        unsigned char v = 0;
        for (int j = (8 * g) / 3; j <= (8 * g + 7) / 3; j++) {
            if (j >= TQ_HD) break;
            const int shift = 3 * j;
            const unsigned char c = (unsigned char)(kcs[j] & 7u);
            if (shift >= 8 * g) {                      // starts in this byte
                const int lo = shift - 8 * g;          // 0..7; a 3-bit code may straddle
                const int take = min(3, 8 - lo);       // bits of c that fit in this byte
                v |= (unsigned char)((c & ((1u << take) - 1u)) << lo);
            } else {                                   // started in the previous byte
                const int hi = 8 * g - shift;          // 1..2
                v |= (unsigned char)(c >> hi);
            }
        }
        krow[g] = v;
    }
#else
    if (t < 32) {                                      // 4 two-bit codes per byte (LSB-first)
        unsigned char v = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) v |= (unsigned char)((kcs[4 * t + j] & 3u) << (2 * j));
        krow[t] = v;
    }
#endif
    if (t < 16) {                                      // 8 sign bits per byte (bit j = coord j)
        unsigned char v = 0;
        #pragma unroll
        for (int j = 0; j < 8; j++) v |= (unsigned char)(kss[8 * t + j] << j);
        krow[TQ_SIGN_OFF + t] = v;
    }
    if (t == 0) {                                      // fp16 LE norms (round-to-nearest-even)
        *(unsigned short*)(krow + TQ_RN_OFF) = __half_as_ushort(__float2half(rn));
        *(unsigned short*)(krow + TQ_KN_OFF) = __half_as_ushort(__float2half(kn));
    }
    // ---- stage + pack V row ----
    vcs[t] = (unsigned char)(int)vidx[t];
    __syncthreads();
    if (t < 48) {                                      // 3-bit LSB-first bitstream
        const int g = t;
        unsigned char v = 0;
        for (int j = (8 * g) / 3; j <= (8 * g + 7) / 3; j++) {
            if (j >= TQ_HD) break;
            const int shift = 3 * j;
            const unsigned char c = (unsigned char)(vcs[j] & 7u);
            if (shift >= 8 * g) {                      // starts in this byte
                const int lo = shift - 8 * g;          // 0..7; a 3-bit code may straddle
                const int take = min(3, 8 - lo);       // bits of c that fit in this byte
                v |= (unsigned char)((c & ((1u << take) - 1u)) << lo);
            } else {                                   // started in the previous byte
                const int hi = 8 * g - shift;          // 1..2
                v |= (unsigned char)(c >> hi);
            }
        }
        vrow[g] = v;
    }
    if (t == 0) *(unsigned short*)(vrow + 48) = __half_as_ushort(__float2half(vn));
    __syncthreads();
    // K rows are TQ_ROW_BYTES (52 for b=2, 68 for b=3); V rows carry 50 meaningful bytes + pad.
    if (t < TQ_ROW_BYTES) kout[t] = krow[t];
    if (t < 52) vout[t] = vrow[t];
}

// write_kv_b_tq — pack-at-write one token's K/V (decode/verify path). One block per (b, kvh).
// grid(B*nkv), blockDim 128, smem TQ_ENCODE_SMEM_BYTES.
extern "C" __global__ void write_kv_b_tq(unsigned char* k_cache, unsigned char* v_cache,
        const __nv_bfloat16* k_new, const __nv_bfloat16* v_new, const float* tables,
        const int* pos, int stride, int nkv, int B, const int* slot_ids) {
    const int idx = blockIdx.x;
    if (idx >= B * nkv) return;
    const int b = idx / nkv;
    const int h = idx % nkv;
    const int slot = slot_ids[b];
    const long long crow = ((long long)slot * nkv + h) * (long long)stride + pos[b];
    extern __shared__ float sh[];
    tq_encode_rows(tables,
        k_new + ((long long)b * nkv + h) * TQ_HD,
        v_new + ((long long)b * nkv + h) * TQ_HD,
        k_cache + crow * TQ_ROW_BYTES, v_cache + crow * TQ_ROW_BYTES, sh);
}

// write_kv_prefill_tq — pack-at-write N tokens' K/V for one slot at pos_start..pos_start+N-1.
extern "C" __global__ void write_kv_prefill_tq(unsigned char* k_cache, unsigned char* v_cache,
        const __nv_bfloat16* k_new, const __nv_bfloat16* v_new, const float* tables,
        int stride, int nkv, int N, int pos_start) {
    const int idx = blockIdx.x;
    if (idx >= N * nkv) return;
    const int t = idx / nkv;
    const int h = idx % nkv;
    const long long crow = ((long long)h * stride) + (pos_start + t);
    extern __shared__ float sh[];
    tq_encode_rows(tables,
        k_new + ((long long)t * nkv + h) * TQ_HD,
        v_new + ((long long)t * nkv + h) * TQ_HD,
        k_cache + crow * TQ_ROW_BYTES, v_cache + crow * TQ_ROW_BYTES, sh);
}

// rotate_q_tq — the "rotate once per (token, head)" step: qr = Pi·q and qs = S·q into an
// interleaved f32 buffer qrqs[b*nh*2*hd + qh*2*hd + {2j, 2j+1}] = (qr_j, qs_j). One block per
// (b, qh), 128 threads. Deterministic per (b, qh) — fixed mapping, ascending-order dots.
// F0: `q_pitch` is the q row pitch (nh*hd packed, or the fused qkv mtot for an offset view).
extern "C" __global__ void rotate_q_tq(float* qrqs, const __nv_bfloat16* q,
        const float* tables, int nh, int hd, int B, int q_pitch) {
    const int idx = blockIdx.x;
    if (idx >= B * nh) return;
    const int b = idx / nh;
    const int qh = idx % nh;
    const float* pi = tables + TQ_TAB_PI;
    const float* s  = tables + TQ_TAB_S;
    const __nv_bfloat16* qrow = q + (long long)b * q_pitch + (long long)qh * hd;
    float* out = qrqs + ((long long)b * nh + qh) * 2 * hd;
    extern __shared__ float sh[];                      // hd floats (hd == 128 == blockDim)
    sh[threadIdx.x] = b2f(qrow[threadIdx.x]);
    __syncthreads();
    const int t = threadIdx.x;
    float acc = 0.0f;
    const float* pirow = pi + t * TQ_HD;
    #pragma unroll
    for (int i = 0; i < TQ_HD; i++) acc += pirow[i] * sh[i];
    out[2 * t] = acc;
    acc = 0.0f;
    const float* srow = s + t * TQ_HD;
    #pragma unroll
    for (int i = 0; i < TQ_HD; i++) acc += srow[i] * sh[i];
    out[2 * t + 1] = acc;
}

// Dequant rows [pos_start, pos_start+n_pos) of the packed cache into scratch in CACHE layout
// ([kvh][pos][hd]): row (h, pos_start+t) lands at (h*out_stride + pos_start + t)*hd. K dequant is
// the reference's decode_k (unbiased reconstruction): k~ = kn·(Pi^T cb2[idx] + qjl·rn·S^T sign);
// V dequant is decode_v: v~ = vn·Pi^T cb3[idx]. One block per (t, kvh), 128 threads, smem 1536 B.
template<bool F32>
__device__ __forceinline__ void tq_dequant_rows(
        const float* tables, const unsigned char* kcache, const unsigned char* vcache,
        long long row, long long out_base, void* out_k, void* out_v, float* sh) {
    const int t = threadIdx.x;
    const float* pit = tables + TQ_TAB_PIT;
    const float* st  = tables + TQ_TAB_ST;
    const float* cb2 = tables + TQ_TAB_CB2;
    const float* cb3 = tables + TQ_TAB_CB3;
    const float qjl_scale = tables[TQ_TAB_SCALE];
    const unsigned char* krow = kcache + row * TQ_ROW_BYTES;
    const unsigned char* vrow = vcache + row * TQ_ROW_BYTES;
    float* kcode = sh;          // [0,128)
    float* ksgn  = sh + 128;    // [128,256)
    float* vcode = sh + 256;    // [256,384)
#if TQ_K_BITS == 3
    {   // 3-bit coord t: bits [3t, 3t+3) — two bytes read separately (byte-unaligned safe)
        const int b0 = (3 * t) >> 3;
        const int off = (3 * t) & 7;
        const unsigned short kw = (unsigned short)krow[b0] | ((unsigned short)krow[b0 + 1] << 8);
        kcode[t] = (float)((kw >> off) & 7);
    }
#else
    kcode[t] = (float)((krow[t >> 2] >> (2 * (t & 3))) & 3);
#endif
    ksgn[t]  = ((krow[TQ_SIGN_OFF + (t >> 3)] >> (t & 7)) & 1) ? 1.0f : -1.0f;
    {   // 3-bit coord t: bits [3t, 3t+3) — two bytes read separately (byte-unaligned safe)
        const int b0 = (3 * t) >> 3;
        const int off = (3 * t) & 7;
        const unsigned short vw = (unsigned short)vrow[b0] | ((unsigned short)vrow[b0 + 1] << 8);
        vcode[t] = (float)((vw >> off) & 7);
    }
    __syncthreads();
    const float rn = tq_half_at(krow, TQ_RN_OFF), kn = tq_half_at(krow, TQ_KN_OFF);
    const float vn = tq_half_at(vrow, 48);
    const float* pitrow = pit + t * TQ_HD;
    float acc = 0.0f;
#if TQ_K_BITS == 3
    #pragma unroll
    for (int j = 0; j < TQ_HD; j++) acc += pitrow[j] * cb3[(int)kcode[j]];
#else
    #pragma unroll
    for (int j = 0; j < TQ_HD; j++) acc += pitrow[j] * cb2[(int)kcode[j]];
#endif
    float acc2 = 0.0f;
    const float* strow = st + t * TQ_HD;
    #pragma unroll
    for (int j = 0; j < TQ_HD; j++) acc2 += strow[j] * ksgn[j];
    const float kval = kn * (acc + qjl_scale * rn * acc2);
    float accv = 0.0f;
    #pragma unroll
    for (int j = 0; j < TQ_HD; j++) accv += pitrow[j] * cb3[(int)vcode[j]];
    const float vval = vn * accv;
    if (F32) {
        ((float*)out_k)[out_base + t] = kval;
        ((float*)out_v)[out_base + t] = vval;
    } else {
        ((__nv_bfloat16*)out_k)[out_base + t] = f2b(kval);
        ((__nv_bfloat16*)out_v)[out_base + t] = f2b(vval);
    }
}

extern "C" __global__ void dequant_kv_tq(__nv_bfloat16* out_k, __nv_bfloat16* out_v,
        const unsigned char* kcache, const unsigned char* vcache, const float* tables,
        int nkv, int stride, int n_pos, int pos_start, int out_stride) {
    const int idx = blockIdx.x;
    if (idx >= n_pos * nkv) return;
    const int t = idx / nkv;
    const int h = idx % nkv;
    extern __shared__ float sh[];
    const long long row = (long long)h * stride + (pos_start + t);
    // OUT row = the launch's t (the q4 convention): the one-shot scratch passes pos_start=0,
    // the incremental mirror shifts the OUT base pointer by the watermark rows instead.
    const long long out_base = ((long long)h * out_stride + t) * TQ_HD;
    tq_dequant_rows<false>(tables, kcache, vcache, row, out_base, out_k, out_v, sh);
}

// dequant_kv_tq_full — debug/oracle twin with f32 output (host-checkable full dequant).
extern "C" __global__ void dequant_kv_tq_full(float* out_k, float* out_v,
        const unsigned char* kcache, const unsigned char* vcache, const float* tables,
        int nkv, int stride, int n_pos, int pos_start, int out_stride) {
    const int idx = blockIdx.x;
    if (idx >= n_pos * nkv) return;
    const int t = idx / nkv;
    const int h = idx % nkv;
    extern __shared__ float sh[];
    const long long row = (long long)h * stride + (pos_start + t);
    const long long out_base = ((long long)h * out_stride + t) * TQ_HD;
    tq_dequant_rows<true>(tables, kcache, vcache, row, out_base, out_k, out_v, sh);
}

// compact_kv_tq — verbatim copy of packed 52-B rows between the cache and a snapshot buffer
// (rollback / prefix-cache paths; position-addressable rows, so a byte copy suffices).
extern "C" __global__ void compact_kv_tq(unsigned char* k_cache, unsigned char* v_cache,
    unsigned char* ks, unsigned char* vs, const int* src_pos, int len, int pos_start,
    int slot, int nkv, int stride, int dir) {
    const int rb = TQ_ROW_BYTES;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = len * nkv * rb;
    if (idx >= total) return;
    const int k = idx / (nkv * rb);
    const int rem = idx % (nkv * rb);
    const int h = rem / rb;
    const int dv = rem % rb;
    const long long cache_pos = (dir == 0) ? (pos_start + src_pos[k]) : (pos_start + k);
    const long long coff = (((long long)slot * nkv + h) * stride + cache_pos) * rb + dv;
    const long long soff = ((long long)k * nkv + h) * rb + dv;
    if (dir == 0) { ks[soff] = k_cache[coff]; vs[soff] = v_cache[coff]; }
    else          { k_cache[coff] = ks[soff]; v_cache[coff] = vs[soff]; }
}

extern "C" __global__ void write_kv_b(__nv_bfloat16* k_cache, __nv_bfloat16* v_cache, const __nv_bfloat16* k_new, const __nv_bfloat16* v_new,
                                       const int* pos, int stride, int nkv, int hd, int B, const int* slot_ids) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * nkv * hd;
    if (idx >= total) return;
    int b = idx / (nkv * hd);
    int rem = idx % (nkv * hd);
    int h = rem / hd;
    int d = rem % hd;
    int slot = slot_ids[b];
    long long coff = ((long long)slot * nkv + h) * stride + pos[b];
    k_cache[coff * hd + d] = k_new[(long long)b * nkv * hd + h * hd + d];
    v_cache[coff * hd + d] = v_new[(long long)b * nkv * hd + h * hd + d];
}


// gqa_attn_flash USED TO LIVE HERE and it is deliberately gone.
//
// It was the "n_splits == 1" fast path for decode attention. Having two kernels that both compute
// decode attention is precisely what broke the lossless-MTP contract: a decode and a verify at the
// same position could pick DIFFERENT ONES (see the note on gqa_attn_splitk), and they did not agree
// to the last bit -- this one divided in fp32, the other round-tripped the numerator through bf16.
// One kernel, one code path. gqa_attn_splitk with ns=1 is this kernel, and costs the same.

// ---- batched sigmoid gate (in place): attn[b,nh*hd] *= sigmoid(gate[b,nh*hd]) ----
extern "C" __global__ void sigmoid_gate_b(__nv_bfloat16* attn, const __nv_bfloat16* gate, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) attn[i] = f2b(b2f(attn[i]) * (1.0f / (1.0f + __expf(-b2f(gate[i])))));
}

// ---- gather RoPE cos/sin from pre-computed tables [max_pos, rdim] ----
extern "C" __global__ void gather_rope_b(float* out_cos, float* out_sin,
    const float* cos_table, const float* sin_table,
    const int* pos, int rdim, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * rdim) return;
    int b = idx / rdim;
    int d = idx % rdim;
    int p = pos[b];
    out_cos[idx] = cos_table[(long long)p * rdim + d];
    out_sin[idx] = sin_table[(long long)p * rdim + d];
}

// ---- gather embedding rows from bf16 table → bf16 output ----
// embed_table: [vocab, h] bf16 (row-major); tokens: [B]; out: [h*B] bf16 (col-major)
extern "C" __global__ void embed_gather_b(__nv_bfloat16* out, const void* embed_table_v,
    const int* tokens, int h, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * h) return;
    int b = idx / h;
    int d = idx % h;
    int tok = tokens[b];
    const __nv_bfloat16* embed_table = (const __nv_bfloat16*)embed_table_v;
    out[(long long)b * h + d] = embed_table[(long long)tok * h + d];
}

// ---- logits penalty: repetition (multiplicative) + presence (flat) + frequency (count) ----
// pen_tokens: [max_pen, B] i32 (-1 = unused); pen_counts: [max_pen, B] i16
// rep_factor > 1.0 penalizes repeated tokens multiplicatively (HF/CTRL formula)
// presence > 0 subtracts flat from any token that appeared
// frequency > 0 subtracts frequency * count from each token
extern "C" __global__ void rep_penalty_b(__nv_bfloat16* logits, const int* pen_tokens,
    const short* pen_counts, int n_pen,
    const float* rep_factors, const float* presences, const float* frequencies,
    int vocab, int B) {
    int b = blockIdx.x;
    if (b >= B) return;
    int tid = threadIdx.x;
    __nv_bfloat16* col = logits + (long long)b * vocab;
    float rep_factor = rep_factors[b];
    float presence = presences[b];
    float frequency = frequencies[b];
    for (int i = tid; i < n_pen; i += blockDim.x) {
        int tok = pen_tokens[(long long)b * n_pen + i];
        if (tok < 0 || tok >= vocab) continue;
        short count = pen_counts[(long long)b * n_pen + i];
        float v = b2f(col[tok]);
        if (rep_factor > 1.0f) v = v > 0.0f ? v / rep_factor : v * rep_factor;
        v -= presence;
        v -= frequency * (float)count;
        col[tok] = f2b(v);
    }
}

// ---- rep_penalty over FP32 logits (hy_v3 enable_lm_head_fp32): same formula, float in/out ----
extern "C" __global__ void rep_penalty_f32_b(float* logits, const int* pen_tokens,
    const short* pen_counts, int n_pen,
    const float* rep_factors, const float* presences, const float* frequencies,
    int vocab, int B) {
    int b = blockIdx.x;
    if (b >= B) return;
    int tid = threadIdx.x;
    float* col = logits + (long long)b * vocab;
    float rep_factor = rep_factors[b];
    float presence = presences[b];
    float frequency = frequencies[b];
    for (int i = tid; i < n_pen; i += blockDim.x) {
        int tok = pen_tokens[(long long)b * n_pen + i];
        if (tok < 0 || tok >= vocab) continue;
        short count = pen_counts[(long long)b * n_pen + i];
        float v = col[tok];
        if (rep_factor > 1.0f) v = v > 0.0f ? v / rep_factor : v * rep_factor;
        v -= presence;
        v -= frequency * (float)count;
        col[tok] = v;
    }
}

// ---- whole-output penalties (vLLM semantics), dense per-slot state (LOOPFIX 2026-11-06) ----
// The list kernels above penalize only what the host packs into a short window (the trunk sampler
// used the last 64 generated tokens), so a repeated paragraph longer than the window was never
// penalized and small models looped. These kernels read a per-slot DENSE state instead:
//   cnt[slot][v]   u32 — occurrences of token v in the lane's OUTPUT so far (committed tokens)
//   pbits[slot][w] u32 — bitset of tokens present in the lane's PROMPT
// Per row r (a decode lane or a verify column): row_slot[r] names the state (-1 = no penalty);
// row_extra[r][0..extra_stride) are tokens the row adds on top of that state (a verify column's
// preceding drafts), -1 padded (extra_stride 0 = none). Semantics = vLLM apply_penalties:
// repetition (HF/CTRL, multiplicative) on every token in the prompt OR the output; presence (flat)
// and frequency (per occurrence) on output tokens only. The formula per token is the list
// kernel's, so a token both kernels penalize gets the identical value.
#define PEN_EXTRA_MAX 16
__device__ __forceinline__ float pen_ld(__nv_bfloat16 x) { return b2f(x); }
__device__ __forceinline__ float pen_ld(float x) { return x; }
__device__ __forceinline__ void pen_st(__nv_bfloat16* p, float x) { *p = f2b(x); }
__device__ __forceinline__ void pen_st(float* p, float x) { *p = x; }
template <typename T>
__device__ __forceinline__ void pen_apply_row(T* col, int b, const int* row_slot, const int* row_extra,
    int extra_stride, const unsigned int* cnt, const unsigned int* pbits,
    const float* rep_factors, const float* presences, const float* frequencies, int vocab, int words) {
    const float rep = rep_factors[b], pres = presences[b], freq = frequencies[b];
    const bool use_rep = rep > 1.0f;
    if (!use_rep && pres == 0.0f && freq == 0.0f) return;          // neutral row: no work
    const int slot = row_slot[b];
    if (slot < 0) return;
    __shared__ int ex[PEN_EXTRA_MAX];
    __shared__ int nex;
    if (threadIdx.x == 0) {
        int n = 0;
        for (int j = 0; j < extra_stride && n < PEN_EXTRA_MAX; j++) {
            const int t = row_extra[(long long)b * extra_stride + j];
            if (t >= 0 && t < vocab) ex[n++] = t;
        }
        nex = n;
    }
    __syncthreads();
    const unsigned int* c = cnt + (long long)slot * vocab;
    const unsigned int* pb = pbits + (long long)slot * words;
    const int ne = nex;
    for (int v = threadIdx.x; v < vocab; v += blockDim.x) {
        unsigned int k = c[v];
        for (int j = 0; j < ne; j++) k += (ex[j] == v) ? 1u : 0u;
        const bool in_prompt = use_rep && ((pb[v >> 5] >> (v & 31)) & 1u);
        if (k == 0u && !in_prompt) continue;
        float x = pen_ld(col[v]);
        if (use_rep) x = x > 0.0f ? x / rep : x * rep;
        if (k > 0u) { x -= pres; x -= freq * (float)k; }
        pen_st(&col[v], x);
    }
}
// grid (B,1,1), block (256,1,1) — the list kernels' launch shape (graph captures unchanged in form).
extern "C" __global__ void pen_apply_b(__nv_bfloat16* logits, const int* row_slot, const int* row_extra,
    int extra_stride, const unsigned int* cnt, const unsigned int* pbits,
    const float* rep_factors, const float* presences, const float* frequencies, int vocab, int words, int B) {
    const int b = blockIdx.x;
    if (b >= B) return;
    pen_apply_row<__nv_bfloat16>(logits + (long long)b * vocab, b, row_slot, row_extra, extra_stride, cnt,
                                 pbits, rep_factors, presences, frequencies, vocab, words);
}
extern "C" __global__ void pen_apply_f32_b(float* logits, const int* row_slot, const int* row_extra,
    int extra_stride, const unsigned int* cnt, const unsigned int* pbits,
    const float* rep_factors, const float* presences, const float* frequencies, int vocab, int words, int B) {
    const int b = blockIdx.x;
    if (b >= B) return;
    pen_apply_row<float>(logits + (long long)b * vocab, b, row_slot, row_extra, extra_stride, cnt,
                         pbits, rep_factors, presences, frequencies, vocab, words);
}
// Count newly committed output tokens: cnt[slots[i]][toks[i]] += 1. grid (ceil(n/256)), block 256.
extern "C" __global__ void pen_count_add_b(unsigned int* cnt, const int* slots, const int* toks, int n, int vocab) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int s = slots[i], t = toks[i];
    if (s < 0 || t < 0 || t >= vocab) return;
    atomicAdd(&cnt[(long long)s * vocab + t], 1u);
}
// A lane takes over a slot: zero its output counts and its prompt bitset. grid-stride, block 256.
extern "C" __global__ void pen_slot_reset_b(unsigned int* cnt, unsigned int* pbits, int slot, int vocab, int words) {
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < vocab; i += stride)
        cnt[(long long)slot * vocab + i] = 0u;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < words; i += stride)
        pbits[(long long)slot * words + i] = 0u;
}
// Mark the prompt's tokens in the slot's bitset (repetition penalty). grid (ceil(n/256)), block 256.
extern "C" __global__ void pen_prompt_set_b(unsigned int* pbits, int slot, const int* toks, int n, int vocab, int words) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int t = toks[i];
    if (t < 0 || t >= vocab) return;
    atomicOr(&pbits[(long long)slot * words + (t >> 5)], 1u << (t & 31));
}

// ---- JSON-schema logit mask (constrained decoding) ----------------------------------
// Bitset mask over the vocabulary: every entry whose bit is CLEAR is written to -inf.
// Entries whose bit is SET are never touched — the mask leaves allowed logits
// BIT-IDENTICAL, so the fixed-order reductions in sample_*/argmax_* (and therefore
// batch-invariance, §2.4) are preserved by construction. No mask ⇒ no launch ⇒ the
// greedy-lossless path is untouched (§2.7).
//   mask_words: [B][ceil(vocab/32)] u32, little-endian bit t = token t allowed.
//   flags:      [B] i32; 0 = lane unconstrained (the kernel returns immediately).
extern "C" __global__ void logit_mask_b(__nv_bfloat16* logits, const unsigned int* mask_words,
    const int* flags, int vocab, int B) {
    int b = blockIdx.x;
    if (b >= B) return;
    if (flags != 0 && flags[b] == 0) return;
    int nw = (vocab + 31) >> 5;
    const unsigned int* w = mask_words + (long long)b * nw;
    __nv_bfloat16* col = logits + (long long)b * vocab;
    const __nv_bfloat16 neg_inf = __ushort_as_bfloat16((unsigned short)0xff80u);
    for (int wi = threadIdx.x + blockIdx.y * blockDim.x; wi < nw; wi += gridDim.y * blockDim.x) {
        unsigned int word = w[wi];
        if (word == 0xFFFFFFFFu) continue;          // whole 32-token bucket allowed: skip it
        int base = wi << 5;
        int end = min(32, vocab - base);
        for (int j = 0; j < end; ++j) {
            if (!((word >> j) & 1u)) col[base + j] = neg_inf;
        }
    }
}

// Same for hy_v3's fp32 logits arm.
extern "C" __global__ void logit_mask_f32_b(float* logits, const unsigned int* mask_words,
    const int* flags, int vocab, int B) {
    int b = blockIdx.x;
    if (b >= B) return;
    if (flags != 0 && flags[b] == 0) return;
    int nw = (vocab + 31) >> 5;
    const unsigned int* w = mask_words + (long long)b * nw;
    float* col = logits + (long long)b * vocab;
    const float neg_inf = __int_as_float(0xff800000);
    for (int wi = threadIdx.x + blockIdx.y * blockDim.x; wi < nw; wi += gridDim.y * blockDim.x) {
        unsigned int word = w[wi];
        if (word == 0xFFFFFFFFu) continue;
        int base = wi << 5;
        int end = min(32, vocab - base);
        for (int j = 0; j < end; ++j) {
            if (!((word >> j) & 1u)) col[base + j] = neg_inf;
        }
    }
}

// ---- batched argmax: logits [vocab, B] bf16 col-major → token_ids [B] ----
extern "C" __global__ void argmax_b(int* token_ids, const __nv_bfloat16* logits, int vocab, int B) {
    int b = blockIdx.x;
    if (b >= B) return;
    extern __shared__ char smem[];
    float* s_vals = (float*)smem;
    int* s_idxs = (int*)(smem + blockDim.x * sizeof(float));
    int tid = threadIdx.x;
    const __nv_bfloat16* col = logits + (long long)b * vocab;
    float my_max = -1e30f;
    int my_idx = 0;
    for (int i = tid; i < vocab; i += blockDim.x) {
        float v = b2f(col[i]);
        if (v > my_max) { my_max = v; my_idx = i; }
    }
    s_vals[tid] = my_max;
    s_idxs[tid] = my_idx;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2) {
            if (s_vals[tid + s2] > s_vals[tid]) {
                s_vals[tid] = s_vals[tid + s2];
                s_idxs[tid] = s_idxs[tid + s2];
            }
        }
        __syncthreads();
    }
    if (tid == 0) token_ids[b] = s_idxs[0];
}

// ---- argmax over FP32 logits (hy_v3 enable_lm_head_fp32): same scan + tree, float reads ----
extern "C" __global__ void argmax_f32_b(int* token_ids, const float* logits, int vocab, int B) {
    int b = blockIdx.x;
    if (b >= B) return;
    extern __shared__ char smem[];
    float* s_vals = (float*)smem;
    int* s_idxs = (int*)(smem + blockDim.x * sizeof(float));
    int tid = threadIdx.x;
    const float* col = logits + (long long)b * vocab;
    float my_max = -1e30f;
    int my_idx = 0;
    for (int i = tid; i < vocab; i += blockDim.x) {
        float v = col[i];
        if (v > my_max) { my_max = v; my_idx = i; }
    }
    s_vals[tid] = my_max;
    s_idxs[tid] = my_idx;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2) {
            if (s_vals[tid + s2] > s_vals[tid]) {
                s_vals[tid] = s_vals[tid + s2];
                s_idxs[tid] = s_idxs[tid + s2];
            }
        }
        __syncthreads();
    }
    if (tid == 0) token_ids[b] = s_idxs[0];
}

// ===================== DEVICE-RESIDENT TOKEN LOOP (EXPERT_DEVICE_ARGMAX_LOOP_RESPONSE) =========
// Four small kernels that move the per-step host bookkeeping into the captured graphs so a clean
// decode step becomes: replay graph -> dtoh [batch] i32 -> bookkeeping -> replay again. All are
// pool-free (persistent DecodeBuffers only), host-sync-free, allocation-free: capture-safe by
// construction. Bit-exactness strategy: the selection kernels (argmax_*/sample_*) and the penalty
// APPLICATION kernels (rep_penalty_*) are NOT touched — only the window/seed BOOKKEEPING moves
// on-device, and it is pure integer set-membership, so the rebuilt arrays are byte-identical to
// today's host rebuild (src/batch.rs batched_decode). Keep the tie-break-relevant kernels' block
// sizes and comparison order untouched — block size is part of the argmax tie-break.

// ---- graph-head id flow: device copy + position advance. grid (1,1,1), block (B,1,1), B<=16.
// tokens[b] <- token_ids[b] (last step's selected id becomes this step's embed input);
// pos[b] += 1 (every Phase-B lane advances exactly one position per step).
extern "C" __global__ void ids_advance_b(int* tokens, const int* token_ids, int* pos, int B) {
    int b = threadIdx.x;
    if (b >= B) return;
    tokens[b] = token_ids[b];
    pos[b] += 1;
}

// ---- penalty ring push (graph epilogue, after the selection kernel).
// grid (B,1,1), block (1,1,1). ring layout: ring[b*n_pen .. b*n_pen+n_pen), most-recent at head-1.
// ring_state[b] packs (head << 8) | len; len saturates at n_pen (64 = MAX_PEN_TOKENS).
extern "C" __global__ void penalty_ring_push_b(int* ring, int* ring_state,
                                               const int* token_ids, int n_pen, int B) {
    int b = blockIdx.x;
    if (b >= B) return;
    int st = ring_state[b];
    int head = st >> 8, len = st & 0xff;
    ring[b * n_pen + head] = token_ids[b];
    ring_state[b] = (((head + 1) % n_pen) << 8) | (len < n_pen ? len + 1 : n_pen);
}

// ---- penalty window rebuild: ring (duplicates allowed, MRU first) -> deduped (token,count) list
// in pen_tokens/pen_counts, BYTE-IDENTICAL to the host rebuild (MRU->LRU; first occurrence keeps
// the slot; count = total occurrences inside the window). grid (B,1,1), block (n_pen,1,1).
// -1 sentinel: rep_penalty_*_b skips it.
extern "C" __global__ void penalty_window_b(const int* ring, const int* ring_state,
                                            int* pen_tokens, short* pen_counts,
                                            int n_pen, int B) {
    int b = blockIdx.x;
    if (b >= B) return;
    int st = ring_state[b];
    int head = st >> 8, len = st & 0xff;
    int i = threadIdx.x;                       // slot i == i-th most-recent entry
    int tok = -1; short cnt = 0;
    if (i < len) {
        int pos_i = (head - 1 - i + n_pen) % n_pen;
        tok = ring[b * n_pen + pos_i];
        bool first = true;
        for (int j = 0; j < i; j++) {          // any more-recent duplicate? then not first
            int pos_j = (head - 1 - j + n_pen) % n_pen;
            if (ring[b * n_pen + pos_j] == tok) { first = false; break; }
        }
        if (first) {
            cnt = 1;
            for (int j = i + 1; j < len; j++) {
                int pos_j = (head - 1 - j + n_pen) % n_pen;
                if (ring[b * n_pen + pos_j] == tok) cnt++;
            }
        } else { tok = -1; cnt = 0; }
    }
    pen_tokens[b * n_pen + i] = tok;
    pen_counts[b * n_pen + i] = cnt;
}

// ---- device seed schedule (sample graph only), bit-identical to the host schedule
// (splitmix64 + rng_u32 with RNG_DOM_SAMPLE at idx 0, src/batch.rs:367-383).
// grid (1,1,1), block (B,1,1). keys advance exactly once per EXECUTED step; capture warmups
// never see real keys (they are uploaded at admission, after capture).
__device__ __forceinline__ unsigned long long splitmix64_dev(unsigned long long z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
#define RNG_DOM_SAMPLE_DEV 0x53414D504C450001ULL   // == RNG_DOM_SAMPLE, src/batch.rs:377
extern "C" __global__ void seed_advance_b(unsigned int* seeds_out,
                                          unsigned long long* keys, int B) {
    int b = threadIdx.x;
    if (b >= B) return;
    unsigned long long k = keys[b];
    // rng_u32(key, DOM, idx=0): the idx term is 0*GOLDEN == 0, so key ^ DOM ^ 0.
    seeds_out[b] = (unsigned int)(splitmix64_dev(k ^ RNG_DOM_SAMPLE_DEV) >> 32);
    keys[b] = splitmix64_dev(k);               // advance exactly once per executed step
}

