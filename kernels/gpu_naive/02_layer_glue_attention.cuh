// gpu_naive/02_layer_glue_attention.cuh: Layer glue: norms, embedding, rope, KV writes, windowed attention with sink, DSA indexer, sparse attention, router and MoE glue.
// Semantics follow the HF reference as encoded by src/naive/cpu.rs (the oracle).
// Section 2 of 3 of gpu_naive.cu: included in order by gpu_naive.cu, one translation unit; text is moved verbatim.
// =============================================================================
// S3b: the layer glue — norms, embedding, rope, KV writes, windowed attention with a sink, the
// DSA indexer (fp8-rounded q/k, scores, exact top-k), gathered sparse attention, router + MoE
// glue. Semantics: the HF reference modeling file, as encoded by src/naive/cpu.rs (the oracle).
// =============================================================================

// ---- embedding gather: out f32 [M][H] = table bf16 [V][H] rows ids[M]. grid (M), block 256.
extern "C" __global__ void nv_embed(float* __restrict__ out, const unsigned short* __restrict__ table,
                                    const int* __restrict__ ids, int H) {
    NV_PDL_ENTRY();
    const int m = blockIdx.x;
    const long long row = (long long)ids[m] * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const unsigned u = (unsigned)table[row + i] << 16;        // bf16 -> f32 is a shift
        out[(long long)m * H + i] = __uint_as_float(u);
    }
}

// ---- LlamaRMSNorm: h f16 [M][H] = x * rsqrt(mean(x^2) + eps) * w  (w f16 [H], the pack's bf16
// norms are converted to f16 at load). grid (M), block 1024, H % 1024 == 0 not required.
extern "C" __global__ void nv_rmsnorm(__half* __restrict__ h, const float* __restrict__ x,
                                      const __half* __restrict__ w, int H, float eps) {
    NV_PDL_ENTRY();
    __shared__ float red[1024];
    const int m = blockIdx.x, tid = threadIdx.x;
    const float* xr = x + (long long)m * H;
    float ss = 0.f;
    for (int i = tid; i < H; i += blockDim.x) { const float v = xr[i]; ss += v * v; }
    red[tid] = ss; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
    const float inv = rsqrtf(red[0] / (float)H + eps);
    for (int i = tid; i < H; i += blockDim.x) h[(long long)m * H + i] = nv_f2h(xr[i] * inv * nv_h2f(w[i]));
}

// ---- residual add: x f32 [n] += y f16 [n]
extern "C" __global__ void nv_add_f16(float* __restrict__ x, const __half* __restrict__ y, long long n) {
    NV_PDL_ENTRY();
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += nv_h2f(y[i]);
}

// ---- GPT-NeoX partial rotary on the first `rd` dims of each head (rotate_half), in place on f16
// rows. One warp per (row, head); lane owns dims lane and lane+rd/2 (rd = 64 -> lanes 0..31).
// cos/sin from double-precision sincos of pos * theta^(-2i/rd) (the f32 reference formula loses
// precision past ~10^5 positions; both sides of every gate here use this f64 form).
__device__ __forceinline__ void nv_rope_head(__half* v, int pos, float theta, int rd, int lane) {
    const int half = rd >> 1;
    if (lane >= half) return;
    const double inv = pow((double)theta, -(2.0 * lane) / (double)rd);
    double sn, cs;
    sincos((double)pos * inv, &sn, &cs);
    const float c = (float)cs, s = (float)sn;
    const float x1 = nv_h2f(v[lane]), x2 = nv_h2f(v[lane + half]);
    v[lane] = nv_f2h(x1 * c - x2 * s);
    v[lane + half] = nv_f2h(x2 * c + x1 * s);
}
// grid (M * heads), block 32; rows of `heads` heads of `hd` dims at pitch heads*hd.
extern "C" __global__ void nv_rope_rows(__half* __restrict__ x, const int* __restrict__ pos, int heads, int hd,
                                        int rd, float theta) {
    NV_PDL_ENTRY();
    const int m = blockIdx.x / heads, h = blockIdx.x % heads;
    nv_rope_head(x + ((long long)m * heads + h) * hd, pos[m], theta, rd, threadIdx.x & 31);
}

// ---- KV cache write. k [M][kv*192] f16 (roped), v [M][kv*192] f16 (per-head padded: the first vd of
// each 192 slice is live). Cache layout: kc [kv][S][hd], vc [kv][S][vd], slot = ring ? pos & (S-1) : pos.
// grid (M * kv), block 192.
extern "C" __global__ void nv_kv_write(__half* __restrict__ kc, __half* __restrict__ vc,
                                       const __half* __restrict__ k, const __half* __restrict__ v,
                                       const int* __restrict__ pos, const int* __restrict__ slot_ids,
                                       int kv, int hd, int vd, int S, int ring) {
    NV_PDL_ENTRY();
    const int m = blockIdx.x / kv, g = blockIdx.x % kv, d = threadIdx.x;
    const int p = pos[m];
    const int slot = ring ? (p & (S - 1)) : p;
    // slot_ids[m] = the request row m writes its cache in: kc [n_slots][kv][S][hd] (vc [n_slots][kv][S][vd])
    const long long skc = (long long)slot_ids[m] * ((long long)kv * S * hd);
    const long long svc = (long long)slot_ids[m] * ((long long)kv * S * vd);
    kc[skc + ((long long)g * S + slot) * hd + d] = k[((long long)m * kv + g) * hd + d];
    if (d < vd) vc[svc + ((long long)g * S + slot) * vd + d] = v[((long long)m * kv + g) * hd + d];
}

// ---- S6 F4: rope(q) + rope(k) + K/V cache write in one launch. grid (M * (nh + kv)), block 32: warps
// < nh rope a q head; the rest rope a k head and copy its k (roped) and v rows into the cache slot
// (the SAME per-element ops as nv_rope_rows + nv_kv_write). nhkv = nh | kv << 16, hdvd = hd | vd << 16.
extern "C" __global__ void nv_rope_kv(__half* __restrict__ q, __half* __restrict__ k, const __half* __restrict__ v,
                                      __half* __restrict__ kc, __half* __restrict__ vc, const int* __restrict__ pos,
                                      const int* __restrict__ slot_ids,
                                      int nhkv, int hdvd, int rd, float theta, int S) {
    NV_PDL_ENTRY();
    const int nh = nhkv & 0xFFFF, kv = nhkv >> 16, hd = hdvd & 0xFFFF, vd = (hdvd >> 16) & 0x3FFF;
    const int ring = (int)((unsigned)hdvd >> 30);   // packed by the host (cudarc's 12-arg launch cap)
    const int per = nh + kv;
    const int m = blockIdx.x / per, h = blockIdx.x % per;
    const int lane = threadIdx.x & 31;
    const int p = pos[m];
    if (h < nh) { nv_rope_head(q + ((long long)m * nh + h) * hd, p, theta, rd, lane); return; }
    const int g = h - nh;
    __half* kr = k + ((long long)m * kv + g) * hd;
    nv_rope_head(kr, p, theta, rd, lane);
    __syncwarp();
    const int slot = ring ? (p & (S - 1)) : p;
    // slot_ids[m] = the request row m writes its cache in: kc [n_slots][kv][S][hd] (vc [n_slots][kv][S][vd])
    __half* kd = kc + (long long)slot_ids[m] * ((long long)kv * S * hd) + ((long long)g * S + slot) * hd;
    __half* vdst = vc + (long long)slot_ids[m] * ((long long)kv * S * vd) + ((long long)g * S + slot) * vd;
    const __half* vr = v + ((long long)m * kv + g) * hd;
    for (int d = lane; d < hd; d += 32) kd[d] = kr[d];
    for (int d = lane; d < vd; d += 32) vdst[d] = vr[d];
}

// ---- sliding-window attention with a learned sink, decode / verify rows (M <= 16).
// q f16 [M][nh*hd] (roped), caches as nv_kv_write (ring of S slots), sink f16 [nh] (has_sink = 0: unused).
// Keys t in [max(0, p-W+1), p]; logits = q.k * hd^-0.5 ; softmax over keys U {sink}; out f16
// [M][nh*vd] = sum p_t V_t (the reference scales V by attention_value_scale BEFORE attention; the
// pack folds that into o_proj, so nothing here). grid (nh, M), block 256 (W <= 256).
extern "C" __global__ void nv_attn_swa(__half* __restrict__ out, const __half* __restrict__ q,
                                       const __half* __restrict__ kc, const __half* __restrict__ vc,
                                       const __half* __restrict__ sink, const int* __restrict__ pos,
                                       const int* __restrict__ slot_ids,
                                       int nhkv, int hdvd, int S, int w_sink) {
    NV_PDL_ENTRY();
    const int nh = nhkv & 0xFFFF, kv = nhkv >> 16, hd = hdvd & 0xFFFF, vd = hdvd >> 16;
    const int W = w_sink & 0xFFFF, has_sink = w_sink >> 16;
    __shared__ float qs[256];        // hd <= 256
    __shared__ float lg[256];        // one logit per key slot in the window
    __shared__ float red[256];
    const int h = blockIdx.x, m = blockIdx.y, tid = threadIdx.x;
    const int g = h / (nh / kv);
    // slot_ids[m] = the request whose cache row m attends over: kc [n_slots][kv][S][hd] (vc [n_slots][kv][S][vd])
    const long long skc = (long long)slot_ids[m] * ((long long)kv * S * hd);
    const long long svc = (long long)slot_ids[m] * ((long long)kv * S * vd);
    const int p = pos[m];
    const int t0 = p - W + 1 > 0 ? p - W + 1 : 0;
    const int nk = p - t0 + 1;                                   // <= W <= 256
    for (int d = tid; d < hd; d += blockDim.x) qs[d] = nv_h2f(q[((long long)m * nh + h) * hd + d]);
    __syncthreads();
    const float scale = rsqrtf((float)hd);
    float lv = -INFINITY;
    if (tid < nk) {
        const int t = t0 + tid;
        const __half* kr = kc + skc + ((long long)g * S + (t & (S - 1))) * hd;
        float dot = 0.f;
        // 8 halves per load (hd % 8 == 0; rows are 16-B aligned), the SAME sequential fp32 accumulation
        // over d as the scalar loop it replaces: bitwise identical, 8x fewer load instructions.
        for (int d0 = 0; d0 < hd; d0 += 8) {
            const uint4 kv4 = *(const uint4*)(kr + d0);
            const __half* kh = (const __half*)&kv4;
            #pragma unroll
            for (int j = 0; j < 8; j++) dot += qs[d0 + j] * nv_h2f(kh[j]);
        }
        lv = dot * scale;
    }
    lg[tid] = lv;
    // max over keys and the sink
    float mx = lv;
    if (has_sink && tid == 0) mx = fmaxf(mx, nv_h2f(sink[h]));
    red[tid] = mx; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]); __syncthreads(); }
    mx = red[0];
    __syncthreads();
    float e = (tid < nk) ? __expf(lv - mx) : 0.f;
    lg[tid] = e;
    red[tid] = e; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
    float den = red[0];
    if (has_sink) den += __expf(nv_h2f(sink[h]) - mx);
    __syncthreads();
    // weighted V: thread d < vd sums over keys ascending
    if (tid < vd) {
        float acc = 0.f;
        for (int i = 0; i < nk; i++) {
            const int t = t0 + i;
            acc += lg[i] * nv_h2f(vc[svc + ((long long)g * S + (t & (S - 1))) * vd + tid]);
        }
        out[((long long)m * nh + h) * vd + tid] = nv_f2h(acc / den);
    }
}

// ---- PREFILL variant of nv_attn_swa (m > 16): ONE BLOCK PER (kv group, row); the gqa (<= 8) query heads of the
// group share the K / V rows (each key's K row is read once for all heads; the V column once per key). Per head
// the arithmetic, summation order (sequential dot over d, the 256-wide tree sums, ascending-key weighted V),
// sink and window semantics are exactly nv_attn_swa's: the output is bitwise identical. grid (kv, M), block 256.
extern "C" __global__ void nv_attn_swa_grp(__half* __restrict__ out, const __half* __restrict__ q,
                                           const __half* __restrict__ kc, const __half* __restrict__ vc,
                                           const __half* __restrict__ sink, const int* __restrict__ pos,
                                           const int* __restrict__ slot_ids,
                                           int nhkv, int hdvd, int S, int w_sink) {
    NV_PDL_ENTRY();
    const int nh = nhkv & 0xFFFF, kv = nhkv >> 16, hd = hdvd & 0xFFFF, vd = hdvd >> 16;
    const int W = w_sink & 0xFFFF, has_sink = w_sink >> 16;
    __shared__ float qs[8][256];
    __shared__ float lg[8][256];
    __shared__ float red[8][256];
    const int g = blockIdx.x, m = blockIdx.y, tid = threadIdx.x;
    const int gqa = nh / kv;                                      // <= 8 (host guarantees)
    // slot_ids[m] = the request whose cache row m attends over: kc [n_slots][kv][S][hd] (vc [n_slots][kv][S][vd])
    const long long skc = (long long)slot_ids[m] * ((long long)kv * S * hd);
    const long long svc = (long long)slot_ids[m] * ((long long)kv * S * vd);
    const int p = pos[m];
    const int t0 = p - W + 1 > 0 ? p - W + 1 : 0;
    const int nk = p - t0 + 1;
    for (int hh = 0; hh < gqa; hh++)
        for (int d = tid; d < hd; d += blockDim.x) qs[hh][d] = nv_h2f(q[((long long)m * nh + g * gqa + hh) * hd + d]);
    __syncthreads();
    const float scale = rsqrtf((float)hd);
    float lv[8];
    #pragma unroll
    for (int hh = 0; hh < 8; hh++) lv[hh] = -INFINITY;
    if (tid < nk) {
        const int t = t0 + tid;
        const __half* kr = kc + skc + ((long long)g * S + (t & (S - 1))) * hd;
        float dot[8];
        #pragma unroll
        for (int hh = 0; hh < 8; hh++) dot[hh] = 0.f;
        for (int d = 0; d < hd; d++) {
            const float kd = nv_h2f(kr[d]);
            #pragma unroll
            for (int hh = 0; hh < 8; hh++) if (hh < gqa) dot[hh] += qs[hh][d] * kd;
        }
        #pragma unroll
        for (int hh = 0; hh < 8; hh++) lv[hh] = dot[hh] * scale;
    }
    #pragma unroll
    for (int hh = 0; hh < 8; hh++) if (hh < gqa) {
        lg[hh][tid] = lv[hh];
        float mx = lv[hh];
        if (has_sink && tid == 0) mx = fmaxf(mx, nv_h2f(sink[g * gqa + hh]));
        red[hh][tid] = mx;
    }
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            #pragma unroll
            for (int hh = 0; hh < 8; hh++) if (hh < gqa) red[hh][tid] = fmaxf(red[hh][tid], red[hh][tid + s]);
        }
        __syncthreads();
    }
    float mxh[8];
    #pragma unroll
    for (int hh = 0; hh < 8; hh++) mxh[hh] = (hh < gqa) ? red[hh][0] : 0.f;
    __syncthreads();
    #pragma unroll
    for (int hh = 0; hh < 8; hh++) if (hh < gqa) {
        const float e = (tid < nk) ? __expf(lv[hh] - mxh[hh]) : 0.f;
        lg[hh][tid] = e;
        red[hh][tid] = e;
    }
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            #pragma unroll
            for (int hh = 0; hh < 8; hh++) if (hh < gqa) red[hh][tid] += red[hh][tid + s];
        }
        __syncthreads();
    }
    float den[8];
    #pragma unroll
    for (int hh = 0; hh < 8; hh++) {
        den[hh] = (hh < gqa) ? red[hh][0] : 1.f;
        if (hh < gqa && has_sink) den[hh] += __expf(nv_h2f(sink[g * gqa + hh]) - mxh[hh]);
    }
    __syncthreads();
    if (tid < vd) {
        float acc[8];
        #pragma unroll
        for (int hh = 0; hh < 8; hh++) acc[hh] = 0.f;
        for (int i = 0; i < nk; i++) {
            const int t = t0 + i;
            const float v = nv_h2f(vc[svc + ((long long)g * S + (t & (S - 1))) * vd + tid]);
            #pragma unroll
            for (int hh = 0; hh < 8; hh++) if (hh < gqa) acc[hh] += lg[hh][i] * v;
        }
        #pragma unroll
        for (int hh = 0; hh < 8; hh++) if (hh < gqa) out[((long long)m * nh + g * gqa + hh) * vd + tid] = nv_f2h(acc[hh] / den[hh]);
    }
}

// ---- DSA indexer. The reference: q = rope(wq(x)) [16][128], k = rope(LayerNorm(wk(x))) [128], both
// rounded to fp8 e4m3 per row (scale = amax.clamp_min(1e-4)/448, RNE, saturate); scores[t] =
// sum_h relu(q_h . k_t) * (weights_proj(x)[h] * 16^-0.5); causal; top-k by stable descending sort.
// Keys are cached as e4m3 bytes + one f32 scale per position: kq [S][ihd] u8, ks [S] f32.

__device__ __forceinline__ float nv_e4m3_round(float v, float scale) {
    // (v / scale).clamp(-448, 448) -> e4m3 (RN, saturate) -> f32 * scale
    const float x = fminf(fmaxf(v / scale, -448.f), 448.f);
    const __nv_fp8_storage_t b = __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
    return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3)) * scale;
}
__device__ __forceinline__ float nv_e4m3_val(unsigned char b) {
    return __half2float(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3));
}

// q side: iq f32 [M][ih][ihd] from the wq chain output f16 [M][ih*ihd]: rope (first rd dims), then
// the per-(row, head) fp8 round-trip. grid (M * ih), block = ihd (128).
extern "C" __global__ void nv_idx_q(float* __restrict__ iq, const __half* __restrict__ raw, const int* __restrict__ pos,
                                    int ih, int ihd, int rd, float theta, int fp8) {
    NV_PDL_ENTRY();
    __shared__ float v[128];
    __shared__ float red[128];
    const int m = blockIdx.x / ih, h = blockIdx.x % ih, d = threadIdx.x;
    const long long base = ((long long)m * ih + h) * ihd;
    float x = nv_h2f(raw[base + d]);
    // rope in f32 on the first rd dims (rotate_half)
    const int half = rd >> 1;
    v[d] = x; __syncthreads();
    if (d < rd) {
        const int i = d < half ? d : d - half;
        const double inv = pow((double)theta, -(2.0 * i) / (double)rd);
        double sn, cs; sincos((double)pos[m] * inv, &sn, &cs);
        const float c = (float)cs, s = (float)sn;
        x = d < half ? (v[d] * c - v[d + half] * s) : (v[d] * c + v[d - half] * s);
    }
    if (fp8) {
        red[d] = fabsf(x); __syncthreads();
        for (int s = ihd >> 1; s > 0; s >>= 1) { if (d < s) red[d] = fmaxf(red[d], red[d + s]); __syncthreads(); }
        const float scale = fmaxf(red[0], 1e-4f) / 448.f;
        x = nv_e4m3_round(x, scale);
    }
    iq[base + d] = x;
}

// k side: raw f16 [M][ihd] from the wk chain; LayerNorm (biased var, eps 1e-5, weight + bias f16),
// rope, fp8 round -> kq[pos][ihd] bytes + ks[pos]. grid (M), block = ihd (128).
extern "C" __global__ void nv_idx_k(unsigned char* __restrict__ kq, float* __restrict__ ks,
                                    const __half* __restrict__ raw, const __half* __restrict__ nw,
                                    const __half* __restrict__ nb, const int* __restrict__ pos,
                                    const int* __restrict__ slot_ids,
                                    int ihd, int rd, float theta, float eps, int max_pos) {
    NV_PDL_ENTRY();
    __shared__ float v[128];
    __shared__ float red[128];
    const int m = blockIdx.x, d = threadIdx.x;
    float x = nv_h2f(raw[(long long)m * ihd + d]);
    red[d] = x; __syncthreads();
    for (int s = ihd >> 1; s > 0; s >>= 1) { if (d < s) red[d] += red[d + s]; __syncthreads(); }
    const float mean = red[0] / (float)ihd;
    __syncthreads();
    red[d] = (x - mean) * (x - mean); __syncthreads();
    for (int s = ihd >> 1; s > 0; s >>= 1) { if (d < s) red[d] += red[d + s]; __syncthreads(); }
    const float inv = rsqrtf(red[0] / (float)ihd + eps);
    x = (x - mean) * inv * nv_h2f(nw[d]) + nv_h2f(nb[d]);
    __syncthreads();
    v[d] = x; __syncthreads();
    const int half = rd >> 1;
    if (d < rd) {
        const int i = d < half ? d : d - half;
        const double invf = pow((double)theta, -(2.0 * i) / (double)rd);
        double sn, cs; sincos((double)pos[m] * invf, &sn, &cs);
        const float c = (float)cs, s = (float)sn;
        x = d < half ? (v[d] * c - v[d + half] * s) : (v[d] * c + v[d - half] * s);
    }
    __syncthreads();
    red[d] = fabsf(x); __syncthreads();
    for (int s = ihd >> 1; s > 0; s >>= 1) { if (d < s) red[d] = fmaxf(red[d], red[d + s]); __syncthreads(); }
    const float scale = fmaxf(red[0], 1e-4f) / 448.f;
    const float xq = fminf(fmaxf(x / scale, -448.f), 448.f);
    // slot_ids[m] = the request whose index cache row m writes: kq [n_slots][max_pos][ihd], ks [n_slots][max_pos]
    kq[(long long)slot_ids[m] * max_pos * ihd + (long long)pos[m] * ihd + d] = (unsigned char)__nv_cvt_float_to_fp8(xq, __NV_SATFINITE, __NV_E4M3);
    if (d == 0) ks[(long long)slot_ids[m] * max_pos + pos[m]] = scale;
}

// scores[m][t] for t <= pos[m]: sum_h relu(iq[m][h] . k_t) * wgt[m][h]; wgt = weights_proj row (f16,
// padded to 128; the first ih live) * ih^-0.5. One warp per (row, key): lane owns 4 dims (ihd 128).
// grid (ceil((maxpos+1)/8), M), block 256 (8 warps = 8 keys). Entries past pos[m] are not written.
extern "C" __global__ void nv_idx_score(float* __restrict__ scores, const float* __restrict__ iq,
                                        const __half* __restrict__ wgt, const unsigned char* __restrict__ kq,
                                        const float* __restrict__ ks, const int* __restrict__ pos,
                                        const int* __restrict__ slot_ids,
                                        int ih, int ihd, int wpitch, long long stride, int max_pos) {
    NV_PDL_ENTRY();
    // S6: a FIXED grid (graph-capturable) that strides over the keys t <= pos[m]
    const int m = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int dpl = ihd >> 5;                                    // 4
    const float wscale = rsqrtf((float)ih);
    const int n = pos[m] + 1;
    // slot_ids[m] = the request whose index cache row m reads: kq [n_slots][max_pos][ihd], ks [n_slots][max_pos]
    const long long skq = (long long)slot_ids[m] * max_pos * ihd;
    const long long sks = (long long)slot_ids[m] * max_pos;
    for (int t = blockIdx.x * 8 + warp; t < n; t += gridDim.x * 8) {
        float kvv[4];
        const unsigned char* kr = kq + skq + (long long)t * ihd + lane * dpl;
        const float sc = ks[sks + t];
        #pragma unroll
        for (int i = 0; i < 4; i++) kvv[i] = (i < dpl) ? nv_e4m3_val(kr[i]) * sc : 0.f;
        float s = 0.f;
        for (int h = 0; h < ih; h++) {
            const float* qr = iq + ((long long)m * ih + h) * ihd + lane * dpl;
            float dot = 0.f;
            #pragma unroll
            for (int i = 0; i < 4; i++) if (i < dpl) dot += qr[i] * kvv[i];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
            s += fmaxf(dot, 0.f) * (nv_h2f(wgt[(long long)m * wpitch + h]) * wscale);
        }
        if (lane == 0) scores[(long long)m * stride + t] = s;
    }
}

// PREFILL variant of nv_idx_score (M > 16 rows, ih == 16, ihd == 128): tensor cores. The 16 index heads of one
// row are exactly one m16 tile: C[h][key] = sum_d iq[m][h][d] * K[key][d] (mma m16n8k16 f16, fp32 accumulate).
// K is staged as raw e4m3 values (exact in f16) per tile of NVW_KT keys; the per-key scale is applied to the
// fp32 accumulator (ks > 0, so relu commutes). Then relu * (w[m][h] * ih^-0.5), summed over h in fp32.
// grid (ceil(nkeys / NVW_KT), ceil(M / NVW_RB)), block 256 (8 warps, each walks rows r = warp, warp+8, ...).
// Entries past pos[m] are not written.
#define NVW_KT 128
#define NVW_RB 32
#define NVW_LD 136
extern "C" __global__ void nv_idx_score_wide(float* __restrict__ scores, const float* __restrict__ iq,
                                             const __half* __restrict__ wgt, const unsigned char* __restrict__ kq,
                                             const float* __restrict__ ks, const int* __restrict__ pos,
                                             const int* __restrict__ slot_ids,
                                             int M, int nkeys, int wpitch, long long stride) {
    // 2026-10-20: the ONE naive entry kernel without the PDL entry. Under --naive-pdl=1 (default since the same
    // night) it could start before nv_idx_q / nv_idx_k finished writing iq / kq and score stale keys: the 7,475-token
    // prompt's sampled stream diverged at ~token 15 while every short-prompt stream stayed identical.
    NV_PDL_ENTRY();
    __shared__ __align__(16) __half Ks[NVW_KT * NVW_LD];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
    const int t0 = blockIdx.x * NVW_KT;
    const int rbase = blockIdx.y * NVW_RB;
    int maxp = -1;
    for (int r = 0; r < NVW_RB && rbase + r < M; r++) maxp = max(maxp, pos[rbase + r]);
    if (t0 > maxp) return;
    const float wscale = 0.25f;                              // 16^-0.5
    // The key tile is staged by ALL threads of the block, so it comes from ONE request's plane: the block's rows are
    // one prefill chunk's (this scorer runs for M > 16 only, i.e. a prefill chunk, i.e. one request). 2026-10-21: each
    // thread used its OWN warp's row slot here, and 0 for a row with no key in this tile (a row before the tile, or past
    // the chunk), so a request in a KV slot other than 0 (--naive-lanes > 1) scored tiles mixed with slot 0's keys;
    // past 2,048 keys (the top-k) its DSA selection, and its text, drifted from the same request in slot 0.
    const long long skq_t = (long long)slot_ids[rbase] * nkeys * 128;
    for (int r = warp; r < NVW_RB; r += 8) {
        const int m = rbase + r;
        const bool live = (m < M) && (t0 <= pos[m]);
        // slot_ids[m] = the request whose index cache row m reads: kq [n_slots][nkeys][128], ks [n_slots][nkeys]
        const long long sks = live ? (long long)slot_ids[m] * nkeys : 0;
        __syncthreads();
        for (int i = tid; i < NVW_KT * 8; i += 256) {      // 8 chunks of 16 bytes per key
            const int key = i >> 3, c = i & 7, t = t0 + key;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (t < nkeys) v = *(const uint4*)(kq + skq_t + (long long)t * 128 + c * 16);
            const unsigned char* b = (const unsigned char*)&v;
            __half2 o[8];
            #pragma unroll
            for (int j = 0; j < 8; j++) o[j] = __floats2half2_rn(nv_e4m3_val(b[2 * j]), nv_e4m3_val(b[2 * j + 1]));
            *(uint4*)&Ks[key * NVW_LD + c * 16] = *(uint4*)&o[0];
            *(uint4*)&Ks[key * NVW_LD + c * 16 + 8] = *(uint4*)&o[4];
        }
        __syncthreads();
        if (!live) continue;
        const int pm = pos[m];
        uint32_t a[8][4];
        const float* q0 = iq + ((long long)m * 16 + gid) * 128;
        const float* q1 = q0 + 8 * 128;
        #pragma unroll
        for (int kc = 0; kc < 8; kc++) {
            const int d = kc * 16 + 2 * tig;
            const float2 x0 = *(const float2*)(q0 + d), x1 = *(const float2*)(q1 + d);
            const float2 x2 = *(const float2*)(q0 + d + 8), x3 = *(const float2*)(q1 + d + 8);
            __half2 h0 = __floats2half2_rn(x0.x, x0.y), h1 = __floats2half2_rn(x1.x, x1.y);
            __half2 h2 = __floats2half2_rn(x2.x, x2.y), h3 = __floats2half2_rn(x3.x, x3.y);
            a[kc][0] = *(uint32_t*)&h0; a[kc][1] = *(uint32_t*)&h1; a[kc][2] = *(uint32_t*)&h2; a[kc][3] = *(uint32_t*)&h3;
        }
        const float w0 = nv_h2f(wgt[(long long)m * wpitch + gid]) * wscale;
        const float w1 = nv_h2f(wgt[(long long)m * wpitch + gid + 8]) * wscale;
        for (int nb = 0; nb < NVW_KT / 8; nb++) {
            const int tb = t0 + nb * 8;
            if (tb > pm) break;
            float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
            const __half* kr = &Ks[(nb * 8 + gid) * NVW_LD + 2 * tig];
            #pragma unroll
            for (int kc = 0; kc < 8; kc++) {
                const uint32_t b0 = *(const uint32_t*)(kr + kc * 16), b1 = *(const uint32_t*)(kr + kc * 16 + 8);
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                             : "r"(a[kc][0]), "r"(a[kc][1]), "r"(a[kc][2]), "r"(a[kc][3]), "r"(b0), "r"(b1));
            }
            const int ta = tb + 2 * tig;
            const float s0 = (ta < nkeys) ? ks[sks + ta] : 0.f, s1 = (ta + 1 < nkeys) ? ks[sks + ta + 1] : 0.f;
            float v0 = fmaxf(c0 * s0, 0.f) * w0 + fmaxf(c2 * s0, 0.f) * w1;
            float v1 = fmaxf(c1 * s1, 0.f) * w0 + fmaxf(c3 * s1, 0.f) * w1;
            #pragma unroll
            for (int o = 4; o < 32; o <<= 1) { v0 += __shfl_xor_sync(0xffffffffu, v0, o); v1 += __shfl_xor_sync(0xffffffffu, v1, o); }
            if (gid == 0) {
                if (ta <= pm) scores[(long long)m * stride + ta] = v0;
                if (ta + 1 <= pm) scores[(long long)m * stride + ta + 1] = v1;
            }
        }
    }
}

// monotonic unsigned key of a float (negative scores are legal here: weights_proj can be negative)
__device__ __forceinline__ unsigned nv_fkey(float f) {
    const unsigned b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// block-wide exclusive scan of two flags (blockDim 1024 = 32 warps); returns prefixes and totals
__device__ __forceinline__ void nv_exscan2(int a, int c, int& pa, int& pc, int& ta, int& tc, int* wa, int* wc) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int ia = a, ic = c;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const int xa = __shfl_up_sync(0xffffffffu, ia, off), xc = __shfl_up_sync(0xffffffffu, ic, off);
        if (lane >= off) { ia += xa; ic += xc; }
    }
    if (lane == 31) { wa[warp] = ia; wc[warp] = ic; }
    __syncthreads();
    int ba = 0, bc = 0, sa = 0, sc = 0;
    const int nw = blockDim.x >> 5;
    for (int w = 0; w < nw; w++) { if (w < warp) { ba += wa[w]; bc += wc[w]; } sa += wa[w]; sc += wc[w]; }
    pa = ba + ia - a; pc = bc + ic - c; ta = sa; tc = sc;
    __syncthreads();
}

// top-K over scores[m][0..pos[m]] (n = pos+1 keys): the K largest under (value desc, index asc) —
// exactly torch's stable descending argsort. Radix select (4 x 8-bit passes, MSB first) over the
// monotonic keys finds the K-th key T and how many == T to keep (kp); the compaction keeps every
// key > T and the first kp keys == T in index order, emitted ASCENDING into sel[m][0..nsel), nsel[m]
// = min(K, n). One 1024-thread block per row (the engine's qsa_topk_b, ratio 1, signed keys).
extern "C" __global__ void __launch_bounds__(1024)
nv_topk(int* __restrict__ sel, int* __restrict__ nsel, const float* __restrict__ scores,
        const int* __restrict__ pos, int K, long long stride, const int* __restrict__ acc,
        const int* __restrict__ force) {
    NV_PDL_ENTRY();
    // sampler fallback gate (the indexer passes nullptr): skipped once a Gumbel try was accepted,
    // UNLESS `force[0]` (the exact speculative path of work item A builds p from the gathered
    // candidate list on EVERY row, on EVERY rank: a skipped gather leaves sel/nsel stale or never
    // written, and the accept kernel then reads an empty list -- the 2026-10-19 `anchor 4286578688`
    // served fatal: keep == 0 -> pick == -1 -> rtok = sp_i[-1] = the -INFINITY pad of sp_v)
    if (acc != nullptr && acc[0] && !(force != nullptr && force[0])) return;
    __shared__ int hist[256];
    __shared__ int s_bin, s_kp;
    __shared__ int wa[32], wc[32];
    const int m = blockIdx.x, tid = threadIdx.x;
    const int n = pos[m] + 1;
    const float* sc = scores + (long long)m * stride;
    int* srow = sel + (long long)m * K;
    unsigned T = 0u; int kp = 0x7fffffff;
    if (n > K) {
        unsigned prefix = 0u, mask = 0u;
        kp = K;
        for (int pass = 3; pass >= 0; pass--) {
            const int shift = pass * 8;
            for (int i = tid; i < 256; i += blockDim.x) hist[i] = 0;
            __syncthreads();
            for (int j = tid; j < n; j += blockDim.x) {
                const unsigned key = nv_fkey(sc[j]);
                if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1);
            }
            __syncthreads();
            if (tid == 0) {
                int cum = 0, bin = 0, rem = kp;
                for (int i = 255; i >= 0; i--) {
                    const int c = hist[i];
                    if (cum + c >= kp) { bin = i; rem = kp - cum; break; }
                    cum += c;
                }
                s_bin = bin; s_kp = rem;
            }
            __syncthreads();
            prefix |= ((unsigned)s_bin) << shift;
            mask |= 0xFFu << shift;
            kp = s_kp;
            __syncthreads();
        }
        T = prefix;
    }
    int run_sel = 0, run_eq = 0;
    for (int base = 0; base < n; base += blockDim.x) {
        const int j = base + tid;
        const bool valid = j < n;
        const unsigned key = valid ? nv_fkey(sc[j]) : 0u;
        const bool gt = valid && (n <= K || key > T);
        const bool eq = valid && n > K && key == T;
        int p_eq, p_d, t_eq, t_d;
        nv_exscan2(eq ? 1 : 0, 0, p_eq, p_d, t_eq, t_d, wa, wc);
        const bool selected = gt || (eq && (run_eq + p_eq) < kp);
        int p_s, t_s;
        nv_exscan2(selected ? 1 : 0, 0, p_s, p_d, t_s, t_d, wa, wc);
        if (selected) srow[run_sel + p_s] = j;
        run_sel += t_s; run_eq += t_eq;
    }
    if (tid == 0) nsel[m] = run_sel;
}

// ---- gathered sparse attention (DSA layers): q f16 [M][nh*hd] roped, full caches kc [kv][S][hd],
// vc [kv][S][vd], key list sel[m][0..nsel[m]) (ascending). Split-K online softmax, no sink: grid
// (M * kv * nsplits), block 256 = 8 warps; warp w owns heads kvh*gqa + w + 8j (j < gqa/8); lane
// owns hd/32 = 6 K dims and vd/32 = 4 V dims. Partials pm/pl [M*nh*nsplits], pacc [..][vd]; empty
// split: pm = -inf, pl = 0. Then nv_attn_combine merges the splits (ascending) -> out f16 [M][nh*vd].
extern "C" __global__ void nv_attn_dsa_split(float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc,
                                             const __half* __restrict__ q, const __half* __restrict__ kc,
                                             const __half* __restrict__ vc, const int* __restrict__ sel,
                                             const int* __restrict__ nsel, const int* __restrict__ slot_ids,
                                             int nhkv, int hdvd, long long geom) {
    NV_PDL_ENTRY();
    const int nh = nhkv & 0xFFFF, kv = nhkv >> 16, hd = hdvd & 0xFFFF, vd = hdvd >> 16;
    // geom layout, packed host-side at the nv_attn_dsa_split launch in model.rs:
    //     K [0:16) | nsplits [16:24) | split_len [24:40) | S [40:64)
    // S gets 24 bits ON PURPOSE: it is the layer's KV row count (= max_pos for a DSA layer) and is
    // the per-head stride g*S+t, and 262144 does not fit in 16. It used to be packed at bit 48 and
    // masked with 0xFFFF, which truncated it to 0 at the serving context and aliased every KV head
    // onto head 0's keys -- silently, and only once max_pos > 65535, so every slice probe stayed green.
    const int K = (int)(geom & 0xFFFF), nsplits = (int)((geom >> 16) & 0xFF), split_len = (int)((geom >> 24) & 0xFFFF), S = (int)((geom >> 40) & 0xFFFFFF);
    const int sp = blockIdx.x % nsplits;
    const int g = (blockIdx.x / nsplits) % kv;
    const int m = blockIdx.x / (nsplits * kv);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gqa = nh / kv;
    const int dq = hd >> 5, dv = vd >> 5;                       // 6, 4
    // slot_ids[m] = the request whose cache row m gathers from: kc [n_slots][kv][S][hd] (vc [n_slots][kv][S][vd])
    const long long skc = (long long)slot_ids[m] * ((long long)kv * S * hd);
    const long long svc = (long long)slot_ids[m] * ((long long)kv * S * vd);
    const int ns = nsel[m];
    const int begin = sp * split_len;
    const int end = min((sp + 1) * split_len, ns);
    const float scale = rsqrtf((float)hd);
    const int* srow = sel + (long long)m * K;
    for (int w = warp; w < gqa; w += 8) {
        const int h = g * gqa + w;
        const __half* qr = q + ((long long)m * nh + h) * hd + lane * dq;
        float qv[6];
        #pragma unroll
        for (int i = 0; i < 6; i++) qv[i] = (i < dq) ? nv_h2f(qr[i]) : 0.f;
        float mx = -INFINITY, l = 0.f, acc[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) acc[i] = 0.f;
        // Software pipeline (bitwise): key r+1's K (dq halves) and V (dv halves) rows are loaded while
        // key r is reduced. The loads are packed u32 pairs of halves at the lane's 4-B-aligned offset
        // (dq = 6, dv = 4 halves per lane); the values and the per-key arithmetic order are those of
        // the scalar loop this replaces. Without this each key cost a full dependent load latency
        // (~1 us/key, 181 us per launch at the owner's contexts).
        uint32_t kw[3] = {0u, 0u, 0u}, vw[2] = {0u, 0u};
        if (begin < end) {
            const int t0 = srow[begin];
            const uint32_t* kp = (const uint32_t*)(kc + skc + ((long long)g * S + t0) * hd + lane * dq);
            const uint32_t* vp = (const uint32_t*)(vc + svc + ((long long)g * S + t0) * vd + lane * dv);
            #pragma unroll
            for (int i = 0; i < 3; i++) if (2 * i < dq) kw[i] = kp[i];
            #pragma unroll
            for (int i = 0; i < 2; i++) if (2 * i < dv) vw[i] = vp[i];
        }
        for (int r = begin; r < end; r++) {
            uint32_t kn[3] = {0u, 0u, 0u}, vn[2] = {0u, 0u};
            if (r + 1 < end) {
                const int tn = srow[r + 1];
                const uint32_t* kp = (const uint32_t*)(kc + skc + ((long long)g * S + tn) * hd + lane * dq);
                const uint32_t* vp = (const uint32_t*)(vc + svc + ((long long)g * S + tn) * vd + lane * dv);
                #pragma unroll
                for (int i = 0; i < 3; i++) if (2 * i < dq) kn[i] = kp[i];
                #pragma unroll
                for (int i = 0; i < 2; i++) if (2 * i < dv) vn[i] = vp[i];
            }
            float dot = 0.f;
            #pragma unroll
            for (int i = 0; i < 6; i++) if (i < dq) dot += qv[i] * nv_h2f(__ushort_as_half((unsigned short)((i & 1) ? (kw[i >> 1] >> 16) : (kw[i >> 1] & 0xFFFFu))));
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
            const float s = dot * scale;
            const float mn = fmaxf(mx, s);
            const float f = __expf(mx - mn), wt = __expf(s - mn);
            l = l * f + wt;
            #pragma unroll
            for (int i = 0; i < 4; i++) if (i < dv) acc[i] = acc[i] * f + wt * nv_h2f(__ushort_as_half((unsigned short)((i & 1) ? (vw[i >> 1] >> 16) : (vw[i >> 1] & 0xFFFFu))));
            mx = mn;
            #pragma unroll
            for (int i = 0; i < 3; i++) kw[i] = kn[i];
            #pragma unroll
            for (int i = 0; i < 2; i++) vw[i] = vn[i];
        }
        const long long hg = (long long)m * nh + h;
        if (lane == 0) { pm[hg * nsplits + sp] = mx; pl[hg * nsplits + sp] = l; }
        #pragma unroll
        for (int i = 0; i < 4; i++) if (i < dv) pacc[(hg * nsplits + sp) * vd + lane * dv + i] = acc[i];
    }
}

// PREFILL variant of nv_attn_dsa_split (m > 16): same grid / partial layout, but each warp walks the key list
// ONCE for its (up to 2) heads w and w + 8, sharing the K / V loads. Per head the arithmetic and order are
// nv_attn_dsa_split's: bitwise identical partials. Requires gqa % 8 == 0 and gqa <= 16.
extern "C" __global__ void nv_attn_dsa_split_grp(float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc,
                                                 const __half* __restrict__ q, const __half* __restrict__ kc,
                                                 const __half* __restrict__ vc, const int* __restrict__ sel,
                                                 const int* __restrict__ nsel, const int* __restrict__ slot_ids,
                                                 int nhkv, int hdvd, long long geom) {
    NV_PDL_ENTRY();
    const int nh = nhkv & 0xFFFF, kv = nhkv >> 16, hd = hdvd & 0xFFFF, vd = hdvd >> 16;
    // geom layout, packed host-side at the nv_attn_dsa_split launch in model.rs:
    //     K [0:16) | nsplits [16:24) | split_len [24:40) | S [40:64)
    // S gets 24 bits ON PURPOSE: it is the layer's KV row count (= max_pos for a DSA layer) and is
    // the per-head stride g*S+t, and 262144 does not fit in 16. It used to be packed at bit 48 and
    // masked with 0xFFFF, which truncated it to 0 at the serving context and aliased every KV head
    // onto head 0's keys -- silently, and only once max_pos > 65535, so every slice probe stayed green.
    const int K = (int)(geom & 0xFFFF), nsplits = (int)((geom >> 16) & 0xFF), split_len = (int)((geom >> 24) & 0xFFFF), S = (int)((geom >> 40) & 0xFFFFFF);
    const int sp = blockIdx.x % nsplits;
    const int g = (blockIdx.x / nsplits) % kv;
    const int m = blockIdx.x / (nsplits * kv);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gqa = nh / kv;
    const int nj = gqa >> 3;                                    // 1 or 2 heads per warp
    const int dq = hd >> 5, dv = vd >> 5;
    // slot_ids[m] = the request whose cache row m gathers from: kc [n_slots][kv][S][hd] (vc [n_slots][kv][S][vd])
    const long long skc = (long long)slot_ids[m] * ((long long)kv * S * hd);
    const long long svc = (long long)slot_ids[m] * ((long long)kv * S * vd);
    const int ns = nsel[m];
    const int begin = sp * split_len;
    const int end = min((sp + 1) * split_len, ns);
    const float scale = rsqrtf((float)hd);
    const int* srow = sel + (long long)m * K;
    float qv[2][6], mx[2], l[2], acc[2][4];
    #pragma unroll
    for (int j = 0; j < 2; j++) {
        const __half* qr = q + ((long long)m * nh + g * gqa + warp + 8 * j) * hd + lane * dq;
        #pragma unroll
        for (int i = 0; i < 6; i++) qv[j][i] = (j < nj && i < dq) ? nv_h2f(qr[i]) : 0.f;
        mx[j] = -INFINITY; l[j] = 0.f;
        #pragma unroll
        for (int i = 0; i < 4; i++) acc[j][i] = 0.f;
    }
    for (int r = begin; r < end; r++) {
        const int t = srow[r];
        const __half* kr = kc + skc + ((long long)g * S + t) * hd + lane * dq;
        const __half* vr = vc + svc + ((long long)g * S + t) * vd + lane * dv;
        float kk[6], vv[4];
        #pragma unroll
        for (int i = 0; i < 6; i++) kk[i] = (i < dq) ? nv_h2f(kr[i]) : 0.f;
        #pragma unroll
        for (int i = 0; i < 4; i++) vv[i] = (i < dv) ? nv_h2f(vr[i]) : 0.f;
        #pragma unroll
        for (int j = 0; j < 2; j++) if (j < nj) {
            float dot = 0.f;
            #pragma unroll
            for (int i = 0; i < 6; i++) if (i < dq) dot += qv[j][i] * kk[i];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
            const float s = dot * scale;
            const float mn = fmaxf(mx[j], s);
            const float f = __expf(mx[j] - mn), wt = __expf(s - mn);
            l[j] = l[j] * f + wt;
            #pragma unroll
            for (int i = 0; i < 4; i++) if (i < dv) acc[j][i] = acc[j][i] * f + wt * vv[i];
            mx[j] = mn;
        }
    }
    #pragma unroll
    for (int j = 0; j < 2; j++) if (j < nj) {
        const long long hg = (long long)m * nh + g * gqa + warp + 8 * j;
        if (lane == 0) { pm[hg * nsplits + sp] = mx[j]; pl[hg * nsplits + sp] = l[j]; }
        #pragma unroll
        for (int i = 0; i < 4; i++) if (i < dv) pacc[(hg * nsplits + sp) * vd + lane * dv + i] = acc[j][i];
    }
}

// grid (M * nh), block vd
extern "C" __global__ void nv_attn_combine(__half* __restrict__ out, const float* __restrict__ pm,
                                           const float* __restrict__ pl, const float* __restrict__ pacc,
                                           int nsplits, int vd) {
    NV_PDL_ENTRY();
    const long long hg = blockIdx.x;
    const int d = threadIdx.x;
    float mg = -INFINITY;
    for (int s = 0; s < nsplits; s++) mg = fmaxf(mg, pm[hg * nsplits + s]);
    float l = 0.f, acc = 0.f;
    for (int s = 0; s < nsplits; s++) {
        const float m = pm[hg * nsplits + s];
        if (m > -INFINITY) { const float w = __expf(m - mg); l += pl[hg * nsplits + s] * w; acc += pacc[(hg * nsplits + s) * vd + d] * w; }
    }
    out[hg * vd + d] = nv_f2h(acc / l);
}

// ---- router GEMV in f32: logits [M][E] = h f16 [M][H] . W f16 [E][H]^T (the pack stores the gate as
// F16; the reference computes in f32). grid (ceil(E/4), M), block 128 = 4 warps = 4 experts.
extern "C" __global__ void nv_router_logits(float* __restrict__ logits, const __half* __restrict__ h,
                                            const __half* __restrict__ w, int M, int E, int H) {
    NV_PDL_ENTRY();
    const int m = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int e = blockIdx.x * 4 + warp;
    if (e >= E) return;
    const __half* hr = h + (long long)m * H;
    const __half* wr = w + (long long)e * H;
    float acc = 0.f;
    for (int i = lane; i < H; i += 32) acc += nv_h2f(hr[i]) * nv_h2f(wr[i]);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) logits[(long long)m * E + e] = acc;
}

// ---- sigmoid + noaux_tc selection (moe_router_topk_sigmoid_f32_b, row-major ids/wts [M][K]):
// select on sigmoid(l) + bias (ties -> lowest id), weights = the un-biased sigmoid at the picks,
// renormalized (norm_topk) and scaled. grid (M), block 256, smem E floats.
extern "C" __global__ void nv_router_topk(int* __restrict__ ids, float* __restrict__ wts, const float* __restrict__ logits,
                                          const float* __restrict__ bias, int E, int K, int norm_topk, float scaling) {
    NV_PDL_ENTRY();
    const int m = blockIdx.x;
    extern __shared__ float s[];
    for (int e = threadIdx.x; e < E; e += blockDim.x) s[e] = 1.f / (1.f + __expf(-logits[(long long)m * E + e])) + bias[e];
    __syncthreads();
    if (threadIdx.x < 32) {
        const int lane = threadIdx.x;
        for (int j = 0; j < K; j++) {
            float lv = -1e30f; int li = 0x7fffffff;
            for (int e = lane; e < E; e += 32) if (s[e] > lv || (s[e] == lv && e < li)) { lv = s[e]; li = e; }
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_down_sync(0xffffffffu, lv, o);
                const int oi = __shfl_down_sync(0xffffffffu, li, o);
                if (ov > lv || (ov == lv && oi < li)) { lv = ov; li = oi; }
            }
            const int best = __shfl_sync(0xffffffffu, li, 0);
            if (lane == 0) ids[(long long)m * K + j] = best;
            __syncwarp();
            if (lane == 0) s[best] = -1e30f;
            __syncwarp();
        }
        if (lane == 0) {
            float ev[16], wsum = 0.f;
            for (int j = 0; j < K; j++) {
                const float l = logits[(long long)m * E + ids[(long long)m * K + j]];
                ev[j] = 1.f / (1.f + __expf(-l)); wsum += ev[j];
            }
            const float winv = norm_topk ? 1.f / (wsum + 1e-20f) : 1.f;
            for (int j = 0; j < K; j++) wts[(long long)m * K + j] = ev[j] * winv * scaling;
        }
    }
}

// ---- device-side routing (xq_moe_route): first-seen slots over ids[0..M*K] row-major; per-slot
// trellis word offsets for the three expert tables; esel[0] = live expert count. grid (1), block 256.
extern "C" __global__ void nv_moe_route(const int* __restrict__ ids, int mke,
                                        int* __restrict__ slotmap, int* __restrict__ idxmap, int* __restrict__ esel,
                                        unsigned long long* __restrict__ offs_g, unsigned long long* __restrict__ offs_u,
                                        unsigned long long* __restrict__ offs_d,
                                        unsigned long long g_words, unsigned long long u_words, unsigned long long d_words,
                                        const int* __restrict__ local_of) {
    NV_PDL_ENTRY();
    // TP (NAIVE §5.1): local_of[e] = this rank's LOCAL index of expert e (its position in the rank's
    // concatenated trellis / suh / svh), or -1 = another rank's expert: it gets no slot here (slotmap
    // stays -1, so nv_moe_combine skips it on this rank; the owning rank adds it, the fp32 all-reduce
    // of the partial combines sums the ranks). idxmap carries LOCAL indices; slotmap is keyed GLOBAL.
    const int M = mke & 0xFF, K = (mke >> 8) & 0xFF, E = mke >> 16;
    __shared__ int cnt;
    const int tid = threadIdx.x;
    for (int e = tid; e < E; e += blockDim.x) slotmap[e] = -1;
    __syncthreads();
    if (tid == 0) {
        int c = 0;
        for (int t = 0; t < M * K; t++) {
            const int e = ids[t];
            if (e >= 0 && e < E && slotmap[e] == -1) {
                const int le = local_of[e];
                if (le < 0) continue;
                slotmap[e] = c; idxmap[c] = le; c++;
            }
        }
        cnt = c; esel[0] = c;
    }
    __syncthreads();
    for (int s2 = tid; s2 < cnt; s2 += blockDim.x) {
        const int le = idxmap[s2];
        offs_g[s2] = (unsigned long long)le * g_words;
        offs_u[s2] = (unsigned long long)le * u_words;
        offs_d[s2] = (unsigned long long)le * d_words;
    }
}

// ---- per-expert input Hadamard (xq_had_suh_multi): xh[e][m][:] = H(x[m][:] * suh[idxmap[e]][:]);
// per_expert_in: x is itself [E][M][K]. grid E*M*(K/128), block 32.
extern "C" __global__ void nv_had_suh_multi(const __half* __restrict__ x, const __half* __restrict__ suh_all,
                                            const int* __restrict__ idxmap, __half* __restrict__ xh, int M, int K,
                                            const int* __restrict__ esel, int per_expert_in) {
    NV_PDL_ENTRY();
    const int nblk = K >> 7;
    const int blk = blockIdx.x % nblk;
    const long long em = blockIdx.x / nblk;
    const int m = (int)(em % M), e = (int)(em / M);
    if (e >= *esel) return;
    const int lane = threadIdx.x & 31;
    const long long base_in = (long long)(per_expert_in ? e : 0) * M * K + (long long)m * K + (long long)blk * 128 + lane * 4;
    const long long base_out = ((long long)e * M + m) * K + (long long)blk * 128 + lane * 4;
    const __half* sr = suh_all + (long long)idxmap[e] * K + (long long)blk * 128 + lane * 4;
    float v[4];
    #pragma unroll
    for (int r = 0; r < 4; r++) v[r] = nv_h2f(x[base_in + r]) * nv_h2f(sr[r]);
    nv_had128(v, lane);
    #pragma unroll
    for (int r = 0; r < 4; r++) xh[base_out + r] = nv_f2h(v[r]);
}

// ---- per-expert output Hadamard (xq_had_svh_multi): y[e][m][:] = H(yraw[e][m][:]) * svh[idxmap[e]][:].
extern "C" __global__ void nv_had_svh_multi(const __half* __restrict__ yraw, const __half* __restrict__ svh_all,
                                            const int* __restrict__ idxmap, __half* __restrict__ y, int M, int N,
                                            const int* __restrict__ esel) {
    NV_PDL_ENTRY();
    const int nblk = N >> 7;
    const int blk = blockIdx.x % nblk;
    const long long em = blockIdx.x / nblk;
    const int m = (int)(em % M), e = (int)(em / M);
    if (e >= *esel) return;
    const int lane = threadIdx.x & 31;
    const long long base = ((long long)e * M + m) * N + (long long)blk * 128 + lane * 4;
    const __half* sr = svh_all + (long long)idxmap[e] * N + (long long)blk * 128 + lane * 4;
    float v[4];
    #pragma unroll
    for (int r = 0; r < 4; r++) v[r] = nv_h2f(yraw[base + r]);
    nv_had128(v, lane);
    #pragma unroll
    for (int r = 0; r < 4; r++) y[base + r] = nv_f2h(v[r] * nv_h2f(sr[r]));
}

// ---- din[i] = silu(g[i]) * u[i] over n f16 (the expert gate/up products; grids padded to the
// top-k cap are sized by the host from E_cap*M*mi; absent slots compute on zeros harmlessly).
extern "C" __global__ void nv_silu_mul(__half* __restrict__ din, const __half* __restrict__ g,
                                       const __half* __restrict__ u, long long n) {
    NV_PDL_ENTRY();
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gv = nv_h2f(g[i]);
    din[i] = nv_f2h(gv / (1.f + __expf(-gv)) * nv_h2f(u[i]));
}

// ---- MoE combine into the f32 residual: x[m][:] += sum_j wts[m][j] * y_d[slot(ids[m][j])][m][:].
// grid (M), block 1024.
extern "C" __global__ void nv_moe_combine(float* __restrict__ x, const __half* __restrict__ y_d,
                                          const int* __restrict__ ids, const float* __restrict__ wts,
                                          const int* __restrict__ slotmap, int K, int H, int M, int accumulate) {
    NV_PDL_ENTRY();
    // accumulate = 1: x += combine (the residual); 0: x = combine (TP: this rank's fp32 partial)
    const int m = blockIdx.x;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < K; j++) {
            const int slot = slotmap[ids[(long long)m * K + j]];
            if (slot >= 0) acc += wts[(long long)m * K + j] * nv_h2f(y_d[((long long)slot * M + m) * H + i]);
        }
        if (accumulate) x[(long long)m * H + i] += acc; else x[(long long)m * H + i] = acc;
    }
}

// ---- TP glue (NAIVE §5.1): the partial sums travel as fp32 through the doorbell all-reduce.
// dst[i] = (float) src[i]
extern "C" __global__ void nv_cvt_f16_f32(float* __restrict__ dst, const __half* __restrict__ src, long long n) {
    NV_PDL_ENTRY();
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = nv_h2f(src[i]);
}
// x[i] += y[i] (both f32)
extern "C" __global__ void nv_add_f32(float* __restrict__ x, const float* __restrict__ y, long long n) {
    NV_PDL_ENTRY();
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}
// Column compaction: dst[m][0..n) = src[m][off .. off + n) for M rows (the 128-block-rounded k/v
// slices of a rank whose kv-head range is not 128-aligned: TP4 on the DSA layers). grid ceil(M*n/256).
extern "C" __global__ void nv_cols_copy(__half* __restrict__ dst, const __half* __restrict__ src, int M, int src_n, int off, int n) {
    NV_PDL_ENTRY();
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)M * n) return;
    const int m = (int)(i / n), c = (int)(i % n);
    dst[i] = src[(long long)m * src_n + off + c];
}

