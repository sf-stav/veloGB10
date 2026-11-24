// gpu_inkling2/b0_vit.cuh — the Inkling-2 vision tower (leg S5). Section of gpu_inkling2.cu; kernels `inkv_*`.
// Host side: src/inkling2/vision/gpu.rs. Oracle: src/inkling2/vision/cpu.rs (transformers Inkling2VisionModel).
//
// Data flow of one image (S = 1 + 4 T sequence rows, CLS first, patches block-major; Sp = S rounded up to 64):
//   inkv_patches   pixels [T][40][40][3] f32 -> Xp [S][2432] bf16 = [hi | lo] split of each 1200-value patch (+16 zero
//                  pads per half): one bf16 GEMM against [W | W] gives the f32-input patch embedding (pixels are not
//                  bf16-exact; the split keeps ~16 bits).
//   GEMM (ink_proj_wide)  -> Y [S][1024] f32
//   inkv_embed_ln  + bilinear position embedding (or CLS), optional embed tap, ln_pre -> H f32 (the residual stream)
//   per layer:  inkv_ln (ln_1 -> X bf16) ; GEMM qkv -> Y [S][3072] ; inkv_qkv_rope (+bias, axial RoPE, Q * 1/8, f16
//               Q / K [16][Sp][64] and V transposed [16][64][Sp]) ; inkv_attn (tensor-core flash attention) -> O bf16 ;
//               GEMM out_proj -> Y ; inkv_resid_ln (H += ls_1 (Y + b), ln_2 -> X) ; GEMM fc1 -> Y [S][4096] ;
//               inkv_gelu_bf (+b, erf GELU -> F bf16) ; GEMM fc2 -> Y ; inkv_resid_ln (H += ls_2 (Y + b), next ln_1 -> X)
//   merger:     inkv_cast_bf (H rows 1.. = [T][4096]) ; GEMM proj_in ; inkv_gelu_bf ; GEMM proj_out -> [T][2304] f32
// The residual stream stays f32; GEMM inputs are bf16 (as the reference's bf16 load); attention runs on f16 operands
// (Q, K, V, P: 11-bit mantissas) with fp32 scores / softmax / accumulation.

#define IKV_D 1024
#define IKV_HD 64
#define IKV_NH 16
#define IKV_KP 1216          // padded patch K (1200 -> multiple of 32)
#define IKV_ST 72            // attention smem row stride (halves): conflict-free 32-bit fragment loads

__device__ __forceinline__ void ikv_rc(int p, int cols, int& r, int& c) {
    const int t = p >> 2, tr = t / cols, tc = t - tr * cols;
    r = 2 * tr + ((p & 3) >> 1);
    c = 2 * tc + (p & 1);
}

// LayerNorm statistics of one 1024-row held 4 per thread (block 256), fixed reduction order.
__device__ __forceinline__ void ikv_ln_stats(const float (&v)[4], float* red, float eps, float& mean, float& inv) {
    float s[1] = {v[0] + v[1] + v[2] + v[3]};
    ika_bsum<1, 8>(s, red);
    mean = s[0] * (1.f / IKV_D);
    float q[1] = {0.f};
    #pragma unroll
    for (int i = 0; i < 4; ++i) { const float d = v[i] - mean; q[0] += d * d; }
    ika_bsum<1, 8>(q, red);
    inv = rsqrtf(q[0] * (1.f / IKV_D) + eps);
}

// grid S, block 256
extern "C" __global__ __launch_bounds__(256) void inkv_patches(const float* __restrict__ px, uint16_t* __restrict__ Xp, int cols) {
    INK2_PDL_ENTRY();  // PV1: the module launches with PDL (S7): wait for the predecessor first
    const int r = blockIdx.x;
    uint16_t* x = Xp + (size_t)r * (2 * IKV_KP);
    for (int k = threadIdx.x; k < IKV_KP; k += 256) {
        float v = 0.f;
        if (r > 0 && k < 1200) {
            const int p = r - 1, t = p >> 2, my = (p & 3) >> 1, mx = p & 1;
            const int py = k / 60, pxx = (k % 60) / 3, c = k % 3;
            v = px[(size_t)t * 4800 + ((my * 20 + py) * 40 + mx * 20 + pxx) * 3 + c];
        }
        const uint16_t hi = ika_f2b(v);
        x[k] = hi;
        x[IKV_KP + k] = ika_f2b(v - ika_b2f(hi));
    }
}

// grid S, block 256: E (optional tap) = CLS or Y + interpolated position; H = ln_pre(E)
extern "C" __global__ __launch_bounds__(256) void inkv_embed_ln(const float* __restrict__ Y, const float* __restrict__ pos,
        const float* __restrict__ cls, const float* __restrict__ w, const float* __restrict__ b, float* __restrict__ E,
        float* __restrict__ H, int rows, int cols, int side, float eps) {
    INK2_PDL_ENTRY();  // PV1: the module launches with PDL (S7): wait for the predecessor first
    __shared__ float red[8];
    const int r = blockIdx.x;
    float v[4];
    if (r == 0) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = cls[threadIdx.x + 256 * i];
    } else {
        int pr, pc;
        ikv_rc(r - 1, cols, pr, pc);
        int ti[4];
        float tw[4];
        {
            const int sz[2] = {2 * rows, 2 * cols}, ix[2] = {pr, pc};
            int t[2][2];
            float wt[2][2];
            #pragma unroll
            for (int a = 0; a < 2; ++a) {
                const float src = (((float)ix[a] + 0.5f) * (float)side) / (float)sz[a] - 0.5f;
                const float fl = floorf(src);
                #pragma unroll
                for (int o = 0; o < 2; ++o) {
                    t[a][o] = min(max((int)fl + o, 0), side - 1);
                    wt[a][o] = fmaxf(1.f - fabsf(src - fl - (float)o), 0.f);
                }
            }
            #pragma unroll
            for (int a = 0; a < 2; ++a)
                #pragma unroll
                for (int c = 0; c < 2; ++c) { ti[a * 2 + c] = t[0][a] * side + t[1][c]; tw[a * 2 + c] = wt[0][a] * wt[1][c]; }
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int d = threadIdx.x + 256 * i;
            float s = 0.f;
            #pragma unroll
            for (int q = 0; q < 4; ++q) s += pos[(size_t)ti[q] * IKV_D + d] * tw[q];
            v[i] = Y[(size_t)r * IKV_D + d] + s;
        }
    }
    if (E) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) E[(size_t)r * IKV_D + threadIdx.x + 256 * i] = v[i];
    }
    float mean, inv;
    ikv_ln_stats(v, red, eps, mean, inv);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int d = threadIdx.x + 256 * i;
        H[(size_t)r * IKV_D + d] = (v[i] - mean) * inv * w[d] + b[d];
    }
}

// grid S, block 256: X = bf16(LN(H))
extern "C" __global__ __launch_bounds__(256) void inkv_ln(const float* __restrict__ H, const float* __restrict__ w,
        const float* __restrict__ b, uint16_t* __restrict__ X, float eps) {
    INK2_PDL_ENTRY();  // PV1: the module launches with PDL (S7): wait for the predecessor first
    __shared__ float red[8];
    const int r = blockIdx.x;
    float v[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = H[(size_t)r * IKV_D + threadIdx.x + 256 * i];
    float mean, inv;
    ikv_ln_stats(v, red, eps, mean, inv);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int d = threadIdx.x + 256 * i;
        X[(size_t)r * IKV_D + d] = ika_f2b((v[i] - mean) * inv * w[d] + b[d]);
    }
}

// grid S, block 256: H += g * (Y + bias); if w: X = bf16(LN(H))
extern "C" __global__ __launch_bounds__(256) void inkv_resid_ln(float* __restrict__ H, const float* __restrict__ Y,
        const float* __restrict__ bias, const float* __restrict__ g, const float* __restrict__ w, const float* __restrict__ b,
        uint16_t* __restrict__ X, float eps) {
    INK2_PDL_ENTRY();  // PV1: the module launches with PDL (S7): wait for the predecessor first
    __shared__ float red[8];
    const int r = blockIdx.x;
    float v[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int d = threadIdx.x + 256 * i;
        const size_t o = (size_t)r * IKV_D + d;
        v[i] = H[o] + g[d] * (Y[o] + bias[d]);
        H[o] = v[i];
    }
    if (!w) return;
    float mean, inv;
    ikv_ln_stats(v, red, eps, mean, inv);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int d = threadIdx.x + 256 * i;
        X[(size_t)r * IKV_D + d] = ika_f2b((v[i] - mean) * inv * w[d] + b[d]);
    }
}

// grid ceil(n / 1024), block 256: X = bf16(gelu(Y + bias)) over rows of `n` (bias may be null); exact erf GELU
extern "C" __global__ __launch_bounds__(256) void inkv_gelu_bf(const float* __restrict__ Y, const float* __restrict__ bias,
        uint16_t* __restrict__ X, long long total, int n) {
    INK2_PDL_ENTRY();  // PV1: the module launches with PDL (S7): wait for the predecessor first
    const long long i0 = ((long long)blockIdx.x * 256 + threadIdx.x) * 4;
    if (i0 >= total) return;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const long long i = i0 + j;
        if (i >= total) break;
        float x = Y[i] + (bias ? bias[i % n] : 0.f);
        X[i] = ika_f2b(0.5f * x * (1.f + erff(x * 0.70710678118654752f)));
    }
}

// grid ceil(n / 1024), block 256: X = bf16(H) elementwise
extern "C" __global__ __launch_bounds__(256) void inkv_cast_bf(const float* __restrict__ H, uint16_t* __restrict__ X, long long total) {
    INK2_PDL_ENTRY();  // PV1: the module launches with PDL (S7): wait for the predecessor first
    const long long i0 = ((long long)blockIdx.x * 256 + threadIdx.x) * 4;
    #pragma unroll
    for (int j = 0; j < 4; ++j) if (i0 + j < total) X[i0 + j] = ika_f2b(H[i0 + j]);
}

// grid (Sp / 32, 16 heads), block 256 = 32 rows x 8 threads (8 dims = 4 rotation pairs each).
// Y [S][3072] = q | k | v before bias. Rows >= S are written as zeros.
extern "C" __global__ __launch_bounds__(256) void inkv_qkv_rope(const float* __restrict__ Y, const float* __restrict__ bias,
        const float* __restrict__ inv_freq, __half* __restrict__ Q, __half* __restrict__ K, __half* __restrict__ Vt,
        int S, int Sp, int cols) {
    INK2_PDL_ENTRY();  // PV1: the module launches with PDL (S7): wait for the predecessor first
    __shared__ __half sv[IKV_HD][33];
    const int h = blockIdx.y, lr = threadIdx.x >> 3, part = threadIdx.x & 7;
    const int r = blockIdx.x * 32 + lr;
    const int d0 = part * 8;                  // head dims d0..d0+7 = pairs d0/2 .. d0/2+3
    float q[8], k[8], v[8];
    if (r < S) {
        const float* y = Y + (size_t)r * 3 * IKV_D + h * IKV_HD + d0;
        const float* bb = bias + h * IKV_HD + d0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            q[i] = y[i] + bb[i];
            k[i] = y[IKV_D + i] + bb[IKV_D + i];
            v[i] = y[2 * IKV_D + i] + bb[2 * IKV_D + i];
        }
        int pr = -1, pc = -1;
        if (r > 0) ikv_rc(r - 1, cols, pr, pc);
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int j = d0 / 2 + u;            // pair index 0..31
            const float pos = (j < 16) ? (float)(pc + 1) : (float)(pr + 1);
            const float ang = pos * inv_freq[j & 15];
            float s, c;
            sincosf(ang, &s, &c);
            const float q1 = q[2 * u], q2 = q[2 * u + 1], k1 = k[2 * u], k2 = k[2 * u + 1];
            q[2 * u] = (q1 * c - q2 * s) * 0.125f;
            q[2 * u + 1] = (q2 * c + q1 * s) * 0.125f;
            k[2 * u] = k1 * c - k2 * s;
            k[2 * u + 1] = k2 * c + k1 * s;
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 8; ++i) { q[i] = 0.f; k[i] = 0.f; v[i] = 0.f; }
    }
    if (r < Sp) {
        __half* qo = Q + ((size_t)h * Sp + r) * IKV_HD + d0;
        __half* ko = K + ((size_t)h * Sp + r) * IKV_HD + d0;
        #pragma unroll
        for (int i = 0; i < 8; i += 2) {
            *reinterpret_cast<__half2*>(qo + i) = __floats2half2_rn(q[i], q[i + 1]);
            *reinterpret_cast<__half2*>(ko + i) = __floats2half2_rn(k[i], k[i + 1]);
        }
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) sv[d0 + i][lr] = __float2half_rn(v[i]);
    __syncthreads();
    // transposed store: 64 dims x 32 rows
    for (int e = threadIdx.x; e < IKV_HD * 32; e += 256) {
        const int dim = e >> 5, rr = e & 31, row = blockIdx.x * 32 + rr;
        if (row < Sp) Vt[((size_t)h * IKV_HD + dim) * Sp + row] = sv[dim][rr];
    }
}

__device__ __forceinline__ void ikv_cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src));
}
__device__ __forceinline__ void ikv_mma_h(float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t ikv_h2(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}

// Non-causal attention of one image. grid (Sp / 64, 16 heads), block 128 = 4 warps x 16 query rows.
// Q (pre-scaled by 1/8), K: [16][Sp][64] f16; Vt: [16][64][Sp] f16; O: [S][1024] bf16.
extern "C" __global__ __launch_bounds__(128) void inkv_attn(const __half* __restrict__ Q, const __half* __restrict__ K,
        const __half* __restrict__ Vt, uint16_t* __restrict__ O, int S, int Sp) {
    INK2_PDL_ENTRY();  // PV1: the module launches with PDL (S7): wait for the predecessor first
    __shared__ __align__(16) __half sK[2][64 * IKV_ST];
    __shared__ __align__(16) __half sV[2][64 * IKV_ST];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int h = blockIdx.y, q0 = blockIdx.x * 64 + warp * 16;
    const __half* Qh = Q + (size_t)h * Sp * IKV_HD;
    const __half* Kh = K + (size_t)h * Sp * IKV_HD;
    const __half* Vh = Vt + (size_t)h * IKV_HD * Sp;
    uint32_t qa[4][4];
    #pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
        const __half* r0 = Qh + (size_t)(q0 + gid) * IKV_HD + ks * 16 + 2 * tig;
        const __half* r1 = r0 + 8 * IKV_HD;
        qa[ks][0] = *reinterpret_cast<const uint32_t*>(r0);
        qa[ks][1] = *reinterpret_cast<const uint32_t*>(r1);
        qa[ks][2] = *reinterpret_cast<const uint32_t*>(r0 + 8);
        qa[ks][3] = *reinterpret_cast<const uint32_t*>(r1 + 8);
    }
    float o[8][4];
    #pragma unroll
    for (int i = 0; i < 8; ++i) { o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f; }
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    const int ntile = Sp / 64;
    auto load = [&](int kt, int buf) {
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int c = threadIdx.x + 128 * u, row = c >> 3, ch = c & 7;
            ikv_cp16(&sK[buf][row * IKV_ST + ch * 8], Kh + (size_t)(kt * 64 + row) * IKV_HD + ch * 8);
            ikv_cp16(&sV[buf][row * IKV_ST + ch * 8], Vh + (size_t)row * Sp + kt * 64 + ch * 8);
        }
        asm volatile("cp.async.commit_group;");
    };
    load(0, 0);
    const float L2E = 1.4426950408889634f;
    for (int kt = 0; kt < ntile; ++kt) {
        const int buf = kt & 1;
        if (kt + 1 < ntile) { load(kt + 1, buf ^ 1); asm volatile("cp.async.wait_group 1;"); }
        else asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        const __half* k_ = sK[buf];
        const __half* v_ = sV[buf];
        float s[8][4];
        // B fragments by ldmatrix.x4: rows = 8 keys, 4 matrices = dims [32 kp, +8, +16, +24) -> (b0, b1) of k-steps 2kp, 2kp + 1
        const int lrow = lane & 7, lcol = (lane >> 3) * 8;
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
            #pragma unroll
            for (int kp = 0; kp < 2; ++kp) {
                uint32_t b0, b1, b2, b3;
                ika_ldsm4(b0, b1, b2, b3, k_ + (nt * 8 + lrow) * IKV_ST + kp * 32 + lcol);
                ikv_mma_h(s[nt], qa[2 * kp][0], qa[2 * kp][1], qa[2 * kp][2], qa[2 * kp][3], b0, b1);
                ikv_mma_h(s[nt], qa[2 * kp + 1][0], qa[2 * kp + 1][1], qa[2 * kp + 1][2], qa[2 * kp + 1][3], b2, b3);
            }
        }
        if (kt * 64 + 64 > S) {
            #pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                const int key = kt * 64 + nt * 8 + 2 * tig;
                if (key >= S) { s[nt][0] = -INFINITY; s[nt][2] = -INFINITY; }
                if (key + 1 >= S) { s[nt][1] = -INFINITY; s[nt][3] = -INFINITY; }
            }
        }
        float mx0 = -INFINITY, mx1 = -INFINITY;
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) { mx0 = fmaxf(mx0, fmaxf(s[nt][0], s[nt][1])); mx1 = fmaxf(mx1, fmaxf(s[nt][2], s[nt][3])); }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
        const float a0 = exp2f((m0 - n0) * L2E), a1 = exp2f((m1 - n1) * L2E);
        m0 = n0; m1 = n1;
        l0 *= a0; l1 *= a1;
        #pragma unroll
        for (int i = 0; i < 8; ++i) { o[i][0] *= a0; o[i][1] *= a0; o[i][2] *= a1; o[i][3] *= a1; }
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            s[nt][0] = exp2f((s[nt][0] - n0) * L2E); s[nt][1] = exp2f((s[nt][1] - n0) * L2E);
            s[nt][2] = exp2f((s[nt][2] - n1) * L2E); s[nt][3] = exp2f((s[nt][3] - n1) * L2E);
            l0 += s[nt][0] + s[nt][1]; l1 += s[nt][2] + s[nt][3];
        }
        #pragma unroll
        for (int kp = 0; kp < 2; ++kp) {
            uint32_t pa[2][4];
            #pragma unroll
            for (int u = 0; u < 2; ++u) {
                const int kk = 2 * kp + u;
                pa[u][0] = ikv_h2(s[2 * kk][0], s[2 * kk][1]); pa[u][1] = ikv_h2(s[2 * kk][2], s[2 * kk][3]);
                pa[u][2] = ikv_h2(s[2 * kk + 1][0], s[2 * kk + 1][1]); pa[u][3] = ikv_h2(s[2 * kk + 1][2], s[2 * kk + 1][3]);
            }
            #pragma unroll
            for (int dn = 0; dn < 8; ++dn) {
                uint32_t b0, b1, b2, b3;
                ika_ldsm4(b0, b1, b2, b3, v_ + (dn * 8 + lrow) * IKV_ST + kp * 32 + lcol);
                ikv_mma_h(o[dn], pa[0][0], pa[0][1], pa[0][2], pa[0][3], b0, b1);
                ikv_mma_h(o[dn], pa[1][0], pa[1][1], pa[1][2], pa[1][3], b2, b3);
            }
        }
        __syncthreads();
    }
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1); l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1); l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float i0 = 1.f / l0, i1 = 1.f / l1;
    const int r0 = q0 + gid, r1 = q0 + gid + 8;
    #pragma unroll
    for (int dn = 0; dn < 8; ++dn) {
        const int col = h * IKV_HD + dn * 8 + 2 * tig;
        if (r0 < S) {
            const __nv_bfloat162 v = __floats2bfloat162_rn(o[dn][0] * i0, o[dn][1] * i0);
            *reinterpret_cast<__nv_bfloat162*>(O + (size_t)r0 * IKV_D + col) = v;
        }
        if (r1 < S) {
            const __nv_bfloat162 v = __floats2bfloat162_rn(o[dn][2] * i1, o[dn][3] * i1);
            *reinterpret_cast<__nv_bfloat162*>(O + (size_t)r1 * IKV_D + col) = v;
        }
    }
}

// grid ceil(n / 256), block 256: dst rows <- src rows (f32 [rows][n]); the splice of image features into the text stream
extern "C" __global__ __launch_bounds__(256) void inkv_rows_copy(float* __restrict__ dst, const float* __restrict__ src, long long n) {
    INK2_PDL_ENTRY();  // PV1: the module launches with PDL (S7): wait for the predecessor first
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i < n) dst[i] = src[i];
}
