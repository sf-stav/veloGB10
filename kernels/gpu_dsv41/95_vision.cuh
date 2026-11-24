// =============================================================================
// kernels/gpu_dsv41/95_vision.cuh — DeepSeek-V4.1-Flash Vision Tower + Aligner (leg VIS1 / M4).
//
// Kernels (all sm_121a, D41_PDL_ENTRY, 0 stack frame, 0 register spills, smem <= 48 KB):
//   d41_vis_gemm_bf16    Tiled BF16/F16 m16n8k16 tensor-core GEMM (64x64x32 double-buffered,
//                        arbitrary M/N/K incl. K=588, optional bias, epilogues NONE / RESIDUAL / GELU_ERF)
//   d41_vis_rmsnorm      RMSNorm(dim=1024, eps=1e-6) in F32 -> BF16
//   d41_vis_qkv_rope     Split wqkv [N, 3072] -> head-major Q, K, V [16, N, 64] + 2D RoPE on Q, K
//   d41_vis_sdpa_bf16    Tiled FlashAttention-2 bidirectional SDPA (16 heads, head_dim=64, 64x64 tiles,
//                        m16n8k16 BF16 Q@K^T and P@V with online F32 softmax, 0 global [N,N] matrix)
//   d41_vis_swiglu       ViT MLP SwiGLU [N, 5632] -> [N, 2816] BF16 (silu(gate) * up, no clamp)
//   d41_vis_unfold3x3    Aligner bottom/right zero-pad + 3x3 channel-outer unfold -> [N_llm, 9216] BF16
//   d41_vis_splice_embed Rank-0 prefill chunk embedding overwrite (image_start/end/newline + aligner rows)
// =============================================================================
#pragma once

#define D41V_EPI_NONE     0
#define D41V_EPI_RESIDUAL 1
#define D41V_EPI_GELU_ERF 2

struct d41_vis_gemm_args {
    const void* x;          // BF16 [m, lda]
    const void* w;          // BF16 or F16 [n, ldw]
    const void* bias;       // optional [n] (null = 0)
    const void* resid;      // optional BF16 [m, ldr] for D41V_EPI_RESIDUAL
    void* y;                // BF16 [m, ldy]
    int m, n, k;
    int lda, ldw, ldy, ldr;
    unsigned w_fmt;         // D41_FMT_BF16 (1) or D41_FMT_F16 (2)
    unsigned bias_fmt;      // 0 = none, 1 = BF16, 2 = F16, 3 = F32
    int epi;                // D41V_EPI_*
    int pad0;
};

struct d41_vis_norm_args {
    const void* x;          // BF16 [rows, dim]
    const void* w;          // BF16/F16/F32 [dim]
    void* y;                // BF16 [rows, dim]
    int rows, dim;
    unsigned w_fmt;
    float eps;
};

struct d41_vis_qkv_rope_args {
    const void* qkv;        // BF16 [n_tok, 3072]
    void* q_out;            // BF16 [16, n_tok, 64]
    void* k_out;            // BF16 [16, n_tok, 64]
    void* v_out;            // BF16 [16, n_tok, 64]
    int n_tok;
    int n_w;
    float inv_freq[16];     // 1.0 / (theta ** (arange(0, 32, 2) / 32))
};

struct d41_vis_sdpa_args {
    const void* q;          // BF16 [n_heads, n_tok, 64]
    const void* k;          // BF16 [n_heads, n_tok, 64]
    const void* v;          // BF16 [n_heads, n_tok, 64]
    void* out;              // BF16 [n_tok, n_heads * 64]
    int n_tok;
    int n_heads;
    float scale;            // 0.125f = 1 / sqrt(64)
    int pad0;
};

struct d41_vis_swiglu_args {
    const void* x;          // BF16 [rows, 2 * inter]
    void* y;                // BF16 [rows, inter]
    int rows;
    int inter;              // 2816
};

struct d41_vis_unfold_args {
    const void* x;          // BF16 [n_h * n_w, dim]
    void* y;                // BF16 [n_llm_h * n_llm_w, dim * 9]
    int n_h, n_w;
    int n_llm_h, n_llm_w;
    int dim;                // 1024
    int pad0;
};

struct d41_vis_splice_args {
    void* embed;            // BF16 [rows, dim] in-place target
    const void* aligner;    // BF16 [n_aligner_rows, dim]
    const void* img_start;  // BF16 [dim]
    const void* img_newline;// BF16 [dim]
    const void* img_end;    // BF16 [dim]
    const int* row_ops;     // i32 [rows]: -1=TEXT, -2=START, -3=NEWLINE, -4=END, >=0 = aligner row index
    int rows;
    int dim;                // 5120
};

__device__ __forceinline__ float d41v_ld_param(const void* p, unsigned fmt, int i) {
    if (fmt == D41_FMT_BF16) return __bfloat162float(((const __nv_bfloat16*)p)[i]);
    if (fmt == D41_FMT_F16) return __half2float(((const __half*)p)[i]);
    return ((const float*)p)[i];
}

__device__ __forceinline__ uint16_t d41v_to_bf16_bits(const void* p, unsigned fmt, size_t idx) {
    if (fmt == D41_FMT_BF16) return ((const uint16_t*)p)[idx];
    const float v = __half2float(((const __half*)p)[idx]);
    return __bfloat16_as_ushort(__float2bfloat16_rn(v));
}

__device__ __forceinline__ uint32_t d41v_f2_bf16x2(float lo, float hi) {
    const unsigned short a = __bfloat16_as_ushort(__float2bfloat16_rn(lo));
    const unsigned short b = __bfloat16_as_ushort(__float2bfloat16_rn(hi));
    return (uint32_t)a | ((uint32_t)b << 16);
}

__device__ __forceinline__ void d41v_ldsm4(unsigned smem_addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_addr));
}

__device__ __forceinline__ void d41v_mma_bf16(
    float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Conflict-free swizzle for [64 rows][32 bf16 = 4 x 16B chunks] (4096 B).
__device__ __forceinline__ unsigned d41v_swz32(int row, int ch) {
    return (unsigned)(row * 64 + ((ch ^ ((row >> 1) & 3)) << 4));
}

// Conflict-free swizzle for [64 rows][64 bf16 = 8 x 16B chunks] (8192 B).
__device__ __forceinline__ unsigned d41v_swz64(int row, int ch) {
    return (unsigned)(row * 128 + ((ch ^ (row & 7)) << 4));
}

// ---- 1. Tiled BF16 Tensor-Core GEMM (BM=64, BN=64, BK=32, 4 warps = 128 threads, 16 KB smem) ----

extern "C" __global__ void __launch_bounds__(128) d41_vis_gemm_bf16(d41_vis_gemm_args a) {
    D41_PDL_ENTRY();
    __shared__ __align__(128) unsigned char smem[2 * 8192]; // 2 stages x (4 KB A + 4 KB B) = 16 KB

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm = warp >> 1;   // 0..2 (32 M rows per warp)
    const int wn = warp & 1;    // 0..2 (32 N cols per warp)

    const int m0 = blockIdx.y * 64;
    const int n0 = blockIdx.x * 64;
    if (m0 >= a.m || n0 >= a.n) return;

    const unsigned s_base = (unsigned)__cvta_generic_to_shared(smem);
    const int n_stages = (a.k + 31) >> 5;

    auto load_stage = [&](int st) {
        const int buf = st & 1;
        unsigned char* sA = smem + buf * 8192;
        unsigned char* sB = sA + 4096;
        const int k0 = st * 32;
        const bool in_a = (m0 + 64 <= a.m) && (k0 + 32 <= a.k);
        const bool in_b = (n0 + 64 <= a.n) && (k0 + 32 <= a.k) && (a.w_fmt == D41_FMT_BF16);
        const bool a_align16 = ((a.lda & 7) == 0) && ((((uintptr_t)a.x) & 15u) == 0u);
        const bool a_align8  = ((a.lda & 3) == 0) && ((((uintptr_t)a.x) & 7u) == 0u);
        const bool b_align16 = ((a.ldw & 7) == 0) && ((((uintptr_t)a.w) & 15u) == 0u);
        const bool b_align8  = ((a.ldw & 3) == 0) && ((((uintptr_t)a.w) & 7u) == 0u);

        #pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int idx = tid + p * 128; // 0..256: 64 rows x 4 chunks of 8 bf16
            const int r = idx >> 2;
            const int ch = idx & 3;
            const unsigned off = d41v_swz32(r, ch);
            if (in_a && a_align16) {
                const uint4 v = *(const uint4*)((const uint16_t*)a.x + (size_t)(m0 + r) * a.lda + k0 + ch * 8);
                *(uint4*)(sA + off) = v;
            } else if (in_a && a_align8) {
                const uint2* sp = (const uint2*)((const uint16_t*)a.x + (size_t)(m0 + r) * a.lda + k0 + ch * 8);
                const uint2 v0 = sp[0], v1 = sp[1];
                *(uint4*)(sA + off) = make_uint4(v0.x, v0.y, v1.x, v1.y);
            } else {
                uint16_t* dst = (uint16_t*)(sA + off);
                const int gr = m0 + r;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int gk = k0 + ch * 8 + j;
                    dst[j] = (gr < a.m && gk < a.k) ? ((const uint16_t*)a.x)[(size_t)gr * a.lda + gk] : 0u;
                }
            }

            if (in_b && b_align16) {
                const uint4 v = *(const uint4*)((const uint16_t*)a.w + (size_t)(n0 + r) * a.ldw + k0 + ch * 8);
                *(uint4*)(sB + off) = v;
            } else if (in_b && b_align8) {
                const uint2* sp = (const uint2*)((const uint16_t*)a.w + (size_t)(n0 + r) * a.ldw + k0 + ch * 8);
                const uint2 v0 = sp[0], v1 = sp[1];
                *(uint4*)(sB + off) = make_uint4(v0.x, v0.y, v1.x, v1.y);
            } else {
                uint16_t* dst = (uint16_t*)(sB + off);
                const int gn = n0 + r;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int gk = k0 + ch * 8 + j;
                    dst[j] = (gn < a.n && gk < a.k) ? d41v_to_bf16_bits(a.w, a.w_fmt, (size_t)gn * a.ldw + gk) : 0u;
                }
            }
        }
    };

    float acc[2][4][4];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt)
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[mt][nt][i] = 0.0f;

    load_stage(0);
    __syncthreads();

    for (int st = 0; st < n_stages; ++st) {
        const int buf = st & 1;
        const unsigned au = s_base + (unsigned)(buf * 8192);
        const unsigned bu = au + 4096u;

        if (st + 1 < n_stages) {
            load_stage(st + 1);
        }

        #pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            uint32_t af[2][4];
            #pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const int mr = wm * 32 + mt * 16 + (lane & 15);
                const int ach = 2 * kk + (lane >> 4);
                d41v_ldsm4(au + d41v_swz32(mr, ach), af[mt][0], af[mt][1], af[mt][2], af[mt][3]);
            }
            #pragma unroll
            for (int np = 0; np < 2; ++np) {
                uint32_t b0_0, b0_1, b1_0, b1_1;
                const int nr = wn * 32 + np * 16 + (lane & 7) + 8 * (lane >> 4);
                const int bch = 2 * kk + ((lane >> 3) & 1);
                d41v_ldsm4(bu + d41v_swz32(nr, bch), b0_0, b0_1, b1_0, b1_1);
                #pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    d41v_mma_bf16(acc[mt][2 * np + 0], af[mt][0], af[mt][1], af[mt][2], af[mt][3], b0_0, b0_1);
                    d41v_mma_bf16(acc[mt][2 * np + 1], af[mt][0], af[mt][1], af[mt][2], af[mt][3], b1_0, b1_1);
                }
            }
        }
        __syncthreads();
    }

    const int gid = lane >> 2;
    const int tig = lane & 3;
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
        const int r0 = m0 + wm * 32 + mt * 16 + gid;
        const int r1 = r0 + 8;
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const int c0 = n0 + wn * 32 + nt * 8 + 2 * tig;
            const int c1 = c0 + 1;
            const float b0 = (a.bias && a.bias_fmt && c0 < a.n) ? d41v_ld_param(a.bias, a.bias_fmt, c0) : 0.0f;
            const float b1 = (a.bias && a.bias_fmt && c1 < a.n) ? d41v_ld_param(a.bias, a.bias_fmt, c1) : 0.0f;

            auto finish_elt = [&](float val, float b, int gr, int gc) {
                if (gr >= a.m || gc >= a.n) return;
                float v = val + b;
                __nv_bfloat16 out_bf = __float2bfloat16_rn(v);
                if (a.epi == D41V_EPI_RESIDUAL && a.resid) {
                    const float rv = __bfloat162float(((__nv_bfloat16 const*)a.resid)[(size_t)gr * a.ldr + gc]);
                    out_bf = __float2bfloat16_rn(rv + __bfloat162float(out_bf));
                } else if (a.epi == D41V_EPI_GELU_ERF) {
                    const float xf = __bfloat162float(out_bf);
                    out_bf = __float2bfloat16_rn(0.5f * xf * (1.0f + erff(xf * 0.7071067811865475244f)));
                }
                ((__nv_bfloat16*)a.y)[(size_t)gr * a.ldy + gc] = out_bf;
            };

            finish_elt(acc[mt][nt][0], b0, r0, c0);
            finish_elt(acc[mt][nt][1], b1, r0, c1);
            finish_elt(acc[mt][nt][2], b0, r1, c0);
            finish_elt(acc[mt][nt][3], b1, r1, c1);
        }
    }
}

// ---- 2. ViT RMSNorm (dim = 1024, eps = 1e-6, 1 CTA of 256 threads per row) ----

extern "C" __global__ void __launch_bounds__(256) d41_vis_rmsnorm(d41_vis_norm_args a) {
    D41_PDL_ENTRY();
    __shared__ float warp_sum[8];
    const int row = blockIdx.x;
    if (row >= a.rows) return;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const __nv_bfloat16* xr = (const __nv_bfloat16*)a.x + (size_t)row * a.dim;
    __nv_bfloat16* yr = (__nv_bfloat16*)a.y + (size_t)row * a.dim;

    float sq = 0.0f;
    for (int i = tid; i < a.dim; i += 256) {
        const float v = __bfloat162float(xr[i]);
        sq = __fmaf_rn(v, v, sq);
    }
    #pragma unroll
    for (int m = 16; m >= 1; m >>= 1) {
        sq += __shfl_xor_sync(0xFFFFFFFFu, sq, m);
    }
    if (lane == 0) warp_sum[warp] = sq;
    __syncthreads();

    if (warp == 0) {
        float s = (lane < 8) ? warp_sum[lane] : 0.0f;
        #pragma unroll
        for (int m = 4; m >= 1; m >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, m);
        if (lane == 0) warp_sum[0] = rsqrtf(s / (float)a.dim + a.eps);
    }
    __syncthreads();

    const float inv = warp_sum[0];
    for (int i = tid; i < a.dim; i += 256) {
        const float v = __bfloat162float(xr[i]) * inv;
        const float w = d41v_ld_param(a.w, a.w_fmt, i);
        yr[i] = __float2bfloat16_rn(v * w);
    }
}

// ---- 3. Split wqkv [N, 3072] -> Head-Major Q, K, V [16, N, 64] + 2D Half-Split RoPE ----

extern "C" __global__ void __launch_bounds__(128) d41_vis_qkv_rope(d41_vis_qkv_rope_args a) {
    D41_PDL_ENTRY();
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31; // 0..32 covers half-split pairs (lane, lane + 32) of head_dim = 64
    const int row = blockIdx.x * 4 + warp;
    const int head = blockIdx.y;
    if (row >= a.n_tok || head >= 16) return;

    const int hpos = row / a.n_w;
    const int wpos = row - hpos * a.n_w;
    const float pos = (lane < 16) ? (float)hpos : (float)wpos;
    const float freq = a.inv_freq[lane & 15];
    float s, c;
    d41_sincos(__fmul_rn(pos, freq), s, c);

    const __nv_bfloat16* src = (const __nv_bfloat16*)a.qkv + (size_t)row * 3072;
    const size_t dst_off = ((size_t)head * a.n_tok + row) * 64;

    // Q (cols 0..1024)
    {
        const float q1 = __bfloat162float(src[head * 64 + lane]);
        const float q2 = __bfloat162float(src[head * 64 + 32 + lane]);
        __nv_bfloat16* qd = (__nv_bfloat16*)a.q_out + dst_off;
        qd[lane] = __float2bfloat16_rn(q1 * c - q2 * s);
        qd[32 + lane] = __float2bfloat16_rn(q2 * c + q1 * s);
    }
    // K (cols 1024..2048)
    {
        const float k1 = __bfloat162float(src[1024 + head * 64 + lane]);
        const float k2 = __bfloat162float(src[1024 + head * 64 + 32 + lane]);
        __nv_bfloat16* kd = (__nv_bfloat16*)a.k_out + dst_off;
        kd[lane] = __float2bfloat16_rn(k1 * c - k2 * s);
        kd[32 + lane] = __float2bfloat16_rn(k2 * c + k1 * s);
    }
    // V (cols 2048..3072, no RoPE)
    {
        __nv_bfloat16* vd = (__nv_bfloat16*)a.v_out + dst_off;
        vd[lane] = src[2048 + head * 64 + lane];
        vd[32 + lane] = src[2048 + head * 64 + 32 + lane];
    }
}

// ---- 4. Tiled FlashAttention-2 Bidirectional SDPA (BR=64, BC=64, D=64, 16 KB smem, 0 [N,N] matrix) ----

extern "C" __global__ void __launch_bounds__(128) d41_vis_sdpa_bf16(d41_vis_sdpa_args a) {
    D41_PDL_ENTRY();
    __shared__ __align__(128) unsigned char smem[16384]; // 8 KB Ks + 8 KB Vs_T (Qs loaded once through Ks)

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid = lane >> 2;
    const int tig = lane & 3;

    const int q_start = blockIdx.x * 64;
    const int head = blockIdx.y;
    if (q_start >= a.n_tok || head >= a.n_heads) return;

    const unsigned s_base = (unsigned)__cvta_generic_to_shared(smem);
    const unsigned ks_u = s_base;
    const unsigned vs_u = s_base + 8192u;
    unsigned char* sK = smem;
    unsigned char* sVT = smem + 8192;

    const uint16_t* q_head = (const uint16_t*)a.q + (size_t)head * a.n_tok * 64;
    const uint16_t* k_head = (const uint16_t*)a.k + (size_t)head * a.n_tok * 64;
    const uint16_t* v_head = (const uint16_t*)a.v + (size_t)head * a.n_tok * 64;

    // Stage Q [64, 64] through sK once and load each warp's [16, 64] slice into registers.
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        const int idx = tid + p * 128; // 0..512: 64 rows x 8 chunks of 8 bf16
        const int r = idx >> 3;
        const int ch = idx & 7;
        const int gr = q_start + r;
        const uint4 v = (gr < a.n_tok) ? *(const uint4*)(q_head + (size_t)gr * 64 + ch * 8) : make_uint4(0, 0, 0, 0);
        *(uint4*)(sK + d41v_swz64(r, ch)) = v;
    }
    __syncthreads();

    uint32_t q_frag[4][4];
    #pragma unroll
    for (int kd = 0; kd < 4; ++kd) {
        const int qr = warp * 16 + (lane & 15);
        const int qch = 2 * kd + (lane >> 4);
        d41v_ldsm4(ks_u + d41v_swz64(qr, qch), q_frag[kd][0], q_frag[kd][1], q_frag[kd][2], q_frag[kd][3]);
    }
    __syncthreads();

    float o_acc[8][4];
    #pragma unroll
    for (int dt = 0; dt < 8; ++dt)
        #pragma unroll
        for (int i = 0; i < 4; ++i) o_acc[dt][i] = 0.0f;

    float m0_max = -INFINITY, m1_max = -INFINITY;
    float l0_sum = 0.0f, l1_sum = 0.0f;

    for (int kv_start = 0; kv_start < a.n_tok; kv_start += 64) {
        // Load K [64, 64] into sK and V [64, 64] transposed into sVT [d=64, kv=64].
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int idx = tid + p * 128;
            const int r = idx >> 3;      // kv row 0..64
            const int ch = idx & 7;      // d chunk 0..8
            const int gk = kv_start + r;
            const uint4 kv_vec = (gk < a.n_tok) ? *(const uint4*)(k_head + (size_t)gk * 64 + ch * 8) : make_uint4(0, 0, 0, 0);
            *(uint4*)(sK + d41v_swz64(r, ch)) = kv_vec;

            const uint4 vv_vec = (gk < a.n_tok) ? *(const uint4*)(v_head + (size_t)gk * 64 + ch * 8) : make_uint4(0, 0, 0, 0);
            const uint16_t* vp = (const uint16_t*)&vv_vec;
            const int kv_ch = r >> 3;
            const int kv_in = r & 7;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int d = ch * 8 + j;
                uint16_t* dst = (uint16_t*)(sVT + d41v_swz64(d, kv_ch));
                dst[kv_in] = vp[j];
            }
        }
        __syncthreads();

        // S = Q_w [16, 64] @ K_tile [64, 64]^T
        float s_acc[8][4];
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) s_acc[nt][i] = 0.0f;

        #pragma unroll
        for (int kd = 0; kd < 4; ++kd) {
            #pragma unroll
            for (int np = 0; np < 4; ++np) {
                uint32_t b0_0, b0_1, b1_0, b1_1;
                const int kr = np * 16 + (lane & 7) + 8 * (lane >> 4);
                const int kch = 2 * kd + ((lane >> 3) & 1);
                d41v_ldsm4(ks_u + d41v_swz64(kr, kch), b0_0, b0_1, b1_0, b1_1);
                d41v_mma_bf16(s_acc[2 * np + 0], q_frag[kd][0], q_frag[kd][1], q_frag[kd][2], q_frag[kd][3], b0_0, b0_1);
                d41v_mma_bf16(s_acc[2 * np + 1], q_frag[kd][0], q_frag[kd][1], q_frag[kd][2], q_frag[kd][3], b1_0, b1_1);
            }
        }

        // Scale & mask out-of-bounds keys, compute tile row max across the 4 tig lanes per gid.
        float r0_tile_max = -INFINITY, r1_tile_max = -INFINITY;
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const int kc0 = kv_start + nt * 8 + 2 * tig;
            const int kc1 = kc0 + 1;
            s_acc[nt][0] = (kc0 < a.n_tok) ? (s_acc[nt][0] * a.scale) : -INFINITY;
            s_acc[nt][1] = (kc1 < a.n_tok) ? (s_acc[nt][1] * a.scale) : -INFINITY;
            s_acc[nt][2] = (kc0 < a.n_tok) ? (s_acc[nt][2] * a.scale) : -INFINITY;
            s_acc[nt][3] = (kc1 < a.n_tok) ? (s_acc[nt][3] * a.scale) : -INFINITY;
            r0_tile_max = fmaxf(r0_tile_max, fmaxf(s_acc[nt][0], s_acc[nt][1]));
            r1_tile_max = fmaxf(r1_tile_max, fmaxf(s_acc[nt][2], s_acc[nt][3]));
        }
        #pragma unroll
        for (int m = 1; m <= 2; m <<= 1) {
            r0_tile_max = fmaxf(r0_tile_max, __shfl_xor_sync(0xFFFFFFFFu, r0_tile_max, m));
            r1_tile_max = fmaxf(r1_tile_max, __shfl_xor_sync(0xFFFFFFFFu, r1_tile_max, m));
        }

        const float m0_new = fmaxf(m0_max, r0_tile_max);
        const float m1_new = fmaxf(m1_max, r1_tile_max);
        const float alpha0 = (m0_max == -INFINITY) ? 0.0f : expf(m0_max - m0_new);
        const float alpha1 = (m1_max == -INFINITY) ? 0.0f : expf(m1_max - m1_new);

        #pragma unroll
        for (int dt = 0; dt < 8; ++dt) {
            o_acc[dt][0] *= alpha0;
            o_acc[dt][1] *= alpha0;
            o_acc[dt][2] *= alpha1;
            o_acc[dt][3] *= alpha1;
        }
        l0_sum *= alpha0;
        l1_sum *= alpha1;
        m0_max = m0_new;
        m1_max = m1_new;

        float p0_loc = 0.0f, p1_loc = 0.0f;
        uint32_t p_frag[4][4];
        #pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            float e0[2], e1[2], e2[2], e3[2];
            e0[0] = (s_acc[2 * ks][0] == -INFINITY) ? 0.0f : expf(s_acc[2 * ks][0] - m0_new);
            e0[1] = (s_acc[2 * ks][1] == -INFINITY) ? 0.0f : expf(s_acc[2 * ks][1] - m0_new);
            e1[0] = (s_acc[2 * ks][2] == -INFINITY) ? 0.0f : expf(s_acc[2 * ks][2] - m1_new);
            e1[1] = (s_acc[2 * ks][3] == -INFINITY) ? 0.0f : expf(s_acc[2 * ks][3] - m1_new);

            e2[0] = (s_acc[2 * ks + 1][0] == -INFINITY) ? 0.0f : expf(s_acc[2 * ks + 1][0] - m0_new);
            e2[1] = (s_acc[2 * ks + 1][1] == -INFINITY) ? 0.0f : expf(s_acc[2 * ks + 1][1] - m0_new);
            e3[0] = (s_acc[2 * ks + 1][2] == -INFINITY) ? 0.0f : expf(s_acc[2 * ks + 1][2] - m1_new);
            e3[1] = (s_acc[2 * ks + 1][3] == -INFINITY) ? 0.0f : expf(s_acc[2 * ks + 1][3] - m1_new);

            p0_loc += (e0[0] + e0[1]) + (e2[0] + e2[1]);
            p1_loc += (e1[0] + e1[1]) + (e3[0] + e3[1]);

            p_frag[ks][0] = d41v_f2_bf16x2(e0[0], e0[1]);
            p_frag[ks][1] = d41v_f2_bf16x2(e1[0], e1[1]);
            p_frag[ks][2] = d41v_f2_bf16x2(e2[0], e2[1]);
            p_frag[ks][3] = d41v_f2_bf16x2(e3[0], e3[1]);
        }
        // Ordered 4-lane butterfly sum across tig=0..3 so all 4 lanes compute identical l_sum bits.
        #pragma unroll
        for (int m = 1; m <= 2; m <<= 1) {
            const float o0 = __shfl_xor_sync(0xFFFFFFFFu, p0_loc, m);
            const float o1 = __shfl_xor_sync(0xFFFFFFFFu, p1_loc, m);
            p0_loc = (tig & m) ? __fadd_rn(o0, p0_loc) : __fadd_rn(p0_loc, o0);
            p1_loc = (tig & m) ? __fadd_rn(o1, p1_loc) : __fadd_rn(p1_loc, o1);
        }
        l0_sum = __fadd_rn(l0_sum, p0_loc);
        l1_sum = __fadd_rn(l1_sum, p1_loc);

        // O_acc += P_w [16, 64] @ V_tile [64, 64] (reading V^T [64, 64] from sVT)
        #pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            #pragma unroll
            for (int dp = 0; dp < 4; ++dp) {
                uint32_t b0_0, b0_1, b1_0, b1_1;
                const int dr = dp * 16 + (lane & 7) + 8 * (lane >> 4);
                const int vch = 2 * ks + ((lane >> 3) & 1);
                d41v_ldsm4(vs_u + d41v_swz64(dr, vch), b0_0, b0_1, b1_0, b1_1);
                d41v_mma_bf16(o_acc[2 * dp + 0], p_frag[ks][0], p_frag[ks][1], p_frag[ks][2], p_frag[ks][3], b0_0, b0_1);
                d41v_mma_bf16(o_acc[2 * dp + 1], p_frag[ks][0], p_frag[ks][1], p_frag[ks][2], p_frag[ks][3], b1_0, b1_1);
            }
        }
        __syncthreads();
    }

    const int qr0 = q_start + warp * 16 + gid;
    const int qr1 = qr0 + 8;
    const float inv0 = (l0_sum > 0.0f) ? (1.0f / l0_sum) : 0.0f;
    const float inv1 = (l1_sum > 0.0f) ? (1.0f / l1_sum) : 0.0f;
    const int out_ld = a.n_heads * 64;

    #pragma unroll
    for (int dt = 0; dt < 8; ++dt) {
        const int col = head * 64 + dt * 8 + 2 * tig;
        if (qr0 < a.n_tok) {
            *(uint32_t*)((uint16_t*)a.out + (size_t)qr0 * out_ld + col) =
                d41v_f2_bf16x2(o_acc[dt][0] * inv0, o_acc[dt][1] * inv0);
        }
        if (qr1 < a.n_tok) {
            *(uint32_t*)((uint16_t*)a.out + (size_t)qr1 * out_ld + col) =
                d41v_f2_bf16x2(o_acc[dt][2] * inv1, o_acc[dt][3] * inv1);
        }
    }
}

// ---- 5. ViT MLP SwiGLU ([rows, 2 * inter] -> [rows, inter], F.silu(gate) * up, no clamp) ----

extern "C" __global__ void __launch_bounds__(256) d41_vis_swiglu(d41_vis_swiglu_args a) {
    D41_PDL_ENTRY();
    const int row = blockIdx.y;
    const int col = blockIdx.x * 256 + threadIdx.x;
    if (row >= a.rows || col >= a.inter) return;

    const __nv_bfloat16* xr = (const __nv_bfloat16*)a.x + (size_t)row * (2 * a.inter);
    const float g = __bfloat162float(xr[col]);
    const float u = __bfloat162float(xr[a.inter + col]);
    const float sg = __bfloat162float(__float2bfloat16_rn(g / (1.0f + expf(-g))));
    ((__nv_bfloat16*)a.y)[(size_t)row * a.inter + col] = __float2bfloat16_rn(sg * u);
}

// ---- 6. Aligner Zero-Pad + 3x3 Unfold (vision.py:115-118) ----

extern "C" __global__ void __launch_bounds__(256) d41_vis_unfold3x3(d41_vis_unfold_args a) {
    D41_PDL_ENTRY();
    const int out_row = blockIdx.x;
    const int total_out = a.n_llm_h * a.n_llm_w;
    if (out_row >= total_out) return;

    const int lh = out_row / a.n_llm_w;
    const int lw = out_row - lh * a.n_llm_w;
    const int out_dim = a.dim * 9;
    const uint16_t* x = (const uint16_t*)a.x;
    uint16_t* yr = (uint16_t*)a.y + (size_t)out_row * out_dim;

    for (int col = threadIdx.x; col < out_dim; col += 256) {
        const int c = col / 9;
        const int rem = col - c * 9;
        const int kh = rem / 3;
        const int kw = rem - kh * 3;
        const int sh = lh * 3 + kh;
        const int sw = lw * 3 + kw;
        yr[col] = (sh < a.n_h && sw < a.n_w) ? x[((size_t)sh * a.n_w + sw) * a.dim + c] : 0u;
    }
}

// ---- 7. Rank-0 Prefill Chunk Embedding Splice (model.py:1231-1243) ----

extern "C" __global__ void __launch_bounds__(256) d41_vis_splice_embed(d41_vis_splice_args a) {
    D41_PDL_ENTRY();
    const int r = blockIdx.x;
    if (r >= a.rows) return;
    const int op = a.row_ops[r];
    if (op == -1) return; // TEXT row: keep gathered token embedding untouched

    const uint16_t* src = nullptr;
    if (op >= 0) {
        src = (const uint16_t*)a.aligner + (size_t)op * a.dim;
    } else if (op == -2) {
        src = (const uint16_t*)a.img_start;
    } else if (op == -3) {
        src = (const uint16_t*)a.img_newline;
    } else if (op == -4) {
        src = (const uint16_t*)a.img_end;
    } else {
        return;
    }

    uint16_t* dst = (uint16_t*)a.embed + (size_t)r * a.dim;
    const int n_vec = a.dim >> 3; // 5120 / 8 = 640 uint4s
    for (int i = threadIdx.x; i < n_vec; i += 256) {
        ((uint4*)dst)[i] = ((const uint4*)src)[i];
    }
}
