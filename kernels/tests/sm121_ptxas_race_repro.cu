
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
using bf16 = __nv_bfloat16;
extern "C" __global__ void __launch_bounds__(128) gqa_attn_band_ring_b(
    __nv_bfloat16* __restrict__ out,          // [nh*hd, B] col-major
    const __nv_bfloat16* __restrict__ q,      // [nh*hd, B] col-major
    const __nv_bfloat16* __restrict__ k_cache,// [nkv, stride, hd]
    const __nv_bfloat16* __restrict__ v_cache,
    const int* pos,                            // [B] block key positions (ABSOLUTE C..C+7)
    unsigned long long ntot_stride, int nh_packed, int window_B, float scale) {
    const int nh  = nh_packed >> 20;
    const int hd  = (nh_packed >> 10) & 0x3FF;
    const int nkv = nh_packed & 0x3FF;
    const int ntot    = (int)(ntot_stride >> 32);
    const int C_ring  = (int)((ntot_stride >> 16) & 0xFFFF);
    const int stride  = (int)(ntot_stride & 0xFFFF);
    const int window  = window_B >> 4;
    const int B       = window_B & 0xF;
    const int C = ntot - B;                    // committed ctx rows
    extern __shared__ float sm[];
    const int b  = blockIdx.x / nh;
    const int qh = blockIdx.x % nh;
    const int kvh = qh / (nh / nkv);
    const int d = threadIdx.x;
    const int qp = pos[b];
    int lo = qp - (window - 1);
    if (lo < 0) lo = 0;

    float* qs = sm;              // [hd]
    float* scores = qs + hd;     // [ntot]
    float* red = scores + ntot;  // [32]
    qs[d] = __bfloat162float(q[(long long)b * (nh * hd) + (long long)qh * hd + d]);
    __syncthreads();

    const __nv_bfloat16* kbase = k_cache + (long long)kvh * (long long)stride * hd;
    const __nv_bfloat16* vbase = v_cache + (long long)kvh * (long long)stride * hd;

    // pass 1: scores + band max, ascending key index (same schedule as gqa_attn_band_b).
    float mx = -1e30f;
    for (int j = lo + d; j < ntot; j += hd) {
        const int rj = (j < C) ? (j % C_ring) : (C_ring + (j - C));
        const __nv_bfloat16* kr = kbase + (long long)rj * hd;
        float s = 0.f;
        for (int dd = 0; dd < hd; ++dd) s = fmaf(qs[dd], __bfloat162float(kr[dd]), s);
        s *= scale;
        scores[j] = s;
        mx = fmaxf(mx, s);
    }
    for (int off = 16; off; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    if ((d & 31) == 0) red[d >> 5] = mx;
    __syncthreads();
    // cross-warp reduce: SERIAL ASCENDING on thread 0 (fixed order = the host mirror's; no
    // partial-mask/runtime-offset shuffles under divergence — the sm_121 nondeterminism hunt)
    if (d == 0) {
        float v = red[0];
        for (int w = 1; w < hd / 32; ++w) v = fmaxf(v, red[w]);
        red[0] = v;
    }
    __syncthreads();
    mx = red[0];

    // pass 2: exp + sum (accurate expf, ~2 ulp — the documented S3F tail).
    float l = 0.f;
    for (int j = lo + d; j < ntot; j += hd) { const float e = expf(scores[j] - mx); scores[j] = e; l += e; }
    for (int off = 16; off; off >>= 1) l += __shfl_xor_sync(0xffffffffu, l, off);
    if ((d & 31) == 0) red[d >> 5] = l;
    __syncthreads();
    // serial ascending sum on thread 0 — the mirror's exact association
    if (d == 0) {
        float v = red[0];
        for (int w = 1; w < hd / 32; ++w) v += red[w];
        red[0] = v;
    }
    __syncthreads();
    const float inv = 1.0f / red[0];

    // pass 3: PV, ascending over the band.
    float acc = 0.f;
    for (int j = lo; j < ntot; ++j) {
        const int rj = (j < C) ? (j % C_ring) : (C_ring + (j - C));
        acc = fmaf(scores[j] * inv, __bfloat162float(vbase[(long long)rj * hd + d]), acc);
    }
    out[(long long)b * (nh * hd) + (long long)qh * hd + d] = __float2bfloat16(acc);
}
int main() { printf("A\n"); fflush(stdout);
    const int nh = 32, nkv = 8, hd = 128, B = 8, C = 512, C_ring = 2048, stride = 2056, ntot = 520;
    bf16 *q, *kc, *vc, *out, *ref;
    cudaMalloc(&q,   (size_t)B*nh*hd*sizeof(bf16));
    cudaMalloc(&kc,  (size_t)nkv*stride*hd*sizeof(bf16));
    cudaMalloc(&vc,  (size_t)nkv*stride*hd*sizeof(bf16));
    cudaMalloc(&out, (size_t)B*nh*hd*sizeof(bf16));
    cudaMalloc(&ref, (size_t)B*nh*hd*sizeof(bf16));
    // pseudo-random deterministic fill (host, LCG)
    #define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("FAIL %s: %s\n", #x, cudaGetErrorString(e_)); return 1; } } while (0)
    CK(cudaMalloc(&q,   (size_t)B*nh*hd*sizeof(bf16)));
    CK(cudaMalloc(&kc,  (size_t)nkv*stride*hd*sizeof(bf16)));
    CK(cudaMalloc(&vc,  (size_t)nkv*stride*hd*sizeof(bf16)));
    CK(cudaMalloc(&out, (size_t)B*nh*hd*sizeof(bf16)));
    CK(cudaMalloc(&ref, (size_t)B*nh*hd*sizeof(bf16)));
    unsigned rng = 123456789u;
    auto fill = [&](bf16* p, size_t n, float sc){ bf16* h = (bf16*)malloc(n*sizeof(bf16));
        for (size_t i=0;i<n;++i){ rng = rng*1664525u+1013904223u; float v = ((int)((rng>>16)&0xFFFF)-32768)/32768.0f; h[i] = __float2bfloat16(v*sc); }
        cudaMemcpy(p, h, n*sizeof(bf16), cudaMemcpyHostToDevice); free(h); };
    printf("B\n"); fflush(stdout);
    fill(q, (size_t)B*nh*hd, 1.0f);
    printf("C\n"); fflush(stdout);
    fill(kc, (size_t)nkv*stride*hd, 1.0f);
    printf("D\n"); fflush(stdout);
    fill(vc, (size_t)nkv*stride*hd, 1.0f);
    printf("E\n"); fflush(stdout);
    int* pos; cudaMalloc(&pos, B*sizeof(int)); int hp[B]; for (int i=0;i<B;++i) hp[i]=C+i;
    cudaMemcpy(pos, hp, B*sizeof(int), cudaMemcpyHostToDevice);
    printf("F\n"); fflush(stdout);
    unsigned long long packed = ((unsigned long long)ntot<<32)|((unsigned long long)C_ring<<16)|stride;
    int nh_packed = (nh<<20)|(hd<<10)|nkv;
    int window_B = (2048<<4)|B;
    float scale = 1.0f/sqrtf((float)hd);
    int smem = (hd + ntot + 32)*4;
    dim3 grid(B*nh), block(hd);
    printf("G-launch\n"); fflush(stdout);
    if (getenv("NOLAUNCH") == NULL) gqa_attn_band_ring_b<<<grid, block, smem>>>(out, q, kc, vc, pos, packed, nh_packed, window_B, scale);
    printf("H-launch-done err=%s\n", cudaGetErrorString(cudaGetLastError())); fflush(stdout);
    bf16* href = (bf16*)malloc((size_t)B*nh*hd*sizeof(bf16));
    bf16* hout = (bf16*)malloc((size_t)B*nh*hd*sizeof(bf16));
    cudaMemcpy(href, out, (size_t)B*nh*hd*sizeof(bf16), cudaMemcpyDeviceToHost);
    int bad_launch = -1, bad_idx = -1;
    for (int it=0; it<10000; ++it) {
        gqa_attn_band_ring_b<<<grid, block, smem>>>(out, q, kc, vc, pos, packed, nh_packed, window_B, scale);
        cudaMemcpy(hout, out, (size_t)B*nh*hd*sizeof(bf16), cudaMemcpyDeviceToHost);
        if (memcmp(hout, href, (size_t)B*nh*hd*sizeof(bf16)) != 0) {
            bad_launch = it;
            for (int i=0;i<B*nh*hd;++i) if (memcmp(&hout[i], &href[i], 2)) { bad_idx = i; break; }
            break;
        }
    }
    cudaError_t e = cudaGetLastError();
    if (bad_launch >= 0)
        printf("FLAKE at launch %d idx %d (b=%d qh=%d d=%d): ref=%f got=%f\n", bad_launch, bad_idx,
               bad_idx/(nh*hd), (bad_idx/hd)%nh, bad_idx%hd,
               __bfloat162float(href[bad_idx]), __bfloat162float(hout[bad_idx]));
    else
        printf("STABLE 10000 launches (err=%s)\n", cudaGetErrorString(e));
    return 0;
}
