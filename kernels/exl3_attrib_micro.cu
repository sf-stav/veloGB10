// =============================================================================
// kernels/exl3_attrib_micro.cu — S-A3-f-a X4 + X5a standalone microbenches.
// NOT registered in build.rs; compiled by hand for the attribution session only:
//   nvcc --gpu-architecture=sm_121 -O3 exl3_attrib_micro.cu -o exl3_attrib_micro
// Run on idle compute box. All timing uses clock64() deltas at the current SM
// clock (printed via nvidia-smi convention separately); cycles/warp-instruction
// are clock-invariant since both time and clock scale together.
//
// X4(a): sustained HMMA m16n8k16 row.col f32.f16.f16.f32 rate per SM.
// X4(b): pipe rates: SHF, LOP3, PRMT, IMAD(U32), IMAD.HI, IDP4A, I2F.F32.S32,
//        F2F.F16.U32 (the decode's cvt.rn.f16.u32), HFMA2, SHFL.
//        Method: 48 CTAs (one per SM) x 128 threads (4 warps, one per scheduler);
//        each warp executes ITERS x UNROLL independent warp-instructions with
//        clock64() bracketing; cycles/warp-instr = delta / (ITERS*UNROLL).
//        8 independent chains defeat dependency stalls; inline asm pins the op.
// X5(a): naive decode->fp16 (the "79-83 GB/s floor" row): decode every 16x16
//        trellis block to a gmem fp16 buffer, then CONSUME it with a reduction
//        kernel inside the timed region (kills the dirty-L2 ambiguity). Prints
//        block rate and all three GB/s conventions.
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define MUL1 0x83DCD12Du
#define CHK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

// ---------------------------------------------------------------- X4 harness
__device__ __forceinline__ unsigned long long sm_cycles() { return clock64(); }

template <int OP>
__global__ void alu_bench(unsigned* sink, int iters) {
    // one warp per scheduler: block 128 = 4 warps
    unsigned v0 = threadIdx.x + 1, v1 = 0x9E3779B9u ^ threadIdx.x,
             v2 = 0x85EBCA6Bu + threadIdx.x, v3 = 0xC2B2AE35u ^ (threadIdx.x << 3),
             v4 = 0x27D4EB2Fu + threadIdx.x, v5 = 0x165667B1u ^ threadIdx.x,
             v6 = 0xFD7046C5u + threadIdx.x, v7 = 0xA766F3ABu ^ threadIdx.x;
    // c0..c7: independent second inputs so each asm is a self-contained chain
    unsigned c0 = v1 ^ 0x1234u, c1 = v2 ^ 0x5678u, c2 = v3 ^ 0x9abcu, c3 = v4 ^ 0xdef0u,
             c4 = v5 ^ 0x0f1fu, c5 = v6 ^ 0x2e3du, c6 = v7 ^ 0x4c5bu, c7 = v0 ^ 0x6a79u;
    unsigned long long t0 = sm_cycles();
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            if (OP == 0) asm volatile("shf.r.wrap.b32 %0, %1, %1, 3;" : "+r"(v0) : "r"(c0)); // SHF
            else if (OP == 1) asm volatile("lop3.b32 %0, %1, %1, %2, 0xCA;" : "+r"(v0) : "r"(c1), "r"(c2)); // LOP3 (a&b)|c
            else if (OP == 2) asm volatile("prmt.b32 %0, %1, %2, 0x5410;" : "+r"(v0) : "r"(c3), "r"(c4)); // PRMT
            else if (OP == 3) asm volatile("mad.lo.u32 %0, %1, %2, %0;" : "+r"(v0) : "r"(c5), "r"(c6)); // IMAD low
            else if (OP == 4) asm volatile("mad.hi.u32 %0, %1, %2, %0;" : "+r"(v0) : "r"(c7), "r"(c0)); // IMAD.HI
            else if (OP == 5) asm volatile("dp4a.u32.u32 %0, %1, %2, %0;" : "+r"(v0) : "r"(c1), "r"(c2)); // dp4a
            else if (OP == 6) asm volatile("cvt.rn.f32.s32 %0, %1;" : "=r"(v0) : "r"(c3 | 1)); // I2F.F32.S32 (XU)
            else if (OP == 7) asm volatile("cvt.rn.f16.u32 %0, %1;" : "=r"(v0) : "r"(c4 | 1)); // F2F.F16.U32 (XU) — the decode cvt
            else if (OP == 8) asm volatile("fma.rn.f16x2 %0, %1, %2, %0;" : "+r"(v0) : "r"(c5), "r"(c6)); // HFMA2
            else if (OP == 9) v0 = __shfl_xor_sync(0xFFFFFFFFu, v0, 1, 32); // SHFL
        }
    }
    unsigned long long t1 = sm_cycles();
    if (threadIdx.x == 0) sink[blockIdx.x] = (unsigned)(v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7);
    if (threadIdx.x == 1) sink[blockIdx.x + 1024] = (unsigned)((t1 - t0) & 0xFFFFFFFFu);
}

__global__ void hmma_bench(unsigned* sink, int iters) {
    // 4 warps/SM; each warp: 8 independent f32 accumulator pairs, m16n8k16.
    unsigned a0 = threadIdx.x ^ 1, a1 = threadIdx.x ^ 2, a2 = threadIdx.x ^ 4, a3 = threadIdx.x ^ 8;
    unsigned b0 = threadIdx.x ^ 16, b1 = threadIdx.x ^ 32;
    float c0[4] = {0,0,0,0}, c1[4] = {0,0,0,0}, c2[4] = {0,0,0,0}, c3[4] = {0,0,0,0};
    unsigned long long t0 = sm_cycles();
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
        #define HMMA(C) asm volatile( \
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " \
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" \
            : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]) \
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))
            HMMA(c0); HMMA(c1); HMMA(c2); HMMA(c3);
        #undef HMMA
        }
    }
    unsigned long long t1 = sm_cycles();
    if (threadIdx.x == 0)
        sink[blockIdx.x] = c0[0]+c1[0]+c2[0]+c3[0];
    if (threadIdx.x == 1) sink[blockIdx.x + 1024] = (unsigned)((t1 - t0) & 0xFFFFFFFFu);
}

// ---------------------------------------------------------------- X5a: naive decode->fp16
// Mirrors exl3_dq_from (kernels/exl3_bench.cu) bit-exactly; grid-stride over
// (k-blocks x n-blocks) of ONE gate_up-shaped expert, M irrelevant (pure decode).
__device__ __forceinline__ uint16_t naive_dq(const uint32_t* __restrict__ ring,
                                             int i0m, int i1m, int sf) {
    const unsigned lo = ring[i1m], hi = ring[i0m];
    const unsigned idx = __funnelshift_r(lo, hi, (unsigned)sf) & 0xFFFFu;
    const unsigned sum = __dp4a(idx * MUL1, 0x01010101u, 0x00000400u);
    const __half h = __uint2half_rn(sum);
    const __half w16 = __hfma(h, __ushort_as_half((unsigned short)0x1EEE),
                                  __ushort_as_half((unsigned short)0xC931));
    return (uint16_t)__half_as_ushort(w16);
}

// 32 threads per (k-block, n-block) pair? — mirror the dump gate: one WARP per block,
// lane l decodes temporal slots t = l and t = l + 32 (16x16 block = 256 elems, 8/lane).
__global__ void naive_decode_fp16(const uint16_t* __restrict__ trellis,
                                  __half* __restrict__ out,
                                  int kb_n, int nb, int bits) {
    const int W = bits * 8;
    long long blk = (long long)blockIdx.x * blockDim.x + threadIdx.x;   // one warp = 32 lanes
    long long nblocks = (long long)kb_n;                                 // TOTAL blocks (all experts, linear)
    (void)nb;
    int lane = threadIdx.x & 31;
    for (; blk / 32 < nblocks; blk += (long long)gridDim.x * blockDim.x) {
        long long b = blk >> 5;
        const uint32_t* ring = (const uint32_t*)(trellis + b * (16 * bits)); // concatenated experts, linear
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            int t = lane + 32 * s;                    // temporal index in block
            int b1 = (t + 257) * bits, i0 = (b1 - 16) >> 5, i1 = (b1 - 1) >> 5;
            uint16_t v = naive_dq(ring, i0 % W, i1 % W, ((i1 + 1) << 5) - b1);
            // scatter to fp16 [256] with SOME placement — identity is fine for bandwidth;
            // consumer only needs to touch the bytes.
            out[((size_t)b << 8) + ((t & 15) << 4) + (t >> 4)] = __ushort_as_half(v);
        }
    }
}

__global__ void reduce_sum(const __half* __restrict__ y, float* __restrict__ sink, long long n) {
    float s = 0.f;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) s += __half2float(y[i]);
    // deterministic single-level atomic is fine for a bandwidth probe
    if (s == 1.0e38f) sink[0] = s; // never true; defeats any store elimination
    atomicAdd(sink, s);
}

static double now_s() { return clock() / (double)CLOCKS_PER_SEC; }

int main(int argc, char** argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 2000;
    // ---------------- X4 ----------------
    {
        unsigned* sink; CHK(cudaMalloc(&sink, 2048 * sizeof(unsigned)));
        CHK(cudaMemset(sink, 0, 2048 * sizeof(unsigned)));
        printf("=== X4 pipe rates (48 SMs x 4 warps, %d iters x 8 ops; cycles/warp-instr) ===\n", iters);
        const char* names[10] = {"SHF.R","LOP3","PRMT","IMAD","IMAD.HI","IDP4A","I2F.F32.S32","F2F.F16.U32","HFMA2","SHFL"};
        for (int op = 0; op < 10; ++op) {
            #define RUN_OP(O) if (op == O) { \
                alu_bench<O><<<48, 128>>>(sink, 100); CHK(cudaDeviceSynchronize()); \
                CHK(cudaMemset(sink + 1024, 0, 48 * sizeof(unsigned))); \
                alu_bench<O><<<48, 128>>>(sink, iters); CHK(cudaDeviceSynchronize()); }
            RUN_OP(0) RUN_OP(1) RUN_OP(2) RUN_OP(3) RUN_OP(4)
            RUN_OP(5) RUN_OP(6) RUN_OP(7) RUN_OP(8) RUN_OP(9)
            unsigned cyc[48]; CHK(cudaMemcpy(cyc, sink + 1024, 48 * sizeof(unsigned), cudaMemcpyDeviceToHost));
            unsigned long long tot = 0; for (int b = 0; b < 48; ++b) tot += cyc[b];
            double per = (double)tot / 48 / iters / 8;
            printf("  %-12s %6.3f cycles/warp-instr   (lane-ops/clk/SM = %5.1f)\n",
                   names[op], per, 4 * 32.0 / per);
        }
        // HMMA: 8 acc-chains x 8 unroll = 64 independent-ish MMAs per iter per warp
        unsigned* fsink; CHK(cudaMalloc(&fsink, 2048 * sizeof(unsigned)));
        CHK(cudaMemset(fsink, 0, 2048 * sizeof(unsigned)));
        hmma_bench<<<48, 128>>>(fsink, 100); CHK(cudaDeviceSynchronize());
        CHK(cudaMemset(fsink + 1024, 0, 48 * sizeof(float)));
        hmma_bench<<<48, 128>>>(fsink, iters); CHK(cudaDeviceSynchronize());
        unsigned cyc[48]; CHK(cudaMemcpy(cyc, fsink + 1024, 48 * sizeof(unsigned), cudaMemcpyDeviceToHost));
        unsigned long long tot = 0; for (int b = 0; b < 48; ++b) tot += cyc[b];
        double per = (double)tot / 48 / iters / 16;   // cycles per HMMA (4 chains x 4 unroll per warp)
        printf("  %-12s %6.3f cycles/warp-instr   => %.1f HMMA/clk/SM, %.0f FMA/clk/SM (2048 FMA/HMMA)\n",
               "HMMA.16816", per, 4.0 / per, 4.0 * 2048.0 / per);
        float fs; CHK(cudaMemcpy(&fs, fsink, sizeof(float), cudaMemcpyDeviceToHost));
        printf("  (sink %.1f)\n", fs);
    }
    // ---------------- X5a ----------------
    {
        // gate_up expert shape: K=2560 (160 k-blocks) x N=640 (40 n-blocks), b3.
        // Pack the SAME expert 512x (streaming footprint like the grouped bench).
        const int KB = 160, NB = 40, BITS = 3; const int E = argc > 2 ? atoi(argv[2]) : 512;
        const size_t per_expert_words = (size_t)KB * NB * (16 * BITS); // uint16 count (48/block at b3)
        const size_t per_expert_u32 = (size_t)KB * NB * (16 * BITS) / 4; // 24 words/block
        std::vector<unsigned short> trellis(per_expert_words * E);
        srand(7);
        for (size_t i = 0; i < trellis.size(); ++i) trellis[i] = (unsigned short)(rand() & 0xFFFF);
        unsigned short* d_tr; CHK(cudaMalloc(&d_tr, trellis.size() * 2));
        CHK(cudaMemcpy(d_tr, trellis.data(), trellis.size() * 2, cudaMemcpyHostToDevice));
        const long long nblocks = (long long)KB * NB * E;
        __half* d_out; CHK(cudaMalloc(&d_out, nblocks * 256 * sizeof(__half)));   // 300 MB for 512E? KB*NB*256*2*E
        float* d_sink; CHK(cudaMalloc(&d_sink, 4));
        CHK(cudaMemset(d_sink, 0, 4));
        printf("=== X5a naive decode->fp16 (512 x gate_up b3, %.0f MB trellis in, %.0f MB fp16 out) ===\n",
               trellis.size() / 2 / 1e6, nblocks * 256 * 2 / 1e6);
        cudaEvent_t e0, e1; CHK(cudaEventCreate(&e0)); CHK(cudaEventCreate(&e1));
        int grid = 8192, block = 256;
        int rgrid = 4096, rblock = 256;
        // warm
        naive_decode_fp16<<<grid, block>>>(d_tr, d_out, (long long)KB * NB * E, NB, BITS); CHK(cudaDeviceSynchronize());
        reduce_sum<<<rgrid, rblock>>>(d_out, d_sink, nblocks * 256); CHK(cudaDeviceSynchronize());
        const int reps = 3;
        double best = 1e30; float chk = 0;
        for (int r = 0; r < reps; ++r) {
            CHK(cudaMemset(d_sink, 0, 4));
            CHK(cudaEventRecord(e0));
            for (int i = 0; i < 3; ++i) {
                naive_decode_fp16<<<grid, block>>>(d_tr, d_out, (long long)KB * NB * E, NB, BITS);
                reduce_sum<<<rgrid, rblock>>>(d_out, d_sink, nblocks * 256);
            }
            CHK(cudaEventRecord(e1)); CHK(cudaEventSynchronize(e1));
            float ms; CHK(cudaEventElapsedTime(&ms, e0, e1));
            best = best < ms / 3 ? best : ms / 3;
            CHK(cudaMemcpy(&chk, d_sink, 4, cudaMemcpyDeviceToHost));
        }
        double bps = nblocks / (best / 1e3);
        printf("  kernel+consume best %.3f ms  block rate %.3f G blocks/s\n", best, bps / 1e9);
        printf("  GB/s conventions: trellis-in %.1f | trellis-in+fp16-out %.1f | fp16-out-only %.1f   (checksum %.1f)\n",
               nblocks * 96.0 / best / 1e6, nblocks * (96.0 + 512.0) / best / 1e6,
               nblocks * 512.0 / best / 1e6, chk);
        CHK(cudaFree(d_tr)); CHK(cudaFree(d_out)); CHK(cudaFree(d_sink));
    }
    printf("MICRO: done\n");
    return 0;
}
