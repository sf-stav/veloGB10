// 09_l2_paced.cuh (leg PD1, included by 18_exl3_dense_dec.cuh): the PACED L2 PREFETCH of the weights the next kernels of a
// decode / verify round read. One CTA of 128 threads on its own stream, forked by an event after a streaming kernel and
// joined to the main stream only at the end of the round (the forward graph's end), so DRAM keeps working through the
// exchanges, mixes, router and attention that follow, and the next latency-bound linears read part of their bytes from L2.
// Mechanism as the external reference code specifies it (PLAN/DSV41_RX4_REPORT.md C.1 / C.2):
//   - wait `delay_ns` after the start (the kernel launched beside it first runs undisturbed);
//   - the ranges (<= 16, in the order the next kernels read them, already cut to the launch's budget by the host) are
//     walked in 8 KiB chunks (16-byte floor of each run). A range is `count` runs of `run` bytes, `stride` bytes apart (a
//     contiguous weight is one run; a column slice of a whole-resident trellis is one run per k16 step: TP4 attention);
//   - waves of 128 chunks (1 MiB): wave w starts when globaltimer - t0 >= w * wave_ns (thread 0 polls with a 256 ns sleep),
//     then each thread issues ONE cp.async.bulk.prefetch.L2 of its chunk. 1 MiB every wave_ns (4,559 ns = 230 GB/s by
//     default: a little under what DRAM serves, so DRAM's queue holds about one wave and the latency-bound kernels beside
//     it keep their load latency);
//   - a plain L2 prefetch: no eviction hint, nothing written, no PDL (it is nobody's predecessor).
// The range table is read from the __grid_constant__ parameter (constant bank): no per-thread array, no stack.
#pragma once

#define D41X_PF_MAXR 16
#define D41X_PF_THREADS 128
#define D41X_PF_CHUNK 8192

struct D41xL2PacedArgs {
    const unsigned char* ptr[D41X_PF_MAXR];
    unsigned long long run[D41X_PF_MAXR];     // bytes of one run (floored to 16 B here)
    unsigned long long stride[D41X_PF_MAXR];  // bytes between runs
    unsigned long long count[D41X_PF_MAXR];   // runs (already cut to the launch's budget by the host)
    int n, pad_;
    unsigned long long delay_ns, wave_ns;
};

__device__ __forceinline__ unsigned long long d41x_gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

extern "C" __global__ void __launch_bounds__(D41X_PF_THREADS, 1) d41x_l2_paced(const __grid_constant__ D41xL2PacedArgs a) {
    __shared__ unsigned long long s_t0;
    const int t = threadIdx.x;
    if (t == 0) {
        const unsigned long long ts = d41x_gtimer();
        unsigned long long now = ts;
        while (now - ts < a.delay_ns) { __nanosleep(500); now = d41x_gtimer(); }
        s_t0 = now;
    }
    __syncthreads();
    const unsigned long long t0 = s_t0;
    // total chunks
    unsigned long long total = 0;
    for (int r = 0; r < a.n; ++r) total += ((a.run[r] & ~15ull) + D41X_PF_CHUNK - 1) / D41X_PF_CHUNK * a.count[r];
    for (unsigned long long w = 0; w * D41X_PF_THREADS < total; ++w) {
        __syncthreads();
        if (t == 0) { while (d41x_gtimer() - t0 < w * a.wave_ns) __nanosleep(256); }
        __syncthreads();
        unsigned long long c = w * D41X_PF_THREADS + t;
        if (c >= total) continue;
        int r = 0;
        unsigned long long cpr = 1;
        for (; r < a.n; ++r) {
            cpr = ((a.run[r] & ~15ull) + D41X_PF_CHUNK - 1) / D41X_PF_CHUNK;
            const unsigned long long nc = cpr * a.count[r];
            if (c < nc) break;
            c -= nc;
        }
        const unsigned long long k = c / cpr, off = (c - k * cpr) * D41X_PF_CHUNK, len = a.run[r] & ~15ull;
        const unsigned sz = (unsigned)(len - off < D41X_PF_CHUNK ? len - off : D41X_PF_CHUNK);
        const unsigned char* src = a.ptr[r] + k * a.stride[r] + off;
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(src), "r"(sz) : "memory");
    }
}
