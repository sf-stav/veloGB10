// gpu_inkling2/20_conv.cuh — static short causal conv (k = 4) + residual, conv-ring store (K1).
// Section of gpu_inkling2.cu.
//
// conv(x)_p = sum_{lag=0..3} w[lag][c] * x_{p-lag} (x_{<0} = 0), out = (conv + x_p) / sqrt(5).
// Taps are stored [4 lag][C] for every conv (lag 0 = CURRENT token); the checkpoint's static [C][1][4]
// (tap 3 = current, D3) is reversed at load, the dynamic `static_w [4][C]` is already lag-ordered.
//
// STATE + SPECULATIVE COMMIT. Each (logical slot, conv) keeps its conv INPUTS in a ring of R = 32 fp32 rows,
// row = pos & (R - 1). A call on rows p0..p0+M-1 reads x_{p-lag} from the call's own input rows when
// p - lag >= p0 and from the ring otherwise, and (M <= R - 3) writes its M inputs to ring rows p0..p0+M-1
// in the same kernel (the rows it reads are p0-3..p0-1: disjoint). A verify of M draft rows therefore
// leaves the ring valid for EVERY prefix: accepting a rows = setting the sequence position to p0 + a.
// No snapshot, no copy, no rollback kernel; rejected rows are overwritten by the next call before any
// read (reads only reach pos - 3). Prefill chunks (M > R - 3) skip the in-kernel write and run
// ink_ring_store afterwards (the last min(M, R) inputs). The same position rule drives the K/V cache,
// so one `pos` per sequence is the whole commit state (48 slots x 4 convs x 32 x C x 4 B = 41 MB / seq).

// grid (ceil(C / 256), M), block 256. base (nullable) [M][C] is added: out = base + conv-out.
// Body shared with the lane-batched decode entry (85_lanes.cuh, t = 0 on per-row pointers).
__device__ __forceinline__ void ika_sconv_body(const float* __restrict__ x, int ldx,
        float* __restrict__ ring, int ring_mask, int ring_write, const float* __restrict__ w,
        const float* __restrict__ base, float* __restrict__ out, int C, int p0, float div, const int t) {
    const int c = blockIdx.x * 256 + threadIdx.x, p = p0 + t;
    if (c >= C) return;
    const float x0 = x[(size_t)t * ldx + c];
    float acc = 0.f;
    #pragma unroll
    for (int lag = 0; lag < 4; ++lag) {
        float xv;
        if (lag == 0) xv = x0;
        else if (t - lag >= 0) xv = x[(size_t)(t - lag) * ldx + c];
        else if (p - lag >= 0) xv = ring[(size_t)((p - lag) & ring_mask) * C + c];
        else xv = 0.f;
        acc = fmaf(xv, w[lag * C + c], acc);
    }
    const float y = (acc + x0) * div;
    out[(size_t)t * C + c] = base ? base[(size_t)t * C + c] + y : y;
    if (ring_write) ring[(size_t)(p & ring_mask) * C + c] = x0;
}
extern "C" __global__ __launch_bounds__(256) void ink_sconv(const float* __restrict__ x, int ldx,
        float* __restrict__ ring, int ring_mask, int ring_write, const float* __restrict__ w,
        const float* __restrict__ base, float* __restrict__ out, int C, int p0, float div) {
    INK2_PDL_ENTRY();
    ika_sconv_body(x, ldx, ring, ring_mask, ring_write, w, base, out, C, p0, div, blockIdx.y);
}

// grid (ceil(C / 256), rows), block 256: ring[(p0 + M - rows + i) & mask] = x[M - rows + i] (the call's last
// `rows` inputs, rows = min(M, R)); x columns at `off` with row stride ldx.
extern "C" __global__ __launch_bounds__(256) void ink_ring_store(const float* __restrict__ x, int ldx, int off,
        float* __restrict__ ring, int ring_mask, int C, int p0, int M, int rows) {
    INK2_PDL_ENTRY();
    const int c = blockIdx.x * 256 + threadIdx.x, i = blockIdx.y;
    if (c >= C) return;
    const int t = M - rows + i;
    ring[(size_t)((p0 + t) & ring_mask) * C + c] = x[(size_t)t * ldx + off + c];
}
