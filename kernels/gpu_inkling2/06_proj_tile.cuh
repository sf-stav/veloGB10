// gpu_inkling2/06_proj_tile.cuh — the prefill-width BF16 projection GEMM, shared-memory tiled (leg S4). Section of
// gpu_inkling2.cu.
//
// Same contract and the SAME per-output arithmetic as ink_proj_wide (05): Y[M][ldy] fp32 = X[M][K] bf16 . W^T, W in
// the B32 fragment layout, up to three activation segments split at 16-row weight-tile boundaries t1 / t2. Every
// output is accumulated by one warp over the 32-k groups g = 0, 1, ... in order, per group the two m16n8k16 MMAs in
// the order of ika_bf_group, with the same register contents (A: row r, the 16 B at k 32g + 8tig; B: the lane's
// 16 B of the 1-KB group block). Only the staging differs (cp.async into shared memory, each operand fetched once per
// CTA instead of once per warp), so the outputs are bit-identical to ink_proj_wide.
//
// CTA: 128 token rows x 8 weight tiles (128 outputs), 256 threads = 8 warps as 2 (rows) x 4 (tiles): a warp owns
// 64 rows (4 m16) x 2 tiles. A CTA's tiles never cross a segment boundary: blockIdx.y enumerates 8-tile blocks
// segment by segment. Pipeline: IKP_ST stages of one 32-k group (A 128 x 64 B, padded rows; B 8 x 1 KB).
// grid (ceil(M / 128), nblk(t1, t2, ntiles)), block 256, dynamic smem IKP_ST x (IKP_ASZ + IKP_BSZ).

#define IKP_AROW 80                     // A smem row stride in bytes (64 B data + 16 pad: conflict-free 16-B reads)
#define IKP_ASZ (128 * IKP_AROW)        // 10 KB
#define IKP_BSZ (8 * 1024)              // 8 KB


template <int IKP_ST>
__device__ __forceinline__ void ika_proj_tile(const uint16_t* __restrict__ W,
        const uint16_t* __restrict__ X0, const uint16_t* __restrict__ X1, const uint16_t* __restrict__ X2,
        int t1, int t2, float* __restrict__ Y, int ldy, int M, int K, int ntiles) {
    extern __shared__ __align__(16) unsigned char smp[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    t1 = min(t1, ntiles); t2 = min(max(t2, t1), ntiles);
    // block -> (segment, first tile)
    const int b1 = (t1 + 7) / 8, b2 = b1 + (t2 - t1 + 7) / 8;
    const int bx = blockIdx.y;   // M-blocks run fastest: the CTAs sharing a weight block run together (one DRAM read)
    int tile0, tend;
    const uint16_t* X;
    if (bx < b1) { tile0 = 8 * bx; tend = t1; X = X0; }
    else if (bx < b2) { tile0 = t1 + 8 * (bx - b1); tend = t2; X = X1; }
    else { tile0 = t2 + 8 * (bx - b2); tend = ntiles; X = X2; }
    const int nt = min(8, tend - tile0);
    const int r0 = blockIdx.x * 128;
    const int G = K >> 5;

    auto stage_a = [&](int s) { return smp + s * (IKP_ASZ + IKP_BSZ); };
    auto stage_b = [&](int s) { return smp + s * (IKP_ASZ + IKP_BSZ) + IKP_ASZ; };
    auto load = [&](int g, int s) {
        unsigned char* sa = stage_a(s);
        unsigned char* sb = stage_b(s);
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int c = threadIdx.x + 256 * i;          // 512 chunks of 16 B: row c / 4, part c % 4
            const int r = c >> 2, part = c & 3;
            const int row = min(r0 + r, M - 1);
            ika_cp16(sa + r * IKP_AROW + part * 16, X + (size_t)row * K + 32 * g + 8 * part, true);
            const int tl = c >> 6, off = c & 63;           // 8 tiles x 64 chunks
            const bool v = tl < nt;
            const uint16_t* src = W + ((size_t)(tile0 + (v ? tl : 0)) * G + g) * 512 + off * 8;
            ika_cp16(sb + tl * 1024 + off * 16, src, v);
        }
    };

    const int wm = warp >> 2, wn = warp & 3;               // rows wm*64.., tiles 2wn, 2wn+1
    float acc[4][2][2][4];
    #pragma unroll
    for (int a = 0; a < 4; ++a)
        #pragma unroll
        for (int b = 0; b < 2; ++b)
            #pragma unroll
            for (int h = 0; h < 2; ++h)
                #pragma unroll
                for (int r = 0; r < 4; ++r) acc[a][b][h][r] = 0.f;

    #pragma unroll
    for (int s = 0; s < IKP_ST - 1; ++s) {
        if (s < G) load(s, s);
        asm volatile("cp.async.commit_group;" ::: "memory");
    }
    for (int g = 0; g < G; ++g) {
        asm volatile("cp.async.wait_group %0;" :: "n"(IKP_ST - 2) : "memory");
        __syncthreads();
        {   // prefetch group g + ST - 1 into the stage freed by group g - 1
            const int gn = g + IKP_ST - 1;
            if (gn < G) load(gn, gn % IKP_ST);
            asm volatile("cp.async.commit_group;" ::: "memory");
        }
        const unsigned char* sa = stage_a(g % IKP_ST);
        const unsigned char* sb = stage_b(g % IKP_ST);
        uint4 bw[2][2];
        #pragma unroll
        for (int b = 0; b < 2; ++b)
            #pragma unroll
            for (int h = 0; h < 2; ++h)
                bw[b][h] = *reinterpret_cast<const uint4*>(sb + (2 * wn + b) * 1024 + h * 512 + lane * 16);
        #pragma unroll
        for (int a = 0; a < 4; ++a) {
            const int ra = wm * 64 + 16 * a + gid;
            const uint4 xa = *reinterpret_cast<const uint4*>(sa + ra * IKP_AROW + tig * 16);
            const uint4 xb = *reinterpret_cast<const uint4*>(sa + (ra + 8) * IKP_AROW + tig * 16);
            #pragma unroll
            for (int b = 0; b < 2; ++b) {
                ika_bf_group(acc[a][b][0], xa, xb, bw[b][0]);
                ika_bf_group(acc[a][b][1], xa, xb, bw[b][1]);
            }
        }
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    #pragma unroll
    for (int b = 0; b < 2; ++b) {
        const int tl = 2 * wn + b;
        if (tl >= nt) continue;
        const int tile = tile0 + tl;
        #pragma unroll
        for (int a = 0; a < 4; ++a) {
            const int ta = r0 + wm * 64 + 16 * a + gid, tb = ta + 8;
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int n = tile * 16 + 8 * h + 2 * tig;
                if (ta < M) *reinterpret_cast<float2*>(&Y[(size_t)ta * ldy + n]) = make_float2(acc[a][b][h][0], acc[a][b][h][1]);
                if (tb < M) *reinterpret_cast<float2*>(&Y[(size_t)tb * ldy + n]) = make_float2(acc[a][b][h][2], acc[a][b][h][3]);
            }
        }
    }
}
// IKP_ST stages: 4 = 72 KB smem (1 CTA / SM), 2 = 36 KB (2 CTAs / SM)
extern "C" __global__ __launch_bounds__(256) void ink_proj_tile(const uint16_t* __restrict__ W,
        const uint16_t* __restrict__ X0, const uint16_t* __restrict__ X1, const uint16_t* __restrict__ X2,
        int t1, int t2, float* __restrict__ Y, int ldy, int M, int K, int ntiles) { INK2_PDL_ENTRY(); ika_proj_tile<4>(W, X0, X1, X2, t1, t2, Y, ldy, M, K, ntiles); }
extern "C" __global__ __launch_bounds__(256, 2) void ink_proj_tile2(const uint16_t* __restrict__ W,
        const uint16_t* __restrict__ X0, const uint16_t* __restrict__ X1, const uint16_t* __restrict__ X2,
        int t1, int t2, float* __restrict__ Y, int ldy, int M, int K, int ntiles) { INK2_PDL_ENTRY(); ika_proj_tile<2>(W, X0, X1, X2, t1, t2, Y, ldy, M, K, ntiles); }
extern "C" __global__ __launch_bounds__(256, 2) void ink_proj_tile3(const uint16_t* __restrict__ W,
        const uint16_t* __restrict__ X0, const uint16_t* __restrict__ X1, const uint16_t* __restrict__ X2,
        int t1, int t2, float* __restrict__ Y, int ldy, int M, int K, int ntiles) { INK2_PDL_ENTRY(); ika_proj_tile<3>(W, X0, X1, X2, t1, t2, Y, ldy, M, K, ntiles); }
