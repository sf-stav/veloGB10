// gpu_inkling2/63_experts_prefill.cuh: routed experts at PREFILL widths — device-routed grouped GEMM over experts.
//   ink2_moe_pf_route (1 CTA): per-expert counts, offsets, the compact row order (expert-major, token ascending
//     inside an expert: deterministic), rowtok[compact] = token, rowidx[t][k] = compact row of pick k of token t,
//     and the tile list tiles[2i] = expert, tiles[2i+1] = first compact row (within the expert) of a 64-row block.
//   GU / DN tiles: grid (N/128, max_tiles), block 256 (8 warps x one 16-row tile = 128 weight rows) x 64 compact rows.
//     NVFP4: smem-staged (the fp4l xf_nv4_tc structure: activations staged once per CTA, 2 cp.async stages, 48 KB);
//     MXFP8: the same smem-staged shape over F64 64-k blocks.
// Per 64-k step a row's MMA sequence and the epilogue are those of the decode kernels (62): a prefill row is
// bit-identical to the same token's decode row (gated by --probe-inkling2-moe).
// Tensor cores: W4A16 / W8A16 f16 mma.sync with the weight dequantised exactly into the B fragment. The sm_121a
// block-scaled fp4 / fp8 MMA (kind::mxf4nvf4 / f8f6f4) needs the ACTIVATIONS in fp4 / fp8 too, i.e. an activation
// quantisation the reference does not perform (it dequantises the weights to bf16): not used (owner decision).

// ------------------------------------------------------------------------------------------------
// ink2_moe_pf_route — grid E, block 256: CTA e owns expert e. ids [M][8] (ink2_router_topk, build_slots = 0). E == 128.
//   Every CTA builds the full histogram (integer smem counts: order-free), so offsets and the tile-list base are
//   derived without a second launch; then it places its own expert's pairs in ascending pair order (block scan).
//   cnt / offs [E], rowtok [8M], rowidx [8M], tiles [2 * max_tiles], ntiles [1]. max_tiles = 8M/64 + E.
// ------------------------------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256) void ink2_moe_pf_route(
        const int* __restrict__ ids, int R, int* __restrict__ cnt, int* __restrict__ offs, int* __restrict__ rowtok,
        int* __restrict__ rowidx, int* __restrict__ tiles, int* __restrict__ ntiles) {
    INK2_PDL_ENTRY();
    constexpr int E = 128, TOPK = 8, ROWS = 64;
    __shared__ int s_cnt[E];
    __shared__ int s_base[2];   // offs[e], tile base of e
    __shared__ int s_wsum[8];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int e = blockIdx.x;
    if (tid < E) s_cnt[tid] = 0;
    __syncthreads();
    for (int p = tid; p < R; p += blockDim.x) atomicAdd(&s_cnt[__ldg(&ids[p])], 1);
    __syncthreads();
    if (tid == 0) {
        int off = 0, tb = 0;
        for (int f = 0; f < e; ++f) { off += s_cnt[f]; tb += (s_cnt[f] + ROWS - 1) / ROWS; }
        s_base[0] = off; s_base[1] = tb;
        cnt[e] = s_cnt[e]; offs[e] = off;
        const int c = s_cnt[e];
        for (int r0 = 0, i = tb; r0 < c; r0 += ROWS, ++i) { tiles[2 * i] = e; tiles[2 * i + 1] = r0; }
        if (e == E - 1) *ntiles = tb + (c + ROWS - 1) / ROWS;
    }
    __syncthreads();
    int run = s_base[0];
    const unsigned lt = (1u << lane) - 1u;
    for (int base = 0; base < R; base += 256) {
        const int p = base + tid;
        const bool hit = p < R && __ldg(&ids[p]) == e;
        const unsigned m = __ballot_sync(0xffffffffu, hit);
        if (lane == 0) s_wsum[warp] = __popc(m);
        __syncthreads();
        int before = 0, tot = 0;
        #pragma unroll
        for (int w = 0; w < 8; ++w) { const int v = s_wsum[w]; before += w < warp ? v : 0; tot += v; }
        if (hit) {
            const int pos = run + before + __popc(m & lt);
            rowtok[pos] = p / TOPK;
            rowidx[p] = pos;
        }
        run += tot;
        __syncthreads();
    }
}

__device__ __forceinline__ void ik2m_cp16(void* smem, const void* gmem) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void ik2m_cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void ik2m_cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }

// ------------------------------------------------------------------------------------------------
// NVFP4 prefill tiles (smem-staged). GU: X = Xh [tokens][K] read through rowmap = rowtok; out Din [R][N/2].
// DN: X = Din [R][K] (rowmap nullptr); out Y [R][N] fp32. K % 128 == 0, N % 128 == 0.
// ------------------------------------------------------------------------------------------------
template <bool GU>
__device__ __forceinline__ void ik2m_nv4_pf_body(
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, const float* __restrict__ GS,
        const __half* __restrict__ X, const int* __restrict__ rowmap, __half* __restrict__ Din, float* __restrict__ Yf,
        const int* __restrict__ tiles, const int* __restrict__ ntiles, const int* __restrict__ offs,
        const int* __restrict__ cnt, int K, int N) {
    __shared__ __align__(128) uint8_t smA[2][64 * 256];
    __shared__ __align__(128) uint8_t smB[2][8 * 1024];
    const int tile = blockIdx.y;
    if (tile >= *ntiles) return;
    const int e = tiles[2 * tile], r0 = tiles[2 * tile + 1];
    const int rows = min(64, cnt[e] - r0);
    const int base = offs[e] + r0;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
    const int n0c = blockIdx.x * 8;   // the CTA's first 16-row tile
    const int KB = K >> 7;
    const size_t tw_stride = (size_t)KB * 1024 + ((K & 64) ? 512 : 0);
    const size_t ts_stride = (size_t)KB * 128 + ((K & 64) ? 64 : 0);
    const uint8_t* We = W + (size_t)e * N * (K >> 1);
    const uint8_t* Se = S + (size_t)e * N * (K >> 4);
    const uint8_t* stp = Se + (size_t)(n0c + warp) * ts_stride + lane * 4;
    const float gs = GS[2 * e + (GU ? 0 : 1)];
    auto load = [&](int st, int b) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = tid + 256 * j, row = c >> 4, ch = c & 15;
            const int rr = base + min(row, rows - 1);
            const int src = rowmap ? rowmap[rr] : rr;
            ik2m_cp16(&smA[st][row * 256 + ((ch ^ (row & 7)) << 4)], X + (size_t)src * K + b * 128 + ch * 8);
        }
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int c = tid + 256 * j, tl = c >> 6, o = (c & 63) << 4;
            ik2m_cp16(&smB[st][tl * 1024 + o], We + (size_t)(n0c + tl) * tw_stride + (size_t)b * 1024 + o);
        }
    };
    float acc[4][2][4];
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc[mt][h][r] = 0.f;
    if (KB > 0) {
        load(0, 0);
        ik2m_cp_commit();
    }
    uint32_t sc = (KB > 0) ? __ldg(reinterpret_cast<const uint32_t*>(stp)) : 0u;
    for (int b = 0; b < KB; ++b) {
        const int st = b & 1;
        if (b + 1 < KB) load(st ^ 1, b + 1);
        ik2m_cp_commit();
        const uint32_t nsc = (b + 1 < KB) ? __ldg(reinterpret_cast<const uint32_t*>(stp + (size_t)(b + 1) * 128)) : sc;
        ik2m_cp_wait<1>();
        __syncthreads();
        const uint4 w0 = *reinterpret_cast<const uint4*>(&smB[st][warp * 1024 + lane * 16]);
        const uint4 w1 = *reinterpret_cast<const uint4*>(&smB[st][warp * 1024 + 512 + lane * 16]);
        #pragma unroll
        for (int s2 = 0; s2 < 2; ++s2) {
            uint32_t b0[8], b1[8];
            ik2m_e2m1x8(s2 ? w0.z : w0.x, &b0[0]); ik2m_e2m1x8(s2 ? w0.w : w0.y, &b0[4]);
            ik2m_e2m1x8(s2 ? w1.z : w1.x, &b1[0]); ik2m_e2m1x8(s2 ? w1.w : w1.y, &b1[4]);
            const uint32_t f0 = ik2m_e4m3_bcast((sc >> (8 * s2)) & 0xFF), f1 = ik2m_e4m3_bcast((sc >> (16 + 8 * s2)) & 0xFF);
            #pragma unroll
            for (int j = 0; j < 8; ++j) { b0[j] = ik2m_hmul2(b0[j], f0); b1[j] = ik2m_hmul2(b1[j], f1); }
            const int c0 = s2 * 8 + 2 * tig;
            #pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                const int ra = mt * 16 + gid, rb = ra + 8;
                const uint4 xa0 = *reinterpret_cast<const uint4*>(&smA[st][ra * 256 + ((c0 ^ (ra & 7)) << 4)]);
                const uint4 xa1 = *reinterpret_cast<const uint4*>(&smA[st][ra * 256 + (((c0 + 1) ^ (ra & 7)) << 4)]);
                const uint4 xb0 = *reinterpret_cast<const uint4*>(&smA[st][rb * 256 + ((c0 ^ (rb & 7)) << 4)]);
                const uint4 xb1 = *reinterpret_cast<const uint4*>(&smA[st][rb * 256 + (((c0 + 1) ^ (rb & 7)) << 4)]);
                ik2m_mma(acc[mt][0], xa0.x, xb0.x, xa0.y, xb0.y, b0[0], b0[1]);
                ik2m_mma(acc[mt][0], xa0.z, xb0.z, xa0.w, xb0.w, b0[2], b0[3]);
                ik2m_mma(acc[mt][0], xa1.x, xb1.x, xa1.y, xb1.y, b0[4], b0[5]);
                ik2m_mma(acc[mt][0], xa1.z, xb1.z, xa1.w, xb1.w, b0[6], b0[7]);
                ik2m_mma(acc[mt][1], xa0.x, xb0.x, xa0.y, xb0.y, b1[0], b1[1]);
                ik2m_mma(acc[mt][1], xa0.z, xb0.z, xa0.w, xb0.w, b1[2], b1[3]);
                ik2m_mma(acc[mt][1], xa1.x, xb1.x, xa1.y, xb1.y, b1[4], b1[5]);
                ik2m_mma(acc[mt][1], xa1.z, xb1.z, xa1.w, xb1.w, b1[6], b1[7]);
            }
        }
        sc = nsc;
        __syncthreads();
    }
    if constexpr (!GU) {
        if (K & 64) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int c = tid + 256 * j, row = c >> 3, ch = c & 7;
                const int rr = base + min(row, rows - 1);
                const int src = rowmap ? rowmap[rr] : rr;
                ik2m_cp16(&smA[0][row * 256 + ((ch ^ (row & 7)) << 4)], X + (size_t)src * K + KB * 128 + ch * 8);
            }
            ik2m_cp_commit();
            ik2m_cp_wait<0>();
            __syncthreads();
            const uint8_t* wh = We + (size_t)(n0c + warp) * tw_stride + (size_t)KB * 1024;
            const uint2 w0h = __ldg(reinterpret_cast<const uint2*>(wh + lane * 8));
            const uint2 w1h = __ldg(reinterpret_cast<const uint2*>(wh + 256 + lane * 8));
            const uint16_t sch = __ldg(reinterpret_cast<const uint16_t*>(Se + (size_t)(n0c + warp) * ts_stride + (size_t)KB * 128 + lane * 2));
            uint32_t b0[8], b1[8];
            ik2m_e2m1x8(w0h.x, &b0[0]); ik2m_e2m1x8(w0h.y, &b0[4]);
            ik2m_e2m1x8(w1h.x, &b1[0]); ik2m_e2m1x8(w1h.y, &b1[4]);
            const uint32_t f0 = ik2m_e4m3_bcast(sch & 0xFF), f1 = ik2m_e4m3_bcast(sch >> 8);
            #pragma unroll
            for (int j = 0; j < 8; ++j) { b0[j] = ik2m_hmul2(b0[j], f0); b1[j] = ik2m_hmul2(b1[j], f1); }
            const int c0 = 2 * tig;
            #pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                const int ra = mt * 16 + gid, rb = ra + 8;
                const uint4 xa0 = *reinterpret_cast<const uint4*>(&smA[0][ra * 256 + ((c0 ^ (ra & 7)) << 4)]);
                const uint4 xa1 = *reinterpret_cast<const uint4*>(&smA[0][ra * 256 + (((c0 + 1) ^ (ra & 7)) << 4)]);
                const uint4 xb0 = *reinterpret_cast<const uint4*>(&smA[0][rb * 256 + ((c0 ^ (rb & 7)) << 4)]);
                const uint4 xb1 = *reinterpret_cast<const uint4*>(&smA[0][rb * 256 + (((c0 + 1) ^ (rb & 7)) << 4)]);
                ik2m_mma(acc[mt][0], xa0.x, xb0.x, xa0.y, xb0.y, b0[0], b0[1]);
                ik2m_mma(acc[mt][0], xa0.z, xb0.z, xa0.w, xb0.w, b0[2], b0[3]);
                ik2m_mma(acc[mt][0], xa1.x, xb1.x, xa1.y, xb1.y, b0[4], b0[5]);
                ik2m_mma(acc[mt][0], xa1.z, xb1.z, xa1.w, xb1.w, b0[6], b0[7]);
                ik2m_mma(acc[mt][1], xa0.x, xb0.x, xa0.y, xb0.y, b1[0], b1[1]);
                ik2m_mma(acc[mt][1], xa0.z, xb0.z, xa0.w, xb0.w, b1[2], b1[3]);
                ik2m_mma(acc[mt][1], xa1.x, xb1.x, xa1.y, xb1.y, b1[4], b1[5]);
                ik2m_mma(acc[mt][1], xa1.z, xb1.z, xa1.w, xb1.w, b1[6], b1[7]);
            }
            __syncthreads();
        }
    }
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
        const int t0 = 16 * mt + gid, t1 = t0 + 8;
        ik2m_expert_store<GU>(acc[mt], gs, n0c + warp, tig, base + t0, base + t1, t0 < rows, t1 < rows, N, Din, Yf);
    }
}
extern "C" __global__ __launch_bounds__(256, 2) void ink2_nv4_gu_pf(
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, const float* __restrict__ GS,
        const __half* __restrict__ X, const int* __restrict__ rowtok, __half* __restrict__ Din,
        const int* __restrict__ tiles, const int* __restrict__ ntiles, const int* __restrict__ offs,
        const int* __restrict__ cnt, int K, int N) {
    INK2_PDL_ENTRY();
    ik2m_nv4_pf_body<true>(W, S, GS, X, rowtok, Din, nullptr, tiles, ntiles, offs, cnt, K, N);
}
extern "C" __global__ __launch_bounds__(256, 2) void ink2_nv4_dn_pf(
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, const float* __restrict__ GS,
        const __half* __restrict__ Din, float* __restrict__ Y,
        const int* __restrict__ tiles, const int* __restrict__ ntiles, const int* __restrict__ offs,
        const int* __restrict__ cnt, int K, int N) {
    INK2_PDL_ENTRY();
    ik2m_nv4_pf_body<false>(W, S, GS, Din, nullptr, nullptr, Y, tiles, ntiles, offs, cnt, K, N);
}

// ------------------------------------------------------------------------------------------------
// MXFP8 prefill tiles, smem-staged like the NVFP4 pair: 64 compact rows x 128 weight rows per CTA, 64-k steps,
// 2 cp.async stages (activations 2 x 8 KB, codes 2 x 8 KB; the scales ride in registers). Per 64-k block a row's
// arithmetic is ik2m_mx8_block, the decode kernel's (bit-identical rows). K % 64 == 0, N % 128 == 0.
// ------------------------------------------------------------------------------------------------
template <bool GU>
__device__ __forceinline__ void ik2m_mx8_pf_body(
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
        const __half* __restrict__ X, const int* __restrict__ rowmap, __half* __restrict__ Din, float* __restrict__ Yf,
        const int* __restrict__ tiles, const int* __restrict__ ntiles, const int* __restrict__ offs,
        const int* __restrict__ cnt, int K, int N) {
    __shared__ __align__(128) uint8_t smA[2][64 * 128];
    __shared__ __align__(128) uint8_t smB[2][8 * 1024];
    const int tile = blockIdx.y;
    if (tile >= *ntiles) return;
    const int e = tiles[2 * tile], r0 = tiles[2 * tile + 1];
    const int rows = min(64, cnt[e] - r0);
    const int base = offs[e] + r0;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
    const int n0c = blockIdx.x * 8;
    const int KB = K >> 6;
    const uint8_t* We = W + (size_t)e * N * K;
    const uint8_t* stp = S + (size_t)e * N * (K >> 5) + (size_t)(n0c + warp) * KB * 32 + tig * 8;
    auto load = [&](int st, int b) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int c = tid + 256 * j, row = c >> 3, ch = c & 7;
            const int rr = base + min(row, rows - 1);
            const int src = rowmap ? rowmap[rr] : rr;
            ik2m_cp16(&smA[st][row * 128 + ((ch ^ (row & 7)) << 4)], X + (size_t)src * K + b * 64 + ch * 8);
        }
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int c = tid + 256 * j, tl = c >> 6, o = (c & 63) << 4;
            ik2m_cp16(&smB[st][tl * 1024 + o], We + ((size_t)(n0c + tl) * KB + b) * 1024 + o);
        }
    };
    float acc[4][2][4];
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc[mt][h][r] = 0.f;
    load(0, 0);
    ik2m_cp_commit();
    uint2 sc = __ldg(reinterpret_cast<const uint2*>(stp));
    for (int b = 0; b < KB; ++b) {
        const int st = b & 1;
        if (b + 1 < KB) load(st ^ 1, b + 1);
        ik2m_cp_commit();
        const uint2 nsc = (b + 1 < KB) ? __ldg(reinterpret_cast<const uint2*>(stp + (size_t)(b + 1) * 32)) : sc;
        ik2m_cp_wait<1>();
        __syncthreads();
        const uint4 w0 = *reinterpret_cast<const uint4*>(&smB[st][warp * 1024 + lane * 16]);
        const uint4 w1 = *reinterpret_cast<const uint4*>(&smB[st][warp * 1024 + 512 + lane * 16]);
        #pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
            const int ra = mt * 16 + gid, rb = ra + 8;
            const uint4 xa0 = *reinterpret_cast<const uint4*>(&smA[st][ra * 128 + ((tig ^ (ra & 7)) << 4)]);
            const uint4 xa1 = *reinterpret_cast<const uint4*>(&smA[st][ra * 128 + (((4 + tig) ^ (ra & 7)) << 4)]);
            const uint4 xb0 = *reinterpret_cast<const uint4*>(&smA[st][rb * 128 + ((tig ^ (rb & 7)) << 4)]);
            const uint4 xb1 = *reinterpret_cast<const uint4*>(&smA[st][rb * 128 + (((4 + tig) ^ (rb & 7)) << 4)]);
            ik2m_mx8_block(acc[mt], w0, w1, sc, xa0, xa1, xb0, xb1);
        }
        sc = nsc;
        __syncthreads();
    }
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
        const int t0 = 16 * mt + gid, t1 = t0 + 8;
        ik2m_expert_store<GU>(acc[mt], 1.0f, n0c + warp, tig, base + t0, base + t1, t0 < rows, t1 < rows, N, Din, Yf);
    }
}
extern "C" __global__ __launch_bounds__(256, 1) void ink2_mx8_gu_pf(
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
        const __half* __restrict__ X, const int* __restrict__ rowtok, __half* __restrict__ Din,
        const int* __restrict__ tiles, const int* __restrict__ ntiles, const int* __restrict__ offs,
        const int* __restrict__ cnt, int K, int N) {
    INK2_PDL_ENTRY();
    ik2m_mx8_pf_body<true>(W, S, X, rowtok, Din, nullptr, tiles, ntiles, offs, cnt, K, N);
}
extern "C" __global__ __launch_bounds__(256, 1) void ink2_mx8_dn_pf(
        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
        const __half* __restrict__ Din, float* __restrict__ Y,
        const int* __restrict__ tiles, const int* __restrict__ ntiles, const int* __restrict__ offs,
        const int* __restrict__ cnt, int K, int N) {
    INK2_PDL_ENTRY();
    ik2m_mx8_pf_body<false>(W, S, Din, nullptr, nullptr, Y, tiles, ntiles, offs, cnt, K, N);
}
