// kernels/tests/dsv41_mxfp4_bench.cu — standalone harness for the DSV4.1 native MXFP4 routed-expert
// kernels (band 30, leg KN2): correctness against a CPU model of the reference arithmetic
// (act_quant + fp4_gemm + SwiGLU + per-expert bf16 + ascending-id FP32 sum), batch invariance
// across row counts / classes, and the byte-floor / prefill timing tables.
// The served path is src/dsv41/weights/native/moe.rs; this file only drives the same kernels.
//
// Build (on .11):  nvcc -O3 -std=c++17 -gencode arch=compute_121a,code=sm_121a -Ikernels \
//                       kernels/tests/dsv41_mxfp4_bench.cu -o dsv41_mxfp4_bench
// Run (GPU box):   ./dsv41_mxfp4_bench --check | --check-wide | --bench | --bench-wide [--experts 384 --topk 6] [--reps N]
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <thread>
#include <algorithm>
#include <random>

#define KERNEL_BUILD_ID 0ULL
#include "gpu_dsv41/00_abi.cuh"
#include "gpu_dsv41/30_mxfp4_moe.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1);} } while (0)

static const int DIM = 5120, INTER = 2304;
static bool getenv_dbg = false;

// ---- host value codes ----------------------------------------------------------------------------
static inline int e4m3_int(uint8_t c) {   // value * 2^9 as a signed integer (exact)
    int s = c >> 7, e = (c >> 3) & 15, m = c & 7;
    int v = e == 0 ? m : (8 + m) << (e - 1);
    return s ? -v : v;
}
static inline int e2m1_int(uint8_t c) {   // value * 2 as a signed integer (exact)
    int e = (c >> 1) & 3, m = c & 1;
    int v = e == 0 ? m : (2 + m) << (e - 1);
    return (c & 8) ? -v : v;
}
static inline float bf16r(float v) { return __bfloat162float(__float2bfloat16(v)); }
static inline float f_of_e8m0(uint8_t e) { return ldexpf(1.0f, (int)e - 127); }
static uint8_t e4m3_rne(float x) {   // RNE satfinite, by the CUDA host conversion (same rounding as cvt.rn.satfinite)
    return (uint8_t)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}
static float pow2_scale(float amax) {
    float v = amax * (1.0f / 448.0f);
    uint32_t b; memcpy(&b, &v, 4);
    int e = (int)((b >> 23) & 0xFF) - 127 + ((b & 0x7FFFFF) ? 1 : 0);
    return ldexpf(1.0f, e);
}
// act_quant of one row of n bf16-valued floats -> codes + scales
static void act_quant_row(const float* x, int n, uint8_t* q, uint8_t* s) {
    for (int g = 0; g < n / 32; g++) {
        float am = 0; for (int i = 0; i < 32; i++) am = fmaxf(am, fabsf(x[g * 32 + i]));
        am = fmaxf(am, 1e-4f);
        float sc = pow2_scale(am);
        for (int i = 0; i < 32; i++) q[g * 32 + i] = e4m3_rne(fminf(fmaxf(x[g * 32 + i] / sc, -448.f), 448.f));
        uint32_t b; memcpy(&b, &sc, 4); s[g] = (uint8_t)((b >> 23) & 0xFF);
    }
}
// one k-block's tensor-core dot as GB10 computes it (KN2 probe 2026-11-18): the exact 32-product sum
// ROUNDED TOWARD ZERO to f32 when it needs more than 24 bits (both TileLang's e4m3 x e4m3 instruction and
// our block-scaled e2m1 x e4m3 one), then scaled by a power of two (exact).
static float dot_rz(long long s, int e2) {
    double ex = (double)s;               // exact (|s| < 2^31)
    float f = (float)ex;                 // RNE
    if (fabs((double)f) > fabs(ex)) f = nextafterf(f, 0.0f);
    return (float)((double)f * ldexp(1.0, e2));
}
// the reference fp4_gemm for one (weight row, activation row): sequential FP32 over k-blocks
static float fp4_dot(const uint8_t* wrow, const uint8_t* wsc, const uint8_t* xq, const uint8_t* xs, int nkb) {
    float acc = 0.0f;
    for (int kb = 0; kb < nkb; kb++) {
        long long s = 0;
        for (int k = 0; k < 32; k++) {
            uint8_t byte = wrow[kb * 16 + (k >> 1)];
            uint8_t c = (k & 1) ? (byte >> 4) : (byte & 15);
            s += (long long)e2m1_int(c) * e4m3_int(xq[kb * 32 + k]);
        }
        acc = acc + dot_rz(s, (int)wsc[kb] + (int)xs[kb] - 254 - 10);
    }
    return acc;
}
static float swiglu_h(float gb, float ub, float w, float L) {
    float u = ub < -L ? -L : (ub > L ? L : ub);
    float g = gb > L ? L : gb;
    float s = g / (1.0f + expf(-g));
    return bf16r(w * (s * u));
}

// ---- raw expert weights (checkpoint layout) and the mx1 packer ----------------------------------
struct RawExpert { std::vector<uint8_t> w1, s1, w3, s3, w2, s2; };   // w [N][K/2], s [N][K/32]
static void gen_raw(RawExpert& r, std::mt19937& rng) {
    std::uniform_int_distribution<int> b(0, 255), sc(112, 126);
    r.w1.resize((size_t)INTER * DIM / 2); r.w3.resize(r.w1.size()); r.w2.resize((size_t)DIM * INTER / 2);
    r.s1.resize((size_t)INTER * DIM / 32); r.s3.resize(r.s1.size()); r.s2.resize((size_t)DIM * INTER / 32);
    for (auto& v : r.w1) v = b(rng); for (auto& v : r.w3) v = b(rng); for (auto& v : r.w2) v = b(rng);
    for (auto& v : r.s1) v = sc(rng); for (auto& v : r.s3) v = sc(rng); for (auto& v : r.s2) v = sc(rng);
}
static uint8_t nib(const uint8_t* row, int k) { uint8_t byte = row[k >> 1]; return (k & 1) ? (byte >> 4) : (byte & 15); }
// pack one m-tile: rowptr[r] (16 rows, K-sliced from k0), scale rows srow[r] (from kb0)
static void pack_mtile(uint8_t* out, const uint8_t* const* rowp, const uint8_t* const* srow, int k0, int nkb) {
    int nkq = (nkb + 3) / 4;
    for (int kb = 0; kb < nkb; kb++)
        for (int lane = 0; lane < 32; lane++) {
            int g = lane >> 2, t = lane & 3;
            uint64_t v = 0;
            for (int n = 0; n < 16; n++) {
                int r = n >> 2, i = n & 3;
                int row = g + 8 * (r & 1), k = kb * 32 + 4 * t + 16 * (r >> 1) + i;
                v |= (uint64_t)nib(rowp[row], k0 + k) << (4 * n);
            }
            memcpy(out + (size_t)kb * 256 + lane * 8, &v, 8);
        }
    uint8_t* sc = out + (size_t)nkb * 256;
    memset(sc, 0, (size_t)nkq * 64);
    for (int q = 0; q < nkq; q++)
        for (int gg = 0; gg < 8; gg++) for (int h = 0; h < 2; h++) for (int v = 0; v < 4; v++) {
            int kb = 4 * q + v; if (kb >= nkb) continue;
            sc[q * 64 + (2 * gg + h) * 4 + v] = srow[gg + 8 * h][(k0 / 32) + kb];
        }
}
static size_t mt_bytes(int nkb) { return (size_t)nkb * 256 + (size_t)((nkb + 3) / 4) * 64; }
// w13 of one expert, inner slice [i0, i0+ns)
static void pack_w13(uint8_t* out, const RawExpert& r, int i0, int ns) {
    int nkb = DIM / 32; size_t mb = mt_bytes(nkb);
    for (int j = 0; j < ns / 8; j++) {
        const uint8_t* rp[16]; const uint8_t* sp[16];
        for (int x = 0; x < 8; x++) {
            int row = i0 + 8 * j + x;
            rp[x] = r.w1.data() + (size_t)row * DIM / 2; sp[x] = r.s1.data() + (size_t)row * DIM / 32;
            rp[x + 8] = r.w3.data() + (size_t)row * DIM / 2; sp[x + 8] = r.s3.data() + (size_t)row * DIM / 32;
        }
        pack_mtile(out + j * mb, rp, sp, 0, nkb);
    }
}
static void pack_w2(uint8_t* out, const RawExpert& r, int i0, int ns) {
    int nkb = ns / 32; size_t mb = mt_bytes(nkb);
    for (int j = 0; j < DIM / 16; j++) {
        const uint8_t* rp[16]; const uint8_t* sp[16];
        for (int x = 0; x < 16; x++) { rp[x] = r.w2.data() + (size_t)(16 * j + x) * INTER / 2; sp[x] = r.s2.data() + (size_t)(16 * j + x) * INTER / 32; }
        pack_mtile(out + j * mb, rp, sp, i0, nkb);
    }
}

// ---- a device layer (packed, at one inner slice) -----------------------------------------------
struct DevLayer { uint8_t* w13 = nullptr; uint8_t* w2 = nullptr; size_t e13 = 0, e2 = 0, mt13 = 0, mt2 = 0; int n = 0, i0 = 0, ns = 0; };
static DevLayer upload(const std::vector<RawExpert>& raws, int nexp_alloc, int i0, int ns) {
    DevLayer L; L.n = nexp_alloc; L.i0 = i0; L.ns = ns;
    L.mt13 = mt_bytes(DIM / 32); L.e13 = L.mt13 * (ns / 8);
    L.mt2 = mt_bytes(ns / 32); L.e2 = L.mt2 * (DIM / 16);
    CK(cudaMalloc(&L.w13, L.e13 * nexp_alloc)); CK(cudaMalloc(&L.w2, L.e2 * nexp_alloc));
    // pack each generated expert once (threads), then place it at every expert id e with e % n == its index
    size_t n = raws.size();
    std::vector<std::vector<uint8_t>> b13(n, std::vector<uint8_t>(L.e13)), b2(n, std::vector<uint8_t>(L.e2));
    std::vector<std::thread> th;
    for (size_t i = 0; i < n; i++) th.emplace_back([&, i] { pack_w13(b13[i].data(), raws[i], i0, ns); pack_w2(b2[i].data(), raws[i], i0, ns); });
    for (auto& t : th) t.join();
    for (int e = 0; e < nexp_alloc; e++) {
        CK(cudaMemcpy(L.w13 + (size_t)e * L.e13, b13[e % n].data(), L.e13, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(L.w2 + (size_t)e * L.e2, b2[e % n].data(), L.e2, cudaMemcpyHostToDevice));
    }
    return L;
}

// ---- one MoE call on the device (grouped path) ---------------------------------------------------
struct Bufs { uint8_t *xq, *xs, *hq, *hs; float* y; float* rout; __nv_bfloat16* dbg; __nv_bfloat16* x; int* ids; float* rw; };
static int ldxs_of() { return ((DIM / 32) + 3) / 4 * 4; }
static int ldhs_of(int ns) { return ((ns / 32) + 3) / 4 * 4; }
static void run_grouped(const DevLayer& L, const Bufs& B, int rows, int topk, int nexp, bool dbg, bool round_bf16, cudaStream_t st,
                        int* fc1_grid_out = nullptr, int phase = 0) {
    int nkb1 = DIM / 32, ldx = DIM, ldxs = ldxs_of(), ldh = L.ns, ldhs = ldhs_of(L.ns);
    int nw = rows * (DIM / 32);
    if (phase != 2) d41_mx_act_quant<<<(nw * 32 + 255) / 256, 256, 0, st>>>(B.x, DIM, rows, DIM, B.xq, ldx, B.xs, ldxs);
    D41MxFc1Args a{};
    a.w = L.w13; a.e_stride = L.e13; a.mt_stride = L.mt13; a.xq = B.xq; a.xs = B.xs; a.ids = B.ids; a.rw = B.rw;
    a.hq = B.hq; a.hs = B.hs; a.dbg = dbg ? B.dbg : nullptr; a.rows = rows; a.top_k = topk; a.e_first = 0; a.n_local = nexp;
    a.nkb = nkb1; a.slice = L.ns; a.ldx = ldx; a.ldxs = ldxs; a.ldh = ldh; a.ldhs = ldhs; a.limit = 10.0f;
    int nt = (rows + 7) / 8; if (nt > 4) { fprintf(stderr, "rows > 32\n"); exit(1); }
    int maxd = std::min(rows * topk, nexp);
    int grid1 = maxd * (L.ns / 32);
    if (fc1_grid_out) *fc1_grid_out = grid1;
    D41MxFc2Args b{};
    b.w = L.w2; b.e_stride = L.e2; b.mt_stride = L.mt2; b.hq = B.hq; b.hs = B.hs; b.ids = B.ids; b.y = B.y;
    b.route_out = dbg ? B.rout : nullptr; b.rows = rows; b.top_k = topk; b.e_first = 0; b.n_local = nexp;
    b.nkb = L.ns / 32; b.dim = DIM; b.ldh = ldh; b.ldhs = ldhs; b.ldy = DIM; b.round_bf16 = round_bf16 ? 1 : 0; b.accumulate = 0;
    int grid2 = DIM / 64;
    bool f1 = phase != 2, f2 = phase != 1;
    switch (nt) {
    case 1: if (f1) d41_mx_fc1_g1<<<grid1, 128, 0, st>>>(a); if (f2) d41_mx_fc2_g1<<<grid2, 128, 0, st>>>(b); break;
    case 2: if (f1) d41_mx_fc1_g2<<<grid1, 128, 0, st>>>(a); if (f2) d41_mx_fc2_g2<<<grid2, 128, 0, st>>>(b); break;
    case 3: if (f1) d41_mx_fc1_g3<<<grid1, 128, 0, st>>>(a); if (f2) d41_mx_fc2_g3<<<grid2, 128, 0, st>>>(b); break;
    default: if (f1) d41_mx_fc1_g4<<<grid1, 128, 0, st>>>(a); if (f2) d41_mx_fc2_g4<<<grid2, 128, 0, st>>>(b); break;
    }
}

// ---- the wide (prefill) path ---------------------------------------------------------------------
// PK3: the token tile (routes per tile) by routes per expert, as src/dsv41/weights/native/mxfp4_launch.rs `wide_tt` picks it
// (keep the two in step); --tt 64|128 forces one. --bf16: slices store bf16 per-route outputs (--dsv41-prefill-bf16-partials).
static int g_tt = 0;
static bool g_bf16 = false;
static int tt_for(int rows, int topk, int nexp) { return (long)rows * topk <= (long)D41_MX_TT64_MAX_RPE * nexp ? 64 : 128; }
static int tt_of(int rows, int topk, int nexp) { return g_tt ? g_tt : tt_for(rows, topk, nexp); }
struct WideBufs { uint8_t *xq, *xs, *hq, *hs; float* out2; int *cnt, *off, *cur, *list, *tile_e, *tile_s, *ntiles; int max_tiles, tt; };
static WideBufs alloc_wide(int rows, int topk, int ns, int nexp, int tt) {
    WideBufs W{}; int nr = rows * topk; W.tt = tt; W.max_tiles = (nr + tt - 1) / tt + nexp;
    CK(cudaMalloc(&W.xq, (size_t)rows * DIM)); CK(cudaMalloc(&W.xs, (size_t)rows * ldxs_of())); CK(cudaMalloc(&W.hq, (size_t)nr * ns));
    CK(cudaMalloc(&W.hs, (size_t)nr * ldhs_of(ns))); CK(cudaMalloc(&W.out2, (size_t)nr * DIM * 4));
    CK(cudaMalloc(&W.cnt, nexp * 4)); CK(cudaMalloc(&W.off, (nexp + 1) * 4)); CK(cudaMalloc(&W.cur, nexp * 4)); CK(cudaMalloc(&W.list, nr * 4));
    CK(cudaMalloc(&W.tile_e, W.max_tiles * 4)); CK(cudaMalloc(&W.tile_s, W.max_tiles * 4)); CK(cudaMalloc(&W.ntiles, 4));
    return W;
}
static void free_wide(WideBufs& W) {
    cudaFree(W.xq); cudaFree(W.xs); cudaFree(W.hq); cudaFree(W.hs); cudaFree(W.out2); cudaFree(W.cnt); cudaFree(W.off); cudaFree(W.cur);
    cudaFree(W.list); cudaFree(W.tile_e); cudaFree(W.tile_s); cudaFree(W.ntiles);
}
// phase: 0 all, 1 act_quant + plan + FC1, 2 FC2 + combine
static void run_wide(const DevLayer& L, const WideBufs& W, const __nv_bfloat16* x, const int* ids, const float* rw, float* y,
                     int rows, int topk, int nexp, bool round_bf16, cudaStream_t st, int phase = 0) {
    int ldxs = ldxs_of(), ldhs = ldhs_of(L.ns);
    if (phase != 2) {
        d41_mx_act_quant_p<<<(rows * (DIM / 32) * 32 + 255) / 256, 256, 0, st>>>(x, DIM, rows, DIM, W.xq, DIM, W.xs, ldxs);
        D41MxPlanWArgs pa{}; pa.ids = ids; pa.cnt = W.cnt; pa.off = W.off; pa.cur = W.cur; pa.list = W.list; pa.tile_e = W.tile_e; pa.tile_s = W.tile_s;
        pa.n_tiles = W.ntiles; pa.rows = rows; pa.top_k = topk; pa.e_first = 0; pa.n_local = nexp; pa.max_tiles = W.max_tiles; pa.tt = W.tt;
        d41_mx_plan_w<<<1, 1024, 0, st>>>(pa);
    }
    D41MxWideArgs a{};
    a.w = L.w13; a.e_stride = L.e13; a.mt_stride = L.mt13; a.aq = W.xq; a.as = W.xs; a.list = W.list; a.off = W.off; a.tile_e = W.tile_e;
    a.tile_s = W.tile_s; a.n_tiles = W.ntiles; a.rw = rw; a.hq = W.hq; a.hs = W.hs; a.out2 = W.out2; a.top_k = topk; a.nkb = DIM / 32;
    a.slice = L.ns; a.dim = DIM; a.lda = DIM; a.ldas = ldxs; a.ldh = L.ns; a.ldhs = ldhs; a.fc1 = 1; a.round_bf16 = round_bf16; a.limit = 10.0f;
    if (phase != 2) {
        if (W.tt == 64) d41_mx_fc1_w64<<<W.max_tiles * (L.ns / D41_MX_FC1W_INNER), D41_MX_FC1W_WARPS * 32, 0, st>>>(a);
        else d41_mx_fc1_w<<<W.max_tiles * (L.ns / D41_MX_FC1W_INNER), D41_MX_FC1W_WARPS * 32, 0, st>>>(a);
    }
    D41MxWideArgs b = a; b.w = L.w2; b.e_stride = L.e2; b.mt_stride = L.mt2; b.aq = W.hq; b.as = W.hs; b.nkb = L.ns / 32; b.lda = L.ns; b.ldas = ldhs; b.fc1 = 0;
    if (phase != 1) {
        if (W.tt == 64) d41_mx_fc2_w64<<<W.max_tiles * (DIM / D41_MX_FC2W_COLS), D41_MX_FC2W_WARPS * 32, 0, st>>>(b);
        else d41_mx_fc2_w<<<W.max_tiles * (DIM / D41_MX_FC2W_COLS), D41_MX_FC2W_WARPS * 32, 0, st>>>(b);
        d41_mx_combine_w<<<dim3(DIM / 256, rows), 256, 0, st>>>(W.out2, ids, y, DIM, rows, topk, DIM, 0, nexp, 0, round_bf16 ? 1 : 0);
    }
}

static void gen_x(std::vector<float>& x, int rows, std::mt19937& rng) {
    std::normal_distribution<float> n(0.f, 1.f); std::uniform_int_distribution<int> o(0, 63);
    x.resize((size_t)rows * DIM);
    for (auto& v : x) { float a = n(rng); if (o(rng) == 0) a *= 30.f; v = bf16r(a); }
}
static void gen_routes(std::vector<int>& ids, std::vector<float>& w, int rows, int topk, int nexp, std::mt19937& rng) {
    ids.resize(rows * topk); w.resize(rows * topk);
    std::uniform_real_distribution<float> u(0.05f, 1.f);
    for (int r = 0; r < rows; r++) {
        std::vector<int> pick;
        while ((int)pick.size() < topk) { int e = rng() % nexp; if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e); }
        float s = 0; std::vector<float> ws(topk); for (auto& v : ws) { v = u(rng); s += v; }
        for (int k = 0; k < topk; k++) { ids[r * topk + k] = pick[k]; w[r * topk + k] = ws[k] / (s + 1e-20f) * 1.5f; }
    }
}

// ---- correctness ---------------------------------------------------------------------------------
static int check(int nexp, int topk) {
    std::mt19937 rng(1234);
    const int NGEN = 24;   // distinct generated experts; the layer reuses them cyclically
    std::vector<RawExpert> raws(NGEN); for (auto& r : raws) gen_raw(r, rng);
    int fails = 0;
    const int slices[] = {INTER, INTER / 2, INTER / 4, INTER / 8};
    for (int si = 0; si < 4; si++) {
        int ns = slices[si];
        int rank = si == 0 ? 0 : (si % 2 == 0 ? 1 : (INTER / ns) - 1);   // also a non-zero rank's slice
        int i0 = rank * ns;
        DevLayer L = upload(raws, nexp, i0, ns);
        Bufs B{};
        const int MR = 32;
        CK(cudaMalloc(&B.x, MR * DIM * 2)); CK(cudaMalloc(&B.xq, MR * DIM)); CK(cudaMalloc(&B.xs, MR * ldxs_of()));
        CK(cudaMalloc(&B.hq, MR * topk * ns)); CK(cudaMalloc(&B.hs, MR * topk * ldhs_of(ns))); CK(cudaMalloc(&B.y, MR * DIM * 4));
        CK(cudaMalloc(&B.rout, (size_t)MR * topk * DIM * 4)); CK(cudaMalloc(&B.dbg, (size_t)MR * topk * 2 * ns * 2));
        CK(cudaMalloc(&B.ids, MR * topk * 4)); CK(cudaMalloc(&B.rw, MR * topk * 4));
        bool whole = ns == INTER;
        std::vector<float> y_rows1;   // batch invariance: y of every row from the 1-row calls
        for (int rows : {1, 2, 3, 6, 8, 9, 12, 16, 17, 24, 32}) {
            std::vector<float> x; gen_x(x, rows, rng);
            std::vector<int> ids; std::vector<float> rw; gen_routes(ids, rw, rows, topk, nexp, rng);
            if (rows == 24) for (int r = 1; r < rows; r++) for (int k = 0; k < topk; k++) ids[r * topk + k] = ids[k];  // all rows share experts
            std::vector<__nv_bfloat16> xb(x.size()); for (size_t i = 0; i < x.size(); i++) xb[i] = __float2bfloat16(x[i]);
            CK(cudaMemcpy(B.x, xb.data(), xb.size() * 2, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(B.ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(B.rw, rw.data(), rw.size() * 4, cudaMemcpyHostToDevice));
            run_grouped(L, B, rows, topk, nexp, true, whole, 0);
            CK(cudaDeviceSynchronize());
            int nr = rows * topk;
            std::vector<uint8_t> xq(rows * DIM), xs(rows * ldxs_of()), hq((size_t)nr * ns), hs((size_t)nr * ldhs_of(ns));
            std::vector<float> y((size_t)rows * DIM), rout((size_t)nr * DIM);
            std::vector<__nv_bfloat16> dbg((size_t)nr * 2 * ns);
            CK(cudaMemcpy(xq.data(), B.xq, xq.size(), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(xs.data(), B.xs, xs.size(), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hq.data(), B.hq, hq.size(), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hs.data(), B.hs, hs.size(), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(y.data(), B.y, y.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(rout.data(), B.rout, rout.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(dbg.data(), B.dbg, dbg.size() * 2, cudaMemcpyDeviceToHost));
            // CPU: act_quant
            long bad_aq = 0;
            std::vector<uint8_t> cq(DIM), cs(DIM / 32);
            for (int r = 0; r < rows; r++) {
                act_quant_row(&x[(size_t)r * DIM], DIM, cq.data(), cs.data());
                for (int i = 0; i < DIM; i++) bad_aq += cq[i] != xq[(size_t)r * DIM + i];
                for (int i = 0; i < DIM / 32; i++) bad_aq += cs[i] != xs[(size_t)r * ldxs_of() + i];
            }
            // CPU: FC1 gate / up (exact), h codes (CPU expf), FC2 from the GPU h (exact), combine
            std::vector<long> bad_gu(nr, 0), bad_h(nr, 0), bad_ro(nr, 0);
            std::vector<float> ycpu((size_t)rows * DIM, 0.f);
            auto work = [&](int r0, int r1) {
                std::vector<float> h(ns); std::vector<uint8_t> hq_c(ns), hs_c(ns / 32);
                for (int route = r0; route < r1; route++) {
                    int row = route / topk; const RawExpert& R = raws[ids[route] % NGEN];
                    const uint8_t* xqr = &xq[(size_t)row * DIM]; const uint8_t* xsr = &xs[(size_t)row * ldxs_of()];
                    for (int i = 0; i < ns; i++) {
                        int ir = i0 + i;
                        float gt = bf16r(fp4_dot(&R.w1[(size_t)ir * DIM / 2], &R.s1[(size_t)ir * DIM / 32], xqr, xsr, DIM / 32));
                        float up = bf16r(fp4_dot(&R.w3[(size_t)ir * DIM / 2], &R.s3[(size_t)ir * DIM / 32], xqr, xsr, DIM / 32));
                        bad_gu[route] += gt != __bfloat162float(dbg[(size_t)route * 2 * ns + i]);
                        bad_gu[route] += up != __bfloat162float(dbg[(size_t)route * 2 * ns + ns + i]);
                        h[i] = swiglu_h(gt, up, rw[route], 10.0f);
                    }
                    act_quant_row(h.data(), ns, hq_c.data(), hs_c.data());
                    for (int i = 0; i < ns; i++) bad_h[route] += hq_c[i] != hq[(size_t)route * ns + i];
                    for (int i = 0; i < ns / 32; i++) bad_h[route] += hs_c[i] != hs[(size_t)route * ldhs_of(ns) + i];
                    // FC2 from the GPU's own h (isolates the expf ulp question)
                    const uint8_t* hqr = &hq[(size_t)route * ns]; const uint8_t* hsr = &hs[(size_t)route * ldhs_of(ns)];
                    for (int o = 0; o < DIM; o++) {
                        // w2 row o, k-slice [i0, i0+ns)
                        float acc = 0.f;
                        const uint8_t* wr = &R.w2[(size_t)o * INTER / 2]; const uint8_t* sr = &R.s2[(size_t)o * INTER / 32];
                        acc = fp4_dot(wr + i0 / 2, sr + i0 / 32, hqr, hsr, ns / 32);
                        float v = whole ? bf16r(acc) : acc;
                        if (v != rout[(size_t)route * DIM + o]) {
                            if (bad_ro[route] < 2 && getenv_dbg) {
                                fprintf(stderr, "  ro mismatch route %d o %d: gpu %.9g cpu %.9g\n", route, o, rout[(size_t)route * DIM + o], v);
                                float a2 = 0.f; const uint8_t* w0 = wr + i0 / 2; const uint8_t* s0 = sr + i0 / 32;
                                for (int kb = 0; kb < ns / 32; kb++) {
                                    long long ss = 0; int mx = 0;
                                    for (int k = 0; k < 32; k++) { uint8_t byte = w0[kb * 16 + (k >> 1)]; uint8_t c = (k & 1) ? (byte >> 4) : (byte & 15);
                                        long long pr = (long long)e2m1_int(c) * e4m3_int(hqr[kb * 32 + k]); ss += pr; }
                                    double vv = (double)ss * ldexp(1.0, (int)s0[kb] + (int)hsr[kb] - 254 - 10);
                                    a2 = a2 + dot_rz(ss, (int)s0[kb] + (int)hsr[kb] - 254 - 10);
                                    fprintf(stderr, "    kb %d dot %lld sa %d sb %d term %.9g acc %.9g repr %d\n", kb, ss, s0[kb], hsr[kb], vv, a2, (double)(float)vv == vv);
                                }
                            }
                            bad_ro[route]++;
                        }
                    }
                }
            };
            unsigned nth = std::max(1u, std::thread::hardware_concurrency());
            std::vector<std::thread> th;
            for (unsigned i = 0; i < nth; i++) { int a0 = nr * i / nth, a1 = nr * (i + 1) / nth; th.emplace_back(work, a0, a1); }
            for (auto& tt : th) tt.join();
            // combine: ascending expert id per row, from the GPU route outputs (checked above)
            long bad_y = 0;
            for (int r = 0; r < rows; r++) {
                std::vector<int> ks(topk); for (int k = 0; k < topk; k++) ks[k] = k;
                std::sort(ks.begin(), ks.end(), [&](int a, int b2) { return ids[r * topk + a] < ids[r * topk + b2]; });
                for (int o = 0; o < DIM; o++) {
                    float acc = 0.f; for (int k : ks) acc += rout[(size_t)(r * topk + k) * DIM + o];
                    bad_y += acc != y[(size_t)r * DIM + o];
                }
            }
            long sgu = 0, sh = 0, sro = 0; for (int i = 0; i < nr; i++) { sgu += bad_gu[i]; sh += bad_h[i]; sro += bad_ro[i]; }
            bool ok = bad_aq == 0 && sgu == 0 && sro == 0 && bad_y == 0 && sh <= (long)nr * 2;
            printf("slice %4d (i0 %4d) rows %2d: act_quant %ld, gate/up %ld of %ld, h codes %ld of %ld, route out %ld of %ld, y %ld of %ld  %s\n",
                   ns, i0, rows, bad_aq, sgu, (long)nr * 2 * ns, sh, (long)nr * (ns + ns / 32), sro, (long)nr * DIM, bad_y, (long)rows * DIM, ok ? "OK" : "FAIL");
            fails += !ok;
            // batch invariance: re-run every row alone (GEMV class) and compare bitwise with this call's row
            if (rows > 1) {
                long bad_bi = 0;
                std::vector<float> y1(DIM);
                for (int r = 0; r < rows; r++) {
                    CK(cudaMemcpy(B.x, xb.data() + (size_t)r * DIM, DIM * 2, cudaMemcpyHostToDevice));
                    CK(cudaMemcpy(B.ids, ids.data() + r * topk, topk * 4, cudaMemcpyHostToDevice));
                    CK(cudaMemcpy(B.rw, rw.data() + r * topk, topk * 4, cudaMemcpyHostToDevice));
                    run_grouped(L, B, 1, topk, nexp, false, whole, 0);
                    CK(cudaMemcpy(y1.data(), B.y, DIM * 4, cudaMemcpyDeviceToHost));
                    for (int o = 0; o < DIM; o++) bad_bi += memcmp(&y1[o], &y[(size_t)r * DIM + o], 4) != 0;
                }
                printf("    batch invariance rows %2d vs 1-row calls: %ld of %ld differ  %s\n", rows, bad_bi, (long)rows * DIM, bad_bi ? "FAIL" : "OK");
                fails += bad_bi != 0;
            }
        }
        cudaFree(B.x); cudaFree(B.xq); cudaFree(B.xs); cudaFree(B.hq); cudaFree(B.hs); cudaFree(B.y); cudaFree(B.rout); cudaFree(B.dbg);
        cudaFree(B.ids); cudaFree(B.rw); cudaFree(L.w13); cudaFree(L.w2);
    }
    printf(fails ? "KN2_CHECK_FAIL (%d)\n" : "KN2_CHECK_OK\n", fails);
    return fails ? 1 : 0;
}

// ---- the wide path equals the grouped path, row by row, bit for bit ------------------------------
static int check_wide(int nexp, int topk) {
    std::mt19937 rng(4321);
    std::vector<RawExpert> raws(12); for (auto& r : raws) gen_raw(r, rng);
    int fails = 0;
    for (int ns : {INTER, INTER / 4}) {
        DevLayer L = upload(raws, nexp, ns == INTER ? 0 : ns, ns);
        for (int rows : {33, 128, 700, 2500})
        for (int tt : {64, 128}) {
            std::vector<float> x; gen_x(x, rows, rng);
            std::vector<int> ids; std::vector<float> rw; gen_routes(ids, rw, rows, topk, nexp, rng);
            std::vector<__nv_bfloat16> xb(x.size()); for (size_t i = 0; i < x.size(); i++) xb[i] = __float2bfloat16(x[i]);
            __nv_bfloat16* dx; int* dids; float *drw, *dy; CK(cudaMalloc(&dx, xb.size() * 2)); CK(cudaMalloc(&dids, ids.size() * 4));
            CK(cudaMalloc(&drw, rw.size() * 4)); CK(cudaMalloc(&dy, (size_t)rows * DIM * 4));
            CK(cudaMemcpy(dx, xb.data(), xb.size() * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(drw, rw.data(), rw.size() * 4, cudaMemcpyHostToDevice));
            WideBufs W = alloc_wide(rows, topk, ns, nexp, tt);
            run_wide(L, W, dx, dids, drw, dy, rows, topk, nexp, ns == INTER, 0);
            CK(cudaDeviceSynchronize());
            std::vector<float> yw((size_t)rows * DIM); CK(cudaMemcpy(yw.data(), dy, yw.size() * 4, cudaMemcpyDeviceToHost));
            long badb = -1;
            if (ns != INTER) {
                // PK3 bf16 partials: y must equal the ascending-id f32 sum of bf16(each route's f32 partial) of the f32 run, bit for bit
                std::vector<float> o2((size_t)rows * topk * DIM); CK(cudaMemcpy(o2.data(), W.out2, o2.size() * 4, cudaMemcpyDeviceToHost));
                run_wide(L, W, dx, dids, drw, dy, rows, topk, nexp, true, 0);
                CK(cudaDeviceSynchronize());
                std::vector<float> yb((size_t)rows * DIM); CK(cudaMemcpy(yb.data(), dy, yb.size() * 4, cudaMemcpyDeviceToHost));
                badb = 0;
                for (int r = 0; r < rows; r++) {
                    std::vector<int> ord(topk); for (int k = 0; k < topk; k++) ord[k] = k;
                    std::sort(ord.begin(), ord.end(), [&](int a2, int b2) { return ids[r * topk + a2] < ids[r * topk + b2]; });
                    for (int c = 0; c < DIM; c++) {
                        float acc = 0.0f;
                        for (int k : ord) acc += bf16r(o2[((size_t)r * topk + k) * DIM + c]);
                        badb += memcmp(&acc, &yb[(size_t)r * DIM + c], 4) != 0;
                    }
                }
            }
            // grouped path, 32 rows at a time
            Bufs B{};
            CK(cudaMalloc(&B.xq, 32 * DIM)); CK(cudaMalloc(&B.xs, 32 * ldxs_of())); CK(cudaMalloc(&B.hq, 32 * topk * ns)); CK(cudaMalloc(&B.hs, 32 * topk * ldhs_of(ns)));
            long bad = 0;
            for (int r0 = 0; r0 < rows; r0 += 32) {
                int n = std::min(32, rows - r0);
                B.x = dx + (size_t)r0 * DIM; B.ids = dids + r0 * topk; B.rw = drw + r0 * topk; B.y = dy + (size_t)r0 * DIM;
                run_grouped(L, B, n, topk, nexp, false, ns == INTER, 0);
            }
            CK(cudaDeviceSynchronize());
            std::vector<float> yg((size_t)rows * DIM); CK(cudaMemcpy(yg.data(), dy, yg.size() * 4, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < yg.size(); i++) bad += memcmp(&yg[i], &yw[i], 4) != 0;
            printf("wide (tile %3d) vs grouped: %dx%d slice %4d rows %4d: %ld of %ld differ  %s\n", tt, nexp, topk, ns, rows, bad, (long)rows * DIM, bad ? "FAIL" : "OK");
            fails += bad != 0;
            if (badb >= 0) {
                printf("wide (tile %3d) bf16 partials vs host sum of bf16(f32 partials): slice %4d rows %4d: %ld of %ld differ  %s\n", tt, ns, rows, badb,
                       (long)rows * DIM, badb ? "FAIL" : "OK");
                fails += badb != 0;
            }
            cudaFree(B.xq); cudaFree(B.xs); cudaFree(B.hq); cudaFree(B.hs); free_wide(W);
            cudaFree(dx); cudaFree(dids); cudaFree(drw); cudaFree(dy);
        }
        cudaFree(L.w13); cudaFree(L.w2);
    }
    printf(fails ? "KN2_WIDE_FAIL (%d)\n" : "KN2_WIDE_OK\n", fails);
    return fails ? 1 : 0;
}

static int g_only_slice = 0, g_only_rows = 0;   // PK2: --slice N / --rows M restrict --bench-wide
static int bench_wide(int nexp, int topk, int reps) {
    std::mt19937 rng(99);
    std::vector<RawExpert> raws(8); for (auto& r : raws) gen_raw(r, rng);
    printf("| experts x top-k | slice (TP) | rows | FC1 ms | FC2 + combine ms | total ms | rows/s (one layer) | TFLOP/s | GB/s of weights (once) |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n");
    const int slices[] = {INTER, INTER / 2, INTER / 4, INTER / 8};
    const char* tp[] = {"2304 (TP1)", "1152 (TP2)", "576 (TP4)", "288 (TP8)"};
    for (int si = 0; si < 4; si++) {
        int ns = slices[si];
        if (g_only_slice && ns != g_only_slice) continue;
        DevLayer L = upload(raws, nexp, 0, ns);
        for (int rows : {512, 1024, 2048, 3072, 4096, 6144, 8192, 16384}) {
            if (g_only_rows && rows != g_only_rows) continue;
            std::vector<float> x; gen_x(x, rows, rng);
            std::vector<int> ids; std::vector<float> rw; gen_routes(ids, rw, rows, topk, nexp, rng);
            std::vector<__nv_bfloat16> xb(x.size()); for (size_t i = 0; i < x.size(); i++) xb[i] = __float2bfloat16(x[i]);
            __nv_bfloat16* dx; int* dids; float *drw, *dy; CK(cudaMalloc(&dx, xb.size() * 2)); CK(cudaMalloc(&dids, ids.size() * 4));
            CK(cudaMalloc(&drw, rw.size() * 4)); CK(cudaMalloc(&dy, (size_t)rows * DIM * 4));
            CK(cudaMemcpy(dx, xb.data(), xb.size() * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(drw, rw.data(), rw.size() * 4, cudaMemcpyHostToDevice));
            const int tt = tt_of(rows, topk, nexp);
            const bool bf = ns == INTER || g_bf16;
            WideBufs W = alloc_wide(rows, topk, ns, nexp, tt);
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            double ms3[3];
            int rp = std::max(2, reps * 512 / rows / 8);
            for (int ph = 0; ph < 3; ph++) {
                run_wide(L, W, dx, dids, drw, dy, rows, topk, nexp, bf, 0, ph);
                CK(cudaDeviceSynchronize());
                cudaEventRecord(e0);
                for (int i = 0; i < rp; i++) run_wide(L, W, dx, dids, drw, dy, rows, topk, nexp, bf, 0, ph);
                cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
                float ms; cudaEventElapsedTime(&ms, e0, e1); ms3[ph] = ms / rp;
            }
            double flop = (double)rows * topk * 3.0 * 2.0 * DIM * ns;
            double wbytes = (double)nexp * (L.e13 + L.e2);
            printf("| %d x %d | %s | %d (tile %d%s) | %.2f | %.2f | %.2f | %.0f | %.1f | %.0f |\n", nexp, topk, tp[si], rows, tt, bf && ns != INTER ? ", bf16" : "", ms3[1], ms3[2], ms3[0],
                   rows / (ms3[0] * 1e-3), flop / (ms3[0] * 1e-3) / 1e12, wbytes / (ms3[0] * 1e-3) / 1e9);
            free_wide(W); cudaFree(dx); cudaFree(dids); cudaFree(drw); cudaFree(dy);
        }
        cudaFree(L.w13); cudaFree(L.w2);
    }
    return 0;
}

// ---- byte-floor bench ----------------------------------------------------------------------------
static int bench(int nexp, int topk, int reps) {
    std::mt19937 rng(77);
    std::vector<RawExpert> raws(8); for (auto& r : raws) gen_raw(r, rng);
    printf("| experts x top-k | slice (TP) | rows | distinct experts | MB read | FC1 us | FC2 us | total us | GB/s | of 238 |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    const int slices[] = {INTER, INTER / 2, INTER / 4, INTER / 8};
    const char* tp[] = {"2304 (TP1)", "1152 (TP2)", "576 (TP4)", "288 (TP8)"};
    for (int si = 0; si < 4; si++) {
        int ns = slices[si];
        DevLayer L = upload(raws, nexp, 0, ns);
        Bufs B{};
        const int MR = 32, NSETS = 64;
        CK(cudaMalloc(&B.x, MR * DIM * 2)); CK(cudaMalloc(&B.xq, MR * DIM)); CK(cudaMalloc(&B.xs, MR * ldxs_of()));
        CK(cudaMalloc(&B.hq, MR * topk * ns)); CK(cudaMalloc(&B.hs, MR * topk * ldhs_of(ns))); CK(cudaMalloc(&B.y, MR * DIM * 4));
        int* ids_sets; float* rw_sets; CK(cudaMalloc(&ids_sets, NSETS * MR * topk * 4)); CK(cudaMalloc(&rw_sets, NSETS * MR * topk * 4));
        std::vector<float> x; gen_x(x, MR, rng);
        std::vector<__nv_bfloat16> xb(x.size()); for (size_t i = 0; i < x.size(); i++) xb[i] = __float2bfloat16(x[i]);
        CK(cudaMemcpy(B.x, xb.data(), xb.size() * 2, cudaMemcpyHostToDevice));
        for (int rows : {1, 2, 6, 12, 24}) {
            std::vector<int> all_ids; std::vector<float> all_w; double dist_sum = 0;
            for (int s = 0; s < NSETS; s++) {
                std::vector<int> ids; std::vector<float> rw; gen_routes(ids, rw, rows, topk, nexp, rng);
                // the verify rows of one lane (6 rows) share half their experts with the previous row
                // (T0 §5.3 assumes ~3 new experts per extra row); lanes are independent
                for (int r = 1; r < rows; r++) {
                    if (r % 6 == 0) continue;
                    std::vector<int> row;
                    for (int k = 0; k < topk / 2; k++) row.push_back(ids[(r - 1) * topk + topk / 2 + k]);
                    while ((int)row.size() < topk) { int e = rng() % nexp; if (std::find(row.begin(), row.end(), e) == row.end()) row.push_back(e); }
                    for (int k = 0; k < topk; k++) ids[r * topk + k] = row[k];
                }
                std::vector<int> u(ids); std::sort(u.begin(), u.end()); dist_sum += std::unique(u.begin(), u.end()) - u.begin();
                ids.resize(MR * topk, 0); rw.resize(MR * topk, 0.f);
                all_ids.insert(all_ids.end(), ids.begin(), ids.end()); all_w.insert(all_w.end(), rw.begin(), rw.end());
            }
            CK(cudaMemcpy(ids_sets, all_ids.data(), all_ids.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(rw_sets, all_w.data(), all_w.size() * 4, cudaMemcpyHostToDevice));
            double dist = dist_sum / NSETS;
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            double us3[3];
            for (int ph = 0; ph < 3; ph++) {
                for (int i = 0; i < 4; i++) { B.ids = ids_sets + (i % NSETS) * MR * topk; B.rw = rw_sets + (i % NSETS) * MR * topk; run_grouped(L, B, rows, topk, nexp, false, ns == INTER, 0, nullptr, ph); }
                CK(cudaDeviceSynchronize());
                cudaEventRecord(e0);
                for (int i = 0; i < reps; i++) { B.ids = ids_sets + (i % NSETS) * MR * topk; B.rw = rw_sets + (i % NSETS) * MR * topk; run_grouped(L, B, rows, topk, nexp, false, ns == INTER, 0, nullptr, ph); }
                cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
                float ms; cudaEventElapsedTime(&ms, e0, e1);
                us3[ph] = ms * 1000.0 / reps;
            }
            double us = us3[0];
            double mb = dist * (double)(L.e13 + L.e2) / 1e6;
            double gbs = mb / us * 1e3;
            printf("| %d x %d | %s | %d | %.1f | %.2f | %.1f | %.1f | %.1f | %.1f | %.2f |\n", nexp, topk, tp[si], rows, dist, mb, us3[1], us3[2], us, gbs, gbs / 238.0);
        }
        cudaFree(B.x); cudaFree(B.xq); cudaFree(B.xs); cudaFree(B.hq); cudaFree(B.hs); cudaFree(B.y); cudaFree(ids_sets); cudaFree(rw_sets);
        cudaFree(L.w13); cudaFree(L.w2);
    }
    return 0;
}

int main(int argc, char** argv) {
    int nexp = 384, topk = 6, reps = 200; bool do_check = false, do_bench = false, do_wide = false, do_bench_wide = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--check") do_check = true;
        else if (a == "--bench") do_bench = true;
        else if (a == "--check-wide") do_wide = true;
        else if (a == "--bench-wide") do_bench_wide = true;
        else if (a == "--verbose") getenv_dbg = true;
        else if (a == "--experts" && i + 1 < argc) nexp = atoi(argv[++i]);
        else if (a == "--topk" && i + 1 < argc) topk = atoi(argv[++i]);
        else if (a == "--reps" && i + 1 < argc) reps = atoi(argv[++i]);
        else if (a == "--slice" && i + 1 < argc) g_only_slice = atoi(argv[++i]);
        else if (a == "--rows" && i + 1 < argc) g_only_rows = atoi(argv[++i]);
        else if (a == "--tt" && i + 1 < argc) g_tt = atoi(argv[++i]);
        else if (a == "--bf16") g_bf16 = true;
        else { fprintf(stderr, "usage: %s --check | --bench [--experts N --topk K --reps R]\n", argv[0]); return 2; }
    }
    int rc = 0;
    if (do_check) rc |= check(nexp, topk);
    if (do_wide) rc |= check_wide(nexp, topk);
    if (do_bench) rc |= bench(nexp, topk, reps);
    if (do_bench_wide) rc |= bench_wide(nexp, topk, reps);
    return rc;
}
