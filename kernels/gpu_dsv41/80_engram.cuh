// =============================================================================
// kernels/gpu_dsv41/80_engram.cuh — band 80 (leg EIO): the Engram row runtime on the GPU.
// Host side: src/dsv41/gpu/engram.rs (control block / descriptor mirrors, launches, the CPU twin).
//
// What arrives here: the host reader (src/dsv41/engram/{io,cache,prefetch}.rs) has put the 264-B
// records (256 E4M3 values + 8 E8M0 scales, ENGb's pack format) of one step into slots of the
// pinned, device-mapped row cache, written the step's slot table into a descriptor ring slot, and
// bumped the channel's published sequence word (st.release from the host).
//
// d41_engram_step_decode is the bounded in-stream wait plus the decode, graph-capture safe: no
// per-step value is a kernel argument. The step it consumes is the channel's device counter
// (dev[0], advanced by the kernel's last block), so a captured graph replays the next step on every
// launch. It waits (one poller per block, relaxed .sys loads + backoff, one acquire fence) until
// the host has published that step or the timeout passes; a step that had to wait is a STALL
// (counted, its wait in ns summed); a timeout or a host error status is FAIL-STOP: the rows are not
// written, the error code lands in the mapped control block and the step is consumed (the host
// fails the round; nothing ever decodes an unwritten slot).
//
// Row decode = the reference's lookup (model.py ParallelEngramEmbedding.forward):
//   bf16( float(e4m3[i]) * float(e8m0[i / 32]) )   for i in 0..256,
// the product in f32 (exact: a power-of-two scale), one round-to-nearest-even to bf16.
// One warp per row, lane L owns values 8L..8L+7 (one 8-B load, one 16-B store), fixed order.
//
// Output (mode bit 0): bf16 rows at out + ((l * max_tok + t) * n_own + j) * 256, j = the rank's
// owned column in ascending column order (c = g * n_heads + h). That is the K-slice input of the
// layer's `wkv` (partial-sum plug-in). Mode bit 1 also copies the raw records to
// out_rec + ((l * max_tok + t) * n_own + j) * 264 (the gather plug-in: records cross the wire,
// the row owner decodes them with d41_engram_records_decode).
// =============================================================================
#pragma once

#define D41_ENG_ROW_VALUES 256
#define D41_ENG_REC_BYTES 264
#define D41_ENG_SLOT_STRIDE 272u        // row-cache slot stride (Rust: gpu::engram::SLOT_STRIDE; 264 B record, 16-B aligned)
#define D41_ENG_ZERO_SLOT 0xFFFFFFFFu   // DEAD position (gate 0): the row is written as zeros, nothing is read
#define D41_ENG_ERR_TIMEOUT 1u
#define D41_ENG_ERR_HOST 2u            // the host published the step with a non-zero status (I/O error)
#define D41_ENG_ERR_SEQ 3u             // descriptor sequence != the device counter (protocol bug)
#define D41_ENG_ERR_SHAPE 4u           // n_tok / n_own outside the launch's max_tok / max_own

// Host-mapped control block of one channel (Rust mirror: gpu::engram::Ctrl). Each word the host
// and the device both touch sits on its own 64-B line.
struct d41_engram_ctrl {
    unsigned long long pub_seq;        // host -> device: steps published (st.release by the host)
    unsigned long long pad0[7];
    unsigned long long done_seq;       // device -> host: steps consumed (decoded or failed)
    unsigned long long pad1[7];
    unsigned long long stalls;         // steps whose publish came after the kernel reached the wait
    unsigned long long stall_ns;       // total ns waited by those steps
    unsigned long long max_stall_ns;
    unsigned long long last_wait_ns;   // the latest step's wait (0 = no stall)
    unsigned long long timeouts;
    unsigned long long host_errs;
    unsigned int err;                  // first error code (sticky until the host clears it)
    unsigned int err_seq_lo;           // low 32 bits of the step that set it
    unsigned long long last_decode_ns; // the latest step: block 0 saw the publish -> the last block finished
    unsigned long long sum_decode_ns;
    unsigned long long max_decode_ns;
    unsigned long long steps;          // steps consumed
    unsigned long long pad2[5];
};

// Descriptor header (one per ring slot; the slot table of u32 follows it, n_rows = 2 * n_tok * n_own entries,
// order (l, t, j)). Rust mirror: gpu::engram::DescHead.
struct d41_engram_desc {
    unsigned long long seq;            // the step this slot carries
    unsigned int n_tok;                // T fed positions of this step (all lanes)
    unsigned int n_own;                // owned columns per Engram layer
    unsigned int n_layers;             // Engram layers (2)
    unsigned int status;               // 0 = rows ready; else the host's error code
    unsigned int pad[2];
};

__device__ __forceinline__ unsigned long long d41_eng_ld_relaxed(const unsigned long long* p) {
    unsigned long long v;
    asm volatile("ld.relaxed.sys.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void d41_eng_st_release(unsigned long long* p, unsigned long long v) {
    asm volatile("st.release.sys.b64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ unsigned long long d41_eng_now() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// One warp decodes one 264-B record into 256 bf16 (lane L: values 8L..8L+7).
// Plain loads. The slots and descriptors are host-written mapped memory that an SM may have read for an
// earlier step (slot reuse after eviction, ring-slot reuse). Visibility comes from the acquire pattern, as
// in the TP transport: thread 0 observes the published sequence (ld.relaxed.sys), executes fence.acquire.sys
// (which also discards the SM's L1), and bar.sync orders every thread's later loads after it; GB10's L2 is
// coherent with the CPU over C2C. ld.global.cv was measured 3x slower for the same rows (144 rows:
// 38 us vs 12 us, .14, 2026-11-18) and adds nothing to this ordering.
__device__ __forceinline__ void d41_eng_decode_row(const unsigned char* rec, __nv_bfloat16* dst, unsigned lane) {
    const uint2 q = *reinterpret_cast<const uint2*>(rec + 8u * lane);
    const unsigned sw = *reinterpret_cast<const unsigned*>(rec + D41_ENG_ROW_VALUES + 4u * (lane >> 4));
    const float sc = d41_e8m0((sw >> (8u * ((lane >> 2) & 3u))) & 0xFFu);
    __align__(16) __nv_bfloat16 o[8];
#pragma unroll
    for (int k = 0; k < 4; ++k) o[k] = __float2bfloat16_rn(d41_e4m3((q.x >> (8 * k)) & 0xFFu) * sc);
#pragma unroll
    for (int k = 0; k < 4; ++k) o[4 + k] = __float2bfloat16_rn(d41_e4m3((q.y >> (8 * k)) & 0xFFu) * sc);
    *reinterpret_cast<uint4*>(dst + 8u * lane) = *reinterpret_cast<const uint4*>(o);
}
__device__ __forceinline__ void d41_eng_zero_row(__nv_bfloat16* dst, unsigned lane) {
    *reinterpret_cast<uint4*>(dst + 8u * lane) = make_uint4(0u, 0u, 0u, 0u);
}
// One warp copies one 264-B record (33 x 8 B; 8-B aligned source slots and destinations).
__device__ __forceinline__ void d41_eng_copy_rec(const unsigned char* rec, unsigned char* dst, unsigned lane) {
    const unsigned long long* s = reinterpret_cast<const unsigned long long*>(rec);
    unsigned long long* d = reinterpret_cast<unsigned long long*>(dst);
    d[lane] = s[lane];
    if (lane == 0) d[32] = s[32];
}

// ctrl: the channel's mapped control block (device pointer of the host mapping).
// dev: device-memory words of the channel: [0] = next step to consume, [1] = block arrival counter,
//      [2] = this step's wait ns (block 0 writes, the last block reports), [3] = sticky error of this step,
//      [4] = global time block 0 saw the publish (the decode-duration start).
// desc_base / desc_stride: the mapped descriptor ring (ring slots of desc_stride bytes, slot = seq % ring).
// cache: the mapped row cache (slots of D41_ENG_SLOT_STRIDE bytes). out / out_rec: see the header comment (mode bits 0 / 1).
// max_tok / max_own: the launch's output geometry (the wkv input buffer). timeout_ns: the bounded wait.
extern "C" __global__ void __launch_bounds__(256) d41_engram_step_decode(
    d41_engram_ctrl* ctrl, unsigned long long* dev, const unsigned char* desc_base, unsigned desc_stride, unsigned ring,
    const unsigned char* cache, __nv_bfloat16* out, unsigned char* out_rec,
    unsigned max_tok, unsigned max_own, unsigned mode, unsigned long long timeout_ns) {
    D41_PDL_WAIT_ONLY();
    __shared__ int s_ok;
    __shared__ unsigned long long s_seq;
    __shared__ unsigned s_hdr[4];        // n_tok, n_own, n_layers, error code (one uncached read per block)
    const unsigned tid = threadIdx.x;
    if (tid == 0) {
        const unsigned long long seq = dev[0];
        const unsigned long long t0 = d41_eng_now();
        unsigned long long t = t0;
        int ok = 1;
        unsigned ns = 32;
        while (d41_eng_ld_relaxed(&ctrl->pub_seq) <= seq) {
            t = d41_eng_now();
            if (t - t0 > timeout_ns) { ok = 0; break; }
            if (t - t0 > 2000ull) { __nanosleep(ns); if (ns < 256u) ns <<= 1; }
        }
        asm volatile("fence.acquire.sys;" ::: "memory");
        if (blockIdx.x == 0) {
            const unsigned long long seen = d41_eng_now();
            dev[2] = (t == t0) ? 0ull : (seen - t0);
            dev[4] = seen;
        }
        unsigned code = 0;
        if (!ok) {
            code = D41_ENG_ERR_TIMEOUT;
        } else {
            const d41_engram_desc* d = reinterpret_cast<const d41_engram_desc*>(desc_base + (size_t)(seq % ring) * desc_stride);
            const unsigned status = d->status;
            s_hdr[0] = d->n_tok;
            s_hdr[1] = d->n_own;
            s_hdr[2] = d->n_layers;
            if (status != 0u) code = D41_ENG_ERR_HOST;
            else if (d->seq != seq) code = D41_ENG_ERR_SEQ;
            else if (s_hdr[0] > max_tok || s_hdr[1] > max_own || s_hdr[2] == 0u) code = D41_ENG_ERR_SHAPE;
        }
        if (code) atomicCAS(reinterpret_cast<unsigned int*>(&dev[3]), 0u, code);
        s_hdr[3] = code;
        s_ok = ok;
        s_seq = seq;
    }
    __syncthreads();
    D41_PDL_RELEASE();
    const unsigned long long seq = s_seq;
    if (s_ok && s_hdr[3] == 0u) {
        const unsigned n_tok = s_hdr[0], n_own = s_hdr[1];
        const unsigned n_rows = s_hdr[2] * n_tok * n_own;
        const unsigned* slots = reinterpret_cast<const unsigned*>(desc_base + (size_t)(seq % ring) * desc_stride + sizeof(d41_engram_desc));
        const unsigned lane = tid & 31u, warp = tid >> 5, wpb = blockDim.x >> 5;
        for (unsigned r = blockIdx.x * wpb + warp; r < n_rows; r += gridDim.x * wpb) {
            const unsigned l = r / (n_tok * n_own), rem = r % (n_tok * n_own);
            const unsigned t = rem / n_own, j = rem % n_own;
            const size_t o = ((size_t)l * max_tok + t) * max_own + j;
            unsigned s = 0;
            if (lane == 0) s = slots[r];               // one load per row, broadcast
            s = __shfl_sync(0xFFFFFFFFu, s, 0);
            if (s == D41_ENG_ZERO_SLOT) {
                if (mode & 1u) d41_eng_zero_row(out + o * D41_ENG_ROW_VALUES, lane);
                if (mode & 2u) {
                    unsigned long long* z = reinterpret_cast<unsigned long long*>(out_rec + o * D41_ENG_REC_BYTES);
                    z[lane] = 0ull;
                    if (lane == 0) z[32] = 0ull;
                }
            } else {
                const unsigned char* rec = cache + (size_t)s * D41_ENG_SLOT_STRIDE;
                if (mode & 1u) d41_eng_decode_row(rec, out + o * D41_ENG_ROW_VALUES, lane);
                if (mode & 2u) d41_eng_copy_rec(rec, out_rec + o * D41_ENG_REC_BYTES, lane);
            }
        }
    }
    // the last block to arrive advances the channel and reports
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        const unsigned prev = atomicAdd(reinterpret_cast<unsigned int*>(&dev[1]), 1u);
        if (prev == gridDim.x - 1u) {
            __threadfence();
            const unsigned long long w = *(volatile unsigned long long*)&dev[2];
            const unsigned e = *(volatile unsigned int*)&dev[3];
            const unsigned long long dn = d41_eng_now() - *(volatile unsigned long long*)&dev[4];
            ctrl->last_wait_ns = w;
            if (w) { ctrl->stalls += 1ull; ctrl->stall_ns += w; if (w > ctrl->max_stall_ns) ctrl->max_stall_ns = w; }
            ctrl->last_decode_ns = dn;
            ctrl->sum_decode_ns += dn;
            if (dn > ctrl->max_decode_ns) ctrl->max_decode_ns = dn;
            ctrl->steps += 1ull;
            if (e == D41_ENG_ERR_TIMEOUT) ctrl->timeouts += 1ull;
            if (e == D41_ENG_ERR_HOST) ctrl->host_errs += 1ull;
            if (e && ctrl->err == 0u) { ctrl->err = e; ctrl->err_seq_lo = (unsigned)seq; }
            dev[1] = 0ull;
            dev[2] = 0ull;
            dev[3] = 0ull;
            dev[0] = seq + 1ull;
            __threadfence_system();
            d41_eng_st_release(&ctrl->done_seq, seq + 1ull);
        }
    }
}

// Decode n contiguous records (rec + i * rec_stride) into out + i * 256 bf16 (the gather plug-in's
// row-owner side, and the identity probe). One warp per row.
extern "C" __global__ void __launch_bounds__(256) d41_engram_records_decode(
    const unsigned char* rec, unsigned rec_stride, unsigned n, __nv_bfloat16* out) {
    D41_PDL_ENTRY();
    const unsigned lane = threadIdx.x & 31u, wpb = blockDim.x >> 5;
    for (unsigned r = blockIdx.x * wpb + (threadIdx.x >> 5); r < n; r += gridDim.x * wpb)
        d41_eng_decode_row(rec + (size_t)r * rec_stride, out + (size_t)r * D41_ENG_ROW_VALUES, lane);
}

// Bench helper (the synthetic decode of `--probe-dsv41-engram-io`): occupy the stream for ns
// nanoseconds (one thread spins on the global timer), then, if flag != 0, store value to the mapped
// word *flag with release semantics (the "round end: the next tokens are known" doorbell).
extern "C" __global__ void d41_engram_bench_spin(unsigned long long ns, unsigned long long* flag, unsigned long long value) {
    D41_PDL_ENTRY();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const unsigned long long t0 = d41_eng_now();
        while (d41_eng_now() - t0 < ns) { }
        if (flag) { __threadfence_system(); d41_eng_st_release(flag, value); }
    }
}
