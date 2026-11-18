//! Small kernel-argument helpers (`d`, `grid`, `fbits`, `u`, `fork_blocking_stream`) and the geometry-word packers
//! (`pack_nh`, `pack_bs`, `pack_stride_nkv`, `pack_qsa_geom`). The packers are release asserts: a field one past its
//! width would land in the neighbouring field, i.e. a wrong cache stride. Map: `mod.rs`.

use super::*;

/// Create the engine's compute stream as a BLOCKING stream (CU_STREAM_DEFAULT) rather than a
/// NonBlocking fork. This is the crux of MTP correctness on GB10.
///
/// cudarc 0.9.15 runs ALL of its memory operations on the device's legacy NULL stream (handle 0):
/// `alloc_zeros`/`memset_zeros` (cuMemAllocAsync/cuMemsetAsync), `htod_*`/`dtoh_*`, and
/// `CudaSlice::drop` → `cuMemFreeAsync`. Our kernels + cuBLAS + dtod copies run on this compute
/// stream. `fork_default_stream()` makes a NonBlocking stream, which by definition performs NO
/// implicit synchronization with the NULL stream — so a buffer produced by `cuMemAllocAsync` on the
/// NULL stream can be touched by a kernel on the compute stream *before the allocation is ordered*
/// (compute-sanitizer: "accessed before it is allocated"), and a `cuMemFreeAsync` on drop can recycle
/// memory a still-in-flight compute kernel is using. That cross-stream, stream-ordered-allocator race
/// is the source of the nondeterministic MTP corruption (it only bites the MTP loop because that path
/// churns fresh allocations — embed/verify/logits — between GPU-in-flight kernels; the pooled decode
/// path masks it because `Pool::get` force-syncs after each fresh alloc).
///
/// A BLOCKING stream implicitly synchronizes with the NULL stream in BOTH directions: a kernel on it
/// waits for prior NULL-stream work (allocs/memsets/htod), and NULL-stream work (frees/dtoh) waits for
/// prior kernels on it. That establishes the exact stream-order edges the async allocator requires,
/// eliminating the race with zero hot-path changes. There is no concurrency loss: the single-lane
/// decode/MTP loop never overlaps the two streams anyway.
pub fn fork_blocking_stream(dev: &std::sync::Arc<CudaDevice>) -> cudarc::driver::CudaStream {
    use cudarc::driver::result::stream::{create, destroy, StreamKind};
    let mut s = dev.fork_default_stream().expect("fork stream");
    unsafe {
        destroy(s.stream).expect("destroy nonblocking stream");
        s.stream = create(StreamKind::Default).expect("create blocking stream");
    }
    s
}

pub(super) fn d<T>(s: &cudarc::driver::CudaSlice<T>) -> u64 { *s.device_ptr() }
pub(super) fn grid(n: usize) -> (u32, u32, u32) { (((n + 255) / 256) as u32, 1, 1) }
pub(super) fn fbits(x: f32) -> u64 { x.to_bits() as u64 }
pub(super) fn u(x: usize) -> u64 { x as u64 }

/// `nh_packed` = nkv(bits 0-9) | hd(10-19) | nh(20-30); gqa_attn_splitk / reduce / sel_prefill unpack
/// `>> 20`, `(>> 10) & 0x3FF`, `& 0x3FF` from a signed int (nh < 2048 keeps the sign bit clear).
pub(crate) fn pack_nh(nh: usize, hd: usize, nkv: usize) -> i32 {
    assert!(nh < 2048 && hd <= 1023 && nkv <= 1023,
            "nh_packed field overflow (nh {nh} < 2048, hd {hd} <= 1023, nkv {nkv} <= 1023)");
    ((nh << 20) | (hd << 10) | nkv) as i32
}

/// `bs_packed` = stride(bits 0-18) | ns_grid(19-24) | batch(25-30) | q_pitch(31-49), a u64 (the
/// splitk family unpacks `& 0x7FFFF`, `(>> 19) & 0x3F`, `(>> 25) & 0x3F`, `(>> 31) & 0x7FFFF`).
/// `stride` is the KV cache's positions per slot, i.e. the served window.
pub(crate) fn pack_bs(stride: usize, ns_grid: usize, batch: usize, q_pitch: usize) -> u64 {
    assert!(stride < (1 << 19) && ns_grid < (1 << 6) && batch < (1 << 6) && q_pitch < (1 << 19),
            "bs_packed field overflow (stride {stride} and q_pitch {q_pitch} < 2^19, ns_grid {ns_grid} and \
             batch {batch} < 2^6) - lower --max-seq-len");
    ((q_pitch as u64) << 31) | ((batch as u64) << 25) | ((ns_grid as u64) << 19) | (stride as u64)
}

/// `stride_nkv` = kv_stride(bits 0-18) | nkv(19-31) for rmsnorm_rope_kvwrite_b / _q4_b (`& 0x7FFFF`,
/// `(unsigned) >> 19`). `kv_stride` is the KV cache's positions per slot.
pub(crate) fn pack_stride_nkv(nkv: usize, kv_stride: usize) -> u32 {
    assert!(kv_stride < (1 << 19) && nkv < (1 << 13),
            "stride_nkv field overflow (kv_stride {kv_stride} < 2^19 cache positions per slot, nkv {nkv} < 2^13) \
             - lower --max-seq-len");
    ((nkv as u32) << 19) | (kv_stride as u32)
}

/// The QSA geometry word = stride(bits 0-19) | qk_dim(20-39) | nblk_stride(40-59); the qsa_* kernels
/// unpack three `& 0xFFFFF` fields (bits 60+ are kernel flags, OR-ed in by the caller). `stride` is the
/// key plane's positions per slot: the served window (`max_pos` on the EXL3 engine).
pub(crate) fn pack_qsa_geom(stride: usize, qk_dim: usize, nblk_stride: usize) -> u64 {
    assert!(stride < (1 << 20) && qk_dim < (1 << 20) && nblk_stride < (1 << 20),
            "qsa geom field overflow (stride {stride}, qk_dim {qk_dim}, nblk_stride {nblk_stride}: 20 bits each) \
             - lower --max-seq-len");
    (stride as u64) | ((qk_dim as u64) << 20) | ((nblk_stride as u64) << 40)
}
