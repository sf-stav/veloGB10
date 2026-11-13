//! EMB1: every PTX module the engine can load, embedded in the binary at compile time.
//!
//! build.rs compiles `kernels/*.cu` into `src/ptx/*.ptx` BEFORE the crate compiles; the
//! `include_str!` targets below embed those bytes into the executable, so a deploy is ONE
//! file and no serving box needs `src/ptx/` on disk (no CWD dependence).
//!
//! Freshness: `include_str!` targets enter rustc's dep-info, so a `.cu` edit -> build.rs
//! rerun -> new PTX bytes -> the crate recompiles. The embedded modules can never be stale
//! against the binary that carries them. The KERNEL_BUILD_ID handshake stays as the runtime
//! guard: build.rs bakes the same stamp into the PTX (`-DKERNEL_BUILD_ID`) and the binary
//! (`env!("KERNEL_BUILD_ID")`), and `assert_kernel_build_id` / `Dsv4Kernels::load` refuse a
//! mismatch loudly -- which, with embedding, is only possible across a mixed head/node.
//!
//! Module-identity note (S-B9-CF-R16): several paths load the SAME PTX text as separate
//! CUmodules on purpose (the raw-module copy that can carry dynamic-smem opt-ins / PDL
//! launches vs the cudarc copy). The consts below are just bytes: every loader still builds
//! its own module from them, so per-function attributes keep affecting only the copy they
//! were set on. Do not "share" a CUmodule across loader sites.

/// `gpu_batch.cu` (sm_121): the trunk batch kernels, TurboQuant b=2 golden-anchored build
/// (the `[kv-tq]=1` default).
pub const GPU_BATCH: &str = include_str!("ptx/gpu_batch.ptx");
/// `gpu_batch.cu -DTQ_B3` (sm_121): same kernel names, 3-bit K codes / 68-B TQ rows,
/// selected by `--kv-tq=3`. Same KERNEL_BUILD_ID (same .cu bytes).
pub const GPU_BATCH_B3: &str = include_str!("ptx/gpu_batch_b3.ptx");
/// `gpu_kernels.cu` (sm_121): per-token decode kernels (Qwen trunk, DFlash2, DSpark).
pub const GPU_KERNELS: &str = include_str!("ptx/gpu_kernels.ptx");
/// `gpu_dsv4.cu` (sm_121): DSV4 spine (raw-module load with the build-id stamp assert).
pub const GPU_DSV4: &str = include_str!("ptx/gpu_dsv4.ptx");
/// `gpu_dsv4_attn.cu` (sm_121): DSV4 lane-3A/3C attention kernels.
pub const GPU_DSV4_ATTN: &str = include_str!("ptx/gpu_dsv4_attn.ptx");
/// `gpu_dsv4_comp.cu` (sm_121): DSV4 compressor/indexer kernels.
pub const GPU_DSV4_COMP: &str = include_str!("ptx/gpu_dsv4_comp.ptx");
/// `gpu_dflash.cu` (sm_121): DFlash kernels.
pub const GPU_DFLASH: &str = include_str!("ptx/gpu_dflash.ptx");
/// `gpu_dspark.cu` (sm_121): DSpark ring-attention kernels.
pub const GPU_DSPARK: &str = include_str!("ptx/gpu_dspark.ptx");
/// `gpu_vision.cu` (sm_121): FP32 vision-tower kernels.
pub const GPU_VISION: &str = include_str!("ptx/gpu_vision.ptx");
/// `gpu_fp4l.cu` (sm_121a): FP4L weight kernels (MXFP8 / f16 / NVFP4 experts) for the EXL3 trunk,
/// loaded only for an nvidia/Qwen3.8-Flash-Next-FP4-Local pack (src/exl3_forward/fp4l_fwd.rs).
pub const GPU_FP4L: &str = include_str!("ptx/gpu_fp4l.ptx");
/// `mxfp4_bench.cu` (sm_121a): standalone OMMA microbench (probe-only, lazy).
pub const MXFP4_BENCH: &str = include_str!("ptx/mxfp4_bench.ptx");
/// `gpu_mxfp4.cu` (sm_121a): NVFP4 native GEMV kernels.
pub const GPU_MXFP4: &str = include_str!("ptx/gpu_mxfp4.ptx");
/// `gpu_mxfp4_moe.cu` (sm_121a): NVFP4 MoE grouped/slot GEMVs.
pub const GPU_MXFP4_MOE: &str = include_str!("ptx/gpu_mxfp4_moe.ptx");
/// `gpu_w4a4.cu` (sm_121a): NVFP4 W4A4 prefill kernels.
pub const GPU_W4A4: &str = include_str!("ptx/gpu_w4a4.ptx");
/// `exl3_bench.cu` (sm_121): EXL3 weight-path GEMM -- the SERVING module of the EXL3
/// trunk (raw copy for xq_* / PDL + cudarc copy), not probe-only.
pub const EXL3_BENCH: &str = include_str!("ptx/exl3_bench.ptx");
/// `gpu_naive.cu` (sm_121): the Naive-N0.5-Flash family kernels (half-K trellis GEMM, ...; PLAN/NAIVE_N05_PORT.md).
pub const GPU_NAIVE: &str = include_str!("ptx/gpu_naive.ptx");
/// `gpu_naive_draft.cu` (sm_121): the Naive DSpark drafter round (src/naive/drafter_gpu.rs).
pub const GPU_NAIVE_DRAFT: &str = include_str!("ptx/gpu_naive_draft.ptx");

