// =============================================================================
// kernels/gpu_dsv41/01_abi_probe.cuh — band 01 (leg I0): the ABI probe kernels.
// `d41_kernel_build_id` is the stale-PTX handshake every loader runs; `d41_abi_probe` reports the
// ABI constants and struct sizes and decodes every E4M3 / E8M0 code and every E2M1 nibble, so the
// host can check the device and the Rust mirror (src/dsv41/gpu/abi.rs) agree bit for bit.
// =============================================================================

extern "C" __global__ void d41_kernel_build_id(unsigned long long* out) {
    D41_PDL_ENTRY();
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = KERNEL_BUILD_ID;
}

// u32 out[18]: [0] ABI version, [1..7] struct sizes, [8..15] format tags / row-class bounds,
// [16] SWA ring slots, [17] backtracking rows.
// f32 dec[256 + 256 + 16]: E4M3 codes 0..255, E8M0 codes 0..255, E2M1 codes 0..15 (one per thread).
extern "C" __global__ void d41_abi_probe(unsigned* out, float* dec) {
    D41_PDL_ENTRY();
    const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t == 0) {
        out[0] = D41_ABI_VERSION;
        out[1] = (unsigned)sizeof(d41_kv_main_row);
        out[2] = (unsigned)sizeof(d41_index_k_q);
        out[3] = (unsigned)sizeof(d41_index_k_s);
        out[4] = (unsigned)sizeof(d41_swa_q);
        out[5] = (unsigned)sizeof(d41_swa_s);
        out[6] = (unsigned)sizeof(d41_engram_row);
        out[7] = (unsigned)D41_SMEM_OPTIN_MAX;
        out[8] = D41_FMT_BF16;
        out[9] = D41_FMT_FP8_B32X32_UE8M0;
        out[10] = D41_FMT_FP8_B1X32_UE8M0;
        out[11] = D41_FMT_MXFP4;
        out[12] = D41_FMT_NVFP4;
        out[13] = D41_FMT_EXL3(3);
        out[14] = D41_GEMV_MAX_ROWS;
        out[15] = D41_COOP_MAX_ROWS;
        out[16] = D41_SWA_SLOTS;
        out[17] = D41_BACKTRACK;
    }
    if (t < 256u) {
        dec[t] = d41_e4m3(t);
        dec[256u + t] = d41_e8m0(t);
    }
    if (t < 16u) dec[512u + t] = d41_e2m1(t);
}
