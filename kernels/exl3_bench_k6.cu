// CF-P1g (4.05 bpw): the EXL3 kernel module with K = 6 added to every bit-width switch of exl3_bench.cu. Same kernels,
// same names, same signatures — the host loads THIS ptx instead of exl3_bench.ptx for a pack that stores a K = 6 module
// (exl3_bench::bench_ptx_path); every K <= 5 pack keeps the unmodified module.
#define EXL3_K6_BUILD 1
#include "exl3_bench.cu"
