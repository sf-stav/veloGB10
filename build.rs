use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::process::Command;

/// Compile every kernel the binary actually loads at runtime, and FAIL LOUDLY if nvcc does.
///
/// This file used to compile four .cu files that nothing loads (gemm_nvfp4, fused_decode, rms_norm,
/// silu_gate) while skipping the only two that `GpuModel::load` actually reads -- gpu_batch.ptx and
/// gpu_kernels.ptx. So `cargo build` did not rebuild the main kernel file. A .cu edit was invisible
/// until someone remembered to run nvcc by hand, and the resulting binary ran OLD KERNELS against NEW
/// launch parameters. That is not a theoretical hazard: it cost a debugging cycle on the day this was
/// written, when a rewritten kernel that uses no shared memory was launched with smem=0 while the
/// stale PTX still contained the version that writes to it -- CUDA_ERROR_ILLEGAL_ADDRESS, and the
/// deployed stable_build was silently running pre-fix kernels.
///
/// It also swallowed nvcc failures with an eprintln and let the build go green on the stale PTX.
/// A kernel that does not compile is now a build error, as it should always have been.
fn main() {
    println!("cargo:rustc-env=CUDA_ARCH=sm_121");
    println!("cargo:rustc-env=CUDA_ARCHITECTURES=121");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cudart");

    let cuda_home = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    println!("cargo:include={}/include", cuda_home);

    // ---- TP=2 comm transport shim (native/net_shim.c): flat C ABI over libibverbs + cudaHostAlloc ----
    // Keeps ibverbs struct complexity in C; src/net.rs is a thin FFI. Links libibverbs (data plane) and
    // cudart (already linked) for cudaHostAlloc. Compiled into the crate; a compile error fails the build.
    println!("cargo:rerun-if-changed=native/net_shim.c");
    println!("cargo:rerun-if-changed=native/tp_doorbell.h");
    cc::Build::new()
        .file("native/net_shim.c")
        .include("native")
        .include(format!("{cuda_home}/include"))
        .flag("-O3")
        .warnings(false)
        .compile("net_shim");
    println!("cargo:rustc-link-lib=ibverbs");

    let nvcc = PathBuf::from(&cuda_home).join("bin/nvcc");
    let kernel_dir = PathBuf::from("kernels");
    let out_dir = PathBuf::from("src/ptx");
    let _ = std::fs::create_dir_all(&out_dir);

    // Exactly the modules gpu.rs loads. If you add a load_ptx, add it here.
    let kernels = [
        ("gpu_batch.cu", "gpu_batch.ptx"),
        ("gpu_kernels.cu", "gpu_kernels.ptx"),
    ];

    for (src_name, _) in &kernels {
        println!("cargo:rerun-if-changed={}", kernel_dir.join(src_name).display());
    }
    println!("cargo:rerun-if-changed=build.rs");

    // ---- Make a stale PTX STRUCTURALLY IMPOSSIBLE, not merely checked ----
    //
    // A deploy is three files (binary + two PTX) and the PTX is loaded from disk at runtime, so a
    // fresh binary can sit next to old kernels. That shipped once: new Rust launched an old kernel and
    // the result was CUDA_ERROR_ILLEGAL_ADDRESS from correct code. scripts/deploy.sh checks it, but a
    // procedural check only protects the paths it is run on.
    //
    // So: hash the exact .cu bytes we are about to compile, bake that ID into the PTX as a global, and
    // have GpuModel::load assert the PTX's ID equals the one compiled into the binary. Now a mismatched
    // pair fails LOUDLY at startup, on any box, however it was assembled -- including ones this repo's
    // deploy script never touched.
    let mut hasher = DefaultHasher::new();
    for (src_name, _) in &kernels {
        std::fs::read(kernel_dir.join(src_name))
            .unwrap_or_else(|e| panic!("cannot read kernels/{src_name}: {e}"))
            .hash(&mut hasher);
    }
    // gpu_batch.cu #includes native/tp_doorbell.h (the TP=2 wire layout, shared with net_shim.c). It is
    // NOT a .cu, so hashing only the kernel sources would let an offset change slip through with an
    // unchanged build ID -- exactly the stale-PTX failure this scheme exists to prevent.
    std::fs::read("native/tp_doorbell.h")
        .expect("cannot read native/tp_doorbell.h")
        .hash(&mut hasher);
    let build_id = format!("{:016x}", hasher.finish());
    println!("cargo:rustc-env=KERNEL_BUILD_ID={build_id}");

    // Rust-side sources change wire/compute behavior (weight sharders, the cluster protocol, the
    // scheduler) WITHOUT touching the kernels — so the KERNEL_BUILD_ID handshake can't see them.
    // That hole shipped a wrong model once: TP legs with NO_STAGE=1 after a Rust-only change ran a
    // stale peer binary past the version check (the "garbage token-11" legs, 2026-07-20). Cover
    // every src/*.rs + the C shim with a second stamp, appended to the handshake token.
    let mut shasher = DefaultHasher::new();
    let mut rust_files: Vec<PathBuf> = std::fs::read_dir("src")
        .expect("read_dir src")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().map(|x| x == "rs").unwrap_or(false))
        .collect();
    rust_files.sort();          // stable order → stable hash
    for f in &rust_files {
        println!("cargo:rerun-if-changed={}", f.display());
        std::fs::read(f).unwrap_or_else(|e| panic!("cannot read {}: {e}", f.display()))
            .hash(&mut shasher);
    }
    println!("cargo:rerun-if-changed=native/net_shim.c");
    std::fs::read("native/net_shim.c").expect("cannot read native/net_shim.c").hash(&mut shasher);
    let source_id = format!("{:016x}", shasher.finish());
    println!("cargo:rustc-env=SOURCE_BUILD_ID={source_id}");

    if !nvcc.exists() {
        // Cross-compiling or a CI box with no CUDA: the checked-in PTX has to do. Say so clearly --
        // a silent fallback here is exactly how a stale kernel ships.
        println!("cargo:warning=nvcc not found at {}; using the CHECKED-IN PTX in src/ptx/. \
                  Any kernels/*.cu edit will NOT take effect.", nvcc.display());
        return;
    }

    for (src_name, dst_name) in &kernels {
        let src = kernel_dir.join(src_name);
        let dst = out_dir.join(dst_name);
        if !src.exists() {
            panic!("{} is missing but the binary loads {}", src.display(), dst_name);
        }
        let out = Command::new(&nvcc)
            .args([
                "--gpu-architecture=sm_121",
                "--ptx",
                "--default-stream=per-thread",
                "-O3",
                "-Inative",
                &format!("-DKERNEL_BUILD_ID=0x{build_id}ULL"),
                "-o",
            ])
            .arg(&dst)
            .arg(&src)
            .output()
            .unwrap_or_else(|e| panic!("failed to run {}: {e}", nvcc.display()));

        if !out.status.success() {
            panic!(
                "nvcc failed on {} -- the build must NOT proceed on the stale {}:\n{}",
                src.display(),
                dst.display(),
                String::from_utf8_lossy(&out.stderr)
            );
        }
    }
}
