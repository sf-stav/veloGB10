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
    // mxfp4_bench.cu is the Phase-0 standalone OMMA microbench (probe-only, loaded lazily by
    // --probe-mxfp4): the mxf4nvf4 mma and cvt.e2m1x2 are sm_121a family-specific features that
    // plain sm_121 REJECTS, so it gets its own arch per file (the serving manifest stays sm_121
    // and byte-identical — the non-destruction contract).
    let kernels = [
        ("gpu_batch.cu", "gpu_batch.ptx", "sm_121"),
        ("gpu_kernels.cu", "gpu_kernels.ptx", "sm_121"),
        ("gpu_dsv4.cu", "gpu_dsv4.ptx", "sm_121"),
        ("gpu_dsv4_attn.cu", "gpu_dsv4_attn.ptx", "sm_121"),
        ("gpu_dsv4_comp.cu", "gpu_dsv4_comp.ptx", "sm_121"),
        ("gpu_dflash.cu", "gpu_dflash.ptx", "sm_121"),
        ("gpu_dspark.cu", "gpu_dspark.ptx", "sm_121"),
        ("gpu_vision.cu", "gpu_vision.ptx", "sm_121"),
        ("mxfp4_bench.cu", "mxfp4_bench.ptx", "sm_121a"),
        ("gpu_mxfp4.cu", "gpu_mxfp4.ptx", "sm_121a"),
        ("gpu_mxfp4_moe.cu", "gpu_mxfp4_moe.ptx", "sm_121a"),
        // PR veloGB10#4: NVFP4 W4A4 prefill kernels. Self-contained module (own helpers,
        // own kernel_build_id), sm_121a like the other mxf4nvf4 OMMA users.
        ("gpu_w4a4.cu", "gpu_w4a4.ptx", "sm_121a"),
        // S-A3-c: EXL3 weight-path GEMM (trellis decode fused into fp16 HMMA).
        // Probe/bench-only module — loaded lazily by --probe-exl3-kernel / --probe-exl3-binv /
        // --bench-exl3-gemm (mxfp4_bench precedent); NOT in the serving manifest.
        // sm_121: fp16 mma.m16n8k16 + dp4a are baseline CC 12.1.
        ("exl3_bench.cu", "exl3_bench.ptx", "sm_121"),
        // CF-P1g (4.05 bpw): the same source with a `case 6` in every bit-width switch (see the file).
        // NAIVE (PLAN/NAIVE_N05_PORT.md S3): the Naive-N0.5-Flash family module — copies of the EXL3
        // kernels with half-integer trellis rates; loaded by src/naive/gpu.rs only.
        ("gpu_naive.cu", "gpu_naive.ptx", "sm_121"),
        // NAIVE S8: the DSpark drafter round (plain f16 weights, small-M mma GEMM, Markov + confidence);
        // loaded by src/naive/drafter_gpu.rs only.
        ("gpu_naive_draft.cu", "gpu_naive_draft.ptx", "sm_121"),
        // FP4L (PLAN/FP4L_DESIGN.md): nvidia/Qwen3.8-Flash-Next-FP4-Local weight kernels (MXFP8 W8A16,
        // f16-weight, NVFP4 experts) for the EXL3 trunk; sm_121a for the e2m1x2 -> f16x2 cvt (SM12x family form).
        ("gpu_fp4l.cu", "gpu_fp4l.ptx", "sm_121a"),
    ];

    for (src_name, _, _) in &kernels {
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
    for (src_name, _, _) in &kernels {
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

    // ---- TUNE T0 (PLAN/AUTOTUNE_DESIGN.md §7.2): TUNE_BUILD_ID, the autotune table's build stamp ----
    // Narrower than SOURCE_BUILD_ID (every src/*.rs, so a server.rs edit would needlessly invalidate
    // a tuned table): a content hash over exactly the files that decide kernel code and decode-graph
    // composition. FNV-1a 64 over (name, len, bytes) — a stable algorithm, NOT std's DefaultHasher,
    // so a toolchain upgrade can never alias or silently change it. Missing files hash as absent.
    {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut feed = |b: &[u8]| {
            for &x in b {
                h ^= x as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        // T2 (review): + the files that decide which graphs are captured and how (the WP27 capture
        // DAG, the DHEAD draft tail, the served posture / precapture in exl3_serve, MODULE_FNS and
        // smem opt-ins in exl3_bench).
        for f in ["kernels/exl3_bench.cu", "kernels/gpu_batch.cu", "native/tp_doorbell.h",
                  "src/exl3_forward.rs", "src/exl3_tune.rs", "src/exl3.rs", "src/exl3_wp27.rs",
                  "src/exl3_forward/dhead.rs", "src/exl3_serve.rs", "src/exl3_bench.rs",
                  // T0c (p5e): the W4 packages' host sides (dispatch, grids, registry entries)
                  "src/exl3_forward/dense.rs", "src/exl3_forward/w4s.rs"] {
            println!("cargo:rerun-if-changed={f}");
            let bytes = std::fs::read(f).unwrap_or_default();
            feed(f.as_bytes());
            feed(&(bytes.len() as u64).to_le_bytes());
            feed(&bytes);
        }
        println!("cargo:rustc-env=TUNE_BUILD_ID={h:016x}");
        // Provenance only (receipts / table provenance.commit): the HEAD commit and dirty flag AS OF
        // THIS build-script run (cargo reruns it on any tracked source change, not on a bare commit).
        let git = |a: &[&str]| Command::new("git").args(a).output().ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
        // T2 (review): rerun on a bare commit / checkout too, so the stamp never cites a stale commit
        // (only paths that exist: a missing rerun-if-changed path would rerun on every build).
        let mut watch: Vec<String> = Vec::new();
        if let Some(p) = git(&["rev-parse", "--git-path", "HEAD"]) { watch.push(p); }
        if let Some(r) = git(&["symbolic-ref", "-q", "HEAD"]) {
            if let Some(p) = git(&["rev-parse", "--git-path", &r]) { watch.push(p); }
        }
        if let Some(p) = git(&["rev-parse", "--git-path", "packed-refs"]) { watch.push(p); }
        for p in watch {
            if std::path::Path::new(&p).exists() { println!("cargo:rerun-if-changed={p}"); }
        }
        let commit = git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into());
        let dirty = match git(&["status", "--porcelain", "--untracked-files=no"]) {
            Some(s) if s.is_empty() => "0",
            Some(_) => "1",
            None => "unknown",
        };
        println!("cargo:rustc-env=TUNE_GIT_COMMIT={commit}");
        println!("cargo:rustc-env=TUNE_GIT_DIRTY={dirty}");
    }

    // Rust-side sources change wire/compute behavior (weight sharders, the cluster protocol, the
    // scheduler) WITHOUT touching the kernels — so the KERNEL_BUILD_ID handshake can't see them.
    // That hole shipped a wrong model once: TP legs with NO_STAGE=1 after a Rust-only change ran a
    // stale peer binary past the version check (the "garbage token-11" legs, 2026-07-20). Cover
    // the engine's Rust sources + the C shim with a second stamp, appended to the handshake token.
    //
    // S-B9-CF-AGREE16B (W1): the stamp used to read the `.rs` FILES OF `src/` ONLY, so the 20 engine
    // files in subdirectories (src/exl3_forward/** with the EXL3 TP sharder and forward, src/dflash2/*,
    // src/dspark/*), the data baked in with include_str!/include_bytes!, Cargo.lock and this script
    // were outside it: a head and a node that differed only there paired and ran different code.
    // Now: every `.rs` under src/ at any depth, except src/ptx (this script writes it) and src/bin
    // (separate executables), plus the included data files, the manifests and this script. Each
    // file is hashed as (relative path, bytes) in sorted order, and watched PER FILE. `src/` itself
    // is never watched as a directory (this script writes src/ptx: it would rerun on every build);
    // the subdirectories are, so a file added to one reruns the script and joins the list.
    // The list is baked in for the unit test `cluster::build_id_tests`, which walks the tree itself
    // and fails on an engine source or an included file that is in neither stamp.
    let mut src_files: Vec<String> = Vec::new();
    let mut src_dirs: Vec<String> = Vec::new();
    let mut stack = vec![PathBuf::from("src")];
    while let Some(dir) = stack.pop() {
        for e in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display())).filter_map(|e| e.ok()) {
            let p = e.path();
            let rel = p.to_string_lossy().replace('\\', "/");
            if p.is_dir() {
                if rel == "src/ptx" || rel == "src/bin" { continue; }
                src_dirs.push(rel);
                stack.push(p);
            } else if p.is_file() && p.extension().map(|x| x == "rs").unwrap_or(false) {
                // `is_file()`: a dangling symlink or an editor lock file (`src/x/.#y.rs`) is not a source
                // and must not be read (the read below would panic the build).
                src_files.push(rel);
            }
        }
    }
    // the data compiled into the binary (include_str! / include_bytes! targets outside the kernels)
    for (dir, ext) in [("data/ep_deal", "json"), ("src/tq_consts", "bin")] {
        src_dirs.push(dir.to_string());
        for e in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {dir}: {e}")).filter_map(|e| e.ok()) {
            let p = e.path();
            if p.is_file() && p.extension().map(|x| x == ext).unwrap_or(false) {
                src_files.push(p.to_string_lossy().replace('\\', "/"));
            }
        }
    }
    for f in ["native/net_shim.c", "Cargo.lock", "Cargo.toml", "build.rs"] {
        src_files.push(f.to_string());
    }
    // NOT in the stamp: tests/fixtures/loop_detect_vectors.txt (included by src/loop_detect.rs inside
    // `#[cfg(test)]`). It is not an engine input, and a stamp that hashed it "when present" gave one tree
    // two tokens depending on build history (with / without tests/); the include-target test
    // (`cluster::build_id_tests`) names it as its one stated `#[cfg(test)]` exception.
    src_files.sort();           // byte order of the relative path: stable order → stable hash
    src_files.dedup();
    src_dirs.sort();
    src_dirs.dedup();
    // FNV-1a 64 over (path, len, bytes) per file, in sorted order -- a stable, specified algorithm (as
    // TUNE_BUILD_ID above), NOT std's DefaultHasher, whose output is unspecified across Rust releases: two
    // machines that build the same commit with different toolchains must get the same token or they
    // would be refused at the Hello. (KERNEL_BUILD_ID keeps DefaultHasher on purpose: it is baked into the
    // PTX bytes, and moving it would change every PTX file without a kernel change.)
    let mut sh: u64 = 0xcbf2_9ce4_8422_2325;
    let mut feed = |b: &[u8]| {
        for &x in b {
            sh ^= x as u64;
            sh = sh.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for f in &src_files {
        println!("cargo:rerun-if-changed={f}");
        let bytes = std::fs::read(f).unwrap_or_else(|e| panic!("cannot read {f}: {e}"));
        feed(f.as_bytes());
        feed(&(bytes.len() as u64).to_le_bytes());
        feed(&bytes);
    }
    for d in &src_dirs { println!("cargo:rerun-if-changed={d}"); }
    let source_id = format!("{sh:016x}");
    println!("cargo:rustc-env=SOURCE_BUILD_ID={source_id}");
    println!("cargo:rustc-env=SOURCE_BUILD_FILES={}", src_files.join(";"));
    // what KERNEL_BUILD_ID hashes (the other half of `cluster::binary_version`), for the same test
    let mut kernel_files: Vec<String> = kernels.iter().map(|(s, _, _)| format!("kernels/{s}")).collect();
    kernel_files.push("native/tp_doorbell.h".to_string());
    println!("cargo:rustc-env=KERNEL_BUILD_FILES={}", kernel_files.join(";"));
    // EMB1: cu:ptx pairs for every module this script emits (the loaders embed via src/ptx_embed.rs).
    // cluster.rs build-id test exempts exactly these include targets as derived artifacts.
    let mut manifest: Vec<String> = kernels.iter().map(|(s, d, _)| format!("{s}:{d}")).collect();
    manifest.push(String::from("gpu_batch.cu:gpu_batch_b3.ptx")); // -DTQ_B3 variant of the first entry
    println!("cargo:rustc-env=EMBEDDED_PTX_MANIFEST={}", manifest.join(";"));

    if !nvcc.exists() {
        // No nvcc: the only buildable path is off the src/ptx/*.ptx a PREVIOUS build left in the
        // tree (nothing is checked in; .gitignore has *.ptx). Since EMB1 embeds those bytes, that
        // bakes possibly-stale kernels into the binary -- the KERNEL_BUILD_ID handshake refuses
        // a mismatch at boot, but fail HERE when a target file is missing at all, because the
        // include_str! compile error is otherwise a confusing rustc message.
        let missing: Vec<&str> = kernels.iter()
            .filter(|(_, dst, _)| !out_dir.join(dst).exists())
            .map(|(_, dst, _)| *dst).collect();
        if !missing.is_empty() {
            panic!("nvcc not found at {} and src/ptx/ is missing {:?} -- cannot build: EMB1 embeds \
                    these bytes (build once on a box with nvcc, or install CUDA)",
                   nvcc.display(), missing);
        }
        println!("cargo:warning=nvcc not found at {}; embedding the src/ptx/*.ptx a previous \
                  build left in the tree. Any kernels/*.cu edit will NOT take effect.",
                 nvcc.display());
        return;
    }

    for (src_name, dst_name, arch) in &kernels {
        let src = kernel_dir.join(src_name);
        let dst = out_dir.join(dst_name);
        if !src.exists() {
            panic!("{} is missing but the binary loads {}", src.display(), dst_name);
        }
        let out = Command::new(&nvcc)
            .args([
                &format!("--gpu-architecture={arch}"),
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

    // TurboQuant b=3 K variant (GB10_KV_TQ=3): the same gpu_batch.cu compiled with -DTQ_B3 (K
    // codes at 3 bits -> 68-B rows). Loaded INSTEAD of gpu_batch.ptx only when GB10_KV_TQ=3;
    // gpu_batch.ptx stays the golden-anchored b=2 build. Same KERNEL_BUILD_ID (same source
    // bytes), so the assert_kernel_build_id handshake covers both files.
    {
        let src = kernel_dir.join("gpu_batch.cu");
        let dst = out_dir.join("gpu_batch_b3.ptx");
        let out = Command::new(&nvcc)
            .args([
                "--gpu-architecture=sm_121",
                "--ptx",
                "--default-stream=per-thread",
                "-O3",
                "-Inative",
                "-DTQ_B3",
                &format!("-DKERNEL_BUILD_ID=0x{build_id}ULL"),
                "-o",
            ])
            .arg(&dst)
            .arg(&src)
            .output()
            .unwrap_or_else(|e| panic!("failed to run {nvcc:?} for gpu_batch_b3.ptx: {e}"));
        if !out.status.success() {
            panic!(
                "nvcc failed on gpu_batch.cu (-DTQ_B3) -- the b=3 build must NOT proceed on the stale {}:\n{}",
                dst.display(),
                String::from_utf8_lossy(&out.stderr)
            );
        }
    }
}
