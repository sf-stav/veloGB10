//! `impl GpuModel`: `load_from_dir_impl`, the NVFP4 / bf16 trunk loader (one function, kept whole: a single item over
//! the file budget). Callers and helpers: `construct.rs`, `shard_ops.rs`. Map: `mod.rs`.

use super::*;

impl GpuModel {
    pub(super) fn load_from_dir_impl(model_dir: &str, tp_rank: Option<i32>, world: i32) -> anyhow::Result<(Self, crate::qwen::Config)> {
        use safetensors::{SafeTensors, Dtype};
        // Read-ahead hint for the NEXT shard: the kernel starts pulling it into the page cache
        // while we process this one, so the serial `std::fs::read` finds its pages already in
        // flight. No mmap, no extra memory — WILLNEED is a self-throttling kernel hint. Declared
        // by hand (no libc dep): glibc has had it forever.
        extern "C" { fn posix_fadvise(fd: std::os::raw::c_int, offset: i64, len: i64,
                                      advice: std::os::raw::c_int) -> std::os::raw::c_int; }
        const POSIX_FADV_WILLNEED: std::os::raw::c_int = 3;
        let readahead = |p: &std::path::Path| {
            if let Ok(f) = std::fs::File::open(p) {
                use std::os::fd::AsRawFd;
                unsafe { posix_fadvise(f.as_raw_fd(), 0, 0, POSIX_FADV_WILLNEED); }
            }
        };
        // Phase timers for the [load] summary (load-speed campaign Phase 0): every phase of the
        // load accounted, one line at the end. Zero behavior change — Instant reads only.
        let t_load0 = std::time::Instant::now();
        let mut t_read = std::time::Duration::ZERO;      // shard file reads (disk/page-cache)
        let mut t_up_shard = std::time::Duration::ZERO;  // htod uploads inside the shard loop
        let mut t_shard_loop = std::time::Duration::ZERO;// whole shard loop (cpu = loop - read - upload)
        let mut t_repack_inline = std::time::Duration::ZERO; // hy3 shard-at-load inline expert repack (in-loop)
        let mut t_fuse = std::time::Duration::ZERO;      // fuse_nvfp4/fuse_fp8 concat copies
        let mut t_repack = std::time::Duration::ZERO;    // repack_*_mma (call-site wall time, assembly phase)
        let mut t_hshard = std::time::Duration::ZERO;    // host-side TP shard of repacked weights
        let mut t_up_post = std::time::Duration::ZERO;   // htod uploads inside `gwn`
        // The one user-facing message for a broken/stale/wrong-format checkpoint (owner directive
        // 2026-08-27): name the path, classify, and say what to do — instead of a panic/OOM/core-dump
        // from garbage metadata. Called at every graceful-exit site in this function.
        let load_error = |classification: &str, cause: &str| -> anyhow::Error {
            anyhow::anyhow!(
                "Error loading model from {}: {}\n\
                 \x20 The model checkpoint appears to be {}.\n\
                 \x20 Fix: re-download the FULL set cleanly (all shards + the index + config), verify with\n\
                 \x20 sha256sum that the shards match the source, and confirm the directory is the engine's\n\
                 \x20 CONVERTED NVFP4 model (not a raw or differently-quantized checkpoint). See the README.",
                model_dir, cause, classification)
        };
        // S-B9-REL-PPL: an EXL3 pack is a different container (`.trellis`/`.suh`/`.svh`, no `.weight`
        // tensors). This loader cannot read it; before this check it reached the first shard, hit
        // the alignment panic and then hung. Refuse with the way forward instead (a Result, exit 1).
        if crate::exl3_serve::is_exl3_pack(model_dir) {
            return Err(anyhow::anyhow!(
                "Error loading model from {model_dir}: this is an EXL3 pack (trellis-coded weights), which the \
                 generic NVFP4/bf16 loader cannot read.\n\
                 \x20 Use an EXL3-aware entry point: --server, --perplexity, --bench-mtp, --probe-state, --probe-exl3-*."));
        }
        let config_path = format!("{}/config.json", model_dir.trim_end_matches('/'));
        // FIX1 #77: a DFlash2 DRAFT artifact booted as --model-dir used to fall through to the
        // tensor loop and die as `missing tensor: <name>` — a message that says nothing about the
        // real mistake. Detect the artifact by its own marker (config.json's df2_quant — the
        // field dflash2::load bakes into every draft artifact) and say WHAT / WHERE / HOW TO FIX.
        if let Ok(cfg_txt) = std::fs::read_to_string(&config_path) {
            if cfg_txt.contains("df2_quant") || cfg_txt.contains("dflash_config") || cfg_txt.contains("num_target_layers") {
                let d = model_dir.trim_end_matches('/');
                anyhow::bail!(
                    "--model-dir points at a DFlash2 DRAFT artifact, not a model\n  \
                     WHAT: {config_path} carries a draft-head marker (df2_quant / dflash_config / num_target_layers) — this directory is the \
                     DFlash2 DRAFT head (5-ish draft layers over the target model), so the main model's tensors are not in it\n  \
                     WHERE: the engine was loading the MAIN model from --model-dir {d} \
                     (tensors: {d}/model.safetensors; before FIX1 this died later as a generic \
                     'missing tensor: <name>' panic)\n  \
                     HOW TO FIX: pass the draft as --draft-dir {d} and point --model-dir at the \
                     target model directory",
                );
            }
        }
        let cfg = crate::qwen::Config::from_config_json(&config_path)
            .map_err(|e| load_error("corrupted",
                &format!("config.json is missing or does not parse: {e}")))?;
        // The attention kernels take hd at launch but size per-lane register slices for hd/32 <= 16
        // (SK_DPL_MAX in gpu_batch.cu: qwen3.5 256, hy_v3 128, DeepSeek 512). Fail loudly on anything
        // outside that envelope rather than corrupting attention silently.
        assert!(cfg.head_dim % 32 == 0 && cfg.head_dim <= 512,
                "head_dim {} unsupported: the attention kernels need a positive multiple of 32 <= 512",
                cfg.head_dim);
        let _h = cfg.hidden_size;
        let _pref = "model.language_manager".rsplit_once('.').map(|(p, _)| format!("{}.{}", p, "language_model")).unwrap_or("model.language_model".to_string());

        // ---- MXFP4-NATIVE MODE (--mxfp4=on, QWEN_MXFP4_NATIVE_DESIGN.md) ----
        // Single-node AND TP=2 (Phase 2: the flag rides TpConfig; the node installs it as env
        // before load, so both ranks build the same repacks — SPMD). When on, every fp4 tensor
        // additionally gets the lossless OMMA repack (src/mxfp4.rs) computed in the assembly
        // workers and keyed by its qweight device pointer; the default buffers are untouched
        // (the bf16 chain and the prefill dequant path keep reading them). TP attach re-keys the
        // repacks when the weights get sharded (shard_mxfp4_col_segs / shard_mxfp4_row twins).
        let mxfp4_mode = crate::opts::var(crate::opt!("mxfp4")).is_ok();
        // Sensitive-tensor allowlist (design §8.1 risk #1 — MEASURED, 2026-08-07): the MTP head's
        // own e2m1 activation quant erodes near-tie draft acceptance (~2-3 pts overall, −6..−11
        // pts in the 0.3-0.7 confidence buckets) because its logits decide acceptance. Keeping
        // the MTP head on the bf16 chain (its bytes are ~1% of the step budget) recovers
        // acceptance to the bf16-chain baseline (73.9 -> 75.9 vs 76.7 off-mode). DEFAULT ON —
        // the MTP head joins the design §3.3 stay-list. --mxfp4-mtp-native=1 is the escape
        // hatch back to the all-native chain (the Phase 3 variant, acceptance-gated).
        let mxfp4_mtp_native = crate::opts::var(crate::opt!("mxfp4-mtp-native")).is_ok();
        let t_omma = std::sync::Arc::new(std::sync::Mutex::new(std::time::Duration::ZERO)); // OMMA repack timer (assembly phase, mode on only)
        // mxfp4-native mode: OMMA repacks keyed by the qweight device pointer. Filled by the
        // draft-head site below AND by the assembly uploader thread (extend, never replace).
        let mut omma_map: std::collections::HashMap<u64, crate::mxfp4::OmmaEntry> =
            std::collections::HashMap::new();
        // Sensitive-tensor allowlist: qweight ptrs that run the bf16 chain even in native mode.
        let mut allow_bf16: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut mixer4: std::collections::HashSet<u64> = std::collections::HashSet::new();
        if mxfp4_mode {
            println!("mxfp4-native mode ON: fp4 decode/verify GEMMs will run the sm_121a OMMA path \
                      (lossless load-time repack; bf16 chain preserved){}",
                     if mxfp4_mtp_native { "; MTP head runs NATIVE (--mxfp4-mtp-native)" }
                     else { "; MTP head allowlisted to the bf16 chain (acceptance-gated default)" });
        }

        // ---- TP=2 SHARD-AT-LOAD (hy_v3 only) ----
        // hy_v3's 169.6 GB cannot be loaded whole and sharded afterwards (`attach_tp`'s in-place
        // `tp_shard_weights` needs the FULL model resident on one GPU first), so under TP the
        // sharding happens HERE, host-side, per tensor, before upload — the same byte math as the
        // GPU-side sharders (16-row tile bands / per-tile k-block ranges), applied pre-upload.
        // Trigger: TP load (`load_from_dir_tp`) + mixer sharding on (env on the head, the
        // head-shipped TpConfig on the node). hy_v3 always; qwen now too — the 122B-class does
        // not fit one node for the in-place shard (each rank must load only its shard). The
        // attach asserts tp_shard_mixers() for pre-sharded loads, so a mixers-off TP load
        // refuses there rather than running with disagreeing head counts.
        let sal: Option<usize> = match tp_rank {
            None => None,
            Some(r) => {
                let mixers = crate::opts::var(crate::opt!("tp-shard-mixers")).is_ok()
                    || crate::tp::tp_config().map(|c| c.shard_mixers).unwrap_or(false);
                if cfg.family == crate::qwen::Family::HyV3 {
                    assert!(mixers, "hy_v3 under TP=2 requires mixer sharding ([tp-shard-mixers] / \
                                     TpConfig.shard_mixers): without it nothing halves and the full model \
                                     cannot fit one node. Refusing to load.");
                }
                assert!(r >= 0 && (r as usize) < world as usize, "shard-at-load: rank {r} out of range [0, {world})");
                crate::deprintln!("[tp] shard-at-load: rank {r}/{world} — weights will be sliced host-side before upload");
                if mixers { Some(r as usize) } else { None }
            }
        };
        assert!(sal.is_none() || crate::opts::var(crate::opt!("dequant-at-load")).is_err(),
                "shard-at-load is incompatible with --dequant-at-load (a debug-only path)");
        // B32: a shard dir (gb10_shard.json) must be this load's (nvfp4, world, rank); its PRE-SLICED tensors (the
        // rank's stacked-expert bands, cut by the same `col_band`) skip the load-time slice — and only those.
        let presliced: std::collections::HashSet<String> = {
            let d = std::path::Path::new(model_dir);
            let info = match sal {
                Some(r) => crate::shard_plan::check_shard_dir(d, crate::shard_plan::ShardFormat::Nvfp4, world.max(1) as usize, r)?,
                None => {
                    if let Some(i) = crate::shard_plan::read_shard_info(d)? {
                        anyhow::bail!("{model_dir} is a B32 shard dir (nvfp4 rank {} of world {}) — it loads only under TP \
                                       shard-at-load with mixer sharding", i.rank, i.world);
                    }
                    None
                }
            };
            if let Some(i) = &info {
                crate::deprintln!("[tp] B32 shard dir: nvfp4 rank {} of world {} — {} tensors arrive pre-sliced (their load-time \
                           slice is skipped)", i.rank, i.world, i.presliced.len());
            }
            // the loader's slice decision is per ASSEMBLED tensor (its `.weight` base name, whose raw parts are
            // `.weight_packed` / `.weight_scale` on disk) — key the set by that base
            info.map(|i| i.presliced.into_iter().map(|n| {
                n.strip_suffix("_packed").or_else(|| n.strip_suffix("_scale")).map(str::to_string).unwrap_or(n)
            }).collect()).unwrap_or_default()
        };

        // Find and load all safetensors shards
        let dir = std::path::Path::new(model_dir);
        let index_path = dir.join("model.safetensors.index.json");
        let safetensors_files: Vec<std::path::PathBuf> = if index_path.exists() {
            let index_raw = std::fs::read_to_string(&index_path)
                .map_err(|e| load_error("corrupted",
                    &format!("model.safetensors.index.json is unreadable: {e}")))?;
            let index: serde_json::Value = serde_json::from_str(&index_raw)
                .map_err(|e| load_error("corrupted",
                    &format!("model.safetensors.index.json does not parse as JSON: {e}")))?;
            let wm = index["weight_map"].as_object().ok_or_else(|| {
                load_error("wrong-format",
                    "model.safetensors.index.json has no \"weight_map\" object — not a safetensors sharded-model index")
            })?;
            let mut files: std::collections::BTreeSet<std::path::PathBuf> = std::collections::BTreeSet::new();
            for (_, v) in wm {
                if let Some(s) = v.as_str() {
                    let p = dir.join(s);
                    if !p.exists() {
                        // A stale/mismatched index: some shards present, others missing.
                        return Err(load_error("stale",
                            &format!("the index references shard \"{s}\" which is not present on disk — \
                                      partial or stale download (which shards are missing is what matters)")));
                    }
                    files.insert(p);
                }
            }
            if files.is_empty() {
                return Err(load_error("corrupted",
                    "the index declares no shards — empty or partial download"));
            }
            files.into_iter().collect()
        } else {
            let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
                .map_err(|e| load_error("corrupted", &format!("cannot read the model directory: {e}")))?
                .filter_map(|e| {
                    let e = e.ok()?;
                    let n = e.file_name().to_string_lossy().to_string();
                    if n.ends_with(".safetensors") { Some(e.path()) } else { None }
                }).collect();
            if files.is_empty() {
                return Err(load_error("corrupted",
                    "no .safetensors shards found in the directory — empty or partial download"));
            }
            files
        };

        // Process shards ONE AT A TIME — load, upload to GPU, drop raw bytes.
        // Peak memory: one shard (~5 GB) + accumulated GPU weights (~55.6 GB) ≈ 61 GB.
        // hy_v3 keeps the transformer root at `model.*` (qwen3.5 nests it at `model.language_model.*`).
        let pref = if cfg.family == crate::qwen::Family::HyV3 { "model" } else { "model.language_model" };
        let dev = CudaDevice::new(0)?;
        let stream = fork_blocking_stream(&dev);
        let blas = CudaBlas::new(dev.clone())?;
        unsafe { blas.set_stream(Some(&stream))?; }
        let t_ptx0 = std::time::Instant::now();
        Self::init_ptx(&dev)?;
        let (k, bk) = Self::load_batch_kernels(&dev)?;
        let t_ptx = t_ptx0.elapsed();
        let t_rope0 = std::time::Instant::now();
        // E5/YARN1: trunk YaRN — the flag wins, then the shipped TpConfig (node), then the
        // pack config's rope_scaling (one precedence rule: qwen::pick_yarn).
        let mut cfg = cfg;
        let (yarn, yarn_src) = crate::qwen::resolved_yarn(&cfg);
        cfg.rope_yarn_factor = yarn;
        let (cos_table, sin_table) = Self::build_rope_tables(&dev, &cfg, yarn_src)?;
        let t_rope = t_rope0.elapsed();
        let sc_pos = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        let sc_rope = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        let sc_slot = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        let sc_winsrc = dev.alloc_zeros::<i32>(MAX_VERIFY * 8).unwrap();
        let sc_parent = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        let sc_path = dev.alloc_zeros::<u8>(MAX_VERIFY * MAX_VERIFY).unwrap();
        let sc_tok = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        let sc_pstart = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        let moe_ids = dev.alloc_zeros::<i32>(16 * MAX_VERIFY).unwrap();
        let moe_wts = dev.alloc_zeros::<f32>(16 * MAX_VERIFY).unwrap();
        let sc_i1a = dev.alloc_zeros::<i32>(1).unwrap();
        let sc_i1b = dev.alloc_zeros::<i32>(1).unwrap();
        let sv_pf = dev.alloc_zeros::<f32>(3 * MAX_VERIFY).unwrap();
        let sv_ki = dev.alloc_zeros::<i32>(2 * MAX_VERIFY).unwrap();
        let sv_cand = dev.alloc_zeros::<u64>(MAX_VERIFY * 16).unwrap();
        let sv_sd = dev.alloc_zeros::<u32>(MAX_VERIFY).unwrap();
        let sv_p  = dev.alloc_zeros::<f32>(MAX_VERIFY).unwrap();
        let sv_r  = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        let mr_tok = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        let mr_pos = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        // alloc_zeros is cuMemAllocAsync and does NOT zero. This buffer MUST be all zeros.
        let mut mtp_sids = dev.alloc_zeros::<i32>(MAX_VERIFY).unwrap();
        dev.memset_zeros(&mut mtp_sids).unwrap();
        let deq_scratch = std::sync::Mutex::new(None);
        let fp8e_map: std::sync::Mutex<std::collections::HashMap<u64, cudarc::driver::CudaSlice<u8>>> = std::sync::Mutex::new(std::collections::HashMap::new());
        let fp8e_scratch: std::sync::Mutex<Option<(cudarc::driver::CudaSlice<u8>, cudarc::driver::CudaSlice<f32>)>> = std::sync::Mutex::new(None);
        let splitk_partials = dev.alloc_zeros::<f32>(SPLITK_MAX_TILES * SPLITK_MAX_S * 256).unwrap();
        // R9 (world>2): fp32 upcast scratch for the no-rounding all-reduce. Sized for the widest
        // single-slot reduction (TP_SLOT_BYTES/4 fp32 elems = the whole ring slot in fp32).
        let tp_f32_scratch = dev.alloc_zeros::<f32>(crate::tp::TP_SLOT_BYTES / 2).unwrap();
        dev.synchronize().unwrap();

        let mut gpu_bf16: std::collections::HashMap<String, B> = std::collections::HashMap::new();
        // compressed-tensors `input_global_scale` (W4A4 activation scale) per packed stem, if present
        let mut igs_by_name: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
        let mut gpu_f32: std::collections::HashMap<String, S> = std::collections::HashMap::new();

        // Simulated NVFP4 (--fake-quant). Round-trips a weight through the 4-bit codec and
        // back to bf16: the bytes stay bf16 so the engine is unmodified, but the VALUES carry exactly
        // the error the real kernel would produce. This is how we test — in the real engine, before
        // any kernel exists — whether the LM head and the GDN projections actually need high
        // precision, or whether that is just something everyone repeats.
        let fq = crate::quant::fake_quant_spec();
        // S5F2 clean-tap ladder: per-LAYER overrides (`layers:<sel>:<fmt>` tokens in the same
        // env var) win over the group map — L1a fp8 tap-column, L1b bf16 prefix.
        let fq_layers = crate::quant::fake_quant_layer_rules();
        let mut fq_stats: Vec<(String, f32, f32)> = Vec::new();
        if let Some(map) = &fq {
            let mut names: Vec<String> = map.iter()
                .map(|&(g, f)| format!("{}:{}", crate::quant::group_name(g), crate::quant::fmt_name(f)))
                .collect();
            for r in &fq_layers {
                names.push(format!("layers:{}-{}:{}", r.lo, r.hi, crate::quant::fmt_name(r.fmt)));
            }
            println!("SIMULATED QUANTIZATION (fake-quant): {}", names.join(" "));
        }

        fn needs_f32(name: &str) -> bool {
            name.ends_with("A_log") || name.ends_with("dt_bias") ||
            (name.contains("norm") && name.ends_with(".weight")) ||
            name.ends_with("conv1d.weight") ||
            name.ends_with("expert_bias")   // hy_v3 router bias: fp32 semantics (kept fp32 by the reference)
        }

        // A quantized artifact stores each quantized tensor as a GROUP of entries rather than one
        // `.weight`. Reassemble them here and dequantize to bf16 on the way to the GPU.
        //
        // This is deliberately the slow path: it proves the artifact end-to-end (encoder -> file ->
        // loader -> dequant) against the simulated-quantization perplexity, with zero kernel work.
        // It does NOT make decode faster — the weights land in VRAM as bf16, so the same bytes stream
        // per token. The speed comes from `gemm_binv_fp4_b` keeping them packed; this is the
        // correctness rung below it.
        //   NVFP4: {stem}.weight_packed [M,K/2] u8 + {stem}.weight_scale [M,K/16] e4m3
        //          + {stem}.weight_global_scale [1] f32
        //   FP8:   {stem}.weight (F8_E4M3) [M,K]  + {stem}.weight_scale [M] f32
        let mut n_dq4 = 0usize;
        let mut n_dq8 = 0usize;
        // Escape hatch: dequantize to bf16 at load. Same numbers, no speedup — useful to prove the
        // fused kernels against, since the two paths must agree exactly.
        let dequant_at_load = crate::opts::var(crate::opt!("dequant-at-load")).is_ok();
        // Quantized tensors are held HOST-side until every shard is read, then fused, repacked into
        // mma-fragment order, and uploaded. Deferring is what makes fusion possible: the four GDN
        // input projections must be concatenated along M *before* the mma permutation, and they are
        // not guaranteed to arrive in the same shard.
        type Q4H = (Vec<u8>, Vec<u8>, f32, usize, usize);   // qweight, scales, 1/global_scale, m, k
        type Q8H = (Vec<u8>, Vec<f32>, usize, usize, usize);   // qweight RAW, scales, m, k, nb_k
        // nb_k == 0 → legacy per-row scheme, scales [M] f32 (w = code·scale_row).
        // nb_k == k/128 → Qwen block-128 scheme, scales flat [(m/128)*nb_k] f32
        // (w = code·scale_inv[(row/128)*nb_k + (col/128)]). RAW bytes either way — the worker
        // shards RAW and repacks to MMA order last (the pipeline's existing contract).
        let mut host_q4: std::collections::HashMap<String, Q4H> = std::collections::HashMap::new();
        let mut host_q8: std::collections::HashMap<String, Q8H> = std::collections::HashMap::new();
        // ---- STREAMING LOAD pre-pass (122B OOM fix, 2026-08-07) ----
        // Classify every weight from the shard HEADERS ONLY (no data read), so the record pass
        // and the job specs exist BEFORE any shard data arrives. The assembly pipeline then drains
        // each shard's parts while the next shard is read — previously ALL shards' raw sat in
        // host_q4 (~56 GB for the 122B) until the last shard was read, and the crossover with the
        // growing device copy OOM'd the 121 GB unified pool even on an otherwise-idle box.
        fn shard_header(path: &std::path::Path) -> anyhow::Result<serde_json::Value> {
            use std::io::Read;
            let mut f = std::fs::File::open(path)?;
            let mut lenb = [0u8; 8];
            f.read_exact(&mut lenb)?;
            let hlen = u64::from_le_bytes(lenb);
            // GARBAGE-HEADER GUARD: a corrupt/truncated/shadowed shard can put a ~2^61 value in the
            // 8-byte length prefix, and `vec![0u8; hlen]` is exactly the "memory allocation of
            // 2336927755350992246 bytes failed" OOM. Bound before allocating. A real safetensors
            // header is far smaller than 256 MiB.
            if hlen == 0 || hlen > (1 << 28) {
                return Err(anyhow::anyhow!(
                    "corrupt safetensors header length {hlen} (a truncated or wrong-format shard)"));
            }
            let mut hdr = vec![0u8; hlen as usize];
            f.read_exact(&mut hdr)?;
            Ok(serde_json::from_slice(&hdr)?)
        }
        // variant: 1 = NVFP4 (packed), 2 = FP8, 3 = bf16 — keyed by the FINAL weight name
        // (e.g. "model.layers.0.mlp.gate_proj.weight"), exactly the key the shard loop inserts.
        let mut variants: std::collections::HashMap<String, u8> = std::collections::HashMap::new();
        n_dq4 = 0;
        n_dq8 = 0;
        let mut q4_bytes = 0usize;
        let mut q8_bytes = 0usize;
        let mut bf16_bytes = 0usize;
        // Wrong-format detection: if the shards hold tensors but NONE match the engine's recognized
        // scheme (packed NVFP4 / FP8 / bf16 weight names), the directory is a raw/different layout —
        // say so explicitly instead of a downstream missing-tensor panic.
        let mut total_tensors = 0usize;
        let mut recognized = 0usize;
        for sf_path in &safetensors_files {
            let hdr = shard_header(sf_path).map_err(|e| load_error("corrupted",
                &format!("shard {} has a corrupt/truncated header: {e}", sf_path.display())))?;
            let obj = hdr.as_object().ok_or_else(|| {
                load_error("corrupted", &format!(
                    "shard {} header is not a JSON object — corrupt or wrong-format file", sf_path.display()))
            })?;
            for (name, meta) in obj {
                if name == "__metadata__" { continue; }
                total_tensors += 1;
                let dtype = meta.get("dtype").and_then(|d| d.as_str()).unwrap_or("");
                let shape: Vec<u64> = meta.get("shape").and_then(|s| s.as_array())
                    .map(|a| a.iter().filter_map(|v| v.as_u64()).collect()).unwrap_or_default();
                // GARBAGE-SHAPE GUARD: a corrupt tensor metadata can declare a 0 or a ~2^61 shape;
                // the downstream `m*k*elems` allocation is the OOM. Reject before allocating.
                let mut elems: u64 = 1;
                let mut bad_shape = shape.iter().any(|&d| d == 0);
                if !bad_shape {
                    for &d in &shape {
                        elems = elems.saturating_mul(d);
                        if elems > (1 << 40) { bad_shape = true; break; }
                    }
                }
                if bad_shape {
                    return Err(load_error("corrupted", &format!(
                        "tensor \"{name}\" in {} declares a corrupt shape {shape:?} (zero or implausibly large dims)",
                        sf_path.display())));
                }
                let (m, k) = (shape.first().copied().unwrap_or(0) as usize,
                              shape.get(1).copied().unwrap_or(0) as usize);
                if name.ends_with(".weight_packed") {
                    // packed [M, K/2] + E4M3 scales [M, K/16]
                    variants.insert(format!("{}.weight", name.trim_end_matches(".weight_packed")), 1);
                    n_dq4 += 1;
                    recognized += 1;
                    q4_bytes += m * k + m * (k / 8);
                } else if name.ends_with(".weight") && dtype == "F8_E4M3" {
                    variants.insert(name.clone(), 2);
                    n_dq8 += 1;
                    recognized += 1;
                    q8_bytes += m * k + m * 4;
                } else if name.ends_with(".weight") && (dtype == "BF16" || dtype == "F16") {
                    variants.insert(name.clone(), 3);
                    recognized += 1;
                    bf16_bytes += m * k * 2;
                }
            }
        }
        // Wrong-format: shards present but no recognized weight scheme at all.
        if total_tensors > 0 && recognized == 0 {
            return Err(load_error("wrong-format",
                "this checkpoint is not the converted NVFP4 format the engine expects — the shards \
                 contain tensors but none match the engine's converted layout (no *_weight_packed / \
                 FP8 / bf16 weight scheme); it is a raw HF or differently-quantized checkpoint"));
        }
        // Reserved LM-head/embed parts for the FR-Spec draft head: the head's own assembly job
        // consumes them from host_q4/host_q8, so the draft build (post-loop) reads these copies.
        let mut draft_parts_q4: Option<Q4H> = None;
        // Draft-head FP8 parts — row-scheme ONLY (nb_k==0): the DFlash2 borrowed-head subsetter
        // (subset_rows_fp8) has no block-scale variant, and every checkpoint that needs it
        // (Qwen official FP8) keeps lm_head bf16 anyway, so the blk path never populates this.
        let mut draft_parts_q8: Option<(Vec<u8>, Vec<f32>, usize, usize)> = None;
        // The economy/dual-layout decision (used by the workers AND the fit guard): computed
        // here from the header pre-pass totals — the maps drain during the load, so a mid-load
        // sum would be wrong.
        let mxfp4_economy = (mxfp4_mode && crate::opts::var(crate::opt!("mxfp4-economy")).is_ok())
            || (mxfp4_mode && q4_bytes > (40 << 30));
        if mxfp4_economy {
            println!("mxfp4-native ECONOMY mode: fp4 weights exceed the dual-layout budget — the \
                      OMMA repack is the primary storage (prefill dequantizes from it)");
        }
        // ---- SINGLE-NODE FIT GUARD (user directive 2026-08-07) ----
        // A model too big for one node must REFUSE to load (with a clear message) instead of
        // OOM-killing the box. The footprint estimate: the resident device layouts (dual =
        // standard + OMMA for fp4, economy = OMMA-primary; plus FP8/bf16) + the KV cache at the
        // model's max context + pools/scratch. Under TP the rank-local footprint is what matters
        // (shard-at-load halves it) — the guard uses the rank-local estimate there.
        // Bound on the pipeline's in-flight host bytes (raw parts + their repacked twins). 24 GB
        // was sized for ~5 GB shards; on the 100 GB qwen4_exp artifact that transient, on top of the
        // raw shard + its part copies, exhausted the box. --load-pipe-cap-gb overrides.
        let pipe_cap_gb: usize = crate::opts::var(crate::opt!("load-pipe-cap-gb")).ok().and_then(|v| v.parse().ok())
            .unwrap_or(if cfg.is_q4() { 8 } else { 24 });
        #[allow(non_snake_case)]
        let PIPE_CAP_BYTES: usize = pipe_cap_gb << 30;
        // --load-force bypasses the guard — except on qwen4_exp, where exhausting the unified
        // memory hung the box twice (2026-08-28); there only `--load-force=unsafe` does.
        let force = crate::opts::var(crate::opt!("load-force")).ok();
        let force_ok = match force.as_deref() {
            None => false,
            Some("unsafe") => true,
            Some(_) => !cfg.is_q4(),
        };
        if force.is_some() && !force_ok {
            eprintln!("[warn] --load-force is ignored on this family (memory exhaustion hangs the machine); \
                       use --ple-offload ssd, a smaller --max-seq-len, or --load-force=unsafe.");
        }
        if !force_ok {
            // Mode-aware resident fp4: mode-off stores the standard layout once; mxfp4=on
            // dual-layout stores standard + OMMA; economy stores the OMMA only.
            let fp4_resident = if !mxfp4_mode { q4_bytes }
                               else if mxfp4_economy { q4_bytes } else { q4_bytes * 2 };
            let kv_slots = if sal.is_some() { 2 } else { 2 };
            let kv_bytes = cfg.num_kv_heads * cfg.head_dim * cfg.max_position_embeddings
                * 2 * kv_slots * 2;
            // Single-node estimate: device layouts + KV + the streaming raw transient
            // (q4/3 rule of thumb) + a 16 GB margin for base/page-cache/other processes.
            // qwen4_exp: the device-resident PLE table (96-B records) is outside the safetensors.
            let ple_resident: usize = if cfg.is_q4() && !Self::ple_offload_ssd() {
                crate::ple::PleTableMeta::load(std::path::Path::new(model_dir))
                    .map(|m| m.total_rows * m.record_bytes).unwrap_or(0)
            } else { 0 };
            // Single node, MEASURED accounting (owner 2026-11-07, "how can an NVFP4 pack of the 122B not fit in
            // ~115 GB?"): the streaming raw transient is ~1 read-ahead shard plus its parts (posix_fadvise DONTNEED
            // drops each shard after its parts drain — the same J9 measurement the TP branch below uses), NOT q4/3 of
            // the whole model, and MemAvailable already excludes the OS and every other process, so one 8 GB margin
            // is the buffer. The old `q4/3 + 16 GB + 8 GB` refused the 122B NVFP4 at 117.8 + 8 > 124.3 GB available
            // while the forced load ran at ~106 GB used / 19.3 GB free on .12 (TP=1, the runner's 2026-11-07 case).
            let nshards1 = safetensors_files.len().max(1);
            let raw_transient1 = 2 * (q4_bytes / nshards1).max(2 << 30);
            let mut est = fp4_resident + q8_bytes + bf16_bytes + kv_bytes + (4 << 30)
                + raw_transient1 + (8 << 30) + ple_resident;
            let mut extra = 0;   // the 8 GB margin above is the buffer
            if cfg.is_q4() && sal.is_none() {
                // qwen4_exp, MEASURED accounting (2026-08-28): the load-time transient is the
                // largest raw shard (read whole) + a copy of its parts + the pipeline cap + the
                // next shard's read-ahead cache; steady state adds KV/pools (~6 GB) and the PLE
                // table when resident. Margin: 6 GB for the OS/page cache/other processes.
                let max_shard = safetensors_files.iter()
                    .map(|p| std::fs::metadata(p).map(|m| m.len() as usize).unwrap_or(0)).max().unwrap_or(0);
                let transient = max_shard * 3 + PIPE_CAP_BYTES;
                let steady = fp4_resident + q8_bytes + bf16_bytes + kv_bytes + ple_resident + (6 << 30);
                est = steady.max(fp4_resident + q8_bytes + bf16_bytes + transient);
                extra = 6 << 30;
                if crate::boot_report::debug() {
                eprintln!("[mem] qwen4_exp footprint: device {:.1} GB (+PLE {:.1} GB resident) | load transient ~{:.1} GB \
                           (shard {:.1} GB x3 + pipeline {:.1} GB) | steady ~{:.1} GB | peak ~{:.1} GB",
                          (fp4_resident + q8_bytes + bf16_bytes) as f64 / 1e9, ple_resident as f64 / 1e9,
                          transient as f64 / 1e9, max_shard as f64 / 1e9, PIPE_CAP_BYTES as f64 / 1e9,
                          steady as f64 / 1e9, est as f64 / 1e9);
                }
            }
            if sal.is_some() {
                // Rank-local TP=2, computed DIRECTLY (not est/2). Device fp4/fp8 halve; bf16
                // (embed/lm_head/router/norms) is replicated → full; KV halves by head. The
                // streaming raw transient is ~1 read-ahead shard (posix_fadvise DONTNEED drops
                // each shard after its parts drain — measured 5-19 GB, J9), NOT q4/3 of the whole
                // model. The old `est/2 + q4/3 + 8G` DOUBLE-counted the raw term (once inside est
                // halved, once outside) and predicted ~186 GB for hy3 vs the ~102 GB measured
                // peak — forcing --load-force on every hy3 load. This matches the measured
                // rank-local peak (hy3: ~108 GB estimate, ~102 GB actual; 122B-mixed ~48 vs ~45).
                let nshards = safetensors_files.len().max(1);
                let raw_transient = (q4_bytes / nshards).max(2 << 30);
                est = fp4_resident / 2 + q8_bytes / 2 + bf16_bytes + kv_bytes / 2
                    + raw_transient + (8 << 30);
                extra = 0;   // the 8 GB margin above is the buffer; no double-margin for TP
            }
            let avail = std::fs::read_to_string("/proc/meminfo").ok()
                .and_then(|s| s.lines().find(|l| l.starts_with("MemAvailable:")).map(|l| {
                    l.split_whitespace().nth(1).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0) * 1024
                })).unwrap_or(0);
            if est as u64 + extra > avail {
                eprintln!("ERROR: model footprint ~{:.1} GB exceeds the ~{:.1} GB available on this box \
                          ({} mode). Free memory, or run with --tp (each rank loads only its shard, \
                          ~{:.1} GB/rank). --load-force=1 bypasses this check.",
                          est as f64 / 1e9, avail as f64 / 1e9,
                          if sal.is_some() { "TP rank-local" } else { "single-node" },
                          est as f64 / 2e9);
                std::process::exit(2);
            }
        }
        // TP=2 shard-at-load (hy_v3): the big stacked EXPERT tensors never enter host_q4 at all —
        // accumulating them host-side would peak at ~156 GB on a 128 GB unified box. They are a
        // contiguous per-rank expert BAND of the [ne·me, K] stack, so the rank's rows are sliced,
        // repacked and uploaded INLINE while the shard file is open, and land here already final.
        // `gwn` consults this map first. Holding the whole rest (attn/dense/shared, ~4 GB) in
        // host_q4 as usual is fine.
        let mut gpu_q4_sharded: std::collections::HashMap<String, W> = std::collections::HashMap::new();

        let t_asm0 = std::time::Instant::now();   // whole post-shard assembly phase
        // Memory probe at phase boundaries (122B economy-mode OOM hunt, 2026-08-07): VmRSS +
        // MemAvailable so the assembly peak is attributable to a phase.
        fn mem_probe(tag: &str) {
            let rss = std::fs::read_to_string("/proc/self/status").ok()
                .and_then(|s| s.lines().find(|l| l.starts_with("VmRSS:")).map(|l| l.to_string()));
            let avail = std::fs::read_to_string("/proc/meminfo").ok()
                .and_then(|s| s.lines().find(|l| l.starts_with("MemAvailable:")).map(|l| l.to_string()));
            if crate::boot_report::debug() {
                eprintln!("[mem] {tag}: {rss:?} {avail:?}", );
            }
        }
        mem_probe("pre-shards");
        let mut t_gwn = std::time::Duration::ZERO; // sum of whole `gwn` calls (gwn-cpu = gwn - its parts)
        let is_hy3 = cfg.family == crate::qwen::Family::HyV3;
        let hy3_mtp_prefix = format!("model.layers.{}", cfg.num_layers);
        let hy3_num_layers = cfg.num_layers;   // Copy capture for the mxfp4 MTP-head allowlist (threads)
        let mtp_present = if is_hy3 {
            variants.contains_key(&format!("{hy3_mtp_prefix}.eh_proj.weight"))
        } else if cfg.is_q4() {
            variants.contains_key("mtp.fc_embedding.weight")
        } else {
            variants.contains_key("mtp.fc.weight")
        };
        let mtp_moe_ok = !mtp_present || !cfg.is_moe || {
            let k = if is_hy3 { format!("{hy3_mtp_prefix}.mlp.experts.gate_up_proj") }
                    else { "mtp.layers.0.mlp.experts.gate_up_proj".to_string() };
            variants.keys().any(|n| n.starts_with(&k[..]))
        };
        let has_mtp = mtp_present && mtp_moe_ok;
        if mtp_present && !mtp_moe_ok {
            println!("  MTP head present but its MoE experts are stored un-fused (per-expert) — \
                      skipping MTP (optional speculative-decode head; model decodes fine without it)");
        }
        let mut gb = |n: &str| -> B { gpu_bf16.remove(n).unwrap_or_else(|| panic!("missing bf16 tensor: {}", n)) };

        // Build ONE weight from one-or-more source tensors, concatenated along M.
        //
        // Concatenating is the point: several projections in this architecture read the SAME
        // activation and are separate tensors only because the checkpoint stores them that way. Fused,
        // they cost ONE GEMM — and the M=32 GDN beta/alpha projections stop being their own
        // catastrophically under-parallel launch (grid=2 on a 48-SM GPU; 26 us to move 74 KB; 4.7% of
        // all GEMM time for 0.03% of the bytes).
        //
        // Fusion happens BEFORE the mma permutation, and the NVFP4 tensor scale becomes a per-16-row
        // -tile lookup (every segment boundary is 16-aligned, so a tile never straddles two segments).
        // bf16 is NOT fused: it is not the serving path, and the legacy single-stream forward that the
        // PyTorch reference tests anchor indexes the projections individually.
        // ---- FR-Spec: a smaller LM head, for picking DRAFT tokens only ----
        //
        // Built by subsetting the rows of the real head, which is EXACT (rows do not interact in any
        // codec). DEFAULT ON for the Qwen3.5/3.6/3.8 family (owner 2026-10-27; README documents the
        // trade: non-Latin scripts are mostly outside the id-order subset, `--draft-vocab off` for
        // those workloads); opt-in elsewhere (hy_v3 unverified; qwen4_exp never). `--draft-vocab off`
        // = no draft head and no reserved host copy.
        //
        // A row count is the number of most-frequent token ids to keep (plus the whole tail of the id
        // range, where every special token lives -- a drafter that cannot propose <|im_end|> cannot
        // draft the end of a chat turn). A FRACTION ("0.30") or PERCENT ("30%") is taken of the full
        // vocab. NOTE the ranking is BPE ID ORDER (earlier merges ≈ more frequent) — a proxy, not
        // a measured ranking. `--draft-vocab-file=<path>` replaces the list with explicit
        // newline-separated ids from a corpus-ranked artifact.
        let draft_top: usize = match crate::opts::var(crate::opt!("draft-vocab")).ok() {
            // validated at boot (opts::canon)
            Some(v) => crate::quant::DraftVocab::parse(&v).unwrap_or(crate::quant::DraftVocab::Off),
            None => crate::quant::DraftVocab::default_for(cfg.family),
        }.top(cfg.vocab_size);
        let head_name = if cfg.tie_word_embeddings { format!("{}.embed_tokens.weight", pref) }
                        else { "lm_head.weight".to_string() };
        // FR-Spec draft head: the ROWS are precomputed here (data-independent); the subset build
        // needs the LM-head's bytes, which the streaming pipeline consumes — it is built AFTER the
        // shard loop (section 3b) from the reserved `draft_parts_*` copies or the bf16 map.
        let mut draft_rows: Vec<u32> = if draft_top == 0 && crate::opts::var(crate::opt!("draft-vocab-file")).is_err() {
            Vec::new()
        } else {
            match crate::opts::var(crate::opt!("draft-vocab-file")) {
                Ok(path) => crate::quant::draft_vocab_rows_file(&path, cfg.vocab_size)
                    .unwrap_or_else(|e| panic!("--draft-vocab-file {path}: {e}")),
                Err(_) => crate::quant::draft_vocab_rows(draft_top, cfg.vocab_size),
            }
        };
        // Graceful fallbacks (AGENTS §1b-bis 8): one line, then the full-vocabulary drafter.
        if !draft_rows.is_empty() && cfg.family == crate::qwen::Family::Qwen4Exp {
            crate::rprintln!("--draft-vocab: not applied to this qwen4_exp (Flash-Next) model — its MTP drafter \
                      keeps the full vocabulary (the reduced draft head is built for the Qwen3.5/3.6/3.8 \
                      and hy_v3 families only)");
            draft_rows.clear();
        }
        if !draft_rows.is_empty() && draft_rows.len() >= cfg.vocab_size {
            crate::rprintln!("--draft-vocab: the requested subset ({} rows) covers the whole {}-token vocabulary — \
                      no reduced draft head is built (it would only duplicate the LM head)",
                     draft_rows.len(), cfg.vocab_size);
            draft_rows.clear();
        }




        // A bf16 MTP head (recipe `-mtp`, to fix speculative acceptance) inside an otherwise-quantized
        // model needs the LOADER to treat the MTP head per-weight, not globally: its attention can't be
        // FUSED (a quantized-only trick), and its stacked experts are named WITHOUT the `.weight` suffix
        // the packed path adds. `mtp_quant` says whether the MTP head is quantized (checked before `gwn`
        // borrows the maps).
        let mtp_quant = if is_hy3 {
            variants.get(&format!("{hy3_mtp_prefix}.eh_proj.weight")).map_or(false, |v| *v != 3)
        } else {
            variants.get(if cfg.is_q4() { "mtp.fc_embedding.weight" } else { "mtp.fc.weight" }).map_or(false, |v| *v != 3)
        };

        // ---- PARALLEL ASSEMBLY (load-speed campaign, Phase 1 item 3) ----
        // The weight assembly runs the SAME code twice:
        //   1. a RECORD pass whose gwn only notes the call sequence (dummy weights; maps untouched);
        //   2. the REAL pass whose gwn looks up finished weights.
        // In between, a worker pool (<=8 std threads) runs fuse+repack(+host TP shard) per weight —
        // weights are independent and each output byte is a pure function of the input parts, so
        // per-weight bytes are unchanged — feeding ONE uploader thread that owns every CUDA call
        // during the pipeline (NULL-stream htod, like everywhere else; the blocking compute stream
        // is untouched). This overlaps the CPU assembly with the uploads, and the big host-buffer
        // frees land on the workers instead of serializing on the main thread.
        // Fuse if the artifact is quantized (`gwn` concatenates along M); leave bf16 split.
        let quantized = n_dq4 + n_dq8 > 0 && !dequant_at_load;
        let hy3 = cfg.family == crate::qwen::Family::HyV3;
        let q4 = cfg.is_q4();
        // True during the RECORD pass (below): rec_gwn hands out 16x16 dummies of the right
        // VARIANT so the assembly's own dispatch sees the shapes — the shape asserts must run
        // on the REAL pass only (they'd fire on every dummy otherwise).
        let record_pass = std::cell::Cell::new(true);
        let mut assemble = |mut gwn: &mut dyn FnMut(&[String]) -> W,
                            mut gf: &mut dyn FnMut(&str) -> S|
            -> (W, S, Option<W>, Vec<GpuLayer>, Option<GpuMtpLayer>, Option<GpuHc>)
        {
            // qwen4_exp: a hyper-connection (gated residual) block. `block_inject_weight` is
            // [hc, hc*h] (M=4 — never quantized, the quantizer copies it through as bf16).
            let load_hc = |gwn: &mut dyn FnMut(&[String]) -> W, gf: &mut dyn FnMut(&str) -> S,
                           p: &str, inject: bool| -> GpuHc {
                GpuHc {
                    norm: gf(&format!("{p}.hc_norm.weight")),
                    down: gwn(&[format!("{p}.input_mix_weight_down.weight")]),
                    up:   gwn(&[format!("{p}.input_mix_weight_up.weight")]),
                    inject: if inject {
                        match gwn(&[format!("{p}.block_inject_weight.weight")]) {
                            W::Bf16(b) => Some(b),
                            _ => panic!("{p}.block_inject_weight must be bf16 (M=4 cannot be packed)"),
                        }
                    } else { None },
                }
            };
            // qwen4_exp has no layer norms at all: 1-element placeholders keep the struct shape.
            let dummy_s = || dev.alloc_zeros::<f32>(1).unwrap();
            // qwen4_exp QSA indexer: the projection, the two per-head norms, and the device params block.
            let (cos_ptr, sin_ptr) = (*cos_table.device_ptr() as u64, *sin_table.device_ptr() as u64);
            let load_indexer = |gwn: &mut dyn FnMut(&[String]) -> W, gf: &mut dyn FnMut(&str) -> S, p: &str| -> GpuIndexer {
                let q_norm = gf(&format!("{p}.q_layernorm.weight"));
                let k_norm = gf(&format!("{p}.k_layernorm.weight"));
                let params = Self::qsa_params(&dev, &cfg, cos_ptr, sin_ptr, *k_norm.device_ptr() as u64, *q_norm.device_ptr() as u64);
                GpuIndexer { qk_proj: gwn(&[format!("{p}.index_qk_proj.weight")]), q_norm, k_norm, params }
            };
            // Fusion is decided PER TENSOR GROUP: only an all-packed group fuses (bf16 weights are
            // not fusable). A mixed artifact — e.g. `--recipe all,gdn:bf16`, or a `--gptq` output
            // that keeps the GDN / hyper-connections in bf16 — loads its bf16 groups split.
            // (H0.3a: the PR expressed this as a `packed` closure — variants != 3 — which is
            // semantically identical to dev's `Some(1 | 2)` checks in attn_in/gdn_in below; dev's
            // side won, the closure itself is not needed.)
            let attn_in = |gwn: &mut dyn FnMut(&[String]) -> W, lp: &str, fuse: bool| -> AttnIn {
                let n = |s: &str| format!("{}.self_attn.{}.weight", lp, s);
                // Fuse only when EVERY part is quantized: foreign checkpoints (Qwen official
                // FP8) may leave small linears bf16, and a mixed group can't assemble into one
                // packed weight. The split arm dispatches per-part through gemm_act, so parts
                // may be any variant; under TP a split site is replicated (legacy full-width
                // contract — the ColSegs rule keys on the fused 3-name group).
                let parts = [n("q_proj"), n("k_proj"), n("v_proj")];
                let fuse = fuse && parts.iter().all(|nm| matches!(variants.get(nm), Some(1 | 2)));
                if fuse { AttnIn::Fused(gwn(&parts)) }
                else { AttnIn::Split { q: gwn(&[n("q_proj")]), k: gwn(&[n("k_proj")]), v: gwn(&[n("v_proj")]) } }
            };
            let gdn_in = |gwn: &mut dyn FnMut(&[String]) -> W, lp: &str| -> GdnIn {
                let n = |s: &str| format!("{}.linear_attn.{}.weight", lp, s);
                // Same dtype-aware rule: Qwen official FP8 quantizes in_proj_qkv/z but leaves
                // in_proj_b/a bf16 ([48,5120] rows don't tile into 128-blocks), so the 4-name
                // fused group can never assemble there — run the legacy split path.
                let parts = [n("in_proj_qkv"), n("in_proj_z"), n("in_proj_b"), n("in_proj_a")];
                let fuse = quantized && parts.iter().all(|nm| matches!(variants.get(nm), Some(1 | 2)));
                if fuse { GdnIn::Fused(gwn(&parts)) }
                else { GdnIn::Split { qkv: gwn(&[n("in_proj_qkv")]), z: gwn(&[n("in_proj_z")]),
                                      b: gwn(&[n("in_proj_b")]), a: gwn(&[n("in_proj_a")]) } }
            };
            // FFN loader: dense MLP (qwen3_5, hy_v3 layer 0) or the MoE block (qwen3_5_moe, hy_v3).
            // Expert tensors are STACKED and gate+up FUSED, stored WITHOUT a `.weight` suffix; router and
            // shared-MLP carry `.weight`. hy_v3 names differ from qwen3_5_moe: `mlp.router.gate.weight`
            // (not `mlp.gate.weight`), `mlp.expert_bias` [E] fp32, `mlp.shared_mlp.*` (not
            // `shared_expert.*`), and NO `shared_expert_gate` (the shared expert is ungated).
            let load_ffn = |gwn: &mut dyn FnMut(&[String]) -> W, gf: &mut dyn FnMut(&str) -> S,
                            lp: &str, is_moe: bool, q: bool| -> Ffn {
                // The bf16 checkpoint names the stacked experts WITHOUT a `.weight` suffix; the quantizer
                // packs them as `<name>.weight_packed`, which the packed-ingestion keys as `<name>.weight`.
                // `q` = are THESE experts quantized (per-head, so a bf16 MTP head in a quantized model works).
                let esuf = if q { ".weight" } else { "" };
                if is_moe {
                    if hy3 {
                        // The shared expert's gate/up are FUSED at load into one [2*si, h] tensor
                        // (quantized path only — bf16 tensors can't fuse): two [si, h] GEMMs at
                        // si=1536 run at ~115 GB/s (small-M latency); the fused one shares the
                        // experts' layout and `moe_silu_bf16_b` (== silu_mul_b bitwise).
                        // shared.gate/up are never read when shared_gate_up is Some — placeholders.
                        let (smlp, sgu) = if q {
                            (GpuMlp {
                                gate: W::Bf16(dev.alloc_zeros::<half::bf16>(1).unwrap()),
                                up:   W::Bf16(dev.alloc_zeros::<half::bf16>(1).unwrap()),
                                down: gwn(&[format!("{}.mlp.shared_mlp.down_proj.weight", lp)]),
                            },
                            Some(gwn(&[format!("{}.mlp.shared_mlp.gate_proj.weight", lp),
                                       format!("{}.mlp.shared_mlp.up_proj.weight", lp)])))
                        } else {
                            (GpuMlp {
                                gate: gwn(&[format!("{}.mlp.shared_mlp.gate_proj.weight", lp)]),
                                up:   gwn(&[format!("{}.mlp.shared_mlp.up_proj.weight", lp)]),
                                down: gwn(&[format!("{}.mlp.shared_mlp.down_proj.weight", lp)]),
                            }, None)
                        };
                        // E8 (revised): the shared gate+up (fused ColSegs) AND down (Row) are sliced
                        // by hy3_load_shard_op under shard-at-load — each rank reads HALF the shared
                        // weights. Capture the flag BEFORE sgu moves into the struct.
                        // --e8-no-shard=1 disables (the loader then keeps the shared replicated
                        // and the fold runs the replicated path — the byte-identical escape).
                        let e8_on = crate::opts::var(crate::opt!("e8-no-shard")).map_or(true, |v| v != "1");
                        let shared_sharded = e8_on && shared_expert_shard_factor_for(&cfg, world.max(1) as usize, true) > 1
                            && sal.is_some() && lp != &hy3_mtp_prefix && sgu.is_some();
                        // E12: the shared slot's weight geometry for the fold kernels — device
                        // i64[16], written once at load (the fold GEMMs take the descriptor POINTER
                        // as their single extra arg; cudarc launch tuples cap at 12 elements).
                        // [0..4)  gate_up desc { qweight, scales, gs, (SK<<32)|sk_off } with SK = the
                        //         tensor's K (= h, full — the col-shard splits M, not K) and
                        //         sk_off = 0;
                        // [4..8)  down desc with SK = sdn_k (si, or si/2 under the row-parallel
                        //         shard) and sk_off = rank·sdn_k — the rank's K-offset into the
                        //         full act, the exact slice the old act_half copy fed the down GEMM;
                        // [8]     the shared slot's gu M (2·si, or 2·si_local under the shard) — the
                        //         fold GU GEMMs' per-slot M (the launch M stays the routed 2·mi);
                        // [12]    the shared slot's down M (= h, always the launch M) — written so
                        //         the uniform desc[8] read is in-bounds for the down launches too.
                        // None for bf16 shared tensors.
                        let shared_slot_desc = sgu.as_ref().and_then(|guw| match (guw, &smlp.down) {
                            (W::Nvfp4 { qweight: gw, scales: gs_, gs: ggs, m: gu_m, k: gu_k, .. },
                             W::Nvfp4 { qweight: dw, scales: ds_, gs: dgs, m: dwn_m, k, .. }) => {
                                let rank = sal.unwrap_or(0) as i64;
                                let sdn_k = *k as i64;
                                let sk_off = if shared_sharded { rank * sdn_k } else { 0 };
                                let host: Vec<i64> = vec![
                                    *gw.device_ptr() as i64, *gs_.device_ptr() as i64, *ggs.device_ptr() as i64,
                                    (*gu_k as i64) << 32,
                                    *dw.device_ptr() as i64, *ds_.device_ptr() as i64, *dgs.device_ptr() as i64,
                                    (sdn_k << 32) | sk_off,
                                    *gu_m as i64, 0, 0, 0,
                                    *dwn_m as i64, 0, 0, 0,
                                ];
                                Some(dev.htod_sync_copy(&host).unwrap())
                            }
                            _ => None,
                        });
                        return Ffn::Moe(GpuMoe {
                            router:      gwn(&[format!("{}.mlp.router.gate.weight", lp)]),
                            gate_up:     gwn(&[format!("{}.mlp.experts.gate_up_proj{}", lp, esuf)]),
                            down:        gwn(&[format!("{}.mlp.experts.down_proj{}", lp, esuf)]),
                            shared: smlp,
                            shared_gate: None,   // hy_v3's shared expert is UNGATED
                            shared_gate_up: sgu,
                            expert_bias: Some(gf(&format!("{}.mlp.expert_bias", lp))),
                            // Set at load ONLY under shard-at-load (hy_v3 TP) and only for MAIN layers —
                            // the MTP block (layers.80) is replicated (barrier-free drafting) even under
                            // shard-at-load; the qwen TP path still flips it in `tp_shard_weights`.
                            // Requires the expert band to actually be divisible by world, so the flag
                            // agrees with what the slicer did.
                            experts_sharded: experts_shard_factor_for(&cfg, world.max(1) as usize, true) > 1
                                && sal.is_some() && lp != &hy3_mtp_prefix,
                            shared_sharded,
                            shared_slot_desc,
                        });
                    }
                    Ffn::Moe(GpuMoe {
                        router:      gwn(&[format!("{}.mlp.gate.weight", lp)]),
                        gate_up:     gwn(&[format!("{}.mlp.experts.gate_up_proj{}", lp, esuf)]),
                        down:        gwn(&[format!("{}.mlp.experts.down_proj{}", lp, esuf)]),
                        shared: GpuMlp {
                            gate: gwn(&[format!("{}.mlp.shared_expert.gate_proj.weight", lp)]),
                            up:   gwn(&[format!("{}.mlp.shared_expert.up_proj.weight", lp)]),
                            down: gwn(&[format!("{}.mlp.shared_expert.down_proj.weight", lp)]),
                        },
                        shared_gate: Some(gwn(&[format!("{}.mlp.shared_expert_gate.weight", lp)])),
                        shared_gate_up: None,
                        shared_sharded: false,   // E8: flipped by tp_shard_weights at attach (qwen)
                        expert_bias: None,
                        // TP shard-at-load: the experts are sliced to the rank's band — the expert
                        // all-reduce (moe_batch) must run — only when the band is divisible by world.
                        experts_sharded: experts_shard_factor_for(&cfg, world.max(1) as usize, true) > 1
                            && sal.is_some(),
                        // E12: qwen's shared gate/up are separate tensors (no fused gate_up) → the
                        // fold never fires → no descriptor.
                        shared_slot_desc: None,
                    })
                } else {
                    Ffn::Dense(GpuMlp {
                        gate: gwn(&[format!("{}.mlp.gate_proj.weight", lp)]),
                        up:   gwn(&[format!("{}.mlp.up_proj.weight", lp)]),
                        down: gwn(&[format!("{}.mlp.down_proj.weight", lp)]),
                    })
                }
            };

            let embed = gwn(&[format!("{}.embed_tokens.weight", pref)]);
            let final_norm = if q4 { dummy_s() } else { gf(&format!("{}.norm.weight", pref)) };
            let lm_head = if cfg.tie_word_embeddings { None } else { Some(gwn(&["lm_head.weight".to_string()])) };

            let mut layers = Vec::with_capacity(cfg.num_layers);
            for i in 0..cfg.num_layers {
                let lpref = format!("{}.layers.{}", pref, i);
                let lt = cfg.layer_types[i];
                let (la, fa) = match lt {
                    crate::qwen::LayerType::LinearAttention => {
                        let la = GpuLinearAttn {
                            in_proj: gdn_in(&mut *gwn, &lpref),
                            conv1d: gf(&format!("{}.linear_attn.conv1d.weight", lpref)),
                            a_log: gf(&format!("{}.linear_attn.A_log", lpref)),
                            dt_bias: gf(&format!("{}.linear_attn.dt_bias", lpref)),
                            norm: gf(&format!("{}.linear_attn.norm.weight", lpref)),
                            out_proj: gwn(&[format!("{}.linear_attn.out_proj.weight", lpref)]),
                        };
                        (Some(la), None)
                    }
                    crate::qwen::LayerType::FullAttention => {
                        let fa = GpuFullAttn {
                            qkv: attn_in(&mut *gwn, &lpref, quantized),
                            o_proj: gwn(&[format!("{}.self_attn.o_proj.weight", lpref)]),
                            q_norm: gf(&format!("{}.self_attn.q_norm.weight", lpref)),
                            k_norm: gf(&format!("{}.self_attn.k_norm.weight", lpref)),
                            indexer: if cfg.has_indexer() { Some(load_indexer(&mut *gwn, &mut *gf, &format!("{lpref}.self_attn.indexer"))) } else { None },
                        };
                        // LOAD-TIME SHAPE ASSERT — the `attn_output_gate` convention: q_proj packs
                        // q|gate, i.e. 2*nh*hd rows when the output gate is on (MEASURED
                        // Qwen3.8-Flash-Next: q_proj (12288, 2560) = 2 x 24 x 256 x h; o_proj input
                        // 6144 = nh*hd). This is the SAME convention the 27B uses — a checkpoint that
                        // deviates must fail loudly here, not surface as garbage attention.
                        // (Packed weights only: w_m/w_k return 0 for bf16, whose shape the slice
                        // length cannot recover — the pre-existing helper convention.)
                        // S-B4: under shard-at-load the fused qkv/o_proj arrive PER-RANK
                        // (qwen_load_shard_op ColSegs/Row — q|gate split per head, k/v per kv-head,
                        // o_proj row-sharded), so the assert must mirror the op's own decision
                        // (attn_shard_factor_for(cfg, world, true)); a full-width expectation fired
                        // on the CORRECT rank-0 slice (6656x2560 at world=2) and killed the first
                        // TP=2 bring-up. world==1 / no shard-at-load keeps the full-width check.
                        let attn_shard_a = if sal.is_some() { attn_shard_factor_for(&cfg, world as usize, true) } else { 1 };
                        let (nh_a, nkv_a, hd_a) = (cfg.num_heads / attn_shard_a, cfg.num_kv_heads / attn_shard_a, cfg.head_dim);
                        let qmul_a = if cfg.attn_out_gate() { 2 } else { 1 };
                        if !record_pass.get() {
                        match &fa.qkv {
                            AttnIn::Fused(w) => {
                                let (m, k) = (w_m(w), w_k(w));
                                assert!(m == 0 || (m == nh_a * hd_a * qmul_a + 2 * nkv_a * hd_a && k == cfg.hidden_size),
                                    "{lpref}: fused qkv is {m}x{k}, expected {}x{} (q|gate + k + v rows)",
                                    nh_a * hd_a * qmul_a + 2 * nkv_a * hd_a, cfg.hidden_size);
                            }
                            AttnIn::Split { q, .. } => {
                                let (m, k) = (w_m(q), w_k(q));
                                assert!(m == 0 || (m == nh_a * hd_a * qmul_a && k == cfg.hidden_size),
                                    "{lpref}: q_proj is {m}x{k}, expected {}x{} (q|output-gate packing)",
                                    nh_a * hd_a * qmul_a, cfg.hidden_size);
                            }
                        }
                        {
                            let (m, k) = (w_m(&fa.o_proj), w_k(&fa.o_proj));
                            assert!(m == 0 || (m == cfg.hidden_size && k == nh_a * hd_a),
                                "{lpref}: o_proj is {m}x{k}, expected {}x{}", cfg.hidden_size, nh_a * hd_a);
                        }
                        }
                        (None, Some(fa))
                    }
                };
                let mlp = load_ffn(&mut *gwn, &mut *gf, &lpref, cfg.is_moe_layer(i), quantized);
                let (input_ln, post_ln, hc, ple) = if q4 {
                    let hc = (load_hc(&mut *gwn, &mut *gf, &format!("{lpref}.attn_hyper_connection"), true),
                              load_hc(&mut *gwn, &mut *gf, &format!("{lpref}.mlp_hyper_connection"), true));
                    // The PLE layer: projections + norms + the dilated conv here; the 30 GB n-gram
                    // TABLE is attached after assembly (`attach_ple_table`) — assemble runs twice
                    // (record + real) and the table must be read exactly once.
                    let ple = if cfg.ple_layer == Some(i) {
                        let pp = format!("{lpref}.ple");
                        let hash = crate::ple::PleHash::new(&cfg);
                        let heads = hash.ngram_heads();
                        Some(GpuPle {
                            key_proj: gwn(&[format!("{pp}.key_proj.weight")]),
                            value_proj: gwn(&[format!("{pp}.value_proj.weight")]),
                            norm_key: gf(&format!("{pp}.norm_key.weight")),
                            norm_query: gf(&format!("{pp}.norm_query.weight")),
                            norm_conv: gf(&format!("{pp}.norm_conv.weight")),
                            conv1d: gf(&format!("{pp}.conv1d.weight")),
                            hash_tab: dev.htod_sync_copy(&hash.device_table()).unwrap(),
                            hash,
                            table: PleTable::Device(dev.alloc_zeros::<u8>(crate::quant::PLE_REC_BYTES).unwrap()),
                            gs: dev.alloc_zeros::<f32>(1).unwrap(),
                            rows_per_shard: 1,
                            stage_rows: PLE_STAGE_ROWS,
                            stage: dev.alloc_zeros::<u8>(PLE_STAGE_ROWS * heads * crate::quant::PLE_REC_BYTES).unwrap(),
                            ids: dev.alloc_zeros::<i64>(PLE_STAGE_ROWS * heads).unwrap(),
                        })
                    } else { None };
                    (dummy_s(), dummy_s(), Some(hc), ple)
                } else {
                    (gf(&format!("{}.input_layernorm.weight", lpref)),
                     gf(&format!("{}.post_attention_layernorm.weight", lpref)), None, None)
                };
                layers.push(GpuLayer { layer_type: lt, la, fa, mlp, input_ln, post_ln, hc, ple });
            }
            let hc_mixer = if q4 { Some(load_hc(&mut *gwn, &mut *gf, &format!("{pref}.hyper_connection_mixer"), false)) } else { None };

            // Load MTP if present
            let mtp = if has_mtp {
                if crate::boot_report::debug() {
                    println!("Loading MTP head...");
                }
                if is_hy3 {
                    // hy_v3's MTP block is `model.layers.80.*`: eh_proj (=fc), enorm/hnorm (=pre_fc
                    // norms), final_layernorm (=mtp.norm), and a full MoE decoder layer. One-to-one
                    // with GpuMtpLayer; the head it feeds is the MAIN lm_head (shared), as in HF.
                    let mp = &hy3_mtp_prefix;
                    Some(GpuMtpLayer {
                        fc: gwn(&[format!("{mp}.eh_proj.weight")]),
                        pre_fc_norm_hidden: gf(&format!("{mp}.hnorm.weight")),
                        pre_fc_norm_embedding: gf(&format!("{mp}.enorm.weight")),
                        input_ln: gf(&format!("{mp}.input_layernorm.weight")),
                        post_ln: gf(&format!("{mp}.post_attention_layernorm.weight")),
                        fa: GpuFullAttn {
                            qkv: attn_in(&mut *gwn, mp, mtp_quant),
                            o_proj: gwn(&[format!("{mp}.self_attn.o_proj.weight")]),
                            q_norm: gf(&format!("{mp}.self_attn.q_norm.weight")),
                            k_norm: gf(&format!("{mp}.self_attn.k_norm.weight")),
                            indexer: None,
                        },
                        mlp: load_ffn(&mut *gwn, &mut *gf, mp, cfg.is_moe, mtp_quant),
                        final_norm: gf(&format!("{mp}.final_layernorm.weight")),
                        fc_sharded: false, attn_sharded: false, mlp_sharded: false, q4: None,
                    })
                } else if q4 {
                    // qwen4_exp: mtp.fc_embedding / mtp.fc_hidden [h,h], pre_fc_norm_hidden [hc*h],
                    // pre_fc_norm_embedding [h], the layer's two hyper-connections, the head's mixer.
                    let mp = "mtp.layers.0";
                    Some(GpuMtpLayer {
                        fc: W::Bf16(dev.alloc_zeros::<half::bf16>(1).unwrap()),
                        pre_fc_norm_hidden: gf("mtp.pre_fc_norm_hidden.weight"),
                        pre_fc_norm_embedding: gf("mtp.pre_fc_norm_embedding.weight"),
                        input_ln: dummy_s(), post_ln: dummy_s(),
                        fa: GpuFullAttn {
                            qkv: attn_in(&mut *gwn, mp, mtp_quant),
                            o_proj: gwn(&[format!("{mp}.self_attn.o_proj.weight")]),
                            q_norm: gf(&format!("{mp}.self_attn.q_norm.weight")),
                            k_norm: gf(&format!("{mp}.self_attn.k_norm.weight")),
                            indexer: if cfg.has_indexer() { Some(load_indexer(&mut *gwn, &mut *gf, &format!("{mp}.self_attn.indexer"))) } else { None },
                        },
                        mlp: load_ffn(&mut *gwn, &mut *gf, mp, cfg.is_moe, mtp_quant),
                        final_norm: dummy_s(),
                        fc_sharded: false, attn_sharded: false, mlp_sharded: false,
                        q4: Some(GpuMtpQ4 {
                            fc_embedding: gwn(&["mtp.fc_embedding.weight".to_string()]),
                            fc_hidden: gwn(&["mtp.fc_hidden.weight".to_string()]),
                            hc: (load_hc(&mut *gwn, &mut *gf, &format!("{mp}.attn_hyper_connection"), true),
                                 load_hc(&mut *gwn, &mut *gf, &format!("{mp}.mlp_hyper_connection"), true)),
                            mixer: load_hc(&mut *gwn, &mut *gf, "mtp.hyper_connection_mixer", false),
                        }),
                    })
                } else {
                Some(GpuMtpLayer {
                    fc: gwn(&["mtp.fc.weight".to_string()]),
                    pre_fc_norm_hidden: gf("mtp.pre_fc_norm_hidden.weight"),
                    pre_fc_norm_embedding: gf("mtp.pre_fc_norm_embedding.weight"),
                    input_ln: gf("mtp.layers.0.input_layernorm.weight"),
                    post_ln: gf("mtp.layers.0.post_attention_layernorm.weight"),
                    fa: GpuFullAttn {
                        qkv: attn_in(&mut *gwn, "mtp.layers.0", mtp_quant),
                        o_proj: gwn(&["mtp.layers.0.self_attn.o_proj.weight".to_string()]),
                        q_norm: gf("mtp.layers.0.self_attn.q_norm.weight"),
                        k_norm: gf("mtp.layers.0.self_attn.k_norm.weight"),
                        indexer: None,
                    },
                    mlp: load_ffn(&mut *gwn, &mut *gf, "mtp.layers.0", cfg.is_moe, mtp_quant),
                    final_norm: gf("mtp.norm.weight"),
                    fc_sharded: false, attn_sharded: false, mlp_sharded: false, q4: None,
                })
                }
            } else { None };

            (embed, final_norm, lm_head, layers, mtp, hc_mixer)
        };

        // 1) RECORD pass: note the gwn call sequence. Dummy weights of the right VARIANT (the
        // assembly's own variant checks must see the shape the real weight will have); products
        // dropped immediately. gf is cheap map reads in the real pass — dummy here too.
        // R4 (S-H0.4a): the 16x16 dummies rec_gwn hands out below are LOAD-BEARING (variant
        // dispatch, not a bug). The load-time shape asserts skip this pass via the `record_pass`
        // Cell above — a NEW assert must respect that gate, or it will trip on every dummy.
        let mut recorded: Vec<Vec<String>> = Vec::new();
        {
            let mut rec_gwn = |names: &[String]| -> W {
                recorded.push(names.to_vec());
                match variants.get(&names[0]).copied().unwrap_or(3) {
                    1 => W::Nvfp4 { qweight: dev.alloc_zeros::<u8>(1).unwrap(),
                                    scales:  dev.alloc_zeros::<u8>(1).unwrap(),
                                    gs:      dev.alloc_zeros::<f32>(1).unwrap(), m: 16, k: 16 },
                    2 => W::Fp8 { data:      dev.alloc_zeros::<u8>(1).unwrap(),
                                  row_scale: dev.alloc_zeros::<f32>(1).unwrap(), m: 16, k: 16 },
                    _ => W::Bf16(dev.alloc_zeros::<half::bf16>(1).unwrap()),
                }
            };
            let mut rec_gf = |_: &str| -> S { dev.alloc_zeros::<f32>(1).unwrap() };
            drop(assemble(&mut rec_gwn, &mut rec_gf));
        }

        // 2) Jobs: the recorded name-groups (+ shard op + an in-flight-bytes estimate). Parts stay
        //    in the host maps until a worker picks the job up — pulling them ALL out here would
        //    hold ~35 GB (122B) in `jobs` ON TOP of the workers' working sets, which OOM-killed
        //    the 122B load on a 121 GB box (~138 GB peak, measured). The lazy pull keeps the peak
        //    at the serial path's plus one bounded gate. bf16 weights need no assembly — the real
        //    gwn serves them from gpu_bf16.
        enum AssembleJob {
            Q4 { names: Vec<String>, shard_op: Option<LoadShardOp>, est: usize },
            Q8 { names: Vec<String>, shard_op: Option<LoadShardOp>, est: usize },
        }

        // STREAMING LOAD (122B OOM fix, 2026-08-07): the shard loop feeds a LIVE job queue —
        // each shard's parts are drained (fused + repacked + uploaded, then freed) while the
        // next shard is still being read, so the resident raw is ~1-2 shards instead of the
        // whole artifact. The economy/dual-layout decision comes from the header pre-pass
        // totals (the maps drain during the load, so a mid-load sum would be wrong).
        // A fused group's job fires when its LAST part lands (parts may span shards); bf16
        // groups and hy3 inline bands need no job.
        let job_q: std::sync::Arc<(std::sync::Mutex<std::collections::VecDeque<AssembleJob>>,
                                   std::sync::Condvar)> =
            std::sync::Arc::new((std::sync::Mutex::new(std::collections::VecDeque::new()),
                                 std::sync::Condvar::new()));
        let job_done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut pushed: Vec<bool> = vec![false; recorded.len()];
        // Push every recorded group whose parts are ALL present. Called after each shard and
        // once more after the loop (when every group is complete).
        fn push_ready_jobs(recorded: &[Vec<String>], pushed: &mut [bool],
                           hq4: &std::sync::Mutex<std::collections::HashMap<String, Q4H>>,
                           hq8: &std::sync::Mutex<std::collections::HashMap<String, Q8H>>,
                           gpu_q4_sharded: &std::collections::HashMap<String, W>,
                           job_q: &std::sync::Arc<(std::sync::Mutex<std::collections::VecDeque<AssembleJob>>, std::sync::Condvar)>,
                           sal: Option<usize>, cfg: &crate::qwen::Config, world: usize, PIPE_CAP_BYTES: usize,
                           presliced: &std::collections::HashSet<String>) {
            for (idx, names) in recorded.iter().enumerate() {
                if pushed[idx] { continue; }
                if gpu_q4_sharded.contains_key(&names[0]) { pushed[idx] = true; continue; } // hy3 inline bands
                let (g4, g8) = (hq4.lock().unwrap(), hq8.lock().unwrap());
                let in_q4 = g4.contains_key(&names[0]);
                let in_q8 = g8.contains_key(&names[0]);
                if (in_q4 || in_q8) && names.iter().all(|n| g4.contains_key(n) || g8.contains_key(n)) {
                    let shard_op = if sal.is_some() {
                        Some(if presliced.contains(&names[0]) && names.iter().all(|n| presliced.contains(n)) {
                            LoadShardOp::None // B32: shipped already sliced to this rank's band
                        } else if cfg.family == crate::qwen::Family::HyV3 {
                            hy3_load_shard_op(names, cfg, true, world)
                        } else {
                            qwen_load_shard_op(names, cfg, true, world)
                        })
                    } else { None };
                    let job = if in_q4 {
                        let bytes: usize = names.iter().map(|n| {
                            let p = &g4[n]; p.0.len() + p.1.len()
                        }).sum();
                        AssembleJob::Q4 { names: names.clone(), shard_op,
                                          est: (bytes * 2 + 1).min(PIPE_CAP_BYTES) }
                    } else {
                        let bytes: usize = names.iter().map(|n| {
                            let p = &g8[n]; p.0.len() + p.1.len() * 4
                        }).sum();
                        AssembleJob::Q8 { names: names.clone(), shard_op,
                                          est: (bytes * 2 + 1).min(PIPE_CAP_BYTES) }
                    };
                    drop(g4); drop(g8);
                    pushed[idx] = true;
                    job_q.0.lock().unwrap().push_back(job);
                    job_q.1.notify_one();
                }
            }
        }
        if n_dq4 + n_dq8 > 0 {
            if crate::boot_report::debug() {
                println!("  QUANTIZED artifact: {} NVFP4 + {} FP8 tensors ({})", n_dq4, n_dq8,
                     if dequant_at_load { "dequantized to bf16 at load — NO speedup, reference only" }
                     else { "kept packed; dequantized in-register by the fused GEMV" });
            }
        }

        // Per-group round-trip error — the first thing to look at if quality moves.
        if !fq_stats.is_empty() {
            use std::collections::BTreeMap;
            let mut agg: BTreeMap<&str, (usize, f32, f32)> = BTreeMap::new();
            for (n, rel, mx) in &fq_stats {
                let e = agg.entry(crate::quant::group_name(crate::quant::group_of(n)))
                           .or_insert((0, 0.0, 0.0));
                e.0 += 1;
                e.1 += *rel;
                e.2 = e.2.max(*mx);
            }
            println!("  fake-quant round-trip error (relative L2 per tensor, averaged per group):");
            for (g, (n, sr, mx)) in agg {
                println!("    {:<8} {:3} tensors   rel={:.4}   max|Δw|={:.4}", g, n, sr / n as f32, mx);
            }
        }

        // Helper to get weights from the HashMaps
        // The MTP head is a weight like any other and gets quantized too — so this must look in ALL
        // the format maps. Checking only the bf16 map silently drops MTP on every quantized model,
        // costing the entire 1.4-1.6x speculative-decoding lever.
        // The MTP head's MoE experts must be in the FUSED `experts.gate_up_proj` layout the loader
        // ingests. Some checkpoints (122B) store the MTP experts UN-fused (per-expert
        // `experts.<N>.gate_proj/up_proj/down_proj`), which the fused-GEMM MoE path can't consume.
        // MTP is an optional speculative-decode accelerator, so SKIP it (the model decodes correctly
        // without it) rather than panic on a missing fused tensor.
        // hy_v3's MTP block is `model.layers.<num_layers>.*` (80): eh_proj (=fc), enorm/hnorm
        // (=pre_fc norms), final_layernorm (=mtp.norm), and a full decoder layer underneath —
        // one-to-one with GpuMtpLayer (P6).
        // 3) The pipeline: workers assemble host-side (fuse -> repack -> optional host shard) and
        //    hand host buffers to the ONE uploader thread. The handoff is bounded by a bytes gate
        //    (the working set is the peak host memory this adds) and a 2-deep output channel.
        // ECONOMY MODE (memory-gated, HERE — after the shard loop filled host_q4): for models
        // whose fp4 bytes make the dual-layout footprint impossible (122B: ~61 GB × 2 > the
        // 121 GB unified pool — the global OOM of 2026-08-07), the OMMA repack becomes the
        // PRIMARY storage: the W's qweight/scales fields hold the Aimg/SFAw buffers, the standard
        // MMA-repacked buffers are never uploaded, and prefill dequantizes from the OMMA layout
        // (mxfp4_dequant_tiled_b). Below the threshold (27B) both layouts stay resident.
        struct AssembledW { names: Vec<String>, est: usize, w: AssembledInner }
        enum AssembledInner {
            /// `omma` = the lossless OMMA repack (aimg, sfa) — present iff mxfp4-native mode
            /// (computed in the worker next to repack_nvfp4_mma; the default path never sees it).
            Q4 { wt: Vec<u8>, st: Vec<u8>, gsv: Vec<f32>, m: usize, k: usize,
                 omma: Option<(Vec<u8>, Vec<u8>)> },
            Q8 { wt: Vec<u8>, rs: Vec<f32>, m: usize, k: usize, nb_k: usize },
        }
        // RAII gate charge: the worker holds it for the job's lifetime, so a PANICKING worker
        // releases its charge on unwind instead of leaking it and hanging every other worker at
        // the gate (a corrupt artifact must fail the load loudly, like the serial path, never hang).
        struct GateGuard(std::sync::Arc<(std::sync::Mutex<usize>, std::sync::Condvar)>, usize);
        impl GateGuard {
            fn acquire(gate: &std::sync::Arc<(std::sync::Mutex<usize>, std::sync::Condvar)>,
                       est: usize, cap: usize) -> Self {
                let (lock, cvar) = &**gate;
                let mut g = lock.lock().unwrap();
                while *g + est > cap { g = cvar.wait(g).unwrap(); }
                *g += est;
                GateGuard(gate.clone(), est)
            }
        }
        impl Drop for GateGuard {
            fn drop(&mut self) {
                let (lock, cvar) = &*self.0;
                let mut g = lock.lock().unwrap();
                *g = g.saturating_sub(self.1);
                cvar.notify_all();
            }
        }
        let mut results: std::collections::HashMap<String, W> = std::collections::HashMap::new();
        // S9F (TP-DF2 leg): the worker keeps the FULL fused RAW of the lm_head here (before the
        // host-side shard slice) when [df2-tp]=1 — the main thread repacks + uploads it after
        // the pipeline as the round's full-head borrow (the shard-at-load twin of the attach_tp
        // capture; the two paths are mutually exclusive).
        let df2_full_head_raw: std::sync::Arc<std::sync::Mutex<Option<(Vec<u8>, Vec<u8>, Vec<f32>, usize, usize)>>>
            = std::sync::Arc::new(std::sync::Mutex::new(None));
        // Workers pull their job's parts out of these (one short lock per job) — see (2).
        let host_q4 = std::sync::Mutex::new(host_q4);
        let host_q8 = std::sync::Mutex::new(host_q8);
        let nworkers = recorded.len()
            .min(crate::opts::var(crate::opt!("load-workers")).ok().and_then(|v| v.parse().ok()).unwrap_or(8))
            .min(std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))
            .max(1);
        if nworkers > 0 {
            let (tx, rx) = std::sync::mpsc::sync_channel::<AssembledW>(2);
            let gate = std::sync::Arc::new((std::sync::Mutex::new(0usize), std::sync::Condvar::new()));
            let dev_ref = &dev;
            let jq = &job_q;
            let jdone = &job_done;
            let sal_c = sal;
            let world_c = world.max(1) as usize;
            let hq4 = &host_q4;
            let hq8 = &host_q8;
            std::thread::scope(|s| -> anyhow::Result<()> {
                // Bounded async uploads (Phase 1 item 4): issue async htod on the NULL stream and
                // sync every ~2 GB / 64 copies instead of per weight. Pageable async htod reads
                // the source AT EXECUTION, so the source Vecs sit in `keep` until each sync point
                // (the lifetime rule — dropping them early corrupts the upload). The env var is
                // the escape hatch back to per-weight sync copies.
                let sync_uploads = crate::opts::var(crate::opt!("load-sync-upload")).is_ok();
                let up = s.spawn(move || -> (std::collections::HashMap<String, W>,
                                             std::collections::HashMap<u64, crate::mxfp4::OmmaEntry>,
                                             std::collections::HashSet<u64>,
                                             std::time::Duration) {
                    let mut out = std::collections::HashMap::new();
                    // mxfp4-native mode: OMMA repacks keyed by the qweight device pointer.
                    let mut omma_map: std::collections::HashMap<u64, crate::mxfp4::OmmaEntry> = std::collections::HashMap::new();
                    // Sensitive-tensor allowlist (MTP head under GB10_MXFP4_MTP_BF16): tensors
                    // that intentionally run the bf16 chain even in native mode.
                    let mut allow_bf16: std::collections::HashSet<u64> = std::collections::HashSet::new();
                    // --pf-mixer4 (load-time): which mixer projections may run the W4A4 v2
                    // PREFILL GEMM. 'safe' = every is_qkv_projection_name tensor EXCEPT the GDN
                    // fused in_proj (its a/b/z rows feed the recurrence — the documented
                    // amplification point); 'all'/'1' = everything (known MISMATCH, diagnostic).
                    let m4env = crate::opts::var(crate::opt!("pf-mixer4")).unwrap_or_default();
                    let m4_all = m4env == "all" || m4env == "1";
                    let m4_safe = m4_safe_on(&m4env);
                    let mut mixer4: std::collections::HashSet<u64> = std::collections::HashSet::new();
                    let mut t_up = std::time::Duration::ZERO;
                    let mut keep: Vec<AssembledInner> = Vec::new();
                    let mut queued_bytes = 0usize;
                    let mut queued_copies = 0usize;
                    while let Ok(a) = rx.recv() {
                        let t = std::time::Instant::now();
                        let w = match a.w {
                            AssembledInner::Q4 { wt, st, gsv, m, k, omma } => {
                                if mxfp4_economy && omma.is_some() {
                                    // ECONOMY: the OMMA repack is the primary storage — the W's
                                    // qweight/scales ARE Aimg/SFAw (one layout resident, not two;
                                    // prefill dequantizes from it via mxfp4_dequant_tiled_b).
                                    let (aimg, sfa) = omma.unwrap();
                                    let qweight = dev_ref.htod_sync_copy(&aimg).unwrap();
                                    let scales  = dev_ref.htod_sync_copy(&sfa).unwrap();
                                    let gs      = dev_ref.htod_sync_copy(&gsv).unwrap();
                                    omma_map.insert(*qweight.device_ptr() as u64,
                                                    crate::mxfp4::OmmaEntry::Ptr(*qweight.device_ptr() as u64,
                                                                                 *scales.device_ptr() as u64));
                                    W::Nvfp4 { qweight, scales, gs, m, k }
                                } else if is_qkv_projection_name(&a.names[0])
                                    && crate::opts::var(crate::opt!("no-pf-qkv-fast")).is_err() {
                                    // Expert R4 Q3(b) run 1: Q/K/V projections keep the bf16 chain —
                                    // their quant noise feeds the attention softmax, which amplifies
                                    // it at the PF_BR=1024 query-tile boundary (the >1024 ppl cliff).
                                    // 2026-08-26 (P4 TTFT): the OMMA repack bytes are STILL uploaded
                                    // and mapped — decode/verify keep the allowlisted bf16 chain,
                                    // but the v2 PREFILL GEMM (prefill-only callers) reads the OMMA
                                    // layouts (nsys: mixer dequant+cuBLAS was 24.8% = 5.4s at 32K).
                                    let qweight = dev_ref.htod_sync_copy(&wt).unwrap();
                                    allow_bf16.insert(*qweight.device_ptr() as u64);
                                    let m4_ok = m4_all
                                        || (m4_safe && !a.names[0].contains("in_proj"))
                                        || m4_member(&m4env, &a.names[0]);
                                    if m4_ok {
                                        if let Some((aimg, sfa)) = omma {
                                            let d_a = dev_ref.htod_sync_copy(&aimg).unwrap();
                                            let d_s = dev_ref.htod_sync_copy(&sfa).unwrap();
                                            omma_map.insert(*qweight.device_ptr() as u64,
                                                            crate::mxfp4::OmmaEntry::Owned(d_a, d_s));
                                        }
                                        mixer4.insert(*qweight.device_ptr() as u64);
                                    }
                                    W::Nvfp4 {
                                        qweight,
                                        scales:  dev_ref.htod_sync_copy(&st).unwrap(),
                                        gs:      dev_ref.htod_sync_copy(&gsv).unwrap(), m, k }
                                } else if !mxfp4_mtp_native && is_mtp_head_name(&a.names[0], is_hy3, hy3_num_layers) {
                                    // allowlisted: bf16 chain runs this tensor; remember the ptr
                                    let qweight = dev_ref.htod_sync_copy(&wt).unwrap();
                                    allow_bf16.insert(*qweight.device_ptr() as u64);
                                    W::Nvfp4 {
                                        qweight,
                                        scales:  dev_ref.htod_sync_copy(&st).unwrap(),
                                        gs:      dev_ref.htod_sync_copy(&gsv).unwrap(), m, k }
                                } else if design_stay_list(a.names[0].as_str()) || experts_stay_list(a.names[0].as_str()) {
                                    // DESIGN §1/§3.3 stay-list: the LM head and the embed run the
                                    // bf16 chain even in native mode. The LM head's logits decide
                                    // the argmax — a W4A4 head re-injects the activation-quant
                                    // error at the decision point and flips near-tie tokens
                                    // (measured 2026-08-07: prefill hiddens bit-identical across
                                    // chains, yet a0 = 220 native vs 561 bf16 at an 821-token
                                    // prompt). The embed is a gather (no activation quant) but the
                                    // design keeps it on the standard path too. ~1% of the step
                                    // budget, like the MTP head.
                                    let qweight = dev_ref.htod_sync_copy(&wt).unwrap();
                                    allow_bf16.insert(*qweight.device_ptr() as u64);
                                    W::Nvfp4 {
                                        qweight,
                                        scales:  dev_ref.htod_sync_copy(&st).unwrap(),
                                        gs:      dev_ref.htod_sync_copy(&gsv).unwrap(), m, k }
                                } else if sync_uploads {
                                    let qweight = dev_ref.htod_sync_copy(&wt).unwrap();
                                    if let Some((aimg, sfa)) = omma {
                                        let d_a = dev_ref.htod_sync_copy(&aimg).unwrap();
                                        let d_s = dev_ref.htod_sync_copy(&sfa).unwrap();
                                        omma_map.insert(*qweight.device_ptr() as u64,
                                                        crate::mxfp4::OmmaEntry::Owned(d_a, d_s));
                                    }
                                    W::Nvfp4 {
                                        qweight,
                                        scales:  dev_ref.htod_sync_copy(&st).unwrap(),
                                        gs:      dev_ref.htod_sync_copy(&gsv).unwrap(), m, k }
                                } else {
                                    let qweight = unsafe { dev_ref.alloc::<u8>(wt.len()).unwrap() };
                                    let scales  = unsafe { dev_ref.alloc::<u8>(st.len()).unwrap() };
                                    let gs      = unsafe { dev_ref.alloc::<f32>(gsv.len()).unwrap() };
                                    unsafe {
                                        cudarc::driver::result::memcpy_htod_async(
                                            *qweight.device_ptr(), &wt, std::ptr::null_mut()).unwrap();
                                        cudarc::driver::result::memcpy_htod_async(
                                            *scales.device_ptr(), &st, std::ptr::null_mut()).unwrap();
                                        cudarc::driver::result::memcpy_htod_async(
                                            *gs.device_ptr(), &gsv, std::ptr::null_mut()).unwrap();
                                    }
                                    queued_bytes += wt.len() + st.len() + gsv.len() * 4;
                                    queued_copies += 3;
                                    if let Some((aimg, sfa)) = omma {
                                        let d_a = unsafe { dev_ref.alloc::<u8>(aimg.len()).unwrap() };
                                        let d_s = unsafe { dev_ref.alloc::<u8>(sfa.len()).unwrap() };
                                        unsafe {
                                            cudarc::driver::result::memcpy_htod_async(
                                                *d_a.device_ptr(), &aimg, std::ptr::null_mut()).unwrap();
                                            cudarc::driver::result::memcpy_htod_async(
                                                *d_s.device_ptr(), &sfa, std::ptr::null_mut()).unwrap();
                                        }
                                        queued_bytes += aimg.len() + sfa.len();
                                        queued_copies += 2;
                                        omma_map.insert(*qweight.device_ptr() as u64,
                                                        crate::mxfp4::OmmaEntry::Owned(d_a, d_s));
                                        // lifetime rule: sources live until the next sync point
                                        keep.push(AssembledInner::Q4 { wt, st, gsv, m, k, omma: Some((aimg, sfa)) });
                                    } else {
                                        keep.push(AssembledInner::Q4 { wt, st, gsv, m, k, omma: None });
                                    }
                                    W::Nvfp4 { qweight, scales, gs, m, k }
                                }
                            }
                            AssembledInner::Q8 { wt, rs, m, k, nb_k } => {
                                if sync_uploads {
                                    if nb_k > 0 {
                                        W::Fp8Blk {
                                            data:      dev_ref.htod_sync_copy(&wt).unwrap(),
                                            blk_scale: dev_ref.htod_sync_copy(&rs).unwrap(),
                                            nb_k, m, k }
                                    } else {
                                        W::Fp8 {
                                            data:      dev_ref.htod_sync_copy(&wt).unwrap(),
                                            row_scale: dev_ref.htod_sync_copy(&rs).unwrap(), m, k }
                                    }
                                } else {
                                    let data = unsafe { dev_ref.alloc::<u8>(wt.len()).unwrap() };
                                    let scales = unsafe { dev_ref.alloc::<f32>(rs.len()).unwrap() };
                                    unsafe {
                                        cudarc::driver::result::memcpy_htod_async(
                                            *data.device_ptr(), &wt, std::ptr::null_mut()).unwrap();
                                        cudarc::driver::result::memcpy_htod_async(
                                            *scales.device_ptr(), &rs, std::ptr::null_mut()).unwrap();
                                    }
                                    queued_bytes += wt.len() + rs.len() * 4;
                                    queued_copies += 2;
                                    keep.push(AssembledInner::Q8 { wt, rs, m, k, nb_k });
                                    if nb_k > 0 {
                                        W::Fp8Blk { data, blk_scale: scales, nb_k, m, k }
                                    } else {
                                        W::Fp8 { data, row_scale: scales, m, k }
                                    }
                                }
                            }
                        };
                        t_up += t.elapsed();
                        out.insert(akey(&a.names), w);
                        if !sync_uploads && (queued_bytes >= (2 << 30) || queued_copies >= 64) {
                            let t = std::time::Instant::now();
                            dev_ref.synchronize().unwrap();
                            keep.clear();           // every queued copy has executed: sources consumed
                            queued_bytes = 0;
                            queued_copies = 0;
                            t_up += t.elapsed();
                        }
                    }
                    if !sync_uploads {
                        dev_ref.synchronize().unwrap();
                        drop(keep);
                    }
                    (out, omma_map, allow_bf16, t_up)
                });
                let mut handles = Vec::new();
                for _ in 0..nworkers {
                    let gate_w = gate.clone();
                    let tx_w = tx.clone();
                    let jq_w = jq.clone();
                    let jdone_w = jdone.clone();
                    let hq4_w = hq4;
                    let hq8_w = hq8;
                    let world_w = world_c;
                    let t_omma_w = t_omma.clone();
                    let df2_full_head_raw_w = df2_full_head_raw.clone();
                    handles.push(s.spawn(move || -> (std::time::Duration, std::time::Duration, std::time::Duration) {
                        // The pool is the parallelism — no nested repack threads (see quant.rs).
                        crate::quant::set_repack_threading(false);
                        let jq = jq_w;
                        let jdone = jdone_w;
                        let (mut tf, mut tr, mut th) =
                            (std::time::Duration::ZERO, std::time::Duration::ZERO, std::time::Duration::ZERO);
                        loop {
                            let job = {
                                let (lock, cvar) = &*jq;
                                let mut g = lock.lock().unwrap();
                                loop {
                                    if let Some(j) = g.pop_front() { break Some(j); }
                                    if jdone.load(std::sync::atomic::Ordering::Relaxed) { break None; }
                                    g = cvar.wait(g).unwrap();
                                }
                            };
                            let job = match job { Some(j) => j, None => break };
                            let est = match &job { AssembleJob::Q4 { est, .. } => *est,
                                                  AssembleJob::Q8 { est, .. } => *est };
                            // Held until the job is sent (and released on panic unwind).
                            let _charge = GateGuard::acquire(&gate_w, est, PIPE_CAP_BYTES);
                            let a = match &job {
                                AssembleJob::Q4 { names, shard_op, est } => {
                                    // The parts move out of the shared map HERE (see (2)) — their
                                    // memory is freed on this worker as the job completes.
                                    let parts: Vec<Q4H> = {
                                        let mut m4 = hq4_w.lock().unwrap();
                                        names.iter()
                                            .map(|n| m4.remove(n).unwrap_or_else(|| panic!("missing nvfp4 tensor: {}", n)))
                                            .collect()
                                    };
                                    let k = parts[0].4;
                                    let m: usize = parts.iter().map(|p| p.3).sum();
                                    let refs: Vec<(&[u8], &[u8], f32, usize)> =
                                        parts.iter().map(|p| (&p.0[..], &p.1[..], p.2, p.3)).collect();
                                    let t = std::time::Instant::now();
                                    let (qw, sc, gsv) = crate::quant::fuse_nvfp4(&refs, k);
                                    tf += t.elapsed();
                                    // S9F (TP-DF2 leg): keep the FULL fused RAW of the lm_head for
                                    // the DFlash2 round BEFORE the host-side shard slice — the
                                    // round's borrowed head GEMM needs all v rows; the rank-local
                                    // shard keeps only v/world. [df2-tp]=1 is set by the head
                                    // (spec-source is a DF2 variant) and installed by the node from
                                    // the shipped config — both ranks keep the full head (SPMD).
                                    if crate::opts::var(crate::opt!("df2-tp")).is_ok() && names.len() == 1
                                        && names[0] == "lm_head.weight" {
                                        let full = (qw.clone(), sc.clone(), gsv.clone(), m, k);
                                        *df2_full_head_raw_w.lock().unwrap() = Some(full);
                                    }
                                    // Shard-at-load: slice the FUSED RAW to rank r's shard BEFORE
                                    // the repacks — byte-identical to repacking then slicing, and
                                    // the ONLY option in economy mode (no standard repack exists).
                                    let (qw, sc, gsv, m, k) = if let Some(op) = &shard_op {
                                        shard_raw_nvfp4(&qw, &sc, &gsv, m, k, op, sal_c.unwrap(), world_w)
                                    } else { (qw, sc, gsv, m, k) };
                                    let t = std::time::Instant::now();
                                    let repack_omma = mxfp4_mode && (mxfp4_mtp_native || !is_mtp_head_name(&names[0], is_hy3, hy3_num_layers));
                                    let (wt, st) = if mxfp4_economy && repack_omma {
                                        // The OMMA repack is built from the fused RAW below; the
                                        // standard MMA repack is never materialized — the workers
                                        // then hold ~1.1x per job instead of ~2.25x (earlyoom
                                        // lesson: the 122B double-repack transient tripped the
                                        // daemon and it killed bystanders too, 2026-08-07).
                                        // ALLOWLISTED tensors (the MTP head) still get the
                                        // standard repack — their bf16 chain reads it.
                                        (Vec::new(), Vec::new())
                                    } else {
                                        crate::quant::repack_nvfp4_mma(&qw, &sc, m, k)
                                    };
                                    tr += t.elapsed();
                                    let omma = if repack_omma {
                                        let t = std::time::Instant::now();
                                        let r = if mxfp4_economy {
                                            crate::mxfp4::repack_nvfp4_omma_raw(&qw, &sc, m, k)
                                        } else {
                                            crate::mxfp4::repack_nvfp4_omma(&wt, &st, m, k)
                                        };
                                        *t_omma_w.lock().unwrap() += t.elapsed();
                                        Some(r)
                                    } else { None };
                                    AssembledW { names: names.clone(), est: *est,
                                                 w: AssembledInner::Q4 { wt, st, gsv, m, k, omma } }
                                }
                                AssembleJob::Q8 { names, shard_op, est } => {
                                    let parts: Vec<Q8H> = {
                                        let mut m8 = hq8_w.lock().unwrap();
                                        names.iter()
                                            .map(|n| m8.remove(n).unwrap_or_else(|| panic!("missing fp8 tensor: {}", n)))
                                            .collect()
                                    };
                                    let k = parts[0].3;
                                    let nb_k = parts[0].4;
                                    let m: usize = parts.iter().map(|p| p.2).sum();
                                    let refs: Vec<(&[u8], &[f32], usize)> =
                                        parts.iter().map(|p| (&p.0[..], &p.1[..], p.2)).collect();
                                    let t = std::time::Instant::now();
                                    let (qw, rs) = crate::quant::fuse_fp8_len(&refs, k, nb_k);
                                    tf += t.elapsed();
                                    // Shard-at-load: slice the fused RAW fp8 to rank r's shard.
                                    let (qw, rs, m, k, nb_k) = if let Some(op) = &shard_op {
                                        if nb_k > 0 {
                                            shard_raw_fp8_blk(&qw, &rs, m, k, nb_k, op, sal_c.unwrap(), world_w)
                                        } else {
                                            let (a, b, c, d) = shard_raw_fp8(&qw, &rs, m, k, op, sal_c.unwrap(), world_w);
                                            (a, b, c, d, 0usize)
                                        }
                                    } else { (qw, rs, m, k, nb_k) };
                                    let t = std::time::Instant::now();
                                    let wt = crate::quant::repack_fp8_mma(&qw, m, k);
                                    tr += t.elapsed();
                                    AssembledW { names: names.clone(), est: *est,
                                                 w: AssembledInner::Q8 { wt, rs, m, k, nb_k } }
                                }
                            };
                            // Backpressure lives here too: the channel is 2 deep.
                            let _ = tx_w.send(a);
                        }
                        (tf, tr, th)
                    }));
                }
                // ---- STREAMING SHARD LOOP (inside the pipeline scope) ----
                // Each shard's parts are drained by the workers while the next shard is read,
                // so the resident raw stays ~1-2 shards (the 122B previously held all ~56 GB
                // until every shard was read — the OOM of 2026-08-07).
                // S-B9-REL-PPL (the v0.7.3 `--perplexity` HANG): a PANIC in the shard loop below (the
                // bytemuck alignment panic on an EXL3 pack was the first) skipped the `jdone` store +
                // notify of the Err path, so every worker sat in `cvar.wait` forever and
                // `std::thread::scope` — which joins its threads BEFORE it resumes an unwind — never
                // returned: the panic message, then a silent hang. This guard runs on every exit of
                // the scope body (return, `?`, unwind) and releases the workers; the lock makes the
                // store+notify atomic against a worker that is between its `jdone` check and its wait.
                let _release_workers = ReleaseWorkers(|| {
                    let _held = jq.0.lock().unwrap_or_else(|e| e.into_inner());
                    jdone.store(true, std::sync::atomic::Ordering::Relaxed);
                    jq.1.notify_all();
                });
                let shard_result: anyhow::Result<()> = (|| -> anyhow::Result<()> {
        // BASICS (owner 2026-11-02): the shard read as a percentage of the pack's bytes, at the normal level
        let shard_bytes: Vec<u64> = safetensors_files.iter()
            .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)).collect();
        let load_prog = crate::boot_report::Progress::new("loading weights", shard_bytes.iter().sum(),
                                                          crate::boot_report::Unit::Bytes);
        for (i, sf_path) in safetensors_files.iter().enumerate() {
            let t_iter0 = std::time::Instant::now();
            if safetensors_files.len() > 1 {
                println!("  Shard {}/{}: {}", i+1, safetensors_files.len(),
                         sf_path.file_name().unwrap_or_default().to_string_lossy());
            }
            // Overlap the next shard's disk read with this shard's CPU work (see `readahead`).
            if let Some(next) = safetensors_files.get(i + 1) { readahead(next); }
            let raw = {
                let t = std::time::Instant::now();
                // Drop the shard's PAGE-CACHE pages after the read: the 72 GB 122B artifact
                // would otherwise leave ~40 GB of cache resident while the unified device copy
                // grows — the kernel OOM-kills at the crossover even with the streaming drain
                // (measured 2026-08-07, 22:16:33: total-vm 174 GB at the kill).
                let r = {
                    let mut f = std::fs::File::open(sf_path)?;
                    let mut buf = Vec::with_capacity(f.metadata().map(|m| m.len() as usize).unwrap_or(0));
                    use std::io::Read;
                    f.read_to_end(&mut buf)?;
                    let fd = std::os::unix::io::AsRawFd::as_raw_fd(&f);
                    unsafe { libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED); }
                    buf
                };
                t_read += t.elapsed();
                r
            };
            let st = SafeTensors::deserialize(&raw)?;

            // --- NVFP4 groups: upload PACKED. Keeping them packed IS the speedup; dequantizing
            // here would stream bf16 bytes per token again and gain nothing. ---
            let packed: Vec<String> = st.names().iter()
                .filter(|n| n.ends_with(".weight_packed"))
                .map(|n| n.to_string()).collect();
            for pname in &packed {
                let stem = pname.trim_end_matches(".weight_packed");
                let pv = st.tensor(pname)?;
                let sv = st.tensor(&format!("{}.weight_scale", stem))?;
                let gv = st.tensor(&format!("{}.weight_global_scale", stem))?;
                let (m, k) = (pv.shape()[0], pv.shape()[1] * 2);
                let gs = f32::from_le_bytes(gv.data()[..4].try_into().unwrap());
                if let Ok(iv) = st.tensor(&format!("{}.input_global_scale", stem)) {
                    igs_by_name.insert(stem.to_string(), f32::from_le_bytes(iv.data()[..4].try_into().unwrap()));
                }
                if dequant_at_load {
                    let q = crate::quant::Nvfp4Tensor {
                        qweight: pv.data().to_vec(), scales: sv.data().to_vec(),
                        global_scale: gs, m, k };
                    let bv = crate::quant::dequantize_nvfp4(&q);
                    let t = std::time::Instant::now();
                    gpu_bf16.insert(format!("{}.weight", stem), dev.htod_sync_copy(&bv).unwrap());
                    t_up_shard += t.elapsed();
                } else if sal.is_some() && is_hy3_main_expert(stem, &cfg) {
                    // TP shard-at-load: rank `r` owns the expert band [r·ne/world, +ne/world) of the
                    // stacked [ne·me, K] tensor — a contiguous ROW range (rows are contiguous
                    // pre-repack), so slice → repack → upload, byte-identical to the post-load
                    // `shard_nvfp4_col` result (per-16-row-tile independence of the MMA repack).
                    let rank = sal.unwrap();
                    let ne = cfg.num_experts;
                    let world = world.max(1) as usize;
                    assert!(m % ne == 0 && ne % world == 0,
                            "hy3 expert stack {stem}: M={m} not divisible by {ne} experts across world {world}");
                    let me = m / ne;
                    assert!(me % 16 == 0, "hy3 expert stack {stem}: {me} rows/expert not tile-aligned");
                    let (r0, rn) = (rank * (ne / world) * me, (ne / world) * me);
                    // mxfp4-native mode: the inline expert path must mirror the worker path's
                    // repack rule, or the native MoE arms find no OMMA repack (the C5/J8 bug
                    // class; the qwen 122B never hit it because ITS experts go through the job
                    // pipeline — hy3's inline slice predates mxfp4). In economy the OMMA repack
                    // is the PRIMARY storage (qweight/scales ARE Aimg/SFAw); in dual-layout the
                    // standard repack stays and the OMMA twin is registered alongside.
                    if mxfp4_mode && mxfp4_economy {
                        let t = std::time::Instant::now();
                        let (aimg, sfa) = crate::mxfp4::repack_nvfp4_omma_raw(
                            &pv.data()[r0 * (k / 2)..(r0 + rn) * (k / 2)],
                            &sv.data()[r0 * (k / 16)..(r0 + rn) * (k / 16)], rn, k);
                        t_repack_inline += t.elapsed();
                        let gsv = vec![1.0f32 / gs; rn / 16];
                        let t = std::time::Instant::now();
                        let qweight = dev.htod_sync_copy(&aimg).unwrap();
                        let scales  = dev.htod_sync_copy(&sfa).unwrap();
                        omma_map.insert(*qweight.device_ptr() as u64,
                                        crate::mxfp4::OmmaEntry::Ptr(*qweight.device_ptr() as u64,
                                                                     *scales.device_ptr() as u64));
                        gpu_q4_sharded.insert(format!("{}.weight", stem), W::Nvfp4 {
                            qweight, scales,
                            gs: dev.htod_sync_copy(&gsv).unwrap(), m: rn, k });
                        t_up_shard += t.elapsed();
                    } else {
                        let t = std::time::Instant::now();
                        let (wt, st2) = crate::quant::repack_nvfp4_mma(
                            &pv.data()[r0 * (k / 2)..(r0 + rn) * (k / 2)],
                            &sv.data()[r0 * (k / 16)..(r0 + rn) * (k / 16)], rn, k);
                        t_repack_inline += t.elapsed();
                        let gsv = vec![1.0f32 / gs; rn / 16];
                        let t = std::time::Instant::now();
                        let qweight = dev.htod_sync_copy(&wt).unwrap();
                        if mxfp4_mode {
                            // dual-layout: register the OMMA twin for the native arms
                            let (aimg, sfa) = crate::mxfp4::repack_nvfp4_omma(&wt, &st2, rn, k);
                            let d_a = dev.htod_sync_copy(&aimg).unwrap();
                            let d_s = dev.htod_sync_copy(&sfa).unwrap();
                            omma_map.insert(*qweight.device_ptr() as u64,
                                            crate::mxfp4::OmmaEntry::Owned(d_a, d_s));
                        }
                        gpu_q4_sharded.insert(format!("{}.weight", stem), W::Nvfp4 {
                            qweight,
                            scales: dev.htod_sync_copy(&st2).unwrap(),
                            gs: dev.htod_sync_copy(&gsv).unwrap(),
                            m: rn, k });
                        t_up_shard += t.elapsed();
                    }
                } else {
                    host_q4.lock().unwrap().insert(format!("{}.weight", stem), (
                        pv.data().to_vec(), sv.data().to_vec(),
                        1.0f32 / gs,           // kernel multiplies, so pre-invert once here
                        m, k));
                    // STREAMING LOAD: reserve the LM-head/embed parts for the FR-Spec draft
                    // head BEFORE the head's own job consumes them from host_q4 (only when
                    // --draft-vocab asked for one: the default load holds no extra copy).
                    let key = format!("{}.weight", stem);
                    if !draft_rows.is_empty() && key == head_name && draft_parts_q4.is_none() {
                        draft_parts_q4 = Some((pv.data().to_vec(), sv.data().to_vec(), 1.0f32 / gs, m, k));
                    }
                }

            }

            // --- FP8 tensors (dtype F8_E4M3 + an f32 row-scale sibling) ---
            let fp8: Vec<String> = st.names().iter()
                .filter(|n| n.ends_with(".weight"))
                .filter(|n| st.tensor(n).map(|t| t.dtype() == Dtype::F8_E4M3).unwrap_or(false))
                .map(|n| n.to_string()).collect();
            for wname in &fp8 {
                let stem = wname.trim_end_matches(".weight");
                let wv = st.tensor(wname)?;
                let (m, k) = (wv.shape()[0], wv.shape()[1]);
                // Scale sibling: legacy per-row f32 `weight_scale` [M], or Qwen fine-grained
                // `weight_scale_inv` BF16 [M/128, K/128] — w = code·scale_inv per 128x128 block
                // (direction verified numerically against the BF16 source: rel-L2 2.65e-2).
                let (scales, nb_k) = if let Some(sv) = st.tensor(&format!("{}.weight_scale", stem)).ok() {
                    (crate::podread::pod_vec::<f32>(sv.data()), 0usize)
                } else {
                    let sv = st.tensor(&format!("{}.weight_scale_inv", stem))?;
                    assert_eq!(sv.dtype(), Dtype::BF16, "weight_scale_inv must be BF16: {}", wname);
                    assert!(m % 128 == 0 && k % 128 == 0,
                            "block-128 fp8 needs M,K % 128 == 0: {} ({}x{})", wname, m, k);
                    assert_eq!(sv.shape(), &[m / 128, k / 128],
                               "weight_scale_inv shape mismatch: {}", wname);
                    (crate::podread::pod_cow::<half::bf16>(sv.data()).iter()
                        .map(|x| x.to_f32()).collect::<Vec<f32>>(), k / 128)
                };
                if dequant_at_load {
                    let bv: Vec<half::bf16> = if nb_k > 0 {
                        // Exact block dequant: reuse the row codec with unit row scales, then
                        // apply scale_inv per 128x128 block (one multiply, no requantization).
                        let q = crate::quant::Fp8Tensor {
                            qweight: wv.data().to_vec(), row_scale: vec![1.0f32; m], m, k };
                        crate::quant::dequantize_fp8(&q).iter().enumerate().map(|(i, v)| {
                            let (row, col) = (i / k, i % k);
                            half::bf16::from_f32(v.to_f32() * scales[(row / 128) * nb_k + (col / 128)])
                        }).collect()
                    } else {
                        let q = crate::quant::Fp8Tensor {
                            qweight: wv.data().to_vec(), row_scale: scales.clone(), m, k };
                        crate::quant::dequantize_fp8(&q)
                    };
                    let t = std::time::Instant::now();
                    gpu_bf16.insert(wname.clone(), dev.htod_sync_copy(&bv).unwrap());
                    t_up_shard += t.elapsed();
                } else {
                    host_q8.lock().unwrap().insert(wname.clone(), (wv.data().to_vec(), scales.clone(), m, k, nb_k));
                    if !draft_rows.is_empty() && wname == &head_name && nb_k == 0 && draft_parts_q8.is_none() {
                        draft_parts_q8 = Some((wv.data().to_vec(), scales, m, k));
                    }
                }

            }

            for (name, view) in st.tensors() {
                // Skip anything already handled as part of a quantized group.
                if name.ends_with(".weight_packed") || name.ends_with(".weight_scale")
                    || name.ends_with(".weight_global_scale") || name.ends_with(".input_global_scale") { continue; }
                if view.dtype() == Dtype::F8_E4M3 { continue; }
                let dt = match view.dtype() { Dtype::BF16 => "BF16", Dtype::F16 => "F16", Dtype::F32 => "F32", _ => "OTHER" };
                let data = view.data();
                if needs_f32(&name) {
                    // S-B9-REL-PPL: file-derived bytes are not guaranteed aligned (see podread.rs)
                    let mut fv: Vec<f32> = if dt == "F32" {
                        crate::podread::pod_vec::<f32>(data)
                    } else {
                        crate::podread::pod_cow::<half::bf16>(data).iter().map(|x| x.to_f32()).collect()
                    };
                    // hy_v3's RMSNorm is the PLAIN T5-style `weight * x` (HYV3RMSNorm), while the
                    // rmsnorm kernels hard-code qwen3_5's zero-centered `(1 + weight) * x`. Store
                    // hy3's norm weights as (w - 1) at upload so (1 + (w-1)) == w — the kernels and
                    // qwen numerics stay byte-identical. (w-1)+1 in fp32 deviates from w by < 1 ulp
                    // of 1.0, ~100x below the bf16 rounding the kernel applies to its output.
                    if cfg.family == crate::qwen::Family::HyV3
                        && name.contains("norm") && name.ends_with(".weight") {
                        for x in fv.iter_mut() { *x -= 1.0; }
                    }
                    // TP shard-at-load: the GDN f32 helpers are sliced by rank like the in-place
                    // attach does (shard_f32_segs) — a FULL conv1d/A_log/dt_bias with local-channel
                    // indexing reads the wrong rows (the local index into the full weight = the
                    // global channel's row). Norms and expert_bias stay replicated.
                    if let Some(rank) = sal {
                        let gdn_factor = gdn_shard_factor_for(&cfg, world.max(1) as usize, true);
                        let (kdim, vdim, lin_nv) = (cfg.key_dim(), cfg.value_dim(), cfg.lin_num_v_heads);
                        let (segs, stride): (&[(usize, usize)], usize) = if name.starts_with("mtp.") || gdn_factor <= 1 {
                            (&[], 0)   // the MTP head (and non-divisible GDN) is REPLICATED under TP
                        } else if name.ends_with("conv1d.weight") && !name.contains(".ple.") {
                            // B15 FIX: the PLE layer's `ple.conv1d.weight` is [rw, ple_conv_kernel]
                            // and is consumed FULL-width by ple_dconv_prefill_b (the residual is
                            // full rw under TP). Its row count coincides with the GDN qkv row
                            // count (kdim*2+vdim == 4*h for qwen4_exp), so the old ends_with rule
                            // silently col-sharded it to [rw/2, ck] and the kernel read past the
                            // buffer — rank-asymmetric, run-nondeterministic garbage from the PLE
                            // layer on (S-B15 root cause). PLE f32 helpers stay REPLICATED.
                            (&[(0, kdim), (kdim, kdim), (2 * kdim, vdim)], cfg.conv_kernel)
                        } else if name.ends_with("A_log") || name.ends_with("dt_bias") {
                            (&[(0, lin_nv)], 1)
                        } else {
                            (&[], 0)
                        };
                        if !segs.is_empty() {
                            let mut sliced: Vec<f32> = Vec::with_capacity(fv.len() / gdn_factor);
                            for &(off, rows) in segs {
                                assert!(rows % gdn_factor == 0, "gdn f32 shard: segment of {rows} rows won't split across factor {gdn_factor}");
                                let take = rows / gdn_factor;
                                let base = (off + rank * take) * stride;
                                sliced.extend_from_slice(&fv[base..base + take * stride]);
                            }
                            fv = sliced;
                        }
                    }
                    let t = std::time::Instant::now();
                    gpu_f32.insert(name, dev.htod_sync_copy(&fv).unwrap());
                    t_up_shard += t.elapsed();
                } else {
                    // Upload bf16 directly — no Vec copy needed for bf16 source data
                    let owned: Option<Vec<half::bf16>> = if dt == "BF16" || dt == "F16" {
                        None
                    } else {
                        let fv = crate::podread::pod_cow::<f32>(data);
                        Some(fv.iter().map(|&x| half::bf16::from_f32(x)).collect())
                    };
                    let bv_cow: std::borrow::Cow<[half::bf16]>;
                    let bv: &[half::bf16] = match &owned {
                        Some(v) => v,
                        None => { bv_cow = crate::podread::pod_cow::<half::bf16>(data); &bv_cow }
                    };

                    // Apply simulated NVFP4 if this tensor is in scope. Only 2-D weights with K%16==0
                    // (every reduction dim in this family qualifies); shape is [M, K] row-major with
                    // K = in_features = the reduction dim, which is the axis blocks run along.
                    // Per-layer overrides (S5F2 L1a/L1b) win over the group map.
                    let shape = view.shape();
                    let fmt = fq.as_ref().map_or(crate::quant::Fmt::Bf16,
                                                 |m| crate::quant::fmt_for_layer(m, &fq_layers, &name));
                    let in_scope = fmt != crate::quant::Fmt::Bf16;
                    if in_scope && shape.len() == 2 && shape[1] % crate::quant::BLOCK == 0 {
                        let (m, k) = (shape[0], shape[1]);
                        let mut q = bv.to_vec();
                        crate::quant::fake_quant(&mut q, m, k, fmt);
                        let (rel, mx) = crate::quant::roundtrip_error(bv, &q);
                        fq_stats.push((name.clone(), rel, mx));
                        let t = std::time::Instant::now();
                        gpu_bf16.insert(name, dev.htod_sync_copy(&q).unwrap());
                        t_up_shard += t.elapsed();
                    } else {
                        let t = std::time::Instant::now();
                        gpu_bf16.insert(name, dev.htod_sync_copy(bv).unwrap());
                        t_up_shard += t.elapsed();
                    }
                }
            }
            // raw dropped here
            t_shard_loop += t_iter0.elapsed();
            // STREAMING LOAD: drain this shard's ready groups while the next shard is read.
            push_ready_jobs(&recorded, &mut pushed, &host_q4, &host_q8, &gpu_q4_sharded,
                            &job_q, sal, &cfg, world.max(1) as usize, PIPE_CAP_BYTES, &presliced);
            load_prog.add(shard_bytes[i]);
        }
        load_prog.finish();
                    Ok(())
                })();
                if let Err(e) = shard_result {
                    jdone.store(true, std::sync::atomic::Ordering::Relaxed);
                    jq.1.notify_all();
                    return Err(e);
                }
                // all parts are now present: push every remaining group, then release the workers.
                push_ready_jobs(&recorded, &mut pushed, hq4, hq8, &gpu_q4_sharded, jq, sal_c, &cfg, world_c, PIPE_CAP_BYTES, &presliced);
                jdone.store(true, std::sync::atomic::Ordering::Relaxed);
                jq.1.notify_all();
                drop(tx);   // the uploader's recv ends once the workers' senders drop
                let (res, omma_map_up, allow_up, t_up) = up.join().unwrap();
                omma_map.extend(omma_map_up);
                allow_bf16.extend(allow_up);
                results = res;
                t_up_post += t_up;
                for h in handles {
                    let (tf, tr, th) = h.join().unwrap();
                    t_fuse += tf; t_repack += tr; t_hshard += th;
                }
                Ok(())
            })?;
        }

        // 3b) FR-Spec draft head (`--draft-vocab`, default on for Qwen3.5/3.6/3.8; `--draft-vocab-file`). Built HERE, after the
        //     shard loop: the reserved `draft_parts_*` copies are filled only while the shards stream
        //     through (the streaming load of 14c6d7b5 built the head BEFORE the loop, so every load from
        //     v0.4.1 on silently got none), and a bf16 head is still in `gpu_bf16` until the real pass
        //     below takes it. The subset comes from the FULL head on every TP rank (shard-at-load slices
        //     the head's own job, never these copies), so the draft head is replicated: no gather.
        let t_draft0 = std::time::Instant::now();
        let (draft_head, draft_ids) = if draft_rows.is_empty() {
            (None, Vec::new())
        } else {
            let rows = std::mem::take(&mut draft_rows);
            let w = if let Some((qw, sc, inv_gs, _m, k)) = draft_parts_q4.take() {
                let (sq, ss) = crate::quant::subset_rows_nvfp4(&qw, &sc, k, &rows);
                let (wt, st) = crate::quant::repack_nvfp4_mma(&sq, &ss, rows.len(), k);
                let gsv = vec![inv_gs; rows.len() / 16];
                if mxfp4_mode {
                    // The draft head is part of the native chain: it needs the lossless OMMA
                    // repack too (the pipeline does not see it — it is a row subset), and its W
                    // must carry the OMMA layout as PRIMARY storage (the native prefill dequant
                    // reads qweight/scales as OMMA). Small (170 MB), so no economy math.
                    let (aimg, sfa) = crate::mxfp4::repack_nvfp4_omma(&wt, &st, rows.len(), k);
                    let qweight = dev.htod_sync_copy(&aimg).unwrap();
                    let scales = dev.htod_sync_copy(&sfa).unwrap();
                    omma_map.insert(
                        *qweight.device_ptr() as u64,
                        crate::mxfp4::OmmaEntry::Ptr(
                            *qweight.device_ptr() as u64,
                            *scales.device_ptr() as u64,
                        ),
                    );
                    Some(W::Nvfp4 { qweight, scales, gs: dev.htod_sync_copy(&gsv).unwrap(), m: rows.len(), k })
                } else {
                    Some(W::Nvfp4 {
                        qweight: dev.htod_sync_copy(&wt).unwrap(),
                        scales: dev.htod_sync_copy(&st).unwrap(),
                        gs: dev.htod_sync_copy(&gsv).unwrap(),
                        m: rows.len(),
                        k,
                    })
                }
            } else if let Some((qw, rs, _m, k)) = draft_parts_q8.take() {
                let (sq, srs) = crate::quant::subset_rows_fp8(&qw, &rs, k, &rows);
                let wt = crate::quant::repack_fp8_mma(&sq, rows.len(), k);
                Some(W::Fp8 {
                    data: dev.htod_sync_copy(&wt).unwrap(),
                    row_scale: dev.htod_sync_copy(&srs).unwrap(),
                    m: rows.len(),
                    k,
                })
            } else if let Some(full) = gpu_bf16.get(&head_name) {
                // bf16 head (hy_v3 by recipe; a Qwen pack with a bf16 LM head; --dequant-at-load):
                // copy the kept rows on the device, one D2D per contiguous run of ids (the default
                // subset is two runs), on the compute stream. Rows are independent: the subset is exact.
                let k = full.len() / cfg.vocab_size;
                let sub = dev.alloc_zeros::<half::bf16>(rows.len() * k)?;   // every row is written below
                let mut i = 0;
                while i < rows.len() {
                    let mut j = i + 1;
                    while j < rows.len() && rows[j] == rows[j - 1] + 1 { j += 1; }
                    unsafe {
                        let r = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                            *sub.device_ptr() as u64 + (i * k * 2) as u64,
                            *full.device_ptr() as u64 + (rows[i] as usize * k * 2) as u64,
                            (j - i) * k * 2, stream.stream);
                        assert!(r == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                                "draft head bf16 row copy failed: {r:?}");
                    }
                    i = j;
                }
                dev.synchronize()?;
                Some(W::Bf16(sub))
            } else {
                None
            };
            match w {
                Some(w) => {
                    println!(
                        "  draft head: {} of {} tokens ({:.0}% of the vocabulary) -- the draft \
                         chain reads {:.0}% of the LM head's bytes",
                        rows.len(),
                        cfg.vocab_size,
                        100.0 * rows.len() as f32 / cfg.vocab_size as f32,
                        100.0 * rows.len() as f32 / cfg.vocab_size as f32
                    );
                    crate::metrics::set_draft_vocab(rows.len(), cfg.vocab_size);
                    (Some(w), rows)
                }
                None => {
                    crate::rprintln!("--draft-vocab: this model's LM head ({head_name}) is in a format the reduced \
                              draft head cannot subset (block-scaled FP8) — the MTP drafter keeps the full \
                              vocabulary");
                    (None, Vec::new())
                }
            }
        };
        let t_draft = t_draft0.elapsed();
        mem_probe("post-draft");
        // 4) REAL pass: the same assembly, gwn now a lookup over the pipeline results (identical
        //    dispatch to the old serial gwn: hy3 inline bands -> assembled q4/q8 -> bf16 map).
        record_pass.set(false);
        let mut gwn = |names: &[String]| -> W {
            let t_g = std::time::Instant::now();
            // TP=2 shard-at-load (hy_v3): the expert bands were sliced + repacked + uploaded INLINE
            // during the shard-file loop; nothing fuses across them (names.len() == 1 by construction).
            if let Some(w) = gpu_q4_sharded.remove(&names[0]) {
                assert_eq!(names.len(), 1, "shard-at-load expert tensors are never fused");
                t_gwn += t_g.elapsed();
                return w;
            }
            if let Some(w) = results.remove(&akey(names)) {
                t_gwn += t_g.elapsed();
                return w;
            }
            let shard_op = if sal.is_some() {
                Some(if cfg.family == crate::qwen::Family::HyV3 {
                    hy3_load_shard_op(names, &cfg, false, world.max(1) as usize)
                } else {
                    qwen_load_shard_op(names, &cfg, false, world.max(1) as usize)
                })
            } else { None };
            assert!(matches!(shard_op, None | Some(LoadShardOp::None)),
                    "shard-at-load: bf16 tensor sharding is not implemented (the sharded set is all quantized)");
            assert_eq!(names.len(), 1, "bf16 weights are not fused (see GdnIn/AttnIn)");
            let w = W::Bf16(gpu_bf16.remove(&names[0]).unwrap_or_else(|| panic!("missing tensor: {}", names[0])));
            t_gwn += t_g.elapsed();
            w
        };
        let mut gf = |n: &str| -> S { gpu_f32.remove(n).unwrap_or_else(|| panic!("missing f32 tensor: {}", n)) };
        let (embed, final_norm, mut lm_head, mut layers, mtp, hc_mixer) = assemble(&mut gwn, &mut gf);
        // qwen4_exp: attach the PLE n-gram table (device-resident, or the SSD reader).
        if q4 {
            if let Some(pl) = cfg.ple_layer {
                let ple = layers[pl].ple.as_mut().expect("qwen4_exp: PLE layer not assembled");
                Self::attach_ple_table(&dev, ple, std::path::Path::new(model_dir))?;
            }
        }

        // E29-B3 temporaries (set inside the E7 vocab-shard block below; None otherwise).
        let mut dflash_full_lm_head: Option<B> = None;
        let mut dflash_tap_out: Option<crate::dflash::DflashTapSink> = None;

        // S9F (TP-DF2 leg, shard-at-load twin): the worker kept the FULL fused RAW of the
        // lm_head — repack + upload it as the round's full-head borrow (the round's head GEMM
        // needs all v rows; the rank-local shard keeps only v/world). [df2-tp]=1 is set by
        // the head and installed by the node from the shipped config — both ranks keep it (SPMD).
        let mut df2_full_head_lc: Option<W> = None;
        if crate::opts::var(crate::opt!("df2-tp")).is_ok() {
            if let Some((qw, sc, gsv, m, k)) = df2_full_head_raw.lock().unwrap().take() {
                let (wt, st) = crate::quant::repack_nvfp4_mma(&qw, &sc, m, k);
                let qweight = dev.htod_sync_copy(&wt)?;
                let scales = dev.htod_sync_copy(&st)?;
                let gs = dev.htod_sync_copy(&gsv)?;
                crate::deprintln!("[tp] rank {} — DFlash2 full lm_head kept for the round (at-load, {} x {} rows)",
                          sal.unwrap_or(0), m, cfg.hidden_size);
                df2_full_head_lc = Some(W::Nvfp4 { qweight, scales, gs, m, k });
            }
        }

        // S4F DFlash2 tap capture (DEFAULT-OFF). `--df2-capture` rides TpConfig (the generic
        // knob channel, AGENTS §7); the sink allocates 400 KB of zeroed staging and NOTHING
        // ELSE happens at draft time until a round reads it. Flag-off = the field stays None =
        // the capture sites are a dead branch (R1: timing-free when off).
        let df2_capture_out: Option<std::sync::Arc<crate::dflash2::capture::Df2TapSink>> =
            if crate::tp::tp_config().map(|c| c.df2_capture).unwrap_or(false) {
                Some(std::sync::Arc::new(crate::dflash2::capture::Df2TapSink::new(&dev)))
            } else {
                None
            };

        // P14 DFlash v1 tap sink: active whenever the v1 drafter directory is configured
        // (the drafter-dir env knob) — the drafter's conditioning IS the target's hidden states at the
        // artifact's `target_layer_ids`, so the sink cannot be sized before that config is read.
        // Rank 0 only (the drafter is a rank-0 citizen; the captures are rank-local D2Ds).
        // Reading the config here (JSON only, no tensors) is what makes a second artifact a
        // config change: nothing in the trunk hard-codes 5 taps / 4096 / block 8 any more.
        if let Some(ddir) = crate::draft_dir_env() {
            if !ddir.is_empty() && sal.map(|r| r == 0).unwrap_or(true) {
                let dcfg = crate::dflash::DflashCfg::load(std::path::Path::new(&ddir))?;
                eprintln!("[dflash] v1 tap sink (rank-0): {} taps {:?}, block {}, h {}",
                          dcfg.nctx(), dcfg.tap_layers, dcfg.block, cfg.hidden_size);
                dflash_tap_out = Some(crate::dflash::DflashTapSink::new(
                    &dev, cfg.hidden_size, dcfg.tap_layers.clone()));
            }
        }

        // Mirror of attach_tp's rule: a QUANTIZED untied head was vocab-sharded at load.
        let mut lm_head_sharded_lc = sal.is_some()
            && matches!(lm_head, Some(W::Nvfp4 { .. }) | Some(W::Fp8 { .. }));
        // E7: vocab-parallel the BF16 head too — hy3's head is bf16 BY RECIPE (0.99 GB/rank/token
        // replicated ≈ 4.2 ms, the largest replicated read left). Rank r keeps vocab rows
        // [r*v/2, +v/2): a contiguous band of the row-major [v, h] weight, one async D2D on the
        // compute stream; logits_batch{,_f32} gather the halves (zero-fill + sum, bitwise-exact).
        // Peak GPU memory is unchanged (slice after upload, 0.5 GB one-shot — nothing at 90 GB/rank).
        if let (Some(rank), Some(W::Bf16(wb))) = (sal, &lm_head) {
            if !lm_head_sharded_lc && vocab_shard_factor_for(&cfg, world.max(1) as usize, true) > 1 {
                let (v, h) = (cfg.vocab_size, cfg.hidden_size);
                let world = world.max(1) as usize;
                let hv = v / world;
                // E29-B3: capture the FULL bf16 head for the DFlash draft path BEFORE the
                // vocab-parallel shard slices it (rank 0 only, --tp-dflash=1). The drafter's
                // logits need all v rows; the rank-local shard keeps only [rank*v/2, +v/2).
                let dflash_active = rank == 0
                    && (crate::opts::var(crate::opt!("tp-dflash")).is_ok()
                        || crate::tp::tp_config().map(|c| c.dflash).unwrap_or(false));
                // S9F twin for the BF16-head class: the round borrows the FULL head on EVERY
                // rank ([df2-tp] — set by the head, shipped to the node via TpConfig).
                // Without this the borrow sees the E7 rank-local half and the round's
                // full-VOCAB head GEMM reads past the allocation — TP=2 DF2 acceptance 0.0%
                // with correct committed text (verify rejects every draft), 2026-09-04.
                let df2_tp = crate::opts::var(crate::opt!("df2-tp")).is_ok();
                let dflash_full = if dflash_active || df2_tp {
                    let mut full = dev.alloc_zeros::<half::bf16>(v * h)?;
                    unsafe {
                        let r = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                            *full.device_ptr() as u64,
                            *wb.device_ptr() as u64,
                            v * h * 2, stream.stream);
                        assert!(r == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                                "dflash full lm_head D2D failed: {r:?}");
                    }
                    dev.synchronize()?;
                    crate::deprintln!("[tp] rank {rank} — DFlash full-vocab lm_head kept ({v} x {h} bf16, {:.1} MB)",
                              v as f64 * h as f64 * 2.0 / 1e6);
                    Some(full)
                } else { None };
                if df2_tp {
                    df2_full_head_lc = dflash_full.clone().map(W::Bf16);
                }
                let mut shard = dev.alloc_zeros::<half::bf16>(hv * h)?;
                unsafe {
                    let r = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                        *shard.device_ptr() as u64,
                        *wb.device_ptr() as u64 + (rank * hv * h * 2) as u64,
                        hv * h * 2, stream.stream);
                    assert!(r == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                            "lm_head bf16 shard D2D failed: {r:?}");
                }
                // cudarc frees on the NULL stream — order the copy (on `stream`) before wb drops.
                dev.synchronize()?;
                lm_head = Some(W::Bf16(shard));
                lm_head_sharded_lc = true;
                eprintln!("[tp] rank {rank} — bf16 lm_head vocab-sharded ({v} -> {hv} rows)");
                // E29-B3 tap sink: rank 0 only (the node stays a plain SPMD participant; the
                // D2D captures are rank-local and barrier-free, so the TP lockstep is untouched).
                // (P14: the v1 tap sink is created above, from the artifact's config — not here.)
                dflash_full_lm_head = dflash_full;
            }
        }

        // HashMaps now empty (all weights consumed). Drop explicitly.
        drop(gpu_bf16);
        drop(gpu_f32);
        // Every pre-sharded expert band must have been consumed by a MoE layer — a leftover means a
        // config/tensor-name mismatch, and running on with it would be a silent missing weight.
        assert!(gpu_q4_sharded.is_empty(),
                "shard-at-load: {} expert tensors were never consumed by a layer", gpu_q4_sharded.len());
        drop(gpu_q4_sharded);
        dev.synchronize()?;
        let moe_g = new_moe_grouped_scratch(&dev, &cfg, MAX_VERIFY * cfg.num_experts_per_tok);
        let moe_g_pf = new_moe_grouped_scratch(&dev, &cfg, crate::batch::PREFILL_CHUNK * cfg.num_experts_per_tok);
        let verify_params = dev.alloc_zeros::<i32>(4).unwrap();
        let verify_resid = dev.alloc_zeros::<half::bf16>(cfg.resid_width() * MAX_VERIFY).unwrap();
        let verify_graphs = std::sync::Mutex::new(std::collections::HashMap::new());
        let gdn_slot_tables = std::sync::Mutex::new(std::collections::HashMap::new());

        // [load] summary — every phase accounted (Phase-0 instrumentation of the load-speed
        // campaign). `shards cpu` = the shard loop minus its reads and uploads (parse + host
        // copies + f32 converts); `other` = everything unaccounted (device allocs, map churn).
        {
            let total = t_load0.elapsed().as_secs_f64();
            let shard_cpu = t_shard_loop.saturating_sub(t_read + t_up_shard + t_repack_inline).as_secs_f64();
            let gwn_cpu = t_gwn.saturating_sub(t_fuse + t_repack + t_hshard + t_up_post).as_secs_f64();
            let accounted = t_ptx + t_rope + t_shard_loop + t_asm0.elapsed();
            let other = t_load0.elapsed().saturating_sub(accounted).as_secs_f64();
            if crate::boot_report::debug() {
            eprintln!(
                "[load] total {:.1}s | ptx+jit {:.1}s rope {:.1}s | shards: read {:.1}s cpu {:.1}s repack-inline {:.1}s upload {:.1}s | assemble {:.1}s (draft {:.1}s fuse {:.1}s repack {:.1}s host-shard {:.1}s upload {:.1}s gwn-cpu {:.1}s omma-repack {:.1}s) | other {:.1}s",
                total, t_ptx.as_secs_f64(), t_rope.as_secs_f64(),
                t_read.as_secs_f64(), shard_cpu, t_repack_inline.as_secs_f64(), t_up_shard.as_secs_f64(),
                t_asm0.elapsed().as_secs_f64(), t_draft.as_secs_f64(),
                t_fuse.as_secs_f64(), t_repack.as_secs_f64(), t_hshard.as_secs_f64(),
                t_up_post.as_secs_f64(), gwn_cpu, t_omma.lock().unwrap().as_secs_f64(), other);
            }
        }

        mem_probe("post-pipeline");
        let mxfp4 = crate::mxfp4::Mxfp4State::build(mxfp4_mode, omma_map, allow_bf16, mixer4, &dev, &cfg)?;
        let (kv_quant, kv_tq, kv_tq_b3, kv_k8v4, kv_k8v8) = Self::kv_modes_from_opts();
        let tq_tables = Self::build_tq_tables(&dev)?;
        mem_probe("post-mxfp4-build");
        // `input_global_scale.json` (written by --calib-igs) overrides / supplements the tensors.
        let mut igs_by_name = igs_by_name;
        if let Ok(js) = std::fs::read_to_string(std::path::Path::new(model_dir).join("input_global_scale.json")) {
            if let Ok(serde_json::Value::Object(m)) = serde_json::from_str::<serde_json::Value>(&js) {
                let n0 = igs_by_name.len();
                for (k, v) in m { if let Some(f) = v.as_f64() { igs_by_name.insert(k, f as f32); } }
                println!("input_global_scale.json: {} entries (artifact tensors: {n0})", igs_by_name.len());
            }
        }
        // (dev's vmask allocations — the merge dropped these two lines when the PR rearranged the
        // tq_tables..Ok region; restored verbatim from dev's load_from_dir_impl.)
        let vmask_words = dev.htod_sync_copy(&vec![0u32; mask_words_for(cfg.vocab_size) * MAX_VERIFY]).unwrap();
        let vmask_flags = dev.htod_sync_copy(&vec![0i32; MAX_VERIFY]).unwrap();
        Ok((Self { dev, blas, stream, cfg: cfg.clone(), embed, lm_head, final_norm, layers, mtp, k, bk, cos_table, sin_table, sc_pos, sc_rope, sc_slot, sc_winsrc, sc_parent, sc_path, sc_tok, sc_pstart, moe_ids, moe_wts, moe_g, moe_g_pf, sc_i1a, sc_i1b, sv_pf, sv_ki, sv_sd, sv_cand, sv_p, sv_r, mr_tok, mr_pos, verify_params, verify_resid, verify_graphs, gdn_slot_tables, sv_t20: std::sync::OnceLock::new(), sv_t20_scratch: std::sync::OnceLock::new(), vmask_words, vmask_flags, vmask_words_ptr: std::sync::atomic::AtomicU64::new(0), vmask_flags_ptr: std::sync::atomic::AtomicU64::new(0), deq_scratch, fp8e_map, fp8e_scratch, splitk_partials, tp_f32_scratch, mtp_sids, draft_head, draft_ids, lm_head_sharded: lm_head_sharded_lc, gdn_chunk_raw_fn: gdn_chunk_load_raw_fn(), gdn_chunk_tc_raw_fn: gdn_chunk_tc_load_raw_fn(), tp_sharded_at_load: sal.is_some(), kv_quant, kv_tq, kv_tq_b3, kv_k8v4, kv_k8v8, tq_tables, mxfp4, hc_mixer, tp_rank: 0, tp_world: world, tp_ctx_dptr: 0, head_visits: None, dflash_tap: dflash_tap_out, df2_capture: df2_capture_out, df2_capture_tree: None, df2_prime: None, dspark_capture: None, dspark_capture_tree: None, dspark_prime: None, dflash_lm_head: dflash_full_lm_head, df2_full_head: df2_full_head_lc, df2_q_head: std::sync::OnceLock::new(), gptq_tap: std::sync::Mutex::new(None), rotated: Default::default(), rot_small: parking_lot::Mutex::new(None), w4a4: None, igs_by_name, igs_tap: std::sync::Mutex::new(None) }, cfg))
    }
}
