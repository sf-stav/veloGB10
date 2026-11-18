//! Sequential MR-GPTQ for the 5-layer DFlash2 drafter (`--gptq-dflash2`).
//! Siblings: `run.rs` is the trunk driver this mirrors; `mod.rs` has the map.

use super::*;

/// Sequential MR-GPTQ for the 5-layer DFlash2 drafter. Target prompt taps are captured with
/// the production W4A4 prefill enabled, projected once, and cached beside the output. Each draft
/// layer is then calibrated on the already-quantized prefix of the drafter stack.
pub fn dflash2(source: &Path, target: &Path, out: &Path, calib: &Path,
               opts: GptqOpts, context_vectors: usize) -> Result<()> {
    use crate::dflash2::{BLOCK as DB, head_dim, hidden, inter, n_layers, num_heads, num_kv_heads};
    validate_opts(&opts)?;
    anyhow::ensure!(
        !opts.maca,
        "--gptq-dflash2 does not yet support --maca variable-length captures"
    );
    use crate::dflash2::capture::Df2PrimeSink;
    use crate::dflash2::round::Df2Round;

    anyhow::ensure!(opts.rotate, "--gptq-dflash2 requires --rotate (MR-GPTQ artifact)");
    anyhow::ensure!((1..=8192).contains(&opts.seqlen), "DFlash2 calibration seqlen must be 1..8192");
    anyhow::ensure!(context_vectors > 0 && context_vectors <= opts.seqlen,
                    "--df2-context-vectors must be 1..seqlen");
    prepare_output_dir(out, &[source, target])?;
    let cache = PathBuf::from(format!("{}.calib-cache", out.display()));
    std::fs::create_dir_all(&cache)?;
    let t_all = std::time::Instant::now();
    let manifest = serde_json::json!({
        "source": std::fs::canonicalize(source)?, "target": std::fs::canonicalize(target)?,
        "calib": std::fs::canonicalize(calib)?, "calib_bytes": std::fs::metadata(calib)?.len(),
        "nsamples": opts.nsamples, "seqlen": opts.seqlen, "damp": opts.damp,
        "clip": opts.nclip, "scale_iters": opts.scale_iters,
        "static_act_order": opts.static_act_order, "local_hessian": opts.local_hessian,
        "context_vectors": context_vectors,
    });
    let mp = cache.join("manifest.json");
    if mp.exists() {
        let old: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&mp)?)?;
        anyhow::ensure!(old == manifest, "{} belongs to a different DFlash2 calibration recipe", cache.display());
    } else {
        std::fs::write(&mp, serde_json::to_string_pretty(&manifest)?)?;
    }

    // This is the exact production target prefill requested by the recipe. DFlash2's own
    // projections remain W4A16 at N=8; their IGS is still recorded for future native A4 work.
    crate::opts::set(crate::opt!("w4a4-prefill"), "attn,mlp,gdn");
    if crate::opts::var(crate::opt!("ple-offload")).is_err() { crate::opts::set(crate::opt!("ple-offload"), "ssd"); }
    let src = ShardReader::open(source)?;
    let (mut gpu, cfg) = GpuModel::load_from_dir(&target.to_string_lossy())?;
    anyhow::ensure!(gpu.df2_round_compatible(), "target model is not dimension-compatible with DFlash2");
    let samples = calib_tokens(target, calib, opts.nsamples, opts.seqlen, cfg.vocab_size)?;
    let (head, embed) = gpu.df2_borrow().ok_or_else(|| anyhow!("target lm_head/embed cannot be borrowed"))?;
    let mut round = Df2Round::load(&source.to_string_lossy(), Some(head), Some(embed), opts.seqlen + DB)?;
    round.set_head_hadamard16(gpu.df2_head_hadamard16());
    let prime = std::sync::Arc::new(Df2PrimeSink::new(gpu.gptq_dev(), opts.seqlen));
    gpu.set_df2_prime_sink(prime.clone());
    let mut pool = Pool::new(gpu.gptq_dev().clone());
    let mut state = gpu.new_batch_state(1, 1, opts.seqlen);
    println!("[gptq-df2] cache: {} samples × {} W4A4-prefill tokens → {}",
             samples.len(), opts.seqlen, cache.display());
    for (si, toks) in samples.iter().enumerate() {
        let sample_path = cache.join(format!("{si:06}.bin"));
        if std::fs::metadata(&sample_path).map(|m| m.len() as usize).ok()
            == Some(4 + hidden() * opts.seqlen * 2) { continue; }
        gpu.zero_slot_state(&mut state, 0, opts.seqlen);
        let (anchor, residual) = gpu.prefill_batch_range(&mut pool, toks, &mut state, 0,
            opts.seqlen, 0, 0, cfg.num_layers, None);
        gpu.gptq_sync();
        let th = round.project_taps_host(&prime.taps, opts.seqlen)?;
        let mut raw = Vec::with_capacity(4 + th.len() * 2);
        raw.extend_from_slice(&anchor.to_le_bytes());
        raw.extend_from_slice(bytemuck::cast_slice(&th));
        let tmp = sample_path.with_extension("tmp");
        std::fs::write(&tmp, raw)?;
        std::fs::rename(tmp, sample_path)?;
        pool.release_bf16(residual, hidden() * opts.seqlen);
        if (si + 1) % 8 == 0 || si + 1 == samples.len() {
            println!("[gptq-df2] tap cache {}/{} ({:.1} min)", si + 1, samples.len(), t_all.elapsed().as_secs_f32()/60.0);
        }
    }
    gpu.set_df2_prime_off();
    drop(state); drop(pool); drop(prime);

    let mut cs = Cusolver::new(&gpu)?;
    let mut records: HashMap<String, Rec> = HashMap::new();
    let specs = [
        ("self_attn.q_proj", num_heads() * head_dim(), hidden(), 0usize),
        ("self_attn.k_proj", num_kv_heads() * head_dim(), hidden(), 1usize),
        ("self_attn.v_proj", num_kv_heads() * head_dim(), hidden(), 1usize),
        ("self_attn.o_proj", hidden(), num_heads() * head_dim(), 2usize),
        ("mlp.gate_proj", inter(), hidden(), 3usize),
        ("mlp.up_proj", inter(), hidden(), 3usize),
        ("mlp.down_proj", hidden(), inter(), 4usize),
    ];
    for li in 0..n_layers() {
        let checkpoint = cache.join(format!("layer-{li}.records"));
        if checkpoint.exists() {
            let saved = read_df2_checkpoint(&checkpoint)?;
            anyhow::ensure!(saved.len() == specs.len(), "{} has {} records, expected {}", checkpoint.display(), saved.len(), specs.len());
            for (name, r) in saved {
                let prefix = format!("layers.{li}.");
                let suffix = name.strip_prefix(&prefix).and_then(|s| s.strip_suffix(".weight"))
                    .ok_or_else(|| anyhow!("bad DFlash2 checkpoint tensor {name}"))?;
                round.install_calibrated_projection(li, suffix, &r.qw, &r.sc, r.gs, r.m, r.k, true)?;
                records.insert(name, r);
            }
            println!("[gptq-df2] layer {li}: restored checkpoint {}", checkpoint.display());
            continue;
        }
        let tl = std::time::Instant::now();
        let mut hq = gpu.gptq_hess_new(hidden());
        let mut hkv = gpu.gptq_hess_new(hidden());
        let mut ho = gpu.gptq_hess_new(num_heads() * head_dim());
        let mut hgu = gpu.gptq_hess_new(hidden());
        let mut hdn = gpu.gptq_hess_new(inter());
        for si in 0..samples.len() {
            let raw = std::fs::read(cache.join(format!("{si:06}.bin")))?;
            anyhow::ensure!(raw.len() == 4 + hidden() * opts.seqlen * 2, "corrupt DFlash2 cache sample {si}");
            let anchor = u32::from_le_bytes(raw[..4].try_into().unwrap());
            let th: &[bf16] = bytemuck::cast_slice(&raw[4..]);
            round.reset();
            round.prime_projected(th, opts.seqlen, 0)?;
            let x = round.capture_layer_inputs(anchor, li)?;
            let qd = gpu.gptq_upload_bf16(&x.qkv);
            let od = gpu.gptq_upload_bf16(&x.o);
            let gud = gpu.gptq_upload_bf16(&x.gate_up);
            let dnd = gpu.gptq_upload_bf16(&x.down);
            gpu.gptq_hess_accum_external(&mut hq, &qd, DB, true);
            gpu.gptq_hess_accum_external(&mut hkv, &qd, DB, true);
            gpu.gptq_hess_accum_external(&mut ho, &od, DB, true);
            gpu.gptq_hess_accum_external(&mut hgu, &gud, DB, true);
            gpu.gptq_hess_accum_external(&mut hdn, &dnd, DB, true);
            // Evenly sample prompt context for k/v. Full 2048-token Hessians are redundant and
            // would dominate runtime; 16/sample gives 8192 context vectors at nsamples=512.
            let mut sub = Vec::with_capacity(hidden() * context_vectors);
            for j in 0..context_vectors {
                let c = ((j + 1) * opts.seqlen / (context_vectors + 1)).min(opts.seqlen - 1);
                sub.extend_from_slice(&th[c * hidden()..(c + 1) * hidden()]);
            }
            let td = gpu.gptq_upload_bf16(&sub);
            gpu.gptq_hess_accum_external(&mut hkv, &td, context_vectors, true);
            if (si + 1) % 16 == 0 || si + 1 == samples.len() { gpu.gptq_sync(); }
        }
        println!("[gptq-df2] layer {li}: Hessians q={} kv={} o={} gu={} down={} vectors",
                 hq.n, hkv.n, ho.n, hgu.n, hdn.n);
        for &(suffix, m, k, hk) in &specs {
            let name = format!("layers.{li}.{suffix}.weight");
            let (shape, host) = src.read_bf16(&name)?;
            anyhow::ensure!(shape == vec![m, k], "{name}: shape {shape:?}, expected [{m},{k}]");
            let wd = gpu.gptq_upload_bf16(&host);
            let hess = match hk { 0 => &hq, 1 => &hkv, 2 => &ho, 3 => &hgu, 4 => &hdn, _ => unreachable!() };
            let r = gptq_2d(&gpu, &mut cs, *wd.device_ptr() as u64, m, k, &hess.h,
                            &opts, None, gpu.gptq_amax(hess))?;
            round.install_calibrated_projection(li, suffix, &r.qw, &r.sc, r.gs, m, k, true)?;
            records.insert(name, r);
        }
        gpu.gptq_sync();
        let saved: Vec<(String, &Rec)> = specs.iter().map(|(suffix, _, _, _)| {
            let name = format!("layers.{li}.{suffix}.weight");
            let r = records.get(&name).expect("fresh DFlash2 record");
            (name, r)
        }).collect();
        write_df2_checkpoint(&checkpoint, &saved)?;
        println!("[gptq-df2] layer {li}/{} quantized sequentially in {:.1} min",
                 n_layers() - 1, tl.elapsed().as_secs_f32()/60.0);
    }

    drop(round); drop(cs); drop(gpu);
    let mut writer = Writer::new(out, 4 * 1024 * 1024 * 1024);
    for (name, meta) in &src.metas {
        let stem = name.strip_suffix(".weight").unwrap_or(name);
        if let Some(r) = records.remove(name) {
            writer.push_nvfp4(stem, r.qw, r.sc, r.m, r.k, r.gs, r.igs);
            continue;
        }
        let (_, data) = src.read_bytes(name)?;
        let dtype = match meta.dtype.as_str() {
            "BF16" => safetensors::Dtype::BF16, "F32" => safetensors::Dtype::F32,
            "I64" => safetensors::Dtype::I64, "U8" => safetensors::Dtype::U8,
            o => return Err(anyhow!("dtype {o} on {name}")),
        };
        writer.push(Out { name: name.clone(), dtype, shape: meta.shape.clone(), data });
    }
    anyhow::ensure!(records.is_empty(), "unwritten DFlash2 GPTQ records: {}", records.len());
    writer.finish()?;
    for f in std::fs::read_dir(source)? {
        let f = f?; let n = f.file_name().to_string_lossy().to_string();
        if n.ends_with(".safetensors") || n == "model.safetensors.index.json" { continue; }
        if !f.file_type()?.is_file() && !f.file_type()?.is_symlink() { continue; }
        std::fs::copy(f.path(), out.join(&n))?;
    }
    let cp = out.join("config.json");
    let mut cj: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&cp)?)?;
    cj["quantization_config"] = serde_json::json!({
        "quant_method": "gptq", "format": "nvfp4-pack-quantized",
        "activation_variant": "w4a16", "transform": { "type": "hadamard16", "groups": ["projections"] },
        "gptq": { "nsamples": samples.len(), "seqlen": opts.seqlen, "damp": opts.damp,
          "clip_ratios": opts.nclip, "activation_order": if opts.static_act_order { "static" } else { "none" },
          "global_scale_optimization": { "type": "alternating_least_squares", "iterations": opts.scale_iters },
          "local_scale_optimization": if opts.local_hessian { "local_hessian_fp8_sweep" } else { "clip_mse" },
          "context_vectors_per_sample": context_vectors, "sequential": true },
        "calibration_target": { "model": target.display().to_string(),
          "prefill": "--w4a4-prefill=attn,mlp,gdn" },
        "quantized_groups": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "bf16_groups": ["fc", "conv", "norm", "selector"]
    });
    std::fs::write(&cp, serde_json::to_string_pretty(&cj)?)?;
    if let Err(e) = std::fs::remove_dir_all(&cache) {
        eprintln!("[gptq-df2] warning: could not remove cache {}: {e}", cache.display());
    }
    println!("[gptq-df2] done in {:.1} min → {}", t_all.elapsed().as_secs_f32()/60.0, out.display());
    Ok(())
}

