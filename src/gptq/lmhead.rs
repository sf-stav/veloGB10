//! LM-head-only requantization of an existing MR-GPTQ artifact.
//! Siblings: `run.rs` has the trunk driver whose layer installers it reuses; `mod.rs` has the map.

use super::*;

/// Quantize only the dense LM head of an existing MR-GPTQ artifact. This deliberately reuses the
/// already-quantized trunk and calibrates on its real outputs, avoiding a second 64-layer GPTQ run.
pub fn lmhead(source: &Path, base: &Path, out: &Path, calib: &Path, opts: GptqOpts) -> Result<()> {
    validate_opts(&opts)?;
    prepare_output_dir(out, &[source, base])?;
    crate::opts::unset(crate::opt!("w4a4-prefill"));
    crate::opts::unset(crate::opt!("w4a4-lmhead-narrow"));
    if crate::opts::var(crate::opt!("ple-offload")).is_err() { crate::opts::set(crate::opt!("ple-offload"), "ssd"); }
    let t_all = std::time::Instant::now();
    let src = ShardReader::open(source)?;
    let basr = ShardReader::open(base)?;
    let (gpu, cfg) = GpuModel::load_from_dir(&base.to_string_lossy())?;
    anyhow::ensure!(
        matches!(cfg.family, crate::qwen::Family::Qwen35) && !cfg.is_moe,
        "--gptq-lmhead currently requires a qwen3_5 dense artifact, got {:?} (is_moe={})",
        cfg.family,
        cfg.is_moe
    );
    anyhow::ensure!(
        opts.rotate,
        "--gptq-lmhead requires --rotate for an MR-GPTQ head"
    );
    let samples = calib_tokens_mode(
        base,
        calib,
        opts.nsamples,
        opts.seqlen,
        cfg.vocab_size,
        opts.maca,
    )?;
    let ns = samples.len();
    let h = cfg.hidden_size;
    let mut hess = gpu.gptq_hess_new(h);
    let normed = gpu.gptq_dev().alloc_zeros::<bf16>(h * opts.seqlen)?;
    let mut pool = Pool::new(gpu.gptq_dev().clone());
    let mut state = gpu.new_batch_state(1, 1, opts.seqlen);
    println!("[gptq-lmhead] {} samples × {} tokens; H {}x{}, damp {}, clip {}, rotate {}",
             ns, opts.seqlen, h, h, opts.damp, opts.nclip, opts.rotate);
    for (s, toks) in samples.iter().enumerate() {
        gpu.zero_slot_state(&mut state, 0, opts.seqlen);
        let (_, residual) = gpu.prefill_batch_range(
            &mut pool,
            toks,
            &mut state,
            0,
            opts.seqlen,
            0,
            0,
            cfg.num_layers,
            None,
        );
        let weight = if opts.maca {
            1.0 / toks.len() as f32
        } else {
            1.0
        };
        gpu.gptq_lmhead_hess_accum(
            &mut hess,
            &residual,
            &normed,
            toks.len(),
            opts.rotate,
            weight,
        );
        pool.release_bf16(residual, h * toks.len());
        if (s + 1) % 16 == 0 || s + 1 == ns {
            gpu.gptq_sync();
            println!("[gptq-lmhead] Hessian {}/{} samples ({:.1} min)", s + 1, ns, t_all.elapsed().as_secs_f32() / 60.0);
        }
    }
    let total_tokens: usize = samples.iter().map(Vec::len).sum();
    anyhow::ensure!(
        hess.n == total_tokens,
        "lm_head Hessian coverage mismatch: {} vectors, expected {total_tokens}",
        hess.n
    );
    let x_amax = gpu.gptq_amax(&hess);
    let (shape, head) = src.read_bf16("lm_head.weight")?;
    anyhow::ensure!(shape == vec![cfg.vocab_size, h], "lm_head shape {:?}, expected [{}, {}]", shape, cfg.vocab_size, h);
    let head_dev = gpu.gptq_upload_bf16(&head);
    drop(head);
    let mut cs = Cusolver::new(&gpu)?;
    let rec = gptq_2d(&gpu, &mut cs, *head_dev.device_ptr() as u64, cfg.vocab_size, h,
                      &hess.h, &opts, None, x_amax)?;
    let head_igs = rec.igs;
    println!("[gptq-lmhead] quantized lm_head [{}, {}] with {} activation vectors; IGS {:?}",
             cfg.vocab_size, h, hess.n, head_igs);
    drop(cs); drop(head_dev); drop(hess); drop(normed); drop(pool); drop(state); drop(gpu);

    // Rewrite the artifact, replacing the old RTN/bf16 head family and copying every other tensor
    // byte-for-byte. The writer keeps each packed family in one shard.
    let mut writer = Writer::new(out, 4 * 1024 * 1024 * 1024);
    let mut rec = Some(rec);
    let mut inserted = false;
    for (name, meta) in basr.metas.iter() {
        if name.starts_with("lm_head.") {
            if !inserted {
                let r = rec.take().unwrap();
                writer.push_nvfp4("lm_head", r.qw, r.sc, r.m, r.k, r.gs, r.igs);
                inserted = true;
            }
            continue;
        }
        let (_, data) = basr.read_bytes(name)?;
        let dtype = match meta.dtype.as_str() {
            "BF16" => safetensors::Dtype::BF16, "F32" => safetensors::Dtype::F32,
            "I64" => safetensors::Dtype::I64, "F16" => safetensors::Dtype::F16,
            "U8" => safetensors::Dtype::U8, "F8_E4M3" => safetensors::Dtype::F8_E4M3,
            o => return Err(anyhow!("dtype {o} on {name}")),
        };
        writer.push(Out { name: name.clone(), dtype, shape: meta.shape.clone(), data });
    }
    anyhow::ensure!(inserted, "base artifact has no lm_head tensor family");
    writer.finish()?;
    for f in std::fs::read_dir(base)? {
        let f = f?; let n = f.file_name().to_string_lossy().to_string();
        if n.ends_with(".safetensors") || n == "model.safetensors.index.json" { continue; }
        let dst = out.join(&n);
        let big = n.starts_with("ple_ngram") && n.ends_with(".bin");
        if !(big && std::fs::hard_link(f.path(), &dst).is_ok()) { std::fs::copy(f.path(), &dst)?; }
    }
    let cp = out.join("config.json");
    let mut cj: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&cp)?)?;
    let add = |v: &mut serde_json::Value, g: &str| {
        let a = v.as_array_mut().expect("quantization group list");
        if !a.iter().any(|x| x.as_str() == Some(g)) { a.push(serde_json::json!(g)); }
    };
    let remove = |v: &mut serde_json::Value, g: &str| {
        if let Some(a) = v.as_array_mut() { a.retain(|x| x.as_str() != Some(g)); }
    };
    add(&mut cj["quantization_config"]["gptq"]["groups"], "lmhead");
    remove(&mut cj["quantization_config"]["gptq"]["rtn_groups"], "lmhead");
    cj["quantization_config"]["gptq"]["nsamples"] = serde_json::json!(ns);
    cj["quantization_config"]["gptq"]["seqlen"] = serde_json::json!(opts.seqlen);
    cj["quantization_config"]["gptq"]["damp"] = serde_json::json!(opts.damp);
    cj["quantization_config"]["gptq"]["clip_ratios"] = serde_json::json!(opts.nclip);
    cj["quantization_config"]["gptq"]["global_scale_optimization"] = serde_json::json!({ "type": "alternating_least_squares", "iterations": opts.scale_iters });
    cj["quantization_config"]["gptq"]["local_scale_optimization"] = serde_json::json!(if opts.local_hessian { "local_hessian_fp8_sweep" } else { "clip_mse" });
    cj["quantization_config"]["gptq"]["activation_order"] = serde_json::json!(if opts.static_act_order { "static" } else { "none" });
    add(&mut cj["quantization_config"]["transform"]["groups"], "lmhead");
    cj["quantization_config"]["activation_variant"] = serde_json::json!("runtime-selectable-a16-or-a4");
    std::fs::write(&cp, serde_json::to_string_pretty(&cj)?)?;
    if let Some(igs) = head_igs {
        let p = out.join("input_global_scale.json");
        let mut j: serde_json::Value = if p.exists() { serde_json::from_str(&std::fs::read_to_string(&p)?)? } else { serde_json::json!({}) };
        j.as_object_mut().ok_or_else(|| anyhow!("{} is not a JSON object", p.display()))?
            .insert("lm_head".to_string(), serde_json::json!(igs));
        std::fs::write(&p, serde_json::to_string_pretty(&j)?)?;
    }
    println!("[gptq-lmhead] done in {:.1} min → {}", t_all.elapsed().as_secs_f32() / 60.0, out.display());
    Ok(())
}

