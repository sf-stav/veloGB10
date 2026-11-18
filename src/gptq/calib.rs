//! Calibration data for GPTQ: token sets, the Hessian / profile passes (plain and sequential) and vision
//! calibration samples. Siblings: `run.rs` consumes these; `mod.rs` has the directory map.

use super::*;

// ---------------------------------------------------------------- calibration data
pub fn calib_tokens(
    model_dir: &Path,
    calib: &Path,
    nsamples: usize,
    seqlen: usize,
    vocab: usize,
) -> Result<Vec<Vec<u32>>> {
    calib_tokens_mode(model_dir, calib, nsamples, seqlen, vocab, false)
}

fn calib_tokens_mode(
    model_dir: &Path,
    calib: &Path,
    nsamples: usize,
    seqlen: usize,
    vocab: usize,
    variable_lengths: bool,
) -> Result<Vec<Vec<u32>>> {
    if calib.to_string_lossy() == "random" {
        // Smoke-test / synthetic-model mode: seeded random token ids.
        let mut st = 0x9E3779B97F4A7C15u64;
        let mut next = || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; st };
        return Ok((0..nsamples).map(|_| (0..seqlen).map(|_| (next() % (vocab as u64).max(4)) as u32 + 4).map(|t| t.min(vocab as u32 - 1)).collect()).collect());
    }
    let raw = std::fs::read_to_string(calib).with_context(|| format!("read {}", calib.display()))?;
    // `calib_compose` emits sample-aligned JSONL with the exact token IDs audited in its manifest.
    // Consume those IDs directly: re-rendering/re-tokenizing would destroy both category boundaries
    // and the exact token ratios. A pre-tokenized file is deliberately all-or-nothing.
    if calib.extension().map_or(false, |e| e == "jsonl") {
        let mut samples = Vec::new();
        let mut saw_pretokenized = false;
        for (line_no, line) in raw.lines().filter(|l| !l.trim().is_empty()).enumerate() {
            let v: serde_json::Value = serde_json::from_str(line)
                .with_context(|| format!("{}:{}: invalid JSON", calib.display(), line_no + 1))?;
            let Some(ids) = v.get("input_ids") else {
                anyhow::ensure!(!saw_pretokenized,
                    "{}:{}: mixed corpus: every row must contain input_ids", calib.display(), line_no + 1);
                break;
            };
            saw_pretokenized = true;
            let ids = ids.as_array().ok_or_else(|| {
                anyhow!(
                    "{}:{}: input_ids is not an array",
                    calib.display(),
                    line_no + 1
                )
            })?;
            if variable_lengths {
                anyhow::ensure!(
                    !ids.is_empty() && ids.len() <= seqlen,
                    "{}:{}: input_ids has {} tokens, expected 1..={seqlen} for MaCa",
                    calib.display(),
                    line_no + 1,
                    ids.len()
                );
            } else {
                anyhow::ensure!(
                    ids.len() == seqlen,
                    "{}:{}: input_ids has {} tokens, expected seqlen={seqlen}",
                    calib.display(),
                    line_no + 1,
                    ids.len()
                );
            }
            let mut sample = Vec::with_capacity(ids.len());
            for (column, id) in ids.iter().enumerate() {
                let id = id.as_u64().ok_or_else(|| anyhow!("{}:{}: input_ids[{column}] is not an unsigned integer", calib.display(), line_no + 1))?;
                anyhow::ensure!(id < vocab as u64,
                    "{}:{}: input_ids[{column}]={id} is outside vocab_size={vocab}", calib.display(), line_no + 1);
                sample.push(id as u32);
            }
            samples.push(sample);
            if samples.len() == nsamples { break; }
        }
        if saw_pretokenized {
            anyhow::ensure!(
                samples.len() == nsamples,
                "pre-tokenized calibration corpus contains {} complete samples, asked {nsamples}",
                samples.len()
            );
            println!(
                "[gptq] loaded {nsamples} pre-tokenized, sample-aligned calibration rows{}",
                if variable_lengths {
                    " (variable length)"
                } else {
                    ""
                }
            );
            return Ok(samples);
        }
    }
    let tok = crate::tokenizer::QwenTokenizer::from_file(&model_dir.join("tokenizer.json").to_string_lossy())?;
    // jsonl with a "text" field, or plain text (blank-line separated documents)
    // jsonl lines: {"text": …} raw documents, or {"messages": [{role, content}, …]} rendered
    // through the model's own chat template (calibration in the served format).
    let docs: Vec<String> = if calib.extension().map_or(false, |e| e == "jsonl") {
        raw.lines().filter(|l| !l.trim().is_empty()).filter_map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).ok()?;
            if let Some(t) = v["text"].as_str() { return Some(t.to_string()); }
            let msgs: Vec<crate::tokenizer::ChatMessage> = v["messages"].as_array()?.iter().map(|m| crate::tokenizer::ChatMessage {
                role: m["role"].as_str().unwrap_or("user").to_string(), content: m["content"].as_str().map(|s| s.to_string()),
                tool_calls: None, tool_call_id: None, name: None, reasoning_content: None, images: vec![], layout: vec![], unsupported_parts: vec![] }).collect();
            tok.apply_chat_template_no_gen(&msgs, None, None).ok()
        }).filter(|s| !s.is_empty()).collect()
    } else { raw.split("\n\n").map(|s| s.to_string()).filter(|s| !s.trim().is_empty()).collect() };
    let mut samples = Vec::new();
    let mut buf: Vec<u32> = Vec::new();
    for d in docs {
        let ids = tok.encode(&d, false)?;
        buf.extend(ids);
        buf.push(tok.encode("\n\n", false)?.first().copied().unwrap_or(198));
        while buf.len() >= seqlen && samples.len() < nsamples {
            samples.push(buf.drain(..seqlen).collect());
        }
        if samples.len() >= nsamples { break; }
    }
    anyhow::ensure!(samples.len() >= nsamples.min(1), "calibration text too short for one sample of {seqlen} tokens");
    if samples.len() < nsamples { println!("[gptq] calibration text gave {} samples (asked {nsamples})", samples.len()); }
    Ok(samples)
}

/// Profile a candidate corpus once, producing compact per-layer activation features for
/// COLA/ACDM plus exact per-layer expert route counts for MoE balancing. The output is keyed by
/// candidate row index so selection can copy the original JSONL records byte-for-byte.
pub fn profile_calibration(
    model_dir: &Path,
    calib: &Path,
    out: &Path,
    nsamples: usize,
    max_seqlen: usize,
    layers: &[usize],
    sketch_dim: usize,
) -> Result<()> {
    anyhow::ensure!(!out.exists(), "refusing to overwrite {}", out.display());
    anyhow::ensure!(
        nsamples > 0 && max_seqlen > 0,
        "profile sample count and max sequence length must be positive"
    );
    anyhow::ensure!(
        (1..=256).contains(&sketch_dim),
        "--profile-sketch-dim must be 1..256"
    );
    if crate::opts::var(crate::opt!("ple-offload")).is_err() {
        crate::opts::set(crate::opt!("ple-offload"), "ssd");
    }
    anyhow::ensure!(
        crate::opts::var_os(crate::opt!("w4a4-prefill")).is_none(),
        "--calib-profile requires unquantized activations; unset --w4a4-prefill"
    );
    let t0 = std::time::Instant::now();
    let (gpu, cfg) = GpuModel::load_from_dir(&model_dir.to_string_lossy())?;
    let profile_layers: Vec<usize> = if layers.is_empty() {
        let points = cfg.num_layers.min(8).max(1);
        let mut auto: Vec<usize> = (0..points)
            .map(|index| index * (cfg.num_layers - 1) / points.saturating_sub(1).max(1))
            .collect();
        auto.sort_unstable();
        auto.dedup();
        auto
    } else {
        layers.to_vec()
    };
    anyhow::ensure!(
        profile_layers.iter().all(|&layer| layer < cfg.num_layers),
        "profile layers {profile_layers:?} exceed model depth {}",
        cfg.num_layers
    );
    let samples = calib_tokens_mode(model_dir, calib, nsamples, max_seqlen, cfg.vocab_size, true)?;
    let mut pool = Pool::new(gpu.gptq_dev().clone());
    let mut state = gpu.new_batch_state(1, 1, max_seqlen);
    let incomplete = PathBuf::from(format!("{}.incomplete", out.display()));
    let mut writer = std::io::BufWriter::new(std::fs::File::create(&incomplete)?);
    for (index, tokens) in samples.iter().enumerate() {
        let tap = GptqTap {
            profile_enabled: true,
            profile_layers: profile_layers.clone(),
            profile_sketch_dim: sketch_dim,
            sample_weight: 1.0,
            ..Default::default()
        };
        gpu.gptq_arm(tap);
        gpu.zero_slot_state(&mut state, 0, max_seqlen);
        let (_, residual) = gpu.prefill_batch_range(
            &mut pool,
            tokens,
            &mut state,
            0,
            max_seqlen,
            0,
            0,
            cfg.num_layers,
            None,
        );
        drop(residual);
        let tap = gpu
            .gptq_disarm()
            .context("calibration profile tap disappeared")?;
        let experts: Vec<serde_json::Value> = tap
            .profile_expert_counts
            .into_iter()
            .map(|(layer, counts)| serde_json::json!({ "layer": layer, "counts": counts }))
            .collect();
        serde_json::to_writer(
            &mut writer,
            &serde_json::json!({
                "format": "veloGB10-calibration-profile-v1",
                "sample_index": index,
                "sequence_length": tokens.len(),
                "activations": tap.profile_activations,
                "experts": experts,
            }),
        )?;
        writer.write_all(b"\n")?;
        if (index + 1) % 16 == 0 || index + 1 == samples.len() {
            gpu.gptq_sync();
            println!(
                "[calib-profile] {}/{} candidates ({:.1} min)",
                index + 1,
                samples.len(),
                t0.elapsed().as_secs_f32() / 60.0
            );
        }
    }
    writer.flush()?;
    drop(writer);
    std::fs::rename(&incomplete, out)?;
    let total_tokens: usize = samples.iter().map(Vec::len).sum();
    let manifest = serde_json::json!({
        "format": "veloGB10-calibration-profile-manifest-v1",
        "model_dir": std::fs::canonicalize(model_dir)?,
        "candidate_corpus": if calib.to_string_lossy() == "random" {
            serde_json::Value::String("random".to_string())
        } else {
            serde_json::to_value(std::fs::canonicalize(calib)?)?
        },
        "output": out,
        "samples": samples.len(),
        "total_tokens": total_tokens,
        "max_seqlen": max_seqlen,
        "activation_layers": profile_layers,
        "sketch": { "type": "signed_count_sketch_of_mean_pooled_channels", "dimension": sketch_dim },
        "expert_counts": if cfg.is_moe { "all_moe_layers_topk_routes" } else { "dense_model" },
    });
    std::fs::write(
        format!("{}.manifest.json", out.display()),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    println!(
        "[calib-profile] wrote {} profiles / {total_tokens} tokens -> {}",
        samples.len(),
        out.display()
    );
    Ok(())
}

/// Profile a BF16 source checkpoint one layer at a time while using a quantized artifact only as
/// the memory-resident model skeleton. This is the profiling analogue of sequential GPTQ: all base
/// layer weights are dropped, source BF16 layers are installed one at a time, and each candidate's
/// hidden stream is retained between layers. The base still supplies the PLE table, matching the
/// actual GPTQ path for qwen4_exp without ever making the full source checkpoint resident.
pub fn profile_calibration_sequential(
    source: &Path,
    base: &Path,
    calib: &Path,
    out: &Path,
    nsamples: usize,
    max_seqlen: usize,
    layers: &[usize],
    sketch_dim: usize,
) -> Result<()> {
    anyhow::ensure!(!out.exists(), "refusing to overwrite {}", out.display());
    anyhow::ensure!(
        nsamples > 0 && max_seqlen > 0,
        "profile sample count and max sequence length must be positive"
    );
    anyhow::ensure!(
        (1..=256).contains(&sketch_dim),
        "--profile-sketch-dim must be 1..256"
    );
    if crate::opts::var(crate::opt!("ple-offload")).is_err() {
        crate::opts::set(crate::opt!("ple-offload"), "ssd");
    }
    anyhow::ensure!(
        crate::opts::var_os(crate::opt!("w4a4-prefill")).is_none(),
        "--calib-profile requires unquantized activations; unset --w4a4-prefill"
    );

    let t0 = std::time::Instant::now();
    let src = ShardReader::open(source)?;
    let (mut gpu, cfg) = GpuModel::load_from_dir(&base.to_string_lossy())?;
    gpu.gptq_reset_rotation();
    anyhow::ensure!(
        matches!(cfg.family, crate::qwen::Family::Qwen35 | crate::qwen::Family::Qwen4Exp),
        "sequential calibration profiling is not implemented for {:?}",
        cfg.family
    );
    let profile_layers: Vec<usize> = if layers.is_empty() {
        let points = cfg.num_layers.min(8).max(1);
        let mut auto: Vec<usize> = (0..points)
            .map(|index| index * (cfg.num_layers - 1) / points.saturating_sub(1).max(1))
            .collect();
        auto.sort_unstable();
        auto.dedup();
        auto
    } else {
        layers.to_vec()
    };
    anyhow::ensure!(
        profile_layers.iter().all(|&layer| layer < cfg.num_layers),
        "profile layers {profile_layers:?} exceed model depth {}",
        cfg.num_layers
    );
    let samples = calib_tokens_mode(
        base,
        calib,
        nsamples,
        max_seqlen,
        cfg.vocab_size,
        true,
    )?;
    let total_tokens: usize = samples.iter().map(Vec::len).sum();

    for li in 0..gpu.gptq_num_layers() {
        gpu.gptq_drop_layer_weights(li);
    }
    gpu.gptq_sync();

    // Layer 0 must begin from the source embedding rather than the base's RTN embedding. The
    // lm_head/final mixer are never reached by this profiling pass and can be released.
    let embed_name = "model.language_model.embed_tokens.weight";
    let (_, embed_host) = src.read_bf16(embed_name)?;
    let embed = gpu.gptq_w_bf16(&embed_host);
    drop(embed_host);
    gpu.gptq_install_nonlayer(Some(embed), None, None);

    let ns = samples.len();
    let mut hidden: Vec<Option<B>> = (0..ns).map(|_| None).collect();
    let mut activations: Vec<Vec<CalibLayerProfile>> =
        (0..ns).map(|_| Vec::with_capacity(profile_layers.len())).collect();
    let mut experts: Vec<std::collections::BTreeMap<usize, Vec<u64>>> =
        (0..ns).map(|_| std::collections::BTreeMap::new()).collect();
    let mut pool = Pool::new(gpu.gptq_dev().clone());
    let mut state = gpu.new_batch_state(1, 1, max_seqlen);

    println!(
        "[calib-profile-sequential] {} candidates / {total_tokens} tokens; BF16 source layers, base skeleton {}; activation layers {:?}",
        ns,
        base.display(),
        profile_layers
    );
    for li in 0..cfg.num_layers {
        let layer_t0 = std::time::Instant::now();
        install_source_bf16_layer(&mut gpu, &cfg, &src, li)?;
        let capture_activation = profile_layers.contains(&li);
        for s in 0..ns {
            let tap = GptqTap {
                profile_enabled: true,
                profile_layers: if capture_activation { vec![li] } else { Vec::new() },
                profile_sketch_dim: sketch_dim,
                sample_weight: 1.0,
                ..Default::default()
            };
            gpu.gptq_arm(tap);
            gpu.zero_slot_state(&mut state, 0, max_seqlen);
            let inc = hidden[s].take();
            let (_, outb) = gpu.prefill_batch_range(
                &mut pool,
                &samples[s],
                &mut state,
                0,
                max_seqlen,
                0,
                li,
                li + 1,
                inc,
            );
            hidden[s] = Some(outb);
            let tap = gpu
                .gptq_disarm()
                .context("sequential calibration profile tap disappeared")?;
            activations[s].extend(tap.profile_activations);
            experts[s].extend(tap.profile_expert_counts);
            if (s + 1) % 128 == 0 || s + 1 == ns {
                gpu.gptq_sync();
                println!(
                    "[calib-profile-sequential] layer {li}/{}: {}/{} candidates ({:.1} min)",
                    cfg.num_layers - 1,
                    s + 1,
                    ns,
                    t0.elapsed().as_secs_f32() / 60.0
                );
            }
        }
        gpu.gptq_drop_layer_weights(li);
        gpu.gptq_sync();
        println!(
            "[calib-profile-sequential] layer {li}/{} complete in {:.1}s; MemAvailable {:.1} GB",
            cfg.num_layers - 1,
            layer_t0.elapsed().as_secs_f32(),
            mem_available_gb()
        );
    }

    drop(hidden);
    drop(state);
    drop(pool);
    drop(gpu);

    let incomplete = PathBuf::from(format!("{}.incomplete", out.display()));
    let mut writer = std::io::BufWriter::new(std::fs::File::create(&incomplete)?);
    for index in 0..ns {
        let expert_rows: Vec<serde_json::Value> = std::mem::take(&mut experts[index])
            .into_iter()
            .map(|(layer, counts)| serde_json::json!({ "layer": layer, "counts": counts }))
            .collect();
        serde_json::to_writer(
            &mut writer,
            &serde_json::json!({
                "format": "veloGB10-calibration-profile-v1",
                "sample_index": index,
                "sequence_length": samples[index].len(),
                "activations": std::mem::take(&mut activations[index]),
                "experts": expert_rows,
            }),
        )?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    drop(writer);
    std::fs::rename(&incomplete, out)?;
    let manifest = serde_json::json!({
        "format": "veloGB10-calibration-profile-manifest-v1",
        "mode": "sequential_bf16_layers",
        "model_dir": std::fs::canonicalize(source)?,
        "base_model": std::fs::canonicalize(base)?,
        "candidate_corpus": if calib.to_string_lossy() == "random" {
            serde_json::Value::String("random".to_string())
        } else {
            serde_json::to_value(std::fs::canonicalize(calib)?)?
        },
        "output": out,
        "samples": ns,
        "total_tokens": total_tokens,
        "max_seqlen": max_seqlen,
        "activation_layers": profile_layers,
        "sketch": { "type": "signed_count_sketch_of_mean_pooled_channels", "dimension": sketch_dim },
        "expert_counts": if cfg.is_moe { "all_moe_layers_topk_routes" } else { "dense_model" },
        "base_supplied": ["model_skeleton", "ple_table"],
    });
    std::fs::write(
        format!("{}.manifest.json", out.display()),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    println!(
        "[calib-profile-sequential] wrote {ns} profiles / {total_tokens} tokens in {:.1} min -> {}",
        t0.elapsed().as_secs_f32() / 60.0,
        out.display()
    );
    Ok(())
}

struct VisionCalibSample {
    tokens: Vec<u32>,
    image_embeds: Vec<f32>,
    spans: Vec<crate::vision_encoder::ImageSpan>,
}

fn local_image_data_url(url: &str) -> Result<String> {
    if url.starts_with("data:") { return Ok(url.to_string()); }
    let path = Path::new(url);
    anyhow::ensure!(path.is_file(), "vision calibration requires a local file or data URL, got {url}");
    let mime = match path.extension().and_then(|s| s.to_str()).unwrap_or("").to_ascii_lowercase().as_str() {
        "png" => "image/png", "jpg" | "jpeg" => "image/jpeg", "webp" => "image/webp",
        other => return Err(anyhow!("unsupported calibration image extension {other:?}: {}", path.display())),
    };
    let data = base64::engine::general_purpose::STANDARD.encode(std::fs::read(path)?);
    Ok(format!("data:{mime};base64,{data}"))
}

/// Return `Some` only for a raw multimodal JSONL. The visual tower and preprocessing are the same
/// ones as the serving request path; this is intentionally used by `calib_igs`, not the 2048-token
/// GPTQ Hessian run.
fn vision_calib_samples(model_dir: &Path, calib: &Path, nsamples: usize, seqlen: usize) -> Result<Option<Vec<VisionCalibSample>>> {
    if calib.extension().map_or(true, |e| e != "jsonl") { return Ok(None); }
    let raw = std::fs::read_to_string(calib).with_context(|| format!("read {}", calib.display()))?;
    let first = match raw.lines().find(|line| !line.trim().is_empty()) { Some(line) => line, None => return Ok(None) };
    let first_value: serde_json::Value = serde_json::from_str(first)?;
    if first_value.get("input_ids").is_some() { return Ok(None); }
    let first_messages: Vec<crate::tokenizer::ChatMessage> = match first_value.get("messages") {
        Some(messages) => serde_json::from_value(messages.clone())?, None => return Ok(None),
    };
    if !first_messages.iter().any(|message| !message.images.is_empty()) { return Ok(None); }

    let tok = crate::tokenizer::QwenTokenizer::from_file(&model_dir.join("tokenizer.json").to_string_lossy())?;
    let tower = crate::vision_tower::VisualTower::load(&model_dir.to_string_lossy())
        .with_context(|| format!("load visual tower from {}", model_dir.display()))?;
    let mut samples = Vec::new();
    for (line_no, line) in raw.lines().filter(|line| !line.trim().is_empty()).enumerate() {
        if samples.len() == nsamples { break; }
        let value: serde_json::Value = serde_json::from_str(line)
            .with_context(|| format!("{}:{}: invalid JSON", calib.display(), line_no + 1))?;
        let messages: Vec<crate::tokenizer::ChatMessage> = serde_json::from_value(value["messages"].clone())
            .with_context(|| format!("{}:{}: invalid multimodal messages", calib.display(), line_no + 1))?;
        let mut urls = Vec::new();
        for image in messages.iter().flat_map(|message| &message.images) {
            urls.push(local_image_data_url(image.url.as_deref().ok_or_else(|| anyhow!("{}:{}: image URL is absent", calib.display(), line_no + 1))?)?);
        }
        anyhow::ensure!(!urls.is_empty(), "{}:{}: mixed text/vision corpus", calib.display(), line_no + 1);
        let rendered = tok.apply_chat_template_no_gen(&messages, None, None)?;
        let prompt_tokens = tok.encode(&rendered, true)?;
        let mut prepared = crate::vision_encoder::prepare_vision_request(&tower, &urls, &prompt_tokens)?;
        let vision_end = prepared.spans.iter().map(|span| span.start + span.num_tokens).max().unwrap_or(0);
        anyhow::ensure!(vision_end <= seqlen,
            "{}:{}: image span ends at token {vision_end}, beyond seqlen={seqlen}", calib.display(), line_no + 1);
        anyhow::ensure!(prepared.expanded_tokens.len() >= seqlen,
            "{}:{}: multimodal prompt has only {} expanded tokens, expected at least {seqlen}",
            calib.display(), line_no + 1, prepared.expanded_tokens.len());
        prepared.expanded_tokens.truncate(seqlen);
        samples.push(VisionCalibSample { tokens: prepared.expanded_tokens, image_embeds: prepared.image_embeds, spans: prepared.spans });
    }
    anyhow::ensure!(samples.len() == nsamples,
        "vision calibration corpus contains {} usable samples, asked {nsamples}", samples.len());
    Ok(Some(samples))
}

fn prepare_output_dir(out: &Path, inputs: &[&Path]) -> Result<()> {
    let input_paths: Vec<PathBuf> = inputs.iter().map(|p| {
        let resolved = std::fs::canonicalize(p).with_context(|| format!("resolve input {}", p.display()))?;
        Ok(resolved)
    }).collect::<Result<_>>()?;
    if out.exists() {
        let out_path = std::fs::canonicalize(out).with_context(|| format!("resolve output {}", out.display()))?;
        anyhow::ensure!(!input_paths.iter().any(|p| p == &out_path),
            "output directory {} must differ from every input directory", out.display());
        anyhow::ensure!(std::fs::read_dir(out)?.next().is_none(),
            "output directory {} already exists and is not empty", out.display());
    } else {
        std::fs::create_dir_all(out)?;
    }
    Ok(())
}

