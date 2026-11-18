//! The `--gptq` driver (`run`), the output-directory check and the per-layer weight installers.
//! Siblings: `calib.rs` (inputs), `dflash2.rs` / `lmhead.rs` (variants of this driver); `mod.rs` has the map.

use super::*;

// ---------------------------------------------------------------- the driver
pub fn run(source: &Path, base: &Path, out: &Path, calib: &Path, opts: GptqOpts) -> Result<()> {
    validate_opts(&opts)?;
    prepare_output_dir(out, &[source, base])?;
    if crate::opts::var(crate::opt!("ple-offload")).is_err() { crate::opts::set(crate::opt!("ple-offload"), "ssd"); }
    let t_all = std::time::Instant::now();
    let src = ShardReader::open(source)?;
    let basr = ShardReader::open(base)?;
    let (mut gpu, cfg) = GpuModel::load_from_dir(&base.to_string_lossy())?;
    gpu.gptq_reset_rotation();
    anyhow::ensure!(
        matches!(
            cfg.family,
            crate::qwen::Family::Qwen35 | crate::qwen::Family::Qwen4Exp
        ),
        "--gptq is implemented for qwen3_5 dense and qwen4_exp, got {:?}",
        cfg.family
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
    let total_tokens: usize = samples.iter().map(Vec::len).sum();
    let mut length_hist = std::collections::BTreeMap::<usize, usize>::new();
    for sample in &samples {
        *length_hist.entry(sample.len()).or_default() += 1;
    }
    println!("[gptq] {} samples / {} tokens / lengths {:?}; groups GPTQ {:?}, RTN {:?}, rotate {}, damp {}, clip ratios {}, scale iters {}, act-order {}, MaCa {}",
             ns, total_tokens, length_hist, opts.gptq_groups.iter().map(|g| quant::group_name(*g)).collect::<Vec<_>>(),
             opts.nvfp4_groups.iter().map(|g| quant::group_name(*g)).collect::<Vec<_>>(), opts.rotate, opts.damp, opts.nclip,
             opts.scale_iters, if opts.static_act_order { "static" } else { "none" }, opts.maca);
    println!(
        "[gptq] local-Hessian FP8 scale sweep: {}",
        opts.local_hessian
    );
    for li in 0..gpu.gptq_num_layers() {
        gpu.gptq_drop_layer_weights(li);
    }
    gpu.gptq_sync();
    println!("[gptq] base layer weights dropped (rebuilt per layer from the source); MemAvailable {:.1} GB", mem_available_gb());
    let mut pool = Pool::new(gpu.gptq_dev().clone());
    let seqlen = opts.seqlen;
    let mut state = gpu.new_batch_state(1, 1, seqlen);
    let mut cs = Cusolver::new(&gpu)?;
    let mut writer = Writer::new(out, 4 * 1024 * 1024 * 1024);
    let (h, ne, mi) = (cfg.hidden_size, cfg.num_experts, cfg.moe_intermediate_size);
    let is_gptq = |name: &str| opts.gptq_groups.contains(&quant::group_of(name));
    let is_rtn = |name: &str| opts.nvfp4_groups.contains(&quant::group_of(name));
    let is_fp8 = |name: &str| opts.fp8_groups.contains(&quant::group_of(name));
    let lm = "model.language_model";
    // The non-layer tensors the artifact takes from the SOURCE (embed, lm_head, final mixer) are
    // served during the calibration exactly as they will be served — source bf16, or the
    // quantizer's own RTN / FP8 when their group asks for it — not as the base's RTN copies.
    {
        let mk = |name: &str| -> Result<Option<W>> {
            let Some(meta) = src.metas.get(name) else { return Ok(None) };
            if meta.dtype != "BF16" || meta.shape.len() != 2 { return Ok(None); }
            let (shape, v) = src.read_bf16(name)?;
            let (m, k) = (shape[0], shape[1]);
            let q = m % 16 == 0 && k % 16 == 0;
            Ok(Some(if q && is_rtn(name) { let r = rtn_2d(&v, m, k); gpu.gptq_w_nvfp4(&r.qw, &r.sc, m, k, r.gs) }
                    else if q && is_fp8(name) { gpu.gptq_w_fp8(quant::quantize_fp8(&v, m, k)) }
                    else { gpu.gptq_w_bf16(&v) }))
        };
        let embed = mk(&format!("{lm}.embed_tokens.weight"))?;
        let head = mk("lm_head.weight")?;
        let mixer = match (mk(&format!("{lm}.hyper_connection_mixer.input_mix_weight_down.weight"))?,
                           mk(&format!("{lm}.hyper_connection_mixer.input_mix_weight_up.weight"))?) {
            (Some(a), Some(b)) => Some((a, b)), _ => None };
        let n = embed.is_some() as usize + head.is_some() as usize + mixer.is_some() as usize * 2;
        gpu.gptq_install_nonlayer(embed, head, mixer);
        println!("[gptq] {n} non-layer tensors (embed / lm_head / final mixer) reinstalled from the source as the artifact will serve them");
    }
    // Per-sample residual streams between layers (None before layer 0: the prefill embeds).
    let mut hidden: Vec<Option<B>> = (0..ns).map(|_| None).collect();
    let nl = cfg.num_layers;
    for li in 0..nl {
        let t_l = std::time::Instant::now();
        let lp = format!("{lm}.layers.{li}");
        // 1. the layer's linears in bf16, swapped in; taps registered for the GPTQ groups
        let mut tap = GptqTap::default();
        // name -> (the quantizer's bf16 copy, m, k, tap key = device pointer of the LAYER's copy)
        let mut bf: HashMap<String, (B, usize, usize, u64)> = HashMap::new();
        let up = |gpu: &GpuModel, name: &str, tap: &mut GptqTap, bf: &mut HashMap<String, (B, usize, usize, u64)>| -> Result<W> {
            let (shape, v) = src.read_bf16(name)?;
            let (m, k) = if shape.len() == 3 { (shape[0] * shape[1], shape[2]) } else { (shape[0], shape[1]) };
            let b = gpu.gptq_upload_bf16(&v);
            // The LAYER owns `b`: `gemm_act` taps by the device pointer it is called with, so the Hessian
            // must be keyed by the pointer the layer sees (keying the quantizer's copy left every dense
            // Hessian empty -> RTN). The quantizer keeps its own copy.
            let key = *b.device_ptr() as u64;
            if is_gptq(name) && !name.contains(".mlp.experts.") { tap.by_ptr.insert(key, gpu.gptq_hess_new(k)); }
            let mine = gpu.gptq_clone_b(&b);
            bf.insert(name.to_string(), (mine, m, k, key));
            Ok(W::Bf16(b))
        };
        let is_attn = matches!(cfg.layer_types[li], crate::qwen::LayerType::FullAttention);
        let mut names: Vec<String> = Vec::new();
        {
            let layer = gpu.gptq_layer(li);
            if is_attn {
                for t in ["q_proj", "k_proj", "v_proj", "o_proj"] { names.push(format!("{lp}.self_attn.{t}.weight")); }
                if layer.fa.as_ref().unwrap().indexer.is_some() { names.push(format!("{lp}.self_attn.indexer.index_qk_proj.weight")); }
            } else {
                for t in ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"] { names.push(format!("{lp}.linear_attn.{t}.weight")); }
            }
            match &layer.mlp {
                Ffn::Moe(_) => {
                    names.push(format!("{lp}.mlp.experts.gate_up_proj"));
                    names.push(format!("{lp}.mlp.experts.down_proj"));
                    for t in ["gate_proj", "up_proj", "down_proj"] { names.push(format!("{lp}.mlp.shared_expert.{t}.weight")); }
                    names.push(format!("{lp}.mlp.gate.weight"));
                }
                Ffn::Dense(_) => {
                    for t in ["gate_proj", "up_proj", "down_proj"] { names.push(format!("{lp}.mlp.{t}.weight")); }
                }
            }
            if layer.hc.is_some() {
                for hcn in ["attn_hyper_connection", "mlp_hyper_connection"] {
                    for t in ["input_mix_weight_down", "input_mix_weight_up"] { names.push(format!("{lp}.{hcn}.{t}.weight")); }
                }
            }
            if layer.ple.is_some() { for t in ["key_proj", "value_proj"] { names.push(format!("{lp}.ple.{t}.weight")); } }
        }
        let mut ws: HashMap<String, W> = HashMap::new();
        for n in &names { let w = up(&gpu, n, &mut tap, &mut bf)?; ws.insert(n.clone(), w); }
        let gptq_experts = cfg.is_moe && is_gptq(&format!("{lp}.mlp.experts.gate_up_proj"));
        if gptq_experts {
            tap.moe_gu = (0..ne).map(|_| gpu.gptq_hess_new(h)).collect();
            tap.moe_dn = (0..ne).map(|_| gpu.gptq_hess_new(mi)).collect();
            tap.moe_all = Some(gpu.gptq_hess_new(h));
            tap.moe_all_cap = MOE_SUB_TOKENS;
            tap.moe_all_x = Some(gpu.gptq_dev().alloc_zeros::<half::bf16>(h * MOE_SUB_TOKENS)?);
        }
        install_layer(&mut gpu, li, is_attn, &lp, &mut ws);
        // 2. pass 1: Hessians (the layer output is discarded: the bf16 experts kernel is skipped)
        tap.skip_experts = true;
        gpu.gptq_arm(tap);
        for s in 0..ns {
            gpu.gptq_set_sample_weight(if opts.maca {
                1.0 / samples[s].len() as f32
            } else {
                1.0
            });
            gpu.zero_slot_state(&mut state, 0, seqlen);
            let inc = hidden[s].as_ref().map(|b| gpu.gptq_clone_b(b));
            let (_, outb) = gpu.prefill_batch_range(&mut pool, &samples[s], &mut state, 0, seqlen, 0, li, li + 1, inc);
            drop(outb);
        }
        let tap = gpu.gptq_disarm().unwrap();
        let t_fwd = t_l.elapsed().as_secs_f32();
        if !tap.moe_gu.is_empty() {
            // Calibration coverage of the routed experts: how many tokens each expert's Hessian saw.
            let mut ns: Vec<usize> = tap.moe_gu.iter().map(|h| h.n).collect(); ns.sort_unstable();
            let under = ns.iter().filter(|&&n| n < 256).count();
            println!("[gptq] layer {li} expert coverage: tokens/expert min {} median {} max {}; {} of {} experts under 256 tokens",
                     ns[0], ns[ns.len() / 2], ns[ns.len() - 1], under, ns.len());
        }
        // 3. quantize
        let mut recs: HashMap<String, Rec> = HashMap::new();
        for n in &names {
            let (b, m, k, key) = bf.get(n).unwrap();
            if n.ends_with("experts.gate_up_proj") || n.ends_with("experts.down_proj") {
                if !gptq_experts { if is_rtn(n) { let (_, v) = src.read_bf16(n)?; recs.insert(n.clone(), rtn_2d(&v, *m, *k)); } continue; }
                let is_gu = n.ends_with("gate_up_proj");
                let (me, ke) = if is_gu { (2 * mi, h) } else { (h, mi) };
                let hs = if is_gu { &tap.moe_gu } else { &tap.moe_dn };
                let base_ptr = *b.device_ptr() as u64;
                // Under-calibrated experts (fewer than 2·K routed tokens: a rank-deficient Hessian)
                // fall back to the layer-wide statistics: the all-token Hessian for gate_up, and
                // for down a Hessian built by running the expert on the token subsample.
                let thr = 2 * ke;
                let gu_b: Option<&B> = bf.get(&format!("{lp}.mlp.experts.gate_up_proj")).map(|(b, _, _, _)| b);
                let mut n_fallback = 0usize;
                // one global scale per stacked tensor (the artifact's convention): amax over all (rotated) experts
                let mut amax = 0f32;
                for e in 0..ne { let w32 = gpu.gptq_w32(base_ptr + (e * me * ke * 2) as u64, me, ke, opts.rotate); amax = amax.max(gpu.gptq_absmax_f32(&w32, me * ke)); }
                let initial_st = e4m3_scale_of(amax);
                let st = optimize_bf16_scale(&gpu, base_ptr, ne * me * ke, initial_st, &opts);
                let mut qw = Vec::with_capacity(ne * me * ke / 2); let mut sc = Vec::with_capacity(ne * me * ke / 16);
                // one input global scale per stacked tensor: the activation amax over all experts
                let x_amax = (0..ne).map(|e| gpu.gptq_amax(&hs[e])).fold(0f32, f32::max);
                for e in 0..ne {
                    let fallback: Option<GptqHess> = if hs[e].n < thr {
                        n_fallback += 1;
                        if is_gu { None } else {
                            let xs = tap.moe_all_x.as_ref().unwrap();
                            Some(gpu.gptq_down_hess_from(
                                gu_b.unwrap(),
                                e,
                                xs,
                                tap.moe_all_n,
                                h,
                                mi,
                                &tap.moe_all_segments,
                            ))
                        }
                    } else { None };
                    let hess: &S = if hs[e].n < thr && is_gu { &tap.moe_all.as_ref().unwrap().h } else { fallback.as_ref().map(|f| &f.h).unwrap_or(&hs[e].h) };
                    let r = gptq_2d(&gpu, &mut cs, base_ptr + (e * me * ke * 2) as u64, me, ke, hess, &opts, Some(st), x_amax)?;
                    qw.extend_from_slice(&r.qw); sc.extend_from_slice(&r.sc);
                }
                if n_fallback > 0 { println!("[gptq] layer {li} {}: {n_fallback} experts under {thr} tokens used the all-token fallback", if is_gu { "gate_up" } else { "down" }); }
                recs.insert(n.clone(), Rec { qw, sc, m: ne * me, k: ke, gs: 1.0 / st, igs: igs_of(x_amax) });
            } else if is_gptq(n) {
                let mut acc = tap.by_ptr.get(key).ok_or_else(|| anyhow!("no Hessian for {n} (the calibration never reached this GEMM)"))?;
                if *k % 16 != 0 || *m % 16 != 0 { continue; }   // e.g. in_proj_b/a [nh, h] — kept bf16 by the quantizer too
                if acc.n == 0 && n.ends_with(".self_attn.indexer.index_qk_proj.weight") {
                    // QSA is off below `qsa_limit` visible tokens, so the indexer never runs at calibration
                    // seqlen; its input is the same normalized hidden as q/k/v -> reuse q_proj's Hessian.
                    if let Some((_, _, _, qkey)) = bf.get(&format!("{lp}.self_attn.q_proj.weight")) {
                        if let Some(a) = tap.by_ptr.get(qkey) { if a.k == acc.k && a.n > 0 { println!("[gptq] {n}: indexer never ran (QSA off at this seqlen) — using q_proj's Hessian ({} tokens)", a.n); acc = a; } }
                    }
                }
                anyhow::ensure!(acc.n > 0, "{n}: empty Hessian — no calibration token reached this GEMM (tap pointer mismatch?)");
                if acc.n < 2 * *k { println!("[gptq] warning: {n} Hessian over only {} tokens (K = {k})", acc.n); }
                let hess = &acc.h;
                let x_amax = gpu.gptq_amax(acc);
                recs.insert(n.clone(), gptq_2d(&gpu, &mut cs, *b.device_ptr() as u64, *m, *k, hess, &opts, None, x_amax)?);
            } else if is_rtn(n) && *k % 16 == 0 && *m % 16 == 0 {
                let (_, v) = src.read_bf16(n)?; recs.insert(n.clone(), rtn_2d(&v, *m, *k));
            }
        }
        drop(tap);
        let t_q = t_l.elapsed().as_secs_f32() - t_fwd;
        // 4. swap the quantized weights in (sequential GPTQ: the next layer calibrates on them)
        let mut ws2: HashMap<String, W> = HashMap::new();
        for n in &names {
            let w = match recs.get(n) {
                // stacked experts use the same MMA-repacked layout as the loader's (W::Nvfp4, gs per 16-row tile)
                Some(r) => gpu.gptq_w_nvfp4(&r.qw, &r.sc, r.m, r.k, r.gs),
                None if is_fp8(n) => {
                    let (_, m, k, _) = bf.remove(n).unwrap();
                    if m % 16 == 0 && k % 16 == 0 { let (_, v) = src.read_bf16(n)?; gpu.gptq_w_fp8(quant::quantize_fp8(&v, m, k)) }
                    else { let (_, v) = src.read_bf16(n)?; gpu.gptq_w_bf16(&v) }
                }
                None => { let (b, _, _, _) = bf.remove(n).unwrap(); W::Bf16(b) }
            };
            // only GPTQ'd tensors were quantized in the rotated basis (RTN groups are not, and are not in
            // `transform.groups`) — marking them would rotate their input at calibration but not at serving
            if opts.rotate && recs.contains_key(n) && is_gptq(n) { gpu.gptq_mark_rotated(&w); }
            ws2.insert(n.clone(), w);
        }
        drop(bf);
        install_layer(&mut gpu, li, is_attn, &lp, &mut ws2);
        // 5. pass 2: the quantized layer's outputs become the next layer's inputs
        for s in 0..ns {
            gpu.zero_slot_state(&mut state, 0, seqlen);
            let inc = hidden[s].take();
            let (_, outb) = gpu.prefill_batch_range(&mut pool, &samples[s], &mut state, 0, seqlen, 0, li, li + 1, inc);
            hidden[s] = Some(outb);
        }
        // 5b. the quantized layer is never run again (the next layer calibrates on `hidden`, the LM
        //     head on the final residual streams): release its device weights now instead of keeping
        //     every quantized layer resident (~1.4 GB/layer, 68 GB by layer 47 — the 1M-token
        //     calibration hit the memory watchdog at layer 43). The rotation markers are keyed by
        //     device pointer, so clear them with the buffers they described.
        gpu.gptq_drop_layer_weights(li);
        gpu.gptq_reset_rotation();
        gpu.gptq_sync();
        // 6. stream the layer's tensors out: GPTQ/RTN records, everything else verbatim from the source
        let mut n_q = 0;
        for (name, meta) in src.metas.range(format!("{lp}.")..).take_while(|(k, _)| k.starts_with(&format!("{lp}."))) {
            if name.contains(".ngram_embedding.shard_") { continue; }   // the PLE table comes from the base artifact
            let stem = name.strip_suffix(".weight").unwrap_or(name).to_string();
            if let Some(r) = recs.remove(name) { writer.push_nvfp4(&stem, r.qw, r.sc, r.m, r.k, r.gs, r.igs); n_q += 1; continue; }
            if is_fp8(name) && meta.dtype == "BF16" && meta.shape.len() == 2 && meta.shape[0] % 16 == 0 && meta.shape[1] % 16 == 0 {
                let (shape, v) = src.read_bf16(name)?; writer.push_fp8(&stem, quant::quantize_fp8(&v, shape[0], shape[1])); n_q += 1; continue;
            }
            let (_, data) = src.read_bytes(name)?;
            let dtype = match meta.dtype.as_str() { "BF16" => safetensors::Dtype::BF16, "F32" => safetensors::Dtype::F32, "I64" => safetensors::Dtype::I64, "F16" => safetensors::Dtype::F16, "U8" => safetensors::Dtype::U8, o => return Err(anyhow!("dtype {o} on {name}")) };
            writer.push(Out { name: name.clone(), dtype, shape: meta.shape.clone(), data });
        }
        println!("[gptq] layer {li}/{nl} ({}): forward {:.1}s, quantize {:.1}s, {n_q} tensors quantized, total {:.1}s",
                 if is_attn { "attn" } else { "gdn" }, t_fwd, t_q, t_l.elapsed().as_secs_f32());
    }
    // 6b. the LM head (group `lmhead` in --gptq-groups): its input is the final mixer / norm of the
    //     residual streams the last layer left in `hidden` — Hessian over every calibration token.
    let mut lm_rec: Option<Rec> = None;
    if is_gptq("lm_head.weight") {
        if let Some((ptr, rows)) = gpu.gptq_lm_head_bf16() {
            let t0 = std::time::Instant::now();
            let hess = gpu.gptq_hess_new(h);
            for s in 0..ns {
                let len = samples[s].len();
                let weight = if opts.maca { 1.0 / len as f32 } else { 1.0 };
                gpu.gptq_lm_head_hess_accum(
                    &mut pool,
                    hidden[s].as_ref().unwrap(),
                    len,
                    &hess,
                    weight,
                );
            }
            let x_amax = gpu.gptq_amax(&hess);
            lm_rec = Some(gptq_2d(
                &gpu, &mut cs, ptr, rows, h, &hess.h, &opts, None, x_amax,
            )?);
            println!("[gptq] lm_head [{rows}, {h}] GPTQ over {total_tokens} tokens (activation amax {x_amax:.3}) in {:.1}s", t0.elapsed().as_secs_f32());
        } else {
            println!("[gptq] warning: lm_head is not served in bf16 (tied / missing) — left as the source has it");
        }
    }
    drop(hidden);
    // 7. non-layer tensors: source verbatim (embed, final norm, mixer, vision, lm_head) with the
    //    RTN groups quantized; the MTP head and the PLE table straight from the base artifact.
    for (name, meta) in src.metas.iter() {
        if name.starts_with(&format!("{lm}.layers.")) || name.starts_with("mtp.") { continue; }
        let stem = name.strip_suffix(".weight").unwrap_or(name).to_string();
        let quantizable = meta.dtype == "BF16" && meta.shape.len() == 2 && meta.shape[1] % 16 == 0 && meta.shape[0] % 16 == 0 && !name.contains(".visual.");
        if name == "lm_head.weight" { if let Some(r) = lm_rec.take() { writer.push_nvfp4(&stem, r.qw, r.sc, r.m, r.k, r.gs, r.igs); continue; } }
        if quantizable && is_rtn(name) {
            let (shape, v) = src.read_bf16(name)?; let r = rtn_2d(&v, shape[0], shape[1]);
            writer.push_nvfp4(&stem, r.qw, r.sc, r.m, r.k, r.gs, r.igs); continue;
        }
        if quantizable && is_fp8(name) {
            let (shape, v) = src.read_bf16(name)?; writer.push_fp8(&stem, quant::quantize_fp8(&v, shape[0], shape[1])); continue;
        }
        let (_, data) = src.read_bytes(name)?;
        let dtype = match meta.dtype.as_str() { "BF16" => safetensors::Dtype::BF16, "F32" => safetensors::Dtype::F32, "I64" => safetensors::Dtype::I64, "F16" => safetensors::Dtype::F16, "U8" => safetensors::Dtype::U8, o => return Err(anyhow!("dtype {o} on {name}")) };
        writer.push(Out { name: name.clone(), dtype, shape: meta.shape.clone(), data });
    }
    for (name, meta) in basr.metas.iter() {
        if !name.starts_with("mtp.") { continue; }
        let (_, data) = basr.read_bytes(name)?;
        let dtype = match meta.dtype.as_str() { "BF16" => safetensors::Dtype::BF16, "F32" => safetensors::Dtype::F32, "U8" => safetensors::Dtype::U8, "F8_E4M3" => safetensors::Dtype::F8_E4M3, "I64" => safetensors::Dtype::I64, o => return Err(anyhow!("dtype {o} on {name}")) };
        writer.push(Out { name: name.clone(), dtype, shape: meta.shape.clone(), data });
    }
    writer.finish()?;
    for f in ["tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja", "merges.txt", "vocab.json", "preprocessor_config.json"] {
        let s = base.join(f); if s.exists() { std::fs::copy(&s, out.join(f))?; }
    }
    let ple_side = base.join("ple_ngram_nvfp4.json");
    if ple_side.exists() {
        let side: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&ple_side)?)?;
        let ple_file = side["file"].as_str().unwrap_or("ple_ngram_nvfp4.bin").to_string();
        std::fs::copy(&ple_side, out.join("ple_ngram_nvfp4.json"))?;
        if std::fs::hard_link(base.join(&ple_file), out.join(&ple_file)).is_err() { std::fs::copy(base.join(&ple_file), out.join(&ple_file))?; }
    }
    let mut cj: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(base.join("config.json"))?)?;
    cj["quantization_config"]["gptq"] = serde_json::json!({ "nsamples": ns, "seqlen": seqlen,
        "total_tokens": total_tokens, "length_histogram": length_hist, "maca": opts.maca,
        "hessian_sequence_normalization": if opts.maca { "1/sequence_length" } else { "none" },
        "damp": opts.damp, "clip_ratios": opts.nclip,
        "global_scale_optimization": { "type": "alternating_least_squares", "iterations": opts.scale_iters },
        "local_scale_optimization": if opts.local_hessian { "local_hessian_fp8_sweep" } else { "clip_mse" },
        "activation_order": if opts.static_act_order { "static" } else { "none" },
        "groups": opts.gptq_groups.iter().map(|g| quant::group_name(*g)).collect::<Vec<_>>(),
        "rtn_groups": opts.nvfp4_groups.iter().map(|g| quant::group_name(*g)).collect::<Vec<_>>(),
        "fp8_groups": opts.fp8_groups.iter().map(|g| quant::group_name(*g)).collect::<Vec<_>>() });
    if opts.rotate {
        cj["quantization_config"]["transform"] = serde_json::json!({ "type": "hadamard16", "groups": opts.gptq_groups.iter().map(|g| quant::group_name(*g)).collect::<Vec<_>>() });
    } else if let Some(qc) = cj["quantization_config"].as_object_mut() {
        qc.remove("transform");
    }
    std::fs::write(out.join("config.json"), serde_json::to_string_pretty(&cj)?)?;
    println!("[gptq] done in {:.1} min → {}", t_all.elapsed().as_secs_f32() / 60.0, out.display());
    Ok(())
}

/// Put the layer's linears (`ws`, by source name) into the engine's layer structs.
fn install_layer(gpu: &mut GpuModel, li: usize, is_attn: bool, lp: &str, ws: &mut HashMap<String, W>) {
    let mut take = |n: String| ws.remove(&n).unwrap_or_else(|| panic!("install_layer: missing {n}"));
    let layer = gpu.gptq_layer_mut(li);
    if is_attn {
        let fa = layer.fa.as_mut().unwrap();
        fa.qkv = AttnIn::Split { q: take(format!("{lp}.self_attn.q_proj.weight")), k: take(format!("{lp}.self_attn.k_proj.weight")), v: take(format!("{lp}.self_attn.v_proj.weight")) };
        fa.o_proj = take(format!("{lp}.self_attn.o_proj.weight"));
        if let Some(ix) = fa.indexer.as_mut() { ix.qk_proj = take(format!("{lp}.self_attn.indexer.index_qk_proj.weight")); }
    } else {
        let la = layer.la.as_mut().unwrap();
        la.in_proj = GdnIn::Split { qkv: take(format!("{lp}.linear_attn.in_proj_qkv.weight")), z: take(format!("{lp}.linear_attn.in_proj_z.weight")),
                                    b: take(format!("{lp}.linear_attn.in_proj_b.weight")), a: take(format!("{lp}.linear_attn.in_proj_a.weight")) };
        la.out_proj = take(format!("{lp}.linear_attn.out_proj.weight"));
    }
    match &mut layer.mlp {
        Ffn::Moe(moe) => {
            moe.gate_up = take(format!("{lp}.mlp.experts.gate_up_proj"));
            moe.down = take(format!("{lp}.mlp.experts.down_proj"));
            moe.shared.gate = take(format!("{lp}.mlp.shared_expert.gate_proj.weight"));
            moe.shared.up = take(format!("{lp}.mlp.shared_expert.up_proj.weight"));
            moe.shared.down = take(format!("{lp}.mlp.shared_expert.down_proj.weight"));
            moe.router = take(format!("{lp}.mlp.gate.weight"));
        }
        Ffn::Dense(mlp) => {
            mlp.gate = take(format!("{lp}.mlp.gate_proj.weight"));
            mlp.up = take(format!("{lp}.mlp.up_proj.weight"));
            mlp.down = take(format!("{lp}.mlp.down_proj.weight"));
        }
    }
    if let Some(hc) = layer.hc.as_mut() {
        hc.0.down = take(format!("{lp}.attn_hyper_connection.input_mix_weight_down.weight"));
        hc.0.up = take(format!("{lp}.attn_hyper_connection.input_mix_weight_up.weight"));
        hc.1.down = take(format!("{lp}.mlp_hyper_connection.input_mix_weight_down.weight"));
        hc.1.up = take(format!("{lp}.mlp_hyper_connection.input_mix_weight_up.weight"));
    }
    if let Some(p) = layer.ple.as_mut() {
        p.key_proj = take(format!("{lp}.ple.key_proj.weight"));
        p.value_proj = take(format!("{lp}.ple.value_proj.weight"));
    }
}

/// Read and install exactly one source layer in BF16. The layer structure comes from the already
/// loaded base artifact; only its weights are replaced. Keeping this separate from Hessian setup
/// lets calibration profiling reuse the same out-of-core layer traversal as GPTQ.
fn install_source_bf16_layer(
    gpu: &mut GpuModel,
    cfg: &crate::qwen::Config,
    src: &ShardReader,
    li: usize,
) -> Result<()> {
    let lp = format!("model.language_model.layers.{li}");
    let is_attn = matches!(cfg.layer_types[li], crate::qwen::LayerType::FullAttention);
    let mut names = Vec::new();
    {
        let layer = gpu.gptq_layer(li);
        if is_attn {
            for tensor in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                names.push(format!("{lp}.self_attn.{tensor}.weight"));
            }
            if layer.fa.as_ref().unwrap().indexer.is_some() {
                names.push(format!("{lp}.self_attn.indexer.index_qk_proj.weight"));
            }
        } else {
            for tensor in ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"] {
                names.push(format!("{lp}.linear_attn.{tensor}.weight"));
            }
        }
        match &layer.mlp {
            Ffn::Moe(_) => {
                names.push(format!("{lp}.mlp.experts.gate_up_proj"));
                names.push(format!("{lp}.mlp.experts.down_proj"));
                for tensor in ["gate_proj", "up_proj", "down_proj"] {
                    names.push(format!("{lp}.mlp.shared_expert.{tensor}.weight"));
                }
                names.push(format!("{lp}.mlp.gate.weight"));
            }
            Ffn::Dense(_) => {
                for tensor in ["gate_proj", "up_proj", "down_proj"] {
                    names.push(format!("{lp}.mlp.{tensor}.weight"));
                }
            }
        }
        if layer.hc.is_some() {
            for connection in ["attn_hyper_connection", "mlp_hyper_connection"] {
                for tensor in ["input_mix_weight_down", "input_mix_weight_up"] {
                    names.push(format!("{lp}.{connection}.{tensor}.weight"));
                }
            }
        }
        if layer.ple.is_some() {
            for tensor in ["key_proj", "value_proj"] {
                names.push(format!("{lp}.ple.{tensor}.weight"));
            }
        }
    }

    let mut weights = HashMap::new();
    for name in names {
        let (_, host) = src.read_bf16(&name)?;
        weights.insert(name, W::Bf16(gpu.gptq_upload_bf16(&host)));
    }
    install_layer(gpu, li, is_attn, &lp, &mut weights);
    Ok(())
}

