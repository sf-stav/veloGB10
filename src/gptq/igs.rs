//! Group-list parsing, input-global-scale (IGS) calibration (`calib_igs`) and `--gptq-refmt` (`refmt`).
//! Siblings: `run.rs` / `calib.rs` feed the histograms it reads; `mod.rs` has the map.

use super::*;

pub fn parse_groups(s: &str) -> Result<Vec<Group>> {
    s.split(',').map(|t| t.trim()).filter(|t| !t.is_empty()).map(|t| match t {
        "expert" => Ok(Group::Expert), "attn" => Ok(Group::Attn), "mlp" => Ok(Group::Mlp), "gdn" => Ok(Group::Gdn),
        "hc" => Ok(Group::Hc), "ple" => Ok(Group::Ple), "lmhead" => Ok(Group::LmHead), "embed" => Ok(Group::Embed),
        "mtp" => Ok(Group::Mtp), "router" => Ok(Group::Router), o => Err(anyhow!("unknown group {o}")) }).collect()
}


/// Policy used to derive the reciprocal runtime input scale from mergeable block-amax statistics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IgsMethod {
    Max,
    Headroom,
}

impl IgsMethod {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "max" => Ok(Self::Max),
            "headroom" => Ok(Self::Headroom),
            other => Err(anyhow!("unknown --igs-method {other}; expected max or headroom")),
        }
    }

    fn name(self) -> &'static str {
        match self { Self::Max => "max", Self::Headroom => "headroom" }
    }
}

/// NVIDIA-style NVFP4 activation headroom parameters. The artifact stores the reciprocal convention:
/// `input_global_scale = 6*448/amax`, whereas ModelOpt exports `input_scale = amax/(6*448)`.
#[derive(Clone, Copy, Debug)]
pub struct IgsCalibConfig {
    pub method: IgsMethod,
    pub anchor_percentile: f32,
    pub upper_percentile: f32,
    pub rho: f32,
}

impl Default for IgsCalibConfig {
    fn default() -> Self {
        Self { method: IgsMethod::Headroom, anchor_percentile: 1.0, upper_percentile: 99.99, rho: 16384.0 }
    }
}

impl IgsCalibConfig {
    pub fn validate(self) -> Result<Self> {
        anyhow::ensure!(self.anchor_percentile > 0.0 && self.anchor_percentile <= 100.0,
                        "--igs-anchor-percentile must be in (0, 100]");
        anyhow::ensure!(self.upper_percentile > 0.0 && self.upper_percentile <= 100.0,
                        "--igs-upper-percentile must be in (0, 100]");
        anyhow::ensure!(self.rho > 0.0 && self.rho < 28672.0,
                        "--igs-rho must be in (0, 28672)");
        Ok(self)
    }
}

#[derive(Clone, Debug)]
struct IgsScaleDiagnostic {
    input_global_scale: f32,
    selected_amax: f32,
    anchor: Option<f32>,
    upper: f32,
    span: Option<f32>,
    has_headroom: bool,
    range_exceeds_e4m3: bool,
}

fn igs_hist_bin_index(value: f32) -> usize {
    let frac = (value.log2() - IGS_HIST_LOG2_MIN) / (IGS_HIST_LOG2_MAX - IGS_HIST_LOG2_MIN);
    (frac * IGS_HIST_BINS as f32).floor().clamp(0.0, (IGS_HIST_BINS - 1) as f32) as usize
}

fn igs_hist_bin_center(index: usize) -> f32 {
    let log2 = IGS_HIST_LOG2_MIN
        + (index as f32 + 0.5) / IGS_HIST_BINS as f32 * (IGS_HIST_LOG2_MAX - IGS_HIST_LOG2_MIN);
    2.0f32.powf(log2)
}

fn igs_hist_percentile(hist: &[u64], percentile: f32, floor_value: Option<f32>) -> Option<f32> {
    if hist.len() != IGS_HIST_BINS { return None; }
    let floor_bin = floor_value.filter(|v| *v > 0.0).map(igs_hist_bin_index).unwrap_or(0);
    let total: u64 = hist[floor_bin..].iter().sum();
    if total == 0 { return None; }
    let target = percentile as f64 / 100.0 * total as f64;
    let mut cumulative = 0u64;
    for (index, &count) in hist.iter().enumerate().skip(floor_bin) {
        cumulative = cumulative.saturating_add(count);
        if cumulative as f64 >= target {
            return Some(igs_hist_bin_center(index));
        }
    }
    None
}

fn igs_scale_from_hist(stats: &IgsHistogram, cfg: IgsCalibConfig) -> Option<IgsScaleDiagnostic> {
    if !stats.running_max.is_finite() || stats.running_max <= 0.0 || stats.invalid_blocks != 0 {
        return None;
    }
    if cfg.method == IgsMethod::Max {
        return igs_of(stats.running_max).map(|scale| IgsScaleDiagnostic {
            input_global_scale: scale,
            selected_amax: stats.running_max,
            anchor: None,
            upper: stats.running_max,
            span: None,
            has_headroom: false,
            range_exceeds_e4m3: false,
        });
    }

    let upper = if cfg.upper_percentile >= 100.0 {
        stats.running_max
    } else {
        igs_hist_percentile(&stats.bins, cfg.upper_percentile, None).unwrap_or(stats.running_max)
    };
    let anchor = igs_hist_percentile(&stats.bins, cfg.anchor_percentile, Some(upper / 1.0e6));
    let Some(anchor) = anchor.filter(|v| v.is_finite() && *v > 0.0) else {
        return igs_of(stats.running_max).map(|scale| IgsScaleDiagnostic {
            input_global_scale: scale,
            selected_amax: stats.running_max,
            anchor: None,
            upper: stats.running_max,
            span: None,
            has_headroom: false,
            range_exceeds_e4m3: false,
        });
    };
    let span = upper / anchor;
    let selected_amax = upper.max(cfg.rho * anchor);
    igs_of(selected_amax).map(|scale| IgsScaleDiagnostic {
        input_global_scale: scale,
        selected_amax,
        anchor: Some(anchor),
        upper,
        span: Some(span),
        has_headroom: cfg.rho * anchor > upper,
        range_exceeds_e4m3: span > 28672.0,
    })
}

/// Calibrate W4A4 input scales on an existing final W4 artifact. Statistics are collected after any
/// MR rotation. The default headroom policy follows NVIDIA's per-16 block-amax histogram method and
/// emits a mergeable sidecar; `--igs-method max` retains the legacy literal-max behavior.
pub fn calib_igs(
    model_dir: &Path,
    out: &Path,
    calib: &Path,
    nsamples: usize,
    seqlen: usize,
    igs_cfg: IgsCalibConfig,
) -> Result<()> {
    let igs_cfg = igs_cfg.validate()?;
    if crate::opts::var(crate::opt!("ple-offload")).is_err() { crate::opts::set(crate::opt!("ple-offload"), "ssd"); }
    anyhow::ensure!(crate::opts::var_os(crate::opt!("w4a4-prefill")).is_none(),
                    "--calib-igs must collect the unquantized GEMM inputs; unset --w4a4-prefill before calibration");
    anyhow::ensure!(crate::opts::var_os(crate::opt!("w4a4-verify")).is_none(),
                    "--calib-igs must collect the unquantized GEMM inputs; unset --w4a4-verify before calibration");
    std::fs::create_dir_all(out)?;
    let t_all = std::time::Instant::now();
    let (gpu, cfg) = GpuModel::load_from_dir(&model_dir.to_string_lossy())?;
    let vision_samples = vision_calib_samples(model_dir, calib, nsamples, seqlen)?;
    // Text calibration corpora may be MaCa sample-aligned JSONL with mixed sequence lengths.
    // `seqlen` is the state/allocation ceiling, not a requirement that every row be padded to it.
    let samples = if vision_samples.is_none() {
        Some(calib_tokens_mode(model_dir, calib, nsamples, seqlen, cfg.vocab_size, true)?)
    } else {
        None
    };
    let sample_count = vision_samples.as_ref().map_or_else(|| samples.as_ref().unwrap().len(), Vec::len);
    let total_tokens: usize = vision_samples
        .as_ref()
        .map_or_else(|| samples.as_ref().unwrap().iter().map(Vec::len).sum(), |rows| rows.iter().map(|row| row.tokens.len()).sum());
    let weights = gpu.nvfp4_weights_by_name();
    let ptrs: Vec<u64> = weights.iter().map(|w| w.1).collect();
    println!("[calib-igs] method={} — {} {} samples / {total_tokens} tokens (max seqlen {seqlen}) over {} NVFP4 weights ({} stems)",
             igs_cfg.method.name(), sample_count, if vision_samples.is_some() { "vision" } else { "text" },
             { let mut p = ptrs.clone(); p.sort(); p.dedup(); p.len() }, weights.len());
    gpu.igs_arm(&ptrs);
    let mut pool = Pool::new(gpu.gptq_dev().clone());
    let mut state = gpu.new_batch_state(1, 1, seqlen);
    let nl = cfg.num_layers;
    for s in 0..sample_count {
        gpu.zero_slot_state(&mut state, 0, seqlen);
        let toks = if let Some(vision) = &vision_samples {
            let sample = &vision[s];
            let embeds: Vec<bf16> = sample.image_embeds.iter().copied().map(bf16::from_f32).collect();
            state.vision_embeds = Some(gpu.gptq_dev().htod_sync_copy(&embeds)?);
            state.vision_spans = sample.spans.clone();
            &sample.tokens
        } else { &samples.as_ref().unwrap()[s] };
        let sample_seqlen = toks.len();
        let _ = gpu.prefill_batch_range(
            &mut pool,
            toks,
            &mut state,
            0,
            sample_seqlen,
            0,
            0,
            nl,
            None,
        );
        if (s + 1) % 32 == 0 || s + 1 == sample_count {
            gpu.gptq_sync();
            println!("[calib-igs] {}/{} samples, {:.1} min", s + 1, sample_count, t_all.elapsed().as_secs_f32() / 60.0);
        }
    }

    let histograms = gpu.igs_disarm();
    let mut scales = serde_json::Map::new();
    let mut stem_stats = serde_json::Map::new();
    let (mut n_ok, mut n_unfed, mut n_invalid, mut n_wide) = (0, 0, 0, 0);
    for (stem, ptr, _k) in &weights {
        let Some(stats) = histograms.get(ptr) else { n_unfed += 1; continue; };
        let diag = igs_scale_from_hist(stats, igs_cfg);
        if stats.invalid_blocks > 0 { n_invalid += 1; }
        if let Some(d) = &diag {
            scales.insert(stem.clone(), serde_json::json!(d.input_global_scale));
            n_ok += 1;
            if d.range_exceeds_e4m3 { n_wide += 1; }
        } else {
            n_unfed += 1;
        }
        let nonzero_blocks: u64 = stats.bins.iter().sum();
        stem_stats.insert(
            stem.clone(),
            serde_json::json!({
            "histogram": stats.bins,
            "running_max": stats.running_max,
            "zero_blocks": stats.zero_blocks,
            "invalid_blocks": stats.invalid_blocks,
            "nonzero_blocks": nonzero_blocks,
            "anchor": diag.as_ref().and_then(|d| d.anchor),
            "upper": diag.as_ref().map(|d| d.upper),
            "span": diag.as_ref().and_then(|d| d.span),
            "selected_amax": diag.as_ref().map(|d| d.selected_amax),
            "input_global_scale": diag.as_ref().map(|d| d.input_global_scale),
            "has_headroom": diag.as_ref().map(|d| d.has_headroom),
            "range_exceeds_e4m3": diag.as_ref().map(|d| d.range_exceeds_e4m3),
            }),
        );
    }

    let stats_doc = serde_json::json!({
        "format": "veloGB10-igs-hist-v2",
        "scale_convention": "input_global_scale = 2688 / activation_amax",
        "block_size": 16,
        "histogram": {
            "bins": IGS_HIST_BINS,
            "log2_min": IGS_HIST_LOG2_MIN,
            "log2_max": IGS_HIST_LOG2_MAX,
        },
        "policy": {
            "method": igs_cfg.method.name(),
            "anchor_percentile": igs_cfg.anchor_percentile,
            "upper_percentile": igs_cfg.upper_percentile,
            "rho": igs_cfg.rho,
        },
        "calibration": {
            "model_dir": model_dir,
            "corpus": calib,
            "nsamples": sample_count,
            "seqlen": seqlen,
            "total_tokens": total_tokens,
        },
        "stems": stem_stats,
    });
    let stats_path = out.join("input_global_scale.stats.json");
    std::fs::write(&stats_path, serde_json::to_string_pretty(&stats_doc)? + "\n")?;
    anyhow::ensure!(n_invalid == 0,
                    "{n_invalid} stems observed non-finite activations; refusing to write scales (stats: {})",
                    stats_path.display());

    let scale_path = out.join("input_global_scale.json");
    std::fs::write(&scale_path, serde_json::to_string_pretty(&serde_json::Value::Object(scales))? + "\n")?;
    println!("[calib-igs] {n_ok} scales written ({n_unfed} unfed, {n_wide} spans > E4M3 normal range) → {} in {:.1} min",
             scale_path.display(), t_all.elapsed().as_secs_f32() / 60.0);
    println!("[calib-igs] mergeable histogram stats → {}", stats_path.display());
    Ok(())
}

/// `--gptq-refmt`: re-format an existing artifact without recalibrating — the bf16 2-D weights of
/// `fp8_groups` become row-scaled FP8 (`quant::quantize_fp8`), those of `nvfp4_groups` RTN NVFP4;
/// every other tensor (packed triples included) is copied verbatim, the PLE files are linked.
pub fn refmt(input: &Path, out: &Path, fp8_groups: &[Group], nvfp4_groups: &[Group]) -> Result<()> {
    prepare_output_dir(out, &[input])?;
    let rd = ShardReader::open(input)?;
    let mut writer = Writer::new(out, 4 * 1024 * 1024 * 1024);
    let (mut n8, mut n4, mut nc) = (0, 0, 0);
    for (name, meta) in rd.metas.iter() {
        let stem = name.strip_suffix(".weight").unwrap_or(name).to_string();
        let g = quant::group_of(name);
        let quantizable = meta.dtype == "BF16" && meta.shape.len() == 2 && meta.shape[0] % 16 == 0 && meta.shape[1] % 16 == 0
            && !name.contains(".visual.") && name.ends_with(".weight");
        if quantizable && fp8_groups.contains(&g) {
            let (shape, v) = rd.read_bf16(name)?; writer.push_fp8(&stem, quant::quantize_fp8(&v, shape[0], shape[1])); n8 += 1; continue;
        }
        if quantizable && nvfp4_groups.contains(&g) {
            let (shape, v) = rd.read_bf16(name)?; let r = rtn_2d(&v, shape[0], shape[1]);
            writer.push_nvfp4(&stem, r.qw, r.sc, r.m, r.k, r.gs, r.igs); n4 += 1; continue;
        }
        let (_, data) = rd.read_bytes(name)?;
        let dtype = match meta.dtype.as_str() { "BF16" => safetensors::Dtype::BF16, "F32" => safetensors::Dtype::F32, "I64" => safetensors::Dtype::I64,
            "F16" => safetensors::Dtype::F16, "U8" => safetensors::Dtype::U8, "F8_E4M3" => safetensors::Dtype::F8_E4M3, o => return Err(anyhow!("dtype {o} on {name}")) };
        writer.push(Out { name: name.clone(), dtype, shape: meta.shape.clone(), data }); nc += 1;
    }
    writer.finish()?;
    for f in std::fs::read_dir(input)? {
        let f = f?; let n = f.file_name().to_string_lossy().to_string();
        if n.ends_with(".safetensors") || n == "model.safetensors.index.json" { continue; }
        let dst = out.join(&n);
        // only the big PLE table is hard-linked; everything else is COPIED (config.json is edited below —
        // a hard link would edit the input artifact's config in place)
        let big = n.starts_with("ple_ngram") && n.ends_with(".bin");
        if !(big && std::fs::hard_link(f.path(), &dst).is_ok()) { std::fs::copy(f.path(), &dst)?; }
    }
    let cp = out.join("config.json");
    let mut cj: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&cp)?)?;
    cj["quantization_config"]["refmt"] = serde_json::json!({ "fp8_groups": fp8_groups.iter().map(|g| quant::group_name(*g)).collect::<Vec<_>>(),
        "rtn_groups": nvfp4_groups.iter().map(|g| quant::group_name(*g)).collect::<Vec<_>>() });
    std::fs::write(&cp, serde_json::to_string_pretty(&cj)?)?;
    println!("[gptq-refmt] {n8} tensors → fp8, {n4} → nvfp4 (RTN), {nc} copied → {}", out.display());
    Ok(())
}


