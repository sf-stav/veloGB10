//! `--gptq`: GPTQ (optionally micro-rotated, "MR-GPTQ") quantization of qwen3_5 dense and
//! qwen4_exp MoE checkpoints to the engine's NVFP4 artifact, ONE LAYER AT A TIME on one GB10.
//!
//! The memory trick: the already-quantized `--base` artifact is loaded as the model (embedding,
//! norms, PLE table on the SSD, MTP head…), and for each layer l its linear weights are swapped
//! for the bf16 originals read straight from the source shards (~2.5 GB). The calibration forward
//! is the engine's own prefill (`prefill_batch_range(lo=l, hi=l+1)`), so the activations GPTQ sees
//! are the ones the serving kernels compute; `gemm_act` / `moe_batch` taps accumulate the input
//! Hessians (per routed expert for the MoE). GPTQ then runs on the GPU (cuSOLVER Cholesky, a
//! row-parallel block sweep with NVFP4 group scales, cuBLAS propagation), the layer is swapped to
//! its quantized weights and re-run to produce the next layer's inputs (sequential GPTQ), and the
//! quantized records stream to the output shards. Peak footprint ≈ base artifact + one bf16 layer
//! + the Hessians (512 experts × 2560² f32 = 13 GB) + the calibration hidden states.
//!
//! `--rotate` applies the 16-point Hadamard micro-rotation (W' = W·R, H' = R·H·R, R = H16/4) before
//! quantizing; such an artifact needs the engine to rotate activations (`transform: hadamard16` in
//! its config.json) — see `GpuModel::rotated_ptrs`.
use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use anyhow::{anyhow, Context, Result};
use half::bf16;
use cudarc::driver::DevicePtr;
use base64::Engine;
use crate::gpu::{
    GpuModel, GptqTap, GptqHess, IgsHistogram, CalibLayerProfile, W, B, S, Pool, AttnIn, GdnIn, Ffn,
    IGS_HIST_BINS, IGS_HIST_LOG2_MIN, IGS_HIST_LOG2_MAX,
};
use crate::quant::{self, Group};

//!
//! Map of this directory (every public item keeps its `crate::gptq::` path via `pub use self::<sub>::*`):
//! - `mod.rs` (this file): options, the DF2 checkpoint records, scale fitting and the per-matrix GPTQ / RTN solvers.
//! - `io.rs`: cuSOLVER Cholesky, the safetensors range reader and the output artifact writer.
//! - `calib.rs`: calibration token sets, the Hessian / profile passes and vision calibration samples.
//! - `run.rs`: the `--gptq` driver (`run`) and the per-layer weight installers.
//! - `dflash2.rs`: sequential MR-GPTQ for the DFlash2 drafter.
//! - `lmhead.rs`: LM-head-only requantization.
//! - `igs.rs`: group parsing, input-global-scale calibration (`calib_igs`) and `refmt`.
//! - `tests.rs`: the inline unit-test module.

mod io;
mod calib;
mod run;
mod dflash2;
mod lmhead;
mod igs;

pub use self::io::*;
pub use self::calib::*;
pub use self::run::*;
pub use self::dflash2::*;
pub use self::lmhead::*;
pub use self::igs::*;

// ---------------------------------------------------------------- options
#[derive(Clone)]
pub struct GptqOpts {
    pub nsamples: usize,
    pub seqlen: usize,
    pub damp: f32,
    pub nclip: usize,
    pub rotate: bool,
    /// Matryoshka Calibration: variable-length rows with equal Hessian mass per sequence.
    pub maca: bool,
    pub scale_iters: usize,
    pub static_act_order: bool,
    pub local_hessian: bool,
    pub gptq_groups: Vec<Group>,
    pub nvfp4_groups: Vec<Group>, // GPTQ'd / RTN'd; everything else bf16
    pub fp8_groups: Vec<Group>,   // row-scaled FP8 (E4M3): the speed/accuracy middle ground
}

fn validate_opts(opts: &GptqOpts) -> Result<()> {
    anyhow::ensure!(!opts.local_hessian || opts.static_act_order,
        "--local-hessian requires static activation order (remove --no-act-order)");
    Ok(())
}

/// `igs`: the W4A4 input global scale (6·448 / calibration activation amax), written as
/// `{stem}.input_global_scale` — None for tensors the calibration never fed (RTN groups).
struct Rec { qw: Vec<u8>, sc: Vec<u8>, m: usize, k: usize, gs: f32, igs: Option<f32> }

fn write_df2_checkpoint(path: &Path, recs: &[(String, &Rec)]) -> Result<()> {
    let meta: Vec<serde_json::Value> = recs
        .iter()
        .map(|(name, r)| {
            serde_json::json!({
            "name": name, "m": r.m, "k": r.k, "gs": r.gs, "igs": r.igs,
            "qw": r.qw.len(), "sc": r.sc.len()
                })
        })
        .collect();
    let hdr = serde_json::to_vec(&meta)?;
    let tmp = path.with_extension("tmp");
    let mut f = std::fs::File::create(&tmp)?;
    f.write_all(&(hdr.len() as u64).to_le_bytes())?;
    f.write_all(&hdr)?;
    for (_, r) in recs { f.write_all(&r.qw)?; f.write_all(&r.sc)?; }
    f.sync_all()?;
    std::fs::rename(tmp, path)?;
    Ok(())
}

fn read_df2_checkpoint(path: &Path) -> Result<Vec<(String, Rec)>> {
    let mut f = std::fs::File::open(path)?;
    let mut lb = [0u8; 8]; f.read_exact(&mut lb)?;
    let mut hb = vec![0u8; u64::from_le_bytes(lb) as usize]; f.read_exact(&mut hb)?;
    let meta: Vec<serde_json::Value> = serde_json::from_slice(&hb)?;
    let mut out = Vec::with_capacity(meta.len());
    for m in meta {
        let qn = m["qw"].as_u64().ok_or_else(|| anyhow!("bad DFlash2 checkpoint qw"))? as usize;
        let sn = m["sc"].as_u64().ok_or_else(|| anyhow!("bad DFlash2 checkpoint sc"))? as usize;
        let mut qw = vec![0u8; qn]; let mut sc = vec![0u8; sn];
        f.read_exact(&mut qw)?; f.read_exact(&mut sc)?;
        out.push((m["name"].as_str().ok_or_else(|| anyhow!("bad DFlash2 checkpoint name"))?.to_string(), Rec {
            qw, sc, m: m["m"].as_u64().unwrap() as usize, k: m["k"].as_u64().unwrap() as usize,
            gs: m["gs"].as_f64().unwrap() as f32, igs: m["igs"].as_f64().map(|x| x as f32),
        }));
    }
    let mut tail = [0u8; 1];
    anyhow::ensure!(f.read(&mut tail)? == 0, "trailing bytes in {}", path.display());
    Ok(out)
}
/// Token subsample kept per layer for the down-projection fallback Hessians.
const MOE_SUB_TOKENS: usize = 16384;
fn mem_available_gb() -> f64 {
    std::fs::read_to_string("/proc/meminfo").ok().and_then(|s| s.lines().find(|l| l.starts_with("MemAvailable:"))
        .and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<f64>().ok())).map(|kb| kb / 1048576.0).unwrap_or(0.0)
}

fn e4m3_scale_of(amax: f32) -> f32 { if amax > 0.0 { amax / (quant::E2M1_MAX * quant::E4M3_MAX) } else { 1.0 } }
pub(crate) fn static_activation_order(diag: &[f32]) -> Vec<i32> {
    let mut order: Vec<usize> = (0..diag.len()).collect();
    order.sort_by(|&a, &b| match (diag[a].is_finite(), diag[b].is_finite()) {
        (true, true) => diag[b].total_cmp(&diag[a]).then_with(|| a.cmp(&b)),
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        (false, false) => a.cmp(&b),
    });
    order.into_iter().map(|i| i as i32).collect()
}

#[derive(Debug, Clone, Copy)]
struct ScaleFit {
    scale: f32,
    initial_mse: f64,
    final_mse: f64,
    accepted: usize,
}

/// Alternate between NVFP4 local-scale/code assignment and the closed-form least-squares global
/// tensor scale. A guarded line search makes this monotonic even across E4M3 assignment boundaries.
fn alternating_scale_fit<F>(initial: f32, iterations: usize, mut stats: F) -> ScaleFit
where F: FnMut(f32) -> (f64, f64, f64) {
    if iterations == 0 || !initial.is_finite() || initial <= 0.0 {
        return ScaleFit { scale: initial, initial_mse: f64::NAN, final_mse: f64::NAN, accepted: 0 };
    }
    let (_, _, initial_mse) = stats(initial);
    if !initial_mse.is_finite() {
        return ScaleFit { scale: initial, initial_mse, final_mse: initial_mse, accepted: 0 };
    }
    let mut scale = initial;
    let mut mse = initial_mse;
    let lo = initial * 0.25;
    let hi = initial * 4.0;
    let mut accepted = 0usize;
    for _ in 0..iterations {
        let (num, den, _) = stats(scale);
        if !num.is_finite() || !den.is_finite() || den <= 0.0 { break; }
        let mut candidate = (num / den) as f32;
        if !candidate.is_finite() || candidate <= 0.0 { break; }
        candidate = candidate.clamp(lo, hi);
        if (candidate - scale).abs() <= scale.abs().max(f32::MIN_POSITIVE) * 1e-6 { break; }

        let mut trial = candidate;
        let mut trial_mse = stats(trial).2;
        let mut found = trial_mse.is_finite() && trial_mse < mse;
        for _ in 0..8 {
            if found { break; }
            trial = 0.5 * (scale + trial);
            trial_mse = stats(trial).2;
            found = trial_mse.is_finite() && trial_mse < mse;
        }
        if !found { break; }
        scale = trial;
        mse = trial_mse;
        accepted += 1;
    }
    ScaleFit { scale, initial_mse, final_mse: mse, accepted }
}

fn optimize_w32_scale(gpu: &GpuModel, w32: &S, n: usize, initial: f32, opts: &GptqOpts) -> f32 {
    let fit = alternating_scale_fit(initial, opts.scale_iters,
        |s| gpu.gptq_scale_stats_f32(w32, n, s, opts.nclip));
    if fit.accepted > 0 {
        println!("[gptq] NVFP4 scale: {:.6e} -> {:.6e}, SSE {:.6e} -> {:.6e} ({} steps)",
                 initial, fit.scale, fit.initial_mse, fit.final_mse, fit.accepted);
    }
    fit.scale
}

fn optimize_bf16_scale(gpu: &GpuModel, w_ptr: u64, n: usize, initial: f32, opts: &GptqOpts) -> f32 {
    let fit = alternating_scale_fit(initial, opts.scale_iters,
        |s| gpu.gptq_scale_stats_bf16(w_ptr, n, opts.rotate, s, opts.nclip));
    if fit.accepted > 0 {
        println!("[gptq] stacked NVFP4 scale: {:.6e} -> {:.6e}, SSE {:.6e} -> {:.6e} ({} steps)",
                 initial, fit.scale, fit.initial_mse, fit.final_mse, fit.accepted);
    }
    fit.scale
}

/// GPTQ one 2-D weight (bf16 on device at `w_ptr`, [m, k] row-major) with its Hessian.
fn igs_of(amax: f32) -> Option<f32> { if amax > 0.0 && amax.is_finite() { Some(6.0 * 448.0 / amax) } else { None } }
fn gptq_2d(gpu: &GpuModel, cs: &mut Cusolver, w_ptr: u64, m: usize, k: usize, hess: &S, opts: &GptqOpts, s_tensor: Option<f32>, x_amax: f32) -> Result<Rec> {
    validate_opts(opts)?;
    let w32 = gpu.gptq_w32(w_ptr, m, k, opts.rotate);
    let initial_st = s_tensor.unwrap_or_else(|| e4m3_scale_of(gpu.gptq_absmax_f32(&w32, m * k)));
    // Stacked MoE tensors pass their already jointly optimized scale; ordinary matrices optimize
    // their own global scale from the original rotated f32 weight.
    let st = if s_tensor.is_some() { initial_st } else { optimize_w32_scale(gpu, &w32, m * k, initial_st, opts) };

    // Adaptive damping (GPTQModel-style): 1 % → 5 % → 10 % of mean(diag) before falling back to
    // RTN (U = I: the sweep then rounds without error feedback, same scales and static groups).
    let mut u: Option<S> = None;
    let mut perm = None;
    let mut used_damp = opts.damp;
    for damp in [opts.damp, 0.05, 0.10] {
        let (h32, p) = gpu.gptq_h32(hess, k, opts.rotate, damp, opts.static_act_order);
        perm = p;
        match cs.chol_inv_chol(gpu, &h32, k) {
            Ok(()) => { used_damp = damp; u = Some(h32); break; }
            Err(e) => eprintln!("[gptq] cholesky failed at damp {damp}: {e} — retrying"),
        }
    }
    let u = match u { Some(u) => u, None => {
        eprintln!("[gptq] Hessian not usable — RTN fallback for this tensor");
        let mut id = vec![0f32; k * k]; for i in 0..k { id[i * k + i] = 1.0; }
        gpu.gptq_dev().htod_sync_copy(&id)?
    } };
    let (qw, sc) = if let Some(perm) = perm {
        // Freeze block scales before error feedback, then work in importance order and write codes
        // directly into their original columns. The artifact and serving path remain unchanged.
        let qs = if opts.local_hessian {
            // Inspect every positive finite E4M3 local scale and minimize dw^T H_block dw.
            // H stays in the original column layout, so scales remain artifact-compatible even
            // though the GPTQ sweep itself runs in activation-importance order.
            let (h_local, _) = gpu.gptq_h32(hess, k, opts.rotate, used_damp, false);
            gpu.gptq_static_scales_hessian(&w32, &h_local, m, k, st)
        } else {
            gpu.gptq_static_scales(&w32, m, k, st, opts.nclip)
        };
        let wp = gpu.gptq_permute_w(&w32, m, k, &perm);
        gpu.gptq_sweep_static(&wp, &u, &perm, &qs, m, k, st)
    } else {
        gpu.gptq_sweep(&w32, &u, m, k, st, opts.nclip)
    };
    Ok(Rec { qw, sc, m, k, gs: 1.0 / st, igs: igs_of(x_amax) })
}

/// Plain RTN through the quantizer's own codec (weights the calibration never touches: the MTP head).
fn rtn_2d(w: &[bf16], m: usize, k: usize) -> Rec {
    let q = quant::quantize_nvfp4(w, m, k);
    Rec { qw: q.qweight, sc: q.scales, m, k, gs: q.global_scale, igs: None }
}


#[cfg(test)]
mod tests;
