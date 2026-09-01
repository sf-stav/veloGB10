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

// ---------------------------------------------------------------- cuSOLVER (dense Cholesky)
#[link(name = "cusolver")]
extern "C" {
    fn cusolverDnCreate(handle: *mut *mut std::ffi::c_void) -> i32;
    fn cusolverDnDestroy(handle: *mut std::ffi::c_void) -> i32;
    fn cusolverDnSpotrf_bufferSize(h: *mut std::ffi::c_void, uplo: i32, n: i32, a: *mut f32, lda: i32, lwork: *mut i32) -> i32;
    fn cusolverDnSpotrf(h: *mut std::ffi::c_void, uplo: i32, n: i32, a: *mut f32, lda: i32, work: *mut f32, lwork: i32, info: *mut i32) -> i32;
    fn cusolverDnSpotri_bufferSize(h: *mut std::ffi::c_void, uplo: i32, n: i32, a: *mut f32, lda: i32, lwork: *mut i32) -> i32;
    fn cusolverDnSpotri(h: *mut std::ffi::c_void, uplo: i32, n: i32, a: *mut f32, lda: i32, work: *mut f32, lwork: i32, info: *mut i32) -> i32;
}
const CUBLAS_FILL_MODE_LOWER: i32 = 0;

struct Cusolver { h: *mut std::ffi::c_void, work: Option<(cudarc::driver::CudaSlice<f32>, usize)>, info: cudarc::driver::CudaSlice<i32> }
impl Cusolver {
    fn new(gpu: &GpuModel) -> Result<Self> {
        let mut h: *mut std::ffi::c_void = std::ptr::null_mut();
        let rc = unsafe { cusolverDnCreate(&mut h) };
        if rc != 0 { return Err(anyhow!("cusolverDnCreate failed ({rc})")); }
        Ok(Self { h, work: None, info: gpu.gptq_dev().alloc_zeros::<i32>(1)? })
    }
    /// In place on the device buffer `a` (f32 [n, n], symmetric PD): a ← upper Cholesky factor U
    /// of a⁻¹ in ROW-MAJOR terms (a⁻¹ = UᵀU), i.e. cuSOLVER's lower factor of the lower-mode inverse.
    fn chol_inv_chol(&mut self, gpu: &GpuModel, a: &S, n: usize) -> Result<()> {
        let dev = gpu.gptq_dev().clone();
        gpu.gptq_sync();
        let mut lw1 = 0i32; let mut lw2 = 0i32;
        let ap = *a.device_ptr() as *mut f32;
        unsafe {
            cusolverDnSpotrf_bufferSize(self.h, CUBLAS_FILL_MODE_LOWER, n as i32, ap, n as i32, &mut lw1);
            cusolverDnSpotri_bufferSize(self.h, CUBLAS_FILL_MODE_LOWER, n as i32, ap, n as i32, &mut lw2);
        }
        let lw = lw1.max(lw2).max(1) as usize;
        if self.work.as_ref().map_or(true, |(_, cap)| *cap < lw) { self.work = Some((dev.alloc_zeros::<f32>(lw)?, lw)); }
        let wp = *self.work.as_ref().unwrap().0.device_ptr() as *mut f32;
        let ip = *self.info.device_ptr() as *mut i32;
        let check = |tag: &str, rc: i32, info: &cudarc::driver::CudaSlice<i32>| -> Result<()> {
            if rc != 0 { return Err(anyhow!("cusolver {tag} returned {rc}")); }
            dev.synchronize()?;
            let v = dev.dtoh_sync_copy(info)?;
            if v[0] != 0 { return Err(anyhow!("cusolver {tag}: info = {} (not positive definite?)", v[0])); }
            Ok(())
        };
        unsafe {
            let rc = cusolverDnSpotrf(self.h, CUBLAS_FILL_MODE_LOWER, n as i32, ap, n as i32, wp, lw as i32, ip); check("potrf", rc, &self.info)?;
            let rc = cusolverDnSpotri(self.h, CUBLAS_FILL_MODE_LOWER, n as i32, ap, n as i32, wp, lw as i32, ip); check("potri", rc, &self.info)?;
            let rc = cusolverDnSpotrf(self.h, CUBLAS_FILL_MODE_LOWER, n as i32, ap, n as i32, wp, lw as i32, ip); check("potrf(inv)", rc, &self.info)?;
        }
        Ok(())
    }
}
impl Drop for Cusolver { fn drop(&mut self) { unsafe { cusolverDnDestroy(self.h); } } }

// ---------------------------------------------------------------- safetensors range reader
#[derive(Clone)]
pub struct TensorMeta { pub dtype: String, pub shape: Vec<usize>, pub off: (u64, u64), pub file: PathBuf, pub data_start: u64 }

pub struct ShardReader { pub metas: BTreeMap<String, TensorMeta> }
impl ShardReader {
    pub fn open(dir: &Path) -> Result<Self> {
        let mut files: Vec<PathBuf> = Vec::new();
        let idx = dir.join("model.safetensors.index.json");
        if idx.exists() {
            let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&idx)?)?;
            let mut set = std::collections::BTreeSet::new();
            for (_, v) in j["weight_map"].as_object().unwrap() { set.insert(v.as_str().unwrap().to_string()); }
            files.extend(set.into_iter().map(|f| dir.join(f)));
        } else { files.push(dir.join("model.safetensors")); }
        let mut metas = BTreeMap::new();
        for f in &files {
            let mut fh = std::fs::File::open(f).with_context(|| format!("open {}", f.display()))?;
            let mut lenb = [0u8; 8]; fh.read_exact(&mut lenb)?;
            let hlen = u64::from_le_bytes(lenb);
            let mut hb = vec![0u8; hlen as usize]; fh.read_exact(&mut hb)?;
            let hj: serde_json::Value = serde_json::from_slice(&hb)?;
            for (name, v) in hj.as_object().unwrap() {
                if name == "__metadata__" { continue; }
                let shape: Vec<usize> = v["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
                let o = v["data_offsets"].as_array().unwrap();
                metas.insert(name.clone(), TensorMeta { dtype: v["dtype"].as_str().unwrap().to_string(), shape,
                    off: (o[0].as_u64().unwrap(), o[1].as_u64().unwrap()), file: f.clone(), data_start: 8 + hlen });
            }
        }
        Ok(Self { metas })
    }
    pub fn read_bytes(&self, name: &str) -> Result<(TensorMeta, Vec<u8>)> {
        let m = self.metas.get(name).ok_or_else(|| anyhow!("missing tensor {name}"))?.clone();
        let mut fh = std::fs::File::open(&m.file)?;
        fh.seek(SeekFrom::Start(m.data_start + m.off.0))?;
        let mut buf = vec![0u8; (m.off.1 - m.off.0) as usize];
        fh.read_exact(&mut buf)?;
        Ok((m, buf))
    }
    pub fn read_bf16(&self, name: &str) -> Result<(Vec<usize>, Vec<bf16>)> {
        let (m, b) = self.read_bytes(name)?;
        anyhow::ensure!(m.dtype == "BF16", "{name}: expected BF16, got {}", m.dtype);
        Ok((m.shape, bytemuck::cast_slice::<u8, bf16>(&b).to_vec()))
    }
}

// ---------------------------------------------------------------- output artifact writer
struct Out { name: String, dtype: safetensors::Dtype, shape: Vec<usize>, data: Vec<u8> }
struct Writer { dir: PathBuf, outs: Vec<Out>, bytes: usize, shard_idx: usize, weight_map: serde_json::Map<String, serde_json::Value>, total: u64, shard_bytes: usize }
impl Writer {
    fn new(dir: &Path, shard_bytes: usize) -> Self { Self { dir: dir.to_path_buf(), outs: Vec::new(), bytes: 0, shard_idx: 0, weight_map: Default::default(), total: 0, shard_bytes } }
    // A shard boundary may only fall BETWEEN tensor families: the loader pairs an NVFP4 triple
    // (weight_packed / weight_scale / weight_global_scale) within one shard.
    fn push(&mut self, o: Out) {
        // Verbatim copies of a packed family arrive in name order (weight_global_scale, weight_packed,
        // weight_scale): hold the shard boundary until the family's last member.
        // (input_global_scale sorts first: ".input_global_scale" < ".weight_*" — it holds too)
        let hold = o.name.ends_with(".weight_global_scale") || o.name.ends_with(".weight_packed") || o.name.ends_with(".input_global_scale");
        self.push_raw(o);
        if !hold && self.bytes >= self.shard_bytes { self.flush(); }
    }
    fn push_raw(&mut self, o: Out) { self.bytes += o.data.len(); self.total += o.data.len() as u64; self.outs.push(o); }
    fn push_fp8(&mut self, stem: &str, q: quant::Fp8Tensor) {
        let sc: Vec<u8> = q.row_scale.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.push_raw(Out { name: format!("{stem}.weight"), dtype: safetensors::Dtype::F8_E4M3, shape: vec![q.m, q.k], data: q.qweight });
        self.push_raw(Out { name: format!("{stem}.weight_scale"), dtype: safetensors::Dtype::F32, shape: vec![q.m], data: sc });
        if self.bytes >= self.shard_bytes { self.flush(); }
    }
    fn push_nvfp4(&mut self, stem: &str, qw: Vec<u8>, sc: Vec<u8>, m: usize, k: usize, gs: f32, igs: Option<f32>) {
        self.push_raw(Out { name: format!("{stem}.weight_packed"), dtype: safetensors::Dtype::U8, shape: vec![m, k / 2], data: qw });
        self.push_raw(Out { name: format!("{stem}.weight_scale"), dtype: safetensors::Dtype::F8_E4M3, shape: vec![m, k / 16], data: sc });
        self.push_raw(Out { name: format!("{stem}.weight_global_scale"), dtype: safetensors::Dtype::F32, shape: vec![1], data: gs.to_le_bytes().to_vec() });
        if let Some(g) = igs {
            self.push_raw(Out { name: format!("{stem}.input_global_scale"), dtype: safetensors::Dtype::F32, shape: vec![1], data: g.to_le_bytes().to_vec() });
        }
        if self.bytes >= self.shard_bytes { self.flush(); }
    }
    fn flush(&mut self) {
        if self.outs.is_empty() { return; }
        self.shard_idx += 1;
        let fname = format!("model-{:05}.safetensors", self.shard_idx);
        let views: Vec<(String, safetensors::tensor::TensorView)> = self.outs.iter()
            .map(|o| (o.name.clone(), safetensors::tensor::TensorView::new(o.dtype, o.shape.clone(), &o.data).expect("view"))).collect();
        let meta: std::collections::HashMap<String, String> = [("format".to_string(), "pt".to_string())].into_iter().collect();
        safetensors::serialize_to_file(views, Some(meta), &self.dir.join(&fname)).expect("write shard");
        for o in &self.outs { self.weight_map.insert(o.name.clone(), serde_json::Value::String(fname.clone())); }
        println!("  wrote {fname} ({:.2} GB, {} tensors)", self.bytes as f64 / 1e9, self.outs.len());
        self.outs.clear(); self.bytes = 0;
    }
    fn finish(mut self) -> Result<()> {
        self.flush();
        // The loader pairs a packed family within ONE shard: refuse an index that splits one
        // (a served artifact would otherwise die at its first start with "tensor … not found").
        let mut split = Vec::new();
        for (k, v) in &self.weight_map {
            if let Some(stem) = k.strip_suffix(".weight_packed") {
                for suf in [".weight_scale", ".weight_global_scale", ".input_global_scale"] {
                    if let Some(sv) = self.weight_map.get(&format!("{stem}{suf}")) { if sv != v { split.push(format!("{stem}{suf}")); } }
                    else if suf != ".input_global_scale" { split.push(format!("{stem}{suf} (missing)")); }
                }
            }
            if let Some(stem) = k.strip_suffix(".weight_scale") {
                if self.weight_map.contains_key(&format!("{stem}.weight")) && self.weight_map.get(&format!("{stem}.weight")) != Some(v) { split.push(format!("{stem}.weight (fp8)")); }
            }
        }
        anyhow::ensure!(split.is_empty(), "artifact writer: {} tensor families split across shards: {:?}", split.len(), &split[..split.len().min(4)]);
        let index = serde_json::json!({ "metadata": { "total_size": self.total }, "weight_map": self.weight_map });
        std::fs::write(self.dir.join("model.safetensors.index.json"), serde_json::to_string_pretty(&index)?)?;
        println!("[gptq] index written: {} tensors in {} shards, every packed family within one shard", self.weight_map.len(), self.shard_idx);
        Ok(())
    }
}

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
                tool_calls: None, tool_call_id: None, name: None, reasoning_content: None, images: vec![] }).collect();
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
    if std::env::var("GB10_PLE_OFFLOAD").is_err() {
        std::env::set_var("GB10_PLE_OFFLOAD", "ssd");
    }
    anyhow::ensure!(
        std::env::var_os("GB10_W4A4_PREFILL").is_none(),
        "--calib-profile requires unquantized activations; unset GB10_W4A4_PREFILL"
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
    if std::env::var("GB10_PLE_OFFLOAD").is_err() {
        std::env::set_var("GB10_PLE_OFFLOAD", "ssd");
    }
    anyhow::ensure!(
        std::env::var_os("GB10_W4A4_PREFILL").is_none(),
        "--calib-profile requires unquantized activations; unset GB10_W4A4_PREFILL"
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

// ---------------------------------------------------------------- the driver
pub fn run(source: &Path, base: &Path, out: &Path, calib: &Path, opts: GptqOpts) -> Result<()> {
    validate_opts(&opts)?;
    prepare_output_dir(out, &[source, base])?;
    if std::env::var("GB10_PLE_OFFLOAD").is_err() { std::env::set_var("GB10_PLE_OFFLOAD", "ssd"); }
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

/// Sequential MR-GPTQ for the 5-layer DFlash2 drafter. Target prompt taps are captured with
/// the production W4A4 prefill enabled, projected once, and cached beside the output. Each draft
/// layer is then calibrated on the already-quantized prefix of the drafter stack.
pub fn dflash2(source: &Path, target: &Path, out: &Path, calib: &Path,
               opts: GptqOpts, context_vectors: usize) -> Result<()> {
    use crate::dflash2::{BLOCK as DB, HEAD_DIM, HIDDEN, INTER, N_LAYERS, NUM_HEADS, NUM_KV_HEADS};
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
    std::env::set_var("GB10_W4A4_PREFILL", "attn,mlp,gdn");
    if std::env::var("GB10_PLE_OFFLOAD").is_err() { std::env::set_var("GB10_PLE_OFFLOAD", "ssd"); }
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
            == Some(4 + HIDDEN * opts.seqlen * 2) { continue; }
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
        pool.release_bf16(residual, HIDDEN * opts.seqlen);
        if (si + 1) % 8 == 0 || si + 1 == samples.len() {
            println!("[gptq-df2] tap cache {}/{} ({:.1} min)", si + 1, samples.len(), t_all.elapsed().as_secs_f32()/60.0);
        }
    }
    gpu.set_df2_prime_off();
    drop(state); drop(pool); drop(prime);

    let mut cs = Cusolver::new(&gpu)?;
    let mut records: HashMap<String, Rec> = HashMap::new();
    let specs = [
        ("self_attn.q_proj", NUM_HEADS * HEAD_DIM, HIDDEN, 0usize),
        ("self_attn.k_proj", NUM_KV_HEADS * HEAD_DIM, HIDDEN, 1usize),
        ("self_attn.v_proj", NUM_KV_HEADS * HEAD_DIM, HIDDEN, 1usize),
        ("self_attn.o_proj", HIDDEN, NUM_HEADS * HEAD_DIM, 2usize),
        ("mlp.gate_proj", INTER, HIDDEN, 3usize),
        ("mlp.up_proj", INTER, HIDDEN, 3usize),
        ("mlp.down_proj", HIDDEN, INTER, 4usize),
    ];
    for li in 0..N_LAYERS {
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
        let mut hq = gpu.gptq_hess_new(HIDDEN);
        let mut hkv = gpu.gptq_hess_new(HIDDEN);
        let mut ho = gpu.gptq_hess_new(NUM_HEADS * HEAD_DIM);
        let mut hgu = gpu.gptq_hess_new(HIDDEN);
        let mut hdn = gpu.gptq_hess_new(INTER);
        for si in 0..samples.len() {
            let raw = std::fs::read(cache.join(format!("{si:06}.bin")))?;
            anyhow::ensure!(raw.len() == 4 + HIDDEN * opts.seqlen * 2, "corrupt DFlash2 cache sample {si}");
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
            let mut sub = Vec::with_capacity(HIDDEN * context_vectors);
            for j in 0..context_vectors {
                let c = ((j + 1) * opts.seqlen / (context_vectors + 1)).min(opts.seqlen - 1);
                sub.extend_from_slice(&th[c * HIDDEN..(c + 1) * HIDDEN]);
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
                 N_LAYERS - 1, tl.elapsed().as_secs_f32()/60.0);
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
          "prefill": "GB10_W4A4_PREFILL=attn,mlp,gdn" },
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

/// Quantize only the dense LM head of an existing MR-GPTQ artifact. This deliberately reuses the
/// already-quantized trunk and calibrates on its real outputs, avoiding a second 64-layer GPTQ run.
pub fn lmhead(source: &Path, base: &Path, out: &Path, calib: &Path, opts: GptqOpts) -> Result<()> {
    validate_opts(&opts)?;
    prepare_output_dir(out, &[source, base])?;
    std::env::remove_var("GB10_W4A4_PREFILL");
    std::env::remove_var("GB10_W4A4_LMHEAD_NARROW");
    if std::env::var("GB10_PLE_OFFLOAD").is_err() { std::env::set_var("GB10_PLE_OFFLOAD", "ssd"); }
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
    if std::env::var("GB10_PLE_OFFLOAD").is_err() { std::env::set_var("GB10_PLE_OFFLOAD", "ssd"); }
    anyhow::ensure!(std::env::var_os("GB10_W4A4_PREFILL").is_none(),
                    "--calib-igs must collect the unquantized GEMM inputs; unset GB10_W4A4_PREFILL before calibration");
    anyhow::ensure!(std::env::var_os("GB10_W4A4_VERIFY").is_none(),
                    "--calib-igs must collect the unquantized GEMM inputs; unset GB10_W4A4_VERIFY before calibration");
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


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn igs_headroom_uses_block_distribution_not_literal_outlier() {
        let mut stats = IgsHistogram {
            bins: vec![0; IGS_HIST_BINS],
            running_max: 100.0,
            zero_blocks: 0,
            invalid_blocks: 0,
        };
        let common_bin = igs_hist_bin_index(1.0);
        stats.bins[common_bin] = 10_000;
        stats.bins[igs_hist_bin_index(100.0)] = 1;
        let diag = igs_scale_from_hist(&stats, IgsCalibConfig::default()).unwrap();
        let anchor = igs_hist_bin_center(common_bin);
        assert_eq!(diag.anchor, Some(anchor));
        assert!((diag.selected_amax - 16384.0 * anchor).abs() < diag.selected_amax * 1e-6);
        assert!(diag.has_headroom);
        assert!(!diag.range_exceeds_e4m3);
    }

    #[test]
    fn igs_max_mode_preserves_legacy_reciprocal_scale() {
        let stats = IgsHistogram {
            bins: vec![0; IGS_HIST_BINS],
            running_max: 42.0,
            zero_blocks: 3,
            invalid_blocks: 0,
        };
        let cfg = IgsCalibConfig { method: IgsMethod::Max, ..IgsCalibConfig::default() };
        let diag = igs_scale_from_hist(&stats, cfg).unwrap();
        assert_eq!(diag.selected_amax, 42.0);
        assert!((diag.input_global_scale - 2688.0 / 42.0).abs() < 1e-6);
    }

    #[test]
    fn igs_headroom_flags_ranges_no_single_e4m3_global_scale_can_cover() {
        let mut stats = IgsHistogram {
            bins: vec![0; IGS_HIST_BINS],
            running_max: 2.0f32.powi(20),
            zero_blocks: 0,
            invalid_blocks: 0,
        };
        // Keep the anchor above the deliberate upper/1e6 noise-floor cutoff while still
        // exceeding the E4M3 normal-range ratio.
        stats.bins[igs_hist_bin_index(2.0)] = 10_000;
        stats.bins[igs_hist_bin_index(2.0f32.powi(20))] = 10_000;
        let diag = igs_scale_from_hist(&stats, IgsCalibConfig::default()).unwrap();
        assert!(diag.range_exceeds_e4m3);
        assert!(!diag.has_headroom);
    }

    #[test]
    fn igs_rejects_non_finite_activation_blocks() {
        let stats = IgsHistogram {
            bins: vec![1; IGS_HIST_BINS],
            running_max: 1.0,
            zero_blocks: 0,
            invalid_blocks: 1,
        };
        assert!(igs_scale_from_hist(&stats, IgsCalibConfig::default()).is_none());
    }

    #[test]
    fn pretokenized_calibration_preserves_sample_boundaries() {
        let path = std::env::temp_dir().join(format!("gptq-pretokenized-test-{}.jsonl", std::process::id()));
        std::fs::write(&path,
            "{\"input_ids\":[1,2,3,4],\"text\":\"ignored\"}\n{\"input_ids\":[5,6,7,8],\"text\":\"ignored too\"}\n").unwrap();
        let samples = calib_tokens(Path::new("/tokenizer-is-deliberately-absent"), &path, 2, 4, 16).unwrap();
        assert_eq!(samples, vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn maca_accepts_variable_rows_but_fixed_loader_rejects_them() {
        let path =
            std::env::temp_dir().join(format!("gptq-maca-test-{}.jsonl", std::process::id()));
        std::fs::write(&path,
            "{\"input_ids\":[1,2],\"text\":\"short\"}\n{\"input_ids\":[3,4,5,6],\"text\":\"long\"}\n").unwrap();
        assert!(calib_tokens(
            Path::new("/tokenizer-is-deliberately-absent"),
            &path,
            2,
            4,
            16
        )
        .is_err());
        let samples = calib_tokens_mode(
            Path::new("/tokenizer-is-deliberately-absent"),
            &path,
            2,
            4,
            16,
            true,
        )
        .unwrap();
        assert_eq!(samples, vec![vec![1, 2], vec![3, 4, 5, 6]]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn dflash2_checkpoint_roundtrip_is_exact() {
        let path = std::env::temp_dir().join(format!("gptq-df2-checkpoint-{}.bin", std::process::id()));
        let r = Rec { qw: vec![1, 2, 3, 4], sc: vec![9, 8], m: 16, k: 32,
                      gs: 123.25, igs: Some(77.5) };
        write_df2_checkpoint(&path, &[("layers.0.self_attn.q_proj.weight".into(), &r)]).unwrap();
        let mut got = read_df2_checkpoint(&path).unwrap();
        assert_eq!(got.len(), 1);
        let (name, q) = got.pop().unwrap();
        assert_eq!(name, "layers.0.self_attn.q_proj.weight");
        assert_eq!((q.qw, q.sc, q.m, q.k, q.gs, q.igs),
                   (r.qw, r.sc, r.m, r.k, r.gs, r.igs));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn output_dir_must_be_distinct_and_empty() {
        let root = std::env::temp_dir().join(format!("gptq-output-test-{}", std::process::id()));
        let input = root.join("input");
        let out = root.join("out");
        std::fs::create_dir_all(&input).unwrap();
        let err = prepare_output_dir(&input, &[&input]).unwrap_err().to_string();
        assert!(err.contains("must differ"));
        std::fs::create_dir_all(&out).unwrap();
        std::fs::write(out.join("partial.bin"), b"x").unwrap();
        let err = prepare_output_dir(&out, &[&input]).unwrap_err().to_string();
        assert!(err.contains("not empty"));
        std::fs::remove_file(out.join("partial.bin")).unwrap();
        prepare_output_dir(&out, &[&input]).unwrap();
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Families pushed around forced shard boundaries (tiny shard size) must never be split.
    #[test]
    fn writer_keeps_packed_families_in_one_shard() {
        let dir = std::env::temp_dir().join(format!("gptq-writer-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut w = Writer::new(&dir, 1000);   // 1 KB shards
        for i in 0..7 {
            let stem = format!("layers.{i}.w");
            // a GPTQ triple (~700 B) …
            w.push_nvfp4(&stem, vec![1u8; 512], vec![2u8; 64], 16, 64, 1.0, None);
            // … a verbatim triple in index (name) order, as the base-artifact copy emits it
            let vs = format!("mtp.{i}.w");
            w.push(Out { name: format!("{vs}.weight_global_scale"), dtype: safetensors::Dtype::F32, shape: vec![1], data: vec![0u8; 4] });
            w.push(Out { name: format!("{vs}.weight_packed"), dtype: safetensors::Dtype::U8, shape: vec![16, 32], data: vec![3u8; 512] });
            w.push(Out { name: format!("{vs}.weight_scale"), dtype: safetensors::Dtype::F8_E4M3, shape: vec![16, 4], data: vec![4u8; 64] });
            // and a plain bf16 tensor
            w.push(Out { name: format!("norm.{i}.weight"), dtype: safetensors::Dtype::BF16, shape: vec![64], data: vec![0u8; 128] });
        }
        w.finish().expect("no split families");
        let idx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("model.safetensors.index.json")).unwrap()).unwrap();
        let wm = idx["weight_map"].as_object().unwrap();
        assert!(wm.len() == 7 * 7);
        let shards: std::collections::BTreeSet<&str> = wm.values().map(|v| v.as_str().unwrap()).collect();
        assert!(shards.len() > 5, "the tiny shard size must have produced many shards ({})", shards.len());
        std::fs::remove_dir_all(&dir).unwrap();
    }
    fn host_scale_stats(w: &[f32], s_tensor: f32, nclip: usize) -> (f64, f64, f64) {
        const RATIOS: [f32; 7] = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7];
        let mut out = (0.0, 0.0, 0.0);
        for x in w.chunks_exact(16) {
            let amax = x.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let mut best = (u8::default(), f64::INFINITY);
            for &ratio in RATIOS.iter().take(nclip) {
                let raw = if amax > 0.0 { amax * ratio / quant::E2M1_MAX / s_tensor } else { 0.0 };
                let scode = quant::f32_to_e4m3(raw);
                let scale = quant::e4m3_to_f32(scode) * s_tensor;
                let mse = x.iter().map(|&v| {
                    let q = if scale > 0.0 { quant::e2m1_to_f32(quant::f32_to_e2m1(v / scale)) * scale } else { 0.0 };
                    let d = (q - v) as f64;
                    d * d
                }).sum::<f64>();
                if mse < best.1 { best = (scode, mse); }
            }
            let es = quant::e4m3_to_f32(best.0);
            let scale = es * s_tensor;
            for &v in x {
                let code = if scale > 0.0 { quant::f32_to_e2m1(v / scale) } else { 0 };
                let z = es * quant::e2m1_to_f32(code);
                out.0 += v as f64 * z as f64;
                out.1 += z as f64 * z as f64;
            }
            out.2 += best.1;
        }
        out
    }

    fn host_hessian_loss(w: &[f32; 16], h: &[f32; 256], s_tensor: f32, code: u8) -> f64 {
        let scale = quant::e4m3_to_f32(code) * s_tensor;
        let mut dw = [0.0f64; 16];
        for i in 0..16 {
            let q = quant::e2m1_to_f32(quant::f32_to_e2m1(w[i] / scale)) * scale;
            dw[i] = (w[i] - q) as f64;
        }
        let mut loss = 0.0f64;
        for i in 0..16 {
            for j in 0..16 {
                loss += dw[i] * h[i * 16 + j] as f64 * dw[j];
            }
        }
        loss
    }

    fn host_local_hessian_code(w: &[f32; 16], h: &[f32; 256], s_tensor: f32) -> u8 {
        (1u8..=126).min_by(|&a, &b| {
            host_hessian_loss(w, h, s_tensor, a)
                .total_cmp(&host_hessian_loss(w, h, s_tensor, b))
                .then_with(|| a.cmp(&b))
        }).unwrap()
    }

    fn host_clip_code(w: &[f32; 16], s_tensor: f32) -> u8 {
        const RATIOS: [f32; 7] = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7];
        let amax = w.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        RATIOS.into_iter().map(|ratio| {
            quant::f32_to_e4m3(amax * ratio / quant::E2M1_MAX / s_tensor)
        }).min_by(|&a, &b| {
            let mse = |code| {
                let scale = quant::e4m3_to_f32(code) * s_tensor;
                w.iter().map(|&v| {
                    let q = quant::e2m1_to_f32(quant::f32_to_e2m1(v / scale)) * scale;
                    let d = (v - q) as f64;
                    d * d
                }).sum::<f64>()
            };
            mse(a).total_cmp(&mse(b)).then_with(|| a.cmp(&b))
        }).unwrap()
    }

    #[test]
    fn static_activation_order_is_descending_stable_and_finite_first() {
        let order = static_activation_order(&[2.0, 9.0, 9.0, f32::NAN, 1.0, f32::INFINITY]);
        assert_eq!(order, vec![1, 2, 0, 4, 3, 5]);
    }

    #[test]
    fn alternating_scale_fit_never_increases_nvfp4_error() {
        let w: Vec<f32> = (0..256).map(|i| {
            let x = i as f32 - 127.5;
            (x * 0.071).sin() * (0.2 + (i % 29) as f32 * 0.037) + x * 0.0009
        }).collect();
        let amax = w.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let initial = e4m3_scale_of(amax) * 1.8;
        let fit = alternating_scale_fit(initial, 6, |s| host_scale_stats(&w, s, 7));
        assert!(fit.final_mse <= fit.initial_mse);
        assert!(fit.accepted > 0, "synthetic bad initial scale should admit an improving step");
        assert!((fit.scale - initial).abs() > initial * 1e-5);
    }

    #[test]
    fn alternating_scale_fit_keeps_zero_tensor_scale() {
        let w = [0.0f32; 32];
        let fit = alternating_scale_fit(1.0, 4, |s| host_scale_stats(&w, s, 7));
        assert_eq!(fit.scale, 1.0);
        assert_eq!(fit.final_mse, 0.0);
        assert_eq!(fit.accepted, 0);
    }

    #[test]
    fn local_hessian_fp8_sweep_beats_clip_mse_on_its_objective() {
        let mut w = [0.0f32; 16];
        w[0] = 0.37;
        w[1] = -0.43;
        for (i, v) in w[2..].iter_mut().enumerate() {
            *v = if i & 1 == 0 { 4.0 + i as f32 * 0.14 } else { -4.3 - i as f32 * 0.11 };
        }
        let mut h = [0.0f32; 256];
        for i in 0..16 { h[i * 16 + i] = if i < 2 { 1.0e6 } else { 1.0 }; }
        let amax = w.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let s_tensor = e4m3_scale_of(amax);
        let local = host_local_hessian_code(&w, &h, s_tensor);
        let clip = host_clip_code(&w, s_tensor);
        let local_loss = host_hessian_loss(&w, &h, s_tensor, local);
        let clip_loss = host_hessian_loss(&w, &h, s_tensor, clip);
        assert_ne!(local, clip, "the weighted fixture must distinguish the two objectives");
        assert!(local_loss < clip_loss, "local={local_loss:e}, clip={clip_loss:e}");
    }
}
