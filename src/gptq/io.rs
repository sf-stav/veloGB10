//! Low-level plumbing of the GPTQ pass: cuSOLVER dense Cholesky, the safetensors range reader (`ShardReader`)
//! and the output artifact writer. Siblings: `mod.rs` has the solvers that use these and the directory map.

use super::*;

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
        Ok((m.shape, crate::podread::pod_vec::<bf16>(&b)))
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

