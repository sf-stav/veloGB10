//! The vision tower on the GPU (kernels `inkv_*`, kernels/gpu_inkling2/b0_vit.cuh). GEMMs: cuBLAS bf16 x bf16 -> f32
//! (tensor cores, plain row-major weights; default) or the K1 projection `ink_proj_wide` on B32-packed weights
//! (`--inkling2-vision-gemm-k1`, the probe's comparison; about 2x slower at tower shapes, measured A/B under the same load). One image per call;
//! scratch grows to the largest image seen (bounded by the per-image token cap).

use super::preproc::PreImage;
use super::{rope_inv_freq, VisionCfg, VisionHost};
use crate::inkling2::gpu::k1::ProjW;
use crate::inkling2::gpu::{dp, Ink2Kernels, KArgs};
use anyhow::{ensure, Result};
use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaDevice, CudaSlice};
use half::f16;
use std::sync::Arc;

/// The entries of band b0 (loaded with the module; see `gpu::FNS_EXTRA`).
pub const FNS: &[&str] = &["inkv_patches", "inkv_embed_ln", "inkv_ln", "inkv_resid_ln", "inkv_gelu_bf", "inkv_cast_bf",
                           "inkv_qkv_rope", "inkv_attn", "inkv_rows_copy"];

const KP: usize = 1216;

/// One linear (no bias): `Y[M][ldy] = X[M][K] . W^T`, W `[N][K]` bf16.
struct Lin { packed: Option<ProjW>, plain: Option<CudaSlice<u16>>, n: usize, k: usize }
impl Lin {
    fn new(ks: &Ink2Kernels, segs: &[&[f32]], k: usize, blas: bool) -> Result<Lin> {
        let n = segs.iter().map(|s| s.len() / k).sum();
        if blas {
            let w: Vec<u16> = segs.iter().flat_map(|s| s.iter().map(|&x| crate::inkling2::gpu::k1::bf16_bits(x))).collect();
            Ok(Lin { packed: None, plain: Some(ks.dev.htod_sync_copy(&w)?), n, k })
        } else {
            Ok(Lin { packed: Some(ProjW::new(ks, segs, k)?), plain: None, n, k })
        }
    }
    fn bytes(&self) -> usize { self.n * self.k * 2 }
}

struct LayerDev {
    ln1_w: CudaSlice<f32>, ln1_b: CudaSlice<f32>, ln2_w: CudaSlice<f32>, ln2_b: CudaSlice<f32>,
    ls1: CudaSlice<f32>, ls2: CudaSlice<f32>,
    qkv: Lin, qkv_b: CudaSlice<f32>, o: Lin, o_b: CudaSlice<f32>,
    fc1: Lin, fc1_b: CudaSlice<f32>, fc2: Lin, fc2_b: CudaSlice<f32>,
}

#[derive(Default)]
struct Scratch {
    s_cap: usize,
    px: Option<CudaSlice<f32>>,
    xp: Option<CudaSlice<u16>>,
    h: Option<CudaSlice<f32>>,
    x: Option<CudaSlice<u16>>,
    y: Option<CudaSlice<f32>>,
    f: Option<CudaSlice<u16>>,
    q: Option<CudaSlice<f16>>,
    k: Option<CudaSlice<f16>>,
    vt: Option<CudaSlice<f16>>,
    o: Option<CudaSlice<u16>>,
}

pub struct VisionGpu {
    pub cfg: VisionCfg,
    dev: Arc<CudaDevice>,
    k: &'static Ink2Kernels,
    patch: Lin,
    blas: Option<CudaBlas>,
    pos: CudaSlice<f32>,
    cls: CudaSlice<f32>,
    ln_pre_w: CudaSlice<f32>,
    ln_pre_b: CudaSlice<f32>,
    inv_freq: CudaSlice<f32>,
    layers: Vec<LayerDev>,
    pin: Lin,
    pout: Lin,
    sc: Scratch,
    /// bytes of weights on the device
    pub weight_bytes: usize,
}

/// Optional per-block taps (downloaded to the host) for the probe.
#[derive(Default)]
pub struct GpuTaps { pub embed: Vec<f32>, pub ln_pre: Vec<f32>, pub layers: Vec<Vec<f32>> }

fn up(dev: &Arc<CudaDevice>, v: &[f32]) -> Result<CudaSlice<f32>> { Ok(dev.htod_sync_copy(v)?) }
fn pad_s(s: usize) -> usize { s.div_ceil(64) * 64 }

impl VisionGpu {
    /// `blas`: cuBLAS GEMMs (the default everywhere); false = the K1 projection kernel (comparison only).
    pub fn new(dev: &Arc<CudaDevice>, host: &VisionHost, blas: bool) -> Result<VisionGpu> {
        let k = Ink2Kernels::global(dev)?;
        let c = host.cfg.clone();
        let d = c.hidden;
        // patch weight [1024][1200] -> [W | 0 | W | 0] with K 2 x 1216 (hi / lo activation halves)
        let mut pw = vec![0f32; d * 2 * KP];
        for r in 0..d {
            for half in 0..2 { pw[r * 2 * KP + half * KP..r * 2 * KP + half * KP + 1200].copy_from_slice(&host.patch_w[r * 1200..(r + 1) * 1200]); }
        }
        let patch = Lin::new(k, &[&pw], 2 * KP, blas)?;
        let mut layers = Vec::new();
        let mut wb = patch.bytes();
        for l in &host.layers {
            let qkv = Lin::new(k, &[&l.q_w, &l.k_w, &l.v_w], d, blas)?;
            let qkv_b: Vec<f32> = [l.q_b.as_slice(), &l.k_b, &l.v_b].concat();
            let o = Lin::new(k, &[&l.o_w], d, blas)?;
            let fc1 = Lin::new(k, &[&l.fc1_w], d, blas)?;
            let fc2 = Lin::new(k, &[&l.fc2_w], c.inter, blas)?;
            wb += qkv.bytes() + o.bytes() + fc1.bytes() + fc2.bytes();
            layers.push(LayerDev {
                ln1_w: up(dev, &l.ln1_w)?, ln1_b: up(dev, &l.ln1_b)?, ln2_w: up(dev, &l.ln2_w)?, ln2_b: up(dev, &l.ln2_b)?,
                ls1: up(dev, &l.ls1)?, ls2: up(dev, &l.ls2)?,
                qkv, qkv_b: up(dev, &qkv_b)?, o, o_b: up(dev, &l.o_b)?, fc1, fc1_b: up(dev, &l.fc1_b)?, fc2, fc2_b: up(dev, &l.fc2_b)?,
            });
        }
        let pin = Lin::new(k, &[&host.proj_in], 4 * d, blas)?;
        let pout = Lin::new(k, &[&host.proj_out], c.proj_hidden, blas)?;
        wb += pin.bytes() + pout.bytes() + host.pos.len() * 4;
        Ok(VisionGpu {
            blas: if blas { Some(CudaBlas::new(dev.clone())?) } else { None },
            dev: dev.clone(), k, patch, pos: up(dev, &host.pos)?, cls: up(dev, &host.cls)?,
            ln_pre_w: up(dev, &host.ln_pre_w)?, ln_pre_b: up(dev, &host.ln_pre_b)?, inv_freq: up(dev, &rope_inv_freq(&c))?,
            layers, pin, pout, sc: Scratch::default(), weight_bytes: wb, cfg: c,
        })
    }

    /// Grow the scratch to images of `s` sequence rows (1 + 4 x tokens); the server sizes it for the cap at boot.
    pub fn ensure_scratch(&mut self, s: usize) -> Result<()> {
        if self.sc.s_cap >= s { return Ok(()); }
        let sp = pad_s(s);
        let (d, i) = (self.cfg.hidden, self.cfg.inter.max(self.cfg.proj_hidden));
        let t = (s - 1) / 4 + 1;
        let dev = &self.dev;
        self.sc = Scratch::default();
        self.sc.px = Some(dev.alloc_zeros::<f32>(t * 4800)?);
        self.sc.xp = Some(dev.alloc_zeros::<u16>(s * 2 * KP)?);
        self.sc.h = Some(dev.alloc_zeros::<f32>(s * d)?);
        self.sc.x = Some(dev.alloc_zeros::<u16>(s * d)?);
        self.sc.y = Some(dev.alloc_zeros::<f32>(s * i.max(3 * d))?);
        self.sc.f = Some(dev.alloc_zeros::<u16>(s * i)?);
        self.sc.q = Some(dev.alloc_zeros::<f16>(16 * sp * 64)?);
        self.sc.k = Some(dev.alloc_zeros::<f16>(16 * sp * 64)?);
        self.sc.vt = Some(dev.alloc_zeros::<f16>(16 * sp * 64)?);
        self.sc.o = Some(dev.alloc_zeros::<u16>(s * d)?);
        self.sc.s_cap = s;
        Ok(())
    }

    /// Scratch bytes held for an image of `tokens` LM tokens.
    pub fn scratch_bytes(&self, tokens: usize) -> usize {
        let s = 1 + 4 * tokens;
        let (d, i) = (self.cfg.hidden, self.cfg.inter.max(self.cfg.proj_hidden));
        tokens * 4800 * 4 + s * 2 * KP * 2 + s * d * 4 + s * d * 2 + s * i.max(3 * d) * 4 + s * i * 2 + 3 * 16 * pad_s(s) * 64 * 2 + s * d * 2
    }

    fn launch(&self, name: &str, grid: (u32, u32, u32), block: u32, a: &mut KArgs) -> Result<()> {
        self.k.launch(name, grid, (block, 1, 1), 0, a)
    }

    fn gemm(&self, w: &Lin, x: u64, y: u64, ldy: usize, m: usize) -> Result<()> {
        if let (Some(b), Some(wp)) = (&self.blas, &w.plain) {
            use cudarc::cublas::sys::{cublasComputeType_t, cublasGemmAlgo_t, cublasOperation_t as OP, cudaDataType};
            let (one, zero) = (1f32, 0f32);
            // column-major view: Y^T [N x M] = W [K x N]^T . X^T [K x M]
            unsafe {
                cudarc::cublas::result::gemm_ex(*b.handle(), OP::CUBLAS_OP_T, OP::CUBLAS_OP_N, w.n as i32, m as i32, w.k as i32,
                    &one as *const f32 as *const _, dp(wp) as *const _, cudaDataType::CUDA_R_16BF, w.k as i32,
                    x as *const _, cudaDataType::CUDA_R_16BF, w.k as i32, &zero as *const f32 as *const _,
                    y as *mut _, cudaDataType::CUDA_R_32F, ldy as i32, cublasComputeType_t::CUBLAS_COMPUTE_32F, cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT)
                    .map_err(|e| anyhow::anyhow!("vision cuBLAS gemm {}x{}x{}: {e:?}", m, w.n, w.k))
            }
        } else {
            let w = w.packed.as_ref().ok_or_else(|| anyhow::anyhow!("vision linear has no packed weight"))?;
            let nt = w.ntiles();
            self.k.proj(w, x, x, x, nt, nt, y, ldy, m, 0)
        }
    }

    /// One image -> `out` ([tokens][2304] f32, row r = LM token r of the image). `taps`: per-block host copies.
    pub fn forward(&mut self, pre: &PreImage, out: u64, mut taps: Option<&mut GpuTaps>) -> Result<()> {
        let g = pre.grid;
        let (s, t, d) = (g.seq(), g.tokens(), self.cfg.hidden);
        ensure!(pre.pixels.len() == t * 4800, "pixels {} vs {t} tiles", pre.pixels.len());
        self.ensure_scratch(s)?;
        let sp = pad_s(s);
        let eps = self.cfg.eps;
        {
            let px = self.sc.px.as_mut().unwrap();
            self.dev.htod_sync_copy_into(&pre.pixels, &mut px.slice_mut(0..t * 4800))?;
        }
        let sc = &self.sc;
        let (px, xp, h, x, y, f, q, kk, vt, o) = (dp(sc.px.as_ref().unwrap()), dp(sc.xp.as_ref().unwrap()), dp(sc.h.as_ref().unwrap()),
            dp(sc.x.as_ref().unwrap()), dp(sc.y.as_ref().unwrap()), dp(sc.f.as_ref().unwrap()), dp(sc.q.as_ref().unwrap()),
            dp(sc.k.as_ref().unwrap()), dp(sc.vt.as_ref().unwrap()), dp(sc.o.as_ref().unwrap()));
        let sg = (s as u32, 1, 1);
        self.launch("inkv_patches", sg, 256, KArgs::new().v(px).v(xp).v(g.cols as i32))?;
        self.gemm(&self.patch, xp, y, d, s)?;
        // embed tap: write E into o's bytes? use F (bf16 buffer is too small): stage through Y's tail when tapping
        let e_tap: Option<CudaSlice<f32>> = if taps.is_some() { Some(self.dev.alloc_zeros::<f32>(s * d)?) } else { None };
        let e_ptr = e_tap.as_ref().map_or(0, dp);
        self.launch("inkv_embed_ln", sg, 256, KArgs::new().v(y).v(dp(&self.pos)).v(dp(&self.cls)).v(dp(&self.ln_pre_w)).v(dp(&self.ln_pre_b))
            .v(e_ptr).v(h).v(g.rows as i32).v(g.cols as i32).v(self.cfg.grid as i32).v(eps))?;
        if let Some(tp) = taps.as_deref_mut() {
            tp.embed = self.dev.dtoh_sync_copy(e_tap.as_ref().unwrap())?;
            tp.ln_pre = self.dev.dtoh_sync_copy(sc.h.as_ref().unwrap())?; tp.ln_pre.truncate(s * d);
        }
        let nl = self.layers.len();
        let l0 = &self.layers[0];
        self.launch("inkv_ln", sg, 256, KArgs::new().v(h).v(dp(&l0.ln1_w)).v(dp(&l0.ln1_b)).v(x).v(eps))?;
        for li in 0..nl {
            let l = &self.layers[li];
            self.gemm(&l.qkv, x, y, 3 * d, s)?;
            self.launch("inkv_qkv_rope", ((sp / 32) as u32, 16, 1), 256,
                KArgs::new().v(y).v(dp(&l.qkv_b)).v(dp(&self.inv_freq)).v(q).v(kk).v(vt).v(s as i32).v(sp as i32).v(g.cols as i32))?;
            self.launch("inkv_attn", ((sp / 64) as u32, 16, 1), 128, KArgs::new().v(q).v(kk).v(vt).v(o).v(s as i32).v(sp as i32))?;
            self.gemm(&l.o, o, y, d, s)?;
            self.launch("inkv_resid_ln", sg, 256, KArgs::new().v(h).v(y).v(dp(&l.o_b)).v(dp(&l.ls1)).v(dp(&l.ln2_w)).v(dp(&l.ln2_b)).v(x).v(eps))?;
            self.gemm(&l.fc1, x, y, self.cfg.inter, s)?;
            let tot = (s * self.cfg.inter) as i64;
            self.launch("inkv_gelu_bf", ((tot as usize).div_ceil(1024) as u32, 1, 1), 256, KArgs::new().v(y).v(dp(&l.fc1_b)).v(f).v(tot).v(self.cfg.inter as i32))?;
            self.gemm(&l.fc2, f, y, d, s)?;
            let (nw, nb) = if li + 1 < nl { (dp(&self.layers[li + 1].ln1_w), dp(&self.layers[li + 1].ln1_b)) } else { (0, 0) };
            self.launch("inkv_resid_ln", sg, 256, KArgs::new().v(h).v(y).v(dp(&l.fc2_b)).v(dp(&l.ls2)).v(nw).v(nb).v(x).v(eps))?;
            if let Some(tp) = taps.as_deref_mut() { let mut v = self.dev.dtoh_sync_copy(sc.h.as_ref().unwrap())?; v.truncate(s * d); tp.layers.push(v); }
        }
        // merger: patch rows (CLS dropped) as [T][4096]
        let tot = (t * 4 * d) as i64;
        self.launch("inkv_cast_bf", ((tot as usize).div_ceil(1024) as u32, 1, 1), 256, KArgs::new().v(h + (d * 4) as u64).v(f).v(tot))?;
        self.gemm(&self.pin, f, y, self.cfg.proj_hidden, t)?;
        let tot2 = (t * self.cfg.proj_hidden) as i64;
        self.launch("inkv_gelu_bf", ((tot2 as usize).div_ceil(1024) as u32, 1, 1), 256, KArgs::new().v(y).v(0u64).v(x).v(tot2).v(self.cfg.proj_hidden as i32))?;
        self.gemm(&self.pout, x, out, self.cfg.text_hidden, t)?;
        Ok(())
    }
}

impl Ink2Kernels {
    /// `dst[0..n)` = `src[0..n)` (f32) on the module's stream (the image-feature splice).
    pub fn vis_rows_copy(&self, dst: u64, src: u64, n: usize) -> Result<()> {
        self.launch("inkv_rows_copy", (n.div_ceil(256) as u32, 1, 1), (256, 1, 1), 0, KArgs::new().v(dst).v(src).v(n as i64))
    }
}
