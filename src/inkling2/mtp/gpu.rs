//! S6: the bundled drafter on the GPU (kernels/gpu_inkling2/a0_mtp.cuh) and the exact accept of one speculative round.
//!
//! Per lane: the drafter's context K / V cache (3 layers; layers 0 / 2 a 2,048-row ring for their 1,024-position
//! window, layer 1 linear up to the served context) and the Markov head's f_rnn row (bf16). Both are written at
//! COMMIT from the target's tap features (`commit_ctx`): prefill chunks and every verify's committed rows.
//!
//! One round for a lane at committed length P with anchor token x_P (sampled from the last target row):
//!   draft:  block rows [x_P, mask x 7] at positions P..P+7 through the 3 layers (non-causal over ctx ++ block),
//!           base logits of rows 1..7 through the target LM head, then 7 sequential Markov steps (folded head):
//!           step j's logits -> confidence (top probability at the request temperature) and the draft d_{j+1}
//!           (argmax for greedy, a Gumbel draw at the request temperature for sampled requests).
//!   verify: the target forward over [x_P, d_1..d_c] (c chosen by the policy, `policy.rs`) at positions P..P+c.
//!   accept: `accept` here (sequential exact rule on the device), then the emitted token after the accepted drafts.

use super::super::config::MtpConfig;
use super::super::gpu::k1::{bf16_bits, ProjW};
use super::super::gpu::{dp, Ink2Kernels, KArgs};
use super::super::model::{GpuModel, CHUNK, D};
use super::super::pack::Inkling2Pack;
use anyhow::{ensure, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, DeviceSlice};
use std::sync::Arc;

/// Block rows (anchor + mask rows) and the most drafts one round can verify.
pub const B: usize = 8;
pub const GMAX: usize = B - 1;
const NH: usize = 24;
const LSE_G: usize = 128;
const FOLD_G: usize = 384;
const SMP_G: usize = 128;
const SMAX: usize = 64;
const LOCAL_CAP: usize = 2048;
const INTER: usize = 9216;

/// Store f32 rows `x` [m][d] as bf16 tap `slot` of `taps` [m][4][d].
pub fn tap_store(k: &Ink2Kernels, x: u64, taps: u64, d: usize, slot: usize, m: usize) -> Result<()> {
    k.launch("ink6_tap_store", (m as u32, 1, 1), (256, 1, 1), 0, KArgs::new().v(x).v(taps).v(d as i32).v(slot as i32))
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MixArgs { w: [u64; 4], al: [i32; 4] }
unsafe impl DeviceRepr for MixArgs {}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CtxArgs { y: [u64; 3], kn: [u64; 3], kc: [u64; 3], vc: [u64; 3], cap_mask: [u32; 3] }
unsafe impl DeviceRepr for CtxArgs {}

struct Layer {
    local: bool,
    qkv: ProjW,
    kvctx: ProjW,
    o: ProjW,
    gu: ProjW,
    down: ProjW,
    in_norm: CudaSlice<f32>,
    post_norm: CudaSlice<f32>,
    q_norm: CudaSlice<f32>,
    k_norm: CudaSlice<f32>,
    k_norm_ctx: CudaSlice<f32>,
    dflare: CudaSlice<f32>,
}

pub struct MtpDev {
    pub cfg: MtpConfig,
    layers: Vec<Layer>,
    /// softmax(fuse_weight) rounded to bf16 [4][4] (row 3 = the Markov head)
    alpha: CudaSlice<f32>,
    pub alpha_host: Vec<f32>,
    norm: CudaSlice<f32>,
    mask: CudaSlice<f32>,
    mk_dflare: CudaSlice<f32>,
    w1: CudaSlice<u16>,
    rnn: ProjW,
    rnn_b: CudaSlice<f32>,
    fold_w: CudaSlice<u16>,
    fold_b: CudaSlice<f32>,
    /// vocabulary slice this rank proposes from (= this rank's LM head rows `[voff .. voff + n)`)
    pub n: usize,
    /// global vocabulary row offset of this rank (`0` at TP1, `g.rank.v0` at TP2/4/8)
    pub voff: usize,
    cap: f32,
    target_eps: f32,
    /// diagnostic (probe only): added to the block rows' RoPE positions
    pub pos_shift: i64,
}

pub struct MtpLane {
    kc: Vec<CudaSlice<u16>>,
    vc: Vec<CudaSlice<u16>>,
    cap: Vec<usize>,
    frnn: CudaSlice<u16>,
}

/// Prefix-cache snapshot of one drafter lane at a prompt end: the local layers' rings (whole) and f_rnn. The full
/// layer's rows [0, p) stay in place (linear by position; valid while the lane has not started another prompt, the
/// same rule as the target's full-attention slots, `model::prefix`).
pub struct MtpSnap {
    kv: Vec<Option<(CudaSlice<u16>, CudaSlice<u16>)>>,
    frnn: CudaSlice<u16>,
}

fn d2d(dst: u64, src: u64, bytes: usize) -> Result<()> {
    if bytes == 0 { return Ok(()); }
    let r = unsafe { cudarc::driver::sys::cuMemcpyDtoDAsync_v2(dst, src, bytes, std::ptr::null_mut()) };
    ensure!(r == cudarc::driver::sys::CUresult::CUDA_SUCCESS, "drafter prefix copy: {r:?}");
    Ok(())
}

pub struct MtpWork {
    mixb: CudaSlice<u16>,
    ycx: CudaSlice<f32>,
    h: CudaSlice<f32>,
    xb: CudaSlice<u16>,
    y: CudaSlice<f32>,
    qf: CudaSlice<f32>,
    bk: CudaSlice<u16>,
    bv: CudaSlice<u16>,
    pm: CudaSlice<f32>,
    pl: CudaSlice<f32>,
    pacc: CudaSlice<f32>,
    att: CudaSlice<u16>,
    ao: CudaSlice<f32>,
    act: CudaSlice<u16>,
    fin: CudaSlice<f32>,
    finb: CudaSlice<u16>,
    pub base: CudaSlice<f32>,
    pub step: CudaSlice<f32>,
    state: CudaSlice<u16>,
    latent: CudaSlice<u16>,
    rin: CudaSlice<u16>,
    ry: CudaSlice<f32>,
    pub tok: CudaSlice<i32>,
    seeds: CudaSlice<u64>,
    invt: CudaSlice<f32>,
    fpm: CudaSlice<f32>,
    fps: CudaSlice<f32>,
    gpm: CudaSlice<f32>,
    gpi: CudaSlice<i32>,
    pub conf: CudaSlice<f32>,
    lse_q: CudaSlice<f32>,
    lse_p: CudaSlice<f32>,
    lpm: CudaSlice<f32>,
    lps: CudaSlice<f32>,
    tgt: CudaSlice<i32>,
    res: CudaSlice<i32>,
    hpm: CudaSlice<f32>,
    hpi: CudaSlice<i32>,
}

/// What one draft produced (host copy).
#[derive(Clone, Debug)]
pub struct Draft {
    /// d_1..d_7
    pub toks: Vec<u32>,
    /// conf[j] = the top probability of the drafter's step-j distribution (the one d_{j+1} is drawn from), which
    /// depends on d_1..d_j only
    pub conf: Vec<f32>,
}

fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

impl MtpDev {
    /// Load `mtp.safetensors` (and fold the Markov head through the target's LM head on the GPU).
    pub fn load(g: &GpuModel, pack: &Inkling2Pack) -> Result<MtpDev> {
        let t0 = std::time::Instant::now();
        let cfg = g.cfg.mtp.clone().context("config.json has no mtp_config")?;
        let st = pack.mtp.as_ref().context("the pack has no mtp.safetensors")?;
        ensure!(cfg.hidden == D && cfg.layers == 3 && cfg.heads == NH && cfg.kv_heads == 8 && cfg.head_dim == 128 && cfg.inter == INTER
                && cfg.block_size == B && cfg.markov_rank == 256 && cfg.markov_head_type == "rnn_hidden" && cfg.target_layer_ids.len() == 4,
                "inkling2 drafter: unsupported mtp_config {cfg:?}");
        ensure!(*cfg.target_layer_ids.last().unwrap() == g.cfg.visits(), "drafter taps {:?}: the last must be the final-norm sentinel {}",
                cfg.target_layer_ids, g.cfg.visits());
        let k = g.k;
        let dev = &g.dev;
        let f = |n: &str| -> Result<Vec<f32>> { st.plain_f32(n) };
        let up = |v: &[f32]| -> Result<CudaSlice<f32>> { Ok(dev.htod_sync_copy(v)?) };
        let mut layers = Vec::new();
        for l in 0..3 {
            let p = format!("layers.{l}.");
            let a = format!("{p}self_attn.");
            layers.push(Layer {
                local: cfg.local_layer_ids.contains(&l),
                qkv: ProjW::new(k, &[&f(&format!("{a}q_proj.weight"))?, &f(&format!("{a}k_proj.weight"))?, &f(&format!("{a}v_proj.weight"))?], D)?,
                kvctx: ProjW::new(k, &[&f(&format!("{a}k_proj_ctx.weight"))?, &f(&format!("{a}v_proj_ctx.weight"))?], D)?,
                o: ProjW::new(k, &[&f(&format!("{a}o_proj.weight"))?], NH * 128)?,
                gu: ProjW::new(k, &[&f(&format!("{p}mlp.gate_proj.weight"))?, &f(&format!("{p}mlp.up_proj.weight"))?], D)?,
                down: ProjW::new(k, &[&f(&format!("{p}mlp.down_proj.weight"))?], INTER)?,
                in_norm: up(&f(&format!("{p}input_layernorm.weight"))?)?,
                post_norm: up(&f(&format!("{p}post_attention_layernorm.weight"))?)?,
                q_norm: up(&f(&format!("{a}q_norm.weight"))?)?,
                k_norm: up(&f(&format!("{a}k_norm.weight"))?)?,
                k_norm_ctx: up(&f(&format!("{a}k_norm_ctx.weight"))?)?,
                dflare: up(&f(&format!("{p}dflare_norm.weight"))?)?,
            });
        }
        let alpha_host = fuse_alpha(&f("fuse_weight")?);
        // Markov head
        let w1raw = st.raw("markov_head.markov_w1.weight")?;
        let w1: Vec<u16> = w1raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let rnn = ProjW::new(k, &[&f("markov_head.rnn_proj.weight")?], 2 * 256 + 2 * D)?;
        // fold: unembed[voff..voff+n] . [hidden_out.weight | bias] (GPU GEMM, fp32 accumulation of exact bf16 products)
        let (_, _, head) = g.shared_for_mtp();
        let n = head.n;
        let voff = g.rank.v0;
        let how = f("markov_head.hidden_out.weight")?; // [2304][256]
        let mut wt = vec![0f32; 256 * D];
        for d in 0..D { for r in 0..256 { wt[r * D + d] = how[d * 256 + r]; } }
        let hb = f("markov_head.hidden_out.bias")?;
        let fp = ProjW::new(k, &[&wt, &hb], D)?;
        let ub = pack.st.raw_range(
            &format!("{}unembed.weight", super::super::pack::P),
            (voff * D * 2) as u64,
            ((voff + n) * D * 2) as u64,
        )?;
        let ux: CudaSlice<u16> = dev.htod_sync_copy(&ub.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect::<Vec<u16>>())?;
        drop(ub);
        let ldy = fp.n_pad;
        let yf = dev.alloc_zeros::<f32>(n * ldy)?;
        let mut fold_w = dev.alloc_zeros::<u16>(n * 256)?;
        let mut fold_b = dev.alloc_zeros::<f32>(n)?;
        dev.memset_zeros(&mut fold_w)?;
        dev.memset_zeros(&mut fold_b)?;
        let mut off = 0;
        while off < n {
            let m = (n - off).min(65536);
            k.proj(&fp, dp(&ux) + (off * D * 2) as u64, 0, 0, 1 << 30, 1 << 30, dp(&yf) + (off * ldy * 4) as u64, ldy, m, 0)?;
            off += m;
        }
        k.launch("ink6_fold_pack", (n as u32, 1, 1), (256, 1, 1), 0, KArgs::new().v(dp(&yf)).v(ldy as i32).v(dp(&fold_w)).v(dp(&fold_b)).v(n as i32))?;
        k.sync()?;
        drop(yf);
        drop(ux);
        let m = MtpDev {
            layers, alpha: up(&alpha_host)?, alpha_host, norm: up(&f("norm.weight")?)?, mask: up(&f("mask_embed")?)?,
            mk_dflare: up(&f("markov_head.dflare_norm.weight")?)?, w1: dev.htod_sync_copy(&w1)?, rnn,
            rnn_b: up(&f("markov_head.rnn_proj.bias")?)?, fold_w, fold_b, n, voff, cap: g.cfg.logit_softcap, target_eps: g.cfg.eps, pos_shift: 0, cfg,
        };
        eprintln!("INK2_MTP loaded the drafter (3 layers, folded rnn_hidden head [{voff}..{}]x256) in {:.1}s", voff + n, t0.elapsed().as_secs_f64());
        Ok(m)
    }

    /// Target tap visits for the model's `Work::tap_visits` ([5, 23, 41, 48] on the stock pack).
    pub fn tap_visits(&self) -> [usize; 4] {
        let t = &self.cfg.target_layer_ids;
        [t[0], t[1], t[2], t[3]]
    }

    pub fn new_lane(&self, dev: &Arc<CudaDevice>, max_ctx: usize) -> Result<MtpLane> {
        let full = max_ctx.max(16).next_power_of_two();
        let (mut kc, mut vc, mut cap) = (Vec::new(), Vec::new(), Vec::new());
        for l in &self.layers {
            let c = if l.local { LOCAL_CAP.min(full.max(LOCAL_CAP)) } else { full.max(LOCAL_CAP) };
            let mut a = dev.alloc_zeros::<u16>(c * 1024)?;
            let mut b = dev.alloc_zeros::<u16>(c * 1024)?;
            dev.memset_zeros(&mut a)?;
            dev.memset_zeros(&mut b)?;
            kc.push(a); vc.push(b); cap.push(c);
        }
        let mut frnn = dev.alloc_zeros::<u16>(D)?;
        dev.memset_zeros(&mut frnn)?;
        Ok(MtpLane { kc, vc, cap, frnn })
    }

    pub fn new_snap(&self, dev: &Arc<CudaDevice>, lane: &MtpLane) -> Result<MtpSnap> {
        let mut kv = Vec::new();
        for (l, ly) in self.layers.iter().enumerate() {
            kv.push(if ly.local { Some((dev.alloc_zeros::<u16>(lane.kc[l].len())?, dev.alloc_zeros::<u16>(lane.vc[l].len())?)) } else { None });
        }
        Ok(MtpSnap { kv, frnn: dev.alloc_zeros::<u16>(D)? })
    }
    /// Copy the lane's prompt-end state into `s` (`to_snap`) or back (stream-ordered, NULL stream).
    pub fn snap_copy(&self, lane: &MtpLane, s: &MtpSnap, to_snap: bool) -> Result<()> {
        let one = |a: u64, b: u64, n: usize| if to_snap { d2d(b, a, n * 2) } else { d2d(a, b, n * 2) };
        for (l, x) in s.kv.iter().enumerate() {
            if let Some((k, v)) = x {
                one(dp(&lane.kc[l]), dp(k), lane.kc[l].len())?;
                one(dp(&lane.vc[l]), dp(v), lane.vc[l].len())?;
            }
        }
        one(dp(&lane.frnn), dp(&s.frnn), D)
    }

    pub fn new_work(&self, dev: &Arc<CudaDevice>) -> Result<MtpWork> {
        let f = |n: usize| -> Result<CudaSlice<f32>> { let mut x = dev.alloc_zeros::<f32>(n)?; dev.memset_zeros(&mut x)?; Ok(x) };
        let h = |n: usize| -> Result<CudaSlice<u16>> { let mut x = dev.alloc_zeros::<u16>(n)?; dev.memset_zeros(&mut x)?; Ok(x) };
        let i = |n: usize| -> Result<CudaSlice<i32>> { Ok(dev.htod_sync_copy(&vec![0i32; n])?) };
        let n = self.n;
        let hg = n.div_ceil(128);
        Ok(MtpWork {
            mixb: h(3 * CHUNK * D)?, ycx: f(3 * CHUNK * 2048)?, h: f(B * D)?, xb: h(B * D)?, y: f(B * 2 * INTER)?,
            qf: f(B * NH * 128)?, bk: h(B * 1024)?, bv: h(B * 1024)?, pm: f(SMAX * B * NH)?, pl: f(SMAX * B * NH)?,
            pacc: f(SMAX * B * NH * 128)?, att: h(B * NH * 128)?, ao: f(B * D)?, act: h(B * INTER)?, fin: f(B * D)?, finb: h(B * D)?,
            base: f(GMAX * n)?, step: f(GMAX * n)?, state: h(256)?, latent: h(256)?, rin: h(2 * 256 + 2 * D)?, ry: f(16 * self.rnn.n_pad)?,
            tok: i(B)?, seeds: dev.htod_sync_copy(&vec![0u64; 16])?, invt: f(16)?, fpm: f(FOLD_G)?, fps: f(FOLD_G)?,
            gpm: f(16 * SMP_G)?, gpi: i(16 * SMP_G)?, conf: f(B)?, lse_q: f(B)?, lse_p: f(16)?, lpm: f(16 * LSE_G)?, lps: f(16 * LSE_G)?,
            tgt: i(16)?, res: i(4)?, hpm: f(16 * hg)?, hpi: i(16 * hg)?,
        })
    }

    /// Commit `rows` positions p0..p0+rows of `lane` into the drafter: context K / V of every layer from the target's
    /// taps (bf16 [rows][4][D] at `taps`), and f_rnn from the last row.
    pub fn commit_ctx(&self, k: &Ink2Kernels, w: &mut MtpWork, lane: &MtpLane, taps: u64, rows: usize, p0: usize) -> Result<()> {
        ensure!(rows >= 1 && rows <= CHUNK, "drafter commit of {rows} rows");
        let eps = self.cfg.eps;
        let ldo = (CHUNK * D) as i64;
        let mut ma = MixArgs::default();
        for l in 0..3 { ma.w[l] = dp(&self.layers[l].dflare); ma.al[l] = l as i32; }
        k.launch("ink6_mix_norm", (rows as u32, 3, 1), (256, 1, 1), 0,
                 KArgs::new().v(taps).v(dp(&self.alpha)).v(ma).v(dp(&w.mixb)).v(0u64).v(ldo).v(D as i32).v(eps))?;
        let mut ca = CtxArgs::default();
        for l in 0..3 {
            let ly = &self.layers[l];
            let y = dp(&w.ycx) + (l * CHUNK * 2048 * 4) as u64;
            let x = dp(&w.mixb) + (l * CHUNK * D * 2) as u64;
            k.proj(&ly.kvctx, x, x, x, 1 << 30, 1 << 30, y, 2048, rows, 0)?;
            ca.y[l] = y; ca.kn[l] = dp(&ly.k_norm_ctx); ca.kc[l] = dp(&lane.kc[l]); ca.vc[l] = dp(&lane.vc[l]);
            ca.cap_mask[l] = (lane.cap[l] - 1) as u32;
        }
        k.launch("ink6_ctx_kv", (rows as u32, 3, 1), (256, 1, 1), 0,
                 KArgs::new().v(ca).v(2048i32).v(p0 as i32).v(eps).v(self.cfg.rope_theta as f32))?;
        let mut mf = MixArgs::default();
        mf.w[0] = dp(&self.mk_dflare); mf.al[0] = 3;
        k.launch("ink6_mix_norm", (1, 1, 1), (256, 1, 1), 0,
                 KArgs::new().v(taps + ((rows - 1) * 4 * D * 2) as u64).v(dp(&self.alpha)).v(mf).v(dp(&lane.frnn)).v(0u64).v(0i64).v(D as i32).v(eps))
    }

    /// Recompute the lane's f_rnn from one tap row (bf16 [4][D] at `tap_row`).
    pub fn set_frnn(&self, k: &Ink2Kernels, lane: &MtpLane, tap_row: u64) -> Result<()> {
        let mut mf = MixArgs::default();
        mf.w[0] = dp(&self.mk_dflare); mf.al[0] = 3;
        k.launch("ink6_mix_norm", (1, 1, 1), (256, 1, 1), 0,
                 KArgs::new().v(tap_row).v(dp(&self.alpha)).v(mf).v(dp(&lane.frnn)).v(0u64).v(0i64).v(D as i32).v(self.cfg.eps))
    }

    /// Run the drafter for `lane` (committed length `p`, anchor token at position p): fills `w.tok[1..8]`, the step
    /// logits `w.step` and `w.conf`; `t` = the request temperature (<= 0: greedy drafts); returns the host copy.
    pub fn draft(&self, g: &GpuModel, w: &mut MtpWork, lane: &MtpLane, anchor: u32, p: usize, t: f32, seed: u64) -> Result<Draft> {
        self.draft_until(g, w, lane, anchor, p, t, seed, GMAX, &mut |_, _| true)
    }

    /// `draft` with an early stop: after Markov step j the host reads conf[j] and calls `more(j, &conf[..=j])`; false
    /// stops (steps j + 1.. are not run). `more` must decide only from what it is given (the exactness contract of
    /// `policy.rs`: conf[j] depends on d_1..d_j, never on d_{j+1}). The returned draft has the steps that ran.
    #[allow(clippy::too_many_arguments)]
    pub fn draft_until(&self, g: &GpuModel, w: &mut MtpWork, lane: &MtpLane, anchor: u32, p: usize, t: f32, seed: u64, max_steps: usize,
                       more: &mut dyn FnMut(usize, &[f32]) -> bool) -> Result<Draft> {
        let max_steps = max_steps.clamp(1, GMAX);
        let k = g.k;
        let dev = &g.dev;
        let eps = self.cfg.eps;
        let theta = self.cfg.rope_theta as f32;
        let n = self.n;
        let it = if t > 0.0 { 1.0 / t } else { 0.0 };
        let mut tk = [0i32; B];
        tk[0] = anchor as i32;
        k.sync()?;
        dev.htod_sync_copy_into(&tk, &mut w.tok)?;
        let seeds: Vec<u64> = (0..16).map(|j| mix64(seed ^ mix64(0xd5a7_0000 + j as u64))).collect();
        dev.htod_sync_copy_into(&seeds, &mut w.seeds)?;
        dev.htod_sync_copy_into(&[it; 16], &mut w.invt)?;
        let (emb, embn, head) = g.shared_for_mtp();
        k.launch("ink_embed_norm", (1, 1, 1), (256, 1, 1), 0, KArgs::new().v(dp(&w.tok)).v(emb).v(embn).v(dp(&w.h)).v(D as i32).v(self.target_eps))?;
        k.launch("ink6_block_init", ((B - 1) as u32, 1, 1), (256, 1, 1), 0, KArgs::new().v(dp(&self.mask)).v(dp(&w.h)).v(D as i32))?;
        for (l, ly) in self.layers.iter().enumerate() {
            let delta = if l == 0 { 0u64 } else { dp(&w.ao) };
            k.launch("ink6_add_norm", (B as u32, 1, 1), (256, 1, 1), 0,
                     KArgs::new().v(dp(&w.h)).v(delta).v(dp(&ly.in_norm)).v(dp(&w.xb)).v(0u64).v(D as i32).v(eps))?;
            k.proj(&ly.qkv, dp(&w.xb), dp(&w.xb), dp(&w.xb), 1 << 30, 1 << 30, dp(&w.y), ly.qkv.n_pad, B, 0)?;
            k.launch("ink6_qk_rope", (B as u32, 1, 1), (256, 1, 1), 0,
                     KArgs::new().v(dp(&w.y)).v(ly.qkv.n_pad as i32).v(dp(&ly.q_norm)).v(dp(&ly.k_norm)).v(dp(&w.qf)).v(dp(&w.bk)).v(dp(&w.bv))
                         .v((p as i64 + self.pos_shift) as i32).v(eps).v(theta))?;
            let lo = if ly.local { p.saturating_sub(self.cfg.sliding_window) } else { 0 };
            let ntot = p - lo + B;
            let per = (ntot.div_ceil(SMAX).max(256)).div_ceil(32) * 32;
            let s = ntot.div_ceil(per);
            k.launch("ink6_dattn", (8, s as u32, 1), (256, 1, 1), 0,
                     KArgs::new().v(dp(&w.qf)).v(dp(&lane.kc[l])).v(dp(&lane.vc[l])).v((lane.cap[l] - 1) as u32).v(dp(&w.bk)).v(dp(&w.bv))
                         .v(lo as i32).v(p as i32).v(per as i32).v(dp(&w.pm)).v(dp(&w.pl)).v(dp(&w.pacc)))?;
            k.launch("ink6_dattn_comb", (NH as u32, B as u32, 1), (128, 1, 1), 0,
                     KArgs::new().v(dp(&w.pm)).v(dp(&w.pl)).v(dp(&w.pacc)).v(s as i32).v(dp(&w.att)))?;
            k.proj(&ly.o, dp(&w.att), dp(&w.att), dp(&w.att), 1 << 30, 1 << 30, dp(&w.ao), D, B, 0)?;
            k.launch("ink6_add_norm", (B as u32, 1, 1), (256, 1, 1), 0,
                     KArgs::new().v(dp(&w.h)).v(dp(&w.ao)).v(dp(&ly.post_norm)).v(dp(&w.xb)).v(0u64).v(D as i32).v(eps))?;
            k.proj(&ly.gu, dp(&w.xb), dp(&w.xb), dp(&w.xb), 1 << 30, 1 << 30, dp(&w.y), ly.gu.n_pad, B, 0)?;
            k.launch("ink6_silu_mul", ((INTER / 256) as u32, B as u32, 1), (256, 1, 1), 0,
                     KArgs::new().v(dp(&w.y)).v(ly.gu.n_pad as i32).v(INTER as i32).v(dp(&w.act)))?;
            k.proj(&ly.down, dp(&w.act), dp(&w.act), dp(&w.act), 1 << 30, 1 << 30, dp(&w.ao), D, B, 0)?;
        }
        k.launch("ink6_add_norm", (B as u32, 1, 1), (256, 1, 1), 0,
                 KArgs::new().v(dp(&w.h)).v(dp(&w.ao)).v(dp(&self.norm)).v(dp(&w.finb)).v(dp(&w.fin)).v(D as i32).v(eps))?;
        // base logits of rows 1..7 (no softcap: the cap is applied after the Markov bias)
        k.launch("ink2_lm_head", (head.grid(), 1, 1), (256, 1, 1), 0,
                 KArgs::new().v(dp(&head.w)).v(dp(&w.fin) + (D * 4) as u64).v(dp(&w.base)).v(dp(&w.hpm)).v(dp(&w.hpi)).v(GMAX as i32)
                     .v(head.k as i32).v(n as i32).v(n as i32).v(0i32).v(1.0e30f32))?;
        dev.memset_zeros(&mut w.state)?;
        let itc = if t > 0.0 { it } else { 1.0 };
        let mut steps = 0;
        let mut conf_h: Vec<f32> = Vec::with_capacity(GMAX);
        for j in 0..max_steps {
            k.launch("ink6_rnn_in", (20, 1, 1), (256, 1, 1), 0,
                     KArgs::new().v(dp(&w.state)).v(dp(&w.tok)).v(j as i32).v(dp(&self.w1)).v(dp(&w.finb) + ((j + 1) * D * 2) as u64)
                         .v(dp(&lane.frnn)).v(dp(&w.rin)).v(256i32).v(D as i32))?;
            k.proj(&self.rnn, dp(&w.rin), dp(&w.rin), dp(&w.rin), 1 << 30, 1 << 30, dp(&w.ry), self.rnn.n_pad, 1, 0)?;
            k.launch("ink6_rnn_gates", (1, 1, 1), (256, 1, 1), 0,
                     KArgs::new().v(dp(&w.ry)).v(dp(&self.rnn_b)).v(dp(&w.state)).v(dp(&w.latent)).v(256i32))?;
            let base = dp(&w.base) + (j * n * 4) as u64;
            let step = dp(&w.step) + (j * n * 4) as u64;
            k.launch("ink6_fold", (FOLD_G as u32, 1, 1), (256, 1, 1), 0,
                     KArgs::new().v(dp(&self.fold_w)).v(dp(&self.fold_b)).v(base).v(dp(&w.latent)).v(step).v(n as i32).v(self.cap).v(itc)
                         .v(dp(&w.fpm)).v(dp(&w.fps)))?;
            k.launch("ink6_lse_fin", (1, 1, 1), (32, 1, 1), 0,
                     KArgs::new().v(dp(&w.fpm)).v(dp(&w.fps)).v(FOLD_G as i32).v(dp(&w.lse_q) + (j * 4) as u64).v(dp(&w.conf) + (j * 4) as u64))?;
            k.launch("ink3_gumbel_part", (SMP_G as u32, 1, 1), (256, 1, 1), 0,
                     KArgs::new().v(step).v(n as i32).v(n as i32).v(dp(&w.seeds) + (j * 8) as u64).v(dp(&w.invt)).v(dp(&w.gpm)).v(dp(&w.gpi)))?;
            k.launch("ink3_gumbel_pick", (1, 1, 1), (256, 1, 1), 0,
                     KArgs::new().v(dp(&w.gpm)).v(dp(&w.gpi)).v(SMP_G as i32).v(dp(&w.tok) + ((j + 1) * 4) as u64))?;
            steps = j + 1;
            if j + 1 < max_steps {
                let mut cj = [0f32; 1];
                k.sync()?;
                dev.dtoh_sync_copy_into(&w.conf.slice(j..j + 1), &mut cj)?;
                conf_h.push(cj[0]);
                if !more(j, &conf_h) { break; }
            }
        }
        let mut tok = [0i32; B];
        k.sync()?;
        dev.dtoh_sync_copy_into(&w.tok, &mut tok)?;
        let mut conf = [0f32; B];
        dev.dtoh_sync_copy_into(&w.conf, &mut conf)?;
        Ok(Draft { toks: tok[1..=steps].iter().map(|&x| x as u32).collect(), conf: conf[..steps].to_vec() })
    }

    /// Exact accept of `c` drafts (`w.tok[1..=c]`, drawn by the last `draft` at temperature `t`) against the target
    /// logits rows `tl` [c + 1][n] (softcapped, the verify rows of [anchor, d_1..d_c]). Returns (accepted a, the
    /// emitted token after them): greedy = the target argmax of row a; sampled = a draw from (p_a - q_a)+ (a < c) or
    /// from p_c (a == c).
    pub fn accept(&self, g: &GpuModel, w: &mut MtpWork, tl: u64, c: usize, t: f32, seed: u64) -> Result<(usize, u32)> {
        let k = g.k;
        let n = self.n;
        let it = if t > 0.0 { 1.0 / t } else { 0.0 };
        k.sync()?;
        g.dev.htod_sync_copy_into(&[it; 16], &mut w.invt)?;
        let sa = mix64(seed ^ 0xacce_97ed);
        let sf = mix64(seed ^ 0xf1a1_70c3);
        if t <= 0.0 {
            // the target argmax of every row (the same kernels as the plain greedy path: T = 0 Gumbel = argmax)
            k.launch("ink3_gumbel_part", (SMP_G as u32, (c + 1) as u32, 1), (256, 1, 1), 0,
                     KArgs::new().v(tl).v(n as i32).v(n as i32).v(dp(&w.seeds)).v(dp(&w.invt)).v(dp(&w.gpm)).v(dp(&w.gpi)))?;
            k.launch("ink3_gumbel_pick", ((c + 1) as u32, 1, 1), (256, 1, 1), 0,
                     KArgs::new().v(dp(&w.gpm)).v(dp(&w.gpi)).v(SMP_G as i32).v(dp(&w.tgt)))?;
            k.launch("ink6_accept", (1, 1, 1), (32, 1, 1), 0,
                     KArgs::new().v(dp(&w.tok)).v(c as i32).v(dp(&w.tgt)).v(tl).v(n as i32).v(dp(&w.step)).v(n as i32).v(dp(&w.lse_p))
                         .v(dp(&w.lse_q)).v(0f32).v(sa).v(dp(&w.res)))?;
            let mut r = [0i32; 4];
            k.sync()?;
            g.dev.dtoh_sync_copy_into(&w.res, &mut r)?;
            let a = r[0] as usize;
            let mut tg = [0i32; 16];
            g.dev.dtoh_sync_copy_into(&w.tgt, &mut tg)?;
            return Ok((a, tg[a] as u32));
        }
        k.launch("ink6_lse", (LSE_G as u32, (c + 1) as u32, 1), (256, 1, 1), 0,
                 KArgs::new().v(tl).v(n as i32).v(n as i32).v(dp(&w.invt)).v(dp(&w.lpm)).v(dp(&w.lps)))?;
        k.launch("ink6_lse_fin", ((c + 1) as u32, 1, 1), (32, 1, 1), 0,
                 KArgs::new().v(dp(&w.lpm)).v(dp(&w.lps)).v(LSE_G as i32).v(dp(&w.lse_p)).v(0u64))?;
        k.launch("ink6_accept", (1, 1, 1), (32, 1, 1), 0,
                 KArgs::new().v(dp(&w.tok)).v(c as i32).v(dp(&w.tgt)).v(tl).v(n as i32).v(dp(&w.step)).v(n as i32).v(dp(&w.lse_p))
                     .v(dp(&w.lse_q)).v(it).v(sa).v(dp(&w.res)))?;
        k.launch("ink6_final_part", (SMP_G as u32, 1, 1), (256, 1, 1), 0,
                 KArgs::new().v(tl).v(n as i32).v(dp(&w.step)).v(n as i32).v(n as i32).v(dp(&w.lse_p)).v(dp(&w.lse_q)).v(it).v(dp(&w.res))
                     .v(c as i32).v(sf).v(dp(&w.gpm)).v(dp(&w.gpi)))?;
        k.launch("ink3_gumbel_pick", (1, 1, 1), (256, 1, 1), 0,
                 KArgs::new().v(dp(&w.gpm)).v(dp(&w.gpi)).v(SMP_G as i32).v(dp(&w.res) + 4))?;
        let mut r = [0i32; 4];
        k.sync()?;
        g.dev.dtoh_sync_copy_into(&w.res, &mut r)?;
        Ok((r[0] as usize, r[1] as u32))
    }

    /// Drop 1 (#1): vocab-parallel exact accept on `tl` (`[c + 1][self.n]` — this rank's slice of `ws.logits`).
    /// Zero full-vocabulary logits gather: exchanges only `<= 1 KiB` per lane (greedy or sampled `t > 0`) over
    /// `velo-fabric`. At TP1 delegates directly to `accept`.
    pub fn accept_tp(&self, g: &GpuModel, w: &mut MtpWork, tl: u64, c: usize, t: f32, seed: u64) -> Result<(usize, u32)> {
        let Some(tp) = &g.tp else { return self.accept(g, w, tl, c, t, seed); };
        use super::super::tp::kern as tpk;
        let k = g.k;
        let vn = self.n;
        let voff = self.voff;
        let (rank, world) = (g.rank.rank, g.rank.world);
        let it = if t > 0.0 { 1.0 / t } else { 0.0 };
        k.sync()?;
        g.dev.htod_sync_copy_into(&[it; 16], &mut w.invt)?;
        let sa = mix64(seed ^ 0xacce_97ed);
        let sf = mix64(seed ^ 0xf1a1_70c3);
        if t <= 0.0 {
            tpk::gumbel_rec(k, tl, vn, vn, voff, dp(&w.seeds), dp(&w.invt), dp(&w.gpm), dp(&w.gpi),
                            SMP_G, c + 1, dp(&tp.rec), rank, world)?;
            tp.x.all_reduce(&g.dev, dp(&tp.rec), (c + 1) * world * 2)?;
            tpk::rank_pick(k, dp(&tp.rec), world, c + 1, dp(&w.tgt))?;
            k.launch("ink6_accept", (1, 1, 1), (32, 1, 1), 0,
                     KArgs::new().v(dp(&w.tok)).v(c as i32).v(dp(&w.tgt)).v(tl).v(vn as i32).v(dp(&w.step)).v(vn as i32).v(dp(&w.lse_p))
                         .v(dp(&w.lse_q)).v(0f32).v(sa).v(dp(&w.res)))?;
            let mut r = [0i32; 4];
            k.sync()?;
            g.dev.dtoh_sync_copy_into(&w.res, &mut r)?;
            tp.x.check("after vocab-parallel greedy accept")?;
            let a = r[0] as usize;
            let mut tg = [0i32; 16];
            g.dev.dtoh_sync_copy_into(&w.tgt, &mut tg)?;
            return Ok((a, tg[a] as u32));
        }
        // Sampled (t > 0): Stage 1 = exact global lse_p[0..=c] + accept decision (~1 KiB exchange)
        k.launch("ink6_lse", (LSE_G as u32, (c + 1) as u32, 1), (256, 1, 1), 0,
                 KArgs::new().v(tl).v(vn as i32).v(vn as i32).v(dp(&w.invt)).v(dp(&w.lpm)).v(dp(&w.lps)))?;
        tpk::accept_pack(k, dp(&w.lpm), dp(&w.lps), LSE_G, dp(&w.tok), c,
                         tl, vn, dp(&w.step), vn, voff, vn, dp(&tp.rec), rank, world)?;
        tp.x.all_reduce(&g.dev, dp(&tp.rec), (c + 1) * world * 4)?;
        tpk::accept_decide(k, dp(&tp.rec), world, c, dp(&w.lse_q), dp(&w.lse_p), it, sa, dp(&w.res))?;
        // Stage 2 = vocab-parallel final draw from (p_a - q_a)+ or p_c (<= 64 B exchange)
        tpk::final_part(k, tl, vn, dp(&w.step), vn, vn, voff, dp(&w.lse_p), dp(&w.lse_q),
                        it, dp(&w.res), c, sf, SMP_G, dp(&w.gpm), dp(&w.gpi))?;
        k.launch("ink_tp_gumbel_rec", (1, 1, 1), (256, 1, 1), 0,
                 KArgs::new().v(dp(&w.gpm)).v(dp(&w.gpi)).v(SMP_G as i32).v(dp(&tp.rec)).v(rank as i32).v(world as i32))?;
        tp.x.all_reduce(&g.dev, dp(&tp.rec), world * 2)?;
        tpk::rank_pick(k, dp(&tp.rec), world, 1, dp(&w.res) + 4)?;
        let mut r = [0i32; 4];
        k.sync()?;
        g.dev.dtoh_sync_copy_into(&w.res, &mut r)?;
        tp.x.check("after vocab-parallel sampled accept")?;
        Ok((r[0] as usize, r[1] as u32))
    }

    /// Host copies of the drafter's last block (for the oracle probe): final hidden rows [8][D] and step logits [7][n].
    pub fn dump(&self, g: &GpuModel, w: &MtpWork) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        g.k.sync()?;
        let fin = g.dev.dtoh_sync_copy(&w.fin)?;
        let base = g.dev.dtoh_sync_copy(&w.base)?;
        let step = g.dev.dtoh_sync_copy(&w.step)?;
        Ok((fin, base, step))
    }
}

/// softmax(fuse_weight) per row in fp32, rounded to bf16 (the reference: `softmax(w.float()).to(bf16)`).
pub fn fuse_alpha(fw: &[f32]) -> Vec<f32> {
    let mut a = vec![0f32; 16];
    for r in 0..4 {
        let row = &fw[r * 4..r * 4 + 4];
        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = row.iter().map(|&x| (x - mx).exp()).collect();
        let s: f32 = e.iter().sum();
        for t in 0..4 { a[r * 4 + t] = super::super::gpu::k1::bf16_to_f32(super::super::gpu::k1::f32_to_bf16(e[t] / s)); }
    }
    let _ = bf16_bits;
    a
}

/// Scratch of a multi-lane draft (up to 4 lanes x 8 block rows run the backbone and the base head together).
pub struct MtpBatch {
    h: CudaSlice<f32>,
    xb: CudaSlice<u16>,
    y: CudaSlice<f32>,
    qf: CudaSlice<f32>,
    bk: CudaSlice<u16>,
    bv: CudaSlice<u16>,
    pm: CudaSlice<f32>,
    pl: CudaSlice<f32>,
    pacc: CudaSlice<f32>,
    att: CudaSlice<u16>,
    ao: CudaSlice<f32>,
    act: CudaSlice<u16>,
    fin: CudaSlice<f32>,
    finb: CudaSlice<u16>,
    base: CudaSlice<f32>,
    state: CudaSlice<u16>,
    latent: CudaSlice<u16>,
    rin: CudaSlice<u16>,
    ry: CudaSlice<f32>,
}

/// Most lanes one multi-lane draft carries.
pub const LMAX: usize = 4;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct FoldArgs { base: [u64; 4], out: [u64; 4], it: [f32; 4], pmax: [u64; 4], psum: [u64; 4] }
unsafe impl DeviceRepr for FoldArgs {}

/// One lane of a multi-lane draft.
#[derive(Clone, Copy, Debug)]
pub struct DraftReq { pub lane: usize, pub anchor: u32, pub p: usize, pub t: f32, pub seed: u64, pub cap: usize }

impl MtpDev {
    pub fn new_batch(&self, dev: &Arc<CudaDevice>) -> Result<MtpBatch> {
        let f = |n: usize| -> Result<CudaSlice<f32>> { let mut x = dev.alloc_zeros::<f32>(n)?; dev.memset_zeros(&mut x)?; Ok(x) };
        let h = |n: usize| -> Result<CudaSlice<u16>> { let mut x = dev.alloc_zeros::<u16>(n)?; dev.memset_zeros(&mut x)?; Ok(x) };
        let r = LMAX * B;
        Ok(MtpBatch {
            h: f(r * D)?, xb: h(r * D)?, y: f(r * 2 * INTER)?, qf: f(r * NH * 128)?, bk: h(r * 1024)?, bv: h(r * 1024)?,
            pm: f(SMAX * B * NH)?, pl: f(SMAX * B * NH)?, pacc: f(SMAX * B * NH * 128)?, att: h(r * NH * 128)?, ao: f(r * D)?,
            act: h(r * INTER)?, fin: f(r * D)?, finb: h(r * D)?, base: f(r * self.n)?, state: h(LMAX * 256)?, latent: h(LMAX * 256)?,
            rin: h(LMAX * (2 * 256 + 2 * D))?, ry: f(16 * self.rnn.n_pad)?,
        })
    }

    /// Draft for up to 4 lanes at once (TP1 or TP2/4/8 head). Under TP, `head` (`1/W` vocab) and `fold_w` (`1/W`
    /// vocab) run vocab-parallel across all ranks, exchanging only `[act.len()][W][4]` floats (`<= 512 B`) per Markov
    /// step and a 1-byte early-stop bitmask over the head's control streams.
    pub fn draft_multi(&self, g: &GpuModel, b: &mut MtpBatch, ws: &mut [MtpWork], lanes: &[MtpLane], reqs: &[DraftReq],
                       more: &mut dyn FnMut(usize, usize, &[f32]) -> bool) -> Result<Vec<Draft>> {
        self.draft_multi_inner(g, b, ws, lanes, reqs, Some(more), None)
    }

    /// Node-side lockstep twin of `draft_multi` for TP2/4/8 (reads the 1-byte early-stop bitmask from `ctl_stream`).
    pub fn draft_multi_node(&self, g: &GpuModel, b: &mut MtpBatch, ws: &mut [MtpWork], lanes: &[MtpLane], reqs: &[DraftReq],
                            ctl_stream: &mut std::net::TcpStream) -> Result<()> {
        let _ = self.draft_multi_inner(g, b, ws, lanes, reqs, None, Some(ctl_stream))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn draft_multi_inner(&self, g: &GpuModel, b: &mut MtpBatch, ws: &mut [MtpWork], lanes: &[MtpLane], reqs: &[DraftReq],
                         mut more: Option<&mut dyn FnMut(usize, usize, &[f32]) -> bool>,
                         mut node_stream: Option<&mut std::net::TcpStream>) -> Result<Vec<Draft>> {
        use std::io::Read;
        let nl = reqs.len();
        ensure!((1..=LMAX).contains(&nl), "multi-lane draft of {nl} lanes");
        let k = g.k;
        let dev = &g.dev;
        let eps = self.cfg.eps;
        let theta = self.cfg.rope_theta as f32;
        let n = self.n;
        let rows = nl * B;
        let (emb, embn, head) = g.shared_for_mtp();
        k.sync()?;
        for (l, r) in reqs.iter().enumerate() {
            let w = &mut ws[r.lane];
            let it = if r.t > 0.0 { 1.0 / r.t } else { 0.0 };
            let mut tk = [0i32; B];
            tk[0] = r.anchor as i32;
            dev.htod_sync_copy_into(&tk, &mut w.tok)?;
            let seeds: Vec<u64> = (0..16).map(|j| mix64(r.seed ^ mix64(0xd5a7_0000 + j as u64))).collect();
            dev.htod_sync_copy_into(&seeds, &mut w.seeds)?;
            dev.htod_sync_copy_into(&[it; 16], &mut w.invt)?;
            let hl = dp(&b.h) + (l * B * D * 4) as u64;
            k.launch("ink_embed_norm", (1, 1, 1), (256, 1, 1), 0, KArgs::new().v(dp(&w.tok)).v(emb).v(embn).v(hl).v(D as i32).v(self.target_eps))?;
            k.launch("ink6_block_init", ((B - 1) as u32, 1, 1), (256, 1, 1), 0, KArgs::new().v(dp(&self.mask)).v(hl).v(D as i32))?;
        }
        for (li, ly) in self.layers.iter().enumerate() {
            let delta = if li == 0 { 0u64 } else { dp(&b.ao) };
            k.launch("ink6_add_norm", (rows as u32, 1, 1), (256, 1, 1), 0,
                     KArgs::new().v(dp(&b.h)).v(delta).v(dp(&ly.in_norm)).v(dp(&b.xb)).v(0u64).v(D as i32).v(eps))?;
            k.proj(&ly.qkv, dp(&b.xb), dp(&b.xb), dp(&b.xb), 1 << 30, 1 << 30, dp(&b.y), ly.qkv.n_pad, rows, 0)?;
            for (l, r) in reqs.iter().enumerate() {
                let lane = &lanes[r.lane];
                let (qf, bk, bv) = (dp(&b.qf) + (l * B * NH * 128 * 4) as u64, dp(&b.bk) + (l * B * 1024 * 2) as u64, dp(&b.bv) + (l * B * 1024 * 2) as u64);
                k.launch("ink6_qk_rope", (B as u32, 1, 1), (256, 1, 1), 0,
                         KArgs::new().v(dp(&b.y) + (l * B * ly.qkv.n_pad * 4) as u64).v(ly.qkv.n_pad as i32).v(dp(&ly.q_norm)).v(dp(&ly.k_norm))
                             .v(qf).v(bk).v(bv).v((r.p as i64 + self.pos_shift) as i32).v(eps).v(theta))?;
                let lo = if ly.local { r.p.saturating_sub(self.cfg.sliding_window) } else { 0 };
                let ntot = r.p - lo + B;
                let per = (ntot.div_ceil(SMAX).max(256)).div_ceil(32) * 32;
                let s = ntot.div_ceil(per);
                k.launch("ink6_dattn", (8, s as u32, 1), (256, 1, 1), 0,
                         KArgs::new().v(qf).v(dp(&lane.kc[li])).v(dp(&lane.vc[li])).v((lane.cap[li] - 1) as u32).v(bk).v(bv)
                             .v(lo as i32).v(r.p as i32).v(per as i32).v(dp(&b.pm)).v(dp(&b.pl)).v(dp(&b.pacc)))?;
                k.launch("ink6_dattn_comb", (NH as u32, B as u32, 1), (128, 1, 1), 0,
                         KArgs::new().v(dp(&b.pm)).v(dp(&b.pl)).v(dp(&b.pacc)).v(s as i32).v(dp(&b.att) + (l * B * NH * 128 * 2) as u64))?;
            }
            k.proj(&ly.o, dp(&b.att), dp(&b.att), dp(&b.att), 1 << 30, 1 << 30, dp(&b.ao), D, rows, 0)?;
            k.launch("ink6_add_norm", (rows as u32, 1, 1), (256, 1, 1), 0,
                     KArgs::new().v(dp(&b.h)).v(dp(&b.ao)).v(dp(&ly.post_norm)).v(dp(&b.xb)).v(0u64).v(D as i32).v(eps))?;
            k.proj(&ly.gu, dp(&b.xb), dp(&b.xb), dp(&b.xb), 1 << 30, 1 << 30, dp(&b.y), ly.gu.n_pad, rows, 0)?;
            k.launch("ink6_silu_mul", ((INTER / 256) as u32, rows as u32, 1), (256, 1, 1), 0,
                     KArgs::new().v(dp(&b.y)).v(ly.gu.n_pad as i32).v(INTER as i32).v(dp(&b.act)))?;
            k.proj(&ly.down, dp(&b.act), dp(&b.act), dp(&b.act), 1 << 30, 1 << 30, dp(&b.ao), D, rows, 0)?;
        }
        k.launch("ink6_add_norm", (rows as u32, 1, 1), (256, 1, 1), 0,
                 KArgs::new().v(dp(&b.h)).v(dp(&b.ao)).v(dp(&self.norm)).v(dp(&b.finb)).v(dp(&b.fin)).v(D as i32).v(eps))?;
        // Base logits of every block row on THIS rank's 1/W LM head slice (W times faster under TP!)
        k.launch("ink6_head32", (head.grid(), 1, 1), (256, 1, 1), 0,
                 KArgs::new().v(dp(&head.w)).v(dp(&b.fin)).v(dp(&b.base)).v(rows as i32).v(head.k as i32).v(n as i32).v(n as i32))?;
        dev.memset_zeros(&mut b.state)?;
        let mut conf_h: Vec<Vec<f32>> = vec![Vec::new(); nl];
        let mut live: Vec<bool> = reqs.iter().map(|r| r.cap >= 1).collect();
        let mut steps = vec![0usize; nl];
        let rw = 2 * 256 + 2 * D;
        for j in 0..GMAX {
            let act: Vec<usize> = (0..nl).filter(|&l| live[l] && j < reqs[l].cap.min(GMAX)).collect();
            if act.is_empty() { break; }
            for (a, &l) in act.iter().enumerate() {
                let r = &reqs[l];
                k.launch("ink6_rnn_in", (20, 1, 1), (256, 1, 1), 0,
                         KArgs::new().v(dp(&b.state) + (l * 512) as u64).v(dp(&ws[r.lane].tok)).v(j as i32).v(dp(&self.w1))
                             .v(dp(&b.finb) + ((l * B + j + 1) * D * 2) as u64).v(dp(&lanes[r.lane].frnn)).v(dp(&b.rin) + (a * rw * 2) as u64)
                             .v(256i32).v(D as i32))?;
            }
            k.proj(&self.rnn, dp(&b.rin), dp(&b.rin), dp(&b.rin), 1 << 30, 1 << 30, dp(&b.ry), self.rnn.n_pad, act.len(), 0)?;
            let mut fa = FoldArgs::default();
            for (a, &l) in act.iter().enumerate() {
                let r = &reqs[l];
                k.launch("ink6_rnn_gates", (1, 1, 1), (256, 1, 1), 0,
                         KArgs::new().v(dp(&b.ry) + (a * self.rnn.n_pad * 4) as u64).v(dp(&self.rnn_b)).v(dp(&b.state) + (l * 512) as u64)
                             .v(dp(&b.latent) + (a * 512) as u64).v(256i32))?;
                let w = &ws[r.lane];
                fa.base[a] = dp(&b.base) + ((l * B + j + 1) * n * 4) as u64;
                fa.out[a] = dp(&w.step) + (j * n * 4) as u64;
                fa.it[a] = if r.t > 0.0 { 1.0 / r.t } else { 1.0 };
                fa.pmax[a] = dp(&w.fpm);
                fa.psum[a] = dp(&w.fps);
            }
            k.launch("ink6_fold4", (FOLD_G as u32, 1, 1), (256, 1, 1), 0,
                     KArgs::new().v(dp(&self.fold_w)).v(dp(&self.fold_b)).v(dp(&b.latent)).v(fa).v(act.len() as i32).v(n as i32).v(self.cap))?;
            if let Some(tp) = &g.tp {
                use super::super::tp::kern as tpk;
                let mut pa = tpk::MtpStepPackArgs::default();
                let mut oa = tpk::MtpStepFinArgs::default();
                for (a, &l) in act.iter().enumerate() {
                    let w = &ws[reqs[l].lane];
                    let step = dp(&w.step) + (j * n * 4) as u64;
                    k.launch("ink_tp_gumbel_part", (SMP_G as u32, 1, 1), (256, 1, 1), 0,
                             KArgs::new().v(step).v(n as i32).v(n as i32).v(self.voff as i32)
                                 .v(dp(&w.seeds) + (j * 8) as u64).v(dp(&w.invt)).v(dp(&w.gpm)).v(dp(&w.gpi)))?;
                    pa.fpm[a] = dp(&w.fpm); pa.fps[a] = dp(&w.fps);
                    pa.gpm[a] = dp(&w.gpm); pa.gpi[a] = dp(&w.gpi);
                    oa.lse_q[a] = dp(&w.lse_q) + (j * 4) as u64;
                    oa.conf[a] = dp(&w.conf) + (j * 4) as u64;
                    oa.tok[a] = dp(&w.tok) + ((j + 1) * 4) as u64;
                }
                let (rank, world) = (g.rank.rank, g.rank.world);
                tpk::mtp_step_pack(k, pa, act.len(), FOLD_G, SMP_G, dp(&tp.rec), rank, world)?;
                tp.x.all_reduce(dev, dp(&tp.rec), act.len() * world * 4)?;
                tpk::mtp_step_fin(k, dp(&tp.rec), world, oa, act.len())?;
            } else {
                for &l in &act {
                    let w = &ws[reqs[l].lane];
                    let step = dp(&w.step) + (j * n * 4) as u64;
                    k.launch("ink6_lse_fin", (1, 1, 1), (32, 1, 1), 0,
                             KArgs::new().v(dp(&w.fpm)).v(dp(&w.fps)).v(FOLD_G as i32).v(dp(&w.lse_q) + (j * 4) as u64).v(dp(&w.conf) + (j * 4) as u64))?;
                    k.launch("ink3_gumbel_part", (SMP_G as u32, 1, 1), (256, 1, 1), 0,
                             KArgs::new().v(step).v(n as i32).v(n as i32).v(dp(&w.seeds) + (j * 8) as u64).v(dp(&w.invt)).v(dp(&w.gpm)).v(dp(&w.gpi)))?;
                    k.launch("ink3_gumbel_pick", (1, 1, 1), (256, 1, 1), 0,
                             KArgs::new().v(dp(&w.gpm)).v(dp(&w.gpi)).v(SMP_G as i32).v(dp(&w.tok) + ((j + 1) * 4) as u64))?;
                }
            }
            let mut need_mask_sync = false;
            for &l in &act {
                steps[l] = j + 1;
                if j + 1 >= reqs[l].cap.min(GMAX) { live[l] = false; }
                else { need_mask_sync = true; }
            }
            if need_mask_sync {
                if let Some(st) = node_stream.as_deref_mut() {
                    let mut buf = [0u8; 1];
                    st.read_exact(&mut buf).context("inkling2 TP node: read draft early-stop mask")?;
                    let mask = buf[0];
                    for &l in &act { if live[l] && (mask & (1u8 << l)) == 0 { live[l] = false; } }
                } else {
                    k.sync()?;
                    let mut mask = 0u8;
                    for &l in &act {
                        if !live[l] { continue; }
                        let mut cj = [0f32; 1];
                        dev.dtoh_sync_copy_into(&ws[reqs[l].lane].conf.slice(j..j + 1), &mut cj)?;
                        conf_h[l].push(cj[0]);
                        if let Some(cb) = more.as_deref_mut() {
                            if !cb(l, j, &conf_h[l]) { live[l] = false; }
                        }
                        if live[l] { mask |= 1u8 << l; }
                    }
                    if let Some(tp) = &g.tp { tp.broadcast_u8(mask)?; }
                }
            }
        }
        if let Some(tp) = &g.tp { tp.x.check("after vocab-parallel draft_multi")?; }
        if node_stream.is_some() { return Ok(Vec::new()); }
        k.sync()?;
        let mut out = Vec::with_capacity(nl);
        for (l, r) in reqs.iter().enumerate() {
            let w = &ws[r.lane];
            let mut tok = [0i32; B];
            dev.dtoh_sync_copy_into(&w.tok, &mut tok)?;
            let mut conf = [0f32; B];
            dev.dtoh_sync_copy_into(&w.conf, &mut conf)?;
            out.push(Draft { toks: tok[1..=steps[l]].iter().map(|&x| x as u32).collect(), conf: conf[..steps[l]].to_vec() });
        }
        Ok(out)
    }
}
