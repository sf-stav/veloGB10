//! DeepSeek-V4.1-Flash GPU Vision Tower (32-layer ViT) + 3x3 Unfold Aligner + Rank-0 Splice (`95_vision.cuh`).
//!
//! Semantics match `inference/vision.py` and `inference/model.py:1227-1243`:
//! * `PatchEmbed`: `[N, 588] -> [N, 1024]` + bias
//! * `32 x Block`: `RMSNorm(1024, 1e-6)` -> `wqkv` -> 2D RoPE on `q, k` -> bidirectional SDPA (`scale = 0.125`)
//!   -> `wo` + residual -> `RMSNorm(1024, 1e-6)` -> `mlp.w1` -> `silu(gate) * up` (unclamped) -> `mlp.w2` + residual
//! * `vision.norm`: `RMSNorm(1024, 1e-6)`
//! * `Aligner`: bottom/right zero-pad to multiple of 3 -> `3x3` channel-outer unfold `[N_llm, 9216]`
//!   -> `aligner.w1` + bias -> exact `erff` GELU -> `aligner.w2` + bias -> `[N_llm, 5120]` BF16.

use anyhow::{bail, ensure, Context, Result};
use cudarc::driver::DeviceRepr;

use crate::dsv41::census::classify::{Role, TensorClass as C};
use crate::dsv41::census::LinearFormat;
use crate::dsv41::config::Vision;
use crate::dsv41::fwd::dev::{bf16, Buf, Gpu};
use crate::dsv41::fwd::glue::{self, Scr};
use crate::dsv41::fwd::layer::{bytemuck_i32, bytemuck_u16, fmt_of, Bump};
use crate::dsv41::vision::preproc::{PreparedImage, TOK_IMAGE, TOK_IMAGE_END, TOK_IMAGE_NEW_LINE, TOK_IMAGE_START};
use crate::dsv41::weights::{Accum, BackendKind, DenseWeight, DevTensor};

pub const FNS: &[&str] = &[
    "d41_vis_gemm_bf16",
    "d41_vis_rmsnorm",
    "d41_vis_qkv_rope",
    "d41_vis_sdpa_bf16",
    "d41_vis_swiglu",
    "d41_vis_unfold3x3",
    "d41_vis_splice_embed",
];

pub const EPI_NONE: i32 = 0;
pub const EPI_RESIDUAL: i32 = 1;
pub const EPI_GELU_ERF: i32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct VisGemmArgs {
    pub x: u64,
    pub w: u64,
    pub bias: u64,
    pub resid: u64,
    pub y: u64,
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub lda: i32,
    pub ldw: i32,
    pub ldy: i32,
    pub ldr: i32,
    pub w_fmt: u32,
    pub bias_fmt: u32,
    pub epi: i32,
    pub pad0: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct VisNormArgs {
    pub x: u64,
    pub w: u64,
    pub y: u64,
    pub rows: i32,
    pub dim: i32,
    pub w_fmt: u32,
    pub eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct VisQkvRopeArgs {
    pub qkv: u64,
    pub q_out: u64,
    pub k_out: u64,
    pub v_out: u64,
    pub n_tok: i32,
    pub n_w: i32,
    pub inv_freq: [f32; 16],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct VisSdpaArgs {
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub out: u64,
    pub n_tok: i32,
    pub n_heads: i32,
    pub scale: f32,
    pub pad0: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct VisSwigluArgs {
    pub x: u64,
    pub y: u64,
    pub rows: i32,
    pub inter: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct VisUnfoldArgs {
    pub x: u64,
    pub y: u64,
    pub n_h: i32,
    pub n_w: i32,
    pub n_llm_h: i32,
    pub n_llm_w: i32,
    pub dim: i32,
    pub pad0: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct VisSpliceArgs {
    pub embed: u64,
    pub aligner: u64,
    pub img_start: u64,
    pub img_newline: u64,
    pub img_end: u64,
    pub row_ops: u64,
    pub rows: i32,
    pub dim: i32,
}

unsafe impl DeviceRepr for VisGemmArgs {}
unsafe impl DeviceRepr for VisNormArgs {}
unsafe impl DeviceRepr for VisQkvRopeArgs {}
unsafe impl DeviceRepr for VisSdpaArgs {}
unsafe impl DeviceRepr for VisSwigluArgs {}
unsafe impl DeviceRepr for VisUnfoldArgs {}
unsafe impl DeviceRepr for VisSpliceArgs {}

/// One ViT or Aligner linear: takes the fast `m16n8k16` tensor-core GEMM (`d41_vis_gemm_bf16`) for BF16/F16
/// weights (all native and EXL3 checkpoints), with a fallback to `glue::linear` if a pack quantizes it.
#[derive(Clone)]
pub struct VisLinear {
    pub dense: (DenseWeight, BackendKind),
    pub plain_w: Option<(u64, u32)>,
    pub plain_bias: Option<(u64, u32)>,
    pub out: usize,
    pub inn: usize,
}

impl VisLinear {
    pub fn from_dense(dense: (DenseWeight, BackendKind)) -> Result<Self> {
        let out = dense.0.out as usize;
        let inn = dense.0.inn as usize;
        let plain_w = match &dense.0.format {
            LinearFormat::Plain(dt) if dt == "BF16" || dt == "F16" => {
                let wt = dense.0.tensors.iter().find(|t| t.role == Role::Weight).context("vision linear missing weight")?;
                Some((wt.ptr.0, fmt_of(wt)?))
            }
            _ => None,
        };
        let plain_bias = match dense.0.tensors.iter().find(|t| t.role == Role::Bias) {
            Some(bt) => Some((bt.ptr.0, fmt_of(bt)?)),
            None => None,
        };
        Ok(VisLinear { dense, plain_w, plain_bias, out, inn })
    }

    pub fn run(&self, g: &Gpu, b: &mut Bump, scr: &Scr, x: u64, rows: usize, y: u64, epi: i32, resid: u64) -> Result<()> {
        ensure!(rows >= 1, "VisLinear::run: empty rows");
        if let Some((wp, w_fmt)) = self.plain_w {
            let (bp, b_fmt) = self.plain_bias.unwrap_or((0, 0));
            let a = VisGemmArgs {
                x,
                w: wp,
                bias: bp,
                resid,
                y,
                m: rows as i32,
                n: self.out as i32,
                k: self.inn as i32,
                lda: self.inn as i32,
                ldw: self.inn as i32,
                ldy: self.out as i32,
                ldr: self.out as i32,
                w_fmt,
                bias_fmt: b_fmt,
                epi,
                pad0: 0,
            };
            let gx = (self.out as u32).div_ceil(64);
            let gy = (rows as u32).div_ceil(64);
            return g.launch1("d41_vis_gemm_bf16", (gx, gy, 1), (128, 1, 1), 0, &a);
        }
        // Quantized vision-linear fallback through the standard DenseWeight backend (`glue::linear` applies bias).
        glue::linear(g, "vis.linear", &self.dense, &bf16(x, rows, self.inn, self.inn), &bf16(y, rows, self.out, self.out), Accum::Overwrite, scr)?;
        if epi != EPI_NONE {
            bail!("quantized vision linear epilogue {epi} requires plain BF16/F16 weights (got {:?})", self.dense.0.format);
        }
        let _ = b;
        Ok(())
    }
}

pub struct VisBlock {
    pub norm1: DevTensor,
    pub wqkv: VisLinear,
    pub wo: VisLinear,
    pub norm2: DevTensor,
    pub w1: VisLinear,
    pub w2: VisLinear,
}

/// Rank-0 resident Vision Tower + Aligner + learned delimiter embeddings (`~0.904 GiB`).
pub struct VisionWeights {
    pub vis: Vision,
    pub hidden: usize,
    pub inv_freq: [f32; 16],
    pub patch_embed: VisLinear,
    pub blocks: Vec<VisBlock>,
    pub norm: DevTensor,
    pub aligner_w1: VisLinear,
    pub aligner_w2: VisLinear,
    pub image_start: DevTensor,
    pub image_newline: DevTensor,
    pub image_end: DevTensor,
    pub bytes: u64,
    pub _arena: Vec<Buf>,
}

/// Exact `vision.py:10`: `1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))` (`dim = 32`).
pub fn vision_inv_freq(rope_dim: usize, theta: f64) -> Result<[f32; 16]> {
    ensure!(rope_dim == 32, "vision 2D RoPE expects rope_dim = 32 (got {rope_dim})");
    let mut out = [0.0f32; 16];
    let th = theta as f32;
    for (i, slot) in out.iter_mut().enumerate() {
        let exp = (2 * i) as f32 / (rope_dim as f32);
        *slot = 1.0f32 / th.powf(exp);
    }
    Ok(out)
}

impl VisionWeights {
    pub fn rmsnorm(&self, g: &Gpu, x: u64, w: &DevTensor, rows: usize, y: u64) -> Result<()> {
        let a = VisNormArgs {
            x,
            w: w.ptr.0,
            y,
            rows: rows as i32,
            dim: self.vis.hidden as i32,
            w_fmt: fmt_of(w)?,
            eps: 1e-6,
        };
        g.launch1("d41_vis_rmsnorm", (rows as u32, 1, 1), (256, 1, 1), 0, &a)
    }

    /// Run full 32-layer ViT + 3x3 Aligner on `img` (`patches` uploaded to `b`), writing `[n_llm_h * n_llm_w, 5120]` BF16 into `out_ptr`.
    pub fn encode_image(&self, g: &Gpu, b: &mut Bump, scr: &Scr, img: &PreparedImage, out_ptr: u64) -> Result<()> {
        let v = &self.vis;
        let n_tok = img.n_patches();
        let patch_dim = 3 * v.patch * v.patch;
        ensure!(img.patches.len() == n_tok * patch_dim, "patches len {} != {} * {patch_dim}", img.patches.len(), n_tok);
        ensure!(v.hidden == 1024 && v.heads == 16 && v.downsample == 3, "unsupported vision shape: {v:?}");

        let mark = b.off;
        let patches_dev = b.take(img.patches.len() * 2)?;
        g.h2d(patches_dev, bytemuck_u16(&img.patches))?;

        let x = b.take(n_tok * v.hidden * 2)?;
        let norm_buf = b.take(n_tok * v.hidden * 2)?;
        let sub_mark = b.off;

        self.patch_embed.run(g, b, scr, patches_dev, n_tok, x, EPI_NONE, 0)?;

        for blk in &self.blocks {
            b.off = sub_mark;
            let qkv = b.take(n_tok * 3 * v.hidden * 2)?;
            let q = b.take(v.heads * n_tok * 64 * 2)?;
            let k = b.take(v.heads * n_tok * 64 * 2)?;
            let vv = b.take(v.heads * n_tok * 64 * 2)?;
            let attn_out = b.take(n_tok * v.hidden * 2)?;

            self.rmsnorm(g, x, &blk.norm1, n_tok, norm_buf)?;
            blk.wqkv.run(g, b, scr, norm_buf, n_tok, qkv, EPI_NONE, 0)?;

            let ra = VisQkvRopeArgs {
                qkv,
                q_out: q,
                k_out: k,
                v_out: vv,
                n_tok: n_tok as i32,
                n_w: img.n_vit_w as i32,
                inv_freq: self.inv_freq,
            };
            g.launch1("d41_vis_qkv_rope", ((n_tok as u32).div_ceil(4), v.heads as u32, 1), (128, 1, 1), 0, &ra)?;

            let sa = VisSdpaArgs {
                q,
                k,
                v: vv,
                out: attn_out,
                n_tok: n_tok as i32,
                n_heads: v.heads as i32,
                scale: 0.125,
                pad0: 0,
            };
            g.launch1("d41_vis_sdpa_bf16", ((n_tok as u32).div_ceil(64), v.heads as u32, 1), (128, 1, 1), 0, &sa)?;
            blk.wo.run(g, b, scr, attn_out, n_tok, x, EPI_RESIDUAL, x)?;

            b.off = sub_mark;
            let mlp_gu = b.take(n_tok * 2 * v.inter * 2)?;
            let mlp_act = b.take(n_tok * v.inter * 2)?;

            self.rmsnorm(g, x, &blk.norm2, n_tok, norm_buf)?;
            blk.w1.run(g, b, scr, norm_buf, n_tok, mlp_gu, EPI_NONE, 0)?;
            let sw = VisSwigluArgs { x: mlp_gu, y: mlp_act, rows: n_tok as i32, inter: v.inter as i32 };
            g.launch1("d41_vis_swiglu", ((v.inter as u32).div_ceil(256), n_tok as u32, 1), (256, 1, 1), 0, &sw)?;
            blk.w2.run(g, b, scr, mlp_act, n_tok, x, EPI_RESIDUAL, x)?;
        }

        b.off = sub_mark;
        self.rmsnorm(g, x, &self.norm, n_tok, norm_buf)?;

        let n_llm = img.n_aligner_tokens();
        let unfold_dim = v.hidden * v.downsample * v.downsample;
        let unfolded = b.take(n_llm * unfold_dim * 2)?;
        let ua = VisUnfoldArgs {
            x: norm_buf,
            y: unfolded,
            n_h: img.n_vit_h as i32,
            n_w: img.n_vit_w as i32,
            n_llm_h: img.n_llm_h as i32,
            n_llm_w: img.n_llm_w as i32,
            dim: v.hidden as i32,
            pad0: 0,
        };
        g.launch1("d41_vis_unfold3x3", (n_llm as u32, 1, 1), (256, 1, 1), 0, &ua)?;

        let mid = b.take(n_llm * self.hidden * 2)?;
        self.aligner_w1.run(g, b, scr, unfolded, n_llm, mid, EPI_GELU_ERF, 0)?;
        self.aligner_w2.run(g, b, scr, mid, n_llm, out_ptr, EPI_NONE, 0)?;

        b.off = mark;
        let _ = C::VisPatchEmbed;
        Ok(())
    }

    /// Overwrite image-span rows inside Rank 0's `[rows, dim]` BF16 embedding buffer (`embed_ptr`) for chunk `[c0, c1)`.
    pub fn splice_chunk_embeddings(
        &self,
        g: &Gpu,
        b: &mut Bump,
        embed_ptr: u64,
        aligner_ptr: u64,
        row_ops: &[i32],
    ) -> Result<()> {
        if row_ops.is_empty() {
            return Ok(());
        }
        let ops_dev = b.take(row_ops.len() * 4)?;
        g.h2d_async(ops_dev, row_ops)?;
        let a = VisSpliceArgs {
            embed: embed_ptr,
            aligner: aligner_ptr,
            img_start: self.image_start.ptr.0,
            img_newline: self.image_newline.ptr.0,
            img_end: self.image_end.ptr.0,
            row_ops: ops_dev,
            rows: row_ops.len() as i32,
            dim: self.hidden as i32,
        };
        let _ = bytemuck_i32;
        g.launch1("d41_vis_splice_embed", (row_ops.len() as u32, 1, 1), (256, 1, 1), 0, &a)
    }
}

/// Build the per-prompt `row_ops: Vec<i32>` table mapping every prompt position `0..plen` to:
/// `-1` (`TEXT`), `-2` (`IMAGE_START`), `-3` (`IMAGE_NEW_LINE`), `-4` (`IMAGE_END`), or `>= 0` (row index inside
/// Rank 0's concatenated `aligner_embeds` buffer for encoded images).
/// Images completely inside `0..skip_before` (`img.end <= skip_before`, i.e., covered by a prefix-cache hit)
/// do NOT need ViT encoding and are omitted from the encoded buffer (`aligner_row` counts only images with `img.end > skip_before`).
pub fn build_vl_row_ops(token_types: &[i8], encoded_spans: &[(usize, usize)]) -> (Vec<i32>, usize) {
    let mut ops = vec![-1i32; token_types.len()];
    let mut aligner_idx = 0i32;
    let mut span_it = encoded_spans.iter().copied().peekable();

    for (pos, &tt) in token_types.iter().enumerate() {
        while span_it.peek().is_some_and(|&(_, end)| pos >= end) {
            span_it.next();
        }
        let in_encoded = span_it.peek().is_some_and(|&(st, en)| pos >= st && pos < en);
        ops[pos] = match tt {
            TOK_IMAGE_START => -2,
            TOK_IMAGE_NEW_LINE => -3,
            TOK_IMAGE_END => -4,
            TOK_IMAGE if in_encoded => {
                let r = aligner_idx;
                aligner_idx += 1;
                r
            }
            _ => -1,
        };
    }
    (ops, aligner_idx as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vis_arg_struct_sizes_match_cuda_abi() {
        assert_eq!(std::mem::size_of::<VisGemmArgs>(), 88);
        assert_eq!(std::mem::size_of::<VisNormArgs>(), 40);
        assert_eq!(std::mem::size_of::<VisQkvRopeArgs>(), 104);
        assert_eq!(std::mem::size_of::<VisSdpaArgs>(), 48);
        assert_eq!(std::mem::size_of::<VisSwigluArgs>(), 24);
        assert_eq!(std::mem::size_of::<VisUnfoldArgs>(), 40);
        assert_eq!(std::mem::size_of::<VisSpliceArgs>(), 56);
    }

    #[test]
    fn build_vl_row_ops_assigns_sequential_aligner_rows_and_skips_cached_spans() {
        let mut tt = vec![-1i8; 2];
        tt.extend_from_slice(&[TOK_IMAGE_START, TOK_IMAGE, TOK_IMAGE, TOK_IMAGE_NEW_LINE, TOK_IMAGE_END]); // 2..7 (cached)
        tt.push(-1); // 7
        tt.extend_from_slice(&[TOK_IMAGE_START, TOK_IMAGE, TOK_IMAGE, TOK_IMAGE_NEW_LINE, TOK_IMAGE_END]); // 8..13 (uncached)
        let (ops, n_al) = build_vl_row_ops(&tt, &[(8, 13)]);
        assert_eq!(n_al, 2);
        assert_eq!(&ops[8..13], &[-2, 0, 1, -3, -4]);
        assert_eq!(ops[0], -1);
        assert_eq!(ops[7], -1);
    }
}
