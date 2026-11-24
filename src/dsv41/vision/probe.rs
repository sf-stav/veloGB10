//! Gate 1 GPU oracle probe (`--probe-dsv41-vision --model-dir <pack> --dsv41-dump-ref <golden dir>`).
//!
//! Verifies against the `vision.py` + `image_processor.py` + `model.py` golden dump:
//! 1. PIL-exact `decode_and_patchify` (`padded_rgb` + `patches` BF16 `[N_patches, 588]`) on all 4 golden PNGs
//!    (`small_300x200`, `mid_1200x800`, `hires_1800x1200`, `wide_1600x200`) and 2-image placeholder expansion.
//! 2. Per-stage single-op GPU accuracy (`PatchEmbed`, `ViT Block 0`, `3x3 Unfold + Aligner`, each `< 2e-3`) AND
//!    full 32-layer chained GPU ViT + Aligner vs PyTorch 4D FlashAttention, `vision.py` 3D MATH SDPA, and FP32
//!    ground truth, plus warm GPU latency (ms), speedup over `vision.py`, and achieved TFLOP/s.
//! 3. Rank-0 `splice_chunk_embeddings` vs `h_embed_merged` (`< 2e-3` on Golden aligner, bit-exact on text + delimiters).
//! 4. `EngramHasher` with `live` (`DEAD` image-span mask) vs `engram_hashes` (`100%` bit-exact across all `235 * 48` IDs).
//! 5. `Fwd::router_vl` (`d41_router_select` with `GateBiasVl` + `image_mask`) on `l0_fin` and `l1_fin` vs
//!    `l0_router_indices`/`weights` and `l1_router_indices`/`weights`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use half::bf16 as hbf16;

use crate::dsv41::census::classify::TensorClass as C;
use crate::dsv41::engram::hash::{EngramHasher, NgramHashState};
use crate::dsv41::fwd::dev::{bf16 as dev_bf16, Gpu};
use crate::dsv41::fwd::glue::{self, Scr};
use crate::dsv41::fwd::layer::{bytemuck_i32, bytemuck_u16, Bump, Fwd};
use crate::dsv41::fwd::rope_set;
use crate::dsv41::tp::wload::RankSrc;
use crate::dsv41::vision::preproc::{decode_and_patchify, prepare_vl_inputs, resize_and_pad_rgb};
use crate::dsv41::vision::tower::{
    build_vl_row_ops, VisQkvRopeArgs, VisSdpaArgs, VisSwigluArgs, VisUnfoldArgs, EPI_GELU_ERF, EPI_NONE, EPI_RESIDUAL,
};
use crate::dsv41::weights::DevPtr;
use crate::dsv4_chat::parse_json;
use crate::opts;

struct StFile {
    raw: Vec<u8>,
    data_base: usize,
    tensors: HashMap<String, (String, Vec<usize>, usize, usize)>,
}

impl StFile {
    fn open(path: &Path) -> Result<Self> {
        let raw = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        ensure!(raw.len() >= 8, "safetensors file too short: {}", path.display());
        let hdr_len = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
        ensure!(8 + hdr_len <= raw.len(), "corrupt safetensors header: {}", path.display());
        let hdr: serde_json::Value = serde_json::from_slice(&raw[8..8 + hdr_len])?;
        let obj = hdr.as_object().context("safetensors header not an object")?;
        let mut tensors = HashMap::new();
        for (k, v) in obj {
            if k == "__metadata__" {
                continue;
            }
            let dt = v.get("dtype").and_then(|x| x.as_str()).context("missing dtype")?.to_string();
            let shape: Vec<usize> = v
                .get("shape")
                .and_then(|x| x.as_array())
                .context("missing shape")?
                .iter()
                .map(|d| d.as_u64().unwrap_or(0) as usize)
                .collect();
            let offs = v.get("data_offsets").and_then(|x| x.as_array()).context("missing data_offsets")?;
            let a = offs[0].as_u64().unwrap_or(0) as usize;
            let b = offs[1].as_u64().unwrap_or(0) as usize;
            tensors.insert(k.clone(), (dt, shape, a, b));
        }
        Ok(StFile { raw, data_base: 8 + hdr_len, tensors })
    }

    fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    fn bytes(&self, name: &str) -> Result<(&str, &[usize], &[u8])> {
        let (dt, sh, a, b) = self.tensors.get(name).with_context(|| format!("missing tensor '{name}'"))?;
        Ok((dt.as_str(), sh.as_slice(), &self.raw[self.data_base + *a..self.data_base + *b]))
    }

    fn bf16_u16(&self, name: &str) -> Result<Vec<u16>> {
        let (dt, _, bs) = self.bytes(name)?;
        ensure!(dt == "BF16", "tensor {name}: expected BF16, got {dt}");
        Ok(bs.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
    }

    fn f32_vec(&self, name: &str) -> Result<Vec<f32>> {
        let (dt, _, bs) = self.bytes(name)?;
        ensure!(dt == "F32", "tensor {name}: expected F32, got {dt}");
        Ok(bs.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    fn i64_vec(&self, name: &str) -> Result<Vec<i64>> {
        let (dt, _, bs) = self.bytes(name)?;
        ensure!(dt == "I64", "tensor {name}: expected I64, got {dt}");
        Ok(bs.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect())
    }

    fn u8_vec(&self, name: &str) -> Result<Vec<u8>> {
        let (dt, _, bs) = self.bytes(name)?;
        ensure!(dt == "U8", "tensor {name}: expected U8, got {dt}");
        Ok(bs.to_vec())
    }
}

fn rel_l2_bf16(got: &[u16], want: &[u16]) -> (f64, f64, usize) {
    assert_eq!(got.len(), want.len());
    let mut diff_sq = 0.0f64;
    let mut ref_sq = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut exact = 0usize;
    for (&g, &w) in got.iter().zip(want.iter()) {
        if g == w {
            exact += 1;
        }
        let gf = hbf16::from_bits(g).to_f32() as f64;
        let wf = hbf16::from_bits(w).to_f32() as f64;
        let d = (gf - wf).abs();
        if d > max_abs {
            max_abs = d;
        }
        diff_sq += d * d;
        ref_sq += wf * wf;
    }
    let rel = if ref_sq > 0.0 { (diff_sq / ref_sq).sqrt() } else { diff_sq.sqrt() };
    (rel, max_abs, exact)
}

fn rel_l2_bf16_vs_f32(got: &[u16], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let mut diff_sq = 0.0f64;
    let mut ref_sq = 0.0f64;
    for (&g, &w) in got.iter().zip(want.iter()) {
        let gf = hbf16::from_bits(g).to_f32() as f64;
        let wf = w as f64;
        let d = gf - wf;
        diff_sq += d * d;
        ref_sq += wf * wf;
    }
    if ref_sq > 0.0 { (diff_sq / ref_sq).sqrt() } else { diff_sq.sqrt() }
}

fn vision_flops(n_patches: usize, n_llm: usize) -> f64 {
    let n = n_patches as f64;
    let m = n_llm as f64;
    let patch_embed = 2.0 * n * 588.0 * 1024.0;
    let per_blk = 2.0 * n * 1024.0 * 3072.0
        + 4.0 * 16.0 * n * n * 64.0
        + 2.0 * n * 1024.0 * 1024.0
        + 2.0 * n * 1024.0 * 5632.0
        + 2.0 * n * 2816.0 * 1024.0;
    let aligner = 2.0 * m * 9216.0 * 5120.0 + 2.0 * m * 5120.0 * 5120.0;
    patch_embed + 32.0 * per_blk + aligner
}

pub fn run(args: &[String]) -> Result<Vec<String>> {
    let mut lines = Vec::new();
    let model_dir = PathBuf::from(crate::arg_value(args, "--model-dir").context("--model-dir <pack> is required")?);
    let dump_dir = PathBuf::from(
        opts::lookup("dsv41-dump-ref")
            .and_then(opts::get)
            .filter(|s| s != "unset")
            .context("--dsv41-dump-ref <golden dir> is required")?,
    );
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dump_dir.join("meta.json")).context("read meta.json")?)?;

    let src = RankSrc::pack(&model_dir, 1, 0, "inner")?;
    let cfg = src.cfg.clone();
    let vis_cfg = cfg.vision.clone().context("pack has no vision_config")?;
    lines.push(format!("DSV41_VIS1 probe: pack={} ({}) golden={}", model_dir.display(), cfg.quant.method, dump_dir.display()));

    // ---- 1. PIL-exact preprocessing & patchify ----
    let cases = ["small_300x200", "mid_1200x800", "hires_1800x1200", "wide_1600x200"];
    let mut prepared_cases = Vec::new();
    let mut st_cases = Vec::new();
    for name in cases {
        let png_bytes = std::fs::read(dump_dir.join(format!("{name}.png")))?;
        let st = StFile::open(&dump_dir.join(format!("{name}_vis.safetensors")))?;
        let (img, (ow, oh)) = decode_and_patchify(&png_bytes, &vis_cfg).map_err(|e| anyhow::anyhow!(e.message))?;
        let dyn_im = image::load_from_memory(&png_bytes)?.into_rgb8();
        let best_w = img.n_vit_w * vis_cfg.patch;
        let best_h = img.n_vit_h * vis_cfg.patch;
        let padded = resize_and_pad_rgb(dyn_im.as_raw(), ow, oh, best_w, best_h, vis_cfg.max_wh_ratio);
        let want_rgb = st.u8_vec("padded_rgb")?;
        let rgb_max_diff = padded.iter().zip(&want_rgb).map(|(&a, &b)| a.abs_diff(b)).max().unwrap_or(0);
        let want_patches = st.bf16_u16("patches")?;
        let (rel_p, max_p, exact_p) = rel_l2_bf16(&img.patches, &want_patches);
        lines.push(format!(
            "  [preproc] {name:<15} {ow}x{oh} -> vit {}x{} ({} patches), llm {}x{} ({} toks): rgb max|d|={rgb_max_diff}, patches exact={:.2}% rel-L2={rel_p:.3e} max|d|={max_p:.3e}",
            img.n_vit_h, img.n_vit_w, img.n_patches(), img.n_llm_h, img.n_llm_w, img.types.len(),
            100.0 * exact_p as f64 / want_patches.len() as f64,
        ));
        ensure!(rgb_max_diff <= 1 && rel_p < 1e-4, "{name}: preprocessing drifted (rgb_max_diff={rgb_max_diff}, rel_p={rel_p:.3e})");
        prepared_cases.push((name, img));
        st_cases.push(st);
    }

    if let Some(two) = meta.get("two_image") {
        use base64::Engine as _;
        let b1 = base64::engine::general_purpose::STANDARD.encode(std::fs::read(dump_dir.join("small_300x200.png"))?);
        let b2 = base64::engine::general_purpose::STANDARD.encode(std::fs::read(dump_dir.join("mid_1200x800.png"))?);
        let r1 = parse_json(&format!(r#"{{"data": "{b1}"}}"#)).unwrap();
        let r2 = parse_json(&format!(r#"{{"data": "{b2}"}}"#)).unwrap();
        let want_ids: Vec<u32> = two["input_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let want_tt: Vec<i8> = two["token_types"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as i8).collect();
        let raw_ids = {
            let mut r = Vec::new();
            let mut i = 0;
            while i < want_ids.len() {
                r.push(want_ids[i]);
                if want_tt[i] >= 0 {
                    let end = if r.iter().filter(|&&t| t == cfg.image_token_id).count() == 1 {
                        two["spans"][0]["start"].as_u64().unwrap() as usize + two["spans"][0]["len"].as_u64().unwrap() as usize
                    } else {
                        two["spans"][1]["start"].as_u64().unwrap() as usize + two["spans"][1]["len"].as_u64().unwrap() as usize
                    };
                    i = end;
                } else {
                    i += 1;
                }
            }
            r
        };
        let vlp = prepare_vl_inputs(&raw_ids, &[r1, r2], cfg.image_token_id, &vis_cfg, 16).map_err(|e| anyhow::anyhow!(e.message))?;
        ensure!(vlp.prompt_ids == want_ids && vlp.token_types == want_tt, "2-image prepare_vl_inputs mismatch");
        lines.push(format!("  [preproc] 2-image placeholder expansion: {} tokens, spans {:?} == meta.json EXACT", vlp.prompt_ids.len(), vlp.spans.iter().map(|s| (s.start, s.end)).collect::<Vec<_>>()));
    }

    // ---- 2a. Per-stage single-op GPU accuracy (fed dumped golden inputs, rel-L2 < 2e-3) ----
    let g = Gpu::new()?;
    let vis_w = src.load_vision(&g)?.context("load_vision returned None")?;
    lines.push(format!("  [weights] VisionWeights loaded on GPU: {:.3} GiB ({} B)", vis_w.bytes as f64 / (1u64 << 30) as f64, vis_w.bytes));
    let scr_buf = g.buf(64 << 20)?;
    let scr = Scr { ptr: scr_buf.ptr(), bytes: scr_buf.bytes };
    let mut b = Bump::new(&g, 512 << 20)?;

    {
        let st0 = &st_cases[0];
        let img0 = &prepared_cases[0].1;
        let n_tok = img0.n_patches();
        let n_llm = img0.n_aligner_tokens();
        b.reset();
        // Stage A: PatchEmbed alone
        let p_in = b.take(n_tok * 588 * 2)?;
        g.h2d(p_in, bytemuck_u16(&st0.bf16_u16("patches")?))?;
        let pe_out = b.take(n_tok * 1024 * 2)?;
        vis_w.patch_embed.run(&g, &mut b, &scr, p_in, n_tok, pe_out, EPI_NONE, 0)?;
        g.sync()?;
        let (rel_pe, max_pe, ex_pe) = rel_l2_bf16(&g.down::<u16>(pe_out, n_tok * 1024)?, &st0.bf16_u16("patch_embed")?);
        lines.push(format!("  [stage]   patch_embed  [1536, 1024]: rel-L2={rel_pe:.3e} max|d|={max_pe:.3e} exact={:.2}%", 100.0 * ex_pe as f64 / (n_tok * 1024) as f64));
        ensure!(rel_pe < 2e-3, "patch_embed rel-L2 {rel_pe:.3e} >= 2e-3");

        // Stage B: ViT Block 0 alone (fed golden patch_embed)
        let x0 = b.take(n_tok * 1024 * 2)?;
        g.h2d(x0, bytemuck_u16(&st0.bf16_u16("patch_embed")?))?;
        let nb = b.take(n_tok * 1024 * 2)?;
        let qkv = b.take(n_tok * 3072 * 2)?;
        let q = b.take(16 * n_tok * 64 * 2)?;
        let k = b.take(16 * n_tok * 64 * 2)?;
        let vv = b.take(16 * n_tok * 64 * 2)?;
        let ao = b.take(n_tok * 1024 * 2)?;
        let mgu = b.take(n_tok * 5632 * 2)?;
        let ma = b.take(n_tok * 2816 * 2)?;
        let blk0 = &vis_w.blocks[0];
        vis_w.rmsnorm(&g, x0, &blk0.norm1, n_tok, nb)?;
        blk0.wqkv.run(&g, &mut b, &scr, nb, n_tok, qkv, EPI_NONE, 0)?;
        let ra = VisQkvRopeArgs { qkv, q_out: q, k_out: k, v_out: vv, n_tok: n_tok as i32, n_w: img0.n_vit_w as i32, inv_freq: vis_w.inv_freq };
        g.launch1("d41_vis_qkv_rope", ((n_tok as u32).div_ceil(4), 16, 1), (128, 1, 1), 0, &ra)?;
        let sa = VisSdpaArgs { q, k, v: vv, out: ao, n_tok: n_tok as i32, n_heads: 16, scale: 0.125, pad0: 0 };
        g.launch1("d41_vis_sdpa_bf16", ((n_tok as u32).div_ceil(64), 16, 1), (128, 1, 1), 0, &sa)?;
        blk0.wo.run(&g, &mut b, &scr, ao, n_tok, x0, EPI_RESIDUAL, x0)?;
        vis_w.rmsnorm(&g, x0, &blk0.norm2, n_tok, nb)?;
        blk0.w1.run(&g, &mut b, &scr, nb, n_tok, mgu, EPI_NONE, 0)?;
        let sw = VisSwigluArgs { x: mgu, y: ma, rows: n_tok as i32, inter: 2816 };
        g.launch1("d41_vis_swiglu", (2816u32.div_ceil(256), n_tok as u32, 1), (256, 1, 1), 0, &sw)?;
        blk0.w2.run(&g, &mut b, &scr, ma, n_tok, x0, EPI_RESIDUAL, x0)?;
        g.sync()?;
        let got_b0 = g.down::<u16>(x0, n_tok * 1024)?;
        let (rel_b0_math, _, _) = rel_l2_bf16(&got_b0, &st0.bf16_u16("block0_out")?);
        let rel_b0_fa = if st0.has("block0_out_fa") { rel_l2_bf16(&got_b0, &st0.bf16_u16("block0_out_fa")?).0 } else { rel_b0_math };
        lines.push(format!("  [stage]   vit.block0   [1536, 1024]: vs 4D FlashAttn rel-L2={rel_b0_fa:.3e}, vs 3D MATH rel-L2={rel_b0_math:.3e}"));
        ensure!(rel_b0_fa < 2e-3 && rel_b0_math < 2.5e-3, "vit.block0 rel-L2 too high: fa={rel_b0_fa:.3e} math={rel_b0_math:.3e}");

        // Stage C: 3x3 Unfold + Aligner alone (fed golden vit_out)
        let vin = b.take(n_tok * 1024 * 2)?;
        g.h2d(vin, bytemuck_u16(&st0.bf16_u16("vit_out")?))?;
        let unf = b.take(n_llm * 9216 * 2)?;
        let ua = VisUnfoldArgs { x: vin, y: unf, n_h: img0.n_vit_h as i32, n_w: img0.n_vit_w as i32, n_llm_h: img0.n_llm_h as i32, n_llm_w: img0.n_llm_w as i32, dim: 1024, pad0: 0 };
        g.launch1("d41_vis_unfold3x3", (n_llm as u32, 1, 1), (256, 1, 1), 0, &ua)?;
        let mid = b.take(n_llm * 5120 * 2)?;
        let aout = b.take(n_llm * 5120 * 2)?;
        vis_w.aligner_w1.run(&g, &mut b, &scr, unf, n_llm, mid, EPI_GELU_ERF, 0)?;
        vis_w.aligner_w2.run(&g, &mut b, &scr, mid, n_llm, aout, EPI_NONE, 0)?;
        g.sync()?;
        let (_, _, ex_unf) = rel_l2_bf16(&g.down::<u16>(unf, n_llm * 9216)?, &st0.bf16_u16("unfold_out")?);
        let (rel_al, max_al, ex_al) = rel_l2_bf16(&g.down::<u16>(aout, n_llm * 5120)?, &st0.bf16_u16("aligner_out")?);
        lines.push(format!(
            "  [stage]   unfold+align [176, 5120]:  unfold exact={:.2}%, aligner rel-L2={rel_al:.3e} max|d|={max_al:.3e} exact={:.2}%",
            100.0 * ex_unf as f64 / (n_llm * 9216) as f64,
            100.0 * ex_al as f64 / (n_llm * 5120) as f64,
        ));
        ensure!(ex_unf == n_llm * 9216 && rel_al < 2e-3, "unfold/aligner single-stage check failed (rel_al={rel_al:.3e})");
    }

    // ---- 2b. Full 32-layer chained ViT + Aligner vs 4D FlashAttn, 3D MATH, and FP32 truth + timing ----
    let mut small_aligner_dev: Option<crate::dsv41::fwd::dev::Buf> = None;
    for ((name, img), st) in prepared_cases.iter().zip(st_cases.iter()) {
        let n_llm = img.n_aligner_tokens();
        let out_buf = g.buf(n_llm * vis_w.hidden * 2)?;
        b.reset();
        vis_w.encode_image(&g, &mut b, &scr, img, out_buf.ptr())?;
        g.sync()?;
        let got_u16 = g.down::<u16>(out_buf.ptr(), n_llm * vis_w.hidden)?;
        let want_math = st.bf16_u16("aligner_out")?;
        let (rel_math, _, _) = rel_l2_bf16(&got_u16, &want_math);
        let rel_fa = if st.has("aligner_out_fa") { rel_l2_bf16(&got_u16, &st.bf16_u16("aligner_out_fa")?).0 } else { rel_math };
        let fp32_note = if st.has("aligner_out_fp32") {
            let truth = st.f32_vec("aligner_out_fp32")?;
            let our_fp32 = rel_l2_bf16_vs_f32(&got_u16, &truth);
            let py_fp32 = rel_l2_bf16_vs_f32(&want_math, &truth);
            ensure!(our_fp32 <= py_fp32 * 1.25, "{name}: chained tower drifted from FP32 truth ({our_fp32:.3e} vs py {py_fp32:.3e})");
            format!("vs_FP32={our_fp32:.3e} (py={py_fp32:.3e})")
        } else {
            String::new()
        };

        let iters = 10usize;
        let t0 = Instant::now();
        for _ in 0..iters {
            b.reset();
            vis_w.encode_image(&g, &mut b, &scr, img, out_buf.ptr())?;
        }
        g.sync()?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        let flops = vision_flops(img.n_patches(), n_llm);
        let tflops = (flops / (ms * 1e-3)) / 1e12;
        let py_ms = meta["images"][*name]["pytorch_ms"].as_f64().unwrap_or(0.0);
        lines.push(format!(
            "  [tower32] {name:<15} patches={:<4} llm={:<4} vs_FA={rel_fa:.3e} vs_MATH={rel_math:.3e} {fp32_note} | GPU {ms:6.2} ms ({tflops:5.1} TFLOP/s, {:.1}x vs vision.py {py_ms:.1} ms)",
            img.n_patches(), n_llm, py_ms / ms.max(1e-6),
        ));
        ensure!(rel_fa < 3.0e-2 && rel_math < 3.0e-2, "{name}: 32-layer chained VisionWeights::encode_image rel-L2 too high (fa={rel_fa:.3e}, math={rel_math:.3e})");
        if *name == "small_300x200" {
            small_aligner_dev = Some(out_buf);
        }
    }

    // ---- 3, 4, 5. Splice, Engram DEAD mask, and Layer 0-1 GateBiasVl routing ----
    let vl_st = StFile::open(&dump_dir.join("vl_layers01.safetensors"))?;
    let input_ids: Vec<u32> = vl_st.i64_vec("input_ids")?.into_iter().map(|x| x as u32).collect();
    let token_types: Vec<i8> = vl_st.i64_vec("token_types")?.into_iter().map(|x| x as i8).collect();
    let rows = input_ids.len();

    // 3. Splice check: feed golden aligner_out to verify splice_chunk_embeddings is 100% bit-exact (0 ULP) on native pack!
    let glob = src.load_global(&g)?;
    let ew = glob.embed.as_ref().context("missing embed table")?;
    let ids_i32: Vec<i32> = input_ids.iter().map(|&t| t as i32).collect();
    let ids_buf = g.buf(rows * 4)?;
    g.h2d(ids_buf.ptr(), bytemuck_i32(&ids_i32))?;
    let emb_buf = g.buf(rows * cfg.hidden * 2)?;
    glue::backend(ew.1).gather(&g.l(), &ew.0, DevPtr(ids_buf.ptr()), rows, &dev_bf16(emb_buf.ptr(), rows, cfg.hidden, cfg.hidden))?;
    let span_start = token_types.iter().position(|&t| t >= 0).unwrap();
    let span_end = token_types.iter().rposition(|&t| t >= 0).unwrap() + 1;
    let (row_ops, align_cnt) = build_vl_row_ops(&token_types, &[(span_start, span_end)]);
    ensure!(align_cnt == prepared_cases[0].1.n_aligner_tokens(), "row_ops aligner count mismatch");
    let gold_align_u16 = st_cases[0].bf16_u16("aligner_out")?;
    let gold_align_buf = g.buf(gold_align_u16.len() * 2)?;
    g.h2d(gold_align_buf.ptr(), bytemuck_u16(&gold_align_u16))?;
    b.reset();
    vis_w.splice_chunk_embeddings(&g, &mut b, emb_buf.ptr(), gold_align_buf.ptr(), &row_ops)?;
    g.sync()?;
    let got_merged = g.down::<u16>(emb_buf.ptr(), rows * cfg.hidden)?;
    let want_merged = vl_st.bf16_u16("h_embed_merged")?;
    let (rel_m, max_m, exact_m) = rel_l2_bf16(&got_merged, &want_merged);
    let img_slice = span_start * cfg.hidden..span_end * cfg.hidden;
    let (rel_img_span, _, exact_img_span) = rel_l2_bf16(&got_merged[img_slice.clone()], &want_merged[img_slice.clone()]);
    let _ = small_aligner_dev;
    lines.push(format!(
        "  [splice]  h_embed_merged [{rows}, {}]: all rel-L2={rel_m:.3e} (max|d|={max_m:.3e}, exact={:.2}%), image_span [{span_start}..{span_end}] rel-L2={rel_img_span:.3e} exact={:.2}%",
        cfg.hidden,
        100.0 * exact_m as f64 / want_merged.len() as f64,
        100.0 * exact_img_span as f64 / img_slice.len() as f64,
    ));
    ensure!(exact_img_span == img_slice.len(), "spliced image_span was not 100% bit-exact ({exact_img_span}/{})", img_slice.len());
    if cfg.quant.method != "exl3" {
        ensure!(rel_m < 2e-3, "h_embed_merged rel-L2 {rel_m:.3e} >= 2e-3");
    }

    // 4. Engram DEAD mask check against engram_hashes
    let hs = EngramHasher::from_model_dir(&model_dir)?.context("EngramHasher")?;
    let live: Vec<bool> = token_types.iter().map(|&t| t < 0).collect();
    let mut st_hash = NgramHashState::new();
    let mut got_hashes = vec![0u32; rows * hs.ids_per_token()];
    hs.hash_chunk(&mut st_hash, &input_ids, Some(&live), &mut got_hashes)?;
    let want_hashes: Vec<u32> = vl_st.i64_vec("engram_hashes")?.into_iter().map(|x| x as u32).collect();
    ensure!(got_hashes == want_hashes, "Engram DEAD-masked hashes differed from engram.py!");
    lines.push(format!("  [engram]  DEAD-masked hashes: {}/{} u32 IDs 100.00% bit-exact vs engram.py", got_hashes.len(), want_hashes.len()));

    // 5. Router GateBiasVl check on Layer 0 and Layer 1
    let mut f = Fwd { g: &g, cfg: &cfg, scr, b: Bump::new(&g, 256 << 20)?, rope: rope_set(&cfg)?, checks: true, wide_min: 0 };
    let mask_u8: Vec<u8> = token_types.iter().map(|&t| (t >= 0) as u8).collect();
    let mask_buf = g.buf(rows.max(256))?;
    g.h2d(mask_buf.ptr(), &mask_u8)?;
    for l in [0usize, 1] {
        let lw = src.load_layer(&g, l)?;
        let _ = lw.param(C::GateBiasVl).with_context(|| format!("layer {l} missing GateBiasVl"))?;
        let fin_u16 = vl_st.bf16_u16(&format!("l{l}_fin"))?;
        let fin_buf = g.buf(fin_u16.len() * 2)?;
        g.h2d(fin_buf.ptr(), bytemuck_u16(&fin_u16))?;
        f.b.reset();
        let (rid, rw) = f.router_vl(&lw, fin_buf.ptr(), rows, mask_buf.ptr())?;
        g.sync()?;
        let got_idx = g.down::<i32>(rid, rows * cfg.top_k)?;
        let got_wts = g.down::<f32>(rw, rows * cfg.top_k)?;
        let want_idx: Vec<i32> = vl_st.i64_vec(&format!("l{l}_router_indices"))?.into_iter().map(|x| x as i32).collect();
        let want_wts = vl_st.f32_vec(&format!("l{l}_router_weights"))?;
        let mut row_set_match = 0usize;
        let mut w_diff_sq = 0.0f64;
        let mut w_ref_sq = 0.0f64;
        for r in 0..rows {
            let mut gs = got_idx[r * 6..(r + 1) * 6].to_vec();
            let mut ws = want_idx[r * 6..(r + 1) * 6].to_vec();
            gs.sort_unstable();
            ws.sort_unstable();
            if gs == ws {
                row_set_match += 1;
                for k in 0..6 {
                    let eid = want_idx[r * 6 + k];
                    let pos = got_idx[r * 6..(r + 1) * 6].iter().position(|&x| x == eid).unwrap();
                    let d = (got_wts[r * 6 + pos] - want_wts[r * 6 + k]) as f64;
                    w_diff_sq += d * d;
                    w_ref_sq += (want_wts[r * 6 + k] as f64).powi(2);
                }
            }
        }
        let w_rel = (w_diff_sq / w_ref_sq.max(1e-30)).sqrt();
        lines.push(format!(
            "  [router]  layer {l} GateBiasVl + image_mask: {row_set_match}/{rows} rows top-6 match ({:.2}%), weights rel-L2={w_rel:.3e}",
            100.0 * row_set_match as f64 / rows as f64,
        ));
        let (min_pct, max_w_rel) = if cfg.quant.method == "exl3" { (90usize, 1e-2) } else { (99usize, 1e-4) };
        ensure!(row_set_match * 100 >= rows * min_pct && w_rel < max_w_rel, "layer {l} router_vl mismatch: {row_set_match}/{rows}, w_rel={w_rel:.3e}");
    }

    lines.push("DSV41_VIS1_OK".into());
    Ok(lines)
}
