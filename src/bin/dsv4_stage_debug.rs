//! dsv4_stage_debug — stage-by-stage dump of the layer-0 attention input path
//! on REAL weights + the oracle's pre.x, for localizing the G1 kv_cache/attn
//! divergence. Dumps each intermediate as npy under --out (default
//! /tmp/dsv4_stage/rust): collapsed, xn, act codes/scales for the shared GEMM
//! input, wq_a/wq_b/wkv GEMM outputs, qr, q post-rescale / post-RoPE, kv post
//! kv_norm / post-RoPE (pre-sim) / post-sim.

use anyhow::{Context, Result};
use gb10_inference::dsv4_cpu::*;
use gb10_inference::dsv4_load::{self, NpyData};
use std::path::{Path, PathBuf};

fn read_f32(dir: &Path, key: &str) -> Result<(Vec<usize>, Vec<f32>)> {
    let p = dir.join(format!("{key}.npy"));
    let (shape, data) = dsv4_load::read_npy(&p).with_context(|| format!("reading {}", p.display()))?;
    match data {
        NpyData::F32(v) => Ok((shape, v)),
        _ => anyhow::bail!("{}: expected <f4", p.display()),
    }
}

fn dump(out: &Path, key: &str, shape: &[usize], data: &[f32]) -> Result<()> {
    std::fs::create_dir_all(out)?;
    dsv4_load::write_npy_f32(&out.join(format!("{key}.npy")), shape, data)?;
    eprintln!("  wrote {key} {shape:?}");
    Ok(())
}

fn main() -> Result<()> {
    let mut bundle = PathBuf::from("/mnt/models/DeepSeek-V4-Flash-DSpark");
    let mut dir_in = PathBuf::from("/tmp/dsv4_in/swa");
    let mut dir_out = PathBuf::from("/tmp/dsv4_stage/rust");
    let mut layer_id = 0usize;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--bundle" => bundle = PathBuf::from(it.next().unwrap()),
            "--in" => dir_in = PathBuf::from(it.next().unwrap()),
            "--out" => dir_out = PathBuf::from(it.next().unwrap()),
            "--layer" => layer_id = it.next().unwrap().parse().unwrap(),
            other => anyhow::bail!("unknown arg {other}"),
        }
    }
    let cfg = dsv4_load::load_config(&bundle)?;
    let kind = cfg.layer_kind(layer_id);
    let layer = dsv4_load::load_layer(&bundle, &cfg, layer_id)?;
    let cpu = cpu_layer_from_dsv4(layer, &cfg, kind)?;
    let (xshape, pre_x) = read_f32(&dir_in, "pre.x")?;
    let (s, hc, dim) = (xshape[0], xshape[1], xshape[2]);
    assert_eq!(hc, cfg.hc_mult);
    assert_eq!(dim, cfg.dim);
    let rope = layer_rope_table(&cfg, kind, s + 8);
    let hd = cfg.head_dim;
    let rd = cfg.rope_head_dim;
    let nh = cfg.n_heads;

    // hc_pre collapse + attn_norm
    let (y, _posts, _combs) = hc_pre_all(&pre_x, s, &cpu.hc_attn, &cfg);
    dump(&dir_out, "collapsed", &[s, dim], &y)?;
    let xn = rms_norm(&y, s, dim, &cpu.attn_norm, cfg.norm_eps);
    dump(&dir_out, "xn", &[s, dim], &xn)?;

    // act_quant codes+scales for the shared GEMM input (group 128, ue8m0, §C.1)
    let (codes, scales) = act_quant_codes(&xn, s, dim, 128);
    dump(&dir_out, "act_codes_xn", &[s, dim], &codes)?;
    dump(&dir_out, "act_scales_xn", &[s, dim / 128], &scales)?;

    // q path
    let wqa_out = quant_gemm(&xn, s, dim, &cpu.attn.wq_a, cfg.q_lora_rank, 128);
    dump(&dir_out, "wqa_out", &[s, cfg.q_lora_rank], &wqa_out)?;
    let qr = rms_norm(&wqa_out, s, cfg.q_lora_rank, &cpu.attn.q_norm, cfg.norm_eps);
    dump(&dir_out, "qr", &[s, cfg.q_lora_rank], &qr)?;
    let mut q = quant_gemm(&qr, s, cfg.q_lora_rank, &cpu.attn.wq_b, nh * hd, 128);
    dump(&dir_out, "q_prerescale", &[s, nh * hd], &q)?;
    for i in 0..s * nh {
        let row = &mut q[i * hd..(i + 1) * hd];
        let mut sq: Vec<f32> = row.iter().map(|&v| bf(v * v)).collect();
        let ss = pairwise_sum(&mut sq);
        let mean = bf(ss / hd as f32);
        let arg = bf(mean + cfg.norm_eps);
        let r = bf(arg.sqrt().recip());
        for v in row.iter_mut() {
            *v = bf(*v * r);
        }
    }
    dump(&dir_out, "q_rescale", &[s, nh * hd], &q)?;
    {
        let rows = s * nh;
        let pos: Vec<usize> = (0..rows).map(|i| i / nh).collect();
        apply_rope(&mut q, rows, hd, &rope, &pos, false);
    }
    dump(&dir_out, "q_rope", &[s, nh * hd], &q)?;

    // kv path
    let wkv_out = quant_gemm(&xn, s, dim, &cpu.attn.wkv, hd, 128);
    dump(&dir_out, "wkv_out", &[s, hd], &wkv_out)?;
    let mut kv = rms_norm(&wkv_out, s, hd, &cpu.attn.kv_norm, cfg.norm_eps);
    dump(&dir_out, "kv_norm_out", &[s, hd], &kv)?;
    {
        let pos: Vec<usize> = (0..s).collect();
        apply_rope(&mut kv, s, hd, &rope, &pos, false);
    }
    dump(&dir_out, "kv_rope", &[s, hd], &kv)?;
    {
        let nope = hd - rd;
        let mut tmp = vec![0.0f32; s * nope];
        for i in 0..s {
            tmp[i * nope..(i + 1) * nope].copy_from_slice(&kv[i * hd..i * hd + nope]);
        }
        act_quant_sim(&mut tmp, s, nope, 64);
        for i in 0..s {
            kv[i * hd..i * hd + nope].copy_from_slice(&tmp[i * nope..(i + 1) * nope]);
        }
    }
    dump(&dir_out, "kv_sim", &[s, hd], &kv)?;
    eprintln!("dsv4_stage_debug: done");
    Ok(())
}
