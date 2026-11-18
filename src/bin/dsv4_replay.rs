//! dsv4_replay — G1 npz replay driver (Lane B owns this file).
//!
//! Reads oracle input arrays (`.npy`, exported by `scripts/dsv4_diff.py export`),
//! runs the `dsv4_cpu` model on `dsv4_load`-loaded weights, and writes output
//! arrays (`.npy`) under the oracle's key names for `scripts/dsv4_diff.py diff`.
//!
//! CLI (contract with Lane C):
//!   dsv4_replay --bundle /mnt/models/DeepSeek-V4-Flash-DSpark \
//!       --piece {swa|csa|hca|dspark|head} --in <dir-with-{ids,pre.x,dec*.x}.npy> \
//!       --out <dir-to-write-{pre.y,dec*.y,attn_out,...}.npy> [--max-seq-len 8192]
//!
//! `--max-seq-len` sizes the attention/indexer KV caches (the mech oracle
//! profile ran with 8192 → CSA cache [128+2048, 512], HCA [128+64, 512]).

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

use gb10_inference::dsv4_cpu;
use gb10_inference::dsv4_load::{self, NpyData};

struct Args {
    bundle: PathBuf,
    piece: String,
    dir_in: PathBuf,
    dir_out: PathBuf,
    max_seq_len: usize,
}

fn parse_args() -> Result<Args> {
    let mut bundle = None;
    let mut piece = None;
    let mut dir_in = None;
    let mut dir_out = None;
    let mut max_seq_len = 8192usize;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--bundle" => bundle = Some(PathBuf::from(it.next().context("--bundle needs a value")?)),
            "--piece" => piece = Some(it.next().context("--piece needs a value")?),
            "--in" => dir_in = Some(PathBuf::from(it.next().context("--in needs a value")?)),
            "--out" => dir_out = Some(PathBuf::from(it.next().context("--out needs a value")?)),
            "--max-seq-len" => {
                max_seq_len = it.next().context("--max-seq-len needs a value")?.parse().context("--max-seq-len must be an integer")?
            }
            other => bail!("unknown argument: {other}"),
        }
    }
    Ok(Args {
        bundle: bundle.context("missing --bundle")?,
        piece: piece.context("missing --piece {swa|csa|hca|dspark|head}")?,
        dir_in: dir_in.context("missing --in")?,
        dir_out: dir_out.context("missing --out")?,
        max_seq_len,
    })
}

fn read_f32(dir: &Path, key: &str) -> Result<(Vec<usize>, Vec<f32>)> {
    let p = dir.join(format!("{key}.npy"));
    let (shape, data) = dsv4_load::read_npy(&p).with_context(|| format!("reading {}", p.display()))?;
    match data {
        NpyData::F32(v) => Ok((shape, v)),
        _ => bail!("{}: expected <f4 npy", p.display()),
    }
}

fn read_i64(dir: &Path, key: &str) -> Result<(Vec<usize>, Vec<i64>)> {
    let p = dir.join(format!("{key}.npy"));
    let (shape, data) = dsv4_load::read_npy(&p).with_context(|| format!("reading {}", p.display()))?;
    match data {
        NpyData::I64(v) => Ok((shape, v)),
        _ => bail!("{}: expected <i8 npy", p.display()),
    }
}

fn write_outputs(dir: &Path, outs: &dsv4_cpu::PieceOutputs) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    for (key, shape, data) in &outs.f32_arrays {
        dsv4_load::write_npy_f32(&dir.join(format!("{key}.npy")), shape, data)
            .with_context(|| format!("writing {key}.npy"))?;
        eprintln!("  wrote {key}.npy {:?} ({} f32)", shape, data.len());
    }
    for (key, shape, data) in &outs.i64_arrays {
        dsv4_load::write_npy_i64(&dir.join(format!("{key}.npy")), shape, data)
            .with_context(|| format!("writing {key}.npy"))?;
        eprintln!("  wrote {key}.npy {:?} ({} i64)", shape, data.len());
    }
    Ok(())
}

/// Echo a replay input bit-exact into the output dir — the differ requires the
/// full oracle key set (rule d); echoed inputs prove the replay ran on them.
fn echo_f32(dir_in: &Path, dir_out: &Path, key: &str) -> Result<()> {
    let (shape, data) = read_f32(dir_in, key)?;
    dsv4_load::write_npy_f32(&dir_out.join(format!("{key}.npy")), &shape, &data)
        .with_context(|| format!("echoing {key}.npy"))
}

fn echo_i64(dir_in: &Path, dir_out: &Path, key: &str) -> Result<()> {
    let (shape, data) = read_i64(dir_in, key)?;
    dsv4_load::write_npy_i64(&dir_out.join(format!("{key}.npy")), &shape, &data)
        .with_context(|| format!("echoing {key}.npy"))
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let cfg = dsv4_load::load_config(&args.bundle).context("load_config")?;
    eprintln!(
        "dsv4_replay: piece={} bundle={} in={} out={} max_seq_len={}",
        args.piece,
        args.bundle.display(),
        args.dir_in.display(),
        args.dir_out.display(),
        args.max_seq_len
    );
    match args.piece.as_str() {
        "swa" | "csa" | "hca" => {
            // layer id: meta.layer_id.npy if the exporter kept it, else the piece default
            let default_layer = match args.piece.as_str() {
                "swa" => 0usize,
                "csa" => 2,
                _ => 3,
            };
            let layer_id = read_i64(&args.dir_in, "meta.layer_id").map(|(_, v)| v[0] as usize).unwrap_or(default_layer);
            let (ids_shape, ids) = read_i64(&args.dir_in, "ids")?;
            let s_total = ids_shape.iter().product::<usize>();
            let (xshape, pre_x) = read_f32(&args.dir_in, "pre.x")?;
            anyhow::ensure!(xshape.len() == 3 && xshape[1] == cfg.hc_mult && xshape[2] == cfg.dim, "pre.x shape {xshape:?}");
            let s = xshape[0];
            let d = s_total - s;
            let mut dec_xs = Vec::with_capacity(d);
            for i in 0..d {
                let (dshape, dx) = read_f32(&args.dir_in, &format!("dec{i}.x"))?;
                anyhow::ensure!(dshape == vec![1, cfg.hc_mult, cfg.dim], "dec{i}.x shape {dshape:?}");
                dec_xs.push(dx);
            }
            eprintln!("loading trunk layer {layer_id} ({}) ...", args.piece);
            let layer = dsv4_load::load_layer(&args.bundle, &cfg, layer_id).context("load_layer")?;
            anyhow::ensure!(cfg.layer_kind(layer_id) == match args.piece.as_str() {
                "swa" => dsv4_load::LayerKind::Swa,
                "csa" => dsv4_load::LayerKind::Csa,
                _ => dsv4_load::LayerKind::Hca,
            }, "piece/layer kind mismatch for layer {layer_id}");
            let cpu = dsv4_cpu::cpu_layer_from_dsv4(layer, &cfg, cfg.layer_kind(layer_id)).context("cpu_layer_from_dsv4")?;
            eprintln!("running prefill S={s} + {d} decode steps ...");
            let outs = dsv4_cpu::run_layer_piece(&cfg, &cpu, &ids, &pre_x, &dec_xs, args.max_seq_len);
            write_outputs(&args.dir_out, &outs)?;
            echo_i64(&args.dir_in, &args.dir_out, "ids")?;
            echo_f32(&args.dir_in, &args.dir_out, "pre.x")?;
            for i in 0..d {
                echo_f32(&args.dir_in, &args.dir_out, &format!("dec{i}.x"))?;
            }
        }
        "dspark" => {
            let (_, warm) = read_f32(&args.dir_in, "warm.main_hidden")?;
            let (_, draft) = read_f32(&args.dir_in, "draft.main_hidden")?;
            let (_, real) = read_i64(&args.dir_in, "draft.real_token")?;
            eprintln!("loading 3 DSpark stages + trunk embed/head ...");
            let top = dsv4_load::load_trunk_top(&args.bundle, &cfg).context("load_trunk_top")?;
            let trunk = dsv4_cpu::trunk_top_from(top, &cfg).context("trunk_top_from")?;
            let stages: Vec<dsv4_cpu::CpuLayer> = (0..cfg.n_mtp_layers)
                .map(|s| {
                    let l = dsv4_load::load_mtp_stage(&args.bundle, &cfg, s).with_context(|| format!("load_mtp_stage {s}"))?;
                    dsv4_cpu::cpu_stage_from_dsv4(l, &cfg, s).with_context(|| format!("cpu_stage_from_dsv4 {s}"))
                })
                .collect::<Result<_>>()?;
            let stages: [dsv4_cpu::CpuLayer; 3] = stages.try_into().map_err(|_| anyhow::anyhow!("expected 3 stages"))?;
            eprintln!("running warm + draft ...");
            let outs = dsv4_cpu::run_dspark_piece(&cfg, &stages, &trunk.embed, &trunk.head, &warm, &draft, real[0]);
            write_outputs(&args.dir_out, &outs)?;
            echo_f32(&args.dir_in, &args.dir_out, "warm.main_hidden")?;
            echo_f32(&args.dir_in, &args.dir_out, "draft.main_hidden")?;
            echo_i64(&args.dir_in, &args.dir_out, "draft.real_token")?;
        }
        "head" => {
            let (_, x) = read_f32(&args.dir_in, "x")?;
            eprintln!("loading trunk top ...");
            let top = dsv4_load::load_trunk_top(&args.bundle, &cfg).context("load_trunk_top")?;
            let trunk = dsv4_cpu::trunk_top_from(top, &cfg).context("trunk_top_from")?;
            eprintln!("running head collapse ...");
            let outs = dsv4_cpu::run_head_piece(&cfg, &trunk.hc_head, &trunk.norm, &trunk.head, &x);
            write_outputs(&args.dir_out, &outs)?;
            echo_f32(&args.dir_in, &args.dir_out, "x")?;
        }
        other => bail!("unknown --piece {other} (want swa|csa|hca|dspark|head)"),
    }
    eprintln!("dsv4_replay: done");
    Ok(())
}
