//! dsv4_attn_replay — lane-3A/3C GPU oracle-replay driver (SWA/CSA/HCA trunk layers).
//!
//! GPU twin of `src/bin/dsv4_replay.rs` (same CLI contract):
//!   dsv4_attn_replay --bundle /mnt/models/DeepSeek-V4-Flash-DSpark \
//!       --piece {swa|csa|hca} --in <dir-with-{ids,pre.x,dec*.x}.npy> \
//!       --out <dir-to-write-{pre.y,dec*.y,attn_out,...}.npy>
//!
//! Reads oracle input arrays (`.npy`, exported by `scripts/dsv4_diff.py export`), runs
//! the kind-dispatched `dsv4_attn` GPU layer assembly (hc_pre → norm → attn → hc_post →
//! hc_pre → norm → router+MoE → hc_post) on `dsv4_load`-loaded weights, and writes
//! output arrays under the oracle's key names for `scripts/dsv4_diff.py diff`.
//!
//! Cache sizing: `max_seq_len` is derived from the oracle's `kv_cache.shape[0]`
//! (CSA: (rows-window)*4; HCA: (rows-window)*128; SWA: window-only). For the long8k
//! and yarn profiles, `pre.y`/`pre.attn_out`/`pre.ffn_out` are written FULL — the diff
//! harness reads `meta.sample_rows` and subsamples the Rust arrays itself.

use anyhow::{bail, Context, Result};
use half::bf16;
use std::path::{Path, PathBuf};

use cudarc::driver::{CudaDevice, CudaSlice};

use gb10_inference::dsv4_attn::{Dsv4AttnRuntime, B, S};
use gb10_inference::dsv4_cpu;
use gb10_inference::dsv4_load::{self, NpyData};
use gb10_inference::gpu;

struct Args {
    bundle: PathBuf,
    piece: String,
    dir_in: PathBuf,
    dir_out: PathBuf,
}

fn parse_args() -> Result<Args> {
    let mut bundle = None;
    let mut piece = None;
    let mut dir_in = None;
    let mut dir_out = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--bundle" => bundle = Some(PathBuf::from(it.next().context("--bundle needs a value")?)),
            "--piece" => piece = Some(it.next().context("--piece needs a value")?),
            "--in" => dir_in = Some(PathBuf::from(it.next().context("--in needs a value")?)),
            "--out" => dir_out = Some(PathBuf::from(it.next().context("--out needs a value")?)),
            other => bail!("unknown argument: {other}"),
        }
    }
    Ok(Args {
        bundle: bundle.context("missing --bundle")?,
        piece: piece.context("missing --piece {swa|csa|hca}")?,
        dir_in: dir_in.context("missing --in")?,
        dir_out: dir_out.context("missing --out")?,
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

/// Echo a replay input bit-exact into the output dir (dsv4_replay's contract — the differ
/// gates inputs at 0.0; the echo proves the replay ran on the exported arrays).
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
        "dsv4_attn_replay: piece={} bundle={} in={} out={}",
        args.piece, args.bundle.display(), args.dir_in.display(), args.dir_out.display()
    );

    // ---- inputs (mirror dsv4_replay's parsing) ----
    let layer_id = read_i64(&args.dir_in, "meta.layer_id").map(|(_, v)| v[0] as usize).unwrap_or(match args.piece.as_str() {
        "swa" => 0,
        "csa" => 2,
        "hca" => 3,
        _ => bail!("unknown piece {:?} and no meta.layer_id", args.piece),
    });
    let kind = cfg.layer_kind(layer_id);
    let kind_str = match kind {
        dsv4_load::LayerKind::Swa => "swa",
        dsv4_load::LayerKind::Csa => "csa",
        dsv4_load::LayerKind::Hca => "hca",
    };
    anyhow::ensure!(
        kind_str == args.piece,
        "piece {:?} doesn't match layer {} kind {}",
        args.piece, layer_id, kind_str
    );
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
    eprintln!("loading trunk layer {layer_id} ({kind_str}) on GPU ...");

    // ---- max_seq_len: derived from the oracle's kv_cache shape (the piece sizes its
    // caches for the next pow2 above S+D; reading the oracle's shape is exact). ----
    let win = cfg.window_size;
    let max_seq_len = match kind {
        dsv4_load::LayerKind::Swa => win,
        dsv4_load::LayerKind::Csa | dsv4_load::LayerKind::Hca => {
            let (kc_shape, _) = read_f32(&args.dir_in, "kv_cache")
                .with_context(|| "reading oracle kv_cache.npy to size caches")?;
            let cache_rows = kc_shape[0];
            anyhow::ensure!(cache_rows > win, "kv_cache rows {cache_rows} <= window {win}");
            let ratio = match kind {
                dsv4_load::LayerKind::Csa => 4,
                dsv4_load::LayerKind::Hca => 128,
                _ => unreachable!(),
            };
            (cache_rows - win) * ratio
        }
    };
    eprintln!("  max_seq_len={max_seq_len} (kv_cache rows {} for {kind_str})", match kind {
        dsv4_load::LayerKind::Swa => win,
        _ => win + max_seq_len / match kind { dsv4_load::LayerKind::Csa => 4, _ => 128 },
    });

    // ---- GPU subsystem ----
    let dev = CudaDevice::new(0).context("CUDA device 0")?;
    let positions = s + d + 8; // dsv4_cpu::run_layer_piece's table sizing
    let rt = Dsv4AttnRuntime::new(&dev, kind, positions, &cfg).context("Dsv4AttnRuntime::new")?;
    let layer = rt.upload_layer(&args.bundle, &cfg, layer_id, 0, 1).context("upload_layer")?;
    let mut st = match kind {
        dsv4_load::LayerKind::Swa => rt.new_state_swa(&cfg, 320).context("new_state_swa")?,
        _ => rt.new_state(&cfg, &layer, max_seq_len, s).context("new_state")?,
    };
    let mut scratch = gpu::new_moe_grouped_scratch_raw(&dev, cfg.n_routed_experts, cfg.dim, cfg.moe_inter_dim, cfg.n_activated_experts, s.max(16), 16 * cfg.n_activated_experts);

    // ---- replay: prefill S + d decode steps ----
    eprintln!("running prefill S={s} + {d} decode steps on GPU ...");
    let to_dev = |v: &[f32]| -> Result<cudarc::driver::CudaSlice<bf16>> {
        let b: Vec<bf16> = v.iter().map(|&x| bf16::from_f32(x)).collect();
        Ok(dev.htod_sync_copy(&b)?)
    };
    let from_dev = |b: &cudarc::driver::CudaSlice<bf16>| -> Result<Vec<f32>> {
        Ok(dev.dtoh_sync_copy(b)?.iter().map(|v| v.to_f32()).collect())
    };
    let ids_i32: Vec<i32> = ids.iter().map(|&v| v as i32).collect();

    let mut out = dsv4_cpu::PieceOutputs::default();
    // prefill
    {
        let x_dev = to_dev(&pre_x)?;
        let ids_dev = dev.htod_sync_copy(&ids_i32[..s])?;
        let o = rt.block_forward::<B, S, CudaSlice<i32>, CudaSlice<u8>, CudaSlice<u32>>(&layer, &mut st, &mut scratch, &x_dev, s, 0, &ids_dev, &cfg)
            .context("block_forward prefill")?;
        dev.synchronize()?;
        out.push_f32("pre.y", &[s, cfg.hc_mult, cfg.dim], from_dev(&o.y)?);
        out.push_f32("pre.attn_out", &[s, cfg.dim], from_dev(&o.attn_out)?);
        out.push_f32("pre.ffn_out", &[s, cfg.dim], from_dev(&o.ffn_out)?);
        out.push_f32("pre.router_w", &[s, cfg.n_activated_experts], dev.dtoh_sync_copy(&o.router_w)?);
        let ri: Vec<i64> = dev.dtoh_sync_copy(&o.router_idx)?.iter().map(|&v| v as i64).collect();
        out.push_i64("pre.router_idx", &[s, cfg.n_activated_experts], ri);
        let ti: Vec<i64> = dev.dtoh_sync_copy(&o.topk_idx)?.iter().map(|&v| v as i64).collect();
        out.push_i64("pre.topk_idx", &[1, s, o.topk_t], ti);
        eprintln!("  prefill done (topk_t={})", o.topk_t);
    }
    // decode steps
    for (i, dx) in dec_xs.iter().enumerate() {
        let sp = s + i;
        let x_dev = to_dev(dx)?;
        let ids_dev = dev.htod_sync_copy(&ids_i32[sp..sp + 1])?;
        let o = rt.block_forward::<B, S, CudaSlice<i32>, CudaSlice<u8>, CudaSlice<u32>>(&layer, &mut st, &mut scratch, &x_dev, 1, sp, &ids_dev, &cfg)
            .with_context(|| format!("block_forward dec{i}"))?;
        dev.synchronize()?;
        out.push_f32(&format!("dec{i}.y"), &[1, cfg.hc_mult, cfg.dim], from_dev(&o.y)?);
        out.push_f32(&format!("dec{i}.attn_out"), &[1, cfg.dim], from_dev(&o.attn_out)?);
        // oracle saves dec router_w squeezed to [6], router_idx as [1,6] (run_layer_piece contract)
        out.push_f32(&format!("dec{i}.router_w"), &[cfg.n_activated_experts], dev.dtoh_sync_copy(&o.router_w)?);
        let ri: Vec<i64> = dev.dtoh_sync_copy(&o.router_idx)?.iter().map(|&v| v as i64).collect();
        out.push_i64(&format!("dec{i}.router_idx"), &[1, cfg.n_activated_experts], ri);
        let ti: Vec<i64> = dev.dtoh_sync_copy(&o.topk_idx)?.iter().map(|&v| v as i64).collect();
        out.push_i64(&format!("dec{i}.topk_idx"), &[1, 1, o.topk_t], ti);
        eprintln!("  dec{i} done (topk_t={})", o.topk_t);
    }
    // kv cache (the full unified buffer after the last decode — the oracle stores the
    // attention kv_cache [cache_rows, head_dim] which for CSA/HCA includes the comp tail).
    let kv_rows = match kind {
        dsv4_load::LayerKind::Swa => win,
        dsv4_load::LayerKind::Csa => win + max_seq_len / 4,
        dsv4_load::LayerKind::Hca => win + max_seq_len / 128,
    };
    out.push_f32("kv_cache", &[kv_rows, cfg.head_dim], from_dev(&st.kv_cache)?);

    write_outputs(&args.dir_out, &out)?;
    echo_i64(&args.dir_in, &args.dir_out, "ids")?;
    echo_f32(&args.dir_in, &args.dir_out, "pre.x")?;
    for i in 0..d {
        echo_f32(&args.dir_in, &args.dir_out, &format!("dec{i}.x"))?;
    }
    eprintln!("dsv4_attn_replay: done");
    Ok(())
}
