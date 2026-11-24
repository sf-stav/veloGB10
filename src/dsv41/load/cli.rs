//! DSV4.1 loader (leg LS): the two tool modes `--dsv41-shard-plan` (dry run) and `--dsv41-ship` (sync one node).
//! Result lines are direct prints (a tool entry point, like `dsv41::cli`).

use super::{dryrun, ship};
use anyhow::{bail, Context, Result};
use std::path::PathBuf;

fn val(args: &[String], f: &str) -> Option<String> { crate::arg_value(args, f).map(str::to_string) }

pub fn shard_plan(args: &[String]) -> Result<()> {
    let dir = PathBuf::from(val(args, "--model-dir").context("--model-dir <pack> is required")?);
    let worlds: Vec<usize> = match val(args, "--tp") { Some(w) => vec![w.parse().context("--tp")?], None => vec![2, 4, 8] };
    let lanes = val(args, "--max-batch").map(|s| s.parse()).transpose().context("--max-batch")?.unwrap_or(4);
    let seq = val(args, "--max-seq-len").map(|s| s.parse()).transpose().context("--max-seq-len")?;
    let cache = val(args, "--engram-row-cache-mib").map(|s| s.parse()).transpose().context("--engram-row-cache-mib")?;
    let (md, ok) = dryrun::run(&dir, &worlds, &ship::deal(), lanes, seq, cache)?;
    println!("{md}");
    if !ok { bail!("a proof failed"); }
    println!("DSV41_SHARD_PLAN_OK");
    Ok(())
}

/// `--dsv41-stage`: rank 0's shard as the local fast-reboot stage (what a serving boot builds), once, for a real pack.
pub fn stage_cli(args: &[String]) -> Result<()> {
    let dir = PathBuf::from(val(args, "--model-dir").context("--model-dir <pack> is required")?);
    let world: usize = val(args, "--tp").context("--tp W is required")?.parse().context("--tp")?;
    let c = ship::census_of(&dir)?;
    let o = ship::plan_opts(&c, world, &ship::deal())?;
    let sel = super::plan::select(&c, &o, 0)?;
    let (files, _) = ship::rank_files(&c, &dir, &o, &sel)?;
    let source = dir.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "pack".into());
    let mode = super::stage::Mode::parse(val(args, "--dsv41-load-cache").as_deref());
    let t0 = std::time::Instant::now();
    let s = super::stage::stage_or_skip(&files, &source, world, 0, mode, (0, world));
    println!("DSV41_STAGE_OK outcome={} bytes={} dir={} secs={:.1}", format!("{:?}", s.outcome).to_lowercase(), s.bytes,
             s.dir.as_ref().map_or("-".into(), |d| d.display().to_string()), t0.elapsed().as_secs_f64());
    Ok(())
}

pub fn ship_cli(args: &[String]) -> Result<()> {
    use crate::cluster;
    let dir = PathBuf::from(val(args, "--model-dir").context("--model-dir <pack> is required")?);
    let world: u32 = val(args, "--tp").context("--tp W is required")?.parse().context("--tp")?;
    let rank: usize = val(args, "--ship-rank").context("--ship-rank R is required")?.parse().context("--ship-rank")?;
    let node = val(args, "--nodes").context("--nodes <ip:port> is required")?;
    let addr: std::net::SocketAddr = if node.contains(':') { node.parse() } else { format!("{node}:29500").parse() }.context("--nodes")?;
    if rank == 0 || rank >= world as usize { bail!("--ship-rank must be 1..{}", world - 1); }
    let t0 = std::time::Instant::now();
    let ships = cluster::ship_plans_for(&dir, world, &[rank])?;
    println!("planned and hashed rank {rank} of {world}: {} files, {:.2} GB in {:.1}s", ships[rank].artifacts.len(),
             ships[rank].artifacts.iter().map(|a| a.size).sum::<u64>() as f64 / 1e9, t0.elapsed().as_secs_f64());
    let mut cfg = crate::tp::TpConfig::from_opts();
    cfg.world = world;
    cfg.sync_only = true;
    let t1 = std::time::Instant::now();
    let (n, bytes, total) = cluster::sync_one_rank(&dir, addr, rank as i32, &ships[rank], &cfg)?;
    println!("synced {addr} in {:.1}s", t1.elapsed().as_secs_f64());
    println!("DSV41_SHIP_OK blobs={n} bytes={bytes} total={total}");
    Ok(())
}
