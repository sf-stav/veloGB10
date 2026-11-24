//! DSV4.1 (leg I0): the ONE mode hook for every `--dsv41-*` / `--probe-dsv41-*` mode. main.rs calls
//! `dispatch` once (right after the option registry is filled); a later leg adds its mode by filling
//! the arm of its RESERVED flag here (the flag already exists in src/opts_table.rs), never by
//! editing main.rs. These are tool entry points: their result lines are direct prints (counted in
//! the logq per-file baseline for this file); progress goes through `boot_report::Progress`.

use crate::dsv41::census::{report, Census};
use crate::dsv41::config;
use crate::dsv41::{dump, gpu, load, rankmap, slice, state};
use crate::opts;
use anyhow::{bail, Context, Result};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// The reserved modes of later legs: (flag, leg, what it will do).
pub const RESERVED: &[(&str, &str, &str)] = &[
    ("probe-dsv41-engram-hash", "ENGa", "Engram row ids vs engram.py on a token file"),
    ("probe-dsv41-oracle", "OA / OM / OF / OD", "a CPU oracle part (--dsv41-oracle-part) against REFb dumps"),
    ("probe-dsv41-kernel", "KE / KN1 / KN2 / KC / KI / KA / KM", "one kernel set (--dsv41-kernel-set) against its host reference"),
    ("probe-dsv41-slice-model", "SL", "the GPU slice model, every layer class against the oracles"),
    ("probe-dsv41-ced", "CE", "CED prefill vs the reference CED for every chunking"),
    ("probe-dsv41-dspark", "DS", "DSpark on the GPU: losslessness, sampled exactness, acceptance"),
    ("dsv41-encode-check", "ENC", "the chat encoding against encoding.py on a corpus"),
];

fn flag_on(name: &str) -> bool { opts::lookup(name).map_or(false, opts::on) }
fn flag_val(name: &str) -> Option<String> { opts::lookup(name).and_then(opts::get) }

/// Run a dsv41 mode if the command line names one; returns the process exit code.
pub fn dispatch(args: &[String]) -> Option<i32> {
    let run = |r: Result<()>, tag: &str| -> i32 {
        match r { Ok(()) => 0, Err(e) => { eprintln!("{tag}_FAIL: {e:#}"); 1 } }
    };
    if flag_on("probe-dsv41-abi") { return Some(run(probe_abi(), "DSV41_ABI")); }
    if flag_on("probe-dsv41-xchg") { return Some(run(crate::dsv41::tp::xprobe::run().map(|l| print_lines(&l)), "DSV41_XCHG")); }   // leg PX2
    if flag_on("dsv41-shard-plan") { return Some(run(load::cli::shard_plan(args), "DSV41_SHARD_PLAN")); }
    if flag_on("dsv41-stage") { return Some(run(load::cli::stage_cli(args), "DSV41_STAGE")); }
    if flag_on("dsv41-ship") { return Some(run(load::cli::ship_cli(args), "DSV41_SHIP")); }
    if flag_on("probe-dsv41-kernel") && flag_val("dsv41-kernel-set").as_deref() == Some("attn") {
        return Some(run(probe_kernel_attn(), "DSV41_KERNEL_ATTN"));
    }
    if flag_on("dsv41-census") { return Some(run(census_cli(args), "DSV41_CENSUS")); }
    if flag_on("dsv41-slice-pack") { return Some(run(slice_cli(args), "DSV41_SLICE")); }
    if let Some(d) = flag_val("dsv41-dump-ls") { return Some(run(dump_ls(Path::new(&d), flag_val("dsv41-dump-ref").as_deref().map(Path::new)), "DSV41_DUMP")); }
    if let Some(d) = flag_val("dsv41-dump-demo") {
        return Some(run(dump::write_demo(Path::new(&d)).map(|p| println!("{}\nDSV41_DUMP_DEMO_OK", p.display())), "DSV41_DUMP_DEMO"));
    }
    if flag_on("probe-dsv41-kernel") && flag_val("dsv41-kernel-set").as_deref() == Some("kv") {
        return Some(run(kv_probe(), "DSV41_KC"));
    }
    if flag_on("probe-dsv41-kernel") && flag_val("dsv41-kernel-set").as_deref() == Some("fp8") {
        return Some(run(model_dir(args).and_then(|d| crate::dsv41::weights::native::fp8_probe::run(&d)).map(|l| l.iter().for_each(|s| println!("{s}"))), "DSV41_KN1_FP8"));
    }
    if flag_on("probe-dsv41-oracle") && matches!(flag_val("dsv41-oracle-part").as_deref(), Some("moe" | "engram")) {
        return Some(run(crate::dsv41::cpu::moe::oracle_cli::run(args), "DSV41_ORACLE_OF"));
    }
    if flag_on("probe-dsv41-oracle") && flag_val("dsv41-oracle-part").as_deref() == Some("mhc") {
        return Some(run(oracle_mhc(args), "DSV41_ORACLE_MHC"));
    }
    if flag_on("probe-dsv41-kernel") && matches!(flag_val("dsv41-kernel-set").as_deref(), Some("trellis") | Some("plain")) {
        return Some(run(crate::dsv41::weights::exl3::probe::run(args), "DSV41_KE"));   // leg KE
    }
    if flag_on("probe-dsv41-kernel") && flag_val("dsv41-kernel-set").as_deref() == Some("mxfp4") {
        // KN2: native MXFP4 experts against the host reference, batch invariance, byte floor
        return Some(run(crate::dsv41::weights::native::moe::probe::run(args).map(|ls| for l in ls { println!("{l}") }), "DSV41_KERNEL_MXFP4"));
    }
    if flag_on("probe-dsv41-tp") { return Some(run(crate::dsv41::tp::launch::probe_cli(args), "DSV41_TP")); }   // leg TPC
    if flag_on("probe-dsv41-engram-io") { return Some(run(crate::dsv41::engram::io_probe::run_cli(args), "DSV41_ENGRAM_IO")); }
    if flag_on("probe-dsv41-kernel") && flag_val("dsv41-kernel-set").as_deref() == Some("indexer") {
        return Some(run(probe_indexer(args), "DSV41_KERNEL_INDEXER"));   // leg KI
    }
    if flag_on("probe-dsv41-oracle") && flag_val("dsv41-oracle-part").as_deref() == Some("dspark") { return Some(run(crate::dsv41::cpu::dspark::cli(args), "DSV41_ORACLE_DSPARK")); }
    if flag_on("probe-dsv41-oracle") && flag_val("dsv41-oracle-part").as_deref() == Some("attn") {
        return Some(run(crate::dsv41::cpu::attn::probe_cli(args), "DSV41_ORACLE_ATTN"));
    }
    if flag_on("probe-dsv41-slice-model") { return Some(run(crate::dsv41::fwd::run(args).map(|_| ()), "DSV41_SL")); }   // leg SL (prints as it goes)
    if flag_on("probe-dsv41-dspark") { return Some(run(crate::dsv41::gpu::dspark::gate::run(args).map(|_| ()), "DSV41_DSPARK")); }   // leg DS (prints as it goes)
    if flag_on("probe-dsv41-ced") { return Some(run(crate::dsv41::ced::run(args).map(|_| ()), "DSV41_CE")); }   // leg CE (prints as it goes)
    if flag_on("probe-dsv41-vision") { return Some(run(crate::dsv41::vision::probe::run(args).map(|l| print_lines(&l)), "DSV41_VIS1")); }   // leg VIS1 / M4
    if flag_on("probe-dsv41-q1") {   // leg Q1: our EXL3 decode vs exllamav3's own decode, bit for bit
        let fx = flag_val("dsv41-kernel-fixture").filter(|s| s != "unset");
        return Some(run(model_dir(args).and_then(|d| match fx {
            Some(f) => crate::dsv41::weights::exl3::q1_oracle::run(&d, Path::new(&f)),
            None => bail!("--probe-dsv41-q1 needs --dsv41-kernel-fixture <oracle dir from tools/dsv41_q1/oracle_gen.py>"),
        }), "DSV41_Q1"));
    }
    for (f, leg, what) in RESERVED {
        if flag_on(f) {
            eprintln!("--{f} is RESERVED for leg {leg} ({what}) and is not implemented in this build");
            return Some(2);
        }
    }
    None
}

/// `--probe-dsv41-kernel --dsv41-kernel-set kv --dsv41-dump-ref <fixture dir>` (leg KC; fixture: tools/dsv41_ref/kc_fixture.py).
fn kv_probe() -> Result<()> {
    let d = flag_val("dsv41-dump-ref").context("--dsv41-dump-ref <kc fixture dir> is required (tools/dsv41_ref/kc_fixture.py --out DIR)")?;
    for l in gpu::kv::probe::run(Path::new(&d))? { println!("{l}"); }
    Ok(())
}

fn print_lines(lines: &[String]) {
    for l in lines { println!("{l}"); }
}

/// Leg KI: `--probe-dsv41-kernel --dsv41-kernel-set indexer [--model-dir PACK] [--dsv41-dump-ref FIXTURE_DIR]
/// [--max-seq-len N]`: the indexer gates, the REFb fixture check, the timing table to N (default 1M).
fn probe_indexer(args: &[String]) -> Result<()> {
    let cfg = match crate::arg_value(args, "--model-dir") { Some(d) => Some(config::Dsv41Config::load(Path::new(d))?), None => None };
    let fixture = flag_val("dsv41-dump-ref").map(PathBuf::from);
    let max_ctx: usize = opts::get(crate::opt!("max-seq-len")).map(|s| s.parse()).transpose().context("--max-seq-len")?.unwrap_or(1 << 20);
    for l in gpu::indexer::probe::run(cfg.as_ref(), fixture.as_deref(), max_ctx, true)? { println!("{l}"); }
    Ok(())
}

fn probe_abi() -> Result<()> {
    print_lines(&gpu::abi::probe()?);
    Ok(())
}

/// OM: `--probe-dsv41-oracle --dsv41-oracle-part mhc --model-dir <pack> --dsv41-dump-ref <REFb dump> [--layers 0-39]`
/// (the mHC / norm / router oracle against REFb's dumps; RESULT: DSV41_ORACLE_MHC_OK).
fn oracle_mhc(args: &[String]) -> Result<()> {
    use crate::dsv41::cpu::mhc::check;
    let dump = flag_val("dsv41-dump-ref").context("--dsv41-dump-ref <REFb dump dir> is required")?;
    let mut o = check::CheckOpts::default();
    if let Some(l) = crate::arg_value(args, "--layers") { o.layers = Some(slice::parse_layers(l)?); }
    let r = check::run(&model_dir(args)?, Path::new(&dump), &o)?;
    let mut lines = r.lines;
    if r.failures.is_empty() { lines.push("DSV41_ORACLE_MHC_OK".into()); }
    print_lines(&lines);
    if !r.failures.is_empty() { bail!("{} bound(s) failed (listed above)", r.failures.len()); }
    Ok(())
}

/// `--probe-dsv41-kernel --dsv41-kernel-set attn [--dsv41-kernel-fixture DIR[,DIR]] [--dsv41-kernel-timing 4k,128k,1m]` (leg KA)
fn probe_kernel_attn() -> Result<()> {
    print_lines(&gpu::attn::probe::run(flag_val("dsv41-kernel-fixture").as_deref(), flag_val("dsv41-kernel-timing").as_deref())?);
    Ok(())
}

fn model_dir(args: &[String]) -> Result<PathBuf> {
    crate::arg_value(args, "--model-dir").map(PathBuf::from).context("--model-dir <pack> is required")
}

fn build_census(dir: &Path) -> Result<Census> {
    let mut p: Option<crate::boot_report::Progress> = None;
    let c = Census::build_with(dir, &mut |done, total| {
        let pr = p.get_or_insert_with(|| load::progress(load::Phase::Headers, None, total));
        pr.set(done);
    })?;
    if let Some(pr) = p { pr.finish(); }
    Ok(c)
}

/// `--dsv41-census --model-dir A [--dsv41-census-also B] [--dsv41-census-out FILE] [--max-batch N] [--max-seq-len N]`
fn census_cli(args: &[String]) -> Result<()> {
    let a = model_dir(args)?;
    let also = flag_val("dsv41-census-also").map(PathBuf::from);
    let lanes: usize = crate::arg_value(args, "--max-batch").map(str::parse).transpose().context("--max-batch")?.unwrap_or(4);
    let mut packs = vec![build_census(&a)?];
    if let Some(b) = &also { packs.push(build_census(b)?); }
    let seq: usize = opts::get(crate::opt!("max-seq-len")).map(|s| s.parse()).transpose().context("--max-seq-len")?.unwrap_or(packs[0].cfg.max_pos);
    let mut md = String::new();
    let _ = writeln!(md, "# DSV4.1 census\n");
    let _ = writeln!(md, "Generated by `gb10_inference {}` (commit {}, kernel build {}) on {} with `{}`.\n",
                     env!("CARGO_PKG_VERSION"), option_env!("TUNE_GIT_COMMIT").unwrap_or("unknown").get(..8).unwrap_or("unknown"),
                     env!("KERNEL_BUILD_ID"), chrono::Utc::now().format("%Y-%m-%d %H:%M UTC"), args.join(" "));
    let _ = writeln!(md, "| pack | quantization | tensors | index keys | bytes | index total_size | census = index | problems |\n|---|---|---:|---:|---:|---:|---|---:|");
    let mut ok = true;
    for c in &packs {
        let eq = c.index_keys == c.entries.len() && c.index_total == Some(c.total_bytes()) && c.problems.is_empty();
        ok &= eq;
        let _ = writeln!(md, "| `{}` | {} | {} | {} | {} | {} | **{}** | {} |", c.dir.display(), c.cfg.quant.method, c.entries.len(), c.index_keys,
                         c.total_bytes(), c.index_total.map_or("-".into(), |t| t.to_string()), if eq { "EQUAL" } else { "NOT EQUAL" }, c.problems.len());
        println!("{}: {} tensors = {} index keys, {} bytes = index total {:?}: {}", c.dir.display(), c.entries.len(), c.index_keys,
                 c.total_bytes(), c.index_total, if eq { "EQUAL" } else { "NOT EQUAL" });
        for p in c.problems.iter().take(10) { println!("  problem: {p}"); }
    }
    // ---- config ----
    let cfg = &packs[0].cfg;
    let facts = match cfg.validate() {
        Ok(f) => f,
        Err(e) => { ok = false; println!("config validation FAILED: {e:#}"); Vec::new() }
    };
    let _ = writeln!(md, "\n## Config\n\n{}\n\n`validate()`: {} invariants hold{}.\n", cfg.summary(), facts.len(),
                     if facts.is_empty() { " — VALIDATION FAILED" } else { "" });
    for f in facts.iter().filter(|f| !f.starts_with("layer ")) { let _ = writeln!(md, "- {f}"); }
    let _ = writeln!(md, "\nAttention wiring (each compressing layer's KV source and index source were checked to share its ratio):\n");
    let _ = writeln!(md, "| layers | ratio | KV source | top-k from index source | own indexer | masks with layer-{} candidates |\n|---|---:|---:|---:|---|---|",
                     cfg.candidate_source.map_or("-".into(), |c| c.to_string()));
    for (a0, a1, r, kv, ix, own, cand) in cfg.wiring() {
        let o = |x: Option<usize>| x.map_or("-".to_string(), |v| v.to_string());
        let _ = writeln!(md, "| {} | {r} | {} | {} | {} | {} |", if a0 == a1 { a0.to_string() } else { format!("{a0}-{a1}") }, o(kv), o(ix),
                         if own { "yes" } else { "-" }, if cand { "yes" } else { "-" });
    }
    for c in &packs {
        if !c.cfg.unknown_keys.is_empty() { let _ = writeln!(md, "- `{}`: config keys this parser does not know: {:?}", c.dir.display(), c.cfg.unknown_keys); }
    }
    if let Some(b) = &also {
        let d = config::diff_dirs(&a, b)?;
        let _ = writeln!(md, "\n### Differences between the two packs' config.json (`A | B`)\n");
        for l in &d { let _ = writeln!(md, "- `{l}`"); }
        // parameters stored in different dtypes by the two packs (precision notes for the kernel legs)
        let mut dd: std::collections::BTreeMap<(String, String, String), usize> = std::collections::BTreeMap::new();
        let key = |m: &crate::dsv41::census::Module| (m.class, m.layer, m.sub);
        let other: std::collections::HashMap<_, _> = packs[1].modules.iter().map(|m| (key(m), m)).collect();
        for m in &packs[0].modules {
            if let (crate::dsv41::census::LinearFormat::Param(da), Some(o)) = (&m.format, other.get(&key(m))) {
                if let crate::dsv41::census::LinearFormat::Param(db) = &o.format {
                    if da != db { *dd.entry((m.class.name().to_string(), da.clone(), db.clone())).or_default() += 1; }
                }
            }
        }
        if !dd.is_empty() {
            let _ = writeln!(md, "\nParameters the two packs store in different dtypes (`A | B`):\n");
            for ((c, a1, b1), n) in &dd { let _ = writeln!(md, "- {c}: {a1} | {b1} ({n} modules)"); }
        }
        let text_same = packs[0].cfg.layers == packs[1].cfg.layers && !d.iter().any(|l| l.starts_with("text_config") || l.starts_with("vision_config"));
        let _ = writeln!(md, "\ntext_config and vision_config identical: **{}**", if text_same { "yes" } else { "NO" });
        println!("config.json differences: {} path(s); text_config / vision_config identical: {}", d.len(), text_same);
        ok &= text_same;
    }
    for c in &packs {
        let refp = c.dir.join("inference/config.json");
        if refp.exists() {
            let bad = config::check_reference_args(&c.cfg, &refp)?;
            let _ = writeln!(md, "\nReference `inference/config.json` (ModelArgs names) vs the parsed text_config: {}",
                             if bad.is_empty() { "**every mapped field agrees**".to_string() } else { format!("**{} MISMATCHES**: {bad:?}", bad.len()) });
            println!("reference ModelArgs check ({}): {}", refp.display(), if bad.is_empty() { "agree" } else { "MISMATCH" });
            ok &= bad.is_empty();
        }
    }
    // ---- state and rank map ----
    let _ = writeln!(md, "\n{}", state::report(cfg, lanes, seq));
    let ln = state::LaneState::plan(cfg, seq).bytes();
    println!("state per lane at {seq} tokens: {:.3} GB (reference buffers {:.3} GB); {lanes} lanes {:.3} GB",
             ln.total() as f64 / 1e9, state::reference_bytes_per_lane(cfg, seq).total() as f64 / 1e9, (ln.total() * lanes as u64) as f64 / 1e9);
    let _ = writeln!(md, "\n## Rank map (src/dsv41/rankmap.rs)\n");
    for p in [rankmap::Posture::Tp2, rankmap::Posture::Tp4, rankmap::Posture::Tp8] {
        for l in rankmap::describe(cfg, p) { let _ = writeln!(md, "- {l}"); }
    }
    for (i, c) in packs.iter().enumerate() {
        let title = format!("Pack {}: {}", (b'A' + i as u8) as char, c.dir.file_name().map(|x| x.to_string_lossy().to_string()).unwrap_or_default());
        let _ = writeln!(md, "\n{}", report::section(c, &title));
        let plan = crate::dsv41::weights::WeightPlan::from_census(c);
        let _ = writeln!(md, "Weight plan (linears per backend): {}", plan.as_ref().map(|p| p.summary()).unwrap_or_else(|e| format!("FAILED: {e:#}")));
        println!("{}: weights {}; plan {}", c.dir.display(), c.weights_line(), plan.map(|p| p.summary()).unwrap_or_else(|e| format!("FAILED: {e:#}")));
    }
    if let Some(out) = flag_val("dsv41-census-out") {
        std::fs::write(&out, &md).with_context(|| format!("write {out}"))?;
        println!("report written: {out}");
    }
    if !ok { bail!("the census does not equal the index, or the config does not validate (see above)"); }
    println!("DSV41_CENSUS_OK");
    Ok(())
}

/// `--dsv41-slice-pack --model-dir X --layers 0,1,2 --out DIR [--dsv41-slice-engram-tables on] [--dsv41-slice-vision on]`
fn slice_cli(args: &[String]) -> Result<()> {
    let src = model_dir(args)?;
    let layers = slice::parse_layers(crate::arg_value(args, "--layers").context("--layers <list> is required")?)?;
    let out = PathBuf::from(crate::arg_value(args, "--out").context("--out <dir> is required")?);
    let o = slice::SliceOpts { layers, engram_tables: flag_on("dsv41-slice-engram-tables"), vision: flag_on("dsv41-slice-vision") };
    let mut p: Option<crate::boot_report::Progress> = None;
    let r = slice::write_slice(&src, &out, &o, &mut |done, total| {
        p.get_or_insert_with(|| load::progress(load::Phase::Slice, None, total)).set(done);
    })?;
    if let Some(p) = p { p.finish(); }
    println!("slice {}: layers {:?}, {} tensors, {:.2} GB in {} files; re-read by the census: clean, spot-checked against the source",
             r.out.display(), o.layers, r.tensors, r.bytes as f64 / 1e9, r.files);
    for e in &r.excluded { println!("  left out: {e}"); }
    for m in &r.missing_deps { println!("  cross-layer read not in the slice: {m}"); }
    println!("DSV41_SLICE_OK");
    Ok(())
}

fn dump_ls(dir: &Path, reference: Option<&Path>) -> Result<()> {
    let d = dump::Dump::open(dir)?;
    let layout = if d.refb().is_some() { "REFb layout" } else { dump::FORMAT };
    println!("{}: {} mode {}, {} prompt ids, {} points", dir.display(), layout, d.meta["mode"].as_str().unwrap_or("?"), d.prompt_ids().len(), d.points.len());
    for (l, p, dt, sh) in &d.points { println!("  {:40} {dt:8} {sh:?}", d.label(*l, p)); }
    if let Some(r) = reference {
        let rd = dump::Dump::open(r)?;
        for (f, v) in dump::compare(&d, &rd) {
            match v { Ok(x) => println!("  {f:40} rel-L2 {x:.3e}"), Err(e) => println!("  {f:40} NOT COMPARABLE: {e}") }
        }
    }
    println!("DSV41_DUMP_OK");
    Ok(())
}
