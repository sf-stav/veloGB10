//! DSV4 offline converter CLI — emits the engine-native artifact from a bundle.
//!   dsv4_convert --bundle <b> --out <dir> [--layers N] [--sharded --world 2]
//! Default: one full artifact (`layer{N}.safetensors` + `trunk_top.safetensors` + manifest.json).
//! `--sharded --world 2`: PER-RANK self-contained shards at `<dir>/rank{0..world}/` — each carries
//! the replicated parts + that rank's expert slice only (~84 GB). The head ships rank1/ to the node
//! once; each node loads only its part. This is the load-speed lane's TP=2 design.
//! Resumable — re-run after interruption picks up where it left off (atomic per-file writes).
use gb10_inference::{dsv4_convert, dsv4_load};
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let grab = |flag: &str| -> Option<&str> {
        args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).map(|s| s.as_str())
    };
    let bundle = grab("--bundle").expect("--bundle <bundle-dir>");
    let out = grab("--out").expect("--out <artifact-dir>");
    let n_layers = grab("--layers").map(|s| s.parse::<usize>().expect("--layers N"));
    let cfg = dsv4_load::load_config(Path::new(bundle)).expect("load_config");
    let n = n_layers.unwrap_or(cfg.n_layers);
    let t0 = std::time::Instant::now();
    if args.iter().any(|a| a == "--sharded") {
        let world: usize = grab("--world").map(|s| s.parse().expect("--world N")).unwrap_or(2);
        eprintln!("[convert] {bundle} → {out}/rank{{0..{world}}} ({n} layers, {world}-way expert shard)");
        dsv4_convert::write_artifact_sharded(Path::new(bundle), &cfg, Path::new(out), n, world).expect("write_artifact_sharded");
    } else {
        eprintln!("[convert] {bundle} → {out} ({n} layers, full)");
        dsv4_convert::write_artifact(Path::new(bundle), &cfg, Path::new(out), n).expect("write_artifact");
    }
    eprintln!("[convert] done in {:.1}s", t0.elapsed().as_secs_f64());
}

