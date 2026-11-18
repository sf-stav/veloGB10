//! GPU vision tower, embeddings only: for each image, the sha256 of the merged embedding rows the
//! tower hands to the scheduler (f32, little-endian bytes). Two builds that print the same hashes
//! for the same images compute bit-identical image embeddings (S-B9-REL-VISIONFIX gate 2).
//!
//!   vision_gpu_hash --model-dir <DIR> --data-url-file <FILE> [--data-url-file <FILE> ...] [--repeat N]
//!
//! Each file holds one `data:image/...;base64,...` URL. Every image is forwarded `--repeat` times
//! (default 2: the second pass runs on reused, non-zeroed scratch) in the order given, then the
//! whole list once more in reverse, so that every buffer size is reused after another image's.
//! Flags only; no environment variables.
use sha2::{Digest, Sha256};

fn main() {
    use base64::Engine;
    let argv: Vec<String> = std::env::args().collect();
    let one = |name: &str| argv.iter().position(|a| a == name).and_then(|i| argv.get(i + 1).cloned());
    let all = |name: &str| -> Vec<String> {
        argv.iter().enumerate().filter(|(_, a)| *a == name).filter_map(|(i, _)| argv.get(i + 1).cloned()).collect()
    };
    let md = one("--model-dir").expect("usage: vision_gpu_hash --model-dir <DIR> --data-url-file <FILE> [...] [--repeat N]");
    let files = all("--data-url-file");
    assert!(!files.is_empty(), "pass at least one --data-url-file <FILE>");
    let repeat: usize = one("--repeat").map(|s| s.parse().expect("--repeat N")).unwrap_or(2);

    let tower = gb10_inference::vision_tower::VisualTower::load(&md).expect("load tower");
    let dev = cudarc::driver::CudaDevice::new(0).expect("device");
    let mut gvt = gb10_inference::vision_gpu::GpuVisualTower::new(dev, &tower).expect("gpu tower");
    let pres: Vec<_> = files.iter().map(|f| {
        let url = std::fs::read_to_string(f).unwrap_or_else(|e| panic!("{f}: {e}"));
        let b64 = url.split("base64,").nth(1).expect("a base64 data url");
        let bytes = base64::engine::general_purpose::STANDARD.decode(b64.trim()).expect("b64");
        let img = image::load_from_memory(&bytes).expect("image").to_rgb8();
        let (w, h) = img.dimensions();
        (f.clone(), w, h, gb10_inference::vision_preproc::preprocess_image(h as usize, w as usize, img.as_raw(), &tower.preproc))
    }).collect();
    let mut order: Vec<usize> = (0..pres.len()).flat_map(|i| std::iter::repeat(i).take(repeat)).collect();
    order.extend((0..pres.len()).rev());
    let mut first: Vec<Option<String>> = vec![None; pres.len()];
    // the tower's buffer pool after every forward: it may grow only on the first pass of an image size
    let (mut pool_prev, mut grew_late): ((usize, usize), Vec<usize>) = (gvt.pool_stats(), Vec::new());
    for (pass, &i) in order.iter().enumerate() {
        let (f, w, h, pre) = &pres[i];
        let t = std::time::Instant::now();
        let (merged, _) = gvt.forward(&pre.pixel_values, pre.grid_h, pre.grid_w, false).expect("gpu forward");
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let mut hsh = Sha256::new();
        for v in &merged { hsh.update(v.to_le_bytes()); }
        let hex: String = hsh.finalize().iter().map(|b| format!("{b:02x}")).collect();
        let same = match &first[i] { None => { first[i] = Some(hex.clone()); "first" } Some(p) if *p == hex => "same", Some(_) => "DIFFERS FROM ITS FIRST PASS" };
        let name = std::path::Path::new(f).file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        let pool = gvt.pool_stats();
        if pool != pool_prev && same != "first" { grew_late.push(pass); }
        pool_prev = pool;
        println!("VISION_GPU_HASH pass {pass} {name} {w}x{h} grid {}x{} rows {} sha256 {hex} ({same}, {ms:.1} ms) pool {} B in {} allocations",
                 pre.grid_h, pre.grid_w, merged.len() / tower.dims.out_hidden, pool.0, pool.1);
        if same.starts_with("DIFFERS") { std::process::exit(1); }
    }
    println!("VISION_GPU_HASH_DONE {} images, {} forwards; tower pool at the end {} B in {} allocations; passes on which it grew \
              after an image's first pass: {:?} ({})", pres.len(), order.len(), pool_prev.0, pool_prev.1, grew_late,
             if grew_late.is_empty() { "POOL_FLAT" } else { "POOL_GREW" });
}
