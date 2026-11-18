//! V1 golden test: compare the engine's Rust preprocessing against the HF reference oracle.
//!
//! Self-contained bin (no cudarc/lib link; includes the preproc module via #[path]). The HF
//! reference (`tool_probe/w2/ref_vision_oracle.py --mode preproc`) writes
//!   <outdir>/pixel_values.bin   (raw f32, first two i64 = [rows, 1536])
//!   <outdir>/manifest.json      (grid_thw, num_image_tokens, grid_sha per image)
//! This binary decodes the same images, runs `vision_preproc`, concatenates the flatten in image
//! order, and reports max-abs / mean-abs / grid-SHA vs the reference.

#[path = "../vision_preproc.rs"]
mod vision_preproc;

use vision_preproc::{preprocess_image, QWEN27B_PREPROC};
use std::fs;

fn read_f32_raw(path: &str) -> (Vec<f32>, Vec<i64>) {
    let bytes = fs::read(path).expect("read ref");
    let mut shape = Vec::new();
    for off in [0usize, 8] {
        if off + 8 <= bytes.len() {
            shape.push(i64::from_le_bytes(bytes[off..off + 8].try_into().unwrap()));
        }
    }
    let data = &bytes[16..];
    let n = data.len() / 4;
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        v.push(f32::from_le_bytes([
            data[i * 4],
            data[i * 4 + 1],
            data[i * 4 + 2],
            data[i * 4 + 3],
        ]));
    }
    (v, shape)
}

/// A required `--flag <VALUE>` of this dev tool (no built-in lab path: S-B9-REL-SMALLFIXES item 8).
fn required_flag(name: &str) -> String {
    let argv: Vec<String> = std::env::args().collect();
    argv.iter().position(|a| a == name).and_then(|i| argv.get(i + 1).cloned())
        .unwrap_or_else(|| panic!("usage: pass {name} <DIR>"))
}

fn main() {
    // --w2-dir <DIR>: the W2 fixture directory (images/ and refout/; tool_probe/w2 of the dev tree)
    let w2 = required_flag("--w2-dir");
    let imgs: Vec<(&str, String)> = ["landscape.png", "portrait.png", "small.png"].iter()
        .map(|n| (*n, format!("{w2}/images/{n}"))).collect();
    let outdir = format!("{w2}/refout");
    let manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(format!("{outdir}/manifest.json")).unwrap()).unwrap();
    let (ref_rows, ref_shape) = read_f32_raw(&format!("{outdir}/pixel_values.bin"));
    let ref_rd = ref_shape[1] as usize;
    let imgs_arr = manifest["images"].as_array().unwrap();
    let mut per_img: Vec<(usize, usize)> = Vec::new(); // (gh, gw)
    for e in imgs_arr {
        let g = e["grid_thw"].as_array().unwrap();
        per_img.push((g[1].as_u64().unwrap() as usize, g[2].as_u64().unwrap() as usize));
    }

    let mut got: Vec<f32> = Vec::new();
    let mut gh = Vec::new();
    let mut gw = Vec::new();
    let mut shape_ok = true;
    for (i, (name, path)) in imgs.iter().enumerate() {
        let dimg = image::open(path).unwrap().to_rgb8();
        let (w, h) = dimg.dimensions();
        let pre = preprocess_image(h as usize, w as usize, dimg.as_raw(), &QWEN27B_PREPROC);
        gh.push(pre.grid_h);
        gw.push(pre.grid_w);
        let want = per_img[i];
        if pre.grid_h != want.0 || pre.grid_w != want.1 {
            eprintln!("PREPROC_GRID_MISMATCH {name}: rust=({},{}) manifest=({},{})",
                pre.grid_h, pre.grid_w, want.0, want.1);
            shape_ok = false;
        }
        got.extend_from_slice(&pre.pixel_values);
    }
    eprintln!("[rust] grids {:?} x {:?}", gh, gw);

    if got.len() != ref_rows.len() {
        eprintln!("PREPROC_ROW_MISMATCH rust={} ref={}", got.len(), ref_rows.len());
        println!("PREPROC_MISMATCH (row count)");
        return;
    }

    let mut maxabs = 0.0f64;
    let mut mean = 0.0f64;
    let mut maxidx = 0usize;
    for k in 0..got.len() {
        let d = (got[k] as f64 - ref_rows[k] as f64).abs();
        if d > maxabs {
            maxabs = d;
            maxidx = k;
        }
        mean += d;
    }
    mean /= got.len() as f64;
    let tol = 1e-3;
    println!("PREPROC_COMPARE maxabs={maxabs:.6} meanabs={mean:.6} rows={} max_idx={maxidx} (row={},elem={})",
        got.len(), maxidx / ref_rd, maxidx % ref_rd);
    println!("PREPROC_COMPARE grid_sha shapes={shape_ok} rust=({gh:?}x{gw:?}) manifest={per_img:?}");
    if shape_ok && maxabs <= tol {
        println!("PREPROC_MATCH");
    } else {
        println!("PREPROC_MISMATCH shape_ok={shape_ok} maxabs={maxabs:.6} tol={tol}");
    }
}
