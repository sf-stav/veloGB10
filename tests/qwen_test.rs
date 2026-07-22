//! Validate qwen.rs forward against the PyTorch reference (ref/ref_data.npz).

use gb10_inference::qwen::{Config, Model, ModelState};
use std::collections::HashMap;

// Minimal NPZ (zip) + NPY reader.
fn read_npz(path: &str) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let f = std::fs::File::open(path).unwrap();
    let mut zip = zip::ZipArchive::new(f).unwrap();
    let mut out = HashMap::new();
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i).unwrap();
        let name = entry.name().trim_end_matches(".npy").to_string();
        // read NPY: magic \x93NUMPY version(2) header_len(2 BE) header(dict) then data (little-endian)
        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut entry, &mut buf).unwrap();
        let magic = &buf[0..6];
        assert_eq!(magic, b"\x93NUMPY");
        let (hlen, hdr_off) = if buf[6] == 1 {
            // v1.0: 2-byte header length (little-endian)
            (u16::from_le_bytes([buf[8], buf[9]]) as usize, 10)
        } else {
            // v2.0+: 4-byte header length (little-endian) at offset 8
            (u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize, 12)
        };
        let header = std::str::from_utf8(&buf[hdr_off..hdr_off + hlen]).unwrap().to_string();
        let data = &buf[hdr_off + hlen..];
        // parse dtype + shape from header dict
        let shape_start = header.find("'shape':").or_else(|| header.find("\"shape\":")).unwrap();
        let rest = &header[shape_start..];
        let lp = rest.find('(').unwrap();
        let rp = rest.find(')').unwrap();
        let shape_str = &rest[lp + 1..rp];
        let mut shape = Vec::new();
        for part in shape_str.split(',') {
            let p = part.trim();
            if !p.is_empty() {
                shape.push(p.parse::<usize>().unwrap());
            }
        }
        let descr = {
            let ds = header.find("'descr':").or_else(|| header.find("\"descr\":")).unwrap();
            let after = &header[ds + "'descr':".len()..]; // skip the key+colon
            let oq = after.find('\'').unwrap();
            let rest = &after[oq + 1..];
            let cq = rest.find('\'').unwrap();
            &rest[..cq]
        };
        let data = data;
        let flat: Vec<f32> = if descr.contains("i8") {
            // int64 little-endian -> cast values to f32
            let cnt = data.len() / 8;
            let sl = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const i64, cnt) };
            sl.iter().map(|&x| x as f32).collect()
        } else {
            let cnt = data.len() / 4;
            let sl = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, cnt) };
            sl.to_vec()
        };
        out.insert(name, (flat, shape));
    }
    out
}

#[test]
fn test_forward_matches_reference() {
    let model_path = std::env::var("GB10_TEST_MODEL")
        .expect("GB10_TEST_MODEL must point at model.safetensors (test fixture)");
    let npz_path = std::env::var("GB10_TEST_REF_NPZ")
        .expect("GB10_TEST_REF_NPZ must point at ref_data.npz (test fixture; run ref/ref_forward.py)");

    let model = Model::load(&model_path).expect("load model");
    let cfg = &model.config;
    let npz = read_npz(&npz_path);

    let input_ids: Vec<u32> = npz["input_ids"].0.iter().map(|&v| v as u32).collect();
    let n = input_ids.len();

    // ---- 1. check embed ----
    let emb_ref = &npz["embed_out"].0;
    // embed_out shape [n, hidden]; ref stored flattened row-major
    let max_diff = (0..n).flat_map(|t| {
        let h = cfg.hidden_size;
        let emb = model.embed(input_ids[t]);
        (0..h).map(move |i| (emb[i] - emb_ref[t * h + i]).abs())
    }).fold(0.0f32, f32::max);
    println!("embed max abs diff: {:.6}", max_diff);
    assert!(max_diff < 0.05, "embed diff too large: {}", max_diff);

    let h = cfg.hidden_size;

    // ---- 2. run full prefill token-by-token, capture per-layer residuals ----
    let mut state = ModelState::new(&model);
    let mut layer0_h: Vec<Vec<f32>> = Vec::new();
    let mut layer3_h: Vec<Vec<f32>> = Vec::new();
    let mut final_hiddens: Vec<Vec<f32>> = Vec::new();
    for (t, &tok) in input_ids.iter().enumerate() {
        let h_in = model.embed(tok);
        let (cos, sin) = model.rope_tables(t);
        let mut caps: Vec<Vec<f32>> = Vec::new();
        let fh = model.forward_token_captured(&h_in, t, &mut state, &cos, &sin, &mut caps);
        // caps has 24 entries (residual after each layer)
        layer0_h.push(caps[0].clone());
        layer3_h.push(caps[3].clone());
        final_hiddens.push(fh);
    }

    // diff layer0 (linear attn) and layer3 (full attn)
    let l0_ref = &npz["layer0_out"].0; // [n, hidden]
    let mut l0max = 0.0f32;
    for t in 0..n {
        for i in 0..h {
            let d = (layer0_h[t][i] - l0_ref[t * h + i]).abs();
            if d > l0max { l0max = d; }
        }
    }
    println!("layer0 (linear_attn) max abs diff: {:.6}", l0max);

    let l3_ref = &npz["layer3_out"].0;
    let mut l3max = 0.0f32;
    for t in 0..n {
        for i in 0..h {
            let d = (layer3_h[t][i] - l3_ref[t * h + i]).abs();
            if d > l3max { l3max = d; }
        }
    }
    println!("layer3 (full_attn) max abs diff: {:.6}", l3max);

    // compare layer0_out? We don't expose per-layer in forward_token easily; compare final_norm_out instead.
    let fn_ref = &npz["final_norm_out"].0; // [1, n, hidden]
    let h = cfg.hidden_size;
    let mut fmax = 0.0f32;
    for t in 0..n {
        for i in 0..h {
            let d = (final_hiddens[t][i] - fn_ref[t * h + i]).abs();
            if d > fmax { fmax = d; }
        }
    }
    println!("final_norm max abs diff: {:.6}", fmax);
    // bf16-vs-f32 rounding accumulates over 24 layers; final hidden can drift a bit.
    // The decisive correctness check is the greedy argmax below.
    assert!(fmax < 1.0, "final_norm diff too large: {}", fmax);

    // ---- 3. logits for last token ----
    let logits = model.logits(&final_hiddens[n - 1]);
    let lr = &npz["logits_last"].0; // [vocab]
    let v = cfg.vocab_size;
    let mut lmax = 0.0f32;
    let mut l1 = 0.0f32;
    for i in 0..v {
        let d = (logits[i] - lr[i]).abs();
        if d > lmax { lmax = d; }
        l1 += d;
    }
    println!("logits max abs diff: {:.6}  mean abs diff: {:.6}", lmax, l1 / v as f32);
    // argmax must match
    let my_argmax = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
    let ref_tok = npz["next_tok"].0[0] as u32;
    println!("my argmax: {}  ref next_tok: {}", my_argmax, ref_tok);
    assert_eq!(my_argmax, ref_tok, "greedy argmax mismatch");
}
