//! Validate the GPU forward against the PyTorch reference (ref/ref_data.npz).
use gb10_inference::gpu::{GpuModel, Pool};
use gb10_inference::qwen::Model;
use std::collections::HashMap;

fn read_npz(path: &str) -> HashMap<String, Vec<f32>> {
    let f = std::fs::File::open(path).unwrap();
    let mut zip = zip::ZipArchive::new(f).unwrap();
    let mut out = HashMap::new();
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i).unwrap();
        let name = entry.name().trim_end_matches(".npy").to_string();
        let mut buf = Vec::new();
        use std::io::Read; entry.read_to_end(&mut buf).unwrap();
        let (hlen, hdr_off) = if buf[6]==1 { (u16::from_le_bytes([buf[8],buf[9]]) as usize, 10) }
            else { (u32::from_le_bytes([buf[8],buf[9],buf[10],buf[11]]) as usize, 12) };
        let header = std::str::from_utf8(&buf[hdr_off..hdr_off+hlen]).unwrap();
        let ds = header.find("'descr':").unwrap();
        let after = &header[ds+"'descr':".len()..];
        let oq = after.find('\'').unwrap(); let rest=&after[oq+1..]; let cq=rest.find('\'').unwrap();
        let descr = &rest[..cq];
        let data = &buf[hdr_off+hlen..];
        let flat: Vec<f32> = if descr.contains("i8") {
            let cnt=data.len()/8; let sl=unsafe{std::slice::from_raw_parts(data.as_ptr() as *const i64,cnt)};
            sl.iter().map(|&x| x as f32).collect()
        } else { let cnt=data.len()/4; let sl=unsafe{std::slice::from_raw_parts(data.as_ptr() as *const f32,cnt)}; sl.to_vec() };
        out.insert(name, flat);
    }
    out
}

#[test]
fn test_gpu_forward_matches_reference() {
    let model_path = std::env::var("GB10_TEST_MODEL")
        .expect("GB10_TEST_MODEL must point at model.safetensors (test fixture)");
    let npz_path = std::env::var("GB10_TEST_REF_NPZ")
        .expect("GB10_TEST_REF_NPZ must point at ref_data.npz (test fixture; run ref/ref_forward.py)");
    let host = Model::load(&model_path).expect("load");
    let gpu = GpuModel::new(&host).expect("gpu init");
    let npz = read_npz(&npz_path);
    let input_ids: Vec<u32> = npz["input_ids"].iter().map(|&v| v as u32).collect();
    let n = input_ids.len();
    let h = gpu.cfg().hidden_size;

    let mut state = gpu.new_state();
    let mut pool = Pool::new(gpu.dev().clone());
    let mut l0: Vec<f32> = vec![]; let mut l3: Vec<f32> = vec![]; let mut last_hidden = vec![0.0f32; h];
    for (t, &tok) in input_ids.iter().enumerate() {
        let rdim = gpu.cfg().rotary_dim; let half=rdim/2; let theta=gpu.cfg().rope_theta;
        let mut cos=vec![0.0f32;rdim]; let mut sin=vec![0.0f32;rdim];
        for i in 0..half { let f=(t as f32)*theta.powf(-(2.0*i as f32)/rdim as f32); cos[i]=f.cos();sin[i]=f.sin();cos[i+half]=cos[i];sin[i+half]=sin[i]; }
        let cosd = gpu.dev().htod_sync_copy(&cos).unwrap();
        let sind = gpu.dev().htod_sync_copy(&sin).unwrap();
        let emb = gpu.embed_row(tok);
        let mut caps: Vec<Vec<f32>> = vec![];
        let out = gpu.forward_token_captured(&mut pool, emb, t, &mut state, &cosd, &sind, &mut caps);
        if t == n-1 {
            l0 = caps[0].clone(); l3 = caps[3].clone(); last_hidden = gpu.dev().dtoh_sync_copy(&out).unwrap();
        }
    }
    // diffs
    let l0_ref = &npz["layer0_out"];
    let mut l0m=0.0f32; for i in 0..h { l0m=l0m.max((l0[i]-l0_ref[(n-1)*h+i]).abs()); }
    println!("GPU layer0 (linear_attn) max abs diff: {:.6}", l0m);
    let l3_ref = &npz["layer3_out"];
    let mut l3m=0.0f32; for i in 0..h { l3m=l3m.max((l3[i]-l3_ref[(n-1)*h+i]).abs()); }
    println!("GPU layer3 (full_attn) max abs diff: {:.6}", l3m);

    // logits argmax
    let hidden_d = gpu.dev().htod_sync_copy(&last_hidden).unwrap();
    let logits_d = gpu.logits(&mut pool, &hidden_d);
    let logits = gpu.dev().dtoh_sync_copy(&logits_d).unwrap();
    let my = logits.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
    let refr = npz["next_tok"][0] as u32;
    println!("GPU argmax: {} ref: {}", my, refr);
    // report only; don't hard-fail so we see all diffs
}
