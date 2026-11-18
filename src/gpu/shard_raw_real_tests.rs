//! Unit tests for the host shard helpers on real packs (former inline test module `shard_raw_real_tests` of `gpu.rs`; file map in `mod.rs`).

    use super::*;

    /// The real-model check: fuse the 27B's GDN in_proj parts from the actual artifact, then
    /// slice-then-repack vs repack-then-host-shard must be byte-identical (both ranks).
    #[test]
    fn real_gdn_inproj_slice_roundtrip() {
        use safetensors::{SafeTensors, Dtype};
        let dir = std::env::var("GB10_TEST_MODEL_DIR").unwrap_or_default();
        if dir.is_empty() { eprintln!("skip: GB10_TEST_MODEL_DIR not set"); return; }
        let files: Vec<_> = std::fs::read_dir(dir).unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |x| x == "safetensors"))
            .collect();
        if files.is_empty() { eprintln!("skip: no model"); return; }
        // find the tensors of layer 0's linear_attn in_proj
        let mut parts: Vec<(Vec<u8>, Vec<u8>, f32, usize)> = Vec::new();
        let mut k = 0usize;
        for f in &files {
            let raw = std::fs::read(f).unwrap();
            let st = SafeTensors::deserialize(&raw).unwrap();
            for name in st.names() {
                let stem = name.strip_suffix(".weight_packed");
                if let Some(stem) = stem {
                    if stem.ends_with("layers.0.linear_attn.in_proj_qkv")
                        || stem.ends_with("layers.0.linear_attn.in_proj_z")
                        || stem.ends_with("layers.0.linear_attn.in_proj_b")
                        || stem.ends_with("layers.0.linear_attn.in_proj_a") {
                        let pv = st.tensor(name).unwrap();
                        let sv = st.tensor(&format!("{stem}.weight_scale")).unwrap();
                        let gv = st.tensor(&format!("{stem}.weight_global_scale")).unwrap();
                        let (m, k2) = (pv.shape()[0], pv.shape()[1]);
                        let gs = f32::from_le_bytes(gv.data()[..4].try_into().unwrap());
                        k = k2 * 2;
                        parts.push((pv.data().to_vec(), sv.data().to_vec(), 1.0 / gs, m));
                    }
                }
            }
        }
        assert_eq!(parts.len(), 4, "found {} in_proj parts", parts.len());
        let refs: Vec<(&[u8], &[u8], f32, usize)> = parts.iter()
            .map(|p| (&p.0[..], &p.1[..], p.2, p.3)).collect();
        let (qw, sc, gsv) = crate::quant::fuse_nvfp4(&refs, k);
        let m = parts.iter().map(|p| p.3).sum();
        eprintln!("fused in_proj {m}x{k}");
        // The real 27B segment widths are 512 (=2^9, divides 16·world up to 32) with 48-row
        // split=false b/a segments (16-aligned); validate the whole power-of-two ladder.
        for world in [1usize, 2, 4, 8, 16] {
            let segs = vec![
                (0, 512, true), (512, 512, true), (1024, 512, true),
                (1536, 512, true), (2048, 48, false), (2096, 48, false)];
            let op = LoadShardOp::ColSegs(segs.clone());
            for rank in 0..world {
                let (sq, ss, sg, sm, sk) = shard_raw_nvfp4(&qw, &sc, &gsv, m, k, &op, rank, world);
                let (wt2, st2) = crate::quant::repack_nvfp4_mma(&sq, &ss, sm, sk);
                let (wt, st) = crate::quant::repack_nvfp4_mma(&qw, &sc, m, k);
                let (rw, rs, rg, rm) = host_shard_nvfp4_col_segs(&wt, &st, &gsv, k, &segs, rank, world);
                assert_eq!(wt2, rw, "real in_proj world {world} rank {rank}: qweight mismatch");
                assert_eq!(st2, rs, "real in_proj world {world} rank {rank}: scales mismatch");
                assert_eq!(sg, rg, "real in_proj world {world} rank {rank}: gs mismatch");
                assert_eq!(sm, rm, "real in_proj world {world} rank {rank}: m mismatch");
            }
        }
    }
