//! Unit tests (module `ple_ngram_tests`) moved out of the old inline block in `exl3_forward.rs`; text unchanged.
//! Declared from `mod.rs`; the body keeps its original indentation and `use super::*` imports.

    use super::*;
    use serde_json::json;

    const KEY: &str = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding";
    const ROWS: i64 = 4096; // synthetic small row count; the planner is size-agnostic

    fn shard_name(i: usize) -> String { format!("{KEY}.shard_{i}.trellis") }
    fn names() -> Vec<String> { (0..128).map(shard_name).collect() }

    /// Synthetic sidecar header: 128 contiguous I16 shards [ROWS, 10*k+1], optionally with a
    /// mutation applied to one shard entry (hooks for the failure cases). Returns a made-up
    /// header length; the planner only uses it to make offsets absolute.
    fn synth_sidecar(k: usize, mutate: Option<(usize, fn(serde_json::Value) -> serde_json::Value)>) -> (u64, serde_json::Value) {
        let width = 10 * k + 1;
        let mut hdr = serde_json::Map::new();
        hdr.insert("__metadata__".to_string(), json!({"format": "exl3_ngram_trellis", "K": k.to_string(), "shard_rows": ROWS}));
        let per = ROWS as u64 * width as u64 * 2;
        let mut off = 0u64;
        for i in 0..128usize {
            let mut ent = json!({"dtype": "I16", "shape": [ROWS, width], "data_offsets": [off, off + per]});
            if let Some((idx, f)) = &mutate { if *idx == i { ent = f(ent); } }
            hdr.insert(shard_name(i), ent);
            off += per;
        }
        (500u64, serde_json::Value::Object(hdr))
    }

    fn wmap_with_shards(file: &str) -> serde_json::Map<String, serde_json::Value> {
        let mut m = serde_json::Map::new();
        for n in names() { m.insert(n, json!(file)); }
        m
    }

    #[test]
    fn sidecar_k5_plan_ok() {
        let (hlen, hdr) = synth_sidecar(5, None);
        let (rows, words, offs) = ple_sidecar_shard_plan(&hdr, hlen, &names()).unwrap();
        assert_eq!((rows, words), (ROWS, 51));
        assert_eq!(offs.len(), 128);
        let per = ROWS as u64 * 51 * 2;
        assert_eq!(offs[0], 8 + hlen, "shard 0 starts right after the header");
        for i in 1..128 { assert_eq!(offs[i], offs[0] + per * i as u64, "contiguous in order, shard {i}"); }
    }

    #[test]
    fn sidecar_k6_plan_ok() {
        let (hlen, hdr) = synth_sidecar(6, None);
        let (rows, words, offs) = ple_sidecar_shard_plan(&hdr, hlen, &names()).unwrap();
        assert_eq!((rows, words), (ROWS, 61));
        assert_eq!(offs.len(), 128);
        assert_eq!(offs[127], offs[0] + ROWS as u64 * 61 * 2 * 127);
    }

    #[test]
    fn sidecar_missing_shard_named() {
        let (hlen, mut hdr) = synth_sidecar(6, None);
        hdr.as_object_mut().unwrap().remove(shard_name(42).as_str());
        let err = ple_sidecar_shard_plan(&hdr, hlen, &names()).unwrap_err().to_string();
        assert!(err.contains("shard 42"), "error names the shard: {err}");
    }

    #[test]
    fn sidecar_non_contiguous_named() {
        let gap = |e: serde_json::Value| {
            let mut o = e;
            let offs = o["data_offsets"].as_array().unwrap().clone();
            o["data_offsets"] = json!([offs[0].as_u64().unwrap() + 16, offs[1].as_u64().unwrap() + 16]);
            o
        };
        let (hlen, hdr) = synth_sidecar(5, Some((7, gap)));
        let err = ple_sidecar_shard_plan(&hdr, hlen, &names()).unwrap_err().to_string();
        assert!(err.contains("shard 7") && err.contains("contiguous"), "error names the shard and the rule: {err}");
    }

    #[test]
    fn sidecar_rejected_widths_named() {
        // #26: the loader accepts any 10*K+1 row with K = 1..=8, so K=4 (41) and
        // K=7 (71) now plan. Only a non-10K+1 width (52) and a mixed-width set fail.
        // non-10K+1 width (52): refused, shard named. (fn pointer = non-capturing closures only)
        {
            let (hlen, hdr) = synth_sidecar(5, Some((9, |mut e| { e["shape"] = json!([ROWS, 52]); e })));
            let err = ple_sidecar_shard_plan(&hdr, hlen, &names()).unwrap_err().to_string();
            assert!(err.contains("shard 9"), "error names the shard (w=52): {err}");
        }
        // K=4 (41) and K=7 (71) rows are valid now: plan, do not refuse.
        {
            let (hlen, hdr) = synth_sidecar(4, Some((9, |mut e| { e["shape"] = json!([ROWS, 41]); e })));
            assert_eq!(ple_sidecar_shard_plan(&hdr, hlen, &names()).expect("K=4 sidecar must plan").1, 41);
        }
        {
            let (hlen, hdr) = synth_sidecar(7, Some((9, |mut e| { e["shape"] = json!([ROWS, 71]); e })));
            assert_eq!(ple_sidecar_shard_plan(&hdr, hlen, &names()).expect("K=7 sidecar must plan").1, 71);
        }
        // ISSUE #38 (2026-11-13): an ng8 pack (8 experts) was REFUSED by the public v0.7.8 cut
        // because that cut's PLE admission test was `matches!((swidth - 1) / 10, 5 | 6)`. The
        // loader accepts every 10*K+1 row for K = 1..=8, so K=8 (81 words) MUST plan. This test
        // pins the admission set so a narrowed check cannot come back unnoticed: the failure
        // mode is a loud refusal of a pack the engine can serve, and it only shows up on a
        // user's model, never in a unit test that exercised only K = 5 / 6.
        {
            let (hlen, hdr) = synth_sidecar(8, None);
            let (rows, words, offs) = ple_sidecar_shard_plan(&hdr, hlen, &names())
                .expect("K=8 (ng8) sidecar must plan — issue #38");
            assert_eq!((rows, words), (ROWS, 81), "K=8 rows are 10*8+1 = 81 words");
            assert_eq!(offs.len(), 128);
            assert_eq!(offs[127], offs[0] + ROWS as u64 * 81 * 2 * 127);
        }
        {
            let (hlen, hdr) = synth_sidecar(8, Some((9, |mut e| { e["shape"] = json!([ROWS, 81]); e })));
            assert_eq!(ple_sidecar_shard_plan(&hdr, hlen, &names())
                .expect("K=8 sidecar must plan when the width is uniform").1, 81);
        }
        // The whole admitted set, in one place: K = 1..=8 all plan.
        for k in 1usize..=8 {
            let (hlen, hdr) = synth_sidecar(k, None);
            assert_eq!(ple_sidecar_shard_plan(&hdr, hlen, &names())
                .expect("K must plan").1, 10 * k + 1, "K={k} (ng{k}) sidecar must plan");
        }
        // Mixed widths across shards (51 vs 61) also fails, naming the shard.
        let (hlen, hdr) = synth_sidecar(5, Some((9, |mut e| { e["shape"] = json!([ROWS, 61]); e })));
        let err = ple_sidecar_shard_plan(&hdr, hlen, &names()).unwrap_err().to_string();
        assert!(err.contains("shard 9") && err.contains("uniform"), "{err}");
    }

    #[test]
    fn sidecar_bad_dtype_and_rowcount_named() {
        let (hlen, hdr) = synth_sidecar(6, Some((3, |mut e| { e["dtype"] = json!("F16"); e })));
        let err = ple_sidecar_shard_plan(&hdr, hlen, &names()).unwrap_err().to_string();
        assert!(err.contains("shard 3") && err.contains("I16"), "{err}");
        let (hlen, hdr) = synth_sidecar(6, Some((4, |mut e| { e["shape"] = json!([ROWS + 1, 61]); e })));
        let err = ple_sidecar_shard_plan(&hdr, hlen, &names()).unwrap_err().to_string();
        assert!(err.contains("shard 4") && err.contains("rows"), "{err}");
    }

    #[test]
    fn sidecar_shard_rows_metadata_mismatch_is_a_note_not_an_error() {
        let (hlen, mut hdr) = synth_sidecar(5, None);
        hdr["__metadata__"]["shard_rows"] = json!(ROWS + 12345);
        assert!(ple_sidecar_shard_plan(&hdr, hlen, &names()).is_ok(),
                "shapes win over metadata (single-tensor-path semantic); mismatch only logs");
    }

    #[test]
    fn index_listed_shards_untouched_no_sidecar_consulted() {
        // Today's behaviour: the index lists every shard -> all Index sources, the sidecar
        // planner is never invoked (a garbage sidecar header must not matter).
        let wmap = wmap_with_shards("ngram_embedding.safetensors");
        let (srcs, layout) = ple_shard_sources(Some(&wmap), None, &names()).unwrap();
        assert!(layout.is_none());
        assert!(srcs.iter().all(|s| matches!(s, PleShardSrc::Index(f) if f == "ngram_embedding.safetensors")));
        // Even a present-but-broken sidecar is not consulted when the index lists the shards.
        let (hlen, hdr) = synth_sidecar(5, Some((0, |mut e| { e["dtype"] = json!("F32"); e })));
        let (srcs2, layout2) = ple_shard_sources(Some(&wmap), Some(("ngram_embedding.safetensors", hlen, &hdr)), &names()).unwrap();
        assert!(layout2.is_none() && srcs2 == srcs);
    }

    #[test]
    fn mixed_index_and_sidecar_subset_loads() {
        // K8 (REL v0.7.3 review): the index lists shards 0..63; the sidecar holds ONLY
        // 64..127. The old code planned ALL 128 names against the sidecar header and
        // failed "shard 0 missing"; the subset planner must load the mixed layout with
        // correct absolute offsets.
        let (hlen, hdr) = synth_sidecar(6, None);
        let mut wmap = serde_json::Map::new();
        for n in names().into_iter().take(64) { wmap.insert(n, json!("model-00001.safetensors")); }
        let (srcs, layout) = ple_shard_sources(Some(&wmap), Some(("ngram_embedding.safetensors", hlen, &hdr)), &names()).unwrap();
        assert_eq!(layout, Some((ROWS, 61)));
        let per = ROWS as u64 * 61 * 2;
        for (i, s) in srcs.iter().enumerate() {
            match s {
                PleShardSrc::Index(f) => assert!(i < 64 && f == "model-00001.safetensors", "src {i}: {f:?}"),
                PleShardSrc::Sidecar { fname, abs_off } => {
                    assert!(i >= 64 && fname == "ngram_embedding.safetensors", "src {i}");
                    assert_eq!(*abs_off, 8 + hlen + per * i as u64, "shard {i} absolute offset");
                }
            }
        }
        // A sidecar shard of the subset missing from the sidecar header is named.
        let (hlen2, mut hdr2) = synth_sidecar(6, None);
        hdr2.as_object_mut().unwrap().remove(shard_name(100).as_str());
        let err = ple_shard_sources(Some(&wmap), Some(("ngram_embedding.safetensors", hlen2, &hdr2)), &names())
            .unwrap_err().to_string();
        assert!(err.contains("shard 100") && err.contains("missing"), "{err}");
        // Two overlapping subset spans are rejected (order-independent).
        let (hlen3, hdr3) = synth_sidecar(6, Some((70, |mut e| {
            let offs = e["data_offsets"].as_array().unwrap().clone();
            let d0 = offs[0].as_u64().unwrap(); let d1 = offs[1].as_u64().unwrap();
            e["data_offsets"] = json!([d0 - 8, d1 - 8]);
            e
        })));
        let err = ple_shard_sources(Some(&wmap), Some(("ngram_embedding.safetensors", hlen3, &hdr3)), &names())
            .unwrap_err().to_string();
        assert!(err.contains("overlap"), "{err}");
        // Out-of-order sidecar entries (explicit absolute offsets) are fine for a subset.
        let (hlen4, mut hdr4) = synth_sidecar(6, None);
        let h = hdr4.as_object_mut().unwrap();
        for i in 64..128usize {
            let e = h.get(&shard_name(i)).cloned().unwrap();
            h.insert(format!("zz_{i}"), e.clone());
            h.insert(shard_name(i), json!({}));
            h.insert(shard_name(i), e);
        }
        // (same entries, just re-inserted — real out-of-order below)
        let (srcs4, _) = ple_shard_sources(Some(&wmap), Some(("ngram_embedding.safetensors", hlen4, &hdr4)), &names()).unwrap();
        assert!(matches!(srcs4[127], PleShardSrc::Sidecar { .. }));
    }

    #[test]
    fn sidecar_fallback_when_index_lacks_shards() {
        // Issue #9 layout: the index lists nothing n-gram; the sidecar holds the 128 shards.
        let (hlen, hdr) = synth_sidecar(6, None);
        let (srcs, layout) = ple_shard_sources(None, Some(("ngram_embedding.safetensors", hlen, &hdr)), &names()).unwrap();
        assert_eq!(layout, Some((ROWS, 61)));
        assert!(srcs.iter().all(|s| matches!(s, PleShardSrc::Sidecar { fname, .. } if fname == "ngram_embedding.safetensors")));
        let offs: Vec<u64> = srcs.iter().map(|s| if let PleShardSrc::Sidecar { abs_off, .. } = s { *abs_off } else { unreachable!() }).collect();
        assert_eq!(offs[0], 8 + hlen);
        // K=5 too...
        let (hlen5, hdr5) = synth_sidecar(5, None);
        let (_, l5) = ple_shard_sources(None, Some(("ngram_embedding.safetensors", hlen5, &hdr5)), &names()).unwrap();
        assert_eq!(l5, Some((ROWS, 51)));
        // ...and the failure text when there is no sidecar at all keeps the original words.
        let err = ple_shard_sources(None, None, &names()).unwrap_err().to_string();
        assert!(err.contains("ple shard 0 not in index"), "original v0.7.2 message preserved: {err}");
        // A sidecar that exists but lacks the shard names it clearly.
        let (hlenb, hdrb) = synth_sidecar(6, None);
        let mut small = json!({"__metadata__": {}});
        small[shard_name(0).as_str()] = hdrb[shard_name(0).as_str()].clone();
        let err = ple_shard_sources(None, Some(("ngram_embedding.safetensors", hlenb, &small)), &names()).unwrap_err().to_string();
        assert!(err.contains("shard 1") && err.contains("missing"), "{err}");
    }

    #[test]
    fn mixed_index_and_sidecar_sources_validated() {
        // Half index, half sidecar: the sidecar subset is still validated in full and the
        // per-shard offsets stay right for both sources.
        let (hlen, hdr) = synth_sidecar(6, None);
        let mut wmap = serde_json::Map::new();
        for n in names().iter().take(64) { wmap.insert(n.clone(), json!("model-00001-of-00007.safetensors")); }
        let (srcs, layout) = ple_shard_sources(Some(&wmap), Some(("ngram_embedding.safetensors", hlen, &hdr)), &names()).unwrap();
        assert_eq!(layout, Some((ROWS, 61)));
        assert!(matches!(srcs[0], PleShardSrc::Index(_)));
        assert!(matches!(srcs[63], PleShardSrc::Index(_)));
        assert!(matches!(srcs[64], PleShardSrc::Sidecar { .. }));
        assert!(matches!(srcs[127], PleShardSrc::Sidecar { .. }));
        if let PleShardSrc::Sidecar { abs_off, .. } = &srcs[127] {
            assert_eq!(*abs_off, 8 + hlen + ROWS as u64 * 61 * 2 * 127);
        } else { unreachable!() }
    }
