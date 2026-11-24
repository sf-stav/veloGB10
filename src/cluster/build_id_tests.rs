    use super::*;
    use std::collections::BTreeSet;

    /// `(including file, include target)` pairs that are test-only: the include sits inside `#[cfg(test)]`,
    /// the target is a lab fixture under `tests/` (absent from a source tree shipped without `tests/`),
    /// and it is deliberately in NO stamp (see build.rs). Adding a pair here is a review decision.
    const CFG_TEST_ONLY_INCLUDES: &[(&str, &str)] =
        &[("src/loop_detect.rs", "tests/fixtures/loop_detect_vectors.txt")];

    fn root() -> PathBuf { PathBuf::from(env!("CARGO_MANIFEST_DIR")) }
    fn hashed() -> BTreeSet<&'static str> { env!("SOURCE_BUILD_FILES").split(';').collect() }
    fn kernel() -> BTreeSet<&'static str> { env!("KERNEL_BUILD_FILES").split(';').collect() }

    /// Every `.rs` under `src/`, at any depth, as a path relative to the crate root. The walk is this
    /// test's own (build.rs has another): `src/bin` (separate executables) and `src/ptx` (build output)
    /// are the only directories left out, and they are named here, so a new one is walked.
    fn engine_sources() -> Vec<String> {
        let (root, mut out, mut stack) = (root(), Vec::new(), vec![root().join("src")]);
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).unwrap().flatten() {
                let p = e.path();
                let rel = p.strip_prefix(&root).unwrap().to_string_lossy().to_string();
                if p.is_dir() {
                    if rel != "src/bin" && rel != "src/ptx" { stack.push(p); }
                } else if p.is_file() && p.extension().map_or(false, |x| x == "rs") {
                    out.push(rel);
                }
            }
        }
        out.sort();
        out
    }

    /// The W1 hole: `SOURCE_BUILD_ID` read the `.rs` files of `src/` only, so a head and a node that
    /// differed only in `src/exl3_forward/xtp.rs` (or any of 19 more files) passed the handshake.
    #[test]
    fn the_source_stamp_covers_every_engine_source_file_at_any_depth() {
        let (hashed, on_disk) = (hashed(), engine_sources());
        let missing: Vec<&String> = on_disk.iter().filter(|f| !hashed.contains(f.as_str())).collect();
        assert!(missing.is_empty(), "engine sources OUTSIDE the head/node build stamp (a pair that differs only there would \
                 not be refused). If the file is new and build.rs has not rerun since, touch build.rs; otherwise fix the walk \
                 in build.rs:\n{missing:#?}");
        // the files the old stamp missed, by name, and the count: 20 in subdirectories at the time of the fix
        for f in ["src/exl3_forward/xtp/mod.rs", "src/exl3_forward/xtp/ep_deal4.rs", "src/exl3_forward/dense.rs",
                  "src/exl3_forward/dhead.rs", "src/exl3_forward/w4s.rs", "src/exl3_forward/wp16.rs",
                  "src/dflash2/round/mod.rs", "src/dflash2/mod.rs", "src/dspark/round.rs", "src/dspark/mod.rs",
                  "src/cluster/mod.rs", "src/net/mod.rs", "src/main.rs", "src/opts_table.rs"] {
            assert!(hashed.contains(f), "{f} is not in the build stamp");
        }
        let nested = on_disk.iter().filter(|f| f.matches('/').count() >= 2).count();
        assert!(nested >= 20, "the walk found only {nested} engine sources below src/<dir>/ (20 when this test was written): \
                 is the test still walking subdirectories?");
        // not the engine: separate executables and the build's own output
        assert!(!hashed.iter().any(|f| f.starts_with("src/bin/") || f.starts_with("src/ptx/")), "{hashed:?}");
        // every listed file exists (a stale list would mean the stamp was computed from another tree)
        for f in &hashed { assert!(root().join(f).is_file(), "{f} is in the build stamp but not on disk"); }
    }

    /// The other inputs of the binary: the manifests, the build script, the C shim, and every file an
    /// engine source pulls in with `include_str!` / `include_bytes!` / `include!`. Each include target
    /// must be in the source stamp or the kernel stamp (the two halves of `binary_version`).
    #[test]
    fn the_stamps_cover_the_manifests_and_every_included_file() {
        let (hashed, kernel, root) = (hashed(), kernel(), root());
        for f in ["Cargo.lock", "Cargo.toml", "build.rs", "native/net_shim.c", "data/ep_deal/qwen38_flash_next_tp2.json",
                  "data/ep_deal/qwen38_flash_next_tp4.json", "src/tq_consts/Pi.bin", "src/tq_consts/Pit.bin",
                  "src/tq_consts/S.bin", "src/tq_consts/codebooks.bin", "src/tq_consts/scale.bin"] {
            assert!(hashed.contains(f), "{f} is not in the source stamp");
        }
        assert!(kernel.contains("native/tp_doorbell.h") && kernel.contains("kernels/gpu_batch.cu")
                && kernel.contains("kernels/exl3_bench.cu"), "{kernel:?}");
        let (mut seen, mut bad) = (0usize, Vec::new());
        for src in engine_sources() {
            // SPLIT1: the test-only source-text helper (`src_text!`) includes src/*.rs files by computed
            // (CARGO_MANIFEST_DIR) paths; all of them are inside the source stamp (build.rs walks src/).
            if src == "src/src_text.rs" { continue; }
            if src == "src/cluster/build_id_tests.rs" { continue; }   // SPLIT1: this module's own file spells the macro names as needles
            let text = std::fs::read_to_string(root.join(&src)).unwrap();
            // (this module spells the three macro names as needles: scan this file up to it)
            let text = if src == "src/cluster/mod.rs" { &text[..text.find("mod build_id_tests").unwrap()] } else { &text[..] };
            let dir = Path::new(&src).parent().unwrap().to_path_buf();
            for mac in ["include_str!(", "include_bytes!(", "include!("] {
                let mut from = 0;
                while let Some(k) = text[from..].find(mac) {
                    let at = from + k + mac.len();
                    from = at;
                    // a literal path is the only spelling in the tree; anything else (concat!, a macro
                    // variable) is reported, because this test could not say what it pulls in
                    let rest = text[at..].trim_start();
                    let Some(lit) = rest.strip_prefix('"').and_then(|r| r.split('"').next()) else {
                        bad.push(format!("{src}: a non-literal {mac}..) the test cannot resolve: {:?}", rest.chars().take(40).collect::<String>()));
                        continue;
                    };
                    // normalize dir/lit (".." pops) into a path relative to the crate root
                    let mut parts: Vec<String> = dir.iter().map(|c| c.to_string_lossy().to_string()).collect();
                    for c in lit.split('/') {
                        match c { ".." => { parts.pop(); } "." | "" => {} c => parts.push(c.to_string()) }
                    }
                    let target = parts.join("/");
                    seen += 1;
                    // The one stated exception: a TEST fixture included from inside `#[cfg(test)]` is not an
                    // input of the engine binary, so it is in no stamp (a stamp that hashed it "when present"
                    // gave one tree two tokens). The exception is narrow: this exact (file, target) pair, and
                    // only an include that sits after a `#[cfg(test)]` line of that file.
                    if CFG_TEST_ONLY_INCLUDES.contains(&(src.as_str(), target.as_str()))
                        && text[..at].contains("#[cfg(test)]") {
                        continue;
                    }
                    // EMB1: build.rs-EMITTED PTX embeds are derived artifacts. Their bytes are a
                    // pure function of inputs that ARE stamped: the paired kernels/*.cu in the
                    // kernel stamp, and build.rs (owner of the nvcc flags and of this manifest)
                    // in the source stamp. cargo dep-info pins the crate rebuild on the emitted
                    // bytes; the boot-time KERNEL_BUILD_ID assert pins module-binary pairing.
                    // The manifest comes from build.rs itself: the exemption cannot grow.
                    if let Some((cu, _)) = env!("EMBEDDED_PTX_MANIFEST")
                        .split(';').filter_map(|p| p.split_once(':'))
                        .find(|&(_, ptx)| format!("src/ptx/{ptx}") == target)
                    {
                        assert!(kernel.contains(format!("kernels/{cu}").as_str()),
                                "{target} is emitted from kernels/{cu}, not in the kernel stamp");
                        continue;
                    }
                    if !hashed.contains(target.as_str()) && !kernel.contains(target.as_str()) {
                        bad.push(format!("{src} includes {target}: in neither the source stamp nor the kernel stamp (add it in build.rs)"));
                    }
                }
            }
        }
        assert!(bad.is_empty(), "{}", bad.join("\n"));
        assert!(seen >= 30, "the scan resolved only {seen} include targets (40 when written): did it stop matching?");
    }

    /// DSV41: `kernels/gpu_dsv41.cu` is include-split into `kernels/gpu_dsv41/*.cuh` bands; every band on disk
    /// (and the head file) must be in the KERNEL stamp, or a head and a node that differ only in a band pair.
    #[test]
    fn the_kernel_stamp_covers_every_dsv41_band_file() {
        let (kernel, root) = (kernel(), root());
        assert!(kernel.contains("kernels/gpu_dsv41.cu"), "{kernel:?}");
        let mut n = 0;
        for e in std::fs::read_dir(root.join("kernels/gpu_dsv41")).unwrap().flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if !name.ends_with(".cuh") { continue; }
            let rel = format!("kernels/gpu_dsv41/{name}");
            assert!(kernel.contains(rel.as_str()), "{rel} is on disk but outside the kernel build stamp");
            n += 1;
        }
        assert!(n >= 2, "found only {n} band files under kernels/gpu_dsv41: is the walk right?");
    }

    /// W2: the lockstep words of AGREE16 are a protocol change, named as one.
    #[test]
    fn the_version_token_names_protocol_kernels_and_sources() {
        assert_eq!(PROTOCOL_VERSION, 6, "AGREE16 changed every pairwise lockstep word: protocol 6");
        let v = binary_version();
        assert_eq!(v, format!("v6-k{}-r{}", env!("KERNEL_BUILD_ID"), env!("SOURCE_BUILD_ID")));
        assert!(env!("SOURCE_BUILD_ID").len() == 16 && env!("KERNEL_BUILD_ID").len() == 16);
        assert_eq!(version_match(&v, &v), Ok(()));
        // a difference in any one part refuses, and the message names both tokens
        for other in [v.replacen("v6-", "v5-", 1), format!("{v}0"), v.replacen("-r", "-r0", 1), String::new()] {
            let e = version_match(&v, &other).unwrap_err();
            assert_eq!(e, format!("binary mismatch: head {v} vs node {other}"));
            assert!(version_match(&other, &v).is_err(), "the comparison is symmetric");
        }
    }

    /// FLASHNEXT-DF2-C (ISSUES 116): the drafter decision is ONE helper for both head entries, and an
    /// EXL3 serve config with a DF2 source + a resolved draft dir ships the artifact (config.json
    /// included — the node's geometry install reads it); a non-DF2 source or an empty dir ships nothing.
    #[test]
    fn exl3_serve_config_ships_the_dflash2_drafter_like_the_nvfp4_one() {
        let d = std::env::temp_dir().join(format!("df2c_manifest_{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("config.json"), "{\"hidden_size\": 2560}").unwrap();
        std::fs::write(d.join("model.safetensors"), vec![7u8; 4096]).unwrap();
        let mut cfg = crate::tp::TpConfig::from_opts();
        cfg.mode_serve = true; cfg.exl3 = true; cfg.world = 2;
        cfg.spec_source = "dflash2".into();
        cfg.df2_draft_dir = d.to_string_lossy().to_string();
        let (id, arts) = draft_for_cfg(&cfg).expect("an EXL3 serve with a DF2 source ships its drafter");
        assert_eq!(id, d.file_name().unwrap().to_string_lossy());
        let mut names: Vec<&str> = arts.iter().map(|a| a.logical.as_str()).collect();
        names.sort();
        assert_eq!(names, ["config.json", "model.safetensors"]);
        cfg.spec_source = "mtp".into();
        assert!(draft_for_cfg(&cfg).is_none(), "a non-DF2 source ships no drafter");
        cfg.spec_source = "dflash2-auto".into(); cfg.df2_draft_dir.clear();
        assert!(draft_for_cfg(&cfg).is_none(), "no resolved draft dir: nothing to ship");
        // both head entries hand the SAME decision to head_sync_one (the retain entry passed None)
        let src = src_text!("cluster");
        let retain = &src[src.find("pub fn run_head_retain(").unwrap()..src.find("fn sync_only_exit(").unwrap()];
        assert!(retain.contains("let draft = draft_for_cfg(cfg);") && retain.contains("cfg, draft.as_ref(), srcs)")
                && retain.contains("head_sync_one(node, mid, &per_rank[rank], draft.as_ref())"),
                "run_head_retain must ship draft_for_cfg's decision");
        assert!(!retain.contains("cfg, None, srcs)"), "the EXL3 head may no longer pass None as the drafter");
        std::fs::remove_dir_all(&d).ok();
    }
    /// The node's half, the real `node_handle` over loopback: a head with another version token is told
    /// `binary mismatch` and nothing follows (no manifest is read).
    #[test]
    fn a_node_refuses_a_head_built_from_other_sources() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let node = std::thread::spawn(move || { let (s, _) = l.accept().unwrap(); node_handle(s).map(|_| ()) });
        let mut h = TcpStream::connect(addr).unwrap();
        let theirs = binary_version().replacen("-r", "-rdead", 1);
        send_msg(&mut h, &Msg::Hello { version: theirs.clone(), role: "head".into(), hostname: "t".into() }).unwrap();
        match recv_msg(&mut h).unwrap() {
            Msg::Error { msg } => assert_eq!(msg, format!("binary mismatch: head {theirs} vs node {}", binary_version())),
            other => panic!("the node must refuse, got {other:?}"),
        }
        let e = node.join().unwrap().expect_err("node_handle must fail");
        assert!(format!("{e:#}").contains("binary mismatch"), "{e:#}");
    }

    /// The head's half, the real `head_hello` over loopback. Before AGREE16B the head accepted any Hello:
    /// a node that does not compare (here: a fake one that answers Hello whatever it was sent) got the
    /// manifest and the model. Now the head compares the node's token itself.
    #[test]
    fn a_head_refuses_a_node_built_from_other_sources_even_if_the_node_does_not_check() {
        let run = |node_version: String| -> Result<()> {
            let l = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = l.local_addr().unwrap();
            let fake = std::thread::spawn(move || {
                let (mut s, _) = l.accept().unwrap();
                let got = recv_msg(&mut s).unwrap();
                send_msg(&mut s, &Msg::Hello { version: node_version, role: "node".into(), hostname: "fake".into() }).unwrap();
                // what the head sends next, if anything (None = it hung up: nothing was shipped)
                (got, recv_msg(&mut s).ok())
            });
            let mut s = TcpStream::connect(addr).unwrap();
            let ni = NodeInfo { hostname: "fake".into(), addr };
            let r = head_hello(&mut s, &ni, &binary_version());
            if r.is_ok() { send_msg(&mut s, &Msg::Manifest { model_id: "m".into(), artifacts: Vec::new(), draft: Vec::new() }).unwrap(); }
            drop(s);
            let (first, next) = fake.join().unwrap();
            assert!(matches!(first, Msg::Hello { ref role, .. } if role == "head"));
            assert_eq!(next.is_some(), r.is_ok(), "a refused node is sent nothing after the Hello");
            r
        };
        assert!(run(binary_version()).is_ok(), "the same build pairs");
        let other = binary_version().replacen("-r", "-rdead", 1);
        let e = format!("{:#}", run(other.clone()).expect_err("the head must refuse a node with another version token"));
        assert!(e.contains(&format!("binary mismatch: head {} vs node {other}", binary_version())), "{e}");
        let old = binary_version().replacen("v6-", "v5-", 1);
        assert!(run(old).is_err(), "a v5 node is refused by protocol number");
    }

    /// The two call sites, pinned: `head_sync_one` goes through `head_hello` before it ships the manifest,
    /// and discovery still filters on the same token.
    #[test]
    fn the_head_checks_the_version_before_the_manifest() {
        let src = src_text!("cluster");
        let body = &src[src.find("fn head_sync_one(").unwrap()..];
        let body = &body[..body.find("\nfn ").unwrap()];
        let (hello, manifest) = (body.find("head_hello(&mut s, node, &binary_version())?;").expect("head_sync_one calls head_hello"),
                                 body.find("Msg::Manifest").expect("head_sync_one ships the manifest"));
        assert!(hello < manifest, "the version check comes before the manifest");
        assert!(src.contains("if r.version != binary_version() {"), "discovery filters on the version token");
    }

    /// EXL3-SHARDSHIP — the ISSUES row's contract, exercised end to end on the REAL code path:
    /// a minimal qwen4_exp EXL3 pack is planned by the real `ship_plans` (B32), rank 1's blobs
    /// are materialized exactly as the head's shipper produces the wire bytes (Whole = the
    /// file's bytes, Segment = `write_segment`, Bytes = inline), published and assembled by the
    /// real blob-cache path (`publish_blob` + `assemble_model_dir`), and the REAL node loader
    /// (`Exl3Pack::open`) must open the assembled rank dir: the rank's own expert modules +
    /// every common module and every sidecar the serve boot reads, no foreign expert, the
    /// regenerated index and gb10_shard.json in place — and strictly less than the whole pack.
    /// The wiring pins close the row's two ends: the head ships whenever the plan covers the
    /// model and stamps `ship_shards`; the node's EXL3 boot swaps the head's local pack dir for
    /// the shipped one (never the head's path).
    #[test]
    fn exl3_rank_ship_round_trip_through_the_blob_cache_and_the_node_loader() {
        use std::collections::BTreeMap;
        // V0710: it writes into cache_root() (the process-global --tp-cache): hold the lock like the other cache tests,
        // or a concurrent a_short_node_drive_stops_the_head_before_anything_ships sees its blobs and fails (flaky full runs)
        let _g = CACHE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let deal = crate::exl3_forward::xtp::ep_deal_kind();
        assert_eq!(deal, "interleave", "the fixture pins the default expert deal");

        // ---- the fixture pack: 2 trunk layers x 4 experts (gate/up/down), a common attn
        // module per layer, lm_head, and the sidecars the node's serve boot reads by name.
        let root = std::env::temp_dir().join(format!("shardship_exl3_{}", std::process::id()));
        let pack = root.join("pack");
        std::fs::create_dir_all(&pack).unwrap();
        let (nl, ne) = (2usize, 4usize);
        let mut names: Vec<(String, Vec<usize>)> = Vec::new();
        let mut module = |v: &mut Vec<(String, Vec<usize>)>, name: String| {
            v.push((format!("{name}.trellis"), vec![1, 1, 32])); // words 32 -> k2 4 (kernel-supported)
            for s in ["suh", "svh", "mul1"] { v.push((format!("{name}.{s}"), vec![1])); }
        };
        for l in 0..nl {
            module(&mut names, format!("model.language_model.layers.{l}.self_attn.q_proj"));
        }
        module(&mut names, "lm_head".to_string());
        for l in 0..nl { for e in 0..ne { for proj in ["gate_proj", "up_proj", "down_proj"] {
            module(&mut names, format!("model.language_model.layers.{l}.mlp.experts.{e}.{proj}"));
        } } }
        let mut header: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        let mut data: Vec<u8> = Vec::new();
        for (name, shape) in &names {
            let bytes = shape.iter().product::<usize>() * 4; // F32
            let start = data.len();
            data.extend(std::iter::repeat(0u8).take(bytes));
            header.insert(name.clone(), serde_json::json!({
                "dtype": "F32", "shape": shape, "data_offsets": [start, start + bytes]
            }));
        }
        let hj = serde_json::to_string(&header).unwrap();
        let mut st = (hj.len() as u64).to_le_bytes().to_vec();
        st.extend_from_slice(hj.as_bytes());
        st.extend_from_slice(&data);
        let st_name = "model-00001-of-00001.safetensors";
        std::fs::write(pack.join(st_name), &st).unwrap();
        let mut wmap = serde_json::Map::new();
        for (name, _) in &names { wmap.insert(name.clone(), serde_json::json!(st_name)); }
        std::fs::write(pack.join("model.safetensors.index.json"),
                       serde_json::json!({ "weight_map": wmap }).to_string()).unwrap();
        std::fs::write(pack.join("config.json"), serde_json::json!({
            "model_type": "qwen4_exp",
            "text_config": { "hidden_size": 256, "num_hidden_layers": nl, "num_experts": ne,
                             "num_experts_per_tok": 2, "moe_intermediate_size": 128 }
        }).to_string()).unwrap();
        std::fs::write(pack.join("quantization_config.json"),
                       serde_json::json!({ "quant_method": "exl3", "codebook": "mul1", "bits": 2.0 }).to_string()).unwrap();
        std::fs::write(pack.join("tokenizer.json"), "{}").unwrap();
        std::fs::write(pack.join("generation_config.json"), "{}").unwrap();

        // ---- the row's front door: the plan covers this pack at world 2, never at world 1
        assert_eq!(crate::shard_plan::detect_format(&pack), Some(crate::shard_plan::ShardFormat::Exl3));
        assert_eq!(crate::shard_plan::applies(&pack, 2), Some(crate::shard_plan::ShardFormat::Exl3));
        assert_eq!(crate::shard_plan::applies(&pack, 1), None, "world < 2 never ships shards");

        // ---- the head: the real B32 planning + hashing
        let ships = ship_plans(&pack, 2).expect("the covered pack must plan");
        assert_eq!(ships.len(), 2, "ships[0] is the head (nothing shipped), ships[1] is rank 1");
        assert!(ships[0].artifacts.is_empty() && ships[0].srcs.is_empty());
        let ship = &ships[1];
        let src_total = std::fs::metadata(pack.join(st_name)).unwrap().len()
            + ["config.json", "quantization_config.json", "tokenizer.json", "generation_config.json"]
                .iter().map(|f| std::fs::metadata(pack.join(f)).unwrap().len()).sum::<u64>();
        let ship_total: u64 = ship.artifacts.iter().map(|a| a.size).sum();
        assert!(ship_total < src_total,
                "rank 1 must receive only its shards, not the whole pack ({ship_total} vs {src_total} B)");

        // ---- the wire: every artifact's bytes exactly as the shipper produces them, published
        // into the real blob cache and assembled by the real node-side assembly
        let mut blobs = Vec::new();
        std::fs::create_dir_all(cache_root().join("blobs")).unwrap(); // first publish on a fresh box creates it
        for a in &ship.artifacts {
            let src = ship.srcs.get(&a.logical).expect("every artifact has its plan source");
            let bytes: Vec<u8> = match src {
                crate::shard_plan::PlanSrc::Whole(p) => std::fs::read(p).unwrap(),
                crate::shard_plan::PlanSrc::Bytes(b) => b.clone(),
                crate::shard_plan::PlanSrc::Generated(g) => { let mut v = Vec::new(); g.write_to(&mut v).unwrap(); v }
                crate::shard_plan::PlanSrc::Segment(ps) => {
                    let mut v = Vec::new();
                    crate::shard_plan::write_segment(ps, &mut v).unwrap();
                    v
                }
            };
            assert_eq!(bytes.len() as u64, a.size, "{}: wire bytes must match the manifest size", a.logical);
            let tmp = cache_root().join("blobs").join(format!("tmp.shardship.{}", a.hash));
            std::fs::write(&tmp, &bytes).unwrap();
            publish_blob(&a.hash, &tmp).unwrap();
            blobs.push(blob_path(&a.hash));
        }
        let dir = assemble_model_dir(&ship.model_id, &ship.artifacts).unwrap();

        // ---- the node: the REAL loader opens the assembled rank dir
        let p = crate::exl3::Exl3Pack::open(&dir.to_string_lossy())
            .expect("the assembled rank shard dir must open as an EXL3 pack");
        assert_eq!(p.cfg.num_experts, ne, "the shipped config.json parses");
        let mut expert_modules = 0usize;
        for m in &p.modules {
            if let Some(rest) = m.name.strip_prefix("model.language_model.layers.") {
                if let Some((l, tail)) = rest.split_once('.') {
                    if let Some(experts) = tail.strip_prefix("mlp.experts.") {
                        let (e, _) = experts.split_once('.').expect("expert tensor name");
                        let own = crate::exl3_forward::xtp::ep_layer_owners_kind(&deal, ne, 2, l.parse().unwrap()).unwrap();
                        assert_eq!(own[e.parse::<usize>().unwrap()], 1,
                                   "rank 1's dir must hold no foreign expert: {}", m.name);
                        expert_modules += 1;
                    }
                }
            }
        }
        assert_eq!(expert_modules, nl * (ne / 2) * 3, "own gate/up/down modules only, every layer");
        for f in ["config.json", "quantization_config.json", "tokenizer.json",
                  "generation_config.json", "model.safetensors.index.json", "gb10_shard.json"] {
            assert!(dir.join(f).is_file(), "{f} must ship to the rank dir");
        }
        let info: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("gb10_shard.json")).unwrap()).unwrap();
        assert_eq!(info["rank"].as_u64(), Some(1), "the shard info names this rank");
        assert!(dir.join("config.json").symlink_metadata().unwrap().file_type().is_symlink(),
                "the assembled files are blob-cache links: the node never needed the head's path");

        // ---- the row's two wiring ends, pinned against regression
        let src = src_text!("cluster");
        let retain = &src[src.find("pub fn run_head_retain(").unwrap()..src.find("fn sync_only_exit(").unwrap()];
        assert!(retain.contains("applies(model_dir, world"),
                "the head must ship whenever the shard plan covers the model");
        let one = &src[src.find("fn head_sync_one(").unwrap()..];
        assert!(one.contains("node_cfg.ship_shards = srcs.is_some()"),
                "the head must stamp ship_shards when the rank's shards shipped");
        let msrc = src_text!("main");
        assert!(msrc.contains("if tpc.ship_shards {") &&
                msrc.contains("tpc.exl3_pack_dir = dir.to_string_lossy().to_string();"),
                "the node's EXL3 boot must load the shipped shard dir, not the head's local path");

        std::fs::remove_dir_all(&dir).ok();
        for b in blobs { std::fs::remove_file(b).ok(); }
        std::fs::remove_dir_all(&root).ok();
    }
