    use super::*;

    /// A synthetic EXL3-shaped pack: an index referencing two shards + a sidecar, the loader-read
    /// small files, and files the engine never reads (README, LICENSE, pack_scan.json, *.native).
    fn make_pack(root: &Path, name: &str, readme: &str, shard_b: &[u8], gen_cfg: Option<&str>) -> PathBuf {
        let d = root.join(name);
        std::fs::create_dir_all(&d).unwrap();
        let idx = serde_json::json!({"metadata": {}, "weight_map": {
            "w.a.trellis": "model-00001-of-00002.safetensors",
            "w.b.trellis": "model-00002-of-00002.safetensors",
            "mtp.x": "mtp_hyper_connection_mixer_patch.safetensors"}});
        std::fs::write(d.join("model.safetensors.index.json"), idx.to_string()).unwrap();
        std::fs::write(d.join("model-00001-of-00002.safetensors"), b"shard-a").unwrap();
        std::fs::write(d.join("model-00002-of-00002.safetensors"), shard_b).unwrap();
        std::fs::write(d.join("mtp_hyper_connection_mixer_patch.safetensors"), b"patch").unwrap();
        std::fs::write(d.join("ngram_embedding.safetensors"), b"ngram").unwrap();   // sidecar, not in the index
        std::fs::write(d.join("config.json"), b"{\"model_type\":\"qwen4_exp\"}").unwrap();
        std::fs::write(d.join("quantization_config.json"), b"{\"quant_method\":\"exl3\"}").unwrap();
        std::fs::write(d.join("tokenizer.json"), b"{}").unwrap();
        if let Some(g) = gen_cfg { std::fs::write(d.join("generation_config.json"), g).unwrap(); }
        std::fs::write(d.join("README.md"), readme).unwrap();
        std::fs::write(d.join("LICENSE"), b"license").unwrap();
        std::fs::write(d.join("pack_scan.json"), b"{}").unwrap();
        std::fs::write(d.join("config.json.native"), b"{}").unwrap();
        std::fs::write(d.join("qbench_prompts.md"), b"x").unwrap();
        d
    }

    #[test]
    fn manifest_is_the_loader_read_set_and_ignores_readme() {
        let root = std::env::temp_dir().join(format!("packfix_ut_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        // private hash cache for the test process (never the user's ~/.cache/gb10_tp)
        let _g = CACHE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner()); // tp-cache is process-global
        crate::opts::set(crate::opt!("tp-cache"), root.join("cache").to_string_lossy());
        let a = make_pack(&root, "a", "upstream card", b"shard-b", Some("{\"eos_token_id\": 1}"));
        let b = make_pack(&root, "b", "the owner's model card, different length", b"shard-b", Some("{\"eos_token_id\": 1}"));
        let ma = pack_manifest(&a).unwrap();
        let mb = pack_manifest(&b).unwrap();
        let names: Vec<&str> = ma.files.iter().map(|f| f.0.as_str()).collect();
        assert_eq!(names, vec!["config.json", "generation_config.json", "model-00001-of-00002.safetensors",
                               "model-00002-of-00002.safetensors", "model.safetensors.index.json",
                               "mtp_hyper_connection_mixer_patch.safetensors", "ngram_embedding.safetensors",
                               "quantization_config.json", "tokenizer.json"]);
        for ig in ["LICENSE", "README.md", "config.json.native", "pack_scan.json", "qbench_prompts.md"] {
            assert!(ma.ignored.iter().any(|x| x == ig), "{ig} must be ignored: {:?}", ma.ignored);
        }
        assert_eq!(ma.hash, mb.hash, "a README difference must not change the manifest");
        assert!(pack_manifest_diff(&ma.files, &mb.files).is_empty());
        // warm cache: nothing re-hashed
        assert_eq!(pack_manifest(&a).unwrap().hashed_now, 0);

        // one flipped byte in a weight shard (same size) -> mismatch naming exactly that file
        let c = make_pack(&root, "c", "upstream card", b"shard-B", Some("{\"eos_token_id\": 1}"));
        let mc = pack_manifest(&c).unwrap();
        assert_ne!(ma.hash, mc.hash);
        let d = pack_manifest_diff(&ma.files, &mc.files);
        assert_eq!(d.len(), 1, "{d:?}");
        assert!(d[0].contains("model-00002-of-00002.safetensors") && d[0].contains("!="), "{d:?}");

        // a whitespace change in a loader-read JSON -> mismatch naming it
        let e = make_pack(&root, "e", "upstream card", b"shard-b", Some("{\"eos_token_id\":  1}"));
        let me = pack_manifest(&e).unwrap();
        assert_ne!(ma.hash, me.hash);
        let d = pack_manifest_diff(&ma.files, &me.files);
        assert!(d.len() == 1 && d[0].contains("generation_config.json"), "{d:?}");

        // an optional read file present on one side only -> mismatch naming it
        let f = make_pack(&root, "f", "upstream card", b"shard-b", None);
        let mf = pack_manifest(&f).unwrap();
        let d = pack_manifest_diff(&ma.files, &mf.files);
        assert!(d.len() == 1 && d[0].contains("generation_config.json") && d[0].contains("ABSENT on the node"), "{d:?}");

        // an index-referenced shard missing on the node -> recorded, named
        let g = make_pack(&root, "g", "upstream card", b"shard-b", Some("{\"eos_token_id\": 1}"));
        std::fs::remove_file(g.join("model-00001-of-00002.safetensors")).unwrap();
        let mg = pack_manifest(&g).unwrap();
        let d = pack_manifest_diff(&ma.files, &mg.files);
        assert!(d.len() == 1 && d[0].contains("model-00001-of-00002.safetensors") && d[0].contains(PACK_MISSING), "{d:?}");

        // a symlinked view of a pack hashes identically and reuses the cache (canonical key)
        let v = root.join("view");
        std::fs::create_dir_all(&v).unwrap();
        for ent in std::fs::read_dir(&a).unwrap() {
            let ent = ent.unwrap();
            std::os::unix::fs::symlink(ent.path(), v.join(ent.file_name())).unwrap();
        }
        let mv = pack_manifest(&v).unwrap();
        assert_eq!(mv.hash, ma.hash);
        assert_eq!(mv.hashed_now, 0, "a symlink view must hit the real files' cached hashes");

        // the parallel hasher is bitwise the sequential one
        let jobs: Vec<(PathBuf, u64)> = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "LICENSE"]
            .iter().map(|n| (a.join(n), std::fs::metadata(a.join(n)).unwrap().len())).collect();
        let par = sha256_files_parallel(&jobs, 3).unwrap();
        let seq: Vec<String> = jobs.iter().map(|(p, _)| sha256_file_with(p, None).unwrap().0).collect();
        assert_eq!(par, seq);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn peeked_frames_classify() {
        let mut err = Vec::new();
        send_msg(&mut err, &Msg::Error { msg: "PACK MANIFEST MISMATCH: x".into() }).unwrap();
        assert!(matches!(peek_frame(&err), PeekFrame::Error(m) if m.contains("MISMATCH")));
        assert!(matches!(peek_frame(&err[..err.len() - 1]), PeekFrame::Incomplete));
        assert!(matches!(peek_frame(&err[..3]), PeekFrame::Incomplete));
        // the node's legitimate first frame on the served path (its mirror Ready) is not a failure
        let mut ready = Vec::new();
        send_json(&mut ready, &crate::tp_serve::ServingMsg::Ready).unwrap();
        assert!(matches!(peek_frame(&ready), PeekFrame::Other));
    }

    fn getsockopt_i32(s: &TcpStream, level: i32, name: i32) -> i32 {
        use std::os::fd::AsRawFd;
        let mut v: i32 = -1;
        let mut len = std::mem::size_of::<i32>() as libc::socklen_t;
        let r = unsafe { libc::getsockopt(s.as_raw_fd(), level, name, &mut v as *mut i32 as *mut libc::c_void, &mut len) };
        assert_eq!(r, 0, "getsockopt({level},{name}): {}", std::io::Error::last_os_error());
        v
    }

    /// Audit C3/C9: the retained control stream gets keepalive + user timeout (read back from the kernel). What it
    /// does NOT prove: that a silent peer host is then detected — that needs a dropped link, a hardware drill.
    #[test]
    fn harden_control_stream_sets_keepalive_and_user_timeout() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let c = TcpStream::connect(l.local_addr().unwrap()).unwrap();
        let (s, _) = l.accept().unwrap();
        for st in [&c, &s] {
            assert_eq!(getsockopt_i32(st, libc::SOL_SOCKET, libc::SO_KEEPALIVE), 0, "default: keepalive off");
            harden_control_stream(st).unwrap();
            assert_ne!(getsockopt_i32(st, libc::SOL_SOCKET, libc::SO_KEEPALIVE), 0);
            assert_eq!(getsockopt_i32(st, libc::IPPROTO_TCP, libc::TCP_KEEPIDLE), CTL_KEEPIDLE_S);
            assert_eq!(getsockopt_i32(st, libc::IPPROTO_TCP, libc::TCP_KEEPINTVL), CTL_KEEPINTVL_S);
            assert_eq!(getsockopt_i32(st, libc::IPPROTO_TCP, libc::TCP_KEEPCNT), CTL_KEEPCNT);
            assert_eq!(getsockopt_i32(st, libc::IPPROTO_TCP, libc::TCP_USER_TIMEOUT), CTL_USER_TIMEOUT_MS);
        }
    }

    /// Audit C4: a silent stray connection (health check, telnet) and a close-without-data connection are dropped
    /// and the node keeps listening; the real head's connection is then accepted, hardened, with reads blocking again.
    #[test]
    fn accept_head_drops_silent_strays_and_serves_the_real_head() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let t0 = std::time::Instant::now();
        let stray_silent = TcpStream::connect(addr).unwrap(); // held open, sends nothing
        let stray_closed = TcpStream::connect(addr).unwrap();
        drop(stray_closed);
        let partial = {
            let mut p = TcpStream::connect(addr).unwrap();
            p.write_all(&[0, 0]).unwrap(); // 2 of the 4 length bytes, then silence
            p
        };
        let head = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            let mut h = TcpStream::connect(addr).unwrap();
            send_msg(&mut h, &Msg::Hello { version: "1".into(), role: "head".into(), hostname: "t".into() }).unwrap();
            h
        });
        let (mut s, _) = accept_head(&l, Duration::from_millis(300)).unwrap();
        assert!(t0.elapsed() >= Duration::from_millis(600), "the two silent strays cost one first-frame wait each: {:?}", t0.elapsed());
        assert!(t0.elapsed() < Duration::from_secs(5));
        assert_ne!(getsockopt_i32(&s, libc::SOL_SOCKET, libc::SO_KEEPALIVE), 0, "the accepted head stream is hardened");
        assert_eq!(s.read_timeout().unwrap(), None, "reads block again after the first-frame check");
        match recv_msg(&mut s).unwrap() {
            Msg::Hello { role, .. } => assert_eq!(role, "head", "the first frame on the accepted stream is the real head's Hello (peek consumed nothing)"),
            other => panic!("expected the head's Hello, got {other:?}"),
        }
        let _keep = (stray_silent, partial, head.join().unwrap());
    }

    #[test]
    fn node_watch_disarm_restores_blocking_reads() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let _peer = TcpStream::connect(addr).unwrap();
        let (s, _) = l.accept().unwrap();
        let streams = vec![s];
        let w = watch_nodes_during_bring_up(&streams).unwrap();
        std::thread::sleep(Duration::from_millis(250));
        w.disarm(&streams);
        assert_eq!(streams[0].read_timeout().unwrap(), None);
    }

    fn space_report(need: u64, free: u64) -> Option<SpaceReport> {
        Some(SpaceReport { need, free, dir: "/c".into(), host: "box".into() })
    }

    /// Owner 2026-11-02, the rule itself: a node is short when its drive cannot hold what it would store plus
    /// `SHIP_HEADROOM`; a node with nothing to store or no report is not judged; the line says how much is needed.
    #[test]
    fn a_node_is_short_when_its_drive_cannot_hold_the_shipment_plus_headroom() {
        let rows = vec![
            (1, "10.0.0.1".to_string(), space_report(40_000_000_000, 40_000_000_000 + SHIP_HEADROOM)), // exactly fits
            (2, "10.0.0.2".to_string(), space_report(40_000_000_000, 8_000_000_000)),                  // short
            (3, "10.0.0.3".to_string(), space_report(0, 0)),                                            // stores nothing
            (4, "10.0.0.4".to_string(), None),                                                          // no report
            (5, "10.0.0.5".to_string(), space_report(40_000_000_000, 40_000_000_000)),                 // the headroom is missing
        ];
        let short = space_shortfalls(&rows);
        assert_eq!(short.len(), 2, "{short:?}");
        assert_eq!(short[0], "rank 2, node box (10.0.0.2): needs ~41.0 GB (40.0 GB of model files it does not have yet + \
1.0 GB headroom), has 8.0 GB free on the drive of /c — short by ~33.0 GB");
        assert!(short[1].starts_with("rank 5, node box (10.0.0.5): needs ~41.0 GB") && short[1].ends_with("short by ~1.0 GB"), "{}", short[1]);
        let msg = space_refusal(&short, 5);
        assert!(msg.starts_with("not enough drive space on 2 of 5 TP node(s)") && msg.contains("Nothing was shipped")
                && msg.contains("--cached-models-remove") && msg.contains("--tp-cache"), "{msg}");
    }

    fn blob_artifact(dir: &Path, logical: &str, bytes: &[u8]) -> Artifact {
        std::fs::write(dir.join(logical), bytes).unwrap();
        let mut h = Sha256::new(); h.update(bytes);
        Artifact { logical: logical.into(), hash: hex(&h.finalize()), size: bytes.len() as u64 }
    }
    /// One real `node_handle` on loopback, returning its model dir and its rank on success.
    fn loopback_node() -> (NodeInfo, std::thread::JoinHandle<Result<(PathBuf, i32)>>) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let t = std::thread::spawn(move || {
            let (s, _) = l.accept().unwrap();
            node_handle(s).map(|(dir, _, cfg, _)| (dir, cfg.node_rank))
        });
        (NodeInfo { hostname: addr.ip().to_string(), addr }, t)
    }

    /// Owner 2026-11-02, over loopback with the real node and head code: one node's drive cannot hold its share, so
    /// the head stops after the manifest round, before any byte ships. Each node reported what it would store
    /// (the drafter files announced with the manifest included, a file it gets in both counted once), each node is
    /// told why and stops, and nothing was written to the cache.
    #[test]
    fn a_short_node_drive_stops_the_head_before_anything_ships() {
        let tmp = std::env::temp_dir().join(format!("shipspace_short_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let _g = CACHE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner()); // tp-cache is process-global
        let cache = tmp.join("cache"); // not created: the free space comes from its nearest existing ancestor
        crate::opts::set(crate::opt!("tp-cache"), &cache.to_string_lossy());
        let md = tmp.join("model");
        std::fs::create_dir_all(&md).unwrap();
        let small = blob_artifact(&md, "a.bin", &[1u8; 1000]);
        let dfile = blob_artifact(&md, "d.bin", &[2u8; 5000]);
        let huge = Artifact { logical: "b.bin".into(), hash: "f".repeat(64), size: 1 << 60 }; // no drive holds 1.15 EB
        let draft = ("drafter".to_string(), vec![dfile, small.clone()]);
        let (n1, t1) = loopback_node();
        let (n2, t2) = loopback_node();
        let nodes = vec![n1, n2];
        let mut offers = vec![head_sync_one(&nodes[0], "m", &[small.clone()], Some(&draft)).unwrap(),
                              head_sync_one(&nodes[1], "m", &[small.clone(), huge], None).unwrap()];
        let r1 = offers[0].space.clone().expect("a node reports its drive");
        assert_eq!((r1.need, r1.dir.clone()), (1000 + 5000, cache.display().to_string()), "{r1:?}");
        assert!(r1.free > 0 && r1.host == hostname(), "{r1:?}");
        assert_eq!(offers[1].space.as_ref().unwrap().need, 1000 + (1 << 60));
        let e = format!("{:#}", space_check(&nodes, &mut offers).expect_err("the head must stop"));
        eprintln!("head error: {e}"); // the exact text the head prints (receipt with --nocapture)
        assert!(e.starts_with("not enough drive space on 1 of 2 TP node(s)") && e.contains("Nothing was shipped")
                && e.contains(&format!("rank 2, node {} (127.0.0.1): needs ~", hostname())) && !e.contains("rank 1,")
                && e.contains("short by ~"), "{e}");
        drop(offers);
        for t in [t1, t2] {
            let ne = format!("{:#}", t.join().unwrap().expect_err("each node is told and stops"));
            assert!(ne.starts_with("the head stopped the sync: not enough drive space"), "{ne}");
        }
        assert!(!cache.join("blobs").exists(), "nothing was stored on any node");
        std::fs::remove_dir_all(&tmp).ok();
    }

    /// The two-phase sync still syncs: a node with room passes the check and gets its files, config and rank.
    #[test]
    fn a_node_with_room_is_synced_after_the_check() {
        let tmp = std::env::temp_dir().join(format!("shipspace_room_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let _g = CACHE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        crate::opts::set(crate::opt!("tp-cache"), &tmp.join("cache").to_string_lossy());
        let md = tmp.join("model");
        std::fs::create_dir_all(&md).unwrap();
        let small = blob_artifact(&md, "a.bin", &[3u8; 1000]);
        let (n1, t1) = loopback_node();
        let nodes = vec![n1];
        let mut offers = vec![head_sync_one(&nodes[0], "m", &[small.clone()], None).unwrap()];
        assert_eq!(offers[0].space.as_ref().unwrap().need, 1000);
        space_check(&nodes, &mut offers).expect("room for 1 KB");
        let topo = vec!["127.0.0.1".to_string(); 2];
        let cfg = crate::tp::TpConfig::from_opts();
        let _s = head_ship(&nodes[0], 1, offers.pop().unwrap(), &md, &[small.clone()], &topo, &cfg, None, None).unwrap();
        let (dir, rank) = t1.join().unwrap().expect("the node synced");
        assert_eq!(rank, 1);
        assert_eq!(std::fs::read(dir.join("a.bin")).unwrap(), vec![3u8; 1000]);
        std::fs::remove_dir_all(&tmp).ok();
    }
