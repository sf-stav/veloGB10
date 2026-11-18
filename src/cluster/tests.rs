    use super::*;

    /// B32: the cache ops on per-rank shard models in a THROWAWAY cache (--tp-cache): truthful per-model totals
    /// with shared blobs, remove deletes only the model's unshared blobs (a shared blob, another model's orphan
    /// and an in-flight tmp.* partial stay), an empty manifest leaves no placeholder dir, remove-all clears.
    #[test]
    fn b32_cache_ops_on_shard_models() {
        let tmp = std::env::temp_dir().join(format!("b32_cache_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let _g = CACHE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner()); // tp-cache is process-global
        crate::opts::set(crate::opt!("tp-cache"), &tmp.to_string_lossy());
        let mk = |name: &str, bytes: &[u8]| -> Artifact {
            let mut h = Sha256::new(); h.update(bytes);
            let hash = hex(&h.finalize());
            let p = blob_path(&hash);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, bytes).unwrap();
            Artifact { logical: name.into(), hash, size: bytes.len() as u64 }
        };
        let shared = mk("ngram_embedding.safetensors", &[7u8; 4096]);
        let a1 = mk("seg/experts-L00.safetensors", &[1u8; 1000]);
        let b1 = mk("seg/experts-L00.safetensors", &[2u8; 2000]);
        let orphan = mk("x", &[9u8; 10]);
        std::fs::write(cache_root().join("blobs").join("tmp.123.abc"), b"partial").unwrap();
        let da = assemble_model_dir("M@exl3-w2-r1-interleave", &[shared.clone(), a1.clone()]).unwrap();
        assemble_model_dir("M@exl3-w4-r1-interleave", &[shared.clone(), b1.clone()]).unwrap();
        assert!(da.join("seg/experts-L00.safetensors").exists());
        // empty manifest (legacy EXL3): no placeholder dir
        let pe = assemble_model_dir("Legacy-exl3", &[]).unwrap();
        assert!(!pe.exists(), "an empty manifest must not leave a placeholder model dir");
        assert_eq!(model_blob_hashes("M@exl3-w2-r1-interleave").len(), 2);
        list_cached_models().unwrap();
        // re-assembling the same id with another manifest drops the stale link
        assemble_model_dir("M@exl3-w2-r1-interleave", &[shared.clone()]).unwrap();
        assert!(!da.join("seg/experts-L00.safetensors").exists(), "stale link survived a re-assembly");
        assemble_model_dir("M@exl3-w2-r1-interleave", &[shared.clone(), a1.clone()]).unwrap();
        remove_cached_model("M@exl3-w2-r1-interleave").unwrap();
        assert!(!blob_path(&a1.hash).exists(), "the removed model's unshared blob must go");
        assert!(blob_path(&shared.hash).exists(), "a blob shared with another model must stay");
        assert!(blob_path(&b1.hash).exists(), "another model's blob must stay");
        assert!(blob_path(&orphan.hash).exists(), "another model's orphan is not this model's to delete");
        assert!(cache_root().join("blobs/tmp.123.abc").exists(), "an in-flight partial must stay");
        remove_all_cached_models().unwrap();
        assert!(!blob_path(&shared.hash).exists() && !cache_root().join("models").exists());
        let _ = std::fs::remove_dir_all(&tmp);
    }


    fn node(hostname: &str, ip: &str, port: u16) -> NodeInfo {
        NodeInfo {
            hostname: hostname.to_string(),
            addr: SocketAddr::new(ip.parse().unwrap(), port),
        }
    }

    // Synthetic RoCE rails: rail 1 = 192.0.2.0/24, rail 2 = 198.51.100.0/24.
    fn rails() -> Vec<Roce> {
        vec![
            Roce {
                ib: "rocep1s0f1".into(),
                netdev: "r1".into(),
                ip: "192.0.2.1".parse().unwrap(),
                bcast: "192.0.2.255".parse().unwrap(),
                mask: 0xffffff00,
                rail: 1,
            },
            Roce {
                ib: "roceP2p1s0f1".into(),
                netdev: "r2".into(),
                ip: "198.51.100.1".parse().unwrap(),
                bcast: "198.51.100.255".parse().unwrap(),
                mask: 0xffffff00,
                rail: 2,
            },
        ]
    }

    #[test]
    fn deterministic_rank_order_prefers_rail_then_ip() {
        let ifaces = rails();
        // rail-1 .13, rail-1 .12, rail-2 .99, non-RoCE .1 -> expected order.
        let mut nodes = vec![
            node("a", "10.0.0.1", 29500),
            node("b", "198.51.100.99", 29500),
            node("c", "192.0.2.13", 29500),
            node("d", "192.0.2.12", 29500),
        ];
        assign_ranks(&mut nodes, &ifaces);
        let order: Vec<String> = nodes.iter().map(|n| n.hostname.clone()).collect();
        assert_eq!(order, vec!["d", "c", "b", "a"], "rail desc, then ip asc");

        // Reassign with the same set in a different input order -> identical result.
        let mut shuffled = vec![
            node("d", "192.0.2.12", 29500),
            node("b", "198.51.100.99", 29500),
            node("a", "10.0.0.1", 29500),
            node("c", "192.0.2.13", 29500),
        ];
        assign_ranks(&mut shuffled, &ifaces);
        let order2: Vec<String> = shuffled.iter().map(|n| n.hostname.clone()).collect();
        assert_eq!(order, order2, "order must be input-order independent");
    }

    #[test]
    fn topology_is_full_and_indexed_by_rank() {
        let ifaces = rails();
        let mut nodes = vec![
            node("b", "198.51.100.99", 29500),
            node("c", "192.0.2.13", 29500),
            node("d", "192.0.2.12", 29500),
        ];
        assign_ranks(&mut nodes, &ifaces);
        let topo = build_topology(&nodes, &ifaces, 4).unwrap();
        assert_eq!(topo.len(), 4);
        assert_eq!(topo[0], "192.0.2.1"); // head = rail 1
        assert_eq!(topo[1], "192.0.2.12"); // rail 1, lowest ip
        assert_eq!(topo[2], "192.0.2.13"); // rail 1, next ip
        assert_eq!(topo[3], "198.51.100.99"); // rail 2 last
    }

    #[test]
    fn manifest_filtering_per_rank_and_world2_identity() {
        // Logical paths exactly as model_files() would emit relative to the model dir.
        let logicals = vec![
            "config.json",
            "inference/config.json",
            "rank1/weights.safetensors",
            "rank1/dspark_stage0.safetensors",
            "rank2/weights.safetensors",
            "rank3/weights.safetensors",
            "rank0/weights.safetensors",
        ];

        // world==2 with rank1/ present: rank1 manifest == the pre-P4 set (rank1/* + 2 root files).
        let got = |rank: usize, sharded: bool| -> Vec<&str> {
            logicals.iter().copied().filter(|rel| include_for_rank(rel, rank, sharded)).collect()
        };
        assert_eq!(got(1, true), vec![
            "config.json",
            "inference/config.json",
            "rank1/weights.safetensors",
            "rank1/dspark_stage0.safetensors",
        ]);

        // world==4 with rank1/..rank3/ present: each node gets only its own shard.
        assert_eq!(got(2, true), vec![
            "config.json",
            "inference/config.json",
            "rank2/weights.safetensors",
        ]);
        assert_eq!(got(3, true), vec![
            "config.json",
            "inference/config.json",
            "rank3/weights.safetensors",
        ]);

        // rank0's own shard must never leak into any node manifest.
        assert!(!got(1, true).contains(&"rank0/weights.safetensors"));
        assert!(!got(2, true).contains(&"rank0/weights.safetensors"));
        assert!(!got(3, true).contains(&"rank0/weights.safetensors"));

        // Replicate-if-not-divisible: a rank with no dir gets the whole model.
        let all: Vec<&str> = logicals.iter().copied().filter(|rel| include_for_rank(rel, 2, false)).collect();
        assert_eq!(all.len(), logicals.len());
    }
