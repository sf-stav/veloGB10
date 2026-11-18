    /// TPSHIP (2026-11-07): an EXL3 pack the shard plan does not cover ships whole (no empty-manifest branch is left
    /// in the EXL3 head sync) and the node is told to load that copy (ship_shards), never the head's path.
    #[test]
    fn exl3_packs_outside_the_shard_plan_ship_whole_and_load_the_shipped_copy() {
        let src = src_text!("cluster");
        let retain = &src[src.find("pub fn run_head_retain(").unwrap()..];
        let retain = &retain[..retain.find("\n}\n").unwrap()];
        assert!(!retain.contains("vec![Vec::new(); world as usize]"), "the EXL3 empty-manifest (load at the head's path) branch is gone");
        assert!(retain.contains("build_manifest(model_dir, world)?"), "a pack outside the plan ships whole");
        // ship-space (2026-11-02) split the per-node sync: the shipment and its ship_shards stamp live in head_ship
        let sync = &src[src.find("fn head_ship(").unwrap()..];
        let sync = &sync[..sync.find("\nfn ").unwrap()];
        assert!(sync.contains("node_cfg.ship_shards = srcs.is_some() || (cfg.exl3 && !artifacts.is_empty());"),
                "a node that received an EXL3 pack loads the shipped copy");
    }
