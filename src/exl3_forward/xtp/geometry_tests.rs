    use super::*;

    #[test]
    fn round_pad_aligns_to_a_multiple() {
        for rounds in [2usize, 3, 4] {
            for e in 0..40u64 {
                let pad = round_pad(e, rounds);
                assert!(pad < rounds);
                assert_eq!((e + pad as u64) % rounds as u64, 0, "e {e} rounds {rounds}");
                assert_eq!(pad == 0, e % rounds as u64 == 0);
            }
        }
    }

    #[test]
    fn world_gt2_refusals_leave_world_2_alone() {
        let sp = XtpOpts { sp_fault: Some((3, 1)), ..Default::default() };
        assert!(world_gt2_flag_refusals(2).is_ok() && world_gt2_flag_refusals(1).is_ok());
        assert!(world_gt2_opts_refusals(2, &sp).is_ok() && world_gt2_opts_refusals(1, &sp).is_ok());
        assert!(world_gt2_opts_refusals(4, &XtpOpts::default()).is_ok());
        // TP-4S: the fault is the W=4 SP gate's negative control, so it is admitted at world 4 with SP on and refused
        // everywhere no SP run exists (world 4 with SP off, worlds above 4)
        assert!(opts_refusals_with_sp(4, &sp, true).is_ok() && opts_refusals_with_sp(4, &XtpOpts::default(), false).is_ok());
        for (w, on) in [(4, false), (8, true), (8, false)] {
            let e = opts_refusals_with_sp(w, &sp, on).unwrap_err().to_string();
            assert!(e.contains("--xtp-sp-fault") && e.contains(&format!("world {w}")) && e.contains("silent negative control"), "{e}");
        }
        for w in [3, 5, 8] { assert!(world_gt2_flag_refusals(w).is_err(), "world {w}"); }
        // TP-4E: the DEFAULT dual-rail prefill transport is admitted at world 4 (its second rail is an N-way link)
        assert!(world_gt2_flag_refusals(4).is_ok(), "the default posture must be admitted at world 4: {:?}",
                world_gt2_flag_refusals(4).err());
    }

    /// PACK4-REVIEW B1: `--lane-pack on` is refused by the SAME pre-network hook the other flag
    /// refusals use (head: before discovery; node: before the link), at every world above the gated
    /// one, and the default (off / unset) boots everywhere. Order pinned in source: the hook runs it
    /// BEFORE the world <= 2 early return.
    #[test]
    fn lane_pack_is_refused_by_the_pre_network_hook() {
        let max = crate::lane_pack::LANE_PACK_MAX_WORLD as i32;
        assert_eq!(max, 4, "TP=1, TP=2 and TP=4 (owner 2026-10-10: the TP=4 refusal is lifted)");
        for w in 1..=8 {
            for v in [None, Some(""), Some("off"), Some(" off ")] {
                assert!(lane_pack_world_refusal(w, v).is_ok(), "world {w} {v:?}: the default must boot");
            }
            for v in [Some("on"), Some("2")] {
                assert_eq!(lane_pack_world_refusal(w, v).is_err(), w > max, "world {w} {v:?}");
                if w > max {
                    let e = lane_pack_world_refusal(w, v).unwrap_err().to_string();
                    assert!(e.contains("--lane-pack") && e.contains(&format!("world {w}")), "{e}");
                }
            }
        }
        let src = src_text!("exl3_forward/xtp");
        let f = &src[src.find("pub fn world_gt2_flag_refusals(").unwrap()..];
        let f = &f[..f.find("\n}\n").unwrap()];
        let call = f.find("lane_pack_world_refusal(world, crate::opts::var(crate::opt!(\"lane-pack\")).ok().as_deref())?;").expect("wired");
        assert!(call < f.find("if world <= 2 { return Ok(()); }").unwrap(), "before the early return");
        // the call sites that matter: the serve head before discovery, the node before its link
        let main = src_text!("main");
        let head = &main[main.find("fn run_exl3_tp_serve_head(").unwrap()..];
        assert!(head.find("world_gt2_flag_refusals(world as i32)").unwrap() < head.find("run_head_retain(").unwrap());
        let node = &main[main.find("fn exl3_tp_node_prelink(").unwrap()..];
        assert!(node.contains("world_gt2_flag_refusals(tpc.world as i32)?;"));
    }

    #[test]
    fn overlap_and_sp_refusals_are_world_gated() {
        let fold = parse_pf_overlap(Some("fold")).unwrap();
        assert!(fold & PF_FOLD != 0);
        // world 2 and 1: every code passes untouched
        for w in [1, 2] {
            for c in [0, PF_OVERLAP_DEFAULT, fold, 64, 64 | PF_DUAL] { assert!(world_gt2_overlap_refusal(w, c).is_ok(), "world {w} code {c:#x}"); }
        }
        // world 4: the program-driven modes are admitted, fold is refused loudly
        for s in [None, Some("on"), Some("dual"), Some("dual:64"), Some("inline"), Some("inline:256"), Some("off")] {
            let c = parse_pf_overlap(s).unwrap();
            assert!(world_gt2_overlap_refusal(4, c).is_ok(), "{s:?}");
        }
        let e = world_gt2_overlap_refusal(4, fold).unwrap_err().to_string();
        assert!(e.contains("fold") && e.contains("world 4") && e.contains("world-2"), "{e}");
        // SP: world 2 untouched; world 4 (TP-4S) admits explicit on/hc/router; above world 4 an explicit on/hc/router is
        // refused (no schedule), default and off always fine
        for code in [0, SP_HC, SP_HC | SP_ROUTER] {
            for ex in [false, true] { assert!(world_gt2_sp_refusal(2, ex, code).is_ok()); assert!(world_gt2_sp_refusal(4, ex, code).is_ok()); }
        }
        assert!(world_gt2_sp_refusal(8, false, SEQ_PAR_DEFAULT).is_ok(), "the default ON yields at runtime, it is not a refusal");
        assert!(world_gt2_sp_refusal(8, true, 0).is_ok(), "an explicit off is fine");
        for code in [SP_HC, SP_HC | SP_ROUTER] {
            let e = world_gt2_sp_refusal(8, true, code).unwrap_err().to_string();
            assert!(e.contains("--tp-seq-parallel") && e.contains("world 8") && e.contains("no schedule"), "{e}");
        }
    }

    #[test]
    fn rail2_port_plan_is_disjoint_and_ordered() {
        // world 4: the documented layout
        assert_eq!(rail2_nway_base(4).unwrap(), crate::tp::TP_PORT + 20);
        let r1: Vec<u32> = crate::net::nway_link_ports(crate::tp::TP_PORT, 4);
        let base2 = rail2_nway_base(4).unwrap();
        let r2: Vec<u32> = crate::net::nway_link_ports(base2, 4);
        let t = crate::tp::TP_PORT as u32;
        assert_eq!(r1, vec![t + 1, t + 3, t + 4, t + 5, t + 8, t + 9, t + 13], "rail 1 = liveness + 6 pair ports");
        assert_eq!(r2.len(), 7);
        assert!(r2.iter().all(|p| !r1.contains(p)) && !r1.contains(&(base2 as u32)) && !r2.contains(&(t)), "rails must not share a port");
        // the world-2 rail-2 port (TP_PORT + 10, liveness + 11) is outside both world-4 sets: no stale-listener overlap
        let w2 = [t + TP_AUX_PORT_OFF as u32, t + TP_AUX_PORT_OFF as u32 + 1];
        assert!(w2.iter().all(|p| !r1.contains(p) && !r2.contains(p)));
        // every pair port is unique within a rail (each ordered pair has its own listener)
        for w in [4, 8] {
            let mut v = crate::net::nway_link_ports(crate::tp::TP_PORT, w);
            let n = v.len();
            v.sort(); v.dedup();
            assert_eq!(v.len(), n, "world {w}: duplicate pair port");
            assert_eq!(n, 1 + (w * (w - 1) / 2) as usize);
        }
        // the collision check fires on an overlapping base, and on a u16 overflow; world 2 has no N-way base
        let e = check_rail_ports(4, crate::tp::TP_PORT, crate::tp::TP_PORT + 4).unwrap_err().to_string();
        assert!(e.contains("collision"), "{e}");
        assert!(check_rail_ports(4, crate::tp::TP_PORT, crate::tp::TP_PORT).is_err());
        assert!(check_rail_ports(4, 65000, 65530).unwrap_err().to_string().contains("65535"));
        assert!(rail2_nway_base(2).is_err());
        assert!(rail2_nway_base(8).is_ok());
    }

    /// TP-4F2 (found by the first served boot, F2.2 leg B1: "kernel xq_tp_wait_add_dec_single missing from exl3_bench"): a
    /// kernel launched by name in the serving path must be in the module load list (`MODULE_FNS`, else `get_func` finds
    /// nothing at the first launch) and in the capture pre-resolve list, and defined in the .cu. The micro-probe has its
    /// own load list, so only the served boot exposes this; this test is what would have caught it before hardware.
    #[test]
    fn single_stage_k2_kernel_is_registered_for_serving() {
        const K: &str = "xq_tp_wait_add_dec_single";
        assert!(src_text!("kernels/exl3_bench").contains(&format!("void {K}(")), "{K} is not defined in exl3_bench.cu");
        assert!(crate::exl3_bench::MODULE_FNS.contains(&K), "{K} missing from MODULE_FNS (served boot would fail at the first launch)");
        assert!(src_text!("exl3_forward").contains(&format!("\"{K}\", \"xq_argmax_rows_vp\"")),
                "{K} missing from the graph-capture pre-resolve list in exl3_forward.rs");
    }

    /// TP-4F2: the single-stage ctx's port plan is disjoint from both rails (and from the world-2 rail-2 ports), and the
    /// flag's refusals are exactly: single at a world other than 4, single with the serial decode pair; default rd never refuses.
    #[test]
    fn tp_reduce_default_resolves_per_world() {
        // TP-4Z1 (owner D-TP4-7): `auto` (the default / unset) = single at world 4 on the folded decode transport,
        // rd elsewhere; an explicit value always wins.
        for raw in [None, Some("auto"), Some(" AUTO ")] {
            assert!(tp_reduce_resolve(raw, 4, 1), "auto at world 4 resolves to single ({raw:?})");
            assert!(!tp_reduce_resolve(raw, 2, 1), "auto at world 2 stays rd ({raw:?})");
            assert!(!tp_reduce_resolve(raw, 1, 1), "auto at world 1 stays rd ({raw:?})");
            assert!(!tp_reduce_resolve(raw, 4, 0), "auto with --tp-dec-xport 0 falls back to rd, never refuses ({raw:?})");
        }
        for w in [1, 2, 4] { for dx in [0u8, 1] {
            assert!(!tp_reduce_resolve(Some("rd"), w, dx), "explicit rd always wins (w{w} dx{dx})");
            assert!(tp_reduce_resolve(Some("single"), w, dx), "explicit single always resolves true (refusal is separate) (w{w} dx{dx})");
        } }
        assert!(tp_reduce_resolve(Some("RD "), 4, 1) == false && tp_reduce_resolve(Some(" Single"), 4, 1));
        // the refusal only ever fires on an explicit single outside world 4 / without the folded decode transport
        assert!(tp_reduce_refusal(tp_reduce_resolve(None, 2, 1) , 2, 1).is_ok(), "auto at world 2 never reaches a refusal");
        assert!(tp_reduce_refusal(true, 2, 1).is_err());
    }

    #[test]
    fn tp_reduce_single_ports_and_refusals() {
        let t = crate::tp::TP_PORT as u32;
        let b3 = single_nway_base(4).unwrap();
        assert_eq!(b3 as u32, t + 40);
        let p3 = rail_ports(b3, 4);
        assert_eq!(p3.len(), 8, "base + liveness + 6 pair ports");
        for other in [rail_ports(crate::tp::TP_PORT, 4), rail_ports(rail2_nway_base(4).unwrap(), 4), vec![t + 10, t + 11]] {
            assert!(p3.iter().all(|p| !other.contains(p)), "single-stage ports collide: {p3:?} vs {other:?}");
        }
        assert!(single_nway_base(2).is_err(), "no N-way base at world 2 (the flag is refused there first)");
        // refusals
        for w in [1, 2, 4, 8] { for dx in [0u8, 1] { assert!(tp_reduce_refusal(false, w, dx).is_ok(), "rd never refuses (w{w} dx{dx})"); } }
        assert!(tp_reduce_refusal(true, 4, 1).is_ok());
        for w in [1, 2, 8] { assert!(tp_reduce_refusal(true, w, 1).unwrap_err().to_string().contains("world-4"), "world {w}"); }
        assert!(tp_reduce_refusal(true, 4, 0).unwrap_err().to_string().contains("--tp-dec-xport"));
    }

    #[test]
    fn nway_pair_port_mirrors_the_shim_formula() {
        // native/net_shim.c nway_pair_port: tcp_port + 2 + lo * world + hi. Rebuild it from the C source text
        // when it is reachable (the test runs from the crate root), so a change to the C formula fails here.
        let shim = Ok::<&str, ()>(src_text!("native/net_shim"));
        if let Ok(src) = shim {
            assert!(src.contains("return tcp_port + 2 + lo * world + hi;"), "net_shim.c nway_pair_port changed: update net::nway_link_ports");
            assert!(src.contains("c->liveness_port = tcp_port + 1;"), "net_shim.c liveness port changed: update net::nway_link_ports");
        }
        let v = crate::net::nway_link_ports(1000, 4);
        let expect: Vec<u32> = std::iter::once(1001).chain([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            .iter().map(|&(lo, hi)| 1000 + 2 + lo * 4 + hi)).collect();
        assert_eq!(v, expect);
    }

    #[test]
    fn rail2_preflight_words_and_device_state() {
        assert_ne!(preflight_word(true), preflight_word(false));
        assert!(preflight_failures(&[preflight_word(true); 4]).is_empty());
        let f = preflight_failures(&[preflight_word(true), preflight_word(true), preflight_word(false), preflight_word(true)]);
        assert_eq!(f.len(), 1);
        assert!(f[0].contains("rank 2") && f[0].contains("not usable"), "{f:?}");
        let f = preflight_failures(&[preflight_word(true), 0xDEAD_BEEF, preflight_word(true), 0]);
        assert_eq!(f.len(), 2);
        assert!(f.iter().all(|m| m.contains("desync")), "{f:?}");
        // device state against a fake sysfs tree (state + GID 3 + its type)
        let root = std::env::temp_dir().join(format!("tp4e_ib_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        const GOOD_GID: &str = "0000:0000:0000:0000:0000:ffff:c000:020d\n"; // 192.0.2.13 (RFC 5737), IPv4-mapped
        let mk = |dev: &str, state: Option<&str>, gid: Option<&str>, ty: Option<&str>| {
            let p = root.join(dev).join("ports/1");
            for d in ["gids", "gid_attrs/types"] { std::fs::create_dir_all(p.join(d)).unwrap(); }
            for (rel, v) in [("state", state), ("gids/3", gid), ("gid_attrs/types/3", ty)] {
                if let Some(v) = v { std::fs::write(p.join(rel), v).unwrap(); }
            }
        };
        mk("up", Some("4: ACTIVE\n"), Some(GOOD_GID), Some("RoCE v2\n"));
        mk("down", Some("1: DOWN\n"), Some(GOOD_GID), Some("RoCE v2\n"));
        mk("init", Some("2: INIT\n"), Some(GOOD_GID), Some("RoCE v2\n"));
        mk("defer", Some("5: ACTIVE_DEFER\n"), Some(GOOD_GID), Some("RoCE v2\n"));
        mk("nostate", None, Some(GOOD_GID), Some("RoCE v2\n"));
        mk("nogid", Some("4: ACTIVE\n"), None, Some("RoCE v2\n"));
        mk("zerogid", Some("4: ACTIVE\n"), Some("0000:0000:0000:0000:0000:0000:0000:0000\n"), Some("RoCE v2\n"));
        mk("v1", Some("4: ACTIVE\n"), Some(GOOD_GID), Some("IB/RoCE v1\n"));
        mk("notype", Some("4: ACTIVE\n"), Some(GOOD_GID), None);
        let (ok, why) = rail_device_state(&root, "up", 3);
        assert!(ok && why.contains("ACTIVE") && why.contains("RoCE v2"), "{why}");
        for (d, needle) in [("down", "ports/1/state"), ("init", "ports/1/state"), ("defer", "ports/1/state"),
                            ("nostate", "cannot read"), ("nogid", "cannot read"), ("zerogid", "non-zero GID"),
                            ("v1", "RoCE v2"), ("notype", "cannot read")] {
            let (ok, why) = rail_device_state(&root, d, 3);
            assert!(!ok && why.contains(d) && why.contains(needle), "{d}: {why}");
        }
        // a GID index the fixture does not have is unreadable, never assumed usable
        let (ok, why) = rail_device_state(&root, "up", 5);
        assert!(!ok && why.contains("gids/5") && why.contains("cannot read"), "{why}");
        let (ok, why) = rail_device_state(&root, "absent", 3);
        assert!(!ok && why.contains("does not exist") && why.contains("absent"), "{why}");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The pure decision over fake file text: good, zero GID, wrong type, unreadable, state not 4. Every refusal names
    /// the device, the file path and the expected value.
    #[test]
    fn rail_files_verdict_fails_closed_and_names_everything() {
        let dir = std::path::Path::new("/sys/class/infiniband/roceP2p1s0f1");
        let dev = "roceP2p1s0f1";
        let gid = "fe80:0000:0000:0000:ba3f:d2ff:fe5a:1c31";
        let good = RailFiles { state: Some("4: ACTIVE\n"), gid: Some(gid), gid_type: Some("RoCE v2\n") };
        let (ok, why) = rail_files_verdict(dev, dir, 3, &good);
        assert!(ok && why.contains(dev) && why.contains(gid), "{why}");
        let refused = |f: RailFiles, path: &str, expected: &str| {
            let (ok, why) = rail_files_verdict(dev, dir, 3, &f);
            assert!(!ok, "{path}: must be refused: {why}");
            assert!(why.contains(dev) && why.contains(path) && why.contains(expected), "names device, path, expected: {why}");
        };
        // state: any number but 4, including the substring traps
        for st in ["1: DOWN\n", "2: INIT\n", "3: ARMED\n", "5: ACTIVE_DEFER\n", "INACTIVE\n", "14: ACTIVE\n", "", "\n"] {
            refused(RailFiles { state: Some(st), ..good }, "ports/1/state", "4: ACTIVE");
        }
        refused(RailFiles { state: None, ..good }, "ports/1/state", "4: ACTIVE");
        // GID: unreadable, all zero (several spellings), empty, not a GID
        refused(RailFiles { gid: None, ..good }, "ports/1/gids/3", "non-zero GID");
        for z in ["0000:0000:0000:0000:0000:0000:0000:0000\n", "::\n", "0\n", "", "\n", "zzzz:0001\n"] {
            refused(RailFiles { gid: Some(z), ..good }, "ports/1/gids/3", "non-zero GID");
        }
        // type: unreadable, RoCE v1, empty, a superstring
        refused(RailFiles { gid_type: None, ..good }, "ports/1/gid_attrs/types/3", "RoCE v2");
        for t in ["IB/RoCE v1\n", "RoCE v1\n", "", "RoCE v2 (x)\n"] {
            refused(RailFiles { gid_type: Some(t), ..good }, "ports/1/gid_attrs/types/3", "RoCE v2");
        }
        // the index in the paths follows the caller's GID index (the bench's --gid)
        let (ok, why) = rail_files_verdict(dev, dir, 7, &RailFiles { gid: None, ..good });
        assert!(!ok && why.contains("ports/1/gids/7"), "{why}");
    }

    #[test]
    fn prefill_xport_why_names_the_choice_and_labels_its_numbers() {
        // every (world, mode, explicit) cell yields a non-empty reason that says default|explicit
        for w in [2, 4] {
            for m in [0u8, 1, 2] {
                for ex in [false, true] {
                    let s = prefill_xport_why(w, m, ex);
                    assert!(s.contains(if ex { "explicit" } else { "default" }), "{w}/{m}/{ex}: {s}");
                    assert!(s.len() > 40);
                }
            }
        }
        // the world-4 default names dual-rail and labels TP-4A's numbers as projections, not measurements
        let s = prefill_xport_why(4, 2, false);
        assert!(s.contains("dual-rail") && s.contains("PROJECTION") && s.contains("831") && s.contains("1,064") && s.contains("no silent fallback"), "{s}");
        // single/serial at world 4 are flagged as perf opt-outs
        assert!(prefill_xport_why(4, 1, true).contains("OPT-OUT"));
        assert!(prefill_xport_why(4, 0, true).contains("OPT-OUT"));
        // world 2 quotes the MEASURED TP-F numbers
        assert!(prefill_xport_why(2, 2, false).contains("measured") && prefill_xport_why(2, 2, false).contains("993"));
        // the default flag resolves to dual at every world
        assert_eq!(prefill_xport_mode(), 2);
    }

    #[test]
    fn vp_sampled_resolution_is_world_gated() {
        // world <= 2: exactly parse_vp_sampled (no note) — the TP=2 behaviour is untouched
        for w in [1u32, 2] {
            for v in [None, Some("on"), Some("off"), Some("1"), Some("0")] {
                assert_eq!(resolve_vp_sampled(w, v).unwrap(), (parse_vp_sampled(v).unwrap(), None), "world {w} {v:?}");
            }
        }
        for w in [1u32, 2, 4] { assert!(resolve_vp_sampled(w, Some("maybe")).is_err(), "garbage is refused at world {w}"); }
        // world 4 (TP-4H2, D-TP4-3): absent -> ON (no note); explicit on -> ON; explicit off / 0 -> OFF quietly
        assert_eq!(resolve_vp_sampled(4, None).unwrap(), (1, None));
        assert_eq!(resolve_vp_sampled(4, Some("on")).unwrap(), (1, None));
        assert_eq!(resolve_vp_sampled(4, Some("1")).unwrap(), (1, None));
        assert_eq!(resolve_vp_sampled(4, Some("off")).unwrap(), (0, None));
        assert_eq!(resolve_vp_sampled(4, Some("0")).unwrap(), (0, None));
        // any other world > 2: no kernels -> absent = OFF with a banner note, off quiet, explicit on refused loudly
        for w in [3u32, 8] {
            let (code, note) = resolve_vp_sampled(w, None).unwrap();
            assert_eq!(code, 0, "world {w}");
            assert!(note.map_or(false, |n| n.contains("defaults to OFF")), "world {w}");
            assert_eq!(resolve_vp_sampled(w, Some("off")).unwrap(), (0, None), "world {w}");
            let e = resolve_vp_sampled(w, Some("on")).unwrap_err().to_string();
            assert!(e.contains("world-2 / world-4") && e.contains(&format!("world {w}")), "{e}");
        }
    }

    /// The pack's config under `$HOME/models` (lab boxes); skipped where it is absent.
    fn pack_config() -> std::path::PathBuf {
        std::path::Path::new(&std::env::var("HOME").unwrap_or_default()).join("models/Qwen3.8-Flash-Next-exl3-3.05bpw/config.json")
    }
    const EMBEDDED: &str = r#"{"model_type":"qwen4_exp","text_config":{"hidden_size":2560,"head_dim":256,
        "num_attention_heads":24,"num_key_value_heads":2,"linear_num_key_heads":16,"linear_num_value_heads":48,
        "linear_key_head_dim":128,"linear_value_head_dim":128,"num_experts":512,"num_experts_per_tok":10,
        "moe_intermediate_size":640,"shared_expert_intermediate_size":640,"num_hidden_layers":48,
        "vocab_size":248320,"indexer_n_heads":4,"indexer_head_dim":128,"indexer_budget":2048,
        "indexer_compress_ratio":4,"indexer_kv_heads":1}}"#;

    fn parse(txt: &str, tag: &str) -> crate::qwen::Config {
        static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!("tp4b_cfg_{}_{tag}_{n}.json", std::process::id()));
        std::fs::write(&p, txt).unwrap();
        let c = crate::qwen::Config::from_config_json(p.to_str().unwrap()).unwrap();
        let _ = std::fs::remove_file(&p);
        c
    }

    fn cfg() -> crate::qwen::Config { parse(EMBEDDED, "emb") }

    fn geometry(c: &crate::qwen::Config) -> [usize; 10] {
        [c.num_heads, c.num_kv_heads, c.head_dim, c.lin_num_k_heads, c.lin_num_v_heads, c.lin_k_dim, c.lin_v_dim,
         c.num_experts, c.moe_intermediate_size, c.shared_expert_intermediate_size]
    }

    fn partitions(mut v: Vec<(usize, usize)>, base: usize, total: usize) {
        v.sort();
        let mut at = base;
        for (a, b) in v { assert_eq!(a, at, "gap or overlap at {a}"); at = b; }
        assert_eq!(at, base + total);
    }

    #[test]
    fn embedded_numbers_match_the_real_pack_when_present() {
        let c = cfg();
        assert_eq!(geometry(&c), [24, 2, 256, 16, 48, 128, 128, 512, 640, 640]);
        if let Ok(txt) = std::fs::read_to_string(pack_config()) {
            let real = parse(&txt, "real");
            assert_eq!(geometry(&real), geometry(&c), "the embedded numbers drifted from the pack's config.json");
            assert_eq!(real.vocab_size, 248320);
        }
    }

    #[test]
    fn kv_span_mapping() {
        let table = |nkv: usize, w: usize| (0..w).map(|r| kv_span(nkv, r, w).unwrap()).collect::<Vec<_>>();
        assert_eq!(table(2, 1), [(0, 2)]);
        assert_eq!(table(2, 2), [(0, 1), (1, 1)]);
        assert_eq!(table(2, 4), [(0, 1), (0, 1), (1, 1), (1, 1)]);
        assert_eq!(table(2, 8), [(0, 1), (0, 1), (0, 1), (0, 1), (1, 1), (1, 1), (1, 1), (1, 1)]);
        assert_eq!(table(8, 4), [(0, 2), (2, 2), (4, 2), (6, 2)]);
        assert_eq!(table(4, 4), [(0, 1), (1, 1), (2, 1), (3, 1)]);
        assert!(kv_span(3, 0, 2).is_err() && kv_span(3, 0, 4).is_err(), "neither-way geometries must refuse");
        assert!(kv_span(1, 0, 2).is_err() && kv_span(1, 1, 2).is_err(), "world 2 with one KV head stays refused (as before TP-4B)");
        assert_eq!(table(1, 4), [(0, 1), (0, 1), (0, 1), (0, 1)]);
        assert!(kv_span(2, 4, 4).is_err() && kv_span(0, 0, 2).is_err());
    }

    #[test]
    fn trunk_geometry_per_rank_w1_w2_w4() {
        let c = cfg();
        // (world, q heads, kv heads, gdn k heads, gdn v heads) per rank
        for (w, nh, nkv, lk, lv) in [(1, 24, 2, 16, 48), (2, 12, 1, 8, 24), (4, 6, 1, 4, 12)] {
            let t = trunk_cfg(&c, w).unwrap();
            assert_eq!((t.num_heads, t.num_kv_heads, t.lin_num_k_heads, t.lin_num_v_heads), (nh, nkv, lk, lv), "world {w}");
            assert_eq!((t.head_dim, t.lin_k_dim, t.lin_v_dim, t.hidden_size), (256, 128, 128, 2560));
            assert_eq!(t.num_experts, 512, "the expert count stays global; the deal divides it");
        }
        assert!(trunk_cfg(&c, 3).is_err() && trunk_cfg(&c, 0).is_err());
        let mut one_kv = cfg();
        one_kv.num_kv_heads = 1;
        assert!(trunk_cfg(&one_kv, 2).is_err(), "world 2 with nkv == 1 must keep refusing");
    }

    #[test]
    fn world_1_and_2_equal_the_pre_tp4b_results() {
        let c = cfg();
        for w in [1usize, 2] {
            let t = trunk_cfg(&c, w).unwrap();
            assert_eq!((t.num_heads, t.num_kv_heads, t.lin_num_k_heads, t.lin_num_v_heads),
                       (c.num_heads / w, c.num_kv_heads / w, c.lin_num_k_heads / w, c.lin_num_v_heads / w));
            for rank in 0..w {
                let ts = TrunkSplit { rank, world: w };
                let old_kv = { let s = c.num_kv_heads / w * c.head_dim; (rank * s, (rank + 1) * s) };
                assert_eq!(ts.attn_kv(&c).unwrap(), old_kv, "attn_kv world {w} rank {rank}");
            }
        }
    }

    #[test]
    fn every_rank_query_head_reads_a_kv_head_the_rank_holds() {
        let c = cfg();
        let group = c.num_heads / c.num_kv_heads;
        for w in [1usize, 2, 4] {
            let t = trunk_cfg(&c, w).unwrap();
            let gqa_local = t.num_heads / t.num_kv_heads;
            for r in 0..w {
                let (kv0, kvn) = kv_span(c.num_kv_heads, r, w).unwrap();
                assert_eq!(kvn, t.num_kv_heads);
                for j in 0..t.num_heads {
                    let kv_global = (r * t.num_heads + j) / group;
                    assert!(kv_global >= kv0 && kv_global < kv0 + kvn, "world {w} rank {r} local q {j}");
                    assert_eq!(kv_global - kv0, j / gqa_local, "the kernel's local KV index must be the held head");
                }
            }
        }
        let t4 = trunk_cfg(&c, 4).unwrap();
        assert_eq!(t4.num_heads / t4.num_kv_heads, 6, "the W=4 per-rank GQA ratio");
    }

    #[test]
    fn trunk_slices_partition_every_sharded_dimension() {
        let c = cfg();
        let (nh, nkv, hd) = (c.num_heads, c.num_kv_heads, c.head_dim);
        let (nk, kd, nv, vd) = (c.lin_num_k_heads, c.lin_k_dim, c.lin_num_v_heads, c.lin_v_dim);
        for w in [1usize, 2, 4] {
            let ts = |rank| TrunkSplit { rank, world: w };
            partitions((0..w).map(|r| ts(r).attn_q(&c)).collect(), 0, nh * 2 * hd);
            partitions((0..w).map(|r| ts(r).attn_o(&c)).collect(), 0, nh * hd);
            partitions((0..w).map(|r| ts(r).gdn_v(&c)).collect(), 0, nv * vd);
            partitions((0..w).map(|r| ts(r).gdn_heads(&c)).collect(), 0, nv);
            partitions((0..w).map(|r| ts(r).gdn_qkv(&c)[0]).collect(), 0, nk * kd);
            partitions((0..w).map(|r| ts(r).gdn_qkv(&c)[1]).collect(), nk * kd, nk * kd);
            partitions((0..w).map(|r| ts(r).gdn_qkv(&c)[2]).collect(), 2 * nk * kd, nv * vd);
            // K/V columns: every KV head is held by exactly max(1, w / nkv) ranks, and ranks sharing a head
            // slice the identical columns
            let mut held = vec![0usize; nkv];
            for r in 0..w {
                let (a, b) = ts(r).attn_kv(&c).unwrap();
                assert_eq!(b - a, trunk_cfg(&c, w).unwrap().num_kv_heads * hd);
                for h in a / hd..b / hd { held[h] += 1; }
                let (kv0, _) = kv_span(nkv, r, w).unwrap();
                assert_eq!(a, kv0 * hd);
            }
            assert!(held.iter().all(|&n| n == (w / nkv).max(1)), "world {w}: holders per KV head {held:?}");
        }
    }

    #[test]
    fn shared_expert_and_expert_deals_at_w4() {
        let c = cfg();
        let sh = |w: usize| (0..w).map(|r| TrunkSplit { rank: r, world: w }.shared(c.moe_intermediate_size).unwrap()).collect::<Vec<_>>();
        assert_eq!(sh(1), [(0, 640)]);
        assert_eq!(sh(2), [(0, 384), (384, 640)], "the W=2 384|256 deal is unchanged");
        assert_eq!(sh(4), [(0, 256), (256, 384), (384, 512), (512, 640)], "640 = 5 x 128 -> 2/1/1/1 blocks at W=4");
        for kind in ["interleave", "contig"] {
            for w in [2usize, 4] {
                let mut per = vec![0usize; w];
                for e in 0..c.num_experts { per[ep_owner(e, c.num_experts, w, kind)] += 1; }
                assert!(per.iter().all(|&n| n == c.num_experts / w), "{kind} world {w}: {per:?}");
            }
        }
        assert_eq!(c.num_experts / 4, 128);
        assert_eq!((c.vocab_size / 4, c.vocab_size / 4 / 128), (62080, 485), "vocab shard at W=4");
    }
