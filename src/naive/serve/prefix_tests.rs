//! Inline unit tests, moved out of `serve.rs` verbatim (module `prefix_tests`).
//! Declared from `mod.rs`; the body keeps its original indentation and its `use super::*` imports.
    use super::*;

    #[test]
    fn ring_runs_cover_the_window_in_position_order() {
        assert_eq!(ring_runs(0, 128, 4096), vec![(0, 128)]);
        assert_eq!(ring_runs(7168 - 128, 128, 4096), vec![(7040 & 4095, 128)]);
        // a window that wraps the ring: the tail of the ring, then its head
        assert_eq!(ring_runs(4096 - 28, 128, 4096), vec![(4068, 28), (0, 100)]);
        assert_eq!(ring_runs(8192 + 4000, 128, 4096), vec![(4000, 96), (0, 32)]);
        for p0 in [0usize, 1, 4000, 4095, 12345] {
            let r = ring_runs(p0, 128, 4096);
            assert_eq!(r.iter().map(|x| x.1).sum::<usize>(), 128);
            assert_eq!(r[0].0, p0 & 4095);
        }
    }

    #[test]
    fn the_plan_resumes_at_the_held_chunk_boundary_and_snapshots_the_next() {
        let p: Vec<u32> = (0..7475u32).collect();
        // first request: nothing held; the snapshot is the last chunk boundary at or before prompt - margin
        assert_eq!(plan_prefix(None, &p, None, 1024, 128), (0, Some(7168)));
        // the template's message boundary wins over the margin
        assert_eq!(plan_prefix(None, &p, Some(5000), 1024, 128), (0, Some(4096)));
        // the next turn repeats the first 7168 tokens and goes on: resume there; its own snapshot further on
        let mut q = p.clone(); q.extend(9000..12000u32);
        assert_eq!(plan_prefix(Some(&p[..7168]), &q, None, 1024, 128), (7168, Some(10240)));
        // no boundary beyond the held one: resume, keep the held snapshot (None = no new one)
        let mut r = p[..7168].to_vec(); r.extend(1..500u32);
        assert_eq!(plan_prefix(Some(&p[..7168]), &r, None, 1024, 128), (7168, None));
        // a prompt that differs inside the held prefix, or only repeats it, prefills whole
        let mut d = p.clone(); d[100] = 999_999;
        assert_eq!(plan_prefix(Some(&p[..7168]), &d, None, 1024, 128).0, 0);
        assert_eq!(plan_prefix(Some(&p[..7168]), &p[..7168], None, 1024, 128).0, 0);
        // a short prompt: no snapshot (below one chunk), nothing to resume
        assert_eq!(plan_prefix(None, &p[..900], None, 1024, 128), (0, None));
    }

    #[test]
    fn lanes_follow_max_batch() {
        let a = |v: &[&str]| super::lanes_asked(&v.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        assert_eq!(a(&["--max-batch", "4"]), (4, 4));
        assert_eq!(a(&["--max-batch", "2"]), (2, 2));
        assert_eq!(a(&["--max-batch", "16"]), (4, 16)); // capped and announced, not refused
        assert_eq!(a(&[]), (4, 8)); // the generic --max-batch default
        assert_eq!(a(&["--max-batch", "4", "--naive-lanes", "1"]), (1, 1)); // the override wins
        assert_eq!(a(&["--max-batch", "3", "--naive-lanes", "0"]), (3, 3)); // 0 = follow --max-batch
    }

    #[test]
    fn an_admit_without_the_prefix_fields_still_decodes() {
        let j = r#"{"Admit":{"prompt":[1,2,3],"max_new":4,"seed":5,"sp":{"temperature":1.0,"top_p":0.95,"top_k":0,"min_p":0.0,"rep_penalty":1.0,"presence_penalty":0.0,"frequency_penalty":0.0},"min_new":0,"ignore_eos":false}}"#;
        match serde_json::from_str::<Ctl>(j).expect("decode") {
            Ctl::Admit { reuse, snap, .. } => { assert_eq!(reuse, 0); assert_eq!(snap, None); }
            other => panic!("{other:?}"),
        }
    }
