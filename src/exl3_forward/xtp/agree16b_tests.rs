    #[test]
    fn the_harness_passes_compare_the_peers_agree_word() {
        let src: String = src_text!("exl3_forward/xtp").lines().map(|l| l.split("//").next().unwrap_or("")).collect::<Vec<_>>().join("\n");
        let src = &src[..src.find("mod agree16b_tests").unwrap()];   // (not this module's own needles)
        let raw = ["agree", "_ext("].concat();
        assert_eq!(src.matches(&raw).count(), 0, "a harness pass calls the raw agree and can only test `is_none()` on it");
        let probe = ["probe", "_agree("].concat();
        assert_eq!(src.matches(&probe).count(), 4, "one definition and the three passes (identity, greedy, timing)");
        let def = &src[src.find("fn probe_agree(").expect("probe_agree")..];
        let def = &def[..def.find("\n}").unwrap()];
        assert!(def.contains("crate::exl3_serve::tp_agree_eq(step, 0, 1, argmax)") && def.contains("abort_link()"), "{def}");
        // what that comparison does with a peer whose argmax differs (the case the old sites let through)
        let peer = |am: u32| crate::net::agree_token_reply(crate::net::agree_token(7, 0, 1, am));
        assert_eq!(crate::exl3_serve::tp_agree_decide(0, 1, 1234, Some(peer(1234))), Ok(()));
        for other in [1235u32, 0, 1234 ^ (1 << 31), 1234 ^ (1 << 27)] {
            assert!(crate::exl3_serve::tp_agree_decide(0, 1, 1234, Some(peer(other))).is_err(), "argmax 1234 vs {other}");
        }
        assert!(crate::exl3_serve::tp_agree_decide(0, 1, 1234, None).is_err(), "an aborted link");
    }

    /// S-B9-REL-INTEG A5: a divergence that flips an argmax fails the per-step tripwire INSIDE pass 1; the class
    /// table of what was compared must be printed before the error propagates (it used to return through `?` first).
    #[test]
    fn a_pass_one_error_still_prints_the_class_table() {
        let mut nb = std::collections::BTreeMap::new();
        nb.insert("yatt".to_string(), 3usize);
        nb.insert("prefill:xpre".to_string(), 1usize);
        let l = super::rank_identity_abort_line(480, 10, 7, 96, 2, 1, &nb, Some("step 9 buffer yatt12"),
                                                "TP agree() failed at step 9 (argmax 5): ranks diverged");
        assert!(l.starts_with("G-T1-b RANK_IDENTITY FAIL"), "{l}");
        for needle in ["mismatches by class {\"prefill:xpre\": 1, \"yatt\": 3}", "steps fully identical 7/10",
                       "first: step 9 buffer yatt12", "2 chunks (1 bad)", "the error: TP agree() failed at step 9"] {
            assert!(l.contains(needle), "{needle:?} missing from {l}");
        }
        // the wiring: pass 1's Result is inspected (Err prints the line, then returns it), not `?`-propagated
        let src: String = src_text!("exl3_forward/xtp").lines().map(|l| l.split("//").next().unwrap_or("")).collect::<Vec<_>>().join("\n");
        let src = &src[..src.find("mod agree16b_tests").unwrap()];
        let at = src.find("let r1 = trace_pass(").expect("pass 1 keeps its Result");
        let tail = &src[at..];
        let tail = &tail[..tail.find("let pass1_s").unwrap()];
        assert!(tail.contains("rank_identity_abort_line(") && tail.contains("return Err(e)"), "{tail}");
    }
