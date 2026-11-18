//! Unit tests of `/metrics`: render shape, counters, KV pool gauges, scheduler age. Map: `mod.rs`.

    /// H3 tests drive the same scheduler statics — serialize them so busy/idle reads are exact.
    /// (pub(crate): the keep-alive test in server.rs reads the same gauges.)
    pub(crate) static SCHED_GAUGE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// S-B9-TOOLPARSE3 T1: the parser's shadow counters render with every series present (zeros
    /// included, so "no disagreement" and "not running" differ), a disagreement lands in its own
    /// cell, and the HELP texts say what a cell means.
    #[test]
    fn toolcall_shadow_counters_render_and_count() {
        use crate::tools::{DiffKind, Fewer, Other, Rules, Shadow, ShadowDiff};
        let _g = SCHED_GAUGE_LOCK.lock().unwrap();
        let before = super::render(true);
        let get = |s: &str, name: &str| -> f64 {
            s.lines().find(|l| l.starts_with(name)).and_then(|l| l.rsplit(' ').next()).and_then(|v| v.parse().ok()).unwrap_or(f64::NAN)
        };
        let cell = "velogb10_toolcall_parse_disagreement_total{other=\"cost\",kind=\"call_count\",fewer=\"other\"}";
        assert_eq!(before.matches("velogb10_toolcall_parse_disagreement_total{").count(), 36, "every series is present");
        let c0 = get(&before, cell);
        let b0 = get(&before, "velogb10_toolcall_blocks_total");
        let sh = Shadow {
            skipped: None,
            blocks: 2,
            compared: [0, 2, 2, 1],
            cost_declined: 0,
            diffs: vec![ShadowDiff { other: Other::Cost, kind: DiffKind::CallCount, fewer: Fewer::Other, hash: "0123456789abcdef".into(), len: 300, calls_served: 2, calls_other: 1 }],
        };
        super::toolcall_shadow(&sh, &Rules::V073);
        super::toolcall_blocks(2);
        let after = super::render(true);
        // (other tests parse through `parse_served` on parallel threads and bump the same process-wide
        // counters, so a delta is a lower bound)
        assert!(get(&after, cell) >= c0 + 1.0);
        assert!(get(&after, "velogb10_toolcall_blocks_total") >= b0 + 2.0);
        assert!(after.contains("velogb10_toolcall_rules_info{pairing=\"off\",block=\"after-function\",unclosed=\"drop\"} 1"));
        for h in ["velogb10_toolcall_parse_disagreement_total", "velogb10_toolcall_shadow_compared_total", "velogb10_toolcall_blocks_total"] {
            let help = after.lines().find(|l| l.starts_with(&format!("# HELP {h} "))).expect(h);
            assert!(help.len() > 120, "{h}: the HELP text says what a cell means");
        }
        assert!(after.contains("fewer says which of the two readings returned FEWER calls"));
    }

    #[test]
    fn render_has_longrun_gauges() {
        let _g = SCHED_GAUGE_LOCK.lock().unwrap();
        // Same value the sibling test asserts (2): initialises START and can never race it.
        super::set_max_batch(2);
        super::reject_request();
        super::stream_backlog_cancel();
        super::sched_step();
        super::sched_idle();
        super::set_graph_entries(3);
        let s = super::render(true);
        for name in [
            "velogb10_process_rss_bytes", "velogb10_thread_count", "velogb10_open_fds",
            "velogb10_mem_available_bytes", "velogb10_scheduler_steps_total",
            "velogb10_scheduler_busy", "velogb10_scheduler_last_step_age_seconds",
            "velogb10_graph_cache_entries", "velogb10_graph_cache_mixed_step_entries", "velogb10_log_lines_dropped_total",
            "velogb10_graph_capture_invalidated_total", "velogb10_graph_capture_failed_total",
            "velogb10_requests_rejected_total", "velogb10_streams_cancelled_backlog_total",
            "velogb10_lane_pack_rounds_total", "velogb10_lane_pack_skip_total", "velogb10_lane_pack_lane_rounds_total",
            "velogb10_lane_pack_disabled",
            "velogb10_spec_lanes_est_tok_s", "velogb10_spec_lanes_shared_scale", "velogb10_spec_lanes_busy_lanes",
            "velogb10_spec_lanes_decisions_total", "velogb10_spec_lanes_switches_total",
            "velogb10_spec_lanes_arm_ms_total", "velogb10_spec_lanes_arm_tok_total",
            "velogb10_spec_lanes_measured_tok_s", "velogb10_spec_lanes_probes_total", "velogb10_spec_lanes_meter", "velogb10_spec_lanes_meter_tp_sync_total", "velogb10_spec_lanes_meter_replace_total",
        ] {
            assert!(s.contains(name), "missing gauge {name}");
        }
        for l in s.lines().filter(|l| !l.starts_with('#')) {
            let v = l.rsplit(' ').next().unwrap();
            assert!(v.parse::<f64>().is_ok(), "not a sample line: {l}");
        }
        assert!(s.lines().any(|l| l.starts_with("velogb10_requests_rejected_total") && l.ends_with(" 1")));
        // SURFACE2 item 6: HELP text is read by operators — no internal audit tag (LR-n), and the two
        // `_entries` gauges say what a gauge holds (graphs held NOW), not "captured since boot".
        for l in s.lines().filter(|l| l.starts_with("# HELP ")) {
            assert!(!l.contains("LR-"), "internal audit tag in a HELP text: {l}");
        }
        for name in ["velogb10_graph_cache_entries", "velogb10_graph_cache_mixed_step_entries"] {
            let h = s.lines().find(|l| l.starts_with(&format!("# HELP {name} "))).unwrap();
            assert!(h.contains("holds now") && !h.contains("since boot"), "{h}");
            assert!(s.contains(&format!("# TYPE {name} gauge\n")));
        }
    }

    /// H3 (REL_V0_7_3): the busy/idle transitions and the END-of-work age timestamp, on the
    /// REAL gauge functions (what the schedulers call): idle -> 0, round start -> 1, and
    /// S-B9-REL-SURFACE2B R6(d): the HELP texts say what the code does. The steps counter moves
    /// when a round STARTS (`sched_step`), and busy is set there and nowhere else, so an idle
    /// server prefilling its first request reads busy 0 (source pins: the two writers).
    #[test]
    fn scheduler_help_texts_say_started_and_decode_round() {
        let text = super::render(true);
        let help = |name: &str| text.lines().find(|l| l.starts_with(&format!("# HELP {name} "))).unwrap_or_else(|| panic!("{name}")).to_string();
        let steps = help("velogb10_scheduler_steps_total");
        assert!(steps.contains("rounds STARTED since boot") && !steps.contains("completed since"), "{steps}");
        let busy = help("velogb10_scheduler_busy");
        assert!(busy.contains("from the start of a decode round") && busy.contains("still reads 0 while it admits and prefills its first request"), "{busy}");
        let src = src_text!("metrics");
        let code = &src[..src.find("mod tests").unwrap()];
        assert_eq!(code.matches(&["SCHED_STEPS.", "fetch_add"].concat()).count(), 1);
        assert_eq!(code.matches(&["SCHED_BUSY.", "store(1"].concat()).count(), 1);
        let step = &code[code.find("pub fn sched_step()").unwrap()..code.find("pub fn sched_touch()").unwrap()];
        assert!(step.contains("SCHED_STEPS.fetch_add") && step.contains("SCHED_BUSY.store(1"), "both are written when a round starts");
    }

    /// `sched_touch` is what writes the age (a completed unit of work), so a scrape right
    /// after work reads a small age even though the round started earlier.
    #[test]
    fn sched_gauges_busy_idle_and_age_transitions() {
        let _g = SCHED_GAUGE_LOCK.lock().unwrap();
        super::set_max_batch(2); // initialises START (age timestamps need it)
        let busy = || super::render(true).lines()
            .find(|l| l.starts_with("velogb10_scheduler_busy")).unwrap()
            .rsplit(' ').next().unwrap().to_string();
        super::sched_idle();
        assert_eq!(busy(), "0", "blocked idle-waiting must read busy 0 (H3: it stuck at 1)");
        super::sched_step();
        assert_eq!(busy(), "1", "a started round must read busy 1");
        std::thread::sleep(std::time::Duration::from_millis(80));
        super::sched_touch(); // the round COMPLETES now — the age measures from HERE
        std::thread::sleep(std::time::Duration::from_millis(30));
        let age = super::render(true).lines()
            .find(|l| l.starts_with("velogb10_scheduler_last_step_age_seconds")).unwrap()
            .rsplit(' ').next().unwrap().parse::<f64>().unwrap();
        assert!(age < 0.060, "age {age} must count from sched_touch (end of work): from the round start it would be >= 0.110");
        assert!(age >= 0.0, "age is a sane non-negative gauge");
        // S-B9-REL-BENCHFIX2 (W4): the keep-alive's liveness read is these same two gauges
        let st = super::sched_state();
        assert!(st.busy, "the round has not gone idle yet");
        let work = st.work_age.expect("a unit of work completed");
        assert!(work >= std::time::Duration::from_millis(30) && work < std::time::Duration::from_secs(30),
                "counts from sched_touch: {work:?}");
        super::sched_idle();
        assert!(!super::sched_state().busy);
        assert_eq!(busy(), "0", "idle again reads 0");
    }

    fn sample(s: &str, name: &str) -> u64 {
        s.lines().find(|l| l.starts_with(name) && l[name.len()..].starts_with(' '))
            .unwrap_or_else(|| panic!("no sample line for {name}"))
            .rsplit(' ').next().unwrap().parse().unwrap()
    }

    /// Issue #24: the rendering of the KV pool gauges — names, HELP/TYPE lines, the per-lane label,
    /// the byte components, and the documented relations between the series.
    #[test]
    fn kv_pool_renders_the_documented_series() {
        let p = super::kv_pool_new(4, 32768);
        // lane 0 runs (1,500 positions), lane 1 is prefilling (4,095 so far), lane 2 is free and
        // keeps a 900-token prefix snapshot, lane 3 never served
        p.publish([(1500, 0), (4095, 0), (0, 900), (0, 0)].into_iter());
        let mut s = String::new();
        p.render(&mut s, Some([1_000_000, 20_000, 300]));
        assert_eq!(sample(&s, "velogb10_kv_positions_capacity"), 4 * 32768);
        assert_eq!(sample(&s, "velogb10_kv_positions_used"), 5595);
        assert_eq!(sample(&s, "velogb10_kv_positions_peak"), 5595);
        assert_eq!(sample(&s, "velogb10_kv_positions_cached"), 900);
        for (lane, used, cached) in [(0, 1500, 0), (1, 4095, 0), (2, 0, 900), (3, 0, 0)] {
            assert!(s.contains(&format!("velogb10_kv_lane_positions_used{{lane=\"{lane}\"}} {used}\n")), "{s}");
            assert!(s.contains(&format!("velogb10_kv_lane_positions_peak{{lane=\"{lane}\"}} {used}\n")));
            assert!(s.contains(&format!("velogb10_kv_lane_positions_cached{{lane=\"{lane}\"}} {cached}\n")));
            assert!(s.contains(&format!("velogb10_kv_lane_positions_capacity{{lane=\"{lane}\"}} 32768\n")));
        }
        assert!(s.contains("velogb10_kv_allocated_bytes{kind=\"trunk_kv\"} 1000000\n"));
        assert!(s.contains("velogb10_kv_allocated_bytes{kind=\"indexer_keys\"} 20000\n"));
        assert!(s.contains("velogb10_kv_allocated_bytes{kind=\"draft_head\"} 300\n"));
        // every series has exactly one HELP and one TYPE line, and every sample line parses
        for name in ["velogb10_kv_positions_capacity", "velogb10_kv_positions_used", "velogb10_kv_positions_peak",
                     "velogb10_kv_positions_cached", "velogb10_kv_lane_positions_capacity", "velogb10_kv_lane_positions_used",
                     "velogb10_kv_lane_positions_peak", "velogb10_kv_lane_positions_cached", "velogb10_kv_allocated_bytes"] {
            assert_eq!(s.matches(&format!("# HELP {name} ")).count(), 1, "HELP of {name}");
            assert_eq!(s.matches(&format!("# TYPE {name} gauge\n")).count(), 1, "TYPE of {name}");
        }
        for l in s.lines().filter(|l| !l.starts_with('#')) {
            assert!(l.rsplit(' ').next().unwrap().parse::<f64>().is_ok(), "not a sample line: {l}");
        }
        // S-B9-REL-SURFACE2B R3: `_total` is the counter suffix. These are gauges, so none of them
        // carries it (`promtool check metrics` flags a non-counter named `_total`, and dashboards
        // offer rate() on one): capacity for the two position series, allocated_bytes for the third.
        // The same holds for every gauge of the whole /metrics page.
        for text in [s.clone(), super::render(true)] {
            for l in text.lines().filter(|l| l.starts_with("# TYPE ") && l.ends_with(" gauge")) {
                let name = l.split(' ').nth(2).unwrap();
                assert!(!name.ends_with("_total"), "a gauge named like a counter: {name}");
            }
        }
        assert!(!s.contains("kv_positions_total") && !s.contains("kv_lane_positions_total") && !s.contains("kv_bytes_total"));
        // HELP honesty: positions are not bytes, and "used" is what RUNNING requests hold
        let help = |n: &str| s.lines().find(|l| l.starts_with(&format!("# HELP {n} "))).unwrap().to_string();
        assert!(help("velogb10_kv_positions_capacity").contains("not a byte"));
        assert!(help("velogb10_kv_positions_used").contains("running now"));
        assert!(help("velogb10_kv_positions_cached").contains("Not part of positions_used"));
        // the byte series is absent when the engine registered no byte counts
        let mut s2 = String::new();
        p.render(&mut s2, None);
        assert!(!s2.contains("velogb10_kv_allocated_bytes"));
        // the finished request frees lane 0; lane 1 becomes a running lane
        p.publish([(0, 1400), (4200, 0), (0, 900), (0, 0)].into_iter());
        let mut s3 = String::new();
        p.render(&mut s3, None);
        assert_eq!(sample(&s3, "velogb10_kv_positions_used"), 4200);
        assert_eq!(sample(&s3, "velogb10_kv_positions_peak"), 5595, "the peak is a high-water mark");
        assert_eq!(sample(&s3, "velogb10_kv_positions_cached"), 2300);
        assert!(s3.contains("velogb10_kv_lane_positions_peak{lane=\"0\"} 1500\n"));
        assert!(s3.contains("velogb10_kv_lane_positions_peak{lane=\"1\"} 4200\n"));
    }

    /// Issue #24 invariants over an arbitrary publication sequence (including values an engine bug
    /// could produce: past the window): used <= total, peak >= used, a peak never decreases, the
    /// per-lane series sum to the aggregate, and each lane's peak bounds its own series.
    #[test]
    fn kv_pool_peak_never_decreases_and_used_never_exceeds_total() {
        let (lanes, window) = (5usize, 1000u64);
        let p = super::kv_pool_new(lanes, window as usize);
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut rnd = move || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x };
        let (mut last_peak, mut last_lane_peak) = (0u64, vec![0u64; lanes]);
        for step in 0..20_000 {
            let vals: Vec<(usize, usize)> = (0..lanes).map(|_| match rnd() % 4 {
                0 => (0, (rnd() % 1500) as usize),       // free lane, maybe an over-long "snapshot"
                1 => ((rnd() % 1300) as usize, 0),       // running, maybe past the window
                _ => ((rnd() % 900) as usize, 0),
            }).collect();
            // more lanes than the pool has are ignored, fewer leave the rest untouched
            let extra = if step % 97 == 0 { vec![(5000usize, 5000usize)] } else { vec![] };
            p.publish(vals.iter().copied().chain(extra));
            let used = p.used.load(super::Relaxed);
            let peak = p.peak.load(super::Relaxed);
            assert!(used <= lanes as u64 * window, "used {used} exceeds total");
            assert!(peak >= used && peak >= last_peak, "peak {peak} (was {last_peak}) vs used {used}");
            last_peak = peak;
            let mut sum = 0;
            for (i, l) in p.lanes.iter().enumerate() {
                let (u, pk) = (l.used.load(super::Relaxed), l.peak.load(super::Relaxed));
                assert!(u <= window && l.cached.load(super::Relaxed) <= window);
                assert_eq!(u, (vals[i].0 as u64).min(window));
                assert!(pk >= u && pk >= last_lane_peak[i] && pk <= peak);
                last_lane_peak[i] = pk;
                sum += u;
            }
            assert_eq!(sum, used, "the per-lane series sum to the aggregate");
        }
        assert!(last_peak > 0);
    }

    /// Issue #24: a concurrent scrape never sees `peak < used` (the peak is raised first), and the
    /// label cardinality is bounded.
    #[test]
    fn kv_pool_scrape_is_consistent_under_a_live_writer_and_lane_series_are_bounded() {
        let p = std::sync::Arc::new(super::kv_pool_new(2, 1_000_000));
        let w = p.clone();
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop_w = stop.clone();
        let writer = std::thread::spawn(move || {
            let mut i = 0usize;
            while !stop_w.load(super::Relaxed) { i += 1; w.publish([(i, 0), (i / 2, 0)].into_iter()); }
        });
        for _ in 0..20_000 {
            // read used first (Acquire), then the peak: what the writer stored before `used` is visible
            let used = p.used.load(super::Acquire);
            let peak = p.peak.load(super::Relaxed);
            assert!(peak >= used, "scrape saw peak {peak} < used {used}");
        }
        stop.store(true, super::Relaxed);
        writer.join().unwrap();
        let big = super::kv_pool_new(super::KV_LANE_SERIES_MAX + 6, 10);
        big.publish((0..super::KV_LANE_SERIES_MAX + 6).map(|_| (3, 0)));
        let mut s = String::new();
        big.render(&mut s, None);
        assert_eq!(s.matches("velogb10_kv_lane_positions_used{").count(), super::KV_LANE_SERIES_MAX);
        assert_eq!(sample(&s, "velogb10_kv_positions_used"), 3 * (super::KV_LANE_SERIES_MAX as u64 + 6), "totals cover every lane");
    }

    /// Issue #24: the global wiring — nothing is emitted until an engine registers a pool, then the
    /// scheduler's `kv_publish` reaches `/metrics`.
    #[test]
    fn kv_pool_global_is_absent_until_an_engine_registers_it() {
        // (this is the only test that touches the global pool, so "absent before init" is exact)
        // the pool family only: velogb10_kv_spill_* is a separate family (LANERES2) and may be present
        assert!(!super::render(true).contains("velogb10_kv_pool")
            && !super::render(true).contains("velogb10_kv_positions")
            && !super::render(true).contains("velogb10_kv_allocated_bytes"), "no engine registered a pool yet");
        super::kv_publish([(5, 0)].into_iter()); // a publish before init is a no-op, not a panic
        super::kv_pool_init(3, 4096);
        super::kv_pool_init(9, 9); // first call wins
        super::kv_pool_add_bytes(0, 4096);
        super::kv_pool_add_bytes(0, 4096);
        super::kv_pool_add_bytes(7, 1); // unknown kind: ignored
        super::kv_publish([(100, 0), (0, 50), (0, 0)].into_iter());
        let s = super::render(true);
        assert_eq!(sample(&s, "velogb10_kv_positions_capacity"), 3 * 4096);
        assert_eq!(sample(&s, "velogb10_kv_positions_used"), 100);
        assert_eq!(sample(&s, "velogb10_kv_positions_cached"), 50);
        assert!(s.contains("velogb10_kv_allocated_bytes{kind=\"trunk_kv\"} 8192\n"));
        for l in s.lines().filter(|l| !l.starts_with('#')) {
            assert!(l.rsplit(' ').next().unwrap().parse::<f64>().is_ok(), "not a sample line: {l}");
        }
    }

    #[test]
    fn render_is_prometheus_text_and_counts_a_request() {
        super::set_max_batch(2);
        {
            let mut r = super::Req::start(100);
            r.tok(); r.tok(); r.tok();
            r.cached(40); // v0.7.3 #8.2: the request's prefix-cache hit
            r.finish("length");
        }
        { let _dropped = super::Req::start(7); } // never finished: a cancelled request
        let s = super::render(true);
        assert!(s.contains("velogb10_engine_alive 1"));
        assert!(s.contains("velogb10_max_batch 2"));
        assert!(s.lines().any(|l| l.starts_with("velogb10_requests_total{finish=\"length\"}") && !l.ends_with(" 0")));
        assert!(s.lines().any(|l| l.starts_with("velogb10_requests_total{finish=\"cancelled\"}") && !l.ends_with(" 0")));
        assert!(s.contains("velogb10_time_to_first_token_seconds_bucket{le=\"+Inf\"}"));
        // v0.7.3 #8.2: the cached-prompt-tokens counter, HELP + TYPE + a sample >= 40.
        assert!(s.contains("# HELP velogb10_prompt_tokens_cached_total"));
        assert!(s.contains("# TYPE velogb10_prompt_tokens_cached_total counter"));
        let cv: f64 = s.lines().find(|l| l.starts_with("velogb10_prompt_tokens_cached_total "))
            .expect("cached counter sample line").rsplit(' ').next().unwrap().parse().unwrap();
        assert!(cv >= 40.0, "cached counter should count the request's 40 cached tokens, got {cv}");
        for l in s.lines().filter(|l| !l.starts_with('#')) {
            let v = l.rsplit(' ').next().unwrap();
            assert!(v.parse::<f64>().is_ok(), "not a sample line: {l}");
        }
    }

    /// S-B9-REL-VISIONFIX2 V2b: every lost capture is counted, split by cause; the line prints once
    /// per (kind, shape, cause) and never again; /metrics shows both counters.
    #[test]
    fn lost_captures_are_counted_and_logged_once_per_shape() {
        let _g = SCHED_GAUGE_LOCK.lock().unwrap();
        let (i0, f0) = (super::graph_captures_invalidated(), super::graph_captures_failed());
        let inv = super::CAPTURE_INVALIDATED_STATUS;
        assert!(super::graph_capture_lost("unit-test-kind", "width 8", inv), "the first loss of a shape logs");
        assert!(!super::graph_capture_lost("unit-test-kind", "width 8", inv), "the second loss of the same shape stays silent");
        assert!(super::graph_capture_lost("unit-test-kind", "width 9", inv), "another shape logs");
        assert!(super::graph_capture_lost("unit-test-kind", "width 8", "CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED"), "another cause logs");
        assert_eq!(super::graph_captures_invalidated() - i0, 3);
        assert_eq!(super::graph_captures_failed() - f0, 1);
        let s = super::render(true);
        for name in ["velogb10_graph_capture_invalidated_total", "velogb10_graph_capture_failed_total"] {
            assert!(s.contains(&format!("# TYPE {name} counter")), "{name}");
            assert!(s.lines().any(|l| l.starts_with(name) && !l.starts_with('#')), "{name} has a sample");
        }
    }

    /// S-B9-REL-INTEG C1: the line and the counters' texts say what is counted and what happens next, per
    /// graph family (they promised "captured again at its next use" for three families that are captured
    /// once, at boot), and every failure kind the code now reports is counted as `failed`.
    #[test]
    fn the_lost_capture_texts_say_what_happens_next_per_family() {
        let inv = super::CAPTURE_INVALIDATED_STATUS;
        for kind in ["step", "verify", "packed-verify", "draft-pass", "draft-chain", "reprime", "nvfp4-verify", "df2-inject"] {
            let l = super::capture_lost_line(kind, "width 8", inv);
            assert!(l.contains("captured again at its next use"), "{kind}: {l}");
            assert!(l.contains("from another thread") && l.contains("capturing thread itself"), "{kind}: the cause is not over-attributed: {l}");
        }
        for (kind, needle) in [("nvfp4-decode", "drops ALL decode graphs"), ("nvfp4-decode-sample", "never retried"),
                               ("df2-draft-round", "never retried")] {
            for st in [inv, "BeginCapture CUDA_ERROR_NOT_SUPPORTED", "instantiate CUDA_ERROR_OUT_OF_MEMORY"] {
                let l = super::capture_lost_line(kind, "s", st);
                assert!(l.contains(needle) && !l.contains("captured again at its next use"), "{kind} {st}: {l}");
                assert!(l.contains(if st == inv { "velogb10_graph_capture_invalidated_total" } else { "velogb10_graph_capture_failed_total" }), "{l}");
            }
        }
        // the statuses the newly counted sites pass are `failed`, not `invalidated`
        let _g = SCHED_GAUGE_LOCK.lock().unwrap();
        let (i0, f0) = (super::graph_captures_invalidated(), super::graph_captures_failed());
        for st in ["BeginCapture CUDA_ERROR_UNKNOWN", "instantiate CUDA_ERROR_OUT_OF_MEMORY", "tp_spine_audit: a barrier has a concurrent node"] {
            super::graph_capture_lost("unit-test-sites", st, st);
        }
        assert_eq!((super::graph_captures_invalidated() - i0, super::graph_captures_failed() - f0), (0, 3));
        // the /metrics texts name what is counted
        let s = super::render(true);
        for needle in ["cuStreamBeginCapture refused", "cuGraphInstantiate failed", "graph-spine audit refused",
                       "the three boot-time graphs", "from the capturing thread itself"] {
            assert!(s.contains(needle), "{needle:?} missing from the HELP texts");
        }
        assert!(!s.contains("a thread besides the scheduler touched the device"));
    }
