//! The `/metrics` body: process gauges and the `render` function that writes every series in Prometheus text format 0.0.4.
//! Reads the counters of the sibling files; scrape-time work only (module header in `mod.rs`). Map: `mod.rs`.

use super::*;

fn proc_status_kb(field: &str) -> Option<u64> {
    std::fs::read_to_string("/proc/self/status").ok()?.lines()
        .find(|l| l.starts_with(field))?
        .split_whitespace().nth(1)?.parse().ok()
}

fn open_fds() -> Option<u64> {
    let mut n = 0u64;
    let mut rd = std::fs::read_dir("/proc/self/fd").ok()?;
    while let Some(_e) = rd.next() { n += 1; }
    Some(n)
}

/// The `/metrics` body (Prometheus text format 0.0.4).
pub fn render(engine_alive: bool) -> String {
    let mut o = String::with_capacity(8192);
    o.push_str("# HELP velogb10_build_info Build identity (value is always 1).\n# TYPE velogb10_build_info gauge\n");
    o.push_str(&format!("velogb10_build_info{{version=\"{}\",commit=\"{}\",dirty=\"{}\",kernel_build=\"{}\"}} 1\n",
                        env!("CARGO_PKG_VERSION"), option_env!("TUNE_GIT_COMMIT").unwrap_or("unknown"),
                        option_env!("TUNE_GIT_DIRTY").unwrap_or("unknown"), env!("KERNEL_BUILD_ID")));
    o.push_str("# HELP velogb10_engine_alive 1 while the engine serves, 0 once it is dead (/health answers 503).\n");
    o.push_str(&format!("# TYPE velogb10_engine_alive gauge\nvelogb10_engine_alive {}\n", engine_alive as u8));
    if let Some(s) = START.get() {
        o.push_str(&format!("# HELP velogb10_uptime_seconds Seconds since the server started.\n# TYPE velogb10_uptime_seconds gauge\nvelogb10_uptime_seconds {:.1}\n",
                            s.elapsed().as_secs_f64()));
    }
    o.push_str("# HELP velogb10_df2_rounds_total DFlash2 draft rounds served on this boot (the --draft-dir speculator; nvfp4 trunk-path arm; scripts/owner_check.py prints it).\n# TYPE velogb10_df2_rounds_total counter\n");
    o.push_str(&format!("velogb10_df2_rounds_total {}\n", DF2_ROUNDS.load(Relaxed)));
    o.push_str("# HELP velogb10_tp_proxy_parks_total CPUIDLE: times a TP proxy on this process left its spin for the idle tier (a WFE wait after --proxy-idle-us with no epoch owed). 0 on a TP server that has sat idle = the tier never ran; it should grow about once per quiet gap (between requests), not once per step.\n# TYPE velogb10_tp_proxy_parks_total counter\n");
    o.push_str(&format!("velogb10_tp_proxy_parks_total {}\n", crate::net::proxy_parks()));
    o.push_str("# HELP velogb10_tp_proxy_parked_seconds_total CPUIDLE: seconds TP proxies on this process spent in the idle tier instead of spinning a core.\n# TYPE velogb10_tp_proxy_parked_seconds_total counter\n");
    o.push_str(&format!("velogb10_tp_proxy_parked_seconds_total {:.3}\n", crate::net::proxy_parked_secs()));
    {
        let (vf_world, vf_dual, vf_gpu, vf_sp) = crate::dsv41::tp::rdma::fabric_telemetry();
        if vf_world >= 2 {
            o.push_str("# HELP velogb10_velo_fabric_world Active velo-fabric TP world size (2 / 4 / 8).\n# TYPE velogb10_velo_fabric_world gauge\n");
            o.push_str(&format!("velogb10_velo_fabric_world {vf_world}\n"));
            o.push_str("# HELP velogb10_velo_fabric_dual_rail 1 when dual-rail RoCEv2 (400 Gb/s) is active on velo-fabric.\n# TYPE velogb10_velo_fabric_dual_rail gauge\n");
            o.push_str(&format!("velogb10_velo_fabric_dual_rail {vf_dual}\n"));
            o.push_str("# HELP velogb10_velo_fabric_gpu_recv 1 when zero-CPU-bounce GPU-validated direct NIC tail receive is active.\n# TYPE velogb10_velo_fabric_gpu_recv gauge\n");
            o.push_str(&format!("velogb10_velo_fabric_gpu_recv {vf_gpu}\n"));
            o.push_str("# HELP velogb10_velo_fabric_sp_ready 1 when sequence-parallel prefill collectives are active.\n# TYPE velogb10_velo_fabric_sp_ready gauge\n");
            o.push_str(&format!("velogb10_velo_fabric_sp_ready {vf_sp}\n"));
        }
    }
    {
        let (w, slept, early) = crate::host_wait::counters();
        o.push_str("# HELP velogb10_gpu_wait_hybrid_waits_total CPUIDLE --gpu-wait hybrid: GPU waits that slept part of the wait instead of spinning (0 with requests served = the hybrid wait never ran: --gpu-wait spin, or no wait was long enough).\n# TYPE velogb10_gpu_wait_hybrid_waits_total counter\n");
        o.push_str(&format!("velogb10_gpu_wait_hybrid_waits_total {w}\n"));
        o.push_str("# HELP velogb10_gpu_wait_slept_seconds_total CPUIDLE: seconds the waiting thread slept instead of spinning a core.\n# TYPE velogb10_gpu_wait_slept_seconds_total counter\n");
        o.push_str(&format!("velogb10_gpu_wait_slept_seconds_total {slept:.3}\n"));
        o.push_str("# HELP velogb10_gpu_wait_early_total CPUIDLE: hybrid waits whose GPU finished before the spin window opened (each one woke up to one 200 us slice late; should stay a small fraction of the waits).\n# TYPE velogb10_gpu_wait_early_total counter\n");
        o.push_str(&format!("velogb10_gpu_wait_early_total {early}\n"));
        // NIGHT1: the policy and what the driver reported back (block = 4: every wait sleeps until the GPU signals)
        o.push_str("# HELP velogb10_gpu_wait_policy NIGHT1 --gpu-wait: 1 for the resolved policy (spin|hybrid|block).\n# TYPE velogb10_gpu_wait_policy gauge\n");
        o.push_str(&format!("velogb10_gpu_wait_policy{{policy=\"{}\"}} 1\n", crate::host_wait::mode_name()));
        o.push_str("# HELP velogb10_gpu_wait_context_flags NIGHT1: the CUDA primary context scheduling flags this process set and read back (1 = spin, 4 = blocking sync, 0 = not set: the driver's default).\n# TYPE velogb10_gpu_wait_context_flags gauge\n");
        o.push_str(&format!("velogb10_gpu_wait_context_flags {}\n", crate::host_wait::applied_flags()));
    }
    {
        // NIGHT1: the thermal governor (crate::thermal)
        let (on, hot, peak, soft, hard, paced, held, holds, calls) = crate::thermal::counters();
        o.push_str("# HELP velogb10_thermal_guard NIGHT1 --thermal-guard: 1 = GPU work is paced by the box temperature, 0 = off (or not started: no GPU work yet / no thermal zone).\n# TYPE velogb10_thermal_guard gauge\n");
        o.push_str(&format!("velogb10_thermal_guard {}\n", on as u8));
        o.push_str("# HELP velogb10_thermal_celsius NIGHT1: the hottest ACPI board zone now, its peak since boot, and the governor's soft / hard limits (C).\n# TYPE velogb10_thermal_celsius gauge\n");
        o.push_str(&format!("velogb10_thermal_celsius{{which=\"now\"}} {hot:.1}\nvelogb10_thermal_celsius{{which=\"peak\"}} {peak:.1}\nvelogb10_thermal_celsius{{which=\"soft\"}} {soft:.1}\nvelogb10_thermal_celsius{{which=\"hard\"}} {hard:.1}\n"));
        o.push_str("# HELP velogb10_thermal_paced_seconds_total NIGHT1: seconds GPU work was paused between the soft and the hard limit.\n# TYPE velogb10_thermal_paced_seconds_total counter\n");
        o.push_str(&format!("velogb10_thermal_paced_seconds_total {paced:.3}\n"));
        o.push_str("# HELP velogb10_thermal_held_seconds_total NIGHT1: seconds new GPU work waited at or above the hard limit.\n# TYPE velogb10_thermal_held_seconds_total counter\n");
        o.push_str(&format!("velogb10_thermal_held_seconds_total {held:.3}\n"));
        o.push_str("# HELP velogb10_thermal_holds_total NIGHT1: pauses at or above the hard limit.\n# TYPE velogb10_thermal_holds_total counter\n");
        o.push_str(&format!("velogb10_thermal_holds_total {holds}\n"));
        o.push_str("# HELP velogb10_thermal_paced_total NIGHT1: pauses between the soft and the hard limit.\n# TYPE velogb10_thermal_paced_total counter\n");
        o.push_str(&format!("velogb10_thermal_paced_total {calls}\n"));
    }
    o.push_str("# HELP velogb10_exl3_ple_width_accepted_total PLE n-gram rows whose group width K was outside {5,6} but inside 1..=8, loaded and served (B9 width closure; 0 = never ran).\n# TYPE velogb10_exl3_ple_width_accepted_total counter\n");
    o.push_str(&format!("velogb10_exl3_ple_width_accepted_total {}\n", EXL3_PLE_WIDTH_ACCEPTED.load(Relaxed)));
    o.push_str("# HELP velogb10_ple_warm_ahead_chunks_total prefill chunks whose next chunk's n-gram (PLE) rows were read ahead into the page cache (SSD-resident table only; 0 = never ran).\n# TYPE velogb10_ple_warm_ahead_chunks_total counter\n");
    o.push_str(&format!("velogb10_ple_warm_ahead_chunks_total {}\n", PLE_WARM_AHEAD_CHUNKS.load(Relaxed)));
    o.push_str("# HELP velogb10_ple_warm_ahead_rows_total n-gram (PLE) rows covered by the read-ahead.\n# TYPE velogb10_ple_warm_ahead_rows_total counter\n");
    o.push_str(&format!("velogb10_ple_warm_ahead_rows_total {}\n", PLE_WARM_AHEAD_ROWS.load(Relaxed)));
    o.push_str("# HELP velogb10_ple_warm_ahead_reads_total deduplicated 4 KB page reads the read-ahead issued.\n# TYPE velogb10_ple_warm_ahead_reads_total counter\n");
    o.push_str(&format!("velogb10_ple_warm_ahead_reads_total {}\n", PLE_WARM_AHEAD_READS.load(Relaxed)));
    o.push_str("# HELP velogb10_exl3_w4moe_width_served_total persistent grouped-expert (--w4moe) launches at a non-3 expert width (B10 width closure; 0 = never ran).\n# TYPE velogb10_exl3_w4moe_width_served_total counter\n");
    o.push_str(&format!("velogb10_exl3_w4moe_width_served_total {}\n", EXL3_W4MOE_WIDTH_SERVED.load(Relaxed)));
    o.push_str("# HELP velogb10_exl3_devroute_fallback_total #26 TP-B: prefill rounds that asked for the host-routed MoE arm under TP and were served by the device-routed EP arm instead (one clear line at boot; this counter moves when it ran).\n# TYPE velogb10_exl3_devroute_fallback_total counter\n");
    o.push_str(&format!("velogb10_exl3_devroute_fallback_total {}\n", EXL3_DEVROUTE_FALLBACK.load(Relaxed)));
    o.push_str("# HELP velogb10_exl3_pf_kernel_width_fallback_total #26 Fix 3: prefill layers whose width the pipelined expert kernel (xq_moe_pf*: K 2..7 and 1.5/2.5/3.5) cannot decode, served by the wide tiles kernel instead (one clear line per boot; 0 = never ran).\n# TYPE velogb10_exl3_pf_kernel_width_fallback_total counter\n");
    o.push_str(&format!("velogb10_exl3_pf_kernel_width_fallback_total {}\n", EXL3_PF_KERNEL_WIDTH_FALLBACK.load(Relaxed)));
    o.push_str("# HELP velogb10_exl3_kv_cache_fallback_total EXL3 boots that were asked for --kv-cache bf16 (the EXL3 engine has no bf16 KV store) and served f16 instead (the same 2 bytes per value; one clear line at boot). 0 = never happened.\n# TYPE velogb10_exl3_kv_cache_fallback_total counter\n");
    o.push_str(&format!("velogb10_exl3_kv_cache_fallback_total {}\n", EXL3_KV_CACHE_FALLBACK.load(Relaxed)));

    o.push_str("# HELP velogb10_exl3_mtp_skipped_total #26 Fix 4: boots whose draft head exceeded the served width set and were served WITHOUT speculation\n# TYPE velogb10_exl3_mtp_skipped_total counter\n");
    o.push_str(&format!("velogb10_exl3_mtp_skipped_total {}\n", EXL3_MTP_SKIPPED.load(Relaxed)));
    o.push_str("# HELP velogb10_exl3_ngram_embed_rows_total Token-embedding rows decoded from ngram trellis rings at load (exl3 1.6.3 packs; 0 = the plain embed path ran).\n# TYPE velogb10_exl3_ngram_embed_rows_total counter\n");
    o.push_str(&format!("velogb10_exl3_ngram_embed_rows_total {}\n", EXL3_NGRAM_EMBED_ROWS.load(Relaxed)));
    o.push_str("# HELP velogb10_exl3_frac_pf_bridge_rows_total Prefill rows served by the fractional-k2 decode-moe bridge (LEG-B; no pf kernel decodes k2=3/5/7).\n# TYPE velogb10_exl3_frac_pf_bridge_rows_total counter\n");
    o.push_str(&format!("velogb10_exl3_frac_pf_bridge_rows_total {}\n", EXL3_FRAC_PF_BRIDGE_ROWS.load(Relaxed)));
    o.push_str("# HELP velogb10_exl3_frac_expert_rounds_total K35: MoE layer rounds whose routed experts ran the fractional-width grouped body (K = 1.5/2.5/3.5, label k2 = 3/5/7; decode, verify, MTP draft and the prefill bridge; eager calls plus graph replays — a capture executes nothing and is not counted). Absent = that width never served.\n# TYPE velogb10_exl3_frac_expert_rounds_total counter\n");
    for k2 in 0..8usize {
        let v = EXL3_FRAC_EXPERT_ROUNDS[k2].load(Relaxed);
        if v > 0 { o.push_str(&format!("velogb10_exl3_frac_expert_rounds_total{{k2=\"{k2}\"}} {v}\n")); }
    }
    o.push_str("# HELP velogb10_exl3_k78_expert_rounds_total K8EXP: MoE layer rounds whose routed experts are K = 7 or K = 8 (label k2 = 14 / 16; decode, verify, MTP draft; eager calls plus graph replays). K = 7 runs the integer expert family (a1b7 / WP20 b7); K = 8 runs the plain grouped chain on its own entry (xq_gemm_grouped_b8). Absent = that width never served.\n# TYPE velogb10_exl3_k78_expert_rounds_total counter\n");
    for k2 in [14usize, 16] {
        let v = EXL3_FRAC_EXPERT_ROUNDS[k2].load(Relaxed);
        if v > 0 { o.push_str(&format!("velogb10_exl3_k78_expert_rounds_total{{k2=\"{k2}\"}} {v}\n")); }
    }
    o.push_str("# HELP velogb10_exl3_frac_pf_wide_rows_total K35: prefill expert rows (token x top-k) served by the fractional-width wide-tiles body (exl3_hmma_gemm_wide_tiles, EXL3_FRAC_BITS case), by k2. Absent = no fractional prefill ran there (the bridge counter above covers the host-routed arm).\n# TYPE velogb10_exl3_frac_pf_wide_rows_total counter\n");
    for k2 in 0..8usize {
        let v = EXL3_FRAC_PF_WIDE_ROWS[k2].load(Relaxed);
        if v > 0 { o.push_str(&format!("velogb10_exl3_frac_pf_wide_rows_total{{k2=\"{k2}\"}} {v}\n")); }
    }
    o.push_str("# HELP velogb10_exl3_frac_pf_pipelined_rows_total PFHALF: prefill expert rows (token x top-k) served by the pipelined kernel's fractional-width HALF body (xq_moe_pf, EXL3_FRAC_BITS case), by k2. Absent = no fractional layer prefilled on the pipelined kernel (it took the wide-tiles body above, or never ran).\n# TYPE velogb10_exl3_frac_pf_pipelined_rows_total counter\n");
    for k2 in 0..8 {
        let v = EXL3_FRAC_PF_PIPE_ROWS[k2].load(Relaxed);
        if v > 0 { o.push_str(&format!("velogb10_exl3_frac_pf_pipelined_rows_total{{k2=\"{k2}\"}} {v}\n")); }
    }
    o.push_str("# HELP velogb10_df2_mask_row_override 1 if the draft's mask row measured DAMAGED at load and the artifact's BF16 copy override is engaged (0 otherwise).\n# TYPE velogb10_df2_mask_row_override gauge\n");
    o.push_str(&format!("velogb10_df2_mask_row_override {}\n", DF2_MASK_OVERRIDE.load(Relaxed)));
    o.push_str("# HELP velogb10_draft_vocab_rows Token ids the MTP drafter may propose (--draft-vocab, FR-Spec); 0 = no reduced draft head, the drafter reads the full vocabulary.\n# TYPE velogb10_draft_vocab_rows gauge\n");
    o.push_str(&format!("velogb10_draft_vocab_rows {}\n", DRAFT_VOCAB_ROWS.load(Relaxed)));
    o.push_str("# HELP velogb10_draft_vocab_full_rows The model's full vocabulary (the denominator of velogb10_draft_vocab_rows; 0 when no reduced draft head was built).\n# TYPE velogb10_draft_vocab_full_rows gauge\n");
    o.push_str(&format!("velogb10_draft_vocab_full_rows {}\n", DRAFT_VOCAB_FULL.load(Relaxed)));
    o.push_str("# HELP velogb10_draft_vocab_steps_total MTP draft tokens picked from the reduced draft head (--draft-vocab engaged).\n# TYPE velogb10_draft_vocab_steps_total counter\n");
    o.push_str(&format!("velogb10_draft_vocab_steps_total {}\n", DRAFT_VOCAB_STEPS.load(Relaxed)));
    let inflight = IN_FLIGHT.load(Relaxed);
    let cap = MAX_BATCH.load(Relaxed);
    let running = if cap > 0 { inflight.min(cap) } else { inflight };
    o.push_str(&format!("# HELP velogb10_max_batch Lane capacity (--max-batch).\n# TYPE velogb10_max_batch gauge\nvelogb10_max_batch {cap}\n"));
    let req = MAX_BATCH_REQUESTED.load(Relaxed);
    o.push_str("# HELP velogb10_max_batch_requested The --max-batch the command line asked for, before the 16-lane serving cap (EXL3 and NVFP4 trunk servers; PREHARD1).\n# TYPE velogb10_max_batch_requested gauge\n");
    o.push_str(&format!("velogb10_max_batch_requested {req}\n"));
    let dsk = EXL3_DENSE_SPLITK_CALLS.load(Relaxed);
    o.push_str("# HELP velogb10_exl3_dense_splitk_calls_total DSPLITK: EXL3 trunk dense-attention calls served split-K over the context (--exl3-dense-splitk on; eager calls and graph captures, replays not counted). Absent = never.\n# TYPE velogb10_exl3_dense_splitk_calls_total counter\n");
    if dsk > 0 { o.push_str(&format!("velogb10_exl3_dense_splitk_calls_total {dsk}\n")); }
    let (ca, cs) = (EXL3_CTX_ASKED.load(Relaxed), EXL3_CTX_SERVED.load(Relaxed));
    o.push_str("# HELP velogb10_exl3_ctx_clamped 1 when this EXL3 TP=1 boot could not fit even one lane at the asked --max-seq-len and booted one lane at the longest context that fits (CTXCLAMP, owner 2026-11-07); labels: asked, served. Absent = no context clamp.\n# TYPE velogb10_exl3_ctx_clamped gauge\n");
    if ca > 0 { o.push_str(&format!("velogb10_exl3_ctx_clamped{{asked=\"{ca}\",served=\"{cs}\"}} 1\n")); }
    let pft = EXL3_PF_TC_ROWS.load(Relaxed);
    o.push_str("# HELP velogb10_exl3_pf_tc_rows_total PFTC: EXL3 prefill rows (per attention layer) whose dense attention ran on the tensor-core kernels (--exl3-dense-pf-tc on). Absent = never.\n# TYPE velogb10_exl3_pf_tc_rows_total counter\n");
    if pft > 0 { o.push_str(&format!("velogb10_exl3_pf_tc_rows_total {pft}\n")); }
    o.push_str(&format!("# HELP velogb10_requests_running Requests holding a lane.\n# TYPE velogb10_requests_running gauge\nvelogb10_requests_running {running}\n"));
    o.push_str(&format!("# HELP velogb10_requests_waiting Requests queued for a lane.\n# TYPE velogb10_requests_waiting gauge\nvelogb10_requests_waiting {}\n",
                        inflight - running));
    // MAXBATCH1: the F2 cap's engagement counter — arrivals that found every lane busy. Queued ran
    // when a lane freed; rejected hit the LR-4 waiting-limit 503. scripts/owner_check.py prints it.
    o.push_str("# HELP velogb10_over_cap_arrivals_total Requests that arrived beyond what the lane cap can run at once (a busy server, or a burst the queue must absorb): queued = waited for a lane, then served; rejected = waiting limit, 503 server_busy.\n# TYPE velogb10_over_cap_arrivals_total counter\n");
    o.push_str(&format!("velogb10_over_cap_arrivals_total{{disposition=\"queued\"}} {}\n", OVER_CAP_QUEUED.load(Relaxed)));
    o.push_str(&format!("velogb10_over_cap_arrivals_total{{disposition=\"rejected\"}} {}\n", OVER_CAP_REJECTED.load(Relaxed)));
    // TOPKM1: the batched GPU samplers' top_k cap (0 / negative / > 64 -> 64). scripts/owner_check.py prints it.
    o.push_str("# HELP velogb10_top_k_capped_total Sampled requests that asked for no top-k truncation (top_k 0 or negative) or more than 64 candidates on a model family whose GPU sampler keeps at most 64; served with top_k 64 (the EXL3 server and --cpu-sample apply no such limit).\n# TYPE velogb10_top_k_capped_total counter\n");
    o.push_str(&format!("velogb10_top_k_capped_total {}\n", TOP_K_CAPPED.load(Relaxed)));
    // THINKBUDGET1: the per-request thinking budget (vLLM's thinking_token_budget). scripts/owner_check.py prints it.
    o.push_str("# HELP velogb10_thinking_budget_requests_total Requests that sent thinking_token_budget: applied = the budget was enforced; unsupported = this model or serving path cannot enforce it, so the request ran without it.\n# TYPE velogb10_thinking_budget_requests_total counter\n");
    o.push_str(&format!("velogb10_thinking_budget_requests_total{{disposition=\"applied\"}} {}\n", THINK_BUDGET_APPLIED.load(Relaxed)));
    o.push_str(&format!("velogb10_thinking_budget_requests_total{{disposition=\"unsupported\"}} {}\n", THINK_BUDGET_UNSUPPORTED.load(Relaxed)));
    o.push_str("# HELP velogb10_thinking_budget_forced_total Reasoning blocks the thinking budget ended (the end-of-thinking marker was emitted by the server at the budget).\n# TYPE velogb10_thinking_budget_forced_total counter\n");
    o.push_str(&format!("velogb10_thinking_budget_forced_total {}\n", THINK_BUDGET_FORCED.load(Relaxed)));
    // PR #30 / #31 ports: log probabilities, json_schema enforcement, and the lanes they ran without speculation.
    o.push_str("# HELP velogb10_logprobs_requests_total Requests that asked for logprobs: served = returned; unsupported = this serving engine does not compute them; withheld = --scrub-tool-secrets is on; stream = stream=true, not wired yet. All but served get logprobs null.\n# TYPE velogb10_logprobs_requests_total counter\n");
    for (d, c) in [("served", &LOGPROBS_SERVED), ("unsupported", &LOGPROBS_UNSUPPORTED), ("withheld", &LOGPROBS_WITHHELD), ("stream", &LOGPROBS_STREAM)] {
        o.push_str(&format!("velogb10_logprobs_requests_total{{disposition=\"{d}\"}} {}\n", c.load(Relaxed)));
    }
    o.push_str("# HELP velogb10_logprobs_spec_rows_total Generated tokens whose log probabilities were read from a speculative round's verify rows (logprobs keep speculation).\n# TYPE velogb10_logprobs_spec_rows_total counter\n");
    o.push_str(&format!("velogb10_logprobs_spec_rows_total {}\n", LOGPROBS_SPEC_ROWS.load(Relaxed)));
    o.push_str("# HELP velogb10_json_schema_requests_total response_format json_schema requests: enforced = token mask armed (x-json-schema-enforced: full); unenforced = served unconstrained (x-json-schema-enforced: none).\n# TYPE velogb10_json_schema_requests_total counter\n");
    o.push_str(&format!("velogb10_json_schema_requests_total{{disposition=\"enforced\"}} {}\n", SCHEMA_ENFORCED.load(Relaxed)));
    o.push_str(&format!("velogb10_json_schema_requests_total{{disposition=\"unenforced\"}} {}\n", SCHEMA_UNENFORCED.load(Relaxed)));
    o.push_str("# HELP velogb10_spec_off_lanes_total EXL3 lanes that decoded without speculation because the request asked for logprobs or a schema mask.\n# TYPE velogb10_spec_off_lanes_total counter\n");
    o.push_str(&format!("velogb10_spec_off_lanes_total{{why=\"logprobs\"}} {}\n", SPEC_OFF_LOGPROBS.load(Relaxed)));
    o.push_str(&format!("velogb10_spec_off_lanes_total{{why=\"schema\"}} {}\n", SPEC_OFF_SCHEMA.load(Relaxed)));
    // CLIPRUNE1: rendered since it was set (F7b) but never shown — scripts/owner_check.py
    // already reads it and tolerates absence; keep that guard working.
    o.push_str("# HELP velogb10_unknown_options Unknown '--' options seen at startup (warned and ignored; F7b).\n# TYPE velogb10_unknown_options gauge\n");
    o.push_str(&format!("velogb10_unknown_options {}\n", UNKNOWN_OPTIONS.load(Relaxed)));
    o.push_str("# HELP velogb10_requests_total Finished requests by finish reason.\n# TYPE velogb10_requests_total counter\n");
    for (i, f) in FINISHES.iter().enumerate() {
        o.push_str(&format!("velogb10_requests_total{{finish=\"{f}\"}} {}\n", REQ_BY_FINISH[i].load(Relaxed)));
    }
    o.push_str("# HELP velogb10_scrub_hits_total Spans redacted by --scrub-tool-secrets since boot, by layer (advisory liveness; a span split by the pending cap can count twice). layer=provenance: matches of strings extracted from this conversation's tool results; layer=pattern: key-shaped text. Zero forever when the flag is off (the default).\n# TYPE velogb10_scrub_hits_total counter\n");
    o.push_str(&format!("velogb10_scrub_hits_total{{layer=\"provenance\"}} {}\nvelogb10_scrub_hits_total{{layer=\"pattern\"}} {}\n",
                       SCRUB_PROV_HITS.load(Relaxed), SCRUB_PAT_HITS.load(Relaxed)));
    o.push_str(&format!("# HELP velogb10_prompt_tokens_total Prompt tokens of finished requests.\n# TYPE velogb10_prompt_tokens_total counter\nvelogb10_prompt_tokens_total {}\n",
                        PROMPT_TOKENS.load(Relaxed)));
    o.push_str(&format!("# HELP velogb10_prompt_tokens_cached_total Prompt tokens served from the prefix cache (prefill skipped) of finished requests.\n# TYPE velogb10_prompt_tokens_cached_total counter\nvelogb10_prompt_tokens_cached_total {}\n",
                        CACHED_PROMPT.load(Relaxed)));
    o.push_str(&format!("# HELP velogb10_generation_tokens_total Generated tokens of finished requests.\n# TYPE velogb10_generation_tokens_total counter\nvelogb10_generation_tokens_total {}\n",
                        GEN_TOKENS.load(Relaxed)));
    TTFT.render(&mut o, "velogb10_time_to_first_token_seconds", "Request receipt to first generated token (queueing + prefill).");
    E2E.render(&mut o, "velogb10_request_duration_seconds", "Request receipt to the end of its response.");
    PREFILL.render(&mut o, "velogb10_prefill_tokens_per_second", "Prompt tokens / time to first token, per request (includes queueing).");
    DECODE.render(&mut o, "velogb10_decode_tokens_per_second", "Per-request generation rate after the first token.");
    // v0.7.3 long-run gauges (REL_LONGRUN §3) — /proc reads happen here, at scrape time.
    let rss_kb = proc_status_kb("VmRSS:");
    if let Some(kb) = rss_kb {
        o.push_str("# HELP velogb10_process_rss_bytes Resident set size of the server process.\n# TYPE velogb10_process_rss_bytes gauge\n");
        o.push_str(&format!("velogb10_process_rss_bytes {}\n", kb * 1024));
    }
    if let Some(t) = proc_status_kb("Threads:") {
        o.push_str("# HELP velogb10_thread_count OS threads of the server process (leak tripwire).\n# TYPE velogb10_thread_count gauge\n");
        o.push_str(&format!("velogb10_thread_count {t}\n"));
    }
    if let Some(f) = open_fds() {
        o.push_str("# HELP velogb10_open_fds Open file descriptors (leak tripwire).\n# TYPE velogb10_open_fds gauge\n");
        o.push_str(&format!("velogb10_open_fds {f}\n"));
    }
    if let Some(a) = crate::memwatch::mem_available_bytes() {
        o.push_str("# HELP velogb10_mem_available_bytes Box MemAvailable (the memwatch metric).\n# TYPE velogb10_mem_available_bytes gauge\n");
        o.push_str(&format!("velogb10_mem_available_bytes {a}\n"));
    }
    let min_av = crate::memwatch::min_available_gb();
    if min_av.is_finite() {
        o.push_str("# HELP velogb10_mem_min_available_bytes Minimum MemAvailable seen since boot.\n# TYPE velogb10_mem_min_available_bytes gauge\n");
        o.push_str(&format!("velogb10_mem_min_available_bytes {}\n", (min_av * 1e9) as u64));
    }
    o.push_str("# HELP velogb10_scheduler_steps_total Scheduler rounds STARTED since boot (a round in progress, or one that never completes, is counted; so is a pass whose lanes were all cancelled before it ran).\n# TYPE velogb10_scheduler_steps_total counter\n");
    o.push_str(&format!("velogb10_scheduler_steps_total {}\n", SCHED_STEPS.load(Relaxed)));
    o.push_str("# HELP velogb10_scheduler_busy 1 from the start of a decode round until the scheduler next blocks idle-waiting for a request, else 0. Only a decode round sets it: a server that was idle still reads 0 while it admits and prefills its first request.\n# TYPE velogb10_scheduler_busy gauge\n");
    o.push_str(&format!("velogb10_scheduler_busy {}\n", SCHED_BUSY.load(Relaxed)));
    if let Some(s0) = START.get() {
        let now_ms = s0.elapsed().as_millis() as u64;
        let last = SCHED_LAST_MS.load(Relaxed);
        let age = if last == 0 { now_ms } else { now_ms.saturating_sub(last) }; // H9: no wrap
        o.push_str("# HELP velogb10_scheduler_last_step_age_seconds Seconds since the scheduler last COMPLETED a round or a prefill chunk (uptime if none yet; grows while idle: alert on a high value only together with scheduler_busy == 1, or with requests_running > 0 for a stall during the first request's prefill).\n# TYPE velogb10_scheduler_last_step_age_seconds gauge\n");
        o.push_str(&format!("velogb10_scheduler_last_step_age_seconds {:.3}\n", age as f64 / 1000.0));
    }
    o.push_str("# HELP velogb10_graph_cache_entries CUDA graphs the engine holds now (captured at boot and, lazily, the first time a new step shape runs).\n# TYPE velogb10_graph_cache_entries gauge\n");
    o.push_str(&format!("velogb10_graph_cache_entries {}\n", GRAPH_ENTRIES.load(Relaxed)));
    o.push_str("# HELP velogb10_graph_cache_mixed_step_entries Mixed-regime decode-step CUDA graphs the engine holds now (per-row attention regime: one is captured the first time a step shape is seen; they are never dropped and not capped, so the value only grows; they are also counted in velogb10_graph_cache_entries).\n# TYPE velogb10_graph_cache_mixed_step_entries gauge\n");
    o.push_str(&format!("velogb10_graph_cache_mixed_step_entries {}\n", MIXED_STEP_GRAPHS.load(Relaxed)));
    o.push_str("# HELP velogb10_graph_capture_invalidated_total CUDA graph captures that ended CUDA_ERROR_STREAM_CAPTURE_INVALIDATED since boot: a CUDA call that is illegal inside stream capture landed in the capture window (from another thread, e.g. the vision tower, or from the capturing thread itself), so no graph was produced and the step ran eagerly. The lazily captured shapes (EXL3 step/verify/packed-verify/draft/reprime graphs, the NVFP4 verify graph, the DFlash2 inject graph) are captured again at their next use; the three boot-time graphs (NVFP4 decode set, NVFP4 decode+sample per batch size, the DFlash2 draft round) are not retried and stay eager for the boot (the [graph] log line says which). Expected 0.\n# TYPE velogb10_graph_capture_invalidated_total counter\n");
    o.push_str(&format!("velogb10_graph_capture_invalidated_total {}\n", CAPTURE_INVALIDATED.load(Relaxed)));
    o.push_str("# HELP velogb10_graph_capture_failed_total CUDA graph captures that produced no graph for a reason other than an invalidated capture, since boot: cuStreamBeginCapture refused, cuStreamEndCapture returned another status, a body error during capture, cuGraphInstantiate failed, or the TP graph-spine audit refused the graph (also counted by the idle line's spine tally). Same recovery per graph family as the invalidated counter (eager now; captured again at the next use for the lazy shapes, not retried for the three boot-time graphs). Expected 0.\n# TYPE velogb10_graph_capture_failed_total counter\n");
    o.push_str(&format!("velogb10_graph_capture_failed_total {}\n", CAPTURE_FAILED.load(Relaxed)));
    if let Some(p) = KV_POOL.get() {
        let bytes = (KV_BYTES_SET.load(Relaxed) != 0).then(|| [0, 1, 2].map(|k| KV_BYTES[k].load(Relaxed)));
        p.render(&mut o, bytes);
    }
    o.push_str("# HELP velogb10_lane_pack_rounds_total EXL3 scheduler rounds by arm since boot: packed = packed speculative verifies (ONE verify serving two requests; lane packing is on by default, --lane-pack off disables it), serial = lone speculative rounds (one request each; a request's control rounds and its plain steps at the end of its window or budget count here), shared_plain = shared plain steps of the load-adaptive policy (--spec-lanes-max). The plain batch of MTP-off requests is not counted.\n# TYPE velogb10_lane_pack_rounds_total counter\n");
    for (i, arm) in LANE_PACK_ARMS.iter().enumerate() {
        o.push_str(&format!("velogb10_lane_pack_rounds_total{{arm=\"{arm}\"}} {}\n", LANE_PACK_ROUNDS[i].load(Relaxed)));
    }
    o.push_str("# HELP velogb10_lane_pack_lane_rounds_total LANE rounds (one per request per packed verify: two per packed round) served by a packed speculative verify since boot, by the kind of request: sampled_real_q = a sampled request whose rows verify by the real-q ratio rule (--spec-sampling ratio, the default), sampled_match = a sampled request under --spec-sampling match, greedy = temperature 0. Always 0 with --lane-pack off.\n# TYPE velogb10_lane_pack_lane_rounds_total counter\n");
    for (i, kind) in LANE_PACK_KINDS.iter().enumerate() {
        o.push_str(&format!("velogb10_lane_pack_lane_rounds_total{{kind=\"{kind}\"}} {}\n", LANE_PACK_LANES[i].load(Relaxed)));
    }

    o.push_str("# HELP velogb10_mtp_prime_batch_total Shared plain-step primes served by the BATCHED single-launch path (xq_gather_tap_rows + one re-prime pass + xq_attn_kv_prefill_pairs) since boot; velogb10_mtp_prime_batch_lanes_total counts the head-row lanes those steps carried. 0 with --exl3-prime-batch off; steps that fall back to the per-lane loop are NOT counted.\n# TYPE velogb10_mtp_prime_batch_total counter\n");
    o.push_str(&format!("velogb10_mtp_prime_batch_total {}\n", PRIME_BATCH.load(Relaxed)));
    o.push_str("# TYPE velogb10_mtp_prime_batch_lanes_total counter\n");
    o.push_str(&format!("velogb10_mtp_prime_batch_lanes_total {}\n", PRIME_BATCH_LANES.load(Relaxed)));    o.push_str("# HELP velogb10_lane_pack_skip_total LANE rounds (one per request per scheduler step, not requests) that ran their own lone round outside a packed verify while lane packing was active and at least two requests were busy, by the reason that lane was not packed. NOT counted: steps the load-adaptive policy gave to the shared plain step (--spec-lanes-max; see velogb10_lane_pack_rounds_total{arm=\"shared_plain\"}), and a lone request. When a pair falls back to serial rounds for the control cadence BOTH of its lanes are counted as `control`, although only one of them owes the control round. Always 0 with --lane-pack off.\n# TYPE velogb10_lane_pack_skip_total counter\n");
    for (i, why) in crate::lane_pack::Skip::ALL.iter().enumerate() {
        o.push_str(&format!("velogb10_lane_pack_skip_total{{reason=\"{}\"}} {}\n", why.label(), LANE_PACK_SKIP[i].load(Relaxed)));
    }
    o.push_str("# HELP velogb10_lane_pack_disabled 1 once a packed round failed and the packed arm was turned off for the rest of this boot (the server then runs the serial arm; restart to re-enable). Always 0 with --lane-pack off and on a healthy boot.\n# TYPE velogb10_lane_pack_disabled gauge\n");
    o.push_str("# HELP velogb10_spec_lanes_est_tok_s The spec-lanes auto policy's estimate of the AGGREGATE tok/s the busy clients would see on each arm, at its last decision that compared real measurements (the busy lanes' speculation EMAs known): serial = the busy requests taking turns on their own speculative rounds (packed pairs where two lanes pack); shared = one plain step for all of them. The arm with the higher estimate runs, with 4% hysteresis. 0 means no such decision yet this boot (the policy rides its fallback threshold).\n# TYPE velogb10_spec_lanes_est_tok_s gauge\n");
    o.push_str(&format!("velogb10_spec_lanes_est_tok_s{{arm=\"serial\"}} {}\n", f64::from_bits(SPEC_LANES_EST_SERIAL.load(Relaxed))));
    o.push_str(&format!("velogb10_spec_lanes_est_tok_s{{arm=\"shared\"}} {}\n", f64::from_bits(SPEC_LANES_EST_SHARED.load(Relaxed))));
    o.push_str("# HELP velogb10_spec_lanes_shared_scale The learned correction the shared plain step's cost model carries for this boot's topology and content (1 = the fitted prior; learned from real shared steps; frozen at 1 until one runs).\n# TYPE velogb10_spec_lanes_shared_scale gauge\n");
    o.push_str(&format!("velogb10_spec_lanes_shared_scale {}\n", SPEC_LANES_SCALE_MILLI.load(Relaxed) as f64 / 1000.0));
    o.push_str("# HELP velogb10_spec_lanes_busy_lanes Busy lanes at the spec-lanes policy's last measured decision (the n of the shared-step cost model at that moment).\n# TYPE velogb10_spec_lanes_busy_lanes gauge\n");
    o.push_str(&format!("velogb10_spec_lanes_busy_lanes {}\n", SPEC_LANES_BUSY.load(Relaxed)));
    o.push_str("# HELP velogb10_spec_lanes_decisions_total Spec-lanes auto decisions that compared real measurements (both arms' estimates known); decisions taken on the fallback threshold (an estimate missing) are not counted.\n# TYPE velogb10_spec_lanes_decisions_total counter\n");
    o.push_str(&format!("velogb10_spec_lanes_decisions_total {}\n", SPEC_LANES_DECISIONS.load(Relaxed)));
    o.push_str("# HELP velogb10_spec_lanes_switches_total Times the spec-lanes auto pick FLIPPED between arms since boot (to = the arm it flipped to). A large count says thrash; zero with all rounds on one arm says the policy is stuck (see the est gauges for why).\n# TYPE velogb10_spec_lanes_switches_total counter\n");
    for (i, to) in SPEC_LANES_TO.iter().enumerate() {
        o.push_str(&format!("velogb10_spec_lanes_switches_total{{to=\"{to}\"}} {}\n", SPEC_LANES_SWITCHES[i].load(Relaxed)));
    }
    o.push_str("# HELP velogb10_spec_lanes_arm_ms_total Cumulative milliseconds the arm ran since boot (serial speculation rounds vs shared plain steps) and velogb10_spec_lanes_arm_tok_total the tokens it delivered: the exact time split behind the pick. kind=millisecond_count\n");
    for (i, arm) in SPEC_LANES_TO.iter().enumerate() {
        o.push_str(&format!("velogb10_spec_lanes_arm_ms_total{{arm=\"{arm}\"}} {}\n", SPEC_ARM_MS[i].load(Relaxed)));
        o.push_str(&format!("velogb10_spec_lanes_arm_tok_total{{arm=\"{arm}\"}} {}\n", SPEC_ARM_TOK[i].load(Relaxed)));
    }
    o.push_str("# HELP velogb10_spec_lanes_measured_tok_s The MEASURED delivered aggregate tok/s of each arm (EWMA over the windows its rounds last folded; 0 = never ran recently): shared = plain steps for all busy lanes, serial = speculative rounds taking turns (packed pairs where two lanes pack). The auto pick compares these when both are fresh at the current busy count; the est_tok_s gauges are its modelled fallback. kind=rate_tokens_per_second\n");
    o.push_str("# TYPE velogb10_spec_lanes_measured_tok_s gauge\n");
    for (i, arm) in SPEC_LANES_TO.iter().enumerate() {
        o.push_str(&format!("velogb10_spec_lanes_measured_tok_s{{arm=\"{arm}\"}} {}\n", f64::from_bits(SPEC_LANES_MEASURED[i].load(Relaxed))));
    }
    // POSTURE1: which meter feeds the measured rates (--spec-arm-meter): 1 = v1 (per round, counts within one), 2 = v2
    // (per step, exact busy count, never inside a prefill)
    o.push_str("# HELP velogb10_spec_lanes_meter Which meter feeds the measured arm rates (--spec-arm-meter): 1 = v1 (each arm's rounds, rates shared across busy counts within one of each other), 2 = v2 (whole steps, per exact busy count, not inside a prefill). kind=mode\n");
    o.push_str("# TYPE velogb10_spec_lanes_meter gauge\n");
    o.push_str(&format!("velogb10_spec_lanes_meter {}\n", if crate::opts::choice(crate::opt!("spec-arm-meter")) == "v2" { 2 } else { 1 }));
    o.push_str("# HELP velogb10_spec_lanes_meter_tp_sync_total --spec-arm-meter v2 under TP: metered steps whose step time every rank took from the head (one lockstep point per metered step, so the ranks fold identical rates and pick identical arms). 0 at TP=1. kind=step_count\n");
    o.push_str("# TYPE velogb10_spec_lanes_meter_tp_sync_total counter\n");
    o.push_str(&format!("velogb10_spec_lanes_meter_tp_sync_total {}\n", spec_meter_tp_syncs()));
    o.push_str("# HELP velogb10_spec_lanes_meter_replace_total --spec-arm-meter v2: probe windows whose measured rate contradicted the stored one by more than 15% and replaced it (instead of averaging). kind=fold_count\n");
    o.push_str("# TYPE velogb10_spec_lanes_meter_replace_total counter\n");
    o.push_str(&format!("velogb10_spec_lanes_meter_replace_total {}\n", spec_meter_replaces()));
    o.push_str("# HELP velogb10_spec_lanes_probes_total Arm-rate probe windows started: while one arm rides, a short window of the idle arm refreshes its measured rate so the pick stays grounded in delivery. kind=window_count\n");
    o.push_str("# TYPE velogb10_spec_lanes_probes_total counter\n");
    for (i, arm) in SPEC_LANES_TO.iter().enumerate() {
        o.push_str(&format!("velogb10_spec_lanes_probes_total{{arm=\"{arm}\"}} {}\n", SPEC_LANES_PROBES[i].load(Relaxed)));
    }
    o.push_str(&format!("velogb10_lane_pack_disabled {}\n", LANE_PACK_DISABLED.load(Relaxed)));
    // F1: the boot's --lane-pack decision — active, fell back with why, or off. The boot line prints
    // the same reason (`lane packing (--lane-pack, on by default) is NOT ACTIVE on this boot: ...; requests
    // are served without packed rounds`). A --lane-pack off boot records the flag as the reason.
    let (lpa, lpr) = match crate::lane_pack::boot_pack_word() {
        Some(w) if w.is_empty() => (1u64, String::new()),
        Some(w) => (0u64, w.clone()),
        None => (0u64, "--lane-pack off".to_string()),
    };
    o.push_str("# HELP velogb10_lane_pack_active 1 when lane packing (--lane-pack, on by default) is ACTIVE on this boot (packed rounds can run); 0 when this boot serves without packed rounds: reason = why (the boot line prints the same), or `--lane-pack off`.\n# TYPE velogb10_lane_pack_active gauge\n");
    o.push_str(&format!("velogb10_lane_pack_active{{reason=\"{}\"}} {}\n",
                        lpr.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n"), lpa));
    // F3: modules the wide trellis kernel cannot decode (widths outside 3..=7 since P7 B5) served through the
    // reconstruct + cuBLASLt arm whatever --exl3-recon-min said; 0 on a 3-, 4- or 5-bit pack.
    o.push_str("# HELP velogb10_exl3_recon_forced_total EXL3 prefill-chain GEMMs that a module the wide trellis kernel cannot decode (bit width outside 3..=7 since P7 B5) served through the reconstruct + cuBLASLt arm instead of the arm --exl3-recon-min asked for, by the module's bit width. The boot log prints the once-per-boot line the first time it happens.\n# TYPE velogb10_exl3_recon_forced_total counter\n");
    o.push_str("# HELP velogb10_exl3_psk_width_fallback_total EXL3 split-K chains that asked for the persistent PSK grid (--psk=1) but whose module bit width is outside the persistent bodies (2..=6): served by the plain split-K grid instead, by bit width. The boot log prints the once-per-boot line the first time it happens.\n# TYPE velogb10_exl3_psk_width_fallback_total counter\n");
    o.push_str("# HELP velogb10_exl3_wp21_width_fallback_total EXL3 split-K chains that were eligible for the WP21 fused kernel but whose module k2 is outside the fused bodies (3..=14, e.g. k2 16 = the 8-bit MTP-head class): served by the plain 3-launch tail instead, by k2. The boot log prints one WARN the first time it happens.\n# TYPE velogb10_exl3_wp21_width_fallback_total counter\n");
    for (b, c) in PSK_WIDTH_FALLBACK.iter().enumerate().skip(1) {
        let v = c.load(Relaxed);
        if v > 0 {
            o.push_str(&format!("velogb10_exl3_psk_width_fallback_total{{bits=\"{b}\"}} {v}\n"));
        }
    }
    for (b, c) in WP21_WIDTH_FALLBACK.iter().enumerate().skip(1) {
        let v = c.load(Relaxed);
        if v > 0 {
            o.push_str(&format!("velogb10_exl3_wp21_width_fallback_total{{k2=\"{b}\"}} {v}\n"));
        }
    }
    for b in 1..RECON_FORCED.len() {
        let v = RECON_FORCED[b].load(Relaxed);
        if v > 0 {
            o.push_str(&format!("velogb10_exl3_recon_forced_total{{bits=\"{b}\"}} {v}\n"));
        }
    }
    o.push_str("# HELP velogb10_fp4l_pack 1 when the served model is an FP4-Local (NVIDIA ModelOpt MXFP8 + NVFP4) pack.\n# TYPE velogb10_fp4l_pack gauge\n");
    o.push_str(&format!("velogb10_fp4l_pack {}\n", FP4L_PACK.load(Relaxed)));
    o.push_str("# HELP velogb10_fp4l_dispatch_total FP4-Local weight-kernel host dispatches by weight class (mxfp8 dense, f16 dense, bf16 = stored bf16 bits, nvfp4 experts, exl3_mtp_sidecar = the opt-in EXL3 trellis MTP-head linears of --fp4l-mtp-sidecar) and row class (decode = 1 row, verify = 2..16 rows incl. packed lanes, prefill = more). Graph replays do not dispatch from the host: in graphed decode these grow at capture.\n# TYPE velogb10_fp4l_dispatch_total counter\n");
    for (ki, kn) in ["mxfp8", "f16", "bf16", "nvfp4_experts", "exl3_mtp_sidecar"].iter().enumerate() {
        for (ri, rn) in ["decode", "verify", "prefill"].iter().enumerate() {
            let v = FP4L_DISPATCH[ki][ri].load(Relaxed);
            if v > 0 { o.push_str(&format!("velogb10_fp4l_dispatch_total{{weights=\"{kn}\",rows=\"{rn}\"}} {v}\n")); }
        }
    }
    if let Some(p) = FP4L_PREC.get() {
        o.push_str("# HELP velogb10_fp4l_precision The FP4-Local precision levers this boot resolved (--fp4l-plain, --fp4l-norms, --fp4l-lm-head, --fp4l-draft-head, --fp4l-prefill-dense); the value is always 1.\n# TYPE velogb10_fp4l_precision gauge\n");
        o.push_str(&format!("velogb10_fp4l_precision{{{p}}} 1\n"));
    }
    o.push_str("# HELP velogb10_fp4l_twin_total Host dispatches of the FP4-Local exact-precision twins of trunk kernels by op (stored bf16 mixers / router / embedding / shared gate / n-gram weights, fp32 norms). 0 under --fp4l-plain f16 --fp4l-norms f16. Graph replays do not dispatch from the host: in graphed decode these grow at capture.\n# TYPE velogb10_fp4l_twin_total counter\n");
    for (i, op) in FP4L_TWIN_OPS.iter().enumerate() {
        let v = FP4L_TWIN[i].load(Relaxed);
        if v > 0 { o.push_str(&format!("velogb10_fp4l_twin_total{{op=\"{op}\"}} {v}\n")); }
    }
    o.push_str("# HELP velogb10_log_lines_dropped_total Log lines dropped because whatever reads the server's output stalled (the log queue was full).\n# TYPE velogb10_log_lines_dropped_total counter\n");
    o.push_str(&format!("velogb10_log_lines_dropped_total {}\n", crate::logq::dropped()));
    o.push_str("# HELP velogb10_requests_rejected_total Requests refused at admission (--max-waiting exceeded).\n# TYPE velogb10_requests_rejected_total counter\n");
    o.push_str(&format!("velogb10_requests_rejected_total {}\n", REQ_REJECTED.load(Relaxed)));
    o.push_str("# HELP velogb10_streams_cancelled_backlog_total Streams cancelled for exceeding --stream-backlog-events (the client stopped reading).\n# TYPE velogb10_streams_cancelled_backlog_total counter\n");
    o.push_str(&format!("velogb10_streams_cancelled_backlog_total {}\n", STREAMS_CANCELLED_BACKLOG.load(Relaxed)));
    o.push_str("# HELP velogb10_lane_reserve_admits_total Admissions decided by the --lane-reserve policy (EXL3; head-authoritative at TP>=2 since LANERES3: the head ships each admit's scoping, every rank picks the same lane), by rule: own = the waiting request's prompt-end snapshot sat in a free lane (admitted first, into that lane); unreserved = a free lane no reservation protects; forced = the queue head out-waited --lane-reserve-patience and took a reserved lane (rate-limited by --lane-reserve-swap). All zero when --lane-reserve is off (default).\n# TYPE velogb10_lane_reserve_admits_total counter\n");
    o.push_str(&format!("velogb10_lane_reserve_admits_total{{rule=\"own\"}} {}\n", LANE_RESERVE_ADMITS[0].load(Relaxed)));
    o.push_str(&format!("velogb10_lane_reserve_admits_total{{rule=\"unreserved\"}} {}\n", LANE_RESERVE_ADMITS[1].load(Relaxed)));
    o.push_str(&format!("velogb10_lane_reserve_admits_total{{rule=\"forced\"}} {}\n", LANE_RESERVE_ADMITS[2].load(Relaxed)));
    o.push_str(&format!("velogb10_lane_reserve_tp_requeued_total {}\n", LANE_RESERVE_TP_REQUEUED.load(Relaxed)));
    o.push_str("# HELP velogb10_lane_reserve_waiting Requests waiting while the --lane-reserve policy holds every free lane reserved for their conversations (0 when the policy is off or nothing waits).\n# TYPE velogb10_lane_reserve_waiting gauge\n");
    o.push_str(&format!("velogb10_lane_reserve_waiting {}\n", LANE_RESERVE_WAITING.load(Relaxed)));
    {
        let (pc, sc) = (NAIVE_TP_PRIMARY_CTX.load(Relaxed), NAIVE_TP_SINGLE_CTX.load(Relaxed));
        if pc != 0 {
            o.push_str("# HELP velogb10_naive_tp_reduce Naive TP all-reduce paths (device epoch counters): single_stage = 1 when --tp-reduce resolved to the one-shot ctx (world 4); single_epochs_total = reduces that took it (one epoch each); tree_epochs_total = exchanges on the primary link (log2(world) per tree reduce: prefill chunks, and every reduce when the single stage is off)\n");
            o.push_str(&format!("velogb10_naive_tp_reduce_single_stage {}\n", (sc != 0) as u8));
            o.push_str(&format!("velogb10_naive_tp_reduce_single_gpu_recv {}\n", (sc != 0 && NAIVE_TP_SINGLE_GPU_RECV.load(Relaxed)) as u8));
            o.push_str(&format!("velogb10_naive_tp_reduce_single_k1_multiblock {}\n", (sc != 0 && NAIVE_TP_SINGLE_K1_MB.load(Relaxed)) as u8));
            o.push_str(&format!("velogb10_naive_tp_reduce_single_epochs_total {}\n", if sc != 0 { crate::net::ctx_device_epoch(sc) } else { 0 }));
            o.push_str(&format!("velogb10_naive_tp_reduce_tree_epochs_total {}\n", crate::net::ctx_device_epoch(pc)));
        }
        // every velogb10_naive_* series renders ONLY on a Naive server (lanes > 0): owner_check keys its NAIVE-N0.5
        // section on their presence, and any other model's server must export none of them
        let nl = NAIVE_LANES.load(Relaxed);
        if nl > 0 {
            o.push_str("# HELP velogb10_naive_config Naive serve configuration of this boot: moe_shard_tp = 1 when every rank holds a slice of every expert (--naive-moe-shard tp), lanes = concurrent requests (--naive-lanes)\n");
            o.push_str(&format!("velogb10_naive_moe_shard_tp {}\n", NAIVE_MOE_TP.load(Relaxed) as u8));
            o.push_str(&format!("velogb10_naive_lanes {nl}\n"));
            o.push_str(&format!("velogb10_naive_lane_ctx {}\n", NAIVE_LANE_CTX.load(Relaxed) as u8));
        o.push_str("# HELP velogb10_naive_spec_sampled Naive exact speculative sampling (sampled requests): rounds, accepted drafts, residual fallbacks by cause\n");
        o.push_str(&format!("velogb10_naive_spec_sampled_rounds_total {}\n", NAIVE_SPEC_SAMPLED_ROUNDS.load(Relaxed)));
        o.push_str(&format!("velogb10_naive_spec_sampled_accepted_total {}\n", NAIVE_SPEC_SAMPLED_ACCEPTED.load(Relaxed)));
        o.push_str(&format!("velogb10_naive_spec_fallback_nolist_total {}\n", NAIVE_SPEC_FALLBACK_NOLIST.load(Relaxed)));
        o.push_str(&format!("velogb10_naive_spec_fallback_empty_residual_total {}\n", NAIVE_SPEC_FALLBACK_EMPTY.load(Relaxed)));
            o.push_str("# HELP velogb10_naive_prefix Naive prefix reuse (--prefix-cache on): state 1 = on (one conversation per KV slot), 0 = not asked (2 = asked but off: builds before 2026-10-21 with --naive-lanes > 1); hits = prompts that resumed after the previous prompt's prefix, misses = prompts prefilled whole, reused_tokens = prompt tokens not prefilled again\n");
            o.push_str(&format!("velogb10_naive_prefix_state {}\n", NAIVE_PREFIX_ON.load(Relaxed)));
            o.push_str(&format!("velogb10_naive_prefix_hits_total {}\n", NAIVE_PREFIX_HITS.load(Relaxed)));
            o.push_str(&format!("velogb10_naive_prefix_misses_total {}\n", NAIVE_PREFIX_MISSES.load(Relaxed)));
            o.push_str(&format!("velogb10_naive_prefix_reused_tokens_total {}\n", NAIVE_PREFIX_REUSED.load(Relaxed)));
        }
    }
    o.push_str("# HELP velogb10_prefix_pool Prefix pool: prompt states kept in memory across lanes (--prefix-pool-gb). cap/bytes/entries = now; captures = lane states copied in (capture_us_total = their in-round copy time); skips = already held or too short; refused = over the cap, under the memory floor, or a failed copy (any rank); hits = prompts resumed from an entry (hit_tokens = prompt tokens not read again; restore_us_total = the copy time); misses = no lane snapshot and no entry deeper than the lane's own checkpoint\n");
    o.push_str(&format!("velogb10_prefix_pool_cap_bytes {}\n", PFX_POOL_CAP.load(Relaxed)));
    o.push_str(&format!("velogb10_prefix_pool_bytes {}\n", PFX_POOL_BYTES.load(Relaxed)));
    o.push_str(&format!("velogb10_prefix_pool_entries {}\n", PFX_POOL_ENTRIES.load(Relaxed)));
    o.push_str(&format!("velogb10_prefix_pool_captures_total {}\n", PFX_POOL_CAPTURES.load(Relaxed)));
    o.push_str(&format!("velogb10_prefix_pool_capture_us_total {}\n", PFX_POOL_CAPTURE_US.load(Relaxed)));
    o.push_str(&format!("velogb10_prefix_pool_skips_total {}\n", PFX_POOL_SKIPS.load(Relaxed)));
    o.push_str(&format!("velogb10_prefix_pool_refused_total {}\n", PFX_POOL_REFUSED.load(Relaxed)));
    o.push_str(&format!("velogb10_prefix_pool_evictions_total {}\n", PFX_POOL_EVICTIONS.load(Relaxed)));
    o.push_str(&format!("velogb10_prefix_pool_emergency_frees_total {}\n", PFX_POOL_EMERGENCY.load(Relaxed)));
    o.push_str(&format!("velogb10_prefix_pool_hits_total {}\n", PFX_POOL_HITS.load(Relaxed)));
    o.push_str(&format!("velogb10_prefix_pool_hit_tokens_total {}\n", PFX_POOL_HIT_TOKENS.load(Relaxed)));
    o.push_str(&format!("velogb10_prefix_pool_restore_us_total {}\n", PFX_POOL_RESTORE_US.load(Relaxed)));
    o.push_str(&format!("velogb10_prefix_pool_misses_total {}\n", PFX_POOL_MISSES.load(Relaxed)));
    crate::turn_resume::render_metrics(&mut o); // TURNRESUME: velogb10_turn_*
    crate::dsv41::engram::io_metrics::render(&mut o); // DSV41 EIO: velogb10_dsv41_engram_* (nothing unless an Engram runtime opened)
    crate::dsv41::fwd::receipts::render(&mut o); // DSV41 SL: velogb10_dsv41_fwd_calls_total (nothing unless a dsv41 forward ran)
    crate::dsv41::gpu::dspark::metrics::render(&mut o); // DSV41 DS: velogb10_dsv41_spec_* (nothing unless a DSpark round ran or a source was announced)
    o.push_str("# HELP velogb10_kv_spill SSD spill tier for evicted lane snapshots (--kv-spill)\n");
    o.push_str(&format!("velogb10_kv_spill_writes_total {}\n", KV_SPILL_WRITES.load(Relaxed)));
    o.push_str("# HELP velogb10_prefill_pack_passes_total Prefill passes run by the packed admission (--prefill-pack); packed = passes carrying 2+ prompts\n");
    o.push_str(&format!("velogb10_prefill_pack_passes_total {}\n", PREFILL_PACK_PASSES.load(Relaxed)));
    o.push_str(&format!("velogb10_prefill_pack_packed_passes_total {}\n", PREFILL_PACK_PACKED.load(Relaxed)));
    o.push_str(&format!("velogb10_prefill_pack_segments_total {}\n", PREFILL_PACK_SEGS.load(Relaxed)));
    o.push_str(&format!("velogb10_prefill_pack_rows_total {}\n", PREFILL_PACK_ROWS.load(Relaxed)));
    o.push_str(&format!("velogb10_prefill_pack_prompts_total {}\n", PREFILL_PACK_PROMPTS.load(Relaxed)));
    o.push_str(&format!("velogb10_prefill_pack_holds_total {}\n", PREFILL_PACK_HOLDS.load(Relaxed)));
    for (i, r) in PREFILL_PACK_SKIP_REASONS.iter().enumerate() {
        o.push_str(&format!("velogb10_prefill_pack_skipped_total{{reason=\"{r}\"}} {}\n", PREFILL_PACK_SKIPS[i].load(Relaxed)));
    }
    o.push_str("# HELP velogb10_prefill_interleave_steps_total Decode steps run between prefill chunks of joining prompts (--prefill-interleave)\n");
    o.push_str(&format!("velogb10_prefill_interleave_steps_total {}\n", PREFILL_INTERLEAVE_STEPS.load(Relaxed)));
    o.push_str(&format!("velogb10_prefill_interleave_points_total {}\n", PREFILL_INTERLEAVE_POINTS.load(Relaxed)));
    o.push_str(&format!("velogb10_kv_spill_hits_total {}\n", KV_SPILL_HITS.load(Relaxed)));
    o.push_str(&format!("velogb10_kv_spill_misses_recompute_total {}\n", KV_SPILL_MISSES.load(Relaxed)));
    o.push_str(&format!("velogb10_kv_spill_durability_waits_total {}\n", KV_SPILL_DURABILITY_WAITS.load(Relaxed)));
    o.push_str(&format!("velogb10_kv_spill_durability_joins_total {}\n", KV_SPILL_DURABILITY_JOINS.load(Relaxed)));
    o.push_str(&format!("velogb10_kv_spill_durability_wait_us_total {}\n", KV_SPILL_DURABILITY_WAIT_US.load(Relaxed)));
    o.push_str(&format!("velogb10_kv_spill_truncated_restores_total {}\n", KV_SPILL_TRUNCATED_RESTORES.load(Relaxed)));
    o.push_str(&format!("velogb10_kv_spill_capture_peer_drop_total {}\n", KV_SPILL_CAPTURE_PEER_DROPS.load(Relaxed)));
    o.push_str(&format!("velogb10_kv_spill_restore_peer_miss_total {}\n", KV_SPILL_RESTORE_PEER_MISSES.load(Relaxed)));
    o.push_str(&format!("velogb10_kv_spill_bytes {}\n", KV_SPILL_BYTES.load(Relaxed)));
    o.push_str(&format!("velogb10_kv_spill_tokens_recompute_saved_total {}\n", KV_SPILL_SAVED.load(Relaxed)));
    {
        // KVPOOL (veloGB10 #43): the shared KV page pool behind the EXL3 position planes (--kv-pool).
        use std::sync::atomic::Ordering::Relaxed as R;
        o.push_str("# HELP velogb10_kv_vmm_mode what backs the EXL3 position planes: 0 = the standard per-lane allocation, 1 = --kv-pool full, 2 = --kv-pool lazy\n");
        o.push_str(&format!("velogb10_kv_vmm_mode {}\n", crate::kv_vmm::engaged()));
        o.push_str("# HELP velogb10_kv_vmm_fallback an asked-for --kv-pool that does not run here: 1 with the reason (tp = a TP rank, model = the model is not served by pool-backed planes)\n");
        for (i, why) in [(1u64, "tp"), (2, "model")] {
            o.push_str(&format!("velogb10_kv_vmm_fallback{{reason=\"{why}\"}} {}\n", (crate::kv_vmm::fallback_reason() == i) as u8));
        }
        o.push_str("# HELP velogb10_kv_vmm_bytes the shared KV pool: pool = physical pages, mapped = pages under lane rows now, reserved = address space\n");
        o.push_str(&format!("velogb10_kv_vmm_pool_bytes {}\n", crate::kv_vmm::POOL_BYTES.load(R)));
        o.push_str(&format!("velogb10_kv_vmm_mapped_bytes {}\n", crate::kv_vmm::MAPPED_BYTES.load(R)));
        o.push_str(&format!("velogb10_kv_vmm_reserved_bytes {}\n", crate::kv_vmm::RESERVED_BYTES.load(R)));
        o.push_str("# HELP velogb10_kv_vmm_events_total pool full: idle lanes evicted, requests ended finish=length (the pool could not grow them), prompts that waited for room\n");
        o.push_str(&format!("velogb10_kv_vmm_evictions_total {}\n", crate::kv_vmm::EVICTIONS.load(R)));
        o.push_str(&format!("velogb10_kv_vmm_length_ends_total {}\n", crate::kv_vmm::LENGTH_ENDS.load(R)));
        o.push_str(&format!("velogb10_kv_vmm_waits_total {}\n", crate::kv_vmm::WAITS.load(R)));
    }
    o.push_str("# HELP velogb10_vision_cache vision-tower embed cache (--vision-embed-cache-mb): bytes = merged embeddings held now; hits = images served from the cache (the tower never ran for them); misses = images that ran the tower and were inserted; evictions = images shifted out by the LRU cap\n");
    o.push_str(&format!("velogb10_vision_cache_bytes {}\n", VISION_CACHE_BYTES.load(Relaxed)));
    o.push_str(&format!("velogb10_vision_cache_hits_total {}\n", VISION_CACHE_HITS.load(Relaxed)));
    o.push_str(&format!("velogb10_vision_cache_misses_total {}\n", VISION_CACHE_MISSES.load(Relaxed)));
    o.push_str(&format!("velogb10_vision_cache_evictions_total {}\n", VISION_CACHE_EVICTIONS.load(Relaxed)));
    let (steps, drafts, accepted, emitted) = crate::tel::spec_totals();
    o.push_str("# HELP velogb10_spec_draft_tokens_total Speculative draft tokens proposed (MTP / DFlash2 / DFlash).\n# TYPE velogb10_spec_draft_tokens_total counter\n");
    o.push_str(&format!("velogb10_spec_draft_tokens_total {drafts}\n"));
    o.push_str("# HELP velogb10_spec_accepted_tokens_total Speculative draft tokens accepted by the verify.\n# TYPE velogb10_spec_accepted_tokens_total counter\n");
    o.push_str(&format!("velogb10_spec_accepted_tokens_total {accepted}\n"));
    o.push_str("# HELP velogb10_spec_rounds_total Speculative rounds (verify forwards).\n# TYPE velogb10_spec_rounds_total counter\n");
    o.push_str(&format!("velogb10_spec_rounds_total {steps}\n"));
    o.push_str("# HELP velogb10_spec_emitted_tokens_total Tokens emitted by speculative rounds (accepted + 1 per round).\n# TYPE velogb10_spec_emitted_tokens_total counter\n");
    o.push_str(&format!("velogb10_spec_emitted_tokens_total {emitted}\n"));
    let (mtp_off, reprobes, reprobe_ok) = crate::tel::mtp_reprobe_totals();
    o.push_str("# HELP velogb10_spec_mtp_off_events_total MTP off EVENTS, not requests (one request can go off several times; re-offs after failed probes count; flushed at request end with [mtp-stats]).\n# TYPE velogb10_spec_mtp_off_events_total counter\n");
    o.push_str(&format!("velogb10_spec_mtp_off_events_total {mtp_off}\n"));
    o.push_str("# HELP velogb10_spec_mtp_reprobes_total Re-probe windows the engine opened after an MTP off (speculation re-armed to test the workload again).\n# TYPE velogb10_spec_mtp_reprobes_total counter\n");
    o.push_str(&format!("velogb10_spec_mtp_reprobes_total {reprobes}\n"));
    o.push_str("# HELP velogb10_spec_mtp_reprobe_ok_total Re-probe windows that ended with speculation back on.\n# TYPE velogb10_spec_mtp_reprobe_ok_total counter\n");
    o.push_str(&format!("velogb10_spec_mtp_reprobe_ok_total {reprobe_ok}\n"));
    crate::exl3_serve::df2_metrics_append(&mut o);
    crate::exl3_serve::copy_span_metrics_append(&mut o); // WIDEVERIFY1: velogb10_copy_span_*
    render_toolcall(&mut o);
    crate::dsv41::load::metrics_rows::append(&mut o); // DSV4.1 loader rows (nothing unless a dsv41 load ran)
    crate::dsv41::tp::metrics::append(&mut o); // DSV4.1 TP comm rows (nothing unless a dsv41 TP comm attached)
    o
}
