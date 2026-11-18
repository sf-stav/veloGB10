//! Test-only source-text helper: `src_text!("<name>")` expands to the text of the source file(s)
//! of module `<name>` (path relative to `src/`, no `.rs`).
//!
//! About 75 source-text tests assert on function bodies by reading whole source files. When a file
//! is split into a module directory, those bodies move; each arm below is then changed from the
//! single `include_str!` to `concat!(include_str!(..), include_str!(..), ...)` over ALL files of
//! that module in a fixed order, so the assertions keep working unchanged. Paths are absolute
//! (`CARGO_MANIFEST_DIR`), so the macro works from any file depth and from the bin crate too.
//! Declared first in `lib.rs` (and via `#[path]` in `main.rs`) so its textual scope covers all modules.

macro_rules! src_text {
    ("batch") => { concat!(
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/batch/mod.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/batch/admission.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/batch/copy_span.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/batch/decode.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/batch/df2.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/batch/df2_carry.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/batch/dspark.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/batch/helpers.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/batch/lifecycle.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/batch/loopfix_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/batch/mtp_forest.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/batch/mtp_lane.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/batch/mtp_policy.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/batch/mtp_policy_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/batch/prehard1_serve_cap_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/batch/tree_accept_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/batch/vision_image_cache_tests.rs"))
    ) };
    // Needles are first occurrences of `pub fn run_head_retain(` / `fn head_sync_one(` etc.: every code file must precede the test files
    // (whose text spells those needles), and head.rs keeps head_sync_one .. head_ship .. run_head_retain .. sync_only_exit in the old order.
    ("cluster") => { concat!(
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cluster/discovery.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cluster/head.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cluster/mod.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cluster/models.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cluster/node.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cluster/build_id_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cluster/ipv4_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cluster/pack_manifest_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cluster/tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cluster/tpship_tests.rs"))
    ) };
    ("exl3_autotune") => { concat!(
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_autotune/mod.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_autotune/cli_main.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_autotune/digest.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_autotune/passes.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_autotune/session.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_autotune/tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_autotune/tune_cli.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_autotune/tuner_defs.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_autotune/tuner_gate.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_autotune/tuner_plan.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_autotune/tuner_race.rs"))
    ) };
    // Two source-text tests cut the text at the first `mod wide_bits_guard_tests` / `mod dsplitk_tests` (all code before the
    // test module), so every code file must precede mod.rs (which holds those declarations); the test files come last.
    ("exl3_forward") => { concat!(
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/attn.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/attn_gates.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/bench.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/chain.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/consts.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/cublaslt.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/df2.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/draft.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/forward_step.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/graph_keys.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/hc.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/ladder.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/lmh.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/load.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/moe.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/moe_wp20.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/mpk.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/mtp_tap.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/pack.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/ple.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/prefill.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/prefill_attn.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/prefill_hc.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/prefill_moe.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/prefill_warmup.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/probes_lt.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/probes_pack23.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/probes_packstate.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/probes_prefillx.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/psk.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/reprime.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/rounds.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/router_r16.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/sampling.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/scratch.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/slot_state.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/spec_pass.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/tensors.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/tune_hc.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/verify.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/vision.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/widths.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/wp21.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/wp_levers.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/mod.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/dsplitk_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/flash256_stage_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/l10_prefillx_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/pfx1_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/ple_headroom_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/ple_k_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/ple_ngram_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/qsaregime_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/rel_fixb_router_selection.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/rel_gates2_reprime_words_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/reprime_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/rope_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/served_window_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/spill_region_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/tp4g_dv3_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/vis2_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/wide_bits_guard_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/wp05_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/wp18_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/wp20_b5_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/wp24_key_tests.rs"))
    ) };
    ("exl3_forward/fp4l_fwd") => { concat!(
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/fp4l_fwd/mod.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/fp4l_fwd/chains.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/fp4l_fwd/draft_head.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/fp4l_fwd/launch.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/fp4l_fwd/moe.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/fp4l_fwd/probe_gpu.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/fp4l_fwd/tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/fp4l_fwd/weights.rs"))
    ) };
    // One source-text test cuts the text at the first `mod agree16b_tests` (everything before it is the old "code + geometry_tests"), so
    // every code file and geometry_tests precede mod.rs (whose declaration of that module is its last item) and agree16b_tests comes last.
    ("exl3_forward/xtp") => { concat!(
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/xtp/chain_ep.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/xtp/ep.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/xtp/gates.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/xtp/slice_probe.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/xtp/spine.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/xtp/tp_config.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/xtp/tp_state.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/xtp/xtp_ref.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/xtp/geometry_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/xtp/mod.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_forward/xtp/agree16b_tests.rs"))
    ) };
    ("exl3_kvnll") => { include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_kvnll.rs")) };
    ("exl3_ppl") => { concat!(
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_ppl/mod.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_ppl/tests.rs"))
    ) };
    ("exl3_regime") => { include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_regime.rs")) };
    // exl3_serve: code files, then mod.rs, then the test-module files: three tests cut the text at `mod <name>_tests` (the first hit must be the
    // declaration in mod.rs, AFTER all code) and the pins find code before the test text that repeats their needles.
    ("exl3_serve") => { concat!(
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/admission.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/builder.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/copy.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/dds.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/df2.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/fit.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/lanes.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/lockstep.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/mtp_round.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/pack.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/pack_prefill.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/policy.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/pool.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/prefill.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/scheduler.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/serve.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/spec_program.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/step.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/tp_serve.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/wp24_gate.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/mod.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/agree16b_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/copy_span_arm_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/ctx_clamp_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/dds_prior_w4_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/df2_gate_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/fit_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/lane_pack_penalty_warning_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/lane_pack_sampled_tp_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/lane_pack_wiring_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/mtp_switch_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/prefill_interleave_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/regime_sched_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/review_p0_pins.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/s1_eq_aware_arg_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/spec_lanes_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/spill_agree_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/step_image_payload_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/stop_info_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/token_logprobs_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/tp_agree_width_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/tp_hub_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/tp_preflight_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/tpship_kv_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/vision_drill_flag_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/exl3_serve/wp23_tests.rs")),
    ) };
    ("gpu") => { concat!(
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/mod.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/attn_batch.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/bandwidth.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/batch_types.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/bench_mtp.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/bench_tree.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/construct.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/debug.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/decode.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/df2_capture.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/dsv4_moe.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/e9.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/eff.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/embed.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/forward_batch.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/forward_token.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/gates.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/gemm_policy.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/gemm_prefill.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/generate.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/geom_pack_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/gptq_gpu.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/gptq_rot.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/graphs.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/launch_args.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/launch_gemm.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/load.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/moe_batch.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/mtp.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/pool.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/prefill.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/probes_gemm.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/probes_mxfp4.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/probes_verify.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/qsa.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/qwen4_exp.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/release_workers_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/shard_factor_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/shard_ops.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/shard_raw_fp8_blk_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/shard_raw_real_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/shard_raw_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/state.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/tp_comm.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/tp_shard.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/verify.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/verify_sample.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/w4a4.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/xchain.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/xchain2_tests.rs"))
    ) };
    ("lane_pack") => { concat!(
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/lane_pack/mod.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/lane_pack/tests.rs"))
    ) };
    ("lib") => { include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/lib.rs")) };
    ("main") => { concat!(
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/main.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cli/mod.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cli/args.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cli/bench_basic.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cli/bench_df2.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cli/bench_mtp.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cli/dsv4_serve.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cli/dsv4_serve/dspark_driver.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cli/help.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cli/probe_df2_draft.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cli/probe_df2_misc.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cli/probe_df2_round.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cli/probe_dspark.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cli/probe_dspark_round.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cli/probe_tq.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cli/probes_dsv4.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cli/probes_serve.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cli/probes_state.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cli/quantize.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cli/replay.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cli/serve.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cli/stq.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cli/tp_cluster.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cli/tp_heads.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cli/tp_serve_trunk.rs")),
        // the test module last: its own text names `fn run_cli`, which must not be the first match
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cli/prehard1_cli_model_dir_tests.rs"))
    ) };
    // The tests cut the text at the first `mod tests`: every code file precedes mod.rs (which declares it); tests.rs last.
    ("metrics") => { concat!(
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/metrics/capture.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/metrics/exl3.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/metrics/features.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/metrics/fp4l.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/metrics/lane_pack.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/metrics/render.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/metrics/req.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/metrics/sched.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/metrics/toolcall.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/metrics/mod.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/metrics/tests.rs"))
    ) };
    ("opts_table") => { include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/opts_table.rs")) };
    // The benchfix tests cut the text at the first `mod benchfix_sse_tests`, so every code file must
    // precede mod.rs (which holds that declaration); the test files come last.
    ("server") => { concat!(
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/server/admin.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/server/chat.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/server/chat_params.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/server/chat_stream.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/server/completions.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/server/engine.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/server/helpers.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/server/logprobs.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/server/reasoning.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/server/sse.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/server/tokenize.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/server/vision.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/server/mod.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/server/benchfix_sse_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/server/completions_usage_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/server/context_budget_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/server/gen_line_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/server/grow_find_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/server/harden_lr5_toktx_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/server/harden_review_h1_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/server/harden_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/server/host_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/server/logprobs_param_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/server/logprobs_span_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/server/loopfix_card_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/server/model_id_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/server/nonetools_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/server/prompt_token_id_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/server/reasoning_start_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/server/s10_models_route_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/server/s8_sse_escape_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/server/thinking_budget_parse_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/server/top_k_parse_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/server/vision_fallback_tests.rs")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/server/wp08_validation_tests.rs")),
    ) };
    ("vision_gpu") => { include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/vision_gpu.rs")) };
    // SPLIT1 Wave 3: kernels/exl3_bench.cu is include-split; the text is the .cu then every section in include order.
    ("kernels/exl3_bench") => { concat!(
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/exl3_bench.cu")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/exl3_bench/01_tune_lmh_gemm.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/exl3_bench/02_wide_grouped_forward.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/exl3_bench/03_gdn_router_dhead.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/exl3_bench/04_sampler_penalties.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/exl3_bench/05_draft_head_pen.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/exl3_bench/06_router_fold_r16.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/exl3_bench/07_moe_wp20_epilogues.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/exl3_bench/08_moe_persistent.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/exl3_bench/09_moe_pk_widths_ple.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/exl3_bench/10_prefill_chunk.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/exl3_bench/11_prefill_attention.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/exl3_bench/12_prefill_moe_compact.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/exl3_bench/13_prefill_moe_pf2.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/exl3_bench/14_hc_mixer.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/exl3_bench/15_splitk_chain.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/exl3_bench/16_dense_dks_verify.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/exl3_bench/17_attn_dense_v3.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/exl3_bench/18_draft_slice_qsa.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/exl3_bench/19_qsa_xcheck_sel.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/exl3_bench/20_sel_tc_posture1.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/exl3_bench/21_small_levers.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/exl3_bench/22_dsplitk_ep.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/exl3_bench/23_tp_allreduce.cuh"))
    ) };
    // SPLIT1 Wave 3: kernels/gpu_naive.cu is include-split; the text is the main file then every section in include order.
    ("kernels/gpu_naive") => { concat!(
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/gpu_naive.cu")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/gpu_naive/01_decode_hadamard_gemm_wide.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/gpu_naive/02_layer_glue_attention.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/gpu_naive/03_gpu_sampling.cuh"))
    ) };
    // SPLIT1 Wave 3: kernels/gpu_fp4l.cu is include-split; the text is the main file then every section in include order.
    ("kernels/gpu_fp4l") => { concat!(
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/gpu_fp4l.cu")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/gpu_fp4l/01_mx8_nv4_gemms_dequant.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/gpu_fp4l/02_precision_levers_router_hc.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/gpu_fp4l/03_embed_gate_ple_heads.cuh"))
    ) };
    // SPLIT1 Wave 3: native/net_shim.c is include-split; the text is the main file then every section in include order.
    ("native/net_shim") => { concat!(
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/net_shim.c")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/net_shim/01_helpers_proxy_nway.inc")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/net_shim/02_init_accessors_bench.inc")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/net_shim/03_hot_path.inc")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/net_shim/04_startup_audit_channel.inc"))
    ) };
    // SPLIT1 Wave 3: kernels/gpu_batch.cu is include-split; the text is the .cu then every section in include order.
    ("kernels/gpu_batch") => { concat!(
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/gpu_batch.cu")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/gpu_batch/01_norms_tp_transport.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/gpu_batch/02_tp_prefill_conv_delta.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/gpu_batch/03_gdn_prefill_kv_q4.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/gpu_batch/04_kv_k8v4_tq_misc.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/gpu_batch/05_prefill_attn_splitk.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/gpu_batch/06_splitk_k8v4_k8v8.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/gpu_batch/07_splitk_tq_gq.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/gpu_batch/08_attn_reduce_sample.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/gpu_batch/09_gemm_mma_fp4.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/gpu_batch/10_gemm_dsv4_moe_ref.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/gpu_batch/11_moe_fp4_warp_tc.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/gpu_batch/12_moe_fold_descriptor.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/gpu_batch/13_gdn_rollback_fp8_ple.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/gpu_batch/14_qsa_indexer.cuh")),
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/gpu_batch/15_gptq.cuh"))
    ) };
}
