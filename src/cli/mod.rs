//! The binary's mode drivers, probes and benches, moved out of `main.rs` (which keeps `fn main`, the mode
//! dispatch). Items called from `main.rs` are `pub(crate)` and re-exported here; the rest are private to their file.
//!
//! File map:
//! - `args.rs`: argv helpers, option bridge, draft-source resolution, model loader for probes, interactive CLI (`run_cli`)
//! - `bench_basic.rs`: the basic prefill/decode/batch/tree/lane/accept/tau benches
//! - `bench_df2.rs`: DFlash2 sampling benches (sample, real-question, matrix)
//! - `bench_mtp.rs`: MTP verify/profile/sample/file benches and the MTP calibration cache helpers
//! - `dsv4_serve.rs`: DSV4 TP server loop with the DSpark depth policy and serve state (+ child `dsv4_serve/dspark_driver.rs`: DSpark serve driver, node loop)
//! - `help.rs`: `print_help`, the hand-written help text
//! - `probe_df2_draft.rs`: DFlash/DFlash2 draft-side probes, round loaders, spec-scheduler builder
//! - `probe_df2_misc.rs`: DFlash2 lossless, tap-capture, prime and graph probes
//! - `probe_df2_round.rs`: `run_probe_df2_round` (one long probe, kept whole)
//! - `probe_dspark.rs`: DSpark bind/synth/main/extract/rollback probes
//! - `probe_dspark_round.rs`: DSpark round probe and its draft-dir loader
//! - `probe_tq.rs`: `run_probe_tq`
//! - `probes_dsv4.rs`: DSV4 main, prefix and TP-simulation probes
//! - `probes_serve.rs`: pinned serve probe, its arm runner, HTTP helpers, vision tower probe
//! - `probes_state.rs`: small single-model probes (Q4, state, verify width, reject, GEMM, MoE, perplexity, capture, argmax dump)
//! - `quantize.rs`: offline quantizer, GDN requant, requant simulation, DF2 bake and fidelity tools
//! - `replay.rs`: DFlash2 golden-case replays
//! - `serve.rs`: `run_server`, the server entry (kept whole)
//! - `stq.rs`: STQ bake, check and verify modes
//! - `tp_cluster.rs`: TP reduce/barrier benches, network test, cluster node and head, memory-budget report
//! - `tp_heads.rs`: per-family TP head and node drivers (EXL3, naive, DSV4) and `node_serve_tp`
//! - `tp_serve_trunk.rs`: `tp_serve` (trunk TP serve driver) and DSV4 bundle load/serve entry points
//! - `prehard1_cli_model_dir_tests.rs`: source-text test of the bare CLI path.

use super::*;

mod args;
mod bench_basic;
mod bench_df2;
mod bench_mtp;
mod dsv4_serve;
mod help;
mod probe_df2_draft;
mod probe_df2_misc;
mod probe_df2_round;
mod probe_dspark;
mod probe_dspark_round;
mod probe_tq;
mod probes_dsv4;
mod probes_serve;
mod probes_state;
mod quantize;
mod replay;
mod serve;
mod stq;
mod tp_cluster;
mod tp_heads;
mod tp_serve_trunk;

pub(crate) use self::args::*;
pub(crate) use self::bench_basic::*;
pub(crate) use self::bench_df2::*;
pub(crate) use self::bench_mtp::*;
use self::dsv4_serve::*;
pub(crate) use self::help::*;
pub(crate) use self::probe_df2_draft::*;
pub(crate) use self::probe_df2_misc::*;
pub(crate) use self::probe_df2_round::*;
pub(crate) use self::probe_dspark::*;
pub(crate) use self::probe_dspark_round::*;
pub(crate) use self::probe_tq::*;
pub(crate) use self::probes_dsv4::*;
pub(crate) use self::probes_serve::*;
pub(crate) use self::probes_state::*;
pub(crate) use self::quantize::*;
pub(crate) use self::replay::*;
pub(crate) use self::serve::*;
pub(crate) use self::stq::*;
pub(crate) use self::tp_cluster::*;
pub(crate) use self::tp_heads::*;
use self::tp_serve_trunk::*;


/// PREHARD1 [ISSUES 84]: the bare CLI path (no subcommand) honors --model-dir — it stands in for
/// --model when --model is absent, the tokenizer defaults INSIDE that dir, and a missing model
/// path is ONE clear line + exit 2, not a bare "os error 2" from a CWD-relative default.
#[cfg(test)]
mod prehard1_cli_model_dir_tests;
