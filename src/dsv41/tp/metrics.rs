//! DSV4.1 TP (leg TPC): `/metrics` rows of the comm layer (engagement receipts, AGENTS §1b-bis 2). Rendered by
//! `metrics::render` through one line; nothing is printed unless a dsv41 TP comm attached in this process, so
//! every other family's `/metrics` is byte-identical.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

static ATTACHED: AtomicU64 = AtomicU64::new(0);
static WORLD: AtomicU64 = AtomicU64::new(0);
static COLLECTIVES: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
static ROUNDS: AtomicU64 = AtomicU64::new(0);
static AGREES: AtomicU64 = AtomicU64::new(0);
static ABORTS: AtomicU64 = AtomicU64::new(0);
static GRAPH_REPLAYS: AtomicU64 = AtomicU64::new(0);

pub fn attached(world: usize) { ATTACHED.store(1, Relaxed); WORLD.store(world as u64, Relaxed); }
/// one collective enqueued (bytes on the wire per rank)
pub fn collective(bytes: usize) { COLLECTIVES.fetch_add(1, Relaxed); BYTES.fetch_add(bytes as u64, Relaxed); }
pub fn round() { ROUNDS.fetch_add(1, Relaxed); }
pub fn agreed() { AGREES.fetch_add(1, Relaxed); }
pub fn abort() { ABORTS.fetch_add(1, Relaxed); }
pub fn graph_replays(n: u64) { GRAPH_REPLAYS.fetch_add(n, Relaxed); }
pub fn collectives() -> u64 { COLLECTIVES.load(Relaxed) }

// leg FW: the serving engine's paths (lanes, PK1's indexer decode plan, CED prefill rows, graphs)
static FW_CALLS: AtomicU64 = AtomicU64::new(0);
static FW_MIXED: AtomicU64 = AtomicU64::new(0);
static FW_ROWS: AtomicU64 = AtomicU64::new(0);
static FW_IDX_PLAN: [AtomicU64; 4] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
/// leg P4: prefill calls whose last chunk rode the chunk before it (`--dsv41-prefill-tail-merge`)
pub static PF_TAIL_MERGED: AtomicU64 = AtomicU64::new(0);
static FW_IDX_CHUNKED: AtomicU64 = AtomicU64::new(0);
static FW_CED_ROWS: AtomicU64 = AtomicU64::new(0);
static FW_EXACT_ROWS: AtomicU64 = AtomicU64::new(0);
static FW_GRAPH_CAPTURES: AtomicU64 = AtomicU64::new(0);
static FW_GRAPH_EAGER: AtomicU64 = AtomicU64::new(0);
/// One forward call of `lanes` lanes and `rows` rows.
pub fn call(lanes: usize, rows: usize) { FW_CALLS.fetch_add(1, Relaxed); FW_ROWS.fetch_add(rows as u64, Relaxed); if lanes > 1 { FW_MIXED.fetch_add(1, Relaxed); } }
/// An index layer served by PK1's decode plan with R rows per block and G key chunks.
pub fn idx_plan(r: usize, g: usize) {
    let i = match r { 1 => 0, 2 => 1, 4 => 2, _ => 3 };
    FW_IDX_PLAN[i].fetch_add(1, Relaxed);
    if g > 1 { FW_IDX_CHUNKED.fetch_add(1, Relaxed); }
}
/// Layer-rows a prefill computed (CED) and the layer-rows the every-layer prefill would have computed.
pub fn ced_rows(ced: u64, exact: u64) { FW_CED_ROWS.fetch_add(ced, Relaxed); FW_EXACT_ROWS.fetch_add(exact, Relaxed); }
pub fn graph_capture() { FW_GRAPH_CAPTURES.fetch_add(1, Relaxed); }
// leg PF1: the head over a round's lanes (one launch per run of consecutive lanes) and the PDL launches built
static FW_HEAD_LAUNCHES: AtomicU64 = AtomicU64::new(0);
static FW_HEAD_SEGS: AtomicU64 = AtomicU64::new(0);
/// A round's head: `launches` head launches served `segs` lane segments.
pub fn head_launches(launches: usize, segs: usize) { FW_HEAD_LAUNCHES.fetch_add(launches as u64, Relaxed); FW_HEAD_SEGS.fetch_add(segs as u64, Relaxed); }
pub fn graph_eager() { FW_GRAPH_EAGER.fetch_add(1, Relaxed); }
/// leg PS3: captured graphs destroyed by the LRU (`mem::GRAPH_CAP`), and the distinct call shapes (keys) seen so far
static FW_GRAPH_EVICTED: AtomicU64 = AtomicU64::new(0);
static FW_GRAPH_SHAPES: AtomicU64 = AtomicU64::new(0);
pub fn graph_evicted() { FW_GRAPH_EVICTED.fetch_add(1, Relaxed); }
pub fn graph_shapes(n: usize) { FW_GRAPH_SHAPES.store(n as u64, Relaxed); }

pub fn append(o: &mut String) {
    if ATTACHED.load(Relaxed) == 0 { return; }
    use std::fmt::Write as _;
    let rows: [(&str, &str, u64); 8] = [
        ("velogb10_dsv41_tp_world", "TP ranks of the dsv41 comm layer", WORLD.load(Relaxed)),
        ("velogb10_dsv41_tp_collectives_total", "collectives enqueued (layer reduces, head and drafter exchanges)", COLLECTIVES.load(Relaxed)),
        ("velogb10_dsv41_tp_bytes_total", "bytes sent per rank by those collectives", BYTES.load(Relaxed)),
        ("velogb10_dsv41_tp_rounds_total", "decode rounds (or smoke rounds) run over the comm layer", ROUNDS.load(Relaxed)),
        ("velogb10_dsv41_tp_agrees_total", "per-step agree tokens exchanged (all equal, or the run stopped)", AGREES.load(Relaxed)),
        ("velogb10_dsv41_tp_aborts_total", "transport aborts noticed (a rank stops; never served)", ABORTS.load(Relaxed)),
        ("velogb10_dsv41_tp_graph_replays_total", "captured rounds replayed", GRAPH_REPLAYS.load(Relaxed)),
        ("velogb10_dsv41_tp_attached", "1 = a dsv41 TP comm layer attached in this process", 1),
    ];
    for (n, h, v) in rows { let _ = writeln!(o, "# HELP {n} {h}\n# TYPE {n} counter\n{n} {v}"); }
    let fw: [(&str, &str, u64); 11] = [
        ("velogb10_dsv41_fw_head_launches_total", "round head launches (leg PF1: one launch per run of consecutive lanes)", FW_HEAD_LAUNCHES.load(Relaxed)),
        ("velogb10_dsv41_fw_head_segments_total", "lane segments those head launches served (segments > launches = lanes shared a launch)", FW_HEAD_SEGS.load(Relaxed)),
        ("velogb10_dsv41_pdl_launches_total", "kernel launches built with programmatic dependent launch (--dsv41-pdl; graph replays re-run captured launches without counting)", crate::dsv41::gpu::pdl_launches()),
        ("velogb10_dsv41_fw_calls_total", "TP forward calls (prefill chunks, decode and verify rounds)", FW_CALLS.load(Relaxed)),
        ("velogb10_dsv41_fw_mixed_lane_calls_total", "TP forward calls that carried rows of two or more lanes", FW_MIXED.load(Relaxed)),
        ("velogb10_dsv41_fw_rows_total", "rows of those calls", FW_ROWS.load(Relaxed)),
        ("velogb10_dsv41_fw_idx_chunked_total", "index layers whose decode plan cut the keys into more than one chunk", FW_IDX_CHUNKED.load(Relaxed)),
        ("velogb10_dsv41_fw_ced_layer_rows_total", "layer-rows computed by CED prefills", FW_CED_ROWS.load(Relaxed)),
        ("velogb10_dsv41_fw_exact_layer_rows_total", "layer-rows an every-layer prefill of the same prompts would compute", FW_EXACT_ROWS.load(Relaxed)),
        ("velogb10_dsv41_fw_graph_captures_total", "decode / verify round shapes captured as CUDA graphs", FW_GRAPH_CAPTURES.load(Relaxed)),
        ("velogb10_dsv41_fw_graph_eager_total", "decode / verify calls run eagerly (no graph for that shape)", FW_GRAPH_EAGER.load(Relaxed)),
    ];
    for (n, h, v) in fw { let _ = writeln!(o, "# HELP {n} {h}\n# TYPE {n} counter\n{n} {v}"); }
    let _ = writeln!(o, "# HELP velogb10_dsv41_fw_graph_evictions_total captured call graphs destroyed by the least-recently-used cap ({} live; a later call of that shape is captured again)\n# TYPE velogb10_dsv41_fw_graph_evictions_total counter\nvelogb10_dsv41_fw_graph_evictions_total {}",
                     super::mem::GRAPH_CAP, FW_GRAPH_EVICTED.load(Relaxed));
    let rows_keys = crate::opts::lookup("dsv41-graph-keys").and_then(crate::opts::get).as_deref() != Some("split");
    let _ = writeln!(o, "# HELP velogb10_dsv41_fw_graph_keys How decode / verify graphs are keyed (rows = per total rows, the lanes' split as data, leg PG1; split = per (lane, rows) vector)\n# TYPE velogb10_dsv41_fw_graph_keys gauge\nvelogb10_dsv41_fw_graph_keys{{mode=\"{}\"}} 1", if rows_keys { "rows" } else { "split" });
    let _ = writeln!(o, "# HELP velogb10_dsv41_fw_graph_shapes_seen distinct call shapes (lanes and rows per lane, context class, head rows, draft steps) seen by the graph cache\n# TYPE velogb10_dsv41_fw_graph_shapes_seen gauge\nvelogb10_dsv41_fw_graph_shapes_seen {}",
                     FW_GRAPH_SHAPES.load(Relaxed));
    let _ = writeln!(o, "# HELP velogb10_dsv41_fw_idx_decode_plan_total index layers served by PK1's decode plan, by rows per block\n# TYPE velogb10_dsv41_fw_idx_decode_plan_total counter");
    for (i, r) in [1, 2, 4, 8].iter().enumerate() { let _ = writeln!(o, "velogb10_dsv41_fw_idx_decode_plan_total{{r=\"{r}\"}} {}", FW_IDX_PLAN[i].load(Relaxed)); }
    // leg PP1: the prefill paths (same bits as the paths they replace; switches --dsv41-pf-router-wide / -gemv-rows / -engram-ahead)
    let pp1: [(&str, &str, u64); 2] = [
        ("velogb10_dsv41_pf_router_wide_total", "prefill router calls (> 32 rows) served by the wide entry in one launch (leg PP1; 0 with --dsv41-pf-router-wide off)",
         crate::dsv41::gpu::PF_ROUTER_WIDE.load(Relaxed)),
        ("velogb10_dsv41_pf_gemv_rows_total", "plain GEMV calls (> 32 rows) spread along the rows over several block rows (leg PP1; 0 with --dsv41-pf-gemv-rows off)",
         crate::dsv41::gpu::PF_GEMV_ROWS.load(Relaxed)),
    ];
    let _ = writeln!(o, "# HELP velogb10_dsv41_pf_tail_merged_total prefill calls whose last chunk of --dsv41-prefill-tail-merge rows or fewer rode the chunk before it (leg P4; 0 with the flag at 0)\n# TYPE velogb10_dsv41_pf_tail_merged_total counter\nvelogb10_dsv41_pf_tail_merged_total {}", PF_TAIL_MERGED.load(Relaxed));
    let _ = writeln!(o, "# HELP velogb10_dsv41_pf_gemv_small_total plain GEMV calls of narrow weights (<= 256 columns) at 1..=32 rows served by the one-warp-per-block entry (leg P4; 0 with --dsv41-pf-gemv-small off; counts graph captures and eager calls, not replays)\n# TYPE velogb10_dsv41_pf_gemv_small_total counter\nvelogb10_dsv41_pf_gemv_small_total {}", crate::dsv41::gpu::PF_GEMV_SMALL.load(Relaxed));
    for (n, h, v) in pp1 { let _ = writeln!(o, "# HELP {n} {h}\n# TYPE {n} counter\n{n} {v}"); }
    super::pd1::append(o);
    super::xchg::append(o);
    super::pa1::append(o);
    super::mem::append(o);
    super::idle::append(o);
    super::pit::append(o);
}

/// The FW counters in one line (the program's receipt; the server renders them on /metrics).
pub fn fw_line() -> String {
    format!("fw counters: calls {} (mixed-lane {}, rows {}), index layers on the decode plan r1/r2/r4/r8 {}/{}/{}/{} (key-chunked {}), CED layer-rows {} of {} exact, graphs captured {} / eager calls {}, head launches {} for {} lane segments, PDL launches built {}",
            FW_CALLS.load(Relaxed), FW_MIXED.load(Relaxed), FW_ROWS.load(Relaxed), FW_IDX_PLAN[0].load(Relaxed), FW_IDX_PLAN[1].load(Relaxed), FW_IDX_PLAN[2].load(Relaxed),
            FW_IDX_PLAN[3].load(Relaxed), FW_IDX_CHUNKED.load(Relaxed), FW_CED_ROWS.load(Relaxed), FW_EXACT_ROWS.load(Relaxed), FW_GRAPH_CAPTURES.load(Relaxed), FW_GRAPH_EAGER.load(Relaxed),
            FW_HEAD_LAUNCHES.load(Relaxed), FW_HEAD_SEGS.load(Relaxed), crate::dsv41::gpu::pdl_launches())
}
