//! WP27 (PLAN/SURPASS_PLAN_2026-09-26.md) — graph DAG concurrency via capture-dependency editing.
//!
//! The EXL3 decode-step / verify / draft graphs are recorded by stream capture on the ONE blocking
//! compute stream (AGENTS §2.1: no second stream, never `fork_default_stream`). A captured graph's
//! topology is exactly the dependency set each launch saw at capture time: `cuStreamGetCaptureInfo_v2`
//! reads the node set the NEXT captured launch will depend on, and `cuStreamUpdateCaptureDependencies`
//! (SET) replaces it. So a run of launches that is data-independent of the run captured just before
//! it can be re-pointed at an earlier node set (the fork base) and becomes a sibling branch of the
//! graph; a later SET of the union of the branch tails joins them. The graph executor runs sibling
//! branches concurrently — concurrency lives in graph topology only (PLAN/11_E2_RECON §6).
//!
//! Eager launches (the stream is not capturing) are never touched: every `Cap` is inert there and
//! the same kernels run in stream order. Numerics: same kernels, same inputs, same launch shapes —
//! only the ordering between independent nodes changes (bitwise class).
//!
//! Rules the call sites follow (the buffer-by-buffer audit is in the WP27 commit message):
//! - a branch is a run of kernels whose reads/writes are disjoint from every sibling's writes
//!   (duplicated scratch where two chains would otherwise share one split-K workspace);
//! - no grid-barrier kernel (the hc fuse family: `xq_grid_barrier` spin waits needing co-residency)
//!   is ever concurrent with a branch: every fork opens after an hc mixer and joins before the next;
//! - a last-arrival (self re-arming counter) kernel sits on a branch only when its counter is
//!   private to that one launch site (the router fold's `fold_cnt`, the S-A3-z pair's `route_cnt`):
//!   no two co-running kernels ever share a counter or a workspace;
//! - inert under `--exl3-pdl=1` (programmatic edges carry edge data the v2 query cannot return).
//!
//! p3c re-audit (after WP20 rung 2 / DHEAD / REPRIME / WP13 / LMH / PSK joined the graphs): the
//! MoE fork is kept (WP20 rung 2 returns before its cut, so its epilogues stay linear; the fork
//! only splits the pre-WP20 fallback sequence, whose routed branch holds the router fold's
//! private counter as audited above); the GDN fork is kept (both branches: chain GEMMs incl. PSK
//! persistent split-K — static item striding, no inter-CTA wait — GEMVs, copies, conv, scan);
//! the PLE fork is DROPPED (its branch overlapped layer 0's WP20 rung-2 epilogues).
//!
//! WP21-v3 re-audit (split-K fixup epilogue, default on): every ks > 1 chain GEMM on a branch is now
//! a last-arrival kernel (exl3_hmma_gemm_ks_fx(_x2)) whose counter set is its weight matrix's own
//! `Quad.cnt` — private to that Quad, which no graph launches twice on sibling branches: GDN fork =
//! in_qkv (base) then in_z (from in_qkv's tail: ordered, own z scratch) beside counter-free a|b /
//! saves / conv / scan; MoE fork (pre-WP20 fallback) and W4/MOE SHOVL = the shared expert's
//! sh_gate|sh_up (two distinct slices, enforced) + sh_down beside a routed side with no Quad
//! counter (fold_cnt / moe_cnt only). No spin wait anywhere => no co-residency requirement. The
//! full note is at exl3_forward.rs `wp21_on`.
//!
//! A5-K1 (w6/SHOVL): `moe.shovl` (default ON; escape --moe-shovl=0) re-opens the MoE fork on the
//! DEFAULT WP20 rung-2 path: routed branch = router fold -> xq_had_suh_multi -> xq_moe_gu_epi
//! (private fold_cnt / moe_cnt), shared branch = sh_gate|sh_up + silu + sh_down (private Quad.cnt),
//! joined before xq_moe_dn_epi (whose combine reads ysh). The p3c "rung 2 stays linear" rule was a
//! conservative default, not a hazard: neither branch spin-waits, so any residency is deadlock-free.
//!
//! Switches: `--wp27-off=1` = linear capture (the pre-WP27 graphs and launch sequence exactly);
//! `--wp27-forks=moe,gdn` (default: both; `ple` is ignored since p3c) picks the forks; `--wp27-graphs=verify,draft,step`
//! (default: all) picks which graph classes get the dependency edits; `--wp27-prio=1` (opt-in)
//! instantiates the forked graphs with per-node priorities: every kernel node gets the context's
//! greatest priority EXCEPT the side branches (shared expert, GDN z chain, PLE prefix), so freed SM
//! slots go to the critical path first (scheduling only — no numeric effect).

use anyhow::Result;
use cudarc::driver::sys;
use std::cell::RefCell;
use crate::exl3_tune as tune;

/// Fork kinds (--wp27-forks).
pub const FORK_MOE: u8 = 1; // shared expert ∥ routed path (router .. yd), join at xq_moe_combine
pub const FORK_GDN: u8 = 2; // in_z chain (own scratch) ∥ conv -> scan; a|b ∥ in_qkv; joins before the scan / the gate
pub const FORK_PLE: u8 = 4; // PLE key/value projections + key norm ∥ layer 0's MoE window — DROPPED in p3c (cfg
                            // masks it: that window now holds WP20 rung-2 last-arrival epilogues)

/// Graph classes (--wp27-graphs) — set per body into `Scratch::wp27_cls`.
pub const CLS_VERIFY: u8 = 1;
pub const CLS_DRAFT: u8 = 2;
pub const CLS_STEP: u8 = 4;

/// The env-resolved state (printed once); the decisions are registry lookups since TUNE T0.
#[allow(dead_code)]
struct Cfg {
    forks: u8,
    graphs: u8,
    prio: bool,
}

fn parse_list(var: crate::opts::OptId, names: &[(&str, u8)]) -> u8 {
    parse_list_str(&var.to_string(), crate::opts::var(var).ok().as_deref(), names)
}

/// The list parse proper (CLI-1: split out so the unit test feeds values directly, no option set).
fn parse_list_str(var: &str, val: Option<&str>, names: &[(&str, u8)]) -> u8 {
    let all = names.iter().fold(0u8, |a, (_, b)| a | b);
    match val.ok_or(()) {
        Err(_) => all,
        Ok(s) => {
            let mut v = 0u8;
            for t in s.split(',').map(|t| t.trim().to_ascii_lowercase()).filter(|t| !t.is_empty()) {
                if t == "all" { v |= all; continue; }
                if t == "none" { continue; }
                match names.iter().find(|(n, _)| *n == t) {
                    Some((_, b)) => v |= b,
                    None => eprintln!("WP27: {var}: unknown entry '{t}' ignored"),
                }
            }
            v
        }
    }
}

fn names_of(v: u8, names: &[(&str, u8)]) -> String {
    let s: Vec<&str> = names.iter().filter(|(_, b)| v & b != 0).map(|(n, _)| *n).collect();
    if s.is_empty() { "none".into() } else { s.join(",") }
}

const FORK_NAMES: [(&str, u8); 3] = [("moe", FORK_MOE), ("gdn", FORK_GDN), ("ple", FORK_PLE)];
const CLS_NAMES: [(&str, u8); 3] = [("verify", CLS_VERIFY), ("draft", CLS_DRAFT), ("step", CLS_STEP)];

/// The env-resolved switch state — printed once (the boot receipt line); the DECISIONS read the
/// registry (`forks_now` / `graphs_now` / `prio_now`), whose env aliases parse these same variables
/// exactly, so with no tune table the two agree bit for bit.
fn cfg() -> &'static Cfg {
    static C: std::sync::OnceLock<Cfg> = std::sync::OnceLock::new();
    C.get_or_init(|| {
        if crate::opts::var(crate::opt!("wp27-off")).as_deref() == Ok("1") {
            println!("WP27: OFF (--wp27-off=1) — linear graph capture, pre-WP27 launch sequence");
            return Cfg { forks: 0, graphs: 0, prio: false };
        }
        if crate::opts::var(crate::opt!("exl3-pdl")).as_deref() == Ok("1") {
            println!("WP27: OFF under --exl3-pdl=1 (programmatic edges) — linear graph capture");
            return Cfg { forks: 0, graphs: 0, prio: false };
        }
        let mut forks = parse_list(crate::opt!("wp27-forks"), &FORK_NAMES);
        // p3c re-audit (preview/p3c, 2026-09-27): the PLE fork is DROPPED — its branch spans layer
        // 0's whole MoE window, which in p3c holds the A5 WP20 rung-2 epilogues (self re-arming
        // last-arrival counters, the default MoE path). Rule: no last-arrival-counter epilogue
        // (WP20 rung 2, DHEAD rescore) and no grid-barrier kernel (WP11 hc family) may run on a
        // parallel branch beside another kernel. Never enabled, even if listed; the linear
        // sequence (ple_prefix at the layer-1 boundary) is the pre-WP27 launch order exactly.
        if forks & FORK_PLE != 0 {
            if crate::opts::var(crate::opt!("wp27-forks")).map_or(false, |v| v.to_ascii_lowercase().contains("ple")) {
                println!("WP27: fork 'ple' ignored — dropped in p3c (it would run beside layer 0's WP20 rung-2 MoE epilogues)");
            }
            forks &= !FORK_PLE;
        }
        let graphs = parse_list(crate::opt!("wp27-graphs"), &CLS_NAMES);
        let prio = forks != 0 && graphs != 0 && crate::opts::var(crate::opt!("wp27-prio")).as_deref() == Ok("1");
        println!("WP27: capture-DAG forks [{}] in [{}] graphs, node priorities {} (--wp27-off=1 = linear \
                  capture; --wp27-forks / --wp27-graphs select; --wp27-prio=1 = critical-path priority)",
                 names_of(forks, &FORK_NAMES), names_of(graphs, &CLS_NAMES), if prio { "ON" } else { "off" });
        Cfg { forks, graphs, prio }
    })
}

// TUNE T0 (p3c): the WP27 switches as registry entries (S class: ordering between independent
// nodes only). wp27.forks is per graph family (a table may fork verify but not step); graphs and
// priorities are global. Under PDL (programmatic edges) every fork is inert, as before.
pub(crate) const T_WP27_FORKS: tune::TunableDef = tune::TunableDef {
    slot: 40, id: "wp27.forks", class: tune::Class::S, scope: tune::Scope::Fam,
    fams: &[tune::Fam::Step, tune::Fam::Verify, tune::Fam::DraftPass, tune::Fam::DraftChain, tune::Fam::Util],
    // bit 1 = moe, bit 2 = gdn (the PLE fork is dropped since p3c and masked out)
    domain: &[0, 1, 2, 3], default: (FORK_MOE | FORK_GDN) as i32, valid: tune::valid_any,
    xcheck: "WP27 capture-DAG audit (same kernels/inputs/launch shapes; per-sample live digest)",
    env: "--wp27-forks|--wp27-off",
    env_parse: || {
        if crate::opts::var(crate::opt!("wp27-off")).as_deref() == Ok("1") { return Some(0); }
        crate::opts::var(crate::opt!("wp27-forks")).ok()
            .map(|_| (parse_list(crate::opt!("wp27-forks"), &FORK_NAMES) & !FORK_PLE) as i32)
    },
    rev: 1, wp: "WP27",
};
pub(crate) const T_WP27_GRAPHS: tune::TunableDef = tune::TunableDef {
    slot: 41, id: "wp27.graphs", class: tune::Class::S, scope: tune::Scope::Global,
    fams: &[tune::Fam::Step, tune::Fam::Verify, tune::Fam::DraftPass, tune::Fam::DraftChain, tune::Fam::Util],
    domain: &[0, 1, 2, 3, 4, 5, 6, 7], default: (CLS_VERIFY | CLS_DRAFT | CLS_STEP) as i32, valid: tune::valid_any,
    xcheck: "WP27 capture-DAG audit",
    env: "--wp27-graphs|--wp27-off",
    env_parse: || {
        if crate::opts::var(crate::opt!("wp27-off")).as_deref() == Ok("1") { return Some(0); }
        crate::opts::var(crate::opt!("wp27-graphs")).ok().map(|_| parse_list(crate::opt!("wp27-graphs"), &CLS_NAMES) as i32)
    },
    rev: 1, wp: "WP27",
};
pub(crate) const T_WP27_PRIO: tune::TunableDef = tune::TunableDef {
    slot: 42, id: "wp27.prio", class: tune::Class::S, scope: tune::Scope::Global,
    fams: &[tune::Fam::Step, tune::Fam::Verify, tune::Fam::DraftPass, tune::Fam::DraftChain, tune::Fam::Util],
    domain: &[0, 1], default: 0, valid: tune::valid_any, xcheck: "scheduling only (node priorities; no numeric effect)",
    env: "--wp27-prio", env_parse: || tune::env_on_if_1(crate::opt!("wp27-prio")), rev: 1, wp: "WP27",
};

/// Forks in force for the current family scope (0 under PDL: programmatic edges carry edge data
/// the v2 capture query cannot return).
fn forks_now() -> u8 {
    let _ = cfg(); // the one-time receipt line
    if tune::get_cur(&crate::exl3_forward::T_PDL) != 0 { return 0; }
    (tune::get_cur(&T_WP27_FORKS).clamp(0, 7) as u8) & !FORK_PLE
}
fn graphs_now() -> u8 {
    tune::get_cur(&T_WP27_GRAPHS).clamp(0, 7) as u8
}
pub(crate) fn prio_now() -> bool {
    forks_now() != 0 && graphs_now() != 0 && tune::get_cur(&T_WP27_PRIO) != 0
}

/// Launch-sequence switch for one fork kind (duplicated scratch, early PLE prefix). Decided the
/// same way eager and captured, so an eager step runs the same buffers and kernel order as the
/// graph it is later captured into (only the captured dependencies differ).
pub fn fork_on(kind: u8) -> bool {
    forks_now() & kind != 0
}

type Node = sys::CUgraphNode;

/// A branch tail: the node set the stream's next launch depended on when the branch was cut.
/// Empty (and ignored) when the owning `Cap` is inert.
#[derive(Default)]
pub struct Tail(Vec<Node>);

/// (capture id, the node set the next captured launch on `stream` depends on); None when the
/// stream is not actively capturing.
fn deps(stream: sys::CUstream) -> Result<Option<(u64, Vec<Node>)>> {
    let mut st = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
    let mut id: sys::cuuint64_t = 0;
    let mut graph: sys::CUgraph = std::ptr::null_mut();
    let mut dp: *const Node = std::ptr::null();
    let mut n: usize = 0;
    let r = unsafe { sys::cuStreamGetCaptureInfo_v2(stream, &mut st, &mut id, &mut graph, &mut dp, &mut n) };
    anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "WP27: cuStreamGetCaptureInfo_v2 failed ({r:?})");
    if st != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE {
        return Ok(None);
    }
    // The array is owned by the driver and valid only until the next call on the stream: copy it.
    let v = if n == 0 || dp.is_null() { Vec::new() } else { unsafe { std::slice::from_raw_parts(dp, n) }.to_vec() };
    Ok(Some((id, v)))
}

fn deps_active(stream: sys::CUstream) -> Result<Vec<Node>> {
    Ok(deps(stream)?.ok_or_else(|| anyhow::anyhow!("WP27: capture ended inside a fork"))?.1)
}

/// Replace the dependency set of the next captured launch on `stream`.
fn set(stream: sys::CUstream, nodes: &[Node]) -> Result<()> {
    let mut v = nodes.to_vec();
    let p = if v.is_empty() { std::ptr::null_mut() } else { v.as_mut_ptr() };
    let r = unsafe {
        sys::cuStreamUpdateCaptureDependencies(stream, p, v.len(),
            sys::CUstreamUpdateCaptureDependencies_flags::CU_STREAM_SET_CAPTURE_DEPENDENCIES as std::ffi::c_uint)
    };
    anyhow::ensure!(r == sys::CUresult::CUDA_SUCCESS, "WP27: cuStreamUpdateCaptureDependencies failed ({r:?})");
    Ok(())
}

fn union_into(dst: &mut Vec<Node>, src: &[Node]) {
    for &n in src {
        if !dst.contains(&n) { dst.push(n); }
    }
}

/// Dependencies of one captured node (empty on a query failure).
fn node_deps(n: Node) -> Vec<Node> {
    let mut k: usize = 0;
    unsafe {
        if sys::cuGraphNodeGetDependencies(n, std::ptr::null_mut(), &mut k) != sys::CUresult::CUDA_SUCCESS || k == 0 {
            return Vec::new();
        }
        let mut d: Vec<Node> = vec![std::ptr::null_mut(); k];
        let mut k2 = k;
        if sys::cuGraphNodeGetDependencies(n, d.as_mut_ptr(), &mut k2) != sys::CUresult::CUDA_SUCCESS {
            return Vec::new();
        }
        d.truncate(k2.min(k));
        d
    }
}

/// The nodes of a linear branch: everything reachable backwards from `end` without crossing
/// `start` (the dependency set the branch was launched from). Bounded.
fn branch_nodes(end: &[Node], start: &[Node]) -> Vec<Node> {
    let mut out: Vec<Node> = Vec::new();
    let mut stack: Vec<Node> = end.to_vec();
    while let Some(n) = stack.pop() {
        if start.contains(&n) || out.contains(&n) { continue; }
        out.push(n);
        if out.len() > 256 { break; } // a branch is a handful of launches; never walk the graph
        for d in node_deps(n) {
            if !start.contains(&d) && !out.contains(&d) { stack.push(d); }
        }
    }
    out
}

thread_local! {
    /// Side-branch nodes recorded during the current capture (capture id, node addresses) —
    /// consumed by `finish` for the --wp27-prio node priorities.
    static SIDE: RefCell<(u64, Vec<usize>)> = RefCell::new((0, Vec::new()));
}

struct Inner {
    stream: sys::CUstream,
    id: u64,
    base: Vec<Node>,
    open: Vec<Node>, // nodes of cut branch tails not yet joined back into the stream
    side_start: Option<Vec<Node>>,
}

/// One fork point inside a capture. Inert (every method a no-op) unless the fork kind and the
/// graph class are enabled AND the stream is actively capturing.
pub struct Cap {
    inner: Option<Inner>,
}

impl Cap {
    pub fn none() -> Self {
        Cap { inner: None }
    }

    /// Open a fork at the current capture position (the base = the next launch's dependency set).
    pub fn open(stream: sys::CUstream, cls: u8, kind: u8) -> Self {
        if forks_now() & kind == 0 || graphs_now() & cls == 0 {
            return Cap::none();
        }
        match deps(stream) {
            Ok(Some((id, base))) => {
                SIDE.with(|s| {
                    let mut s = s.borrow_mut();
                    if s.0 != id { s.0 = id; s.1.clear(); } // a new capture: drop stale records
                });
                Cap { inner: Some(Inner { stream, id, base, open: Vec::new(), side_start: None }) }
            }
            Ok(None) => Cap::none(),
            Err(e) => {
                // Linear capture is always correct: a failed query only costs the overlap.
                static WARN: std::sync::Once = std::sync::Once::new();
                WARN.call_once(|| eprintln!("WP27: capture query failed ({e:#}) — this fork stays linear"));
                Cap::none()
            }
        }
    }

    /// The current capture position as a tail (no edit): a later `cut_to` can resume from it.
    pub fn mark(&mut self) -> Result<Tail> {
        let Some(i) = self.inner.as_mut() else { return Ok(Tail::default()) };
        Ok(Tail(deps_active(i.stream)?))
    }

    /// End the current branch: return its tail and restart the next launch from the fork base.
    pub fn cut(&mut self) -> Result<Tail> {
        let Some(i) = self.inner.as_mut() else { return Ok(Tail::default()) };
        let cur = deps_active(i.stream)?;
        set(i.stream, &i.base)?;
        union_into(&mut i.open, &cur);
        Ok(Tail(cur))
    }

    /// End the current branch and resume from `from` (a `mark` or an earlier cut's tail).
    pub fn cut_to(&mut self, from: &Tail) -> Result<Tail> {
        let Some(i) = self.inner.as_mut() else { return Ok(Tail::default()) };
        let cur = deps_active(i.stream)?;
        set(i.stream, &from.0)?;
        union_into(&mut i.open, &cur);
        i.open.retain(|n| !from.0.contains(n));
        Ok(Tail(cur))
    }

    /// Join: the next launch depends on the current position AND on the branch tail `t`.
    pub fn join(&mut self, t: &Tail) -> Result<()> {
        let Some(i) = self.inner.as_mut() else { return Ok(()) };
        if t.0.is_empty() { return Ok(()); }
        let mut cur = deps_active(i.stream)?;
        union_into(&mut cur, &t.0);
        set(i.stream, &cur)?;
        i.open.retain(|n| !t.0.contains(n));
        Ok(())
    }

    /// --wp27-prio: the launches until `side_end` form a side branch (keeps the default, lowest
    /// node priority; every other kernel node of the graph is raised). No-op otherwise.
    pub fn side_begin(&mut self) -> Result<()> {
        if !prio_now() { return Ok(()); }
        let Some(i) = self.inner.as_mut() else { return Ok(()) };
        i.side_start = Some(deps_active(i.stream)?);
        Ok(())
    }

    pub fn side_end(&mut self) -> Result<()> {
        let Some(i) = self.inner.as_mut() else { return Ok(()) };
        let Some(start) = i.side_start.take() else { return Ok(()) };
        let end = deps_active(i.stream)?;
        let nodes = branch_nodes(&end, &start);
        let id = i.id;
        SIDE.with(|s| {
            let mut s = s.borrow_mut();
            if s.0 != id { s.0 = id; s.1.clear(); }
            s.1.extend(nodes.iter().map(|&n| n as usize));
        });
        Ok(())
    }
}

impl Drop for Cap {
    /// Safety net: a scope that leaves with a cut branch still unjoined (an early return, or a
    /// later merge that bypasses the join) joins it here, so no consumer can outrun a branch.
    fn drop(&mut self) {
        let Some(i) = self.inner.as_mut() else { return };
        if i.open.is_empty() { return; }
        if let Ok(Some((_, mut cur))) = deps(i.stream) {
            union_into(&mut cur, &i.open);
            let ok = set(i.stream, &cur).is_ok();
            static WARN: std::sync::Once = std::sync::Once::new();
            WARN.call_once(|| eprintln!("WP27: a fork left scope with an unjoined branch — joined at scope end ({})",
                                        if ok { "ok" } else { "FAILED: capture will be discarded" }));
        }
    }
}

/// The context's greatest (numerically lowest) stream priority, cached.
fn greatest_priority() -> i32 {
    static P: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    *P.get_or_init(|| {
        let (mut least, mut greatest) = (0i32, 0i32);
        let r = unsafe { sys::cuCtxGetStreamPriorityRange(&mut least, &mut greatest) };
        if r == sys::CUresult::CUDA_SUCCESS { greatest } else { 0 }
    })
}

/// Post-capture bookkeeping for one graph (call after a successful EndCapture, before
/// `instantiate`): with --wp27-prio, raise every kernel node that is not on a recorded side
/// branch to the greatest priority; then print the structural receipt (bounded) — nodes, edges,
/// forks (fan-out > 1), joins (fan-in > 1). A linear capture has edges = nodes - 1 and 0 forks.
pub fn finish(graph: sys::CUgraph, what: &str, m: usize) {
    let prio = prio_now();
    if forks_now() == 0 { return; }
    let side: Vec<usize> = SIDE.with(|s| std::mem::take(&mut s.borrow_mut().1));
    let mut n: usize = 0;
    unsafe {
        if sys::cuGraphGetNodes(graph, std::ptr::null_mut(), &mut n) != sys::CUresult::CUDA_SUCCESS { return; }
    }
    let mut raised = 0usize;
    if prio && n > 0 {
        let mut nodes: Vec<Node> = vec![std::ptr::null_mut(); n];
        let mut n2 = n;
        let ok = unsafe { sys::cuGraphGetNodes(graph, nodes.as_mut_ptr(), &mut n2) } == sys::CUresult::CUDA_SUCCESS;
        if ok {
            let p = greatest_priority();
            for &nd in &nodes[..n2.min(n)] {
                if side.contains(&(nd as usize)) { continue; }
                let mut t = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
                if unsafe { sys::cuGraphNodeGetType(nd, &mut t) } != sys::CUresult::CUDA_SUCCESS
                    || t != sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL { continue; }
                let mut v: sys::CUkernelNodeAttrValue = unsafe { std::mem::zeroed() };
                v.priority = p;
                let r = unsafe { sys::cuGraphKernelNodeSetAttribute(nd,
                    sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PRIORITY, &v) };
                if r == sys::CUresult::CUDA_SUCCESS { raised += 1; }
            }
        }
    }
    // bounded: graphs are keyed by (width, regime, QSA bucket, slot, ...) — the first 64 suffice
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) >= 64 { return; }
    let mut e: usize = 0;
    unsafe {
        if sys::cuGraphGetEdges(graph, std::ptr::null_mut(), std::ptr::null_mut(), &mut e) != sys::CUresult::CUDA_SUCCESS { return; }
    }
    let mut from: Vec<Node> = vec![std::ptr::null_mut(); e];
    let mut to: Vec<Node> = vec![std::ptr::null_mut(); e];
    let mut e2 = e;
    if e > 0 {
        let r = unsafe { sys::cuGraphGetEdges(graph, from.as_mut_ptr(), to.as_mut_ptr(), &mut e2) };
        if r != sys::CUresult::CUDA_SUCCESS { return; }
    }
    let mut outs: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut ins: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for k in 0..e2.min(e) {
        *outs.entry(from[k] as usize).or_insert(0) += 1;
        *ins.entry(to[k] as usize).or_insert(0) += 1;
    }
    let forks = outs.values().filter(|&&c| c > 1).count();
    let joins = ins.values().filter(|&&c| c > 1).count();
    let pr = if prio { format!(", side nodes {} / raised {} (prio {})", side.len(), raised, greatest_priority()) }
             else { String::new() };
    println!("WP27_DAG {what} m={m}: nodes {n} edges {e} forks {forks} joins {joins}{pr} \
              (linear = {} edges, 0 forks)", n.saturating_sub(1));
}

/// Instantiate a captured graph: with --wp27-prio the per-node priorities set by `finish` are
/// honoured (CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY); otherwise the original call exactly.
pub unsafe fn instantiate(exec: &mut sys::CUgraphExec, graph: sys::CUgraph) -> sys::CUresult {
    if prio_now() {
        sys::cuGraphInstantiateWithFlags(exec, graph,
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY as std::ffi::c_ulonglong)
    } else {
        sys::cuGraphInstantiate_v2(exec, graph, std::ptr::null_mut(), std::ptr::null_mut(), 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_lists() {
        // unset -> all; explicit subsets; unknown entries ignored
        // (CLI-1: the parse is fed directly — the old GB10_WP27_TEST_LIST env round trip is gone)
        assert_eq!(parse_list_str("test", None, &FORK_NAMES), FORK_MOE | FORK_GDN | FORK_PLE);
        assert_eq!(parse_list_str("test", Some("moe, PLE"), &FORK_NAMES), FORK_MOE | FORK_PLE);
        assert_eq!(parse_list_str("test", Some("none"), &FORK_NAMES), 0);
        assert_eq!(parse_list_str("test", Some("bogus,step"), &CLS_NAMES), CLS_STEP);
        assert_eq!(names_of(FORK_MOE | FORK_GDN, &FORK_NAMES), "moe,gdn");
        assert_eq!(names_of(0, &CLS_NAMES), "none");
    }

    #[test]
    fn union_dedups() {
        let a = 0x10usize as Node;
        let b = 0x20usize as Node;
        let c = 0x30usize as Node;
        let mut v = vec![a, b];
        union_into(&mut v, &[b, c, c]);
        assert_eq!(v, vec![a, b, c]);
    }

    #[test]
    fn inert_cap_is_noop() {
        let mut f = Cap::none();
        let q = f.mark().unwrap();
        assert!(q.0.is_empty());
        f.side_begin().unwrap();
        f.side_end().unwrap();
        let t = f.cut().unwrap();
        assert!(t.0.is_empty());
        let t2 = f.cut_to(&q).unwrap();
        f.join(&t2).unwrap();
        f.join(&t).unwrap();
    }

    #[test]
    fn branch_walk_stops_at_start() {
        // end == start: nothing on the branch (no driver query is made for start nodes)
        let a = 0x10usize as Node;
        assert!(branch_nodes(&[a], &[a]).is_empty());
    }
}
