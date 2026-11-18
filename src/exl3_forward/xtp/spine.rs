//! TP-C: the graph spine audit — every doorbell node must be a cut vertex of the captured DAG. See `mod.rs` for the map.

use super::*;

// ---------------------------------------------------------------------------------------------
// TP-C: the graph spine audit (design §2.3). Every doorbell node (K1 tp_gate_copy_signal / K2
// tp_wait_add — identified by their first kernel argument, this rank's doorbell ctx pointer) must
// be a CUT VERTEX of the captured DAG: every other node is its ancestor or its descendant, i.e. no
// kernel (a WP27 sibling branch, a grid-barrier hc kernel) can ever run concurrently with a
// barrier, and no barrier runs inside a fork. A graph that fails is discarded (the eager path runs
// the same launches linearly), loudly.
// ---------------------------------------------------------------------------------------------

static SPINE_LOG: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static SPINE_SEEN: std::sync::Mutex<std::collections::BTreeSet<(String, usize, usize)>> =
    std::sync::Mutex::new(std::collections::BTreeSet::new());
pub static SPINE_BAD: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static SPINE_OK: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

impl FwdModel {
    /// true = the graph may be instantiated (TP=1: always; TP: every doorbell on the spine).
    pub(in crate::exl3_forward) fn tp_spine_audit(&self, graph: cudarc::driver::sys::CUgraph, what: &str, m: usize) -> bool {
        use cudarc::driver::sys;
        let Some(tp) = self.tp.as_ref() else { return true };
        let mut n: usize = 0;
        unsafe {
            if sys::cuGraphGetNodes(graph, std::ptr::null_mut(), &mut n) != sys::CUresult::CUDA_SUCCESS {
                crate::dprintln!("TP-SPINE {what} m={m}: node query failed — graph discarded (eager)");
                crate::metrics::graph_capture_lost(what, &format!("spine audit, width/index {m}"), "tp_spine_audit: node query failed");
                return false;
            }
        }
        let mut nodes: Vec<sys::CUgraphNode> = vec![std::ptr::null_mut(); n];
        let mut e: usize = 0;
        let ok_q = unsafe {
            sys::cuGraphGetNodes(graph, nodes.as_mut_ptr(), &mut n) == sys::CUresult::CUDA_SUCCESS
                && sys::cuGraphGetEdges(graph, std::ptr::null_mut(), std::ptr::null_mut(), &mut e) == sys::CUresult::CUDA_SUCCESS
        };
        let mut from: Vec<sys::CUgraphNode> = vec![std::ptr::null_mut(); e];
        let mut to: Vec<sys::CUgraphNode> = vec![std::ptr::null_mut(); e];
        let ok_e = ok_q && (e == 0 || unsafe {
            sys::cuGraphGetEdges(graph, from.as_mut_ptr(), to.as_mut_ptr(), &mut e) == sys::CUresult::CUDA_SUCCESS
        });
        if !ok_e {
            crate::dprintln!("TP-SPINE {what} m={m}: edge query failed — graph discarded (eager)");
            crate::metrics::graph_capture_lost(what, &format!("spine audit, width/index {m}"), "tp_spine_audit: edge query failed");
            return false;
        }
        let idx: std::collections::HashMap<usize, usize> = nodes.iter().enumerate().map(|(i, &p)| (p as usize, i)).collect();
        let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut pred: Vec<Vec<usize>> = vec![Vec::new(); n];
        for k in 0..e {
            let (Some(&a), Some(&b)) = (idx.get(&(from[k] as usize)), idx.get(&(to[k] as usize))) else { continue };
            succ[a].push(b);
            pred[b].push(a);
        }
        // doorbell nodes: kernel nodes whose first argument is this rank's doorbell ctx
        let mut bells: Vec<usize> = Vec::new();
        for (i, &nd) in nodes.iter().enumerate() {
            let mut t = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
            if unsafe { sys::cuGraphNodeGetType(nd, &mut t) } != sys::CUresult::CUDA_SUCCESS
                || t != sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL { continue; }
            let mut p: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
            if unsafe { sys::cuGraphKernelNodeGetParams(nd, &mut p) } != sys::CUresult::CUDA_SUCCESS { continue; }
            if p.kernelParams.is_null() { continue; }
            let a0 = unsafe { *p.kernelParams };
            if a0.is_null() { continue; }
            // TP-G: any grid — the folded K1 producers and the multi-block decode K2 carry the ctx first too
            let a0v = unsafe { *(a0 as *const u64) };
            if a0v == tp.ctx || tp.single.as_ref().map_or(false, |s| s.ctx == a0v) { bells.push(i); }   // TP-4F2: the single-stage ctx's doorbells too
        }
        let reach = |start: usize, adj: &Vec<Vec<usize>>| -> usize {
            let mut seen = vec![false; n];
            let mut st = vec![start];
            let mut c = 0usize;
            seen[start] = true;
            while let Some(x) = st.pop() {
                for &y in &adj[x] { if !seen[y] { seen[y] = true; c += 1; st.push(y); } }
            }
            c
        };
        let mut bad = 0usize;
        let mut worst = 0usize;
        for &b in &bells {
            let conc = n - 1 - reach(b, &succ) - reach(b, &pred);
            if conc > 0 { bad += 1; worst = worst.max(conc); }
        }
        let ok = bad == 0;
        if ok { SPINE_OK.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
        else {
            SPINE_BAD.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // S-B9-REL-INTEG C1: the refusal discards the graph (the shape runs eager): it is a lost capture
            crate::metrics::graph_capture_lost(what, &format!("spine audit, width/index {m}"), "tp_spine_audit: a barrier has a concurrent node");
        }
        // TP-H2: also log the first graph of every new (class, width, doorbell count) — the sampled-row gather
        // graphs are captured at runtime, after the boot precapture has used up the first 24 lines
        let new_shape = SPINE_SEEN.lock().map(|mut s| s.insert((what.to_string(), m, bells.len()))).unwrap_or(false);
        if !ok || SPINE_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 24 || new_shape {
            crate::dprintln!("TP-SPINE {what} m={m}: {n} nodes, {e} edges, {} doorbell nodes (K1+K2), {bad} with a concurrent node{} -> {}",
                     bells.len(), if bad > 0 { format!(" (worst {worst})") } else { String::new() },
                     if ok { "OK (every barrier is a cut vertex of the DAG)" } else { "VIOLATION — graph discarded, this shape runs eager" });
        }
        ok
    }
}
