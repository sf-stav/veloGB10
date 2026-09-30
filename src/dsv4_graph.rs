//! CUDA-graph decode for DSV4 (R3A.1 / E2 — the 12.4%-of-step launch-gap lever).
//!
//! Design (DSV4_R3A.md §8, the full audit + inventory):
//! - ONE whole-forward graph per FIRE VARIANT (the comp epilogue launches are the only
//!   token-to-token kernel-sequence difference; fire ⇔ (sp+1)%ratio==0 is position-
//!   deterministic): V0 (no fire), V4 (CSA ratio-4 fires), V128 (CSA+HCA fire). The host
//!   picks the variant by position math — free.
//! - Position/depth scalars (start_pos, sp%win, block counts, gather t/n, topk nb/k) are
//!   baked as kernel-node params at capture; per replay they are rewritten with
//!   cuGraphExecKernelNodeSetParams (~220 nodes/token at ~1-2 µs each ≪ the ~12 ms gap pool
//!   at decode). NO kernel changes — bitwise by construction (same kernels, same args,
//!   same per-element order; only the launch vehicle changes).
//! - Classification needs no launch recorder: each kernel node's `func` (from
//!   cuGraphKernelNodeGetParams) names the kernel, and its BAKED capture-time arg values
//!   encode the layer context (e.g. gather slot6 == win+(T+1)/4 ⇒ CSA pair, == win+(T+1)/128
//!   ⇒ HCA pair, == win ⇒ SWA static). Every inference is VERIFIED against the capture-time
//!   formula; any mismatch → hard error → the caller falls back to eager (loud, never wrong).
//! - Depth-growth allocations are max-sized at capture (the --graph arm of the SEQ
//!   attention arm allocs idxs for win+index_topk / win+max_blocks) so NO re-capture is
//!   ever needed; t/n/k are pure param updates.
//! - cudarc allocations are stream-ordered (malloc_async) ⇒ capture-legal; the device
//!   default mempool's release threshold is raised once so graph pool memory is never
//!   returned to the OS. The ids upload + logits readout stay OUTSIDE the graph.
//! - Regime limit: the CSA indexer's hierarchical topk (nblocks > 16384, i.e. context
//!   > ~65K tokens) changes the kernel SEQUENCE with depth → those tokens run eager
//!   (recorded limitation; the 1M path is unaffected in its current eager form).
//!
//! CURRENT STATUS (2026-07-30, measured): BLOCKED on the persistent-workspace refactor.
//! The classifier/policies/integration/gate are built and compile, but capture fails at
//! the driver boundary: cudarc issues EVERY alloc/free/memset on the device LEGACY stream
//! (CudaDevice.stream is null, fixed at construction, no alternate constructor in 0.9.15).
//! Legacy-stream memory ops during capture are hard errors — malloc/memset invalidate it
//! (CUDA_ERROR_STREAM_CAPTURE_INVALIDATED at EndCapture) and joining legacy via an
//! event-fork is refused (CUDA_ERROR_STREAM_CAPTURE_IMPLICIT: "operation would make the
//! legacy stream depend on a capturing blocking stream"). The fix is the GSlice refactor:
//! a repo-local CudaSlice equivalent whose alloc/free/memset run on rt.stream (capture-
//! legal), swapped into the ~150 alloc_zeros sites of the decode forward — the multi-day
//! workspace item from the original audit. --graph stays OFF by default; the env-gated
//! arms are inert scaffolding until that lands.

use anyhow::{anyhow, Result};
use cudarc::driver::sys;
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceSlice};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;

use crate::dsv4_gpu::S;

/// Per-slot scalar expression, evaluated per replay with the token's start_pos.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Expr {
    StartPos,        // sp
    SpModWin,        // sp % win (win read from the node's baked slot-4 arg)
    TCompHca,        // (sp+1)/128
    NbCsa,           // (sp+1)/4
    KCsa,            // min(512, (sp+1)/4)
    WinPlusTCompHca, // win + (sp+1)/128
    WinPlusNbCsa,    // win + (sp+1)/4
    WinPlusKCsa,     // win + min(512, (sp+1)/4)
    FirePos(i64),    // sp+1-ratio
    FireIdx(i64),    // sp/ratio
    // Drafter (DSpark) expressions — the draft attention's index-list geometry:
    DraftTwin,   // min(win, sp+1) — window part of the 133-entry non-causal list
    DraftT(i64), // min(win, sp+1) + block — total idx length / gather t
    DraftN(i64), // win + block — gather n (constant; the Fixed check is verify-only)
}

impl Expr {
    fn eval(self, sp: i64, win: i64) -> i32 {
        let v = match self {
            Expr::StartPos => sp,
            Expr::SpModWin => sp % win,
            Expr::TCompHca => (sp + 1) / 128,
            Expr::NbCsa => (sp + 1) / 4,
            Expr::KCsa => ((sp + 1) / 4).min(512),
            Expr::WinPlusTCompHca => win + (sp + 1) / 128,
            Expr::WinPlusNbCsa => win + (sp + 1) / 4,
            Expr::WinPlusKCsa => win + ((sp + 1) / 4).min(512),
            Expr::FirePos(r) => sp + 1 - r,
            Expr::FireIdx(r) => sp / r,
            Expr::DraftTwin => (sp + 1).min(win),
            Expr::DraftT(b) => (sp + 1).min(win) + b,
            Expr::DraftN(b) => win + b,
        };
        v as i32
    }
}

/// A slot whose expression is fixed by kernel name, or inferred from the baked value.
#[derive(Clone, Copy, Debug)]
enum SlotExpr {
    Fixed(Expr),
    InferIota,    // dsv4_iota_b slot 1: StartPos | FirePos(4) | FirePos(128)
    InferFireIdx, // dsv4_comp_copy_rows_b slot 1: FireIdx(4) | FireIdx(128)
    InferGatherT, // gather slot 5: static (SWA) | WinPlusKCsa | WinPlusTCompHca
    InferGatherN, // gather slot 6: static (SWA) | WinPlusNbCsa | WinPlusTCompHca
    InferTopkNb,  // topk slot 3: static 256 (MoE router) | NbCsa (CSA indexer)
    InferTopkK,   // topk slot 4: static 8 (MoE router) | KCsa (CSA indexer)
}

#[derive(Clone, Copy, Debug)]
struct GridUpdate {
    y: bool, // false → gridDimX, true → gridDimY
    expr: Expr,
    unit: i64, // ceil divisor: (mul*expr + unit - 1) / unit, min 1
    mul: i64,  // row multiplier (the drafter's idxs grid is ceil(block*t/256))
}

struct Policy {
    slots: Vec<(usize, SlotExpr)>,
    grid: Option<GridUpdate>,
}

fn policy_for(name: &str, drafter: bool, draft_block: i64) -> Policy {
    use SlotExpr::Fixed as F;
    // Drafter table: the DSpark draft attention (SWA-only stages, NO compressor/indexer).
    // Position-dependent kernels: the main_kv ring write (sp), the non-causal draft idx
    // list (t_win = min(win, sp+1), t = t_win + block), and the batched gather's t/n.
    // rope_last_b reads positions from PERSISTENT device buffers (refreshed outside the
    // graph per step) — no scalar position args, no policy entries.
    if drafter {
        let b = draft_block;
        return match name {
            "dsv4_ring_write_b" => Policy { slots: vec![(3, F(Expr::StartPos))], grid: None },
            "dsv4_dspark_draft_idxs_b" => Policy {
                slots: vec![(2, F(Expr::DraftTwin)), (3, F(Expr::DraftT(b)))],
                grid: Some(GridUpdate { y: false, expr: Expr::DraftT(b), unit: 256, mul: b }),
            },
            "dsv4_gather_attn" => Policy {
                slots: vec![(5, F(Expr::DraftT(b))), (6, F(Expr::DraftN(b)))],
                grid: None,
            },
            _ => Policy { slots: Vec::new(), grid: None },
        };
    }
    match name {
        "dsv4_rope_pair_b" => Policy { slots: vec![(4, F(Expr::StartPos))], grid: None },
        "dsv4_rescale_rope_sim_b" => Policy { slots: vec![(4, F(Expr::StartPos))], grid: None },
        "dsv4_rope_q_inline_b" => Policy { slots: vec![(3, F(Expr::StartPos))], grid: None },
        "dsv4_ring_write_b" => Policy { slots: vec![(3, F(Expr::StartPos))], grid: None },
        "dsv4_window_idxs_b" => Policy { slots: vec![(2, F(Expr::StartPos))], grid: None },
        "dsv4_comp_decode_b" => Policy { slots: vec![(7, F(Expr::StartPos))], grid: None },
        "dsv4_compress_idxs_b" => Policy {
            slots: vec![(2, F(Expr::StartPos)), (5, F(Expr::TCompHca))],
            grid: Some(GridUpdate { y: false, expr: Expr::TCompHca, unit: 256, mul: 1 }),
        },
        "dsv4_gather_attn" => Policy {
            slots: vec![(5, SlotExpr::InferGatherT), (6, SlotExpr::InferGatherN)],
            grid: None,
        },
        "dsv4_idxs_place_b" => Policy {
            slots: vec![(3, F(Expr::KCsa)), (4, F(Expr::WinPlusKCsa))],
            grid: Some(GridUpdate { y: false, expr: Expr::KCsa, unit: 256, mul: 1 }),
        },
        "dsv4_iota_b" => Policy { slots: vec![(1, SlotExpr::InferIota)], grid: None },
        "dsv4_comp_copy_rows_b" => Policy { slots: vec![(1, SlotExpr::InferFireIdx)], grid: None },
        "dsv4_comp_index_score_b" => Policy {
            slots: vec![(4, F(Expr::NbCsa)), (5, F(Expr::StartPos))],
            grid: Some(GridUpdate { y: true, expr: Expr::NbCsa, unit: 1024, mul: 1 }),
        },
        "dsv4_comp_idx_remask_b" => Policy {
            slots: vec![(2, F(Expr::KCsa)), (3, F(Expr::StartPos)), (5, F(Expr::NbCsa))],
            grid: Some(GridUpdate { y: false, expr: Expr::KCsa, unit: 256, mul: 1 }),
        },
        "dsv4_topk" => Policy {
            slots: vec![(3, SlotExpr::InferTopkNb), (4, SlotExpr::InferTopkK)],
            grid: None,
        },
        _ => Policy { slots: Vec::new(), grid: None },
    }
}

struct NodeUpdate {
    node: sys::CUgraphNode,
    params: sys::CUDA_KERNEL_NODE_PARAMS,
    slots: Vec<(usize, Expr)>,
    grid: Option<GridUpdate>,
    win: i64,
}

pub struct Graph {
    exec: sys::CUgraphExec,
    graph: sys::CUgraph, // kept alive: node param storage lives in the graph object
    updates: Vec<NodeUpdate>,
    n_kernel_nodes: usize,
}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe {
            sys::cuGraphExecDestroy(self.exec);
            sys::cuGraphDestroy(self.graph);
        }
    }
}

pub enum Slot {
    Unborn,
    Ready(Graph),
    /// capture/classify failed once — eager forever (loud; never retry mid-session).
    Poisoned,
}

pub struct DecodeGraphs {
    pub v0: Slot,
    pub v4: Slot,
    pub v128: Slot,
    pub ids_dev: CudaSlice<i32>,
    /// Persistent eager buffer the graph's final memcpy node writes logits into
    /// (in-graph memory is freed at launch end under AUTO_FREE — outputs must land here).
    pub logits_out: CudaSlice<f32>,
    /// kernel func handle → name, for node classification (spine + attn + comp modules).
    func_names: HashMap<usize, &'static str>,
    win: i64,
    /// MoE router topk geometry (n_routed_experts, n_activated_experts) — the router's
    /// dsv4_topk nodes are STATIC with these values (vs the CSA indexer's depth-driven topk).
    router_nb: i32,
    router_k: i32,
    /// Drafter mode (DSpark draft graph): switches the policy table to the drafter entries
    /// (`policy_for`) and sizes `logits_out` at block*vocab. Only the V0 slot is used
    /// (SWA stages have no compressor fire variants); V4/V128 stay Poisoned.
    drafter: bool,
    draft_block: i64,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Variant {
    V0,
    V4,
    V128,
}

impl Variant {
    pub fn of(start_pos: usize) -> Variant {
        if (start_pos + 1) % 128 == 0 {
            Variant::V128
        } else if (start_pos + 1) % 4 == 0 {
            Variant::V4
        } else {
            Variant::V0
        }
    }
}

impl DecodeGraphs {
    pub fn new(
        dev: &Arc<CudaDevice>,
        func_names: HashMap<usize, &'static str>,
        win: usize,
        router_nb: usize,
        router_k: usize,
        vocab: usize,
    ) -> Result<Self> {
        let ids_dev = dev.alloc_zeros::<i32>(16).map_err(|e| anyhow!("graph ids alloc: {e:?}"))?;
        let logits_out = dev.alloc_zeros::<f32>(vocab).map_err(|e| anyhow!("graph logits_out alloc: {e:?}"))?;
        Ok(Self {
            v0: Slot::Unborn,
            v4: Slot::Unborn,
            v128: Slot::Unborn,
            ids_dev,
            logits_out,
            func_names,
            win: win as i64,
            router_nb: router_nb as i32,
            router_k: router_k as i32,
            drafter: false,
            draft_block: 0,
        })
    }

    /// Drafter (DSpark) variant: ONE graph (no fire variants), `ids_dev` sized to the draft
    /// block, `logits_out` [block, vocab] (the Markov tail reads all rows). V4/V128 are
    /// Poisoned so an accidental variant pick errors loudly.
    pub fn new_drafter(
        dev: &Arc<CudaDevice>,
        func_names: HashMap<usize, &'static str>,
        win: usize,
        draft_block: usize,
        vocab: usize,
    ) -> Result<Self> {
        let ids_dev = dev.alloc_zeros::<i32>(draft_block.max(16)).map_err(|e| anyhow!("drafter graph ids alloc: {e:?}"))?;
        let logits_out = dev.alloc_zeros::<f32>(draft_block * vocab).map_err(|e| anyhow!("drafter graph logits_out alloc: {e:?}"))?;
        Ok(Self {
            v0: Slot::Unborn,
            v4: Slot::Poisoned,
            v128: Slot::Poisoned,
            ids_dev,
            logits_out,
            func_names,
            win: win as i64,
            router_nb: 0,
            router_k: 0,
            drafter: true,
            draft_block: draft_block as i64,
        })
    }

    pub fn slot_mut(&mut self, v: Variant) -> &mut Slot {
        match v {
            Variant::V0 => &mut self.v0,
            Variant::V4 => &mut self.v4,
            Variant::V128 => &mut self.v128,
        }
    }

    pub fn slot_ref(&self, v: Variant) -> &Slot {
        match v {
            Variant::V0 => &self.v0,
            Variant::V4 => &self.v4,
            Variant::V128 => &self.v128,
        }
    }

    /// Read one i32 from a node's baked arg storage.
    unsafe fn baked_i32(params: &sys::CUDA_KERNEL_NODE_PARAMS, slot: usize) -> Result<i32> {
        let arr = params.kernelParams;
        if arr.is_null() {
            return Err(anyhow!("kernelParams null"));
        }
        let p = *arr.add(slot);
        if p.is_null() {
            return Err(anyhow!("arg slot {slot} null"));
        }
        Ok(*(p as *const i32))
    }

    /// Classify all kernel nodes of a freshly-captured graph: match func → policy, resolve
    /// inferences from baked values (each verified against its capture-time formula at `sp`).
    /// Returns the update list + total kernel-node count (for the report).
    pub fn classify(
        &self,
        graph: sys::CUgraph,
        sp: i64,
    ) -> Result<(Vec<NodeUpdate>, usize)> {
        let win = self.win;
        let mut n_nodes: usize = 0;
        let r = unsafe { sys::cuGraphGetNodes(graph, std::ptr::null_mut(), &mut n_nodes) };
        if r != sys::CUresult::CUDA_SUCCESS {
            return Err(anyhow!("cuGraphGetNodes(count): {r:?}"));
        }
        let mut nodes = vec![std::ptr::null_mut::<sys::CUgraphNode_st>(); n_nodes];
        let r = unsafe { sys::cuGraphGetNodes(graph, nodes.as_mut_ptr(), &mut n_nodes) };
        if r != sys::CUresult::CUDA_SUCCESS {
            return Err(anyhow!("cuGraphGetNodes: {r:?}"));
        }
        let mut updates = Vec::new();
        let mut n_kernel = 0usize;
        for &node in &nodes {
            let mut ty = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL;
            let r = unsafe { sys::cuGraphNodeGetType(node, &mut ty) };
            if r != sys::CUresult::CUDA_SUCCESS {
                return Err(anyhow!("cuGraphNodeGetType: {r:?}"));
            }
            if ty != sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
                continue; // memset/memcpy nodes: static sizes at decode, no updates
            }
            n_kernel += 1;
            let mut params: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
            let r = unsafe { sys::cuGraphKernelNodeGetParams(node, &mut params) };
            if r != sys::CUresult::CUDA_SUCCESS {
                return Err(anyhow!("cuGraphKernelNodeGetParams: {r:?}"));
            }
            let name = match self.func_names.get(&(params.func as usize)) {
                Some(n) => *n,
                None => continue, // unknown (bk/cudarc-safe kernels): static, no updates
            };
            let pol = policy_for(name, self.drafter, self.draft_block);
            if pol.slots.is_empty() && pol.grid.is_none() {
                continue;
            }
            let mut slots: Vec<(usize, Expr)> = Vec::with_capacity(pol.slots.len());
            for &(slot, se) in &pol.slots {
                let baked = unsafe { Self::baked_i32(&params, slot) }
                    .map_err(|e| anyhow!("{name} slot {slot}: {e}"))?;
                let expr = match se {
                    SlotExpr::Fixed(e) => {
                        let want = e.eval(sp, win);
                        if baked != want {
                            return Err(anyhow!(
                                "{name} slot {slot}: baked {baked} != capture-formula {want} (sp={sp})"
                            ));
                        }
                        e
                    }
                    SlotExpr::InferIota => {
                        if baked == sp as i32 {
                            Expr::StartPos
                        } else if baked == (sp + 1 - 4) as i32 {
                            Expr::FirePos(4)
                        } else if baked == (sp + 1 - 128) as i32 {
                            Expr::FirePos(128)
                        } else {
                            return Err(anyhow!("iota slot 1: baked {baked} matches no formula (sp={sp})"));
                        }
                    }
                    SlotExpr::InferFireIdx => {
                        if baked == (sp / 4) as i32 && (sp + 1) % 4 == 0 {
                            Expr::FireIdx(4)
                        } else if baked == (sp / 128) as i32 && (sp + 1) % 128 == 0 {
                            Expr::FireIdx(128)
                        } else {
                            return Err(anyhow!("copy_rows slot 1: baked {baked} matches no fire formula (sp={sp})"));
                        }
                    }
                    SlotExpr::InferGatherT => {
                        let kcsa = win + ((sp + 1) / 4).min(512);
                        let thca = win + (sp + 1) / 128;
                        if baked == win as i32 {
                            continue; // SWA: static, no update
                        } else if baked == thca as i32 {
                            Expr::WinPlusTCompHca
                        } else if baked == kcsa as i32 && thca != kcsa {
                            Expr::WinPlusKCsa
                        } else {
                            return Err(anyhow!(
                                "gather slot 5: baked {baked} matches no formula (sp={sp}, win={win})"
                            ));
                        }
                    }
                    SlotExpr::InferGatherN => {
                        let nb = win + (sp + 1) / 4;
                        let thca = win + (sp + 1) / 128;
                        if baked == win as i32 {
                            continue;
                        } else if baked == thca as i32 {
                            Expr::WinPlusTCompHca
                        } else if baked == nb as i32 && thca != nb {
                            Expr::WinPlusNbCsa
                        } else {
                            return Err(anyhow!(
                                "gather slot 6: baked {baked} matches no formula (sp={sp}, win={win})"
                            ));
                        }
                    }
                    SlotExpr::InferTopkNb => {
                        // MoE-router topk (256 experts, k=8 — static) vs CSA indexer topk
                        // (nb=(sp+1)/4, k=min(512,nb) — depth). Pair-checked both ways.
                        let other = unsafe { Self::baked_i32(&params, 4) }
                            .map_err(|e| anyhow!("topk slot 4: {e}"))?;
                        let nb = (sp + 1) / 4;
                        if baked == self.router_nb && other == self.router_k {
                            continue; // router: static, no update
                        } else if baked == nb as i32 {
                            Expr::NbCsa
                        } else {
                            return Err(anyhow!("topk slot 3: baked {baked} (slot4={other}) matches no formula (sp={sp})"));
                        }
                    }
                    SlotExpr::InferTopkK => {
                        let other = unsafe { Self::baked_i32(&params, 3) }
                            .map_err(|e| anyhow!("topk slot 3: {e}"))?;
                        let kcsa = ((sp + 1) / 4).min(512);
                        if baked == self.router_k && other == self.router_nb {
                            continue;
                        } else if baked == kcsa as i32 {
                            Expr::KCsa
                        } else {
                            return Err(anyhow!("topk slot 4: baked {baked} (slot3={other}) matches no formula (sp={sp})"));
                        }
                    }
                };
                slots.push((slot, expr));
            }
            if let Some(gu) = &pol.grid {
                let want = ((gu.mul * gu.expr.eval(sp, win) as i64 + gu.unit - 1) / gu.unit).max(1) as u32;
                let have = if gu.y { params.gridDimY } else { params.gridDimX };
                if have != want {
                    return Err(anyhow!("{name} grid: baked {have} != formula {want} (sp={sp})"));
                }
            }
            updates.push(NodeUpdate { node, params, slots, grid: pol.grid, win });
            if crate::opts::var(crate::opt!("graph-debug")).is_ok() && updates.len() <= 2 {
                eprintln!(
                    "[graph-dbg] node {} func={:?} kp={:?} extra={:?} grid=({},{},{}) smem={}",
                    updates.len(), params.func, params.kernelParams, params.extra,
                    params.gridDimX, params.gridDimY, params.gridDimZ, params.sharedMemBytes
                );
            }
        }
        Ok((updates, n_kernel))
    }

    /// Instantiate + classify a captured graph into a ready `Graph`.
    pub fn instantiate(
        &self,
        graph: sys::CUgraph,
        sp: usize,
    ) -> Result<Graph> {
        let mut exec: sys::CUgraphExec = std::ptr::null_mut();
        // Plain instantiate: the workspace pattern (GSlice bump slices of a persistent
        // slab) puts NO alloc nodes in the graph — nothing to re-execute or collide at
        // replay; kernel-arg pointers (slab addresses) are stable by construction.
        // (Measured 2026-07-30: capture-time driver allocs are unsound across launches —
        // alloc nodes re-execute (INVALID_VALUE at launch #2), and AUTO_FREE_ON_LAUNCH
        // re-allocs are not address-stable for multi-alloc graphs (garbage output).)
        let r = unsafe { sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0) };
        if r != sys::CUresult::CUDA_SUCCESS {
            return Err(anyhow!("cuGraphInstantiate: {r:?}"));
        }
        let (updates, n_kernel_nodes) = self.classify(graph, sp as i64)?;
        eprintln!("[dsv4-graph] instantiated: {n_kernel_nodes} kernel nodes, {} policy nodes", updates.len());
        Ok(Graph { exec, graph, updates, n_kernel_nodes })
    }
}

impl Graph {
    /// Explicit one-time upload (the driver's per-launch re-validation is ~35 ms host-side
    /// on this graph — cuGraphUpload moves that cost to setup; launches stay cheap).
    pub fn upload_once(&self, stream: &CudaStream) -> Result<()> {
        let r = unsafe { sys::cuGraphUpload(self.exec, stream.stream) };
        if r != sys::CUresult::CUDA_SUCCESS {
            return Err(anyhow!("cuGraphUpload(once): {r:?}"));
        }
        Ok(())
    }

    /// Node param updates + launch (steps 2–3 of [`replay`](Self::replay)), exposed so the
    /// DSpark drafter graph can drive the same machinery with its own input refresh (draft
    /// ids / main_hidden / position buffers) and multi-row logits readout.
    pub fn apply_updates_and_launch(&self, stream: &CudaStream, sp: usize) -> Result<()> {
        let dbg = crate::opts::var(crate::opt!("graph-debug")).is_ok();
        let skip_updates = crate::opts::var(crate::opt!("graph-noupdates")).is_ok();
        for (ui, u) in self.updates.iter().enumerate() {
            if skip_updates {
                break;
            }
            let mut params = u.params;
            for &(slot, expr) in &u.slots {
                let v = expr.eval(sp as i64, u.win);
                unsafe {
                    let p = *params.kernelParams.add(slot);
                    *(p as *mut i32) = v;
                }
            }
            if let Some(gu) = &u.grid {
                let g = ((gu.mul * gu.expr.eval(sp as i64, u.win) as i64 + gu.unit - 1) / gu.unit).max(1) as u32;
                if gu.y {
                    params.gridDimY = g;
                } else {
                    params.gridDimX = g;
                }
            }
            let r = unsafe { sys::cuGraphExecKernelNodeSetParams(self.exec, u.node, &params) };
            if r != sys::CUresult::CUDA_SUCCESS {
                return Err(anyhow!("cuGraphExecKernelNodeSetParams[{ui}]: {r:?}"));
            }
            if dbg {
                eprintln!("[graph-dbg] setparams[{ui}] ok grid=({},{},{})", params.gridDimX, params.gridDimY, params.gridDimZ);
            }
        }
        if dbg {
            eprintln!("[graph-dbg] launch exec={:?} stream={:?} sp={sp}", self.exec, stream.stream);
        }
        let t_upd = std::time::Instant::now();
        if crate::opts::var(crate::opt!("graph-upload")).is_ok() {
            // variant: explicit upload before launch (isolates lazy-upload failures)
            let r = unsafe { sys::cuGraphUpload(self.exec, stream.stream) };
            if r != sys::CUresult::CUDA_SUCCESS {
                return Err(anyhow!("cuGraphUpload: {r:?}"));
            }
        }
        let r = unsafe { sys::cuGraphLaunch(self.exec, stream.stream) };
        if r != sys::CUresult::CUDA_SUCCESS {
            return Err(anyhow!("cuGraphLaunch: {r:?}"));
        }
        if dbg {
            eprintln!("[graph-dbg] updates+launch host-side: {:.2} ms", t_upd.elapsed().as_secs_f64() * 1e3);
        }
        Ok(())
    }

    /// Replay for one token: upload id, patch nodes, launch, copy logits out (fresh alloc).
    /// `logits_out` is the persistent eager buffer the graph's final memcpy node wrote
    /// (in-graph memory is freed at launch end — never read it directly).
    pub fn replay(
        &self,
        dev: &Arc<CudaDevice>,
        stream: &CudaStream,
        ids_dev: &CudaSlice<i32>,
        logits_out: &CudaSlice<f32>,
        id: i32,
        sp: usize,
    ) -> Result<S> {
        // 1. next-token id → the persistent ids buffer (stream-ordered, outside the graph).
        unsafe {
            cudarc::driver::result::memcpy_htod_async(
                *ids_dev.device_ptr(),
                &[id],
                stream.stream,
            )
            .map_err(|e| anyhow!("graph ids htod: {e}"))?;
        }
        // 2+3. node param updates + launch.
        self.apply_updates_and_launch(stream, sp)?;
        // 4. fresh logits out (0.5 MB D2D from the persistent eager buffer the graph's
        //    final memcpy node wrote — replay runs OUTSIDE capture, so the legacy-stream
        //    cudarc alloc is legal here).
        let out = dev.alloc_zeros::<f32>(logits_out.len()).map_err(|e| anyhow!("graph logits alloc: {e:?}"))?;
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(
                *out.device_ptr(),
                *logits_out.device_ptr(),
                logits_out.len() * 4,
                stream.stream,
            )
            .map_err(|e| anyhow!("graph logits dtod: {e}"))?;
        }
        Ok(out)
    }
}

/// One-time setup: raise the default mempool release threshold so graph pool memory is
/// never returned to the OS (the documented requirement for graphs owning memory nodes).
pub fn raise_mempool_threshold(dev: &Arc<CudaDevice>) -> Result<()> {
    static ONCE: std::sync::Once = std::sync::Once::new();
    let mut err: Option<anyhow::Error> = None;
    ONCE.call_once(|| {
        unsafe {
            let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
            let r = sys::cuDeviceGetDefaultMemPool(&mut pool, *dev.cu_device());
            if r != sys::CUresult::CUDA_SUCCESS {
                err = Some(anyhow!("cuDeviceGetDefaultMemPool: {r:?}"));
                return;
            }
            let threshold: u64 = u64::MAX;
            let r = sys::cuMemPoolSetAttribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                &threshold as *const u64 as *mut c_void,
            );
            if r != sys::CUresult::CUDA_SUCCESS {
                err = Some(anyhow!("cuMemPoolSetAttribute: {r:?}"));
            }
        }
    });
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}
