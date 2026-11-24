//! TPPOOL (design v3 step 4): the TP head's capacity gate for the shared KV pool.
//!
//! Under TP an admit whose prompt finds the pool exhausted cannot wait the way the single box does: the apply
//! (`admit_req`) runs on every rank, and an exhausted `vmm_room` there is a fatal on all of them. So the head asks
//! BEFORE it ships the Admit: how many pages does this prompt need on the lane the apply will pick, and are they free?
//! The answer comes from dry runs over the very page tables and unit math the apply uses (`FwdModel::vmm_pages_needed`,
//! `vmm_pages_freed`), never from a second formula. The node never runs the gate: holding a request is head-only
//! state (the waiting deque), and a refused request never ships.
//!
//! Dormant today: on a TP rank `kv_vmm::mode_from_opts` falls back to the stock allocation (one notice line at bind),
//! and the gate runs only with a lazy pool under TP. Contributed by @AbsolomBane (veloGB10 #43).

use super::*;

/// What the head does with one waiting request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum GateDecision {
    /// Ship the Admit; `pages` = what it was charged.
    Ship { pages: usize },
    /// It does not fit yet, but a running lane will free room: keep it at the head of the queue.
    Hold,
    /// It cannot fit and nothing running will make room: answer it with this error, ship nothing.
    Refuse(String),
}

/// Under TP the idle-lane eviction is still single-box only (`vmm_evict_idle` returns early when `self.tp` is set),
/// so the gate must not count pages an eviction would free: none would be freed. Turn this on in the same change
/// that lets the eviction run under TP.
pub(super) const TP_EVICTS: bool = false;

/// The decision, pure: `need` pages for the prompt; `free` pages unmapped now; `reserved` pages already charged to
/// Admits this head shipped but no rank has applied yet; `victims` = pages each idle victim would free, in the
/// eviction's order (`vmm_idle_victims`); `evicts` = whether an eviction would really run; `running` = whether a
/// lane will still decode after this step's cancels.
/// - A victim may free ZERO pages (all its pages straddle a boundary): only real per-victim counts are credited.
/// - Victims are credited one at a time, only as many as the shortfall needs (the eviction takes them one by one).
pub(super) fn gate_decide(need: usize, free: usize, reserved: usize, victims: &[usize], evicts: bool, running: bool)
                          -> GateDecision {
    let mut have = free.saturating_sub(reserved);
    if have >= need { return GateDecision::Ship { pages: need }; }
    if evicts {
        for &v in victims {
            have += v;
            if have >= need { return GateDecision::Ship { pages: need }; }
        }
    }
    if running { return GateDecision::Hold; }
    GateDecision::Refuse(format!(
        "error: the shared KV pool cannot hold this prompt ({need} pages of 2 MiB needed, {have} free and nothing running to free more)"))
}

impl Exl3Scheduler {
    /// Rows a prompt of `plen` tokens maps on its lane at apply: the prompt itself (`vmm_room(slot, plen)`), or a
    /// restore of a snapshot of at most `plen` tokens (`spill_restore` / the prefix pool map `pool_rows_for(tokens)`,
    /// which rounds to the pooled block and can cross the next grid line: plen 11264 maps 12288 rows directly but
    /// 16384 on a restore). The gate charges the larger: a restore is decided at apply, after the Admit ships.
    pub(super) fn gate_rows(&self, plen: usize) -> usize {
        plen.max(self.model.pool_rows_for(plen))
    }

    /// True when the gate decides admissions: a lazy pool under TP. The head then ships ONE Admit per iteration (as
    /// LANERES3 does for the reservation policy): with `--prefill-interleave` the first admit's prefill runs nested
    /// decode steps that map pages before a second admit of the same iteration would map its own, so a second verdict
    /// would be stale. One admit per iteration keeps every verdict exact for the state its apply finds.
    pub(super) fn tp_pool_gated(&self) -> bool { self.tp.is_some() && self.model.vmm_lazy() }

    /// The head's gate for one request (see the module doc). `pending_free`: lanes a Cancel in this step frees before
    /// the admit applies; `reserved`: pages charged to Admits shipped earlier in this step. Runs only with a lazy pool
    /// under TP; everything else ships as today.
    pub(super) fn tp_pool_gate(&self, req: &BatchRequest, scoped: Option<&[usize]>,
                               pending_free: &[usize], reserved: usize) -> GateDecision {
        if self.tp.is_none() || !self.model.vmm_lazy() { return GateDecision::Ship { pages: 0 }; }
        let Some(free) = self.model.vmm_free_pages() else { return GateDecision::Ship { pages: 0 } };
        // no lane to pick: the apply answers "server busy" exactly as today, nothing is mapped
        let Some(lane) = self.pick_lane(req, scoped, pending_free) else { return GateDecision::Ship { pages: 0 } };
        let need = match self.model.vmm_pages_needed(lane, self.gate_rows(req.prompt.len())) {
            Ok(n) => n,
            Err(e) => return GateDecision::Refuse(format!("error: kv pool gate failed: {e:#}")),
        };
        let victims: Vec<usize> = if TP_EVICTS {
            self.vmm_idle_victims(&[lane]).iter()
                .filter(|&&(_, v)| !pending_free.contains(&v))
                .map(|&(_, v)| self.model.vmm_pages_freed(v, 0).unwrap_or(0)).collect()
        } else { Vec::new() };
        let running = self.lanes.iter().enumerate().any(|(s, l)| l.is_some() && !pending_free.contains(&s));
        gate_decide(need, free, reserved, &victims, TP_EVICTS, running)
    }
}
