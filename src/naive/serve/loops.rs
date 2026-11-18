//! The serving loops: the head's one-at-a-time loop (`serve_one`), the lane loop (`serve_lanes`) and the node's
//! lockstep loop (`node_lanes`). Head-authoritative sampling: the node never diverges on a tie or RNG detail.
//! Siblings: `head.rs` brings the cluster up around these; `engine.rs` has what they call; `mod.rs` has the map.

use super::*;

// ---------------------------------------------------------------------------------------------
// Sampling (CPU, the head): the OpenAI request knobs over the f32 logits row.
// ---------------------------------------------------------------------------------------------

pub(super) fn argmax(l: &[f32]) -> usize { l.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0 }

fn seed_of(req: &BatchRequest) -> u64 {
    match req.seed {
        Some(s) => s,
        // no client seed: a fresh random one (RandomState is seeded from the OS); the node receives it in Admit
        None => { use std::hash::{BuildHasher, Hasher}; std::collections::hash_map::RandomState::new().build_hasher().finish() }
    }
}

// ---------------------------------------------------------------------------------------------
// The head
// ---------------------------------------------------------------------------------------------

/// Run one request on the head: ship Admit, prefill, then sample / ship / stream token by token.
pub(super) fn serve_one(eng: &mut Engine, req: BatchRequest, stop_ids: &[u32], nodes: &mut [std::net::TcpStream]) -> Result<()> {
    crate::think_budget::note_unsupported(req.thinking_budget, "Naive"); // THINKBUDGET1
    let t0 = std::time::Instant::now();
    let prompt = req.prompt.clone();
    let room = eng.max_pos.saturating_sub(prompt.len());
    if prompt.is_empty() || room == 0 {
        let _ = req.tx.send(TokEvent::Finish { reason: "length".into() });
        bail!("prompt of {} tokens leaves no room in the context ({})", prompt.len(), eng.max_pos);
    }
    let max_new = req.max_new.min(room);
    let seed = seed_of(&req);
    let (reuse, snap) = eng.prefix_plan(0, &prompt, req.ckpt_at, true);
    for n in nodes.iter_mut() { crate::cluster::send_json(n, &Ctl::Admit { prompt: prompt.clone(), max_new, seed, sp: sample_params(&req), min_new: req.min_new, ignore_eos: req.ignore_eos, reuse, snap })?; }
    let _ = req.tx.send(TokEvent::Admitted { cached_tokens: reuse as u32 });
    eng.begin_request(&sample_params(&req), seed, &prompt)?;
    let row = eng.prefill_resume(0, &prompt, reuse, snap, true)?.context("head prefill without logits")?;
    let (mut t, mut cancel) = eng.dec.sample_row(&eng.model, row)?;
    let ttft = t0.elapsed();
    let mut generated: Vec<u32> = Vec::new();
    let mut pos = prompt.len();
    let mut reason = "length";
    let td = std::time::Instant::now();
    let mut step_secs = 0f64;
    if eng.ver.is_some() {
        // S8: speculative rounds. `t` is the first sampled token (position `pos`, not yet in the caches).
        let mut rounds = 0usize; let mut accepted = 0usize;
        // the first sampled token (the first anchor) is emitted once, here; every round then emits only its
        // committed tokens (the bonus becomes the next anchor and is never emitted twice)
        let mut stop0 = false;
        {
            let is_stop = stop_ids.contains(&t) && !req.ignore_eos && generated.len() + 1 > req.min_new;
            if is_stop { reason = "stop"; stop0 = true; }
            else {
                // a dead socket is NOT a stop on this rank alone (the node cannot see it): record the reason,
                // keep the bookkeeping identical, and let the exchanged cancel flag stop every rank together
                if req.tx.send(TokEvent::Tok(t)).is_err() { reason = "cancelled"; }
                generated.push(t); if generated.len() >= max_new { stop0 = true; }
            }
            for n in nodes.iter_mut() { crate::cluster::send_json(n, &Ctl::Round { toks: vec![t], stop: stop0 })?; }
            if !stop0 { eng.note_commit_all(&[t], false)?; }
        }
        let mut guard = SpecGuard::with_every(eng.guard.0, eng.guard.1, eng.guard.2);
        let mut plain_steps = 0usize;
        let mut width_sum = 0usize;
        // confidence calibration: per 0.2-wide confidence bucket, (verified, accepted) and (skipped) counts,
        // printed on the request line so `--spec-conf` can be set from data
        let (mut hv, mut ha, mut hs) = ([0usize; 5], [0usize; 5], [0usize; 5]);
        'rounds: while !stop0 {
            let ts = std::time::Instant::now();
            let mut from_plain = false;
            let (committed, a, c) = if guard.speculate() {
                let (drafts, conf) = eng.draft(t, pos)?;
                let k = eng.pick_width(&conf);
                let r = eng.spec_round(Some((&drafts, k)), t, pos)?;
                guard.observe(r.1);
                rounds += 1; accepted += r.1; width_sum += k;
                for (i, &c) in conf.iter().enumerate().take(eng.block - 1) {
                    let b = ((c.clamp(0.0, 0.999)) * 5.0) as usize;
                    if i < k { hv[b] += 1; if i < r.1 { ha[b] += 1; } } else { hs[b] += 1; }
                }
                if k == 0 { from_plain = true; }
                r
            } else {
                // a plain step (the guard says speculation does not pay right now); same bookkeeping shape
                let (nt, c) = eng.plain_step_spec(t, pos)?;
                plain_steps += 1; from_plain = true;
                (vec![nt], 0, c)
            };
            step_secs += ts.elapsed().as_secs_f64();
            let toks_out: Vec<u32> = committed.clone();
            let mut stop = false;
            let mut emitted = Vec::new();
            for &tok in &toks_out {
                let is_stop = stop_ids.contains(&tok) && !req.ignore_eos && generated.len() + 1 > req.min_new;
                if is_stop { reason = "stop"; stop = true; break; }
                // same rule as the plain loop: a dead socket only sets the reason; the stop that ships to the
                // nodes must be the pure function every rank computes (2026-10-18 TP=4 cancel abort:
                // `node [..]/false, head [..]/true` -- identical tokens, the head stopped one round early)
                if req.tx.send(TokEvent::Tok(tok)).is_err() { reason = "cancelled"; }
                generated.push(tok); emitted.push(tok);
                if generated.len() >= max_new { stop = true; break; }
            }
            if c || cancel { stop = true; }
            for n in nodes.iter_mut() { crate::cluster::send_json(n, &Ctl::Round { toks: toks_out.clone(), stop })?; }
            eng.note_commit_all(&emitted, from_plain)?;
            if stop { break 'rounds; }
            // the next anchor: the bonus sample (last committed), at position pos + a + 1
            pos += a + 1;
            t = *committed.last().unwrap();
            cancel = c;
            eng.set_cancel_all(req.tx.is_closed())?;
            if pos + eng.block > eng.max_pos { reason = "length"; break 'rounds; }
        }
        let _ = req.tx.send(TokEvent::Finish { reason: reason.into() });
        let dt = td.elapsed().as_secs_f64();
        println!("[naive-serve] req: prompt {} tok ({reuse} reused), ttft {:.2} s ({:.0} tok/s prefill), {} generated in {dt:.2} s ({:.1} tok/s; {rounds} spec rounds ({:.2} accepted/round, {:.2} verified/round of {}) + {plain_steps} plain steps, {:.1} ms per round/step, draft {:.1} ms/round {}, sampler {}; sampled {}, residual fallbacks {} (no list {})), finish {reason}",
                 prompt.len(), ttft.as_secs_f64(), (prompt.len() - reuse) as f64 / ttft.as_secs_f64().max(1e-9), generated.len(),
                 generated.len() as f64 / dt.max(1e-9), accepted as f64 / rounds.max(1) as f64, width_sum as f64 / rounds.max(1) as f64, eng.block - 1,
                 step_secs / (rounds + plain_steps).max(1) as f64 * 1e3, eng.draft_secs / rounds.max(1) as f64 * 1e3, if eng.sharded { "sharded" } else { "head-only" }, if eng.batched { "batched" } else { "rows" }, eng.sp_cur.temperature > 0.0, eng.smp_fallbacks, eng.smp_nolist);
        println!("[naive-serve] conf: verified(acc/n) by bucket .0 .2 .4 .6 .8: {} | skipped: {:?}",
                 (0..5).map(|b| format!("{}/{}", ha[b], hv[b])).collect::<Vec<_>>().join(" "), hs);
        return Ok(());
    }
    for i in 0..max_new {
        // the stop decision is a pure function of (token, i, pos, the exchanged cancel flag): every rank
        // computes the same one; the head's Tok message is a lagging lockstep check on the node
        let is_stop = stop_ids.contains(&t) && !req.ignore_eos && generated.len() + 1 > req.min_new;
        let last = is_stop || i + 1 == max_new || pos + 1 >= eng.max_pos || cancel;
        for n in nodes.iter_mut() { crate::cluster::send_json(n, &Ctl::Tok { t, stop: last })?; }
        if is_stop { reason = "stop"; break; }
        if req.tx.send(TokEvent::Tok(t)).is_err() { reason = "cancelled"; }
        else { generated.push(t); }
        if last { break; }
        // the client's departure is seen by the NEXT step's exchange (one step late, on every rank)
        eng.dec.set_cancel(&eng.model, req.tx.is_closed())?;
        let ts = std::time::Instant::now();
        (t, cancel) = eng.decode_step(t, pos)?;
        step_secs += ts.elapsed().as_secs_f64();
        pos += 1;
    }
    let _ = req.tx.send(TokEvent::Finish { reason: reason.into() });
    let dt = td.elapsed().as_secs_f64();
    println!("[naive-serve] req: prompt {} tok ({reuse} reused), ttft {:.2} s ({:.0} tok/s prefill), {} generated in {dt:.2} s ({:.1} tok/s; step {:.1} ms of {:.1} ms per token), finish {reason}",
             prompt.len(), ttft.as_secs_f64(), (prompt.len() - reuse) as f64 / ttft.as_secs_f64().max(1e-9), generated.len(),
             generated.len() as f64 / dt.max(1e-9), step_secs / generated.len().max(1) as f64 * 1e3, dt / generated.len().max(1) as f64 * 1e3);
    Ok(())
}

/// G1: one active request on the head.
struct HLane { req: BatchRequest, slot: usize, pos: usize, t: u32, gen: usize, max_new: usize, owner: bool, t0: std::time::Instant,
               ttft: f64, prompt_len: usize, steps: usize, rounds: usize, accepted: usize, reason: &'static str, last: Vec<u32>,
               spec: LaneSpec, guard: SpecGuard }

/// G1: a lane's own speculation state (admission order id, sampler knobs, seed, its accept-draw round counter); loaded
/// into the engine when the lane speculates alone. Every rank keeps the same, in the same order.
#[derive(Clone, Copy)]
struct LaneSpec { id: u64, sp: SampleParams, seed: u64, smp_round: u64 }

/// G1: lane 0 is about to run a speculative round or a drafter-fed plain step alone: its context into the drafter, its
/// speculation state into the engine (only when another request held it).
fn lane_spec_load(eng: &mut Engine, ls: &LaneSpec) -> Result<()> {
    eng.ctx_make_live(0)?;
    if eng.spec_id != ls.id {
        eng.sp_cur = ls.sp; eng.seed_cur = ls.seed; eng.smp_round = ls.smp_round;
        eng.smp_fallbacks = 0; eng.smp_nolist = 0; eng.draft_secs = 0.0; eng.node_drafts = None;
        eng.spec_id = ls.id;
    }
    Ok(())
}

/// G1 (`--naive-lanes N`): the head's multi-request loop. Up to N requests are active at once; between steps the head
/// admits waiting requests (prefill into a free KV slot, first token), and every step serves every active request:
/// one row each in a multi-request step, or — when ONE request is active and it owns the drafter context — the
/// speculative round of the single-stream path. The nodes are pure followers (`node_lanes`): the head ships every
/// decision (finishes, admissions, the step kind) before the step, so no rank decides anything on its own.
pub(super) fn serve_lanes(eng: &mut Engine, srx: &mut tokio::sync::mpsc::UnboundedReceiver<BatchRequest>, stop_ids: &[u32],
               nodes: &mut [std::net::TcpStream]) -> Result<bool> {
    let n_l = eng.lanes;
    let mut act: Vec<HLane> = Vec::new();
    let mut free: Vec<usize> = (0..n_l).rev().collect();
    let mut fin: Vec<usize> = Vec::new();
    let mut next_id = 0u64;
    let send_all = |nodes: &mut [std::net::TcpStream], c: &Ctl| -> Result<()> { for n in nodes.iter_mut() { crate::cluster::send_json(n, c)?; } Ok(()) };
    let finish = |l: HLane| {
        let _ = l.req.tx.send(TokEvent::Finish { reason: l.reason.into() });
        let dt = l.t0.elapsed().as_secs_f64() - l.ttft;
        println!("[naive-serve] req (lane): prompt {} tok, ttft {:.2} s, {} generated in {dt:.2} s ({:.1} tok/s; {} steps, {} spec rounds ({:.2} accepted/round)), finish {}",
                 l.prompt_len, l.ttft, l.gen, l.gen as f64 / dt.max(1e-9), l.steps, l.rounds, l.accepted as f64 / l.rounds.max(1) as f64, l.reason);
    };
    loop {
        // 1. the finishes of the previous step, descending (swap-remove compaction; the sampler state follows)
        fin.sort_unstable(); fin.dedup(); fin.reverse();
        let fin_now = fin.clone();
        for &j in &fin_now {
            let last = act.len() - 1;
            if j != last { eng.copy_lane(last, j)?; }
            let l = act.swap_remove(j);
            free.push(l.slot);
            finish(l);
        }
        fin.clear();
        // 2. admissions (blocking only when idle)
        let mut reqs: Vec<BatchRequest> = Vec::new();
        while act.len() + reqs.len() < n_l {
            let r = if act.is_empty() && reqs.is_empty() {
                match srx.blocking_recv() { Some(r) => r, None => { send_all(nodes, &Ctl::LIter { fin: fin_now.clone(), admits: vec![] })?; return Ok(false); } }
            } else { match srx.try_recv() { Ok(r) => r, Err(_) => break } };
            reqs.push(r);
        }
        let mut admits = Vec::new();
        let mut new_lanes = Vec::new();
        for r in reqs {
            let prompt = r.prompt.clone();
            let room = eng.max_pos.saturating_sub(prompt.len());
            if prompt.is_empty() || room < 2 {
                let _ = r.tx.send(TokEvent::Finish { reason: "length".into() });
                println!("[naive-serve] req refused before admission: prompt {} tokens leaves no room in the context ({})", prompt.len(), eng.max_pos);
                continue;
            }
            let lane = act.len() + new_lanes.len();
            let owner = act.is_empty() && new_lanes.is_empty() && eng.ver.is_some();
            ensure!(!free.is_empty(), "no free KV slot");
            // the free slot whose prefix snapshot this prompt resumes furthest into (prefix reuse per slot)
            let (fi, reuse, snap) = eng.prefix_pick(&free, &prompt, r.ckpt_at, owner || eng.per_lane_ctx());
            let slot = free.remove(fi);
            let max_new = r.max_new.min(room - 1);
            let seed = seed_of(&r);
            crate::think_budget::note_unsupported(r.thinking_budget, "Naive"); // THINKBUDGET1
            admits.push(LAdmit { lane, slot, prompt: prompt.clone(), max_new, seed, sp: sample_params(&r), owner, reuse, snap });
            new_lanes.push((r, slot, max_new, owner, prompt.len()));
        }
        send_all(nodes, &Ctl::LIter { fin: fin_now, admits: admits.clone() })?;
        // 3. prefill every admitted request into its slot, sample its first token
        let mut drop: Vec<usize> = Vec::new();
        for (a, (r, slot, max_new, owner, plen)) in admits.iter().zip(new_lanes) {
            let t0 = std::time::Instant::now();
            let _ = r.tx.send(TokEvent::Admitted { cached_tokens: a.reuse as u32 });
            eng.dec.begin_request_lane(&eng.model, &a.sp, a.seed, &a.prompt, a.lane)?;
            eng.set_slot_all(slot)?;
            // per-lane contexts: every request's prompt feeds ITS context (else only the owner's feeds the one context)
            if eng.per_lane_ctx() { eng.ctx_make_live(a.lane)?; }
            let row = eng.prefill_resume(slot, &a.prompt, a.reuse, a.snap, owner || eng.per_lane_ctx())?.context("prefill without logits")?;
            let (t, _) = eng.dec.sample_row_lane(&eng.model, row, a.lane)?;
            let spec = LaneSpec { id: next_id, sp: a.sp, seed: a.seed, smp_round: 0 };
            next_id += 1;
            let mut l = HLane { req: r, slot, pos: plen, t, gen: 0, max_new, owner, t0, ttft: t0.elapsed().as_secs_f64(), prompt_len: plen,
                                steps: 0, rounds: 0, accepted: 0, reason: "length", last: vec![t],
                                spec, guard: SpecGuard::with_every(eng.guard.0, eng.guard.1, eng.guard.2) };
            let is_stop = stop_ids.contains(&t) && !l.req.ignore_eos && l.req.min_new == 0;
            if is_stop { l.reason = "stop"; drop.push(a.lane); }
            else {
                if l.req.tx.send(TokEvent::Tok(t)).is_err() { l.reason = "cancelled"; drop.push(a.lane); }
                l.gen = 1;
                if l.gen >= l.max_new && !drop.contains(&a.lane) { drop.push(a.lane); }
            }
            act.push(l);
        }
        drop.sort_unstable(); drop.reverse();
        for &j in &drop {
            let last = act.len() - 1;
            if j != last { eng.copy_lane(last, j)?; }
            let l = act.swap_remove(j);
            free.push(l.slot);
            finish(l);
        }
        if act.is_empty() { send_all(nodes, &Ctl::LStep { drop, kind: 255, check: vec![] })?; continue; }
        // 4. the step
        // a request alone speculates: any lane with per-lane contexts, only the owner of the one shared context without
        let solo_spec = act.len() == 1 && (act[0].owner || eng.per_lane_ctx()) && eng.ver.is_some();
        let kind: u8 = if solo_spec { if act[0].guard.speculate() { 1 } else { 2 } } else { 0 };
        let check: Vec<Vec<u32>> = act.iter().map(|l| l.last.clone()).collect();
        send_all(nodes, &Ctl::LStep { drop, kind, check })?;
        let outs: Vec<(Vec<u32>, usize)> = match kind {
            1 => {
                let (t, pos, slot) = (act[0].t, act[0].pos, act[0].slot);
                eng.set_slot_all(slot)?;
                lane_spec_load(eng, &act[0].spec)?;
                let (drafts, conf) = eng.draft(t, pos)?;
                let k = eng.pick_width(&conf);
                let (committed, a, _) = eng.spec_round(Some((&drafts, k)), t, pos)?;
                act[0].guard.observe(a);
                act[0].spec.smp_round = eng.smp_round;
                act[0].rounds += 1; act[0].accepted += a;
                if !eng.last_round_plain() { eng.dec.note_commit_lane(&eng.model, &committed, 0)?; }
                vec![(committed, a)]
            }
            2 => {
                let (t, pos, slot) = (act[0].t, act[0].pos, act[0].slot);
                eng.set_slot_all(slot)?;
                lane_spec_load(eng, &act[0].spec)?;
                let (nt, _) = eng.plain_step_spec(t, pos)?;
                vec![(vec![nt], 0)]
            }
            _ => {
                let ids: Vec<u32> = act.iter().map(|l| l.t).collect();
                let pos: Vec<usize> = act.iter().map(|l| l.pos).collect();
                let slots: Vec<usize> = act.iter().map(|l| l.slot).collect();
                let owner = act.iter().position(|l| l.owner);
                let toks = eng.step_lanes(&ids, &pos, &slots, owner)?;
                toks.into_iter().map(|t| (vec![t], 0)).collect()
            }
        };
        // 5. deliver, stop rules (the head decides; the nodes learn the finishes at the next iteration)
        for (j, (toks, a)) in outs.into_iter().enumerate() {
            let l = &mut act[j];
            l.steps += 1;
            let mut done = false;
            for &tok in &toks {
                let is_stop = stop_ids.contains(&tok) && !l.req.ignore_eos && l.gen + 1 > l.req.min_new;
                if is_stop { l.reason = "stop"; done = true; break; }
                if l.req.tx.send(TokEvent::Tok(tok)).is_err() { l.reason = "cancelled"; done = true; break; }
                l.gen += 1;
                if l.gen >= l.max_new { l.reason = "length"; done = true; break; }
            }
            l.pos += a + 1;
            l.t = *toks.last().unwrap();
            l.last = toks;
            if l.req.tx.is_closed() { l.reason = "cancelled"; done = true; }
            if l.pos + eng.block + 1 > eng.max_pos { l.reason = "length"; done = true; }
            if done { fin.push(j); }
        }
    }
}

/// G1: the node's follower of `serve_lanes`: the same admissions, compactions and steps, in the head's order.
pub(super) fn node_lanes(eng: &mut Engine, stream: &mut std::net::TcpStream, first: Ctl) -> Result<()> {
    struct NLane { slot: usize, pos: usize, t: u32, owner: bool, last: Vec<u32>, spec: LaneSpec }
    let mut act: Vec<NLane> = Vec::new();
    let mut next_id = 0u64;
    let mut guard_obs: Vec<usize> = Vec::new(); let _ = &mut guard_obs;
    let mut msg = Some(first);
    loop {
        let m = match msg.take() { Some(m) => m, None => crate::cluster::recv_json::<Ctl>(stream).context("control stream (head gone?)")? };
        let (fin, admits) = match m { Ctl::LIter { fin, admits } => (fin, admits), Ctl::Shutdown => return Ok(()), other => bail!("lanes: unexpected {other:?}") };
        for &j in &fin { let last = act.len() - 1; if j != last { eng.copy_lane(last, j)?; } act.swap_remove(j); }
        if admits.is_empty() && act.is_empty() && !fin.is_empty() {
            // the head's request channel closed after these finishes (its last LIter); keep following
        }
        for a in &admits {
            ensure!(a.lane == act.len(), "lanes: admitted lane {} but {} active", a.lane, act.len());
            eng.dec.begin_request_lane(&eng.model, &a.sp, a.seed, &a.prompt, a.lane)?;
            eng.set_slot_all(a.slot)?;
            if eng.per_lane_ctx() { eng.ctx_make_live(a.lane)?; }
            let row = eng.prefill_resume(a.slot, &a.prompt, a.reuse, a.snap, a.owner || eng.per_lane_ctx())?.context("node prefill without logits")?;
            let (t, _) = eng.dec.sample_row_lane(&eng.model, row, a.lane)?;
            let spec = LaneSpec { id: next_id, sp: a.sp, seed: a.seed, smp_round: 0 };
            next_id += 1;
            act.push(NLane { slot: a.slot, pos: a.prompt.len(), t, owner: a.owner, last: vec![t], spec });
        }
        let (drop, kind, check) = match crate::cluster::recv_json::<Ctl>(stream)? {
            Ctl::LStep { drop, kind, check } => (drop, kind, check),
            Ctl::Shutdown => return Ok(()),
            other => bail!("lanes: unexpected {other:?} before a step"),
        };
        for &j in &drop { let last = act.len() - 1; if j != last { eng.copy_lane(last, j)?; } act.swap_remove(j); }
        if kind == 255 { continue; }
        ensure!(check.len() == act.len(), "NAIVE TP LOCKSTEP BROKEN (lanes): head {} lanes, node {}", check.len(), act.len());
        for (j, c) in check.iter().enumerate() {
            ensure!(*c == act[j].last, "NAIVE TP LOCKSTEP BROKEN (lane {j}): node {:?}, head {c:?}", act[j].last);
        }
        let outs: Vec<(Vec<u32>, usize)> = match kind {
            1 => {
                let (t, pos, slot) = (act[0].t, act[0].pos, act[0].slot);
                eng.set_slot_all(slot)?;
                lane_spec_load(eng, &act[0].spec)?;
                if eng.sharded { let (d, cf) = eng.draft(t, pos)?; let k = eng.pick_width(&cf); eng.node_drafts = Some((d, k)); }
                let (committed, a, _) = eng.spec_round(None, t, pos)?;
                act[0].spec.smp_round = eng.smp_round;
                if !eng.last_round_plain() { eng.dec.note_commit_lane(&eng.model, &committed, 0)?; }
                vec![(committed, a)]
            }
            2 => {
                let (t, pos, slot) = (act[0].t, act[0].pos, act[0].slot);
                eng.set_slot_all(slot)?;
                lane_spec_load(eng, &act[0].spec)?;
                let (nt, _) = eng.plain_step_spec(t, pos)?;
                vec![(vec![nt], 0)]
            }
            _ => {
                let ids: Vec<u32> = act.iter().map(|l| l.t).collect();
                let pos: Vec<usize> = act.iter().map(|l| l.pos).collect();
                let slots: Vec<usize> = act.iter().map(|l| l.slot).collect();
                let owner = act.iter().position(|l| l.owner);
                eng.step_lanes(&ids, &pos, &slots, owner)?.into_iter().map(|t| (vec![t], 0)).collect()
            }
        };
        for (j, (toks, a)) in outs.into_iter().enumerate() {
            act[j].pos += a + 1;
            act[j].t = *toks.last().unwrap();
            act[j].last = toks;
        }
    }
}

