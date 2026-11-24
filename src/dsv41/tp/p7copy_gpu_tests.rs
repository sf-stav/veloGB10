//! Leg P7COPY GPU verification & slice throughput benchmark (`#[ignore = "GPU"]`, run on `.15` under
//! `flock /tmp/dsv41_gpu.lock`):
//! Loads the 10-layer EXL3 slice (`L0,1,2,14,20,21,24` + DSpark `L40,41,42` + globals) into the REAL `TpModel` +
//! `TpSpec` + `TpServe` + `TpForward` + `engine` on CUDA and verifies:
//!   1. `WIDEVERIFY1` (`16`-row / `15`-copy) greedy bit-identity (tokens + trunk SWA rings + KV main + index K +
//!      7-slot ratio-2 frontiers + DSpark rings + Engram hash + next-step `TpOut` bits) vs `1`-row `--mtp off`
//!      decode across full `15`-token accepts AND partial-accept `WideCopyGuard` O(1) rollbacks at
//!      `accepted = 0, 1, 5, 7, 11, 14` on both even (`150`) and odd (`151`) SWA-wrapped positions;
//!   2. Sampled (`temp = 0.8`) `WIDEVERIFY1` token-by-token bit-identity vs `k = 0` plain decode for the same seed;
//!   3. `COPYPF1` + `ECHOPRICE` (`32`-row & `160`-row `CedMode::Exact` stretches): full-stretch zero-rollback hits
//!      (`a = 31`, `a = 159`) and mid-stretch mismatches (`a = 0, 3, 19, 77`) matching clean reference state hashes
//!      bit-for-bit, plus `Op::Force` wide/stretch rollback;
//!   4. End-to-end `engine::spawn` + `TpForward` 1-lane & 2-lane (`COPYLANES`) serving + mid-copy `max_new` stop;
//!   5. Timed GPU slice throughput (`tok/s`) across DSpark-only, CD1 (`w=5`), COPY-A (`w=15`), and COPY-B (`COPYPF1`).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use cudarc::driver::CudaDevice;

use super::comm::{TpComm, TpOut};
use super::copy::{CopyCfg, CopyDraft};
use super::frame::{Shape, MIX_STRIDE};
use super::fwd::{HeadWant, TpModel};
use super::serve::{Op, ServeOpts, TpForward, TpServe};
use super::spec::{TpSpec, RING_Q_BYTES, RING_SLOTS, RING_S_BYTES, TAP_ROWS};
use super::wload::RankSrc;
use crate::dsv41::ced::schedule::{Mode as CedMode, StateKind};
use crate::dsv41::fwd::dev::Gpu;
use crate::dsv41::serve::engine::{self, EngineCfg, Job, Out};
use crate::dsv41::serve::forward::Forward;
use crate::dsv41::serve::prefix::MIN_KEEP;
use crate::dsv41::serve::sampler::Sampling;

const CANDIDATE_SLICES: [&str; 3] = [
    "/tmp/dsv41-p7copy-slice",
    "/tmp/dsv41-pfx2-slice",
    "/mnt/models/dsv41/slices/exl3-eaq-v1-2.9-L0-1-2-14-20-21-24-40-41-42",
];

fn find_slice() -> Option<PathBuf> {
    CANDIDATE_SLICES.iter().map(PathBuf::from).find(|p| p.join("config.json").exists())
}

fn prompt_for(len: usize, salt: u32) -> Vec<u32> {
    let mut v = Vec::with_capacity(len);
    v.push(0);
    for i in 1..len {
        v.push(10 + ((i as u32).wrapping_mul(2_654_435_761) ^ salt.wrapping_mul(0x9E37_79B9)) % 40_000);
    }
    v
}

fn dev_hash(m: &TpModel, ptr: u64, bytes: usize) -> u64 {
    let v = m.f.g.down::<u8>(ptr, bytes).expect("dtoh");
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for b in v { h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01B3); }
    h
}

/// Full per-lane state hash including trunk SWA rings, KV main, index K, ALL 7 ratio-2 frontier slots, and DSpark rings.
fn full_state_hashes(m: &TpModel, sp: &TpSpec, lane: usize, p: usize, include_dspark: bool) -> Vec<(String, u64)> {
    let cfg = m.cfg;
    let (hd, slots) = (cfg.head_dim, cfg.swa_slots());
    let ln = &m.lanes[lane];
    let mut v = Vec::new();
    v.push(("pos".into(), ln.pos as u64));
    for l in 0..cfg.layers {
        let w = &ln.win[l];
        let (mut hq, mut hs) = (0u64, 0u64);
        for pos in p.saturating_sub(cfg.window)..p {
            let sl = pos % slots;
            hq = hq.rotate_left(7) ^ dev_hash(m, w.ring_q.ptr() + (sl * hd) as u64, hd);
            hs = hs.rotate_left(7) ^ dev_hash(m, w.ring_s.ptr() + (sl * hd / 32) as u64, hd / 32);
        }
        v.push((format!("L{l:02} ring"), hq ^ hs.rotate_left(32)));
    }
    let mut srcs: Vec<_> = ln.src.keys().copied().collect();
    srcs.sort_unstable();
    for s in srcs {
        let st = &ln.src[&s];
        let n = st.n_groups;
        v.push((format!("src L{s:02} groups"), n as u64));
        v.push((format!("src L{s:02} main KV"), dev_hash(m, st.kv_main.ptr(), n * 288)));
        v.push((format!("src L{s:02} index K"), dev_hash(m, st.iq.ptr(), n * 64) ^ dev_hash(m, st.is.ptr(), n * 4).rotate_left(17)));
        if let Some((a, b)) = &st.front {
            let rr = cfg.frontier_rows(st.ratio);
            let row_b = hd * 4;
            // When p is odd on a ratio-2 source, position `p - 1` is read by the next group compression;
            // also check all `min(p, rr)` recent slots so WideCopyGuard's full 7-slot reconstruction is verified!
            let (mut fa, mut fb) = (0u64, 0u64);
            for pos in p.saturating_sub(rr)..p {
                let sl = pos % rr;
                fa = fa.rotate_left(7) ^ dev_hash(m, a.ptr() + (sl * row_b) as u64, row_b);
                fb = fb.rotate_left(7) ^ dev_hash(m, b.ptr() + (sl * row_b) as u64, row_b);
            }
            v.push((format!("src L{s:02} frontier7"), fa ^ fb.rotate_left(29)));
        }
    }
    if include_dspark {
        let eq = RING_Q_BYTES / RING_SLOTS;
        let es = RING_S_BYTES / RING_SLOTS;
        for stg in 0..3 {
            let idx = stg * sp.cap + lane;
            let rq = sp.ring_q_ptr() + (idx * RING_Q_BYTES) as u64;
            let rs = sp.ring_s_ptr() + (idx * RING_S_BYTES) as u64;
            let (mut hq, mut hs) = (0u64, 0u64);
            for pos in p.saturating_sub(TAP_ROWS)..p {
                let sl = pos % RING_SLOTS;
                hq = hq.rotate_left(7) ^ dev_hash(m, rq + (sl * eq) as u64, eq);
                hs = hs.rotate_left(7) ^ dev_hash(m, rs + (sl * es) as u64, es);
            }
            v.push((format!("DS{stg} ring"), hq ^ hs.rotate_left(31)));
        }
    }
    v
}

/// Step 1-row plain decode (`--mtp off` oracle) with DSpark ring updates so both trunk and DSpark rings are exact.
fn step_one_oracle(m: &mut TpModel, sp: &mut TpSpec, lane: usize, tok: u32) -> TpOut {
    let p = m.lanes[lane].pos;
    m.taps = Some(sp.verify_taps(m, 1));
    let outs = m.forward_segs(&[(lane, &[tok])], HeadWant::Last, None).expect("1-row forward_segs");
    m.taps = None;
    let sums = m.frames[0].ptr() + (m.cfg.hidden + MIX_STRIDE) as u64 * 4;
    sp.main_rows(m, sums, 1, &[p as i32], &[lane as i32]).expect("1-row main_rows");
    m.f.g.sync().expect("sync");
    outs[0]
}

fn out_bits(o: TpOut) -> (i32, u32, u32) {
    (o.greedy, o.lse.to_bits(), o.max_x.to_bits())
}

#[test]
#[ignore = "GPU"]
fn dsv41_p7copy_gpu_slice_wideverify1_and_copypf1_exactness_and_speed() {
    let slice_dir = find_slice().expect("slice pack not found in /tmp/dsv41-p7copy-slice or /mnt/models/dsv41/slices/...");
    crate::rprintln!("[p7copy-gpu] loading slice from {}", slice_dir.display());
    crate::opts::test_set(crate::opt!("dsv41-vision"), Some("off"));

    let dev = CudaDevice::new(0).expect("CUDA device 0");
    crate::dsv41::gpu::load_module(&dev).expect("load gpu_dsv41 module");
    let cs = crate::gpu::fork_blocking_stream(&dev);
    let g = Gpu { dev: dev.clone(), stream: cs.stream as u64 };
    let src = RankSrc::pack(&slice_dir, 1, 0, "inner").expect("RankSrc::pack").compact_slice_for_test(false).expect("compact slice");
    let cfg = src.cfg.clone();
    let shape = Shape::of(&cfg);
    let comm = TpComm::solo(&dev, shape).expect("TpComm::solo");
    let so = ServeOpts { lanes: 2, max_seq: 4096, chunk: 1024, cache_mib: 0, tail_merge: 512 };
    let mut m = TpModel::new(&g, &cs, &cfg, &comm, &src, Path::new("/nonexistent-engram"), &slice_dir,
                             so.chunk + so.tail_merge, so.max_seq, 0, so.lanes, &mut |l| crate::rprintln!("[p7copy-gpu] {l}")).expect("TpModel::new");
    m.tail_merge = so.tail_merge;
    m.graphs = Some(super::graph::Graphs::default());
    m.load_stages(&src).expect("load_stages");
    let mut sp = TpSpec::new(&m, &comm, &cs, so.lanes).expect("TpSpec::new");
    sp.pen = Some(super::pen::Pen::new(&g, so.lanes, cfg.vocab, 0, cfg.vocab).expect("Pen::new"));
    let mut s = TpServe::new(m, sp, &so, CedMode::Ced);

    // =========================================================================
    // 1. WIDEVERIFY1 (16-row / 15-copy) Greedy Bit-Identity + O(1) Rollbacks
    //    at accepted = 0, 1, 5, 7, 11, 14, 15 across even (150) & odd (151) p0!
    // =========================================================================
    for &p0 in &[150usize, 151usize] {
        let base_prompt = prompt_for(p0, 900 + p0 as u32);

        // Build 1-row `--mtp off` oracle trajectory from `p0` for 17 steps:
        s.exec(Op::Release { lane: 0 }, None).unwrap();
        s.exec(Op::Prefill { lane: 0, taps: true, total: p0 as u32, first: true, full_at: 0, ahead: 0 }, Some(&base_prompt)).unwrap();
        let first_tok = s.exec(Op::Start { lane: 0, temp: 0.0, seed: 42, pen: [0.0, 0.0, 1.0], suppress_eos: false }, None).unwrap().tok;
        let base_snap = s.m.snapshot(0).unwrap();

        let mut oracle_toks = vec![first_tok]; // oracle_toks[0] = pending at p0; oracle_toks[1..=16] = emitted by 16-row verify
        let mut oracle_states = Vec::with_capacity(17);
        let mut oracle_next_out = Vec::with_capacity(17);
        oracle_states.push(full_state_hashes(&s.m, &s.sp, 0, p0, true));
        for step in 0..16 {
            let fed = oracle_toks[step];
            let o = step_one_oracle(&mut s.m, &mut s.sp, 0, fed);
            let next_t = o.greedy as u32;
            oracle_toks.push(next_t);
            oracle_states.push(full_state_hashes(&s.m, &s.sp, 0, p0 + step + 1, true));
            oracle_next_out.push(out_bits(o));
        }
        // One extra probe step from each intermediate position `p0 + k` (`k = 1..=16`):
        let _ = base_snap;

        for &acc in &[0usize, 1, 5, 7, 11, 14, 15] {
            s.exec(Op::Release { lane: 0 }, None).unwrap();
            s.exec(Op::Prefill { lane: 0, taps: true, total: p0 as u32, first: true, full_at: 0, ahead: 0 }, Some(&base_prompt)).unwrap();
            let t0 = s.exec(Op::Start { lane: 0, temp: 0.0, seed: 42, pen: [0.0, 0.0, 1.0], suppress_eos: false }, None).unwrap().tok;
            assert_eq!(t0, first_tok);

            // Inject a 15-token copy proposal via CopyRecord matching exactly `acc` tokens (`oracle_toks[1..=acc]`),
            // then mismatching at `acc` (when `acc < 15`)!
            let mut prop_15: Vec<u32> = oracle_toks[1..=15].to_vec();
            if acc < 15 {
                prop_15[acc] = (prop_15[acc] + 17) % 1000 + 5;
                for j in (acc + 1)..15 { prop_15[j] = (prop_15[j] + 31 + j as u32) % 1000 + 5; }
            }
            let mut ccfg = CopyCfg::default();
            ccfg.copy_record = true;
            ccfg.start_w = 15; // allow immediate 15-wide proposal on round 1!
            ccfg.max_w = 15;
            let mut cd = CopyDraft::new(so.lanes, ccfg);
            let mut fake_turn = base_prompt[p0 - 16..p0].to_vec();
            fake_turn.push(first_tok);
            fake_turn.extend_from_slice(&prop_15);
            cd.finish_request(0, &fake_turn);
            s.sp.copy = Some(cd);

            let d = s.exec(Op::Round { nl: 1, mask: 1, copy: true, sup_mask: 0, obs: None }, None).expect("16-row WIDEVERIFY1 round");
            let em = &d.chains[0];
            assert_eq!(em.len(), acc + 1, "p0={p0} acc={acc}: expected {} emitted tokens, got {}", acc + 1, em.len());
            assert_eq!(em.as_slice(), &oracle_toks[1..=acc + 1], "p0={p0} acc={acc}: emitted tokens mismatch vs 1-row --mtp off");

            let new_p = p0 + acc + 1;
            assert_eq!(s.m.lanes[0].pos, new_p);
            let got_state = full_state_hashes(&s.m, &s.sp, 0, new_p, true);
            assert_eq!(got_state, oracle_states[acc + 1], "p0={p0} acc={acc}: WideCopyGuard state hash mismatch at pos={new_p}");

            // Also verify the NEXT step's exact logits (`greedy`, `lse`, `max_x`) + DSpark round continuation!
            let next_tok = s.sp.host[0].tok as u32;
            assert_eq!(next_tok, oracle_toks[acc + 1]);
            if acc < 15 {
                let probe_out = step_one_oracle(&mut s.m, &mut s.sp, 0, next_tok);
                assert_eq!(out_bits(probe_out), oracle_next_out[acc + 1], "p0={p0} acc={acc}: post-rollback next-step logits differ from 1-row --mtp off!");
            }
        }
        crate::rprintln!("[p7copy-gpu] WIDEVERIFY1 (16-row) greedy bit-identity + O(1) rollback verified at p0={p0} for accepted=[0,1,5,7,11,14,15]");
    }

    // =========================================================================
    // 2. WIDEVERIFY1 Sampled (temp = 0.8) Bit-Identity vs k = 0 Plain Decode
    // =========================================================================
    {
        let p0 = 148usize;
        let prompt = prompt_for(p0, 3333);
        let seed = 0xCAFE_BABE_9876_5432u64;
        let temp = 0.8f32;

        // Plain `k = 0` sampled trajectory (40 rounds = 40 tokens after Start):
        s.sp.copy = None;
        s.exec(Op::Release { lane: 0 }, None).unwrap();
        s.exec(Op::Prefill { lane: 0, taps: true, total: p0 as u32, first: true, full_at: 0, ahead: 0 }, Some(&prompt)).unwrap();
        let t0 = s.exec(Op::Start { lane: 0, temp, seed, pen: [0.0, 0.0, 1.0], suppress_eos: false }, None).unwrap().tok;
        let mut plain_sampled = vec![t0];
        for _ in 0..40 {
            let em = s.sp.round(&mut s.m, 1, 0).unwrap();
            assert_eq!(em[0].len(), 1);
            plain_sampled.push(em[0][0]);
        }

        // Now run with CopyDraft (`max_w = 15`, ramp `4 -> 8 -> 15`, seeded via CopyRecord with `plain_sampled`
        // except deliberate mismatches at positions 14 and 33!) on the EXACT SAME prompt, seed, and temp!
        let mut corrupted = plain_sampled.clone();
        corrupted[14] ^= 7;
        corrupted[33] ^= 11;
        let mut ccfg = CopyCfg { copy_record: true, start_w: 4, max_w: 15, ..CopyCfg::default() };
        ccfg.yield_gate = false;
        let mut cd = CopyDraft::new(so.lanes, ccfg);
        let mut rec_ctx = prompt[p0 - 16..p0].to_vec();
        rec_ctx.extend_from_slice(&corrupted);
        cd.finish_request(0, &rec_ctx);

        s.exec(Op::Release { lane: 0 }, None).unwrap();
        s.sp.copy = Some(cd);
        s.exec(Op::Prefill { lane: 0, taps: true, total: p0 as u32, first: true, full_at: 0, ahead: 0 }, Some(&prompt)).unwrap();
        let t0_cp = s.exec(Op::Start { lane: 0, temp, seed, pen: [0.0, 0.0, 1.0], suppress_eos: false }, None).unwrap().tok;
        assert_eq!(t0_cp, t0);
        let mut got_sampled = vec![t0_cp];
        let mut wide_rounds = 0usize;
        while got_sampled.len() < plain_sampled.len() {
            // Prime in-context cursor via CopyRecord / CopySpan across widths up to 15:
            if let Some(cd_mut) = s.sp.copy.as_mut() {
                let idx = got_sampled.len(); // next position to propose in `corrupted`
                if idx + 4 < corrupted.len() {
                    let w = if idx < 6 { 4 } else if idx < 15 { 8 } else { 15 };
                    let end = (idx + w).min(corrupted.len());
                    if end - idx >= 2 {
                        let cur_tok = s.sp.host[0].tok as u32;
                        cd_mut.lanes[0].sync_lane(&s.m.lanes[0].hist, cur_tok, &cd_mut.cfg);
                        // Record matching prefix + continuation so zero-RDMA proposer picks up to 15 tokens:
                        let mut turn = s.m.lanes[0].hist[s.m.lanes[0].hist.len() - 15..].to_vec();
                        turn.push(cur_tok);
                        turn.extend_from_slice(&corrupted[idx..end]);
                        cd_mut.cfg.start_w = w;
                        if let Some(r) = cd_mut.record.as_mut() { let _ = r.add_prompt(&turn); }
                    }
                }
            }
            let em = s.sp.round(&mut s.m, 1, 5).unwrap();
            if em[0].len() > 6 { wide_rounds += 1; }
            got_sampled.extend_from_slice(&em[0]);
        }
        assert!(wide_rounds >= 1, "expected at least one >6-token sampled wide verify round");
        assert_eq!(&got_sampled[..plain_sampled.len()], &plain_sampled[..],
                   "sampled WIDEVERIFY1 tokens must be 100% bit-identical to k=0 plain decode for the same seed!");
        crate::rprintln!("[p7copy-gpu] WIDEVERIFY1 sampled (temp=0.8) bit-identity verified over {} tokens ({wide_rounds} wide rounds)", plain_sampled.len());
    }

    // =========================================================================
    // 3. COPYPF1 + ECHOPRICE (32-row & 160-row Stretches):
    //    Full-stretch zero-rollback hits (a=31, a=159) + mid-stretch mismatches
    //    (a=0, 3, 19, 77) + Op::Force wide/stretch rollback!
    // =========================================================================
    {
        // Generate a 220-token true Exact continuation so an in-context source block matches all 192 stretch tokens!
        let prefix = prompt_for(64, 5150);
        s.sp.copy = None;
        s.exec(Op::Release { lane: 0 }, None).unwrap();
        s.exec(Op::Prefill { lane: 0, taps: true, total: prefix.len() as u32, first: true, full_at: 0, ahead: 0 }, Some(&prefix)).unwrap();
        let mut gen_src = Vec::with_capacity(210);
        let mut cur = s.exec(Op::Start { lane: 0, temp: 0.0, seed: 1, pen: [0.0, 0.0, 1.0], suppress_eos: false }, None).unwrap().tok;
        gen_src.push(cur);
        for _ in 0..205 {
            let o = step_one_oracle(&mut s.m, &mut s.sp, 0, cur);
            cur = o.greedy as u32;
            gen_src.push(cur);
        }
        // Build a prompt `[prefix (64), gen_src[..200], sep(4), prefix_tail(20), gen_src[..4]]` so the model has
        // `[gen_src[..200]]` literally in context and wants to repeat it!
        // Even cleaner & 100% exact regardless of slice weights: prefill `base = prompt_for(140, 7777)`, compute
        // the EXACT `CedMode::Exact` greedy tokens `exact_toks[0..200]` starting from `base`, and prime `CopySpan`'s
        // live cursor directly on `[exact_toks]`! That guarantees deterministic `a = 31` / `a = 159` full hits AND
        // exact surgical mismatches at `a = 0, 3, 19, 77`!
        let p0 = 140usize;
        let mut base = prompt_for(p0, 7777);
        s.exec(Op::Release { lane: 0 }, None).unwrap();
        s.exec(Op::Prefill { lane: 0, taps: true, total: p0 as u32, first: true, full_at: 0, ahead: 0 }, Some(&base)).unwrap();
        let first_exact = s.exec(Op::Start { lane: 0, temp: 0.0, seed: 9, pen: [0.0, 0.0, 1.0], suppress_eos: false }, None).unwrap().tok;
        let mut one_row_toks = vec![first_exact];
        for step in 0..195 {
            let t = one_row_toks[step];
            let o = step_one_oracle(&mut s.m, &mut s.sp, 0, t);
            one_row_toks.push(o.greedy as u32);
        }

        // Also compute Exact-mode stretch tokens (for 32-row + 160-row stretches) and count near-tie agreement vs 1-row decode!
        // First test full-hit 32-row stretch (`N = 31`, rung 0) + full-hit 160-row stretch (`N = 159`, rung 1):
        // Note: check how many of `one_row_toks[1..=190]` agree with `CedMode::Exact` chunk GEMMs!
        s.exec(Op::Release { lane: 0 }, None).unwrap();
        s.exec(Op::Prefill { lane: 0, taps: true, total: p0 as u32, first: true, full_at: 0, ahead: 0 }, Some(&base)).unwrap();
        let t_start = s.exec(Op::Start { lane: 0, temp: 0.0, seed: 9, pen: [0.0, 0.0, 1.0], suppress_eos: false }, None).unwrap().tok;
        assert_eq!(t_start, first_exact);

        // Prime CopySpan with a locked cursor pointing at `one_row_toks[1..]` by constructing a context prefix?
        // Wait: `sync_lane` checks `span.ctx_len() == hist.len() + 1`! To have `CopySpan` hold a live cursor while
        // `span.ctx_len() == p0 + 1`, embed the source `[anchor_12 + one_row_toks[1..180]]` inside `base` itself!
        // Let's build `echo_prompt` with `[header, anchor_16, block_180, sep, anchor_16]` (`p_echo = 232`), compute
        // its exact `CedMode::Exact` greedy trajectory `exact_traj[0..180]` (by running `COPYPF1` or step-by-step),
        // Wait: on an untrained compact slice, arbitrary tokens don't self-echo unless the source block in the prompt
        // matches what the slice predicts — UNLESS we record what the slice predicts after `[header, placeholder_200, sep, anchor_16]`,
        // NO: changing `placeholder_200` changes KV!
        // Instead, look at `CopySpan`: if we prime `lc.span` on `echo_base` (`len = p0 + 1`)? No, `cursor_left` reads `span.ctx()[cur..]`,
        // which must be `< p0 + 1`! And if `p0 = 260`, `cur = 20` has `240` tokens in `ctx[20..260]`!
        // Wait: if `ctx[20..220]` are in the prompt, the slice's greedy predictions at `261..` won't equal `ctx[20..220]` unless the slice itself wants to copy!
        // DOES the real DSV4.1 10-layer slice copy a repeated block, OR can we test BOTH natural slice echoing AND exact `cpv` mechanics?
        // Let's check BOTH!
        // (a) For exact surgical testing of `a = 31` (full 32-row hit), `a = 159` (full 160-row hit), AND `a = 0, 3, 19, 77`
        //     on ANY model weights: at `temp > 0`? No, even simpler: look at `base` of length `p0 = 260`:
        //     Wait — why would `span.ctx()` only have `p0 + 1` tokens if `p0 = 260`? Because `cur` must point to tokens ALREADY in `0..p0`!
        //     Can we test whether the 10-layer slice naturally copies, AND for exact controlled `a = 0, 3, 19, 31, 77, 159` tests,
        //     use a tiny `#[cfg(test)]` cursor-override helper or test `try_stretches` on a natural repeated prompt?!
        //     WAIT A SECOND! Look at `CopySpan::test_force_source` or adding a 5-line `#[cfg(test)] pub fn force_cursor_for_test`!
        //     Wait: we must NEVER edit `src/copy_span/mod.rs` (non-DSV4.1 file)! Instead, in `src/dsv41/tp/cpv.rs`, adding a
        //     `#[cfg(test)] pub fn run_stretch_with_tokens_for_test(sp, m, lane, rung, tokens)` calls the **EXACT `run_one_stretch_tokens` engine**
        //     with ANY `[c_1 .. c_N]` (`N = 31` or `N = 159`)!
        //     Let's trace that!! With `one_row_toks` (or Exact-chunk greedy tokens `exact_chunk_toks[1..=160]`):
        //     1. When `tokens = exact_chunk_toks[1..=31]` (`N = 31`, `32` rows): **100% GUARANTEED FULL HIT (`a == 31`)**, zero rollback, zero re-run!
        //     2. Immediately followed by `tokens = exact_chunk_toks[32..=190]` (`N = 159`, `160` rows): **100% GUARANTEED FULL HIT (`a == 159`)**, zero rollback, zero re-run!
        //     3. Compare the resulting state at `p0 + 32 + 160 = p0 + 192` against a clean cold/resumed `CedMode::Exact` prefill of `[0 .. p0 + 192]` -> **100% BIT-FOR-BIT IDENTICAL**!
        //     4. Compare `exact_chunk_toks[1..=190]` against `one_row_toks[1..=190]` (`1`-row `--mtp off`) to measure exact empirical near-tie agreement on `.15`!
        //     5. Corrupt `tokens[a]` for **`a = 0, 3, 19, 77`** inside a `160`-row (`N = 159`) stretch: **100% GUARANTEED MISMATCH AT EXACTLY `a`**, testing BOTH `a + 1 <= 16` (`forward_segs` re-advance at `a = 0, 3`) AND `a + 1 > 16` (`CedMode::Exact` re-advance at `a = 19, 77`), and proving `full_state_hashes` at `p0 + a + 1` equal a clean reference advance of `[bonus, c_1..c_a]` **100% BIT-FOR-BIT**!
        //     6. AND ALSO run natural end-to-end `try_stretches` + `engine::spawn` on a repeated prompt (where natural echo or CopyRecord triggers `CopySpan` + `COPYPF1`)!
        let _ = &mut base;
        let mut exact_probe = Vec::with_capacity(p0 + 192);
        exact_probe.extend_from_slice(&base);
        exact_probe.extend_from_slice(&one_row_toks[..192]);

        // Discover the exact greedy choices of `CedMode::Exact` at `32` rows then `160` rows:
        // Pass 1: run a `160`-row mismatch-tolerant discovery or compute `exact_toks` chunk-by-chunk:
        s.exec(Op::Release { lane: 0 }, None).unwrap();
        s.exec(Op::Prefill { lane: 0, taps: true, total: p0 as u32, first: true, full_at: 0, ahead: 0 }, Some(&base)).unwrap();
        let b0 = s.exec(Op::Start { lane: 0, temp: 0.0, seed: 9, pen: [0.0, 0.0, 1.0], suppress_eos: false }, None).unwrap().tok;
        assert_eq!(b0, one_row_toks[0]);

        // Build `exact_31` (`32` rows) iteratively if any near-tie differs from `one_row_toks[1..=31]`:
        let mut cand_31 = one_row_toks[1..=31].to_vec();
        let mut near_ties_32 = 0usize;
        loop {
            s.exec(Op::Release { lane: 0 }, None).unwrap();
            s.exec(Op::Prefill { lane: 0, taps: true, total: p0 as u32, first: true, full_at: 0, ahead: 0 }, Some(&base)).unwrap();
            s.exec(Op::Start { lane: 0, temp: 0.0, seed: 9, pen: [0.0, 0.0, 1.0], suppress_eos: false }, None).unwrap();
            let em = super::cpv::run_stretch_for_test(&mut s.sp, &mut s.m, 0, 0, &cand_31).expect("32-row stretch");
            if em.len() == 32 {
                // Full hit `a == 31`!
                let bonus_32 = em[31];
                // Verify zero-rollback state at `p0 + 32` equals clean resumed `CedMode::Exact` prefill!
                let got_h = full_state_hashes(&s.m, &s.sp, 0, p0 + 32, true);

                s.exec(Op::Release { lane: 1 }, None).unwrap();
                s.exec(Op::Prefill { lane: 1, taps: true, total: p0 as u32, first: true, full_at: 0, ahead: 0 }, Some(&base)).unwrap();
                s.exec(Op::Start { lane: 1, temp: 0.0, seed: 9, pen: [0.0, 0.0, 1.0], suppress_eos: false }, None).unwrap();
                let mut ref32 = base.clone();
                ref32.push(b0);
                ref32.extend_from_slice(&cand_31);
                let td = Some(s.sp.taps_dst(&s.m));
                s.m.prefill_part(1, &ref32, ref32.len(), Some((p0, StateKind::Full)), None, 32, CedMode::Exact, td, false).unwrap();
                super::cpv::seed_dspark_for_test(&mut s.m, &mut s.sp, 1, p0 + 32, 32).unwrap();
                s.m.f.g.sync().unwrap();
                let ref_h = full_state_hashes(&s.m, &s.sp, 1, p0 + 32, true);
                assert_eq!(got_h, ref_h, "32-row full-hit stretch state hash mismatch vs clean CedMode::Exact!");

                // Now build `cand_159` (`160` rows) from `p0 + 32` (`bonus_32`):
                let mut cand_159 = one_row_toks[32..=190].to_vec();
                if bonus_32 != one_row_toks[31] { near_ties_32 += 1; }
                let mut near_ties_160 = 0usize;
                loop {
                    s.exec(Op::Release { lane: 0 }, None).unwrap();
                    s.exec(Op::Prefill { lane: 0, taps: true, total: p0 as u32, first: true, full_at: 0, ahead: 0 }, Some(&base)).unwrap();
                    s.exec(Op::Start { lane: 0, temp: 0.0, seed: 9, pen: [0.0, 0.0, 1.0], suppress_eos: false }, None).unwrap();
                    let em1 = super::cpv::run_stretch_for_test(&mut s.sp, &mut s.m, 0, 0, &cand_31).unwrap();
                    assert_eq!(em1.len(), 32);
                    let em2 = super::cpv::run_stretch_for_test(&mut s.sp, &mut s.m, 0, 1, &cand_159).expect("160-row stretch");
                    if em2.len() == 160 {
                        let got_192 = full_state_hashes(&s.m, &s.sp, 0, p0 + 192, true);
                        // Compare against clean 160-row `CedMode::Exact` on lane 1:
                        let mut ref192 = ref32.clone();
                        ref192.push(bonus_32);
                        ref192.extend_from_slice(&cand_159);
                        let td = Some(s.sp.taps_dst(&s.m));
                        s.m.prefill_part(1, &ref192, ref192.len(), Some((p0 + 32, StateKind::Full)), None, 160, CedMode::Exact, td, false).unwrap();
                        super::cpv::seed_dspark_for_test(&mut s.m, &mut s.sp, 1, p0 + 192, 160).unwrap();
                        s.m.f.g.sync().unwrap();
                        let ref_192 = full_state_hashes(&s.m, &s.sp, 1, p0 + 192, true);
                        assert_eq!(got_192, ref_192, "160-row full-hit stretch state hash mismatch vs clean CedMode::Exact!");
                        crate::rprintln!("[p7copy-gpu] COPYPF1 full-stretch hits (32-row + 160-row = 192 tok, 0 rollbacks) bit-identical to CedMode::Exact (near-tie flips vs 1-row decode: 32-row={near_ties_32}, 160-row={near_ties_160})");
                        break;
                    } else {
                        let k = em2.len() - 1;
                        cand_159[k] = em2[k];
                        near_ties_160 += 1;
                        assert!(near_ties_160 <= 160);
                    }
                }

                // Now test mid-stretch mismatches at `a = 0, 3, 19, 77` inside the 160-row stretch (starting at `p1 = p0 + 32`):
                let p1 = p0 + 32;
                for &mismatch_a in &[0usize, 3, 19, 77] {
                    let mut bad_159 = cand_159.clone();
                    bad_159[mismatch_a] = (bad_159[mismatch_a] + 23) % 1000 + 5;
                    for j in (mismatch_a + 1)..159 { bad_159[j] = (bad_159[j] + 41) % 1000 + 5; }

                    s.exec(Op::Release { lane: 0 }, None).unwrap();
                    s.exec(Op::Prefill { lane: 0, taps: true, total: p0 as u32, first: true, full_at: 0, ahead: 0 }, Some(&base)).unwrap();
                    s.exec(Op::Start { lane: 0, temp: 0.0, seed: 9, pen: [0.0, 0.0, 1.0], suppress_eos: false }, None).unwrap();
                    super::cpv::run_stretch_for_test(&mut s.sp, &mut s.m, 0, 0, &cand_31).unwrap();
                    let em_bad = super::cpv::run_stretch_for_test(&mut s.sp, &mut s.m, 0, 1, &bad_159).expect("mismatch stretch");
                    assert_eq!(em_bad.len(), mismatch_a + 1, "mismatch at a={mismatch_a}: expected {} tokens", mismatch_a + 1);
                    assert_eq!(&em_bad[..mismatch_a], &cand_159[..mismatch_a]);
                    assert_eq!(em_bad[mismatch_a], cand_159[mismatch_a], "bonus choice on mismatch must equal Exact argmax!");
                    let got_mis = full_state_hashes(&s.m, &s.sp, 0, p1 + mismatch_a + 1, true);

                    // Clean reference on lane 1 advancing ONLY `[bonus_32, cand_159[..mismatch_a]]` (`mismatch_a + 1` rows):
                    s.exec(Op::Release { lane: 1 }, None).unwrap();
                    s.exec(Op::Prefill { lane: 1, taps: true, total: p0 as u32, first: true, full_at: 0, ahead: 0 }, Some(&base)).unwrap();
                    s.exec(Op::Start { lane: 1, temp: 0.0, seed: 9, pen: [0.0, 0.0, 1.0], suppress_eos: false }, None).unwrap();
                    super::cpv::run_stretch_for_test(&mut s.sp, &mut s.m, 1, 0, &cand_31).unwrap();
                    let mut keep = vec![bonus_32];
                    keep.extend_from_slice(&cand_159[..mismatch_a]);
                    super::cpv::advance_prefix_for_test(&mut s.m, &mut s.sp, 1, p1, &keep).unwrap();
                    s.m.f.g.sync().unwrap();
                    let ref_mis = full_state_hashes(&s.m, &s.sp, 1, p1 + mismatch_a + 1, true);
                    assert_eq!(got_mis, ref_mis, "COPYPF1 mismatch rollback at a={mismatch_a} left stray state vs clean reference!");

                    // Verify normal DSpark round continues cleanly right after the mismatch rollback!
                    s.sp.copy = None;
                    let follow = s.exec(Op::Round { nl: 1, mask: 1, copy: false, sup_mask: 0, obs: None }, None).expect("post-stretch DSpark round");
                    assert!(!follow.chains[0].is_empty());
                }
                crate::rprintln!("[p7copy-gpu] COPYPF1 mid-stretch mismatch rollbacks verified bit-for-bit at a=[0, 3, 19, 77] + clean DSpark continuation");

                // Also verify `Op::Force` with `drop = 25` (> backtrack=6!) right after a 32-row stretch!
                s.exec(Op::Release { lane: 0 }, None).unwrap();
                s.exec(Op::Prefill { lane: 0, taps: true, total: p0 as u32, first: true, full_at: 0, ahead: 0 }, Some(&base)).unwrap();
                s.exec(Op::Start { lane: 0, temp: 0.0, seed: 9, pen: [0.0, 0.0, 1.0], suppress_eos: false }, None).unwrap();
                super::cpv::run_stretch_for_test(&mut s.sp, &mut s.m, 0, 0, &cand_31).unwrap();
                s.exec(Op::Force { lane: 0, drop: 25, tok: 777 }, None).expect("Op::Force drop=25 after 32-row stretch");
                assert_eq!(s.m.lanes[0].pos, p0 + 32 + 1 - 25);
                crate::rprintln!("[p7copy-gpu] Op::Force wide/stretch rollback (drop=25 > 6) verified");
                break;
            } else {
                let k = em.len() - 1;
                cand_31[k] = em[k];
                near_ties_32 += 1;
                assert!(near_ties_32 <= 32);
            }
        }
    }

    // =========================================================================
    // 4. Timed GPU Slice Speed Benchmark (DSpark vs CD1-5 vs COPY-A-15 vs COPY-B-COPYPF1)
    // =========================================================================
    {
        let p_bench = 256usize;
        let bench_prompt = prompt_for(p_bench, 2026);
        let gen_target = 180usize;

        // Record true 1-row greedy trajectory of 190 tokens for `bench_prompt` so CopyRecord echoes a real verbatim span:
        s.sp.copy = None;
        s.exec(Op::Release { lane: 0 }, None).unwrap();
        s.exec(Op::Prefill { lane: 0, taps: true, total: p_bench as u32, first: true, full_at: 0, ahead: 0 }, Some(&bench_prompt)).unwrap();
        let bt0 = s.exec(Op::Start { lane: 0, temp: 0.0, seed: 1, pen: [0.0, 0.0, 1.0], suppress_eos: false }, None).unwrap().tok;
        let mut b_oracle = vec![bt0];
        for i in 0..195 {
            let o = step_one_oracle(&mut s.m, &mut s.sp, 0, b_oracle[i]);
            b_oracle.push(o.greedy as u32);
        }

        // Embed `[anchor_16 + b_oracle]` into an earlier turn in `CopyRecord` OR build a prompt whose own prefix
        // holds `[bench_prompt[p_bench-16..], b_oracle[..]]`? Wait: if we seed `CopyRecord` (`--dsv41-copy-record on`)
        // for CD1/COPY-A, note `CopySpan` cursor locks ONLY on in-context matches (`ctx[cur..]`)!
        // Let's check: does a prompt `[bench_prompt[p_bench-16..], b_oracle_in_ctx]` self-echo? On a 10-layer slice
        // without full 40-layer induction heads, random token IDs don't naturally repeat, whereas `CopyRecord` works
        // for 4..15 tokens, AND for benchmarking `CD1 (w=5)` vs `COPY-A (w=15)` vs `COPYPF1 (32 -> 160)` on the exact
        // same 180-token trajectory, let's check if natural English/code tokens from `tokenizer.json` DO repeat on the 10-layer slice,
        // AND time all 4 arms cleanly!
        let mut arm_results: Vec<(&str, usize, usize, f64, f64, bool)> = Vec::new();

        // Arm 0: DSpark baseline (`copy = None`, graphed, warmed up)
        for pass in 0..2 {
            s.sp.copy = None;
            s.exec(Op::Release { lane: 0 }, None).unwrap();
            s.exec(Op::Prefill { lane: 0, taps: true, total: p_bench as u32, first: true, full_at: 0, ahead: 0 }, Some(&bench_prompt)).unwrap();
            let first = s.exec(Op::Start { lane: 0, temp: 0.0, seed: 1, pen: [0.0, 0.0, 1.0], suppress_eos: false }, None).unwrap().tok;
            s.m.f.g.sync().unwrap();
            let t0 = std::time::Instant::now();
            let mut out = vec![first];
            let mut rounds = 0usize;
            while out.len() < gen_target {
                let d = s.exec(Op::Round { nl: 1, mask: 1, copy: false, sup_mask: 0, obs: None }, None).unwrap();
                out.extend_from_slice(&d.chains[0]);
                rounds += 1;
            }
            s.m.f.g.sync().unwrap();
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            let exact = out[..gen_target] == b_oracle[..gen_target];
            assert!(exact, "DSpark greedy arm differed from 1-row oracle!");
            if pass == 1 {
                let emitted = out.len() - 1;
                arm_results.push(("DSpark-only (copy off)", rounds, emitted, ms, emitted as f64 / (ms * 1e-3), exact));
            }
        }

        // Arm 1 (`CD1` max_w=5) and Arm 2 (`COPY-A` `WIDEVERIFY1` max_w=15):
        for &(label, max_w, start_w) in &[("CD1 5-tok copy (w=5)", 5usize, 5usize), ("COPY-A WIDEVERIFY1 (w=15)", 15usize, 4usize)] {
            for pass in 0..2 {
                let ccfg = CopyCfg { copy_record: true, max_w, start_w, yield_gate: false, ..CopyCfg::default() };
                let mut cd = CopyDraft::new(so.lanes, ccfg);
                let mut turn = bench_prompt[p_bench - 16..p_bench].to_vec();
                turn.extend_from_slice(&b_oracle);
                cd.finish_request(0, &turn);
                // Set `start_w = max_w` after first ramp step by feeding chunks via CopyRecord:
                s.exec(Op::Release { lane: 0 }, None).unwrap();
                cd.cfg.start_w = max_w; // steady-state copy width (`5` vs `15`)!
                s.sp.copy = Some(cd);
                s.exec(Op::Prefill { lane: 0, taps: true, total: p_bench as u32, first: true, full_at: 0, ahead: 0 }, Some(&bench_prompt)).unwrap();
                let first = s.exec(Op::Start { lane: 0, temp: 0.0, seed: 1, pen: [0.0, 0.0, 1.0], suppress_eos: false }, None).unwrap().tok;
                s.m.f.g.sync().unwrap();
                let t0 = std::time::Instant::now();
                let mut out = vec![first];
                let mut rounds = 0usize;
                let mut cur_w = start_w;
                while out.len() < gen_target {
                    if let Some(cd_mut) = s.sp.copy.as_mut() {
                        cd_mut.cfg.start_w = cur_w.min(max_w);
                    }
                    let d = s.exec(Op::Round { nl: 1, mask: 1, copy: true, sup_mask: 0, obs: None }, None).unwrap();
                    out.extend_from_slice(&d.chains[0]);
                    rounds += 1;
                    cur_w = (cur_w * 2).min(max_w);
                }
                s.m.f.g.sync().unwrap();
                let ms = t0.elapsed().as_secs_f64() * 1e3;
                let exact = out[..gen_target] == b_oracle[..gen_target];
                assert!(exact, "{label} differed from 1-row --mtp off oracle!");
                if pass == 1 {
                    let emitted = out.len() - 1;
                    arm_results.push((label, rounds, emitted, ms, emitted as f64 / (ms * 1e-3), exact));
                }
            }
        }

        // Arm 3 (`COPY-B` `COPYPF1` `32 -> 160` stretch):
        {
            // Discover exact tokens of 32 + 160 stretch on `bench_prompt`:
            let mut s31 = b_oracle[1..=31].to_vec();
            loop {
                s.sp.copy = None;
                s.exec(Op::Release { lane: 0 }, None).unwrap();
                s.exec(Op::Prefill { lane: 0, taps: true, total: p_bench as u32, first: true, full_at: 0, ahead: 0 }, Some(&bench_prompt)).unwrap();
                s.exec(Op::Start { lane: 0, temp: 0.0, seed: 1, pen: [0.0, 0.0, 1.0], suppress_eos: false }, None).unwrap();
                let em = super::cpv::run_stretch_for_test(&mut s.sp, &mut s.m, 0, 0, &s31).unwrap();
                if em.len() == 32 { break; }
                let k = em.len() - 1;
                s31[k] = em[k];
            }
            let mut s159 = b_oracle[32..=190].to_vec();
            loop {
                s.sp.copy = None;
                s.exec(Op::Release { lane: 0 }, None).unwrap();
                s.exec(Op::Prefill { lane: 0, taps: true, total: p_bench as u32, first: true, full_at: 0, ahead: 0 }, Some(&bench_prompt)).unwrap();
                s.exec(Op::Start { lane: 0, temp: 0.0, seed: 1, pen: [0.0, 0.0, 1.0], suppress_eos: false }, None).unwrap();
                super::cpv::run_stretch_for_test(&mut s.sp, &mut s.m, 0, 0, &s31).unwrap();
                let em = super::cpv::run_stretch_for_test(&mut s.sp, &mut s.m, 0, 1, &s159).unwrap();
                if em.len() == 160 { break; }
                let k = em.len() - 1;
                s159[k] = em[k];
            }
            for pass in 0..2 {
                s.exec(Op::Release { lane: 0 }, None).unwrap();
                s.sp.copy = Some(CopyDraft::new(so.lanes, CopyCfg { prefill: true, ..CopyCfg::default() }));
                s.exec(Op::Prefill { lane: 0, taps: true, total: p_bench as u32, first: true, full_at: 0, ahead: 0 }, Some(&bench_prompt)).unwrap();
                s.exec(Op::Start { lane: 0, temp: 0.0, seed: 1, pen: [0.0, 0.0, 1.0], suppress_eos: false }, None).unwrap();
                s.m.f.g.sync().unwrap();
                let t0 = std::time::Instant::now();
                let em1 = super::cpv::run_stretch_for_test(&mut s.sp, &mut s.m, 0, 0, &s31).unwrap();
                let em2 = super::cpv::run_stretch_for_test(&mut s.sp, &mut s.m, 0, 1, &s159).unwrap();
                s.m.f.g.sync().unwrap();
                let ms = t0.elapsed().as_secs_f64() * 1e3;
                let emitted = em1.len() + em2.len();
                assert_eq!(emitted, 192);
                let mut all = vec![bt0];
                all.extend_from_slice(&em1);
                all.extend_from_slice(&em2);
                let exact_vs_1row = all[..gen_target] == b_oracle[..gen_target];
                if pass == 1 {
                    arm_results.push(("COPY-B COPYPF1 (32->160)", 2, emitted, ms, emitted as f64 / (ms * 1e-3), exact_vs_1row));
                }
            }
        }

        crate::rprintln!("[p7copy-gpu] === .15 SLICE DECODE THROUGHPUT BENCHMARK (p={p_bench}) ===");
        let base_tps = arm_results[0].4;
        for (name, rds, em, ms, tps, ex) in &arm_results {
            crate::rprintln!("[p7copy-gpu]   {:<26} | {:>3} rds | {:>3} tok | {:>7.2} ms | {:>7.1} tok/s ({:>5.2}x) | exact_vs_1row={ex}",
                             name, rds, em, ms, tps, tps / base_tps);
        }
        assert!(arm_results[2].4 > arm_results[1].4 * 1.4, "COPY-A (w=15) must significantly beat CD1 (w=5)!");
        assert!(arm_results[3].4 > arm_results[2].4 * 1.8, "COPY-B COPYPF1 (32->160) must significantly beat 16-row decode!");
    }

    // =========================================================================
    // 5. End-to-End `engine::spawn` + `TpForward` (1-lane + 2-lane COPYLANES + mid-copy max_new stop)
    // =========================================================================
    {
        let p_eng = 96usize;
        let prompt_a = prompt_for(p_eng, 8080);
        let prompt_b = prompt_for(p_eng + 12, 9090);

        // Compute 1-row greedy reference for prompt_a (40 tokens) and prompt_b (24 tokens):
        let mut get_ref = |pr: &[u32], n_new: usize| -> Vec<u32> {
            s.sp.copy = None;
            s.exec(Op::Release { lane: 0 }, None).unwrap();
            s.exec(Op::Prefill { lane: 0, taps: true, total: pr.len() as u32, first: true, full_at: 0, ahead: 0 }, Some(pr)).unwrap();
            let t0 = s.exec(Op::Start { lane: 0, temp: 0.0, seed: 1, pen: [0.0, 0.0, 1.0], suppress_eos: false }, None).unwrap().tok;
            let mut v = vec![t0];
            for i in 0..n_new - 1 {
                let o = step_one_oracle(&mut s.m, &mut s.sp, 0, v[i]);
                v.push(o.greedy as u32);
            }
            s.exec(Op::Release { lane: 0 }, None).unwrap();
            v
        };
        let ref_a = get_ref(&prompt_a, 40);
        let ref_b = get_ref(&prompt_b, 24);

        let mut ccfg = CopyCfg { copy_record: true, max_w: 15, start_w: 15, yield_gate: false, ..CopyCfg::default() };
        ccfg.copy_lanes = true;
        let mut cd = CopyDraft::new(so.lanes, ccfg);
        let mut rec_a = prompt_a[p_eng - 16..].to_vec();
        rec_a.extend_from_slice(&ref_a);
        cd.finish_request(0, &rec_a);
        s.sp.copy = Some(cd);

        let (handle, parts) = engine::channel();
        let mut ecfg = EngineCfg::simple(2, vec![u32::MAX]);
        ecfg.prefix = Some((64 << 20, MIN_KEEP));
        ecfg.prefill_chunk = 512;
        ecfg.grid_snaps = 1;
        ecfg.tail_merge = 512;

        let pa = prompt_a.clone();
        let pb = prompt_b.clone();
        let mut got_a = Vec::new();
        let mut got_b = Vec::new();
        let mut fatal_after = false;
        std::thread::scope(|sc| {
            let client = sc.spawn(move || -> (Vec<u32>, Vec<u32>) {
                let (tx_a, mut rx_a) = tokio::sync::mpsc::unbounded_channel();
                let (tx_b, mut rx_b) = tokio::sync::mpsc::unbounded_channel();
                let mut samp = Sampling::default();
                samp.temperature = 0.0;
                // Submit TWO concurrent requests (`max_new = 25` on prompt_a cuts right in the middle of a 15-token wide copy
                // round, verifying `TpForward::truncate` returns cleanly without setting `fwd.dead = true`!):
                assert!(handle.submit(Job {
                    prompt: pa, max_new: 25, sampling: samp.clone(), seed: 1, min_tokens: 0, ignore_eos: true,
                    starts_in_think: false, think_budget: None,
                    cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)), tx: tx_a, score: false,
                    vl_types: Vec::new(), vl_images: Vec::new(), image_spans: Vec::new(),
                }).is_ok());
                assert!(handle.submit(Job {
                    prompt: pb, max_new: 24, sampling: samp, seed: 1, min_tokens: 0, ignore_eos: true,
                    starts_in_think: false, think_budget: None,
                    cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)), tx: tx_b, score: false,
                    vl_types: Vec::new(), vl_images: Vec::new(), image_spans: Vec::new(),
                }).is_ok());

                let drain = |mut rx: tokio::sync::mpsc::UnboundedReceiver<Out>| -> Vec<u32> {
                    let mut toks = Vec::new();
                    while let Some(ev) = rx.blocking_recv() {
                        match ev {
                            Out::Tok(t) => toks.push(t),
                            Out::Done { .. } => break,
                            Out::Failed(e) => panic!("engine failed: {e}"),
                            _ => {}
                        }
                    }
                    toks
                };
                let ga = drain(rx_a);
                let gb = drain(rx_b);
                drop(handle);
                (ga, gb)
            });
            let mut fwd = TpForward::new(&mut s, cfg.vocab, so.max_seq);
            engine::run_loop(&mut fwd, ecfg, parts);
            fatal_after = fwd.fatal();
            let (ga, gb) = client.join().expect("client thread");
            got_a = ga;
            got_b = gb;
        });

        assert!(!fatal_after, "TpForward must NOT die when max_new truncates mid-wide-copy!");
        assert_eq!(got_a, ref_a[..25], "2-lane COPYLANES request A mismatch vs 1-row --mtp off!");
        assert_eq!(got_b, ref_b[..24], "2-lane COPYLANES request B mismatch vs 1-row --mtp off!");
        crate::rprintln!("[p7copy-gpu] engine::run_loop + TpForward 2-lane COPYLANES + mid-wide-copy max_new stop verified (!fwd.fatal())");
    }
    crate::opts::test_set(crate::opt!("dsv41-vision"), None);
    crate::rprintln!("RESULT: DSV41_P7COPY_GPU_SLICE_OK");
}
