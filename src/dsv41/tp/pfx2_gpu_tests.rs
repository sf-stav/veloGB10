//! Leg PFX2 GPU bit-exactness verification (`#[ignore = "GPU"]`, run on `.14` under `flock /tmp/dsv41_gpu.lock`):
//! loads the 10-layer EXL3 slice (`L0,1,2,14,20,21,24` + DSpark `L40,41,42` + globals) into the REAL `TpModel` +
//! `TpSpec` + `TpServe` + `TpForward` + `engine` on CUDA, and proves bit-identical decode state, first-token logits
//! (`greedy`, `lse`, `max_x`), DSpark `Start`, and greedy speculative `Round` continuations across:
//!   * short prompts `< 128`: `pos = 16, 31, 46, 47, 68, 79, 87` (odd & even `at`: `pos - 1`, `pos - 5`, `pos - 6`,
//!     short extension `< 128`, and cross-`TAP_ROWS` extension `47 -> 150`);
//!   * long prompts `>= 128`: `pos = 1174, 2324` (odd & even `at`: `pos - 1`, `pos - 4`, `pos - 6`);
//!   * full `engine::spawn` + `TpForward` + `PrefixPool` two-pass serving over `[31, 46, 47, 87, 1174, 2324]`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use cudarc::driver::CudaDevice;

use super::comm::TpComm;
use super::frame::Shape;
use super::fwd::TpModel;
use super::serve::{Op, ServeOpts, TpForward, TpServe};
use super::spec::TpSpec;
use super::wload::RankSrc;
use crate::dsv41::ced::schedule::Mode as CedMode;
use crate::dsv41::engram::hash::{EngramHasher, NgramHashState};
use crate::dsv41::fwd::dev::Gpu;
use crate::dsv41::serve::engine::{self, EngineCfg, Job, Out};
use crate::dsv41::serve::prefix::MIN_KEEP;
use crate::dsv41::serve::sampler::Sampling;

const CANDIDATE_SLICES: [&str; 2] = [
    "/tmp/dsv41-pfx2-slice",
    "/mnt/models/dsv41/slices/exl3-eaq-v1-2.9-L0-1-2-14-20-21-24-40-41-42",
];

fn find_slice() -> Option<PathBuf> {
    CANDIDATE_SLICES.iter().map(PathBuf::from).find(|p| p.join("config.json").exists())
}

fn prompt_for(len: usize, salt: u32) -> Vec<u32> {
    let mut v = Vec::with_capacity(len);
    v.push(0); // BOS
    for i in 1..len {
        v.push(10 + ((i as u32).wrapping_mul(2_654_435_761) ^ salt.wrapping_mul(0x9E37_79B9)) % 50_000);
    }
    v
}

fn dev_hash(m: &TpModel, ptr: u64, bytes: usize) -> u64 {
    let v = m.f.g.down::<u8>(ptr, bytes).expect("dtoh");
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for b in v { h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01B3); }
    h
}

fn state_hashes(m: &TpModel, lane: usize, p: usize) -> Vec<(String, u64)> {
    let cfg = m.cfg;
    let (hd, slots) = (cfg.head_dim, cfg.swa_slots());
    let ln = &m.lanes[lane];
    let mut v = Vec::new();
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
            // Hash the active frontier slots that a future row or <=6-row rollback could read: [p.saturating_sub(7), p)
            let rr = cfg.frontier_rows(st.ratio);
            let row_b = hd * 4;
            let (mut fa, mut fb) = (0u64, 0u64);
            for pos in p.saturating_sub(rr)..p {
                let sl = pos % rr;
                fa = fa.rotate_left(7) ^ dev_hash(m, a.ptr() + (sl * row_b) as u64, row_b);
                fb = fb.rotate_left(7) ^ dev_hash(m, b.ptr() + (sl * row_b) as u64, row_b);
            }
            v.push((format!("src L{s:02} frontier"), fa ^ fb.rotate_left(29)));
        }
    }
    v
}

#[test]
#[ignore = "GPU"]
fn dsv41_pfx2_gpu_slice_short_and_long_tail_resumes_are_bit_identical() {
    let slice_dir = find_slice().expect("slice pack not found in /tmp/dsv41-pfx2-slice or /mnt/models/dsv41/slices/...");
    crate::rprintln!("[pfx2-gpu] loading slice from {}", slice_dir.display());

    // 1. Verify EngramHasher rollback state_at == cold hash_chunk on all odd/even short & long positions
    if let Some(hs) = EngramHasher::from_model_dir(&slice_dir).expect("hasher parse") {
        let per = hs.ids_per_token();
        for &p in &[16usize, 31, 46, 47, 68, 79, 87, 150, 1174, 2324] {
            let toks = prompt_for(p, 55 + p as u32);
            let mut cold_st = NgramHashState::new();
            let mut cold_ids = vec![0u32; p * per];
            hs.hash_chunk(&mut cold_st, &toks, None, &mut cold_ids).unwrap();
            for d in [1usize, 2, 5, 6] {
                if d >= p { continue; }
                let at = p - d;
                let tail = &toks[at.saturating_sub(3)..at];
                let mut rb_st = hs.state_at(at as u64, tail, None).unwrap();
                let mut rb_ids = vec![0u32; d * per];
                hs.hash_chunk(&mut rb_st, &toks[at..], None, &mut rb_ids).unwrap();
                assert_eq!(&cold_ids[at * per..], &rb_ids[..], "Engram hash mismatch at p={p}, at={at}");
            }
        }
        crate::rprintln!("[pfx2-gpu] EngramHasher rollback state_at == cold hash_chunk verified for all p in [16..2324]");
    }

    // 2. Boot real 7-trunk + 3-DSpark TpServe on CUDA device 0
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
                             so.chunk + so.tail_merge, so.max_seq, 0, so.lanes, &mut |l| crate::rprintln!("[pfx2-gpu] {l}")).expect("TpModel::new");
    m.tail_merge = so.tail_merge;
    m.graphs = Some(super::graph::Graphs::default());
    m.load_stages(&src).expect("load_stages");
    let mut sp = TpSpec::new(&m, &comm, &cs, so.lanes).expect("TpSpec::new");
    sp.pen = Some(super::pen::Pen::new(&g, so.lanes, cfg.vocab, 0, cfg.vocab).expect("Pen::new"));
    let _ = sp.attach_markov_cache(&mut m).expect("attach_markov_cache");
    let mut ts = TpServe::new(m, sp, &so, CedMode::Ced);

    // Helper: run cold prefill + head bits + DSpark greedy Start + 3 speculative rounds on `toks`, returning
    // (state_hashes_after_prefill, head_bits, generated_tokens, snap_id_if_requested)
    let run_cold = |ts: &mut TpServe, lane: usize, toks: &[u32], snap_id: Option<u32>| -> (Vec<(String, u64)>, [u32; 3], Vec<u32>) {
        ts.exec(Op::Release { lane }, None).unwrap();
        let total = toks.len();
        let mut off = 0usize;
        let mut first = true;
        while off < total {
            let mut end = (off + so.chunk).min(total);
            if end < total && total - end <= so.tail_merge + 1 { end = total; }
            ts.exec(Op::Prefill { lane, taps: true, total: total as u32, first, full_at: 0, ahead: 0 }, Some(&toks[off..end])).unwrap();
            off = end;
            first = false;
        }
        let st = state_hashes(&ts.m, lane, total);
        let hd = ts.m.head_last(ts.m.lanes[lane].pos.min(1) * 0).ok(); // note: last_row is stored inside TpServe, check via head_last on r
        let _ = hd;
        if let Some(id) = snap_id {
            let sd = ts.exec(Op::Snap { lane, id }, None).unwrap();
            assert!(sd.snap_full && sd.snap_tail && sd.snap_pos == total, "snap at {total}: {sd:?}");
        }
        let d0 = ts.exec(Op::Start { lane, temp: 0.0, seed: 42, pen: [0.0, 0.0, 1.0], suppress_eos: false }, None).unwrap();
        let mut gen = vec![d0.tok];
        for _ in 0..3 {
            let rd = ts.exec(Op::Round { nl: lane + 1, mask: 1 << lane, copy: false, sup_mask: 0, obs: None }, None).unwrap();
            gen.extend_from_slice(&rd.chains[lane]);
        }
        // Also capture exact greedy/lse/max_x bits on one probe token at `total + gen`? Even stronger: capture head_last right after prefill!
        ts.exec(Op::Release { lane }, None).unwrap();
        (st, [d0.tok, gen.len() as u32, gen.iter().fold(0u32, |a, &x| a.wrapping_mul(31) ^ x)], gen)
    };

    // Helper: capture exact first-token head TpOut bits (`greedy`, `lse.to_bits()`, `max_x.to_bits()`) after prefill
    let head_bits_cold = |ts: &mut TpServe, toks: &[u32]| -> [u32; 3] {
        ts.m.reset_lane_n(0).unwrap();
        let td = Some(ts.sp.taps_dst(&ts.m));
        let r = ts.m.prefill(0, toks, so.chunk, CedMode::Ced, td, true).unwrap();
        let h = r.head.expect("head");
        [h.greedy as u32, h.lse.to_bits(), h.max_x.to_bits()]
    };

    let poison = prompt_for(200, 0xDEAD_BEEF);
    let poison_lane0_and_taps = |ts: &mut TpServe| {
        ts.exec(Op::Release { lane: 0 }, None).unwrap();
        ts.exec(Op::Prefill { lane: 0, taps: true, total: poison.len() as u32, first: true, full_at: 0, ahead: 0 }, Some(&poison)).unwrap();
        ts.exec(Op::Release { lane: 0 }, None).unwrap();
    };

    let head_bits_resumed = |ts: &mut TpServe, snap_id: u32, at: usize, toks: &[u32]| -> [u32; 3] {
        poison_lane0_and_taps(ts);
        ts.exec(Op::Restore { lane: 0, id: snap_id }, None).unwrap();
        if at < ts.m.lanes[0].pos {
            ts.exec(Op::Truncate { lane: 0, pos: at as u32 }, None).unwrap();
        }
        let r = ts.m.prefill_part(0, toks, toks.len(), None, None, so.chunk, CedMode::Ced, Some(ts.sp.taps_dst(&ts.m)), true).unwrap();
        let h = r.head.expect("resumed head");
        ts.exec(Op::Release { lane: 0 }, None).unwrap();
        [h.greedy as u32, h.lse.to_bits(), h.max_x.to_bits()]
    };

    let mut next_id = 1u32;
    for &p in &[16usize, 31, 46, 47, 68, 79, 87, 1174, 2324] {
        let base = prompt_for(p, 1000 + p as u32);
        let sid = next_id;
        next_id += 1;
        // Take snapshot `sid` on lane 1, while capturing cold reference on lane 0 (`nl = 1`)
        let _ = run_cold(&mut ts, 1, &base, Some(sid));
        let hb_cold = head_bits_cold(&mut ts, &base);
        let (st_cold, _, gen_cold) = run_cold(&mut ts, 0, &base, None);

        // Check rollbacks at d = 1 (identical prompt), d = 2 (opposite parity `at`!), d = 5, d = 6 (max TAIL_BACK!)
        for d in [1usize, 2, 5, 6] {
            if d >= p { continue; }
            let at = p - d;
            let mut query = base[..at].to_vec();
            if d == 1 {
                query.push(base[p - 1]); // identical prompt
            } else {
                for j in 0..d { query.push(20_000 + (p * 13 + d * 7 + j) as u32); }
            }
            // Expected cold result for `query` on lane 0
            let (want_st, want_hb, want_gen) = if d == 1 {
                (st_cold.clone(), hb_cold, gen_cold.clone())
            } else {
                let hb = head_bits_cold(&mut ts, &query);
                let (s, _, g) = run_cold(&mut ts, 0, &query, None);
                (s, hb, g)
            };

            // Resumed TpOut head bits (greedy, lse, max_x) after poisoning lane 0
            let got_hb = head_bits_resumed(&mut ts, sid, at, &query);
            assert_eq!(got_hb, want_hb, "p={p} at={at} (odd={}): head TpOut bits differ: got {got_hb:?} vs cold {want_hb:?}", at & 1 == 1);

            // Resumed full TpServe Op::Restore -> Op::Truncate -> Op::Prefill (with tap copy!) -> Op::Start -> 3 Op::Round on poisoned lane 0
            poison_lane0_and_taps(&mut ts);
            ts.exec(Op::Restore { lane: 0, id: sid }, None).unwrap();
            ts.exec(Op::Truncate { lane: 0, pos: at as u32 }, None).unwrap();
            ts.exec(Op::Prefill { lane: 0, taps: true, total: query.len() as u32, first: true, full_at: 0, ahead: 0 }, Some(&query[at..])).unwrap();
            let got_st = state_hashes(&ts.m, 0, query.len());
            assert_eq!(got_st, want_st, "p={p} at={at} (odd={}): lane state hashes differ after resumed prefill", at & 1 == 1);

            let d0 = ts.exec(Op::Start { lane: 0, temp: 0.0, seed: 42, pen: [0.0, 0.0, 1.0], suppress_eos: false }, None).unwrap();
            let mut got_gen = vec![d0.tok];
            for _ in 0..3 {
                let rd = ts.exec(Op::Round { nl: 1, mask: 1 << 0, copy: false, sup_mask: 0, obs: None }, None).unwrap();
                got_gen.extend_from_slice(&rd.chains[0]);
            }
            assert_eq!(got_gen, want_gen, "p={p} at={at} (odd={}): DSpark speculative greedy continuation differs: got {got_gen:?} vs cold {want_gen:?}", at & 1 == 1);
            ts.exec(Op::Release { lane: 0 }, None).unwrap();
        }

        // For short prompt p = 47, also test extensions to 80 (< 128) and 150 (crosses TAP_ROWS = 128, t0 = 22 < 47)
        if p == 47 {
            for ext_total in [80usize, 150] {
                let mut ext = base.clone();
                ext.extend((0..(ext_total - p) as u32).map(|i| 30_000 + i));
                let want_hb = head_bits_cold(&mut ts, &ext);
                let (want_st, _, want_gen) = run_cold(&mut ts, 0, &ext, None);
                let got_hb = head_bits_resumed(&mut ts, sid, p, &ext);
                assert_eq!(got_hb, want_hb, "ext 47->{ext_total}: head bits differ");

                poison_lane0_and_taps(&mut ts);
                ts.exec(Op::Restore { lane: 0, id: sid }, None).unwrap();
                ts.exec(Op::Prefill { lane: 0, taps: true, total: ext_total as u32, first: true, full_at: 0, ahead: 0 }, Some(&ext[p..])).unwrap();
                assert_eq!(state_hashes(&ts.m, 0, ext_total), want_st, "ext 47->{ext_total}: state hashes differ");
                let d0 = ts.exec(Op::Start { lane: 0, temp: 0.0, seed: 42, pen: [0.0, 0.0, 1.0], suppress_eos: false }, None).unwrap();
                let mut got_gen = vec![d0.tok];
                for _ in 0..3 {
                    let rd = ts.exec(Op::Round { nl: 1, mask: 1 << 0, copy: false, sup_mask: 0, obs: None }, None).unwrap();
                    got_gen.extend_from_slice(&rd.chains[0]);
                }
                assert_eq!(got_gen, want_gen, "ext 47->{ext_total}: DSpark greedy continuation differs");
                ts.exec(Op::Release { lane: 0 }, None).unwrap();
            }
        }
        ts.exec(Op::Free { id: sid }, None).unwrap();
        crate::rprintln!("[pfx2-gpu] p={p:4}: cold == resumed bit-identical across rollbacks d=1,2,5,6 (state hashes, head TpOut bits, DSpark Start + 3 speculative rounds)");
    }

    // 3. End-to-end `engine::run_loop` + `TpForward` + `PrefixPool` over `[31, 46, 47, 87, 1174, 2324]` x 2 runs on GPU
    let suite_lens = [31usize, 46, 47, 87, 1174, 2324];
    let suite_prompts: Vec<Vec<u32>> = suite_lens.iter().enumerate().map(|(i, &n)| prompt_for(n, 9000 + i as u32)).collect();
    let (handle, parts) = engine::channel();
    let mut ecfg = EngineCfg::simple(2, vec![u32::MAX]); // never stop early so every request generates `max_new` tokens
    ecfg.prefix = Some((256 << 20, MIN_KEEP));
    ecfg.prefill_chunk = 1024;
    ecfg.grid_snaps = 1;
    ecfg.tail_merge = 512;

    std::thread::scope(|sc| {
        let client = sc.spawn(move || {
            let mut run1_toks = Vec::new();
            for (run, expect_hit) in [(1usize, false), (2usize, true)] {
                for (pi, pr) in suite_prompts.iter().enumerate() {
                    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
                    let mut samp = Sampling::default();
                    samp.temperature = 0.0;
                    assert!(handle.submit(Job {
                        prompt: pr.clone(), max_new: 12, min_tokens: 0, sampling: samp, seed: 77, ignore_eos: true,
                        starts_in_think: false, think_budget: None,
                        cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)), tx, score: false,
                        vl_types: Vec::new(), vl_images: Vec::new(), image_spans: Vec::new(),
                    }).is_ok());
                    let mut cached = None;
                    let mut toks = Vec::new();
                    let mut err = None;
                    while let Some(ev) = rx.blocking_recv() {
                        match ev {
                            Out::Admitted { cached: c } => cached = Some(c),
                            Out::Tok(t) => toks.push(t),
                            Out::Done { .. } => break,
                            Out::Failed(e) => { err = Some(e); break; }
                            Out::Scores(_) => {}
                        }
                    }
                    assert!(err.is_none(), "run {run} prompt #{pi}: {err:?}");
                    assert_eq!(toks.len(), 12, "run {run} prompt #{pi}");
                    if !expect_hit {
                        assert_eq!(cached, Some(0), "run 1 prompt #{pi} cold");
                        run1_toks.push(toks);
                    } else {
                        assert_eq!(cached, Some(pr.len() - 1), "run 2 prompt #{pi} (len {}) must hit tail_resume", pr.len());
                        assert_eq!(toks, run1_toks[pi], "run 2 prompt #{pi} (len {}) greedy tokens must equal cold run 1", pr.len());
                    }
                }
            }
            drop(handle); // closes channel so `run_loop` exits cleanly
        });
        let mut fwd = TpForward::new(&mut ts, cfg.vocab, so.max_seq);
        engine::run_loop(&mut fwd, ecfg, parts);
        client.join().expect("client thread");
    });
    crate::rprintln!("[pfx2-gpu] end-to-end engine::run_loop + TpForward + PrefixPool: 6/6 Run 2 tail_resume hits (short & long) with 100% bit-identical 12-token greedy speculative outputs! DSV41_PFX2_GPU_OK");
}
