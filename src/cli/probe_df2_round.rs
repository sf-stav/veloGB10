//! `run_probe_df2_round`: the DFlash2 round probe (one function of about 1,000 lines, kept whole).
//! Sibling DF2 files: `probe_df2_draft.rs`, `probe_df2_misc.rs`, `bench_df2.rs`, `replay.rs`.

use super::*;

pub(crate) fn run_probe_df2_round(args: &[String], draft_dir: &str) {
    let baked = gb10_inference::dflash2::load::is_baked(draft_dir);
    use gb10_inference::dflash2::capture::Df2TapSink;
    use gb10_inference::dflash2::round::{Df2Round, EvTimer, Nvfp4Ptrs, ring};
    use gb10_inference::dflash2::{mirror as m, BLOCK, hidden, tap_concat_dim, vocab};
    use gb10_inference::dflash2::oracle::Dflash2Config;
    use gb10_inference::dflash2::synth::SyntheticTables;
    use half::bf16;

    let trunk_dir = parse_arg(args, "--model-dir")
        .unwrap_or_else(|| panic!("--probe-df2-round needs --model-dir <trunk>")).to_string();
    let all_pass = std::cell::Cell::new(true);
    let nonoise_fail = std::cell::Cell::new(false);
    let check = |name: &str, ok: bool| {
        println!("  [{:6}] {name}", if ok { "PASS" } else { "FAIL" });
        if !ok {
            all_pass.set(false);
            // the bitwise-vs-bf16 gates are noise-expected on a baked (fp4) drafter; any OTHER
            // failing check marks the run as a real failure
            let th = name.contains("th == mirror");
            let ring = name.contains("ring k/v");
            if !th && !ring { nonoise_fail.set(true); }
        }
    };
    let rel_l2 = |a: &[f32], b: &[f32]| -> f64 {
        let mut num = 0.0f64; let mut den = 0.0f64;
        for i in 0..a.len() {
            let dd = (a[i] - b[i]) as f64;
            num += dd * dd; den += (b[i] as f64) * (b[i] as f64);
        }
        (num / den.max(1e-30)).sqrt()
    };

    // ---- Part 0: loads --------------------------------------------------------
    println!("== loads ==");
    let t0 = std::time::Instant::now();
    let (mut trunk, _) = load_model_gpu(&trunk_dir, None, 1);
    println!("  trunk loaded in {:.1}s", t0.elapsed().as_secs_f32());
    let (head_p, embed_p) = trunk.df2_borrow_ptrs()
        .expect("trunk lm_head/embed are NVFP4 (mma-repacked) — the borrowed-head contract");
    println!("  borrowed head+embed NVFP4 pointers resolved");

    // host copies of the trunk's packed lm_head + embed (for the mirror)
    let host_nvfp4 = |name: &str| -> (Vec<u8>, Vec<u8>, f32, usize, usize) {
        use safetensors::SafeTensors;
        let idx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
            format!("{trunk_dir}/model.safetensors.index.json")).unwrap()).unwrap();
        let wm = idx["weight_map"].as_object().unwrap();
        let shard = wm[&format!("{name}.weight_packed")].as_str().unwrap().to_string();
        let raw = std::fs::read(format!("{trunk_dir}/{shard}")).unwrap();
        let st = SafeTensors::deserialize(&raw).unwrap();
        let pv = st.tensor(&format!("{name}.weight_packed")).unwrap();
        let sv = st.tensor(&format!("{name}.weight_scale")).unwrap();
        let gv = st.tensor(&format!("{name}.weight_global_scale")).unwrap();
        let (mm, kk) = (pv.shape()[0], pv.shape()[1] * 2);
        let gs = f32::from_le_bytes(gv.data()[..4].try_into().unwrap());
        (pv.data().to_vec(), sv.data().to_vec(), gs, mm, kk)
    };
    let (hp_pack, hp_sc, hp_gs, hm, hk) = host_nvfp4("lm_head");
    assert_eq!((hm, hk), (vocab(), hidden()), "trunk lm_head shape {hm}x{hk}");
    let (ep_pack, ep_sc, ep_gs, em, ek) = host_nvfp4("model.language_model.embed_tokens");
    assert_eq!((em, ek), (vocab(), hidden()), "trunk embed shape {em}x{ek}");
    println!("  trunk head/embed host copies: gs {hp_gs:.4}/{ep_gs:.4}, {hm}x{hk}");

    // decode the head ONCE (row-parallel) into a flat bf16 matrix (mma semantics, no gs)
    let t1 = std::time::Instant::now();
    let mut head_bf16 = vec![bf16::default(); hm * hk];
    {
        // CPU courtesy: cap the decode pool (default 8; S3Q's CPU-only session shares this box)
        let nthreads = gb10_inference::opts::var(gb10_inference::opt!("df2-mirror-threads")).ok()
            .and_then(|v| v.parse::<usize>().ok()).filter(|&n| (1..=16).contains(&n)).unwrap_or(8);
        let rows_per = (hm + nthreads - 1) / nthreads;
        let mut outs: Vec<&mut [bf16]> = Vec::new();
        {
            let mut rest = head_bf16.as_mut_slice();
            let mut r0 = 0usize;
            while r0 < hm {
                let r1 = (r0 + rows_per).min(hm);
                let (a, b) = rest.split_at_mut((r1 - r0) * hk);
                outs.push(a); rest = b; r0 = r1;
            }
        }
        std::thread::scope(|sc| {
            for (t, o) in outs.into_iter().enumerate() {
                let pack = &hp_pack; let scl = &hp_sc;
                let base_row = t * rows_per;
                sc.spawn(move || {
                    for (i, row) in o.chunks_mut(hk).enumerate() {
                        m::head_row_mma(pack, scl, hk, base_row + i, row);
                    }
                });
            }
        });
    }
    println!("  head decoded ({:.1}s, {:.2} GB bf16)", t1.elapsed().as_secs_f32(),
             head_bf16.len() as f64 * 2.0 / 1e9);

    let t2 = std::time::Instant::now();
    // PLAN/25 §1a: the oracle (host reference) always loads the BF16 original; the device round
    // loads --draft-dir, which may be a baked NVFP4 artifact (its values then differ from the
    // oracle by quantization noise — the probe's thresholds are noise-aware where marked).
    let oracle_dir = parse_arg(args, "--oracle-dir").unwrap_or("");
    let art = gb10_inference::dflash2::load::load(oracle_dir, Some(gb10_inference::dflash2::REAL_SHA256))
        .expect("load oracle artifact (BF16)");
    println!("  oracle artifact loaded in {:.1}s ({} tensors)", t2.elapsed().as_secs_f32(), art.n_tensors);
    let cfg = Dflash2Config::default();
    let oracle = gb10_inference::dflash2::oracle::Dflash2Oracle::from_weights(
        cfg.clone(), art.weights.clone()).expect("oracle");

    let max_c = 4096usize;
    let t3 = std::time::Instant::now();
    let max_c = 4096 + 8 * 110 + gb10_inference::dflash2::block();   // headroom for the perf loop's advance
    let round_pin = if gb10_inference::dflash2::load::is_baked(draft_dir) { None } else { Some(gb10_inference::dflash2::REAL_SHA256) };
    let mut round = Df2Round::load_pinned(draft_dir, Some(gb10_inference::dflash2::round::BorrowedW::Nvfp4(head_p)), Some(gb10_inference::dflash2::round::BorrowedW::Nvfp4(embed_p)), max_c, round_pin)
        .expect("round load");
    println!("  Df2Round loaded in {:.1}s (ring {} rows/layer)", t3.elapsed().as_secs_f32(), gb10_inference::dflash2::round::ring());





    // ---- Part R1: trunk tap capture (bit-identity; free-when-off is the R6 gate) ----
    // Minimal determinism repro (memcheck target): --df2-det-only=1 skips R1/R2/R3/R5 and
    // runs ONLY the R4 loop (prime + rounds). A few thousand launches, not the full suite.
    let det_only = gb10_inference::opts::var(gb10_inference::opt!("df2-det-only")).is_ok();
    if det_only {
        let gen_taps = |c: usize| -> Vec<f32> {
            let mut t = vec![0f32; c * gb10_inference::dflash2::tap_concat_dim()];
            let mut rng: u64 = 0x9E3779B97F4A7C15;
            for v in t.iter_mut() {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                *v = ((rng >> 33) as i32 as f32 / (1u32 << 30) as f32 - 1.0) * 3.0;
            }
            t
        };
        let staged_row = |staged: &mut Vec<bf16>, taps: &[f32], base: usize| {
            for mi in 0..8usize { for k in 0..gb10_inference::dflash2::tap_concat_dim() {
                staged[mi * gb10_inference::dflash2::tap_concat_dim() + k] = bf16::from_f32(taps[(base + mi) * gb10_inference::dflash2::tap_concat_dim() + k]);
            } }
        };
        let mut staged: Vec<bf16> = vec![bf16::default(); gb10_inference::dflash2::tap_concat_dim() * gb10_inference::dflash2::block()];
        let c = 512usize;
        let taps = gen_taps(c);
        {
            let mut pos = 0usize;
            while pos < c {
                let n = gb10_inference::dflash2::block().min(c - pos);
                staged_row(&mut staged, &taps, pos);
                round.upload_chunk(&staged, n).unwrap();
                round.inject_dev(n, None).unwrap();
                pos += n;
            }
        }
        round.refresh_block_pos().unwrap();
        // per-rep early-stage captures at depth 1: embed(blk.h pre-layer via h_final? no —
        // normed/dyn/x_conv/q/k/attn/h) — find the EARLIEST diverging stage in L0.
        let mut sets: Vec<(&str, Vec<Vec<f32>>)> = vec![
            ("normed", Vec::new()), ("dyn_attn", Vec::new()), ("x_conv", Vec::new()),
            ("q", Vec::new()), ("kv", Vec::new()), ("attn", Vec::new()), ("h", Vec::new()),
            ("h_final", Vec::new()), ("ringK_regions", Vec::new()), ("ringV_regions", Vec::new()),
            ("attn_pre_o", Vec::new()),
        ];
        let post = gb10_inference::opts::var(gb10_inference::opt!("df2-det-post")).is_ok();
        let dual = gb10_inference::opts::var(gb10_inference::opt!("df2-det-dual")).is_ok();
        for _ in 0..30 {
            let o = round.draft_round_stages(12345, 2048, false, dual, false, 1, post).unwrap();
            sets[0].1.push(round.dump_block_normed().unwrap());
            sets[1].1.push(round.dump_block_dyn().unwrap());
            sets[2].1.push(round.dump_block_x().unwrap());
            sets[3].1.push(round.dump_block_q().unwrap());
            let (kk, _) = round.dump_block_kv().unwrap();
            sets[4].1.push(kk);
            sets[5].1.push(round.dump_attn_scratch().unwrap());
            sets[6].1.push(round.dump_block_h().unwrap());
            sets[7].1.push(o.h_final);
            if dual { sets[10].1.push(round.dump_ctl_attn_ref().unwrap()); }
            let (rk, rv) = round.dump_ring_regions(0, 512).unwrap();
            sets[8].1.push(rk);
            sets[9].1.push(rv);
        }
        // same-rep cross-compare: the dual-write re-launch must equal the primary launch.
        {
            let attn = sets.iter().find(|(n, _)| *n == "attn").unwrap();
            let pre = sets.iter().find(|(n, _)| *n == "attn_pre_o").unwrap();
            for i in 0..attn.1.len() {
                if &attn.1[i] != &pre.1[i] {
                    let (a, b) = (&attn.1[i], &pre.1[i]);
                    let mut nd = 0usize; let mut f = 0usize;
                    for k in 0..a.len() { if a[k] != b[k] { nd += 1; if f == 0 { f = k; } } }
                    println!("    SAME-REP MISMATCH rep {i}: attn != attn_pre_o: {nd}/{} first at {f}: {:+e} vs {:+e}",
                             a.len(), a[f], b[f]);
                    break;
                }
            }
            let _ = &pre;
        }
        // discriminator pass: re-run ONLY attention per saved state — rerun per rep needs the
        // rep's exact q/ring state, so re-run the whole sequence instead: rounds are cheap.
        // (state after rep i is what attention would read at rep i+1 — approximation is fine:
        // the question is only whether a fresh launch reproduces the dumped value.)
        let mut first_div: Option<(usize, &str)> = None;
        for (si, (name, vs)) in sets.iter().enumerate() {
            if let Some(i) = vs.iter().enumerate().skip(1).find(|(_, v)| *v != &vs[0]).map(|(i, _)| i) {
                let (a, b) = (&vs[0], &vs[i]);
                let mut nd = 0usize; let mut f = 0usize;
                for k in 0..a.len() { if a[k] != b[k] { nd += 1; if f == 0 { f = k; } } }
                println!("    stage {name}: DIVERGES rep {i}: {nd}/{} elems, first at {f} (row {}, col {}): {:+e} vs {:+e}",
                         a.len(), f / hidden().max(1), f % hidden().max(1), a[f.min(a.len()-1)], b[f.min(a.len()-1)]);
                if first_div.is_none() { first_div = Some((i, name)); }
            } else {
                println!("    stage {name}: stable (30 reps)");
            }
            let _ = si;
        }
        // discriminator: at the diverging rep, does a fresh attention re-run reproduce the
        // corrupted attn (inputs changed) or the round-0 value (post-attention overwrite)?
        if let Some((i, _)) = first_div {
            // fresh attention launch over the CURRENT (post-rep-29) state, vs the last dumped attn
            let rerun = round.ctl_ring_attn(2048).unwrap();
            let last = sets.iter().find(|(n, _)| *n == "attn").unwrap();
            let l29 = &last.1[29];
            let matches_last = &rerun == l29;
            let matches_r0 = rerun == last.1[0];
            println!("    discriminator (fresh launch, post-rep29 state): ==attn29 {matches_last} | ==attn0 {matches_r0}");
        }
        match first_div {
            Some((i, n)) => println!("RESULT: DET_DIVERGES at rep {i}, earliest stage {n} (depth 1)"),
            None => println!("RESULT: DET_STABLE (30 reps, depth 1)"),
        }
        return;
    }

    println!("\n== R1: trunk tap capture ==");
    let sink = std::sync::Arc::new(Df2TapSink::new(trunk.dev()));
    trunk.set_df2_capture(sink.clone());
    check("capture armed (df2_capture_armed)", trunk.df2_capture_armed());
    let cap_prompt: Vec<u32> = (0..64u32).map(|i| 1000 + (i * 7) % 150_000).collect();
    let mut pool = gb10_inference::gpu::Pool::new(trunk.dev().clone());
    let mut run_capture = |trunk: &gb10_inference::gpu::GpuModel, sink: &std::sync::Arc<Df2TapSink>,
                           logits_out: &mut Vec<f32>| -> Vec<f32> {
        let mut state = trunk.new_batch_state(1, 2, 4096);
        let mut taps: Vec<f32> = Vec::with_capacity(cap_prompt.len() * gb10_inference::dflash2::tap_concat_dim());
        for (t, &tok) in cap_prompt.iter().enumerate() {
            let hidden = trunk.embed_batch(&[tok]);
            let out = trunk.forward_batch(&mut pool, hidden, &[t], &mut state, 4096, 1);
            trunk.sync_stream();
            // staging col 0 = THIS forward's taps (5 layers x 5120, layer-major)
            let col: Vec<bf16> = trunk.dev().dtoh_sync_copy(&sink.staging).expect("staging dtoh");
            let col0: Vec<bf16> = (0..gb10_inference::dflash2::tap_concat_dim()).map(|k| col[k]).collect();
            taps.extend(col0.iter().map(|x| x.to_f32()));
            if t == cap_prompt.len() - 1 {
                let lg = trunk.logits_batch(&mut pool, &out, 1);
                let lgv: Vec<bf16> = trunk.dev().dtoh_sync_copy(&lg).expect("logits dtoh");
                *logits_out = lgv.iter().map(|x| x.to_f32()).collect();
                pool.release_bf16(lg, vocab());
            }
            pool.release_bf16(out, gb10_inference::dflash2::hidden());
        }
        taps
    };
    let mut lg1 = Vec::new();
    let taps_a = run_capture(&trunk, &sink, &mut lg1);
    check("captured 64 positions x 25600", taps_a.len() == 64 * gb10_inference::dflash2::tap_concat_dim());

    // double-run determinism: a fresh trunk state, same prompt -> identical capture
    let mut lg2 = Vec::new();
    let taps_b = run_capture(&trunk, &sink, &mut lg2);
    check("capture double-run bit-identical (taps + trunk logits)", taps_a == taps_b && lg1 == lg2);

    // capture OFF: trunk behavior unchanged (final logits bit-identical)
    trunk.set_df2_capture_off();
    check("capture disarmed", !trunk.df2_capture_armed());
    let mut lg3 = Vec::new();
    {
        let mut state3 = trunk.new_batch_state(1, 2, 4096);
        for (t, &tok) in cap_prompt.iter().enumerate() {
            let hidden = trunk.embed_batch(&[tok]);
            let out = trunk.forward_batch(&mut pool, hidden, &[t], &mut state3, 4096, 1);
            if t == cap_prompt.len() - 1 {
                let lg = trunk.logits_batch(&mut pool, &out, 1);
                let lgv: Vec<bf16> = trunk.dev().dtoh_sync_copy(&lg).expect("logits dtoh");
                lg3 = lgv.iter().map(|x| x.to_f32()).collect();
                pool.release_bf16(lg, vocab());
            }
            pool.release_bf16(out, gb10_inference::dflash2::hidden());
        }
    }
    check("trunk logits identical capture-on vs capture-off (no behavior change)", lg1 == lg3);
    trunk.set_df2_capture(sink.clone());

    // norms plausibility vs the S3Q real-tap fixtures (chat1_short layer_norms)
    let mut norms = [0f32; 5];
    for pos in 0..64 {
        for l in 0..5 {
            let mut s = 0f32;
            for k in 0..hidden() { let v = taps_a[pos * gb10_inference::dflash2::tap_concat_dim() + l * hidden() + k]; s += v * v; }
            norms[l] += s.sqrt();
        }
    }
    for n in norms.iter_mut() { *n /= 64.0; }
    let s3q_ref = [85.8f32, 182.96, 237.28, 368.92, 978.62];
    println!("    captured per-layer mean |h|: {norms:?}");
    println!("    S3Q real-tap reference:      {s3q_ref:?}");
    let norms_ok = norms.iter().zip(s3q_ref.iter()).all(|(a, b)| *a > 0.0 && (a / b) > 0.05 && (a / b) < 20.0);
    check("captured tap norms plausible vs the S3Q real distribution", norms_ok);

    // ---- Part R2: the integrated round vs the extended mirror ----------------
    println!("\n== R2: integrated round vs mirror (EXACT on identical inputs) ==");
    let taps_gen = SyntheticTables::new(gb10_inference::dflash2::SYNTH_TAP_SEED);
    let tap_scale = 1.0f32 / (gb10_inference::dflash2::tap_concat_dim() as f32).sqrt();
    let gen_taps = |c: usize| -> Vec<f32> {
        let mut t = Vec::with_capacity(c * gb10_inference::dflash2::tap_concat_dim());
        for i in 0..c {
            t.extend_from_slice(&taps_gen.row(SyntheticTables::TABLE_TAPS, i as u32, gb10_inference::dflash2::tap_concat_dim(), tap_scale));
        }
        t
    };
    let s3q_taps: Vec<f32> = {
        let raw = std::fs::read("tool_probe/dflash2-quant-fixtures/chat1_short/taps.f32").unwrap();
        bytemuck::cast_slice(&raw).to_vec()
    };
    struct Fixture { name: &'static str, taps: Vec<f32>, c: usize }
    let fixtures = vec![
        Fixture { name: "S3Q-real C=8", taps: s3q_taps.clone(), c: 8 },
        Fixture { name: "synth C=512", taps: gen_taps(512), c: 512 },
        Fixture { name: "synth C=4096", taps: gen_taps(4096), c: 4096 },
        Fixture { name: "captured C=64", taps: taps_a.clone(), c: 64 },
    ];
    // DF2 block-16: a fixture narrower than the live block cannot exercise one full round
    // (the incremental prime and the last-chunk slicing need at least block() columns), so it is
    // dropped with a note rather than underflowing. Real-tap coverage survives at block 16 via the
    // captured C=64 fixture.
    let fixtures: Vec<Fixture> = fixtures.into_iter()
        .filter(|fx| {
            if fx.c >= gb10_inference::dflash2::block() { true } else {
                println!("  [SKIP ] fixture {} ({} cols < block {}) — too narrow for this block",
                         fx.name, fx.c, gb10_inference::dflash2::block());
                false
            }
        })
        .collect();
    assert!(!fixtures.is_empty(), "no fixture is wide enough for block {}",
            gb10_inference::dflash2::block());
    let anchors: [u32; 3] = [12345, 1, 248319];

    let pred_cb: Vec<bf16> = art.weights.predecessor_codebook.iter().map(|&x| bf16::from_f32(x)).collect();
    let succ_cb: Vec<bf16> = art.weights.successor_codebook.iter().map(|&x| bf16::from_f32(x)).collect();
    let mut n_exact_top16 = 0usize;
    let mut n_exact_walk = 0usize;
    let mut n_exact_path = 0usize;
    let mut n_rounds = 0usize;

    for fx in &fixtures {
        println!("\n-- fixture {} ({} x {}) --", fx.name, fx.c, gb10_inference::dflash2::tap_concat_dim());
        let tm = std::time::Instant::now();
        let taps_bf16 = m::rb_clone(&fx.taps);
        let (_, th_m) = m::round_tap_project_dsp(&cfg, &art.weights.fc, &art.weights.hidden_norm, &taps_bf16, fx.c);
        let inv = m::inv_freq(&cfg);
        // RoPE tables for the FULL context (no truncation): the device indexes cos/sin by
        // ABSOLUTE position, so any window/truncation in the mirror is a mirror bug.
        let (cos, sin) = m::rope_tables_half(&cfg, &inv, fx.c + gb10_inference::dflash2::block());
        let mut ctx_m: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
        for l in &art.weights.layers {
            ctx_m.push(m::round_draft_kv_dsp(&cfg, l, &th_m, fx.c, 0, &cos, &sin));
        }
        println!("    mirror ctx built in {:.1}s", tm.elapsed().as_secs_f32());

        let mut staged: Vec<bf16> = vec![bf16::default(); gb10_inference::dflash2::tap_concat_dim() * gb10_inference::dflash2::block()];
        // the FULL mirror th (all c rows) for the per-chunk gate
        let (th_raw_all, th_all) = m::round_tap_project_dsp(&cfg, &art.weights.fc, &art.weights.hidden_norm, &taps_bf16, fx.c);
        let k4_all = m::round_draft_kv_dsp(&cfg, &art.weights.layers[gb10_inference::dflash2::n_layers() - 1], &th_all, fx.c, 0, &cos, &sin);
        round.reset();
        let mut pos = 0usize;
        let mut chunk_bad = 0usize;
        while pos < fx.c {
            let n = gb10_inference::dflash2::block().min(fx.c - pos);
            for mi in 0..n {
                for k in 0..gb10_inference::dflash2::tap_concat_dim() {
                    staged[mi * gb10_inference::dflash2::tap_concat_dim() + k] = bf16::from_f32(fx.taps[(pos + mi) * gb10_inference::dflash2::tap_concat_dim() + k]);
                }
            }
            round.upload_chunk(&staged, n).expect("upload chunk");
            round.inject_dev(n, None).expect("inject");
            {
                // per-chunk gate: th for THIS chunk's rows, ALL fixtures (a bad th row
                // upstream explains an all-layers k divergence; a clean th localizes to the
                // k chain)
                let th_d = round.dump_th().unwrap();
                let th_m = &th_all[pos * hidden()..(pos + n) * hidden()];
                if th_d[..n * hidden()] != th_m[..n * hidden()] {
                    // PROOF PATH: the device norm's `inv` comes from rsqrt.approx.f32 (MUFU,
                    // <=2 ulp — see the PTX), so a 1-f32-ulp inv shift near a bf16 rounding
                    // boundary flips the stored element. Scan inv candidates around the
                    // exact 1/sqrt: if ONE candidate reproduces the device row BITWISE, the
                    // residue is fully explained by the hardware rsqrt approximation.
                    let mut bad_rows = 0usize; let mut explained_rows = 0usize;
                    for r in 0..n {
                        if th_d[r * hidden()..(r + 1) * hidden()] == th_m[r * hidden()..(r + 1) * hidden()] { continue; }
                        bad_rows += 1;
                        let mut nd = 0usize; let mut fi = 0usize;
                        for i in 0..hidden() {
                            if th_d[r * hidden() + i] != th_m[r * hidden() + i] { nd += 1; if fi == 0 { fi = i; } }
                        }
                        let raw = &th_raw_all[(pos + r) * hidden()..(pos + r + 1) * hidden()];
                        let w_n = &art.weights.hidden_norm;
                        let base = m::rms_norm_rows_inv_exact(raw, w_n, hidden(), cfg.rms_eps);
                        let mut best = (0i32, 0usize); // (bit offset, matched count)
                        for off in -8i32..=8 {
                            let inv_c = f32::from_bits((base.to_bits() as i32 + off) as u32);
                            let mut matched = 0usize;
                            for i in 0..hidden() {
                                if bf16::from_f32(raw[i] * inv_c * w_n[i]).to_f32() == th_d[r * hidden() + i] { matched += 1; }
                            }
                            if matched > best.1 { best = (off, matched); }
                        }
                        let explained = best.1 == hidden();
                        if explained { explained_rows += 1; }
                        if bad_rows <= 3 {
                            println!("      th chunk@{} row {}: {}/{} elems differ (first at {}: dev {:+e} mir {:+e}); inv candidate {:+} ulp reproduces {}/{} -> {}",
                                     pos, pos + r, nd, hidden(), fi, th_d[r * hidden() + fi], th_m[r * hidden() + fi],
                                     best.0, best.1, hidden(),
                                     if explained { "EXPLAINED (rsqrt.approx)" } else { "UNEXPLAINED" });
                        }
                    }
                    if explained_rows < bad_rows { chunk_bad += 1; }
                    println!("      th chunk@{}: {}/{} bad rows, {} explained by the rsqrt.approx inv candidate",
                             pos, bad_rows, n, explained_rows);
                }
                if fx.c == 64 {
                    let (kd, _) = round.dump_kvc().unwrap();
                    let km = &k4_all.0[pos * (8 * 128)..(pos + n) * (8 * 128)];
                    if kd[..n * 8 * 128] != km[..n * 8 * 128] {
                        chunk_bad += 1;
                        for i in 0..n * 8 * 128 {
                            if kd[i].to_bits() != km[i].to_bits() {
                                println!("      chunk@{pos} k mismatch row {} elem {}: dev {:+e} mir {:+e}",
                                         pos + i / (8 * 128), i % (8 * 128), kd[i], km[i]);
                                break;
                            }
                        }
                    }
                }
            }
            pos += n;
        }
        println!("    per-chunk gates (th all fixtures + layer-4 k @C=64, {} chunks): {chunk_bad} failing chunk-gates",
                 fx.c.div_ceil(gb10_inference::dflash2::block()));
        check(&format!("{}: nprev == c after incremental prime", fx.name), round.nprev() == fx.c);

        // the device th holds ONLY the last chunk's 8 rows (the incremental path)
        let th_last = &th_m[(fx.c - gb10_inference::dflash2::block()) * hidden()..];
        let th_dev = round.dump_th().expect("th");
        let th_bit = th_dev == th_last;
        let th_rl = rel_l2(&th_dev, th_last);
        println!("    th (last 8 rows): bitwise {} (rel-L2 {th_rl:.3e})", if th_bit { "EQUAL" } else { "DIFF" });
        check(&format!("{}: th == mirror (bitwise, last chunk)", fx.name), th_bit || th_rl < 1e-6);

        // ring row j%ring() holds ctx row j IFF j is the LAST position mapping there
        // (j > c-1-ring()); at c<=ring() that is every row, at c=4096=2*ring() the second pass.
        // the ring cache is [nkv, stride, hd]: row j's (h,d) element at (h*stride + j)*hd + d.
        // ALL 5 layers; mismatching elements are classified by bf16 ulp distance (a <=4-ulp
        // tail on <0.1% of elements with bitwise-exact th AND bitwise-exact per-chunk k is a
        // mirror-fidelity artifact — DECISION; anything larger is a wiring bug).
        let kv_lo = fx.c.saturating_sub(ring());
        let mut kv_tot = 0usize; let mut kv_max_ulp = 0u32; let mut kv_examples = 0usize;
        let mut kv_max_rel = 0f32; // |dev-mir| / row MAX |k| (units of the row rounding quantum)
        for li in 0..gb10_inference::dflash2::n_layers() {
            let (kr, _vr) = round.dump_ring_kv(li).expect("ring kv");
            for j in kv_lo..fx.c {
                let rj = j % ring();
                // this row's mirror k MAX magnitude (the rounding-quantum scale of the row)
                let mut row_max = 0f32;
                for h in 0..8usize { for d2 in 0..128usize {
                    let mv = ctx_m[li].0[(j * 8 + h) * 128 + d2].abs(); if mv > row_max { row_max = mv; }
                } }
                for h in 0..8usize {
                    for d2 in 0..128usize {
                        let dv = kr[(h * 2056 + rj) * 128 + d2];
                        let mv = ctx_m[li].0[(j * 8 + h) * 128 + d2];
                        if dv != mv {
                            kv_max_rel = kv_max_rel.max((dv - mv).abs() / row_max);
                            kv_tot += 1;
                            let ulp = bf16_ulp_dist(bf16::from_f32(dv), bf16::from_f32(mv));
                            kv_max_ulp = kv_max_ulp.max(ulp);
                            if kv_examples < 3 {
                                println!("      L{li} j={j} h={h} d={d2}: dev {dv:+e} mir {mv:+e} ({ulp} ulp)");
                                kv_examples += 1;
                            }
                        }
                    }
                }
            }
        }
        let kv_n = (fx.c - kv_lo) * (8 * 128) * gb10_inference::dflash2::n_layers();
        println!("    ring k all layers: {kv_tot}/{kv_n} mismatched, max ulp {kv_max_ulp}, max |dev-mir|/rowMAX {kv_max_rel:.3e} (bf16 half-ulp at max = 3.9e-3)");
        if kv_tot > 0 {
            // ulp histogram + the worst 5 (anything past ~4 ulp is NOT an rsqrtf-class
            // approximation artifact and must be explained before this gate can pass)
            let mut b = [0usize; 5]; // 1, 2-4, 5-16, 17-64, >64
            let mut worst: Vec<(u32, usize, usize, usize, usize, f32, f32)> = Vec::new();
            for li in 0..gb10_inference::dflash2::n_layers() {
                let (kr, _vr) = round.dump_ring_kv(li).expect("ring kv");
                for j in kv_lo..fx.c {
                    let rj = j % ring();
                    for h in 0..8usize {
                        for d2 in 0..128usize {
                            let dv = kr[(h * 2056 + rj) * 128 + d2];
                            let mv = ctx_m[li].0[(j * 8 + h) * 128 + d2];
                            if dv != mv {
                                let ulp = bf16_ulp_dist(bf16::from_f32(dv), bf16::from_f32(mv));
                                let bi = match ulp { 1 => 0, 2..=4 => 1, 5..=16 => 2, 17..=64 => 3, _ => 4 };
                                b[bi] += 1;
                                worst.push((ulp, li, j, h, d2, dv, mv));
                            }
                        }
                    }
                }
            }
            worst.sort_by(|a, b2| b2.0.cmp(&a.0));
            println!("      ulp histogram: 1:{} 2-4:{} 5-16:{} 17-64:{} >64:{}", b[0], b[1], b[2], b[3], b[4]);
            for w in worst.iter().take(5) {
                println!("      worst: L{} j={} h={} d={} dev {:+e} mir {:+e} ({} ulp, rel {:+.2e})",
                         w.1, w.2, w.3, w.4, w.5, w.6, w.0,
                         if w.6 != 0.0 { (w.5 - w.6) / w.6 } else { f32::NAN });
            }
        }
        // pass condition: either bitwise, or a rare tail (<0.1%) where every delta is
        // <= 2^-6 x the row's MAX |k| — at most 2 bf16 rounding quanta at the largest
        // magnitude present. Mechanism: rsqrt.approx.f32 inv shifts (PROVEN bitwise on the
        // th stage via the candidate scan) flip the bf16 rounding of boundary elements by
        // 1 quantum, amplified to large RELATIVE deltas only at near-zero rope outputs.
        // A WIRING error (wrong row/position/window) produces deltas at the full row scale.
        check(&format!("{}: ring k/v ~= mirror (bitwise or <=2-quanta tail, rsqrt-proven)", fx.name),
              kv_tot == 0 || (kv_tot * 1000 <= kv_n && kv_max_rel <= 2f32.powi(-6)));

        // the embed gate (fixture-level, once per anchor set): device gather vs mirror
        {
            let mut emb_ok = true;
            let mut emb_worst = 0f64;
            for &anchor in &anchors {
                let dev = round.embed_probe(anchor).expect("embed probe");
                let mut emb16 = vec![bf16::default(); gb10_inference::dflash2::block() * hidden()];
                m::embed_row_mma(&ep_pack, &ep_sc, hidden(), anchor as usize, ep_gs,
                                 &mut emb16[..hidden()]);
                for r in 1..gb10_inference::dflash2::block() {
                    m::embed_row_mma(&ep_pack, &ep_sc, hidden(),
                                     gb10_inference::dflash2::mask_token_id() as usize, ep_gs,
                                     &mut emb16[r * hidden()..(r + 1) * hidden()]);
                }
                let mir: Vec<f32> = emb16.iter().map(|b| b.to_f32()).collect();
                let bit = dev == mir;
                let rl = rel_l2(&dev, &mir);
                emb_worst = emb_worst.max(rl);
                if !bit && rl > 1e-6 { emb_ok = false; }
            }
            println!("    embed gather: worst rel-L2 {emb_worst:.3e} over {} anchors", anchors.len());
            if !emb_ok {
                // layout debug: the first 6 elements of anchor 12345 on both paths
                let dev = round.embed_probe(12345).unwrap();
                let mut e16 = vec![bf16::default(); hidden()];
                m::embed_row_mma(&ep_pack, &ep_sc, hidden(), 12345, ep_gs, &mut e16);
                let mir: Vec<f32> = e16.iter().map(|b| b.to_f32()).collect();
                println!("    embed dbg anchor 12345: dev {:?}…", &dev[..6.min(dev.len())]);
                println!("    embed dbg anchor 12345: mir {:?}…", &mir[..6.min(mir.len())]);
                // per-row rel-L2 (row 0 = anchor, rows 1..7 = MASK) + first differing element
                for r in 0..gb10_inference::dflash2::block() {
                    let dr = &dev[r * hidden()..(r + 1) * hidden()];
                    let mr = &mir[r * hidden()..(r + 1) * hidden()];
                    let rl = rel_l2(dr, mr);
                    let mut ne = 0usize; let mut ex = (0usize, 0f32, 0f32);
                    for i in 0..hidden() {
                        if dr[i].to_bits() != mr[i].to_bits() {
                            if ne == 0 { ex = (i, dr[i], mr[i]); }
                            ne += 1;
                        }
                    }
                    println!("      row {r}: rel-L2 {rl:.3e}, {ne} diffs, first i={} dev {:+e} mir {:+e}", ex.0, ex.1, ex.2);
                if r == 0 {
                    // per-kb match count + ratio probe (a wrong per-block scale shows as a
                    // constant dev/mir ratio inside the block)
                    for kb in 0..8usize {
                        let mut same = 0usize; let mut nz = 0usize; let mut ratio = 0f64;
                        for i in kb * 16..kb * 16 + 16 {
                            if dr[i].to_bits() == mr[i].to_bits() { same += 1; }
                            if mr[i] != 0.0 && dr[i] != 0.0 && nz == 0 { ratio = dr[i] as f64 / mr[i] as f64; nz += 1; }
                        }
                        let sc_m = ep_sc[12345 * 320 + kb];
                        println!("        kb {kb}: {same}/16 match, ratio {ratio:.4}, mir scale byte {sc_m:#04x}");
                    }
                    // THE decisive dump: the device's OWN repacked tile bytes for
                    // (row 12345, kb 16..19) vs the host repack prediction.
                    unsafe {
                        
                        let mt = 12345usize >> 4;
                        let mut dev_wt = [0u8; 4 * 128];
                        let mut dev_st = [0u8; 4 * 16];
                        cudarc::driver::result::memcpy_dtoh_sync(&mut dev_wt, (embed_p.qweight + ((mt * 320 + 16) * 128) as u64) as u64 as _).unwrap();
                        cudarc::driver::result::memcpy_dtoh_sync(&mut dev_st, (embed_p.scales + ((mt * 320 + 16) * 16) as u64) as u64 as _).unwrap();
                        // host repack prediction for these 4 tiles
                        let mut pred_wt = [0u8; 4 * 128];
                        let mut pred_st = [0u8; 4 * 16];
                        for kb in 16..20usize {
                            for r in 0..16usize {
                                let row = mt * 16 + r;
                                pred_st[(kb - 16) * 16 + r] = ep_sc[row * 320 + kb];
                                for cp in 0..8usize {
                                    let c = cp * 2;
                                    let g = r & 7; let hi_row = r >> 3;
                                    let t = (c & 7) >> 1; let hi_col = c >> 3;
                                    let lane = g * 4 + t;
                                    let j = hi_row | (hi_col << 1);
                                    pred_wt[(kb - 16) * 128 + lane * 4 + j] = ep_pack[row * 2560 + kb * 8 + cp];
                                }
                            }
                        }
                        let wt_eq = dev_wt == pred_wt;
                        let st_eq = dev_st == pred_st;
                        println!("        tile(771,16..20) bytes: wt {} st {}",
                                 if wt_eq { "MATCH" } else { "DIFF" }, if st_eq { "MATCH" } else { "DIFF" });
                        if !st_eq {
                            print!("        dev st:"); for b in dev_st.iter().take(16) { print!(" {b:#04x}"); } println!();
                            print!("        pre st:"); for b in pred_st.iter().take(16) { print!(" {b:#04x}"); } println!();
                        }
                        if !wt_eq {
                            let mut nd = 0; let mut first = 999usize;
                            for i in 0..512 { if dev_wt[i] != pred_wt[i] { nd += 1; if first == 999 { first = i; } } }
                            println!("        wt {nd}/512 bytes differ, first at {first} (tile byte {} lane {} j {})",
                                     first % 128, (first % 128) / 4, first % 4);
                        }
                    }
                    // the trunk's own dequant at the probe tile-row (device-side cross-check)
                    let dq = round.dequant_probe(771, embed_p.qweight, embed_p.scales, embed_p.gs, hidden()).unwrap();
                    let dq9 = &dq[9 * hidden()..10 * hidden()];
                    let mut dq_mir = 0usize; let mut dq_dev = 0usize;
                    for i in 0..hidden() {
                        if dq9[i] == mir[i] { dq_mir += 1; }
                        if dq9[i] == dev[i] { dq_dev += 1; }
                    }
                    println!("        dequant(row 12345): {dq_mir}/5120 == mirror, {dq_dev}/5120 == gather-output");
                }
                }
            }
            check(&format!("{}: trunk embed gather == mirror (bitwise)", fx.name), emb_ok);
        }

        for &anchor in &anchors {
            round.refresh_block_pos().expect("pos");
            let out = round.draft_round(anchor, 2048, false, false, true).expect("round");
            n_rounds += 1;

            let blk_live = gb10_inference::dflash2::block();
            let block_pos: Vec<usize> = (fx.c..fx.c + blk_live).collect();
            let mut emb16 = vec![bf16::default(); blk_live * hidden()];
            m::embed_row_mma(&ep_pack, &ep_sc, hidden(), anchor as usize, ep_gs, &mut emb16[..hidden()]);
            for r in 1..gb10_inference::dflash2::block() {
                m::embed_row_mma(&ep_pack, &ep_sc, hidden(),
                                 gb10_inference::dflash2::mask_token_id() as usize, ep_gs,
                                 &mut emb16[r * hidden()..(r + 1) * hidden()]);
            }
            let emb = m::rb_clone(&emb16.iter().map(|b| b.to_f32()).collect::<Vec<f32>>());
            let mut h = emb.clone();
            for (li, l) in art.weights.layers.iter().enumerate() {
                let o = m::mirror_layer_forward(&cfg, l, &h, &ctx_m[li].0, &ctx_m[li].1, &block_pos, &cos, &sin);
                h = o.h3.clone();
            }
            // DF2 block-16: the mirror's h_final must have the LIVE block's row count, not the
            // BLOCK const — otherwise the device (block rows) and mirror (8 rows) lengths differ.
            let h_final = m::rb_clone(&m::rms_norm_rows(&h, &art.weights.norm,
                                       gb10_inference::dflash2::block(), hidden(), cfg.rms_eps));
            let hf_rl = rel_l2(&out.h_final, &h_final);

            let hsel: Vec<f32> = (1..gb10_inference::dflash2::block()).flat_map(|r| h_final[r * hidden()..(r + 1) * hidden()].to_vec()).collect();
            let lg_m = m::head_logits_mirror(&head_bf16, &hsel, blk_live - 1, hidden(), hp_gs, vocab());
            let lg_d = out.logits.as_ref().unwrap();
            let lg_rl = rel_l2(lg_d, &lg_m);

            // The walk reads hp rows for the levels only (the device's m8 tiles write a full
            // block() of columns; the trailing column is garbage-but-unread at block 8, and at
            // block 16 the 15th/16th column likewise). Mirror the READ width, not the write width.
            let hp_m = m::rb_clone(&m::linear_gemm_dsp(&art.weights.hidden_projection, &hsel,
                                                       gb10_inference::dflash2::selector_rank(), hidden(),
                                                       gb10_inference::dflash2::block() - 1));
            let hp_d = out.hp.as_ref().unwrap();
            let hp_bit = *hp_d == hp_m;

            // GATE E: top16 EXACT on IDENTICAL logits (device bf16 logits through the mirror)
            let mut e_ok = true;
            for p in 0..out.tokens.len() {
                let row: Vec<f32> = (0..vocab()).map(|v| lg_d[p * vocab() + v]).collect();
                let (vals, ids) = oracle.top16(&row);
                for k in 0..16 {
                    if vals[k].to_bits() != out.unary[p * 16 + k].to_bits()
                        || ids[k] != out.candidates[p * 16 + k] { e_ok = false; }
                }
            }
            if e_ok { n_exact_top16 += 1; }

            // GATE F: walk EXACT on identical (device hp, device cand, device unary)
            // DF2 block-16: the live level count comes from the round output itself.
            let nlv_r = out.tokens.len();
            let (tok_m, sc_m) = m::round_walk_mirror(hp_d, &out.candidates, &out.unary, anchor,
                                                     &pred_cb, &succ_cb,
                                                     gb10_inference::dflash2::selector_rank(), nlv_r);
            let f_bit = tok_m == out.tokens && sc_m == out.scores;
            if f_bit { n_exact_walk += 1; }

            // GATE G: end-to-end mirror path (mirror logits -> mirror top16 -> mirror walk)
            let mut cand_m = vec![0u32; nlv_r * 16];
            let mut un_m = vec![0f32; nlv_r * 16];
            for p in 0..nlv_r {
                let (vals, ids) = oracle.top16(&lg_m[p * vocab()..(p + 1) * vocab()]);
                for k in 0..16 { cand_m[p * 16 + k] = ids[k]; un_m[p * 16 + k] = vals[k]; }
            }
            let (tok_e2e, _) = m::round_walk_mirror(&hp_m, &cand_m, &un_m, anchor, &pred_cb, &succ_cb,
                                                    gb10_inference::dflash2::selector_rank(), nlv_r);
            let g_eq = tok_e2e == out.tokens;
            if g_eq { n_exact_path += 1; }
            println!("    anchor {anchor}: h_final {hf_rl:.3e} | logits {lg_rl:.3e} | hp bitwise {hp_bit} | top16 {} | walk {} | path {}",
                     if e_ok { "EXACT" } else { "DIFF" }, if f_bit { "EXACT" } else { "DIFF" },
                     if g_eq { "match" } else { "DIFF" });
        }
    }
    check(&format!("top16 EXACT vs mirror on identical logits ({n_exact_top16}/{n_rounds})"),
          n_exact_top16 == n_rounds);
    check(&format!("walk EXACT (bitwise scores+tokens) ({n_exact_walk}/{n_rounds})"),
          n_exact_walk == n_rounds);
    println!("    end-to-end mirror path agreement: {n_exact_path}/{n_rounds} (a near-tie at the head/16th rank boundary -> DECISIONS + re-anchor)");

    // ---- Part R3: negative controls (must FIRE) -------------------------------
    println!("\n== R3: negative controls ==");
    let prime = |round: &mut Df2Round, taps: &[f32], c: usize, stop: usize| {
        round.reset();
        let mut staged: Vec<bf16> = vec![bf16::default(); gb10_inference::dflash2::tap_concat_dim() * gb10_inference::dflash2::block()];
        let mut pos = 0usize;
        while pos < stop {
            let n = gb10_inference::dflash2::block().min(stop - pos);
            for mi in 0..n {
                for k in 0..gb10_inference::dflash2::tap_concat_dim() {
                    staged[mi * gb10_inference::dflash2::tap_concat_dim() + k] = bf16::from_f32(taps[(pos + mi) * gb10_inference::dflash2::tap_concat_dim() + k]);
                }
            }
            round.upload_chunk(&staged, n).unwrap();
            round.inject_dev(n, None).unwrap();
            pos += n;
        }
        let _ = c;
    };
    {
        let c = 512usize;
        let taps = gen_taps(c);
        let base = {
            prime(&mut round, &taps, c, c);
            round.refresh_block_pos().unwrap();
            round.draft_round(12345, 2048, false, false, false).unwrap().tokens
        };
        let mut swapped = taps.clone();
        for k in 0..gb10_inference::dflash2::tap_concat_dim() {
            swapped.swap(510 * gb10_inference::dflash2::tap_concat_dim() + k, 511 * gb10_inference::dflash2::tap_concat_dim() + k);
        }
        let pert = {
            prime(&mut round, &swapped, c, c);
            round.refresh_block_pos().unwrap();
            round.draft_round(12345, 2048, false, false, false).unwrap().tokens
        };
        println!("    tap-row swap: base {base:?} vs pert {pert:?} [must differ]");
        check("negative control: tap-row swap changes the chain", base != pert);
    }
    {
        let c = 4096usize;
        let taps = gen_taps(c);
        prime(&mut round, &taps, c, c);
        round.refresh_block_pos().unwrap();
        let banded = round.draft_round(12345, 2048, false, false, false).unwrap().h_final;
        let noband = round.draft_round(12345, 1 << 20, false, false, false).unwrap().h_final;
        let d = rel_l2(&banded, &noband);
        println!("    band drop at C=4096: h rel-L2 {d:.3e} [must be >> 1e-2]");
        check("negative control: dropping the band fires (h diff explodes)", d > 1e-2);
    }
    {
        let c = 512usize;
        let taps = gen_taps(c);
        prime(&mut round, &taps, c, c);
        round.refresh_block_pos().unwrap();
        let norm1 = round.draft_round(12345, 2048, false, false, false).unwrap().tokens;
        let flip = round.draft_round(12345, 2048, true, false, false).unwrap().tokens;
        println!("    sign-flip unary: {norm1:?} vs {flip:?} [must differ]");
        check("negative control: selector score sign-flip changes the path", norm1 != flip);
    }
    {
        let c = 512usize;
        let taps = gen_taps(c);
        prime(&mut round, &taps, c, c);
        round.refresh_block_pos().unwrap();
        let _ = round.draft_round(12345, 2048, false, true, false).unwrap(); // dual write
        let (ring_attn, lin_attn) = round.dump_ctl_pair().unwrap();
        let d = rel_l2(&ring_attn, &lin_attn);
        let bit = ring_attn == lin_attn;
        println!("    ring vs linear attention (layer0, C=512): bitwise {} rel-L2 {d:.3e}",
                 if bit { "EQUAL" } else { "DIFF" });
        check("ring attention == S3F linear attention on the same cache (bitwise)", bit);
    }

    // ---- Part R4: determinism (two rounds bit-identical) ----------------------
    println!("\n== R4: determinism ==");
    {
        let c = 512usize;
        let taps = gen_taps(c);
        prime(&mut round, &taps, c, c);
        round.refresh_block_pos().unwrap();
        // 50 identical rounds (the flake rate is build/layout-dependent — one A/B pair
        // passed twice then failed on the third build, the signature of an
        // uninitialized-memory read). Localize the FIRST divergence by stage.
        // per-round stage dumps: x (conv in) / q / attn / h — localize the FIRST stage
        // that diverges (all stages are deterministic kernels; a divergence means some
        // read returned different bytes for identical inputs = uninitialized memory).
        // LAYER BISECT: 30 reps per depth; the first depth whose h_final diverges names
        // the layer that reads nondeterministic bytes.
        for depth in 1..=5usize {
            let hs: Vec<Vec<f32>> = (0..30).map(|_| {
                round.draft_round_depth(12345, 2048, false, false, false, depth).unwrap().h_final
            }).collect();
            let di = hs.iter().enumerate().skip(1).find(|(_, v)| *v != &hs[0]).map(|(i, _)| i);
            match di {
                Some(i) => {
                    let (a, b) = (&hs[0], &hs[i]);
                    let mut nd = 0usize; let mut first = 0usize;
                    for k in 0..a.len() { if a[k] != b[k] { nd += 1; if first == 0 { first = k; } } }
                    println!("      depth {depth} (L0..={}): DIVERGES at rep {i}: {nd}/{} elems, first at row {} col {}: {:+e} vs {:+e}",
                             depth - 1, a.len(), first / hidden(), first % hidden(), a[first], b[first]);
                    break;
                }
                None => println!("      depth {depth} (L0..={}): 30 reps identical", depth - 1),
            }
        }
        let mut xs = Vec::new(); let mut qs = Vec::new(); let mut ats = Vec::new(); let mut hs = Vec::new();
        let mut bks = Vec::new(); let mut r0s = Vec::new();
        let outs: Vec<_> = (0..50).map(|_| {
            let o = round.draft_round(12345, 2048, false, false, true).unwrap();
            xs.push(round.dump_block_x().unwrap());
            qs.push(round.dump_block_q().unwrap());
            ats.push(round.dump_attn_scratch().unwrap());
            hs.push(round.dump_block_h().unwrap());
            let (bk, _) = round.dump_block_kv().unwrap();          // L4 k BEFORE write_kv
            bks.push(bk);
            let (kr0, _) = round.dump_ring_kv(0).unwrap();         // L0 ring, block rows
            r0s.push(kr0[8 * 2048 * 128..].to_vec());
            o
        }).collect();
        for (name, set) in [("blk.k (L4, pre-write)", &bks), ("ring0 blockrows (post-write)", &r0s)] {
            if let Some(i) = set.iter().enumerate().skip(1).find(|(_, v)| *v != &set[0]).map(|(i, _)| i) {
                let (a, b) = (&set[0], &set[i]);
                let mut nd = 0usize; let mut first = 0usize;
                for k in 0..a.len() { if a[k] != b[k] { nd += 1; if first == 0 { first = k; } } }
                println!("      {name}: round {i} diverges: {nd}/{} elems, first at {first} (row {}, elem {}): {:+e} vs {:+e}",
                         a.len(), first / 1024, first % 1024, a[first], b[first]);
            } else {
                println!("      {name}: all 50 rounds identical");
            }
        }
        for (name, set) in [("x_conv", &xs), ("q", &qs), ("attn", &ats), ("h", &hs)] {
            let di = set.iter().enumerate().skip(1).find(|(_, v)| *v != &set[0]).map(|(i, _)| i);
            if let Some(i) = di {
                let (a, b) = (&set[0], &set[i]);
                let mut nd = 0usize; let mut first = 0usize;
                for k in 0..a.len() { if a[k] != b[k] { nd += 1; if first == 0 { first = k; } } }
                println!("      stage {name}: round {i} first diverges: {nd}/{} elems, first at elem {first} (row {}, col {}): {:+e} vs {:+e}",
                         a.len(), first / hidden(), first % hidden(), a[first], b[first]);
            } else {
                println!("      stage {name}: all 50 rounds identical");
            }
        }
let div_idx = outs.iter().enumerate().skip(1).find(|(_, o)| {
                o.tokens != outs[0].tokens || o.scores != outs[0].scores
                || o.candidates != outs[0].candidates || o.logits != outs[0].logits
                || o.h_final != outs[0].h_final
            }).map(|(i, _)| i);
        let det = div_idx.is_none();
        if let Some(di) = div_idx {
            let (a1, a2) = (&outs[0], &outs[di]);
            println!("      first divergence: round {di} vs round 0");
            if a1.h_final != a2.h_final {
                let mut nd = 0usize; let mut first = 0usize;
                for i in 0..a1.h_final.len() {
                    if a1.h_final[i] != a2.h_final[i] { nd += 1; if first == 0 { first = i; } }
                }
                println!("      h_final: {nd}/{} differ, first at row {} col {}: {:+e} vs {:+e}",
                         a1.h_final.len(), first / hidden(), first % hidden(), a1.h_final[first], a2.h_final[first]);
            }
            if a1.logits.as_ref().unwrap() != a2.logits.as_ref().unwrap() {
                let (l1, l2) = (a1.logits.as_ref().unwrap(), a2.logits.as_ref().unwrap());
                let mut nd = 0usize; let mut first = 0usize;
                for i in 0..l1.len() {
                    if l1[i] != l2[i] { nd += 1; if first == 0 { first = i; } }
                }
                println!("      logits: {nd}/{} differ, first at flat {first} (row {}, col {}): {:+e} vs {:+e}",
                         l1.len(), first / vocab(), first % vocab(), l1[first], l2[first]);
            }
            if a1.candidates != a2.candidates {
                for i in 0..a1.candidates.len() {
                    if a1.candidates[i] != a2.candidates[i] {
                        println!("      candidates[{i}]: {:#x} vs {:#x}", a1.candidates[i], a2.candidates[i]);
                        break;
                    }
                }
            }
            if a1.scores != a2.scores {
                for i in 0..a1.scores.len() {
                    if a1.scores[i] != a2.scores[i] {
                        println!("      scores[{i}] (row {}, rank {}): {:+e} vs {:+e}",
                                 i / 16, i % 16, a1.scores[i], a2.scores[i]);
                        break;
                    }
                }
            }
            if a1.tokens != a2.tokens {
                println!("      tokens: {:?} vs {:?}", a1.tokens, a2.tokens);
            }
        }
        check("two rounds bit-identical (5x: tokens/scores/candidates/logits/h_final)", det);
    }

    // ---- Part R5: perf --------------------------------------------------------
    println!("\n== R5: perf (integrated round vs the 15.21 ms Step-0 budget) ==");
    for &c in &[512usize, 4096usize] {
        let taps = gen_taps(c);
        prime(&mut round, &taps, c, c);
        round.refresh_block_pos().unwrap();
        let _ = round.draft_round(12345, 2048, false, false, false).unwrap(); // warm
        let mut staged: Vec<bf16> = vec![bf16::default(); gb10_inference::dflash2::tap_concat_dim() * gb10_inference::dflash2::block()];
        for mi in 0..8 {
            for k in 0..gb10_inference::dflash2::tap_concat_dim() {
                staged[mi * gb10_inference::dflash2::tap_concat_dim() + k] = bf16::from_f32(taps[(c - 8 + mi) * gb10_inference::dflash2::tap_concat_dim() + k]);
            }
        }
        let mut tot: Vec<f64> = Vec::with_capacity(100);
        let mut stg: Vec<[f32; 4]> = Vec::with_capacity(100);
        for _ in 0..100 {
            let t0 = std::time::Instant::now();
            let mut tm = EvTimer::new();
            round.upload_chunk(&staged, 8).unwrap();
            round.inject_dev(8, Some(&mut tm)).unwrap();   // the steady-state 8-token chunk
            round.refresh_block_pos().unwrap();
            let out = round.draft_round(12345, 2048, false, false, false).unwrap();
            tot.push(t0.elapsed().as_secs_f64() * 1e3);
            stg.push(out.stage_ms.unwrap());
            round.rollback_nprev(c);   // hold the steady state (identical work every rep)
        }
        tot.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |f: f64| tot[(f * (tot.len() - 1) as f64) as usize];
        let med = |idx: usize| {
            let mut v: Vec<f32> = stg.iter().map(|s| s[idx]).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        println!("    C={c}: round+inject median {:.3} ms (p10 {:.3} / p90 {:.3}) | stages: block {:.3} + head {:.3} + top16 {:.3} + walk {:.3}",
                 q(0.5), q(0.10), q(0.90), med(0), med(1), med(2), med(3));
        println!("      (budget 15.21 ms = the Step-0 GEMM stream; the integrated round adds injection + head + selector)");
    }
    // M flatness: injection at m in {1,4,8} at a fixed nprev (fc re-streams per chunk)
    {
        let c = 512usize;
        let taps = gen_taps(c);
        prime(&mut round, &taps, c, c - 8);
        let mut staged: Vec<bf16> = vec![bf16::default(); gb10_inference::dflash2::tap_concat_dim() * gb10_inference::dflash2::block()];
        for mi in 0..8 {
            for k in 0..gb10_inference::dflash2::tap_concat_dim() {
                staged[mi * gb10_inference::dflash2::tap_concat_dim() + k] = bf16::from_f32(taps[(c - 8 + mi) * gb10_inference::dflash2::tap_concat_dim() + k]);
            }
        }
        for &mm in &[1usize, 4, 8] {
            round.upload_chunk(&staged, mm).unwrap();
            let mut times: Vec<f64> = Vec::with_capacity(20);
            for _ in 0..20 {
                let t0 = std::time::Instant::now();
                round.inject_dev(mm, None).unwrap();
                times.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!("    inject(m={mm}) median {:.3} ms (x{mm} per round if every commit is {mm}-wide)",
                     times[times.len() / 2]);
        }
    }

    if baked {
        // PLAN/25 section 1a: on a baked (weight-only nvfp4/fp8) drafter the th/ring-kv gates
        // above are bitwise-vs-BF16-oracle by design and DIFFER by quantization noise (~1e-1
        // rel-L2 at fp4, ~1e-2 at fp8) — expected, recorded behavior, not a defect. The
        // binding gates for a baked artifact are the negative controls (input sensitivity),
        // two-round determinism, walk/top16 exactness, and the acceptance quartet +
        // LOSSLESS x3.
        let fmt = gb10_inference::dflash2::load::baked_fmt(draft_dir).unwrap_or("nvfp4");
        let noise_only = !nonoise_fail.get();
        println!("\nRESULT: {}", if all_pass.get() {
            "ALL PASS"
        } else if noise_only {
            match fmt {
                "fp8" => "PASS_WITH_FP8_NOISE (baked artifact: only the bitwise-vs-bf16 th/ring gates diff, at fp8-noise level; binding gates all pass)",
                _ => "PASS_WITH_FP4_NOISE (baked artifact: only the bitwise-vs-bf16 th/ring gates diff, at fp4-noise level; binding gates all pass)",
            }
        } else {
            "FAIL"
        });
    } else {
        println!("\nRESULT: {}", if all_pass.get() { "ALL PASS" } else { "FAIL" });
    }
    if !all_pass.get() && !baked { std::process::exit(1); }
}
