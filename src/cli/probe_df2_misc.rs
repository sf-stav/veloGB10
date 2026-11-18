//! DFlash2 round-level probes: lossless check, tap capture, prime and graph probes.
//! The big round probe is in `probe_df2_round.rs`; the DF2 benches are in `bench_df2.rs`.

use super::*;

/// S5F — `--probe-df2-lossless`: the greedy bit-identity gate (workdoc §3.1, BINDING).
/// DFlash2-on ≡ DFlash2-off (plain decode) ≡ MTP-on at temp 0 — token-stream bit-identity on
/// ≥3 prompts × ≥2 context scales. Rejected drafts are FREE losslessness (acceptance only
/// affects speed); this proves the emitted stream is the target's argmax at every position
/// under the DFlash2 source. Also asserts the drafter's ring KV never aliases trunk KV slots.
pub(crate) fn run_probe_df2_lossless(args: &[String]) {
    use cudarc::driver::{DevicePtr, DeviceSlice};
    use gb10_inference::batch::SpecSource;
    let trunk_dir = parse_arg(args, "--model-dir")
        .expect("--probe-df2-lossless needs --model-dir <trunk>").to_string();
    let max_seq_len: usize = gb10_inference::opt_or_arg(args, "--max-seq-len")
        .and_then(|s| s.parse().ok()).unwrap_or(4096);
    let max_new: usize = parse_arg(args, "--max-new-tokens")
        .and_then(|s| s.parse().ok()).unwrap_or(32);
    let mtp_depth: usize = parse_arg(args, "--mtp-depth")
        .and_then(|s| s.parse().ok()).unwrap_or(4);
    let all_pass = std::cell::Cell::new(true);
    let check = |name: &str, ok: bool| {
        println!("  [{:6}] {name}", if ok { "PASS" } else { "FAIL" });
        if !ok { all_pass.set(false); }
    };

    let tokenizer = QwenTokenizer::from_file(&format!("{}/tokenizer.json", trunk_dir.trim_end_matches('/')))
        .expect("tokenizer");
    let eos = tokenizer.stop_token_ids(config_eos(trunk_dir.trim_end_matches('/')));
    let (mut gpu, _) = load_model_gpu(&trunk_dir, None, 1);
    let df2 = load_df2_round(&mut gpu, max_seq_len);
    let (round, sink, prime) = match df2 {
        Some(x) => x,
        None => {
            println!("RESULT: FAIL — the DFlash2 round is not loadable (see [df2] WARN) — the DFlash2 \
                      leg of the bit-identity gate cannot run");
            std::process::exit(1);
        }
    };

    // The no-alias assert (workdoc §3.1): the drafter's ring KV must never intersect the trunk's
    // KV-cache ranges. Both are big independent allocations; an overlap would corrupt trunk KV on
    // write_kv (the round's write path) with no gate catching it.
    {
        let mut state = gpu.new_batch_state(1, 2, max_seq_len);
        let mut ranges: Vec<(u64, u64)> = round.ring_kv_ptr_ranges();
        for (k, v) in state.k_cache.iter().zip(state.v_cache.iter()) {
            for b in [k, v].into_iter().flatten() {
                ranges.push((*b.device_ptr() as u64, (b.len() * 2) as u64));
            }
        }
        let mut ok = true;
        for i in 0..ranges.len() {
            for j in (i + 1)..ranges.len() {
                let (a0, a1) = (ranges[i].0, ranges[i].0 + ranges[i].1);
                let (b0, b1) = (ranges[j].0, ranges[j].0 + ranges[j].1);
                if a0 < b1 && b0 < a1 {
                    ok = false;
                    println!("    ALIAS: [{a0:#x}, {a1:#x}) overlaps [{b0:#x}, {b1:#x})");
                }
            }
        }
        check("drafter ring KV does not alias trunk KV slots", ok);
    }

    // 3 prompts × 2 context scales (short + ~2K filler — the ring's full window engages).
    let filler: Vec<u32> = tokenizer.encode(
        "The quick brown fox jumps over the lazy dog. ", true).expect("filler");
    // One prompt per domain (gsm1, cod1, chat1) — the gate needs domain spread.
    let short_prompts: Vec<(&str, &str)> = ["math", "code", "chat"].iter()
        .map(|d| S3T3_PROMPTS.iter().find(|p| p.1 == *d).unwrap())
        .map(|p| (p.0, p.2)).collect();
    let mut prompts: Vec<(String, Vec<u32>)> = Vec::new();
    for (name, text) in &short_prompts {
        let tok = tokenizer.encode(text, true).expect("encode prompt");
        prompts.push((name.to_string(), tok.clone()));
        // Long scale: pad with the filler to >= 2048 tokens.
        let mut long = tok.clone();
        while long.len() < 2048 { long.extend_from_slice(&filler); }
        long.truncate(2048);
        prompts.push((format!("{name}-2k"), long));
    }
    // `name` is a &&str here; fix the lifetime by re-borrowing.
    let prompts: Vec<(String, Vec<u32>)> = prompts.into_iter()
        .map(|(n, t)| (n.to_string(), t)).collect();

    // One scheduler, three sources per prompt (Plain, DFlash2, Mtp) — same process, same load.
    use gb10_inference::batch::SpecBenchJob;
    // DF2 block-16: the tree source is refused above block 8 (it needs up to 31 verify columns
    // vs MAX_VERIFY 16), so the sweep SKIPS that arm rather than dying on the deliberate refusal
    // — a gate that cannot run its own control is not a gate. `n_src` keeps the result indexing
    // aligned with the arms that actually ran.
    let tree_ok = gb10_inference::dflash2::block() == 8;
    let n_src: usize = if tree_ok { 4 } else { 3 };
    if !tree_ok {
        println!("  [SKIP ] dflash2-tree arm: unsupported at --df2-block {} (needs <= 8)",
                 gb10_inference::dflash2::block());
    }
    let mut jobs = Vec::new();
    for (_, ptoks) in &prompts {
        for src in [SpecSource::Plain, SpecSource::DFlash2, SpecSource::DFlash2Tree, SpecSource::Mtp] {
            if src == SpecSource::DFlash2Tree && !tree_ok { continue; }
            jobs.push(SpecBenchJob {
                prompt: ptoks.clone(), max_new, temperature: 0.0, top_p: 1.0, top_k: 0,
                seed: 0x5EED_0001, source: src,
                domain: gb10_inference::batch::Domain::General,
            });
        }
    }
    let scheduler = build_spec_scheduler(gpu, max_seq_len, eos, SpecSource::Mtp, mtp_depth,
                                         Some(round), Some(sink), Some(prime),
                                         None, None, None, None);
    let rt = tokio::runtime::Builder::new_current_thread().enable_all()
        .build().expect("runtime");
    let (streams, _, _) = rt.block_on(scheduler.run_spec_bench(jobs, &[]));
    assert_eq!(streams.len(), prompts.len() * n_src);

    let mut all_ok = true;
    for (pi, (pname, _)) in prompts.iter().enumerate() {
        let (plain, df2s) = (&streams[pi * n_src], &streams[pi * n_src + 1]);
        let (tree, mtp) = if tree_ok {
            (&streams[pi * n_src + 2], &streams[pi * n_src + 3])
        } else {
            (&streams[pi * n_src], &streams[pi * n_src + 2])   // tree arm skipped: self-compare
        };
        let eq_df2 = plain == df2s;
        let eq_tree = !tree_ok || plain == tree;
        let eq_mtp = plain == mtp;
        let div = |a: &[u32], b: &[u32]| -> Option<usize> {
            a.iter().zip(b.iter()).position(|(x, y)| x != y)
        };
        println!("  prompt {pname} ({} tokens): plain==df2 {} plain==tree {} plain==mtp {}",
                 plain.len(), eq_df2, eq_tree, eq_mtp);
        if !eq_df2 {
            let d = div(plain, df2s);
            println!("    DF2 divergence at {:?}: plain {:?} vs df2 {:?}",
                     d, d.map(|i| plain[i]), d.map(|i| df2s[i]));
        }
        if !eq_mtp {
            let d = div(plain, mtp);
            println!("    MTP divergence at {:?}: plain {:?} vs mtp {:?}",
                     d, d.map(|i| plain[i]), d.map(|i| mtp[i]));
        }
        if !eq_tree {
            let d = div(plain, tree);
            println!("    TREE divergence at {:?}: plain {:?} vs tree {:?}",
                     d, d.map(|i| plain[i]), d.map(|i| tree[i]));
        }
        all_ok &= eq_df2 && eq_tree && eq_mtp;
        check(&format!("{pname}: DFlash2-on == tree == off == MTP-on (bit-identity, temp 0)"),
              eq_df2 && eq_tree && eq_mtp);
    }
    check(&format!("greedy bit-identity over {} prompts × 2 scales", prompts.len() / 2), all_ok);
    println!("RESULT: {}", if all_pass.get() { "ALL PASS" } else { "FAIL" });
    if !all_pass.get() { std::process::exit(1); }
}

/// S5F2 — `--probe-df2-tapcap`: capture the trunk's DFlash2 tap hiddens for the S3T3 chat1
/// prompt (token-by-token prefill, tap sink armed), write `[plen, 25600]` f32 to `--out`.
/// Dtype-generic (the NVFP4 serving trunk and the plain-BF16 trunk class both work). The
/// output is the engine-side tap snapshot: vs the BF16 text model's reference taps
/// (`tool_probe/b8/s5f_bf16_taps.py` output) it is the S5F2 L0 tap-cleanliness check; the
/// NVFP4-trunk captures are the L1c fit input (paired with the BF16 references).
pub(crate) fn run_probe_df2_tapcap(args: &[String]) {
    use gb10_inference::dflash2::capture::Df2TapSink;
    use gb10_inference::dflash2::{hidden, tap_concat_dim};
    use half::bf16;
    let trunk_dir = parse_arg(args, "--model-dir")
        .expect("--probe-df2-tapcap needs --model-dir <trunk>").to_string();
    let out_path = parse_arg(args, "--out")
        .expect("--probe-df2-tapcap needs --out <file>").to_string();
    let prompt_text = "Explain how a transformer's self-attention mechanism works, in three short sentences.";

    let tokenizer = QwenTokenizer::from_file(&format!("{}/tokenizer.json", trunk_dir.trim_end_matches('/')))
        .expect("tokenizer");
    let ids = tokenizer.encode(prompt_text, true).expect("encode chat1");
    println!("[tapcap] prompt plen={} ids={:?}", ids.len(), &ids[..ids.len().min(6)]);

    let (mut gpu, _) = load_model_gpu(&trunk_dir, None, 1);
    let sink = std::sync::Arc::new(Df2TapSink::new(gpu.dev()));
    gpu.set_df2_capture(sink.clone());
    assert!(gpu.df2_capture_armed(), "tap capture not armed");
    let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
    let mut state = gpu.new_batch_state(1, 2, 4096);
    let mut taps: Vec<f32> = Vec::with_capacity(ids.len() * gb10_inference::dflash2::tap_concat_dim());
    for (t, &tok) in ids.iter().enumerate() {
        let hidden = gpu.embed_batch(&[tok]);
        let out = gpu.forward_batch(&mut pool, hidden, &[t], &mut state, 4096, 1);
        gpu.sync_stream();
        let col: Vec<bf16> = gpu.dev().dtoh_sync_copy(&sink.staging).expect("staging dtoh");
        for k in 0..gb10_inference::dflash2::tap_concat_dim() { taps.push(col[k].to_f32()); }
        pool.release_bf16(out, gb10_inference::dflash2::hidden());
    }
    std::fs::write(&out_path, bytemuck::cast_slice(&taps)).expect("write taps");
    println!("[tapcap] wrote {out_path}: [{}, {}] f32 ({} bytes)", ids.len(), gb10_inference::dflash2::tap_concat_dim(), taps.len() * 4);
    println!("RESULT: TAPCAP_DONE");
}

/// S5F — `--probe-df2-prime`: the prompt-prime correctness gate. Part A (synthetic, deterministic):
/// the SAME taps through the probe's PROVEN chunked path (upload_chunk + inject_dev) vs the
/// engine's `prime_window` — the ring k/v must agree bitwise (or within the documented ≤2-quanta
/// tail) and the draft tokens must be identical (the prime is a pure function of the taps).
/// Part B (real prompt): the PREFILL-captured taps (the engine path) vs the DECODE-captured taps
/// (the R1-proven per-token capture) — reports the tap delta + drafts (prefill-vs-decode tap
/// deltas are documented findings, not gates — drafts are verify-rejected either way).
pub(crate) fn run_probe_df2_prime(args: &[String]) {
    use gb10_inference::dflash2::capture::{Df2PrimeSink, Df2TapSink};
    use gb10_inference::dflash2::round::{Df2Round, ring, ring_stride};
    use gb10_inference::dflash2::{BLOCK, hidden, tap_concat_dim};
    use gb10_inference::dflash2::synth::SyntheticTables;
    use half::bf16;
    let trunk_dir = parse_arg(args, "--model-dir")
        .expect("--probe-df2-prime needs --model-dir <trunk>").to_string();
    let max_c: usize = gb10_inference::opt_or_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(4096);
    let draft_dir = required_dir_arg(args, "--draft-dir", "the DFlash2 draft artifact");
    let all_pass = std::cell::Cell::new(true);
    let check = |name: &str, ok: bool| {
        println!("  [{:6}] {name}", if ok { "PASS" } else { "FAIL" });
        if !ok { all_pass.set(false); }
    };
    let (mut gpu, _) = load_model_gpu(&trunk_dir, None, 1);
    let (head_p, embed_p) = gpu.df2_borrow_ptrs().expect("nvfp4 head/embed");
    let mut round = Df2Round::load(&draft_dir, Some(gb10_inference::dflash2::round::BorrowedW::Nvfp4(head_p)), Some(gb10_inference::dflash2::round::BorrowedW::Nvfp4(embed_p)), max_c).expect("round load");
    let sink = std::sync::Arc::new(Df2TapSink::new(gpu.dev()));
    round.attach_sink(&sink);
    let prime = std::sync::Arc::new(Df2PrimeSink::new(gpu.dev(), 8192));

    // ---- Part A: synthetic taps through BOTH prime paths. ----
    println!("== A: synthetic taps — chunked inject vs prime_window ==");
    let taps_gen = SyntheticTables::new(gb10_inference::dflash2::SYNTH_TAP_SEED);
    let tap_scale = 1.0f32 / (gb10_inference::dflash2::tap_concat_dim() as f32).sqrt();
    let c = 130usize;   // a realistic short-prompt length (> 8, exercising the wide path)
    let taps_cm: Vec<bf16> = {
        // col-major [25600, c]: column j = the tap row j (the prime_window layout).
        let mut t = vec![bf16::default(); gb10_inference::dflash2::tap_concat_dim() * c];
        for j in 0..c {
            let row = taps_gen.row(SyntheticTables::TABLE_TAPS, j as u32, gb10_inference::dflash2::tap_concat_dim(), tap_scale);
            for k in 0..gb10_inference::dflash2::tap_concat_dim() { t[j * gb10_inference::dflash2::tap_concat_dim() + k] = bf16::from_f32(row[k]); }
        }
        t
    };
    // (a1) the PROVEN chunked path: upload_chunk + inject_dev in BLOCK chunks.
    let mut staged: Vec<bf16> = vec![bf16::default(); gb10_inference::dflash2::tap_concat_dim() * gb10_inference::dflash2::block()];
    round.reset();
    let mut pos = 0usize;
    while pos < c {
        let n = gb10_inference::dflash2::block().min(c - pos);
        for mi in 0..n {
            for k in 0..gb10_inference::dflash2::tap_concat_dim() {
                staged[mi * gb10_inference::dflash2::tap_concat_dim() + k] = taps_cm[(pos + mi) * gb10_inference::dflash2::tap_concat_dim() + k];
            }
        }
        round.upload_chunk(&staged, n).unwrap();
        round.inject_dev(n, None).unwrap();
        pos += n;
    }
    let ring_a: Vec<(Vec<f32>, Vec<f32>)> = (0..5).map(|li| round.dump_ring_kv(li).unwrap()).collect();
    round.refresh_block_pos().unwrap();
    let draft_a = round.draft_round_dev(12345).unwrap();

    // (a2) prime_window with the SAME taps (a fresh round on the round's own device).
    let mut round2 = Df2Round::load(&draft_dir, Some(gb10_inference::dflash2::round::BorrowedW::Nvfp4(head_p)), Some(gb10_inference::dflash2::round::BorrowedW::Nvfp4(embed_p)), max_c).expect("round2 load");
    round2.attach_sink(&sink);
    round2.reset();
    let taps_slice = round2.dev.htod_sync_copy(&taps_cm).expect("round taps cm");
    round2.prime_window(&taps_slice, c, 0).unwrap();
    let ring_b: Vec<(Vec<f32>, Vec<f32>)> = (0..5).map(|li| round2.dump_ring_kv(li).unwrap()).collect();
    round2.refresh_block_pos().unwrap();
    let draft_b = round2.draft_round_dev(12345).unwrap();

    let mut n_mm = 0usize; let mut n_tot = 0usize; let mut max_rel = 0f32;
    // The S4F ring gate's metric: |dev−mir| relative to the ROW's max |k| (the rounding-quantum
    // scale) — raw ulps are meaningless near zero, where the rsqrt/rope class flips ~1e-4 values.
    for li in 0..5 {
        let rowmax = |row: usize, buf: &Vec<f32>| -> f32 {
            let mut m = 0f32;
            for h in 0..8usize {
                for d in 0..128usize {
                    let v = buf[(row + h * ring_stride() as usize) * 128 + d].abs();
                    if v > m { m = v; }
                }
            }
            m
        };
        // rows actually compared: the ring rows the prime wrote (0..c % ring()) for both k and v.
        for (ki, buf) in [&ring_a[li].0, &ring_a[li].1].iter().enumerate() {
            for row in 0..c.min(ring()) {
                let scale = rowmax(row, buf).max(1e-9);
                let other = if ki == 0 { &ring_b[li].0 } else { &ring_b[li].1 };
                for h in 0..8usize {
                    for d in 0..128usize {
                        let idx = (row + h * ring_stride() as usize) * 128 + d;
                        n_tot += 1;
                        if buf[idx].to_bits() != other[idx].to_bits() {
                            n_mm += 1;
                            max_rel = max_rel.max((buf[idx] - other[idx]).abs() / scale);
                        }
                    }
                }
            }
        }
    }
    println!("  ring k/v: {n_mm}/{n_tot} mismatched (max |Δ|/rowMAX {max_rel:.3e}, bar 2^-6)");
    // The two GEMM paths (gemm_tiled vs gemm_dsp) have different k-accumulation orders, so the
    // ring deltas are the documented device-order class (≤2 rounding quanta of the row max) —
    // the FUNCTIONAL gate is the drafts: identical outputs prove the prime is a pure function.
    check("Part A: prime_window == chunked inject (ring deltas <=2-quanta; drafts identical)",
          max_rel <= 2f32.powi(-6));
    println!("  drafts: chunked {draft_a:?}");
    println!("  drafts: prime_w {draft_b:?}");
    check("Part A: drafts identical (prime is a pure function of the taps)", draft_a == draft_b);

    // ---- Part B: real chat1 — prefill-captured vs decode-captured taps. ----
    println!("== B: real chat1 — prefill-captured vs decode-captured taps ==");
    let tokenizer = QwenTokenizer::from_file(&format!("{}/tokenizer.json", trunk_dir.trim_end_matches('/')))
        .expect("tokenizer");
    let prompt = tokenizer.encode(
        "Explain how a transformer's self-attention mechanism works, in three short sentences.", true)
        .expect("encode");
    let plen = prompt.len();
    let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
    let mut state = gpu.new_batch_state(1, 2, max_c);
    gpu.set_df2_prime_sink(prime.clone());
    gpu.zero_slot_state(&mut state, 0, max_c);
    let (a0, hout) = gpu.prefill_batch(&mut pool, &prompt, &mut state, 0, max_c, 0);
    pool.release_bf16(hout, hidden() * plen);
    gpu.set_df2_prime_off();
    round2.reset();
    round2.prime_window(&prime.taps, plen, 0).unwrap();
    round2.refresh_block_pos().unwrap();
    let draft_prefill = round2.draft_round_dev(a0).unwrap();

    // the DECODE-captured reference: forward_batch per token with the 8-col sink armed.
    gpu.set_df2_capture(sink.clone());
    round.reset();
    let mut dcap: Vec<bf16> = Vec::with_capacity(plen * gb10_inference::dflash2::tap_concat_dim());
    let mut state2 = gpu.new_batch_state(1, 2, max_c);
    gpu.zero_slot_state(&mut state2, 0, max_c);
    for (t, &tok) in prompt.iter().enumerate() {
        let hidden = gpu.embed_batch(&[tok]);
        let out = gpu.forward_batch(&mut pool, hidden, &[t], &mut state2, max_c, 1);
        gpu.sync_stream();
        let col: Vec<bf16> = gpu.dev().dtoh_sync_copy(&sink.staging).expect("staging dtoh");
        dcap.extend_from_slice(&col[..gb10_inference::dflash2::tap_concat_dim()]);
        pool.release_bf16(out, gb10_inference::dflash2::hidden());
    }
    gpu.set_df2_capture_off();
    let mut staged: Vec<bf16> = vec![bf16::default(); gb10_inference::dflash2::tap_concat_dim() * gb10_inference::dflash2::block()];
    let mut pos = 0usize;
    while pos < plen {
        let n = gb10_inference::dflash2::block().min(plen - pos);
        for mi in 0..n {
            for k in 0..gb10_inference::dflash2::tap_concat_dim() {
                staged[mi * gb10_inference::dflash2::tap_concat_dim() + k] = dcap[(pos + mi) * gb10_inference::dflash2::tap_concat_dim() + k];
            }
        }
        round.upload_chunk(&staged, n).unwrap();
        round.inject_dev(n, None).unwrap();
        pos += n;
    }
    round.refresh_block_pos().unwrap();
    let draft_decode = round.draft_round_dev(a0).unwrap();
    let pf_taps: Vec<bf16> = gpu.dev().dtoh_sync_copy(&prime.taps).expect("pf taps dtoh")
        .into_iter().take(plen * gb10_inference::dflash2::tap_concat_dim()).collect();
    let mut tap_mm = 0usize; let mut tap_tot = 0usize; let mut tap_ulp = 0u32;
    for (x, y) in pf_taps.iter().zip(dcap.iter()) {
        tap_tot += 1;
        if x.to_bits() != y.to_bits() {
            tap_mm += 1;
            tap_ulp = tap_ulp.max(bf16_ulp_dist(*x, *y));
        }
    }
    println!("  taps: {tap_mm}/{tap_tot} prefill-vs-decode mismatched (max {tap_ulp} ulp)");
    // per-layer rel-L2 — the prefill-vs-decode hidden delta is the documented bf16 path class
    // (gqa_attn_prefill vs splitk), NOT a wiring bug; the drafts diverging only at position 5+
    // shows the delta is second-order for the round.
    let pf_f: Vec<f32> = pf_taps.iter().map(|b| b.to_f32()).collect();
    let dc_f: Vec<f32> = dcap.iter().map(|b| b.to_f32()).collect();
    let mut worst_pd = 0f64;
    for k in 0..5usize {
        let mut num = 0.0f64; let mut den = 0.0f64;
        for j in 0..plen {
            for i in 0..hidden() {
                let d = (pf_f[j * gb10_inference::dflash2::tap_concat_dim() + k * hidden() + i]
                         - dc_f[j * gb10_inference::dflash2::tap_concat_dim() + k * hidden() + i]) as f64;
                num += d * d;
                den += (dc_f[j * gb10_inference::dflash2::tap_concat_dim() + k * hidden() + i] as f64)
                         * (dc_f[j * gb10_inference::dflash2::tap_concat_dim() + k * hidden() + i] as f64);
            }
        }
        worst_pd = worst_pd.max((num / den.max(1e-30)).sqrt());
    }
    println!("  prefill-vs-decode tap rel-L2 worst {worst_pd:.4} (the documented bf16 path class)");
    println!("  drafts: prefill-taps {draft_prefill:?}");
    println!("  drafts: decode-taps {draft_decode:?}");
    check("Part B: prefill taps within the bf16 path-noise class (worst rel-L2 < 0.1)",
          worst_pd < 0.1);

    // ---- Part C: dump the engine's chat1 PREFILL taps for the BF16-reference comparison. ----
    // The .14-side extraction (`tool_probe/b8/s5f_bf16_taps.py`) computes the SAME prompt's tap
    // hiddens on the BF16 TEXT model; the rel-L2 per tap layer quantifies the documented
    // NVFP4-vs-BF16 conditioning delta (the offline studies' predicted shift).
    {
        let mut f = std::fs::File::create("/tmp/s5f/chat1_engine_taps.f32").expect("tap dump");
        use std::io::Write;
        let pf_f: Vec<f32> = pf_taps.iter().map(|b| b.to_f32()).collect();
        f.write_all(bytemuck::cast_slice(&pf_f[..plen * gb10_inference::dflash2::tap_concat_dim()])).expect("write taps");
        println!("  engine chat1 prefill taps ({plen} cols) -> /tmp/s5f/chat1_engine_taps.f32");
    }
    println!("RESULT: {}", if all_pass.get() { "ALL PASS" } else { "FAIL" });
    if !all_pass.get() { std::process::exit(1); }
}

/// S5F — `--probe-df2-graph`: the draft-round CUDA graph gate (workdoc §3.4). Captures the round
/// graph, then: (a) determinism UNDER capture — repeated graph replays must be bit-identical;
/// (b) graph == eager — the SAME ring state driven through the graph vs the eager kernel sequence
/// must produce bit-identical walk tokens (the R13 volatile kernels are stable under capture);
/// (c) captured-vs-eager round time (the launch-amortization report).
pub(crate) fn run_probe_df2_graph(args: &[String]) {
    use gb10_inference::dflash2::capture::{Df2PrimeSink, Df2TapSink};
    use gb10_inference::dflash2::round::Df2Round;
    use gb10_inference::dflash2::{BLOCK, hidden, tap_concat_dim};
    use gb10_inference::dflash2::synth::SyntheticTables;
    use half::bf16;
    let trunk_dir = parse_arg(args, "--model-dir")
        .expect("--probe-df2-graph needs --model-dir <trunk>").to_string();
    let max_c: usize = gb10_inference::opt_or_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(4096);
    let draft_dir = required_dir_arg(args, "--draft-dir", "the DFlash2 draft artifact");
    let all_pass = std::cell::Cell::new(true);
    let check = |name: &str, ok: bool| {
        println!("  [{:6}] {name}", if ok { "PASS" } else { "FAIL" });
        if !ok { all_pass.set(false); }
    };
    let (mut gpu, _) = load_model_gpu(&trunk_dir, None, 1);
    let (head_p, embed_p) = gpu.df2_borrow_ptrs().expect("nvfp4 head/embed");
    let mut round = Df2Round::load(&draft_dir, Some(gb10_inference::dflash2::round::BorrowedW::Nvfp4(head_p)), Some(gb10_inference::dflash2::round::BorrowedW::Nvfp4(embed_p)), max_c).expect("round load");
    let sink = std::sync::Arc::new(Df2TapSink::new(gpu.dev()));
    round.attach_sink(&sink);
    let _prime = std::sync::Arc::new(Df2PrimeSink::new(gpu.dev(), 8192));

    // Prime with deterministic synthetic taps (the chunked path).
    let taps_gen = SyntheticTables::new(gb10_inference::dflash2::SYNTH_TAP_SEED);
    let tap_scale = 1.0f32 / (gb10_inference::dflash2::tap_concat_dim() as f32).sqrt();
    let c = 512usize;
    let mut staged: Vec<bf16> = vec![bf16::default(); gb10_inference::dflash2::tap_concat_dim() * gb10_inference::dflash2::block()];
    let mut pos = 0usize;
    while pos < c {
        let n = gb10_inference::dflash2::block().min(c - pos);
        for mi in 0..n {
            let row = taps_gen.row(SyntheticTables::TABLE_TAPS, (pos + mi) as u32, gb10_inference::dflash2::tap_concat_dim(), tap_scale);
            for k in 0..gb10_inference::dflash2::tap_concat_dim() { staged[mi * gb10_inference::dflash2::tap_concat_dim() + k] = bf16::from_f32(row[k]); }
        }
        round.upload_chunk(&staged, n).unwrap();
        round.inject_dev(n, None).unwrap();
        pos += n;
    }
    round.refresh_block_pos().unwrap();

    // (a) eager reference tokens.
    let eager_tokens = round.draft_round_dev(12345).unwrap();
    println!("  eager   : {eager_tokens:?}");

    // (b) capture + replay.
    if !round.capture_round_graph() {
        println!("RESULT: FAIL — round graph capture failed (unsupported)");
        std::process::exit(1);
    }
    let graph_tokens = round.draft_round_graph(12345).unwrap();
    println!("  graph   : {graph_tokens:?}");
    check("graph == eager (bit-identical walk tokens on the same state)", graph_tokens == eager_tokens);
    // determinism under capture: replay twice more, must be bit-identical.
    let g1 = round.draft_round_graph(12345).unwrap();
    let g2 = round.draft_round_graph(12345).unwrap();
    check("determinism under capture (3 replays bit-identical)", g1 == g2 && g1 == graph_tokens);

    // (b2) varying-nprev replays: the graph must replay correctly at ANY nprev <= max_c (the
    // capture sizes smem for the max; the replayed ntot is device-driven).
    {
        let mut ok = true;
        for &np in &[511usize, 256, 64] {   // descending: rollback only rewinds
            round.rollback_nprev(np);
            round.refresh_block_pos().unwrap();
            let eg = round.draft_round_dev(999).unwrap();
            let gp = round.draft_round_graph(999).unwrap();
            if eg != gp {
                ok = false;
                println!("    nprev={np}: eager {eg:?} vs graph {gp:?}");
                // first-diverging surface: the post-final-norm block hidden (the head's input).
                let hf = round.dump_h_final().unwrap();
                let h = gb10_inference::dflash2::hidden();
                let mut nd = 0usize; let mut f = 0usize;
                for i in 0..hf.len() {
                    if hf[i].to_bits() != hf[i].to_bits() { nd += 1; if f == 0 { f = i; } }
                }
                println!("      h_final (post-eager-round): {} diffs vs itself is 0-check", nd);
                // re-run the eager then the graph, dumping h_final after EACH, and compare.
                let _ = round.draft_round_dev(999).unwrap();
                let hf_e = round.dump_h_final().unwrap();
                let _ = round.draft_round_graph(999).unwrap();
                let hf_g = round.dump_h_final().unwrap();
                let mut nd2 = 0usize; let mut f2 = 0usize; let mut max_d = 0f32;
                for i in 0..hf_e.len() {
                    if hf_e[i].to_bits() != hf_g[i].to_bits() {
                        nd2 += 1;
                        if f2 == 0 { f2 = i; }
                        max_d = max_d.max((hf_e[i] - hf_g[i]).abs());
                    }
                }
                println!("      h_final eager-vs-graph: {nd2}/{} differ, first at row {} col {} (max |d| {max_d:.3e})",
                         hf_e.len(), f2 / h, f2 % h);
            }
        }
        check("varying-nprev replays (nprev in {64,256,511}) == eager bitwise", ok);
        // rollback can only REWIND — leave nprev where the loop ended (64); the timing section
        // below is nprev-agnostic (eager vs graph at the same state).
    }

    // (c) captured-vs-eager round time (median of 30, the launch-amortization report).
    let mut eager_ms = Vec::new();
    for _ in 0..30 {
        let t0 = std::time::Instant::now();
        let _ = round.draft_round_dev(12345).unwrap();
        eager_ms.push(t0.elapsed().as_secs_f32() * 1e3);
    }
    let mut graph_ms = Vec::new();
    for _ in 0..30 {
        let t0 = std::time::Instant::now();
        let _ = round.draft_round_graph(12345).unwrap();
        graph_ms.push(t0.elapsed().as_secs_f32() * 1e3);
    }
    let mut med = |v: &mut Vec<f32>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };
    let me = med(&mut eager_ms); let mg = med(&mut graph_ms);
    println!("  round time (median 30): eager {me:.3} ms | graph {mg:.3} ms (Δ {:.2}%)",
             100.0 * (mg - me) / me.max(1e-6));
    println!("RESULT: {}", if all_pass.get() { "ALL PASS" } else { "FAIL" });
    if !all_pass.get() { std::process::exit(1); }
}

// ===========================================================================================
// S4F — `--probe-df2-round <draft-dir> --model-dir <trunk-dir>`: the INTEGRATED draft round on
// REAL weights: trunk tap capture -> incremental injection (gemm_dsp, ring KV) -> 5-layer block
// pass -> borrowed NVFP4 LM-head logits -> radix top-16 -> selector chain -> 7 tokens.
// Gates vs the extended mirror (bitwise where inputs are identical, rel-L2 where the device's
// hardware mma order cannot be mirrored), negative controls, determinism, perf vs 15.21 ms.
// ===========================================================================================
/// bf16 ulp distance between two values (same-sign assumption; 0xFFFF for sign flips).
pub(super) fn bf16_ulp_dist(a: half::bf16, b: half::bf16) -> u32 {
    let (x, y) = (a.to_bits(), b.to_bits());
    if x == y { return 0; }
    let sx = (x >> 15) & 1; let sy = (y >> 15) & 1;
    if sx != sy { return 0xFFFF; }
    if sx == 1 { (y as i32 - x as i32).unsigned_abs() } else { (x as i32 - y as i32).unsigned_abs() as u32 }
}
