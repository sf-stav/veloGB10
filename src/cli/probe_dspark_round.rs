//! DSpark round probe (`run_probe_dspark_round`) and its draft-dir loader.
//! One long probe function, kept whole; sibling DSpark probes are in `probe_dspark.rs`.

use super::*;

/// WI1 GATE: the DSpark round on GPU vs the CPU oracle (`dspark::oracle`), self-contained —
/// NO trunk load: the borrowed embed/head are the oracle's own deterministic synthetic tables
/// (SYNTH_EMBED_HEAD_SEED), materialized once to bf16 so the device gathers EXACTLY what the
/// oracle generates (up to bf16 rounding, which the rel-L2 gates absorb; the chain gate runs
/// the host markov chain on the DEVICE logits, so token EXACTness carries zero tolerance
/// risk). Synthetic taps at ctx 8 (prime_window), then +8 injected (inject_dev) → ctx 16 —
/// the second round exercises the ring attention over injected context.
pub(crate) fn run_probe_dspark_round(args: &[String], draft_dir: &str) {
    use gb10_inference::dflash2::round::BorrowedW;
    use gb10_inference::dflash2::synth::SyntheticTables;
    use cudarc::driver::DevicePtr;
    use gb10_inference::dspark::oracle::{DsparkConfig, DsparkOracle, RoundCtx};
    use gb10_inference::dspark::{load as ds_load, HIDDEN as DH, MARKOV_RANK, TAP_CONCAT_DIM, VOCAB as DV};
    use half::bf16;

    let pin = parse_arg(args, "--sha256");
    let all_pass = std::cell::Cell::new(true);
    let check = |name: &str, ok: bool| {
        println!("  [{:6}] {name}", if ok { "PASS" } else { "FAIL" });
        if !ok { all_pass.set(false); }
    };
    let rel_l2 = |a: &[f32], b: &[f32]| -> f64 {
        let mut num = 0.0f64; let mut den = 0.0f64;
        for i in 0..a.len().min(b.len()) {
            let dd = (a[i] - b[i]) as f64;
            num += dd * dd; den += (b[i] as f64) * (b[i] as f64);
        }
        (num / den.max(1e-30)).sqrt()
    };

    // ---- Part 0: artifact + oracle (host) ---------------------------------
    println!("== loads ==");
    let t0 = std::time::Instant::now();
    let art = ds_load::load(draft_dir, pin).expect("DSpark artifact load");
    println!("  artifact: {} tensors, {} params, sha {}… ({:.1}s)",
             art.n_tensors, art.n_params, &art.sha256[..16], t0.elapsed().as_secs_f32());
    let oracle = DsparkOracle::from_weights(DsparkConfig::default(), art.weights.clone())
        .expect("oracle build");

    // ---- Part 1: the round on GPU with synthetic borrowed tables ----------
    let dev = cudarc::driver::CudaDevice::new(0).expect("CudaDevice");
    let synth = SyntheticTables::new(gb10_inference::dspark::SYNTH_EMBED_HEAD_SEED);
    let scale = 1.0f32 / (DH as f32).sqrt();
    fn materialize(synth: &SyntheticTables, table: u64, scale: f32,
                   dev: &std::sync::Arc<cudarc::driver::CudaDevice>)
                   -> std::sync::Arc<cudarc::driver::CudaSlice<bf16>> {
        let mut out = vec![bf16::default(); DV * DH];
        let nthreads = gb10_inference::opts::var(gb10_inference::opt!("df2-mirror-threads")).ok()
            .and_then(|v| v.parse::<usize>().ok()).filter(|&n| (1..=16).contains(&n)).unwrap_or(8);
        let rows_per = (DV + nthreads - 1) / nthreads;
        std::thread::scope(|sc| {
            let mut rest = out.as_mut_slice();
            let mut r0 = 0usize;
            for t in 0..nthreads {
                let r1 = if t == nthreads - 1 { DV } else { (r0 + rows_per).min(DV) };
                if r0 >= r1 { break; }
                let (a, b) = rest.split_at_mut((r1 - r0) * DH);
                rest = b;
                let base = r0;
                r0 = r1;
                sc.spawn(move || {
                    for (i, row) in a.chunks_mut(DH).enumerate() {
                        let v = synth.row(table, (base + i) as u32, DH, scale);
                        for (d, x) in row.iter_mut().enumerate() {
                            *x = bf16::from_f32(v[d]);
                        }
                    }
                });
            }
        });
        std::sync::Arc::new(dev.htod_sync_copy(&out).expect("table upload"))
    }
    let t1 = std::time::Instant::now();
    let embed_tbl = materialize(&synth, SyntheticTables::TABLE_EMBED, scale, &dev);
    let head_tbl = materialize(&synth, SyntheticTables::TABLE_HEAD, scale, &dev);
    println!("  synthetic embed+head tables materialized ({:.1}s)", t1.elapsed().as_secs_f32());
    let head_w = BorrowedW::Bf16 { ptr: *head_tbl.device_ptr() as u64 };
    let embed_w = BorrowedW::Bf16 { ptr: *embed_tbl.device_ptr() as u64 };
    let mut round = gb10_inference::dspark::round::DsparkRound::load_pinned(
        draft_dir, Some(head_w), Some(embed_w), 4096, pin).expect("DsparkRound load");

    // ---- REAL-TAPS mode: replay the SERVED dspark steps offline on the captured taps with the
    // REAL trunk embed+lm_head (outside.safetensors), walking prime → per-step inject → draft
    // exactly as the lane did. Agreement with the served drafts proves the round consumed the
    // captured taps faithfully; disagreement with the TRUNK then localizes to tap content or
    // artifact semantics — never to the lane plumbing.
    if let Some(dir) = parse_arg(args, "--real-taps") {
        let trunk_dir = parse_arg(args, "--embed-from")
            .unwrap_or("models/Qwen3.8-27B-FP8").to_string();
        let load_outside = |key: &str| -> Vec<bf16> {
            use std::io::{Read, Seek, SeekFrom};
            let path = format!("{}/outside.safetensors", trunk_dir.trim_end_matches('/'));
            let mut f = std::fs::File::open(&path).expect("open outside.safetensors");
            let mut hb = [0u8; 8];
            f.read_exact(&mut hb).unwrap();
            let n = u64::from_le_bytes(hb) as usize;
            let mut hdr = vec![0u8; n];
            f.read_exact(&mut hdr).unwrap();
            let h: serde_json::Value = serde_json::from_slice(&hdr).expect("header json");
            let e = h.get(key).unwrap_or_else(|| panic!("{key} missing from outside.safetensors"));
            let off = e["data_offsets"][0].as_u64().unwrap();
            let len = e["data_offsets"][1].as_u64().unwrap() - off;
            assert_eq!(e["dtype"].as_str().unwrap(), "BF16", "{key} not BF16");
            f.seek(SeekFrom::Start(8 + n as u64 + off)).unwrap();
            let mut raw = vec![0u8; len as usize];
            f.read_exact(&mut raw).expect("tensor bytes");
            raw.chunks_exact(2).map(|c| bf16::from_le_bytes([c[0], c[1]])).collect()
        };
        let t_embed = std::time::Instant::now();
        let embed_real = dev.htod_sync_copy(&load_outside("model.language_model.embed_tokens.weight"))
            .expect("embed upload");
        let head_real = dev.htod_sync_copy(&load_outside("lm_head.weight")).expect("head upload");
        println!("  real trunk embed+lm_head loaded ({:.1}s)", t_embed.elapsed().as_secs_f32());
        drop(round);   // the synth-table round; rebuild with the real tables
        drop((embed_tbl, head_tbl));   // free the 5 GB synth tables before the trunk loads
        let head_w = BorrowedW::Bf16 { ptr: *head_real.device_ptr() as u64 };
        let embed_w = BorrowedW::Bf16 { ptr: *embed_real.device_ptr() as u64 };
        let mut round = gb10_inference::dspark::round::DsparkRound::load_pinned(
            draft_dir, Some(head_w), Some(embed_w), 4096, pin).expect("DsparkRound load (real tables)");

        // captured inputs: walk every request dir r{N} (--dspark-dump-steps steps each)
        let mut reqdirs: Vec<usize> = std::fs::read_dir(dir).expect("dump dir")
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let n = e.file_name().to_string_lossy().to_string();
                n.strip_prefix('r').and_then(|s| s.parse::<usize>().ok())
            })
            .collect();
        reqdirs.sort_unstable();
        let list = |meta: &str, key: &str| -> Vec<u32> {
            meta.lines().find(|l| l.starts_with(key))
                .and_then(|l| l.split('[').nth(1)).and_then(|r| r.split(']').next())
                .map(|r| r.split(',').filter_map(|x| x.trim().parse::<u32>().ok()).collect())
                .unwrap_or_default()
        };
        let mut grand_match = 0usize; let mut grand_steps = 0usize; let mut grand_d0 = 0usize;
        // chain-formula tally: cur (base+resid, as served) vs base-only, per chain position
        let mut cur_hit = [0usize; 7]; let mut base_hit = [0usize; 7]; let mut chain_n = 0usize;
        let rank = MARKOV_RANK;
        let w1 = &oracle.weights.w1;
        let w2 = &oracle.weights.w2;
        let argmx = |v: &[f32]| -> usize {
            let mut bi = 0usize; let mut bv = f32::NEG_INFINITY;
            for (i, x) in v.iter().enumerate() { if *x > bv { bv = *x; bi = i; } }
            bi
        };
        for &r in &reqdirs {
            let rdir = format!("{dir}/r{r}");
            let raw = match std::fs::read(format!("{rdir}/taps_wide.bin")) { Ok(x) => x, Err(_) => continue };
            let plen = raw.len() / (2 * gb10_inference::dflash2::tap_concat_dim());
            let mut taps: Vec<bf16> = Vec::with_capacity(plen * gb10_inference::dflash2::tap_concat_dim());
            for c in raw.chunks_exact(2) { taps.push(bf16::from_le_bytes([c[0], c[1]])); }
            let steps_meta: Vec<(u32, Vec<u32>, Vec<u32>, usize)> = (0..24).filter_map(|k| {
                let meta = std::fs::read_to_string(format!("{rdir}/step{k}/meta.txt")).ok()?;
                let anchor = meta.lines().find(|l| l.starts_with("anchor"))?
                    .split_whitespace().nth(1)?.parse().ok()?;
                let nacc: usize = meta.lines().find(|l| l.starts_with("nacc"))?
                    .split_whitespace().nth(1)?.parse().ok()?;
                Some((anchor, list(&meta, "drafts"), list(&meta, "preds"), nacc))
            }).collect();
            if steps_meta.is_empty() { continue; }
            println!("== r{r}: {plen} prime cols, {} captured steps ==", steps_meta.len());
            round.reset();
            let taps_dev = dev.htod_sync_copy(&taps).expect("taps upload");
            round.prime_window(&taps_dev, plen, 0).expect("prime real taps");
            for (k, &(anchor, ref served, ref preds, nacc)) in steps_meta.iter().enumerate() {
                let dev_toks = round.draft_round_dev(anchor).expect("draft replay");
                let m = (nacc + 1).min(8);
                let same = dev_toks[..7.min(dev_toks.len())] == served[..7.min(served.len())];
                let d0ok = dev_toks.first() == preds.first();
                grand_match += same as usize; grand_steps += 1; grand_d0 += d0ok as usize;
                // per-position formula table: the round now runs the REFERENCE chain (bias at
                // every position, prev seeded by the ANCHOR); the dump's row k contains
                // resid(prev_k) with prev_0 = anchor. Strip with the SERVED chain's prevs to
                // recover base rows, then re-chain base-only host-side. (On OLD dumps —
                // pre-anchor-bias rounds — row 0 carried no resid, so only rows k>=1 compare.)
                if let Ok(lraw) = std::fs::read(format!("{rdir}/step{k}/logits.bin")) {
                    if lraw.len() >= 7 * DV * 4 {
                        let lg: Vec<f32> = lraw.chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                        chain_n += 1;
                        let mut toks_b = [0u32; 7];
                        let mut prev_s = anchor as usize;
                        for kk in 0..7 {
                            if preds.get(kk).copied() == Some(served.get(kk).copied().unwrap_or(0)) {
                                cur_hit[kk] += 1;
                            }
                            // strip the served chain's resid (prev = served chain's own prev)
                            let w1s = &w1[prev_s * rank..(prev_s + 1) * rank];
                            let mut base = vec![0.0f32; DV];
                            for o in 0..DV {
                                let w2row = &w2[o * rank..(o + 1) * rank];
                                let mut s = 0.0f32;
                                for i in 0..rank { s += w2row[i] * w1s[i]; }
                                base[o] = lg[kk * DV + o] - s;
                            }
                            if kk + 1 < served.len() { prev_s = served[kk] as usize; }
                            toks_b[kk] = argmx(&base) as u32;
                            if preds.get(kk) == Some(&toks_b[kk]) { base_hit[kk] += 1; }
                        }
                    }
                }
                if k < 2 || !same || !d0ok {
                    println!("  r{r} step{k}: anchor {anchor} | replay {dev_toks:?} | served {served:?} | trunk d0 {} | replay==served {same} d0==trunk {d0ok}",
                             preds.first().map(|x| x.to_string()).unwrap_or_default());
                }
                let sraw = std::fs::read(format!("{rdir}/step{k}/staging.bin")).expect("staging");
                let st: Vec<bf16> = sraw.chunks_exact(2).map(|c| bf16::from_le_bytes([c[0], c[1]])).collect();
                assert_eq!(st.len(), 8 * gb10_inference::dflash2::tap_concat_dim(), "staging dump must be 8 columns");
                let mut chunk_f = vec![0.0f32; 8 * gb10_inference::dflash2::tap_concat_dim()];
                for (i, b) in st.iter().enumerate() { chunk_f[i] = b.to_f32(); }
                round.upload_chunk(&chunk_f).expect("upload chunk");
                round.inject_dev(m).expect("inject");
            }
        }
        println!("  replay==served on {grand_match}/{grand_steps} steps; d0==trunk on {grand_d0}/{grand_steps} steps");
        println!("  chain per-position trunk-hits (n={chain_n}): served(base+resid) {cur_hit:?} | base-only {base_hit:?}");
        check("real-taps replay reproduces served drafts", grand_steps > 0 && grand_match == grand_steps);

        // ---- Ground-truth compare (--with-trunk): prefill-capture the SAME positions the
        // decode-capture fed the ring, and diff. The prime (prefill) capture is PROVEN good —
        // step 0's d0 matched the trunk — so its columns for positions 61..63 are ground truth.
        if args.iter().any(|a| a == "--with-trunk" || a == "--gt-reprime") && reqdirs.first() == Some(&0) {
            let gt_reprime = args.iter().any(|a| a == "--gt-reprime");
            let rdir0 = format!("{dir}/r0");
            let prow = std::fs::read(format!("{rdir0}/prompt.bin")).expect("prompt.bin");
            let prompt: Vec<u32> = prow.chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            let (mut gpu, _) = load_model_gpu(&trunk_dir, None, 1);
            let (_r2, _sink, prime2) = load_dspark_round_dir(&mut gpu, 4096, draft_dir, None)
                .expect("dspark triple");
            gpu.set_dspark_prime_sink(prime2.clone());
            let max_seq_len = (prompt.len() + 256).next_power_of_two();
            let mut pool = gb10_inference::gpu::Pool::new(gpu.dev().clone());
            let kv_stride = max_seq_len;
            let mut prime_cols = |ptoks: &[u32]| -> Vec<f32> {
                let mut state = gpu.new_batch_state(1, 1, max_seq_len);
                let (_t, _h) = gpu.prefill_batch(&mut pool, ptoks, &mut state, 0, kv_stride, 0);
                gpu.sync_stream();
                let cols: Vec<half::bf16> = gpu.dev().dtoh_sync_copy(&prime2.taps).unwrap();
                let n = ptoks.len().min(cols.len() / gb10_inference::dflash2::tap_concat_dim());
                cols[..n * gb10_inference::dflash2::tap_concat_dim()].iter().map(|b| b.to_f32()).collect()
            };
            let col = |taps: &[f32], c: usize| -> Vec<f32> {
                taps[c * gb10_inference::dflash2::tap_concat_dim()..(c + 1) * gb10_inference::dflash2::tap_concat_dim()].to_vec()
            };
            // ---- The tap-noise discriminator: re-prime the ring from PREFILL ground-truth taps
            // of the true committed stream (zero decode-capture/FP8-verify noise) and measure
            // the d0 hit-rate vs the served 9/24 baseline. Jump ⇒ the gap is tap noise; flat ⇒
            // artifact semantics. true stream = prompt ++ [anchor_0] ++ per-step preds[0..nacc+1);
            // at step k the ring holds taps[0..nprev) = all but the last token (lag-one design).
            if gt_reprime {
                let steps_meta: Vec<(u32, Vec<u32>, Vec<u32>, usize)> = (0..24).filter_map(|k| {
                    let meta = std::fs::read_to_string(format!("{rdir0}/step{k}/meta.txt")).ok()?;
                    let anchor = meta.lines().find(|l| l.starts_with("anchor"))?
                        .split_whitespace().nth(1)?.parse().ok()?;
                    let nacc: usize = meta.lines().find(|l| l.starts_with("nacc"))?
                        .split_whitespace().nth(1)?.parse().ok()?;
                    let ls = |key: &str| -> Vec<u32> {
                        meta.lines().find(|l| l.starts_with(key))
                            .and_then(|l| l.split('[').nth(1)).and_then(|r| r.split(']').next())
                            .map(|r| r.split(',').filter_map(|x| x.trim().parse().ok()).collect())
                            .unwrap_or_default()
                    };
                    Some((anchor, ls("drafts"), ls("preds"), nacc))
                }).collect();
                let mut stream = prompt.clone();
                stream.push(steps_meta[0].0);   // anchor_0 at position 61
                let mut hits = 0usize;
                println!("  [gt-reprime] re-priming from prefill GT taps per step …");
                for (k, (anchor, _served, preds, nacc)) in steps_meta.iter().enumerate() {
                    let np = stream.len() - 1;          // nprev: ring rows [0..np)
                    let mut gt = prime_cols(&stream[..np]); // GT taps for [0..np)
                    // A/B: reverse the 5 tap chunks per column (fc pairing convention)
                    if gb10_inference::opts::var(gb10_inference::opt!("dspark-tap-rev")).map(|v| v == "1").unwrap_or(false) {
                        let c5 = gb10_inference::dflash2::tap_concat_dim() / 5;
                        for col in gt.chunks_mut(gb10_inference::dflash2::tap_concat_dim()) {
                            for i in 0..2 {
                                for j in 0..c5 {
                                    let (a, b) = (i * c5 + j, (4 - i) * c5 + j);
                                    col.swap(a, b);
                                }
                            }
                        }
                    }
                    let tb: Vec<bf16> = gt.iter().map(|&x| bf16::from_f32(x)).collect();
                    round.reset();
                    let tdev = dev.htod_sync_copy(&tb).expect("gt taps upload");
                    round.prime_window(&tdev, np, 0).expect("gt prime");
                    let toks = round.draft_round_dev(*anchor).expect("gt draft");
                    let ok = toks.first() == preds.first();
                    hits += ok as usize;
                    if k < 4 || ok {
                        println!("  [gt-reprime] step{k}: anchor {anchor} d0 {} (trunk {}) {}",
                                 toks[0], preds.first().unwrap_or(&0), if ok { "HIT" } else { "miss" });
                    }
                    stream.extend(preds.iter().take(nacc + 1));
                }
                println!("  [gt-reprime] d0==trunk on {hits}/{} steps (served-noise baseline: 9/24)",
                         steps_meta.len());
                check("gt-reprime improves d0 hit-rate over the noise baseline", hits > 9);
            }
            // sanity: prime capture of the prompt == the serving taps_wide
            let raw0 = std::fs::read(format!("{rdir0}/taps_wide.bin")).expect("taps_wide r0");
            let taps0: Vec<f32> = raw0.chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect();
            let cap61 = prime_cols(&prompt);
            let rel = rel_l2(&cap61, &taps0);
            println!("  [gt] prime-capture(prompt) vs serving taps_wide relL2 {rel:.3e} (expect ~0)");
            // ground truth for positions 61, 62 (the injected rows)
            let mut p2 = prompt.clone(); p2.push(1596);
            let cap62 = prime_cols(&p2);
            let mut p3 = p2.clone(); p3.push(1144);
            let cap63 = prime_cols(&p3);
            // decode-captured staging (step0) cols 0,1
            let sraw0 = std::fs::read(format!("{rdir0}/step0/staging.bin")).expect("staging0");
            let st0: Vec<f32> = sraw0.chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect();
            let r61 = rel_l2(&col(&st0, 0), &col(&cap62, 61));
            let r62 = rel_l2(&col(&st0, 1), &col(&cap63, 62));
            println!("  [gt] decode-capture col0 (1596@61) vs prefill GT: relL2 {r61:.3e}");
            println!("  [gt] decode-capture col1 (1144@62) vs prefill GT: relL2 {r62:.3e}");
            check("gt: decode capture matches prefill GT (col0)", r61 < 5e-2);
            check("gt: decode capture matches prefill GT (col1)", r62 < 5e-2);
        }
        if !all_pass.get() { std::process::exit(1); }
        println!("RESULT: REAL_TAPS_OK");
        return;
    }

    // (The earlier oracle-on-real-taps branch was folded into the replay above: with the REAL
    // embed+head the replay subsumes it, and the synth-table oracle comparison was uninformative
    // — the anchor embed was synthetic garbage.)

    // synthetic taps: 16 rows (8 prime + 8 inject), one deterministic generator
    let tap_scale = 0.5f32;
    let mut taps_rm = vec![0.0f32; 16 * gb10_inference::dflash2::tap_concat_dim()];   // row-major [16, 25600]
    for i in 0..16 {
        let r = synth.row(SyntheticTables::TABLE_TAPS, i as u32, gb10_inference::dflash2::tap_concat_dim(), tap_scale);
        taps_rm[i * gb10_inference::dflash2::tap_concat_dim()..(i + 1) * gb10_inference::dflash2::tap_concat_dim()].copy_from_slice(&r);
    }
    // the gemm x convention is TOKEN-major (column c at c*inn) — the prime buffer is just the
    // first 8 tap ROWS, contiguous (row-major [8, 25600])
    let taps_b: Vec<bf16> = taps_rm[..8 * gb10_inference::dflash2::tap_concat_dim()].iter().map(|&x| bf16::from_f32(x)).collect();
    let taps_dev = dev.htod_sync_copy(&taps_b).expect("taps upload");

    // host replicate of the ctx-KV write (k/v_proj + perhead k_norm + rope at offset
    // positions) — isolates prime/inject ring content from the block pass
    let ctx_kv_rep = |li: usize, th_rows: &[f32], m: usize, pos0: usize| -> (Vec<f32>, Vec<f32>) {
        let l = &art.weights.layers[li];
        let lin = |wv: &[f32], x: &[f32], outn: usize, inn: usize, rows: usize| -> Vec<f32> {
            let mut o = vec![0.0f32; rows * outn];
            for r in 0..rows {
                for oi in 0..outn {
                    let wr = &wv[oi * inn..(oi + 1) * inn];
                    let mut acc = 0.0f32;
                    for i in 0..inn { acc += wr[i] * x[r * inn + i]; }
                    o[r * outn + oi] = acc;
                }
            }
            o
        };
        let freqs = gb10_inference::dspark::oracle::yaarn_freqs(128, 1e7, 32.0, 8192, 32, 1);
        let mut k = lin(&l.k_proj, th_rows, 8 * 128, DH, m);
        let v = lin(&l.v_proj, th_rows, 8 * 128, DH, m);
        for r in 0..m {
            for h2 in 0..8usize {
                let base = (r * 8 + h2) * 128;
                let ss: f32 = k[base..base + 128].iter().map(|x| x * x).sum();
                let inv = 1.0f32 / (ss / 128.0 + 1e-6).sqrt();
                for d2 in 0..128 { k[base + d2] *= inv * l.k_norm[d2]; }
            }
        }
        // rope: NeoX half-split (j, j+64) with the YaRN mscale folded into cos/sin — the
        // reference conventions now shared by the round, the oracle, and this replicate.
        let mscale = 0.1f32 * 32.0f32.ln() + 1.0f32;
        for r in 0..m {
            for h2 in 0..8usize {
                let base = (r * 8 + h2) * 128;
                for j in 0..64usize {
                    let ang = (pos0 + r) as f32 * freqs[j];
                    let (c, sn) = (ang.cos() * mscale, ang.sin() * mscale);
                    let x1 = k[base + j];
                    let x2 = k[base + 64 + j];
                    k[base + j] = x1 * c - x2 * sn;
                    k[base + 64 + j] = x2 * c + x1 * sn;
                }
            }
        }
        (k, v)
    };
    let kv_gate = |tag: &str, li: usize, rows_off: usize, m: usize, th_rows: &[f32],
                   round: &gb10_inference::dspark::round::DsparkRound| {
        let (kd, vd) = round.dump_kv_rows(li, rows_off + m).expect("dump_kv_rows");
        let (kh, vh) = ctx_kv_rep(li, th_rows, m, rows_off);
        // both sides → (kvh, r_local, d): the device dump covers rows [0, off+m) per kvh
        // (R = off+m, row r of kvh h at h*R*stride + r*stride); the host replicate covers the
        // m new rows (r, kvh, d). Restrict the device to [off, off+m) and reorder to match.
        let big_r = rows_off + m;
        // device dump (kvh, r, d): element (h, r) starts at h*big_r*128 + r*128
        let mut kd2 = Vec::with_capacity(8 * m * 128);
        let mut vd2 = Vec::with_capacity(8 * m * 128);
        for h in 0..8usize {
            for r in rows_off..big_r {
                let base = h * big_r * 128 + r * 128;
                kd2.extend_from_slice(&kd[base..base + 128]);
                vd2.extend_from_slice(&vd[base..base + 128]);
            }
        }
        // host replicate (r, kvh, d): row r starts at r*(8*128)
        let mut kh2 = Vec::with_capacity(8 * m * 128);
        let mut vh2 = Vec::with_capacity(8 * m * 128);
        for h in 0..8usize {
            for r in 0..m {
                let base = r * (8 * 128) + h * 128;
                kh2.extend_from_slice(&kh[base..base + 128]);
                vh2.extend_from_slice(&vh[base..base + 128]);
            }
        }
        let rlk = rel_l2(&kd2, &kh2);
        let rlv = rel_l2(&vd2, &vh2);
        check(&format!("{tag} L{li} ctx-KV k relL2 {rlk:.3e} < 5e-3"), rlk < 5e-3);
        check(&format!("{tag} L{li} ctx-KV v relL2 {rlv:.3e} < 5e-3"), rlv < 5e-3);
    };

    let run_round_checks = |tag: &str, anchor_tok: u32, ctx_rows: usize,
                            round: &mut gb10_inference::dspark::round::DsparkRound,
                            oracle: &DsparkOracle| {
        println!("== {tag}: ctx {ctx_rows}, anchor {anchor_tok} ==");
        // device round
        let toks_dev = round.draft_round_dev(anchor_tok).expect("draft_round_dev");
        let h_dev = round.read_h_final().expect("read_h_final");
        let lg_dev = round.dump_logits().expect("dump_logits");
        let lat_dev = round.read_latents().expect("read_latents");
        // oracle on the SAME taps
        let ctx = RoundCtx {
            tap_hiddens: taps_rm[..ctx_rows * gb10_inference::dflash2::tap_concat_dim()].to_vec(),
            anchor: anchor_tok,
            confidence_threshold: 0.15,
        };
        let out = oracle.run_round(&ctx);
        // gate 1: h_final rel-L2 (bf16 device staging vs f32 oracle) — all 7 rows
        let h_or = out.h.clone();
        let rl = rel_l2(&h_dev, &h_or);
        // gate 1b: the device round is a BF16-STAGING implementation (every stage boundary
        // rounds to bf16) vs the oracle's f32; the gap compounds through the 5 layers +
        // attention (~1e-1 at ctx 16). DF2 head-gate precedent: documented rel-L2, not a
        // defect. A STRUCTURAL bug (layout/rope/ring) shows up as ~1.0; the gate catches that.
        check(&format!("{tag} h_final relL2 {rl:.3e} < 2e-1 (bf16-staging documented)"), rl < 2e-1);
        // gate 2: the markov chain on the DEVICE logits (exact); dump_logits is token-major
        // [7, VOCAB] — the oracle chain's own layout.
        let lg0 = lg_dev;
        let mo = oracle.markov_chain(&lg0, anchor_tok);
        check(&format!("{tag} chain tokens EXACT (dev {:?} vs host {:?})",
                       &toks_dev[..7], &mo.tokens[..]),
              toks_dev[..7] == mo.tokens);
        // gate 3: latents exact (both f32 upcasts of the same bf16 W1 rows)
        let maxd = lat_dev.iter().zip(mo.latents.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        check(&format!("{tag} latents EXACT (max|d| {maxd:.3e})"), maxd == 0.0);
        // gate 4: confidence head. 4a (SHARP): the host mirror vs the oracle on the SAME
        // device h readback + latents — identical f32 inputs, identical accumulation order →
        // EXACT. This isolates mirror correctness from inherited h noise. 4b (diagnostic):
        // survival vs the oracle's own f32 h — the sigmoid amplifies the bf16-staging h gap
        // (relL2 ~2.7e-2) nonlinearly; under the reference norm magnitudes one position can
        // swing ~0.5. Adaptive verify is OFF in serving (confidence is diagnostic-only), so
        // this arm documents the inherited variance rather than gating on it.
        let (_, survival, k_verify) = round.confidence(&h_dev, &lat_dev);
        let co_mirror = oracle.confidence(&h_dev, &lat_dev, 0.15);
        let mmax = survival.iter().zip(co_mirror.survival.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        check(&format!("{tag} confidence mirror on same h (max|d| {mmax:.3e} < 1e-6; f32 sigmoid-branch ulps)"), mmax < 1e-6);
        let mut pmax = 0.0f32;
        for k in 0..6 {
            pmax = pmax.max((survival[k] - out.survival[k]).abs());
        }
        check(&format!("{tag} confidence survival (oracle-f32-h) max|d| {pmax:.3e} < 5e-1 (k_verify {k_verify} vs {}; inherited bf16-staging h gap, documented)",
                       out.k_verify), pmax < 5e-1);
        // gate 5: full logits rel-L2 vs the oracle's synth head (the borrowed-head path gate;
        // bf16 rounding of the table + binv accumulation order → loose bound)
        let rl_lg = rel_l2(&lg0, &out.logits0);
        check(&format!("{tag} logits relL2 {rl_lg:.3e} < 5e-1 (bf16-staging documented)"), rl_lg < 5e-1);
    };

    let run_round_checks_ctx0 = |tag: &str, anchor_tok: u32,
                                 round: &mut gb10_inference::dspark::round::DsparkRound,
                                 oracle: &DsparkOracle| {
        println!("== {tag} ==");
        let toks_dev = round.draft_round_dev(anchor_tok).expect("draft_round_dev ctx0");
        let h_dev = round.read_h_final().expect("read_h_final ctx0");
        let lg_dev = round.dump_logits().expect("dump_logits ctx0");
        let ctx = RoundCtx {
            tap_hiddens: Vec::new(),
            anchor: anchor_tok,
            confidence_threshold: 0.15,
        };
        let out = oracle.run_round(&ctx);
        let h_or = out.h.clone();
        let rl = rel_l2(&h_dev, &h_or);
        // gate 1b: the device round is a BF16-STAGING implementation (every stage boundary
        // rounds to bf16) vs the oracle's f32; the gap compounds through the 5 layers +
        // attention (~1e-1 at ctx 16). DF2 head-gate precedent: documented rel-L2, not a
        // defect. A STRUCTURAL bug (layout/rope/ring) shows up as ~1.0; the gate catches that.
        check(&format!("{tag} h_final relL2 {rl:.3e} < 2e-1 (bf16-staging documented)"), rl < 2e-1);
        // dump_logits is already token-major [7, VOCAB] = the oracle's logits0 layout
        check(&format!("{tag} chain tokens EXACT"),
              { let mo = oracle.markov_chain(&lg_dev, anchor_tok); toks_dev[..7] == mo.tokens });
        let rl_lg = rel_l2(&lg_dev, &out.logits0);
        check(&format!("{tag} logits relL2 {rl_lg:.3e} < 5e-1 (bf16-staging documented)"), rl_lg < 5e-1);
    };

    // ---- Round 0: ctx 0 (no prime) — isolates the BLOCK PASS (embed → layers → head) from
    // ---- the prime/ring-write path. The oracle runs the same empty-ctx round.
    run_round_checks_ctx0("round0 ctx0", 777, &mut round, &oracle);

    // ---- Round 1: prime ctx 8, draft from anchor 1234 ---------------------
    round.prime_window(&taps_dev, 8, 0).expect("prime_window 8");
    {
        // th gate: prime's fc+hidden_norm vs the oracle's tap projection (rows 0..8)
        let th_dev = round.dump_th(8).expect("dump_th");
        let ctx = RoundCtx {
            tap_hiddens: taps_rm[..8 * gb10_inference::dflash2::tap_concat_dim()].to_vec(),
            anchor: 1234,
            confidence_threshold: 0.15,
        };
        let out = oracle.run_round(&ctx);
        let rl = rel_l2(&th_dev, &out.th);
        check(&format!("prime th relL2 {rl:.3e} < 2e-2"), rl < 2e-2);
    }
    run_round_checks("round1 ctx8", 1234, 8, &mut round, &oracle);
    // ctx-KV gates: prime rows [0..8) at positions [0..8) — the oracle's own th feeds the host
    // replicate, so this isolates k/v_proj + k_norm + rope + the ring write from the block pass.
    for li in [0usize, 4] {
        let th_prime = oracle.tap_project(&taps_rm[..8 * gb10_inference::dflash2::tap_concat_dim()], 8);
        kv_gate("prime-ctx8", li, 0, 8, &th_prime, &round);
    }

    // ---- K-pipeline stage bisect: gemm → norm → rope → ring write, one layer, vs host. ----
    {
        let li = 0usize;
        let th8 = oracle.tap_project(&taps_rm[..8 * gb10_inference::dflash2::tap_concat_dim()], 8);
        let (g, nr, rp, ring) = round.probe_k_pipeline(li, &th8, 8, 0).expect("probe_k_pipeline");
        // host stages
        let l = &art.weights.layers[li];
        let lin = |wv: &[f32], x: &[f32], outn: usize, inn: usize, rows: usize| -> Vec<f32> {
            let mut o = vec![0.0f32; rows * outn];
            for r in 0..rows {
                for oi in 0..outn {
                    let wr = &wv[oi * inn..(oi + 1) * inn];
                    let mut acc = 0.0f32;
                    for i in 0..inn { acc += wr[i] * x[r * inn + i]; }
                    o[r * outn + oi] = acc;
                }
            }
            o
        };
        let hg = lin(&l.k_proj, &th8, 8 * 128, DH, 8);
        let mut hn = hg.clone();
        for r in 0..8 {
            for h2 in 0..8usize {
                let base = (r * 8 + h2) * 128;
                let ss: f32 = hn[base..base + 128].iter().map(|x| x * x).sum();
                let inv = 1.0f32 / (ss / 128.0 + 1e-6).sqrt();
                for d2 in 0..128 { hn[base + d2] *= inv * l.k_norm[d2]; }
            }
        }
        let freqs = gb10_inference::dspark::oracle::yaarn_freqs(128, 1e7, 32.0, 8192, 32, 1);
        let mscale = 0.1f32 * 32.0f32.ln() + 1.0f32;
        let mut hr = hn.clone();
        for r in 0..8 {
            for h2 in 0..8usize {
                let base = (r * 8 + h2) * 128;
                for j in 0..64usize {
                    let ang = r as f32 * freqs[j];
                    let (c, sn) = (ang.cos() * mscale, ang.sin() * mscale);
                    let x1 = hr[base + j];
                    let x2 = hr[base + 64 + j];
                    hr[base + j] = x1 * c - x2 * sn;
                    hr[base + 64 + j] = x2 * c + x1 * sn;
                }
            }
        }
        // device stage buffers are token-major [8, 1024]; restrict to rows [0..8) (all of them)
        let rl_g = rel_l2(&g, &hg);
        let rl_n = rel_l2(&nr, &hn);
        let rl_r = rel_l2(&rp, &hr);
        println!("  [kpipe] L{li} relL2 gemm {rl_g:.3e} norm {rl_n:.3e} rope {rl_r:.3e}");
        check(&format!("kpipe gemm relL2 {rl_g:.3e} < 1e-2"), rl_g < 1e-2);
        check(&format!("kpipe norm relL2 {rl_n:.3e} < 1e-2"), rl_n < 1e-2);
        check(&format!("kpipe rope relL2 {rl_r:.3e} < 1e-2"), rl_r < 1e-2);
        let _ = ring;
    }

    // ---- Round 2: inject 8 more committed taps (the serving step shape), ctx 16
    {
        // the inject's staging is the NEXT 8 committed tap rows, row-major (upload_chunk doc)
        round.upload_chunk(&taps_rm[8 * gb10_inference::dflash2::tap_concat_dim()..]).expect("upload_chunk");
        round.inject_dev(8).expect("inject_dev 8");
    }
    // ctx-KV gates: inject rows [8..16) at positions [8..16)
    for li in [0usize, 4] {
        let th_inj = oracle.tap_project(&taps_rm[8 * gb10_inference::dflash2::tap_concat_dim()..16 * gb10_inference::dflash2::tap_concat_dim()], 8);
        kv_gate("inject-ctx16", li, 8, 8, &th_inj, &round);
    }
    run_round_checks("round2 ctx16", 4321, 16, &mut round, &oracle);

    // ---- Round 2b: the SAME ctx 16 but primed in ONE window (no inject) — isolates whether
    // the ctx16 failure lives in the INJECT path's state or in the ctx16 shape itself.
    {
        round.reset();
        let mut taps64 = vec![0.0f32; 16 * gb10_inference::dflash2::tap_concat_dim()];
        taps64[..16 * gb10_inference::dflash2::tap_concat_dim()].copy_from_slice(&taps_rm[..16 * gb10_inference::dflash2::tap_concat_dim()]);
        let w16: Vec<bf16> = taps64.iter().map(|&x| bf16::from_f32(x)).collect();
        let w16_dev = dev.htod_sync_copy(&w16).expect("w16 upload");
        round.prime_window(&w16_dev, 16, 0).expect("prime 16");
        run_round_checks("round2b ctx16-prime-only", 4321, 16, &mut round, &oracle);
    }
    // ---- Round 2c: per-layer bisect at ctx16 (the inject-then-draft failing shape). ----
    {
        // rebuild the failing state: prime 8 + inject 8 (round2's path), then layer-by-layer
        round.reset();
        {
            let mut taps8 = vec![0.0f32; 8 * gb10_inference::dflash2::tap_concat_dim()];
            taps8.copy_from_slice(&taps_rm[..8 * gb10_inference::dflash2::tap_concat_dim()]);
            let w8: Vec<bf16> = taps8.iter().map(|&x| bf16::from_f32(x)).collect();
            let w8_dev = dev.htod_sync_copy(&w8).expect("w8 upload");
            round.prime_window(&w8_dev, 8, 0).expect("prime 8");
        }
        {
            let mut chunk = vec![0.0f32; 8 * gb10_inference::dflash2::tap_concat_dim()];
            chunk.copy_from_slice(&taps_rm[8 * gb10_inference::dflash2::tap_concat_dim()..16 * gb10_inference::dflash2::tap_concat_dim()]);
            round.upload_chunk(&chunk).expect("upload 8");
            round.inject_dev(8).expect("inject 8");
        }
        // host replicate of the block pass at ctx16 (embed + 5 layers), oracle primitives
        let w = &art.weights;
        let hnorm2 = |x: &mut Vec<f32>, wv: &[f32], rows: usize, n: usize| {
            for r in 0..rows {
                let ss: f32 = x[r * n..(r + 1) * n].iter().map(|v| v * v).sum();
                let inv = 1.0f32 / (ss / n as f32 + 1e-6).sqrt();
                for i in 0..n { x[r * n + i] = x[r * n + i] * inv * wv[i]; }
            }
        };
        let lin2 = |wv: &[f32], x: &[f32], outn: usize, inn: usize, rows: usize| -> Vec<f32> {
            let mut o = vec![0.0f32; rows * outn];
            for r in 0..rows {
                for oi in 0..outn {
                    let wr = &wv[oi * inn..(oi + 1) * inn];
                    let mut acc = 0.0f32;
                    for i in 0..inn { acc += wr[i] * x[r * inn + i]; }
                    o[r * outn + oi] = acc;
                }
            }
            o
        };
        // th for the ctx rows → the oracle's ctx KV (public API) at positions [0..16)
        let th16 = oracle.tap_project(&taps_rm[..16 * gb10_inference::dflash2::tap_concat_dim()], 16);
        let ctxkv = oracle.draft_kv_write(&th16, 16);
        let toks = [777u32, 248077, 248077, 248077, 248077, 248077, 248077];
        let freqs = gb10_inference::dspark::oracle::yaarn_freqs(128, 1e7, 32.0, 8192, 32, 1);
        let mut hh = vec![0.0f32; 7 * DH];
        for (r, t) in toks.iter().enumerate() {
            hh[r * DH..(r + 1) * DH].copy_from_slice(&synth.row(SyntheticTables::TABLE_EMBED, *t, DH, scale));
        }
        let silu = |x: f32| x / (1.0 + (-x).exp());
        for (li, l) in w.layers.iter().enumerate() {
            let mut ln = hh.clone();
            hnorm2(&mut ln, &l.input_ln, 7, DH);
            let q = lin2(&l.q_proj, &ln, 40 * 128, DH, 7);
            let (mut qh, mut kh) = (q.clone(), lin2(&l.k_proj, &ln, 8 * 128, DH, 7));
            let norm_heads = |x: &mut Vec<f32>, wv: &[f32], heads: usize| {
                for r in 0..7 {
                    for h2 in 0..heads {
                        let base = (r * heads + h2) * 128;
                        let ss: f32 = x[base..base + 128].iter().map(|v| v * v).sum();
                        let inv = 1.0f32 / (ss / 128.0 + 1e-6).sqrt();
                        for d2 in 0..128 { x[base + d2] *= inv * wv[d2]; }
                    }
                }
            };
            norm_heads(&mut qh, &l.q_norm, 40);
            norm_heads(&mut kh, &l.k_norm, 8);
            // NeoX half-split + mscale (reference conventions)
            let mscale = 0.1f32 * 32.0f32.ln() + 1.0f32;
            let rope_on = |x: &mut Vec<f32>, heads: usize| {
                for r in 0..7usize {
                    for h2 in 0..heads {
                        let base = (r * heads + h2) * 128;
                        for j in 0..64usize {
                            let ang = (16 + r) as f32 * freqs[j];
                            let (c, sn) = (ang.cos() * mscale, ang.sin() * mscale);
                            let x1 = x[base + j];
                            let x2 = x[base + 64 + j];
                            x[base + j] = x1 * c - x2 * sn;
                            x[base + 64 + j] = x2 * c + x1 * sn;
                        }
                    }
                }
            };
            rope_on(&mut qh, 40);
            rope_on(&mut kh, 8);
            let vb = lin2(&l.v_proj, &ln, 8 * 128, DH, 7);
            // attention over ctx [0..16) + block [16..23)
            let ntot = 23usize;
            let mut attn = vec![0.0f32; 7 * 40 * 128];
            let sc = 1.0f32 / (128.0f32).sqrt();
            for r in 0..7 {
                for h2 in 0..40 {
                    let kvh = h2 / 5;
                    let qrow = &qh[(r * 40 + h2) * 128..(r * 40 + h2 + 1) * 128];
                    let mut scores = vec![0.0f32; ntot];
                    let mut mmax = f32::NEG_INFINITY;
                    for j in 0..ntot {
                        let ksrc: &[f32] = if j < 16 { &ctxkv.layers[li].k } else { &kh };
                        let jr = if j < 16 { j } else { j - 16 };
                        let krow = &ksrc[(jr * 8 + kvh) * 128..(jr * 8 + kvh + 1) * 128];
                        let mut sacc = 0.0f32;
                        for d2 in 0..128 { sacc += qrow[d2] * krow[d2]; }
                        sacc *= sc;
                        scores[j] = sacc;
                        if sacc > mmax { mmax = sacc; }
                    }
                    let mut sum = 0.0f32;
                    for s2 in scores.iter_mut() { *s2 = (*s2 - mmax).exp(); sum += *s2; }
                    let orow = &mut attn[(r * 40 + h2) * 128..(r * 40 + h2 + 1) * 128];
                    for j in 0..ntot {
                        let wgt = scores[j] / sum;
                        let vsrc: &[f32] = if j < 16 { &ctxkv.layers[li].v } else { &vb };
                        let jr = if j < 16 { j } else { j - 16 };
                        let vrow = &vsrc[(jr * 8 + kvh) * 128..(jr * 8 + kvh + 1) * 128];
                        for d2 in 0..128 { orow[d2] += wgt * vrow[d2]; }
                    }
                }
            }
            let ao = lin2(&l.o_proj, &attn, DH, 40 * 128, 7);
            let mut h2v = vec![0.0f32; 7 * DH];
            for i in 0..7 * DH { h2v[i] = hh[i] + ao[i]; }
            let mut h2n = h2v.clone();
            hnorm2(&mut h2n, &l.post_ln, 7, DH);
            let gate = lin2(&l.gate_proj, &h2n, 10240, DH, 7);
            let up = lin2(&l.up_proj, &h2n, 10240, DH, 7);
            let mut ffn = vec![0.0f32; 7 * 10240];
            for i in 0..7 * 10240 { ffn[i] = silu(gate[i]) * up[i]; }
            let down = lin2(&l.down_proj, &ffn, DH, 10240, 7);
            for i in 0..7 * DH { hh[i] = h2v[i] + down[i]; }

            round.draft_round_partial_dev(777, li + 1).expect("partial ctx16");
            let hd = round.dump_h(7).expect("dump_h");
            let rl = rel_l2(&hd, &hh);
            check(&format!("bisect ctx16 L{} h relL2 {rl:.3e} < 3e-1", li), rl < 3e-1);
        }
        round.reset();
    }
    // ---- Round 2d: THE FAILING SEQUENCE EXACTLY — prime 8, FULL ctx8 draft (head+chain),
    // inject 8, then the per-layer bisect at ctx16. If this fails where 2c passed, the ctx8
    // draft's own execution poisons the next round.
    {
        round.reset();
        {
            let mut taps8 = vec![0.0f32; 8 * gb10_inference::dflash2::tap_concat_dim()];
            taps8.copy_from_slice(&taps_rm[..8 * gb10_inference::dflash2::tap_concat_dim()]);
            let w8: Vec<bf16> = taps8.iter().map(|&x| bf16::from_f32(x)).collect();
            let w8_dev = dev.htod_sync_copy(&w8).expect("w8 upload");
            round.prime_window(&w8_dev, 8, 0).expect("prime 8");
        }
        let _ = round.draft_round_dev(777).expect("ctx8 draft");
        {
            let mut chunk = vec![0.0f32; 8 * gb10_inference::dflash2::tap_concat_dim()];
            chunk.copy_from_slice(&taps_rm[8 * gb10_inference::dflash2::tap_concat_dim()..16 * gb10_inference::dflash2::tap_concat_dim()]);
            round.upload_chunk(&chunk).expect("upload 8");
            round.inject_dev(8).expect("inject 8");
        }
        // (the host-side ctx16 replicate hh was left at the final-layer values by 2c; rebuild it)
        // — cheap enough to just rerun the chain of partials and compare layer-by-layer against
        // the oracle's public final hidden instead: compare each partial's h to the 2c target is
        // not possible here, so gate the FINAL full draft's h vs the oracle ctx16 h.
        let toks_dev = round.draft_round_dev(777).expect("full ctx16 draft after sequence");
        let h_dev = round.read_h_final().expect("read_h_final");
        let ctx = RoundCtx {
            tap_hiddens: taps_rm[..16 * gb10_inference::dflash2::tap_concat_dim()].to_vec(),
            anchor: 777,
            confidence_threshold: 0.15,
        };
        let out = oracle.run_round(&ctx);
        let h_or = out.h.clone();
        let rl = rel_l2(&h_dev, &h_or);
        check(&format!("round2d ctx16-after-sequence h relL2 {rl:.3e} < 2e-1"), rl < 2e-1);
        check(&format!("round2d chain tokens EXACT"),
              { let mo = oracle.markov_chain(&round.dump_logits().expect("lg"), 777); toks_dev[..7] == mo.tokens });
        round.reset();
    }

    // ---- Round 3: the SERVING shape — a WIDE prime window (n=61, the chat-prompt case the
    // serving prefill hits in one PREFILL_CHUNK window) then a 3-col inject. The ctx16 round
    // above only primes at n=8; a wide-n bug in the prime path (rmsnorm/gather/rope/write_kv
    // at window widths) would show here and NOWHERE else.
    {
        round.reset();
        let mut taps64 = vec![0.0f32; 64 * gb10_inference::dflash2::tap_concat_dim()];
        for i in 0..64 {
            let r = synth.row(SyntheticTables::TABLE_TAPS, (100 + i) as u32, gb10_inference::dflash2::tap_concat_dim(), tap_scale);
            taps64[i * gb10_inference::dflash2::tap_concat_dim()..(i + 1) * gb10_inference::dflash2::tap_concat_dim()].copy_from_slice(&r);
        }
        let w61: Vec<bf16> = taps64[..61 * gb10_inference::dflash2::tap_concat_dim()].iter().map(|&x| bf16::from_f32(x)).collect();
        let w61_dev = dev.htod_sync_copy(&w61).expect("wide taps upload");
        round.prime_window(&w61_dev, 61, 0).expect("wide prime 61");
        let toks_dev = round.draft_round_dev(777).expect("draft ctx61");
        let h_dev = round.read_h_final().expect("read_h_final");
        let lg_dev = round.dump_logits().expect("dump_logits");
        let ctx = RoundCtx {
            tap_hiddens: taps64[..61 * gb10_inference::dflash2::tap_concat_dim()].to_vec(),
            anchor: 777,
            confidence_threshold: 0.15,
        };
        let out = oracle.run_round(&ctx);
        let h_or = out.h.clone();
        let rl = rel_l2(&h_dev, &h_or);
        check(&format!("round3 ctx61 h_final relL2 {rl:.3e} < 2e-1 (bf16-staging documented)"), rl < 2e-1);
        let mo = oracle.markov_chain(&lg_dev, 777);
        check(&format!("round3 ctx61 chain tokens EXACT"), toks_dev[..7] == mo.tokens);
        // then the serving step shape: 3 committed taps injected (nacc=2 style)
        {
            let mut chunk = vec![0.0f32; 8 * gb10_inference::dflash2::tap_concat_dim()];
            chunk[..3 * gb10_inference::dflash2::tap_concat_dim()].copy_from_slice(&taps64[61 * gb10_inference::dflash2::tap_concat_dim()..64 * gb10_inference::dflash2::tap_concat_dim()]);
            round.upload_chunk(&chunk).expect("upload_chunk 3");
            round.inject_dev(3).expect("inject_dev 3");
        }
        let toks_dev = round.draft_round_dev(555).expect("draft ctx64");
        let lg_dev = round.dump_logits().expect("dump_logits");
        let ctx = RoundCtx {
            tap_hiddens: taps64[..64 * gb10_inference::dflash2::tap_concat_dim()].to_vec(),
            anchor: 555,
            confidence_threshold: 0.15,
        };
        let out = oracle.run_round(&ctx);
        check(&format!("round3 ctx64 chain tokens EXACT"),
              { let mo = oracle.markov_chain(&lg_dev, 555); toks_dev[..7] == mo.tokens });
        let _ = out;
    }

    // ---- Layer bisect (ctx0): device h after each layer vs a host replicate. ----
    if gb10_inference::opts::var(gb10_inference::opt!("dspark-bisect")).is_ok() {
        println!("== layer bisect (ctx0) ==");
        // host replicate of the block pass (the oracle's exact math on the artifact weights)
        let w = &art.weights;
        let hnorm = |x: &mut Vec<f32>, wv: &[f32], rows: usize, n: usize| {
            for r in 0..rows {
                let ss: f32 = x[r * n..(r + 1) * n].iter().map(|v| v * v).sum();
                let inv = 1.0f32 / (ss / n as f32 + 1e-6).sqrt();
                for i in 0..n {
                    x[r * n + i] = x[r * n + i] * inv * wv[i];
                }
            }
        };
        let lin = |wv: &[f32], x: &[f32], outn: usize, inn: usize, rows: usize| -> Vec<f32> {
            let mut o = vec![0.0f32; rows * outn];
            for r in 0..rows {
                for oi in 0..outn {
                    let wr = &wv[oi * inn..(oi + 1) * inn];
                    let mut acc = 0.0f32;
                    for i in 0..inn {
                        acc += wr[i] * x[r * inn + i];
                    }
                    o[r * outn + oi] = acc;
                }
            }
            o
        };
        let row = |t: u32| -> Vec<f32> { synth.row(SyntheticTables::TABLE_EMBED, t, DH, scale) };
        let mut hh = vec![0.0f32; 7 * DH];
        let toks = [777u32, 248077, 248077, 248077, 248077, 248077, 248077];
        for (r, t) in toks.iter().enumerate() {
            hh[r * DH..(r + 1) * DH].copy_from_slice(&row(*t));
        }
        let silu = |x: f32| x / (1.0 + (-x).exp());
        for (li, l) in w.layers.iter().enumerate() {
            let mut ln = hh.clone();
            hnorm(&mut ln, &l.input_ln, 7, DH);
            let q = lin(&l.q_proj, &ln, 40 * 128, DH, 7);
            let (mut qh, mut kh): (Vec<f32>, Vec<f32>) = (q.clone(), lin(&l.k_proj, &ln, 8 * 128, DH, 7));
            let norm_heads = |x: &mut Vec<f32>, wv: &[f32], heads: usize| {
                for r in 0..7 {
                    for h2 in 0..heads {
                        let base = (r * heads + h2) * 128;
                        let ss: f32 = x[base..base + 128].iter().map(|v| v * v).sum();
                        let inv = 1.0f32 / (ss / 128.0 + 1e-6).sqrt();
                        for d2 in 0..128 {
                            x[base + d2] *= inv * wv[d2];
                        }
                    }
                }
            };
            norm_heads(&mut qh, &l.q_norm, 40);
            norm_heads(&mut kh, &l.k_norm, 8);
            // rope at positions 0..7 (ctx0) — freqs from the same yaarn table; NeoX half-split
            // pairing + mscale, the reference conventions.
            let freqs = gb10_inference::dspark::oracle::yaarn_freqs(
                128, 1e7, 32.0, 8192, 32, 1);
            let mscale = 0.1f32 * 32.0f32.ln() + 1.0f32;
            let rope_on = |x: &mut Vec<f32>, heads: usize| {
                for r in 0..7usize {
                    for h2 in 0..heads {
                        let base = (r * heads + h2) * 128;
                        for j in 0..64usize {
                            let ang = r as f32 * freqs[j];
                            let (c, sn) = (ang.cos() * mscale, ang.sin() * mscale);
                            let x1 = x[base + j];
                            let x2 = x[base + 64 + j];
                            x[base + j] = x1 * c - x2 * sn;
                            x[base + 64 + j] = x2 * c + x1 * sn;
                        }
                    }
                }
            };
            rope_on(&mut qh, 40);
            rope_on(&mut kh, 8);
            let vb = lin(&l.v_proj, &ln, 8 * 128, DH, 7);
            // attention over the 7 block keys only (ctx0)
            let mut attn = vec![0.0f32; 7 * 40 * 128];
            let sc = 1.0f32 / (128.0f32).sqrt();
            for r in 0..7 {
                for h2 in 0..40 {
                    let kvh = h2 / 5;
                    let qrow = &qh[(r * 40 + h2) * 128..(r * 40 + h2 + 1) * 128];
                    let mut scores = vec![0.0f32; 7];
                    let mut m = f32::NEG_INFINITY;
                    for j in 0..7 {
                        let krow = &kh[(j * 8 + kvh) * 128..(j * 8 + kvh + 1) * 128];
                        let mut sacc = 0.0f32;
                        for d2 in 0..128 {
                            sacc += qrow[d2] * krow[d2];
                        }
                        sacc *= sc;
                        scores[j] = sacc;
                        if sacc > m { m = sacc; }
                    }
                    let mut sum = 0.0f32;
                    for s2 in scores.iter_mut() { *s2 = (*s2 - m).exp(); sum += *s2; }
                    let orow = &mut attn[(r * 40 + h2) * 128..(r * 40 + h2 + 1) * 128];
                    for j in 0..7 {
                        let wgt = scores[j] / sum;
                        let vrow = &vb[(j * 8 + kvh) * 128..(j * 8 + kvh + 1) * 128];
                        for d2 in 0..128 {
                            orow[d2] += wgt * vrow[d2];
                        }
                    }
                }
            }
            let ao = lin(&l.o_proj, &attn, DH, 40 * 128, 7);
            let mut h2v = vec![0.0f32; 7 * DH];
            for i in 0..7 * DH { h2v[i] = hh[i] + ao[i]; }
            let mut h2n = h2v.clone();
            hnorm(&mut h2n, &l.post_ln, 7, DH);
            let gate = lin(&l.gate_proj, &h2n, 10240, DH, 7);
            let up = lin(&l.up_proj, &h2n, 10240, DH, 7);
            let mut ffn = vec![0.0f32; 7 * 10240];
            for i in 0..7 * 10240 { ffn[i] = silu(gate[i]) * up[i]; }
            let down = lin(&l.down_proj, &ffn, DH, 10240, 7);
            for i in 0..7 * DH { hh[i] = h2v[i] + down[i]; }

            // device side: partial round to layer li+1, dump h
            round.reset();
            round.draft_round_partial_dev(777, li + 1).expect("partial round");
            let hd = round.dump_h(7).expect("dump_h");
            let rl = rel_l2(&hd, &hh);
            // compounding bf16-staging gap (documented above); must stay well below ~1.0
            check(&format!("bisect after layer {} h relL2 {rl:.3e} < 3e-1 (bf16-staging documented)", li),
                  rl < 3e-1);
        }
        round.reset();
    }

    println!("== summary ==");
    if all_pass.get() {
        println!("DSPARK_ROUND_PROBE: ALL PASS");
    } else {
        println!("DSPARK_ROUND_PROBE: FAILURES PRESENT");
        std::process::exit(1);
    }
}

/// WI1: load the Qwen3.8-27B-DSpark round (the `--spec-source dspark` serving drafter) —
/// the DF2 loader's fail-soft shape: any refusal (ctx bound, trunk mismatch, missing/failed
/// artifact) logs a WARN and returns None so the lane degrades to MTP, never a hard failure.
pub(super) fn load_dspark_round_dir(gpu: &mut gb10_inference::gpu::GpuModel, max_c: usize, draft_dir: &str,
                         sha_pin: Option<&str>)
    -> Option<(gb10_inference::dspark::round::DsparkRound,
               std::sync::Arc<gb10_inference::dflash2::capture::Df2TapSink>,
               std::sync::Arc<gb10_inference::dflash2::capture::Df2PrimeSink>)> {
    use gb10_inference::dflash2::capture::{Df2PrimeSink, Df2TapSink};
    // F8 acceptance A2 (owner directive 2026-09-06): the packed-arg geometry fields are 21
    // bits wide (gpu_dspark.cu) — the round carries the full 512K context class. Beyond the
    // 2^21 field width the round refuses loudly here rather than at the first long-context
    // request (the old 16-bit fields capped at 65520 — lifted).
    if max_c + 15 >= (1 << 21) {
        eprintln!("[dspark] WARN: --max-seq-len {max_c} exceeds the DSpark round's packed-arg \
                   ctx bound ({}) — serving via MTP (auto-fallback)", (1 << 21) - 15);
        return None;
    }
    // The drafter is dimension-fixed to the 3.8-27B trunk shapes (borrowed head/embed + taps
    // at layers <= 52); an incompatible trunk refuses here (the standing fallback).
    if !gpu.df2_round_compatible() {
        let (h, v, l) = gpu.df2_trunk_dims();
        eprintln!("[dspark] WARN: trunk hidden={h} vocab={v} layers={l} is not DSpark-compatible \
                   (the drafter borrows the 3.8-27B head/embed; taps reach layer 52) — serving \
                   via MTP (auto-fallback)");
        return None;
    }
    let (head, embed) = match gpu.df2_borrow() {
        Some(p) => p,
        None => {
            eprintln!("[dspark] WARN: trunk lm_head/embed are not NVFP4/BF16 (or absent) — the \
                       borrowed-head path is unavailable; serving via MTP");
            return None;
        }
    };
    let mut round = match gb10_inference::dspark::round::DsparkRound::load_pinned(
        draft_dir, Some(head), Some(embed), max_c, sha_pin) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[dspark] WARN: DSpark round load FAILED ({e:#}) — absent/failed artifact \
                       is never a hard failure; serving via MTP (standing directive)");
            return None;
        }
    };
    let sink = std::sync::Arc::new(Df2TapSink::new(gpu.dev()));
    round.attach_sink(&sink);
    let prime = std::sync::Arc::new(Df2PrimeSink::new(
        gpu.dev(), gb10_inference::batch::PREFILL_CHUNK));
    gpu.set_dspark_capture(sink.clone());
    Some((round, sink, prime))
}
