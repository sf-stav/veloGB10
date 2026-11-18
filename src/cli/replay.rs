//! DFlash2 golden-case replays: `run_replay_df2_bisect` and `run_replay_df2_round`, with the golden-case
//! reader. They replay captured rounds against the engine and report relative L2 differences.

use super::*;

/// rel-L2 = ‖a−b‖₂ / ‖b‖₂ (f64 accumulation for the metric only).
pub(super) fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for i in 0..a.len() {
        let d = (a[i] - b[i]) as f64;
        num += d * d;
        den += (b[i] as f64) * (b[i] as f64);
    }
    (num / den.max(1e-300)).sqrt()
}

/// S5F3 — `--replay-df2-bisect`: the S1 layer/ring bisect. For one replayed step: rebuild the
/// round's ring from OUR captured taps (taps.bin rows [0..pos), via prime_window — the SAME
/// taps the oracle consumed), run the ENGINE's round at nlayers k=1..5 (the probe's
/// draft_round_depth), and compare each k's post-norm hidden against BOTH the engine's
/// ORIGINAL h_final (hfinal.bin, step s) and the ORACLE's per-layer hiddens (the replay
/// JSON). Discriminates: (a) the rebuild == original engine (the dump reconstruction is
/// faithful; the divergence vs the oracle is in the ENGINE's round math — S1 with the layer
/// pinned), vs (b) the rebuild == oracle (the ORIGINAL engine's ring differed from taps.bin —
/// a dump/ring-reconstruction artifact).
pub(crate) fn run_replay_df2_bisect(args: &[String]) {
    use gb10_inference::dflash2::oracle::{Dflash2Oracle, RoundCtx};
    use gb10_inference::dflash2::round::{BorrowedW, Df2Round};
    use half::bf16;
    let draft_dir = required_dir_arg(args, "--draft-dir", "the DFlash2 draft artifact");
    let taps_path = parse_arg(args, "--taps").expect("--taps <taps.bin>").to_string();
    let jsonl_path = parse_arg(args, "--jsonl").expect("--jsonl <steps.jsonl>").to_string();
    let job_tag = parse_arg(args, "--job").expect("--job <tag>").to_string();
    let step: usize = parse_arg(args, "--step").and_then(|s| s.parse().ok()).expect("--step <n>");
    let head_path = parse_arg(args, "--head").expect("--head").to_string();
    let embed_path = parse_arg(args, "--embed").expect("--embed").to_string();
    let trunk = parse_arg(args, "--model-dir").expect("--model-dir <trunk>").to_string();
    let (hidden, vocab, mask) = (gb10_inference::dflash2::hidden(), gb10_inference::dflash2::vocab(),
                                 gb10_inference::dflash2::mask_token_id() as usize);
    let block = 8usize;
    let tap_dim = gb10_inference::dflash2::tap_concat_dim();

    // oracle + real head/embed
    let art = gb10_inference::dflash2::load::load(&draft_dir, Some(gb10_inference::dflash2::REAL_SHA256))
        .expect("load draft artifact");
    let oracle = Dflash2Oracle::from_weights(gb10_inference::dflash2::oracle::Dflash2Config::default(),
                                             art.weights).expect("oracle");
    let read_bf16 = |path: &str| -> Vec<f32> {
        let bytes = std::fs::read(path).expect("read bf16 table");
        bytes.chunks_exact(2).map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect()
    };
    let head = read_bf16(&head_path);
    let embed = read_bf16(&embed_path);

    // find the step record + the job's taps rows
    #[derive(Clone)]
    struct StepRec { pos: usize, committed: u32, drafts: Vec<u32>, p_draft: Vec<f32>, nacc: usize }
    let mut cur_steps: Vec<StepRec> = Vec::new();
    let mut found_step: Option<StepRec> = None;
    for line in std::fs::read_to_string(&jsonl_path).expect("jsonl").lines() {
        let v: serde_json::Value = serde_json::from_str(line).expect("line");
        if v.get("job_end").is_some() { continue; }
        if v.get("tag").is_some() { continue; }
        if v.get("pos").is_none() { continue; }
        let s = v.get("step").unwrap().as_u64().unwrap() as usize;
        if s == step && v.get("pos").is_some() {
            found_step = Some(StepRec {
                pos: v["pos"].as_u64().unwrap() as usize,
                committed: v["committed"].as_u64().unwrap() as u32,
                drafts: v["drafts"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect(),
                p_draft: v["p_draft"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect(),
                nacc: v["nacc"].as_u64().unwrap() as usize,
            });
        }
        let _ = &mut cur_steps;
    }
    let st = found_step.expect("step not found");
    let pos = st.pos;
    println!("[bisect] job {job_tag} step {step}: pos={pos} anchor={} nacc={} drafts={:?}",
             st.committed, st.nacc, &st.drafts[..st.drafts.len().min(4)]);

    // taps rows [0..pos)
    let taps_all: Vec<f32> = bytemuck::cast_slice(&std::fs::read(&taps_path).expect("taps.bin")).to_vec();
    let taps = &taps_all[..pos * tap_dim];
    let mut taps_cm: Vec<bf16> = vec![bf16::default(); tap_dim * pos]; // col-major [25600, pos]
    for p in 0..pos {
        for k in 0..tap_dim { taps_cm[p * tap_dim + k] = bf16::from_f32(taps[p * tap_dim + k]); }
    }

    // ---- the engine's rebuilt round (ring from taps.bin via prime_window) ----
    let (mut gpu, _) = load_model_gpu(&trunk, None, 1);
    let (head_p, embed_p) = gpu.df2_borrow().expect("head/embed (bf16 or nvfp4)");
    let max_c = (pos + block + 64).max(1024);
    let mut round = Df2Round::load(&draft_dir, Some(head_p), Some(embed_p), max_c)
        .expect("round load");
    let taps_dev = round.dev.htod_sync_copy(&taps_cm).expect("taps dev");
    round.prime_window(&taps_dev, pos, 0).expect("prime");
    println!("[bisect] ring rebuilt: nprev={} (expected {pos})", round.nprev());

    // engine h_final at nlayers 1..5 (post final-norm of the k-layer hidden)
    for k in 1..=5usize {
        round.refresh_block_pos().expect("refresh");
        let out = round.draft_round_depth(st.committed, gb10_inference::dflash2::sliding_window(), false, false, true, k)
            .expect("round depth");
        // the oracle's per-layer hiddens on the same taps
        let mut tap_h = Vec::with_capacity(pos * tap_dim);
        for p in 0..pos { tap_h.extend_from_slice(&taps[p * tap_dim..(p + 1) * tap_dim]); }
        let ctx = RoundCtx { tap_hiddens: tap_h, anchor: st.committed };
        let th = oracle.tap_project(&ctx.tap_hiddens, pos);
        let kv = oracle.draft_kv_write(&th, pos, 0);
        let mut emb = vec![0.0f32; block * hidden];
        emb[..hidden].copy_from_slice(&embed[st.committed as usize * hidden..(st.committed as usize + 1) * hidden]);
        for r in 1..block {
            emb[r * hidden..(r + 1) * hidden].copy_from_slice(&embed[mask * hidden..(mask + 1) * hidden]);
        }
        let (layer_hiddens, _h) = oracle.backbone_forward(&emb, &kv, pos);
        let oracle_k = if k <= layer_hiddens.len() { &layer_hiddens[k - 1] } else { &_h };
        // note: engine h_final = norm(h_after_k); oracle layer_hiddens[k-1] is PRE-norm -> apply
        // the oracle's final norm for the comparison
        let oracle_k_norm = if k <= layer_hiddens.len() {
            // plain RMSNorm over the LAST axis (the oracle's convention) — inline here.
            let w = &oracle.weights.norm;
            let x = &layer_hiddens[k - 1];
            let eps = oracle.cfg.rms_eps;
            let mut out = vec![0.0f32; block * hidden];
            for r in 0..block {
                let xr = &x[r * hidden..(r + 1) * hidden];
                let mut sum_sq = 0.0f32;
                for &v in xr { sum_sq += v * v; }
                let inv = 1.0f32 / (sum_sq / hidden as f32 + eps).sqrt();
                for i in 0..hidden { out[r * hidden + i] = xr[i] * inv * w[i]; }
            }
            out
        } else { _h.clone() };
        let eng = &out.h_final;
        let rel = |a: &[f32], b: &[f32]| -> f64 {
            let mut num = 0.0f64; let mut den = 0.0f64;
            for i in 0..a.len().min(b.len()) { let d = a[i] as f64 - b[i] as f64; num += d * d; den += b[i] as f64 * b[i] as f64; }
            (num / den.max(1e-30)).sqrt()
        };
        let corr = |a: &[f32], b: &[f32]| -> f64 {
            let n = a.len().min(b.len());
            let mut me = 0.0f64; let mut mo = 0.0f64;
            for i in 0..n { me += a[i] as f64; mo += b[i] as f64; }
            me /= n as f64; mo /= n as f64;
            let mut c = 0.0f64; let mut ve = 0.0f64; let mut vo = 0.0f64;
            for i in 0..n { let ae = a[i] as f64 - me; let ao = b[i] as f64 - mo; c += ae * ao; ve += ae * ae; vo += ao * ao; }
            c / (ve.max(1e-30) * vo.max(1e-30)).sqrt()
        };
        println!("[bisect] k={k}: engine-rebuilt vs ORACLE(k) relL2 {:.4e} corr {:.4}",
                 rel(&eng, &oracle_k_norm), corr(&eng, &oracle_k_norm));
    }
    // the engine's ORIGINAL h_final for this step (hfinal.bin)
    let hf_path = std::path::Path::new(&jsonl_path).parent().unwrap().join("hfinal.bin");
    if hf_path.exists() {
        let hfinal: Vec<f32> = bytemuck::cast_slice(&std::fs::read(&hf_path).expect("hfinal.bin")).to_vec();
        let off = step * block * hidden;
        if off + block * hidden <= hfinal.len() {
            let orig = &hfinal[off..off + block * hidden];
            round.refresh_block_pos().expect("refresh");
            let out5 = round.draft_round_depth(st.committed, gb10_inference::dflash2::sliding_window(), false, false, true, 5)
                .expect("round 5");
            let rel = |a: &[f32], b: &[f32]| -> f64 {
                let mut num = 0.0f64; let mut den = 0.0f64;
                for i in 0..a.len().min(b.len()) { let d = a[i] as f64 - b[i] as f64; num += d * d; den += b[i] as f64 * b[i] as f64; }
                (num / den.max(1e-30)).sqrt()
            };
            println!("[bisect] engine-rebuilt(k=5) vs engine ORIGINAL h_final: relL2 {:.4e}",
                     rel(&out5.h_final, orig));
        }
    }
    let lo = pos.saturating_sub(2);
    let mut tap_h_all = Vec::with_capacity(pos * tap_dim);
    for p in 0..pos { tap_h_all.extend_from_slice(&taps[p * tap_dim..(p + 1) * tap_dim]); }
    // ---- ISOLATION: rebuild via the CHUNKED+INJECT path and compare vs the prime-only
    // rebuild — discriminates inject-MATH (broken) vs live-staging-timing (the inject saw
    // different contents than the dump recorded).
    {
        let mut round2 = Df2Round::load(&draft_dir, Some(gb10_inference::dflash2::round::BorrowedW::Bf16 { ptr: 0 }),
                                        Some(gb10_inference::dflash2::round::BorrowedW::Bf16 { ptr: 0 }), max_c)
            .expect("round2 load (isolation; ptrs unused — ring writes only)");
        // prime the prompt rows [0..plen) via the chunked path (upload_chunk + inject_dev in
        // BLOCK chunks — the S4F probe's proven path), then inject the generated rows
        // [plen..pos) from the staging (uploaded with the SAME taps.bin rows the live staging
        // held — the dump verified taps.bin == the live staging's true taps).
        let plen = st.pos.min(181);   // gsm1's prompt length (the first job's taps rows 0..181)
        let _ = plen;
        // NOTE: this path needs the REAL head/embed for the block input (the anchor's embed
        // gather) — the round2 with ptr-0 borrows would crash on embed_gather. The isolation
        // is therefore limited to the ring writes (fc/inject), which need NO embed. The full
        // block pass isolation is the S4F Part A on real taps (run separately).
        round2.reset();
        let chunk = 8usize;
        let mut staged: Vec<half::bf16> = vec![half::bf16::default(); tap_dim * chunk];
        let mut p = 0usize;
        // prompt rows [0..plen)
        while p < 181 {
            let n = chunk.min(181 - p);
            for mi in 0..n {
                for k in 0..tap_dim { staged[mi * tap_dim + k] = half::bf16::from_f32(taps[(p + mi) * tap_dim + k]); }
            }
            round2.upload_chunk(&staged, n).expect("upload");
            round2.inject_dev(n, None).expect("inject");
            p += n;
        }
        // generated rows [181..pos) via the staging + inject (the LIVE path's exact inputs)
        let mut g = 181usize;
        while g < pos {
            let n = chunk.min(pos - g);
            for mi in 0..n {
                for k in 0..tap_dim { staged[mi * tap_dim + k] = half::bf16::from_f32(taps[(g + mi) * tap_dim + k]); }
            }
            round2.upload_chunk(&staged, n).expect("upload gen");
            round2.inject_dev(n, None).expect("inject gen");
            g += n;
        }
        let (rk2, rv2) = round2.dump_ring_rows(0, lo, pos + 8).expect("ring2");
        let (rk1, _) = round.dump_ring_rows(0, lo, pos + 8).expect("ring1");
        // also compare the INJECT's fc output (th, [5120, 8], cols [0..m) = the last inject's
        // chunk) vs the ORACLE's exact tap_project on the SAME taps — isolates the m8 fc.
        let th2: Vec<f32> = round2.dump_th().unwrap_or_default();
        let n_last = pos - 181;   // the generated rows count
        let mut taps_g: Vec<f32> = Vec::with_capacity(n_last * tap_dim);
        for p_ in 181..pos { taps_g.extend_from_slice(&taps[p_ * tap_dim..(p_ + 1) * tap_dim]); }
        let th_oracle = oracle.tap_project(&taps_g, n_last);
        let rel2 = |a: &[f32], b: &[f32]| -> f64 {
            let mut num = 0.0f64; let mut den = 0.0f64;
            for i in 0..a.len().min(b.len()) { let d = a[i] as f64 - b[i] as f64; num += d * d; den += b[i] as f64 * b[i] as f64; }
            (num / den.max(1e-30)).sqrt()
        };
        println!("[bisect] ISOLATION: inject th (cols 0..{n_last}) vs oracle tap_project: relL2 {:.4e}",
                 rel2(&th2[..n_last * hidden], &th_oracle));
        let rel = |a: &[f32], b: &[f32]| -> f64 {
            let mut num = 0.0f64; let mut den = 0.0f64;
            for i in 0..a.len().min(b.len()) { let d = a[i] as f64 - b[i] as f64; num += d * d; den += b[i] as f64 * b[i] as f64; }
            (num / den.max(1e-30)).sqrt()
        };
        println!("[bisect] ISOLATION: chunked+inject ring vs prime-only ring (L0, rows {lo}..{}): k relL2 {:.4e}",
                 pos + 8, rel(&rk2, &rk1));
        // prompt-row comparison: rows [0..8) + [176..181)
        let (pk2, _) = round2.dump_ring_rows(0, 0, 8).expect("prompt ring2");
        let (pk1, _) = round.dump_ring_rows(0, 0, 8).expect("prompt ring1");
        println!("[bisect] ISOLATION: prompt rows [0..8): k relL2 {:.4e}", rel(&pk2, &pk1));
        let (mk2, _) = round2.dump_ring_rows(0, 176, 181).expect("mid ring2");
        let (mk1, _) = round.dump_ring_rows(0, 176, 181).expect("mid ring1");
        println!("[bisect] ISOLATION: prompt rows [176..181): k relL2 {:.4e}", rel(&mk2, &mk1));
        // the inject's ring k rows [181..pos) vs the ORACLE's k rows (post k_norm+rope) — the
        // m8-k_proj path vs the vendor-exact reference on the SAME taps.
        let (injk, _) = round2.dump_ring_rows(0, 181, pos).expect("inj ring");
        let th_oracle_all = oracle.tap_project(&tap_h_all, pos);
        let kv_oracle = oracle.draft_kv_write(&th_oracle_all, pos, 0);
        let k_oracle = &kv_oracle.layers[0].k;
        let nkv = gb10_inference::dflash2::num_kv_heads();
        let hd = gb10_inference::dflash2::head_dim();
        let mut kor = Vec::with_capacity((pos - 181) * nkv * hd);
        for r in 181..pos { kor.extend_from_slice(&k_oracle[r * nkv * hd..(r + 1) * nkv * hd]); }
        println!("[bisect] ISOLATION: inject ring k rows [181..{pos}) vs ORACLE k: relL2 {:.4e}",
                 rel2(&injk[..(pos - 181) * nkv * hd], &kor));
        // layout-correct oracle-k comparison: write the oracle's k rows [lo..pos+8) per layer
        // in the engine's HEAD-MAJOR layout (h, row, hd) and compare directly.
        let rows: Vec<usize> = (lo..pos).collect();   // the oracle's ctx k only covers [0..pos)
        let mut okm = Vec::new();   // [5 layers][nkv][rows][hd]
        let kv_all = oracle.draft_kv_write(&th_oracle_all, pos, 0);
        for li in 0..5 {
            let k = &kv_all.layers[li].k;
            for h in 0..nkv {
                for &r in &rows {
                    okm.extend_from_slice(&k[r * nkv * hd + h * hd..r * nkv * hd + h * hd + hd]);
                }
            }
        }
        // compare the INJECT ring (round2) vs the oracle for the same rows, per layer
        for li in 0..5 {
            let (inj, _) = round2.dump_ring_rows(li, lo, pos + 8).expect("inj ring li");
            let orac = &okm[li * nkv * rows.len() * hd..(li + 1) * nkv * rows.len() * hd];
            println!("[bisect] ISOLATION: inject ring vs ORACLE k (head-major, L{li}): relL2 {:.4e}",
                     rel2(&inj, orac));
            let (pr, _) = round.dump_ring_rows(li, lo, pos + 8).expect("prime ring li");
            println!("[bisect] ISOLATION: prime  ring vs ORACLE k (head-major, L{li}): relL2 {:.4e}",
                     rel2(&pr, orac));
        }
        // and the PRIME's ring k rows [0..8) vs the ORACLE k rows [0..8) — the tiled path's
        // fidelity (expected ~1e-2).
        let (prk, _) = round.dump_ring_rows(0, 0, 8).expect("prime ring");
        let mut kor2 = Vec::with_capacity(8 * nkv * hd);
        for r in 0..8 { kor2.extend_from_slice(&k_oracle[r * nkv * hd..(r + 1) * nkv * hd]); }
        println!("[bisect] ISOLATION: prime ring k rows [0..8) vs ORACLE k: relL2 {:.4e}",
                 rel2(&prk[..8 * nkv * hd], &kor2));
        // write the inject ring + oracle k rows for the external comparison
        let (injk0, _) = round2.dump_ring_rows(0, lo, pos).unwrap_or_default();
        let th_all = oracle.tap_project(&tap_h_all, pos);
        let kv_all2 = oracle.draft_kv_write(&th_all, pos, 0);
        let nkv2 = gb10_inference::dflash2::num_kv_heads();
        let hd2 = gb10_inference::dflash2::head_dim();
        let mut okor = Vec::with_capacity((pos - lo) * nkv2 * hd2);
        for h in 0..nkv2 {
            for r in lo..pos {
                okor.extend_from_slice(&kv_all2.layers[0].k[r * nkv2 * hd2 + h * hd2..r * nkv2 * hd2 + h * hd2 + hd2]);
            }
        }
        use std::io::Write as _;
        let out_d = parse_arg(args, "--out").map(str::to_string).unwrap_or_else(|| "/tmp/s5f3".to_string());
        let f1 = format!("{out_d}/inject_ring_lo{lo}_L0.bin");
        let mut fp1 = std::fs::File::create(&f1).expect("inj out");
        let _ = fp1.write_all(bytemuck::cast_slice(&injk0));
        let f2 = format!("{out_d}/oracle_k_lo{lo}_L0.bin");
        let mut fp2 = std::fs::File::create(&f2).expect("oracle out");
        let _ = fp2.write_all(bytemuck::cast_slice(&okor));
        println!("[bisect] wrote {f1} + {f2}");
    }
    // dump the rebuilt round's ring rows near C (the same rows the live dump captured) for
    // the direct live-vs-rebuilt comparison.
    let mut rk = Vec::new();
    let mut rv = Vec::new();
    for li in 0..5 {
        if let Ok((k, v)) = round.dump_ring_rows(li, lo, pos + 8) {
            rk.push(k); rv.push(v);
        }
    }
    let out_dir = parse_arg(args, "--out").map(str::to_string).unwrap_or_else(|| "/tmp/s5f3".to_string());
    let f = format!("{out_dir}/bisect_ringrows_{job_tag}_s{step}.bin");
    let mut fp = std::fs::File::create(&f).expect("ringrows out");
    use std::io::Write;
    for li in 0..rk.len() {
        let _ = fp.write_all(bytemuck::cast_slice(&rk[li]));
        let _ = fp.write_all(bytemuck::cast_slice(&rv[li]));
    }

    println!("RESULT: BISECT_DONE");
}

/// S5F3 — `--replay-df2-round`: the oracle replay (workdoc §2.2). Feed OUR captured taps
/// (from a --df2-step-dump run) through the S2F GOLDEN oracle (vendor-exact f32 math) with
/// the REAL trunk head + embed (the BF16 trunk's lm_head/embed_tokens — extracted by
/// tool_probe/b8/s5f3/extract_replay_tables.py), and compare per replayed step:
///   * oracle drafts (real head, greedy chain) vs the engine's captured drafts;
///   * the oracle's block h_final vs the engine's h_final (hfinal.bin — the S1 conditioning
///     surface; expect ~2e-2-class device-vs-oracle deltas, order-1 = divergence);
///   * the oracle drafts' target p (from the engine's captured top-20 table — the SGLang
///     renorm semantics) vs the engine drafts' p (p_of_draft);
///   * the oracle's chain on the ENGINE's candidate sets (isolates the chain from the head).
/// Verdict (workdoc §2.2): oracle drafts ≈ engine drafts AND both off-nucleus ⇒ S2 (taps);
/// oracle drafts high-p ⇒ S1 (engine's draft computation diverges — bisect surfaces below).
pub(crate) fn run_replay_df2_round(args: &[String]) {
    use gb10_inference::dflash2::oracle::{Dflash2Oracle, RoundCtx};
    let draft_dir = required_dir_arg(args, "--draft-dir", "the DFlash2 draft artifact");
    let taps_path = parse_arg(args, "--taps").expect("--replay-df2-round needs --taps <taps.bin>").to_string();
    let jsonl_path = parse_arg(args, "--jsonl").expect("--replay-df2-round needs --jsonl <steps.jsonl>").to_string();
    let job_tag = parse_arg(args, "--job").expect("--replay-df2-round needs --job <tag>").to_string();
    let head_path = parse_arg(args, "--head").expect("--replay-df2-round needs --head <head.bf16>").to_string();
    let embed_path = parse_arg(args, "--embed").expect("--replay-df2-round needs --embed <embed.bf16>").to_string();
    let out_dir = parse_arg(args, "--out").map(str::to_string).unwrap_or_else(|| "/tmp/s5f3/replay".to_string());
    let max_steps: usize = parse_arg(args, "--steps").and_then(|s| s.parse().ok()).unwrap_or(16);
    let (hidden, vocab, mask) = (gb10_inference::dflash2::hidden(), gb10_inference::dflash2::vocab(),
                                 gb10_inference::dflash2::mask_token_id() as usize);
    let block = 8usize;
    let tap_dim = gb10_inference::dflash2::tap_concat_dim();

    // ---- load the oracle ----
    let art = gb10_inference::dflash2::load::load(&draft_dir, Some(gb10_inference::dflash2::REAL_SHA256))
        .expect("load draft artifact");
    let oracle = Dflash2Oracle::from_weights(gb10_inference::dflash2::oracle::Dflash2Config::default(),
                                             art.weights).expect("oracle");

    // ---- load the real head/embed (u16 bf16 bits -> f32) ----
    let read_bf16 = |path: &str| -> Vec<f32> {
        let bytes = std::fs::read(path).expect("read bf16 table");
        assert_eq!(bytes.len() % 2, 0);
        bytes.chunks_exact(2).map(|c| {
            let u = u16::from_le_bytes([c[0], c[1]]);
            half::bf16::from_bits(u).to_f32()
        }).collect()
    };
    let head = read_bf16(&head_path);
    assert_eq!(head.len(), vocab * hidden, "head table size");
    let embed = read_bf16(&embed_path);
    assert_eq!(embed.len(), vocab * hidden, "embed table size");
    println!("[replay] oracle + real head/embed loaded ({:.2} GB head, {:.2} GB embed)",
             head.len() as f64 * 4.0 / 1e9, embed.len() as f64 * 4.0 / 1e9);

    // ---- parse the JSONL: find the job + its steps + the taps.bin row layout ----
    let jl = std::fs::read_to_string(&jsonl_path).expect("read jsonl");
    #[derive(Clone)]
    struct StepRec { pos: usize, committed: u32, drafts: Vec<u32>, p_draft: Vec<f32>,
                     candidates: Vec<u32>, unary: Vec<f32>, scores: Vec<f32>, top20: Vec<u64>,
                     nacc: usize, greedy: bool, realq: bool, step: u64 }
    let mut cur_job: Option<String> = None;
    let mut cur_steps: Vec<StepRec> = Vec::new();
    let mut jobs: Vec<(String, Vec<StepRec>)> = Vec::new();
    for line in jl.lines() {
        let v: serde_json::Value = serde_json::from_str(line).expect("jsonl line");
        if let Some(tag) = v.get("tag").and_then(|t| t.as_str()) {
            if v.get("job_end").is_some() {
                if let Some(t) = &cur_job { jobs.push((t.clone(), std::mem::take(&mut cur_steps))); }
                cur_job = None;
            } else {
                if let Some(t) = &cur_job { jobs.push((t.clone(), std::mem::take(&mut cur_steps))); }
                cur_job = Some(tag.to_string());
            }
            continue;
        }
        if cur_job.is_none() { continue; }
        // a step record
        let g = |k: &str| v.get(k);
        let gv = |k: &str| -> Vec<u64> { g(k).map(|x| x.as_array().unwrap().iter()
            .map(|e| e.as_u64().unwrap()).collect()).unwrap_or_default() };
        let gvf = |k: &str| -> Vec<f32> { g(k).map(|x| x.as_array().unwrap().iter()
            .map(|e| e.as_f64().unwrap() as f32).collect()).unwrap_or_default() };
        let gvu = |k: &str| -> Vec<u32> { g(k).map(|x| x.as_array().unwrap().iter()
            .map(|e| e.as_u64().unwrap() as u32).collect()).unwrap_or_default() };
        if g("pos").is_none() { continue; }
        cur_steps.push(StepRec {
            pos: g("pos").unwrap().as_u64().unwrap() as usize,
            committed: g("committed").unwrap().as_u64().unwrap() as u32,
            drafts: gvu("drafts"), p_draft: gvf("p_draft"),
            candidates: gvu("candidates"), unary: gvf("unary"), scores: gvf("scores"),
            top20: gv("top20"),
            nacc: g("nacc").unwrap().as_u64().unwrap() as usize,
            greedy: g("greedy").map(|x| x.as_bool().unwrap()).unwrap_or(false),
            realq: g("realq").map(|x| x.as_bool().unwrap()).unwrap_or(false),
            step: g("step").unwrap().as_u64().unwrap(),
        });
    }
    if let Some(t) = &cur_job { jobs.push((t.clone(), std::mem::take(&mut cur_steps))); }
    let (_, steps) = jobs.iter().find(|(t, _)| t == &job_tag)
        .unwrap_or_else(|| panic!("job {job_tag} not found (have: {:?})", jobs.iter().map(|(t,_)| t.clone()).collect::<Vec<_>>()));
    println!("[replay] job {job_tag}: {} steps captured", steps.len());

    // ---- load the job's taps rows ----
    let taps_all: Vec<f32> = bytemuck::cast_slice(&std::fs::read(&taps_path).expect("read taps.bin")).to_vec();
    assert_eq!(taps_all.len() % tap_dim, 0, "taps.bin not [N, 25600]");
    // the job's rows are [0..max_pos) of its accumulation — find the max pos of its steps
    let max_pos = steps.iter().map(|s| s.pos).max().unwrap_or(0) + block;
    assert!(max_pos <= taps_all.len() / tap_dim, "taps.bin rows {} < job need {max_pos}", taps_all.len() / tap_dim);
    let taps = &taps_all[..max_pos * tap_dim];
    let tap_at = |p: usize| -> &[f32] { &taps[p * tap_dim..(p + 1) * tap_dim] };

    // ---- the engine's h_final (hfinal.bin: [8, 5120] f32 per step, steps < RAW_STEPS) ----
    let hf_path = std::path::Path::new(&jsonl_path).parent().unwrap().join("hfinal.bin");
    let hfinal: Vec<f32> = if hf_path.exists() {
        bytemuck::cast_slice(&std::fs::read(&hf_path).expect("read hfinal.bin")).to_vec()
    } else { Vec::new() };

    // ---- replay each step ----
    let mut out = String::from("{\"job\":\"");
    out.push_str(&job_tag);
    out.push_str("\",\"steps\":[");
    let n_replay = steps.len().min(max_steps);
    let mut t_all = std::time::Instant::now();
    for (si, st) in steps.iter().take(n_replay).enumerate() {
        let t0 = std::time::Instant::now();
        let pos = st.pos;
        // tap_hiddens = taps[0..pos) (the ring's committed rows, positions 0..pos-1), anchor
        let mut tap_h = Vec::with_capacity(pos * tap_dim);
        for p in 0..pos { tap_h.extend_from_slice(tap_at(p)); }
        let ctx = RoundCtx { tap_hiddens: tap_h, anchor: st.committed };
        // oracle round: tap_project -> kv -> real-embed block -> backbone -> REAL-head logits
        let th = oracle.tap_project(&ctx.tap_hiddens, pos);
        let kv = oracle.draft_kv_write(&th, pos, 0);
        let mut emb = vec![0.0f32; block * hidden];
        emb[..hidden].copy_from_slice(&embed[st.committed as usize * hidden..(st.committed as usize + 1) * hidden]);
        for r in 1..block {
            emb[r * hidden..(r + 1) * hidden].copy_from_slice(&embed[mask * hidden..(mask + 1) * hidden]);
        }
        let (_lh, h) = oracle.backbone_forward(&emb, &kv, pos);
        let h_sel = &h[hidden..block * hidden];
        let logits = oracle.linear(&head, h_sel, vocab, hidden, block - 1);
        let sel = oracle.select_path(h_sel, &logits, st.committed);

        // --- comparisons ---
        // 1. oracle drafts vs engine drafts
        let mut tok_match = 0;
        for j in 0..7 { if sel.tokens[j] == st.drafts.get(j).copied().unwrap_or(u32::MAX) { tok_match += 1; } }
        // 2. oracle h_final vs engine h_final (hfinal.bin)
        let mut hf_rel = f64::NAN; let mut hf_corr = f64::NAN;
        let eng_hf = if st.step < 64 && (st.step as usize) * block * hidden + block * hidden <= hfinal.len() {
            Some(&hfinal[(st.step as usize) * block * hidden..(st.step as usize + 1) * block * hidden])
        } else { None };
        if let Some(ehf) = eng_hf {
            let mut num = 0.0f64; let mut den = 0.0f64;
            let mut me = 0.0f64; let mut mo = 0.0f64; let mut m2e = 0.0f64; let mut m2o = 0.0f64; let mut mco = 0.0f64;
            let n = block * hidden;
            for i in 0..n {
                let a = h[i] as f64; let b = ehf[i] as f64;
                let d = a - b; num += d * d; den += b * b;
                me += a; mo += b; m2e += a * a; m2o += b * b; mco += a * b;
            }
            hf_rel = (num / den.max(1e-30)).sqrt();
            let cov = mco / n as f64 - (me / n as f64) * (mo / n as f64);
            let ve = m2e / n as f64 - (me / n as f64) * (me / n as f64);
            let vo = m2o / n as f64 - (mo / n as f64) * (mo / n as f64);
            hf_corr = cov / (ve.max(1e-30) * vo.max(1e-30)).sqrt();
        }
        // 3. p of the oracle drafts vs the engine drafts (the captured top-20 table, column j
        //    = the distribution at position pos+1+j)
        let table_p = |tok: u32, col: usize, top20: &[u64]| -> f32 {
            if top20.len() < 8 * 20 { return f32::NAN; }
            for s in 0..20 {
                let w = top20[col * 20 + s];
                if (w & 0xFFFFFFFF) as u32 == tok {
                    return f32::from_bits(((w >> 32) & 0xFFFFFFFF) as u32);
                }
            }
            0.0  // outside the top-20 (=> outside the nucleus)
        };
        // DF2 block-16: the live level count, never a constant 7.
        let nlv = block - 1;
        let mut p_oracle = vec![f32::NAN; nlv];
        for j in 0..nlv { p_oracle[j] = table_p(sel.tokens[j], j, &st.top20); }
        let p_engine = &st.p_draft;
        let p_oracle_sum: f32 = p_oracle.iter().sum();
        let p_engine_sum: f32 = p_engine.iter().sum();
        // 4. oracle chain on the ENGINE's candidate sets (isolate the chain)
        let mut chain_on_engine =
            [0u32; gb10_inference::dflash2::oracle::MAX_SELECT_LEVELS];
        if st.candidates.len() == nlv * 16 && st.unary.len() == nlv * 16 {
            let mut cands = [[0u32; 16]; gb10_inference::dflash2::oracle::MAX_SELECT_LEVELS];
            let mut unars = [[0.0f32; 16]; gb10_inference::dflash2::oracle::MAX_SELECT_LEVELS];
            for p_ in 0..nlv { for k in 0..16 {
                cands[p_][k] = st.candidates[p_ * 16 + k];
                unars[p_][k] = st.unary[p_ * 16 + k];
            }}
            let co = oracle.select_chain(h_sel, &cands, &unars, st.committed);
            chain_on_engine = co.tokens;
        }
        let mut ce_match = 0;
        for j in 0..nlv { if chain_on_engine[j] == st.drafts.get(j).copied().unwrap_or(u32::MAX) { ce_match += 1; } }

        let dt = t0.elapsed().as_secs_f64();
        out.push_str(&format!(
            "{{\"step\":{},\"pos\":{},\"nacc\":{},\"greedy\":{},\"realq\":{},\
             \"oracle_drafts\":{:?},\"engine_drafts\":{:?},\"tok_match\":{},\
             \"chain_on_engine\":{:?},\"chain_match\":{},\
             \"h_final_relL2\":{:.4e},\"h_final_corr\":{:.4},\
             \"p_oracle\":{:?},\"p_engine\":{:?},\"p_oracle_sum\":{:.4},\"p_engine_sum\":{:.4},\
             \"wall_s\":{:.2}}}",
            st.step, pos, st.nacc, st.greedy, st.realq,
            sel.tokens.iter().map(|t| *t as u64).collect::<Vec<_>>(),
            st.drafts.iter().map(|t| *t as u64).collect::<Vec<_>>(),
            tok_match,
            chain_on_engine.iter().map(|t| *t as u64).collect::<Vec<_>>(),
            ce_match,
            hf_rel, hf_corr,
            p_oracle.iter().map(|p| *p as f64).collect::<Vec<_>>(),
            p_engine.iter().map(|p| *p as f64).collect::<Vec<_>>(),
            p_oracle_sum, p_engine_sum, dt));
        if si + 1 < n_replay { out.push(','); }
        println!("[replay] step {} pos {}: oracle-match {}/7 | chain-match {}/7 | hf relL2 {:.3e} corr {:.3} | p_oracle_sum {:.3} vs p_engine_sum {:.3} ({:.1}s)",
                 st.step, pos, tok_match, ce_match, hf_rel, hf_corr, p_oracle_sum, p_engine_sum, dt);
    }
    out.push_str("]}");
    std::fs::create_dir_all(&out_dir).expect("out dir");
    let out_path = format!("{out_dir}/replay_{job_tag}.json");
    std::fs::write(&out_path, &out).expect("write replay out");
    println!("[replay] wrote {out_path} ({n_replay} steps, {:.0}s total)", t_all.elapsed().as_secs_f64());
    println!("RESULT: REPLAY_DONE");
}
