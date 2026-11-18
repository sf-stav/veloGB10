//! `impl Tuner`, phase 3: the DDS round, generation helpers and the final gate (LI whole-round A/B plus the automatic hash gate).
//! See `tuner_defs.rs` for the type and `mod.rs` for the file map.

use super::*;

impl<'a> Tuner<'a> {
    /// One served DDS round on slot 0 (exl3_serve::step_mtp's DDS branch: the calibrator's keep,
    /// the WP23 target, labels, decay, the lane EMAs and cost fit). `det_cost` feeds the WP23 guard
    /// a deterministic per-round cost (29 + 6 ms/draft, §6.6's fit) instead of the wall clock, so
    /// two arms' draft decisions are comparable (the hash gate); the LI A/B uses the wall clock.
    fn dds_round(&mut self, lane: &mut DdsLane, b: i32, p: usize, tap_row: usize, det_cost: bool,
                 samp: Option<(f32, f32, u64)>) -> Result<(MtpRoundOut, f64, bool)> {
        let k = self.k;
        let k_eff = k.min(self.model.max_pos.saturating_sub(p + 2));
        anyhow::ensure!(k_eff >= 1, "dds_round at the end of the window");
        let rows: Vec<Option<SampParams>> = (0..k + 1).map(|i| samp.and_then(|(t, tp, seed)|
            SampParams::sampled(t, tp, 0, 0.0, seed, lane.ctr.wrapping_add(i as u32)))).collect();
        self.model.set_sampling(&mut self.sc, &rows)?;
        let pens: Vec<Option<PenParams>> = vec![None; k + 1];
        self.model.set_penalties(&mut self.sc, &pens)?;
        let guard = crate::exl3_serve::wp23_guard_on(k);
        let target = crate::exl3_serve::wp23_target(self.draft_conf, guard, lane.rounds, lane.fit.slope(), lane.ema_tok, lane.ema_ms);
        let g0 = self.model.graph_count();
        let t0 = std::time::Instant::now();
        let (o, confs) = {
            let cal = &lane.cal;
            let mut reach = 1.0f64;
            let mut keep = |i: usize, c: f32| { reach *= cal.estimate_at(i, c); reach >= target };
            self.model.mtp_round_adaptive(&mut self.sc, self.head, 0, k_eff, b, p, tap_row, &mut keep, None)?
        };
        self.model.dev.synchronize()?; // retire the async re-prime: the round's whole cost
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let captured = self.model.graph_count() != g0;
        for (i, &c) in confs.iter().enumerate() {
            if i < o.a { lane.cal.add_label(i, c, true); } else { lane.cal.add_label(i, c, false); break; }
        }
        lane.cal.decay_step();
        lane.ctr = lane.ctr.wrapping_add((k + 1) as u32);
        lane.rounds += 1;
        lane.drafted += o.drafts.len();
        lane.accepted += o.a;
        let toks = (o.a + 1) as f64;
        let w = o.drafts.len();
        let cost = if det_cost { 29.0 + 6.0 * w as f64 } else { ms };
        if lane.rounds > 4 {
            if lane.ema_tok <= 0.0 { lane.ema_ms = cost; lane.ema_tok = toks; }
            else { lane.ema_ms = 0.9375 * lane.ema_ms + 0.0625 * cost; lane.ema_tok = 0.9375 * lane.ema_tok + 0.0625 * toks; }
            if (det_cost || !captured) && (lane.ema_ms <= 0.0 || cost <= 3.0 * lane.ema_ms) { lane.fit.add(w as f64, cost); }
        }
        Ok((o, ms, captured))
    }

    /// A fresh generation on slot 0: positional planes zeroed over the whole span it can reach
    /// (every arm and every repeat starts from the same device state), the served seam prime.
    fn setup_gen(&mut self, ids: &[u32], max_new: usize) -> Result<(i32, usize)> {
        let pred0 = pass_setup(self.model, &mut self.sc, ids, max_new, self.chunk, &mut self.psc)?;
        self.model.mtp_seam_prime(&mut self.sc, self.psc.as_ref(), self.head, *ids.last().unwrap() as i32, ids.len(), 0)?;
        Ok((pred0 as i32, ids.len()))
    }

    /// One whole generation in DDS (det_cost) or fixed-chain mode — the hash gate's unit.
    fn gen_once(&mut self, ids: &[u32], max_new: usize, dds: bool) -> Result<GenOut> {
        let (mut b, mut p) = self.setup_gen(ids, max_new)?;
        let mut out = GenOut { tokens: vec![b as u32], rounds: 0, drafted: 0, accepted: 0 };
        let mut tap_row = 0usize;
        let mut lane = DdsLane::new();
        while !EOS_IDS.contains(&b) && out.tokens.len() < max_new && p + self.k + 2 < self.model.max_pos {
            let r = if dds {
                self.dds_round(&mut lane, b, p, tap_row, true, None)?.0
            } else {
                let rows: Vec<Option<SampParams>> = vec![None; self.k + 1];
                self.model.set_sampling(&mut self.sc, &rows)?;
                self.model.mtp_round(&mut self.sc, self.head, 0, self.k, b, p, tap_row)?
            };
            out.rounds += 1;
            out.drafted += r.drafts.len();
            out.accepted += r.a;
            out.tokens.extend(r.emitted.iter().map(|&e| e as u32));
            b = r.last_tok;
            p = r.pos_next;
            tap_row = r.tap_row;
        }
        Ok(out)
    }

    /// Run `f` with arm B in force (the table overlay + B's graph set swapped into the served map)
    /// or arm A (defaults, the served map as is).
    fn with_arm<T>(&mut self, b_arm: bool, map_b: &mut HashMap<usize, crate::gpu::CudaGraph>,
                   assign: &[(&'static tune::TunableDef, Option<tune::Key>, i32)],
                   f: &mut dyn FnMut(&mut Self) -> Result<T>) -> Result<T> {
        if !b_arm || assign.is_empty() { return f(self); }
        std::mem::swap(&mut *self.model.graphs.lock().unwrap(), map_b);
        let r = {
            let _ov = tune::overlay(assign)?;
            f(self)
        };
        std::mem::swap(&mut *self.model.graphs.lock().unwrap(), map_b);
        r
    }

    pub(super) fn final_gate(&mut self, li_rounds: usize, hash_tokens: usize, prompts: &[(&'static str, Vec<u32>, bool)])
        -> Result<(bool, serde_json::Value)> {
        // exactly what boot will put in force: the table's decisions through build_decisions
        let probe = tune::Table { format: tune::TABLE_FORMAT.into(), origin: "user".into(), fingerprint: Default::default(),
                                  posture: Default::default(), provenance: serde_json::Value::Null,
                                  decisions: self.table_decisions(), families: serde_json::Value::Null,
                                  confirm: serde_json::Value::Null, table_sha256: String::new() };
        let (assign, ignored) = tune::table_assign(&probe, false);
        if !ignored.is_empty() {
            self.say(format!("FINAL GATE: WARNING {} decision(s) boot would ignore: {}", ignored.len(), ignored.join("; ")));
        }
        if assign.is_empty() {
            // L10: a clean 0-decision run IS a passed gate (the table is the built-in defaults: the
            // A/B has no B arm and the hashes are identical by construction) — it must say so, or
            // boot's auto selection refuses it as "unconfirmed". Not clean = decisions boot would
            // ignore (the table claims something it cannot put in force) or a DEVIATING known answer.
            let known = match self.known_answer { None => "not run", Some(true) => "holds", Some(false) => "deviates" };
            let clean = ignored.is_empty() && self.known_answer != Some(false);
            let reason = if clean {
                "0 decisions: the table is the built-in defaults (no B arm to A/B; hashes identical by construction)".to_string()
            } else if !ignored.is_empty() {
                format!("0 decisions in force but {} decision(s) boot would ignore — the table is not what it claims", ignored.len())
            } else {
                "0 decisions but the B3 known answer DEVIATES — the run is not clean".to_string()
            };
            self.say(format!("FINAL GATE: no decision adopted — {} ({reason}; known answer {known})",
                             if clean { "PASS" } else { "FAIL" }));
            return Ok((clean, serde_json::json!({"skipped": "no decision adopted", "pass": clean, "reason": reason,
                                                 "known_answer": known, "hashes_identical": true,
                                                 "ignored": ignored})));
        }
        // decode decisions are judged by the whole-round A/B; prefill (Chunk) decisions change no
        // decode round — they were confirmed by their own paired race and ride the hash gate
        let decode_decisions = self.decisions.iter().any(|d| !d.family.starts_with("prefill."));
        // graph sets: A = the served defaults, B = the table (both captured capture-only, slot 0)
        let t_cap = std::time::Instant::now();
        self.model.precapture_decode_graphs(&mut self.sc, 1, self.k, self.draft_conf > 0.0)?;
        let mut map_b: HashMap<usize, crate::gpu::CudaGraph> = HashMap::new();
        {
            let a = assign.clone();
            let dds = self.draft_conf > 0.0;
            self.with_arm(true, &mut map_b, &a, &mut |t: &mut Self| t.model.precapture_decode_graphs(&mut t.sc, 1, t.k, dds))?;
        }
        self.say(format!("FINAL GATE: graph sets captured (defaults + table, slot 0) in {:.1} s", t_cap.elapsed().as_secs_f64()));
        // ---- LI whole-round A/B, served posture (DDS + the WP23 guard on the wall clock)
        let mut li = serde_json::Map::new();
        let mut pass_li = true;
        let mut sum_est = 0.0f64;
        let mut measured_ansic: Option<f64> = None;
        for (name, ids, sampled) in prompts.iter().filter(|x| x.0 != "python") {
            if !decode_decisions { break; }
            self.tenants_ok()?;
            let samp = if *sampled { Some((1.0f32, 0.95f32, 0x5C1F1u64)) } else { None };
            // v2: a deterministic Williams schedule (n = 2: A B | B A | ... — every arm in every
            // in-block position and after each arm equally often) instead of random blocks
            let mut sched = LiSched::new(2, ids.len() as u64);
            let (mut t_a, mut t_b): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
            // (arm, in-block position, ms) of every timed round, for the position view
            let mut placed: Vec<(usize, u8, f64)> = Vec::new();
            let mut warm_ms: Vec<f64> = Vec::new();
            let mut gen: Option<(i32, usize, usize, usize, DdsLane)> = None;
            let mut guard_rounds = 0usize;
            let mut warm_left = 0usize;
            while (t_a.len() < li_rounds || t_b.len() < li_rounds) && guard_rounds < 6 * li_rounds + 64 {
                guard_rounds += 1;
                let fresh = match &gen { None => true, Some((b, p, _, e, _)) =>
                    EOS_IDS.contains(b) || *e >= self.max_new.max(1024) || p + self.k + 2 >= self.model.max_pos };
                if fresh {
                    let (b, p) = self.setup_gen(ids, self.max_new.max(1024))?;
                    gen = Some((b, p, 0, 1, DdsLane::new()));
                    // v2: the first round(s) after a fresh generation (prefill + seam prime) are an
                    // untimed warm-up under the defaults — a cold start no arm should pay
                    warm_left = self.warmup;
                }
                let (b, p, tap_row, emitted, mut lane) = gen.take().unwrap();
                let warm = warm_left > 0;
                let (arm, apos) = if warm { warm_left -= 1; (0usize, NO_APOS) } else { sched.next() };
                let r = self.with_arm(arm == 1, &mut map_b, &assign, &mut |t: &mut Self| t.dds_round(&mut lane, b, p, tap_row, false, samp))?;
                let (o, ms, captured) = r;
                if warm {
                    warm_ms.push(ms);
                } else if !captured {
                    if arm == 1 { t_b.push(ms) } else { t_a.push(ms) }
                    placed.push((arm, apos, ms));
                }
                gen = Some((o.last_tok, o.pos_next, o.tap_row, emitted + o.emitted.len(), lane));
            }
            let e = tune::hl_two_sample(&t_a, &t_b);
            let (ma, mb) = (tune::median(&t_a).unwrap_or(f64::NAN), tune::median(&t_b).unwrap_or(f64::NAN));
            let regress = matches!(e, Some(x) if x.lo > 0.0);
            pass_li &= !regress;
            if let Some(x) = e { sum_est += x.est; }
            if *name == "ansic" { measured_ansic = e.map(|x| x.est); }
            let pos_med = |a: usize| -> Vec<Option<f64>> { (0..2u8).map(|p| tune::median(&placed.iter()
                .filter(|x| x.0 == a && x.1 == p).map(|x| x.2).collect::<Vec<_>>())).collect() };
            let pm = |v: &[Option<f64>]| v.iter().map(|x| x.map_or("-".into(), |x| format!("{x:.2}"))).collect::<Vec<_>>().join(", ");
            let (pa, pb) = (pos_med(0), pos_med(1));
            self.say(format!("FINAL LI A/B {name}: default {ma:.2} ms/round (n {}) | table {mb:.2} ms/round (n {}) | table-default HL {} -> {} \
                              | by in-block position: default [{}] table [{}] | warm-up rounds {} (median {})",
                             t_a.len(), t_b.len(),
                             e.map_or("n/a".into(), |x| format!("{:+.3} CI95 [{:+.3}, {:+.3}]", x.est, x.lo, x.hi)),
                             if regress { "REGRESSION" } else { "ok" }, pm(&pa), pm(&pb), warm_ms.len(),
                             tune::median(&warm_ms).map_or("-".into(), |x| format!("{x:.2}"))));
            li.insert(name.to_string(), serde_json::json!({"default": ma, "table": mb, "n_default": t_a.len(), "n_table": t_b.len(),
                "hl_delta_ms": e.map(|x| x.est), "ci95_ms": e.map(|x| [x.lo, x.hi]), "regression": regress, "sampled": sampled,
                "median_ms_by_block_pos": {"default": pa, "table": pb}, "warmup_rounds": warm_ms.len(),
                "warmup_median_ms": tune::median(&warm_ms)}));
        }
        let beats = !decode_decisions || sum_est < 0.0;
        if !decode_decisions {
            self.say("FINAL LI A/B: skipped — only prefill (chunk-class) decisions were adopted; the hash gate still runs".into());
        }
        // prediction check (§6.5): measured round gain >= 0.5 x the sum of the adopted unit gains
        let predicted: f64 = self.decisions.iter().map(|d| d.gain_ms).sum();
        let pred_ok = match measured_ansic {
            Some(m) if predicted < 0.0 => m <= 0.5 * predicted,
            _ => true,
        };
        self.say(format!("PREDICTION CHECK: measured AnsiC round delta {} vs adopted unit gains {predicted:+.3} ms -> {}",
                         measured_ansic.map_or("n/a".into(), |m| format!("{m:+.3}")),
                         if pred_ok { "OK" } else { "FLAGGED for attribution (the parts do not add up)" }));
        // ---- the automatic hash gate: fixed chain AND DDS (deterministic guard cost), A vs B
        let mut hashes = serde_json::Map::new();
        let mut pass_hash = true;
        for (name, ids, sampled) in prompts.iter() {
            if *sampled { continue; }
            for dds in [false, true] {
                self.tenants_ok()?;
                let a = self.gen_once(ids, hash_tokens, dds)?;
                let bo = self.with_arm(true, &mut map_b, &assign, &mut |t: &mut Self| t.gen_once(ids, hash_tokens, dds))?;
                let same_tok = a.tokens == bo.tokens;
                let same_ctr = (a.rounds, a.drafted, a.accepted) == (bo.rounds, bo.drafted, bo.accepted);
                pass_hash &= same_tok && same_ctr;
                let mode = if dds { "dds" } else { "fixed" };
                self.say(format!("HASH GATE {name} {mode}: default {:016x} ({} tok, rounds {} drafted {} accepted {}) | table {:016x} \
                                  ({} tok, rounds {} drafted {} accepted {}) -> {}",
                                 fnv64(&a.tokens), a.tokens.len(), a.rounds, a.drafted, a.accepted,
                                 fnv64(&bo.tokens), bo.tokens.len(), bo.rounds, bo.drafted, bo.accepted,
                                 if same_tok && same_ctr { "IDENTICAL" } else if same_tok { "TOKENS equal, COUNTERS DIFFER" } else { "DIFFERENT" }));
                hashes.insert(format!("{name}.{mode}"), serde_json::json!({"default": format!("{:016x}", fnv64(&a.tokens)),
                    "table": format!("{:016x}", fnv64(&bo.tokens)), "tokens": a.tokens.len(),
                    "mtp_stats_default": [a.rounds, a.drafted, a.accepted], "mtp_stats_table": [bo.rounds, bo.drafted, bo.accepted],
                    "identical": same_tok && same_ctr}));
            }
        }
        drop(map_b); // table graphs (the served map keeps the defaults)
        let pass = pass_li && beats && pass_hash;
        let conf = serde_json::json!({
            "li_round_ms": li, "beats_defaults": beats, "sum_hl_delta_ms": sum_est, "no_regression": pass_li,
            "prediction_check": {"measured_ansic_ms": measured_ansic, "predicted_ms": predicted, "ok": pred_ok},
            "hashes": hashes, "mtp_stats_identical": pass_hash, "pass": pass,
            "note": "LI = served DDS rounds (wall-clock WP23 guard), arms alternated per round on one trajectory \
                     (v2: a Williams A B | B A schedule; the first round(s) of every generation are an untimed warm-up); \
                     rounds that captured a graph are excluded; control rounds (1/128 plain) are not run",
        });
        Ok((pass, conf))
    }
}
