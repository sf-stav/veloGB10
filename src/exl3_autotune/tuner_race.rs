//! `impl Tuner`, phase 2: sampling, the paired race per family, the sweep search and the prefill point.
//! See `tuner_defs.rs` for the type and `mod.rs` for the file map.

use super::*;

impl<'a> Tuner<'a> {
    /// One family race + its decisions. Returns (adopted, a clock / thermal event happened).
    fn run_family(&mut self, fam: &Family, target: Target, sweep: usize) -> Result<(bool, bool)> {
        let t_fam = std::time::Instant::now();
        let e0 = self.env_snapshot(&format!("family {} [{}] start", fam.name, target.name()));
        let base = self.base_assign(Some(&fam.name));
        // sweep 2: the family's own adopted value becomes a candidate again, the default the base
        let mut fam = fam.clone();
        if sweep > 1 {
            let mine: Vec<i32> = fam.knobs.iter().map(|k| self.decisions.iter()
                .find(|d| d.family == fam.name && d.knob.def.slot == k.def.slot && d.knob.key == k.key && d.target == target)
                .map(|d| d.value).unwrap_or(k.def.default)).collect();
            let dflt: Vec<i32> = fam.knobs.iter().map(|k| k.def.default).collect();
            if mine != dflt && !fam.cands.contains(&mine) { fam.cands.push(mine); }
            fam.cands.retain(|c| *c != dflt);
        }
        let tau = self.tau_for(target, fam.regime);
        let rcfg = self.rcfg;
        let mut err: Option<anyhow::Error> = None;
        let out = {
            let mut sampler = |active: &[usize], want: usize, confirm: bool| -> Vec<tune::Sampled> {
                if err.is_some() { return active.iter().map(|_| tune::Sampled::Out("ERROR".into())).collect(); }
                match self.sample(&fam, target, &base, active, want, confirm) {
                    Ok(v) => v,
                    Err(e) => {
                        err = Some(e);
                        active.iter().map(|_| tune::Sampled::Out("ERROR".into())).collect()
                    }
                }
            };
            tune::race(fam.cands.len(), &rcfg, tau, &mut sampler)
        };
        with_session(|s| { s.clear_graphs(); Ok(()) })?;
        if let Some(e) = err { return Err(e); }
        let e1 = self.env_snapshot(&format!("family {} [{}] end", fam.name, target.name()));
        let drift = match (sm_clock(&e0), sm_clock(&e1)) {
            (Some(a), Some(b)) if a > 0.0 => ((b - a) / a).abs() > 0.05,
            _ => false,
        };
        // a NEW throttle flag during the block (a box that sits power-capped throughout is steady state)
        let thr = throttled(&e1) && !throttled(&e0);
        // receipts: every candidate's fate + estimate
        let cand_json: Vec<serde_json::Value> = out.cands.iter().map(|c| serde_json::json!({
            "value": fam.cands.get(c.idx), "fate": c.fate, "n": c.obs.len(),
            "hl_delta_ms": c.est.map(|e| e.est), "ci95_ms": c.est.map(|e| [e.lo, e.hi]) })).collect();
        let _ = writeln!(self.jsonl, "{}", serde_json::json!({"family": fam.name, "target": target.name(), "sweep": sweep,
            "regime": fam.regime, "tau_ms": tau, "verdict": out.verdict, "candidates": cand_json,
            "confirm": out.confirm.map(|e| serde_json::json!({"n": e.n, "hl": e.est, "ci95": [e.lo, e.hi]})),
            "calls": out.calls, "seconds": t_fam.elapsed().as_secs_f64(), "clock_drift": drift, "throttled": thr }));
        let key = format!("{}{} [{}]", fam.name, if fam.regime == 1 { " (sparse)" } else { "" }, target.name());
        let lead = out.leader.and_then(|i| out.cands[i].est.map(|e| (fam.cands[i].clone(), e)));
        let mut line = format!("FAMILY {key} sweep {sweep}: {} | tau {tau:.4} ms | {} candidate(s) | {:.1} s",
                               out.verdict, fam.cands.len(), t_fam.elapsed().as_secs_f64());
        if let Some((v, e)) = &lead { line += &format!(" | leader {v:?} HL {:+.4} CI95 [{:+.4}, {:+.4}] n {}", e.est, e.lo, e.hi, e.n); }
        if let Some(c) = out.confirm { line += &format!(" | confirm HL {:+.4} CI95 [{:+.4}, {:+.4}] n {}", c.est, c.lo, c.hi, c.n); }
        for c in &out.cands {
            if c.fate.starts_with("XCHECK") || c.fate.starts_with("INERT") {
                line += &format!(" | {:?}: {}", fam.cands[c.idx], c.fate);
            }
        }
        if drift || thr { line += " | CLOCK/THERMAL EVENT during the family (flagged in the receipts)"; }
        self.say(line);
        // decisions
        let mut adopted = false;
        // a re-race replaces this family's previous decision for this target either way
        self.decisions.retain(|d| !(d.family == fam.name && d.target == target));
        if let Some(w) = out.winner {
            let e = out.cands[w].est.unwrap();
            let conf = out.confirm.unwrap();
            let values = fam.cands[w].clone();
            let runner = out.cands.iter().filter(|c| c.idx != w && c.est.is_some())
                .min_by(|a, b| a.est.unwrap().est.partial_cmp(&b.est.unwrap().est).unwrap())
                .map(|c| serde_json::json!({"value": fam.cands[c.idx], "hl_delta_ms": c.est.unwrap().est}));
            let split = tune::width_split(&out.confirm_obs);
            let rg = if fam.regime == 1 { "sparse" } else { "dense" };
            let base_ms = match target {
                Target::Verify => self.unit_ms("verify", fam.regime),
                Target::Draft => self.unit_ms("draft", fam.regime),
                Target::Round => self.unit_ms("verify", fam.regime) + self.unit_ms("draft", fam.regime),
            };
            for (i, kn) in fam.knobs.iter().enumerate() {
                let (scopes, keys): (Vec<tune::DecScope>, Vec<tune::Key>) = match (target, kn.def.scope) {
                    (Target::Verify, tune::Scope::Width) => {
                        let mut ms: Vec<u16> = self.widths_all.iter().map(|&m| m as u16)
                            .filter(|m| split.iter().find(|x| x.0 == *m).map_or(true, |x| x.2)).collect();
                        ms.sort_unstable();
                        let sc = tune::DecScope { fam: Some("Verify".into()), m: Some(ms.clone()), regime: Some(rg.into()), ..Default::default() };
                        let keys = ms.iter().map(|&m| tune::Key::W(Fam::Verify as u8, m, fam.regime)).collect();
                        (vec![sc], keys)
                    }
                    (Target::Verify, tune::Scope::Fam) => (vec![tune::DecScope { fam: Some("Verify".into()), ..Default::default() }],
                                                           vec![tune::Key::F(Fam::Verify as u8)]),
                    (Target::Draft, tune::Scope::Width) => (
                        [Fam::DraftChain, Fam::DraftPass].iter().map(|f| tune::DecScope { fam: Some(f.name().into()), m: Some(vec![1]),
                            regime: Some(rg.into()), ..Default::default() }).collect(),
                        vec![tune::Key::W(Fam::DraftChain as u8, 1, fam.regime), tune::Key::W(Fam::DraftPass as u8, 1, fam.regime)]),
                    (Target::Draft, tune::Scope::Fam) => (
                        [Fam::DraftChain, Fam::DraftPass].iter().map(|f| tune::DecScope { fam: Some(f.name().into()), ..Default::default() }).collect(),
                        vec![tune::Key::F(Fam::DraftChain as u8), tune::Key::F(Fam::DraftPass as u8)]),
                    (_, tune::Scope::Shape) => match kn.key {
                        Some(tune::Key::S(k, n, b)) => (vec![tune::DecScope { k: Some(k), n: Some(n), bits: Some(b), ..Default::default() }],
                                                        vec![tune::Key::S(k, n, b)]),
                        _ => continue,
                    },
                    (_, tune::Scope::Global) => (vec![tune::DecScope::default()], vec![tune::Key::G]),
                    _ => continue,
                };
                // boot drops a decision any of whose keys fails valid() (and a Width scope with no m):
                // adopt exactly what boot will load, per scope
                let v = values[i];
                let scopes: Vec<tune::DecScope> = scopes.into_iter().filter(|sc| {
                    match tune::decision_keys(kn.def, sc) {
                        Ok(ks) => !ks.is_empty() && ks.iter().all(|k| tune::key_valid(kn.def, k, v)),
                        Err(_) => false,
                    }
                }).collect();
                let keys: Vec<tune::Key> = keys.into_iter().filter(|k| tune::key_valid(kn.def, k, v)).collect();
                if scopes.is_empty() || keys.is_empty() {
                    self.say(format!("FAMILY {key}: {}={v} adopted but invalid at every key of this target (valid()) — not recorded",
                                     kn.def.id));
                    continue;
                }
                let ev = serde_json::json!({
                    "family": fam.name, "unit": target.name(), "n": e.n, "hl_delta_ms": e.est, "ci95_ms": [e.lo, e.hi],
                    "tau_ms": tau, "base_ms": base_ms, "runner_up": runner, "digest_ok": e.n + conf.n,
                    "confirm": {"n": conf.n, "hl_delta_ms": conf.est, "ci95_ms": [conf.lo, conf.hi]},
                    "widths": split.iter().map(|(m, x, keep)| serde_json::json!({"m": m, "n": x.map(|x| x.n),
                        "hl": x.map(|x| x.est), "keep": keep})).collect::<Vec<_>>(),
                    "regime": rg, "sweep": sweep, "joint": fam.knobs.len() > 1,
                });
                let gain = match target {
                    Target::Draft => conf.est * DRAFT_PASSES_PER_ROUND / self.k.max(1) as f64,
                    _ => conf.est,
                } / fam.knobs.len() as f64;
                self.decisions.push(Dec { family: fam.name.clone(), target, knob: kn.clone(), value: values[i], scopes, keys,
                                          evidence: ev, gain_ms: gain });
                adopted = true;
            }
        }
        let status = if adopted { format!("TUNED: {} (sweep {sweep})", out.verdict) } else { out.verdict.clone() };
        self.families.insert(key, serde_json::json!({"status": status, "tau_ms": tau,
            "candidates": out.cands.iter().map(|c| serde_json::json!({"value": fam.cands.get(c.idx), "fate": c.fate,
                "hl": c.est.map(|e| e.est), "ci95": c.est.map(|e| [e.lo, e.hi]), "n": c.obs.len()})).collect::<Vec<_>>(),
            "seconds": t_fam.elapsed().as_secs_f64(), "clock_or_thermal_event": drift || thr }));
        Ok((adopted, drift || thr))
    }

    /// §5.4: a family whose block saw the SM clock move > 5% or a throttle flag is re-run once.
    fn run_family_checked(&mut self, fam: &Family, target: Target, sweep: usize) -> Result<()> {
        let (_, event) = self.run_family(fam, target, sweep)?;
        if event {
            self.say(format!("FAMILY {} [{}]: clock / thermal event — discarded and re-run once (§5.4)", fam.name, target.name()));
            let (_, again) = self.run_family(fam, target, sweep)?;
            if again {
                self.say(format!("FAMILY {} [{}]: the event repeated — decision kept but flagged", fam.name, target.name()));
            }
        }
        Ok(())
    }

    /// Sweep 1 over the plan in order, then sweep 2 (+) over the adopted families until nothing
    /// changes (§6.2), all within the budget.
    pub(super) fn search(&mut self, plan: Vec<(Family, Target)>, max_sweeps: usize) -> Result<()> {
        let n = plan.len();
        let mut skipped = 0usize;
        for (f, t) in plan.iter() {
            // §6.6: a family that no longer fits is left UNTUNED (budget); a cheaper later one may still fit
            let est = self.est_s(f, *t);
            if self.elapsed_s() + est > self.budget_s {
                let key = format!("{}{} [{}]", f.name, if f.regime == 1 { " (sparse)" } else { "" }, t.name());
                self.families.insert(key, serde_json::json!({"status": "UNTUNED (budget)", "est_s": est}));
                skipped += 1;
                continue;
            }
            self.tenants_ok()?;
            self.run_family_checked(f, *t, 1)?;
        }
        if skipped > 0 {
            self.say(format!("BUDGET: {skipped} of {n} families left UNTUNED (budget {:.0} s, spent {:.0} s)",
                             self.budget_s, self.elapsed_s()));
        }
        for sweep in 2..=max_sweeps {
            let again: Vec<(Family, Target)> = plan.iter().filter(|(f, t)| self.decisions.iter()
                .any(|d| d.family == f.name && d.target == *t)).cloned().collect();
            if again.is_empty() { break; }
            let snapshot = |t: &Self| -> Vec<(String, i32)> {
                let mut v: Vec<(String, i32)> = t.decisions.iter()
                    .map(|d| (format!("{}|{:?}|{:?}|{}", d.family, d.target, d.knob.key, d.knob.def.id), d.value)).collect();
                v.sort();
                v
            };
            let before = snapshot(self);
            for (f, t) in &again {
                if self.elapsed_s() + self.est_s(f, *t) > self.budget_s {
                    self.say(format!("BUDGET: sweep {sweep} stopped before {} (budget)", f.name));
                    return Ok(());
                }
                self.tenants_ok()?;
                self.run_family_checked(f, *t, sweep)?;
            }
            let after = snapshot(self);
            if before == after {
                self.say(format!("SWEEP {sweep}: no decision changed — converged"));
                break;
            }
            self.say(format!("SWEEP {sweep}: decisions changed ({} -> {}) — another sweep", before.len(), after.len()));
        }
        Ok(())
    }

    // ---- the prefill point (§5.1 prefill(c), §5.7.4 the B3 known answer)

    pub(super) fn prefill_phase(&mut self, tiled: &[u32]) -> Result<()> {
        let def = &T_WIDE_MT;
        if let Some((_, env, v)) = tune::env_overrides().into_iter().find(|(id, _, _)| *id == def.id) {
            self.families.insert("prefill.wide_mt".into(),
                serde_json::json!({"status": format!("NOT SEARCHED: OVERRIDDEN(flag {env}={v}) — the override flag wins over every arm")}));
            return Ok(());
        }
        if !self.model.prefix_snapshots() {
            self.families.insert("prefill.wide_mt".into(), serde_json::json!({"status": "UNTUNED (no prefix snapshots: --prefix-cache off posture)"}));
            return Ok(());
        }
        let mut sizes: Vec<usize> = [26usize, 72, 160, 512, self.chunk].iter().copied().filter(|&c| c > 16 && c <= self.chunk).collect();
        sizes.sort_unstable();
        sizes.dedup();
        let p0 = 64usize;
        let mut known: Vec<String> = Vec::new();
        let mut deviate: Vec<String> = Vec::new();
        for c in sizes {
            if tiled.len() < p0 + c { continue; }
            let est = if c >= 2048 { 60.0 } else { 12.0 };
            // the prefill point has its own 2-minute reserve beyond the search budget
            if self.elapsed_s() + est > self.budget_s + 120.0 {
                self.families.insert(format!("prefill.wide_mt c={c}"), serde_json::json!({"status": "UNTUNED (budget)"}));
                continue;
            }
            self.tenants_ok()?;
            let prefix: Vec<i32> = tiled[..p0].iter().map(|&t| t as i32).collect();
            let toks: Vec<i32> = tiled[p0..p0 + c].iter().map(|&t| t as i32).collect();
            // the chunk's pre-state: a real prefix, snapshotted (restored before every replay)
            self.model.reset_slot(0)?;
            {
                let psc = self.psc.as_mut().context("prefill scratch")?;
                self.model.prefill_chunk(psc, &prefix, 0, 0, false, None)?;
                self.model.snapshot_slot(0, psc)?;
            }
            self.model.dev.synchronize()?;
            let class = tune::chunk_class(c);
            // p5e: the default (0) IS the PFX1 rule (B3's MT by chunk rows for the MoE expert GEMMs);
            // a fixed MT equal to the rule's value at this c launches exactly the base's kernels, so
            // it is not a candidate
            let rule_mt = pfx1_moe_mt(c) as i32;
            let cands: Vec<i32> = def.domain.iter().copied().filter(|&v| v != def.default && v != rule_mt).collect();
            let arms: Vec<i32> = std::iter::once(def.default).chain(cands.iter().copied()).collect();
            let mut regs: Vec<Reg> = Vec::new();
            for kc in self.model.k_cache.iter() { kv_rows(self.model, dptr(kc), 0, p0, c, &mut regs)?; }
            let mut err: Option<anyhow::Error> = None;
            let rcfg = if c >= 1024 { tune::RaceCfg { n_stage1: 6, n_max: 8, n_confirm: 6, step: 2 } }
                       else { tune::RaceCfg { n_stage1: 6, n_max: 12, n_confirm: 6, step: 3 } };
            let mut base_times: Vec<f64> = Vec::new();
            // v2: timed replays per block size for the position model; (warm-up, by-position) medians
            let mut hist: HashMap<usize, Vec<tune::PosObs>> = HashMap::new();
            let mut warm_times: Vec<f64> = Vec::new();
            let mut rep_id: u64 = 0;
            let (warmup, stratify) = (self.warmup, self.stratify);
            let mut porder = std::mem::take(&mut self.pf_order);
            let out = {
                let mut sampler = |active: &[usize], want: usize, _confirm: bool| -> Vec<tune::Sampled> {
                    let mut res: Vec<Vec<tune::Obs>> = vec![Vec::new(); active.len()];
                    let mut bad: Vec<Option<String>> = vec![None; active.len()];
                    if err.is_some() { return active.iter().map(|_| tune::Sampled::Out("ERROR".into())).collect(); }
                    // one timed replay of arm `a` (usize::MAX = the base) from the restored pre-state
                    let mut replay = |a: usize| -> Result<(f64, u64)> {
                        let v = if a == usize::MAX { arms[0] } else { arms[a + 1] };
                        {
                            let psc = self.psc.as_mut().context("prefill scratch")?;
                            self.model.restore_slot(0, psc)?;
                        }
                        self.model.dev.synchronize()?;
                        let t0 = std::time::Instant::now();
                        {
                            let _ov = if v != def.default { Some(tune::overlay(&[(def, None, v)])?) } else { None };
                            let psc = self.psc.as_mut().context("prefill scratch")?;
                            self.model.prefill_chunk(psc, &toks, p0, 0, false, None)?;
                        }
                        self.model.dev.synchronize()?;
                        let ms = t0.elapsed().as_secs_f64() * 1e3;
                        let dg = with_session(|s| s.digest_now(self.model, &regs))?.unwrap_or(0);
                        Ok((ms, dg))
                    };
                    let npos = active.len() + 1;
                    let r = (|| -> Result<()> {
                        // v2: the call starts after host-side race work (and, the first time, a family
                        // boundary): untimed base warm-up replay(s) first — every timed arm is then
                        // preceded by a replay + restore + sync, whatever its place
                        let mut warm_dg: Option<u64> = None;
                        for _ in 0..warmup {
                            let (ms, dg) = replay(usize::MAX)?;
                            warm_times.push(ms);
                            warm_dg = Some(dg);
                        }
                        let mut reps: Vec<HashMap<usize, (f64, u64, usize)>> = Vec::new();
                        for _ in 0..want {
                            // v2: the balanced order (Williams over base + active, per chunk and block
                            // size, continuing across race calls); row index 0 = the base
                            let row = porder.next(c as u64, npos);
                            let mut t: HashMap<usize, (f64, u64, usize)> = HashMap::new();
                            for (place, &ix) in row.iter().enumerate() {
                                let a = if ix == 0 { usize::MAX } else { active[ix - 1] };
                                let (ms, dg) = replay(a)?;
                                t.insert(a, (ms, dg, place));
                            }
                            rep_id += 1;
                            let h = hist.entry(npos).or_default();
                            for (&a, &(ms, _, place)) in t.iter() {
                                let v = if a == usize::MAX { arms[0] } else { arms[a + 1] };
                                h.push(tune::PosObs { round: rep_id, arm: v as u32 as u64, pos: place, ms });
                            }
                            reps.push(t);
                        }
                        let f = match hist.get(&npos) {
                            Some(h) if stratify => tune::position_offsets(h, npos),
                            _ => vec![0.0; npos],
                        };
                        for t in &reps {
                            let (tb, db, pb) = t[&usize::MAX];
                            base_times.push(tb);
                            if let Some(wd) = warm_dg {
                                anyhow::ensure!(wd == db, "prefill c={c}: the base chunk is not deterministic (warm-up KV digest \
                                                           {wd:016x} != base {db:016x}) — harness failure");
                            }
                            for (j, &a) in active.iter().enumerate() {
                                let (ta, da, pa) = t[&a];
                                if da != db {
                                    bad[j] = Some(format!("XCHECK FAIL: prefill c={c} MT={} KV digest {da:016x} != base {db:016x}", arms[a + 1]));
                                }
                                res[j].push((c.min(u16::MAX as usize) as u16, (ta - f[pa]) - (tb - f[pb])));
                            }
                        }
                        Ok(())
                    })();
                    if let Err(e) = r { err = Some(e); }
                    res.into_iter().zip(bad).map(|(o, b)| match b { Some(w) => tune::Sampled::Out(w), None => tune::Sampled::Obs(o) }).collect()
                };
                // tau from a first estimate of the base chunk time (0.1% of it, >= 0.02 ms)
                let tau = tune::tau_ms(if c >= 1024 { 2000.0 } else { 300.0 });
                tune::race(cands.len(), &rcfg, tau, &mut sampler)
            };
            self.pf_order = porder;
            if let Some(e) = err { return Err(e); }
            self.model.reset_slot(0)?;
            // v2: the prefill point's position effect (per block size), for the family record
            let pf_pos: Vec<serde_json::Value> = {
                let mut ks: Vec<usize> = hist.keys().copied().collect();
                ks.sort_unstable();
                ks.into_iter().map(|np| {
                    let h = &hist[&np];
                    let by: Vec<Option<f64>> = (0..np).map(|p| tune::median(&h.iter().filter(|o| o.pos == p)
                        .map(|o| o.ms).collect::<Vec<_>>())).collect();
                    serde_json::json!({"n_pos": np, "reps": h.len() / np.max(1), "median_ms_by_pos": by,
                                       "offsets_ms": tune::position_offsets(h, np)})
                }).collect()
            };
            let base_med = tune::median(&base_times).unwrap_or(f64::NAN);
            // the effective MT after the race: the adopted fixed MT, else the rule's value at c
            let pick = out.winner.map(|w| cands[w]).unwrap_or(rule_mt);
            let lead = out.leader.and_then(|i| out.cands[i].est.map(|e| (cands[i], e)));
            let mut line = format!("PREFILL c={c} (class {class}): base = the PFX1 rule (MT {rule_mt}) {:.2} ms | {} -> MT {pick}",
                                   base_med, out.verdict);
            if let Some((v, e)) = lead { line += &format!(" | leader MT {v} HL {:+.3} ms CI95 [{:+.3}, {:+.3}] n {}", e.est, e.lo, e.hi, e.n); }
            for cc in &out.cands { if cc.fate.starts_with("XCHECK") { line += &format!(" | MT {}: {}", cands[cc.idx], cc.fate); } }
            self.say(line);
            known.push(format!("c={c}->{pick}"));
            if pick != rule_mt { deviate.push(format!("c={c}: MT {pick} beat the rule's MT {rule_mt}")); }
            self.families.insert(format!("prefill.wide_mt c={c}"), serde_json::json!({"status": out.verdict, "base_ms": base_med,
                "pick": pick, "candidates": out.cands.iter().map(|x| serde_json::json!({"mt": cands[x.idx], "fate": x.fate,
                    "hl": x.est.map(|e| e.est), "ci95": x.est.map(|e| [e.lo, e.hi]), "n": x.obs.len()})).collect::<Vec<_>>(),
                "warmup_median_ms": tune::median(&warm_times), "position": pf_pos}));
            if let (Some(w), Some(conf)) = (out.winner, out.confirm) {
                let e = out.cands[w].est.unwrap();
                self.decisions.push(Dec {
                    family: format!("prefill.wide_mt c={c}"), target: Target::Round,
                    knob: Knob { def, key: None, label: def.id.into() }, value: cands[w],
                    scopes: vec![tune::DecScope { chunk: Some(vec![class]), regime: Some("dense".into()), ..Default::default() }],
                    keys: vec![tune::Key::C(class, 0)],
                    evidence: serde_json::json!({"unit": format!("prefill(c={c})"), "n": e.n, "hl_delta_ms": e.est,
                        "ci95_ms": [e.lo, e.hi], "base_ms": base_med, "digest_ok": e.n + conf.n,
                        "confirm": {"n": conf.n, "hl_delta_ms": conf.est, "ci95_ms": [conf.lo, conf.hi]}}),
                    gain_ms: 0.0, // TTFT, not a decode round: outside the round prediction check
                });
            }
        }
        // p5e: the served default IS B3's rule (PFX1 (a): MT 1 for c <= 112, 2 below 384, 4 from 384), so
        // the known answer is that no fixed MT beats it by tau at any chunk size
        self.say(format!("KNOWN ANSWER (B3, §5.7.4): MT by chunk {} — expected the PFX1 rule (MT 1 up to c ~100, 2 for \
                          ~100-350, 4 from 512) to stand: {}", known.join(" "),
                         if deviate.is_empty() { "HOLDS (no fixed MT beat the rule by tau)".to_string() }
                         else { format!("DEVIATES ({}) — a finding for the B3 ledger, flagged", deviate.join("; ")) }));
        self.known_answer = Some(deviate.is_empty());
        Ok(())
    }

    // ---- the final gate (§6.5): LI whole-round A/B + the automatic hash gate
}
