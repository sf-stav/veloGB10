//! `impl Tuner`, phase 1: bookkeeping, the base probe, static pruning and the search plan.
//! See `tuner_defs.rs` for the type and `mod.rs` for the file map.

use super::*;

impl<'a> Tuner<'a> {
    pub(super) fn elapsed_s(&self) -> f64 { self.t0.elapsed().as_secs_f64() }

    pub(super) fn say(&mut self, line: String) {
        println!("{line}");
        let _ = writeln!(self.jsonl, "{}", serde_json::json!({"log": line, "t_s": self.elapsed_s()}));
        let _ = self.jsonl.flush();
    }

    pub(super) fn env_snapshot(&mut self, phase: &str) -> serde_json::Value {
        let e = gpu_env(phase);
        if throttled(&e) { self.throttle_events += 1; }
        let _ = writeln!(self.jsonl, "{}", serde_json::json!({"env": e.clone(), "t_s": self.elapsed_s()}));
        self.gpu_log.push(e.clone());
        e
    }

    /// §6.4.3: abort (keeping the receipts) the moment another tenant appears.
    pub(super) fn tenants_ok(&mut self) -> Result<()> {
        if let Some(a) = resident_apps() {
            if !a.is_empty() { bail!("a tenant appeared on the GPU mid-run: {a:?} — aborting (receipts kept)"); }
        }
        let o = other_engines();
        if !o.is_empty() { bail!("another gb10_inference appeared mid-run (pids {o:?}) — aborting (receipts kept)"); }
        Ok(())
    }

    // ---- the tuning trajectory (§5.2): a real greedy fixed-depth generation, SR armed per round

    fn start_prompt(&mut self) -> Result<()> {
        let idx = self.gen_next % self.corpus.len();
        self.gen_next += 1;
        let ids = self.corpus[idx].clone();
        with_session(|s| { s.prompt = idx; Ok(()) })?;
        let pred0 = pass_setup(self.model, &mut self.sc, &ids, self.max_new, self.chunk, &mut self.psc)?;
        self.model.mtp_seam_prime(&mut self.sc, self.psc.as_ref(), self.head, *ids.last().unwrap() as i32, ids.len(), 0)?;
        self.gen = Some((pred0 as i32, ids.len(), 0, 1));
        Ok(())
    }

    fn one_round(&mut self) -> Result<()> {
        let fresh = match self.gen {
            None => true,
            Some((b, p, _, e)) => EOS_IDS.contains(&b) || e >= self.max_new || p + self.k + 2 >= self.model.max_pos,
        };
        if fresh { self.start_prompt()?; }
        let (b, p, tap_row, emitted) = self.gen.unwrap();
        if EOS_IDS.contains(&b) { self.gen = None; return Ok(()); }
        let t = std::time::Instant::now();
        ARMED.store(true, Ordering::SeqCst);
        let r = self.model.mtp_round(&mut self.sc, self.head, 0, self.k, b, p, tap_row);
        ARMED.store(false, Ordering::SeqCst);
        let r = r?;
        self.round_ms.push(t.elapsed().as_secs_f64() * 1e3);
        self.gen = Some((r.last_tok, r.pos_next, r.tap_row, emitted + r.emitted.len()));
        Ok(())
    }

    fn run_rounds(&mut self, n: usize) -> Result<()> {
        for _ in 0..n { self.one_round()?; }
        let model = self.model;
        with_session(|s| s.finish(model))?;
        Ok(())
    }

    fn set_arms(&mut self, arms: Vec<Arm>, verify_on: bool, draft_on: bool, widths: Vec<usize>) -> Result<usize> {
        let (k, cap) = (self.k, self.mem_cap);
        Ok(with_session(|s| {
            s.set_arms(arms)?;
            s.cfg = SrCfg { verify_on, draft_on, li_on: false, verify_widths: widths, draft_s: k, mem_cap: cap };
            Ok(s.phase)
        })?.context("tune session missing")?)
    }

    /// The adopted decisions as table records (one per scope).
    pub(super) fn table_decisions(&self) -> Vec<tune::Decision> {
        let mut v = Vec::new();
        for d in &self.decisions {
            for sc in &d.scopes {
                v.push(tune::Decision { id: d.knob.def.id.into(), rev: d.knob.def.rev, class: d.knob.def.class.name().into(),
                                        scope: sc.clone(), value: d.value, default: Some(d.knob.def.default),
                                        evidence: d.evidence.clone() });
            }
        }
        v
    }

    /// The base assignment (every adopted decision, keyed), optionally without one family's.
    pub(super) fn base_assign(&self, skip_family: Option<&str>) -> Vec<(&'static tune::TunableDef, Option<tune::Key>, i32)> {
        let mut v = Vec::new();
        for d in &self.decisions {
            if Some(d.family.as_str()) == skip_family { continue; }
            for k in &d.keys { v.push((d.knob.def, Some(*k), d.value)); }
        }
        v
    }

    // ---- probe (§6.1 / §4.8a / tau and budget inputs)

    pub(super) fn probe(&mut self, regime: u8) -> Result<()> {
        let widths = self.widths_all.clone();
        let phase = self.set_arms(vec![Arm::base()], true, true, widths.clone())?;
        self.run_rounds(2 * widths.len() + 4)?;
        let samples: Vec<Sample> = with_session(|s| Ok(s.samples.iter().filter(|x| x.phase == phase && x.arm == 0 && x.digest_ok)
            .cloned().collect::<Vec<_>>()))?.unwrap_or_default();
        let mut groups: HashMap<(&'static str, usize), Vec<f64>> = HashMap::new();
        for s in &samples {
            let unit = if s.unit == "verify_dry" { "verify" } else if s.unit == "draft" { "draft" } else { continue };
            groups.entry((unit, s.m)).or_default().push(s.ms);
        }
        let mut line = format!("PROBE ({}): base unit medians", if regime == 1 { "sparse" } else { "dense" });
        let mut keys: Vec<_> = groups.keys().copied().collect();
        keys.sort();
        for key in keys {
            let v = &groups[&key];
            let med = tune::median(v).unwrap_or(f64::NAN);
            let mad = tune::median(&v.iter().map(|x| (x - med).abs()).collect::<Vec<_>>()).unwrap_or(f64::NAN);
            self.base_ms.insert((key.0, key.1, regime), med);
            self.base_mad.insert((key.0, key.1, regime), mad);
            line += &format!(" | {} m{} {:.3} ms (MAD {:.3}, n {})", key.0, key.1, med, mad, v.len());
        }
        let canon = tune::median(&self.round_ms[self.round_ms.len().saturating_sub(16)..]).unwrap_or(f64::NAN);
        line += &format!(" | canonical round {canon:.1} ms wall");
        self.say(line);
        let xf = with_session(|s| Ok(s.xfail.clone()))?.unwrap_or_default();
        if !xf.is_empty() {
            for l in xf.iter().take(8) { self.say(l.clone()); }
            bail!("the BASE replay differs from the canonical execution in the probe — the harness is not \
                   trustworthy on this build (fix before tuning)");
        }
        Ok(())
    }

    pub(super) fn unit_ms(&self, unit: &'static str, regime: u8) -> f64 {
        let v: Vec<f64> = self.base_ms.iter().filter(|((u, _, r), _)| *u == unit && *r == regime).map(|(_, v)| *v).collect();
        if v.is_empty() { return if unit == "verify" { 45.0 } else { 8.0 }; }
        v.iter().sum::<f64>() / v.len() as f64
    }

    pub(super) fn tau_for(&self, target: Target, regime: u8) -> f64 {
        let base = match target {
            Target::Verify => self.unit_ms("verify", regime),
            Target::Draft => self.unit_ms("draft", regime),
            Target::Round => self.unit_ms("verify", regime) + self.unit_ms("draft", regime) * DRAFT_PASSES_PER_ROUND / self.k as f64,
        };
        tune::tau_ms(base)
    }

    /// Estimated seconds for one family race (stage 1 + ~half of stage 2 + a 50% confirmation).
    pub(super) fn est_s(&self, f: &Family, t: Target) -> f64 {
        let rounds = self.rcfg.n_stage1 as f64 + (self.rcfg.n_max - self.rcfg.n_stage1) as f64 * 0.3
            + 0.5 * self.rcfg.n_confirm as f64;
        let canon = tune::median(&self.round_ms).unwrap_or(70.0);
        let unit = match t {
            Target::Verify => self.unit_ms("verify", f.regime),
            Target::Draft => self.unit_ms("draft", f.regime),
            Target::Round => self.unit_ms("verify", f.regime) + self.unit_ms("draft", f.regime),
        } + 0.3;
        // v2: + the base warm-up replay(s) every block
        rounds * (canon + (f.cands.len() + 1 + self.warmup) as f64 * unit) / 1e3
    }

    // ---- static pruning (§6.1) + the plan (§6.2)

    /// Why a registry entry is not searched (None = searchable).
    pub(super) fn prune_reason(&self, def: &tune::TunableDef) -> Option<String> {
        if let Some((_, why)) = NOT_SEARCHED.iter().find(|(id, _)| *id == def.id) { return Some(why.to_string()); }
        if def.class != tune::Class::S { return Some(format!("class {}: not searched by default (never without consent)", def.class.name())); }
        if def.xcheck.is_empty() { return Some("DOMAIN UNPROVEN (no bit-diff harness)".into()); }
        if def.scope == tune::Scope::Load { return Some("Load scope: both layouts would have to be resident".into()); }
        if def.scope == tune::Scope::Chunk { return Some("prefill unit: registered, not searched in this build".into()); }
        if tune::env_overrides().iter().any(|(id, _, _)| *id == def.id) {
            return Some(format!("OVERRIDDEN(flag {}) — the override flag wins over every arm", def.env));
        }
        None
    }

    /// Units (per the probe captures) that consulted `def`.
    fn consulted_units(&self, def: &tune::TunableDef) -> Vec<&'static str> {
        with_session(|s| Ok(s.consulted_by.get(&def.slot).map(|x| x.iter().copied().collect::<Vec<_>>()).unwrap_or_default()))
            .ok().flatten().unwrap_or_default()
    }

    fn targets_for(&self, knobs: &[Knob]) -> Vec<Target> {
        let mut v = false;
        let mut d = false;
        for k in knobs {
            let u = self.consulted_units(k.def);
            v |= u.contains(&"verify");
            d |= u.contains(&"draft");
        }
        let per_fam = knobs.iter().all(|k| matches!(k.def.scope, tune::Scope::Width | tune::Scope::Fam));
        if per_fam {
            let mut t = Vec::new();
            if v { t.push(Target::Verify); }
            if d { t.push(Target::Draft); }
            t
        } else if v || d {
            vec![Target::Round]
        } else {
            Vec::new()
        }
    }

    /// Candidate values of one knob: its domain minus the base value, valid at the representative
    /// contexts of its targets.
    fn knob_cands(&self, k: &Knob, regime: u8) -> Vec<i32> {
        let base_v = self.decisions.iter().find(|d| d.knob.def.slot == k.def.slot && d.knob.key == k.key)
            .map(|d| d.value).unwrap_or(k.def.default);
        let ctxs: Vec<Ctx> = match k.key {
            Some(tune::Key::S(kk, n, b)) => vec![Ctx::UTIL.shape(kk, n, b as i32)],
            _ => self.widths_rep.iter().map(|&m| Ctx::fam(Fam::Verify, m).regime(regime == 1))
                .chain(std::iter::once(Ctx::fam(Fam::DraftChain, 1).regime(regime == 1))).collect(),
        };
        k.def.domain.iter().copied()
            .filter(|&v| v != base_v)
            .filter(|&v| ctxs.iter().any(|c| (k.def.valid)(c, v)))
            .collect()
    }

    /// The model's split-K chain shapes (singles and same-shape pairs), as psk_log_shapes walks them.
    fn psk_shapes(&self) -> (Vec<(i32, i32, i32, u32)>, Vec<(i32, i32, i32, u32)>) {
        use std::collections::BTreeSet;
        let mut singles: BTreeSet<(i32, i32, i32, u32)> = BTreeSet::new();
        let mut pairs: BTreeSet<(i32, i32, i32, u32)> = BTreeSet::new();
        let mut visit = |attn: Option<&AttnLayer>, gdn: Option<&GdnLayer>, moe: &MoeLayer| {
            let mut one = |q: &Quad| { if q.ks > 1 { singles.insert((q.k, q.n, q.bits, q.ks)); } };
            if let Some(g) = gdn { one(&g.in_qkv); one(&g.in_z); one(&g.out_proj); }
            if let Some(a) = attn {
                one(&a.q_proj); one(&a.o_proj); one(&a.k_proj); one(&a.v_proj);
                if let Some(ix) = &a.idx { one(&ix.qk); }
            }
            one(&moe.sh_gate); one(&moe.sh_up); one(&moe.sh_down);
            let mut two = |a: &Quad, b: &Quad| {
                if a.ks > 1 && a.ks == b.ks && a.k == b.k && a.n == b.n && a.bits == b.bits {
                    pairs.insert((a.k, a.n, a.bits, a.ks));
                }
            };
            if let Some(a) = attn { two(&a.k_proj, &a.v_proj); }
            two(&moe.sh_gate, &moe.sh_up);
        };
        for l in &self.model.layers {
            match &l.mixer {
                Mixer::Gdn(g) => visit(None, Some(g), &l.moe),
                Mixer::Attn(a) => visit(Some(a), None, &l.moe),
            }
        }
        visit(Some(&self.head.attn), None, &self.head.moe);
        for q in [&self.head.fc_hidden, &self.head.fc_embed] { if q.ks > 1 { singles.insert((q.k, q.n, q.bits, q.ks)); } }
        (singles.into_iter().collect(), pairs.into_iter().collect())
    }

    /// Build the dense or sparse plan; every registry entry ends up either in a family or in the
    /// report's not-searched list (with its reason).
    pub(super) fn build_plan(&mut self, regime: u8) -> Vec<(Family, Target)> {
        let reg = |id: &str| tune::REGISTRY.iter().copied().find(|t| t.id == id);
        let mut fams: Vec<Family> = Vec::new();
        let mut paired: Vec<&str> = Vec::new();
        if regime == 0 {
            for (a, av, b, bv, ev) in PAIRS {
                let (Some(da), Some(db)) = (reg(a), reg(b)) else { continue };
                if self.prune_reason(da).is_some() || self.prune_reason(db).is_some() { continue; }
                paired.push(a);
                paired.push(b);
                let ka = Knob { def: da, key: None, label: a.to_string() };
                let kb = Knob { def: db, key: None, label: b.to_string() };
                let base = (da.default, db.default);
                let mut cands: Vec<Vec<i32>> = Vec::new();
                for &x in av.iter() {
                    for &y in bv.iter() {
                        if (x, y) != base { cands.push(vec![x, y]); }
                    }
                }
                cands.truncate(9);
                fams.push(Family { name: format!("{a} x {b}"), knobs: vec![ka, kb], cands, regime, ev_ms: *ev });
            }
        }
        let specs: &[(&str, f64)] = if regime == 0 { DENSE_SPECS } else { SPARSE_SPECS };
        for (id, ev) in specs {
            if paired.contains(id) { continue; }
            let Some(def) = reg(id) else { continue };
            if self.prune_reason(def).is_some() { continue; }
            let k = Knob { def, key: None, label: id.to_string() };
            let cands: Vec<Vec<i32>> = self.knob_cands(&k, regime).into_iter().map(|v| vec![v]).collect();
            fams.push(Family { name: id.to_string(), knobs: vec![k], cands, regime, ev_ms: *ev });
        }
        if regime == 0 {
            // T4e: psk.g per single shape, psk.g_pair per pair shape (Shape scope, keyed overlay)
            let (singles, pairs) = self.psk_shapes();
            let (s1, s2) = psk_slots(&self.model.dev);
            for (list, def, members, slots) in [(&singles, &T_PSK_G, 1u32, s1), (&pairs, &T_PSK_G2, 2u32, s2)] {
                if self.prune_reason(def).is_some() { continue; }
                for &(kk, n, bits, ks) in list.iter() {
                    let items = ((n.max(0) as u32) / 128) * ks * members;
                    let rule = items.min(slots) as i32;
                    let key = tune::Key::S(kk, n, bits.clamp(0, 255) as u8);
                    let k = Knob { def, key: Some(key), label: format!("{}[{kk}x{n} b{bits}]", def.id) };
                    let mut c: Vec<i32> = self.knob_cands(&k, regime).into_iter().filter(|&v| v > 0 && v != rule).collect();
                    c.sort_by_key(|&v| (v - rule).abs());
                    c.truncate(6);
                    fams.push(Family { name: k.label.clone(), knobs: vec![k], cands: c.into_iter().map(|v| vec![v]).collect(),
                                       regime, ev_ms: PSK_EV_PER_SHAPE });
                }
            }
            // chain.pair per pair shape
            if let Some(def) = reg("chain.pair") {
                if self.prune_reason(def).is_none() {
                    for &(kk, n, bits, _) in pairs.iter() {
                        let key = tune::Key::S(kk, n, bits.clamp(0, 255) as u8);
                        let k = Knob { def, key: Some(key), label: format!("chain.pair[{kk}x{n} b{bits}]") };
                        let c: Vec<Vec<i32>> = self.knob_cands(&k, regime).into_iter().map(|v| vec![v]).collect();
                        fams.push(Family { name: k.label.clone(), knobs: vec![k], cands: c, regime, ev_ms: 0.05 });
                    }
                }
            }
        }
        // targets (per the probe's consulted map), INERT (not consulted), no candidates
        let mut plan: Vec<(Family, Target)> = Vec::new();
        for f in fams {
            let rname = if regime == 1 { format!("{} [sparse]", f.name) } else { f.name.clone() };
            if f.cands.is_empty() {
                self.families.insert(rname, serde_json::json!({"status": "NO VALID CANDIDATE (static pruning)"}));
                continue;
            }
            let ts = self.targets_for(&f.knobs);
            if ts.is_empty() {
                self.families.insert(rname, serde_json::json!({"status": "INERT (not consulted by any unit)"}));
                continue;
            }
            for t in ts { plan.push((f.clone(), t)); }
        }
        let mut items: Vec<tune::PlanItem> = plan.iter().enumerate()
            .map(|(i, (f, t))| tune::PlanItem { name: format!("{i}"), ev_ms: f.ev_ms, est_s: self.est_s(f, *t) }).collect();
        tune::order_by_ev(&mut items);
        items.iter().map(|it| plan[it.name.parse::<usize>().unwrap()].clone()).collect()
    }

    // ---- the race sampler: fresh SR rounds with arms [base] + the active candidates

    fn obs_for(&self, samples: &[Sample], offs: &PosTable, phase: usize, target: Target, arms: usize, regime: u8)
        -> Vec<Vec<tune::Obs>> {
        pair_obs(samples, offs, phase, target, arms, self.k, regime)
    }

    pub(super) fn sample(&mut self, fam: &Family, target: Target, base: &[(&'static tune::TunableDef, Option<tune::Key>, i32)],
              active: &[usize], want: usize, confirm: bool) -> Result<Vec<tune::Sampled>> {
        let mut arms = vec![Arm { name: "base".into(), assign: base.to_vec() }];
        for &c in active {
            let mut asg = base.to_vec();
            for (i, kn) in fam.knobs.iter().enumerate() { asg.push((kn.def, kn.key, fam.cands[c][i])); }
            arms.push(Arm { name: format!("{}={:?}", fam.name, fam.cands[c]), assign: asg });
        }
        let n_arms = arms.len();
        let widths = if confirm && target != Target::Draft { self.widths_all.clone() } else { self.widths_rep.clone() };
        let phase = self.set_arms(arms, target != Target::Draft, target != Target::Verify, widths)?;
        let cap_rounds = want * 3 + 12;
        let mut rounds = 0usize;
        let (mut obs, mut outs): (Vec<Vec<tune::Obs>>, Vec<Option<String>>);
        let stratify = self.stratify;
        loop {
            self.run_rounds(4)?;
            rounds += 4;
            let (samples, xfail, inert, offs): (Vec<Sample>, Vec<String>, Vec<Option<bool>>, PosTable) = with_session(|s| {
                let mine: Vec<Sample> = s.samples.iter().filter(|x| x.phase == phase).cloned().collect();
                // v2: the position model of this phase's cells, fitted over the whole session's
                // replays in those cells (a short phase alone cannot separate arm from position)
                let offs = if stratify {
                    let cells: std::collections::HashSet<PosCell> = mine.iter().filter(|x| x.mode == "sr").map(pos_cell).collect();
                    pos_table(&s.samples, &|c| cells.contains(c))
                } else { PosTable::new() };
                Ok((mine, s.xfail.clone(), (0..n_arms).map(|a| s.arm_inert(a)).collect(), offs))
            })?.context("tune session missing")?;
            obs = self.obs_for(&samples, &offs, phase, target, n_arms, fam.regime);
            outs = vec![None; n_arms];
            for a in 1..n_arms {
                if samples.iter().any(|x| x.arm == a && !x.digest_ok) {
                    let first = xfail.iter().rev().find(|l| l.contains(&format!("'{}=", fam.name)))
                        .or_else(|| xfail.iter().rev().find(|l| l.contains("differs from base")))
                        .cloned().unwrap_or_default();
                    outs[a] = Some(format!("XCHECK FAIL (a kernel bug on live activations): {first}"));
                } else if inert[a] == Some(true) {
                    outs[a] = Some("INERT (identical graph: same kernels, geometry, arguments and edges as the base)".into());
                }
            }
            if samples.iter().any(|x| (x.arm == 0 || x.arm == WARMUP_ARM) && (x.mode == "sr" || x.mode == "warm") && !x.digest_ok) {
                bail!("the BASE replay stopped matching the canonical execution (family {}) — harness failure", fam.name);
            }
            let done = (1..n_arms).all(|a| outs[a].is_some() || obs[a].len() >= want);
            if done || rounds >= cap_rounds { break; }
        }
        Ok((1..n_arms).map(|a| match outs[a].take() {
            Some(why) => tune::Sampled::Out(why),
            None => tune::Sampled::Obs(std::mem::take(&mut obs[a])),
        }).collect())
    }
}
