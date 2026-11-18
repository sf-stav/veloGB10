//! The DSV4 TP server loop (`dsv4_tp_serve_server`) with the DSpark depth policy and per-server state it drives.
//! Child module `dspark_driver` (the DSpark serve driver and node loop) shares those private types.
//! Pinned by source-text tests in `dsv4_sse.rs`: the streaming branch writes no chunk before
//! `decode_request`, and the greedy path starts its request clock before the prefill; the server stays
//! followed by a private fn so those tests find its end.

mod dspark_driver;
pub(super) use self::dspark_driver::*;
use super::*;

/// Item 3.3 — adaptive draft depth for the DSpark drafter (the qwen `MtpPolicy` template).
///
/// A depth-D DSpark step drafts D rows (ONE parallel-row network pass at width D) and verifies
/// D+1 trunk rows, emitting `1 + accepted` tokens. The step pays iff `yield(D) > r(D)` with
/// `r(D) = (carry + draft(D) + verify(D+1)) / carry` — MEASURED at serve start (the
/// `--dspark-phase-ms` protocol) — and `yield(D)` from the live per-position hazards.
/// The policy re-picks the depth every 128 steps and drops to the minimum (D=1) when no depth
/// beats a plain decode. LOSSLESS is structural: the committed/corrected tokens are always the
/// trunk's argmaxes, so a depth change never touches acceptance semantics.
struct DsparkDepthPolicy {
    pin: Option<usize>,
    depth: usize,
    /// (drafted rows D, r(D)) measured by `dspark_calibrate_rd`.
    r: Vec<(usize, f32)>,
    // Rolling evaluation window.
    win_steps: u64,
    win_emitted: u64,
    win_drafts: u64,
    win_accepted: u64,
    /// Per-position conditional hazards: `hz[i]` = P(draft i+1 accepted | drafts 1..i accepted),
    /// as (accepted, offered) counts — the whole basis of the depth decision.
    hz: [(f64, f64); 8],
    /// (decode step, chosen depth, score) per completed window — printed for the A/B record.
    windows: Vec<(u64, usize, f32)>,
}

/// DSpark depth-policy evaluation window (steps per re-pick).
const DSPARK_EVAL_WINDOW: u64 = 128;

/// A challenger depth must beat the incumbent by this factor to be worth switching to.
const DSPARK_DEPTH_MARGIN: f32 = 1.05;

impl DsparkDepthPolicy {
    fn new(block: usize, pin: Option<usize>, r: Vec<(usize, f32)>) -> Self {
        // Open in the MIDDLE of the range (like qwen): the first window needs real hazards at
        // several positions to reason from, not an extrapolation from position 1 alone.
        let depth = pin.unwrap_or(4).clamp(1, block);
        Self { pin, depth, r, win_steps: 0, win_emitted: 0, win_drafts: 0, win_accepted: 0,
               hz: [(0.0, 0.0); 8], windows: Vec::new() }
    }
    fn depth(&self) -> usize { self.depth }
    fn set_r(&mut self, r: Vec<(usize, f32)>) { self.r = r; }
    fn r_at(&self, d: usize) -> f32 {
        self.r.iter().find(|&&(x, _)| x == d).map(|&(_, r)| r).unwrap_or(f32::INFINITY)
    }
    /// Cumulative per-position conditional acceptance counts (accepted, offered).
    fn hazard_counts(&self) -> Vec<(u64, u64)> {
        let mut v: Vec<(u64, u64)> = self.hz.iter().map(|&(a, n)| (a as u64, n as u64)).collect();
        while v.last().map_or(false, |&(_, n)| n == 0) { v.pop(); }
        v
    }

    /// Cumulative yield curve `P(k >= j+1)` from the conditional hazards (the docs' reference
    /// format: 95/86/77/73/73 on the old model). `cum[j] = Π_{i≤j} p_i`.
    fn cumulative_yield(&self) -> Vec<f64> {
        let mut out = Vec::new();
        let mut acc = 1.0f64;
        for &(a, n) in self.hz.iter() {
            if n < 8.0 { break; }
            acc *= a / n;
            out.push(acc);
        }
        out
    }

    /// Expected tokens emitted by a depth-D step: `1 + Σ_{j=1..D} Π_{i≤j} p_i` (the bonus token
    /// + each accepted draft). Hazards decay with depth (each draft is conditioned on its own
    /// guesses); unobserved positions carry the last observed hazard, never the first.
    fn yield_at(&self, d: usize) -> f32 {
        const MIN_OBS: f64 = 8.0;
        let mut last = 0.5f64;
        let mut acc = 1.0f64;
        let mut chain = 1.0f64;
        for j in 1..=d {
            let p = match self.hz.get(j - 1) {
                Some(&(a, n)) if n >= MIN_OBS => { last = a / n; last }
                _ => last,
            };
            chain *= p;
            acc += chain;
        }
        acc as f32
    }

    /// Record one completed step. `accepted` is the accepted prefix length (drafts 1..=accepted
    /// taken, draft accepted+1 offered-and-rejected when present) — exactly the per-position
    /// hazard information.
    fn record_step(&mut self, drafts: usize, accepted: usize, emitted: u64) {
        self.win_steps += 1;
        self.win_drafts += drafts as u64;
        self.win_accepted += accepted as u64;
        self.win_emitted += emitted;
        for i in 0..accepted.min(self.hz.len()) {
            self.hz[i].0 += 1.0;
            self.hz[i].1 += 1.0;
        }
        if accepted < drafts {
            if let Some(e) = self.hz.get_mut(accepted) { e.1 += 1.0; }
        }
    }

    /// Re-evaluate the depth at the window boundary. All inputs (hazards, r table) are identical
    /// on both ranks (TP item E pattern), so the decision — and every log line — is byte-identical
    /// across the pair; the node must NOT re-derive anything from its own clocks.
    fn tick(&mut self, step: u64) {
        if self.pin.is_some() { return; }
        if self.win_steps < DSPARK_EVAL_WINDOW { return; }
        let observed = self.win_emitted as f32 / self.win_steps as f32;
        let acc = self.win_accepted as f32 / self.win_drafts.max(1) as f32;
        self.win_steps = 0; self.win_emitted = 0; self.win_drafts = 0; self.win_accepted = 0;

        let cur = self.yield_at(self.depth) / self.r_at(self.depth);
        let (mut best_d, mut best) = (self.depth, cur);
        for &(d, r) in &self.r {
            if r <= 0.0 || !r.is_finite() || d == self.depth { continue; }
            let s = self.yield_at(d) / r;
            if s > best * DSPARK_DEPTH_MARGIN { best = s; best_d = d; }
        }
        eprintln!("[dspark-depth] window: d={} yield {:.2} acc {:.1}% | cur {:.2}x best d={} {:.2}x",
                  self.depth, observed, acc * 100.0, cur, best_d, best);
        self.windows.push((step, self.depth, best));

        if best < 1.0 {
            // No depth beats a plain decode: drop to the minimum draft (D=1) and keep
            // re-evaluating (the workload may change). Full speculation-off would be a greedy
            // serve — a structural change out of scope; the floor keeps the verify honest.
            eprintln!("[dspark-depth] DISABLED: no depth beats plain decode (best {:.2}x) — drafting 1 row", best);
            self.depth = 1;
            return;
        }
        if best_d != self.depth {
            let hzs: Vec<String> = (0..self.depth.max(best_d))
                .map(|i| match self.hz.get(i) {
                    Some(&(a, n)) if n >= 8.0 => format!("{:.2}", a / n),
                    _ => "?".to_string(),
                }).collect();
            eprintln!("[dspark-depth] depth {} -> {} ({:.2} -> {:.2} tok/step, r {:.2} -> {:.2}, {:.2}x -> {:.2}x) hazards [{}]",
                      self.depth, best_d, self.yield_at(self.depth), self.yield_at(best_d),
                      self.r_at(self.depth), self.r_at(best_d), cur, best, hzs.join(" "));
            self.depth = best_d;
        }
    }
}

/// Item 3.4 — DSpark decode machinery for the persistent DSV4 server (the `--server-dspark`
/// path). The HEAD and the NODE each own one of these and run the IDENTICAL SPMD sequence per
/// request: prefix-cached prefill with main_hidden capture → draft-ring warm → the same
/// draft/verify/rollback step loop as the one-shot `dsv4_tp_dspark_serve`. Owns the trunk
/// model, the drafter, the depth policy, and the once-per-process r(D) calibration cache.
struct DsparkServeState {
    m: gb10_inference::dsv4_model::Dsv4GpuModel,
    ds: gb10_inference::dsv4_dspark::Dsv4DSpark,
    dev: std::sync::Arc<cudarc::driver::CudaDevice>,
    policy: DsparkDepthPolicy,
    eos: u32,
    block: usize,
    vocab: usize,
    phase_sync: bool,
    /// r(D) calibration (item 3.4a + T4/E17 per-ctx-bucket re-calibration): the first request
    /// calibrates at its (short) context — the once-per-process cache; then, when the carried
    /// context grows past a 16K bucket (a long single request or a growing conversation), the
    /// table is re-measured AT THE REAL CONTEXT — the ≥16K KV-bound regime is where adaptive
    /// truncation pays (session-11's flat short-ctx table is 1.14–1.35×; at ≥16K the deeper
    /// verify rows cost real DRAM, the r(D) ratios grow and the policy can truncate). The
    /// machinery is the SPMD-safe, head-shipped one running at the NEXT_WATERMARK position (ctx
    /// position to trigger the next re-calibration). Never per-request (item 3.4a caching).
    next_calib_at: usize,
    // Process-lifetime stats (printed per request as deltas; the policy stays process-level so
    // the 128-step re-pick window amortizes across requests, not per short turn).
    t_carryfwd: f64,
    t_draft: f64,
    t_verify: f64,
    t_reforward: f64,
    n_steps: u64,
    n_offered: u64,
    n_accepted: u64,
    n_fast: u64,
    reforward_toks: u64,
}

/// T4/E17: re-calibrate r(D) at each 16K-context bucket (the KV-bound regime where the
/// short-ctx flat table under-prices the deeper verify rows).
const DSPARK_CTX_BUCKET: usize = 16384;

impl DsparkServeState {
    /// Load the trunk (already loaded by the caller) + the DSpark stages (replicated per-rank
    /// `rank{rank}/dspark_stage*.safetensors`). `pin_eff` derives from the SHIPPED TpConfig
    /// (`--dspark-depth`), identical on both ranks by construction — the server path does NOT
    /// arm the graphed drafter (the one-shot bench path keeps that knob), so there is no
    /// head-local env divergence to desync the depth decision.
    fn new(
        dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
        model_dir: &std::path::Path,
        cfg: &gb10_inference::dsv4_load::Dsv4Config,
        max_seq_len: usize,
        m: gb10_inference::dsv4_model::Dsv4GpuModel,
        rank: usize,
        is_head: bool,
    ) -> anyhow::Result<Self> {
        use gb10_inference::dsv4_dspark::Dsv4DSpark;
        let role = if is_head { "HEAD" } else { "NODE" };
        let rank_dir = model_dir.join(format!("rank{rank}"));
        let dspark_path = rank_dir.join("dspark_stage0.safetensors");
        anyhow::ensure!(dspark_path.exists(),
            "[server-dspark] {role}: dspark_stage0.safetensors missing at {} — the DSpark server \
             needs the converted sharded dir (rank0/rank1 with dspark_stage*.safetensors)",
            rank_dir.display());
        let t0 = std::time::Instant::now();
        let embed = m.embed.clone();
        let head = m.head.clone();
        let ds = Dsv4DSpark::load_from_artifact(dev, &rank_dir, cfg, max_seq_len, embed, head)?;
        println!("{role} — DSpark stages loaded (3 stages, replicated) in {:.1}s", t0.elapsed().as_secs_f64());
        let depth_pin = gb10_inference::tp::tp_config()
            .map(|c| c.dspark_depth.map(|d| d as usize)).unwrap_or(None);
        let block = cfg.dspark_block_size;
        let policy = DsparkDepthPolicy::new(block, depth_pin, Vec::new());
        match depth_pin {
            Some(p) => println!("{role} — DSpark depth: PINNED at {p} (fixed-width draft — the adaptive-depth disable path)"),
            None => println!("{role} — DSpark depth: ADAPTIVE (r(D) calibrated once at the first request, re-picked every {} steps; pin --dspark-depth N to disable)",
                             DSPARK_EVAL_WINDOW),
        }
        Ok(Self {
            m, ds,
            dev: dev.clone(),
            policy,
            eos: 1u32,
            block,
            vocab: cfg.vocab_size,
            phase_sync: gb10_inference::opts::var(gb10_inference::opt!("dspark-phase-ms")).is_ok(),
            next_calib_at: 0,
            t_carryfwd: 0.0, t_draft: 0.0, t_verify: 0.0, t_reforward: 0.0,
            n_steps: 0, n_offered: 0, n_accepted: 0, n_fast: 0, reforward_toks: 0,
        })
    }

    /// The shared per-request SPMD DSpark decode (head + node call the same sequence; the r(D)
    /// calibration ships head→node over the retained control stream on the first request).
    /// Returns the generated token ids (the head frames the OpenAI response; the node discards
    /// — both computed the identical sequence). `emit` (head-only, SSE streaming) is invoked
    /// per generated token AS the loop produces it (item 3.4 — one delta per token).
    #[allow(clippy::too_many_arguments)]
    fn decode_request(
        &mut self,
        prompt_i32: &[i32],
        priming_len: usize,
        max_new: usize,
        cache: &mut gb10_inference::dsv4_model::PrefixCache,
        prefix_cache_on: bool,
        ctl: Option<&mut std::net::TcpStream>,
        is_head: bool,
        mut emit: Option<&mut dyn FnMut(u32) -> anyhow::Result<()>>,
    ) -> anyhow::Result<Vec<u32>> {
        let role = if is_head { "HEAD" } else { "NODE" };
        let dev = &self.dev;
        let m = &mut self.m;
        let ds = &mut self.ds;
        let mut ctl = ctl;
        let prompt_len = prompt_i32.len();
        let eos = self.eos;
        let block = self.block;
        let vocab = self.vocab;
        let phase_sync = self.phase_sync;

        // Prefill (prefix-cached with main_hidden capture — the DSpark draft ring warm needs
        // the per-position hidden means; a cache HIT restores the prefix's stored rows).
        let (xt_l, main_hidden) = if prefix_cache_on {
            m.forward_prefix_cached_capture_main(prompt_i32, priming_len, cache)?
        } else {
            m.reset_states()?;
            m.forward_capture_main(prompt_i32, 0)?
        };
        let mh = main_hidden.as_ref()
            .ok_or_else(|| anyhow::anyhow!("[server-dspark] {role}: no main_hidden capture (need dspark_target_layer_ids 40/41/42)"))?;
        ds.warm(mh, prompt_len)?;

        let mut logits_host: Vec<f32> = dev.dtoh_sync_copy(&xt_l)?;
        let mut carry = dsv4_argmax(&logits_host) as u32;
        let mut carry_pos = prompt_len;
        let mut generated: Vec<u32> = Vec::with_capacity(max_new);
        let (s0_steps, s0_offered, s0_accepted, s0_fast, s0_re, s0_c, s0_d, s0_v) =
            (self.n_steps, self.n_offered, self.n_accepted, self.n_fast, self.t_reforward,
             self.t_carryfwd, self.t_draft, self.t_verify);
        let t_req = std::time::Instant::now();

        loop {
            // forward the carry → logits predict carry_pos+1, main_hidden@carry_pos.
            dspark_stream_sync(m, ds, phase_sync);
            let _tc = std::time::Instant::now();
            let (lc, mh_carry) = m.forward_capture_main(std::slice::from_ref(&(carry as i32)), carry_pos)?;
            dspark_stream_sync(m, ds, phase_sync);
            self.t_carryfwd += _tc.elapsed().as_secs_f64();
            logits_host = dev.dtoh_sync_copy(&lc)?;
            generated.push(carry);
            if let Some(f) = emit.as_mut() { f(carry)?; }
            if carry == eos { eprintln!("[server-dspark] {role} — EOS (carry); stopping."); break; }
            if generated.len() >= max_new { break; }

            let r = dsv4_argmax(&logits_host) as i32; // greedy token at carry_pos+1 (ALWAYS committed)
            // r(D) calibration (item 3.4a + T4/E17): first request's step 0 (short ctx) then re-calibrate
            // at each 16K-context bucket — the ≥16K KV-bound regime where adaptive truncation pays.
            // Same SPMD sequence as the one-shot path; the head ships its table over the retained
            // control stream, the node discards its own. Both ranks see the same carry_pos →
            // identical trigger (a divergence would desync the DsparkRd exchange).
            if carry_pos >= self.next_calib_at
                && gb10_inference::tp::tp_config().and_then(|c| c.dspark_depth).is_none() {
                let once = self.next_calib_at == 0;
                let rtab = dspark_calibrate_rd(m, ds, mh_carry.as_ref().unwrap(), r, carry_pos, block)?;
                ds.t_chain = 0.0;
                ds.t_markov = 0.0;
                self.policy.set_r(rtab.clone());
                self.next_calib_at = carry_pos + DSPARK_CTX_BUCKET;
                if is_head {
                    let table: Vec<(u32, f32)> = rtab.iter().map(|&(d, r)| (d as u32, r)).collect();
                    let s = ctl.as_mut()
                        .ok_or_else(|| anyhow::anyhow!("[server-dspark] adaptive depth needs the retained control stream (head)"))?;
                    gb10_inference::tp_serve::send_serving(&mut **s, &gb10_inference::tp_serve::ServingMsg::DsparkRd { table })?;
                    println!("{role} — DSpark r(D) table shipped to node ({}, next re-calib @{} +{})",
                             if once { "first request" } else { "ctx-bucket re-calibration" },
                             carry_pos, DSPARK_CTX_BUCKET);
                } else {
                    let got = gb10_inference::tp_serve::recv_serving(ctl.as_mut()
                        .expect("[server-dspark] adaptive depth needs the retained control stream (node)"))?;
                    match got {
                        gb10_inference::tp_serve::ServingMsg::DsparkRd { table } => {
                            self.policy.set_r(table.iter().map(|&(d, r)| (d as usize, r)).collect());
                        }
                        other => anyhow::bail!("[server-dspark] expected DsparkRd from head, got {other:?}"),
                    }
                }
                println!("{role} — r(D) table ({}):", if once { "once per process, first request" } else { "per-ctx-bucket re-calibration" });
                for &(d, r) in &rtab {
                    println!("    D={}: {:.2}x decode — pays if yield > {:.2} tok/step", d, r, r);
                }
            }
            dspark_stream_sync(m, ds, phase_sync);
            let depth = self.policy.depth();
            let _td = std::time::Instant::now();
            let draft_out = ds.draft_n(mh_carry.as_ref().unwrap(), r, carry_pos, depth)?;
            dspark_stream_sync(m, ds, phase_sync);
            self.t_draft += _td.elapsed().as_secs_f64();
            // verify [r, d1..dD] at carry_pos+1 (depth+1 rows)
            let verify_ids: Vec<i32> = std::iter::once(r).chain(draft_out.drafts.iter().copied()).collect();
            let snap = m.snapshot_verify_state()?;
            dspark_stream_sync(m, ds, phase_sync);
            let _tv = std::time::Instant::now();
            let (vl, vmh) = m.forward_verify_capture_main(&verify_ids, carry_pos + 1)?;
            dspark_stream_sync(m, ds, phase_sync);
            let vh: Vec<f32> = dev.dtoh_sync_copy(&vl)?;
            self.t_verify += _tv.elapsed().as_secs_f64();
            // acceptance: row j-1 (at carry_pos+j) predicts carry_pos+j+1; argmax vs draft_out.drafts[j-1].
            let mut k = 0usize;
            for j in 1..=depth {
                let row_arg = dsv4_argmax(&vh[(j - 1) * vocab..j * vocab]);
                if row_arg == draft_out.drafts[j - 1] as usize { k = j; } else { break; }
            }
            self.n_steps += 1;
            self.n_offered += depth as u64;
            self.n_accepted += k as u64;
            self.policy.record_step(depth, k, 1 + k as u64);
            self.policy.tick(self.n_steps);
            // r is the greedy token at carry_pos+1 — always correct (the trunk's own prediction,
            // confirmed by the verify's row-0 forward). Emit it, then the accepted drafts.
            generated.push(r as u32);
            if let Some(f) = emit.as_mut() { f(r as u32)?; }
            if r as u32 == eos { eprintln!("[server-dspark] {role} — EOS (verify real token); stopping."); break; }
            // max_new boundary: stop BEFORE committing drafts (greedy parity — the server must
            // never emit past max_new; the one-shot bench path has the same latent overshoot,
            // noted in DSV4_SESSION_12_HANDOFF).
            if generated.len() >= max_new { break; }
            // commit d1..dk — never past the max_new boundary (greedy parity; the one-shot
            // bench path has the same latent overshoot, noted in DSV4_SESSION_12_HANDOFF).
            let mut hit_eos = false;
            for j in 0..k {
                if generated.len() >= max_new { break; }
                let t = draft_out.drafts[j] as u32;
                generated.push(t);
                if let Some(f) = emit.as_mut() { f(t)?; }
                if t == eos { hit_eos = true; break; }
            }
            if hit_eos { eprintln!("[server-dspark] {role} — EOS (accepted draft); stopping."); break; }
            if generated.len() >= max_new { break; }

            if k < depth {
                // rejection: rollback trunk + re-apply committed prefix SELECTIVELY (R4).
                let corrected = dsv4_argmax(&vh[k * vocab..(k + 1) * vocab]) as u32; // row k → carry_pos+k+2
                m.restore_verify_state(&snap)?;
                dspark_stream_sync(m, ds, phase_sync);
                let _tr = std::time::Instant::now();
                m.readvance_committed(carry_pos + 1, k + 1)?;
                dspark_stream_sync(m, ds, phase_sync);
                self.reforward_toks += (k + 1) as u64;
                self.t_reforward += _tr.elapsed().as_secs_f64();
                // re-prime the draft ring for the committed positions (real verify hiddens).
                ds.warm_range(vmh.as_ref().unwrap(), k + 1, carry_pos + 1)?;
                carry = corrected;
                carry_pos = carry_pos + k + 2;
            } else {
                // all accepted: bonus = row depth (the (depth+1)th verify row) → carry_pos + depth + 2.
                let bonus = dsv4_argmax(&vh[depth * vocab..(depth + 1) * vocab]) as u32;
                self.n_fast += 1;
                ds.warm_range(vmh.as_ref().unwrap(), depth + 1, carry_pos + 1)?;
                carry = bonus;
                carry_pos = carry_pos + depth + 2;
            }
        }
        let dt = t_req.elapsed().as_secs_f64();
        if is_head {
            let n = generated.len();
            let steps = self.n_steps - s0_steps;
            let off = self.n_offered - s0_offered;
            let acc = self.n_accepted - s0_accepted;
            let acc_pct = if off > 0 { 100.0 * acc as f64 / off as f64 } else { 0.0 };
            let e_adv = if steps > 0 { n as f64 / steps as f64 } else { 0.0 };
            println!("[server-dspark] HEAD request: {n} tok in {dt:.2}s ({:.1} tok/s) | steps {steps}, \
                      drafts {off}, accepted {acc} ({acc_pct:.1}%), E[advance] {e_adv:.2}, fast {}/{steps}, \
                      carry {:.1} ms draft {:.1} ms verify {:.1} ms reforward {:.1} ms (per step)",
                      n as f64 / dt.max(1e-9),
                      self.n_fast - s0_fast,
                      (self.t_carryfwd - s0_c) / steps.max(1) as f64 * 1000.0,
                      (self.t_draft - s0_d) / steps.max(1) as f64 * 1000.0,
                      (self.t_verify - s0_v) / steps.max(1) as f64 * 1000.0,
                      (self.t_reforward - s0_re as f64) / steps.max(1) as f64 * 1000.0);
            let hz = self.policy.hazard_counts();
            if !hz.is_empty() {
                print!("[server-dspark] HEAD yield curve (P(draft j accepted | earlier accepted)):");
                for (j, &(a, o)) in hz.iter().enumerate() {
                    let p = if o > 0 { 100.0 * a as f64 / o as f64 } else { 0.0 };
                    print!(" d{}={:.0}%({})", j + 1, p, o);
                }
                println!();
            }
            if !self.policy.windows.is_empty() {
                let hist: Vec<String> = self.policy.windows.iter()
                    .map(|&(s, d, x)| format!("@{}:d{}:{x:.2}x", s, d)).collect();
                println!("[server-dspark] adaptive depth history: {}", hist.join(" "));
            }
        }
        Ok(generated)
    }
}

/// DSV4 TP=2 persistent HTTP server (the `--server-port` path). Loads the model once,
/// attaches the TP link WITHOUT consuming the TpContext (so `broadcast_prompt` can be
/// called repeatedly), then loops: accept HTTP → parse → encode → reset state → broadcast
/// → SPMD greedy decode → OpenAI JSON response → continue. Handles bad/probe requests
/// (empty body, GET, missing Content-Type) by returning 400 and keeping the loop alive.
/// Model id reported by the DSV4 minimal server — in completions AND /v1/models (keep in sync with
/// what OpenWebUI will send back as `model`).
/// Item 3.4 (`--server-dspark on`): each request instead runs the DSpark draft/verify/rollback
/// loop via [`DsparkServeState`] (both ranks SPMD; the head ships the r(D) table over the
/// retained control stream once per process). Default OFF = the pre-flag greedy path byte-for-byte.
const DSV4_MODEL_ID: &str = "DeepSeek-V4-Flash-DSpark";

pub(super) fn dsv4_tp_serve_server(
    model_dir: &str,
    ctx_result: anyhow::Result<gb10_inference::tp::TpContext>,
    port: u16,
    default_max_tokens: usize,
    ctl: Option<std::net::TcpStream>,
    tpc: &gb10_inference::tp::TpConfig,
) -> anyhow::Result<()> {
    use gb10_inference::dsv4_load;
    use cudarc::driver::CudaDevice;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    let mut ctx = ctx_result?;
    ctx.sanity()?;
    if gb10_inference::boot_report::debug() {
        println!("HEAD (rank 0/2) — DSV4 TP LINK UP (server mode, port {port})");
    }
    let st_weights = gb10_inference::boot_report::Stage::begin("weights");

    let cfg = dsv4_load::load_config(std::path::Path::new(model_dir))
        .map_err(|e| anyhow::anyhow!("load_config: {e:#}"))?;
    let model_id_str: String = std::path::Path::new(model_dir.trim_end_matches('/'))
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| DSV4_MODEL_ID.to_string());
    if gb10_inference::boot_report::debug() {
        println!("[dsv4-server] model ID: {model_id_str}");
    }
    let dev = CudaDevice::new(0)?;
    // max_seq_len (KV-ring depth): env --max-seq-len (operator override) > the shipped
    // TpConfig (--max-seq-len on the head, zero-config to the node) > 4096.
    let max_seq_len = gb10_inference::opts::var(gb10_inference::opt!("max-seq-len")).ok()
        .and_then(|v| v.parse::<usize>().ok())
        .or_else(|| gb10_inference::tp::tp_config().map(|c| c.max_seq_len).filter(|&v| v > 0))
        .unwrap_or(4096);
    let s_max = gb10_inference::dsv4_model::PREFILL_CHUNK;
    let (m0, load_secs) = dsv4_load_for_serve(&dev, std::path::Path::new(model_dir), &cfg,
                                                   max_seq_len, s_max, 0, 2)?;
    st_weights.ok();
    if gb10_inference::boot_report::debug() {
        println!("[dsv4-server] head shard loaded (43 layers, rank 0/2) in {load_secs:.1}s — listening on port {port} (max_seq_len={max_seq_len})");
    }

    // NOTE: TP attach is deferred to the first request (matches the old one-shot timing:
    // broadcast_prompt uses the raw RDMA exchange, not the proxy; the proxy is only needed
    // for the per-step all-reduce during decode. Attaching after broadcast_prompt keeps the
    // QP fresh — attaching before a 100s load gap causes "transport retry exceeded").

    let tok_path = format!("{}/tokenizer.json", model_dir.trim_end_matches('/'));
    let tok = QwenTokenizer::from_file(&tok_path)?;
    let addr = gb10_inference::server::http_bind_addr(port)?;
    let listener = TcpListener::bind(addr).map_err(|e| anyhow::anyhow!("cannot listen on {addr}: {e}"))?;
    if gb10_inference::boot_report::debug() {
        println!("[dsv4-server] listening on http://{addr}/v1/chat/completions");
    }
    {
        // OUT1: the one idle line for the DSV4 family, resolved from the registry.
        // (UNVERIFIED ON GPU — DSV4 serving needs a TP=2 pair; .11 has none.)
        let kv_b = if gb10_inference::opts::var(gb10_inference::opt!("kv-k8v8")).ok().as_deref() == Some("1") { "k8v8" }
                   else if gb10_inference::opts::var(gb10_inference::opt!("kv-k8v4")).ok().as_deref() == Some("1") { "k8v4" } else { "bf16" };
        gb10_inference::boot_report::serving(&gb10_inference::boot_report::ServingLine {
            family: "dsv4", model: &model_id_str, tp: 2, ctx: max_seq_len, batch: tpc.max_batch, kv: kv_b,
            prefix: if tpc.prefix_cache { "on" } else { "off" },
            mtp: "off",
            spec: if tpc.server_dspark { "dspark" } else { "plain" },
            port,
            url: format!("http://{addr}"),
        });
    }

    let mut tp_attached = false;
    // R2.3 prefix-cache (+ item 2.3 LRU): snapshot/restore at 128-aligned conversation-prefix
    // boundaries. Turn 2+ forwards only the delta — bitwise-identical to a full re-prefill
    // (§12.B.5 extends across request boundaries). Both ranks run the same local cache (SPMD —
    // the forward sequence is identical, only the prefill row count changes). Gated by the
    // head's --prefix-cache flag (rides TpConfig; default ON for the dsv4 server).
    let prefix_cache_on = gb10_inference::tp::tp_config().map(|c| c.prefix_cache).unwrap_or(true);
    let mut prefix_cache = gb10_inference::dsv4_model::PrefixCache::new(8);
    // Item 3.4: the DSpark serve state (drafter + depth policy + once-per-process calibration).
    // Only constructed when --server-dspark on — the greedy default stays untouched (no extra
    // load, byte-identical behavior). The trunk model moves INTO the state in that case; the
    // greedy path keeps it as `m0` (see the per-request branch below).
    let mut m: Option<gb10_inference::dsv4_model::Dsv4GpuModel> = Some(m0);
    let mut dspark_state: Option<DsparkServeState> = if tpc.server_dspark {
        Some(DsparkServeState::new(&dev, std::path::Path::new(model_dir), &cfg, max_seq_len,
                                   m.take().expect("fresh model"), 0, true)?)
    } else {
        None
    };
    let mut ctl_mut = ctl;
    loop {
        let (mut sock, peer) = match listener.accept() {
            Ok(s) => s,
            Err(e) => { eprintln!("[dsv4-server] accept error: {e}"); continue; }
        };
        let _ = sock.set_read_timeout(Some(std::time::Duration::from_secs(10)));
        eprintln!("[dsv4-server] connection from {peer}");

        // ---- robust HTTP read (handles \r\n\r\n and \n\n; skips probes/GETs) ----
        let mut buf = Vec::with_capacity(8192);
        let mut tmp = [0u8; 8192];
        let mut clen: Option<usize> = None;
        let mut got_headers = false;
        loop {
            let n = match sock.read(&mut tmp) { Ok(0) => break, Ok(n) => n, Err(_) => break };
            buf.extend_from_slice(&tmp[..n]);
            let s = String::from_utf8_lossy(&buf);
            if !got_headers {
                let hdr_end = s.find("\r\n\r\n").map(|i| i + 4)
                    .or_else(|| s.find("\n\n").map(|i| i + 2));
                if hdr_end.is_some() {
                    got_headers = true;
                    let is_crlf = s.contains("\r\n\r\n");
                    let head = &s[..hdr_end.unwrap() - if is_crlf { 4 } else { 2 }];
                    if let Some(line) = head.lines().find(|l| l.to_ascii_lowercase().starts_with("content-length:")) {
                        clen = line.split(':').nth(1).and_then(|v| v.trim().parse().ok());
                    }
                }
            }
            if got_headers {
                if let Some(cl) = clen {
                    let hdr_end = s.find("\r\n\r\n").map(|i| i + 4)
                        .or_else(|| s.find("\n\n").map(|i| i + 2)).unwrap();
                    if buf.len() >= hdr_end + cl { break; }
                } else {
                    break;
                }
            }
        }

        if !got_headers {
            let _ = sock.write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
            continue;
        }

        let req_text = String::from_utf8_lossy(&buf);
        let body_start = req_text.find("\r\n\r\n").map(|i| i + 4)
            .or_else(|| req_text.find("\n\n").map(|i| i + 2))
            .unwrap_or(req_text.len());

        // GETs: the OpenAI model-list endpoints (same shape as src/server.rs list_models/get_model —
        // OpenWebUI probes /v1/models before it will talk to the server at all). Anything else GET
        // stays a 400 probe answer.
        let request_line = req_text[..body_start].lines().next().unwrap_or("");
        let mut rl = request_line.split_whitespace();
        let (method, path) = (rl.next().unwrap_or(""), rl.next().unwrap_or(""));
        if method.eq_ignore_ascii_case("get") {
            let created = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs()).unwrap_or(0);
            let card = format!("{{\"id\":\"{model_id_str}\",\"object\":\"model\",\"created\":{created},\"owned_by\":\"rust_infer\"}}");
            let (code, ctype, body_out) = if path == "/v1/models" {
                ("200 OK", "application/json", format!("{{\"object\":\"list\",\"data\":[{card}]}}"))
            } else if let Some(id) = path.strip_prefix("/v1/models/") {
                if id == model_id_str { ("200 OK", "application/json", card) }
                else { ("404 Not Found", "text/plain", format!("Model '{id}' not found. Available: {model_id_str}")) }
            } else {
                ("400 Bad Request", "application/json", String::new())
            };
            let http = format!("HTTP/1.1 {code}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body_out}", body_out.len());
            let _ = sock.write_all(http.as_bytes());
            if code != "400 Bad Request" { eprintln!("[dsv4-server] {method} {path} → {code}"); }
            continue;
        }

        let body_str = req_text[body_start..].trim_start_matches(['\r', '\n']);

        if body_str.is_empty() {
            let _ = sock.write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
            continue;
        }

        let body_json = match gb10_inference::dsv4_chat::parse_json(body_str) {
            Ok(j) => j,
            Err(e) => {
                eprintln!("[dsv4-server] JSON parse error at byte {}: {}", e.0, e.1);
                let _ = sock.write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
                continue;
            }
        };

        let max_new = body_json.get("max_tokens").and_then(|v| v.as_u64()).map(|n| n as usize).unwrap_or(default_max_tokens);
        let thinking_mode = match body_json.get("thinking").and_then(|v| v.as_bool()) {
            Some(false) => "chat", _ => "thinking",
        };
        let stream = body_json.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
        let include_usage = body_json.get("stream_options")
            .and_then(|v| v.get("include_usage"))
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let (prompt_ids, raw_completion): (Vec<u32>, bool) = if let Some(p) = body_json.get("prompt").and_then(|v| v.as_str()) {
            eprintln!("[dsv4-server] RAW completion: prompt={p:?}, max_tokens={max_new}");
            (tok.encode(p, true)?, true)
        } else {
            let messages = body_json.get("messages").and_then(|m| m.as_array())
                .ok_or_else(|| anyhow::anyhow!("/v1/chat/completions needs `messages`"))?
                .to_vec();
            let mut messages = messages;
            let top_tools = body_json.get("tools").cloned();
            let any_msg_has_tools = messages.iter().any(|m| m.get("tools").is_some());
            if let Some(tools) = top_tools {
                if !any_msg_has_tools {
                    let idx = messages.iter().position(|m| {
                        matches!(m.get("role").and_then(|r| r.as_str()), Some("system") | Some("developer"))
                    }).unwrap_or(0);
                    if let Some(gb10_inference::dsv4_chat::Json::Object(o)) = messages.get_mut(idx) {
                        o.push(("tools".into(), tools));
                    }
                }
            }
            let opts = gb10_inference::dsv4_chat::EncodeOptions { thinking_mode: thinking_mode.into(), ..Default::default() };
            let wire = gb10_inference::dsv4_chat::encode_messages(&messages, &opts);
            let ids = tok.encode(&wire, false)?;
            eprintln!("[dsv4-server] chat ({thinking_mode} mode): {} messages → {} prompt tokens, max_tokens={max_new}",
                      messages.len(), ids.len());
            (ids, false)
        };
        let prompt_len = prompt_ids.len();

        // WORK 1a: context-length guard (head-side only, before broadcast — the node never sees
        // rejected requests, so no SPMD desync is possible). Over-limit → HTTP 400
        // `context_length_exceeded`, server keeps listening. A boundary prompt of exactly
        // `max_seq_len − max_new` passes (strict `>`); the guard uses the SAME `max_seq_len`
        // binding the shard was loaded with (env --max-seq-len > shipped TpConfig > 4096).
        if prompt_ids.len() + max_new > max_seq_len {
            let n = prompt_ids.len();
            let msg = format!(
                "prompt length {n} + max_tokens {max_new} exceeds this server's max_seq_len {max_seq_len} — shorten the conversation or restart with --max-seq-len > {n}+{max_new}");
            let err = serde_json::json!({"error":{"message":msg,"type":"invalid_request_error","code":"context_length_exceeded"}});
            let body_out = serde_json::to_string(&err)?;
            let http = format!("HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body_out.len(), body_out);
            let _ = sock.write_all(http.as_bytes());
            eprintln!("[dsv4-server] REJECT prompt {n}+{max_new} > {max_seq_len} → 400 context_length_exceeded (listening)");
            continue;
        }

        // R2.3 prefix-cache: snapshot/restore at the 128-aligned conversation prefix (before the
        // generation priming). Turn 2+ forwards only the delta — bitwise-identical to a full
        // re-prefill (§12.B.5 extends across request boundaries). Both ranks do the same local
        // split + snapshot/restore (SPMD: both see the same prompt_ids, same deterministic ops).
        // The broadcast sends the FULL prompt_ids; the prefix cache is a local GPU optimization.
        let priming_len = {
            let priming_str = if thinking_mode == "thinking" {
                format!("{}{}", gb10_inference::dsv4_chat::ASSISTANT_SP_TOKEN, gb10_inference::dsv4_chat::THINKING_START_TOKEN)
            } else {
                format!("{}{}", gb10_inference::dsv4_chat::ASSISTANT_SP_TOKEN, gb10_inference::dsv4_chat::THINKING_END_TOKEN)
            };
            tok.encode(&priming_str, false).map(|ids| ids.len()).unwrap_or(0)
        };

        // Broadcast the FULL prompt + priming_len to the node (both ranks do the prefix cache locally).
        let (prompt, _, _) = ctx.broadcast_prompt(Some((prompt_ids.as_slice(), max_new, priming_len)))?;

        // Attach TP on the first request (deferred from load — see comment above). The model
        // lives either in the greedy local (`m`) or inside the DSpark state (item 3.4).
        if !tp_attached {
            let mm = if let Some(st) = dspark_state.as_mut() { &mut st.m } else { m.as_mut().unwrap() };
            let nbytes = mm.cfg.dim * 2;
            ctx.link.set_payload(nbytes, false)?;
            mm.rt.tp_ctx_dptr = ctx.link.ctx_device_ptr();
            mm.tp_rank = 0; // R3.2: enables the vocab-parallel maxloc head (rank 0 = first vocab half)
            let ctx_addr = ctx.link.ctx_addr();
            gb10_inference::net::spawn_proxy(ctx_addr, 19);
            eprintln!("[dsv4-tp] rank 0/2 — RDMA proxy up ({nbytes} B/decode-ring)");
            tp_attached = true;
        }

        // Prefix-cache forward: snapshot/restore at the 128-aligned conversation prefix.
        let prompt_i32: Vec<i32> = prompt.iter().map(|&v| v as i32).collect();
        let aligned_now = ((prompt_i32.len() - priming_len) / 128) * 128;
        let was_hit = prefix_cache_on
            && prefix_cache.lookup(&prompt_i32, aligned_now).map(|e| e.len).is_some();
        if prefix_cache_on {
            if was_hit {
                let hit_len = prefix_cache.lookup(&prompt_i32, aligned_now).map(|e| e.len).unwrap_or(0);
                eprintln!("[dsv4-server] prefix-cache HIT: {} tok cached, delta from {hit_len}",
                    prompt_i32.len());
                prefix_cache.touch(hit_len);
            } else {
                eprintln!("[dsv4-server] prefix-cache MISS: full prefill {} tok (priming {priming_len})",
                    prompt_i32.len());
            }
        }
        let eos = 1u32;
        // R3.2: under TP the next token comes from the vocab-parallel maxloc head (bitwise ==
        // full-vocab argmax, no 129 KB logits dtoh per token); single-process keeps full logits.
        // (Computed in the greedy path — the DSpark branch owns the model inside the state.)

        // Item 3.4 (--server-dspark on): every request runs the DSpark draft/verify/rollback
        // loop (both ranks SPMD; the head ships the r(D) table over the retained control stream
        // once per process). LOSSLESS by construction — the committed/corrected tokens are the
        // trunk's argmaxes, identical to the greedy decode below. SSE streaming (item 3.4 —
        // the tool-eval harness streams every perf request): one delta per generated token via
        // Dsv4SseStreamer, the same reasoning→content split as the greedy path.
        if let Some(st) = dspark_state.as_mut() {
            let _gen = if stream {
                let t0 = std::time::Instant::now();
                let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
                sock.write_all(head.as_bytes())?;
                // S-B9-REL-BENCHFIX2 (W1): no role chunk here. `decode_request` begins with the
                // prefill; a `choices` chunk written before it made a client's "first chunk" a
                // few milliseconds (the defect fixed in src/server.rs). The streamer writes the
                // role chunk with the first generated token (or before the finish chunk when
                // none is generated): gb10_inference::dsv4_sse.
                let decode = |t: u32| tok.decode(std::slice::from_ref(&t), false).ok();
                let mut streamer = Dsv4SseStreamer::new(decode, &mut sock, raw_completion, thinking_mode, &model_id_str, include_usage, t0);
                let gen = st.decode_request(&prompt_i32, priming_len, max_new, &mut prefix_cache,
                                            prefix_cache_on, ctl_mut.as_mut(), true, Some(&mut |t| streamer.feed(t)))?;
                let hit_eos = gen.last().copied() == Some(eos);
                streamer.finish(hit_eos, prompt_len)?;
                eprintln!("[dsv4-server] streamed {} DSpark tokens (finish={}, {} mode) — ready for next request",
                          gen.len(), if hit_eos { "stop" } else { "length" }, if raw_completion { "raw" } else { thinking_mode });
                gen
            } else {
                let t0 = std::time::Instant::now();
                let gen = st.decode_request(&prompt_i32, priming_len, max_new, &mut prefix_cache,
                                            prefix_cache_on, ctl_mut.as_mut(), true, None)?;
                let hit_eos = gen.last().copied() == Some(eos);
                let completion_tokens = if hit_eos { gen.len() - 1 } else { gen.len() };
                let timings = gb10_inference::make_timings(t0, None, prompt_len, completion_tokens);
                let usage_json = serde_json::json!({
                    "prompt_tokens": prompt_len,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_len + completion_tokens,
                });
                let timings_json = serde_json::json!({
                    "prompt_ms": timings.prompt_ms,
                    "predicted_ms": timings.predicted_ms,
                    "prompt_per_second": timings.prompt_per_second,
                    "predicted_per_second": timings.predicted_per_second,
                });
                let text = tok.decode(&gen, false)?;
                eprintln!("[dsv4-server] DSpark generated {} tokens (finish={}, {} mode)", gen.len(),
                          if hit_eos { "stop" } else { "length" }, if raw_completion { "raw" } else { thinking_mode });
                let resp_json: serde_json::Value = if raw_completion {
                    serde_json::json!({
                        "id": "dsv4-tp-1", "object": "chat.completion", "model": model_id_str,
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                     "finish_reason": if hit_eos { "stop" } else { "length" }}],
                        "usage": usage_json,
                        "timings": timings_json,
                    })
                } else {
                    let parsed = gb10_inference::dsv4_chat::parse_completion(&text, thinking_mode);
                    if !parsed.reasoning_content.is_empty() {
                        eprintln!("[dsv4-server] reasoning head: {:?}", parsed.reasoning_content.chars().take(160).collect::<String>());
                    }
                    if !parsed.content.is_empty() {
                        eprintln!("[dsv4-server] content head:    {:?}", parsed.content.chars().take(160).collect::<String>());
                    }
                    let finish = if !parsed.tool_calls.is_empty() { "tool_calls" } else if hit_eos { "stop" } else { "length" };
                    let tool_calls: Vec<serde_json::Value> = parsed.tool_calls.iter().enumerate().map(|(i, tc)| {
                        let name = tc.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str()).unwrap_or("").to_string();
                        let args = tc.get("function").and_then(|f| f.get("arguments")).and_then(|a| a.as_str()).unwrap_or("").to_string();
                        serde_json::json!({"index": i, "id": format!("call_{i:03}"), "type": "function",
                                           "function": {"name": name, "arguments": args}})
                    }).collect();
                    let mut message = serde_json::json!({"role": "assistant"});
                    if !parsed.tool_calls.is_empty() {
                        message["content"] = if parsed.content.is_empty() { serde_json::Value::Null } else { serde_json::Value::String(parsed.content.clone()) };
                        message["tool_calls"] = serde_json::Value::Array(tool_calls);
                    } else {
                        message["content"] = serde_json::Value::String(parsed.content.clone());
                    }
                    if !parsed.reasoning_content.is_empty() {
                        message["reasoning_content"] = serde_json::Value::String(parsed.reasoning_content);
                    }
                    serde_json::json!({
                        "id": "dsv4-tp-1", "object": "chat.completion", "model": model_id_str,
                        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                        "usage": usage_json,
                        "timings": timings_json,
                    })
                };
                let body_out = serde_json::to_string(&resp_json)?;
                let http = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body_out.len(), body_out);
                sock.write_all(http.as_bytes())?;
                eprintln!("[dsv4-server] responded ({} DSpark tokens) — ready for next request", gen.len());
                gen
            };
            continue;
        }
        // Greedy path: shadow the Option with the owned model (the DSpark branch above took it
        // when --server-dspark on — the greedy default is untouched byte-for-byte).
        let m = m.as_mut().expect("greedy server owns the model");
        // R3.2: under TP the next token comes from the vocab-parallel maxloc head.
        let maxloc = m.rt.tp_ctx_dptr != 0 && m.tp_rank >= 0;
        // The request clock of the `timings` block starts HERE, before the prefill: `prompt_ms`
        // is request -> first token, so `prompt_per_second` is the prefill rate. It used to be
        // taken after the prefill (in both branches below), and the first token already exists
        // when the decode loop starts: prompt tokens divided by a few microseconds, a fictional
        // rate (S-B9-REL-SURFACE2B R6f; the DSpark branch above always started its clock first).
        let t0 = std::time::Instant::now();
        let mut next: u32 = if maxloc {
            if prefix_cache_on {
                m.forward_prefix_cached_next(&prompt_i32, priming_len, &mut prefix_cache)?
                    .expect("maxloc head under TP")
            } else {
                m.reset_states()?;
                // chunked full prefill (forward_streams processes <= s_max rows per call —
                // the RUN-4 crash class; forward_prefill_chunked is bitwise one-shot).
                let (xt, tl) = if prompt_i32.len() > gb10_inference::dsv4_model::PREFILL_CHUNK {
                    m.forward_prefill_chunked(&prompt_i32, gb10_inference::dsv4_model::PREFILL_CHUNK)?
                } else {
                    (m.forward_streams(&prompt_i32, 0)?, prompt_i32.len())
                };
                m.forward_head_next(&xt, tl)?.expect("maxloc head under TP")
            }
        } else {
            let l = if prefix_cache_on {
                m.forward_prefix_cached(&prompt_i32, priming_len, &mut prefix_cache)?
            } else {
                m.reset_states()?;
                let (xt, tl) = if prompt_i32.len() > gb10_inference::dsv4_model::PREFILL_CHUNK {
                    m.forward_prefill_chunked(&prompt_i32, gb10_inference::dsv4_model::PREFILL_CHUNK)?
                } else {
                    (m.forward_streams(&prompt_i32, 0)?, prompt_i32.len())
                };
                let (_, l) = m.forward_head(&xt, tl)?;
                l
            };
            dsv4_argmax(&dev.dtoh_sync_copy(&l)?) as u32
        };
        let mut generated: Vec<u32> = Vec::with_capacity(max_new);
        let mut pos = prompt_i32.len();
        let mut hit_eos = false;
        if stream {
            // WORK 3: SSE streaming path. The decode loop runs the IDENTICAL forward sequence
            // as the non-stream path (SPMD with the node — the node sees no change); only the
            // head's HTTP framing differs. The reasoning→content split mirrors src/server.rs
            // ~389-456 and uses the SAME boundary marker `parse_completion` recognizes
            // (THINKING_END_TOKEN), so a client concatenating deltas reconstructs the split
            // `parse_completion` would produce. decode is per-token with skip_special=false so
            // the close marker stays visible for the split; a partial close-tag prefix
            // straddling two decode chunks is held back (dsv4_partial_overlap).
            let mut first_tok: Option<std::time::Instant> = None;
            let mut n_emitted: usize = 0;
            let think_close = gb10_inference::dsv4_chat::THINKING_END_TOKEN;
            let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
            sock.write_all(head.as_bytes())?;
            // The role chunk (the first chunk with `choices`) is written HERE, after the prefill:
            // `next` above is already the first generated token. (The headers go out here too, so
            // this path sends nothing at all during its prefill.)
            let role = serde_json::json!({"id":"dsv4-tp-1","object":"chat.completion.chunk","model":model_id_str,
                "choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]});
            dsv4_sse_chunk(&mut sock, &serde_json::to_string(&role)?)?;

            let mut acc = String::new();
            let mut reason_emitted: usize = 0;
            let mut content_emitted: usize = 0;
            // thinking mode starts INSIDE the think block (prompt primed `imd`); raw completion
            // and chat mode are all content (no reasoning split).
            let mut content_start: Option<usize> = if !raw_completion && thinking_mode == "thinking" { None } else { Some(0) };

            for step in 0..max_new {
                if next == eos { eprintln!("[dsv4-server] EOS at step {step}"); hit_eos = true; break; }
                if first_tok.is_none() { first_tok = Some(std::time::Instant::now()); }
                n_emitted += 1;
                generated.push(next);
                if let Ok(piece) = tok.decode(std::slice::from_ref(&next), false) {
                    if !piece.is_empty() {
                        acc.push_str(&piece);
                        match content_start {
                            None => {
                                if let Some(idx) = acc[reason_emitted..].find(think_close).map(|i| reason_emitted + i) {
                                    if idx > reason_emitted {
                                        let d = serde_json::json!({"id":"dsv4-tp-1","object":"chat.completion.chunk","model":model_id_str,
                                            "choices":[{"index":0,"delta":{"reasoning_content":&acc[reason_emitted..idx]},"finish_reason":null}]});
                                        dsv4_sse_chunk(&mut sock, &serde_json::to_string(&d)?)?;
                                    }
                                    let cs = idx + think_close.len();
                                    let mut lead = cs;
                                    while lead < acc.len() && matches!(acc.as_bytes()[lead], b'\n'|b'\r'|b' '|b'\t') { lead += 1; }
                                    if lead < acc.len() {
                                        let d = serde_json::json!({"id":"dsv4-tp-1","object":"chat.completion.chunk","model":model_id_str,
                                            "choices":[{"index":0,"delta":{"content":&acc[lead..acc.len()]},"finish_reason":null}]});
                                        dsv4_sse_chunk(&mut sock, &serde_json::to_string(&d)?)?;
                                    }
                                    content_start = Some(lead);
                                    content_emitted = acc.len();
                                } else {
                                    let overlap = dsv4_partial_overlap(&acc, think_close);
                                    let safe = acc.len() - overlap;
                                    if safe > reason_emitted {
                                        let d = serde_json::json!({"id":"dsv4-tp-1","object":"chat.completion.chunk","model":model_id_str,
                                            "choices":[{"index":0,"delta":{"reasoning_content":&acc[reason_emitted..safe]},"finish_reason":null}]});
                                        dsv4_sse_chunk(&mut sock, &serde_json::to_string(&d)?)?;
                                        reason_emitted = safe;
                                    }
                                }
                            }
                            Some(_) => {
                                if acc.len() > content_emitted {
                                    let d = serde_json::json!({"id":"dsv4-tp-1","object":"chat.completion.chunk","model":model_id_str,
                                        "choices":[{"index":0,"delta":{"content":&acc[content_emitted..acc.len()]},"finish_reason":null}]});
                                    dsv4_sse_chunk(&mut sock, &serde_json::to_string(&d)?)?;
                                    content_emitted = acc.len();
                                }
                            }
                        }
                    }
                }
                next = if maxloc {
                    m.forward_next(std::slice::from_ref(&(next as i32)), pos)?.expect("maxloc head under TP")
                } else {
                    let l = m.forward(std::slice::from_ref(&(next as i32)), pos)?;
                    dsv4_argmax(&dev.dtoh_sync_copy(&l)?) as u32
                };
                pos += 1;
            }
            // Flush any held-back tail (no close tag found, or trailing content).
            match content_start {
                Some(_) => {
                    if acc.len() > content_emitted {
                        let d = serde_json::json!({"id":"dsv4-tp-1","object":"chat.completion.chunk","model":model_id_str,
                            "choices":[{"index":0,"delta":{"content":&acc[content_emitted..acc.len()]},"finish_reason":null}]});
                        dsv4_sse_chunk(&mut sock, &serde_json::to_string(&d)?)?;
                    }
                }
                None => {
                    if acc.len() > reason_emitted {
                        let d = serde_json::json!({"id":"dsv4-tp-1","object":"chat.completion.chunk","model":model_id_str,
                            "choices":[{"index":0,"delta":{"reasoning_content":&acc[reason_emitted..acc.len()]},"finish_reason":null}]});
                        dsv4_sse_chunk(&mut sock, &serde_json::to_string(&d)?)?;
                    }
                }
            }
            // Final chunk + usage chunk (if enabled) + the OpenAI SSE terminator.
            let finish = if hit_eos { "stop" } else { "length" };
            let final_chunk = serde_json::json!({"id":"dsv4-tp-1","object":"chat.completion.chunk","model":model_id_str,
                "choices":[{"index":0,"delta":{},"finish_reason":finish}]});
            dsv4_sse_chunk(&mut sock, &serde_json::to_string(&final_chunk)?)?;
            if include_usage {
                let timings = gb10_inference::make_timings(t0, first_tok, prompt_len, n_emitted);
                let usage_chunk = serde_json::json!({
                    "id": "dsv4-tp-1", "object": "chat.completion.chunk", "model": model_id_str,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": prompt_len,
                        "completion_tokens": n_emitted,
                        "total_tokens": prompt_len + n_emitted,
                    },
                    "timings": {
                        "prompt_ms": timings.prompt_ms,
                        "predicted_ms": timings.predicted_ms,
                        "prompt_per_second": timings.prompt_per_second,
                        "predicted_per_second": timings.predicted_per_second,
                    },
                });
                dsv4_sse_chunk(&mut sock, &serde_json::to_string(&usage_chunk)?)?;
            }
            sock.write_all(b"data: [DONE]\r\n\r\n")?;
            sock.flush()?;
            eprintln!("[dsv4-server] streamed {} tokens (finish={finish}, {} mode) — ready for next request", generated.len(), if raw_completion { "raw" } else { thinking_mode });
        } else {
            let mut first_tok: Option<std::time::Instant> = None;
            for step in 0..max_new {
                if next == eos { eprintln!("[dsv4-server] EOS at step {step}"); hit_eos = true; break; }
                if first_tok.is_none() { first_tok = Some(std::time::Instant::now()); }
                generated.push(next);
                next = if maxloc {
                    m.forward_next(std::slice::from_ref(&(next as i32)), pos)?.expect("maxloc head under TP")
                } else {
                    let l = m.forward(std::slice::from_ref(&(next as i32)), pos)?;
                    dsv4_argmax(&dev.dtoh_sync_copy(&l)?) as u32
                };
                pos += 1;
            }

            let completion_tokens = generated.len(); // EOS breaks before push
            let timings = gb10_inference::make_timings(t0, first_tok, prompt_len, completion_tokens);
            let usage_json = serde_json::json!({
                "prompt_tokens": prompt_len,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_len + completion_tokens,
            });
            let timings_json = serde_json::json!({
                "prompt_ms": timings.prompt_ms,
                "predicted_ms": timings.predicted_ms,
                "prompt_per_second": timings.prompt_per_second,
                "predicted_per_second": timings.predicted_per_second,
            });

            let text = tok.decode(&generated, false)?;
            eprintln!("[dsv4-server] generated {} tokens (finish={}, {} mode)", generated.len(),
                      if hit_eos { "stop" } else { "length" }, if raw_completion { "raw" } else { thinking_mode });

            let resp_json: serde_json::Value = if raw_completion {
                serde_json::json!({
                    "id": "dsv4-tp-1", "object": "chat.completion", "model": model_id_str,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                 "finish_reason": if hit_eos { "stop" } else { "length" }}],
                    "usage": usage_json,
                    "timings": timings_json,
                })
            } else {
                let parsed = gb10_inference::dsv4_chat::parse_completion(&text, thinking_mode);
                if !parsed.reasoning_content.is_empty() {
                    eprintln!("[dsv4-server] reasoning head: {:?}", parsed.reasoning_content.chars().take(160).collect::<String>());
                }
                if !parsed.content.is_empty() {
                    eprintln!("[dsv4-server] content head:    {:?}", parsed.content.chars().take(160).collect::<String>());
                }
                let finish = if !parsed.tool_calls.is_empty() { "tool_calls" } else if hit_eos { "stop" } else { "length" };
                let tool_calls: Vec<serde_json::Value> = parsed.tool_calls.iter().enumerate().map(|(i, tc)| {
                    let name = tc.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str()).unwrap_or("").to_string();
                    let args = tc.get("function").and_then(|f| f.get("arguments")).and_then(|a| a.as_str()).unwrap_or("").to_string();
                    serde_json::json!({"index": i, "id": format!("call_{i:03}"), "type": "function",
                                       "function": {"name": name, "arguments": args}})
                }).collect();
                let mut message = serde_json::json!({"role": "assistant"});
                if !parsed.tool_calls.is_empty() {
                    message["content"] = if parsed.content.is_empty() { serde_json::Value::Null } else { serde_json::Value::String(parsed.content.clone()) };
                    message["tool_calls"] = serde_json::Value::Array(tool_calls);
                } else {
                    message["content"] = serde_json::Value::String(parsed.content.clone());
                }
                if !parsed.reasoning_content.is_empty() {
                    message["reasoning_content"] = serde_json::Value::String(parsed.reasoning_content);
                }
                serde_json::json!({
                    "id": "dsv4-tp-1", "object": "chat.completion", "model": model_id_str,
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                    "usage": usage_json,
                    "timings": timings_json,
                })
            };
            let body_out = serde_json::to_string(&resp_json)?;
            let http = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body_out.len(), body_out);
            sock.write_all(http.as_bytes())?;
            eprintln!("[dsv4-server] responded ({} generated tokens) — ready for next request", generated.len());
        }
    }
}

/// GPU-sync both streams (the --dspark-phase-ms protocol — `dev.synchronize()` does NOT cover
/// `rt.stream`). Free fn so the step loop's `&mut` borrows of `m`/`ds` stay disjoint.
fn dspark_stream_sync(m: &gb10_inference::dsv4_model::Dsv4GpuModel, ds: &gb10_inference::dsv4_dspark::Dsv4DSpark, phase_sync: bool) {
    if phase_sync {
        unsafe {
            cudarc::driver::result::stream::synchronize(m.rt.stream.stream).ok();
            cudarc::driver::result::stream::synchronize(ds.rt.stream.stream).ok();
        }
    }
}

/// Measure r(D) = (carry + draft(D) + verify(D+1)) / carry for D in 1..=block at the serve's
/// CURRENT context, on the real state (step 0's carry just forwarded — the values are
/// timing-neutral, but real main_hidden/token keep every kernel on its true path).
///
/// TP discipline (item 3.3 / TP item E): BOTH ranks run this identical SPMD sequence — the
/// verify forwards' all-reduces are barriers the head waits on — and each rank times its own
/// phases; the head's table is then SHIPPED over the retained control stream and the node
/// discards its own, so both policies drive from one identical set of numbers (a per-rank
/// timing difference must never diverge the depth decision).
///
/// State hygiene: the trunk is snapshot/restored around the calibration (rollback-gate
/// semantics, bitwise); the draft's single ring write at `carry_pos` is overwritten by step 0's
/// real draft. `x_cap` capture buffers are re-captured by the real verify before any rollback.
fn dspark_calibrate_rd(
    m: &mut gb10_inference::dsv4_model::Dsv4GpuModel,
    ds: &mut gb10_inference::dsv4_dspark::Dsv4DSpark,
    mh_carry: &gb10_inference::dsv4_attn::B,
    real_token: i32,
    carry_pos: usize,
    block: usize,
) -> anyhow::Result<Vec<(usize, f32)>> {
    let sync = |m: &gb10_inference::dsv4_model::Dsv4GpuModel, ds: &gb10_inference::dsv4_dspark::Dsv4DSpark| {
        unsafe {
            cudarc::driver::result::stream::synchronize(m.rt.stream.stream).ok();
            cudarc::driver::result::stream::synchronize(ds.rt.stream.stream).ok();
        }
    };
    let snap = m.snapshot_verify_state()?;
    let mut time_it = |name: &str, n: usize, f: &mut dyn FnMut() -> anyhow::Result<()>| -> anyhow::Result<f64> {
        f()?; // warm
        let t0 = std::time::Instant::now();
        for _ in 0..n { f()?; }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        eprintln!("[dspark-depth] calib {name}: {ms:.1} ms");
        Ok(ms)
    };

    let mut time_decode = || {
        let _ = m.forward_capture_main(std::slice::from_ref(&real_token), carry_pos)?;
        sync(m, ds);
        Ok::<_, anyhow::Error>(())
    };
    let decode = time_it("decode", 3, &mut time_decode)?;

    let mut table: Vec<(usize, f32)> = Vec::new();
    for d in 1..=block {
        let ids: Vec<i32> = vec![real_token; d + 1]; // values are timing-neutral
        let mut time_pair = || {
            let _ = ds.draft_n(mh_carry, real_token, carry_pos, d)?;
            sync(m, ds);
            let _ = m.forward_verify_capture_main(&ids, carry_pos + 1)?;
            sync(m, ds);
            Ok::<_, anyhow::Error>(())
        };
        let step = time_it(&format!("d={} (draft+verify)", d), 3, &mut time_pair)?;
        let r = (step / decode.max(1e-9)) as f32;
        eprintln!("[dspark-depth] calib r({}) = {:.2}x decode", d, r);
        table.push((d, r));
    }
    m.restore_verify_state(&snap)?;
    sync(m, ds);
    Ok(table)
}
