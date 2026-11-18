//! The `--probe-exl3-xtp` harness pieces: XtpOpts, the traced / timed passes and the G-T1-c (verify column 0 ==
//! decode) / G-T1-d (GDN state after an N-wide verify) gates, plus `verify_teacher`. See `mod.rs` for the map.

use super::*;

pub(super) fn fnv64(v: &[f32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for x in v {
        for b in x.to_bits().to_le_bytes() { h ^= b as u64; h = h.wrapping_mul(0x100_0000_01b3); }
    }
    h
}

pub(super) fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) { let d = *x as f64 - *y as f64; num += d * d; den += (*y as f64) * (*y as f64); }
    (num / den.max(1e-300)).sqrt()
}

pub(super) fn median(v: &mut Vec<f64>) -> f64 {
    if v.is_empty() { return f64::NAN; }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

// ---------------------------------------------------------------------------------------------
// TP-B: the xtp program options (ride TpConfig.exl3_mode: "xtp[:pf=N][:gates=N][:ident][:time=N]")
// ---------------------------------------------------------------------------------------------

/// What the xtp program runs. `pf_chunk` = prefill the prompt (all but its last token) through
/// the SERVED chunked prefill (prefill_chunk, chunk N) instead of token-by-token forward_steps;
/// `gates` = N > 0 runs G-T1-c (verify column 0 == decode) and G-T1-d (GDN state after an N-wide
/// verify == N sequential steps) for widths 1..=N; `ident_only` = no TP=1 reference (rank
/// identity + greedy agreement only, e.g. the 8K prompt); `time_tokens` = an untraced greedy
/// timing pass of that many tokens (rough eager ms/token — not a perf claim).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct XtpOpts {
    pub pf_chunk: Option<usize>,
    pub gates: usize,
    pub ident_only: bool,
    pub time_tokens: usize,
    /// TP-SP2 (`--xtp-sp-fault L[:R]`): the G-T1-b fault injection — in the sequence-parallel run of the
    /// FIRST split chunk, rank R (default 1) flips one ulp of its first own stream row after layer L.
    /// The redesigned gate must report RANK_IDENTITY_FAIL (the proof that it catches a one-rank
    /// divergence of an owned row). A probe flag only; never on a served path.
    pub sp_fault: Option<(usize, usize)>,
}

impl XtpOpts {
    pub fn parse(mode: &str) -> Result<Self> {
        let mut it = mode.split(':');
        anyhow::ensure!(it.next() == Some("xtp"), "EXL3 TP: unknown program '{mode}'");
        let mut o = XtpOpts::default();
        for kv in it {
            let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
            match k {
                "pf" => o.pf_chunk = Some(v.parse().context("xtp pf=N")?),
                "gates" => o.gates = v.parse().context("xtp gates=N")?,
                "ident" => o.ident_only = true,
                "time" => o.time_tokens = v.parse().context("xtp time=N")?,
                "spfault" => {
                    let (a, b) = v.split_once('.').unwrap_or((v, "1"));
                    o.sp_fault = Some((a.parse().context("xtp spfault=L.R")?, b.parse().context("xtp spfault=L.R")?));
                }
                _ => anyhow::bail!("EXL3 TP: unknown xtp option '{kv}' in '{mode}'"),
            }
        }
        Ok(o)
    }
    pub fn to_mode(&self) -> String {
        let mut s = "xtp".to_string();
        if let Some(c) = self.pf_chunk { s += &format!(":pf={c}"); }
        if self.gates > 0 { s += &format!(":gates={}", self.gates); }
        if self.ident_only { s += ":ident"; }
        if self.time_tokens > 0 { s += &format!(":time={}", self.time_tokens); }
        if let Some((l, r)) = self.sp_fault { s += &format!(":spfault={l}.{r}"); }
        s
    }
}

/// TP-SP2: the armed fault injection of this xtp run ((layer + 1) | rank << 16; 0 = none), consumed by
/// the first split chunk's sequence-parallel run of the digest pass.
pub(super) static SP_FAULT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// One traced pass's results. Steps are indexed from `t0` (0 = token-by-token from position 0;
/// plen-1 = the prompt's first plen-1 tokens went through the chunked prefill).
pub(super) struct Pass {
    pub(super) seq: Vec<u32>,
    pub(super) am: Vec<u32>,
    pub(super) lg: Vec<Vec<f32>>,
    pub(super) at_cmp: Vec<(String, Vec<f32>)>,
    pub(super) t0: usize,
    pub(super) pf_s: f64,
}

pub(super) type Caps = [(String, Vec<f32>)];

/// Zero every QSA key/pool plane (the slot's positional side planes) + reset slot 0.
fn fresh_slot(model: &FwdModel) -> Result<()> {
    let l = Launcher { dev: &model.dev, stream: &model.stream };
    for k in model.qsa_keys.iter().chain(model.qsa_pool.iter()) {
        let n = k.len() as i64;
        xqlaunch!(l, "xq_memset_u16", (((n + 255) / 256) as u32, 1, 1), (256, 1, 1), 0, (*k.device_ptr() as u64, n, 0i32))?;
    }
    model.dev.synchronize()?;
    model.reset_slot(0)
}

/// Q1's floor: flip mantissa bit k in ~half of every GDN S element (this rank's state).
fn perturb_state(model: &FwdModel, k: u32, seed: u64, t: usize) -> Result<()> {
    let l = Launcher { dev: &model.dev, stream: &model.stream };
    let per_s = model.tc.lin_num_v_heads * model.tc.lin_k_dim * model.tc.lin_v_dim;
    for (li, ss) in model.s_state.iter().enumerate() {
        let mut hh: Vec<f32> = model.dev.dtoh_sync_copy(ss)?;
        for (j, x) in hh[..per_s].iter_mut().enumerate() {
            let mut z = (j as u64) ^ ((t as u64) << 40) ^ ((li as u64) << 52) ^ seed.wrapping_mul(0xD1B5_4A32_D192_ED03);
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let b = x.to_bits();
            if z & 1 == 1 && (b & 0x7F80_0000) != 0 { *x = f32::from_bits(b ^ (1u32 << k)); }
        }
        let tmp = model.dev.htod_copy(hh[..per_s].to_vec())?;
        let dst = *ss.device_ptr() as u64;
        xqlaunch!(l, "xq_copy_f32", (((per_s + 255) / 256) as u32, 1, 1), (256, 1, 1), 0,
                  (dst, &tmp, per_s as i64, 0i64, 0i64))?;
        model.dev.synchronize()?;
    }
    Ok(())
}

/// Prefill prompt[..n] of slot 0 through the served chunked prefill (chunk = psc.c). With
/// `digest`, every chunk's replicated buffers are captured as whole-buffer digests and handed to
/// `on_pf(chunk index, digests)`. Returns the wall seconds of the prefill (incl. captures if on).
///
/// TP-SP2 (the G-T1-b redesign for row ownership): with `digest` and a chunk that runs SEQUENCE-
/// PARALLEL, the chunk runs TWICE on the same slot state: first REPLICATED (the flag cleared on both
/// ranks for this call: the reference class — every buffer whole on both ranks, so the replicated
/// G-T1-b holds on it), its whole-buffer digests + every rank block's digests of the own-row classes
/// captured as `R:<name>`; then the recurrent state is restored from a WP16 checkpoint taken before it
/// (every positional row is rewritten by the second run) and the chunk runs sequence-parallel with
/// the own block's digests. Both runs take the last row's logits (the final-mix gather path) and
/// digest the gathered final streams + the head-fill carry. on_pf gets both runs' entries.
fn prefill_prompt_chunks(model: &FwdModel, psc: &mut PrefillScratch, prompt: &[u32], digest: bool,
                         on_pf: &mut dyn FnMut(usize, &Caps) -> Result<()>) -> Result<f64> {
    let nl = model.cfg.layer_types.len();
    let c = psc.c;
    let toks: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let t0 = std::time::Instant::now();
    let (mut pos, mut k) = (0usize, 0usize);
    let mut ck: Option<super::wp16::RecurCkpt> = None;
    while pos < toks.len() {
        let cc = c.min(toks.len() - pos);
        let split = if digest { model.sp_rows_for(psc, cc) } else { None };
        if let (Some(sp), Some(tp)) = (split, model.tp.as_ref()) {
            let world = tp.world as usize;
            let blocks: Vec<(usize, usize)> = (0..world).map(|r| crate::tp_xport::row_block(cc, r, world)).collect();
            let own = tp.rank as usize;
            anyhow::ensure!(blocks[own] == sp.own, "TP-SP2 G-T1-b: own block {:?} vs the chunk's {:?}", blocks[own], sp.own);
            let mk = |only: Option<usize>, fault_layer: Option<usize>| LadderCtx {
                dir: String::new(), step: 0, layers: (0..nl).collect(), only_step: None, mem: Some(Mutex::new(Vec::new())),
                digest: true, sp: Some(SpDigest { cap: c, c: cc, blocks: blocks.clone(), only, fault_layer }) };
            if ck.is_none() { ck = Some(model.ckpt_alloc()?); }
            let b = ck.as_mut().unwrap();
            model.ckpt_save(0, psc, b)?;
            // R: the replicated reference run of this chunk (SPMD: both ranks clear the flag here)
            let code = seq_parallel_code();
            set_seq_parallel(0);
            let ld_r = mk(None, None);
            let rr = model.prefill_chunk(psc, &toks[pos..pos + cc], pos, 0, true, Some(&ld_r));
            set_seq_parallel(code);
            rr?;
            model.ckpt_restore(0, psc, ck.as_ref().unwrap())?;
            // S: the sequence-parallel run (the fault, if armed for this rank, is consumed here)
            let f = SP_FAULT.swap(0, AtOrd::Relaxed);
            let fault = (f != 0 && (f >> 16) == own).then(|| (f & 0xFFFF) - 1);
            if f != 0 && fault.is_none() { SP_FAULT.store(0, AtOrd::Relaxed); }
            let ld_s = mk(Some(own), fault);
            model.prefill_chunk(psc, &toks[pos..pos + cc], pos, 0, true, Some(&ld_s))?;
            let mut ent: Vec<(String, Vec<f32>)> = ld_r.mem.expect("mem capture").into_inner().unwrap()
                .into_iter().map(|(n, v)| (format!("R:{n}"), v)).collect();
            ent.extend(ld_s.mem.expect("mem capture").into_inner().unwrap());
            on_pf(k, &ent)?;
        } else if digest {
            let ld = LadderCtx { dir: String::new(), step: 0, layers: (0..nl).collect(), only_step: None,
                                 mem: Some(Mutex::new(Vec::new())), digest: true, sp: None };
            model.prefill_chunk(psc, &toks[pos..pos + cc], pos, 0, false, Some(&ld))?;
            let ent = ld.mem.expect("mem capture").into_inner().unwrap();
            on_pf(k, &ent)?;
        } else {
            model.prefill_chunk(psc, &toks[pos..pos + cc], pos, 0, false, None)?;
        }
        pos += cc;
        k += 1;
    }
    model.dev.synchronize()?;
    Ok(t0.elapsed().as_secs_f64())
}

/// One traced pass. `forced` = teacher forcing (the reference sequence) — else greedy from the
/// argmax. `perturb` = the Q1 floor (flip mantissa bit k in ~half of every GDN S element after
/// every step but the last; after the chunked prefill too). `psc` = prefill prompt[..plen-1]
/// through the served chunked prefill first (TP-B), digesting its replicated buffers into
/// `on_pf` when `pf_digest`. `on_step(step, entries, argmax)` sees every decode step's captured
/// replicated buffers (the rank identity hook).
#[allow(clippy::too_many_arguments)]
pub(super) fn trace_pass(model: &FwdModel, sc: &mut Scratch, psc: Option<&mut PrefillScratch>, pf_digest: bool,
              prompt: &[u32], forced: Option<&[u32]>, gen: usize, cmp: usize,
              perturb: Option<(u32, u64)>, keep_logits: bool,
              on_pf: &mut dyn FnMut(usize, &Caps) -> Result<()>,
              on_step: &mut dyn FnMut(usize, &Caps, u32) -> Result<()>) -> Result<Pass> {
    let nl = model.cfg.layer_types.len();
    fresh_slot(model)?;
    let plen = prompt.len();
    let total = plen + gen - 1;
    let mut seq: Vec<u32> = prompt.to_vec();
    let (mut am_v, mut lg_v, mut at_cmp) = (Vec::new(), Vec::new(), Vec::new());
    let mut pf_s = 0.0;
    let t0 = match psc {
        Some(p) if plen >= 2 => {
            pf_s = prefill_prompt_chunks(model, p, &prompt[..plen - 1], pf_digest, on_pf)?;
            if let Some((k, seed)) = perturb {
                if k <= 22 { perturb_state(model, k, seed, plen - 2)?; }
            }
            plen - 1
        }
        _ => 0,
    };
    for t in t0..total {
        let mut ld = LadderCtx { dir: String::new(), step: 0, layers: (0..nl).collect(), only_step: None,
                                 mem: Some(Mutex::new(Vec::new())), digest: false, sp: None };
        let ids = model.forward_step(sc, &[seq[t] as i32], &[t], &[0], Some(&mut ld))?;
        let am = ids[0] as u32;
        let ent = ld.mem.take().unwrap().into_inner().unwrap();
        on_step(t, &ent, am)?;
        if keep_logits {
            lg_v.push(ent.iter().find(|(n, _)| n == "logits").map(|(_, v)| v.clone()).context("no logits capture")?);
        }
        if t == cmp { at_cmp = ent.into_iter().filter(|(n, _)| n != "logits").collect(); }
        am_v.push(am);
        if t + 1 >= seq.len() {
            seq.push(match forced { Some(f) => f[t + 1], None => am });
        }
        if let Some((k, seed)) = perturb {
            if t + 1 < total && k <= 22 { perturb_state(model, k, seed, t)?; }
        }
    }
    Ok(Pass { seq, am: am_v, lg: lg_v, at_cmp, t0, pf_s })
}

/// Untraced greedy pass (no ladder: the served step path; eager under TP): prefill + `n` decode
/// tokens. Returns (prefill seconds, decode ms/token over tokens 2..n, the generated ids).
pub(super) fn time_pass(model: &FwdModel, sc: &mut Scratch, psc: Option<&mut PrefillScratch>, prompt: &[u32], n: usize,
             on_tok: &mut dyn FnMut(usize, u32) -> Result<()>) -> Result<(f64, f64, Vec<u32>)> {
    fresh_slot(model)?;
    let plen = prompt.len();
    let t_pf = std::time::Instant::now();
    let t0 = match psc {
        Some(p) if plen >= 2 => { prefill_prompt_chunks(model, p, &prompt[..plen - 1], false, &mut |_, _| Ok(()))?; plen - 1 }
        _ => 0,
    };
    let mut cur = 0u32;
    for t in t0..plen {
        cur = model.forward_step(sc, &[prompt[t] as i32], &[t], &[0], None)?[0] as u32;
    }
    let pf_s = t_pf.elapsed().as_secs_f64();
    let mut out = vec![cur];
    on_tok(0, cur)?;
    let mut t1 = std::time::Instant::now();
    for i in 1..n.max(1) {
        if i == 1 { t1 = std::time::Instant::now(); }
        cur = model.forward_step(sc, &[cur as i32], &[plen + i - 1], &[0], None)?[0] as u32;
        on_tok(i, cur)?;
        out.push(cur);
    }
    let ms = if n > 1 { t1.elapsed().as_secs_f64() * 1e3 / (n - 1) as f64 } else { f64::NAN };
    Ok((pf_s, ms, out))
}

impl FwdModel {
    /// TP-B (G-T1-c/d): a TEACHER-forced verify of rows (b, drafts...) at positions p.. of `slot`
    /// — the served verify block (shadow -> kernels -> taps -> accept -> device-keyed commit),
    /// EAGER, with the drafts uploaded instead of drafted by the head. Mirrors mtp_verify's
    /// staging (non-WP05 branch) and its PLE host-ring restore. Returns (accepted a, emitted).
    pub(in crate::exl3_forward) fn verify_teacher(&self, sc: &mut Scratch, slot: usize, p: usize, b: i32, drafts: &[i32])
                                 -> Result<(usize, Vec<i32>)> {
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        let k = drafts.len();
        let m = k + 1;
        anyhow::ensure!(k >= 1 && m <= MTP_MAX_K + 1 && sc.toks.len() >= m && sc.slots.len() >= 2 * m
                        && p + k < self.max_pos, "verify_teacher: width {m} / position {p} out of range");
        let mut toks_v = vec![b];
        toks_v.extend_from_slice(drafts);
        let mut toks_w = toks_v.clone();
        toks_w.resize(sc.toks.len(), 0);
        self.dev.htod_copy_into(toks_w, &mut sc.toks)?;
        let mut pos_v: Vec<i32> = (p..p + m).map(|x| x as i32).collect();
        pos_v.resize(sc.pos.len(), 0);
        self.dev.htod_copy_into(pos_v, &mut sc.pos)?;
        let mut slotpos: Vec<i32> = vec![0; sc.slots.len()];
        for r in 0..m {
            slotpos[r * 2] = slot as i32;
            slotpos[r * 2 + 1] = (p + r) as i32;
        }
        self.dev.htod_copy_into(slotpos, &mut sc.slots)?;
        self.dev.htod_copy_into(vec![slot as i32; sc.slot_ids.len()], &mut sc.slot_ids)?;
        let hist_pre = {
            let mut emb_bits = vec![0u16; sc.ple_emb.len()];
            let hp = { let hist = self.ple_hist.lock().unwrap(); (hist[slot * 2], hist[slot * 2 + 1]) };
            self.ple_build_embed(m, &toks_v, &vec![slot; m], &mut emb_bits)?;
            self.dev.htod_copy_into(emb_bits, &mut sc.ple_emb)?;
            hp
        };
        let mut d = vec![0i32; sc.d_dev.len()];
        d[..k].copy_from_slice(drafts);
        self.dev.htod_copy_into(d, &mut sc.d_dev)?;
        let qsa = self.qsa_live(p + k);
        let (qg, _) = self.qsa_grid_bucket_key(p + k);
        sc.qsa_grid_nblk = qg;
        self.verify_block(&l, sc, slot, m, k, qsa)?;
        self.dsync()?;
        let out = self.dtoh(&sc.accept_out)?;
        let a = out[k + 2] as usize;
        anyhow::ensure!(a <= k, "verify_teacher: accept out of range");
        {
            let mut hist = self.ple_hist.lock().unwrap();
            if a == 0 { hist[slot * 2] = hist_pre.1; hist[slot * 2 + 1] = b; }
            else { hist[slot * 2] = toks_v[a - 1]; hist[slot * 2 + 1] = toks_v[a]; }
        }
        Ok((a, out[..=a].to_vec()))
    }
}

/// TP-B G-T1-c + G-T1-d on THIS rank (SPMD; `sync(tag, value)` exchanges one word with the peer
/// and returns the peer's (TP=1: returns `value`) — a mismatch is a rank-identity failure). Base state = the prompt consumed (chunked prefill of all but its last token
/// when `psc`, then one step), b = its argmax. Reference = `nmax` sequential decode steps (logits
/// L_i, argmax e_i, this rank's GDN/conv state S_i after i steps). For N = 2..=nmax: restore the
/// base, verify (b, e_0..e_{N-2}) teacher-forced (all accepted), then
///   G-T1-c  verify row 0's logits == L_0 bitwise (every row i vs L_i also reported);
///   G-T1-d  the committed state == S_N bitwise (max |diff| must be 0.0).
/// N = 1 is the decode itself (re-run from the base: determinism). Returns (all ok, report).
pub(super) fn gates_bv_state(model: &FwdModel, psc: Option<&mut PrefillScratch>, prompt: &[u32], nmax: usize,
                  sync: &mut dyn FnMut(u64, u32) -> Result<u32>) -> Result<(bool, Vec<String>)> {
    // Graphs are keyed by width, not by scratch: a step graph captured earlier on another Scratch
    // (the timing pass) would replay against THAT scratch's buffers. The gates run eager (TP=2 is
    // eager anyway; at TP=1 this is the eager control).
    let prev_ng = crate::opts::var(crate::opt!("exl3-no-graph")).ok();
    crate::opts::set(crate::opt!("exl3-no-graph"), "1");
    let r = gates_inner(model, psc, prompt, nmax, sync);
    GATE_VP_OFF.store(false, AtOrd::Relaxed);
    GATE_VP_GATHER.store(false, AtOrd::Relaxed);
    match prev_ng { Some(v) => crate::opts::set(crate::opt!("exl3-no-graph"), v), None => crate::opts::unset(crate::opt!("exl3-no-graph")) }
    r
}

fn gates_inner(model: &FwdModel, psc: Option<&mut PrefillScratch>, prompt: &[u32], nmax: usize,
               sync: &mut dyn FnMut(u64, u32) -> Result<u32>) -> Result<(bool, Vec<String>)> {
    let v = model.cfg.vocab_size;
    let w = nmax.max(2);
    let mut sc = Scratch::new(&model.dev, &model.cfg, w, model.cfg.rotary_dim, model.max_pos)?;
    let plen = prompt.len();
    anyhow::ensure!(plen >= 2 && nmax >= 1 && nmax <= MTP_MAX_K + 1, "gates: prompt >= 2 tokens, 1 <= N <= {}", MTP_MAX_K + 1);
    let mut psc = psc;
    let chunked = psc.is_some();
    let mut base = |model: &FwdModel, sc: &mut Scratch| -> Result<i32> {
        fresh_slot(model)?;
        let t0 = match psc.as_deref_mut() {
            Some(p) => { prefill_prompt_chunks(model, p, &prompt[..plen - 1], false, &mut |_, _| Ok(()))?; plen - 1 }
            None => 0,
        };
        let mut cur = 0i32;
        for t in t0..plen { cur = model.forward_step(sc, &[prompt[t] as i32], &[t], &[0], None)?[0]; }
        Ok(cur)
    };
    let row = |lg: &[u16], i: usize| lg[i * v..(i + 1) * v].to_vec();
    // ---- sequential reference. TP-H: on the REPLICATED head (full-V rows), so the vocab-parallel
    // tail below is compared against ground truth of a different class, not against itself.
    GATE_VP_OFF.store(true, AtOrd::Relaxed);
    let b = base(model, &mut sc)?;
    let mut e: Vec<i32> = Vec::new();
    let mut lref: Vec<Vec<u16>> = Vec::new();
    let mut sref: Vec<(Vec<Vec<u32>>, Vec<Vec<u32>>)> = Vec::new();
    let mut cur = b;
    for i in 0..nmax {
        let id = model.forward_step(&mut sc, &[cur], &[plen + i], &[0], None)?[0];
        lref.push(row(&model.logits_host(&sc)?, 0));
        sref.push(snap_slot0_state(model)?);
        e.push(id);
        cur = id;
    }
    GATE_VP_OFF.store(false, AtOrd::Relaxed);
    let eh = e.iter().fold(0u32, |h, &x| h.wrapping_mul(31).wrapping_add(x as u32));
    let mut ok = true;
    let mut rep = Vec::new();
    if sync(0xB0, eh)? != eh { ok = false; rep.push("GATES rank identity: the sequential greedy reference DIFFERS from the peer's".into()); }
    rep.push(format!("GATES base: plen {plen}{} b {b} greedy e[0..{nmax}] {:?}",
                     if chunked { " (chunked prefill)" } else { " (token-by-token)" }, e));
    // ---- N = 1: the decode re-run from the base (determinism of the reference itself)
    {
        let b2 = base(model, &mut sc)?;
        let id = model.forward_step(&mut sc, &[b2], &[plen], &[0], None)?[0];
        // TP-H: the re-run takes the SERVED tail (vp shard when live): compare its shard slice with
        // the replicated reference's same columns (the merged id is checked through `id == e[0]`)
        let lg1 = model.logits_host(&sc)?;
        let (md, neq) = state_diff(&sref[0], &snap_slot0_state(model)?);
        let vp1 = model.vp_greedy(&sc, 1);
        let lneq = if vp1 {
            let vp = model.lm_head_vp.as_ref().expect("vp_greedy implies the shard");
            let (bs, ns) = (vp.base as usize, vp.n_sh as usize);
            lg1[..ns].iter().zip(&lref[0][bs..bs + ns]).filter(|(a, c)| a != c).count()
        } else {
            row(&lg1, 0).iter().zip(&lref[0]).filter(|(a, c)| a != c).count()
        };
        let pass = b2 == b && id == e[0] && lneq == 0 && neq == 0;
        ok &= pass;
        rep.push(format!("G-T1-c N=1 (decode re-run{}): logits row0 bit-neq {lneq}/{} argmax {} | G-T1-d state max|diff| {md:.1} bit-neq {neq} -> {}",
                         if vp1 { ", vp shard tail" } else { "" },
                         if vp1 { model.lm_head_vp.as_ref().map_or(v, |q| q.n_sh as usize) } else { v },
                         if id == e[0] { "==" } else { "!=" }, if pass { "EXACT" } else { "DIVERGES" }));
        let _ = sync(0xC1, pass as u32)?;
    }
    for n in 2..=nmax {
        let b2 = base(model, &mut sc)?;
        anyhow::ensure!(b2 == b, "gates: base argmax moved between runs ({b2} vs {b}) — nondeterministic base");
        let drafts: Vec<i32> = e[..n - 1].to_vec();
        let (a, emitted) = model.verify_teacher(&mut sc, 0, plen, b, &drafts)?;
        let lg = model.logits_host(&sc)?;
        let am = model.dev.dtoh_sync_copy(&sc.argmax)?;
        // TP-H #4: verify_teacher takes the vocab-parallel greedy tail, so sc.logits holds this
        // rank's SHARD rows (stride n_sh, global column = base + j), not full-V rows. Compare the
        // shard slice bit-exactly against the same columns of the decode reference (the replicated
        // head), and check the merged argmax (the tail's real output) against decode's.
        let vpm = model.vp_greedy(&sc, n);
        let neq_row = |i: usize| -> usize {
            if vpm {
                let vp = model.lm_head_vp.as_ref().expect("vp_greedy implies the shard");
                let (bs, ns) = (vp.base as usize, vp.n_sh as usize);
                lg[i * ns..(i + 1) * ns].iter().zip(&lref[i][bs..bs + ns]).filter(|(x, y)| x != y).count()
            } else {
                row(&lg, i).iter().zip(&lref[i]).filter(|(x, y)| x != y).count()
            }
        };
        let r0neq = neq_row(0);
        let rows_neq: Vec<usize> = (0..n).map(|i| neq_row(i)).collect();
        let cmp_cols = if vpm { model.lm_head_vp.as_ref().map(|q| q.n_sh as usize).unwrap_or(v) } else { v };
        let am_ok = (0..n).all(|i| am[i] == e[i]);
        let (md, neq) = state_diff(&sref[n - 1], &snap_slot0_state(model)?);
        let pass_c = r0neq == 0 && am_ok;
        let pass_d = a == n - 1 && md == 0.0 && neq == 0;
        ok &= pass_c && pass_d;
        rep.push(format!("G-T1-c N={n}: verify col0 logits bit-neq {r0neq}/{cmp_cols}{} (rows {:?}) argmax rows {} -> {} | \
                          G-T1-d accepted {a}/{} emitted {:?} state max|diff| {md:.1} bit-neq {neq} -> {}",
                         if vpm { " [vp shard slice vs decode columns; merged argmax vs decode]" } else { "" },
                         rows_neq, if am_ok { "== decode" } else { "!= decode" }, if pass_c { "EXACT" } else { "DIVERGES" },
                         n - 1, emitted, if pass_d { "EXACT" } else { "DIVERGES" }));
        // rank identity of the verify's replicated output (G-T1-b on the verify path)
        // (under the vp tail the logits are per-rank shards by design — the replicated output is
        // the merged argmax ids, so those are what must be identical across ranks)
        let d = if vpm {
            crate::exl3_forward::digest_words(am[..n].iter().map(|&x| x as u32 as u64))
        } else {
            crate::exl3_forward::digest_words(lg[..n * v].iter().map(|&x| x as u64))
        };
        let dm = (d as u32) ^ ((d >> 32) as u32);
        if sync(0xC0 + n as u64, dm)? != dm {
            ok = false;
            rep.push(format!("G-T1-b verify N={n}: the verify {} DIFFER from the peer's (rank identity FAIL)",
                             if vpm { "merged argmax ids" } else { "logits rows" }));
        }
    }
    // ---- TP-H2 G-T1-v: the row all-gather tail (the --tp-vp-sampled path) forced on EVERY row. Its
    // output is the FULL logits rows, so every row is compared bit for bit against the replicated-head
    // reference (lref: a different class — the replicated lm_head GEMM), plus the ids, the committed state
    // and the rows' rank identity. N = 1 = the plain-step tail (forward_step), N >= 2 = the verify tail.
    // TP-4H2: the section runs at world 2 (one exchange per group) AND world 4 (recursive doubling, two per
    // group of <= 4 rows; N = 1..nmax <= 8 covers groups 1-2 incl. the 4 -> 5 change; m = 9..16 is the
    // `--tp-reduce-bench --mode decwvpg` arm's). At any other world > 2 it is reported SKIPPED (never a silent
    // omission: the report line names why, and the sampled rows' replicated head is what serves)
    let vp_world = model.tp_rank().map_or(0, |(_, w)| w);
    if model.lm_head_vp.is_some() && vp_world > 2 && vp_world != 4 {
        rep.push(format!("G-T1-v: SKIPPED at world {vp_world} — the TP-H2 row all-gather has kernels at world 2 and 4 only; \
                          sampled / penalized / ratio-rule rows take the replicated lm_head"));
    }
    if model.lm_head_vp.is_some() && (vp_world == 2 || vp_world == 4) {
        GATE_VP_GATHER.store(true, AtOrd::Relaxed); // cleared by gates_bv_state on every exit
        {
            for n in 1..=nmax {
                let b2 = base(model, &mut sc)?;
                anyhow::ensure!(b2 == b, "gates: base argmax moved between runs ({b2} vs {b})");
                let (a, ids) = if n == 1 {
                    anyhow::ensure!(model.vp_gather(&sc, 1), "G-T1-v: the gather tail is not live at m = 1");
                    let id = model.forward_step(&mut sc, &[b], &[plen], &[0], None)?[0];
                    (0usize, vec![id])
                } else {
                    anyhow::ensure!(model.vp_gather(&sc, n), "G-T1-v: the gather tail is not live at m = {n}");
                    let drafts: Vec<i32> = e[..n - 1].to_vec();
                    let (a, _) = model.verify_teacher(&mut sc, 0, plen, b, &drafts)?;
                    (a, model.dev.dtoh_sync_copy(&sc.argmax)?[..n].to_vec())
                };
                let lg = model.logits_host(&sc)?;
                let rows_neq: Vec<usize> = (0..n).map(|i| row(&lg, i).iter().zip(&lref[i]).filter(|(x, y)| x != y).count()).collect();
                let ids_ok = (0..n).all(|i| ids[i] == e[i]);
                let (md, neq) = state_diff(&sref[n - 1], &snap_slot0_state(model)?);
                let pass = rows_neq.iter().all(|&x| x == 0) && ids_ok && a == n - 1 && md == 0.0 && neq == 0;
                ok &= pass;
                let d = crate::exl3_forward::digest_words(lg[..n * v].iter().map(|&x| x as u64));
                let dm = (d as u32) ^ ((d >> 32) as u32);
                let same = sync(0xD0 + n as u64, dm)? == dm;
                ok &= same;
                rep.push(format!("G-T1-v N={n} (vp row all-gather tail, {}{}): full logits rows bit-neq {:?}/{v} vs the replicated head, \
                                  ids {} decode, accepted {a}/{}, state max|diff| {md:.1} bit-neq {neq}, rows rank-identical {} -> {}",
                                 if vp_world == 2 { String::new() } else { format!("world {vp_world}, ") },
                                 if n == 1 { "plain step" } else { "verify" }, rows_neq, if ids_ok { "==" } else { "!=" },
                                 n - 1, if same { "yes" } else { "NO" }, if pass && same { "EXACT" } else { "DIVERGES" }));
            }
        }
        GATE_VP_GATHER.store(false, AtOrd::Relaxed);
    }
    Ok((ok, rep))
}
