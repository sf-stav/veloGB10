//! WP16 (PLAN/SURPASS_PLAN_2026-09-26.md §WP16) — intermediate recurrent checkpoints.
//!
//! The prefix cache keeps ONE prompt-end snapshot per slot, so "same document, new question"
//! (or a new conversation sharing a long system prompt) re-prefilled everything: 148.8 s at 128K.
//! WP16 keeps extra per-slot checkpoints of the RECURRENT state — the prefix snapshot's own
//! contents (GDN S + conv of every GDN layer, the PLE f16 conv ring row, the host PLE n-gram pair,
//! the MTP head-fill carry) — at chunk ends of the served chunked prefill. Everything positional
//! (trunk KV rows incl. q8 scales, QSA raw/pooled key planes, the draft head's KV + QSA keys) is
//! already in the slot for every position below the checkpoint and is reused as the prefix cache
//! reuses it.
//!
//! BITWISE contract: a resumed prefill runs the SAME chunk grid as a fresh prefill from 0, so it
//! reproduces the fresh state, output bytes and MTP trajectory exactly:
//! - checkpoints are taken only at chunk ends of ALIGNED runs (chunk grid starting at 0, or at an
//!   aligned checkpoint — never after an unaligned prompt-end prefix-cache resume, which is a
//!   different grid and a different fp result class), at `end % 8192 == 0` plus every C boundary
//!   within the run's last 2 chunks (`wp16_due`);
//! - they are keyed by the chunk width C (a checkpoint taken at C is never resumed at another C);
//! - a resume takes the deepest checkpoint <= min(LCP(prompt, the checkpointed prefix), plen-1),
//!   skips reset_slot and drops every checkpoint above the resume point (those rows get rewritten).
//!
//! Escapes (diagnostic only, AGENTS §1b): --wp16-off=1 = today's behaviour (no checkpoint is
//! taken or resumed); --prefix-ckpt-mem-gb 0 does the same. --wp16-xcheck=1: every checkpoint
//! resume also runs the fresh full prefill into a scratch slot and bit-compares GDN/conv/PLE state
//! + carry + the first-token logits (exl3_serve.rs). `--probe-exl3-wp16` is the single-process
//! gate of the same property (probe_wp16).

use super::*;

/// Chunk ends at multiples of this (counted from position 0) are always checkpointed.
pub const WP16_STRIDE: usize = 8192;

/// Is chunk end `e` of an ALIGNED prefill run (chunk width `c`, prefill end `n` = plen-1) a
/// checkpoint position? `e % 8192 == 0`, or `e` bounds one of the run's last 2 chunks. `e` must
/// be a C multiple (a partial final chunk's end is not a grid point of any longer prompt).
pub fn wp16_due(e: usize, n: usize, c: usize) -> bool {
    c > 0 && e > 0 && e <= n && e % c == 0 && (stride_hit(e, c) || e + 2 * c >= n)
}

/// Does the stride checkpoint rule pick chunk end `e` (a C multiple, e >= c)? c <= 2048: `e % 8192
/// == 0`, exactly today's rule. c > 2048 (TP-4X1, `--prefill-chunk 4095`): C multiples almost never
/// hit a multiple of 8192, so the rule is "the chunk [e-c, e) crosses one" - a checkpoint about
/// every 8192 rows, the same density and memory as C=2048 (identical to `e % 8192 == 0` whenever C
/// divides 8192, e.g. 4096).
fn stride_hit(e: usize, c: usize) -> bool {
    if c > 2048 { e / WP16_STRIDE > (e - c) / WP16_STRIDE } else { e % WP16_STRIDE == 0 }
}

fn env_flag(o: crate::opts::OptId) -> bool {
    crate::opts::var(o).is_ok_and(|v| !v.is_empty() && v != "0")
}

/// --wp16-off=1: diagnostic escape — today's behaviour (prompt-end snapshot only).
pub fn wp16_off() -> bool { env_flag(crate::opt!("wp16-off")) }

/// --wp16-xcheck=1: on every checkpoint resume, also prefill fresh into a scratch slot and
/// bit-compare the state and the first-token logits (diagnostic; costs a full prefill).
pub fn wp16_xcheck_on() -> bool { env_flag(crate::opt!("wp16-xcheck")) }

/// C1 (A5 session C1): TAIL checkpoints. A multi-turn follow-up diverges from the previous prompt
/// right at its generation prompt (the next turn re-renders everything up to the message boundary
/// verbatim and replaces `<|im_start|>assistant\n<think>\n` with the answer), so the deepest
/// useful resume point is the previous prompt's MESSAGE BOUNDARY `t` (server.rs `ckpt_at`: the
/// token count of the prompt without its generation prompt) — never a 2,048-aligned WP16 point.
/// With the knob on, a prefill run (a) splits its chunk grid at `t` and checkpoints there, and
/// (b) realigns a run that starts off the C grid (a tail resume, a prompt-end prefix hit) to the
/// next C multiple, so its later chunk ends are C multiples again and the WP16 rule keeps working.
///
/// OUTPUT-CHANGING (class N, owner consent): the split moves a chunk boundary of the prompt that
/// takes the checkpoint, and a tail resume runs a chunk grid a fresh prefill never runs — both are
/// reassociation-class changes (this model amplifies rounding chaotically, Q1_prefillx.md), so the
/// bytes of those requests differ from knob-off. Knob off = today's grid and checkpoints exactly.
/// A split is taken only when it saves at least TAIL_MIN_GAIN tokens over the aligned checkpoint
/// below `t` (short prompts, e.g. the greedy gate prompts, keep today's single-chunk grid).
pub(crate) const T_PREFIX_TAIL_CKPT: tune::TunableDef = tune::TunableDef {
    slot: 94, id: "prefix.tail_ckpt", class: tune::Class::N, scope: tune::Scope::Global, fams: &[tune::Fam::Util],
    domain: &[0, 1, 2], default: 1, valid: tune::valid_any,
    xcheck: "--probe-exl3-wp16 --wp16-probe-tail (N: output-changing, relL2 vs fresh; owner consent)",
    env: "--prefix-tail-ckpt",
    env_parse: || tune::env_str(crate::opt!("prefix-tail-ckpt")).map(|v| match v.as_str() { "1" => 1, "2" => 2, _ => 0 }),
    rev: 2, wp: "A5-C1",
};

/// C1: tail checkpoints on (registry prefix.tail_ckpt, alias --prefix-tail-ckpt=1|2).
/// 1 = only tail splits merge_grid absorbs (never an extra chunk: cold single-chunk prompts keep
///     today's grid and cost); 2 = also splits that cost one extra chunk (~180 ms) when they
///     promise >= TAIL_MIN_GAIN_SPLIT tokens (the v2 A/B configuration).
pub fn tail_ckpt_on() -> bool { tune::get_cur(&T_PREFIX_TAIL_CKPT) != 0 }
/// C1: knob value 2 (see tail_ckpt_on).
pub fn tail_ckpt_extra_chunk() -> bool { tune::get_cur(&T_PREFIX_TAIL_CKPT) == 2 }

/// C1: a tail split must save at least this many tokens over the aligned grid point below it.
pub const TAIL_MIN_GAIN: usize = 128;

/// C1: the checkpoint sits this many tokens before the message boundary. A follow-up that EXTENDS
/// the last user message (a growing document: the owner's depth sweep, LCP = plen - 7..9) diverges
/// at its `<|im_end|>\n` (2 tokens before the boundary) or a BPE merge a token or two earlier; a
/// new chat turn diverges after the boundary. 8 covers both at a cost of 8 recomputed tokens.
pub const TAIL_BACKOFF: usize = 8;

/// C1: the tail split point for a run over [from, n) at chunk width c, or None: t = boundary -
/// TAIL_BACKOFF strictly inside the run and at least TAIL_MIN_GAIN past both the run start and the
/// C multiple below it (what a resume without the tail would restart from).
pub fn tail_point(boundary: Option<usize>, from: usize, n: usize, c: usize, extra_chunk: bool) -> Option<usize> {
    let t = boundary?.checked_sub(TAIL_BACKOFF)?;
    if c == 0 || t <= from || t >= n { return None; }
    // TP-4X1: tail grain = min(C, 2048). At C <= 2048 this is C (unchanged); at C > 2048 (steady
    // 4,095-row chunks) it keeps today's tail-checkpoint availability (a 2-4K-row prompt still gets
    // its message-boundary checkpoint, at the price of the one small tail chunk it costs today)
    // instead of silently dropping it.
    let g = c.min(2048);
    let s = from.max(t / g * g);
    if s == from && !extra_chunk { return None; }
    // v2: when no C boundary lies between the run start and t, merge_grid cannot absorb the split
    // and the run pays one extra chunk (~120-190 ms of fixed weight streaming, v1 A/B): only worth
    // it when the follow-up saves >= TAIL_MIN_GAIN_SPLIT tokens.
    let need = if s == from { TAIL_MIN_GAIN_SPLIT } else { TAIL_MIN_GAIN };
    (t - s >= need).then_some(t)
}

/// C1 v2: the gain a tail split must promise when it costs an extra chunk (see tail_point).
pub const TAIL_MIN_GAIN_SPLIT: usize = 1024;

/// C1 v2: remove the extra chunk the tail split / realign add (v1 measured ~180 ms per extra chunk:
/// a small chunk still streams every weight). With the prefill scratch sized for 2C - 1 rows:
/// (1) the C boundary just before the tail merges into one chunk [(k-1)C, t) when <= 2C - 1 rows,
/// (2) a realigned lead chunk [from, next C multiple) merges with the chunk after it likewise.
/// Required boundaries (t, n) never move. Knob off never calls this.
pub fn merge_grid(from: usize, ends: Vec<usize>, tail: Option<usize>, c: usize) -> Vec<usize> {
    merge_grid_cap(from, ends, tail, c, 2 * c.max(1) - 1)
}

/// TP-4X1: the scratch / all-reduce partial buffer holds at most `TP_MAX_ROWS` = 4,096 rows
/// (`SP_MAX_ROWS` = 4,095 for a sequence-parallel chunk), so under TP a merged chunk is capped at
/// min(2C - 1, 4,095). Identical to 2C - 1 for every C <= 2,048 (every config legal before this
/// change); C in 2,049..=4,095 was refused at boot under TP while C1 tail checkpoints were on.
pub fn merge_cap(c: usize, tp: bool) -> usize {
    let cap = 2 * c.max(1) - 1;
    if tp { cap.min(super::xtp::TP_MAX_ROWS - 1) } else { cap }
}

/// merge_grid with an explicit row cap (see merge_cap); `merge_grid` is `cap = 2C - 1`.
pub fn merge_grid_cap(from: usize, ends: Vec<usize>, tail: Option<usize>, c: usize, cap: usize) -> Vec<usize> {
    let mut e = ends;
    let start = |e: &Vec<usize>, i: usize| if i == 0 { from } else { e[i - 1] };
    if let Some(t) = tail {
        if let Some(i) = e.iter().position(|&x| x == t) {
            // e[i-1] is the boundary before t: drop it if [start(i-1), t) fits
            if i >= 1 && Some(e[i - 1]) != tail && t - start(&e, i - 1) <= cap { e.remove(i - 1); }
        }
    }
    if from % c.max(1) != 0 && e.len() >= 2 && Some(e[0]) != tail && e[1] - from <= cap {
        e.remove(0);
    }
    e
}

/// TP-4X1 ragged-tail folding: ALWAYS on since v0.7.1 (owner 2026-10-02, D-TP4-6; no off switch).
/// `--prefill-absorb-tail` is still accepted so older command lines keep working.
pub fn absorb_on() -> bool { true }

/// The ragged-chunk threshold when absorb is on: chunks below `recon_min_rows()` (17) are the replicated
/// eager small-M class, each streaming every weight for a handful of rows.
pub fn absorb_rmin() -> Option<usize> { absorb_on().then(super::recon_min_rows) }

/// TP-4X1: fold ragged chunks. A chunk of fewer than `rmin` rows is merged into its neighbour (dropping
/// the boundary that separates them) when (a) the merged chunk fits `cap` rows, and (b) the dropped
/// boundary is not required: never the tail checkpoint `tail`, never `n`, and - in a checkpointing run
/// (`ckpts`) - never a boundary the WP16 rule checkpoints (`wp16_due`) UNLESS the tail checkpoint sits
/// fewer than `rmin` rows before it (that checkpoint dominates it: a resume loses < `rmin` rows).
/// Forward merge (drop the chunk's end) is tried first, then backward (drop its start). Cache
/// granularity cost: zero checkpoints beyond `rmin - 1` rows are lost, and `tail` / `n` never move.
pub fn absorb_grid(from: usize, ends: Vec<usize>, tail: Option<usize>, n: usize, c: usize, cap: usize,
                   ckpts: bool, rmin: usize) -> Vec<usize> {
    let mut e = ends;
    let droppable = |b: usize| -> bool {
        if b >= n || Some(b) == tail { return false; }
        if !ckpts || !wp16_due(b, n, c) { return true; }
        tail.is_some_and(|t| t < b && b - t < rmin)
    };
    let mut i = 0;
    while i < e.len() {
        let s = if i == 0 { from } else { e[i - 1] };
        if e[i] - s < rmin {
            if i + 1 < e.len() && droppable(e[i]) && e[i + 1] - s <= cap { e.remove(i); continue; }
            if i >= 1 && droppable(e[i - 1]) {
                let s2 = if i >= 2 { e[i - 2] } else { from };
                if e[i] - s2 <= cap { e.remove(i - 1); i -= 1; continue; }
            }
        }
        i += 1;
    }
    e
}

/// TP-4X1: the served chunk grid of a run over [from, n): `realign` = a tracked (C1) run - run_grid with
/// the tail split, merged to `cap` rows; else today's plain grid. `absorb` = Some(rmin) applies
/// absorb_grid. `absorb = None` is exactly the pre-TP-4X1 grid (merge_grid_cap with the old cap).
pub fn plan_grid(from: usize, n: usize, c: usize, realign: bool, tail: Option<usize>, cap: usize,
                 absorb: Option<usize>, ckpts: bool) -> Vec<usize> {
    let g = if realign {
        merge_grid_cap(from, run_grid(from, n, c, true, tail), tail, c, cap)
    } else {
        run_grid(from, n, c, false, None)
    };
    match absorb { Some(rmin) => absorb_grid(from, g, tail, n, c, cap, ckpts, rmin), None => g }
}

/// C1: chunk ends of a run over [from, n) at width c. `realign`: the first chunk ends at the next
/// C multiple (a run that starts off the grid returns to it; from a C multiple this IS the plain
/// grid). `tail`: an extra boundary there. realign=false, tail=None = today's grid (from + k*C).
pub fn run_grid(from: usize, n: usize, c: usize, realign: bool, tail: Option<usize>) -> Vec<usize> {
    let c = c.max(1);
    let mut ends = Vec::new();
    let mut pos = from;
    while pos < n {
        let mut e = if realign { (pos / c + 1) * c } else { pos + c }.min(n);
        if let Some(t) = tail { if pos < t && t < e { e = t; } }
        ends.push(e);
        pos = e;
    }
    ends
}

/// f32-word layout of one checkpoint buffer:
/// [conv_0 .. conv_{L-1} | S_0 .. S_{L-1} | PLE ring row | head-fill carry].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CkptLayout {
    pub n_gdn: usize,
    pub per_conv: usize,
    pub per_s: usize,
    pub ple_words: usize,
    pub carry_words: usize,
}

impl CkptLayout {
    fn conv_off(&self, i: usize) -> usize { i * self.per_conv }
    fn s_off(&self, i: usize) -> usize { self.n_gdn * self.per_conv + i * self.per_s }
    fn ple_off(&self) -> usize { self.n_gdn * (self.per_conv + self.per_s) }
    fn carry_off(&self) -> usize { self.ple_off() + self.ple_words }
    pub fn words(&self) -> usize { self.carry_off() + self.carry_words }
    pub fn bytes(&self) -> usize { self.words() * 4 }
    /// One boot line: bytes per checkpoint and what they hold.
    pub fn describe(&self) -> String {
        let mb = |w: usize| w as f64 * 4.0 / 1e6;
        format!("{:.1} MB per checkpoint (GDN S {:.1} MB + conv {:.1} MB over {} GDN layers, PLE ring {:.2} MB, \
                 head-fill carry {:.3} MB, + the host PLE n-gram pair)",
                mb(self.words()), mb(self.n_gdn * self.per_s), mb(self.n_gdn * self.per_conv), self.n_gdn,
                mb(self.ple_words), mb(self.carry_words))
    }
}

/// One intermediate recurrent checkpoint of a slot: the device words (CkptLayout) + the host PLE
/// n-gram pair (the last two ids before the checkpoint position).
pub struct RecurCkpt {
    buf: CudaSlice<f32>,
    hist: [i32; 2],
}

/// Bitwise comparison of two checkpoints, per component (mismatching f32 words / total).
#[derive(Clone, Copy, Debug, Default)]
pub struct CkptDiff {
    pub conv: (usize, usize),
    pub s: (usize, usize),
    pub ple: (usize, usize),
    pub carry: (usize, usize),
    pub hist_eq: bool,
}

impl CkptDiff {
    pub fn exact(&self) -> bool {
        self.conv.0 == 0 && self.s.0 == 0 && self.ple.0 == 0 && self.carry.0 == 0 && self.hist_eq
    }
    pub fn line(&self) -> String {
        format!("GDN S differ {}/{}, conv differ {}/{}, PLE ring differ {}/{}, carry differ {}/{}, PLE hist {}",
                self.s.0, self.s.1, self.conv.0, self.conv.1, self.ple.0, self.ple.1,
                self.carry.0, self.carry.1, if self.hist_eq { "equal" } else { "DIFFER" })
    }
}

impl FwdModel {
    pub fn ckpt_layout(&self) -> CkptLayout {
        let cfg = &self.tc; // TP-B: the trunk's per-rank recurrent state
        CkptLayout {
            n_gdn: self.s_state.len().min(self.conv_state.len()),
            per_conv: (cfg.key_dim() * 2 + cfg.value_dim()) * cfg.conv_kernel,
            per_s: cfg.lin_num_v_heads * cfg.lin_k_dim * cfg.lin_v_dim,
            ple_words: 10240 * 9 / 2, // PLE f16 ring row = 10240*9 halves (copy_slot_state's unit)
            carry_words: cfg.hc_count.max(1) * cfg.hidden_size,
        }
    }

    /// A fresh checkpoint buffer (contents undefined until `ckpt_save` fills all of it).
    pub fn ckpt_alloc(&self) -> Result<RecurCkpt> {
        let w = self.ckpt_layout().words();
        Ok(RecurCkpt { buf: self.dev.alloc_zeros::<f32>(w)?, hist: [248046i32; 2] })
    }

    /// Copy `slot`'s recurrent state (+ the head-fill carry in `psc`) into `ck`. Device copies on
    /// the compute stream, ordered after the prefill chunk that produced the state: bit-exact.
    pub fn ckpt_save(&self, slot: usize, psc: &PrefillScratch, ck: &mut RecurCkpt) -> Result<()> {
        let p = *ck.buf.device_ptr() as u64;
        self.ckpt_copy(slot, psc, p, &mut ck.hist, true)
    }

    /// Restore `slot`'s recurrent state (+ the carry into `psc`) from `ck` — the resume replaces
    /// reset_slot: rows below the checkpoint (trunk/head KV, QSA planes) stay as the slot has them.
    pub fn ckpt_restore(&self, slot: usize, psc: &mut PrefillScratch, ck: &RecurCkpt) -> Result<()> {
        let p = *ck.buf.device_ptr() as u64;
        let mut h = ck.hist;
        self.ckpt_copy(slot, psc, p, &mut h, false)
    }

    fn ckpt_copy(&self, slot: usize, psc: &PrefillScratch, buf: u64, hist: &mut [i32; 2], save: bool) -> Result<()> {
        let lay = self.ckpt_layout();
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        let grid = |n: usize| ((((n as u64) + 255) / 256) as u32, 1, 1);
        // live plane element offset `loff`, checkpoint word offset `coff`
        let cp = |live: u64, loff: usize, coff: usize, n: usize| -> Result<()> {
            let (dst, src, doff, soff) = if save { (buf, live, coff, loff) } else { (live, buf, loff, coff) };
            xqlaunch!(l, "xq_copy_f32", grid(n), (256, 1, 1), 0, (dst, src, n as i64, doff as i64, soff as i64))?;
            Ok(())
        };
        for i in 0..lay.n_gdn {
            cp(*self.conv_state[i].device_ptr() as u64, slot * lay.per_conv, lay.conv_off(i), lay.per_conv)?;
            cp(*self.s_state[i].device_ptr() as u64, slot * lay.per_s, lay.s_off(i), lay.per_s)?;
        }
        cp(*self.ple_state.device_ptr() as u64, slot * lay.ple_words, lay.ple_off(), lay.ple_words)?;
        // head-fill carry: psc holds ONE carry (the slot being prefilled) — present iff a draft head
        // exists; else keep the checkpoint's region defined (zero) so comparisons stay meaningful.
        let rw = lay.carry_words;
        if psc.hfill_carry.len() >= rw {
            cp(*psc.hfill_carry.device_ptr() as u64, 0, lay.carry_off(), rw)?;
        } else if save {
            xqlaunch!(l, "xq_memset_f32", grid(rw), (256, 1, 1), 0, (buf, lay.carry_off() as i64, rw as i64))?;
        }
        let mut live = self.ple_hist.lock().unwrap();
        if save { *hist = [live[slot * 2], live[slot * 2 + 1]]; }
        else { live[slot * 2] = hist[0]; live[slot * 2 + 1] = hist[1]; }
        Ok(())
    }

    /// C1 diagnostics: relative L2 ||a-b|| / ||b|| of the GDN S state per layer (median, max) and of
    /// the conv state over all layers (b = the reference).
    pub fn ckpt_rel(&self, a: &RecurCkpt, b: &RecurCkpt) -> Result<(f64, f64, f64)> {
        let lay = self.ckpt_layout();
        self.dev.synchronize()?;
        let ha = self.dev.dtoh_sync_copy(&a.buf)?;
        let hb = self.dev.dtoh_sync_copy(&b.buf)?;
        let rel = |o: usize, n: usize| -> f64 {
            let (mut d, mut r) = (0f64, 0f64);
            for (x, y) in ha[o..o + n].iter().zip(&hb[o..o + n]) {
                let (x, y) = (*x as f64, *y as f64);
                d += (x - y) * (x - y);
                r += y * y;
            }
            if r > 0.0 { (d / r).sqrt() } else { d.sqrt() }
        };
        let mut s: Vec<f64> = (0..lay.n_gdn).map(|i| rel(lay.s_off(i), lay.per_s)).collect();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med = s.get(s.len() / 2).copied().unwrap_or(0.0);
        let max = s.last().copied().unwrap_or(0.0);
        Ok((med, max, rel(lay.conv_off(0), lay.n_gdn * lay.per_conv)))
    }

    /// Bitwise diff of two checkpoints (host readback; diagnostics only).
    pub fn ckpt_diff(&self, a: &RecurCkpt, b: &RecurCkpt) -> Result<CkptDiff> {
        let lay = self.ckpt_layout();
        self.dev.synchronize()?;
        let ha = self.dev.dtoh_sync_copy(&a.buf)?;
        let hb = self.dev.dtoh_sync_copy(&b.buf)?;
        let cnt = |o: usize, n: usize| -> (usize, usize) {
            (ha[o..o + n].iter().zip(&hb[o..o + n]).filter(|(x, y)| x.to_bits() != y.to_bits()).count(), n)
        };
        let conv = cnt(lay.conv_off(0), lay.n_gdn * lay.per_conv);
        let s = cnt(lay.s_off(0), lay.n_gdn * lay.per_s);
        Ok(CkptDiff {
            conv,
            s,
            ple: cnt(lay.ple_off(), lay.ple_words),
            carry: cnt(lay.carry_off(), lay.carry_words),
            hist_eq: a.hist == b.hist,
        })
    }

    /// PFX1 (b) gate on a WP16 resume (--pfx1-headkv-poison=1 with the head fill + PFX1 headkv
    /// on): NaN-poison the slot's head-KV rows [q, max_pos) — the rows the resumed request must
    /// write before any draft reads them (rows < q are the checkpointed prefix and stay). Hashes +
    /// drafted/accepted identical to an unpoisoned run prove the invariant on the resume path.
    pub fn ckpt_poison_head_kv_from(&self, slot: usize, q: usize) -> Result<()> {
        if !(pfx1_on(PFX1_HEADKV) && head_fill_on() && pfx1_headkv_poison()) { return Ok(()); }
        let Some(head) = self.mtp.as_ref() else { return Ok(()) };
        if q >= self.max_pos { return Ok(()); }
        let (nkv, hd) = (self.cfg.num_kv_heads, self.cfg.head_dim);
        let rb = kv_rowbytes(self.kv_fmt, hd);
        let base = *head.kv.device_ptr() as u64;
        anyhow::ensure!(head.kv.len() >= self.width * 2 * nkv * self.max_pos * rb && rb % 2 == 0,
                        "WP16 poison: head KV layout mismatch");
        let l = Launcher { dev: &self.dev, stream: &self.stream };
        let n16 = ((self.max_pos - q) * rb / 2) as i64;
        for g in 0..2 * nkv {
            let off = ((slot * 2 * nkv + g) * self.max_pos + q) * rb;
            xqlaunch!(l, "xq_memset_u16", ((((n16 + 255) / 256) as u32), 1, 1), (256, 1, 1), 0,
                      (base + off as u64, n16, 0x7FC0i32))?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The checkpoint store (host bookkeeping; generic over the buffer so the policy is unit-tested
// without a GPU). One list per slot + the token prefix it describes; an LRU byte cap over all.
// ---------------------------------------------------------------------------

struct CkEnt<B> {
    pos: usize,
    used: u64,
    b: B,
}

struct SlotCk<B> {
    /// prompt tokens of the slot's ALIGNED-grid prefix: rows [0, toks.len()) of the slot were written
    /// by an aligned chunked prefill of exactly these tokens, and have not been overwritten since.
    toks: Vec<u32>,
    /// chunk width C the checkpoints were taken at (the key: a resume at another C never matches)
    c: usize,
    list: Vec<CkEnt<B>>,
}

pub struct CkptStore<B> {
    /// the byte cap in buffers (floor(cap_bytes / bytes per checkpoint))
    pub cap: usize,
    /// buffers allocated so far (live + free) — never exceeds `cap`
    pub n_alloc: usize,
    free: Vec<B>,
    slots: Vec<SlotCk<B>>,
    tick: u64,
    alloc_failed: bool,
}

impl<B> CkptStore<B> {
    pub fn new(width: usize, cap: usize) -> Self {
        Self {
            cap,
            n_alloc: 0,
            free: Vec::new(),
            slots: (0..width).map(|_| SlotCk { toks: Vec::new(), c: 0, list: Vec::new() }).collect(),
            tick: 0,
            alloc_failed: false,
        }
    }

    /// (resume position, LCP): the deepest checkpoint <= min(LCP(prompt, the slot's aligned
    /// prefix), end) taken at chunk width `c`; 0 = none.
    pub fn resume_point(&self, slot: usize, prompt: &[u32], end: usize, c: usize) -> (usize, usize) {
        let Some(s) = self.slots.get(slot) else { return (0, 0) };
        let lcp = s.toks.iter().zip(prompt).take_while(|(a, b)| a == b).count();
        if s.c != c { return (0, lcp); }
        let lim = lcp.min(end);
        (s.list.iter().filter(|e| e.pos <= lim).map(|e| e.pos).max().unwrap_or(0), lcp)
    }

    fn drop_above(&mut self, slot: usize, q: usize) -> usize {
        let s = &mut self.slots[slot];
        let mut n = 0;
        let mut i = 0;
        while i < s.list.len() {
            if s.list[i].pos > q {
                let e = s.list.remove(i);
                self.free.push(e.b);
                n += 1;
            } else {
                i += 1;
            }
        }
        n
    }

    /// An ALIGNED run starts at `from` (0, or a checkpoint position of this slot): rows >= from
    /// get rewritten, so every checkpoint above `from` goes (returns how many) and the aligned
    /// prefix is cut to `from`; the run's chunks extend it again (`extend`).
    pub fn begin_aligned(&mut self, slot: usize, from: usize, c: usize) -> usize {
        if self.slots[slot].c != c || self.slots[slot].toks.len() < from {
            // another chunk grid, or an inconsistent prefix: nothing here is resumable at c
            let n = self.slots[slot].list.len();
            self.clear_slot(slot);
            self.slots[slot].c = c;
            return n;
        }
        let n = self.drop_above(slot, from);
        self.slots[slot].toks.truncate(from);
        n
    }

    /// An UNALIGNED run (a prompt-end prefix-cache hit resuming at `from`): rows >= from are
    /// rewritten in a different chunk grid — nothing there is checkpoint-grade (no-op in
    /// practice: the aligned prefix never outruns the prompt-end snapshot).
    pub fn begin_unaligned(&mut self, slot: usize, from: usize) {
        self.drop_above(slot, from);
        let s = &mut self.slots[slot];
        s.toks.truncate(s.toks.len().min(from));
    }

    /// An aligned run wrote the rows of `toks` (the next chunk).
    pub fn extend(&mut self, slot: usize, toks: &[u32]) {
        self.slots[slot].toks.extend_from_slice(toks);
    }

    pub fn toks_len(&self, slot: usize) -> usize { self.slots[slot].toks.len() }

    /// C1: the slot's tracked prefix covers prompt[..upto] exactly and was taken at chunk width `c`
    /// (a prompt-end prefix hit may then extend it: its rows below `upto` are this prefix's rows).
    pub fn prefix_ok(&self, slot: usize, prompt: &[u32], upto: usize, c: usize) -> bool {
        let s = &self.slots[slot];
        s.c == c && s.toks.len() >= upto && prompt.len() >= upto && s.toks[..upto] == prompt[..upto]
    }

    /// A buffer for a new checkpoint: the free list, else a new allocation under the cap, else the
    /// least-recently-used live checkpoint of ANY slot. None = cap 0. An allocation failure (device
    /// memory) freezes the cap at what is allocated and falls back to eviction.
    pub fn acquire(&mut self, alloc: impl FnOnce() -> Result<B>) -> Option<B> {
        if let Some(b) = self.free.pop() { return Some(b); }
        if self.n_alloc < self.cap && !self.alloc_failed {
            match alloc() {
                Ok(b) => { self.n_alloc += 1; return Some(b); }
                Err(e) => {
                    crate::rprintln!("[exl3-serve] WP16: checkpoint allocation failed after {} buffer(s) ({e:#}) — \
                              the cap is frozen there (LRU eviction from now on)", self.n_alloc);
                    self.alloc_failed = true;
                }
            }
        }
        let (si, ei) = self.slots.iter().enumerate()
            .flat_map(|(si, s)| s.list.iter().enumerate().map(move |(ei, e)| (e.used, si, ei)))
            .min().map(|(_, si, ei)| (si, ei))?;
        Some(self.slots[si].list.remove(ei).b)
    }

    /// Record checkpoint `b` of `slot` at `pos` (replaces one already there).
    pub fn insert(&mut self, slot: usize, pos: usize, b: B) {
        self.tick += 1;
        let used = self.tick;
        let s = &mut self.slots[slot];
        match s.list.iter().position(|e| e.pos == pos) {
            Some(i) => {
                let old = std::mem::replace(&mut s.list[i], CkEnt { pos, used, b });
                self.free.push(old.b);
            }
            None => {
                let i = s.list.iter().position(|e| e.pos > pos).unwrap_or(s.list.len());
                s.list.insert(i, CkEnt { pos, used, b });
            }
        }
    }

    /// A buffer that was acquired but not inserted (a failed save) goes back to the free list.
    pub fn release(&mut self, b: B) { self.free.push(b); }

    /// LRU: a resume from `pos` counts as a use.
    pub fn touch(&mut self, slot: usize, pos: usize) {
        self.tick += 1;
        let t = self.tick;
        if let Some(e) = self.slots[slot].list.iter_mut().find(|e| e.pos == pos) { e.used = t; }
    }

    pub fn get(&self, slot: usize, pos: usize) -> Option<&B> {
        self.slots.get(slot)?.list.iter().find(|e| e.pos == pos).map(|e| &e.b)
    }

    /// Forget everything about `slot` (its rows are being overwritten, or its state is unknown).
    pub fn clear_slot(&mut self, slot: usize) {
        let s = &mut self.slots[slot];
        s.toks.clear();
        for e in s.list.drain(..) { self.free.push(e.b); }
    }

    pub fn live(&self) -> usize { self.slots.iter().map(|s| s.list.len()).sum() }
    pub fn positions(&self, slot: usize) -> Vec<usize> { self.slots[slot].list.iter().map(|e| e.pos).collect() }
}

// ---------------------------------------------------------------------------
// --probe-exl3-wp16: the single-process gate. Prompt A prefills on slot 0 (fresh, aligned, with
// the served checkpoint rule); prompt B (A's first S tokens + a different suffix) then resumes on
// slot 0 from the deepest checkpoint <= min(LCP, plen-1), and prefills FRESH on slot 1. The two
// B states (GDN S / conv / PLE ring / PLE hist / head-fill carry), the seam step's first-token
// logits and a greedy continuation must all be bit-identical.
// ---------------------------------------------------------------------------

pub struct Wp16ProbeArgs {
    pub base: FwdArgs,
    /// B = A[..split] + reversed(A[split..]) (default split: len(A) - min(256, len(A)/4))
    pub split: Option<usize>,
    /// explicit prompt B (ids file; overrides split)
    pub b_ids: Option<String>,
}

fn read_ids(f: &str) -> Result<Vec<u32>> {
    let s = std::fs::read_to_string(f)?;
    serde_json::from_str(&s).or_else(|_| {
        s.split_whitespace().map(|x| x.parse::<u32>()).collect::<std::result::Result<Vec<_>, _>>()
            .map_err(anyhow::Error::from)
    })
}

/// The serve's chunk loop (prefill_range) for prompt[from..to) on `slot`, optionally taking the
/// served checkpoints into `store` (an aligned run). Returns the checkpoint positions taken.
fn probe_prefill(model: &FwdModel, psc: &mut PrefillScratch, slot: usize, prompt: &[u32], from: usize,
                 to: usize, c: usize, mut store: Option<&mut CkptStore<RecurCkpt>>) -> Result<Vec<usize>> {
    let mut pos = from;
    let mut taken = Vec::new();
    while pos < to {
        let end = (pos + c).min(to);
        let toks: Vec<i32> = prompt[pos..end].iter().map(|&t| t as i32).collect();
        model.prefill_chunk(psc, &toks, pos, slot, false, None)?;
        if let Some(st) = store.as_deref_mut() {
            st.extend(slot, &prompt[pos..end]);
            if wp16_due(end, to, c) {
                if let Some(mut b) = st.acquire(|| model.ckpt_alloc()) {
                    model.ckpt_save(slot, psc, &mut b)?;
                    st.insert(slot, end, b);
                    taken.push(end);
                }
            }
        }
        pos = end;
    }
    Ok(taken)
}

/// C1 probe (--wp16-probe-tail=<t>, 0 = len(A) - 6: a chat generation-prompt tail): how far the
/// tail-checkpoint path moves the numbers (it is output-changing by construction).
///   A: fresh on today's grid (slot 1 = the knob-off bytes) vs split at t (slot 0; checkpoint at t).
///   B = A[..t] + A[t/2..] (LCP t): resumed from the tail checkpoint on the realigned grid (slot 0)
///   vs fresh on today's grid (slot 1 = what knob off serves: a WP16 aligned resume is bitwise fresh).
/// Per arm: bit counts + relL2 of the GDN S (per-layer median / max) and conv state, first-token
/// logits (differing count, max |d|, argmax + the fresh argmax's rank in the tail arm) and the first
/// differing greedy continuation index; plus the prefill times.
fn probe_tail(a: &FwdArgs, c: usize, pa: &[u32], t0: usize) -> Result<()> {
    let t = if t0 == 0 { pa.len().saturating_sub(6) } else { t0 };
    anyhow::ensure!(t >= 2 && t + 1 < pa.len(), "tail {t} must lie inside prompt A ({} tokens)", pa.len());
    let mut pb = pa[..t].to_vec();
    pb.extend_from_slice(&pa[t / 2..]);
    let cont = a.max_new.max(1);
    let need = pa.len().max(pb.len()) + cont + 2;
    anyhow::ensure!(need <= a.max_pos, "prompts + continuation need {need} positions > --max-seq-len {}", a.max_pos);
    let model = FwdModel::load(&a.dir, 2, a.max_pos)?;
    let mut sc = Scratch::new(&model.dev, &model.cfg, 2, model.cfg.rotary_dim, model.max_pos())?;
    let mut psc = model.prefill_scratch(2 * c - 1)?; // v2: merged chunks up to 2C - 1 rows
    let v = model.cfg.vocab_size;
    println!("C1 TAIL probe: A {} tokens, tail t={t}, B {} tokens (LCP t), C {c}, continuation {cont}", pa.len(), pb.len());
    let run = |psc: &mut PrefillScratch, slot: usize, p: &[u32], from: usize, grid: &[usize]| -> Result<f64> {
        let t = std::time::Instant::now();
        let mut pos = from;
        for &e in grid {
            let toks: Vec<i32> = p[pos..e].iter().map(|&x| x as i32).collect();
            model.prefill_chunk(psc, &toks, pos, slot, false, None)?;
            pos = e;
        }
        model.dev.synchronize()?;
        Ok(t.elapsed().as_secs_f64() * 1e3)
    };
    let tail_of = |sc: &mut Scratch, slot: usize, p: &[u32]| -> Result<(Vec<u16>, Vec<u32>)> {
        let e = p.len() - 1;
        let mut id = model.forward_step(sc, &[p[e] as i32], &[e], &[slot], None)?[0];
        let lg = model.logits_host(sc)?[..v].to_vec();
        let mut toks = vec![id as u32];
        for i in 1..cont {
            id = model.forward_step(sc, &[id], &[e + i], &[slot], None)?[0];
            toks.push(id as u32);
        }
        Ok((lg, toks))
    };
    let report = |tag: &str, d: &CkptDiff, rel: (f64, f64, f64), lt: &[u16], lf: &[u16], tt: &[u32], tf: &[u32]| {
        let f = |b: u16| half::f16::from_bits(b).to_f32();
        let nl = lt.iter().zip(lf).filter(|(x, y)| x != y).count();
        let md = lt.iter().zip(lf).map(|(&x, &y)| (f(x) - f(y)).abs()).fold(0f32, f32::max);
        let am = |l: &[u16]| l.iter().enumerate().max_by(|x, y| f(*x.1).partial_cmp(&f(*y.1)).unwrap_or(std::cmp::Ordering::Equal)).map(|x| x.0).unwrap_or(0);
        let (at, af) = (am(lt), am(lf));
        let rank = lt.iter().filter(|&&x| f(x) > f(lt[af])).count() + 1;
        let fd = tt.iter().zip(tf).position(|(x, y)| x != y);
        println!("C1 TAIL {tag}: state {} | S relL2 med {:.3e} max {:.3e} | conv relL2 {:.3e} | first-token logits differ \
                  {nl}/{v} (max|d| {md:.4}), argmax tail {at} vs fresh {af} (fresh argmax rank {rank} in tail) | greedy \
                  continuation {}", d.line(), rel.0, rel.1, rel.2,
                 match fd { None => format!("IDENTICAL over {} tokens", tt.len()), Some(i) => format!("first differs at token {i} of {}", tt.len()) });
    };
    // A fresh (today's grid) on slot 1
    let end_a = pa.len() - 1;
    model.reset_slot_for_prefill(1)?;
    let ms_af = run(&mut psc, 1, pa, 0, &run_grid(0, end_a, c, false, None))?;
    let mut af = model.ckpt_alloc()?;
    model.ckpt_save(1, &psc, &mut af)?;
    let (lg_af, tk_af) = tail_of(&mut sc, 1, pa)?;
    // A split at t on slot 0, checkpoint at t
    model.reset_slot_for_prefill(0)?;
    let g = merge_grid(0, run_grid(0, end_a, c, true, Some(t)), Some(t), c);
    let i = g.iter().position(|&e| e == t).context("tail not on the grid")?;
    let ms_a1 = run(&mut psc, 0, pa, 0, &g[..=i])?;
    let mut ck = model.ckpt_alloc()?;
    model.ckpt_save(0, &psc, &mut ck)?;
    let ms_a2 = run(&mut psc, 0, pa, t, &g[i + 1..])?;
    let mut at = model.ckpt_alloc()?;
    model.ckpt_save(0, &psc, &mut at)?;
    let (lg_at, tk_at) = tail_of(&mut sc, 0, pa)?;
    println!("C1 TAIL A: prefill fresh {ms_af:.1} ms ({} chunks) vs split {:.1} ms ({} chunks: grid {g:?})",
             run_grid(0, end_a, c, false, None).len(), ms_a1 + ms_a2, g.len());
    report("A (split at t vs fresh)", &model.ckpt_diff(&at, &af)?, model.ckpt_rel(&at, &af)?, &lg_at, &lg_af, &tk_at, &tk_af);
    // B resumed from the tail checkpoint (slot 0, realigned) vs fresh (slot 1)
    let end_b = pb.len() - 1;
    model.ckpt_restore(0, &mut psc, &ck)?;
    let gb = merge_grid(t, run_grid(t, end_b, c, true, None), None, c);
    let ms_br = run(&mut psc, 0, &pb, t, &gb)?;
    let mut br = model.ckpt_alloc()?;
    model.ckpt_save(0, &psc, &mut br)?;
    let (lg_br, tk_br) = tail_of(&mut sc, 0, &pb)?;
    model.reset_slot_for_prefill(1)?;
    let ms_bf = run(&mut psc, 1, &pb, 0, &run_grid(0, end_b, c, false, None))?;
    let mut bf = model.ckpt_alloc()?;
    model.ckpt_save(1, &psc, &mut bf)?;
    let (lg_bf, tk_bf) = tail_of(&mut sc, 1, &pb)?;
    println!("C1 TAIL B: resumed at {t}: prefill {ms_br:.1} ms ({} tokens, grid {gb:?}) vs fresh {ms_bf:.1} ms ({} tokens)",
             end_b - t, end_b);
    report("B (tail resume vs fresh)", &model.ckpt_diff(&br, &bf)?, model.ckpt_rel(&br, &bf)?, &lg_br, &lg_bf, &tk_br, &tk_bf);
    println!("C1 TAIL PROBE: DONE");
    Ok(())
}

pub fn probe_wp16(p: &Wp16ProbeArgs) -> Result<()> {
    let a = &p.base;
    let c = a.prefill_chunk.filter(|&c| c > 0).unwrap_or(2048);
    let pa = gate_prompt_ids(a)?;
    anyhow::ensure!(pa.len() >= 3, "prompt A too short ({} tokens)", pa.len());
    if let Some(t) = crate::opts::var(crate::opt!("wp16-probe-tail")).ok().and_then(|v| v.trim().parse::<usize>().ok()) {
        return probe_tail(a, c, &pa, t);
    }
    let pb: Vec<u32> = match &p.b_ids {
        Some(f) => read_ids(f)?,
        None => {
            let s = p.split.unwrap_or(pa.len() - (pa.len() / 4).min(256)).clamp(1, pa.len() - 1);
            let mut b = pa[..s].to_vec();
            b.extend(pa[s..].iter().rev());
            b
        }
    };
    anyhow::ensure!(pb.len() >= 2, "prompt B too short");
    let cont = a.max_new.max(1);
    let need = pa.len().max(pb.len()) + cont + 2;
    anyhow::ensure!(need <= a.max_pos, "prompts + continuation need {need} positions > --max-seq-len {}", a.max_pos);
    let model = FwdModel::load(&a.dir, 2, a.max_pos)?;
    let lay = model.ckpt_layout();
    println!("WP16 probe: A {} tokens, B {} tokens, C {c}, continuation {cont}; {}", pa.len(), pb.len(), lay.describe());
    let mut sc = Scratch::new(&model.dev, &model.cfg, 2, model.cfg.rotary_dim, model.max_pos())?;
    let mut psc = model.prefill_scratch(c)?;
    let v = model.cfg.vocab_size;
    let mut store: CkptStore<RecurCkpt> = CkptStore::new(2, 64);

    // A: fresh aligned prefill on slot 0 with the served checkpoint rule
    let end_a = pa.len() - 1;
    model.reset_slot_for_prefill(0)?;
    store.begin_aligned(0, 0, c);
    let t = std::time::Instant::now();
    let taken = probe_prefill(&model, &mut psc, 0, &pa, 0, end_a, c, Some(&mut store))?;
    println!("WP16 probe: A prefilled on slot 0 in {:.1} ms; checkpoints at {taken:?}", t.elapsed().as_secs_f64() * 1e3);

    // B resumed on slot 0
    let end_b = pb.len() - 1;
    let (q, lcp) = store.resume_point(0, &pb, end_b, c);
    anyhow::ensure!(q > 0, "no checkpoint <= min(LCP {lcp}, plen-1 {end_b}) — B must share a checkpointed prefix with A");
    let dropped = store.begin_aligned(0, q, c);
    store.touch(0, q);
    model.ckpt_restore(0, &mut psc, store.get(0, q).unwrap())?;
    model.ckpt_poison_head_kv_from(0, q)?;
    let t = std::time::Instant::now();
    probe_prefill(&model, &mut psc, 0, &pb, q, end_b, c, Some(&mut store))?;
    model.dev.synchronize()?;
    let ms_res = t.elapsed().as_secs_f64() * 1e3;
    let mut r = model.ckpt_alloc()?;
    model.ckpt_save(0, &psc, &mut r)?;
    let mut run_tail = |slot: usize| -> Result<(Vec<u16>, Vec<u32>)> {
        let mut id = model.forward_step(&mut sc, &[pb[end_b] as i32], &[end_b], &[slot], None)?[0];
        let lg = model.logits_host(&sc)?[..v].to_vec();
        let mut toks = vec![id as u32];
        for i in 1..cont {
            id = model.forward_step(&mut sc, &[id], &[end_b + i], &[slot], None)?[0];
            toks.push(id as u32);
        }
        Ok((lg, toks))
    };
    let (lg_r, tk_r) = run_tail(0)?;

    // B fresh on slot 1
    model.reset_slot_for_prefill(1)?;
    let t = std::time::Instant::now();
    probe_prefill(&model, &mut psc, 1, &pb, 0, end_b, c, None)?;
    model.dev.synchronize()?;
    let ms_fresh = t.elapsed().as_secs_f64() * 1e3;
    let mut f = model.ckpt_alloc()?;
    model.ckpt_save(1, &psc, &mut f)?;
    let d = model.ckpt_diff(&r, &f)?;
    let (lg_f, tk_f) = run_tail(1)?;
    let nl = lg_r.iter().zip(&lg_f).filter(|(x, y)| x != y).count();
    let nt = tk_r.iter().zip(&tk_f).filter(|(x, y)| x != y).count();
    println!("WP16 probe: B resumed at {q} (LCP {lcp}, plen {}; {dropped} checkpoint(s) above dropped): prefill {:.1} ms \
              resumed vs {:.1} ms fresh", pb.len(), ms_res, ms_fresh);
    println!("WP16 probe: state after prefill (resumed slot 0 vs fresh slot 1): {}", d.line());
    println!("WP16 probe: first-token logits differ {nl}/{v} (argmax {} vs {}); greedy continuation differs at {nt}/{cont} tokens",
             tk_r[0], tk_f[0]);
    if d.exact() && nl == 0 && nt == 0 {
        println!("WP16 PROBE: EXACT");
        Ok(())
    } else {
        println!("WP16 PROBE: MISMATCH");
        anyhow::bail!("WP16 resumed prefill is not bit-identical to the fresh prefill")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn due_rule_aligned_ends() {
        let c = 2048;
        // sys 5K + message: end 5100 -> the two C boundaries of the last 2 chunks
        let e: Vec<usize> = (1..=3).map(|k| (k * c).min(5100)).filter(|&e| wp16_due(e, 5100, c)).collect();
        assert_eq!(e, vec![2048, 4096]);
        // 8K sys prompt: end 8300 -> 6144 (last-2) and 8192 (stride + last-2)
        let e: Vec<usize> = (1..=5).map(|k| (k * c).min(8300)).filter(|&e| wp16_due(e, 8300, c)).collect();
        assert_eq!(e, vec![6144, 8192]);
        // 128K doc + question: every 8192, plus the last 2 chunk boundaries
        let n = 131_072 + 700;
        let e: Vec<usize> = (1..=n / c + 1).map(|k| (k * c).min(n)).filter(|&e| wp16_due(e, n, c)).collect();
        assert_eq!(e.len(), 16 + 1); // 8192..131072 (16) + 129024
        assert!(e.contains(&129_024) && e.contains(&131_072) && !e.contains(&126_976));
        // an aligned end is itself a checkpoint; a partial final chunk's end never is
        assert!(wp16_due(6144, 6144, c));
        assert!(!wp16_due(6000, 6000, c));
        assert!(!wp16_due(0, 100, c));
        assert!(!wp16_due(2048, 2047, c));
        assert!(!wp16_due(2048, 4096, 0));
    }

    #[test]
    fn c1_tail_grid() {
        let c = 2048;
        // knob off: today's grid exactly (from + k*C)
        assert_eq!(run_grid(0, 5000, c, false, None), vec![2048, 4096, 5000]);
        assert_eq!(run_grid(2213, 6581, c, false, None), vec![4261, 6309, 6581]);
        // realign from a C multiple = the plain grid
        assert_eq!(run_grid(2048, 6581, c, true, None), run_grid(2048, 6581, c, false, None));
        // tail split inside the last chunk; realign after an off-grid resume
        assert_eq!(run_grid(0, 6581, c, true, Some(6573)), vec![2048, 4096, 6144, 6573, 6581]);
        assert_eq!(run_grid(6573, 10940, c, true, Some(10934)), vec![8192, 10240, 10934, 10940]);
        assert_eq!(run_grid(10, 20, c, true, Some(15)), vec![15, 20]);
        // tail point rule (boundary - 8)
        assert_eq!(tail_point(Some(6577), 0, 6581, c, true), Some(6569));
        assert_eq!(tail_point(Some(6577), 6500, 6581, c, true), None); // < 1024 past the run start (extra chunk)
        assert_eq!(tail_point(Some(6250), 0, 6581, c, true), None);    // < 128 past the C multiple 6144
        assert_eq!(tail_point(Some(100), 0, 108, c, true), None);      // short prompt: today's grid
        assert_eq!(tail_point(Some(1515), 0, 1519, c, true), Some(1507)); // < C: extra chunk, gain >= 1024
        assert_eq!(tail_point(Some(547), 0, 551, c, true), None);         // < C and < 1024: no split
        assert_eq!(tail_point(Some(1515), 0, 1519, c, false), None);        // value 1: never an extra chunk
        assert_eq!(tail_point(Some(6577), 0, 6581, c, false), Some(6569)); // value 1: absorbed splits stay
        // v2 merges: the fresh-prompt count is kept
        assert_eq!(merge_grid(0, run_grid(0, 6581, c, true, Some(6569)), Some(6569), c), vec![2048, 4096, 6569, 6581]);
        assert_eq!(merge_grid(6569, run_grid(6569, 10940, c, true, Some(10928)), Some(10928), c), vec![8192, 10928, 10940]);
        assert_eq!(merge_grid(7790, run_grid(7790, 12329, c, true, None), None, c), vec![10240, 12288, 12329]);
        assert_eq!(merge_grid(0, run_grid(0, 1519, c, true, Some(1507)), Some(1507), c), vec![1507, 1519]);
        assert_eq!(merge_grid(0, run_grid(0, 5000, c, true, None), None, c), vec![2048, 4096, 5000]);
        assert_eq!(tail_point(Some(6590), 0, 6581, c, true), None);    // not inside the run
        assert_eq!(tail_point(Some(4), 0, 6581, c, true), None);
        assert_eq!(tail_point(None, 0, 6581, c, true), None);
    }

    /// The pre-TP-4X1 rules, verbatim: the flags-off reference every new function must equal.
    mod old {
        pub const WP16_STRIDE: usize = 8192;
        pub fn wp16_due(e: usize, n: usize, c: usize) -> bool {
            c > 0 && e > 0 && e <= n && e % c == 0 && (e % WP16_STRIDE == 0 || e + 2 * c >= n)
        }
        pub fn merge_grid(from: usize, ends: Vec<usize>, tail: Option<usize>, c: usize) -> Vec<usize> {
            let cap = 2 * c.max(1) - 1;
            let mut e = ends;
            let start = |e: &Vec<usize>, i: usize| if i == 0 { from } else { e[i - 1] };
            if let Some(t) = tail {
                if let Some(i) = e.iter().position(|&x| x == t) {
                    if i >= 1 && Some(e[i - 1]) != tail && t - start(&e, i - 1) <= cap { e.remove(i - 1); }
                }
            }
            if from % c.max(1) != 0 && e.len() >= 2 && Some(e[0]) != tail && e[1] - from <= cap {
                e.remove(0);
            }
            e
        }
    }

    /// TP-4X1 flags-off identity: for every C <= 2,048 (every config legal before) the new
    /// due rule, merge cap and merged grid equal the old ones, over a sweep of runs and tails.
    #[test]
    fn tp4x1_flags_off_equals_old_grid() {
        for &c in &[64usize, 256, 512, 1000, 1024, 1500, 2047, 2048] {
            // the cap is the old 2C - 1 under TP and not
            assert_eq!(merge_cap(c, true), 2 * c - 1);
            assert_eq!(merge_cap(c, false), 2 * c - 1);
            for &n in &[1usize, 100, 1519, 2048, 3000, 4096, 6581, 8195, 8300, 20_000, 32_000, 131_772] {
                for e in (0..=n + 3).step_by(1).filter(|e| e % c == 0 || e % 7 == 0) {
                    assert_eq!(wp16_due(e, n, c), old::wp16_due(e, n, c), "due e={e} n={n} c={c}");
                }
                for &from in &[0usize, 7, c, c + 13, 2 * c + 1, 5000, 6573] {
                    if from >= n { continue; }
                    for &tp in &[false, true] {
                        let mut tails = vec![None];
                        for d in [14usize, 12, 9, 200, 1000] { if n > d && n - d > from { tails.push(Some(n - d)); } }
                        for tail in tails {
                            let g = run_grid(from, n, c, true, tail);
                            assert_eq!(merge_grid_cap(from, g.clone(), tail, c, merge_cap(c, tp)),
                                       old::merge_grid(from, g.clone(), tail, c), "from={from} n={n} c={c} tail={tail:?}");
                            assert_eq!(merge_grid(from, g.clone(), tail, c), old::merge_grid(from, g, tail, c));
                        }
                    }
                }
            }
        }
    }

    /// TP-4X1: C = 4,095 under TP caps merged chunks at 4,095 rows (the all-reduce partial buffer
    /// holds 4,096); non-TP keeps 2C - 1. Boundaries at 4,095 multiples, tail split kept.
    #[test]
    fn tp4x1_steady_4095_grid() {
        let c = 4095;
        assert_eq!(merge_cap(c, true), 4095);
        assert_eq!(merge_cap(c, false), 8189);
        assert_eq!(merge_cap(4096, true), 4095);
        assert_eq!(merge_cap(2048, true), 4095);
        let cap = merge_cap(c, true);
        // the served 8K request (n = 8195, message boundary - 8 = 8181): 4 chunks (4095, 4086, 9, 5)
        let g = merge_grid_cap(0, run_grid(0, 8195, c, true, Some(8181)), Some(8181), c, cap);
        assert_eq!(g, vec![4095, 8181, 8190, 8195]);
        // no merged chunk exceeds the cap, whatever the run
        for &n in &[1519usize, 3000, 4095, 4096, 8195, 12_286, 32_000, 128_000, 250_000] {
            for tail in [None, Some(n.saturating_sub(14).max(1))] {
                let tail = tail.filter(|&t| t < n);
                let g = merge_grid_cap(0, run_grid(0, n, c, true, tail), tail, c, cap);
                let mut s = 0;
                for &e in &g { assert!(e - s <= cap && e > s, "n={n} chunk [{s},{e})"); s = e; }
                assert_eq!(s, n);
                assert!(tail.map_or(true, |t| g.contains(&t)));
            }
        }
    }

    /// TP-4X1: stride checkpoints at C > 2,048 follow the crossing rule (a chunk crossing a
    /// multiple of 8,192): the same density as C = 2,048; and = `e % 8192 == 0` for C = 4,096.
    #[test]
    fn tp4x1_stride_rule_at_wide_chunks() {
        let n = 131_072 + 700;
        let due = |c: usize| -> Vec<usize> {
            let ends = run_grid(0, n, c, false, None);
            ends.into_iter().filter(|&e| wp16_due(e, n, c)).collect()
        };
        // C = 4,095: ends 12285, 20475, ... one per crossed multiple of 8,192 (16), plus the last 2
        let d = due(4095);
        assert!(d.starts_with(&[12_285, 20_475]), "{d:?}");
        let stride_only: Vec<usize> = d.iter().copied().filter(|&e| e + 2 * 4095 < n).collect();
        // 8192*k crossed for k = 1..=14 before the last 2 chunks (k = 15, 16 land in them)
        assert_eq!(stride_only.len(), 14, "{stride_only:?}");
        for w in stride_only.windows(2) { assert!((8190..=12_285).contains(&(w[1] - w[0])), "{w:?}"); }
        // the last 2 chunk boundaries are still checkpoints
        assert!(d.contains(&(n / 4095 * 4095)) && d.contains(&(n / 4095 * 4095 - 4095)));
        // C = 4,096 divides 8,192: the crossing rule IS the stride rule
        for e in (4096..=n).step_by(4096) { assert_eq!(stride_hit(e, 4096), e % 8192 == 0); }
        // C = 2,048 unchanged (crossing == stride for a divisor of 8,192)
        for e in (2048..=n).step_by(2048) { assert_eq!(stride_hit(e, 2048), e % 8192 == 0); }
    }

    const RMIN: usize = 17; // recon_min_rows() default: chunks below it are the replicated eager small-M class

    /// TP-4X1 absorb off == the pre-TP-4X1 grid for every run (tracked: merge_grid; plain: run_grid).
    #[test]
    fn tp4x1_absorb_off_equals_old_grid() {
        for &c in &[256usize, 1024, 2048] {
            for &n in &[1usize, 17, 100, 1519, 2048, 3000, 8195, 8300, 32_000, 128_000] {
                for &from in &[0usize, 13, c, 5000, 8181] {
                    if from >= n { continue; }
                    let mut tails = vec![None];
                    for d in [14usize, 12, 9, 200, 1000] { if n > d && n - d > from { tails.push(Some(n - d)); } }
                    for tail in tails {
                        let cap = merge_cap(c, true);
                        let new = plan_grid(from, n, c, true, tail, cap, None, true);
                        assert_eq!(new, old::merge_grid(from, run_grid(from, n, c, true, tail), tail, c));
                        assert_eq!(plan_grid(from, n, c, false, None, cap, None, false), run_grid(from, n, c, false, None));
                    }
                }
            }
        }
    }

    /// TP-4X1 absorb: the served 8K request (n = 8195, tail 8181) loses the 11-row replicated chunk;
    /// the tail checkpoint and n never move; the 8192 checkpoint is dropped only because the tail
    /// checkpoint sits 11 < 17 rows before it.
    #[test]
    fn tp4x1_absorb_served_8k() {
        let (n, t) = (8195, 8181);
        let cap = merge_cap(2048, true);
        let old = plan_grid(0, n, 2048, true, Some(t), cap, None, true);
        assert_eq!(old, vec![2048, 4096, 8181, 8192, 8195]); // 2048, 2048, 4085, 11, 3
        assert_eq!(plan_grid(0, n, 2048, true, Some(t), cap, Some(RMIN), true), vec![2048, 4096, 8181, 8195]);
        // steady 4095 chunks
        let cap = merge_cap(4095, true);
        assert_eq!(plan_grid(0, n, 4095, true, Some(t), cap, None, true), vec![4095, 8181, 8190, 8195]);
        assert_eq!(plan_grid(0, n, 4095, true, Some(t), cap, Some(RMIN), true), vec![4095, 8181, 8195]);
        // a middle ragged chunk followed by a real one: 8181 | 8192 | 8300
        assert_eq!(plan_grid(0, 8300, 2048, true, Some(8181), merge_cap(2048, true), Some(RMIN), true),
                   vec![2048, 4096, 8181, 8300]);
        // no tail checkpoint: the stride checkpoint 8192 is NOT dominated -> kept (cache granularity wins);
        // an untracked run has no checkpoints, so its 3-row remainder folds into [6144, 8195)
        let cap = merge_cap(2048, true);
        assert_eq!(plan_grid(0, n, 2048, true, None, cap, Some(RMIN), true), vec![2048, 4096, 6144, 8192, 8195]);
        assert_eq!(plan_grid(0, n, 2048, false, None, cap, Some(RMIN), false), vec![2048, 4096, 6144, 8195]);
        // the inherent final chunk after the tail checkpoint (n - t = 14 rows) stays: tail never moves
        assert_eq!(plan_grid(0, 3000, 2048, true, Some(2986), cap, Some(RMIN), true), vec![2986, 3000]);
        assert_eq!(plan_grid(0, 1519, 2048, true, None, cap, Some(RMIN), true), vec![1519]);
        // a lone tiny prompt is never merged away
        assert_eq!(plan_grid(0, 10, 2048, true, None, cap, Some(RMIN), true), vec![10]);
        // the cap holds: a ragged chunk does not fold into a full chunk
        assert_eq!(plan_grid(0, 4099, 4095, false, None, merge_cap(4095, true), Some(RMIN), false), vec![4095, 4099]);
    }

    /// TP-4X1: tail-checkpoint availability at C = 4,095 equals today's C = 2,048 for every prompt and
    /// run start (grain min(C, 2048)); 2-4K-row prompts keep their message-boundary checkpoint.
    #[test]
    fn tp4x1_tail_grain_keeps_availability() {
        for n in (1..=60usize).map(|k| k * 211 + 3).chain([8195, 32_000, 128_000, 250_000]) {
            for &from in &[0usize, 9, 2048, 4095, 8181] {
                for extra in [false, true] {
                    for d in [6usize, 40] {
                        let b = n.saturating_sub(d);
                        let a = tail_point(Some(b), from, n, 2048, extra);
                        let w = tail_point(Some(b), from, n, 4095, extra);
                        assert_eq!(a, w, "n={n} from={from} extra={extra} d={d}");
                    }
                }
            }
        }
        // 3,000-row prompt: boundary tail kept at C = 4,095 (grid [2986, 3000], today's cost), not dropped
        assert_eq!(tail_point(Some(2994), 0, 3000, 4095, false), Some(2986));
    }

    /// TP-4X1 invariants over a sweep: ends strictly increase to n, tail and n survive, every chunk <= cap,
    /// never more chunks than without absorb, and a lost checkpoint is always within RMIN - 1 rows after the
    /// tail checkpoint (so a resume loses < RMIN rows).
    #[test]
    fn tp4x1_absorb_invariants() {
        for &c in &[1024usize, 2048, 4095] {
            let cap = merge_cap(c, true);
            for n in (1..=40).chain((1..=120).map(|k| k * 311 + 7)).chain([8195, 8300, 32_000, 128_000, 250_000]) {
                for &from in &[0usize, 9, c, c + 5, 8181] {
                    if from >= n { continue; }
                    let mut tails = vec![None];
                    for d in [14usize, 12, 9, 5, 200] { if n > d && n - d > from { tails.push(Some(n - d)); } }
                    for tail in tails {
                        for &ck in &[true, false] {
                            let realign = ck;
                            let base = plan_grid(from, n, c, realign, tail.filter(|_| realign), cap, None, ck);
                            let tl = tail.filter(|_| realign);
                            let g = plan_grid(from, n, c, realign, tl, cap, Some(RMIN), ck);
                            let mut s = from;
                            for &e in &g {
                                assert!(e > s && e <= n, "{g:?}");
                                assert!(e - s <= cap.max(c).max(base.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0)).max(base[0] - from), "cap {g:?}");
                                s = e;
                            }
                            assert_eq!(s, n);
                            assert!(g.len() <= base.len());
                            if let Some(t) = tl { if base.contains(&t) { assert!(g.contains(&t), "tail lost {g:?}"); } }
                            if ck {
                                for &b in base.iter().filter(|&&b| b < n && !g.contains(&b)) {
                                    if wp16_due(b, n, c) {
                                        let t = tl.expect("a due checkpoint was dropped without a tail");
                                        assert!(t < b && b - t < RMIN, "lost due ckpt {b} (tail {t})");
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Prints the TP-4X1 grid tables (cargo test --release --lib tp4x1_print_tables -- --ignored --nocapture).
    #[test]
    #[ignore]
    fn tp4x1_print_tables() {
        let tail_of = |n: usize, c: usize| tail_point(Some(n - 6), 0, n, c, false); // served: message boundary - 8 = n - 14
        for &n in &[1519usize, 3000, 8195, 32_000, 128_000, 250_000] {
            for (name, c, absorb) in [("A today C=2048", 2048usize, None), ("B C=4095", 4095, None),
                                      ("C C=2048+absorb", 2048, Some(RMIN)), ("D C=4095+absorb", 4095, Some(RMIN))] {
                let tail = tail_of(n, c);
                let cap = merge_cap(c, true);
                let g = plan_grid(0, n, c, true, tail, cap, absorb, true);
                let mut rows = Vec::new();
                let mut s = 0;
                for &e in &g { rows.push(e - s); s = e; }
                let ragged = rows.iter().filter(|&&r| r < RMIN).count();
                let ck: Vec<usize> = g.iter().copied().filter(|&e| wp16_due(e, n, c) || tail == Some(e)).collect();
                let shown = if rows.len() > 10 { format!("{:?} .. {:?}", &rows[..3], &rows[rows.len() - 4..]) } else { format!("{rows:?}") };
                let cks = if ck.len() > 6 { format!("{} (last {:?})", ck.len(), &ck[ck.len() - 4..]) } else { format!("{ck:?}") };
                println!("TABLE n={n:>7} {name:<16} tail={tail:?} chunks={:>3} ragged={ragged} rows={shown} | ckpts {cks}", rows.len());
            }
        }
    }

    fn take(st: &mut CkptStore<u32>, slot: usize, pos: usize, next: &mut u32) -> bool {
        let mut n = *next;
        let b = st.acquire(|| { n += 1; Ok(n) });
        *next = n;
        match b { Some(b) => { st.insert(slot, pos, b); true } None => false }
    }

    #[test]
    fn resume_point_is_deepest_checkpoint_within_lcp() {
        let mut st: CkptStore<u32> = CkptStore::new(2, 16);
        let mut id = 0;
        let a: Vec<u32> = (0..10_000).collect();
        st.begin_aligned(0, 0, 2048);
        st.extend(0, &a[..9999]);
        for p in [2048, 4096, 6144, 8192] { assert!(take(&mut st, 0, p, &mut id)); }
        // B shares 7000 tokens
        let mut b = a[..7000].to_vec();
        b.extend(50_000..53_000u32);
        assert_eq!(st.resume_point(0, &b, b.len() - 1, 2048), (6144, 7000));
        // keyed by C
        assert_eq!(st.resume_point(0, &b, b.len() - 1, 4096).0, 0);
        // bounded by plen-1: B = A[..4097] resumes at 4096 (the seam token stays a decode step)
        assert_eq!(st.resume_point(0, &a[..4097], 4096, 2048).0, 4096);
        assert_eq!(st.resume_point(0, &a[..4096], 4095, 2048).0, 2048);
        // no shared prefix / other slot
        assert_eq!(st.resume_point(0, &[7, 7, 7], 2, 2048).0, 0);
        assert_eq!(st.resume_point(1, &b, b.len() - 1, 2048).0, 0);
        // resume at 6144: drops 8192, keeps <= 6144, cuts the prefix
        assert_eq!(st.begin_aligned(0, 6144, 2048), 1);
        assert_eq!(st.positions(0), vec![2048, 4096, 6144]);
        assert_eq!(st.toks_len(0), 6144);
        st.extend(0, &b[6144..b.len() - 1]);
        assert_eq!(st.resume_point(0, &b, b.len() - 1, 2048).0, 6144);
        // the prompt-end prefix hit (unaligned) keeps everything at or below its resume point
        st.begin_unaligned(0, b.len() - 1);
        assert_eq!(st.positions(0), vec![2048, 4096, 6144]);
        // a fresh aligned run from 0 drops the slot's checkpoints into the free list
        assert_eq!(st.begin_aligned(0, 0, 2048), 3);
        assert_eq!(st.live(), 0);
        assert_eq!(st.n_alloc, 4);
        assert!(take(&mut st, 0, 2048, &mut id));
        assert_eq!(st.n_alloc, 4, "reused a freed buffer");
    }

    #[test]
    fn lru_cap_evicts_least_recently_used_across_slots() {
        let mut st: CkptStore<u32> = CkptStore::new(2, 3);
        let mut id = 0;
        st.begin_aligned(0, 0, 2048);
        st.extend(0, &vec![1u32; 8192]);
        st.begin_aligned(1, 0, 2048);
        st.extend(1, &vec![2u32; 8192]);
        assert!(take(&mut st, 0, 2048, &mut id));
        assert!(take(&mut st, 1, 2048, &mut id));
        assert!(take(&mut st, 0, 4096, &mut id));
        st.touch(0, 2048); // slot 0 @2048 is now more recent than slot 1 @2048
        assert!(take(&mut st, 1, 4096, &mut id)); // cap 3: evicts slot 1 @2048 (LRU)
        assert_eq!(st.n_alloc, 3);
        assert_eq!(st.positions(0), vec![2048, 4096]);
        assert_eq!(st.positions(1), vec![4096]);
        assert!(take(&mut st, 1, 6144, &mut id)); // evicts slot 0 @4096
        assert_eq!(st.positions(0), vec![2048]);
        // cap 0 never takes one
        let mut z: CkptStore<u32> = CkptStore::new(1, 0);
        z.begin_aligned(0, 0, 2048);
        assert!(!take(&mut z, 0, 2048, &mut id));
    }

    #[test]
    fn allocation_failure_freezes_the_cap() {
        let mut st: CkptStore<u32> = CkptStore::new(1, 8);
        st.begin_aligned(0, 0, 2048);
        assert_eq!(st.acquire(|| Ok(1)), Some(1));
        st.insert(0, 2048, 1);
        assert_eq!(st.acquire(|| anyhow::bail!("oom")), Some(1)); // falls back to eviction
        st.insert(0, 4096, 1);
        assert_eq!(st.acquire(|| Ok(9)), Some(1)); // frozen: no new allocation
        assert_eq!(st.n_alloc, 1);
    }

    #[test]
    fn a_changed_chunk_width_or_short_prefix_clears_the_slot() {
        let mut st: CkptStore<u32> = CkptStore::new(1, 8);
        let mut id = 0;
        st.begin_aligned(0, 0, 2048);
        st.extend(0, &vec![3u32; 4096]);
        assert!(take(&mut st, 0, 2048, &mut id));
        assert_eq!(st.begin_aligned(0, 0, 4096), 1);
        assert_eq!(st.live(), 0);
        st.extend(0, &vec![3u32; 100]);
        // a resume position beyond the aligned prefix is inconsistent: clear, never trust it
        assert!(take(&mut st, 0, 100, &mut id));
        assert_eq!(st.begin_aligned(0, 4096, 4096), 1);
        assert_eq!(st.toks_len(0), 0);
    }

    #[test]
    fn layout_offsets_tile_the_buffer() {
        let l = CkptLayout { n_gdn: 3, per_conv: 10, per_s: 100, ple_words: 7, carry_words: 5 };
        assert_eq!(l.conv_off(2), 20);
        assert_eq!(l.s_off(0), 30);
        assert_eq!(l.s_off(2), 230);
        assert_eq!(l.ple_off(), 330);
        assert_eq!(l.carry_off(), 337);
        assert_eq!(l.words(), 342);
        assert_eq!(l.bytes(), 1368);
    }
}
