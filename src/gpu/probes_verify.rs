//! `impl GpuModel`: verify-side probes and benches (`probe_ppsplit`, `bench_verify`, `verify_state_diff`, reject-path probes,
//! bucket bit-exactness). Probes only; not on the serving path. Map: `mod.rs`.

use super::*;

impl GpuModel {
    /// PP-prefill (PLAN/14) on-box split validation: the layer-range refactor must be EXACT.
    ///
    /// Ground truth = the monolithic prefill (all 64 layers, one call) into slot 0. Treatment =
    /// the same window split at `split` (default 32): layers [0, split) into slot 1, then layers
    /// [split, 64) into slot 1 CONTINUING from the returned residual — the exact on-box analogue
    /// of the 2-box pipeline, minus the bf16 wire copy (which is value-preserving by
    /// construction). Asserts, on success symptoms (AGENTS §3):
    ///   1. first token identical;
    ///   2. final pre-norm hidden (the MTP-prime artifact) byte-identical;
    ///   3. per-slot post-prefill state byte-identical: KV (both caches, all FA layers), GDN
    ///      s_state and conv state (all GDN layers) — the decode handoff contract.
    /// Prints per-half timings (the balance measurement for the split-index choice, PLAN/14 §2).
    pub fn probe_ppsplit(&self, pool: &mut Pool, state: &mut BatchGpuState, prompt: &[u32],
                         kv_stride: usize, split: usize, state_slots: usize) -> bool {
        let h = self.cfg.hidden_size;
        let n = prompt.len();

        // Ground truth: monolithic.
        self.zero_slot_state(state, 0, kv_stride);
        let t0 = std::time::Instant::now();
        let (tok_m, h_m) = self.prefill_batch(pool, prompt, state, 0, kv_stride, 0);
        self.sync_stream();
        let t_mono = t0.elapsed().as_secs_f64();

        // Treatment: split at `split`, continuing from the lower range's residual.
        self.zero_slot_state(state, 1, kv_stride);
        let t0 = std::time::Instant::now();
        let (_, h_lo) = self.prefill_batch_range(pool, prompt, state, 1, kv_stride, 0,
                                                 0, split, None);
        self.sync_stream();
        let t_lo = t0.elapsed().as_secs_f64();
        let t0 = std::time::Instant::now();
        let (tok_s, h_hi) = self.prefill_batch_range(pool, prompt, state, 1, kv_stride, 0,
                                                     split, self.layers.len(), Some(h_lo));
        self.sync_stream();
        let t_hi = t0.elapsed().as_secs_f64();

        // 1. First token.
        let tok_ok = tok_m == tok_s;
        // 2. Final pre-norm hidden, byte-identical.
        // bf16 dtoh as raw bytes (slice reinterpret — compare BITS, not f32 values).
        let bits = |b: &B| -> Vec<u8> {
            let mut v = vec![0u8; b.len() * 2];
            unsafe { cudarc::driver::result::memcpy_dtoh_sync(&mut v, *b.device_ptr() as u64) }
                .expect("ppsplit hidden dtoh");
            v
        };
        let hm_b = bits(&h_m);
        let hh_b = bits(&h_hi);
        let hidden_ok = hm_b == hh_b;
        pool.release_bf16(h_m, h * n);
        pool.release_bf16(h_hi, h * n);

        // 3. Post-prefill slot state, byte-identical. Per-slot regions mirror the launch offsets
        // exactly: KV = [slot, nkv, kv_stride, elems] per cache (K and V row sizes DIVERGE under
        // k8v4 — each cache gets its own length), s_state/conv = [slot, ...] f32. Both slots ran
        // the same window from the same zeroed start, so every written byte must match.
        let mut state_ok = true;
        let mut bytes_cmp = 0usize;
        let read_range = |base: u64, off: u64, len: usize| -> Vec<u8> {
            let mut v = vec![0u8; len];
            unsafe { cudarc::driver::result::memcpy_dtoh_sync(&mut v, base + off) }
                .expect("ppsplit state dtoh");
            v
        };
        for li in 0..self.layers.len() {
            let nkv = self.eff_num_kv_heads();
            if let (Some(kc), Some(vc)) = (state.k_cache[li].as_ref(), state.v_cache[li].as_ref()) {
                let per_k = nkv * kv_stride * self.kv_k_elems_per_pos() * 2;
                let per_v = nkv * kv_stride * self.kv_v_elems_per_pos() * 2;
                for (buf, per) in [(*kc.device_ptr() as u64, per_k), (*vc.device_ptr() as u64, per_v)] {
                    let a = read_range(buf, 0, per);
                    let b = read_range(buf, per as u64, per);
                    state_ok &= a == b;
                    bytes_cmp += per * 2;
                }
            }
            if let Some(ss) = state.s_state[li].as_ref() {
                let per = (ss.len() / state_slots) * 4;
                let a = read_range(*ss.device_ptr() as u64, 0, per);
                let b = read_range(*ss.device_ptr() as u64, per as u64, per);
                state_ok &= a == b;
                bytes_cmp += per * 2;
            }
            if let Some(cs) = state.conv_state[li].as_ref() {
                let per = (cs.len() / state_slots) * 4;
                let a = read_range(*cs.device_ptr() as u64, 0, per);
                let b = read_range(*cs.device_ptr() as u64, per as u64, per);
                state_ok &= a == b;
                bytes_cmp += per * 2;
            }
        }
        println!("[ppsplit] n={n} split={split} layers={}  mono={:.1}ms  lo={:.1}ms  hi={:.1}ms  lo+hi={:.1}ms",
                 self.layers.len(), t_mono * 1e3, t_lo * 1e3, t_hi * 1e3, (t_lo + t_hi) * 1e3);
        println!("[ppsplit] token={}  hidden_bytes={}  state_bytes={}  -> {}",
                 if tok_ok { "IDENTICAL" } else { { state_ok = false; "DIVERGED" } },
                 if hidden_ok { "IDENTICAL" } else { { state_ok = false; "DIVERGED" } },
                 bytes_cmp,
                 if state_ok { "PP-SPLIT EXACT (token+hidden+state byte-identical)" } else { "PP-SPLIT MISMATCH" });
        tok_ok && hidden_ok && state_ok
    }

    /// Lossless-correctness probe for `verify_forward` (the MTP verify primitive). Prefills the same
    /// prompt into slots 0 and 1 (identical starting state), runs `depth` sequential greedy decodes
    /// on slot 0 (ground truth), then runs `verify_forward` on slot 1 over the first `depth`
    /// ground-truth tokens. The verify predictions must equal the ground-truth tokens shifted by one:
    /// verify token at position plen+i predicts the token at plen+i+1, so preds[i] must equal
    /// seq_tokens[i+1]. This is the non-negotiable lossless gate — the K-token append path (using
    /// delta_step_prefill + gqa_attn_prefill) must match the 1-token decode path (delta_step_b +
    /// gqa_attn_splitk) token-for-token. Returns (seq_tokens, preds, per_position_match).
    pub fn bench_verify(&self, pool: &mut Pool, state: &mut BatchGpuState, prompt: &[u32],
                        kv_stride: usize, depth: usize) -> (Vec<u32>, Vec<u32>, Vec<bool>) {
        let max_batch = 2usize;
        let mut bufs = self.new_decode_buffers(max_batch);

        // Prefill the same prompt into slots 0 and 1 (identical starting state).
        let h = self.cfg.hidden_size;
        self.zero_slot_state(state, 0, kv_stride);
        let (s0a, h0) = self.prefill_batch(pool, prompt, state, 0, kv_stride, 0);
        pool.release_bf16(h0, h * prompt.len());
        self.zero_slot_state(state, 1, kv_stride);
        let (s0b, h1) = self.prefill_batch(pool, prompt, state, 1, kv_stride, 0);
        pool.release_bf16(h1, h * prompt.len());
        if s0a != s0b { crate::gpu::dump_dual_prefill_diff(prompt.len()); }
        assert_eq!(s0a, s0b, "prefill divergence between slot 0 and slot 1");

        let plen = prompt.len();
        let mut seq_tokens = vec![s0a];   // token at position plen

        // Sequential greedy decode on slot 0 (batch=1): produce ground-truth tokens.
        let mut pos = plen;
        for _ in 0..depth {
            let tok = *seq_tokens.last().unwrap() as i32;
            let toks_i32 = vec![tok, 0];
            let pos_i32 = vec![pos as i32, 0];
            self.dev.htod_sync_copy_into(&toks_i32, &mut bufs.tokens_dev).unwrap();
            self.dev.htod_sync_copy_into(&pos_i32, &mut bufs.pos_dev).unwrap();
            // slot_ids_dev stays identity [0,1] from new_decode_buffers; lane 0 -> physical slot 0.
            self.dev.synchronize().unwrap();
            let max_pc = pos + 1;
            let next = self.forward_decode(pool, &mut bufs, state, kv_stride, max_pc, 1);
            seq_tokens.push(next[0]);
            pos += 1;
        }

        // Verify on slot 1: feed [s_0..s_{depth-1}] at positions plen..plen+depth-1.
        let verify_input: Vec<u32> = seq_tokens[..depth].to_vec();
        let (preds, vout) = self.verify_forward(pool, &verify_input, state, 1, kv_stride, plen, None, None);
        pool.release_bf16(vout, h * depth);

        // preds[i] must equal seq_tokens[i+1] (token predicted at pos plen+i is the one at plen+i+1).
        let matches: Vec<bool> = (0..depth).map(|i| preds[i] == seq_tokens[i + 1]).collect();
        (seq_tokens, preds, matches)
    }

    /// PLAN/25 Phase 0 — the verify-WIDTH cost curve (bench-only, `--bench-verify
    /// --time-widths`): per-requested-width median of ONE `verify_forward(W columns)` vs the
    /// per-token median of sequential `forward_decode`, measured ALTERNATING within each rep
    /// (same-window alternation, the E1b lesson — never compare across windows). Every leg
    /// starts from a fresh identical prefill state (slot 0 for decode, slot 1 for verify), so
    /// legs are independent and the verify legs are real chain verifies (lossless path, graphs
    /// as shipped). Returns (width, decode_ms_per_tok_median, verify_ms_median).
    pub fn time_verify_widths(&self, pool: &mut Pool, state: &mut BatchGpuState, prompt: &[u32],
                              kv_stride: usize, widths: &[usize], reps: usize)
                              -> anyhow::Result<Vec<(usize, f64, f64)>> {
        let h = self.cfg.hidden_size;
        let maxw = *widths.iter().max().expect("time_verify_widths: no widths");
        let mut bufs = self.new_decode_buffers(2);
        let plen = prompt.len();
        let mut dec_ms: Vec<f64> = Vec::new();
        let mut ver_ms: Vec<Vec<f64>> = vec![Vec::new(); widths.len()];
        for _ in 0..reps {
            // ---- decode leg (slot 0): maxw sequential greedy steps, per-step wall. ----
            self.zero_slot_state(state, 0, kv_stride);
            let (s0, h0) = self.prefill_batch(pool, prompt, state, 0, kv_stride, 0);
            pool.release_bf16(h0, h * plen);
            let mut seq = vec![s0];
            let mut pos = plen;
            for _ in 0..maxw {
                let tok = *seq.last().unwrap() as i32;
                self.dev.htod_sync_copy_into(&[tok, 0], &mut bufs.tokens_dev).unwrap();
                self.dev.htod_sync_copy_into(&[pos as i32, 0], &mut bufs.pos_dev).unwrap();
                self.dev.synchronize().unwrap();
                let t0 = std::time::Instant::now();
                let nx = self.forward_decode(pool, &mut bufs, state, kv_stride, pos + 1, 1);
                self.dev.synchronize().unwrap();
                dec_ms.push(t0.elapsed().as_secs_f64() * 1e3);
                seq.push(nx[0]);
                pos += 1;
            }
            // ---- verify legs (slot 1): one verify_forward(W) per width, fresh state each. ----
            for (wi, &w) in widths.iter().enumerate() {
                self.zero_slot_state(state, 1, kv_stride);
                let (_s1, h1) = self.prefill_batch(pool, prompt, state, 1, kv_stride, 0);
                pool.release_bf16(h1, h * plen);
                let inputs: Vec<u32> = seq[..w].to_vec();
                self.dev.synchronize().unwrap();
                let t0 = std::time::Instant::now();
                let (_p, vout) = self.verify_forward(pool, &inputs, state, 1, kv_stride, plen, None, None);
                self.dev.synchronize().unwrap();
                ver_ms[wi].push(t0.elapsed().as_secs_f64() * 1e3);
                pool.release_bf16(vout, h * w);
            }
        }
        let med = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.total_cmp(b));
            v[v.len() / 2]
        };
        Ok(widths.iter().enumerate()
            .map(|(wi, &w)| (w, med(&mut dec_ms), med(&mut ver_ms[wi])))
            .collect())
    }

    /// State-divergence probe: prefills both slots identically, advances slot 0 by ONE
    /// `verify_forward(N tokens)` call and slot 1 by N individual `forward_decode` calls, then
    /// returns the max abs f32 diff over all GDN `s_state` (recurrent) entries between the two slots.
    /// If verify_forward (delta_step_prefill) and forward_decode (delta_step_b) produce identical
    /// recurrent state, the diff is 0. A nonzero diff localizes the divergence to the GDN recurrence
    /// path (or the projections feeding it) — distinguishing a real kernel bug from cuBLAS N=1-vs-N
    /// rounding. `n` is the verify batch size (1 = single-token append, 2 = two-token append).
    /// Phase-7: the GDN decode-vs-verify state probe on the SHARDED path. Both ranks run this in
    /// SPMD lockstep (every forward's all-reduces pair up); each compares its own state slices.
    pub fn probe_state_tp(&self, pool: &mut Pool, prompt: &[u32], kv_stride: usize, max_n: usize) -> bool {
        let plen = prompt.len().min(512);
        let prompt: &[u32] = &prompt[..plen];
        // Phase-8 A1: the probe MUST run eager. The verify CUDA-graph capture trips the TP
        // transport tripwire mid-probe (Phase-7 N=4 abort: device epoch != gpu_ready — the capture
        // records the barrier kernels instead of executing them, which desyncs the SPMD lockstep),
        // and the probe's job is the numerics of the EAGER path anyway. Setting the env var here
        // covers both ranks (SPMD: the node runs this same function). Not a §1b feature disable:
        // graphs return for serving the moment the probe exits the process.
        crate::opts::set(crate::opt!("no-verify-graph"), "1");
        // Phase-8 A2: arm the per-layer residual tap (--state-tap=0 disarms, for A/B).
        if crate::opts::var(crate::opt!("state-tap")).map_or(true, |v| v != "0") {
            state_tap_arm();
        }
        let mut all_ok = true;
        println!("=== TP GDN state probe (world={} rank={}): verify vs decode, plen={plen} (eager verify) ===",
                 self.tp_world, self.tp_rank);
        for n in 1..=max_n {
            // Phase-8 A2 root cause: this WAS n.max(2) slots, but verify_state_diff's per-slot
            // extent is len/2 — with >=3 slots the compare read slot-1 data at a 1.5-slot offset
            // (and zeroed slot-2 tail), reporting a PHANTOM MISMATCH for every N>=3. The probe
            // uses exactly slots {0=verify, 1=decode}; 2 slots matches run_probe_state (world=1).
            // The trunk state contract itself was clean at every N.
            let mut state = self.new_batch_state(2, 2, kv_stride);
            let d = self.verify_state_diff(pool, &mut state, prompt, kv_stride, n);
            let ok = d == 0.0;
            all_ok &= ok;
            println!("  N={n}: verify_state_diff = {d:.7} {}", if ok { "EXACT" } else { "MISMATCH" });
        }
        state_tap_disarm();
        println!("{}", if all_ok { "PROBE-STATE-TP: PASS" } else { "PROBE-STATE-TP: FAIL" });
        all_ok
    }

    pub fn verify_state_diff(&self, pool: &mut Pool, state: &mut BatchGpuState, prompt: &[u32],
                             kv_stride: usize, n: usize) -> f32 {
        let h = self.cfg.hidden_size;
        let mut bufs = self.new_decode_buffers(2);
        self.zero_slot_state(state, 0, kv_stride);
        let (s0a, h0) = self.prefill_batch(pool, prompt, state, 0, kv_stride, 0);
        pool.release_bf16(h0, h * prompt.len());
        self.zero_slot_state(state, 1, kv_stride);
        let (s0b, h1) = self.prefill_batch(pool, prompt, state, 1, kv_stride, 0);
        pool.release_bf16(h1, h * prompt.len());
        assert_eq!(s0a, s0b);

        let plen = prompt.len();
        // Generate n ground-truth tokens by decoding on slot 0 first (to know the inputs), then we
        // will RE-zero slot 0 and use verify there. Decode n tokens on slot 0 to get the token list.
        let mut toks = vec![s0a];
        let mut pos = plen;
        for _ in 0..n {
            let t = *toks.last().unwrap() as i32;
            self.dev.htod_sync_copy_into(&[t, 0], &mut bufs.tokens_dev).unwrap();
            self.dev.htod_sync_copy_into(&[pos as i32, 0], &mut bufs.pos_dev).unwrap();
            self.dev.synchronize().unwrap();
            let nx = self.forward_decode(pool, &mut bufs, state, kv_stride, pos + 1, 1);
            toks.push(nx[0]);
            pos += 1;
        }
        // Now slot 0 has been advanced by decode; RE-zero it and re-prefill so it's clean, then
        // advance it by ONE verify_forward of the same n tokens.
        let inputs: Vec<u32> = toks[..n].to_vec();
        self.zero_slot_state(state, 0, kv_stride);
        let (_s, hh) = self.prefill_batch(pool, prompt, state, 0, kv_stride, 0);
        pool.release_bf16(hh, h * prompt.len());
        let tap = state_tap_active();
        if tap { let _ = state_tap_take(); }   // drop the generation-decode records
        let (_p, vout) = self.verify_forward(pool, &inputs, state, 0, kv_stride, plen, None, None);
        pool.release_bf16(vout, h * n);
        let ver_tap = if tap { state_tap_take() } else { Vec::new() };
        // Phase-8 bracket: slot-0 state snapshot immediately after the verify returns.
        let snap_s1 = if tap {
            state.s_state.iter().map(|b| b.as_ref().map(|s| self.dev.dtoh_sync_copy(s).unwrap()))
                .collect::<Vec<_>>()
        } else { Vec::new() };
        // Phase-8 bracket (in-verify tail): the mid-verify GDN taps hold slot 0 right after each
        // layer's delta commit; snap_s1 holds slot 0 at verify return. A difference on layer li
        // means something between layer li's delta and the verify's return corrupted it.
        if tap && n <= 4 {
            let vs: Vec<&Vec<Vec<f32>>> = ver_tap.iter().filter(|c| c.tag == "verify").map(|c| &c.s).collect();
            if let Some(vss) = vs.first() {
                let mut gi = 0usize;
                let mut any = false;
                for (li, s1) in snap_s1.iter().enumerate() {
                    if !matches!(self.cfg.layer_types.get(li), Some(LayerType::LinearAttention)) { continue; }
                    let (Some(s1), Some(tap_s)) = (s1, vss.get(gi)) else { gi += 1; continue; };
                    let per = s1.len() / 2;
                    let mut w = 0f32;
                    for j in 0..per { let dd = (s1[j] - tap_s[j]).abs(); if dd > w { w = dd; } }
                    if w > 0.0 {
                        eprintln!("    [bracket] N={n} layer {li:>2} (GDN): slot-0 changed during the VERIFY TAIL after its GDN tap, worst={w:.6e}");
                        any = true;
                    }
                    gi += 1;
                }
                if !any { eprintln!("    [bracket] N={n}: slot-0 s_state clean through the verify tail (all GDN layers)"); }
            }
        }

        // Slot 1 was advanced only by the prefill above (the decode loop used lane 0 → slot 0, then
        // we re-zeroed slot 0). Advance slot 1 by n decodes with the same tokens.
        self.dev.htod_sync_copy_into(&[1i32, 0], &mut bufs.slot_ids_dev).unwrap();
        let mut s1pos = plen;
        // Phase-8 per-step watch: does slot_ids stay [1,0], and which decode step first corrupts
        // slot-0's GDN state (layer 0)? If slot 0 changes with slot_ids == [1,0], a decode kernel
        // is writing outside its lane; if slot_ids drifts, the writer is a stray host copy.
        let watch = tap && n >= 2;
        let mut prev0 = if watch { self.dev.dtoh_sync_copy(state.s_state[0].as_ref().unwrap()).unwrap() } else { Vec::new() };
        for i in 0..n {
            let t = inputs[i] as i32;
            self.dev.htod_sync_copy_into(&[t, 0], &mut bufs.tokens_dev).unwrap();
            self.dev.htod_sync_copy_into(&[s1pos as i32, 0], &mut bufs.pos_dev).unwrap();
            if watch {
                let sids = self.dev.dtoh_sync_copy(&bufs.slot_ids_dev).unwrap();
                eprintln!("    [step] N={n} i={i} tok={t} pos={s1pos} slot_ids={sids:?}");
            }
            self.dev.synchronize().unwrap();
            let _nx = self.forward_decode(pool, &mut bufs, state, kv_stride, s1pos + 1, 1);
            s1pos += 1;
            if watch {
                let cur = self.dev.dtoh_sync_copy(state.s_state[0].as_ref().unwrap()).unwrap();
                let per = cur.len() / 2;
                let (mut w, mut fi) = (0f32, 0usize);
                for j in 0..per { let dd = (cur[j] - prev0[j]).abs(); if dd > w { w = dd; fi = j; } }
                if w > 0.0 {
                    let lin_nh0 = self.eff_lin_v_heads();
                    let (kdd, vdd) = (self.cfg.lin_k_dim, self.cfg.lin_v_dim);
                    let head = fi / (kdd * vdd);
                    let rem = fi - head * kdd * vdd;
                    eprintln!("    [step] N={n} i={i} done: slot-0 s_state[0] delta this step = {w:.6e} at head={head} row={} col={} old={} new={} slot1_now={}",
                        rem / vdd, rem - (rem / vdd) * vdd, prev0[fi], cur[fi], cur[per + fi]);
                } else {
                    eprintln!("    [step] N={n} i={i} done: slot-0 s_state[0] delta this step = 0.000000e0");
                }
                prev0 = cur;
            }
        }
        drop(bufs);
        let dec_tap = if tap { state_tap_take() } else { Vec::new() };
        // Phase-8 bracket: second snapshot after the decode leg. If slot 0 changed between the
        // two snapshots, the DECODE leg is the corruptor (writes outside slot 1). If slot 0 is
        // stable but already differs from the mid-forward taps, the VERIFY corrupted its own
        // commit after its last GDN-layer tap (the FA/logits tail).
        if tap {
            let mut rep = false;
            for (li, s1) in snap_s1.iter().enumerate() {
                let Some(s1) = s1 else { continue };
                let s2 = self.dev.dtoh_sync_copy(state.s_state[li].as_ref().unwrap()).unwrap();
                let per = s2.len() / 2;
                let mut w = 0f32;
                for j in 0..per { let dd = (s1[j] - s2[j]).abs(); if dd > w { w = dd; } }
                if w > 0.0 {
                    let ty = if matches!(self.cfg.layer_types.get(li), Some(LayerType::LinearAttention)) { "GDN" } else { "FA" };
                    eprintln!("    [bracket] N={n} layer {li:>2} ({ty}): SLOT-0 s_state changed during the decode leg, worst={w:.6e}");
                    rep = true;
                }
            }
            if !rep { eprintln!("    [bracket] N={n}: slot-0 s_state STABLE across the decode leg (all layers)"); }
        }

        // Compare slot 0 (verify path) vs slot 1 (decode path): s_state, conv_state, AND k/v cache.
        // All three must be ~0 for verify_forward to be a bit-identical stand-in for decode (the
        // lossless-MTP contract); s_state alone is insufficient.
        // TP: the state buffers are allocated at FULL head shape (the head-proof red zone), so the
        // per-slot extent is each buffer's own half-length, not the global-config product.
        let per_slot = state.s_state.iter().flatten().next().map(|b| b.len() / 2).unwrap_or(0);
        let per_slot_conv = state.conv_state.iter().flatten().next().map(|b| b.len() / 2).unwrap_or(0);
        let _ = self.cfg.num_kv_heads;
        let _ = kv_stride;
        let mut worst_s = 0.0f32; let mut worst_conv = 0.0f32; let mut worst_kv = 0.0f32;
        let mut per_layer: Vec<(usize, f32, f32, f32)> = Vec::new();
        for (li, lt) in self.cfg.layer_types.iter().enumerate() {
            let (mut wl_s, mut wl_c, mut wl_k) = (0.0f32, 0.0f32, 0.0f32);
            match lt {
                LayerType::LinearAttention => {
                    let s = self.dev.dtoh_sync_copy(state.s_state[li].as_ref().unwrap()).unwrap();
                    let c = self.dev.dtoh_sync_copy(state.conv_state[li].as_ref().unwrap()).unwrap();
                    for j in 0..per_slot { let dd=(s[per_slot+j]-s[j]).abs(); if dd>worst_s{worst_s=dd;} if dd>wl_s {wl_s=dd;} }
                    for j in 0..per_slot_conv { let dd=(c[per_slot_conv+j]-c[j]).abs(); if dd>worst_conv{worst_conv=dd;} if dd>wl_c {wl_c=dd;} }
                }
                LayerType::FullAttention => {
                    let kb = state.k_cache[li].as_ref().unwrap();
                    let vb = state.v_cache[li].as_ref().unwrap();
                    let per_slot_k = kb.len() / 2;
                    let per_slot_v = vb.len() / 2;
                    let k = self.dev.dtoh_sync_copy(kb).unwrap();
                    let v = self.dev.dtoh_sync_copy(vb).unwrap();
                    for j in 0..per_slot_k {
                        let dk=(half::bf16::to_f32(k[per_slot_k+j])-half::bf16::to_f32(k[j])).abs(); if dk>worst_kv{worst_kv=dk;} if dk>wl_k {wl_k=dk;}
                    }
                    for j in 0..per_slot_v {
                        let dv=(half::bf16::to_f32(v[per_slot_v+j])-half::bf16::to_f32(v[j])).abs(); if dv>worst_kv{worst_kv=dv;} if dv>wl_k {wl_k=dv;}
                    }
                }
            }
            per_layer.push((li, wl_s, wl_c, wl_k));
        }
        eprintln!("    [probe] N={}  s_state={}, conv_state={}, kv_cache={}", n, worst_s, worst_conv, worst_kv);
        if tap {
            for (li, ws, wc, wk) in &per_layer {
                if *ws > 0.0 || *wc > 0.0 || *wk > 0.0 {
                    let ty = if matches!(self.cfg.layer_types.get(*li), Some(LayerType::LinearAttention)) { "GDN" } else { "FA" };
                    eprintln!("    [state-diff] N={n} layer {li:>2} ({ty}): s={ws:.4} conv={wc:.4} kv={wk:.4}");
                }
            }
            // Per-layer residual compare: verify column t vs decode step t (the FIRST divergent
            // layer+column localizes the F9 residual).
            let vc: Vec<Vec<half::bf16>> = ver_tap.iter().find(|c| c.tag == "verify")
                .map(|c| c.layers.clone()).unwrap_or_default();
            let vg = ver_tap.iter().find(|c| c.tag == "verify").map(|c| c.gdn.clone()).unwrap_or_default();
            let vs_s = ver_tap.iter().find(|c| c.tag == "verify").map(|c| c.s.clone()).unwrap_or_default();
            let dc: Vec<&Vec<Vec<half::bf16>>> = dec_tap.iter().map(|c| &c.layers).collect();
            let dg: Vec<&Vec<(Vec<half::bf16>, Vec<half::bf16>)>> = dec_tap.iter().map(|c| &c.gdn).collect();
            let ds_s: Vec<&Vec<Vec<f32>>> = dec_tap.iter().map(|c| &c.s).collect();
            eprintln!("    [tap] N={n}: verify record layers={} gdn={}; decode calls={} (layers0={} gdn0={})",
                vc.len(), vg.len(), dc.len(), dc.first().map(|l| l.len()).unwrap_or(0), dg.first().map(|g| g.len()).unwrap_or(0));
            if !vc.is_empty() && dc.len() == n {
                for (li, vl) in vc.iter().enumerate() {
                    let (mut wl, mut wt, mut wi) = (0f32, 0usize, 0usize);
                    let mut first: Option<(usize, usize, f32, f32)> = None;
                    for t in 0..n {
                        let dl = &dc[t][li];
                        for i in 0..h {
                            let a = half::bf16::to_f32(vl[t * h + i]);
                            let b = half::bf16::to_f32(dl[i]);
                            let dd = (a - b).abs();
                            if dd > wl { wl = dd; wt = t; wi = i; }
                            if dd > 0.0 && first.is_none() { first = Some((t, i, a, b)); }
                        }
                    }
                    if wl > 0.0 {
                        let ty = if matches!(self.cfg.layer_types.get(li), Some(LayerType::LinearAttention)) { "GDN" } else { "FA" };
                        eprintln!("    [resid-diff] N={n} layer {li:>2} ({ty}): worst={wl:.6e} at col={wt} elem={wi}{}",
                            first.map(|(t, i, a, b)| format!(" first=col{t} elem{i} verify={a:.6e} decode={b:.6e}"))
                                 .unwrap_or_default());
                    }
                }
                // Post-delta S compare (n<=4): verify's slot-0 committed state vs decode step t's
                // slot-1 committed state, per GDN layer, with (head,row,col) localization.
                let lin_nh = self.eff_lin_v_heads();
                let (kdd, vdd) = (self.cfg.lin_k_dim, self.cfg.lin_v_dim);
                let ext = lin_nh * kdd * vdd;
                for (gi, vss) in vs_s.iter().enumerate() {
                    for t in 0..n {
                        let dss = match ds_s.get(t).and_then(|s| s.get(gi)) {
                            Some(x) => x, None => continue,
                        };
                        if vss.len() < 2 * ext || dss.len() < 2 * ext { continue; }
                        let (mut w, mut fi) = (0f32, 0usize);
                        for i in 0..ext {
                            let a = vss[i];
                            let b = dss[ext + i];
                            let dd = (a - b).abs();
                            if dd > w { w = dd; fi = i; }
                        }
                        if w > 0.0 {
                            let head = fi / (kdd * vdd);
                            let rem = fi - head * kdd * vdd;
                            let row = rem / vdd;
                            let col = rem - row * vdd;
                            eprintln!("    [s-diff] N={n} gdn#{gi:>2} after-col={t}: worst={w:.6e} at head={head} row={row} col={col} (verify={} decode={})",
                                vss[fi], dss[ext + fi]);
                        }
                    }
                }
                let conv_dim = self.eff_conv_dim();
                let lin_nh = self.eff_lin_v_heads();
                let vd = self.cfg.lin_v_dim;
                for (gi, (vconvo, vcore)) in vg.iter().enumerate() {
                    // Decode call t holds column t's row 0 (width-1 forwards): its conv rows are the
                    // first conv_dim entries of the (possibly mtot-pitched) tap buffer.
                    for t in 0..n {
                        let (dconvo, dcore) = match dg.get(t).and_then(|g| g.get(gi)) {
                            Some(x) => x, None => continue,
                        };
                        let (mut wc, mut wcc) = (0f32, 0f32);
                        let mut fc: Option<(usize, f32, f32)> = None;
                        for c in 0..conv_dim.min(dconvo.len()) {
                            let a = half::bf16::to_f32(vconvo[t * conv_dim + c]);
                            let b = half::bf16::to_f32(dconvo[c]);
                            let dd = (a - b).abs();
                            if dd > wc { wc = dd; }
                            if dd > 0.0 && fc.is_none() { fc = Some((c, a, b)); }
                        }
                        for i in 0..(lin_nh * vd).min(dcore.len()) {
                            let a = half::bf16::to_f32(vcore[t * lin_nh * vd + i]);
                            let b = half::bf16::to_f32(dcore[i]);
                            let dd = (a - b).abs();
                            if dd > wcc { wcc = dd; }
                        }
                        if wc > 0.0 || wcc > 0.0 {
                            eprintln!("    [gdn-diff] N={n} gdn#{gi:>2} col={t}: convo={wc:.6e}{} core={wcc:.6e}",
                                fc.map(|(c, a, b)| format!(" (first ch{c} verify={a:.6e} decode={b:.6e})")).unwrap_or_default());
                        }
                    }
                }
            } else if !vc.is_empty() {
                eprintln!("    [resid-diff] N={n}: tap pairing skipped (verify layers={}, decode calls={})", vc.len(), dc.len());
            }
        }
        worst_s
    }

    /// B8/G1 GATE: verify-bucket bit-exactness at M<=8 on REAL activations. For every chain width
    /// n in 1..=8, runs the same token chain as (i) an exact-width verify, (ii) a BUCKET-PADDED
    /// verify (width verify_bucket_width(n), extra columns padded with a fixed token), and (iii) an
    /// n-step sequential DECODE ground truth; asserts preds[..n] and the [h, n] residual columns
    /// are BIT-IDENTICAL across all three. This is the "padded buckets must not change per-column
    /// reduction order" gate (the U2 class-1 lesson) plus the ladder's item 1 (TP=1 col k of an
    /// 8-wide verify == an 8-token decode sequence) at single-process level. Uses two GDN
    /// checkpoint slots so rollback machinery stays in the loop. Returns (widths, bit-mismatches).
    pub fn verify_bucket_bitexact(&self, pool: &mut Pool, state: &mut BatchGpuState,
                                  prompt: &[u32], kv_stride: usize, max_n: usize)
                                  -> Vec<(usize, usize)> {
        let h = self.cfg.hidden_size;
        let plen = prompt.len();
        let vocab = self.cfg.vocab_size;
        assert!(max_n <= MAX_VERIFY, "bucket probe width {max_n} exceeds MAX_VERIFY");
        // Scratch slots: 2..(2+max_n) hold per-column GDN checkpoints (the ring layout the serving
        // loop uses), contiguous so `slot (2 + t) holds verify column t` holds for every t < max_n.
        let n_slots_needed = 2 + max_n;
        let mut results = Vec::new();

        // 1) Prefill slot 0 and slot 1 identically (slot 1 is the decode reference lane).
        self.zero_slot_state(state, 0, kv_stride);
        let (a0, h0) = self.prefill_batch(pool, prompt, state, 0, kv_stride, 0);
        pool.release_bf16(h0, h * plen);
        self.zero_slot_state(state, 1, kv_stride);
        let (a1, _h1) = self.prefill_batch(pool, prompt, state, 1, kv_stride, 0);
        assert_eq!(a0, a1, "bucket probe: prefill divergence slot0 vs slot1");
        let _ = vocab;

        // 2) Ground-truth decode continuation of n_tokens tokens on slot 1 (greedy, from a0).
        let mut gt: Vec<u32> = Vec::with_capacity(max_n);
        let mut gt_tok = a0;
        let mut bufs = self.new_decode_buffers(2);
        self.dev.htod_sync_copy_into(&[1i32, 0], &mut bufs.slot_ids_dev).unwrap();
        self.dev.synchronize().unwrap();
        let mut dpos = plen;
        for _ in 0..max_n {
            self.dev.htod_sync_copy_into(&[gt_tok as i32, 0], &mut bufs.tokens_dev).unwrap();
            self.dev.htod_sync_copy_into(&[dpos as i32, 0], &mut bufs.pos_dev).unwrap();
            self.dev.synchronize().unwrap();
            let nxt = self.forward_decode(pool, &mut bufs, state, kv_stride, dpos + 1, 1);
            gt.push(gt_tok);
            gt_tok = nxt[0];
            dpos += 1;
        }
        // `gt` now holds the greedy continuation tokens at positions plen..plen+max_n-1; the
        // decode's own logits predicted gt_tok as the token AFTER the last one — the verify's
        // preds[n-1] analog. The chain to verify is exactly `gt[..n]` for each width n.

        // 3) For each width: exact verify vs padded verify vs decode ground truth.
        for n in 1..=max_n {
            let chain: Vec<u32> = gt[..n].to_vec();
            let bucket = verify_bucket_width(n);

            // (i) exact-width verify on slot 0 (fresh prefill so state is clean per width).
            self.zero_slot_state(state, 0, kv_stride);
            let (_a, _hh) = self.prefill_batch(pool, prompt, state, 0, kv_stride, 0);
            let (preds_exact, vout_exact) = self.verify_forward(pool, &chain, state, 0, kv_stride, plen, Some(2), None);
            self.sync_stream();
            let he: Vec<half::bf16> = self.dev.dtoh_sync_copy(&vout_exact).unwrap();
            pool.release_bf16(vout_exact, h * n);

            // (ii) bucket-padded verify: same first n tokens, pad to `bucket` with a fixed token.
            let mut chain_p = chain.clone();
            let pad_tok = 1u32; // deterministic; padded columns are never read
            for _ in n..bucket { chain_p.push(pad_tok); }
            self.zero_slot_state(state, 0, kv_stride);
            let (_a, _hh) = self.prefill_batch(pool, prompt, state, 0, kv_stride, 0);
            let (preds_pad, vout_pad) = self.verify_forward(pool, &chain_p, state, 0, kv_stride, plen, Some(2), None);
            self.sync_stream();
            let hp: Vec<half::bf16> = self.dev.dtoh_sync_copy(&vout_pad).unwrap();
            pool.release_bf16(vout_pad, h * bucket);

            // BITWISE compare: preds[..n], residual columns [0..n), and (if padded) also confirm
            // the pad columns did not perturb anything we read.
            let mut mism = 0usize;
            for i in 0..n {
                if preds_exact[i] != preds_pad[i] { mism += 1; }
            }
            for i in 0..h * n {
                if he[i].to_bits() != hp[i].to_bits() { mism += 1; }
            }
            // (iii) decode ground truth: chain[i] IS the greedy decode token at plen+i; the exact
            // verify must reproduce the model's own greedy continuation: preds_exact[n-1] is the
            // argmax AFTER the last chain token; the decode ground truth's corresponding token is
            // gt[n] when n < max_n. Compare what exists.
            if n < max_n {
                // preds_exact[n-1] must equal the token decode produced after consuming chain[..n]
                // (that is gt[n]: decode consumed gt[..=n-1] and emitted gt[n]).
                if preds_exact[n - 1] != gt[n] { mism += 1; }
            }
            let _ = n_slots_needed;
            results.push((n, mism));
        }
        drop(bufs);
        results
    }

    /// B8/G1 GATE: reject-path probe at WIDTH 8 (the ping-pong checkpoint at the widest verify).
    /// Same three-way shape as `probe_reject_path` but the verified chain is 8 columns wide with
    /// only the first 1-2 accepted (the rest deliberately wrong), so the GDN rollback must restore
    /// the state of the LAST ACCEPTED column among the 8 written checkpoints — the exact failure
    /// mode B8_VERIFY_M8 flags as highest-risk (ring off-by-one at width 8). Returns max diff of
    /// (restored slot vs checkpoint slot) and (restored slot vs decode reference), which must be 0.
    pub fn probe_reject_path_w8(&self, pool: &mut Pool, state: &mut BatchGpuState,
                                prompt: &[u32], kv_stride: usize) -> (f32, f32, f32) {
        let h = self.cfg.hidden_size;
        let plen = prompt.len();
        let w = 8usize;
        let mut bufs = self.new_decode_buffers(2);
        self.zero_slot_state(state, 0, kv_stride);
        let (a0, h0) = self.prefill_batch(pool, prompt, state, 0, kv_stride, 0);
        pool.release_bf16(h0, h * plen);
        self.zero_slot_state(state, 1, kv_stride);
        let (a1, h1) = self.prefill_batch(pool, prompt, state, 1, kv_stride, 0);
        pool.release_bf16(h1, h * plen);
        assert_eq!(a0, a1, "w8 reject probe: prefill divergence");
        let committed = a0;

        // Reference: decode the committed token on slot 1 (reference S1 for column 0).
        self.dev.htod_sync_copy_into(&[1i32, 0], &mut bufs.slot_ids_dev).unwrap();
        self.dev.htod_sync_copy_into(&[committed as i32, 0], &mut bufs.tokens_dev).unwrap();
        self.dev.htod_sync_copy_into(&[plen as i32, 0], &mut bufs.pos_dev).unwrap();
        self.dev.synchronize().unwrap();
        let _ref_tok = self.forward_decode(pool, &mut bufs, state, kv_stride, plen + 1, 1);

        // 8-wide verify: [committed, 7 deliberately-wrong drafts] -> acceptance 0 (or near it).
        // Wrong = every draft is a token the model just emitted (near-certain mismatch at pos>0).
        let vi: Vec<u32> = std::iter::once(committed).chain(std::iter::repeat(committed).take(w - 1)).collect();
        let (_preds, vout) = self.verify_forward(pool, &vi, state, 0, kv_stride, plen, Some(2), None);
        pool.release_bf16(vout, h * w);
        // nacc == 0 in general -> rollback target is checkpoint slot (2 + 0).
        let nacc = 0usize;
        self.copy_gdn_slot(state, 2 + nacc, 0);
        self.sync_stream();

        let lin_nh = self.cfg.lin_num_v_heads; let kd = self.cfg.lin_k_dim; let vd = self.cfg.lin_v_dim;
        let per_slot = lin_nh * kd * vd;
        let conv_dim = self.cfg.key_dim()*2 + self.cfg.value_dim(); let ck = self.cfg.conv_kernel;
        let per_slot_conv = conv_dim * ck;
        let mut b_s = 0.0f32; let mut b_c = 0.0f32; let mut c_s = 0.0f32; let mut c_c = 0.0f32;
        for (li, lt) in self.cfg.layer_types.iter().enumerate() {
            if matches!(lt, LayerType::LinearAttention) {
                let s = self.dev.dtoh_sync_copy(state.s_state[li].as_ref().unwrap()).unwrap();
                let c = self.dev.dtoh_sync_copy(state.conv_state[li].as_ref().unwrap()).unwrap();
                for j in 0..per_slot {
                    let (l, sn, rf) = (s[j], s[2*per_slot+j], s[per_slot+j]);
                    let d1=(l-sn).abs(); if d1>b_s{b_s=d1;}
                    let d2=(l-rf).abs(); if d2>c_s{c_s=d2;}
                }
                for j in 0..per_slot_conv {
                    let (l, sn, rf) = (c[j], c[2*per_slot_conv+j], c[per_slot_conv+j]);
                    let d1=(l-sn).abs(); if d1>b_c{b_c=d1;}
                    let d2=(l-rf).abs(); if d2>c_c{c_c=d2;}
                }
            }
        }
        drop(bufs);
        (b_s.max(b_c), c_s.max(c_c), 0.0)
    }

    /// Three-way reject-path probe (expert advice §3/§5). Forces a rejection and checks whether the
    /// ping-pong checkpoint/rollback is bit-exact — the decisive reject-path equality test.
    ///   slot A (0): verify([committed, wrong_draft], ckpt=Some(2)) → snapshots S1 into slot 2
    ///   slot B (1): decode committed once → reference S1
    /// Compares (per GDN layer conv+s, and the logical KV region):
    ///   (a) snap(slot 2) vs reference(slot 1)   — must be 0; else the checkpoint WRITE is wrong
    ///   (b) restored(slot 0) vs snap(slot 2)     — must be 0; else the D2D rollback copy is wrong
    ///   (c) restored(slot 0) vs reference(slot 1)— must be 0; the decisive reject-path equality
    ///   (d) KV positions 0..plen (committed region) slot 0 vs slot 1 — must be 0
    /// With f32 state + D2D rollback, every diff must be EXACTLY 0; any nonzero is a discrete bug.
    pub fn probe_reject_path(&self, pool: &mut Pool, state: &mut BatchGpuState, prompt: &[u32],
                             kv_stride: usize) {
        let h = self.cfg.hidden_size;
        let mut bufs = self.new_decode_buffers(2);
        self.zero_slot_state(state, 0, kv_stride);
        let (a0, h0) = self.prefill_batch(pool, prompt, state, 0, kv_stride, 0);
        pool.release_bf16(h0, h * prompt.len());
        self.zero_slot_state(state, 1, kv_stride);
        let (a1, h1) = self.prefill_batch(pool, prompt, state, 1, kv_stride, 0);
        pool.release_bf16(h1, h * prompt.len());
        assert_eq!(a0, a1, "prefill divergence");
        let plen = prompt.len();
        let committed = a0;
        let wrong_draft = a0;  // deliberately "wrong" → forces nacc=0 reject in general

        // slot B (1): decode committed once → reference S1 (state after committed at pos plen).
        self.dev.htod_sync_copy_into(&[1i32, 0], &mut bufs.slot_ids_dev).unwrap();
        self.dev.htod_sync_copy_into(&[committed as i32, 0], &mut bufs.tokens_dev).unwrap();
        self.dev.htod_sync_copy_into(&[plen as i32, 0], &mut bufs.pos_dev).unwrap();
        self.dev.synchronize().unwrap();
        let _ref_tok = self.forward_decode(pool, &mut bufs, state, kv_stride, plen + 1, 1);

        // slot A (0): verify([committed, wrong_draft], ckpt=Some(2)) → snapshot S1 into slot 2.
        let vi = vec![committed, wrong_draft];
        let (preds, vout) = self.verify_forward(pool, &vi, state, 0, kv_stride, plen, Some(2), None);
        pool.release_bf16(vout, h * 2);
        eprintln!("[reject] committed={} wrong_draft={} verify preds={:?} (forcing nacc=0 reject)", committed, wrong_draft, preds);

        let lin_nh = self.cfg.lin_num_v_heads; let kd = self.cfg.lin_k_dim; let vd = self.cfg.lin_v_dim;
        let per_slot = lin_nh * kd * vd;
        let conv_dim = self.cfg.key_dim()*2 + self.cfg.value_dim(); let ck = self.cfg.conv_kernel;
        let per_slot_conv = conv_dim * ck;
        let nkv = self.cfg.num_kv_heads; let hd = self.cfg.head_dim;

        // (a) snap(slot 2) vs reference(slot 1), pre-rollback.
        let mut a_s = 0.0f32; let mut a_c = 0.0f32;
        for (li, lt) in self.cfg.layer_types.iter().enumerate() {
            if matches!(lt, LayerType::LinearAttention) {
                let s = self.dev.dtoh_sync_copy(state.s_state[li].as_ref().unwrap()).unwrap();
                let c = self.dev.dtoh_sync_copy(state.conv_state[li].as_ref().unwrap()).unwrap();
                for j in 0..per_slot { let dd=(s[2*per_slot+j]-s[per_slot+j]).abs(); if dd>a_s{a_s=dd;} }
                for j in 0..per_slot_conv { let dd=(c[2*per_slot_conv+j]-c[per_slot_conv+j]).abs(); if dd>a_c{a_c=dd;} }
            }
        }
        eprintln!("[reject] (a) snap(slot2) vs ref(slot1):   s_state={}, conv={}", a_s, a_c);

        // Rollback: copy_gdn_slot(snap=2 → A=0), then sync before host reads.
        self.copy_gdn_slot(state, 2, 0);
        self.sync_stream();

        // (b) restored(slot 0) vs snap(slot 2); (c) restored(slot 0) vs ref(slot 1).
        let mut b_s = 0.0f32; let mut b_c = 0.0f32; let mut c_s = 0.0f32; let mut c_c = 0.0f32;
        for (li, lt) in self.cfg.layer_types.iter().enumerate() {
            if matches!(lt, LayerType::LinearAttention) {
                let s = self.dev.dtoh_sync_copy(state.s_state[li].as_ref().unwrap()).unwrap();
                let c = self.dev.dtoh_sync_copy(state.conv_state[li].as_ref().unwrap()).unwrap();
                for j in 0..per_slot {
                    let (l, sn, rf) = (s[j], s[2*per_slot+j], s[per_slot+j]);
                    let d1=(l-sn).abs(); if d1>b_s{b_s=d1;}
                    let d2=(l-rf).abs(); if d2>c_s{c_s=d2;}
                }
                for j in 0..per_slot_conv {
                    let (l, sn, rf) = (c[j], c[2*per_slot_conv+j], c[per_slot_conv+j]);
                    let d1=(l-sn).abs(); if d1>b_c{b_c=d1;}
                    let d2=(l-rf).abs(); if d2>c_c{c_c=d2;}
                }
            }
        }
        eprintln!("[reject] (b) restored(slot0) vs snap(slot2): s_state={}, conv={}", b_s, b_c);
        eprintln!("[reject] (c) restored(slot0) vs ref(slot1):   s_state={}, conv={}", c_s, c_c);

        // (d) KV positions 0..plen (committed region) slot 0 vs slot 1. Bitwise on the RAW cache
        // bytes: under the packed modes the per-position stride is the per-channel packed size
        // (k8v4: K 20 B/16, V 12 B/16 — they diverge), under bf16 both are hd, identical to the
        // old indexing.
        let mut d_kv = 0.0f32;
        let per_slot_k = nkv * kv_stride * self.kv_k_elems_per_pos();
        let per_slot_v = nkv * kv_stride * self.kv_v_elems_per_pos();
        let k_pp = self.kv_k_elems_per_pos();
        let v_pp = self.kv_v_elems_per_pos();
        for (li, lt) in self.cfg.layer_types.iter().enumerate() {
            if matches!(lt, LayerType::FullAttention) {
                let k = self.dev.dtoh_sync_copy(state.k_cache[li].as_ref().unwrap()).unwrap();
                let v = self.dev.dtoh_sync_copy(state.v_cache[li].as_ref().unwrap()).unwrap();
                for head in 0..nkv {
                    for p in 0..=plen {
                        for j in 0..k_pp {
                            let idx = (head*kv_stride + p)*k_pp + j;
                            let dk=(half::bf16::to_f32(k[per_slot_k+idx])-half::bf16::to_f32(k[idx])).abs();
                            if dk>d_kv { d_kv=dk; }
                        }
                        for j in 0..v_pp {
                            let idx = (head*kv_stride + p)*v_pp + j;
                            let dv=(half::bf16::to_f32(v[per_slot_v+idx])-half::bf16::to_f32(v[idx])).abs();
                            if dv>d_kv { d_kv=dv; }
                        }
                    }
                }
            }
        }
        eprintln!("[reject] (d) KV[0..plen] slot0 vs slot1 (committed region): {}", d_kv);
        eprintln!("[reject] all four MUST be 0. Nonzero (a)→checkpoint write; (b)→D2D copy; (c)→reject not bit-exact; (d)→KV write/pos.");

        // ---- Second step (expert §5 "next step"): verify→reject→rollback→verify-again, vs decode×2.
        // bonus = preds[0] from the first verify (the greedy next token after committed).
        let bonus = preds[0];
        // slot B (1): decode bonus once more → reference "after [committed, bonus]".
        self.dev.htod_sync_copy_into(&[bonus as i32, 0], &mut bufs.tokens_dev).unwrap();
        self.dev.htod_sync_copy_into(&[(plen + 1) as i32, 0], &mut bufs.pos_dev).unwrap();
        self.dev.synchronize().unwrap();
        let _ref_tok2 = self.forward_decode(pool, &mut bufs, state, kv_stride, plen + 2, 1);
        // slot A (0): verify([bonus, wrong_draft2], ckpt=Some(2)) at pos plen+1.
        let wrong_draft2 = bonus;
        let vi2 = vec![bonus, wrong_draft2];
        let (preds2, vout2) = self.verify_forward(pool, &vi2, state, 0, kv_stride, plen + 1, Some(2), None);
        pool.release_bf16(vout2, h * 2);
        self.copy_gdn_slot(state, 2, 0);
        self.sync_stream();
        let mut e_s = 0.0f32; let mut e_c = 0.0f32; let mut e_kv = 0.0f32;
        let k_pp = self.kv_k_elems_per_pos();   // packed per-channel strides (k8v4 K/V diverge)
        let v_pp = self.kv_v_elems_per_pos();
        let valid_k = |head: usize, p: usize| (head * kv_stride + p) * k_pp;
        let valid_v = |head: usize, p: usize| (head * kv_stride + p) * v_pp;
        for (li, lt) in self.cfg.layer_types.iter().enumerate() {
            match lt {
                LayerType::LinearAttention => {
                    let s = self.dev.dtoh_sync_copy(state.s_state[li].as_ref().unwrap()).unwrap();
                    let c = self.dev.dtoh_sync_copy(state.conv_state[li].as_ref().unwrap()).unwrap();
                    for j in 0..per_slot { let dd=(s[j]-s[per_slot+j]).abs(); if dd>e_s{e_s=dd;} }
                    for j in 0..per_slot_conv { let dd=(c[j]-c[per_slot_conv+j]).abs(); if dd>e_c{e_c=dd;} }
                }
                LayerType::FullAttention => {
                    let k = self.dev.dtoh_sync_copy(state.k_cache[li].as_ref().unwrap()).unwrap();
                    let v = self.dev.dtoh_sync_copy(state.v_cache[li].as_ref().unwrap()).unwrap();
                    for head in 0..nkv {
                        for p in 0..=plen+1 {
                            for j in 0..k_pp {
                                let idx = valid_k(head, p) + j;
                                let dk=(half::bf16::to_f32(k[per_slot_k+idx])-half::bf16::to_f32(k[idx])).abs(); if dk>e_kv{e_kv=dk;}
                            }
                            for j in 0..v_pp {
                                let idx = valid_v(head, p) + j;
                                let dv=(half::bf16::to_f32(v[per_slot_v+idx])-half::bf16::to_f32(v[idx])).abs(); if dv>e_kv{e_kv=dv;}
                            }
                        }
                    }
                }
            }
        }
        eprintln!("[reject] (e) AFTER 2nd verify+rollback vs decode×2: s_state={}, conv={}, kv[0..plen+1]={}", e_s, e_c, e_kv);
        eprintln!("[reject] 2nd verify preds={:?} (bonus={}, greedy-after-committed={})", preds2, bonus, preds[0]);
        drop(bufs);
    }
    /// `offset` sequential greedy decodes (so both reach the same state at plen+offset), then decodes
    /// `depth` more on slot 0 (ground truth) while verify_forward-ing `depth` on slot 1 at
    /// plen+offset. Isolates whether the causal-append verify path stays lossless after many appends.
    pub fn bench_verify_at_offset(&self, pool: &mut Pool, state: &mut BatchGpuState, prompt: &[u32],
                                  kv_stride: usize, offset: usize, depth: usize)
                                 -> (Vec<u32>, Vec<u32>, Vec<bool>) {
        let h = self.cfg.hidden_size;
        let mut bufs = self.new_decode_buffers(2);
        self.zero_slot_state(state, 0, kv_stride);
        let (s0a, h0) = self.prefill_batch(pool, prompt, state, 0, kv_stride, 0);
        pool.release_bf16(h0, h * prompt.len());
        self.zero_slot_state(state, 1, kv_stride);
        let (s0b, h1) = self.prefill_batch(pool, prompt, state, 1, kv_stride, 0);
        pool.release_bf16(h1, h * prompt.len());
        if s0a != s0b { crate::gpu::dump_dual_prefill_diff(prompt.len()); }
        assert_eq!(s0a, s0b, "prefill divergence slot0 vs slot1");

        let plen = prompt.len();
        let mut cur = s0a;
        let mut pos = plen;
        // Advance BOTH slots identically by `offset` tokens.
        for _ in 0..offset {
            let toks_i32 = vec![cur as i32, cur as i32];
            let pos_i32 = vec![pos as i32, pos as i32];
            self.dev.htod_sync_copy_into(&toks_i32, &mut bufs.tokens_dev).unwrap();
            self.dev.htod_sync_copy_into(&pos_i32, &mut bufs.pos_dev).unwrap();
            self.dev.synchronize().unwrap();
            let next = self.forward_decode(pool, &mut bufs, state, kv_stride, pos + 1, 2);
            cur = next[0]; // both lanes identical → next[0]==next[1]
            pos += 1;
        }

        // Ground truth: decode `depth` more on slot 0 (lane 0 → physical slot 0).
        let mut seq_tokens = vec![cur];
        for _ in 0..depth {
            let tok = *seq_tokens.last().unwrap() as i32;
            let toks_i32 = vec![tok, 0];
            let pos_i32 = vec![pos as i32, 0];
            self.dev.htod_sync_copy_into(&toks_i32, &mut bufs.tokens_dev).unwrap();
            self.dev.htod_sync_copy_into(&pos_i32, &mut bufs.pos_dev).unwrap();
            self.dev.synchronize().unwrap();
            let next = self.forward_decode(pool, &mut bufs, state, kv_stride, pos + 1, 1);
            seq_tokens.push(next[0]);
            pos += 1;
        }

        // Verify `depth` tokens on slot 1 at position plen+offset.
        let verify_input: Vec<u32> = seq_tokens[..depth].to_vec();
        let (preds, vout) = self.verify_forward(pool, &verify_input, state, 1, kv_stride, plen + offset, None, None);
        pool.release_bf16(vout, h * depth);
        drop(bufs);

        let matches: Vec<bool> = (0..depth).map(|i| preds[i] == seq_tokens[i + 1]).collect();
        (seq_tokens, preds, matches)
    }
}
