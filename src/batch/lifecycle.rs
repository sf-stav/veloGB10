//! Construction, setters and the run loops of the scheduler: `new`/`with_df2`, the TP head/mirror loops and `run_spec_bench`.
//! Another `impl BatchScheduler` block (file map in `mod.rs`); the methods keep their exact text.
//! Binding invariants: greedy MTP stays bitwise lossless; spec gating is per source (AGENTS.md section 4).

use super::*;

impl BatchScheduler {
    pub fn new(gpu: GpuModel, max_batch: usize, kv_stride: usize, eos: Vec<u32>,
               rx: mpsc::UnboundedReceiver<BatchRequest>,
               mtp: MtpPolicy, prefix_cache: bool, ngram_draft: usize, tree_draft: bool, mtp_lanes: bool) -> Self {
        BatchScheduler::with_df2(gpu, max_batch, kv_stride, eos, rx, mtp, prefix_cache,
                                 ngram_draft, tree_draft, mtp_lanes,
                                 None, None, None, None, None, None, None)
    }

    /// P11 W5a: the --prefill-sched inline escape (default false = the cursor schedule).
    pub fn set_pf_inline(&mut self, on: bool) { self.pf_inline = on; }

    /// P3(b) L1: set the prose-lane routing (default `false` = rq sampled selector; `true` =
    /// greedy drafts for General-domain requests under `DFlash2Auto`).
    pub fn set_prose_lane_greedy(&mut self, on: bool) {
        self.prose_lane_greedy = on;
    }

    /// DF2_CARRY: `--df2-carry on|off` (default off — SPEC §7 gates the default flip). SPMD:
    /// ships on TpConfig, so every rank resolves the same value and makes the same admit choice.
    pub fn set_df2_carry(&mut self, on: bool) {
        self.df2_carry_enabled = on;
    }

    /// S5F: `new` + the DFlash2 round (loaded by the caller; `None` = absent/failed artifact → the
    /// source falls back to MTP per the standing directive) and its tap-sink twins.
    pub fn with_df2(mut gpu: GpuModel, max_batch: usize, kv_stride: usize, eos: Vec<u32>,
                    rx: mpsc::UnboundedReceiver<BatchRequest>,
                    mtp: MtpPolicy, prefix_cache: bool, ngram_draft: usize, tree_draft: bool,
                    mtp_lanes: bool,
                    df2: Option<crate::dflash2::round::Df2Round>,
                    df2_sink: Option<std::sync::Arc<crate::dflash2::capture::Df2TapSink>>,
                    df2_prime: Option<std::sync::Arc<crate::dflash2::capture::Df2PrimeSink>>,
                    mut dspark: Option<crate::dspark::round::DsparkRound>,
                    dspark_sink: Option<std::sync::Arc<crate::dflash2::capture::Df2TapSink>>,
                    dspark_prime: Option<std::sync::Arc<crate::dflash2::capture::Df2PrimeSink>>,
                    step_dump: Option<crate::dflash2::stepdump::StepDump>) -> Self {
        // When MTP is on, reserve one extra physical slot as a shared GDN-rollback snapshot target
        // (MTP lanes run one at a time, so a single snapshot slot suffices). It is never assigned to
        // a lane; `copy_gdn_slot(state, phys, snapshot_slot)` snapshots before verify, the reverse
        // restores on partial reject.
        // Reserve the snapshot slot whenever the model HAS an MTP head, not merely when MTP is
        // currently active: the policy can switch MTP on later and the slot must already exist.
        let mtp_has_head = mtp.head_present();
        // Size for the DEEPEST depth the policy may choose, not the one it opens at — the policy
        // re-picks depth from live acceptance every window, and a buffer sized to the initial depth
        // would be silently overrun the moment it went deeper.
        let mtp_depth = crate::gpu::MAX_AUTO_DEPTH;
        // ONE checkpoint slot per verify column we might roll back to: nacc can be 0..depth-2, so
        // we need depth-1 of them, contiguous from `mtp_snapshot_slot`. The GDN kernels write
        // column t's post-state into slot (mtp_snapshot_slot + t) via a derived stride.
        //
        // A single slot is only correct at depth 2. With depth >= 3 it silently rolled the recurrent
        // state back past accepted drafts, which is why greedy MTP was not lossless above depth 2.
        let mtp_snapshot_slot = max_batch;
        // PLAN/25 Phase 1: the DFlash2 TREE lane can engage lazily on any scheduler that carries
        // the DF2 round (e.g. the lossless probe switches sources per job), and its verify writes
        // per-column GDN checkpoints for up to MAX_VERIFY columns — size the band for that
        // capability whenever the round is present, not just when the source is the tree.
        let src_is_tree = matches!(mtp.spec_source(), SpecSource::DFlash2Tree);
        // P14: the DFlash v1 lane (--spec-source dflash) verifies the artifact's WHOLE block — for
        // the 3.6-35B drafter that is `1 + (block-1)` = 16 columns, exactly MAX_VERIFY — and the GDN
        // scan writes one checkpoint per column at `mtp_snapshot_slot + t`, t < n-1, with no bounds
        // check (`kernels/gpu_batch.cu:1299`, `:4120`). Sized off `mtp_depth` (8 ⇒ 7 checkpoints) the
        // 16-column verify overran the band by 8 slots per GDN layer: every emitted token came back
        // token 0 (`!`) because the verify's own logits were computed from corrupted state, at ~400
        // tok/s wall — the "step speed without acceptance is not a result" signature
        // (HANDOFF 2026-09-12). Capability, not current source: the v1 lane can engage on ANY
        // scheduler that carries the artifact (the probe/harness flip sources per job), exactly the
        // rule the DF2 tree already follows. `install_dflash_lane` runs after construction, so the
        // capability is detected from the artifact's presence OR the resolved source.
        let dflash_v1 = matches!(mtp.spec_source(), SpecSource::DFlash)
            || crate::draft_dir_env().is_some();
        // The band below is sized MAX_VERIFY; a block constant WIDER than that would make the v1
        // verify overrun it. Compile-time, not a boot warning: this relation is a property of two
        // constants, and a silent runtime warning is exactly the class of notice nobody reads.
        const _: () = assert!(crate::dflash::MAX_BLOCK <= crate::gpu::MAX_VERIFY,
            "DFlash MAX_BLOCK must not exceed MAX_VERIFY: the v1 verify width is bounded by \
             MAX_VERIFY (past it the verify leaves the batch-invariant path) and the GDN \
             checkpoint band is sized MAX_VERIFY");
        // COPY27B: resolve --copy-span for this trunk boot (one line; off = None and nothing below
        // changes). A copy round verifies up to MAX_VERIFY columns, so it needs the full checkpoint band.
        let copy_arm = crate::exl3_serve::copy_span_resolve_trunk(
            (gpu.tp_world().max(1)) as usize, (mtp_has_head || df2.is_some()) && mtp.active());
        // ULTRA2: --copy-span-lanes is EXL3-only; asked here = one boot line + velogb10_copy_span_lanes_state 5 and
        // the fallback counter (never a refusal). Not asked = nothing (no state, no line).
        let _ = crate::copy_lanes::resolve_other_server();
        let n_ckpt = if mtp_has_head || dflash_v1 {
            if df2.is_some() || tree_draft || mtp_lanes || src_is_tree || dflash_v1 || copy_arm.is_some() {
                crate::gpu::MAX_VERIFY
            } else {
                mtp_depth.saturating_sub(1).max(1)
            }
        } else { 0 };
        // One PROMPT checkpoint slot per lane, after the MTP snapshot slots. These hold the GDN state
        // as it stood at the END OF PREFILL — see `prompt_ckpt_slot`. They are pure state: no KV, and
        // none at all when prefix caching is off (51 MB/slot on 9B, 154 MB on 27B).
        let prompt_ckpt_slot = max_batch + n_ckpt;
        let ring_ckpt_slot = prompt_ckpt_slot + max_batch;
        // A6: the periodic ring checkpoints add max_batch*RING_CKPT_K state slots (154 MB/slot
        // class on 27B; K=16 x stride 512 keeps the same 8K coverage window as the old K=4 x
        // 2048 at 4x finer boundaries → ~2.5 GB per lane; the cached-turn slack drops from a
        // 2047-token re-prefill (~3.2 s at the mid-M lane) to <=511 tokens. b=1 deployments.)
        let n_state_slots = max_batch + n_ckpt
            + if prefix_cache { max_batch * (1 + RING_CKPT_K) } else { 0 };
        // ---- S-B16: the env-gated in-server lossless auditor (--mtp-lossless-audit=1) ----
        // A lockstep shadow decode of the greedy MTP lane's committed stream (bench_mtp_steps'
        // slot-1 ground truth, moved into the serving path) that pins the FIRST divergence to an
        // exact (request, emitted-index, position). DIAGNOSTIC ONLY: one extra plain decode
        // forward per emitted token roughly halves serving throughput while on; off, it is one
        // bool. TP>1 refuses the audit — see the field doc above.
        let audit_on = crate::opts::var(crate::opt!("mtp-lossless-audit")).is_ok();
        let audit_on = if audit_on && gpu.tp_world() > 1 {
            crate::reprintln!("[lossless-audit] REFUSED under TP={} — the shadow decode is single-node SPMD; run the audit at TP=1",
                      gpu.tp_world());
            false
        } else { audit_on };
        let audit_slot = n_state_slots; // one extra state slot appended after the normal band
        let n_state_slots = n_state_slots + if audit_on { 1 } else { 0 };
        if prefix_cache {
            eprintln!("[cache] prefix cache ON — prompt ckpt + ring {RING_CKPT_K}x{RING_CKPT_STRIDE} \
                       GDN checkpoints per slot ({n_state_slots} state slots)");
        }
        crate::memwatch::phase(&format!("sched:pre-state slots={n_state_slots} kv_stride={kv_stride} max_batch={max_batch}"));
        // S-B16: the auditor's shadow slot lives at index audit_slot in the STATE band, and KV
        // rows are addressed as base + slot*k_bytes — so the KV cache must span that slot INDEX,
        // not just max_batch. (v2/v3 bug: with kv_slots = max_batch = 1 the shadow's KV writes
        // landed ~8 slots past the allocation — silent corruption in v2 (the bogus 99% accept),
        // a wedged first request in v3.) Diagnostic-only: kv_slots = n_state_slots costs a few
        // GB of extra KV allocation on an audit boot; the serving layout is UNCHANGED when off.
        let kv_alloc_slots = if audit_on { n_state_slots } else { max_batch };
        let mut state = gpu.new_batch_state(kv_alloc_slots, n_state_slots, kv_stride);
        gpu.dev().synchronize().unwrap(); // ensure state allocs visible to non-blocking stream
        let mut pool = Pool::new(gpu.dev().clone());
        let mut bufs = gpu.new_decode_buffers(max_batch);
        crate::memwatch::phase("sched:post-state+bufs");
        // The auditor's shadow decode buffers: batch 2 like the bench's slot-1 ground truth, with
        // slot_ids preloaded so every shadow step reads/writes ONLY the audit slot's KV + GDN
        // state (mirror of bench_mtp_steps' [1, 0] preload on its slot-1 bufs2).
        let audit_bufs = if audit_on {
            let mut ab = gpu.new_decode_buffers(2);
            gpu.dev().htod_sync_copy_into(&[audit_slot as i32, 0i32], &mut ab.slot_ids_dev).unwrap();
            crate::reprintln!("[lossless-audit] ARMED slot={audit_slot} — one shadow decode per emitted token (throughput halves)");
            Some(ab)
        } else {
            None
        };

        // Capture decode graphs for all batch sizes, IF every kernel in the captured region fits the
        // default 48 KB/block limit.
        //
        // THIS USED TO BE `(kv_stride + head_dim) * 4`, which described the PRE-split-K attention
        // kernel: it held a kv_stride-sized score array in shared memory, so smem grew with context and
        // graphs were skipped beyond ~12K positions. The old comment even said "split-K will fix this".
        // Split-K shipped — `gqa_attn_splitk` sizes its smem by WARP COUNT, not context — but the guard
        // was never updated, so every long-context deployment silently lost CUDA graphs: 129 KB computed
        // at seq 32768 (the 122B preset), and at the 256K target envelope graphs would never capture at
        // all, on any model.
        //
        // The real requirement is the MAX over the kernels inside `forward_decode_gpu`, all of which are
        // constant in context length:
        //   GDN delta_step_b  18.4 KB  (gdn_launch: kd*(GDN_C+1) + kd + 3*GDN_C + 2*kd floats)  <- max
        //   attention splitk   8.1 KB  ((nw*hd + 2*nw) * 4, nw = hd/32)
        //   argmax_b           8.0 KB
        //   rmsnorm_b          4.0 KB
        // Capture already degrades safely (`capture_decode_graph` returns None and we fall back), so this
        // guard is belt-and-braces — but it must not be the thing that disables graphs everywhere.
        let head_dim = 256;
        let c = gpu.cfg();
        let gdn_smem = crate::gpu::gdn_launch(c.lin_k_dim, c.lin_v_dim).1 as usize;
        let attn_smem = ((head_dim / 32) * head_dim + 2 * (head_dim / 32)) * 4;
        let smem_bytes = gdn_smem.max(attn_smem).max(1024 * 8);
        // A/B switch: --no-decode-graphs=1 forces the non-graph path, so the value of capture can be
        // measured on any config without rebuilding (and gives an escape hatch if capture ever misbehaves).
        let smem_bytes = if crate::opts::var(crate::opt!("no-decode-graphs")).is_ok() { usize::MAX } else { smem_bytes };
        // qwen4_exp with the PLE table on SSD: the forward has a host round-trip (row reads) —
        // not capturable. Everything else keeps the graph path (PR #4 width/graph contract).
        let smem_bytes = if gpu.decode_graphs_supported() { smem_bytes } else {
            println!("Skipping CUDA graphs: PLE table is SSD-resident (host gather inside the forward).");
            usize::MAX
        };
        let mut graphs = std::collections::HashMap::new();
        if smem_bytes <= 48 * 1024 {
            if crate::boot_report::debug() {
            print!("Attempting CUDA graph capture for batch sizes 1..={}... ", max_batch);
            }
            let mut ok = true;
            for b in 1..=max_batch {
                match gpu.capture_decode_graph(&mut pool, &mut bufs, &mut state,
                                                 kv_stride, kv_stride, b) {
                    Some(g) => { graphs.insert(b, g); }
                    None => { ok = false; break; }
                }
            }
            // Graph capture (and its warmup) advanced the stateful GDN recurrent state — reset.
            gpu.zero_state(&mut state);
            gpu.dev().synchronize().unwrap(); // ensure zeroed state visible to non-blocking stream
            if ok {
                if crate::boot_report::debug() {
                    println!("captured (smem {} KB).", smem_bytes / 1024);
                }
            } else {
                graphs.clear();
                println!("unsupported (legacy stream); using non-graph decode.");
            }
        } else {
            crate::boot_report::notice(&format!("Skipping CUDA graphs: attention smem {} KB > 48 KB. Using non-graph decode.",
                                          smem_bytes / 1024));
        }

        // When GPU sampling is enabled, also capture a decode+sample graph per batch size so that
        // sampling requests get the same graph speedup greedy does. Falls back to the non-graph
        // sample path if capture is unsupported.
        let gpu_sample = crate::opts::var(crate::opt!("cpu-sample")).is_err();
        let mut sample_graphs: std::collections::HashMap<usize, CudaGraph> = std::collections::HashMap::new();
        if gpu_sample && !graphs.is_empty() {
            for b in 1..=max_batch {
                if let Some(g) = gpu.capture_decode_sample_graph(&mut pool, &mut bufs, &mut state,
                                                                  kv_stride, kv_stride, b) {
                    sample_graphs.insert(b, g);
                }
            }
            gpu.zero_state(&mut state);
            gpu.dev().synchronize().unwrap();
            if crate::boot_report::debug() {
                println!("captured {} GPU-sample graph(s).", sample_graphs.len());
            }
        }
        crate::memwatch::phase("sched:post-graphs");

        // Allocate per-slot MTP state when MTP is enabled. The MTP KV (`[nkv, kv_stride, hd]` bf16
        // per slot) must be zeroed: alloc_zeros is cuMemAllocAsync which does NOT zero, and stale
        // GPU garbage in unwritten MTP KV positions would yield nondeterministic drafts. Zero on the
        // COMPUTE stream (ordered with all later kernels), then sync.
        let (mtp_kc, mtp_vc, mtp_h_prev, mtp_h_save, mtp_h_scratch, mtp_cur_hidden,
             mtp_pen_slots, mtp_pen_extra, mtp_pen_rep, mtp_pen_presence, mtp_pen_freq,
             mtp_draft_pen_tokens, mtp_draft_pen_counts, mtp_draft_pen_rep, mtp_draft_pen_presence,
             mtp_draft_pen_freq) =
            if mtp_has_head {
                let cfg = gpu.cfg();
                // PR #4 width contract: the hidden columns the scheduler moves for the MTP head
                // (h_prev/save/scratch/cur) are the BACKBONE width — the hc-stream stack on
                // qwen4_exp (4*hidden), the hidden size elsewhere (resid_width() == hidden_size
                // whenever hc_count == 1, so this is a no-op for every existing model).
                let h = gpu.mtp_hidden_width();
                debug_assert_eq!(h, cfg.resid_width(),
                    "MTP hidden columns must be the hc-stream (resid) width");
                // §4.1: with --tp-shard-mtp the MTP attention is head-sharded and the draft cache
                // holds only this rank's kv heads (mtp_kv_heads == num_kv_heads when unsharded).
                let nkv = gpu.mtp_kv_heads();
                let hd = cfg.head_dim;
                let kv_bytes = nkv * kv_stride * hd * 2; // bf16 — the DRAFT cache stays bf16 even
                // under the 4-bit main KV cache (quantized draft KV costs real acceptance).
                let dev = gpu.dev().clone();
                let mut kc: Vec<B> = Vec::with_capacity(max_batch);
                let mut vc: Vec<B> = Vec::with_capacity(max_batch);
                let mut hp: Vec<B> = Vec::with_capacity(max_batch);
                for _ in 0..max_batch {
                    // qwen4_exp QSA: the head's raw-key cache rides at the end of its K buffer
                    // (mtp_kc_elems) so the (kc, vc, kv_stride) plumbing stays untouched.
                    let k = dev.alloc_zeros::<half::bf16>(gpu.mtp_kc_elems(kv_stride)).unwrap();
                    let v = dev.alloc_zeros::<half::bf16>(nkv * kv_stride * hd).unwrap();
                    let p = dev.alloc_zeros::<half::bf16>(h).unwrap();
                    gpu.memset_compute_stream(*k.device_ptr(), gpu.mtp_kc_elems(kv_stride) * 2);
                    gpu.memset_compute_stream(*v.device_ptr(), kv_bytes);
                    kc.push(k);
                    vc.push(v);
                    hp.push(p);
                }
                let save = dev.alloc_zeros::<half::bf16>(h).unwrap();
                let scr = dev.alloc_zeros::<half::bf16>(h).unwrap();
                let cur = dev.alloc_zeros::<half::bf16>(h).unwrap();
                gpu.sync_stream(); // ensure MTP KV zeroing visible before any primed lane reads it
                // Penalty buffers for the verify path (depth positions). bf16 lanes already keep
                // penalties via the normal path; MTP greedy lanes now keep theirs too.
                let mp = crate::gpu::MAX_PEN_TOKENS;
                // Sized to MAX_VERIFY (not just the chain depth): a FOREST verify has up to MAX_VERIFY
                // columns spanning several lanes, each column carrying ITS lane's penalty (per-column).
                let pcap = crate::gpu::MAX_VERIFY;
                // LOOPFIX: whole-output verify penalty = per-column (lane slot, preceding drafts);
                // the per-slot state itself lives in DecodeBuffers (pen_cnt_dev / pen_pbits_dev).
                let pen_tokens = dev.htod_sync_copy(&vec![-1i32; pcap]).unwrap();
                let pen_counts = dev.htod_sync_copy(&vec![-1i32; pcap * crate::gpu::PEN_EXTRA]).unwrap();
                let pen_rep = dev.alloc_zeros::<f32>(pcap).unwrap();
                let pen_presence = dev.alloc_zeros::<f32>(pcap).unwrap();
                let pen_freq = dev.alloc_zeros::<f32>(pcap).unwrap();
                let dp_tokens = dev.alloc_zeros::<i32>(mp).unwrap();
                let dp_counts = dev.alloc_zeros::<i16>(mp).unwrap();
                let dp_rep = dev.alloc_zeros::<f32>(1usize).unwrap();
                let dp_presence = dev.alloc_zeros::<f32>(1usize).unwrap();
                let dp_freq = dev.alloc_zeros::<f32>(1usize).unwrap();
                (kc, vc, hp, Some(save), Some(scr), Some(cur),
                 Some(pen_tokens), Some(pen_counts), Some(pen_rep), Some(pen_presence), Some(pen_freq),
                 Some(dp_tokens), Some(dp_counts), Some(dp_rep), Some(dp_presence), Some(dp_freq))
            } else {
                // MTP absent: allocate MINIMAL 1-element dummy per-lane buffers so the unconditional
                // `mtp_kc[phys]`/`mtp_vc[phys]` indexing on the prefill/decode paths never panics. These
                // pointers are never dereferenced — every real MTP use is gated by `will_use_mtp` /
                // `self.mtp.active()`, both false without a head.
                let dev = gpu.dev().clone();
                let dummy = |n: usize| -> Vec<B> {
                    (0..n).map(|_| dev.alloc_zeros::<half::bf16>(1).unwrap()).collect()
                };
                (dummy(max_batch), dummy(max_batch), dummy(max_batch),
                 None, None, None, None, None, None, None, None,
                 None, None, None, None, None)
            };

        // The graph replays with per-step (anchor, nprev) written to device ints; the R13
        // volatile kernels are stable under capture (the probe asserts determinism). Env
        // --no-df2-graph=1 keeps the eager path (the captured-vs-eager measurement).
        // PLAN/25 Phase 1: `dflash2-tree` arms the WIDE (MAX_VERIFY-col) tap sink on the trunk —
        // the topo verify's n > BLOCK per-column taps land there; the accepted path is gathered
        // into the round's 8-col staging at commit. No-op for every other source.
        // WI1: arm the DSpark sinks BEFORE `gpu` moves into Self (the DF2 twins' pre-move
        // pattern), and bind the round to its sink (S5F3: keep the Arc; the lane copies the
        // sink's LIVE staging before each inject).
        crate::memwatch::phase("sched:post-mtp-state");
        if let Some(ds) = dspark.as_mut() {
            if let Some(sk) = &dspark_sink { gpu.set_dspark_capture(sk.clone()); ds.attach_sink(sk); }
            if let Some(ps) = &dspark_prime { gpu.set_dspark_prime_sink(ps.clone()); }
        }
        let df2_sink_tree = if matches!(mtp.spec_source(), SpecSource::DFlash2Tree) {
            let ws = std::sync::Arc::new(crate::dflash2::capture::Df2TapSink::new_cols(gpu.dev(), crate::gpu::MAX_VERIFY));
            gpu.set_df2_capture_tree(ws.clone());
            Some(ws)
        } else if copy_arm.is_some() && df2.is_some() {
            // COPY27B: a copy round wider than the DFlash2 block verifies more than 8 columns; their
            // taps land in this WIDE sink (the capture arm for n > block) and the accepted span is
            // injected into the ring from it in block-sized pieces. Only verifies wider than the block
            // write it (copy rounds; the forest verify when --mtp-lanes is on).
            let ws = std::sync::Arc::new(crate::dflash2::capture::Df2TapSink::new_cols(gpu.dev(), crate::gpu::MAX_VERIFY));
            gpu.set_df2_capture_tree(ws.clone());
            Some(ws)
        } else { None };
        let mut s = Self {
            dflash: None,
            gpu, pool, state, bufs, graphs, kv_stride, eos, max_batch, rx,
            lanes: (0..max_batch).map(|_| None).collect(),
            free_slots: (0..max_batch).rev().collect(),
            slot_cache: vec![Vec::new(); max_batch],
            slot_ckpt_seq: vec![Vec::new(); max_batch],
            slot_cache_images: vec![Vec::new(); max_batch],
            slot_ckpt_images: vec![Vec::new(); max_batch],
            prompt_ckpt_slot,
            ring_ckpt_slot,
            slot_ring_len: vec![vec![0usize; RING_CKPT_K]; max_batch],
            ring_cursor: vec![0usize; max_batch],
            slot_dspark_len: vec![0usize; max_batch],
            prefix_cache,
            ngram_draft,
            tree_draft,
            mtp_lanes,
            pf_inline: false,
            gpu_sample,
            sample_graphs,
            mtp,
            mtp_kc,
            mtp_vc,
            mtp_h_prev,
            mtp_h_save,
            mtp_h_scratch,
            mtp_cur_hidden,
            mtp_snapshot_slot,
            audit_on,
            audit_slot,
            audit_bufs,
            audit_cur: (0..max_batch).map(|_| None).collect(),
            audit_req_n: 0,
            mtp_pen_slots,
            mtp_pen_extra,
            mtp_pen_rep,
            mtp_pen_presence,
            mtp_pen_freq,
            mtp_draft_pen_tokens,
            mtp_draft_pen_counts,
            mtp_draft_pen_rep,
            mtp_draft_pen_presence,
            mtp_draft_pen_freq,
            pen_const_key: None,
            pen_had: false,
            df2,
            df2_sink,
            df2_sink_tree,
            df2_prime,
            dspark,
            dspark_sink,
            dspark_prime,
            df2_fallback_logged: false,
            dflash_fallback_logged: false,
            dflash_prime_armed: false,
            dflash_prime_done: false,
            dflash_carry_logged: false,
            dspark_fallback_logged: false,
            dspark_stat_steps: 0,
            dspark_stat_drafts: 0,
            dspark_stat_accepted: 0,
            dspark_stat_emitted: 0,
            dspark_stat_published: 0,
            prose_lane_greedy: false,
            df2_carry_enabled: false,
            cov_trace: step_dump.is_some(),
            cov_req_n: 0,
            step_dump,
            mtp_stat_steps: 0,
            mtp_stat_drafts: 0,
            mtp_stat_accepted: 0,
            mtp_stat_emitted: 0,
            mtp_stat_verify_fwds: 0,
            mtp_stat_published: 0,
            df2_tree_stat_steps: 0,
            df2_tree_stat_rescues: 0,
            df2_tree_stat_nodes: 0,
            df2_tree_stat_drafts: 0,
            df2_tree_stat_accepted: 0,
            df2_tree_stat_emitted: 0,
            df2_tree_stat_published: 0,
            df2_stat_steps: 0,
            df2_stat_drafts: 0,
            df2_stat_accepted: 0,
            df2_stat_emitted: 0,
            df2_stat_published: 0,
            dflash_stat_steps: 0,
            dflash_stat_drafts: 0,
            dflash_stat_accepted: 0,
            dflash_stat_emitted: 0,
            dflash_acc_a: [0; crate::dflash::MAX_BLOCK],
            dflash_acc_n: [0; crate::dflash::MAX_BLOCK],
            spec_steps: Vec::new(),
            spec_steps_on: false,
            mtp_draft_log: crate::opts::var(crate::opt!("mtp-draft-log")).ok().and_then(|p| {
                match std::fs::File::create(&p) {
                    Ok(f) => { eprintln!("[mtp] draft log -> {}", p); Some(f) }
                    Err(e) => { eprintln!("[mtp] WARN: cannot open --mtp-draft-log {}: {}", p, e); None }
                }
            }),
            mtp_curve_path: crate::opts::var(crate::opt!("mtp-curve-file")).ok(),
            tp_serving: false,
            // Device-resident token loop: the first step must upload everything (the capture
            // warmup left stale values in token_ids_dev/pos/ring), so the state starts dirty.
            resident_dirty: true,
            pf: None,
            copy_arm,
        };
        // COPY27B: the copy arm keeps the head's depth policy exploring (see MtpPolicy::explore_every).
        if s.copy_arm.is_some() { s.mtp.set_explore(COPY_MTP_EXPLORE_EVERY); s.mtp.open_deep(); }
        // S5F: capture the DFlash2 draft-round CUDA graph once (the MTP verify-graph pattern).
        // The graph replays with per-step (anchor, nprev) written to device ints; the R13
        // volatile kernels are stable under capture (the probe asserts determinism). Env
        // --no-df2-graph=1 keeps the eager path (the captured-vs-eager measurement).
        if crate::opts::var(crate::opt!("no-df2-graph")).is_err() {
            if let Some(df2) = s.df2.as_mut() {
                if df2.capture_round_graph() {
                    crate::reprintln!("[df2] draft-round CUDA graph captured (eager fallback via --no-df2-graph)");
                } else {
                    crate::reprintln!("[df2] draft-round graph capture unsupported — staying eager");
                }
            }
        }
        s
    }

    /// PREHARD1 [ISSUES 104]: count over-cap arrivals since the last sample. The trunk scheduler
    /// has no VecDeque queue — requests beyond the cap wait IN the channel (plus the TP head's
    /// `pending`) — so the observable is the backlog: every request present above `max_batch`
    /// (busy lanes + in-flight cursor + waiting) is an over-cap arrival; count the NEW ones (the
    /// delta since the last sample). Called once per loop iteration from both serve loops
    /// (`run`, `run_tp_head`); the series is the same
    /// `velogb10_over_cap_arrivals_total{disposition="queued"}` the EXL3 server counts.
    fn count_over_cap_arrivals(busy: usize, waiting: usize, cap: usize, last_excess: &mut usize) {
        let excess = over_cap_excess(busy, waiting, cap);
        let d = excess.saturating_sub(*last_excess);
        if d > 0 { crate::metrics::over_cap_queued_n(d as u64); }
        *last_excess = excess;
    }

    /// Run the scheduler loop until the request channel closes and no lanes remain.
    ///
    /// PLAN/SCHEDULER_2LANE_WORKDOC.md §3.1/§3.3 — the two-lane (cursor) loop. Shape per iteration:
    ///
    ///   1. DRAIN queued requests (cheap: admission only creates a cursor),
    ///   2. BLOCK for work only when there is none — a pending cursor IS work,
    ///   3. ADVANCE at most one prefill window of the single in-flight cursor,
    ///   4. DECODE the `b` lanes that are actually eligible (a mid-prefill request is not a lane).
    ///
    /// The ONE structural difference from the pre-cursor loop is the position of the window work:
    /// it used to run to completion INSIDE `admit()` (all `ceil(plen/PREFILL_CHUNK)` windows before
    /// that iteration's `decode_step`), so a co-resident lane's next token waited for the whole
    /// prompt — measured at 25,370 ms across a 16,800-token admission (P0). Now it waits for at
    /// most one window (12.6 s), and the window formula itself is unchanged (§3.2).
    /// LOOPFIX (2026-11-06): a request taken off the server channel (single node, or the TP head —
    /// never a mirror, which replays the head's shipped WireRequest). A request without a `seed`
    /// gets a fresh random one HERE, before the head ships its `Admit`, so every rank samples the
    /// same stream (`--seedless-seed random`, the default; the EXL3 server's `entropy_seed` rule).
    /// `--seedless-seed prompt-hash` keeps the old behaviour: `admit` derives the seed from a hash
    /// of the prompt, so an identical request always returns identical text.
    fn intake(mut req: BatchRequest) -> BatchRequest {
        if req.seed.is_none() && !matches!(crate::opts::var(crate::opt!("seedless-seed")).as_deref(), Ok("prompt-hash")) {
            req.seed = Some(rand::random::<u64>());
        }
        req
    }

    pub async fn run(mut self) {
        self.tp_drill_notice();   // S-B9-CF-AGREE16B (W3): a lockstep drill flag at TP=1 is INERT, said once
        let mut oc_last: usize = 0; // PREHARD1 [ISSUES 104]: over-cap arrival accounting state
        loop {
            // F8 diagnosis (temporary, --loop-trace): split the wall around decode_step so the
            // serve-vs-bench per-step gap (261 vs 188 ms) can be attributed.
            let loop_trace = crate::opts::var(crate::opt!("loop-trace")).is_ok();
            let t_admit = if loop_trace { Some(std::time::Instant::now()) } else { None };
            // Admit queued requests into free lanes (front-packed). Capacity counts BOTH the
            // decode-eligible lanes and the in-flight cursor: a mid-prefill request already owns a
            // physical slot, so it must not be oversubscribed.
            // `self.pf.is_none()` is re-tested EVERY iteration: a successful admit creates a
            // cursor, and at most one may exist (the shared df2/dspark rounds are primed
            // window-by-window). Without that term a max-batch>=3 server would try to admit a
            // second request and silently drop the first cursor's prefill.
            while self.pf.is_none()
                  && self.num_active() + usize::from(self.pf.is_some()) < self.max_batch
                  && self.lanes[self.num_active()].is_none() {
                match self.rx.try_recv() {
                    Ok(req) => self.admit(Self::intake(req)),
                    Err(_) => break,
                }
            }
            if self.num_active() == 0 && self.pf.is_none() {
                crate::metrics::sched_idle(); // v0.7.3 gauge: about to block for work
                match self.rx.recv().await {
                    Some(req) => self.admit(Self::intake(req)),
                    // Fall THROUGH (no `continue`): the cursor just created must get its first
                    // window this iteration, or the loop blocks on an empty channel forever.
                    None => break,
                }
            }
            // PREHARD1 [ISSUES 104]: over-cap arrival accounting (waiting = the channel backlog).
            Self::count_over_cap_arrivals(self.num_active() + usize::from(self.pf.is_some()),
                                          self.rx.len(), self.max_batch, &mut oc_last);
            // One window of the in-flight prefill (no-op when nothing is prefilling).
            self.pf_advance();
            let b = self.num_active();
            let t_step = if loop_trace { Some(std::time::Instant::now()) } else { None };
            let t_yield0 = if loop_trace { Some(std::time::Instant::now()) } else { None };
            if b > 0 { self.decode_step(b); }
            let decode_ms = t_step.map(|t0| t0.elapsed().as_secs_f32() * 1e3);
            // Yield to tokio so streaming handlers can flush SSE events between decode steps.
            tokio::task::yield_now().await;
            if let (Some(t0), Some(ty0), Some(ta0), Some(dm)) = (t_step, t_yield0, t_admit, decode_ms) {
                static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                static LAST: std::sync::OnceLock<std::sync::Mutex<std::time::Instant>> =
                    std::sync::OnceLock::new();
                let last = LAST.get_or_init(|| std::sync::Mutex::new(std::time::Instant::now()));
                let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n >= 5 && n < 35 {
                    let mut prev = last.lock().unwrap();
                    let since_prev = prev.elapsed().as_secs_f32() * 1e3;
                    *prev = std::time::Instant::now();
                    let yield_ms = ty0.elapsed().as_secs_f32() * 1e3 - dm;
                    let _ = t0;
                    eprintln!("[loop-trace] step {n}: decode_step={dm:.1}ms post_yield={yield_ms:.1}ms \
                               admit_ms={:.1} since_prev_total={since_prev:.1}ms",
                              ta0.elapsed().as_secs_f32() * 1e3);
                }
            }
        }
    }

    /// S-B9-CF-AGREE16B (W3): one boot line per lockstep drill flag this process carries: armed, or
    /// why it is inert here (`net::drill_plan`). Called where the rank is known (the TP serve loops).
    fn tp_drill_notice(&self) {
        let tp = if self.gpu.tp_world() > 1 { Some((self.gpu.tp_rank(), self.gpu.tp_world())) } else { None };
        for l in crate::net::drill_plan(crate::opts::get(crate::opt!("tp-agree-drill")).as_deref(),
                                        crate::opts::get(crate::opt!("tp-width-drill")).as_deref(), false, tp).lines {
            crate::reprintln!("{l}");
        }
    }

    /// TP item D — per-step SPMD divergence guard. After every executed decode step, both ranks
    /// exchange `(step, cumulative emitted, state hash)` over the link's agreement channel
    /// (`net::agree`, 10 s deadline) and REFUSE to continue on mismatch or timeout. The hash folds
    /// the per-lane output state (`last_tok`, `generated`, `pos`, `mtp_pos`) plus the MTP policy
    /// decision (`active`, `depth`) — identical on both ranks by construction, so any acceptance
    /// or policy divergence flips it within one step. On mismatch: abort the link cooperatively
    /// (kernels no-op through the stream, I9) and return Err — the head's die-with-it guard exits
    /// the server and the mirror's supervisor re-arms, rather than serve one more divergent token.
    /// `--tp-agree-drill=<step>` corrupts this rank's hash at one step (the forced-divergence
    /// drill gate; a CLI flag, test-only, NVFP4 packs).
    fn tp_agree_step(&self, step: u64) -> anyhow::Result<()> {
        let mut h: u32 = 0x811c9dc5;                     // FNV-1a over the lane/policy state
        let mut mix = |x: u32| { h ^= x; h = h.wrapping_mul(0x0100_0193); };
        let mut total_generated: u64 = 0;
        for l in self.lanes.iter().flatten() {
            mix(l.last_tok);
            mix(l.generated as u32);
            mix(l.pos as u32);
            mix(l.mtp_pos as u32);
            total_generated += l.generated as u64;
        }
        // PLAN/SCHEDULER_2LANE §3.4/§3.5: a mid-prefill request is NOT a lane, so its progress is
        // folded EXPLICITLY. Without this term two ranks that advanced different window counts would
        // agree on everything visible (lanes, policy, width) while holding different prefill state —
        // the divergence would only surface as a KV/state mismatch much later.
        if let Some(c) = self.pf.as_ref() {
            mix(c.phys as u32);
            mix(c.w0 as u32);
            mix(c.plen as u32);
        }
        mix(self.mtp.active() as u32);
        mix(self.mtp.depth() as u32);
        // P11/FOREST: the pack flag changes the verify shape per step; fold it so a one-sided
        // mtp_lanes resolution (mixed builds on one link) fails LOUD at step 0, not as drift.
        mix(self.mtp_lanes as u32);
        // B8/G1: k_verify (the verify WIDTH this step ran) joins the extended agree token — ranks
        // that disagree on the width execute different barrier sequences (I9 class). agree_ext
        // carries it in bits 27..31 of the token's low word; the depth IS the width for chain MTP.
        // S-B9-CF-AGREE16B (W5): the range check sees the whole usize, BEFORE the narrowing (a
        // `depth() as u8` first would turn 256 + k into k and slip past agree_ext's own check).
        let k_real: usize = if self.mtp.active() { self.mtp.depth() } else { 0 };
        let Some(k_verify) = crate::net::agree_width(k_real) else {
            crate::net::abort_link();
            anyhow::bail!("TP agree REFUSED at step {step}: verify width {k_real} does not fit the lockstep token (max {}) — \
                           link aborted instead of sending a width the peer could mistake for another",
                          crate::net::AGREE_WIDTH_MAX);
        };
        if let Ok(d) = crate::opts::var(crate::opt!("tp-agree-drill")) {
            if d.parse::<u64>().ok() == Some(step) {
                eprintln!("[tp-agree] DRILL: corrupting this rank's hash at step {step}");
                // S-B9-CF-AGREE16B (W3): rank-dependent (it was `h ^= 0xDEAD` on every rank: two armed
                // ranks corrupted alike and agreed)
                h ^= crate::net::drill_hash_flip(self.gpu.tp_rank());
            }
        }
        let count = (total_generated & 0xFF) as u8;
        // B8: agree_ext sends net::agree_word(k_verify, h) (width in bits 27..31, the digest folded to
        // 27 bits); compare against the SAME function's word. A raw-hash comparison mismatches whenever
        // MTP is active (k_verify = depth > 0) — the step-0 stall masked this.
        let (pc, ph) = match crate::net::agree_ext(step, count, k_verify, h) {
            Some(x) => x,
            None => {
                eprintln!("[tp-agree] link aborted or peer timeout at step {step} — aborting");
                crate::net::abort_link();
                anyhow::bail!("TP agree: link aborted/timeout at step {step}");
            }
        };
        // (after agree_ext: it has refused a width the word cannot carry)
        let h_ext = crate::net::agree_word(k_verify, h);
        if pc != count || ph != h_ext {
            eprintln!("[tp-agree] MISMATCH at step {step}: local (count {count}, hash {h_ext:#010x}) \
                       vs peer (count {pc}, hash {ph:#010x}) — aborting rather than serving \
                       divergent output");
            // R9 LOCALIZATION note: the per-layer xchain dump happens in net::agree (which sees the
            // mismatch on the head AND the sentinel on the nodes); this branch only runs when the
            // consensus token itself disagrees, which is a subset of that path.
            crate::net::abort_link();
            anyhow::bail!("TP AGREE MISMATCH at step {step}: local ({count}, {h_ext:#010x}) vs peer ({pc}, {ph:#010x})");
        }
        Ok(())
    }

    /// TP serving head loop (item A, N-way): the same loop shape as `run()`, plus a per-step
    /// rendezvous with EVERY node's mirror over the retained sync streams. Per step:
    ///   1. Drain admissions under the SAME capacity gate as `run()`, but DEFER the `admit()` calls
    ///      until after the Step message is sent — admission prefill runs SPMD all-reduces that the
    ///      mirror can only join once it has seen the admission, so shipping first is what keeps
    ///      the barrier sequences paired instead of deadlocked. Each drained request becomes a
    ///      `TpEvent::Admit`; `admit()` itself runs unmodified (its context-length reject is
    ///      deterministic from identical state, so the mirror rejects exactly the same requests).
    ///   2. Cancel detection: lanes whose client went away are marked `tp_cancelled` NOW (host
    ///      state, no race with decode_step's sweep) and shipped as `TpEvent::Cancel`. Indices are
    ///      the post-admission front-packed table; admissions only append past the active region,
    ///      so they cannot renumber these.
    ///   3. Send `ServingMsg::Step` — every executed step, even with an empty event list; it is the
    ///      rendezvous the mirror waits on.
    ///   4. Admit the pendings (in order), then one `decode_step` over the new front-packed table.
    /// On request-channel close (server shutdown): keep stepping until the live lanes drain, then
    /// send `ServingMsg::Shutdown` and return.
    pub async fn run_tp_head(mut self, mut streams: Vec<std::net::TcpStream>) -> anyhow::Result<()> {
        self.tp_serving = true;
        self.tp_drill_notice();
        let mut step: u64 = 0;
        let mut closed = false;
        // PERSISTENT across iterations (cursor scheduler). With one prefill cursor at a time, a
        // request can be drained while another is still prefilling; if `pending` were rebuilt every
        // iteration (as it was when admission was unconditional) that request would be DROPPED at
        // the end of the iteration — its `tx` goes with it and the client's SSE stream closes with
        // no tokens and no error. The 2-lane bar caught exactly that: lane 1 was admitted never,
        // and its client saw the body end 1.5 s in.
        let mut pending: Vec<BatchRequest> = Vec::new();
        let mut oc_last: usize = 0; // PREHARD1 [ISSUES 104]: over-cap arrival accounting state
        loop {
            let mut events: Vec<crate::tp_serve::TpEvent> = Vec::new();
            if !closed {
                loop {
                    // Capacity counts the in-flight prefill cursor too (it owns a physical slot).
                    let projected = self.num_active() + pending.len() + usize::from(self.pf.is_some());
                    if projected >= self.max_batch || self.lanes[projected].is_some() { break; }
                    match self.rx.try_recv() {
                        // rejected here, before any `Admit` is shipped: the mirror would admit it
                        Ok(req) => pending.extend(Self::refuse_logprobs(Self::intake(req))),
                        Err(mpsc::error::TryRecvError::Empty) => break,
                        Err(mpsc::error::TryRecvError::Disconnected) => { closed = true; break; }
                    }
                }
                // Idle: block for the next request, exactly like run() — the mirror blocks on the
                // session stream meanwhile, so both ranks sleep in step.
                if self.num_active() == 0 && pending.is_empty() && self.pf.is_none() && !closed {
                    crate::metrics::sched_idle(); // H3: about to block for work (busy must not stick at 1)
                    match self.rx.recv().await {
                        Some(req) => pending.extend(Self::refuse_logprobs(Self::intake(req))),
                        None => closed = true,
                    }
                }
            }
            // PREHARD1 [ISSUES 104]: over-cap arrival accounting (waiting = pending + channel
            // backlog — the TP head owns both halves of the trunk queue).
            Self::count_over_cap_arrivals(self.num_active() + usize::from(self.pf.is_some()),
                                          pending.len() + self.rx.len(), self.max_batch, &mut oc_last);
            // AT MOST ONE admission per step, and its `Admit` event is shipped in THIS step's event
            // list — the mirror admits exactly what the wire says, so the two ranks can never hold a
            // different number of cursors. (Draining several requests and admitting them all, as the
            // pre-cursor loop did, is impossible now: only one prefill cursor may exist.)
            let admit_now = self.pf.is_none() && !pending.is_empty();
            // Shutdown only when NOTHING is left — including an in-flight prefill cursor: sending
            // Shutdown while a cursor is mid-prefill would exit the mirror with a half-prefilled
            // lane on the head (§3.4's one new teardown hazard).
            if closed && pending.is_empty() && self.num_active() == 0 && self.pf.is_none() {
                for s in streams.iter_mut() {
                    crate::tp_serve::send_serving(s, &crate::tp_serve::ServingMsg::Shutdown)?;
                }
                return Ok(());
            }
            for i in 0..self.num_active() {
                if self.lanes[i].as_ref()
                    .map_or(false, |l| !l.tp_cancelled
                        && (l.tx.is_closed()
                            || crate::server::stream_backlog_exceeded(l.tx.len()))) { // LR-5
                    if !self.lanes[i].as_ref().map_or(false, |l| l.tx.is_closed()) {
                        crate::metrics::stream_backlog_cancel();
                        crate::reprintln!("[req] lane {i}: stream cancelled — unconsumed event backlog at the --stream-backlog-events limit (reader stalled; LR-5; Cancel shipped to the mirrors)");
                    }
                    self.lanes[i].as_mut().unwrap().tp_cancelled = true;
                    events.push(crate::tp_serve::TpEvent::Cancel { lane: i });
                }
            }
            if admit_now {
                events.push(crate::tp_serve::TpEvent::Admit((&pending[0]).into()));
            }
            // Fan the SAME per-step event list out to every node (world-1 mirrors). The head is the
            // hub; every mirror must replay an identical scheduler state at the same step index.
            let msg = crate::tp_serve::ServingMsg::Step(
                crate::tp_serve::StepEvents { step, events });
            for s in streams.iter_mut() {
                crate::tp_serve::send_serving(s, &msg)?;
            }
            if admit_now { self.admit(pending.remove(0)); }
            // §3.5: the window advance sits at a FIXED POINT in the step — after the Step message
            // has been shipped and after admissions, before `decode_step`. That is what keeps every
            // prefill all-reduce pair-able by the mirror (the barrier structure changes in
            // ORDERING relative to decode's own collectives, never in COUNT).
            self.pf_advance();
            let b = self.num_active();
            if b > 0 { self.decode_step(b); self.tp_agree_step(step)?; }
            // A cursor can exist while b == 0 (the whole batch is prefilling): nothing to agree
            // about yet, but the step index advances on both ranks identically because the cursor
            // policy is a pure function of state (one cursor, one window, no wall clock).

            step += 1;
            // D3: serving-mode barrier histogram. --tp-trace already timestamps every barrier
            // (K1 duration / peer bounce / K2 wait / whole barrier / gap) and the bench path dumps
            // it at exit; this makes the same table available while SERVING, every 512 steps, so
            // the fixed per-step cost can be split sync vs read without a bench harness.
            if b > 0 && step % 512 == 0 && crate::opts::var(crate::opt!("tp-trace")).is_ok() {
                self.gpu.tp_trace_dump(&format!("serve step {step}"));
            }
            // Yield to tokio so streaming handlers can flush SSE events between decode steps.
            tokio::task::yield_now().await;
        }
    }

    /// TP=2 serving mirror loop (node, rank 1): block on the head's per-step events, replay them in
    /// order, and run the identical `decode_step`. Scheduler state stays identical BY CONSTRUCTION —
    /// the forward's all-reduces keep both ranks bitwise in lockstep and every admission/cancel is
    /// replayed from the wire at the same step index. Tokens decode into a dummy channel whose
    /// receiver is held in `keepalive` forever, so a mirror lane's `tx.is_closed()` never fires on
    /// its own: cancels arrive exclusively as wire events (see `Lane::tp_cancelled`).
    pub async fn run_tp_mirror(mut self, mut stream: std::net::TcpStream) -> anyhow::Result<()> {
        self.tp_serving = true;
        self.tp_drill_notice();
        let mut keepalive: Vec<mpsc::UnboundedReceiver<TokEvent>> = Vec::new();
        let mut step: u64 = 0;
        loop {
            let msg = match crate::tp_serve::recv_serving(&mut stream) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("[tp-mirror] session stream ended ({e:#}) — mirror exit at step {step}");
                    return Ok(());
                }
            };
            match msg {
                crate::tp_serve::ServingMsg::Shutdown => {
                    eprintln!("[tp-mirror] head shut down at step {step} — mirror exit");
                    return Ok(());
                }
                crate::tp_serve::ServingMsg::Step(se) => {
                    anyhow::ensure!(se.step == step,
                        "TP step desync: head sent step {}, mirror is at step {step}", se.step);
                    for ev in se.events {
                        match ev {
                            crate::tp_serve::TpEvent::Admit(w) => {
                                let (tx, rx, _bl) = crate::server::tok_channel(); // mirror keepalive; LR-5 gated off under TP
                                keepalive.push(rx);
                                self.admit(w.into_request(tx));
                            }
                            crate::tp_serve::TpEvent::RecordClear => {} // COPYREC1: EXL3 only (never shipped here)
                            crate::tp_serve::TpEvent::Cancel { lane } => {
                                match self.lanes.get_mut(lane).and_then(|l| l.as_mut()) {
                                    Some(l) => l.tp_cancelled = true,
                                    None => anyhow::bail!(
                                        "TP cancel for empty lane {lane} at step {step} — state desync"),
                                }
                            }
                        }
                    }
                    // Symmetric with the head: replay the admissions, advance the SAME one window
                    // of the SAME cursor, then decode. The replayed Admit events are the only input
                    // either rank uses, so the two cursors cannot diverge.
                    self.pf_advance();
                    let b = self.num_active();
                    if b > 0 { self.decode_step(b); self.tp_agree_step(step)?; }
                    step += 1;
                    tokio::task::yield_now().await;
                }
                other => anyhow::bail!("unexpected serving message on the step stream: {other:?}"),
            }
        }
    }

    /// True when the shared DFlash2 ring's frontier is exactly `pos` (the lane owns the ring).
    pub(super) fn df2_ring_at(&self, pos: usize) -> bool { self.df2.as_ref().map_or(false, |d| d.nprev() == pos) }

    /// The DSpark twin of `df2_ring_at`.
    pub(super) fn dspark_ring_at(&self, pos: usize) -> bool { self.dspark.as_ref().map_or(false, |d| d.nprev() == pos) }

    pub(super) fn num_active(&self) -> usize {
        self.lanes.iter().take_while(|l| l.is_some()).count()
    }

    /// S5F — record one speculation step (the τ-matrix harness's per-step telemetry). No-op
    /// unless `spec_steps_on` (set only by `run_spec_bench`).
    pub(super) fn rec_step(&mut self, r: SpecStepRec) {
        if self.spec_steps_on { self.spec_steps.push(r); }
    }

    /// S5F — the on-engine τ-matrix driver: runs the REAL scheduler loop over a list of jobs,
    /// switching the speculation source between jobs (ONE process, ONE model load — the matrix's
    /// amortization; each job is a fresh lane: admit → prefill (+ DFlash2 prime) → decode steps →
    /// finish). Exercises the exact code path the server runs (Phase A/B routing, the lane steps,
    /// EOS handling, the policy tick under a forced/pinned policy). Returns (token streams,
    /// per-step records per job) — the τ/tok-s/step-breakdown inputs.
    ///
    /// The scheduler is CONSUMED (single use, like `run()`); the tokio runtime must be a
    /// current-thread runtime (the scheduler + the round are single-task by contract).
    pub async fn run_spec_bench(mut self, jobs: Vec<SpecBenchJob>, dump_tags: &[String])
        -> (Vec<Vec<u32>>, Vec<Vec<SpecStepRec>>, Vec<f32>) {
        assert!(self.step_dump.is_none() || dump_tags.len() == jobs.len(),
                "run_spec_bench: dump_tags must align with jobs");
        let mut streams: Vec<Vec<u32>> = Vec::with_capacity(jobs.len());
        let mut step_recs: Vec<Vec<SpecStepRec>> = Vec::with_capacity(jobs.len());
        let mut walls: Vec<f32> = Vec::with_capacity(jobs.len());
        for (k, job) in jobs.into_iter().enumerate() {
            self.mtp.set_spec_source(job.source);
            self.spec_steps.clear();
            self.spec_steps_on = true;
            if let Some(d) = self.step_dump.as_mut() {
                d.job_start(&dump_tags[k], &job.prompt, self.gpu.dev());
            }
            let (tx, mut rx, bl) = crate::server::tok_channel(); // LR-5 (H6): drained counted below
            let job_t0 = std::time::Instant::now();
            let job_plen = job.prompt.len();
            self.admit(BatchRequest {
                prompt: job.prompt,
                max_new: job.max_new,
                temperature: job.temperature,
                top_p: job.top_p,
                top_k: job.top_k,
                rep_penalty: 1.0,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                min_p: 0.0,
                min_new: 0,
                ignore_eos: false,
                tx,
                seed: Some(job.seed),
                ckpt_at: None,
                domain: job.domain,
                received_at: std::time::Instant::now(),
                image_embeds: None,
                image_spans: Vec::new(),
                schema: None,
                thinking_budget: None,
                logprobs: None,
            });
            // The bench harness has no scheduler loop of its own, so it drains the cursor inline —
            // the SAME window sequence (one window per call), just without interleaving. This keeps
            // `run_spec_bench` byte-equivalent to the pre-cursor inline admit.
            while self.pf.is_some() { self.pf_advance(); }
            // The prefill filled the prime sink — copy the prompt's tap rows into the dump.
            if let Some(d) = self.step_dump.as_mut() {
                d.job_prime(job_plen, self.df2_prime.as_ref(), self.gpu.dev());
            }
            // Step until the lane finishes (the admit may have REJECTED a too-long prompt — the
            // lane count stays 0 and the Finish event already carries the reason).
            loop {
                let b = self.num_active();
                if b == 0 { break; }
                self.decode_step(b);
                tokio::task::yield_now().await;
            }
            let mut toks = Vec::new();
            while let Some(ev) = crate::server::try_recv_counted(&mut rx, &bl) { // LR-5 (H6)
                match ev {
                    TokEvent::Tok(t) => toks.push(t),
                    TokEvent::Finish { .. } => {}
                    // #8.2: admission telemetry — no action for a bench drain.
                    TokEvent::Admitted { .. } => {}
                    TokEvent::StopInfo { .. } => {} // [gen]-line data only
                    TokEvent::Logprob { .. } => {} // not requested by this drain
                }
            }
            // Keep the source set for the next job's ADMIT (the lane priming consults it).
            self.spec_steps_on = false;
            if let Some(d) = self.step_dump.as_mut() { d.job_end(); }
            walls.push(job_t0.elapsed().as_secs_f32());
            streams.push(toks);
            step_recs.push(std::mem::take(&mut self.spec_steps));
        }
        if let Some(d) = self.step_dump.as_mut() { d.finish(); }
        (streams, step_recs, walls)
    }

    /// Prefill `req` into a free physical slot; emits the first generated token to the client.
    ///
    /// PREFIX REUSE. OpenWebUI and opencode resend the ENTIRE conversation every turn, so without this a
    /// chat pays prefill of its whole history on every turn — per-turn TTFT grows linearly and a session
    /// costs O(T²) in total prefill. We pick the free slot whose cached sequence is the longest prefix of
    /// this prompt and prefill only the suffix.
    /// PLAN/25 Phase 0: flip the coverage-trace op sequence on a rank whose `step_dump` is None
    /// (the TP node mirror — the flag arrives via TpConfig; on the head it is implied by the dump).
    pub fn set_cov_trace(&mut self, on: bool) { self.cov_trace = on; }
}
