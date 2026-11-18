//! `impl GpuModel`: stream sync, DFlash2 / DSpark capture and prime-sink setters and the `df2_*` borrow helpers that hand
//! the trunk dimensions and device pointers to the drafter. Map: `mod.rs`.

use super::*;

impl GpuModel {
    /// Synchronize the inference stream (blocks until all GPU work on our stream completes).
    /// CPUIDLE: the wait follows --gpu-wait (crate::host_wait); each call site keeps its own prediction.
    #[track_caller]
    pub fn sync_stream(&self) {
        crate::host_wait::stream_wait(self.stream.stream).unwrap();
    }

    /// S4F: install the DFlash2 tap-capture sink on a model loaded WITHOUT `--df2-capture`
    /// (the single-process `--probe-df2-round` path — the trunk and drafter load as two
    /// independent objects; the probe owns the sink and hands the trunk its writer twin).
    /// After this, the trunk's decode/verify forwards copy the DFlash2 tap layers.
    pub fn set_df2_capture(&mut self, sink: std::sync::Arc<crate::dflash2::capture::Df2TapSink>) {
        self.df2_capture = Some(sink);
    }

    /// S4F: disarm the DFlash2 tap capture (the probe's capture-off control).
    pub fn set_df2_capture_off(&mut self) {
        self.df2_capture = None;
    }
    /// PLAN/25 Phase 1: arm the WIDE tree sink (the topo verify's n > BLOCK columns).
    pub fn set_df2_capture_tree(&mut self, sink: std::sync::Arc<crate::dflash2::capture::Df2TapSink>) {
        self.df2_capture_tree = Some(sink);
    }

    /// S4F: is the DFlash2 tap capture armed? (the probe's negative controls toggle it).
    pub fn df2_capture_armed(&self) -> bool {
        self.df2_capture.is_some()
    }

    /// S5F: install the DFlash2 PROMPT-PRIME sink (the prefill path's wide tap buffer). Armed
    /// only while a DFlash2 lane is being admitted; `cols` = the max prefill window width.
    pub fn set_df2_prime_sink(&mut self, sink: std::sync::Arc<crate::dflash2::capture::Df2PrimeSink>) {
        self.df2_prime = Some(sink);
    }

    /// S5F: disarm the prompt-prime capture (after the DFlash2 lane's admit completes — the
    /// prefill capture must not run for non-DFlash2 lanes or other requests).
    pub fn set_df2_prime_off(&mut self) {
        self.df2_prime = None;
    }

    // ---- WI1: the DSpark twins (same shapes, dspark::TAP_LAYERS layers) ----

    /// Diagnostic: the DSpark tap-layer index shift (--dspark-tap-shift, default 0). The
    /// artifact's target_layer_ids semantics (post-FFN residual of layer l = hidden_states[l+1])
    /// vs (the layer's input = hidden_states[l]) differ by exactly one capture site; the shift
    /// lets the serving acceptance measurement decide which the trainer used.
    pub fn dspark_tap_shift() -> i32 {
        static S: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
        *S.get_or_init(|| {
            crate::opts::var(crate::opt!("dspark-tap-shift")).ok().and_then(|v| v.parse::<i32>().ok()).unwrap_or(0)
        })
    }

    pub fn set_dspark_capture(&mut self, sink: std::sync::Arc<crate::dflash2::capture::Df2TapSink>) {
        self.dspark_capture = Some(sink);
    }
    pub fn set_dspark_capture_off(&mut self) {
        self.dspark_capture = None;
    }
    pub fn set_dspark_capture_tree(&mut self, sink: std::sync::Arc<crate::dflash2::capture::Df2TapSink>) {
        self.dspark_capture_tree = Some(sink);
    }
    pub fn set_dspark_prime_sink(&mut self, sink: std::sync::Arc<crate::dflash2::capture::Df2PrimeSink>) {
        self.dspark_prime = Some(sink);
    }
    pub fn set_dspark_prime_off(&mut self) {
        self.dspark_prime = None;
    }

    /// S4F: the TRUNK's lm_head + embed_tokens as device pointers for the DFlash2 round's
    /// borrowed head / block-input embed gather, in either supported trunk dtype:
    /// `BorrowedW::Nvfp4` (MMA-repacked qweight/scales + per-tile f32 gs — the serving quant's
    /// contract) or `BorrowedW::Bf16` (the plain-BF16 trunk class — the S5F2 L0 dtype
    /// diagnostic; the round dispatches to `embed_gather_b` / `gemm_binv_b`). Returns None
    /// when either tensor is neither NVFP4 nor BF16.
    ///
    /// S9F (TP-DF2 leg): under TP with [df2-tp]=1 the round gets the FULL pre-shard lm_head
    /// (`df2_full_head` — the rank-local half is vocab-sharded and would draft from v/world
    /// rows); single-node and non-DF2 TP fall back to the plain `lm_head`.
    pub fn df2_borrow(&self) -> Option<(crate::dflash2::round::BorrowedW, crate::dflash2::round::BorrowedW)> {
        use crate::dflash2::round::BorrowedW;
        let ptrs = |w: &W| -> Option<BorrowedW> {
            match w {
                W::Nvfp4 { qweight, scales, gs, .. } => Some(BorrowedW::Nvfp4(crate::dflash2::round::Nvfp4Ptrs {
                    qweight: *qweight.device_ptr() as u64,
                    scales: *scales.device_ptr() as u64,
                    gs: *gs.device_ptr() as u64,
                })),
                W::Bf16(b) => Some(BorrowedW::Bf16 { ptr: *b.device_ptr() as u64 }),
                _ => None,
            }
        };
        // F8-CATCHUP two-stage draft head (DEFAULT ON since the 2026-09-06 battery): a Bf16
        // borrow costs the round a 2.54 GB `gemm_binv_b` pass (measured 13.5 ms = 45% of the
        // round). The two-stage path (coarse NVFP4 0.68 GB -> top-256 shortlist -> EXACT bf16
        // re-rank vs the original hiddens) measured OUTPUT-EXACT (served-text sha byte-identical
        // to the bf16 head) on FOUR paired cells — cod1 chat ×3 (+3.1%), synthetic (+4.7%),
        // gsm1 math (+4.2%), ~4.6K-token long-context (+3.3%) — with acceptance/τ at bf16 level.
        // The plain NVFP4-head path (no re-rank) measured a τ wash and is NOT used by default.
        // Escape hatch (diagnostic): --no-df2-q2stage=1 keeps the bf16 `gemm_binv_b` path;
        // --df2-qhead alone (without Q2STAGE) still selects the plain NVFP4 head for A/B.
        let head_w: &W = match &self.df2_full_head {
            Some(h) => h,
            None => self.lm_head.as_ref()?,
        };
        // Semantics: default = two-stage; --no-df2-q2stage=1 = plain bf16 escape hatch;
        // --df2-qhead=1 (without the NO_ escape) = plain NVFP4 head, no re-rank (A/B arm).
        let is_bf16_borrow = matches!(head_w, W::Bf16(_));
        let q2s_off = crate::opts::var(crate::opt!("no-df2-q2stage")).is_ok();
        let want_2stage = is_bf16_borrow && !q2s_off;
        let want_plain_q = is_bf16_borrow && !q2s_off && crate::opts::var(crate::opt!("df2-qhead")).is_ok();
        let head = if want_2stage || want_plain_q {
            match self.df2_q_head_borrow() {
                Some(W::Nvfp4 { qweight, scales, gs, .. }) if want_2stage => {
                    let W::Bf16(b) = head_w else {
                        unreachable!("TwoStage arm requires a Bf16 borrow");
                    };
                    BorrowedW::TwoStage {
                        q: crate::dflash2::round::Nvfp4Ptrs {
                            qweight: *qweight.device_ptr() as u64,
                            scales: *scales.device_ptr() as u64,
                            gs: *gs.device_ptr() as u64,
                        },
                        bf16_ptr: *b.device_ptr() as u64,
                    }
                }
                Some(w) => ptrs(w)?,
                None => ptrs(head_w)?,
            }
        } else {
            ptrs(head_w)?
        };
        let embed = ptrs(&self.embed)?;
        Some((head, embed))
    }

    /// F8-CATCHUP: the lazily-built NVFP4-MMA copy of a Bf16 borrowed head (None when the head
    /// is not Bf16 — quantized trunks already draft from their packed head). One ~4 s host
    /// round-trip at first borrow; every later round reads 0.68 GB instead of 2.54 GB.
    fn df2_q_head_borrow(&self) -> Option<&W> {
        self.df2_q_head.get_or_init(|| self.build_df2_q_head()).as_ref()
    }

    fn build_df2_q_head(&self) -> Option<W> {
        // TP FIX (2026-09-06, caught by the WI0 plain-server TP=2 boot): the quantize source
        // MUST be the same full-vocab tensor the borrow hands the round. Under TP the serving
        // `lm_head` is the rank-local vocab shard (E7, v/world rows), while `df2_full_head`
        // holds the FULL pre-shard head (S9F twin, bf16-head class). Quantizing the shard
        // with the full `m` fired quantize_nvfp4's shape assert and crashed BOTH ranks at
        // TP=2 boot (master dea4249+; single-node was unaffected — shard == full there).
        let head_src: &W = match &self.df2_full_head {
            Some(h) => h,
            None => self.lm_head.as_ref()?,
        };
        let b = match head_src {
            W::Bf16(b) => b,
            _ => return None, // quantized full head: the borrow already uses it directly
        };
        let (m, k) = (self.cfg.vocab_size, self.cfg.hidden_size);
        if m % 16 != 0 || k % 16 != 0 {
            return None;
        }
        eprintln!("[df2] quantizing borrowed head bf16->NVFP4-mma in memory ({m} x {k}) — one-time");
        let t0 = std::time::Instant::now();
        let host: Vec<half::bf16> = self.dev.dtoh_sync_copy(b).ok()?;
        crate::quant::set_repack_threading(true);
        let t = crate::quant::quantize_nvfp4(&host, m, k);
        drop(host);
        let (wt, st) = crate::quant::repack_nvfp4_mma(&t.qweight, &t.scales, m, k);
        let inv_gs = 1.0 / t.global_scale; // kernel convention: gs = 1/global_scale per 16-row tile
        let gsv = vec![inv_gs; m / 16];
        let qweight = self.dev.htod_sync_copy(&wt).ok()?;
        let scales = self.dev.htod_sync_copy(&st).ok()?;
        let gs = self.dev.htod_sync_copy(&gsv).ok()?;
        eprintln!("[df2] head quantized in {:.1}s ({} MB packed, was {} MB bf16) — draft head now reads ~1/3.7x",
                  t0.elapsed().as_secs_f32(), wt.len() / 1_000_000, m * k * 2 / 1_000_000);
        Some(W::Nvfp4 { qweight, scales, gs, m, k })
    }

    /// Whether the exact lm_head borrowed by DFlash2 is marked Hadamard16-rotated.
    pub fn df2_head_hadamard16(&self) -> bool {
        self.df2_full_head.as_ref().or(self.lm_head.as_ref())
            .map(|w| self.is_rotated(w))
            .unwrap_or(false)
    }

    /// S10' §4 — is this trunk dimension-compatible with the DFlash2 round? The round (the
    /// 3.8-27B drafter artifact) is hard-wired to HIDDEN=5120 (tap concat + fc + head GEMM),
    /// VOCAB=248320 (the borrowed lm_head's row count the head GEMM reads), and TAP_LAYERS
    /// [5,19,33,47,61] (the captured trunk layers). A smaller/different trunk (any older model)
    /// would OOB the borrowed head/embed at the first round — the round must simply not load.
    /// The check is dimension-only by design: a dimension-compatible trunk (e.g. 3.6-27B) may
    /// load the round (lossless by construction — the verify checks every draft); only its τ
    /// quality is the user's choice.
    pub fn df2_round_compatible(&self) -> bool {
        crate::dflash2::geom().trunk_compatible(
            self.cfg.hidden_size,
            self.cfg.vocab_size,
            self.cfg.num_layers,
        )
    }

    /// S10' §4 — the trunk's (hidden, vocab, layers) for the compat-guard WARN line.
    pub fn df2_trunk_dims(&self) -> (usize, usize, usize) {
        (self.cfg.hidden_size, self.cfg.vocab_size, self.cfg.num_layers)
    }

    /// S4F: the dtype-generic borrow narrowed to the NVFP4 pair (the probes that require the
    /// MMA-repacked head/embed contract, e.g. the mirror probes that decode the packed bytes).
    pub fn df2_borrow_ptrs(&self) -> Option<(crate::dflash2::round::Nvfp4Ptrs, crate::dflash2::round::Nvfp4Ptrs)> {
        use crate::dflash2::round::BorrowedW;
        match self.df2_borrow()? {
            (BorrowedW::Nvfp4(h), BorrowedW::Nvfp4(e)) => Some((h, e)),
            _ => None,
        }
    }

    /// P2 (round sharding): the DFlash2 round's TP all-reduce context — a rank-local snapshot
    /// of the transport surfaces the round's sharded path drives (the faithful mirror of
    /// `tp_all_reduce_bf16`'s world>2 branch, launched on the ROUND's stream). The buffers are
    /// process-lifetime (the transport ctx device struct + the persistent fp32 scratch), so the
    /// handle stays valid across the round's CUDA-graph capture/replays. None when TP is not
    /// attached or world <= 2 (the round shard engages only at world > 2 — the quad target;
    /// TP=2 qwen+DF2 is not a shipped configuration).
    pub fn df2_ar_ctx(&self) -> Option<crate::dflash2::round::Df2ArCtx> {
        if self.tp_world <= 2 || self.tp_ctx_dptr == 0 {
            return None;
        }
        Some(crate::dflash2::round::Df2ArCtx {
            ctx_dptr: self.tp_ctx_dptr,
            f32_scratch: self.tp_f32_scratch_ptr(),
            world: self.tp_world,
            rank: self.tp_rank,
            k2_gpu_recv: self.tp_gpu_recv_on(),
            oneshot: self.tp_oneshot_on(),
            payload_bytes: self.tp_default_payload_bytes(),
        })
    }
}
