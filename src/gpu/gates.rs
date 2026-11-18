//! `impl GpuModel`: feature gates read from the option registry (TP recv / reduce-fuse, MoE variants, GDN prefill modes)
//! and `launch_moe_fold_gemm`, `ffn_batch`. Map: `mod.rs`.

use super::*;

impl GpuModel {
    /// Dispatch a layer's FFN: dense MLP or MoE block. Keeps the four batched forward sites uniform.
    /// `sharded` says whether THIS ffn's weights were split by `tp_shard_weights`. It must not be
    /// inferred from `tp_world`: the MTP head is replicated even under TP=2, so a global flag would
    /// halve `im` on a full-width weight and then all-reduce two identical halves — wrong output, and
    /// wrong timing too. Sharding is a property of the tensor, not of the process.
    /// v2 receive mode (EXPERT_GPU_ALLREDUCE §8): GPU-direct payload-tail validation (K2') instead
    /// of the CPU-bounced cpu_done gate. Resolved ONCE per process (env-first, then the shipped
    /// TpConfig, default OFF until the §12 battery passes on both boxes); both ranks resolve from
    /// the same shipped config (SPMD). --tp-gpu-recv=0/1 overrides; the node installs the env
    /// from the shipped config before load like every other SPMD knob.
    pub(super) fn tp_gpu_recv_on(&self) -> bool {
        static G: std::sync::LazyLock<Option<bool>> = std::sync::LazyLock::new(|| {
            if let Ok(v) = crate::opts::var(crate::opt!("tp-gpu-recv")) {
                let v = v.trim().to_ascii_lowercase();
                return match v.as_str() {
                    "1" | "on" | "true" => Some(true),
                    "0" | "off" | "false" => Some(false),
                    _ => None,   // garbage: treat as unset (config + world force)
                };
            }
            None
        });
        // R9 (world>2): force the v2 GPU-direct receive. The v1 CPU-bounce gate (`cpu_done[partner]
        // >= e`) is a per-partner WATERMARK — a backlog drain that releases a later epoch of the same
        // partner lets K2 pass `>= e` for THIS epoch before its slot is validated, so K2 consumes a
        // stale/crossed payload (the world=4 divergence + the deadlock the R9 checksum rings caught:
        // rank1<->rank3 payloads swapped at the stall epoch). The v2 gate validates the slot's
        // generation-tagged TAIL by EQUALITY (tp_wait_add_g), which cannot pass for the wrong epoch.
        // world==2 keeps the proven v1 CPU-bounce path (single QP, single watermark — sound there).
        match *G {
            // B8 §1.7-3: an explicit --tp-gpu-recv=0/off/false now FORCES v1 even at world>2 —
            // previously the `*G || world>2` force swallowed it, so the =0 escape hatch was dead.
            Some(v) => v,
            None => crate::tp::tp_config().map(|c| c.gpu_recv.unwrap_or(false)).unwrap_or(false)
                        || self.tp_world > 2,
        }
    }
    pub(crate) fn tp_gpu_recv_on_pub() -> bool {
        // Instance-free twin for the maxloc kernel selector: world comes from the shipped TpConfig
        // (SPMD — identical on every rank), matching the world>2 force in `tp_gpu_recv_on`.
        static G: std::sync::LazyLock<Option<bool>> = std::sync::LazyLock::new(|| {
            if let Ok(v) = crate::opts::var(crate::opt!("tp-gpu-recv")) {
                let v = v.trim().to_ascii_lowercase();
                return match v.as_str() {
                    "1" | "on" | "true" => Some(true),
                    "0" | "off" | "false" => Some(false),
                    _ => None,
                };
            }
            None
        });
        match *G {
            // B8 §1.7-3: =0/off/false forces v1 even at world>2 (the old =0 escape hatch was dead).
            Some(v) => v,
            None => {
                let world = crate::tp::tp_config().map(|c| c.world).unwrap_or(2);
                crate::tp::tp_config().map(|c| c.gpu_recv.unwrap_or(false)).unwrap_or(false) || world > 2
            }
        }
    }

    /// AR landing 2: fused reduce+residual+norm epilogue (--tp-reduce-fuse; value-based — any
    /// value except "0" enables, matching the TpConfig read). The fused kernel replaces the
    /// K2/K2' + fused_res_rmsnorm_b(/q) two-launch chain at the mixer/FFN epilogue sites. The FFN
    /// site is ALSO gated on `fuse_residual_norm()` (§4 — the unfused add_residual_b + rmsnorm_b
    /// chain has a DIFFERENT rounding placement, so a fused run with the unfused config would not
    /// be bit-equal); the mixer site is not (its norm kernel is always the fused flavor-S one).
    /// Both ranks resolve from the same shipped config (SPMD). DEFAULT OFF until the §12 battery.
    pub(super) fn tp_reduce_fuse_on() -> bool {
        static R: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
            if let Ok(v) = crate::opts::var(crate::opt!("tp-reduce-fuse")) {
                return v != "0";
            }
            crate::tp::tp_config().map(|c| c.reduce_fuse.unwrap_or(false)).unwrap_or(false)
        });
        *R
    }

    /// E16: first batch width that routes to the device-only grouped-MoE path (the crossover A/B
    /// knob — grouped wins once tokens/expert > 1; the doc's measured-baseline question is whether
    /// that is N=2, 4, or 8). Default 17 = the SLOT arm for every decode/verify width 2..=16:
    /// EXPERT_BATCH_SCALING_RESPONSE (2026-08-12) settled the crossover — with the E16 static-bound
    /// elementwise passes (gather/quant/silu over ppad_max=6304 rows) the grouped arm never wins at
    /// any width 2..=16 on hy3: at batch=4 it pays ~212 MB/layer of padding passes to dedup ~8% of
    /// expert reads (170 vs 157 MB of weights), ~66 ms of a 190 ms step. The slot arm is the pre-E16
    /// shipping path and the N=1 bit-identity reference (grouped was proven bit-identical TO it).
    /// --moe-grouped-min overrides without a rebuild (env must be in 2..=127 — 17 = never grouped
    /// at verify widths; values >16 were silently ignored pre-2026-08-12 because the filter capped
    /// at MAX_VERIFY). Read once: the dispatch choice is a host-side shape constant, stable under
    /// graph capture. NOTE: the verify arm change alters MTP r(d) — let the policy re-calibrate.
    /// 2026-08-17: native (W4A4) grouped MoE at PREFILL widths — measured 3-9% TP-prefill
    /// REGRESSION vs the bf16 mma grouped kernel (B2-gate A/B #2). Default OFF at wide batches;
    /// --moe-native-pf=1 restores the old behavior for A/B.
    pub(super) fn moe_native_pf_on() -> bool {
        static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *V.get_or_init(|| crate::opts::var(crate::opt!("moe-native-pf")).is_ok())
    }

    /// P4 B3 chunked GDN prefill (2026-08-17).
    /// 2026-09-06 DEFAULT ON (F8 catch-up, measured on master @ 10631-tok prefill: 11768 ms →
    /// 9806 ms, −16.7%, with binv PASS / state EXACT / LOSSLESS_OK; the identical flip is already
    /// proven on the parked PP-serve line 7d5e4c1). Prefill-only — decode byte-identical by
    /// construction. `--gdn-chunk2=0` restores the legacy sequential scan (VALUE check, not
    /// existence — the idiom that burned a benchmark session).
    /// S-A3-w: --gdn-xcheck for EVERY chunked GDN prefill arm (scalar / tensor-core / split).
    /// Runs the sequential delta_step_prefill from the SAME S_in snapshot on the same qkv/b/a into a
    /// fresh core, then prints the per-layer LOCAL divergence (o and S rel-L2) of the chunked arm.
    /// Was only wired on the scalar arm — the default tensor-core arm was never cross-checked, and it
    /// carried the A-gram decay off-by-one (S-A3-v/w).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gdn_xcheck_seq(&self, pool: &mut Pool, li: usize, tag: &str, core: &B, s_copy: S, s_ptr: u64,
                      qkv: u64, b: u64, a: u64, stride_pack: u32, kd: usize, vd: usize,
                      alog: u64, dtb: u64, n_nkh: i32, n: usize, lin_nh: usize, nchunk: usize, smem: u32) {
        let s_words = lin_nh * kd * vd;
        let core2 = pool.get_bf16(lin_nh * vd * n);
        blaunch!(self, "delta_step_prefill", ((lin_nh*nchunk) as u32,1,1), (kd as u32,1,1), smem,
            (*core2.device_ptr(), qkv, *s_copy.device_ptr(), b, a, stride_pack as i32,
             (kd as i32) | ((vd as i32) << 16), alog, dtb, n_nkh, 0u64, 0u64));
        let s_real = pool.get(s_words);
        unsafe { cudarc::driver::result::memcpy_dtod_async(
            *s_real.device_ptr(), s_ptr, s_words * 4, self.stream.stream)
            .expect("xcheck S readback dtod"); }
        self.sync_stream();
        let o1 = self.dev.dtoh_sync_copy(core).expect("xcheck core dtoh");
        let o2 = self.dev.dtoh_sync_copy(&core2).expect("xcheck core2 dtoh");
        let (mut dd, mut nn) = (0f64, 0f64);
        for i in 0..lin_nh * vd * n {
            let x = o1[i].to_f32() as f64; let y = o2[i].to_f32() as f64;
            dd += (x - y) * (x - y); nn += y * y;
        }
        let s1 = self.dev.dtoh_sync_copy(&s_real).expect("xcheck s_real dtoh");
        let s2 = self.dev.dtoh_sync_copy(&s_copy).expect("xcheck s_copy dtoh");
        let (mut sd, mut sn, mut smax) = (0f64, 0f64, 0f32);
        for i in 0..s_words {
            let x = s1[i] as f64; let y = s2[i] as f64;
            sd += (x - y) * (x - y); sn += y * y; smax = smax.max((s1[i] - s2[i]).abs());
        }
        eprintln!("[gdn-xcheck] {tag} layer {:>2} n {n}  o rel-L2 {:.3e}   S rel-L2 {:.3e}   S max|d| {:.3e}",
                  li, (dd / nn.max(1e-30)).sqrt(), (sd / sn.max(1e-30)).sqrt(), smax);
        pool.release_bf16(core2, lin_nh * vd * n);
        pool.release(s_real, s_words);
        pool.release(s_copy, s_words);
    }

    pub(super) fn gdn_chunk2_prefill_on() -> bool {
        static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *V.get_or_init(|| match crate::opts::var(crate::opt!("gdn-chunk2")) {
            Ok(v) => !matches!(v.as_str(), "" | "0" | "false" | "off"),
            Err(_) => true,
        })
    }
    pub(super) fn gdn_chunk_prefill_on() -> bool {
        static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        // VALUE check, not existence: `--gdn-chunk=0` must mean OFF. The `.is_ok()` existence
        // idiom burned a benchmark session (both A/B arms ran chunked because the off arm set =0).
        *V.get_or_init(|| match crate::opts::var(crate::opt!("gdn-chunk")) {
            Ok(v) => !matches!(v.as_str(), "" | "0" | "false" | "off"),
            Err(_) => false,
        })
    }

    /// F8 P2c' (2026-09-06): SPLIT chunked GDN prefill — the fused gdn_chunk_tc_b walks chunks
    /// serially in a 48-block grid; the split runs the per-chunk grams/W/subst/O in a
    /// (chunks × heads × vchunks)-parallel kernel after a LEAN serial state-recurrence kernel
    /// materializes every chunk's entry state (FlashInfer's decomposition). MEASURED REGRESSION
    /// at n=10631 (this box): split H+WUO ≈ 21.7-22.3 ms/layer vs the fused gdn_chunk_tc_b at
    /// 10.4-11.2 ms/layer (--gdn-time, both idle) — the fused kernel's SMEM-resident state
    /// plus all-MMA chunks beat the split's 522 MB/layer of state-scratch traffic. Default OFF;
    /// kept as a DIAGNOSTIC variant (--gdn-split=1 to A/B). The analysis doc's GDN-first
    /// priority was based on a stale ~37 ms/layer estimate that this measurement refutes.
    pub(super) fn gdn_split_prefill_on() -> bool {
        static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *V.get_or_init(|| match crate::opts::var(crate::opt!("gdn-split")) {
            Ok(v) => !matches!(v.as_str(), "" | "0" | "false" | "off"),
            Err(_) => false,
        })
    }

    pub(super) fn moe_grouped_min() -> usize {
        static N: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
            crate::opts::var(crate::opt!("moe-grouped-min")).ok()
                .and_then(|v| v.parse().ok()).filter(|n| (2..=127).contains(n)).unwrap_or(17)
        });
        *N
    }

    /// Fused activation-quant dispatch (EXPERT_FUSED_QUANT_RESPONSE.md §8/R9): ONE env-once
    /// predicate per process, read once (static, like `moe_grouped_min`). --mxfp4-fused=0
    /// escapes EVERY site to today's separate quant(+silu)+GEMM launches (byte-identical).
    /// Default OFF (2026-08-12): --probe-mxfp4-fused hit CUDA_ERROR_ILLEGAL_ADDRESS at
    /// production shapes (dense N=1 K=1536) — the fused kernels are NOT yet cleared for the
    /// serving path; --mxfp4-fused=1 opts the A/B in until the OOB is fixed and the gate
    /// chain (probe + binv + LOSSLESS + xchain2) passes.
    pub(super) fn mxfp4_fused_on() -> bool {
        static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
            crate::opts::var(crate::opt!("mxfp4-fused")).map_or(false, |v| v == "1")
        });
        *ON
    }

    /// Prefill-arm escape (§10 R3): --mxfp4-fused-prefill=0 keeps the separate pair at the
    /// grouped-prefill windows (the L2-thrash contingency for the widest x_perm windows);
    /// default fused.
    pub(super) fn mxfp4_fused_prefill_on() -> bool {
        static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
            crate::opts::var(crate::opt!("mxfp4-fused-prefill")).map_or(true, |v| v != "0")
        });
        *ON
    }

    /// E12/E13: is the shared-expert fold enabled? DEFAULT OFF (2026-08-11, E11 matrix): fold-on +
    /// MTP degenerates hy3 output (the fold's MTP-head draft path diverges wrong-but-consistent and
    /// the verify accepts the garbage — repetition loops; every self-consistency gate stayed green).
    /// --moe-fold=1 opts in; the legacy --moe-no-fold=1 escape still wins. Read once per
    /// process — the fold's launch sequence must be stable under CUDA-graph capture, so it cannot
    /// flip between decode and verify.
    pub(super) fn moe_fold_on() -> bool {
        static F: std::sync::LazyLock<bool> =
            std::sync::LazyLock::new(|| {
                let opt_in = crate::opts::var(crate::opt!("moe-fold")).map_or(false, |v| v == "1");
                let escape = crate::opts::var(crate::opt!("moe-no-fold")).map_or(true, |v| v != "1");
                opt_in && escape
            });
        *F
    }

    /// E11: the slot-fold GEMM variant selector. --moe-variant=plain|u4|x2|rast|lb5|lb4|pdl
    /// (default plain = the shipped fold kernel). Read once per process (graph-capture stability).
    fn moe_variant() -> &'static str {
        static V: std::sync::LazyLock<&'static str> = std::sync::LazyLock::new(|| {
            let v = crate::opts::var(crate::opt!("moe-variant")).unwrap_or_else(|_| "plain".to_string());
            match v.as_str() {
                "plain" | "u4" | "x2" | "rast" | "lb5" | "lb4" | "pdl" => {}
                other => panic!("--moe-variant: unknown variant '{other}' (plain|u4|x2|rast|lb5|lb4|pdl)"),
            }
            Box::leak(v.into_boxed_str())
        });
        *V
    }

    /// E11: the decode-arm slot-fold GEMM launch through the configured variant. Every variant
    /// preserves the exact per-element k-chain (pair partition p = warp + 8k ascending, per-pair
    /// block order 2p then 2p+1, per-warp fp32 accumulation, fixed warp-order epilogue reduce) —
    /// only the block→tile map or instruction scheduling differ, so col-0 bit-identity and the
    /// oracle-identical MoE gate are untouched. `g` is the PLAIN (M/16, bkf) grid; x2 halves gx
    /// (requires M%32==0), rast swaps the axes, pdl launches the E9 programmatic-stream-secondary
    /// kernel (down GEMM only — `pdl_ok`), falling back to the plain path when E9 is off.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch_moe_fold_gemm(&self, name_plain: &str, g: (u32, u32), pdl_ok: bool,
                            a: (u64, u64, u64, u64, u64, u64, i32, i32, i64, i32, i32, u64)) {
        let v = Self::moe_variant();
        let grid = match v {
            "u4"   => (g.0, g.1, 1),
            "x2"   => { assert!(g.0 % 2 == 0, "x2 fold variant needs M%32==0 (grid.x even)"); (g.0 / 2, g.1, 1) }
            "rast" => (g.1, g.0, 1),
            "lb5"  => (g.0, g.1, 1),
            "lb4"  => (g.0, g.1, 1),
            "pdl"  => (g.0, g.1, 1),
            _      => (g.0, g.1, 1),
        };
        let name = match v {
            "u4"   => "gemm_moe_mma_fp4_fold_u4",
            "x2"   => "gemm_moe_mma_fp4_fold_x2",
            "rast" => "gemm_moe_mma_fp4_fold_rast",
            "lb5"  => "gemm_moe_mma_fp4_fold_lb5",
            "lb4"  => "gemm_moe_mma_fp4_fold_lb4",
            "pdl"  => "gemm_moe_mma_fp4_fold_pdl",
            _      => name_plain,
        };
        if v == "pdl" {
            // Programmatic-stream-secondary launch: the down GEMM's weight stream (independent of
            // the silu's output) may start while the silu still runs; the kernel's prologue
            // griddepcontrol.wait gates the X reads. Falls back to the plain pdl=0 launch when E9
            // is off (single-node / escape / module-load failure) — byte-identical to the fold.
            if pdl_ok {
                if let Some(f) = self.e9_handle(name) {
                    // Named locals for every expression as_kernel_param() points into (see launch_pss).
                    let (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11) = a;
                    let (m1, k1, kx, b1, e1, sd) = (a6, a7, a8, a9, a10, a11);
                    let pdl_one = 1i32;
                    let mut params = [
                        a0.as_kernel_param(), a1.as_kernel_param(), a2.as_kernel_param(),
                        a3.as_kernel_param(), a4.as_kernel_param(), a5.as_kernel_param(),
                        m1.as_kernel_param(), k1.as_kernel_param(), kx.as_kernel_param(),
                        b1.as_kernel_param(), e1.as_kernel_param(), sd.as_kernel_param(),
                        pdl_one.as_kernel_param(),
                    ];
                    unsafe { self.launch_pss(f, grid, (256, 1, 1), 0, &mut params); }
                    return;
                }
            }
            let (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11) = a;
            let pdl_zero = 0i32;
            let mut params = [
                a0.as_kernel_param(), a1.as_kernel_param(), a2.as_kernel_param(),
                a3.as_kernel_param(), a4.as_kernel_param(), a5.as_kernel_param(),
                a6.as_kernel_param(), a7.as_kernel_param(), a8.as_kernel_param(),
                a9.as_kernel_param(), a10.as_kernel_param(), a11.as_kernel_param(),
                pdl_zero.as_kernel_param(),
            ];
            unsafe { self.launch_raw(name, grid, (256, 1, 1), 0, &mut params); }
            return;
        }
        blaunch!(self, name, grid, (256,1,1), 0,
            (a.0, a.1, a.2, a.3, a.4, a.5, a.6, a.7, a.8, a.9, a.10, a.11));
    }

    /// `fuse` = AR landing 2: the FFN epilogue runs the fused `tp_reduce_resnorm_b`, so the final
    /// reduce may stop after K1 and hand the LOCAL partial up instead of the full-rank bf16 output.
    pub(super) fn ffn_batch(&self, pool: &mut Pool, x: &B, ffn: &Ffn, batch: usize, sharded: bool, fuse: bool) -> ReduceOut {
        match ffn {
            Ffn::Dense(m)   => self.mlp_batch(pool, x, m, batch, sharded, fuse),
            Ffn::Moe(moe)   => self.moe_batch(pool, x, moe, batch, fuse),
        }
    }
}
