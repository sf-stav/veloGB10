//! `impl GpuModel`: `prefill_batch_range`, the chunked prefill (one function, kept whole: a single item over the file
//! budget). Map: `mod.rs`.

use super::*;

impl GpuModel {
    /// Layer-range prefill (PP-prefill, PLAN/14): runs layers [lo, hi) over the window.
    ///
    /// - `lo == 0`: embeds `prompt` (plus vision splice) as usual — the normal entry.
    /// - `lo > 0`:  takes the INCOMING residual stream (`incoming`, the bf16 [n, h] hidden the
    ///   lower-range box produced at its layer hi' == this lo) and continues the per-layer loop
    ///   from it. Everything the loop touches — KV cache rows, GDN s_state, conv state — is
    ///   LAYER-LOCAL, so the only value that crosses a range boundary is `residual` itself, in
    ///   bf16, value-exactly. The rope tables are still built (upper-range FA layers need them).
    /// - `hi < layers.len()`: skips the final norm + LM head tail and returns `(0, residual)` —
    ///   the caller ships the residual (2-box PP) or resumes in-process (on-box validation).
    /// - `hi == layers.len()`: the full tail — identical to the monolithic path.
    pub fn prefill_batch_range(&self, pool: &mut Pool, prompt: &[u32],
                         state: &mut BatchGpuState, slot: usize, kv_stride: usize,
                         pos_start: usize, lo: usize, hi: usize, incoming: Option<B>) -> (u32, B) {
        crate::thermal::pace(); // NIGHT1: the thermal governor (once per prefill range)
        // F8 diagnostic (--trace-prefill): every call's shape + caller. The decode-phase
        // prefill-class kernels (conv1d_prefill/gdn_chunk/attn_prefill_fa/dequant) needed a
        // caller attribution — this is the front door of that whole family.
        static PF_N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        if crate::opts::var(crate::opt!("trace-prefill")).is_ok() {
            use std::sync::atomic::Ordering::Relaxed;
            let c = PF_N.fetch_add(1, Relaxed);
            if c < 6 || c % 200 == 0 {
                let bt = std::backtrace::Backtrace::force_capture();
                let frames: Vec<String> = format!("{bt}").lines()
                    .filter(|l| l.contains("gb10_inference"))
                    .map(|l| l.trim().to_string()).take(6).collect();
                eprintln!("[pf-trace #{c}] n={} pos_start={} slot={}\n   {}", prompt.len(), pos_start, slot,
                          frames.join("\n   "));
            }
        }
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        // TP-local attention heads when the mixers are sharded (matches the sharded qkv/o_proj and the
        // head-sharded KV cache these paths write into).
        let nh = self.eff_num_heads(); let nkv = self.eff_num_kv_heads(); let hd = cfg.head_dim; let rdim = cfg.rotary_dim;
        // TP-local GDN geometry (see eff_lin_v_heads). Same rule as the decode path: every dim on a
        // sharded path comes from an eff_* accessor, never from cfg directly.
        let lin_nh = self.eff_lin_v_heads(); let kd = cfg.lin_k_dim; let vd = cfg.lin_v_dim;
        let conv_dim = self.eff_conv_dim(); let ck = cfg.conv_kernel;
        let value_dim = self.eff_value_dim();
        let v = cfg.vocab_size;
        let n = prompt.len();
        let _max_pc = n;

        // --prefill-trace (TTFT fix 0, EXPERT_TTFT_PREFILL_RESPONSE §3.4): per-phase wall
        // attribution for the hy3/qwen prefill path (until this landing the knob fired only in
        // the DSV4 code). The boundary syncs distort — this pass is for attribution only; the
        // knob is off by default.
        let trace_pf = crate::opts::var(crate::opt!("prefill-trace")).ok().is_some();
        let pf_t0 = std::time::Instant::now();
        let mut pf_phase = pf_t0;
        let mut pf_prologue = 0.0f64;
        let mut pf_mixer = 0.0f64;
        let mut pf_ffn = 0.0f64;
        let mut pf_tail = 0.0f64;
        let mut pf_layer_mixer = vec![0.0f64; self.layers.len()];
        let mut pf_layer_ffn = vec![0.0f64; self.layers.len()];

        let pos_i32: Vec<i32> = (0..n).map(|i| (pos_start + i) as i32).collect();
        let pos_dev = self.dev.htod_sync_copy(&pos_i32).expect("htod pos");
        let cos = pool.get(n * rdim);
        let sin = pool.get(n * rdim);
        self.dev.synchronize().unwrap(); // NULL stream htod must complete before non-blocking stream reads
        blaunch!(self, "gather_rope_b", grid(n*rdim), (256,1,1), 0,
            (d(&cos), d(&sin), d(&self.cos_table), d(&self.sin_table),
             *pos_dev.device_ptr() as u64, rdim as i32, n as i32));

        // qwen4_exp: the PLE hash reads the window's token ids on device (kept alive over the loop).
        let q4 = cfg.is_q4();
        let rw = cfg.resid_width();
        let hcn = cfg.hc_count.max(1);
        let q4_toks: Option<cudarc::driver::CudaSlice<i32>> = if q4 {
            Some(self.dev.htod_sync_copy(&prompt.iter().map(|&t| t as i32).collect::<Vec<_>>()).expect("htod q4 tokens"))
        } else { None };
        let inj: Option<S> = if q4 { Some(pool.get(hcn * n)) } else { None };
        let residual = if lo == 0 {
            if xchain_active() { xchain_new(n); }
            // V3 vision-embed splice: overwrite the image_pad span rows of the embedded hidden with the
            // merged image embeddings (width == text hidden). Must run BEFORE the layer loop — and, on
            // qwen4_exp, before the hyper-connection expansion (inside embed_batch_resid_vision).
            let vision = state.vision_embeds.as_ref().map(|ve| (ve, state.vision_spans.as_slice()));
            self.embed_batch_resid_vision(prompt, vision, pos_start)
        } else {
            // PP-prefill: continue from the lower range's residual stream (bf16, value-exact).
            incoming.expect("prefill_batch_range: lo > 0 requires the incoming hidden")
        };
        let normed = pool.get_bf16_fp("prefill_batch_range:normed", h * self.pf8_pad(n), h * self.pf8_pad(n));
        let ckpt_slot: Option<usize> = None; // prefill never needs the MTP ping-pong checkpoint
        if trace_pf {
            self.sync_stream();
            let now = std::time::Instant::now();
            pf_prologue += now.duration_since(pf_phase).as_secs_f64();
            pf_phase = now;
        }
        for (li, layer) in self.layers.iter().enumerate().skip(lo).take(hi - lo) {
            if q4 {
                if let Some(ple) = &layer.ple {
                    self.ple_forward(pool, d(&residual), ple, state, *q4_toks.as_ref().unwrap().device_ptr() as u64,
                                     0, slot, 0, true, n, None, false);
                }
                self.hc_pre(pool, &normed, inj.as_ref(), d(&residual), &layer.hc.as_ref().unwrap().0, n);
            } else {
            blaunch!(self, "rmsnorm_b", (n as u32,1,1), (1024,1,1), (4096) as u32,
                (d(&normed), d(&residual), d(&layer.input_ln), h as i32, n as i32, fbits(cfg.rms_eps)));
            }
            self.gptq_profile_layer(li, &normed, h, n);
            let mixer = match layer.layer_type {
                LayerType::LinearAttention => {
                    let la = layer.la.as_ref().unwrap();
                    let conv_off = (slot * conv_dim * ck * 4) as u64;
                    let s_off = (slot * lin_nh * kd * vd * 4) as u64;
                    let conv_ptr = *state.conv_state[li].as_ref().unwrap().device_ptr() + conv_off;
                    let s_ptr = *state.s_state[li].as_ref().unwrap().device_ptr() + s_off;
                    // MTP depth-2 ping-pong: if a checkpoint slot is given, the GDN kernels snapshot
                    // the recurrent state after the always-accepted committed token (S1) into that
                    // slot, so a rejected draft restores S1 with no second model forward.
                    let mid_conv_ptr: u64 = match ckpt_slot {
                        Some(ss) => *state.conv_state[li].as_ref().unwrap().device_ptr()
                            + (ss as u64 * conv_dim as u64 * ck as u64 * 4),
                        None => 0,
                    };
                    let mid_s_ptr: u64 = match ckpt_slot {
                        Some(ss) => *state.s_state[li].as_ref().unwrap().device_ptr()
                            + (ss as u64 * lin_nh as u64 * kd as u64 * vd as u64 * 4),
                        None => 0,
                    };
                    let np = self.pf8_pad(n);
                    let mut qkv = pool.get_bf16(conv_dim * np);
                    let mut z = pool.get_bf16(value_dim * np);
                    let mut b = pool.get_bf16(lin_nh * np);
                    let mut a = pool.get_bf16(lin_nh * np);
                    // b/a row geometry for gdn_prep_b: the FUSED arm splits to LOCAL-width rows;
                    // the SPLIT arm GEMMs the replicated-full b/a at the FULL head count and
                    // hands the kernel the rank's band via pointer offset + full row stride.
                    let mut ba_stride_pf = lin_nh;
                    let mut ba_off_pf: u64 = 0;
                    match &la.in_proj {
                        GdnIn::Fused(w) => {
                            let mtot = self.eff_gdn_fused_m();
                            let mut fused = pool.get_bf16_fp("prefill_batch_range:fused_qkv", mtot * self.pf8_pad(n), mtot * self.pf8_pad(n));
                            self.gemm_act(w, &normed, &mut fused, h, mtot, n);
                            blaunch!(self, "split_gdn_b", grid(mtot*n), (256,1,1), 0,
                                (d(&qkv), d(&z), d(&b), d(&a), d(&fused),
                                 conv_dim as i32, value_dim as i32, lin_nh as i32, n as i32,
                                 self.cfg.lin_num_v_heads as i32, self.gdn_head0() as i32));
                            pool.release_bf16(fused, mtot * n);
                        }
                        GdnIn::Split { qkv: pq, z: pz, b: pb, a: pa } => {
                            // Under TP the quantized qkv/z parts arrive SHARDED (rank-local head
                            // bands — the split arm computes local dims); b/a stay replicated-full
                            // (bf16 rows have no shard pipeline), so GEMM them at the FULL head
                            // count and let gdn_prep_b read this rank's band. world==1: identical
                            // to the old local call.
                            let nh_src = self.cfg.lin_num_v_heads;
                            self.gemm_act(pq, &normed, &mut qkv, h, conv_dim, n);
                            self.gemm_act(pz, &normed, &mut z, h, value_dim, n);
                            let mut bb = pool.get_bf16_fp("prefill_batch_range:mixer_bb", nh_src * self.pf8_pad(n), nh_src * self.pf8_pad(n));
                            let mut ab = pool.get_bf16_fp("prefill_batch_range:mixer_ab", nh_src * self.pf8_pad(n), nh_src * self.pf8_pad(n));
                            self.gemm_act(pb, &normed, &mut bb, h, nh_src, n);
                            self.gemm_act(pa, &normed, &mut ab, h, nh_src, n);
                            b = bb; a = ab;
                            ba_stride_pf = nh_src;
                            ba_off_pf = (self.tp_rank as usize * lin_nh * 2) as u64;
                        }
                    }
                    // conv1d is a causal depthwise STENCIL, not a recurrence: out[t] reads in[t-3..t].
                    // It is now fully parallel over (channel, position), which forced it out of place --
                    // a thread reads inputs its neighbours are writing. So: convolve into a fresh buffer,
                    // swap, and carry the final window back into `state` in a separate launch (the main
                    // kernel reads `state` from every thread; writing it there would race).
                    let mut convo = pool.get_bf16(conv_dim * n);
                    blaunch!(self, "conv1d_prefill", grid(conv_dim * n), (256,1,1), 0,
                        (d(&convo), d(&qkv), conv_ptr, d(&la.conv1d),
                         conv_dim as i32, ck as i32, n as i32, mid_conv_ptr, 0u64, 0u64, conv_dim as i32));
                    blaunch!(self, "conv1d_prefill_state", grid(conv_dim), (256,1,1), 0,
                        (conv_ptr, d(&qkv), conv_dim as i32, ck as i32, (n - 1) as i32, n as i32, 0u64, conv_dim as i32));
                    std::mem::swap(&mut qkv, &mut convo);
                    pool.release_bf16(convo, conv_dim * n);
                    let core = pool.get_bf16(lin_nh * vd * n);
                    let (nchunk, smem) = gdn_launch(kd, vd);
                    // F0: packed qkv/z/b/a strides — (qkv_stride<<15)|ba_stride = (conv_dim<<15)|lin_nh.
                    let stride_pack = ((conv_dim as u32) << 15) | (ba_stride_pf as u32);
                    // S-A3-w: --gdn-xcheck snapshot of S_in (before any chunked arm mutates it).
                    let gx_on = crate::opts::var(crate::opt!("gdn-xcheck")).is_ok() && n >= 256 && mid_s_ptr == 0;
                    let gx_snap = |pool: &mut Pool| -> Option<S> {
                        if !gx_on { return None; }
                        let w = lin_nh * kd * vd;
                        let buf = pool.get(w);
                        unsafe { cudarc::driver::result::memcpy_dtod_async(
                            *buf.device_ptr(), s_ptr, w * 4, self.stream.stream).expect("xcheck S snapshot dtod"); }
                        Some(buf)
                    };
                    let gx_nkh = (n as i32 & 0xFFFFFF) | ((self.eff_lin_k_heads() as i32 & 0xFF) << 24);
                    // P4 B3: chunked WY/UT scan for prefill (--gdn-chunk). Prefill-only —
                    // decode/verify keep delta_step_prefill (bit-exact contract). VC=16 columns
                    // per block keeps dynamic smem (~33KB) under the 48KB default: plain launch.
                    // Validated on metal vs the sequential scan: o/S rel-L2 ~7.6e-5.
                    // --gdn-chunk2 (2026-08-26): TENSOR-CORE chunked GDN scan (gdn_chunk_tc_b).
                    // 21x over the sequential scan (9.4 vs 198 ms/layer at 8K); consumes the
                    // same gdn_prep_b P0 scratch; bf16 mma phases (o/S rel-L2 ~2.2e-2 vs the
                    // f32 seq oracle — the bf16-operand envelope, non-compounding over N).
                    // Prefill-only; decode/verify keep delta_step_prefill (bit-exact contract).
                    // F8 P2c': SPLIT chunked GDN — H (serial, lean state recurrence +
                    // per-chunk entry states) then WUO (per-chunk-parallel grams/W/subst/O).
                    // When the raw fns are absent (stale PTX), the condition below is false and
                    // the fused gdn_chunk_tc_b branch takes over unchanged.
                    if Self::gdn_split_prefill_on() && n >= 256 && kd == 128 && vd == 128
                        && gdn_split_fns().is_some() {
                        let (h_fn, wuo_fn) = gdn_split_fns().unwrap();
                        static SPLIT_MARK: std::sync::Once = std::sync::Once::new();
                        SPLIT_MARK.call_once(|| eprintln!("[gsc] SPLIT chunked GDN ENGAGED (--gdn-split, n={n})"));
                        debug_assert!(mid_s_ptr == 0, "chunked GDN cannot snapshot mid-state");
                        {
                            let kds = kd + 4;
                            let Qs = pool.get(n * lin_nh * kds);
                            let Ks = pool.get(n * lin_nh * kds);
                            let Vs = pool.get(n * lin_nh * vd);
                            let Ps = pool.get(n * lin_nh * 2);
                            blaunch!(self, "gdn_prep_b", (n as u32, lin_nh as u32, 1), (kd as u32,1,1), (2*kd*4) as u32,
                                (d(&qkv), d(&b) + ba_off_pf, d(&a) + ba_off_pf, stride_pack as i32, (kd as i32) | ((vd as i32) << 16),
                                 d(&la.a_log), d(&la.dt_bias),
                                 (n as i32 & 0xFFFFFF) | ((self.eff_lin_k_heads() as i32 & 0xFF) << 24),
                                 d(&Qs), d(&Ks), d(&Vs), d(&Ps)));
                            let gdn_time = crate::opts::var(crate::opt!("gdn-time")).is_ok();
                            if gdn_time { self.sync_stream(); }
                            let t_split = std::time::Instant::now();
                            const GSC_C: usize = 32;
                            let nc = n.div_ceil(GSC_C);
                            let vcb = vd / 64;                 // v-chunks per head (2)
                            // states scratch [chunk][head][KD][VD] bf16
                            let states = pool.get_bf16(nc * lin_nh * kd * vd);
                            let n_nkh = (n as i32 & 0xFFFFFF)
                                | ((self.eff_lin_k_heads() as i32 & 0xFF) << 24);
                            let gx_s = gx_snap(pool);
                            // --- H ---
                            let mut a_st = s_ptr;
                            let mut a_ks = d(&Ks); let mut a_vs = d(&Vs); let mut a_ps = d(&Ps);
                            let mut a_scr = *states.device_ptr() as u64;
                            let mut a_kdvd_h = (kd as i32) | ((vd as i32) << 16);
                            let mut a_n = n_nkh;
                            let mut args_h: [*mut std::ffi::c_void; 7] = [
                                &mut a_st as *mut u64 as *mut _, &mut a_ks as *mut u64 as *mut _,
                                &mut a_vs as *mut u64 as *mut _, &mut a_ps as *mut u64 as *mut _,
                                &mut a_scr as *mut u64 as *mut _,
                                &mut a_kdvd_h as *mut i32 as *mut _, &mut a_n as *mut i32 as *mut _,
                            ];
                            let mut cf = cudarc::driver::sys::CUlaunchConfig {
                                gridDimX: lin_nh as u32 * vcb as u32, gridDimY: 1, gridDimZ: 1,
                                blockDimX: 256, blockDimY: 1, blockDimZ: 1,
                                sharedMemBytes: 34816,
                                hStream: self.stream.stream,
                                attrs: std::ptr::null_mut(), numAttrs: 0,
                            };
                            let r = unsafe {
                                cudarc::driver::sys::cuLaunchKernelEx(&mut cf, h_fn,
                                    args_h.as_mut_ptr(), std::ptr::null_mut())
                            };
                            if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                                panic!("gdn_chunk_states_b launch failed ({r:?})");
                            }
                            // --- WUO ---
                            let mut a_core2 = d(&core);
                            let mut a_qs2 = d(&Qs); let mut a_ks2 = d(&Ks);
                            let mut a_vs2 = d(&Vs); let mut a_ps2 = d(&Ps);
                            let mut a_scr2 = *states.device_ptr() as u64;
                            let mut a_n2 = n_nkh;
                            let mut args_w: [*mut std::ffi::c_void; 7] = [
                                &mut a_core2 as *mut u64 as *mut _, &mut a_qs2 as *mut u64 as *mut _,
                                &mut a_ks2 as *mut u64 as *mut _, &mut a_vs2 as *mut u64 as *mut _,
                                &mut a_ps2 as *mut u64 as *mut _,
                                &mut a_scr2 as *mut u64 as *mut _, &mut a_n2 as *mut i32 as *mut _,
                            ];
                            let mut cf2 = cudarc::driver::sys::CUlaunchConfig {
                                gridDimX: nc as u32, gridDimY: lin_nh as u32, gridDimZ: vcb as u32,
                                blockDimX: 256, blockDimY: 1, blockDimZ: 1,
                                sharedMemBytes: 83204,
                                hStream: self.stream.stream,
                                attrs: std::ptr::null_mut(), numAttrs: 0,
                            };
                            let r = unsafe {
                                cudarc::driver::sys::cuLaunchKernelEx(&mut cf2, wuo_fn,
                                    args_w.as_mut_ptr(), std::ptr::null_mut())
                            };
                            if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                                panic!("gdn_chunk_wuo_b launch failed ({r:?})");
                            }
                            pool.release_bf16(states, nc * lin_nh * kd * vd);
                            if let Some(sc) = gx_s {
                                self.gdn_xcheck_seq(pool, li, "split", &core, sc, s_ptr, d(&qkv), d(&b) + ba_off_pf, d(&a) + ba_off_pf,
                                    stride_pack, kd, vd, d(&la.a_log), d(&la.dt_bias), gx_nkh, n, lin_nh, nchunk, smem);
                            }
                            pool.release(Qs, n * lin_nh * kds);
                            pool.release(Ks, n * lin_nh * kds);
                            pool.release(Vs, n * lin_nh * vd);
                            pool.release(Ps, n * lin_nh * 2);
                            if gdn_time {
                                self.sync_stream();
                                eprintln!("[gdn-time] layer {li} SPLIT kernels: {:.2} ms",
                                          t_split.elapsed().as_secs_f32()*1e3);
                            }
                        }
                    } else
                    if Self::gdn_chunk2_prefill_on() && n >= 256 && kd == 128 && vd == 128 {
                        static CHUNK2_MARK: std::sync::Once = std::sync::Once::new();
                        CHUNK2_MARK.call_once(|| eprintln!("[gtc] tensor-core chunked GDN ENGAGED (--gdn-chunk2, n={n})"));
                        debug_assert!(mid_s_ptr == 0, "chunked GDN cannot snapshot mid-state");
                        let kds = kd + 4;
                        let Qs = pool.get(n * lin_nh * kds);
                        let Ks = pool.get(n * lin_nh * kds);
                        let Vs = pool.get(n * lin_nh * vd);
                        let Ps = pool.get(n * lin_nh * 2);
                        blaunch!(self, "gdn_prep_b", (n as u32, lin_nh as u32, 1), (kd as u32,1,1), (2*kd*4) as u32,
                            (d(&qkv), d(&b) + ba_off_pf, d(&a) + ba_off_pf, stride_pack as i32, (kd as i32) | ((vd as i32) << 16),
                             d(&la.a_log), d(&la.dt_bias),
                             (n as i32 & 0xFFFFFF) | ((self.eff_lin_k_heads() as i32 & 0xFF) << 24),
                             d(&Qs), d(&Ks), d(&Vs), d(&Ps)));
                        let gdn_time = crate::opts::var(crate::opt!("gdn-time")).is_ok();
                        if gdn_time { self.sync_stream(); }
                        let t_gtc = std::time::Instant::now();
                        let gx_s = gx_snap(pool);
                        let f = self.gdn_chunk_tc_raw_fn.expect("gdn_chunk_tc raw fn (load-time failure?)");
                        let mut a_core = d(&core); let mut a_st = s_ptr;
                        let mut a_qs = d(&Qs); let mut a_ks = d(&Ks);
                        let mut a_vs = d(&Vs); let mut a_ps = d(&Ps);
                        let mut a_kdvd = (kd as i32) | ((vd as i32) << 16);
                        let mut a_n = (n as i32 & 0xFFFFFF) | ((self.eff_lin_k_heads() as i32 & 0xFF) << 24);
                        let mut args: [*mut std::ffi::c_void; 8] = [
                            &mut a_core as *mut u64 as *mut _, &mut a_st as *mut u64 as *mut _,
                            &mut a_qs as *mut u64 as *mut _, &mut a_ks as *mut u64 as *mut _,
                            &mut a_vs as *mut u64 as *mut _, &mut a_ps as *mut u64 as *mut _,
                            &mut a_kdvd as *mut i32 as *mut _, &mut a_n as *mut i32 as *mut _,
                        ];
                        let grid = (lin_nh * (vd / 64)) as u32;      // 48*2 = 96 blocks
                        let mut cf = cudarc::driver::sys::CUlaunchConfig {
                            gridDimX: grid, gridDimY: 1, gridDimZ: 1,
                            blockDimX: 256, blockDimY: 1, blockDimZ: 1,
                            sharedMemBytes: 83204,
                            hStream: self.stream.stream,
                            attrs: std::ptr::null_mut(), numAttrs: 0,
                        };
                        let r = unsafe {
                            cudarc::driver::sys::cuLaunchKernelEx(&mut cf, f, args.as_mut_ptr(),
                                std::ptr::null_mut())
                        };
                        if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                            panic!("gdn_chunk_tc_b launch failed ({r:?})");
                        }
                        if let Some(sc) = gx_s {
                            self.gdn_xcheck_seq(pool, li, "tc", &core, sc, s_ptr, d(&qkv), d(&b) + ba_off_pf, d(&a) + ba_off_pf,
                                stride_pack, kd, vd, d(&la.a_log), d(&la.dt_bias), gx_nkh, n, lin_nh, nchunk, smem);
                        }
                        pool.release(Qs, n * lin_nh * kds);
                        pool.release(Ks, n * lin_nh * kds);
                        pool.release(Vs, n * lin_nh * vd);
                        pool.release(Ps, n * lin_nh * 2);
                        if gdn_time {
                            self.sync_stream();
                            eprintln!("[gdn-time] layer {li} TC kernel: {:.2} ms", t_gtc.elapsed().as_secs_f32()*1e3);
                        }
                    } else
                    if Self::gdn_chunk_prefill_on() && n >= 256 {
                        // Positive engagement marker (once per process): benches MUST see this
                        // line in the flag-on arm — an A/B without it measured two identical
                        // configurations (the =0-means-ON parse bug did exactly that).
                        static CHUNK_MARK: std::sync::Once = std::sync::Once::new();
                        CHUNK_MARK.call_once(|| eprintln!("[gdn] chunked prefill path ENGAGED (--gdn-chunk, n={n})"));
                        // P0 (F step 3): prep pass normalizes q/k + beta/log-g ONCE per layer
                        // at full occupancy, writing dense scratch; the scan stages by pure
                        // coalesced copies. Scratch ~ N*nh*(2*(kd+4) + vd + 2) f32.
                        let kds = kd + 4;
                        let Qs = pool.get(n * lin_nh * kds);
                        let Ks = pool.get(n * lin_nh * kds);
                        let Vs = pool.get(n * lin_nh * vd);
                        let Ps = pool.get(n * lin_nh * 2);
                        let gdn_time = crate::opts::var(crate::opt!("gdn-time")).is_ok();
                        if gdn_time { self.sync_stream(); }   // F: drain the queue BEFORE the stopwatch
                        let t_p = std::time::Instant::now();
                        blaunch!(self, "gdn_prep_b", (n as u32, lin_nh as u32, 1), (kd as u32,1,1), (2*kd*4) as u32,
                            (d(&qkv), d(&b) + ba_off_pf, d(&a) + ba_off_pf, stride_pack as i32, (kd as i32) | ((vd as i32) << 16),
                             d(&la.a_log), d(&la.dt_bias),
                             (n as i32 & 0xFFFFFF) | ((self.eff_lin_k_heads() as i32 & 0xFF) << 24),
                             d(&Qs), d(&Ks), d(&Vs), d(&Ps)));
                        if gdn_time {
                            self.sync_stream();
                            eprintln!("[gdn-time] layer {} prep kernel: {:.2} ms", li, t_p.elapsed().as_secs_f32()*1e3);
                        }
                        // VC=64 needs ~86KB dynamic smem — over the 48KB default: raw launch on
                        // a CUfunction whose attribute was opted in once at load (E9 pattern).
                        let vcb = 64usize;
                        debug_assert!(mid_s_ptr == 0, "chunked GDN cannot snapshot mid-state");
                        let smem_c = (kd * vcb + 32*(kd+4) + 32*(kd+4) + 32*vcb + 32*vcb
                                      + 32*32 + 32*32 + 32 + 33) as u32 * 4;
                        let t_k = std::time::Instant::now();
                        let f = self.gdn_chunk_raw_fn.expect("gdn_chunk raw fn (load-time failure?)");
                        // --gdn-xcheck: cross-chain check on real activations (AGENTS §3) —
                        // snapshot S_in BEFORE the chunked scan mutates it, so the seq reference
                        // below runs from the identical starting state.
                        let xcheck = crate::opts::var(crate::opt!("gdn-xcheck")).is_ok();
                        let s_words = lin_nh * kd * vd;
                        let s_copy = if xcheck {
                            let buf = pool.get(s_words);
                            unsafe { cudarc::driver::result::memcpy_dtod_async(
                                *buf.device_ptr(), s_ptr, s_words * 4, self.stream.stream)
                                .expect("xcheck S snapshot dtod"); }
                            Some(buf)
                        } else { None };
                        let mut a_core = d(&core); let mut a_qkv = d(&qkv); let mut a_sp = s_ptr;
                        let mut a_b = d(&b); let mut a_a = d(&a);
                        let mut a_stride = stride_pack as i32;
                        let mut a_kdvd = (kd as i32) | ((vd as i32) << 16);
                        let mut a_alog = d(&la.a_log); let mut a_dtb = d(&la.dt_bias);
                        let mut a_n = (n as i32 & 0xFFFFFF) | ((self.eff_lin_k_heads() as i32 & 0xFF) << 24);
                        let mut a_qs = d(&Qs); let mut a_ks = d(&Ks);
                        let mut a_vs = d(&Vs); let mut a_ps = d(&Ps);
                        // ONE array, exactly as many entries as the kernel has params (14), every
                        // entry pointing at a named `let` binding that outlives the launch (the E9
                        // rule at ~1283: never point kernelParams at temporaries). The P0 segfault
                        // was launching with a 12-entry array while the kernel takes 14 params —
                        // cuLaunchKernelEx read kernelParams[12..14] past the array's end and
                        // dereferenced stack garbage as param pointers (host SIGSEGV).
                        let mut args: [*mut std::ffi::c_void; 14] = [
                            &mut a_core as *mut u64 as *mut _, &mut a_qkv as *mut u64 as *mut _,
                            &mut a_sp as *mut u64 as *mut _, &mut a_b as *mut u64 as *mut _,
                            &mut a_a as *mut u64 as *mut _, &mut a_stride as *mut i32 as *mut _,
                            &mut a_kdvd as *mut i32 as *mut _, &mut a_alog as *mut u64 as *mut _,
                            &mut a_dtb as *mut u64 as *mut _, &mut a_n as *mut i32 as *mut _,
                            &mut a_qs as *mut u64 as *mut _, &mut a_ks as *mut u64 as *mut _,
                            &mut a_vs as *mut u64 as *mut _, &mut a_ps as *mut u64 as *mut _,
                        ];
                        unsafe {
                            self.launch_plain_raw(f, ((lin_nh * (vd / vcb)) as u32, 1, 1),
                                                  (256, 1, 1), smem_c, args.as_mut_slice());
                        }
                        pool.release(Qs, n * lin_nh * kds);
                        pool.release(Ks, n * lin_nh * kds);
                        pool.release(Vs, n * lin_nh * vd);
                        pool.release(Ps, n * lin_nh * 2);
                        if gdn_time {
                            self.sync_stream();
                            eprintln!("[gdn-time] layer {} chunked kernel: {:.2} ms", li, t_k.elapsed().as_secs_f32()*1e3);
                        }
                        if let Some(s_copy) = s_copy {
                            // Reference chain: seq kernel from the SAME S_in (the snapshot) and the
                            // same qkv/b/a, into a fresh core. Compares per-layer LOCAL divergence
                            // (chunked vs seq), not compounding: o rel-L2 and S rel-L2. Drift class
                            // is ~1e-4 (metal validation: 7.6e-5); 1e-2+ means a staging bug.
                            let core2 = pool.get_bf16(lin_nh * vd * n);
                            blaunch!(self, "delta_step_prefill", ((lin_nh*nchunk) as u32,1,1), (kd as u32,1,1), smem,
                                (d(&core2), d(&qkv), *s_copy.device_ptr(), d(&b), d(&a), stride_pack as i32,
                                 (kd as i32) | ((vd as i32) << 16), d(&la.a_log), d(&la.dt_bias),
                                 (n as i32 & 0xFFFFFF) | ((self.eff_lin_k_heads() as i32 & 0xFF) << 24),
                                 0u64, 0u64));
                            let s_real = pool.get(s_words);
                            unsafe { cudarc::driver::result::memcpy_dtod_async(
                                *s_real.device_ptr(), s_ptr, s_words * 4, self.stream.stream)
                                .expect("xcheck S readback dtod"); }
                            self.sync_stream();
                            let o1 = self.dev.dtoh_sync_copy(&core).expect("xcheck core dtoh");
                            let o2 = self.dev.dtoh_sync_copy(&core2).expect("xcheck core2 dtoh");
                            let m = lin_nh * vd * n;
                            let (mut dd, mut nn) = (0f64, 0f64);
                            for i in 0..m {
                                let a = o1[i].to_f32() as f64; let b = o2[i].to_f32() as f64;
                                dd += (a - b) * (a - b); nn += b * b;
                            }
                            let s1 = self.dev.dtoh_sync_copy(&s_real).expect("xcheck s_real dtoh");
                            let s2 = self.dev.dtoh_sync_copy(&s_copy).expect("xcheck s_copy dtoh");
                            let (mut sd, mut sn) = (0f64, 0f64);
                            for i in 0..s_words {
                                let a = s1[i] as f64; let b = s2[i] as f64;
                                sd += (a - b) * (a - b); sn += b * b;
                            }
                            eprintln!("[gdn-xcheck] layer {:>2}  o rel-L2 {:.3e}   S rel-L2 {:.3e}",
                                      li, (dd / nn.max(1e-30)).sqrt(), (sd / sn.max(1e-30)).sqrt());
                            pool.release_bf16(core2, lin_nh * vd * n);
                            pool.release(s_real, s_words);
                            pool.release(s_copy, s_words);
                        }
                    } else {
                    let gdn_time = crate::opts::var(crate::opt!("gdn-time")).is_ok();
                    if gdn_time { self.sync_stream(); }   // F: drain the queue BEFORE the stopwatch
                    let t_k = std::time::Instant::now();
                    blaunch!(self, "delta_step_prefill", ((lin_nh*nchunk) as u32,1,1), (kd as u32,1,1), smem,
                        (d(&core), d(&qkv), s_ptr, d(&b), d(&a), stride_pack as i32, (kd as i32) | ((vd as i32) << 16),
                         d(&la.a_log), d(&la.dt_bias), (n as i32 & 0xFFFFFF) | ((self.eff_lin_k_heads() as i32 & 0xFF) << 24),
                         mid_s_ptr, 0u64));
                    if gdn_time {
                        self.sync_stream();
                        eprintln!("[gdn-time] layer {} seq kernel: {:.2} ms", li, t_k.elapsed().as_secs_f32()*1e3);
                    }
                    }
                    let gnormed = pool.get_bf16(value_dim * n);
                    let z_off_z_stride = ((lin_nh as u32 * vd as u32) << 15) | 0;   // packed z: (z_stride<<15)|z_off
                    blaunch!(self, self.gdn_gate_kernel(), ((n*lin_nh) as u32,1,1), (vd as u32,1,1), (vd*4) as u32,
                        (d(&gnormed), d(&core), d(&z), d(&la.norm), vd as i32, lin_nh as i32, n as i32, fbits(cfg.rms_eps), z_off_z_stride as i32));
                    let mut out = pool.get_bf16_fp("prefill_batch_range:o_proj", h * self.pf8_pad(n), h * self.pf8_pad(n));
                    if self.gdn_shard_factor() > 1 {
                        // AR landing 2: prefill is never fused (chunked, and its FFN epilogue is the
                        // unfused add_residual_b chain — §4) — the fused outcome is impossible.
                        let _ = self.tp_reduce_site(pool, &la.out_proj, &gnormed, &mut out, value_dim, h, n, false);
                    } else {
                        self.gemm_act(&la.out_proj, &gnormed, &mut out, value_dim, h, n);
                    }
                    pool.release_bf16(qkv, conv_dim*n); pool.release_bf16(z, value_dim*n);
                    pool.release_bf16(b, lin_nh*n); pool.release_bf16(a, lin_nh*n);
                    pool.release_bf16(core, lin_nh*vd*n); pool.release_bf16(gnormed, value_dim*n);
                    out
                }
                LayerType::FullAttention => {
                    let fa = layer.fa.as_ref().unwrap();
                    // K and V slot bases diverge under k8v4 (K rows 20 B/16, V rows 12 B/16) —
                    // the prefill write/dequant kernels address each cache with its own row size.
                    let k_off = (slot * nkv * kv_stride * self.kv_k_elems_per_pos() * 2) as u64;
                    let v_off = (slot * nkv * kv_stride * self.kv_v_elems_per_pos() * 2) as u64;
                    let kc_ptr = *state.k_cache[li].as_ref().unwrap().device_ptr() + k_off;
                    let vc_ptr = *state.v_cache[li].as_ref().unwrap().device_ptr() + v_off;
                    // qwen3_5 fuses a sigmoid output-gate into q_proj ([q|gate]); hy_v3's q_proj is bare.
                    let qmul = if cfg.attn_out_gate() { 2 } else { 1 };
                    let np = self.pf8_pad(n);
                    let mut qg = pool.get_bf16(nh*hd*qmul*np);
                    let mut k = pool.get_bf16(nkv*hd*np);
                    let mut v = pool.get_bf16(nkv*hd*np);
                    match &fa.qkv {
                        AttnIn::Fused(w) => {
                            let mtot = if self.attn_shard_factor() > 1 { nh*hd*qmul + 2*nkv*hd } else { AttnIn::fused_m(cfg) };
                            let mut fused = pool.get_bf16_fp("prefill_batch_range:fused_qkv_2", mtot * self.pf8_pad(n), mtot * self.pf8_pad(n));
                            self.gemm_act(w, &normed, &mut fused, h, mtot, n);
                            blaunch!(self, "split_qkv_b", grid(mtot*n), (256,1,1), 0,
                                (d(&qg), d(&k), d(&v), d(&fused),
                                 (nh*hd*qmul) as i32, (nkv*hd) as i32, n as i32));
                            pool.release_bf16(fused, mtot * n);
                        }
                        AttnIn::Split { q: qp, k: kp, v: vp } => {
                            self.gemm_act(qp, &normed, &mut qg, h, nh*hd*qmul, n);
                            self.gemm_act(kp, &normed, &mut k, h, nkv*hd, n);
                            self.gemm_act(vp, &normed, &mut v, h, nkv*hd, n);
                        }
                    }
                    let (q, gate) = if qmul == 2 {
                        let q = pool.get_bf16(nh*hd*n);
                        let g = pool.get_bf16(nh*hd*n);
                        // packed qg row pitch: nh*hd*2
                        blaunch!(self, "split_qgate_b", grid(nh*hd*n), (256,1,1), 0, (d(&q), d(&g), d(&qg), nh as i32, hd as i32, n as i32, (nh*hd*2) as i32));
                        pool.release_bf16(qg, nh*hd*2*n);
                        (q, Some(g))
                    } else { (qg, None) };
                    // E14: fused per-head norm+rope (byte-identical; see full_attn_batch).
                    if crate::opts::var(crate::opt!("no-fused-perhead-rope")).is_ok() {
                        blaunch!(self, "rmsnorm_perhead_b", ((n*nh) as u32,1,1), (hd as u32,1,1), (hd*4) as u32, (d(&q), d(&q), d(&fa.q_norm), nh as i32, hd as i32, n as i32, fbits(cfg.rms_eps)));
                        blaunch!(self, "rmsnorm_perhead_b", ((n*nkv) as u32,1,1), (hd as u32,1,1), (hd*4) as u32, (d(&k), d(&k), d(&fa.k_norm), nkv as i32, hd as i32, n as i32, fbits(cfg.rms_eps)));
                        blaunch!(self, "rope_b", grid(n*nh*(rdim/2)), (256,1,1), 0, (d(&q), d(&cos), d(&sin), nh as i32, hd as i32, rdim as i32, n as i32));
                        blaunch!(self, "rope_b", grid(n*nkv*(rdim/2)), (256,1,1), 0, (d(&k), d(&cos), d(&sin), nkv as i32, hd as i32, rdim as i32, n as i32));
                    } else {
                        // F0 packed q/k row pitches: (off, stride) = (0, nh*hd) / (0, nkv*hd).
                        blaunch!(self, "rmsnorm_rope_b", ((n*nh) as u32,1,1), (hd as u32,1,1), (hd*4) as u32, (d(&q), d(&fa.q_norm), d(&cos), d(&sin), nh as i32, hd as i32, rdim as i32, n as i32, fbits(cfg.rms_eps), 0i32, (nh*hd) as i32));
                        blaunch!(self, "rmsnorm_rope_b", ((n*nkv) as u32,1,1), (hd as u32,1,1), (hd*4) as u32, (d(&k), d(&fa.k_norm), d(&cos), d(&sin), nkv as i32, hd as i32, rdim as i32, n as i32, fbits(cfg.rms_eps), 0i32, (nkv*hd) as i32));
                    }
                    let mut attn = pool.get_bf16(nh*hd*n);
                    let scale = 1.0f32/(hd as f32).sqrt();
                    // [NW][hd] partial accumulators + [NW] m + [NW] l, for the final softmax merge.
                    // The INNER loop needs no shared memory at all — it is warp shuffles and registers.
                    let nw = (hd / 32) as u32;                 // warps per block (blockDim = hd)
                    let smem = (nw * hd as u32 + 2 * nw) * 4;
                    if self.kv_quant || self.kv_tq || self.kv_k8v4 || self.kv_k8v8 {
                        // Quantized cache: quantize at write, dequantize for the attention readers
                        // (the TQ dequant materializes decode_k/decode_v, the values the splitk
                        // score path is consistent with). E2 Fix 2: the dequant is INCREMENTAL
                        // into a persistent per-(layer, slot) bf16 mirror — the old code re-
                        // dequantized the whole prefix from position 0 into a fresh scratch PER
                        // LAYER PER WINDOW (O(L²): 15 windows × 79 layers × full prefix at 128K).
                        // The mirror allocates lazily on the slot's SECOND window (single-window
                        // prompts — the common case — pay nothing), advances a watermark by only
                        // the new range, and feeds the same readers via their stride argument
                        // (mirror head-stride = budget). Values are bit-identical by construction:
                        // the dequant is position-stateless, so mirror rows equal the corresponding
                        // rows of a full re-dequant. All the risk is the bookkeeping below.
                        // k8v4: K packs via kv8_pack16 (20 B/16), V via kvq16_pack (12 B/16); the
                        // mirror pair is dequant_kv_k8v4 (K) + dequant_kv_q4 (V) under ONE watermark.
                        if self.kv_tq {
                            blaunch!(self, "write_kv_prefill_tq", ((n*nkv) as u32,1,1), (128,1,1),
                                if self.kv_tq_b3 { TQ_ENCODE_SMEM_BYTES_B3 } else { TQ_ENCODE_SMEM_BYTES },
                                (kc_ptr, vc_ptr, d(&k), d(&v), d(&self.tq_tables), kv_stride as i32, nkv as i32, n as i32, pos_start as i32));
                        } else if self.kv_k8v8 {
                            blaunch!(self, "write_kv_prefill_k8v8", grid(n*nkv*(hd/16)), (256,1,1), 0,
                                (kc_ptr, vc_ptr, d(&k), d(&v), kv_stride as i32, nkv as i32, hd as i32, n as i32, pos_start as i32));
                        } else if self.kv_k8v4 {
                            blaunch!(self, "write_kv_prefill_k8v4", grid(n*nkv*(hd/16)), (256,1,1), 0,
                                (kc_ptr, vc_ptr, d(&k), d(&v), kv_stride as i32, nkv as i32, hd as i32, n as i32, pos_start as i32));
                        } else {
                            blaunch!(self, "write_kv_prefill_q4", grid(n*nkv*(hd/16)), (256,1,1), 0,
                                (kc_ptr, vc_ptr, d(&k), d(&v), kv_stride as i32, nkv as i32, hd as i32, n as i32, pos_start as i32));
                        }
                        let npos = pos_start + n;
                        let budget = self.kv_mirror_budget(kv_stride);
                        // qwen4_exp QSA on a packed cache (k8v4; qsa_enabled refuses q4/tq): raw keys
                        // -> the slot's indexer cache, then per-query block selection. The selected
                        // attention below reads the bf16 dequant (mirror or scratch), never the
                        // packed rows — the same values the tiled path would read.
                        let qsa_sel_p: Option<(S, S)> = self.qsa_in(state, fa, li, kv_stride, 0, 0, 0).and_then(|qs| {
                            let keys_ptr = qs.keys_ptr + (slot * kv_stride * cfg.indexer_head_dim * 2) as u64;
                            self.qsa_select_prefill(pool, qs.idx, &normed, keys_ptr, kv_stride, pos_start, n, &cos, &sin)
                        });
                        if npos <= budget && (pos_start > 0 || state.kv_mirror[li][slot].is_some()) {
                            // Mirror arm. The watermark is the whole rule: rows [0, up_to) are
                            // valid, so the range to dequant is [up_to, npos) — a fresh mirror
                            // catches up [0, npos) once (== the old full dequant), a contiguous
                            // window only its own tokens (up_to == pos_start), a prefix-hit clamp
                            // only the tail past the clamp. Invalidation keeps [0, up_to) honest.
                            let m = state.kv_mirror[li][slot].get_or_insert_with(|| KvMirror {
                                k: self.dev.alloc_zeros::<half::bf16>(nkv * budget * hd).unwrap(),
                                v: self.dev.alloc_zeros::<half::bf16>(nkv * budget * hd).unwrap(),
                                up_to: 0,
                            });
                            let (mk, mv, wm) = (d(&m.k), d(&m.v), m.up_to);
                            if npos > wm {
                                let len = npos - wm;
                                // Rows land at (head*budget + wm + t): out head-stride = budget,
                                // base shifted by wm rows (head-uniform shift — see the kernel).
                                if self.kv_tq {
                                    blaunch!(self, "dequant_kv_tq", ((len*nkv) as u32,1,1), (128,1,1), TQ_DEQ_SMEM_BYTES,
                                        (mk + (wm * hd * 2) as u64, mv + (wm * hd * 2) as u64, kc_ptr, vc_ptr,
                                         d(&self.tq_tables), nkv as i32, kv_stride as i32,
                                         len as i32, wm as i32, budget as i32));
                                } else if self.kv_k8v8 {
                                    // K AND V at int8+fp16 blocks — channel-agnostic reader, twice.
                                    blaunch!(self, "dequant_kv_k8v4", grid(len*nkv*(hd/16)), (256,1,1), 0,
                                        (mk + (wm * hd * 2) as u64, kc_ptr, nkv as i32, kv_stride as i32,
                                         hd as i32, len as i32, wm as i32, budget as i32));
                                    blaunch!(self, "dequant_kv_k8v4", grid(len*nkv*(hd/16)), (256,1,1), 0,
                                        (mv + (wm * hd * 2) as u64, vc_ptr, nkv as i32, kv_stride as i32,
                                         hd as i32, len as i32, wm as i32, budget as i32));
                                } else if self.kv_k8v4 {
                                    // K at int8+fp16, V at q4 — the two dequants share one watermark.
                                    blaunch!(self, "dequant_kv_k8v4", grid(len*nkv*(hd/16)), (256,1,1), 0,
                                        (mk + (wm * hd * 2) as u64, kc_ptr, nkv as i32, kv_stride as i32,
                                         hd as i32, len as i32, wm as i32, budget as i32));
                                    blaunch!(self, "dequant_kv_q4", grid(len*nkv*(hd/16)), (256,1,1), 0,
                                        (mv + (wm * hd * 2) as u64, vc_ptr, nkv as i32, kv_stride as i32,
                                         hd as i32, len as i32, wm as i32, budget as i32));
                                } else {
                                    blaunch!(self, "dequant_kv_q4", grid(len*nkv*(hd/16)), (256,1,1), 0,
                                        (mk + (wm * hd * 2) as u64, kc_ptr, nkv as i32, kv_stride as i32,
                                         hd as i32, len as i32, wm as i32, budget as i32));
                                    blaunch!(self, "dequant_kv_q4", grid(len*nkv*(hd/16)), (256,1,1), 0,
                                        (mv + (wm * hd * 2) as u64, vc_ptr, nkv as i32, kv_stride as i32,
                                         hd as i32, len as i32, wm as i32, budget as i32));
                                }
                                m.up_to = npos;
                            }
                            if let Some((sel, pos_sel)) = &qsa_sel_p {
                                self.qsa_attn_prefill_ptrs(&mut attn, &q, mk, mv, budget, nh, nkv, hd, scale, n, d(sel), d(pos_sel));
                            } else if n >= PF_MIN && !prefill_scalar() {
                                self.attn_prefill_tiled(pool, &q, mk, mv, budget, nh, nkv, n, pos_start, &mut attn);
                            } else {
                                let qt = (hd / 32).max(1);
                                let ntile = n.div_ceil(qt);
                                blaunch!(self, "gqa_attn_prefill", ((ntile*nh) as u32,1,1), (hd as u32,1,1), 0,
                                    (d(&attn), d(&q), mk, mv, budget as i32, nh as i32, nkv as i32, hd as i32,
                                     fbits(scale), n as i32, pos_start as i32));
                            }
                        } else {
                        // Full re-dequant scratch: a slot's FIRST window (mirror not yet paid for)
                        // and every window past the budget — the 128K case keeps the old O(L²)
                        // beyond 32K, trading bounded memory for it; the 26K-class win is entirely
                        // in the mirror arm above.
                        let mut kdq = pool.get_bf16(nkv * npos * hd);
                        let mut vdq = pool.get_bf16(nkv * npos * hd);
                        if self.kv_tq {
                            blaunch!(self, "dequant_kv_tq", ((npos*nkv) as u32,1,1), (128,1,1), TQ_DEQ_SMEM_BYTES,
                                (d(&kdq), d(&vdq), kc_ptr, vc_ptr, d(&self.tq_tables),
                                 nkv as i32, kv_stride as i32, npos as i32, 0i32, npos as i32));
                        } else if self.kv_k8v8 {
                            // K AND V at int8+fp16 blocks — dequant_kv_k8v4 is channel-agnostic.
                            blaunch!(self, "dequant_kv_k8v4", grid(npos*nkv*(hd/16)), (256,1,1), 0,
                                (d(&kdq), kc_ptr, nkv as i32, kv_stride as i32, hd as i32, npos as i32, 0i32, npos as i32));
                            blaunch!(self, "dequant_kv_k8v4", grid(npos*nkv*(hd/16)), (256,1,1), 0,
                                (d(&vdq), vc_ptr, nkv as i32, kv_stride as i32, hd as i32, npos as i32, 0i32, npos as i32));
                        } else if self.kv_k8v4 {
                            // K at int8+fp16, V at q4 — the same pair the mirror arm launches.
                            blaunch!(self, "dequant_kv_k8v4", grid(npos*nkv*(hd/16)), (256,1,1), 0,
                                (d(&kdq), kc_ptr, nkv as i32, kv_stride as i32, hd as i32, npos as i32, 0i32, npos as i32));
                            blaunch!(self, "dequant_kv_q4", grid(npos*nkv*(hd/16)), (256,1,1), 0,
                                (d(&vdq), vc_ptr, nkv as i32, kv_stride as i32, hd as i32, npos as i32, 0i32, npos as i32));
                        } else {
                            blaunch!(self, "dequant_kv_q4", grid(npos*nkv*(hd/16)), (256,1,1), 0,
                                (d(&kdq), kc_ptr, nkv as i32, kv_stride as i32, hd as i32, npos as i32, 0i32, npos as i32));
                            blaunch!(self, "dequant_kv_q4", grid(npos*nkv*(hd/16)), (256,1,1), 0,
                                (d(&vdq), vc_ptr, nkv as i32, kv_stride as i32, hd as i32, npos as i32, 0i32, npos as i32));
                        }
                        let (kqs, vqs) = (d(&kdq), d(&vdq));
                        if let Some((sel, pos_sel)) = &qsa_sel_p {
                            self.qsa_attn_prefill_ptrs(&mut attn, &q, kqs, vqs, npos, nh, nkv, hd, scale, n, d(sel), d(pos_sel));
                        } else if n >= PF_MIN && !prefill_scalar() {
                            self.attn_prefill_tiled(pool, &q, kqs, vqs, npos, nh, nkv, n, pos_start, &mut attn);
                        } else {
                            let qt = (hd / 32).max(1);
                            let ntile = n.div_ceil(qt);
                            blaunch!(self, "gqa_attn_prefill", ((ntile*nh) as u32,1,1), (hd as u32,1,1), 0,
                                (d(&attn), d(&q), kqs, vqs, npos as i32, nh as i32, nkv as i32, hd as i32,
                                 fbits(scale), n as i32, pos_start as i32));
                        }
                        pool.release_bf16(kdq, nkv*npos*hd); pool.release_bf16(vdq, nkv*npos*hd);
                        }
                    } else {
                    blaunch!(self, "write_kv_prefill", grid(n*nkv*hd), (256,1,1), 0, (kc_ptr, vc_ptr, d(&k), d(&v), kv_stride as i32, nkv as i32, hd as i32, n as i32, pos_start as i32));
                    // qwen4_exp QSA: raw keys → the slot's cache, then per-query block selection.
                    let qsa_sel: Option<(S, S)> = self.qsa_in(state, fa, li, kv_stride, 0, 0, 0).and_then(|qs| {
                        let keys_ptr = qs.keys_ptr + (slot * kv_stride * cfg.indexer_head_dim * 2) as u64;
                        self.qsa_select_prefill(pool, qs.idx, &normed, keys_ptr, kv_stride, pos_start, n, &cos, &sin)
                    });
                    if let Some((sel, pos_sel)) = &qsa_sel {
                        let ta = std::time::Instant::now();
                        self.qsa_attn_prefill_ptrs(&mut attn, &q, kc_ptr, vc_ptr, kv_stride, nh, nkv, hd, scale, n, d(sel), d(pos_sel));
                        if crate::opts::var(crate::opt!("qsa-time")).is_ok() { self.sync_stream(); eprintln!("[qsa-time] prefill n={n}: selected attention {:.2} ms", ta.elapsed().as_secs_f32()*1e3); }
                    } else if n >= PF_MIN && !prefill_scalar() {
                        self.attn_prefill_tiled(pool, &q, kc_ptr, vc_ptr, kv_stride, nh, nkv, n, pos_start, &mut attn);
                    } else {
                        let qt = (hd / 32).max(1);              // one query per warp (GQA_PF_QT at hd=256)
                        let ntile = n.div_ceil(qt);
                        blaunch!(self, "gqa_attn_prefill", ((ntile*nh) as u32,1,1), (hd as u32,1,1), 0, (d(&attn), d(&q), kc_ptr, vc_ptr, kv_stride as i32, nh as i32, nkv as i32, hd as i32, fbits(scale), n as i32, pos_start as i32));
                    }
                    if let Some((a, b)) = qsa_sel { self.qsa_release(pool, a, b, n); }
                    }
                    if let Some(g) = &gate {
                        blaunch!(self, "sigmoid_gate_b", grid(nh*hd*n), (256,1,1), 0, (d(&attn), d(g), (nh*hd*n) as i32));
                    }
                    // FIX #3 (2026-09-11): pad to pf8_pad(n) — the pf8 prefill lane writes
                    // tp64 = ceil(n/64)*64 rows (gemm_fp8_prefill_e asserts out covers them);
                    // an unpadded h*n alloc panics whenever the pool's power-of-2 capacity
                    // lands below tp64*h (observed: prefix-hit tail chunk n=408 -> tp64=448,
                    // out 2^21 < 448*5120, ladder d0 8.6k_r2 deterministic panic). Mirrors the
                    // gdn out_proj / batch-mixer pad from 74a4611; the downstream release at
                    // h*n is capacity-keyed, not length-keyed, so it stays as-is.
                    let mut out = pool.get_bf16_fp("prefill_batch_range:o_proj_fullattn", h * self.pf8_pad(n), h * self.pf8_pad(n));
                    if self.attn_shard_factor() > 1 {
                        // AR landing 2: prefill is never fused (chunked, and its FFN epilogue is the
                        // unfused add_residual_b chain — §4) — the fused outcome is impossible.
                        let _ = self.tp_reduce_site(pool, &fa.o_proj, &attn, &mut out, nh*hd, h, n, false);
                    } else {
                        self.gemm_act(&fa.o_proj, &attn, &mut out, nh*hd, h, n);
                    }
                    pool.release_bf16(k, nkv*hd*n); pool.release_bf16(v, nkv*hd*n);
                    pool.release_bf16(q, nh*hd*n); pool.release_bf16(attn, nh*hd*n);
                    if let Some(g) = gate { pool.release_bf16(g, nh*hd*n); }
                    out
                }
            };
            if trace_pf {
                self.sync_stream();
                let now = std::time::Instant::now();
                let dt = now.duration_since(pf_phase).as_secs_f64();
                pf_mixer += dt; pf_layer_mixer[li] += dt;
                pf_phase = now;
            }
            if q4 {
                self.hc_post(d(&residual), &mixer, inj.as_ref().unwrap(), n);
                self.hc_pre(pool, &normed, inj.as_ref(), d(&residual), &layer.hc.as_ref().unwrap().1, n);
            } else {
            blaunch!(self, "fused_res_rmsnorm_b", (n as u32,1,1), (1024,1,1), (4096) as u32,
                (d(&normed), d(&residual), d(&mixer), d(&layer.post_ln), h as i32, n as i32, fbits(cfg.rms_eps)));
            // --layer-csum: post-MIXER residual (pre-FFN) — splits GDN/attn corruption from MLP.
            if crate::opts::var(crate::opt!("layer-csum")).is_ok() {
                let base = *residual.device_ptr() as cudarc::driver::sys::CUdeviceptr;
                let mut acc = String::new();
                for r in [0usize, n / 2, n - 1] {
                    let mut tmp = vec![0u8; h * 2];
                    unsafe {
                        cudarc::driver::result::memcpy_dtoh_sync(&mut tmp, base + (r * h * 2) as u64)
                    }.expect("csum dtoh");
                    let (mut s, mut mx) = (0.0f64, 0.0f32);
                    for ch in tmp.chunks_exact(2) {
                        let f = half::bf16::from_bits(u16::from_le_bytes([ch[0], ch[1]])).to_f32();
                        s += f as f64; if f.abs() > mx { mx = f.abs(); }
                    }
                    acc.push_str(&format!(" MX{s}:{s:.4}/{mx:.4}"));
                }
                eprintln!("[csum] L{li} n{n}{acc}");
            }
            // Phase-8 [fh]: WHOLE-buffer FNV of the residual at both per-layer points — the
            // 3-row csum can miss a divergence that starts between the sampled rows.
            if crate::opts::var(crate::opt!("layer-fullhash")).is_ok() {
                let host = self.dev.dtoh_sync_copy(&residual).unwrap_or_default();
                let n_real = (h * n).min(host.len());
                let mut x: u64 = 1469598103934665603;
                for v in host.iter().take(n_real) { for b in v.to_bits().to_le_bytes() { x ^= b as u64; x = x.wrapping_mul(1099511628211); } }
                let host2 = self.dev.dtoh_sync_copy(&normed).unwrap_or_default();
                let mut x2: u64 = 1469598103934665603;
                for v in host2.iter().take(n_real) { for b in v.to_bits().to_le_bytes() { x2 ^= b as u64; x2 = x2.wrapping_mul(1099511628211); } }
                eprintln!("[fh] L{li} n{n} MX {:012x} nx {:012x}", x & 0xFFFFFFFFFFFF, x2 & 0xFFFFFFFFFFFF);
            }
            }
            let ReduceOut::Full(mlp_out) = self.ffn_batch(pool, &normed, &layer.mlp, n, self.ffn_shard_factor() > 1, false) else {
                unreachable!("AR landing 2: prefill never fuses the FFN epilogue")
            };
            if trace_pf {
                self.sync_stream();
                let now = std::time::Instant::now();
                let dt = now.duration_since(pf_phase).as_secs_f64();
                pf_ffn += dt; pf_layer_ffn[li] += dt;
                pf_phase = now;
            }
            let tot = h * n;
            if q4 {
                self.hc_post(d(&residual), &mlp_out, inj.as_ref().unwrap(), n);
            } else {
            blaunch!(self, "add_residual_b", grid(tot), (256,1,1), 0, (d(&residual), d(&residual), d(&mlp_out), tot as i32));
            }
            // Flake hunt (2026-08-26): fingerprint the post-FFN residual per layer (rows
            // 0 / n/2 / n-1 as f32) so a dual-prefill divergence names its FIRST layer.
            if xchain_active() {
                let rows = [0usize, n / 2, n - 1];
                let mut fp = Vec::with_capacity(rows.len() * h);
                let base = *residual.device_ptr() as cudarc::driver::sys::CUdeviceptr;
                for r in rows {
                    let mut tmp = vec![0u8; h * 2];
                    unsafe {
                        cudarc::driver::result::memcpy_dtoh_sync(&mut tmp, base + (r * h * 2) as u64)
                    }.expect("fp dtoh");
                    for ch in tmp.chunks_exact(2) {
                        fp.push(half::bf16::from_bits(u16::from_le_bytes([ch[0], ch[1]])).to_f32());
                    }
                }
                xchain_capture(|c| c.layer_outputs.push(fp));
            }
            // --layer-csum (diagnostic): 3-row checksum of the post-FFN residual per layer —
            // TP bring-up bisect: same prompt at TP=1 vs TP=N, first divergent layer names the
            // corrupt stage. Rows (not the whole buffer) keep the per-layer cost trivial.
            if crate::opts::var(crate::opt!("layer-csum")).is_ok() {
                let base = *residual.device_ptr() as cudarc::driver::sys::CUdeviceptr;
                let mut acc = String::new();
                for r in [0usize, n / 2, n - 1] {
                    let mut tmp = vec![0u8; h * 2];
                    unsafe {
                        cudarc::driver::result::memcpy_dtoh_sync(&mut tmp, base + (r * h * 2) as u64)
                    }.expect("csum dtoh");
                    let (mut s, mut mx) = (0.0f64, 0.0f32);
                    for ch in tmp.chunks_exact(2) {
                        let f = half::bf16::from_bits(u16::from_le_bytes([ch[0], ch[1]])).to_f32();
                        s += f as f64; if f.abs() > mx { mx = f.abs(); }
                    }
                    acc.push_str(&format!(" r{s}:{s:.4}/{mx:.4}"));
                }
                eprintln!("[csum] L{li} n{n}{acc}");
            }
            if crate::opts::var(crate::opt!("layer-fullhash")).is_ok() {
                let host = self.dev.dtoh_sync_copy(&residual).unwrap_or_default();
                let n_real = (h * n).min(host.len());
                let mut x: u64 = 1469598103934665603;
                for v in host.iter().take(n_real) { for b in v.to_bits().to_le_bytes() { x ^= b as u64; x = x.wrapping_mul(1099511628211); } }
                eprintln!("[fh] L{li} n{n} r {:012x}", x & 0xFFFFFFFFFFFF);
            }
            // P14 DFlash v1 PROMPT capture. The post-FFN-add residual IS hidden_states[l+1] — the
            // same tap source the decode/verify paths capture — written POSITION-ADDRESSED into the
            // wide ctx feature at the window's absolute column (`wide_pos`, set per window by the
            // scheduler). This is the ONLY site allowed to use the wide surface: the prefill is
            // EAGER, so the destination column is a live host value, whereas inside a captured graph
            // it would be baked at capture time (see the verify site below).
            //
            // This site did not exist before the P14 repair. Without it the serving lane's ctx
            // feature was NEVER written by any forward: `dflash_prime` D2D'd an `alloc_zeros`
            // surface (`alloc_zeros` does not zero — it is an async alloc) into the lane's feature,
            // so the drafter conditioned on garbage for every prompt position. The single-process
            // probe never showed it because it primes token-by-token through the ring staging.
            if let Some(tap) = &self.dflash_tap {
                if let Some(w) = &tap.wide {
                    if let Some(k) = tap.tap_index(li) {
                        let p0 = tap.wide_pos.load(std::sync::atomic::Ordering::Relaxed);
                        if p0 + n <= tap.wide_stride {
                            crate::dflash::tap_capture_to(&self.dev, self.stream.stream,
                                *w.device_ptr() as u64, tap.nctx * h, p0,
                                *residual.device_ptr() as u64, h, n, k);
                        } else {
                            crate::dflash::warn_wide_skip(p0, n, tap.wide_stride);
                        }
                    }
                }
            }
            // S5F DFlash2 prompt-prime capture: the post-FFN-add residual IS hidden_states[l+1] —
            // the same tap source the decode/verify path captures. The prefill runs at window
            // width n (up to PREFILL_CHUNK ≫ 8), so the tap lands in the WIDE prime sink, not the
            // 8-column staging. DEFAULT OFF (df2_prime == None): one Option read per layer — free.
            if let Some(ps) = &self.df2_prime {
                if let Some(k) = crate::dflash2::tap_layers().iter().position(|&l| l == li) {
                    crate::dflash2::capture::capture_cols_into(&self.dev, &self.stream,
                        *ps.taps.device_ptr() as u64, crate::dflash2::tap_concat_dim(),
                        *residual.device_ptr() as u64, h, hcn, n, k);
                }
            }
            // WI1 DSpark prompt-prime twin (same wide buffer shape, dspark taps).
            if let Some(ps) = &self.dspark_prime {
                if let Some(k) = crate::dspark::TAP_LAYERS.iter().position(|&l| (l as i32 + Self::dspark_tap_shift()) == li as i32) {
                    crate::dflash2::capture::capture_cols_into(&self.dev, &self.stream,
                        *ps.taps.device_ptr() as u64, crate::dspark::TAP_CONCAT_DIM,
                        *residual.device_ptr() as u64, h, hcn, n, k);
                }
            }
            pool.release_bf16(mixer, h*n); pool.release_bf16(mlp_out, h*n);
        }
        if let Some(i) = inj { pool.release(i, hcn * n); }
        if hi < self.layers.len() {
            // PP-prefill lower range: no final norm / LM head here. The residual stream is the
            // chunk's crossing artifact (bf16 [n, h]) — return it; the caller ships it to the
            // upper-range box or resumes in-process (on-box split validation).
            pool.release_bf16(normed, h*n);
            return (0u32, residual);
        }
        let out = pool.get_bf16(h * n);
        if q4 {
            self.hc_pre(pool, &out, None, d(&residual), self.hc_mixer.as_ref().unwrap(), n);
        } else {
        blaunch!(self, "rmsnorm_b", (n as u32,1,1), (1024,1,1), (4096) as u32,
            (d(&out), d(&residual), d(&self.final_norm), h as i32, n as i32, fbits(cfg.rms_eps)));
        }
        let _ = rw;
        pool.release_bf16(normed, h*n);
        // NOTE: `residual` (the pre-final-RMSNorm backbone hidden) is kept and returned — the MTP
        // head consumes the PRE-norm hidden (it applies its own pre_fc_norm_hidden). Returning the
        // post-norm `out` would double-normalize and materially cut MTP acceptance.

        // Argmax at last position only — so run the LM head (the model's single largest GEMM) on
        // the LAST hidden column alone, not on all n positions. Previously this computed full-vocab
        // logits for every prompt position and discarded columns 0..n-2 unread: at an 8K window that
        // is a 248320×8192 GEMM of waste plus a multi-GB logits buffer and its first-touch memset,
        // paid PER prefill window. Column n-1 of `out` ([h, n] column-major) is contiguous.
        // Flake-hunt fix candidate (2026-08-26): persistent NON-pooled tail buffers. The
        // dual-prefill divergence corrupted `last`/`logits` with foreign real values
        // (timing-dependent flavor: uniform NaN one run, wrong-row data the next) — the
        // signature of two live pool slices aliasing one allocation. --tail-fixed=1
        // takes the whole first-token tail out of the pool.
        if crate::opts::var(crate::opt!("tail-fixed")).is_ok() {
            static TL: std::sync::OnceLock<std::sync::Mutex<Option<B>>> = std::sync::OnceLock::new();
            let cell = TL.get_or_init(|| std::sync::Mutex::new(None));
            let mut gtl = cell.lock().unwrap();
            if gtl.is_none() { *gtl = Some(self.dev.alloc_zeros::<half::bf16>(h).unwrap()); }
            let tl = gtl.as_ref().unwrap();
            self.copy_col_w(*tl.device_ptr() as u64, &out, n - 1, h);
            static TG: std::sync::OnceLock<std::sync::Mutex<Option<B>>> = std::sync::OnceLock::new();
            let cellg = TG.get_or_init(|| std::sync::Mutex::new(None));
            let mut gtg = cellg.lock().unwrap();
            if gtg.is_none() { *gtg = Some(self.dev.alloc_zeros::<half::bf16>(v).unwrap()); }
            let tlog = gtg.as_mut().unwrap();
            let w = self.lm_head.as_ref().unwrap_or(&self.embed);
            self.gemm_act(w, tl, tlog, h, v, 1);
            let token_id_dev = self.dev.alloc_zeros::<i32>(1).unwrap();
            let block = 1024u32;
            blaunch!(self, "argmax_b", (1,1,1), (block,1,1), (block*8),
                (*token_id_dev.device_ptr() as u64, d(tlog), v as i32, 1));
            pool.release_bf16(out, h*n);
            self.sync_stream();
            let tid = self.dev.dtoh_sync_copy(&token_id_dev).unwrap();
            if xchain_active() {
                let pl: Vec<f32> = self.dev.dtoh_sync_copy(tlog).unwrap()[..v].iter().map(|b| half::bf16::to_f32(*b)).collect();
                let mut g = XCHAIN.get_or_init(|| parking_lot::Mutex::new(Vec::new())).lock();
                if let Some(e) = g.last_mut() { e.prefill_logits = pl; }
            }
            return (tid[0] as u32, residual);
        }
        let mut last = pool.get_bf16(h);
        if xchain_active() {
            // capture `out` row n-1 (the copy SOURCE) BEFORE the dtod
            let mut tmp = vec![0u8; h * 2];
            unsafe {
                cudarc::driver::result::memcpy_dtoh_sync(&mut tmp,
                    (*out.device_ptr() as cudarc::driver::sys::CUdeviceptr) + ((n - 1) * h * 2) as u64)
            }.expect("out row dtoh");
            let src_bits: Vec<u32> = tmp.chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]) as u32).collect();
            let mut g = XCHAIN.get_or_init(|| parking_lot::Mutex::new(Vec::new())).lock();
            if let Some(e) = g.last_mut() { e.layer_inputs.push(src_bits.iter().map(|&b| f32::from_bits(b << 16)).collect()); }
        }
        self.copy_col_w(d(&last), &out, n - 1, h);
        let block = 1024u32;
        let token_id_dev = self.dev.alloc_zeros::<i32>(1).unwrap();
        if self.cfg.lm_head_fp32 {
            // hy_v3: the first generated token comes from UNROUNDED fp32 logits (f32 twin kernels).
            let logits = self.logits_batch_f32(pool, &last, 1);
            blaunch!(self, "argmax_f32_b", (1,1,1), (block,1,1), (block*8),
                (*token_id_dev.device_ptr() as u64, d(&logits), v as i32, 1));
            pool.release(logits, v);
        } else {
            let logits = self.logits_batch(pool, &last, 1);
            // R9: capture the PREFILL logits (the last-position argmax input) so the cross-rank diff
            // can prove/disprove "prefill is bit-identical end-to-end" — isolating the decode-side
            // divergence to the recurrent GDN state vs the sampled first token.
            if xchain_active() {
                // Flake hunt: fingerprint the GEMV INPUT (the `last` row) as raw u16 bits —
                // if this differs, the divergence is upstream of the head; if identical while
                // logits differ, the head GEMV / logits buffer is the culprit.
                let last_bits: Vec<u32> = {
                    let raw = self.dev.dtoh_sync_copy(&last).unwrap();
                    raw.iter().map(|b| b.to_bits() as u32).collect()
                };
                let pl = match self.dev.dtoh_sync_copy(&logits) {
                    Ok(mut lh) => { lh.truncate(v); lh.iter().map(|b| half::bf16::to_f32(*b)).collect() }
                    Err(_) => Vec::new(),
                };
                let post_last: Vec<u32> = {
                    let raw = self.dev.dtoh_sync_copy(&last).unwrap();
                    raw.iter().map(|b| b.to_bits() as u32).collect()
                };
                if !pl.is_empty() {
                    let mut g = XCHAIN.get_or_init(|| parking_lot::Mutex::new(Vec::new())).lock();
                    if let Some(e) = g.last_mut() { e.layer_inputs.push(post_last.iter().map(|&b| f32::from_bits(b << 16)).collect()); }
                    // prefill_batch has no per-forward sink entry — attach to the most recent entry
                    // (or create one) under a dedicated field.
                    if g.is_empty() { g.push(XChainCapture::default()); }
                    if let Some(last) = g.last_mut() {
                        last.prefill_logits = pl;
                        last.final_hidden = last_bits.iter().map(|&b| f32::from_bits(b)).collect();
                        last.logits = vec![last_bits.iter().map(|&b| b as f64).sum::<f64>() as f32];
                    }
                }
            }
            blaunch!(self, "argmax_b", (1,1,1), (block,1,1), (block*8),
                (*token_id_dev.device_ptr() as u64, d(&logits), v as i32, 1));
            pool.release_bf16(logits, v);
        }
        pool.release_bf16(last, h);
        pool.release_bf16(out, h*n);
        self.sync_stream();
        let tid = self.dev.dtoh_sync_copy(&token_id_dev).unwrap();
        if trace_pf {
            let now = std::time::Instant::now();
            pf_tail += now.duration_since(pf_phase).as_secs_f64();
            let total = now.duration_since(pf_t0).as_secs_f64() * 1000.0;
            eprintln!("[pf] prefill n={n} pos_start={pos_start} prologue={:.2}ms mixer={:.2}ms ffn={:.2}ms tail={:.2}ms total={:.2}ms",
                      pf_prologue * 1000.0, pf_mixer * 1000.0, pf_ffn * 1000.0, pf_tail * 1000.0, total);
            for (li, (m, f)) in pf_layer_mixer.iter().zip(&pf_layer_ffn).enumerate() {
                if *m > 1e-6 || *f > 1e-6 {
                    eprintln!("[pf]   layer {li}: mixer={:.2}ms ffn={:.2}ms", m * 1000.0, f * 1000.0);
                }
            }
        }
        // Return first token AND the pre-final-norm [h, n] backbone hidden (MTP consumes the
        // PRE-norm hidden; caller extracts h at the last position to seed drafting).
        (tid[0] as u32, residual)
    }
}
