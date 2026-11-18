//! `Df2Round` construction: weights, scratch and RoPE tables for the plain, pinned and tensor-parallel loaders.
//! Siblings: `state.rs` / `inject.rs` / `draft.rs` operate on the loaded round; `mod.rs` has the map.

use super::*;

impl Df2Round {
    /// Load the DFlash2 artifact + upload (per-tensor bf16, norms f32 w−1 — the S3F layout
    /// freeze) + allocate the ring KV + scratch. `head`/`embed` are the TRUNK's borrowed
    /// tensors in either supported dtype (`GpuModel::df2_borrow`). The pre-P2 replicated load
    /// (single-box probes and every non-sharded path).
    pub fn load(dir: &str, head: Option<BorrowedW>, embed: Option<BorrowedW>, max_c: usize)
        -> Result<Self> {
        Self::load_tp(dir, head, embed, max_c, None)
    }

    /// S9F+ (2026-08-29): load with an explicit artifact sha256 pin override.
    /// `None` = the published REAL_SHA256; `Some("off")` = no sha check (inventory/shape/
    /// dtype guard still runs); `Some(hex)` = pin to that hash. Used by the serve path so a
    /// retrained selector artifact (new sha) loads without weakening the default.
    pub fn load_pinned(dir: &str, head: Option<BorrowedW>, embed: Option<BorrowedW>, max_c: usize,
                       sha_pin: Option<&str>) -> Result<Self> {
        Self::load_tp_pinned(dir, head, embed, max_c, None, sha_pin)
    }

    /// P2 — the TP load: `load` plus an all-reduce context (`GpuModel::df2_ar_ctx`, Some only
    /// at world > 2 with `--df2-round-shard on`). When `ar` is Some the BIG per-layer weights
    /// are sliced at load (the proven trunk shard-at-load pattern) and the round carries two
    /// all-reduce sites per layer:
    ///
    /// * q/k/v_proj — HEAD-split rows (q: NUM_HEADS/world heads; k/v: NUM_KV_HEADS/world kv
    ///   heads; GQA's 4:1 ratio is preserved locally). Full-K on the replicated x_conv.
    /// * gate/up_proj — row bands (INTER/world rows), full-K on the replicated normed2.
    /// * o_proj/down_proj — K-split column bands: the rank computes a [5120×B] PARTIAL of the
    ///   true output over its input slice; the all-reduce lands the full sum.
    /// * Everything else (fc, norms, conv kp/base, head/embed borrow, selector) stays
    ///   REPLICATED: the conv coefficients are channel-group-local and the projections after
    ///   them need the full hidden — splitting them would force K-split qkv and extra
    ///   all-reduces for ~0.1 ms/layer of reads; the head must see full-dim inputs (the S9F
    ///   garbage-drafts hazard dies by construction).
    ///
    /// The ring KV shrinks to the rank's kv-heads (`[NUM_KV_HEADS/world, RING_STRIDE, hd]` —
    /// compile-time 2056×128 rows, ctx-free by construction post-S10R). Scratch buffers keep
    /// their FULL sizes (the local slices are prefixes — stale tails are never read by any
    /// consumer, same discipline as `upload_chunk`'s m-padding).
    pub fn load_tp(dir: &str, head: Option<BorrowedW>, embed: Option<BorrowedW>, max_c: usize,
                   ar: Option<Df2ArCtx>) -> Result<Self> {
        Self::load_tp_pinned(dir, head, embed, max_c, ar, None)
    }

    /// `load_tp` with an explicit artifact sha256 pin override (see `load_pinned`).
    pub fn load_tp_pinned(dir: &str, head: Option<BorrowedW>, embed: Option<BorrowedW>, max_c: usize,
                          ar: Option<Df2ArCtx>, sha_pin: Option<&str>) -> Result<Self> {
        // DFLASH2-GEN: install the artifact geometry (from its config.json) BEFORE anything is
        // sized, then pin it — after this point the process cannot silently re-shape the round.
        crate::dflash2::geom::install_geom_from_dir(dir)?;
        crate::dflash2::geom::pin_geom();
        let pin: Option<&str> = match sha_pin {
            Some("off") => None,
            Some(hex) => Some(hex),
            None => Some(crate::dflash2::REAL_SHA256),
        };
        // PLAN/25 §1a: a baked (NVFP4/FP8 weight-only) artifact loads through the quantized sidecar;
        // the BF16 original keeps its exact loader. PLAN/ROUND_SHARD_W2_WORKDOC.md §3 lifts the old
        // "packed-tile sharding not supported yet" refusal: the sharded branch now slices the PACKED
        // sidecar (load.rs `slice_rows`/`slice_k`) and uploads per-rank through the same
        // `upload_nvfp4`/`upload_fp8` path. The flag stays the gate; a *geometry* that cannot be
        // sharded still refuses loudly below (never a silent redundant round).
        let baked = crate::dflash2::load::is_baked(dir);
        let (art_weights, quant): (crate::dflash2::oracle::Dflash2Weights,
                                   Option<Vec<crate::dflash2::load::QuantTensor>>) = if baked {
            let (a, q) = crate::dflash2::load::load_quantized(dir)?;
            // The BF16 artifact's default pin is the source checkpoint's sha — the baked config
            // carries the same value under df2_quant_source_sha256; honor a requested pin vs it.
            if let Some(p) = pin {
                if p != "off" {
                    // Multi-pin rule (12528992): a published, owner-validated artifact must never
                    // be orphaned by re-pinning. A pin that names a KNOWN published hash accepts
                    // any KNOWN source stamp; an exotic explicit pin stays strict.
                    let pin_known = crate::dflash2::KNOWN_SHA256.iter().any(|k| p.eq_ignore_ascii_case(k));
                    let src_known = crate::dflash2::KNOWN_SHA256.iter().any(|k| q.source_sha256.eq_ignore_ascii_case(k));
                    let ok = q.source_sha256.eq_ignore_ascii_case(p) || (pin_known && src_known);
                    anyhow::ensure!(ok,
                        "baked artifact source sha {} != pin {p} (accepted: KNOWN_SHA256 when the pin is a published hash)", q.source_sha256);
                }
            }
            (a.weights, Some(q.tensors))
        } else {
            let a = crate::dflash2::load::load(dir, pin)?;
            (a.weights, None)
        };
        let w = &art_weights;
        let cfg = crate::dflash2::oracle::Dflash2Config::default();
        let max_pos = max_c + block() + 1;

        // Resolve the shard geometry ONCE (a pure function of constants + world ⇒ identical on
        // every rank; a non-divisible world refuses the load loudly — the head's round outcome
        // then ships false via CalibTable and every rank serves MTP in lockstep).
        let (sharded, shard_rank, world) = match &ar {
            Some(a) => {
                let world = a.world as usize;
                anyhow::ensure!(world > 1, "df2 round shard: world {} <= 1", world);
                anyhow::ensure!(num_heads() % world == 0 && num_kv_heads() % world == 0
                                && inter() % world == 0,
                    "df2 round shard: world {world} does not divide heads {}/{} or inter {}", num_heads(), num_kv_heads(), inter());
                anyhow::ensure!((0..world).contains(&(a.rank as usize)),
                    "df2 round shard: rank {} out of range for world {world}", a.rank);
                (true, a.rank as usize, world)
            }
            None => (false, 0, 1),
        };
        let (nq_l, nkv_l, ni_l) = if sharded {
            (num_heads() / world, num_kv_heads() / world, inter() / world)
        } else {
            (num_heads(), num_kv_heads(), inter())
        };

        let dev = CudaDevice::new(0).context("CudaDevice")?;
        let stream = fork_blocking_stream(&dev);

        let bptx = Ptx::from_src(crate::ptx_embed::GPU_BATCH);
        let mut bfnames = ["rmsnorm_b", "rmsnorm_perhead_b", "rope_b", "gather_rope_b",
            "write_kv_b", "add_residual_b", "silu_mul_b", "kernel_build_id",
            "gemm_mma_fp4_b", "gemm_mma_fp8_b", "embed_gather_fp4_tiled_b", "dequant_fp4_tiled_b",
            "embed_gather_b", "gemm_binv_b"].to_vec();
        // P2: the all-reduce handshake kernels live in gpu_batch.cu (the trunk AR path's module);
        // f32tobf16 lives in gpu_kernels.cu — it joins kfnames below (a name from the wrong
        // module is a load-time CUDA_ERROR_NOT_FOUND, caught live at the first SHARD=on boot).
        if ar.is_some() {
            bfnames.extend(["tp_gate_copy_signal", "tp_wait_add", "tp_wait_add_g",
                "tp_wait_add_4way"]);
        }
        // bf16tof32 lives in gpu_kernels.cu (next to the df2 kernels).
        dev.load_ptx(bptx, "gpu_batch", &bfnames)?;
        crate::gpu::GpuModel::assert_kernel_build_id(&dev, "gpu_batch")?;
        let kptx = Ptx::from_src(crate::ptx_embed::GPU_KERNELS);
        // S5F adds gemm_tiled_b (the S3F large-M ctx GEMM) for the prompt-prime path
        // (prime_window runs fc + per-layer k/v at M = the whole prefill window).
        let kfnames = ["gemm_dsp_b_m8_r4", "gemm_tiled_b", "gqa_attn_band_b", "gqa_attn_band_ring_b",
            "top16_b", "df2_sel_walk_b", "df2_sel_walk_sample_b", "bf16tof32", "conv2_dynamic_b",
            "f32tobf16", "kernel_build_id",
            // F8-CATCHUP two-stage draft head (coarse NVFP4 + exact bf16 re-rank)
            "df2_top256_b", "df2_head_rerank_b",
            // FLASHNEXT-DF2-B: the trunk's hc-mean tap capture (hyper-connection trunks)
            "df2_tap_hcmean_b"];
        dev.load_ptx(kptx, "gpu_kernels", &kfnames)?;
        crate::gpu::GpuModel::assert_kernel_build_id(&dev, "gpu_kernels")?;
        let mut bk = HashMap::new();
        for n in bfnames.iter().chain(kfnames.iter()) {
            let module = if kfnames.contains(n) { "gpu_kernels" } else { "gpu_batch" };
            bk.insert(n.to_string(), dev.get_func(module, n).with_context(|| format!("kernel {n} not in ptx"))?);
        }

        let up_b = |data: &[f32]| -> CudaSlice<bf16> {
            let b: Vec<bf16> = data.iter().map(|&x| bf16::from_f32(x)).collect();
            dev.htod_sync_copy(&b).expect("upload bf16")
        };
        let mut layers = Vec::with_capacity(n_layers());
        for (li, l) in w.layers.iter().enumerate() {
            // P2 shard-at-load (host-side slices BEFORE the bf16 upload — the trunk's proven
            // pattern). Row-major [out, in]: head/row bands are contiguous row slices; the
            // K-split (o/down) takes a per-row column band. Full-K everywhere a band is an
            // OUTPUT band; K-sliced bands produce PARTIAL outputs the all-reduce lands.
            //
            // K1 fix (2026-09-12, found by the A0s gate arm): this arm is the **bf16-domain**
            // shard and must not claim a BAKED artifact. It sliced `l.q_proj` and friends, whose
            // f32 linear fields are EMPTY for a baked draft (the quant arm below says so itself:
            // "never touch them"), so `--df2-round-shard on` + a baked draft panicked on the
            // ranks with `range start index 5242880 out of range for slice of length 0`
            // (round.rs:499) ~31 s into the boot, and the packed-domain shard below (line ~558)
            // was UNREACHABLE. Guard on `quant.is_none()` so the baked case falls through to the
            // packed-domain branch, which is what commit 000951e intended.
            if sharded && quant.is_none() {
                let rows = |v: &[f32], cols: usize, r0: usize, r1: usize| v[r0 * cols..r1 * cols].to_vec();
                let cols = |v: &[f32], rows_n: usize, cols_n: usize, c0: usize, c1: usize| -> Vec<f32> {
                    let mut o = Vec::with_capacity(rows_n * (c1 - c0));
                    for r in 0..rows_n { o.extend_from_slice(&v[r * cols_n + c0..r * cols_n + c1]); }
                    o
                };
                let nq_rows = nq_l * head_dim();    // 1024 @ world=4
                let nkv_rows = nkv_l * head_dim();  // 256
                layers.push(GpuLayer {
                    q_proj: Df2W::Bf16(upload_bf16(&dev, &rows(&l.q_proj, hidden(), shard_rank * nq_rows, (shard_rank + 1) * nq_rows))),
                    k_proj: Df2W::Bf16(upload_bf16(&dev, &rows(&l.k_proj, hidden(), shard_rank * nkv_rows, (shard_rank + 1) * nkv_rows))),
                    k_proj_bf16: None,
                    v_proj: Df2W::Bf16(upload_bf16(&dev, &rows(&l.v_proj, hidden(), shard_rank * nkv_rows, (shard_rank + 1) * nkv_rows))),
                    v_proj_bf16: None,
                    o_proj: Df2W::Bf16(upload_bf16(&dev, &cols(&l.o_proj, hidden(), num_heads() * head_dim(),
                                                    shard_rank * nq_rows, (shard_rank + 1) * nq_rows))),
                    gate_proj: Df2W::Bf16(upload_bf16(&dev, &rows(&l.gate_proj, hidden(), shard_rank * ni_l, (shard_rank + 1) * ni_l))),
                    up_proj: Df2W::Bf16(upload_bf16(&dev, &rows(&l.up_proj, hidden(), shard_rank * ni_l, (shard_rank + 1) * ni_l))),
                    down_proj: Df2W::Bf16(upload_bf16(&dev, &cols(&l.down_proj, hidden(), inter(),
                                                       shard_rank * ni_l, (shard_rank + 1) * ni_l))),
                    q_norm: upload_norm(&dev, &l.q_norm),
                    k_norm: upload_norm(&dev, &l.k_norm),
                    input_ln: upload_norm(&dev, &l.input_ln),
                    post_ln: upload_norm(&dev, &l.post_ln),
                    attn_kp: Df2W::Bf16(upload_bf16(&dev, &l.attention_conv.kernel_projection)),
                    attn_base: upload_bf16(&dev, &l.attention_conv.base_kernel),
                    mlp_kp: Df2W::Bf16(upload_bf16(&dev, &l.mlp_conv.kernel_projection)),
                    mlp_base: upload_bf16(&dev, &l.mlp_conv.base_kernel),
                });
            } else if let Some(q) = &quant {
                // PLAN/25 §1a: the 9 block linears come off the packed sidecar (MMA-repacked at
                // upload); k/v keep BF16 twins for prime_window; norms/bases come off the kept
                // bf16 side (the f32 struct's linear fields are EMPTY here — never touch them).
                // PLAN/ROUND_SHARD_W2_WORKDOC.md §3.1: when `sharded`, every packed tensor is
                // sliced in the PACKED domain first (M-band = row slice; K-band = aligned byte
                // slice) and the bf16 k/v twins are row-sliced with them (§3.2: a replicated twin
                // would still compute full k/v and throw the rest away).
                let _ = l;
                use crate::dflash2::load::QuantTensor as QT;
                let pick = |name: &str| -> QT {
                    q.iter().find(|t| t.name() == name)
                        .unwrap_or_else(|| panic!("baked artifact missing packed tensor {name}"))
                        .clone()
                };
                let up4 = |t: QT| -> Df2W {
                    match t {
                        QT::Nvfp4(p) => Df2W::Nvfp4(upload_nvfp4(&dev, &p)),
                        QT::Fp8(p) => Df2W::Fp8(upload_fp8(&dev, &p)),
                    }
                };
                let sl_rows = |t: QT, r0: usize, r1: usize| -> QT {
                    match t { QT::Nvfp4(p) => QT::Nvfp4(p.slice_rows(r0, r1)),
                              QT::Fp8(p) => QT::Fp8(p.slice_rows(r0, r1)) }
                };
                let sl_k = |t: QT, k0: usize, k1: usize| -> QT {
                    match t { QT::Nvfp4(p) => QT::Nvfp4(p.slice_k(k0, k1)),
                              QT::Fp8(p) => QT::Fp8(p.slice_k(k0, k1)) }
                };
                let ln = |sfx: &str| format!("layers.{li}.{sfx}");
                if sharded {
                    let rows16 = |v: &[f32], cols: usize, r0: usize, r1: usize| -> Vec<f32> {
                        assert!(r1 <= v.len() / cols, "bf16 twin slice {r0}..{r1} out of range");
                        v[r0 * cols..r1 * cols].to_vec()
                    };
                    let nq_rows = nq_l * head_dim();
                    let nkv_rows = nkv_l * head_dim();
                    let (rq0, rq1) = (shard_rank * nq_rows, (shard_rank + 1) * nq_rows);
                    let (rk0, rk1) = (shard_rank * nkv_rows, (shard_rank + 1) * nkv_rows);
                    let (ri0, ri1) = (shard_rank * ni_l, (shard_rank + 1) * ni_l);
                    layers.push(GpuLayer {
                        q_proj: up4(sl_rows(pick(&ln("self_attn.q_proj.weight")), rq0, rq1)),
                        k_proj: up4(sl_rows(pick(&ln("self_attn.k_proj.weight")), rk0, rk1)),
                        k_proj_bf16: Some(upload_bf16(&dev, &rows16(&l.k_proj, hidden(), rk0, rk1))),
                        v_proj: up4(sl_rows(pick(&ln("self_attn.v_proj.weight")), rk0, rk1)),
                        v_proj_bf16: Some(upload_bf16(&dev, &rows16(&l.v_proj, hidden(), rk0, rk1))),
                        o_proj: up4(sl_k(pick(&ln("self_attn.o_proj.weight")), rq0, rq1)),
                        gate_proj: up4(sl_rows(pick(&ln("mlp.gate_proj.weight")), ri0, ri1)),
                        up_proj: up4(sl_rows(pick(&ln("mlp.up_proj.weight")), ri0, ri1)),
                        down_proj: up4(sl_k(pick(&ln("mlp.down_proj.weight")), ri0, ri1)),
                        q_norm: upload_norm(&dev, &l.q_norm),
                        k_norm: upload_norm(&dev, &l.k_norm),
                        input_ln: upload_norm(&dev, &l.input_ln),
                        post_ln: upload_norm(&dev, &l.post_ln),
                        attn_kp: up4(pick(&ln("attention_conv.kernel_projection.weight"))),
                        attn_base: upload_bf16(&dev, &l.attention_conv.base_kernel),
                        mlp_kp: up4(pick(&ln("mlp_conv.kernel_projection.weight"))),
                        mlp_base: upload_bf16(&dev, &l.mlp_conv.base_kernel),
                    });
                    continue;
                }
                layers.push(GpuLayer {
                    q_proj: up4(pick(&ln("self_attn.q_proj.weight"))),
                    k_proj: up4(pick(&ln("self_attn.k_proj.weight"))),
                    k_proj_bf16: Some(up_b(&l.k_proj)),
                    v_proj: up4(pick(&ln("self_attn.v_proj.weight"))),
                    v_proj_bf16: Some(up_b(&l.v_proj)),
                    o_proj: up4(pick(&ln("self_attn.o_proj.weight"))),
                    gate_proj: up4(pick(&ln("mlp.gate_proj.weight"))),
                    up_proj: up4(pick(&ln("mlp.up_proj.weight"))),
                    down_proj: up4(pick(&ln("mlp.down_proj.weight"))),
                    q_norm: upload_norm(&dev, &l.q_norm),
                    k_norm: upload_norm(&dev, &l.k_norm),
                    input_ln: upload_norm(&dev, &l.input_ln),
                    post_ln: upload_norm(&dev, &l.post_ln),
                    attn_kp: up4(pick(&ln("attention_conv.kernel_projection.weight"))),
                    attn_base: upload_bf16(&dev, &l.attention_conv.base_kernel),
                    mlp_kp: up4(pick(&ln("mlp_conv.kernel_projection.weight"))),
                    mlp_base: upload_bf16(&dev, &l.mlp_conv.base_kernel),
                });
            } else {
                layers.push(GpuLayer {
                    q_proj: Df2W::Bf16(upload_bf16(&dev, &l.q_proj)),
                    k_proj: Df2W::Bf16(upload_bf16(&dev, &l.k_proj)),
                    k_proj_bf16: None,
                    v_proj: Df2W::Bf16(upload_bf16(&dev, &l.v_proj)),
                    v_proj_bf16: None,
                    o_proj: Df2W::Bf16(upload_bf16(&dev, &l.o_proj)),
                    gate_proj: Df2W::Bf16(upload_bf16(&dev, &l.gate_proj)),
                    up_proj: Df2W::Bf16(upload_bf16(&dev, &l.up_proj)),
                    down_proj: Df2W::Bf16(upload_bf16(&dev, &l.down_proj)),
                    q_norm: upload_norm(&dev, &l.q_norm),
                    k_norm: upload_norm(&dev, &l.k_norm),
                    input_ln: upload_norm(&dev, &l.input_ln),
                    post_ln: upload_norm(&dev, &l.post_ln),
                    attn_kp: Df2W::Bf16(upload_bf16(&dev, &l.attention_conv.kernel_projection)),
                    attn_base: upload_bf16(&dev, &l.attention_conv.base_kernel),
                    mlp_kp: Df2W::Bf16(upload_bf16(&dev, &l.mlp_conv.kernel_projection)),
                    mlp_base: upload_bf16(&dev, &l.mlp_conv.base_kernel),
                });
            }
        }
        let glob = match &quant {
            Some(q) => {
                let p = q.iter().find(|t| t.name() == "fc.weight").expect("baked: fc.weight packed");
                let fc_w = match p {
                    crate::dflash2::load::QuantTensor::Nvfp4(p) => Df2W::Nvfp4(upload_nvfp4(&dev, p)),
                    crate::dflash2::load::QuantTensor::Fp8(p) => Df2W::Fp8(upload_fp8(&dev, p)),
                };
                GpuGlobal {
                    fc: fc_w,
                    fc_bf16: Some(up_b(&w.fc)),
                    hidden_norm: upload_norm(&dev, &w.hidden_norm),
                    norm: upload_norm(&dev, &w.norm),
                }
            }
            None => GpuGlobal {
                fc: Df2W::Bf16(upload_bf16(&dev, &w.fc)),
                fc_bf16: None,
                hidden_norm: upload_norm(&dev, &w.hidden_norm),
                norm: upload_norm(&dev, &w.norm),
            },
        };
        let hp_w = up_b(&w.hidden_projection);
        let pred_cb = up_b(&w.predecessor_codebook);
        let succ_cb = up_b(&w.successor_codebook);
        anyhow::ensure!(head.is_some(), "Df2Round needs the trunk's lm_head (df2_borrow)");
        anyhow::ensure!(embed.is_some(), "Df2Round needs the trunk's embed (df2_borrow)");

        let inv = crate::dflash2::mirror::inv_freq(&cfg);
        let (cos_t, sin_t) = crate::dflash2::mirror::rope_tables(&cfg, &inv, max_pos);
        let cos_table = dev.htod_sync_copy(&cos_t)?;
        let sin_table = dev.htod_sync_copy(&sin_t)?;

        let dev2 = dev.clone();
        let alloc_z = move |n: usize| dev2.alloc_zeros::<bf16>(n).expect("alloc bf16");
        let dev3 = dev.clone();
        let alloc_zf = move |n: usize| dev3.alloc_zeros::<f32>(n).expect("alloc f32");
        let dev4 = dev.clone();
        let alloc_zi = move |n: usize| dev4.alloc_zeros::<i32>(n).expect("alloc i32");
        let dev5 = dev.clone();
        let dev6 = dev.clone();
        let mut k_ring = Vec::with_capacity(n_layers());
        let mut v_ring = Vec::with_capacity(n_layers());
        // P7-TP2: the ring KV is read partially before its rows are written (block-16 decode
        // spans ahead of the ring head). alloc_zeros does NOT zero (AGENTS §2.2) — at TP=1 the
        // stale bytes were locally deterministic so every gate stayed green, but the two TP
        // ranks hold DIFFERENT stale bytes and the first mtp-round digest diverges (agree()
        // FAIL at lockstep step 8, run-varying digests = uninitialized reads). Zero the whole
        // ring at install: both ranks read identical zeros wherever a row is unwritten.
        // (~21 MB one-time boot cost.)
        let ring_zero = vec![bf16::default(); nkv_l * ring_stride() * head_dim()];
        for _ in 0..n_layers() {
            // P2: the ring holds ONLY this rank's kv-heads post-shard (heads are independent —
            // no cross-rank attention traffic; the ring rows stay compile-time 2056×128,
            // ctx-free by construction post-S10R).
            k_ring.push(dev.htod_sync_copy(&ring_zero)?);
            v_ring.push(dev.htod_sync_copy(&ring_zero)?);
        }
        // Explicit zeroing of the persistent buffers a first round might read partially
        // (AGENTS §2.2: alloc_zeros does NOT zero).
        let zero8: Vec<bf16> = vec![bf16::default(); tap_concat_dim() * block()];
        let staging = dev.htod_sync_copy(&zero8)?;
        let blk = BlkScratch {
            h: alloc_z(hidden() * block()),
            normed: alloc_z(hidden() * block()),
            x_conv: alloc_z(hidden() * block()),
            dyn_attn: alloc_z(2 * conv_kernel() * conv_groups() * block()),
            q: alloc_z(num_heads() * head_dim() * block()),
            k: alloc_z(num_kv_heads() * head_dim() * block()),
            v: alloc_z(num_kv_heads() * head_dim() * block()),
            attn: alloc_z(num_heads() * head_dim() * block()),
            attn_out: alloc_z(hidden() * block()),
            fin: alloc_z(hidden() * block()),
            normed2: alloc_z(hidden() * block()),
            x_conv2: alloc_z(hidden() * block()),
            dyn_mlp: alloc_z(2 * conv_kernel() * conv_groups() * block()),
            gate: alloc_z(inter() * block()),
            up: alloc_z(inter() * block()),
            mlp_out: alloc_z(hidden() * block()),
            fin2: alloc_z(hidden() * block()),
            // gemm_dsp (m8) reads X rows 1..=8 at +HIDDEN for the selector projection: the
            // 9th row is a permanently-zero guard row (row 8 never written, only read by the
            // hp GEMM's 8th column, which no consumer reads).
            h_final: dev6.htod_sync_copy(&vec![bf16::default(); hidden() * (block() + 1)])
                .expect("h_final zeroed 9 rows"),
            cos8: alloc_zf(block() * head_dim()),
            sin8: alloc_zf(block() * head_dim()),
            slot_ids: dev.htod_sync_copy(&vec![0i32; block()])?,
        };
        let pos_blk: Vec<i32> = (0..block()).map(|b| b as i32).collect();
        let pos_blk = dev.htod_sync_copy(&pos_blk)?;
        let wrow_blk: Vec<i32> = (0..block()).map(|b| (ring() + b) as i32).collect();
        let wrow_blk = dev.htod_sync_copy(&wrow_blk)?;
        let wrow_ctl: Vec<i32> = (0..block()).map(|b| b as i32).collect();
        let wrow_ctl = dev.htod_sync_copy(&wrow_ctl)?;
        let toks_blk = dev.htod_sync_copy(&vec![0i32; block()])?;

        dev.synchronize()?;
        if sharded {
            eprintln!("[df2] round SHARDED across {world} ranks (rank {shard_rank}): \
                       qkv/gate/up col-split ({}q/{}kv heads, {} inter rows/rank), o/down K-split, \
                       ring KV [{} x {} x {}]/layer, 2 all-reduce sites/layer on the round stream",
                      nq_l, nkv_l, ni_l, nkv_l, ring_stride(), head_dim());
        }
        let mut me = Self {
            dev, stream, bk, layers, glob, hp_w, pred_cb, succ_cb, head, embed,
            cos_table, sin_table, k_ring, v_ring, nprev: 0, ring_slot: None, ring_len: 0,
            staging,
            sink: None,
            mask_ovr_row: None,
            mask_ovr_toks: None,
            th_raw: alloc_z(hidden() * block()),
            th: alloc_z(hidden() * block()),
            kc: alloc_z(num_kv_heads() * head_dim() * block()),
            vc: alloc_z(num_kv_heads() * head_dim() * block()),
            cos_c: alloc_zf(block() * head_dim()),
            sin_c: alloc_zf(block() * head_dim()),
            pos_c: alloc_zi(block()),
            wrow_c: alloc_zi(block()),
            slot_c: dev5.htod_sync_copy(&vec![0i32; block()])?,
            blk, pos_blk, wrow_blk, wrow_ctl, toks_blk,
            logits: alloc_z(vocab() * levels()),
            coarse_idx: dev5.alloc_zeros::<u32>(levels() * 256).expect("alloc coarse_idx"),
            h2d_stage: dev5.alloc_zeros::<i32>(2 * block() + 2).expect("alloc h2d_stage"),
            out_vals: alloc_zf(levels() * 16),
            out_ids: dev5.alloc_zeros::<u32>(levels() * 16).expect("alloc u32"),
            hp_bf16: alloc_z(selector_rank() * block()),  // the hp GEMM is launched per m8 tile; block() cols
            hp_f32: alloc_zf(selector_rank() * block()),
            unary_ctl: alloc_zf(levels() * 16),
            walk_tokens: dev5.alloc_zeros::<u32>(levels()).expect("alloc u32"),
            ctl_attn_ref: alloc_z(num_heads() * head_dim() * block()),
            ctl_attn: alloc_z(num_heads() * head_dim() * block()),
            walk_scores: alloc_zf(levels() * 16),
            walk_out_tok: dev5.alloc_zeros::<u32>(levels() + levels() * 16).expect("alloc u32"),
            walk_out_q: alloc_zf(levels() + levels() * 16),
            walk_seeds: dev5.alloc_zeros::<u32>(levels()).expect("alloc u32"),
            max_c,
            ntot_dev: 0,
            anchor_dev: 0,
            round_graph: None,
            pos_at: None,
            inject_graphs: std::collections::HashMap::new(),
            ntot_buf: None,
            anchor_buf: None,
            ar,
            nq_l,
            nkv_l,
            ni_l,
        };
        // DFLASH2-GEN (mask sub-item): measure the trunk's quantized mask row against the
        // artifact's shipped BF16 copy ONCE at boot; logs, never fails the load.
        me.mask_row_check(dir);
        Ok(me)
    }

}
