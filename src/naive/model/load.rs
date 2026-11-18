//! Loading `GpuModel`: slice / layer upload from the pack (staged), including the TP shard split.
//! Siblings: `step.rs` runs the loaded layers; `mod.rs` has the directory map.

use super::*;

impl GpuModel {
    /// Load layers `0..n_layers` (+ embeddings, final norm, lm_head). `max_pos` sizes the DSA caches
    /// and the indexer key planes; `ring` (power of two >= window + max chunk) the SWA rings;
    /// `n_slots` the per-request slot dimension in front of every pos-keyed cache.
    pub fn load_slice(pack: &NaivePack, n_layers: usize, max_pos: usize, ring: usize, n_slots: usize) -> Result<GpuModel> {
        let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
        Self::load_slice_on(dev, pack, n_layers, max_pos, ring, n_slots, super::shard::Shard::single())
    }

    /// The first `n_layers` layers, this rank's shard, on `dev` (the TP transport attaches afterwards).
    pub fn load_slice_on(dev: Arc<CudaDevice>, pack: &NaivePack, n_layers: usize, max_pos: usize, ring: usize, n_slots: usize, shard: super::shard::Shard) -> Result<GpuModel> {
        let c = pack.cfg.clone();
        if let Some((w, r)) = pack.shard {
            ensure!(w == shard.world && r == shard.rank, "{} is rank {r}'s shard of world {w}; this load is rank {} of world {}", pack.dir, shard.rank, shard.world);
        }
        ensure!(n_layers <= c.num_layers, "n_layers {n_layers}");
        ensure!(ring.is_power_of_two() && ring >= 2 * c.sliding_window, "ring {ring} must be a power of two >= 2*window");
        load_module(&dev)?;
        let t0 = std::time::Instant::now();
        // LOADFAST (stage.rs): staged host images — cache + worker pipeline. The pre section
        // (embed/norm/lm_head) arrives first, then the layers, uploaded in order as staged.
        let opts = super::stage::LoadOpts::from_opts();
        let etp = etp_of(pack, &shard);
        crate::metrics::NAIVE_MOE_TP.store(etp, std::sync::atomic::Ordering::Relaxed);
        // expert TP: every expert is local (identity map); else the interleaved ownership map
        let local_of = dev.htod_sync_copy(&if etp { (0..c.num_experts as i32).collect::<Vec<i32>>() } else { shard.local_of(c.num_experts) })?;
        if etp && shard.rank == 0 {
            println!("[naive] experts: TENSOR-parallel (--naive-moe-shard tp; auto = tp at TP=4): every rank holds {} of each expert's {} intermediate columns — no per-layer expert-count skew; \
                      the MoE partial sums are over K slices (a different fp32 summation order from the expert-parallel deal: tolerance class)",
                     c.moe_intermediate_size / shard.world, c.moe_intermediate_size);
        } else if shard.world > 1 && shard.rank == 0 {
            println!("[naive] experts: expert-parallel (whole experts per rank; --naive-moe-shard auto = ep at TP=2, where expert TP has not run on \
                      hardware; --naive-moe-shard tp asks for it)");
        }
        let mut layers = Vec::with_capacity(n_layers);
        let (mut embed, mut norm, mut lm_head) = (None, None, None);
        let (mut v0, mut vl) = (0usize, c.vocab_size);
        super::stage::run(pack, n_layers, shard, &opts, |item| -> Result<()> {
            match item {
                super::stage::Item::Pre(pre) => {
                    embed = Some(dev.htod_sync_copy(pre.plain("model.embed_tokens.weight")?)?);
                    norm = Some(dev.htod_sync_copy(pre.plain("model.norm.weight")?)?);
                    lm_head = Some(pre.quad("lm_head")?.upload(&dev)?);
                    if shard.world != 1 {
                        let (a, n) = shard.span(c.vocab_size)?;
                        v0 = a; vl = n;
                    }
                    Ok(())
                }
                super::stage::Item::Layer(l, s) => {
                    layers.push(Self::upload_layer(&dev, &c, l, max_pos, ring, n_slots, &shard, etp, s)?);
                    crate::dprintln!("[naive] layer {l} ({:?}, {}) loaded, {:.1} s", c.attn[l], if c.moe[l] { "moe" } else { "dense" }, t0.elapsed().as_secs_f32());
                    Ok(())
                }
            }
        })?;
        let embed = embed.context("staged pre section: embed missing")?;
        let norm = norm.context("staged pre section: norm missing")?;
        let lm_head = lm_head.context("staged pre section: lm_head missing")?;
        let stream = dev.fork_default_stream().context("fork stream")?;
        // AGENTS.md S2 rule 2: `alloc_zeros` does NOT zero (async alloc). Both buffers below are read by
        // kernels on the FIRST decode step. `sids_buf` in particular is written by ONLY the
        // --probe-naive-slots path; the serving path never uploads it, so every decode step read
        // uninitialised pool memory as the slot id and addressed garbage KV planes -- silent, no
        // bounds check, corrupt logits (every draft rejected, output degenerating to '*').
        // cudarc 0.9.15 has no memset_* API, so zero explicitly the way decode.rs:290 does.
        let zero_sink = dev.htod_sync_copy(&vec![0u16; c.num_heads])?;
        // the per-row slot ids every cache-addressing kernel takes (uploaded per step by the caller)
        // MAX_PREFILL_ROWS, NOT MAX_ROWS: every cache-addressing kernel indexes slot_ids[m] for
        // m in 0..m, and the serving path feeds THIS ONE buffer chunks of up to `chunk` rows
        // (PREFILL_CHUNK = 1024, clamped to MAX_PREFILL_ROWS). Sized at MAX_ROWS (16) it silently
        // read past the allocation for rows >= 16, and the garbage slot id is used as the KV plane
        // stride `slot * (kv * S * hd)` with S = max_pos = 262144 -- an enormous stride, so every
        // prefill wider than 16 rows died with CUDA_ERROR_ILLEGAL_ADDRESS while short prompts
        // (m <= 16) served fine. The decode/verify path still never exceeds MAX_ROWS.
        let sids_buf = dev.htod_sync_copy(&vec![0i32; MAX_PREFILL_ROWS])?;
        let sids: u64 = dptr(&sids_buf);
        dev.synchronize()?;
        // F13 counters: an explicit zero upload (alloc_zeros does not zero: AGENTS §2.2), then the kernels
        // keep them zero (the last CTA of every tile resets its counter)
        let fuse_cnt = dev.htod_sync_copy(&vec![0i32; super::gpu::FUSE_COUNTERS])?;
        dev.synchronize()?;
        Ok(GpuModel { dev, stream, cfg: c, zero_sink, shard, tp: None, local_of, layers, embed, norm, lm_head, v0, vl, max_pos, ring, n_slots, sids, sids_buf, fuse_cnt })
    }

    /// One layer's weights + caches on the device: this rank's shard (`Shard::single()` = whole layer).
    /// One layer's weights + caches on the device: this rank's shard (`Shard::single()` = whole layer).
    /// The classic one-shot path (stage + upload inline, no cache) — probes keep this shape.
    pub fn load_layer(dev: &Arc<CudaDevice>, pack: &NaivePack, l: usize, max_pos: usize, ring: usize, n_slots: usize, shard: &super::shard::Shard) -> Result<GpuLayer> {
        let s = super::stage::stage_layer(pack, l, shard)?;
        Self::upload_layer(dev, &pack.cfg, l, max_pos, ring, n_slots, shard, etp_of(pack, shard), &s)
    }

    /// The device half of a staged layer (the old `load_layer`'s upload side, verbatim): quads are
    /// found by name prefix (staging decided whole/slice; the name records which), plains by name.
    /// The interm_div guard runs HERE, so cold (pack) and warm (cache) loads validate identically.
    fn upload_layer(dev: &Arc<CudaDevice>, c: &NaiveConfig, l: usize, max_pos: usize, ring: usize, n_slots: usize,
                    shard: &super::shard::Shard, etp: bool, s: &super::stage::StagedLayer) -> Result<GpuLayer> {
        let (h, hd, vd) = (c.hidden_size, c.head_dim, c.v_head_dim);
        {
            let p = format!("model.layers.{l}.");
            let kv_total = c.kv_heads(l);
            let (q0, nh_l) = shard.q_heads(c.num_heads)?;
            let (k0, kv_l) = shard.kv_heads(c.num_heads, kv_total)?;
            let kv = kv_l;
            let slots = if c.attn[l] == AttnKind::Swa { ring } else { max_pos };
            let kc = dev.alloc_zeros::<u16>(kv * slots * hd * n_slots)?;
            let vc = dev.alloc_zeros::<u16>(kv * slots * vd * n_slots)?;
            // attention projections: q columns of this rank's heads, k/v columns of its kv heads
            // (v is stored per-head padded to hd), o rows of its heads' values
            let (q, k, v, o, k_off, v_off) = if shard.world == 1 {
                (s.quad(&format!("{p}self_attn.q_proj"))?.upload(dev)?,
                 s.quad(&format!("{p}self_attn.k_proj"))?.upload(dev)?,
                 s.quad(&format!("{p}self_attn.v_proj"))?.upload(dev)?,
                 s.quad(&format!("{p}self_attn.o_proj"))?.upload(dev)?,
                 0, 0)
            } else {
                // a staged shard carries o_proj PRE-SLICED to this rank's rows (k == nh_l * vd);
                // the whole pack was sliced at staging time
                let ks = s.quad(&format!("{p}self_attn.k_proj"))?;
                let vs = s.quad(&format!("{p}self_attn.v_proj"))?;
                let k_off = ks.col_off;
                let v_off = vs.col_off;
                (s.quad(&format!("{p}self_attn.q_proj"))?.upload(dev)?, ks.upload(dev)?, vs.upload(dev)?,
                 s.quad(&format!("{p}self_attn.o_proj"))?.upload(dev)?, k_off, v_off)
            };
            let idx = if c.attn[l] == AttnKind::Dsa {
                Some(GpuIndexer {
                    wq: s.quad(&format!("{p}self_attn.indexer.wq"))?.upload(dev)?,
                    wk: s.quad(&format!("{p}self_attn.indexer.wk"))?.upload(dev)?,
                    wp: s.quad(&format!("{p}self_attn.indexer.weights_proj"))?.upload(dev)?,
                    kn_w: dev.htod_sync_copy(s.plain(&format!("{p}self_attn.indexer.k_norm.weight"))?)?,
                    kn_b: dev.htod_sync_copy(s.plain(&format!("{p}self_attn.indexer.k_norm.bias"))?)?,
                    kq: dev.alloc_zeros::<u8>(max_pos * c.index_head_dim * n_slots)?,
                    ks: dev.alloc_zeros::<f32>(max_pos * n_slots)?,
                })
            } else { None };
            let sink = if c.has_sink(l) {
                Some(dev.htod_sync_copy(s.plain(&format!("{p}self_attn.attention_sink_bias"))?)?)
            } else { None };
            let mlp = if c.moe[l] {
                let (e, mi) = (c.num_experts, c.moe_intermediate_size);
                let mi_l = if etp { mi / shard.world } else { mi };
                let experts: Vec<usize> = if etp { (0..e).collect() } else { shard.local_experts(e) };
                let el = experts.len();
                let m = s.moe.as_ref().with_context(|| format!("layer {l}: staged image has no MoE planes"))?;
                super::stage::check_interm_div(l, m, c.moe_interm_div[l])?;
                let g_words = ((h / 16) * (mi_l / 16) * m.bpb) as u64;
                let d_words = ((mi_l / 16) * (h / 16) * m.bpb) as u64;
                ensure!(m.g_tr.len() as u64 == el as u64 * g_words && m.d_tr.len() as u64 == el as u64 * d_words && m.svh_g.len() == el * mi_l && m.suh_d.len() == el * mi_l,
                        "layer {l}: expert planes do not match {el} experts x mi {mi_l}");
                let offs_g = dev.htod_sync_copy(&(0..el as u64).map(|i| i * g_words).collect::<Vec<u64>>())?;
                let offs_d = dev.htod_sync_copy(&(0..el as u64).map(|i| i * d_words).collect::<Vec<u64>>())?;
                GpuMlp::Moe(GpuMoe {
                    router: dev.htod_sync_copy(s.plain(&format!("{p}mlp.gate.weight"))?)?,
                    bias: dev.htod_sync_copy(s.plain32(&format!("{p}mlp.gate.e_score_correction_bias"))?)?,
                    g_tr: dev.htod_sync_copy(&m.g_tr)?, u_tr: dev.htod_sync_copy(&m.u_tr)?, d_tr: dev.htod_sync_copy(&m.d_tr)?,
                    suh_g: dev.htod_sync_copy(&m.suh_g)?, svh_g: dev.htod_sync_copy(&m.svh_g)?,
                    suh_u: dev.htod_sync_copy(&m.suh_u)?, svh_u: dev.htod_sync_copy(&m.svh_u)?,
                    suh_d: dev.htod_sync_copy(&m.suh_d)?, svh_d: dev.htod_sync_copy(&m.svh_d)?,
                    g_words, u_words: g_words, d_words, bpb: m.bpb, offs_g, offs_d,
                    scale: c.pack_router_scale(l), mi_l, el,
                })
            } else {
                // dense MLP: the staged quad under each name is whole (world 1) or already
                // column/row-sliced to this rank (world > 1) — staging made that decision
                GpuMlp::Dense {
                    gate: s.quad(&format!("{p}mlp.gate_proj"))?.upload(dev)?,
                    up: s.quad(&format!("{p}mlp.up_proj"))?.upload(dev)?,
                    down: s.quad(&format!("{p}mlp.down_proj"))?.upload(dev)?,
                }
            };
            Ok(GpuLayer {
                kind: c.attn[l], nh_l, kv_l, k_off, v_off,
                in_norm: dev.htod_sync_copy(s.plain(&format!("{p}input_layernorm.weight"))?)?,
                post_norm: dev.htod_sync_copy(s.plain(&format!("{p}post_attention_layernorm.weight"))?)?,
                q, k, v, o, sink, idx, mlp, kc, vc, slots,
            })
        }
    }

}
