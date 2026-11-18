//! Construction of `Dsv4AttnRuntime` (single-kind and multi-kind), layer and MTP-stage weight upload,
//! and attention-state allocation. Siblings: `ops.rs` / `attn.rs` run what is built here; `mod.rs` has the map.

use super::*;

impl Dsv4AttnRuntime {
    /// Build the runtime: load all modules + the per-kind RoPE table (`positions` rows).
    /// `kind` selects the RoPE table (YaRN for CSA/HCA, plain for SWA) and whether the
    /// comp module is loaded (CSA/HCA only — SWA skips the load to stay 3A-byte-identical).
    pub fn new(dev: &Arc<CudaDevice>, kind: dsv4_load::LayerKind, positions: usize, cfg: &dsv4_load::Dsv4Config) -> Result<Self> {
        let stream = dsv4_gpu::blocking_compute_stream(dev);
        let spine = Dsv4Kernels::load(dev, SPINE_FUNCS).context("load gpu_dsv4.ptx (spine)")?;
        spine
            .set_dynamic_smem("dsv4_gather_attn", GATHER_SMEM)
            .context("set_dynamic_smem gather_attn")?;
        spine
            .set_dynamic_smem("dsv4_fused_gather_b", GATHER_SMEM)
            .context("set_dynamic_smem fused_gather_b")?;
        spine
            .set_dynamic_smem("dsv4_hc_pre_fused_b", HC_PRE_FUSED_SMEM)
            .context("set_dynamic_smem hc_pre_fused_b")?;
        let attn = Dsv4Kernels::load_module(dev, crate::ptx_embed::GPU_DSV4_ATTN, "gpu_dsv4_attn", ATTN_FUNCS)
            .context("load gpu_dsv4_attn.ptx (lane 3A/3C)")?;
        // CSA/HCA need the comp module (compressor + indexer kernels). SWA skips it
        // (no compressor/indexer constructed — stays 3A-byte-identical).
        let comp = if matches!(kind, dsv4_load::LayerKind::Csa | dsv4_load::LayerKind::Hca) {
            Some(CompKernels::load(dev).context("CompKernels::load (lane 3C)")?)
        } else {
            None
        };

        let bptx = Ptx::from_src(crate::ptx_embed::GPU_BATCH);
        dev.load_ptx(bptx, "gpu_batch", BATCH_FUNCS).context("load_ptx gpu_batch")?;
        let dptx = Ptx::from_src(crate::ptx_embed::GPU_DSV4);
        dev.load_ptx(dptx, "gpu_dsv4", MOE_DFUNCS).context("load_ptx gpu_dsv4 (moe)")?;
        let collect = |m: &str, names: &[&'static str]| -> HashMap<String, CudaFunction> {
            names
                .iter()
                .map(|n| (n.to_string(), dev.get_func(m, n).unwrap_or_else(|| panic!("missing {m}::{n}"))))
                .collect()
        };
        let bk = collect("gpu_batch", BATCH_FUNCS);
        let df = collect("gpu_dsv4", MOE_DFUNCS);

        // RoPE table for this layer kind (dsv4_cpu::layer_rope_table semantics).
        let table = dsv4_cpu::layer_rope_table(cfg, kind, positions);
        let rope = DevRope::from_cpu(dev, &table).context("upload rope table")?;
        Ok(Self {
            dev: dev.clone(),
            stream,
            spine,
            attn,
            comp,
            bk,
            df,
            rope,
            ropes: HashMap::new(),
            window: cfg.window_size,
            tp_ctx_dptr: 0,
        })
    }

    /// Full-trunk constructor: one runtime serving ALL three layer kinds (SWA + CSA + HCA),
    /// used by `Dsv4GpuModel`. Loads the comp module unconditionally (CSA/HCA need it; SWA
    /// doesn't construct a compressor/indexer, so the extra module is harmless) and builds
    /// all three RoPE tables (§B.1.3 — SWA plain θ vs CSA/HCA YaRN). `attn_forward` selects
    /// the rope via [`rope_for`](Self::rope_for); the lane `new` path stays unaffected.
    pub fn new_multikind(
        dev: &Arc<CudaDevice>,
        positions: usize,
        cfg: &dsv4_load::Dsv4Config,
    ) -> Result<Self> {
        let stream = dsv4_gpu::blocking_compute_stream(dev);
        let spine = Dsv4Kernels::load(dev, SPINE_FUNCS).context("load gpu_dsv4.ptx (spine)")?;
        spine
            .set_dynamic_smem("dsv4_gather_attn", GATHER_SMEM)
            .context("set_dynamic_smem gather_attn")?;
        spine
            .set_dynamic_smem("dsv4_fused_gather_b", GATHER_SMEM)
            .context("set_dynamic_smem fused_gather_b")?;
        spine
            .set_dynamic_smem("dsv4_hc_pre_fused_b", HC_PRE_FUSED_SMEM)
            .context("set_dynamic_smem hc_pre_fused_b")?;
        let attn = Dsv4Kernels::load_module(dev, crate::ptx_embed::GPU_DSV4_ATTN, "gpu_dsv4_attn", ATTN_FUNCS)
            .context("load gpu_dsv4_attn.ptx (lane 3A/3C)")?;
        // Multikind always needs the comp module (CSA + HCA layers construct compressors).
        let comp = Some(CompKernels::load(dev).context("CompKernels::load (multikind)")?);

        let bptx = Ptx::from_src(crate::ptx_embed::GPU_BATCH);
        dev.load_ptx(bptx, "gpu_batch", BATCH_FUNCS).context("load_ptx gpu_batch")?;
        let dptx = Ptx::from_src(crate::ptx_embed::GPU_DSV4);
        dev.load_ptx(dptx, "gpu_dsv4", MOE_DFUNCS).context("load_ptx gpu_dsv4 (moe)")?;
        let collect = |m: &str, names: &[&'static str]| -> HashMap<String, CudaFunction> {
            names
                .iter()
                .map(|n| (n.to_string(), dev.get_func(m, n).unwrap_or_else(|| panic!("missing {m}::{n}"))))
                .collect()
        };
        let bk = collect("gpu_batch", BATCH_FUNCS);
        let df = collect("gpu_dsv4", MOE_DFUNCS);

        // All three RoPE tables (§B.1.3). SWA forces YaRN off (plain θ=10000); CSA/HCA YaRN.
        // DevRope owns a CudaSlice (not Clone), so SWA lives in the `rope` fallback field and
        // `ropes` holds only CSA/HCA — `rope_for(Swa)` falls through to `rope`.
        let rope = DevRope::from_cpu(dev, &dsv4_cpu::layer_rope_table(cfg, dsv4_load::LayerKind::Swa, positions))
            .context("upload rope table (multikind SWA)")?;
        let mut ropes = HashMap::new();
        for kind in [dsv4_load::LayerKind::Csa, dsv4_load::LayerKind::Hca] {
            let table = dsv4_cpu::layer_rope_table(cfg, kind, positions);
            ropes.insert(kind, DevRope::from_cpu(dev, &table).context("upload rope table (multikind)")?);
        }
        Ok(Self {
            dev: dev.clone(),
            stream,
            spine,
            attn,
            comp,
            bk,
            df,
            rope,
            ropes,
            window: cfg.window_size,
            tp_ctx_dptr: 0,
        })
    }

    /// RoPE table for `kind`: the multikind table if present (full trunk), else the lane's
    /// single-kind `rope` (the 3A/3C replay path).
    pub fn rope_for(&self, kind: dsv4_load::LayerKind) -> &DevRope {
        self.ropes.get(&kind).unwrap_or(&self.rope)
    }

    /// Synchronize the compute stream (surfaces any prior async kernel error).
    pub fn synchronize(&self) -> Result<()> {
        self.dev.synchronize().map_err(|e| anyhow!("stream sync: {e:?}"))
    }

    /// Input embedding gather + replicate to hc streams (§1.1, model.py:916 forward_embed):
    /// out [s, hc*dim] bf16 = embed[ids[t], :] copied into every stream. `embed` is [vocab, dim].
    pub fn embed_tokens<X: dsv4_gpu::Dsv4Buf<bf16>>(&self, embed: &B, ids: &CudaSlice<i32>, s: usize, cfg: &dsv4_load::Dsv4Config) -> Result<X> {
        let (hc, dim) = (cfg.hc_mult, cfg.dim);
        let out = X::alloc_zeros(&self.dev, self.stream.stream, s * hc * dim)?;
        let (s_i, dim_i, hc_i) = (s as i32, dim as i32, hc as i32);
        dsv4_launch!(self.spine, "dsv4_embed_b", self.stream.stream, (s as u32, 1, 1), (256, 1, 1), 0,
            (&out, embed, ids, &s_i, &dim_i, &hc_i))?;
        Ok(out)
    }

    /// DSpark trunk interface (§B.10): mean-over-streams of 3 layer outputs (layers 40/41/42)
    /// concatenated → main_hidden [s, 3*dim]. `dsv4_main_hidden_b` fuses mean + concat.
    pub fn compute_main_hidden(&self, x40: &B, x41: &B, x42: &B, s: usize, cfg: &dsv4_load::Dsv4Config) -> Result<B> {
        let (hc, dim) = (cfg.hc_mult, cfg.dim);
        let out = self.dev.alloc_zeros::<bf16>(s * 3 * dim)?;
        let total = (s * 3 * dim) as u32;
        let (s_i, hc_i, dim_i) = (s as i32, hc as i32, dim as i32);
        dsv4_launch!(self.spine, "dsv4_main_hidden_b", self.stream.stream,
            (ceil256(total as usize), 1, 1), (256, 1, 1), 0,
            (&out, x40, x41, x42, &s_i, &hc_i, &dim_i))?;
        Ok(out)
    }

    /// Load + upload trunk layer `layer_id` (SWA for this lane; the loader is kind-generic).
    /// `rank`/`world` TP-shard the routed-expert bank when `world > 1` (§5: each rank uploads its
    /// `[rank·ne/world, (rank+1)·ne/world)` expert slice via `Dsv4MoeGpu::upload_sharded`;
    /// everything else — attn, mHC, router, shared, norms — is replicated). `world == 1` is the
    /// unsharded single-process path (`Dsv4MoeGpu::upload`, e_base=0/e_span=ne).
    pub fn upload_layer(
        &self,
        bundle: &Path,
        cfg: &dsv4_load::Dsv4Config,
        layer_id: usize,
        rank: usize,
        world: usize,
    ) -> Result<Dsv4GpuLayer> {
        let dev = &self.dev;
        let fp8 = |key: &str| -> Result<Fp8Weight> {
            let name = format!("layers.{layer_id}.{key}");
            let (shape, codes, sb) = dsv4_load::read_raw_fp8(bundle, &name)
                .with_context(|| format!("read_raw_fp8 {name}"))?;
            let (m, k) = (shape[0], shape[1]);
            let wt = quant::repack_fp8_mma(&codes, m, k);
            Ok(Fp8Weight {
                wt: dev.htod_sync_copy(&wt).with_context(|| format!("upload {name} wt"))?,
                sb: dev.htod_sync_copy(&sb).with_context(|| format!("upload {name} sb"))?,
                m,
                k,
            })
        };
        let wq_a = fp8("attn.wq_a.weight")?;
        let wq_b = fp8("attn.wq_b.weight")?;
        let wkv = fp8("attn.wkv.weight")?;
        let wo_b = fp8("attn.wo_b.weight")?;
        // Shared expert: fuse [w1; w3] into one [4096, 4096] gate_up GEMM (per-16-row-tile
        // repack makes concat == fused; scale blocks stack at the 128-row boundary).
        let sh_w1 = fp8("ffn.shared_experts.w1.weight")?;
        let sh_w3 = fp8("ffn.shared_experts.w3.weight")?;
        let sh_w2 = fp8("ffn.shared_experts.w2.weight")?;
        let sh_gu = {
            let (m1, m3, k) = (sh_w1.m, sh_w3.m, sh_w1.k);
            anyhow::ensure!(k == sh_w3.k && m1 == m3 && (m1 + m3) % 128 == 0, "shared gu geometry");
            let name = |base: &str| format!("layers.{layer_id}.{base}");
            let (_, c1, s1) = dsv4_load::read_raw_fp8(bundle, &name("ffn.shared_experts.w1.weight"))?;
            let (_, c3, s3) = dsv4_load::read_raw_fp8(bundle, &name("ffn.shared_experts.w3.weight"))?;
            let mut codes = c1;
            codes.extend_from_slice(&c3);
            let mut sb = s1;
            sb.extend_from_slice(&s3);
            let wt = quant::repack_fp8_mma(&codes, m1 + m3, k);
            Fp8Weight {
                wt: dev.htod_sync_copy(&wt)?,
                sb: dev.htod_sync_copy(&sb)?,
                m: m1 + m3,
                k,
            }
        };

        // Strict-loaded layer for everything else (wo_a bf16 dequant, norms, hc, gate, experts).
        let layer = dsv4_load::load_layer(bundle, cfg, layer_id).context("load_layer")?;
        let mut map = layer.tensors;
        let wo_a: Vec<bf16> = match map.remove("attn.wo_a.weight") {
            Some(dsv4_load::HostTensor::BF16 { data, shape }) => {
                anyhow::ensure!(shape == vec![cfg.o_groups * cfg.o_lora_rank, cfg.dim], "wo_a shape {shape:?}");
                data
            }
            other => return Err(anyhow!("attn.wo_a.weight: expected BF16 dequant, got {:?}", other.map(|t| t.shape().to_vec()))),
        };
        let gate_w_bf: Vec<bf16> = match map.remove("ffn.gate.weight") {
            Some(dsv4_load::HostTensor::BF16 { data, .. }) => data,
            other => return Err(anyhow!("ffn.gate.weight: expected BF16, got {:?}", other.map(|t| t.shape().to_vec()))),
        };
        let gate_w: Vec<f32> = gate_w_bf.iter().map(|v| v.to_f32()).collect();
        let tid2eid = if cfg.is_hash_layer(layer_id) {
            let t = dsv4_cpu::take_i32(&mut map, "ffn.gate.tid2eid", cfg.vocab_size * cfg.n_activated_experts)?;
            Some(dev.htod_sync_copy(&t)?)
        } else {
            None
        };
        let gate_bias = if cfg.is_hash_layer(layer_id) {
            None
        } else {
            let b = dsv4_cpu::take_f32(&mut map, "ffn.gate.bias", cfg.n_routed_experts)?;
            Some(dev.htod_sync_copy(&b)?)
        };
        let q_norm = dsv4_cpu::take_f32(&mut map, "attn.q_norm.weight", cfg.q_lora_rank)?;
        let kv_norm = dsv4_cpu::take_f32(&mut map, "attn.kv_norm.weight", cfg.head_dim)?;
        let attn_norm = dsv4_cpu::take_f32(&mut map, "attn_norm.weight", cfg.dim)?;
        let ffn_norm = dsv4_cpu::take_f32(&mut map, "ffn_norm.weight", cfg.dim)?;
        let sink = dsv4_cpu::take_f32(&mut map, "attn.attn_sink", cfg.n_heads)?;
        let hc_attn_fn = dsv4_cpu::take_f32(&mut map, "hc_attn_fn", 24 * cfg.hc_mult * cfg.dim)?;
        let hc_attn_base = dsv4_cpu::take_f32(&mut map, "hc_attn_base", 24)?;
        let hc_attn_scale = dsv4_cpu::take_f32(&mut map, "hc_attn_scale", 3)?;
        let hc_ffn_fn = dsv4_cpu::take_f32(&mut map, "hc_ffn_fn", 24 * cfg.hc_mult * cfg.dim)?;
        let hc_ffn_base = dsv4_cpu::take_f32(&mut map, "hc_ffn_base", 24)?;
        let hc_ffn_scale = dsv4_cpu::take_f32(&mut map, "hc_ffn_scale", 3)?;

        // Lane 3C: attention compressor + indexer weights (CSA/HCA only). Held on the
        // host; uploaded into GpuCompressor/GpuIndexer at `new_state`. The keys and
        // geometry mirror dsv4_cpu::cpu_layer_core's CSA/Hca arms exactly (§A.2).
        let kind = cfg.layer_kind(layer_id);
        let (hd, qlr, ihd, inh) = (cfg.head_dim, cfg.q_lora_rank, cfg.index_head_dim, cfg.index_n_heads);
        let comp_load = match kind {
            dsv4_load::LayerKind::Swa => None,
            dsv4_load::LayerKind::Csa => Some(CompLoad {
                spec: CompSpec::csa_attn(cfg.dim, cfg.rope_head_dim),
                w: dsv4_cpu::CompressorWeights {
                    wkv: dsv4_cpu::take_f32(&mut map, "attn.compressor.wkv.weight", 2 * hd * cfg.dim)?,
                    wgate: dsv4_cpu::take_f32(&mut map, "attn.compressor.wgate.weight", 2 * hd * cfg.dim)?,
                    norm: dsv4_cpu::take_f32(&mut map, "attn.compressor.norm.weight", hd)?,
                    ape: dsv4_cpu::take_f32(&mut map, "attn.compressor.ape", 4 * 2 * hd)?,
                    ratio: 4, head_dim: hd, rope_dim: cfg.rope_head_dim,
                    overlap: true, rotate: false, sim_group: 64, dim: cfg.dim,
                },
            }),
            dsv4_load::LayerKind::Hca => Some(CompLoad {
                spec: CompSpec::hca_attn(cfg.dim, cfg.rope_head_dim),
                w: dsv4_cpu::CompressorWeights {
                    wkv: dsv4_cpu::take_f32(&mut map, "attn.compressor.wkv.weight", hd * cfg.dim)?,
                    wgate: dsv4_cpu::take_f32(&mut map, "attn.compressor.wgate.weight", hd * cfg.dim)?,
                    norm: dsv4_cpu::take_f32(&mut map, "attn.compressor.norm.weight", hd)?,
                    ape: dsv4_cpu::take_f32(&mut map, "attn.compressor.ape", 128 * hd)?,
                    ratio: 128, head_dim: hd, rope_dim: cfg.rope_head_dim,
                    overlap: false, rotate: false, sim_group: 64, dim: cfg.dim,
                },
            }),
        };
        // CSA indexer: wq_b (raw fp8, MMA-repacked — NOT the dequant in `map`),
        // weights_proj (bf16-valued f32), and its own rotate compressor (ratio 4, hd 128).
        let idx_load = if matches!(kind, dsv4_load::LayerKind::Csa) {
            let idx_name = format!("layers.{layer_id}.attn.indexer.wq_b.weight");
            let (shape, codes, sb) = dsv4_load::read_raw_fp8(bundle, &idx_name)
                .with_context(|| format!("read_raw_fp8 {idx_name}"))?;
            anyhow::ensure!(shape[0] == inh * ihd && shape[1] == qlr, "indexer wq_b shape {shape:?}");
            let wq_b_wt = quant::repack_fp8_mma(&codes, inh * ihd, qlr);
            let weights_proj = dsv4_cpu::take_bf16_as_f32(
                &mut map, "attn.indexer.weights_proj.weight", inh * cfg.dim)?;
            Some(IndexerLoad {
                comp: CompLoad {
                    spec: CompSpec::indexer(cfg.dim, cfg.rope_head_dim),
                    w: dsv4_cpu::CompressorWeights {
                        wkv: dsv4_cpu::take_f32(&mut map, "attn.indexer.compressor.wkv.weight", 2 * ihd * cfg.dim)?,
                        wgate: dsv4_cpu::take_f32(&mut map, "attn.indexer.compressor.wgate.weight", 2 * ihd * cfg.dim)?,
                        norm: dsv4_cpu::take_f32(&mut map, "attn.indexer.compressor.norm.weight", ihd)?,
                        ape: dsv4_cpu::take_f32(&mut map, "attn.indexer.compressor.ape", 4 * 2 * ihd)?,
                        ratio: 4, head_dim: ihd, rope_dim: cfg.rope_head_dim,
                        overlap: true, rotate: true, sim_group: 32, dim: cfg.dim,
                    },
                },
                wq_b_wt,
                wq_b_sb: sb,
                weights_proj,
            })
        } else {
            None
        };

        let host = dsv4_moe::pack_moe_layer(
            &dsv4_load::Dsv4Layer {
                tensors: std::mem::take(&mut map),
                experts_w1: layer.experts_w1,
                experts_w2: layer.experts_w2,
                experts_w3: layer.experts_w3,
            },
            cfg,
        )
        .context("pack_moe_layer")?;
        let moe = if world > 1 {
            let e_span = cfg.n_routed_experts / world;
            anyhow::ensure!(e_span * world == cfg.n_routed_experts, "ne {} not divisible by world {world}", cfg.n_routed_experts);
            Dsv4MoeGpu::upload_sharded(dev, &host, rank * e_span, e_span).context("Dsv4MoeGpu::upload_sharded")?
        } else {
            Dsv4MoeGpu::upload(dev, &host).context("Dsv4MoeGpu::upload")?
        };

        let up = |v: &[f32]| -> Result<S> { Ok(dev.htod_sync_copy(v)?) };
        Ok(Dsv4GpuLayer {
            kind,
            wq_a,
            wq_b,
            wkv,
            wo_b,
            sh_gu,
            sh_w2,
            wo_a: dev.htod_sync_copy(&wo_a)?,
            wo_a_q: Some(quantize_wo_a(dev, &wo_a, cfg.o_groups, cfg.o_lora_rank, cfg.dim)?),
            q_norm: up(&q_norm)?,
            kv_norm: up(&kv_norm)?,
            attn_norm: up(&attn_norm)?,
            ffn_norm: up(&ffn_norm)?,
            sink: up(&sink)?,
            hc_attn_fn: up(&hc_attn_fn)?,
            hc_attn_base: up(&hc_attn_base)?,
            hc_attn_scale: up(&hc_attn_scale)?,
            hc_ffn_fn: up(&hc_ffn_fn)?,
            hc_ffn_base: up(&hc_ffn_base)?,
            hc_ffn_scale: up(&hc_ffn_scale)?,
            gate_w: up(&gate_w)?,
            tid2eid,
            gate_bias,
            moe,
            comp_load,
            idx_load,
        })
    }

    /// Load + upload a DSpark stage (`mtp.{stage}.*`) — a full SWA-kind MoE block (no
    /// compressor/indexer; bias-routed, never hash). Mirrors [`upload_layer`](Self::upload_layer)
    /// but with the `mtp.{stage}.` key prefix and `load_mtp_stage` for the strict load. Returns
    /// the uploaded layer AND the leftover tensor map (carrying the stage extras: main_proj/
    /// main_norm for stage 0; norm/hc_head/markov/confidence for stage 2). DSpark runs
    /// single-process on the head (full 256 experts, world=1) — no TP sharding here.
    pub fn upload_mtp_stage(
        &self,
        bundle: &Path,
        cfg: &dsv4_load::Dsv4Config,
        stage: usize,
    ) -> Result<(Dsv4GpuLayer, std::collections::HashMap<String, dsv4_load::HostTensor>)> {
        let dev = &self.dev;
        let pfx = format!("mtp.{stage}.");
        let fp8 = |key: &str| -> Result<Fp8Weight> {
            let name = format!("{pfx}{key}");
            let (shape, codes, sb) = dsv4_load::read_raw_fp8(bundle, &name)
                .with_context(|| format!("read_raw_fp8 {name}"))?;
            let (m, k) = (shape[0], shape[1]);
            let wt = quant::repack_fp8_mma(&codes, m, k);
            Ok(Fp8Weight {
                wt: dev.htod_sync_copy(&wt).with_context(|| format!("upload {name} wt"))?,
                sb: dev.htod_sync_copy(&sb).with_context(|| format!("upload {name} sb"))?,
                m, k,
            })
        };
        let wq_a = fp8("attn.wq_a.weight")?;
        let wq_b = fp8("attn.wq_b.weight")?;
        let wkv = fp8("attn.wkv.weight")?;
        let wo_b = fp8("attn.wo_b.weight")?;
        // Shared expert fuse [w1; w3].
        let sh_w1 = fp8("ffn.shared_experts.w1.weight")?;
        let sh_w3 = fp8("ffn.shared_experts.w3.weight")?;
        let sh_w2 = fp8("ffn.shared_experts.w2.weight")?;
        let sh_gu = {
            let (m1, m3, k) = (sh_w1.m, sh_w3.m, sh_w1.k);
            anyhow::ensure!(k == sh_w3.k && m1 == m3 && (m1 + m3) % 128 == 0, "dspark shared gu geometry");
            let nm = |base: &str| format!("{pfx}{base}");
            let (_, c1, s1) = dsv4_load::read_raw_fp8(bundle, &nm("ffn.shared_experts.w1.weight"))?;
            let (_, c3, s3) = dsv4_load::read_raw_fp8(bundle, &nm("ffn.shared_experts.w3.weight"))?;
            let mut codes = c1; codes.extend_from_slice(&c3);
            let mut sb = s1; sb.extend_from_slice(&s3);
            let wt = quant::repack_fp8_mma(&codes, m1 + m3, k);
            Fp8Weight { wt: dev.htod_sync_copy(&wt)?, sb: dev.htod_sync_copy(&sb)?, m: m1 + m3, k }
        };
        // Strict load (mtp.{stage}.* — embed/head skipped as tied).
        let layer = dsv4_load::load_mtp_stage(bundle, cfg, stage).context("load_mtp_stage")?;
        let mut map = layer.tensors;
        let wo_a: Vec<bf16> = match map.remove("attn.wo_a.weight") {
            Some(dsv4_load::HostTensor::BF16 { data, shape }) => {
                anyhow::ensure!(shape == vec![cfg.o_groups * cfg.o_lora_rank, cfg.dim], "dspark wo_a shape {shape:?}");
                data
            }
            other => return Err(anyhow!("dspark attn.wo_a.weight: expected BF16, got {:?}", other.map(|t| t.shape().to_vec()))),
        };
        let gate_w_bf: Vec<bf16> = match map.remove("ffn.gate.weight") {
            Some(dsv4_load::HostTensor::BF16 { data, .. }) => data,
            other => return Err(anyhow!("dspark ffn.gate.weight: expected BF16, got {:?}", other.map(|t| t.shape().to_vec()))),
        };
        let gate_w: Vec<f32> = gate_w_bf.iter().map(|v| v.to_f32()).collect();
        // DSpark stages are always bias-routed (gate.bias present, never tid2eid).
        let gate_bias = {
            let b = dsv4_cpu::take_f32(&mut map, "ffn.gate.bias", cfg.n_routed_experts)?;
            Some(dev.htod_sync_copy(&b)?)
        };
        let q_norm = dsv4_cpu::take_f32(&mut map, "attn.q_norm.weight", cfg.q_lora_rank)?;
        let kv_norm = dsv4_cpu::take_f32(&mut map, "attn.kv_norm.weight", cfg.head_dim)?;
        let attn_norm = dsv4_cpu::take_f32(&mut map, "attn_norm.weight", cfg.dim)?;
        let ffn_norm = dsv4_cpu::take_f32(&mut map, "ffn_norm.weight", cfg.dim)?;
        let sink = dsv4_cpu::take_f32(&mut map, "attn.attn_sink", cfg.n_heads)?;
        let hc_attn_fn = dsv4_cpu::take_f32(&mut map, "hc_attn_fn", 24 * cfg.hc_mult * cfg.dim)?;
        let hc_attn_base = dsv4_cpu::take_f32(&mut map, "hc_attn_base", 24)?;
        let hc_attn_scale = dsv4_cpu::take_f32(&mut map, "hc_attn_scale", 3)?;
        let hc_ffn_fn = dsv4_cpu::take_f32(&mut map, "hc_ffn_fn", 24 * cfg.hc_mult * cfg.dim)?;
        let hc_ffn_base = dsv4_cpu::take_f32(&mut map, "hc_ffn_base", 24)?;
        let hc_ffn_scale = dsv4_cpu::take_f32(&mut map, "hc_ffn_scale", 3)?;
        // Pull the DSpark stage extras out of the map BEFORE pack_moe_layer takes it. These
        // are the stage-0 main_proj/main_norm and the stage-2 head/markov/confidence tensors
        // (absent on the other stages — `map.remove` leaves them None there).
        let mut extras: std::collections::HashMap<String, dsv4_load::HostTensor> = std::collections::HashMap::new();
        for k in ["main_norm.weight", "main_proj.weight", "main_proj.scale",
                  "norm.weight", "hc_head_fn", "hc_head_base", "hc_head_scale",
                  "markov_head.markov_w1.weight", "markov_head.markov_w2.weight",
                  "confidence_head.proj.weight"] {
            if let Some(t) = map.remove(k) { extras.insert(k.to_string(), t); }
        }
        let host = dsv4_moe::pack_moe_layer(
            &dsv4_load::Dsv4Layer {
                tensors: std::mem::take(&mut map),
                experts_w1: layer.experts_w1,
                experts_w2: layer.experts_w2,
                experts_w3: layer.experts_w3,
            }, cfg,
        ).context("pack_moe_layer (dspark stage)")?;
        let moe = Dsv4MoeGpu::upload(dev, &host).context("Dsv4MoeGpu::upload (dspark stage)")?;
        let up = |v: &[f32]| -> Result<S> { Ok(dev.htod_sync_copy(v)?) };
        let gl = Dsv4GpuLayer {
            kind: dsv4_load::LayerKind::Swa,
            wq_a, wq_b, wkv, wo_b, sh_gu, sh_w2,
            wo_a: dev.htod_sync_copy(&wo_a)?,
            wo_a_q: Some(quantize_wo_a(dev, &wo_a, cfg.o_groups, cfg.o_lora_rank, cfg.dim)?),
            q_norm: up(&q_norm)?, kv_norm: up(&kv_norm)?,
            attn_norm: up(&attn_norm)?, ffn_norm: up(&ffn_norm)?,
            sink: up(&sink)?,
            hc_attn_fn: up(&hc_attn_fn)?, hc_attn_base: up(&hc_attn_base)?, hc_attn_scale: up(&hc_attn_scale)?,
            hc_ffn_fn: up(&hc_ffn_fn)?, hc_ffn_base: up(&hc_ffn_base)?, hc_ffn_scale: up(&hc_ffn_scale)?,
            gate_w: up(&gate_w)?,
            tid2eid: None, gate_bias, moe,
            comp_load: None, idx_load: None,
        };
        Ok((gl, extras))
    }

    /// Fresh per-layer attention state. SWA: window ring only (3A path). CSA/HCA:
    /// unified `kv_cache[window + max_comp_rows, hd]` (ring ++ compressor-cache tail,
    /// zero-init — unwritten compressor rows are never indexed), prefill scratch
    /// `[max_s + max_comp_rows, hd]`, and the attention compressor (+ CSA indexer)
    /// constructed from the layer's load weights.
    ///
    /// `max_seq_len` sizes the compressor caches (the oracle's `kv_cache.shape[0]`
    /// gives the right value per piece: e.g. CSA@8K → max_seq=16384 → cache 128+4096).
    /// `s_max` sizes the compressor GEMM scratch (the largest prefill chunk).
    #[allow(clippy::too_many_arguments)]
    pub fn new_state(
        &self,
        cfg: &dsv4_load::Dsv4Config,
        layer: &Dsv4GpuLayer,
        max_seq_len: usize,
        s_max: usize,
    ) -> Result<Dsv4AttnState> {
        let dev = &self.dev;
        let win = cfg.window_size;
        let hd = cfg.head_dim;
        let comp_ks = self.comp.as_ref().expect("CSA/HCA state needs comp kernels (runtime built without)");
        let comp_load = layer.comp_load.as_ref().expect("CSA/HCA layer has no comp_load");
        let max_comp_rows = match layer.kind {
            dsv4_load::LayerKind::Swa => 0,
            dsv4_load::LayerKind::Csa => max_seq_len / 4,
            dsv4_load::LayerKind::Hca => max_seq_len / 128,
        };
        // The kind's compressor ratio (SWA never reaches the scratch sizing below).
        let comp_ratio = match layer.kind {
            dsv4_load::LayerKind::Swa => 1,
            dsv4_load::LayerKind::Csa => 4,
            dsv4_load::LayerKind::Hca => 128,
        };
        // Unified kv_cache: [window + max_comp_rows, hd], zero-init (AGENTS §2.2 —
        // alloc_zeros doesn't zero, so explicit htod of a zero vec).
        let kv_rows = win + max_comp_rows;
        let zeros = vec![bf16::from_f32(0.0); kv_rows * hd];
        let kv_cache = dev.htod_sync_copy(&zeros).context("kv_cache alloc")?;

        // Prefill scratch: R2.1 resized for the batched-continuation path. The start_pos==0
        // path uses [s + nb_committed] rows; the R2.1 continuation path (start_pos>0) uses
        // [win(prefix) + s(new) + nb_committed(comp)] rows. The worst case (last chunk at
        // max_seq_len) needs win + s_max + max_comp_rows. The delta vs the old chunk-only
        // sizing is win + max_comp_rows rows/layer — ~50 MB at SEQ=8192, ~320 MB at 32K
        // (the 1M case is R5's budget concern; the comp_cache part is the max_comp_rows tail
        // the old comment warned about — but the batched attention needs it in the scratch
        // for a single-buffer gather, and the sequential alternative is ~36% slower per chunk).
        let scratch_rows = win + s_max + max_comp_rows;
        let zeros_sc = vec![bf16::from_f32(0.0); scratch_rows * hd];
        let kv_attn_scratch = Some(dev.htod_sync_copy(&zeros_sc).context("kv_attn_scratch alloc")?);

        // Attention compressor: owns its cache (3B) + frontier state. The compressor's
        // cache is ALIASED to kv_cache[win..] (Item 3: the epilogue writes directly to the
        // kv_cache tail, eliminating the per-step d2d mirror and recovering ~6 GB at 1M).
        // The gather reads from the unified kv_cache (ring + compressor-cache tail) contiguously.
        let mut attn_compressor = GpuCompressor::new(
            dev, comp_ks, &self.stream, comp_load.spec, &comp_load.w,
            cfg.norm_eps, max_comp_rows, s_max,
        ).context("GpuCompressor::new (attention)")?;
        let tail_dptr = *kv_cache.device_ptr() + (win * hd * 2) as u64;
        attn_compressor.set_cache_alias(tail_dptr);
        // Reclaim the full-size cache allocation — the alias replaces it. Keep a 1-element
        // dummy so the CudaSlice field is valid (snapshot/restore size from cache_rows).
        attn_compressor.cache = dev.alloc_zeros::<bf16>(1)?;
        let attn_compressor = Some(attn_compressor);

        // CSA indexer: own rotate compressor + indexer kv_cache [max_seq/4, 128].
        let indexer = if let Some(idx_load) = layer.idx_load.as_ref() {
            Some(GpuIndexer::new(
                dev, comp_ks, &self.stream, cfg.dim, cfg.rope_head_dim, cfg.q_lora_rank,
                cfg.index_n_heads, cfg.index_head_dim, cfg.index_topk,
                &idx_load.comp.w, cfg.norm_eps, &idx_load.wq_b_wt, &idx_load.wq_b_sb,
                &idx_load.weights_proj, max_seq_len, s_max,
            ).context("GpuIndexer::new")?)
        } else {
            None
        };

        Ok(Dsv4AttnState { kv_cache, kv_attn_scratch, attn_compressor, indexer })
    }

    /// Fresh SWA ring state (window × 512 bf16, explicitly zeroed — AGENTS §2.2).
    /// R2.1: SWA continuation chunks need a unified scratch [win + s_max] for the
    /// batched gather (prefix window + new kv). The start_pos==0 path gathers from
    /// `kv` directly (no scratch); the continuation path gathers from the scratch.
    pub fn new_state_swa(&self, cfg: &dsv4_load::Dsv4Config, s_max: usize) -> Result<Dsv4AttnState> {
        let win = cfg.window_size;
        let hd = cfg.head_dim;
        let zeros = vec![bf16::from_f32(0.0); win * hd];
        // R2.1: [win + s_max, hd] — prefix window + one chunk of new kv.
        let scratch_rows = win + s_max;
        let zeros_sc = vec![bf16::from_f32(0.0); scratch_rows * hd];
        Ok(Dsv4AttnState {
            kv_cache: self.dev.htod_sync_copy(&zeros)?,
            kv_attn_scratch: Some(self.dev.htod_sync_copy(&zeros_sc)?),
            attn_compressor: None,
            indexer: None,
        })
    }

    // ---------------------------------------------------------------------------------------------
    // Primitive wrappers (all launches on the blocking compute stream unless noted)
    // ---------------------------------------------------------------------------------------------

}
