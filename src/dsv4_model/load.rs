//! Loaders of `Dsv4GpuModel`: trunk-top only, layers from the strict loader or a converted artifact,
//! `attach_tp` (TP=2 link + proxy start) and the trunk-top tensor extraction.
//! Siblings: `mod.rs` has the struct and the directory map; `forward.rs` runs what is loaded here.

use super::*;

impl Dsv4GpuModel {
    /// Load the trunk-top weights only (embed/norm/head/hc_head). Used by the head-only gate
    /// (`forward_head`) — no layers, no attention state, minimal memory. `load_trunk_top` is the
    /// strict loader; we extract + upload the four tensors here (mirrors `dsv4_cpu::trunk_top_from`).
    pub fn load_trunk_top(dev: &Arc<CudaDevice>, bundle: &Path, cfg: &Dsv4Config) -> Result<Self> {
        let rt = Dsv4AttnRuntime::new_multikind(dev, 16, cfg).context("Dsv4AttnRuntime::new_multikind")?;
        let top = dsv4_load::load_trunk_top(bundle, cfg).context("load_trunk_top")?;
        let (embed, norm, head, hc_head_fn, hc_head_base, hc_head_scale) = extract_trunk_top(dev, top, cfg)?;
        Ok(Self {
            dev: dev.clone(),
            cfg: cfg.clone(),
            rt,
            layers: Vec::new(),
            states: Vec::new(),
            scratch: gpu::new_moe_grouped_scratch_raw(dev, cfg.n_routed_experts, cfg.dim, cfg.moe_inter_dim, cfg.n_activated_experts, 16, 16 * cfg.n_activated_experts),
            embed,
            norm,
            head,
            hc_head_fn,
            hc_head_base,
            hc_head_scale,
            tp_rank: -1,
            x_cap: None,
            graphs: None,
        })
    }

    /// Load the full (or sliced: `n_layers <= cfg.n_layers`) trunk: trunk top + layers 0..n_layers-1
    /// + per-layer attention state. `max_seq_len` sizes the compressor/indexer caches (the context
    /// budget); `s_max` sizes the prefill GEMM scratch (the largest prefill chunk).
    pub fn load(
        dev: &Arc<CudaDevice>,
        bundle: &Path,
        cfg: &Dsv4Config,
        max_seq_len: usize,
        s_max: usize,
        n_layers: usize,
    ) -> Result<Self> {
        Self::load_tp(dev, bundle, cfg, max_seq_len, s_max, n_layers, 0, 1)
    }

    /// TP=2 shard-at-load constructor (§5): identical to [`load`](Self::load) but each rank uploads
    /// only its `ne/world`-expert slice (the rest of the layer — attn/mHC/router/shared/norms — is
    /// replicated). The all-reduce boundary (routed combine BEFORE the shared add) is driven by the
    /// caller via `Dsv4AttnRuntime::block_forward_tp_*`. `rank`/`world` ∈ {0,1}/{1,2} for TP=2.
    pub fn load_tp(
        dev: &Arc<CudaDevice>,
        bundle: &Path,
        cfg: &Dsv4Config,
        max_seq_len: usize,
        s_max: usize,
        n_layers: usize,
        rank: usize,
        world: usize,
    ) -> Result<Self> {
        anyhow::ensure!(n_layers <= cfg.n_layers, "n_layers {n_layers} > cfg.n_layers {}", cfg.n_layers);
        let positions = max_seq_len.max(64) + 8;
        let rt = Dsv4AttnRuntime::new_multikind(dev, positions, cfg).context("Dsv4AttnRuntime::new_multikind")?;
        let top = dsv4_load::load_trunk_top(bundle, cfg).context("load_trunk_top")?;
        let (embed, norm, head, hc_head_fn, hc_head_base, hc_head_scale) = extract_trunk_top(dev, top, cfg)?;

        eprintln!(
            "[dsv4] loading trunk: {n_layers} layer(s), max_seq_len={max_seq_len}, s_max={s_max} (rank {rank}/{world}) ..."
        );
        let mut layers = Vec::with_capacity(n_layers);
        let mut states = Vec::with_capacity(n_layers);
        // BASICS (owner 2026-11-02): a percentage at the normal level next to the per-layer lines
        let load_prog = crate::boot_report::Progress::new("loading weights", n_layers as u64,
                                                          crate::boot_report::Unit::Items("layers"));
        for layer_id in 0..n_layers {
            let kind = cfg.layer_kind(layer_id);
            let layer = rt
                .upload_layer(bundle, cfg, layer_id, rank, world)
                .with_context(|| format!("upload_layer {layer_id}"))?;
            let st = match kind {
                dsv4_load::LayerKind::Swa => rt.new_state_swa(cfg, s_max).with_context(|| format!("new_state_swa {layer_id}"))?,
                _ => rt
                    .new_state(cfg, &layer, max_seq_len, s_max)
                    .with_context(|| format!("new_state {layer_id}"))?,
            };
            layers.push(layer);
            states.push(st);
            eprintln!("[dsv4]   layer {layer_id:2} ({kind:?}) loaded");
            load_prog.add(1);
        }
        load_prog.finish();
        Ok(Self {
            dev: dev.clone(),
            cfg: cfg.clone(),
            rt,
            layers,
            states,
            scratch: gpu::new_moe_grouped_scratch_raw(dev, cfg.n_routed_experts, cfg.dim, cfg.moe_inter_dim, cfg.n_activated_experts, s_max, s_max.min(gpu::MOE_D_CAP) * cfg.n_activated_experts),
            embed,
            norm,
            head,
            hc_head_fn,
            hc_head_base,
            hc_head_scale,
            tp_rank: -1,
            x_cap: None,
            graphs: None,
        })
    }

    /// Build one `Dsv4GpuLayer` from a converted artifact file (the inverse of `prepare_layer`).
    /// Pure read + `htod` — no cast/repack/fuse (those ran offline). Bitwise-identical to
    /// `upload_layer` because the artifact carries the same bytes (A/B-gated).
    fn layer_from_artifact(
        dev: &Arc<CudaDevice>,
        art: &ArtFile,
        cfg: &Dsv4Config,
        layer_id: usize,
        rank: usize,
        world: usize,
        moe_band: Option<(usize, usize)>,
    ) -> Result<Dsv4GpuLayer> {
        let fp8w = |name: &str| -> Result<Fp8Weight> {
            let (m, k) = art.mk_of(&format!("{name}.wt"))?;
            Ok(Fp8Weight {
                wt: dev.htod_sync_copy(art.u8_slice(&format!("{name}.wt"))?)?,
                sb: dev.htod_sync_copy(art.u8_slice(&format!("{name}.sb"))?)?,
                m,
                k,
            })
        };
        let kind = cfg.layer_kind(layer_id);
        let (hd, ihd, inh) = (cfg.head_dim, cfg.index_head_dim, cfg.index_n_heads);
        // Attention compressor + indexer (CSA/HCA) — reconstruct CompLoad/IndexerLoad from cfg
        // geometry (same spec constructors upload_layer uses) + the artifact's f32 weights.
        let comp_load = match kind {
            dsv4_load::LayerKind::Swa => None,
            dsv4_load::LayerKind::Csa => Some(CompLoad {
                spec: CompSpec::csa_attn(cfg.dim, cfg.rope_head_dim),
                w: CompressorWeights {
                    wkv: art.f32_of("comp.wkv")?, wgate: art.f32_of("comp.wgate")?,
                    norm: art.f32_of("comp.norm")?, ape: art.f32_of("comp.ape")?,
                    ratio: 4, head_dim: hd, rope_dim: cfg.rope_head_dim,
                    overlap: true, rotate: false, sim_group: 64, dim: cfg.dim,
                },
            }),
            dsv4_load::LayerKind::Hca => Some(CompLoad {
                spec: CompSpec::hca_attn(cfg.dim, cfg.rope_head_dim),
                w: CompressorWeights {
                    wkv: art.f32_of("comp.wkv")?, wgate: art.f32_of("comp.wgate")?,
                    norm: art.f32_of("comp.norm")?, ape: art.f32_of("comp.ape")?,
                    ratio: 128, head_dim: hd, rope_dim: cfg.rope_head_dim,
                    overlap: false, rotate: false, sim_group: 64, dim: cfg.dim,
                },
            }),
        };
        let idx_load = if matches!(kind, dsv4_load::LayerKind::Csa) {
            Some(IndexerLoad {
                comp: CompLoad {
                    spec: CompSpec::indexer(cfg.dim, cfg.rope_head_dim),
                    w: CompressorWeights {
                        wkv: art.f32_of("idx.comp.wkv")?, wgate: art.f32_of("idx.comp.wgate")?,
                        norm: art.f32_of("idx.comp.norm")?, ape: art.f32_of("idx.comp.ape")?,
                        ratio: 4, head_dim: ihd, rope_dim: cfg.rope_head_dim,
                        overlap: true, rotate: true, sim_group: 32, dim: cfg.dim,
                    },
                },
                wq_b_wt: art.u8_slice("idx.wq_b.wt")?.to_vec(),
                wq_b_sb: art.u8_slice("idx.wq_b.sb")?.to_vec(),
                weights_proj: art.f32_of("idx.weights_proj")?,
            })
        } else {
            None
        };
        // MoE. Three modes:
        //  (a) `moe_band = Some(e_base, e_span)` — PER-RANK artifact: the moe bytes ARE this band's
        //      slice (the converter pre-sliced it). Upload direct, no re-slice. Each node loads only
        //      its ~84 GB part — the load-speed lane's TP=2 design.
        //  (b) flat artifact, world>1 — full moe on disk; re-slice per rank via upload_sharded.
        //  (c) flat artifact, world==1 — full moe, e_base=0/e_span=ne.
        let (ne, hdim, inter) = (cfg.n_routed_experts, cfg.dim, cfg.moe_inter_dim);
        let moe = if let Some((e_base, e_span)) = moe_band {
            Dsv4MoeGpu {
                gu_wt: dev.htod_sync_copy(art.u8_slice("moe.gu_wt")?)?,
                gu_st: dev.htod_sync_copy(art.u8_slice("moe.gu_st")?)?,
                gu_gs: dev.htod_sync_copy(&art.f32_of("moe.gu_gs")?)?,
                dn_wt: dev.htod_sync_copy(art.u8_slice("moe.dn_wt")?)?,
                dn_st: dev.htod_sync_copy(art.u8_slice("moe.dn_st")?)?,
                dn_gs: dev.htod_sync_copy(&art.f32_of("moe.dn_gs")?)?,
                ne, h: hdim, inter, e_base, e_span,
            }
        } else if world > 1 {
            let host = Dsv4MoeHost {
                gu_wt: art.u8_slice("moe.gu_wt")?.to_vec(), gu_st: art.u8_slice("moe.gu_st")?.to_vec(),
                gu_gs: art.f32_of("moe.gu_gs")?,
                dn_wt: art.u8_slice("moe.dn_wt")?.to_vec(), dn_st: art.u8_slice("moe.dn_st")?.to_vec(),
                dn_gs: art.f32_of("moe.dn_gs")?,
                ne, h: hdim, inter,
            };
            let e_span = ne / world;
            Dsv4MoeGpu::upload_sharded(dev, &host, rank * e_span, e_span)?
        } else {
            Dsv4MoeGpu {
                gu_wt: dev.htod_sync_copy(art.u8_slice("moe.gu_wt")?)?,
                gu_st: dev.htod_sync_copy(art.u8_slice("moe.gu_st")?)?,
                gu_gs: dev.htod_sync_copy(&art.f32_of("moe.gu_gs")?)?,
                dn_wt: dev.htod_sync_copy(art.u8_slice("moe.dn_wt")?)?,
                dn_st: dev.htod_sync_copy(art.u8_slice("moe.dn_st")?)?,
                dn_gs: dev.htod_sync_copy(&art.f32_of("moe.dn_gs")?)?,
                ne, h: hdim, inter, e_base: 0, e_span: ne,
            }
        };
        let up = |v: Vec<f32>| -> Result<S> { Ok(dev.htod_sync_copy(&v)?) };
        let tid2eid = if cfg.is_hash_layer(layer_id) {
            Some(dev.htod_sync_copy(&art.i32_of("tid2eid")?)?)
        } else { None };
        let gate_bias = if cfg.is_hash_layer(layer_id) { None } else { Some(up(art.f32_of("gate_bias")?)?) };
        let wo_a_bf = art.bf16_of("wo_a")?;
        let wo_a_q = {
            let (wt, sb) = crate::quant::quantize_fp8_bsb(&wo_a_bf, cfg.o_groups * cfg.o_lora_rank, cfg.dim);
            crate::dsv4_attn::Fp8Weight {
                wt: dev.htod_sync_copy(&wt)?,
                sb: dev.htod_sync_copy(&sb)?,
                m: cfg.o_groups * cfg.o_lora_rank,
                k: cfg.dim,
            }
        };
        Ok(Dsv4GpuLayer {
            kind,
            wq_a: fp8w("wq_a")?, wq_b: fp8w("wq_b")?, wkv: fp8w("wkv")?, wo_b: fp8w("wo_b")?,
            sh_gu: fp8w("sh_gu")?, sh_w2: fp8w("sh_w2")?,
            wo_a: dev.htod_sync_copy(&wo_a_bf)?,
            wo_a_q: Some(wo_a_q),
            q_norm: up(art.f32_of("q_norm")?)?, kv_norm: up(art.f32_of("kv_norm")?)?,
            attn_norm: up(art.f32_of("attn_norm")?)?, ffn_norm: up(art.f32_of("ffn_norm")?)?,
            sink: up(art.f32_of("attn_sink")?)?,
            hc_attn_fn: up(art.f32_of("hc_attn_fn")?)?, hc_attn_base: up(art.f32_of("hc_attn_base")?)?,
            hc_attn_scale: up(art.f32_of("hc_attn_scale")?)?,
            hc_ffn_fn: up(art.f32_of("hc_ffn_fn")?)?, hc_ffn_base: up(art.f32_of("hc_ffn_base")?)?,
            hc_ffn_scale: up(art.f32_of("hc_ffn_scale")?)?,
            gate_w: up(art.f32_of("gate_w")?)?,
            tid2eid, gate_bias, moe, comp_load, idx_load,
        })
    }

    /// FAST load from a converted artifact dir (the load-speed lane's reader). Reads `manifest.json`
    /// + one safetensors per layer + trunk_top, uploads directly (no cast/repack/fuse — that ran
    /// offline at convert time). Bitwise-identical to [`load`](Self::load) by the A/B gate
    /// (`tests/dsv4_convert_test.rs`): the artifact carries the same bytes the streaming path
    /// uploads, so this is `htod` of the same bytes. `rank`/`world` TP-slice the MoE bank at load.
    pub fn load_converted(
        dev: &Arc<CudaDevice>,
        artifact_dir: &Path,
        cfg: &Dsv4Config,
        max_seq_len: usize,
        s_max: usize,
        n_layers: usize,
        rank: usize,
        world: usize,
    ) -> Result<Self> {
        anyhow::ensure!(n_layers <= cfg.n_layers, "n_layers {n_layers} > cfg.n_layers {}", cfg.n_layers);
        // Per-rank shard: if artifact_dir/rank{rank}/ exists, load ONLY that rank's self-contained
        // shard (the converter pre-sliced its experts). Each node reads + loads ~84 GB, not 156 GB.
        let rank_subdir = artifact_dir.join(format!("rank{rank}"));
        let (load_dir, moe_band): (std::path::PathBuf, Option<(usize, usize)>) = if rank_subdir.exists() {
            let e_span = cfg.n_routed_experts / world;
            eprintln!("[dsv4] per-rank shard detected: rank{rank}/ (e[{}..{}) — loading only this node's part",
                rank * e_span, (rank + 1) * e_span);
            (rank_subdir, Some((rank * e_span, e_span)))
        } else {
            (artifact_dir.to_path_buf(), None)
        };
        let manifest = crate::dsv4_convert::read_manifest(&load_dir)
            .map_err(|e| anyhow::anyhow!("read_manifest {}: {e:#}", load_dir.display()))?;
        anyhow::ensure!(manifest.model_type == "deepseek_v4" && manifest.layout_version == 1,
            "artifact: unsupported model_type/layout_version {:?}/{}", manifest.model_type, manifest.layout_version);
        let positions = max_seq_len.max(64) + 8;
        let rt = Dsv4AttnRuntime::new_multikind(dev, positions, cfg)?;
        // Trunk top.
        let top = ArtFile::open(&load_dir.join("trunk_top.safetensors"))?;
        let embed = dev.htod_sync_copy(&top.bf16_of("embed")?)?;
        let norm = dev.htod_sync_copy(&top.f32_of("norm")?)?;
        let head = dev.htod_sync_copy(&top.bf16_of("head")?)?;
        let hc_head_fn = dev.htod_sync_copy(&top.f32_of("hc_head_fn")?)?;
        let hc_head_base = dev.htod_sync_copy(&top.f32_of("hc_head_base")?)?;
        let hc_head_scale = dev.htod_sync_copy(&top.f32_of("hc_head_scale")?)?;
        eprintln!("[dsv4] converted-load: {n_layers} layer(s), max_seq_len={max_seq_len}, s_max={s_max} (rank {rank}/{world}) ...");
        let mut layers = Vec::with_capacity(n_layers);
        let mut states = Vec::with_capacity(n_layers);
        // BASICS (owner 2026-11-02): a percentage at the normal level next to the per-layer lines
        let load_prog = crate::boot_report::Progress::new("loading weights", n_layers as u64,
                                                          crate::boot_report::Unit::Items("layers"));
        for layer_id in 0..n_layers {
            let kind = cfg.layer_kind(layer_id);
            let art = ArtFile::open(&load_dir.join(format!("layer{layer_id}.safetensors")))
                .map_err(|e| anyhow::anyhow!("ArtFile::open layer{layer_id}: {e:#}"))?;
            let layer = Self::layer_from_artifact(dev, &art, cfg, layer_id, rank, world, moe_band)?;
            let st = match kind {
                dsv4_load::LayerKind::Swa => rt.new_state_swa(cfg, s_max)?,
                _ => rt.new_state(cfg, &layer, max_seq_len, s_max)?,
            };
            layers.push(layer);
            states.push(st);
            eprintln!("[dsv4]   layer {layer_id:2} ({kind:?}) loaded (converted)");
            load_prog.add(1);
        }
        load_prog.finish();
        Ok(Self {
            dev: dev.clone(), cfg: cfg.clone(), rt, layers, states,
            scratch: gpu::new_moe_grouped_scratch_raw(dev, cfg.n_routed_experts, cfg.dim, cfg.moe_inter_dim, cfg.n_activated_experts, s_max, s_max.min(gpu::MOE_D_CAP) * cfg.n_activated_experts),
            embed, norm, head, hc_head_fn, hc_head_base, hc_head_scale, tp_rank: -1, x_cap: None, graphs: None,
        })
    }

    /// Attach the TP=2 doorbell link (call AFTER `load_tp`, before the first forward). Mirrors
    /// `gpu.rs::attach_tp` (2652): set the doorbell payload to the decode all-reduce size
    /// (`dim·2` B bf16 — the routed-expert partial at s=1; verify/prefill pass per-call nbytes),
    /// hand the RDMA ctx device pointer to the runtime (so `block_forward`'s TP path can launch the
    /// `tp_gate_copy_signal`/`tp_wait_add` handshake), start the persistent proxy thread, and
    /// `mem::forget` the link (the proxy owns the transport from here; its Drop would shut the ctx).
    pub fn attach_tp(&mut self, rank: i32, world: i32, mut link: crate::net::TpLink) {
        assert_eq!(world, 2, "Dsv4GpuModel::attach_tp: only TP=2 implemented");
        let nbytes = self.cfg.dim * 2; // bf16 routed partial at s=1 (decode)
        assert!(nbytes <= crate::tp::TP_SLOT_BYTES,
            "FATAL: DSV4 TP all-reduce payload {nbytes} B > TP_SLOT_BYTES ({}) B", crate::tp::TP_SLOT_BYTES);
        link.set_payload(nbytes, false).expect("net_set_payload");
        if crate::opts::var(crate::opt!("tp-trace")).is_ok()
            || crate::tp::tp_config().map(|c| c.trace).unwrap_or(false) {
            crate::net::trace_enable(&mut link);
            eprintln!("[dsv4-tp] per-barrier tracing ON (--tp-trace)");
        }
        self.tp_rank = rank;
        self.rt.tp_ctx_dptr = link.ctx_device_ptr();
        let ctx_addr = link.ctx_addr();
        std::mem::forget(link);
        crate::net::spawn_proxy(ctx_addr, 19);
        eprintln!("[dsv4-tp] rank {rank}/{world} — RDMA proxy up (doorbell all-reduce on the routed partial, {nbytes} B/decode-ring)");
    }

}
/// Extract + upload the trunk-top tensors from the strict loader's map (mirrors
/// `dsv4_cpu::trunk_top_from`). `head.weight` ships as F32 (cast rule) but the values are
/// bf16-exact → we downcast to bf16 for `gemm_binv_f32_b` (bit-identical to the reference's
/// fp32 param read, §12.A.4).
fn extract_trunk_top(
    dev: &Arc<CudaDevice>,
    mut map: std::collections::HashMap<String, HostTensor>,
    cfg: &Dsv4Config,
) -> Result<(B, S, B, S, S, S)> {
    let (vocab, dim, hc) = (cfg.vocab_size, cfg.dim, cfg.hc_mult);
    // embed.weight [vocab, dim] BF16.
    let embed_vec: Vec<bf16> = match map.remove("embed.weight") {
        Some(HostTensor::BF16 { data, shape }) => {
            anyhow::ensure!(shape == vec![vocab, dim], "embed.weight shape {shape:?}");
            data
        }
        other => return Err(anyhow!("embed.weight: expected BF16, got {:?}", other.map(|t| t.shape().to_vec()))),
    };
    let embed = dev.htod_sync_copy(&embed_vec).context("upload embed")?;
    // norm.weight [dim] F32 (bf16→f32 upcast at load).
    let norm_vec = dsv4_cpu::take_f32(&mut map, "norm.weight", dim)?;
    let norm = dev.htod_sync_copy(&norm_vec)?;
    // head.weight [vocab, dim] F32 (bf16-exact values) → downcast to bf16 for the GEMM.
    let head_f32 = dsv4_cpu::take_f32(&mut map, "head.weight", vocab * dim)?;
    let head_bf16: Vec<bf16> = head_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let head = dev.htod_sync_copy(&head_bf16).context("upload head")?;
    // hc_head_* (F32). hc_head_fn is [hc, hc*dim]; base [hc]; scale [1].
    let hc_head_fn = dev.htod_sync_copy(&dsv4_cpu::take_f32(&mut map, "hc_head_fn", hc * hc * dim)?)?;
    let hc_head_base = dev.htod_sync_copy(&dsv4_cpu::take_f32(&mut map, "hc_head_base", hc)?)?;
    let hc_head_scale = dev.htod_sync_copy(&dsv4_cpu::take_f32(&mut map, "hc_head_scale", 1)?)?;
    Ok((embed, norm, head, hc_head_fn, hc_head_base, hc_head_scale))
}
