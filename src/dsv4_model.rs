//! Dsv4GpuModel — the full DeepSeek-V4-Flash-DSpark trunk on GPU (G3 integration, §6 step 2).
//!
//! Wraps the lane-3-proven `Dsv4AttnRuntime` (SWA/CSA/HCA kind-generic — one runtime dispatches
//! all 43 layers) into the complete trunk:
//!   `embed → [layer 0..n-1: hc_pre → norm → attn → hc_post → hc_pre → norm → MoE → hc_post]
//!    → hc_head (sigmoid-only collapse) → final RMSNorm → LM head (bf16→fp32 logits)`
//! and exposes a per-step forward returning the last-row fp32 logits (the greedy target).
//!
//! DSpark stages (§B.10) are SKIPPED here — Phase 5. TP=2 sharding (§5) lands next; this is the
//! single-process correctness path (a slice of N layers fits one node; the full 43-layer / 167 GB
//! trunk needs TP=2 and is gated on the shard-at-load constructor, added in a follow-up commit).
//!
//! ## The two G3 gates this model backs
//! - **Head gate** (`forward_head`): load trunk-top only, run hc_head+norm+lm_head on the oracle's
//!   `dsv4_head.npz` `x` [s,4,4096], diff `collapsed`/`logits` vs the npz. Validates the NEW
//!   trunk-top code (`dsv4_hc_head_b` + `gemm_binv_f32_b` LM head) in isolation.
//! - **Trunk gate** (`forward`): embed → N layers → head on a short prompt, diff vs `dsv4_cpu`'s
//!   full-trunk chain. Proves the integration glue (embed, kind dispatch, position/state tracking,
//!   stream replication) before TP=2 sharding.
//!
//! Stream invariant (AGENTS §2.1): every launch rides `rt.stream` (the runtime's blocking compute
//! stream). The G2 MoE runtime still sync-brackets default-stream launches — promoted to the
//! blocking stream in the next commit (G3.3).

use anyhow::{anyhow, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DeviceSlice};
use half::bf16;
use std::path::Path;
use std::sync::Arc;

use crate::dsv4_attn::{
    CompLoad, Dsv4AttnRuntime, Dsv4AttnState, Dsv4GpuLayer, Fp8Weight, IndexerLoad, B,
};
pub use crate::dsv4_attn::LayerVerifySnap;
use crate::dsv4_comp::CompSpec;
use crate::dsv4_convert::ArtFile;
use crate::dsv4_cpu::{self, CompressorWeights};
use crate::dsv4_gpu::S;
use crate::dsv4_load::{self, Dsv4Config, HostTensor};
use crate::dsv4_moe::Dsv4MoeHost;
use crate::gpu::{self, Dsv4MoeGpu, MoeGroupedScratch};

/// Prefill chunk size (§12.B.5 — multiple of 128 so chunk boundaries land on compressor-block
/// boundaries; also a multiple of the grouped-MoE N=16 tile). Prompts longer than this are
/// processed in chunks by [`Dsv4GpuModel::forward`], carrying the compressor frontier
/// (kv_state/score_state), indexer cache tail, and window ring across chunks in `states` —
/// bitwise-identical to one-shot prefill (the 3B compressor gate + the model-level
/// `--probe-dsv4 --chunked` gate). Serving caps `s_max` at this value, so ALL prefill
/// scratch is chunk-sized, not prompt-sized (the 200K OOM fix — DSV4_LONG_CONTEXT_1M §3).
pub const PREFILL_CHUNK: usize = 4096;

/// Full DSV4 trunk model (single-process; TP=2 lands in a follow-up).
pub struct Dsv4GpuModel {
    pub dev: Arc<CudaDevice>,
    pub cfg: Dsv4Config,
    pub rt: Dsv4AttnRuntime,
    /// Trunk layers 0..n_layers-1 (a slice for the probe; the full trunk is n_layers = cfg.n_layers).
    pub layers: Vec<Dsv4GpuLayer>,
    /// Per-layer attention state (KV ring + compressor/indexer caches), parallel to `layers`.
    pub states: Vec<Dsv4AttnState>,
    /// G2 grouped-MoE scratch (E16 metadata pipeline). p_max = 16 * topk (the ≤16-row chunk cap).
    pub scratch: MoeGroupedScratch,
    // ---- trunk top (§A.1) ----
    /// `embed.weight` [vocab, dim] bf16 (`ParallelEmbedding`).
    pub embed: B,
    /// `norm.weight` [dim] f32 — the final trunk RMSNorm (before hc_head? no — see forward_head).
    pub norm: S,
    /// `head.weight` [vocab, dim] bf16 (fp32-valued; the GEMM reads bf16 → fp32 logits, bit-identical
    /// to the reference's fp32 param read at half the bytes — §12.A.4).
    pub head: B,
    /// `hc_head_fn` [hc, hc*dim] = [4, 16384] f32 (final sigmoid-only collapse, §B.8).
    pub hc_head_fn: S,
    /// `hc_head_base` [hc] = [4] f32.
    pub hc_head_base: S,
    /// `hc_head_scale` [1] f32.
    pub hc_head_scale: S,
    /// TP=2 rank (set by attach_tp; -1 = single-process). Drives the R3.2 vocab-parallel
    /// maxloc head (`forward_next`): rank r computes logits rows [r*vocab/2, (r+1)*vocab/2).
    pub tp_rank: i32,
    /// DSpark selective rollback (R4): per-layer verify activations [layers][verify_width, dim],
    /// captured during `forward_verify_capture_main` so a rejection re-applies compressor and
    /// KV state for the committed prefix WITHOUT a full re-forward (ring writes from the
    /// verify are valid for committed positions; only compressor/indexer state and the
    /// committed ring rows need re-application). Lazily allocated on first verify.
    pub x_cap: Option<Vec<B>>,
    /// CUDA-graph decode (R3A.1/E2, `--graph=1`): per-fire-variant whole-forward graphs,
    /// lazily captured on the first decode token of each variant. None = eager (default).
    pub graphs: Option<crate::dsv4_graph::DecodeGraphs>,
}

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
        }
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

    /// Trunk-top forward: `hc_head` (sigmoid-only collapse) → final RMSNorm → LM head on the LAST
    /// row. `x` [s, hc*dim] bf16 streams → (collapsed [s, dim] bf16, logits [vocab] f32). Mirrors
    /// `dsv4_cpu::run_head_piece` exactly (§B.8 hc_head + §A.1 ParallelHead, last-position-sliced).
    pub fn forward_head<X: crate::dsv4_gpu::Dsv4Buf<bf16>, F: crate::dsv4_gpu::Dsv4Buf<f32>>(&self, x: &X, s: usize) -> Result<(X, F)> {
        use cudarc::driver::result;
        let (dim, vocab) = (self.cfg.dim, self.cfg.vocab_size);
        let collapsed = self
            .rt
            .hc_head(x, &self.hc_head_fn, &self.hc_head_base, &self.hc_head_scale, s, &self.cfg)
            .context("hc_head")?;
        let yn = self
            .rt
            .rmsnorm(&collapsed, &self.norm, s, dim, self.cfg.norm_eps)
            .context("final rmsnorm")?;
        // LM head on the LAST row only (the reference slices yn[-1:] — §A.1 last-position). d2d
        // the last row into a fresh dim-wide buffer (8 KB — noise on the blocking stream, no
        // host round-trip).
        let last = X::alloc_zeros(&self.dev, self.rt.stream.stream, dim)?;
        unsafe {
            result::memcpy_dtod_async(
                last.dptr(),
                yn.dptr() + ((s - 1) * dim * 2) as u64,
                dim * 2,
                self.rt.stream.stream,
            )
            .map_err(|e| anyhow!("forward_head last-row d2d: {e}"))?;
        }
        let logits = self.rt.lm_head(&self.head, &last, 1, dim, vocab).context("lm_head")?;
        Ok((collapsed, logits))
    }

    /// R3.2 (L2): trunk top → next token via the vocab-parallel maxloc head. Returns `None`
    /// when TP is not attached (single-process — the caller uses the full-logits path).
    /// Bitwise: the winner equals the full-vocab host argmax exactly (see `lm_head_maxloc_tp`).
    pub fn forward_head_next(&self, x: &B, s: usize) -> Result<Option<u32>> {
        if self.rt.tp_ctx_dptr == 0 || self.tp_rank < 0 {
            return Ok(None);
        }
        use cudarc::driver::result;
        let (dim, vocab) = (self.cfg.dim, self.cfg.vocab_size);
        let collapsed = self
            .rt
            .hc_head(x, &self.hc_head_fn, &self.hc_head_base, &self.hc_head_scale, s, &self.cfg)
            .context("hc_head")?;
        let yn = self
            .rt
            .rmsnorm(&collapsed, &self.norm, s, dim, self.cfg.norm_eps)
            .context("final rmsnorm")?;
        let mut last = self.dev.alloc_zeros::<bf16>(dim)?;
        unsafe {
            result::memcpy_dtod_async(
                *last.device_ptr(),
                *yn.device_ptr() + ((s - 1) * dim * 2) as u64,
                dim * 2,
                self.rt.stream.stream,
            )
            .map_err(|e| anyhow!("forward_head_next last-row d2d: {e}"))?;
        }
        Ok(Some(self.rt.lm_head_maxloc_tp(&self.head, &last, dim, vocab, self.tp_rank as usize)?))
    }

    /// Trunk forward + next token in one call (the R3.2 greedy-decode step). `None` when TP is
    /// not attached (the caller falls back to `forward` + host argmax).
    pub fn forward_next(&mut self, ids: &[i32], start_pos: usize) -> Result<Option<u32>> {
        if self.rt.tp_ctx_dptr == 0 || self.tp_rank < 0 {
            return Ok(None);
        }
        let s = ids.len();
        let x = self.forward_streams(ids, start_pos)?;
        self.forward_head_next(&x, s)
    }

    /// Embed `ids` [s] → [s, hc*dim] bf16 streams (gather + replicate ×hc, §1.1 forward_embed).
    pub fn embed_tokens<X: crate::dsv4_gpu::Dsv4Buf<bf16>>(&self, ids: &CudaSlice<i32>, s: usize) -> Result<X> {
        self.rt.embed_tokens(&self.embed, ids, s, &self.cfg)
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
        }
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

    /// Trunk streams forward (the chunkable body of [`forward`](Self::forward)): embed → N
    /// layers (block_forward) → final `x` streams [s, hc*dim] bf16. NO trunk top (hc_head /
    /// norm / LM head) — the caller runs [`forward_head`](Self::forward_head) once on the
    /// tail. `ids` [s] are the input token ids (hash-router table keys for layers 0–2).
    /// `start_pos` is the absolute position of `ids[0]` (0 for prefill; current len for
    /// decode / chunk continuation). One call processes ≤ s_max rows; chunk boundaries at
    /// multiples of 128 carry all recurrent state (compressor frontier, indexer cache,
    /// window ring) in `self.states` — bitwise-identical to one-shot (§12.B.5).
    pub fn forward_streams(&mut self, ids: &[i32], start_pos: usize) -> Result<B> {
        let s = ids.len();
        let ids_dev = self.dev.htod_sync_copy(ids)?;
        self.forward_streams_dev::<B, S, CudaSlice<i32>, CudaSlice<u8>, CudaSlice<u32>>(&ids_dev, s, start_pos)
    }

    /// `forward_streams` with the ids already on device (the CUDA-graph path: the graph's
    /// persistent ids buffer is re-uploaded OUTSIDE the graph before each replay).
    pub fn forward_streams_dev<X: crate::dsv4_gpu::Dsv4Buf<bf16>, F: crate::dsv4_gpu::Dsv4Buf<f32>, I: crate::dsv4_gpu::Dsv4Buf<i32>, C: crate::dsv4_gpu::Dsv4Buf<u8>, U: crate::dsv4_gpu::Dsv4Buf<u32>>(&mut self, ids_dev: &CudaSlice<i32>, s: usize, start_pos: usize) -> Result<X> {
        anyhow::ensure!(
            self.layers.len() == self.states.len(),
            "layers/states length mismatch ({} vs {}) — load_trunk_top has no layers",
            self.layers.len(),
            self.states.len()
        );
        let mut x = self.embed_tokens::<X>(ids_dev, s)?;
        for i in 0..self.layers.len() {
            let o = self
                .rt
                .block_forward::<X, F, I, C, U>(&self.layers[i], &mut self.states[i], &mut self.scratch, &x, s, start_pos, ids_dev, &self.cfg)
                .with_context(|| format!("block_forward layer {i}"))?;
            x = o.y;
        }
        Ok(x)
    }

    /// Chunked prefill (§12.B.5): process `ids` (a full prompt, start_pos 0) in `chunk`-sized
    /// pieces — `chunk` MUST be a multiple of 128 (compressor-block alignment; PREFILL_CHUNK
    /// in production, smaller in gates) — and return the LAST chunk's streams [cs, hc*dim]
    /// bf16 + its row count (the trunk top runs once on that tail). Recurrent state carries
    /// across chunks in `self.states`; the result is bitwise-identical to one-shot prefill.
    pub fn forward_prefill_chunked(&mut self, ids: &[i32], chunk: usize) -> Result<(B, usize)> {
        anyhow::ensure!(chunk % 128 == 0 && chunk > 0, "prefill chunk {chunk} not a multiple of 128 (§12.B.5)");
        let s = ids.len();
        anyhow::ensure!(s > 0, "forward_prefill_chunked: empty prompt");
        let trace = crate::opts::var(crate::opt!("prefill-trace")).ok().is_some();
        let t0 = if trace { Some(std::time::Instant::now()) } else { None };
        let mut tail: Option<(B, usize)> = None;
        let mut c0 = 0usize;
        let mut n_chunks = 0usize;
        while c0 < s {
            let cs = chunk.min(s - c0);
            let tc = if trace { Some(std::time::Instant::now()) } else { None };
            let x = self
                .forward_streams(&ids[c0..c0 + cs], c0)
                .with_context(|| format!("prefill chunk @{c0} ({cs}/{s} tok)"))?;
            if let (Some(tc), Some(t0)) = (tc, t0) {
                self.rt.dev.synchronize().ok();
                eprintln!("[dsv4-pf] chunk @{c0} cs={cs}/{s} chunk_ms={} total_ms={}", tc.elapsed().as_millis(), t0.elapsed().as_millis());
            }
            tail = Some((x, cs));
            c0 += cs;
            n_chunks += 1;
        }
        if trace {
            eprintln!("[dsv4-pf] forward_prefill_chunked: {s} tok in {n_chunks} chunks, total_ms={}", t0.map(|t| t.elapsed().as_millis()).unwrap_or(0));
        }
        Ok(tail.expect("non-empty prompt produced no chunks"))
    }

    /// Full trunk forward: embed → N layers (block_forward) → hc_head → norm → LM head.
    /// `ids` [s] are the input token ids (hash-router table keys for layers 0–2). `start_pos` is
    /// the absolute position of `ids[0]` (0 for prefill; current len for decode). Returns the
    /// last-row fp32 logits [vocab]. `x` streams flow layer-to-layer in fp32-bf16 (hc_post output).
    /// A prefill longer than PREFILL_CHUNK is chunked (see forward_prefill_chunked) — this is
    /// what makes >4K prompts (200K, 1M) fit chunk-sized prefill scratch.
    pub fn forward(&mut self, ids: &[i32], start_pos: usize) -> Result<S> {
        let s = ids.len();
        let trace = crate::opts::var(crate::opt!("prefill-trace")).ok().is_some();
        let t0 = if trace { Some(std::time::Instant::now()) } else { None };
        if start_pos == 0 && s > PREFILL_CHUNK {
            let (x_tail, tail_len) = self.forward_prefill_chunked(ids, PREFILL_CHUNK)?;
            let (_collapsed, logits) = self.forward_head(&x_tail, tail_len)?;
            if let Some(t0) = t0 {
                self.rt.dev.synchronize().ok();
                eprintln!("[dsv4-pf] forward: {s} tok (chunked) total_ms={} tok/s={:.1}",
                    t0.elapsed().as_millis(), s as f64 / t0.elapsed().as_secs_f64());
            }
            return Ok(logits);
        }
        // CUDA-graph decode (R3A.1/E2, opt-in --graph=1): single-token decode replays a
        // per-fire-variant whole-forward graph instead of ~1600 eager launches. Bitwise by
        // construction (same kernels/args; only the launch vehicle changes) — verified by the
        // classifier's baked-value checks at capture + the replay≡eager gate in the tests.
        // Eager fallback: any capture/classify error poisons that variant (loud, once).
        if s == 1 && start_pos > 0 && crate::dsv4_gpu::env_flag_once(crate::opt!("graph")) {
            // Classifier disambiguation floor (gather CSA-vs-HCA baked values diverge) and
            // the hierarchical-topk regime limit (>16384 blocks ⇒ kernel sequence changes).
            if start_pos >= 130 && (start_pos + 1) / 4 <= 16384 {
                match self.forward_decode_graphed(ids[0], start_pos) {
                    Ok(logits) => return Ok(logits),
                    Err(e) => eprintln!("[dsv4-graph] eager fallback at sp={start_pos}: {e:#}"),
                }
            }
        }
        let x = self.forward_streams(ids, start_pos)?;
        let (_collapsed, logits) = self.forward_head(&x, s)?;
        if let Some(t0) = t0 {
            self.rt.dev.synchronize().ok();
            eprintln!("[dsv4-pf] forward: {s} tok (one-shot sp={start_pos}) total_ms={} tok/s={:.1}",
                t0.elapsed().as_millis(), s as f64 / t0.elapsed().as_secs_f64());
        }
        Ok(logits)
    }

    /// CUDA-graph decode step (the `--graph=1` arm of [`forward`](Self::forward)).
    /// Lazy per-variant capture: the capture run RECORDS (does not execute), so the token's
    /// output is produced by the replay that immediately follows — state advances exactly
    /// once, identical to an eager step. Capture/classify errors poison the variant
    /// (eager forever after, loudly) — they never execute partial GPU work, so the eager
    /// fallback is always safe. Replay errors after cuGraphLaunch propagate (state unknown).
    /// Pub for the replay≡eager bitwise gate (tests/dsv4_graph_test.rs).
    pub fn forward_decode_graphed(&mut self, id: i32, start_pos: usize) -> Result<S> {
        use crate::dsv4_graph::DecodeGraphs;
        // lazy init: func map from the runtime's kernel modules (spine + attn + comp both).
        if self.graphs.is_none() {
            let mut func_names: std::collections::HashMap<usize, &'static str> = std::collections::HashMap::new();
            for (n, f) in self.rt.spine.func_handles() {
                func_names.insert(f as usize, Box::leak(n.to_string().into_boxed_str()));
            }
            for (n, f) in self.rt.attn.func_handles() {
                func_names.insert(f as usize, Box::leak(n.to_string().into_boxed_str()));
            }
            if let Some(ck) = &self.rt.comp {
                for (n, f) in ck.comp.func_handles() {
                    func_names.insert(f as usize, Box::leak(n.to_string().into_boxed_str()));
                }
                for (n, f) in ck.spine.func_handles() {
                    func_names.insert(f as usize, Box::leak(n.to_string().into_boxed_str()));
                }
            }
            crate::dsv4_graph::raise_mempool_threshold(&self.dev)?;
            // create the dedicated GSlice pool EAGERLY (lazy creation inside capture would
            // be an illegal device-state change mid-capture).
            let _ = crate::dsv4_gpu::graph_mempool(&self.dev);
            // the decode-graph workspace slab: all transient GSlice allocs inside capture
            // become bump slices of it (NO alloc nodes in the graph — the only sound
            // multi-launch pattern, measured 2026-07-30). 256 MB >> the ~20 MB forward
            // transient footprint; overflow panics loudly with the required size.
            crate::dsv4_gpu::graph_ws_init(&self.dev, 256 * 1024 * 1024)?;
            self.graphs = Some(DecodeGraphs::new(
                &self.dev, func_names, self.rt.window,
                self.cfg.n_routed_experts, self.cfg.n_activated_experts,
                self.cfg.vocab_size,
            )?);
        }
        // Take the graphs out so the capture closure can call &mut self methods freely;
        // restored on every path (poison marks included).
        let mut graphs = self.graphs.take().unwrap();
        let result = self.graphed_step(&mut graphs, id, start_pos);
        self.graphs = Some(graphs);
        result
    }

    fn graphed_step(
        &mut self,
        graphs: &mut crate::dsv4_graph::DecodeGraphs,
        id: i32,
        start_pos: usize,
    ) -> Result<S> {
        use crate::dsv4_graph::{Slot, Variant};
        let variant = Variant::of(start_pos);
        if matches!(graphs.slot_ref(variant), Slot::Poisoned) {
            return Err(anyhow::anyhow!("variant {variant:?} poisoned (earlier capture failure)"));
        }
        if matches!(graphs.slot_ref(variant), Slot::Unborn) {
            // ---- capture (records, does NOT execute) ----
            let cap_result = (|| {
                // upload this token's id BEFORE capture (the graph reads the buffer, baked).
                unsafe {
                    cudarc::driver::result::memcpy_htod_async(
                        *graphs.ids_dev.device_ptr(),
                        &[id],
                        self.rt.stream.stream,
                    )
                    .map_err(|e| anyhow::anyhow!("graph capture ids upload: {e}"))?;
                }
                let r = unsafe {
                    cudarc::driver::sys::cuStreamBeginCapture_v2(
                        self.rt.stream.stream,
                        cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
                    )
                };
                if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    return Err(anyhow::anyhow!("cuStreamBeginCapture: {r:?}"));
                }
                // The GSlice family: every alloc/free/memset in the forward runs on
                // rt.stream (capture-legal — the blocker that killed the CudaSlice
                // attempt: INVALIDATED / STREAM_CAPTURE_IMPLICIT, DSV4_R3A.md §8).
                // GRAPH_CAPTURE_ACTIVE makes GSlice::drop leak to the graph pool instead
                // of freeing graph-owned memory outside the graph.
                crate::dsv4_gpu::GRAPH_CAPTURE_ACTIVE.store(true, std::sync::atomic::Ordering::Relaxed);
                crate::dsv4_gpu::graph_ws_begin_capture();
                // ONE slab-wide memset as the graph's first node (replaces ~150 per-alloc
                // memset nodes — the bump regions are contiguous; zero-at-start is
                // value-identical since no two sites share a region within one forward).
                unsafe {
                    let (d, b) = crate::dsv4_gpu::graph_ws_span();
                    cudarc::driver::result::memset_d8_async(d, 0, b, self.rt.stream.stream)
                        .map_err(|e| anyhow::anyhow!("graph slab memset: {e}"))?;
                }
                let fwd = (|| {
                    let x = self.forward_streams_dev::<
                        crate::dsv4_gpu::GB, crate::dsv4_gpu::GS,
                        crate::dsv4_gpu::GSlice<i32>, crate::dsv4_gpu::GSlice<u8>, crate::dsv4_gpu::GSlice<u32>,
                    >(&graphs.ids_dev, 1, start_pos)?;
                    let (_collapsed, logits) = self.forward_head::<crate::dsv4_gpu::GB, crate::dsv4_gpu::GS>(&x, 1)?;
                    // AUTO_FREE_ON_LAUNCH frees every in-graph allocation at launch end —
                    // the output MUST be copied to the persistent eager buffer INSIDE the
                    // graph (a memcpy node; the source is valid during the launch).
                    unsafe {
                        cudarc::driver::result::memcpy_dtod_async(
                            *graphs.logits_out.device_ptr(),
                            logits.dptr(),
                            logits.len() * 4,
                            self.rt.stream.stream,
                        )
                        .map_err(|e| anyhow::anyhow!("graph logits-out memcpy: {e}"))?;
                    }
                    Ok::<_, anyhow::Error>(())
                })();
                // ALWAYS close the capture (even when fwd failed) — an open capture turns
                // every subsequent GSlice drop into a pool-ownership violation.
                let mut graph: cudarc::driver::sys::CUgraph = std::ptr::null_mut();
                let r = unsafe { cudarc::driver::sys::cuStreamEndCapture(self.rt.stream.stream, &mut graph) };
                crate::dsv4_gpu::GRAPH_CAPTURE_ACTIVE.store(false, std::sync::atomic::Ordering::Relaxed);
                if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    return Err(anyhow::anyhow!("cuStreamEndCapture: {r:?}"));
                }
                fwd?;
                let g = graphs.instantiate(graph, start_pos)?;
                g.upload_once(&self.rt.stream)?;
                eprintln!("[dsv4-graph] captured variant {variant:?} at sp={start_pos} (ws high-water {} MB)",
                    crate::dsv4_gpu::graph_ws_high_water() / (1024 * 1024));
                *graphs.slot_mut(variant) = Slot::Ready(g);
                Ok::<_, anyhow::Error>(())
            })();
            if let Err(e) = cap_result {
                *graphs.slot_mut(variant) = Slot::Poisoned;
                return Err(e);
            }
        }
        // ---- replay ----
        let g = match graphs.slot_ref(variant) {
            Slot::Ready(g) => g,
            _ => unreachable!("post-capture slot is Ready"),
        };
        g.replay(&self.dev, &self.rt.stream, &graphs.ids_dev, &graphs.logits_out, id, start_pos)
    }

    /// Trunk forward that ALSO captures main_hidden = h.mean(streams) @ layers 40/41/42
    /// (§B.10 DSpark interface). Returns (last-row logits [vocab], main_hidden [s, 3*dim] bf16).
    /// `main_hidden` is None when the trunk slice doesn't contain all 3 target layers.
    /// Auto-chunks prompts > PREFILL_CHUNK (§12.B.5): main_hidden is captured per chunk and
    /// concatenated → bitwise-identical to one-shot (the chunked-prefill gate). The DSpark warm
    /// then primes the stage rings from the full main_hidden.
    pub fn forward_capture_main(&mut self, ids: &[i32], start_pos: usize) -> Result<(S, Option<B>)> {
        let s = ids.len();
        if start_pos == 0 && s > PREFILL_CHUNK {
            return self.forward_capture_main_chunked(ids, PREFILL_CHUNK);
        }
        self.forward_capture_main_oneshot(ids, start_pos)
    }

    /// One-shot capture (probe-scale; s must fit s_max). The original path.
    fn forward_capture_main_oneshot(&mut self, ids: &[i32], start_pos: usize) -> Result<(S, Option<B>)> {
        let s = ids.len();
        let ids_dev = self.dev.htod_sync_copy(ids)?;
        let mut x = self.embed_tokens(&ids_dev, s)?;
        let targets = &self.cfg.dspark_target_layer_ids;
        let have_all = targets.iter().all(|&t| t < self.layers.len());
        let mut caps: [Option<B>; 3] = [None, None, None];
        for i in 0..self.layers.len() {
            let o = self
                .rt
                .block_forward::<B, S, CudaSlice<i32>, CudaSlice<u8>, CudaSlice<u32>>(&self.layers[i], &mut self.states[i], &mut self.scratch, &x, s, start_pos, &ids_dev, &self.cfg)
                .with_context(|| format!("block_forward layer {i}"))?;
            x = o.y;
            if have_all {
                for (k, &t) in targets.iter().enumerate() {
                    if i == t { caps[k] = Some(x.clone()); }
                }
            }
        }
        let (_collapsed, logits) = self.forward_head(&x, s)?;
        let main_hidden = if have_all {
            Some(self.rt.compute_main_hidden(
                caps[0].as_ref().unwrap(),
                caps[1].as_ref().unwrap(),
                caps[2].as_ref().unwrap(),
                s, &self.cfg,
            )?)
        } else { None };
        Ok((logits, main_hidden))
    }

    /// Chunked prefill + main_hidden capture (§12.B.5): process `ids` in `chunk`-sized 128-aligned
    /// pieces, capturing h.mean(streams) @ 40/41/42 per chunk → concatenated main_hidden
    /// [s, 3*dim]. Bitwise-identical to one-shot (the recurrent state carries across chunks).
    fn forward_capture_main_chunked(&mut self, ids: &[i32], chunk: usize) -> Result<(S, Option<B>)> {
        anyhow::ensure!(chunk % 128 == 0 && chunk > 0, "capture chunk {chunk} not a multiple of 128");
        let s = ids.len();
        let targets = &self.cfg.dspark_target_layer_ids;
        let have_all = targets.iter().all(|&t| t < self.layers.len());
        let three_d = 3 * self.cfg.dim;
        let mut full_mh: Option<B> = if have_all { Some(self.dev.alloc_zeros::<bf16>(s * three_d)?) } else { None };
        let mut tail: Option<(B, usize)> = None;
        let mut c0 = 0usize;
        while c0 < s {
            let cs = chunk.min(s - c0);
            let ids_dev = self.dev.htod_sync_copy(&ids[c0..c0 + cs])?;
            let mut x = self.embed_tokens(&ids_dev, cs)?;
            let mut caps: [Option<B>; 3] = [None, None, None];
            for i in 0..self.layers.len() {
                let o = self
                    .rt
                    .block_forward::<B, S, CudaSlice<i32>, CudaSlice<u8>, CudaSlice<u32>>(&self.layers[i], &mut self.states[i], &mut self.scratch, &x, cs, c0, &ids_dev, &self.cfg)
                    .with_context(|| format!("block_forward layer {i} (chunked capture @{c0})"))?;
                x = o.y;
                if have_all {
                    for (k, &t) in targets.iter().enumerate() {
                        if i == t { caps[k] = Some(x.clone()); }
                    }
                }
            }
            if let Some(full) = full_mh.as_mut() {
                let cmh = self.rt.compute_main_hidden(
                    caps[0].as_ref().unwrap(), caps[1].as_ref().unwrap(), caps[2].as_ref().unwrap(),
                    cs, &self.cfg,
                )?;
                // d2d the chunk's main_hidden [cs, 3*dim] into the full buffer at row c0.
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(
                        (*full.device_ptr()) + (c0 * three_d * 2) as u64,
                        *cmh.device_ptr(), cs * three_d * 2, self.rt.stream.stream,
                    ).map_err(|e| anyhow!("chunked main_hidden dtod: {e}"))?;
                }
            }
            tail = Some((x, cs));
            c0 += cs;
        }
        let (x_tail, tail_len) = tail.expect("non-empty prompt produced no chunks");
        let (_collapsed, logits) = self.forward_head(&x_tail, tail_len)?;
        Ok((logits, full_mh))
    }
    /// Trunk verify forward returning ALL `s` rows of fp32 logits (for DSpark verify: compare
    /// each draft position's argmax). Same as `forward` but the LM head runs over every row
    /// (not last-only). Used at start_pos>0 over [real + drafts] (the sequential per-token
    /// attention path handles s>1 correctly — §6a.1).
    pub fn forward_verify_logits(&mut self, ids: &[i32], start_pos: usize) -> Result<S> {
        let (logits, _mh) = self.forward_verify_capture_main(ids, start_pos)?;
        Ok(logits)
    }

    /// Verify forward + main_hidden capture (§B.10 — h.mean(streams) @ 40/41/42) for ALL `s`
    /// verify rows. The DSpark serve loop uses the committed-prefix rows to re-prime the draft
    /// ring (AGENTS §2.6 "re-prime with real verify hiddens") so the draft attends to a
    /// contiguous main_kv window (no gaps at the accepted-draft positions). Returns
    /// (per-row logits [s, vocab] fp32, main_hidden [s, 3*dim] bf16).
    pub fn forward_verify_capture_main(&mut self, ids: &[i32], start_pos: usize) -> Result<(S, Option<B>)> {
        let s = ids.len();
        let ids_dev = self.dev.htod_sync_copy(ids)?;
        let mut x: B = self.embed_tokens(&ids_dev, s)?;
        // R4 selective rollback: lazily allocate the per-layer verify-activation capture and
        // record each layer's input x (committed-prefix re-application without a full re-forward).
        // Item 3.3: the verify WIDTH is no longer fixed (adaptive depth drafts D rows, verifies
        // D+1) — grow the capture when a wider verify arrives (contents are fully re-captured
        // per forward, so a grow is a fresh zeroed alloc, never a copy).
        let n_hc_dim = self.cfg.hc_mult * self.cfg.dim;
        let need = s * n_hc_dim;
        if self.x_cap.as_ref().map_or(true, |c| c[0].len() < need) {
            let caps: Result<Vec<_>, _> = (0..self.layers.len())
                .map(|_| self.dev.alloc_zeros::<bf16>(need).map_err(|e| anyhow!("x_cap alloc: {e}")))
                .collect();
            self.x_cap = Some(caps?);
        }
        let targets = &self.cfg.dspark_target_layer_ids;
        let have_all = targets.iter().all(|&t| t < self.layers.len());
        let mut caps: [Option<B>; 3] = [None, None, None];
        for i in 0..self.layers.len() {
            {
                // capture this layer's input (the committed rows are what a full re-forward
                // would have seen — width-bitwise: x@p at s=6 == s=1).
                let xc = &mut self.x_cap.as_mut().unwrap()[i];
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(
                        *xc.device_ptr(), *x.device_ptr(), x.len() * 2, self.rt.stream.stream,
                    )
                    .map_err(|e| anyhow!("x_cap d2d layer {i}: {e}"))?;
                }
            }
            let o = self
                .rt
                .block_forward::<B, S, CudaSlice<i32>, CudaSlice<u8>, CudaSlice<u32>>(&self.layers[i], &mut self.states[i], &mut self.scratch, &x, s, start_pos, &ids_dev, &self.cfg)
                .with_context(|| format!("block_forward layer {i} (verify+capture)"))?;
            x = o.y;
            if have_all {
                for (k, &t) in targets.iter().enumerate() {
                    if i == t { caps[k] = Some(x.clone()); }
                }
            }
        }
        let (dim, vocab) = (self.cfg.dim, self.cfg.vocab_size);
        let collapsed = self.rt.hc_head(&x, &self.hc_head_fn, &self.hc_head_base, &self.hc_head_scale, s, &self.cfg)?;
        let yn = self.rt.rmsnorm(&collapsed, &self.norm, s, dim, self.cfg.norm_eps)?;
        let logits = self.rt.lm_head(&self.head, &yn, s, dim, vocab)?;
        let main_hidden = if have_all {
            Some(self.rt.compute_main_hidden(
                caps[0].as_ref().unwrap(), caps[1].as_ref().unwrap(), caps[2].as_ref().unwrap(),
                s, &self.cfg,
            )?)
        } else { None };
        Ok((logits, main_hidden))
    }

    /// DSpark selective rollback (R4): after a rejected verify, re-apply the committed prefix
    /// `start_pos .. start_pos+n-1` (verify rows 0..n-1) WITHOUT a full re-forward. The verify
    /// captured every layer's input x (width-bitwise == a re-forward's), so:
    ///   - the KV ring is rebuilt by `dspark_attn_warm_range` (batched, the gated warm path);
    ///   - compressor + indexer state is re-applied by `forward_tokens` (batched GEMM + the
    ///     same sequential state chain as sequential decode — gates prove the equivalence).
    /// The attention gather output, MoE and LM head are NOT recomputed — nothing downstream
    /// consumes them (the next step's carry forward recomputes from state + the new token).
    /// Final state is identical to the old per-token full re-forward.
    pub fn readvance_committed(&mut self, start_pos: usize, n: usize) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let dim = self.cfg.dim;
        let eps = self.cfg.norm_eps;
        let ks = self.rt.comp.as_ref().expect("readvance_committed needs comp kernels");
        let x_cap = self.x_cap.as_ref().expect("readvance_committed before any verify capture");
        for i in 0..self.layers.len() {
            let kind = self.layers[i].kind;
            let rope = self.rt.rope_for(kind);
            let layer = &self.layers[i];
            let st = &mut self.states[i];
            // the compressor/KV path consumes yn = rmsnorm(hc_pre(x)) (the attention input
            // [n, dim]), not the raw [n, hc*dim] block input — recompute it from the capture.
            let xc = &x_cap[i];
            let (y, _posts, _combs) = self
                .rt
                .hc_pre::<B, S>(xc, n, &layer.hc_attn_fn, &layer.hc_attn_base, &layer.hc_attn_scale, &self.cfg)
                .with_context(|| format!("readvance hc_pre layer {i}"))?;
            let yn = self
                .rt
                .rmsnorm(&y, &layer.attn_norm, n, dim, eps)
                .with_context(|| format!("readvance norm layer {i}"))?;
            self.rt
                .dspark_attn_warm_range(layer, st, &yn, n, start_pos, rope, &self.cfg)
                .with_context(|| format!("readvance warm layer {i}"))?;
            if let Some(comp) = &st.attn_compressor {
                comp.forward_tokens::<B, S, CudaSlice<i32>, CudaSlice<u32>>(&self.dev, ks, &self.rt.stream, &yn, n, start_pos, rope)
                    .with_context(|| format!("readvance compressor layer {i}"))?;
            }
            if let Some(idx) = &st.indexer {
                idx.comp
                    .forward_tokens::<B, S, CudaSlice<i32>, CudaSlice<u32>>(&self.dev, ks, &self.rt.stream, &yn, n, start_pos, rope)
                    .with_context(|| format!("readvance indexer layer {i}"))?;
            }
        }
        Ok(())
    }

    /// Snapshot the full per-layer attention state (KV ring + compressor + indexer) for DSpark
    /// verify rollback. Call BEFORE `forward_verify_logits`; [`restore_verify_state`] rewinds.
    pub fn snapshot_verify_state(&self) -> Result<Vec<LayerVerifySnap>> {
        let mut snaps = Vec::with_capacity(self.states.len());
        for st in &self.states {
            snaps.push(st.snapshot_verify(&self.dev, &self.rt.stream)?);
        }
        Ok(snaps)
    }

    /// Rewind every layer's attention state to its snapshot (D2D on the compute stream).
    pub fn restore_verify_state(&self, snaps: &[LayerVerifySnap]) -> Result<()> {
        anyhow::ensure!(snaps.len() == self.states.len(), "restore: {} snaps vs {} layers", snaps.len(), self.states.len());
        for (st, snap) in self.states.iter().zip(snaps) {
            st.restore_verify(snap, &self.dev, &self.rt.stream)?;
        }
        Ok(())
    }

    /// Reset ALL per-layer attention state to the prefill-start condition (zeroed KV ring,
    /// compressor-cache tail, and frontier state re-initialized). Used by the persistent
    /// server loop between requests so each request starts from a clean model state.
    pub fn reset_states(&mut self) -> Result<()> {
        let dev = &self.dev;
        let stream = &self.rt.stream;
        let comp_ks = self.rt.comp.as_ref().expect("CSA/HCA needs comp kernels");
        for st in &mut self.states {
            // Zero the unified kv_cache (ring + compressor-cache tail).
            let n = st.kv_cache.len();
            let zeros = vec![bf16::ZERO; n];
            dev.htod_sync_copy_into(&zeros, &mut st.kv_cache)
                .map_err(|e| anyhow!("reset kv_cache: {e}"))?;
            // Reset compressor frontier (kv=0, score=−inf via the state-init kernel).
            if let Some(comp) = &st.attn_compressor {
                comp.reset(comp_ks, stream)?;
            }
            if let Some(idx) = &st.indexer {
                idx.comp.reset(comp_ks, stream)?;
            }
        }
        dev.synchronize().map_err(|e| anyhow!("reset sync: {e}"))?;
        Ok(())
    }

    /// R2.3 prefix-cache forward: carry model state across chat turns via snapshot/restore at the
    /// conversation prefix (before the generation priming). Turn 2+ forwards only the delta (the
    /// new tokens), not the full conversation — bitwise-identical to a full re-prefill (the
    /// §12.B.5 chunked-prefill proof extends across request boundaries: the recurrent state
    /// carries in self.states, and a request boundary IS a chunk boundary).
    ///
    /// The split point is the conversation prefix length (prompt_len - priming_len), padded DOWN
    /// to a 128-aligned position (compressor block alignment). The "prefix" is prefill+snapshot'd;
    /// the "tail" (non-aligned remainder + priming, ≤ 129 tokens) is forwarded after the snapshot.
    /// On turn 2+, the longest matching snapshot is restored, the delta (prefix growth + new
    /// tail) is forwarded. Item 2.3 (session 9): the single snapshot became a small LRU of
    /// 128-aligned checkpoints — see [`PrefixCache`] and docs/DSV4_PREFIX_CACHE_DESIGN.md.
    ///
    /// Caller owns a `cache` (per server session). `priming_len` = the generation priming token
    /// count (ASSISTANT_SP + THINKING_START/END).
    /// Returns the last-row fp32 logits [vocab] for the first decode token.
    pub fn forward_prefix_cached(
        &mut self,
        ids: &[i32],
        priming_len: usize,
        cache: &mut PrefixCache,
    ) -> Result<S> {
        let (x, tail_len, _mh) = self.forward_prefix_cached_core(ids, priming_len, cache, false)?;
        let (_c, logits) = self.forward_head(&x, tail_len)?;
        Ok(logits)
    }

    /// R3.2 (L2): prefix-cached prefill + next token via the vocab-parallel maxloc head
    /// (`None` when TP is not attached — the caller uses `forward_prefix_cached` + argmax).
    pub fn forward_prefix_cached_next(
        &mut self,
        ids: &[i32],
        priming_len: usize,
        cache: &mut PrefixCache,
    ) -> Result<Option<u32>> {
        let (x, tail_len, _mh) = self.forward_prefix_cached_core(ids, priming_len, cache, false)?;
        self.forward_head_next(&x, tail_len)
    }

    /// Item 3.4 (DSpark-in-server): prefix-cached prefill that ALSO captures the per-position
    /// main_hidden (h.mean(streams) @ layers 40/41/42) for the DSpark draft ring warm — the
    /// cache-entry prefix's hidden means are stored in the entry (see [`PrefixCacheEntry`]) and
    /// restored on a hit, so the returned buffer covers ALL `s` positions either way. Bitwise
    /// identical to a full `forward_capture_main` re-prefill (the R2.3/§12.B.5 proof — the
    /// chunked forward sequence is the same; only the checkpoint/restore is added). Greedy
    /// callers use `forward_prefix_cached` (no capture cost). Returns (last-row logits, main_hidden).
    pub fn forward_prefix_cached_capture_main(
        &mut self,
        ids: &[i32],
        priming_len: usize,
        cache: &mut PrefixCache,
    ) -> Result<(S, Option<B>)> {
        let (x, tail_len, main_hidden) = self.forward_prefix_cached_core(ids, priming_len, cache, true)?;
        let (_c, logits) = self.forward_head(&x, tail_len)?;
        Ok((logits, main_hidden))
    }

    /// The shared prefix-cache prefill body (R2.3 + item 2.3): restore-or-recompute the 128-aligned
    /// conversation prefix, then forward the tail. Returns the tail streams [tail, hc*dim] bf16
    /// + the tail length; the caller runs a trunk-top variant (`forward_head` for full logits,
    /// `forward_head_next` for the R3.2 maxloc head) on it. `capture_main` additionally captures
    /// the full-prompt main_hidden into the returned `Option` (DSpark server path).
    ///
    /// Item 2.3 (session 9): longest-prefix match over the LRU at ANY 128-aligned boundary.
    /// Checkpoint policy: cold prefill snapshots at each PREFILL_CHUNK boundary (4096, itself
    /// 128-aligned) + the final aligned boundary; growth/delta forwards snapshot at EVERY
    /// 128-boundary (1–3 per turn, ~0.4 ms each). All forward/snapshot sequences are the same
    /// chunked-prefill code path the R2.3 proof covers → bitwise-identical to a full re-prefill.
    fn forward_prefix_cached_core(
        &mut self,
        ids: &[i32],
        priming_len: usize,
        cache: &mut PrefixCache,
        capture_main: bool,
    ) -> Result<(B, usize, Option<B>)> {
        let s = ids.len();
        anyhow::ensure!(s > priming_len, "forward_prefix_cached: prompt {s} ≤ priming {priming_len}");
        let conv_prefix_len = s - priming_len;
        let aligned_len = (conv_prefix_len / 128) * 128;

        // Item 3.4: a full-prompt main_hidden capture buffer [s, 3*dim] — filled per chunk by
        // forward_chunk_capture (greedy callers pass capture_main=false → None, zero overhead).
        let three_d = 3 * self.cfg.dim;
        let targets = &self.cfg.dspark_target_layer_ids;
        let have_all = targets.iter().all(|&t| t < self.layers.len());
        let mut full_mh: Option<B> = if capture_main && have_all {
            Some(self.dev.alloc_zeros::<bf16>(s * three_d)?)
        } else {
            None
        };
        // A MISS with aligned_len==0 (very short prompt < 128 tokens) cannot take a snapshot —
        // the lookup itself excludes it (entries require len > 0, len <= aligned_len == 0), so
        // no stale entry can falsely match a future request. A HIT entry WITHOUT stored
        // main_hidden (a greedy-process entry — can't happen in a DSpark process, but be
        // defensive) is treated as a miss: the warm needs the prefix's hidden means.
        let hit = cache.lookup(ids, aligned_len)
            .filter(|e| !capture_main || e.main_hidden.is_some());

        if hit.is_none() {
            // Full re-prefill: reset, clear stale cache, prefill the 128-aligned prefix with
            // checkpoints, forward the tail. (aligned_len==0 leaves the cache untouched — the
            // lookup can never match it.)
            self.reset_states()?;
            if aligned_len > 0 {
                // R2.3 fix (RUN-4 crash): the aligned prefix can EXCEED s_max (any prompt > 4226
                // tok — the harness's 6148 row SIGSEGV'd both ranks). forward_streams processes
                // ≤ s_max rows per call — the prefix MUST go through the §12.B.5 chunked path
                // (bitwise-identical to one-shot, so the snapshot is unchanged). Checkpoint at
                // each PREFILL_CHUNK boundary (128-aligned) + the final boundary — the coarse
                // cross-conversation reuse points.
                let mut c0 = 0usize;
                while c0 < aligned_len {
                    let cs = PREFILL_CHUNK.min(aligned_len - c0);
                    let _x = self.forward_chunk_capture(&ids[c0..c0 + cs], c0, full_mh.as_mut(), c0)?;
                    c0 += cs;
                    let mh = prefix_mh_clone(&self.dev, &self.rt.stream, full_mh.as_ref(), c0, three_d)?;
                    cache.insert(c0, &ids[..c0], self.snapshot_verify_state()?, mh);
                }
            }
            // Forward the tail (non-aligned remainder + priming) and get logits.
            let tail = &ids[aligned_len..];
            let x = self.forward_chunk_capture(tail, aligned_len, full_mh.as_mut(), aligned_len)?;
            Ok((x, tail.len(), full_mh))
        } else {
            // Prefix cache HIT: restore the longest match, forward the delta (prefix growth) with
            // a checkpoint at EVERY 128-boundary, forward tail.
            let entry = hit.expect("checked above");
            self.restore_verify_state(&entry.snap)?;
            // Item 3.4: restore the hit prefix's main_hidden rows into the capture buffer (the
            // warm needs ALL positions; the delta/tail rows are freshly captured below).
            if let (Some(full), Some(emh)) = (full_mh.as_mut(), entry.main_hidden.as_ref()) {
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(
                        *full.device_ptr(), *emh.device_ptr(), entry.len * three_d * 2, self.rt.stream.stream,
                    ).map_err(|e| anyhow!("prefix main_hidden restore d2d: {e}"))?;
                }
            }
            let start = entry.len;
            if aligned_len > start {
                // Forward the prefix growth (128-aligned start and end → 128-token slices land
                // on compressor-block boundaries; the checkpoint after each slice makes every
                // aligned position a future hit point). A >4K-token turn is legal — the slices
                // ride the same start_pos chain as the chunked-prefill path.
                let mut g0 = start;
                while g0 < aligned_len {
                    let cs = 128.min(aligned_len - g0);
                    let _x = self.forward_chunk_capture(&ids[g0..g0 + cs], g0, full_mh.as_mut(), g0)?;
                    g0 += cs;
                    let mh = prefix_mh_clone(&self.dev, &self.rt.stream, full_mh.as_ref(), g0, three_d)?;
                    cache.insert(g0, &ids[..g0], self.snapshot_verify_state()?, mh);
                }
            }
            // Forward the tail + priming and get logits.
            let tail = &ids[aligned_len..];
            let x = self.forward_chunk_capture(tail, aligned_len, full_mh.as_mut(), aligned_len)?;
            Ok((x, tail.len(), full_mh))
        }
    }

    /// One prefill chunk forward with OPTIONAL main_hidden capture (item 3.4): rows
    /// `[row0..row0+cs]` of the caller's `[s, 3*dim]` buffer. With `full_mh: None` this is the
    /// exact `forward_streams` sequence (bitwise); with capture it mirrors
    /// `forward_capture_main_chunked`'s per-chunk loop (h.mean(streams) @ 40/41/42 per chunk,
    /// d2d'd into the caller's buffer).
    fn forward_chunk_capture(
        &mut self,
        ids_chunk: &[i32],
        start_pos: usize,
        full_mh: Option<&mut B>,
        row0: usize,
    ) -> Result<B> {
        let s = ids_chunk.len();
        let ids_dev = self.dev.htod_sync_copy(ids_chunk)?;
        let mut x = self.embed_tokens(&ids_dev, s)?;
        if let Some(full) = full_mh {
            let targets = &self.cfg.dspark_target_layer_ids;
            let have_all = targets.iter().all(|&t| t < self.layers.len());
            let three_d = 3 * self.cfg.dim;
            let mut caps: [Option<B>; 3] = [None, None, None];
            for i in 0..self.layers.len() {
                let o = self
                    .rt
                    .block_forward::<B, S, CudaSlice<i32>, CudaSlice<u8>, CudaSlice<u32>>(&self.layers[i], &mut self.states[i], &mut self.scratch, &x, s, start_pos, &ids_dev, &self.cfg)
                    .with_context(|| format!("block_forward layer {i} (prefix capture @{start_pos})"))?;
                x = o.y;
                if have_all {
                    for (k, &t) in targets.iter().enumerate() {
                        if i == t { caps[k] = Some(x.clone()); }
                    }
                }
            }
            if have_all {
                let cmh = self.rt.compute_main_hidden(
                    caps[0].as_ref().unwrap(), caps[1].as_ref().unwrap(), caps[2].as_ref().unwrap(),
                    s, &self.cfg,
                )?;
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(
                        (*full.device_ptr()) + (row0 * three_d * 2) as u64,
                        *cmh.device_ptr(), s * three_d * 2, self.rt.stream.stream,
                    ).map_err(|e| anyhow!("prefix main_hidden dtod @{row0}: {e}"))?;
                }
            }
        } else {
            for i in 0..self.layers.len() {
                let o = self
                    .rt
                    .block_forward::<B, S, CudaSlice<i32>, CudaSlice<u8>, CudaSlice<u32>>(&self.layers[i], &mut self.states[i], &mut self.scratch, &x, s, start_pos, &ids_dev, &self.cfg)
                    .with_context(|| format!("block_forward layer {i} (prefix @{start_pos})"))?;
                x = o.y;
            }
        }
        Ok(x)
    }
}

/// Item 3.4: d2d clone of the captured main_hidden rows [0..len] for a prefix-cache entry (a
/// later HIT restores them for the DSpark draft-ring warm). `None` when the capture is off
/// (greedy-process entries store no hidden means). A free helper so the caller's borrows of
/// `self` (forward/snapshot) stay disjoint from the entry write.
fn prefix_mh_clone(
    dev: &Arc<CudaDevice>,
    stream: &cudarc::driver::CudaStream,
    full_mh: Option<&B>,
    len: usize,
    three_d: usize,
) -> Result<Option<B>> {
    match full_mh {
        None => Ok(None),
        Some(fm) => {
            let c = dev.alloc_zeros::<bf16>(len * three_d)?;
            unsafe {
                cudarc::driver::result::memcpy_dtod_async(
                    *c.device_ptr(), *fm.device_ptr(), len * three_d * 2, stream.stream,
                ).map_err(|e| anyhow!("prefix main_hidden entry d2d: {e}"))?;
            }
            Ok(Some(c))
        }
    }
}

/// Item 2.3 prefix cache: a small LRU of 128-aligned exact-prefix checkpoints (docs/
/// DSV4_PREFIX_CACHE_DESIGN.md). Each entry is a full per-layer attention-state snapshot
/// ([`LayerVerifySnap`]) at an aligned prefix length, keyed by the EXACT prefix tokens
/// (the u64 hash is a fast-rejection prefilter only — the authoritative key is `ids`).
/// Eviction: LRU at capacity (K=8 → ~840 MB @ SEQ 8192, ~0.8% of the working set).
pub struct PrefixCacheEntry {
    pub len: usize,        // aligned prefix length (multiple of 128, > 0)
    pub hash: u64,         // FNV-1a over ids[..len] — rejection prefilter, never the authority
    pub ids: Vec<i32>,     // exact prefix tokens — the authoritative key
    pub snap: Vec<LayerVerifySnap>,
    /// Item 3.4 (DSpark-in-server): the prefix's main_hidden rows [len, 3*dim] bf16 — captured
    /// during a DSpark-process prefill so a later HIT can re-warm the draft ring without a
    /// re-forward. Greedy-process entries store None (zero capture cost). ~12 KB/token.
    pub main_hidden: Option<B>,
    pub last_use: u64,
}

pub struct PrefixCache {
    pub entries: Vec<PrefixCacheEntry>,
    pub cap: usize,
    tick: u64,
}

impl PrefixCache {
    pub fn new(cap: usize) -> Self {
        PrefixCache { entries: Vec::new(), cap: cap.max(1), tick: 0 }
    }

    fn fnv1a(ids: &[i32]) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for &t in ids {
            for b in t.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        h
    }

    /// Longest-prefix match: the entry with the greatest `len <= aligned_len` whose exact
    /// token prefix equals `ids[..len]`. O(cap × len) memcmp-class — a few µs on a miss.
    pub fn lookup(&self, ids: &[i32], aligned_len: usize) -> Option<&PrefixCacheEntry> {
        let mut best: Option<&PrefixCacheEntry> = None;
        for e in &self.entries {
            if e.len == 0 || e.len > aligned_len {
                continue;
            }
            if e.hash != Self::fnv1a(&ids[..e.len]) {
                continue;
            }
            if ids[..e.len] == e.ids[..] && best.map_or(true, |b| e.len > b.len) {
                best = Some(e);
            }
        }
        best
    }

    /// Insert (or refresh, replacing any same-`len` entry — a re-checkpoint of the same
    /// boundary is value-identical, the LRU tick is what changes) + LRU-evict at capacity.
    /// `main_hidden` = the entry's captured prefix hidden means (None on the greedy path).
    pub fn insert(&mut self, len: usize, ids: &[i32], snap: Vec<LayerVerifySnap>, main_hidden: Option<B>) {
        self.tick = self.tick.wrapping_add(1);
        if let Some(existing) = self.entries.iter_mut().find(|e| e.len == len) {
            existing.hash = Self::fnv1a(ids);
            existing.ids.clear();
            existing.ids.extend_from_slice(ids);
            existing.snap = snap;
            existing.main_hidden = main_hidden;
            existing.last_use = self.tick;
            return;
        }
        self.entries.push(PrefixCacheEntry {
            len,
            hash: Self::fnv1a(ids),
            ids: ids.to_vec(),
            snap,
            main_hidden,
            last_use: self.tick,
        });
        while self.entries.len() > self.cap {
            let oldest = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_use)
                .map(|(i, _)| i)
                .expect("non-empty");
            self.entries.remove(oldest);
        }
    }

    /// Touch an entry after a successful hit (LRU recency).
    pub fn touch(&mut self, len: usize) {
        self.tick = self.tick.wrapping_add(1);
        if let Some(e) = self.entries.iter_mut().find(|e| e.len == len) {
            e.last_use = self.tick;
        }
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
