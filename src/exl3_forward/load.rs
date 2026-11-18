//! `FwdModel::load` and `load_tp`: the EXL3 / FP4-Local loader (per-rank expert dealing under TP). One single
//! function of ~1.96K lines kept whole: splitting an item is a code change, not part of this move.
//! See `mod.rs` for the file map.

use super::*;

impl FwdModel {
    pub fn load(pack_dir: &str, width: usize, max_pos: usize) -> Result<Arc<FwdModel>> {
        Self::load_tp(pack_dir, width, max_pos, None)
    }

    /// TP-A: `load` with an optional tensor-parallel attachment (rank/world + the RDMA link). With
    /// `tp = Some`, every TRUNK MoE layer loads only this rank's statically dealt routed experts
    /// (xtp::ep_local_experts; load-time shape asserts in xtp::EpShard::new) and the link is
    /// attached (payload set, proxy thread spawned) before the model is returned. Everything else
    /// (hc mixers, router, shared expert, attention/GDN, lm_head, PLE, the MTP head) is REPLICATED.
    pub fn load_tp(pack_dir: &str, width: usize, max_pos: usize, tp: Option<xtp::TpAttach>) -> Result<Arc<FwdModel>> {
        // Public PR #21: every later packing of max_pos (the QSA geometry word, rdim_mp) relies on this.
        check_max_pos_packable(max_pos)?;
        // TUNE §4.6: the table is frozen before any knob is read (Load-scope knobs are read below);
        // entry points that booted no table run the built-in defaults (today's behaviour).
        tune::ensure_frozen("FwdModel::load");
        // FP4L: an FP4-Local dir (ModelOpt MXFP8 dense + NVFP4 csf experts) loads its weights through
        // fp4l_fwd; everything else in this loader (state, KV, QSA, graphs, TP) is shared.
        let fxp = fp4l_fwd::open_if_fp4l(pack_dir)?;
        // single-process entry points (bench, probes) get the same stored-precision default as the server;
        // under TP the head applied it and every rank installed the head's options
        if fxp.is_some() && tp.is_none() { crate::fp4l::apply_pack_defaults(pack_dir); }
        // FP4L precision levers (PLAN/FP4L_DESIGN.md §3a): the trunk-held bf16 tensors as stored (--fp4l-plain
        // exact) and the fp32 norms (--fp4l-norms f32). EXL3 packs: both false, every path below unchanged.
        let fx_plain = fxp.is_some() && fp4l_fwd::plain_exact();
        let fx_norms = fxp.is_some() && fp4l_fwd::norms_f32();
        if fx_plain { fp4l_fwd::set_fx_scratch(); }
        if fx_norms && tp.is_some() && xtp::pf_overlap_fold() {
            // the fold's fused K2m + inject + norm kernel has no fp32-norm twin: run the default overlap (every
            // rank takes the same decision from the same shipped flags, so the TP boot agree still matches)
            xtp::set_pf_overlap(xtp::PF_OVERLAP_DEFAULT);
            crate::rprintln!("[fp4l] --tp-prefill-overlap fold asked; this boot runs {} instead: the fold kernel has no \
                              fp32-norm variant (--fp4l-norms f32). --fp4l-norms f16 keeps fold", xtp::pf_overlap_desc(xtp::PF_OVERLAP_DEFAULT));
        }
        let pack_opt = if fxp.is_none() { Some(Exl3Pack::open(pack_dir)?) } else { None };
        let shard_fmt = if fxp.is_some() { crate::shard_plan::ShardFormat::Fp4l } else { crate::shard_plan::ShardFormat::Exl3 };
        let pack_shards: Vec<String> = match (pack_opt.as_ref(), fxp.as_deref()) {
            (Some(p), _) => p.shards.clone(),
            (None, Some(f)) => f.file_names.clone(),
            (None, None) => unreachable!(),
        };
        // B32: a shard dir (gb10_shard.json) must be THIS attach's (format, world, rank); a whole pack passes.
        let shard_info = match tp.as_ref() {
            Some(a) => crate::shard_plan::check_shard_dir(std::path::Path::new(pack_dir),
                shard_fmt, a.world as usize, a.rank as usize)?,
            None => crate::shard_plan::read_shard_info(std::path::Path::new(pack_dir))?
                .map(|i| -> Result<_> { anyhow::bail!("{pack_dir} is a B32 shard dir (rank {} of world {}) — it cannot be loaded without TP", i.rank, i.world) })
                .transpose()?,
        };
        let shard_tag = shard_info.as_ref().map(|i| format!(" (B32 shard dir: exl3 rank {} of world {}, deal {} — the head's shipped plan omitted it)", i.rank, i.world, i.deal)).unwrap_or_default();
        let mut cfg = match pack_opt.as_ref() {
            Some(p) => p.cfg.clone(),
            None => crate::qwen::Config::from_config_json(&format!("{}/config.json", pack_dir.trim_end_matches('/')))
                .with_context(|| format!("FP4L {pack_dir}: config.json"))?,
        };
        // EXL27B (Qwen3.5-class dense packs): the MLP is one dense projection triple. The config
        // carries it as ONE expert (num_experts 1, top-1, moe_intermediate = intermediate) so the
        // shared sizing code sees its width; the layer itself is MoeLayer::dense (no MoE code runs).
        let dense_mlp = cfg.num_experts == 0 && cfg.intermediate_size > 0;
        // EXL27B: a Qwen3.5-class MoE pack (plain residual, routed experts): MoeLayer::split
        let split_moe = cfg.num_experts > 1 && cfg.hc_lowrank == 0;
        if dense_mlp {
            cfg.num_experts = 1;
            cfg.num_experts_per_tok = 1;
            cfg.moe_intermediate_size = cfg.intermediate_size;
        }
        let kv_fmt = kv_fmt_from_opts();
        // The PLE n-gram context's EOS (history fill at a slot start + the reset token): the trellis packs keep
        // the trunk's historical 248046; an FP4-Local pack follows the reference (config.eos_token_id).
        let ple_eos: i32 = if fxp.is_some() { cfg.eos_token_id as i32 } else { 248046 };
        let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
        crate::exl3_bench::load_module_pub(&dev)?;
        if fxp.is_some() { fp4l_fwd::load_module(&dev)?; } // FP4L: gpu_fp4l + its KERNEL_BUILD_ID handshake
        // A5 WP20: log the MoE expert-path plan for this model + verify the pinned suh sequence
        // eagerly (before any graph capture can be the first eligible call).
        wp20_announce(&dev, &cfg);
        // TUNE T0 / AUTOTUNE_DESIGN finding §14.1: the served EXL3 modules had no stale-kernel
        // handshake — a fresh binary next to an old exl3_bench.ptx ran silently whenever the kernel
        // NAMES still resolved. Launch the loaded module's kernel_build_id and compare.
        crate::gpu::GpuModel::assert_kernel_build_id(&dev, MODULE)
            .context("EXL3 kernel handshake (exl3_bench)")?;
        w4moe_announce(&dev, &cfg, width); // W4/MOE: opt in + plan the persistent expert kernels (outside any capture)
        pfx1_announce(); // PFX1: live parts + the MoE prefill MT rule, once
        w4qsa_announce(); // W4/QSA: one boot line with the verify gather in use (feature marker)
        // S-A3-i: the selection chain reuses the gpu_batch-side qsa_* kernels. In serve mode
        // gpu.rs has already loaded the module; in bench/probe mode nothing else does, so
        // load it here with just the four names (idempotent guard via get_func).
        if dev.get_func("gpu_batch", "qsa_key_write_b").is_none() {
            let ptx = cudarc::nvrtc::Ptx::from_src(crate::ptx_embed::GPU_BATCH);
            dev.load_ptx(ptx, "gpu_batch", &["qsa_key_write_b", "qsa_score_b", "qsa_block_keys_b", "qsa_score_prefill_b",
                                            "qsa_pool_update_b", "qsa_pool_range_b", "qsa_score_pooled_b",
                                            "qsa_score_pooled_mr"])
                .context("load gpu_batch module (qsa subset)")?;
        }

        // TUNE T0 (§14.1): the gpu_batch half of the EXL3 stale-kernel handshake.
        tune::assert_gpu_batch_build_id(&dev).context("EXL3 kernel handshake (gpu_batch)")?;
        let h = cfg.hidden_size;

        let mut plain: BTreeMap<String, PlainTensor> = BTreeMap::new();
        for s in &pack_shards {
            for (k, mut v) in read_hdr(&format!("{}/{}", pack_dir.trim_end_matches('/'), s))? {
                v.shard = s.clone(); // B32: the index's RELATIVE path (a shard dir keeps its segments under seg/)
                plain.entry(k).or_insert(v);
            }
        }
        // FP4L: the MTP head's plain tensors are the pack's own. A loader step that shadows mtp.* entries from an
        // mtp-*.safetensors file in the model directory (an EXL3 draft-sidecar convention) must not reach an FP4-Local
        // head: its sidecar serves the dense trellis linears only (fp4l_fwd::mtp_sidecar_*), and the pack's MTP norms
        // are fp32 / bf16 where such a file carries bf16. Restored below, after every edit of this map.
        let fx_mtp_plain: Vec<(String, PlainTensor)> = if fxp.is_some() {
            plain.iter().filter(|(k, _)| k.starts_with("mtp.")).map(|(k, v)| (k.clone(), PlainTensor {
                dtype: v.dtype.clone(), offset: v.offset, n_bytes: v.n_bytes, shard: v.shard.clone(), shape: v.shape.clone() })).collect()
        } else { Vec::new() };
        // CF-P1g: the n-gram sidecar by name (the 4.05 pack's index lists neither its trellis nor its aux tensors; the
        // 3.05 index lists them, so `or_insert` keeps those entries as they are).
        {
            let ng = format!("{}/ngram_embedding.safetensors", pack_dir.trim_end_matches('/'));
            if std::path::Path::new(&ng).is_file() {
                for (k, mut v) in read_hdr(&ng)? {
                    v.shard = "ngram_embedding.safetensors".to_string();
                    plain.entry(k).or_insert(v);
                }
            }
        }
        // PCKG-ALL b12: plain-tensor mirrors of the MTP sidecars (exl3.rs 4b/4c redirect the
        // trellis shapes; the plain map here also carries the .weight tensors those heads read).
        {
            let pd = pack_dir.trim_end_matches('/');
            // (a) mixer patch: insert-if-absent (spec 6.3 main-shard preference).
            let mp = format!("{pd}/mtp_hyper_connection_mixer_patch.safetensors");
            if std::path::Path::new(&mp).is_file() {
                for (k, mut v) in read_hdr(&mp)? {
                    v.shard = "mtp_hyper_connection_mixer_patch.safetensors".to_string();
                    plain.entry(k).or_insert(v);
                }
            }
            // (b) selected mtp-* draft sidecar (lexicographically last, same pick as exl3.rs 4b):
            // replaces any existing plain mtp.* entries — the shadow semantics.
            let mut sc: Vec<String> = std::fs::read_dir(pd)?
                .filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().to_string())
                .filter(|f| f.starts_with("mtp-") && f.ends_with(".safetensors"))
                .collect();
            sc.sort();
            if let Some(fname) = sc.last() {
                for (k, mut v) in read_hdr(&format!("{pd}/{fname}"))? {
                    if k.starts_with("mtp.") {
                        v.shard = fname.clone();
                        plain.insert(k, v);
                    }
                }
            }
        }
        let modules: BTreeMap<&str, &ModuleMeta> =
            pack_opt.iter().flat_map(|p| p.modules.iter()).map(|m| (m.name.as_str(), m)).collect();
        if fxp.is_some() {
            plain.retain(|k, _| !k.starts_with("mtp."));
            plain.extend(fx_mtp_plain);
        }
        // FP4L: "does this linear exist" for either container (the MTP-head / fc-cat probes below)
        let has_lin = |stem: &str| -> bool {
            modules.contains_key(stem) || fxp.as_deref().map_or(false, |f| f.has(&format!("{stem}.weight")))
        };

        let t_load = std::time::Instant::now();
        crate::dprintln!(
            "EXL3-FWD load: {} layers ({} gdn / {} attn), {} experts, hc={} hidden={} vocab={}",
            cfg.layer_types.len(),
            cfg.layer_types.iter().filter(|l| matches!(l, crate::qwen::LayerType::LinearAttention)).count(),
            cfg.layer_types.iter().filter(|l| matches!(l, crate::qwen::LayerType::FullAttention)).count(),
            cfg.num_experts,
            cfg.hc_count,
            h,
            cfg.vocab_size
        );

        let get_plain_f16 = |name: &str| -> Result<CudaSlice<u16>> {
            let t = plain.get(name).with_context(|| format!("missing plain tensor {name}"))?;
            // FP4L: an MXFP8 tensor the trunk keeps as a plain f16 weight (GDN in_proj_a / in_proj_b)
            if let (Some(fp), "F8_E4M3") = (fxp.as_deref(), t.dtype.as_str()) {
                return Ok(dev.htod_sync_copy(&fp4l_fwd::mx8_plain_f16(fp, name)?)?);
            }
            let bits = to_f16_bits(&t.read(pack_dir)?, &t.dtype)?;
            Ok(dev.htod_sync_copy(&bits)?)
        };
        // FP4L --fp4l-plain exact: a BF16 tensor's stored bits, uploaded as they are (no conversion).
        let get_plain_bf = |name: &str| -> Result<CudaSlice<u16>> {
            let t = plain.get(name).with_context(|| format!("missing plain tensor {name}"))?;
            anyhow::ensure!(t.dtype == "BF16", "FP4L --fp4l-plain exact: {name} is {} (the stored-bf16 path reads BF16)", t.dtype);
            let raw = t.read(pack_dir)?;
            Ok(dev.htod_sync_copy(&raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect::<Vec<u16>>())?)
        };
        // a trunk-held bf16 weight: the stored bits under --fp4l-plain exact, else the trunk's f16 copy
        let get_plain_w = |name: &str| -> Result<CudaSlice<u16>> {
            if fx_plain { get_plain_bf(name) } else { get_plain_f16(name) }
        };
        // a trunk-held bf16 GEMM weight [n][k] (router, hc mixers): under --fp4l-plain exact the stored bits in the
        // B32 layout the stored-bf16 GEMMs read (fp4l_fwd::bf16_frag, a permutation), else the trunk's f16 copy
        let get_plain_wg = |name: &str, n: usize, k: usize| -> Result<CudaSlice<u16>> {
            if !fx_plain { return get_plain_f16(name); }
            let t = plain.get(name).with_context(|| format!("missing plain tensor {name}"))?;
            anyhow::ensure!(t.dtype == "BF16", "FP4L --fp4l-plain exact: {name} is {} (the stored-bf16 path reads BF16)", t.dtype);
            let bits: Vec<u16> = t.read(pack_dir)?.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            anyhow::ensure!(bits.len() == n * k, "FP4L {name}: {} values, expected [{n}][{k}]", bits.len());
            Ok(dev.htod_sync_copy(&fp4l_fwd::bf16_frag(&bits, n, k))?)
        };
        // FP4L: the f16 copy of a norm, converted quietly — the precision boot line states how FP4L norms are served
        // (fp32 under --fp4l-norms f32: this copy is then only the diagnostics' reference; the rounding counts are in
        // --probe-fp4l FP4L_LEDGER), instead of one "rounded f32->f16" line per tensor (97 of them)
        let get_norm_f16 = |name: &str| -> Result<CudaSlice<u16>> {
            if fxp.is_none() { return get_plain_f16(name); }
            let t = plain.get(name).with_context(|| format!("missing plain tensor {name}"))?;
            let raw = t.read(pack_dir)?;
            let v = if t.dtype == "BF16" { bf16_to_f32(&raw) } else { plain_to_f32(&raw, &t.dtype)? };
            Ok(dev.htod_sync_copy(&v.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect::<Vec<u16>>())?)
        };
        // FP4L --fp4l-norms f32: a norm weight's stored values in fp32 (fp32 as stored; bf16 widened exactly)
        let get_norm_f = |name: &str| -> Result<Option<CudaSlice<f32>>> {
            if !fx_norms { return Ok(None); }
            let t = plain.get(name).with_context(|| format!("missing plain tensor {name}"))?;
            let raw = t.read(pack_dir)?;
            let v = if t.dtype == "BF16" { bf16_to_f32(&raw) } else { plain_to_f32(&raw, &t.dtype)? };
            Ok(Some(dev.htod_sync_copy(&v)?))
        };
        // EXL27B: a plain-residual (hc_lowrank == 0) slot: the layer's RMSNorm weight; down/up are
        // never read (the hc wrappers take the plain-residual dispatch before any mixer).
        let plain_hc = |norm: &str| -> Result<HcDev> {
            Ok(HcDev { norm: get_plain_f16(norm)?, down: dev.alloc_zeros::<u16>(1)?, up: dev.alloc_zeros::<u16>(1)?,
                       winj: None, q8: None, norm_f: None, bf: false })
        };
        let get_f32_bf16 = |name: &str| -> Result<CudaSlice<f32>> {
            let t = plain.get(name).with_context(|| format!("missing plain tensor {name}"))?;
            let raw = t.read(pack_dir)?;
            Ok(dev.htod_sync_copy(&if t.dtype == "BF16" { bf16_to_f32(&raw) } else { plain_to_f32(&raw, &t.dtype)? })?)
        };
        // FP4L --fp4l-mtp-sidecar auto (default, draft-only): the mtp-*.safetensors the pack directory ships (the head ships it to
        // every node, so every rank loads the same file) — its dense linears replace the pack's own mtp.* linears (get_quad /
        // get_quad_sl below); off = the pack's own head. No external-path form (owner 2026-11-01: the pack decides).
        let fx_side: Option<fp4l_fwd::MtpSidecar> = match (fxp.as_deref(), fp4l_fwd::mtp_sidecar_path(pack_dir)) {
            (Some(_), Some(p)) => {
                let sc = fp4l_fwd::mtp_sidecar_open(&p).with_context(|| format!(
                    "FP4-Local MTP sidecar {p} (the pack directory's mtp-*.safetensors)"))?;
                crate::rprintln!("[fp4l] MTP sidecar {p}: {} tensors; its dense mtp.* trellis linears serve the draft passes \
                                  (the routed experts, norms, mixers and router stay the pack's; draft-only; --fp4l-mtp-sidecar off = \
                                  the pack's own)", sc.hdr.len());
                Some(sc)
            }
            (Some(_), None) => {
                let why = if crate::fp4l::mtp_sidecar_in_dir(std::path::Path::new(pack_dir)).is_some() {
                    "--fp4l-mtp-sidecar off; the model directory has one"
                } else { "no mtp-*.safetensors sidecar in the model directory" };
                crate::rprintln!("[fp4l] MTP head: the pack's own ({why})");
                None
            }
            _ => None,
        };
        let side_quad = |name: &str| -> Result<Option<Quad>> {
            match (fx_side.as_ref(), name.starts_with("mtp.")) {
                (Some(sc), true) => fp4l_fwd::mtp_sidecar_quad(&dev, sc, name),
                _ => Ok(None),
            }
        };
        let get_quad = |name: &str| -> Result<Quad> {
            if let Some(q) = side_quad(name)? { return Ok(q); }
            if let Some(fp) = fxp.as_deref() { return fp4l_fwd::fx_quad(&dev, fp, name, None, None, false); }
            let m = modules.get(name).with_context(|| format!("missing quad module {name}"))?;
            let tr: Vec<u16> = m
                .trellis
                .read_bytes(pack_dir)?
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            let (qk, qn) = (m.k as i32, m.n as i32);
            Ok(Quad {
                tr: dev.htod_sync_copy(&tr)?,
                suh: dev.htod_sync_copy(&le_f16_bits(&m.suh.read_bytes(pack_dir)?))?,
                svh: dev.htod_sync_copy(&le_f16_bits(&m.svh.read_bytes(pack_dir)?))?,
                bits: (m.k2 / 2) as i32, k2: m.k2 as i32,
                k: qk,
                n: qn,
                ks: chain_ks(qk, qn),
                lmh: false,
                cnt: dev.htod_sync_copy(&vec![0u32; ((qn.max(0) as usize) / 128).max(1)])?, // W4/DENSE, explicit zero
                rowpar: false,
                fx: None,
            })
        };

        // TP-B: this rank's trunk shard (None = TP=1: every trunk tensor whole) + the TRUNK's
        // per-rank geometry (local head counts; == cfg at TP=1). The MTP head keeps `cfg`.
        let tp_rw: Option<(usize, usize)> = tp.as_ref().map(|t| (t.rank as usize, t.world as usize));
        let tsplit = tp_rw.map(|(rank, world)| xtp::TrunkSplit { rank, world });
        let tc = match tp_rw { Some((_, w)) => xtp::trunk_cfg(&cfg, w)?, None => cfg.clone() };
        // POSTURE1 --exl3-sel-tc: a boot that asked for it but cannot serve it (KV cache not 8-bit, or a per-rank head
        // shape the kernel does not take: Flash-Next at TP=4 holds 6 query heads per replicated KV head) says so once and
        // counts it; the default gather runs (AGENTS 1b-bis.8: fall back, no refusal). Judged on THIS RANK's trunk geometry.
        if sel_tc_on() && tc.has_indexer() {
            let gqa = tc.num_heads / tc.num_kv_heads.max(1);
            let why = if kv_fmt < KV_Q8 { Some(format!("the KV cache is {} (it serves the 8-bit cache, --kv-cache q8)", kv_fmt_name(kv_fmt))) }
                      else if tc.head_dim != 256 || !(gqa == 12 || (gqa == 6 && tc.indexer_compress_ratio == 4)) {
                          Some(format!("this rank holds head_dim {} / {} query heads per key head (indexer blocks of {}; it serves \
head_dim 256 with 12 query heads per key head, or 6 with blocks of 4)", tc.head_dim, gqa, tc.indexer_compress_ratio)) }
                      else { None };
            match why {
                Some(w) => {
                    SEL_TC_FALLBACK.store(1, std::sync::atomic::Ordering::Relaxed);
                    crate::rprintln!("[exl3] --exl3-sel-tc on was asked, but {w}: prompt reading uses today's gather on this boot \
(xq_attn_sel_pq8 / splitk4); /metrics velogb10_exl3_sel_tc_fallback = 1");
                }
                None => crate::rprintln!("[exl3] --exl3-sel-tc on: prompt reading past the sparse-attention switch gathers on the tensor \
cores ({}: this rank holds {gqa} query heads per key head); /metrics velogb10_exl3_sel_tc_rows_total counts the rows{}",
                    if gqa == 6 { "xq_attn_sel_tcu8, 8 prompt tokens per block" } else { "xq_attn_sel_tcu, 4 prompt tokens per block" },
                    if gqa == 6 { ", velogb10_exl3_sel_tc8_calls_total the 6-head launches" } else { "" }),
            }
        }
        // ShardSpec::{Col, Row} quad: `rows` / `cols` = element ranges (128-aligned, refused
        // otherwise), None = whole. A copy of whole 16x16 trellis blocks + the matching suh / svh
        // slices — never a requant (G-T1-0). `rowpar` marks the K-sliced (partial-output) class.
        let get_quad_sl = |name: &str, rows: Option<&[(usize, usize)]>, cols: Option<&[(usize, usize)]>,
                           rowpar: bool| -> Result<Quad> {
            if rows.is_none() && cols.is_none() && !rowpar { if let Some(q) = side_quad(name)? { return Ok(q); } }
            if let Some(fp) = fxp.as_deref() { return fp4l_fwd::fx_quad(&dev, fp, name, rows, cols, rowpar); }
            let m = modules.get(name).with_context(|| format!("missing quad module {name}"))?;
            let tr: Vec<u16> = m.trellis.read_bytes(pack_dir)?.chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            let suh = le_f16_bits(&m.suh.read_bytes(pack_dir)?);
            let svh = le_f16_bits(&m.svh.read_bytes(pack_dir)?);
            let (kb, nb) = (m.k / 16, m.n / 16);
            anyhow::ensure!(tr.len() == kb * nb * 8 * m.k2 && suh.len() == m.k && svh.len() == m.n,
                "{name}: trellis/suh/svh shapes do not match k={} n={} bits={}", m.k, m.n, m.k2 / 2);
            let rr = match rows { Some(r) => xtp::r16(r)?, None => vec![(0, kb)] };
            let cc = match cols { Some(c) => xtp::r16(c)?, None => vec![(0, nb)] };
            anyhow::ensure!(rr.iter().all(|&(_, b)| b <= kb) && cc.iter().all(|&(_, b)| b <= nb),
                "{name}: slice outside k={} n={}", m.k, m.n);
            let t2 = xtp::slice_trellis(&tr, nb, m.k2, &rr, &cc);
            let suh2: Vec<u16> = rr.iter().flat_map(|&(a, b)| (a * 16..b * 16).map(|i| suh[i])).collect();
            let svh2: Vec<u16> = cc.iter().flat_map(|&(a, b)| (a * 16..b * 16).map(|i| svh[i])).collect();
            let (qk, qn) = (suh2.len() as i32, svh2.len() as i32);
            anyhow::ensure!(t2.len() == (qk as usize / 16) * (qn as usize / 16) * 8 * m.k2,
                "{name}: sliced trellis {} words != {}x{} blocks", t2.len(), qk / 16, qn / 16);
            Ok(Quad {
                tr: dev.htod_sync_copy(&t2)?,
                suh: dev.htod_sync_copy(&suh2)?,
                svh: dev.htod_sync_copy(&svh2)?,
                bits: (m.k2 / 2) as i32, k2: m.k2 as i32,
                k: qk,
                n: qn,
                // TP-I #8: a TP-sliced quad may take the TP split-K target (unset = chain_ks)
                ks: if tp_rw.is_some() { xtp::tp_chain_ks(qk, qn) } else { chain_ks(qk, qn) },
                lmh: false,
                cnt: dev.htod_sync_copy(&vec![0u32; ((qn.max(0) as usize) / 128).max(1)])?,
                rowpar,
                fx: None,
            })
        };
        let plain_bits = |name: &str| -> Result<Vec<u16>> {
            let t = plain.get(name).with_context(|| format!("missing plain tensor {name}"))?;
            if let (Some(fp), "F8_E4M3") = (fxp.as_deref(), t.dtype.as_str()) {
                return fp4l_fwd::mx8_plain_f16(fp, name); // FP4L: TP-sliced a / b rows
            }
            to_f16_bits(&t.read(pack_dir)?, &t.dtype)
        };
        let plain_f32_bf16 = |name: &str| -> Result<Vec<f32>> {
            let t = plain.get(name).with_context(|| format!("missing plain tensor {name}"))?;
            let raw = t.read(pack_dir)?;
            if t.dtype == "BF16" { Ok(bf16_to_f32(&raw)) } else { plain_to_f32(&raw, &t.dtype) }
        };

        // 1.66-class packs (exl3 1.6.3): no plain embed_tokens.weight; the token table is
        // trellis rings + signs, decoded at load by the ngram codec (exl3_ngram_embed.rs),
        // bit-exact with the wheel's TableEmbedding dequant. Counter: ngram_embed_rows.
        let embed = if let Some(emb_name) = plain.keys().find(|k| k.ends_with("embed_tokens.weight")) {
            let emb_name = emb_name.clone();
            get_plain_w(&emb_name)? // FP4L merge: stored bf16 bits for FP4-Local; == get_plain_f16 for EXL3
        } else {
            let tr = plain
                .keys()
                .find(|k| k.ends_with("embed_tokens.trellis"))
                .context("embed_tokens.weight not found and no embed_tokens.trellis (ngram) table")?;
            let (tr_name, base) = (tr.clone(), tr.trim_end_matches(".trellis").to_string());
            let sg_name = format!("{base}.signs");
            let t_tr = plain.get(&tr_name).context("ngram embed: trellis gone?")?;
            let t_sg = plain.get(&sg_name)
                .with_context(|| format!("ngram embed: missing signs tensor {sg_name}"))?;
            let rows = t_tr.shape.first().copied()
                .context("ngram embed: trellis shape missing from shard header")?;
            let row_words = t_tr.shape.get(1).copied()
                .context("ngram embed: trellis is not 2-D")?;
            let (sg_r, sg_c) = (t_sg.shape.first().copied().unwrap_or(0), t_sg.shape.get(1).copied().unwrap_or(0));
            if !(sg_r * sg_c > 0 && row_words > 0 && rows > 0) {
                bail!("ngram embed: bad shapes trellis [{rows},{row_words}] signs [{sg_r},{sg_c}]");
            }
            let t0 = std::time::Instant::now();
            // trellis rings are I16: the u16 words ARE the payload (not float values);
            // reinterpret LE words directly. signs is F16 -> pass-through bits.
            let tr_raw = t_tr.read(pack_dir)?;
            let tr_bits: Vec<u16> = if t_tr.dtype == "I16" {
                tr_raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
            } else {
                to_f16_bits(&tr_raw, &t_tr.dtype)?
            };
            let sg_bits = to_f16_bits(&t_sg.read(pack_dir)?, &t_sg.dtype)?;
            let decoded = crate::exl3_ngram_embed::dequant_embed(
                &tr_bits, &sg_bits, rows, row_words, sg_r, sg_c)?;
            crate::metrics::exl3_ngram_embed_decoded(rows as u64);
            eprintln!("[exl3-load] ngram token-embed decode: {rows} rows x {row_words} words,                        {} groups x {sg_c}, K={:?} -> f16 in {:.1}s",
                      sg_r, ((row_words / sg_r) - 1) * 16 / sg_c, t0.elapsed().as_secs_f32());
            dev.htod_sync_copy(&decoded)?
        };

        let mut layers = Vec::new();
        // S-A3-i: per-full-attn-layer raw-key planes, in layer order (attn layers only).
        let mut qsa_keys_vec: Vec<CudaSlice<u16>> = Vec::new();
        let mut qsa_pool_vec: Vec<CudaSlice<u16>> = Vec::new();
        let pool_on = crate::opts::var(crate::opt!("qsa-pool")).as_deref() != Ok("0")
            && dev.get_func("gpu_batch", "qsa_pool_update_b").is_some()
            && dev.get_func("gpu_batch", "qsa_score_pooled_b").is_some();
        let mut qsa_keys_plane: Option<CudaSlice<u16>> = None;
        let mut qsa_pool_plane: Option<CudaSlice<u16>> = None;
        let mut bytes_round = 0u64;
        // Stacked-experts loader, shared by the trunk layers and the MTP draft head (S-A3-e M1).
        let mk_moe = |p: String, tag: String, ep_rank: Option<(usize, usize)>| -> Result<MoeLayer> {
            let ne = cfg.num_experts;
            let mi = cfg.moe_intermediate_size;
            // TP-A: this rank's routed experts in LOCAL stacked order (all of them when replicated).
            // TP-I #6: the deal may be per layer (--ep-deal=freq); the MTP head is never EP.
            let li: usize = tag.strip_prefix('L').and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
            if dense_mlp {
                // EXL27B: the dense MLP as three plain quads (MoeLayer::dense), each with its own suh /
                // svh / bit width. EXL27C TP-B: under a rank pair the quads are rank-sliced TRUE
                // partials — gate/up along n (this rank's intermediate columns), down along k with
                // rowpar (its partial of the full h; chain_rowpar sums the ranks in FP32, one f16
                // rounding + svh after the reduce). At TP=1 nothing is sliced: every path unchanged.
                let (mut g, mut u, mut d, mi_r) = match ep_rank {
                    Some((rank, world)) => {
                        anyhow::ensure!(mi % world == 0 && (mi / world) % 128 == 0,
                                        "dense MLP TP slice @{tag}{shard_tag}: intermediate {mi} not \
                                         divisible into 128-aligned {world}-way slices");
                        let span = mi / world;
                        let r0 = rank * span;
                        static DTP_ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                        if !DTP_ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                            eprintln!("[exl3-load] dense MLP TP-sharded: gate/up n-slice + down k-slice \
                                       rowpar partials (world={world})");
                        }
                        (get_quad_sl(&format!("{p}.gate_proj"), None, Some(&[(r0, r0 + span)]), false)?,
                         get_quad_sl(&format!("{p}.up_proj"), None, Some(&[(r0, r0 + span)]), false)?,
                         get_quad_sl(&format!("{p}.down_proj"), Some(&[(r0, r0 + span)]), None, true)?, span)
                    }
                    None => (get_quad(&format!("{p}.gate_proj"))?, get_quad(&format!("{p}.up_proj"))?,
                             get_quad(&format!("{p}.down_proj"))?, mi),
                };
                anyhow::ensure!(g.n as usize == mi_r && u.n as usize == mi_r && d.k as usize == mi_r
                                && g.k == u.k && d.n == g.k, "dense MLP geometry mismatch @{tag}{shard_tag}");
                for q in [&mut g, &mut u, &mut d] { q.ks = dense_chain_ks(q.k, q.n); }
                let one = || -> Result<CudaSlice<u16>> { Ok(dev.htod_sync_copy(&[0u16])?) };
                let bits = g.bits as usize;
                let gk2 = g.k2 as usize;
                return Ok(MoeLayer {
                    fx: None, router_bf: false, sg_bf: None, // FP4L fields
                    router: one()?, sg: one()?, gu_tr: one()?, d_tr: one()?,
                    suh_gu: one()?, svh_gu: one()?, suh_d: one()?, svh_d: one()?,
                    sh_gate: g, sh_up: u, sh_down: d, gu_words: 0, d_words: 0, bits, k2: gk2, ep: None, li, dense: true,
                    split: None,
                    // dense MLP: the device-routed MoE arm never runs (dense_mlp_prefill returns first)
                    pf_offs_gu: dev.htod_sync_copy(&[0u64])?, pf_offs_d: dev.htod_sync_copy(&[0u64])?,
                });
            }
            if split_moe {
                // EXL27B: Qwen3.5-class routed experts — gate, up and down each keep their own suh / svh
                // (the FN loader below fuses gate|up under the gate's suh). EP (2026-10-16, EXL27C):
                // each rank stacks only its LOCAL experts (xtp::ep_local_experts, deal from `li`);
                // the shared expert is dealt like the FN branch (gate/up col-slice, down row-slice;
                // its partial folds into the EP combine). The MTP head loads with ep_rank=None
                // (replicated on every rank — never EP). TP=1: local == 0..ne, bit-identical.
                let local: Vec<usize> = match ep_rank {
                    Some((r, w)) => xtp::ep_local_experts(ne, r, w, li)?,
                    None => (0..ne).collect(),
                };
                let sh_split: Option<(usize, usize)> = match ep_rank {
                    Some((r, w)) if crate::opts::var(crate::opt!("tp-shared-repl")).is_err() =>
                        Some(xtp::TrunkSplit { rank: r, world: w }.shared(mi)?),
                    _ => None,
                };
                {
                    static EP_SPLIT_ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                    if ep_rank.is_some() && !EP_SPLIT_ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        eprintln!("[exl3-load] split-MoE EP deal: {} of {} experts on this rank (shared expert split: {})",
                                  local.len(), ne, sh_split.is_some());
                    }
                }
                let h = cfg.hidden_size;
                let rd16 = |b: Vec<u8>| -> Vec<u16> { b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect() };
                let (mut g_tr, mut u_tr, mut d_tr) = (Vec::new(), Vec::new(), Vec::new());
                let (mut suh_g, mut suh_u, mut suh_d) = (Vec::with_capacity(ne * h), Vec::with_capacity(ne * h), Vec::with_capacity(ne * mi));
                let (mut svh_g, mut svh_u, mut svh_d) = (Vec::with_capacity(ne * mi), Vec::with_capacity(ne * mi), Vec::with_capacity(ne * h));
                let (mut g_words, mut d_words, mut ebits, mut ek2) = (0u64, 0u64, 0usize, 0usize);
                for &e in &local {
                    let q = |w: &str| -> Result<&ModuleMeta> {
                        modules.get(format!("{p}.experts.{e}.{w}").as_str()).copied()
                            .with_context(|| format!("expert {e} {w} missing @{tag}"))
                    };
                    let (g, u, d) = (q("gate_proj")?, q("up_proj")?, q("down_proj")?);
                    anyhow::ensure!(g.k == h && u.k == h && g.n == mi && u.n == mi && d.k == mi && d.n == h,
                                    "expert geometry mismatch @{tag} e{e}");
                    // K35: compare k2 (K = 3 and K = 3.5 share k2 / 2 = 3 but not a ring width); a layer that
                    // mixes expert widths is the mtp-2.6 sidecar class, not served on this line.
                    anyhow::ensure!(g.k2 == u.k2 && g.k2 == d.k2 && (ek2 == 0 || ek2 == g.k2),
                                    "expert widths differ within a layer @{tag} e{e} (gate k2 {} / up k2 {} / down k2 {} / layer k2 {ek2}): mixed expert widths within one layer are not served: this build decodes each MoE layer at ONE expert width (packs that allocate widths per expert, e.g. SAGE-4.77, need the planned mixed-width support) — serve a pack with one expert width per layer",
                                    g.k2, u.k2, d.k2);
                    ebits = exl3_kernel_bits(g.k2);
                    ek2 = g.k2 as usize;
                    g_tr.extend(rd16(g.trellis.read_bytes(pack_dir)?));
                    u_tr.extend(rd16(u.trellis.read_bytes(pack_dir)?));
                    d_tr.extend(rd16(d.trellis.read_bytes(pack_dir)?));
                    suh_g.extend(le_f16_bits(&g.suh.read_bytes(pack_dir)?));
                    suh_u.extend(le_f16_bits(&u.suh.read_bytes(pack_dir)?));
                    suh_d.extend(le_f16_bits(&d.suh.read_bytes(pack_dir)?));
                    svh_g.extend(le_f16_bits(&g.svh.read_bytes(pack_dir)?));
                    svh_u.extend(le_f16_bits(&u.svh.read_bytes(pack_dir)?));
                    svh_d.extend(le_f16_bits(&d.svh.read_bytes(pack_dir)?));
                    g_words = ((h / 16) * (mi / 16) * 16 * g.k2 / 2) as u64;
                    d_words = ((mi / 16) * (h / 16) * 16 * d.k2 / 2) as u64;
                }
                let nl = local.len();
                anyhow::ensure!(g_tr.len() == nl * g_words as usize && u_tr.len() == g_tr.len()
                                && d_tr.len() == nl * d_words as usize, "expert trellis sizes @{tag}");
                let offs_e: Vec<u64> = (0..nl as u64).map(|i| i * g_words).collect();
                let offs_d: Vec<u64> = (0..nl as u64).map(|i| i * d_words).collect();
                return Ok(MoeLayer {
                    router: get_plain_f16(&format!("{p}.gate.weight"))?,
                    sg: get_plain_f16(&format!("{p}.shared_expert_gate.weight"))?,
                    gu_tr: dev.htod_sync_copy(&g_tr)?,
                    d_tr: dev.htod_sync_copy(&d_tr)?,
                    suh_gu: dev.htod_sync_copy(&suh_g)?,
                    svh_gu: dev.htod_sync_copy(&svh_g)?,
                    suh_d: dev.htod_sync_copy(&suh_d)?,
                    svh_d: dev.htod_sync_copy(&svh_d)?,
                    // TP-B: shared expert dealt on EP layers (gate/up col-slice, down row-slice);
                    // its partial needs NO extra reduce — each rank adds sgv * its own partial
                    // inside the EP combine (xtp::EpShard::shared_split; the FN pattern).
                    sh_gate: match sh_split { Some(r) => get_quad_sl(&format!("{p}.shared_expert.gate_proj"), None, Some(&[r]), false)?,
                                              None => get_quad(&format!("{p}.shared_expert.gate_proj"))? },
                    sh_up: match sh_split { Some(r) => get_quad_sl(&format!("{p}.shared_expert.up_proj"), None, Some(&[r]), false)?,
                                            None => get_quad(&format!("{p}.shared_expert.up_proj"))? },
                    sh_down: match sh_split { Some(r) => get_quad_sl(&format!("{p}.shared_expert.down_proj"), Some(&[r]), None, false)?,
                                              None => get_quad(&format!("{p}.shared_expert.down_proj"))? },
                    gu_words: g_words, d_words, bits: ebits, k2: ek2, li, dense: false,
                    pf_offs_gu: dev.htod_sync_copy(&(0..local.len() as u64).map(|e| e * g_words).collect::<Vec<u64>>())?,
                    pf_offs_d: dev.htod_sync_copy(&(0..local.len() as u64).map(|e| e * d_words).collect::<Vec<u64>>())?,
                    ep: match ep_rank {
                        Some((r, w)) => {
                            let mut e = xtp::EpShard::new(&dev, ne, r, w, &local, &tag,
                                                          g_tr.len(), d_tr.len(), suh_g.len(), svh_g.len(),
                                                          suh_d.len(), svh_d.len(), g_words, d_words,
                                                          cfg.hidden_size, mi, true)?;
                            e.shared_split = sh_split.is_some();
                            Some(e)
                        }
                        None => None,
                    },
                    split: Some(SplitGu {
                        up_tr: dev.htod_sync_copy(&u_tr)?,
                        suh_u: dev.htod_sync_copy(&suh_u)?,
                        svh_u: dev.htod_sync_copy(&svh_u)?,
                        offs_e: dev.htod_sync_copy(&offs_e)?,
                        offs_d: dev.htod_sync_copy(&offs_d)?,
                    }),
                    fx: None, router_bf: false, sg_bf: None,
                });
            }
            if let Some(fp) = fxp.as_deref() {
                // FP4L: NVFP4 routed experts (csf decoded) stacked in local order; the shared expert is
                // MXFP8 (FP4L Quads, dealt like the FN branch below on EP layers).
                let local: Vec<usize> = match ep_rank {
                    Some((r, w)) => xtp::ep_local_experts(ne, r, w, li)?,
                    None => (0..ne).collect(),
                };
                let sh_split: Option<(usize, usize)> = match ep_rank {
                    Some((r, w)) if crate::opts::var(crate::opt!("tp-shared-repl")).is_err() =>
                        Some(xtp::TrunkSplit { rank: r, world: w }.shared(mi)?),
                    _ => None,
                };
                let fxm = fp4l_fwd::fx_experts(&dev, fp, &p, &local, cfg.hidden_size, mi)
                    .with_context(|| format!("FP4L experts @{tag}{shard_tag}"))?;
                let one = || -> Result<CudaSlice<u16>> { Ok(dev.htod_sync_copy(&[0u16])?) };
                let n_loc = local.len();
                // --fp4l-plain exact: the router as stored (bf16) and the shared-expert gate through xf_sg_gate:
                // `sg` then holds the channel constants the trunk's combine kernels pair with the gate channels
                let sgn = format!("{p}.shared_expert_gate.weight");
                return Ok(MoeLayer {
                    router: get_plain_wg(&format!("{p}.gate.weight"), cfg.num_experts, cfg.hidden_size)?,
                    sg: if fx_plain { dev.htod_sync_copy(&fp4l_fwd::sg_channel_constants(cfg.hidden_size))? } else { get_plain_f16(&sgn)? },
                    router_bf: fx_plain,
                    sg_bf: if fx_plain { Some(get_plain_bf(&sgn)?) } else { None },
                    gu_tr: one()?, d_tr: one()?, suh_gu: one()?, svh_gu: one()?, suh_d: one()?, svh_d: one()?,
                    sh_gate: match sh_split { Some(r) => get_quad_sl(&format!("{p}.shared_expert.gate_proj"), None, Some(&[r]), false)?,
                                              None => get_quad(&format!("{p}.shared_expert.gate_proj"))? },
                    sh_up: match sh_split { Some(r) => get_quad_sl(&format!("{p}.shared_expert.up_proj"), None, Some(&[r]), false)?,
                                            None => get_quad(&format!("{p}.shared_expert.up_proj"))? },
                    sh_down: match sh_split { Some(r) => get_quad_sl(&format!("{p}.shared_expert.down_proj"), Some(&[r]), None, false)?,
                                              None => get_quad(&format!("{p}.shared_expert.down_proj"))? },
                    // the route kernels' per-expert offsets are unused by the FP4L kernels (they index by idxmap)
                    gu_words: 0, d_words: 0, bits: 0, k2: 0, li, dense: false, split: None,
                    // trellis-only per-layer prefill offset tables: the FP4L prefill is its own copy and never reads them
                    pf_offs_gu: dev.htod_sync_copy(&[0u64])?, pf_offs_d: dev.htod_sync_copy(&[0u64])?,
                    ep: match ep_rank {
                        Some((r, w)) => {
                            let mut e = xtp::EpShard::new_fx(&dev, ne, r, w, &local, &tag, n_loc)?;
                            e.shared_split = sh_split.is_some();
                            Some(e)
                        }
                        None => None,
                    },
                    fx: Some(fxm),
                });
            }
            let local: Vec<usize> = match ep_rank {
                Some((r, w)) => xtp::ep_local_experts(ne, r, w, li)?,
                None => (0..ne).collect(),
            };
            // TP-B: shared-expert intermediate range of this rank (EP layers only; --tp-shared-repl=1
            // keeps TP-A's replicated shared expert — a diagnostic A/B, rides TpConfig.exl3_env).
            let sh_split: Option<(usize, usize)> = match ep_rank {
                Some((r, w)) if crate::opts::var(crate::opt!("tp-shared-repl")).is_err() =>
                    Some(xtp::TrunkSplit { rank: r, world: w }.shared(mi)?),
                _ => None,
            };
            let mut gu_words = 0u64;
            let mut d_words = 0u64;
            let (mut ebits, mut ek2) = (0usize, 0usize);
            let mut gu_tr_h: Vec<u16> = Vec::new();
            let mut d_tr_h: Vec<u16> = Vec::new();
            let mut suh_gu_h: Vec<u16> = Vec::with_capacity(ne * 2560);
            let mut svh_gu_h: Vec<u16> = Vec::with_capacity(ne * mi * 2);
            let mut suh_d_h: Vec<u16> = Vec::with_capacity(ne * mi);
            let mut svh_d_h: Vec<u16> = Vec::with_capacity(ne * 2560);
            for &e in &local {
                let gname = format!("{p}.experts.{e}.gate_proj");
                let g = modules.get(gname.as_str()).with_context(|| format!("expert {e} gate ({gname}) missing @{tag}{shard_tag}"))?;
                let u = modules
                    .get(format!("{p}.experts.{e}.up_proj").as_str())
                    .with_context(|| format!("expert {e} up ({p}.experts.{e}.up_proj) missing @{tag}{shard_tag}"))?;
                let d = modules
                    .get(format!("{p}.experts.{e}.down_proj").as_str())
                    .with_context(|| format!("expert {e} down ({p}.experts.{e}.down_proj) missing @{tag}{shard_tag}"))?;
                if g.k2 != u.k2 || g.n != mi || u.n != mi || d.k != mi {
                    bail!("expert geometry mismatch @{tag} e{e} (gate k2 {} / up k2 {}; n {} / {} vs mi {mi}; down k {})", g.k2, u.k2, g.n, u.n, d.k);
                }
                let gw = 16 * g.k2 / 2;
                let kb = g.k / 16;
                let gnb = g.n / 16;
                let g_tr: Vec<u16> = g
                    .trellis
                    .read_bytes(pack_dir)?
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect();
                let u_tr: Vec<u16> = u
                    .trellis
                    .read_bytes(pack_dir)?
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect();
                for r in 0..kb {
                    gu_tr_h.extend_from_slice(&g_tr[r * gnb * gw..(r + 1) * gnb * gw]);
                    gu_tr_h.extend_from_slice(&u_tr[r * gnb * gw..(r + 1) * gnb * gw]);
                }
                d_tr_h.extend(
                    d.trellis
                        .read_bytes(pack_dir)?
                        .chunks_exact(2)
                        .map(|c| u16::from_le_bytes([c[0], c[1]])),
                );
                suh_gu_h.extend(le_f16_bits(&g.suh.read_bytes(pack_dir)?));
                svh_gu_h.extend(le_f16_bits(&g.svh.read_bytes(pack_dir)?));
                svh_gu_h.extend(le_f16_bits(&u.svh.read_bytes(pack_dir)?));
                suh_d_h.extend(le_f16_bits(&d.suh.read_bytes(pack_dir)?));
                svh_d_h.extend(le_f16_bits(&d.svh.read_bytes(pack_dir)?));
                gu_words = (kb * gnb * 2 * gw) as u64;
                d_words = ((d.k / 16) * (d.n / 16) * 16 * d.k2 / 2) as u64;
                // K35: compare k2, not k2 / 2 (K = 3 vs K = 3.5); mixed widths inside one layer are not served here.
                if (ek2 != 0 && (ek2 != g.k2 || ek2 != d.k2)) || g.k2 != u.k2 || g.k2 != d.k2 {
                    bail!("expert widths differ within a layer @{tag} e{e} (gate k2 {} / up k2 {} / down k2 {} / layer k2 {ek2}): mixed expert widths within one layer are not served: this build decodes each MoE layer at ONE expert width (packs that allocate widths per expert, e.g. SAGE-4.77, need the planned mixed-width support) — serve a pack with one expert width per layer", g.k2, u.k2, d.k2);
                }
                ebits = exl3_kernel_bits(g.k2);
                ek2 = g.k2 as usize;
            }
            Ok(MoeLayer {
                router: get_plain_f16(&format!("{p}.gate.weight"))?,
                sg: get_plain_f16(&format!("{p}.shared_expert_gate.weight"))?,
                gu_tr: dev.htod_sync_copy(&gu_tr_h)?,
                d_tr: dev.htod_sync_copy(&d_tr_h)?,
                suh_gu: dev.htod_sync_copy(&suh_gu_h)?,
                svh_gu: dev.htod_sync_copy(&svh_gu_h)?,
                suh_d: dev.htod_sync_copy(&suh_d_h)?,
                svh_d: dev.htod_sync_copy(&svh_d_h)?,
                // TP-B: the shared expert's intermediate is dealt 384 | 256 (uneven 128-block, G-T1-0)
                // on EP layers — gate/up column slices, down row slice. Its partial needs NO extra
                // reduce: each rank adds sgv * its own partial inside the EP combine, and the MoE
                // layer's one all-reduce sums them (xtp::EpShard::shared_split).
                sh_gate: match sh_split { Some(r) => get_quad_sl(&format!("{p}.shared_expert.gate_proj"), None, Some(&[r]), false)?,
                                          None => get_quad(&format!("{p}.shared_expert.gate_proj"))? },
                sh_up: match sh_split { Some(r) => get_quad_sl(&format!("{p}.shared_expert.up_proj"), None, Some(&[r]), false)?,
                                        None => get_quad(&format!("{p}.shared_expert.up_proj"))? },
                sh_down: match sh_split { Some(r) => get_quad_sl(&format!("{p}.shared_expert.down_proj"), Some(&[r]), None, false)?,
                                          None => get_quad(&format!("{p}.shared_expert.down_proj"))? },
                gu_words,
                d_words,
                pf_offs_gu: dev.htod_sync_copy(&(0..local.len() as u64).map(|e| e * gu_words).collect::<Vec<u64>>())?,
                pf_offs_d: dev.htod_sync_copy(&(0..local.len() as u64).map(|e| e * d_words).collect::<Vec<u64>>())?,
                bits: ebits,
                k2: ek2,
                li,
                dense: false,
                split: None,
                fx: None,
                router_bf: false,
                sg_bf: None,
                ep: match ep_rank {
                    Some((r, w)) => {
                        let mut e = xtp::EpShard::new(&dev, ne, r, w, &local, &tag,
                                                      gu_tr_h.len(), d_tr_h.len(), suh_gu_h.len(), svh_gu_h.len(),
                                                      suh_d_h.len(), svh_d_h.len(), gu_words, d_words,
                                                      cfg.hidden_size, mi, false)?;
                        e.shared_split = sh_split.is_some();
                        Some(e)
                    }
                    None => None,
                },
            })
        };
        // BASICS (owner 2026-11-02): the layer loop is the bulk of the read; a percentage at the normal level
        let load_prog = crate::boot_report::Progress::new("loading weights", cfg.layer_types.len() as u64,
                                                          crate::boot_report::Unit::Items("layers"));
        for (li, lt) in cfg.layer_types.iter().enumerate() {
            let p = format!("model.language_model.layers.{li}");
            let mut mk_hc = |base: String, inject: bool| -> Result<HcDev> {
                bytes_round += 2 * (320 * 10240 + 10240 * 320) as u64 * 2;
                Ok(HcDev {
                    norm: get_norm_f16(&format!("{base}.hc_norm.weight"))?,
                    down: get_plain_wg(&format!("{base}.input_mix_weight_down.weight"), cfg.hc_lowrank, cfg.hidden_size * cfg.hc_count.max(1))?,
                    up: get_plain_wg(&format!("{base}.input_mix_weight_up.weight"), cfg.hidden_size * cfg.hc_count.max(1), cfg.hc_lowrank)?,
                    winj: if inject {
                        Some(get_plain_w(&format!("{base}.block_inject_weight.weight"))?)
                    } else {
                        None
                    },
                    q8: None,
                    norm_f: get_norm_f(&format!("{base}.hc_norm.weight"))?,
                    bf: fx_plain,
                })
            };
            let (hc_attn, hc_mlp) = if cfg.hc_lowrank == 0 {
                // EXL27B: plain residual (Qwen3.5-class) — the slots carry the layer's RMSNorms
                (plain_hc(&format!("{p}.input_layernorm.weight"))?, plain_hc(&format!("{p}.post_attention_layernorm.weight"))?)
            } else {
                (mk_hc(format!("{p}.attn_hyper_connection"), true)?, mk_hc(format!("{p}.mlp_hyper_connection"), true)?)
            };

            let mixer = match lt {
                crate::qwen::LayerType::LinearAttention if tsplit.is_some() => {
                    // TP-B: this rank's GDN heads (design §1.3): in_qkv re-tiled [q | k | v] of its
                    // key/value head groups, in_z its value heads' columns, out_proj their ROWS
                    // (row-parallel partial -> one FP32 all-reduce), a|b rows + A_log/dt_bias of its
                    // value heads, conv1d over its qkv channels; the gated-norm weight is per-dim.
                    let ts = tsplit.as_ref().unwrap();
                    bytes_round += (2560 * 10240 + 2560 * 6144 + 6144 * 2560) as u64 * 5 / 8 / ts.world as u64;
                    let qkv = ts.gdn_qkv(&cfg);
                    let vr = ts.gdn_v(&cfg);
                    let (h0, h1) = ts.gdn_heads(&cfg);
                    let rows_of = |v: Vec<u16>, rw: usize| -> Vec<u16> { v[h0 * rw..h1 * rw].to_vec() };
                    let a_w = rows_of(plain_bits(&format!("{p}.linear_attn.in_proj_a.weight"))?, h);
                    let b_w = rows_of(plain_bits(&format!("{p}.linear_attn.in_proj_b.weight"))?, h);
                    let a_log = plain_f32_bf16(&format!("{p}.linear_attn.A_log"))?;
                    let dt_bias = plain_f32_bf16(&format!("{p}.linear_attn.dt_bias"))?;
                    let conv_w = plain_f32_bf16(&format!("{p}.linear_attn.conv1d.weight"))?;
                    let ck = cfg.conv_kernel;
                    let conv_full = cfg.key_dim() * 2 + cfg.value_dim();
                    anyhow::ensure!(a_log.len() == cfg.lin_num_v_heads && dt_bias.len() == cfg.lin_num_v_heads
                                    && conv_w.len() == conv_full * ck,
                        "TP-B GDN L{li}: A_log {} dt_bias {} conv {} vs nv {} conv_dim*ck {}",
                        a_log.len(), dt_bias.len(), conv_w.len(), cfg.lin_num_v_heads, conv_full * ck);
                    let conv_l: Vec<f32> = qkv.iter().flat_map(|&(a, b)| conv_w[a * ck..b * ck].iter().copied()).collect();
                    let norm_bits = plain_bits(&format!("{p}.linear_attn.norm.weight"))?;
                    anyhow::ensure!(norm_bits.len() == cfg.lin_v_dim,
                        "TP-B GDN L{li}: gated-norm weight len {} != v_dim {} (per-head norms would need a slice)",
                        norm_bits.len(), cfg.lin_v_dim);
                    let g = GdnLayer {
                        in_qkv: get_quad_sl(&format!("{p}.linear_attn.in_proj_qkv"), None, Some(&qkv), false)?,
                        in_z: get_quad_sl(&format!("{p}.linear_attn.in_proj_z"), None, Some(&[vr]), false)?,
                        out_proj: get_quad_sl(&format!("{p}.linear_attn.out_proj"), Some(&[vr]), None, true)?,
                        a_w: dev.htod_sync_copy(&a_w)?,
                        b_w: dev.htod_sync_copy(&b_w)?,
                        a_log: dev.htod_sync_copy(&a_log[h0..h1])?,
                        dt_bias: dev.htod_sync_copy(&dt_bias[h0..h1])?,
                        conv_w: dev.htod_sync_copy(&conv_l)?,
                        norm: dev.htod_sync_copy(&norm_bits)?,
                        ab_fx: match fxp.as_deref() {
                            Some(fp) => { let (a, b) = fp4l_fwd::fx_gdn_ab(&dev, fp, &p, Some((h0, h1)))?;
                                          let ws = dev.alloc_zeros::<f32>(fp4l_fwd::ab_ws_len(&a, &b))?;
                                          Some(Box::new((a, b, ws))) }
                            None => None,
                        },
                    };
                    // load-time shape asserts (design §1.3: the boot fails on a mismatch)
                    let conv_l_dim = tc.key_dim() * 2 + tc.value_dim();
                    anyhow::ensure!(g.in_qkv.n as usize == conv_l_dim && g.in_z.n as usize == tc.value_dim()
                                    && g.out_proj.k as usize == tc.value_dim() && g.out_proj.n as usize == h
                                    && g.a_w.len() == tc.lin_num_v_heads * h && g.conv_w.len() == conv_l_dim * ck,
                        "TP-B GDN L{li}: per-rank shapes (qkv {} z {} out {}x{} a {} conv {}) != local geometry \
                         (conv_dim {conv_l_dim}, value_dim {}, nv {}) — refusing to boot",
                        g.in_qkv.n, g.in_z.n, g.out_proj.k, g.out_proj.n, g.a_w.len(), g.conv_w.len(),
                        tc.value_dim(), tc.lin_num_v_heads);
                    Mixer::Gdn(g)
                }
                crate::qwen::LayerType::LinearAttention => {
                    bytes_round += (2560 * 10240 + 2560 * 6144 + 6144 * 2560) as u64 * 5 / 8;
                    Mixer::Gdn(GdnLayer {
                        in_qkv: get_quad(&format!("{p}.linear_attn.in_proj_qkv"))?,
                        in_z: get_quad(&format!("{p}.linear_attn.in_proj_z"))?,
                        out_proj: get_quad(&format!("{p}.linear_attn.out_proj"))?,
                        a_w: get_plain_f16(&format!("{p}.linear_attn.in_proj_a.weight"))?,
                        b_w: get_plain_f16(&format!("{p}.linear_attn.in_proj_b.weight"))?,
                        a_log: get_f32_bf16(&format!("{p}.linear_attn.A_log"))?,
                        dt_bias: get_f32_bf16(&format!("{p}.linear_attn.dt_bias"))?,
                        conv_w: get_f32_bf16(&format!("{p}.linear_attn.conv1d.weight"))?,
                        norm: get_plain_f16(&format!("{p}.linear_attn.norm.weight"))?,
                        ab_fx: match fxp.as_deref() {
                            Some(fp) => { let (a, b) = fp4l_fwd::fx_gdn_ab(&dev, fp, &p, None)?;
                                          let ws = dev.alloc_zeros::<f32>(fp4l_fwd::ab_ws_len(&a, &b))?;
                                          Some(Box::new((a, b, ws))) }
                            None => None,
                        },
                    })
                }
                crate::qwen::LayerType::FullAttention => {
                    if tsplit.is_none() { bytes_round += (2560 * 13312 + 6144 * 2560) as u64 * 5 / 8; }
                    let qn_t = plain.get(&format!("{p}.self_attn.q_norm.weight")).context("q_norm")?;
                    let kn_t = plain.get(&format!("{p}.self_attn.k_norm.weight")).context("k_norm")?;
                    let qn = to_f16_bits(&qn_t.read(pack_dir)?, &qn_t.dtype)?;
                    let kn = to_f16_bits(&kn_t.read(pack_dir)?, &kn_t.dtype)?;
                    let qnkn: Vec<u16> = qn.into_iter().chain(kn).collect();
                    // S-A3-i: load the QSA indexer (the qwen4_exp native attention)
                    // when the pack carries one. The raw-key plane is per-slot bf16
                    // [width][max_pos][hdx], explicitly zeroed (alloc_zeros does NOT).
                    let idx = if cfg.has_indexer() {
                        let f32_of = |name: &str| -> Result<CudaSlice<f32>> {
                            let t = plain.get(&format!("{p}.self_attn.indexer.{name}.weight"))
                                .with_context(|| format!("indexer {name}"))?;
                            let bits = to_f16_bits(&t.read(pack_dir)?, &t.dtype)?;
                            let v: Vec<f32> = bits.iter().map(|b| h16_to_f32(*b)).collect();
                            Ok(dev.htod_sync_copy(&v)?)
                        };
                        Some(IndexerDev {
                            qk: get_quad(&format!("{p}.self_attn.indexer.index_qk_proj"))?,
                            qn: f32_of("q_layernorm")?,
                            kn: f32_of("k_layernorm")?,
                            params: None, // built once the rope tables exist (after cos_tab)
                        })
                    } else { None };
                    if idx.is_some() {
                        let plane = width * max_pos * cfg.indexer_head_dim;
                        qsa_keys_plane = Some(dev.htod_sync_copy(&vec![0u16; plane])?);
                        crate::metrics::kv_pool_add_bytes(1, 2 * plane as u64); // issue #24 (bf16 plane)
                        if pool_on {
                            // Issue #41: exactly the kernels' per-lane block stride (floor; the kernels never
                            // write a partial last block), so the spill / prefix-pool copies agree with them.
                            let pl = qsa_pool_plane_elems(width, max_pos, cfg.indexer_compress_ratio, cfg.indexer_head_dim);
                            qsa_pool_plane = Some(dev.htod_sync_copy(&vec![0u16; pl])?);
                            crate::metrics::kv_pool_add_bytes(1, 2 * pl as u64);
                        }
                    }
                    match tsplit.as_ref() {
                        // TP-B: this rank's query heads (q|gate columns), its KV head (k/v columns;
                        // the KV cache holds only it), o_proj ROWS (row-parallel partial -> one FP32
                        // all-reduce). q/k norms are per head-dim (replicated); the QSA indexer is
                        // replicated (the selection is per layer, shared by all heads — design §1.3).
                        Some(ts) => {
                            let (qr, kvr, orr) = (ts.attn_q(&cfg), ts.attn_kv(&cfg)?, ts.attn_o(&cfg));
                            let a = AttnLayer {
                                q_proj: get_quad_sl(&format!("{p}.self_attn.q_proj"), None, Some(&[qr]), false)?,
                                k_proj: get_quad_sl(&format!("{p}.self_attn.k_proj"), None, Some(&[kvr]), false)?,
                                v_proj: get_quad_sl(&format!("{p}.self_attn.v_proj"), None, Some(&[kvr]), false)?,
                                o_proj: get_quad_sl(&format!("{p}.self_attn.o_proj"), Some(&[orr]), None, true)?,
                                qnkn: dev.htod_sync_copy(&qnkn)?,
                                idx,
                            };
                            let (hd, nhl, nkvl) = (tc.head_dim, tc.num_heads, tc.num_kv_heads);
                            anyhow::ensure!(a.q_proj.n as usize == nhl * 2 * hd && a.k_proj.n as usize == nkvl * hd
                                            && a.v_proj.n as usize == nkvl * hd && a.o_proj.k as usize == nhl * hd
                                            && a.o_proj.n as usize == h && qnkn.len() == 2 * hd,
                                "TP-B attn L{li}: per-rank shapes (q {} k {} v {} o {}x{} qnkn {}) != local geometry \
                                 ({nhl} heads, {nkvl} kv heads, hd {hd}) — refusing to boot",
                                a.q_proj.n, a.k_proj.n, a.v_proj.n, a.o_proj.k, a.o_proj.n, qnkn.len());
                            bytes_round += (2560 * 13312 + 6144 * 2560) as u64 * 5 / 8 / ts.world as u64;
                            Mixer::Attn(a)
                        }
                        None => Mixer::Attn(AttnLayer {
                            q_proj: get_quad(&format!("{p}.self_attn.q_proj"))?,
                            k_proj: get_quad(&format!("{p}.self_attn.k_proj"))?,
                            v_proj: get_quad(&format!("{p}.self_attn.v_proj"))?,
                            o_proj: get_quad(&format!("{p}.self_attn.o_proj"))?,
                            qnkn: dev.htod_sync_copy(&qnkn)?,
                            idx,
                        }),
                    }
                }
            };

            let moe = mk_moe(format!("{p}.mlp"), format!("L{li}"), tp.as_ref().map(|t| (t.rank as usize, t.world as usize)))?;
            bytes_round += (10 * (moe.gu_words + moe.d_words)) * 2;
            bytes_round += (2560 * 640 * 3) as u64 * 5 / 8;
            bytes_round += (512 * 2560) as u64 * 2;
            if let Some(k) = qsa_keys_plane.take() { qsa_keys_vec.push(k); }
            if let Some(k) = qsa_pool_plane.take() { qsa_pool_vec.push(k); }
            layers.push(LayerDev { hc_attn, hc_mlp, mixer, moe });
            load_prog.add(1);
            if li % 12 == 0 {
                crate::dprintln!("  layer {li} ({lt:?}) loaded ({:.0}s)", t_load.elapsed().as_secs_f32());
            }
        }
        load_prog.finish();

        let hc_mixer = if cfg.hc_lowrank == 0 {
            // EXL27B: plain residual — the final norm is the trunk's model.language_model.norm
            plain_hc("model.language_model.norm.weight")?
        } else {
            let names: Vec<String> = plain
                .keys()
                .filter(|k| k.contains("hyper_connection_mixer") && !k.contains(".layers.") && !k.contains("mtp"))
                .cloned()
                .collect();
            let pick = |suffix: &str| -> Result<String> {
                names
                    .iter()
                    .find(|k| k.ends_with(suffix))
                    .cloned()
                    .with_context(|| format!("final mixer *{suffix} not found (have {names:?})"))
            };
            let rd = |name: &str| -> Result<CudaSlice<u16>> {
                let t = plain.get(name).context("mixer tensor")?;
                let bits = to_f16_bits(&t.read(pack_dir)?, &t.dtype)?;
                Ok(dev.htod_sync_copy(&bits)?)
            };
            HcDev {
                norm: if fxp.is_some() { get_norm_f16(&pick("hc_norm.weight")?)? } else { rd(&pick("hc_norm.weight")?)? },
                down: if fx_plain { get_plain_wg(&pick("input_mix_weight_down.weight")?, cfg.hc_lowrank, cfg.hidden_size * cfg.hc_count.max(1))? }
                      else { rd(&pick("input_mix_weight_down.weight")?)? },
                up: if fx_plain { get_plain_wg(&pick("input_mix_weight_up.weight")?, cfg.hidden_size * cfg.hc_count.max(1), cfg.hc_lowrank)? }
                    else { rd(&pick("input_mix_weight_up.weight")?)? },
                winj: None,
                q8: None,
                norm_f: get_norm_f(&pick("hc_norm.weight")?)?,
                bf: fx_plain,
            }
        };

        let mut lm_head = get_quad("lm_head")?;
        lm_head.lmh = true; // W3/LMH
        if lm_head.fx.is_none() { lmh_prewarm(lm_head.bits as usize, lm_head.k as usize); } // FP4L: f16 head, no trellis entry
        bytes_round += (2560 * cfg.vocab_size) as u64 * 5 / 8;

        // ---- MTP draft head (S-A3-e M1): load mtp.* as EXL3 through the same weight path. ----
        // EXL27B: Qwen3.5-class packs carry the standard head — ONE `mtp.fc` over
        // cat[norm_e(embed), norm_h(hidden)] (K = 2h) — instead of FN's fc_hidden + fc_embedding.
        let mtp_fc_cat = !has_lin("mtp.fc_hidden") && has_lin("mtp.fc");
        let mut mtp = if has_lin("mtp.fc_hidden") || mtp_fc_cat {
            let t0h = std::time::Instant::now();
            let hcn = cfg.hc_count.max(1);
            let hp = "mtp.layers.0";
            let mk_hc = |base: String, inject: bool| -> Result<HcDev> {
                Ok(HcDev {
                    norm: get_norm_f16(&format!("{base}.hc_norm.weight"))?,
                    down: get_plain_wg(&format!("{base}.input_mix_weight_down.weight"), cfg.hc_lowrank, cfg.hidden_size * cfg.hc_count.max(1))?,
                    up: get_plain_wg(&format!("{base}.input_mix_weight_up.weight"), cfg.hidden_size * cfg.hc_count.max(1), cfg.hc_lowrank)?,
                    winj: if inject {
                        Some(get_plain_w(&format!("{base}.block_inject_weight.weight"))?)
                    } else {
                        None
                    },
                    q8: None,
                    norm_f: get_norm_f(&format!("{base}.hc_norm.weight"))?,
                    bf: fx_plain,
                })
            };
            let (hc_attn, hc_mlp) = if cfg.hc_lowrank == 0 {
                // EXL27B: the head block is a standard pre-norm decoder layer (its input_layernorm
                // normalizes the fc output, which is also the residual)
                (plain_hc(&format!("{hp}.input_layernorm.weight"))?, plain_hc(&format!("{hp}.post_attention_layernorm.weight"))?)
            } else {
                (mk_hc(format!("{hp}.attn_hyper_connection"), true)?, mk_hc(format!("{hp}.mlp_hyper_connection"), true)?)
            };
            let qn_t = plain.get(&format!("{hp}.self_attn.q_norm.weight")).context("mtp q_norm")?;
            let kn_t = plain.get(&format!("{hp}.self_attn.k_norm.weight")).context("mtp k_norm")?;
            let qnkn: Vec<u16> = to_f16_bits(&qn_t.read(pack_dir)?, &qn_t.dtype)?
                .into_iter()
                .chain(to_f16_bits(&kn_t.read(pack_dir)?, &kn_t.dtype)?)
                .collect();
            let attn = AttnLayer {
                q_proj: get_quad(&format!("{hp}.self_attn.q_proj"))?,
                k_proj: get_quad(&format!("{hp}.self_attn.k_proj"))?,
                v_proj: get_quad(&format!("{hp}.self_attn.v_proj"))?,
                o_proj: get_quad(&format!("{hp}.self_attn.o_proj"))?,
                qnkn: dev.htod_sync_copy(&qnkn)?,
                // Head QSA (2026-09-26): load the MTP layer's own indexer (the S-A3-i session
                // left it unwired by scope, not by finding). Past qsa_limit the draft attends
                // its QSA selection — the model spec and the reference implementation's draft — instead of a dense
                // scan whose cost grew with context (4.7 ms/pass at 128K, fp32 K/V).
                idx: if cfg.has_indexer() && plain.get(&format!("{hp}.self_attn.indexer.q_layernorm.weight")).is_some() {
                    let f32_of = |name: &str| -> Result<CudaSlice<f32>> {
                        let t = plain.get(&format!("{hp}.self_attn.indexer.{name}.weight"))
                            .with_context(|| format!("mtp indexer {name}"))?;
                        let bits = to_f16_bits(&t.read(pack_dir)?, &t.dtype)?;
                        let v: Vec<f32> = bits.iter().map(|b| h16_to_f32(*b)).collect();
                        Ok(dev.htod_sync_copy(&v)?)
                    };
                    Some(IndexerDev {
                        qk: get_quad(&format!("{hp}.self_attn.indexer.index_qk_proj"))?,
                        qn: f32_of("q_layernorm")?,
                        kn: f32_of("k_layernorm")?,
                        params: None,
                    })
                } else { None },
            };
            let head_qsa_keys = if attn.idx.is_some() {
                crate::metrics::kv_pool_add_bytes(2, 2 * (width * max_pos * cfg.indexer_head_dim) as u64); // issue #24
                Some(dev.htod_sync_copy(&vec![0u16; width * max_pos * cfg.indexer_head_dim])?)
            } else { None };
            let head_qsa_pool = if attn.idx.is_some() && pool_on {
                let pl = qsa_pool_plane_elems(width, max_pos, cfg.indexer_compress_ratio, cfg.indexer_head_dim); // issue #41
                crate::metrics::kv_pool_add_bytes(2, 2 * pl as u64);
                Some(dev.htod_sync_copy(&vec![0u16; pl])?)
            } else { None };
            // TP-A (T1): the MTP head is REPLICATED on every rank — no EP shard, no reduce sites.
            let moe = mk_moe(format!("{hp}.mlp"), "mtp".to_string(), None)?;
            let (fc_hidden, fc_embed) = if mtp_fc_cat {
                // EXL27B: split the K = 2h fc at h (128-aligned: the input Hadamard blocks never straddle
                // it; the output svh/Hadamard is linear) into the embed half [0, h) and the hidden half
                // [h, 2h) — fc(cat[e, x]) = W_e e + W_x x, the FN head's two-projection sum.
                let h = cfg.hidden_size;
                (get_quad_sl("mtp.fc", Some(&[(h, 2 * h)]), None, false)?,
                 get_quad_sl("mtp.fc", Some(&[(0, h)]), None, false)?)
            } else {
                (get_quad("mtp.fc_hidden")?, get_quad("mtp.fc_embedding")?)
            };
            let pre_fc_norm_hidden = get_plain_f16("mtp.pre_fc_norm_hidden.weight")?;
            let pre_e = get_plain_f16("mtp.pre_fc_norm_embedding.weight")?;
            let pre_e_h: Vec<u16> = dev.dtoh_sync_copy(&pre_e)?;
            let mut tiled: Vec<u16> = Vec::with_capacity(pre_e_h.len() * hcn);
            for _ in 0..hcn {
                tiled.extend_from_slice(&pre_e_h);
            }
            let pre_fc_norm_embedding = dev.htod_sync_copy(&tiled)?;
            // FP4L --fp4l-norms f32: the two MTP entry norms as their stored values in fp32 (bf16 widened, exact;
            // the embedding one tiled hc x like the f16 copy) for xf_hc_norm_f
            let pre_f = if fx_norms {
                let hid = plain_f32_bf16("mtp.pre_fc_norm_hidden.weight")?;
                let emb = plain_f32_bf16("mtp.pre_fc_norm_embedding.weight")?;
                let emb_t: Vec<f32> = (0..hcn).flat_map(|_| emb.iter().copied()).collect();
                anyhow::ensure!(hid.len() == emb_t.len(), "FP4L MTP pre-fc norms: hidden {} vs tiled embedding {}", hid.len(), emb_t.len());
                Some((dev.htod_sync_copy(&hid)?, dev.htod_sync_copy(&emb_t)?))
            } else { None };
            let mixer = if cfg.hc_lowrank == 0 { plain_hc("mtp.norm.weight")? } else { HcDev {
                norm: get_norm_f16("mtp.hyper_connection_mixer.hc_norm.weight")?,
                down: get_plain_wg("mtp.hyper_connection_mixer.input_mix_weight_down.weight", cfg.hc_lowrank, cfg.hidden_size * cfg.hc_count.max(1))?,
                up: get_plain_wg("mtp.hyper_connection_mixer.input_mix_weight_up.weight", cfg.hidden_size * cfg.hc_count.max(1), cfg.hc_lowrank)?,
                winj: None,
                q8: None,
                norm_f: get_norm_f("mtp.hyper_connection_mixer.hc_norm.weight")?,
                bf: fx_plain,
            } };
            let nkv = cfg.num_kv_heads;
            let hd = cfg.head_dim;
            let kv = dev.alloc_zeros::<u8>(width * 2 * nkv * max_pos * kv_rowbytes(kv_fmt, hd))?;
            crate::metrics::kv_pool_add_bytes(2, (width * 2 * nkv * max_pos * kv_rowbytes(kv_fmt, hd)) as u64); // issue #24
            // PR #27 (lesserevil): the widths this pack's head actually carries (the line used to print the 3.05
            // pack's M1 widths for every pack)
            let qw = |q: &Quad| if q.fx.is_some() { "stored precision".to_string() }
                                else { format!("{}-bit", exl3_width_name(exl3_kernel_bits(q.k2 as usize))) };
            crate::dprintln!(
                "  MTP draft head (mtp.*) loaded: fc_hidden {}, fc_embed {}, experts {} ({:.0}s)",
                qw(&fc_hidden), qw(&fc_embed),
                if moe.fx.is_some() { "NVFP4".to_string() } else { format!("{}-bit", exl3_width_name(moe.bits)) },
                t0h.elapsed().as_secs_f32()
            );
            Some(DraftHead {
                hc_attn,
                hc_mlp,
                attn,
                moe,
                fc_hidden,
                fc_embed,
                pre_fc_norm_hidden,
                pre_fc_norm_embedding,
                pre_f,
                mixer,
                kv,
                qsa_keys: head_qsa_keys,
                qsa_pool: head_qsa_pool,
            })
        } else {
            None
        };

        // TP-B: the trunk's recurrent / conv / KV state is per-rank (its heads / channels only).
        let kd = tc.lin_k_dim;
        let vd = tc.lin_v_dim;
        let nh = tc.lin_num_v_heads;
        let conv_dim = tc.key_dim() * 2 + tc.value_dim();
        // Expert-flagged stride contract: xq_gdn_step derives qkv_stride in-kernel as
        // 2*n_k_heads*kd + nh*vd; this host conv_dim must equal it or lane rows misalign.
        debug_assert_eq!(conv_dim, 2 * tc.lin_num_k_heads * tc.lin_k_dim + tc.lin_num_v_heads * tc.lin_v_dim);
        let ck = cfg.conv_kernel;
        let n_attn = cfg
            .layer_types
            .iter()
            .filter(|l| matches!(l, crate::qwen::LayerType::FullAttention))
            .count();
        let mut conv_state = Vec::new();
        let mut s_state = Vec::new();
        for lt in &cfg.layer_types {
            if matches!(lt, crate::qwen::LayerType::LinearAttention) {
                conv_state.push(dev.alloc_zeros::<f32>(width * conv_dim * ck)?);
                s_state.push(dev.alloc_zeros::<f32>(width * nh * kd * vd)?);
            }
        }
        let mut k_cache = Vec::new();
        for _ in 0..n_attn {
            k_cache.push(dev.alloc_zeros::<u8>(width * 2 * tc.num_kv_heads * max_pos * kv_rowbytes(kv_fmt, tc.head_dim))?);
            crate::metrics::kv_pool_add_bytes(0, (width * 2 * tc.num_kv_heads * max_pos * kv_rowbytes(kv_fmt, tc.head_dim)) as u64); // issue #24
        }

        if fxp.is_some() {
            // FP4L: the byte model from the loaded weights (the trellis estimates above do not apply)
            bytes_round = fp4l_fwd::byte_model(&layers, &hc_mixer, &lm_head, cfg.num_experts_per_tok, cfg.hidden_size,
                                               cfg.moe_intermediate_size);
        }
        crate::dprintln!(
            "EXL3-FWD loaded in {:.1}s; W={width} max_pos={max_pos}; byte model ~{:.2} GB/round (W=1)",
            t_load.elapsed().as_secs_f32(),
            bytes_round as f64 / 1e9
        );

        // ---- PLE (S-A3-d Phase 0b): optional — present in packs with n-gram injection.
        let ple = {
            const PLE_KEY: &str = "model.language_model.layers.1.ple";
            match plain.get(&format!("{PLE_KEY}.key_proj.weight")) {
                None => None,
                Some(_) => {
                    use std::io::Read;
                    // WP09: host bits once -> the row-major device copy (prefill) + the k-major one
                    // (decode/verify). Same bits as get_plain_f16.
                    let plain_bits = |name: &str| -> Result<Vec<u16>> {
                        let t = plain.get(name).with_context(|| format!("missing plain tensor {name}"))?;
                        to_f16_bits(&t.read(pack_dir)?, &t.dtype)
                    };
                    // FP4L --fp4l-plain exact: the stored bf16 bits (the WP09 k-major f16 relayouts do not apply)
                    let ple_bf = fx_plain;
                    let stored_bits = |name: &str| -> Result<Vec<u16>> {
                        let t = plain.get(name).with_context(|| format!("missing plain tensor {name}"))?;
                        anyhow::ensure!(t.dtype == "BF16", "FP4L --fp4l-plain exact: {name} is {} (needs BF16)", t.dtype);
                        Ok(t.read(pack_dir)?.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
                    };
                    let (key_bits, value_bits) = if ple_bf {
                        (stored_bits(&format!("{PLE_KEY}.key_proj.weight"))?, stored_bits(&format!("{PLE_KEY}.value_proj.weight"))?)
                    } else {
                        (plain_bits(&format!("{PLE_KEY}.key_proj.weight"))?, plain_bits(&format!("{PLE_KEY}.value_proj.weight"))?)
                    };
                    anyhow::ensure!(key_bits.len() == 10240 * 2560 && value_bits.len() == 2560 * 2560,
                                    "ple key/value shapes {} / {}", key_bits.len(), value_bits.len());
                    // FP4L exact: the stored bits in the B32 layout the stored-bf16 GEMMs read (a permutation)
                    let key_w = if ple_bf { dev.htod_sync_copy(&fp4l_fwd::bf16_frag(&key_bits, 10240, 2560))? } else { dev.htod_sync_copy(&key_bits)? };
                    let value_w = if ple_bf { dev.htod_sync_copy(&fp4l_fwd::bf16_frag(&value_bits, 2560, 2560))? } else { dev.htod_sync_copy(&value_bits)? };
                    let (key_km, value_km) = if wp09_ple_on() && !ple_bf {
                        (Some(dev.htod_sync_copy(&relayout_km16(&key_bits, 10240, 2560))?),
                         Some(dev.htod_sync_copy(&relayout_km16(&value_bits, 2560, 2560))?))
                    } else {
                        (None, None)
                    };
                    drop((key_bits, value_bits));
                    let conv_w = get_plain_f16(&format!("{PLE_KEY}.conv1d.weight"))?;
                    let mut normw = Vec::new();
                    for nm in ["norm_key", "norm_query", "norm_conv"] {
                        let t = plain.get(&format!("{PLE_KEY}.{nm}.weight"))
                            .with_context(|| format!("ple {nm}"))?;
                        if fxp.is_some() {
                            // FP4L: quiet f16 copy (see get_norm_f16)
                            let raw = t.read(pack_dir)?;
                            let v = if t.dtype == "BF16" { bf16_to_f32(&raw) } else { plain_to_f32(&raw, &t.dtype)? };
                            normw.push(v.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect());
                        } else {
                        normw.push(to_f16_bits(&t.read(pack_dir)?, &t.dtype)?);
                        }
                    }
                    let norm_key_w = { let mut d = dev.alloc_zeros::<u16>(10240)?; dev.htod_copy_into(normw.remove(0), &mut d)?; d };
                    let norm_query_w = { let mut d = dev.alloc_zeros::<u16>(10240)?; dev.htod_copy_into(normw.remove(0), &mut d)?; d };
                    let norm_conv_w = { let mut d = dev.alloc_zeros::<u16>(10240)?; dev.htod_copy_into(normw.remove(0), &mut d)?; d };
                    // FP4L: the conv taps widened to fp32 (exact) and the stored fp32 norms, for the xf_ple_* twins
                    let conv_f = if ple_bf {
                        Some(dev.htod_sync_copy(&plain_f32_bf16(&format!("{PLE_KEY}.conv1d.weight"))?)?)
                    } else { None };
                    let norm_f = if fx_norms {
                        let f = |nm: &str| -> Result<CudaSlice<f32>> {
                            Ok(dev.htod_sync_copy(&plain_f32_bf16(&format!("{PLE_KEY}.{nm}.weight"))?)?)
                        };
                        Some(Box::new([f("norm_key")?, f("norm_query")?, f("norm_conv")?]))
                    } else { None };
                    if let Some(fp) = fxp.as_deref() {
                        // FP4L: the NVFP4 n-gram table (fp4l_fwd::PleFx): 128 shards of [rows, 80] nibbles +
                        // [rows, 10] E4M3 scales, one multiplier; hash tables under ple_embedding.*; the n-gram
                        // context resets on config.eos_token_id (the reference Qwen4ExpTextNGramEmbedding).
                        let i64s = |nm: &str| fp.i64s(&format!("{PLE_KEY}.ple_embedding.{nm}"));
                        let head_offsets = i64s("ngram_heads_offsets")?;
                        let head_vocab_sizes = i64s("ngram_heads_vocab_sizes")?;
                        let lm = i64s("layer_multipliers")?;
                        anyhow::ensure!(lm.len() == 3 && head_offsets.len() == 16 && head_vocab_sizes.len() == 16,
                                        "FP4L PLE hash tables: multipliers {} offsets {} vocab sizes {}", lm.len(),
                                        head_offsets.len(), head_vocab_sizes.len());
                        let (shards, pfx, rows_per_shard) = fp4l_fwd::fx_ple_table(fp, PLE_KEY)?;
                        let gib = (rows_per_shard as u64 * 128 * fp4l_fwd::PLE_FX_ROW as u64) as f64 / (1u64 << 30) as f64;
                        let ram = std::sync::OnceLock::new();
                        if ple_mode() == "ram" {
                            let t0 = std::time::Instant::now();
                            let _ = ram.set(fp4l_fwd::ple_fx_ram(&shards, &pfx, rows_per_shard)?);
                            crate::drprintln!("  PLE n-gram table (FP4L NVFP4) RAM-resident: {gib:.1} GiB in {:.1}s (--ple-ram ram)",
                                             t0.elapsed().as_secs_f32());
                        } else {
                            crate::drprintln!("  PLE n-gram table (FP4L NVFP4, {gib:.1} GiB, 2 preads per row) on SSD via the page cache (--ple-ram {}; \
                                             `auto` promotes it to RAM after the boot when it fits)", ple_mode());
                        }
                        let shard_data_off = pfx.w_off.clone();
                        Some(PleDev {
                            key_w, value_w, key_km, value_km, conv_w, norm_key_w, norm_query_w, norm_conv_w,
                            shards, ram, shard_data_off, rows_per_shard, row_k: 0, row_bytes: fp4l_fwd::PLE_FX_ROW,
                            head_offsets, head_vocab_sizes, layer_multipliers: [lm[0], lm[1], lm[2]],
                            head_bias_bits: Vec::new(), head_bias_f32: Vec::new(), codebook: Vec::new(),
                            eos: ple_eos, fx: Some(pfx), bf: ple_bf, conv_f, norm_f,
                        })
                    } else {
                    let aux = |nm: &str| -> Result<Vec<i64>> {
                        let t = plain.get(&format!("{PLE_KEY}.ple_embedding.ngram_embedding.{nm}"))
                            .with_context(|| format!("ple {nm}"))?;
                        let raw = t.read(pack_dir)?;
                        Ok(raw.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect())
                    };
                    let head_offsets = aux("head_offsets")?;
                    let head_vocab_sizes = aux("head_vocab_sizes")?;
                    let lm = aux("layer_multipliers")?;
                    anyhow::ensure!(lm.len() == 3, "ple layer_multipliers");
                    let hb_t = plain.get(&format!("{PLE_KEY}.ple_embedding.ngram_embedding.head_bias"))
                        .with_context(|| "ple head_bias".to_string())?;
                    let head_bias_bits = to_f16_bits(&hb_t.read(pack_dir)?, &hb_t.dtype)?;
                    let index: serde_json::Value = serde_json::from_reader(
                        std::fs::File::open(format!("{pack_dir}/model.safetensors.index.json"))
                            .with_context(|| "ple index".to_string())?)?;
                    let wmap = index["weight_map"].as_object().context("ple weight_map")?;
                    let mut shards = Vec::new();
                    let mut shard_data_off = Vec::new();
                    let mut rows_per_shard = 0i64;
                    // #26 (Fix 2): placeholder only — every source path (single-tensor / index /
                    // sidecar) sets the real width from the pack before PleDev is built. It must
                    // NOT be used as a fallback: a stale K=5 here would serve wrong PLE rows.
                    let mut row_words = 0usize;
                    // CF-P1g: ONE trellis tensor [rows, 10K+1] (4.05 pack) = the 128 shards concatenated: modelled as 128
                    // virtual shards of the one file, so everything downstream (RAM/SSD placement, row fetch) is unchanged.
                    let single_name = format!("{PLE_KEY}.ple_embedding.ngram_embedding.trellis");
                    let single_loaded = if let Some(t) = plain.get(&single_name) {
                        let fpath = format!("{}/{}", pack_dir.trim_end_matches('/'), t.shard);
                        let mut f = std::fs::File::open(&fpath).with_context(|| format!("ple open {fpath}"))?;
                        let mut lb = [0u8; 8];
                        f.read_exact(&mut lb)?;
                        let mut hb = vec![0u8; u64::from_le_bytes(lb) as usize];
                        f.read_exact(&mut hb)?;
                        let hdr: serde_json::Value = serde_json::from_slice(&hb)?;
                        let shape: Vec<i64> = hdr[single_name.as_str()]["shape"].as_array().context("ple trellis shape")?
                            .iter().map(|v| v.as_i64().unwrap_or(0)).collect();
                        anyhow::ensure!(shape.len() == 2 && (shape[1] - 1) % 10 == 0 && matches!((shape[1] - 1) / 10, 1..=8),
                                        "ple trellis shape {shape:?}: rows must be 10*K+1 words with K = 1..=8");
                        anyhow::ensure!(shape[0] % 128 == 0, "ple trellis rows {} not a multiple of 128", shape[0]);
                        row_words = shape[1] as usize;
                        rows_per_shard = shape[0] / 128;
                        // #26 (Fix 2): the width counter is incremented ONCE, post-layout,
                        // below — a single site covering every source path (single-tensor,
                        // via-index, via-sidecar). AGENTS 1b-bis.2: a path that never runs
                        // must not read as done; a path that RUNS must not read as never-run
                        // (the sidecar path served K=3 with the counter stuck at 0).
                        for vs in 0..128i64 {
                            shard_data_off.push(t.offset as u64 + (vs * rows_per_shard) as u64 * row_words as u64 * 2);
                            shards.push(f.try_clone()?);
                        }
                        true
                    } else { false };
                    // REL-NGRAM (issue #9): resolve each shard through the index when listed
                    // (today's path, byte-for-byte) and otherwise through the sidecar's own
                    // safetensors header (exllamav3 1.5.x packs; validated by ple_shard_sources).
                    let (shard_names, shard_srcs, sc_layout) = if single_loaded {
                        (Vec::new(), Vec::new(), None)
                    } else {
                        let names: Vec<String> = (0..128)
                            .map(|snum| format!("{PLE_KEY}.ple_embedding.ngram_embedding.shard_{snum}.trellis"))
                            .collect();
                        let sidecar = ple_sidecar_header(pack_dir)?;
                        let (srcs, layout) = ple_shard_sources(
                            Some(wmap),
                            sidecar.as_ref().map(|(f, h, hdr)| (f.as_str(), *h, hdr)),
                            &names,
                        )?;
                        (names, srcs, layout)
                    };
                    let mut n_via_index = 0usize;
                    for snum in 0..(if single_loaded { 0usize } else { 128usize }) {
                        let name = &shard_names[snum];
                        match &shard_srcs[snum] {
                            PleShardSrc::Index(fname) => {
                                n_via_index += 1;
                                let fname = fname.as_str();
                                let mut f = std::fs::File::open(format!("{pack_dir}/{fname}"))
                                    .with_context(|| format!("ple open {fname}"))?;
                                let mut lb = [0u8; 8];
                                f.read_exact(&mut lb)?;
                                let hlen = u64::from_le_bytes(lb) as usize;
                                let mut hb = vec![0u8; hlen];
                                f.read_exact(&mut hb)?;
                                let hdr: serde_json::Value = serde_json::from_slice(&hb)?;
                                let ent = &hdr[name.as_str()];
                                let shape: Vec<i64> = ent["shape"].as_array().context("ple shard shape")?
                                    .iter().map(|v| v.as_i64().unwrap_or(0)).collect();
                                if snum == 0 {
                                    rows_per_shard = shape[0];
                                    // All-index packs (no single-tensor, no sidecar-sourced shard):
                                    // nothing upstream set the width — take it from shard 0 once.
                                    // Never clobbers the single-tensor/sidecar value (Fix 2 intent).
                                    if row_words == 0 { row_words = shape[1] as usize; }
                                    anyhow::ensure!(shape[1] as usize == row_words,
                                                     "ple shard row width {:?} != row_words {row_words}", shape);
                                    // #26 (Fix 2): the width was already validated above; do NOT
                                    // clobber it to 51 (that hard-codes K=5 and would corrupt a
                                    // K=3 pack's row_k derivation).
                                }
                                // Mixed-source packs only (all-index packs keep today's checks
                                // exactly): an index shard must agree with the sidecar layout.
                                if let Some((rows, words)) = sc_layout {
                                    anyhow::ensure!(shape[0] == rows && shape[1] as usize == words,
                                        "ple shard {snum} shape {shape:?} disagrees with the sidecar shards' [{rows}, {words}]");
                                }
                                let off0 = ent["data_offsets"][0].as_u64().context("ple shard offset")?;
                                shard_data_off.push(8 + hlen as u64 + off0);
                                shards.push(f);
                            }
                            PleShardSrc::Sidecar { fname, abs_off } => {
                                if snum == 0 {
                                    let (rows, words) = sc_layout.expect("sidecar layout present");
                                    rows_per_shard = rows;
                                    row_words = words;
                                }
                                let mut f = std::fs::File::open(format!("{pack_dir}/{fname}"))
                                    .with_context(|| format!("ple open {fname}"))?;
                                shard_data_off.push(*abs_off);
                                shards.push(f);
                            }
                        }
                    }
                    // REL-NGRAM: once-per-boot line naming which n-gram layout served the pack
                    // (the single-tensor / via-index / via-sidecar distinction). boot_report
                    // notice = ALWAYS visible (the width receipt: K= and the source path).
                    if single_loaded {
                        crate::boot_report::notice(&format!("PLE n-gram table: single trellis tensor, K={}, rows/shard {} (128 virtual shards)",
                                 (row_words - 1) / 10, rows_per_shard));
                    } else if n_via_index == 128 {
                        crate::boot_report::notice(&format!("PLE n-gram table: 128 shards via model index, K={}, rows/shard {}",
                                 (row_words - 1) / 10, rows_per_shard));
                    } else if n_via_index == 0 {
                        crate::boot_report::notice(&format!("PLE n-gram table: 128 shards via sidecar header (exllamav3 1.5.x layout, issue #9), K={}, rows/shard {}",
                                 (row_words - 1) / 10, rows_per_shard));
                    } else {
                        crate::boot_report::notice(&format!("PLE n-gram table: {n_via_index} shards via model index + {} via sidecar header, K={}, rows/shard {}",
                                 128 - n_via_index, (row_words - 1) / 10, rows_per_shard));
                    }
                    // #26 (Fix 3): the unified width receipt — one increment for every source
                    // path, counted only for widths outside the historical {5, 6}.
                    if !matches!((row_words - 1) / 10, 5 | 6) {
                        crate::metrics::exl3_ple_width_accepted();
                    }
                    let ram = std::sync::OnceLock::new();
                    let row_bytes = row_words * 2;
                    if let Some(t) = ple_ram_load(&shards, &shard_data_off, rows_per_shard, row_bytes, width)? { let _ = ram.set(t); }
                    let pd = PleDev {
                        key_w,
                        value_w,
                        key_km,
                        value_km,
                        conv_w,
                        norm_key_w,
                        norm_query_w,
                        norm_conv_w,
                        shards,
                        ram,
                        shard_data_off,
                        rows_per_shard,
                        row_k: (row_words - 1) / 10,
                        row_bytes,
                        head_offsets,
                        head_vocab_sizes,
                        layer_multipliers: [lm[0], lm[1], lm[2]],
                        head_bias_f32: head_bias_bits.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect(),
                        head_bias_bits,
                        codebook: PleDev::mul1_codebook(),
                        eos: 248046,
                        fx: None,
                        bf: false,
                        conv_f: None,
                        norm_f: None,
                    };
                    if let Ok(path) = crate::opts::var(crate::opt!("exl3-ple-dump")) {
                        pd.dump_oracle(&path)?;
                        crate::dprintln!("[exl3] --exl3-ple-dump: wrote {path}; exiting");
                        std::process::exit(0);
                    }
                    Some(pd)
                    } // FP4L else
                }
            }
        };

        let ple_state = dev.alloc_zeros::<u16>(width * 10240 * 9)?;

        // S-A3-f Item 1: full RoPE table, host-computed with the SAME powf/cos/sin
        // code the per-step loop used (bitwise-identical bits, paid once at load),
        // and ONE blocking compute stream for the model's lifetime.
        let rdim_t = cfg.rotary_dim;
        // YARN1: ONE rope source for the whole EXL3 trunk — decode cos_tab, the prefill host
        // loop, the QSA indexer tables and the vision mrope rows all take this RopeFreqs.
        // Factor 1.0 (no flag, no shipped TpConfig value, no config rope_scaling) reproduces
        // the legacy tables bit-for-bit; a factor > 1 prints the boot line naming factor+source.
        let fr = RopeFreqs::for_opts(rdim_t, cfg.rope_theta, cfg.yarn_native_window, &cfg);
        let mut cos_tab_h: Vec<f32> = Vec::with_capacity(max_pos * 2 * rdim_t);
        for p in 0..max_pos {
            for i in 0..rdim_t / 2 { cos_tab_h.push(fr.cos_sin(p, i).0); }
            for _ in 0..rdim_t / 2 { cos_tab_h.push(0.0); }
            for i in 0..rdim_t / 2 { cos_tab_h.push(fr.cos_sin(p, i).1); }
            for _ in 0..rdim_t / 2 { cos_tab_h.push(0.0); }
        }
        let cos_tab = dev.htod_sync_copy(&cos_tab_h)?;
        // VIS-2: per-slot mrope buffers, zero = text-only (alloc_zeros does not zero: upload zeros)
        let rope_map = dev.htod_sync_copy(&vec![0i32; width * max_pos])?;
        let rope_plen = dev.htod_sync_copy(&vec![0i32; width])?;
        let rope_delta = dev.htod_sync_copy(&vec![0i32; width])?;
        let mrope_tab = dev.htod_sync_copy(&vec![0f32; width * VIS_MROPE_ROWS * 2 * rdim_t])?;
        // WP05: the prefill prologue gathers its RoPE rows from cos_tab. Compare the table with
        // the prefill's own host loop (prefill_cossin_host, the pre-WP05 code) on sampled rows —
        // all of [0, 2048), every 127th row after, the last 128 — and refuse the gather (keep the
        // host loop, loudly) on any bit mismatch.
        let cos_gather_ok = {
            let row = 2 * rdim_t;
            let mut ps: Vec<usize> = (0..max_pos.min(2048)).collect();
            ps.extend((2048..max_pos).step_by(127));
            ps.extend(max_pos.saturating_sub(128).max(2048)..max_pos);
            let mut bad = 0usize;
            for &p in &ps {
                let hrow = prefill_cossin_host(&fr, rdim_t, p, 1);
                let trow = &cos_tab_h[p * row..(p + 1) * row];
                bad += hrow.iter().zip(trow).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            }
            if bad > 0 {
                crate::dprintln!("WP05: cos_tab differs from the prefill host RoPE loop in {bad} values over {} sampled rows — prefill keeps the host loop", ps.len());
            } else {
                crate::dprintln!("WP05: prefill RoPE gather armed — cos_tab == prefill host loop bit-for-bit on {} sampled rows", ps.len());
            }
            bad == 0
        };
        // S-A3-i: deinterleaved indexer rope tables + per-layer QsaParams blocks. Same
        // powf/sin/cos formula and order as cos_tab (identical bits); the block layout
        // mirrors gpu.rs qsa_params so the shared bf16 selection kernels read it as-is.
        let (mut qsa_cos_h, mut qsa_sin_h): (Vec<f32>, Vec<f32>) = (Vec::new(), Vec::new());
        // --qsa-key-rope full: both rotary halves (the reference / the NVFP4 engine's table); half
        // (pre-v0.7.1 bytes, A/B only) = the old table with dims rdim/2..rdim left at zero. Default full (owner 2026-10-02).
        let qsa_full = qsa_key_rope_full();
        for p in 0..max_pos {
            qsa_cos_h.resize(qsa_cos_h.len() + rdim_t, 0.0);
            qsa_sin_h.resize(qsa_sin_h.len() + rdim_t, 0.0);
            for i in 0..rdim_t / 2 {
                let (c, s) = fr.cos_sin(p, i);
                qsa_cos_h[p * rdim_t + i] = c;
                qsa_sin_h[p * rdim_t + i] = s;
                if qsa_full {
                    qsa_cos_h[p * rdim_t + i + rdim_t / 2] = c;
                    qsa_sin_h[p * rdim_t + i + rdim_t / 2] = s;
                }
            }
        }
        // VIS-3: per-slot indexer image rows (qsa_cos/qsa_sin layout) + the QsaMrope block every
        // indexer's QsaParams.mr points at (gpu_batch.cu; the shared rope_map/plen/delta).
        let qm_cos = dev.htod_sync_copy(&vec![0f32; width * VIS_MROPE_ROWS * rdim_t])?;
        let qm_sin = dev.htod_sync_copy(&vec![0f32; width * VIS_MROPE_ROWS * rdim_t])?;
        let qsa_mrope = {
            let mut b: Vec<u8> = Vec::with_capacity(48);
            for ptr in [*rope_map.device_ptr() as u64, *rope_plen.device_ptr() as u64, *rope_delta.device_ptr() as u64,
                        *qm_cos.device_ptr() as u64, *qm_sin.device_ptr() as u64] {
                b.extend_from_slice(&ptr.to_le_bytes());
            }
            b.extend_from_slice(&(width as i32).to_le_bytes());
            b.extend_from_slice(&(max_pos as i32).to_le_bytes());
            dev.htod_sync_copy(&b)?
        };
        let qsa_cos = dev.htod_sync_copy(&qsa_cos_h)?;
        let qsa_sin = dev.htod_sync_copy(&qsa_sin_h)?;
        if cfg.has_indexer() {
            let cos_p = *qsa_cos.device_ptr() as u64;
            let sin_p = *qsa_sin.device_ptr() as u64;
            let (hdx, heads, ratio) = (cfg.indexer_head_dim, cfg.indexer_n_heads, cfg.indexer_compress_ratio);
            let head_attn: Option<&mut AttnLayer> = mtp.as_mut().map(|h| &mut h.attn);
            for a in layers.iter_mut()
                .filter_map(|ld| match &mut ld.mixer { Mixer::Attn(a) => Some(a), _ => None })
                .chain(head_attn) {
                let ix = match a.idx.as_mut() { Some(ix) => ix, None => continue };
                let mut b: Vec<u8> = Vec::with_capacity(64);
                for ptr in [cos_p, sin_p, *ix.kn.device_ptr() as u64, *ix.qn.device_ptr() as u64] {
                    b.extend_from_slice(&ptr.to_le_bytes());
                }
                b.extend_from_slice(&(cfg.rms_eps as f32).to_le_bytes());
                for v in [hdx, heads, cfg.rotary_dim, ratio, cfg.indexer_budget / ratio,
                          cfg.indexer_budget + ratio - 1, 0usize] {
                    b.extend_from_slice(&(v as i32).to_le_bytes());
                }
                b.extend_from_slice(&(*qsa_mrope.device_ptr() as u64).to_le_bytes()); // VIS-3 QsaParams.mr
                ix.params = Some(dev.htod_sync_copy(&b)?);
            }
            crate::dprintln!("  QSA indexer loaded: budget {} ratio {} sel_max {} (dense <= {} visible); pooled-key rope {}",
                     cfg.indexer_budget, ratio, cfg.indexer_budget + ratio - 1,
                     cfg.indexer_budget + ratio - 1,
                     if qsa_full { "full (the reference)" } else { "HALF (--qsa-key-rope half: the pre-v0.7.1 table, dims rdim/2..rdim zero)" });
        }
        let stream = fork_blocking_stream(&dev);

        // S-A3-f-f ITEM 1: grid-barrier co-residency must be MEASURED, not assumed
        // (a deadlock here would hang the graph). 0 disables the fused mixer entirely.
        let hc_fuse_cap = hc_fuse_co_residency(&dev);
        crate::dprintln!(
            "EXL3-FWD hc_fuse co-residency: {} blocks (256t){}",
            hc_fuse_cap,
            if hc_fuse_cap > 0 { "" } else { " -> fused hc mixer DISABLED (old path, all m)" }
        );
        // WP11: the k-major twins are enabled (q_up relayout at quantize time) only when THEIR
        // own measured residency covers every width the fused int8 arm can see (m <= 60 there:
        // bx = clamp(cap/m, 4, 40) with bx*m <= hc_fuse_cap*5/6).
        let wp11_cap = if wp11_r3_on() && hc_fuse_cap > 0 {
            let c = (wp11_capacity(&dev, "xq_hc_fuse_i8k"), wp11_capacity(&dev, "xq_hc_fuse_i8_rbk"));
            if (hc_fuse_cap.min(c.0) * 5) / 6 >= 64 { c } else { (0, 0) }
        } else { (0, 0) };
        crate::dprintln!(
            "EXL3-FWD WP11 hc rungs: {} (caps i8k {} rbk {}; R1 pf={} R2={} R4={} cvt={}; --wp11-off=1 = old path)",
            if wp11_off() { "OFF" } else if wp11_cap.0 > 0 { "R3 k-major twins" } else { "R3 off (old fuse kernels)" },
            wp11_cap.0, wp11_cap.1, wp11_pf_mode(), wp11_r2_on() as u8, wp11_r4_on() as u8,
            if wp11_cvt_i2f() { "i2f" } else { "prmt" }
        );
        // HC (w4): the regridded mix-fused mixer needs the WP11 k-major up weights and its shape
        // contract (4 streams, h % 64, rw % 1024, lr % 16 <= 512); its grid is clamped to ITS OWN
        // measured co-residency x 5/6 (pinned-carveout handle), so the grid barrier is safe.
        let (w4hc_grid, w4hc_cap) = {
            let (h, hcn, lr) = (cfg.hidden_size, cfg.hc_count.max(1), cfg.hc_lowrank);
            let shape_ok = hcn == 4 && h % 64 == 0 && h <= 0xFFFF && (h * hcn) % 1024 == 0
                && lr % 16 == 0 && lr >= 16 && lr <= 512;
            // TUNE T0c: probed whenever a lookup can turn the mixer on (the default, a table value,
            // or the --autotune overlay); w4hc_takes still asks w4hc.fuse per width
            let fuse_reachable = tune::reachable(&T_W4HC_OFF).contains(&0) && tune::reachable(&T_W4HC_FUSE).contains(&1);
            if fuse_reachable && !wp11_off() && wp11_cap.0 > 0 && hc_fuse_cap > 0 && shape_ok {
                let cap = w4hc_capacity();
                let g = w4hc_default_grid(cap, h);
                crate::dprintln!("EXL3-FWD HC-W4 mixer: xq_hc_w4 co-residency {cap} blocks (256t, carveout max) -> grid {g}{}",
                         if g > 0 { "" } else { " -> OFF (fewer co-resident CTAs than the phase-B minimum ceil(h/32))" });
                (g, cap)
            } else {
                (0, 0)
            }
        };
        crate::dprintln!(
            "EXL3-FWD HC-W4: fuse {} (grid {w4hc_grid}, m=1 {}, mix-fused {}, geom flags {:#x}), inject+norm fusion {}{}; \
             --w4hc-off=1 = p4c hc path",
            if w4hc_grid > 0 { "ON" } else { "OFF" }, if w4hc_m1_on() { "on" } else { "off" },
            if w4hc_mix_on() { "on" } else { "off" }, w4hc_geom_flags(),
            if w4hc_inj_on() && cfg.hidden_size <= 4096 { "ON" } else { "OFF" },
            if w4hc_xcheck() { " [XCHECK]" } else { "" }
        );

        // S-A3-f-g: tmap-inverse LUT for exl3_decode_dump (the reconstruct front-end).
        let rc_lut = dev.htod_sync_copy(&crate::exl3_bench::tmap_inv_table())?;
        let qsa_dbg = match crate::opts::var(crate::opt!("qsa-dump-layer")).ok().and_then(|v| v.parse::<usize>().ok()) {
            Some(k) if cfg.has_indexer() => {
                let nbs = (max_pos / cfg.indexer_compress_ratio).max(1);
                let sm = cfg.indexer_budget + cfg.indexer_compress_ratio;
                Some((k, dev.htod_sync_copy(&vec![0i32; sm])?, dev.htod_sync_copy(&vec![0f32; nbs])?,
                      dev.htod_sync_copy(&vec![0i32; 2048 * sm])?, dev.htod_sync_copy(&vec![0f32; 2048 * nbs])?))
            }
            _ => None,
        };
        let qsa_rowdbg = match crate::opts::var(crate::opt!("qsa-rowdump-layer")).ok().and_then(|v| v.parse::<usize>().ok()) {
            Some(k) if cfg.has_indexer() => {
                let sm = cfg.indexer_budget + cfg.indexer_compress_ratio - 1;
                Some((k, dev.htod_sync_copy(&vec![-7i32; QSA_ROWDUMP_CAP * sm])?, dev.htod_sync_copy(&vec![-7i32; QSA_ROWDUMP_CAP])?,
                      dev.htod_sync_copy(&vec![-7i32; QSA_ROWDUMP_CAP * sm])?, dev.htod_sync_copy(&vec![-7i32; QSA_ROWDUMP_CAP])?))
            }
            _ => None,
        };
        let esel_hist = if crate::opts::var(crate::opt!("exl3-esel-hist")).is_ok() {
            Some(dev.htod_sync_copy(&vec![0i32; 17 * ESEL_HIST_W])?)
        } else { None };
        let route_log = match crate::opts::var(crate::opt!("exl3-route-log")) {
            Ok(dir) if !dir.is_empty() && dir != "0" => {
                std::fs::create_dir_all(&dir).with_context(|| format!("--exl3-route-log {dir}"))?;
                crate::dprintln!("[exl3] --exl3-route-log diagnostic: per-call expert ids -> {dir}/route_NNNN.bin (one file per request)");
                Some((dev.alloc_zeros::<i32>(ROUTE_LOG_CAP * ROUTE_REC)?, dev.htod_sync_copy(&[0u32])?, dir,
                      std::sync::atomic::AtomicUsize::new(0)))
            }
            _ => None,
        };
        let expert_hist = match crate::opts::var(crate::opt!("tp-ep-hist")) {
            Ok(path) if !path.is_empty() && path != "0" => {
                crate::dprintln!("[exl3] --tp-ep-hist diagnostic: per-(layer, expert) routing histogram -> {path} (cumulative, at every request finish)");
                Some((dev.htod_sync_copy(&vec![0i32; cfg.num_layers * 2 * cfg.num_experts])?, path))
            }
            _ => None,
        };

        // S-A3-f-d: pruned draft lm_head (reference patch 0002). Column slice of the
        // shared head's trellis: nb_keep = n2/16 leading 16-col blocks per k-block.
        // n2 = min(--exl3-mtp-head-n || 65536, n_full) rounded DOWN to /128.
        // 0 disables. svh is n-indexed -> plain prefix slice.
        let lm_head_draft: Option<Quad> = if mtp.is_some() {
            let n_full = lm_head.n;
            let env_n: usize = crate::opts::var(crate::opt!("exl3-mtp-head-n"))
                .ok().and_then(|v| v.parse().ok()).unwrap_or(65536);
            let n2 = ((env_n.min(n_full as usize)) / 128 * 128) as i32;
            if n2 > 0 && n2 < n_full && fxp.is_some() {
                // FP4L: rows [0, n2) of the verify head in its own format (--fp4l-draft-head head, the
                // default) or an MXFP8 RTN copy of the stored rows (--fp4l-draft-head mxfp8). Draft-only: the
                // verify keeps the verify head, so served tokens / distributions are unchanged (acceptance may move).
                let fp = fxp.as_deref().unwrap();
                let mut d = if fp4l_fwd::draft_head_mode() == "mxfp8" {
                    fp4l_fwd::draft_head_mx8(&dev, fp, n2 as usize)?
                } else {
                    // head, or screen (the screen rescores these rows exactly)
                    fp4l_fwd::fx_rows_of(&dev, &stream, &lm_head, 0, n2 as usize)?
                };
                d.lmh = true;
                crate::drprintln!("FP4L pruned draft head: n {n_full} -> {n2} as {} ({:.0} MB per draft pass; --fp4l-draft-head head|mxfp8)",
                                 d.fx.as_deref().map_or("?", |f| f.kind()), d.fx.as_deref().map_or(0, |f| f.bytes()) as f64 / 1e6);
                Some(d)
            } else if n2 > 0 && n2 < n_full {
                let l0 = Launcher { dev: &dev, stream: &stream };
                let nb_full = (n_full / 16) as usize;
                let nb_keep = (n2 / 16) as usize;
                let bw = (16 * lm_head.bits) as i64;
                let words = (lm_head.tr.len() / nb_full) * nb_keep;
                let mut tr2 = dev.alloc_zeros::<u16>(words)?;
                let total = words as u32;
                xqlaunch!(l0, "xq_slice_trellis_u16",
                          (((total) + 255) / 256, 1, 1), (256, 1, 1), 0,
                          (&mut tr2, &lm_head.tr, nb_full as i32, nb_keep as i32, bw))?;
                let mut svh2 = dev.alloc_zeros::<u16>(n2 as usize)?;
                xqlaunch!(l0, "xq_copy_u16", (((n2 as u32) + 255) / 256, 1, 1), (256, 1, 1), 0,
                          (&mut svh2, &lm_head.svh, n2 as i64, 0i64, 0i64))?;
                let mut suh2 = dev.alloc_zeros::<u16>(lm_head.suh.len())?;
                xqlaunch!(l0, "xq_copy_u16", (((lm_head.suh.len() as u32) + 255) / 256, 1, 1), (256, 1, 1), 0,
                          (&mut suh2, &lm_head.suh, lm_head.suh.len() as i64, 0i64, 0i64))?;
                l0.dev.synchronize()?;
                let draft = Quad {
                    tr: tr2,
                    suh: suh2,
                    svh: svh2,
                    bits: lm_head.bits, k2: lm_head.k2,
                    k: lm_head.k,
                    n: n2,
                    ks: chain_ks(lm_head.k, n2),
                    lmh: true, // W3/LMH: the draft slice is the same lm_head class
                    cnt: dev.htod_sync_copy(&vec![0u32; ((n2.max(0) as usize) / 128).max(1)])?, // W4/DENSE, explicit zero
                    rowpar: false,
                    fx: None,
                };
                crate::dprintln!("EXL3-MTP pruned draft head: n {n_full} -> {n2} (trellis {:.0} MB -> {:.0} MB)",
                         (lm_head.tr.len() * 2) as f64 / 1e6, (words * 2) as f64 / 1e6);
                Some(draft)
            } else {
                if env_n == 0 { crate::dprintln!("EXL3-MTP pruned draft head disabled (--exl3-mtp-head-n=0)"); }
                None
            }
        } else { None };
        // FP4L --fp4l-draft-head screen (draft-only): the int4 screen of the draft head's stored rows. The draft head is
        // replicated (every TP rank drafts the same rows of the full head), so every rank builds the same screen.
        // Needs the stored-bf16 head; anything else serves the full draft head with one line saying so.
        let fx_dhs: Option<fp4l_fwd::FxDhs> = match (fxp.as_deref(), lm_head_draft.as_ref()) {
            (Some(fp), Some(qd)) if fp4l_fwd::draft_head_mode() == "screen" => {
                let bf = matches!(qd.fx.as_deref(), Some(fp4l_fwd::FxW::Bf16 { .. }));
                if !bf {
                    crate::rprintln!("[fp4l] --fp4l-draft-head screen: serving the full draft head instead (the draft head is not \
                                      stored bf16: --fp4l-lm-head)");
                    None
                } else {
                    let t0 = std::time::Instant::now();
                    let d = fp4l_fwd::dhs_build(&dev, fp, qd.n as usize)?;
                    crate::rprintln!("[fp4l] draft-head screen: int4 copy of the {} draft-head rows ({:.0} MB, {:.1} s) picks {} candidate \
                                      128-row blocks per draft pass, rescored exactly (draft-only: served output unchanged)",
                                     d.n2, (d.n2 * d.k / 2) as f64 / 1e6, t0.elapsed().as_secs_f32(), fp4l_fwd::DHS_T);
                    Some(d)
                }
            }
            _ => None,
        };
        // DHEAD (w2): the draft passes' screen + exact rescore over the slice above (draft-only).
        let dhead: Option<dhead::DHead> = match lm_head_draft.as_ref() {
            Some(qd) if qd.fx.is_some() => {
                crate::drprintln!("DHEAD: the screen + exact rescore is trellis-only; the FP4L draft passes run the {} draft head GEMV",
                                 qd.fx.as_deref().map_or("?", |f| f.kind()));
                None
            }
            Some(qd) if !dhead::dhead_off() => match dhead::DHead::build(&dev, &stream, qd) {
                Ok(d) => Some(d),
                Err(e) => { crate::dprintln!("DHEAD: NOT built ({e:#}) — the old draft head serves"); None }
            },
            Some(_) => { crate::dprintln!("DHEAD: off (--dhead-off=1) — the old draft head serves"); None }
            None => None,
        };

        // S-A3-p: --hc-int8 — quantize every decode/verify mixer (trunk, final, MTP head).
        let mut layers = layers;
        let mut hc_mixer = hc_mixer;
        let mut mtp = mtp;
        if hc_int8_enabled() && cfg.hc_lowrank > 0 { // EXL27B: plain-residual packs have no mixer to quantize
            let t0q = std::time::Instant::now();
            let (lr, rw) = (cfg.hc_lowrank, cfg.hidden_size * cfg.hc_count.max(1));
            let mut n = 0usize;
            for l in layers.iter_mut() {
                hc_quantize_i8(&dev, &mut l.hc_attn, lr, rw, wp11_cap.0 > 0)?;
                hc_quantize_i8(&dev, &mut l.hc_mlp, lr, rw, wp11_cap.0 > 0)?;
                n += 2;
            }
            hc_quantize_i8(&dev, &mut hc_mixer, lr, rw, wp11_cap.0 > 0)?;
            n += 1;
            if let Some(h) = mtp.as_mut() {
                hc_quantize_i8(&dev, &mut h.hc_attn, lr, rw, wp11_cap.0 > 0)?;
                hc_quantize_i8(&dev, &mut h.hc_mlp, lr, rw, wp11_cap.0 > 0)?;
                hc_quantize_i8(&dev, &mut h.mixer, lr, rw, wp11_cap.0 > 0)?;
                n += 3;
            }
            crate::dprintln!("EXL3 hc mixers: {n} quantized to int8 (per-row fp32 scales) in {:.1}s — DEFAULT; --hc-fp16 = bit-exact fp16 mixer",
                     t0q.elapsed().as_secs_f32());
        }

        let snap = if crate::opts::var(crate::opt!("exl3-prefix")).as_deref() == Ok("1") {
            let rw = cfg.hc_count.max(1) * cfg.hidden_size;
            let mk = |v: &Vec<CudaSlice<f32>>| -> Result<Vec<CudaSlice<f32>>> {
                v.iter().map(|b| Ok(dev.alloc_zeros::<f32>(b.len())?)).collect()
            };
            let ps = PrefixSnap {
                conv: mk(&conv_state)?,
                s: mk(&s_state)?,
                ple: dev.alloc_zeros::<u16>(ple_state.len())?,
                carry: dev.alloc_zeros::<f32>(width * rw)?,
                hist: std::sync::Mutex::new(vec![ple_eos; width * 2]),
            };
            let mb: usize = ps.s.iter().map(|b| b.len() * 4).sum::<usize>() + ps.conv.iter().map(|b| b.len() * 4).sum::<usize>();
            crate::dprintln!("  prefix cache: per-slot recurrent-state snapshots ({:.0} MB for {width} slot(s))", mb as f64 / 1e6);
            Some(ps)
        } else { None };

        // W4/DENSE: the per-job grid plan + entry prewarm (feature marker); when it is off (p5e
        // default; opt-in --w4dense=1), the w3-PSK persistent split-K grid chosen for every chain
        // shape (p3: OFF unless --psk=1), logged once — the p3 boot sequence.
        if fxp.is_none() { // FP4L: W4/DENSE and PSK are trellis chain packages (FP4L chains never reach them)
            dense::log_and_prewarm(&dev, &layers, mtp.as_ref());
            if !dense::dense_on() { psk_log_shapes(&dev, &layers, mtp.as_ref()); }
            else if tune::reachable(&T_PSK_ON).contains(&1) { let _ = psk_slots(&dev); } // TUNE T0c: outside any capture
        }
        wp21_log_boot(); // WP21 feature marker (the split-K chain tail in effect)
        let w4s_lptr = w4s::w4s_build_lptr(&dev, &s_state, &conv_state)?;
        // TP-H #4 (--tp-vp-head, default on): this rank's column shard of the lm_head for the
        // vocab-parallel greedy verify / plain-step tail (xtp.rs vp_greedy_tail). The lmh GEMM is
        // column-tile independent (ks = 1, 128-column tiles), so shard columns [r*n/2, (r+1)*n/2)
        // are bitwise the replicated head's. The replicated head stays for prefill, the draft
        // slice and the sampled/penalized tails. World 2 or 4 (TP-4C: a multi-round key merge; the TP-H2 row
        // all-gather: world 2 = one exchange per group, world 4 = TP-4H2's recursive doubling, two).
        let vp_rw = tp.as_ref().map(|t| (t.rank, t.world));
        let lm_head_vp: Option<VpHead> = match vp_rw {
            Some((rank, world)) if lm_head.fx.is_some() && xtp::vp_head_on() && (world == 2 || world == 4)
                && xtp::dec_xport_mode() == 1 && (lm_head.n as usize) % (world as usize * 128) == 0
                && crate::opts::var(crate::opt!("mtp-dump")).map_or(true, |v| v.is_empty())
                && crate::opts::var(crate::opt!("step-topk-dir")).is_err() => {
                // FP4L: this rank's rows of the f16 head (the xf_f16w GEMM is N-tile independent at ks = 1,
                // so the shard's logits are bitwise the replicated head's columns)
                let n_sh = lm_head.n / world;
                let q = fp4l_fwd::fx_rows_of(&dev, &stream, &lm_head, (rank * n_sh) as usize, ((rank + 1) * n_sh) as usize)?;
                anyhow::ensure!(q.ks == 1, "FP4L vocab shard ks {} != 1", q.ks);
                xtp::set_vp_rows_cols(n_sh as usize);
                crate::drprintln!("TP-H vocab-parallel lm_head (FP4L {}): rank {rank}/{world} shard rows [{}, {}) ({:.0} MB)",
                                 q.fx.as_deref().map_or("?", |f| f.kind()), rank * n_sh, (rank + 1) * n_sh,
                                 q.fx.as_deref().map_or(0, |f| f.bytes()) as f64 / 1e6);
                Some(VpHead { q, base: rank * n_sh, n_sh })
            }
            Some((rank, world)) if lm_head.fx.is_none() && xtp::vp_head_on() && (world == 2 || world == 4) && xtp::dec_xport_mode() == 1
                && lm_head.lmh && lm_head.ks == 1 && (lm_head.n as usize) % (world as usize * 128) == 0
                && crate::opts::var(crate::opt!("mtp-dump")).map_or(true, |v| v.is_empty())
                && crate::opts::var(crate::opt!("step-topk-dir")).is_err() => {
                let l0 = Launcher { dev: &dev, stream: &stream };
                let n_full = lm_head.n;
                let n_sh = n_full / world;
                let nb_full = (n_full / 16) as usize;
                let nb_keep = (n_sh / 16) as usize;
                let nb_off = rank as usize * nb_keep;
                let bw = (16 * lm_head.bits) as i64;
                anyhow::ensure!(lm_head.tr.len() % nb_full == 0, "lm_head trellis {} words not a multiple of nb {nb_full}", lm_head.tr.len());
                let words = (lm_head.tr.len() / nb_full) * nb_keep;
                let mut tr2 = dev.alloc_zeros::<u16>(words)?;
                xqlaunch!(l0, "xq_slice_trellis_u16_off",
                          ((words as u32 + 255) / 256, 1, 1), (256, 1, 1), 0,
                          (&mut tr2, &lm_head.tr, nb_full as i32, nb_keep as i32, bw, nb_off as i32))?;
                let mut svh2 = dev.alloc_zeros::<u16>(n_sh as usize)?;
                xqlaunch!(l0, "xq_copy_u16", (((n_sh as u32) + 255) / 256, 1, 1), (256, 1, 1), 0,
                          (&mut svh2, &lm_head.svh, n_sh as i64, 0i64, (rank * n_sh) as i64))?;
                let mut suh2 = dev.alloc_zeros::<u16>(lm_head.suh.len())?;
                xqlaunch!(l0, "xq_copy_u16", (((lm_head.suh.len() as u32) + 255) / 256, 1, 1), (256, 1, 1), 0,
                          (&mut suh2, &lm_head.suh, lm_head.suh.len() as i64, 0i64, 0i64))?;
                l0.dev.synchronize()?;
                let q = Quad {
                    tr: tr2, suh: suh2, svh: svh2, bits: lm_head.bits, k2: lm_head.k2, k: lm_head.k, n: n_sh,
                    ks: chain_ks(lm_head.k, n_sh), lmh: true,
                    cnt: dev.htod_sync_copy(&vec![0u32; ((n_sh.max(0) as usize) / 128).max(1)])?, // explicit zero
                    rowpar: false,
                    fx: None,
                };
                anyhow::ensure!(q.ks == 1, "vocab shard chain ks {} != 1 (the replicated head's reduction order)", q.ks);
                crate::dprintln!("TP-H vocab-parallel lm_head: rank {rank}/{world} shard columns [{}, {}) ({:.0} MB trellis) — \
                          greedy verify/step tail = shard GEMM + argmax + {} key exchange(s) (--tp-vp-head=0 = replicated)",
                         rank * n_sh, (rank + 1) * n_sh, (words * 2) as f64 / 1e6, world.trailing_zeros());
                if world == 2 {
                    // TP-H2: the sampled / penalized / ratio-rule rows' tail (row all-gather of the shard rows)
                    xtp::set_vp_rows_cols(n_sh as usize);
                    crate::dprintln!("TP-H2 vocab-parallel sampled rows: {} — sampled / penalized / ratio-rule step+verify tail = shard GEMM + \
                              row all-gather ({} rows per 1 MiB epoch, {:.0} KB per row per rank) + the unchanged sampler \
                              (--tp-vp-sampled {})",
                             if xtp::vp_sampled_on() { "ON" } else { "OFF (the replicated head)" }, xtp::VP_GATHER_ROWS,
                             (n_sh as f64) * 2.0 / 1e3, xtp::vp_sampled_desc(xtp::vp_sampled_code()));
                } else if world == 4 {
                    // TP-4H2: the same tail at world 4 — recursive-doubling row all-gather (2 epochs per group of
                    // <= 4 rows: the own shard, then the two shards held), default ON (D-TP4-3). The state printed
                    // is the RESOLVED code (the head's flag, installed on every rank before the model load).
                    xtp::set_vp_rows_cols(n_sh as usize);
                    crate::dprintln!("TP-4H2 vocab-parallel sampled rows (world 4): {} — sampled / penalized / ratio-rule step+verify tail = \
                              shard GEMM + 2-round recursive-doubling row all-gather ({} rows per group = 2 epochs of <= {:.0} KB, \
                              {:.0} KB per row per rank) + the unchanged sampler (--tp-vp-sampled {})",
                             if xtp::vp_sampled_on() { "ON" } else { "OFF (the replicated head)" }, xtp::VP_GATHER_ROWS,
                             (2 * xtp::VP_GATHER_ROWS as u64 * n_sh as u64 * 2) as f64 / 1e3,
                             (n_sh as f64) * 2.0 / 1e3, xtp::vp_sampled_desc(xtp::vp_sampled_code()));
                } else {
                    // any other world: no row all-gather kernels — the replicated head (still resident) serves the
                    // sampled rows; vp_rows keeps its 1-element placeholder
                    crate::dprintln!("TP-H2 vocab-parallel sampled rows: OFF at world {world} (the row all-gather has kernels at world 2 and 4 \
                              only; sampled / penalized / ratio-rule rows take the replicated head)");
                }
                Some(VpHead { q, base: rank * n_sh, n_sh })
            }
            _ => None,
        };
        // TP-H #3 (--tp-dh-shard, default on): each rank screens 1/world of the draft head's blocks and the
        // doorbell reassembles the replicated block-maxima array bit for bit (dhead.rs / xtp.rs).
        let mut dhead = dhead;
        if let (Some(dh), Some((rank, world))) = (dhead.as_mut(), vp_rw) {
            if xtp::dh_shard_on() && (world == 2 || world == 4) && xtp::dec_xport_mode() == 1 {
                dh.enable_shard(&dev, rank as usize, world as usize)?;
                if let Some((a, b, nblk)) = dh.shard_span() {
                    crate::dprintln!("TP-H sharded MTP-head screen: rank {rank}/{world} screens blocks [{a}, {b}) of {nblk} \
                              (1/{world} of the screen bytes per draft pass; bmax reassembled through the doorbell, \
                              bitwise the replicated screen; --tp-dh-shard=0 = replicated)");
                }
            }
        }
        // TP-A: attach the RDMA link LAST (every weight is resident and shape-asserted first).
        let tp_state = match tp {
            Some(t) => Some(xtp::TpState::attach(&dev, &stream, t, cfg.hidden_size)?),
            None => None,
        };

        // FP4L: the precision context + ONE boot line naming every lever's resolved value (the A/B the owner
        // decides the defaults from reads it; /metrics velogb10_fp4l_precision carries the same values)
        let fxc = if fxp.is_some() {
            let hc_cap = if fx_plain && cfg.hc_lowrank > 0 { fp4l_fwd::xf_co_residency("xf_hc_fuse_b32") } else { 0 };
            let kv = kv_fmt_name(kv_fmt);
            crate::rprintln!("[fp4l] precision levers: --fp4l-plain {} | --fp4l-norms {} | --fp4l-lm-head {} | --fp4l-draft-head {} | \
                              --fp4l-prefill-dense {} | --kv-cache {kv}{} (defaults = the full-precision ends, provisional until the A/B; \
                              PLAN/FP4L_DESIGN.md §3a){}",
                             if fx_plain { "exact" } else { "f16" }, if fx_norms { "f32" } else { "f16" },
                             lm_head.fx.as_deref().map_or("?", |f| f.kind()), fp4l_fwd::draft_head_mode(),
                             if fp4l_fwd::prefill_dense_lt() { "lt" } else { "exact" },
                             if kv != "q8" && crate::lane_pack::kv_any() { " (lane packing ungated on it: --lane-pack-kv any)" } else { "" },
                             if fx_plain && cfg.hc_lowrank > 0 && fp4l_fwd::hc_fuse_grid(hc_cap * 5 / 6) == 0 {
                                 " — xf_hc_fuse_b32 co-residency too small (< 20 CTAs): the exact mixers run unfused (down GEMM, silu, up GEMM) at every width"
                             } else { "" });
            crate::metrics::fp4l_precision(fx_plain, fx_norms, lm_head.fx.as_deref().map_or("?", |f| f.kind()),
                                           fp4l_fwd::draft_head_mode(), !fp4l_fwd::prefill_dense_lt());
            Some(fp4l_fwd::FxCtx { plain: fx_plain, norms: fx_norms, hc_cap })
        } else { None };

        Ok(Arc::new(FwdModel {
            tc,
            dev,
            cfg,
            rope: fr,
            embed,
            embed_bf: fx_plain,
            fxc,
            layers,
            hc_mixer,
            lm_head,
            lm_head_draft,
            fx_dhs,
            lm_head_vp,
            dhead,
            conv_state,
            s_state,
            w4s_lptr,
            width,
            k_cache,
            kv_fmt,
            max_pos,
            bytes_per_round: bytes_round,
            ple,
            ple_state,
            ple_hist: std::sync::Mutex::new(vec![ple_eos; width * 2]),
            ple_eos,
            snap,
            mtp,
            stream,
            cos_tab,
            rope_map,
            rope_plen,
            rope_delta,
            mrope_tab,
            qm_cos,
            qm_sin,
            qsa_mrope,
            qsa_full,
            cos_gather_ok,
            graphs: std::sync::Mutex::new(std::collections::HashMap::new()),
            qsa_keys: qsa_keys_vec,
            qsa_pool: qsa_pool_vec,
            qsa_cos,
            qsa_sin,
            hc_fuse_cap,
            wp11_cap,
            w4hc_grid,
            w4hc_cap,
            rc_lut,
            esel_hist,
            expert_hist,
            route_log,
            qsa_dbg,
            qsa_rowdbg,
            tp: tp_state,
            // EXL27C-DF2 (fwd): capture disarmed at construction (zero-cost until the serve
            // agent calls set_df2_capture); the boot buffers build lazily in df2_boot_ptrs.
            df2_sink: None,
            df2_prime: None,
            df2_head_bf16: std::sync::Mutex::new(None),
            df2_embed_bf16: std::sync::Mutex::new(None),
            df2_q_head: std::sync::OnceLock::new(),
        }))
    }
}
