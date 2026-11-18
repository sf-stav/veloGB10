//! `impl GpuModel`: the qwen4_exp (Flash-Next) side of the batch kernels: hyper-connections, PLE (incl. the SSD-offloaded
//! table), `ple_forward`, `embed_*_resid*`, and the MTP mixing helpers. Everything is family-gated by `cfg.is_q4()` at call sites. Map: `mod.rs`.

use super::*;

// =====================================================================================================
// qwen4_exp (Qwen3.8-Flash-Next): hyper-connections + PLE — the GpuModel side of the kernels at the
// end of kernels/gpu_batch.cu. Everything here is family-gated by `cfg.is_q4()` at the call sites.
// =====================================================================================================
impl GpuModel {
    /// `--ple-offload ssd` (env [ple-offload]=ssd): keep the n-gram table on disk and `pread` the
    /// rows each forward needs. Costs one host round-trip per forward (graphs off); saves ~31 GB.
    pub fn ple_offload_ssd() -> bool {
        crate::opts::var(crate::opt!("ple-offload")).map(|v| v.eq_ignore_ascii_case("ssd")).unwrap_or(false)
    }

    /// The GDN output-gate norm kernel: silu gate (qwen3_5) or sigmoid gate (qwen4_exp).
    pub(crate) fn gdn_gate_kernel(&self) -> &'static str {
        if self.cfg.gdn_gate_sigmoid { "rmsnorm_gated_sig_b" } else { "rmsnorm_gated_b" }
    }

    /// CUDA-graph decode is impossible when the PLE table is on SSD (the gather is a host round-trip
    /// inside the forward). Everything else captures as before.
    pub fn decode_graphs_supported(&self) -> bool {
        !self.layers.iter().any(|l| matches!(l.ple.as_ref().map(|p| &p.table), Some(PleTable::Ssd(_))))
    }

    /// Attach the PLE n-gram table to an assembled PLE layer: device-resident (the whole 96-B record
    /// file uploaded in 1 GB pieces) or the SSD reader. Validates the sidecar against the hash
    /// geometry derived from config.json.
    pub(super) fn attach_ple_table(dev: &Arc<CudaDevice>, ple: &mut GpuPle, model_dir: &std::path::Path) -> anyhow::Result<()> {
        let meta = crate::ple::PleTableMeta::load(model_dir)?;
        anyhow::ensure!(meta.total_rows == ple.hash.total_rows,
            "PLE table has {} rows, config.json implies {} (vocab base / heads mismatch)", meta.total_rows, ple.hash.total_rows);
        ple.gs = dev.htod_sync_copy(&meta.shard_global_scales)?;
        ple.rows_per_shard = meta.rows_per_shard;
        let t0 = std::time::Instant::now();
        if Self::ple_offload_ssd() {
            let ssd = crate::ple::PleSsd::open(meta)?;
            println!("PLE n-gram table: SSD-resident ({}, {} rows, {:.1} GB on disk) — rows are read per forward",
                     ssd.meta.file.display(), ssd.meta.total_rows, (ssd.meta.total_rows * ssd.meta.record_bytes) as f64 / 1e9);
            ple.table = PleTable::Ssd(ssd);
        } else {
            use std::io::Read;
            let total = meta.total_rows * meta.record_bytes;
            let buf = dev.alloc_zeros::<u8>(total)?;
            let mut f = std::fs::File::open(&meta.file)?;
            let mut chunk = vec![0u8; 1 << 30];
            let mut off = 0usize;
            while off < total {
                let n = (total - off).min(chunk.len());
                f.read_exact(&mut chunk[..n])?;
                unsafe {
                    cudarc::driver::result::memcpy_htod_sync(*buf.device_ptr() + off as u64, &chunk[..n])?;
                }
                off += n;
            }
            dev.synchronize()?;
            println!("PLE n-gram table: device-resident ({:.1} GB, {} rows) in {:.1}s",
                     total as f64 / 1e9, meta.total_rows, t0.elapsed().as_secs_f32());
            ple.table = PleTable::Device(buf);
        }
        Ok(())
    }

    /// Probe helper: the LM-head logits (f32, host) of the LAST column of a prefill's returned
    /// residual (`hout`, the pre-final-norm/mixer backbone hidden). Family-agnostic.
    pub fn probe_logits_of_hidden_col(&self, pool: &mut Pool, hout: &B, col: usize) -> Vec<f32> {
        let cfg = &self.cfg;
        let h = cfg.hidden_size; let rw = cfg.resid_width(); let v = cfg.vocab_size;
        let src = *hout.device_ptr() as u64 + (col * rw * 2) as u64;
        let out = pool.get_bf16(h);
        if cfg.is_q4() {
            self.hc_pre(pool, &out, None, src, self.hc_mixer.as_ref().unwrap(), 1);
        } else {
            blaunch!(self, "rmsnorm_b", (1u32,1,1), (1024,1,1), 4096u32, (d(&out), src, d(&self.final_norm), h as i32, 1i32, fbits(cfg.rms_eps)));
        }
        let logits = self.logits_batch(pool, &out, 1);
        self.sync_stream();
        let lh: Vec<half::bf16> = self.dev.dtoh_sync_copy(&logits).unwrap();
        pool.release_bf16(out, h); pool.release_bf16(logits, v);
        lh[..v].iter().map(|x| x.to_f32()).collect()
    }

    /// Probe helper: one decode step returning the raw (pre-penalty) logits on the host and the
    /// greedy token. `bufs.tokens_dev/pos_dev` must be written and synced by the caller.
    pub fn probe_decode_logits(&self, pool: &mut Pool, bufs: &mut DecodeBuffers, state: &mut BatchGpuState,
                               kv_stride: usize, max_pc: usize) -> (Vec<f32>, u32) {
        let v = self.cfg.vocab_size;
        let lg = self.forward_decode_logits(pool, bufs, state, kv_stride, max_pc, 1);
        self.sync_stream();
        let lh: Vec<f32> = match lg {
            DecodeLogits::Bf16(b) => { let x: Vec<half::bf16> = self.dev.dtoh_sync_copy(&b).unwrap(); pool.release_bf16(b, v); x[..v].iter().map(|x| x.to_f32()).collect() }
            DecodeLogits::F32(f) => { let x: Vec<f32> = self.dev.dtoh_sync_copy(&f).unwrap(); pool.release(f, v); x[..v].to_vec() }
        };
        let mut best = 0usize;
        for i in 1..v { if lh[i] > lh[best] { best = i; } }
        (lh, best as u32)
    }

    /// Zero a slot's PLE state (conv window) and reset its token ring to EOS (= empty history).
    pub(super) fn ple_zero_slot(&self, state: &mut BatchGpuState, slot: usize) {
        let cfg = &self.cfg;
        let Some(cs) = state.ple_conv_state.as_ref() else { return };
        let ring = state.ple_ring.as_ref().unwrap();
        let l = cfg.ple_conv_state_len(); let rw = cfg.resid_width();
        let ri = cfg.ple_ngram_size - 1;
        let stream = self.stream.stream;
        unsafe {
            let sb = l * rw * 4;
            cudarc::driver::sys::cuMemsetD8Async((*cs.device_ptr() as u64 + (slot * sb) as u64) as cudarc::driver::sys::CUdeviceptr, 0, sb, stream);
            cudarc::driver::sys::cuMemsetD32Async((*ring.device_ptr() as u64 + (slot * ri * 4) as u64) as cudarc::driver::sys::CUdeviceptr, cfg.eos_token_id as u32, ri, stream);
        }
    }

    /// Copy a slot's PLE state + ring (snapshot / rollback), on the compute stream.
    pub(super) fn ple_copy_slot(&self, state: &BatchGpuState, src: usize, dst: usize) {
        self.ple_copy_slot_idx(state, src, dst, 0, 0);
    }
    /// Same, with the destination slot read on device from `dst_ids[idx]` when `dst_ids != 0`.
    fn ple_copy_slot_idx(&self, state: &BatchGpuState, src: usize, dst: usize, dst_ids: u64, idx: usize) {
        let cfg = &self.cfg;
        let Some(cs) = state.ple_conv_state.as_ref() else { return };
        let ring = state.ple_ring.as_ref().unwrap();
        let sf = cfg.ple_conv_state_len() * cfg.resid_width();
        let ri = cfg.ple_ngram_size - 1;
        blaunch!(self, "ple_slot_copy_b", grid(sf.max(ri)), (256,1,1), 0,
            (d(cs), *ring.device_ptr() as u64, sf as u64, ri as i32, src as i32, dst as i32, dst_ids, idx as i32));
    }

    /// Hyper-connection PRE: from the stream stack at `resid_ptr` [rw, batch] produce the sublayer
    /// input `out` [h, batch] and (when `inj` is given and the block has an inject weight) the
    /// per-stream injection weights `inj` [hc, batch] f32.
    pub(crate) fn hc_pre(&self, pool: &mut Pool, out: &B, inj: Option<&S>, resid_ptr: u64, hc: &GpuHc, batch: usize) {
        let cfg = &self.cfg;
        let h = cfg.hidden_size; let hcn = cfg.hc_count; let rw = cfg.resid_width(); let lr = cfg.hc_lowrank;
        let hn = pool.get_bf16(rw * batch);
        blaunch!(self, "hc_norm_b", ((batch * hcn) as u32,1,1), (1024,1,1), 4096u32,
            (d(&hn), resid_ptr, d(&hc.norm), h as i32, hcn as i32, batch as i32, fbits(cfg.rms_eps)));
        let mut dd = pool.get_bf16(lr * batch);
        self.gemm_act(&hc.down, &hn, &mut dd, rw, lr, batch);
        blaunch!(self, "silu_div_b", grid(lr * batch), (256,1,1), 0, (d(&dd), hcn as f32, (lr * batch) as i32));
        let mut uu = pool.get_bf16(rw * batch);
        self.gemm_act(&hc.up, &dd, &mut uu, lr, rw, batch);
        let (inj_ptr, winj_ptr): (u64, u64) = match (inj, &hc.inject) {
            (Some(i), Some(w)) => (d(i), d(w)),
            _ => (0, 0),
        };
        blaunch!(self, "hc_mix_b", (batch as u32,1,1), (1024,1,1), 4096u32,
            (d(out), inj_ptr, d(&hn), d(&uu), winj_ptr, h as i32, hcn as i32, batch as i32));
        pool.release_bf16(hn, rw * batch);
        pool.release_bf16(dd, lr * batch);
        pool.release_bf16(uu, rw * batch);
    }

    /// Hyper-connection POST: resid[s] += inj[s] * out, for every stream s.
    pub(crate) fn hc_post(&self, resid_ptr: u64, out: &B, inj: &S, batch: usize) {
        let cfg = &self.cfg;
        let h = cfg.hidden_size; let hcn = cfg.hc_count; let rw = cfg.resid_width();
        blaunch!(self, "hc_inject_b", grid(rw * batch), (256,1,1), 0,
            (resid_ptr, d(out), d(inj), h as i32, hcn as i32, batch as i32));
    }

    /// Embedding gather into the RESIDUAL layout: [h, batch] on the classic families, the hc-stream
    /// stack [rw, batch] (the embedding replicated per stream) on qwen4_exp.
    pub(crate) fn embed_gather_resid(&self, pool: &mut Pool, out_ptr: u64, toks_ptr: u64, batch: usize) {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        if !cfg.is_q4() { self.embed_gather(out_ptr, toks_ptr, h, batch); return; }
        let tmp = pool.get_bf16(h * batch);
        self.embed_gather(d(&tmp), toks_ptr, h, batch);
        let hcn = cfg.hc_count;
        blaunch!(self, "hc_expand_b", grid(h * hcn * batch), (256,1,1), 0,
            (out_ptr, d(&tmp), h as i32, hcn as i32, batch as i32));
        pool.release_bf16(tmp, h * batch);
    }

    /// `embed_batch` in the residual layout (see `embed_gather_resid`).
    pub fn embed_batch_resid(&self, tokens: &[u32]) -> B { self.embed_batch_resid_vision(tokens, None, 0) }

    /// `embed_batch_resid` with the V3 image-embed splice applied to the token embeddings BEFORE
    /// the hyper-connection expansion (the residual is `[hc*h, n]`; the spans are h-wide rows of
    /// the embedding, exactly what `splice_vision` overwrites on the other families).
    pub fn embed_batch_resid_vision(&self, tokens: &[u32], vision: Option<(&B, &[crate::vision_encoder::ImageSpan])>, pos_start: usize) -> B {
        if !self.cfg.is_q4() {
            let r = self.embed_batch(tokens);
            if let Some((ve, spans)) = vision { self.splice_vision(ve, spans, &r, pos_start, tokens.len()); }
            return r;
        }
        let h = self.cfg.hidden_size; let rw = self.cfg.resid_width();
        let b = tokens.len();
        let toks_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let toks_dev = self.dev.htod_sync_copy(&toks_i32).expect("htod tokens");
        let tmp = self.dev.alloc_zeros::<half::bf16>(h * b).unwrap();
        self.embed_gather(d(&tmp), *toks_dev.device_ptr() as u64, h, b);
        if let Some((ve, spans)) = vision { self.splice_vision(ve, spans, &tmp, pos_start, b); }
        let hidden = self.dev.alloc_zeros::<half::bf16>(rw * b).unwrap();
        blaunch!(self, "hc_expand_b", grid(rw * b), (256,1,1), 0,
            (d(&hidden), d(&tmp), h as i32, self.cfg.hc_count as i32, b as i32));
        self.sync_stream();   // tmp/toks_dev (non-pool) drop at return
        hidden
    }

    /// The PLE layer forward: resid[rw, n] += PLE(resid, tokens). `decode`: n independent lanes,
    /// each on its own slot (`slot_ids_ptr`), in-place state update. Otherwise the n columns are a
    /// chain (`parent_ptr == 0`) or a verify tree (`parent_ptr` = the packed DFS parents) over one
    /// slot (`slot0`, or per-column `slot_ids_ptr` for a forest); the committed state of the
    /// source slot is read, the last column's state is committed to it (through the scratch
    /// slot), and `ckpt_slot` writes one checkpoint per column at ckpt+t (the GDN contract).
    pub(crate) fn ple_forward(&self, pool: &mut Pool, resid_ptr: u64, ple: &GpuPle, state: &BatchGpuState,
                              tokens_ptr: u64, slot_ids_ptr: u64, slot0: usize, parent_ptr: u64, chain: bool,
                              n: usize, ckpt_slot: Option<usize>, decode: bool) {
        assert!(tokens_ptr != 0, "qwen4_exp PLE needs the column token ids on device");
        assert!(n <= ple.stage_rows, "PLE staging holds {} tokens, forward has {n}", ple.stage_rows);
        let cfg = &self.cfg;
        let h = cfg.hidden_size; let hcn = cfg.hc_count; let rw = cfg.resid_width();
        let heads = ple.hash.ngram_heads(); let ng = ple.hash.ngram_size; let hpn = ple.hash.heads_per_ngram;
        // S-B15 tripwire: the PLE conv is FULL-width — a sharded (rw/2) weight here means a
        // load-time shard rule leaked onto a PLE tensor again; fail loudly, never run garbage.
        assert_eq!(ple.conv1d.len(), rw * cfg.ple_conv_kernel,
                   "ple.conv1d is {len} f32, expected rw*ck = {rw}*{ck} — a TP shard rule leaked onto a PLE tensor",
                   len = ple.conv1d.len(), ck = cfg.ple_conv_kernel);
        let pdim = cfg.ple_embed_dim;
        let l = cfg.ple_conv_state_len(); let kk = cfg.ple_conv_kernel; let dil = ng;
        let cs = state.ple_conv_state.as_ref().expect("PLE state");
        let ring = state.ple_ring.as_ref().unwrap();
        let ring_ptr = *ring.device_ptr() as u64;
        let chain_i = if chain { 1i32 } else { 0i32 };
        // B15 diagnostic (--ple-trace=<dir>): dump PLE intermediates per call — localize the
        // TP=2 layer-1 nondeterminism. DIAGNOSTIC ONLY; off in production.
        let ple_trace = crate::opts::var(crate::opt!("ple-trace")).ok();
        static PLE_TN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let tn = if ple_trace.is_some() { PLE_TN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) } else { 0 };
        if let Some(dir) = &ple_trace {
            self.sync_stream();
            let mut h = vec![0u8; rw * n * 2];
            unsafe { cudarc::driver::result::memcpy_dtoh_sync(&mut h, resid_ptr).expect("ple trace resid"); }
            let _ = std::fs::write(format!("{dir}/tn{tn:02}_rank{}_resid_in.bin", self.tp_rank), &h);
            let nr_ring = (state.state_slots + 1) * (ng as usize - 1);
            let mut hr = vec![0i32; nr_ring];
            unsafe { cudarc::driver::result::memcpy_dtoh_sync(&mut hr, ring_ptr).expect("ple trace ring"); }
            let _ = std::fs::write(format!("{dir}/tn{tn:02}_rank{}_ring.bin", self.tp_rank), bytemuck::cast_slice(&hr));
            let mut hc = vec![0f32; cs.len()];
            unsafe { cudarc::driver::result::memcpy_dtoh_sync(&mut hc, *cs.device_ptr()).expect("ple trace cs"); }
            let _ = std::fs::write(format!("{dir}/tn{tn:02}_rank{}_cs.bin", self.tp_rank), bytemuck::cast_slice(&hc));
        }
        // B15: capture the hash INPUTS (token ids from state + the t=0 ring tokens ple_hash_b reads)
        if let Some(dir) = &ple_trace {
            let mut toks = vec![0i32; n];
            unsafe { cudarc::driver::result::memcpy_dtoh_sync(&mut toks, tokens_ptr).expect("ple trace tokens"); }
            let _ = std::fs::write(format!("{dir}/tn{tn:02}_rank{}_tokens.bin", self.tp_rank), bytemuck::cast_slice(&toks));
            self.sync_stream();
            let ri = ng as usize - 1;
            let mut ring0 = vec![0i32; ri];
            unsafe { cudarc::driver::result::memcpy_dtoh_sync(&mut ring0, ring_ptr + (slot0 as u64) * (ri as u64) * 4).expect("ple trace ring0"); }
            let _ = std::fs::write(format!("{dir}/tn{tn:02}_rank{}_ring0.bin", self.tp_rank), bytemuck::cast_slice(&ring0));
        }
        // 1. row ids
        let hgeom = (ng as i32) | ((hpn as i32) << 8) | ((heads as i32) << 16) | (chain_i << 24);
        blaunch!(self, "ple_hash_b", grid(n), (256,1,1), 0,
            (d(&ple.ids), tokens_ptr, ring_ptr, slot_ids_ptr, slot0 as i32, parent_ptr,
             *ple.hash_tab.device_ptr() as u64, hgeom, ple.hash.eos, n as i32));
        let nrec = n * heads;
        if let Some(dir) = &ple_trace {
            self.sync_stream();
            let mut hi = vec![0i64; nrec];
            unsafe { cudarc::driver::result::memcpy_dtoh_sync(&mut hi, *ple.ids.device_ptr()).expect("ple trace ids"); }
            let _ = std::fs::write(format!("{dir}/tn{tn:02}_rank{}_ids.bin", self.tp_rank), bytemuck::cast_slice(&hi));
        }
        // 2. records → stage
        match &ple.table {
            PleTable::Device(t) => {
                blaunch!(self, "ple_gather_rows_b", grid(nrec * 24), (256,1,1), 0,
                    (d(&ple.stage), d(t), d(&ple.ids), nrec as i32));
            }
            PleTable::Ssd(ssd) => {
                self.sync_stream();
                let mut ids = vec![0i64; nrec];
                unsafe { cudarc::driver::result::memcpy_dtoh_sync(&mut ids, *ple.ids.device_ptr()).expect("ple ids dtoh"); }
                let mut host = vec![0u8; nrec * crate::quant::PLE_REC_BYTES];
                ssd.gather(&ids, &mut host).expect("PLE SSD gather");
                unsafe { cudarc::driver::result::memcpy_htod_sync(*ple.stage.device_ptr(), &host).expect("ple stage htod"); }
            }
        }
        // 3. dequant → emb [pdim, n]
        let emb = pool.get_bf16(pdim * n);
        blaunch!(self, "ple_dequant_rows_b", grid(nrec * 10), (256,1,1), 0,
            (d(&emb), d(&ple.stage), d(&ple.ids), d(&ple.gs), ple.rows_per_shard as i32, heads as i32, n as i32));
        if let Some(dir) = &ple_trace {
            self.sync_stream();
            let mut he = vec![0u8; pdim * n * 2];
            unsafe { cudarc::driver::result::memcpy_dtoh_sync(&mut he, *emb.device_ptr()).expect("ple trace emb"); }
            let _ = std::fs::write(format!("{dir}/tn{tn:02}_rank{}_emb.bin", self.tp_rank), &he);
        }
        // 4. key / value projections, grouped norms, gate
        let mut key = pool.get_bf16(rw * n);
        self.gemm_act(&ple.key_proj, &emb, &mut key, pdim, rw, n);
        let kn = pool.get_bf16(rw * n);
        blaunch!(self, "hc_norm_b", ((n * hcn) as u32,1,1), (1024,1,1), 4096u32,
            (d(&kn), d(&key), d(&ple.norm_key), h as i32, hcn as i32, n as i32, fbits(cfg.rms_eps)));
        let mut val = pool.get_bf16(h * n);
        self.gemm_act(&ple.value_proj, &emb, &mut val, pdim, h, n);
        let qn = pool.get_bf16(rw * n);
        blaunch!(self, "hc_norm_b", ((n * hcn) as u32,1,1), (1024,1,1), 4096u32,
            (d(&qn), resid_ptr, d(&ple.norm_query), h as i32, hcn as i32, n as i32, fbits(cfg.rms_eps)));
        let gv = pool.get_bf16(rw * n);
        blaunch!(self, "ple_gate_b", ((n * hcn) as u32,1,1), (1024,1,1), 4096u32,
            (d(&gv), d(&kn), d(&qn), d(&val), h as i32, hcn as i32, n as i32));
        let gvn = pool.get_bf16(rw * n);
        blaunch!(self, "hc_norm_b", ((n * hcn) as u32,1,1), (1024,1,1), 4096u32,
            (d(&gvn), d(&gv), d(&ple.norm_conv), h as i32, hcn as i32, n as i32, fbits(cfg.rms_eps)));
        // 5. dilated conv + add into the residual; state / ring commits
        if let Some(dir) = &ple_trace {
            self.sync_stream();
            let mut hv = vec![0u8; h * n * 2];
            unsafe { cudarc::driver::result::memcpy_dtoh_sync(&mut hv, *val.device_ptr()).expect("ple trace val"); }
            let _ = std::fs::write(format!("{dir}/tn{tn:02}_rank{}_val.bin", self.tp_rank), &hv);
            let mut hw = vec![0f32; ple.conv1d.len()];
            unsafe { cudarc::driver::result::memcpy_dtoh_sync(&mut hw, *ple.conv1d.device_ptr()).expect("ple trace conv1d"); }
            let _ = std::fs::write(format!("{dir}/tn{tn:02}_rank{}_conv1d.bin", self.tp_rank), bytemuck::cast_slice(&hw));
        }
        if decode {
            blaunch!(self, "ple_dconv_decode_b", grid(rw * n), (256,1,1), 0,
                (resid_ptr, d(&gv), d(&gvn), d(cs), d(&ple.conv1d), slot_ids_ptr, rw as i32, l as i32, kk as i32, dil as i32, n as i32));
            blaunch!(self, "ple_ring_commit_b", (n as u32,1,1), (32,1,1), 0,
                (ring_ptr, 0i32, ring_ptr, tokens_ptr, slot_ids_ptr, slot0 as i32, parent_ptr, chain_i, ng as i32, n as i32, -2i32));
        } else {
            let cgeom = (l as i32) | ((kk as i32) << 8) | ((dil as i32) << 16) | (chain_i << 24);
            blaunch!(self, "ple_dconv_prefill_b", grid(rw * n), (256,1,1), 0,
                (resid_ptr, d(&gv), d(&gvn), d(cs), slot_ids_ptr, slot0 as i32, parent_ptr,
                 d(&ple.conv1d), rw as i32, cgeom, n as i32));
            let gx = ((rw * l).div_ceil(256)) as u32;
            if let Some(ck) = ckpt_slot {
                blaunch!(self, "ple_dconv_state_b", (gx, n as u32, 1), (256,1,1), 0,
                    (d(cs), ck as i32, d(cs), slot_ids_ptr, slot0 as i32, parent_ptr, chain_i, d(&gvn), rw as i32, l as i32, n as i32, -1i32));
                blaunch!(self, "ple_ring_commit_b", (n as u32,1,1), (32,1,1), 0,
                    (ring_ptr, ck as i32, ring_ptr, tokens_ptr, slot_ids_ptr, slot0 as i32, parent_ptr, chain_i, ng as i32, n as i32, -1i32));
            }
            // Main-slot commit: the last column's state → scratch slot → the column's own slot.
            let scratch = state.ple_scratch_slot;
            blaunch!(self, "ple_dconv_state_b", (gx, 1, 1), (256,1,1), 0,
                (d(cs), scratch as i32, d(cs), slot_ids_ptr, slot0 as i32, parent_ptr, chain_i, d(&gvn), rw as i32, l as i32, n as i32, (n - 1) as i32));
            blaunch!(self, "ple_ring_commit_b", (1u32,1,1), (32,1,1), 0,
                (ring_ptr, scratch as i32, ring_ptr, tokens_ptr, slot_ids_ptr, slot0 as i32, parent_ptr, chain_i, ng as i32, n as i32, (n - 1) as i32));
            // Destination = the last column's lane slot, read ON DEVICE (slot_ids[n-1]) so the
            // commit stays capture-legal (no dtoh); uniform slot0 when there is no slot array.
            self.ple_copy_slot_idx(state, scratch, slot0, slot_ids_ptr, n - 1);
        }
        if let Some(dir) = &ple_trace {
            self.sync_stream();
            let mut hg = vec![0u8; rw * n * 2];
            unsafe { cudarc::driver::result::memcpy_dtoh_sync(&mut hg, *gvn.device_ptr()).expect("ple trace gvn"); }
            let _ = std::fs::write(format!("{dir}/tn{tn:02}_rank{}_gvn.bin", self.tp_rank), &hg);
            let mut hr2 = vec![0u8; rw * n * 2];
            unsafe { cudarc::driver::result::memcpy_dtoh_sync(&mut hr2, resid_ptr).expect("ple trace resid_out"); }
            let _ = std::fs::write(format!("{dir}/tn{tn:02}_rank{}_resid_out.bin", self.tp_rank), &hr2);
        }
        pool.release_bf16(emb, pdim * n);
        pool.release_bf16(key, rw * n); pool.release_bf16(kn, rw * n);
        pool.release_bf16(val, h * n); pool.release_bf16(qn, rw * n);
        pool.release_bf16(gv, rw * n); pool.release_bf16(gvn, rw * n);
    }
}

impl GpuModel {
    /// Width of the backbone / MTP hidden columns the scheduler moves around (`hout`, `vout`,
    /// `mtp_h_prev`): the hc-stream stack on qwen4_exp, the hidden size elsewhere.
    pub fn mtp_hidden_width(&self) -> usize { self.cfg.resid_width() }

    /// Copy one column of pitch `w` (bf16 elems) out of a column-major buffer.
    pub fn copy_col_w(&self, dst_ptr: u64, src: &B, col: usize, w: usize) {
        let stream = self.stream.stream;
        unsafe {
            let src_ptr = *src.device_ptr() as u64 + (col as u64) * (w as u64) * 2;
            cudarc::driver::result::memcpy_dtod_async(dst_ptr, src_ptr, w * 2, stream).unwrap();
        }
    }

    /// qwen4_exp: the MTP head's mixer (streams → hidden) before the LM head. None on the other
    /// families (their MTP output already IS the head input).
    pub(crate) fn mtp_mix_for_head(&self, pool: &mut Pool, streams: &B, batch: usize) -> Option<B> {
        let q = self.mtp.as_ref()?.q4.as_ref()?;
        let out = pool.get_bf16(self.cfg.hidden_size * batch);
        self.hc_pre(pool, &out, None, d(streams), &q.mixer, batch);
        Some(out)
    }

    /// qwen4_exp MTP head forward (Qwen4ExpForCausalLMMTP, sglang qwen4_exp_mtp.py):
    ///   streams = fc_hidden(rmsnorm_full(hidden_streams)) + fc_embedding(rmsnorm(embed(tok)))  [per stream]
    ///   one hyper-connected decoder layer (attention with the head's own KV, MoE)
    /// Returns the OUTPUT STREAMS [rw, batch] — the next draft step's input; `mtp_mix_for_head`
    /// turns them into the LM-head input.
    pub(super) fn mtp_forward_q4(&self, pool: &mut Pool, hidden: &B, token_ptr: u64,
                      pos_dev: &cudarc::driver::CudaSlice<i32>,
                      mtp_kc_ptr: u64, mtp_vc_ptr: u64, max_pc: usize,
                      cos: &S, sin: &S, kv_stride: usize, batch: usize,
                      prefill_pos_start: Option<usize>) -> B {
        let slot_ids_ptr = *self.mtp_sids.device_ptr() as u64;
        let cfg = &self.cfg;
        let h = cfg.hidden_size; let hcn = cfg.hc_count; let rw = cfg.resid_width();
        let mtp = self.mtp.as_ref().unwrap();
        let q = mtp.q4.as_ref().unwrap();
        // 1. hidden streams: full-width RMSNorm (1+w over all hc*h), then fc_hidden per stream — the
        //    [rw, batch] column-major buffer IS a [h, hc*batch] matrix.
        let norm_h = pool.get_bf16(rw * batch);
        blaunch!(self, "rmsnorm_b", (batch as u32,1,1), (1024,1,1), 4096u32,
            (d(&norm_h), d(hidden), d(&mtp.pre_fc_norm_hidden), rw as i32, batch as i32, fbits(cfg.rms_eps)));
        let mut streams = pool.get_bf16(rw * batch);
        self.gemm_act(&q.fc_hidden, &norm_h, &mut streams, h, h, hcn * batch);
        pool.release_bf16(norm_h, rw * batch);
        // 2. embedding term, broadcast into every stream
        let norm_e = pool.get_bf16(h * batch);
        self.embed_gather(*norm_e.device_ptr() as u64, token_ptr, h, batch);
        blaunch!(self, "rmsnorm_b", (batch as u32,1,1), (1024,1,1), 4096u32,
            (d(&norm_e), d(&norm_e), d(&mtp.pre_fc_norm_embedding), h as i32, batch as i32, fbits(cfg.rms_eps)));
        let mut fce = pool.get_bf16(h * batch);
        self.gemm_act(&q.fc_embedding, &norm_e, &mut fce, h, h, batch);
        blaunch!(self, "hc_add_bcast_b", grid(rw * batch), (256,1,1), 0,
            (d(&streams), d(&fce), h as i32, hcn as i32, batch as i32));
        pool.release_bf16(norm_e, h * batch);
        pool.release_bf16(fce, h * batch);
        // 3. the hyper-connected decoder layer
        let inj = pool.get(hcn * batch);
        let normed = pool.get_bf16(h * batch);
        self.hc_pre(pool, &normed, Some(&inj), d(&streams), &q.hc.0, batch);
        let qsa = self.qsa_in_mtp(&mtp.fa, mtp_kc_ptr, kv_stride, *pos_dev.device_ptr() as u64);
        let ReduceOut::Full(mixer) = self.full_attn_batch(pool, &normed, &mtp.fa, pos_dev, max_pc, kv_stride,
                                         mtp_kc_ptr, mtp_vc_ptr, cos, sin, slot_ids_ptr, batch,
                                         prefill_pos_start, mtp.attn_sharded, KVCacheMode::Bf16, false, qsa) else {
            unreachable!("qwen4_exp MTP: fused TP reduce is not supported")
        };
        self.hc_post(d(&streams), &mixer, &inj, batch);
        pool.release_bf16(mixer, h * batch);
        self.hc_pre(pool, &normed, Some(&inj), d(&streams), &q.hc.1, batch);
        let ReduceOut::Full(mlp_out) = self.ffn_batch(pool, &normed, &mtp.mlp, batch, mtp.mlp_sharded, false) else {
            unreachable!("qwen4_exp MTP: fused TP reduce is not supported")
        };
        self.hc_post(d(&streams), &mlp_out, &inj, batch);
        pool.release_bf16(mlp_out, h * batch);
        pool.release_bf16(normed, h * batch);
        pool.release(inj, hcn * batch);
        streams
    }
}
