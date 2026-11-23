//! `Df2Round` ctx injection: chunk upload, the GEMM / rmsnorm / rope kernel wrappers (`klaunch!`), the
//! incremental inject (optionally graph-captured) and `prime_window`. Ring KV at true absolute positions.
//! Siblings: `draft.rs` consumes the injected ring; `mod.rs` has the map.

use super::*;

impl Df2Round {
    /// Upload `m ≤ 8` tap columns (col-major `[25600, m]`, host bf16) into the staging buffer.
    /// Only the first m columns are written; cols m..8 keep their previous bytes (their GEMM
    /// outputs are garbage columns that nothing downstream reads — column-independent math).
    pub fn upload_chunk(&mut self, cols: &[bf16], m: usize) -> Result<()> {
        // htod_sync_copy_into requires EQUAL lens, so the copy is always the full BLOCK width
        // (row-major [8, 25600]); rows >= m are garbage-but-unread (column-independent math).
        assert!(m <= block() && m >= 1);
        let need = tap_concat_dim() * block();
        // DF2 block-16: a caller may hold only the m columns it actually has (the real-tap
        // fixtures are narrower than a 16-wide block). Only columns < m are read downstream —
        // the math is column-independent — so a short slab is padded rather than refused. A
        // full-width caller (the prime path) keeps the allocation-free copy.
        if cols.len() >= need {
            self.dev.htod_sync_copy_into(&cols[..need], &mut self.staging)
                .context("upload tap chunk")?;
        } else {
            assert!(cols.len() >= tap_concat_dim() * m,
                "upload_chunk needs >= m={m} columns, got {}", cols.len() / tap_concat_dim());
            let mut buf = vec![bf16::default(); need];
            buf[..cols.len()].copy_from_slice(cols);
            self.dev.htod_sync_copy_into(&buf, &mut self.staging)
                .context("upload tap chunk")?;
        }
        Ok(())
    }

    // ---- kernel helpers (S3F's launch shapes) --------------------------------

    /// DF2 block-16: the bf16 block-pass GEMM instance is m8 (8 token columns). At block 16 the
    /// block has 16 columns, so `block()/8` m8 tiles are launched back to back — column-major
    /// slabs, so tile t starts at column 8t of both `x` and `out`. At block 8 this is exactly one
    /// tile with zero offsets, i.e. today's single launch.
    pub(super) fn gemm_dsp(&self, out: &CudaSlice<bf16>, w: &CudaSlice<bf16>, x_ptr: u64, outn: usize, inn: usize) {
        let g = ((outn + 3) / 4) as u32; // R=4
        for t in 0..(block() / 8) {
            let ob = (outn * 8 * t * std::mem::size_of::<bf16>()) as u64;
            let xb = (inn * 8 * t * std::mem::size_of::<bf16>()) as u64;
            klaunch!(self, "gemm_dsp_b_m8_r4", (g, 1, 1), (256, 1, 1), 0,
                (d(out) + ob, d(w), x_ptr + xb, outn as i32, inn as i32));
        }
    }

    /// PLAN/25 §1a: dispatch one block-pass linear on its storage. Nvfp4 runs the TRUNK's
    /// persistent `gemm_mma_fp4_b`, Fp8 its sibling `gemm_mma_fp8_b` (both M=outn rows,
    /// K=inn, N=BLOCK token columns — the N≤16-clamped X reads cover the round's fixed
    /// width; X/out layouts match `gemm_dsp` exactly). The drafter's own numbers gate on
    /// the acceptance quartet, not on binv (a trunk-verify property).
    pub(super) fn gemm_lin(&self, out: &CudaSlice<bf16>, w: &Df2W, x_ptr: u64, outn: usize, inn: usize) {
        match w {
            Df2W::Bf16(w) => self.gemm_dsp(out, w, x_ptr, outn, inn),
            Df2W::Nvfp4(q) => {
                debug_assert_eq!(outn, q.m);
                debug_assert_eq!(inn, q.k);
                let persistent = (crate::gpu::GB10_SMS * 6).min(q.m / 16) as u32;
                klaunch!(self, "gemm_mma_fp4_b", (persistent, 1, 1), (256, 1, 1), 0,
                    (d(out), d(&q.wt), d(&q.st), d(&q.gs), x_ptr,
                     q.m as i32, q.k as i32, block() as i32, 0u64, 0i32));
            }
            Df2W::Fp8(q) => {
                debug_assert_eq!(outn, q.m);
                debug_assert_eq!(inn, q.k);
                // One CTA per 16-row tile (the kernel's own grid choice — the trunk serving
                // path runs it exactly like this); NO PDL arg (unlike gemm_mma_fp4_b).
                klaunch!(self, "gemm_mma_fp8_b", (q.m as u32 / 16, 1, 1), (256, 1, 1), 0,
                    (d(out), d(&q.wt), d(&q.rs), x_ptr,
                     q.m as i32, q.k as i32, block() as i32, 0u64));
            }
        }
    }


    /// The weight `prime_window` runs fc against: the BF16 arm's own tensor, or the baked
    /// artifact's dedicated twin (`gemm_mma_fp4_b` is N≤16; prime is not).
    fn fc_prime(&self) -> &CudaSlice<bf16> {
        match &self.glob.fc {
            Df2W::Bf16(s) => s,
            Df2W::Nvfp4(_) | Df2W::Fp8(_) =>
                self.glob.fc_bf16.as_ref().expect("fc bf16 twin (baked artifact)"),
        }
    }
    /// Same for the prime path's per-layer k/v.
    fn kv_prime(&self, li: usize, which: Kv) -> &CudaSlice<bf16> {
        let l = &self.layers[li];
        let (w, twin) = match which {
            Kv::K => (&l.k_proj, &l.k_proj_bf16),
            Kv::V => (&l.v_proj, &l.v_proj_bf16),
        };
        match w {
            Df2W::Bf16(s) => s,
            Df2W::Nvfp4(_) | Df2W::Fp8(_) => twin.as_ref().expect("k/v bf16 twin (baked artifact)"),
        }
    }

    /// Large-M GEMM (S3F's ctx-side kernel): out [outn, m] = w [outn, k] x x [k, m] col-major bf16.
    fn gemm_tiled(&self, out: &CudaSlice<bf16>, w: &CudaSlice<bf16>, x_ptr: u64, outn: usize, k: usize, m: usize) {
        let mx = ((m + 127) / 128) as u32;
        let nx = ((outn + 127) / 128) as u32;
        klaunch!(self, "gemm_tiled_b", (mx, nx, 1), (16, 16, 1), 0,
            (d(out), d(w), x_ptr, outn as i32, k as i32, m as i32));
    }

    pub(super) fn rmsnorm(&self, out: &CudaSlice<bf16>, x: &CudaSlice<bf16>, w: &CudaSlice<f32>, n: usize, b: usize) {
        klaunch!(self, "rmsnorm_b", (b as u32, 1, 1), (1024, 1, 1), 4096,
            (d(out), d(x), d(w), n as i32, b as i32, fbits(rms_eps())));
    }

    pub(super) fn rmsnorm_perhead(&self, x: &CudaSlice<bf16>, w: &CudaSlice<f32>, heads: usize, b: usize) {
        klaunch!(self, "rmsnorm_perhead_b", ((b * heads) as u32, 1, 1), (head_dim() as u32, 1, 1), (head_dim() * 4) as u32,
            (d(x), d(x), d(w), heads as i32, head_dim() as i32, b as i32, fbits(rms_eps())));
    }

    pub(super) fn rope(&self, x: &CudaSlice<bf16>, cos: &CudaSlice<f32>, sin: &CudaSlice<f32>, heads: usize, b: usize) {
        klaunch!(self, "rope_b", grid(b * heads * (head_dim() / 2)), (256, 1, 1), 0,
            (d(x), d(cos), d(sin), heads as i32, head_dim() as i32, head_dim() as i32, b as i32));
    }

    pub(super) fn gather_rope(&self, out_cos: &CudaSlice<f32>, out_sin: &CudaSlice<f32>, pos: u64, b: usize) {
        klaunch!(self, "gather_rope_b", grid(b * head_dim()), (256, 1, 1), 0,
            (d(out_cos), d(out_sin), d(&self.cos_table), d(&self.sin_table), pos, head_dim() as i32, b as i32));
    }

    pub(super) fn conv2(&self, out: &CudaSlice<bf16>, x: &CudaSlice<bf16>, dyn_all: &CudaSlice<bf16>,
             base_ptr: u64, n: usize, side: usize) {
        let dyn_side = (side * conv_kernel() * conv_groups()) as i32;
        let dyn_stride = (2 * conv_kernel() * conv_groups()) as i32;
        klaunch!(self, "conv2_dynamic_b", grid(n * hidden()), (256, 1, 1), 0,
            (d(out), d(x), d(dyn_all), base_ptr, hidden() as i32, n as i32,
             conv_groups() as i32, conv_group() as i32, dyn_side, dyn_stride));
    }

    // ---- §3.2 incremental ctx injection (M ≤ 8, gemm_dsp path) ---------------

    /// Inject `m ≤ 8` committed tap columns (read from `self.staging` cols `[0, m)` — the
    /// trunk capture's staging or an upload). Positions are ABSOLUTE `[nprev, nprev+m)`:
    /// RoPE at true positions; ring write rows `pos % ring()`. `timer` (optional) brackets the
    /// whole injection (fc + norm + 5×(k/v proj + norm + rope + write)).
    pub fn inject_dev(&mut self, m: usize, mut timer: Option<&mut EvTimer>) -> Result<()> {
        assert!(m <= block(), "inject chunk {m} > block {}", block());
        assert!(self.nprev + m <= self.max_c, "nprev {} + {m} > max_c {}", self.nprev, self.max_c);
        let n0 = self.nprev;
        let ring = ring();

        // rope positions (absolute) + ring write rows for this chunk (padded to BLOCK width:
        // htod_sync_copy_into needs equal lens; rows >= m are garbage-but-unread)
        let mut pos: Vec<i32> = (0..m).map(|j| (n0 + j) as i32).collect();
        let mut wrow: Vec<i32> = (0..m).map(|j| ((n0 + j) % ring) as i32).collect();
        pos.resize(block(), 0);
        wrow.resize(block(), 0);
        self.dev.htod_sync_copy_into(&pos, &mut self.pos_c)?;
        self.dev.htod_sync_copy_into(&wrow, &mut self.wrow_c)?;
        if let Some(t) = timer.as_deref_mut() { t.mark(self.stream.stream); }

        // E0: replay the captured injection graph when this width has one (the sequence is
        // width-contingent, so the map is keyed by m). Same kernels, same order, same buffers —
        // the probe's graph-vs-eager determinism assert covers the pair. The eager call at a
        // fresh m below doubles as the warmup; the capture that follows only RECORDS.
        if self.inject_graphs.contains_key(&m) {
            if let Some(g) = self.inject_graphs.get(&m) { g.launch(); }
            if let Some(t) = timer.as_deref_mut() { t.mark(self.stream.stream); }
            self.nprev = n0 + m;
            return Ok(());
        }
        self.inject_kernels(m);
        if crate::opts::var(crate::opt!("no-df2-inject-graph")).is_err() {
            let _ = self.capture_inject_graph(m);
        }
        if let Some(t) = timer.as_deref_mut() { t.mark(self.stream.stream); }
        self.nprev = n0 + m;
        Ok(())
    }

    /// The pure-launch body of `inject_dev` (fc + hidden_norm at M=m, then per-layer
    /// k/v projections + ring writes). Eager execution and CUDA-graph capture call the SAME
    /// function — the capture records exactly the eager sequence. No allocs, no syncs, no
    /// all-reduce (kv projections are rank-local) — capture-safe by construction.
    fn inject_kernels(&self, m: usize) {
        // fc + hidden_norm at M=m (gemm_dsp; cols ≥ m are garbage-but-unread)
        self.gemm_lin(&self.th_raw, &self.glob.fc, d(&self.staging), hidden(), tap_concat_dim());
        self.rmsnorm(&self.th, &self.th_raw, &self.glob.hidden_norm, hidden(), m);

        // per-layer k/v: k = rope(k_norm(k_proj(th))) @ ring rows; v raw
        // P2: the projections run at the rank-local kv-head width (heads are independent —
        // no all-reduce; each rank writes only ITS kv-heads' ring rows).
        let nkv = self.nkv_l;
        self.gather_rope(&self.cos_c, &self.sin_c, d(&self.pos_c), m);
        for li in 0..n_layers() {
            let l = &self.layers[li];
            self.gemm_lin(&self.kc, &l.k_proj, d(&self.th), nkv * head_dim(), hidden());
            self.rmsnorm_perhead(&self.kc, &l.k_norm, nkv, m);
            self.rope(&self.kc, &self.cos_c, &self.sin_c, nkv, m);
            self.gemm_lin(&self.vc, &l.v_proj, d(&self.th), nkv * head_dim(), hidden());
            klaunch!(self, "write_kv_b", grid(m * nkv * head_dim()), (256, 1, 1), 0,
                (d(&self.k_ring[li]), d(&self.v_ring[li]), d(&self.kc), d(&self.vc),
                 d(&self.wrow_c), ring_stride() as i32, nkv as i32, head_dim() as i32,
                 m as i32, d(&self.slot_c)));
        }
    }

    /// E0: capture `inject_kernels(m)` into a per-width CUDA graph (the `capture_round_graph`
    /// pattern). The DF2 stream is quiesced first (the eager inject just enqueued); the capture
    /// pass records only. Failure is non-fatal: the eager path stays.
    fn capture_inject_graph(&mut self, m: usize) -> bool {
        use cudarc::driver::sys;
        if self.dev.synchronize().is_err() { return false; }
        let stream = self.stream.stream;
        let r = unsafe { sys::cuStreamBeginCapture_v2(stream, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) };
        if r != sys::CUresult::CUDA_SUCCESS {
            crate::metrics::graph_capture_lost("df2-inject", &format!("width {m}"), &format!("BeginCapture {r:?}")); // S-B9-REL-INTEG C1
            return false;
        }
        self.inject_kernels(m);
        let mut graph: sys::CUgraph = std::ptr::null_mut();
        let r = unsafe { sys::cuStreamEndCapture(stream, &mut graph) };
        if r != sys::CUresult::CUDA_SUCCESS {
            crate::metrics::graph_capture_lost("df2-inject", &format!("width {m}"), &format!("{r:?}")); // S-B9-REL-VISIONFIX2 V2b
            return false;
        }
        let mut exec: sys::CUgraphExec = std::ptr::null_mut();
        let r = unsafe { sys::cuGraphInstantiate_v2(&mut exec, graph, std::ptr::null_mut(), std::ptr::null_mut(), 0) };
        unsafe { sys::cuGraphDestroy(graph); }
        if r != sys::CUresult::CUDA_SUCCESS {
            crate::metrics::graph_capture_lost("df2-inject", &format!("width {m}"), &format!("instantiate {r:?}")); // S-B9-REL-INTEG C1
            return false;
        }
        self.inject_graphs.insert(m, crate::gpu::CudaGraph::from_exec(exec, stream));
        true
    }

    /// S5F — engine prompt-prime: inject `n` tap columns at ONE large M (the whole prefill
    /// window) via the S3F tiled GEMM, instead of n/8 skinny-M chunks. `taps` is the prefill
    /// capture buffer `[TAP_CONCAT_DIM, n]` col-major bf16 (the same per-layer-row layout as the
    /// sink's staging), cols [0, n) = the window's taps. Positions are ABSOLUTE
    /// `[pos_start, pos_start+n)`; ring write rows `pos % ring()`. Sets `nprev = pos_start + n`.
    /// Bit-identical k/v to the probe's chunked path (same kernels, same reduction orders — the
    /// per-chunk probe asserts th/k bitwise; M only changes the GEMM tile count, not the math per
    /// element). Runs on the round's stream; the caller must have synced the capture D2Ds.
    pub fn prime_window(&mut self, taps: &CudaSlice<bf16>, n: usize, pos_start: usize) -> Result<()> {
        assert!(n >= 1 && n <= 8192, "prime window {n} out of range (1..8192)");
        assert!(pos_start + n <= self.max_c, "prime {}..{} > max_c {}",
                pos_start, pos_start + n - 1, self.max_c);
        let ring = ring();

        let pos: Vec<i32> = (0..n).map(|j| (pos_start + j) as i32).collect();
        let wrow: Vec<i32> = (0..n).map(|j| ((pos_start + j) % ring) as i32).collect();
        // slot ids are ALL ZERO (single-slot ring) but `write_kv_b` indexes slot_ids[b] for
        // b in 0..B — a buffer sized for BLOCK=8 would be an OOB READ at n > 8 (the first real
        // S5F bug this session caught by the lossless gate's crash).
        let slots: Vec<i32> = vec![0i32; n];
        let pos_dev = self.dev.htod_sync_copy(&pos).context("prime pos")?;
        let wrow_dev = self.dev.htod_sync_copy(&wrow).context("prime wrow")?;
        let slot_dev = self.dev.htod_sync_copy(&slots).context("prime slots")?;
        let cos_c = self.dev.alloc_zeros::<f32>(n * head_dim()).context("prime cos")?;
        let sin_c = self.dev.alloc_zeros::<f32>(n * head_dim()).context("prime sin")?;
        self.gather_rope(&cos_c, &sin_c, d(&pos_dev), n);

        // fc + hidden_norm at M=n (one weight read for the whole window).
        let th_raw = self.dev.alloc_zeros::<bf16>(hidden() * n).context("prime th_raw")?;
        let th = self.dev.alloc_zeros::<bf16>(hidden() * n).context("prime th")?;
        self.gemm_tiled(&th_raw, self.fc_prime(), d(taps), hidden(), tap_concat_dim(), n);
        self.rmsnorm(&th, &th_raw, &self.glob.hidden_norm, hidden(), n);

        // per-layer k/v at M=n, written at ring rows (pos_start+j) % ring().
        // P2: rank-local kv-head width (same as inject_dev).
        let nkv = self.nkv_l;
        let kc = self.dev.alloc_zeros::<bf16>(nkv * head_dim() * n).context("prime k")?;
        let vc = self.dev.alloc_zeros::<bf16>(nkv * head_dim() * n).context("prime v")?;
        for li in 0..n_layers() {
            let l = &self.layers[li];
            self.gemm_tiled(&kc, self.kv_prime(li, Kv::K), d(&th), nkv * head_dim(), hidden(), n);
            self.rmsnorm_perhead(&kc, &l.k_norm, nkv, n);
            self.rope(&kc, &cos_c, &sin_c, nkv, n);
            self.gemm_tiled(&vc, self.kv_prime(li, Kv::V), d(&th), nkv * head_dim(), hidden(), n);
            klaunch!(self, "write_kv_b", grid(n * nkv * head_dim()), (256, 1, 1), 0,
                (d(&self.k_ring[li]), d(&self.v_ring[li]), d(&kc), d(&vc),
                 d(&wrow_dev), ring_stride() as i32, nkv as i32, head_dim() as i32,
                 n as i32, d(&slot_dev)));
        }
        self.dsync()?;
        self.nprev = pos_start + n;
        Ok(())
    }

    // ---- §3.3/§3.4 the block pass + borrowed head + selector -----------------

}
