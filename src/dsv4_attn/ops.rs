//! Building-block ops of `Dsv4AttnRuntime`: g128 activation quant, FP8 block-scaled GEMMs, TP all-reduce, rmsnorm,
//! hc_pre / hc_post / hc_head and the LM head. Batch-invariant per-row kernels: keep their launch order.
//! Siblings: `attn.rs`, `moe.rs`, `dspark.rs` compose these; `mod.rs` has the map.

use super::*;

impl Dsv4AttnRuntime {
    /// §C.1 FP8 act-quant (non-inplace): bf16 x [rows, n] → codes [rows, n] + UE8M0 [rows, n/128].
    pub fn quant_g128<X: dsv4_gpu::Dsv4Buf<bf16>, C: dsv4_gpu::Dsv4Buf<u8>>(&self, x: &X, rows: usize, n: usize) -> Result<(C, C)> {
        let mut y = C::alloc_zeros(&self.dev, self.stream.stream, rows * n)?;
        let mut s = C::alloc_zeros(&self.dev, self.stream.stream, rows * (n / 128))?;
        self.quant_g128_into(&mut y, &mut s, x, rows, n)?;
        Ok((y, s))
    }

    /// R3.1: `quant_g128` into caller buffers (the MoE workspace — zero per-call allocs).
    pub fn quant_g128_into<X: dsv4_gpu::Dsv4Buf<bf16>, C: dsv4_gpu::Dsv4Buf<u8>>(&self, y: &mut C, s: &mut C, x: &X, rows: usize, n: usize) -> Result<()> {
        let blocks = ceil256(rows * (n / 128) * 32);
        let (r_i, n_i) = (rows as i32, n as i32);
        dsv4_launch!(self.spine, "dsv4_act_quant_g128", self.stream.stream, (blocks, 1, 1), (256, 1, 1), 0,
            (x, &mut *y, &mut *s, &r_i, &n_i))?;
        Ok(())
    }

    /// §C.3 FP8 block-scale GEMM at activation width s (chunked ≤16 — the G2-proven
    /// N-invariant regime; chunking is bitwise-identical to any other ≤16 decomposition).
    /// `codes`/`sa` are the full [s, k] / [s, k/128] activation quant. Output [s, m] bf16.
    pub fn fp8_bsb_rows<X: dsv4_gpu::Dsv4Buf<bf16>, C: dsv4_gpu::Dsv4Buf<u8>>(&self, w: &Fp8Weight, codes: &C, sa: &C, s: usize) -> Result<X> {
        let (m, _k) = (w.m, w.k);
        let mut c = X::alloc_zeros(&self.dev, self.stream.stream, s * m)?;
        self.fp8_bsb_rows_into(&mut c, w, codes, sa, s)?;
        Ok(c)
    }

    /// R3A.1 E2 (first rung): ONE fused launch for two independent fp8 projections sharing
    /// the activation (production: wq_a + wkv per layer). Tier-1 item 1.4 (RUN 16): the
    /// launch is `gemm_dsv4_fp8_bsb1q` — ONE 16-row tile per CTA (96 CTAs at the decode
    /// shapes) instead of the two-tile bsb2q (48 CTAs, ramp-bound at ~1 CTA/SM): 28.4 µs vs
    /// 42.5 µs isolated (221.7 vs 148.2 GB/s). Per-element chains identical to the two
    /// separate bsb launches — gated bitwise at the production shapes, N ∈ {1,6,16}.
    /// Only for s ≤ 16 (decode/verify widths; prefill keeps the pf path per projection).
    pub fn fp8_bsb2q_rows<X: dsv4_gpu::Dsv4Buf<bf16>, C: dsv4_gpu::Dsv4Buf<u8>>(&self, w0: &Fp8Weight, w1: &Fp8Weight, codes: &C, sa: &C, s: usize) -> Result<(X, X)> {
        anyhow::ensure!(s >= 1 && s <= 16, "fp8_bsb2q is a ≤16-row launch (s={s})");
        let (m0, m1, k) = (w0.m, w1.m, w0.k);
        anyhow::ensure!(w1.k == k && m0 % 128 == 0 && m1 % 128 == 0 && k % 128 == 0,
            "fp8_bsb2q geometry {m0}x{m1}x{k}");
        let c0 = X::alloc_zeros(&self.dev, self.stream.stream, s * m0)?;
        let c1 = X::alloc_zeros(&self.dev, self.stream.stream, s * m1)?;
        let f = &self.bk["gemm_dsv4_fp8_bsb1q"];
        let cfg = LaunchConfig {
            grid_dim: ((m0 / 16 + m1 / 16) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let m01: u64 = (m0 as u64) | ((m1 as u64) << 32);
        let kn: u64 = (k as u64) | ((s as u64) << 32);
        let (c0v, c1v) = (c0.view(0, s * m0), c1.view(0, s * m1));
        let (cv, sav) = (codes.view(0, s * k), sa.view(0, s * (k / 128)));
        unsafe {
            f.clone()
                .launch_on_stream(&self.stream, cfg,
                    (&c0v, &w0.wt, &w0.sb, &c1v, &w1.wt, &w1.sb, &cv, &sav, m01, kn, 0u64))
                .map_err(|e| anyhow!("gemm_dsv4_fp8_bsb1q s={s}: {e:?}"))?;
        }
        Ok((c0, c1))
    }

    /// R3.1: `fp8_bsb_rows` into a caller buffer (the MoE workspace — zero per-call allocs).
    /// R3A.1: launches the two-tiles-per-CTA `gemm_dsv4_fp8_bsb2` (identical per-element
    /// chains, two independent weight streams per warp — E1b; grid halves to (m+31)/32).
    /// R3A.4 (P1): at s>16 ONE weight-stationary launch (chunk loop inside; each weight
    /// tile read once per launch, L2-hot across chunks) instead of ceil(s/16) full-weight
    /// re-read launches. Tier-2 2.2 (session 6): the launch is `gemm_dsv4_fp8_bsb_pf2` at
    /// K<=4096 — TWO 16-row weight tiles per CTA sharing each chunk's X fragments (the
    /// bsb2 decode trick at width), with a grid.y fill floor — and stays
    /// `gemm_dsv4_fp8_bsb_pf` at K>4096 (wo_b: pf2's dual weight stream measures 4-6%
    /// slower there). Per-element chains identical to the <=16-row decomposition — gated
    /// bitwise at s in {17, 63, 64, 65, 130, 2048}.
    pub fn fp8_bsb_rows_into<OB: dsv4_gpu::Dsv4Buf<bf16>, C: dsv4_gpu::Dsv4Buf<u8>>(&self, c: &mut OB, w: &Fp8Weight, codes: &C, sa: &C, s: usize) -> Result<()> {
        let (m, k) = (w.m, w.k);
        anyhow::ensure!(m % 128 == 0 && k % 128 == 0, "fp8_bsb geometry {m}x{k}");
        anyhow::ensure!(c.len() >= s * m, "fp8_bsb_rows_into: out {} < {s}*{m}", c.len());
        if s > 16 {
            // grid.y = chunk groups: ~48k CTAs of parallelism without re-serializing
            // chunks (weight tiles re-read <= gridDim.y times, L2-hot within).
            let nchunks = s.div_ceil(16);
            let gy8 = nchunks.div_ceil(8).max(1); // ~8 chunks per CTA (pf grouping)
            if k > 4096 {
                // wo_b-class (long K): pf2's dual weight stream per CTA measures 4-6%
                // SLOWER at K=8192 (session-6 bench) — keep the single-tile pf here.
                let f = &self.bk["gemm_dsv4_fp8_bsb_pf"];
                let cfg = LaunchConfig {
                    grid_dim: ((m / 16) as u32, gy8 as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let cv = c.view(0, s * m);
                let (xv, sav) = (codes.view(0, s * k), sa.view(0, s * (k / 128)));
                unsafe {
                    f.clone()
                        .launch_on_stream(&self.stream, cfg, (&cv, &w.wt, &w.sb, &xv, &sav, m as i32, k as i32, s as i32, 0u64))
                        .map_err(|e| anyhow!("gemm_dsv4_fp8_bsb_pf s={s}: {e:?}"))?;
                }
                return Ok(());
            }
            // pf2 (two 16-row tiles per CTA sharing each chunk's X fragments; session-6
            // bench 1.10-1.30x on every K<=4096 projection). grid.x = tile pairs; the
            // fill floor (2 waves of the 3 CTA/SM residency) keeps small-M shapes from
            // underfilling the machine (wkv [512,4096] @512: 0.88x unfilled -> 1.30x).
            let gx = (m + 31) / 32;
            let gy = gy8.max((288usize.div_ceil(gx)).min(nchunks)) as u32;
            let f = &self.bk["gemm_dsv4_fp8_bsb_pf2"];
            let cfg = LaunchConfig {
                grid_dim: (gx as u32, gy, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let cv = c.view(0, s * m);
            let (xv, sav) = (codes.view(0, s * k), sa.view(0, s * (k / 128)));
            unsafe {
                f.clone()
                    .launch_on_stream(&self.stream, cfg, (&cv, &w.wt, &w.sb, &xv, &sav, m as i32, k as i32, s as i32, 0u64))
                    .map_err(|e| anyhow!("gemm_dsv4_fp8_bsb_pf2 s={s}: {e:?}"))?;
            }
            return Ok(());
        }
        let f = &self.bk["gemm_dsv4_fp8_bsb2"];
        let mut r0 = 0usize;
        while r0 < s {
            let n = (s - r0).min(16);
            let xv = codes.view(r0 * k, n * k);
            let sav = sa.view(r0 * (k / 128), n * (k / 128));
            let cv = c.view(r0 * m, n * m);
            let cfg = LaunchConfig {
                grid_dim: (((m + 31) / 32) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                f.clone()
                    .launch_on_stream(&self.stream, cfg, (&cv, &w.wt, &w.sb, &xv, &sav, m as i32, k as i32, n as i32, 0u64))
                    .map_err(|e| anyhow!("gemm_dsv4_fp8_bsb2 chunk r0={r0} n={n}: {e:?}"))?;
            }
            r0 += n;
        }
        Ok(())
    }

    /// TP=2 sum-all-reduce of a bf16 device buffer across the two boxes — the doorbell two-kernel
    /// handshake (`tp_gate_copy_signal` → `tp_wait_add`), launched on THIS runtime's blocking compute
    /// stream. Mirrors `gpu.rs::tp_all_reduce_bf16` (3183) but on the DSV4 runtime's stream/bk.
    /// SPIN-WAIT comm (no main-thread sync); the persistent RDMA proxy (spawned at `attach_tp`)
    /// watches the device-side epoch and ships the payload. Chunked to the ring slot. No-op when
    /// `tp_ctx_dptr == 0` (single-process). `buf` is the rank-local PARTIAL; on return it holds the
    /// full Σ (bf16: ship bf16, sum in fp32 in K2, one bf16 round — the standard TP-partials class).
    pub fn tp_all_reduce_bf16(&self, buf: &mut B, n: usize) -> Result<()> {
        if self.tp_ctx_dptr == 0 || n == 0 {
            return Ok(());
        }
        let buf_ptr = *buf.device_ptr();
        let chunk = ((crate::tp::TP_SLOT_BYTES - 64) / 2) & !7; // bf16 elems per ring slot
        let mut off = 0usize;
        while off < n {
            let c = (n - off).min(chunk);
            let p = buf_ptr + (off * 2) as u64;
            let nbytes = (c * 2) as u32;
            let c_i = c as i32;
            {
                let f = &self.bk["tp_gate_copy_signal"];
                let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (512, 1, 1), shared_mem_bytes: 0 };
                unsafe {
                    f.clone().launch_on_stream(&self.stream, cfg, (self.tp_ctx_dptr, p, nbytes))
                        .map_err(|e| anyhow!("tp_gate_copy_signal: {e:?}"))?;
                }
            }
            {
                let f = &self.bk["tp_wait_add"];
                let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (512, 1, 1), shared_mem_bytes: 0 };
                unsafe {
                    f.clone().launch_on_stream(&self.stream, cfg, (self.tp_ctx_dptr, p, p, c_i, 0i32))
                        .map_err(|e| anyhow!("tp_wait_add: {e:?}"))?;
                }
            }
            off += c;
        }
        Ok(())
    }

    /// fp32 RMSNorm (bf16 in/out): y = rmsnorm(x, w) row-wise, dim columns.
    pub fn rmsnorm<X: dsv4_gpu::Dsv4Buf<half::bf16>>(&self, x: &X, w: &S, rows: usize, dim: usize, eps: f32) -> Result<X> {
        let y = X::alloc_zeros(&self.dev, self.stream.stream, rows * dim)?;
        let (r_i, d_i) = (rows as i32, dim as i32);
        dsv4_launch!(self.spine, "dsv4_rmsnorm_b", self.stream.stream, (rows as u32, 1, 1), (256, 1, 1), 0,
            (&y, x, w, &r_i, &d_i, &eps))?;
        Ok(y)
    }

    /// mHC hc_pre (§B.8): returns (y [s,dim] bf16, post [s,4] f32, comb [s,16] f32).
    ///
    /// T6: at PREFILL width (s ≥ 256) ONE fused launch (`dsv4_hc_pre_fused_b`) replaces the
    /// 4-launch chain (rsqrt + mixes + sinkhorn + collapse) — bitwise-identical per element
    /// (same per-thread fmaf chains + stride-halving trees), 3 tokens/768-thread block so the
    /// hc_fn rows are read once per 3 tokens instead of once per (m, t) block. At small s
    /// (decode s=1, verify s=6) the chain stays: the 24 dots need 24 blocks of parallelism
    /// there, and a single fused block serializes them (the measured negative precedent from
    /// the qwen era — parallelism-bound at decode, byte-redundant at width — matches).
    pub fn hc_pre<X: dsv4_gpu::Dsv4Buf<bf16>, F: dsv4_gpu::Dsv4Buf<f32>>(&self, x: &X, s: usize, hc_fn: &S, hc_base: &S, hc_scale: &S, cfg: &dsv4_load::Dsv4Config) -> Result<(X, F, F)> {
        let (hc, dim) = (cfg.hc_mult, cfg.dim);
        let hcdim = hc * dim;
        let rsqrt = F::alloc_zeros(&self.dev, self.stream.stream, s)?;
        let mixes = F::alloc_zeros(&self.dev, self.stream.stream, s * 24)?;
        let pre = F::alloc_zeros(&self.dev, self.stream.stream, s * hc)?;
        let post = F::alloc_zeros(&self.dev, self.stream.stream, s * hc)?;
        let comb = F::alloc_zeros(&self.dev, self.stream.stream, s * hc * hc)?;
        let y = X::alloc_zeros(&self.dev, self.stream.stream, s * dim)?;
        let (s_i, hcd_i, dim_i, hc_i) = (s as i32, hcdim as i32, dim as i32, hc as i32);
        let eps = cfg.norm_eps;
        if s >= 256 {
            let grid = ((s + 2) / 3) as u32;
            dsv4_launch!(self.spine, "dsv4_hc_pre_fused_b", self.stream.stream, (grid, 1, 1), (768, 1, 1), HC_PRE_FUSED_SMEM,
                (&rsqrt, &mixes, &pre, &post, &comb, &y, x, hc_fn, hc_scale, hc_base, &s_i, &hcd_i, &dim_i, &hc_i, &eps))?;
        } else {
            dsv4_launch!(self.spine, "dsv4_hc_pre_rsqrt_b", self.stream.stream, (s as u32, 1, 1), (256, 1, 1), 0,
                (&rsqrt, x, &s_i, &hcd_i, &eps))?;
            dsv4_launch!(self.spine, "dsv4_hc_mixes_b", self.stream.stream, (24u32, s as u32, 1), (256, 1, 1), 0,
                (&mixes, hc_fn, x, &rsqrt, &s_i, &hcd_i))?;
            dsv4_launch!(self.spine, "dsv4_hc_split_sinkhorn", self.stream.stream, (ceil256(s), 1, 1), (256, 1, 1), 0,
                (&mixes, hc_scale, hc_base, &pre, &post, &comb, &s_i))?;
            dsv4_launch!(self.spine, "dsv4_hc_collapse_b", self.stream.stream, (s as u32, 1, 1), (256, 1, 1), 0,
                (&y, x, &pre, &s_i, &dim_i, &hc_i))?;
        }
        Ok((y, post, comb))
    }

    /// mHC hc_post (§B.8): out [s, hc*dim] = post[k]·sub_out + Σ_j comb[j,k]·resid[j].
    pub fn hc_post<X: dsv4_gpu::Dsv4Buf<bf16>, F: dsv4_gpu::Dsv4Buf<f32>>(&self, sub_out: &X, resid: &X, post: &F, comb: &F, s: usize, cfg: &dsv4_load::Dsv4Config) -> Result<X> {
        let (hc, dim) = (cfg.hc_mult, cfg.dim);
        let out = X::alloc_zeros(&self.dev, self.stream.stream, s * hc * dim)?;
        let (s_i, dim_i, hc_i) = (s as i32, dim as i32, hc as i32);
        dsv4_launch!(self.spine, "dsv4_hc_post_b", self.stream.stream, (s as u32, 1, 1), (256, 1, 1), 0,
            (&out, sub_out, resid, post, comb, &s_i, &dim_i, &hc_i))?;
        Ok(out)
    }

    /// `hc_head` (§B.8, model.py:709-716): final trunk collapse, sigmoid-only (no post/comb/
    /// Sinkhorn). `x` [s, hc*dim] bf16 streams → `collapsed` [s, dim] bf16. One block/token;
    /// reduction trees match hc_pre (tolerance-level vs the CPU pairwise tree).
    pub fn hc_head<X: dsv4_gpu::Dsv4Buf<bf16>>(
        &self,
        x: &X,
        hc_fn: &S,
        hc_base: &S,
        hc_scale: &S,
        s: usize,
        cfg: &dsv4_load::Dsv4Config,
    ) -> Result<X> {
        let (hc, dim) = (cfg.hc_mult, cfg.dim);
        let y = X::alloc_zeros(&self.dev, self.stream.stream, s * dim)?;
        let (s_i, dim_i, hc_i) = (s as i32, dim as i32, hc as i32);
        dsv4_launch!(self.spine, "dsv4_hc_head_b", self.stream.stream, (s as u32, 1, 1), (256, 1, 1), 0,
            (&y, x, hc_fn, hc_base, hc_scale, &s_i, &dim_i, &hc_i, &cfg.norm_eps, &cfg.hc_eps))?;
        Ok(y)
    }

    /// LM head (§A.1): bf16 `head` [vocab, dim] (fp32-valued) @ `x` [dim, n] bf16 → fp32 logits
    /// [vocab, n] via `gemm_binv_f32_b` (the hy_v3 enable_lm_head_fp32 twin — batch-invariant
    /// strided-k tree, f32 epilogue, no bf16 round). n ≤ 8 uses the templated fast path.
    pub fn lm_head<X: dsv4_gpu::Dsv4Buf<bf16>, F: dsv4_gpu::Dsv4Buf<f32>>(&self, head: &B, x: &X, n: usize, dim: usize, vocab: usize) -> Result<F> {
        let logits = F::alloc_zeros(&self.dev, self.stream.stream, vocab * n)?;
        let f = &self.bk["gemm_binv_f32_b"];
        // gemm_binv_impl<N> uses `extern __shared__ float sh[N][256]` for the tree reduce —
        // size the smem to N·256 floats (the trunk's n=1 path uses 256·4 = 1024 B; multi-row
        // DSpark verify / markov need the full N·256·4).
        let cfg = LaunchConfig {
            grid_dim: (vocab as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: (n * 256 * 4) as u32,
        };
        let logits_v = logits.view(0, vocab * n);
        let xv = x.view(0, n * dim);
        unsafe {
            f.clone()
                .launch_on_stream(&self.stream, cfg, (&logits_v, head, &xv, vocab as i32, dim as i32, n as i32))
                .map_err(|e| anyhow!("gemm_binv_f32_b lm_head: {e:?}"))?;
        }
        Ok(logits)
    }

    /// R3.2 (L2): vocab-parallel LM head + the 8 B maxloc exchange under TP=2. This rank computes
    /// ONLY its vocab half's fp32 logits (the `gemm_binv_f32_b` kernel is row-independent — the
    /// half rows are per-element bitwise-identical to the unsharded head), runs the single-block
    /// argmax (val desc, idx asc — the same total order as the host `dsv4_argmax`), ships the 8 B
    /// (val, global idx) on the doorbell (K1) and combines with the peer's (K2 maxloc, same total
    /// order). The winner is bitwise the unsharded full-vocab argmax — SPMD lockstep holds with
    /// no token broadcast. Requires the TP doorbell attached. Greedy decode only — sampler/
    /// confidence-head paths keep the full-logits [`lm_head`](Self::lm_head).
    pub fn lm_head_maxloc_tp(&self, head: &B, x: &B, dim: usize, vocab: usize, rank: usize) -> Result<u32> {
        anyhow::ensure!(self.tp_ctx_dptr != 0, "lm_head_maxloc_tp: TP not attached");
        anyhow::ensure!(vocab % 2 == 0, "lm_head_maxloc_tp: odd vocab {vocab}");
        let rows = vocab / 2;
        let row0 = rank * rows;
        let mut half = self.dev.alloc_zeros::<f32>(rows)?;
        let hv = head.slice(row0 * dim..(row0 + rows) * dim);
        {
            let f = &self.bk["gemm_binv_f32_b"];
            let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: (256 * 4) as u32 };
            unsafe {
                f.clone()
                    .launch_on_stream(&self.stream, cfg, (&mut half, &hv, x, rows as i32, dim as i32, 1i32))
                    .map_err(|e| anyhow!("gemm_binv_f32_b lm_head_rows: {e:?}"))?;
            }
        }
        let mut pair = self.dev.alloc_zeros::<u32>(2)?;
        {
            let f = &self.bk["dsv4_argmax_pair_b"];
            let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
            unsafe {
                f.clone()
                    .launch_on_stream(&self.stream, cfg, (&half, rows as i32, &mut pair, row0 as i32))
                    .map_err(|e| anyhow!("dsv4_argmax_pair_b: {e:?}"))?;
            }
        }
        let mut out_idx = self.dev.alloc_zeros::<i32>(1)?;
        let pair_ptr = *pair.device_ptr();
        let out_ptr = *out_idx.device_ptr();
        {
            let f = &self.bk["tp_gate_copy_signal"];
            let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (512, 1, 1), shared_mem_bytes: 0 };
            unsafe {
                f.clone()
                    .launch_on_stream(&self.stream, cfg, (self.tp_ctx_dptr, pair_ptr, 8u32))
                    .map_err(|e| anyhow!("tp_gate_copy_signal maxloc: {e:?}"))?;
            }
        }
        {
            // v2 receive mode (EXPERT_GPU_ALLREDUCE §3.2): the maxloc barrier inherits the
            // GPU-direct two-stage gate automatically.
            let k2 = if crate::gpu::GpuModel::tp_gpu_recv_on_pub() { "tp_wait_maxloc_g" } else { "tp_wait_maxloc_b" };
            let f = &self.bk[k2];
            let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (512, 1, 1), shared_mem_bytes: 0 };
            unsafe {
                f.clone()
                    .launch_on_stream(&self.stream, cfg, (self.tp_ctx_dptr, pair_ptr, out_ptr))
                    .map_err(|e| anyhow!("tp_wait_maxloc: {e:?}"))?;
            }
        }
        let v: Vec<i32> = self.dev.dtoh_sync_copy(&out_idx)?;
        Ok(v[0] as u32)
    }

}
