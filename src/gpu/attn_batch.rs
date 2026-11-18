//! `impl GpuModel`: batched full attention: tiled prefill attention, `attn_dispatch` (decode / verify / prefill kernels) and
//! `full_attn_batch`. Map: `mod.rs`.

use super::*;

impl GpuModel {
    /// Tiled prefill attention: `S = QᵀK` and `O += P·Vᵀ` through cuBLAS tensor-core GEMMs, with an
    /// online-softmax kernel between them.
    ///
    /// The scalar `gqa_attn_prefill` is bound by instruction issue, not memory — query tiling cut its
    /// L1/L2 traffic 8× for 11%, and 4 independent loads in flight per warp bought 0%. A *perfect*
    /// scalar kernel bottoms out near 25 ms/layer; a *mediocre* tensor-core one starts at ~9. So the
    /// fix is which hardware unit runs the two GEMMs, not how the scalar loop is tuned.
    ///
    /// GQA: a group's 4 query heads share one K/V, and cuBLAS cannot express "stride 0 for four, then
    /// jump" — so we issue one strided-batched GEMM per index-within-group `r`, batched over kv heads.
    ///
    /// Prefill sits OUTSIDE the batch-invariance contract (it feeds decode and verify identically), so
    /// tiling and reordering are free here. Still deterministic: fixed tiles, fixed order.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attn_prefill_tiled(&self, pool: &mut Pool, q: &B, kc_ptr: u64, vc_ptr: u64,
                          stride: usize, nh: usize, nkv: usize, n: usize, pos_start: usize, attn: &mut B) {
        use cudarc::cublas::sys::{cudaDataType, cublasComputeType_t, cublasGemmAlgo_t};
        let cfg = &self.cfg;
        // Head counts come from the CALLER, never cfg: under TP mixer sharding the q/k/v tensors and
        // the KV cache hold nh/2 and nkv/2 (local) heads — reading cfg's full counts here over-reads
        // every input (masked by pool slack at small n, an ILLEGAL ADDRESS at large n) and writes
        // `attn` with the wrong layout. Same root cause as the attn_dispatch half-heads bug:
        // sharding is a property of the tensors passed in, never of the process.
        let hd = cfg.head_dim;
        let gqa = nh / nkv;
        let pc = pos_start + n;                       // total keys any query here may attend to
        let scale = 1.0f32 / (hd as f32).sqrt();

        // Tile sizes. Note this function is ALSO the verify path (`verify_forward` calls it with
        // n = depth, not a prompt length), so it must stay bit-identical across n — which it is: the
        // per-query accumulation order does not depend on br. Bounding the Pool is `bucket_up`'s job,
        // not this line's.
        let br = PF_BR.min(n);
        let bc = PF_BC;

        // Fused flash-attention prefill (ONE kernel replaces the whole QK^T/softmax/PV tile walk —
        // no S/P global round-trip, no per-tile init/finalize launches). Same layouts and the same
        // online-softmax numerics as the tiled path (P rounded to bf16 before PV, exactly like the
        // tiled path's bf16 p_buf).
        // Measured on .13 (fa_tc harness): 15.3/44.7/74.0/103.5 ms at N=8192, pc=8K/16K/24K/32K
        // = 4.0x under the tiled path's per-call bar at every geometry (~54 TF/s effective).
        // 2026-09-06 DEFAULT ON (F8 catch-up: the vLLM recipe's FA2 prefill measures 11.6 ms/call
        // vs our tiled path ~220 ms/mixer at N=10631 hd=256 — see PLAN/F8_CATCHUP_ANALYSIS.md
        // §10.4; with this + GDN chunk2 the 10631-tok cell runs 11768→8123 ms, binv PASS, state
        // EXACT, LOSSLESS_OK). The tiled path below stays the ONLY path verify can ever take:
        // verify calls this function with n = depth <= 16 < PF_MIN, and this branch requires
        // n >= PF_MIN — the lever is prefill-only and decode is byte-identical by construction.
        // `--fa-prefill=0` restores the tiled path (VALUE check, not existence — the idiom
        // that burned a benchmark session).
        if n >= PF_MIN
            && crate::opts::var(crate::opt!("fa-prefill")).map_or(true, |v| {
                !matches!(v.as_str(), "" | "0" | "false" | "off")
            })
        {
            // Tensor-core body covers hd in {128,256} and gqa dividing 96 (g=6 -> 16 tokens/block)
            if (hd == 256 || hd == 128) && 96 % gqa == 0 {
                if let Some(f) = fa_tc_raw_fn() {
                    let br_t = 96 / gqa;
                    let gx = ((n + br_t - 1) / br_t) as u32;
                    let smem = (4 * 32 * hd * 2) as u32;      // K[2]+V[2] tiles
                    // Named locals: the driver marshals params during the call (E9 convention).
                    let (qp, kp, vp, op) = (*q.device_ptr() as u64, kc_ptr, vc_ptr, d(attn));
                    let (nn_, nh_, nkv_, hd_) = (n as i32, nh as i32, nkv as i32, hd as i32);
                    let (st_, ps_, g_) = (stride as i32, pos_start as i32, gqa as i32);
                    let mut params = [qp.as_kernel_param(), kp.as_kernel_param(), vp.as_kernel_param(),
                                      op.as_kernel_param(), nn_.as_kernel_param(), nh_.as_kernel_param(),
                                      nkv_.as_kernel_param(), hd_.as_kernel_param(), st_.as_kernel_param(),
                                      ps_.as_kernel_param(), g_.as_kernel_param()];
                    unsafe { self.launch_plain_raw(f, (gx, nkv as u32, 1), (192, 1, 1), smem, &mut params); }
                    return;
                }
            }
            // Scalar oracle fallback (slow; correctness twin of the tensor-core body)
            let gx = ((n + 7) / 8) as u32;
            blaunch!(self, "attn_prefill_fa_sc_b", (gx, nh as u32, 1), (256, 1, 1), 0,
                (*q.device_ptr() as u64, kc_ptr, vc_ptr, d(attn),
                 n as i32, nh as i32, nkv as i32, hd as i32, stride as i32, pos_start as i32));
            return;
        }

        let s_buf = pool.get(br * bc * nkv);          // S: [Br x Bc] f32, one slab per kv head
        let p_buf = pool.get_bf16(br * bc * nkv);     // P: same, bf16, fed to the PV GEMM
        let o_buf = pool.get(br * hd * nh);           // O: [Br x hd] f32 per query head, carried
        let m_buf = pool.get(br * nh);
        let l_buf = pool.get(br * nh);

        let handle = *self.blas.handle();
        let one: f32 = 1.0;
        let zero: f32 = 0.0;
        let bf = cudaDataType::CUDA_R_16BF;
        let f32t = cudaDataType::CUDA_R_32F;
        let comp = cublasComputeType_t::CUBLAS_COMPUTE_32F;
        let algo = cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT;

        let mut t0 = 0usize;
        while t0 < n {
            let rows = (n - t0).min(br);              // live rows in this query tile
            blaunch!(self, "attn_tile_init", grid(nh * br * hd), (256,1,1), 0,
                (d(&o_buf), d(&m_buf), d(&l_buf), br as i32, hd as i32, nh as i32));

            // The last key any query in this tile may see. Key tiles beyond it are fully masked.
            let kmax = pos_start + t0 + rows - 1;
            let mut s0 = 0usize;
            while s0 <= kmax {
                let kcols = bc.min(pc - s0);
                for r in 0..gqa {
                    // ---- Sᵀ = Kᵀ · Q_tile, batched over kv heads ----
                    // TRANSPOSED on purpose: the softmax walks one query's keys, and it must read them
                    // contiguously. C is [kcols x rows], ld = bc, so row i of S lives at i*bc.
                    unsafe {
                        cudarc::cublas::result::gemm_strided_batched_ex(
                            handle,
                            OP::CUBLAS_OP_T, OP::CUBLAS_OP_N,
                            kcols as i32, rows as i32, hd as i32,
                            &scale as *const f32 as *const _,      // fold 1/sqrt(hd) into alpha
                            (kc_ptr + (s0 * hd * 2) as u64) as *const _,
                            bf, hd as i32, (stride * hd) as i64,
                            (*q.device_ptr() as u64 + ((r * hd + t0 * nh * hd) * 2) as u64) as *const _,
                            bf, (nh * hd) as i32, (gqa * hd) as i64,
                            &zero as *const f32 as *const _,
                            *s_buf.device_ptr() as *mut _,
                            f32t, bc as i32, (br * bc) as i64,
                            nkv as i32, comp, algo,
                        ).expect("prefill K^T Q");
                    }

                    // Packed args: cudarc's launch tuple caps at 12. scale rides in the GEMM alpha.
                    let nh_gqa_r = (nh * 10000 + gqa * 100 + r) as i32;
                    let s0_pc = (((s0 as i64) << 32) | (pc as i64)) as i64;
                    let smem = 256u32 * 4;
                    blaunch!(self, "attn_softmax_tile", (br as u32, nkv as u32, 1), (256,1,1), smem,
                        (d(&s_buf), d(&p_buf), d(&o_buf), d(&m_buf), d(&l_buf),
                         // Bc is the SLAB STRIDE (the allocated tile width), not the valid column
                         // count: the GEMM strides S/P slabs by br*bc, so indexing them by kcols
                         // sends every kv head after the first to the wrong slab. Columns past the
                         // key range are masked out by the causal test anyway.
                         br as i32, bc as i32, hd as i32, nh_gqa_r, rows as i32,
                         (pos_start + t0) as i32, s0_pc));

                    // ---- O += P · V_tileᵀ (beta=1 accumulates across key tiles) ----
                    // P is stored transposed too ([kcols x rows], ld = bc), so A is transposed here.
                    unsafe {
                        cudarc::cublas::result::gemm_strided_batched_ex(
                            handle,
                            OP::CUBLAS_OP_T, OP::CUBLAS_OP_T,
                            rows as i32, hd as i32, kcols as i32,
                            &one as *const f32 as *const _,
                            *p_buf.device_ptr() as *const _,
                            bf, bc as i32, (br * bc) as i64,
                            (vc_ptr + (s0 * hd * 2) as u64) as *const _,
                            bf, hd as i32, (stride * hd) as i64,
                            &one as *const f32 as *const _,
                            (*o_buf.device_ptr() as u64 + (r * br * hd * 4) as u64) as *mut _,
                            f32t, br as i32, (gqa * br * hd) as i64,
                            nkv as i32, comp, algo,
                        ).expect("prefill PV");
                    }
                }
                s0 += bc;
            }

            assert!(hd <= 256, "attn_finalize smem tile is sized for hd <= 256");
            blaunch!(self, "attn_finalize", ((br.div_ceil(32)) as u32, nh as u32, 1), (32u32,8u32,1), 0,
                (d(attn), d(&o_buf), d(&l_buf), br as i32, hd as i32, nh as i32, t0 as i32, n as i32));
            t0 += br;
        }

        pool.release(s_buf, br * bc * nkv);
        pool.release_bf16(p_buf, br * bc * nkv);
        pool.release(o_buf, br * hd * nh);
        pool.release(m_buf, br * nh);
        pool.release(l_buf, br * nh);
    }

    /// Canonical attention KV-write + softmax dispatch, shared by decode (`full_attn_batch`) and
    /// MTP verify (`verify_forward`). Writes new K/V via `write_kv_b` (slot-indexed) then runs
    /// `gqa_attn_splitk`+`gqa_attn_reduce` — the ONLY decode-attention path there is.
    /// Both call sites MUST use this so the verify forward (batch N) column 0 is
    /// bit-identical to a single-token decode (N=1) at the same position — the lossless-MTP contract.
    /// kc_ptr/vc_ptr are BASE cache pointers (slot offset applied inside the kernels via slot_ids);
    /// `max_pc` is the largest `pos[b]+1` across the batch.
    /// F4/F0: `q_pitch` is the q row pitch (nh*hd for a packed q, the fused qkv mtot for an offset
    /// view — carried to the splitk readers via bs_packed and to rotate_q_tq directly);
    /// `kv_prewritten` skips the whole write phase when the fused k-side rope kernel
    /// (rmsnorm_rope_kvwrite_b / _q4_b) already wrote the cache — then k/v are not read here.
    pub(super) fn attn_dispatch(&self, pool: &mut Pool, q: &B, k: &B, v: &B, q_pitch: usize, kv_prewritten: bool,
                     pos_dev: &cudarc::driver::CudaSlice<i32>,
                     kc_ptr: u64, vc_ptr: u64, slot_ids_ptr: u64, max_pc: usize, kv_stride: usize, batch: usize,
                     prefill_pos_start: Option<usize>, path_ptr: u64, rope_ptr: u64, col_pos_start_ptr: u64,
                     sharded: bool, kv_mode: KVCacheMode, chain_ok: bool, qsa_sel: Option<(u64, u64)>) -> B {
        let cfg = &self.cfg;
        // Head counts must match the CALLER's tensors, not a global flag. The MTP head is replicated
        // under TP, so it hands us full-width q/k/v and a full-width KV cache; deriving the counts from
        // eff_num_heads() here ran its attention at HALF the heads and produced garbage drafts
        // (acceptance collapsed 58% -> 2.7%). Third instance of the same root cause: sharding is a
        // property of the tensors passed in, never of the process.
        let (nh, nkv, hd) = if sharded { (self.eff_num_heads(), self.eff_num_kv_heads(), cfg.head_dim) }
                            else { (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim) };
        let stride = kv_stride;
        let scale = 1.0f32/(hd as f32).sqrt();

        // PREFILL mode: a wide causal-append into ONE slot's cache, positions pos_start..pos_start+N-1.
        // Used by the prompt prefill and by the MTP head's prompt prime. The slot-indexed decode
        // kernels are wrong here for two reasons: they need a slot_ids entry per column (the MTP head's
        // is a single permanently-zero slot, sized for MAX_VERIFY, so a wide batch reads past it), and
        // their split-K fan-out would launch batch*nh*n_splits blocks -- millions at an 8K prompt.
        let nw = (hd / 32) as u32;                      // warps per block (blockDim = hd = 256)
        let smem = (nw * hd as u32 + 2 * nw) * 4;
        if kv_mode == KVCacheMode::Tq {
            assert_eq!(hd, 128, "the TurboQuant KV layout is d=128 (Hy3) only");
        }
        if let Some(ps) = prefill_pos_start {
            match kv_mode {
            KVCacheMode::Q4 => {
                // 4-bit cache: quantize the chunk at write, then dequantize the whole prefix into a
                // bf16 scratch for the attention readers (identical formula to the splitk reader).
                // NOTE (E2 Fix 2): this arm keeps the FULL per-call re-dequant — the incremental
                // mirror lives in prefill_batch, the only production path that re-dequants per
                // window. This arm's callers take it once per slot at most: mtp_prime_prompt's MTP
                // head passes kv_mode=Bf16 (the draft cache stays bf16 by design, mtp_forward), and
                // capture_prefill is a debug probe at pos_start=0 (a first window, which the lazy
                // mirror would route here anyway). Plumbing BatchGpuState through attn_dispatch for
                // zero behavioral change was rejected. out_stride = npos: one-shot scratch layout.
                blaunch!(self, "write_kv_prefill_q4", grid(batch*nkv*(hd/16)), (256,1,1), 0,
                    (kc_ptr, vc_ptr, d(k), d(v), stride as i32, nkv as i32, hd as i32, batch as i32, ps as i32));
                let npos = ps + batch;
                let mut kdq = pool.get_bf16(nkv * npos * hd);
                let mut vdq = pool.get_bf16(nkv * npos * hd);
                blaunch!(self, "dequant_kv_q4", grid(npos*nkv*(hd/16)), (256,1,1), 0,
                    (d(&kdq), kc_ptr, nkv as i32, stride as i32, hd as i32, npos as i32, 0i32, npos as i32));
                blaunch!(self, "dequant_kv_q4", grid(npos*nkv*(hd/16)), (256,1,1), 0,
                    (d(&vdq), vc_ptr, nkv as i32, stride as i32, hd as i32, npos as i32, 0i32, npos as i32));
                let (kqs, vqs) = (d(&kdq), d(&vdq));
                let mut attn = pool.get_bf16(nh*hd*batch);
                if batch >= PF_MIN && !prefill_scalar() {
                    self.attn_prefill_tiled(pool, q, kqs, vqs, npos, nh, nkv, batch, ps, &mut attn);
                } else {
                    let qt = (hd / 32).max(1);
                    let ntile = batch.div_ceil(qt);
                    blaunch!(self, "gqa_attn_prefill", ((ntile*nh) as u32,1,1), (hd as u32,1,1), 0,
                        (d(&attn), d(q), kqs, vqs, npos as i32, nh as i32, nkv as i32, hd as i32,
                         fbits(scale), batch as i32, ps as i32));
                }
                pool.release_bf16(kdq, nkv*npos*hd); pool.release_bf16(vdq, nkv*npos*hd);
                return attn;
            }
            KVCacheMode::K8v4 => {
                // k8v4 prefill: pack-at-write (write_kv_prefill_k8v4 — int8-K + fp16-scale blocks
                // at the 20 B/16 stride, V at the q4 12 B/16 stride), then dequant the whole
                // prefix into a bf16 scratch for the attention readers: dequant_kv_k8v4 for K
                // (bf16(fp32(int8) × fp32(fp16 scale))) + dequant_kv_q4 for V (unchanged), the
                // exact pair the splitk readers use. Same one-shot scratch semantics as q4.
                blaunch!(self, "write_kv_prefill_k8v4", grid(batch*nkv*(hd/16)), (256,1,1), 0,
                    (kc_ptr, vc_ptr, d(k), d(v), stride as i32, nkv as i32, hd as i32, batch as i32, ps as i32));
                let npos = ps + batch;
                let mut kdq = pool.get_bf16(nkv * npos * hd);
                let mut vdq = pool.get_bf16(nkv * npos * hd);
                blaunch!(self, "dequant_kv_k8v4", grid(npos*nkv*(hd/16)), (256,1,1), 0,
                    (d(&kdq), kc_ptr, nkv as i32, stride as i32, hd as i32, npos as i32, 0i32, npos as i32));
                blaunch!(self, "dequant_kv_q4", grid(npos*nkv*(hd/16)), (256,1,1), 0,
                    (d(&vdq), vc_ptr, nkv as i32, stride as i32, hd as i32, npos as i32, 0i32, npos as i32));
                let (kqs, vqs) = (d(&kdq), d(&vdq));
                let mut attn = pool.get_bf16(nh*hd*batch);
                if batch >= PF_MIN && !prefill_scalar() {
                    self.attn_prefill_tiled(pool, q, kqs, vqs, npos, nh, nkv, batch, ps, &mut attn);
                } else {
                    let qt = (hd / 32).max(1);
                    let ntile = batch.div_ceil(qt);
                    blaunch!(self, "gqa_attn_prefill", ((ntile*nh) as u32,1,1), (hd as u32,1,1), 0,
                        (d(&attn), d(q), kqs, vqs, npos as i32, nh as i32, nkv as i32, hd as i32,
                         fbits(scale), batch as i32, ps as i32));
                }
                pool.release_bf16(kdq, nkv*npos*hd); pool.release_bf16(vdq, nkv*npos*hd);
                return attn;
            }
            KVCacheMode::K8v8 => {
                // k8v8 prefill: pack-at-write (write_kv_prefill_k8v8 — int8 codes + fp16 block
                // scales at the 20 B/16 stride on BOTH channels), then dequant the whole prefix
                // to bf16 scratch for the prefill readers: dequant_kv_k8v4 is channel-agnostic
                // (the int8-block layout is identical on K and V here), so it runs twice. Same
                // one-shot scratch semantics as q4/k8v4. The DECODE path reads the cache
                // directly (gqa_attn_verify_e_k8v8) — this dequant is prefill-only.
                blaunch!(self, "write_kv_prefill_k8v8", grid(batch*nkv*(hd/16)), (256,1,1), 0,
                    (kc_ptr, vc_ptr, d(k), d(v), stride as i32, nkv as i32, hd as i32, batch as i32, ps as i32));
                let npos = ps + batch;
                let mut kdq = pool.get_bf16(nkv * npos * hd);
                let mut vdq = pool.get_bf16(nkv * npos * hd);
                blaunch!(self, "dequant_kv_k8v4", grid(npos*nkv*(hd/16)), (256,1,1), 0,
                    (d(&kdq), kc_ptr, nkv as i32, stride as i32, hd as i32, npos as i32, 0i32, npos as i32));
                blaunch!(self, "dequant_kv_k8v4", grid(npos*nkv*(hd/16)), (256,1,1), 0,
                    (d(&vdq), vc_ptr, nkv as i32, stride as i32, hd as i32, npos as i32, 0i32, npos as i32));
                let (kqs, vqs) = (d(&kdq), d(&vdq));
                let mut attn = pool.get_bf16(nh*hd*batch);
                if batch >= PF_MIN && !prefill_scalar() {
                    self.attn_prefill_tiled(pool, q, kqs, vqs, npos, nh, nkv, batch, ps, &mut attn);
                } else {
                    let qt = (hd / 32).max(1);
                    let ntile = batch.div_ceil(qt);
                    blaunch!(self, "gqa_attn_prefill", ((ntile*nh) as u32,1,1), (hd as u32,1,1), 0,
                        (d(&attn), d(q), kqs, vqs, npos as i32, nh as i32, nkv as i32, hd as i32,
                         fbits(scale), batch as i32, ps as i32));
                }
                pool.release_bf16(kdq, nkv*npos*hd); pool.release_bf16(vdq, nkv*npos*hd);
                return attn;
            }
            KVCacheMode::Tq => {
                // TurboQuant prefill: pack-at-write (write_kv_prefill_tq), then dequant the whole
                // prefix into a bf16 scratch for the prefill readers (dequant_kv_tq materializes
                // decode_k/decode_v — the UNBIASED K reconstruction + the MSE V, the values the
                // splitk score path is consistent with). Same one-shot scratch semantics as q4.
                blaunch!(self, "write_kv_prefill_tq", ((batch*nkv) as u32,1,1), (128,1,1),
                    if self.kv_tq_b3 { TQ_ENCODE_SMEM_BYTES_B3 } else { TQ_ENCODE_SMEM_BYTES },
                    (kc_ptr, vc_ptr, d(k), d(v), d(&self.tq_tables), stride as i32, nkv as i32, batch as i32, ps as i32));
                let npos = ps + batch;
                let mut kdq = pool.get_bf16(nkv * npos * hd);
                let mut vdq = pool.get_bf16(nkv * npos * hd);
                blaunch!(self, "dequant_kv_tq", ((npos*nkv) as u32,1,1), (128,1,1), TQ_DEQ_SMEM_BYTES,
                    (d(&kdq), d(&vdq), kc_ptr, vc_ptr, d(&self.tq_tables),
                     nkv as i32, stride as i32, npos as i32, 0i32, npos as i32));
                let (kqs, vqs) = (d(&kdq), d(&vdq));
                let mut attn = pool.get_bf16(nh*hd*batch);
                if batch >= PF_MIN && !prefill_scalar() {
                    self.attn_prefill_tiled(pool, q, kqs, vqs, npos, nh, nkv, batch, ps, &mut attn);
                } else {
                    let qt = (hd / 32).max(1);
                    let ntile = batch.div_ceil(qt);
                    blaunch!(self, "gqa_attn_prefill", ((ntile*nh) as u32,1,1), (hd as u32,1,1), 0,
                        (d(&attn), d(q), kqs, vqs, npos as i32, nh as i32, nkv as i32, hd as i32,
                         fbits(scale), batch as i32, ps as i32));
                }
                pool.release_bf16(kdq, nkv*npos*hd); pool.release_bf16(vdq, nkv*npos*hd);
                return attn;
            }
            KVCacheMode::Bf16 => {
            blaunch!(self, "write_kv_prefill", grid(batch*nkv*hd), (256,1,1), 0,
                (kc_ptr, vc_ptr, d(k), d(v), stride as i32, nkv as i32, hd as i32, batch as i32, ps as i32));
            let mut attn = pool.get_bf16(nh*hd*batch);
            if let Some((sel_ptr, pos_sel_ptr)) = qsa_sel {
                // qwen4_exp QSA: causal attention over each query's selection list.
                self.qsa_attn_prefill_ptrs(&mut attn, q, kc_ptr, vc_ptr, stride, nh, nkv, hd, scale, batch, sel_ptr, pos_sel_ptr);
            } else if batch >= PF_MIN && !prefill_scalar() {
                // Tensor-core path: the two GEMMs go through cuBLAS. See attn_prefill_tiled.
                self.attn_prefill_tiled(pool, q, kc_ptr, vc_ptr, stride, nh, nkv, batch, ps, &mut attn);
            } else {
                // Short chunk: the scalar kernel's fixed costs are lower than the tiled path's.
                let qt = (hd / 32).max(1);      // one query per warp (GQA_PF_QT at hd=256)
                let ntile = batch.div_ceil(qt);
                blaunch!(self, "gqa_attn_prefill", ((ntile*nh) as u32,1,1), (hd as u32,1,1), 0,
                    (d(&attn), d(q), kc_ptr, vc_ptr, stride as i32, nh as i32, nkv as i32, hd as i32,
                     fbits(scale), batch as i32, ps as i32));
            }
            return attn;
            }
        }
        }

        let pos_ptr = *pos_dev.device_ptr() as u64;
        let logical_ptr = if rope_ptr != 0 { rope_ptr } else { pos_ptr };  // splitk/reduce use LOGICAL pos (rope); write_kv uses the KV SLOT (pos)
        // TQ: the rotate-once query buffer (qr = Pi·q, qs = S·q, interleaved f32), produced by
        // rotate_q_tq and consumed by the splitk kernels below (kept alive across the dispatch).
        let mut tq_qrqs: Option<S> = None;
        if !kv_prewritten {
        match kv_mode {
        KVCacheMode::Q4 => {
            blaunch!(self, "write_kv_b_q4", grid(batch*nkv*(hd/16)), (256,1,1), 0,
                (kc_ptr, vc_ptr, d(k), d(v), pos_ptr, stride as i32, nkv as i32, hd as i32, batch as i32, slot_ids_ptr));
        }
        KVCacheMode::K8v4 => {
            blaunch!(self, "write_kv_b_k8v4", grid(batch*nkv*(hd/16)), (256,1,1), 0,
                (kc_ptr, vc_ptr, d(k), d(v), pos_ptr, stride as i32, nkv as i32, hd as i32, batch as i32, slot_ids_ptr));
        }
        KVCacheMode::K8v8 => {
            blaunch!(self, "write_kv_b_k8v8", grid(batch*nkv*(hd/16)), (256,1,1), 0,
                (kc_ptr, vc_ptr, d(k), d(v), pos_ptr, stride as i32, nkv as i32, hd as i32, batch as i32, slot_ids_ptr));
        }
        KVCacheMode::Tq => {
            // pack-at-write (one block per (b, kvh) row), then rotate every query head once
            // (the "rotate once, never rotate back" principle of the reference).
            blaunch!(self, "write_kv_b_tq", ((batch*nkv) as u32,1,1), (128,1,1),
                if self.kv_tq_b3 { TQ_ENCODE_SMEM_BYTES_B3 } else { TQ_ENCODE_SMEM_BYTES },
                (kc_ptr, vc_ptr, d(k), d(v), d(&self.tq_tables), pos_ptr, stride as i32, nkv as i32, batch as i32, slot_ids_ptr));
            let mut qrqs = pool.get(batch * nh * 2 * hd);
            blaunch!(self, "rotate_q_tq", ((batch*nh) as u32,1,1), (128,1,1), TQ_ROTATE_SMEM_BYTES,
                (d(&qrqs), d(q), d(&self.tq_tables), nh as i32, hd as i32, batch as i32, q_pitch as i32));
            tq_qrqs = Some(qrqs);
        }
        KVCacheMode::Bf16 => {
            blaunch!(self, "write_kv_b", grid(batch*nkv*hd), (256,1,1), 0,
                (kc_ptr, vc_ptr, d(k), d(v), pos_ptr, stride as i32, nkv as i32, hd as i32, batch as i32, slot_ids_ptr));
        }
        }
        }
        let attn = pool.get_bf16(nh*hd*batch);
        // nh_packed carries (nh, hd, nkv) bit-packed: nh<2048 (bits 20+), hd<=1023 (bits 10-19),
        // nkv<=1023 (bits 0-9). gqa_attn_splitk/reduce unpack the same layout.
        // (pack_nh asserts the three widths in release builds too.)
        // TRAP-7 envelope: gqa_attn_sel_prefill2 sizes fixed per-lane arrays for hd/32 <= 8 (hd <= 256;
        // Flash-Next is exactly 256) — the dispatch below routes wider hd to gqa_attn_sel_prefill,
        // which is parameterized by SK_DPL_MAX (hd <= SK_HD_MAX == 512). Assert the whole QSA
        // envelope so a wider config fails here instead of producing silent garbage (the hd=128
        // hardcode OOB precedent, AGENTS §7).
        assert!(hd <= 512, "qwen4_exp QSA selection kernels cover hd <= 512 only (got hd={hd}); raise SK_HD_MAX / SEL arrays first");
        let nh_packed = pack_nh(nh, hd, nkv);

        // ===================== THE LOSSLESS-MTP CONTRACT =====================
        //
        // Attention for a query at position p must be a pure function of (q, KV[0..p], pc). It may NOT
        // depend on `batch`, on `max_pc`, or on anything else derived from the other columns -- because
        // column k of an N-wide verify has to be BIT-IDENTICAL to a 1-token decode at that position.
        //
        // Two versions of this got it wrong, and the second is the instructive one:
        //   * n_splits from batch*nh -> a decode split the keys 6 ways, a 4-wide verify 2 ways.
        //   * n_splits from max_pc   -> STILL WRONG. A decode has max_pc = pos+1, a verify has
        //     max_pc = pos_start+N. They straddle a 256-boundary regularly, and when one landed on
        //     n_splits==1 it ran a DIFFERENT KERNEL (gqa_attn_flash) with different rounding. MTP was
        //     silently non-lossless and STILL PASSED THE GATE most runs, because a 1-ulp difference
        //     rarely flips an argmax. --bench-mtp only caught it once an unrelated change perturbed
        //     the trajectory. A gate that fails as a coin toss is not a gate.
        //
        // So the split count is now computed INSIDE the kernel from each column's OWN pc (sk_nsplits),
        // there is only ONE kernel (gqa_attn_flash is deleted), and the partials are fp32. `ns_grid` is
        // purely a launch bound: pc_b <= max_pc implies ns_b <= ns_grid, so every block a column needs
        // exists, and the surplus ones return immediately.
        // F8/C: tensor-core verify attention (the dspark ring_e pattern ported to the trunk
        // GQA decode/verify). Gated to identity-path, single-slot, chain shapes; everything
        // else (tree verify, multi-lane batch) stays on the scalar splitk, bit-identical.
        let ratio_now = nh / nkv.max(1);
        // F9: the attention kernel choice is a property of the REQUEST, never of the verify
        // width. `batch * ratio_now <= 48` used to switch a wide verify (b>=9 at QG=6) onto the
        // splitk lane while a decode stayed on _e — a column-0 cliff. The _e lane now covers up
        // to MAX_VERIFY columns via a row-group grid dimension (grid.z = ceil(NQ/48)), so the
        // gate is the contract width.
        debug_assert_eq!(MAX_VERIFY, 16, "gpu_batch.cu GVE_MAX_VERIFY must match gpu.rs MAX_VERIFY");
        let use_e = chain_ok && matches!(kv_mode, KVCacheMode::Bf16 | KVCacheMode::K8v8)
            && (hd == 128 || hd == 256)
            && batch >= 1 && batch <= MAX_VERIFY
            && crate::opts::var(crate::opt!("no-attn-e")).is_err();
        // k8v8 multi-lane: the `_e` tensor-core kernel addresses ONE slot for all its columns, so a call whose columns
        // are independent sequences (plain batched decode of several lanes) cannot be ONE `_e` launch. It CAN be one
        // batch-1 `_e` launch per column (pointer offsets, below): column j then runs the exact kernel and arithmetic
        // a lone request runs, so a lane's bytes never depend on how many other lanes are busy. Valid for identity-path
        // calls only (a tree column attends through `path`, which `_e` does not read); those take the int8 split-K pair.
        let use_e_cols = !use_e && kv_mode == KVCacheMode::K8v8 && path_ptr == 0 && qsa_sel.is_none()
            && (hd == 128 || hd == 256) && batch >= 2 && batch <= MAX_VERIFY
            && crate::opts::var(crate::opt!("no-attn-e")).is_err();
        let use_e_lane = use_e || use_e_cols;   // partials in the `_e` layout (ns_grid == SEG), reduced by gqa_attn_reduce_e
        if crate::opts::var(crate::opt!("attn-e-debug")).is_ok() {
            eprintln!("[attn_e] chain_ok={chain_ok} kv={:?} hd={hd} batch={batch} ratio={ratio_now} max_pc={max_pc} use_e={use_e} use_e_cols={use_e_cols}", kv_mode);
        }
        let seg_e: usize = if use_e_lane {
            // grid = nkv * SEG: cover the SMs (48) with ~2 waves, then keep the per-segment
            // serial scan bounded (~8K keys) at long ctx. SEG <= 63 (bs_packed field is 6 bits).
            // F9: SEG comes from kv_stride — a per-request constant, so a decode and a verify of
            // the same request resolve the SAME SEG (the pre-fix bug: decode passed pos+1 and
            // verify passed kv_stride, so SEG could differ). The key partition itself comes from
            // pos[0] + MAX_VERIFY inside the kernel, never from this launch bound.
            let base = ((192 + nkv - 1) / nkv).max(4);
            base.max((kv_stride + 8191) / 8192).min(63)
        } else { 0 };
        let ns_grid = if use_e_lane { seg_e } else { (max_pc / 256).clamp(1, 32) };
        // F9: _e row groups (6 warps x 8 real rows per block); 1 for every serving width <= 8.
        let rg_e = if use_e { (batch * ratio_now).div_ceil(48).max(1) } else { 1 };
        let n_partial = batch * nh * ns_grid;
        let pm = pool.get(n_partial);
        let pl = pool.get(n_partial);
        let pa = pool.get(n_partial * hd);          // fp32: the bf16 round-trip was lossy for nothing
        // bs_packed = stride(bits 0-18) | ns_grid(19-24) | batch(25-30) | q_pitch(31-49), a u64
        // (the splitk family unpacks long long) — stride packed to fit the 12-arg launch cap; ranges:
        // stride<=262144<2^19, ns_grid<=32<2^6, batch<2^6, q_pitch<2^19 (the fused qkv mtot / nh*hd).
        let bs_packed: u64 = pack_bs(stride, ns_grid, batch, q_pitch);
        // GQA ratio and escape hatch are shared by the q4 and bf16 dispatch below.
        let gqa_ratio = nh / nkv.max(1);
        // A/B + escape hatch: --no-gqpack=1 forces the per-head kernel. Read per call (the
        // H16/--no-decode-graphs pattern); under graph capture it is bound at capture time.
        let no_gqpack = crate::opts::var(crate::opt!("no-gqpack")).is_ok();
        // TQ splitk smem: per-head = sacc(NW*hd) + sm/sl + tacc(hd); gq adds the 8-head qr/qs stage.
        let tq_smem = (nw * hd as u32 + 2 * nw + hd as u32) * 4;
        let tq_smem_gq = (2 * 8 * hd as u32 + nw * hd as u32 + 2 * nw + hd as u32) * 4;
        match kv_mode {
        KVCacheMode::Q4 => {
            // GQA-packed split-K (E1): one block per (kv head, split) reads the chunk ONCE for the
            // whole query group — the per-head kernel re-reads it gqa_ratio times (8× at Hy3, the
            // "32K anomaly": 53.6 ms/token of re-reads at 26K ctx). Bit-identical per head, so
            // decode == verify col-0 is preserved. hd==128 && ratio in 2..=8 only; else per-head.
            if (hd == 128 || hd == 256) && (2..=8).contains(&gqa_ratio) && !no_gqpack {
                blaunch!(self, "gqa_attn_splitk_q4_gq", ((batch * nkv * ns_grid) as u32,1,1), (hd as u32,1,1), smem,
                    (d(&pm), d(&pl), d(&pa), d(q), kc_ptr, vc_ptr,
                     logical_ptr, bs_packed, nh_packed, slot_ids_ptr, path_ptr, col_pos_start_ptr));
            } else {
            blaunch!(self, "gqa_attn_splitk_q4", ((batch * nh * ns_grid) as u32,1,1), (hd as u32,1,1), smem,
                (d(&pm), d(&pl), d(&pa), d(q), kc_ptr, vc_ptr,
                 logical_ptr, bs_packed, nh_packed, slot_ids_ptr, path_ptr, col_pos_start_ptr));
            }
        }
        KVCacheMode::K8v4 => {
            // k8v4 split-K: same GQA-packed structure and bit-identity argument as the q4 pair —
            // the K reader swaps to the u32-int8 + fp16-scale dequant (dequant_kv_k8v4's formula),
            // the V channel stays the q4 nibble reader. K rows are 20 B/16, V rows 12 B/16, so
            // the packed kernels carry both row sizes; the launch shapes are unchanged.
            if let Some((sel_ptr, pos_sel_ptr)) = qsa_sel {
                // qwen4_exp QSA over the k8v4 cache: the selection-list reader with the k8v4 row
                // dequant (gqa_attn_sel_splitk_k8v4); the reduce below reads the same pos_sel.
                blaunch!(self, "gqa_attn_sel_splitk_k8v4", ((batch * nh * ns_grid) as u32,1,1), (hd as u32,1,1), smem,
                    (d(&pm), d(&pl), d(&pa), d(q), kc_ptr, vc_ptr,
                     pos_sel_ptr, bs_packed, nh_packed, slot_ids_ptr, sel_ptr, self.qsa_limit() as i32));
            } else if (hd == 128 || hd == 256) && (2..=8).contains(&gqa_ratio) && !no_gqpack {
                blaunch!(self, "gqa_attn_splitk_k8v4_gq", ((batch * nkv * ns_grid) as u32,1,1), (hd as u32,1,1), smem,
                    (d(&pm), d(&pl), d(&pa), d(q), kc_ptr, vc_ptr,
                     logical_ptr, bs_packed, nh_packed, slot_ids_ptr, path_ptr, col_pos_start_ptr));
            } else {
            blaunch!(self, "gqa_attn_splitk_k8v4", ((batch * nh * ns_grid) as u32,1,1), (hd as u32,1,1), smem,
                (d(&pm), d(&pl), d(&pa), d(q), kc_ptr, vc_ptr,
                 logical_ptr, bs_packed, nh_packed, slot_ids_ptr, path_ptr, col_pos_start_ptr));
            }
        }
        KVCacheMode::K8v8 => {
            if use_e {
                // Single-request lane (decode b=1 / chain verify): the p8b tensor-core kernel (p3 core +
                // in-register block dequant) reads ONE slot for every column.
                blaunch!(self, "gqa_attn_verify_e_k8v8", (nkv as u32, seg_e as u32, rg_e as u32), (192,1,1), 0,
                    (d(&pm), d(&pl), d(&pa), d(q), kc_ptr, vc_ptr,
                     logical_ptr, bs_packed, nh_packed, slot_ids_ptr, col_pos_start_ptr));
            } else if use_e_cols {
                // Independent columns (plain batched decode of several lanes): one batch-1 `_e` launch per
                // column, outputs/inputs offset to the column — the same kernel a lone request runs.
                let bs1: u64 = pack_bs(stride, ns_grid, 1, q_pitch);
                let (qp, pmp, plp, pap) = (d(q), d(&pm), d(&pl), d(&pa));
                for c in 0..batch as u64 {
                    let part = c * (nh * ns_grid) as u64;
                    blaunch!(self, "gqa_attn_verify_e_k8v8", (nkv as u32, seg_e as u32, 1u32), (192,1,1), 0,
                        (pmp + part * 4, plp + part * 4, pap + part * hd as u64 * 4, qp + c * q_pitch as u64 * 2,
                         kc_ptr, vc_ptr, logical_ptr + c * 4, bs1, nh_packed, slot_ids_ptr + c * 4, col_pos_start_ptr));
                }
            } else {
                // Everything the `_e` lane cannot serve — tree verify (a column attends through `path`), an
                // `--no-attn-e` A/B, widths above MAX_VERIFY: the int8 split-K readers (the k8v4 pair's structure,
                // V read as int8+fp16-block like K, dequant rounded to bf16 exactly like the `_e` staging). A non-e
                // dispatch must never fall through to a bf16 reader (a silent bf16 read of an int8 buffer is the
                // mojibake-class hazard), so this arm names its kernels explicitly.
                if (hd == 128 || hd == 256) && (2..=8).contains(&gqa_ratio) && !no_gqpack {
                    blaunch!(self, "gqa_attn_splitk_k8v8_gq", ((batch * nkv * ns_grid) as u32,1,1), (hd as u32,1,1), smem,
                        (d(&pm), d(&pl), d(&pa), d(q), kc_ptr, vc_ptr,
                         logical_ptr, bs_packed, nh_packed, slot_ids_ptr, path_ptr, col_pos_start_ptr));
                } else {
                    blaunch!(self, "gqa_attn_splitk_k8v8", ((batch * nh * ns_grid) as u32,1,1), (hd as u32,1,1), smem,
                        (d(&pm), d(&pl), d(&pa), d(q), kc_ptr, vc_ptr,
                         logical_ptr, bs_packed, nh_packed, slot_ids_ptr, path_ptr, col_pos_start_ptr));
                }
            }
        }
        KVCacheMode::Tq => {
            // TurboQuant split-K: same GQA-packed structure (one block per (kvh, split), the E1
            // win), same per-head bit-identity, reading the pre-rotated qrqs (rotate_q_tq above).
            // TQ is hd==128 only; the q4_gq launch gate's hd==256 arm doesn't apply.
            let qrqs_ptr = d(tq_qrqs.as_ref().expect("TQ dispatch requires the qrqs buffer"));
            if (2..=8).contains(&gqa_ratio) && !no_gqpack {
                blaunch!(self, "gqa_attn_splitk_tq_gq", ((batch * nkv * ns_grid) as u32,1,1), (hd as u32,1,1), tq_smem_gq,
                    (d(&pm), d(&pl), d(&pa), qrqs_ptr, kc_ptr, vc_ptr, d(&self.tq_tables),
                     logical_ptr, bs_packed, nh_packed, slot_ids_ptr, path_ptr));
            } else {
            blaunch!(self, "gqa_attn_splitk_tq", ((batch * nh * ns_grid) as u32,1,1), (hd as u32,1,1), tq_smem,
                (d(&pm), d(&pl), d(&pa), qrqs_ptr, kc_ptr, vc_ptr, d(&self.tq_tables),
                 logical_ptr, bs_packed, nh_packed, slot_ids_ptr, path_ptr));
            }
        }
        KVCacheMode::Bf16 => {
            // F8/C: the tensor-core lane (ring_e pattern; see gqa_attn_verify_e). Runs both
            // matmuls on mma.sync bf16 with Q row-fragments persistent in registers; partials
            // land in the splitk layout and the _e reduce merges them (SEG == ns_grid).
            if use_e {
                // p3 (round-23): oracle-validated restructure, +18% at 30K-class ctx; A/B knob
                // until it earns standing-config (diagnostics-class env per §7; promote to CLI
                // flag with the fp8-KV integration).
                static ATTN_P3: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                let attn_p3 = *ATTN_P3.get_or_init(|| crate::opts::var(crate::opt!("attn-p3")).is_ok());
                let kname = if attn_p3 { "gqa_attn_verify_e_p3" } else { "gqa_attn_verify_e" };
                blaunch!(self, kname, (nkv as u32, seg_e as u32, rg_e as u32), (192,1,1), 0,
                    (d(&pm), d(&pl), d(&pa), d(q), kc_ptr, vc_ptr,
                     logical_ptr, bs_packed, nh_packed, slot_ids_ptr, col_pos_start_ptr));
            }
            // GQA-packed bf16 split-K (E1c): same re-read pathology as q4 — the per-head kernel
            // reads each group's KV chunk gqa_ratio times (the bf16 "32K anomaly"). Packed reads it
            // ONCE per (kvh, split); per-head bit-identical. hd in {128,256} && ratio 2..=8.
            if let Some((sel_ptr, pos_sel_ptr)) = qsa_sel {
                // qwen4_exp QSA: the per-head kernel over the column's selection list (pc =
                // pos_sel[b]+1 selected keys; the reduce below reads the same pos_sel).
                blaunch!(self, "gqa_attn_sel_splitk", ((batch * nh * ns_grid) as u32,1,1), (hd as u32,1,1), smem,
                    (d(&pm), d(&pl), d(&pa), d(q), kc_ptr, vc_ptr,
                     pos_sel_ptr, bs_packed, nh_packed, slot_ids_ptr, sel_ptr, self.qsa_limit() as i32));
            } else if (hd == 128 || hd == 256) && (2..=8).contains(&gqa_ratio) && !no_gqpack {
                blaunch!(self, "gqa_attn_splitk_gq", ((batch * nkv * ns_grid) as u32,1,1), (hd as u32,1,1), smem,
                    (d(&pm), d(&pl), d(&pa), d(q), kc_ptr, vc_ptr,
                     logical_ptr, bs_packed, nh_packed, slot_ids_ptr, path_ptr, col_pos_start_ptr));
            } else {
            blaunch!(self, "gqa_attn_splitk", ((batch * nh * ns_grid) as u32,1,1), (hd as u32,1,1), smem,
                (d(&pm), d(&pl), d(&pa), d(q), kc_ptr, vc_ptr,
                 logical_ptr, bs_packed, nh_packed, slot_ids_ptr, path_ptr, col_pos_start_ptr));
            }
        }
        }
        // TRAP-1: qsa_sel short-circuits the _e/_p3 lanes entirely. The sel kernels' partials
        // are reduced by plain gqa_attn_reduce with pos_sel below; letting use_e claim a QSA
        // column would reduce _e partials with pos_sel — wrong attention, no error.
        // TRAP-2: the (Bf16|K8v4, Some) arm is the ONLY sparse arm; qsa_enabled() asserts
        // kv_quant/kv_tq/kv_k8v8 so the fallthrough (_ => logical_ptr) is unreachable for QSA.
        let reduce_pos_ptr = match (kv_mode, qsa_sel) { (KVCacheMode::Bf16 | KVCacheMode::K8v4, Some((_, p))) => p, _ => logical_ptr };
        if use_e_lane && qsa_sel.is_none() {
            // _e twin: the partial count is ns_grid (== SEG) by construction — no sk_nsplits
            // recompute (the splitk reduce would read a different split count than we wrote).
            blaunch!(self, "gqa_attn_reduce_e", ((batch*nh) as u32,1,1), (hd as u32,1,1), 0,
                (d(&attn), d(&pm), d(&pl), d(&pa), logical_ptr, ns_grid as i32, batch as i32, nh_packed));
        } else {
        blaunch!(self, "gqa_attn_reduce", ((batch*nh) as u32,1,1), (hd as u32,1,1), 0,
            (d(&attn), d(&pm), d(&pl), d(&pa), reduce_pos_ptr, ns_grid as i32, batch as i32, nh_packed));
        }
        if let Some(qrqs) = tq_qrqs { pool.release(qrqs, batch * nh * 2 * hd); }
        pool.release(pm, n_partial);
        pool.release(pl, n_partial);
        pool.release(pa, n_partial * hd);
        attn
    }

    pub(super) fn full_attn_batch(&self, pool: &mut Pool, hidden: &B, fa: &GpuFullAttn, pos_dev: &cudarc::driver::CudaSlice<i32>,
                       max_pc: usize, kv_stride: usize, kc_ptr: u64, vc_ptr: u64, cos: &S, sin: &S, slot_ids_ptr: u64, batch: usize,
                       prefill_pos_start: Option<usize>, sharded: bool, kv_mode: KVCacheMode, fuse: bool,
                       qsa: Option<QsaIn>) -> ReduceOut {
        if crate::opts::var(crate::opt!("f9-xhash")).as_deref() == Ok("3") {
            let prev = F9X_ATTN.with(|c| c.replace(usize::MAX));
            if prev == usize::MAX { F9X_ATTN.with(|c| c.set(0)); }
            else { F9X_ATTN.with(|c| c.set(prev + 1)); }
        }
        let cfg = &self.cfg;
        let (h, hd, rdim) = (cfg.hidden_size, cfg.head_dim, cfg.rotary_dim);
        // qwen4_exp QSA: raw-key write + per-column selection lists (reads the same normed input as
        // the qkv projection). None => dense attention, the pre-indexer path byte for byte.
        let qsa_sel: Option<(S, S)> = qsa.as_ref().and_then(|qs| match prefill_pos_start {
            Some(ps) => self.qsa_select_prefill(pool, qs.idx, hidden, qs.keys_ptr, kv_stride, ps, batch, cos, sin),
            None => Some(self.qsa_select_cols(pool, qs.idx, hidden, qs.keys_ptr, kv_stride, *pos_dev.device_ptr() as u64,
                                              qs.logical_ptr, slot_ids_ptr, qs.path_ptr, qs.cps_ptr, cos, sin, batch, max_pc)),
        });
        let qsa_sel_ptrs: Option<(u64, u64)> = qsa_sel.as_ref().map(|(a, b)| (d(a), d(b)));
        // TP=2: each box owns half the heads (12Q/2KV). Weights + KV cache are sharded to match, so the
        // whole mixer runs at local head counts and a single all-reduce on the o_proj partial stitches
        // the boxes back together. world==1 → full counts, no all-reduce.
        let (nh, nkv) = if sharded { (self.eff_num_heads(), self.eff_num_kv_heads()) }
                        else { (self.cfg.num_heads, self.cfg.num_kv_heads) };
        // qwen3_5 full-attn fuses a sigmoid output-gate into q_proj (rows [q|gate] per head); hy_v3's
        // q_proj is bare (qmul=1) — no split_qgate, no sigmoid_gate.
        let qmul = if cfg.attn_out_gate() { 2 } else { 1 };
        let qg_dim = nh*hd*qmul;                 // local fused width: [q|gate | k | v]
        let kv_dim = nkv*hd;
        let mtot = qg_dim + 2*kv_dim;
        // F0/F4: the fused qkv GEMM output IS [token, [q|k|v]] — under the default the split is
        // deleted and q/k/v are offset views (base + b·mtot), so the fused buffer must live until
        // the mixer consumes it. The escapes (--no-qkv-view / --no-fused-perhead-rope), the
        // prefill arm, and the TQ/K8v4 modes (no fused write variant) restore the packed split —
        // the split kernels stay callable for the A/B byte gate.
        let old_pipeline = prefill_pos_start.is_some()
            || crate::opts::var(crate::opt!("no-qkv-view")).is_ok()
            || crate::opts::var(crate::opt!("no-fused-perhead-rope")).is_ok()
            || matches!(kv_mode, KVCacheMode::Tq | KVCacheMode::K8v4 | KVCacheMode::K8v8);
        let mut qg: Option<B> = None;            // packed q(|gate) — old pipeline / split weights
        let mut k: Option<B> = None;             // packed k — old pipeline / split weights
        let mut v: Option<B> = None;             // packed v — old pipeline / split weights
        let mut fused_qkv: Option<B> = None;      // F0: the fused qkv buffer, views handed to consumers
        match &fa.qkv {
            AttnIn::Fused(w) => {
                // ONE GEMM for q(|gate), k and v: they all read the same activation, and k/v alone are
                // M=1024 -- only 84 GB/s against a 234 GB/s machine, because grid=64 barely fills it.
                let mut fb = pool.get_bf16(mtot * self.pf8_pad(batch));
                // E9: the qkv projection is the first consumer GEMM of the TP barrier chain —
                // weight prefetch runs under the K2 cpu_done wait (plain `gemm_act` elsewhere).
                self.gemm_act_pdl(w, hidden, &mut fb, h, mtot, batch);
                if old_pipeline {
                    let qgb = pool.get_bf16(qg_dim*self.pf8_pad(batch));
                    let kb = pool.get_bf16(kv_dim*self.pf8_pad(batch));
                    let vb = pool.get_bf16(kv_dim*self.pf8_pad(batch));
                    blaunch!(self, "split_qkv_b", grid(mtot*batch), (256,1,1), 0,
                        (d(&qgb), d(&kb), d(&vb), d(&fb), qg_dim as i32, kv_dim as i32, batch as i32));
                    pool.release_bf16(fb, mtot * batch);
                    qg = Some(qgb); k = Some(kb); v = Some(vb);
                } else {
                    // F0: q/k/v are offset views of the fused output — one buffer, one release (tail).
                    fused_qkv = Some(fb);
                }
            }
            AttnIn::Split { q: qp, k: kp, v: vp } => {
                // E9: the FIRST (q) GEMM carries the programmatic launch; k/v follow it plainly.
                let mut qgb = pool.get_bf16(qg_dim*self.pf8_pad(batch));
                let mut kb = pool.get_bf16(kv_dim*self.pf8_pad(batch));
                let mut vb = pool.get_bf16(kv_dim*self.pf8_pad(batch));
                self.gemm_act_pdl(qp, hidden, &mut qgb, h, qg_dim, batch);
                self.gemm_act(kp, hidden, &mut kb, h, kv_dim, batch);
                self.gemm_act(vp, hidden, &mut vb, h, kv_dim, batch);
                qg = Some(qgb); k = Some(kb); v = Some(vb);
            }
        }
        // q|gate split: qmul==2 (qwen) always splits — from the qg VIEW (F0) or the packed qg — into
        // packed q + gate; qmul==1 (hy3): q IS the qg buffer (packed) or the fused slice itself (view).
        let (q, gate): (Option<B>, Option<B>) = if qmul == 2 {
            // qg source: the packed qg (old pipeline / split weights) or the fused qkv view (F0).
            let (qgs, qg_stride) = match (&qg, &fused_qkv) {
                (Some(qgs), _) => (d(qgs), qg_dim),
                (None, Some(f)) => (d(f), mtot),
                _ => unreachable!("qg source"),
            };
            let qb = pool.get_bf16(nh*hd*batch);
            let gb = pool.get_bf16(nh*hd*batch);
            blaunch!(self, "split_qgate_b", grid(nh*hd*batch), (256,1,1), 0,
                (d(&qb), d(&gb), qgs, nh as i32, hd as i32, batch as i32, qg_stride as i32));
            if let Some(qgs) = qg.take() { pool.release_bf16(qgs, qg_dim*batch); }
            (Some(qb), Some(gb))
        } else if fused_qkv.is_some() {
            (None, None)                       // q is the fused slice itself (view, pitch mtot)
        } else {
            (qg.take(), None)
        };
        // q row geometry for the rope / splitk readers: packed (pitch nh*hd) or the fused view (pitch mtot).
        let (q_ptr, q_pitch): (u64, usize) = match (&q, &fused_qkv) {
            (Some(qb), _) => (d(qb), nh*hd),
            (None, Some(f)) => (d(f), mtot),
            (None, None) => unreachable!("q source"),
        };
        // E14: the per-head q/k RMSNorm + RoPE are ONE fused kernel per buffer (byte-identical to
        // the old four-launch sequence — normed values bf16-rounded to smem, then the same rope
        // arithmetic). F4 folds the KV-cache write into the k-side kernel (norm+rope k in place,
        // then the roped k row AND the raw v row to the cache — three launches become two).
        // --no-fused-perhead-rope=1 keeps the old 4-launch sequence for the A/B byte gate.
        let eps = fbits(cfg.rms_eps);
        let attn = if !old_pipeline {
            if let Some(fb) = &fused_qkv {
                // F0+F4: q is a view at 0 (hy3) or the packed split_qgate output (qwen); k/v are views.
                blaunch!(self, "rmsnorm_rope_b", ((batch*nh) as u32,1,1), (hd as u32,1,1), (hd*4) as u32,
                    (q_ptr, d(&fa.q_norm), d(cos), d(sin), nh as i32, hd as i32, rdim as i32, batch as i32, eps, 0i32, q_pitch as i32));
                // k-side fused norm+rope+cache-write (F4; bf16 + q4 write variants — TQ/K8v4 are in
                // old_pipeline above). grid (b·nkv), block 256, smem hd fp32; 12-arg packed launch:
                // stride_nkv = nkv<<19|kv_stride, hd_rdim = hd<<16|rdim, k_off = qg_dim.
                let kvwrite = match kv_mode {
                    KVCacheMode::Q4 => "rmsnorm_rope_kvwrite_q4_b",
                    KVCacheMode::Bf16 => "rmsnorm_rope_kvwrite_b",
                    _ => unreachable!("F4 fused kvwrite: TQ/K8v4 use the old pipeline"),
                };
                let stride_nkv = pack_stride_nkv(nkv, kv_stride);
                let hd_rdim = ((hd as u32) << 16) | (rdim as u32);
                blaunch!(self, kvwrite, ((batch*nkv) as u32,1,1), (256,1,1), (hd*4) as u32,
                    (kc_ptr, vc_ptr, d(fb), d(&fa.k_norm), d(cos), d(sin), *pos_dev.device_ptr() as u64, slot_ids_ptr,
                     stride_nkv as i32, hd_rdim as i32, qg_dim as i32, eps));
                // F0 q-dispatch fix (Class B): qmul==2 (qwen) computed the normed+roped packed q in
                // `qb` above — dispatch THAT as q, not the raw fused view (the split_qgate_b +
                // rmsnorm_rope_b work was otherwise discarded and the attention read raw q).
                // k/v stay fused views (kv_prewritten=true: the fused kvwrite already cached them).
                let (q_ref, q_pitch_d) = match &q {
                    Some(qb) => (qb, nh * hd),
                    None => (fb, mtot),
                };
                {
                    let at = self.attn_dispatch(pool, q_ref, fb, fb, q_pitch_d, true, pos_dev, kc_ptr, vc_ptr, slot_ids_ptr, max_pc, kv_stride, batch, None, 0u64, 0u64, 0u64, sharded, kv_mode, batch == 1, qsa_sel_ptrs);
                    if crate::opts::var(crate::opt!("f9-xhash")).as_deref() == Ok("3")
                       && F9X_ATTN.with(|c| c.get()) == 0 {
                        let tag = |buf: &B, wdt: usize, nm: &str| {
                            let host = self.dev.dtoh_sync_copy(buf).unwrap_or_default();
                            let mut hv: u64 = 0xcbf29ce484222325;
                            for i in 0..wdt { let b = host.get(i).map(|v| v.to_bits()).unwrap_or(0); hv ^= b as u64; hv = hv.wrapping_mul(0x100000001b3); }
                            eprintln!("  [f9x-L3-D] {nm} 0x{hv:016x} (b={batch})");
                        };
                        tag(q_ref, q_pitch_d * batch, "q");
                        tag(fb, mtot * batch, "qkv");
                        tag(&at, nh*hd * batch, "attn");
                    }
                    at
                }
            } else {
                // AttnIn::Split weights — no fused buffer to view; packed fused rope + dispatch write.
                let qb = q.as_ref().expect("q buffer");
                let kb = k.as_ref().expect("k buffer");
                let vb = v.as_ref().expect("v buffer");
                blaunch!(self, "rmsnorm_rope_b", ((batch*nh) as u32,1,1), (hd as u32,1,1), (hd*4) as u32, (q_ptr, d(&fa.q_norm), d(cos), d(sin), nh as i32, hd as i32, rdim as i32, batch as i32, eps, 0i32, q_pitch as i32));
                blaunch!(self, "rmsnorm_rope_b", ((batch*nkv) as u32,1,1), (hd as u32,1,1), (hd*4) as u32, (d(kb), d(&fa.k_norm), d(cos), d(sin), nkv as i32, hd as i32, rdim as i32, batch as i32, eps, 0i32, kv_dim as i32));
                {
                    let at = self.attn_dispatch(pool, qb, kb, vb, nh*hd, false, pos_dev, kc_ptr, vc_ptr, slot_ids_ptr, max_pc, kv_stride, batch, prefill_pos_start, 0u64, 0u64, 0u64, sharded, kv_mode, batch == 1 && prefill_pos_start.is_none(), qsa_sel_ptrs);
                    if crate::opts::var(crate::opt!("f9-xhash")).as_deref() == Ok("3")
                       && F9X_ATTN.with(|c| c.get()) == 0 {
                        let tag = |buf: &B, wdt: usize, nm: &str| {
                            let host = self.dev.dtoh_sync_copy(buf).unwrap_or_default();
                            let mut hv: u64 = 0xcbf29ce484222325;
                            for i in 0..wdt { let b = host.get(i).map(|v| v.to_bits()).unwrap_or(0); hv ^= b as u64; hv = hv.wrapping_mul(0x100000001b3); }
                            eprintln!("  [f9x-L3-D] {nm} 0x{hv:016x} (b={batch})");
                        };
                        tag(qb, nh*hd*batch, "q");
                        tag(kb, kv_dim*batch, "k");
                        tag(&at, nh*hd*batch, "attn");
                    }
                    at
                }
            }
        } else {
            // Old pipeline: packed q/k/v; the rope is fused unless the F4 escape is set.
            let qb = q.as_ref().expect("q buffer");
            let kb = k.as_ref().expect("k buffer");
            let vb = v.as_ref().expect("v buffer");
            if crate::opts::var(crate::opt!("no-fused-perhead-rope")).is_ok() {
                blaunch!(self, "rmsnorm_perhead_b", ((batch*nh) as u32,1,1), (hd as u32,1,1), (hd*4) as u32, (d(qb), d(qb), d(&fa.q_norm), nh as i32, hd as i32, batch as i32, eps));
                blaunch!(self, "rmsnorm_perhead_b", ((batch*nkv) as u32,1,1), (hd as u32,1,1), (hd*4) as u32, (d(kb), d(kb), d(&fa.k_norm), nkv as i32, hd as i32, batch as i32, eps));
                blaunch!(self, "rope_b", grid(batch*nh*(rdim/2)), (256,1,1), 0, (d(qb), d(cos), d(sin), nh as i32, hd as i32, rdim as i32, batch as i32));
                blaunch!(self, "rope_b", grid(batch*nkv*(rdim/2)), (256,1,1), 0, (d(kb), d(cos), d(sin), nkv as i32, hd as i32, rdim as i32, batch as i32));
            } else {
                blaunch!(self, "rmsnorm_rope_b", ((batch*nh) as u32,1,1), (hd as u32,1,1), (hd*4) as u32, (q_ptr, d(&fa.q_norm), d(cos), d(sin), nh as i32, hd as i32, rdim as i32, batch as i32, eps, 0i32, q_pitch as i32));
                blaunch!(self, "rmsnorm_rope_b", ((batch*nkv) as u32,1,1), (hd as u32,1,1), (hd*4) as u32, (d(kb), d(&fa.k_norm), d(cos), d(sin), nkv as i32, hd as i32, rdim as i32, batch as i32, eps, 0i32, kv_dim as i32));
            }
            {
                let at = self.attn_dispatch(pool, qb, kb, vb, nh*hd, false, pos_dev, kc_ptr, vc_ptr, slot_ids_ptr, max_pc, kv_stride, batch, prefill_pos_start, 0u64, 0u64, 0u64, sharded, kv_mode, batch == 1 && prefill_pos_start.is_none(), qsa_sel_ptrs);
                if crate::opts::var(crate::opt!("f9-xhash")).as_deref() == Ok("3")
                   && F9X_ATTN.with(|c| c.get()) == 0 {
                    let tag = |buf: &B, wdt: usize, nm: &str| {
                        let host = self.dev.dtoh_sync_copy(buf).unwrap_or_default();
                        let mut hv: u64 = 0xcbf29ce484222325;
                        for i in 0..wdt { let b = host.get(i).map(|v| v.to_bits()).unwrap_or(0); hv ^= b as u64; hv = hv.wrapping_mul(0x100000001b3); }
                        eprintln!("  [f9x-L3-D] {nm} 0x{hv:016x} (b={batch})", );
                    };
                    tag(qb, nh*hd*batch, "q");
                    tag(kb, kv_dim*batch, "k");
                    tag(&at, nh*hd*batch, "attn");
                }
                at
            }
        };
        if crate::opts::var(crate::opt!("cap-debug")).is_ok() {
            // Cheap absmax probes to localize a numeric blow-up (CPU reference deltas). Under F0 the
            // q/k/v probes read the fused buffer's segments (q/k/v are views); q is packed for qwen.
            let absmax = |s: &B| { self.sync_stream(); self.dev.dtoh_sync_copy(s).unwrap().iter()
                .fold(0f32, |a, x| a.max(x.to_f32().abs())) };
            let seg = |f: &B| { let fv: Vec<f32> = self.dev.dtoh_sync_copy(f).unwrap().iter().map(|x| x.to_f32()).collect();
                move |off: usize, wid: usize| fv.chunks(mtot).take(batch).flat_map(|r| r[off..off+wid].iter().copied()).fold(0f32, f32::max) };
            let qmx = match &q { Some(qb) => absmax(qb), None => seg(fused_qkv.as_ref().expect("fused"))(0, nh*hd) };
            let kmx = match &fused_qkv { Some(f) => seg(f)(qg_dim, kv_dim), None => absmax(k.as_ref().expect("k")) };
            let vmx = match &fused_qkv { Some(f) => seg(f)(qg_dim + kv_dim, kv_dim), None => absmax(v.as_ref().expect("v")) };
            eprintln!("  capdbg attn: |q|={:.4} |k|={:.4} |v|={:.4} |attn|={:.4}", qmx, kmx, vmx, absmax(&attn));
        }
        if let Some((a, b)) = qsa_sel { self.qsa_release(pool, a, b, batch); }
        if let Some(g) = &gate {
            blaunch!(self, "sigmoid_gate_b", grid(nh*hd*batch), (256,1,1), 0, (d(&attn), d(g), (nh*hd*batch) as i32));
        }
        let mut out = pool.get_bf16(h*self.pf8_pad(batch));
        // AR landing 2: with `fuse` the row-parallel o_proj reduce may stop after K1 and hand the
        // LOCAL partial up for the fused mixer epilogue (single-barrier only).
        let fused = if sharded {
            self.tp_reduce_site(pool, &fa.o_proj, &attn, &mut out, nh*hd, h, batch, fuse)
        } else {
            self.gemm_act(&fa.o_proj, &attn, &mut out, nh*hd, h, batch);
            None
        };
        // F0: under the default q/k/v are views into `fused` — one release; packed otherwise.
        if let Some(f) = fused_qkv { pool.release_bf16(f, mtot * batch); }
        if let Some(kb) = k { pool.release_bf16(kb, kv_dim*batch); }
        if let Some(vb) = v { pool.release_bf16(vb, kv_dim*batch); }
        if let Some(qb) = q { pool.release_bf16(qb, nh*hd*batch); }
        pool.release_bf16(attn, nh*hd*batch);
        if let Some(g) = gate { pool.release_bf16(g, nh*hd*batch); }
        match fused {
            Some(FusedPartial::F32(p)) => ReduceOut::Fused(FusedReduce { sink: out, partial: Some(p), mode: 1 }),
            Some(FusedPartial::F32Scratch) => ReduceOut::Fused(FusedReduce { sink: out, partial: None, mode: 1 }),
            Some(FusedPartial::Bf16) => ReduceOut::Fused(FusedReduce { sink: out, partial: None, mode: 0 }),
            None => ReduceOut::Full(out),
        }
    }
}
