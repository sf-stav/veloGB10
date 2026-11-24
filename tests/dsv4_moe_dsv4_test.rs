//! Phase 2 / Gate G2 (LANE 2B): the DSV4 routed-expert MoE pipeline on the engine's NVFP4
//! machinery, gated against the G1-proven CPU reference on REAL layer weights.
//!
//! Pipeline (src/gpu.rs DSV4 section): act-sim x (§C.1 g128 UE8M0, in place) → NVFP4
//! gate_up GEMM → `dsv4_swiglu_clamp` (§B.9 asymmetric clamps + routing weight folded
//! before w2) → act-sim h → NVFP4 down GEMM → slot-order fp32 combine (weight already
//! folded, so the combine multiplies by exact 1.0f).
//!
//! Gates:
//!   1. `dsv4_swiglu_clamp{,_shared}` unit: elementwise vs an exact-order CPU ref
//!      (≤1 bf16 ulp, libdevice expf vs glibc exp — the Lane-D sinkhorn situation).
//!   2. Real weights (load_layer(0), load_layer(2) — both HASH-routed layers; the pipeline
//!      consumes ids/wts directly, so hash vs sqrtsoftplus routing is irrelevant here —
//!      the router is Phase 3): GPU vs `dsv4_cpu::expert_forward_token` accumulated in
//!      fp32 slot order (moe_forward's accumulator semantics; its expert-ascending merge
//!      differs by reorder-class only, per the G1 amendments) — rel-L2 bar measured, ≤2e-3.
//!   3. Engine contract: N=1 (decode) vs grouped N=2..16 (verify) outputs BITWISE-IDENTICAL
//!      (same MMA k-order per element, zero-filled padding, slot-order fp32 combine).
//!
//! Run: cargo test --release --test dsv4_moe_dsv4_test -- --nocapture
//! The heavyweight test streams real checkpoint shards from
//! /mnt/models/DeepSeek-V4-Flash-DSpark (READ-ONLY) and serializes on a static gate.

use cudarc::driver::{CudaDevice, CudaFunction, DevicePtr, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use gb10_inference::{dsv4_cpu, dsv4_load, dsv4_moe, gpu, quant};
use half::bf16;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

const BUNDLE: &str = "/mnt/models/DeepSeek-V4-Flash-DSpark";

/// One GPU job per process (tests run on threads; the GPU is serialized across lanes too).
static GATE: Mutex<()> = Mutex::new(());

fn gate() -> MutexGuard<'static, ()> {
    GATE.lock().unwrap_or_else(|e| e.into_inner())
}

fn bundle() -> PathBuf {
    PathBuf::from(BUNDLE)
}

// ---------------------------------------------------------------------------
// exact conversion helpers (RNE, matching cvt.rn / __float2bfloat16)
// ---------------------------------------------------------------------------

fn f32_to_bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return 0x7FC0;
    }
    let lsb = (b >> 16) & 1;
    let rounded = b.wrapping_add(0x7FFF + lsb as u32);
    (rounded >> 16) as u16
}

fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn f32(&mut self) -> f32 {
        // uniform in [-1, 1)
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

// ---------------------------------------------------------------------------
// module loading
// ---------------------------------------------------------------------------

const BATCH_FUNCS: [&str; 13] = [
    "gemm_moe_mma_fp4",
    "gemm_moe_mma_fp4_x2",
    "gemm_moe_mma_fp4_u4",
    "gemm_moe_grouped_mma_fp4",
    "gemm_moe_grouped_mma_fp4_x2",
    "gemm_moe_grouped_mma_fp4_u4",
    "moe_count_b",
    "moe_offsets_b",
    "moe_scatter_b",
    "moe_tilemap_b",
    "moe_gather_x_b",
    "moe_combine_experts_b",
    "moe_combine_grouped_b",
];
const DSV4_FUNCS: [&str; 3] = [
    "dsv4_act_quant_sim_g128",
    "dsv4_swiglu_clamp",
    "dsv4_swiglu_clamp_shared",
];

fn load_modules(dev: &Arc<CudaDevice>) -> (HashMap<String, CudaFunction>, HashMap<String, CudaFunction>) {
    let bptx = Ptx::from_src(
        std::fs::read_to_string("src/ptx/gpu_batch.ptx").expect("src/ptx/gpu_batch.ptx (cargo build first)"),
    );
    dev.load_ptx(bptx, "gpu_batch", &BATCH_FUNCS).expect("load_ptx gpu_batch");
    let dptx = Ptx::from_src(
        std::fs::read_to_string("src/ptx/gpu_dsv4.ptx").expect("src/ptx/gpu_dsv4.ptx (cargo build first)"),
    );
    dev.load_ptx(dptx, "gpu_dsv4", &DSV4_FUNCS).expect("load_ptx gpu_dsv4");
    let collect = |module: &str, names: &[&str]| {
        names
            .iter()
            .map(|n| (n.to_string(), dev.get_func(module, n).unwrap_or_else(|| panic!("missing {n}"))))
            .collect()
    };
    (collect("gpu_batch", &BATCH_FUNCS), collect("gpu_dsv4", &DSV4_FUNCS))
}

// ---------------------------------------------------------------------------
// CPU references
// ---------------------------------------------------------------------------

/// §B.9 SwiGLU elementwise, exact op order of dsv4_cpu::expert_forward_token.
fn cpu_swiglu_clamp(gu: &[u16], rw: Option<&[f32]>, i: usize, bk: usize, limit: f32) -> Vec<u16> {
    let mut out = vec![0u16; bk * i];
    for row in 0..bk {
        for r in 0..i {
            let g0 = bf16_bits_to_f32(gu[row * 2 * i + r]);
            let u0 = bf16_bits_to_f32(gu[row * 2 * i + i + r]);
            let u = u0.clamp(-limit, limit);
            let g = g0.min(limit);
            let sg = 1.0f32 / (1.0f32 + (-g).exp());
            let mut v = (g * sg) * u;
            if let Some(w) = rw {
                v *= w[row];
            }
            out[row * i + r] = f32_to_bf16_bits(v);
        }
    }
    out
}

/// Routed-experts-only MoE forward (§B.9): per (token, slot) `expert_forward_token`
/// (the G1-proven expert math, inner_block 32 for FP4), fp32 accumulator in slot order,
/// final bf16 round — `moe_forward`'s accumulator semantics with a slot-order merge
/// (its expert-ascending merge is a reorder-class difference, G1-amendment tolerance class).
#[allow(clippy::too_many_arguments)]
fn cpu_moe_routed(
    bank: &dsv4_cpu::ExpertBank,
    x: &[f32], // [batch, dim] bf16-valued f32 (act-quant happens inside quant_gemm)
    batch: usize,
    dim: usize,
    inter: usize,
    ids: &[i32],
    wts: &[f32],
    topk: usize,
    limit: f32,
) -> Vec<f32> {
    let mut y = vec![0.0f32; batch * dim];
    for t in 0..batch {
        for j in 0..topk {
            let e = ids[t * topk + j] as usize;
            let exp = bank.get(e);
            let yt = dsv4_cpu::expert_forward_token(
                &x[t * dim..(t + 1) * dim],
                &exp.w1,
                &exp.w2,
                &exp.w3,
                dim,
                inter,
                32,
                limit,
                Some(wts[t * topk + j]),
            );
            for d in 0..dim {
                y[t * dim + d] += yt[d];
            }
        }
    }
    dsv4_cpu::round_bf16(&mut y);
    y
}

fn rel_l2(a: &[u16], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        let d = bf16_bits_to_f32(*x) as f64 - *y as f64;
        num += d * d;
        den += (*y as f64) * (*y as f64);
    }
    (num / den).sqrt()
}

/// Seeded routing: 6 distinct experts per token (partial Fisher–Yates), weights U(0.05,1)
/// renormalized to sum route_scale (= gate_forward's renorm × 1.5 semantics).
fn gen_routing(
    rng: &mut XorShift,
    batch: usize,
    ne: usize,
    topk: usize,
    route_scale: f32,
) -> (Vec<i32>, Vec<f32>) {
    let mut ids = vec![0i32; batch * topk];
    let mut wts = vec![0.0f32; batch * topk];
    for t in 0..batch {
        let mut perm: Vec<i32> = (0..ne as i32).collect();
        for j in 0..topk {
            let r = j + (rng.next() as usize % (ne - j));
            perm.swap(j, r);
            ids[t * topk + j] = perm[j];
        }
        let mut s = 0.0f32;
        for j in 0..topk {
            let w = 0.05 + 0.95 * (rng.f32() * 0.5 + 0.5);
            wts[t * topk + j] = w;
            s += w;
        }
        for j in 0..topk {
            wts[t * topk + j] = wts[t * topk + j] / s * route_scale;
        }
    }
    (ids, wts)
}

/// Seeded synthetic hidden states (post-RMSNorm scale), bf16-valued.
fn gen_x(rng: &mut XorShift, batch: usize, dim: usize) -> Vec<u16> {
    (0..batch * dim).map(|_| f32_to_bf16_bits(rng.f32() * 1.5)).collect()
}

fn x_as_f32(x: &[u16]) -> Vec<f32> {
    x.iter().map(|&b| bf16_bits_to_f32(b)).collect()
}

fn to_bf16(bits: &[u16]) -> Vec<bf16> {
    bits.iter().map(|&b| bf16::from_bits(b)).collect()
}

fn to_bits(v: &[bf16]) -> Vec<u16> {
    v.iter().map(|x| x.to_bits()).collect()
}

// ---------------------------------------------------------------------------
// 0. pack_moe_layer layout gate (host-only, no GPU, no bundle): per-expert
//    fuse+repack concatenation must be byte-identical to the naive full-stack
//    fuse+repack, with gs tiles in (expert, w1-then-w3) order.
// ---------------------------------------------------------------------------

fn fake_nvfp4(m: usize, k: usize, fill: u8, global_scale: f32) -> quant::Nvfp4Tensor {
    quant::Nvfp4Tensor {
        qweight: (0..m * k / 2).map(|i| fill ^ (i as u8)).collect(),
        scales: (0..m * k / 16).map(|i| fill.wrapping_add(i as u8)).collect(),
        global_scale,
        m,
        k,
    }
}

#[test]
fn dsv4_moe_pack_layout() {
    let cfg = dsv4_load::Dsv4Config {
        vocab_size: 0,
        dim: 64,
        moe_inter_dim: 32,
        n_layers: 0,
        n_hash_layers: 0,
        n_mtp_layers: 0,
        dspark_block_size: 0,
        dspark_noise_token_id: 0,
        dspark_target_layer_ids: vec![],
        dspark_markov_rank: 0,
        n_heads: 0,
        n_routed_experts: 2,
        n_shared_experts: 0,
        n_activated_experts: 0,
        route_scale: 1.0,
        swiglu_limit: 10.0,
        q_lora_rank: 0,
        head_dim: 0,
        rope_head_dim: 0,
        o_groups: 0,
        o_lora_rank: 0,
        window_size: 0,
        original_seq_len: 0,
        rope_theta: 0.0,
        rope_factor: 0.0,
        beta_fast: 0,
        beta_slow: 0,
        index_n_heads: 0,
        index_head_dim: 0,
        index_topk: 0,
        hc_mult: 0,
        hc_sinkhorn_iters: 0,
        compress_rope_theta: 0.0,
        compress_ratios: vec![],
        norm_eps: 0.0,
        hc_eps: 0.0,
    };
    let (h, inter) = (cfg.dim, cfg.moe_inter_dim);
    let mk = |e: usize| {
        (
            fake_nvfp4(inter, h, 0x10 * (e as u8 + 1), 2f32.powi((4 * e + 1) as i32)), // w1
            fake_nvfp4(h, inter, 0x40 * (e as u8 + 1), 2f32.powi((4 * e + 3) as i32)), // w2
            fake_nvfp4(inter, h, 0x80 * (e as u8 + 1), 2f32.powi((4 * e + 2) as i32)), // w3
        )
    };
    let (e0w1, e0w2, e0w3) = mk(0);
    let (e1w1, e1w2, e1w3) = mk(1);
    let layer = dsv4_load::Dsv4Layer {
        tensors: HashMap::new(),
        experts_w1: vec![e0w1.clone(), e1w1.clone()],
        experts_w2: vec![e0w2.clone(), e1w2.clone()],
        experts_w3: vec![e0w3.clone(), e1w3.clone()],
    };
    let got = dsv4_moe::pack_moe_layer(&layer, &cfg).expect("pack_moe_layer");

    // naive full-stack reference: one fuse over all parts, one repack
    let (qw, sc, gs) = quant::fuse_nvfp4(
        &[
            (&e0w1.qweight, &e0w1.scales, 1.0 / e0w1.global_scale, inter),
            (&e0w3.qweight, &e0w3.scales, 1.0 / e0w3.global_scale, inter),
            (&e1w1.qweight, &e1w1.scales, 1.0 / e1w1.global_scale, inter),
            (&e1w3.qweight, &e1w3.scales, 1.0 / e1w3.global_scale, inter),
        ],
        h,
    );
    let (wt, st) = quant::repack_nvfp4_mma(&qw, &sc, 2 * 2 * inter, h);
    assert_eq!(got.gu_wt, wt, "gate_up weight bytes != full-stack repack");
    assert_eq!(got.gu_st, st, "gate_up scale bytes != full-stack repack");
    assert_eq!(got.gu_gs, gs, "gate_up per-tile gs order != (expert, w1, w3)");
    let (qw, sc, gs) = quant::fuse_nvfp4(
        &[
            (&e0w2.qweight, &e0w2.scales, 1.0 / e0w2.global_scale, h),
            (&e1w2.qweight, &e1w2.scales, 1.0 / e1w2.global_scale, h),
        ],
        inter,
    );
    let (wt, st) = quant::repack_nvfp4_mma(&qw, &sc, 2 * h, inter);
    assert_eq!(got.dn_wt, wt, "down weight bytes != full-stack repack");
    assert_eq!(got.dn_st, st, "down scale bytes != full-stack repack");
    assert_eq!(got.dn_gs, gs, "down per-tile gs order != expert order");
    println!("pack layout: per-expert concat == full-stack fuse+repack, gs order OK");
}

// ---------------------------------------------------------------------------
// 1. dsv4_swiglu_clamp unit gate (synthetic; clamps actually engaged)
// ---------------------------------------------------------------------------

#[test]
fn dsv4_swiglu_clamp_unit() {
    let _g = gate();
    let dev = CudaDevice::new(0).expect("CUDA device 0");
    let (_bk, df) = load_modules(&dev);

    let (bk_rows, i) = (37usize, 2048usize);
    let limit = 10.0f32;
    let mut rng = XorShift(0xD5E4_2024_0713);
    // gate ~ [-14, 14] (upper clamp engages, lower must NOT), up ~ [-16, 16] (both engage)
    let mut gu = vec![0u16; bk_rows * 2 * i];
    for row in 0..bk_rows {
        for r in 0..i {
            gu[row * 2 * i + r] = f32_to_bf16_bits(rng.f32() * 14.0);
            gu[row * 2 * i + i + r] = f32_to_bf16_bits(rng.f32() * 16.0);
        }
    }
    let rw: Vec<f32> = (0..bk_rows).map(|_| 0.05 + 0.7 * (rng.f32() * 0.5 + 0.5)).collect();

    let gu_d = dev.htod_sync_copy(&to_bf16(&gu)).unwrap();
    let rw_d = dev.htod_sync_copy(&rw).unwrap();
    let mut h_d = dev.alloc_zeros::<bf16>(bk_rows * i).unwrap();
    let mut hs_d = dev.alloc_zeros::<bf16>(bk_rows * i).unwrap();
    let grid = ((bk_rows * i + 255) / 256) as u32;
    unsafe {
        df.get("dsv4_swiglu_clamp")
            .unwrap()
            .clone()
            .launch(
                LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                (&mut h_d, &gu_d, &rw_d, limit, i as i32, bk_rows as i32),
            )
            .unwrap();
        df.get("dsv4_swiglu_clamp_shared")
            .unwrap()
            .clone()
            .launch(
                LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                (&mut hs_d, &gu_d, limit, i as i32, bk_rows as i32),
            )
            .unwrap();
    }
    dev.synchronize().unwrap();
    let got = to_bits(&dev.dtoh_sync_copy(&h_d).unwrap());
    let got_s = to_bits(&dev.dtoh_sync_copy(&hs_d).unwrap());
    let want = cpu_swiglu_clamp(&gu, Some(&rw), i, bk_rows, limit);
    let want_s = cpu_swiglu_clamp(&gu, None, i, bk_rows, limit);

    // bf16-output comparison: identical, or adjacent pattern (libdevice expf vs glibc exp
    // is ≤1 ulp in the sigmoid — it can flip the bf16 RNE on exact ties only).
    let check = |got: &[u16], want: &[u16], name: &str| {
        let mut exact = 0usize;
        let mut bad = 0usize;
        for (g, w) in got.iter().zip(want.iter()) {
            if g == w {
                exact += 1;
            } else {
                let d = (bf16_bits_to_f32(*g) - bf16_bits_to_f32(*w)).abs();
                let m = bf16_bits_to_f32(*w).abs().max(1e-6);
                if d / m > 0.008 {
                    bad += 1; // more than ~1 bf16 ulp (2^-7 relative)
                }
            }
        }
        let n = got.len();
        println!(
            "swiglu {name}: {exact}/{n} bit-exact ({:.2}%), >1ulp mismatches {bad}",
            exact as f64 / n as f64 * 100.0
        );
        assert!(exact as f64 / n as f64 > 0.99, "{name}: <99% bit-exact");
        assert_eq!(bad, 0, "{name}: mismatch beyond 1 bf16 ulp");
    };
    check(&got, &want, "routed");
    check(&got_s, &want_s, "shared");

    // clamp semantics spot checks: gate=-20 must NOT clamp low; gate=+20 clamps to 10;
    // up=±20 clamps to ±10.
    let g_lo = f32_to_bf16_bits(-20.0);
    let g_hi = f32_to_bf16_bits(20.0);
    let u_hi = f32_to_bf16_bits(20.0);
    let u_lo = f32_to_bf16_bits(-20.0);
    let one = f32_to_bf16_bits(1.0);
    let cases = [
        ([g_lo, one], (1.0f32 / (1.0f32 + 20.0f32.exp())) * -20.0 * 1.0),  // silu(-20)·1, no low clamp
        ([g_hi, one], 10.0 * 1.0),                                          // silu(10)·1 ≈ 10
        ([one, u_hi], (1.0 / (1.0 + (-1.0f32).exp())) * 10.0),              // up → +10
        ([one, u_lo], (1.0 / (1.0 + (-1.0f32).exp())) * -10.0),             // up → −10
    ];
    for ([g2, u2], want_v) in cases {
        let gu2 = vec![bf16::from_bits(g2), bf16::from_bits(u2)];
        let gu2_d = dev.htod_sync_copy(&gu2).unwrap();
        let rw1 = vec![1.0f32];
        let rw1_d = dev.htod_sync_copy(&rw1).unwrap();
        let mut h2_d = dev.alloc_zeros::<bf16>(1).unwrap();
        unsafe {
            df.get("dsv4_swiglu_clamp")
                .unwrap()
                .clone()
                .launch(
                    LaunchConfig { grid_dim: (1, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                    (&mut h2_d, &gu2_d, &rw1_d, limit, 1i32, 1i32),
                )
                .unwrap();
        }
        dev.synchronize().unwrap();
        let got_v = dev.dtoh_sync_copy(&h2_d).unwrap()[0].to_f32();
        assert!(
            (got_v - want_v).abs() / want_v.abs().max(1e-6) < 0.01,
            "clamp case g={g2:#06x} u={u2:#06x}: got {got_v}, want {want_v}"
        );
    }
    println!("swiglu unit: routed+shared variants OK, asymmetric clamps OK");
}

// ---------------------------------------------------------------------------
// 2/3. Real-weights pipeline gate: CPU-ref rel-L2 + bitwise N-invariance
// ---------------------------------------------------------------------------

#[test]
fn dsv4_moe_pipeline_real_weights() {
    if !bundle().exists() { return; } // lab-model-bundle gate
    let _g = gate();
    let dev = CudaDevice::new(0).expect("CUDA device 0");
    let (bk, df) = load_modules(&dev);
    let cfg = dsv4_load::load_config(&bundle()).expect("load_config");
    let (ne, topk, dim, inter) = (cfg.n_routed_experts, cfg.n_activated_experts, cfg.dim, cfg.moe_inter_dim);
    assert_eq!((dim, inter, ne, topk), (4096, 2048, 256, 6), "§D real geometry");
    let limit = cfg.swiglu_limit; // 10.0

    for layer_idx in [0usize, 2usize] {
        // Both hash-routed layers (0..2 route by tid2eid table, §B.9.4): this pipeline is
        // routing-agnostic (ids/wts in), so hash vs sqrtsoftplus selection cannot matter here.
        println!("== layer {layer_idx}: loading (streams real shards, ~1 min) ==");
        let layer = dsv4_load::load_layer(&bundle(), &cfg, layer_idx).expect("load_layer");
        let host = dsv4_moe::pack_moe_layer(&layer, &cfg).expect("pack_moe_layer");
        let gm = gpu::Dsv4MoeGpu::upload(&dev, &host).expect("upload");
        drop(host);
        let bank = dsv4_cpu::ExpertBank::from_nvfp4(layer.experts_w1, layer.experts_w2, layer.experts_w3);
        let mut scratch = gpu::new_moe_grouped_scratch_raw(&dev, ne, dim, inter, topk, 16, 16 * topk);
        let stream = gb10_inference::dsv4_gpu::blocking_compute_stream(&dev);

        // --- (2) CPU-reference diff at batch 8 ---
        let batch = 8usize;
        let mut rng = XorShift(0xD5E4_0000_0000 + layer_idx as u64);
        let x_bits = gen_x(&mut rng, batch, dim);
        let (ids, wts) = gen_routing(&mut rng, batch, ne, topk, cfg.route_scale);
        let y_ref = cpu_moe_routed(&bank, &x_as_f32(&x_bits), batch, dim, inter, &ids, &wts, topk, limit);

        // R3.1: the test drives the SERVING-path `_ws` helpers (compute stream + persistent
        // workspace) — same kernels/launch configs as the Phase-2 reference helpers, so the
        // bitwise contract below gates exactly what serving runs. `x2` selects the x2-tile
        // streaming variants (queue #3); per-element K chains are unchanged by construction,
        // and the cross-variant equivalence is gated by `dsv4_moe_x2_bitwise_vs_original`.
        let mut run = |x2: bool, grouped: bool, batch: usize, x_bits: &[u16], ids: &[i32], wts: &[f32]| -> Vec<u16> {
            let x_d = dev.htod_sync_copy(&to_bf16(x_bits)).unwrap();
            let ids_d = dev.htod_sync_copy(ids).unwrap();
            let wts_d = dev.htod_sync_copy(wts).unwrap();
            {
                let ws = scratch.dsv4_ws_mut();
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(*ws.xc.device_ptr(), *x_d.device_ptr(), batch * dim * 2, stream.stream).unwrap();
                    cudarc::driver::result::memcpy_dtod_async(*ws.idc.device_ptr(), *ids_d.device_ptr(), batch * topk * 4, stream.stream).unwrap();
                    cudarc::driver::result::memcpy_dtod_async(*ws.wtc.device_ptr(), *wts_d.device_ptr(), batch * topk * 4, stream.stream).unwrap();
                }
            }
            if grouped {
                if x2 {
                    gpu::dsv4_moe_experts_grouped_ws_x2(
                        &dev, &stream, &bk, &df, &gm, &mut scratch, batch, topk, limit,
                    )
                    .unwrap();
                } else {
                    gpu::dsv4_moe_experts_grouped_ws(
                        &dev, &stream, &bk, &df, &gm, &mut scratch, batch, topk, limit,
                    )
                    .unwrap();
                }
            } else {
                if x2 {
                    gpu::dsv4_moe_experts_n1_ws_x2(
                        &dev, &stream, &bk, &df, &gm, scratch.dsv4_ws_mut(), batch, topk, limit,
                    )
                    .unwrap();
                } else {
                    gpu::dsv4_moe_experts_n1_ws(
                        &dev, &stream, &bk, &df, &gm, scratch.dsv4_ws_mut(), batch, topk, limit,
                    )
                    .unwrap();
                }
            }
            dev.synchronize().unwrap();
            let full = dev.dtoh_sync_copy(&scratch.dsv4_ws_ref().outc).unwrap();
            to_bits(&full[..batch * dim])
        };

        let out_n1 = run(false, false, batch, &x_bits, &ids, &wts);
        let out_g = run(false, true, batch, &x_bits, &ids, &wts);
        let e_n1 = rel_l2(&out_n1, &y_ref);
        let e_g = rel_l2(&out_g, &y_ref);
        let bit_eq = out_n1.iter().zip(&y_ref).filter(|(a, b)| {
            let d = (bf16_bits_to_f32(**a) - **b).abs();
            d == 0.0
        }).count();
        println!(
            "layer {layer_idx} batch {batch}: rel-L2 n1={e_n1:.3e} grouped={e_g:.3e}; \
             n1-vs-ref bitwise-equal {bit_eq}/{}",
            batch * dim
        );
        // Bar: single-MoE-forward diffs are GEMM-reorder + ≤1ulp sigmoid class only (the
        // act-sim and weights are bit-exact by G1/Lane-D proofs), far below the full-layer
        // G1 floors (1.24e-2). 2e-3 carries ~4× margin over the measured values.
        assert!(e_n1 < 2e-3, "layer {layer_idx} n1 rel-L2 {e_n1:.3e} >= 2e-3");
        assert!(e_g < 2e-3, "layer {layer_idx} grouped rel-L2 {e_g:.3e} >= 2e-3");

        // --- (3) N-invariance: N=1 vs grouped N=2..16, to_bits identical (serving path) ---
        for n in 2..=16usize {
            let mut rng_n = XorShift(0xD5E4_1000_0000 + (layer_idx as u64) * 100 + n as u64);
            let xn = gen_x(&mut rng_n, n, dim);
            let (idn, wtn) = gen_routing(&mut rng_n, n, ne, topk, cfg.route_scale);
            let a = run(false, false, n, &xn, &idn, &wtn);
            let b = run(false, true, n, &xn, &idn, &wtn);
            assert_eq!(a, b, "layer {layer_idx}: N=1 vs grouped N={n} NOT bitwise-identical");
        }
        println!("layer {layer_idx}: N=1 vs grouped bitwise-identical for every N in 2..=16 ✓");
        drop(bank);
        drop(gm);
    }
    println!("DSV4 MoE pipeline gate: PASS");
}

/// R3A.4 gate: ONE wide dispatch (batch up to the full prefill-chunk width) must be
/// bitwise-identical to the old 16-row-chunked decomposition of the same inputs — the
/// grouped GEMM's per-element K-order is ppad-independent by construction; this is the proof.
#[test]
fn wide_dispatch_bitwise_vs_chunked() {
    let _g = gate();
    if !bundle().exists() {
        eprintln!("SKIP wide_dispatch_bitwise_vs_chunked: bundle not mounted");
        return;
    }
    let dev = CudaDevice::new(0).unwrap();
    let (bk, df) = load_modules(&dev);
    let cfg = dsv4_load::load_config(&bundle()).unwrap();
    let (dim, inter, ne, topk, limit) = (
        cfg.dim,
        cfg.moe_inter_dim,
        cfg.n_routed_experts,
        cfg.n_activated_experts,
        cfg.swiglu_limit,
    );
    let layer_idx = 3usize; // first MoE trunk layer (matches the pipeline gate's class)
    let layer = dsv4_load::load_layer(&bundle(), &cfg, layer_idx).expect("load_layer");
    let host = dsv4_moe::pack_moe_layer(&layer, &cfg).expect("pack_moe_layer");
    let gm = gpu::Dsv4MoeGpu::upload(&dev, &host).expect("upload");
    drop(host);
    let stream = gb10_inference::dsv4_gpu::blocking_compute_stream(&dev);

    for batch in [96usize, 130, 2048] {
        let mut scratch = gpu::new_moe_grouped_scratch_raw(&dev, ne, dim, inter, topk, batch, batch * topk);
        let mut rng = XorShift(0xD5E4_2000_0000 + batch as u64);
        let x_bits = gen_x(&mut rng, batch, dim);
        let (ids, wts) = gen_routing(&mut rng, batch, ne, topk, cfg.route_scale);
        let x_d = dev.htod_sync_copy(&to_bf16(&x_bits)).unwrap();
        let ids_d = dev.htod_sync_copy(&ids).unwrap();
        let wts_d = dev.htod_sync_copy(&wts).unwrap();

        // (a) ONE wide dispatch (the R3A.4 production path).
        {
            let ws = scratch.dsv4_ws_mut();
            unsafe {
                cudarc::driver::result::memcpy_dtod_async(*ws.xc.device_ptr(), *x_d.device_ptr(), batch * dim * 2, stream.stream).unwrap();
                cudarc::driver::result::memcpy_dtod_async(*ws.idc.device_ptr(), *ids_d.device_ptr(), batch * topk * 4, stream.stream).unwrap();
                cudarc::driver::result::memcpy_dtod_async(*ws.wtc.device_ptr(), *wts_d.device_ptr(), batch * topk * 4, stream.stream).unwrap();
            }
        }
        gpu::dsv4_moe_experts_grouped_ws(&dev, &stream, &bk, &df, &gm, &mut scratch, batch, topk, limit).unwrap();
        dev.synchronize().unwrap();
        let wide = dev.dtoh_sync_copy(&scratch.dsv4_ws_ref().outc).unwrap();
        let wide = to_bits(&wide[..batch * dim]);

        // (b) the old decomposition: sequential 16-row dispatches (stage slice, dispatch,
        // fold into outc[r0*dim..]), final buffer compared whole.
        let mut chunked_full = dev.alloc_zeros::<bf16>(batch * dim).unwrap();
        let mut r0 = 0usize;
        while r0 < batch {
            let n = (batch - r0).min(16);
            {
                let ws = scratch.dsv4_ws_mut();
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(*ws.xc.device_ptr(), *x_d.device_ptr() + (r0 * dim * 2) as u64, n * dim * 2, stream.stream).unwrap();
                    cudarc::driver::result::memcpy_dtod_async(*ws.idc.device_ptr(), *ids_d.device_ptr() + (r0 * topk * 4) as u64, n * topk * 4, stream.stream).unwrap();
                    cudarc::driver::result::memcpy_dtod_async(*ws.wtc.device_ptr(), *wts_d.device_ptr() + (r0 * topk * 4) as u64, n * topk * 4, stream.stream).unwrap();
                }
            }
            gpu::dsv4_moe_experts_grouped_ws(&dev, &stream, &bk, &df, &gm, &mut scratch, n, topk, limit).unwrap();
            {
                let ws = scratch.dsv4_ws_mut();
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(*chunked_full.device_ptr() + (r0 * dim * 2) as u64, *ws.outc.device_ptr(), n * dim * 2, stream.stream).unwrap();
                }
            }
            r0 += n;
        }
        dev.synchronize().unwrap();
        let chunked = to_bits(&dev.dtoh_sync_copy(&chunked_full).unwrap());

        let mism = wide.iter().zip(&chunked).filter(|(a, b)| a != b).count();
        println!("batch {batch}: wide dispatch vs 16-row-chunked mismatches {mism} / {}", batch * dim);
        assert_eq!(mism, 0, "batch {batch}: wide dispatch NOT bitwise-identical to the 16-row decomposition");
    }
    println!("DSV4 MoE wide-dispatch gate: PASS");
}

// ---------------------------------------------------------------------------
// Queue #3 gate: the streaming variants (x2-tile: two 16-row weight tiles per CTA sharing X;
// u4: four pair-iterations' loads hoisted) must be BITWISE-IDENTICAL to the single-tile kernels
// at every serving width (they share the per-element K chains and warp-order epilogue reduction;
// only the block→tile map and the instruction schedule differ). Plus an isolated wall-clock A/B.
// ---------------------------------------------------------------------------

/// MoeKernelVariant — which GEMM kernel the dispatch helpers use.
#[derive(Clone, Copy, PartialEq)]
enum MoeVariant { Single, X2, U4 }

fn dispatch(
    dev: &Arc<CudaDevice>,
    stream: &cudarc::driver::CudaStream,
    bk: &HashMap<String, CudaFunction>,
    df: &HashMap<String, CudaFunction>,
    gm: &gpu::Dsv4MoeGpu,
    scratch: &mut gpu::MoeGroupedScratch,
    grouped: bool,
    batch: usize,
    topk: usize,
    limit: f32,
    v: MoeVariant,
) -> anyhow::Result<()> {
    match (grouped, v) {
        (true, MoeVariant::Single) => gpu::dsv4_moe_experts_grouped_ws(dev, stream, bk, df, gm, scratch, batch, topk, limit),
        (true, MoeVariant::X2) => gpu::dsv4_moe_experts_grouped_ws_x2(dev, stream, bk, df, gm, scratch, batch, topk, limit),
        (true, MoeVariant::U4) => gpu::dsv4_moe_experts_grouped_ws_u4(dev, stream, bk, df, gm, scratch, batch, topk, limit),
        (false, MoeVariant::Single) => gpu::dsv4_moe_experts_n1_ws(dev, stream, bk, df, gm, scratch.dsv4_ws_mut(), batch, topk, limit),
        (false, MoeVariant::X2) => gpu::dsv4_moe_experts_n1_ws_x2(dev, stream, bk, df, gm, scratch.dsv4_ws_mut(), batch, topk, limit),
        (false, MoeVariant::U4) => gpu::dsv4_moe_experts_n1_ws_u4(dev, stream, bk, df, gm, scratch.dsv4_ws_mut(), batch, topk, limit),
    }
}

#[test]
fn dsv4_moe_streaming_bitwise_vs_original() {
    if !bundle().exists() { return; } // lab-model-bundle gate
    let _g = gate();
    let dev = CudaDevice::new(0).expect("CUDA device 0");
    let (bk, df) = load_modules(&dev);
    let cfg = dsv4_load::load_config(&bundle()).expect("load_config");
    let (ne, topk, dim, inter) = (cfg.n_routed_experts, cfg.n_activated_experts, cfg.dim, cfg.moe_inter_dim);
    assert_eq!((dim, inter, ne, topk), (4096, 2048, 256, 6), "§D real geometry");
    let limit = cfg.swiglu_limit;

    let layer = dsv4_load::load_layer(&bundle(), &cfg, 0).expect("load_layer");
    let host = dsv4_moe::pack_moe_layer(&layer, &cfg).expect("pack_moe_layer");
    let gm = gpu::Dsv4MoeGpu::upload(&dev, &host).expect("upload");
    drop(host);
    let mut scratch = gpu::new_moe_grouped_scratch_raw(&dev, ne, dim, inter, topk, 16, 16 * topk);
    let stream = gb10_inference::dsv4_gpu::blocking_compute_stream(&dev);

    for n in [1usize, 6, 16] {
        let mut rng = XorShift(0xD5E4_3000_0000 + n as u64);
        let x_bits = gen_x(&mut rng, n, dim);
        let (ids, wts) = gen_routing(&mut rng, n, ne, topk, cfg.route_scale);
        let x_d = dev.htod_sync_copy(&to_bf16(&x_bits)).unwrap();
        let ids_d = dev.htod_sync_copy(&ids).unwrap();
        let wts_d = dev.htod_sync_copy(&wts).unwrap();
        let mut run = |v: MoeVariant, grouped: bool| -> Vec<u16> {
            {
                let ws = scratch.dsv4_ws_mut();
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(*ws.xc.device_ptr(), *x_d.device_ptr(), n * dim * 2, stream.stream).unwrap();
                    cudarc::driver::result::memcpy_dtod_async(*ws.idc.device_ptr(), *ids_d.device_ptr(), n * topk * 4, stream.stream).unwrap();
                    cudarc::driver::result::memcpy_dtod_async(*ws.wtc.device_ptr(), *wts_d.device_ptr(), n * topk * 4, stream.stream).unwrap();
                }
            }
            dispatch(&dev, &stream, &bk, &df, &gm, &mut scratch, grouped, n, topk, limit, v).expect("moe dispatch");
            dev.synchronize().unwrap();
            let full = dev.dtoh_sync_copy(&scratch.dsv4_ws_ref().outc).unwrap();
            to_bits(&full[..n * dim])
        };
        let a = run(MoeVariant::Single, false);
        let b = run(MoeVariant::X2, false);
        let c = run(MoeVariant::U4, false);
        let d = run(MoeVariant::Single, true);
        let e = run(MoeVariant::X2, true);
        let f = run(MoeVariant::U4, true);
        assert_eq!(a, b, "N={n}: n1_x2 NOT bitwise-identical to n1");
        assert_eq!(a, c, "N={n}: n1_u4 NOT bitwise-identical to n1");
        assert_eq!(d, e, "N={n}: grouped_x2 NOT bitwise-identical to grouped");
        assert_eq!(d, f, "N={n}: grouped_u4 NOT bitwise-identical to grouped");
        assert_eq!(c, f, "N={n}: n1_u4 vs grouped_u4 NOT bitwise-identical (contract)");
        println!("N={n}: x2/u4 == original bitwise (n1 and grouped), cross-contract holds");
    }
    println!("DSV4 MoE streaming bitwise gate: PASS");
}

/// Isolated wall-clock A/B (real weights, interleaved medians): single-tile vs x2-tile vs u4 at
/// the two serving shapes (N=1 decode dispatch; N=6 verify-width grouped dispatch).
#[test]
fn dsv4_moe_streaming_bench() {
    if !bundle().exists() { return; } // lab-model-bundle gate
    let _g = gate();
    let dev = CudaDevice::new(0).expect("CUDA device 0");
    let (bk, df) = load_modules(&dev);
    let cfg = dsv4_load::load_config(&bundle()).expect("load_config");
    let (ne, topk, dim, inter) = (cfg.n_routed_experts, cfg.n_activated_experts, cfg.dim, cfg.moe_inter_dim);
    let limit = cfg.swiglu_limit;

    let layer = dsv4_load::load_layer(&bundle(), &cfg, 2).expect("load_layer");
    let host = dsv4_moe::pack_moe_layer(&layer, &cfg).expect("pack_moe_layer");
    let gm = gpu::Dsv4MoeGpu::upload(&dev, &host).expect("upload");
    drop(host);
    let mut scratch = gpu::new_moe_grouped_scratch_raw(&dev, ne, dim, inter, topk, 16, 16 * topk);
    let stream = gb10_inference::dsv4_gpu::blocking_compute_stream(&dev);

    let mut bench = |grouped: bool, batch: usize, v: MoeVariant, reps: usize| -> f64 {
        let mut rng = XorShift(0xD5E4_4000_0000 + (grouped as u64) * 31 + batch as u64);
        let x_bits = gen_x(&mut rng, batch, dim);
        let (ids, wts) = gen_routing(&mut rng, batch, ne, topk, cfg.route_scale);
        let x_d = dev.htod_sync_copy(&to_bf16(&x_bits)).unwrap();
        let ids_d = dev.htod_sync_copy(&ids).unwrap();
        let wts_d = dev.htod_sync_copy(&wts).unwrap();
        let mut times = Vec::with_capacity(reps);
        for _ in 0..reps {
            dispatch(&dev, &stream, &bk, &df, &gm, &mut scratch, grouped, batch, topk, limit, v).expect("moe dispatch");
            // GPU-synced phase timing (dev.synchronize does NOT cover rt.stream — session-2 trap).
            unsafe { cudarc::driver::result::stream::synchronize(stream.stream).unwrap() };
            {
                let ws = scratch.dsv4_ws_mut();
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(*ws.xc.device_ptr(), *x_d.device_ptr(), batch * dim * 2, stream.stream).unwrap();
                    cudarc::driver::result::memcpy_dtod_async(*ws.idc.device_ptr(), *ids_d.device_ptr(), batch * topk * 4, stream.stream).unwrap();
                    cudarc::driver::result::memcpy_dtod_async(*ws.wtc.device_ptr(), *wts_d.device_ptr(), batch * topk * 4, stream.stream).unwrap();
                }
            }
            // GPU-synced timing of the NEXT dispatch: the d2d staging is inside the timed window
            // (identical across arms) and stays ahead of the kernel stream.
            unsafe { cudarc::driver::result::stream::synchronize(stream.stream).unwrap() };
            let t0 = std::time::Instant::now();
            dispatch(&dev, &stream, &bk, &df, &gm, &mut scratch, grouped, batch, topk, limit, v).expect("moe dispatch");
            unsafe { cudarc::driver::result::stream::synchronize(stream.stream).unwrap() };
            times.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        times[times.len() / 2]
    };

    let variants = [MoeVariant::Single, MoeVariant::X2, MoeVariant::U4];
    let vname = |v: MoeVariant| match v { MoeVariant::Single => "single-tile", MoeVariant::X2 => "x2-tile", MoeVariant::U4 => "u4-tile" };
    for (grouped, batch, label) in [(false, 1usize, "n1   (decode dispatch, batch=1)"), (true, 6, "grp6 (verify dispatch, batch=6)")] {
        // Interleaved A/B/C (AGENTS §5: sequential arms drift ±3%): 3 warmups per arm, then
        // 9 interleaved reps per arm, medians compared.
        for _ in 0..3 {
            for v in variants { let _ = bench(grouped, batch, v, 1); }
        }
        let mut all: Vec<f64> = Vec::new();
        for v in variants {
            let mut t = Vec::new();
            for _ in 0..9 { t.push(bench(grouped, batch, v, 1)); }
            t.sort_by(|a, b| a.partial_cmp(b).unwrap());
            all.push(t[t.len() / 2]);
            println!("  {label} {:<12} median {:.3} ms", vname(v), t[t.len() / 2]);
        }
        println!(
            "{label}: single {:.3} ms  x2 {:.3} ms ({:.3}x)  u4 {:.3} ms ({:.3}x)",
            all[0], all[1], all[0] / all[1], all[2], all[0] / all[2]
        );
    }
}
