//! The `run_probe_tq` probe (one long mode, kept whole). Called from the mode dispatch in `fn main`.

use super::*;

/// `--probe-tq [goldens-dir]` — TurboQuant KV engine kernels vs the E4 reference goldens
/// (/tmp/tq_ref2/goldens; REPORT.md + tq.py + golden.py). No model load: a bare CUDA device +
/// the gpu_batch.ptx kernels (single-node probe usage). The golden q/k/v are converted to the
/// ENGINE's bf16 input representation (the engine's q/k/v after the qkv GEMM are bf16), so the
/// numbers include the input-representation gap the engine will actually have. Validates:
///   write_kv_b_tq (pack-at-write)     -> packed rows: byte-match vs k_packed/v_packed
///                                        (informational; boundary flips expected), K codebook
///                                        cost mean(||r||²) (~0.116 — the K quality gate),
///   dequant_kv_tq_full                -> V reconstruction rel-L2 (~0.034), K rel-L2 (~0.18,
///                                        QJL reconstruction noise by design, NOT a gate)
///   rotate_q_tq + gqa_attn_splitk_tq  -> dual-dot scores + the empirical score bias (~0) and
///   + gqa_attn_reduce                   the full PV path. The golden scores/pv used f32 q/k/v
///                                        while the engine's q/k/v are bf16 — that input gap
///                                        dominates the raw golden comparison (torch's own
///                                        bf16-k row delta is 0.0880), so the probe ALSO recomputes
///                                        scores/PV on the host (f64) from the ENGINE's packed
///                                        rows + bf16(q) and gates the KERNEL math on that
///                                        (expect ~1e-3).
///   gqa_attn_splitk_tq_gq             -> the serving hot path: bit-identical to the per-head
///                                        kernel on the same queries
///   compact_kv_tq                     -> byte-identical row-copy round trip
/// Also verifies the embedded constants (src/tq_consts) are byte-identical to the golden files
/// (the engine and the reference must share Pi/S/codebooks/scale).
pub(crate) fn run_probe_tq(args: &[String]) {
    use cudarc::driver::{CudaDevice, DevicePtr, LaunchAsync, LaunchConfig};
    use cudarc::nvrtc::Ptx;
    use half::bf16;

    const D: usize = 128;                 // head_dim (the TQ layout is d=128)
    const LAYERS: usize = 80;
    const KV_HEADS: usize = 4;
    const N_TOK: usize = 32;
    const NROWS: usize = LAYERS * KV_HEADS * N_TOK;   // 10240 golden rows
    // K channel bit width from the env: [kv-tq]=3 -> b=3 K (68-B rows, gpu_batch_b3.ptx);
    // otherwise the golden-anchored b=2 layout (52-B rows). V is 3-bit in both.
    let b3 = gb10_inference::opts::var(gb10_inference::opt!("kv-tq")).ok().as_deref() == Some("3");
    let row_bytes = if b3 { 68usize } else { 52usize };   // TQ_ROW_BYTES (_B3)
    const ROW2: usize = 52;               // b=2 row size (the golden k_packed layout)
    const ENC_SMEM_B3: u32 = 4176;        // TQ_ENCODE_SMEM_BYTES_B3
    const ENC_SMEM: u32 = 4160;           // TQ_ENCODE_SMEM_BYTES
    const DEQ_SMEM: u32 = 3 * 128 * 4;    // TQ_DEQ_SMEM_BYTES
    const ROT_SMEM: u32 = 128 * 4;        // TQ_ROTATE_SMEM_BYTES
    const SK_SMEM: u32 = ((4 * D + 2 * 4 + D) * 4) as u32;    // per-head splitk smem (blockDim 128, NW=4)
    let enc_smem = if b3 { ENC_SMEM_B3 } else { ENC_SMEM };

    let dir = parse_arg(args, "--probe-tq")
        .map(|s| s.to_string())
        .unwrap_or_else(|| "/tmp/tq_ref2/goldens".to_string());
    let g = |n: &str| std::fs::read(format!("{dir}/{n}"))
        .unwrap_or_else(|e| panic!("cannot read {dir}/{n}: {e}"));
    let f32v = |b: Vec<u8>| -> Vec<f32> {
        b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    };
    let bf2f = |v: &[bf16]| -> Vec<f32> { v.iter().map(|&x| bf16::to_f32(x)).collect() };

    println!("=== TQ kernel validation (E4) vs {dir} ===");

    // ---- 1. embedded constants == golden files ----
    let embedded: Vec<(&str, &[u8])> = vec![
        ("Pi.bin", include_bytes!("../tq_consts/Pi.bin")),
        ("Pit.bin", include_bytes!("../tq_consts/Pit.bin")),
        ("S.bin", include_bytes!("../tq_consts/S.bin")),
        ("codebooks.bin", include_bytes!("../tq_consts/codebooks.bin")),
        ("scale.bin", include_bytes!("../tq_consts/scale.bin")),
    ];
    let mut consts_ok = true;
    for (name, eb) in &embedded {
        let gb = g(name);
        if gb.as_slice() != *eb {
            consts_ok = false;
            println!("  CONST MISMATCH: {name} (embedded vs golden) — the engine would NOT match the reference");
        }
    }
    println!("  embedded constants == golden files: {}", if consts_ok { "OK" } else { "FAIL" });

    // ---- device + kernels ----
    let dev = CudaDevice::new(0).expect("CUDA device 0");
    let ptx_name = if b3 { "src/ptx/gpu_batch_b3.ptx" } else { "src/ptx/gpu_batch.ptx" };
    let ptx = Ptx::from_src(std::fs::read_to_string(ptx_name).expect(ptx_name));
    let names = ["write_kv_b_tq", "write_kv_prefill_tq", "rotate_q_tq", "dequant_kv_tq_full", "compact_kv_tq",
                 "gqa_attn_splitk_tq", "gqa_attn_splitk_tq_gq", "gqa_attn_splitk_tq_dbg_scores",
                 "gqa_attn_reduce"];
    let mod_name = if b3 { "tqprobe_b3" } else { "tqprobe" };
    dev.load_ptx(ptx, mod_name, &names).expect("load TQ kernels");
    let f = |n: &str| dev.get_func(mod_name, n).unwrap_or_else(|| panic!("missing kernel {n}"));
    let (kwrite, kwrite_pf, krotate, kdeq, kcompact) = (f("write_kv_b_tq"), f("write_kv_prefill_tq"), f("rotate_q_tq"), f("dequant_kv_tq_full"), f("compact_kv_tq"));
    let (ksplitk, kscores, kreduce) = (f("gqa_attn_splitk_tq"), f("gqa_attn_splitk_tq_dbg_scores"), f("gqa_attn_reduce"));

    let tables = gb10_inference::gpu::GpuModel::build_tq_tables(&dev).expect("build_tq_tables");
    let tbl_ptr = *tables.device_ptr() as u64;

    // ---- 2. data: golden q/k/v f32 -> bf16 (the engine's input representation) ----
    let q_f32 = f32v(g("q.bin"));
    let k_f32 = f32v(g("k.bin"));
    let v_f32 = f32v(g("v.bin"));
    assert_eq!((q_f32.len(), k_f32.len(), v_f32.len()), (NROWS * D, NROWS * D, NROWS * D));
    let qb = dev.htod_sync_copy(&q_f32.iter().map(|&x| bf16::from_f32(x)).collect::<Vec<_>>()).unwrap();
    let kb = dev.htod_sync_copy(&k_f32.iter().map(|&x| bf16::from_f32(x)).collect::<Vec<_>>()).unwrap();
    let vb = dev.htod_sync_copy(&v_f32.iter().map(|&x| bf16::from_f32(x)).collect::<Vec<_>>()).unwrap();
    // The b=2 layout validates against the golden k_packed.bin (52-B rows); the b=3 layout has
    // no committed golden — k_packed_b3.bin was regenerated with the reference math
    // (encode_k at CODEBOOK_BITS_K=3, 68-B rows, /tmp/gen_b3_golden.py). V rows are 50 B (no
    // pad) in both references; the engine stores them at the uniform row stride with the tail
    // bytes unused — compare the meaningful 50 B per row.
    let k_golden = if b3 { g("k_packed_b3.bin") } else { g("k_packed.bin") };
    let v_golden = g("v_packed.bin");
    let krow_golden = if b3 { row_bytes } else { ROW2 };
    assert_eq!(k_golden.len(), NROWS * krow_golden);
    assert_eq!(v_golden.len(), NROWS * 50);

    // ---- 3. write_kv_b_tq: pack all 10240 rows (B=NROWS, nkv=1, stride=NROWS, pos[b]=b) ----
    let pos: Vec<i32> = (0..NROWS as i32).collect();
    let slot_ids = vec![0i32; NROWS];
    let pos_dev = dev.htod_sync_copy(&pos).unwrap();
    let slots_dev = dev.htod_sync_copy(&slot_ids).unwrap();
    let kcache = dev.alloc_zeros::<u8>(NROWS * row_bytes).unwrap();
    let vcache = dev.alloc_zeros::<u8>(NROWS * row_bytes).unwrap();
    unsafe {
        kwrite.clone().launch(LaunchConfig { grid_dim: (NROWS as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: enc_smem },
            (&kcache, &vcache, &kb, &vb, &tables, &pos_dev, NROWS as i32, 1i32, NROWS as i32, &slots_dev)).unwrap();
    }
    dev.synchronize().unwrap();
    let kp = dev.dtoh_sync_copy(&kcache).unwrap();
    let vp = dev.dtoh_sync_copy(&vcache).unwrap();
    let kmatch = kp.iter().zip(&k_golden).filter(|(a, b)| a == b).count();
    let mut vmatch = 0usize;
    for r in 0..NROWS {
        for j in 0..50 {
            if vp[r * row_bytes + j] == v_golden[r * 50 + j] { vmatch += 1; }
        }
    }
    println!("  packed byte-match vs golden ({} K rows): K {:.3}%  V {:.3}%  (engine packs the bf16 input; the reference packs f32 — boundary flips expected)",
             if b3 { "b=3" } else { "b=2" },
             100.0 * kmatch as f64 / (NROWS * krow_golden) as f64, 100.0 * vmatch as f64 / (NROWS * 50) as f64);

    // ---- 4. K codebook cost from the ENGINE rows (the fp16-rounded ||r||) ----
    let mut rn2 = 0.0f64;
    for r in 0..NROWS {
        let u = u16::from_le_bytes([kp[r * row_bytes + if b3 { 64 } else { 48 }], kp[r * row_bytes + if b3 { 65 } else { 49 }]]);
        let rn = half::f16::from_bits(u).to_f32() as f64;
        rn2 += rn * rn;
    }
    let k_cost = rn2 / NROWS as f64;
    let (k_cost_ref, k_cost_paper, k_cost_tag) = if b3 { (0.034, 0.03, "b=3") } else { (0.116, 0.117, "b=2") };
    let k_cost_ok = (k_cost - k_cost_ref).abs() < 0.01;
    println!("  K codebook cost mean(||r||^2) = {:.5}   ({k_cost_tag}: reference ~{k_cost_ref}, paper {k_cost_paper}) {}",
             k_cost, if k_cost_ok { "OK" } else { "FAIL" });

    // ---- 4b. write_kv_prefill_tq: the N-token writer shares tq_encode_rows; verify its
    // indexing by re-packing every row through the prefill form and byte-comparing with the
    // decode-path rows (identical content, different launch mapping).
    let kcache2 = dev.alloc_zeros::<u8>(NROWS * row_bytes).unwrap();
    let vcache2 = dev.alloc_zeros::<u8>(NROWS * row_bytes).unwrap();
    unsafe {
        kwrite_pf.clone().launch(LaunchConfig { grid_dim: (NROWS as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: enc_smem },
            (&kcache2, &vcache2, &kb, &vb, &tables, NROWS as i32, 1i32, NROWS as i32, 0i32)).unwrap();
    }
    dev.synchronize().unwrap();
    let kp_pf = dev.dtoh_sync_copy(&kcache2).unwrap();
    let vp_pf = dev.dtoh_sync_copy(&vcache2).unwrap();
    // Compare the meaningful bytes only: K rows are fully written (row_bytes); V rows carry 50 B
    // + PAD, which is never written (alloc_zeros does not zero, so the pad is garbage).
    let pf_ok = kp_pf == kp
        && (0..NROWS).all(|r| (0..50).all(|j| vp_pf[r * row_bytes + j] == vp[r * row_bytes + j]));
    if !pf_ok {
        let mut shown = 0;
        'outer: for r in 0..NROWS {
            for j in 0..row_bytes {
                if kp_pf[r * row_bytes + j] != kp[r * row_bytes + j] && shown < 4 {
                    println!("    K diff row {r} byte {j}: prefill={:#04x} decode={:#04x}", kp_pf[r * row_bytes + j], kp[r * row_bytes + j]);
                    shown += 1;
                    if shown >= 4 { break 'outer; }
                }
            }
        }
        let mut shown = 0;
        'outer: for r in 0..NROWS {
            for j in 0..row_bytes {
                if vp_pf[r * row_bytes + j] != vp[r * row_bytes + j] && shown < 4 {
                    println!("    V diff row {r} byte {j}: prefill={:#04x} decode={:#04x}", vp_pf[r * row_bytes + j], vp[r * row_bytes + j]);
                    shown += 1;
                    if shown >= 4 { break 'outer; }
                }
            }
        }
    }
    println!("  write_kv_prefill_tq rows == write_kv_b_tq rows: {}", if pf_ok { "OK" } else { "FAIL" });

    // ---- 5. dequant_kv_tq_full: K/V reconstruction (oracle path, f32 out) ----
    let kdq = dev.alloc_zeros::<f32>(NROWS * D).unwrap();
    let vdq = dev.alloc_zeros::<f32>(NROWS * D).unwrap();
    unsafe {
        kdeq.clone().launch(LaunchConfig { grid_dim: (NROWS as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: DEQ_SMEM },
            (&kdq, &vdq, &kcache, &vcache, &tables, 1i32, NROWS as i32, NROWS as i32, 0i32, NROWS as i32)).unwrap();
    }
    dev.synchronize().unwrap();
    let kd = dev.dtoh_sync_copy(&kdq).unwrap();
    let vd = dev.dtoh_sync_copy(&vdq).unwrap();
    let rel2 = |a: &[f32], b: &[f32]| -> f64 {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in 0..a.len() {
            let d = (a[i] - b[i]) as f64;
            num += d * d;
            den += (b[i] as f64) * (b[i] as f64);
        }
        num / den
    };
    let v_rel = rel2(&vd, &bf2f(&v_f32.iter().map(|&x| bf16::from_f32(x)).collect::<Vec<_>>()));
    let v_rel_ok = (v_rel - 0.034).abs() < 0.01;
    println!("  V dequant rel-L2 vs bf16(v) = {:.5}  (vs f32 v: {:.5})   (reference ~0.034) {}",
             v_rel, rel2(&vd, &v_f32), if v_rel_ok { "OK" } else { "FAIL" });
    println!("  K dequant rel-L2 vs bf16(k) = {:.5}  (QJL reconstruction noise ~0.18 by design — NOT a gate; the codebook cost above is the K gate)",
             rel2(&kd, &bf2f(&k_f32.iter().map(|&x| bf16::from_f32(x)).collect::<Vec<_>>())));

    // ---- 6. rotate_q_tq: every query head once (B=NROWS, nh=1) ----
    let qrqs = dev.alloc_zeros::<f32>(NROWS * 2 * D).unwrap();
    unsafe {
        krotate.clone().launch(LaunchConfig { grid_dim: (NROWS as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: ROT_SMEM },
            (&qrqs, &qb, &tables, 1i32, D as i32, NROWS as i32, D as i32)).unwrap();
    }

    // ---- 7. per-(layer, head) slices: scores + PV (the real engine path) ----
    let scores_g = f32v(g("scores_tq.bin"));
    let scores_exact_g = f32v(g("scores_exact.bin"));
    let pv_g = f32v(g("pv_tq.bin"));
    // Host bf16-input references: the golden scores_tq/pv_tq were computed from f32 q/k/v, but
    // the ENGINE's q/k/v are bf16 — that input gap dominates the golden comparison (torch's own
    // bf16-k row delta is 0.0880, matching the kernel's 0.088). To isolate the KERNEL math we
    // recompute scores/PV on the host (f64) from the ENGINE's packed rows + bf16(q) — the exact
    // inputs the kernels consumed — and gate on THAT instead.
    let qb_f32: Vec<f32> = q_f32.iter().map(|&x| bf16::to_f32(bf16::from_f32(x))).collect();
    let cb_all = f32v(g("codebooks.bin"));
    let pi_f64: Vec<f64> = f32v(g("Pi.bin")).iter().map(|&x| x as f64).collect();
    let s_f64: Vec<f64> = f32v(g("S.bin")).iter().map(|&x| x as f64).collect();
    // K codebook: cb2[2..6) for b=2, cb3[6..14) for b=3 (both in the embedded/golden codebooks).
    let cbk_f64: Vec<f64> = cb_all[if b3 { 6..14 } else { 2..6 }].iter().map(|&x| x as f64).collect();
    let qjl_scale = f32v(g("scale.bin"))[0] as f64;
    // host unpack of an ENGINE K row (row_bytes): (codes[128], signs[128] +-1, rn, kn) — the
    // fp16 norms are the packed values, exactly what the kernels read back. b=3 K codes are the
    // LSB-first 3-bit bitstream (coord j at bits [3j, 3j+3)); signs at TQ_SIGN_OFF; rn/kn at
    // TQ_RN_OFF/TQ_KN_OFF (48/64, 50/66).
    let sign_off = if b3 { 48 } else { 32 };
    let rn_off = if b3 { 64 } else { 48 };
    let kn_off = if b3 { 66 } else { 50 };
    let unpack_k = |row: &[u8]| -> (Vec<i32>, Vec<f32>, f32, f32) {
        let mut codes = vec![0i32; D];
        let mut signs = vec![0.0f32; D];
        for j in 0..D {
            if b3 {
                let b0 = (3 * j) >> 3;
                let off = (3 * j) & 7;
                let w = (row[b0] as u16) | ((row[b0 + 1] as u16) << 8);
                codes[j] = ((w >> off) & 7) as i32;
            } else {
                codes[j] = ((row[j >> 2] >> (2 * (j & 3))) & 3) as i32;
            }
            signs[j] = if ((row[sign_off + (j >> 3)] >> (j & 7)) & 1) == 1 { 1.0 } else { -1.0 };
        }
        let rn = half::f16::from_bits(u16::from_le_bytes([row[rn_off], row[rn_off + 1]])).to_f32();
        let kn = half::f16::from_bits(u16::from_le_bytes([row[kn_off], row[kn_off + 1]])).to_f32();
        (codes, signs, rn, kn)
    };
    // Per-query rotated/projected query vectors (bf16 q, f64) — shared by the score and PV refs.
    let mut qr_host: Vec<Vec<f64>> = Vec::with_capacity(NROWS);
    let mut qs_host: Vec<Vec<f64>> = Vec::with_capacity(NROWS);
    for t in 0..NROWS {
        let qb: Vec<f64> = (0..D).map(|i| qb_f32[t * D + i] as f64).collect();
        let mut qr = vec![0.0f64; D];
        let mut qs = vec![0.0f64; D];
        for j in 0..D {
            let mut ar = 0.0f64;
            let mut as2 = 0.0f64;
            for i in 0..D {
                ar += pi_f64[j * D + i] * qb[i];
                as2 += s_f64[j * D + i] * qb[i];
            }
            qr[j] = ar;
            qs[j] = as2;
        }
        qr_host.push(qr);
        qs_host.push(qs);
    }
    let pos32 = vec![31i32; N_TOK];              // pc = 32: full causal row == the golden's full row
    let slots32 = vec![0i32; N_TOK];
    let pos32_dev = dev.htod_sync_copy(&pos32).unwrap();
    let slots32_dev = dev.htod_sync_copy(&slots32).unwrap();
    let bs_packed = (32u64 << 25) | (1u64 << 19) | (NROWS as u64);   // batch 32, ns_grid 1, stride 10240 (q_pitch 0)
    let nh_packed = (1i32 << 20) | ((D as i32) << 10) | 1i32;        // nh 1, hd 128, nkv 1
    let scores_dev = dev.alloc_zeros::<f32>(N_TOK * N_TOK).unwrap();
    let pm = dev.alloc_zeros::<f32>(N_TOK).unwrap();
    let pl = dev.alloc_zeros::<f32>(N_TOK).unwrap();
    let pa = dev.alloc_zeros::<f32>(N_TOK * D).unwrap();
    let attn = dev.alloc_zeros::<bf16>(N_TOK * D).unwrap();
    let mut s_rel = 0.0f64;      // kernel vs golden scores_tq (f32 q/k — includes the bf16-input gap)
    let mut s_den = 0.0f64;
    let mut sr_rel = 0.0f64;     // kernel vs host bf16-input reference (kernel math only) — the gate
    let mut sr_den = 0.0f64;
    let mut bias_sum = 0.0f64;
    let mut bias_sq = 0.0f64;
    let mut pv_rel_num = 0.0f64; // engine PV vs golden pv_tq (f32 — includes the bf16-input gap)
    let mut pv_rel_den = 0.0f64;
    let mut pvr_num = 0.0f64;    // engine PV vs host exact-arithmetic PV on the same inputs — the gate
    let mut pvr_den = 0.0f64;
    for l in 0..LAYERS {
        for h in 0..KV_HEADS {
            let slice = (l * KV_HEADS + h) * N_TOK;
            let qo = (slice * 2 * D * 4) as u64;       // qrqs byte offset (f32)
            let ko = (slice * row_bytes) as u64;      // cache byte offset (u8 rows)
            let (qrp, kcp, vcp) = (*qrqs.device_ptr() as u64 + qo, *kcache.device_ptr() as u64 + ko, *vcache.device_ptr() as u64 + ko);
            unsafe {
                kscores.clone().launch(LaunchConfig { grid_dim: (N_TOK as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 },
                    (&scores_dev, qrp, kcp, tbl_ptr, &pos32_dev, bs_packed, nh_packed, &slots32_dev)).unwrap();
                ksplitk.clone().launch(LaunchConfig { grid_dim: (N_TOK as u32, 1, 1), block_dim: (D as u32, 1, 1), shared_mem_bytes: SK_SMEM },
                    (&pm, &pl, &pa, qrp, kcp, vcp, tbl_ptr, &pos32_dev, bs_packed, nh_packed, &slots32_dev, 0u64)).unwrap();
                kreduce.clone().launch(LaunchConfig { grid_dim: (N_TOK as u32, 1, 1), block_dim: (D as u32, 1, 1), shared_mem_bytes: 0 },
                    (&attn, &pm, &pl, &pa, &pos32_dev, 1i32, N_TOK as i32, nh_packed)).unwrap();
            }
            let sc = dev.dtoh_sync_copy(&scores_dev).unwrap();
            let at = dev.dtoh_sync_copy(&attn).unwrap();
            // host bf16-input score reference for this slice (f64): the reference formula on the
            // ENGINE's packed rows + bf16(q) — the exact inputs the kernels consumed.
            let rows: Vec<(Vec<i32>, Vec<f32>, f32, f32)> =
                (0..N_TOK).map(|r| unpack_k(&kp[(slice + r) * row_bytes..(slice + r + 1) * row_bytes])).collect();
            for q in 0..N_TOK {
                // exact-arithmetic PV reference: softmax(s_kernel) x V_dequant (original domain —
                // Pi^T(softmax . vn.cb3) == softmax . (Pi^T vn.cb3), so the dequant values vd are
                // the right basis; the engine output must match up to its fp32 softmax ordering).
                let mut srow: Vec<f64> = Vec::with_capacity(N_TOK);
                for r in 0..N_TOK {
                    let (codes, signs, rn, kn) = &rows[r];
                    let mut code = 0.0f64;
                    let mut qjl = 0.0f64;
                    for j in 0..D {
                        code += qr_host[slice + q][j] * cbk_f64[codes[j] as usize];
                        qjl += qs_host[slice + q][j] * signs[j] as f64;
                    }
                    let sr = *kn as f64 * (code + qjl_scale * *rn as f64 * qjl);
                    srow.push(sr);
                }
                let smax = srow.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mut wsum = 0.0f64;
                let mut w: Vec<f64> = Vec::with_capacity(N_TOK);
                for &sr in &srow {
                    let e = (sr - smax).exp();
                    w.push(e);
                    wsum += e;
                }
                for r in 0..N_TOK {
                    let e = scores_g[slice * N_TOK + q * N_TOK + r] as f64;
                    let a = sc[q * N_TOK + r] as f64;
                    let d = a - e;
                    s_rel += d * d;
                    s_den += e * e;
                    let ex = scores_exact_g[slice * N_TOK + q * N_TOK + r] as f64;
                    let dd = a - ex;
                    bias_sum += dd;
                    bias_sq += dd * dd;
                    // kernel score vs host bf16-input reference
                    let dd2 = a - srow[r];
                    sr_rel += dd2 * dd2;
                    sr_den += srow[r] * srow[r];
                }
                for d in 0..D {
                    let e = pv_g[slice * D + q * D + d] as f64;
                    let a = bf16::to_f32(at[q * D + d]) as f64;
                    let dd = a - e;
                    pv_rel_num += dd * dd;
                    pv_rel_den += e * e;
                    // engine PV vs exact-arithmetic PV on the same inputs
                    let mut pr = 0.0f64;
                    for r in 0..N_TOK {
                        pr += w[r] / wsum * vd[(slice + r) * D + d] as f64;
                    }
                    let dd2 = a - pr;
                    pvr_num += dd2 * dd2;
                    pvr_den += pr * pr;
                }
            }
        }
    }
    dev.synchronize().unwrap();
    let score_rel = (s_rel / s_den).sqrt();        // vs golden (f32 q/k)
    let score_kernel_rel = (sr_rel / sr_den).sqrt(); // vs host bf16-input reference (kernel math)
    let pv_rel = (pv_rel_num / pv_rel_den).sqrt();   // vs golden pv_tq (f32)
    let pv_kernel_rel = (pvr_num / pvr_den).sqrt();  // vs exact-arithmetic PV on the same inputs
    let n = (NROWS * N_TOK) as f64;
    let bias = bias_sum / n;
    let bias_std = (bias_sq / n - bias * bias).max(0.0).sqrt();
    let score_ok = score_kernel_rel < 1e-2;
    let pv_ok = pv_kernel_rel < 1e-2;
    println!("  score rel-L2 vs scores_tq.bin = {:.4e}   (the bf16 q/k input gap — torch's own bf16-k row delta is 0.0880, matching)",
             score_rel);
    println!("  score rel-L2 vs host bf16-input reference = {:.4e}  (kernel math only)  {}",
             score_kernel_rel, if score_ok { "OK" } else { "FAIL" });
    println!("  score bias mean = {:.3}  std = {:.3}   (reference: ~0; the fixed-S empirical mean over all {} golden queries)",
             bias, bias_std, n as usize);
    println!("  PV rel-L2 vs pv_tq.bin = {:.4e}   (includes the bf16-input score gap)", pv_rel);
    println!("  PV rel-L2 vs exact-arithmetic PV on the same inputs = {:.4e}  (kernel math only)  {}",
             pv_kernel_rel, if pv_ok { "OK" } else { "FAIL" });

    // ---- 8. gqa_attn_splitk_tq_gq: the serving hot path, per-head bit-identical ----
    // Hy3's 8:1 GQA will run ONLY the GQA-packed kernel at serving time, so a bug there must not
    // pass the probe. Re-run one slice with nh=2/nkv=1 (16 tokens x 2 heads, gqa_ratio=2) and
    // assert the gq outputs equal the per-head kernel's for the same 32 queries, bit-for-bit
    // (the "per-head bit-identical by construction" claim, checked).
    let f_gq = f("gqa_attn_splitk_tq_gq");
    const GQ_SMEM: u32 = ((2 * 8 * D + 4 * D + 2 * 4 + D) * 4) as u32;   // gq smem: qr/qs stage + merge
    let (l, h) = (3usize, 2usize);
    let slice = (l * KV_HEADS + h) * N_TOK;
    let qo = (slice * 2 * D * 4) as u64;
    let ko = (slice * row_bytes) as u64;
    let mut q2: Vec<bf16> = Vec::with_capacity(16 * 2 * D);
    for b in 0..16 {
        for qh in 0..2 {
            for i in 0..D {
                q2.push(bf16::from_f32(q_f32[(slice + b * 2 + qh) * D + i]));
            }
        }
    }
    let q2_dev = dev.htod_sync_copy(&q2).unwrap();
    let qrqs2 = dev.alloc_zeros::<f32>(16 * 2 * 2 * D).unwrap();
    unsafe {
        krotate.clone().launch(LaunchConfig { grid_dim: (32u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: ROT_SMEM },
            (&qrqs2, &q2_dev, &tables, 2i32, D as i32, 16i32)).unwrap();
    }
    let pos16 = vec![31i32; 16];
    let slots16 = vec![0i32; 16];
    let pos16_dev = dev.htod_sync_copy(&pos16).unwrap();
    let slots16_dev = dev.htod_sync_copy(&slots16).unwrap();
    let bs16 = (16i32 << 25) | (1i32 << 19) | (NROWS as i32);
    let nh2 = (2i32 << 20) | ((D as i32) << 10) | 1i32;
    let pm2 = dev.alloc_zeros::<f32>(16 * 2).unwrap();
    let pl2 = dev.alloc_zeros::<f32>(16 * 2).unwrap();
    let pa2 = dev.alloc_zeros::<f32>(16 * 2 * D).unwrap();
    let attn2 = dev.alloc_zeros::<bf16>(16 * 2 * D).unwrap();
    let pm1 = dev.alloc_zeros::<f32>(32).unwrap();
    let pl1 = dev.alloc_zeros::<f32>(32).unwrap();
    let pa1 = dev.alloc_zeros::<f32>(32 * D).unwrap();
    let attn1 = dev.alloc_zeros::<bf16>(32 * D).unwrap();
    unsafe {
        f_gq.clone().launch(LaunchConfig { grid_dim: (16u32, 1, 1), block_dim: (D as u32, 1, 1), shared_mem_bytes: GQ_SMEM },
            (&pm2, &pl2, &pa2, &qrqs2, *kcache.device_ptr() as u64 + ko, *vcache.device_ptr() as u64 + ko,
             tbl_ptr, &pos16_dev, bs16, nh2, &slots16_dev, 0u64)).unwrap();
        kreduce.clone().launch(LaunchConfig { grid_dim: (32u32, 1, 1), block_dim: (D as u32, 1, 1), shared_mem_bytes: 0 },
            (&attn2, &pm2, &pl2, &pa2, &pos16_dev, 1i32, 16i32, nh2)).unwrap();
        // per-head reference for the same 32 queries (nh=1, batch=32)
        ksplitk.clone().launch(LaunchConfig { grid_dim: (32u32, 1, 1), block_dim: (D as u32, 1, 1), shared_mem_bytes: SK_SMEM },
            (&pm1, &pl1, &pa1, *qrqs.device_ptr() as u64 + qo, *kcache.device_ptr() as u64 + ko, *vcache.device_ptr() as u64 + ko,
             tbl_ptr, &pos32_dev, bs_packed, nh_packed, &slots32_dev, 0u64)).unwrap();
        kreduce.clone().launch(LaunchConfig { grid_dim: (32u32, 1, 1), block_dim: (D as u32, 1, 1), shared_mem_bytes: 0 },
            (&attn1, &pm1, &pl1, &pa1, &pos32_dev, 1i32, 32i32, nh_packed)).unwrap();
    }
    dev.synchronize().unwrap();
    let at2 = dev.dtoh_sync_copy(&attn2).unwrap();
    let at1 = dev.dtoh_sync_copy(&attn1).unwrap();
    let mut gq_bits = 0usize;
    for q in 0..32 {
        for d in 0..D {
            if at2[q * D + d].to_bits() != at1[q * D + d].to_bits() { gq_bits += 1; }
        }
    }
    let gq_ok = gq_bits == 0;
    println!("  gqa_attn_splitk_tq_gq vs per-head (l=3,h=2, nh=2/nkv=1): {} differing bf16 elems of {}  {}",
             gq_bits, 32 * D, if gq_ok { "OK (bit-identical)" } else { "FAIL" });

    // ---- 9. compact_kv_tq byte round trip (gather + scatter) ----
    let src_pos: Vec<i32> = (0..NROWS as i32).collect();
    let sp = dev.htod_sync_copy(&src_pos).unwrap();
    let ks = dev.alloc_zeros::<u8>(NROWS * row_bytes).unwrap();
    let vsn = dev.alloc_zeros::<u8>(NROWS * row_bytes).unwrap();
    unsafe {
        kcompact.clone().launch(LaunchConfig { grid_dim: ((NROWS * row_bytes) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
            (&kcache, &vcache, &ks, &vsn, &sp, NROWS as i32, 0i32, 0i32, 1i32, NROWS as i32, 0i32)).unwrap();
        dev.synchronize().unwrap();
        kcompact.clone().launch(LaunchConfig { grid_dim: ((NROWS * row_bytes) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
            (&kcache, &vcache, &ks, &vsn, &sp, NROWS as i32, 0i32, 0i32, 1i32, NROWS as i32, 1i32)).unwrap();
    }
    dev.synchronize().unwrap();
    let kp2 = dev.dtoh_sync_copy(&kcache).unwrap();
    let vp2 = dev.dtoh_sync_copy(&vcache).unwrap();
    let compact_ok = kp2 == kp && vp2 == vp;
    println!("  compact_kv_tq round trip byte-identical: {}", if compact_ok { "OK" } else { "FAIL" });

    let pass = consts_ok && k_cost_ok && v_rel_ok && score_ok && pv_ok && gq_ok && pf_ok && compact_ok;
    println!("=== TQ probe {} ===", if pass { "PASS" } else { "FAIL" });
    if !pass { std::process::exit(1); }
}
