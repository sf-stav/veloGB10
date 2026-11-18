//! Unit tests for the fp8 block-scale shard helpers (former inline test module `shard_raw_fp8_blk_tests` of `gpu.rs`; file map in `mod.rs`).

    use super::*;

    /// Synthetic block-128 FP8 tensor: m=256 (2 bands), k=256 (nb_k=2). Byte i of the raw is
    /// (i % 251) as u8; scale [b, kb] = 1.0 + b*10 + kb. Roundtrip: shard to `world` ranks,
    /// reassemble in rank order, and require byte-exact equality of BOTH the codes and the
    /// flat scale matrix (the sharder must preserve the code<->scale pairing exactly).
    fn synth() -> (Vec<u8>, Vec<f32>, usize, usize, usize) {
        let (m, k) = (512usize, 256usize);
        let nb_k = k / 128;
        let qw: Vec<u8> = (0..m * k).map(|i| (i % 251) as u8).collect();
        let sc: Vec<f32> = (0..(m / 128) * nb_k).map(|i| 1.0 + (i / nb_k) as f32 * 10.0 + (i % nb_k) as f32).collect();
        (qw, sc, m, k, nb_k)
    }

    fn dequant_ref(qw: &[u8], sc: &[f32], m: usize, k: usize, nb_k: usize) -> Vec<f32> {
        qw.iter().enumerate().map(|(i, &c)| {
            let (r, c2) = (i / k, i % k);
            (c as f32) * sc[(r / 128) * nb_k + (c2 / 128)]
        }).collect()
    }

    #[test]
    fn col_shard_roundtrip() {
        let (qw, sc, m, k, nb_k) = synth();
        let mut gq = Vec::new(); let mut gs = Vec::new();
        for rank in 0..2usize {
            let (q, s, mo, kk, nb) =
                shard_raw_fp8_blk(&qw, &sc, m, k, nb_k, &LoadShardOp::Col, rank, 2);
            assert_eq!(mo, m / 2); assert_eq!(kk, k); assert_eq!(nb, nb_k);
            assert_eq!(q.len(), mo * kk); assert_eq!(s.len(), (mo / 128) * nb);
            gq.extend(q); gs.extend(s);
        }
        assert_eq!(gq, qw, "Col: reassembled codes differ");
        assert_eq!(gs, sc, "Col: reassembled scales differ");
    }

    #[test]
    fn col_segs_shard_roundtrip() {
        let (qw, sc, m, k, nb_k) = synth();
        // two segments [0,256) and [256,512), both split across world=2 (take=128, band-aligned
        // — the real fused-qkv geometry: every segment is a whole number of 128-row bands).
        // Per-rank layout is the FUSED rank-local order [local seg0 | local seg1], so the test
        // asserts each rank's piece against its expected slice, not against the original order.
        let segs = vec![(0usize, 256usize, true), (256usize, 256usize, true)];
        for rank in 0..2usize {
            let (q, s, mo, kk, nb) =
                shard_raw_fp8_blk(&qw, &sc, m, k, nb_k, &LoadShardOp::ColSegs(segs.clone()), rank, 2);
            assert_eq!(mo, m / 2); assert_eq!(kk, k); assert_eq!(nb, nb_k);
            let (off0, take0, off1, take1) = (0usize, 128usize, 256usize, 128usize);
            // rank r holds seg i's r-th half: [off_i + r*take_i .. +take_i]
            let (s0, s1) = (off0 + rank * take0, off1 + rank * take1);
            let mut expect_q = Vec::new();
            expect_q.extend_from_slice(&qw[s0 * k..(s0 + take0) * k]);
            expect_q.extend_from_slice(&qw[s1 * k..(s1 + take1) * k]);
            assert_eq!(q.len(), expect_q.len());
            let bad = q.iter().zip(&expect_q).position(|(a, b)| a != b);
            assert!(bad.is_none(), "ColSegs rank {rank}: first code diff at byte {:?} (got {} want {})",
                    bad, q[bad.unwrap()], expect_q[bad.unwrap()]);
            // scale bands follow the ROWS: band (s/128) for each rank-local segment slice
            let mut expect_s = Vec::new();
            expect_s.extend_from_slice(&sc[(s0 / 128) * nb_k..((s0 + take0) / 128) * nb_k]);
            expect_s.extend_from_slice(&sc[(s1 / 128) * nb_k..((s1 + take1) / 128) * nb_k]);
            assert_eq!(s, expect_s, "ColSegs rank {rank}: scales differ");
        }
    }

    #[test]
    fn row_shard_roundtrip_and_dequant() {
        let (qw, sc, m, k, nb_k) = synth();
        let mut gq: Vec<Option<u8>> = vec![None; m * k];
        let mut gs: Vec<Option<f32>> = vec![None; (m / 128) * nb_k];
        for rank in 0..2usize {
            let (q, s, mo, kk, nb) =
                shard_raw_fp8_blk(&qw, &sc, m, k, nb_k, &LoadShardOp::Row, rank, 2);
            assert_eq!(mo, m); assert_eq!(kk, k / 2); assert_eq!(nb, nb_k / 2);
            assert_eq!(s.len(), (m / 128) * nb);
            // scatter this rank's pieces back to global positions
            let k0 = rank * kk; let kb0 = rank * nb;
            for r in 0..mo {
                for c in 0..kk { gq[r * k + k0 + c] = Some(q[r * kk + c]); }
            }
            for b in 0..(m / 128) {
                for kb in 0..nb { gs[b * nb_k + kb0 + kb] = Some(s[b * nb + kb]); }
            }
        }
        let rq: Vec<u8> = gq.into_iter().map(|x| x.unwrap()).collect();
        let rs: Vec<f32> = gs.into_iter().map(|x| x.unwrap()).collect();
        assert_eq!(rq, qw, "Row: reassembled codes differ");
        assert_eq!(rs, sc, "Row: reassembled scales differ");
        // and the pairing: dequant of the reassembly equals the reference dequant
        let dq = dequant_ref(&rq, &rs, m, k, nb_k);
        let dr = dequant_ref(&qw, &sc, m, k, nb_k);
        assert_eq!(dq, dr, "Row: reassembled dequant differs");
    }

    #[test]
    fn misaligned_segment_panics() {
        let (qw, sc, m, k, nb_k) = synth();
        let segs = vec![(0usize, 96usize, true)]; // 96 rows: NOT a 128 multiple
        let r = std::panic::catch_unwind(|| {
            let _ = shard_raw_fp8_blk(&qw, &sc, m, k, nb_k, &LoadShardOp::ColSegs(segs), 0, 2);
        });
        assert!(r.is_err(), "misaligned ColSegs split must panic loudly");
    }
