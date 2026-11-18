//! Unit tests for the host shard byte helpers (former inline test module `shard_raw_tests` of `gpu.rs`; file map in `mod.rs`).

    use super::*;

    fn rng(n: usize, seed: u32) -> Vec<u8> {
        let mut s = seed;
        (0..n).map(|_| { s = s.wrapping_mul(1664525).wrapping_add(1013904223); (s >> 24) as u8 }).collect()
    }

    /// The shard-at-load invariant: slicing the RAW then repacking must equal repacking then
    /// slicing the MMA repack (the post-load sharders) — for Col, ColSegs, and Row — at every world
    /// in the power-of-two ladder. `m` (512 = 2·256) and the split segments (256) divide 16·world
    /// at every rung up to 16; K=256 divides row-shards.
    #[test]
    fn raw_slice_then_repack_equals_repack_then_slice() {
        for world in [1usize, 2, 4, 8, 16] {
            // K=512 so k/world stays %32 at world=16 (row-shard validity); M=512 divides 16·world.
            let k = 512usize;
            let m = 512usize;
            let qw = rng(m * k / 2, 3);
            let sc = rng(m * k / 16, 5);
            let gsv: Vec<f32> = (0..m / 16).map(|i| i as f32 * 0.5).collect();
            let segs = vec![(0usize, 256, true), (256, 256, true)];
            for rank in 0..world {
                for op in [LoadShardOp::Col, LoadShardOp::ColSegs(segs.clone()), LoadShardOp::Row] {
                    let (sq, ss, sg, sm, sk) = shard_raw_nvfp4(&qw, &sc, &gsv, m, k, &op, rank, world);
                    let (wt2, st2) = crate::quant::repack_nvfp4_mma(&sq, &ss, sm, sk);
                    let (wt, st) = crate::quant::repack_nvfp4_mma(&qw, &sc, m, k);
                    let (rw, rs, rg, rm) = if matches!(op, LoadShardOp::Row) {
                        let (q, s2, kl) = host_shard_nvfp4_row(&wt, &st, m, k, rank, world);
                        (q, s2, gsv.clone(), m)
                    } else {
                        match &op {
                            LoadShardOp::Col => host_shard_nvfp4_col_segs(&wt, &st, &gsv, k, &[(0, m, true)], rank, world),
                            LoadShardOp::ColSegs(segs) => host_shard_nvfp4_col_segs(&wt, &st, &gsv, k, segs, rank, world),
                            LoadShardOp::Row => unreachable!(),
                            LoadShardOp::None => unreachable!(),
                        }
                    };
                    assert_eq!(sm, rm, "op {op:?} world {world} rank {rank}: m mismatch");
                    assert_eq!(sk, if matches!(op, LoadShardOp::Row) { k / world } else { k },
                               "op {op:?} world {world} rank {rank}: local k wrong");
                    assert_eq!(wt2, rw, "op {op:?} world {world} rank {rank}: qweight mismatch ({} vs {})", wt2.len(), rw.len());
                    assert_eq!(st2, rs, "op {op:?} world {world} rank {rank}: REPACKED scales mismatch");
                    assert_eq!(sg, rg, "op {op:?} world {world} rank {rank}: gs mismatch");
                }
            }
        }
    }

    /// Partition + replication invariants: (a) concatenating every rank's Col slice reproduces the
    /// full tensor byte-for-byte; (b) a split=false segment is byte-identical across all ranks.
    #[test]
    fn raw_shard_partitions_and_replicates() {
        for world in [1usize, 2, 4, 8, 16] {
            let k = 256usize;
            let m_col = 512usize;                       // divides 16·world at every rung
            let qw = rng(m_col * k / 2, 7);
            let sc = rng(m_col * k / 16, 11);
            let gsv: Vec<f32> = (0..m_col / 16).map(|i| i as f32 * 0.5).collect();

            // (a) whole-M Col: concatenated rank slices == original.
            let mut cat_q = Vec::new(); let mut cat_s = Vec::new(); let mut cat_g = Vec::new();
            for rank in 0..world {
                let (sq, ss, sg, sm, sk) = shard_raw_nvfp4(&qw, &sc, &gsv, m_col, k, &LoadShardOp::Col, rank, world);
                assert_eq!((sm, sk), (m_col / world, k), "Col world {world} rank {rank}: local shape");
                cat_q.extend_from_slice(&sq); cat_s.extend_from_slice(&ss); cat_g.extend_from_slice(&sg);
            }
            assert_eq!(cat_q, qw, "Col world {world}: qweight partition");
            assert_eq!(cat_s, sc, "Col world {world}: scales partition");
            assert_eq!(cat_g, gsv, "Col world {world}: gs partition");

            // (b) split=false segment (16 rows = 1 output tile) is identical across ranks.
            let seg_w = 16 * world;
            let m_rep = 2 * seg_w + 16;                 // split segments divide; the 16-row tail is split=false
            let qw_rep = rng(m_rep * k / 2, 13);
            let sc_rep = rng(m_rep * k / 16, 17);
            let gsv_rep: Vec<f32> = (0..m_rep / 16).map(|i| i as f32 * 0.5).collect();
            let segs = vec![(0usize, seg_w, true), (seg_w, seg_w, true), (2 * seg_w, 16, false)];
            let mut first_replicated: Option<(Vec<u8>, Vec<u8>, Vec<f32>)> = None;
            for rank in 0..world {
                let (sq, ss, sg, _sm, _sk) =
                    shard_raw_nvfp4(&qw_rep, &sc_rep, &gsv_rep, m_rep, k, &LoadShardOp::ColSegs(segs.clone()), rank, world);
                // last split=false segment is the final output tile (16 rows)
                let (q_tail, s_tail, g_tail) = (
                    sq[sq.len() - (k / 16) * 128..].to_vec(),
                    ss[ss.len() - (k / 16) * 16..].to_vec(),
                    sg[sg.len() - 1..].to_vec(),
                );
                match &first_replicated {
                    None => first_replicated = Some((q_tail, s_tail, g_tail)),
                    Some((q0, s0, g0)) => {
                        assert_eq!(q0, &q_tail, "world {world} rank {rank}: replicated qweight tail differs");
                        assert_eq!(s0, &s_tail, "world {world} rank {rank}: replicated scales tail differs");
                        assert_eq!(g0, &g_tail, "world {world} rank {rank}: replicated gs tail differs");
                    }
                }
            }
        }
    }

    /// E8: the shared expert's FUSED [gate (si) | up (si)] ColSegs split (hy3 shapes) — rank r keeps
    /// gate[i]·up[i] for ITS OWN i-half (original rows [r·si/world, +si/world) of gate and
    /// [si + r·si/world, +si/world) of up), the local tensor is the standard fused layout with
    /// si_local = si/world, and slice-then-repack equals repack-then-slice (the shard-at-load invariant).
    #[test]
    fn e8_shared_gateup_pairing_and_invariant() {
        // si=1536 = 16·96, so it divides 16·world for world in {1,2,4,8,16} (16·16=256 ≤ 1536).
        let (si, h) = (1536usize, 5120usize);
        let (m, k) = (2 * si, h);
        let qw = rng(m * k / 2, 11);
        let sc = rng(m * k / 16, 13);
        let gsv: Vec<f32> = (0..m / 16).map(|i| i as f32 * 0.25).collect();
        let segs = vec![(0usize, si, true), (si, si, true)];
        let op = LoadShardOp::ColSegs(segs.clone());
        for world in [1usize, 2, 4, 8, 16] {
            let si_local = si / world;
            for rank in 0..world {
                let (sq, ss, sg, sm, sk) = shard_raw_nvfp4(&qw, &sc, &gsv, m, k, &op, rank, world);
                assert_eq!(sm, 2 * si_local, "world {world} rank {rank}: local fused M");
                assert_eq!(sk, k, "world {world} rank {rank}: K stays full (the col-shard splits M, not K)");
                // The pairing: local rows [0, si_local) are the rank's gate half and [si_local, 2·si_local)
                // its up half — local gate[i] and up[i] are the ORIGINAL gate[i]·up[i] pair.
                for r in 0..si_local {
                    let lo = r * (k / 2);
                    let ro = (rank * si_local + r) * (k / 2);
                    assert_eq!(&sq[lo..lo + k / 2], &qw[ro..ro + k / 2], "world {world} rank {rank}: gate row {r}");
                    let lo = (si_local + r) * (k / 2);
                    let ro = (si + rank * si_local + r) * (k / 2);
                    assert_eq!(&sq[lo..lo + k / 2], &qw[ro..ro + k / 2], "world {world} rank {rank}: up row {r}");
                }
                // The invariant for this exact segment shape.
                let (wt2, st2) = crate::quant::repack_nvfp4_mma(&sq, &ss, sm, sk);
                let (wt, st) = crate::quant::repack_nvfp4_mma(&qw, &sc, m, k);
                let (rw, rs, rg, rm) = host_shard_nvfp4_col_segs(&wt, &st, &gsv, k, &segs, rank, world);
                assert_eq!(sm, rm, "world {world} rank {rank}: m mismatch");
                assert_eq!(wt2, rw, "world {world} rank {rank}: qweight mismatch");
                assert_eq!(st2, rs, "world {world} rank {rank}: scales mismatch");
                assert_eq!(sg, rg, "world {world} rank {rank}: gs mismatch");
            }
        }
    }

    /// Block-128 FP8 (`W::Fp8Blk`) shard geometry. Both load paths share this function: the
    /// load-time path calls it directly, and the in-place path (`shard_fp8blk_col_segs` /
    /// `shard_fp8blk_row`, used when mixer sharding is off, e.g. `--dump-rank-weights --world N`)
    /// now delegates to it — before 2026-09-09 that path panicked on Fp8Blk.
    /// The property tested is a full ROUND TRIP: placing every rank's shard back at its source
    /// offset must reproduce the original codes AND the original scale grid exactly. The scale
    /// buffer starts as NaN so an unfilled element fails the comparison instead of passing empty.
    #[test]
    fn fp8blk_shard_roundtrip_reassembles_the_original() {
        let m = 2048usize;         // 16 bands of 128
        let k = 1024usize;         // nb_k = 8 -> row-shard valid up to world 8
        let nb_k = k / 128;
        let bands = m / 128;
        let qw = rng(m * k, 11);
        let sc: Vec<f32> = (0..bands * nb_k).map(|i| i as f32 * 0.25).collect();
        // Segment takes must stay 128-aligned at every world in the ladder: 1024/world is a
        // multiple of 128 for world <= 8. (A finer segment would legitimately panic — the
        // real Qwen qkv/GDN segments are head-multiples of 128 or more, so they pass.)
        let segs = vec![(0usize, 1024, true), (1024usize, 1024, true)];

        for world in [1usize, 2, 4, 8] {
            for op in [LoadShardOp::Col, LoadShardOp::ColSegs(segs.clone()), LoadShardOp::Row] {
                let mut rec_q = vec![0u8; m * k];
                let mut rec_s = vec![f32::NAN; bands * nb_k];
                let mut rows_total = 0usize;
                for rank in 0..world {
                    let (q, s, mo, ko, nb) = shard_raw_fp8_blk(&qw, &sc, m, k, nb_k, &op, rank, world);
                    let row_split = matches!(op, LoadShardOp::Row);
                    assert_eq!(ko, if row_split { k / world } else { k }, "op {op:?} world {world} rank {rank}: local k");
                    assert_eq!(nb, if row_split { nb_k / world } else { nb_k }, "op {op:?} world {world} rank {rank}: local nb_k");
                    assert_eq!(q.len(), mo * ko, "op {op:?} world {world} rank {rank}: code bytes");
                    assert_eq!(s.len(), (mo / 128) * nb, "op {op:?} world {world} rank {rank}: scale count");
                    if row_split {
                        assert_eq!(mo, m, "row shard keeps M");
                        for r in 0..m {
                            rec_q[r * k + rank * ko..r * k + (rank + 1) * ko]
                                .copy_from_slice(&q[r * ko..r * ko + ko]);
                        }
                        for b in 0..bands {
                            rec_s[b * nb_k + rank * nb..b * nb_k + (rank + 1) * nb]
                                .copy_from_slice(&s[b * nb..b * nb + nb]);
                        }
                    } else {
                        // Row ranges for this rank, derived test-side from the op (independent of
                        // the function under test), so a placement bug cannot cancel out.
                        let rows: Vec<(usize, usize)> = match &op {
                            LoadShardOp::Col => vec![(rank * (m / world), m / world)],
                            LoadShardOp::ColSegs(segs) => segs.iter().map(|&(off, cnt, split)| {
                                let take = if split { cnt / world } else { cnt };
                                (off + if split { rank * take } else { 0 }, take)
                            }).collect(),
                            _ => unreachable!(),
                        };
                        let (mut qoff, mut soff) = (0usize, 0usize);
                        for (r0, take) in rows {
                            rec_q[r0 * k..(r0 + take) * k].copy_from_slice(&q[qoff..qoff + take * k]);
                            qoff += take * k;
                            rec_s[(r0 / 128) * nb_k..((r0 + take) / 128) * nb_k]
                                .copy_from_slice(&s[soff..soff + (take / 128) * nb]);
                            soff += (take / 128) * nb;
                        }
                        assert_eq!(qoff, q.len(), "op {op:?} world {world} rank {rank}: codes not fully consumed");
                        assert_eq!(soff, s.len(), "op {op:?} world {world} rank {rank}: scales not fully consumed");
                        rows_total += mo;
                    }
                }
                assert_eq!(rec_q, qw, "op {op:?} world {world}: reconstructed codes differ");
                assert_eq!(rec_s, sc, "op {op:?} world {world}: reconstructed scales differ");
                if !matches!(op, LoadShardOp::Row) {
                    assert_eq!(rows_total, m, "op {op:?} world {world}: ranks do not tile M");
                }
            }
        }
    }

    /// A 128-row-misaligned ColSegs take must PANIC, not silently produce a scale band that
    /// straddles two segments (a silent numerical error is the failure mode this guards).
    #[test]
    #[should_panic(expected = "not 128-aligned")]
    fn fp8blk_col_seg_must_be_128_aligned() {
        let (m, k, nb_k) = (512usize, 1024usize, 8usize);
        let qw = rng(m * k, 3);
        let sc = vec![0f32; (m / 128) * nb_k];
        let _ = shard_raw_fp8_blk(&qw, &sc, m, k, nb_k,
                                  &LoadShardOp::ColSegs(vec![(0usize, 64, false)]), 0, 2);
    }

    /// A row split whose nb_k does not divide the world must PANIC (world 16 vs nb_k 8).
    #[test]
    #[should_panic(expected = "won't split across world")]
    fn fp8blk_row_shard_requires_divisible_nb_k() {
        let (m, k, nb_k) = (512usize, 1024usize, 8usize);
        let qw = rng(m * k, 3);
        let sc = vec![0f32; (m / 128) * nb_k];
        let _ = shard_raw_fp8_blk(&qw, &sc, m, k, nb_k, &LoadShardOp::Row, 0, 16);
    }
