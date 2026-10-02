//! TP-4H2 — the WORLD-4 vocab-parallel row ALL-GATHER (`--tp-vp-sampled`, sampled / penalized / ratio-rule
//! rows) as a pure schedule: launch tables, the device index math (mirrored here in Rust), and a CPU model
//! of the epoch + ring-slot protocol with the tests that pin it. No GPU, no CUDA types: everything the
//! launcher in `exl3_forward/xtp.rs::vp_gather_rows_w4` iterates comes from `gather4_plan`, so the tables
//! the tests check ARE the tables the device sees (the kernel bodies are checked on hardware, see
//! PLAN/TP-4H2_GATES.md).
//!
//! What is gathered. Each rank owns V/4 = 62,080 logits columns (the TP-H #4 shard GEMM; every column bitwise
//! the replicated head's, ks = 1, so the gather is a BIT COPY). The sampler needs the full `[m][V]` f16
//! rows, so the four shards are all-gathered into `sc.logits` and the UNCHANGED `xq_pen_rows` /
//! `argmax_rows` / `xq_sample_rows(_rq)` run on rows that are bit-for-bit the replicated head's.
//!
//! Schedule: recursive doubling over the decode doorbell, TWO exchanges per group of <= 4 rows.
//!   * `partner(e) = rank ^ (1 << (e % rounds))`, rounds = 2 (native/tp_doorbell.h, TP-4B). K1 pre-increments
//!     (`e = c->epoch + 1`), and the R10 invariant keeps the counter a multiple of `rounds` between ops, so a
//!     gather's first epoch is ODD (round 1, partner `rank ^ 2`). The kernels derive every mask from the
//!     epoch itself, so the schedule is correct from either phase (tested below) — the phase only decides
//!     WHICH pairing comes first.
//!   * Stage A (epoch e): the own shard `[nr][n_sh]` f16 goes to `rank ^ m0`, `m0 = 1 << (e % rounds)`.
//!   * Stage B (epoch e + 1): the two shards held after A, `{rank, rank ^ m0}`, go to `rank ^ m1`,
//!     `m1 = 1 << ((e + 1) % rounds)` (`!= m0`). Because the FIRST mask decides the pairing, the stage-B
//!     block is {own, own ^ m0}: two shards that are NOT adjacent in the column space when m0 = 2.
//!     The kernels therefore index by shard number, never by "a contiguous 2-shard block".
//!   * Stage B's payload is 2 x nr x n_sh x 2 B = 993,280 B at nr = 4 (+ an 8 B tail): that, not the
//!     receive-slot rule, limits a group to 4 rows (5 rows = 1,241,600 B > 1 MiB).
//!   * Per group the stream order is K1(A) K2(A) K1(B) K2(B), every K2 with lag 0: stage B's K1 reads
//!     columns that stage A's K2 wrote (a hard data dependency), and the serial lookahead-1 pattern is the
//!     one whose recv-slot safety is already proven for the decode arms (`serial_schedule_recv_slot_safety`
//!     in tp_xport.rs; re-proved here for the real gather programs).
//!   * Epochs per gather = 2 * ceil(m / 4) (2, 4, 6, 8 for m = 1..4, 5..8, 9..12, 13..16): always a multiple
//!     of `rounds`, so the R10 phase is invariant across gathers and decode arms.

use crate::tp_xport::RING_SLOTS;

/// The one world this schedule serves. World 2 keeps the TP-H2 kernels (`xq_vp_gather_k1/k2`) untouched.
pub const WORLD: usize = 4;
/// Recursive-doubling rounds at world 4 (log2).
pub const ROUNDS: usize = 2;
/// Rows per group: 4 x 2 shards x 62,080 x 2 B + 8 B tail = 993,288 B fits one 1 MiB ring slot.
pub const GROUP_ROWS: usize = 4;
/// Verify width cap of the vp tails (`vp_greedy` / `vp_gather`: `(1..=16).contains(&m)`).
pub const MAX_ROWS: usize = 16;
/// native/tp_doorbell.h TP_RING_SLOTS.
pub const RING: usize = RING_SLOTS;
/// uint4 units one thread block moves (256 threads x 8): the W=2 launch-shape constant, reused.
pub const UNITS_PER_BLOCK: usize = 256 * 8;
/// Block cap of the gather kernels (the W=2 clamp).
pub const MAX_BLOCKS: u32 = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Half {
    K1,
    K2,
}

/// One kernel launch of the gather. `stage` 0 = A (own shard, partner `rank ^ m0`), 1 = B (two shards).
/// `lag` is the K2's `c->epoch - lag` offset (always 0 in the serial plan; the field exists so the model
/// can also run the W=2-style lagged shape as a negative control).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Launch {
    pub half: Half,
    pub stage: u32,
    pub r0: usize,
    pub nr: usize,
    pub blocks: u32,
    pub lag: u32,
}

/// Row groups of an m-row gather: `(r0, nr)`, nr <= GROUP_ROWS, in order, partitioning `0..m`.
pub fn groups(m: usize) -> Vec<(usize, usize)> {
    (0..m.div_ceil(GROUP_ROWS)).map(|j| (j * GROUP_ROWS, GROUP_ROWS.min(m - j * GROUP_ROWS))).collect()
}

/// Device epochs one m-row gather consumes (two per group).
pub fn epochs_per_gather(m: usize) -> usize {
    2 * m.div_ceil(GROUP_ROWS)
}

/// uint4 units (8 f16 = 16 B) in one group's payload: stage A carries 1 shard, stage B 2.
pub fn payload_units(nr: usize, rq: usize, stage: u32) -> usize {
    nr * rq * (stage as usize + 1)
}

/// Payload bytes of one epoch (the 8 B generation tail sits after it, `align8(nbytes)`).
pub fn payload_bytes(nr: usize, n_sh: usize, stage: u32) -> usize {
    payload_units(nr, n_sh / 8, stage) * 16
}

/// Thread blocks of a launch: the W=2 rule `(units / 2048).ceil().clamp(1, 32)` applied to the stage's payload.
pub fn blocks_for(nr: usize, rq: usize, stage: u32) -> u32 {
    (payload_units(nr, rq, stage).div_ceil(UNITS_PER_BLOCK) as u32).clamp(1, MAX_BLOCKS)
}

/// Does a `rows`-row group's stage-B payload (+ the 8 B tail) fit a ring slot of `slot_bytes`?
pub fn slot_fits(rows: usize, n_sh: usize, slot_bytes: usize) -> bool {
    payload_bytes(rows, n_sh, 1) + 8 <= slot_bytes
}

/// THE LAUNCH TABLE: per group, K1(A) K2(A) K1(B) K2(B), all lag 0. `n_sh % 8 == 0`, `1 <= m <= 16`
/// (host-checked by the launcher).
pub fn gather4_plan(m: usize, n_sh: usize) -> Vec<Launch> {
    let rq = n_sh / 8;
    let mut v = Vec::with_capacity(4 * m.div_ceil(GROUP_ROWS));
    for (r0, nr) in groups(m) {
        for stage in 0..2u32 {
            let blocks = blocks_for(nr, rq, stage);
            v.push(Launch { half: Half::K1, stage, r0, nr, blocks, lag: 0 });
            v.push(Launch { half: Half::K2, stage, r0, nr, blocks, lag: 0 });
        }
    }
    v
}

// ---- the device index math, mirrored (kernels/exl3_bench.cu: xq_vp_gather4_k1 / _k2) -------------------

/// Partner mask of epoch `e` (`xtp_partner`: `rank ^ (1 << (e % rounds))`).
pub fn mask(e: u64, rounds: usize) -> usize {
    1usize << (e % rounds as u64)
}

/// The mask of epoch `e - 1` (written without an underflow): stage B's "first" pairing.
pub fn first_mask(e: u64, rounds: usize) -> usize {
    1usize << ((e + rounds as u64 - 1) % rounds as u64)
}

/// Shards carried by rank `r`'s payload at stage `stage` of epoch `e`, in slot order (block 0 first).
pub fn payload_shards(r: usize, stage: u32, e: u64, rounds: usize) -> Vec<usize> {
    match stage {
        0 => vec![r],
        _ => vec![r, r ^ first_mask(e, rounds)],
    }
}

/// Split a payload unit index into (block, row-in-group, uint4-in-row): `i = (blk * nr + row) * rq + j`.
pub fn unit_decode(i: usize, nr: usize, rq: usize) -> (usize, usize, usize) {
    let per = nr * rq;
    let blk = i / per;
    let rem = i - blk * per;
    let row = rem / rq;
    (blk, row, rem - row * rq)
}

/// Deterministic f16 bit pattern of the synthetic shard element (shard `s`, row, global column): the bench
/// fills each rank's shard rows with this (position dependent, so a misplaced column can never match) and
/// every rank checks the assembled rows against it. Not a NaN filter: the gather moves raw bits.
pub fn ref_half(s: usize, row: usize, col: usize) -> u16 {
    let x = (s as u64) * 0x9E37_79B9_7F4A_7C15 ^ (row as u64 + 1) * 0xBF58_476D_1CE4_E5B9 ^ (col as u64 + 7) * 0x94D0_49BB_1331_11EB;
    let x = (x ^ (x >> 29)).wrapping_mul(0xD6E8_FEB8_6659_FD93);
    (x >> 40) as u16
}

// ---- the W=2 reference (TP-H2), transcribed so a test can pin it -------------------------------------

/// The WORLD-2 launch table exactly as `vp_gather_rows` computes it inline today: every K1 (4-row groups),
/// then every K2 with `lag = ne - 1 - j`; blocks = `((nr * n_sh / 8).div_ceil(2048)).clamp(1, 32)`.
/// Pinned against literals and against the function body's text hash in the tests; the W=2 path itself is
/// NOT routed through this (it stays byte-identical code).
pub fn w2_plan(m: usize, n_sh: usize) -> Vec<(Half, usize, usize, u32, u32)> {
    let ne = m.div_ceil(GROUP_ROWS);
    let blocks = |nr: usize| ((nr * n_sh / 8).div_ceil(256 * 8) as u32).clamp(1, 32);
    let mut v = Vec::new();
    for j in 0..ne {
        let (r0, nr) = (j * GROUP_ROWS, GROUP_ROWS.min(m - j * GROUP_ROWS));
        v.push((Half::K1, r0, nr, blocks(nr), 0u32));
    }
    for j in 0..ne {
        let (r0, nr) = (j * GROUP_ROWS, GROUP_ROWS.min(m - j * GROUP_ROWS));
        v.push((Half::K2, r0, nr, blocks(nr), (ne - 1 - j) as u32));
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashSet, VecDeque};

    const N_SH: usize = 62080;
    const RQ: usize = N_SH / 8;

    // ------------------------------------------------------------------------------------------------
    // plan shape, golden launch tables, slot capacity
    // ------------------------------------------------------------------------------------------------

    #[test]
    fn slot_capacity_is_why_a_group_is_four_rows() {
        assert_eq!(payload_bytes(4, N_SH, 0), 496_640);
        assert_eq!(payload_bytes(4, N_SH, 1), 993_280);
        assert!(slot_fits(4, N_SH, crate::tp::TP_SLOT_BYTES));
        assert!(!slot_fits(5, N_SH, crate::tp::TP_SLOT_BYTES), "5 rows would be {} B", payload_bytes(5, N_SH, 1));
        assert_eq!(payload_bytes(5, N_SH, 1), 1_241_600);
        // the W=2 group has the same byte size by coincidence (1 shard of 124,160 x 4 rows)
        assert_eq!(4 * 124_160 * 2, 993_280);
        assert_eq!(RING, 8);
        assert_eq!(WORLD.trailing_zeros() as usize, ROUNDS);
    }

    /// The exact launch table for m = 1..16 at n_sh = 62,080 (rq = 7760): (r0, nr, blocks A, blocks B) per group.
    /// blocks = ceil(units / 2048).clamp(1, 32): nr 1 -> 4 / 8, nr 2 -> 8 / 16, nr 3 -> 12 / 23, nr 4 -> 16 / 31.
    #[test]
    fn gather4_launch_tables_m1_to_m16() {
        let per_nr = |nr: usize| -> (u32, u32) {
            match nr { 1 => (4, 8), 2 => (8, 16), 3 => (12, 23), 4 => (16, 31), _ => panic!("nr {nr}") }
        };
        for nr in 1..=4 {
            assert_eq!((blocks_for(nr, RQ, 0), blocks_for(nr, RQ, 1)), per_nr(nr));
        }
        for m in 1..=MAX_ROWS {
            let plan = gather4_plan(m, N_SH);
            let ne = m.div_ceil(4);
            assert_eq!(plan.len(), 4 * ne, "m {m}");
            let mut want: Vec<Launch> = Vec::new();
            let mut r0 = 0;
            while r0 < m {
                let nr = 4.min(m - r0);
                let (ba, bb) = per_nr(nr);
                for (stage, b) in [(0u32, ba), (1u32, bb)] {
                    want.push(Launch { half: Half::K1, stage, r0, nr, blocks: b, lag: 0 });
                    want.push(Launch { half: Half::K2, stage, r0, nr, blocks: b, lag: 0 });
                }
                r0 += nr;
            }
            assert_eq!(plan, want, "m {m}");
            assert_eq!(groups(m).iter().map(|g| g.1).sum::<usize>(), m);
            assert_eq!(epochs_per_gather(m), 2 * ne);
            assert_eq!(plan.iter().filter(|l| l.half == Half::K1).count(), epochs_per_gather(m), "one epoch per K1");
            assert_eq!(epochs_per_gather(m) % ROUNDS, 0, "m {m}: epochs per gather must be a multiple of rounds");
            assert!(plan.iter().all(|l| l.blocks >= 1 && l.blocks <= MAX_BLOCKS && l.lag == 0));
        }
        // the group-count changes are where the epoch count steps
        let e: Vec<usize> = (1..=16).map(epochs_per_gather).collect();
        assert_eq!(e, vec![2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6, 8, 8, 8, 8]);
    }

    // ------------------------------------------------------------------------------------------------
    // W=2 invariance pins
    // ------------------------------------------------------------------------------------------------

    /// FNV-1a 64 over the UTF-8 text.
    fn fnv(s: &str) -> u64 {
        s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3))
    }

    /// `text[start ..= first end_pat after start]`.
    fn extract<'a>(text: &'a str, start: &str, end_pat: &str) -> &'a str {
        let i = text.find(start).unwrap_or_else(|| panic!("`{start}` not found"));
        let j = text[i..].find(end_pat).unwrap_or_else(|| panic!("end of `{start}` not found")) + i + end_pat.len();
        &text[i..j]
    }

    /// The W=2 gather is untouched: the text of the host launcher, both kernels and both multi-block gates
    /// hashes to the value measured at the base commit 4f6d6df (before TP-4H2). Editing any of them fails
    /// here; a deliberate W=2 change must update the hash in the same commit and say why.
    #[test]
    fn w2_gather_code_is_byte_identical_to_the_base_commit() {
        let xtp = include_str!("exl3_forward/xtp.rs");
        let cu = include_str!("../kernels/exl3_bench.cu");
        let cases: [(&str, &str, &str, &str, usize, u64); 5] = [
            // vp_gather_rows: len/hash re-pinned for d6ac6c4, which changed ONLY the world-2 `ensure!` message text (no logic).
            ("vp_gather_rows", xtp, "    pub(super) fn vp_gather_rows(", "\n    }\n", 2401, 0x43f198b746bfb54e),
            ("xq_vp_gather_k1", cu, "extern \"C\" __global__ void __launch_bounds__(256) xq_vp_gather_k1(", "\n}\n", 2290, 0x0dc64682830e60da),
            ("xq_vp_gather_k2", cu, "extern \"C\" __global__ void __launch_bounds__(256) xq_vp_gather_k2(", "\n}\n", 1091, 0xc9bc24fe24c374c0),
            ("xtp_k1_gate_mb", cu, "__device__ __forceinline__ int xtp_k1_gate_mb(", "\n}\n", 968, 0xd2be2b4fdff7280c),
            ("xtp_k2_gate_mb", cu, "__device__ __forceinline__ int xtp_k2_gate_mb(", "\n}\n", 1015, 0x58a7d4a384957301),
        ];
        for (name, text, start, end, len, hash) in cases {
            let body = extract(text, start, end);
            assert_eq!(body.len(), len, "{name}: length changed");
            assert_eq!(fnv(body), hash, "{name}: body text changed (W=2 must stay byte-identical)");
        }
    }

    /// The W=2 launch table, literally, for every m: K1 of each 4-row group, then K2 with lag ne-1-j; blocks
    /// from the 124,160-column shard. (Transcription of the inline loops of `vp_gather_rows`; the body hash
    /// above ties the transcription to the code.)
    #[test]
    fn w2_launch_tables_are_the_old_ones() {
        const N2: usize = 124_160;
        // (nr) -> blocks at W=2: units = nr * 15520 -> 8 / 16 / 23 / 31 (clamp 32 never binds)
        let b = |nr: usize| match nr { 1 => 8u32, 2 => 16, 3 => 23, 4 => 31, _ => unreachable!() };
        for m in 1..=16usize {
            let ne = m.div_ceil(4);
            let mut want = Vec::new();
            for j in 0..ne {
                let nr = 4.min(m - 4 * j);
                want.push((Half::K1, 4 * j, nr, b(nr), 0u32));
            }
            for j in 0..ne {
                let nr = 4.min(m - 4 * j);
                want.push((Half::K2, 4 * j, nr, b(nr), (ne - 1 - j) as u32));
            }
            assert_eq!(w2_plan(m, N2), want, "m {m}");
        }
        // spot literals: m = 8 is K1(0,4) K1(4,4) K2(0,4,lag 1) K2(4,4,lag 0), 31 blocks each
        assert_eq!(
            w2_plan(8, N2),
            vec![(Half::K1, 0, 4, 31, 0), (Half::K1, 4, 4, 31, 0), (Half::K2, 0, 4, 31, 1), (Half::K2, 4, 4, 31, 0)]
        );
        assert_eq!(w2_plan(1, N2), vec![(Half::K1, 0, 1, 8, 0), (Half::K2, 0, 1, 8, 0)]);
    }

    // ------------------------------------------------------------------------------------------------
    // the device index math
    // ------------------------------------------------------------------------------------------------

    /// After the two stages, rank r holds every shard exactly once: own (A-K1), the A partner's, and the two
    /// the B partner forwards — for either phase of the counter and every rank.
    #[test]
    fn shards_partition_across_the_two_stages() {
        for e_a in [1u64, 2, 3, 4, 9, 10, 101, 102] {
            for r in 0..WORLD {
                let e_b = e_a + 1;
                let pa = r ^ mask(e_a, ROUNDS);
                let pb = r ^ mask(e_b, ROUNDS);
                assert_ne!(mask(e_a, ROUNDS), mask(e_b, ROUNDS));
                let recv_a = payload_shards(pa, 0, e_a, ROUNDS);
                let recv_b = payload_shards(pb, 1, e_b, ROUNDS);
                // what rank r's stage-B payload is, versus what it actually holds after stage A
                let held_after_a: HashSet<usize> = [r].into_iter().chain(recv_a.iter().copied()).collect();
                let sends_b: HashSet<usize> = payload_shards(r, 1, e_b, ROUNDS).into_iter().collect();
                assert_eq!(sends_b, held_after_a, "e_a {e_a} rank {r}: B must forward exactly what A left it holding");
                let mut all: Vec<usize> = vec![r];
                all.extend(recv_a.iter());
                all.extend(recv_b.iter());
                all.sort_unstable();
                assert_eq!(all, vec![0, 1, 2, 3], "e_a {e_a} rank {r}: shards after A+B");
                // the odd-phase pairing carries a NON-contiguous block in B (m0 = 2 => {r, r^2})
                if mask(e_a, ROUNDS) == 2 {
                    assert_eq!((r ^ 2).abs_diff(r), 2);
                }
            }
        }
    }

    #[test]
    fn unit_decode_is_a_bijection_on_the_payload() {
        for (nr, rq, stage) in [(1usize, 7usize, 0u32), (4, 7, 1), (3, 13, 1), (4, RQ, 1)] {
            let n = payload_units(nr, rq, stage);
            let mut seen = vec![false; n];
            for i in 0..n {
                let (blk, row, j) = unit_decode(i, nr, rq);
                assert!(blk <= stage as usize && row < nr && j < rq);
                let back = (blk * nr + row) * rq + j;
                assert_eq!(back, i);
                assert!(!seen[back]);
                seen[back] = true;
            }
        }
    }

    // ------------------------------------------------------------------------------------------------
    // the program, the epoch accounting and the CPU model of the protocol
    // ------------------------------------------------------------------------------------------------

    #[derive(Clone, Copy, Debug)]
    enum Step {
        K1(Launch, usize),
        K2(Launch, usize),
        T1,
        T2(u32),
    }

    #[derive(Clone, Copy, Debug)]
    enum Op {
        /// an m-row vocab gather through the real `gather4_plan`
        Gather(usize),
        /// one decode-path LOGICAL reduce = `rounds` exchanges (R10: every logical op consumes a multiple
        /// of `rounds` epochs), each a plain K1 + K2 of a one-unit token
        Dec,
        /// lookahead-L token schedule: L K1s, then L K2s with lag L-1..0 (negative controls / the rule)
        Synth(usize),
    }

    fn program(ops: &[Op], rq: usize) -> (Vec<Step>, Vec<usize>) {
        let mut p = Vec::new();
        let mut gm = Vec::new();
        for op in ops {
            match *op {
                Op::Gather(m) => {
                    let gid = gm.len();
                    gm.push(m);
                    for l in gather4_plan(m, rq * 8) {
                        p.push(if l.half == Half::K1 { Step::K1(l, gid) } else { Step::K2(l, gid) });
                    }
                }
                Op::Dec => {
                    for _ in 0..ROUNDS {
                        p.push(Step::T1);
                        p.push(Step::T2(0));
                    }
                }
                Op::Synth(l) => {
                    for _ in 0..l {
                        p.push(Step::T1);
                    }
                    for j in 0..l {
                        p.push(Step::T2((l - 1 - j) as u32));
                    }
                }
            }
        }
        (p, gm)
    }

    /// Per program step: (is K1, the epoch it publishes / consumes), from a counter starting at `e0`.
    fn step_epochs(prog: &[Step], e0: u64) -> Vec<(bool, u64)> {
        let mut e = e0;
        prog.iter()
            .map(|s| match *s {
                Step::K1(..) | Step::T1 => {
                    e += 1;
                    (true, e)
                }
                Step::K2(l, _) => (false, e - l.lag as u64),
                Step::T2(lag) => (false, e - lag as u64),
            })
            .collect()
    }

    #[test]
    fn every_prefix_of_ops_keeps_the_round_phase() {
        let ops = [Op::Gather(1), Op::Dec, Op::Gather(8), Op::Gather(16), Op::Dec, Op::Dec, Op::Gather(5), Op::Gather(13), Op::Gather(4)];
        for e0 in [0u64, 2, 6, 8, 30] {
            let mut e = e0;
            for op in ops {
                let (p, _) = program(&[op], RQ);
                let k1s = step_epochs(&p, 0).iter().filter(|s| s.0).count() as u64;
                e += k1s;
                assert_eq!(e % ROUNDS as u64, e0 % ROUNDS as u64, "{op:?} moved the phase");
            }
        }
        for m in 1..=16 {
            let (p, _) = program(&[Op::Gather(m)], RQ);
            let k1s = step_epochs(&p, 0).iter().filter(|s| s.0).count();
            assert_eq!(k1s, epochs_per_gather(m));
            assert_eq!(k1s % ROUNDS, 0);
        }
    }

    // ---- the happens-before DAG proof of recv-slot safety -------------------------------------------

    /// Per-rank stream order + the wire edge K1_partner(e) -> K2_r(e). Slot rule: epochs e and e - R share a
    /// recv slot (round, e & (R-1)); the write of epoch e into rank r's slot (its partner's K1(e)) must be
    /// ordered AFTER rank r's K2(e - R) read. Returns the first violation.
    fn dag_slot_safety(prog: &[Step], e0: u64) -> Result<(), String> {
        let ep = step_epochs(prog, e0);
        let n = prog.len();
        let node = |r: usize, pc: usize| r * n + pc;
        let nodes = WORLD * n;
        let mut k1_at = std::collections::HashMap::new();
        let mut k2_at = std::collections::HashMap::new();
        for (pc, &(is_k1, e)) in ep.iter().enumerate() {
            if is_k1 { k1_at.insert(e, pc); } else { k2_at.insert(e, pc); }
        }
        let mut succ: Vec<Vec<usize>> = vec![Vec::new(); nodes];
        for r in 0..WORLD {
            for pc in 0..n {
                if pc + 1 < n { succ[node(r, pc)].push(node(r, pc + 1)); }
                let (is_k1, e) = ep[pc];
                if !is_k1 {
                    let p = r ^ mask(e, ROUNDS);
                    let w = *k1_at.get(&e).ok_or_else(|| format!("K2 of epoch {e} has no K1"))?;
                    succ[node(p, w)].push(node(r, pc));
                }
            }
        }
        let reaches = |from: usize, to: usize| {
            let mut seen = vec![false; nodes];
            let mut st = vec![from];
            seen[from] = true;
            while let Some(x) = st.pop() {
                if x == to { return true; }
                for &y in &succ[x] {
                    if !seen[y] { seen[y] = true; st.push(y); }
                }
            }
            false
        };
        let last = ep.iter().map(|x| x.1).max().unwrap_or(e0);
        for r in 0..WORLD {
            for e in (e0 + RING as u64 + 1)..=last {
                let (Some(&w), Some(&rd)) = (k1_at.get(&e), k2_at.get(&(e - RING as u64))) else { continue };
                let writer = node(r ^ mask(e, ROUNDS), w);
                let reader = node(r, rd);
                if !reaches(reader, writer) {
                    return Err(format!("rank {r} epoch {e}: the slot write is not ordered after the K2 read of epoch {}", e - RING as u64));
                }
            }
        }
        Ok(())
    }

    #[test]
    fn dag_proves_recv_slot_safety_of_the_real_gather_programs() {
        let mixes: [&[Op]; 5] = [
            &[Op::Gather(1); 12],
            &[Op::Gather(4); 6],
            &[Op::Gather(8), Op::Dec, Op::Gather(16), Op::Dec, Op::Dec, Op::Gather(5), Op::Gather(13), Op::Gather(2), Op::Gather(9)],
            &[Op::Gather(16), Op::Gather(16), Op::Gather(16), Op::Gather(16)],
            &[Op::Dec, Op::Gather(3), Op::Dec, Op::Gather(7), Op::Dec, Op::Gather(11), Op::Dec, Op::Gather(15), Op::Dec, Op::Gather(6)],
        ];
        for ops in mixes {
            for e0 in [0u64, 2, 8, 14] {
                let (p, _) = program(ops, RQ);
                dag_slot_safety(&p, e0).unwrap_or_else(|e| panic!("{ops:?} e0 {e0}: {e}"));
            }
        }
    }

    /// The decode arms' own serial schedule passes the same checker (cross-check of the checker against the
    /// existing proof in tp_xport.rs), and the lookahead rule is exactly what the model says: serial (L=1)
    /// and L <= R/2 that are multiples of rounds are safe; L above R/2 is not.
    #[test]
    fn dag_lookahead_rule_and_negative_controls() {
        let run = |l: usize| dag_slot_safety(&program(&[Op::Synth(l); 12], RQ).0, 0);
        assert!(dag_slot_safety(&program(&[Op::Dec; 40], RQ).0, 0).is_ok(), "serial T1 T2 pairs");
        assert!(run(2).is_ok(), "L = 2 = rounds");
        assert!(run(4).is_ok(), "L = 4 = R/2, a multiple of rounds");
        assert!(run(6).is_err(), "L = 6 > R/2 must be refused by the checker");
        assert!(run(8).is_err(), "L = 8 = R must be refused by the checker");
    }

    // ---- the protocol simulator ------------------------------------------------------------------------

    #[derive(Clone)]
    struct Slot {
        gen: u64,
        data: Vec<u32>,
    }

    struct Rank {
        pc: usize,
        epoch: u64,
        shard: Vec<u32>,
        logits: Vec<u32>,
        written: Vec<u8>,
        cur_gid: usize,
        send: Vec<Slot>,
        recv: Vec<Slot>,
        consumed: HashSet<u64>,
        queue: Vec<VecDeque<u64>>,
        retired: Vec<u64>,
    }

    const SENTINEL: u32 = 0xFFFF_FFFF;

    fn val(s: usize, row: usize, j: usize) -> u32 {
        ((s as u32) << 28) | ((row as u32) << 20) | j as u32
    }

    fn tok(r: usize, e: u64) -> u32 {
        0xD000_0000 | ((r as u32) << 20) | (e as u32 & 0xF_FFFF)
    }

    struct Sim {
        prog: Vec<Step>,
        gm: Vec<usize>,
        rq: usize,
        ranks: Vec<Rank>,
        events: u64,
        max_ahead: u64,
    }

    impl Sim {
        fn new(prog: Vec<Step>, gm: Vec<usize>, rq: usize, e0: u64) -> Sim {
            let ranks = (0..WORLD)
                .map(|r| Rank {
                    pc: 0,
                    epoch: e0,
                    shard: (0..MAX_ROWS * rq).map(|i| val(r, i / rq, i % rq)).collect(),
                    logits: vec![SENTINEL; MAX_ROWS * WORLD * rq],
                    written: vec![0; MAX_ROWS * WORLD * rq],
                    cur_gid: usize::MAX,
                    send: vec![Slot { gen: 0, data: vec![] }; RING],
                    recv: vec![Slot { gen: 0, data: vec![] }; ROUNDS * RING],
                    consumed: HashSet::new(),
                    queue: vec![VecDeque::new(); WORLD],
                    retired: vec![e0; WORLD],
                })
                .collect();
            Sim { prog, gm, rq, ranks, events: 0, max_ahead: 0 }
        }

        fn peer(r: usize, e: u64) -> usize {
            r ^ mask(e, ROUNDS)
        }

        fn enabled(&self, r: usize) -> bool {
            let rk = &self.ranks[r];
            match self.prog[rk.pc] {
                Step::K1(..) | Step::T1 => {
                    let e = rk.epoch + 1;
                    // I3: the send slot of epoch e - R must have been retired to the same peer
                    e <= RING as u64 || rk.retired[Self::peer(r, e)] >= e - RING as u64
                }
                Step::K2(l, _) => self.recv_ready(r, rk.epoch - l.lag as u64),
                Step::T2(lag) => self.recv_ready(r, rk.epoch - lag as u64),
            }
        }

        fn recv_ready(&self, r: usize, e: u64) -> bool {
            self.ranks[r].recv[(e % ROUNDS as u64) as usize * RING + (e as usize & (RING - 1))].gen == e
        }

        fn step(&mut self, r: usize) -> Result<(), String> {
            let rq = self.rq;
            let pc = self.ranks[r].pc;
            match self.prog[pc] {
                Step::K1(l, gid) => {
                    let e = self.ranks[r].epoch + 1;
                    self.begin_gather(r, gid);
                    let rk = &mut self.ranks[r];
                    let mut data = Vec::with_capacity(payload_units(l.nr, rq, l.stage));
                    if l.stage == 0 {
                        for row in 0..l.nr {
                            for j in 0..rq {
                                let v = rk.shard[(l.r0 + row) * rq + j];
                                data.push(v);
                                let at = (l.r0 + row) * WORLD * rq + r * rq + j;
                                rk.logits[at] = v;
                                rk.written[at] += 1;
                            }
                        }
                    } else {
                        for s in payload_shards(r, 1, e, ROUNDS) {
                            for row in 0..l.nr {
                                for j in 0..rq {
                                    let at = (l.r0 + row) * WORLD * rq + s * rq + j;
                                    if rk.written[at] == 0 {
                                        return Err(format!("rank {r} epoch {e}: K1(B) reads shard {s} row {} unit {j} BEFORE it was written", l.r0 + row));
                                    }
                                    data.push(rk.logits[at]);
                                }
                            }
                        }
                    }
                    self.publish(r, e, data);
                }
                Step::K2(l, gid) => {
                    let e = self.ranks[r].epoch - l.lag as u64;
                    let p = Self::peer(r, e);
                    let idx = (e % ROUNDS as u64) as usize * RING + (e as usize & (RING - 1));
                    let slot = self.ranks[r].recv[idx].clone();
                    let rk = &mut self.ranks[r];
                    let shards = payload_shards(p, l.stage, e, ROUNDS);
                    if slot.data.len() != payload_units(l.nr, rq, l.stage) {
                        return Err(format!("rank {r} epoch {e}: K2 payload has {} units, stage {} nr {} wants {}",
                                           slot.data.len(), l.stage, l.nr, payload_units(l.nr, rq, l.stage)));
                    }
                    for (i, &v) in slot.data.iter().enumerate() {
                        let (blk, row, j) = unit_decode(i, l.nr, rq);
                        let at = (l.r0 + row) * WORLD * rq + shards[blk] * rq + j;
                        rk.logits[at] = v;
                        rk.written[at] += 1;
                    }
                    rk.consumed.insert(e);
                    self.finish_gather_if_last(r, pc, gid)?;
                }
                Step::T1 => {
                    let e = self.ranks[r].epoch + 1;
                    self.publish(r, e, vec![tok(r, e)]);
                }
                Step::T2(lag) => {
                    let e = self.ranks[r].epoch - lag as u64;
                    let idx = (e % ROUNDS as u64) as usize * RING + (e as usize & (RING - 1));
                    let got = self.ranks[r].recv[idx].data.first().copied();
                    let want = tok(Self::peer(r, e), e);
                    if got != Some(want) {
                        return Err(format!("rank {r} epoch {e}: token {got:?}, want {want:#x} (stale or wrong slot)"));
                    }
                    self.ranks[r].consumed.insert(e);
                }
            }
            self.ranks[r].pc += 1;
            Ok(())
        }

        fn begin_gather(&mut self, r: usize, gid: usize) {
            let rk = &mut self.ranks[r];
            if rk.cur_gid != gid {
                rk.cur_gid = gid;
                for w in rk.written.iter_mut() { *w = 0; }
            }
        }

        fn publish(&mut self, r: usize, e: u64, data: Vec<u32>) {
            let rk = &mut self.ranks[r];
            rk.send[e as usize & (RING - 1)] = Slot { gen: e, data };
            rk.epoch = e;
            rk.queue[Self::peer(r, e)].push_back(e);
        }

        /// the proxy: deliver the head epoch of (sender p -> peer) into the peer's recv slot
        fn deliver(&mut self, p: usize, peer: usize) -> Result<(), String> {
            let e = self.ranks[p].queue[peer].pop_front().unwrap();
            let src = self.ranks[p].send[e as usize & (RING - 1)].clone();
            if src.gen != e {
                return Err(format!("rank {p} epoch {e}: the SEND slot was overwritten (gen {}) before the proxy sent it", src.gen));
            }
            let idx = (e % ROUNDS as u64) as usize * RING + (e as usize & (RING - 1));
            let old_gen = self.ranks[peer].recv[idx].gen;
            if old_gen != 0 && !self.ranks[peer].consumed.contains(&old_gen) {
                return Err(format!("rank {peer}: delivery of epoch {e} from rank {p} OVERWRITES the recv slot of unconsumed epoch {old_gen}"));
            }
            self.ranks[peer].recv[idx] = src;
            self.ranks[p].retired[peer] = e;
            Ok(())
        }

        fn finish_gather_if_last(&mut self, r: usize, pc: usize, gid: usize) -> Result<(), String> {
            let last = match self.prog.get(pc + 1) {
                Some(Step::K1(_, g)) | Some(Step::K2(_, g)) => *g != gid,
                _ => true,
            };
            if !last { return Ok(()); }
            let (m, rq) = (self.gm[gid], self.rq);
            let rk = &self.ranks[r];
            for row in 0..MAX_ROWS {
                for s in 0..WORLD {
                    for j in 0..rq {
                        let at = row * WORLD * rq + s * rq + j;
                        if row < m {
                            if rk.written[at] != 1 {
                                return Err(format!("rank {r} gather {gid}: (row {row}, shard {s}, unit {j}) written {} times", rk.written[at]));
                            }
                            if rk.logits[at] != val(s, row, j) {
                                return Err(format!("rank {r} gather {gid}: (row {row}, shard {s}, unit {j}) holds {:#x}, want {:#x}", rk.logits[at], val(s, row, j)));
                            }
                        } else if rk.written[at] != 0 {
                            return Err(format!("rank {r} gather {gid}: row {row} >= m {m} was written"));
                        }
                    }
                }
            }
            Ok(())
        }

        fn rng(state: &mut u64) -> u64 {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            *state
        }

        /// Random interleaving of rank steps and proxy deliveries until every rank ran its program.
        /// `step_pct` = probability (percent) that a rank step is chosen over a delivery when both are
        /// possible: high values let the ranks race ahead of the wire (the adversarial direction for slot
        /// reuse), low values deliver eagerly.
        fn run(&mut self, seed: u64, step_pct: u64) -> Result<(), String> {
            let mut st = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
            loop {
                let mut steps = Vec::new();
                for r in 0..WORLD {
                    if self.ranks[r].pc < self.prog.len() && self.enabled(r) { steps.push(r); }
                }
                let mut dels = Vec::new();
                for p in 0..WORLD {
                    for q in 0..WORLD {
                        if !self.ranks[p].queue[q].is_empty() { dels.push((p, q)); }
                    }
                }
                let done = self.ranks.iter().all(|r| r.pc >= self.prog.len());
                if steps.is_empty() && dels.is_empty() {
                    return if done { Ok(()) } else { Err("DEADLOCK: no rank step or delivery is enabled".to_string()) };
                }
                self.events += 1;
                let pick_step = !steps.is_empty() && (dels.is_empty() || Self::rng(&mut st) % 100 < step_pct);
                if pick_step {
                    let r = steps[(Self::rng(&mut st) % steps.len() as u64) as usize];
                    self.step(r)?;
                    let lo = self.ranks.iter().map(|k| k.epoch).min().unwrap();
                    let hi = self.ranks.iter().map(|k| k.epoch).max().unwrap();
                    self.max_ahead = self.max_ahead.max(hi - lo);
                } else {
                    let (p, q) = dels[(Self::rng(&mut st) % dels.len() as u64) as usize];
                    self.deliver(p, q)?;
                }
            }
        }
    }

    fn simulate(ops: &[Op], rq: usize, e0: u64, seed: u64, step_pct: u64) -> Result<Sim, String> {
        let (prog, gm) = program(ops, rq);
        let mut sim = Sim::new(prog, gm, rq, e0);
        sim.run(seed, step_pct)?;
        Ok(sim)
    }

    /// Every (row, column) of the assembled logits is written exactly once by the intended shard, bit for
    /// bit, on all four ranks, for m = 1..16 at the REAL geometry (n_sh = 62,080, rq = 7,760), from both
    /// counter phases. (`finish_gather_if_last` asserts written == 1 and the value at every cell.)
    #[test]
    fn exactly_once_assembly_real_geometry_every_m() {
        for m in 1..=MAX_ROWS {
            for (e0, seed, pct) in [(0u64, m as u64, 50u64), (2, 100 + m as u64, 90), (1, 200 + m as u64, 50)] {
                let sim = simulate(&[Op::Gather(m)], RQ, e0, seed, pct).unwrap_or_else(|e| panic!("m {m} e0 {e0}: {e}"));
                for r in 0..WORLD {
                    assert_eq!(sim.ranks[r].epoch, e0 + epochs_per_gather(m) as u64, "m {m} rank {r}");
                    assert_eq!(sim.ranks[r].pc, sim.prog.len());
                }
            }
        }
    }

    /// Many random interleavings at a reduced geometry over mixed programs: no slot is overwritten before its
    /// consumer finished, no send slot before it was sent, no deadlock, every gather assembled exactly.
    #[test]
    fn random_interleavings_mixed_programs() {
        let ops: Vec<Op> = vec![
            Op::Gather(1), Op::Dec, Op::Gather(8), Op::Gather(16), Op::Dec, Op::Dec, Op::Gather(5), Op::Gather(13),
            Op::Gather(4), Op::Gather(9), Op::Dec, Op::Gather(3), Op::Gather(16), Op::Gather(12), Op::Gather(7), Op::Gather(2),
        ];
        let mut runs = 0u32;
        let mut ahead = 0u64;
        for e0 in [0u64, 2, 6, 8, 14, 1] {
            for pct in [5u64, 30, 50, 80, 95, 100] {
                for seed in 1..=6u64 {
                    let sim = simulate(&ops, 13, e0, seed * 7919 + e0 * 31 + pct, pct)
                        .unwrap_or_else(|e| panic!("e0 {e0} pct {pct} seed {seed}: {e}"));
                    ahead = ahead.max(sim.max_ahead);
                    runs += 1;
                }
            }
        }
        // the ranks did drift apart in some schedules (the test is not vacuous). The spread is bounded by the
        // protocol itself (a rank cannot publish epoch e + 1 before its K2(e) saw its partner's K1(e)), so a
        // large spread is neither expected nor required: the adversarial direction is the WIRE lagging the
        // ranks, which the delivery-vs-step weights (5..100 %) and the overwrite detector cover.
        assert!(ahead >= 1, "the schedules never separated the ranks (max epoch spread {ahead})");
        assert!(ahead <= 2, "epoch spread {ahead} exceeds what the serial lookahead-1 dependency chain allows");
        assert_eq!(runs, 6 * 6 * 6);
    }

    /// A few random runs at the real geometry over m = 16 back-to-back (8 epochs per gather, the deepest
    /// schedule: the slot ring wraps inside one gather) with decode exchanges in between.
    #[test]
    fn real_geometry_back_to_back_m16_with_decode_exchanges() {
        let ops = [Op::Gather(16), Op::Dec, Op::Gather(16), Op::Gather(13), Op::Dec, Op::Dec];
        for (seed, pct) in [(3u64, 95u64), (4, 50), (5, 100)] {
            simulate(&ops, RQ, 0, seed, pct).unwrap_or_else(|e| panic!("seed {seed} pct {pct}: {e}"));
        }
    }

    // ---- negative controls: the model CAN see each failure class ---------------------------------------

    /// The W=2-shaped schedule (all K1s, then all K2s with lags) is WRONG at world 4: stage B's K1 would read
    /// columns stage A's K2 has not written. The model must flag it, not assemble garbage silently.
    #[test]
    fn negative_control_w2_shaped_schedule_reads_before_write() {
        let rq = 13;
        let (mut prog, gm) = program(&[Op::Gather(8)], rq);
        // reorder: every K1 first (A, B per group), then every K2 with lag = epochs - 1 - j
        let k1s: Vec<Step> = prog.iter().copied().filter(|s| matches!(s, Step::K1(..))).collect();
        let n = k1s.len();
        let mut k2s: Vec<Step> = prog.iter().copied().filter(|s| matches!(s, Step::K2(..))).collect();
        for (j, s) in k2s.iter_mut().enumerate() {
            if let Step::K2(l, g) = s { *s = Step::K2(Launch { lag: (n - 1 - j) as u32, ..*l }, *g); }
        }
        prog = k1s.into_iter().chain(k2s).collect();
        let mut sim = Sim::new(prog, gm, rq, 0);
        let err = sim.run(1, 50).expect_err("the W=2 shape must be rejected at world 4");
        assert!(err.contains("BEFORE it was written") || err.contains("DEADLOCK"), "unexpected failure: {err}");
    }

    /// A lookahead of 6 (> R/2) lets a delivery land on an unconsumed slot under an adversarial schedule; the
    /// serial lookahead 1 (the real plan) never does. Also proves the simulator's overwrite detector fires.
    #[test]
    fn negative_control_overwrite_is_detected() {
        let mut found = 0;
        for seed in 1..=40u64 {
            if let Err(e) = simulate(&[Op::Synth(6); 8], 4, 0, seed, 98) {
                assert!(e.contains("OVERWRITES") || e.contains("SEND slot") || e.contains("token"), "unexpected: {e}");
                found += 1;
            }
        }
        assert!(found > 0, "no adversarial schedule exposed the unsafe lookahead-6 program");
        for seed in 1..=40u64 {
            simulate(&[Op::Synth(1); 40], 4, 0, seed, 98).unwrap();
            simulate(&[Op::Synth(2); 20], 4, 0, seed, 98).unwrap();
            simulate(&[Op::Synth(4); 10], 4, 0, seed, 98).unwrap();
        }
    }

    /// A wrong partner mask (forgetting that stage B forwards {rank, rank ^ m0} and sending the contiguous
    /// pair {rank & !1, rank | 1} instead) breaks the assembly at m0 = 2: the index mirror would place a
    /// shard in the wrong columns. Checked on the mirror functions directly.
    #[test]
    fn negative_control_contiguous_pair_assumption_is_wrong() {
        let (rounds, e_a) = (ROUNDS, 1u64);       // odd first epoch => m0 = 2
        let mut wrong = 0;
        for r in 0..WORLD {
            let held: HashSet<usize> = payload_shards(r, 1, e_a + 1, rounds).into_iter().collect();
            let contiguous: HashSet<usize> = [r & !1, r | 1].into_iter().collect();
            if held != contiguous { wrong += 1; }
        }
        assert_eq!(wrong, WORLD, "with the odd first epoch every rank's B block is non-contiguous");
        // and with the even stage-A epoch (m0 = 1, stage B at the odd epoch 3) the block IS the contiguous
        // pair: the kernels cover both phases
        for r in 0..WORLD {
            let held: HashSet<usize> = payload_shards(r, 1, 3, rounds).into_iter().collect();
            assert_eq!(held, [r & !1, r | 1].into_iter().collect());
        }
    }

    #[test]
    fn ref_half_is_position_dependent() {
        let mut seen = HashSet::new();
        for s in 0..4 {
            for row in 0..4 {
                for col in 0..512 {
                    seen.insert(ref_half(s, row, col));
                }
            }
        }
        // 8192 draws from a 16-bit space: a healthy spread, far from a constant or a short cycle
        assert!(seen.len() > 6000, "only {} distinct values", seen.len());
        assert_ne!(ref_half(0, 0, 5), ref_half(1, 0, 5));
        assert_ne!(ref_half(0, 0, 5), ref_half(0, 1, 5));
        assert_ne!(ref_half(0, 0, 5), ref_half(0, 0, 6));
    }
}
