//! Inline unit tests, moved out of `model.rs` verbatim (module `t2d_tests`).
//! Declared from `mod.rs`; the body keeps its original indentation and its `use super::*` imports.
    use super::{chunk_ranges, MAX_PREFILL_ROWS, MAX_ROWS};
    use crate::naive::serve::PREFILL_CHUNK;

    #[test]
    fn chunks_tile_the_prompt_exactly() {
        for &total in &[1usize, 16, 2047, 2048, 2049, 4096, 5000, 32768] {
            let r = chunk_ranges(total, MAX_PREFILL_ROWS);
            assert_eq!(r.iter().map(|&(_, n)| n).sum::<usize>(), total, "total {total}");
            let mut at = 0usize;
            for (c0, n) in &r {
                assert_eq!(*c0, at, "chunks must be contiguous (total {total})");
                assert!(*n > 0 && *n <= MAX_PREFILL_ROWS, "chunk width out of range");
                at += n;
            }
        }
    }

    #[test]
    fn a_prompt_within_the_chunk_is_one_pass() {
        assert_eq!(chunk_ranges(2048, MAX_PREFILL_ROWS), vec![(0, 2048)], "must stay single-shot");
        assert_eq!(chunk_ranges(1, MAX_PREFILL_ROWS), vec![(0, 1)]);
    }

    /// REGRESSION GATE (the 16-row sids overrun). Every cache-addressing kernel indexes
    /// `slot_ids[m]` for `m` in `0..m`, and `sids_buf` is the ONE allocation all of them read.
    /// It was sized `MAX_ROWS` (16) while prefill feeds it chunks of up to `PREFILL_CHUNK`
    /// (1024) rows: rows >= 16 read past the allocation, the garbage slot id became the KV plane
    /// stride `slot * (kv * S * hd)` with S = max_pos = 262144, and every prefill wider than 16
    /// rows aborted with CUDA_ERROR_ILLEGAL_ADDRESS. Prompts of <= 16 rows were fine, which is
    /// exactly why short gates stayed green.
    ///
    /// THE INVARIANT: `sids_buf` is `vec![0i32; MAX_PREFILL_ROWS]`, every chunk reads from
    /// OFFSET 0 (n_slots = 1 in serving, so all rows are slot 0 and the buffer is the
    /// guaranteed-zero array), so the buffer must cover the WIDEST CHUNK any path can hand it.
    /// It must never be indexed by a cumulative row offset.
    #[test]
    fn slot_id_buffer_covers_the_widest_chunk_any_path_feeds_it() {
        // decode/verify rows must fit too (they were the original MAX_ROWS cap)
        assert!(MAX_PREFILL_ROWS >= MAX_ROWS, "the decode cap must not exceed the prefill cap");
        // the SERVING chunk cap must fit, or raising it silently reintroduces the overrun
        assert!(
            PREFILL_CHUNK <= MAX_PREFILL_ROWS,
            "serving chunk cap {PREFILL_CHUNK} exceeds the {MAX_PREFILL_ROWS}-row slot-id buffer"
        );
        // every chunk of any prompt length must fit, read from base offset 0
        for &total in &[1usize, 16, 17, 128, 1024, 2048, 2049, 5000, 32768] {
            for (c0, n) in chunk_ranges(total, MAX_PREFILL_ROWS) {
                assert!(n <= MAX_PREFILL_ROWS, "chunk ({c0}, {n}) of {total} exceeds the buffer");
            }
        }
    }

    /// WHY THE CHUNK MUST READ SLOT IDS FROM BASE OFFSET 0.
    ///
    /// The old code passed `sids + (c0 * 4)` — a CUMULATIVE row offset. `chunk_ranges` puts
    /// the 2nd chunk at `c0 = width of the 1st`, so for any prompt longer than one chunk that
    /// offset is already >= the buffer width and the kernel reads past the allocation. This
    /// test pins that fact: if `chunk_ranges` ever stopped producing cumulative offsets (e.g.
    /// by clamping `total`), the guard below would stop firing and the "base offset" comment
    /// would need re-deriving, so a silent change here is worth a red build.
    #[test]
    fn a_cumulative_slot_offset_would_overrun_on_the_second_chunk() {
        for &total in &[2049usize, 5000, 32768] {
            let r = chunk_ranges(total, MAX_PREFILL_ROWS);
            assert!(r.len() >= 2, "total {total} must split into >= 2 chunks at this cap");
            let (c0_2, n_2) = r[1];
            assert_eq!(c0_2, r[0].1, "the 2nd chunk starts where the 1st ended");
            assert!(
                c0_2 >= MAX_PREFILL_ROWS,
                "2nd chunk of {total} starts at {c0_2}: a cumulative `sids + c0*4` would overrun"
            );
            assert!(n_2 <= MAX_PREFILL_ROWS, "chunk width must fit the buffer read from base");
        }
    }

    /// A zero (or otherwise absurd) cap must still TERMINATE — `while c0 < total` with `cap == 0`
    /// would spin forever. The floor is 1, so it degrades to one row per pass.
    #[test]
    fn a_zero_cap_cannot_spin() {
        let r = chunk_ranges(10, 0);
        assert_eq!(r.iter().map(|&(_, n)| n).sum::<usize>(), 10, "must still tile exactly");
        assert!(r.iter().all(|&(_, n)| n >= 1), "no zero-width chunk");
        assert!(r.len() <= 10, "terminates in bounded passes");
    }
