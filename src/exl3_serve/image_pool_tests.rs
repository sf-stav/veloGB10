//! PFXPOOL4 unit tests: why image-bearing prompts are pool-safe at full depth, host-only (no GPU).
//!
//! The restore chain the pool runs is: key-prefix match -> copy rows + the recurrent snapshot into ANOTHER lane
//! -> the new request prefills the suffix and decodes. Byte-correctness of the restored IMAGE rows in the new
//! lane therefore rests on exactly two facts, and these tests pin both:
//! 1. KEY STABILITY: the cache key replaces image rows with `image_row_key(pixel_hash, j)` — same image, same
//!    span position => byte-identical key across turns; any different image => a mismatch at the span's rows.
//! 2. ROPE STABILITY: RoPE rotation is applied at row WRITE from the slot's per-REQUEST vision arm (admit arms it
//!    for the FULL new prompt before any restore), so restored rows are correct iff the new prompt's 3-axis
//!    positions for those rows equal the old prompt's — `mrope_positions` is prefix-stable (and the post-prompt
//!    delta constant) under prompt extension, which `vision_encoder`'s own vis2_tests anchor against HF.
//! Plus the pool-side depth rule: a key mismatch truncates the resume at the provable prefix, never past it.

use super::image_row_key;
use crate::prefix_pool::{Pool, Resume};
use crate::vision_encoder::{mrope_positions, ImageSpan};

/// The cache key of a rendered prompt: filler vocab ids with each span's rows replaced exactly the way
/// `vision_setup` does it (the shared helper guarantees the expression is the same one, not a copy).
fn keyed(len: usize, spans: &[(usize, usize, u64)]) -> Vec<u32> {
    let mut k: Vec<u32> = (0..len as u32).map(|i| 1000 + (i % 500)).collect();
    for &(start, n, hash) in spans {
        for j in 0..n {
            k[start + j] = image_row_key(hash, j);
        }
    }
    k
}

// 1. Key stability: the same image at the same span position re-renders byte-identical ids across turns;
// a different image mismatches; ids stay out of the vocab range and out of each other's way.
#[test]
fn image_row_key_is_stable_across_turns_and_image_specific() {
    let (h1, h2) = (0xdead_beef_cafe_1234u64, 0x0123_4567_89ab_cdefu64);
    // turn 1 and turn N render the same span: every row id must be identical (this is what the pool/spill/
    // own-lane prefix scans compare — any drift would silently turn every hit into a miss)
    for j in 0..256usize {
        assert_eq!(image_row_key(h1, j), image_row_key(h1, j), "row {j} of one span");
    }
    // a different image mismatches from the FIRST row (the pool then resumes only up to the provable prefix)
    assert_ne!(image_row_key(h1, 0), image_row_key(h2, 0));
    assert_ne!(image_row_key(h1, 5), image_row_key(h2, 5));
    // ids are out of the vocabulary range and pairwise distinct within a span (the key is injective in j, so
    // the span's rows cannot alias each other or a text row)
    let ids: Vec<u32> = (0..1024).map(|j| image_row_key(h1, j)).collect();
    assert!(ids.iter().all(|&t| t >= 0x8000_0000));
    let uniq: std::collections::HashSet<u32> = ids.iter().copied().collect();
    assert_eq!(uniq.len(), ids.len());
    // and the rendered keys compose: the same image at the same position in two differently-sized prompts
    // leaves the span rows byte-identical (a grown conversation keeps earlier spans put)
    assert_eq!(keyed(400, &[(100, 256, h1)])[100..356], keyed(1500, &[(100, 256, h1)])[100..356]);
}

// 2. Rope stability: growing a prompt (the only way a pool entry is ever reused) does NOT move the 3-axis
// positions of the prefix rows nor the prompt's rope delta. Rotated-K rows written under the old arm are
// therefore exactly the rows a fresh prefill of the new prompt would write — a cross-lane restore is byte-correct.
#[test]
fn mrope_positions_and_delta_are_prefix_stable_under_growth() {
    let spans_old = [ImageSpan { start: 100, num_tokens: 256, grid_h: 16, grid_w: 16, pixel_hash: 7 },
                     ImageSpan { start: 500, num_tokens: 64, grid_h: 8, grid_w: 8, pixel_hash: 9 }];
    let (p_old, d_old) = mrope_positions(1000, &spans_old).unwrap();
    // text growth only: same spans, longer prompt
    let (p_txt, d_txt) = mrope_positions(1500, &spans_old).unwrap();
    assert_eq!(&p_txt[..1000], &p_old[..], "text growth moved prefix positions");
    assert_eq!(d_txt, d_old, "text growth moved the rope delta");
    // growth with a NEW image appended: prefix rows unchanged (the new delta applies only to rows past the
    // new prompt, which the new request's arm re-ropes on its own)
    let spans_new = [spans_old[0], spans_old[1],
                     ImageSpan { start: 1200, num_tokens: 16, grid_h: 4, grid_w: 4, pixel_hash: 11 }];
    let (p_img, _) = mrope_positions(1500, &spans_new).unwrap();
    assert_eq!(&p_img[..1000], &p_old[..], "a later image moved prefix positions");
    // a re-render that DROPS the images: the key (raw placeholder rows) mismatches at the first span row long
    // before any rope question arises — the arm for a text-only request is plain cos_tab and matches nothing
    // in an image-keyed entry deeper than the LCP. (Same-image-different-grid is the one key collision the
    // hash cannot see; it already breaks the own-lane snapshot and the spill tier identically, and
    // preprocessing is deterministic per (image bytes, --image-max-edge) on one boot.)
    let (p_drop, d_drop) = mrope_positions(1000, &[]).unwrap();
    assert_eq!(p_drop[150], [150; 3], "text-only positions must be the plain count");
    assert_ne!(p_old[150], [150; 3], "row 150 sits inside the first span here (h/w axes diverge)");
    assert_ne!(d_drop, d_old); // the deltas differ: only key-equal prompts may share rows, and these do not
}

// 3. Pool depth on image keys: a grown prompt hits the entry at FULL depth (across lanes — the pool is
// lane-agnostic by construction); a changed image drops the resume to what the entry can PROVE.
#[test]
fn image_keyed_pool_entry_hits_full_depth_and_never_overmatches() {
    let (h1, h2) = (0xdead_beef_cafe_1234u64, 0x0123_4567_89ab_cdefu64);
    let mut p: Pool<u8> = Pool::new(100_000);
    let turn1 = keyed(400, &[(100, 256, h1)]);
    p.insert(turn1.clone(), 400, 1);
    // turn 2: same key prefix (the reply + a new question only append) — full-depth hit for another lane
    let mut turn2 = turn1.clone();
    turn2.extend((0..600u32).map(|i| 2000 + (i % 400)));
    assert_eq!(p.lookup(&turn2, turn2.len() - 1), Some((0, 400)));
    assert_eq!(p.best(&turn2, turn2.len() - 1), Some(Resume { i: 0, at: 400, full: true }));
    // a re-render with a DIFFERENT image: no entry prefixes it, and with no checkpoints there is nothing
    // provable below the mismatch — the kept depth is 0, never deeper than the LCP (the first span row, 100)
    let other = keyed(400, &[(100, 256, h2)]);
    assert_eq!(p.lookup(&other, 399), None);
    assert_eq!(p.best(&other, 399), None);
    // with a carried checkpoint BEFORE the first image row, the resume depth is exactly that checkpoint:
    // the pool hands back the provable text prefix and re-prefills from the image on (guarded partial)
    let mut q: Pool<u8> = Pool::new(100_000);
    q.insert_marked(turn1.clone(), 400, 1, vec![96], None, vec![(400, 1)]);
    assert_eq!(q.best(&other, 399), Some(Resume { i: 0, at: 96, full: false }));
    // a same-hash image moved to a different span start mismatches at the OLD start (ids encode the span's
    // internal j and the position; a move changes which raw row carries the first image id)
    let moved = keyed(400, &[(120, 256, h1)]);
    assert_eq!(q.lookup(&moved, 399), None);
    assert_eq!(q.best(&moved, 399), Some(Resume { i: 0, at: 96, full: false })); // provable prefix only
}
