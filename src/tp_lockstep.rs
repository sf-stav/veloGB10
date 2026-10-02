//! TP-4D: the world > 2 head-hub LOCKSTEP frames of the EXL3 serve path.
//!
//! At world 2 a lockstep point is three pairwise steps over the shared hot-path ring memory (`agree` fence A,
//! a u32 `exchange`, `agree` fence B) — `src/exl3_serve.rs` keeps that code and its wire protocol untouched.
//! At world > 2 every lockstep point is ONE `net::hub_all` over the dedicated control slots: each rank sends one
//! frame, the head gathers, then broadcasts the concatenation, so EVERY rank holds every rank's frame and runs
//! the same pure verdict function over the same data — all ranks reach the same verdict and abort together
//! (nobody is left parked in a later exchange). The hub's two rounds are themselves the barrier.
//!
//! This file is pure frame layout + verdicts, no transport: the ops in `exl3_serve.rs` (`TpSync::*_hub`) glue
//! them to `hub_all`, and the tests here run the REAL `hub_all` over the in-memory `net::hub_mock` transport
//! with one thread per rank.
//!
//! Frame layouts (payload words; the hub adds 2 words of tail headroom, `*_WIRE`). Every frame leads with a
//! `tag` word (a constant class prefix | low counter bits) and the `field` word (`agree_field(step, kind)`, the
//! lockstep counter + kind) which ALL ranks must agree on — that is the "same lockstep point" proof.
//!   GO    [tag 0x60.., field, head step_no, head n_events]              nodes send 0 for the head words
//!   PV    [tag 0x9E.., field, width, drafts hash, head passes launched] all words compared
//!   RND   [tag 0xC0DE.., field, accept, width, words hash, epoch lo, epoch hi, ms lo, ms hi, plain lo, plain hi,
//!          captured]                                                    words 0..5 compared; 5,6 diagnostic; 7.. head's
//!   IDENT [tag 0xD1.., n, digest lo/hi ...]                             per-digest compare against the head's
//!   ALLOC [tag 0xA110.., checkpoints allocated so far, ok]              WP16 checkpoint allocation outcome (audit C2)
//! The wire sizes are all EVEN (the tail tag is one aligned u64 at `nbytes - 8`) and differ between classes;
//! `net_exchange_one` zeroes every consumed frame so a stale larger frame can never leave a tag-shaped word pair
//! where a smaller frame's tail is read (`net::hub_tests::stale_payload_can_fake_a_tag_until_the_frame_is_retired`).

pub(crate) const GO_PAYLOAD: usize = 4;
pub(crate) const GO_WIRE: usize = GO_PAYLOAD + 2;
pub(crate) const PV_PAYLOAD: usize = 5;
pub(crate) const PV_WIRE: usize = 8; // payload 5 + 2 tail words = 7, rounded up so the tail tag stays 8-byte aligned
pub(crate) const RND_PAYLOAD: usize = 12;
pub(crate) const RND_WIRE: usize = RND_PAYLOAD + 2;
pub(crate) const ID_HDR: usize = 2;
pub(crate) const ALLOC_PAYLOAD: usize = 3;
pub(crate) const ALLOC_WIRE: usize = 10; // payload 3 + 2 tail words = 5; padded to a size no other class uses (even: 8-byte aligned tail tag)

/// Wire words of an IDENT frame carrying `n` digests.
pub(crate) fn id_wire(n: usize) -> usize { ID_HDR + 2 * n + 2 }

const GO_TAG: u32 = 0x6000_0000;
const PV_TAG: u32 = 0x9E00_0000;
const RND_TAG: u32 = 0xC0DE_0000;
const ID_TAG: u32 = 0xD100_0000;
const ALLOC_TAG: u32 = 0xA110_0000;

/// Compare the listed words of every frame against rank 0's; Err names up to six offenders.
fn same_words(what: &str, names: &[&str], frames: &[Vec<u32>], cmp: &[usize]) -> Result<(), String> {
    let mut bad: Vec<String> = Vec::new();
    for (r, f) in frames.iter().enumerate().skip(1) {
        for &i in cmp {
            if f[i] != frames[0][i] {
                bad.push(format!("rank {r} {} = {:#010x} vs head {:#010x}", names[i], f[i], frames[0][i]));
            }
        }
    }
    if bad.is_empty() { return Ok(()); }
    let n = bad.len();
    bad.truncate(6);
    Err(format!("{what}: {} word(s) differ from the head's: {}{}", n, bad.join("; "), if n > 6 { "; ..." } else { "" }))
}

fn check_shape(what: &str, frames: &[Vec<u32>], world: usize, payload: usize) -> Result<(), String> {
    if frames.len() != world || frames.iter().any(|f| f.len() != payload) {
        return Err(format!("{what}: malformed hub result ({} frames of lengths {:?}, want {world} x {payload})",
                           frames.len(), frames.iter().map(|f| f.len()).collect::<Vec<_>>()));
    }
    Ok(())
}

// ---------------------------------------------------------------- GO (step_go)

/// The per-step "go": the head publishes (step number, event count); every rank sends its lockstep counter
/// field so the hub also proves all ranks are at the same lockstep point.
pub(crate) fn go_frame(is_head: bool, field: u32, step_no: u64, n_events: usize) -> [u32; GO_PAYLOAD] {
    [GO_TAG | (field & 0xFF_FFFF), field,
     if is_head { step_no as u32 } else { 0 }, if is_head { n_events as u32 } else { 0 }]
}

/// Ok((head step number low 32 bits, head event count)). Words 0,1 (tag, counter field) must match on all ranks.
pub(crate) fn go_verdict(frames: &[Vec<u32>], world: usize) -> Result<(u32, u32), String> {
    check_shape("step-go", frames, world, GO_PAYLOAD)?;
    if frames[0][0] & 0xFF00_0000 != GO_TAG {
        return Err(format!("step-go: the head's frame carries tag {:#010x}, not a step-go frame", frames[0][0]));
    }
    same_words("step-go", &["tag", "counter field", "step", "events"], frames, &[0, 1])?;
    Ok((frames[0][2], frames[0][3]))
}

// ---------------------------------------------------------------- PV (pre_verify)

pub(crate) fn pv_frame(field: u32, w: usize, h: u32, launched: usize) -> [u32; PV_PAYLOAD] {
    [PV_TAG | (field & 0xFFFF), field, w as u32, h, launched as u32]
}

/// Ok = every rank chose the same verify width, drafts and speculative head-pass launches at the same lockstep
/// point. Err names which rank/word differs.
pub(crate) fn pv_verdict(frames: &[Vec<u32>], world: usize) -> Result<(), String> {
    check_shape("pre-verify", frames, world, PV_PAYLOAD)?;
    same_words("pre-verify", &["tag", "counter field", "width", "drafts hash", "head passes launched"], frames,
               &[0, 1, 2, 3, 4])
}

// ---------------------------------------------------------------- RND (tp_round)

pub(crate) struct RoundIn {
    pub step: u64,
    pub field: u32,
    pub accept: usize,
    pub width: usize,
    pub hash: u32,
    pub epoch: u64,
    pub ms: f64,
    pub plain_ms: f64,
    pub captured: bool,
}

pub(crate) fn rnd_frame(r: &RoundIn) -> [u32; RND_PAYLOAD] {
    let (mb, pb) = (r.ms.to_bits(), r.plain_ms.to_bits());
    [RND_TAG | (r.step as u32 & 0xFFFF), r.field, r.accept as u32, r.width as u32, r.hash,
     r.epoch as u32, (r.epoch >> 32) as u32, mb as u32, (mb >> 32) as u32, pb as u32, (pb >> 32) as u32,
     r.captured as u32]
}

pub(crate) struct RoundOut {
    /// the head's timing words: every rank adopts these, so every timing-derived decision is the head's
    pub ms: f64,
    pub plain_ms: f64,
    pub captured: bool,
    /// (rank, device epoch delta vs the head) for every rank whose barrier counter differs — a DIAGNOSTIC, the
    /// hot-path epoch-divergence probe of the W>2 agree; it does not fail the round
    pub epoch_skew: Vec<(usize, i64)>,
}

/// Ok = every rank reached this lockstep point with the same tag, counter field, accept count, verify width and
/// words hash (the AGENTS §2.10 tripwire, now compared by every rank).
pub(crate) fn rnd_verdict(frames: &[Vec<u32>], world: usize) -> Result<RoundOut, String> {
    check_shape("round", frames, world, RND_PAYLOAD)?;
    same_words("round", &["tag", "counter field", "accept count", "verify width", "words hash"], frames,
               &[0, 1, 2, 3, 4])?;
    let ep = |f: &Vec<u32>| f[5] as u64 | ((f[6] as u64) << 32);
    let epoch_skew = frames.iter().enumerate().skip(1)
        .filter(|(_, f)| ep(f) != ep(&frames[0]))
        .map(|(r, f)| (r, ep(f) as i64 - ep(&frames[0]) as i64)).collect();
    let h = &frames[0];
    Ok(RoundOut {
        ms: f64::from_bits(h[7] as u64 | ((h[8] as u64) << 32)),
        plain_ms: f64::from_bits(h[9] as u64 | ((h[10] as u64) << 32)),
        captured: h[11] != 0,
        epoch_skew,
    })
}

// ---------------------------------------------------------------- IDENT (G-T1-b digests)

pub(crate) fn id_frame(step: u64, digests: &[u64]) -> Vec<u32> {
    let mut v = vec![ID_TAG | (step as u32 & 0xFFFF), digests.len() as u32];
    v.extend(digests.iter().flat_map(|&x| [x as u32, (x >> 32) as u32]));
    v
}

/// Ok(number of digests on which at least one rank differs from the head). Err on a tag / count mismatch.
pub(crate) fn id_verdict(frames: &[Vec<u32>], world: usize, n: usize) -> Result<usize, String> {
    check_shape("ident", frames, world, ID_HDR + 2 * n)?;
    same_words("ident", &["tag", "digest count"], frames, &[0, 1])?;
    if frames[0][1] as usize != n {
        return Err(format!("ident: the head sent {} digests, this rank {n}", frames[0][1]));
    }
    Ok((0..n).filter(|&i| frames.iter().skip(1).any(|f| {
        f[ID_HDR + 2 * i] != frames[0][ID_HDR + 2 * i] || f[ID_HDR + 2 * i + 1] != frames[0][ID_HDR + 2 * i + 1]
    })).count())
}

// ---------------------------------------------------------------- ALLOC (WP16 checkpoint allocation, audit C2)

/// One rank's checkpoint-allocation outcome: how many checkpoint buffers it holds (`n_alloc`, which every rank's
/// store must agree on) and whether the allocation it just attempted succeeded.
pub(crate) fn alloc_frame(n_alloc: usize, ok: bool) -> [u32; ALLOC_PAYLOAD] {
    [ALLOC_TAG | (n_alloc as u32 & 0xFFFF), n_alloc as u32, ok as u32]
}

/// Ok(ranks whose allocation FAILED) — empty = every rank got its buffer, so all keep it; non-empty = every rank
/// (including those that succeeded) must treat the allocation as failed and freeze the store's cap, so all stores
/// stay identical. Err = the ranks' stores already disagree on how many buffers they hold (or a frame of another
/// class arrived): a desync, never a recoverable allocation failure.
pub(crate) fn alloc_verdict(frames: &[Vec<u32>], world: usize) -> Result<Vec<usize>, String> {
    check_shape("wp16 alloc", frames, world, ALLOC_PAYLOAD)?;
    if frames[0][0] & 0xFFF0_0000 != ALLOC_TAG {
        return Err(format!("wp16 alloc: the head's frame carries tag {:#010x}, not a checkpoint-allocation frame", frames[0][0]));
    }
    same_words("wp16 alloc", &["tag", "allocated buffers", "ok"], frames, &[0, 1])?;
    Ok(frames.iter().enumerate().filter(|(_, f)| f[2] == 0).map(|(r, _)| r).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_same<const N: usize>(f: [u32; N], world: usize) -> Vec<Vec<u32>> { vec![f.to_vec(); world] }

    #[test]
    fn go_adopts_head_words_and_needs_same_counter() {
        let world = 4;
        let frames: Vec<Vec<u32>> = (0..world).map(|r| go_frame(r == 0, 0xC0_0007, 0x1_0000_0042, 3).to_vec()).collect();
        assert_eq!(go_verdict(&frames, world), Ok((0x42, 3)), "nodes adopt the head's step number (low 32 bits) and event count");
        let mut bad = frames.clone();
        bad[2][1] = 0xC0_0008; // rank 2 is one lockstep point ahead
        let e = go_verdict(&bad, world).unwrap_err();
        assert!(e.contains("rank 2 counter field"), "{e}");
        assert!(!e.contains("rank 1") && !e.contains("rank 3"), "only the offender is named: {e}");
        let mut notgo = frames.clone();
        notgo[0][0] = 0xC0DE_0001; // the head sent some other frame class
        assert!(go_verdict(&notgo, world).unwrap_err().contains("not a step-go frame"));
    }

    #[test]
    fn pv_compares_every_word() {
        let world = 4;
        let ok = all_same(pv_frame(0x40_0005, 6, 0xDEAD_BEEF, 7), world);
        assert_eq!(pv_verdict(&ok, world), Ok(()));
        for (word, name) in [(2usize, "width"), (3, "drafts hash"), (4, "head passes launched"), (1, "counter field")] {
            let mut bad = ok.clone();
            bad[3][word] ^= 1;
            let e = pv_verdict(&bad, world).unwrap_err();
            assert!(e.contains(&format!("rank 3 {name}")), "word {word}: {e}");
        }
        // every rank mismatching at once is reported as a count, capped at six names
        let mut all_bad = ok.clone();
        for f in all_bad.iter_mut().skip(1) { for w in f.iter_mut() { *w ^= 0xFF; } }
        let e = pv_verdict(&all_bad, world).unwrap_err();
        assert!(e.contains("15 word(s) differ") && e.ends_with("; ..."), "{e}");
    }

    fn rin(step: u64, a: usize, m: usize, h: u32, epoch: u64, ms: f64) -> RoundIn {
        RoundIn { step, field: (step & 0x3F_FFFF) as u32, accept: a, width: m, hash: h, epoch, ms, plain_ms: ms * 1.5, captured: true }
    }

    #[test]
    fn rnd_verdict_ships_the_heads_timing_and_flags_divergence() {
        let world = 4;
        let frames: Vec<Vec<u32>> = (0..world)
            .map(|r| rnd_frame(&rin(9, 3, 5, 0xABCD, 1000 + (r == 2) as u64, if r == 0 { 11.25 } else { 99.0 })).to_vec()).collect();
        let out = rnd_verdict(&frames, world).unwrap();
        assert_eq!((out.ms, out.plain_ms, out.captured), (11.25, 16.875, true), "timing words are the head's, not a node's own");
        assert_eq!(out.epoch_skew, vec![(2, 1)], "the epoch probe names the skewed rank but does not fail the round");
        for (what, edit) in [("accept count", 2usize), ("verify width", 3), ("words hash", 4), ("counter field", 1), ("tag", 0)] {
            let mut bad = frames.clone();
            bad[1][edit] ^= 0x10;
            let e = rnd_verdict(&bad, world).err().unwrap_or_else(|| panic!("{what}: a divergent word must fail the round"));
            assert!(e.contains(&format!("rank 1 {what}")), "{what}: {e}");
        }
    }

    #[test]
    fn ident_counts_digests_where_any_rank_differs() {
        let world = 4;
        let d: Vec<u64> = (0..5).map(|i| 0x1111_0000_0000_0000u64 * (i + 1)).collect();
        let mut frames: Vec<Vec<u32>> = (0..world).map(|_| id_frame(9, &d)).collect();
        assert_eq!(id_verdict(&frames, world, 5), Ok(0));
        frames[3][ID_HDR + 2 * 1] ^= 1; // rank 3, digest 1, low word
        frames[2][ID_HDR + 2 * 4 + 1] ^= 1; // rank 2, digest 4, high word
        frames[1][ID_HDR + 2 * 4] ^= 1; // rank 1 differs on the same digest 4: still ONE bad digest
        assert_eq!(id_verdict(&frames, world, 5), Ok(2));
        assert!(id_verdict(&frames, world, 4).unwrap_err().contains("malformed"), "a wrong digest count is a shape error");
    }

    #[test]
    fn alloc_verdict_names_failed_ranks_and_rejects_desynced_stores() {
        let world = 4;
        let all_ok: Vec<Vec<u32>> = (0..world).map(|_| alloc_frame(3, true).to_vec()).collect();
        assert_eq!(alloc_verdict(&all_ok, world), Ok(vec![]), "every rank allocated: keep the buffer");
        let mut one_failed = all_ok.clone();
        one_failed[2] = alloc_frame(3, false).to_vec();
        assert_eq!(alloc_verdict(&one_failed, world), Ok(vec![2]), "one failing rank fails the allocation for ALL ranks");
        let mut head_failed = all_ok.clone();
        head_failed[0] = alloc_frame(3, false).to_vec();
        assert_eq!(alloc_verdict(&head_failed, world), Ok(vec![0]));
        // the stores already disagree on the buffer count: a desync, not an allocation failure
        let mut desync = all_ok.clone();
        desync[1] = alloc_frame(2, true).to_vec();
        let e = alloc_verdict(&desync, world).unwrap_err();
        assert!(e.contains("rank 1 allocated buffers"), "{e}");
        let mut other = all_ok.clone();
        other[0][0] = 0x6000_0001;
        assert!(alloc_verdict(&other, world).unwrap_err().contains("not a checkpoint-allocation frame"));
    }

    #[test]
    fn wire_sizes_fit_the_control_slot() {
        // every class keeps the tail tag 8-byte aligned (even wire words) in both the node frame and the head's
        // world-wide broadcast, and fits the control slot at world 4 and 8
        for wire in [GO_WIRE, PV_WIRE, RND_WIRE, ALLOC_WIRE, id_wire(0), id_wire(7)] {
            assert_eq!(wire % 2, 0, "wire {wire} would put the tail tag at a 4-byte offset");
            for world in [4usize, 8] { assert!(world * wire * 4 <= crate::tp::TP_SLOT_BYTES); }
        }
        assert_ne!(GO_WIRE, PV_WIRE);
        assert_ne!(PV_WIRE, RND_WIRE);
        for other in [GO_WIRE, PV_WIRE, RND_WIRE] { assert_ne!(ALLOC_WIRE, other, "frame classes of one hub lane must differ in wire size"); }
        // an IDENT frame of the widest realistic digest window (rows 0..16 x (logits + resid + taps)) fits at world 8
        assert!(8 * id_wire(16 * 3 + 64) * 4 <= crate::tp::TP_SLOT_BYTES);
        assert!(GO_PAYLOAD * 4 + 8 <= GO_WIRE * 4 && PV_PAYLOAD * 4 + 8 <= PV_WIRE * 4 && RND_PAYLOAD * 4 + 8 <= RND_WIRE * 4
                && ALLOC_PAYLOAD * 4 + 8 <= ALLOC_WIRE * 4,
                "every frame keeps 8 bytes of tail headroom");
    }
}
