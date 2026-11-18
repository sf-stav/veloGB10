//! Free helpers of the scheduler: per-step RNG (splitmix64 domains), prefix compare, the DF2 carry check,
//! vision-image identities for prefix reuse, and the MTP tree accept walk. Pure host-side code.

/// Per-step RNG for stochastic MTP. The lane holds a 64-bit key; every draw is an independent
/// splitmix64 of (key, domain, index), and the key advances once per decode step.
///
/// Two properties this has to have, both learned the hard way:
///
/// 1. **Uniforms must actually land in [0,1).** The device kernels run a *32-bit* LCG and take
///    `r = (s >> 8) / 2^24`, which is a 24-bit mantissa. Carrying that state in a u64 and shifting
///    right by 8 leaves a full 32-bit value, so `r` lands in [0, 256) and an `r < ratio` test with
///    `ratio <= 1` succeeds only ~1/256 of the time — silently rejecting ~255/256 of perfectly
///    acceptable drafts (this is what pinned stochastic acceptance at ~1%).
///
/// 2. **Draws must not share a stream across consumers.** Each `spec_verify_b` column advances its
///    seed internally, so handing the columns *consecutive* LCG states makes column `j`'s draw come
///    from the state that seeds column `j+1` — fine only as long as every column draws exactly once,
///    and an outright collision the moment one draws twice. Domain separation removes the coupling
///    entirely, so the host accept decisions, the per-column residual samples, and the bonus sample
///    are independent by construction.
#[inline]
pub(crate) fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Domain separators keeping the host accept stream disjoint from the device column seeds.
pub(crate) const RNG_DOM_VERIFY: u64 = 0x5645_5249_4659_0001; // device seeds for spec_verify_b columns
pub(crate) const RNG_DOM_ACCEPT: u64 = 0x4143_4345_5054_0001; // host accept/reject uniforms
pub(crate) const RNG_DOM_SAMPLE: u64 = 0x5341_4D50_4C45_0001; // device seeds for sample_b (batched path)
pub const RNG_DOM_DF2_SEL: u64 = 0x4446_3253_454C_0001; // device seeds for the sampled selector walk (S5F2 L2)

/// One independent 32-bit draw from the lane's step key, keyed by domain + index.
#[inline]
pub fn rng_u32(key: u64, domain: u64, idx: usize) -> u32 {
    (splitmix64(key ^ domain ^ (idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 32) as u32
}

/// One independent uniform in [0,1), matching the device's 24-bit mantissa convention.
#[inline]
pub(crate) fn rng_uniform(key: u64, domain: u64, idx: usize) -> f32 {
    (rng_u32(key, domain, idx) >> 8) as f32 * (1.0f32 / 16777216.0)
}

/// Length of the longest common prefix of two token sequences.
pub(super) fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// DF2_CARRY decision, as a pure function so the cases are unit-testable (see `mod df2_carry`).
///
/// The DFlash2 ring is a single already-resident shared buffer, so carrying it across a
/// prefix-cache hit costs **zero bytes** — the only question is whether the rows the draft will
/// read still hold the right positions. Two conditions make that true, and the second is why a
/// length-only copy of DSpark's A6 carry (`slot_dspark_len[phys] >= reuse`) is WRONG here:
///
/// 1. **Identity** — `ring_slot == Some(phys)`. The round is ONE shared buffer while the
///    per-slot bookkeeping is per-slot, so "this slot once ran the drafter" does not imply the
///    ring still holds THIS slot's prefix. DSpark used to guard on length alone and was therefore
///    exposed to a stale ring after any other slot used the drafter in between; with
///    `max_batch > 1` that interleaving is reachable, so DFlash2 records the identity — and
///    DSpark now does too (`DsparkRound::ring_slot`, PLAN/DSPARK_RING_IDENTITY_SPEC.md).
///
/// 2. **Frontier** — `reuse <= ring_len <= plen`. The DF2 ring is a MODULO ring (`pos % RING`,
///    RING = 2048); DSpark's is LINEAR (`c_ring = max_c`, no wrap), which is exactly why A6's
///    length check suffices there and does not here. After a lane that ran to frontier `N` the
///    ring holds precisely positions `[N-RING, N)`; every older row has been recycled. The
///    draft at the anchor reads the trailing band `[plen-RING+1, plen)`; the suffix prime writes
///    `[reuse, plen)`; those two are complementary residues of that band, so the CARRIED rows
///    must cover `[plen-RING+1, reuse)`, which requires `reuse <= N` (they were written) and
///    `N <= plen` (they have not yet been overwritten). Both bounds are equalities in the
///    natural multi-turn flow (`N` is the previous turn's prompt + its emitted tokens, and the
///    new prompt is that plus the client's next message), which is why this fires at all.
///
/// Carrying with `N > plen` would feed the draft keys from the wrong ring rows. Verify still
/// rejects bad drafts, so the output text would stay correct and losslessness would stay green
/// while tau quietly fell — the "silently wrong but self-consistent" class AGENTS §3 warns
/// about. Refuse instead.
#[allow(clippy::too_many_arguments)]
pub(super) fn df2_carry_ok(enabled: bool, is_df2_src: bool, reuse: usize, plen: usize,
                phys: usize, ring_slot: Option<usize>, ring_len: usize,
                len_only: bool) -> bool {
    enabled
        && is_df2_src
        && reuse > 0
        && ring_slot == Some(phys)
        && ring_len >= reuse
        // `len_only` is a DIAGNOSTICS-ONLY instrument (--df2-carry-len-only=1): it drops the
        // frontier term, reproducing DSpark's length-only shape so SPEC §7 gate 7 can force a
        // known-bad carry and confirm the measurement catches it. Never a serving option.
        && (len_only || ring_len <= plen)
}

/// An image's content identity inside a slot's cached sequence: the token-span start plus a hash
/// of the merged embeddings that were actually spliced there. The slot-level prefix reuse matches
/// purely on token identity, and two images of the same resolution expand to IDENTICAL `image_pad`
/// token runs — so a token-only prefix key would happily reuse image N's spliced KV/state for image
/// N+1. This fingerprint is the image-content half of that key.
#[derive(Clone, Copy)]
pub(super) struct ImageIdentity {
    pub(super) start: usize,        // token-span start in the EXPANDED stream
    pub(super) hash: u64,           // FNV-1a over the image's merged-embedding bytes
}

/// The per-image content identities for a request's concatenated `image_embeds` (f32 merged
/// embeddings, concatenated in `spans` order). Empty when the request carries no images. A hash of
/// the RAW f32 merged embeddings (the bytes the prefill converts to bf16 and splices) is the most
/// faithful content key: the same image always yields the same embeddings, a different image
/// essentially never does.
pub(super) fn request_image_identities(
    embeds: Option<&Vec<f32>>,
    spans: &[crate::vision_encoder::ImageSpan],
) -> Vec<ImageIdentity> {
    // Row width = the vision out width (== text hidden for every supported tower). Derive it
    // from the request itself: len == sum(spans.num_tokens) * width, so ANY tower geometry
    // (0.8b 1024 … 27B 5120) is handled without plumbing model-specific constants.
    let total: usize = spans.iter().map(|s| s.num_tokens).sum();
    let out_h = if total > 0 {
        embeds.map(|e| e.len() / total).unwrap_or(0)
    } else {
        0
    };
    let mut out = Vec::new();
    let Some(emb) = embeds else { return out; };
    let mut off = 0usize;
    for s in spans {
        let n = s.num_tokens;
        let row0 = off * out_h;
        let row1 = (off + n) * out_h;
        let slice = if row1 <= emb.len() { &emb[row0..row1] }
                    else if row0 < emb.len() { &emb[row0..emb.len()] }
                    else { &[][..] };
        // Hash the f32 bit patterns directly (no `from_raw_parts` on a possibly-empty slice).
        let mut h: u64 = 0xcbf29ce484222325;
        for &v in slice {
            for b in v.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        out.push(ImageIdentity { start: s.start, hash: h });
        off += n;
    }
    out
}

/// True iff reusing the prefix `[0, l)` from a slot is safe for this request's images: every image
/// whose span begins inside the reused prefix (`start < l`) must match the slot's recorded image at
/// the same position. A differing image at a reused position means the slot's KV/GDN state was built
/// from the OLD image — reusing it would replay that image's content.
pub(super) fn images_compatible(slot_imgs: &[ImageIdentity], req_imgs: &[ImageIdentity], l: usize) -> bool {
    for r in req_imgs {
        if r.start < l {
            match slot_imgs.iter().find(|s| s.start == r.start) {
                Some(s) if s.hash == r.hash => {}
                _ => return false,
            }
        }
    }
    true
}

/// Greedy tree accept-walk. Follow the target's argmax down the tree: at the current node, if any child
/// carries the token the target predicts (`preds[current]`), descend into it; else stop. Column 0 is the
/// committed token (always the start). This is the tree generalisation of the chain "accept longest
/// prefix", and greedy exactness is preserved by the same induction: every emitted token is the target's
/// argmax given its accepted prefix, so the output is byte-identical to plain autoregressive decode.
///
/// `parent[c]` is c's tree parent (parent[0] = -1). `tokens[c]` is the draft token at column c (tokens[0]
/// is the committed token). `preds[c]` is the target's argmax AFTER the path ending at c. Returns:
/// - `path`: the accepted node indices, root first (`path[0] == 0`).
/// - `emitted`: the tokens to emit = `preds` along the accepted path = each step's correction/accepted
///   token, ending with the bonus `preds[leaf]`.
///
/// A tie prefers the LOWEST child index (deterministic); duplicate sibling tokens are de-duped upstream.
pub(super) fn tree_accept_walk(parent: &[i32], tokens: &[u32], preds: &[u32]) -> (Vec<usize>, Vec<u32>) {
    let n = parent.len();
    // children[c] in ascending index order (index order == DFS order here, so lowest = first-drafted).
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    for c in 1..n { children[parent[c] as usize].push(c); }

    let mut path = vec![0usize];
    let mut emitted = Vec::new();
    let mut cur = 0usize;
    loop {
        let want = preds[cur];
        emitted.push(want); // the token the target actually chose after `cur` (accepted child or bonus)
        match children[cur].iter().copied().find(|&c| tokens[c] == want) {
            Some(c) => { path.push(c); cur = c; }
            None => break, // no child matched: `want` is the correction/bonus, walk ends
        }
    }
    (path, emitted)
}

