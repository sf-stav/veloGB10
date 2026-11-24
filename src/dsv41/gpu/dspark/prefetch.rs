//! The draft-token Engram prefetch (plan §2.4, T0 §3.6): the next verify feeds `[tok, d1 .. dk]`; their Engram rows
//! (every Engram layer, this rank's columns) are known the moment the drafts are, so they are read into EIO's row
//! cache while the drafter's tail and the host gap run, and the verify's `submit_decode` then finds them resident
//! (or joins the reads in flight; EIO never reads a row twice). Two calls per round:
//! * [`warm_token`] when the drafter STARTS (the bonus / correction token is known: one token's rows);
//! * [`warm_drafts`] when the drafts are on the host (all k + 1 tokens; the already-warm token is skipped by EIO).
//! The hash state is the lane's state BEFORE the verify block (`peek` does not advance it).

use crate::dsv41::engram::hash::{EngramHasher, NgramHashState};
use crate::dsv41::engram::prefetch::EngramIo;
use anyhow::Result;

/// The Engram row ids of `tokens` fed after `state` (ids per token = layers x columns).
pub fn row_ids(h: &EngramHasher, state: &NgramHashState, tokens: &[u32]) -> Result<Vec<u32>> {
    let mut ids = vec![0u32; tokens.len() * h.ids_per_token()];
    h.peek(state, tokens, None, &mut ids)?;
    Ok(ids)
}

/// Read the rows of the bonus / correction token ahead (decode priority). Returns the reads issued.
pub fn warm_token(io: &mut EngramIo, h: &EngramHasher, state: &NgramHashState, tok: u32) -> Result<usize> {
    io.warm(&row_ids(h, state, &[tok])?, None, true)
}

/// Read the rows of the whole next verify block `[tok, drafts..]` ahead. Returns the reads issued.
pub fn warm_drafts(io: &mut EngramIo, h: &EngramHasher, state: &NgramHashState, tok: u32, drafts: &[u32]) -> Result<usize> {
    let mut block = Vec::with_capacity(drafts.len() + 1);
    block.push(tok);
    block.extend_from_slice(drafts);
    io.warm(&row_ids(h, state, &block)?, None, true)
}
