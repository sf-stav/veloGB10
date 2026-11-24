//! Attention-sink detection and draft-zero rejection: OPT-IN, default off (`--dsv41-sink-reject`).
//!
//! DeepSeek's own engine has neither (PLAN Revision 4; XREF_DEEPINFER §5): a sampled row whose last-layer residual
//! column (2455 on V4.1-Flash-1017) is huge predicts a BOS-like token and the reply can end empty (the "Say hi."
//! collapse X-VL documents for an external implementation). The design adopted from that reference (PLAN/DSV41_XREF_SPARKDS41F.md
//! §2.1-2.2), in the shell's terms:
//! * a token sampled at step t is PROVISIONAL: it is not committed (not streamed, not counted as generated) until the
//!   forward has run it as an input row at step t+1;
//! * if that row's sink magnitude is at or above the threshold, the token is rejected: the lane is truncated back to the
//!   row's position and the NEXT ALTERNATE of the draw that produced the token is taken (same logits, rejected ids
//!   excluded, an independent draw), at most `retries` times; after that the sunk token is committed anyway and counted
//!   (`fallthrough`);
//! * a token that ends the request (stop id, max-tokens, context) is committed at once: nothing follows to verify it.
//! Cost: every token is emitted one step later (TTFT + one decode step for a lone request); the step count is unchanged.
//! Drafter rows of speculation (a flagged row masks its draft in the previous row) belong to leg DS: the shell has no
//! speculation, so only draft zero exists here. A flagged LAST PROMPT row is counted (`row="prompt"`) and not acted on.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SinkCfg {
    pub enabled: bool,
    pub threshold: f32,
    pub retries: u32,
}

impl SinkCfg {
    pub const OFF: SinkCfg = SinkCfg { enabled: false, threshold: 85_000.0, retries: 3 };
    /// Read the three flags.
    pub fn from_opts() -> SinkCfg {
        let on = crate::opts::lookup("dsv41-sink-reject").map_or(false, crate::opts::on);
        let f = |n: &str, d: f32| crate::opts::lookup(n).and_then(crate::opts::get).and_then(|v| v.parse().ok()).unwrap_or(d);
        SinkCfg { enabled: on, threshold: f("dsv41-sink-threshold", 85_000.0), retries: f("dsv41-sink-retries", 3.0) as u32 }
    }
    /// Is this row a sink? (an unmeasured row never is)
    pub fn flagged(&self, sink: Option<f32>) -> bool { self.enabled && sink.map_or(false, |m| m >= self.threshold) }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// the row is clean: commit the provisional token
    Accept,
    /// reject it and take the next alternate (`rejected_so_far` + 1 alternates have been tried)
    Reject,
    /// retries exhausted: commit the sunk token and count it
    Fallthrough,
}

/// `flagged`: the row the provisional token was fed at is a sink; `rejected_so_far`: alternates already refused at this position.
pub fn verdict(flagged: bool, rejected_so_far: u32, retries: u32) -> Verdict {
    if !flagged { Verdict::Accept } else if rejected_so_far < retries { Verdict::Reject } else { Verdict::Fallthrough }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verdicts() {
        assert_eq!(verdict(false, 0, 3), Verdict::Accept);
        assert_eq!(verdict(true, 0, 3), Verdict::Reject);
        assert_eq!(verdict(true, 2, 3), Verdict::Reject);
        assert_eq!(verdict(true, 3, 3), Verdict::Fallthrough);
        assert_eq!(verdict(true, 0, 0), Verdict::Fallthrough);
    }

    #[test]
    fn disabled_or_unmeasured_rows_are_never_flagged() {
        assert!(!SinkCfg::OFF.flagged(Some(1e9)));
        let on = SinkCfg { enabled: true, ..SinkCfg::OFF };
        assert!(on.flagged(Some(85_000.0)));
        assert!(!on.flagged(Some(84_999.0)));
        assert!(!on.flagged(None));
    }
}
