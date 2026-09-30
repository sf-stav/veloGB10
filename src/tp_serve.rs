//! TP=2 serving wire protocol (TP item A): the control plane for the TP=2 OpenAI server.
//!
//! The cluster sync TCP connection (`cluster::run_head_session` / `cluster::run_node`) is RETAINED
//! for the whole serving session instead of being dropped after `Msg::Config`. All serving traffic
//! flows over it as length-prefixed JSON (the same framing as the sync protocol, via
//! `cluster::send_json` / `cluster::recv_json`), one `ServingMsg` at a time:
//!
//! ```text
//!   head → node:  CalibTable, Step { Admit | Cancel }*, Shutdown
//!   node → head:  Ready
//! ```
//!
//! The `Step` message is the per-decode-step rendezvous: the head's `BatchScheduler::run_tp_head`
//! ships one per step (even when it carries no events), the node's `run_tp_mirror` applies it and
//! runs the identical `decode_step`, so both schedulers hold identical state by construction. The
//! RDMA link (`net::TpLink`) is untouched by any of this — it stays the inference data plane.

use crate::batch::{BatchRequest, TokEvent};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

/// A `BatchRequest` minus its token channel — everything the mirror needs to replay an admission
/// with bit-identical scheduler state. `seed: None` stays None: the default (DefaultHasher of the
/// prompt) is derived identically on both ranks.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct WireRequest {
    pub prompt: Vec<u32>,
    pub max_new: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub rep_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub seed: Option<u64>,
    pub ckpt_at: Option<usize>,
    /// S8F routing domain (S6F adjudication): rides the wire so a future TP-DF2 lane split is
    /// SPMD-identical; a pure function of the prompt, default `General`.
    pub domain: crate::batch::Domain,
    #[serde(default)]
    pub min_new: usize,
    #[serde(default)]
    pub ignore_eos: bool,
    /// TP-D (EXL3 serve): the request's min-p (the EXL3 sampler honours it; the NVFP4 mirror's
    /// `into_request` keeps 0.0 — that sampler has none). Older peers: absent = 0.
    #[serde(default)]
    pub min_p: f32,
}

/// One scheduler-visible event within a step. Ordering inside a step: all Admits (in admit order),
/// then all Cancels. Cancel lane indices refer to the POST-admission front-packed lane table
/// (admissions only append past the active region, so they never renumber live lanes).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum TpEvent {
    Admit(WireRequest),
    Cancel { lane: usize },
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StepEvents {
    pub step: u64,
    pub events: Vec<TpEvent>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum ServingMsg {
    /// The head's measured MTP cost-per-depth tables, per context bucket (E17): (measurement ctx,
    /// r table at that ctx). The node runs the SAME SPMD calibration forwards (the all-reduces are
    /// barriers the head waits on) but discards its own tables, so both ranks drive `MtpPolicy`
    /// from one identical set of numbers.
    /// S9F (the TP-DF2 leg): `df2_round` = the head's DFlash2-round LOAD OUTCOME (the config only
    /// ships the intent; the head knows the artifact's residency only after its own load). The
    /// node must load the round iff the head did — a one-sided round is a lane-branch mismatch
    /// that desyncs the verify all-reduces (the node panics loudly if its own load contradicts
    /// the shipped flag rather than silently serving a different lane).
    CalibTable { ctx_r: Vec<(u32, Vec<(u32, f32)>)>, df2_round: bool },
    /// The head's measured DSpark depth-cost table (item 3.3): (drafted rows D, r(D) = step cost /
    /// decode cost). Both ranks run the identical SPMD calibration forwards (the verify all-reduces
    /// are barriers the head waits on); the node discards its own timings and drives the depth
    /// policy from the head's table — a per-rank timing difference must never diverge the depth
    /// decision (the serve loop is SPMD).
    DsparkRd { table: Vec<(u32, f32)> },
    /// Node → head: scheduler built (graph capture done), mirror loop armed. The head binds the
    /// HTTP listener only after receiving this, so no client request can arrive before the mirror
    /// is in lockstep.
    Ready,
    /// The per-step rendezvous. Sent every executed step, even with an empty event list.
    Step(StepEvents),
    /// Head's request channel closed (server shutdown) and all lanes drained: end the session.
    Shutdown,
    /// TP-D (EXL3 serve): a head-authoritative boolean decided INSIDE a step (today: "the client
    /// left — cancel this prefill at this chunk boundary"). The node blocks on it at the same
    /// program point and adopts the head's value, so both ranks stop (or continue) the SPMD
    /// prefill at the same chunk.
    HeadFlag { v: bool },
}

impl From<&BatchRequest> for WireRequest {
    fn from(r: &BatchRequest) -> Self {
        WireRequest {
            prompt: r.prompt.clone(),
            max_new: r.max_new,
            temperature: r.temperature,
            top_p: r.top_p,
            top_k: r.top_k,
            rep_penalty: r.rep_penalty,
            presence_penalty: r.presence_penalty,
            frequency_penalty: r.frequency_penalty,
            seed: r.seed,
            ckpt_at: r.ckpt_at,
            domain: r.domain,
            min_new: r.min_new,
            ignore_eos: r.ignore_eos,
            min_p: r.min_p,
        }
    }
}

impl WireRequest {
    /// Rebuild a `BatchRequest` on the mirror. `tx` is a dummy channel whose receiver is held by
    /// the mirror forever — the node's `tx.is_closed()` must NEVER fire on its own, so cancels
    /// arrive exclusively as wire events.
    pub fn into_request(self, tx: mpsc::UnboundedSender<TokEvent>) -> BatchRequest {
        BatchRequest {
            prompt: self.prompt,
            max_new: self.max_new,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            rep_penalty: self.rep_penalty,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            min_p: 0.0, // the NVFP4 sampler has no min-p (WP08)
            tx,
            seed: self.seed,
            ckpt_at: self.ckpt_at,
            domain: self.domain,
            min_new: self.min_new,
            ignore_eos: self.ignore_eos,
            received_at: std::time::Instant::now(),
            image_embeds: None,
            image_spans: Vec::new(),
            // W2: the schema FSM lives on the HEAD (that is where the sampler runs); the node's
            // mirror lane is unconstrained by construction.
            schema: None,
        }
    }
}

/// Send one serving message (length-prefixed JSON, same framing as the cluster sync).
pub fn send_serving(w: &mut impl std::io::Write, m: &ServingMsg) -> anyhow::Result<()> {
    crate::cluster::send_json(w, m)
}

/// Receive one serving message. An `Err` here (clean EOF included) means the session is over.
pub fn recv_serving(r: &mut impl std::io::Read) -> anyhow::Result<ServingMsg> {
    crate::cluster::recv_json(r)
}
