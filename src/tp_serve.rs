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
    /// The head's measured MTP cost-per-depth table. The node runs the SAME SPMD calibration
    /// forwards (the all-reduces are barriers the head waits on) but discards its own table, so
    /// both ranks drive `MtpPolicy` from one identical set of numbers.
    CalibTable { r: Vec<(u32, f32)> },
    /// Node → head: scheduler built (graph capture done), mirror loop armed. The head binds the
    /// HTTP listener only after receiving this, so no client request can arrive before the mirror
    /// is in lockstep.
    Ready,
    /// The per-step rendezvous. Sent every executed step, even with an empty event list.
    Step(StepEvents),
    /// Head's request channel closed (server shutdown) and all lanes drained: end the session.
    Shutdown,
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
            tx,
            seed: self.seed,
            ckpt_at: self.ckpt_at,
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
