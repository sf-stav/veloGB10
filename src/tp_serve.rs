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
    /// VIS-4: the request's image spans (empty = text). Their rows follow the Step frame on the
    /// control stream as raw little-endian fp16 [sum(num_tokens)][hidden] (`image_bytes` bytes, FNV-1a
    /// 64 `image_digest`) — the binary channel; the JSON frame never carries the rows.
    #[serde(default)]
    pub image_spans: Vec<crate::vision_encoder::ImageSpan>,
    #[serde(default)]
    pub image_bytes: u64,
    #[serde(default)]
    pub image_digest: u64,
    /// LANERES3: the reservation policy's scoping for THIS admit, decided by the head
    /// (`lane_reserve::plan`) and shipped so the node's lane pick is the head's pick:
    /// `admit_req`'s choice is a pure function of replicated state, but the policy's
    /// `allowed` set restricts it, and both ranks must restrict identically or their
    /// slot tables diverge. `None` = unscoped (policy off / baseline admission).
    #[serde(default)]
    pub reserve: Option<Scoping>,
}

/// LANERES3: the head-decided admission scoping that rides `WireRequest::reserve`.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct Scoping {
    /// Lane indices `admit_req` may pick from (the policy's `allowed` set).
    pub allowed: Vec<usize>,
    /// `lane_reserve::Kind` as its discriminant (0 own / 1 unreserved / 2 forced).
    pub kind: u8,
}

/// VIS-4: the image rows of a request as the TP wire carries them (fp16 = the precision the splice
/// uses on every rank), and their FNV-1a 64 digest.
pub fn image_rows_wire(embeds: &[f32]) -> (Vec<u8>, u64) {
    let mut out = Vec::with_capacity(embeds.len() * 2);
    for v in embeds {
        out.extend_from_slice(&half::f16::from_f32(*v).to_bits().to_le_bytes());
    }
    let d = fnv64(&out);
    (out, d)
}

/// VIS-4: FNV-1a 64 of a byte payload.
pub fn fnv64(b: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &x in b {
        h ^= x as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
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
    /// FLASHNEXT-DF2-117 (ISSUES 117): the HEAD's sampled DFlash2 draft for one solo round — the drawn
    /// draft tokens, their drawn weights and the selector walk's full candidate table (per position:
    /// the 16 candidate ids + their softmax weights, f32 as bits) — shipped to every node at the same
    /// program point of the same step. A replicated drafter drawing its own multinomial on residuals
    /// that differ across ranks in the low bits produced different drafts (rank divergence at the
    /// lockstep agree); with ONE draft and ONE proposal law on every rank the real-q verify is
    /// rank-identical. `lane`/`p` are the tripwire: a node at another lane or position is out of step.
    Df2Draft { lane: u32, p: u64, tokens: Vec<u32>, q_rows: Vec<u32>, cand_tok: Vec<u32>, cand_q: Vec<u32> },
    /// LANERES3: one lockstep bit of the spill tier. Sent by EVERY rank at the same
    /// admission-phase program point (capture: my local snapshot write succeeded;
    /// restore: my local snapshot read succeeded), send+recv rendezvous, verdict =
    /// the AND of all ranks — so the replicated spill policy only ever records keys
    /// whose local disk state every rank agrees on. A disk that diverged can never
    /// desync the pair: the key is dropped everywhere and counted.
    SpillFlag { ok: bool },
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
            image_spans: r.image_spans.clone(),
            image_bytes: 0,   // set by the head when it ships the rows (exl3_serve)
            image_digest: 0,
            reserve: None, // LANERES3: set by the head when the plan scoped the admit
        }
    }
}

impl WireRequest {
    /// Rebuild a `BatchRequest` on the mirror. `tx` is a dummy channel whose receiver is held by
    /// the mirror forever — the node's `tx.is_closed()` must NEVER fire on its own, so cancels
    /// arrive exclusively as wire events.
    pub fn into_request(self, tx: crate::server::TokTx) -> BatchRequest {
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
            image_embeds: None,              // VIS-4: the node fills it from the binary payload
            image_spans: self.image_spans,
            // W2: the schema FSM lives on the HEAD (that is where the sampler runs); the node's
            // mirror lane is unconstrained by construction.
            schema: None,
            // log probabilities are computed on the head only (the TP path does not send them)
            logprobs: None,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// LANERES3: an old head's wire frames (no `reserve` field) still decode with scoping
    /// absent — the baseline admission path — while a scoped frame round-trips exactly.
    #[test]
    fn wire_request_reserve_serde_default_none() {
        let old = serde_json::json!({
            "prompt": [1, 2, 3], "max_new": 8, "temperature": 1.0, "top_p": 0.95,
            "top_k": 20, "rep_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
            "seed": 7, "ckpt_at": null, "domain": "General",
        });
        let w: WireRequest = serde_json::from_value(old).expect("old frame decodes");
        assert!(w.reserve.is_none(), "old frames must decode without scoping");
        let scoped = WireRequest {
            reserve: Some(Scoping { allowed: vec![2, 3], kind: 0 }),
            ..w
        };
        let s = serde_json::to_string(&scoped).unwrap();
        let back: WireRequest = serde_json::from_str(&s).unwrap();
        assert_eq!(back.reserve, Some(Scoping { allowed: vec![2, 3], kind: 0 }));
    }

    /// FLASHNEXT-DF2-117: the head's sampled draft frame round-trips exactly (f32 weights ride as bits).
    #[test]
    fn df2_draft_frame_serde_round_trips_exactly() {
        let q = [0.5f32, 0.25, 1e-7, 0.0];
        let m = ServingMsg::Df2Draft { lane: 3, p: 811, tokens: vec![13, 4844, 6088], q_rows: q.iter().map(|x| x.to_bits()).collect(),
                                       cand_tok: vec![1, 2, 3, 4], cand_q: q.iter().map(|x| x.to_bits()).collect() };
        let s = serde_json::to_string(&m).unwrap();
        assert!(s.contains("Df2Draft"), "{s}");
        match serde_json::from_str::<ServingMsg>(&s).unwrap() {
            ServingMsg::Df2Draft { lane, p, tokens, q_rows, cand_tok, cand_q } => {
                assert_eq!((lane, p, tokens, cand_tok), (3, 811, vec![13, 4844, 6088], vec![1, 2, 3, 4]));
                assert_eq!(q_rows.iter().map(|&b| f32::from_bits(b)).collect::<Vec<_>>(), q);
                assert_eq!(cand_q.iter().map(|&b| f32::from_bits(b)).collect::<Vec<_>>(), q);
            }
            other => panic!("decoded {other:?}"),
        }
    }
    /// LANERES3: the spill tier's control bit serializes on the same stream as the other
    /// control frames (a node that receives one mid-admit must be able to name it).
    #[test]
    fn spill_flag_serde() {
        let m = ServingMsg::SpillFlag { ok: true };
        let s = serde_json::to_string(&m).unwrap();
        assert!(s.contains("SpillFlag"), "{s}");
        let back: ServingMsg = serde_json::from_str(&s).unwrap();
        matches!(back, ServingMsg::SpillFlag { ok: true });
    }
}
