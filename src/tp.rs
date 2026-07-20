//! TP=2 runtime context: rank/world + the RDMA data-plane link, brought up AFTER the cluster has
//! discovered the peer and synced the model. Head = rank 0 (listens for the QP handshake), node =
//! rank 1 (connects to the head's RoCE IP that it saw during the TCP sync).
//!
//! The link (`net::TpLink`) is the inference data plane for the sharded forward's all-reduces.

use crate::net::TpLink;
use anyhow::Result;
use std::net::IpAddr;

/// net_shim QP-bootstrap TCP port (distinct from the cluster control plane on 29500).
pub const TP_PORT: u16 = 29600;
/// One slot must hold the widest all-reduce payload we will ever ship: hidden * 4 (FP32) * verify batch.
/// 128 KB covers hidden=5120 at FP32 up to batch 6, or bf16 up to batch 12 — i.e. an MTP verify at any
/// depth we would plausibly run. Cost is ~4 MB pinned for both rings; the reason not to size it huge is
/// that the ring addresses are baked into a captured CUDA graph, so changing it later forces a
/// re-capture. Sized once, deliberately. (At 64 KB a batch-8 bf16 forward failed the payload guard.)
pub const TP_SLOT_BYTES: usize = 128 * 1024;
pub const GID_IDX: i32 = 3;
const DEFAULT_RDMA_DEV: &str = "rocep1s0f1";

// ---------------------------------------------------------------------------------------------------
// TP settings distribution (TP item C)
// ---------------------------------------------------------------------------------------------------
// The contract is "nodes just run `--node`": the head ships its TP config to the node during the
// cluster sync (`Msg::Config`), and the node runs with ZERO GB10_TP_* env vars yet reproduces the
// head's behavior. Env vars remain as overrides for benches. Resolution rule at every consumer:
// env var present → env wins; else this process-global config (if installed); else default (same
// default as the no-env behavior). Flags are presence-based, matching the old `is_ok()` semantics.

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TpConfig {
    pub config_version: u32,          // = 2
    pub shard_mixers: bool,           // GB10_TP_SHARD_MIXERS
    pub graph: bool,                  // GB10_TP_GRAPH
    pub fp32_partials: bool,          // GB10_TP_FP32_PARTIALS
    pub trace: bool,                  // GB10_TP_TRACE
    pub mtp: bool,                    // GB10_TP_MTP
    pub mtp_depth: Option<usize>,     // GB10_TP_MTP_DEPTH
    pub batch_probe: Option<usize>,   // GB10_TP_BATCH_PROBE
    pub step_probe: Option<usize>,    // GB10_TP_STEP_PROBE
    // ---- v2: serving mode (TP item A). `from_env` fills these with defaults (bench mode is
    // unaffected); the head's `--server --tp` branch fills them from the server args. The node
    // needs them to build a BatchScheduler identical to the head's with ZERO env of its own.
    pub mode_serve: bool,             // false = one-shot bench session; true = resident OpenAI server
    pub max_seq_len: usize,           // head's (clamped) --max-seq-len = both ranks' kv_stride
    pub max_batch: usize,             // head's --max-batch; also the all-reduce payload width
    pub prefix_cache: bool,           // head's --prefix-cache
    pub ngram_draft: usize,           // head's --ngram-draft
    pub tree_draft: bool,             // head's --tree-draft
    pub mtp_lanes: bool,              // head's --mtp-lanes
    pub mtp_force: Option<bool>,      // head's --mtp=on|off (None = auto)
    pub mtp_depth_pin: Option<usize>, // head's --mtp-depth
    pub no_decode_graphs: bool,       // head's GB10_NO_DECODE_GRAPHS (env-read; node installs as env)
    pub cpu_sample: bool,             // head's RUST_INFER_CPU_SAMPLE (env-read; node installs as env)
    pub eos: Vec<u32>,                // head's stop-token set (node has no tokenizer)
    pub calib_prompt: Vec<u32>,       // head-encoded "The capital of France is" probe ids
}

impl TpConfig {
    /// Snapshot the GB10_TP_* env vars (flags = presence; probes/depth = parse). Serving fields get
    /// their v1-compatible defaults — a bench config is indistinguishable from before.
    pub fn from_env() -> Self {
        TpConfig {
            config_version: 2,
            shard_mixers: std::env::var("GB10_TP_SHARD_MIXERS").is_ok(),
            graph: std::env::var("GB10_TP_GRAPH").is_ok(),
            fp32_partials: std::env::var("GB10_TP_FP32_PARTIALS").is_ok(),
            trace: std::env::var("GB10_TP_TRACE").is_ok(),
            mtp: std::env::var("GB10_TP_MTP").is_ok(),
            mtp_depth: std::env::var("GB10_TP_MTP_DEPTH").ok().and_then(|v| v.parse().ok()),
            batch_probe: std::env::var("GB10_TP_BATCH_PROBE").ok().and_then(|v| v.parse().ok()),
            step_probe: std::env::var("GB10_TP_STEP_PROBE").ok().and_then(|v| v.parse().ok()),
            mode_serve: false,
            max_seq_len: 0,
            max_batch: 0,
            prefix_cache: false,
            ngram_draft: 0,
            tree_draft: false,
            mtp_lanes: false,
            mtp_force: None,
            mtp_depth_pin: None,
            no_decode_graphs: false,
            cpu_sample: false,
            eos: Vec::new(),
            calib_prompt: Vec::new(),
        }
    }
}

static TP_CONFIG: std::sync::OnceLock<TpConfig> = std::sync::OnceLock::new();

/// Install the process-global TP config (head: its env snapshot; node: the head's `Msg::Config`).
pub fn set_tp_config(c: TpConfig) {
    eprintln!("[tp] config installed: shard_mixers={} graph={} fp32_partials={} trace={} mtp={} \
               mtp_depth={:?} batch_probe={:?} step_probe={:?} mode_serve={}",
              c.shard_mixers, c.graph, c.fp32_partials, c.trace, c.mtp,
              c.mtp_depth, c.batch_probe, c.step_probe, c.mode_serve);
    if TP_CONFIG.set(c).is_err() {
        eprintln!("[tp] WARNING: TP config already installed — keeping the first");
    }
}

/// The installed TP config, if any (None on a plain single-box run — env/defaults apply there).
pub fn tp_config() -> Option<&'static TpConfig> { TP_CONFIG.get() }

fn rdma_dev() -> String {
    std::env::var("GB10_RDMA_DEV").ok()
        .and_then(|s| s.split(',').next().map(|x| x.trim().to_string()))
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| DEFAULT_RDMA_DEV.to_string())
}

pub struct TpContext {
    pub rank: i32,
    pub world: i32,
    pub link: TpLink,
}

impl TpContext {
    /// Head side (rank 0): listen for the node's QP handshake on the RoCE device.
    pub fn bring_up_head() -> Result<Self> {
        eprintln!("[tp] rank 0/2 — bringing up RDMA data-plane link on {} (listening) ...", rdma_dev());
        let link = TpLink::connect(0, "", TP_PORT, &rdma_dev(), GID_IDX, TP_SLOT_BYTES)?;
        eprintln!("[tp] rank 0/2 — link UP");
        Ok(TpContext { rank: 0, world: 2, link })
    }

    /// Node side (rank 1): connect the QP to the head's RoCE IP (seen during the cluster sync).
    pub fn bring_up_node(head_ip: IpAddr) -> Result<Self> {
        eprintln!("[tp] rank 1/2 — connecting RDMA data-plane link to head {head_ip} on {} ...", rdma_dev());
        let link = TpLink::connect(1, &head_ip.to_string(), TP_PORT, &rdma_dev(), GID_IDX, TP_SLOT_BYTES)?;
        eprintln!("[tp] rank 1/2 — link UP");
        Ok(TpContext { rank: 1, world: 2, link })
    }

    /// Broadcast the prompt + decode budget from head (rank 0) to node (rank 1) over the live link,
    /// so the node runs the identical SPMD generate loop with zero manual config. Frame (i32):
    /// `[n_prompt, max_new, tok0..tok_{n-1}]`. The exchange is symmetric, so both fill and ship their
    /// whole slot; the head ignores what it reads back, the node reads the head's frame.
    /// Head calls with `Some((prompt, max_new))`; node with `None`.
    pub fn broadcast_prompt(&mut self, head: Option<(&[u32], usize)>) -> Result<(Vec<u32>, usize)> {
        let cap = TP_SLOT_BYTES / 4;                     // i32 slots per RDMA buffer
        match head {
            Some((prompt, max_new)) => {
                anyhow::ensure!(prompt.len() + 2 <= cap, "prompt too long for one TP frame ({} toks)", prompt.len());
                {
                    let slot = self.link.send_host_mut::<i32>(cap);
                    for x in slot.iter_mut() { *x = 0; }
                    slot[0] = prompt.len() as i32;
                    slot[1] = max_new as i32;
                    for (i, &t) in prompt.iter().enumerate() { slot[2 + i] = t as i32; }
                }
                self.link.exchange(cap * 4)?;
                Ok((prompt.to_vec(), max_new))
            }
            None => {
                { let slot = self.link.send_host_mut::<i32>(cap); for x in slot.iter_mut() { *x = 0; } }
                self.link.exchange(cap * 4)?;
                let recv = self.link.recv_host::<i32>(cap);
                let n = recv[0] as usize;
                let max_new = recv[1] as usize;
                anyhow::ensure!(n + 2 <= cap, "bad prompt frame from head (n={n})");
                let prompt: Vec<u32> = recv[2..2 + n].iter().map(|&x| x as u32).collect();
                Ok((prompt, max_new))
            }
        }
    }

    /// Consume the context, handing the link to a GpuModel for the sharded forward. Returns
    /// `(rank, world, link)` so the caller can call `GpuModel::attach_tp`.
    pub fn into_parts(self) -> (i32, i32, TpLink) { (self.rank, self.world, self.link) }

    /// Sanity: exchange a rank-stamped probe and confirm we received the peer's — proves the data-plane
    /// link is live end to end (discover → sync → RDMA), before any sharded forward runs.
    pub fn sanity(&mut self) -> Result<()> {
        let stamp = 0xA0u8 + self.rank as u8;
        for b in self.link.send_host_mut::<u8>(16).iter_mut() { *b = stamp; }
        self.link.exchange(16)?;
        let peer = self.link.recv_host::<u8>(16)[0];
        let expect = 0xA0u8 + (1 - self.rank) as u8;
        if peer != expect { anyhow::bail!("tp sanity: peer stamp {peer:#x}, expected {expect:#x}"); }
        eprintln!("[tp] rank {}/2 — data-plane all-reduce link SANE (peer stamp {peer:#x})", self.rank);
        Ok(())
    }
}
