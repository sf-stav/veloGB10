//! TP knobs and preconditions: decode / prefill transport modes, the vocab-parallel / seq-parallel / prefill-overlap
//! codes, the world>2 flag refusals, the rail-2 port plan and the rail preflight checks. See `mod.rs` for the map.

use super::*;

/// --tp-dec-xport — the decode all-reduce transport (TP-G; bitwise in every mode):
///   1 / on / unset   : DEFAULT (TP-G A/B GO) — K1 folded into the producers' last block + multi-block
///                      K2 with the f16 cvt folded; receive = the proxy's cpu_done OR the GPU-side
///                      payload-tail proof (--tp-dec-grecv, default 1: A/B 3 — the floor ties, the
///                      proxy-jitter outliers go; =0 = cpu_done only)
///   0 / off          : the serial single-block K1/K2 pair + xq_cvt_f32_f16 (the TP-B..F path; escape)
/// Rides TpConfig (the EXL3 env snapshot) so both ranks resolve the same value. Diagnostics:
/// --tp-dec-grecv (default 1; 0 = cpu_done only), --tp-dec-blocks (K2 grid, default 8).
pub fn dec_xport_mode() -> u8 {
    match crate::opts::var(crate::opt!("tp-dec-xport")).map(|v| v.trim().to_ascii_lowercase()) {
        Ok(v) if v == "0" || v == "off" || v == "false" => 0,
        _ => 1,
    }
}

/// --tp-route-fold — TP-G (T2 lever #5): on an EP rank, xq_moe_route_ep (EP id remap/filter) is
/// folded into the router fold's route tail (xq_router_fold_c_ep; the tables are identical, one launch
/// fewer per MoE layer). 1 / unset = on (DEFAULT, TP-G A/B GO), 0 = the separate route_ep launch
/// (escape). Rides TpConfig.
pub fn route_fold_on() -> bool {
    static G: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        !matches!(crate::opts::var(crate::opt!("tp-route-fold")).map(|v| v.trim().to_ascii_lowercase()).as_deref(),
                  Ok("0") | Ok("off") | Ok("false"))
    });
    *G
}

/// --tp-vp-head — TP-H #4 (vocab-parallel lm_head, greedy tail; bitwise in every mode):
///   1 / on / unset : DEFAULT — at world 2 each rank runs the lm_head GEMM over its OWN 124160-column
///                    shard (bitwise the replicated head's columns), reduces each row to one u64
///                    argmax key (value desc, lowest GLOBAL id), and the two ranks exchange the m keys
///                    through the decode doorbell (xq_argmax_rows_vp + K1 fold, xq_tp_wait_keys K2):
///                    the merged ids are bitwise the replicated argmax. Sampled / penalized /
///                    ratio-rule rows keep the replicated head (the sampler needs the full row).
///   0 / off        : the replicated lm_head + xq_argmax_rows everywhere (the TP-G path; escape)
/// Rides TpConfig (the EXL3 env snapshot). TP-4C: world 2 or 4 — at world 4 each rank owns V/4 columns and
/// the keys are merged over `rounds` exchanges (xq_tp_wait_keys merges in place each round; the plain K1
/// republishes the merged keys between rounds): the total order (value desc, lowest id) makes the merge
/// association-free, so the ids are bitwise the replicated argmax at every world.
pub fn vp_head_on() -> bool {
    static G: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        !matches!(crate::opts::var(crate::opt!("tp-vp-head")).map(|v| v.trim().to_ascii_lowercase()).as_deref(),
                  Ok("0") | Ok("off") | Ok("false"))
    });
    *G
}

/// --tp-dh-shard — TP-H #3 (sharded MTP-head screen; bitwise in every mode):
///   1 / on / unset : DEFAULT — at world 2 each rank streams ONLY its half of the draft head's 2-bit
///                    screen (256 of the 512 128-column blocks: 21 MB instead of 42 MB per draft pass)
///                    into a zero-padded block-maxima buffer, the last block publishes it through the
///                    decode doorbell (xq_dh_screen_k1), and xq_tp_wait_add_dec (lower + upper of a
///                    value and an exact 0) reassembles the replicated screen's bmax array bit for bit.
///                    Candidate selection + the exact rescore stay replicated, so drafts, dconf and
///                    every trajectory are bitwise the replicated head's.
///   0 / off        : the replicated screen (the TP-G / TP-H #4 path; escape)
/// Rides TpConfig (the EXL3 env snapshot). TP-4C: world 2 or 4 (at world 4 each rank streams a QUARTER of
/// the blocks and the two-exchange K2 reassembles the array; bitwise — every entry has one non-zero term).
pub fn dh_shard_on() -> bool {
    static G: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        !matches!(crate::opts::var(crate::opt!("tp-dh-shard")).map(|v| v.trim().to_ascii_lowercase()).as_deref(),
                  Ok("0") | Ok("off") | Ok("false"))
    });
    *G
}

/// TP-H2 — `--tp-vp-sampled <on|off>` (a CLI flag, no env): the vocab-parallel lm_head for the rows the
/// TP-H #4 greedy tail refuses (sampled / penalized / ratio-rule plain steps and verify rows). Each rank
/// runs the lm_head GEMM on its OWN shard (V / world columns, bitwise the replicated head's columns), the
/// shard rows are ALL-GATHERED into the full logits rows through the decode doorbell, and the UNCHANGED
/// penalty / argmax / sampler kernels run on the reassembled rows — bitwise the replicated head's buffer by
/// construction. World 2: one exchange per group of <= 4 rows (xq_vp_gather_k1 / xq_vp_gather_k2, TP-H2).
/// World 4 (TP-4H2): recursive doubling, two exchanges per group of <= 4 rows (xq_vp_gather4_k1 /
/// xq_vp_gather4_k2, src/tp_vpgather.rs). Needs the TP-H #4 shard (--tp-vp-head on, world 2 or 4). The head
/// resolves it and ships the code in TpConfig.exl3_vp_sampled (v21); the node installs the head's value
/// (`set_vp_sampled`); the code joins the TP boot agree hash. DEFAULT ON at world 2 (TP-H2 A/B: sampled
/// thinking-on class -0.72 ms/round at identical trajectories and identical sampled bytes, greedy classes
/// untouched by construction and tie; API thinking-on median +2.4 %) and at world 4 (owner ruling D-TP4-3,
/// 2026-10-01: sampled requests take the fastest path; hardware-UNVERIFIED until PLAN/TP-4H2_GATES.md has
/// run). `--tp-vp-sampled off` = the replicated head for those rows (the TP-H..TP-I3 path).
pub const VP_SAMPLED_DEFAULT: u32 = 1;
static VP_SAMPLED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(VP_SAMPLED_DEFAULT);
pub fn set_vp_sampled(code: u32) { VP_SAMPLED.store(code, AtOrd::Relaxed); }
/// The resolved code (1 = on, 0 = off).
pub fn vp_sampled_code() -> u32 { VP_SAMPLED.load(AtOrd::Relaxed) }
pub fn vp_sampled_on() -> bool { vp_sampled_code() != 0 }
pub fn vp_sampled_desc(code: u32) -> &'static str { if code != 0 { "on" } else { "off" } }
/// Parse the flag's value (None = the flag is absent = the default).
pub fn parse_vp_sampled(v: Option<&str>) -> Result<u32> {
    let Some(s) = v.map(|s| s.trim().to_ascii_lowercase()) else { return Ok(VP_SAMPLED_DEFAULT) };
    match s.as_str() {
        "on" | "1" | "true" => Ok(1),
        "off" | "0" | "false" => Ok(0),
        _ => anyhow::bail!("--tp-vp-sampled takes on|off, got '{s}'"),
    }
}
/// The head's `--tp-vp-sampled` code at `world`. Worlds <= 2 return `parse_vp_sampled` unchanged (the TP-H2
/// behaviour, no note). World 4 (TP-4H2: the N-way gather kernels exist): absent => the default, ON;
/// explicit `on` => ON; explicit `off` => OFF; no note. Any other world > 2 has no gather kernels: absent
/// => OFF (the replicated head for the sampled / penalized / ratio-rule rows, correct and still resident;
/// the greedy vp tail is unaffected), explicit `on` => refused loudly (never a silent downgrade), explicit
/// `off` => OFF. Returns (code, note) where `note` is a banner line to print when the default was overridden.
pub fn resolve_vp_sampled(world: u32, v: Option<&str>) -> Result<(u32, Option<&'static str>)> {
    let code = parse_vp_sampled(v)?;
    if world <= 2 || world == 4 { return Ok((code, None)); }
    match (v.is_some(), code) {
        (true, 1) => anyhow::bail!("--tp-vp-sampled on is a world-2 / world-4 option (the row all-gather has kernels for those \
                                    worlds only; world {world}). Leave it unset or pass --tp-vp-sampled off."),
        (false, _) => Ok((0, Some("--tp-vp-sampled defaults to OFF at this world (the row all-gather has kernels at world 2 and \
                                   4 only): sampled / penalized / ratio-rule rows take the replicated lm_head"))),
        _ => Ok((0, None)),
    }
}

/// TP-SP1 — `--tp-seq-parallel <on|off>` (a CLI flag, no env): SEQUENCE-PARALLEL prefill (TP-I3 §3.2).
/// For a chunk of T..=SP_MAX rows, rank r owns rows row_block(c, r, W) (rank 0 the first ceil(c/2)):
/// every hc_pre (inject + norm, the hc down/up GEMMs with the full chunk's cuBLASLt plan PINNED on the
/// half-row layouts, silu, mix) runs on the own rows only; the mixers, the router and the EP experts
/// keep all rows after an ALL-GATHER of x (K1m of the own rows + a copy-K2 into the peer's rows); the 96
/// all-reduces become REDUCE-SCATTERS (each rank ships only the peer's rows of its partial and sums
/// only its own); the final streams are gathered once per chunk for the MTP head fill. Bitwise the
/// replicated prefill by construction wherever the pinned-plan gate (--probe-exl3-sppin) holds. The
/// layer-0 block stays replicated when the PLE follows it (the PLE reads every row at the layer-1
/// boundary). The head resolves it and ships the code in TpConfig.exl3_seq_parallel (v22); the node
/// installs the head's value (`set_seq_parallel`); an ON code joins the TP boot agree hash (an OFF boot's
/// hash is the pre-SP one). DEFAULT ON since TP-SP2 (6+6 prefill A/B GO, bitwise; `off` = the replicated prefill).
/// TP-SP2: the code is a bit set. SP_HC (bit 0) = the SP-1 base above; SP_ROUTER (bit 1, only with
/// SP_HC) = the MoE router (the fp32 router GEMM + softmax/top-k) on the own rows too, then ONE
/// all-gather of the ids and weights rows (the router kernels are per-row: one ascending-k chain per
/// (row, expert), one CTA per row for the top-k — bitwise by construction; gate --probe-exl3-sppin's
/// EXL3-SP-ROUTER section). Flag values: off = 0, hc = SP_HC, router = SP_HC | SP_ROUTER, on =
/// SEQ_PAR_ON (the posture the SP-2 A/B picked).
pub const SP_HC: u32 = 1;
pub const SP_ROUTER: u32 = 2;
pub const SEQ_PAR_ON: u32 = SP_HC;
pub const SEQ_PAR_DEFAULT: u32 = SEQ_PAR_ON;
static SEQ_PAR: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(SEQ_PAR_DEFAULT);
pub fn set_seq_parallel(code: u32) { SEQ_PAR.store(code, AtOrd::Relaxed); }
pub fn seq_parallel_code() -> u32 { SEQ_PAR.load(AtOrd::Relaxed) }
pub fn seq_parallel_on() -> bool { seq_parallel_code() & SP_HC != 0 }
/// The router extension is live (SP_ROUTER on top of SP_HC).
pub fn seq_parallel_router() -> bool { seq_parallel_code() & (SP_HC | SP_ROUTER) == (SP_HC | SP_ROUTER) }
pub fn seq_parallel_desc(code: u32) -> &'static str {
    match code { 0 => "off", c if c == SP_HC => "hc", c if c == SP_HC | SP_ROUTER => "hc+router", _ => "?" }
}
/// Parse the flag's value (None = the flag is absent = the default).
pub fn parse_seq_parallel(v: Option<&str>) -> Result<u32> {
    let Some(s) = v.map(|s| s.trim().to_ascii_lowercase()) else { return Ok(SEQ_PAR_DEFAULT) };
    match s.as_str() {
        "on" | "1" | "true" => Ok(SEQ_PAR_ON),
        "hc" => Ok(SP_HC),
        "router" | "hc+router" => Ok(SP_HC | SP_ROUTER),
        "off" | "0" | "false" => Ok(0),
        _ => anyhow::bail!("--tp-seq-parallel takes on|off|hc|router, got '{s}'"),
    }
}
/// The split threshold T: chunks below it stay replicated. TP-I3 §3.1: the pinned hc-down halves are
/// bit-identical for every c >= 97 (and differ at 30 values in [21, 96]); 128 = the margin.
pub const SP_MIN_ROWS: usize = 128;
/// The largest chunk that splits: every chunk the prefill scratch can hold — the served width C = 2048
/// and the C1 tail-checkpoint merges up to 2C - 1 = 4095 rows (TP_MAX_ROWS - 1). The pinned-plan gate
/// (--probe-exl3-sppin, default range = [SP_MIN_ROWS, SP_MAX_ROWS]) covers every c of it.
pub const SP_MAX_ROWS: usize = TP_MAX_ROWS - 1;

/// TP-SP: this rank's row block of a sequence-parallel chunk. World 2: `peer` = rank ^ 1's block (one exchange
/// round). TP-4S, world 4: `blocks` = every rank's quarter (ceil(c/4) rows each, rank 0 first), `peer` = the
/// rank ^ 1 quarter (the log line only; the exchanges are `tp_xport::*_w4` over `blocks`). World 2 callers read
/// `own` / `peer` / `c` exactly as before.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SpRows {
    pub own: (usize, usize),
    pub peer: (usize, usize),
    pub c: usize,
    pub world: usize,
    pub rank: usize,
    /// every rank's row block (entries >= `world` are empty (c, c))
    pub blocks: crate::tp_xport::Blocks4,
}

impl SpRows {
    /// The row split of a `c`-row chunk at `world` (2 or 4) for `rank`: `row_block(c, r, world)` for every rank.
    pub(crate) fn new(c: usize, rank: usize, world: usize) -> SpRows {
        let mut blocks = [(c, c); 4];
        for r in 0..world.min(4) { blocks[r] = crate::tp_xport::row_block(c, r, world); }
        SpRows { own: crate::tp_xport::row_block(c, rank, world), peer: crate::tp_xport::row_block(c, rank ^ 1, world),
                 c, world, rank, blocks }
    }
}

/// Rows per gather epoch: 4 x 248,320 B = 993,280 B fits one 1 MiB ring slot (the payload capacity is
/// TP_SLOT_BYTES; the 8 B tail sits after it). Host-checked against n_sh at every launch.
pub(in crate::exl3_forward) const VP_GATHER_ROWS: usize = 4;
/// TP-H2 gate harness ONLY: while set, `vp_greedy` reports false and `vp_gather` true for EVERY row
/// (greedy too), so the gate compares the gathered full rows against the replicated reference.
pub(super) static GATE_VP_GATHER: AtomicBool = AtomicBool::new(false);
/// TP-H2: the shard width (n_sh) of the built lm_head shard, 0 = none. Set once at model load (before
/// any Scratch exists); Scratch::new sizes its gather buffer from it (TP=1: a 1-element placeholder).
static VP_ROWS_COLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub(in crate::exl3_forward) fn set_vp_rows_cols(n: usize) { VP_ROWS_COLS.store(n, AtOrd::Relaxed); }
pub(in crate::exl3_forward) fn vp_rows_cols() -> usize { VP_ROWS_COLS.load(AtOrd::Relaxed) }

/// Reduces at or below this size keep the single-block pair (decode/verify: <= 16 rows x 2560 x 4 B
/// = 160 KiB). Prefill chunks (>= 17 rows) take the pipelined pair when the knob is on.
pub(super) const XPORT_MIN_BYTES: usize = 256 * 1024;

/// --tp-prefill-xport — the prefill all-reduce transport (TP-F; bitwise in every mode):
///   2 / dual / unset : DEFAULT (TP-F A/B GO) — pipelined multi-block K1m/K2m over BOTH ConnectX-7
///                      rails (a second link on --rdma-dev's second entry, default roceP2p1s0f1,
///                      proxy pinned to core TP_AUX_PROXY_CORE); a missing second rail fails the boot
///                      loudly (no silent fallback — set 1 to run on one rail)
///   1 / on           : pipelined multi-block K1m/K2m on the primary rail only
///   0 / off          : the serial single-block K1/K2 pair (the TP-B..E path; diagnostic escape)
/// Rides TpConfig (the EXL3 env snapshot) so both ranks resolve the same value. Tuning
/// (diagnostics): --tp-xport-lookahead (epochs in flight per rail), --tp-xport-blocks.
pub fn prefill_xport_mode() -> u8 {
    match crate::opts::var(crate::opt!("tp-prefill-xport")).map(|v| v.trim().to_ascii_lowercase()) {
        Ok(v) if v == "0" || v == "off" || v == "false" => 0,
        Ok(v) if v == "1" || v == "on" || v == "single" => 1,
        _ => 2,
    }
}

/// TP-I2 item 1 — `--tp-prefill-overlap <off|on|inline[:ROWS]|dual[:ROWS]>` (a CLI flag, no env):
/// comm/compute overlap by row sub-chunks in prefill. At the 96 pipelined prefill reduces (the 48
/// row-parallel mixer out-projections and the 48 EP MoE combines) the per-row consumer that follows
/// the reduce (the mixer's output Hadamard + svh, then the NEXT hc_pre's fused inject + rmsnorm,
/// xq_pf_hc_inj_norm) runs on each landed ROWS-row range (default 128) inside the reduce schedule
/// (`tp_xport::reduce_pipe_hooked`):
///   inline (v1): on the compute stream between the K1m/K2m launches — fills only the wire wait;
///   dual   (v2): the K1m/K2m schedule on the transport's own blocking stream, each landed range
///                handed to the compute stream by an event, so the consumer runs CONCURRENTLY with
///                the K1m/K2m of later epochs (the in-model reduce is K1m/K2m-execution-bound);
///   on = dual:128 (DEFAULT); off = the TP-F path.
/// Bitwise by construction: the same canonical sums, the same per-row kernels on row-offset
/// pointers, every consumed row final. The head resolves it and ships the code in
/// TpConfig.exl3_pf_overlap (rows | dual << 31); the node installs the head's value
/// (`set_pf_overlap`), and the code joins the TP boot agree hash.
pub const PF_OVERLAP_ON_ROWS: u32 = 128;
pub(super) const PF_DUAL: u32 = 1 << 31;
/// TP-I3 (c1): `fold` — K2m folded into the consumers (xq_pf_hc_inj_norm_k2 / exl3_had_svh_k2 at K2m's
/// schedule position, whole-row epochs, one stream). Code = PF_FOLD | 1 (TpConfig v20).
pub(super) const PF_FOLD: u32 = 1 << 30;
/// The default posture: `on` = dual:128 (TP-I2 A/B, 6+6 interleaved boots: 32K prefill +1.0%, every pair
/// positive, 128K +1.0%, 2K +1.4%, decode tie, bitwise on every boot). `--tp-prefill-overlap off` = the TP-F path.
pub const PF_OVERLAP_DEFAULT: u32 = PF_OVERLAP_ON_ROWS | PF_DUAL;
static PF_OVERLAP: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(PF_OVERLAP_DEFAULT);
pub fn set_pf_overlap(code: u32) { PF_OVERLAP.store(code, AtOrd::Relaxed); }
/// The resolved code (rows | dual << 31; 0 = off).
pub fn pf_overlap_code() -> u32 { PF_OVERLAP.load(AtOrd::Relaxed) }
/// Hook granularity in rows (0 = off).
pub fn pf_overlap_rows() -> usize { (pf_overlap_code() & !(PF_DUAL | PF_FOLD)) as usize }
/// v2: the transport runs on its own stream.
pub fn pf_overlap_dual() -> bool { pf_overlap_code() & PF_DUAL != 0 }
/// TP-I3 (c1): K2m folded into the consumers.
pub fn pf_overlap_fold() -> bool { pf_overlap_code() & PF_FOLD != 0 }
/// Human form of a code (boot lines).
pub fn pf_overlap_desc(code: u32) -> String {
    if code & PF_FOLD != 0 { return "fold".to_string(); }
    match code & !PF_DUAL { 0 => "off".to_string(), r => format!("{}:{r}", if code & PF_DUAL != 0 { "dual" } else { "inline" }) }
}
/// Parse the flag's value (None = the flag is absent = the default).
pub fn parse_pf_overlap(v: Option<&str>) -> Result<u32> {
    let Some(s) = v.map(|s| s.trim().to_ascii_lowercase()) else { return Ok(PF_OVERLAP_DEFAULT) };
    let bad = || anyhow::anyhow!("--tp-prefill-overlap takes off|on|inline[:ROWS]|dual[:ROWS]|fold, got '{s}'");
    if s == "fold" { return Ok(PF_FOLD | 1); }
    let (mode, rows) = match s.split_once(':') { Some((m, r)) => (m, Some(r)), None => (s.as_str(), None) };
    let rows: u32 = match rows { Some(r) => r.parse().map_err(|_| bad())?, None => PF_OVERLAP_ON_ROWS };
    anyhow::ensure!((1..=TP_MAX_ROWS as u32).contains(&rows), "--tp-prefill-overlap rows {rows} outside 1..={TP_MAX_ROWS}");
    match mode {
        "off" | "0" => Ok(0),
        "on" | "dual" => Ok(rows | PF_DUAL),
        "inline" => Ok(rows),
        _ => Err(bad()),
    }
}

/// TP-I2 item 1: the per-row consumer a hooked prefill reduce runs on each landed row range — the
/// NEXT hc_pre's fused inject + rmsnorm (xq_pf_hc_inj_norm over rows [r0, r1), row-offset pointers).
/// `y` = the reduce's (f16) output, i.e. the pending inject's sublayer output.
#[derive(Clone, Copy, Debug)]
pub(in crate::exl3_forward) struct PfHook {
    pub hn: u64,
    pub resid: u64,
    pub y: u64,
    pub inj: u64,
    pub norm: u64,
    /// FP4L --fp4l-norms f32: `norm` points at the stored fp32 norm and the consumer is xf_pf_hc_inj_norm_f
    pub norm_f32: bool,
    pub h: usize,
    pub hcn: usize,
    pub eps: f32,
    /// hook granularity (rows)
    pub rows: usize,
    /// v2: the K1m/K2m schedule on the transport's own stream (RowHook::side)
    pub dual: bool,
    /// TP-I3 (c1): K2m folded into the consumer (RowHook::fold)
    pub fold: bool,
}

impl PfHook {
    /// xq_pf_hc_inj_norm over rows [r0, r1): grid hc x B, block 128, every pointer offset by r0 rows —
    /// the kernel's (row, stream) CTA reads/writes only its own row, so this is the full-chunk launch's
    /// arithmetic on those rows, bit for bit.
    pub fn inj_norm(&self, l: &Launcher, r0: usize, r1: usize) -> Result<()> {
        let (h, hc, b) = (self.h, self.hcn, r1 - r0);
        if self.norm_f32 {
            return super::fp4l_fwd::pf_hc_inj_norm_f(l, self.hn + (r0 * hc * h * 2) as u64, self.resid + (r0 * hc * h * 4) as u64,
                                                     self.y + (r0 * h * 2) as u64, self.inj + (r0 * hc * 4) as u64, self.norm,
                                                     h, hc, b, self.eps);
        }
        xqlaunch!(l, "xq_pf_hc_inj_norm", ((hc * b) as u32, 1, 1), (128, 1, 1), 0,
                  (self.hn + (r0 * hc * h * 2) as u64, self.resid + (r0 * hc * h * 4) as u64,
                   self.y + (r0 * h * 2) as u64, self.inj + (r0 * hc * 4) as u64, self.norm,
                   h as i32, hc as i32, b as i32, self.eps))
    }

    /// TP-I3 (c1): the MoE-site fold — K2m (fp32 partial `part`, fp32 wire, f16 rounding) + the inject+norm
    /// over the epoch's rows [r0, r1), at K2m's schedule position (ctx, lag). hc == 4 (the kernel's groups).
    pub fn inj_norm_k2(&self, l: &Launcher, ctx: u64, lag: u32, part: u64, r0: usize, r1: usize) -> Result<()> {
        let (h, hc) = (self.h, self.hcn);
        anyhow::ensure!(hc == 4 && h % 512 == 0 && h <= 4096, "prefill fold: hc {hc} h {h} outside the fused kernel's shape");
        // FP4L --fp4l-norms f32 has no fold twin: the loader turns `fold` into the default dual:128 (one boot line)
        anyhow::ensure!(!self.norm_f32, "prefill fold with fp32 norms (FP4L --fp4l-norms f32): the loader should have \
                                         switched --tp-prefill-overlap fold to dual:128");
        xqlaunch!(l, "xq_pf_hc_inj_norm_k2", ((r1 - r0) as u32, 1, 1), (512, 1, 1), 0,
                  (ctx, lag, part + (r0 * h * 4) as u64, self.hn + (r0 * hc * h * 2) as u64,
                   self.resid + (r0 * hc * h * 4) as u64, self.inj + (r0 * hc * 4) as u64, self.norm,
                   h as i32, self.eps))
    }
}
/// The dual-rail transport's second proxy core (a big X925; the TP host-worker mask excludes it).
pub const TP_AUX_PROXY_CORE: i32 = 18;
/// Rail-2 TCP handshake port at WORLD 2 (its liveness responder takes +1). World > 2 uses `rail2_nway_base`.
pub(super) const TP_AUX_PORT_OFF: u16 = 10;
/// TP-4F2 (`--tp-reduce single`): the single-stage reduce ctx's proxy core — a big X925 (cores 5-9 and 15-19 on
/// GB10; 19 = rail 1, 18 = rail 2, 9 = the scheduler) that no other thread is pinned to; the TP host-worker mask
/// excludes it when the flag is on (exl3_serve `tp_worker_mask`).
pub const TP_SINGLE_PROXY_CORE: i32 = 17;
/// TP-4Z2 (diagnostic counters, cumulative): hints armed by the layer loop / K2 launches that carried a non-empty hint. At capture
/// time they count the baked graph nodes, so after boot `armed == used` means every armed hint reached a K2.
pub static PF_ARMED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PF_USED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// TP-4F2: the single-stage ctx's N-way TCP base = TP_PORT + 40 (rail 1 = +0 .. +13, rail 2 = +20 .. +33 at world 4).
const TP_SINGLE_PORT_OFF: u16 = 40;

/// `--tp-reduce` (TP-4F2; default flipped by TP-4Z1, owner D-TP4-7): the PURE resolution of the flag value.
/// `single` = the decode/verify all-reduce is ONE all-peers stage on a dedicated transport (bit-identical sum);
/// `rd` = the two recursive-doubling stages; `auto` (the default, and an unset flag) = `single` at world 4 on the
/// folded decode transport (`--tp-dec-xport` 1), `rd` everywhere else. An explicit value always wins (`single` at
/// another world is a refusal, see `tp_reduce_refusal`). Rides TpConfig (Spmd), so every rank resolves the same
/// value (it is hashed into the boot agree).
pub fn tp_reduce_resolve(raw: Option<&str>, world: i32, dec_xport: u8) -> bool {
    match raw.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
        Some("rd") => false,
        Some("single") => true,
        _ => world == 4 && dec_xport == 1,
    }
}

/// The resolved schedule at `world` (reads `--tp-reduce` and `--tp-dec-xport`).
pub fn tp_reduce_single_for(world: i32) -> bool {
    let raw = crate::opts::var(crate::opt!("tp-reduce")).ok();
    tp_reduce_resolve(raw.as_deref(), world, dec_xport_mode())
}

/// True only when the operator typed `--tp-reduce single` (the refusal check must not fire on `auto`).
pub fn tp_reduce_explicit_single() -> bool {
    matches!(crate::opts::var(crate::opt!("tp-reduce")).map(|v| v.trim().to_ascii_lowercase()).as_deref(), Ok("single"))
}

/// TP-4F2: the refusals of `--tp-reduce single` (pure function of the resolved flag, so it is unit-testable).
/// Single is a world-4 decode/verify schedule that rides the folded decode transport: world 2 has one stage already
/// (nothing to fuse), and `--tp-dec-xport 0` (the serial pair) has no producer-folded K1 to retarget.
pub(super) fn tp_reduce_refusal(single: bool, world: i32, dec_xport: u8) -> Result<()> {
    if !single { return Ok(()); }
    anyhow::ensure!(world == 4, "--tp-reduce single is a world-4 schedule (got world {world}): world 2 already reduces in ONE stage");
    anyhow::ensure!(dec_xport == 1, "--tp-reduce single rides the folded decode transport: leave --tp-dec-xport at its default (1)");
    Ok(())
}

/// TP-4F2: the single-stage ctx's TCP base at `world` 4, verified disjoint from rail 1 (TP_PORT) and rail 2's
/// N-way base so no handshake can mis-pair; a future change of any rail layout fails loudly here.
/// Port-block stride per N-way link at `world > 2`: multiple of 10 clearing `2 + world^2` pair ports
/// (world 4: 20 -> rail2 +20, single +40, dual-single +60; world 8: 70 -> rail2 +70, single +140, dual-single +210).
fn nway_port_stride(world: i32) -> u32 {
    let w = world as u32;
    (2 + w * w).div_ceil(10) * 10
}

pub fn single_nway_base(world: i32) -> Result<u16> {
    let (b1, b2) = (crate::tp::TP_PORT, rail2_nway_base(world)?);
    let off = if world == 4 { TP_SINGLE_PORT_OFF as u32 } else { 2 * nway_port_stride(world) };
    let b3 = b1 as u32 + off;
    anyhow::ensure!(b3 + 2 + (world * world) as u32 <= u16::MAX as u32, "single-stage base port {b3} at world {world} is beyond 65535");
    let b3 = b3 as u16;
    let p3 = rail_ports(b3, world);
    for (name, other) in [("rail 1", rail_ports(b1, world)), ("rail 2", rail_ports(b2, world))] {
        if let Some(p) = p3.iter().find(|p| other.contains(p)) {
            anyhow::bail!("TP single-stage port collision at world {world}: ctx (base {b3}) and {name} both use TCP port {p}");
        }
    }
    Ok(b3)
}

/// Phase 2 / TP=8: rail-2 single-stage twin base port at `world > 2` (+60 at world 4, +210 at world 8),
/// verified disjoint from rail-1 primary, rail-2 primary, and rail-1 single-stage port blocks.
pub fn dual_single_nway_base(world: i32) -> Result<u16> {
    let b1 = crate::tp::TP_PORT;
    let b2 = rail2_nway_base(world)?;
    let b3 = single_nway_base(world)?;
    let b4 = b1 as u32 + 3 * nway_port_stride(world);
    anyhow::ensure!(b4 + 2 + (world * world) as u32 <= u16::MAX as u32, "dual single-stage base port {b4} at world {world} is beyond 65535");
    let b4 = b4 as u16;
    for other in [b1, b2, b3] {
        check_rail_ports(world, other, b4)?;
    }
    Ok(b4)
}

pub(super) fn rail2_dev() -> String {
    crate::tp::rdma_dev_rail2()   // explicit --rdma-dev's second entry, else the auto-detected rail 2
}

/// TP-4E: every TCP port one rail's bring-up may bind or dial at `world` — the world-2 control port `base`
/// itself (reserved: the N-way path never binds it, the world-2 path does), the liveness responder
/// (`base + 1`) and one handshake port per rank pair (`base + 2 + lo * world + hi`, native/net_shim.c
/// `nway_pair_port`, mirrored by `net::nway_link_ports`).
pub(super) fn rail_ports(base: u16, world: i32) -> Vec<u32> {
    let mut v = vec![base as u32];
    v.extend(crate::net::nway_link_ports(base, world));
    v
}

/// TP-4E: the two rails' port sets must be disjoint and inside the u16 range. A rail-2 pair port equal to a
/// rail-1 pair port would make two different handshakes contend for one listener (a mis-pair, or a bind
/// failure that hangs the peer's 300 s connect loop).
pub fn check_rail_ports(world: i32, base1: u16, base2: u16) -> Result<()> {
    let (a, b) = (rail_ports(base1, world), rail_ports(base2, world));
    if let Some(p) = b.iter().find(|p| a.contains(p)) {
        anyhow::bail!("TP rail port collision at world {world}: rail 1 (base {base1}) and rail 2 (base {base2}) both use TCP port {p}");
    }
    let top = a.iter().chain(b.iter()).copied().max().unwrap_or(0);
    anyhow::ensure!(top <= u16::MAX as u32, "TP rail ports at world {world} (bases {base1}, {base2}) reach {top}, beyond 65535");
    Ok(())
}

/// TP-4E: the rail-2 N-way base port at `world` > 2: the first multiple of 10 at or above
/// `2 + world^2` past the rail-1 base (TP_PORT), so the pair-port block of rail 1 (`+2 ..= +1 + world^2`)
/// is cleared at any world (world 4: TP_PORT + 20, liveness +21, pairs +23..+25, +28, +29, +33). It is
/// `check_rail_ports`-verified, so a future change of either rail's layout fails loudly here.
pub fn rail2_nway_base(world: i32) -> Result<u16> { rail2_nway_base_for(world, crate::tp::TP_PORT) }

/// `rail2_nway_base` for an arbitrary rail-1 base port (the `--tp-reduce-bench --world 4` harness runs on its
/// own port range, default 29700, so it can coexist with a serving head on TP_PORT).
pub fn rail2_nway_base_for(world: i32, base1: u16) -> Result<u16> {
    anyhow::ensure!(world > 2, "rail2_nway_base: world {world} takes the world-2 rail-2 port (TP_PORT + {TP_AUX_PORT_OFF})");
    let w = world as u32;
    let base = base1 as u32 + (2 + w * w).div_ceil(10) * 10;
    anyhow::ensure!(base + 2 + w * w <= u16::MAX as u32, "rail-2 base port {base} at world {world} is beyond 65535");
    let base = base as u16;
    check_rail_ports(world, base1, base)?;
    Ok(base)
}

/// TP-4B: the world > 2 refusals that depend only on flags. The head runs it BEFORE the model load (a
/// refused launch must not cost a 100 s load) and `attach` runs it again on every rank. World 2 passes
/// untouched.
///   * worlds other than 2 and 4;
///   * --ep-deal freq outside world 4 (TP-4F-prep: world 4 has its own committed table, validated here);
///   * --tp-oneshot: the bf16-wire all-peers push (tp_wait_add_4way) — no fp32 intermediate, caused the
///     NVFP4 wedge, measured at parity with the two-shot;
///   * --tp-dec-grecv other than 0 (TP-4C): the decode K2's GPU-side receive (payload-tail proof) is not
///     armed at world > 2 — the N-way proxy's CPU-bounce recv loop has no `rx_done` skip, so a K2 that
///     passes on the hint + tail while the proxy lags can see a newer tail and abort (code 2). The decode
///     path at world > 2 receives on cpu_done only (forced at attach; TP-4F lever: add the skip, then arm);
///   * --tp-vp-sampled on at a world other than 2 or 4 (TP-4C / TP-4H2): the TP-H2 row all-gather has kernels
///     at world 2 (peer = rank ^ 1) and world 4 (TP-4H2, recursive doubling). World 4 = ON by default (an
///     explicit `on` is accepted); any other world > 2 = OFF by default (the replicated head, resolved by the
///     head, `resolve_vp_sampled`) and an explicit `--tp-vp-sampled on` bails.
///
/// TP-4E: the dual-rail prefill transport (--tp-prefill-xport unset/dual/2) is NO LONGER refused at world
/// > 2 — its second rail is the N-way `connect_nway` link `attach` builds (`rail2_nway_base`). All three
/// transports (0 serial, 1 single-rail pipelined, 2 dual-rail pipelined) are rounds-aware and admitted;
/// their W=4 hardware proof is the TP-4E gate list (PLAN/TP-4E_GATES.md).
pub fn world_gt2_flag_refusals(world: i32) -> Result<()> {
    tp_reduce_refusal(tp_reduce_explicit_single(), world, dec_xport_mode())?;   // TP-4F2: refusal-only; `auto` / `rd` never refuse
    // PACK4-REVIEW B1: the --lane-pack world rule, before any network: the head runs this ahead of
    // discovery and the config ship, a node ahead of its link bring-up. (Runs at TP=1, TP=2 and TP=4.)
    lane_pack_world_refusal(world, crate::opts::var(crate::opt!("lane-pack")).ok().as_deref())?;
    if world <= 2 { return Ok(()); }
    anyhow::ensure!(world == 4, "EXL3 TP: --tp 2 or --tp 4 only (got {world})");
    anyhow::ensure!(!crate::opts::var(crate::opt!("tp-oneshot")).map_or(false, |v| v != "0"),
        "EXL3 TP world {world}: --tp-oneshot is refused on the EXL3 path (bf16 wire, no fp32 intermediate between \
         rounds; it wedged NVFP4 serving and measured at parity with the two-shot)");
    anyhow::ensure!(crate::opts::var(crate::opt!("tp-dec-grecv")).map_or(true, |v| v.trim() == "0"),
        "EXL3 TP world {world}: --tp-dec-grecv is world 2 only (the N-way proxy's recv loop has no rx_done skip, so the \
         GPU-side payload-tail receive can race it — TP-4F); leave the flag unset or pass --tp-dec-grecv 0");
    ep_deal4::flag_refusal(&ep_deal_kind(), world)?; // --ep-deal freq: world 4 only (table validated here, pre-load)
    Ok(())
}

/// PACK4-REVIEW B1: the `--lane-pack` rule of `world_gt2_flag_refusals`, pure (value = the flag's raw
/// text, None = unset): a world above `lane_pack::LANE_PACK_MAX_WORLD` (4) refuses `on` / `2`.
pub(crate) fn lane_pack_world_refusal(world: i32, value: Option<&str>) -> Result<()> {
    anyhow::ensure!(world <= crate::lane_pack::LANE_PACK_MAX_WORLD as i32 || value.map_or(true, |v| matches!(v.trim(), "" | "off")),
        "EXL3 TP world {world}: --lane-pack runs at TP=1, TP=2 and TP=4 (no packed lockstep frame exists above four boxes) \
         — start the head with --lane-pack off");
    Ok(())
}

/// TP-4B: the world > 2 refusals that depend on the probe's options. `--xtp-sp-fault` arms a fault in the
/// sequence-parallel run; where no SP run exists (`sp_ready`) the gate would pass as a silent
/// negative control, so it is refused before the load. World 2 passes untouched.
pub fn world_gt2_opts_refusals(world: i32, opts: &XtpOpts) -> Result<()> {
    opts_refusals_with_sp(world, opts, seq_parallel_on())
}

/// `world_gt2_opts_refusals` with the resolved sequence-parallel state passed in (the unit tests do not touch the global).
pub(super) fn opts_refusals_with_sp(world: i32, opts: &XtpOpts, sp_on: bool) -> Result<()> {
    if world <= 2 { return Ok(()); }
    // TP-4S: world 4 runs the sequence-parallel prefill, so the fault (the rank-identity gate's negative control)
    // is meaningful there when SP is on; above world 4 (or with SP off) the gate would pass as a silent negative control
    anyhow::ensure!(opts.sp_fault.is_none() || (world == 4 && sp_on),
        "EXL3 TP world {world}: --xtp-sp-fault needs a sequence-parallel schedule (world 2, or world 4 with --tp-seq-parallel \
         on|hc|router); here the run is replicated and the gate would pass as a silent negative control");
    Ok(())
}

/// TP-4E: `--tp-prefill-overlap fold` needs whole-row epochs and ONE exchange per chunk (the fused consumer
/// replaces K2m, which at world > 2 is a per-round kernel over a staged fp32 sum), so it is a world-2
/// schedule. `inline` / `dual` / `on` are program-driven and world-general (`tp_xport::build_program`). The
/// head runs this right after it resolves the flag (before any load); `attach` runs it again on every rank.
/// World 2 passes untouched.
pub fn world_gt2_overlap_refusal(world: i32, code: u32) -> Result<()> {
    if world <= 2 || code & PF_FOLD == 0 { return Ok(()); }
    anyhow::bail!("EXL3 TP world {world}: --tp-prefill-overlap fold is a world-2 schedule (whole-row epochs, K2m replaced by the fused \
                   consumer; the world > 2 reduce is {} rounds of K2m over a staged fp32 sum) — use on|dual|inline|off",
                  world.ilog2())
}

/// TP-4E / TP-4S: an EXPLICIT `--tp-seq-parallel <on|hc|router>` is honoured at world 2 and (TP-4S) world 4, where
/// it runs the quarter-row sequence-parallel prefill (recursive-halving reduce-scatter + recursive-doubling
/// all-gather over the world-4 doorbell rounds, `tp_xport::*_w4`). Above world 4 there is no SP schedule, so an
/// explicit flag would be a silent no-op and is refused; `off` is always fine. (The DEFAULT, flag absent, stays ON
/// in the config and yields at runtime (`sp_ready`) wherever no SP schedule exists.)
pub fn world_gt2_sp_refusal(world: i32, explicit: bool, code: u32) -> Result<()> {
    if world <= 4 || !explicit || code == 0 { return Ok(()); }
    anyhow::bail!("EXL3 TP world {world}: --tp-seq-parallel {} has no schedule at world {world} (sequence-parallel prefill exists at \
                   world 2 (two-rank row exchange) and world 4 (TP-4S quarter rows); it would silently not run here) — drop the \
                   flag (the replicated prefill runs) or pass off",
                  seq_parallel_desc(code))
}

/// TP-4E: the boot-log reason for the resolved `--tp-prefill-xport`, a pure function so the wording is
/// tested. The DEFAULT is dual-rail at EVERY world (flag default `dual`); `single` / `off` are explicit loud
/// opt-outs, never a fallback. Numbers are labelled MEASURED (TP-F, world 2 bench) or PROJECTION (TP-4A, not
/// yet measured at world 4 — the TP-4E hardware gates decide).
pub fn prefill_xport_why(world: i32, mode: u8, explicit: bool) -> String {
    let src = if explicit { "explicit" } else { "default" };
    match (mode, world > 2) {
        (2, false) => format!("{src} dual-rail (TP-F, measured at world 2: 993 us vs 1,700 us single-rail per 20 MiB reduce); both rails must be healthy, a missing second rail fails the boot"),
        (2, true) => format!("{src} dual-rail at world {world}: each chunk reduce runs {} rounds, the later rounds ship fp32, so one rail (~13.6 GB/s per direction, measured) is wire-bound; \
                              TP-4A projects ~831 ms/chunk dual-rail vs ~1,064 ms single-rail (PROJECTIONS, to be measured by the TP-4E gates); \
                              rail 2 is a full N-way link, both must be healthy (preflight agrees it before any dial), no silent fallback", world.ilog2()),
        (1, false) => format!("{src} single-rail pipelined (opt-out of the dual-rail default: expect the TP-F single-rail cost, 1,700 us vs 993 us per 20 MiB reduce, measured)"),
        (1, true) => format!("{src} single-rail pipelined at world {world} — a PERF OPT-OUT of the dual-rail default (TP-4A projects ~1,064 ms/chunk vs ~831 ms dual-rail, a regression vs the TP=2 SP-on baseline; projection)"),
        (_, false) => format!("{src} serial single-block K1/K2 (the TP-B..E path; diagnostic escape, no pipelining, no overlap hooks)"),
        (_, true) => format!("{src} serial single-block K1/K2 at world {world} — a diagnostic escape, no pipelining, no overlap hooks (PERF OPT-OUT)"),
    }
}

/// TP-I (C6): the post-load boot rendezvous of the dual-rail attach. Probe-tolerant (net_exchange):
/// waits as long as the peer is alive (model-load skew of minutes is legitimate), aborts in bounded
/// time when the peer process is gone (liveness probe every 5 s of silence; a killed peer is
/// noticed <= ~5.5 s after the wait starts or after it dies, an unreachable host <= ~10 s).
/// Frame: [magic, mode]; a magic mismatch is an SPMD desync, a mode mismatch a config desync.
///
/// Test-only failure injection: --tp-boot-delay=<rank>:<seconds> (rides the exl3_env snapshot to
/// the node) sleeps that rank for <seconds> BEFORE the rendezvous — a controlled load skew.
pub(super) fn boot_rendezvous(rank: i32, mode: u8) -> Result<()> {
    if let Ok(v) = crate::opts::var(crate::opt!("tp-boot-delay")) {
        if let Some((r, s)) = v.split_once(':') {
            if r.trim().parse::<i32>().ok() == Some(rank) {
                let secs: u64 = s.trim().parse().unwrap_or(0);
                crate::dprintln!("[exl3-tp] rank {rank}: --tp-boot-delay test injection — sleeping {secs}s before the boot rendezvous");
                std::thread::sleep(std::time::Duration::from_secs(secs));
            }
        }
    }
    const MAGIC: u32 = 0xB00F_0C06;
    let t0 = std::time::Instant::now();
    crate::dprintln!("[exl3-tp] rank {rank}: entering the post-load boot rendezvous (probe-tolerant, no overall deadline)");
    let all = crate::net::exchange_u32s_all(&[MAGIC, mode as u32], 4)
        .context("TP-F dual-rail boot rendezvous (probe-tolerant exchange; peer dead or link aborted)")?;
    for (r, peer) in all.iter().enumerate().filter(|(r, _)| *r as i32 != rank) {
        if peer[0] != MAGIC || peer[1] != mode as u32 {
            anyhow::bail!("TP-F dual-rail rendezvous: rank {r} frame {:08x}/{} vs mine {MAGIC:08x}/{mode} (SPMD or mode desync)",
                          peer[0], peer[1]);
        }
    }
    crate::dprintln!("[exl3-tp] rank {rank}: boot rendezvous OK after {:.1}s wait (probe-tolerant; mode {mode})",
             t0.elapsed().as_secs_f64());
    Ok(())
}

/// Pad exchanges that bring a device epoch counter to a multiple of `rounds` (0 when already aligned).
pub(super) fn round_pad(epoch: u64, rounds: usize) -> usize { let r = rounds as u64; ((r - epoch % r) % r) as usize }

/// R10 (world > 2 only; the same rule gpu.rs `tp_align_rounds` applies to the batch engine): every logical
/// all-reduce consumes a whole multiple of `rounds` epochs and epoch e runs round e % rounds, so the
/// association the reduce realises is frozen by the counter's parity at the first reduce. One fixed
/// parity on every rank and every launch keeps the result deterministic and rank-identical. A fresh ctx
/// starts at epoch 0 (the boot rendezvous rides the control slots, not the GPU epoch) so the pad is 0 in
/// practice; if anything ever consumes an odd number of epochs first, pad it out here and say so. The
/// first reduce then runs round (e + 1) % rounds first: world 4 => partner ^ 2 first, (p0+p2)+(p1+p3).
/// World 2 never calls this (rounds == 1, no association to fix).
///
/// TP-4E: one call PER RAIL (`rail` labels the log line; `epoch` reads THAT rail's device counter: rail 1 is
/// the registered agree link `traced_device_epoch`, rail 2 its own ctx `ctx_device_epoch`). The pipelined
/// schedule (`rail_seq`) indexes each rail's epochs from its own counter, so BOTH rails must start on the
/// same phase (a multiple of `rounds`); a pad exchange needs that rail's peers' proxies running, so rail 2
/// is aligned only after its proxy is spawned. Every rank pads identically because every rank's fresh
/// counter is 0 (a nonzero pad on one rank only would desync the partner map — the agree hash catches it).
pub(super) fn align_round_phase(dev: &Arc<CudaDevice>, kern: &crate::tp_xport::Kernels, stream: &CudaStream, ctx: u64,
                     part: &mut CudaSlice<f32>, world: i32, rank: i32, rail: &str,
                     epoch: &dyn Fn() -> u64) -> Result<()> {
    let rounds = crate::tp_xport::rounds_of(world as usize)?;
    dev.synchronize()?;
    let e0 = epoch();
    let pad = round_pad(e0, rounds);
    if pad > 0 {
        dev.memset_zeros(part)?;   // AGENTS §2.2: alloc_zeros does not zero
        dev.synchronize()?;
        let p = *part.device_ptr() as u64;
        for _ in 0..pad {
            crate::tp_xport::k1_serial(kern, stream, ctx, p, 4)?;
            crate::tp_xport::k2_serial(kern, stream, ctx, p, 4)?;
        }
        dev.synchronize()?;
    }
    let e1 = epoch();
    anyhow::ensure!(e1 % rounds as u64 == 0, "TP round phase ({rail}): device epoch {e1} is not a multiple of {rounds} after alignment");
    crate::dprintln!("[exl3-tp] rank {rank}: round phase aligned ({rail}: device epoch {e0} -> {e1}, {pad} pad exchange(s)): the first all-reduce \
              runs round {} first", (e1 + 1) % rounds as u64);
    Ok(())
}

/// TP-4E: the rail-2 device must be present, its port ACTIVE and its RoCE v2 GID usable on EVERY rank before any
/// rank dials rail 2. `connect_nway` blocks in `accept` (no timeout) and a 300 s connect loop: one rank whose
/// second device is missing, down or without the GID the engine binds (`GID_IDX`) would bail alone while its
/// peers hang in the handshake. The check is agreed over rail 1's
/// control plane (`exchange_u32s_all`, which touches neither the GPU epoch nor the rail-2 ports), so a
/// failure anywhere fails EVERY rank, loudly, in seconds. Runs only at world > 2
/// (world 2 keeps its TP-F bring-up unchanged).
pub(super) fn rail2_preflight(rank: i32, dev2: &str) -> Result<()> {
    // S-B9-REL-PRPORT3 (review R6): resolve ONCE and print the RESOLVED index in the refusal —
    // the static GID_IDX default can differ from what the sysfs scan picked for this boot.
    let gid_idx = crate::tp::resolve_gid_idx(dev2);
    let (ok, why) = rail_device_state(std::path::Path::new("/sys/class/infiniband"), dev2, gid_idx);
    if !ok { crate::dprintln!("[exl3-tp] rank {rank}: rail 2 preflight FAILED locally: {why}"); }
    let all = crate::net::exchange_u32s_all(&[preflight_word(ok), rank as u32], 4)
        .context("TP-4E rail 2 preflight exchange (rail 1 control plane)")?;
    let bad = preflight_failures(&all.iter().map(|f| f[0]).collect::<Vec<_>>());
    anyhow::ensure!(bad.is_empty(),
        "TP-4E dual-rail bring-up refused before any rail-2 dial: {} (this rank: {why}). Every rank needs the second RDMA device \
         (--rdma-dev <first>,<second>; default: the auto-detected active port pair) present with port 1 state 4 (ACTIVE) and a non-zero \
         RoCE v2 GID at index {}; run with --tp-prefill-xport single only as an explicit, owner-consented opt-out",
        bad.join("; "), gid_idx);
    Ok(())
}

/// The same local check as a `Result`, for a caller that dials rail 2 without the engine's rail-1 agreement
/// (the `--tp-reduce-bench --world > 2` harness): this box fails by name before it dials, instead of leaving
/// its peers blocked in `accept`.
pub(crate) fn rail_local_preflight(dev: &str, gid_idx: i32) -> Result<()> {
    let (ok, why) = rail_device_state(std::path::Path::new("/sys/class/infiniband"), dev, gid_idx);
    anyhow::ensure!(ok, "rail-2 device preflight failed on this box before any rail-2 dial: {why}");
    Ok(())
}

/// The preflight frame word: a magic plus the verdict (1 = this rank's rail-2 device is usable). The frame is
/// two words because `exchange_u32s_all` needs 8 B of tail headroom in its 16 B frame.
const PREFLIGHT_MAGIC: u32 = 0x5232_0F10;
pub(super) fn preflight_word(ok: bool) -> u32 { PREFLIGHT_MAGIC + ok as u32 }
/// Per-rank verdict words -> the list of failing ranks (empty = every rank usable). A word that is neither
/// verdict is an SPMD desync (different binaries), reported as such.
pub(super) fn preflight_failures(words: &[u32]) -> Vec<String> {
    words.iter().enumerate().filter_map(|(r, &w)| match w {
        w if w == preflight_word(true) => None,
        w if w == preflight_word(false) => Some(format!("rank {r}: rail-2 device not usable")),
        w => Some(format!("rank {r}: frame {w:08x} (SPMD desync)")),
    }).collect()
}

/// The text of the three sysfs files a rail device must pass (`None` = unreadable).
#[derive(Clone, Copy)]
pub(super) struct RailFiles<'a> { pub(super) state: Option<&'a str>, pub(super) gid: Option<&'a str>, pub(super) gid_type: Option<&'a str> }

/// Pure decision over the file contents (so it is testable with fake text). Fails closed: an unreadable file is a
/// failure. Every message names the device, the file and the expected value.
///   * `<dev>/ports/1/state` must be state number 4 ("4: ACTIVE"); "5: ACTIVE_DEFER" and anything else fail;
///   * `<dev>/ports/1/gids/<gid_idx>` must be a readable, non-zero GID (the index `connect_nway` binds);
///   * `<dev>/ports/1/gid_attrs/types/<gid_idx>` must read "RoCE v2" (the engine's QPs use RoCE v2 addressing).
pub(super) fn rail_files_verdict(dev: &str, dir: &std::path::Path, gid_idx: i32, f: &RailFiles) -> (bool, String) {
    let (p_state, p_gid, p_type) = (dir.join("ports/1/state"), dir.join(format!("ports/1/gids/{gid_idx}")),
                                    dir.join(format!("ports/1/gid_attrs/types/{gid_idx}")));
    let st = match f.state {
        None => return (false, format!("{dev}: cannot read {} (expected \"4: ACTIVE\"; an unreadable state is refused, not assumed up)", p_state.display())),
        Some(s) => s.trim(),
    };
    if st.split(':').next().map(str::trim) != Some("4") {
        return (false, format!("{dev}: {} is '{st}' (expected \"4: ACTIVE\", state number 4)", p_state.display()));
    }
    let gid = match f.gid {
        None => return (false, format!("{dev}: cannot read {} (expected a non-zero GID)", p_gid.display())),
        Some(g) => g.trim(),
    };
    if !gid.chars().all(|c| c == ':' || c.is_ascii_hexdigit()) || !gid.chars().any(|c| c.is_ascii_hexdigit() && c != '0') {
        return (false, format!("{dev}: {} is '{gid}' (expected a non-zero GID, the entry the engine binds: GID index {gid_idx})", p_gid.display()));
    }
    let ty = match f.gid_type {
        None => return (false, format!("{dev}: cannot read {} (expected \"RoCE v2\")", p_type.display())),
        Some(t) => t.trim(),
    };
    if ty != "RoCE v2" {
        return (false, format!("{dev}: {} is '{ty}' (expected \"RoCE v2\")", p_type.display()));
    }
    (true, format!("{dev} port 1 ACTIVE, GID {gid_idx} {gid} (RoCE v2)"))
}

/// Is `dev` (a /sys/class/infiniband entry under `root`) present with port 1 ACTIVE and a usable RoCE v2 GID at
/// `gid_idx`? Returns (ok, why). Reads the three sysfs files and defers the decision to `rail_files_verdict`.
pub(super) fn rail_device_state(root: &std::path::Path, dev: &str, gid_idx: i32) -> (bool, String) {
    let d = root.join(dev);
    if !d.exists() { return (false, format!("{dev}: {} does not exist (expected the RDMA device directory)", d.display())); }
    let rd = |rel: String| std::fs::read_to_string(d.join(rel)).ok();
    let (state, gid, gid_type) = (rd("ports/1/state".into()), rd(format!("ports/1/gids/{gid_idx}")),
                                  rd(format!("ports/1/gid_attrs/types/{gid_idx}")));
    rail_files_verdict(dev, &d, gid_idx, &RailFiles { state: state.as_deref(), gid: gid.as_deref(), gid_type: gid_type.as_deref() })
}
