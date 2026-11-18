//! Shared constants and flags: route-log and MTP / copy width limits, the KV format codes and helpers,
//! `MtpOff` / `mtp_off_of`, `AFlags`, `a_flags` and the tap cross-check counters.
//! See `mod.rs` for the file map.

use super::*;

/// WP23: draft depth ceiling 5 -> 7 (verify width m = k+1 <= 8). Every per-round buffer derives
/// from it (d_dev / accept_out / dconf, the per-GDN-layer qkv_save / qkv_post / a_save / b_save /
/// gdn_ring planes), and so does the ring-commit gate (MTP_MAX_K+1 <= GDN_RING_CMAX = 8, the
/// xq_gdn_commit_ring smem extent). The SERVED depth is the runtime cap --exl3-mtp-k
/// (mtp_depth_default, default = this); --exl3-mtp-k=5 reproduces the K5 build's rounds.
/// CF-P1e step 0 route log: record width (= XQ_ROUTE_REC in exl3_bench.cu) and ring capacity (~270 MB).
pub const ROUTE_REC: usize = 2 + 16 * 16;
pub const ROUTE_LOG_CAP: usize = 1 << 18;
pub const MTP_MAX_K: usize = 10; // MTP26/DEPTH: raised 7 -> 10 (owner ask 2026-10-31); ring extents follow
/// PACK2: plane stride (rows) of the qkv_save/qkv_post/a_save/b_save
/// per-layer save planes. 16 >= the packed R_pad (10): the packed save must
/// never cross a plane boundary (8 = MTP_MAX_K+1 overflowed by 2 rows/layer).
pub const SAVE_PLANE_ROWS: usize = 16;
/// Words of `Scratch::accept_out` the SERVED round reads back (the lone `xq_accept` writes
/// `out[0..=k+2]`, so `MTP_MAX_K + 3`). The REPRIME pinned buffer (`RpRt::new`) AND the async copy in
/// `reprime_dev_enqueue` are both sized by THIS, never by `accept_out.len()`: K4 (REL v0.7.3) widened
/// `accept_out` to 16 words for the packed probes and the 16-word copy overran the 10-word pin, so
/// every speculative round failed `REPRIME: 16 accept words > pinned 40 B` (found by GATES2).
pub const RP_ACCEPT_WORDS: usize = MTP_MAX_K + 3;
/// Allocated width of `Scratch::d_dev` / `accept_out` (K4: the packed tail runs to R_pad = 16 rows).
pub(crate) fn accept_out_words() -> usize { accept_out_words_for(MTP_MAX_K + 1) }
/// WIDEVERIFY1 refinement 2: `accept_out_words` for a scratch whose lone verify holds `vrows` rows
/// (`Scratch::vrows`): xq_accept writes `out[0..=k+2]` with k = vrows - 1, the packed tail 16 rows.
pub(crate) fn accept_out_words_for(vrows: usize) -> usize { (vrows + 2).max(16) }
/// WIDEVERIFY1 refinement 2: the REPRIME readback (and its pinned buffer) for a scratch of `vrows`
/// verify rows: `out[0..=k+2]`, k = vrows - 1. `rp_accept_words(MTP_MAX_K + 1) == RP_ACCEPT_WORDS`.
pub(crate) fn rp_accept_words(vrows: usize) -> usize { vrows + 2 }
/// WIDEVERIFY1 refinement 2: `FwdModel::rp_rows_for` over the scratch's (width `w`, verify rows
/// `vrows`) — pure, unit-tested: k <= MTP_MAX_K -> the pre-refinement `rp_rows` (MTP_MAX_K bounded
/// by w - 1); a wide copy round -> COPY_MAX_K bounded by vrows - 1 and w - 1.
pub(crate) fn rp_rows_of(w: usize, vrows: usize, k: usize) -> usize {
    if k <= MTP_MAX_K { MTP_MAX_K.min(w.saturating_sub(1)) } else { COPY_MAX_K.min(vrows.saturating_sub(1)).min(w.saturating_sub(1)) }
}
/// WIDEVERIFY1 refinement 2: the most drafts a COPY round (copy-span: host-proposed ids, no draft
/// pass) may verify on a scratch built with `COPY_VROWS` verify rows (`Scratch::vrows`). The neural
/// MTP / DFlash2 rounds keep `MTP_MAX_K`; a scratch built without the wide rows (the default) keeps
/// every per-round plane, pin and readback exactly at `MTP_MAX_K + 1` rows.
pub const COPY_MAX_K: usize = 15;
/// Verify rows of a wide-copy scratch (`COPY_MAX_K + 1`): the ring / taps planes, the accept words.
pub const COPY_VROWS: usize = COPY_MAX_K + 1;
/// WP23: max rows xq_gdn_commit_ring commits (its kr_sh/g_sh extent + device guard; exl3_bench.cu).
/// WIDEVERIFY1 refinement 2: 11 -> 16 (COPY_VROWS). The extent only bounds the smem arrays; C (the
/// committed row count, a+1) bounds every loop, so a C <= 11 commit runs the same operations.
pub(super) const GDN_RING_CMAX: usize = 16; // C = a+1 <= 16 (kr_sh/g_sh extent, mirrors XQ_GDN_RING_CMAX)
const _: () = assert!(MTP_MAX_K + 1 <= GDN_RING_CMAX && COPY_VROWS <= GDN_RING_CMAX && COPY_VROWS <= SAVE_PLANE_ROWS,
                      "the wide copy verify must fit the ring-commit extent and the 16-row save planes");
/// K1: KV cache storage format. [kv-cache] = f32 (exact) | f16 | fp8 | q8 (the server's
/// --kv-cache). Row bytes per (slot, K|V, kv-head, pos): 4*hd, 2*hd, hd+16 (fp8 + f32 row scale),
/// hd + hd/16 padded to 16 B (q8: int8 codes + one f16 scale per 32-dim group; 272 B at hd=256).
pub const KV_F32: u32 = 0;
pub const KV_F16: u32 = 1;
pub const KV_FP8: u32 = 2;
/// WP25/K3: the reference implementation's "-cq 8" scheme (H32 rotation + int8 midpoint grid + f16 group scale).
pub const KV_Q8: u32 = 3;
/// WP25 probe-only positive control: the q8 layout at 4-bit resolution ([kv-cache]=q4x).
pub const KV_Q4X: u32 = 4;
/// 4-bit packed K/V (docs/KV_CACHE_Q4K.md): the q8 H32 basis with one f16 scale per
/// 32-dim group and a 4-bit midpoint grid, two dims per byte. 144 B per row at hd=256.
pub const KV_Q4K: u32 = 5;
/// The format when neither --kv-cache nor [kv-cache] names one. Owner decision 2026-09-26:
/// q8 becomes the default AFTER the WP25 GPU quality gate passes — it passed on .14 (quality
/// holds vs the noise floor), so the default is now the reference implementation's q8 cache. `--kv-cache f32`
/// (or [kv-cache]=f32) restores the exact f32 cache.
pub const KV_FMT_DEFAULT: u32 = KV_Q8;
pub fn kv_fmt_from_opts() -> u32 {
    match crate::opts::var(crate::opt!("kv-cache")).as_deref() {
        Ok("f32") => KV_F32,
        Ok("f16") => KV_F16,
        Ok("fp8") => KV_FP8,
        Ok("q8") => KV_Q8,
        Ok("q4x") => KV_Q4X,
        Ok("q4k") => KV_Q4K,
        Ok("") | Err(_) => KV_FMT_DEFAULT,
        Ok(other) => {
            crate::deprintln!("[exl3] [kv-cache]={other} is not f32|f16|fp8|q8|q4k; using the default {}",
                      kv_fmt_name(KV_FMT_DEFAULT));
            KV_FMT_DEFAULT
        }
    }
}
pub fn kv_fmt_name(fmt: u32) -> &'static str {
    match fmt {
        KV_F32 => "f32",
        KV_F16 => "f16",
        KV_FP8 => "fp8",
        KV_Q8 => "q8",
        KV_Q4X => "q4x",
        KV_Q4K => "q4k",
        _ => "q4x",
    }
}
/// True for the packed 4-bit format: its row is 144 B at hd=256 (not q8's 272) — 2 dims per byte,
/// with the same per-32-group f16 scales, so its readers take a row's scale offset (hd/2) and a
/// packed slice (4 code bytes for 8 dims) instead of q8's byte-per-dim.
pub fn kv_is_packed4(fmt: u32) -> bool {
    fmt == KV_Q4K
}
pub fn kv_rowbytes(fmt: u32, hd: usize) -> usize {
    match fmt {
        KV_F32 => 4 * hd,
        KV_F16 => 2 * hd,
        KV_FP8 => hd + 16,
        KV_Q4K => (hd / 2 + hd / 16).next_multiple_of(16),
        _ => hd + (hd / 16).next_multiple_of(16),
    }
}
/// Served draft depth. 5 = the reference implementation's -ndt 5: with the dynamic draft stop (exl3_serve DDS) the
/// depth is a CEILING, so d5 no longer pays full-width rounds on prose (API A/B 2026-09-25:
/// k5+DDS ansic 87.9 / python 77.6 / prose 55.8 vs k3 fixed 76.3 / 69.5 / 52.4).
/// WP23: default = MTP_MAX_K = 7 (WP00 B2: AnsiC R6+R7 = 1.418), with the DDS depth-keyed bins +
/// cost guard (exl3_serve) stopping levels 6/7 where they do not pay. --exl3-mtp-k=5 = the K5
/// build's served depth AND its DraftCal semantics (the WP23 confound check).
pub fn mtp_depth_default() -> usize {
    crate::opts::var(crate::opt!("exl3-mtp-k")).ok().and_then(|v| v.parse().ok())
        .unwrap_or(MTP_MAX_K).clamp(1, MTP_MAX_K)
}

impl FwdModel {
    /// WP23: number of captured graphs — the serve's DDS cost fit skips a round that grew it
    /// (a lazy first-use capture is a one-off ms outlier, not the marginal cost of a draft).
    pub fn graph_count(&self) -> usize {
        self.graphs.lock().map(|g| g.len()).unwrap_or(0)
    }
}

/// Which switch turned the EXL3 draft head off.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MtpOff {
    /// `--mtp off` (the documented user switch; also its older spellings `0` / `false`)
    MtpFlag,
    /// `--exl3-no-mtp` (the diagnostics hatch, AGENTS 1b)
    Hatch,
}

impl MtpOff {
    /// The flag as the user typed it (banners, error messages).
    pub fn flag(self) -> &'static str {
        match self { MtpOff::MtpFlag => "--mtp off", MtpOff::Hatch => "--exl3-no-mtp" }
    }
}

/// S-B9-REL-SMALLFIXES item 1: what the EXL3 engine makes of `--mtp <auto|on|off>` and
/// `--exl3-no-mtp`. `mtp` = the raw registry value (None = not passed), `hatch` = --exl3-no-mtp set.
/// None = speculation ON (`auto`, `on` and "not passed" are the same thing here: the head runs with
/// the per-request profitability control). v0.7.3 and earlier did not read `--mtp` in this engine at all.
pub fn mtp_off_of(mtp: Option<&str>, hatch: bool) -> Option<MtpOff> {
    if hatch { return Some(MtpOff::Hatch); }
    match mtp {
        Some("off" | "0" | "false") => Some(MtpOff::MtpFlag),
        _ => None,
    }
}

/// The process's resolved switch (both options are Spmd: a TP node sees the head's values).
pub fn mtp_off_by_opt() -> Option<MtpOff> {
    mtp_off_of(crate::opts::var(crate::opt!("mtp")).ok().as_deref(),
               crate::opts::var(crate::opt!("exl3-no-mtp")).is_ok())
}

/// THE gate every EXL3 site reads (serve boot + every step, the TP boot agree, the graph
/// precapture, the WP24 gate): the draft head is off by `--mtp off` or by the hatch.
pub fn mtp_disabled_by_opt() -> bool {
    mtp_off_by_opt().is_some()
}

/// WP02/WP03 (PLAN/SURPASS_PLAN_2026-09-26) diagnostic escapes — A/B and bisect only, never a
/// default (AGENTS §1b). Every switch restores the pre-WP02/03 serve behaviour of one item:
/// - `--a-off=1`: all of them (the rows below), i.e. the old draft-side trajectory;
/// - `--a-taps=shared`: one taps_keep window for every slot (lanes read each other's taps);
/// - `--a-seam=old`: the old seam prime exactly (1-float tap snapshot + full `head_prime` at
///   plen-1 from the tap AFTER plen-1); `=headprime`: full-row snapshot + the old head_prime;
/// - `--a-ctrl=old`: control rounds write no head row p, leave the tap in sc.resid, 1/128;
///   `=fixed128`: the new head-row write + tap snapshot, but the old fixed 1/128 cadence;
///   `=off` (S-B9-CF-PACK4S, gate diagnostics): NO cadence control round. The cadence counter is
///   scheduler-wide (one tick per lane round of ANY lane), so with it a request's round
///   structure — and with it the sampler counters a SAMPLED request draws — depends on what
///   else the server ran; with `off` (and the plain reference therefore never measured, so the
///   per-request cost control never turns speculation off) a request's rounds are a function of
///   its own content and seeds. Not a serving posture: the profitability control is blind;
/// - `--a-remclamp=0`: no remaining-max_new clamp of the draft window.
/// The context-end clamp (p+k+1 < max_pos) and the cancel sweep have NO escape: they only act
/// where the old path wrote past the KV/QSA/cos buffers or decoded for a client that is gone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AFlags {
    pub taps_shared: bool,
    /// 0 = new (KV-only prime from the prefill carry), 1 = headprime, 2 = old
    pub seam: u8,
    /// 0 = new (head row p + snapshot, cost-gated 1/1024 cadence), 1 = fixed128, 2 = old,
    /// 3 = off (S-B9-CF-PACK4S, gate diagnostics: no cadence control round ever runs)
    pub ctrl: u8,
    pub rem_clamp: bool,
}

pub fn a_flags() -> AFlags {
    static F: std::sync::OnceLock<AFlags> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        let v = |k: crate::opts::OptId| crate::opts::var(k).ok();
        let all_off = v(crate::opt!("a-off")).as_deref() == Some("1");
        let f = AFlags {
            taps_shared: all_off || v(crate::opt!("a-taps")).as_deref() == Some("shared"),
            seam: if all_off { 2 } else { match v(crate::opt!("a-seam")).as_deref() { Some("old") => 2, Some("headprime") => 1, _ => 0 } },
            ctrl: if all_off { 2 } else { match v(crate::opt!("a-ctrl")).as_deref() { Some("old") => 2, Some("fixed128") => 1, Some("off") => 3, _ => 0 } },
            rem_clamp: !(all_off || v(crate::opt!("a-remclamp")).as_deref() == Some("0")),
        };
        if f != (AFlags { taps_shared: false, seam: 0, ctrl: 0, rem_clamp: true }) {
            crate::deprintln!("[wp-a] DIAGNOSTIC ESCAPE active: {f:?} (GB10_A_* — not a serving default)");
        }
        f
    })
}

/// WP03 XCHECK (--a-xcheck=1; diagnostics, syncs every round): the serve records the tap each
/// lane's NEXT round will restore, and bit-compares it at that round's start — any write by
/// another lane's verify / prime / control step in between is printed. With --a-taps=shared
/// and two concurrent streams this reproduces the pre-WP03 cross-lane clobber (the bug receipt).
pub fn a_xcheck_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::opts::var(crate::opt!("a-xcheck")).as_deref() == Ok("1"))
}
pub(super) static TAP_XCHECK: std::sync::Mutex<BTreeMap<usize, (usize, Vec<u32>)>> = std::sync::Mutex::new(BTreeMap::new());
pub(super) static TAP_XCHECK_N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub(super) static TAP_XCHECK_BAD: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
