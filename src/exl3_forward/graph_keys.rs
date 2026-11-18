//! Graph-cache key functions (step / verify / draft / prime / pack keys) and the precapture statics. Every
//! key family lives in its own bit range so keys never collide. See `mod.rs` for the file map.

use super::*;

/// WP04 scorer grid: CTAs per SM of `qsa_score_pooled_mr` (--exl3-score-mr-ctas, default 4 —
/// v2: the kernel's __launch_bounds__(256, 4) caps it at 64 regs, so 4 CTAs = 32 warps/SM stay
/// resident in ONE wave; v1 ran 2) times the SM count — a launch constant, so one graph serves
/// every context.
pub(super) fn qsa_mr_ctas() -> u32 {
    static N: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    let sms = *N.get_or_init(|| {
        use cudarc::driver::sys;
        let mut dev_h: sys::CUdevice = 0;
        let mut sms: std::ffi::c_int = 0;
        let ok = unsafe { sys::cuDeviceGet(&mut dev_h, 0) } == sys::CUresult::CUDA_SUCCESS
            && unsafe { sys::cuDeviceGetAttribute(&mut sms,
                   sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev_h) }
               == sys::CUresult::CUDA_SUCCESS;
        if ok && sms > 0 { sms as u32 } else { 48 }
    });
    // TUNE: CTAs per SM from the registry (env alias parsed exactly as before), times the SM count
    (tune::get_cur(&T_SCORE_MR_CTAS).max(1) as u32) * sms
}
/// Rows per CTA group of `qsa_score_pooled_mr` (mirrors QSA_MR_ROWS in gpu_batch.cu).
pub(super) const QSA_MR_ROWS: usize = 8;
/// WP04: true while the boot precapture runs. It silences the per-graph "EXL3_GRAPH: captured"
/// lines (one summary line instead; failures stay loud) and makes every capture helper
/// cuGraphUpload its fresh exec (the lazy path's first replay follows its capture immediately, so
/// only the boot graphs gain from an early upload).
pub(super) static PRECAP_ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub(super) static PRECAP_UPLOAD_FAIL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// S-B9-REL-VISIONFIX2 V2b: the status `metrics::graph_capture_lost` classifies a lost capture by:
/// the failing `cuStreamEndCapture` result, or "body error" when the capture closed cleanly but the
/// recorded body returned an error.
pub(super) fn capture_lost_status(end: cudarc::driver::sys::CUresult) -> String {
    if end != cudarc::driver::sys::CUresult::CUDA_SUCCESS { format!("{end:?}") } else { "body error".to_string() }
}

pub(super) fn graph_log_on() -> bool { !PRECAP_ACTIVE.load(std::sync::atomic::Ordering::Relaxed) }
/// WP04-v2: during the boot precapture, upload a freshly instantiated exec to the device on the
/// compute stream (cuGraphUpload performs the replay-time instantiation work WITHOUT launching a
/// kernel — device state is untouched). A failure only means the first replay uploads, as before.
pub(super) unsafe fn precap_upload(exec: cudarc::driver::sys::CUgraphExec, stream: cudarc::driver::sys::CUstream) {
    use cudarc::driver::sys;
    if PRECAP_ACTIVE.load(std::sync::atomic::Ordering::Relaxed)
        && sys::cuGraphUpload(exec, stream) != sys::CUresult::CUDA_SUCCESS {
        PRECAP_UPLOAD_FAIL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}
/// WP04: the graph keys, one definition shared by the replay sites and the boot precapture.
/// `qb` = the scorer grid bucket id (0 when the scorer is bucket-free, see qsa_grid_bucket_key).
/// TP-C: graph keys carry the SCRATCH that was captured (bits 48..55). A graph records the device
/// addresses of one Scratch's buffers; replaying it for another Scratch (the TP-B trap: a second
/// scratch of the same width) would read/write the first one's buffers while the host staged the
/// second. The first Scratch of a process has id 0, so the serve's single-scratch keys (and the
/// boot precapture's) are unchanged bit for bit.
pub(super) fn sk(sc: &Scratch) -> usize {
    (sc.gid as usize & 0x7F) << 48
}
/// W2-EXL3: decode-step graph key bit 47 = the step runs xq_vmask_rows (free between the vp tail
/// bits 45-46 and the scratch id at 48+).
pub(super) fn vmask_key(sc: &Scratch) -> usize {
    (sc.vmask_live as usize) << 47
}
/// W2-EXL3: u32 words of one row's allowed-token bitset (= json_schema::vocab_words).
pub fn vmask_words(vocab: usize) -> usize {
    vocab.div_ceil(32)
}
/// W2-EXL3: vocabulary ids one xq_vmask_rows block covers (256 threads x 8 ids).
pub(super) const VMASK_SPAN: usize = 2048;
pub(super) static SCRATCH_IDS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
pub(super) fn gkey_of(m: usize, qsa: bool, qb: usize) -> usize {
    m + if qsa { (1 << 20) | (qb << 21) } else { 0 }
}
/// S-B9-REL-QSAREGIME: the decode-step graph key bits of a mixed step, `md` = its dense-row count
/// (bits 8..15 of the step family; gkey_of keeps m in bits 0..7, the regime in bit 20 and the bucket
/// in bits 21..26). 0 for every non-mixed step, so the pre-existing keys are unchanged.
pub(super) fn mixed_key_bits(md: usize) -> usize {
    debug_assert!(md < 256);
    (md & 0xFF) << 8
}
/// S-B9-REL-QSAREGIME: for a decode step with these row positions, (the number of LEADING rows at or
/// below the sparse switch `lim` = qsa_limit() when some row is past it, the number of rows at/below
/// the switch that sit BEHIND a row past it). (0, 0) = not a mixed step. The leading rows keep the
/// dense kernels; the late ones cannot (the dense launches address rows 0..md) and stay sparse.
pub(crate) fn dense_prefix(pos: &[usize], lim: usize) -> (usize, usize) {
    if !pos.iter().any(|&p| p > lim) { return (0, 0); }
    let md = pos.iter().take_while(|&&p| p <= lim).count();
    (md, pos[md..].iter().filter(|&&p| p <= lim).count())
}
/// S-B9-REL-QSAREGIME: the row order a decode step needs for the per-row regime — the indices of
/// `pos` with the rows at/below the sparse switch first, each class in its given order (stable).
/// The identity whenever the rows are all on one side.
pub fn regime_order(pos: &[usize], lim: usize) -> Vec<usize> {
    let mut o: Vec<usize> = (0..pos.len()).filter(|&i| pos[i] <= lim).collect();
    o.extend((0..pos.len()).filter(|&i| pos[i] > lim));
    o
}
/// S-B9-REL-QSAREGIME: `k` clamped so that rows p..p+k do not straddle the sparse switch `lim`
/// (None = no switch): at/below the switch the round ends on it, past it nothing changes.
pub(crate) fn regime_depth_at(p: usize, k: usize, lim: Option<usize>) -> usize {
    match lim { Some(l) if p <= l => k.min(l - p), _ => k }
}
/// S-B9-REL-SMALLFIXES item 2: what a clamped round at position `p` runs — Some(depth >= 1) = a
/// speculative round of that depth (rows p..p+depth on one side of the switch), None = no draft
/// fits (p is the switch itself, or k == 0): the head-synced plain step.
pub(crate) fn clamped_round_plan(p: usize, k: usize, lim: Option<usize>) -> Option<usize> {
    match regime_depth_at(p, k, lim) { 0 => None, d => Some(d) }
}
/// S-B9-REL-SMALLFIXES 9a(iii): how many distinct mixed step shapes `lanes` lanes can form — a
/// step of width m in 2..=lanes with 1..=m-1 leading dense rows: lanes * (lanes - 1) / 2 (6 at 4
/// lanes, 120 at 16). The graph key also carries the sparse grid bucket and the penalty bit
/// (`gkey_of(m, qsa, qb) | mixed_key_bits(md) | pen_live << 40`), so the number of mixed-step
/// GRAPHS a boot can capture is this many per (bucket, penalty state) it meets, not this many.
pub fn mixed_step_shapes(lanes: usize) -> usize {
    lanes * lanes.saturating_sub(1) / 2
}
/// S-B9-REL-QSAREGIME: per-row attention regime in decode steps. Default on; `--qsa-row-regime 0`
/// restores one regime per step from its largest position (the pre-fix rule), for A/B.
pub fn row_regime_on() -> bool {
    crate::opts::var(crate::opt!("qsa-row-regime")).as_deref() != Ok("0")
}
/// Once-per-boot liveness lines of the per-row regime (AGENTS 4: a new dispatch path names itself).
pub(super) fn row_regime_note(md: usize, long: usize, late: usize) {
    static LIVE: std::sync::Once = std::sync::Once::new();
    static LATE: std::sync::Once = std::sync::Once::new();
    if md > 0 {
        LIVE.call_once(|| crate::drprintln!("[exl3] per-row attention regime LIVE: a decode step with {md} row(s) at/below the sparse \
switch and {long} past it keeps the short row(s) on the dense kernels (--qsa-row-regime 0 = one regime per step)"));
    }
    if late > 0 {
        LATE.call_once(|| crate::drprintln!("[exl3] per-row attention regime: {late} row(s) at/below the sparse switch sit BEHIND a \
longer row in a decode step and stay on the sparse kernels (the caller did not order short rows first)"));
    }
}
pub(super) fn vkey_of(slot: usize, m: usize, qsa: bool, qb: usize) -> usize {
    (1usize << 30) | ((qsa as usize) << 29) | (if qsa { qb << 16 } else { 0 }) | (slot << 8) | m
}
pub(super) fn dkey_of(slot: usize, k: usize, qsa: bool, qb: usize) -> usize {
    (2usize << 30) | (if qsa { (1usize << 20) | (qb << 21) } else { 0 }) | (slot << 8) | k
}
pub(super) fn pkey_of(slot: usize, i: usize, qsa: bool, qb: usize) -> usize {
    (3usize << 30) | (if qsa { (1usize << 20) | (qb << 21) } else { 0 }) | (slot << 8) | i
}
/// PACK3 (S-B9-CF-PACK3): the PACKED verify graph key — its own family
/// (4<<30) so it can never collide with the served vkey/dkey/pkey spaces.
/// Keyed on R_pad and the QSA regime ONLY: the graphable packed body (seg
/// trunk + accept_seg + commit_seg) reads every per-staging value from
/// device buffers (sc.seg and friends), so ONE graph serves any k_seg=2
/// staging with the same R_pad — any slots, any row0/len split. k_seg is
/// not in the key (every packed capture today is k_seg=2; the probe guards
/// this at lookup). Probe-only; the served boot never captures this family.
pub(super) fn pkeyv_of(rpad: usize, qsa: bool, qb: usize) -> usize {
    (4usize << 30) | ((qsa as usize) << 29) | (if qsa { qb << 16 } else { 0 }) | rpad
}
/// PACK4S: the SERVED packed verify key = PACK3's family key + the real-q sampler bit (bit 44,
/// the lone verify key's own `rq_verify_bits`): a pair holding a real-q lane replays the rq
/// family (its body runs `xq_sample_rows_rq`), every other pair the PACK4 family, whose key
/// this is bit for bit with `rq` false.
pub(super) fn pack_key(r: usize, qsa: bool, qb: usize, rq: bool) -> usize {
    pkeyv_of(r, qsa, qb) | rq_verify_bits(rq)
}
/// WP24: draft graph key bits 42-43 = the passes' rq mode (0 = today's tail: the boot precapture's
/// keys and graphs; 1 = + the dump list; 2 = real-q old-tail sampling). Verify keys carry the
/// real-q sampler as bit 44 (rq_verify_key). Both 0 unless --spec-sampling ratio / --wp24-dump.
pub(super) fn rq_draft_key(sc: &Scratch) -> usize {
    rq_draft_bits(sc.rq_mode)
}
pub(super) fn rq_verify_key(sc: &Scratch) -> usize {
    rq_verify_bits(sc.rq_live)
}
pub(super) fn rq_draft_bits(mode: u8) -> usize {
    (mode as usize & 3) << 42
}
pub(super) fn rq_verify_bits(live: bool) -> usize {
    (live as usize) << 44
}
