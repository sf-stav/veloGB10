//! CUDA graph captures that produced no graph: counters, the once-per-shape log line and what happens next per graph family.
//! The caller keeps its own `capture failed` line; this file only counts and logs the first loss of each (kind, shape). Map: `mod.rs`.

use super::*;

/// S-B9-REL-VISIONFIX2 V2b: CUDA graph captures that produced no graph. A capture ends
/// `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED` when a CUDA call that is illegal inside stream capture lands in
/// the capture window: from another thread (the vision tower, measured in
/// tests/vision_capture_race_test.rs) or from the capturing thread itself (a NULL-stream allocation, a
/// library-handle destruction). The step then runs eagerly; what happens to the SHAPE depends on the graph
/// family (`capture_next`): the lazily captured ones are captured again at their next use, three are
/// captured once at boot and are not retried. The first loss of each (kind, shape) is logged; every one is
/// counted. S-B9-REL-INTEG C1 added the counting of the sites that used to return silently (BeginCapture
/// refused at the NVFP4 / DFlash2 sites, instantiate failures, the TP spine audit's refusal).
pub(super) static CAPTURE_INVALIDATED: AtomicU64 = AtomicU64::new(0);
pub(super) static CAPTURE_FAILED: AtomicU64 = AtomicU64::new(0);
/// The (kind, shape) keys already logged this boot; bounded, so a pathological stream of distinct
/// shapes cannot grow it (the counters keep counting past the bound).
static CAPTURE_LOGGED: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
const CAPTURE_LOG_KEYS: usize = 64;
/// The status name `cuStreamEndCapture` returns for an invalidated capture.
pub const CAPTURE_INVALIDATED_STATUS: &str = "CUDA_ERROR_STREAM_CAPTURE_INVALIDATED";

/// Record a capture that produced no graph. `kind` names the graph family ("step", "verify", ...),
/// `shape` its key in words ("width 8"), `status` the `{:?}` of the failing CUDA result (or a short
/// reason for a body error). Returns true when this call logged the line (the first loss of that
/// kind and shape). The caller keeps its own, existing `... capture failed ...` line.
pub fn graph_capture_lost(kind: &str, shape: &str, status: &str) -> bool {
    let invalidated = status == CAPTURE_INVALIDATED_STATUS;
    // (the key and the line are `capture_lost_line`'s: a pure function the tests read)
    if invalidated { CAPTURE_INVALIDATED.fetch_add(1, Relaxed); } else { CAPTURE_FAILED.fetch_add(1, Relaxed); }
    let key = format!("{kind}|{shape}|{}", if invalidated { "invalidated" } else { "failed" });
    let first = {
        let mut seen = CAPTURE_LOGGED.lock().unwrap_or_else(|e| e.into_inner());
        if seen.iter().any(|k| *k == key) || seen.len() >= CAPTURE_LOG_KEYS { false } else { seen.push(key); true }
    };
    if first { crate::rprintln!("{}", capture_lost_line(kind, shape, status)); }
    first
}

/// What happens after a lost capture, by graph family. The lazily captured families (the EXL3 step /
/// verify / packed-verify / draft-pass / draft-chain / reprime graphs, the NVFP4 verify graph, the DFlash2
/// inject graph) run the step eagerly and capture the shape again at its next use. Three are captured ONCE,
/// at boot, and are never retried (S-B9-REL-INTEG C1: the texts used to promise a retry for every family).
fn capture_next(kind: &str) -> &'static str {
    match kind {
        "nvfp4-decode" => "This decode-graph set is captured once, at boot, for every batch size: one loss drops ALL decode \
graphs for this boot and decode runs eagerly (nothing is retried)",
        "nvfp4-decode-sample" => "This graph is captured once per batch size, at boot, and is never retried: that batch size \
takes the non-graph sampling path for this boot",
        "df2-draft-round" => "This graph is captured once, at boot, and is never retried: the DFlash2 draft round runs \
eagerly for this boot",
        _ => "This step ran eagerly and the shape is captured again at its next use",
    }
}

/// The once-per-shape `[graph]` line of a lost capture (pure, so the tests read it).
pub(super) fn capture_lost_line(kind: &str, shape: &str, status: &str) -> String {
    if status == CAPTURE_INVALIDATED_STATUS {
        format!("[graph] a {kind} capture ({shape}) was INVALIDATED ({status}): a CUDA call that is illegal inside stream \
capture landed in its window (from another thread, e.g. the vision tower, or from the capturing thread itself), so no \
graph was produced. {}. Counted in velogb10_graph_capture_invalidated_total; this line prints once per shape per boot",
                capture_next(kind))
    } else {
        format!("[graph] a {kind} capture ({shape}) produced no graph ({status}). {}. Counted in \
velogb10_graph_capture_failed_total; this line prints once per shape per boot", capture_next(kind))
    }
}
pub fn graph_captures_invalidated() -> u64 { CAPTURE_INVALIDATED.load(Relaxed) }
pub fn graph_captures_failed() -> u64 { CAPTURE_FAILED.load(Relaxed) }
