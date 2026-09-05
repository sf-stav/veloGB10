//! OTel generation-telemetry EMITTER (engine side).
//!
//! Streams the generation to an OTel-compatible receiver as OTLP/HTTP-JSON
//! (`ExportLogsServiceRequest` on POST `<--otel-endpoint>/v1/logs`, batched; a `/v1/metrics`
//! interface stub wired for the client's Live Stats view — emission is a follow-up).
//!
//! THE CONTRACT (the client decodes exactly this; PLAN/OTEL_EMITTER_PROMPT.md):
//! - one `LogRecord` per OpenAI-compatible completion chunk the SSE path emits. `body.stringValue`
//!   IS the ACTUAL SSE chunk payload bytes (single source of truth — forwarded, never re-derived
//!   or re-encoded; `chat.completion.chunk` JSON: choices[].delta content/role/tool_calls +
//!   finish_reason). Attributes, fixed order: `model.id`, `topology` (single|tp2|tp4),
//!   `request.id` (= the SESSION key — stable across a conversation's turns; see below),
//!   `token.index` (0 = start, 1..= deltas, last = end, per generation),
//!   `event` (stream_start|stream_delta|stream_end|status), `generation.id` (per POST).
//!   `timeUnixNano`/`observedTimeUnixNano` at push time (64-bit ints as the
//!   decimal STRINGS proto3-JSON mandates).
//! - `event: status` — periodic server record (not per request) carrying the live
//!   `model.id`+`topology` pair; `request.id` is the empty string (never collides with a
//!   session key). This is what refreshes the client's "currently running: model @ topology".
//!
//! SESSION IDENTITY — what `request.id` means (owner report 2026-09-04: every response showed
//! up as a separate session in the client; a CONTINUOUS conversation must carry ONE id):
//! the OpenAI chat API is stateless — turn N+1's `messages` array contains turn N's messages
//! array as an EXACT prefix (the history replays verbatim; only the assistant reply and the
//! new user turn are appended). That gives the engine a precise, guess-free continuation rule:
//! request R′ continues session S iff S's last request's full messages array is a prefix of
//! R′'s. [`SessionRegistry`] applies it: an explicit client key (X-Session-Id header or
//! `metadata.session_id`/`metadata.conversation_id`) always wins; otherwise the engine infers
//! the session from the prefix rule (a bare retry of the same messages also lands in the same
//! session; a regenerated history is genuinely a different array and mints a new one — send
//! the header for exact control). Concurrently ACTIVE sessions are bounded by --max-batch
//! (batch=1 ⇒ one session active at a time), while identity survives across turns. All of this
//! lives behind `--otel-endpoint`: with the emitter off there is NO session work at all, and
//! even on, it is one registry touch per REQUEST — never per token.
//!
//! ABSOLUTE LEAST DECODE INTERFERENCE — the design IS the gate:
//! 1. OFF by default: no `--otel-endpoint` ⇒ `AppState.otel: None`; every hook site compiles to
//!    one `if let Some(..)` branch. Absent = zero cost.
//! 2. Hot-path push = a bounded lock-free Vyukov MPMC ring (pre-allocated slots): per token it is
//!    ONE CAS on the producer index + ONE release-store publish, plus an inline memcpy of the
//!    already-built SSE bytes. NO allocation, NO mutex, NO syscall, NO per-token wakeup.
//!    Chunks > INLINE_CAP (rare: think-flush, tool-call JSON) take one `Arc<str>` instead — a
//!    bounded handful per GENERATION, never per token, still no lock/syscall/wake.
//! 3. Drop-on-full, best-effort: ring full (client slow) ⇒ the row is DROPPED and the hot path
//!    moves on. Decode is sacred; telemetry is droppable; the client can never back-pressure us.
//! 4. Timer-polled sender OFF the compute stream: ONE spawned task drains every
//!    `--otel-batch-interval-ms`, batches, POSTs. It never wakes per token, never runs on a GPU
//!    thread, does NO GPU work — it forwards already-generated tokens. A failed POST drops the
//!    batch (throttled log) and keeps going.
//! 5. Hooks live ONLY in the SSE consumption path of /v1/chat/completions (server.rs) — the
//!    compute path (batch.rs / GPU) is untouched. Non-streaming requests have no SSE deltas and
//!    emit nothing.
//!
//! The HTTP client is hand-rolled over tokio's TcpStream (http:// only — no TLS dependency is
//! added to this tree on purpose).

use std::cell::UnsafeCell;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Inline body cap per ring slot. Per-token SSE deltas are far below this; only large held-back
/// flushes (think-close, tool-call JSON) exceed it and take the `Shared` arm.
const INLINE_CAP: usize = 224;

pub const EVENT_START: u8 = 0;
pub const EVENT_DELTA: u8 = 1;
pub const EVENT_END: u8 = 2;
pub const EVENT_STATUS: u8 = 3;

pub fn event_name(e: u8) -> &'static str {
    match e {
        EVENT_START => "stream_start",
        EVENT_DELTA => "stream_delta",
        EVENT_END => "stream_end",
        EVENT_STATUS => "status",
        _ => "unknown",
    }
}

fn now_unix_nano() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// Resolved, immutable emitter configuration (parsed once in run_server).
#[derive(Clone, Debug)]
pub struct OtelConfig {
    /// Base URL, no trailing '/', e.g. "http://127.0.0.1:4318". `hostport` is derived.
    pub endpoint: String,
    /// Max LogRecords per POST /v1/logs.
    pub batch_size: usize,
    /// Sender drain period (timer-polled; the hot path never wakes the sender).
    pub batch_interval_ms: u64,
    /// false = model/topology + start/end events only (no per-token delta records).
    pub include_tokens: bool,
    /// Overrides for the two identity attributes; `None` = auto from the serve config + world.
    pub model_id: Option<String>,
    pub topology: Option<String>,
}

impl OtelConfig {
    /// (host, port) for the raw HTTP client. `http://` only — the dependency-free client does
    /// not speak TLS; an https endpoint is an operator error and refuses loudly at startup.
    pub fn hostport(&self) -> Result<(String, u16), String> {
        let rest = self.endpoint.strip_prefix("http://").ok_or_else(|| {
            format!("--otel-endpoint must be http://<host>[:port] (got {:?}); the emitter's \
                     dependency-free client does not speak TLS", self.endpoint)
        })?;
        let authority = rest.split('/').next().unwrap_or("");
        if authority.is_empty() {
            return Err(format!("--otel-endpoint has no host: {:?}", self.endpoint));
        }
        // Bracketed IPv6 keeps its brackets for TcpStream::connect's parser.
        if authority.starts_with('[') {
            let has_port = authority.rsplit_once("]:").is_some();
            let hp = if has_port { authority.to_string() } else { format!("{authority}:4318") };
            return Ok((hp, 4318));
        }
        match authority.rsplit_once(':') {
            Some((h, p)) => {
                let port: u16 = p.parse().map_err(|_| format!("--otel-endpoint bad port {p:?}"))?;
                Ok((h.to_string(), port))
            }
            None => Ok((authority.to_string(), 4318)), // OTLP/HTTP default port
        }
    }
}

/// `topology` attribute auto-derived from the serve world (AGENTS §7: one generic vocabulary).
pub fn topology_from_world(world: Option<u32>) -> String {
    match world {
        None => "single".to_string(),
        Some(n) => format!("tp{n}"),
    }
}

/// Per-request identity (built ONCE per request at stream open — never per token).
pub struct ReqCtx {
    pub model_id: Arc<str>,
    pub topology: Arc<str>,
    /// The serving request's id == the SSE chunks' `id` ("chatcmpl-…"): the client can correlate
    /// OTel records and SSE chunks without any extra channel.
    pub request_id: Arc<str>,
    pub generation_id: Arc<str>,
}

/// One telemetry row. Fixed-size metadata + a body that is either an inline copy of the SSE
/// chunk bytes (the per-token steady state) or a shared `Arc<str>` (rare large chunks).
pub struct Record {
    ctx: Arc<ReqCtx>,
    body: Body,
    token_index: u32,
    event: u8,
    ts_unix_nano: u64,
}

enum Body {
    Inline { len: u16, buf: [u8; INLINE_CAP] },
    Shared(Arc<str>),
}

impl Body {
    fn from_chunk(chunk: &str) -> Body {
        let b = chunk.as_bytes();
        if b.len() <= INLINE_CAP {
            let mut buf = [0u8; INLINE_CAP];
            buf[..b.len()].copy_from_slice(b);
            Body::Inline { len: b.len() as u16, buf }
        } else {
            // Rare (>224B chunk): one Arc allocation, bounded by large-chunk count per
            // generation. The steady per-token path NEVER lands here.
            Body::Shared(Arc::from(chunk))
        }
    }
    fn as_str(&self) -> &str {
        match self {
            Body::Inline { len, buf } => std::str::from_utf8(&buf[..*len as usize])
                .unwrap_or(""), // bytes came from a &str: always valid UTF-8
            Body::Shared(s) => s,
        }
    }
}

impl Record {
    fn new(ctx: &Arc<ReqCtx>, event: u8, token_index: u32, chunk: &str) -> Record {
        Record { ctx: Arc::clone(ctx), body: Body::from_chunk(chunk), token_index, event,
                 ts_unix_nano: now_unix_nano() }
    }
    fn status(ctx: &Arc<ReqCtx>) -> Record {
        Record { ctx: Arc::clone(ctx), body: Body::from_chunk("status"), token_index: 0,
                 event: EVENT_STATUS, ts_unix_nano: now_unix_nano() }
    }
}

// ─── The bounded lock-free ring (Vyukov bounded MPMC) ─────────────────────────────────────
//
// Pre-allocated slots, no allocation on push/pop, no lock. push = one CAS on `head` (ticket)
// then ONE release-store on the slot's `seq` (publish fence). pop = the consumer mirror.
// Full ⇒ push returns false and the caller DROPS (drop-on-full, decode is sacred).

struct Slot {
    seq: AtomicUsize,
    rec: UnsafeCell<MaybeUninit<Record>>,
}

pub struct Ring {
    buf: Box<[Slot]>,
    mask: usize,
    head: AtomicUsize, // producers (fetch/CAS)
    tail: AtomicUsize, // consumer (the single sender task)
    dropped: AtomicU64,
}

unsafe impl Send for Ring {}
// Sound: slot access is exclusively mediated by the seq protocol (a slot is written only
// between a won head-CAS and its publish, read only after a won tail-CAS observing the publish).
unsafe impl Sync for Ring {}

impl Ring {
    pub fn new(capacity: usize) -> Ring {
        let cap = capacity.max(2).next_power_of_two();
        let mut slots = Vec::with_capacity(cap);
        for i in 0..cap {
            slots.push(Slot { seq: AtomicUsize::new(i), rec: UnsafeCell::new(MaybeUninit::uninit()) });
        }
        Ring { buf: slots.into_boxed_slice(), mask: cap - 1,
               head: AtomicUsize::new(0), tail: AtomicUsize::new(0), dropped: AtomicU64::new(0) }
    }

    pub fn capacity(&self) -> usize { self.mask + 1 }

    /// Hot-path push. Best-effort: false = ring full ⇒ record DROPPED, counted, move on.
    #[inline]
    pub fn push(&self, rec: Record) -> bool {
        let mut pos = self.head.load(Ordering::Relaxed);
        loop {
            let slot = unsafe { self.buf.get_unchecked(pos & self.mask) };
            let seq = slot.seq.load(Ordering::Acquire);
            let dif = seq as isize - pos as isize;
            if dif == 0 {
                if self.head.compare_exchange_weak(pos, pos.wrapping_add(1),
                                                   Ordering::Relaxed, Ordering::Relaxed).is_ok() {
                    unsafe {
                        // Exclusive by the won ticket: write BEFORE the release publish.
                        (*slot.rec.get()).write(rec);
                        slot.seq.store(pos.wrapping_add(1), Ordering::Release); // ONE publish fence
                    }
                    return true;
                }
                // CAS lost a racing producer: pos was reloaded by compare_exchange_weak; retry.
            } else if dif < 0 {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                return false; // FULL: drop-on-full — the client can never back-pressure decode
            } else {
                pos = self.head.load(Ordering::Relaxed);
            }
        }
    }

    /// Consumer pop (the sender task). None = empty (never blocks — timer-polled).
    pub fn pop(&self) -> Option<Record> {
        let mut pos = self.tail.load(Ordering::Relaxed);
        loop {
            let slot = unsafe { self.buf.get_unchecked(pos & self.mask) };
            let seq = slot.seq.load(Ordering::Acquire);
            let dif = seq as isize - pos.wrapping_add(1) as isize;
            if dif == 0 {
                if self.tail.compare_exchange_weak(pos, pos.wrapping_add(1),
                                                   Ordering::Relaxed, Ordering::Relaxed).is_ok() {
                    let rec = unsafe { (*slot.rec.get()).assume_init_read() };
                    slot.seq.store(pos.wrapping_add(self.mask + 1), Ordering::Release);
                    return Some(rec);
                }
            } else if dif < 0 {
                return None; // empty
            } else {
                pos = self.tail.load(Ordering::Relaxed);
            }
        }
    }

    /// Drain up to `max` records (one batch).
    pub fn pop_batch(&self, out: &mut Vec<Record>, max: usize) -> usize {
        let mut n = 0;
        while n < max {
            match self.pop() {
                Some(r) => { out.push(r); n += 1; }
                None => break,
            }
        }
        n
    }

    pub fn dropped(&self) -> u64 { self.dropped.load(Ordering::Relaxed) }
}

// ─── Live Stats interface (stub) ──────────────────────────────────────────────────────────
//
// POST <endpoint>/v1/metrics (OTLP ExportMetricsServiceRequest) is WIRED end to end — the
// sender calls the source every tick and would export it — but no engine component implements
// the interface yet, so the stub returns None and the POST is skipped. Metrics emission
// (request count, token count, tok/s, accepted) is a follow-up; the client's Live Stats view
// already has its target.

/// Live Stats snapshot the metrics exporter will send once a source is wired.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LiveStats {
    pub requests: u64,
    pub tokens: u64,
    pub tokens_per_sec: f64,
    pub accepted: u64,
}

/// The metrics source interface. `None` = not wired (stub) ⇒ no /v1/metrics POST.
pub trait LiveStatsSource: Send + Sync {
    fn live_stats(&self) -> Option<LiveStats>;
}

/// The wired no-op stub.
pub struct NoMetrics;
impl LiveStatsSource for NoMetrics {
    fn live_stats(&self) -> Option<LiveStats> { None }
}

// ─── Session registry (the conversation key) ──────────────────────────────────────────────
//
// See the module doc for the continuation rule. The registry maps the ROLLING fingerprint of a
// request's full messages array (hash of the per-message canonical JSON, in order) to a session
// id. Entries are append-only per turn (each turn registers its own length so the NEXT turn
// matches in one hash), TTL-bounded and capacity-bounded; one parking_lot mutex touched ONCE
// per request — never on the token path.

const SESSION_TTL: Duration = Duration::from_secs(30 * 60);
const SESSION_CAP: usize = 256;
/// Max entries inspected per resolve (newest first) — bounds the worst-case hashing.
const SESSION_SCAN: usize = 64;

struct SessionEntry {
    /// Message count of the request that registered this fingerprint.
    fp_len: usize,
    /// Hash of the per-message canonical JSON of that FULL array.
    fp: u64,
    session: Arc<str>,
    last: Instant,
}

fn msg_hash(per_msg: &[String]) -> u64 {
    let mut h = DefaultHasher::new();
    for m in per_msg {
        h.write(m.as_bytes());
        h.write(&[0]); // separator: ["ab","c"] must never equal ["a","bc"]
    }
    h.finish()
}

#[derive(Default)]
pub struct SessionRegistry {
    inner: parking_lot::Mutex<Vec<SessionEntry>>,
}

impl SessionRegistry {
    pub fn new() -> Self { Self::default() }

    /// Resolve the session id for one request. `explicit` (client key) wins; otherwise the
    /// messages-prefix continuation rule. `per_msg` = canonical JSON per message, in order —
    /// the caller serializes once per request; this never touches the token path.
    pub fn resolve(&self, explicit: Option<String>, per_msg: &[String]) -> String {
        let mut g = self.inner.lock();
        let now = Instant::now();
        g.retain(|e| now.duration_since(e.last) < SESSION_TTL); // TTL sweep
        let fp_all = msg_hash(per_msg);
        if let Some(x) = explicit {
            // Explicit key: refresh this session's fingerprint (drop any stale entry of the
            // same id first) so LATER bare turns still group by the prefix rule.
            g.retain(|e| e.session.as_ref() != x.as_str());
            g.push(SessionEntry { fp_len: per_msg.len(), fp: fp_all,
                                  session: Arc::from(x.as_str()), last: now });
            Self::evict(&mut g);
            return x;
        }
        // Continuation scan, newest first: an entry whose full messages array is a PREFIX of
        // this request's (same rolling hash at that length) is this session's earlier turn.
        g.sort_by(|a, b| b.last.cmp(&a.last));
        for e in g.iter_mut().take(SESSION_SCAN) {
            if e.fp_len <= per_msg.len() && msg_hash(&per_msg[..e.fp_len]) == e.fp {
                e.last = now;
                let s = e.session.to_string();
                // Register THIS length too, so the next turn matches in a single hash.
                if e.fp_len != per_msg.len() {
                    g.push(SessionEntry { fp_len: per_msg.len(), fp: fp_all,
                                          session: Arc::from(s.as_str()), last: now });
                    Self::evict(&mut g);
                }
                return s;
            }
        }
        // No continuation: a NEW session.
        let s: Arc<str> = Arc::from(
            format!("sess-{}", &uuid::Uuid::new_v4().simple().to_string()[..12]).as_str());
        g.push(SessionEntry { fp_len: per_msg.len(), fp: fp_all, session: Arc::clone(&s), last: now });
        Self::evict(&mut g);
        s.to_string()
    }

    fn evict(g: &mut Vec<SessionEntry>) {
        if g.len() <= SESSION_CAP { return; }
        // Newest-first ordering is not guaranteed after pushes: evict by real timestamp.
        while g.len() > SESSION_CAP {
            let idx = g.iter().enumerate()
                .min_by_key(|(_, e)| e.last).map(|(i, _)| i).unwrap_or(0);
            g.swap_remove(idx);
        }
    }
}

// ─── The sink (what AppState holds) ───────────────────────────────────────────────────────

pub struct OtelSink {
    pub cfg: OtelConfig,
    ring: Ring,
    model_id: Arc<str>,
    topology: Arc<str>,
    metrics: Box<dyn LiveStatsSource>,
    sessions: SessionRegistry,
    sent: AtomicU64,
    failed_batches: AtomicU64,
}

impl OtelSink {
    /// Build the sink. The CALLER spawns [`run_sender`] on the server's tokio runtime.
    pub fn new(cfg: OtelConfig, auto_model_id: &str, auto_topology: &str) -> Arc<OtelSink> {
        let ring = Ring::new(8192); // ~20 s of buffer at batch-8 · 50 tok/s — droppable either way
        Arc::new(OtelSink {
            model_id: Arc::from(cfg.model_id.as_deref().unwrap_or(auto_model_id)),
            topology: Arc::from(cfg.topology.as_deref().unwrap_or(auto_topology)),
            cfg, ring,
            metrics: Box::new(NoMetrics),
            sessions: SessionRegistry::new(),
            sent: AtomicU64::new(0),
            failed_batches: AtomicU64::new(0),
        })
    }

    /// Resolve THIS request's session id: explicit client key first, else the messages-prefix
    /// continuation rule (see [`SessionRegistry`] + module doc). One call per REQUEST.
    pub fn resolve_session(&self, explicit: Option<String>, per_msg_json: &[String]) -> String {
        self.sessions.resolve(explicit, per_msg_json)
    }

    /// Open a per-request telemetry handle. `request_id` is the SESSION key (explicit client
    /// key or engine-inferred continuation — see [`SessionRegistry`]); the SSE chunks still
    /// carry their own `chatcmpl-…` id (OpenAI spec), while `generation_id` is this POST's
    /// execution id (a retry of the same request would mint a new one).
    pub fn open_request(self: &Arc<Self>, request_id: &str, generation_id: &str) -> OtelRequest {
        OtelRequest {
            sink: Arc::clone(self),
            ctx: Arc::new(ReqCtx {
                model_id: Arc::clone(&self.model_id),
                topology: Arc::clone(&self.topology),
                request_id: Arc::from(request_id),
                generation_id: Arc::from(generation_id),
            }),
            next_index: AtomicU32::new(0),
        }
    }

    #[inline]
    fn push_record(&self, ctx: &Arc<ReqCtx>, event: u8, token_index: u32, chunk: &str) {
        let rec = Record::new(ctx, event, token_index, chunk);
        if !self.ring.push(rec) {
            // counted inside the ring; decode moves on untouched
        }
    }

    pub fn stats(&self) -> (u64, u64, u64) {
        (self.sent.load(Ordering::Relaxed), self.ring.dropped(), self.failed_batches.load(Ordering::Relaxed))
    }
}

/// Per-request handle held by ONE streaming response task.
pub struct OtelRequest {
    sink: Arc<OtelSink>,
    ctx: Arc<ReqCtx>,
    next_index: AtomicU32,
}

impl OtelRequest {
    /// stream_start — the SSE role chunk. token.index 0.
    pub fn start(&self, chunk: &str) {
        let i = self.next_index.fetch_add(1, Ordering::Relaxed);
        self.sink.push_record(&self.ctx, EVENT_START, i, chunk);
    }
    /// stream_delta — one SSE completion chunk, forwarded byte-for-byte. Skipped entirely when
    /// --otel-include-tokens=off (start/end/status still flow; the per-token path is one branch).
    pub fn delta(&self, chunk: &str) {
        if !self.sink.cfg.include_tokens { return; }
        let i = self.next_index.fetch_add(1, Ordering::Relaxed);
        self.sink.push_record(&self.ctx, EVENT_DELTA, i, chunk);
    }
    /// stream_end — the SSE finish chunk (carries finish_reason). token.index = last+1.
    pub fn end(&self, chunk: &str) {
        let i = self.next_index.fetch_add(1, Ordering::Relaxed);
        self.sink.push_record(&self.ctx, EVENT_END, i, chunk);
    }
}

// ─── OTLP/HTTP-JSON encoding + the timer-polled sender ────────────────────────────────────

/// Periodic server status record period (refreshes the client's model@topology line).
const STATUS_PERIOD: Duration = Duration::from_secs(10);

fn attr_string(key: &str, v: &str) -> serde_json::Value {
    serde_json::json!({"key": key, "value": {"stringValue": v}})
}
fn attr_int(key: &str, v: u32) -> serde_json::Value {
    // OTLP JSON: 64-bit ints are decimal STRINGS (proto3 JSON mapping).
    serde_json::json!({"key": key, "value": {"intValue": v.to_string()}})
}

fn record_json(rec: &Record) -> serde_json::Value {
    serde_json::json!({
        "timeUnixNano": rec.ts_unix_nano.to_string(),
        "observedTimeUnixNano": rec.ts_unix_nano.to_string(),
        "body": {"stringValue": rec.body.as_str()},
        "attributes": [
            attr_string("model.id", &rec.ctx.model_id),
            attr_string("topology", &rec.ctx.topology),
            attr_string("request.id", &rec.ctx.request_id),
            attr_int("token.index", rec.token_index),
            attr_string("event", event_name(rec.event)),
            attr_string("generation.id", &rec.ctx.generation_id),
        ],
    })
}

/// The batched ExportLogsServiceRequest the contract fixes: one resourceLogs → one scopeLogs →
/// up to batch_size LogRecords per POST.
pub fn otlp_logs_json(sink: &OtelSink, records: &[Record]) -> String {
    let logs = serde_json::json!({
        "resourceLogs": [{
            "resource": {"attributes": [
                attr_string("service.name", "gb10_inference"),
            ]},
            "scopeLogs": [{
                "scope": {"name": "gb10.generation.emitter", "version": env!("CARGO_PKG_VERSION")},
                "logRecords": records.iter().map(record_json).collect::<Vec<_>>(),
            }],
        }]
    });
    let _ = sink; // identity anchor: the resource describes this engine
    logs.to_string()
}

/// Minimal dependency-free HTTP/1.1 POST (connect-per-batch, `Connection: close`, hard timeouts).
/// A failure drops the batch — the sender NEVER blocks the engine, only itself.
async fn post_http(cfg: &OtelConfig, path: &str, body: &str) -> Result<(), String> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let (host, port) = cfg.hostport()?;
    let addr = format!("{host}:{port}");
    let mut stream = tokio::time::timeout(Duration::from_secs(3), tokio::net::TcpStream::connect(&addr))
        .await.map_err(|_| format!("connect {addr}: timeout"))?
        .map_err(|e| format!("connect {addr}: {e}"))?;
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
    let write_fut = async { stream.write_all(req.as_bytes()).await.map_err(|e| e.to_string()) };
    tokio::time::timeout(Duration::from_secs(5), write_fut)
        .await.map_err(|_| "write: timeout".to_string())?
        .map_err(|e| format!("write: {e}"))?;
    let mut buf = [0u8; 1024];
    let read_fut = async {
        let n = stream.read(&mut buf).await.map_err(|e| e.to_string())?;
        Ok::<usize, String>(n)
    };
    let n = tokio::time::timeout(Duration::from_secs(5), read_fut)
        .await.map_err(|_| "read: timeout".to_string())?
        .map_err(|e| format!("read: {e}"))?;
    let head = String::from_utf8_lossy(&buf[..n]);
    let status_ok = head.starts_with("HTTP/1.1 2") || head.starts_with("HTTP/1.0 2");
    if status_ok { Ok(()) } else {
        let first = head.lines().next().unwrap_or("<no status line>").to_string();
        Err(format!("receiver said {first}"))
    }
}

/// THE SENDER: ONE task, timer-polled every `batch_interval_ms`. Never wakes per token, never
/// runs on a GPU thread, does NO GPU work — it only forwards already-generated tokens. Ring
/// full / POST failed ⇒ records are dropped; decode is untouched by construction.
pub async fn run_sender(sink: Arc<OtelSink>) {
    let interval = Duration::from_millis(sink.cfg.batch_interval_ms.max(1));
    // Live Stats stub status: the interface is wired; nothing implements it yet.
    if sink.metrics.live_stats().is_none() {
        eprintln!("[otel] /v1/metrics wired (stub): no Live Stats source yet — emission is a follow-up");
    }
    eprintln!("[otel] sender up: {} every {} ms, batch {}, include_tokens={} (model.id={}, topology={})",
              sink.cfg.endpoint, sink.cfg.batch_interval_ms, sink.cfg.batch_size,
              sink.cfg.include_tokens, sink.model_id, sink.topology);
    let mut last_status = Instant::now();
    let mut last_drop_log = Instant::now();
    let mut drop_logged = 0u64;
    let mut fail_streak = 0u32;
    let server_ctx = Arc::new(ReqCtx {
        model_id: Arc::clone(&sink.model_id),
        topology: Arc::clone(&sink.topology),
        request_id: Arc::from(""),      // server-liveness record: no request key
        generation_id: Arc::from("server"),
    });
    loop {
        tokio::time::sleep(interval).await;
        let mut rows: Vec<Record> = Vec::with_capacity(sink.cfg.batch_size.min(1024));
        sink.ring.pop_batch(&mut rows, sink.cfg.batch_size);
        // Periodic status record goes through the SAME batch (never its own POST).
        if last_status.elapsed() >= STATUS_PERIOD {
            rows.push(Record::status(&server_ctx));
            last_status = Instant::now();
        }
        if rows.is_empty() { continue; }
        let body = otlp_logs_json(&sink, &rows);
        match post_http(&sink.cfg, "/v1/logs", &body).await {
            Ok(()) => {
                sink.sent.fetch_add(rows.len() as u64, Ordering::Relaxed);
                fail_streak = 0;
            }
            Err(e) => {
                sink.failed_batches.fetch_add(1, Ordering::Relaxed);
                fail_streak += 1;
                if fail_streak <= 3 || fail_streak % 100 == 0 {
                    eprintln!("[otel] POST /v1/logs failed (streak {fail_streak}): {e} — {} records dropped, decode unaffected",
                              rows.len());
                }
            }
        }
        let dropped = sink.ring.dropped();
        if dropped != drop_logged && last_drop_log.elapsed() >= Duration::from_secs(5) {
            eprintln!("[otel] {dropped} records dropped ring-full so far (client slower than decode; decode unaffected)");
            drop_logged = dropped;
            last_drop_log = Instant::now();
        }
    }
}

// ─── Tests: ring correctness, drop-on-full, contract shape, lifecycle, same-info ──────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn cfg(include_tokens: bool) -> OtelConfig {
        OtelConfig {
            endpoint: "http://127.0.0.1:4318".into(),
            batch_size: 512, batch_interval_ms: 50,
            include_tokens, model_id: None, topology: None,
        }
    }

    fn sink_and_req(include_tokens: bool, rid: &str) -> (Arc<OtelSink>, OtelRequest) {
        let sink = OtelSink::new(cfg(include_tokens), "test-model", "single");
        let req = sink.open_request(rid, "gen-test");
        (sink, req)
    }

    fn drain(sink: &OtelSink) -> Vec<Record> {
        let mut v = Vec::new();
        while let Some(r) = sink.ring.pop() { v.push(r); }
        v
    }

    fn attrmap(rec: &Record) -> Vec<(String, String)> {
        let j = record_json(rec);
        j["attributes"].as_array().unwrap().iter().map(|a| (
            a["key"].as_str().unwrap().to_string(),
            a["value"]["stringValue"].as_str().map(str::to_string)
                .or_else(|| a["value"]["intValue"].as_str().map(str::to_string)).unwrap(),
        )).collect()
    }

    /// THE hot-path law: push = ticket CAS + publish store; no loss, FIFO order, single producer.
    #[test]
    fn ring_fifo_single_producer() {
        let ring = Ring::new(1024);
        for i in 0..500u32 {
            let ctx = Arc::new(ReqCtx {
                model_id: "m".into(), topology: "single".into(),
                request_id: "r".into(), generation_id: "g".into(),
            });
            assert!(ring.push(Record::new(&ctx, EVENT_DELTA, i, "x")));
        }
        for i in 0..500u32 {
            let r = ring.pop().expect("record present");
            assert_eq!(r.token_index, i, "FIFO order");
        }
        assert!(ring.pop().is_none());
        assert_eq!(ring.dropped(), 0);
    }

    /// MPSC stress: 4 producers × 2500 rows through the lock-free ring — exactly-once delivery.
    #[test]
    fn ring_mpsc_stress_exactly_once() {
        let ring = Arc::new(Ring::new(4096));
        let mut handles = Vec::new();
        for p in 0..4u32 {
            let ring = Arc::clone(&ring);
            handles.push(std::thread::spawn(move || {
                for i in 0..2500u32 {
                    let ctx = Arc::new(ReqCtx {
                        model_id: "m".into(), topology: "single".into(),
                        request_id: format!("p{p}").into(), generation_id: format!("{p}:{i}").into(),
                    });
                    let mut ok = false;
                    // Retry on full so the test measures delivery, not drop policy (that's the next test).
                    while !ok {
                        ok = ring.push(Record::new(&ctx, EVENT_DELTA, i, "x"));
                        if !ok { std::hint::spin_loop(); }
                    }
                }
            }));
        }
        let received = Arc::new(std::sync::Mutex::new(Vec::new()));
        let consumer = {
            let ring = Arc::clone(&ring);
            let received = Arc::clone(&received);
            std::thread::spawn(move || {
                loop {
                    let mut got = 0;
                    while let Some(r) = ring.pop() {
                        received.lock().unwrap().push(r.ctx.generation_id.to_string());
                        got += 1;
                    }
                    if received.lock().unwrap().len() >= 10_000 { break; }
                    if got == 0 { std::thread::sleep(Duration::from_millis(1)); }
                }
            })
        };
        for h in handles { h.join().unwrap(); }
        consumer.join().unwrap();
        let all = received.lock().unwrap().clone();
        assert_eq!(all.len(), 10_000, "every record delivered");
        let uniq: HashSet<&String> = all.iter().collect();
        assert_eq!(uniq.len(), 10_000, "no record lost or duplicated");
    }

    /// Drop-on-full: no consumer ⇒ capacity records accepted, the rest DROPPED, no hang.
    #[test]
    fn ring_full_drops_and_moves_on() {
        let ring = Ring::new(8);
        let ctx = Arc::new(ReqCtx {
            model_id: "m".into(), topology: "single".into(),
            request_id: "r".into(), generation_id: "g".into(),
        });
        let mut accepted = 0;
        for i in 0..1000u32 {
            if ring.push(Record::new(&ctx, EVENT_DELTA, i, "x")) { accepted += 1; }
        }
        assert_eq!(accepted, 8, "exactly capacity accepted");
        assert_eq!(ring.dropped(), 992, "rest dropped, hot path moved on");
    }

    /// Inline/no-alloc law: typical delta bodies stay under INLINE_CAP (the >cap arm exists for
    /// rare flushes only).
    #[test]
    fn body_inline_for_typical_chunks() {
        let small = "{\"choices\":[{\"index\":0,\"delta\":{\"content\":\" hi\"},\"finish_reason\":null}]}";
        match Body::from_chunk(small) {
            Body::Inline { len, .. } => assert_eq!(len as usize, small.len()),
            Body::Shared(_) => panic!("typical delta must stay inline (no per-token alloc)"),
        }
        let big = "x".repeat(INLINE_CAP + 1);
        assert!(matches!(Body::from_chunk(&big), Body::Shared(_)));
    }

    /// CONTRACT: one batched ExportLogsServiceRequest; body = the EXACT forwarded chunk bytes;
    /// attributes model.id/topology/request.id/token.index/event/generation.id in order;
    /// timeUnixNano/observedTimeUnixNano as decimal strings; token.index as an OTLP int-string.
    #[test]
    fn contract_shape_and_same_info() {
        let (sink, req) = sink_and_req(true, "chatcmpl-abc");
        let chunk = "{\"id\":\"chatcmpl-abc\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}";
        req.start("{\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}");
        req.delta(chunk);
        req.end("{\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}");
        let rows = drain(&sink);
        assert_eq!(rows.len(), 3, "batched: 3 records in ONE request");
        let json = otlp_logs_json(&sink, &rows);
        // Reference decode (what the client's decoder does): serde into the OTLP JSON shape.
        #[derive(serde::Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct ExportLogs { resource_logs: Vec<ResourceLogs> }
        #[derive(serde::Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct ResourceLogs { scope_logs: Vec<ScopeLogs> }
        #[derive(serde::Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct ScopeLogs { log_records: Vec<LogRecordRef> }
        #[derive(serde::Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct LogRecordRef {
            time_unix_nano: String,
            observed_time_unix_nano: String,
            body: serde_json::Value,
            attributes: Vec<AttrRef>,
        }
        #[derive(serde::Deserialize)]
        struct AttrRef { key: String, value: serde_json::Value }
        let parsed: ExportLogs = serde_json::from_str(&json).expect("decodes as OTLP JSON");
        assert_eq!(parsed.resource_logs.len(), 1, "one resourceLogs");
        assert_eq!(parsed.resource_logs[0].scope_logs.len(), 1, "one scopeLogs");
        let recs = &parsed.resource_logs[0].scope_logs[0].log_records;
        assert_eq!(recs.len(), 3);
        // SAME-INFO: the delta record's body IS the forwarded SSE chunk bytes, byte-for-byte.
        assert_eq!(recs[1].body["stringValue"].as_str().unwrap(), chunk);
        // Fixed attribute order + values (positional per contract).
        let expect = [
            ("model.id", "test-model"), ("topology", "single"), ("request.id", "chatcmpl-abc"),
            ("token.index", "0"), ("event", "stream_start"), ("generation.id", "gen-test"),
        ];
        let a0 = recs[0].attributes.iter().map(|a| (a.key.as_str(),
            a.value["stringValue"].as_str().or(a.value["intValue"].as_str()).unwrap())).collect::<Vec<_>>();
        assert_eq!(a0, expect, "start record attributes");
        assert_eq!(recs[1].attributes[4].value["stringValue"], "stream_delta");
        assert_eq!(recs[1].attributes[3].value["intValue"], "1");
        assert_eq!(recs[2].attributes[4].value["stringValue"], "stream_end");
        assert_eq!(recs[2].attributes[3].value["intValue"], "2");
        // Timestamps: decimal strings that parse as u64 nanos, observed == event time at push.
        for r in recs {
            assert!(r.time_unix_nano.parse::<u64>().is_ok(), "timeUnixNano decimal string");
            assert_eq!(r.time_unix_nano, r.observed_time_unix_nano);
        }
        // The internal attrmap agrees with the decoded one (belt and braces).
        assert_eq!(attrmap(&rows[0])[4], ("event".into(), "stream_start".into()));
    }

    /// LIFECYCLE: start → n×delta → end per request; two concurrent requests stay independently
    /// keyed (no cross-talk); status carries model@topology with an empty request.id.
    #[test]
    fn lifecycle_and_concurrent_keys() {
        let (sink, ra) = sink_and_req(true, "chatcmpl-A");
        let rb = sink.open_request("chatcmpl-B", "gen-B");
        ra.start("role"); ra.delta("a1"); rb.start("role"); ra.delta("a2");
        rb.delta("b1"); ra.end("end-a"); rb.end("end-b");
        let rows = drain(&sink);
        assert_eq!(rows.len(), 7, "ra 4 (start+2d+end) + rb 3 (start+1d+end)");
        for (rid, n_deltas) in [("chatcmpl-A", 2usize), ("chatcmpl-B", 1)] {
            let mine: Vec<&Record> = rows.iter().filter(|r| r.ctx.request_id.as_ref() == rid).collect();
            assert_eq!(mine.len(), n_deltas + 2);
            let events: Vec<u8> = mine.iter().map(|r| r.event).collect();
            assert_eq!(events.first(), Some(&EVENT_START), "{rid} starts first");
            assert_eq!(events.last(), Some(&EVENT_END), "{rid} ends last");
            assert!(events[1..events.len() - 1].iter().all(|&e| e == EVENT_DELTA));
            let idx: Vec<u32> = mine.iter().map(|r| r.token_index).collect();
            let mut sorted: Vec<u32> = idx.clone();
            sorted.sort_unstable();
            assert_eq!(idx, sorted, "{rid} token.index monotone");
        }
        // status: server-level, live model@topology, never collides with a request key.
        let status = Record::status(&Arc::new(ReqCtx {
            model_id: sink.model_id.clone(), topology: sink.topology.clone(),
            request_id: Arc::from(""), generation_id: Arc::from("server"),
        }));
        let a = attrmap(&status);
        assert_eq!(a[4], ("event".into(), "status".into()));
        assert_eq!(a[0], ("model.id".into(), "test-model".into()));
        assert_eq!(a[1], ("topology".into(), "single".into()));
        assert_eq!(a[2], ("request.id".into(), "".into()));
    }

    /// --otel-include-tokens off: per-token delta records VANISH (the hot path is one branch),
    /// start/end still flow, same-info untouched for what remains.
    #[test]
    fn include_tokens_off_drops_deltas_only() {
        let (sink, req) = sink_and_req(false, "chatcmpl-T");
        req.start("role");
        for _ in 0..50 { req.delta("{\"choices\":[...]}"); }
        req.end("end");
        let rows = drain(&sink);
        assert_eq!(rows.len(), 2, "start + end only");
        assert_eq!(rows[0].event, EVENT_START);
        assert_eq!(rows[1].event, EVENT_END);
    }

    /// topology_from_world: the generic single|tp2|tp4 vocabulary.
    #[test]
    fn topology_vocabulary() {
        assert_eq!(topology_from_world(None), "single");
        assert_eq!(topology_from_world(Some(2)), "tp2");
        assert_eq!(topology_from_world(Some(4)), "tp4");
    }

    // ── Session registry: the continuous-conversation key ────────────────────────────────
    //
    // per_msg mimics the server's canonical per-message JSON (ChatMessage::Serialize).

    fn msgs(msgs: &[(&str, &str)]) -> Vec<String> {
        msgs.iter()
            .map(|(r, c)| format!("{{\"role\":\"{r}\",\"content\":\"{c}\"}}"))
            .collect()
    }

    /// Turn N+1 (history + assistant reply + new user turn) continues turn N's session;
    /// an unrelated conversation gets its own id; both keep grouping across further turns.
    #[test]
    fn session_continuation_and_separation() {
        let reg = SessionRegistry::new();
        let t1 = msgs(&[("user", "hi")]);
        let s1 = reg.resolve(None, &t1);
        // Turn 2 = turn 1's messages + assistant reply + new user turn.
        let mut t2 = t1.clone();
        t2.push("{\"role\":\"assistant\",\"content\":\"hello\"}".into());
        t2.push("{\"role\":\"user\",\"content\":\"bye\"}".into());
        let s2 = reg.resolve(None, &t2);
        assert_eq!(s1, s2, "follow-up turn MUST land in the same session");
        // Turn 3 keeps the id.
        let mut t3 = t2.clone();
        t3.push("{\"role\":\"assistant\",\"content\":\"cya\"}".into());
        t3.push("{\"role\":\"user\",\"content\":\"?\"}".into());
        assert_eq!(reg.resolve(None, &t3), s1, "turn 3 continues the session");
        // An unrelated conversation is a DIFFERENT session, and stays itself on its turn 2.
        let u1 = msgs(&[("user", "completely different")]);
        let u_s1 = reg.resolve(None, &u1);
        assert_ne!(u_s1, s1);
        let mut u2 = u1.clone();
        u2.push("{\"role\":\"assistant\",\"content\":\"ok\"}".into());
        u2.push("{\"role\":\"user\",\"content\":\"more\"}".into());
        assert_eq!(reg.resolve(None, &u2), u_s1, "unrelated session groups internally");
        assert_eq!(reg.resolve(None, &t3), s1, "sessions do not cross-contaminate");
    }

    /// Explicit client key (X-Session-Id / metadata) wins, and a following BARE turn still
    /// groups (the registry learned the fingerprint from the keyed turn).
    #[test]
    fn session_explicit_key_wins_and_learns() {
        let reg = SessionRegistry::new();
        let t1 = msgs(&[("user", "keyed")]);
        let s1 = reg.resolve(Some("my-session-42".into()), &t1);
        assert_eq!(s1, "my-session-42");
        let mut t2 = t1.clone();
        t2.push("{\"role\":\"assistant\",\"content\":\"yes\"}".into());
        t2.push("{\"role\":\"user\",\"content\":\"again\"}".into());
        assert_eq!(reg.resolve(None, &t2), "my-session-42", "bare follow-up continues");
        // A different explicit key on a fresh conversation does not join.
        let other = reg.resolve(Some("other".into()), &msgs(&[("user", "keyed")]));
        assert_eq!(other, "other");
        assert_ne!(other, s1);
    }

    /// A bare RETRY (identical messages) lands in the same session; a truly different array
    /// (regenerated history) mints a new one — with the header as the exact-control escape.
    #[test]
    fn session_retry_same_regenerated_new() {
        let reg = SessionRegistry::new();
        let t = msgs(&[("user", "q"), ("assistant", "a"), ("user", "next")]);
        let s1 = reg.resolve(None, &t);
        assert_eq!(reg.resolve(None, &t), s1, "identical retry = same session");
        let mut regen = t[..2].to_vec();
        regen.push("{\"role\":\"user\",\"content\":\"next (edited)\"}".into());
        let s2 = reg.resolve(None, &regen);
        assert_ne!(s1, s2, "regenerated history is a different session (inferred)");
        assert_eq!(reg.resolve(Some(s1.clone()), &regen), s1, "header forces continuity");
    }

    /// Endpoint parsing: http only, default OTLP port, bracketed IPv6 tolerated.
    #[test]
    fn endpoint_hostport() {
        let mk = |ep: &str| OtelConfig {
            endpoint: ep.into(), batch_size: 1, batch_interval_ms: 1,
            include_tokens: true, model_id: None, topology: None,
        };
        assert_eq!(mk("http://127.0.0.1:4318").hostport().unwrap(), ("127.0.0.1".into(), 4318));
        assert_eq!(mk("http://otel.lan").hostport().unwrap(), ("otel.lan".into(), 4318));
        assert_eq!(mk("http://[::1]:9/").hostport().unwrap(), ("[::1]:9".into(), 4318));
        assert!(mk("https://secure").hostport().is_err(), "no TLS in the dependency-free client");
    }
}
