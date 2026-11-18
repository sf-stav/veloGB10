//! Small request helpers shared by the handlers: generation-room budget, human request log line, partial-marker
//! overlap / hold-back for streamed text (`GrowFind`, `hold_back_end`), token dump, off-thread encode and `log_generation`.
//! Siblings: `chat_stream.rs` and `chat.rs` call these; `mod.rs` has the map.

use super::*;

/// Generation room left in the KV cache after the prompt and the scheduler's decode reserve.
/// None = the prompt (plus reserve) does not fit at all. (PR #4 — mirrors `batch::admit`.)
pub(super) fn generation_room(max_seq_len: usize, prompt_len: usize, decode_headroom: usize) -> Option<usize> {
    let used = prompt_len.checked_add(decode_headroom)?;
    max_seq_len.checked_sub(used).filter(|&room| room > 0)
}

/// (The think close marker is resolved per-request from the model's vocab — see
/// QwenTokenizer::think_tags. Qwen: `</think>`; hy_v3: `</think:opensource>`.)

/// Longest suffix of `s` that is a proper (partial) prefix of `marker` — text that could be the start
/// of the marker arriving across decode chunks, and so must be held back rather than forwarded.
/// `--output-prompts [cap]` — log the chat-completion call in human-readable form: effective
/// parameters, one line per message turn, and the exact rendered prompt the model sees
/// (excerpt up to `cap` chars; --dump-prompt=1 still writes the full string to /tmp
/// for diffing). Diagnostic output only; nothing here touches the serving path.
#[allow(clippy::too_many_arguments)]
pub(super) fn log_request_human(
    req: &ChatCompletionRequest,
    effort: Option<&str>,
    prompt: &str,
    prompt_tokens: usize,
    cap: usize,
    model_name: &str,
    render_ms: f64,
) {
    let opt_f32 = |v: &Option<f32>| v.map(|x| x.to_string()).unwrap_or_else(|| "default".into());
    eprintln!("[prompt] ══ chat completion request ({model_name}) ════════════════════════════");
    eprintln!("  stream={}  max_tokens={}  seed={}",
        req.stream,
        req.max_tokens.map(|t| t.to_string()).unwrap_or_else(|| "server-default".into()),
        req.seed.map(|s| s.to_string()).unwrap_or_else(|| "-".into()));
    eprintln!("  temperature={}  top_p={}  top_k={}", opt_f32(&req.temperature), opt_f32(&req.top_p),
        req.top_k.map(|k| k.to_string()).unwrap_or_else(|| "default".into()));
    eprintln!("  penalties: repetition={}  presence={}  frequency={}",
        opt_f32(&req.repetition_penalty), opt_f32(&req.presence_penalty), opt_f32(&req.frequency_penalty));
    eprintln!("  reasoning_effort={} (effective: {})  stop={:?}  include_usage={}",
        req.reasoning_effort.as_deref().unwrap_or("-"),
        effort.unwrap_or("template-default"),
        req.stop,
        req.stream_options.as_ref().map(|s| s.include_usage).unwrap_or(false));
    match &req.tools {
        Some(ts) if !ts.is_empty() => {
            let names: Vec<&str> = ts.iter().filter_map(|t| t.get("function")
                .and_then(|f| f.get("name")).and_then(|n| n.as_str())).collect();
            eprintln!("  tools ({}): {}", ts.len(), names.join(", "));
        }
        _ => eprintln!("  tools: none"),
    }
    eprintln!("  messages ({}):", req.messages.len());
    for (i, m) in req.messages.iter().enumerate() {
        let mut line = format!("    {}. {:9}", i + 1, m.role);
        if let Some(c) = &m.content {
            let flat: String = c.chars().map(|ch| if ch == '\n' { '⏎' } else { ch }).collect();
            let n = flat.chars().count();
            let head: String = flat.chars().take(160).collect();
            line.push_str(&format!(" ({n} ch): {head}{}", if n > 160 { " …" } else { "" }));
        }
        if let Some(tc) = &m.tool_calls {
            let names: Vec<&str> = tc.iter().map(|c| c.function.name.as_str()).collect();
            line.push_str(&format!("  [tool_calls: {}]", names.join(", ")));
        }
        if !m.images.is_empty() { line.push_str(&format!("  [{} image(s)]", m.images.len())); }
        if let Some(id) = &m.tool_call_id { line.push_str(&format!("  [result of {id}]")); }
        eprintln!("{line}");
    }
    let total = prompt.chars().count();
    let trunc = cap.min(total);
    eprintln!("[prompt] rendered prompt: {prompt_tokens} tokens, {total} chars ({render_ms:.1} ms render):");
    let head: String = prompt.chars().take(trunc).collect();
    for l in head.lines() { eprintln!("    | {l}"); }
    if total > trunc {
        eprintln!("    … (+{} more chars of {total} — full dump: --dump-prompt=1)", total - trunc);
    }
    eprintln!("[prompt] ══════════════════════════════════════════════════════════════════");
}

/// The longest proper prefix of `marker` that `s` ENDS with, in bytes (0 = none). A marker may be a request's
/// stop string, so only whole characters of it count as a prefix.
///
/// LINEAR (S-B9-REL-INTEG B1): Knuth-Morris-Pratt over the last `marker.len() - 1` bytes of `s` (nothing
/// earlier can be part of a PROPER prefix of the marker), with the failure table of the prefix of `marker` that
/// tail can reach. The old loop tried every length k from the top and compared k bytes for each: quadratic
/// in the stop string, which a client chooses (1.7 ms per call at 10 KB, 308 ms at 200 KB, per token and per
/// stop string, on a tokio worker). The cost is now O(min(|s|, |marker| - 1)) with no cap and no behaviour
/// change; `partial_overlap_steps` returns the step count the tests assert on.
pub(super) fn partial_overlap(s: &str, marker: &str) -> usize { partial_overlap_steps(s, marker).0 }

pub(super) fn partial_overlap_steps(s: &str, marker: &str) -> (usize, u64) {
    let (sb, mb) = (s.as_bytes(), marker.as_bytes());
    if mb.len() < 2 || sb.is_empty() { return (0, 0); }
    let t = &sb[sb.len().saturating_sub(mb.len() - 1)..]; // <= marker.len() - 1 bytes
    let l = t.len();                                       // the automaton never passes state l
    let mut steps = 0u64;
    // failure function of marker[..l]
    let mut fail = vec![0u32; l];
    let mut k = 0usize;
    for i in 1..l {
        while k > 0 && mb[i] != mb[k] { k = fail[k - 1] as usize; steps += 1; }
        if mb[i] == mb[k] { k += 1; }
        fail[i] = k as u32;
        steps += 1;
    }
    // match marker's prefixes against the tail: the final state is the longest prefix that is a suffix of t
    let mut q = 0usize;
    for &b in t {
        while q > 0 && mb[q] != b { q = fail[q - 1] as usize; steps += 1; }
        if mb[q] == b { q += 1; }
        steps += 1;
    }
    // q <= l < marker.len(): a proper prefix. (Over valid UTF-8 the borders are whole characters of the
    // marker; the walk is the guard for a tail that starts inside a character.)
    while q > 0 && !marker.is_char_boundary(q) { q = fail[q - 1] as usize; steps += 1; }
    (q, steps)
}

pub(super) fn partial_think_overlap(s: &str, marker: &str) -> usize { partial_overlap(s, marker) }

/// S3+S4 (REL_V0_7_3 review): the ONE hold-back decision, called by EVERY SSE emit site
/// (the same-chunk think-close branch, the steady-state content split, the reasoning split).
/// `tool_hold` is None exactly when tool-call parsing is off (a keep-tools "none" request):
/// a stray `<tool_call` then streams as plain content — nothing is held back for tools.
/// `tool_hit` is the caller's scanner result for the tool marker (a GrowFind hit, or a plain
/// `str::find` made absolute; None both when the scanner found nothing and when tools are
/// off). `emit_limit` is the buffer end (`acc.len()`, or `safe` for the reasoning region);
/// `region` is the un-emitted tail the partial-marker checks run over; `partial_markers`
/// lists markers whose SPLIT tail must not leak across an emit boundary (the think-open tag
/// for the content sites; none for reasoning).
pub(super) fn hold_back_end(
    tool_hold: Option<&'static str>,
    tool_hit: Option<usize>,
    emit_limit: usize,
    region: &str,
    partial_markers: &[&str],
) -> usize {
    let partial = partial_markers.iter().map(|&m| partial_overlap(region, m)).max().unwrap_or(0);
    match tool_hold {
        None => emit_limit - partial,
        Some(m) => match tool_hit {
            Some(i) => i, // a call has started: emit nothing more
            None => emit_limit - partial_overlap(region, m).max(partial),
        },
    }
}

/// WP02: incremental first-occurrence search over a buffer that only GROWS (the streaming `acc`).
/// `find(hay, base, needle)` returns exactly `hay[base..].find(needle).map(|i| base + i)` as long as
/// every `hay` is a prefix of the same growing buffer and `base` never decreases (a decrease falls
/// back to a full scan). It skips what an earlier miss already cleared — no occurrence lies wholly
/// inside [seen_base, seen_upto) — so a long stream costs O(new bytes) per token, not O(n).
#[derive(Default)]
pub(super) struct GrowFind {
    base: usize,
    upto: usize,
}

impl GrowFind {
    pub(super) fn find(&mut self, hay: &str, base: usize, needle: &str) -> Option<usize> {
        if needle.is_empty() {
            return Some(base); // str::find("") == Some(0)
        }
        let mut start = base;
        if base >= self.base {
            let known = self.upto.min(hay.len());
            let s = known.saturating_sub(needle.len() - 1);
            if s > start {
                start = s;
                while !hay.is_char_boundary(start) { start -= 1; } // base is a boundary: stops >= base
            }
        }
        let r = hay[start..].find(needle).map(|i| start + i);
        if r.is_none() && (base < self.base || hay.len() > self.upto) {
            self.base = base;
            self.upto = hay.len();
        }
        r
    }
}

/// Diagnostics-only (env `--dump-tokens=1`): one line per generation with the EXACT
/// generated token ids. Why it exists: a served-text comparison cannot distinguish "the same
/// tokens" from "different tokens that detokenize alike", and the bitwise-losslessness claim for a
/// speculative lane (AGENTS §2.6/§3, P14's `--spec-source dflash` A/B) is a statement about the
/// TOKEN sequence. Harness plumbing only — no response byte, no scheduler behaviour is touched.
/// (Twin of `--dump-prompt`, which does the same for the prompt ids.)
pub(super) fn dump_tokens(id: &str, tokens: &[u32]) {
    if crate::opts::var(crate::opt!("dump-tokens")).is_err() { return; }
    let ids: Vec<String> = tokens.iter().map(|t| t.to_string()).collect();
    crate::reprintln!("[gen-ids] id={id} n={} ids=[{}]", tokens.len(), ids.join(","));
}

/// PERFFIX1: tokenize a large text on Tokio's blocking pool instead of the server's single request thread (render +
/// encode measured ~1 ms per 1,000 prompt tokens: a 100K-token prompt held every other stream's SSE writes, /metrics
/// and /health for ~100 ms). Same tokenizer, same call, same tokens; texts under 64 KiB stay inline (a thread hop
/// would cost more than it saves).
pub(super) async fn encode_off_thread(tok: &Arc<QwenTokenizer>, text: &str, add_special: bool) -> anyhow::Result<Vec<u32>> {
    if text.len() < 64 * 1024 {
        return tok.encode(text, add_special);
    }
    let (tok, text) = (tok.clone(), text.to_owned());
    tokio::task::spawn_blocking(move || tok.encode(&text, add_special)).await
        .map_err(|e| anyhow::anyhow!("tokenizer task failed: {e}"))?
}

/// Phase-2 A3 observability: ONE line per generation carrying exactly the fields the Phase-1
/// ledger had to reconstruct — or could not recover at all — from the harness transcripts:
/// the final finish_reason, the terminal token id and whether it is a model stop token, and the
/// reasoning/content token split. `reasoning_tokens`/`content_tokens` are ENCODE-based (the two
/// text spans are re-tokenized), so their sum can differ from `completion_tokens` by
/// template/special-token effects; they are diagnostics, not billing. Log lines only: no
/// response byte, no sampling parameter and no scheduler behaviour is touched.
pub(super) fn log_generation(tok: &QwenTokenizer, stop_ids: &[u32], id: &str, finish: &str,
                  terminal: Option<u32>, prompt_tokens: usize, completion_tokens: usize,
                  reasoning: Option<&str>, content: &str, tool_calls: usize,
                  tools_offered: usize, effort: Option<&str>, presence_penalty: f32,
                  pp_source: &str, stop_kind: &str, stop_info: Option<(f32, u32, bool)>) {
    // PERFFIX1: the line below prints in debug mode only — skip the two re-tokenizations (up to the whole generation)
    // otherwise; they ran on the request thread after every request for a line nobody saw.
    if !crate::boot_report::debug() { return; }
    let count = |s: Option<&str>| s.map(|x| tok.encode(x, false).map(|v| v.len()).unwrap_or(0))
        .unwrap_or(0);
    let r_tok = count(reasoning);
    let c_tok = count(Some(content));
    let is_stop = terminal.map(|t| stop_ids.contains(&t)).unwrap_or(false);
    crate::dreprintln!("[gen] id={} finish={} terminal_tok={} is_stop_tok={} reasoning_tokens={} \
               content_tokens={} completion_tokens={} prompt_tokens={} tool_calls={} \
               tools_offered={} effort={} presence_penalty={} pp_source={} stop_kind={} {}",
        id, finish, terminal.map(|t| t.to_string()).unwrap_or_else(|| "none".into()), is_stop,
        r_tok, c_tok, completion_tokens, prompt_tokens, tool_calls, tools_offered,
        effort.unwrap_or("<template-default>"), presence_penalty, pp_source,
        stop_kind, stop_info_fields(stop_info));
}

