//! THINKBUDGET1: a per-request thinking budget, compatible with vLLM's `thinking_token_budget`.
//!
//! A request may cap how many tokens the model spends inside its reasoning block. When the cap is
//! reached, the server ends the reasoning itself by emitting the model's end-of-thinking marker
//! (`</think>` for the Qwen family) and the model then writes its answer. Opt-in: a request
//! without the field is served exactly as before.
//!
//! The semantics follow vLLM 0.29 (`vllm/v1/sample/thinking_budget_state.py`, `sampling_params.py`):
//! - field `thinking_token_budget`: a non-negative integer, or `-1` = unlimited; anything else is a
//!   400 with vLLM's message;
//! - reasoning tokens are counted from the LAST start marker: tokens after it in the prompt count
//!   (the thinking-on generation prompt `<think>\n` contributes one), then every generated token
//!   until an end marker;
//! - the token that would be reasoning token number `budget + 1` is replaced by the end sequence
//!   (one token per step for a multi-token sequence); so exactly `budget` reasoning tokens precede
//!   the forced end;
//! - a natural end marker before the budget leaves the request alone; a later start marker opens a
//!   new reasoning block with a fresh count;
//! - a prompt that is not inside a reasoning block (thinking off) is never touched.
//! vLLM forces through the logits; the schedulers here force host-side (the speculative depth is
//! clamped at the budget, and the boundary token of a plain step is replaced before the model
//! consumes it), which emits the same token sequence.

/// vLLM's validation message, verbatim.
pub const INVALID: &str = "`thinking_token_budget` must be a non-negative integer or -1 for unlimited.";

/// Deserialize `thinking_token_budget` with vLLM's rule: absent / null / -1 = None (unlimited);
/// an integer >= 0 = Some(n); a bool, a float, a string, or an integer below -1 = error.
pub fn deserialize<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<usize>, D::Error> {
    use serde::Deserialize;
    let v = Option::<serde_json::Value>::deserialize(d)?;
    parse_value(v.as_ref()).map_err(serde::de::Error::custom)
}

/// The rule behind `deserialize`, on a JSON value (None = field absent or null).
pub fn parse_value(v: Option<&serde_json::Value>) -> Result<Option<usize>, &'static str> {
    let Some(v) = v else { return Ok(None) };
    if v.is_null() {
        return Ok(None);
    }
    let serde_json::Value::Number(n) = v else { return Err(INVALID) };
    if let Some(u) = n.as_u64() {
        return usize::try_from(u).map(Some).map_err(|_| INVALID);
    }
    match n.as_i64() {
        Some(-1) => Ok(None),
        _ => Err(INVALID), // a float, or an integer below -1
    }
}

/// The reasoning markers of a model: the start sequence that opens a reasoning block, the end
/// sequence that closes it naturally, and the sequence the server forces at the budget (vLLM's
/// `reasoning_end_str`: the end marker, optionally preceded by a transition phrase).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Markers {
    pub start: Vec<u32>,
    pub end: Vec<u32>,
    pub force: Vec<u32>,
}

/// vLLM's `--reasoning-config` (the keys the thinking budget reads): `reasoning_start_str` and
/// `reasoning_end_str`, both optional; absent = the model's `<think>` / `</think>`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReasoningCfg {
    pub start_str: Option<String>,
    pub end_str: Option<String>,
}

impl ReasoningCfg {
    /// Parse the flag's JSON object. Unknown keys are ignored but reported (`notes`), so a vLLM
    /// launch line that also sets `reasoning_parser` still boots.
    pub fn parse(json: &str) -> Result<(Self, Vec<String>), String> {
        let v: serde_json::Value = serde_json::from_str(json).map_err(|e| format!("not valid JSON ({e})"))?;
        let obj = v.as_object().ok_or_else(|| "must be a JSON object".to_string())?;
        let mut cfg = Self::default();
        let mut notes = Vec::new();
        for (k, val) in obj {
            let text = || val.as_str().map(str::to_string).ok_or_else(|| format!("`{k}` must be a string"));
            match k.as_str() {
                "reasoning_start_str" => cfg.start_str = Some(text()?).filter(|s| !s.is_empty()),
                "reasoning_end_str" => cfg.end_str = Some(text()?).filter(|s| !s.is_empty()),
                other => notes.push(format!("`{other}` is not used here (the reasoning parser is chosen by the model family)")),
            }
        }
        Ok((cfg, notes))
    }
}

/// THIS server's `--reasoning-config`, parsed once. A value that does not parse falls back to the
/// defaults with one clear line (AGENTS 1b-bis.8): a budgeted request still ends at `</think>`.
pub fn reasoning_cfg() -> &'static ReasoningCfg {
    static CFG: std::sync::OnceLock<ReasoningCfg> = std::sync::OnceLock::new();
    CFG.get_or_init(|| {
        let Ok(raw) = crate::opts::var(crate::opt!("reasoning-config")) else { return ReasoningCfg::default() };
        if raw.trim().is_empty() {
            return ReasoningCfg::default();
        }
        match ReasoningCfg::parse(&raw) {
            Ok((cfg, notes)) => {
                for n in notes {
                    crate::reprintln!("[serve] --reasoning-config: {n}");
                }
                cfg
            }
            Err(e) => {
                crate::reprintln!("[serve] --reasoning-config {e}: using the defaults (<think> / </think>) for thinking_token_budget");
                ReasoningCfg::default()
            }
        }
    })
}

impl Markers {
    /// The markers of a tokenizer whose vocabulary has `<think>` and `</think>` as SINGLE tokens
    /// (the Qwen 3.5 / 3.6 / 3.8 family). `encode` = the tokenizer's encode without special-token
    /// insertion. None = the model has no such markers (the budget cannot apply: a multi-piece
    /// encoding of the literal text would match ordinary prose). `cfg` may override the start
    /// sequence and the forced text; a forced text that does not contain the end marker would never
    /// close the reasoning, so it falls back to the bare marker (said once).
    pub fn from_encoder(encode: impl Fn(&str) -> Option<Vec<u32>>, cfg: &ReasoningCfg) -> Option<Self> {
        let think = encode("<think>")?;
        let end = encode("</think>")?;
        if think.len() != 1 || end.len() != 1 || think == end {
            return None;
        }
        let start = match cfg.start_str.as_deref() {
            Some(s) => encode(s).filter(|v| !v.is_empty()).unwrap_or(think),
            None => think,
        };
        let force = match cfg.end_str.as_deref() {
            Some(s) => match encode(s) {
                Some(v) if v.windows(end.len()).any(|w| w == end.as_slice()) => v,
                _ => {
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| crate::reprintln!("[serve] --reasoning-config: reasoning_end_str does not contain the model's \
</think> marker, so it would never close the reasoning; the budget forces </think> alone"));
                    end.clone()
                }
            },
            None => end.clone(),
        };
        Some(Self { start, end, force })
    }
}

/// One request's budget state. Every input is replicated state (the prompt, the emitted tokens,
/// the budget), so every TP rank that applies it at the same emission points reaches the same
/// decisions.
#[derive(Clone, Debug)]
pub struct ThinkBudget {
    budget: usize,
    markers: Markers,
    in_think: bool,
    /// reasoning tokens counted in the current block (prompt-side tokens after the start marker
    /// included)
    count: usize,
    /// Some(i) = forcing the end sequence; `i` = the index of the next forced token
    forcing: Option<usize>,
    /// the last few emitted tokens, for multi-token marker matching
    recent: Vec<u32>,
    /// end sequences forced so far (statistics and the per-request log)
    pub forced: usize,
    /// the reasoning-token count at which the last forced end began (log only)
    pub forced_at: Option<usize>,
}

impl ThinkBudget {
    /// The state at the start of generation: inside a reasoning block iff the prompt's last start
    /// marker comes after its last end marker; the count = the prompt tokens after that start.
    pub fn new(budget: usize, markers: Markers, prompt: &[u32]) -> Self {
        let last_start = rfind(prompt, &markers.start);
        let last_end = rfind(prompt, &markers.end);
        let (in_think, count) = match (last_start, last_end) {
            (Some(s), e) if e.map_or(true, |e| s > e) => (true, prompt.len() - (s + markers.start.len())),
            _ => (false, 0),
        };
        let keep = markers.start.len().max(markers.end.len());
        let recent = prompt[prompt.len().saturating_sub(keep)..].to_vec();
        Self { budget, markers, in_think, count, forcing: None, recent, forced: 0, forced_at: None }
    }

    /// The tokens this request may still emit before the next one must be forced: Some(0) = the
    /// next emitted token is forced (a speculative round must not run past it); None = the budget
    /// does not constrain the next round (not inside a reasoning block).
    pub fn room(&self) -> Option<usize> {
        if self.forcing.is_some() {
            Some(0)
        } else if self.in_think {
            Some(self.budget.saturating_sub(self.count))
        } else {
            None
        }
    }

    /// Feed the token the model produced at the next position; returns the token to emit and
    /// whether it was forced. Call it for EVERY emitted token, in order.
    pub fn filter(&mut self, t: u32) -> (u32, bool) {
        if self.forcing.is_none() && self.in_think && self.count >= self.budget {
            self.forcing = Some(0);
            self.forced_at = Some(self.count);
        }
        if let Some(i) = self.forcing {
            let f = self.markers.force[i];
            self.push_recent(f);
            if i + 1 >= self.markers.force.len() {
                self.forcing = None;
                self.in_think = false;
                self.count = 0;
                self.forced += 1;
            } else {
                self.forcing = Some(i + 1);
            }
            return (f, true);
        }
        self.push_recent(t);
        if self.in_think {
            if self.recent.ends_with(&self.markers.end) {
                self.in_think = false; // a natural end: nothing to force
                self.count = 0;
            } else {
                self.count += 1;
            }
        } else if self.recent.ends_with(&self.markers.start) {
            self.in_think = true; // a new reasoning block: a fresh budget
            self.count = 0;
        }
        (t, false)
    }

    pub fn budget(&self) -> usize {
        self.budget
    }

    pub fn in_think(&self) -> bool {
        self.in_think
    }

    fn push_recent(&mut self, t: u32) {
        self.recent.push(t);
        let keep = self.markers.start.len().max(self.markers.end.len());
        if self.recent.len() > keep {
            let cut = self.recent.len() - keep;
            self.recent.drain(..cut);
        }
    }
}

/// A serving path that does not enforce the budget: when a request carries one, it is served
/// WITHOUT it — counted (velogb10_thinking_budget_requests_total{disposition="unsupported"}) and
/// said once per boot (AGENTS 1b-bis.8: fall back, say so, never refuse). `path` names the path.
pub fn note_unsupported(budget: Option<usize>, path: &'static str) {
    if budget.is_none() {
        return;
    }
    crate::metrics::think_budget_unsupported();
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| crate::reprintln!("[serve] thinking_token_budget is not enforced on the {path} serving path yet: \
requests that send it run without a thinking budget (counted in velogb10_thinking_budget_requests_total; this line prints once per boot)"));
}

/// Clamp a round's emittable token count by the budget's room (None = unconstrained).
pub fn clamp_room(remaining: usize, room: Option<usize>) -> usize {
    room.map_or(remaining, |r| remaining.min(r))
}

fn rfind(hay: &[u32], needle: &[u32]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    (0..=hay.len() - needle.len()).rev().find(|&i| &hay[i..i + needle.len()] == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: u32 = 248068; // <think>
    const E: u32 = 248069; // </think>
    const NL: u32 = 198;

    fn qwen() -> Markers {
        Markers { start: vec![S], end: vec![E], force: vec![E] }
    }

    /// The Qwen3.8 thinking-on generation prompt ends `<think>` `\n`: inside, one token counted.
    fn think_prompt() -> Vec<u32> {
        vec![1, 2, 3, S, NL]
    }

    /// Run generation: the model would emit `model` (index i at step i); returns what is emitted.
    fn run(tb: &mut ThinkBudget, model: &[u32]) -> Vec<u32> {
        model.iter().map(|&t| tb.filter(t).0).collect()
    }

    #[test]
    fn parse_follows_vllm() {
        use serde_json::json;
        assert_eq!(parse_value(None), Ok(None));
        assert_eq!(parse_value(Some(&json!(null))), Ok(None));
        assert_eq!(parse_value(Some(&json!(-1))), Ok(None));
        assert_eq!(parse_value(Some(&json!(0))), Ok(Some(0)));
        assert_eq!(parse_value(Some(&json!(1024))), Ok(Some(1024)));
        for bad in [json!(-2), json!(-100), json!(1.5), json!(2.0), json!(true), json!("5"), json!([1])] {
            assert_eq!(parse_value(Some(&bad)), Err(INVALID), "{bad}");
        }
    }

    #[test]
    fn markers_need_single_tokens() {
        let enc = |s: &str| Some(match s { "<think>" => vec![S], "</think>" => vec![E], _ => vec![] });
        assert_eq!(Markers::from_encoder(enc, &ReasoningCfg::default()), Some(qwen()));
        // a model without the added tokens: the literal text splits into BPE pieces -> no markers
        let pieces = |s: &str| Some(if s.starts_with("</") { vec![27, 14, 26266, 29] } else { vec![27, 26266, 29] });
        assert_eq!(Markers::from_encoder(pieces, &ReasoningCfg::default()), None);
    }

    #[test]
    fn exactly_budget_reasoning_tokens_then_the_forced_end() {
        // budget 4: the prompt's "\n" is reasoning token 1, so 3 generated ones, then </think>
        let mut tb = ThinkBudget::new(4, qwen(), &think_prompt());
        assert_eq!(tb.room(), Some(3));
        let out = run(&mut tb, &[10, 11, 12, 13, 14, 15]);
        assert_eq!(out, vec![10, 11, 12, E, 14, 15], "token 4 of the model's stream is replaced by </think>");
        assert_eq!((tb.forced, tb.forced_at, tb.in_think()), (1, Some(4), false));
        assert_eq!(tb.room(), None, "after the end the answer is unconstrained");
    }

    #[test]
    fn budget_zero_and_one_force_the_first_token() {
        // vLLM: budget - prompt-side count <= 0 -> the first generated token is the end
        for b in [0, 1] {
            let mut tb = ThinkBudget::new(b, qwen(), &think_prompt());
            assert_eq!(tb.room(), Some(0), "budget {b}");
            assert_eq!(run(&mut tb, &[10, 11]), vec![E, 11], "budget {b}");
        }
    }

    #[test]
    fn a_natural_end_before_the_budget_is_left_alone() {
        let mut tb = ThinkBudget::new(100, qwen(), &think_prompt());
        assert_eq!(run(&mut tb, &[10, E, 20, 21]), vec![10, E, 20, 21]);
        assert_eq!((tb.forced, tb.in_think(), tb.room()), (0, false, None));
    }

    #[test]
    fn thinking_off_prompts_are_never_touched() {
        // the thinking-off template: <think>\n\n</think>\n\n
        let mut tb = ThinkBudget::new(0, qwen(), &[1, S, 271, E, 271]);
        assert_eq!(tb.room(), None);
        assert_eq!(run(&mut tb, &[10, 11, 12]), vec![10, 11, 12]);
        // a raw completion prompt with no markers at all
        let mut tb = ThinkBudget::new(0, qwen(), &[1, 2, 3]);
        assert_eq!(run(&mut tb, &[10, 11]), vec![10, 11]);
    }

    #[test]
    fn a_new_block_after_the_end_gets_a_fresh_budget() {
        let mut tb = ThinkBudget::new(2, qwen(), &think_prompt()); // 1 counted in the prompt
        let out = run(&mut tb, &[10, 99, 20, S, 30, 31, 32]);
        assert_eq!(out, vec![10, E, 20, S, 30, 31, E], "second block: 2 reasoning tokens, then forced");
        assert_eq!(tb.forced, 2);
    }

    #[test]
    fn multi_token_end_sequences_are_forced_one_per_step() {
        let m = Markers { start: vec![7, 8], end: vec![5, 6], force: vec![5, 6] };
        let mut tb = ThinkBudget::new(2, m, &[1, 7, 8]);
        assert_eq!(tb.room(), Some(2));
        let out: Vec<u32> = [10, 11, 12, 13, 14].iter().map(|&t| tb.filter(t).0).collect();
        assert_eq!(out, vec![10, 11, 5, 6, 14]);
        assert_eq!(tb.room(), None);
        // a natural two-token end
        let mut tb = ThinkBudget::new(50, Markers { start: vec![7, 8], end: vec![5, 6], force: vec![5, 6] }, &[1, 7, 8]);
        let out: Vec<u32> = [10, 5, 6, 20].iter().map(|&t| tb.filter(t).0).collect();
        assert_eq!((out, tb.forced, tb.in_think()), (vec![10, 5, 6, 20], 0, false));
    }

    #[test]
    fn reasoning_config_follows_vllm() {
        let (c, notes) = ReasoningCfg::parse(r#"{"reasoning_end_str": "\n\nOK, answering now.\n</think>\n\n", "reasoning_parser": "qwen3"}"#).unwrap();
        assert_eq!(c.end_str.as_deref(), Some("\n\nOK, answering now.\n</think>\n\n"));
        assert_eq!((c.start_str, notes.len()), (None, 1), "reasoning_parser is reported, not fatal");
        assert!(ReasoningCfg::parse("[1]").is_err() && ReasoningCfg::parse("{").is_err());
        assert!(ReasoningCfg::parse(r#"{"reasoning_end_str": 5}"#).is_err());
        // a transition phrase is forced token by token, then the reasoning is closed
        let enc = |s: &str| Some(match s { "<think>" => vec![S], "</think>" => vec![E], "Done.</think>" => vec![40, 41, E], "no marker" => vec![40, 41], _ => vec![] });
        let cfg = ReasoningCfg { start_str: None, end_str: Some("Done.</think>".into()) };
        let m = Markers::from_encoder(enc, &cfg).unwrap();
        assert_eq!(m.force, vec![40, 41, E]);
        let mut tb = ThinkBudget::new(3, m, &think_prompt());
        assert_eq!(run(&mut tb, &[10, 11, 12, 13, 14, 15]), vec![10, 11, 40, 41, E, 15]);
        assert_eq!((tb.forced, tb.in_think()), (1, false));
        // a forced text without </think> would never close the reasoning: the bare marker instead
        let cfg = ReasoningCfg { start_str: None, end_str: Some("no marker".into()) };
        assert_eq!(Markers::from_encoder(enc, &cfg).unwrap().force, vec![E]);
    }

    #[test]
    fn room_matches_the_round_clamp() {
        // a scheduler that emits at most room() tokens per round never forces mid-round
        let mut tb = ThinkBudget::new(10, qwen(), &think_prompt());
        let mut emitted = Vec::new();
        let mut step = 0u32;
        while emitted.len() < 14 {
            let room = clamp_room(64, tb.room());
            let width = room.clamp(1, 4); // room 0 -> a plain step of 1
            for j in 0..width {
                let (t, forced) = tb.filter(1000 + step);
                assert!(!forced || j == 0, "a forced token is always the first of its round");
                emitted.push(t);
                step += 1;
            }
        }
        let end_at = emitted.iter().position(|&t| t == E).unwrap();
        assert_eq!(end_at, 9, "1 prompt-side + 9 generated reasoning tokens = the budget of 10");
    }
}
