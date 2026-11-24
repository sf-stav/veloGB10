//! Special token ids of the V4.1 tokenizer, the stop-token union (AGENTS §3) and image span expansion.
//!
//! The ids are the release's own (`tokenizer.json`, `config.json`); `StopTokens::from_model_dir` reads
//! them from the model so a different pack of the same model is followed, and the constants are
//! asserted against the real files in the tests.

use std::path::Path;

pub const BOS_ID: u32 = 0;
pub const EOS_ID: u32 = 1;
pub const THINK_START_ID: u32 = 128821;
pub const THINK_END_ID: u32 = 128822;
pub const DSML_ID: u32 = 128825;
pub const IMAGE_PLACEHOLDER_ID: u32 = 129264;

/// The union of the end-of-sequence ids: `config.json` `eos_token_id` (int or list),
/// `generation_config.json` the same, and the `tokenizer_config.json` `eos_token` content resolved
/// through the tokenizer's added tokens. Sorted, deduplicated.
pub fn stop_token_ids(model_dir: &Path) -> Vec<u32> {
    let mut ids: Vec<u32> = Vec::new();
    let read = |f: &str| -> Option<serde_json::Value> {
        serde_json::from_str(&std::fs::read_to_string(model_dir.join(f)).ok()?).ok()
    };
    let mut push_val = |v: &serde_json::Value| match v {
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_u64() {
                ids.push(i as u32)
            }
        }
        serde_json::Value::Array(a) => {
            for x in a {
                if let Some(i) = x.as_u64() {
                    ids.push(i as u32)
                }
            }
        }
        _ => {}
    };
    for f in ["config.json", "generation_config.json"] {
        if let Some(c) = read(f) {
            if let Some(e) = c.get("eos_token_id") {
                push_val(e);
            }
        }
    }
    let eos_text = read("tokenizer_config.json").and_then(|c| match c.get("eos_token")? {
        serde_json::Value::String(s) => Some(s.clone()),
        o => o.get("content")?.as_str().map(str::to_string),
    });
    if let (Some(text), Some(tok)) = (eos_text, read("tokenizer.json")) {
        if let Some(added) = tok.get("added_tokens").and_then(|a| a.as_array()) {
            for t in added {
                if t.get("content").and_then(|c| c.as_str()) == Some(text.as_str()) {
                    if let Some(i) = t.get("id").and_then(|i| i.as_u64()) {
                        ids.push(i as u32);
                    }
                }
            }
        }
    }
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// Positions of the image placeholder in a tokenized prompt (one per image, in prompt order).
pub fn placeholder_positions(ids: &[u32], image_token_id: u32) -> Vec<usize> {
    ids.iter().enumerate().filter(|(_, t)| **t == image_token_id).map(|(i, _)| i).collect()
}

/// Expand each placeholder into `span_lens[k]` copies of the image token (the reference
/// `prepare_vl_inputs` does the same; the token TYPES of a span come from the vision leg).
/// Returns the new ids and, per image, the start offset of its span. `Err` when the counts differ.
pub fn expand_image_spans(ids: &[u32], image_token_id: u32, span_lens: &[usize]) -> Result<(Vec<u32>, Vec<usize>), String> {
    let n = ids.iter().filter(|t| **t == image_token_id).count();
    if n != span_lens.len() {
        return Err(format!("Found {n} image tokens but got {} images", span_lens.len()));
    }
    let mut out = Vec::with_capacity(ids.len() + span_lens.iter().sum::<usize>());
    let mut starts = Vec::with_capacity(n);
    let mut k = 0;
    for &t in ids {
        if t == image_token_id {
            starts.push(out.len());
            out.extend(std::iter::repeat(image_token_id).take(span_lens[k]));
            k += 1;
        } else {
            out.push(t);
        }
    }
    Ok((out, starts))
}
