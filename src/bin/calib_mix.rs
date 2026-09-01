//! Build a deterministic GPTQ calibration JSONL with a measured fraction of
//! prompt-injection examples in the token prefix the quantizer actually reads.

use anyhow::{bail, Context, Result};
use gb10_inference::tokenizer::{ChatMessage, QwenTokenizer};
use serde_json::{json, Value};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Clone)]
struct Chunk {
    text: String,
    tokens_with_separator: usize,
    injection: bool,
}

struct Args {
    model_dir: PathBuf,
    input: PathBuf,
    injections: PathBuf,
    output: PathBuf,
    percent: f64,
    nsamples: usize,
    seqlen: usize,
    chunk_tokens: usize,
    reserve_sequences: usize,
}

fn value(args: &[String], flag: &str) -> Result<String> {
    let i = args
        .iter()
        .position(|arg| arg == flag)
        .with_context(|| format!("missing {flag}"))?;
    args.get(i + 1)
        .cloned()
        .with_context(|| format!("missing value after {flag}"))
}

fn optional<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    match args.iter().position(|arg| arg == flag) {
        Some(i) => args
            .get(i + 1)
            .with_context(|| format!("missing value after {flag}"))?
            .parse()
            .map_err(|e| anyhow::anyhow!("invalid {flag}: {e}")),
        None => Ok(default),
    }
}

fn parse_args() -> Result<Args> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        println!(
            "Usage: calib_mix --model-dir DIR --input corpus.jsonl --output mixed.jsonl\n\
             \nOptions:\n\
             \x20 --injections FILE       default: assets/calibration/prompt_injection.jsonl\n\
             \x20 --percent N             token share in the consumed prefix (default: 5)\n\
             \x20 --nsamples N            default: 512\n\
             \x20 --seqlen N              default: 2048\n\
             \x20 --chunk-tokens N         scheduling granularity (default: 512)\n\
             \x20 --reserve-sequences N    output margin after consumed prefix (default: 8)"
        );
        std::process::exit(0);
    }
    let injections = args
        .iter()
        .position(|arg| arg == "--injections")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/calibration/prompt_injection.jsonl")
        });
    let parsed = Args {
        model_dir: PathBuf::from(value(&args, "--model-dir")?),
        input: PathBuf::from(value(&args, "--input")?),
        output: PathBuf::from(value(&args, "--output")?),
        injections,
        percent: optional(&args, "--percent", 5.0)?,
        nsamples: optional(&args, "--nsamples", 512)?,
        seqlen: optional(&args, "--seqlen", 2048)?,
        chunk_tokens: optional(&args, "--chunk-tokens", 512)?,
        reserve_sequences: optional(&args, "--reserve-sequences", 8)?,
    };
    if !(0.0..=50.0).contains(&parsed.percent) || parsed.percent == 0.0 {
        bail!("--percent must be in (0, 50]");
    }
    if parsed.nsamples == 0 || parsed.seqlen == 0 || parsed.chunk_tokens == 0 {
        bail!("--nsamples, --seqlen and --chunk-tokens must be positive");
    }
    if parsed.input == parsed.output || parsed.injections == parsed.output {
        bail!("--output must differ from both input files");
    }
    if parsed.output.exists() {
        bail!("refusing to overwrite {}", parsed.output.display());
    }
    Ok(parsed)
}

fn render_record(tok: &QwenTokenizer, line: &str, path: &Path, lineno: usize) -> Result<String> {
    let record: Value = serde_json::from_str(line)
        .with_context(|| format!("{}:{lineno}: invalid JSON", path.display()))?;
    if let Some(text) = record.get("text").and_then(Value::as_str) {
        return Ok(text.to_string());
    }
    let messages: Vec<ChatMessage> = serde_json::from_value(
        record
            .get("messages")
            .cloned()
            .with_context(|| format!("{}:{lineno}: expected text or messages", path.display()))?,
    )
    .with_context(|| format!("{}:{lineno}: invalid messages", path.display()))?;
    tok.apply_chat_template_no_gen(&messages, None, None)
        .with_context(|| format!("{}:{lineno}: render chat template", path.display()))
}

fn load_chunks(
    tok: &QwenTokenizer,
    path: &Path,
    injection: bool,
    chunk_tokens: usize,
    separator_tokens: usize,
) -> Result<Vec<Chunk>> {
    let raw = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut chunks = Vec::new();
    for (i, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let text = render_record(tok, line, path, i + 1)?;
        let ids = tok.encode(&text, false)?;
        for ids_chunk in ids.chunks(chunk_tokens) {
            let decoded = tok.decode(ids_chunk, false)?;
            let reencoded = tok.encode(&decoded, false)?;
            if reencoded.is_empty() {
                continue;
            }
            chunks.push(Chunk {
                text: decoded,
                tokens_with_separator: reencoded.len() + separator_tokens,
                injection,
            });
        }
    }
    if chunks.is_empty() {
        bail!("{} produced no token chunks", path.display());
    }
    Ok(chunks)
}

fn projected_error(
    total: usize,
    security: usize,
    chunk: &Chunk,
    limit: usize,
    fraction: f64,
) -> f64 {
    let used = chunk.tokens_with_separator.min(limit.saturating_sub(total));
    if used == 0 {
        return f64::INFINITY;
    }
    let next_total = total + used;
    let next_security = security + if chunk.injection { used } else { 0 };
    ((next_security as f64 / next_total as f64) - fraction).abs()
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let tok = QwenTokenizer::from_file(&args.model_dir.join("tokenizer.json").to_string_lossy())?;
    let separator_tokens = tok.encode("\n\n", false)?.len().max(1);
    let normal = load_chunks(
        &tok,
        &args.input,
        false,
        args.chunk_tokens,
        separator_tokens,
    )?;
    let injection = load_chunks(
        &tok,
        &args.injections,
        true,
        args.chunk_tokens,
        separator_tokens,
    )?;

    let consumed_limit = args
        .nsamples
        .checked_mul(args.seqlen)
        .context("calibration token budget overflow")?;
    let output_limit = consumed_limit
        .checked_add(
            args.reserve_sequences
                .checked_mul(args.seqlen)
                .context("reserve overflow")?,
        )
        .context("output token budget overflow")?;
    let fraction = args.percent / 100.0;
    let mut writer = BufWriter::new(
        File::create(&args.output).with_context(|| format!("create {}", args.output.display()))?,
    );
    let mut normal_i = 0usize;
    let mut injection_i = 0usize;
    let mut total = 0usize;
    let mut security = 0usize;
    let mut consumed_security = 0usize;
    let mut records = 0usize;

    while total < output_limit {
        if normal_i >= normal.len() {
            bail!("normal corpus exhausted after {total} scheduled tokens");
        }
        let normal_chunk = &normal[normal_i];
        let injection_chunk = &injection[injection_i % injection.len()];
        let schedule_limit = if total < consumed_limit {
            consumed_limit
        } else {
            output_limit
        };
        let pick_injection =
            projected_error(total, security, injection_chunk, schedule_limit, fraction)
                < projected_error(total, security, normal_chunk, schedule_limit, fraction);
        let chunk = if pick_injection {
            injection_i += 1;
            injection_chunk
        } else {
            normal_i += 1;
            normal_chunk
        };

        let before_consumed = total.min(consumed_limit);
        let after_consumed = (total + chunk.tokens_with_separator).min(consumed_limit);
        if chunk.injection {
            security += chunk.tokens_with_separator;
            consumed_security += after_consumed - before_consumed;
        }
        total += chunk.tokens_with_separator;
        records += 1;
        serde_json::to_writer(
            &mut writer,
            &json!({
                "text": chunk.text,
                "calibration_category": if chunk.injection { "prompt_injection" } else { "baseline" }
            }),
        )?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;

    let actual = 100.0 * consumed_security as f64 / consumed_limit as f64;
    println!("wrote {}", args.output.display());
    println!("records: {records}; scheduled tokens (including separators): {total}");
    println!(
        "consumed prefix: {consumed_limit} tokens = {} samples x {}",
        args.nsamples, args.seqlen
    );
    println!(
        "prompt-injection share in consumed prefix: {consumed_security}/{consumed_limit} = {actual:.4}% (target {:.4}%)",
        args.percent
    );
    println!(
        "source chunks used: {normal_i}/{}; injection chunks used: {injection_i} ({} unique)",
        normal.len(),
        injection.len()
    );
    Ok(())
}
