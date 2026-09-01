//! Compose deterministic, sample-aligned GPTQ calibration JSONL.
//!
//! Every output line carries exactly one tokenized sample in `input_ids`. This
//! prevents the GPTQ loader from re-tokenizing, concatenating unrelated records,
//! or resetting recurrent state in the middle of a scheduled sample.

use anyhow::{bail, Context, Result};
use gb10_inference::tokenizer::{ChatMessage, QwenTokenizer};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

#[derive(Clone)]
struct Chunk {
    ids: Vec<u32>,
    source_id: String,
    window: usize,
    metadata: Value,
}

struct Source {
    name: String,
    target: f64,
    path: PathBuf,
    max_document_tokens: usize,
    chunks: Vec<Chunk>,
    cursor: usize,
    chunk_offset: usize,
    consumed: usize,
    scheduled: usize,
    used_documents: BTreeSet<String>,
    metadata_tokens: BTreeMap<String, usize>,
    window_tokens: BTreeMap<usize, usize>,
    trajectory_packing: bool,
}

struct Args {
    model_dir: PathBuf,
    output: PathBuf,
    sources: Vec<(String, f64, usize, PathBuf)>,
    nsamples: usize,
    seqlen: usize,
    maca_lengths: Option<Vec<usize>>,
    token_budget: Option<usize>,
    reserve_sequences: usize,
    trajectory_packing: bool,
}

#[derive(Default)]
struct BuiltSample {
    ids: Vec<u32>,
    category_tokens: BTreeMap<String, usize>,
    provenance: Vec<Value>,
}

fn optional<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    match args.iter().position(|arg| arg == flag) {
        Some(index) => args
            .get(index + 1)
            .with_context(|| format!("missing value after {flag}"))?
            .parse()
            .map_err(|error| anyhow::anyhow!("invalid {flag}: {error}")),
        None => Ok(default),
    }
}

fn required(args: &[String], flag: &str) -> Result<String> {
    let index = args
        .iter()
        .position(|arg| arg == flag)
        .with_context(|| format!("missing {flag}"))?;
    args.get(index + 1)
        .cloned()
        .with_context(|| format!("missing value after {flag}"))
}

fn parse_source(raw: &str) -> Result<(String, f64, usize, PathBuf)> {
    let (name, rest) = raw
        .split_once('=')
        .with_context(|| format!("invalid --source {raw:?}"))?;
    let mut fields = rest.splitn(3, ':');
    let percent: f64 = fields
        .next()
        .context("missing source percent")?
        .parse()
        .with_context(|| format!("invalid percent in --source {raw:?}"))?;
    let max_document_tokens: usize = fields
        .next()
        .context("missing max document tokens")?
        .parse()
        .with_context(|| format!("invalid max document tokens in --source {raw:?}"))?;
    let path = fields.next().context("missing source path")?;
    if name.is_empty() || percent <= 0.0 || max_document_tokens == 0 || path.is_empty() {
        bail!("invalid --source {raw:?}");
    }
    Ok((
        name.to_string(),
        percent / 100.0,
        max_document_tokens,
        PathBuf::from(path),
    ))
}

fn parse_args() -> Result<Args> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.iter().any(|arg| arg == "-h" || arg == "--help") {
        println!(
            "Usage: calib_compose --model-dir DIR --output FILE \\\n  --source NAME=PERCENT:MAX_DOCUMENT_TOKENS:FILE [--source ...]\n\
             \nOptions:\n\
             \x20 --nsamples N            default: 512\n\
             \x20 --seqlen N              default: 2048\n\
             \x20 --maca-lengths LIST     e.g. 256,512,1024,2048,4096\n\
             \x20 --token-budget N        default: nsamples * seqlen\n\
             \x20 --reserve-sequences N   default: 0\n\
             \x20 --trajectory-packing    keep conversation windows adjacent and consume chunks continuously\n\
             \nWith --maca-lengths, sequence counts are derived under a fixed token budget.\n\
             Each JSONL output record is one exact, pre-tokenized sample."
        );
        std::process::exit(0);
    }
    let mut sources = Vec::new();
    let mut index = 0;
    while index < raw.len() {
        if raw[index] == "--source" {
            sources.push(parse_source(
                raw.get(index + 1).context("missing value after --source")?,
            )?);
            index += 2;
        } else {
            index += 1;
        }
    }
    if sources.is_empty() {
        bail!("at least one --source is required");
    }
    let total: f64 = sources.iter().map(|(_, percent, _, _)| percent).sum();
    if (total - 1.0).abs() > 1e-8 {
        bail!(
            "source percentages total {:.6}%, expected 100%",
            total * 100.0
        );
    }
    let output = PathBuf::from(required(&raw, "--output")?);
    if output.exists() {
        bail!("refusing to overwrite {}", output.display());
    }
    let maca_lengths = raw
        .iter()
        .position(|arg| arg == "--maca-lengths")
        .map(|index| -> Result<Vec<usize>> {
            let value = raw
                .get(index + 1)
                .context("missing value after --maca-lengths")?;
            let mut lengths: Vec<usize> = value
                .split(',')
                .map(|part| {
                    part.parse::<usize>()
                        .with_context(|| format!("invalid MaCa length {part:?}"))
                })
                .collect::<Result<_>>()?;
            lengths.sort_unstable();
            lengths.dedup();
            if lengths.is_empty() || lengths[0] == 0 {
                bail!("--maca-lengths must contain positive lengths");
            }
            Ok(lengths)
        })
        .transpose()?;
    let token_budget = raw
        .iter()
        .position(|arg| arg == "--token-budget")
        .map(|index| {
            raw.get(index + 1)
                .context("missing value after --token-budget")?
                .parse::<usize>()
                .context("invalid --token-budget")
        })
        .transpose()?;
    Ok(Args {
        model_dir: PathBuf::from(required(&raw, "--model-dir")?),
        output,
        sources,
        nsamples: optional(&raw, "--nsamples", 512)?,
        seqlen: optional(&raw, "--seqlen", 2048)?,
        maca_lengths,
        token_budget,
        reserve_sequences: optional(&raw, "--reserve-sequences", 0)?,
        trajectory_packing: raw.iter().any(|arg| arg == "--trajectory-packing"),
    })
}

fn maca_schedule(lengths: &[usize], token_budget: usize) -> Result<Vec<usize>> {
    if token_budget == 0 {
        bail!("MaCa token budget must be positive");
    }
    if lengths
        .iter()
        .any(|&length| length == 0 || length > token_budget)
    {
        bail!("every MaCa length must be in 1..={token_budget}");
    }
    let cycle: usize = lengths.iter().sum();
    let rounds = token_budget / cycle;
    let mut schedule = Vec::with_capacity(rounds * lengths.len() + lengths.len());
    for _ in 0..rounds {
        schedule.extend_from_slice(lengths);
    }
    let mut remaining = token_budget - rounds * cycle;
    for &length in lengths.iter().rev() {
        while length <= remaining {
            schedule.push(length);
            remaining -= length;
        }
    }
    if remaining != 0 {
        bail!("token budget {token_budget} cannot be represented exactly by MaCa lengths {lengths:?} (remainder {remaining})");
    }
    if schedule.is_empty() {
        bail!("MaCa schedule is empty");
    }
    Ok(schedule)
}

fn render_record(tok: &QwenTokenizer, record: &Value, path: &Path, line: usize) -> Result<String> {
    if let Some(text) = record.get("text").and_then(Value::as_str) {
        return Ok(text.to_string());
    }
    let messages: Vec<ChatMessage> = serde_json::from_value(
        record
            .get("messages")
            .cloned()
            .with_context(|| format!("{}:{line}: expected text or messages", path.display()))?,
    )
    .with_context(|| format!("{}:{line}: invalid messages", path.display()))?;
    let tools = record.get("tools").and_then(Value::as_array);
    tok.apply_chat_template_no_gen(&messages, tools.map(Vec::as_slice), None)
        .with_context(|| format!("{}:{line}: render chat template", path.display()))
}

fn load_chunks(
    tok: &QwenTokenizer,
    path: &Path,
    max_document_tokens: usize,
    separator: u32,
    trajectory_packing: bool,
) -> Result<Vec<Chunk>> {
    let raw = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut documents = Vec::new();
    for (line_index, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let record: Value = serde_json::from_str(line)
            .with_context(|| format!("{}:{}: invalid JSON", path.display(), line_index + 1))?;
        let text = render_record(tok, &record, path, line_index + 1)?;
        let mut ids = tok.encode(&text, false)?;
        if ids.is_empty() {
            continue;
        }
        ids.push(separator);
        let metadata = record.get("metadata").cloned().unwrap_or_else(|| json!({}));
        let source_id = metadata
            .get("source_id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| format!("{}:{}", path.display(), line_index + 1));
        documents.push((
            ids,
            source_id,
            metadata,
            record.get("messages").and_then(Value::as_array).is_some(),
        ));
    }
    if documents.is_empty() {
        bail!("{} produced no documents", path.display());
    }
    let mut chunks = Vec::new();
    let conversation_aligned = trajectory_packing && documents.iter().all(|document| document.3);
    if conversation_aligned {
        // Tool and chat trajectories must reach their observations and final
        // answers. Keep every document's windows adjacent instead of consuming
        // the first window of every document before ever seeing a tail.
        for (ids, source_id, metadata, _) in &documents {
            for (window, part) in ids.chunks(max_document_tokens).enumerate() {
                chunks.push(Chunk {
                    ids: part.to_vec(),
                    source_id: source_id.clone(),
                    window,
                    metadata: metadata.clone(),
                });
            }
        }
    } else {
        // Plain documents and repository files remain diversity-first.
        let rounds = documents
            .iter()
            .map(|(ids, _, _, _)| ids.len().div_ceil(max_document_tokens))
            .max()
            .unwrap_or(0);
        for window in 0..rounds {
            let start = window * max_document_tokens;
            for (ids, source_id, metadata, _) in &documents {
                if start >= ids.len() {
                    continue;
                }
                let end = (start + max_document_tokens).min(ids.len());
                chunks.push(Chunk {
                    ids: ids[start..end].to_vec(),
                    source_id: source_id.clone(),
                    window,
                    metadata: metadata.clone(),
                });
            }
        }
    }
    Ok(chunks)
}

fn metadata_key_values(metadata: &Value) -> impl Iterator<Item = String> + '_ {
    ["language", "subtype", "code_language", "scenario"]
        .into_iter()
        .filter_map(|key| {
            metadata
                .get(key)
                .and_then(Value::as_str)
                .map(|value| format!("{key}:{value}"))
        })
}

fn take_from_source(source: &mut Source, amount: usize) -> Result<(Vec<u32>, Value)> {
    if source.chunks.is_empty() {
        bail!("source {} has no chunks", source.name);
    }
    let chunk = &source.chunks[source.cursor % source.chunks.len()];
    let start = if source.trajectory_packing {
        source.chunk_offset
    } else {
        0
    };
    let used = amount.min(chunk.ids.len() - start);
    if used == 0 {
        bail!("source {} yielded an empty chunk", source.name);
    }
    let ids = chunk.ids[start..start + used].to_vec();
    if source.trajectory_packing {
        source.chunk_offset += used;
    }
    if !source.trajectory_packing || source.chunk_offset == chunk.ids.len() {
        source.cursor += 1;
        source.chunk_offset = 0;
    }
    source.used_documents.insert(chunk.source_id.clone());
    for key in metadata_key_values(&chunk.metadata) {
        *source.metadata_tokens.entry(key).or_default() += used;
    }
    *source.window_tokens.entry(chunk.window).or_default() += used;
    let mut provenance = json!({
        "category": source.name,
        "source_id": chunk.source_id,
        "window": chunk.window,
        "tokens": used,
        "metadata": chunk.metadata,
    });
    if source.trajectory_packing {
        provenance
            .as_object_mut()
            .unwrap()
            .insert("window_offset".into(), json!(start));
    }
    Ok((ids, provenance))
}

fn append_category(sample: &mut BuiltSample, source: &mut Source, amount: usize) -> Result<()> {
    let mut remaining = amount;
    while remaining > 0 {
        let cap = remaining.min(source.max_document_tokens);
        let (ids, provenance) = take_from_source(source, cap)?;
        let used = ids.len();
        sample.ids.extend(ids);
        *sample
            .category_tokens
            .entry(source.name.clone())
            .or_default() += used;
        sample.provenance.push(provenance);
        source.consumed += used;
        source.scheduled += used;
        remaining -= used;
    }
    Ok(())
}

fn primary_category(counts: &BTreeMap<String, usize>) -> String {
    counts
        .iter()
        .max_by(|(name_a, count_a), (name_b, count_b)| {
            count_a.cmp(count_b).then_with(|| name_b.cmp(name_a))
        })
        .map(|(name, _)| name.clone())
        .unwrap_or_else(|| "unknown".to_string())
}

fn write_sample(
    writer: &mut BufWriter<File>,
    tok: &QwenTokenizer,
    sample_index: usize,
    sample: BuiltSample,
) -> Result<()> {
    let text = tok.decode(&sample.ids, false)?;
    let primary = primary_category(&sample.category_tokens);
    serde_json::to_writer(
        &mut *writer,
        &json!({
            "format": "veloGB10-calibration-sample-v2",
            "sample_index": sample_index,
            "input_ids": sample.ids,
            "text": text,
            "primary_category": primary,
            "category_tokens": sample.category_tokens,
            "provenance": sample.provenance,
        }),
    )?;
    writer.write_all(b"\n")?;
    Ok(())
}

fn sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn audit(path: &Path, schedule: &[usize], names: &[String]) -> Result<Vec<usize>> {
    let raw = std::fs::read_to_string(path)?;
    let mut totals = vec![0usize; names.len()];
    let mut records = 0usize;
    for (line_index, line) in raw.lines().take(schedule.len()).enumerate() {
        let record: Value = serde_json::from_str(line)?;
        let ids = record["input_ids"]
            .as_array()
            .with_context(|| format!("{}:{}: missing input_ids", path.display(), line_index + 1))?;
        let expected = schedule[line_index];
        if ids.len() != expected {
            bail!(
                "{}:{}: sample has {} tokens, expected {expected}",
                path.display(),
                line_index + 1,
                ids.len()
            );
        }
        let counts = record["category_tokens"]
            .as_object()
            .context("missing category_tokens")?;
        let mut sample_total = 0usize;
        for (name, value) in counts {
            let count = value.as_u64().context("non-integer category token count")? as usize;
            let index = names
                .iter()
                .position(|candidate| candidate == name)
                .with_context(|| format!("unknown category {name}"))?;
            totals[index] += count;
            sample_total += count;
        }
        if sample_total != expected {
            bail!("sample {line_index} category total {sample_total}, expected {expected}");
        }
        records += 1;
    }
    if records != schedule.len() {
        bail!(
            "corpus contains only {records} consumed samples, expected {}",
            schedule.len()
        );
    }
    Ok(totals)
}

fn source_manifest(path: &Path) -> Option<Value> {
    let parent = path.parent()?;
    let manifest_path = parent.join("sources.manifest.json");
    let bytes = std::fs::read(&manifest_path).ok()?;
    let value: Value = serde_json::from_slice(&bytes).ok()?;
    Some(json!({
        "path": manifest_path,
        "sha256": format!("{:x}", Sha256::digest(&bytes)),
        "summary": value,
    }))
}

fn main() -> Result<()> {
    let args = parse_args()?;
    if args.nsamples == 0 || args.seqlen == 0 {
        bail!("--nsamples and --seqlen must be positive");
    }
    let default_budget = args
        .nsamples
        .checked_mul(args.seqlen)
        .context("token budget overflow")?;
    let schedule = match &args.maca_lengths {
        Some(lengths) => maca_schedule(lengths, args.token_budget.unwrap_or(default_budget))?,
        None => {
            if args.token_budget.is_some() {
                bail!("--token-budget requires --maca-lengths");
            }
            vec![args.seqlen; args.nsamples]
        }
    };
    let max_seqlen = *schedule.iter().max().context("empty schedule")?;
    let consumed_tokens: usize = schedule.iter().sum();
    let tok = QwenTokenizer::from_file(&args.model_dir.join("tokenizer.json").to_string_lossy())?;
    let separator = tok.encode("\n\n", false)?.first().copied().unwrap_or(198);
    let mut sources = Vec::new();
    for (name, target, max_document_tokens, path) in args.sources {
        let chunks = load_chunks(
            &tok,
            &path,
            max_document_tokens,
            separator,
            args.trajectory_packing,
        )?;
        println!(
            "[compose] {name}: {} {} chunks from {}",
            chunks.len(),
            if args.trajectory_packing {
                "trajectory-aware"
            } else {
                "diversity-first"
            },
            path.display()
        );
        sources.push(Source {
            name,
            target,
            path,
            max_document_tokens,
            chunks,
            cursor: 0,
            chunk_offset: 0,
            consumed: 0,
            scheduled: 0,
            used_documents: BTreeSet::new(),
            metadata_tokens: BTreeMap::new(),
            window_tokens: BTreeMap::new(),
            trajectory_packing: args.trajectory_packing,
        });
    }
    let mut quotas: Vec<usize> = sources
        .iter()
        .map(|source| (source.target * consumed_tokens as f64).round() as usize)
        .collect();
    let assigned: usize = quotas.iter().take(quotas.len().saturating_sub(1)).sum();
    *quotas.last_mut().context("no sources")? = consumed_tokens - assigned;
    let incomplete = PathBuf::from(format!("{}.incomplete.jsonl", args.output.display()));
    if incomplete.exists() {
        bail!("refusing to overwrite {}", incomplete.display());
    }
    let mut writer = BufWriter::new(File::create(&incomplete)?);
    let mut remaining = quotas.clone();
    for (sample_index, &sample_len) in schedule.iter().enumerate() {
        let mut sample = BuiltSample::default();
        while sample.ids.len() < sample_len {
            let source_index = remaining
                .iter()
                .enumerate()
                .filter(|(_, count)| **count > 0)
                .max_by_key(|(_, count)| **count)
                .map(|(index, _)| index)
                .context("category quotas exhausted before sample was full")?;
            let amount = remaining[source_index].min(sample_len - sample.ids.len());
            append_category(&mut sample, &mut sources[source_index], amount)?;
            remaining[source_index] -= amount;
        }
        write_sample(&mut writer, &tok, sample_index, sample)?;
    }
    // Reserve samples are exact and aligned too, but lie outside the consumed
    // prefix. Pick their primary category using weighted deficit scheduling.
    for reserve_index in 0..args.reserve_sequences {
        let total = sources
            .iter()
            .map(|source| source.scheduled)
            .sum::<usize>()
            .max(1);
        let source_index = sources
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| {
                let left_deficit = left.target - left.scheduled as f64 / total as f64;
                let right_deficit = right.target - right.scheduled as f64 / total as f64;
                left_deficit.total_cmp(&right_deficit)
            })
            .map(|(index, _)| index)
            .context("no reserve source")?;
        let mut sample = BuiltSample::default();
        let reserve_len = args
            .maca_lengths
            .as_ref()
            .map(|lengths| lengths[reserve_index % lengths.len()])
            .unwrap_or(args.seqlen);
        append_category(&mut sample, &mut sources[source_index], reserve_len)?;
        write_sample(&mut writer, &tok, schedule.len() + reserve_index, sample)?;
    }
    writer.flush()?;
    drop(writer);
    let names: Vec<String> = sources.iter().map(|source| source.name.clone()).collect();
    let audited = audit(&incomplete, &schedule, &names)?;
    std::fs::rename(&incomplete, &args.output)?;
    let digest = sha256(&args.output)?;
    let categories: Vec<Value> = sources
        .iter()
        .zip(audited.iter())
        .map(|(source, count)| {
            json!({
                "name": source.name,
                "source": source.path,
                "target_percent": source.target * 100.0,
                "actual_tokens": count,
                "actual_percent": *count as f64 * 100.0 / consumed_tokens as f64,
                "max_document_tokens": source.max_document_tokens,
                "available_chunks": source.chunks.len(),
                "used_chunks": source.cursor + usize::from(source.chunk_offset > 0),
                "full_reuses": source.cursor / source.chunks.len(),
                "unique_documents": source.used_documents.len(),
                "metadata_tokens": source.metadata_tokens,
                "window_tokens": source.window_tokens,
                "late_window_tokens": source.window_tokens.iter()
                    .filter(|(window, _)| **window > 0)
                    .map(|(_, tokens)| *tokens)
                    .sum::<usize>(),
            })
        })
        .collect();
    let source_manifests: Vec<Value> = sources
        .iter()
        .filter_map(|source| source_manifest(&source.path))
        .collect();
    let manifest_path = PathBuf::from(format!("{}.manifest.json", args.output.display()));
    let mut length_histogram = BTreeMap::<String, usize>::new();
    for &length in &schedule {
        *length_histogram.entry(length.to_string()).or_default() += 1;
    }
    let manifest = json!({
        "format": if args.maca_lengths.is_some() { "veloGB10-calibration-corpus-v3-maca" } else { "veloGB10-calibration-corpus-v2" },
        "sample_format": "pretokenized input_ids; one JSONL line per state-reset boundary",
        "model_dir": args.model_dir,
        "output": args.output,
        "sha256": digest,
        "nsamples": schedule.len(),
        "seqlen": max_seqlen,
        "nominal_seqlen": args.seqlen,
        "maca_lengths": args.maca_lengths,
        "length_histogram": length_histogram,
        "hessian_sequence_normalization": if args.maca_lengths.is_some() { "1/sequence_length" } else { "none" },
        "reserve_sequences": args.reserve_sequences,
        "trajectory_packing": args.trajectory_packing,
        "consumed_tokens": consumed_tokens,
        "records": schedule.len() + args.reserve_sequences,
        "categories": categories,
        "source_manifests": source_manifests,
    });
    std::fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
    println!("[compose] wrote {} ({digest})", args.output.display());
    println!(
        "[compose] exact consumed prefix: {} sequences / lengths {:?} = {consumed_tokens} tokens",
        schedule.len(),
        length_histogram
    );
    for (source, count) in sources.iter().zip(audited) {
        println!(
            "[compose] {:>24}: {:>7} tokens = {:>7.4}% (target {:.2}%, {} unique documents)",
            source.name,
            count,
            count as f64 * 100.0 / consumed_tokens as f64,
            source.target * 100.0,
            source.used_documents.len(),
        );
    }
    println!("[compose] manifest: {}", manifest_path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maca_schedule_has_exact_budget_and_all_scales() {
        let lengths = [256, 512, 1024, 2048, 4096];
        let schedule = maca_schedule(&lengths, 512 * 2048).unwrap();
        assert_eq!(schedule.iter().sum::<usize>(), 512 * 2048);
        for length in lengths {
            assert!(schedule.contains(&length));
        }
        let min = schedule.iter().filter(|&&length| length == 256).count();
        let max = schedule.iter().filter(|&&length| length == 4096).count();
        assert!(min.abs_diff(max) <= 1);
    }

    #[test]
    fn maca_schedule_rejects_unrepresentable_budget() {
        assert!(maca_schedule(&[256, 512], 257).is_err());
    }

    #[test]
    fn partial_chunk_consumption_resumes_at_the_exact_suffix() {
        let mut source = Source {
            name: "test".into(),
            target: 1.0,
            path: PathBuf::from("test.jsonl"),
            max_document_tokens: 8,
            chunks: vec![Chunk {
                ids: vec![10, 11, 12, 13, 14],
                source_id: "doc".into(),
                window: 0,
                metadata: json!({"scenario":"continuity"}),
            }],
            cursor: 0,
            chunk_offset: 0,
            consumed: 0,
            scheduled: 0,
            used_documents: BTreeSet::new(),
            metadata_tokens: BTreeMap::new(),
            window_tokens: BTreeMap::new(),
            trajectory_packing: true,
        };

        assert_eq!(take_from_source(&mut source, 3).unwrap().0, [10, 11, 12]);
        assert_eq!(source.cursor, 0);
        assert_eq!(source.chunk_offset, 3);
        assert_eq!(take_from_source(&mut source, 3).unwrap().0, [13, 14]);
        assert_eq!(source.cursor, 1);
        assert_eq!(source.chunk_offset, 0);
    }

    #[test]
    fn v9_packing_keeps_legacy_chunk_advance() {
        let mut source = Source {
            name: "test".into(),
            target: 1.0,
            path: PathBuf::from("test.jsonl"),
            max_document_tokens: 8,
            chunks: vec![
                Chunk {
                    ids: vec![10, 11, 12, 13, 14],
                    source_id: "first".into(),
                    window: 0,
                    metadata: json!({}),
                },
                Chunk {
                    ids: vec![20, 21],
                    source_id: "second".into(),
                    window: 0,
                    metadata: json!({}),
                },
            ],
            cursor: 0,
            chunk_offset: 0,
            consumed: 0,
            scheduled: 0,
            used_documents: BTreeSet::new(),
            metadata_tokens: BTreeMap::new(),
            window_tokens: BTreeMap::new(),
            trajectory_packing: false,
        };

        let (first, provenance) = take_from_source(&mut source, 3).unwrap();
        assert_eq!(first, [10, 11, 12]);
        assert_eq!(source.cursor, 1);
        assert!(provenance.get("window_offset").is_none());
        assert_eq!(take_from_source(&mut source, 3).unwrap().0, [20, 21]);
    }
}
