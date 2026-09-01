//! NVFP4 input-global-scale histogram merging and domain auditing.

use anyhow::{bail, Context, Result};
use serde_json::{json, Map, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};

const HIST_BINS: usize = 512;
const LOG2_MIN: f64 = -40.0;
const LOG2_MAX: f64 = 40.0;
const E4M3_NORMAL_RANGE: f64 = 28_672.0;
const ANCHOR_FLOOR_RATIO: f64 = 1.0e6;
const RECIPROCAL_NUMERATOR: f64 = 6.0 * 448.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Method {
    Auto,
    Headroom,
    Max,
}

impl Method {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "auto" => Ok(Self::Auto),
            "headroom" => Ok(Self::Headroom),
            "max" => Ok(Self::Max),
            _ => bail!("unknown method {value:?}; expected auto, headroom, or max"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Headroom => "headroom",
            Self::Max => "max",
        }
    }
}

#[derive(Debug)]
struct MergeArgs {
    output: PathBuf,
    method: Method,
    anchor_percentile: Option<f64>,
    upper_percentile: Option<f64>,
    rho: Option<f64>,
    inputs: Vec<PathBuf>,
}

#[derive(Debug)]
struct AuditArgs {
    root: PathBuf,
    output: PathBuf,
    warn_ratio: f64,
}

#[derive(Clone, Debug)]
struct StemStats {
    histogram: Vec<u64>,
    running_max: f64,
    zero_blocks: u64,
    invalid_blocks: u64,
}

#[derive(Debug)]
struct DerivedScale {
    anchor: Option<f64>,
    upper: f64,
    span: Option<f64>,
    selected_amax: f64,
    input_global_scale: f64,
    has_headroom: bool,
    range_exceeds_e4m3: bool,
}

fn usage(exit_code: i32) -> ! {
    eprintln!(
        "usage:\n  calib_igs merge --output FILE [--method auto|headroom|max] \
         [--anchor-percentile P] [--upper-percentile P] [--rho R] INPUT...\n  \
         calib_igs audit --root DIR --output FILE [--warn-ratio R]"
    );
    std::process::exit(exit_code);
}

fn take_value(args: &[String], index: &mut usize, flag: &str) -> Result<String> {
    *index += 1;
    args.get(*index)
        .cloned()
        .with_context(|| format!("missing value after {flag}"))
}

fn parse_merge(args: &[String]) -> Result<MergeArgs> {
    let mut output = None;
    let mut method = Method::Auto;
    let mut anchor_percentile = None;
    let mut upper_percentile = None;
    let mut rho = None;
    let mut inputs = Vec::new();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--output" => output = Some(PathBuf::from(take_value(args, &mut index, "--output")?)),
            "--method" => method = Method::parse(&take_value(args, &mut index, "--method")?)?,
            "--anchor-percentile" => {
                anchor_percentile = Some(
                    take_value(args, &mut index, "--anchor-percentile")?
                        .parse()
                        .context("invalid --anchor-percentile")?,
                )
            }
            "--upper-percentile" => {
                upper_percentile = Some(
                    take_value(args, &mut index, "--upper-percentile")?
                        .parse()
                        .context("invalid --upper-percentile")?,
                )
            }
            "--rho" => {
                rho = Some(
                    take_value(args, &mut index, "--rho")?
                        .parse()
                        .context("invalid --rho")?,
                )
            }
            flag if flag.starts_with('-') => bail!("unknown merge argument {flag}"),
            input => inputs.push(PathBuf::from(input)),
        }
        index += 1;
    }
    if inputs.is_empty() {
        bail!("merge requires at least one input scale file");
    }
    Ok(MergeArgs {
        output: output.context("missing required --output")?,
        method,
        anchor_percentile,
        upper_percentile,
        rho,
        inputs,
    })
}

fn parse_audit(args: &[String]) -> Result<AuditArgs> {
    let mut root = None;
    let mut output = None;
    let mut warn_ratio: f64 = 1.5;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--root" => root = Some(PathBuf::from(take_value(args, &mut index, "--root")?)),
            "--output" => output = Some(PathBuf::from(take_value(args, &mut index, "--output")?)),
            "--warn-ratio" => {
                warn_ratio = take_value(args, &mut index, "--warn-ratio")?
                    .parse()
                    .context("invalid --warn-ratio")?
            }
            flag => bail!("unknown audit argument {flag}"),
        }
        index += 1;
    }
    if !warn_ratio.is_finite() || warn_ratio <= 0.0 {
        bail!("--warn-ratio must be a positive finite number");
    }
    Ok(AuditArgs {
        root: root.context("missing required --root")?,
        output: output.context("missing required --output")?,
        warn_ratio,
    })
}

fn read_json(path: &Path) -> Result<Value> {
    serde_json::from_reader(BufReader::new(File::open(path)?))
        .with_context(|| format!("parse JSON {}", path.display()))
}

fn write_json_new(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("refusing to overwrite {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

fn appended_path(path: &Path, suffix: &str) -> PathBuf {
    PathBuf::from(format!("{}{suffix}", path.display()))
}

fn stats_path_for(scale_path: &Path) -> Result<PathBuf> {
    let stem = scale_path
        .file_stem()
        .and_then(|value| value.to_str())
        .context("scale path has no UTF-8 file stem")?;
    Ok(scale_path.with_file_name(format!("{stem}.stats.json")))
}

fn bin_index(value: f64) -> usize {
    let fraction = (value.log2() - LOG2_MIN) / (LOG2_MAX - LOG2_MIN);
    (fraction * HIST_BINS as f64)
        .floor()
        .clamp(0.0, (HIST_BINS - 1) as f64) as usize
}

fn bin_center(index: usize) -> f64 {
    let log2_value = LOG2_MIN + (index as f64 + 0.5) / HIST_BINS as f64 * (LOG2_MAX - LOG2_MIN);
    2.0_f64.powf(log2_value)
}

fn histogram_percentile(
    histogram: &[u64],
    percentile: f64,
    floor_value: Option<f64>,
) -> Option<f64> {
    let start = floor_value
        .filter(|value| *value > 0.0)
        .map(bin_index)
        .unwrap_or(0);
    let total: u128 = histogram[start..].iter().map(|count| *count as u128).sum();
    if total == 0 {
        return None;
    }
    let target = percentile / 100.0 * total as f64;
    let mut cumulative = 0_u128;
    for (index, count) in histogram.iter().enumerate().skip(start) {
        cumulative += *count as u128;
        if cumulative as f64 >= target {
            return Some(bin_center(index));
        }
    }
    None
}

fn validate_policy(
    method: Method,
    anchor_percentile: f64,
    upper_percentile: f64,
    rho: f64,
) -> Result<()> {
    if method == Method::Auto {
        bail!("internal error: auto policy was not resolved");
    }
    if !(0.0 < anchor_percentile && anchor_percentile <= 100.0) {
        bail!("anchor percentile must be in (0, 100]");
    }
    if !(0.0 < upper_percentile && upper_percentile <= 100.0) {
        bail!("upper percentile must be in (0, 100]");
    }
    if !(0.0 < rho && rho < E4M3_NORMAL_RANGE) {
        bail!("rho must be in (0, {E4M3_NORMAL_RANGE})");
    }
    Ok(())
}

fn derive_scale(
    histogram: &[u64],
    running_max: f64,
    method: Method,
    anchor_percentile: f64,
    upper_percentile: f64,
    rho: f64,
) -> Result<DerivedScale> {
    if histogram.len() != HIST_BINS {
        bail!(
            "expected {HIST_BINS} histogram bins, got {}",
            histogram.len()
        );
    }
    if !running_max.is_finite() || running_max <= 0.0 {
        bail!("invalid running max {running_max}");
    }
    if method == Method::Max {
        return Ok(DerivedScale {
            anchor: None,
            upper: running_max,
            span: None,
            selected_amax: running_max,
            input_global_scale: RECIPROCAL_NUMERATOR / running_max,
            has_headroom: false,
            range_exceeds_e4m3: false,
        });
    }
    let upper = if upper_percentile >= 100.0 {
        running_max
    } else {
        histogram_percentile(histogram, upper_percentile, None).unwrap_or(running_max)
    };
    let anchor = histogram_percentile(
        histogram,
        anchor_percentile,
        Some(upper / ANCHOR_FLOOR_RATIO),
    )
    .filter(|value| value.is_finite() && *value > 0.0);
    let (span, selected_amax, has_headroom, range_exceeds_e4m3) = match anchor {
        Some(anchor) => {
            let span = upper / anchor;
            let with_headroom = rho * anchor;
            (
                Some(span),
                upper.max(with_headroom),
                with_headroom > upper,
                span > E4M3_NORMAL_RANGE,
            )
        }
        None => (None, running_max, false, false),
    };
    Ok(DerivedScale {
        anchor,
        upper,
        span,
        selected_amax,
        input_global_scale: RECIPROCAL_NUMERATOR / selected_amax,
        has_headroom,
        range_exceeds_e4m3,
    })
}

fn require_u64(value: Option<&Value>, context: &str) -> Result<u64> {
    value
        .and_then(Value::as_u64)
        .with_context(|| format!("{context} must be a non-negative integer"))
}

fn load_stats(path: &Path) -> Result<(Value, BTreeMap<String, StemStats>)> {
    let document = read_json(path)?;
    if document.get("format").and_then(Value::as_str) != Some("veloGB10-igs-hist-v2") {
        bail!("{}: unsupported stats format", path.display());
    }
    let histogram_geometry = document
        .get("histogram")
        .context("missing histogram geometry")?;
    if histogram_geometry.get("bins").and_then(Value::as_u64) != Some(HIST_BINS as u64)
        || histogram_geometry.get("log2_min").and_then(Value::as_f64) != Some(LOG2_MIN)
        || histogram_geometry.get("log2_max").and_then(Value::as_f64) != Some(LOG2_MAX)
        || document.get("block_size").and_then(Value::as_u64) != Some(16)
    {
        bail!("{}: incompatible histogram geometry", path.display());
    }
    let raw_stems = document
        .get("stems")
        .and_then(Value::as_object)
        .with_context(|| format!("{}: missing stems object", path.display()))?;
    let mut stems = BTreeMap::new();
    for (stem, raw) in raw_stems {
        let raw_histogram = raw
            .get("histogram")
            .and_then(Value::as_array)
            .with_context(|| format!("{stem}: malformed histogram"))?;
        if raw_histogram.len() != HIST_BINS {
            bail!("{stem}: malformed histogram");
        }
        let histogram = raw_histogram
            .iter()
            .enumerate()
            .map(|(index, count)| require_u64(Some(count), &format!("{stem} histogram[{index}]")))
            .collect::<Result<Vec<_>>>()?;
        let running_max = raw
            .get("running_max")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        if !running_max.is_finite() || running_max < 0.0 {
            bail!("{stem}: invalid running_max {running_max}");
        }
        let invalid_blocks = match raw.get("invalid_blocks") {
            Some(value) => require_u64(Some(value), &format!("{stem} invalid_blocks"))?,
            None => 0,
        };
        if invalid_blocks != 0 {
            bail!("{stem}: {invalid_blocks} non-finite activation blocks");
        }
        stems.insert(
            stem.clone(),
            StemStats {
                histogram,
                running_max,
                zero_blocks: raw.get("zero_blocks").and_then(Value::as_u64).unwrap_or(0),
                invalid_blocks,
            },
        );
    }
    Ok((document, stems))
}

fn derived_json(derived: &DerivedScale) -> Value {
    json!({
        "anchor": derived.anchor,
        "upper": derived.upper,
        "span": derived.span,
        "selected_amax": derived.selected_amax,
        "input_global_scale": derived.input_global_scale,
        "has_headroom": derived.has_headroom,
        "range_exceeds_e4m3": derived.range_exceeds_e4m3,
    })
}

fn merge_histogram_stats(
    args: &MergeArgs,
    stats_paths: &[PathBuf],
) -> Result<(Value, Value, Value)> {
    let loaded = stats_paths
        .iter()
        .map(|path| load_stats(path))
        .collect::<Result<Vec<_>>>()?;
    let first_policy = loaded[0].0.get("policy").and_then(Value::as_object);
    let method = if args.method == Method::Auto {
        Method::parse(
            first_policy
                .and_then(|policy| policy.get("method"))
                .and_then(Value::as_str)
                .unwrap_or("headroom"),
        )?
    } else {
        args.method
    };
    let policy_number = |name: &str, default: f64| {
        first_policy
            .and_then(|policy| policy.get(name))
            .and_then(Value::as_f64)
            .unwrap_or(default)
    };
    let anchor_percentile = args
        .anchor_percentile
        .unwrap_or_else(|| policy_number("anchor_percentile", 1.0));
    let upper_percentile = args
        .upper_percentile
        .unwrap_or_else(|| policy_number("upper_percentile", 99.99));
    let rho = args.rho.unwrap_or_else(|| policy_number("rho", 16_384.0));
    validate_policy(method, anchor_percentile, upper_percentile, rho)?;

    let mut merged: BTreeMap<String, StemStats> = BTreeMap::new();
    for (_, document_stems) in loaded {
        for (stem, stats) in document_stems {
            let target = merged.entry(stem).or_insert_with(|| StemStats {
                histogram: vec![0; HIST_BINS],
                running_max: 0.0,
                zero_blocks: 0,
                invalid_blocks: 0,
            });
            for (target_count, count) in target.histogram.iter_mut().zip(stats.histogram) {
                *target_count = target_count
                    .checked_add(count)
                    .context("histogram count overflow")?;
            }
            target.running_max = target.running_max.max(stats.running_max);
            target.zero_blocks = target
                .zero_blocks
                .checked_add(stats.zero_blocks)
                .context("zero block overflow")?;
            target.invalid_blocks = target
                .invalid_blocks
                .checked_add(stats.invalid_blocks)
                .context("invalid block overflow")?;
        }
    }

    let mut scales = Map::new();
    let mut diagnostics = Map::new();
    let mut wide_stems = Vec::new();
    let mut unfed_stems = Vec::new();
    for (stem, stats) in merged {
        let nonzero_blocks: u128 = stats.histogram.iter().map(|count| *count as u128).sum();
        if stats.running_max == 0.0 && nonzero_blocks == 0 {
            unfed_stems.push(stem);
            continue;
        }
        let derived = derive_scale(
            &stats.histogram,
            stats.running_max,
            method,
            anchor_percentile,
            upper_percentile,
            rho,
        )?;
        scales.insert(stem.clone(), json!(derived.input_global_scale));
        if derived.range_exceeds_e4m3 {
            wide_stems.push(stem.clone());
        }
        let mut diagnostic = derived_json(&derived).as_object().cloned().unwrap();
        diagnostic.insert("histogram".into(), json!(stats.histogram));
        diagnostic.insert("running_max".into(), json!(stats.running_max));
        diagnostic.insert("zero_blocks".into(), json!(stats.zero_blocks));
        diagnostic.insert("invalid_blocks".into(), json!(stats.invalid_blocks));
        diagnostic.insert("nonzero_blocks".into(), json!(nonzero_blocks));
        diagnostics.insert(stem, Value::Object(diagnostic));
    }
    let path_strings = |paths: &[PathBuf]| {
        paths
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
    };
    let stats_document = json!({
        "format":"veloGB10-igs-hist-v2",
        "scale_convention":"input_global_scale = 2688 / activation_amax",
        "block_size":16,
        "histogram":{"bins":HIST_BINS,"log2_min":LOG2_MIN,"log2_max":LOG2_MAX},
        "policy":{"method":method.as_str(),"anchor_percentile":anchor_percentile,"upper_percentile":upper_percentile,"rho":rho},
        "merged_from":path_strings(stats_paths),
        "stems":diagnostics,
    });
    let manifest = json!({
        "format":"veloGB10-igs-merge-v2",
        "rule":"sum per-16 block-amax histograms, then derive one global scale per stem",
        "method":method.as_str(),
        "anchor_percentile":anchor_percentile,
        "upper_percentile":upper_percentile,
        "rho":rho,
        "inputs":path_strings(&args.inputs),
        "stats_inputs":path_strings(stats_paths),
        "stems":scales.len(),
        "range_exceeds_e4m3_stems":wide_stems,
        "unfed_stems":unfed_stems,
    });
    Ok((Value::Object(scales), stats_document, manifest))
}

fn merge_legacy_scales(paths: &[PathBuf]) -> Result<(Value, Value)> {
    let mut merged: BTreeMap<String, f64> = BTreeMap::new();
    let mut provenance: BTreeMap<String, String> = BTreeMap::new();
    for path in paths {
        let values = read_json(path)?;
        let object = values
            .as_object()
            .with_context(|| format!("{} is not a JSON object", path.display()))?;
        for (stem, raw_scale) in object {
            let Some(scale) = raw_scale.as_f64() else {
                continue;
            };
            if !scale.is_finite() || scale <= 0.0 {
                continue;
            }
            if merged.get(stem).is_none_or(|selected| scale < *selected) {
                merged.insert(stem.clone(), scale);
                provenance.insert(stem.clone(), path.display().to_string());
            }
        }
    }
    let manifest = json!({
        "format":"veloGB10-igs-merge-v1",
        "rule":"legacy minimum input_global_scale = maximum observed activation amax",
        "inputs":paths.iter().map(|path|path.display().to_string()).collect::<Vec<_>>(),
        "stems":merged.len(),
        "selected_from":provenance,
    });
    Ok((serde_json::to_value(merged)?, manifest))
}

fn merge(args: MergeArgs) -> Result<()> {
    let manifest_path = appended_path(&args.output, ".manifest.json");
    let merged_stats_path = stats_path_for(&args.output)?;
    for path in [&args.output, &manifest_path] {
        if path.exists() {
            bail!("refusing to overwrite {}", path.display());
        }
    }
    let stats_paths = args
        .inputs
        .iter()
        .map(|path| stats_path_for(path))
        .collect::<Result<Vec<_>>>()?;
    let present = stats_paths
        .iter()
        .map(|path| path.exists())
        .collect::<Vec<_>>();
    let (scales, stats_document, manifest) = if present.iter().all(|exists| *exists) {
        if merged_stats_path.exists() {
            bail!("refusing to overwrite {}", merged_stats_path.display());
        }
        let (scales, stats, manifest) = merge_histogram_stats(&args, &stats_paths)?;
        (scales, Some(stats), manifest)
    } else if present.iter().any(|exists| *exists) {
        let missing = stats_paths
            .iter()
            .zip(present)
            .filter(|(_, exists)| !*exists)
            .map(|(path, _)| path.display().to_string())
            .collect::<Vec<_>>();
        bail!("partial histogram inputs; missing: {}", missing.join(", "));
    } else {
        if args.method != Method::Auto
            || args.anchor_percentile.is_some()
            || args.upper_percentile.is_some()
            || args.rho.is_some()
        {
            bail!("headroom/max policy options require input_global_scale.stats.json inputs");
        }
        let (scales, manifest) = merge_legacy_scales(&args.inputs)?;
        (scales, None, manifest)
    };
    let scale_count = scales.as_object().map(Map::len).unwrap_or(0);
    if scale_count == 0 {
        bail!("no valid scales found");
    }
    write_json_new(&args.output, &scales)?;
    if let Some(stats) = stats_document {
        write_json_new(&merged_stats_path, &stats)?;
    }
    write_json_new(&manifest_path, &manifest)?;
    println!(
        "[igs-merge] {scale_count} scales ({}) -> {}",
        manifest["format"].as_str().unwrap_or("unknown"),
        args.output.display()
    );
    Ok(())
}

fn interpolated_percentile(values: &[f64], fraction: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    let position = (ordered.len() - 1) as f64 * fraction;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        Some(ordered[lower])
    } else {
        Some(
            ordered[lower] * (upper as f64 - position) + ordered[upper] * (position - lower as f64),
        )
    }
}

fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    let middle = ordered.len() / 2;
    if ordered.len() % 2 == 0 {
        Some((ordered[middle - 1] + ordered[middle]) / 2.0)
    } else {
        Some(ordered[middle])
    }
}

fn load_positive_scales(path: &Path) -> Result<BTreeMap<String, f64>> {
    let document = read_json(path)?;
    let object = document
        .as_object()
        .with_context(|| format!("{} is not a JSON object", path.display()))?;
    Ok(object
        .iter()
        .filter_map(|(stem, value)| {
            value
                .as_f64()
                .filter(|scale| scale.is_finite() && *scale > 0.0)
                .map(|scale| (stem.clone(), scale))
        })
        .collect())
}

fn audit(args: AuditArgs) -> Result<()> {
    if args.output.exists() {
        bail!("refusing to overwrite {}", args.output.display());
    }
    let mut category_paths = Vec::new();
    for entry in
        fs::read_dir(&args.root).with_context(|| format!("read {}", args.root.display()))?
    {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            let scale_path = entry.path().join("input_global_scale.json");
            if scale_path.is_file() {
                category_paths.push((entry.file_name().to_string_lossy().into_owned(), scale_path));
            }
        }
    }
    category_paths.sort_by(|left, right| left.0.cmp(&right.0));
    let mut categories = BTreeMap::new();
    for (category, path) in category_paths {
        categories.insert(category, load_positive_scales(&path)?);
    }
    if categories.len() < 2 {
        bail!(
            "need at least two category results below {}",
            args.root.display()
        );
    }
    let all_stems: BTreeSet<String> = categories
        .values()
        .flat_map(|values| values.keys().cloned())
        .collect();
    let mut comparisons = Vec::new();
    for stem in all_stems {
        let amax_by_category: BTreeMap<String, f64> = categories
            .iter()
            .filter_map(|(category, values)| {
                values
                    .get(&stem)
                    .map(|scale| (category.clone(), RECIPROCAL_NUMERATOR / scale))
            })
            .collect();
        if amax_by_category.len() < 2 {
            continue;
        }
        let (lowest_category, low) = amax_by_category
            .iter()
            .min_by(|left, right| left.1.total_cmp(right.1))
            .unwrap();
        let (highest_category, high) = amax_by_category
            .iter()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .unwrap();
        comparisons.push(json!({
            "stem":stem,
            "amax_by_category":amax_by_category,
            "max_over_min":high/low,
            "lowest_category":lowest_category,
            "highest_category":highest_category,
        }));
    }
    let ratios = comparisons
        .iter()
        .filter_map(|item| {
            item["max_over_min"]
                .as_f64()
                .filter(|ratio| ratio.is_finite())
        })
        .collect::<Vec<_>>();
    let mut flagged = comparisons
        .iter()
        .filter(|item| {
            item["max_over_min"]
                .as_f64()
                .is_some_and(|ratio| ratio >= args.warn_ratio)
        })
        .cloned()
        .collect::<Vec<_>>();
    flagged.sort_by(|left, right| {
        right["max_over_min"]
            .as_f64()
            .unwrap()
            .total_cmp(&left["max_over_min"].as_f64().unwrap())
            .then_with(|| left["stem"].as_str().cmp(&right["stem"].as_str()))
    });
    let category_counts: BTreeMap<String, usize> = categories
        .iter()
        .map(|(name, values)| (name.clone(), values.len()))
        .collect();
    let report = json!({
        "format":"veloGB10-igs-domain-audit-v1",
        "note":"amax is reconstructed as 2688/input_global_scale; ratios compare domains, not model quality",
        "categories":category_counts,
        "stems_compared":comparisons.len(),
        "warn_ratio":args.warn_ratio,
        "ratio_summary":{
            "median":median(&ratios),
            "p90":interpolated_percentile(&ratios,0.90),
            "p95":interpolated_percentile(&ratios,0.95),
            "maximum":ratios.iter().copied().max_by(f64::total_cmp),
        },
        "flagged_count":flagged.len(),
        "flagged":flagged,
    });
    write_json_new(&args.output, &report)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "categories":report["categories"],
            "stems_compared":report["stems_compared"],
            "ratio_summary":report["ratio_summary"],
            "flagged_count":report["flagged_count"],
        }))?
    );
    println!("[igs-audit] report: {}", args.output.display());
    Ok(())
}

fn main() -> Result<()> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let Some(command) = args.first().map(String::as_str) else {
        usage(2)
    };
    match command {
        "merge" => merge(parse_merge(&args[1..])?),
        "audit" => audit(parse_audit(&args[1..])?),
        "--help" | "-h" => usage(0),
        other => bail!("unknown command {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_policy_uses_running_max() {
        let result =
            derive_scale(&vec![0; HIST_BINS], 12.0, Method::Max, 1.0, 99.99, 16_384.0).unwrap();
        assert_eq!(result.selected_amax, 12.0);
        assert_eq!(result.input_global_scale, 224.0);
        assert!(!result.has_headroom);
    }

    #[test]
    fn histogram_percentile_uses_log_bin_center() {
        let mut histogram = vec![0; HIST_BINS];
        histogram[128] = 3;
        histogram[384] = 1;
        assert_eq!(
            histogram_percentile(&histogram, 50.0, None),
            Some(bin_center(128))
        );
        assert_eq!(
            histogram_percentile(&histogram, 100.0, None),
            Some(bin_center(384))
        );
    }

    #[test]
    fn interpolated_percentiles_match_statistics_recipe() {
        let values = [1.0, 2.0, 4.0, 8.0];
        assert_eq!(median(&values), Some(3.0));
        assert!((interpolated_percentile(&values, 0.90).unwrap() - 6.8).abs() < 1e-12);
    }

    #[test]
    fn stats_path_replaces_only_final_suffix() {
        assert_eq!(
            stats_path_for(Path::new("run/input_global_scale.json")).unwrap(),
            PathBuf::from("run/input_global_scale.stats.json")
        );
    }
}
