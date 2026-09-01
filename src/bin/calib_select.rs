//! Activation-aware calibration selection.
//!
//! COLA contributes activation-space k-means representativeness; ACDM contributes distance to a
//! task-reference activation centroid; MoE models additionally receive a concave expert-coverage
//! reward. Category × sequence-length quotas are preserved exactly by a deterministic repair pass.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct Activation {
    layer: usize,
    mean: f64,
    std: f64,
    rms: f64,
    sketch: Vec<f32>,
}
#[derive(Deserialize)]
struct ExpertRoutes {
    layer: usize,
    counts: Vec<u64>,
}
#[derive(Deserialize)]
struct Profile {
    sample_index: usize,
    sequence_length: usize,
    activations: Vec<Activation>,
    #[serde(default)]
    experts: Vec<ExpertRoutes>,
}

struct Args {
    candidates: PathBuf,
    profiles: PathBuf,
    reference_profiles: Option<PathBuf>,
    output: PathBuf,
    nsamples: usize,
    seed: u64,
    iters: usize,
    cola_weight: f64,
    acdm_weight: f64,
    expert_weight: f64,
    preserve_field: String,
    preserve_lengths: bool,
}

fn value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|arg| arg == flag)
        .and_then(|index| args.get(index + 1))
        .cloned()
}
fn parse_num<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    value(args, flag)
        .map(|raw| {
            raw.parse()
                .map_err(|error| anyhow::anyhow!("invalid {flag}: {error}"))
        })
        .unwrap_or(Ok(default))
}
fn parse_args() -> Result<Args> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.iter().any(|arg| arg == "-h" || arg == "--help") {
        println!("Usage: calib_select --candidates CANDIDATES.jsonl --profiles PROFILES.jsonl \\\n  --output SELECTED.jsonl --nsamples N [--reference-profiles TASK.jsonl]\n\n\
Options:\n  --cola-weight F       k-means representativeness weight [1]\n\
  --acdm-weight F       reference-centroid alignment weight [1]\n\
  --expert-weight F     MoE expert-balance reward [1]\n\
  --kmeans-iters N      Lloyd iterations [6]\n  --seed N              deterministic seed [20260831]\n\
  --preserve-field KEY  exact composition field [primary_category]\n\
  --no-preserve-lengths do not preserve the MaCa length histogram");
        std::process::exit(0);
    }
    let required = |flag: &str| value(&raw, flag).with_context(|| format!("missing {flag}"));
    Ok(Args {
        candidates: PathBuf::from(required("--candidates")?),
        profiles: PathBuf::from(required("--profiles")?),
        reference_profiles: value(&raw, "--reference-profiles").map(PathBuf::from),
        output: PathBuf::from(required("--output")?),
        nsamples: parse_num(&raw, "--nsamples", 512)?,
        seed: parse_num(&raw, "--seed", 20260831)?,
        iters: parse_num(&raw, "--kmeans-iters", 6)?,
        cola_weight: parse_num(&raw, "--cola-weight", 1.0)?,
        acdm_weight: parse_num(&raw, "--acdm-weight", 1.0)?,
        expert_weight: parse_num(&raw, "--expert-weight", 1.0)?,
        preserve_field: value(&raw, "--preserve-field")
            .unwrap_or_else(|| "primary_category".into()),
        preserve_lengths: !raw.iter().any(|arg| arg == "--no-preserve-lengths"),
    })
}

fn load_lines(path: &Path) -> Result<Vec<String>> {
    Ok(std::fs::read_to_string(path)
        .with_context(|| format!("read {}", path.display()))?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(str::to_string)
        .collect())
}
fn load_profiles(path: &Path) -> Result<Vec<Profile>> {
    load_lines(path)?
        .into_iter()
        .enumerate()
        .map(|(line, raw)| {
            serde_json::from_str(&raw)
                .with_context(|| format!("{}:{}: invalid profile", path.display(), line + 1))
        })
        .collect()
}
fn cola_feature(profile: &Profile) -> Vec<f64> {
    let mut activations: Vec<&Activation> = profile.activations.iter().collect();
    activations.sort_by_key(|activation| activation.layer);
    let mut feature = Vec::new();
    for activation in activations {
        feature.extend([activation.mean, activation.std, activation.rms]);
        feature.extend(activation.sketch.iter().map(|&value| value as f64));
    }
    feature
}
fn acdm_feature(profile: &Profile) -> Vec<f64> {
    let mut activations: Vec<&Activation> = profile.activations.iter().collect();
    activations.sort_by_key(|activation| activation.layer);
    activations
        .into_iter()
        .flat_map(|activation| [activation.mean, activation.std])
        .collect()
}
fn expert_feature(profile: &Profile) -> Vec<u64> {
    let mut routes: Vec<&ExpertRoutes> = profile.experts.iter().collect();
    routes.sort_by_key(|routes| routes.layer);
    routes
        .into_iter()
        .flat_map(|routes| routes.counts.iter().copied())
        .collect()
}
fn standardize(mut features: Vec<Vec<f64>>) -> Result<(Vec<Vec<f64>>, Vec<f64>, Vec<f64>)> {
    let dim = features.first().context("no activation profiles")?.len();
    if dim == 0 || features.iter().any(|feature| feature.len() != dim) {
        bail!("inconsistent/empty activation profiles");
    }
    let n = features.len() as f64;
    let mut mean = vec![0.0; dim];
    for feature in &features {
        for (dst, value) in mean.iter_mut().zip(feature) {
            *dst += *value / n;
        }
    }
    let mut scale = vec![0.0; dim];
    for feature in &features {
        for ((dst, value), center) in scale.iter_mut().zip(feature).zip(&mean) {
            *dst += (*value - *center).powi(2) / n;
        }
    }
    for value in &mut scale {
        *value = value.sqrt().max(1e-8);
    }
    for feature in &mut features {
        for ((value, center), scale) in feature.iter_mut().zip(&mean).zip(&scale) {
            *value = (*value - *center) / *scale;
        }
    }
    Ok((features, mean, scale))
}
fn transformed_centroid(
    profiles: &[Profile],
    mean: &[f64],
    scale: &[f64],
    extract: fn(&Profile) -> Vec<f64>,
) -> Result<Vec<f64>> {
    if profiles.is_empty() {
        bail!("reference profile set is empty");
    }
    let mut centroid = vec![0.0; mean.len()];
    for profile in profiles {
        let feature = extract(profile);
        if feature.len() != mean.len() {
            bail!(
                "reference profile dimension {} != candidate dimension {}",
                feature.len(),
                mean.len()
            );
        }
        for i in 0..centroid.len() {
            centroid[i] += (feature[i] - mean[i]) / scale[i] / profiles.len() as f64;
        }
    }
    Ok(centroid)
}
fn dist2(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

fn kmeans(features: &[Vec<f64>], k: usize, iters: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<usize>) {
    let n = features.len();
    let dim = features[0].len();
    let mut chosen = vec![false; n];
    let first = (seed as usize) % n;
    let mut centroids = vec![features[first].clone()];
    chosen[first] = true;
    let mut nearest: Vec<f64> = features
        .iter()
        .map(|feature| dist2(feature, &centroids[0]))
        .collect();
    while centroids.len() < k {
        let next = (0..n)
            .filter(|&index| !chosen[index])
            .max_by(|&a, &b| nearest[a].total_cmp(&nearest[b]))
            .unwrap();
        chosen[next] = true;
        centroids.push(features[next].clone());
        for index in 0..n {
            nearest[index] = nearest[index].min(dist2(&features[index], &features[next]));
        }
    }
    let mut assignment = vec![0usize; n];
    for _ in 0..iters.max(1) {
        for (index, feature) in features.iter().enumerate() {
            assignment[index] = (0..k)
                .min_by(|&a, &b| {
                    dist2(feature, &centroids[a]).total_cmp(&dist2(feature, &centroids[b]))
                })
                .unwrap();
        }
        let mut sums = vec![vec![0.0; dim]; k];
        let mut counts = vec![0usize; k];
        for (feature, &cluster) in features.iter().zip(&assignment) {
            counts[cluster] += 1;
            for i in 0..dim {
                sums[cluster][i] += feature[i];
            }
        }
        for cluster in 0..k {
            if counts[cluster] > 0 {
                for i in 0..dim {
                    centroids[cluster][i] = sums[cluster][i] / counts[cluster] as f64;
                }
            }
        }
    }
    (centroids, assignment)
}

fn stratum(record: &Value, profile: &Profile, field: &str, preserve_lengths: bool) -> String {
    let category = record
        .get(field)
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    if preserve_lengths {
        format!("{category}|L{}", profile.sequence_length)
    } else {
        category.to_string()
    }
}
fn prefix_quotas(strata: &[String], budget: usize) -> BTreeMap<String, usize> {
    let mut quotas = BTreeMap::<String, usize>::new();
    for key in strata.iter().take(budget) {
        *quotas.entry(key.clone()).or_default() += 1;
    }
    quotas
}
fn balance_gain(candidate: &[u64], current: &[u64]) -> f64 {
    let total: u64 = candidate.iter().sum();
    if total == 0 {
        return 0.0;
    }
    candidate
        .iter()
        .zip(current)
        .map(|(&count, &seen)| count as f64 / (1.0 + seen as f64).sqrt())
        .sum::<f64>()
        / total as f64
}
fn coverage_metrics(counts: &[u64]) -> Value {
    if counts.is_empty() {
        return json!({"kind":"dense"});
    }
    let mut sorted = counts.to_vec();
    sorted.sort_unstable();
    let mean = sorted.iter().sum::<u64>() as f64 / sorted.len() as f64;
    let variance = sorted
        .iter()
        .map(|&value| (value as f64 - mean).powi(2))
        .sum::<f64>()
        / sorted.len() as f64;
    json!({"routes": sorted.iter().sum::<u64>(), "min": sorted[0], "median": sorted[sorted.len()/2],
           "max": sorted[sorted.len()-1], "zero": sorted.iter().filter(|&&value| value == 0).count(),
           "coefficient_of_variation": if mean > 0.0 { variance.sqrt()/mean } else { 0.0 }})
}
fn sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn main() -> Result<()> {
    let args = parse_args()?;
    if args.output.exists() {
        bail!("refusing to overwrite {}", args.output.display());
    }
    if args.nsamples == 0 {
        bail!("--nsamples must be positive");
    }
    for (name, value) in [
        ("cola", args.cola_weight),
        ("acdm", args.acdm_weight),
        ("expert", args.expert_weight),
    ] {
        if !value.is_finite() || value < 0.0 {
            bail!("{name} weight must be finite and non-negative");
        }
    }
    let candidate_lines = load_lines(&args.candidates)?;
    let records: Vec<Value> = candidate_lines
        .iter()
        .enumerate()
        .map(|(line, raw)| {
            serde_json::from_str(raw).with_context(|| {
                format!(
                    "{}:{}: invalid candidate",
                    args.candidates.display(),
                    line + 1
                )
            })
        })
        .collect::<Result<_>>()?;
    let profiles = load_profiles(&args.profiles)?;
    if candidate_lines.len() != profiles.len() {
        bail!(
            "{} candidates but {} profiles",
            candidate_lines.len(),
            profiles.len()
        );
    }
    if args.nsamples > profiles.len() {
        bail!(
            "cannot select {} from {} candidates",
            args.nsamples,
            profiles.len()
        );
    }
    for (index, profile) in profiles.iter().enumerate() {
        if profile.sample_index != index {
            bail!(
                "profile row {index} identifies candidate {}",
                profile.sample_index
            );
        }
        let token_len = records[index]["input_ids"]
            .as_array()
            .context("candidate missing input_ids")?
            .len();
        if token_len != profile.sequence_length {
            bail!("candidate/profile length mismatch at row {index}");
        }
    }
    let (features, _, _) = standardize(profiles.iter().map(cola_feature).collect())?;
    let (acdm_features, acdm_mean, acdm_scale_by_dim) =
        standardize(profiles.iter().map(acdm_feature).collect())?;
    let reference_owned;
    let (reference, reference_kind) = if let Some(path) = &args.reference_profiles {
        reference_owned = load_profiles(path)?;
        (
            &reference_owned[..],
            format!("task_reference:{}", path.display()),
        )
    } else {
        (&profiles[..], "candidate_pool_centroid".to_string())
    };
    let reference_centroid =
        transformed_centroid(reference, &acdm_mean, &acdm_scale_by_dim, acdm_feature)?;
    let acdm: Vec<f64> = acdm_features
        .iter()
        .map(|feature| dist2(feature, &reference_centroid).sqrt())
        .collect();
    let acdm_scale = acdm.iter().copied().fold(0.0f64, f64::max).max(1e-9);
    println!(
        "[select] k-means: {} candidates -> {} COLA clusters, dim {}, {} iterations",
        profiles.len(),
        args.nsamples,
        features[0].len(),
        args.iters
    );
    let (centroids, assignment) = kmeans(&features, args.nsamples, args.iters, args.seed);
    let mut clusters = vec![Vec::new(); args.nsamples];
    for (index, &cluster) in assignment.iter().enumerate() {
        clusters[cluster].push(index);
    }
    let strata: Vec<String> = records
        .iter()
        .zip(&profiles)
        .map(|(record, profile)| {
            stratum(record, profile, &args.preserve_field, args.preserve_lengths)
        })
        .collect();
    // The composer's consumed prefix is the exact target recipe. Search the entire reserve pool,
    // but preserve that prefix's category × MaCa-length counts exactly.
    let quotas = prefix_quotas(&strata, args.nsamples);
    let mut remaining = quotas.clone();
    let experts: Vec<Vec<u64>> = profiles.iter().map(expert_feature).collect();
    let expert_dim = experts.first().map(Vec::len).unwrap_or(0);
    if experts.iter().any(|expert| expert.len() != expert_dim) {
        bail!("inconsistent expert route dimensions");
    }
    let mut coverage = vec![0u64; expert_dim];
    let mut selected = Vec::with_capacity(args.nsamples);
    let mut used = vec![false; profiles.len()];
    let mut cluster_order: Vec<usize> = (0..clusters.len()).collect();
    cluster_order.sort_by_key(|&cluster| clusters[cluster].len());
    for cluster in cluster_order {
        if clusters[cluster].is_empty() {
            continue;
        }
        let choice = clusters[cluster]
            .iter()
            .copied()
            .min_by(|&a, &b| {
                let score = |index: usize| {
                    let quota_penalty = if remaining.get(&strata[index]).copied().unwrap_or(0) == 0
                    {
                        1e6
                    } else {
                        0.0
                    };
                    quota_penalty
                        + args.cola_weight * dist2(&features[index], &centroids[cluster]).sqrt()
                        + args.acdm_weight * acdm[index] / acdm_scale
                        - args.expert_weight * balance_gain(&experts[index], &coverage)
                };
                score(a).total_cmp(&score(b)).then_with(|| a.cmp(&b))
            })
            .unwrap();
        selected.push((choice, cluster));
        used[choice] = true;
        if let Some(left) = remaining.get_mut(&strata[choice]) {
            *left = left.saturating_sub(1);
        }
        for (dst, &count) in coverage.iter_mut().zip(&experts[choice]) {
            *dst += count;
        }
    }
    // Empty k-means clusters are possible. Fill them with the best still-unused candidates.
    while selected.len() < args.nsamples {
        let choice = (0..profiles.len())
            .filter(|&index| !used[index])
            .min_by(|&a, &b| {
                let score = |index: usize| {
                    (if remaining.get(&strata[index]).copied().unwrap_or(0) == 0 {
                        1e6
                    } else {
                        0.0
                    }) + args.acdm_weight * acdm[index] / acdm_scale
                        - args.expert_weight * balance_gain(&experts[index], &coverage)
                };
                score(a).total_cmp(&score(b)).then_with(|| a.cmp(&b))
            })
            .unwrap();
        selected.push((choice, usize::MAX));
        used[choice] = true;
        if let Some(left) = remaining.get_mut(&strata[choice]) {
            *left = left.saturating_sub(1);
        }
        for (dst, &count) in coverage.iter_mut().zip(&experts[choice]) {
            *dst += count;
        }
    }
    // Exact quota repair: swap an overrepresented selected row for the closest unused row from an
    // underrepresented stratum. This preserves the COLA geometry as much as possible.
    loop {
        let mut actual = BTreeMap::<String, usize>::new();
        for &(index, _) in &selected {
            *actual.entry(strata[index].clone()).or_default() += 1;
        }
        let under = quotas
            .iter()
            .find(|(key, target)| actual.get(*key).copied().unwrap_or(0) < **target)
            .map(|(key, _)| key.clone());
        let Some(under) = under else { break };
        let replacement = (0..profiles.len())
            .filter(|&index| !used[index] && strata[index] == under)
            .min_by(|&a, &b| acdm[a].total_cmp(&acdm[b]).then_with(|| a.cmp(&b)))
            .context("quota repair has no candidate")?;
        let slot = selected
            .iter()
            .enumerate()
            .filter(|(_, (index, _))| actual[&strata[*index]] > quotas[&strata[*index]])
            .min_by(|(_, (a, _)), (_, (b, _))| {
                dist2(&features[replacement], &features[*a])
                    .total_cmp(&dist2(&features[replacement], &features[*b]))
            })
            .map(|(slot, _)| slot)
            .context("quota repair has no overrepresented stratum")?;
        used[selected[slot].0] = false;
        used[replacement] = true;
        selected[slot] = (replacement, usize::MAX);
    }
    coverage.fill(0);
    for &(index, _) in &selected {
        for (dst, &count) in coverage.iter_mut().zip(&experts[index]) {
            *dst += count;
        }
    }
    let mut writer = BufWriter::new(File::create(&args.output)?);
    for (new_index, &(source_index, cluster)) in selected.iter().enumerate() {
        let mut record = records[source_index].clone();
        record["sample_index"] = json!(new_index);
        record["selection"] = json!({"source_sample_index": source_index, "cola_cluster": if cluster == usize::MAX { Value::Null } else { json!(cluster) },
                                      "acdm_distance": acdm[source_index]});
        serde_json::to_writer(&mut writer, &record)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    drop(writer);
    let mut actual = BTreeMap::<String, usize>::new();
    for &(index, _) in &selected {
        *actual.entry(strata[index].clone()).or_default() += 1;
    }
    let manifest = json!({
        "format": "veloGB10-calibration-selection-v1", "method": "COLA_kmeans_plus_ACDM_plus_MoE_balance",
        "candidates": std::fs::canonicalize(&args.candidates)?, "profiles": std::fs::canonicalize(&args.profiles)?,
        "reference": reference_kind, "output": args.output, "sha256": sha256(&args.output)?,
        "candidate_count": profiles.len(), "selected_count": selected.len(),
        "feature_dimensions": {"cola_count_sketch": features[0].len(), "acdm_mean_std": acdm_features[0].len()},
        "weights": {"cola": args.cola_weight, "acdm": args.acdm_weight, "expert_balance": args.expert_weight},
        "kmeans": {"clusters": args.nsamples, "iterations": args.iters, "seed": args.seed},
        "preservation": {"field": args.preserve_field, "lengths": args.preserve_lengths, "target": quotas, "actual": actual},
        "expert_coverage": coverage_metrics(&coverage),
    });
    let manifest_path = PathBuf::from(format!("{}.manifest.json", args.output.display()));
    std::fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
    println!(
        "[select] wrote {} selected rows -> {}",
        selected.len(),
        args.output.display()
    );
    println!("[select] expert coverage: {}", coverage_metrics(&coverage));
    println!("[select] manifest: {}", manifest_path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quotas_sum_exactly() {
        let strata = ["a", "a", "a", "b", "b", "c"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let quotas = prefix_quotas(&strata, 4);
        assert_eq!(quotas.values().sum::<usize>(), 4);
        assert_eq!(quotas["a"], 3);
    }
    #[test]
    fn balance_rewards_unseen_expert() {
        assert!(balance_gain(&[0, 10], &[100, 0]) > balance_gain(&[10, 0], &[100, 0]));
    }
}
