//! `--dsv41-engram-verify`: spot-check packed records against the native source tables.

use super::manifest::*;
use super::pack::parse_heads;
use super::source;
use anyhow::{anyhow, bail, Result};
use rand::{rngs::StdRng, Rng, SeedableRng};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

pub fn sha256_file(p: &std::path::Path) -> Result<String> {
    let mut f = File::open(p)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

pub fn run_cli(args: &[String]) -> Result<()> {
    let arg = |k: &str| crate::arg_value(args, k).map(|s| s.to_string());
    let dir = PathBuf::from(arg("--engram-dir").or_else(|| arg("--out")).ok_or_else(|| anyhow!("--engram-dir <pack dir> required"))?);
    let sha_only = args.iter().any(|a| a == "--engram-verify-sha-only");
    let want_sha = sha_only || args.iter().any(|a| a == "--engram-verify-sha");
    let source_dir = arg("--engram-source").map(PathBuf::from);
    if !sha_only && source_dir.is_none() {
        bail!("--engram-source <native model dir> required (or --engram-verify-sha-only)");
    }
    let rows: u64 = arg("--engram-verify-rows").and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let threads: usize = arg("--engram-verify-threads").and_then(|s| s.parse().ok()).unwrap_or(16);
    let src = match &source_dir {
        Some(d) => Some(source::load(d)?),
        None => None,
    };
    let heads = parse_heads(&arg("--engram-heads").unwrap_or_else(|| "0-7".into()), 8)?;
    let mut all_ok = true;
    for h in heads {
        let ph = match PackedHead::open(&dir, h) {
            Ok(p) => p,
            Err(e) => {
                println!("head {h}: SKIP ({e})");
                continue;
            }
        };
        let m = &ph.manifest;
        let t0 = std::time::Instant::now();
        let mut ok = true;
        if want_sha {
            let s = sha256_file(&ph.path)?;
            let good = s == m.sha256;
            println!("head {h}: sha256 {} ({:.0}s)", if good { "OK" } else { "MISMATCH" }, t0.elapsed().as_secs_f64());
            ok &= good;
        }
        if let (Some(src), Some(sd)) = (&src, &source_dir) {
            if src.geometry != m.geometry {
                bail!("head {h}: manifest geometry differs from the source config");
            }
            let files = source::SourceFiles::open(sd, &src.layers)?;
            let packed = File::open(&ph.path)?;
            let bad = AtomicU64::new(0);
            let checked = AtomicU64::new(0);
            let per = rows.div_ceil(threads as u64);
            std::thread::scope(|sc| {
                for t in 0..threads {
                    let (ph, files, src, packed, bad, checked) = (&ph, &files, src, &packed, &bad, &checked);
                    sc.spawn(move || {
                        let mut rng = StdRng::seed_from_u64(0x5eed_0000 + (h as u64) * 1000 + t as u64);
                        let n_sec = ph.manifest.sections.len();
                        let (mut got, mut w, mut s) = ([0u8; RECORD_BYTES], [0u8; VALUE_BYTES], [0u8; SCALE_BYTES]);
                        let mut jobs: Vec<(usize, u64)> = vec![];
                        if t == 0 {
                            for (i, sec) in ph.manifest.sections.iter().enumerate() {
                                jobs.extend([(i, 0), (i, sec.rows - 1), (i, sec.rows / 2)]);
                            }
                        }
                        for _ in 0..per {
                            let i = rng.gen_range(0..n_sec);
                            jobs.push((i, rng.gen_range(0..ph.manifest.sections[i].rows)));
                        }
                        for (k, (si, row)) in jobs.into_iter().enumerate() {
                            let sec = &ph.manifest.sections[si];
                            let off = ph.record_offset(sec.layer_idx, sec.g, row).unwrap();
                            let mut good = packed.read_exact_at(&mut got, off).is_ok();
                            let sl = &src.layers[sec.layer_idx];
                            let r = sec.source_row_start + row;
                            good &= files.weight[sec.layer_idx].read_exact_at(&mut w, sl.weight.abs_offset + r * VALUE_BYTES as u64).is_ok();
                            good &= files.scale[sec.layer_idx].read_exact_at(&mut s, sl.scale.abs_offset + r * SCALE_BYTES as u64).is_ok();
                            good &= got[..VALUE_BYTES] == w[..] && got[VALUE_BYTES..] == s[..];
                            // page15: the 136 pad bytes after the 15th record of a page must be zero
                            if good && ph.manifest.layout == RecordLayout::Page15 && k % 64 == 0 {
                                let rec = sec.first_record + row;
                                let page_end = HEADER_BYTES + (rec / RECORDS_PER_PAGE) * PAGE_BYTES + PAGE_BYTES;
                                let mut pad = [0u8; 136];
                                good &= packed.read_exact_at(&mut pad, page_end - 136).is_ok() && pad.iter().all(|b| *b == 0);
                            }
                            checked.fetch_add(1, Ordering::Relaxed);
                            if !good {
                                bad.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    });
                }
            });
            let (c, b) = (checked.load(Ordering::Relaxed), bad.load(Ordering::Relaxed));
            println!("head {h}: {c} random rows checked vs source, {b} mismatches ({:.0}s)", t0.elapsed().as_secs_f64());
            ok &= b == 0 && c >= rows;
        }
        println!("head {h}: {}", if ok { "VERIFY PASS" } else { "VERIFY FAIL" });
        all_ok &= ok;
    }
    if !all_ok {
        bail!("verify failed");
    }
    Ok(())
}
