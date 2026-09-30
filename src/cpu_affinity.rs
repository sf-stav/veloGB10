//! HOST / RO-7 (PLAN/SURPASS_PLAN_2026-09-26.md): keep the serving engine's latency-critical host
//! threads on the GB10's big cores.
//!
//! GB10's CPU is 10 Cortex-X925 (3.9 GHz, `cpu_capacity` 997-1024) + 10 Cortex-A725 (2.8 GHz,
//! capacity 718-731), in two clusters (CPUs 5-9 and 15-19 are the X925s on every box measured).
//! Left to EAS, a thread can land on an A725: the per-pass DDS readback / PLE row work then runs
//! ~1.5-1.9x slower (PLE 0.35 -> 0.66 ms per 6 rows) and the tokio HTTP thread likewise. The rival
//! runs its whole process under `taskset -c 5-9,15-19` (r8c start_handoff_endpoint.sh:80,
//! exl3_native_server.py:86). We pin only the threads that sit on the request / decode critical
//! path (the EXL3 scheduler thread and the HTTP runtime thread) to the SET of big cores — a set,
//! not one core, exactly like taskset — after the model load, so the load's own threads keep
//! every core. Threads those two spawn later (the prefill PLE workers, tokio's blocking pool)
//! inherit the mask.
//!
//! Big cores are DETECTED, never hard-coded: `cpu_capacity` (the kernel's per-CPU compute
//! capacity, 1024 = the biggest) when every online CPU exposes it, else
//! `cpufreq/cpuinfo_max_freq`; a CPU is "big" when its metric is >= 90% of the maximum. A
//! homogeneous machine (every CPU big) is left alone, and the set is intersected with the
//! process's current mask so an outer `taskset` / cgroup cpuset is respected.
//!
//! Scheduling only: which core a host thread runs on cannot change any computed value, so this
//! is bitwise-neutral by construction.

use anyhow::{Context, Result};

/// `--cpu-affinity auto|off` (default auto).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AffinityMode {
    Auto,
    Off,
}

impl AffinityMode {
    pub fn parse(v: Option<&str>) -> Result<Self> {
        match v {
            None | Some("auto") | Some("on") => Ok(Self::Auto),
            Some("off") => Ok(Self::Off),
            Some(other) => anyhow::bail!("--cpu-affinity must be auto or off (got '{other}')"),
        }
    }
}

/// The detected big-core set and how it was found (for the boot log line).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BigCores {
    pub cpus: Vec<usize>,
    /// "cpu_capacity" or "cpuinfo_max_freq"
    pub source: &'static str,
    /// the metric cut (>= this is big) and the maximum it was derived from
    pub cut: u64,
    pub max: u64,
    /// online CPUs considered
    pub online: usize,
}

/// Parse a kernel cpu list ("0-4,10-14", "3", "") into sorted, deduplicated ids.
pub fn parse_cpu_list(s: &str) -> Option<Vec<usize>> {
    let mut v = Vec::new();
    for part in s.trim().split(',').map(str::trim).filter(|p| !p.is_empty()) {
        match part.split_once('-') {
            Some((a, b)) => {
                let (a, b): (usize, usize) = (a.trim().parse().ok()?, b.trim().parse().ok()?);
                if b < a || b - a > 4096 { return None; }
                v.extend(a..=b);
            }
            None => v.push(part.parse().ok()?),
        }
    }
    v.sort_unstable();
    v.dedup();
    Some(v)
}

/// Format ids as a compact kernel cpu list ("5-9,15-19").
pub fn format_cpu_list(cpus: &[usize]) -> String {
    let mut v = cpus.to_vec();
    v.sort_unstable();
    v.dedup();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < v.len() {
        let mut j = i;
        while j + 1 < v.len() && v[j + 1] == v[j] + 1 { j += 1; }
        out.push(if j == i { v[i].to_string() } else { format!("{}-{}", v[i], v[j]) });
        i = j + 1;
    }
    out.join(",")
}

/// Classify (cpu, metric) pairs: big = metric >= 90% of the max. None when the set is empty or
/// covers every CPU (homogeneous: nothing to pin to).
pub fn classify_big(metrics: &[(usize, u64)]) -> Option<(Vec<usize>, u64, u64)> {
    let max = metrics.iter().map(|&(_, m)| m).max()?;
    if max == 0 { return None; }
    let cut = max - max / 10; // 90% of the max, integer (1024 -> 922; 3900000 -> 3510000)
    let big: Vec<usize> = metrics.iter().filter(|&&(_, m)| m >= cut).map(|&(c, _)| c).collect();
    if big.is_empty() || big.len() == metrics.len() { return None; }
    Some((big, cut, max))
}

fn read_u64(path: &str) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

/// Detect the big cores from sysfs (see the module doc). None = not a big.LITTLE machine, or
/// sysfs does not say.
pub fn detect_big_cores() -> Option<BigCores> {
    let online = parse_cpu_list(&std::fs::read_to_string("/sys/devices/system/cpu/online").ok()?)?;
    if online.is_empty() { return None; }
    for (source, leaf) in [("cpu_capacity", "cpu_capacity"), ("cpuinfo_max_freq", "cpufreq/cpuinfo_max_freq")] {
        let metrics: Option<Vec<(usize, u64)>> = online.iter()
            .map(|&c| read_u64(&format!("/sys/devices/system/cpu/cpu{c}/{leaf}")).map(|m| (c, m)))
            .collect();
        if let Some(m) = metrics {
            let (cpus, cut, max) = classify_big(&m)?;
            return Some(BigCores { cpus, source, cut, max, online: online.len() });
        }
    }
    None
}

/// The calling thread's current affinity mask.
pub fn current_thread_affinity() -> Result<Vec<usize>> {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        let r = libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set);
        anyhow::ensure!(r == 0, "sched_getaffinity: {}", std::io::Error::last_os_error());
        Ok((0..libc::CPU_SETSIZE as usize).filter(|&c| libc::CPU_ISSET(c, &set)).collect())
    }
}

/// Pin the CALLING thread (Linux: pid 0 = this thread) to `cpus` and verify the read-back.
pub fn pin_current_thread(cpus: &[usize]) -> Result<()> {
    anyhow::ensure!(!cpus.is_empty(), "empty cpu set");
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        for &c in cpus {
            anyhow::ensure!(c < libc::CPU_SETSIZE as usize, "cpu {c} out of range");
            libc::CPU_SET(c, &mut set);
        }
        let r = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
        anyhow::ensure!(r == 0, "sched_setaffinity({}): {}", format_cpu_list(cpus),
                        std::io::Error::last_os_error());
    }
    let got = current_thread_affinity()?;
    let mut want = cpus.to_vec();
    want.sort_unstable();
    want.dedup();
    anyhow::ensure!(got == want, "affinity read-back {} != requested {}",
                    format_cpu_list(&got), format_cpu_list(&want));
    Ok(())
}

/// Resolve `--cpu-affinity` into the cpu set to pin the critical threads to (None = leave the
/// scheduler alone), with the one boot line that says what was chosen and why. Called on the
/// main thread BEFORE it pins itself, so the current mask is the process's launch mask.
pub fn resolve(mode: AffinityMode) -> Result<(Option<Vec<usize>>, String)> {
    if mode == AffinityMode::Off {
        return Ok((None, "cpu affinity OFF (--cpu-affinity off): host threads left to the OS scheduler".into()));
    }
    let Some(bc) = detect_big_cores() else {
        return Ok((None, "cpu affinity auto: no big.LITTLE split found in sysfs (cpu_capacity / \
                          cpuinfo_max_freq) — host threads left to the OS scheduler".into()));
    };
    let allowed = current_thread_affinity().context("reading the launch cpu mask")?;
    let cpus: Vec<usize> = bc.cpus.iter().copied().filter(|c| allowed.contains(c)).collect();
    if cpus.is_empty() {
        return Ok((None, format!("cpu affinity auto: big cores {} are outside the launch mask {} — \
                                  host threads left as launched", format_cpu_list(&bc.cpus),
                                 format_cpu_list(&allowed))));
    }
    let note = format!("cpu affinity auto: scheduler + HTTP threads -> big cores {} ({} of {} online; \
                        {} >= {} of max {}{}); --cpu-affinity off disables",
                       format_cpu_list(&cpus), cpus.len(), bc.online, bc.source, bc.cut, bc.max,
                       if cpus.len() < bc.cpus.len() { format!(", launch mask {}", format_cpu_list(&allowed)) }
                       else { String::new() });
    Ok((Some(cpus), note))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_list_roundtrip() {
        assert_eq!(parse_cpu_list("0-4,10-14").unwrap(), vec![0, 1, 2, 3, 4, 10, 11, 12, 13, 14]);
        assert_eq!(parse_cpu_list("5-9,15-19\n").unwrap(), vec![5, 6, 7, 8, 9, 15, 16, 17, 18, 19]);
        assert_eq!(parse_cpu_list("3").unwrap(), vec![3]);
        assert_eq!(parse_cpu_list("").unwrap(), Vec::<usize>::new());
        assert!(parse_cpu_list("4-2").is_none());
        assert!(parse_cpu_list("x").is_none());
        assert_eq!(format_cpu_list(&[19, 5, 6, 7, 8, 9, 15, 16, 17, 18]), "5-9,15-19");
        assert_eq!(format_cpu_list(&[0, 2, 3]), "0,2-3");
        assert_eq!(format_cpu_list(&[]), "");
    }

    #[test]
    fn classify_gb10_capacity() {
        // .11's sysfs (same GB10 SKU as .13/.14), 2026-09-26
        let cap = |c: usize| -> u64 {
            match c { 0..=4 => 718, 10..=14 => 731, 5..=9 => 997, 15..=18 => 1017, _ => 1024 }
        };
        let m: Vec<(usize, u64)> = (0..20).map(|c| (c, cap(c))).collect();
        let (big, cut, max) = classify_big(&m).unwrap();
        assert_eq!(big, vec![5, 6, 7, 8, 9, 15, 16, 17, 18, 19]);
        assert_eq!((cut, max), (922, 1024));
        // max-freq fallback
        let f: Vec<(usize, u64)> = (0..20).map(|c| (c, if cap(c) > 900 { 3_900_000 } else { 2_808_000 })).collect();
        assert_eq!(classify_big(&f).unwrap().0, big);
        // homogeneous -> nothing to pin
        assert!(classify_big(&[(0, 1024), (1, 1024)]).is_none());
        assert!(classify_big(&[]).is_none());
    }

    #[test]
    fn detect_on_this_box() {
        // CPU-only: whatever this box is, the auto resolution is self-consistent (on a GB10 it
        // prints the 10 X925s, 5-9,15-19).
        let (cpus, note) = resolve(AffinityMode::Auto).unwrap();
        println!("{note}");
        if let Some(c) = cpus {
            let allowed = current_thread_affinity().unwrap();
            assert!(!c.is_empty() && c.iter().all(|x| allowed.contains(x)));
        }
        assert!(resolve(AffinityMode::Off).unwrap().0.is_none());
        assert!(AffinityMode::parse(Some("bogus")).is_err());
        assert_eq!(AffinityMode::parse(None).unwrap(), AffinityMode::Auto);
    }

    #[test]
    fn pin_roundtrip_on_this_box() {
        // CPU-only: pin this test thread to its own current mask (always valid), read it back.
        let cur = current_thread_affinity().unwrap();
        assert!(!cur.is_empty());
        pin_current_thread(&cur).unwrap();
        assert_eq!(current_thread_affinity().unwrap(), cur);
    }
}

/// TP-D: the mask host worker threads re-pin to when their creator is pinned to ONE core (the TP
/// launch thread, core 9): a spawned thread inherits its creator's mask, so without this the
/// prefill PLE row workers of a TP rank would all serialize on the launch core. Unset (TP=1) =
/// `pin_worker` is a no-op and workers keep the inherited mask, exactly as before.
static WORKER_MASK: std::sync::OnceLock<Vec<usize>> = std::sync::OnceLock::new();

pub fn set_worker_mask(cpus: Vec<usize>) {
    let _ = WORKER_MASK.set(cpus);
}

/// Called at the top of a spawned host worker: re-pin to the worker mask when one is set.
pub fn pin_worker() {
    if let Some(m) = WORKER_MASK.get() {
        let _ = pin_current_thread(m);
    }
}
