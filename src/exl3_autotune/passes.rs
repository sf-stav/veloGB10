//! The `--autotune --selftest` corpus and passes (MTP and plain generation), the idle-box preflight and the GPU environment snapshot.
//! Part of the `--autotune` core (see `mod.rs` for the file map).

use super::*;

pub(super) const CORPUS: [&str; 2] = [
    "Write a complete ANSI C program that reads a text file named on the command line, counts how \
     many times each word occurs using a hash table with separate chaining, and prints the twenty \
     most frequent words with their counts. Include all necessary headers, check every allocation \
     and file operation, and comment each function.\n\n```c\n#include <stdio.h>\n",
    "The history of the printing press is a story about far more than machines. When Johannes \
     Gutenberg began experimenting with movable metal type in Mainz around 1440, he could not have \
     imagined how thoroughly his invention would reshape European society. In the decades that \
     followed,",
];

pub(super) struct PassOut {
    pub(super) tokens: Vec<u32>,
    pub(super) rounds: usize,
    pub(super) drafted: usize,
    pub(super) accepted: usize,
    pub(super) accepts: Vec<usize>,
}

pub(super) fn fnv64(v: &[u32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &t in v {
        for b in t.to_le_bytes() { h ^= b as u64; h = h.wrapping_mul(0x0000_0100_0000_01b3); }
    }
    h
}

/// Common state before every pass: zeroed positional planes + taps (so round 1's tap restore reads
/// the same bytes in every pass), a reset slot, greedy unpenalized rows, then the served prefill
/// (chunked prompt[..n-1] + the seam decode step) and the served head prime.
pub(super) fn pass_setup(model: &FwdModel, sc: &mut Scratch, ids: &[u32], max_new: usize, chunk: usize,
              psc: &mut Option<PrefillScratch>) -> Result<u32> {
    let span = ids.len() + max_new + 2 * (MTP_MAX_K + 2) + 4;
    zero_positional(model, 0, span)?;
    model.reset_slot(0)?;
    {
        let l = Launcher { dev: &model.dev, stream: &model.stream };
        let n = sc.taps_keep.len();
        xqlaunch!(l, "xq_memset_f32", ((((n + 255) / 256).max(1)) as u32, 1, 1), (256, 1, 1), 0,
                  (&sc.taps_keep, 0i64, n as i64))?;
    }
    let rows: Vec<Option<SampParams>> = vec![None; MTP_MAX_K + 1];
    model.set_sampling(sc, &rows)?;
    let pens: Vec<Option<PenParams>> = vec![None; MTP_MAX_K + 1];
    model.set_penalties(sc, &pens)?;
    model.prefill_prompt(sc, ids, Some(chunk), psc)?;
    let pred0 = model.dev.dtoh_sync_copy(&sc.argmax)?[0] as u32;
    Ok(pred0)
}

/// One greedy fixed-depth (k = MTP_MAX_K) MTP generation through the SERVED mtp_round — the tuning
/// trajectory of §5.2. SR runs inside when a session is armed.
pub(super) fn run_mtp_pass(model: &FwdModel, sc: &mut Scratch, head: &DraftHead, ids: &[u32], max_new: usize,
                chunk: usize, psc: &mut Option<PrefillScratch>) -> Result<PassOut> {
    let plen = ids.len();
    let pred0 = pass_setup(model, sc, ids, max_new, chunk, psc)?;
    // the served WP03 seam prime (tap snapshot into slot 0's window + the head's row plen-1)
    model.mtp_seam_prime(sc, psc.as_ref(), head, *ids.last().unwrap() as i32, plen, 0)?;
    let mut out = PassOut { tokens: vec![pred0], rounds: 0, drafted: 0, accepted: 0, accepts: Vec::new() };
    let (mut b, mut p, mut tap_row) = (pred0 as i32, plen, 0usize);
    let k = MTP_MAX_K;
    while !EOS_IDS.contains(&b) && out.tokens.len() < max_new && p + k + 2 < model.max_pos {
        let r = model.mtp_round(sc, head, 0, k, b, p, tap_row)?;
        out.rounds += 1;
        out.drafted += r.drafts.len();
        out.accepted += r.a;
        out.accepts.push(r.a);
        out.tokens.extend(r.emitted.iter().map(|&e| e as u32));
        b = r.last_tok;
        p = r.pos_next;
        tap_row = r.tap_row;
    }
    Ok(out)
}

/// Plain greedy decode (LI segment): the served seam, then `steps` served forward_step calls.
pub(super) fn run_plain_pass(model: &FwdModel, sc: &mut Scratch, ids: &[u32], steps: usize, chunk: usize,
                  psc: &mut Option<PrefillScratch>, li: bool) -> Result<Vec<u32>> {
    let plen = ids.len();
    let pred0 = pass_setup(model, sc, ids, steps, chunk, psc)?;
    let mut toks = vec![pred0];
    if li { ARMED.store(true, Ordering::SeqCst); }
    let r = (|| -> Result<()> {
        for i in 0..steps.saturating_sub(1) {
            let t = *toks.last().unwrap() as i32;
            if EOS_IDS.contains(&t) || plen + i + 1 >= model.max_pos { break; }
            let ids = model.forward_step(sc, &[t], &[plen + i], &[0], None)?;
            toks.push(ids[0] as u32);
        }
        Ok(())
    })();
    ARMED.store(false, Ordering::SeqCst);
    r?;
    Ok(toks)
}

pub(super) fn arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    // F9: delegate to the ONE shared `=`-aware parser (S1 pattern; exl3_serve.rs does the same)
    // so `--autotune-seed=7` behaves exactly like `--autotune-seed 7` here too.
    crate::arg_value(args, name)
}

pub(super) fn local_ipv4s() -> Vec<String> {
    let mut out = Vec::new();
    unsafe {
        let mut ifap: *mut libc::ifaddrs = std::ptr::null_mut();
        if libc::getifaddrs(&mut ifap) != 0 { return out; }
        let mut p = ifap;
        while !p.is_null() {
            let a = (*p).ifa_addr;
            if !a.is_null() && (*a).sa_family as i32 == libc::AF_INET {
                let sin = a as *const libc::sockaddr_in;
                out.push(std::net::Ipv4Addr::from(u32::from_be((*sin).sin_addr.s_addr)).to_string());
            }
            p = (*p).ifa_next;
        }
        libc::freeifaddrs(ifap);
    }
    out
}

/// Other gb10_inference processes on this box (/proc/*/comm; no spawn).
pub(super) fn other_engines() -> Vec<u32> {
    let me = std::process::id();
    let mut v = Vec::new();
    if let Ok(rd) = std::fs::read_dir("/proc") {
        for e in rd.flatten() {
            let Ok(pid) = e.file_name().to_string_lossy().parse::<u32>() else { continue };
            if pid == me { continue; }
            if let Ok(c) = std::fs::read_to_string(e.path().join("comm")) {
                if c.trim() == "gb10_inference" { v.push(pid); }
            }
        }
    }
    v
}

/// Resident compute apps other than this process (None = nvidia-smi unusable: cannot prove idle).
pub(super) fn resident_apps() -> Option<Vec<String>> {
    let me = std::process::id().to_string();
    let out = tune::run_timeout("nvidia-smi", &["--query-compute-apps=pid,process_name,used_memory",
                                                "--format=csv,noheader"], std::time::Duration::from_secs(10))?;
    Some(out.lines().map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty() && !l.contains("No running") && l.split(',').next().map(|p| p.trim()) != Some(me.as_str()))
        .collect())
}

/// §6.4.3-4 / §12 G-T8: only on an idle GPU (no other engine, no resident compute app). Err = the refusal reason.
pub(super) fn preflight() -> std::result::Result<(), String> {
    let others = other_engines();
    if !others.is_empty() {
        return Err(format!("other gb10_inference processes are running (pids {others:?}) — the GPU is not idle"));
    }
    match resident_apps() {
        None => Err("nvidia-smi --query-compute-apps failed — cannot prove the GPU is idle".into()),
        Some(a) if !a.is_empty() => Err(format!("resident compute apps: {a:?}")),
        Some(_) => Ok(()),
    }
    .and_then(|_| {
        for (v, why) in [(crate::opt!("exl3-no-graph"), "graphs off"), (crate::opt!("mtp-dump"), "dump mode"),
                         (crate::opt!("exl3-gdn-ring-check"), "eager ring check"), (crate::opt!("wp12-xcheck"), "eager XCHECK"),
                         (crate::opt!("exl3-draft-graph"), "draft graphs toggled")] {
            if crate::opts::var_os(v).is_some() {
                return Err(format!("{v} is set ({why}) — the harness measures the served graphed path"));
            }
        }
        Ok(())
    })
}

/// nvidia-smi clocks / thermal snapshot (unsupported fields read "unknown", never zero).
pub(super) fn gpu_env(phase: &str) -> serde_json::Value {
    let q = |fields: &str| tune::run_timeout("nvidia-smi", &[&format!("--query-gpu={fields}"),
                                                              "--format=csv,noheader,nounits"],
                                             std::time::Duration::from_secs(5));
    let full = "clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_event_reasons.active";
    let (names, out): (Vec<&str>, Option<String>) = match q(full) {
        Some(o) => (full.split(',').collect(), Some(o)),
        None => { let f = "clocks.sm,clocks.mem,temperature.gpu,power.draw"; (f.split(',').collect(), q(f)) }
    };
    let mut m = serde_json::Map::new();
    m.insert("phase".into(), phase.into());
    let vals: Vec<String> = out.as_deref().and_then(|o| o.lines().next())
        .map(|l| l.split(',').map(|x| x.trim().to_string()).collect()).unwrap_or_default();
    for (i, n) in names.iter().enumerate() {
        let v = vals.get(i).cloned().unwrap_or_default();
        let v = if v.is_empty() || v.contains("N/A") || v.contains("Not Supported") { "unknown".to_string() } else { v };
        m.insert(n.to_string(), v.into());
    }
    serde_json::Value::Object(m)
}
