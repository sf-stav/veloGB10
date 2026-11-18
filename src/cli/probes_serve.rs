//! Serve-level probes: the pinned serve probe with its arm runner and tiny HTTP client helpers, and the
//! vision tower probe. They start servers in-process; never run them on a box held by another session.

use super::*;

/// B29 gate: serve-path probe at pinned MTP depth. See the dispatch-site comment.
pub(crate) fn run_probe_serve_pinned(args: &[String]) -> i32 {
    let model_dir = parse_arg(args, "--model-dir")
        .expect("--probe-serve-pinned requires --model-dir <DIR>").to_string();
    let max_seq_len: usize = gb10_inference::opt_or_arg(args, "--max-seq-len").and_then(|s| s.parse().ok()).unwrap_or(4096);
    let max_tokens: usize = parse_arg(args, "--max-new-tokens").and_then(|s| s.parse().ok()).unwrap_or(128);
    let port0: u16 = parse_arg(args, "--port").and_then(|s| s.parse().ok()).unwrap_or(8140);
    let prompt = parse_arg(args, "--prompt").unwrap_or(
        "What are the necessary preconditions for the possibility of the existence of logic? Answer the question and explain your answer within 1000 words.").to_string();
    let arms: Vec<(&str, Vec<&str>)> = vec![
        ("pinned-d2", vec!["--mtp=on", "--mtp-depth", "2"]),
        ("auto", vec!["--mtp=auto"]),
        ("pinned-d4", vec!["--mtp=on", "--mtp-depth", "4"]),
    ];
    let mut outputs: Vec<(String, String, u64)> = Vec::new();
    let mut failed = false;
    for (i, (label, mtp_args)) in arms.iter().enumerate() {
        let port = port0 + i as u16;
        match serve_probe_arm(&model_dir, mtp_args, max_seq_len, port, &prompt, max_tokens) {
            Ok((content, drafts)) => {
                println!("[serve-pinned] arm {label}: {} chars, completion ok, mtp_drafts={drafts}", content.len());
                outputs.push((label.to_string(), content, drafts));
            }
            Err(e) => {
                eprintln!("[serve-pinned] FAIL arm {label}: {e:#}");
                failed = true;
            }
        }
    }
    if failed { return 1; }
    for (label, content, _) in &outputs {
        if content.trim().is_empty() {
            eprintln!("[serve-pinned] FAIL arm {label}: EMPTY completion — an empty response is a FAIL, not a pass (AGENTS §3)");
            return 1;
        }
    }
    for (label, _, drafts) in &outputs {
        if label.starts_with("pinned") && *drafts == 0 {
            eprintln!("[serve-pinned] FAIL arm {label}: MTP did not engage on the pinned arm (drafts=0) — the gate did not exercise the speculative path");
            return 1;
        }
    }
    let reference = outputs[1].1.clone();
    for (label, content, _) in &outputs {
        if *content != reference {
            eprintln!("[serve-pinned] FAIL arm {label}: greedy output differs from the auto-depth reference — pinned depth must never change OUTPUT (AGENTS §2.7)");
            return 1;
        }
    }
    println!("[serve-pinned] PASS: pinned-d2 == auto == pinned-d4, {} chars byte-identical, MTP engaged on pinned arms, clean completions", reference.len());
    0
}

/// Boot THIS binary as a server with the given MTP flags, wait for /health, send ONE greedy chat
/// request, return (content, mtp_drafts). Kills the child on the way out; server log at
/// <tmp>/gb10_serve_pinned_<port>.log.
fn serve_probe_arm(model_dir: &str, mtp_args: &[&str], max_seq_len: usize, port: u16,
                   prompt: &str, max_tokens: usize) -> Result<(String, u64), anyhow::Error> {
    use std::process::{Command, Stdio};
    let exe = std::env::current_exe()?;
    let log_path = std::env::temp_dir().join(format!("gb10_serve_pinned_{port}.log"));
    let log = std::fs::File::create(&log_path)?;
    let mut child = Command::new(&exe)
        .args(["--server", "--model-dir", model_dir, "--thinking", "off",
               "--max-seq-len", &max_seq_len.to_string(), "--max-batch", "8",
               "--port", &port.to_string()])
        .args(mtp_args)
        .stdout(Stdio::from(log.try_clone()?))
        .stderr(Stdio::from(log))
        .spawn()
        .map_err(|e| anyhow::anyhow!("spawn server: {e}"))?;
    let result = (|| -> Result<(String, u64), anyhow::Error> {
        // Wait for /health (boot can take minutes on a cold page cache).
        let mut ready = false;
        for _ in 0..600 {
            if let Ok(resp) = http_req(port, "GET", "/health", None, 5) {
                if resp.contains(" 200 ") { ready = true; break; }
            }
            if let Some(status) = child.try_wait()? {
                anyhow::bail!("server exited early (status {status}); log {log_path:?}");
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        if !ready {
            anyhow::bail!("server never became healthy within 300s; log {log_path:?}");
        }
        let body = format!(
            "{{\"messages\":[{{\"role\":\"user\",\"content\":{}}}],\"max_tokens\":{max_tokens},\"temperature\":0}}",
            serde_json::to_string(prompt)?);
        let resp = http_req(port, "POST", "/v1/chat/completions", Some(&body), 300)?;
        if !resp.contains(" 200 ") {
            anyhow::bail!("chat request returned {}", resp.lines().next().unwrap_or("?"));
        }
        let j: serde_json::Value = serde_json::from_str(resp_body(&resp))
            .map_err(|e| anyhow::anyhow!("chat response is not JSON: {e}"))?;
        let content = j["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string();
        let health = http_req(port, "GET", "/health", None, 10)?;
        let hj: serde_json::Value = serde_json::from_str(resp_body(&health))
            .map_err(|e| anyhow::anyhow!("health response is not JSON: {e}"))?;
        let drafts = hj["telemetry"]["mtp"]["drafts"].as_u64().unwrap_or(0);
        Ok((content, drafts))
    })();
    let _ = child.kill();
    let _ = child.wait();
    result
}

/// Minimal HTTP/1.1 client (std only): returns "<status-line>\r\n\r\n<decoded-body>".
fn http_req(port: u16, method: &str, path: &str, body: Option<&str>, timeout_s: u64)
    -> Result<String, anyhow::Error> {
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::time::Duration;
    let mut s = TcpStream::connect(("127.0.0.1", port))
        .map_err(|e| anyhow::anyhow!("connect 127.0.0.1:{port}: {e}"))?;
    s.set_read_timeout(Some(Duration::from_secs(timeout_s)))?;
    s.set_write_timeout(Some(Duration::from_secs(30)))?;
    let req = match body {
        Some(b) => format!("{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len()),
        None => format!("{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"),
    };
    s.write_all(req.as_bytes())?;
    let mut raw = Vec::new();
    s.read_to_end(&mut raw)?;
    let text = String::from_utf8_lossy(&raw).to_string();
    let (head, b) = text.split_once("\r\n\r\n")
        .ok_or_else(|| anyhow::anyhow!("malformed HTTP response from {path}"))?;
    let status = head.lines().next().unwrap_or("HTTP/1.1 0 ?").to_string();
    let body = if head.to_ascii_lowercase().contains("transfer-encoding: chunked") {
        let mut out = String::new();
        let mut rest: &str = b;
        loop {
            let Some((size_line, after)) = rest.split_once("\r\n") else { break };
            let size = usize::from_str_radix(size_line.trim(), 16).unwrap_or(0);
            if size == 0 { break; }
            let end = after.len().min(size);
            out.push_str(&after[..end]);
            rest = &after[end..];
            rest = rest.strip_prefix("\r\n").unwrap_or(rest);
        }
        out
    } else {
        b.to_string()
    };
    Ok(format!("{status}\r\n\r\n{body}"))
}

fn resp_body(resp: &str) -> &str {
    resp.split_once("\r\n\r\n").map(|(_, b)| b).unwrap_or("")
}

/// VIS-1 `--probe-vision-tower` (see the dispatch comment in main).
pub(crate) fn probe_vision_tower(args: &[String]) -> anyhow::Result<()> {
    use anyhow::Context;
    let dir = parse_arg(args, "--model-dir").context("--probe-vision-tower requires --model-dir <DIR>")?;
    let t0 = std::time::Instant::now();
    let tower = gb10_inference::vision_tower::VisualTower::load(dir)?;
    let d = tower.dims;
    println!("VIS_LOAD ok in {:.1} s: hidden={} depth={} inter={} heads={} out={} blocks={} source={}",
             t0.elapsed().as_secs_f64(), d.hidden, d.depth, d.inter, d.heads, d.out_hidden, tower.blocks.len(), tower.source);
    let write = |path: &str, v: &[f32]| -> anyhow::Result<()> {
        std::fs::write(path, v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).with_context(|| path.to_string())
    };
    if let Some(wd) = parse_arg(args, "--dump-weights") {
        std::fs::create_dir_all(wd)?;
        let last = tower.blocks.len() - 1;
        let mats: Vec<(String, &[f32], usize, usize)> = vec![
            ("blocks.0.attn.proj".into(), &tower.blocks[0].proj_w, d.hidden, d.hidden),
            ("blocks.0.mlp.linear_fc1".into(), &tower.blocks[0].fc1_w, d.inter, d.hidden),
            ("blocks.0.mlp.linear_fc2".into(), &tower.blocks[0].fc2_w, d.hidden, d.inter),
            (format!("blocks.{last}.mlp.linear_fc2"), &tower.blocks[last].fc2_w, d.hidden, d.inter),
            ("merger.linear_fc1".into(), &tower.merger_fc1_w, d.merge_inter(), d.merge_inter()),
            ("merger.linear_fc2".into(), &tower.merger_fc2_w, d.out_hidden, d.merge_inter()),
        ];
        let mut man = Vec::new();
        for (name, w, o, i) in &mats {
            write(&format!("{wd}/{name}.f32"), w)?;
            man.push(serde_json::json!({"name": name, "out": o, "in": i}));
        }
        std::fs::write(format!("{wd}/manifest.json"), serde_json::to_string_pretty(&man)?)?;
        println!("VIS_DUMP_WEIGHTS {} matrices -> {wd}", mats.len());
    }
    if let Some(px) = parse_arg(args, "--pixels") {
        let grid = parse_arg(args, "--grid").context("--pixels needs --grid H,W")?;
        let hw: Vec<usize> = grid.split(',').map(|x| x.trim().parse()).collect::<Result<_, _>>().context("--grid H,W")?;
        anyhow::ensure!(hw.len() == 2, "--grid H,W");
        let od = parse_arg(args, "--out-dir").context("--pixels needs --out-dir DIR")?;
        std::fs::create_dir_all(od)?;
        let raw = std::fs::read(px).with_context(|| px.to_string())?;
        let pv: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let dev = cudarc::driver::CudaDevice::new(0)?;
        let mut gt = gb10_inference::vision_gpu::GpuVisualTower::new(dev, &tower)?;
        let t1 = std::time::Instant::now();
        let (merged, states) = gt.forward(&pv, hw[0], hw[1], true)?;
        println!("VIS_FORWARD grid {}x{} -> {} tokens in {:.1} ms", hw[0], hw[1], merged.len() / d.out_hidden,
                 t1.elapsed().as_secs_f64() * 1e3);
        for (i, st) in states.iter().enumerate() {
            let name = if i == 0 { "pre_blocks".to_string() } else { format!("block_{}", i - 1) };
            write(&format!("{od}/{name}.f32"), st)?;
        }
        write(&format!("{od}/merger.f32"), &merged)?;
        println!("VIS_DUMP_STATES {} states + merger -> {od}", states.len());
    }
    Ok(())
}
