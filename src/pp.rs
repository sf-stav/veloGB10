//! PP-prefill (PLAN/14): two-box LAYER-SPLIT pipeline prefill.
//!
//! Box A (head, rank 0) runs embed + layers [0, split) at FULL width on each prompt window;
//! box B (node, rank 1) runs layers [split, 64) + the final-norm/LM-head tail. The ONLY
//! cross-box artifact is the chunk-boundary residual stream (bf16 [n, h]) — proven
//! value-exact on-box by `--probe-ppsplit` (token + final hidden + full post-prefill state
//! byte-identical), and a bf16 wire copy preserves bits by construction. KV for a layer is
//! written only by that layer's owner, GDN/conv state likewise — nothing else crosses.
//!
//! v0 (this file) is SERIALIZED: per window, head computes the lower half, ships the
//! residual over the TpLink exchange channel (pinned cudaHostAlloc staging), and only then
//! starts the next window; the node's upper half of window c overlaps nothing. It exists to
//! validate transport + end-to-end exactness and to price the ship; the pipelined overlap
//! (head window c+1 || node window c) is v1 and reuses the same protocol unchanged.
//!
//! Protocol (symmetric `exchange`, world==2 — both sides stage a slot of the SAME size and
//! rendezvous; counts match by construction like `broadcast_prompt`):
//!   per window w:  head sends [hdr 32B][residual h*n*2 B]; node sends zeros.
//!                  hdr = [magic, n, pos_start, rep, final, 0, 0, 0] (u32 each)
//!   after the last window of a rep: 64B ack exchange — node's hdr word 0 = first token.
//!
//! CLI: `--pp-node --model-dir D` (rank 1, serves until killed) and
//! `--pp-bench-prefill --model-dir D --peer <head-ip-from-node|unused-on-head>
//! --seq-len N --reps R [--split S] [--verify]` (rank 0).

use crate::gpu::{GpuModel, Pool};
use crate::net::TpLink;
use cudarc::driver::DevicePtr;

pub const PP_PORT: u16 = 29711;
/// Ring-slot capacity: header 32 B + residual 5120*8192*2 B = 83,886,624 B, rounded up.
pub const PP_SLOT_BYTES: usize = 84 * 1024 * 1024;
const HDR_WORDS: usize = 8;
const MAGIC: u32 = 0x5050_0001;

fn rdma_dev() -> String {
    crate::opts::var(crate::opt!("rdma-dev")).ok().filter(|s| !s.is_empty())
        .unwrap_or_else(|| "rocep1s0f1".to_string())
}

fn link_up(rank: i32, peer_ip: &str) -> anyhow::Result<TpLink> {
    // rank 0 listens (peer "" is allowed); rank 1 dials the head's RoCE IP.
    let peer = if rank == 0 { "" } else { peer_ip };
    TpLink::connect(rank, peer, PP_PORT, &rdma_dev(), 3, PP_SLOT_BYTES)
}

/// Node role: serve upper-range windows until killed.
pub fn pp_node(model_dir: &str) -> anyhow::Result<()> {
    // Link FIRST, then load: the head listens within seconds of launch, and both boxes
    // spend their ~100 s model load AFTER the QP handshake — no start-order sensitivity.
    // Retry the dial: the head's listener comes up a few seconds after ITS launch (process
    // start + CUDA context for the pinned staging buffer) — the node may win that race.
    let peer_ip = crate::opts::var(crate::opt!("pp-head-ip")).unwrap_or_default();
    let mut link = None;
    for attempt in 1..=90 {
        match link_up(1, &peer_ip) {
            Ok(l) => { link = Some(l); break; }
            Err(e) => {
                if attempt % 10 == 1 { eprintln!("[pp-node] dial attempt {attempt}/90: {e}"); }
                std::thread::sleep(std::time::Duration::from_secs(2));
            }
        }
    }
    let mut link = link.ok_or_else(|| anyhow::anyhow!("pp-node: head unreachable after 90 attempts"))?;
    eprintln!("[pp-node] rank 1 link UP; loading model ...");
    let (gpu, _) = GpuModel::load_from_dir(model_dir).expect("pp-node: gpu load");
    let split = gpu.nlayers() / 2;
    let max_seq_len = 262144usize.next_power_of_two(); // state sized for the ladder's top
    let mut pool = Pool::new(gpu.dev().clone());
    let mut state = gpu.new_batch_state(1, 1, max_seq_len);
    let kv_stride = max_seq_len;
    let h = gpu.cfg().hidden_size;
    eprintln!("[pp-node] model loaded (split={split}, h={h}, kv_stride={kv_stride})");

    let mut cur_rep: u32 = u32::MAX;
    let mut dummy: Vec<u32> = Vec::new();
    loop {
        // Stage zeros, rendezvous on the residual frame.
        {
            let s = link.send_host_mut::<u8>(64);
            s.iter_mut().for_each(|b| *b = 0);
        }
        // Peek size first is not possible on this channel: the node must know nbytes.
        // The head always ships header+residual of the CURRENT window; the node learns n
        // from the header it just received — so the exchange size must be agreed BEFORE.
        // We fix it: the head ships a FULL-capacity frame every window (residual padded
        // with zeros to PP_SLOT_BYTES). The node reads n from the header and ignores pad.
        link.exchange(PP_SLOT_BYTES)?;
        let hdr = link.recv_host::<u32>(HDR_WORDS);
        if hdr[0] != MAGIC { anyhow::bail!("pp-node: bad magic {:#x}", hdr[0]); }
        let n = hdr[1] as usize;
        let pos_start = hdr[2] as usize;
        let rep = hdr[3];
        let final_w = hdr[4] != 0;
        // Zero on rep change OR on any prompt's first window (pos_start==0): a restarted
        // head process renumbers reps from 0, which could collide with cur_rep and skip the
        // zeroing — pos_start==0 is prompt-begin by construction, independent of numbering.
        if rep != cur_rep || pos_start == 0 {
            gpu.zero_slot_state(&mut state, 0, kv_stride);
            if rep != cur_rep { eprintln!("[pp-node] rep {rep} begins (window n={n} pos={pos_start})"); }
            cur_rep = rep;
        }
        // Residual → device (full overwrite; alloc_zeros' non-zeroing is irrelevant).
        let recv_bytes = link.recv_host::<u8>(PP_SLOT_BYTES);
        let res_host: &[half::bf16] = unsafe {
            std::slice::from_raw_parts(recv_bytes[HDR_WORDS * 4..].as_ptr() as *const half::bf16, h * n)
        };
        let residual = gpu.dev().htod_sync_copy(res_host).expect("pp-node: residual htod");
        if dummy.len() != n { dummy = vec![0u32; n]; }
        let t0 = std::time::Instant::now();
        let (tok, h_out) = gpu.prefill_batch_range(&mut pool, &dummy, &mut state,
            0, kv_stride, pos_start, split, gpu.nlayers(), Some(residual));
        gpu.sync_stream();
        eprintln!("[pp-node] w pos={pos_start} n={n} upper={:.1}ms", t0.elapsed().as_secs_f64()*1e3);
        pool.release_bf16(h_out, h * n);
        if final_w {
            // Ack exchange: stage the token, rendezvous on 64 B.
            {
                let s = link.send_host_mut::<u32>(16);
                s.iter_mut().for_each(|w| *w = 0);
                s[0] = tok;
            }
            link.exchange(64)?;
            eprintln!("[pp-node] rep {rep} complete (token {tok})");
        }
    }
}

/// Head role: drive reps of chunked prompt windows, lower half local, upper half shipped.
/// `chunks` sweeps window sizes (empty = [PREFILL_CHUNK]); the node is chunk-agnostic — it
/// learns each window's n from the frame header, so ONE node launch serves the whole sweep.
pub fn pp_bench_head(model_dir: &str, seq_len: usize, reps: usize, split: usize, verify: bool, chunks: &[usize]) -> anyhow::Result<()> {
    // Link FIRST (see pp_node): the listener is up within seconds; the model load follows.
    let mut link = link_up(0, "")?;
    eprintln!("[pp-head] rank 0 link UP (listening); loading model ...");
    let (gpu, _) = GpuModel::load_from_dir(model_dir).expect("pp-head: gpu load");
    let h = gpu.cfg().hidden_size;
    let max_seq_len = (seq_len + 128).next_power_of_two();
    let mut pool = Pool::new(gpu.dev().clone());
    // kv_slots MUST be 2 alongside state_slots: zero_slot_state indexes kv_mirror[li][slot],
    // which is sized by kv_slots — (1,2) panicked at gpu.rs:9682 the moment --verify touched
    // slot 1 (index out of bounds: len 1, index 1).
    let mut state = gpu.new_batch_state(2, 2, max_seq_len);
    let kv_stride = max_seq_len;
    let chunks: Vec<usize> = if chunks.is_empty() { vec![crate::batch::PREFILL_CHUNK] } else { chunks.to_vec() };

    let prompt: Vec<u32> = (0..seq_len).map(|i| ((i * 2654435761usize) % 30000 + 5) as u32).collect();
    let split = if split == 0 { gpu.nlayers() / 2 } else { split };
    eprintln!("[pp-head] model loaded (seq={seq_len}, split={split}, chunks={chunks:?}, verify={verify})");

    let mut best_ms = f64::INFINITY;
    let mut verify_ok = true;
    let mut per_chunk_best: Vec<(usize, f64)> = Vec::new();
    for &chunk in &chunks {
    let mut chunk_best = f64::INFINITY;
    for rep in 0..reps {
        gpu.zero_slot_state(&mut state, 0, kv_stride);
        gpu.sync_stream();
        let t0 = std::time::Instant::now();
        let mut tok = 0u32;
        let mut w0 = 0usize;
        while w0 < seq_len {
            let w1 = (w0 + chunk).min(seq_len);
            let n = w1 - w0;
            let final_w = w1 == seq_len;
            // Lower half on the head.
            let t_low = std::time::Instant::now();
            let (_, residual): (u32, crate::gpu::B) = gpu.prefill_batch_range(&mut pool, &prompt[w0..w1],
                &mut state, 0, kv_stride, w0, 0, split, None);
            gpu.sync_stream();
            let low_ms = t_low.elapsed().as_secs_f64()*1e3;
            // D2H the residual straight into the pinned send slot (offset = header).
            {
                let s = link.send_host_mut::<u32>(HDR_WORDS);
                s[0] = MAGIC; s[1] = n as u32; s[2] = w0 as u32; s[3] = rep as u32;
                s[4] = final_w as u32; s[5] = 0; s[6] = 0; s[7] = 0;
            }
            let t_ship = std::time::Instant::now();
            unsafe {
                let dst = link.send_host_mut::<u8>(PP_SLOT_BYTES);
                let dst_ptr = dst.as_mut_ptr().add(HDR_WORDS * 4);
                cudarc::driver::result::memcpy_dtoh_sync(
                    std::slice::from_raw_parts_mut(dst_ptr, h * n * 2),
                    *residual.device_ptr() as cudarc::driver::sys::CUdeviceptr)
            }.expect("pp-head: residual d2h");
            // Pad the tail of the slot with zeros (the frame is capacity-sized every window).
            {
                let s = link.send_host_mut::<u8>(PP_SLOT_BYTES);
                for b in s[HDR_WORDS * 4 + h * n * 2..].iter_mut() { *b = 0; }
            }
            let ship_ms = t_ship.elapsed().as_secs_f64()*1e3;
            let t_x = std::time::Instant::now();
            link.exchange(PP_SLOT_BYTES)?;
            let xchg_ms = t_x.elapsed().as_secs_f64()*1e3;
            eprintln!("[pp-head] w pos={w0} n={n} lower={low_ms:.1}ms d2h+pad={ship_ms:.1}ms xchg={xchg_ms:.1}ms");
            pool.release_bf16(residual, h * n);
            w0 = w1;
        }
        // Ack rendezvous: the node's upper half of the last window runs while the head waits.
        {
            let s = link.send_host_mut::<u32>(16);
            s.iter_mut().for_each(|w| *w = 0);
        }
        link.exchange(64)?;
        tok = link.recv_host::<u32>(16)[0];
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        best_ms = best_ms.min(ms);
        chunk_best = chunk_best.min(ms);
        println!("[pp-head] chunk={chunk} rep {rep}: TTFT {ms:.1} ms  ({:.0} tok/s)  token {tok}",
                 seq_len as f64 / ms * 1e3);

        if verify {
            gpu.zero_slot_state(&mut state, 1, kv_stride);
            let (tok_m, h_m) = gpu.prefill_batch(&mut pool, &prompt, &mut state, 1, kv_stride, 0);
            gpu.sync_stream();
            pool.release_bf16(h_m, h * seq_len);
            let ok = tok_m == tok;
            verify_ok &= ok;
            println!("[pp-head] verify chunk={chunk} rep {rep}: monolithic token {tok_m} vs PP token {tok} -> {}",
                     if ok { "MATCH" } else { "MISMATCH" });
        }
    }
    println!("[pp-head] chunk={chunk} BEST {chunk_best:.1} ms  ({:.0} tok/s)",
             seq_len as f64 / chunk_best * 1e3);
    per_chunk_best.push((chunk, chunk_best));
    }
    println!("[pp-head] BEST TTFT {best_ms:.1} ms  ({:.0} tok/s)  N={seq_len}{}",
             seq_len as f64 / best_ms * 1e3,
             if verify { format!("  verify={}", if verify_ok { "MATCH" } else { "MISMATCH" }) } else { String::new() });
    for (c, ms) in &per_chunk_best { println!("[pp-head] SUMMARY chunk={c}: {ms:.1} ms ({:.0} tok/s)", seq_len as f64 / ms * 1e3); }
    if verify && !verify_ok { std::process::exit(1); }
    Ok(())
}
