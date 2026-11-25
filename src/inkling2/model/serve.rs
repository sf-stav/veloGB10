//! The GPU forward behind the Inkling-2 serve shell (`serve::forward::Forward`): one worker thread owns the model,
//! the work buffers and `lanes` sequences; HTTP handlers hand it jobs through a channel and stream back its tokens.
//!
//! Scheduling: a request waits in the queue for a free lane. Each loop iteration runs at most ONE prefill chunk
//! (2,048 rows, or 512 while other lanes decode, so a long prompt never stalls their streams for more than one
//! chunk) and then ONE decode step over every lane whose prompt is in (one CUDA graph replay, S4).
//! Prefix cache (`--prefix-cache on`, prefix.rs): at each prompt end the lane's sliding-window / conv-ring state is
//! saved; a later request whose prompt starts with that whole prompt is admitted to that lane and resumes there.
//!
//! Sampling: temperature-only requests (the stock pack's default: temperature 1.0, no top-p / top-k) and greedy are
//! drawn on the GPU (Gumbel-max, exact); a request that sets top_p < 1 or top_k > 0 is drawn on the CPU from the
//! downloaded logits row (exact nucleus / top-k), counted in `velogb10_inkling2_cpu_sampled_tokens_total`.

use super::super::serve::forward::{End, Forward, Job};
use super::super::vision::gpu::VisionGpu;
use super::super::vision::serve::{TowerCache, VisionReq};
use super::super::serve::metrics as mt;
use super::prefix::{Entry, Snap};
use crate::inkling2::mtp::gpu::MtpSnap;
use super::{GpuModel, Lane, Seg, Work, CHUNK};
use anyhow::{anyhow, Result};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};

pub(crate) enum Ev { Tok(u32), Done(Result<End>) }

pub(crate) struct Req {
    pub(crate) prompt: Vec<u32>,
    pub(crate) max_new: usize,
    pub(crate) temperature: f32,
    pub(crate) top_p: f32,
    pub(crate) top_k: usize,
    pub(crate) seed: u64,
    pub(crate) vision: Option<Arc<VisionReq>>,
    pub(crate) cancel: Arc<AtomicBool>,
    pub(crate) out: Sender<Ev>,
}

pub struct GpuForward { tx: Mutex<Sender<Req>>, eos: Vec<u32>, vision: bool, name: &'static str }

/// Tower-output cache budget on the device (features of recently seen images).
const TOWER_CACHE_BYTES: usize = 256 << 20;

/// `pf` = prompt tokens prefilled so far (< prompt length: still prefilling, chunk by chunk between decode steps).
pub(crate) struct Active { pub(crate) req: Req, pub(crate) gen: usize, pub(crate) next: u32, pub(crate) step: u64, pub(crate) pf: usize }

pub(crate) fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// Exact top-k / nucleus sampling of one logits row on the CPU.
pub(crate) fn cpu_sample(l: &[f32], t: f32, top_p: f32, top_k: usize, seed: u64) -> u32 {
    if t <= 0.0 {
        let mut b = 0;
        for i in 1..l.len() { if l[i] > l[b] { b = i; } }
        return b as u32;
    }
    let mut idx: Vec<usize> = (0..l.len()).collect();
    idx.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
    let kmax = if top_k > 0 { top_k.min(idx.len()) } else { idx.len() };
    let mx = l[idx[0]] as f64;
    let p: Vec<f64> = idx[..kmax].iter().map(|&i| ((l[i] as f64 - mx) / t as f64).exp()).collect();
    let tot: f64 = p.iter().sum();
    let mut keep = kmax;
    if top_p < 1.0 {
        let mut c = 0.0;
        for (j, &x) in p.iter().enumerate() { c += x / tot; if c >= top_p as f64 { keep = j + 1; break; } }
    }
    let s: f64 = p[..keep].iter().sum();
    let u = ((mix(seed) >> 11) as f64 + 0.5) / (1u64 << 53) as f64 * s;
    let mut c = 0.0;
    for j in 0..keep { c += p[j]; if u < c { return idx[j] as u32; } }
    idx[keep - 1] as u32
}

/// S5: the vision tower (weights + scratch for the per-image cap). A pack without a usable tower serves text and says so.
pub(crate) fn load_vision(dev: &Arc<cudarc::driver::CudaDevice>, dir: &Path, image_max_tokens: usize) -> Option<VisionGpu> {
    match super::super::vision::VisionHost::load(dir).and_then(|h| VisionGpu::new(dev, &h, true)).and_then(|mut v| {
        v.ensure_scratch(1 + 4 * image_max_tokens)?;
        // warm-up: one small image through the tower (cuBLAS handle / heuristics, module first launches) so the first
        // image request does not pay it
        let warm = super::super::vision::preproc::PreImage { grid: super::super::vision::Grid { rows: 2, cols: 2 }, pixels: vec![0.0; 4 * 4800],
            hash: 0, src: (80, 80), resized: None, had_alpha: false };
        let out = dev.alloc_zeros::<f32>(4 * v.cfg.text_hidden)?;
        v.forward(&warm, super::super::gpu::dp(&out), None)?;
        dev.synchronize()?;
        Ok(v)
    }) {
        Ok(v) => {
            let side = ((image_max_tokens as f64).sqrt() * 40.0) as usize;
            crate::boot_report::notice(&format!("vision: tower loaded ({:.0} MB weights, {:.0} MB scratch); images up to --image-max-tokens {image_max_tokens} LM tokens (~{side} x {side} px; larger ones are downscaled to fit), tower cache {} MB",
                v.weight_bytes as f64 / 1e6, v.scratch_bytes(image_max_tokens) as f64 / 1e6, TOWER_CACHE_BYTES >> 20));
            mt::set(&mt::VIS_ON, 1);
            mt::set(&mt::VIS_MAX_TOKENS, image_max_tokens as u64);
            Some(v)
        }
        Err(e) => {
            crate::boot_report::notice(&format!("vision: the tower did not load ({e:#}); this server answers image requests with an error and serves text"));
            None
        }
    }
}

impl GpuForward {
    /// Load the model on device 0 and start the worker with `lanes` sequences of `max_ctx` positions.
    /// `spec`: Some(options) = speculative decoding with the bundled drafter (S6 / S8, `mtp::serve`), None = plain.
    #[allow(clippy::too_many_arguments)]
    pub fn start(dir: &Path, lanes: usize, max_ctx: usize, decode_tc: bool, prefix_on: bool, attn_single_pass: bool,
                 spec: Option<crate::inkling2::mtp::serve::SpecOpts>, image_max_tokens: usize) -> Result<GpuForward> {
        let dev = cudarc::driver::CudaDevice::new(0).map_err(|e| anyhow!("CudaDevice::new(0): {e:?}"))?;
        let g = GpuModel::load(dir, &dev)?;
        Self::start_model(g, dir, lanes, max_ctx, decode_tc, prefix_on, attn_single_pass, spec, image_max_tokens)
    }

    /// S-INK2-TP2: start the worker on an already loaded model (TP: rank 0 with its link and control streams).
    #[allow(clippy::too_many_arguments)]
    pub fn start_model(mut g: GpuModel, dir: &Path, lanes: usize, max_ctx: usize, decode_tc: bool, prefix_on: bool, attn_single_pass: bool,
                       spec: Option<crate::inkling2::mtp::serve::SpecOpts>, image_max_tokens: usize) -> Result<GpuForward> {
        let dev = g.dev.clone();
        let name = match g.rank.world { 1 => "inkling2-gpu-tp1", 2 => "inkling2-gpu-tp2", 4 => "inkling2-gpu-tp4", _ => "inkling2-gpu-tp8" };
        mt::set(&mt::TP_WORLD, g.rank.world as u64);
        // vision (TP: the tower runs on the head only; its features enter the prompt rows of the Forward op)
        let vis = load_vision(&dev, dir, image_max_tokens);
        // the speculative rounds verify through verify_lanes (CUDA-core decode attention, device positions)
        if decode_tc && spec.is_some() {
            crate::boot_report::notice("--inkling2-decode-attn-tc: a plain-decode diagnostic; speculation is on, so decode and verify use the lanes attention (the option is ignored)");
        }
        g.decode_tc = decode_tc && spec.is_none();
        g.attn_single_pass = attn_single_pass;
        let ws = g.new_work()?;
        let mut ws = ws;
        let mut ls = Vec::new();
        for _ in 0..lanes { ls.push(g.new_lane(max_ctx)?); }
        g.bind_lanes(&mut ws, &ls)?;
        let eos = g.cfg.eos.clone();
        let (tx, rx) = channel::<Req>();
        let eos2 = eos.clone();
        let sp = match spec {
            Some(o) => {
                let pack = crate::inkling2::pack::Inkling2Pack::open(dir)?;
                if !g.rank.is_solo() {
                    // Drop 1: vocab-parallel speculation across all W ranks (reusing each rank's 1/W LM head slice,
                    // folding 1/W of Markov head per rank, and running vocab-parallel exact accept with zero full-vocab gather).
                    crate::boot_report::notice(&format!(
                        "inkling2 TP={}: vocab-parallel speculation across all {} ranks (drafter LM head + Markov fold + exact accept sharded 1/{})",
                        g.rank.world, g.rank.world, g.rank.world
                    ));
                }
                Some(crate::inkling2::mtp::serve::SpecState::new(&g, &pack, &mut ws, lanes, max_ctx, o)?)
            }
            None => None,
        };
        let vision = vis.is_some();
        std::thread::Builder::new().name("inkling2-gpu".into()).spawn(move || worker(g, ws, ls, rx, eos2, prefix_on, sp, vis))?;
        Ok(GpuForward { tx: Mutex::new(tx), eos, vision, name })
    }
}

impl Forward for GpuForward {
    fn name(&self) -> &'static str { self.name }
    fn vision(&self) -> bool { self.vision }
    fn generate(&self, job: &Job<'_>, emit: &mut dyn FnMut(u32) -> bool) -> Result<End> {
        let (otx, orx) = channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let seed = job.seed.unwrap_or_else(|| {
            let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
            mix(t ^ (std::process::id() as u64) << 32)
        });
        let req = Req { prompt: job.prompt.to_vec(), max_new: job.max_new, temperature: job.temperature, top_p: job.top_p, top_k: job.top_k,
                        seed, vision: job.vision.clone(), cancel: cancel.clone(), out: otx };
        self.tx.lock().map_err(|_| anyhow!("worker lock"))?.send(req).map_err(|_| anyhow!("the inkling2 GPU worker is gone"))?;
        let _ = &self.eos;
        loop {
            match orx.recv() {
                Ok(Ev::Tok(t)) => { if !cancel.load(Ordering::Relaxed) && !emit(t) { cancel.store(true, Ordering::Relaxed); } }
                Ok(Ev::Done(r)) => return r,
                Err(_) => return Err(anyhow!("the inkling2 GPU worker dropped the request")),
            }
        }
    }
}

/// Rows per prefill chunk while other lanes are decoding (they wait one chunk, never a whole prompt).
const INTERLEAVE_CHUNK: usize = 512;

fn worker(g: GpuModel, mut ws: Work, mut lanes: Vec<Lane>, rx: Receiver<Req>, eos: Vec<u32>, prefix_on: bool,
          mut spec: Option<crate::inkling2::mtp::serve::SpecState>, mut vis: Option<VisionGpu>) {
    if spec.is_some() { mt::set(&mt::SPEC_ON, 1); }
    let mut tcache = TowerCache::new(TOWER_CACHE_BYTES);
    let n = lanes.len();
    let mut act: Vec<Option<Active>> = (0..n).map(|_| None).collect();
    // prefix cache: the saved prompt end of each lane (valid while the lane has not started another prompt)
    let mut cache: Vec<Option<Entry>> = (0..n).map(|_| None).collect();
    let mut spare: Vec<Option<(Snap, Option<MtpSnap>)>> = (0..n).map(|_| None).collect();
    let mut last_use: Vec<u64> = vec![0; n];
    let mut tick = 0u64;
    // S5: each lane's image features for its prompt (lane-owned copies: the tower cache may evict while a long prompt
    // prefills chunk by chunk between other lanes' steps)
    let mut limg: Vec<Vec<(usize, usize, cudarc::driver::CudaSlice<f32>)>> = (0..n).map(|_| Vec::new()).collect();
    let v = g.vocab();
    loop {
        // admit
        let busy = act.iter().any(|a| a.is_some());
        let mut incoming: Vec<Req> = Vec::new();
        if !busy {
            match rx.recv() { Ok(r) => incoming.push(r), Err(_) => return }
        }
        while act.iter().filter(|a| a.is_none()).count() > incoming.len() {
            match rx.try_recv() { Ok(r) => incoming.push(r), Err(_) => break }
        }
        for req in incoming {
            let lane_ctx = lanes[0].max_ctx;
            if req.prompt.is_empty() || req.prompt.len() >= lane_ctx {
                let _ = req.out.send(Ev::Done(Err(anyhow!("prompt of {} tokens does not fit the served context {lane_ctx}", req.prompt.len()))));
                continue;
            }
            let free: Vec<usize> = (0..n).filter(|&i| act[i].is_none()).collect();
            // a free lane whose saved prompt is a strict prefix of this one (longest), else the least recently used
            let hit = free.iter().copied().filter_map(|i| {
                cache[i].as_ref().filter(|e| e.toks.len() < req.prompt.len() && req.prompt.starts_with(&e.toks)
                                          && img_sig(req.vision.as_deref(), e.toks.len()) == e.img).map(|e| (e.toks.len(), i))
            }).max();
            let li = match hit { Some((_, i)) => i, None => *free.iter().min_by_key(|&&i| (cache[i].is_some(), last_use[i])).unwrap() };
            tick += 1;
            last_use[li] = tick;
            let mut pf = 0usize;
            let mut ok = true;
            if let Some((p, _)) = hit {
                let e = cache[li].as_ref().unwrap();
                let r = g.snap_restore(&lanes[li], &e.snap, p).and_then(|_| match (spec.as_ref(), e.mtp.as_ref()) {
                    (Some(s), Some(ms)) => s.snap_restore(&g, li, ms),
                    (Some(_), None) => Err(anyhow!("prefix cache: entry without drafter state")),
                    _ => Ok(()),
                });
                match r {
                    Ok(()) => { pf = p; mt::add(&mt::PREFIX_HIT_TOKENS, p as u64); mt::inc(&mt::PREFIX_HITS); }
                    Err(err) => { ok = false; let _ = req.out.send(Ev::Done(Err(err))); }
                }
            }
            if !ok { continue; }
            // the images whose rows are still to prefill (towers run now, features copied to the lane)
            limg[li].clear();
            if let Some(vr) = req.vision.as_ref() {
                if let Err(e) = splice_images(&g, &mut vis, &mut tcache, vr, pf, &mut limg[li]) {
                    let _ = req.out.send(Ev::Done(Err(e)));
                    continue;
                }
            }
            // the lane now runs another prompt: its saved entry (if any) is no longer backed by its full-attention cache
            if let Some(e) = cache[li].take() { spare[li] = Some((e.snap, e.mtp)); }
            lanes[li].pos = pf;
            if let Some(s) = spec.as_mut() { s.admit(li); }
            act[li] = Some(Active { req, gen: 0, next: 0, step: 1, pf });
        }
        // one prefill chunk (the least advanced prefilling lane), smaller while other lanes decode
        let decoding = (0..n).any(|i| act[i].as_ref().is_some_and(|a| a.pf == a.req.prompt.len()));
        if let Some(li) = (0..n).filter(|&i| act[i].as_ref().is_some_and(|a| a.pf < a.req.prompt.len())).min_by_key(|&i| last_use[i]) {
            let chunk = if decoding { INTERLEAVE_CHUNK } else { CHUNK };
            let a = act[li].as_mut().unwrap();
            let (i0, len) = (a.pf, a.req.prompt.len());
            let m = (len - i0).min(chunk);
            let last = i0 + m == len;
            let outr = [m - 1];
            let img: Vec<(usize, usize, u64)> = limg[li].iter().map(|(s0, n, b)| (*s0, *n, super::super::gpu::dp(b))).collect();
            ws.set_img_chunk(&img, i0, m);
            let r = {
                let refs: Vec<&Lane> = lanes.iter().collect();
                g.forward(&mut ws, &refs, &[Seg { lane: li, p0: lanes[li].pos, m }], &a.req.prompt[i0..i0 + m], if last { &outr[..] } else { &[][..] })
            };
            mt::add(&mt::FWD_PREFILL_TOKENS, m as u64);
            if decoding { mt::inc(&mt::FWD_INTERLEAVED_CHUNKS); }
            let r = r.and_then(|_| {
                // the drafter context of the chunk's rows, from the target's taps (S6)
                if let Some(s) = spec.as_mut() { s.commit_prefill(&g, &ws, li, m, lanes[li].pos)?; }
                lanes[li].pos += m;
                a.pf += m;
                if !last { return Ok(None); }
                if prefix_on {
                    let (mut sn, mut ms) = match spare[li].take() { Some(s) => s, None => (g.new_snap(&lanes[li])?, None) };
                    g.snap_save(&lanes[li], &mut sn)?;
                    if let Some(s) = spec.as_ref() {
                        let mut x = match ms.take() { Some(x) => x, None => s.new_snap(&g, li)? };
                        s.snap_save(&g, li, &mut x)?;
                        ms = Some(x);
                    }
                    cache[li] = Some(Entry { toks: a.req.prompt.clone(), snap: sn, mtp: ms, img: img_sig(a.req.vision.as_deref(), len) });
                }
                first_token(&g, &mut ws, &a.req, 0, v).map(Some)
            });
            match r {
                Ok(None) => {}
                Ok(Some(t)) => {
                    let mut a = act[li].take().unwrap();
                    a.next = t;
                    if emit_tok(&mut a, &eos, lanes[li].max_ctx, lanes[li].pos) { act[li] = Some(a); }
                }
                Err(e) => {
                    tp_fatal(&g, &e);
                    cache[li] = None;
                    if let Some(a) = act[li].take() { let _ = a.req.out.send(Ev::Done(Err(e))); }
                }
            }
        }
        // one decode step over every lane whose prompt is in
        let rows: Vec<usize> = (0..n).filter(|&i| act[i].as_ref().is_some_and(|a| a.pf == a.req.prompt.len())).collect();
        if rows.is_empty() { continue; }
        if let Some(s) = spec.as_mut() {
            // speculative round (or a plain step through the same verify entry, when the controller says so)
            if let Err(e) = s.step(&g, &mut ws, &mut lanes, &mut act, &rows, v, &eos) {
                tp_fatal(&g, &e);
                let msg = format!("{e:#}");
                for &i in &rows { if let Some(a) = act[i].take() { let _ = a.req.out.send(Ev::Done(Err(anyhow!("inkling2 speculative round: {msg}")))); } }
            }
            continue;
        }
        let ids: Vec<u32> = rows.iter().map(|&i| act[i].as_ref().unwrap().next).collect();
        let res = if g.decode_tc {
            // the tensor-core decode attention runs per lane (host positions): eager
            let segs: Vec<Seg> = rows.iter().map(|&i| Seg { lane: i, p0: lanes[i].pos, m: 1 }).collect();
            let outr: Vec<usize> = (0..rows.len()).collect();
            let refs: Vec<&Lane> = lanes.iter().collect();
            g.forward(&mut ws, &refs, &segs, &ids, &outr)
        } else {
            // one launch per op for all lanes, device-side positions: the step is one CUDA graph replay
            let rl: Vec<(usize, u32)> = rows.iter().zip(&ids).map(|(&i, &t)| (i, t)).collect();
            g.decode_lanes(&mut ws, &lanes, &rl, true)
        };
        if let Err(e) = res {
            tp_fatal(&g, &e);
            let msg = format!("{e:#}");
            for &i in &rows { if let Some(a) = act[i].take() { let _ = a.req.out.send(Ev::Done(Err(anyhow!("inkling2 forward: {msg}")))); } }
            continue;
        }
        mt::inc(&mt::FWD_DECODE_STEPS);
        mt::add(&mt::FWD_DECODE_ROWS, rows.len() as u64);
        for &i in &rows { lanes[i].pos += 1; }
        let toks = sample_rows(&g, &mut ws, &rows, &act, v);
        match toks {
            Ok(toks) => {
                for (j, &i) in rows.iter().enumerate() {
                    let mut a = act[i].take().unwrap();
                    a.next = toks[j];
                    a.step += 1;
                    if emit_tok(&mut a, &eos, lanes[i].max_ctx, lanes[i].pos) { act[i] = Some(a); }
                }
            }
            Err(e) => {
                tp_fatal(&g, &e);
                let msg = format!("{e:#}");
                for &i in &rows { if let Some(a) = act[i].take() { let _ = a.req.out.send(Ev::Done(Err(anyhow!("inkling2 sampler: {msg}")))); } }
            }
        }
    }
}

/// Under TP an error in a model call leaves the ranks out of lockstep (or the link is gone): the head stops the
/// process with one clear line instead of serving partial sums or hanging on a dead peer (exit 70).
fn tp_fatal(g: &GpuModel, e: &anyhow::Error) {
    if g.tp.is_none() { return; }
    eprintln!("\n*** FATAL: the Inkling-2 TP={} head failed in a model call: {e:#}. Every node stops with it; exiting (70). ***\n", g.rank.world);
    mt::inc(&mt::REQ_ENGINE_ERR);
    std::thread::sleep(std::time::Duration::from_millis(300));
    std::process::exit(70);
}

pub(crate) fn needs_cpu(r: &Req) -> bool { r.temperature > 0.0 && (r.top_p < 1.0 || r.top_k > 0) }

pub(crate) fn first_token(g: &GpuModel, ws: &mut Work, req: &Req, row: usize, v: usize) -> Result<u32> {
    if needs_cpu(req) {
        mt::inc(&mt::FWD_CPU_SAMPLED);
        let l = g.logits_host(ws, row + 1)?;
        Ok(cpu_sample(&l[row * v..(row + 1) * v], req.temperature, req.top_p, req.top_k, mix(req.seed)))
    } else {
        mt::inc(&mt::FWD_GPU_SAMPLED);
        Ok(g.sample(ws, &[req.temperature], &[mix(req.seed)])?[0])
    }
}

fn sample_rows(g: &GpuModel, ws: &mut Work, rows: &[usize], act: &[Option<Active>], v: usize) -> Result<Vec<u32>> {
    let reqs: Vec<&Active> = rows.iter().map(|&i| act[i].as_ref().unwrap()).collect();
    let t: Vec<f32> = reqs.iter().map(|a| if needs_cpu(&a.req) { 0.0 } else { a.req.temperature }).collect();
    let seeds: Vec<u64> = reqs.iter().map(|a| mix(a.req.seed ^ mix(a.step))).collect();
    let mut out = g.sample(ws, &t, &seeds)?;
    if reqs.iter().any(|a| needs_cpu(&a.req)) {
        let l = g.logits_host(ws, rows.len())?;
        for (j, a) in reqs.iter().enumerate() {
            if needs_cpu(&a.req) {
                mt::inc(&mt::FWD_CPU_SAMPLED);
                out[j] = cpu_sample(&l[j * v..(j + 1) * v], a.req.temperature, a.req.top_p, a.req.top_k, seeds[j]);
            } else { mt::inc(&mt::FWD_GPU_SAMPLED); }
        }
    } else {
        mt::add(&mt::FWD_GPU_SAMPLED, rows.len() as u64);
    }
    Ok(out)
}

/// Emit `a.next`; false when the request is finished (Done sent).
pub(crate) fn emit_tok(a: &mut Active, eos: &[u32], max_ctx: usize, pos: usize) -> bool {
    if a.req.cancel.load(Ordering::Relaxed) { let _ = a.req.out.send(Ev::Done(Ok(End::Stop))); return false; }
    let t = a.next;
    if a.req.out.send(Ev::Tok(t)).is_err() { return false; }
    a.gen += 1;
    if eos.contains(&t) { let _ = a.req.out.send(Ev::Done(Ok(End::Stop))); return false; }
    if a.gen >= a.req.max_new || pos + 1 >= max_ctx { let _ = a.req.out.send(Ev::Done(Ok(End::Length))); return false; }
    true
}

/// The image signature of a prompt's first `upto` rows: (first row, rows, image hash) of every image span starting
/// there (a prefix-cache entry matches only the same images at the same rows, `VisionReq::key`'s content).
pub(crate) fn img_sig(vr: Option<&VisionReq>, upto: usize) -> Vec<(usize, usize, u64)> {
    match vr {
        Some(v) => v.images.iter().zip(&v.spans).filter(|(_, s)| s.0 < upto).map(|(im, s)| (s.0, s.1, im.hash)).collect(),
        None => Vec::new(),
    }
}

/// Run (or fetch from the tower cache) the tower of every image of `vr` whose rows lie at or after prompt row `from`,
/// and copy its features into lane-owned buffers `out` = (first prompt row, rows, [rows][D] f32).
fn splice_images(g: &GpuModel, vis: &mut Option<VisionGpu>, cache: &mut TowerCache, vr: &VisionReq, from: usize,
                 out: &mut Vec<(usize, usize, cudarc::driver::CudaSlice<f32>)>) -> Result<()> {
    let v = vis.as_mut().ok_or_else(|| anyhow!("this server has no vision tower (it did not load at boot): image requests cannot be served"))?;
    let pinned: Vec<u64> = vr.images.iter().map(|i| i.hash).collect();
    let d = v.cfg.text_hidden;
    for (im, &(s0, n)) in vr.images.iter().zip(&vr.spans) {
        if s0 + n <= from { continue; }
        let p = match cache.get(im.hash) {
            Some(p) => { mt::inc(&mt::VIS_CACHE_HITS); p }
            None => {
                let t0 = std::time::Instant::now();
                let buf = g.dev.alloc_zeros::<f32>(im.grid.tokens() * d)?;
                v.forward(im, super::super::gpu::dp(&buf), None)?;
                g.k.sync()?;
                mt::add(&mt::VIS_TOWER_US, t0.elapsed().as_micros() as u64);
                mt::inc(&mt::VIS_TOWERS);
                mt::add(&mt::VIS_TOWER_TOKENS, im.grid.tokens() as u64);
                cache.put(im.hash, buf, &pinned)
            }
        };
        let own = g.dev.alloc_zeros::<f32>(n * d)?;
        let r = unsafe { cudarc::driver::sys::cuMemcpyDtoDAsync_v2(super::super::gpu::dp(&own), p, n * d * 4, std::ptr::null_mut()) };
        anyhow::ensure!(r == cudarc::driver::sys::CUresult::CUDA_SUCCESS, "image features copy: {r:?}");
        out.push((s0, n, own));
        mt::add(&mt::VIS_SPLICED_ROWS, n as u64);
    }
    Ok(())
}
