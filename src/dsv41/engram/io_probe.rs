//! `--probe-dsv41-engram-io` (leg EIO): the Engram runtime on a box's real head files and GPU.
//!
//! ```text
//! gb10_inference --probe-dsv41-engram-io --engram-dir DIR --engram-heads 6,7 --model-dir <config.json + tokenizer.json (+ text)>
//!     [--engram-io-part all|identity|latency|decode|prefill] [--engram-io-lanes 4] [--engram-io-rows 6]
//!     [--engram-io-steps N] [--engram-io-period-us 12000] [--engram-io-cover-us 700] [--engram-io-tokens text|random]
//!     [--engram-io-accept 70] [--engram-io-draft-prefetch on|off] [--engram-readers 2] [--engram-row-cache-mib 2048]
//! ```
//!
//! * layout: every head file's manifest equals `EngramLayout::rows_for_head` (checked at open).
//! * identity: steps of real hash ids (text, speculative blocks, DEAD positions, a small cache that evicts,
//!   warm requests that the next step joins) go through the whole runtime (planner, cache, io_uring readers,
//!   publish, the GPU's bounded wait + decode); every bf16 row and every raw record must equal the CPU path
//!   (a buffered pread of the record at the offset the layout gives + the CPU twin of the decode). Then the
//!   prefill channel with read-ahead, the records-decode kernel, the no-cache mode, and a negative control
//!   (rows of one step against the CPU rows of another must differ). RESULT `DSV41_ENGRAM_IO_IDENTITY_OK`.
//! * latency: the per-step latency table (tokens known -> rows decoded on the GPU) at 1 and 4 lanes x rows,
//!   cold (no cache, random tokens: every row from the NVMe) and steady text (cache on). `..._LATENCY_OK`.
//! * decode: the steady synthetic decode: the GPU runs [cover spin][bounded wait + decode][rest of the round,
//!   doorbells]; the host learns the next tokens from the doorbells exactly when a real round ends, hashes,
//!   prefetches the sampled token's rows when the drafter starts (draft prefetch) and submits the block when
//!   the drafts are known. Stalls are counted by the GPU. `..._DECODE_OK` only with zero stalls / timeouts.
//! * prefill: prompt read-ahead of 16K-token chunks: time to have a chunk's rows resident, then the chunk's
//!   step must be all hits. `..._PREFILL_OK`.

use super::hash::{EngramHasher, NgramHashState};
use super::prefetch::{ch_decode, EngramIo, EngramIoConfig};
use crate::dsv41::gpu::engram as ge;
use crate::opts;
use anyhow::{anyhow, bail, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 { self.next() % n }
}

struct Opts {
    dir: PathBuf,
    heads: Vec<usize>,
    model_dir: PathBuf,
    part: String,
    lanes: usize,
    rows: usize,
    steps: usize,
    period_us: u64,
    cover_us: u64,
    random: bool,
    accept: u64,
    draft_prefetch: bool,
    readers: usize,
    cache_mib: usize,
    keepalive_us: u64,
}

fn ranged(o: opts::OptId) -> Result<i64> { opts::ranged(o).map_err(|e| anyhow!(e)) }

fn parse(args: &[String]) -> Result<Opts> {
    let arg = |k: &str| crate::arg_value(args, k).map(|s| s.to_string());
    let dir = PathBuf::from(arg("--engram-dir").context("--engram-dir <dir of engram_head{h}.bin + manifest> is required")?);
    let heads = super::pack::parse_heads(&arg("--engram-heads").context("--engram-heads <the heads on this box, e.g. 6,7> is required")?, 8)?;
    let model_dir = PathBuf::from(arg("--model-dir").context("--model-dir <dir with config.json + tokenizer.json> is required")?);
    Ok(Opts {
        dir,
        heads,
        model_dir,
        part: opts::choice(crate::opt!("engram-io-part")),
        lanes: ranged(crate::opt!("engram-io-lanes"))? as usize,
        rows: ranged(crate::opt!("engram-io-rows"))? as usize,
        steps: ranged(crate::opt!("engram-io-steps"))? as usize,
        period_us: ranged(crate::opt!("engram-io-period-us"))? as u64,
        cover_us: ranged(crate::opt!("engram-io-cover-us"))? as u64,
        random: opts::choice(crate::opt!("engram-io-tokens")) == "random",
        accept: ranged(crate::opt!("engram-io-accept"))? as u64,
        draft_prefetch: opts::choice(crate::opt!("engram-io-draft-prefetch")) == "on",
        readers: ranged(crate::opt!("engram-readers"))? as usize,
        cache_mib: ranged(crate::opt!("engram-row-cache-mib"))? as usize,
        keepalive_us: ranged(crate::opt!("engram-keepalive-us"))? as u64,
    })
}

/// Real text for the token streams: README.md, then every .md / .py file of `inference/` and `encoding/`
/// in the model dir (the native checkpoint ships them), tokenized with the model's tokenizer.
fn corpus(model_dir: &Path) -> Result<Vec<u32>> {
    let tok = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(|e| anyhow!("tokenizer.json: {e}"))?;
    let mut files: Vec<PathBuf> = vec![model_dir.join("README.md")];
    for sub in ["inference", "encoding"] {
        if let Ok(rd) = std::fs::read_dir(model_dir.join(sub)) {
            let mut v: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "md" || x == "py")).collect();
            v.sort();
            files.extend(v);
        }
    }
    let mut ids = Vec::new();
    for f in files.iter().filter(|f| f.exists()) {
        let text = String::from_utf8_lossy(&std::fs::read(f)?).into_owned();
        let enc = tok.encode(text, false).map_err(|e| anyhow!("tokenize {}: {e}", f.display()))?;
        ids.extend_from_slice(enc.get_ids());
    }
    if ids.len() < 4096 { bail!("{}: only {} text tokens (README.md + inference/ + encoding/); the token streams need >= 4096", model_dir.display(), ids.len()); }
    Ok(ids)
}

/// One synthetic lane: a position in the corpus and its n-gram state.
struct Lane {
    st: NgramHashState,
    pos: usize,
}

struct Streams<'a> {
    hasher: &'a EngramHasher,
    corpus: &'a [u32],
    lanes: Vec<Lane>,
    rows: usize,
    accept: u64,
    random: bool,
    rng: Rng,
    vocab: u64,
}

/// One round's fed tokens of every lane and their ids.
struct Block {
    fed: Vec<Vec<u32>>,
    acc: Vec<usize>,
    live: Vec<bool>,
    ids: Vec<u32>,
}

impl<'a> Streams<'a> {
    fn new(hasher: &'a EngramHasher, corpus: &'a [u32], lanes: usize, rows: usize, accept: u64, random: bool, seed: u64) -> Self {
        let lanes = (0..lanes).map(|i| Lane { st: NgramHashState::new(), pos: i * corpus.len() / lanes.max(1) }).collect();
        Streams { hasher, corpus, lanes, rows, accept, random, rng: Rng(seed | 1), vocab: hasher.token_map().len() as u64 }
    }
    fn tok(&mut self, at: usize) -> u32 {
        if self.random { self.rng.below(self.vocab) as u32 } else { self.corpus[at % self.corpus.len()] }
    }
    /// The next verify block: per lane [sampled, d1..d_{rows-1}]; accepted drafts are the true text, the rest random.
    fn block(&mut self, dead_every: u64) -> Result<Block> {
        let (r, n) = (self.rows, self.lanes.len());
        let ipt = self.hasher.ids_per_token();
        let mut b = Block { fed: Vec::with_capacity(n), acc: Vec::with_capacity(n), live: Vec::with_capacity(n * r), ids: vec![0; n * r * ipt] };
        for li in 0..n {
            let mut j = 0;
            while j + 1 < r && self.rng.below(100) < self.accept { j += 1; }
            let pos = self.lanes[li].pos;
            let mut fed = Vec::with_capacity(r);
            for i in 0..r {
                let t = if i <= j { self.tok(pos + i) } else { self.rng.below(self.vocab) as u32 };
                fed.push(t);
            }
            let live: Vec<bool> = (0..r).map(|_| dead_every == 0 || self.rng.below(dead_every) != 0).collect();
            let st = self.lanes[li].st;
            self.hasher.peek(&st, &fed, Some(&live), &mut b.ids[li * r * ipt..(li + 1) * r * ipt])?;
            b.live.extend_from_slice(&live);
            b.fed.push(fed);
            b.acc.push(j);
        }
        Ok(b)
    }
    /// Commit the accepted prefix (sampled + accepted drafts) of every lane.
    fn commit(&mut self, b: &Block) -> Result<()> {
        let r = self.rows;
        for (li, lane) in self.lanes.iter_mut().enumerate() {
            let j = b.acc[li];
            self.hasher.advance(&mut lane.st, &b.fed[li][..j + 1], Some(&b.live[li * r..li * r + j + 1]))?;
            lane.pos += j + 1;
        }
        Ok(())
    }
    /// The ids of every lane's sampled token alone (position 0 of the block): the draft-token prefetch's
    /// first warm request, known when the drafter starts.
    fn sampled_ids(&self, b: &Block) -> Vec<u32> {
        let (r, ipt) = (self.rows, self.hasher.ids_per_token());
        (0..self.lanes.len()).flat_map(|li| b.ids[li * r * ipt..li * r * ipt + ipt].iter().copied()).collect()
    }
}

fn pct(v: &mut [u64], p: f64) -> f64 {
    if v.is_empty() { return f64::NAN; }
    v.sort_unstable();
    v[((v.len() as f64 * p) as usize).min(v.len() - 1)] as f64 / 1e3
}

fn spin_until(t: Instant) {
    while Instant::now() < t { std::hint::spin_loop(); }
}

pub fn run_cli(args: &[String]) -> Result<()> {
    let o = parse(args)?;
    let hasher = EngramHasher::from_model_dir(&o.model_dir)?.ok_or_else(|| anyhow!("{}: config.json has no Engram layers", o.model_dir.display()))?;
    let layout = Arc::new(hasher.layout().clone());
    let corpus = corpus(&o.model_dir)?;
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    crate::dsv41::gpu::load_module(&dev)?;
    println!("engram io probe: heads {:?} in {}, {} text tokens from {}, readers {}, keep-alive {} us, kernel build {}", o.heads, o.dir.display(), corpus.len(),
             o.model_dir.display(), o.readers, o.keepalive_us, env!("KERNEL_BUILD_ID"));
    for &h in &o.heads {
        let f = super::io::HeadFile::open(&o.dir, h, &layout)?;
        println!("layout: head {h}: manifest geometry and {} sections equal EngramLayout::rows_for_head({h}) ({} records, {} layout)",
                 f.sections.len(), f.total_records, f.layout.name());
    }
    let all = o.part == "all";
    let mut ok = true;
    if all || o.part == "identity" { ok &= identity(&o, &hasher, &layout, &corpus, &dev)?; }
    if all || o.part == "latency" { ok &= latency(&o, &hasher, &layout, &corpus, &dev)?; }
    if all || o.part == "decode" { ok &= steady_decode(&o, &hasher, &layout, &corpus, &dev, o.lanes, o.random)?; }
    if all || o.part == "prefill" { ok &= prefill(&o, &hasher, &layout, &corpus, &dev)?; }
    if !ok { bail!("at least one part failed (see above)"); }
    println!("DSV41_ENGRAM_IO_OK");
    Ok(())
}

fn base_cfg(o: &Opts, decode_max_tok: usize, prefill_max_tok: usize, cache_mib: usize) -> EngramIoConfig {
    let mut c = EngramIoConfig::new(o.dir.clone(), o.heads.clone());
    c.readers = o.readers;
    c.cache_mib = cache_mib;
    c.decode_max_tok = decode_max_tok;
    c.prefill_max_tok = prefill_max_tok;
    c.keepalive = Duration::from_micros(o.keepalive_us);
    c
}

/// Device buffers of one channel's output (`[channel layers][max_tok][n_own][256]` bf16 + the raw records).
struct Out {
    b: CudaSlice<u16>,
    r: CudaSlice<u8>,
}
impl Out {
    fn new(dev: &Arc<CudaDevice>, io: &EngramIo, ch: usize) -> Result<Self> {
        let n = io.channel_layers(ch) * io.channel_max_tok(ch) * io.n_own();
        Ok(Out { b: dev.alloc_zeros::<u16>(n * 256)?, r: dev.alloc_zeros::<u8>(n * 264)? })
    }
    fn launch(&self, io: &EngramIo, ch: usize) -> Result<()> {
        io.launch(ch, *self.b.device_ptr(), *self.r.device_ptr(), ge::MODE_DECODE | ge::MODE_RECORDS)
    }
}

/// The rows of finished steps on the GPU (bf16 bits + raw records) in (Engram layer, t, j) order, from
/// channels that together cover the Engram layers in order.
fn fetch(dev: &Arc<CudaDevice>, io: &EngramIo, parts: &[(usize, &Out)], n_tok: usize) -> Result<(Vec<u16>, Vec<u8>)> {
    let no = io.n_own();
    let (mut b, mut r) = (Vec::new(), Vec::new());
    for (ch, out) in parts {
        let mt = io.channel_max_tok(*ch);
        let all_b = dev.dtoh_sync_copy(&out.b)?;
        let all_r = dev.dtoh_sync_copy(&out.r)?;
        for l in 0..io.channel_layers(*ch) {
            for t in 0..n_tok {
                let o = (l * mt + t) * no;
                b.extend_from_slice(&all_b[o * 256..(o + no) * 256]);
                r.extend_from_slice(&all_r[o * 264..(o + no) * 264]);
            }
        }
    }
    Ok((b, r))
}

/// The CPU path, independent of the runtime's own row mapping: ENGa's `split_id` turns each hash id into
/// (g, head, row); the record is then read with a buffered pread at ENGb's `PackedHead::record_offset` (the
/// packer's manifest), or, when `--model-dir` is the native checkpoint (its index names the Engram tables),
/// straight from the source shards at table row = id (ground truth); then the CPU twin of the decode.
/// DEAD positions are zero rows.
struct CpuPath<'a> {
    hasher: &'a EngramHasher,
    packed: std::collections::HashMap<usize, (super::manifest::PackedHead, std::fs::File)>,
    source: Option<(super::source::SourceFiles, Vec<super::manifest::SourceLayer>)>,
}

impl<'a> CpuPath<'a> {
    fn new(hasher: &'a EngramHasher, o: &Opts) -> Result<Self> {
        let mut packed = std::collections::HashMap::new();
        for &h in &o.heads {
            let ph = super::manifest::PackedHead::open(&o.dir, h).map_err(|e| anyhow!("{e}"))?;
            let f = std::fs::File::open(&ph.path)?;
            packed.insert(h, (ph, f));
        }
        let source = if o.model_dir.join("model.safetensors.index.json").exists() {
            let sc = super::source::load(&o.model_dir)?;
            Some((super::source::SourceFiles::open(&o.model_dir, &sc.layers)?, sc.layers))
        } else {
            None
        };
        Ok(CpuPath { hasher, packed, source })
    }
    fn what(&self) -> &'static str {
        if self.source.is_some() { "the native checkpoint's source rows (shards 47/48)" } else { "the packed head files at ENGb's manifest offsets" }
    }
    fn rows(&self, own_cols: &[usize], ids: &[u32], live: &[bool]) -> Result<(Vec<u16>, Vec<u8>)> {
        let (n_tok, no) = (live.len(), own_cols.len());
        let (ipt, ncol) = (self.hasher.ids_per_token(), self.hasher.layout().n_cols());
        let mut b = vec![0u16; 2 * n_tok * no * 256];
        let mut r = vec![0u8; 2 * n_tok * no * 264];
        let (mut wb, mut sb, mut ob) = (Vec::new(), Vec::new(), Vec::new());
        for l in 0..2 {
            for t in 0..n_tok {
                if !live[t] { continue; }
                for (j, &c) in own_cols.iter().enumerate() {
                    let id = ids[t * ipt + l * ncol + c];
                    let k = (l * n_tok + t) * no + j;
                    let dst = &mut r[k * 264..(k + 1) * 264];
                    if let Some((files, layers)) = &self.source {
                        super::source::read_records(files, layers, l, id as u64, 1, &mut wb, &mut sb, &mut ob)?;
                        dst.copy_from_slice(&ob[..264]);
                    } else {
                        let (g, h, row) = self.hasher.split_id(l, c, id);
                        let (ph, f) = self.packed.get(&h).ok_or_else(|| anyhow!("column {c} belongs to head {h}, not opened"))?;
                        let off = ph.record_offset(l, g, row).map_err(|e| anyhow!("{e}"))?;
                        f.read_exact_at(dst, off)?;
                    }
                    ge::decode_record_cpu(&r[k * 264..(k + 1) * 264], &mut b[k * 256..(k + 1) * 256]);
                }
            }
        }
        Ok((b, r))
    }
}

fn compare(gpu: &(Vec<u16>, Vec<u8>), cpu: &(Vec<u16>, Vec<u8>)) -> (usize, usize) {
    let bad_rows = gpu.0.chunks(256).zip(cpu.0.chunks(256)).filter(|(a, b)| a.iter().zip(b.iter()).any(|(x, y)| !ge::bf16_same(*x, *y))).count();
    let bad_recs = gpu.1.chunks(264).zip(cpu.1.chunks(264)).filter(|(a, b)| a != b).count();
    (bad_rows, bad_recs)
}

fn identity(o: &Opts, hasher: &EngramHasher, layout: &Arc<super::layout::EngramLayout>, corpus: &[u32], dev: &Arc<CudaDevice>) -> Result<bool> {
    let (lanes, rows) = (o.lanes, o.rows);
    let chunk = 512;
    let t0 = Instant::now();
    // a 1 MiB cache is raised to the in-flight floor (~tens of thousands of slots): it evicts within the run
    let mut io = EngramIo::open(dev, layout.clone(), base_cfg(o, lanes * rows, chunk, 1))?;
    let nl = io.n_layers();
    let outs: Vec<Out> = (0..nl).map(|l| Out::new(dev, &io, ch_decode(l))).collect::<Result<_>>()?;
    let parts: Vec<(usize, &Out)> = outs.iter().enumerate().map(|(l, x)| (ch_decode(l), x)).collect();
    let cp = io.ch_prefill();
    let pout = Out::new(dev, &io, cp)?;
    let cpu = CpuPath::new(hasher, o)?;
    let own = io.own_cols();
    println!("identity: the CPU path reads {}; owned columns {:?}", cpu.what(), own);
    let mut s = Streams::new(hasher, corpus, lanes, rows, o.accept, false, 0xE10);
    let (mut n_rows, mut hits, mut misses, mut joined, mut dead, mut bad) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let mut prev: Option<(Vec<u16>, Vec<u8>)> = None;
    let mut neg = 0usize;
    let n_steps = 240;
    let mut next = s.block(23)?;
    for k in 0..n_steps {
        let b = next;
        // even steps launch the waits first (the kernels wait in-stream for the publish), odd steps submit first
        if k % 2 == 0 { for (ch, x) in &parts { x.launch(&io, *ch)?; } }
        let tks = io.submit_decode(&b.ids, Some(&b.live))?;
        if k % 2 == 1 { for (ch, x) in &parts { x.launch(&io, *ch)?; } }
        s.commit(&b)?;
        next = s.block(23)?;
        // every third step: warm the next block before this one's rows are checked (the next submit joins / hits)
        if k % 3 == 0 { io.warm(&next.ids, Some(&next.live), true)?; }
        dev.synchronize()?;
        let g = fetch(dev, &io, &parts, tks[0].n_tok)?;
        let c = cpu.rows(&own, &b.ids, &b.live)?;
        let (br, bc) = compare(&g, &c);
        bad += br + bc;
        if let Some(p) = &prev {
            if p.0.len() == c.0.len() { neg += compare(p, &c).0; }
        }
        prev = Some(g);
        for tk in &tks {
            n_rows += tk.rows;
            hits += tk.hits;
            misses += tk.misses;
            joined += tk.joined;
            dead += tk.dead;
        }
    }
    let errs: Vec<u32> = (0..nl).map(|l| io.gpu_ctrl(ch_decode(l)).err).collect();
    let cs = io.cache_stats();
    println!("identity decode: {n_steps} rounds x {lanes} lanes x {rows} rows ({} positions), {n_rows} rows over {nl} per-layer channels: {hits} hits \
              ({joined} joined an in-flight read), {misses} read, {dead} dead; cache {} slots, {} evictions; GPU err {errs:?}; \
              rows or records differing from the CPU path: {bad}", n_steps * lanes * rows, io.n_slots(), cs.evictions);
    let mut good = bad == 0 && errs.iter().all(|e| *e == 0) && cs.evictions > 0 && joined + hits > 0 && dead > 0;
    // prefill channel: chunks of text with the read-ahead of the next chunk
    let mut st = NgramHashState::new();
    let ipt = hasher.ids_per_token();
    let n_chunks = 6;
    let mut chunk_ids: Vec<(Vec<u32>, Vec<bool>)> = Vec::new();
    for c in 0..n_chunks {
        let toks = &corpus[c * chunk..(c + 1) * chunk];
        let live: Vec<bool> = (0..chunk).map(|i| (i + c) % 101 != 7).collect();
        let mut ids = vec![0u32; chunk * ipt];
        hasher.hash_chunk(&mut st, toks, Some(&live), &mut ids)?;
        chunk_ids.push((ids, live));
    }
    let (mut pbad, mut pmiss) = (0usize, 0usize);
    io.warm(&chunk_ids[0].0, Some(&chunk_ids[0].1), false)?;
    for c in 0..n_chunks {
        let tk = io.submit_prefill(&chunk_ids[c].0, Some(&chunk_ids[c].1))?;
        if c + 1 < n_chunks { io.warm(&chunk_ids[c + 1].0, Some(&chunk_ids[c + 1].1), false)?; }
        pout.launch(&io, cp)?;
        dev.synchronize()?;
        let g = fetch(dev, &io, &[(cp, &pout)], tk.n_tok)?;
        let cr = cpu.rows(&own, &chunk_ids[c].0, &chunk_ids[c].1)?;
        let (br, bc) = compare(&g, &cr);
        pbad += br + bc;
        pmiss += tk.misses;
        // the gather plug-in: decode the step's raw records with the records kernel, same bytes as the CPU decode
        if c == n_chunks - 1 {
            let n = g.1.len() / 264;
            let rec_d = dev.htod_sync_copy(&g.1)?;
            let out_d = dev.alloc_zeros::<u16>(n * 256)?;
            ge::launch_records_decode(dev, *rec_d.device_ptr(), 264, n as u32, *out_d.device_ptr())?;
            let rd = dev.dtoh_sync_copy(&out_d)?;
            let rbad = rd.chunks(256).zip(cr.0.chunks(256)).filter(|(a, b)| a.iter().zip(b.iter()).any(|(x, y)| !ge::bf16_same(*x, *y))).count();
            println!("identity records-decode kernel (gather plug-in): {n} records, rows differing from the CPU decode: {rbad}");
            good &= rbad == 0;
        }
    }
    let pk = io.gpu_ctrl(cp);
    println!("identity prefill: {n_chunks} chunks x {chunk} positions with read-ahead of the next chunk; rows read at submit (not ahead): {pmiss}; \
              GPU err {}; rows or records differing from the CPU path: {pbad}", pk.err);
    good &= pbad == 0 && pk.err == 0;
    drop(io);
    // no cache: every row from disk
    let mut io0 = EngramIo::open(dev, layout.clone(), base_cfg(o, lanes * rows, 0, 0))?;
    let outs0: Vec<Out> = (0..nl).map(|l| Out::new(dev, &io0, ch_decode(l))).collect::<Result<_>>()?;
    let parts0: Vec<(usize, &Out)> = outs0.iter().enumerate().map(|(l, x)| (ch_decode(l), x)).collect();
    let mut s0 = Streams::new(hasher, corpus, lanes, rows, o.accept, false, 0xE11);
    let mut bad0 = 0;
    let mut reads0 = 0;
    for _ in 0..20 {
        let b = s0.block(0)?;
        let tks = io0.submit_decode(&b.ids, Some(&b.live))?;
        for (ch, x) in &parts0 { x.launch(&io0, *ch)?; }
        dev.synchronize()?;
        let g = fetch(dev, &io0, &parts0, tks[0].n_tok)?;
        let (br, bc) = compare(&g, &cpu.rows(&own, &b.ids, &b.live)?);
        bad0 += br + bc;
        reads0 += tks.iter().map(|t| t.misses).sum::<usize>();
        s0.commit(&b)?;
    }
    println!("identity without a cache (--engram-row-cache-mib 0): 20 rounds, {reads0} rows read, rows or records differing: {bad0}");
    good &= bad0 == 0 && reads0 == 20 * lanes * rows * nl * io0.n_own();
    println!("identity negative control: rows of round k against the CPU rows of round k+1: {neg} differ (must be > 0)");
    good &= neg > 0;
    println!("identity: {:.1} s; {}", t0.elapsed().as_secs_f64(), if good { "DSV41_ENGRAM_IO_IDENTITY_OK" } else { "DSV41_ENGRAM_IO_IDENTITY_FAIL" });
    Ok(good)
}

/// Per-round latency: tokens known (submit) -> rows decoded on the GPU (the channel's done word), for the
/// first Engram layer (the critical rows) and for both layers; the host part (submit -> published) and the
/// GPU part (publish seen -> decoded) in brackets. The waits are queued before the tokens are known, as in a
/// round. Rounds are spaced `period` apart.
fn latency(o: &Opts, hasher: &EngramHasher, layout: &Arc<super::layout::EngramLayout>, corpus: &[u32], dev: &Arc<CudaDevice>) -> Result<bool> {
    let n = o.steps.max(100);
    println!("latency: per round, tokens known -> rows decoded on the GPU, rounds every {} us, {n} rounds per cell, heads {:?}, readers {}, keep-alive {} us",
             o.period_us, o.heads, o.readers, o.keepalive_us);
    println!("| lanes x rows | rows / round (layer 1 + layer 14) | tokens | cache | hit rate | layer-1 rows p50 / p99 / max us | both layers p50 / p99 / max us | [host publish p50 / p99 us, GPU decode avg / max us] |\n|---|---:|---|---|---:|---|---|---|");
    for &lanes in &[1usize, 4] {
        for &(random, cache_mib) in &[(true, 0usize), (false, o.cache_mib)] {
            let mut io = EngramIo::open(dev, layout.clone(), base_cfg(o, lanes * o.rows, 0, cache_mib))?;
            let nl = io.n_layers();
            let outs: Vec<Out> = (0..nl).map(|l| Out::new(dev, &io, ch_decode(l))).collect::<Result<_>>()?;
            let mut s = Streams::new(hasher, corpus, lanes, o.rows, o.accept, random, 0xA1 + lanes as u64);
            let warm = if random { 20 } else { 400 };
            let (mut lat0, mut lat_all) = (Vec::with_capacity(n), Vec::with_capacity(n));
            let (mut h, mut m_) = (0usize, 0usize);
            let period = Duration::from_micros(o.period_us);
            let mut next_t = Instant::now();
            let (mut dsum0, mut dn0) = (0u64, 0u64);
            for k in 0..warm + n {
                spin_until(next_t);
                let b = s.block(0)?;
                for (l, x) in outs.iter().enumerate() { x.launch(&io, ch_decode(l))?; }
                let t0 = Instant::now();
                next_t = t0 + period;
                let tks = io.submit_decode(&b.ids, None)?;
                while io.consumed(ch_decode(0)) <= tks[0].seq { std::hint::spin_loop(); }
                let d0 = t0.elapsed().as_nanos() as u64;
                while io.consumed(ch_decode(nl - 1)) <= tks[nl - 1].seq { std::hint::spin_loop(); }
                let da = t0.elapsed().as_nanos() as u64;
                s.commit(&b)?;
                if k >= warm {
                    lat0.push(d0);
                    lat_all.push(da);
                    for tk in &tks { h += tk.hits; m_ += tk.misses; }
                }
                if k + 1 == warm {
                    io.take_latencies(ch_decode(0));
                    let c = io.gpu_ctrl(ch_decode(0));
                    (dsum0, dn0) = (c.sum_decode_ns, c.steps);
                }
            }
            dev.synchronize()?;
            let mut host = io.take_latencies(ch_decode(0));
            let c = io.gpu_ctrl(ch_decode(0));
            let dec_avg = (c.sum_decode_ns - dsum0) as f64 / (c.steps - dn0).max(1) as f64 / 1e3;
            let rows = nl * lanes * o.rows * io.n_own();
            let hr = h as f64 / (h + m_).max(1) as f64;
            let mx = |v: &Vec<u64>| v.iter().copied().max().unwrap_or(0) as f64 / 1e3;
            let (mx0, mxa) = (mx(&lat0), mx(&lat_all));
            println!("| {lanes} x {} | {rows} | {} | {} | {:.1} % | {:.0} / {:.0} / {:.0} | {:.0} / {:.0} / {:.0} | [{:.0} / {:.0}, {:.1} / {:.1}] |", o.rows,
                     if random { "random" } else { "text" }, if cache_mib == 0 { "off".to_string() } else { format!("{cache_mib} MiB") }, 100.0 * hr,
                     pct(&mut lat0, 0.5), pct(&mut lat0, 0.99), mx0, pct(&mut lat_all, 0.5), pct(&mut lat_all, 0.99), mxa,
                     pct(&mut host, 0.5), pct(&mut host, 0.99), dec_avg, c.max_decode_ns as f64 / 1e3);
        }
    }
    println!("DSV41_ENGRAM_IO_LATENCY_OK");
    Ok(true)
}

/// The steady synthetic decode at the target round time; zero stalls is the pass. Round k on the GPU:
/// [cover][layer-1 rows: wait + decode][layers 1..13][layer-14 rows: wait + decode][rest of the verify]
/// [doorbell: sampled token known][drafter][doorbell: drafts known]. The host prefetches the sampled token's
/// rows at the first doorbell (draft prefetch) and submits the round at the second.
fn steady_decode(o: &Opts, hasher: &EngramHasher, layout: &Arc<super::layout::EngramLayout>, corpus: &[u32], dev: &Arc<CudaDevice>,
                 lanes: usize, random: bool) -> Result<bool> {
    let n = o.steps;
    let (period, cover) = (o.period_us * 1000, o.cover_us * 1000);
    // the DSpark drafter at TP4 (T0 §5.2 column A: 2.4 ms): the sampled token is known when it starts
    let drafter = 2_400_000u64.min(period.saturating_sub(cover) / 2);
    let verify = period.saturating_sub(cover + drafter);
    // the second Engram layer (model layer 14) is needed at the layer-13 FFN reduce: 13 of 40 layers in
    let to_l14 = verify * 13 / 40;
    let rest = verify - to_l14;
    let mut io = EngramIo::open(dev, layout.clone(), base_cfg(o, lanes * o.rows, 0, o.cache_mib))?;
    let nl = io.n_layers();
    let outs: Vec<Out> = (0..nl).map(|l| Out::new(dev, &io, ch_decode(l))).collect::<Result<_>>()?;
    let geoms: Vec<ge::StepLaunch> = outs.iter().enumerate().map(|(l, x)| io.step_launch(ch_decode(l), *x.b.device_ptr(), 0, ge::MODE_DECODE)).collect();
    // doorbell: the GPU stores 2k+1 when round k's drafter starts (sampled token known), 2k+2 when it ends
    let bell = ge::MappedHost::new(dev, 64)?;
    let bell_w = unsafe { AtomicU64::from_ptr(bell.host_ptr() as *mut u64) };
    let mut s = Streams::new(hasher, corpus, lanes, o.rows, o.accept, random, 0xDEC0 + lanes as u64);
    let enqueue = |k: u64| -> Result<()> {
        ge::launch_bench_spin(dev, cover, 0, 0)?;
        ge::launch_step_decode(dev, &geoms[0])?;
        ge::launch_bench_spin(dev, to_l14, 0, 0)?;
        for g in &geoms[1..] { ge::launch_step_decode(dev, g)?; }
        ge::launch_bench_spin(dev, rest, bell.dev_ptr(), 2 * k + 1)?;
        ge::launch_bench_spin(dev, drafter, bell.dev_ptr(), 2 * k + 2)?;
        Ok(())
    };
    // warm-up rounds are not counted (a fresh server's cache is cold; X-VL's 83-99 % hits are steady state)
    let warm_steps = if random { 20 } else { 300usize };
    let total = warm_steps + n;
    let mut blocks: std::collections::VecDeque<_> = std::collections::VecDeque::new();
    blocks.push_back(s.block(0)?);
    let ahead = 2u64;
    for k in 0..ahead.min(total as u64) { enqueue(k)?; }
    let (mut h, mut mi, mut j) = ([0usize; 2], [0usize; 2], [0usize; 2]);
    let mut base: Vec<ge::Ctrl> = Vec::new();
    let mut rbase = [0u64; 8];
    let t_start = Instant::now();
    for k in 0..total {
        if k == warm_steps {
            base = (0..nl).map(|l| io.gpu_ctrl(ch_decode(l))).collect();
            rbase = io.reader_totals();
            io.take_latencies(ch_decode(0));
            (h, mi, j) = ([0; 2], [0; 2], [0; 2]);
        }
        if k > 0 {
            // round k-1's drafter starts: the sampled token of block k is known -> draft prefetch
            let want = 2 * (k as u64 - 1) + 1;
            while bell_w.load(Ordering::Acquire) < want { std::hint::spin_loop(); }
            if o.draft_prefetch { io.warm(&s.sampled_ids(blocks.back().unwrap()), None, true)?; }
            while bell_w.load(Ordering::Acquire) < want + 1 { std::hint::spin_loop(); }
        }
        let b = blocks.pop_front().unwrap();
        let tks = io.submit_decode(&b.ids, None)?;
        for (l, tk) in tks.iter().enumerate().take(2) {
            h[l] += tk.hits;
            mi[l] += tk.misses;
            j[l] += tk.joined;
        }
        s.commit(&b)?;
        blocks.push_back(s.block(0)?);
        if (k as u64 + ahead) < total as u64 { enqueue(k as u64 + ahead)?; }
    }
    dev.synchronize()?;
    let wall = t_start.elapsed().as_secs_f64();
    let mut host = io.take_latencies(ch_decode(0));
    let rt = io.reader_totals();
    let rows = lanes * o.rows * io.n_own();
    println!("steady decode: {n} rounds (+{warm_steps} warm-up) of {} us, {lanes} lanes x {} rows on heads {:?} = {rows} rows per Engram layer per round, tokens {}, \
              accept {} % per draft, draft prefetch {}, readers {}, keep-alive {} us; layer-1 rows needed {} us after the drafts are known, layer-14 rows {:.0} us; {:.1} s wall",
             o.period_us, o.rows, o.heads, if random { "random" } else { "text" }, o.accept, if o.draft_prefetch { "on" } else { "off" }, o.readers, o.keepalive_us,
             o.cover_us, (cover + to_l14) as f64 / 1e3, wall);
    let mut good = true;
    for l in 0..nl.min(2) {
        let c = io.gpu_ctrl(ch_decode(l));
        let (st, sns, dn) = (c.stalls - base[l].stalls, c.stall_ns - base[l].stall_ns, c.steps - base[l].steps);
        let dec = (c.sum_decode_ns - base[l].sum_decode_ns) as f64 / dn.max(1) as f64 / 1e3;
        println!("steady decode, Engram layer {} (model layer {}): rows per round {:.1} cached + {:.1} joined a prefetch + {:.1} read at submit; \
                  GPU stalls {st} (waited {:.0} us in total, max {:.0} us since boot), timeouts {}, err {}; GPU decode {dec:.1} us avg",
                 l, layout.cfg.layer_ids[l], (h[l] - j[l]) as f64 / n as f64, j[l] as f64 / n as f64, mi[l] as f64 / n as f64,
                 sns as f64 / 1e3, c.max_stall_ns as f64 / 1e3, c.timeouts, c.err);
        good &= st == 0 && c.timeouts == 0 && c.err == 0;
    }
    println!("steady decode: row reads {} (draft prefetch included), slower than 500 us submit -> completion: {}, longest read since boot {:.0} us",
             rt[0] - rbase[0], rt[6] - rbase[6], rt[7] as f64 / 1e3);
    println!("steady decode: layer-1 rows host submit -> published p50 {:.0} us, p99 {:.0} us, max {:.0} us -> {}", pct(&mut host.clone(), 0.5),
             pct(&mut host.clone(), 0.99), host.iter().copied().max().unwrap_or(0) as f64 / 1e3,
             if good { "DSV41_ENGRAM_IO_DECODE_OK" } else { "DSV41_ENGRAM_IO_DECODE_STALLED" });
    host.clear();
    Ok(good)
}

fn prefill(o: &Opts, hasher: &EngramHasher, layout: &Arc<super::layout::EngramLayout>, corpus: &[u32], dev: &Arc<CudaDevice>) -> Result<bool> {
    let chunk = 16384usize.min(corpus.len() / 2);
    let n_chunks = (corpus.len() / chunk).min(4);
    let mut io = EngramIo::open(dev, layout.clone(), base_cfg(o, 1, chunk, o.cache_mib.max(1024)))?;
    let cp = io.ch_prefill();
    let out = Out::new(dev, &io, cp)?;
    let ipt = hasher.ids_per_token();
    let mut st = NgramHashState::new();
    let mut good = true;
    println!("prefill read-ahead: {n_chunks} chunks of {chunk} text tokens, heads {:?} ({} rows per chunk before cache hits), readers {}",
             o.heads, io.n_layers() * chunk * io.n_own(), o.readers);
    println!("| chunk | rows read ahead | read-ahead time ms | rows/s | tok/s it can feed | rows read at submit |\n|---:|---:|---:|---:|---:|---:|");
    for c in 0..n_chunks {
        let toks = &corpus[c * chunk..(c + 1) * chunk];
        let mut ids = vec![0u32; chunk * ipt];
        hasher.hash_chunk(&mut st, toks, None, &mut ids)?;
        let t0 = Instant::now();
        let reads = io.warm(&ids, None, false)?;
        while io.warm_pending() > 0 {
            io.housekeep();
            std::hint::spin_loop();
        }
        let dt = t0.elapsed().as_secs_f64();
        let tk = io.submit_prefill(&ids, None)?;
        out.launch(&io, cp)?;
        dev.synchronize()?;
        println!("| {c} | {reads} | {:.1} | {:.0} | {:.0} | {} |", dt * 1e3, reads as f64 / dt, chunk as f64 / dt, tk.misses);
        good &= tk.misses == 0 && io.gpu_ctrl(cp).err == 0;
    }
    println!("{}", if good { "DSV41_ENGRAM_IO_PREFILL_OK" } else { "DSV41_ENGRAM_IO_PREFILL_FAIL" });
    Ok(good)
}
