//! DSV4.1 TP (leg TPC): **the boundary between the per-layer forward (leg SL, `src/dsv41/gpu/**`) and TP.**
//!
//! The forward hands TP its partials; TP returns the reduced tensors. Concretely, for every collective `Site`
//! of the round (`frame::Shape::round`):
//! 1. `frame(site, rows)` gives the device frame (`FrameView`); the producer kernels write this rank's FP32
//!    partial into its sections (`hidden`, `mix`, `extra`; the layout is `frame::FrameSpec`);
//! 2. `reduce(stream, site, rows)` sums the frame over the ranks in place, in the transport's fixed association,
//!    enqueued on the compute stream (no host sync, graph-capturable);
//! 3. the consumer reads the reduced tensors from the same frame (bit-identical on every rank), before the
//!    next site of the same parity (`Attn` / `DraftAttn` use frame A, `Ffn` / `DraftFfn` frame B).
//!
//! The head never moves logits: `head` takes the rank's processed-logit shard (`vocab()`: whole 128-row
//! blocks, `load::splits::vocab_range`) and writes per-row `TpOut` (greedy, sample, log-sum, probe log-prob,
//! probe log-prob). Accepting drafts and drawing the residual of a rejected row is DS's sampler (`gpu/dspark/vocab.rs`, D-DSV41-5: ONE sampler ships; leg SE2 removed the duplicate accept / residual path from this exchange). `TpComm::solo` is the world-1 instance, so
//! the single-box forward calls the same trait (its reduces are no-ops).

use super::frame::{FrameSpec, Shape, Site, HEAD_REC, MAX_ROWS};
use super::kern::{LocalArgs, TpKernels};
use super::rdma::RdmaXport;
use anyhow::{ensure, Result};
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, DevicePtr, DeviceSlice};
use std::sync::Arc;

/// One row's sampling parameters (`d41tp_row`, 32 B).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TpRow {
    pub seed: u64,
    /// absolute position of the token this row samples (RNG streams pos*3 + {0 fresh, 1 residual, 2 accept})
    pub pos: u64,
    /// 0 = greedy
    pub temp: f32,
    /// the draft token this row verifies (-1 = none)
    pub probe: i32,
    /// bit 0: a drafter draw (seed + 1,000,000,007)
    pub flags: u32,
    pub pad: u32,
}

/// One row's result (`d41tp_out`, 32 B).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TpOut {
    pub greedy: i32,
    pub sample: i32,
    pub lse: f32,
    pub max_x: f32,
    pub probe_lp: f32,
    /// unused since leg SE2 (-1): the accept decision is DS's sampler's
    pub accept: i32,
    /// unused since leg SE2 (-1): the residual draw is DS's sampler's
    pub residual: i32,
    /// bit 0 zero residual mass; bits 8.. non-finite logits seen
    pub flags: u32,
}

impl TpOut {
    pub fn bits(&self) -> [u32; 8] {
        [self.greedy as u32, self.sample as u32, self.lse.to_bits(), self.max_x.to_bits(), self.probe_lp.to_bits(),
         self.accept as u32, self.residual as u32, self.flags]
    }
}

/// A site's device frame.
#[derive(Clone, Copy, Debug)]
pub struct FrameView {
    pub ptr: u64,
    pub spec: FrameSpec,
}
impl FrameView {
    pub fn hidden(&self) -> u64 { self.ptr + (self.spec.hidden_off() * 4) as u64 }
    pub fn mix(&self) -> u64 { self.ptr + (self.spec.mix_off() * 4) as u64 }
    pub fn extra(&self) -> u64 { self.ptr + (self.spec.extra_off() * 4) as u64 }
}

/// A head exchange: the rank's processed-logit shard rows (`logits + r * ld`, ids `vocab()`), the row parameters
/// and outputs on the device, and (for an accept decision) the drafter exchange's outputs for the same rows.
#[derive(Clone, Copy, Debug)]
pub struct HeadCall {
    pub site: Site,
    pub logits: u64,
    pub ld: usize,
    pub rows: usize,
    pub rows_p: u64,
    pub out: u64,
}

/// What TP offers the per-layer forward. Implemented by `TpComm` (world 1 = solo, 2 / 4 = RDMA).
pub trait Dsv41Comm {
    fn rank(&self) -> usize;
    fn world(&self) -> usize;
    /// this rank's vocabulary ids (whole 128-row blocks)
    fn vocab(&self) -> std::ops::Range<usize>;
    /// the frame the producer writes `site`'s partial into, at `rows`
    fn frame(&self, site: Site, rows: usize) -> Result<FrameView>;
    /// sum `site`'s frame over the ranks, in place, on `stream`
    fn reduce(&self, stream: &CudaStream, site: Site, rows: usize) -> Result<()>;
    /// the vocab-parallel head exchange
    fn head(&self, stream: &CudaStream, call: &HeadCall) -> Result<()>;
    /// fail loudly if the transport aborted (a partial sum must never be served)
    fn check(&self, what: &str) -> Result<()>;
}

enum Xport {
    Solo,
    Rdma(RdmaXport),
}

/// The comm layer of one rank.
pub struct TpComm {
    pub rank: usize,
    pub world: usize,
    pub shape: Shape,
    vocab: std::ops::Range<usize>,
    pub kern: TpKernels,
    x: Xport,
    frames: [CudaSlice<f32>; 2],
    head_rec: CudaSlice<f32>,
    /// the head kernels' per-(row, chunk) scratch
    part: CudaSlice<f32>,
}

fn dptr<T>(s: &CudaSlice<T>) -> u64 { *s.device_ptr() }

impl TpComm {
    fn build(dev: &Arc<CudaDevice>, shape: Shape, rank: usize, world: usize, x: Xport) -> Result<Self> {
        let kern = TpKernels::load(dev)?;
        let max = [1usize, 6, 12, MAX_ROWS].iter().map(|&r| shape.round(r, false).max_floats(&shape, world)).max().unwrap_or(0);
        let alloc = |n: usize| -> Result<CudaSlice<f32>> { let mut b = dev.alloc_zeros::<f32>(n)?; dev.memset_zeros(&mut b)?; Ok(b) };
        let vocab = crate::dsv41::load::splits::vocab_range(shape.vocab, world, rank);
        let c = TpComm { rank, world, vocab, kern, x,
                         frames: [alloc(max)?, alloc(max)?],
                         head_rec: alloc(MAX_ROWS * world * HEAD_REC)?,
                         part: alloc(MAX_ROWS * super::kern::chunks(shape.vocab) * HEAD_REC)?, shape };
        dev.synchronize()?;
        Ok(c)
    }

    /// World 1: every reduce is a no-op, the head exchange combines one record.
    pub fn solo(dev: &Arc<CudaDevice>, shape: Shape) -> Result<Self> { Self::build(dev, shape, 0, 1, Xport::Solo) }

    /// Attach a brought-up link (`TpContext::bring_up_head` / `bring_up_node`).
    pub fn attach(dev: &Arc<CudaDevice>, shape: Shape, ctx: crate::tp::TpContext) -> Result<Self> {
        let row_bytes = (shape.hidden + super::frame::MIX_STRIDE) * 4;
        let x = RdmaXport::attach(dev, ctx, row_bytes)?;
        let (rank, world) = (x.rank, x.world);
        Self::build(dev, shape, rank, world, Xport::Rdma(x))
    }

    /// The once-per-boot receipt naming the transport path (AGENTS §4).
    pub fn describe(&self) -> String {
        match &self.x {
            Xport::Solo => "dsv41 TP: world 1 (no transport; reduces are no-ops)".into(),
            Xport::Rdma(x) => x.describe.clone(),
        }
    }

    /// True when dual-rail RDMA is active on this rank's transport.
    pub fn dual_on(&self) -> bool {
        match &self.x { Xport::Solo => false, Xport::Rdma(x) => x.dual_on() }
    }

    /// The transport path a frame of `bytes` takes (for the report).
    pub fn path_name(&self, bytes: usize) -> &'static str {
        match &self.x {
            Xport::Solo => "solo",
            Xport::Rdma(x) => match x.path(bytes) { super::rdma::Path::Pipe => "pipe", super::rdma::Path::Single => "single", super::rdma::Path::Serial => "pair" },
        }
    }

    fn frame_buf(&self, site: Site) -> &CudaSlice<f32> {
        match site { Site::Attn(_) | Site::DraftAttn(_) => &self.frames[0], _ => &self.frames[1] }
    }

    /// In-place exact fp32 all-reduce of `n` floats at `base` (gather frames, control/sampler/token payloads).
    pub fn all_reduce_raw(&self, stream: &CudaStream, base: u64, n: usize) -> Result<()> {
        super::metrics::collective(n * 4);
        match &self.x { Xport::Solo => Ok(()), Xport::Rdma(x) => x.all_reduce(stream, base, n) }
    }

    /// In-place neural activation/residual partial-sum all-reduce (uses BF16 wire when `--dsv41-wire-bf16 on`).
    pub fn all_reduce_act(&self, stream: &CudaStream, base: u64, n: usize) -> Result<()> {
        super::metrics::collective(n * 4);
        match &self.x { Xport::Solo => Ok(()), Xport::Rdma(x) => x.all_reduce_act(stream, base, n) }
    }

    /// True when sequence-parallel prefill collectives are supported on this rank's transport (`world` 2, 3, 4, 6, or 8).
    pub fn sp_ready(&self) -> bool {
        match &self.x { Xport::Solo => false, Xport::Rdma(x) => x.sp_ready() }
    }

    /// U6a: true when 1-RTT All-to-All (`TP_LEN_FLAG_A2A`) is ready on this rank's transport.
    pub fn a2a_ready(&self) -> bool {
        match &self.x { Xport::Solo => false, Xport::Rdma(x) => x.a2a_ready() }
    }

    /// U6a: 1-RTT dual-rail All-to-All across all ranks (`src` and `dst` each hold `world` slices of `slice_bytes`).
    pub fn all_to_all(&self, stream: &CudaStream, src: u64, dst: u64, slice_bytes: usize) -> Result<()> {
        match &self.x {
            Xport::Solo => Ok(()),
            Xport::Rdma(x) => {
                x.all_to_all(stream, src, dst, slice_bytes)?;
                super::metrics::collective((self.world - 1) * slice_bytes);
                Ok(())
            }
        }
    }

    /// Leg P7SP: row-block in-place FP32 reduce-scatter across `blocks` (`blocks[self.rank]` receives the canonical sum).
    pub fn sp_reduce_scatter(&self, stream: &CudaStream, base: u64, row_len: usize, blocks: &[(usize, usize)]) -> Result<()> {
        match &self.x {
            Xport::Solo => Ok(()),
            Xport::Rdma(x) => {
                let bytes = x.sp_reduce_scatter(stream, base, row_len, blocks)?;
                super::metrics::collective(bytes);
                Ok(())
            }
        }
    }

    /// Leg P7SP: row-block byte-exact all-gather of `(base, row_bytes)` buffers across `blocks`.
    pub fn sp_all_gather_multi(&self, stream: &CudaStream, bufs: &[(u64, usize)], blocks: &[(usize, usize)]) -> Result<()> {
        match &self.x {
            Xport::Solo => Ok(()),
            Xport::Rdma(x) => {
                let bytes = x.sp_all_gather_multi(stream, bufs, blocks)?;
                super::metrics::collective(bytes);
                Ok(())
            }
        }
    }

    pub fn head_rec_ptr(&self) -> u64 { dptr(&self.head_rec) }
    /// the head kernels' scratch (for a caller that runs them outside `head`, e.g. the smoke's single-GPU reference)
    pub fn part_ptr(&self) -> u64 { dptr(&self.part) }
}

impl Dsv41Comm for TpComm {
    fn rank(&self) -> usize { self.rank }
    fn world(&self) -> usize { self.world }
    fn vocab(&self) -> std::ops::Range<usize> { self.vocab.clone() }

    fn frame(&self, site: Site, rows: usize) -> Result<FrameView> {
        let spec = self.shape.frame(site, rows, self.world);
        ensure!(!spec.is_gather(), "{site:?} is a head-style exchange: use head()");
        let buf = self.frame_buf(site);
        ensure!(rows >= 1 && spec.floats() <= buf.len(), "{site:?} at {rows} rows: {} floats exceed the frame ({})", spec.floats(), buf.len());
        Ok(FrameView { ptr: dptr(buf), spec })
    }

    fn reduce(&self, stream: &CudaStream, site: Site, rows: usize) -> Result<()> {
        let f = self.frame(site, rows)?;
        self.all_reduce_raw(stream, f.ptr, f.spec.floats())
    }

    fn head(&self, stream: &CudaStream, c: &HeadCall) -> Result<()> {
        ensure!(c.rows >= 1 && c.rows <= MAX_ROWS, "head exchange: {} rows (1..{MAX_ROWS})", c.rows);
        let rec = dptr(&self.head_rec);
        let a = LocalArgs { logits: c.logits, ld: c.ld as i64, rows_p: c.rows_p, rec, part: dptr(&self.part), vlen: self.vocab.len() as i32,
                            voff: self.vocab.start as i32, rank: self.rank as i32, world: self.world as i32, zero_others: 1, ..Default::default() };
        self.kern.head_local(stream, c.rows, a)?;
        self.all_reduce_raw(stream, rec, c.rows * self.world * HEAD_REC)?;
        self.kern.head_combine(stream, rec, c.rows, self.world, c.rows_p, c.out)
    }

    fn check(&self, what: &str) -> Result<()> {
        if let Xport::Rdma(x) = &self.x {
            let c = x.abort_code();
            if c != 0 { super::metrics::abort(); }
            ensure!(c == 0, "dsv41 TP rank {}: the transport ABORTED (code {c:#x}), noticed {what}: every reduce since returned a partial sum, \
                             so this rank stops (watchdog dump: /tmp/tp_wdog.rank{}.dump)", self.rank, self.rank);
        }
        Ok(())
    }
}
