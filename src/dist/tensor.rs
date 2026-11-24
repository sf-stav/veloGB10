//! `ShardedTensor<B>` — placement-tracked tensor wrapper over either a GPU device buffer ([`GpuBuf`])
//! or a host buffer (`Vec<f32>` for deterministic multi-rank loopback testing).

use super::placement::{local_shape_for, DType, Mesh, Placement};
use anyhow::{ensure, Result};

/// Lightweight non-owning GPU device buffer descriptor (capture-safe: plain `u64` device pointer + element count).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuBuf {
    pub ptr: u64,
    pub elems: usize,
}

impl GpuBuf {
    pub const fn new(ptr: u64, elems: usize) -> Self {
        Self { ptr, elems }
    }
    pub const fn byte_len(self, dtype: DType) -> usize {
        self.elems * dtype.elem_bytes()
    }
    /// Sub-slice starting at `elem_offset` with `elems` elements.
    pub fn slice(self, elem_offset: usize, elems: usize, dtype: DType) -> Result<Self> {
        ensure!(
            elem_offset.checked_add(elems).map_or(false, |end| end <= self.elems),
            "GpuBuf::slice [{elem_offset}..+{elems}) out of bounds for {} elems",
            self.elems
        );
        Ok(Self {
            ptr: self.ptr + (elem_offset * dtype.elem_bytes()) as u64,
            elems,
        })
    }
}

/// A tensor annotated with its global shape, local shard shape, element dtype, [`Mesh`], and [`Placement`].
#[derive(Clone, Debug)]
pub struct ShardedTensor<B> {
    pub global_shape: Vec<usize>,
    pub local_shape: Vec<usize>,
    pub dtype: DType,
    pub placement: Placement,
    pub mesh: Mesh,
    pub buf: B,
}

impl<B> ShardedTensor<B> {
    /// Construct a `ShardedTensor` validating that `local_shape` matches `(global_shape, mesh, placement, align)`.
    pub fn new(
        global_shape: Vec<usize>,
        dtype: DType,
        placement: Placement,
        mesh: Mesh,
        align: usize,
        buf: B,
    ) -> Result<Self> {
        let local_shape = local_shape_for(&global_shape, mesh, placement, align)?;
        Ok(Self {
            global_shape,
            local_shape,
            dtype,
            placement,
            mesh,
            buf,
        })
    }

    pub fn local_elems(&self) -> usize {
        self.local_shape.iter().product()
    }

    pub fn global_elems(&self) -> usize {
        self.global_shape.iter().product()
    }

    pub fn local_bytes(&self) -> usize {
        self.local_elems() * self.dtype.elem_bytes()
    }

    pub const fn is_partial(&self) -> bool {
        self.placement.is_partial()
    }

    pub const fn is_replicated(&self) -> bool {
        self.placement.is_replicated()
    }

    /// Assert that this tensor is `Replicated` before passing it to a replicated consumer.
    pub fn assert_replicated(&self, op: &str) -> Result<()> {
        ensure!(
            self.placement == Placement::Replicated || self.mesh.is_solo(),
            "{op}: expected Replicated tensor, got {:?} on world {}",
            self.placement,
            self.mesh.world
        );
        Ok(())
    }

    /// Assert that this tensor is sharded along `axis`.
    pub fn assert_sharded(&self, axis: usize, op: &str) -> Result<()> {
        if self.mesh.is_solo() {
            return Ok(());
        }
        ensure!(
            self.placement == Placement::Shard { axis },
            "{op}: expected Shard {{ axis: {axis} }}, got {:?}",
            self.placement
        );
        Ok(())
    }
}

impl ShardedTensor<GpuBuf> {
    /// Construct a GPU-backed `ShardedTensor`, asserting `buf.elems >= local_elems()`.
    pub fn from_gpu(
        global_shape: Vec<usize>,
        dtype: DType,
        placement: Placement,
        mesh: Mesh,
        align: usize,
        buf: GpuBuf,
    ) -> Result<Self> {
        let t = Self::new(global_shape, dtype, placement, mesh, align, buf)?;
        ensure!(
            t.buf.elems >= t.local_elems(),
            "ShardedTensor<GpuBuf>: buffer has {} elems, need >= {} for local_shape {:?}",
            t.buf.elems,
            t.local_elems(),
            t.local_shape
        );
        Ok(t)
    }

    pub fn dev_ptr(&self) -> u64 {
        self.buf.ptr
    }
}

impl ShardedTensor<Vec<f32>> {
    /// Construct a host FP32 `ShardedTensor` (used by loopback multi-rank tests).
    pub fn from_host_f32(
        global_shape: Vec<usize>,
        placement: Placement,
        mesh: Mesh,
        align: usize,
        buf: Vec<f32>,
    ) -> Result<Self> {
        let t = Self::new(global_shape, DType::F32, placement, mesh, align, buf)?;
        ensure!(
            t.buf.len() == t.local_elems(),
            "ShardedTensor<Vec<f32>>: len {} != local_elems {} ({:?})",
            t.buf.len(),
            t.local_elems(),
            t.local_shape
        );
        Ok(t)
    }
}
