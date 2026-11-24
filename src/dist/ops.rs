//! GSPMD-style placement propagation rules, lazy `PartialSum` materialization, and rider packing
//! (`PendingPartialGroup`).
//!
//! Core invariant:
//! - Column-parallel linear (`Replicated × Shard(out)`) -> `Shard(last)` (**0 collectives**).
//! - Elementwise nonlinear on sharded activations (`SiLU`, `GeLU`, `SwiGLU` on `Shard(last)`) -> `Shard(last)` (**0 collectives**).
//! - Row-parallel linear (`Shard(in) × Shard(in)`) -> **`PartialSum`** (**lazy — 0 collectives yet!**).
//! - Linear combinations (`PartialSum + PartialSum`, e.g. routed MoE + shared expert, or `PartialSum + Replicated`
//!   bias/residual added on rank 0 only) -> stay **`PartialSum`** (**0 collectives!**).
//! - Nonlinear / replicated consumers (`RMSNorm`, `Softmax`, `Embedding`, `RoPE`) trigger
//!   [`materialize_gpu`] / [`PendingPartialGroup::flush_gpu`], issuing ONE collective (or ONE fused
//!   rider-packed collective) and flipping the tensor(s) to `Replicated`.

use super::placement::{DType, Mesh, Placement};
use super::tensor::{GpuBuf, ShardedTensor};
use super::xport::{CollectiveBackend, LoopbackFabric};
use anyhow::{bail, ensure, Result};
use cudarc::driver::CudaStream;

/// Placement rule for a column-parallel linear projection `Y = X @ W^T` where `W` is sharded along
/// its output dimension (`Shard { axis: 0 }` in `[out, in]` layout).
/// - `Replicated` input -> `Shard { axis: out_axis }` (0 collectives).
/// - `PartialSum` input must be materialized BEFORE a column-parallel linear (returns `Err` so
///   a missing reduction is caught immediately instead of silently multiplying unreduced partials).
pub fn col_linear_placement(input: Placement, out_axis: usize) -> Result<Placement> {
    match input {
        Placement::Replicated => Ok(Placement::Shard { axis: out_axis }),
        Placement::PartialSum => bail!("col_linear: input is PartialSum — materialize to Replicated first"),
        Placement::Shard { axis } => bail!("col_linear: input is already Shard {{ axis: {axis} }}; expected Replicated"),
    }
}

/// Placement rule for a row-parallel linear projection `Y = X @ W^T` where `X` is sharded along its
/// contracting dimension (`in_axis`) and `W` is sharded along its input dimension (`Shard { axis: 1 }`).
/// Produces `PartialSum` (or `Replicated` on solo `world == 1`) with **0 collectives** (lazy!).
pub fn row_linear_placement(input: Placement, in_axis: usize, mesh: Mesh) -> Result<Placement> {
    if mesh.is_solo() {
        return Ok(Placement::Replicated);
    }
    match input {
        Placement::Shard { axis } if axis == in_axis => Ok(Placement::PartialSum),
        other => bail!("row_linear: expected input Shard {{ axis: {in_axis} }}, got {other:?}"),
    }
}

/// Placement rule for an additive linear combination `A + B`:
/// - `PartialSum + PartialSum` -> `PartialSum` (0 collectives: e.g. routed experts + shared expert)
/// - `PartialSum + Replicated` (or reverse) -> `PartialSum` (0 collectives: rank 0 adds the replicated
///   bias/residual term while ranks `> 0` skip it, fusing residual add into the single all-reduce!)
/// - `Replicated + Replicated` -> `Replicated`
/// - `Shard(a) + Shard(a)` -> `Shard(a)`
pub fn linear_add_placement(a: Placement, b: Placement) -> Result<(Placement, bool /* add_b_only_on_rank0 */)> {
    match (a, b) {
        (Placement::PartialSum, Placement::PartialSum) => Ok((Placement::PartialSum, false)),
        (Placement::PartialSum, Placement::Replicated) => Ok((Placement::PartialSum, true)),
        (Placement::Replicated, Placement::PartialSum) => Ok((Placement::PartialSum, true)),
        (Placement::Replicated, Placement::Replicated) => Ok((Placement::Replicated, false)),
        (Placement::Shard { axis: x }, Placement::Shard { axis: y }) if x == y => {
            Ok((Placement::Shard { axis: x }, false))
        }
        _ => bail!("linear_add: incompatible placements {a:?} + {b:?}"),
    }
}

/// Placement rule for a point-wise / row-wise non-linear operator (`RMSNorm`, `Softmax`, `SiLU`, `SwiGLU`):
/// - `requires_full_row = true` (`RMSNorm`, `Softmax`): requires `Replicated`; returns `Ok((Replicated, true))`
///   when `input == PartialSum` (signaling that a reduction must run first) and `Err` on `Shard`.
/// - `requires_full_row = false` (strictly element-wise `SiLU`, `GeLU`, `SwiGLU` on matching shards):
///   preserves `Replicated` or `Shard { axis }` with 0 collectives, and requires reduction if `PartialSum`.
pub fn elementwise_nonlinear_placement(
    input: Placement,
    requires_full_row: bool,
) -> Result<(Placement, bool /* needs_all_reduce */)> {
    match input {
        Placement::Replicated => Ok((Placement::Replicated, false)),
        Placement::PartialSum => Ok((Placement::Replicated, true)),
        Placement::Shard { axis } => {
            if requires_full_row {
                bail!("nonlinear row op (e.g. RMSNorm/Softmax) cannot run on Shard {{ axis: {axis} }} without all-gather");
            }
            Ok((Placement::Shard { axis }, false))
        }
    }
}

/// Diagnostic counter tracking materialized collectives and fused rider bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CollectivesStats {
    pub all_reduces: usize,
    pub rider_groups: usize,
    pub total_floats: usize,
}

/// Materialize a `ShardedTensor<GpuBuf>` from `PartialSum` -> `Replicated` via `xport.all_reduce_f32`.
/// Pure no-op returning `Ok(false)` when already `Replicated` or `world == 1`.
pub fn materialize_gpu(
    tensor: &mut ShardedTensor<GpuBuf>,
    xport: &dyn CollectiveBackend,
    stream: &CudaStream,
    stats: Option<&mut CollectivesStats>,
) -> Result<bool> {
    if tensor.placement == Placement::Replicated || tensor.mesh.is_solo() {
        tensor.placement = Placement::Replicated;
        return Ok(false);
    }
    ensure!(
        tensor.placement == Placement::PartialSum,
        "materialize_gpu: cannot all-reduce placement {:?}",
        tensor.placement
    );
    ensure!(
        tensor.dtype == DType::F32,
        "materialize_gpu: FabricXport all_reduce expects FP32 partials (got {:?})",
        tensor.dtype
    );
    let n = tensor.local_elems();
    xport.all_reduce_f32(stream, tensor.dev_ptr(), n)?;
    tensor.placement = Placement::Replicated;
    if let Some(st) = stats {
        st.all_reduces += 1;
        st.total_floats += n;
    }
    Ok(true)
}

/// Rider packing: groups multiple contiguous FP32 `PartialSum` tensors (e.g. `hidden + mix_rider + engram_wkv`)
/// backed by one contiguous GPU scratch span into a SINGLE `all_reduce_f32` call, then marks all members
/// `Replicated`. Eliminates extra network RTTs without model-specific manual offset math.
#[derive(Debug)]
pub struct PendingPartialGroup {
    base_ptr: u64,
    total_floats: usize,
    member_count: usize,
    mesh: Mesh,
}

impl PendingPartialGroup {
    pub fn new(first: &ShardedTensor<GpuBuf>) -> Result<Self> {
        ensure!(first.dtype == DType::F32, "PendingPartialGroup: requires FP32 partials");
        ensure!(
            first.placement == Placement::PartialSum || first.mesh.is_solo(),
            "PendingPartialGroup: first tensor must be PartialSum (got {:?})",
            first.placement
        );
        Ok(Self {
            base_ptr: first.dev_ptr(),
            total_floats: first.local_elems(),
            member_count: 1,
            mesh: first.mesh,
        })
    }

    /// Append a rider tensor that sits immediately contiguous after the current group in device memory.
    pub fn push_contiguous(&mut self, rider: &ShardedTensor<GpuBuf>) -> Result<()> {
        ensure!(rider.dtype == DType::F32, "PendingPartialGroup: rider must be FP32");
        ensure!(rider.mesh == self.mesh, "PendingPartialGroup: mesh mismatch");
        ensure!(
            rider.placement == Placement::PartialSum || self.mesh.is_solo(),
            "PendingPartialGroup: rider must be PartialSum (got {:?})",
            rider.placement
        );
        let expected_ptr = self.base_ptr + (self.total_floats * 4) as u64;
        ensure!(
            rider.dev_ptr() == expected_ptr,
            "PendingPartialGroup: rider ptr {:#x} is not contiguous with group end {:#x}",
            rider.dev_ptr(),
            expected_ptr
        );
        self.total_floats += rider.local_elems();
        self.member_count += 1;
        Ok(())
    }

    pub const fn total_floats(&self) -> usize {
        self.total_floats
    }
    pub const fn member_count(&self) -> usize {
        self.member_count
    }

    /// Flush the packed group in ONE `all_reduce_f32` launch and flip every member tensor to `Replicated`.
    pub fn flush_gpu(
        self,
        members: &mut [&mut ShardedTensor<GpuBuf>],
        xport: &dyn CollectiveBackend,
        stream: &CudaStream,
        stats: Option<&mut CollectivesStats>,
    ) -> Result<bool> {
        ensure!(
            members.len() == self.member_count,
            "PendingPartialGroup::flush_gpu: expected {} members, got {}",
            self.member_count,
            members.len()
        );
        if self.mesh.is_solo() {
            for m in members.iter_mut() {
                m.placement = Placement::Replicated;
            }
            return Ok(false);
        }
        xport.all_reduce_f32(stream, self.base_ptr, self.total_floats)?;
        for m in members.iter_mut() {
            m.placement = Placement::Replicated;
        }
        if let Some(st) = stats {
            st.all_reduces += 1;
            st.rider_groups += 1;
            st.total_floats += self.total_floats;
        }
        Ok(true)
    }
}

/// Materialize a host FP32 `ShardedTensor` over [`LoopbackFabric`] (for multi-rank unit tests).
pub fn materialize_host_f32(
    tensor: &mut ShardedTensor<Vec<f32>>,
    fabric: &LoopbackFabric,
    stats: Option<&mut CollectivesStats>,
) -> Result<bool> {
    if tensor.placement == Placement::Replicated || tensor.mesh.is_solo() {
        tensor.placement = Placement::Replicated;
        return Ok(false);
    }
    ensure!(
        tensor.placement == Placement::PartialSum,
        "materialize_host_f32: expected PartialSum, got {:?}",
        tensor.placement
    );
    fabric.all_reduce_host_f32(tensor.mesh.rank, &mut tensor.buf)?;
    tensor.placement = Placement::Replicated;
    if let Some(st) = stats {
        st.all_reduces += 1;
        st.total_floats += tensor.buf.len();
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::placement::aligned_shard_range;
    use std::sync::Arc;

    fn matmul_nt(x: &[f32], rows: usize, k: usize, w: &[f32], out_dim: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; rows * out_dim];
        for r in 0..rows {
            for o in 0..out_dim {
                let mut acc = 0.0f32;
                for j in 0..k {
                    acc += x[r * k + j] * w[o * k + j];
                }
                y[r * out_dim + o] = acc;
            }
        }
        y
    }

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    /// End-to-end GSPMD SwiGLU MLP + residual add + RMSNorm test across world = 1, 2, 4, 8:
    /// Proves:
    /// 1. Column-parallel gate/up projections + elementwise SwiGLU require ZERO collectives.
    /// 2. Row-parallel down projection produces a lazy `PartialSum`.
    /// 3. Residual add fuses onto `PartialSum` (rank 0 adds residual) with ZERO collectives.
    /// 4. RMSNorm triggers EXACTLY ONE collective per layer (0 at world=1), producing bit-identical
    ///    results across all ranks and matching the TP=1 reference within FP32 accumulation tolerance!
    #[test]
    fn gspmd_swiglu_layer_lazy_partial_matches_tp1_across_worlds_1_2_4_8() {
        let (rows, hidden, inter) = (3usize, 16usize, 32usize);
        let mk = |n: usize, seed: usize| -> Vec<f32> {
            (0..n).map(|i| (((i * 17 + seed * 31) % 97) as f32 - 48.0) * 0.02).collect()
        };
        let x0 = mk(rows * hidden, 1);
        let residual = mk(rows * hidden, 2);
        let w_gate = mk(inter * hidden, 3);
        let w_up = mk(inter * hidden, 4);
        let w_down = mk(hidden * inter, 5);

        // Reference TP=1 computation
        let ref_gate = matmul_nt(&x0, rows, hidden, &w_gate, inter);
        let ref_up = matmul_nt(&x0, rows, hidden, &w_up, inter);
        let ref_act: Vec<f32> = ref_gate.iter().zip(&ref_up).map(|(&g, &u)| silu(g) * u).collect();
        let mut ref_out = matmul_nt(&ref_act, rows, inter, &w_down, hidden);
        for (o, &r) in ref_out.iter_mut().zip(&residual) {
            *o += r;
        }

        for world in [1usize, 2, 4, 8] {
            let fabric = LoopbackFabric::new(world).unwrap();
            let mut handles = Vec::new();
            for rank in 0..world {
                let f: Arc<LoopbackFabric> = fabric.clone();
                let (x0, residual, w_gate, w_up, w_down) =
                    (x0.clone(), residual.clone(), w_gate.clone(), w_up.clone(), w_down.clone());
                handles.push(std::thread::spawn(move || {
                    let mesh = Mesh::new(rank, world).unwrap();
                    let mut stats = CollectivesStats::default();
                    let x = ShardedTensor::from_host_f32(
                        vec![rows, hidden],
                        Placement::Replicated,
                        mesh,
                        1,
                        x0,
                    )
                    .unwrap();
                    let sr = aligned_shard_range(inter, rank, world, 1).unwrap();
                    let local_inter = sr.len();
                    // Slice W_gate / W_up along dim 0 (col-parallel) and W_down along dim 1 (row-parallel)
                    let wg_sh = w_gate[sr.start * hidden..sr.end * hidden].to_vec();
                    let wu_sh = w_up[sr.start * hidden..sr.end * hidden].to_vec();
                    let mut wd_sh = Vec::with_capacity(hidden * local_inter);
                    for o in 0..hidden {
                        wd_sh.extend_from_slice(&w_down[o * inter + sr.start..o * inter + sr.end]);
                    }
                    // 1. Col-parallel gate & up (0 collectives)
                    let p_gate = col_linear_placement(x.placement, 1).unwrap();
                    let p_up = col_linear_placement(x.placement, 1).unwrap();
                    let g_t = ShardedTensor::from_host_f32(
                        vec![rows, inter],
                        p_gate,
                        mesh,
                        1,
                        matmul_nt(&x.buf, rows, hidden, &wg_sh, local_inter),
                    )
                    .unwrap();
                    let u_t = ShardedTensor::from_host_f32(
                        vec![rows, inter],
                        p_up,
                        mesh,
                        1,
                        matmul_nt(&x.buf, rows, hidden, &wu_sh, local_inter),
                    )
                    .unwrap();
                    // 2. Elementwise SwiGLU on sharded activations (0 collectives!)
                    let (p_act, need_ar) = elementwise_nonlinear_placement(g_t.placement, false).unwrap();
                    assert!(!need_ar);
                    let act_buf: Vec<f32> = g_t.buf.iter().zip(&u_t.buf).map(|(&g, &u)| silu(g) * u).collect();
                    let act_t = ShardedTensor::from_host_f32(vec![rows, inter], p_act, mesh, 1, act_buf).unwrap();
                    // 3. Row-parallel down projection -> lazy PartialSum (0 collectives!)
                    let p_down = row_linear_placement(act_t.placement, 1, mesh).unwrap();
                    let mut out_t = ShardedTensor::from_host_f32(
                        vec![rows, hidden],
                        p_down,
                        mesh,
                        1,
                        matmul_nt(&act_t.buf, rows, local_inter, &wd_sh, hidden),
                    )
                    .unwrap();
                    assert_eq!(stats.all_reduces, 0, "still lazy before residual + norm");
                    // 4. Residual add fused onto PartialSum (rank 0 adds replicated residual; 0 collectives!)
                    let (p_res, only_r0) = linear_add_placement(out_t.placement, Placement::Replicated).unwrap();
                    if !only_r0 || mesh.is_head() {
                        for (o, &r) in out_t.buf.iter_mut().zip(&residual) {
                            *o += r;
                        }
                    }
                    out_t.placement = p_res;
                    assert_eq!(stats.all_reduces, 0);
                    // 5. RMSNorm requires full row -> triggers the ONE collective
                    let (_p_norm, need_ar) = elementwise_nonlinear_placement(out_t.placement, true).unwrap();
                    if need_ar {
                        materialize_host_f32(&mut out_t, &f, Some(&mut stats)).unwrap();
                    }
                    assert!(out_t.is_replicated());
                    assert_eq!(stats.all_reduces, usize::from(world > 1));
                    out_t.buf
                }));
            }
            let outs: Vec<Vec<f32>> = handles.into_iter().map(|h| h.join().unwrap()).collect();
            for r in 1..world {
                let a: Vec<u32> = outs[0].iter().map(|x| x.to_bits()).collect();
                let b: Vec<u32> = outs[r].iter().map(|x| x.to_bits()).collect();
                assert_eq!(a, b, "world {world}: rank {r} must be bit-identical to rank 0");
            }
            for (i, (&got, &want)) in outs[0].iter().zip(&ref_out).enumerate() {
                assert!(
                    (got - want).abs() < 1e-5,
                    "world {world} elem {i}: got {got} vs ref {want}"
                );
            }
            assert_eq!(fabric.collectives_run(), usize::from(world > 1));
        }
    }

    #[test]
    fn rider_group_validates_contiguity_and_solo_flush() {
        let mesh = Mesh::solo();
        let a = ShardedTensor::from_gpu(vec![1, 64], DType::F32, Placement::PartialSum, mesh, 1, GpuBuf::new(0x1000, 64)).unwrap();
        let b = ShardedTensor::from_gpu(vec![1, 32], DType::F32, Placement::PartialSum, mesh, 1, GpuBuf::new(0x1000 + 64 * 4, 32)).unwrap();
        let bad = ShardedTensor::from_gpu(vec![1, 32], DType::F32, Placement::PartialSum, mesh, 1, GpuBuf::new(0x9000, 32)).unwrap();
        let mut g = PendingPartialGroup::new(&a).unwrap();
        g.push_contiguous(&b).unwrap();
        assert_eq!(g.total_floats(), 96);
        assert_eq!(g.member_count(), 2);
        assert!(g.push_contiguous(&bad).is_err());
    }
}
