//! Device mesh, tensor placement algebra (`Replicated` / `Shard { axis }` / `PartialSum`),
//! data types, and quantum-aligned shard slicing (e.g. 128-element EXL3 trellis / FP4 block alignment).

use anyhow::{bail, ensure, Result};

/// Logical 1-D tensor-parallel device mesh.
/// Supports `world = 1` (solo, zero-cost no-op) and power-of-two worlds `2, 4, 8, 16`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Mesh {
    pub rank: usize,
    pub world: usize,
}

impl Mesh {
    pub fn new(rank: usize, world: usize) -> Result<Self> {
        ensure!(matches!(world, 1 | 2 | 3 | 4 | 6 | 8 | 16), "dist::Mesh: world {world} must be in {{1, 2, 3, 4, 6, 8, 16}}");
        ensure!(rank < world, "dist::Mesh: rank {rank} out of bounds for world {world}");
        Ok(Self { rank, world })
    }

    pub const fn solo() -> Self {
        Self { rank: 0, world: 1 }
    }

    pub const fn is_solo(self) -> bool {
        self.world == 1
    }

    pub const fn is_head(self) -> bool {
        self.rank == 0
    }

    /// Recursive-doubling round count (`log2(world)`, 0 for solo).
    pub const fn rounds(self) -> usize {
        self.world.trailing_zeros() as usize
    }

    /// For world=8, returns the 2-stage `2x4` hierarchical plan (`tp_xport::hier8_plan`).
    pub fn hier8_plan(self) -> Result<crate::tp_xport::Hier8Plan> {
        ensure!(self.world == 8, "hier8_plan requires world == 8 (got {})", self.world);
        crate::tp_xport::hier8_plan(self.rank)
    }
}

/// How a tensor's values are distributed across a [`Mesh`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Placement {
    /// Every rank holds the exact same full tensor.
    Replicated,
    /// Partitioned along `axis` across the mesh; rank `r` owns slice `aligned_shard_range(...)`.
    Shard { axis: usize },
    /// Every rank holds an additive partial term of the full tensor; summing across all ranks
    /// yields the canonical `Replicated` tensor. Reduction is **lazy**: linear operations
    /// (`PartialSum + PartialSum`, `PartialSum + Replicated` via rank-0 bias/residual) stay
    /// `PartialSum` with 0 collectives until a non-linear consumer materializes them.
    PartialSum,
}

impl Placement {
    pub const fn is_replicated(self) -> bool {
        matches!(self, Placement::Replicated)
    }
    pub const fn is_partial(self) -> bool {
        matches!(self, Placement::PartialSum)
    }
    pub const fn shard_axis(self) -> Option<usize> {
        match self {
            Placement::Shard { axis } => Some(axis),
            _ => None,
        }
    }
}

/// Element storage type for [`crate::dist::ShardedTensor`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    BF16,
    F16,
    I32,
    U32,
    U8,
}

impl DType {
    pub const fn elem_bytes(self) -> usize {
        match self {
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::BF16 | DType::F16 => 2,
            DType::U8 => 1,
        }
    }
}

/// Element slice `[start, end)` owned by one rank along a sharded axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShardRange {
    pub start: usize,
    pub end: usize,
}

impl ShardRange {
    pub const fn len(self) -> usize {
        self.end - self.start
    }
    pub const fn is_empty(self) -> bool {
        self.start == self.end
    }
}

/// Compute rank `rank`'s contiguous element slice `[start, end)` of `global_dim` across `world` ranks,
/// aligned to whole `align`-element blocks (e.g. `align = 128` for EXL3 trellis tiles or FP4 blocks;
/// `align = 1` for unaligned tensors).
///
/// When `global_dim / align` does not divide evenly by `world`, the remainder blocks are distributed
/// evenly to lower ranks (`+align` each for `rank < rem`), matching DSV4.1's `5+4+5+4`-style block
/// balance while guaranteeing:
/// - every rank's `start` and `len()` are exact multiples of `align`;
/// - all ranks' ranges tile `[0, global_dim)` disjointly in ascending order.
pub fn aligned_shard_range(global_dim: usize, rank: usize, world: usize, align: usize) -> Result<ShardRange> {
    ensure!(world >= 1 && rank < world, "aligned_shard_range: invalid rank {rank} for world {world}");
    let q = align.max(1);
    if world == 1 {
        return Ok(ShardRange { start: 0, end: global_dim });
    }
    ensure!(
        global_dim % q == 0,
        "aligned_shard_range: global_dim {global_dim} is not a multiple of alignment quantum {q}"
    );
    let blocks = global_dim / q;
    ensure!(
        blocks >= world,
        "aligned_shard_range: {blocks} blocks of {q} cannot be sharded across {world} ranks without empty shards"
    );
    let base = blocks / world;
    let rem = blocks % world;
    let start_blocks = rank * base + rank.min(rem);
    let my_blocks = base + usize::from(rank < rem);
    Ok(ShardRange {
        start: start_blocks * q,
        end: (start_blocks + my_blocks) * q,
    })
}

/// Compute the local tensor shape for `global_shape` under `(mesh, placement, align)`.
pub fn local_shape_for(
    global_shape: &[usize],
    mesh: Mesh,
    placement: Placement,
    align: usize,
) -> Result<Vec<usize>> {
    match placement {
        Placement::Replicated | Placement::PartialSum => Ok(global_shape.to_vec()),
        Placement::Shard { axis } => {
            if axis >= global_shape.len() {
                bail!("Shard axis {axis} out of bounds for shape {global_shape:?}");
            }
            let r = aligned_shard_range(global_shape[axis], mesh.rank, mesh.world, align)?;
            let mut out = global_shape.to_vec();
            out[axis] = r.len();
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_shard_range_tiles_exact_and_uneven_dims() {
        // Evenly divisible (hidden=5120, align=128, world=4 -> 10 blocks = 1280 each)
        for world in [1usize, 2, 4, 8] {
            let mut cursor = 0;
            for r in 0..world {
                let sr = aligned_shard_range(5120, r, world, 128).unwrap();
                assert_eq!(sr.start, cursor);
                assert_eq!(sr.len() % 128, 0);
                cursor = sr.end;
            }
            assert_eq!(cursor, 5120);
        }
        // Uneven block count (vocab=129280 = 1010 * 128, world=4 -> 253, 253, 252, 252 blocks)
        let lens: Vec<usize> = (0..4)
            .map(|r| aligned_shard_range(129_280, r, 4, 128).unwrap().len())
            .collect();
        assert_eq!(lens, vec![253 * 128, 253 * 128, 252 * 128, 252 * 128]);
        assert_eq!(lens.iter().sum::<usize>(), 129_280);
        // Misaligned global_dim rejected loudly
        assert!(aligned_shard_range(5121, 0, 4, 128).is_err());
    }

    #[test]
    fn mesh_validates_power_of_two_and_bounds() {
        assert!(Mesh::solo().is_solo());
        assert_eq!(Mesh::new(0, 1).unwrap().rounds(), 0);
        assert!(Mesh::new(2, 3).is_ok());
        assert_eq!(Mesh::new(3, 4).unwrap().rounds(), 2);
        assert!(Mesh::new(5, 6).is_ok());
        assert_eq!(Mesh::new(7, 8).unwrap().rounds(), 3);
        assert!(Mesh::new(0, 5).is_err());
        assert!(Mesh::new(3, 3).is_err());
        assert!(Mesh::new(4, 4).is_err());
    }
}
