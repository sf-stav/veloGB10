//! NVFP4 **W4A4 prefill** (kernels/gpu_w4a4.cu): the prefill GEMMs of the NVFP4 tensors in the
//! enabled groups run on Blackwell's block-scaled FP4 tensor cores with the ACTIVATIONS quantized to
//! E2M1 + UE4M3 per 16 (plus a per-tensor input global scale), reading the engine's standard tiled
//! weights directly — no second weight copy. Decode / verify (batch <= MAX_VERIFY) keep the W4A16
//! chain unless `GB10_W4A4_VERIFY` explicitly opts groups into the experimental A4 path. Although
//! decode and verify share the same GEMM, the full speculative chain is not batch-bit-invariant in
//! this mode; the binding lossless gate fails, so production must leave this variable unset.
//!
//! Env: `GB10_W4A4_PREFILL=1` (groups expert,mlp,attn) or `GB10_W4A4_PREFILL=expert,attn,...`
//! (any of expert mlp attn gdn gdn-in gdn-out hc ple lmhead). `gdn` is the backward-compatible
//! alias for both `gdn-in` (the four recurrent input projections) and `gdn-out` (out_proj).
//! `GB10_W4A4_VERIFY=1` (attn,mlp,gdn) or a group list enables experimental narrow W4A4.
//! `GB10_W4A4_N8=0` restores the wide 128-row GEMM for narrow-kernel A/B.
//! `GB10_W4A4_TRACE=1` logs each dispatch. Per-tensor `input_global_scale` (compressed-tensors) is
//! read from the artifact when present (`{stem}.input_global_scale` F32 [1]); absent = 1.0.
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice};
use cudarc::nvrtc::Ptx;

pub const W4_SMEM: u32 = 4 * 10752;
pub const W4_N8_SMEM: u32 = 4 * 4992;
pub const W4_BN: usize = 128;

pub struct W4a4State {
    pub quant: CudaFunction,     // w4a4_quant_pack_b
    pub gemm: CudaFunction,      // w4a4_gemm_b
    pub gemm_n8: CudaFunction,   // w4a4_gemm_n8_b (1..16 rows, 8-row grid groups)
    pub gemm_moe: CudaFunction,  // w4a4_gemm_moe_b
    pub tilemap: CudaFunction,   // w4a4_moe_tilemap_b
    pub fakequant: CudaFunction, // w4a4_fakequant_b (GB10_W4A4_CHECK only)
    /// Packed-activation scratch: `rows_max` rows at `k_max` (Bp = rows*K/2 B, SFB = rows*K/4 B).
    pub bq: CudaSlice<u8>,
    pub sb: CudaSlice<u8>,
    pub rows_max: usize,
    pub k_max: usize,
    /// MoE 128-row tile map (device-built): [0] = count, [1..1+tiles_max) expert id, then first row.
    pub tmap: CudaSlice<i32>,
    pub tiles_max: usize,
    /// NVFP4 weights (by qweight device pointer) whose prefill GEMM takes the W4A4 path.
    pub enabled: HashSet<u64>,
    /// Explicit opt-in for narrow (decode/verify) W4A4. Kept separate from `enabled` because
    /// ordinary groups preserve the W4A16 batch-invariant path at N <= MAX_VERIFY.
    /// `GB10_W4A4_VERIFY` admits its selected groups here; lm_head also has its historical
    /// controlled opt-in. This is experimental: the end-to-end lossless gate shows that the full
    /// speculative chain is not batch-bit-invariant even though the GEMM selection is shared.
    pub narrow_enabled: HashSet<u64>,
    /// Input global scale per enabled weight (absent = 1.0).
    pub x_gs: HashMap<u64, f32>,
    pub groups: Vec<String>,
    pub trace: bool,
}

/// GB10_W4A4_CHECK: every W4A4 GEMM is recomputed through the bf16 chain and compared (slow, debug).
pub fn check_on() -> bool {
    static C: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("GB10_W4A4_CHECK").is_ok());
    *C
}

/// Narrow 8-row MMA kernel. Enabled by default; `GB10_W4A4_N8=0` restores the wide 128-row
/// implementation for bitwise/performance A/B and as a production rollback.
pub fn n8_on() -> bool {
    static N8: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("GB10_W4A4_N8").map(|v| {
            let v = v.trim(); v != "0" && !v.eq_ignore_ascii_case("off")
        }).unwrap_or(true)
    });
    *N8
}

/// The env request: None = off; Some(groups) = on for these quantizer groups.
fn groups_from_var(name: &str, defaults: &[&str]) -> Option<Vec<String>> {
    let v = std::env::var(name).ok()?;
    let v = v.trim();
    if v.is_empty() || v == "0" { return None; }
    if v == "1" || v.eq_ignore_ascii_case("on") {
        return Some(defaults.iter().map(|s| (*s).to_string()).collect());
    }
    Some(v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect())
}

pub fn groups_from_env() -> Option<Vec<String>> {
    groups_from_var("GB10_W4A4_PREFILL", &["expert", "mlp", "attn"])
}

pub fn verify_groups_from_env() -> Option<Vec<String>> {
    groups_from_var("GB10_W4A4_VERIFY", &["attn", "mlp", "gdn"])
}

#[inline]
pub fn group_on(groups: &[String], name: &str) -> bool {
    groups.iter().any(|g| g == name)
}

/// `gdn` retains the old all-projections behavior while the two granular names let serving keep
/// the recurrent input or output side in A16 independently. This changes only activation dispatch;
/// every selected projection continues to use the same MR-GPTQ NVFP4 weight artifact.
#[inline]
pub fn gdn_part_on(groups: &[String], part: &str) -> bool {
    group_on(groups, "gdn") || group_on(groups, part)
}

impl W4a4State {
    pub fn build(dev: &Arc<CudaDevice>, groups: Vec<String>, enabled: HashSet<u64>, narrow_enabled: HashSet<u64>, x_gs: HashMap<u64, f32>,
                 rows_max: usize, k_max: usize, tiles_max: usize) -> anyhow::Result<Self> {
        let ptx = Ptx::from_src(std::fs::read_to_string("src/ptx/gpu_w4a4.ptx")?);
        dev.load_ptx(ptx, "gpu_w4a4", &["w4a4_quant_pack_b", "w4a4_gemm_b", "w4a4_gemm_n8_b", "w4a4_gemm_moe_b",
                                        "w4a4_moe_tilemap_b", "w4a4_fakequant_b", "kernel_build_id"])?;
        crate::gpu::GpuModel::assert_kernel_build_id(dev, "gpu_w4a4")?;
        let get = |n: &str| dev.get_func("gpu_w4a4", n).unwrap_or_else(|| panic!("{n} not in gpu_w4a4"));
        let rows_max = rows_max.div_ceil(8) * 8;
        let k_max = k_max.max(64);
        // Poisoned (0xFF) rather than zeroed: a GEMM read of never-packed rows changes the output
        // deterministically instead of silently reading stale bytes (same rule as mxfp4's scratch).
        let bq = dev.htod_sync_copy(&vec![0xFFu8; rows_max * (k_max / 2)])?;
        let sb = dev.htod_sync_copy(&vec![0xFFu8; rows_max * (k_max / 4)])?;
        let tmap = dev.alloc_zeros::<i32>(1 + 2 * tiles_max.max(1))?;
        let trace = std::env::var("GB10_W4A4_TRACE").is_ok();
        println!("W4A4 runtime ON: groups {:?} — {} wide + {} narrow NVFP4 weights ({} with an input_global_scale); scratch {} rows x K {} ({:.0} MB)",
                 groups, enabled.len(), narrow_enabled.len(), x_gs.len(), rows_max, k_max,
                 (rows_max * k_max * 3 / 4) as f64 / 1e6);
        Ok(Self { quant: get("w4a4_quant_pack_b"), gemm: get("w4a4_gemm_b"), gemm_n8: get("w4a4_gemm_n8_b"), gemm_moe: get("w4a4_gemm_moe_b"),
                  tilemap: get("w4a4_moe_tilemap_b"), fakequant: get("w4a4_fakequant_b"), bq, sb, rows_max, k_max, tmap, tiles_max,
                  enabled, narrow_enabled, x_gs, groups, trace })
    }
    #[inline] pub fn on(&self, qweight_ptr: u64) -> bool { self.enabled.contains(&qweight_ptr) }
    #[inline] pub fn narrow_on(&self, qweight_ptr: u64) -> bool { self.narrow_enabled.contains(&qweight_ptr) }
    #[inline] pub fn xgs(&self, qweight_ptr: u64) -> f32 { self.x_gs.get(&qweight_ptr).copied().unwrap_or(1.0) }
    /// Dense launch grid (token-fastest raster, group width 8): ceil(tn/8)*8 * tm blocks.
    pub fn dense_grid(mf: usize, nt: usize) -> u32 {
        let tm = mf.div_ceil(128); let tn = nt.div_ceil(128);
        (tn.div_ceil(8) * 8 * tm) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::{gdn_part_on, group_on};

    fn groups(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| (*name).to_string()).collect()
    }

    #[test]
    fn gdn_alias_selects_both_parts() {
        let selected = groups(&["attn", "gdn"]);
        assert!(gdn_part_on(&selected, "gdn-in"));
        assert!(gdn_part_on(&selected, "gdn-out"));
    }

    #[test]
    fn granular_gdn_groups_are_independent() {
        let input_only = groups(&["mlp", "gdn-in"]);
        assert!(gdn_part_on(&input_only, "gdn-in"));
        assert!(!gdn_part_on(&input_only, "gdn-out"));
        assert!(group_on(&input_only, "mlp"));

        let output_only = groups(&["gdn-out"]);
        assert!(!gdn_part_on(&output_only, "gdn-in"));
        assert!(gdn_part_on(&output_only, "gdn-out"));
    }
}
