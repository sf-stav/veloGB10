//! DSV4.1 (leg I0): the Rust mirror of kernels/gpu_dsv41/00_abi.cuh, the host decoders of the
//! value codes (E4M3 / E8M0 / E2M1, the CPU oracles use these), and `--probe-dsv41-abi`, which
//! launches the ABI probe kernel and checks the device agrees with this file bit for bit.

use anyhow::{ensure, Context, Result};
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};

/// The ABI probe's entry kernels.
pub const FNS: &[&str] = &["d41_kernel_build_id", "d41_abi_probe"];

pub const ABI_VERSION: u32 = 1;

/// Weight format tags (00_abi.cuh `D41_FMT_*`).
pub mod fmt {
    pub const BF16: u32 = 1;
    pub const F16: u32 = 2;
    pub const F32: u32 = 3;
    pub const FP8_B32X32_UE8M0: u32 = 10;
    pub const FP8_B1X32_UE8M0: u32 = 11;
    pub const MXFP4: u32 = 20;
    pub const NVFP4: u32 = 21;
    pub const EXL3_BASE: u32 = 32;
    pub const fn exl3(k: u32) -> u32 { EXL3_BASE + k }
}

/// The tag of a census format (None = no kernel format: a parameter or an unknown format).
pub fn tag_of(f: &crate::dsv41::census::LinearFormat) -> Option<u32> {
    use crate::dsv41::census::LinearFormat as L;
    Some(match f {
        L::Plain(d) if d == "BF16" => fmt::BF16,
        L::Plain(d) if d == "F16" => fmt::F16,
        L::Plain(d) if d == "F32" => fmt::F32,
        L::Fp8Block { bo: 32, bi: 32, scale } if scale == "F8_E8M0" => fmt::FP8_B32X32_UE8M0,
        L::Fp8Block { bo: 1, bi: 32, scale } if scale == "F8_E8M0" => fmt::FP8_B1X32_UE8M0,
        L::Fp4 { block: 32, scale, scale2: false } if scale == "F8_E8M0" => fmt::MXFP4,
        L::Fp4 { block: 16, scale, scale2: true } if scale == "F8_E4M3" => fmt::NVFP4,
        L::Exl3 { k, .. } => fmt::exl3(*k as u32),
        _ => return None,
    })
}

/// Struct sizes of the cache layouts (00_abi.cuh), checked on the device by the probe.
pub const KV_MAIN_ROW: u32 = 288;
pub const INDEX_K_Q: u32 = 64;
pub const INDEX_K_S: u32 = 4;
pub const SWA_Q: u32 = 512;
pub const SWA_S: u32 = 16;
pub const ENGRAM_ROW: u32 = 264;
pub const SMEM_OPTIN_MAX: u32 = 101_376;
/// SWA ring slots per layer slot (window 128 + backtracking 6) and the backtracking rows.
pub const SWA_SLOTS: u32 = 134;
pub const BACKTRACK: u32 = 6;

/// E4M3 (fn) decode, host side: identical to `d41_e4m3`.
pub fn e4m3(c: u8) -> f32 {
    let (s, e, m) = (c >> 7, (c >> 3) & 15, c & 7);
    let v = if e == 15 && m == 7 { f32::NAN } else if e == 0 { m as f32 * 2f32.powi(-9) } else { 2f32.powi(e as i32 - 7) * (1.0 + m as f32 / 8.0) };
    if s == 1 { -v } else { v }
}
/// E8M0 decode: 2^(e-127), 255 = NaN.
pub fn e8m0(e: u8) -> f32 { if e == 255 { f32::NAN } else { 2f64.powi(e as i32 - 127) as f32 } }
/// E2M1 decode of one nibble (convert.py FP4_TABLE).
pub fn e2m1(n: u8) -> f32 {
    const T: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];
    T[(n & 15) as usize]
}

fn same(a: f32, b: f32) -> bool { (a.is_nan() && b.is_nan()) || a == b }

/// Launch the probe on device 0 and compare everything. Returns the report lines (the last one
/// is `DSV41_ABI_OK` only when every check holds; any mismatch is an error naming it).
pub fn probe() -> Result<Vec<String>> {
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    super::load_module(&dev)?;
    let mut lines = vec![format!("gpu_dsv41 loaded: {} entry kernels, kernel build {} (handshake ok)", super::all_fns().len(), env!("KERNEL_BUILD_ID"))];
    let f = dev.get_func(super::MODULE, "d41_abi_probe").context("d41_abi_probe missing")?;
    let mut out = dev.alloc_zeros::<u32>(18)?;
    let mut dec = dev.alloc_zeros::<f32>(256 + 256 + 16)?;
    unsafe { f.launch(LaunchConfig { grid_dim: (1, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }, (&mut out, &mut dec)) }?;
    let out = dev.dtoh_sync_copy(&out)?;
    let dec = dev.dtoh_sync_copy(&dec)?;
    let want: [(&str, u32); 18] = [("ABI version", ABI_VERSION), ("kv_main_row", KV_MAIN_ROW), ("index_k_q", INDEX_K_Q), ("index_k_s", INDEX_K_S),
        ("swa_q", SWA_Q), ("swa_s", SWA_S), ("engram_row", ENGRAM_ROW), ("smem opt-in max", SMEM_OPTIN_MAX), ("fmt BF16", fmt::BF16),
        ("fmt FP8 32x32", fmt::FP8_B32X32_UE8M0), ("fmt FP8 1x32", fmt::FP8_B1X32_UE8M0), ("fmt MXFP4", fmt::MXFP4), ("fmt NVFP4", fmt::NVFP4),
        ("fmt EXL3 K3", fmt::exl3(3)), ("GEMV max rows", crate::dsv41::weights::GEMV_MAX_ROWS as u32), ("coop max rows", crate::dsv41::weights::COOP_MAX_ROWS as u32),
        ("SWA ring slots", SWA_SLOTS), ("backtracking rows", BACKTRACK)];
    for (i, (what, w)) in want.iter().enumerate() {
        ensure!(out[i] == *w, "ABI mismatch: {what} device {} != host {w}", out[i]);
    }
    lines.push(format!("ABI v{} constants and struct sizes agree (kv_main_row {} B, index_k {}+{} B, swa {}+{} B x {} slots, engram_row {} B)",
                       ABI_VERSION, KV_MAIN_ROW, INDEX_K_Q, INDEX_K_S, SWA_Q, SWA_S, SWA_SLOTS, ENGRAM_ROW));
    // the host mirrors must agree with the state module's arithmetic too
    ensure!(crate::dsv41::state::main_kv_row_bytes(512) as u32 == KV_MAIN_ROW && crate::dsv41::state::swa_row_bytes(512) as u32 == SWA_Q + SWA_S
            && crate::dsv41::state::index_k_row_bytes(128) as u32 == INDEX_K_Q + INDEX_K_S, "state.rs row bytes disagree with the ABI");
    for c in 0..=255u8 {
        ensure!(same(dec[c as usize], e4m3(c)), "E4M3 code {c:#04x}: device {} host {}", dec[c as usize], e4m3(c));
        ensure!(same(dec[256 + c as usize], e8m0(c)), "E8M0 code {c}: device {} host {}", dec[256 + c as usize], e8m0(c));
    }
    for n in 0..16u8 { ensure!(same(dec[512 + n as usize], e2m1(n)), "E2M1 nibble {n}: device {} host {}", dec[512 + n as usize], e2m1(n)); }
    lines.push("value codes agree: 256 E4M3, 256 E8M0, 16 E2M1 (device decode == host decode)".into());
    lines.push("DSV41_ABI_OK".into());
    Ok(lines)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dsv41_host_value_codes() {
        assert_eq!(e4m3(0x7E), 448.0);
        assert_eq!(e4m3(0x01), 2f32.powi(-9));
        assert_eq!(e4m3(0x38), 1.0);
        assert!(e4m3(0x7F).is_nan() && e4m3(0xFF).is_nan());
        assert_eq!(e4m3(0xB8), -1.0);
        assert_eq!(e8m0(127), 1.0);
        assert_eq!(e8m0(0), 2f64.powi(-127) as f32);
        assert!(e8m0(255).is_nan());
        assert_eq!(e2m1(7), 6.0);
        assert_eq!(e2m1(0xF), -6.0);
        assert_eq!(tag_of(&crate::dsv41::census::LinearFormat::Exl3 { k: 2, mul1: true, suh: true, svh: true }), Some(34));
    }

    /// The ABI kernel launches and agrees with the host. GPU: explicit only, under the box's GPU lock
    /// (`flock /tmp/dsv41_gpu.lock cargo test --release dsv41_abi_probe -- --ignored`); the same
    /// check is `--probe-dsv41-abi`.
    #[test]
    #[ignore = "GPU: run explicitly with --ignored under flock /tmp/dsv41_gpu.lock"]
    fn dsv41_abi_probe_on_the_gpu() {
        let lines = probe().unwrap();
        assert_eq!(lines.last().map(String::as_str), Some("DSV41_ABI_OK"), "{lines:?}");
    }
}
