use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;

#[derive(Debug, Clone)]
pub struct KernelModule {
    pub dev: Arc<CudaDevice>,
    pub kernels: HashMap<String, CudaFunction>,
}

impl KernelModule {
    pub async fn load_from_ptx(dev: Arc<CudaDevice>) -> anyhow::Result<Self> {
        let ptx_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
        let ptx_files = vec![
            ("gemm_nvfp4", "src/ptx/gemm_nvfp4.ptx", vec!["gemm_nvfp4_sm121"]),
            ("fused_decode", "src/ptx/fused_decode.ptx", vec!["fused_decode_single_token"]),
            ("rms_norm", "src/ptx/rms_norm.ptx", vec!["rms_norm_kernel", "silu_gate_mul_kernel", "add_residual_kernel", "apply_rope_kernel"]),
            ("silu_gate", "src/ptx/silu_gate.ptx", vec!["silu_gate_mul_kernel", "add_residual_kernel"]),
        ];

        let mut module = Self {
            dev: dev.clone(),
            kernels: HashMap::new(),
        };

        for (module_name, ptx_path, func_names) in ptx_files {
            let full_path = format!("{}/{}", ptx_dir, ptx_path);
            if !std::path::Path::new(&full_path).exists() {
                eprintln!("WARNING: PTX file not found: {}", full_path);
                continue;
            }

            let ptx_src = std::fs::read_to_string(&full_path)?;
            let ptx = Ptx::from_src(ptx_src);

            match dev.load_ptx(ptx, module_name, &func_names) {
                Ok(_) => {
                    eprintln!("Loaded PTX module: {} from {}", module_name, full_path);
                    for func_name in &func_names {
                        if let Some(func) = dev.get_func(module_name, func_name) {
                            module.kernels.insert(func_name.to_string(), func);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("WARNING: Failed to load PTX module {}: {}", module_name, e);
                }
            }
        }

        Ok(module)
    }

    pub fn get_func(&self, name: &str) -> Option<CudaFunction> {
        self.kernels.get(name).cloned()
    }

    pub fn len(&self) -> usize {
        self.kernels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kernels.is_empty()
    }
}

pub unsafe fn launch_gemm_nvfp4(
    module: &KernelModule,
    out: *mut half::f16,
    a: *const half::f16,
    b: &crate::model::PackedLinear,
    m: usize,
    n: usize,
    k: usize,
) -> anyhow::Result<()> {
    let func = module
        .get_func("gemm_nvfp4_sm121")
        .ok_or_else(|| anyhow::anyhow!("gemm_nvfp4 kernel not loaded"))?;

    let block_size = 256;
    let grid_x = (m + 63) / 64;
    let grid_y = (n + 63) / 64;

    let config = LaunchConfig {
        grid_dim: (grid_x as u32, grid_y as u32, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let params = (
        a as u64,
        b.weights.as_ptr() as u64,
        b.scales.as_ptr() as u64,
        out as u64,
        m as i32,
        n as i32,
        k as i32,
    );

    let stream = module.dev.fork_default_stream()?;
    unsafe { func.launch(config, params)?; }
    stream.wait_for_default()?;
    Ok(())
}

pub unsafe fn launch_rms_norm(
    module: &KernelModule,
    out: *mut half::f16,
    input: *const half::f16,
    weight: *const half::f16,
    n: usize,
    eps: f32,
) -> anyhow::Result<()> {
    let func = module
        .get_func("rms_norm_kernel")
        .ok_or_else(|| anyhow::anyhow!("rms_norm kernel not loaded"))?;

    let block_size = 256;
    let grid_size = (n + block_size - 1) / block_size;
    let shared_mem_bytes = (block_size * std::mem::size_of::<f32>()) as u32;

    let config = LaunchConfig {
        grid_dim: (grid_size as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes,
    };

    let params = (input as u64, out as u64, weight as u64, n as i32, eps);

    let stream = module.dev.fork_default_stream()?;
    unsafe { func.launch(config, params)?; }
    stream.wait_for_default()?;
    Ok(())
}

pub unsafe fn launch_silu_gate_mul(
    module: &KernelModule,
    out: *mut half::f16,
    gate: *const half::f16,
    up: *const half::f16,
    n: usize,
) -> anyhow::Result<()> {
    let func = module
        .get_func("silu_gate_mul_kernel")
        .ok_or_else(|| anyhow::anyhow!("silu_gate_mul kernel not loaded"))?;

    let block_size = 256;
    let grid_size = (n + block_size - 1) / block_size;

    let config = LaunchConfig {
        grid_dim: (grid_size as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let params = (gate as u64, up as u64, out as u64, n as i32);

    let stream = module.dev.fork_default_stream()?;
    unsafe { func.launch(config, params)?; }
    stream.wait_for_default()?;
    Ok(())
}

pub unsafe fn launch_add_residual(
    module: &KernelModule,
    out: *mut half::f16,
    a: *const half::f16,
    b: *const half::f16,
    n: usize,
) -> anyhow::Result<()> {
    let func = module
        .get_func("add_residual_kernel")
        .ok_or_else(|| anyhow::anyhow!("add_residual kernel not loaded"))?;

    let block_size = 256;
    let grid_size = (n + block_size - 1) / block_size;

    let config = LaunchConfig {
        grid_dim: (grid_size as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let params = (a as u64, b as u64, out as u64, n as i32);

    let stream = module.dev.fork_default_stream()?;
    unsafe { func.launch(config, params)?; }
    stream.wait_for_default()?;
    Ok(())
}

pub unsafe fn launch_apply_rope(
    module: &KernelModule,
    out: *mut half::f16,
    input: *const half::f16,
    pos: usize,
    head_dim: usize,
    num_heads: usize,
) -> anyhow::Result<()> {
    let func = module
        .get_func("apply_rope_kernel")
        .ok_or_else(|| anyhow::anyhow!("apply_rope kernel not loaded"))?;

    let total_elements = num_heads * head_dim;
    let block_size = 256;
    let grid_size = (total_elements + block_size - 1) / block_size;

    let config = LaunchConfig {
        grid_dim: (grid_size as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let params = (input as u64, out as u64, pos as i32, num_heads as i32, head_dim as i32, 10000.0f32);

    let stream = module.dev.fork_default_stream()?;
    unsafe { func.launch(config, params)?; }
    stream.wait_for_default()?;
    Ok(())
}

pub unsafe fn launch_fused_attention_decode(
    module: &KernelModule,
    output: *mut half::f16,
    q: *const half::f16,
    k_cache: *const half::f16,
    v_cache: *const half::f16,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    num_kv_heads: usize,
    layer: usize,
) -> anyhow::Result<()> {
    let func = module
        .get_func("fused_decode_single_token")
        .ok_or_else(|| anyhow::anyhow!("fused_decode kernel not loaded"))?;

    let block_size = 128;
    let grid_size = num_heads;

    let config = LaunchConfig {
        grid_dim: (grid_size as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let params = (q as u64, k_cache as u64, v_cache as u64, output as u64, seq_len as i32, num_heads as i32, num_kv_heads as i32, head_dim as i32, layer as i32);

    let stream = module.dev.fork_default_stream()?;
    unsafe { func.launch(config, params)?; }
    stream.wait_for_default()?;
    Ok(())
}
