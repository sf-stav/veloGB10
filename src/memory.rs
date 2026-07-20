use cudarc::driver::sys as cuda_sys;
use cudarc::driver::CudaDevice;
use half::f16;
use std::ptr::NonNull;
use std::sync::Arc;

use crate::model::{DevicePtr, ModelConfig};

pub struct UnifiedMemoryPool {
    pub dev: Arc<CudaDevice>,
    pool_base: cuda_sys::CUdeviceptr,
    pool_size: usize,
    next_offset: std::sync::atomic::AtomicUsize,
}

impl UnifiedMemoryPool {
    pub fn new(dev: Arc<CudaDevice>, size_gb: usize) -> anyhow::Result<Self> {
        let size = size_gb * 1024 * 1024 * 1024;
        let mut pool_base: cuda_sys::CUdeviceptr = 0;

        unsafe {
            let result = cuda_sys::cuMemAllocManaged(
                &mut pool_base,
                size,
                cuda_sys::CUmemAttach_flags::CU_MEM_ATTACH_GLOBAL as u32,
            );
            if result != cuda_sys::cudaError_enum::CUDA_SUCCESS {
                return Err(anyhow::anyhow!("Failed to allocate {} GB unified memory", size_gb));
            }
        }

        Ok(Self {
            dev,
            pool_base,
            pool_size: size,
            next_offset: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    pub fn size(&self) -> usize { self.pool_size }

    pub fn allocate<T>(&self, count: usize) -> anyhow::Result<*mut T> {
        let size = count * std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>().max(16);
        let current = self.next_offset.load(std::sync::atomic::Ordering::Relaxed);
        let aligned = (current + align - 1) & !(align - 1);
        let offset = self.next_offset.fetch_add(size, std::sync::atomic::Ordering::Relaxed);

        if offset + size > self.pool_size {
            return Err(anyhow::anyhow!("Out of unified memory"));
        }

        let ptr = (self.pool_base + (aligned as u64)) as *mut T;
        Ok(ptr)
    }

    pub fn device_ptr(&self, ptr: *mut std::ffi::c_void) -> u64 { ptr as u64 }

    pub fn prefetch_device(&self, ptr: *mut u8, len: usize) -> anyhow::Result<()> {
        if len == 0 { return Ok(()); }
        unsafe {
            let _ = cuda_sys::cuMemPrefetchAsync(
                ptr as u64,
                len,
                self.dev.ordinal() as cuda_sys::CUdevice,
                *self.dev.cu_stream(),
            );
        }
        Ok(())
    }

    pub fn memcpy_h2d(&self, dst: *mut u8, src: *const u8, len: usize) -> anyhow::Result<()> {
        unsafe { std::ptr::copy_nonoverlapping(src, dst, len); }
        self.prefetch_device(dst, len)?;
        Ok(())
    }

    pub fn allocate_kv_cache(
        &self,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> anyhow::Result<KVCachePtrs> {
        let bytes_per_kv_head = head_dim * std::mem::size_of::<f16>();
        let bytes_per_layer_k = num_kv_heads * max_seq_len * bytes_per_kv_head;
        let bytes_per_layer_v = num_kv_heads * max_seq_len * bytes_per_kv_head;
        let total = num_layers * (bytes_per_layer_k + bytes_per_layer_v);

        let base_k = self.allocate::<u8>(total)?;
        let base_v = unsafe { base_k.add(bytes_per_layer_k) };

        Ok(KVCachePtrs {
            base_k: DevicePtr(NonNull::new(base_k as *mut f16).unwrap()),
            base_v: DevicePtr(NonNull::new(base_v as *mut f16).unwrap()),
            num_layers,
            num_kv_heads,
            head_dim,
            max_seq_len,
            bytes_per_k: bytes_per_layer_k,
        })
    }
}

impl Drop for UnifiedMemoryPool {
    fn drop(&mut self) {
        unsafe {
            if self.pool_base != 0 {
                let _ = cuda_sys::cuMemFree_v2(self.pool_base);
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct KVCachePtrs {
    pub base_k: DevicePtr,
    pub base_v: DevicePtr,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub bytes_per_k: usize,
}

impl KVCachePtrs {
    #[inline]
    pub fn k_ptr(&self, layer: usize, pos: usize) -> *mut f16 {
        let layer_offset = layer * (self.bytes_per_k * 2);
        let token_offset = pos * self.num_kv_heads * self.head_dim;
        unsafe { self.base_k.0.as_ptr().add((layer_offset + token_offset) / std::mem::size_of::<f16>()) }
    }

    #[inline]
    pub fn v_ptr(&self, layer: usize, pos: usize) -> *mut f16 {
        let layer_offset = layer * (self.bytes_per_k * 2) + self.bytes_per_k;
        let token_offset = pos * self.num_kv_heads * self.head_dim;
        unsafe {
            self.base_v.0.as_ptr().add((layer_offset + token_offset) / std::mem::size_of::<f16>())
        }
    }

    pub fn stride_k_layer(&self) -> usize {
        self.num_kv_heads * self.max_seq_len * self.head_dim
    }
}

pub fn compute_memory_requirements(config: &ModelConfig) -> MemoryRequirements {
    let embedding_bytes = config.vocab_size * config.hidden_size * std::mem::size_of::<half::f16>();
    let lm_head_bytes = config.vocab_size * config.hidden_size * std::mem::size_of::<half::f16>();
    let layer_norm_bytes = 2 * config.hidden_size * std::mem::size_of::<half::f16>();
    let proj_bytes = 4 * config.hidden_size * config.hidden_size * std::mem::size_of::<half::f16>();
    let mlp_bytes = config.hidden_size * config.intermediate_size * 2 * std::mem::size_of::<half::f16>()
        + config.intermediate_size * config.hidden_size * std::mem::size_of::<half::f16>();
    let layer_bytes = proj_bytes + mlp_bytes + layer_norm_bytes;
    let weights_bytes = embedding_bytes + lm_head_bytes + config.num_layers * layer_bytes;
    let kv_bytes = config.num_layers * config.num_kv_heads * config.head_dim * config.max_seq_len * 2
        * std::mem::size_of::<half::f16>();
    let activation_bytes = config.hidden_size * std::mem::size_of::<half::f16>() * 4;
    MemoryRequirements {
        weights_mb: weights_bytes / 1024 / 1024,
        kv_cache_mb: kv_bytes / 1024 / 1024,
        activations_mb: activation_bytes / 1024 / 1024,
        total_mb: (weights_bytes + kv_bytes + activation_bytes) / 1024 / 1024,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MemoryRequirements {
    pub weights_mb: usize,
    pub kv_cache_mb: usize,
    pub activations_mb: usize,
    pub total_mb: usize,
}
