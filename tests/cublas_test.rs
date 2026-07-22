//! De-risk cuBLAS integration: gemv(trans=T) on a row-major weight must equal CPU matvec.
use cudarc::cublas::{CudaBlas, Gemv, GemvConfig};
use cudarc::cublas::sys::cublasOperation_t as OP;
use cudarc::driver::CudaDevice;
use std::sync::Arc;

#[test]
fn test_cublas_gemv_matches_cpu() {
    let dev = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(e) => { eprintln!("no cuda device: {:?}", e); return; }
    };
    let blas = CudaBlas::new(dev.clone()).expect("cublas handle");

    // Weight W[out,in] row-major: y[out] = W @ x[in]
    let out = 64usize;
    let inn = 48usize;
    let mut w_host = vec![0.0f32; out * inn];
    for i in 0..w_host.len() {
        w_host[i] = ((i as f32) * 0.1 - 1.7).sin();
    }
    let x_host: Vec<f32> = (0..inn).map(|i| (i as f32) * 0.03).collect();

    // CPU reference
    let mut y_cpu = vec![0.0f32; out];
    for r in 0..out {
        let mut s = 0.0f32;
        for c in 0..inn {
            s += w_host[r * inn + c] * x_host[c];
        }
        y_cpu[r] = s;
    }

    // GPU: weight memory (row-major[out,in]) is col-major[in,out] == W^T.
    // gemv(trans=T, m=in, n=out, lda=in) computes (W^T)^T @ x = W @ x.  x len in, y len out.
    let w_dev = dev.htod_sync_copy(&w_host).unwrap();
    let x_dev = dev.htod_sync_copy(&x_host).unwrap();
    let mut y_dev = dev.alloc_zeros::<f32>(out).unwrap();

    let cfg = GemvConfig::<f32> {
        trans: OP::CUBLAS_OP_T,
        m: inn as i32,
        n: out as i32,
        alpha: 1.0,
        lda: inn as i32,
        incx: 1,
        beta: 0.0,
        incy: 1,
    };
    unsafe { blas.gemv(cfg, &w_dev, &x_dev, &mut y_dev).expect("gemv"); }

    let y_gpu = dev.dtoh_sync_copy(&y_dev).unwrap();

    let max_diff = y_cpu.iter().zip(y_gpu.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("cublas gemv vs cpu max abs diff: {:.2e}", max_diff);
    assert!(max_diff < 1e-3, "gemv mismatch: {}", max_diff);

    let _: &Arc<CudaDevice> = &dev;
}
