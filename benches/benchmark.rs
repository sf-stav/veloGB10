use criterion::{black_box, criterion_group, criterion_main, Criterion};
use gb10_inference::engine::*;

fn bench_model_config(_c: &mut Criterion) {
    c.bench_function("model_config", |b| {
        b.iter(|| {
            let config = ModelConfig::qwen3_5_0_8b();
            black_box(config.total_params())
        })
    });
}

fn bench_rms_norm_cpu(_c: &mut Criterion) {
    let n = 1024;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) / 100.0).collect();
    let weight: Vec<f32> = vec![1.0; n];

    c.bench_function("rms_norm_cpu", |b| {
        b.iter(|| {
            let mut sum_sq = 0.0f32;
            for val in &input {
                sum_sq += val * val;
            }
            let _ = (sum_sq / n as f32 + 1e-6).sqrt();
        })
    });
}

fn bench_lm_head_matmul(_c: &mut Criterion) {
    let hidden_size = 1024;
    let vocab_size = 1000; // smaller for benchmark
    let input: Vec<f32> = (0..hidden_size).map(|i| (i as f32) / 100.0).collect();
    let weights: Vec<f32> = (0..vocab_size * hidden_size).map(|i| (i as f32 % 100) / 100.0).collect();

    c.bench_function("lm_head_matmul", |b| {
        b.iter(|| {
            let mut logits = vec![0.0f32; vocab_size];
            unsafe {
                let lm_head = weights.as_ptr();
                for v in 0..vocab_size {
                    let mut sum: f32 = 0.0;
                    for h in 0..hidden_size {
                        sum += input[h] * *lm_head.add(v * hidden_size + h);
                    }
                    logits[v] = sum;
                }
            }
            black_box(logits)
        })
    });
}

criterion_group!(benches, bench_model_config, bench_rms_norm_cpu, bench_lm_head_matmul);
criterion_main!(benches);
