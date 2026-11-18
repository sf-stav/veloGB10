use rand::Rng;

/// Token sampler supporting greedy and top-p sampling
pub struct Sampler {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
}

impl Default for Sampler {
    fn default() -> Self {
        Self {
            temperature: 0.0, // Greedy by default
            top_p: 0.9,
            top_k: 50,
        }
    }
}

impl Sampler {
    pub fn new(temperature: f32, top_p: f32, top_k: usize) -> Self {
        Self {
            temperature,
            top_p: if top_p <= 0.0 { 1.0 } else { top_p },
            top_k: if top_k == 0 { usize::MAX } else { top_k },
        }
    }

    pub fn sample(&self, logits: &[f32]) -> u32 {
        sample(logits, self.temperature, self.top_k, self.top_p)
    }
}

/// The batched GPU samplers of the NVFP4/bf16 family (`sample_b`, `sample_f32_b`, `sample_prob_b`
/// and `spec_verify_b` in kernels/gpu_batch.cu) rank at most this many candidates per row (their
/// `SAMPLE_K_MAX`) and clamp a `top_k` below 1 to 1, i.e. to the argmax.
pub const GPU_SAMPLE_K_MAX: usize = 64;

/// `top_k` as the batched GPU samplers must receive it (TOPKM1, the PR #29 follow-up). A request's
/// 0 — "no top-k truncation", sent as 0 or as a negative value the server maps to 0 — and any value
/// above the kernels' candidate limit both become that limit, the widest law the kernels can draw
/// from. Before, 0 reached the kernels' own `topk < 1 -> 1` clamp and "no truncation" was served
/// as greedy. The CPU sampler (`sample`) takes the request's value unchanged: it handles 0 and
/// k >= vocabulary exactly. Callers count and say once per boot when the cap applies
/// (`velogb10_top_k_capped_total`).
pub fn gpu_top_k(top_k: usize) -> i32 {
    if top_k == 0 || top_k > GPU_SAMPLE_K_MAX { GPU_SAMPLE_K_MAX as i32 } else { top_k as i32 }
}

/// Whether `gpu_top_k` changes the request's value (the cap applies): a sampled request (the
/// caller excludes greedy and top_k 1, which never reach the sampler) asking for 0 or more than
/// the kernels' limit.
pub fn gpu_top_k_capped(top_k: usize) -> bool {
    top_k == 0 || top_k > GPU_SAMPLE_K_MAX
}

/// Greedy argmax.
pub fn argmax(logits: &[f32]) -> u32 {
    logits.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0)
}

/// Full sampling: temperature → top-k → top-p (nucleus) → multinomial.
/// temperature=0 → greedy argmax.
pub fn sample(logits: &[f32], temperature: f32, top_k: usize, top_p: f32) -> u32 {
    let v = logits.len();
    if v == 0 { return 0; }
    if temperature < 1e-6 || top_k == 1 {
        return argmax(logits);
    }

    let inv_temp = 1.0 / temperature;

    // Top-k: find indices of the k largest logits (O(n) via partition). 0 means no truncation.
    let k = if top_k == 0 { v } else { top_k.min(v) };
    let mut indices: Vec<usize> = (0..v).collect();
    if k < v {
        indices.select_nth_unstable_by(k, |&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    }
    // indices[0..k] are the top-k token indices

    // Temperature-scaled softmax over top-k
    let max_logit = indices[..k].iter().map(|&i| logits[i] * inv_temp)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f32)> = indices[..k].iter().map(|&i| {
        (i, (logits[i] * inv_temp - max_logit).exp())
    }).collect();
    let sum: f32 = probs.iter().map(|(_, p)| *p).sum();
    if sum <= 0.0 { return probs[0].0 as u32; }
    for (_, p) in probs.iter_mut() { *p /= sum; }

    // Top-p (nucleus): sort by prob desc, keep smallest set with cumsum >= top_p
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut cumsum = 0.0f32;
    let cutoff = probs.iter().position(|(_, p)| {
        cumsum += p;
        cumsum >= top_p
    }).unwrap_or(probs.len() - 1);
    let nucleus = &probs[..=cutoff];

    // Renormalize and sample
    let nuc_sum: f32 = nucleus.iter().map(|(_, p)| *p).sum();
    let mut rng = rand::thread_rng();
    let r = rng.gen_range(0.0..nuc_sum);
    let mut acc = 0.0f32;
    for &(idx, p) in nucleus {
        acc += p;
        if r < acc { return idx as u32; }
    }
    nucleus[nucleus.len() - 1].0 as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_zero_keeps_full_vocab() {
        let logits = [0.1f32, 2.0, -1.0, 0.5];
        for _ in 0..64 {
            assert!((sample(&logits, 0.7, 0, 1.0) as usize) < logits.len());
        }
    }

    #[test]
    fn top_k_at_or_above_vocab_len() {
        let logits = [0.1f32, 2.0, -1.0, 0.5];
        for k in [4usize, 5, usize::MAX] {
            for _ in 0..16 {
                assert!((sample(&logits, 0.7, k, 0.9) as usize) < logits.len());
            }
        }
    }

    #[test]
    fn top_k_one_is_argmax() {
        let logits = [0.1f32, 2.0, -1.0, 0.5];
        assert_eq!(sample(&logits, 0.7, 1, 0.9), 1);
    }
}

#[cfg(test)]
mod gpu_top_k_tests {
    use super::*;

    #[test]
    fn zero_and_negative_mapped_zero_become_the_kernel_limit_not_argmax() {
        assert_eq!(gpu_top_k(0), GPU_SAMPLE_K_MAX as i32);
        assert!(gpu_top_k_capped(0));
    }

    #[test]
    fn values_inside_the_limit_pass_unchanged() {
        for k in [1usize, 2, 20, 63, 64] {
            assert_eq!(gpu_top_k(k), k as i32, "top_k {k}");
            assert!(!gpu_top_k_capped(k), "top_k {k}");
        }
    }

    #[test]
    fn values_above_the_limit_are_capped_at_it() {
        for k in [65usize, 248320, 248321, u32::MAX as usize, usize::MAX] {
            assert_eq!(gpu_top_k(k), GPU_SAMPLE_K_MAX as i32, "top_k {k}");
            assert!(gpu_top_k_capped(k), "top_k {k}");
        }
    }

    #[test]
    fn the_kernel_limit_matches_gpu_batch_cu() {
        let cu = src_text!("kernels/gpu_batch");
        let line = cu.lines().find(|l| l.starts_with("#define SAMPLE_K_MAX ")).expect("SAMPLE_K_MAX in gpu_batch.cu");
        assert_eq!(line.trim_end(), format!("#define SAMPLE_K_MAX {GPU_SAMPLE_K_MAX}"));
    }
}
