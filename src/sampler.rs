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

    // Top-k: find indices of the k largest logits (O(n) via partition)
    let k = top_k.min(v);
    let mut indices: Vec<usize> = (0..v).collect();
    indices.select_nth_unstable_by(k, |&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
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
