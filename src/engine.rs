//! Inference engine for Qwen3.5-0.8B using the GPU forward (cuBLAS + CUDA kernels).


use crate::gpu::{GpuModel, GpuState, Pool};
use crate::qwen::Model;
use crate::sampler::Sampler;

pub struct GB10InferenceEngine {
    host: Model,
    gpu: GpuModel,
    state: GpuState,
    pool: Pool,
    sampler: Sampler,
}

impl GB10InferenceEngine {
    pub async fn new(model_path: &str, _max_seq_len: usize) -> anyhow::Result<Self> {
        let host = Model::load(model_path)?;
        let gpu = GpuModel::new(&host)?;
        let state = gpu.new_state();
        let pool = Pool::new(gpu.dev().clone());
        Ok(Self { host, gpu, state, pool, sampler: Sampler::new(0.0, 0.9, 50) })
    }

    fn cos_sin_for(&self, pos: usize) -> (Vec<f32>, Vec<f32>) {
        let rdim = self.gpu.cfg().rotary_dim;
        let half = rdim / 2;
        let theta = self.gpu.cfg().rope_theta;
        let mut cos = vec![0.0f32; rdim];
        let mut sin = vec![0.0f32; rdim];
        for i in 0..half {
            let inv = theta.powf(-(2.0 * i as f32) / rdim as f32);
            let f = (pos as f32) * inv;
            cos[i] = f.cos(); sin[i] = f.sin();
            cos[i + half] = cos[i]; sin[i + half] = sin[i];
        }
        (cos, sin)
    }

    pub fn forward_step(&mut self, token_id: u32) -> anyhow::Result<Vec<f32>> {
        let pos = self.state.pos;
        let (cos, sin) = self.cos_sin_for(pos);
        let cos_d = self.gpu.dev().htod_sync_copy(&cos)?;
        let sin_d = self.gpu.dev().htod_sync_copy(&sin)?;
        let hidden = self.gpu.embed_row(token_id);
        let out = self.gpu.forward_token(&mut self.pool, hidden, pos, &mut self.state, &cos_d, &sin_d);
        self.state.pos += 1;
        let logits_d = self.gpu.logits(&mut self.pool, &out);
        let logits = self.gpu.dev().dtoh_sync_copy(&logits_d)?;
        Ok(logits)
    }

    pub fn decode_step(&mut self, token_id: u32) -> anyhow::Result<u32> {
        let logits = self.forward_step(token_id)?;
        Ok(self.sampler.sample(&logits))
    }

    /// Greedy decode step using on-device argmax (no logits host copy).
    pub fn decode_greedy(&mut self, token_id: u32) -> anyhow::Result<u32> {
        let pos = self.state.pos;
        let (cos, sin) = self.cos_sin_for(pos);
        let cos_d = self.gpu.dev().htod_sync_copy(&cos)?;
        let sin_d = self.gpu.dev().htod_sync_copy(&sin)?;
        let hidden = self.gpu.embed_row(token_id);
        let out = self.gpu.forward_token(&mut self.pool, hidden, pos, &mut self.state, &cos_d, &sin_d);
        self.state.pos += 1;
        let logits = self.gpu.logits(&mut self.pool, &out);
        Ok(self.gpu.argmax_gpu(&mut self.pool, &logits))
    }

    pub fn generate(&mut self, prompt_tokens: &[u32], max_new_tokens: usize) -> Vec<u32> {
        let (out, _, _, _) = self.generate_stats(prompt_tokens, max_new_tokens);
        out
    }

    /// Like generate, but also returns (prefill_tok_s, decode_tok_s, num_new_tokens). Stops on EOS.
    pub fn generate_stats(&mut self, prompt_tokens: &[u32], max_new_tokens: usize)
        -> (Vec<u32>, f32, f32, usize) {
        let eos = self.host.config.eos_token_id;
        let mut output: Vec<u32> = prompt_tokens.to_vec();
        let mut logits: Option<Vec<f32>> = None;

        let prefill_start = std::time::Instant::now();
        for &tok in prompt_tokens {
            match self.forward_step(tok) {
                Ok(l) => logits = Some(l),
                Err(e) => { eprintln!("Forward error: {}", e); return (output, 0.0, 0.0, 0); }
            }
        }
        let prefill_dur = prefill_start.elapsed().as_secs_f32();

        if max_new_tokens == 0 { return (output, 0.0, 0.0, 0); }
        let mut next = match logits { Some(l) => self.sampler.sample(&l), None => return (output, 0.0, 0.0, 0) };
        output.push(next);
        let mut n_new = 1usize;
        if next == eos { return (output, tok_s(prompt_tokens.len(), prefill_dur), 0.0, n_new); }

        let decode_start = std::time::Instant::now();
        let greedy = self.sampler.temperature < 1e-6;
        for _ in 1..max_new_tokens {
            let t = if greedy { match self.decode_greedy(next) { Ok(t)=>t, Err(_)=>break } }
                    else { match self.decode_step(next) { Ok(t)=>t, Err(_)=>break } };
            output.push(t); n_new += 1; if t == eos { break; } next = t;
        }
        let decode_dur = decode_start.elapsed().as_secs_f32();
        let n_decode = n_new.saturating_sub(1).max(1);
        (output, tok_s(prompt_tokens.len(), prefill_dur), tok_s(n_decode, decode_dur), n_new)
    }

    pub fn eos_token(&self) -> u32 { self.host.config.eos_token_id }

    pub fn reset(&mut self) {
        self.state = self.gpu.new_state();
    }
    pub fn seq_len(&self) -> usize { self.state.pos }
    pub fn memory_size(&self) -> usize { self.host.embed_tokens.len() * 4 }
    pub fn set_sampler(&mut self, s: Sampler) { self.sampler = s; }
    pub fn sample_logits(&self, l: &[f32]) -> u32 { self.sampler.sample(l) }
    pub fn dump_kernel_names(&self) {}
    pub fn gpu(&self) -> &GpuModel { &self.gpu }
    pub fn host(&self) -> &Model { &self.host }
}

fn tok_s(n: usize, secs: f32) -> f32 {
    if secs > 1e-6 { n as f32 / secs } else { 0.0 }
}
