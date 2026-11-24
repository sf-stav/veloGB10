//! The dsv41 host sampler: DeepSeek's sampling semantics on a logits row (PLAN/DSV41_XREF_DEEPINFER.md §5).
//!
//! * defaults temperature 1.0, top_p 1.0, no top-k, no penalties (the model card; THE POSTURE sends no sampler fields);
//! * top-p and top-k are applied **jointly** on the tempered distribution, ties at the cut keep the **lowest token ids**,
//!   the kept mass is renormalised;
//! * the draw is an inverse CDF over the kept set with a **counter RNG keyed by (seed, position * 3 + stream)**, so a
//!   sampled reply does not depend on batch composition or on the number of steps taken (DeepSeek's design; our bit
//!   stream is our own splitmix64, not theirs);
//! * greedy = argmax with the lowest id on ties (DeepSeek runs greedy as temperature 1e-5);
//! * presence / frequency / repetition penalties exist for clients that send them (DeepSeek's engine has none).
//!
//! This is the CPU reference sampler of the serve shell. The GPU sampler with top-k candidates is a later leg
//! (roadmap P7); this code is its oracle.

/// Below this temperature a request is greedy (DeepSeek: 1e-5).
pub const GREEDY_T: f32 = 1e-5;

#[derive(Clone, Debug, PartialEq)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    /// 0 = off
    pub top_k: usize,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    /// 1.0 = off
    pub repetition_penalty: f32,
}

impl Default for Sampling {
    /// DeepSeek's: temperature 1.0, top_p 1.0, no top-k, no penalties.
    fn default() -> Self {
        Sampling { temperature: 1.0, top_p: 1.0, top_k: 0, presence_penalty: 0.0, frequency_penalty: 0.0, repetition_penalty: 1.0 }
    }
}

impl Sampling {
    pub fn greedy(&self) -> bool { self.temperature <= GREEDY_T }
    fn has_penalty(&self) -> bool {
        self.presence_penalty != 0.0 || self.frequency_penalty != 0.0 || (self.repetition_penalty - 1.0).abs() > 1e-9
    }
}

/// splitmix64 finaliser.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The counter RNG: a uniform in [0, 1) from (seed, counter). 53 bits.
pub fn uniform(seed: u64, counter: u64) -> f64 {
    let r = mix(mix(seed) ^ counter.wrapping_mul(0xD6E8_FEB8_6659_FD93));
    (r >> 11) as f64 / (1u64 << 53) as f64
}

/// The draw stream of a position: `position * 3 + stream` (stream 0 = the token draw; 1 / 2 are reserved as
/// DeepSeek reserves them for draft draws).
pub fn counter(position: usize, stream: u64) -> u64 { position as u64 * 3 + stream }

/// Apply the penalties of `s` in place, over the tokens the request generated so far (`history`).
pub fn apply_penalties(logits: &mut [f32], history: &[u32], s: &Sampling) {
    if !s.has_penalty() || history.is_empty() { return; }
    let mut count: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for &t in history { *count.entry(t).or_insert(0) += 1; }
    for (t, c) in count {
        let Some(l) = logits.get_mut(t as usize) else { continue };
        if s.repetition_penalty != 1.0 { *l = if *l > 0.0 { *l / s.repetition_penalty } else { *l * s.repetition_penalty }; }
        *l -= s.frequency_penalty * c as f32 + s.presence_penalty;
    }
}

/// Argmax, the lowest id on ties; ids in `exclude` never win. None if every id is excluded or non-finite.
pub fn argmax(logits: &[f32], exclude: &[u32]) -> Option<u32> {
    let mut best: Option<(u32, f32)> = None;
    for (i, &l) in logits.iter().enumerate() {
        if l.is_nan() { continue; }
        if exclude.contains(&(i as u32)) { continue; }
        if best.map_or(true, |(_, b)| l > b) { best = Some((i as u32, l)); }
    }
    best.map(|(i, _)| i)
}

/// The kept set after temperature, joint top-k / top-p and renormalisation: (id, probability), ids ascending.
/// `exclude` ids are removed before anything else (the alternates of a sink rejection).
pub fn kept_set(logits: &[f32], s: &Sampling, exclude: &[u32]) -> Vec<(u32, f64)> {
    let t = s.temperature.max(GREEDY_T) as f64;
    // candidates in (logit desc, id asc) order = the cut order
    let mut cand: Vec<(u32, f32)> = logits.iter().enumerate()
        .filter(|(i, l)| l.is_finite() && !exclude.contains(&(*i as u32)))
        .map(|(i, &l)| (i as u32, l)).collect();
    if cand.is_empty() { return Vec::new(); }
    let by = |a: &(u32, f32), b: &(u32, f32)| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0));
    if s.top_k > 0 && s.top_k < cand.len() {
        cand.select_nth_unstable_by(s.top_k - 1, by);
        cand.truncate(s.top_k);
    }
    cand.sort_unstable_by(by);
    let mx = cand[0].1 as f64;
    let mut probs: Vec<f64> = cand.iter().map(|&(_, l)| ((l as f64 - mx) / t).exp()).collect();
    let z: f64 = probs.iter().sum();
    for p in probs.iter_mut() { *p /= z; }
    if s.top_p < 1.0 {
        let mut cum = 0.0;
        let mut cut = probs.len();
        for (i, p) in probs.iter().enumerate() {
            cum += p;
            if cum >= s.top_p as f64 { cut = i + 1; break; }
        }
        cand.truncate(cut);
        probs.truncate(cut);
        let z2: f64 = probs.iter().sum();
        for p in probs.iter_mut() { *p /= z2; }
    }
    let mut out: Vec<(u32, f64)> = cand.iter().map(|c| c.0).zip(probs).collect();
    out.sort_unstable_by_key(|x| x.0);
    out
}

/// Draw one token. `history` = the request's generated tokens (penalties), `seed` / `position` key the draw,
/// `exclude` = ids already rejected at this position (sink alternates), `salt` separates the alternates' draws
/// from the first draw (0 = the position's own draw). Returns None only when nothing can be drawn.
pub fn sample(logits: &[f32], s: &Sampling, history: &[u32], seed: u64, position: usize, exclude: &[u32], salt: u64) -> Option<u32> {
    let mut owned;
    let row: &[f32] = if s.has_penalty() { owned = logits.to_vec(); apply_penalties(&mut owned, history, s); &owned } else { logits };
    if s.greedy() { return argmax(row, exclude); }
    let kept = kept_set(row, s, exclude);
    if kept.is_empty() { return None; }
    // the first draw (salt 0) uses the position's stream 0; an alternate re-keys the seed so its draw is independent
    let u = uniform(seed ^ salt.wrapping_mul(0xA24B_AED4_963E_E407), counter(position, 0));
    let mut cum = 0.0;
    for &(id, p) in &kept {
        cum += p;
        if u < cum { return Some(id); }
    }
    kept.last().map(|x| x.0)
}

/// The seed of a request that sent none. `random` = a fresh 64-bit value; `prompt-hash` = a hash of the prompt ids
/// (an identical request draws identical text, and every TP rank agrees without a message).
pub fn seedless_seed(mode: &str, prompt: &[u32]) -> u64 {
    if mode == "prompt-hash" {
        let mut h: u64 = 0xCBF2_9CE4_8422_2325;
        for &t in prompt { h = (h ^ t as u64).wrapping_mul(0x0000_0100_0000_01B3); }
        return mix(h);
    }
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
    mix(t ^ mix(N.fetch_add(1, Ordering::Relaxed) ^ (std::process::id() as u64) << 32))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(n: usize) -> Vec<f32> { (0..n).map(|i| ((i * 7919) % 101) as f32 * 0.05).collect() }

    #[test]
    fn greedy_is_argmax_lowest_id_on_ties() {
        let l = [1.0, 5.0, 5.0, 2.0];
        let s = Sampling { temperature: 0.0, ..Default::default() };
        assert_eq!(sample(&l, &s, &[], 1, 0, &[], 0), Some(1));
        assert_eq!(sample(&l, &s, &[], 1, 0, &[1], 0), Some(2));
    }

    #[test]
    fn the_draw_is_keyed_by_seed_and_position_only() {
        let l = row(500);
        let s = Sampling::default();
        let a: Vec<_> = (0..40).map(|p| sample(&l, &s, &[], 42, p, &[], 0)).collect();
        let b: Vec<_> = (0..40).rev().map(|p| sample(&l, &s, &[], 42, p, &[], 0)).collect::<Vec<_>>().into_iter().rev().collect();
        assert_eq!(a, b, "order of the steps must not matter");
        let c: Vec<_> = (0..40).map(|p| sample(&l, &s, &[], 43, p, &[], 0)).collect();
        assert_ne!(a, c, "a different seed must draw differently");
        assert!(a.iter().collect::<std::collections::HashSet<_>>().len() > 3, "draws must vary across positions");
    }

    #[test]
    fn top_k_and_top_p_apply_jointly_with_lowest_id_ties() {
        // four tied tokens at the top, top-k 2: the two LOWEST ids stay
        let l = [3.0, 3.0, 3.0, 3.0, 0.0, 0.0];
        let s = Sampling { top_k: 2, ..Default::default() };
        let k = kept_set(&l, &s, &[]);
        assert_eq!(k.iter().map(|x| x.0).collect::<Vec<_>>(), vec![0, 1]);
        assert!((k.iter().map(|x| x.1).sum::<f64>() - 1.0).abs() < 1e-12);
        // top-p 0.3 of four equal tokens (each ~0.25): needs two tokens (0.5 >= 0.3), ids 0 and 1
        let s = Sampling { top_p: 0.3, ..Default::default() };
        let k = kept_set(&[3.0, 3.0, 3.0, 3.0], &s, &[]);
        assert_eq!(k.iter().map(|x| x.0).collect::<Vec<_>>(), vec![0, 1]);
        // joint: top-k 3 then top-p 0.3 -> still the two lowest ids
        let s = Sampling { top_k: 3, top_p: 0.3, ..Default::default() };
        assert_eq!(kept_set(&[3.0, 3.0, 3.0, 3.0], &s, &[]).len(), 1, "three tokens of 1/3 each: the first already passes 0.3");
    }

    #[test]
    fn sampled_frequencies_follow_the_distribution() {
        // p = softmax([0, ln 3]) = [0.25, 0.75]
        let l = [0.0, 3f32.ln(), -100.0];
        let s = Sampling::default();
        let n = 20000;
        let ones = (0..n).filter(|&p| sample(&l, &s, &[], 7, p, &[], 0) == Some(1)).count();
        let f = ones as f64 / n as f64;
        assert!((f - 0.75).abs() < 0.02, "freq {f}");
    }

    #[test]
    fn excluded_ids_are_never_drawn_and_alternates_use_their_own_draw() {
        let l = row(300);
        let s = Sampling::default();
        let first = sample(&l, &s, &[], 5, 9, &[], 0).unwrap();
        for salt in 1..4 {
            let alt = sample(&l, &s, &[], 5, 9, &[first], salt).unwrap();
            assert_ne!(alt, first);
        }
    }

    #[test]
    fn penalties_move_the_logits() {
        let mut l = vec![1.0, 1.0, 1.0];
        let s = Sampling { presence_penalty: 0.5, frequency_penalty: 0.25, ..Default::default() };
        apply_penalties(&mut l, &[1, 1, 2], &s);
        assert_eq!(l, vec![1.0, 1.0 - 0.5 - 0.5, 1.0 - 0.25 - 0.5]);
        let mut l = vec![2.0, -2.0];
        apply_penalties(&mut l, &[0, 1], &Sampling { repetition_penalty: 2.0, ..Default::default() });
        assert_eq!(l, vec![1.0, -4.0]);
    }

    #[test]
    fn seedless_modes() {
        assert_eq!(seedless_seed("prompt-hash", &[1, 2, 3]), seedless_seed("prompt-hash", &[1, 2, 3]));
        assert_ne!(seedless_seed("prompt-hash", &[1, 2, 3]), seedless_seed("prompt-hash", &[1, 2, 4]));
        assert_ne!(seedless_seed("random", &[1]), seedless_seed("random", &[1]));
    }
}
