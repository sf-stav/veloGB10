//! DSV4.1 (leg I0): the rank map — TP2 / TP4 / TP8 -> ranks -> boxes -> what each rank owns
//! (Engram heads, routed / DSpark experts, attention output groups, vocab rows) — as ONE table in
//! code, plus the Engram bucket layout the byte accounting needs.
//!
//! Boxes (owner rulings 2026-10-26 / 2026-11-18): `.11` is the head of every topology. TP2 = `.11`
//! + `.13`; TP4 = `.11 .12 .13 .14`; TP8 = `.11`..`.18`. Engram tables are sharded by head
//! (the unit is one Engram head h = 0..7: its n-gram columns in every Engram layer): TP8 rank r
//! holds head r, TP4 {2r, 2r+1}, TP2 {4r..4r+3}. The split of everything else here is the plan's
//! PRELIMINARY one (EP experts, attention by `o_groups`, vocab-parallel head, the rest
//! replicated); leg T0 owns the final design and refines `Split` / `split_of` in census.

use crate::dsv41::config::Dsv41Config;
use std::ops::Range;

/// A TP posture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Posture { Tp1, Tp2, Tp3, Tp4, Tp6, Tp8 }

impl Posture {
    pub const ALL: [Posture; 6] = [Posture::Tp1, Posture::Tp2, Posture::Tp3, Posture::Tp4, Posture::Tp6, Posture::Tp8];
    pub fn world(self) -> usize { match self { Posture::Tp1 => 1, Posture::Tp2 => 2, Posture::Tp3 => 3, Posture::Tp4 => 4, Posture::Tp6 => 6, Posture::Tp8 => 8 } }
    pub fn from_world(w: usize) -> Option<Posture> {
        Some(match w { 1 => Posture::Tp1, 2 => Posture::Tp2, 3 => Posture::Tp3, 4 => Posture::Tp4, 6 => Posture::Tp6, 8 => Posture::Tp8, _ => return None })
    }
    pub fn name(self) -> &'static str { match self { Posture::Tp1 => "TP1", Posture::Tp2 => "TP2", Posture::Tp3 => "TP3", Posture::Tp4 => "TP4", Posture::Tp6 => "TP6", Posture::Tp8 => "TP8" } }
}

/// One rank's box in the lab topology (RoCE rail-1 address; rank 0 is always the head).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RankBox { pub rank: usize, pub addr: &'static str, pub host: &'static str }

const B11: (&str, &str) = ("<peer-ip>", "<node-name>");
const B12: (&str, &str) = ("<peer-ip>", "<node-name>");
const B13: (&str, &str) = ("<peer-ip>", "<node-name>");
const B14: (&str, &str) = ("<peer-ip>", "<node-name>");
const B15: (&str, &str) = ("<peer-ip>", "<node-name>");
const B16: (&str, &str) = ("<peer-ip>", "<node-name>");
const B17: (&str, &str) = ("<peer-ip>", "<node-name>");
const B18: (&str, &str) = ("<peer-ip>", "<node-name>");

/// THE box table: posture -> boxes in rank order. Rank 0 = head = `.11`.
pub fn boxes(p: Posture) -> &'static [(&'static str, &'static str)] {
    match p {
        Posture::Tp1 => &[B11],
        Posture::Tp2 => &[B11, B13],
        Posture::Tp3 => &[B11, B12, B13],
        Posture::Tp4 => &[B11, B12, B13, B14],
        Posture::Tp6 => &[B11, B12, B13, B14, B15, B16],
        Posture::Tp8 => &[B11, B12, B13, B14, B15, B16, B17, B18],
    }
}

/// What rank `r` of posture `p` owns (the preliminary split; T0 refines).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankSlot {
    pub rank: usize,
    pub world: usize,
    pub addr: &'static str,
    pub host: &'static str,
    /// Engram heads whose table columns this rank holds and reads
    pub engram_heads: Range<usize>,
    /// routed experts of every trunk layer (EP)
    pub experts: Range<usize>,
    /// DSpark-stage routed experts (EP)
    pub dspark_experts: Range<usize>,
    /// attention output groups (`o_groups`), so query heads `groups * heads/o_groups`
    pub o_groups: Range<usize>,
    pub heads: Range<usize>,
    pub index_heads: Range<usize>,
    /// vocab rows of the LM head (and the DSpark Markov head)
    pub vocab: Range<usize>,
}

fn part(n: usize, w: usize, r: usize) -> Range<usize> { (r * n / w)..((r + 1) * n / w) }

/// The full rank map of a posture (one row per rank).
pub fn rank_map(cfg: &Dsv41Config, p: Posture) -> Vec<RankSlot> {
    let w = p.world();
    boxes(p).iter().enumerate().map(|(r, &(addr, host))| RankSlot {
        rank: r, world: w, addr, host,
        engram_heads: part(cfg.engram_heads, w, r),
        experts: part(cfg.n_routed, w, r),
        dspark_experts: part(cfg.dspark_routed, w, r),
        o_groups: part(cfg.o_groups, w, r),
        heads: part(cfg.heads, w, r),
        index_heads: part(cfg.index_heads, w, r),
        vocab: part(cfg.vocab, w, r),
    }).collect()
}

/// Which dimensions a posture cannot partition cleanly (empty = valid posture).
/// Power-of-two worlds divide every axis evenly; arbitrary-W (TP3 / TP6) partitions whole o_groups / Engram heads
/// and exact 128-wide EXL3 Hadamard blocks (`moe_inter / 128` = 18 divides 3 and 6 exactly!).
pub fn posture_divides(cfg: &Dsv41Config, p: Posture) -> Vec<String> {
    let w = p.world();
    let mut bad = Vec::new();
    if w.is_power_of_two() {
        for (what, n) in [("engram heads", cfg.engram_heads), ("routed experts", cfg.n_routed), ("DSpark experts", cfg.dspark_routed),
                          ("o_groups", cfg.o_groups), ("heads", cfg.heads), ("index heads", cfg.index_heads), ("vocab", cfg.vocab)] {
            if n % w != 0 { bad.push(format!("{what} {n} % {w} != 0")); }
        }
    } else {
        if cfg.engram_heads < w { bad.push(format!("engram heads {} < {w}", cfg.engram_heads)); }
        if cfg.o_groups < w { bad.push(format!("o_groups {} < {w}", cfg.o_groups)); }
        if cfg.o_groups == 0 || cfg.heads % cfg.o_groups != 0 { bad.push(format!("heads {} % o_groups {} != 0", cfg.heads, cfg.o_groups)); }
        if cfg.vocab / 128 < w { bad.push(format!("vocab blocks {} < {w}", cfg.vocab / 128)); }
        if cfg.moe_inter / 128 < w { bad.push(format!("moe_inter blocks {} < {w}", cfg.moe_inter / 128)); }
    }
    bad
}

/// The Engram bucket layout (engram.py `EngramLayout`): per Engram layer, per n-gram order
/// (2..=max), per head, a prime bucket size; buckets are laid out in that order (order-major,
/// then head) and never share a prime. A hash column c = (order - 2) * heads + h.
/// Leg ENGa owns the canonical hashing (`src/dsv41/engram/layout.rs`); this copy feeds the byte
/// accounting and the config check (the primes must sum to `engram_num_embeddings`).
#[derive(Clone, Debug, PartialEq)]
pub struct EngramLayout {
    /// [engram layer][order - 2][head] -> bucket rows
    pub primes: Vec<Vec<Vec<u64>>>,
}

impl EngramLayout {
    /// The first row of column `c` in layer index `li` (the cumulative sum of the earlier buckets).
    pub fn offset(&self, li: usize, c: usize) -> u64 {
        self.primes[li].iter().flatten().take(c).sum()
    }
    /// Rows of head `h` in layer index `li` (its column of every n-gram order).
    pub fn head_rows(&self, li: usize, h: usize) -> u64 {
        self.primes[li].iter().map(|per_order| per_order[h]).sum()
    }
}

fn is_prime(n: u64) -> bool {
    if n < 2 { return false; }
    if n % 2 == 0 { return n == 2; }
    let mut d = 3u64;
    while d * d <= n {
        if n % d == 0 { return false; }
        d += 2;
    }
    true
}

/// engram.py `EngramLayout.from_args`: for each layer, for each order, restart the search at
/// `engram_vocab_size - 1` and take the next `heads` primes not handed out before (globally).
pub fn engram_layout(cfg: &Dsv41Config) -> EngramLayout {
    let mut seen = std::collections::HashSet::new();
    let mut primes = Vec::new();
    for _ in &cfg.engram_layers {
        let mut per_layer = Vec::new();
        for _ in 0..cfg.engram_max_ngram.saturating_sub(1) {
            let mut sizes = Vec::new();
            let mut cur = cfg.engram_vocab.saturating_sub(1);
            for _ in 0..cfg.engram_heads {
                let mut cand = cur + 1;
                while !is_prime(cand) || seen.contains(&cand) { cand += 1; }
                seen.insert(cand);
                sizes.push(cand);
                cur = cand;
            }
            per_layer.push(sizes);
        }
        primes.push(per_layer);
    }
    EngramLayout { primes }
}

/// Engram row record bytes: `head_dim` FP8 values + one E8M0 scale per 32 (264 B for V4.1).
pub fn engram_row_bytes(cfg: &Dsv41Config) -> u64 { (cfg.engram_head_dim + cfg.engram_head_dim / 32) as u64 }

/// Bytes of Engram table rows rank `slot` holds (both Engram layers, its heads' columns).
pub fn engram_bytes_for(cfg: &Dsv41Config, lay: &EngramLayout, heads: Range<usize>) -> u64 {
    let rb = engram_row_bytes(cfg);
    (0..cfg.engram_layers.len()).map(|li| heads.clone().map(|h| lay.head_rows(li, h)).sum::<u64>()).sum::<u64>() * rb
}

/// The rank map as printable lines (`--dsv41-census` and the boot topology row).
pub fn describe(cfg: &Dsv41Config, p: Posture) -> Vec<String> {
    let lay = engram_layout(cfg);
    rank_map(cfg, p).iter().map(|s| format!(
        "{} rank {} {} ({}): Engram heads {:?} ({:.1} GB), experts {:?}, DSpark experts {:?}, o_groups {:?} (heads {:?}), index heads {:?}, vocab {:?}",
        p.name(), s.rank, s.addr, s.host, s.engram_heads, engram_bytes_for(cfg, &lay, s.engram_heads.clone()) as f64 / 1e9,
        s.experts, s.dspark_experts, s.o_groups, s.heads, s.index_heads, s.vocab)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsv41::config::{test_config_json, Dsv41Config};

    #[test]
    fn dsv41_rank_map_matches_the_owner_table() {
        let c = Dsv41Config::from_value(&test_config_json(), None).unwrap();
        for p in Posture::ALL { assert!(posture_divides(&c, p).is_empty(), "{p:?}"); }
        let t8 = rank_map(&c, Posture::Tp8);
        for (r, s) in t8.iter().enumerate() { assert_eq!(s.engram_heads, r..r + 1); assert_eq!(s.experts.len(), 48); assert_eq!(s.o_groups.len(), 1); }
        let t4 = rank_map(&c, Posture::Tp4);
        assert_eq!(t4.iter().map(|s| s.addr).collect::<Vec<_>>(), ["<peer-ip>", "<peer-ip>", "<peer-ip>", "<peer-ip>"]);
        for (r, s) in t4.iter().enumerate() { assert_eq!(s.engram_heads, 2 * r..2 * r + 2); assert_eq!(s.experts.len(), 96); }
        let t2 = rank_map(&c, Posture::Tp2);
        assert_eq!(t2[1].addr, "<peer-ip>");
        assert_eq!(t2[1].engram_heads, 4..8);
        assert_eq!(t2[0].experts, 0..192);
        assert_eq!(t2[1].vocab, 64640..129280);
        assert_eq!(rank_map(&c, Posture::Tp1)[0].engram_heads, 0..8);
    }

    #[test]
    fn dsv41_engram_layout_sums_to_the_config_tables() {
        let c = Dsv41Config::from_value(&test_config_json(), None).unwrap();
        let lay = engram_layout(&c);
        assert_eq!(lay.primes.len(), 2);
        assert_eq!(lay.primes[0].len(), 3);
        assert_eq!(lay.primes[0][0][0], 16_000_057, "first prime above 15,999,999");
        let all: Vec<u64> = lay.primes.iter().flatten().flatten().copied().collect();
        let mut u = all.clone(); u.sort(); u.dedup();
        assert_eq!(u.len(), all.len(), "no prime is handed out twice");
        for (li, rows) in c.engram_rows.iter().enumerate() {
            assert_eq!(lay.primes[li].iter().flatten().sum::<u64>(), *rows);
            assert_eq!(lay.offset(li, 24), *rows);
        }
        let total: u64 = (0..8).map(|h| engram_bytes_for(&c, &lay, h..h + 1)).sum();
        assert_eq!(total, (384_006_168 + 384_016_682) * 264);
        assert_eq!(engram_row_bytes(&c), 264);
    }
}
