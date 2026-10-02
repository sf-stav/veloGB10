//! TP-4F-prep: the world-4 frequency-balanced expert deal (`--ep-deal freq` at `--tp 4`; opt-in and
//! output-changing exactly like the world-2 table, default stays `interleave`).
//!
//! The committed table (`data/ep_deal/qwen38_flash_next_tp4.json`, derived by
//! `scripts/ep_deal/derive_deal.py --world 4`) stores, per rank, per trunk layer, the routed experts that rank
//! holds (`ranks[r][layer]`, ne/world each). It is loaded and validated as a whole: every expert of every layer is
//! owned by exactly one rank, every rank holds exactly ne/world of a layer, every rank lists the same number of
//! layers, every id is < ne. A table that fails any of that is refused (at the head BEFORE the model load,
//! through `flag_refusal`; again at every `ep_layer_owners` call). World 2 keeps its own table and loader in
//! `xtp.rs` (`freq_deal`) untouched; `--tp-ep-deal-file` supplies an alternative file in the format of the
//! running world.
use anyhow::Result;

/// The committed world-4 deal (see the module doc).
pub const FREQ_DEAL_W4_JSON: &str = include_str!("../../../data/ep_deal/qwen38_flash_next_tp4.json");

const FORMAT: &str = "gb10-ep-deal/2";

/// A validated deal table: `owners[layer][expert]` = owning rank.
#[derive(Debug)]
pub struct Table {
    pub world: usize,
    pub ne: usize,
    pub owners: Vec<Vec<u8>>,
}

fn uint(v: &serde_json::Value, what: &str) -> std::result::Result<usize, String> {
    v.as_u64().map(|u| u as usize).ok_or_else(|| format!("deal table: {what} is not an unsigned integer"))
}

/// Parse and validate a `gb10-ep-deal/2` table (pure; no global state).
pub fn parse(txt: &str) -> std::result::Result<Table, String> {
    let v: serde_json::Value = serde_json::from_str(txt).map_err(|e| format!("deal table JSON: {e}"))?;
    let fmt = v["format"].as_str().unwrap_or("");
    if fmt != FORMAT {
        return Err(format!("deal table: format {fmt:?}, expected {FORMAT:?} (a world-2 `rank0` table \
                            cannot be used at world 4 — regenerate with derive_deal.py --world 4)"));
    }
    let world = uint(&v["world"], "world")?;
    let ne = uint(&v["ne"], "ne")?;
    let layers = uint(&v["layers"], "layers")?;
    if world < 3 || ne == 0 || ne % world != 0 { return Err(format!("deal table: bad geometry world {world} ne {ne}")); }
    let ranks = v["ranks"].as_array().ok_or("deal table: no ranks array")?;
    if ranks.len() != world { return Err(format!("deal table: {} rank rows for world {world}", ranks.len())); }
    let mut owners = vec![vec![u8::MAX; ne]; layers];
    for (r, rows) in ranks.iter().enumerate() {
        let rows = rows.as_array().ok_or_else(|| format!("deal table: rank {r} is not an array of layers"))?;
        if rows.len() != layers { return Err(format!("deal table: rank {r} lists {} layers, the table says {layers}", rows.len())); }
        for (l, row) in rows.iter().enumerate() {
            let row = row.as_array().ok_or_else(|| format!("deal table: rank {r} layer {l} is not an array"))?;
            if row.len() != ne / world {
                return Err(format!("deal table: layer {l} gives rank {r} {} experts, expected ne/world = {}", row.len(), ne / world));
            }
            for x in row {
                let e = uint(x, "expert id")?;
                if e >= ne { return Err(format!("deal table: layer {l} rank {r} names expert {e} >= ne {ne}")); }
                if owners[l][e] != u8::MAX {
                    return Err(format!("deal table: layer {l} expert {e} is owned twice (ranks {} and {r})", owners[l][e]));
                }
                owners[l][e] = r as u8;
            }
        }
    }
    for (l, o) in owners.iter().enumerate() {
        if let Some(e) = o.iter().position(|&x| x == u8::MAX) { return Err(format!("deal table: layer {l} expert {e} has no owner")); }
    }
    Ok(Table { world, ne, owners })
}

impl Table {
    /// Owner rank of every expert of trunk layer `layer`, checked against the running model / world.
    pub fn layer_owners(&self, ne: usize, world: usize, layer: usize) -> Result<Vec<usize>> {
        anyhow::ensure!(world == self.world,
            "--ep-deal=freq: the table is a world-{} deal (world {world})", self.world);
        anyhow::ensure!(ne == self.ne,
            "--ep-deal=freq: the table is for {} routed experts, the model has {ne} (a different model? regenerate it \
             with scripts/ep_deal/)", self.ne);
        let o = self.owners.get(layer).ok_or_else(|| anyhow::anyhow!(
            "--ep-deal=freq: the table has {} layers, no layer {layer} (a different model? regenerate it with scripts/ep_deal/)",
            self.owners.len()))?;
        Ok(o.iter().map(|&r| r as usize).collect())
    }
}

/// The table in effect at world 4: `--tp-ep-deal-file` (diagnostic; the file must exist on every rank's box)
/// or the committed one. Parsed and validated once.
pub fn table() -> Result<&'static Table> {
    static T: std::sync::OnceLock<std::result::Result<Table, String>> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let txt = match crate::opts::var(crate::opt!("tp-ep-deal-file")) {
            Ok(p) if !p.is_empty() => std::fs::read_to_string(&p).map_err(|e| format!("--tp-ep-deal-file {p}: {e}"))?,
            _ => FREQ_DEAL_W4_JSON.to_string(),
        };
        parse(&txt)
    }).as_ref().map_err(|e| anyhow::anyhow!("--ep-deal=freq (world 4): {e}"))
}

/// `--ep-deal freq` is admitted at world 4 only (world 2 has its own path and never reaches this). At world 4
/// the table is loaded and validated here, so a bad `--tp-ep-deal-file` refuses the launch before the model load.
pub fn flag_refusal(kind: &str, world: i32) -> Result<()> {
    if kind != "freq" { return Ok(()); }
    anyhow::ensure!(world == 4,
        "EXL3 TP world {world}: --ep-deal freq has committed tables for world 2 and world 4 only (use interleave or contig)");
    table().map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exl3_forward::xtp::{ep_layer_owners_kind, ep_owner, world_gt2_flag_refusals};

    const NE: usize = 512;
    const LAYERS: usize = 48;
    /// The pack's config under `$HOME/models` (lab boxes); the test falls back to EMBEDDED where it is absent.
    fn pack_config() -> std::path::PathBuf {
        std::path::Path::new(&std::env::var("HOME").unwrap_or_default()).join("models/Qwen3.8-Flash-Next-exl3-3.05bpw/config.json")
    }
    const EMBEDDED: &str = r#"{"model_type":"qwen4_exp","text_config":{"hidden_size":2560,"head_dim":256,
        "num_attention_heads":24,"num_key_value_heads":2,"linear_num_key_heads":16,"linear_num_value_heads":48,
        "linear_key_head_dim":128,"linear_value_head_dim":128,"num_experts":512,"num_experts_per_tok":10,
        "moe_intermediate_size":640,"shared_expert_intermediate_size":640,"num_hidden_layers":48,
        "vocab_size":248320,"indexer_n_heads":4,"indexer_head_dim":128,"indexer_budget":2048,
        "indexer_compress_ratio":4,"indexer_kv_heads":1}}"#;

    fn cfg_of(txt: &str, tag: &str) -> crate::qwen::Config {
        let p = std::env::temp_dir().join(format!("tp4f_cfg_{}_{tag}.json", std::process::id()));
        std::fs::write(&p, txt).unwrap();
        let c = crate::qwen::Config::from_config_json(p.to_str().unwrap()).unwrap();
        let _ = std::fs::remove_file(&p);
        c
    }

    /// Build a table JSON with `ranks[r][layer]` = the given sets, for the negative tests.
    fn json_of(world: usize, ne: usize, layers: usize, ranks: &[Vec<Vec<usize>>]) -> String {
        serde_json::json!({"format": FORMAT, "world": world, "ne": ne, "layers": layers, "ranks": ranks}).to_string()
    }

    /// A correct 4-rank, ne 8, 2-layer table (interleave) that the negative tests then break one way at a time.
    fn good_ranks() -> Vec<Vec<Vec<usize>>> {
        (0..4).map(|r| (0..2).map(|_| vec![r, r + 4]).collect()).collect()
    }

    #[test]
    fn committed_table_is_a_balanced_partition_of_every_layer() {
        let t = parse(FREQ_DEAL_W4_JSON).expect("committed tp4 table must validate");
        assert_eq!((t.world, t.ne, t.owners.len()), (4, NE, LAYERS));
        for (l, o) in t.owners.iter().enumerate() {
            assert_eq!(o.len(), NE);
            let mut per = [0usize; 4];
            for &r in o { assert!((r as usize) < 4, "layer {l}: owner {r}"); per[r as usize] += 1; }
            assert_eq!(per, [NE / 4; 4], "layer {l}: every rank holds exactly ne/4, every expert exactly once");
        }
        // independent recount straight from the raw JSON (not through `parse`): each expert listed exactly once
        let v: serde_json::Value = serde_json::from_str(FREQ_DEAL_W4_JSON).unwrap();
        for l in 0..LAYERS {
            let mut seen = vec![0u8; NE];
            for r in 0..4 { for x in v["ranks"][r][l].as_array().unwrap() { seen[x.as_u64().unwrap() as usize] += 1; } }
            assert!(seen.iter().all(|&c| c == 1), "layer {l}");
        }
        // a real deal, not the interleave in disguise
        let moved = (0..LAYERS).filter(|&l| (0..NE).any(|e| t.owners[l][e] as usize != e % 4)).count();
        assert!(moved > LAYERS / 2, "only {moved} of {LAYERS} layers differ from interleave");
    }

    #[test]
    fn layer_count_matches_the_models_moe_layers() {
        let c = cfg_of(EMBEDDED, "emb");
        assert!(c.mlp_only_layers.is_empty(), "every trunk layer is MoE, so the table is indexed by trunk layer");
        assert_eq!(c.num_experts, NE);
        let t = parse(FREQ_DEAL_W4_JSON).unwrap();
        assert_eq!(t.owners.len(), c.num_layers - c.mlp_only_layers.len());
        if let Ok(txt) = std::fs::read_to_string(pack_config()) {
            let real = cfg_of(&txt, "real");
            assert_eq!((real.num_layers - real.mlp_only_layers.len(), real.num_experts), (t.owners.len(), t.ne),
                "the table drifted from the real pack's config.json");
        }
    }

    /// The per-rank shared-expert carry the deal was balanced against (table field `shared_carry`, routed-expert
    /// units) is the code's OWN split: `TrunkSplit::shared(640)` widths / 640 at world 4 (256|128|128|128).
    #[test]
    fn shared_expert_carry_in_the_table_is_the_codes_split() {
        use crate::exl3_forward::xtp::TrunkSplit;
        let v: serde_json::Value = serde_json::from_str(FREQ_DEAL_W4_JSON).unwrap();
        let carry: Vec<f64> = v["shared_carry"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        let widths: Vec<usize> = (0..4).map(|r| { let (a, b) = TrunkSplit { rank: r, world: 4 }.shared(640).unwrap(); b - a }).collect();
        assert_eq!(widths, [256, 128, 128, 128]);
        assert_eq!(widths.iter().sum::<usize>(), 640);
        assert_eq!(carry.len(), 4);
        for r in 0..4 { assert!((carry[r] - widths[r] as f64 / 640.0).abs() < 1e-12, "rank {r}: {} vs {}", carry[r], widths[r]); }
        // and the world-2 bias the TP-I table was derived with (0.2) is the same arithmetic on the same function
        let w2: Vec<usize> = (0..2).map(|r| { let (a, b) = TrunkSplit { rank: r, world: 2 }.shared(640).unwrap(); b - a }).collect();
        assert_eq!(w2, [384, 256]);
        assert!(((w2[0] - w2[1]) as f64 / 640.0 - 0.2).abs() < 1e-12);
    }

    #[test]
    fn freq_dispatch_world_4_uses_the_table_and_checks_the_model() {
        let t = parse(FREQ_DEAL_W4_JSON).unwrap();
        for l in 0..LAYERS {
            let own = ep_layer_owners_kind("freq", NE, 4, l).unwrap();
            assert_eq!(own, t.owners[l].iter().map(|&r| r as usize).collect::<Vec<_>>());
            for r in 0..4 { assert_eq!(own.iter().filter(|&&o| o == r).count(), NE / 4); }
        }
        let e = ep_layer_owners_kind("freq", NE, 4, LAYERS).unwrap_err().to_string();
        assert!(e.contains("48 layers") && e.contains("no layer 48"), "{e}");
        let e = ep_layer_owners_kind("freq", 256, 4, 0).unwrap_err().to_string();
        assert!(e.contains("512 routed experts") && e.contains("256"), "{e}");
    }

    #[test]
    fn world_2_path_is_unchanged() {
        // the committed world-2 table through the ORIGINAL path, cross-checked against the raw JSON
        let v: serde_json::Value = serde_json::from_str(include_str!("../../../data/ep_deal/qwen38_flash_next_tp2.json")).unwrap();
        for l in 0..LAYERS {
            let mut want = vec![1usize; NE];
            for x in v["rank0"][l].as_array().unwrap() { want[x.as_u64().unwrap() as usize] = 0; }
            assert_eq!(ep_layer_owners_kind("freq", NE, 2, l).unwrap(), want, "layer {l}");
        }
        let e = ep_layer_owners_kind("freq", NE, 2, LAYERS).unwrap_err().to_string();
        assert!(e.contains("the table has 48 layers, no layer 48"), "{e}");
        // the static deals are untouched at every world
        for w in [2usize, 4] {
            for l in [0usize, 7, 47] {
                assert_eq!(ep_layer_owners_kind("interleave", NE, w, l).unwrap(), (0..NE).map(|e| e % w).collect::<Vec<_>>());
                assert_eq!(ep_layer_owners_kind("contig", NE, w, l).unwrap(), (0..NE).map(|e| ep_owner(e, NE, w, "contig")).collect::<Vec<_>>());
                assert_eq!(ep_layer_owners_kind("contig", NE, w, l).unwrap()[NE - 1], w - 1);
            }
        }
        // world-2 flag refusals are a no-op with or without the new helper
        assert!(world_gt2_flag_refusals(2).is_ok() && world_gt2_flag_refusals(1).is_ok());
        assert!(flag_refusal("interleave", 2).is_ok());
    }

    #[test]
    fn freq_is_refused_at_every_world_but_2_and_4() {
        for w in [3usize, 5, 8] {
            let e = ep_layer_owners_kind("freq", NE, w, 0).unwrap_err().to_string();
            assert!(e.contains("the committed table is a world-2 deal") && e.contains(&format!("world {w}")), "{e}");
            assert!(flag_refusal("freq", w as i32).is_err(), "world {w}");
            assert!(world_gt2_flag_refusals(w as i32).is_err(), "world {w}");
        }
        let e = flag_refusal("freq", 8).unwrap_err().to_string();
        assert!(e.contains("world 8") && e.contains("world 2 and world 4 only"), "{e}");
        assert!(flag_refusal("freq", 4).is_ok(), "the committed table validates at world 4");
        assert!(flag_refusal("interleave", 4).is_ok() && flag_refusal("contig", 4).is_ok());
    }

    #[test]
    fn parser_accepts_a_good_table_and_refuses_every_broken_one() {
        let t = parse(&json_of(4, 8, 2, &good_ranks())).expect("the good table parses");
        assert_eq!(t.owners[1], vec![0, 1, 2, 3, 0, 1, 2, 3]);
        let broken = |f: &dyn Fn(&mut Vec<Vec<Vec<usize>>>)| { let mut r = good_ranks(); f(&mut r); json_of(4, 8, 2, &r) };
        let cases: Vec<(&str, String, &str)> = vec![
            ("expert owned twice", broken(&|r| r[1][0] = vec![0, 5]), "owned twice"),
            ("rank holds too few", broken(&|r| r[2][1] = vec![2]), "expected ne/world"),
            ("rank holds too many", broken(&|r| r[2][1] = vec![2, 6, 7]), "expected ne/world"),
            ("id out of range", broken(&|r| r[3][0] = vec![3, 8]), ">= ne"),
            ("ragged layers", broken(&|r| { r[0].pop(); }), "lists 1 layers"),
            ("missing rank", broken(&|r| { r.pop(); }), "3 rank rows"),
        ];
        for (name, txt, want) in cases {
            let e = parse(&txt).unwrap_err();
            assert!(e.contains(want), "{name}: {e}");
        }
        let e = parse(&json_of(4, 8, 2, &good_ranks()).replace("gb10-ep-deal/2", "gb10-ep-deal/1")).unwrap_err();
        assert!(e.contains("format"), "{e}");
        let e = parse(r#"{"format":"gb10-ep-deal/1","world":2,"rank0":[[0]]}"#).unwrap_err();
        assert!(e.contains("regenerate with derive_deal.py --world 4"), "a world-2 table at world 4: {e}");
        let e = parse(&json_of(2, 8, 2, &good_ranks()[..2])).unwrap_err();
        assert!(e.contains("bad geometry"), "{e}");
        assert!(parse("not json").is_err());
    }
}
