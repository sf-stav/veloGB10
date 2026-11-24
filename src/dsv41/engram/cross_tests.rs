//! Cross-check between ENGa (`layout.rs`, the hashing side) and ENGb (`manifest.rs`, the packer side):
//! the two legs computed the Engram bucket geometry independently, and the shard unit (one head's
//! rows in every Engram layer) must be the same object on both sides, section by section.

use super::layout::{EngramConfig, EngramLayout};
use super::manifest::{sections_for, Geometry};

/// The V4.1-Flash-1017 geometry: the reference dump `tools/dsv41_ref/engram_layout_1017.json`.
fn v41_cfg() -> EngramConfig {
    let j: serde_json::Value = serde_json::from_str(include_str!("../../../tools/dsv41_ref/engram_layout_1017.json")).unwrap();
    EngramConfig {
        layer_ids: j["layer_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect(),
        max_ngram_size: 4,
        vocab_size: j["vocab_size"].as_u64().unwrap(),
        n_heads: 8,
        head_dim: 256,
        num_embeddings: j["num_embeddings"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect(),
        pad_id: 2,
        compressed_vocab_size: 99_092,
    }
}

/// ENGa `rows_for_head(h)` == ENGb `sections_for(geo, h)` for every head and layer: the same
/// (layer_idx, layer_id, g, source_row_start, rows, first_record) and the same total.
#[test]
fn rows_for_head_equals_sections_for_every_head_and_layer() {
    let cfg = v41_cfg();
    let geo = Geometry::compute(&cfg.layer_ids, cfg.max_ngram_size, cfg.n_heads, cfg.vocab_size, &cfg.num_embeddings);
    geo.check_tiles().unwrap();
    let lay = EngramLayout::new(cfg.clone()).unwrap();
    let mut checked = 0usize;
    for head in 0..cfg.n_heads {
        let a = lay.rows_for_head(head);
        let (b, b_total) = sections_for(&geo, head);
        assert_eq!(a.sections.len(), cfg.layer_ids.len() * (cfg.max_ngram_size - 1), "head {head}: section count");
        assert_eq!(a.sections.len(), b.len(), "head {head}");
        assert_eq!(a.total_records, b_total, "head {head}: total records");
        for (x, y) in a.sections.iter().zip(&b) {
            assert_eq!((x.layer_idx, x.layer_id, x.g), (y.layer_idx, y.layer_id, y.g), "head {head}");
            assert_eq!(x.source_row_start, y.source_row_start, "head {head} layer {} g {}", x.layer_idx, x.g);
            assert_eq!(x.rows, y.rows, "head {head} layer {} g {}", x.layer_idx, x.g);
            assert_eq!(x.first_record, y.first_record, "head {head} layer {} g {}", x.layer_idx, x.g);
            assert_eq!(x.head, head);
            checked += 1;
        }
    }
    // 8 heads x 2 layers x 3 n-gram columns: a vacuous loop must fail
    assert_eq!(checked, 48);
}
