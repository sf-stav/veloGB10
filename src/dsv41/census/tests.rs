//! Census tests on synthetic packs (no model dir needed): a full EXL3-structured pack, the two
//! mixed packs the owner named (EAQ v1.1-like, dealignai-like), a native-structured pack with
//! Engram tables, and broken packs that must be reported.

use super::classify::TensorClass as C;
use super::report::{rank_bytes, RankBytes};
use super::synth::{small_config, tmp, write_pack, SFmt};
use super::{Census, LinearFormat};
use crate::dsv41::rankmap::Posture;

fn exl3_policy(c: C, l: Option<u16>, _s: Option<u16>) -> SFmt {
    match c {
        C::Embed | C::DsMarkovEmbed => SFmt::Plain("BF16"),
        C::GateWeight | C::IdxWeightsProj | C::DsMarkovHead | C::DsConfidence => SFmt::Plain("F16"),
        C::Head => SFmt::Exl3(6),
        C::IdxWk => SFmt::Exl3(8),
        C::ExpertW1 | C::ExpertW2 | C::ExpertW3 if l.map_or(false, |l| (18..=22).contains(&l)) => SFmt::Exl3(2),
        C::ExpertW1 | C::ExpertW2 | C::ExpertW3 if l.map_or(false, |l| l >= 40) => SFmt::Exl3(4),
        C::ExpertW1 | C::ExpertW2 | C::ExpertW3 => SFmt::Exl3(3),
        _ => SFmt::Exl3(5),
    }
}

fn count(c: &Census, class: C, f: &LinearFormat) -> usize { c.modules.iter().filter(|m| m.class == class && &m.format == f).count() }

#[test]
fn dsv41_census_full_synthetic_exl3_pack_equals_its_index() {
    let d = tmp("census_full");
    let all: Vec<usize> = (0..43).collect();
    write_pack(&d, &small_config(8, 4), &all, true, false, &exl3_policy);
    let c = Census::build(&d).unwrap();
    c.check().unwrap();
    assert_eq!(c.index_keys, c.entries.len());
    assert_eq!(c.index_total, Some(c.total_bytes()));
    assert_eq!(count(&c, C::ExpertW1, &LinearFormat::Exl3 { k: 2, mul1: true, suh: true, svh: true }), 5 * 8);
    assert_eq!(count(&c, C::ExpertW2, &LinearFormat::Exl3 { k: 4, mul1: true, suh: true, svh: true }), 3 * 4);
    assert_eq!(count(&c, C::AttnWoA, &LinearFormat::Exl3 { k: 5, mul1: true, suh: true, svh: true }), 43 * 8);
    assert_eq!(count(&c, C::IdxWeightsProj, &LinearFormat::Plain("F16".into())), 8);
    assert!(c.notes.iter().any(|n| n.contains("Engram tables are NOT in this pack")), "{:?}", c.notes);
    assert!(c.notes.iter().any(|n| n.contains("tensor_storage:")), "{:?}", c.notes);
    // the weight plan sends the plain linears to the plain backend, the rest to EXL3
    let plan = crate::dsv41::weights::WeightPlan::from_census(&c).unwrap();
    assert_eq!(plan.count(crate::dsv41::weights::BackendKind::Plain), 8 + 43 + 1 + 1 + 1 + 1, "{}", plan.summary());
    std::fs::remove_dir_all(&d).ok();
}

#[test]
fn dsv41_census_eaq_v11_like_mixed_pack_has_27_plain_indexer_compressor_linears() {
    let d = tmp("census_eaq11");
    let all: Vec<usize> = (0..43).collect();
    let pol = |c: C, l: Option<u16>, s: Option<u16>| match c {
        C::IdxWqB | C::IdxWeightsProj | C::IdxWk | C::CompWkv | C::CompWgate => SFmt::Plain("BF16"),
        _ => exl3_policy(c, l, s),
    };
    write_pack(&d, &small_config(4, 4), &all, true, false, &pol);
    let c = Census::build(&d).unwrap();
    c.check().unwrap();
    let plain = c.modules.iter().filter(|m| matches!(m.class, C::IdxWqB | C::IdxWeightsProj | C::IdxWk | C::CompWkv | C::CompWgate)
                                             && m.format == LinearFormat::Plain("BF16".into())).count();
    assert_eq!(plain, 27, "8 wq_b + 8 weights_proj + 4 wk + 4 compressor.wkv + 3 wgate");
    let plan = crate::dsv41::weights::WeightPlan::from_census(&c).unwrap();
    for p in &plan.linears {
        if matches!(p.class, C::IdxWqB | C::IdxWk | C::CompWkv | C::CompWgate) {
            assert_eq!(p.backend, crate::dsv41::weights::BackendKind::Plain, "{}", p.key);
        }
    }
    std::fs::remove_dir_all(&d).ok();
}

#[test]
fn dsv41_census_dealignai_like_mixed_pack_has_27_plain_shared_w2() {
    let d = tmp("census_dealign");
    let all: Vec<usize> = (0..43).collect();
    let pol = |c: C, l: Option<u16>, s: Option<u16>| match (c, l) {
        (C::SharedW2, Some(l)) if l < 27 => SFmt::Plain("BF16"),
        _ => exl3_policy(c, l, s),
    };
    write_pack(&d, &small_config(4, 4), &all, true, false, &pol);
    let c = Census::build(&d).unwrap();
    c.check().unwrap();
    assert_eq!(count(&c, C::SharedW2, &LinearFormat::Plain("BF16".into())), 27);
    assert_eq!(c.modules.iter().filter(|m| m.class == C::SharedW2 && matches!(m.format, LinearFormat::Exl3 { .. })).count(), 16);
    std::fs::remove_dir_all(&d).ok();
}

#[test]
fn dsv41_census_native_structured_pack_formats_and_per_rank_split() {
    let d = tmp("census_native");
    // a slice: SWA + Engram (1), a ratio-2 source (2), the candidate source (20), the DSpark stages
    let layers = vec![0, 1, 2, 20, 40, 41, 42];
    let pol = |c: C, _l: Option<u16>, _s: Option<u16>| match c {
        C::ExpertW1 | C::ExpertW2 | C::ExpertW3 => SFmt::Mxfp4,
        C::Embed | C::Head | C::GateWeight | C::IdxWeightsProj | C::IdxWk | C::CompWkv | C::CompWgate | C::DsMarkovEmbed | C::DsMarkovHead | C::DsConfidence => SFmt::Plain("BF16"),
        _ => SFmt::Fp8,
    };
    let cfg = write_pack(&d, &small_config(8, 8), &layers, false, true, &pol);
    let c = Census::build(&d).unwrap();
    c.check().unwrap();
    assert_eq!(c.kept_layers(), layers);
    assert_eq!(count(&c, C::ExpertW1, &LinearFormat::Fp4 { block: 32, scale: "F8_E8M0".into(), scale2: false }), 7 * 8);
    assert_eq!(count(&c, C::AttnWoA, &LinearFormat::Fp8Block { bo: 32, bi: 32, scale: "F8_E8M0".into() }), 7);
    assert_eq!(count(&c, C::EngramTable, &LinearFormat::Fp8Block { bo: 1, bi: 32, scale: "F8_E8M0".into() }), 1);
    assert!(LinearFormat::Fp4 { block: 32, scale: "F8_E8M0".into(), scale2: false }.label().starts_with("MXFP4"));
    // per-rank shares: EP experts, Engram heads and the split classes sum back to the pack
    let lay = crate::dsv41::rankmap::engram_layout(&cfg);
    for p in [Posture::Tp2, Posture::Tp4, Posture::Tp8] {
        let rb: Vec<RankBytes> = rank_bytes(&c, p);
        let experts: u64 = c.entries.iter().filter(|e| e.class.is_routed_expert()).map(|e| e.bytes).sum();
        assert_eq!(rb.iter().map(|b| b.experts).sum::<u64>(), experts, "{p:?}");
        let tables: u64 = c.entries.iter().filter(|e| e.class == C::EngramTable).map(|e| e.bytes).sum();
        assert_eq!(rb.iter().map(|b| b.engram_tables).sum::<u64>(), tables, "{p:?}");
        // layer 1's table bytes of rank r = its heads' rows x 264 B
        let w = p.world();
        let h0 = 0..(8 / w);
        let want: u64 = h0.map(|h| lay.head_rows(0, h)).sum::<u64>() * 264;
        assert_eq!(rb[0].engram_tables, want, "{p:?}");
        assert!(rb.iter().all(|b| b.replicated == rb[0].replicated), "replicated bytes are the same on every rank");
    }
    std::fs::remove_dir_all(&d).ok();
}

#[test]
fn dsv41_census_reports_broken_packs() {
    let d = tmp("census_broken");
    let all: Vec<usize> = (0..43).collect();
    write_pack(&d, &small_config(2, 2), &all, true, false, &exl3_policy);
    // 1. an index key without a tensor and a name that is no V4.1 tensor
    let ip = d.join("model.safetensors.index.json");
    let mut idx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&ip).unwrap()).unwrap();
    idx["weight_map"]["layers.0.attn.ghost.weight"] = "model-layer00.safetensors".into();
    std::fs::write(&ip, idx.to_string()).unwrap();
    let c = Census::build(&d).unwrap();
    assert!(c.check().is_err());
    assert!(c.problems.iter().any(|p| p.contains("index key layers.0.attn.ghost.weight has no tensor")), "{:?}", c.problems);
    // 2. tensor_storage disagreeing with the header (bits 3 for a K=5 trellis)
    let d2 = tmp("census_broken2");
    write_pack(&d2, &small_config(2, 2), &all, true, false, &exl3_policy);
    let qp = d2.join("quantization_config.json");
    let mut q: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&qp).unwrap()).unwrap();
    q["tensor_storage"]["layers.0.attn.wq_a"]["bits_per_weight"] = 3.into();
    std::fs::write(&qp, q.to_string()).unwrap();
    let c = Census::build(&d2).unwrap();
    assert!(c.problems.iter().any(|p| p.contains("layers.0.attn.wq_a: bits_per_weight 3")), "{:?}", c.problems);
    // 3. a missing module (a slice that lists a layer it does not carry)
    let d3 = tmp("census_broken3");
    write_pack(&d3, &small_config(2, 2), &[0, 1], true, false, &exl3_policy);
    let cp = d3.join("config.json");
    let mut cj: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&cp).unwrap()).unwrap();
    cj["dsv41_slice"]["layers"] = serde_json::json!([0, 1, 2]);
    std::fs::write(&cp, cj.to_string()).unwrap();
    let c = Census::build(&d3).unwrap();
    assert!(c.problems.iter().any(|p| p.starts_with("missing module: attn.sink layer Some(2)")), "{:?}", c.problems);
    for x in [d, d2, d3] { std::fs::remove_dir_all(&x).ok(); }
}

#[test]
fn dsv41_census_report_renders() {
    let d = tmp("census_report");
    write_pack(&d, &small_config(2, 2), &[0, 1, 2, 3, 20, 21, 24, 40, 41, 42], true, false, &exl3_policy);
    let c = Census::build(&d).unwrap();
    c.check().unwrap();
    let md = super::report::section(&c, "synthetic");
    assert!(md.contains("| TP8 | 7 |"), "{md}");
    assert!(md.contains("### EXL3 bit widths (K) per class"), "{md}");
    assert!(md.contains("| 20 | r1 kv-src idx-src cand-src |"), "{md}");
    std::fs::remove_dir_all(&d).ok();
}
