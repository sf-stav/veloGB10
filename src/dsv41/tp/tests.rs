//! Pure tests of the TP comm layer (no GPU): the round program against T0's count and payloads, the frame layout,
//! the association model, the transport routing, the ABI mirrors.
use super::frame::*;

#[test]
fn the_round_has_t0s_93_collectives_and_their_payloads() {
    let s = Shape::v41_flash();
    for rows in [1usize, 6, 12, 24] {
        let r = s.round(rows, false);
        assert_eq!(r.len(), 93, "T0 4.1: 80 layer reduces + head + 6 drafter reduces + candidates + 5 Markov steps");
        assert_eq!(s.round(rows, true).len(), 94);
        let layer = r.sites.iter().filter(|(x, _)| matches!(x, Site::Attn(_) | Site::Ffn(_))).count();
        assert_eq!(layer, 80);
    }
    // T0 4.1: 20.6 KB per row and rank for a layer reduce, + 102.4 KB for Engram, + 20.5 KB for main_proj
    let f = s.frame(Site::Attn(5), 1, 2);
    assert_eq!(f.bytes_per_row_per_rank(), (5120 + 25) * 4);
    assert_eq!(f.bytes(), (5120 + MIX_STRIDE) * 4);
    assert_eq!(s.class(Site::Ffn(0)), Class::LayerEngram);
    assert_eq!(s.class(Site::Ffn(13)), Class::LayerEngram);
    assert_eq!(s.class(Site::Ffn(1)), Class::Layer);
    assert_eq!(s.class(Site::Attn(39)), Class::LayerMainProj);
    assert_eq!(s.frame(Site::Ffn(13), 1, 2).extra, 25_600);
    assert_eq!(s.frame(Site::Attn(39), 1, 2).extra, 5120);
    assert_eq!(s.frame(Site::Head, 6, 4).floats(), 6 * 4 * HEAD_REC);
    // the drafter runs at 5 block rows per lane, the Markov chain one row per lane
    let r = s.round(12, false);
    assert!(r.sites.contains(&(Site::DraftAttn(2), 10)) && r.sites.contains(&(Site::Markov(4), 2)));
}

#[test]
fn every_frame_is_a_whole_number_of_16_byte_vectors() {
    let s = Shape::v41_flash();
    for w in [1usize, 2, 4, 8] {
        for rows in 1..=MAX_ROWS {
            for (site, r) in s.round(rows, true).sites {
                assert_eq!(s.frame(site, r, w).floats() % 4, 0, "{site:?} at {r} rows, world {w}");
            }
        }
    }
}

#[test]
fn frame_sections_tile_the_frame() {
    let s = Shape::v41_flash();
    let f = s.frame(Site::Ffn(0), 6, 2);
    assert_eq!(f.mix_off(), 6 * 5120);
    assert_eq!(f.extra_off(), 6 * (5120 + MIX_STRIDE));
    assert_eq!(f.extra_off() + 6 * f.extra, f.floats());
    assert_eq!(s.mix_used(), 25);
    assert_eq!(s.engram_kv(), crate::dsv41::cpu::moe::engram_mod::KOUT);
}

#[test]
fn the_association_model_is_the_transports() {
    // world 2: p0 + p1; world 4 aligned: (p0 + p2) + (p1 + p3) (tp_xport::single_tree_reference)
    let p = |r: usize| (0..64).map(|i| ((i * 7 + r * 13) as f32).sin() * 1e3 + (r as f32) * 1e-4).collect::<Vec<f32>>();
    let two = tree_reference(&[p(0), p(1)]).unwrap();
    assert!(two.iter().enumerate().all(|(i, v)| v.to_bits() == (p(0)[i] + p(1)[i]).to_bits()));
    let four: Vec<Vec<f32>> = (0..4).map(p).collect();
    assert_eq!(tree_reference(&four).unwrap(), crate::tp_xport::single_tree_reference(&four).unwrap());
    assert_eq!(first_round_bit(2), 0);
    assert_eq!(first_round_bit(4), 1);
}

#[test]
fn routing_sends_wide_frames_to_the_pipe_and_decode_frames_to_the_pair_or_single_stage() {
    use super::rdma::{route, Path, PIPE_MIN_BYTES as P};
    let s = Shape::v41_flash();
    let one = s.frame(Site::Attn(3), 1, 2).bytes();
    assert_eq!(route(one, 2, false, true, P), Path::Serial);
    assert_eq!(route(one, 4, true, true, P), Path::Single);
    let wide = s.frame(Site::Ffn(0), 24, 2).bytes();
    assert!(wide > super::rdma::PIPE_MIN_BYTES);
    assert_eq!(route(s.frame(Site::Attn(3), 2, 2).bytes(), 2, false, true, P), Path::Pipe);   // 41 KB: past the measured crossover
    assert_eq!(route(wide, 2, false, true, P), Path::Pipe);
    assert_eq!(route(s.frame(Site::Head, 24, 4).bytes(), 4, true, true, P), Path::Single);
    // a lower pipe threshold (--dsv41-tp-pipe-min-kib) moves mid-size frames to the pipe; 0 = every 16 B-multiple frame
    assert_eq!(route(s.frame(Site::Attn(3), 6, 2).bytes(), 2, false, true, 64 * 1024), Path::Pipe);
    assert_eq!(route(one, 2, false, true, 0), Path::Pipe);
}

#[test]
fn abi_mirrors_have_the_device_sizes() {
    assert_eq!(std::mem::size_of::<super::comm::TpRow>(), 32);
    assert_eq!(std::mem::size_of::<super::comm::TpOut>(), 32);
    assert_eq!(std::mem::size_of::<super::kern::LocalArgs>(), 96);
    let src = include_str!("../../../kernels/gpu_dsv41/05_tp.cuh");
    for k in super::kern::FNS.iter().filter(|k| k.starts_with("d41tp_")) {
        assert!(src.contains(&format!("void {k}(")) || src.contains(&format!("{k}(const")), "{k} not defined in 05_tp.cuh");
    }
}
