//! Unit tests (module `token_logprobs_tests`), moved out of the old inline block in `exl3_serve.rs`; the test text is unchanged.
//! The module is declared in `mod.rs`; `use super::*` there resolves against the parent module as before.

    use super::token_logprobs;

    fn row(xs: &[f32]) -> Vec<u16> { xs.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect() }

    #[test]
    fn log_softmax_and_top_n() {
        let r = row(&[1.0, 3.0, 2.0, 3.0, f32::NEG_INFINITY]);
        let lse = (1f64.exp() + 3f64.exp() + 2f64.exp() + 3f64.exp()).ln();
        let (lp, top) = token_logprobs(&r, 2, 3).unwrap();
        assert!((lp as f64 - (2.0 - lse)).abs() < 1e-5);
        // logit descending, ties by id ascending
        assert_eq!(top.iter().map(|t| t.0).collect::<Vec<_>>(), vec![1, 3, 2]);
        assert!((top[0].1 as f64 - (3.0 - lse)).abs() < 1e-5);
        let p: f64 = (0..4).map(|i| (token_logprobs(&r, i, 0).unwrap().0 as f64).exp()).sum();
        assert!((p - 1.0).abs() < 1e-5, "probabilities sum to 1 over the finite logits");
    }

    /// host cost per token at the model's vocabulary size (run with --ignored --nocapture)
    #[test]
    #[ignore]
    fn host_cost_full_vocab() {
        let v = 248_320usize;
        let r: Vec<u16> = (0..v).map(|i| half::f16::from_f32(((i * 7919) % 2003) as f32 / 97.0 - 10.0).to_bits()).collect();
        let t = std::time::Instant::now();
        for i in 0..100 { std::hint::black_box(token_logprobs(&r, i as u32, 20)); }
        eprintln!("token_logprobs (248K vocab, top 20): {:.3} ms per call", t.elapsed().as_secs_f64() * 10.0);
    }

    #[test]
    fn masked_tokens_are_never_listed() {
        let r = row(&[f32::NEG_INFINITY, 0.5, f32::NEG_INFINITY]);
        let (lp, top) = token_logprobs(&r, 1, 20).unwrap();
        assert_eq!(lp, 0.0);
        assert_eq!(top, vec![(1, 0.0)]);
        assert!(token_logprobs(&r, 0, 1).is_none(), "a masked token was not emitted by a sampler");
        assert!(token_logprobs(&r, 9, 1).is_none(), "out of the row");
        assert!(token_logprobs(&row(&[f32::NAN, f32::NAN]), 0, 1).is_none());
    }
