//! Unit tests for the xchain2 dump format (former inline test module `xchain2_tests` of `gpu.rs`; file map in `mod.rs`).

    use super::*;

    fn sample_dump() -> XChain2Dump {
        XChain2Dump {
            plen: 5, max_new: 3, nlayers: 2, h: 4, vocab: 8, a0: 7,
            hcol: vec![0.5, -1.0, 2.0, 3.5],
            tokens: vec![7, 8, 9],
            step1_outputs: vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]],
            step_last_outputs: vec![vec![9.0; 4], vec![10.0; 4]],
            final_logits: vec![0.1, -0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        }
    }

    #[test]
    fn xchain2_roundtrip() {
        let d = sample_dump();
        let tmp = std::env::temp_dir().join("xchain2_roundtrip.bin");
        let p = tmp.to_str().unwrap();
        xchain2_write(p, &d).unwrap();
        let r = xchain2_read(p).unwrap();
        std::fs::remove_file(p).ok();
        assert_eq!(r.plen, 5);
        assert_eq!(r.max_new, 3);
        assert_eq!(r.nlayers, 2);
        assert_eq!(r.h, 4);
        assert_eq!(r.vocab, 8);
        assert_eq!(r.a0, 7);
        assert_eq!(r.tokens, vec![7, 8, 9]);
        assert_eq!(r.hcol, d.hcol);
        assert_eq!(r.step1_outputs, d.step1_outputs);
        assert_eq!(r.step_last_outputs, d.step_last_outputs);
        assert_eq!(r.final_logits, d.final_logits);
    }

    #[test]
    fn xchain2_rejects_bad_input() {
        assert!(xchain2_read("/nonexistent/xc2.bin").is_err());
        let tmp = std::env::temp_dir().join("xchain2_bad.bin");
        std::fs::write(&tmp, b"NOTXC2Dxxxx").unwrap();
        assert!(xchain2_read(tmp.to_str().unwrap()).is_err());
        // Truncated header: magic + 2 u32s only.
        let tmp2 = std::env::temp_dir().join("xchain2_trunc.bin");
        let mut b = b"XC2D".to_vec();
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&7u32.to_le_bytes());
        std::fs::write(&tmp2, &b).unwrap();
        assert!(xchain2_read(tmp2.to_str().unwrap()).is_err());
        std::fs::remove_file(&tmp).ok();
        std::fs::remove_file(&tmp2).ok();
    }
