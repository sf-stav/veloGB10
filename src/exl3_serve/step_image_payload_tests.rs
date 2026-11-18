//! Unit tests (module `step_image_payload_tests`), moved out of the old inline block in `exl3_serve.rs`; the test text is unchanged.
//! The module is declared in `mod.rs`; `use super::*` there resolves against the parent module as before.

    //! S-B9-REL-PRPORT3 (review R9): the admit-order alignment and error paths of the
    //! PR #17 payload pre-read.
    use super::*;

    fn f16bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter()
            .flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes())
            .collect()
    }

    fn spec(vals: &[f32]) -> (u64, u64) {
        let b = f16bytes(vals);
        (b.len() as u64, crate::tp_serve::fnv64(&b))
    }

    #[test]
    fn mixed_admits_stay_aligned_in_admit_order() {
        let a = spec(&[1.5, 2.5]);
        let b = spec(&[-0.5, 4.0]);
        let specs = [a, (0, 0), b]; // image, text, image — the text Admit must not shift rows
        let mut stream = std::io::Cursor::new([f16bytes(&[1.5, 2.5]), f16bytes(&[-0.5, 4.0])].concat());
        let out = read_step_image_payloads(&mut stream, 7, &specs).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].as_ref().unwrap(), &[1.5, 2.5]);
        assert!(out[1].is_none(), "a text Admit stays aligned (None), never shifts the next image");
        assert_eq!(out[2].as_ref().unwrap(), &[-0.5, 4.0]);
    }

    #[test]
    fn short_read_errors() {
        let a = spec(&[1.0, 2.0]);
        let mut stream = std::io::Cursor::new(vec![0u8; 2]); // 2 of 4 bytes
        let err = read_step_image_payloads(&mut stream, 9, &[a]).unwrap_err();
        assert!(format!("{err:#}").contains("image payload read"), "got: {err:#}");
    }

    #[test]
    fn digest_mismatch_errors() {
        let a = spec(&[1.0, 2.0]);
        let mut stream = std::io::Cursor::new(f16bytes(&[9.0, 9.0])); // right length, wrong bytes
        let err = read_step_image_payloads(&mut stream, 9, &[a]).unwrap_err();
        assert!(format!("{err:#}").contains("digest mismatch"), "got: {err:#}");
    }
