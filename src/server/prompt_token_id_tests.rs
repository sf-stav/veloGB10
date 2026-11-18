//! Inline unit tests of the server module, moved out of `server.rs` verbatim (module `prompt_token_id_tests`).
//! Declared from `mod.rs`; the body keeps its original indentation and its `use super::*` imports.
    use super::prompt_token_ids;
    use serde_json::json;

    #[test]
    fn every_id_must_be_an_integer_in_the_vocab() {
        assert_eq!(prompt_token_ids(&[json!(0), json!(5), json!(99)], 100), Ok(vec![0, 5, 99]));
        for bad in [json!(-1), json!(1.5), json!(2.0), json!("7"), json!(null), json!(100), json!(4294967296u64),
                    json!(4294967297u64)] {
            let e = prompt_token_ids(&[json!(1), bad.clone(), json!(2)], 100).unwrap_err();
            assert!(e.contains("[0, 100)") && e.contains(&bad.to_string()), "{bad}: {e}");
        }
    }
