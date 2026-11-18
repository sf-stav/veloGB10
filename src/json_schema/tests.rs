//! The unit tests of the constrained-decoding module (compile errors, byte machine, masks over a byte vocabulary).

    use super::*;

    fn byte_vocab() -> Vec<(u32, Vec<u8>)> {
        (0u32..256).map(|b| (b, vec![b as u8])).collect()
    }

    fn compile_err(schema_json: &str) -> String {
        let v: Value = serde_json::from_str(schema_json).expect("test schema is valid JSON");
        let rf = serde_json::json!({
            "type": "json_schema",
            "json_schema": {"name": "t", "strict": true, "schema": v}
        });
        compile_response_format(&rf).expect_err("schema must be refused")
    }

    fn mask(schema_json: &str) -> SchemaMask {
        let v: Value = serde_json::from_str(schema_json).expect("test schema is valid JSON");
        let rf = serde_json::json!({
            "type": "json_schema",
            "json_schema": {"name": "t", "strict": true, "schema": v}
        });
        let mut m = compile_response_format(&rf)
            .unwrap_or_else(|e| panic!("schema unexpectedly refused: {e}"))
            .expect("schema must be enforced");
        m.set_vocab(byte_vocab());
        m
    }

    /// Feed a byte string as a sequence of one-byte tokens; `None` = refused.
    fn walk(m: &SchemaMask, s: &str) -> Option<u32> {
        let mut st = m.start_state();
        for &b in s.as_bytes() {
            st = m.step(st, b as u32)?;
        }
        Some(st)
    }

    /// Full-document acceptance: every byte feeds AND the document is finished.
    fn ok(m: &SchemaMask, s: &str) -> bool {
        match walk(m, s) {
            Some(st) => m.is_complete(st),
            None => false,
        }
    }

    fn bits(ids: &[u32], words: usize) -> Vec<u32> {
        let mut v = vec![0u32; words];
        for &i in ids {
            v[(i >> 5) as usize] |= 1u32 << (i & 31);
        }
        v
    }

    // ---- 0: W2 ENFORCEMENT — the escapes the 0.8B probe actually produced --------------------
    //
    // These are the exact failure shapes seen on hardware (2026-09-14, `w2_probe.py` against the
    // 0.8B trunk): a truncated \uXXXX escape, an integer far outside `maximum`, a raw control
    // character inside a string, and a digit run that never closes the document. The byte-level
    // walk isolates the MACHINE from the token plumbing, so a failure here is the FSM's own bug
    // (and a pass here means the escape came from a path that never armed the mask).

    const OBJ_SCHEMA: &str = r#"{"type":"object","properties":{
        "answer":{"type":"string","enum":["ALPHA","BETA"]},
        "n":{"type":"integer","minimum":1,"maximum":3},
        "why":{"type":"string","minLength":1,"maxLength":40}},
        "required":["answer","n","why"],"additionalProperties":false}"#;

    #[test]
    fn w2_accepts_a_valid_document() {
        let m = mask(OBJ_SCHEMA);
        assert!(ok(&m, r#"{"answer":"ALPHA","n":2,"why":"because"}"#),
                "the positive control must be accepted");
    }

    #[test]
    fn w2_refuses_the_observed_escapes() {
        let m = mask(OBJ_SCHEMA);
        let bad: &[(&str, &str)] = &[
            ("truncated escape", r#"{"answer":"ALPHA","n":2,"why":"\u043!"}"#),
            ("raw newline in string", "{\"answer\":\"ALPHA\",\"n\":2,\"why\":\"a\nb\"}"),
            ("below minimum", r#"{"answer":"ALPHA","n":-1,"why":"x"}"#),
            ("above maximum", r#"{"answer":"ALPHA","n":4,"why":"x"}"#),
            ("far above maximum", r#"{"answer":"ALPHA","n":100000000000,"why":"x"}"#),
            ("enum violation", r#"{"answer":"GAMMA","n":2,"why":"x"}"#),
            ("additionalProperties", r#"{"answer":"ALPHA","n":2,"why":"x","extra":1}"#),
            ("missing required", r#"{"answer":"ALPHA","n":2}"#),
        ];
        let mut escaped = Vec::new();
        for (what, doc) in bad {
            if ok(&m, doc) { escaped.push(*what); }
        }
        assert!(escaped.is_empty(), "these documents were ACCEPTED by the FSM: {escaped:?}");
    }

    #[test]
    fn w2_allowed_set_prunes_at_the_number_bound() {
        // The bitset is what the kernel masks with: after `{"n":3` only `,` / `}` may follow.
        let m = mask(OBJ_SCHEMA);
        let st = walk(&m, r#"{"answer":"ALPHA","n":3"#).expect("prefix must be walkable");
        let bits = m.allowed(st);
        for d in b'0'..=b'9' {
            let t = d as usize;
            assert_eq!(bits[t >> 5] & (1u32 << (t & 31)), 0,
                       "digit '{}' is allowed after the maximum", d as char);
        }
        let st0 = walk(&m, r#"{"answer":"ALPHA","n":"#).expect("prefix must be walkable");
        let b0 = m.allowed(st0);
        for c in [b'-', b'0'] {
            let t = c as usize;
            assert_eq!(b0[t >> 5] & (1u32 << (t & 31)), 0,
                       "'{}' is allowed below the minimum", c as char);
        }
    }

    #[test]
    fn w2_allowed_set_tracks_the_escape_state() {
        let m = mask(OBJ_SCHEMA);
        let st = walk(&m, r#"{"answer":"ALPHA","n":2,"why":"\u04"#).expect("prefix must be walkable");
        let bits = m.allowed(st);
        let allowed = |c: u8| { let t = c as usize; bits[t >> 5] & (1u32 << (t & 31)) != 0 };
        assert!(allowed(b'A') && allowed(b'9'), "a 4th hex digit must be allowed");
        assert!(!allowed(b'!'), "'!' must NOT be allowed while a hex digit is pending");
        assert!(!allowed(b'"'), "the string must not close inside an escape");
    }

    // ---- 1: every rejected keyword is named loudly ------------------------------------------

    #[test]
    fn rejected_keywords_are_named_loudly() {
        let cases: &[(&str, &str)] = &[
            ("pattern", r#"{"type":"string","pattern":"^a$"}"#),
            ("format", r#"{"type":"string","format":"date-time"}"#),
            ("oneOf", r#"{"oneOf":[{"type":"string"}]}"#),
            ("anyOf", r#"{"anyOf":[{"type":"string"}]}"#),
            ("allOf", r#"{"allOf":[{"type":"string"}]}"#),
            ("not", r#"{"not":{"type":"string"}}"#),
            ("$ref", r##"{"$ref":"#/defs/x"}"##),
            ("patternProperties", r#"{"type":"object","patternProperties":{"a":{"type":"string"}}}"#),
            ("minProperties", r#"{"type":"object","minProperties":1}"#),
            ("maxProperties", r#"{"type":"object","maxProperties":3}"#),
            ("$defs", r#"{"$defs":{"a":{"type":"string"}}}"#),
            ("definitions", r#"{"definitions":{"a":{"type":"string"}}}"#),
            ("if", r#"{"if":{"type":"string"}}"#),
            ("then", r#"{"then":{"type":"string"}}"#),
            ("else", r#"{"else":{"type":"string"}}"#),
            ("const", r#"{"const":1}"#),
            ("uniqueItems", r#"{"type":"array","uniqueItems":true}"#),
            ("contains", r#"{"type":"array","contains":{"type":"string"}}"#),
            ("prefixItems", r#"{"type":"array","prefixItems":[{"type":"string"}]}"#),
            ("additionalItems", r#"{"type":"array","additionalItems":false}"#),
            ("propertyNames", r#"{"type":"object","propertyNames":{"type":"string"}}"#),
            ("dependencies", r#"{"type":"object","dependencies":{"a":["b"]}}"#),
            ("dependentSchemas", r#"{"type":"object","dependentSchemas":{"a":{}}}"#),
            ("dependentRequired", r#"{"type":"object","dependentRequired":{"a":["b"]}}"#),
            ("unevaluatedProperties", r#"{"type":"object","unevaluatedProperties":false}"#),
            ("unevaluatedItems", r#"{"type":"array","unevaluatedItems":false}"#),
            // unknown keywords are refused just as loudly
            ("title", r#"{"type":"string","title":"nope"}"#),
            ("x-vendor", r#"{"type":"string","x-vendor":true}"#),
        ];
        for (kw, js) in cases {
            let e = compile_err(js);
            assert!(e.contains(&format!("'{kw}'")), "error must name '{kw}': {e}");
            assert!(e.contains("not supported"), "error for '{kw}' must say so: {e}");
        }
        // ... and the message tells the caller what to do instead.
        let e = compile_err(r#"{"type":"string","pattern":"^a$"}"#);
        assert!(e.contains("V1 supports"), "must list the supported keywords: {e}");
    }

    #[test]
    fn type_dependent_keywords_are_named_loudly() {
        // A supported keyword on the wrong type is refused by name, never ignored.
        for (kw, js) in [
            ("minLength", r#"{"minLength":3}"#),
            ("minItems", r#"{"type":"string","minItems":1}"#),
            ("minimum", r#"{"type":"string","minimum":1}"#),
            ("properties", r#"{"type":"array","properties":{"a":{"type":"string"}}}"#),
            ("enum", r#"{"type":"boolean","enum":["a"]}"#),
        ] {
            let e = compile_err(js);
            assert!(e.contains(&format!("'{kw}'")), "error must name '{kw}': {e}");
        }
        let e = compile_err(r#"{"type":"union_thing"}"#);
        assert!(e.contains("'type'"), "{e}");
        let e = compile_err(r#"{"type":["string","null"]}"#);
        assert!(e.contains("'type'"), "{e}");
    }

    // ---- 2: basic object ---------------------------------------------------------------------

    #[test]
    fn object_required_and_additional_properties() {
        let m = mask(r#"{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}"#);
        assert!(ok(&m, r#"{"a":"x"}"#));
        assert!(ok(&m, r#"{ "a" : "x" }"#)); // whitespace is free
        assert!(ok(&m, r#"{"a":"\u0078"}"#)); // escapes decode to the same string
        assert!(!ok(&m, r#"{"a":1}"#), "string property must reject a number");
        assert!(!ok(&m, r#"{}"#), "missing required key");
        assert!(!ok(&m, r#"{"a":"x","b":1}"#), "additionalProperties defaults to false");
        assert!(!ok(&m, r#"{"a":"x"}extra"#), "trailing junk after the document");
        assert!(!ok(&m, r#"{"a":"x","a":"x"}"#), "duplicate key");
        assert!(!ok(&m, r#"{"a":"x",}"#), "trailing comma in object");
        assert!(!ok(&m, r#"{"a":"x""#), "unterminated document is not complete");
        assert!(!ok(&m, r#"{"b":"x"}"#), "unknown key with additionalProperties false");
        assert!(!ok(&m, r#"[]"#), "wrong value kind");

        // explicit additionalProperties: true re-opens the key set (each at most once)
        let open = mask(r#"{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":true}"#);
        assert!(ok(&open, r#"{"a":"x","b":1}"#));
        assert!(ok(&open, r#"{"b":null,"a":"x"}"#));
        assert!(!ok(&open, r#"{"a":"x","a":"y"}"#), "duplicates are rejected even for declared keys");
        assert!(!ok(&open, r#"{"b":1,"b":2}"#), "duplicates are rejected for extra keys too");
        assert!(!ok(&open, r#"{"b":1}"#), "required still applies");

        // a bare object / json_object accepts any shape of object
        let any_obj = mask(r#"{"type":"json_object"}"#);
        assert!(ok(&any_obj, r#"{}"#));
        assert!(ok(&any_obj, r#"{"anything":[1,{"x":null}]}"#));
        assert!(!ok(&any_obj, r#"[]"#));
        assert!(ok(&mask(r#"{"type":"object"}"#), r#"{"z":1}"#));
    }

    // ---- 3: enum -----------------------------------------------------------------------------

    #[test]
    fn enum_prefix_rule() {
        let m = mask(r#"{"enum":["alpha","beta"]}"#);
        assert!(ok(&m, r#""alpha""#));
        assert!(ok(&m, r#""beta""#));
        assert!(!ok(&m, r#""gamma""#), "must die as soon as the prefix leaves the enum");
        assert!(!ok(&m, r#""alph""#), "an unterminated string is not a complete value");
        // the prefix rule is visible at the byte level: after `"` only a/b may start
        let st = walk(&m, r#"""#).unwrap();
        assert!(m.step(st, b'a' as u32).is_some());
        assert!(m.step(st, b'b' as u32).is_some());
        assert!(m.step(st, b'g' as u32).is_none());

        let typed = mask(r#"{"type":"string","enum":["a","b"]}"#);
        assert!(ok(&typed, r#""a""#));
        assert!(!ok(&typed, r#""c""#));
        assert!(!ok(&typed, r#"1"#), "declared type still applies");

        let ints = mask(r#"{"type":"integer","enum":[1,2]}"#);
        assert!(ok(&ints, "1"));
        assert!(ok(&ints, "2"));
        assert!(!ok(&ints, "3"));
        assert!(!ok(&ints, "12"));
        assert!(!ok(&ints, "1.0"), "integer forbids a fraction");
        assert!(!ok(&ints, r#""1""#));
    }

    // ---- 4: array bounds ---------------------------------------------------------------------

    #[test]
    fn array_item_bounds() {
        let m = mask(r#"{"type":"array","items":{"type":"integer"},"minItems":1,"maxItems":2}"#);
        assert!(!ok(&m, "[]"), "minItems 1");
        assert!(ok(&m, "[1]"));
        assert!(ok(&m, "[1,2]"));
        assert!(ok(&m, "[ 1 , 2 ]"));
        assert!(!ok(&m, "[1,2,3]"), "maxItems 2");
        assert!(!ok(&m, "[[1]]"), "items must be integers");
        assert!(!ok(&m, "[1,]"), "trailing comma");
        assert!(!ok(&m, "[1")); // unterminated
        let unbounded = mask(r#"{"type":"array"}"#);
        assert!(ok(&unbounded, "[]"));
        assert!(ok(&unbounded, r#"[1,"a",null,{"b":[]}]"#));
    }

    // ---- 5: nesting --------------------------------------------------------------------------

    #[test]
    fn nested_object_array_object() {
        let m = mask(
            r#"{"type":"object","properties":{"xs":{"type":"array","items":{"type":"object",
               "properties":{"n":{"type":"integer","minimum":0}},"required":["n"],
               "additionalProperties":false}}},"required":["xs"],"additionalProperties":false}"#,
        );
        assert!(ok(&m, r#"{"xs":[{"n":1},{"n":2}]}"#));
        assert!(ok(&m, r#"{"xs":[]}"#));
        assert!(!ok(&m, r#"{"xs":[{"n":-1}]}"#), "minimum applies at depth 3");
        assert!(!ok(&m, r#"{"xs":[{"n":1,"z":0}]}"#), "additionalProperties applies at depth 3");
        assert!(!ok(&m, r#"{"xs":[{"z":1}]}"#), "required applies at depth 3");
        assert!(!ok(&m, r#"{"xs":[{"n":1}]"#), "unterminated nesting");
        assert!(!ok(&m, r#"{"xs":{"n":1}}"#), "object where the array belongs");
        assert!(m.summary().contains("root=object"), "{}", m.summary());
        assert!(m.summary().contains("max_depth=4"), "{}", m.summary());
    }

    #[test]
    fn depth_cap_is_loud() {
        let deep = |levels: usize| {
            let mut v = serde_json::json!({"type":"string"});
            for _ in 0..levels {
                v = serde_json::json!({"type":"object","properties":{"x": v}});
            }
            v.to_string()
        };
        // 12 nested property levels are fine; 13 are refused by name.
        assert!(compile_response_format(&serde_json::json!({
            "type":"json_schema",
            "json_schema":{"name":"deep","schema": serde_json::from_str::<Value>(&deep(11)).unwrap()}
        }))
        .is_ok());
        let v: Value = serde_json::from_str(&deep(13)).unwrap();
        let rf = serde_json::json!({"type":"json_schema","json_schema":{"name":"deep","schema": v}});
        let e = compile_response_format(&rf).expect_err("13 levels must be refused");
        assert!(e.contains("'properties'"), "{e}");
        assert!(e.contains("deeper than the V1 limit"), "{e}");
    }

    // ---- 6: token-level layer ----------------------------------------------------------------

    #[test]
    fn tiny_vocab_allowed_and_step_agree() {
        let mut m = mask(r#"{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}"#);
        // Pieces exercise the trie: single bytes, mid-document fragments, and whole documents.
        let pieces: Vec<(u32, Vec<u8>)> = vec![
            (1, b"{".to_vec()),
            (2, b"\"a\"".to_vec()),
            (3, b"}".to_vec()),
            (4, b"1".to_vec()),
            (5, b"\"x\"".to_vec()),
            (6, b":".to_vec()),
            (7, b",".to_vec()),
            (8, b"{\"".to_vec()),
            (9, b"a\":".to_vec()),
            (10, b"\"x\"}".to_vec()),
            (11, b" ".to_vec()),
            (12, b"{\"a\":\"x\"}".to_vec()),
        ];
        m.set_vocab(pieces);
        let words = vocab_words(13);
        assert_eq!(words, 1);

        let s0 = m.start_state();
        // At the start only tokens whose whole byte run is accepted may fire: `{`, `{"` and the
        // complete document (no whitespace before the root value).
        assert_eq!(m.allowed(s0).as_slice(), bits(&[1, 8, 12], words).as_slice());
        for bad in [2u32, 3, 4, 5, 6, 7, 9, 10, 11] {
            assert_eq!(m.step(s0, bad), None, "token {bad} must not be usable at the start");
        }

        // Walk a full document with the tokenizer's pieces, asserting each use was allowed.
        let used = [1u32, 2, 6, 5, 3];
        let mut st = s0;
        let mut produced = String::new();
        for tok in used {
            let allowed = m.allowed(st);
            assert!(allowed[(tok >> 5) as usize] & (1u32 << (tok & 31)) != 0, "token {tok} was not allowed");
            st = m.step(st, tok).expect("used token must be steppable");
            produced.push_str(match tok {
                1 => "{",
                2 => "\"a\"",
                6 => ":",
                5 => "\"x\"",
                3 => "}",
                _ => unreachable!(),
            });
        }
        assert_eq!(produced, r#"{"a":"x"}"#);
        assert!(m.is_complete(st), "document finished");

        // State-specific expectations (also shows required-key handling inside `allowed`).
        let after_brace = m.step(s0, 1).unwrap();
        assert_eq!(m.allowed(after_brace).as_slice(), bits(&[2, 11], words).as_slice());
        assert_eq!(m.step(after_brace, 3), None, "cannot close before 'a' is present");

        let after_key = m.step(after_brace, 2).unwrap();
        assert_eq!(m.allowed(after_key).as_slice(), bits(&[6, 11], words).as_slice());

        let after_colon = m.step(after_key, 6).unwrap();
        assert_eq!(m.allowed(after_colon).as_slice(), bits(&[2, 5, 10, 11], words).as_slice());
        assert_eq!(m.step(after_colon, 4), None, "1 is not a string");

        let after_value = m.step(after_colon, 5).unwrap();
        // No `,` (7): the only declared key is present and additional properties are off, so a
        // comma could only lead to a key slot that nothing can fill.
        assert_eq!(m.allowed(after_value).as_slice(), bits(&[3, 11], words).as_slice());

        let done = m.step(after_value, 3).unwrap();
        assert!(m.is_complete(done));
        // Trailing whitespace keeps the machine in the same (interned) completed state.
        assert_eq!(m.step(done, 11), Some(done));
        assert_eq!(m.allowed(done).as_slice(), bits(&[11], words).as_slice());

        // End-of-generation tokens are only ever reachable through end_mask.
        let em = m.end_mask(&[3]);
        assert_eq!(em.as_slice(), bits(&[3], words).as_slice());
        assert_eq!(m.allowed(done).as_slice(), bits(&[11], words).as_slice());
    }

    #[test]
    fn unknown_and_empty_tokens_are_never_allowed() {
        let mut m = mask(r#"{"type":"integer"}"#);
        m.set_vocab(vec![(0, Vec::new()), (1, b"4".to_vec()), (2, b"2".to_vec())]);
        let s0 = m.start_state();
        let a = m.allowed(s0);
        assert_eq!(a.as_slice(), bits(&[1, 2], 1).as_slice());
        assert_eq!(m.step(s0, 0), None, "empty piece cannot advance the machine");
        assert_eq!(m.step(s0, 7), None, "token missing from the table");
        assert_eq!(m.step(s0, 9999), None);
    }

    #[test]
    fn vocab_not_set_fails_closed() {
        // A mask with no vocabulary attached must emit nothing, not everything.
        let bare = compile_response_format(&serde_json::json!({
            "type": "json_schema",
            "json_schema": {"name": "t", "schema": {"type": "string"}}
        }))
        .unwrap()
        .expect("schema is enforced");
        assert!(!bare.is_dead(bare.start_state()), "the machine itself is alive");
        assert!(bare.allowed(bare.start_state()).is_empty(), "no vocab: nothing may be emitted");
        // ... and an explicitly empty vocabulary behaves the same way.
        let mut empty = mask(r#"{"type":"string"}"#);
        empty.set_vocab(Vec::new());
        assert!(empty.allowed(empty.start_state()).is_empty());
    }

    // ---- 7: completeness ---------------------------------------------------------------------

    #[test]
    fn complete_state_tracks_document_end() {
        let m = mask(r#"{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}"#);
        let s0 = m.start_state();
        assert!(!m.is_complete(s0), "empty output is not a document");
        for prefix in [r#"{"#, r#"{"a"#, r#"{"a":"#, "{\"a\":1"] {
            let st = walk(&m, prefix).unwrap_or_else(|| panic!("prefix {prefix} should be steppable"));
            assert!(!m.is_complete(st), "{prefix} is not finished");
        }
        let mid = walk(&m, r#"{"a":1"#).unwrap();
        assert!(!m.is_complete(mid), "the object is still open");
        // A pending number IS finished at the top level: `is_complete` finalizes it.
        let nums = mask(r#"{"type":"integer","maximum":10}"#);
        assert!(nums.is_complete(walk(&nums, "7").unwrap()));
        assert!(walk(&nums, "70").is_none(),
                "an integer prefix past `maximum` can never return to range — refused mid-prefix");
        // The same bound on a FLOAT is only decidable once the number ends: `70e-1` is 7, so the
        // prefix stays allowed and the refusal lands at completion.
        let floats = mask(r#"{"type":"number","maximum":10}"#);
        assert!(!floats.is_complete(walk(&floats, "70").unwrap()), "over the bound");
        let finished = walk(&m, r#"{"a":1}"#).unwrap();
        assert!(m.is_complete(finished));
        assert!(m.is_complete(walk(&m, "{\"a\":1} \n\t").unwrap()), "trailing whitespace");
        assert!(walk(&m, r#"{"a":1} x"#).is_none(), "junk is refused outright");
        assert!(walk(&m, r#"{"a":false}"#).is_none(), "wrong value kind is refused");
        assert!(!m.is_dead(finished), "a finished document is alive");
        assert!(m.is_dead(12345), "unknown states report dead (fail closed)");
    }

    // ---- 8: cache ----------------------------------------------------------------------------

    #[test]
    fn allowed_is_cached_and_idempotent() {
        let m = mask(r#"{"type":"integer","minimum":5,"maximum":10}"#);
        let s0 = m.start_state();
        let a = m.allowed(s0);
        let b = m.allowed(s0);
        assert_eq!(a, b, "same bitset");
        assert!(Arc::ptr_eq(&a, &b), "second call must be the cached Arc");
        let before = m.summary();
        assert!(before.contains("cached_masks=1"), "{before}");
        assert!(before.contains("vocab=256 tokens/8 words"), "{before}");
        let _ = m.allowed(s0);
        assert!(m.summary().contains("cached_masks=1"), "no growth on repeat");
    }

    // ---- additional coverage: numbers, strings, response_format shapes ------------------------

    #[test]
    fn number_grammar_and_bounds() {
        let m = mask(r#"{"type":"integer","minimum":5,"maximum":10}"#);
        assert!(ok(&m, "5"));
        assert!(ok(&m, "10"));
        assert!(!ok(&m, "4"));
        assert!(!ok(&m, "11"));
        assert!(!ok(&m, "05"), "leading zero");
        assert!(!ok(&m, "-5"));
        assert!(!ok(&m, "5.0"), "integer forbids fraction");
        assert!(!ok(&m, "1e1"), "integer forbids exponent");
        assert!(!ok(&m, "5x"));
        assert!(!ok(&m, "-"));

        let n = mask(r#"{"type":"number","exclusiveMinimum":0,"exclusiveMaximum":1}"#);
        assert!(ok(&n, "0.5"));
        assert!(ok(&n, "1e-3"));
        assert!(!ok(&n, "0"), "exclusive bound");
        assert!(!ok(&n, "1"), "exclusive bound");
        assert!(!ok(&n, "-1"));
        assert!(!ok(&n, "1."), "dangling fraction point");
        assert!(!ok(&n, "1e"), "dangling exponent");

        let e = mask(r#"{"type":"number","maximum":2,"exclusiveMaximum":2}"#);
        assert!(!ok(&e, "2"));
    }

    #[test]
    fn string_lengths_are_code_points() {
        let m = mask(r#"{"type":"string","minLength":2,"maxLength":3}"#);
        assert!(ok(&m, r#""ab""#));
        assert!(ok(&m, r#""abc""#));
        assert!(!ok(&m, r#""a""#));
        assert!(!ok(&m, r#""abcd""#));
        assert!(ok(&m, r#""\u0041B""#), "an escape is one code point");
        assert!(ok(&m, r#""\ud83d\ude00x""#), "a surrogate pair is one code point");
        assert!(!ok(&m, r#""\ud83d\ude00""#), "one code point < minLength 2");
        let one = mask(r#"{"type":"string","maxLength":1}"#);
        assert!(ok(&one, "\"\u{1F600}\""), "4 UTF-8 bytes, 1 code point");
        assert!(!ok(&one, r#""ab""#));
        assert!(!ok(&m, "\"a\nb\""), "raw newline must be escaped");
        assert!(ok(&m, r#""a\nb""#), "escaped newline is fine");
        assert!(!ok(&m, r#""\ud83d""#), "lone high surrogate");
        assert!(!ok(&m, r#""\q""#), "unknown escape");
    }

    #[test]
    fn response_format_shapes() {
        // nothing to enforce
        assert!(compile_response_format(&Value::Null).unwrap().is_none());
        assert!(compile_response_format(&serde_json::json!({"type":"text"})).unwrap().is_none());
        // json_object = any object
        let m = compile_response_format(&serde_json::json!({"type":"json_object"}))
            .unwrap()
            .expect("json_object is enforced");
        assert!(m.summary().contains("root=object"), "{}", m.summary());
        // bare form: {"type":"json_schema","schema":{...}}
        let bare = compile_response_format(
            &serde_json::json!({"type":"json_schema","schema":{"type":"string"}}),
        )
        .unwrap()
        .expect("bare form is enforced");
        assert!(bare.summary().contains("root=string"), "{}", bare.summary());
        // strict/name/description are accepted and ignored
        assert!(compile_response_format(&serde_json::json!({
            "type":"json_schema",
            "json_schema":{"name":"n","description":"d","strict":true,"schema":{"type":"null"}}
        }))
        .is_ok());
        // loud failures
        let e = compile_response_format(&serde_json::json!({
            "type":"json_schema","json_schema":{"name":"n"}
        }))
        .expect_err("missing schema");
        assert!(e.contains("'schema'"), "{e}");
        let e = compile_response_format(&serde_json::json!({"type":"yaml"})).expect_err("bad type");
        assert!(e.contains("'type'"), "{e}");
        let e = compile_response_format(&serde_json::json!({"nope":1})).expect_err("no type");
        assert!(e.contains("'type'"), "{e}");
        assert!(compile_response_format(&serde_json::json!([1,2])).is_err());
        assert!(compile_response_format(&serde_json::json!({"type":"json_schema","json_schema":{"name":"n","schema":{"type":"string"}},"extra":1})).is_err());
    }

    #[test]
    fn permissive_and_literal_values() {
        let m = mask("{}");
        assert!(ok(&m, "123"));
        assert!(ok(&m, "-1.5e3"));
        assert!(ok(&m, r#""s""#));
        assert!(ok(&m, "true"));
        assert!(ok(&m, "null"));
        assert!(ok(&m, r#"[1,"a"]"#));
        assert!(!ok(&m, "tru"));
        assert!(!ok(&m, "[1,]"));
        assert!(ok(&mask(r#"{"type":"boolean"}"#), "false"));
        assert!(!ok(&mask(r#"{"type":"boolean"}"#), "truex"));
        assert!(ok(&mask(r#"{"type":"null"}"#), "null"));
        assert!(!ok(&mask(r#"{"type":"null"}"#), "nul"));
    }

    // ---- dead-end freedom: every reachable state allows a token or is complete -----------------

    /// Random walks over the allowed sets (single-byte vocabulary). A state that accepts no byte
    /// and is not a complete document is a dead end: under the token mask the sampler would have
    /// nothing to pick. Every allowed byte must also step.
    fn no_dead_ends(schema: &str, walks: usize, max_len: usize) {
        let m = mask(schema);
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        for w in 0..walks {
            let mut st = m.start_state();
            let mut doc = Vec::new();
            for _ in 0..max_len {
                if m.is_complete(st) { break; }
                let bits = m.allowed(st);
                let ids: Vec<u32> = (0..256u32).filter(|&t| bits[(t >> 5) as usize] >> (t & 31) & 1 == 1).collect();
                assert!(!ids.is_empty(), "walk {w}: dead end after {:?}", String::from_utf8_lossy(&doc));
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let t = ids[((rng >> 33) as usize) % ids.len()];
                st = m.step(st, t).unwrap_or_else(|| panic!("walk {w}: allowed byte {t} refused by step"));
                doc.push(t as u8);
            }
        }
    }

    #[test]
    fn no_dead_end_at_max_length() {
        // At maxLength a started escape or UTF-8 sequence could never be finished.
        no_dead_ends(r#"{"type":"object","properties":{"t":{"type":"string","maxLength":3}},"required":["t"]}"#, 400, 80);
        no_dead_ends(r#"{"type":"array","items":{"type":"string","minLength":1,"maxLength":1},"minItems":2,"maxItems":2}"#, 400, 80);
        let m = mask(r#"{"type":"object","properties":{"t":{"type":"string","maxLength":2}}}"#);
        assert!(walk(&m, r#"{"t":"a\u00"#).is_some(), "an escape may start while there is room for its character");
        assert!(walk(&m, r#"{"t":"ab\"#).is_none(), "no escape at maxLength");
        let at_max = walk(&m, r#"{"t":"ab"#).unwrap();
        assert!(m.step(at_max, 0xC3).is_none(), "no UTF-8 lead byte at maxLength");
        let closed = m.step(at_max, b'"' as u32).unwrap();
        assert!(m.step(closed, b'}' as u32).is_some());
    }

    #[test]
    fn enum_candidates_respect_length_bounds() {
        let m = mask(r#"{"type":"string","enum":["a","bbb"],"minLength":3}"#);
        assert!(walk(&m, r#""a"#).is_none(), "a value shorter than minLength is not a candidate");
        assert!(ok(&m, r#""bbb""#));
        let m = mask(r#"{"type":"string","enum":["abc","de"],"maxLength":2}"#);
        assert!(walk(&m, r#""a"#).is_none(), "a value longer than maxLength is not a candidate");
        assert!(ok(&m, r#""de""#));
        assert!(compile_err(r#"{"type":"string","enum":["a"],"minLength":2}"#).contains("enum"));
        let m = mask(r#"{"type":"integer","enum":[1,20],"maximum":10}"#);
        assert!(walk(&m, "2").is_none(), "an integer outside the bounds is not a candidate");
        assert!(ok(&m, "1"));
    }

    #[test]
    fn integer_prefixes_must_reach_a_valid_value() {
        let m = mask(r#"{"type":"integer"}"#);
        assert!(walk(&m, "1.").is_none(), "no fraction for an integer");
        assert!(walk(&m, "1e").is_none(), "no exponent for an integer");
        assert!(ok(&m, "-12"));
        let m = mask(r#"{"type":"integer","minimum":5,"maximum":10}"#);
        assert!(walk(&m, "2").is_none(), "2, 20.. are all outside 5..10");
        assert!(walk(&m, "1").is_some(), "1 can still become 10");
        assert!(walk(&m, "11").is_none());
        assert!(walk(&m, "-").is_none());
        assert!(ok(&m, "5") && ok(&m, "10") && !ok(&m, "1"));
        let m = mask(r#"{"type":"integer","minimum":-10,"maximum":-5}"#);
        assert!(walk(&m, "-1").is_some() && walk(&m, "-2").is_none() && walk(&m, "0").is_none());
        assert!(ok(&m, "-5") && ok(&m, "-10"));
        let m = mask(r#"{"type":"integer","exclusiveMinimum":4.5,"exclusiveMaximum":7}"#);
        assert!(walk(&m, "4").is_none() && walk(&m, "7").is_none());
        assert!(ok(&m, "5") && ok(&m, "6"));
        let m = mask(r#"{"type":"integer","enum":[1,2]}"#);
        assert!(walk(&m, "3").is_none() && walk(&m, "10").is_none() && walk(&m, "1.").is_none());
        assert!(ok(&m, "1") && ok(&m, "2"));
        let m = mask(r#"{"type":"number","minimum":0}"#);
        assert!(walk(&m, "-1").is_none(), "no negative number when none is allowed");
        assert!(ok(&m, "-0") && ok(&m, "2.5"));
        let m = mask(r#"{"type":"number","maximum":-1}"#);
        assert!(walk(&m, "5").is_none() && walk(&m, "0").is_none() && walk(&m, "-0").is_none());
        assert!(ok(&m, "-1.5"));
        no_dead_ends(r#"{"type":"object","properties":{"a":{"type":"integer","minimum":5,"maximum":10},"b":{"type":"integer","enum":[1,2,30]}},"required":["a","b"]}"#, 400, 80);
    }

    #[test]
    fn untyped_enum_admits_only_its_values() {
        let m = mask(r#"{"enum":["alpha","beta"]}"#);
        assert!(walk(&m, "{").is_none() && walk(&m, "[").is_none());
        assert!(ok(&m, r#""alpha""#));
        let m = mask(r#"{"type":"object","properties":{"k":{"enum":["x",3]}},"required":["k"]}"#);
        assert!(walk(&m, r#"{"k":{"#).is_none() && walk(&m, r#"{"k":["#).is_none());
        assert!(ok(&m, r#"{"k":"x"}"#) && ok(&m, r#"{"k":3}"#));
        assert!(walk(&m, r#"{"k":4"#).is_none());
    }

    #[test]
    fn no_lone_surrogate_escape_prefix() {
        let m = mask(r#"{"type":"string"}"#);
        assert!(walk(&m, r#""\udc"#).is_none(), "a low surrogate cannot start a pair");
        assert!(walk(&m, r#""\ud834\u"#).is_some());
        assert!(walk(&m, r#""\ud834\ue"#).is_none(), "a pair needs a low surrogate second");
        assert!(walk(&m, r#""\ud834\ud8"#).is_none());
        assert!(ok(&m, r#""\ud834\udd1e""#));
        no_dead_ends(r#"{"type":"object","properties":{"t":{"type":"string"}},"required":["t"]}"#, 2000, 60);
    }

    #[test]
    fn no_dead_end_enum_with_escapes_and_spaces() {
        no_dead_ends(r#"{"type":"object","properties":{"color":{"type":"string","enum":["red","green","light blue","naïve","𝄞 clef"]}},"required":["color"]}"#, 400, 200);
    }

    #[test]
    fn no_dead_end_object_keys_and_arrays() {
        no_dead_ends(r#"{"type":"object","properties":{"a":{"type":"integer","minimum":0,"maximum":9},"ab":{"type":"boolean"},"list":{"type":"array","items":{"type":"string","enum":["x","y"]},"maxItems":2}},"required":["a"]}"#, 400, 200);
    }

    #[test]
    fn enum_values_and_keys_use_the_canonical_spelling() {
        let m = mask(r#"{"type":"string","enum":["no","not x","𝄞","say \"hi\"\n"]}"#);
        assert!(ok(&m, r#""not x""#));
        assert!(ok(&m, r#""𝄞""#), "non-ASCII is written raw");
        assert!(ok(&m, r#""say \"hi\"\n""#), "quote and newline use their short escapes");
        assert!(walk(&m, r#""no\u0074"#).is_none(), "an escaped spelling of a plain character is refused");
        assert!(walk(&m, r#""\ud834"#).is_none(), "so is a surrogate escape of a raw-written character");
        assert!(walk(&m, "\"no\u{e9}").is_none(), "a raw multi-byte char no candidate continues with");
        assert!(walk(&m, r#""no\"#).is_none(), "a backslash no candidate needs");
        let k = mask(r#"{"type":"object","properties":{"ab":{"type":"integer"}},"required":["ab"]}"#);
        assert!(ok(&k, r#"{"ab":1}"#));
        assert!(walk(&k, r#"{"\u0061"#).is_none(), "declared keys too");
        assert_eq!(canon_match("a\u{1}b", b"a\\u0001b"), (true, true));
        assert_eq!(canon_match("ab", b"a"), (true, false));
        assert_eq!(canon_match("ab", b"abc"), (false, false));
    }

    #[test]
    fn mask_key_shares_free_string_states() {
        let m = mask(r#"{"type":"object","properties":{"t":{"type":"string"}}}"#);
        let a = walk(&m, r#"{"t":"a"#).unwrap();
        let b = walk(&m, r#"{"t":"abcdef"#).unwrap();
        assert_ne!(a, b, "distinct machine states");
        assert!(Arc::ptr_eq(&m.allowed(a), &m.allowed(b)), "one cached mask for both");
    }

    #[test]
    fn mask_key_is_exact_near_length_bounds() {
        // multi-byte pieces, so a token can add several characters at once
        let mut pieces = byte_vocab();
        let n = pieces.len() as u32;
        for (i, p) in ["ab", "abc", "abcd\"", "\"}", "xyz"].iter().enumerate() {
            pieces.push((n + i as u32, p.as_bytes().to_vec()));
        }
        for schema in [r#"{"type":"object","properties":{"t":{"type":"string","minLength":3,"maxLength":7}}}"#,
                       r#"{"type":"object","properties":{"t":{"type":"string","maxLength":4}}}"#,
                       r#"{"type":"object","properties":{"t":{"type":"string","minLength":5}}}"#] {
            let mut m = mask(schema);
            m.set_vocab(pieces.clone());
            for len in 0..9 {
                let Some(st) = walk(&m, &format!(r#"{{"t":"{}"#, "q".repeat(len))) else { continue };
                assert_eq!(*m.allowed(st), m.compute_allowed(st), "{schema}: {len} chars");
            }
        }
    }

    /// Multi-byte pieces with escapes, UTF-8, closing quotes and structure after the close.
    fn mixed_vocab() -> Vec<(u32, Vec<u8>)> {
        let mut pieces = byte_vocab();
        let n = pieces.len() as u32;
        let extra: [&[u8]; 14] = [b"ab", b"abc", b"abcd\"", b"\"}", b"xyz", b"\\n", b"q\\t",
            "\u{e9}".as_bytes(), "a\u{e9}b\"".as_bytes(), b"\\u00e9", b"\\u00", b"e9\"}",
            b"\", \"", &[0xC3]];
        for (i, p) in extra.iter().enumerate() {
            pieces.push((n + i as u32, p.to_vec()));
        }
        pieces
    }

    #[test]
    fn length_bounded_masks_match_full_walk() {
        // Every state reached by a random walk over multi-byte pieces: the derived mask near the
        // length bounds must equal a full trie walk (including mid-escape / mid-UTF-8 states,
        // which take the full walk).
        for schema in [r#"{"type":"object","properties":{"t":{"type":"string","minLength":3,"maxLength":7}}}"#,
                       r#"{"type":"object","properties":{"t":{"type":"string","maxLength":4},"u":{"type":"string","maxLength":2}}}"#,
                       r#"{"type":"object","properties":{"t":{"type":"string","minLength":5}}}"#,
                       r#"{"type":"array","items":{"type":"string","maxLength":3},"maxItems":4}"#] {
            let mut m = mask(schema);
            m.set_vocab(mixed_vocab());
            let mut rng = 0x2545_F491_4F6C_DD1Du64;
            let mut checked = 0;
            for _ in 0..200 {
                let mut st = m.start_state();
                for _ in 0..24 {
                    if m.is_complete(st) { break; }
                    let bits = m.allowed(st);
                    assert_eq!(*bits, m.compute_allowed(st), "{schema}");
                    checked += 1;
                    let ids: Vec<u32> = (0..m.vocab_size as u32)
                        .filter(|&t| bits[(t >> 5) as usize] >> (t & 31) & 1 == 1).collect();
                    if ids.is_empty() { break; }
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    st = m.step(st, ids[((rng >> 33) as usize) % ids.len()]).expect("allowed token refused");
                }
            }
            assert!(checked > 200, "{schema}: only {checked} states checked");
        }
    }

    #[test]
    fn str_run_counts_characters_not_bytes() {
        assert_eq!(str_run_of(b"abc"), 3);
        assert_eq!(str_run_of(b"ab\"}"), 2 | STR_RUN_CLOSES);
        assert_eq!(str_run_of(b"\\n"), 1);
        assert_eq!(str_run_of(b"\\u00e9x"), 2);
        assert_eq!(str_run_of(b"\\u00"), 1); // escape still open: its character is committed
        assert_eq!(str_run_of("\u{e9}\u{e9}".as_bytes()), 2);
        assert_eq!(str_run_of(&[0xC3]), 1); // UTF-8 sequence still open: same
        assert_eq!(str_run_of(&[0x01]), STR_RUN_BAD);
    }

    #[test]
    fn structural_whitespace_is_capped() {
        let m = mask(r#"{"type":"object","properties":{"a":{"type":"string"}}}"#);
        let lead = " ".repeat(MAX_WS_RUN as usize);
        assert!(walk(&m, " ").is_none(), "no whitespace before the root value");
        assert!(ok(&m, &format!("{{{lead}\"a\":\"x\"}}")), "a run up to the cap is fine");
        assert!(walk(&m, &format!("{{{lead} ")).is_none(), "one more is refused");
        assert!(ok(&m, &format!("{{\"a\":\"{}\"}}", " ".repeat(100))), "string content is not structural");
        assert!(ok(&m, &format!("{{{lead}\"a\"{lead}:{lead}\"x\"{lead}}}")), "the run resets at every token");
    }

    #[test]
    fn no_key_or_item_slot_that_cannot_be_filled() {
        let m = mask(r#"{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}}}"#);
        assert!(walk(&m, r#"{"a":1,"a"#).is_none(), "a duplicate declared key cannot start");
        assert!(ok(&m, r#"{"a":1,"b":2}"#));
        assert!(walk(&m, r#"{"a":1,"b":2,"#).is_none(), "no key left: the comma is refused");
        let arr = mask(r#"{"type":"array","items":{"type":"integer"},"maxItems":2}"#);
        assert!(ok(&arr, "[1,2]"));
        assert!(walk(&arr, "[1,2,").is_none(), "maxItems reached: the comma is refused");
        let none = mask(r#"{"type":"object","properties":{},"additionalProperties":false}"#);
        assert!(ok(&none, "{}"));
        assert!(walk(&none, "{\"").is_none(), "no key can ever be accepted");
    }
