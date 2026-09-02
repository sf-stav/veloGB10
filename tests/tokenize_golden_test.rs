// Token-identity goldens for the /v1/tokenize endpoint (and the encode path it exposes).
//
// Gate 5 of PLAN/ADD_V1_TOKENIZE_PROMPT.md: the engine's ids must match the reference
// (`transformers` / HF `tokenizers` on the SAME tokenizer.json) for the same prompt + flag.
// The expected ids below were generated with python `tokenizers` 0.22.2 against the model dirs'
// own tokenizer.json files (tests/tokenize_endpoint_test.py re-derives them at E2E time).
//
// The Hy3/GLM model cannot boot single-box (TP=2-class footprint-at-load), so its token-identity
// is pinned here at the tokenizer level; the HTTP mechanics run E2E on the 0.8b Qwen server.
use gb10_inference::tokenizer::QwenTokenizer;

fn test_model_dir(dir: &str) -> String {
    // GB10_TEST_MODEL_DIR allows pointing the goldens at a real model dir; the fallback is a
    // repo-relative path so the public tree carries no lab paths.
    std::env::var("GB10_TEST_MODEL_DIR").unwrap_or_else(|_| format!("models/{dir}"))
}

#[test]
fn qwen35_08b_golden_ids_match_reference() {
    let dir = test_model_dir("0.8b-nvfp4-mixed");
    let tok = QwenTokenizer::from_file(&format!("{dir}/tokenizer.json"))
        .expect("0.8b tokenizer loads");
    // Reference (python tokenizers 0.22.2 on the same tokenizer.json): add_special_tokens has NO
    // effect for raw text — the Qwen3.5 post-processor adds nothing, so true == false BY
    // REFERENCE (vLLM would show the same equality on this model).
    for (text, want) in [
        ("Hello world", &[9419u32, 1814] as &[u32]),
        ("The quick brown fox jumps over the lazy dog",
         &[760, 3841, 13477, 37550, 33075, 888, 279, 15217, 5388]),
        ("Hello, world! 你好 🚀",
         &[9419, 11, 1814, 0, 220, 109266, 10838, 248, 222]),
    ] {
        assert_eq!(tok.encode(text, true).expect("encode true"), want, "true: {text:?}");
        assert_eq!(tok.encode(text, false).expect("encode false"), want, "false: {text:?}");
    }
}

#[test]
fn hy3_glm_golden_ids_match_reference() {
    let tok = QwenTokenizer::from_file(&format!("{}/tokenizer.json", test_model_dir("hy3-nvfp4")))
        .expect("hy3 tokenizer loads (pair-merge upgrade)");
    // The tok_load_test oracle golden — the known GLM-family tokenization — must hold through the
    // path /v1/tokenize serves:
    assert_eq!(tok.encode("The history of the railway", true).expect("encode"),
               [628, 4043, 279, 252, 32004]);
    // Reference (python tokenizers on the same file): identical with add_special_tokens=false —
    // Hy3's post-processor adds nothing for raw text either.
    assert_eq!(tok.encode("The history of the railway", false).expect("encode"),
               [628, 4043, 279, 252, 32004]);
    assert_eq!(tok.encode("Hello world", true).expect("encode"), [16883, 2385]);
}

#[test]
fn add_special_tokens_flag_is_actually_wired() {
    // Every REAL model dir on this box ships a no-op post-processor, so true==false there and the
    // flag's wiring is invisible on them. This synthetic Llama-3-style tokenizer (TemplateProcessing
    // single="<BOS> $A <EOS>", built and reference-encoded with python `tokenizers` — see the
    // fixture dir) proves the flag reaches the underlying HF tokenizer: `true` adds the specials,
    // `false` does not, and both match the python reference EXACTLY.
    let tok = QwenTokenizer::from_file(concat!(env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/tokenize_specials/tokenizer.json")).expect("fixture tokenizer loads");
    assert_eq!(tok.vocab_size(), 70); // the bound /v1/tokenize validates id-list prompts against
    assert_eq!(tok.encode("hello world", true).expect("encode"),
               [0, 32, 31, 1]);      // <BOS> hello world <EOS>
    assert_eq!(tok.encode("hello world", false).expect("encode"),
               [32, 31]);            // raw — no specials
    assert_eq!(tok.encode("the world says hello", true).expect("encode"),
               [0, 33, 31, 67, 35, 1]);
    assert_eq!(tok.encode("the world says hello", false).expect("encode"),
               [33, 31, 67, 35]);
}

#[test]
fn detokenize_golden_matches_reference() {
    // The decode half of the pair (vLLM /v1/detokenize): whole-sequence decode must match the
    // python `tokenizers` reference byte-for-byte, including multi-byte characters split across
    // tokens (the client's exact example: [9419,11,1814,0] -> "Hello, world!").
    let tok = QwenTokenizer::from_file(&format!("{}/tokenizer.json", test_model_dir("0.8b-nvfp4-mixed")))
        .expect("0.8b tokenizer loads");
    assert_eq!(tok.decode(&[9419, 11, 1814, 0], false).expect("decode"),
               "Hello, world!");
    assert_eq!(tok.decode(&[], false).expect("decode empty"), "");
    // Multi-byte round-trip: encode -> decode is the identity on the original text.
    let text = "Hello, world! 你好 🚀";
    let ids = tok.encode(text, false).expect("encode");
    assert_eq!(tok.decode(&ids, false).expect("decode"), text);
    // Specials: default (skip=false) INCLUDES the special token's text; skip=true drops it.
    assert_eq!(tok.decode(&[248044, 9419, 1814], false).expect("decode"),
               "<|endoftext|>Hello world");
    assert_eq!(tok.decode(&[248044, 9419, 1814], true).expect("decode"),
               "Hello world");
    // The GLM family decodes through the same path.
    let hy3 = QwenTokenizer::from_file(&format!("{}/tokenizer.json", test_model_dir("hy3-nvfp4")))
        .expect("hy3 tokenizer loads");
    let hy_ids = hy3.encode("The history of the railway", true).expect("encode");
    assert_eq!(hy3.decode(&hy_ids, false).expect("decode"), "The history of the railway");
}
