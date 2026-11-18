//! Inline unit tests of the server module, moved out of `server.rs` verbatim (module `model_id_tests`).
//! Declared from `mod.rs`; the body keeps its original indentation and its `use super::*` imports.
    use super::{resolve_model_name, model_id_from_dir};

    fn temp_model_dir(tag: &str, readme: Option<&str>) -> String {
        let d = std::env::temp_dir().join(format!("velogb10-mid-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        if let Some(r) = readme { std::fs::write(d.join("README.md"), r).unwrap(); }
        d.to_string_lossy().to_string()
    }

    /// 2026-10-20: a quantized pack names ITSELF with `model_name:`; `base_model:` is the original
    /// model and only serves as the id when the card has no `model_name:` line.
    #[test]
    fn model_id_prefers_the_packs_own_name() {
        let card = "---\nlicense: mit\nmodel_name: doth4580/Naive-N0.5-Flash-EXL3-3.5bpw\nbase_model: NaiveAI/Naive-N0.5-Flash\nbase_model_relation: quantized\n---\ncard\n";
        let dir = temp_model_dir("own", Some(card));
        assert_eq!(model_id_from_dir(&dir), "doth4580/Naive-N0.5-Flash-EXL3-3.5bpw");
        let card = "---\nbase_model: NaiveAI/Naive-N0.5-Flash\nmodel_name: \"doth4580/Naive-N0.5-Flash-EXL3-4bpw\"\n---\n";
        let dir2 = temp_model_dir("own2", Some(card));
        assert_eq!(model_id_from_dir(&dir2), "doth4580/Naive-N0.5-Flash-EXL3-4bpw", "order in the card and quotes do not matter");
        let card = "---\nmodel_name:   \nbase_model: NaiveAI/Naive-N0.5-Flash\n---\n";
        let dir3 = temp_model_dir("own3", Some(card));
        assert_eq!(model_id_from_dir(&dir3), "NaiveAI/Naive-N0.5-Flash", "a blank model_name: falls through to base_model:");
        for d in [dir, dir2, dir3] { let _ = std::fs::remove_dir_all(&d); }
    }

    /// v0.7.3 (#8.3): the public model id resolves as flag > alias > model_name > base_model > dir name.
    #[test]
    fn model_id_resolution_order() {
        let card = "---\nbase_model: Qwen/Qwen3.8-27B\n---\nsome card\n";
        let dir = temp_model_dir("order", Some(card));
        assert_eq!(resolve_model_name(Some("flag-id"), Some("alias-id"), &dir), "flag-id");
        assert_eq!(resolve_model_name(None, Some("alias-id"), &dir), "alias-id");
        assert_eq!(resolve_model_name(Some("  flag-id  "), None, &dir), "flag-id", "verbatim + trimmed");
        assert_eq!(resolve_model_name(Some(""), Some("alias-id"), &dir), "alias-id",
                   "an empty flag value falls through to the alias");
        assert_eq!(resolve_model_name(None, None, &dir), "Qwen/Qwen3.8-27B",
                   "no flag: the model card base_model: line");
        let bare = temp_model_dir("bare", None);
        let dirname = std::path::Path::new(&bare).file_name().unwrap().to_string_lossy().to_string();
        assert_eq!(resolve_model_name(None, None, &bare), dirname, "no card: the directory name");
        assert_eq!(model_id_from_dir(&bare), dirname);
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&bare);
    }
