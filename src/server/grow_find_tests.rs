//! Inline unit tests of the server module, moved out of `server.rs` verbatim (module `grow_find_tests`).
//! Declared from `mod.rs`; the body keeps its original indentation and its `use super::*` imports.
    // WP02: the incremental streaming search must return exactly what the O(n) `str::find` did.
    use super::GrowFind;

    #[test]
    fn grow_find_matches_str_find_on_growing_buffers() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5eed);
        let alphabet = ["<", "tool", "_call", "</think>", "<think>", "a", "é", "日本", "\n", "<tool_call>", "</thi", "nk>", "int ", "main"];
        let needles = ["<tool_call", "</think>", "<think>", "int main", "日本", "é", ""];
        for _ in 0..400 {
            let mut acc = String::new();
            let mut finders: Vec<GrowFind> = needles.iter().map(|_| GrowFind::default()).collect();
            let mut base = 0usize;
            for _ in 0..60 {
                acc.push_str(alphabet[rng.gen_range(0..alphabet.len())]);
                // a monotone base on a char boundary, as the stream cursors are
                if rng.gen_bool(0.2) {
                    let mut b = rng.gen_range(base..=acc.len());
                    while !acc.is_char_boundary(b) { b -= 1; }
                    base = b.max(base);
                }
                // hay = a prefix of acc (the reasoning branch searches acc[..safe]), never below base
                let mut cut = acc.len() - rng.gen_range(0..=acc.len().min(4));
                while !acc.is_char_boundary(cut) { cut -= 1; }
                let cut = cut.max(base);
                for (f, n) in finders.iter_mut().zip(needles.iter()) {
                    let hay = &acc[..cut];
                    let want = hay[base..].find(n).map(|i| base + i);
                    assert_eq!(f.find(hay, base, n), want, "hay={hay:?} base={base} needle={n:?}");
                }
            }
        }
    }
