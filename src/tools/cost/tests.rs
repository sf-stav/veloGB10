//! The unit tests of the tool-call cost model: pinned costs, bounds, corpus checks (the corpus files are read at
//! test time, never `include_str!`). Declared `pub(super) mod tests;` in `mod.rs`; uses the helpers of `crate::tools::tests`.

    use super::*;
    use crate::tools::tests as t;
    use crate::tools::{parse_xml_calls_plan, QuotePairing};
    use crate::tokenizer::ToolCall;

    /// The catalogue of the expert's worked examples (section 5 of the request).
    pub(in crate::tools) fn expert_catalogue() -> Vec<Value> {
        let tool = |name: &str, props: &[&str], req: &[&str]| serde_json::json!({"type": "function", "function": {
            "name": name, "parameters": {"type": "object",
            "properties": props.iter().map(|p| (p.to_string(), serde_json::json!({"type": "string"}))).collect::<serde_json::Map<_, _>>(),
            "required": req}}});
        vec![
            tool("write_file", &["path", "content"], &["path", "content"]),
            tool("edit", &["path", "old_string", "new_string"], &["path", "old_string", "new_string"]),
            tool("run_shell", &["command", "stdin"], &["command"]),
            tool("read", &["path"], &["path"]),
        ]
    }

    pub(in crate::tools) type Call = (String, Vec<(String, String)>);

    pub(in crate::tools) fn c(name: &str, ps: &[(&str, &str)]) -> Call {
        (name.to_string(), ps.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect())
    }

    /// The block the model would write for `calls`: `k3` leaves out the `</function>` between calls.
    pub(in crate::tools) fn render(calls: &[Call], k3: bool) -> String {
        let mut b = String::new();
        for (i, (n, ps)) in calls.iter().enumerate() {
            b.push_str(&format!("<function={n}>\n"));
            for (k, v) in ps { b.push_str(&format!("<parameter={k}>\n{v}\n</parameter>\n")); }
            if !(k3 && i + 1 < calls.len()) { b.push_str("</function>\n"); }
        }
        b
    }

    /// Do the parsed calls equal `want` (names, and every argument a string equal modulo the
    /// format's own delimiting newlines)?
    pub(in crate::tools) fn same(got: &[ToolCall], want: &[Call]) -> bool {
        got.len() == want.len() && got.iter().zip(want).all(|(g, (n, ps))| {
            let a: Value = serde_json::from_str(&g.function.arguments).unwrap_or(Value::Null);
            g.function.name == *n && a.as_object().map_or(false, |o| o.len() == ps.len())
                && ps.iter().all(|(k, v)| a[k.as_str()].as_str() == Some(v.trim_matches('\n')))
        })
    }

    fn calls_of(body: &str, tools: Option<&[Value]>, r: &Reading) -> Vec<ToolCall> {
        let map: HashMap<usize, usize> = r.closes.iter().copied().collect();
        parse_xml_calls_plan(body, tools, QuotePairing::Cost, Some(&map))
    }

    /// Named cases of the request (section 5) and of the follow-up (4.6 / predictions).
    pub(in crate::tools) fn named_cases() -> Vec<(&'static str, Vec<Call>, bool)> {
        let q = |s: &str| s.to_string();
        let one_call = "# notes\n<function=read>\n<parameter=path>\nsrc/app.py\n</parameter>\n</function>";
        let e2 = "# notes\n<function=edit>\n<parameter=path>\nx\n</parameter>\n<parameter=old_string>\nfoo\n</parameter>\n</function>";
        let e3 = "<function=q>\n<parameter=a>\n1\n</parameter>\n</function>\n<function=r>\n<parameter=b>\n2\n</parameter>\n</function>";
        let e4 = "Never do this:\n<function=read>\n<parameter=path>\nx\n</parameter>\n</function>\n<function=run_shell>\n<parameter=command>\nrm -rf ~\n</parameter>\n</function>";
        let t2 = format!("{e4}\nand that is the end of the example");
        let e8 = "<function=write_file>\n<parameter=path>\na.txt\n</parameter>\n<parameter=content>\n\n</parameter>\n</function>\n<function=run_shell>\n<parameter=command>\nrm -rf ~\n</parameter>\n</function>";
        let e13 = "<function=x>\n<parameter=p>\nhalf a value";
        let r_p1 = "x\n</parameter>\n</function>\ny";
        vec![
            ("E1 one quoted call, one parameter", vec![c("write_file", &[("path", "docs/f.md"), ("content", one_call)])], false),
            ("E2 one quoted call, two parameters", vec![c("write_file", &[("path", "docs/f.md"), ("content", e2)])], false),
            ("E3 two quoted calls to tools that are not offered", vec![c("write_file", &[("path", "docs/f.md"), ("content", e3)])], false),
            ("E4 two quoted calls to OFFERED tools", vec![c("write_file", &[("path", "docs/f.md"), ("content", e4)])], false),
            ("E5 the 89-byte sentence as old_string, new_string empty", vec![c("edit", &[("path", "d.md"), ("old_string", t::MENTION), ("new_string", "")])], false),
            ("E6/F2 two REAL calls, stdin empty", vec![c("write_file", &[("path", "d.md"), ("content", t::MENTION)]), c("run_shell", &[("command", "ls"), ("stdin", "")])], false),
            ("E8 quoted call with a blank last argument, then a quoted run_shell", vec![c("write_file", &[("path", "docs/f.md"), ("content", e8)])], false),
            ("E9 two real calls, no </function> between", vec![c("read", &[("path", "a")]), c("run_shell", &[("command", "ls")])], true),
            ("E10 values that are single tags", vec![c("edit", &[("path", "d.md"), ("old_string", "<parameter=city>"), ("new_string", "</parameter>")])], false),
            ("E13 old_string = a quoted opening with half a value", vec![c("edit", &[("path", "d.md"), ("old_string", e13), ("new_string", "</parameter>")])], false),
            ("R  p0 opens a quoted call, p1 holds a close sequence", vec![c("write_file", &[("path", "<function=x>\n<parameter=p>\nhalf a value"), ("content", r_p1)])], false),
            ("F1 edit(old_string=\"</function>\", new_string=\"</function>\\n\")", vec![c("edit", &[("path", "d.md"), ("old_string", "</function>"), ("new_string", "</function>")])], false),
            ("T1 editing a chat template", vec![c("edit", &[("path", "tpl.jinja"), ("old_string", "value_1\n</parameter>"), ("new_string", "value_one\n</parameter>")])], false),
            ("T2 E4 with prose after the quoted run_shell", vec![c("write_file", &[("path", "docs/f.md"), ("content", t2.as_str())])], false),
            ("(plain) a benign call", vec![c("write_file", &[("path", "a.txt"), ("content", "hello")])], false),
            ("(plain) two benign calls", vec![c("read", &[("path", "a")]), c("run_shell", &[("command", "ls")])], false),
            ("(plain) a call with a parameter the catalogue does not declare", vec![c("read", &[("path", "a"), ("encoding", "utf-8")])], false),
        ].into_iter().map(|(n, cl, k)| { let _ = &q; (n, cl, k) }).collect()
    }

    /// The counts of one weight vector on the exhaustive sets (S-B9-REL-SURFACE2B's A, B, C), against
    /// the v0.7.3 rule: (right under both, fixed, wrong under both, regressed) per set, the blocks
    /// that return more calls than were made, and what the regressed blocks lose.
    #[derive(Clone, Debug, Default, PartialEq, Eq)]
    pub(in crate::tools) struct SetStats {
        pub totals: [usize; 3],
        pub four: [[usize; 4]; 3],
        pub phantom: [usize; 3],
        pub phantom_offered: [usize; 3],
        /// regressed blocks by loss: (a call that was made is not returned, an argument is missing, other)
        pub regressed_by_loss: [[usize; 3]; 3],
        /// blocks the program did not plan (budget or no call tag) and the v0.7.3 rule read
        pub unplanned: usize,
    }

    pub(in crate::tools) fn eval_sets(w: &Weights, with_catalogue: bool) -> SetStats {
        let cat = t::exhaustive_catalogue();
        let tools: Option<&[Value]> = if with_catalogue { Some(&cat) } else { None };
        let mut st = SetStats::default();
        for (calls, k3) in &t::exhaustive_blocks(t::FRAG.len()) {
            let body = t::exhaustive_render(calls, *k3);
            let hi = calls.iter().flatten().copied().max().unwrap();
            let first_set = t::SETS.iter().position(|&n| hi < n).unwrap();
            let r1 = crate::tools::parse_xml_calls_plan(&body, tools, QuotePairing::Off, None);
            let wf = body.contains("<function=");
            let r3 = match plan(&body, tools, wf, w, Merge::Exact) {
                Some(p) => parse_xml_calls_plan(&body, tools, QuotePairing::Cost, Some(&p.closes)),
                None => { st.unplanned += 1; r1.clone() }
            };
            let (ok1, ok3) = (t::exhaustive_right(&r1, calls), t::exhaustive_right(&r3, calls));
            let made = calls.len();
            for set in first_set..3 {
                st.totals[set] += 1;
                st.four[set][match (ok1, ok3) { (true, true) => 0, (false, true) => 1, (false, false) => 2, (true, false) => 3 }] += 1;
                if r3.len() > made { st.phantom[set] += 1; }
                if r3.iter().filter(|c| c.function.name.starts_with("tool")).count() > made { st.phantom_offered[set] += 1; }
                if ok1 && !ok3 {
                    let lost_arg = r3.iter().zip(calls).any(|(c, vals)| {
                        serde_json::from_str::<Value>(&c.function.arguments).ok().and_then(|a| a.as_object().map(|o| o.len() < vals.len())).unwrap_or(false)
                    });
                    st.regressed_by_loss[set][if r3.len() < made { 0 } else if lost_arg { 1 } else { 2 }] += 1;
                }
            }
        }
        st
    }

    /// Shapes of the sets (run by hand): one call; two calls; two calls with the missing </function>.
    #[test]
    #[ignore]
    fn shapes_table() {
        for (label, w) in [("EXPERT", Weights::EXPERT), ("SHIPPED", Weights::SHIPPED)] {
            for cat in [true, false] {
                let sh = shape_counts(&w, cat, false);
                println!("{label} {} catalogue, set C (blocks, right under v0.7.3, right under R3): 1 call {:?}  2 calls {:?}  2 calls k3 {:?}",
                         if cat { "with" } else { "without" }, sh[0], sh[1], sh[2]);
            }
        }
    }

    /// R3's counts on the exhaustive sets, with and without a catalogue (S-B9-TOOLPARSE3). A change
    /// here is a change of parser behaviour. The right/wrong columns are against v0.7.3 (the rule
    /// of the default).
    #[test]
    fn r3_counts_on_the_exhaustive_sets_are_pinned() {
        let with = eval_sets(&Weights::SHIPPED, true);
        let without = eval_sets(&Weights::SHIPPED, false);
        assert_eq!(with.totals, [68_627, 152_336, 296_039]);
        assert_eq!(with.unplanned + without.unplanned, 0, "no block of the sets exceeds a budget");
        assert_eq!(with.four, [[18_331, 24_689, 18_628, 6_979], [54_435, 53_117, 30_592, 14_192], [69_627, 71_919, 132_966, 21_527]]);
        assert_eq!(without.four, [[18_232, 3_920, 39_397, 7_078], [54_330, 9_651, 74_058, 14_297], [69_458, 16_814, 188_071, 21_696]]);
        assert_eq!((with.phantom, with.phantom_offered), ([0; 3], [0; 3]));
        assert_eq!((without.phantom, without.phantom_offered), ([103, 184, 189], [0; 3]));
        // everything R3 gets wrong that v0.7.3 got right is a CALL that was made and is not returned
        assert_eq!(with.regressed_by_loss, [[6_979, 0, 0], [14_192, 0, 0], [21_527, 0, 0]]);
    }

    /// The dynamic program finds the cheapest reading (the minimum over ALL readings, with the tie
    /// rule) on every block of set A, with and without a catalogue; set A is 68,627 blocks. The
    /// expert's Viterbi shortcut is reported (not asserted): see `exactness_table` for A, B and C.
    #[test]
    fn the_program_equals_brute_force_on_set_a() {
        let cat = t::exhaustive_catalogue();
        let w = Weights::SHIPPED;
        let (mut n, mut bad, mut vit) = (0usize, 0usize, 0usize);
        for (calls, k3) in &t::exhaustive_blocks(13) {
            let body = t::exhaustive_render(calls, *k3);
            let wf = body.contains("<function=");
            for tools in [Some(cat.as_slice()), None] {
                let all = all_readings(&body, tools, wf, &w, 1_000_000).expect("readings");
                let key = |r: &Reading| (r.total, r.ncalls, r.closes.iter().map(|x| x.1).collect::<Vec<_>>());
                let best = all.iter().map(|r| key(r)).min().unwrap();
                n += 1;
                if key(&plan(&body, tools, wf, &w, Merge::Exact).unwrap().reading) != best { bad += 1; }
                if key(&plan(&body, tools, wf, &w, Merge::Viterbi).unwrap().reading) != best { vit += 1; }
            }
        }
        println!("set A: {n} (block, catalogue) pairs: exact-key program differs from brute force in {bad}, Viterbi in {vit}");
        assert_eq!(bad, 0);
    }

    /// The profile (one forward pass per value, built a candidate at a time) is the definition
    /// (`delta_direct`, one walk per candidate) computed faster: every candidate of every value of
    /// every block of the exhaustive sets, with and without the bare-`function=` repair, and of a
    /// soup of tag pieces and prose.
    #[test]
    fn the_profile_equals_the_definition() {
        let w = Weights::SHIPPED;
        let check = |body: &str, wf: bool| -> usize {
            let ctx = Ctx::new(None);
            let Ok(blk) = Block::new(body, ctx.catalogue(), wf, &w, usize::MAX / 4) else { return 0 };
            let mut n = 0;
            for pi in 0..blk.evs.len() {
                if blk.evs[pi].kind != Kind::ParamOpen { continue; }
                let prof = blk.profile(pi).unwrap();
                let mut j = 0;
                while let Some((ci, d)) = blk.cand(pi, &prof, j).unwrap() {
                    assert_eq!(Some(d), blk.delta_direct(pi, ci), "{body:?}: value at event {pi}, candidate {j} (close {ci})");
                    n += 1;
                    j += 1;
                }
            }
            n
        };
        let mut checked = 0usize;
        for (calls, k3) in &t::exhaustive_blocks(13) {
            let body = t::exhaustive_render(calls, *k3);
            checked += check(&body, body.contains("<function="));
        }
        // soup: tags, prose, blank lines, indentation, CRs, carriage returns before tags
        const PIECES: [&str; 18] = ["<function=a>", "</function>", "<parameter=k>", "</parameter>", "\n", "\n", " ", "\t", "\r", "x",
                                    "  <parameter=p>", "</parameter>  ", "prose", "é", "<", ">", "function=b>", "<parameter=j>"];
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || { state ^= state << 13; state ^= state >> 7; state ^= state << 17; state };
        for _ in 0..4000 {
            let n = 2 + (next() % 40) as usize;
            let mut body = String::new();
            for _ in 0..n { body.push_str(PIECES[(next() % PIECES.len() as u64) as usize]); }
            checked += check(&body, body.contains("<function="));
        }
        assert!(checked > 50_000, "{checked}");
    }

    /// Branch and bound changes nothing the program answers: on every block of the exhaustive
    /// sets, the plan of the bounded program is the cheapest of the brute-force readings, ties
    /// broken as before (fewer calls, then the earlier closes). Set A is the one small enough for
    /// brute force; the counts of sets B and C are pinned by `r3_counts_on_the_exhaustive_sets_are_pinned`.
    #[test]
    fn the_bound_never_changes_the_cheapest_reading() {
        let w = Weights::SHIPPED;
        let cat = t::exhaustive_catalogue();
        let mut n = 0usize;
        for (calls, k3) in t::exhaustive_blocks(13).iter().step_by(7) {
            let body = t::exhaustive_render(calls, *k3);
            let wf = body.contains("<function=");
            for tools in [Some(cat.as_slice()), None] {
                let all = all_readings(&body, tools, wf, &w, 1_000_000).unwrap();
                let key = |r: &Reading| (r.total, r.ncalls, r.closes.iter().map(|x| x.1).collect::<Vec<_>>());
                let best = all.iter().map(key).min().unwrap();
                let got = plan(&body, tools, wf, &w, Merge::Exact).unwrap();
                assert_eq!(key(&got.reading), best, "{body:?}");
                n += 1;
            }
        }
        assert!(n > 1000, "{n}");
    }

    /// Print the four counts of a weight vector (run by hand with --ignored --nocapture).
    #[test]
    #[ignore]
    fn exhaustive_sets_table() {
        for (label, w) in [("EXPERT", Weights::EXPERT), ("SHIPPED", Weights::SHIPPED)] {
            for cat in [true, false] {
                let s = eval_sets(&w, cat);
                println!("{label} {} catalogue: totals {:?}", if cat { "with" } else { "without" }, s.totals);
                println!("    (right both, fixed, wrong both, regressed) A {:?}  B {:?}  C {:?}", s.four[0], s.four[1], s.four[2]);
                println!("    phantom {:?}  phantom to an offered tool {:?}  regressed by loss (call, arg, other) {:?}  unplanned {}",
                         s.phantom, s.phantom_offered, s.regressed_by_loss, s.unplanned);
            }
        }
    }

    // ---- corpora ---------------------------------------------------------------------------
    // Read at test time (not `include_str!`: the build stamp covers include targets). In a tree
    // without the lab's PLAN/ directory (the public repository) the corpus tests have nothing to
    // read and return; in the lab tree a missing file is a failure.
    const MANIFEST: &str = env!("CARGO_MANIFEST_DIR");

    fn corpus(rel: &str) -> Option<String> {
        if !std::path::Path::new(&format!("{MANIFEST}/PLAN")).exists() { return None; }
        Some(std::fs::read_to_string(format!("{MANIFEST}/{rel}")).unwrap_or_else(|e| panic!("{rel}: {e}")))
    }

    fn benign_tools() -> Option<String> { corpus("PLAN/TOOLPARSE3_evidence/bench_union_tools.json") }
    fn benign_rows() -> Option<String> { corpus("PLAN/TOOLPARSE3_evidence/benign_calls.jsonl") }
    fn real_outputs() -> Option<String> { corpus("PLAN/REL_SURFACE2_evidence/surface2b/gates/A3.dump_tools.log") }
    fn real_tools() -> Option<String> { corpus("PLAN/REL_SURFACE2_evidence/surface2b/tools.json") }

    fn rules_with(pairing: QuotePairing, w: &Weights) -> crate::tools::Rules {
        crate::tools::Rules { pairing, block: crate::tools::BlockEnd::First, weights: *w, trim: crate::tools::ValueTrim::All, unclosed: crate::tools::Unclosed::Drop }
    }

    fn call_sig(o: &crate::tools::ParsedOutput) -> Vec<(String, String)> {
        o.tool_calls.iter().map(|c| (c.function.name.clone(), c.function.arguments.clone())).collect()
    }

    /// Corpus (i-a): the logged calls, re-rendered. Returns (rows, blocks parsed, diffs against the
    /// v0.7.3 rule, diffs against the logged call). Every row is parsed with the benchmark's tool
    /// catalogue and without one.
    pub(in crate::tools) fn benign_logged(w: &Weights) -> (usize, usize, usize, usize) {
        let (Some(tools), Some(text)) = (benign_tools(), benign_rows()) else { return (0, 0, 0, 0) };
        let tools: Vec<Value> = serde_json::from_str(&tools).unwrap();
        let (mut rows, mut blocks, mut d_r1, mut d_logged) = (0, 0, 0, 0);
        for line in text.lines() {
            let row: Value = serde_json::from_str(line).unwrap();
            let block = row["block"].as_str().unwrap();
            let want: Vec<(String, Value)> = row["calls"].as_array().unwrap().iter()
                .map(|c| (c[0].as_str().unwrap().to_string(), c[1].clone())).collect();
            let text = format!("<tool_call>\n{block}</tool_call>");
            rows += 1;
            for cat in [Some(tools.as_slice()), None] {
                blocks += 1;
                let r1 = crate::tools::parse_rules(&text, cat, &crate::tools::Rules::V073, None);
                let r3 = crate::tools::parse_rules(&text, cat, &rules_with(QuotePairing::Cost, w), None);
                if call_sig(&r1) != call_sig(&r3) { d_r1 += 1; }
                if cat.is_some() {
                    let got: Vec<(String, Value)> = r3.tool_calls.iter()
                        .map(|c| (c.function.name.clone(), serde_json::from_str(&c.function.arguments).unwrap())).collect();
                    if got != want { d_logged += 1; }
                }
            }
        }
        (rows, blocks, d_r1, d_logged)
    }

    /// Corpus (i-b): the 348 real outputs of S-B9-REL-SURFACE2B. Returns (outputs, outputs where
    /// R3 and v0.7.3 differ, of those: outputs whose served calls hold no tag text in a value).
    pub(in crate::tools) fn benign_real_outputs(w: &Weights) -> (usize, usize, usize) {
        let (Some(tools), Some(text)) = (real_tools(), real_outputs()) else { return (0, 0, 0) };
        let tools: Vec<Value> = serde_json::from_str(&tools).unwrap();
        let (mut n, mut diff, mut diff_plain) = (0, 0, 0);
        for line in text.lines() {
            let Some(at) = line.find("[dump-tools] ") else { continue };
            let Ok(v) = serde_json::from_str::<Value>(&line[at + 13..]) else { continue };
            let Some(raw) = v["raw"].as_str() else { continue };
            n += 1;
            let r1 = crate::tools::parse_rules(raw, Some(&tools), &crate::tools::Rules::V073, None);
            let r3 = crate::tools::parse_rules(raw, Some(&tools), &rules_with(QuotePairing::Cost, w), None);
            if call_sig(&r1) != call_sig(&r3) || r1.content != r3.content {
                diff += 1;
                let tagged = |o: &crate::tools::ParsedOutput| o.tool_calls.iter().any(|c| {
                    ["<function=", "<parameter=", "</parameter>", "</function>"].iter().any(|t| c.function.arguments.contains(t))
                });
                if !tagged(&r1) && !tagged(&r3) { diff_plain += 1; }
            }
        }
        (n, diff, diff_plain)
    }

    #[test]
    fn the_benign_corpus_parses_identically_under_every_rule_at_the_shipped_weights() {
        let (rows, blocks, d_r1, d_logged) = benign_logged(&Weights::SHIPPED);
        println!("benign (logged calls re-rendered): {rows} rows, {blocks} parses, {d_r1} differ from v0.7.3, {d_logged} differ from the logged call");
        if benign_rows().is_none() { return; } // the public tree has no PLAN/
        assert!(rows > 800 && blocks > 1600);
        assert_eq!((d_r1, d_logged), (0, 0));
        // and the 348 real outputs of S-B9-REL-SURFACE2B: none whose served calls hold no tag text may differ
        let (n, diff, plain) = benign_real_outputs(&Weights::SHIPPED);
        println!("real outputs: {n}, differing from v0.7.3: {diff} (of which with no tag text in a value: {plain})");
        assert!(n >= 340);
        assert_eq!(plain, 0);
    }

    /// T3 on real output: the structural block rule differs from the plain one in exactly the 5 of the
    /// 348 real outputs that were cut at the close of a complete quoted block, and in no other
    /// (the finish reason is `length`, so that a block lacking its close is not the difference).
    #[test]
    fn the_default_block_rule_differs_from_first_close_in_exactly_the_cut_outputs() {
        let Some(text) = real_outputs() else { return };
        let (mut n, mut differ) = (0usize, Vec::new());
        for line in text.lines() {
            let Some(at) = line.find("[dump-tools] ") else { continue };
            let Ok(v) = serde_json::from_str::<Value>(&line[at + 13..]) else { continue };
            let Some(raw) = v["raw"].as_str() else { continue };
            n += 1;
            let a = crate::tools::block_spans(raw, crate::tools::BlockEnd::First, Some("length"));
            let b = crate::tools::block_spans(raw, crate::tools::BlockEnd::AfterFunction, Some("length"));
            if a != b { differ.push(n); }
        }
        assert!(n >= 340, "{n}");
        assert_eq!(differ.len(), 5, "outputs whose blocks differ: {differ:?}");
        // ... and each of those 5 is read WHOLE by the default rule: one write_file whose content is
        // everything the model wrote between the content tag and the closing sequence, no prose left over
        let mut whole = 0usize;
        let mut k = 0usize;
        for line in text.lines() {
            let Some(at) = line.find("[dump-tools] ") else { continue };
            let Ok(v) = serde_json::from_str::<Value>(&line[at + 13..]) else { continue };
            let Some(raw) = v["raw"].as_str() else { continue };
            k += 1;
            if !differ.contains(&k) { continue; }
            let o = crate::tools::parse_rules(raw, None, &crate::tools::Rules { block: crate::tools::BlockEnd::AfterFunction, trim: crate::tools::ValueTrim::One, ..crate::tools::Rules::V073 }, Some("stop"));
            assert_eq!(o.tool_calls.len(), 1, "output {k}");
            assert_eq!(o.content, "", "output {k}: nothing of the call leaks into the prose");
            let a: Value = serde_json::from_str(&o.tool_calls[0].function.arguments).unwrap();
            let want = raw.split_once("<parameter=content>\n").unwrap().1.strip_suffix("\n</parameter>\n</function>\n</tool_call>").unwrap();
            assert_eq!(a["content"].as_str(), Some(want), "output {k}");
            whole += 1;
        }
        assert_eq!(whole, 5);
    }

    // ---- the named cases as constraints ------------------------------------------------------

    /// For one named case under one catalogue setting: did the chosen reading return the right
    /// calls, and by what margin (the cheapest wrong reading minus the cheapest right one).
    #[derive(Clone, Debug)]
    pub(in crate::tools) struct CaseEval { pub right: bool, pub margin: i64, pub chosen: u32, pub rival: Option<u32>, pub best_right: Option<u32> }

    pub(in crate::tools) fn eval_case(calls: &[Call], k3: bool, tools: Option<&[Value]>, w: &Weights) -> CaseEval {
        let body = render(calls, k3);
        let wf = body.contains("<function=");
        let key = |r: &Reading| (r.total, r.ncalls, r.closes.iter().map(|x| x.1).collect::<Vec<_>>());
        let all = all_readings(&body, tools, wf, w, 500_000).expect("readings");
        let right_of = |r: &Reading| {
            let map: HashMap<usize, usize> = r.closes.iter().copied().collect();
            same(&crate::tools::parse_xml_calls_w(&body, tools, QuotePairing::Cost, Some(&map), w), calls)
        };
        let chosen = all.iter().min_by_key(|r| key(r)).unwrap();
        let best_right = all.iter().filter(|r| right_of(r)).map(|r| r.total).min();
        let rival = all.iter().filter(|r| !right_of(r)).map(|r| r.total).min();
        let right = right_of(chosen);
        let margin = match (best_right, rival) {
            (Some(b), Some(r)) => r as i64 - b as i64,
            (Some(_), None) => 1000,
            _ => -1000,
        };
        CaseEval { right, margin, chosen: chosen.total, rival, best_right }
    }

    /// Named cases that must be right (every named case of the request and the follow-up except
    /// E7, which is undecidable by construction and pinned separately).
    pub(in crate::tools) fn named_eval(w: &Weights) -> (usize, i64) {
        let cat = expert_catalogue();
        let (mut fails, mut min_margin) = (0usize, i64::MAX);
        for (_, calls, k3) in named_cases() {
            for tools in [Some(cat.as_slice()), None] {
                let e = eval_case(&calls, k3, tools, w);
                if !e.right { fails += 1; }
                min_margin = min_margin.min(e.margin);
            }
        }
        (fails, min_margin)
    }

    /// Every named case of the request and of the follow-up, read at the shipped weights, with a
    /// catalogue and without one. The margins are the cheapest wrong reading minus the cheapest
    /// right one, MEASURED (the expert's table predicted others: see the report). Pinned so that a
    /// change of a weight shows which case it moves.
    #[test]
    fn the_named_cases_at_the_shipped_weights() {
        let cat = expert_catalogue();
        let w = Weights::SHIPPED;
        // (name prefix, margin with a catalogue, margin without)
        let want: [(&str, i64, i64); 17] = [
            ("E1 ", 32, 27), ("E2 ", 19, -6), ("E3 ", 19, 4), ("E4 ", 4, 4), ("E5 ", 20, 15), ("E6/F2", 5, 7),
            ("E8 ", 4, 4), ("E9 ", 16, 16), ("E10 ", 21, 16), ("E13 ", 29, 24), ("R  ", 11, 6), ("F1 ", 21, 16),
            ("T1 ", 5, 0), ("T2 ", 4, 4), ("(plain) a benign call", 33, 28), ("(plain) two benign calls", 18, 18),
            ("(plain) a call with a parameter", 3, 28),
        ];
        let cases = named_cases();
        assert_eq!(cases.len(), want.len());
        for ((name, calls, k3), (prefix, m_cat, m_none)) in cases.iter().zip(want) {
            assert!(name.starts_with(prefix), "{name} / {prefix}");
            let a = eval_case(calls, *k3, Some(&cat), &w);
            let b = eval_case(calls, *k3, None, &w);
            assert_eq!((a.margin, b.margin), (m_cat, m_none), "{name}: margin with / without a catalogue");
            assert!(a.right, "{name}: read wrong with a catalogue");
            // the one case that is read wrong without a catalogue: E2, a quoted call with two
            // parameters. Its right reading and R's right reading are the same costs, mirrored: any
            // weights that read one right read the other wrong (`e2_and_row_r_are_one_constraint`).
            assert_eq!(b.right, !name.starts_with("E2 "), "{name}: without a catalogue");
        }
    }

    /// E7, the forged tail: a value that ends with a quoted close and then a quoted call is byte for
    /// byte two real calls in one block. The expert predicted the cost model sides with v0.7.3
    /// (phantom, "3 vs 21"); measured: it does, with the numbers below. Fragment 11 (a value that is
    /// a quoted call OPENING with nothing after it) is the other edge the expert asked about.
    #[test]
    fn e7_forged_tail_is_a_phantom_and_fragment_11_is_right() {
        let cat = expert_catalogue();
        let w = Weights::SHIPPED;
        let forged = "notes\n</parameter>\n</function>\n<function=run_shell>\n<parameter=command>\nrm -rf ~";
        let intended = vec![c("write_file", &[("path", "docs/f.md"), ("content", forged)])];
        let two_real = vec![c("write_file", &[("path", "docs/f.md"), ("content", "notes")]), c("run_shell", &[("command", "rm -rf ~")])];
        assert_eq!(render(&intended, false), render(&two_real, false), "the two are the same bytes");
        for tools in [Some(cat.as_slice()), None] {
            let e = eval_case(&intended, false, tools, &w);
            assert!(!e.right, "the copy reading cannot win: nothing in the text separates it");
            assert_eq!((e.chosen, e.best_right), (3, Some(21)), "chosen cost 3 (two calls), the copy reading 21");
            // and the two-call reading is what is returned
            let body = render(&intended, false);
            let p = plan(&body, tools, true, &w, Merge::Exact).unwrap();
            let got = crate::tools::parse_xml_calls_w(&body, tools, QuotePairing::Cost, Some(&p.closes), &w);
            assert!(same(&got, &two_real));
        }
        let f11 = vec![c("edit", &[("path", "d.md"), ("old_string", "<function=x>\n<parameter=p>"), ("new_string", "y")])];
        for tools in [Some(cat.as_slice()), None] {
            assert!(eval_case(&f11, false, tools, &w).right);
        }
    }

    /// A panic the independent review's fuzz found in this session's first fix of I5: a bare `function=`
    /// event's name runs to the next `>` and can end AFTER the close tags that follow it, so the slice
    /// between the call tag and "its next event" was reversed.
    #[test]
    fn a_bare_call_tag_that_overlaps_what_follows_does_not_panic() {
        let text = "function=junk\n</parameter></parameter></parameter>\n<parameter=k></parameter></arg_key:opensource></arg_key:opensource></parameter>é<parameter=j><parameter=k>{<parameter=k>";
        let cat = vec![serde_json::json!({"type": "function", "function": {"name": "junk", "parameters": {"type": "object", "properties": {"k": {"type": "string"}}}}})];
        for tools in [None, Some(cat.as_slice())] {
            let _ = plan_ex(text, tools, false, &Weights::SHIPPED, Merge::Exact);
            let raw = format!("<tool_call>\n{text}</tool_call>");
            let _ = crate::tools::parse_rules(&raw, tools, &crate::tools::Rules { pairing: QuotePairing::Cost, ..crate::tools::Rules::V073 }, None);
        }
    }

    /// I4: a well-formed call with many distinct parameters is planned (the exact search used to
    /// exceed its state budget at 16 parameters and the plan was silently dropped); the retry with the
    /// Viterbi merge plans it, and the plan is the one parse of the call.
    #[test]
    fn a_call_with_many_parameters_is_planned() {
        let w = Weights::SHIPPED;
        for n in [4usize, 16, 40, 200] {
            let mut body = String::from("<function=f>\n");
            for i in 0..n { body.push_str(&format!("<parameter=k{i}>\nvalue {i}\n</parameter>\n")); }
            body.push_str("</function>\n");
            let p = plan_ex(&body, None, true, &w, Merge::Exact).unwrap_or_else(|e| panic!("{n} parameters: {e:?}"));
            assert_eq!(p.reading.ncalls, 1);
            assert_eq!(p.closes.len(), n, "every value is closed where the plan says");
            assert_eq!(p.reading.total, 0, "a well-formed call costs nothing");
        }
    }

    /// I3: the byte walks are charged: a value of 100 KB of blank lines under 16 empty parameters did
    /// not hit the work budget and took 23 s. Now the block is declined (read by v0.7.3) or planned
    /// inside its units, and what it did is covered by what it charged (counted, not timed).
    #[test]
    fn byte_walks_are_charged_to_the_work_budget() {
        let mut body = String::from("<function=f>\n");
        for _ in 0..16 { body.push_str("<parameter=k>\n</parameter>\n"); }
        body.push_str(&" \n".repeat(50_000));
        for _ in 0..80 { body.push_str("</parameter>\n"); }
        body.push_str("</function>\n");
        OPS.with(|c| c.set(0));
        let (r, units) = plan_metered(&body, None, true, &Weights::SHIPPED, Merge::Exact);
        let ops = OPS.with(|c| c.get());
        println!("16 empty params + 100 KB of blank lines + 80 stray closes: {units} units, {ops} operations ({:?})", r.as_ref().map(|p| p.reading.total).map_err(|e| *e));
        assert!(units <= BLOCK_UNITS + body.len() / WALK_BYTES + 2_000, "{units}");
        assert!(ops <= 32 * units + 2_000, "{ops} operations against {units} units charged");
        // the bare `function=` repair loop: 75,000 candidates, no `<function=` in the block
        let bare = " function=x".repeat(75_000);
        OPS.with(|c| c.set(0));
        let (_, units) = plan_metered(&bare, None, false, &Weights::SHIPPED, Merge::Exact);
        let ops = OPS.with(|c| c.get());
        assert!(units <= BLOCK_UNITS + bare.len() / WALK_BYTES + 2_000, "{units}");
        assert!(ops <= 32 * units + 2_000, "{ops} operations against {units} units charged");
    }

    /// The pinch pair, as arithmetic. E2 without a catalogue is right iff K < 1 + D + S and row R
    /// is right iff K > 1 + S + D (K = skip_param, D = dangling, S = stray): the same sum on both
    /// sides, so no weight vector separates them. At K == 1 + S + D the tie goes to the v0.7.3
    /// close, which is right for R and wrong for E2.
    #[test]
    fn e2_and_row_r_are_one_constraint() {
        let cat = expert_catalogue();
        let find = |p: &str| named_cases().into_iter().find(|(n, _, _)| n.starts_with(p)).unwrap();
        let (_, e2, k2) = find("E2 ");
        let (_, r, kr) = find("R  ");
        for (k, d, s) in [(17u32, 2u32, 8u32), (10, 2, 8), (12, 3, 9), (30, 5, 20), (5, 1, 1)] {
            let w = Weights { skip_param: k, skip_param_cat: k, dangling: d, stray: s, ..Weights::SHIPPED };
            let (e2_right, r_right) = (eval_case(&e2, k2, None, &w).right, eval_case(&r, kr, None, &w).right);
            assert!(!(e2_right && r_right), "K={k} D={d} S={s}: both cannot be right");
            if k < 1 + d + s { assert!(e2_right && !r_right, "K={k} D={d} S={s}"); }
            if k >= 1 + d + s { assert!(!e2_right && r_right, "K={k} D={d} S={s}"); }
        }
        let _ = cat;
    }

    /// The invariant of the follow-up: a real second call whose first call has no `</function>` must
    /// not disappear because of the repair's own cost: C_CALL + C_NO_FCLOSE < C_SKIP_CALL + C_STRAY +
    /// C_TAG_CONTENT + C_DANGLING.
    #[test]
    fn the_missing_function_close_repair_never_hides_a_second_call() {
        for w in [Weights::EXPERT, Weights::SHIPPED] {
            assert!(w.call + w.no_fclose < w.skip_call + w.stray + w.tag_content + w.dangling, "{w:?}");
        }
        // and as behaviour: two real calls, no </function> between, with and without a catalogue,
        // under both tables
        let cat = expert_catalogue();
        let calls = vec![c("read", &[("path", "a")]), c("run_shell", &[("command", "ls"), ("stdin", "")])];
        for w in [Weights::EXPERT, Weights::SHIPPED] {
            for tools in [Some(cat.as_slice()), None] {
                assert!(eval_case(&calls, true, tools, &w).right);
            }
        }
    }

    // ---- training and hold-out shapes ------------------------------------------------------

    /// Per shape of the exhaustive sets: (blocks, right under R1, right under R3) for set C,
    /// with and without a catalogue, shapes: 1 call; 2 calls; 2 calls with the missing </function>.
    pub(in crate::tools) fn shape_counts(w: &Weights, with_cat: bool, only_one_call: bool) -> [(usize, usize, usize); 3] {
        let cat = t::exhaustive_catalogue();
        let tools: Option<&[Value]> = if with_cat { Some(&cat) } else { None };
        let mut out = [(0usize, 0usize, 0usize); 3];
        for (calls, k3) in &t::exhaustive_blocks(t::FRAG.len()) {
            if only_one_call && calls.len() != 1 { continue; }
            let shape = if calls.len() == 1 { 0 } else if !*k3 { 1 } else { 2 };
            let body = t::exhaustive_render(calls, *k3);
            let wf = body.contains("<function=");
            let r1 = crate::tools::parse_xml_calls_plan(&body, tools, QuotePairing::Off, None);
            let r3 = match plan(&body, tools, wf, w, Merge::Exact) {
                Some(p) => crate::tools::parse_xml_calls_w(&body, tools, QuotePairing::Cost, Some(&p.closes), w),
                None => r1.clone(),
            };
            out[shape].0 += 1;
            if t::exhaustive_right(&r1, calls) { out[shape].1 += 1; }
            if t::exhaustive_right(&r3, calls) { out[shape].2 += 1; }
        }
        out
    }

    fn set_field(w: &mut Weights, i: usize, v: u32) {
        match i {
            0 => w.skip_param = v, 1 => w.skip_param_cat = v, 2 => w.skip_call = v, 3 => w.skip_more = v,
            4 => w.call = v, 5 => w.no_fclose = v, 6 => w.empty = v, 7 => w.tag_content = v,
            8 => w.dangling = v, 9 => w.stray = v, 10 => w.tail = v, 11 => w.required = v,
            12 => w.duplicate = v, 13 => w.undeclared = v, 14 => w.unoffered = v, 15 => w.junk = v,
            _ => unreachable!(),
        }
    }

    fn get_field(w: &Weights, i: usize) -> u32 {
        [w.skip_param, w.skip_param_cat, w.skip_call, w.skip_more, w.call, w.no_fclose, w.empty, w.tag_content,
         w.dangling, w.stray, w.tail, w.required, w.duplicate, w.undeclared, w.unoffered, w.junk][i]
    }

    const FIELD_NAMES: [&str; 16] = ["skip_param", "skip_param_cat", "skip_call", "skip_more", "call", "no_fclose", "empty",
        "tag_content", "dangling", "stray", "tail", "required", "duplicate", "undeclared", "unoffered", "junk"];

    /// The objective of the fit, lexicographic: the benign corpus must not move (a vector that moves
    /// it is rejected outright), then named cases wrong, then the smallest named margin (capped), then
    /// the right ONE-CALL blocks of set C with and without a catalogue (the training shapes).
    fn objective(w: &Weights) -> Option<(i64, i64, i64)> {
        let (_, _, d_r1, d_logged) = benign_logged(w);
        let (_, _, d_plain) = benign_real_outputs(w);
        if d_r1 + d_logged + d_plain > 0 { return None; }
        let (fails, min_margin) = named_eval(w);
        let train: usize = [true, false].iter().map(|&c| shape_counts(w, c, true)[0].2).sum();
        Some((-(fails as i64), min_margin.min(6), train as i64))
    }

    /// The fit (run by hand: `--ignored --nocapture fit_weights`): coordinate ascent from the
    /// expert's table on the objective above, then the held-out shapes are reported.
    #[test]
    #[ignore]
    fn fit_weights() {
        let mut w = Weights::EXPERT;
        let show = |w: &Weights| FIELD_NAMES.iter().enumerate().map(|(i, n)| format!("{n}={}", get_field(w, i))).collect::<Vec<_>>().join(" ");
        println!("start {}", show(&w));
        match objective(&w) {
            Some(o) => println!("expert objective {o:?}"),
            None => {
                let (rows, blocks, d_r1, d_logged) = benign_logged(&w);
                let (n, diff, plain) = benign_real_outputs(&w);
                println!("the expert's table MOVES the benign corpus: logged {rows} rows / {blocks} parses: {d_r1} differ from v0.7.3, {d_logged} from the logged call; real outputs {n}: {diff} differ, {plain} of them without tag text");
            }
        }
        // start from the expert's table with the three catalogue-evidence weights brought below the
        // cost of the cheapest swallow (11), the move the benign corpus forces (see the report)
        let mut best: Option<(i64, i64, i64)> = None;
        let mut seeds = vec![w];
        for (u, d, x) in [(5, 5, 3), (8, 8, 5), (3, 3, 3)] {
            let mut v = w; v.undeclared = u; v.duplicate = d; v.unoffered = x; seeds.push(v);
        }
        for seed in seeds {
            if let Some(o) = objective(&seed) {
                if best.map_or(true, |b| o > b) { best = Some(o); w = seed; }
            }
        }
        let mut best = best.expect("no benign-clean seed");
        println!("seed {} -> {best:?}", show(&w));
        let steps: [i32; 6] = [-4, -2, -1, 1, 2, 4];
        for pass in 0..6 {
            let mut moved = false;
            for i in 0..16 {
                for &d in &steps {
                    let cur = get_field(&w, i) as i32;
                    let nv = cur + d;
                    if !(0..=60).contains(&nv) { continue; }
                    let mut v = w; set_field(&mut v, i, nv as u32);
                    if let Some(o) = objective(&v) {
                        if o > best { best = o; w = v; moved = true; println!("pass {pass}: {} {cur}->{nv}: {best:?}", FIELD_NAMES[i]); }
                    }
                }
            }
            if !moved { break; }
        }
        println!("final {}", show(&w));
        println!("objective {best:?}");
        for cat in [true, false] {
            let sh = shape_counts(&w, cat, false);
            println!("{} catalogue, set C shapes (blocks, right under v0.7.3, right under R3): 1 call {:?}, 2 calls {:?}, 2 calls k3 {:?}",
                     if cat { "with" } else { "without" }, sh[0], sh[1], sh[2]);
        }
        let (fails, margin) = named_eval(&w);
        println!("named cases wrong {fails}, smallest margin {margin}");
    }

    /// The benign rows R3 reads differently from v0.7.3 under a weight vector (run by hand).
    #[test]
    #[ignore]
    fn benign_diffs_at_expert_weights() {
        const SHOW_DIFFS: usize = 10; // raise by hand to print more
        let w = Weights::EXPERT;
        let (Some(tools), Some(text)) = (benign_tools(), benign_rows()) else { return };
        let tools: Vec<Value> = serde_json::from_str(&tools).unwrap();
        let mut shown = 0usize;
        let mut kinds: std::collections::BTreeMap<String, usize> = Default::default();
        for line in text.lines() {
            let row: Value = serde_json::from_str(line).unwrap();
            let block = row["block"].as_str().unwrap();
            let text = format!("<tool_call>\n{block}</tool_call>");
            for (label, cat) in [("catalogue", Some(tools.as_slice())), ("none", None)] {
                let r1 = crate::tools::parse_rules(&text, cat, &crate::tools::Rules::V073, None);
                let r3 = crate::tools::parse_rules(&text, cat, &rules_with(QuotePairing::Cost, &w), None);
                if call_sig(&r1) == call_sig(&r3) { continue; }
                let key = format!("{label}: {} -> {} calls ({})", r1.tool_calls.len(), r3.tool_calls.len(), row["source"].as_str().unwrap());
                *kinds.entry(key).or_default() += 1;
                if shown < SHOW_DIFFS {
                    shown += 1;
                    let body = &text["<tool_call>\n".len()..text.len() - "</tool_call>".len()];
                    let wf = body.contains("<function=");
                    let p = plan(body, cat, wf, &w, Merge::Exact).unwrap();
                    let r1k = score(body, cat, wf, &w, &|vs, c| { let o = param_end_v073(&body[vs..])?; c.iter().copied().find(|&x| x == vs + o) }).unwrap();
                    let short = |o: &crate::tools::ParsedOutput| o.tool_calls.iter().map(|c| format!("{}({}B)", c.function.name, c.function.arguments.len())).collect::<Vec<_>>().join(" ");
                    println!("[{label}] {}: R1 {} cost {}; R3 {} cost {}", row["source"], short(&r1), r1k.describe(), short(&r3), p.reading.describe());
                }
            }
        }
        println!("{kinds:#?}");
    }

    /// Per named case: right or wrong and the margin, for the weight vector `candidate()` (run by hand).
    #[test]
    #[ignore]
    fn named_margins_table() {
        let cat = expert_catalogue();
        for (label, w) in [("fit1", Weights { skip_param: 17, skip_param_cat: 15, skip_call: 4, skip_more: 9, call: 3, no_fclose: 2, empty: 0, tag_content: 1, dangling: 9, stray: 7, tail: 0, required: 2, duplicate: 8, undeclared: 8, unoffered: 5, junk: 100, candidates: 64 })] {
            println!("== {label}");
            for (name, calls, k3) in named_cases() {
                let a = eval_case(&calls, k3, Some(&cat), &w);
                let b = eval_case(&calls, k3, None, &w);
                println!("{:<62} cat: {} margin {:>4} (chosen {:>3}, rival {:?}) | none: {} margin {:>4} (chosen {:>3}, rival {:?})",
                         name, if a.right { "ok " } else { "BAD" }, a.margin, a.chosen, a.rival, if b.right { "ok " } else { "BAD" }, b.margin, b.chosen, b.rival);
            }
        }
    }

    /// Which fragments the regressed (right under v0.7.3, wrong under R3) blocks of set A hold, and a
    /// few examples with the cost of the reading R3 took against the cost of the right reading.
    #[test]
    #[ignore]
    fn regressed_blocks_of_set_a() {
        let cat = t::exhaustive_catalogue();
        let w = Weights::EXPERT;
        let mut by_frag = vec![0usize; t::FRAG.len()];
        let mut by_shape: std::collections::BTreeMap<String, usize> = Default::default();
        let mut shown = 0usize;
        let mut total = 0usize;
        for (calls, k3) in &t::exhaustive_blocks(13) {
            let body = t::exhaustive_render(calls, *k3);
            let r1 = crate::tools::parse_xml_calls_plan(&body, Some(&cat), QuotePairing::Off, None);
            let wf = body.contains("<function=");
            let p = plan(&body, Some(&cat), wf, &w, Merge::Exact).unwrap();
            let r3 = parse_xml_calls_plan(&body, Some(&cat), QuotePairing::Cost, Some(&p.closes));
            if !(t::exhaustive_right(&r1, calls) && !t::exhaustive_right(&r3, calls)) { continue; }
            total += 1;
            for f in calls.iter().flatten() { by_frag[*f] += 1; }
            let shape = format!("{} calls, k3={} frags={:?}", calls.len(), k3, calls);
            *by_shape.entry(format!("{}c k3={}", calls.len(), k3)).or_default() += 1;
            if shown < 6 && (shown as usize * 997) % 7 == 0 || shown < 3 {
                shown += 1;
                println!("--- {shape}\n{body}--- R1 reading: {} calls; R3 took: {}", r1.len(), p.reading.describe());
                let all = all_readings(&body, Some(&cat), wf, &w, 100_000).unwrap();
                let key = |r: &Reading| (r.total, r.ncalls, r.closes.iter().map(|x| x.1).collect::<Vec<_>>());
                let good = all.iter().filter(|r| t::exhaustive_right(&parse_xml_calls_plan(&body, Some(&cat), QuotePairing::Cost, Some(&r.closes.iter().copied().collect())), calls)).min_by_key(|r| key(r));
                println!("    right reading: {}", good.map_or("none".into(), |r| r.describe()));
            }
        }
        println!("regressed {total}; by fragment (index: count): {:?}", by_frag.iter().enumerate().filter(|(_, &c)| c > 0).collect::<Vec<_>>());
        println!("by shape {by_shape:?}");
    }

    /// The program against brute force: for every block of the three sets the cheapest reading the
    /// dynamic program returns must cost what the cheapest of ALL readings costs (and equal it on
    /// the tie rule). Returns (blocks, exact mismatches, viterbi mismatches).
    pub(in crate::tools) fn exactness(w: &Weights, with_catalogue: bool) -> (usize, usize, usize) {
        let cat = t::exhaustive_catalogue();
        let tools: Option<&[Value]> = if with_catalogue { Some(&cat) } else { None };
        let (mut n, mut exact_bad, mut vit_bad) = (0usize, 0usize, 0usize);
        for (calls, k3) in &t::exhaustive_blocks(t::FRAG.len()) {
            let body = t::exhaustive_render(calls, *k3);
            let wf = body.contains("<function=");
            let all = all_readings(&body, tools, wf, w, 1_000_000).expect("all readings");
            let key = |r: &Reading| (r.total, r.ncalls, r.closes.iter().map(|x| x.1).collect::<Vec<_>>());
            let best = all.iter().map(|r| key(r)).min().unwrap();
            n += 1;
            let e = plan(&body, tools, wf, w, Merge::Exact).expect("plan");
            if key(&e.reading) != best { exact_bad += 1; }
            let v = plan(&body, tools, wf, w, Merge::Viterbi).expect("plan");
            if key(&v.reading) != best { vit_bad += 1; }
        }
        (n, exact_bad, vit_bad)
    }

    #[test]
    #[ignore]
    fn exactness_table() {
        for cat in [true, false] {
            let (n, e, v) = exactness(&Weights::EXPERT, cat);
            println!("exactness {} catalogue: {n} blocks, exact-key DP differs from brute force in {e}, Viterbi in {v}", if cat { "with" } else { "without" });
        }
    }

    /// The measurement table of the follow-up: every named case, with and without a catalogue, the
    /// cost breakdown of the cheapest reading, of R1's reading and of the cheapest correct reading.
    #[test]
    #[ignore]
    fn named_cases_table() {
        let cat = expert_catalogue();
        for (wname, w) in [("EXPERT", Weights::EXPERT), ("SHIPPED", Weights::SHIPPED)] {
        println!("==== weights {wname}");
        for (name, calls, k3) in named_cases() {
            let body = render(&calls, k3);
            for (label, tools) in [("catalogue", Some(cat.as_slice())), ("no catalogue", None)] {
                let wf = body.contains("<function=");
                let all = all_readings(&body, tools, wf, &w, 200_000).expect("readings");
                let r1 = score(&body, tools, wf, &w, &|vs, cands| {
                    let o = param_end_v073(&body[vs..])?;
                    let p = vs + o;
                    cands.iter().copied().find(|&c| c == p)
                }).expect("r1 reading");
                let best = plan(&body, tools, wf, &w, Merge::Exact).expect("plan");
                let key = |r: &Reading| (r.total, r.ncalls, r.closes.iter().map(|x| x.1).collect::<Vec<_>>());
                let min_ok = all.iter().filter(|r| same(&calls_of(&body, tools, r), &calls)).min_by_key(|r| key(r));
                let min_bad = all.iter().filter(|r| !same(&calls_of(&body, tools, r), &calls)).min_by_key(|r| key(r));
                let got = calls_of(&body, tools, &best.reading);
                println!("{name} [{label}]: R3 returns {} call(s), {}", got.len(), if same(&got, &calls) { "RIGHT" } else { "WRONG" });
                println!("    cheapest: {}", best.reading.describe());
                println!("    R1      : {}", r1.describe());
                println!("    correct : {}", min_ok.map_or("(none of the readings)".to_string(), |r| r.describe()));
                println!("    rival   : {}", min_bad.map_or("(none)".to_string(), |r| r.describe()));
            }
        }
        }
    }
