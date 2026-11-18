//! V0710 (owner rulings 2026-11-13): family- and TP-scoped RESOLVED defaults for the EXL3 server.
//!
//! Features that pay and never change output became default-on, each keeping its `off` switch;
//! anything that can change output stays opt-in. The default-on set was measured on ONE family
//! (Flash-Next, `model_type` qwen4_exp, which includes the FP4-Local pack on the EXL3 trunk), so
//! it resolves ON only there:
//!
//! - `--copy-span` auto -> on, with `--copy-span-max-w` 15 (the wide copy verify);
//! - `--exl3-dense-resident` auto -> on.
//!
//! `--exl3-mid-gemm` was on the owner's default-on list but is NOT output-identical to v0.7.9: it serves a
//! 17..64-row prefill chunk through the decode-class trellis body instead of the reconstruct + vendor GEMM, so a short
//! prompt (one 17..64-row chunk) reads differently (V0710 gate 2026-11-14, Flash-Next 3.05bpw TP=1, greedy: 6 of 7
//! short prompts changed text with it on, 0 of 7 with only it off). Under the same ruling ("anything that can change
//! output stays opt-in, always") it stays opt-in (registry default off) and is not resolved here.
//!
//! Every other family and loader (Qwen3.5/3.6 EXL3 packs, MoE, the NVFP4/bf16 trunk server and its
//! copy arm, DeepSeek V4, Naive) keeps them OFF unless asked: nothing is written to the option store
//! for them, so such a boot is the v0.7.9 boot.
//!
//! `--spec-arm-meter` auto -> v2 at TP=1 and TP=2, v1 at TP=4 (and any larger world), on every
//! EXL3 boot (the only server that reads it).
//!
//! An explicit value on the command line (or shipped by a TP head) always wins. Resolution is a pure
//! function of (family, TP world, the set values), so every TP rank resolves alike whether the head
//! shipped its resolved values or not; it runs once per boot at the start of the EXL3 `build_serve`,
//! before any of these options is read, and prints ONE line. `/metrics` carries
//! `velogb10_resolved_default{option,value,source}` and `owner_check.py` prints it.

use std::sync::Mutex;

/// The Flash-Next copy verify width the default resolves to (16 rows, the batch-invariant ceiling).
pub const FN_COPY_MAX_W: &str = "15";

/// One option's resolution on this boot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resolved {
    /// the flag name without dashes
    pub flag: &'static str,
    /// the value in force (the option store's spelling: "1" / "0" for on / off)
    pub value: String,
    /// "auto" = resolved here; "set" = given explicitly (command line or the TP head's snapshot)
    pub auto: bool,
}

/// The options this module resolves, in the boot line's order.
pub const FLAGS: [&str; 4] = ["copy-span", "copy-span-max-w", "exl3-dense-resident", "spec-arm-meter"];

/// `model_type` (root or `text_config`) of an EXL3 pack's config.json is qwen4_exp / qwen4_exp_text.
pub fn is_flash_next_dir(model_dir: &str) -> bool {
    let p = std::path::Path::new(model_dir).join("config.json");
    let Ok(raw) = std::fs::read_to_string(p) else { return false };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) else { return false };
    let fam = |t: &serde_json::Value| matches!(t.as_str(), Some("qwen4_exp") | Some("qwen4_exp_text"));
    fam(&v["model_type"]) || fam(&v["text_config"]["model_type"])
}

/// Pure decision. `set(flag)` = the value already in the store (explicit). Returns every option's
/// resolution; `auto` rows with `Some` value are the ones to write. A non-Flash-Next family gets no
/// write for the three family-scoped options (their value reads as the registry's off).
pub fn decide(flash_next: bool, world: usize, set: &dyn Fn(&str) -> Option<String>) -> Vec<(Resolved, bool)> {
    let mut v = Vec::new();
    for flag in FLAGS {
        if let Some(x) = set(flag) {
            v.push((Resolved { flag, value: x, auto: false }, false));
            continue;
        }
        let (value, write) = match flag {
            "copy-span" | "exl3-dense-resident" => {
                if flash_next { ("1".to_string(), true) } else { ("0".to_string(), false) }
            }
            "copy-span-max-w" => {
                if flash_next { (FN_COPY_MAX_W.to_string(), true) } else { ("10".to_string(), false) }
            }
            "spec-arm-meter" => (if world <= 2 { "v2" } else { "v1" }.to_string(), true),
            _ => unreachable!(),
        };
        v.push((Resolved { flag, value, auto: true }, write));
    }
    v
}

fn word(flag: &str, value: &str) -> String {
    match flag {
        "copy-span" | "exl3-dense-resident" => if crate::opts::is_on_value(value) { "on" } else { "off" }.to_string(),
        _ => value.to_string(),
    }
}

/// The one boot line.
pub fn boot_line(flash_next: bool, world: usize, rows: &[Resolved]) -> String {
    let fam = if flash_next { "Flash-Next (qwen4_exp) EXL3 family" } else { "not the Flash-Next family" };
    let items: Vec<String> = rows.iter().map(|r| format!("--{} {} ({})", r.flag, word(r.flag, &r.value),
                                                          if r.auto { "auto" } else { "set" })).collect();
    let why = if flash_next {
        "auto turns copy-span (max-w 15) and dense-resident on for this family only (output-identical, measured there); --exl3-mid-gemm stays opt-in (it changes short-prompt output)"
    } else {
        "auto keeps copy-span and dense-resident off outside the Flash-Next family (never measured here)"
    };
    format!("[exl3-serve] resolved defaults (TP={world}, {fam}): {}; {why}; --spec-arm-meter auto = v2 at TP=1/2, v1 at TP=4; \
an explicit on/off always wins", items.join(", "))
}

static RESOLVED: Mutex<Vec<Resolved>> = Mutex::new(Vec::new());

/// Boot: resolve, write the auto values to the option store, print the line, keep the rows for /metrics.
pub fn resolve_exl3(model_dir: &str, world: usize) {
    let fnx = is_flash_next_dir(model_dir);
    let get = |f: &str| crate::opts::lookup(f).and_then(crate::opts::get);
    let d = decide(fnx, world, &get);
    for (r, write) in &d {
        if *write {
            if let Some(id) = crate::opts::lookup(r.flag) { crate::opts::set(id, &r.value); }
        }
    }
    let rows: Vec<Resolved> = d.into_iter().map(|(r, _)| r).collect();
    crate::rprintln!("{}", boot_line(fnx, world, &rows));
    *RESOLVED.lock().unwrap_or_else(|e| e.into_inner()) = rows;
}

/// `velogb10_resolved_default{option,value,source} 1` per resolved option (nothing before an EXL3 boot).
pub fn metrics_append(o: &mut String) {
    let g = RESOLVED.lock().unwrap_or_else(|e| e.into_inner());
    if g.is_empty() { return; }
    o.push_str("# HELP velogb10_resolved_default The value each family/TP-scoped default resolved to on this boot (source auto = resolved by the server, set = given explicitly). kind=mode\n");
    o.push_str("# TYPE velogb10_resolved_default gauge\n");
    for r in g.iter() {
        o.push_str(&format!("velogb10_resolved_default{{option=\"{}\",value=\"{}\",source=\"{}\"}} 1\n",
                            r.flag, word(r.flag, &r.value), if r.auto { "auto" } else { "set" }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn none(_: &str) -> Option<String> { None }
    fn val<'a>(d: &'a [(Resolved, bool)], f: &str) -> &'a (Resolved, bool) { d.iter().find(|(r, _)| r.flag == f).unwrap() }

    #[test]
    fn flash_next_resolves_the_two_on_and_max_w_15() {
        for world in [1, 2, 4] {
            let d = decide(true, world, &none);
            assert!(d.iter().all(|(r, _)| r.flag != "exl3-mid-gemm"), "mid-gemm is opt-in, never resolved");
            for f in ["copy-span", "exl3-dense-resident"] {
                assert_eq!(val(&d, f), &(Resolved { flag: FLAGS.iter().find(|x| **x == f).unwrap(), value: "1".into(), auto: true }, true), "{f} at TP={world}");
            }
            assert_eq!(val(&d, "copy-span-max-w").0.value, "15");
            assert!(val(&d, "copy-span-max-w").1);
        }
    }

    #[test]
    fn other_families_write_nothing_family_scoped() {
        for world in [1, 2, 4] {
            let d = decide(false, world, &none);
            for f in ["copy-span", "copy-span-max-w", "exl3-dense-resident"] {
                let (r, w) = val(&d, f);
                assert!(!w, "{f}: no store write outside Flash-Next");
                assert!(r.auto);
            }
            assert_eq!(val(&d, "copy-span").0.value, "0");
        }
    }

    #[test]
    fn meter_resolves_by_tp_world() {
        for fnx in [true, false] {
            assert_eq!(val(&decide(fnx, 1, &none), "spec-arm-meter").0.value, "v2");
            assert_eq!(val(&decide(fnx, 2, &none), "spec-arm-meter").0.value, "v2");
            assert_eq!(val(&decide(fnx, 4, &none), "spec-arm-meter").0.value, "v1");
            assert_eq!(val(&decide(fnx, 8, &none), "spec-arm-meter").0.value, "v1");
        }
    }

    #[test]
    fn explicit_values_always_win() {
        let set = |f: &str| match f {
            "copy-span" => Some("0".to_string()),
            "exl3-dense-resident" => Some("0".to_string()),
            "copy-span-max-w" => Some("10".to_string()),
            "spec-arm-meter" => Some("v1".to_string()),
            _ => None,
        };
        let d = decide(true, 1, &set);
        assert_eq!(val(&d, "copy-span"), &(Resolved { flag: "copy-span", value: "0".into(), auto: false }, false));
        assert_eq!(val(&d, "exl3-dense-resident"), &(Resolved { flag: "exl3-dense-resident", value: "0".into(), auto: false }, false));
        assert_eq!(val(&d, "copy-span-max-w").0.value, "10");
        assert_eq!(val(&d, "spec-arm-meter"), &(Resolved { flag: "spec-arm-meter", value: "v1".into(), auto: false }, false));
        assert_eq!(val(&d, "copy-span-max-w").1, false, "explicit max-w kept");
        // an explicit on outside the family is kept too
        let on = |f: &str| (f == "copy-span").then(|| "1".to_string());
        assert_eq!(val(&decide(false, 2, &on), "copy-span").0, Resolved { flag: "copy-span", value: "1".into(), auto: false });
        let v2 = |f: &str| (f == "spec-arm-meter").then(|| "v2".to_string());
        assert_eq!(val(&decide(false, 4, &v2), "spec-arm-meter").0.value, "v2");
    }

    #[test]
    fn family_detection_reads_model_type() {
        let d = std::env::temp_dir().join(format!("v0710_fam_{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let w = |s: &str| std::fs::write(d.join("config.json"), s).unwrap();
        w(r#"{"model_type":"qwen4_exp"}"#);
        assert!(is_flash_next_dir(d.to_str().unwrap()));
        w(r#"{"model_type":"x","text_config":{"model_type":"qwen4_exp_text"}}"#);
        assert!(is_flash_next_dir(d.to_str().unwrap()));
        w(r#"{"model_type":"qwen3_5_moe","text_config":{"model_type":"qwen3_5_moe_text"}}"#);
        assert!(!is_flash_next_dir(d.to_str().unwrap()));
        std::fs::remove_dir_all(&d).unwrap();
        assert!(!is_flash_next_dir(d.to_str().unwrap()), "no config = not Flash-Next");
    }

    #[test]
    fn boot_line_names_every_option_and_source() {
        let rows: Vec<Resolved> = decide(true, 4, &|f: &str| (f == "exl3-dense-resident").then(|| "0".to_string()))
            .into_iter().map(|(r, _)| r).collect();
        let l = boot_line(true, 4, &rows);
        for s in ["--copy-span on (auto)", "--copy-span-max-w 15 (auto)", "--exl3-dense-resident off (set)",
                  "--spec-arm-meter v1 (auto)", "TP=4", "Flash-Next", "--exl3-mid-gemm stays opt-in"] {
            assert!(l.contains(s), "missing {s}: {l}");
        }
    }

    #[test]
    fn registry_rows_say_auto() {
        for f in ["copy-span", "exl3-dense-resident", "spec-arm-meter"] {
            assert_eq!(crate::opts::lookup(f).unwrap().def().def, "auto", "{f}");
        }
        assert!(crate::opts::lookup("copy-span-max-w").unwrap().def().help.contains("15 on the Flash-Next"));
        // the opt-in set stays off
        for f in ["exl3-mid-gemm", "copy-span-prefill", "copy-span-lanes", "exl3-sel-tc", "exl3-dense-pf-tc-qsa", "lane-wide-exact", "copy-span-record",
                  "prefill-pack", "turn-resume"] {
            let d = crate::opts::lookup(f).unwrap_or_else(|| panic!("{f}")).def().def;
            assert_eq!(d, "off", "{f} stays opt-in");
        }
        assert_eq!(crate::opts::lookup("lane-wide-verify").unwrap().def().def, "off");
    }
}
