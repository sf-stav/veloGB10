//! PSK (paired chain) knobs: tunables, the grid and config picks, cross-checks, the shape log and
//! `exl3_chain_pair`. See `mod.rs` for the file map.

use super::*;

// ---------------------------------------------------------------------------
// T4e / w3-PSK: persistent split-K chain GEMM (exl3_hmma_gemm_ks_p / exl3_hmma_gemm_ks_p_x2).
// Every fp32 partial is BITWISE the old exl3_hmma_gemm_ks(_x2) partial for EVERY grid size G:
// the logical items are exactly the old grid's CTAs (member, N128 tile, K slab at the SAME
// chain_ks), each computed by the same per-element sequence (kernels/exl3_bench.cu contract).
// The combine is untouched, so G is a pure schedule knob — the autotuner's T4e axis.
// p3 DEFAULT = OFF (preview/p4c @ 6300674, .14, 2026-09-27: PSK-off faster in every greedy class
// over 3 reps, -0.73..-0.93 ms/round — an in-round regression). OFF = the old (N/128, ks[, 2])
// grids everywhere, incl. REPRIME's wp22r1 row slabs.
//   --psk=1          enable the persistent split-K grids (the knobs below apply only then).
//   --psk-off=1      escape: the old (N/128, ks[, 2]) grids, every shape (wins over --psk=1).
//   --psk-g=<n>      G for every shape; or a per-shape list "KxN=G,KxNp=G,..." (p = the pair
//                       kernel), e.g. "2560x6144=240,2560x512p=96". Clamped to [1, items];
//                       unlisted shapes keep the built-in rule (psk_grid).
//   --psk-pf=<d>     trellis L2 prefetch distance in 8-step blocks (default 1; 0 = off).
//   --psk-xcheck=1   eager only: after each PSK launch the old kernel re-runs into a twin
//                       workspace and every fp32 partial is bit-compared (PSK_XCHECK lines).
// ---------------------------------------------------------------------------
pub(super) fn psk_off() -> bool {
    tune::get_cur(&T_PSK_ON) == 0
}
/// --psk-off=1: the explicit escape (as before p3; overrides --psk=1).
fn psk_escape() -> bool {
    crate::opts::var(crate::opt!("psk-off")).as_deref() == Ok("1")
}
/// --psk=1: opt in to the persistent split-K grids (p3: default OFF).
fn psk_opt_in() -> bool {
    crate::opts::var(crate::opt!("psk")).as_deref() == Ok("1")
}
fn psk_pf() -> i32 {
    tune::get_cur(&T_PSK_PF).clamp(0, 127)
}
// TUNE T0 (p3c): the PSK knobs. G is the T4e axis — a pure schedule knob at fixed ks (EXL3-KS-P
// sweeps G in {1, odd, 48-multiples, the default, items-1, items, > items} x M 1..16 x PF 0/1/2).
// p5e: PSK is an OPT-IN package (p3: default OFF, --psk=1 opts in, --psk-off=1 wins) — so
// the registry switch is psk.on (value 1 = the persistent grids), per width: bitwise at every G and
// M 1..16, so the tuner may re-evaluate it per (family, m) in-round (what its standalone estimate lacked).
pub(crate) const T_PSK_ON: tune::TunableDef = tune::TunableDef {
    slot: 24, id: "psk.on", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_CHAIN,
    domain: &[0, 1], default: 0, valid: tune::valid_any, xcheck: "--probe-exl3-binv EXL3-KS-P + --psk-xcheck",
    env: "--psk|--psk-off",
    // either alias set = the old rule exactly (on iff --psk == "1" and --psk-off != "1")
    env_parse: || (tune::env_str(crate::opt!("psk")).is_some() || tune::env_str(crate::opt!("psk-off")).is_some())
        .then(|| (psk_opt_in() && !psk_escape()) as i32),
    rev: 2, wp: "w3/PSK",
};
/// psk.g / psk.g_pair value that means "the legacy --psk-g parse decides" (the env alias: a
/// global G or a per-shape "KxN=G,KxNp=G" list — context-dependent, so the registry carries only
/// this marker and psk_grid applies the old parse verbatim).
const PSK_G_ENV: i32 = -1;
/// G candidates: 0 = the built-in rule min(items, slots); else a CTA count (clamped to [1, items]).
const PSK_G_DOMAIN: &[i32] = &[0, 48, 96, 144, 192, 240, 288, 336, 384];
fn psk_g_valid(cx: &tune::Ctx, v: i32, members: u32) -> bool {
    if v <= 0 { return true; }
    let ks = chain_ks(cx.k, cx.n);
    let items = ((cx.n.max(0) as u32) / 128) * ks * members;
    ks > 1 && (v as u32) <= items
}
pub(crate) const T_PSK_G: tune::TunableDef = tune::TunableDef {
    slot: 25, id: "psk.g", class: tune::Class::S, scope: tune::Scope::Shape, fams: FAMS_CHAIN,
    domain: PSK_G_DOMAIN, default: 0, valid: |cx, v| psk_g_valid(cx, v, 1),
    xcheck: "--probe-exl3-binv EXL3-KS-P (G sweep) + --psk-xcheck",
    env: "--psk-g", env_parse: || tune::env_str(crate::opt!("psk-g")).map(|_| PSK_G_ENV), rev: 1, wp: "w3/PSK",
};
pub(crate) const T_PSK_G2: tune::TunableDef = tune::TunableDef {
    slot: 26, id: "psk.g_pair", class: tune::Class::S, scope: tune::Scope::Shape, fams: FAMS_CHAIN,
    domain: PSK_G_DOMAIN, default: 0, valid: |cx, v| psk_g_valid(cx, v, 2),
    xcheck: "--probe-exl3-binv EXL3-KS-P (pair G sweep) + --psk-xcheck",
    env: "--psk-g", env_parse: || tune::env_str(crate::opt!("psk-g")).map(|_| PSK_G_ENV), rev: 1, wp: "w3/PSK",
};
pub(crate) const T_PSK_PF: tune::TunableDef = tune::TunableDef {
    // p5e: per width (it pairs with psk.on in the tuner's joint grid; bitwise for every PF)
    slot: 27, id: "psk.pf", class: tune::Class::S, scope: tune::Scope::Width, fams: FAMS_CHAIN,
    domain: &[0, 1, 2, 3], default: 1, valid: tune::valid_any, xcheck: "--probe-exl3-binv EXL3-KS-P (PF 0/1/2)",
    env: "--psk-pf",
    env_parse: || tune::env_str(crate::opt!("psk-pf")).and_then(|v| v.trim().parse::<i32>().ok()).map(|v| v.clamp(0, 127)),
    rev: 1, wp: "w3/PSK",
};
/// Families whose graphs run split-K dense chains (every forward + the re-prime's fc slabs).
pub(crate) const FAMS_CHAIN: &[tune::Fam] = &[tune::Fam::Step, tune::Fam::Verify, tune::Fam::DraftPass,
                                              tune::Fam::DraftChain, tune::Fam::Reprime, tune::Fam::Util];
pub(super) fn psk_xcheck() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| crate::opts::var(crate::opt!("psk-xcheck")).map_or(false, |v| !v.is_empty() && v != "0"))
}
/// --psk-g parse: (global G, per-shape [(K, N, pair, G)]). Malformed entries are ignored
/// (and reported once), never fatal.
fn psk_g_env() -> &'static (Option<u32>, Vec<(i32, i32, bool, u32)>) {
    static V: std::sync::OnceLock<(Option<u32>, Vec<(i32, i32, bool, u32)>)> = std::sync::OnceLock::new();
    V.get_or_init(|| {
        let Ok(raw) = crate::opts::var(crate::opt!("psk-g")) else { return (None, Vec::new()) };
        if let Ok(g) = raw.trim().parse::<u32>() { return (Some(g.max(1)), Vec::new()); }
        let mut v = Vec::new();
        for ent in raw.split(',').map(str::trim).filter(|e| !e.is_empty()) {
            let parsed = (|| {
                let (shape, g) = ent.split_once('=')?;
                let (k, n) = shape.trim().split_once(['x', 'X'])?;
                let (n, pair) = match n.trim().strip_suffix(['p', 'P']) { Some(n2) => (n2, true), None => (n.trim(), false) };
                Some((k.trim().parse::<i32>().ok()?, n.parse::<i32>().ok()?, pair, g.trim().parse::<u32>().ok()?.max(1)))
            })();
            match parsed {
                Some(e) => v.push(e),
                None => crate::dprintln!("EXL3-FWD PSK: ignoring malformed --psk-g entry {ent:?} (want KxN=G or KxNp=G)"),
            }
        }
        (None, v)
    })
}
/// Co-resident CTAs (occupancy API x SMs) of the two PSK kernels as compiled; 0 = query failed
/// (stale PTX) => that kernel is never launched (the old grid runs, consistently for all m).
pub(super) fn psk_slots(dev: &Arc<CudaDevice>) -> (u32, u32) {
    static V: std::sync::OnceLock<(u32, u32)> = std::sync::OnceLock::new();
    *V.get_or_init(|| (wp11_capacity(dev, "exl3_hmma_gemm_ks_p"), wp11_capacity(dev, "exl3_hmma_gemm_ks_p_x2")))
}
/// Built-in grid rule: items = (N/128) * ks * members; G = min(items, slots) — one full wave of
/// persistent CTAs (slots = occupancy x 48 SMs, a multiple of 48), never more CTAs than items.
/// 0 = run the old kernel (PSK off, ks <= 1, or no measured capacity).
pub(super) fn psk_grid(dev: &Arc<CudaDevice>, k: i32, n: i32, bits: i32, ks: u32, pair: bool) -> u32 {
    // P7 B6 width guard: exl3_hmma_gemm_ks_p(_x2) carries bits-keyed cases 2..=6 only. A width
    // outside that (K=1 -> bits=1, K=7/8 -> bits 7/8) used to launch anyway and hit
    // `default: break` -> NaN partials (the B1 class; --probe-exl3-binv caught it on the real
    // 2.0bpw pack before case 2 existed). Fractional packs cannot reach a Quad (the loader
    // refuses k2 5/7), so a bits-only guard is sufficient. Refuse here instead: the chain keeps
    // the plain split-K grid (exl3_hmma_gemm_ks, correct at every width the loader admits),
    // one boot line says so, /metrics counts it (velogb10_exl3_psk_width_fallback_total).
    if psk_off() || ks <= 1 { return 0; }
    if !(2..=6).contains(&bits) {
        static WARNED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        WARNED.get_or_init(|| {
            crate::dprintln!(
                "EXL3-FWD PSK: bits={bits} outside the persistent-grid bodies (2..=6) — \
                 keeping the plain split-K grid (exl3_hmma_gemm_ks) for this pack's chains"
            );
        });
        crate::metrics::exl3_psk_width_fallback(bits);
        return 0;
    }
    let items = ((n.max(0) as u32) / 128) * ks * if pair { 2 } else { 1 };
    let (s1, s2) = psk_slots(dev);
    let slots = if pair { s2 } else { s1 };
    if slots == 0 || items == 0 { return 0; }
    // TUNE: psk.g / psk.g_pair per (K, N, bits); the env alias (PSK_G_ENV) keeps the old parse
    let v = tune::get(if pair { &T_PSK_G2 } else { &T_PSK_G }, &tune::current().shape(k, n, bits));
    let g = if v == PSK_G_ENV {
        let env = psk_g_env();
        env.1.iter().find(|e| e.0 == k && e.1 == n && e.2 == pair).map(|e| e.3)
            .or(env.0)
            .unwrap_or_else(|| items.min(slots))
    } else if v > 0 {
        v as u32
    } else {
        items.min(slots)
    };
    g.clamp(1, items)
}
/// Kernel launch word: bits | ks << 8 | pf_d << 24 (mirrors PSK_CFG_* in exl3_bench.cu).
pub(super) fn psk_cfg(bits: i32, ks: u32) -> i32 {
    (bits & 0xFF) | (((ks as i32) & 0xFFFF) << 8) | ((psk_pf() & 0x7F) << 24)
}
/// Bit-compare report for --psk-xcheck (fp32 partials, raw bits).
fn psk_xcheck_report(site: &str, new: &[f32], old: &[f32]) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    static BAD: AtomicUsize = AtomicUsize::new(0);
    let n = new.len().min(old.len());
    let mut mism = 0usize;
    let mut first: Option<usize> = None;
    for i in 0..n {
        if new[i].to_bits() != old[i].to_bits() {
            mism += 1;
            if first.is_none() { first = Some(i); }
        }
    }
    let k = N.fetch_add(1, Ordering::SeqCst) + 1;
    if mism > 0 { BAD.fetch_add(1, Ordering::SeqCst); }
    if mism > 0 || k % 500 == 0 || k == 1 {
        let fi = first.map_or(String::from("-"), |i| format!("{i} (new {:e} old {:e})", new[i], old[i]));
        crate::dprintln!("PSK_XCHECK {site}: checks {k} bad {} | this: n {n} mism {mism} first {fi}",
                 BAD.load(Ordering::SeqCst));
    }
}
/// --psk-xcheck (single): the old exl3_hmma_gemm_ks into a twin workspace, bit-diffed against
/// the PSK partials already in ws[..ks*m*n]. Skipped (noted once) while a graph is capturing.
/// `xh` = device address of the chain's first A row (p3c: raw, so REPRIME's wp22r1 row slabs
/// can check their own slab).
pub(super) fn psk_xcheck_single(l: &Launcher, q: &Quad, xh: u64, ws: &CudaSlice<f32>, m: usize,
                     g: u32) -> Result<()> {
    if stream_capturing(l) {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| crate::dprintln!("PSK_XCHECK: skipped inside graph capture (run eager: --exl3-no-graph=1)"));
        return Ok(());
    }
    let need = q.ks as usize * m * q.n as usize;
    let twin = l.dev.alloc_zeros::<f32>(need)?;       // every entry is written by the old kernel
    xqlaunch!(l, "exl3_hmma_gemm_ks", ((q.n as u32) / 128, q.ks, 1), (256, 1, 1), 0,
             (&q.tr, xh, &twin, m as i32, q.k, q.n, q.k2))?;
    l.dev.synchronize()?;
    let mut newp = vec![0f32; need];
    l.dev.dtoh_sync_copy_into(&ws.slice(0..need), &mut newp)?;
    let oldp = l.dev.dtoh_sync_copy(&twin)?;
    psk_xcheck_report(&format!("K{}xN{} b{} ks{} m{m} G{g}", q.k, q.n, q.bits, q.ks), &newp, &oldp);
    Ok(())
}
/// --psk-xcheck (pair): the old exl3_hmma_gemm_ks_x2 into twins, bit-diffed per member.
#[allow(clippy::too_many_arguments)]
fn psk_xcheck_pair(l: &Launcher, q0: &Quad, q1: &Quad, xh0: &CudaSlice<u16>, xh1: &CudaSlice<u16>,
                   ws0: &CudaSlice<f32>, ws1: &CudaSlice<f32>, m: usize, g: u32) -> Result<()> {
    if stream_capturing(l) {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| crate::dprintln!("PSK_XCHECK: skipped inside graph capture (run eager: --exl3-no-graph=1)"));
        return Ok(());
    }
    let need = q0.ks as usize * m * q0.n as usize;
    let t0 = l.dev.alloc_zeros::<f32>(need)?;
    let t1 = l.dev.alloc_zeros::<f32>(need)?;
    xqlaunch!(l, "exl3_hmma_gemm_ks_x2", ((q0.n as u32) / 128, q0.ks, 2), (256, 1, 1), 0,
             (&q0.tr, &q1.tr, xh0, xh1, &t0, &t1, m as i32, q0.k, q0.n, q0.k2))?;
    l.dev.synchronize()?;
    let (mut n0, mut n1) = (vec![0f32; need], vec![0f32; need]);
    l.dev.dtoh_sync_copy_into(&ws0.slice(0..need), &mut n0)?;
    l.dev.dtoh_sync_copy_into(&ws1.slice(0..need), &mut n1)?;
    let (o0, o1) = (l.dev.dtoh_sync_copy(&t0)?, l.dev.dtoh_sync_copy(&t1)?);
    let site = format!("pair K{}xN{} b{} ks{} m{m} G{g}", q0.k, q0.n, q0.bits, q0.ks);
    psk_xcheck_report(&format!("{site} [0]"), &n0, &o0);
    psk_xcheck_report(&format!("{site} [1]"), &n1, &o1);
    Ok(())
}
/// Load-time log: the chosen G for every chain shape this model splits (singles, and the k/v +
/// shared gate/up pairs), once.
pub(super) fn psk_log_shapes(dev: &Arc<CudaDevice>, layers: &[LayerDev], mtp: Option<&DraftHead>) {
    use std::collections::BTreeSet;
    if psk_off() {
        let why = if psk_escape() {
            if psk_opt_in() { "--psk-off=1 overrides --psk=1" } else { "--psk-off=1" }
        } else { "default" };
        let knobs: Vec<String> = [crate::opt!("psk-g"), crate::opt!("psk-pf"), crate::opt!("psk-xcheck")].into_iter()
            .filter(|k| crate::opts::var_os(*k).is_some()).map(|k| k.to_string()).collect();
        crate::dprintln!("EXL3-FWD PSK: OFF ({why}) — split-K chains run the old (N/128, ks) grids; \
                  --psk=1 = persistent split-K ON{}",
                 if knobs.is_empty() { String::new() } else { format!(" ({} ignored while OFF)", knobs.join("/")) });
        // TUNE T0c: a table value or the --autotune overlay can turn PSK on at some width — measure
        // the kernels' co-residency now, outside any graph capture
        if tune::reachable(&T_PSK_ON).contains(&1) { let _ = psk_slots(dev); }
        return;
    }
    let (s1, s2) = psk_slots(dev);
    let mut singles: BTreeSet<(i32, i32, i32, u32)> = BTreeSet::new();
    let mut pairs: BTreeSet<(i32, i32, i32, u32)> = BTreeSet::new();
    let mut visit = |attn: Option<&AttnLayer>, gdn: Option<&GdnLayer>, moe: &MoeLayer| {
        let mut one = |q: &Quad| { if q.ks > 1 { singles.insert((q.k, q.n, q.bits, q.ks)); } };
        if let Some(g) = gdn { one(&g.in_qkv); one(&g.in_z); one(&g.out_proj); }
        if let Some(a) = attn {
            one(&a.q_proj); one(&a.o_proj); one(&a.k_proj); one(&a.v_proj);
            if let Some(ix) = &a.idx { one(&ix.qk); }
        }
        one(&moe.sh_gate); one(&moe.sh_up); one(&moe.sh_down);
        let mut two = |a: &Quad, b: &Quad| {
            if a.ks > 1 && a.ks == b.ks && a.k == b.k && a.n == b.n && a.bits == b.bits {
                pairs.insert((a.k, a.n, a.bits, a.ks));
            }
        };
        if let Some(a) = attn { two(&a.k_proj, &a.v_proj); }
        two(&moe.sh_gate, &moe.sh_up);
    };
    for l in layers {
        match &l.mixer {
            Mixer::Gdn(g) => visit(None, Some(g), &l.moe),
            Mixer::Attn(a) => visit(Some(a), None, &l.moe),
        }
    }
    if let Some(h) = mtp {
        visit(Some(&h.attn), None, &h.moe);
        for q in [&h.fc_hidden, &h.fc_embed] { if q.ks > 1 { singles.insert((q.k, q.n, q.bits, q.ks)); } }
    }
    crate::dprintln!("EXL3-FWD PSK: persistent split-K ON (--psk=1; default OFF) — slots single {s1} / pair {s2} \
              (occupancy x SMs), trellis prefetch {} block(s) ahead; --psk-off=1 = old grids", psk_pf());
    for (set, pair) in [(&singles, false), (&pairs, true)] {
        for &(k, n, bits, ks) in set.iter() {
            let items = ((n as u32) / 128) * ks * if pair { 2 } else { 1 };
            let g = psk_grid(dev, k, n, bits, ks, pair);
            let slots = if pair { s2 } else { s1 };
            let rounds = if g > 0 { (items + g - 1) / g } else { 0 };
            crate::dprintln!("  PSK {} K={k:<5} N={n:<6} b{bits} ks={ks:<3} items {items:>4} -> G {g:>4}{}",
                     if pair { "pair  " } else { "single" },
                     if g == 0 { format!(" (old grid: capacity {slots})") }
                     else { format!(" ({rounds} item(s)/CTA max, {:.2} waves of {slots})", items as f32 / slots.max(1) as f32) });
        }
    }
}

/// S-A3-o C1: two same-input decode/verify chains (q0, q1 read the same x) in ONE launch
/// per stage (grid.z = member). Each member keeps its own suh/svh/trellis and the SAME ks
/// partition + ascending combine as exl3_chain, through the same device bodies => each
/// output is bitwise its exl3_chain result. Falls back to two exl3_chain calls unless both
/// members share (K, N, bits, ks > 1), m <= 16 and the pair buffers fit.
/// --exl3-chain-pair=0 = A/B escape (always two chains).
#[allow(clippy::too_many_arguments)]
pub(super) fn exl3_chain_pair(l: &Launcher, q0: &Quad, q1: &Quad, x: &CudaSlice<u16>,
                   xh0: &mut CudaSlice<u16>, yraw0: &mut CudaSlice<u16>, ws0: &CudaSlice<f32>,
                   xh1: &mut CudaSlice<u16>, yraw1: &mut CudaSlice<u16>, ws1: &CudaSlice<f32>,
                   y0: &mut CudaSlice<u16>, y1: &mut CudaSlice<u16>, m: usize) -> Result<()> {
    if q0.fx.is_some() || q1.fx.is_some() {
        // FP4L: one paired launch (bitwise per member), else two FP4L chains in order (ws0 reused stream-ordered)
        if m * q0.k as usize <= x.len() && m * q0.n as usize <= y0.len() && m * q1.n as usize <= y1.len()
            && fp4l_fwd::fx_pair(l, q0, q1, *x.device_ptr() as u64, *y0.device_ptr() as u64, *y1.device_ptr() as u64,
                                 (*ws0.device_ptr() as u64, ws0.len()), (*ws1.device_ptr() as u64, ws1.len()), m)? {
            return Ok(());
        }
        exl3_chain_impl(l, q0, x, xh0, yraw0, y0, m, ws0, None)?;
        return exl3_chain_impl(l, q1, x, xh0, yraw0, y1, m, ws0, None);
    }
    let off = tune::get(&T_CHAIN_PAIR, &l.cx().shape(q0.k, q0.n, q0.bits)) == 0;
    if !off && m <= 16 && q0.ks > 1 && q1.ks > 1 && dense::dense_on() {
        // W4/DENSE: both members in ONE launch (suh + ks GEMM + combine + svh each), bitwise the
        // x2 path below. Ok(false) = not taken, nothing launched.
        use dense::{dptr, DOut};
        let o0 = DOut { y: dptr(y0), y_len: y0.len(), yraw: dptr(yraw0), yraw_len: yraw0.len(), xh: dptr(xh0), xh_len: xh0.len() };
        let o1 = DOut { y: dptr(y1), y_len: y1.len(), yraw: dptr(yraw1), yraw_len: yraw1.len(), xh: dptr(xh1), xh_len: xh1.len() };
        if dense::dense_chains(l, &[q0, q1], &[o0, o1], (dptr(x), x.len()), (0, 0), m,
                               &[(dptr(ws0), ws0.len()), (dptr(ws1), ws1.len())])? {
            return Ok(());
        }
    }
    let need_ws = q0.ks as usize * m * q0.n as usize;
    let ok = !off && m <= 16 && q0.ks > 1 && q0.ks == q1.ks && q0.k == q1.k && q0.n == q1.n
        && q0.bits == q1.bits && ((q0.k as u32) / 16) % q0.ks == 0
        && need_ws <= ws0.len() && need_ws <= ws1.len()
        && m * q0.k as usize <= xh1.len() && m * q0.n as usize <= yraw1.len();
    if !ok {
        exl3_chain(l, q0, x, xh0, yraw0, y0, m, ws0)?;
        return exl3_chain(l, q1, x, xh0, yraw0, y1, m, ws0);
    }
    let (k, n) = (q0.k, q0.n);
    xqlaunch!(l, "exl3_had_suh_x2", (m as u32 * (k as u32 / 128), 1, 2), (32, 1, 1), 0,
             (x, &q0.suh, &q1.suh, &mut *xh0, &mut *xh1, k))?;
    // w3-PSK: persistent pair grid (bitwise == exl3_hmma_gemm_ks_x2 for every G); 0 = old grid.
    // PSK is opt-in (--psk=1) and keeps the 3-launch tail (the WP21 fixup is not in its body).
    let g = psk_grid(l.dev, k, n, q0.bits, q0.ks, true);
    if g > 0 {
        // trailing &q0.tr / &q1.tr = the prefetch bases (same addresses; see PSK_CFG_* note)
        xqlaunch!(l, "exl3_hmma_gemm_ks_p_x2", (g, 1, 1), (256, 1, 1), 0,
                 (&q0.tr, &q1.tr, &*xh0, &*xh1, ws0, ws1, m as i32, k, n, psk_cfg(q0.bits, q0.ks),
                  &q0.tr, &q1.tr))?;
        if psk_xcheck() { psk_xcheck_pair(l, q0, q1, &*xh0, &*xh1, ws0, ws1, m, g)?; }
        let mn = (m * n as usize) as u32;
        xqlaunch!(l, "exl3_ks_combine_x2", ((mn + 255) / 256, 1, 2), (256, 1, 1), 0,
                 (ws0, ws1, &mut *yraw0, &mut *yraw1, m as i32, n, q0.ks as i32))?;
        xqlaunch!(l, "exl3_had_svh_x2", (m as u32 * (n as u32 / 128), 1, 2), (32, 1, 1), 0,
                 (&*yraw0, &*yraw1, &q0.svh, &q1.svh, &mut *y0, &mut *y1, n))?;
    } else if wp21_on(q0.ks) && wp21_pair_ok(q0, q1, n) && wp21_fx_width_ok(q0) && wp21_fx_width_ok(q1) {
        // WP21: ks GEMM + ascending combine + H128 svh in ONE launch (last-arriving CTA per
        // N-tile per member does the fixup) — bitwise the 3-launch path below.
        wp21_occupancy_once(l);
        wp21_launch_fx_x2(l, q0, q1, &*xh0, &*xh1, ws0, ws1, &mut *y0, &mut *y1, m)?;
        if wp21_xcheck() {
            wp21_check_pair(l, q0, q1, xh0, yraw0, ws0, xh1, yraw1, ws1, y0, y1, m)?;
        }
    } else {
        xqlaunch!(l, "exl3_hmma_gemm_ks_x2", ((n as u32) / 128, q0.ks, 2), (256, 1, 1), 0,
                 (&q0.tr, &q1.tr, &*xh0, &*xh1, ws0, ws1, m as i32, k, n, q0.k2))?;
        let mn = (m * n as usize) as u32;
        xqlaunch!(l, "exl3_ks_combine_x2", ((mn + 255) / 256, 1, 2), (256, 1, 1), 0,
                 (ws0, ws1, &mut *yraw0, &mut *yraw1, m as i32, n, q0.ks as i32))?;
        xqlaunch!(l, "exl3_had_svh_x2", (m as u32 * (n as u32 / 128), 1, 2), (32, 1, 1), 0,
                 (&*yraw0, &*yraw1, &q0.svh, &q1.svh, &mut *y0, &mut *y1, n))?;
    }
    if crate::opts::var(crate::opt!("exl3-chain-pair-check")).is_ok() {
        // diagnostic (eager only: --exl3-no-graph=1): pair outputs vs two plain chains
        let mn = m * n as usize;
        l.dev.synchronize()?;
        let p0 = l.dev.dtoh_sync_copy(&*y0)?;
        let p1 = l.dev.dtoh_sync_copy(&*y1)?;
        exl3_chain(l, q0, x, xh0, yraw0, y0, m, ws0)?;
        exl3_chain(l, q1, x, xh0, yraw0, y1, m, ws0)?;
        l.dev.synchronize()?;
        let r0 = l.dev.dtoh_sync_copy(&*y0)?;
        let r1 = l.dev.dtoh_sync_copy(&*y1)?;
        let bad = (0..mn).filter(|&i| p0[i] != r0[i] || p1[i] != r1[i]).count();
        use std::sync::atomic::{AtomicUsize, Ordering};
        static N: AtomicUsize = AtomicUsize::new(0);
        static BAD: AtomicUsize = AtomicUsize::new(0);
        let c = N.fetch_add(1, Ordering::SeqCst) + 1;
        if bad > 0 { BAD.fetch_add(1, Ordering::SeqCst); }
        if bad > 0 || c % 500 == 0 {
            crate::drprintln!("CHAIN_PAIR_CHECK: calls {c} bad {} (this: m {m} K {k} N {n} mism {bad})", BAD.load(Ordering::SeqCst));
        }
    }
    Ok(())
}
