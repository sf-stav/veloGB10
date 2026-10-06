//! W4/SMALL (decode wave 4, package SMALL, 2026-09-27) — host side of five latency-bound
//! small-kernel levers. Every lever launches a BITWISE twin of the kernel(s) it replaces (the
//! device-side contracts are in the "W4/SMALL" header of kernels/exl3_bench.cu):
//!
//!   router  xq_router_fold -> xq_router_fold_w4: persistent (1 CTA/SM x 768), x staged once per
//!           CTA in smem (the served fold walked 5*m serialized x round trips per thread).
//!           Serves the verify, the draft passes and the decode step (fn moe).
//!   gdn     per-layer xq_conv_commit + xq_gdn_commit_ring (+ xq_ple_ring_commit) -> ONE
//!           xq_gdn_commit_all_w4 launch for every GDN layer (verify_commit).
//!   dattn   xq_attn_dense_acc4 / _acc1 -> _w4 twins: 5-stage cp.async V ring (was 2).
//!   ple     xq_gemm_f16_rows_km -> xq_gemm_f16_rows_km_w4m<M>: x staged in smem, exact-M body.
//!   draft   the m = 1 head attention: xq_attn_prep (grid nkv) -> xq_attn_dense_prep (grid nh),
//!           xq_attn_dense_splitk4 -> _w4 (smem K|V ring, both heads of a warp in one walk),
//!           xq_attn_sel_combine -> _w4 (pm/pl staged, one __expf per split).
//!
//! Escapes (diagnostic A/B + bisect only — AGENTS §1b): --w4s-off=1|all restores every old
//! launch; or a comma list of items (router,gdn,dattn,ple,draft). Checks: --w4s-xcheck=1|all
//! or a list — EAGER launches only (--exl3-no-graph=1): every changed launch re-runs its old
//! twin on the same inputs and diffs the outputs bitwise, printing `W4S_XCHECK <item> ...
//! bad <n>` per call (first 4 calls, every mismatch, then powers of two).

use super::*;

pub(super) const W4S_ROUTER: u8 = 1;
pub(super) const W4S_GDN: u8 = 2;
pub(super) const W4S_DATTN: u8 = 4;
pub(super) const W4S_PLE: u8 = 8;
pub(super) const W4S_DRAFT: u8 = 16;
const W4S_ALL: u8 = 0x1f;
const W4S_NAMES: [(&str, u8); 5] =
    [("router", W4S_ROUTER), ("gdn", W4S_GDN), ("dattn", W4S_DATTN), ("ple", W4S_PLE), ("draft", W4S_DRAFT)];

fn w4s_parse(var: crate::opts::OptId) -> u8 {
    match crate::opts::var(var) {
        Err(_) => 0,
        Ok(s) => {
            let mut v = 0u8;
            for t in s.split(',').map(|t| t.trim().to_ascii_lowercase()).filter(|t| !t.is_empty()) {
                match t.as_str() {
                    "1" | "all" => v |= W4S_ALL,
                    "0" | "none" => {}
                    _ => match W4S_NAMES.iter().find(|(n, _)| *n == t) {
                        Some((_, b)) => v |= b,
                        None => eprintln!("W4S: {var}: unknown item '{t}' ignored"),
                    },
                }
            }
            v
        }
    }
}

fn w4s_names(v: u8) -> String {
    let s: Vec<&str> = W4S_NAMES.iter().filter(|(_, b)| v & b != 0).map(|(n, _)| *n).collect();
    if s.is_empty() { "none".into() } else { s.join(",") }
}

/// The items ON (default: all). --w4s-off=1|all|<list> turns items off.
/// TUNE T0c (p5e): each item is a registry entry (w4s.router / gdn / dattn / ple / draft, per width —
/// every one a bitwise twin); the alias is parsed once exactly as before and wins; the boot line
/// (the feature marker) still prints once, on the first lookup, from the alias.
pub(super) fn w4s_on(item: u8) -> bool {
    w4s_announce();
    let t = match item {
        W4S_ROUTER => &T_W4S_ROUTER,
        W4S_GDN => &T_W4S_GDN,
        W4S_DATTN => &T_W4S_DATTN,
        W4S_PLE => &T_W4S_PLE,
        W4S_DRAFT => &T_W4S_DRAFT,
        _ => return false,
    };
    tune::get_cur(t) != 0
}

/// --w4s-off as the items-ON mask (None = unset), parsed once (its warnings once).
fn w4s_env_on() -> Option<u8> {
    static M: std::sync::OnceLock<Option<u8>> = std::sync::OnceLock::new();
    *M.get_or_init(|| crate::opts::var(crate::opt!("w4s-off")).ok().map(|_| W4S_ALL & !w4s_parse(crate::opt!("w4s-off"))))
}

/// The W4S boot line (unchanged text), once.
fn w4s_announce() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let on = w4s_env_on().unwrap_or(W4S_ALL);
        let xc = w4s_parse(crate::opt!("w4s-xcheck"));
        println!("W4S: small-kernel levers ON = [{}] (--w4s-off=1|all or router,gdn,dattn,ple,draft = old kernels); \
                  xcheck [{}] (--w4s-xcheck, eager launches only)", w4s_names(on), w4s_names(xc));
    });
}

const fn w4s_def(slot: u16, id: &'static str, fams: &'static [tune::Fam], xcheck: &'static str,
                 env_parse: fn() -> Option<i32>) -> tune::TunableDef {
    tune::TunableDef {
        slot, id, class: tune::Class::S, scope: tune::Scope::Width, fams, domain: &[0, 1], default: 1,
        valid: tune::valid_any, xcheck, env: "--w4s-off", env_parse, rev: 1, wp: "W4/SMALL",
    }
}
pub(crate) const T_W4S_ROUTER: tune::TunableDef = w4s_def(60, "w4s.router", FAMS_DECODE,
    "--probe-exl3-binv EXL3-ROUTERFOLD (fold_w4) + --w4s-xcheck=router",
    || w4s_env_on().map(|m| (m & W4S_ROUTER != 0) as i32));
pub(crate) const T_W4S_GDN: tune::TunableDef = w4s_def(61, "w4s.gdn", &[tune::Fam::Verify],
    "--w4s-xcheck=gdn (commit_all vs the per-layer commits, eager)",
    || w4s_env_on().map(|m| (m & W4S_GDN != 0) as i32));
pub(crate) const T_W4S_DATTN: tune::TunableDef = w4s_def(62, "w4s.dattn", &[tune::Fam::Step, tune::Fam::Verify],
    "--probe-exl3-binv EXL3-ATTN (acc4_w4 / acc1_w4 vs v2) + --w4s-xcheck=dattn",
    || w4s_env_on().map(|m| (m & W4S_DATTN != 0) as i32));
pub(crate) const T_W4S_PLE: tune::TunableDef = w4s_def(63, "w4s.ple", &[tune::Fam::Step, tune::Fam::Verify],
    "--probe-exl3-binv EXL3-F16ROWS-KM (w4m<M>) + --w4s-xcheck=ple",
    || w4s_env_on().map(|m| (m & W4S_PLE != 0) as i32));
pub(crate) const T_W4S_DRAFT: tune::TunableDef = w4s_def(64, "w4s.draft", &[tune::Fam::DraftPass, tune::Fam::DraftChain],
    "--w4s-xcheck=draft (head attention prep / splitk4_w4 / combine_w4, eager)",
    || w4s_env_on().map(|m| (m & W4S_DRAFT != 0) as i32));

/// --w4s-xcheck=1|all|<list>: per-call new-vs-old bitwise checks (eager launches only).
pub(super) fn w4s_xcheck(item: u8) -> bool {
    static X: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *X.get_or_init(|| w4s_parse(crate::opt!("w4s-xcheck"))) & item != 0
}

/// Report one check (first 4 calls per item, every mismatch, then powers of two).
fn w4s_report(item: u8, what: &str, bad: usize, tot: usize, first: String) {
    use std::sync::atomic::AtomicUsize;
    static N: [AtomicUsize; 5] = [AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0),
                                  AtomicUsize::new(0), AtomicUsize::new(0)];
    let k = (0..5).find(|&i| W4S_NAMES[i].1 == item).unwrap_or(0);
    let c = N[k].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if c < 4 || bad > 0 || c.is_power_of_two() {
        println!("W4S_XCHECK {} #{c} {what}: bad {bad} of {tot}{first}", W4S_NAMES[k].0);
    }
}

fn diff_u16(a: &[u16], b: &[u16]) -> (usize, String) {
    let n = a.len().min(b.len());
    let bad = (0..n).filter(|&i| a[i] != b[i]).count();
    let first = (0..n).find(|&i| a[i] != b[i])
        .map(|i| format!(" (first idx {i}: new {:04x} old {:04x})", a[i], b[i])).unwrap_or_default();
    (bad, first)
}

fn diff_f32(a: &[f32], b: &[f32]) -> (usize, String) {
    let n = a.len().min(b.len());
    let bad = (0..n).filter(|&i| a[i].to_bits() != b[i].to_bits()).count();
    let first = (0..n).find(|&i| a[i].to_bits() != b[i].to_bits())
        .map(|i| format!(" (first idx {i}: new {:e} old {:e})", a[i], b[i])).unwrap_or_default();
    (bad, first)
}

/// (5) persistent router fold geometry: 768 threads = 12 experts x 64 slices per CTA.
pub(super) const W4R_EPC: usize = 12;
pub(super) const W4R_NT: u32 = (W4R_EPC * 64) as u32;
const W4R_RS: usize = W4R_EPC * 72 + 1;   // == XQ_W4R_RS

/// Grid of xq_router_fold_w4: one CTA per SM, at least ceil(ne / 12) CTAs.
pub(super) fn w4r_grid(dev: &Arc<CudaDevice>, ne: usize) -> u32 {
    let sms = lmh_sm_count(dev) as usize;
    sms.max(ne.div_ceil(W4R_EPC)).min(ne.max(1)) as u32
}

/// Dynamic smem of xq_router_fold_w4 (x staging | partials | tail scratch, aliased).
pub(super) fn w4r_smem(m: usize, h: usize) -> usize {
    (m * h * 2).max(m * W4R_RS * 4).max((512 + 2 * 8 * 16 + 8) * 4)
}

/// (6) the device table xq_gdn_commit_all_w4 reads: [S_0 .. S_{L-1}, conv_0 .. conv_{L-1}].
pub(super) fn w4s_build_lptr(dev: &Arc<CudaDevice>, s_state: &[CudaSlice<f32>], conv_state: &[CudaSlice<f32>])
                             -> Result<CudaSlice<u64>> {
    let mut v: Vec<u64> = s_state.iter().map(|b| *b.device_ptr() as u64).collect();
    v.extend(conv_state.iter().map(|b| *b.device_ptr() as u64));
    if v.is_empty() { v.push(0); }
    Ok(dev.htod_sync_copy(&v)?)
}

impl FwdModel {
    /// (5) The routing fold through xq_router_fold_w4 (same outputs as xq_router_fold, bitwise).
    /// Ok(false) = item off or not eligible: the caller launches the old fold.
    pub(super) fn w4s_router_fold(&self, l: &Launcher, sc: &mut Scratch, moe: &MoeLayer, m: usize) -> Result<bool> {
        let h = self.cfg.hidden_size;
        let ne = self.cfg.num_experts;
        let topk = self.cfg.num_experts_per_tok;
        if !w4s_on(W4S_ROUTER) { return Ok(false); }
        let g = w4r_grid(&self.dev, ne) as usize;
        let smem = w4r_smem(m, h);
        if m == 0 || m > 8 || ne > 512 || m * topk > 128 || topk > 16 || h % 64 != 0 || (h / 64) % 8 != 0
            || (m * h) % 8 != 0 || ne.div_ceil(g) > W4R_EPC || smem > 48 * 1024 {
            return Ok(false);
        }
        let f = xq_raw_fn("xq_router_fold_w4")
            .ok_or_else(|| anyhow::anyhow!("xq_router_fold_w4 raw fn unavailable (stale PTX?)"))?;
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| println!("W4S router: xq_router_fold_w4 ON (grid {g} x {W4R_NT}, x in smem; --w4s-off=router = xq_router_fold)"));
        let mut a_lg = *sc.logits_r.device_ptr() as u64;
        let mut a_w = *moe.router.device_ptr() as u64;
        let mut a_x = *sc.x.device_ptr() as u64;
        let (mut a_m, mut a_n, mut a_h) = (m as i32, ne as i32, h as i32);
        let mut a_cnt = *sc.fold_cnt.device_ptr() as u64;
        let mut a_ids = *sc.ids.device_ptr() as u64;
        let mut a_wts = *sc.wts.device_ptr() as u64;
        let mut a_k = topk as i32;
        let mut a_sm = *sc.slotmap.device_ptr() as u64;
        let mut a_ix = *sc.idxmap.device_ptr() as u64;
        let mut a_es = *sc.esel.device_ptr() as u64;
        let mut a_og = *sc.offs_gu.device_ptr() as u64;
        let mut a_od = *sc.offs_d.device_ptr() as u64;
        let (mut a_gw, mut a_dw) = (moe.gu_words, moe.d_words);
        let mut args: [*mut std::ffi::c_void; 17] = [
            &mut a_lg as *mut u64 as *mut _, &mut a_w as *mut u64 as *mut _, &mut a_x as *mut u64 as *mut _,
            &mut a_m as *mut i32 as *mut _, &mut a_n as *mut i32 as *mut _, &mut a_h as *mut i32 as *mut _,
            &mut a_cnt as *mut u64 as *mut _, &mut a_ids as *mut u64 as *mut _, &mut a_wts as *mut u64 as *mut _,
            &mut a_k as *mut i32 as *mut _,
            &mut a_sm as *mut u64 as *mut _, &mut a_ix as *mut u64 as *mut _, &mut a_es as *mut u64 as *mut _,
            &mut a_og as *mut u64 as *mut _, &mut a_od as *mut u64 as *mut _,
            &mut a_gw as *mut u64 as *mut _, &mut a_dw as *mut u64 as *mut _,
        ];
        let grid = (g as u32, 1u32, 1u32);
        if pdl_on() {
            unsafe { launch_pss_raw(l.stream.stream, f, grid, (W4R_NT, 1, 1), smem as u32, &mut args) }
                .context("launch xq_router_fold_w4 (pdl)")?;
        } else {
            let r = unsafe {
                cudarc::driver::sys::cuLaunchKernel(f, grid.0, 1, 1, W4R_NT, 1, 1, smem as u32, l.stream.stream,
                    args.as_mut_ptr(), std::ptr::null_mut())
            };
            anyhow::ensure!(r == cudarc::driver::sys::CUresult::CUDA_SUCCESS, "xq_router_fold_w4 launch ({r:?})");
        }
        if w4s_xcheck(W4S_ROUTER) && !self.stream_capturing() {
            // snapshot the new outputs, re-run the OLD fold on the same inputs (it rewrites the same
            // buffers; its self re-arming counter reads 0 after the new kernel's tail), diff bitwise
            self.dev.synchronize()?;
            let (l1, i1, w1) = (self.dev.dtoh_sync_copy(&sc.logits_r)?, self.dev.dtoh_sync_copy(&sc.ids)?,
                                self.dev.dtoh_sync_copy(&sc.wts)?);
            let (s1, x1, e1) = (self.dev.dtoh_sync_copy(&sc.slotmap)?, self.dev.dtoh_sync_copy(&sc.idxmap)?,
                                self.dev.dtoh_sync_copy(&sc.esel)?);
            let (g1, d1) = (self.dev.dtoh_sync_copy(&sc.offs_gu)?, self.dev.dtoh_sync_copy(&sc.offs_d)?);
            let cnt1 = self.dev.dtoh_sync_copy(&sc.fold_cnt)?[0];
            let fo = xq_raw_fn(if wp10_gemv_remap() { "xq_router_fold" } else { "xq_router_fold_m0" })
                .ok_or_else(|| anyhow::anyhow!("xq_router_fold raw fn unavailable"))?;
            let r = unsafe {
                cudarc::driver::sys::cuLaunchKernel(fo, (ne / 4) as u32, 1, 1, 256, 1, 1, 0, l.stream.stream,
                    args.as_mut_ptr(), std::ptr::null_mut())
            };
            anyhow::ensure!(r == cudarc::driver::sys::CUresult::CUDA_SUCCESS, "xq_router_fold (xcheck) launch ({r:?})");
            self.dev.synchronize()?;
            let (l0, i0, w0) = (self.dev.dtoh_sync_copy(&sc.logits_r)?, self.dev.dtoh_sync_copy(&sc.ids)?,
                                self.dev.dtoh_sync_copy(&sc.wts)?);
            let (s0, x0, e0) = (self.dev.dtoh_sync_copy(&sc.slotmap)?, self.dev.dtoh_sync_copy(&sc.idxmap)?,
                                self.dev.dtoh_sync_copy(&sc.esel)?);
            let (g0, d0) = (self.dev.dtoh_sync_copy(&sc.offs_gu)?, self.dev.dtoh_sync_copy(&sc.offs_d)?);
            let (nl, nk) = (m * ne, m * topk);
            let ns = (e0[0].max(0) as usize).min(x0.len());
            let (dl, first) = diff_f32(&l1[..nl], &l0[..nl]);
            let di = (0..nk).filter(|&i| i1[i] != i0[i] || w1[i].to_bits() != w0[i].to_bits()).count();
            let dr = (s1[..ne] != s0[..ne]) as usize + (e1[0] != e0[0]) as usize + (x1[..ns] != x0[..ns]) as usize
                + (g1[..ns] != g0[..ns]) as usize + (d1[..ns] != d0[..ns]) as usize + (cnt1 != 0) as usize;
            w4s_report(W4S_ROUTER, &format!("m={m} xq_router_fold_w4 vs xq_router_fold: logits {dl}/{nl}, ids|wts {di}/{nk}, \
                                             route tables+counter {dr}/6 (esel {})", e1[0]),
                       dl + di + dr, nl + nk + 6, first);
        }
        Ok(true)
    }

    /// (10) PLE key+value projections through xq_gemm_f16_rows_km_w4m<m> (bitwise ==
    /// xq_gemm_f16_rows_km). Ok(false) = item off / not eligible: the caller launches the old kernel.
    pub(super) fn w4s_ple_km(&self, l: &Launcher, sc: &mut Scratch, kk: &CudaSlice<u16>, vk: &CudaSlice<u16>,
                             m: usize) -> Result<bool> {
        if !w4s_on(W4S_PLE) || m == 0 || m > 8 { return Ok(false); }
        const NA: usize = 10240;
        const NB: usize = 2560;
        const K: usize = 2560;
        const BT: usize = 160;
        let smem = (m * K * 2) as u32;
        let (nba, nbb) = (NA.div_ceil(BT), NB.div_ceil(BT));
        const NAMES: [&str; 8] = ["xq_gemm_f16_rows_km_w4m1", "xq_gemm_f16_rows_km_w4m2", "xq_gemm_f16_rows_km_w4m3",
                                  "xq_gemm_f16_rows_km_w4m4", "xq_gemm_f16_rows_km_w4m5", "xq_gemm_f16_rows_km_w4m6",
                                  "xq_gemm_f16_rows_km_w4m7", "xq_gemm_f16_rows_km_w4m8"];
        let kname = NAMES[m - 1];
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| println!("W4S ple: xq_gemm_f16_rows_km_w4m<M> ON ({} x {BT} threads, x in smem; --w4s-off=ple = xq_gemm_f16_rows_km)",
                                   nba + nbb));
        xqlaunch!(l, kname, ((nba + nbb) as u32, 1, 1), (BT as u32, 1, 1), smem,
                 (&mut sc.ple_key16, kk, NA as i32, nba as i32, &mut sc.ple_val, vk, NB as i32,
                  &sc.ple_emb, m as i32, K as i32))?;
        if w4s_xcheck(W4S_PLE) && !self.stream_capturing() {
            let mut ko = self.dev.alloc_zeros::<u16>(m * NA)?;
            let mut vo = self.dev.alloc_zeros::<u16>(m * NB)?;
            xqlaunch!(l, "xq_gemm_f16_rows_km", (100u32, 1, 1), (128, 1, 1), 0,
                     (&mut ko, kk, NA as i32, 80i32, &mut vo, vk, NB as i32, &sc.ple_emb, m as i32, K as i32))?;
            self.dev.synchronize()?;
            let (kn, vn) = (self.dev.dtoh_sync_copy(&sc.ple_key16)?, self.dev.dtoh_sync_copy(&sc.ple_val)?);
            let (ko, vo) = (self.dev.dtoh_sync_copy(&ko)?, self.dev.dtoh_sync_copy(&vo)?);
            let (b1, f1) = diff_u16(&kn[..m * NA], &ko);
            let (b2, f2) = diff_u16(&vn[..m * NB], &vo);
            w4s_report(W4S_PLE, &format!("m={m} {kname} vs xq_gemm_f16_rows_km (key {b1}, value {b2})"),
                       b1 + b2, m * (NA + NB), format!("{f1}{f2}"));
        }
        Ok(true)
    }

    /// (6) the whole verify commit (every GDN layer's conv ring + rank-1 state commit, and the PLE
    /// ring) in ONE xq_gdn_commit_all_w4 launch. Ok(false) = item off / not eligible: the caller
    /// runs the per-layer launches.
    pub(super) fn w4s_commit_all(&self, l: &Launcher, sc: &mut Scratch, ring_mode: u8) -> Result<bool> {
        let cfg = &self.tc; // TP-B: the trunk's per-rank GDN geometry
        let (kd, vd, nh) = (cfg.lin_k_dim, cfg.lin_v_dim, cfg.lin_num_v_heads);
        let conv_dim = cfg.key_dim() * 2 + cfg.value_dim();
        let ck = cfg.conv_kernel;
        let nl = self.s_state.len();
        if !w4s_on(W4S_GDN) || ring_mode != 1 || kd != 128 || vd % 32 != 0 || ck != 4 || nl == 0
            || self.conv_state.len() != nl || nh >= 0x10000 || vd >= 0x10000 || conv_dim >= 0x10000 {
            return Ok(false);
        }
        let ple = self.ple.is_some() && crate::opts::var(crate::opt!("exl3-no-ple")).is_err();
        let nring = nh * (vd / 32);
        let gx = (nring + conv_dim.div_ceil(128)).max(10240usize.div_ceil(128));
        let gy = nl + ple as usize;
        let ring_ls = ((MTP_MAX_K + 1) * nh * GDN_RS) as i64;
        let raw_ls = (SAVE_PLANE_ROWS * conv_dim) as i64; // PACK2/D1: qkv_save planes are SAVE_PLANE_ROWS rows (was MTP_MAX_K+1 — read the wrong layer's raws)
        let nh_kd = (nh | (kd << 16)) as i32;
        let vd_conv = (vd | (conv_dim << 16)) as i32;
        let ple_st = if ple { *self.ple_state.device_ptr() as u64 } else { 0u64 };
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| println!("W4S gdn: xq_gdn_commit_all_w4 ON (one launch, grid {gx} x {gy}: {nl} layers x (ring + conv) \
                                    + {} PLE ring; --w4s-off=gdn = per-layer commits)", if ple { "1" } else { "no" }));
        // XCHECK reference: the OLD per-layer launches on device COPIES of the live states
        // (taken before the new kernel runs), compared after both.
        let xc = w4s_xcheck(W4S_GDN) && !self.stream_capturing();
        let refs = if xc {
            let mut rs: Vec<CudaSlice<f32>> = Vec::with_capacity(nl);
            let mut rc: Vec<CudaSlice<f32>> = Vec::with_capacity(nl);
            for i in 0..nl {
                let (ns, nc) = (self.s_state[i].len(), self.conv_state[i].len());
                let mut a = self.dev.alloc_zeros::<f32>(ns)?;
                let mut b = self.dev.alloc_zeros::<f32>(nc)?;
                xqlaunch!(l, "xq_copy_f32", (((ns as u32) + 255) / 256, 1, 1), (256, 1, 1), 0,
                          (&mut a, &self.s_state[i], ns as i64, 0i64, 0i64))?;
                xqlaunch!(l, "xq_copy_f32", (((nc as u32) + 255) / 256, 1, 1), (256, 1, 1), 0,
                          (&mut b, &self.conv_state[i], nc as i64, 0i64, 0i64))?;
                rs.push(a);
                rc.push(b);
            }
            let np = self.ple_state.len();
            let mut rp = self.dev.alloc_zeros::<u16>(np)?;
            xqlaunch!(l, "xq_copy_u16", (((np as u32) + 255) / 256, 1, 1), (256, 1, 1), 0,
                      (&mut rp, &self.ple_state, np as i64, 0i64, 0i64))?;
            for i in 0..nl {
                xqlaunch!(l, "xq_conv_commit", ((conv_dim as u32 + 255) / 256, 1, 1), (256, 1, 1), 0,
                         ((*rc[i].device_ptr()) as u64, &sc.gdn_ring, conv_dim as i32, ck as i32, &sc.qkv_save,
                          // K5 (REL v0.7.3 review): the gdn reference reads qkv_save with the SAVE plane
                          // stride (16 rows) like every other plane site — the old (MTP_MAX_K+1) stride
                          // read the WRONG LAYER's raws here (false XCHECK mismatches; diagnostic only).
                          (i * SAVE_PLANE_ROWS * conv_dim) as i64, &sc.acc2))?;
                let ring_p = (*sc.gdn_ring.device_ptr()) as u64 + (i * (MTP_MAX_K + 1) * nh * GDN_RS * 4) as u64;
                xqlaunch!(l, "xq_gdn_commit_ring", ((nh * (vd / 32)) as u32, 1, 1), (kd as u32, 1, 1), 0,
                         ((*rs[i].device_ptr()) as u64, ring_p, nh as i32, kd as i32, vd as i32, &sc.acc2))?;
            }
            if ple {
                xqlaunch!(l, "xq_ple_ring_commit", ((10240u32 + 255) / 256, 1, 1), (256, 1, 1), 0,
                         (&sc.ple_normed, (*rp.device_ptr()) as u64, &sc.acc2))?;
            }
            Some((rs, rc, rp))
        } else { None };
        xqlaunch!(l, "xq_gdn_commit_all_w4", (gx as u32, gy as u32, 1), (128, 1, 1), 0,
                 (&self.w4s_lptr, nl as i32, &sc.gdn_ring, ring_ls, &sc.qkv_save, raw_ls,
                  nh_kd, vd_conv, ck as i32, &sc.acc2, &sc.ple_normed, ple_st))?;
        if let Some((rs, rc, rp)) = refs {
            self.dev.synchronize()?;
            let (mut bad, mut tot, mut first) = (0usize, 0usize, String::new());
            for i in 0..nl {
                let (a, b) = (self.dev.dtoh_sync_copy(&self.s_state[i])?, self.dev.dtoh_sync_copy(&rs[i])?);
                let (d, f) = diff_f32(&a, &b);
                if d > 0 && first.is_empty() { first = format!(" [S layer {i}]{f}"); }
                bad += d; tot += a.len();
                let (a, b) = (self.dev.dtoh_sync_copy(&self.conv_state[i])?, self.dev.dtoh_sync_copy(&rc[i])?);
                let (d, f) = diff_f32(&a, &b);
                if d > 0 && first.is_empty() { first = format!(" [conv layer {i}]{f}"); }
                bad += d; tot += a.len();
            }
            if ple {
                let (a, b) = (self.dev.dtoh_sync_copy(&self.ple_state)?, self.dev.dtoh_sync_copy(&rp)?);
                let (d, f) = diff_u16(&a, &b);
                if d > 0 && first.is_empty() { first = format!(" [ple]{f}"); }
                bad += d; tot += a.len();
            }
            let acc = self.dev.dtoh_sync_copy(&sc.acc2)?;
            w4s_report(W4S_GDN, &format!("commit a={} xq_gdn_commit_all_w4 vs per-layer conv/ring{} commits ({nl} layers, whole planes)",
                                         acc.get(1).copied().unwrap_or(-1), if ple { "/ple" } else { "" }),
                       bad, tot, first);
        }
        Ok(true)
    }

    /// (9) The dense v3 accumulate kernel for this width: the W4 twin when on.
    pub(super) fn w4s_dattn_kname(&self, old: &'static str) -> &'static str {
        if !w4s_on(W4S_DATTN) { return old; }
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| println!("W4S dattn: xq_attn_dense_acc4_w4 / _acc1_w4 ON (5-stage V ring; --w4s-off=dattn = v3 acc)"));
        super::dv3_w4_twin(old)   // TP-4G: includes the G=6 (W=4) twins
    }

    /// (9) XCHECK after a W4 accumulate launch (eager only): the old kernel into a side buffer
    /// on the same score plane / chunk maxima / cache (acc writes only the attention rows).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn w4s_dattn_xcheck(&self, l: &Launcher, sc: &Scratch, kv: u64, m: usize, kname: &str,
                                   old: &str, sd: usize, cfg: &crate::qwen::Config, block: u32) -> Result<()> {
        if kname == old || !w4s_xcheck(W4S_DATTN) || self.stream_capturing() { return Ok(()); }
        let nh_p = (cfg.num_heads as i32) | ((cfg.num_kv_heads as i32) << 16);
        let hd_p = (cfg.head_dim as i32) | ((cfg.rotary_dim as i32) << 16);
        let n = m * cfg.num_heads * cfg.head_dim;
        let mut side = self.dev.alloc_zeros::<u16>(n)?;
        xqlaunch!(l, old, ((cfg.head_dim / sd) as u32, cfg.num_kv_heads as u32, m as u32), (block, 1, 1), 0,
                  (&mut side, &sc.dv3_s, &sc.dv3_pm, kv, &sc.qg, &sc.slots, nh_p, hd_p, self.mpf()))?;
        self.dev.synchronize()?;
        let (a, b) = (self.dev.dtoh_sync_copy(&sc.normed)?, self.dev.dtoh_sync_copy(&side)?);
        let (bad, first) = diff_u16(&a[..n], &b);
        let sp = self.dev.dtoh_sync_copy(&sc.slots)?;
        w4s_report(W4S_DATTN, &format!("m={m} p0={} {kname} vs {old}", sp.get(1).copied().unwrap_or(-1)), bad, n, first);
        Ok(())
    }

    /// (8) The m = 1 draft-head dense attention: dense_prep (grid m*nh) -> splitk4_w4 ->
    /// combine_w4 (bitwise == xq_attn_prep -> xq_attn_dense_splitk4 -> xq_attn_sel_combine).
    /// Ok(false) = item off / not eligible: the caller launches the old three.
    pub(super) fn w4s_draft_attn(&self, l: &Launcher, sc: &mut Scratch, head: &DraftHead, a: &AttnLayer,
                                 m: usize) -> Result<bool> {
        let cfg = &self.cfg;
        let nkv = cfg.num_kv_heads;
        let gqa = if nkv > 0 { cfg.num_heads / nkv } else { 0 };
        let rb = kv_rowbytes(self.kv_fmt, cfg.head_dim);
        if !w4s_on(W4S_DRAFT) || cfg.head_dim != 256 || nkv == 0 || gqa == 0 || gqa > 16 || gqa * nkv != cfg.num_heads
            || rb > 1024 || rb % 16 != 0 || DRAFT_ATTN_SPLITS > 256 {
            return Ok(false);
        }
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| println!("W4S draft: head attention = xq_attn_dense_prep (grid m*nh) + xq_attn_dense_splitk4_w4 \
                                    (smem K|V ring) + xq_attn_sel_combine_w4 ON (--w4s-off=draft = the old three)"));
        let nh_p = (cfg.num_heads as i32) | ((nkv as i32) << 16);
        let hd_p = (cfg.head_dim as i32) | ((cfg.rotary_dim as i32) << 16);
        let kvp = (*head.kv.device_ptr()) as u64;
        xqlaunch!(l, "xq_attn_dense_prep", ((m * cfg.num_heads) as u32, 1, 1), (cfg.head_dim as u32, 1, 1), 0,
                  (&mut sc.qstage, kvp, &sc.qg, &sc.kp, &sc.vp, &a.qnkn, &sc.cos, &sc.slots,
                   nh_p, hd_p, self.mpf(), cfg.rms_eps))?;
        let geom: u64 = (DRAFT_ATTN_SPLITS as u64) | ((self.mpf() as u32 as u64) << 32);
        xqlaunch!(l, "xq_attn_dense_splitk4_w4", ((m * nkv * DRAFT_ATTN_SPLITS) as u32, 1, 1), (256, 1, 1), 0,
                  (&mut sc.dpm, &mut sc.dpl, &mut sc.dpacc, &sc.qstage, kvp, &sc.slots, nh_p, geom as i64))?;
        xqlaunch!(l, "xq_attn_sel_combine_w4", ((m * cfg.num_heads) as u32, 1, 1), (cfg.head_dim as u32, 1, 1), 0,
                  (&mut sc.normed, &sc.dpm, &sc.dpl, &sc.dpacc, &sc.qg,
                   nh_p, DRAFT_ATTN_SPLITS as i32, self.rows_kvf(m)))?;
        if w4s_xcheck(W4S_DRAFT) && !self.stream_capturing() {
            // the old chain on side buffers (xq_attn_prep rewrites the same K/V words and qstage
            // rows — identical when the check passes; qstage is compared after it)
            self.dev.synchronize()?;
            let nq = m * cfg.num_heads * cfg.head_dim;
            let np = m * cfg.num_heads * DRAFT_ATTN_SPLITS;
            let q1 = self.dev.dtoh_sync_copy(&sc.qstage)?;
            let (pm1, pl1, pa1) = (self.dev.dtoh_sync_copy(&sc.dpm)?, self.dev.dtoh_sync_copy(&sc.dpl)?,
                                   self.dev.dtoh_sync_copy(&sc.dpacc)?);
            let o1 = self.dev.dtoh_sync_copy(&sc.normed)?;
            let mut xm = self.dev.alloc_zeros::<f32>(np)?;
            let mut xl = self.dev.alloc_zeros::<f32>(np)?;
            let mut xa = self.dev.alloc_zeros::<f32>(np * cfg.head_dim)?;
            let mut xo = self.dev.alloc_zeros::<u16>(nq)?;
            xqlaunch!(l, "xq_attn_prep", ((m * nkv) as u32, 1, 1), (cfg.head_dim as u32, 1, 1), 0,
                      (&mut sc.qstage, kvp, &sc.qg, &sc.kp, &sc.vp, &a.qnkn, &sc.cos, &sc.slots,
                       nh_p, hd_p, self.mpf(), cfg.rms_eps))?;
            xqlaunch!(l, "xq_attn_dense_splitk4", ((m * nkv * DRAFT_ATTN_SPLITS) as u32, 1, 1), (256, 1, 1), 0,
                      (&mut xm, &mut xl, &mut xa, &sc.qstage, kvp, &sc.slots, nh_p, geom as i64))?;
            xqlaunch!(l, "xq_attn_sel_combine", ((m * cfg.num_heads) as u32, 1, 1), (cfg.head_dim as u32, 1, 1), 0,
                      (&mut xo, &xm, &xl, &xa, &sc.qg, nh_p, DRAFT_ATTN_SPLITS as i32, self.rows_kvf(m)))?;
            self.dev.synchronize()?;
            let q0 = self.dev.dtoh_sync_copy(&sc.qstage)?;
            let (pm0, pl0, pa0) = (self.dev.dtoh_sync_copy(&xm)?, self.dev.dtoh_sync_copy(&xl)?, self.dev.dtoh_sync_copy(&xa)?);
            let o0 = self.dev.dtoh_sync_copy(&xo)?;
            let (bq, fq) = diff_f32(&q1[..nq], &q0[..nq]);
            let (bm, fm) = diff_f32(&pm1[..np], &pm0);
            let (bl, _) = diff_f32(&pl1[..np], &pl0);
            let (ba, _) = diff_f32(&pa1[..np * cfg.head_dim], &pa0);
            let (bo, fo) = diff_u16(&o1[..nq], &o0);
            let sp = self.dev.dtoh_sync_copy(&sc.slots)?;
            w4s_report(W4S_DRAFT, &format!("m={m} p={} dense_prep/splitk4_w4/combine_w4 vs prep/splitk4/combine: \
                                            qstage {bq}, pm {bm}, pl {bl}, pacc {ba}, attn {bo}", sp.get(1).copied().unwrap_or(-1)),
                       bq + bm + bl + ba + bo, nq * 2 + np * (2 + cfg.head_dim), format!("{fq}{fm}{fo}"));
        }
        Ok(true)
    }
}
