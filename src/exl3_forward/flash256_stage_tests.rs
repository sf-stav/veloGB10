//! Unit tests (module `flash256_stage_tests`) moved out of the old inline block in `exl3_forward.rs`; text unchanged.
//! Declared from `mod.rs`; the body keeps its original indentation and `use super::*` imports.

    /// One kernel shape: head dim, q rows per block (= warps), kv rows per tile.
    #[derive(Clone, Copy)]
    struct Shape { hd: usize, bt: usize, bkv: usize }
    impl Shape {
        fn nthr(self) -> usize { self.bt * 32 }
        fn g(self) -> usize { self.hd / 8 }                             // 8-dim groups per row
        fn ipt(self) -> usize { self.bkv * 2 * self.g() / self.nthr() } // vector items per thread
        fn spt(self) -> usize { self.bkv * self.hd / self.nthr() }      // scalar elements per thread
    }
    /// xq_attn_prefill_flash256 / 256v / 256c: a3fg_flash_body<256, 8, 16, 260, 8, ..>
    const HD256: Shape = Shape { hd: 256, bt: 8, bkv: 16 };
    /// xq_attn_prefill_flash / 128c: a3fg_flash_body<128, 16, 32, 132, 4, ..>
    const HD128: Shape = Shape { hd: 128, bt: 16, bkv: 32 };

    #[derive(Clone, Copy, PartialEq, Debug)]
    enum Stage { Scalar, V1, V2, Causal }

    struct Rng(u64);
    impl Rng {
        fn u(&mut self) -> f32 {
            self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17;
            ((self.0 >> 11) as f64 / (1u64 << 53) as f64) as f32
        }
        fn g(&mut self) -> f32 { (self.u() + self.u() + self.u() + self.u() - 2.0) * 1.7 }
    }

    /// q rows t0..t0+BT (q: [rows][HD], already normed/roped; cache k/v: [pos][HD]) of a chunk
    /// starting at pos0 -> per-row attention acc / l (the gate is shared and omitted).
    fn block(sh: Shape, st: Stage, q: &[f32], k: &[f32], v: &[f32], pos0: usize, t0: usize) -> Vec<Vec<f32>> {
        let (hd, bt, bkv, nthr, g, ipt) = (sh.hd, sh.bt, sh.bkv, sh.nthr(), sh.g(), sh.ipt());
        // V1 / V2 model the hd=256 twin's code (owner warp of group grp = grp >> 2): blockDim == hd only
        assert!(matches!(st, Stage::Scalar | Stage::Causal) || nthr == hd, "{st:?} needs blockDim == hd");
        let scale = 1.0f32 / (hd as f32).sqrt();         // rsqrtf(256) = 1/16 exact
        let pbase = pos0 + t0;
        let pmax = |w: usize| pbase + w;
        let pend = pbase + bt - 1;                       // the block's last live tile start bound
        let mut ks = vec![f32::NAN; bkv * hd];           // smem starts undefined: NaN exposes a
        let mut vs = vec![f32::NAN; bkv * hd];           // read of an element never staged
        let mut m = vec![f32::NEG_INFINITY; bt];
        let mut l = vec![0f32; bt];
        let mut acc = vec![vec![0f32; hd]; bt];
        let mut jt = 0usize;
        while jt <= pend {
            match st {
                Stage::Scalar => for w in 0..bt {
                    if jt > pmax(w) { continue; }        // this warp left its row loop
                    let jn = bkv.min(pmax(w) + 1 - jt);
                    // for (x = tid; x < BKV*hd; x += blockDim.x): element x is thread x % blockDim's
                    for x in (w * 32..w * 32 + 32).flat_map(|tid| (0..sh.spt()).map(move |kk| tid + kk * nthr)) {
                        let (jj, e) = (x / hd, x % hd);
                        let ok = jj < jn;
                        ks[jj * hd + e] = if ok { k[(jt + jj) * hd + e] } else { 0.0 };
                        vs[jj * hd + e] = if ok { v[(jt + jj) * hd + e] } else { 0.0 };
                    }
                },
                Stage::V1 => for w in 0..bt {
                    if jt > pmax(w) { continue; }
                    let jn = bkv.min(pmax(w) + 1 - jt);
                    for tid in w * 32..w * 32 + 32 {
                        for kk in 0..ipt {
                            let it = tid + kk * nthr;
                            let (grp, jj, isv) = (it % g, (it / g) % bkv, it / (g * bkv));
                            let (ok, jr) = (jj < jn, jj.min(jn - 1));
                            let (src, dst) = if isv != 0 { (v, &mut vs) } else { (k, &mut ks) };
                            for i in 0..8 {
                                let val = src[(jt + jr) * hd + grp * 8 + i];
                                dst[jj * hd + grp * 8 + i] = if ok { val } else { 0.0 };
                            }
                        }
                    }
                },
                Stage::V2 | Stage::Causal => for tid in 0..nthr {
                    for kk in 0..ipt {
                        let it = tid + kk * nthr;
                        let (grp, jj, isv) = (it % g, (it / g) % bkv, it / (g * bkv));
                        let pw = if st == Stage::V2 { pbase + (grp >> 2) } else { pend };
                        let p = (jt + jj).min(pw);
                        let (src, dst) = if isv != 0 { (v, &mut vs) } else { (k, &mut ks) };
                        if jt <= pw {
                            let ok = jt + jj <= pw;
                            for i in 0..8 {
                                let val = src[p * hd + grp * 8 + i];
                                dst[jj * hd + grp * 8 + i] = if ok { val } else { 0.0 };
                            }
                        }
                    }
                },
            }
            for r in 0..bt {
                if jt > pmax(r) { continue; }
                let jn = bkv.min(pmax(r) + 1 - jt);
                let qr = &q[(t0 + r) * hd..(t0 + r + 1) * hd];
                let mut s = [f32::NEG_INFINITY; 32];
                for lane in 0..jn {
                    let mut d = 0f32;
                    for e in 0..hd { d = qr[e].mul_add(ks[lane * hd + e], d); }
                    s[lane] = d * scale;
                }
                let mtile = s.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mnew = m[r].max(mtile);
                let factor = (m[r] - mnew).exp();
                m[r] = mnew;
                l[r] *= factor;
                for a in acc[r].iter_mut() { *a *= factor; }
                let mut p = [0f32; 32];
                for lane in 0..jn { p[lane] = (s[lane] - mnew).exp(); }
                let mut tr = p;
                for off in [16usize, 8, 4, 2, 1] { for lane in 0..off { tr[lane] += tr[lane + off]; } }
                l[r] += tr[0];
                for jj in 0..jn {
                    for d in 0..hd { acc[r][d] = p[jj].mul_add(vs[jj * hd + d], acc[r][d]); }
                }
            }
            jt += bkv;
        }
        (0..bt).map(|r| acc[r].iter().map(|a| a / l[r]).collect()).collect()
    }

    /// Chunk rows [0, c) at pos0: (outputs of rows < c per stage, f64 causal-softmax reference).
    fn chunk(sh: Shape, st: Stage, pos0: usize, c: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<Vec<f64>>) {
        let (hd, bt, bkv) = (sh.hd, sh.bt, sh.bkv);
        let mut rng = Rng(seed);
        let rows = c.div_ceil(bt) * bt;
        let npos = pos0 + rows + bkv;
        let q: Vec<f32> = (0..rows * hd).map(|_| rng.g()).collect();
        let k: Vec<f32> = (0..npos * hd).map(|_| rng.g()).collect();
        let v: Vec<f32> = (0..npos * hd).map(|_| rng.g()).collect();
        let mut out = Vec::new();
        for t0 in (0..rows).step_by(bt) { out.extend(block(sh, st, &q, &k, &v, pos0, t0)); }
        out.truncate(c);
        let rs = (hd as f64).sqrt();
        let reference = (0..c).map(|t| {
            let pm = pos0 + t;
            let sc: Vec<f64> = (0..=pm).map(|j| (0..hd).map(|e| q[t * hd + e] as f64 * k[j * hd + e] as f64)
                                                 .sum::<f64>() / rs).collect();
            let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let w: Vec<f64> = sc.iter().map(|s| (s - mx).exp()).collect();
            let den: f64 = w.iter().sum();
            (0..hd).map(|d| (0..=pm).map(|j| w[j] * v[j * hd + d] as f64).sum::<f64>() / den).collect()
        }).collect();
        (out, reference)
    }

    fn differ(a: &[Vec<f32>], b: &[Vec<f32>]) -> usize {
        a.iter().zip(b).map(|(x, y)| x.iter().zip(y).filter(|(p, q)| p.to_bits() != q.to_bits()).count()).sum()
    }
    fn max_err(a: &[Vec<f32>], r: &[Vec<f64>]) -> f64 {
        a.iter().zip(r).flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| (*p as f64 - q).abs())).fold(0.0, f64::max)
    }

    /// v2 == scalar bitwise at every chunk start (incl. pos0 % 8 != 0, where warps' row loops end
    /// on different tiles and the scalar image keeps stale elements), and the test is not vacuous:
    /// v1 differs from the scalar image, and the scalar image deviates from the causal softmax
    /// reference far beyond the fully staged (causal) image's rounding error.
    #[test]
    fn flash256v_stage_is_the_scalar_image() {
        let sh = HD256;
        for (i, &(pos0, c)) in [(0usize, 40usize), (8, 24), (16, 33), (256, 17), (3, 29), (5, 16), (9, 23), (13, 40),
                                 (1234, 21), (2047, 9)].iter().enumerate() {
            let seed = 0x9E37_79B9_7F4A_7C15 ^ (i as u64 * 0x1000_0001);
            let (sc, rf) = chunk(sh, Stage::Scalar, pos0, c, seed);
            let (v2, _) = chunk(sh, Stage::V2, pos0, c, seed);
            let (v1, _) = chunk(sh, Stage::V1, pos0, c, seed);
            let (ca, _) = chunk(sh, Stage::Causal, pos0, c, seed);
            let nd = differ(&v2, &sc);
            assert_eq!(nd, 0, "pos0 {pos0} c {c}: v2 staging != scalar staging in {nd} of {} outputs", c * sh.hd);
            assert!(sc.iter().flatten().all(|x| x.is_finite()), "pos0 {pos0}: scalar read an unstaged element");
            let (e_sc, e_ca) = (max_err(&sc, &rf), max_err(&ca, &rf));
            crate::dprintln!("pos0 {pos0:5} c {c:3}: v1 vs scalar differ {:6} of {:6} | max |err| vs f64 ref: scalar {e_sc:.3e} \
                      v1 {:.3e} causal {e_ca:.3e}", differ(&v1, &sc), c * sh.hd, max_err(&v1, &rf));
            assert!(e_ca < 1e-4, "pos0 {pos0}: the fully staged image must match the reference ({e_ca:.3e})");
            assert!(e_sc > 1e-3, "pos0 {pos0}: the scalar image's diagonal-tile defect must show ({e_sc:.3e})");
            if pos0 % 8 == 0 { assert!(differ(&v1, &sc) > 0, "pos0 {pos0}: v1 must differ from the scalar image"); }
        }
    }

    /// w5/PFIX, hd=128 (xq_attn_prefill_flash vs xq_attn_prefill_flash128c): the scalar stager's
    /// image (element (jj, e) owned by warp (jj % 4) * 4 + (e >> 5)) carries the diagonal-tile
    /// defect at every chunk start — including blocks whose 16 rows straddle two 32-row tiles
    /// (pos0 % 32 in 17..31, where the first rows' warps leave stale elements in the second
    /// tile) and deep starts — while the CAUSAL image matches the f64 causal softmax.
    #[test]
    fn flash128c_stage_is_causal() {
        let sh = HD128;
        for (i, &(pos0, c)) in [(0usize, 40usize), (8, 24), (16, 33), (24, 48), (256, 17), (3, 29), (5, 16), (9, 23),
                                 (13, 40), (21, 64), (1234, 21), (2047, 19)].iter().enumerate() {
            let seed = 0xD1B5_4A32_D192_ED03 ^ (i as u64 * 0x1000_0001);
            let (sc, rf) = chunk(sh, Stage::Scalar, pos0, c, seed);
            let (ca, _) = chunk(sh, Stage::Causal, pos0, c, seed);
            assert!(sc.iter().flatten().all(|x| x.is_finite()), "pos0 {pos0}: scalar read an unstaged element");
            assert!(ca.iter().flatten().all(|x| x.is_finite()), "pos0 {pos0}: causal read an unstaged element");
            let (e_sc, e_ca) = (max_err(&sc, &rf), max_err(&ca, &rf));
            crate::dprintln!("hd128 pos0 {pos0:5} c {c:3}: causal vs scalar differ {:6} of {:6} | max |err| vs f64 ref: \
                      scalar {e_sc:.3e} causal {e_ca:.3e}", differ(&ca, &sc), c * sh.hd);
            assert!(e_ca < 1e-4, "pos0 {pos0}: the causal image must match the reference ({e_ca:.3e})");
            assert!(e_sc > 1e-3, "pos0 {pos0}: the scalar image's diagonal-tile defect must show ({e_sc:.3e})");
        }
    }
