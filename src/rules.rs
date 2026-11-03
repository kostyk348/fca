//! The cellular upscale rules: `scale2x` (EPX), `fuzzy` (EAFAR S5) and `xbr`.
//!
//! All rules share the skeleton `out = mix(E, candidate, strength)` with
//! `strength = m(p~q) * (1-m(perp1~p)) * (1-m(perp2~q))`. Results stay inside
//! the 3x3 min/max — no ringing by construction. Integer-only, deterministic.
//!
//! In the C original a hand-written AVX2 path was kept bit-exact with this
//! scalar reference; the port keeps the scalar arithmetic as the single
//! source of truth and parallelizes rows instead.

use crate::grid::Grid;
use rayon::prelude::*;

/// Tolerance for "cell state match" (~0.08 in 0..1 == 20 in 0..255, matches GLSL EPS).
pub const K_TOL: i32 = 20;

/// Width of the fuzzy triangle: `kFuzzyW = 2 * kTol` gives a soft zone around
/// the old hard threshold — kills staircase on diagonals and frame-to-frame
/// flicker at tolerance boundaries.
pub const K_FUZZY_W: i32 = 40;

#[inline]
pub fn eq_tol(a: u8, b: u8) -> bool {
    let d = i32::from(a.max(b)) - i32::from(a.min(b));
    d <= K_TOL
}

/// Clamped neighbor access (out-of-range clamped, like a GLSL clamped sampler).
#[inline]
pub fn px(g: &Grid, x: isize, y: isize) -> u8 {
    let x = x.clamp(0, g.w as isize - 1);
    let y = y.clamp(0, g.h as isize - 1);
    g.data[y as usize * g.w + x as usize]
}

// ---------------------------------------------------------------- fuzzy (EAFAR S5)

/// Triangle membership: 255 at d==0, linear to 0 at d>=K_FUZZY_W
/// (`d*51>>3 == d*(255/40)` in fixed point).
#[inline]
pub fn m_fuzzy(d: i32) -> i32 {
    if d >= K_FUZZY_W {
        return 0;
    }
    255 - ((d * 51) >> 3)
}

/// Smooth pick: `out = E + (w*(a-E))/256` via half-shift; bit-exact with the
/// retired AVX2 path (max error vs exact /255 is <= 1 level, deterministic).
#[inline]
pub fn fz_pick(w: i32, a: u8, e: u8) -> u8 {
    let t = (((w + 1) >> 1) * (i32::from(a) - i32::from(e))) >> 7;
    let v = i32::from(e) + t;
    v.clamp(0, 255) as u8
}

/// Fuzzy version of the EPX e0..e3 picks. Per output: candidate `a` on the
/// diagonal line (a,c), contrast perpendicular (b1 vs a, b2 vs c):
/// `w = m(a~c) * (1-m(b1~a)) * (1-m(b2~c))`.
#[allow(clippy::too_many_arguments)]
pub fn fz_pick4(b: u8, d: u8, e: u8, f: u8, h: u8) -> [u8; 4] {
    let ad = |x: i32, y: i32| (x - y).abs();
    let (b, d, e, f, h) = (
        i32::from(b),
        i32::from(d),
        i32::from(e),
        i32::from(f),
        i32::from(h),
    );
    let w0 = (m_fuzzy(ad(b, d)) * (255 - m_fuzzy(ad(f, b))) * (255 - m_fuzzy(ad(h, d)))) >> 16;
    let w1 = (m_fuzzy(ad(f, b)) * (255 - m_fuzzy(ad(d, f))) * (255 - m_fuzzy(ad(h, b)))) >> 16;
    let w2 = (m_fuzzy(ad(d, h)) * (255 - m_fuzzy(ad(b, d))) * (255 - m_fuzzy(ad(f, h)))) >> 16;
    let w3 = (m_fuzzy(ad(f, h)) * (255 - m_fuzzy(ad(d, f))) * (255 - m_fuzzy(ad(b, h)))) >> 16;
    [
        fz_pick(w0, b as u8, e as u8),
        fz_pick(w1, f as u8, e as u8),
        fz_pick(w2, d as u8, e as u8),
        fz_pick(w3, f as u8, e as u8),
    ]
}

// ------------------------------------------------- xBR-style rule (8 neighbors)

/// Weighted pair blend: `(p*wp + q*wq) / (wp+wq)` with rounding.
#[inline]
pub fn fz_blend2(p: i32, q: i32, wp: i32, wq: i32) -> u8 {
    let num = p * wp + q * wq;
    let den = wp + wq;
    ((num + den / 2) / den) as u8
}

/// xBR pick with diagonal corners (A,C,G,I) and weighted pair blend: thin 1px
/// diagonal lines survive. Results stay inside the 3x3 min/max.
#[allow(clippy::too_many_arguments)]
pub fn xbr_pick4(a: u8, b: u8, c: u8, d: u8, e: u8, f: u8, g: u8, h: u8, i: u8) -> [u8; 4] {
    let ad = |x: i32, y: i32| (x - y).abs();
    let (a, b, c, d, e, f, g, h, i) = (
        i32::from(a),
        i32::from(b),
        i32::from(c),
        i32::from(d),
        i32::from(e),
        i32::from(f),
        i32::from(g),
        i32::from(h),
        i32::from(i),
    );
    // e0: p=B q=D r=A, perp1=F perp2=H
    let e0 = {
        let s = (m_fuzzy(ad(b, d)) * (255 - m_fuzzy(ad(f, b))) * (255 - m_fuzzy(ad(h, d)))) >> 16;
        let blend = fz_blend2(b, d, m_fuzzy(ad(a, b)) + 1, m_fuzzy(ad(a, d)) + 1);
        fz_pick(s, blend, e as u8)
    };
    // e1: p=F q=B r=C, perp1=D perp2=H
    let e1 = {
        let s = (m_fuzzy(ad(f, b)) * (255 - m_fuzzy(ad(d, f))) * (255 - m_fuzzy(ad(h, b)))) >> 16;
        let blend = fz_blend2(f, b, m_fuzzy(ad(c, f)) + 1, m_fuzzy(ad(c, b)) + 1);
        fz_pick(s, blend, e as u8)
    };
    // e2: p=D q=H r=G, perp1=B perp2=F
    let e2 = {
        let s = (m_fuzzy(ad(d, h)) * (255 - m_fuzzy(ad(b, d))) * (255 - m_fuzzy(ad(f, h)))) >> 16;
        let blend = fz_blend2(d, h, m_fuzzy(ad(g, d)) + 1, m_fuzzy(ad(g, h)) + 1);
        fz_pick(s, blend, e as u8)
    };
    // e3: p=F q=H r=I, perp1=D perp2=B
    let e3 = {
        let s = (m_fuzzy(ad(f, h)) * (255 - m_fuzzy(ad(d, f))) * (255 - m_fuzzy(ad(b, h)))) >> 16;
        let blend = fz_blend2(f, h, m_fuzzy(ad(i, f)) + 1, m_fuzzy(ad(i, h)) + 1);
        fz_pick(s, blend, e as u8)
    };
    [e0, e1, e2, e3]
}

// ------------------------------------------------------------- full-plane rules

/// EPX picks with strict priority — the branch structure of the C reference,
/// shared by the scalar plane rule and the AVX2 edge pixels.
// identical blocks are intentional: e2/e3 repeat the d/f fallback on purpose
#[allow(clippy::if_same_then_else)]
#[inline]
pub fn epx_pick4(b: u8, d: u8, e: u8, f: u8, h: u8) -> [u8; 4] {
    let (e0, e1, e2, e3);
    if !eq_tol(b, h) && !eq_tol(d, f) {
        e0 = if eq_tol(b, d) {
            b
        } else if eq_tol(b, f) {
            f
        } else if eq_tol(d, f) {
            d
        } else {
            e
        };
        e1 = if eq_tol(b, f) {
            f
        } else if eq_tol(b, d) {
            b
        } else if eq_tol(f, d) {
            d
        } else {
            e
        };
        e2 = if eq_tol(h, d) {
            d
        } else if eq_tol(h, f) {
            f
        } else if eq_tol(d, f) {
            d
        } else {
            e
        };
        e3 = if eq_tol(h, f) {
            f
        } else if eq_tol(h, d) {
            d
        } else if eq_tol(f, d) {
            d
        } else {
            e
        };
    } else {
        e0 = e;
        e1 = e;
        e2 = e;
        e3 = e;
    }
    [e0, e1, e2, e3]
}

/// Forced-scalar EPX plane rule — the bit-exactness reference for the AVX2
/// path and the fallback on CPUs without AVX2.
pub fn epx_scalar_ref(input: &Grid) -> Grid {
    let w = input.w;
    let mut out = Grid::new(w * 2, input.h * 2);
    out.data
        .par_chunks_mut(out.w)
        .with_max_len(1 << 20)
        .enumerate()
        .for_each(|(oy, row)| {
            let y = oy / 2;
            let sub = oy % 2;
            for x in 0..w {
                let b = px(input, x as isize, y as isize - 1);
                let d = px(input, x as isize - 1, y as isize);
                let e = input.data[y * w + x];
                let f = px(input, x as isize + 1, y as isize);
                let h = px(input, x as isize, y as isize + 1);
                let picks = epx_pick4(b, d, e, f, h);
                row[2 * x] = picks[2 * sub];
                row[2 * x + 1] = picks[2 * sub + 1];
            }
        });
    out
}

/// EPX plane rule: scalar reference, AVX2 inner blocks when the CPU has them
/// (bit-exact either way; `FCA_SIMD=0` pins the scalar path before first use).
#[allow(clippy::if_same_then_else)] // branch structure mirrors the C reference bit-for-bit
pub fn scale2x_scalar(input: &Grid) -> Grid {
    #[cfg(target_arch = "x86_64")]
    if crate::simd::has_avx2() && input.w > 2 && input.h > 0 {
        let mut out = Grid::new(input.w * 2, input.h * 2);
        let h = input.h;
        let w2 = out.w;
        out.data
            .par_chunks_mut(2 * w2)
            .with_max_len(1 << 21)
            .enumerate()
            .for_each(|(y, pair)| {
                let (o0, o1) = pair.split_at_mut(w2);
                let yt = if y == 0 { 0 } else { y - 1 };
                let yb = if y + 1 < h { y + 1 } else { h - 1 };
                // SAFETY: AVX2 presence is checked by has_avx2(); the two
                // output rows are disjoint halves of this task's chunk.
                unsafe {
                    crate::simd::epx_rows_avx2(input.row(yt), input.row(y), input.row(yb), o0, o1);
                }
            });
        return out;
    }
    epx_scalar_ref(input)
}

/// Forced-scalar fuzzy plane rule — the bit-exactness reference for the AVX2
/// path and the fallback on CPUs without AVX2.
pub fn fuzzy_scalar_ref(input: &Grid) -> Grid {
    let w = input.w;
    let mut out = Grid::new(w * 2, input.h * 2);
    out.data
        .par_chunks_mut(out.w)
        .with_max_len(1 << 20)
        .enumerate()
        .for_each(|(oy, row)| {
            let y = oy / 2;
            let sub = oy % 2;
            for x in 0..w {
                let b = px(input, x as isize, y as isize - 1);
                let d = px(input, x as isize - 1, y as isize);
                let e = input.data[y * w + x];
                let f = px(input, x as isize + 1, y as isize);
                let h = px(input, x as isize, y as isize + 1);
                let picks = fz_pick4(b, d, e, f, h);
                row[2 * x] = picks[2 * sub];
                row[2 * x + 1] = picks[2 * sub + 1];
            }
        });
    out
}

/// Fuzzy plane rule: scalar reference, AVX2 inner blocks when the CPU has
/// them (bit-exact either way; `FCA_SIMD=0` pins the scalar path).
pub fn scale2x_fuzzy(input: &Grid) -> Grid {
    #[cfg(target_arch = "x86_64")]
    if crate::simd::has_avx2() && input.w > 2 && input.h > 0 {
        let mut out = Grid::new(input.w * 2, input.h * 2);
        let h = input.h;
        let w2 = out.w;
        out.data
            .par_chunks_mut(2 * w2)
            .with_max_len(1 << 21)
            .enumerate()
            .for_each(|(y, pair)| {
                let (o0, o1) = pair.split_at_mut(w2);
                let yt = if y == 0 { 0 } else { y - 1 };
                let yb = if y + 1 < h { y + 1 } else { h - 1 };
                // SAFETY: AVX2 presence is checked by has_avx2(); the two
                // output rows are disjoint halves of this task's chunk.
                unsafe {
                    crate::simd::fuzzy_rows_avx2(
                        input.row(yt),
                        input.row(y),
                        input.row(yb),
                        o0,
                        o1,
                    );
                }
            });
        return out;
    }
    fuzzy_scalar_ref(input)
}

/// Fuzzy + diagonal corners and weighted pair blend: thin 1px diagonal lines survive.
pub fn scale2x_xbr(input: &Grid) -> Grid {
    let w = input.w;
    let mut out = Grid::new(w * 2, input.h * 2);
    out.data
        .par_chunks_mut(out.w)
        .with_max_len(1 << 20)
        .enumerate()
        .for_each(|(oy, row)| {
            let y = oy / 2;
            let sub = oy % 2;
            for x in 0..w {
                let a = px(input, x as isize - 1, y as isize - 1);
                let b = px(input, x as isize, y as isize - 1);
                let c = px(input, x as isize + 1, y as isize - 1);
                let d = px(input, x as isize - 1, y as isize);
                let e = input.data[y * w + x];
                let f = px(input, x as isize + 1, y as isize);
                let g = px(input, x as isize - 1, y as isize + 1);
                let h = px(input, x as isize, y as isize + 1);
                let i = px(input, x as isize + 1, y as isize + 1);
                let picks = xbr_pick4(a, b, c, d, e, f, g, h, i);
                row[2 * x] = picks[2 * sub];
                row[2 * x + 1] = picks[2 * sub + 1];
            }
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn membership_triangle() {
        assert_eq!(m_fuzzy(0), 255);
        assert_eq!(m_fuzzy(40), 0);
        assert_eq!(m_fuzzy(41), 0);
        assert_eq!(m_fuzzy(8), 255 - ((8 * 51) >> 3)); // 204
    }

    #[test]
    fn tolerance_equality() {
        assert!(eq_tol(100, 120));
        assert!(!eq_tol(100, 121));
        assert!(eq_tol(7, 7));
    }

    #[test]
    fn flat_grid_stays_flat() {
        let g = Grid::from_vec(4, 3, vec![128; 12]);
        for f in [scale2x_scalar, scale2x_fuzzy, scale2x_xbr] {
            let o = f(&g);
            assert_eq!(o.w, 8);
            assert_eq!(o.h, 6);
            assert!(o.data.iter().all(|&v| v == 128));
        }
    }

    #[test]
    fn no_ringing_beyond_3x3_minmax() {
        // hard vertical edge
        let mut g = Grid::new(6, 4);
        for y in 0..4 {
            for x in 0..6 {
                g.data[y * 6 + x] = if x < 3 { 20 } else { 220 };
            }
        }
        for f in [scale2x_scalar, scale2x_fuzzy, scale2x_xbr] {
            let o = f(&g);
            assert!(o.data.iter().all(|&v| (20..=220).contains(&v)));
        }
    }

    #[test]
    fn edge_clamped_access() {
        let mut g = Grid::new(2, 2);
        g.data.copy_from_slice(&[1, 2, 3, 4]);
        assert_eq!(px(&g, -5, -5), 1);
        assert_eq!(px(&g, 99, 0), 2);
        assert_eq!(px(&g, 0, 99), 3);
    }
}
