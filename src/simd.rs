//! Hand-written AVX2 kernels for the cellular rules, ported back from the C
//! prototype (`include/fca/rules.hpp`, namespace `avx2`).
//!
//! Contract, same as in C: the vector path is **bit-exact** with the scalar
//! reference in [`crate::rules`]. The scalar code stays the single source of
//! truth; tests in this module compare both paths on random grids for every
//! width/height around the 32-pixel block boundary.
//!
//! Dispatch: runtime-detected once per process (`std::arch::
//! is_x86_feature_detected!`). `FCA_SIMD=0` forces the scalar path (used by
//! the benchmark harness to measure the SIMD gain on the same machine).
//! Non-x86_64 targets always get the scalar path.

#[cfg(target_arch = "x86_64")]
pub use x86::has_avx2;

#[cfg(not(target_arch = "x86_64"))]
pub fn has_avx2() -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    //! Kernels operate on row slices: `fuzzy_rows(top, mid, bot, out0, out1)`
    //! upscales source row `mid` into output rows `out0`/`out1` (each `2*w`).
    //! The caller (rules.rs) feeds edge-clamped `top`/`bot` rows and hands out
    //! disjoint output chunks to rayon.

    use std::arch::x86_64::*;
    use std::sync::OnceLock;

    /// Cached runtime dispatch decision. `FCA_SIMD=0` forces scalar.
    pub fn has_avx2() -> bool {
        static AVX2: OnceLock<bool> = OnceLock::new();
        *AVX2.get_or_init(|| {
            if std::env::var_os("FCA_SIMD")
                .map(|v| v == "0")
                .unwrap_or(false)
            {
                return false;
            }
            std::arch::is_x86_feature_detected!("avx2")
        })
    }

    #[inline(always)]
    unsafe fn ones_v() -> __m256i {
        _mm256_set1_epi8(-1)
    }
    #[inline(always)]
    unsafe fn zero_v() -> __m256i {
        _mm256_setzero_si256()
    }
    #[inline(always)]
    unsafe fn tol_v() -> __m256i {
        _mm256_set1_epi8(crate::rules::K_TOL as i8)
    }

    /// 0xFF where |a-b| <= K_TOL (saturated abs diff — scalar `eq_tol`).
    #[inline(always)]
    unsafe fn eq_tol_v(a: __m256i, b: __m256i) -> __m256i {
        let d1 = _mm256_subs_epu8(a, b);
        let d2 = _mm256_subs_epu8(b, a);
        let absd = _mm256_or_si256(d1, d2);
        let over = _mm256_subs_epu8(absd, tol_v());
        _mm256_cmpeq_epi8(over, zero_v())
    }

    /// Priority pick: m1 -> v1, else m2 -> v2, else m3 -> v3, else e.
    #[inline(always)]
    unsafe fn pick3(
        m1: __m256i,
        v1: __m256i,
        m2: __m256i,
        v2: __m256i,
        m3: __m256i,
        v3: __m256i,
        e: __m256i,
    ) -> __m256i {
        let m2p = _mm256_andnot_si256(m1, m2);
        let m3p = _mm256_andnot_si256(_mm256_or_si256(m1, m2), m3);
        let t = _mm256_blendv_epi8(e, v1, m1);
        let t = _mm256_blendv_epi8(t, v2, m2p);
        _mm256_blendv_epi8(t, v3, m3p)
    }

    /// |a-b| saturated (u8).
    #[inline(always)]
    unsafe fn fz_absdiff(a: __m256i, b: __m256i) -> __m256i {
        _mm256_or_si256(_mm256_subs_epu8(a, b), _mm256_subs_epu8(b, a))
    }

    /// m = 255 - min((d*51)>>3, 255), unpacked into lo/hi 16-bit lanes.
    /// Bit-exact with scalar `m_fuzzy` for every d in 0..=255.
    #[inline(always)]
    unsafe fn fz_membership(d8: __m256i) -> (__m256i, __m256i) {
        let z = zero_v();
        let w51 = _mm256_set1_epi16(51);
        let c255 = _mm256_set1_epi16(255);
        let lo = _mm256_mullo_epi16(_mm256_unpacklo_epi8(d8, z), w51);
        let hi = _mm256_mullo_epi16(_mm256_unpackhi_epi8(d8, z), w51);
        let lo = _mm256_min_epu16(_mm256_srli_epi16(lo, 3), c255);
        let hi = _mm256_min_epu16(_mm256_srli_epi16(hi, 3), c255);
        (_mm256_sub_epi16(c255, lo), _mm256_sub_epi16(c255, hi))
    }

    /// w = (m1 * m2 * m3) >> 16 in lo/hi lanes. m1*m2 <= 65025 fits in u16
    /// (mullo exact); mulhi_epu16 then computes ((m1*m2)*m3) >> 16 exactly
    /// like the scalar i32 expression.
    #[inline(always)]
    unsafe fn fz_weight(
        m1lo: __m256i,
        m1hi: __m256i,
        m2lo: __m256i,
        m2hi: __m256i,
        m3lo: __m256i,
        m3hi: __m256i,
    ) -> (__m256i, __m256i) {
        let wlo = _mm256_mullo_epi16(m1lo, m2lo);
        let whi = _mm256_mullo_epi16(m1hi, m2hi);
        (_mm256_mulhi_epu16(wlo, m3lo), _mm256_mulhi_epu16(whi, m3hi))
    }

    /// out = sat_u8(E + ((w+1)/2 * (a-E)) >> 7) — scalar `fz_pick`.
    /// |w2*(a-E)| <= 128*255 fits in i16, so mullo is exact; `srai_epi16` is
    /// the arithmetic shift; packus saturates like `clamp(0, 255)`.
    #[inline(always)]
    unsafe fn fz_blend(a8: __m256i, e8: __m256i, wlo: __m256i, whi: __m256i) -> __m256i {
        let z = zero_v();
        let alo = _mm256_unpacklo_epi8(a8, z);
        let ahi = _mm256_unpackhi_epi8(a8, z);
        let elo = _mm256_unpacklo_epi8(e8, z);
        let ehi = _mm256_unpackhi_epi8(e8, z);
        let dal = _mm256_sub_epi16(alo, elo);
        let dah = _mm256_sub_epi16(ahi, ehi);
        let one = _mm256_set1_epi16(1);
        let w2l = _mm256_srli_epi16(_mm256_add_epi16(wlo, one), 1);
        let w2h = _mm256_srli_epi16(_mm256_add_epi16(whi, one), 1);
        let plo = _mm256_add_epi16(elo, _mm256_srai_epi16(_mm256_mullo_epi16(w2l, dal), 7));
        let phi = _mm256_add_epi16(ehi, _mm256_srai_epi16(_mm256_mullo_epi16(w2h, dah), 7));
        _mm256_packus_epi16(plo, phi)
    }

    /// Interleave 4 result vectors into two output rows at source column x0:
    /// `out0[2i] = e0[i], out0[2i+1] = e1[i], out1[2i] = e2[i], out1[2i+1] = e3[i]`.
    #[inline(always)]
    unsafe fn store_e0e3(
        out0: &mut [u8],
        out1: &mut [u8],
        x0: usize,
        e0: __m256i,
        e1: __m256i,
        e2: __m256i,
        e3: __m256i,
    ) {
        let mut b0 = [0u8; 32];
        let mut b1 = [0u8; 32];
        let mut b2 = [0u8; 32];
        let mut b3 = [0u8; 32];
        _mm256_storeu_si256(b0.as_mut_ptr().cast(), e0);
        _mm256_storeu_si256(b1.as_mut_ptr().cast(), e1);
        _mm256_storeu_si256(b2.as_mut_ptr().cast(), e2);
        _mm256_storeu_si256(b3.as_mut_ptr().cast(), e3);
        for i in 0..32 {
            out0[2 * (x0 + i)] = b0[i];
            out0[2 * (x0 + i) + 1] = b1[i];
            out1[2 * (x0 + i)] = b2[i];
            out1[2 * (x0 + i) + 1] = b3[i];
        }
    }

    // ------------------------------------------------------------------ EPX

    /// EPX 32-pixel block. Requires x0 >= 1 and x0 + 32 < w: every load stays
    /// inside the row, so no cross-row reads (the C original allowed a 1-byte
    /// overread on the last block; the port does not).
    #[target_feature(enable = "avx2")]
    unsafe fn epx_block32(
        top: &[u8],
        mid: &[u8],
        bot: &[u8],
        out0: &mut [u8],
        out1: &mut [u8],
        x0: usize,
    ) {
        let b = _mm256_loadu_si256(top.as_ptr().add(x0).cast());
        let h = _mm256_loadu_si256(bot.as_ptr().add(x0).cast());
        let d = _mm256_loadu_si256(mid.as_ptr().add(x0 - 1).cast());
        let e = _mm256_loadu_si256(mid.as_ptr().add(x0).cast());
        let f = _mm256_loadu_si256(mid.as_ptr().add(x0 + 1).cast());

        let c_bh = eq_tol_v(b, h);
        let c_df = eq_tol_v(d, f);
        let cond = _mm256_andnot_si256(_mm256_or_si256(c_bh, c_df), ones_v());

        let m_bd = eq_tol_v(b, d);
        let m_bf = eq_tol_v(b, f);
        let m_df = eq_tol_v(d, f);
        let m_hd = eq_tol_v(h, d);
        let m_hf = eq_tol_v(h, f);
        let m_fd = eq_tol_v(f, d);

        let mut e0 = pick3(m_bd, b, m_bf, f, m_df, d, e);
        let mut e1 = pick3(m_bf, f, m_bd, b, m_fd, d, e);
        let mut e2 = pick3(m_hd, d, m_hf, f, m_df, d, e);
        let mut e3 = pick3(m_hf, f, m_hd, d, m_fd, d, e);

        e0 = _mm256_blendv_epi8(e, e0, cond);
        e1 = _mm256_blendv_epi8(e, e1, cond);
        e2 = _mm256_blendv_epi8(e, e2, cond);
        e3 = _mm256_blendv_epi8(e, e3, cond);

        store_e0e3(out0, out1, x0, e0, e1, e2, e3);
    }

    /// AVX2 EPX, one source row -> two output rows. Bit-exact with
    /// `rules::epx_pick4` scalar paths (x=0 and the tail are scalar to keep
    /// the x-clamping semantics).
    #[target_feature(enable = "avx2")]
    pub unsafe fn epx_rows(top: &[u8], mid: &[u8], bot: &[u8], out0: &mut [u8], out1: &mut [u8]) {
        let w = mid.len();
        {
            // x = 0: clamped scalar (same as rules::scale2x_scalar)
            let (b, d, e, f, h) = (top[0], mid[0], mid[0], mid[1.min(w - 1)], bot[0]);
            let picks = crate::rules::epx_pick4(b, d, e, f, h);
            out0[0] = picks[0];
            out0[1] = picks[1];
            out1[0] = picks[2];
            out1[1] = picks[3];
        }
        let mut x = 1;
        while x + 32 < w {
            epx_block32(top, mid, bot, out0, out1, x);
            x += 32;
        }
        while x < w {
            let (b, d, e, f, h) = (top[x], mid[x - 1], mid[x], mid[(x + 1).min(w - 1)], bot[x]);
            let picks = crate::rules::epx_pick4(b, d, e, f, h);
            out0[2 * x] = picks[0];
            out0[2 * x + 1] = picks[1];
            out1[2 * x] = picks[2];
            out1[2 * x + 1] = picks[3];
            x += 1;
        }
    }

    // ---------------------------------------------------------------- fuzzy

    /// Fuzzy 32-pixel block. Same in-row requirement as `epx_block32`.
    #[target_feature(enable = "avx2")]
    unsafe fn fz_block32(
        top: &[u8],
        mid: &[u8],
        bot: &[u8],
        out0: &mut [u8],
        out1: &mut [u8],
        x0: usize,
    ) {
        let b = _mm256_loadu_si256(top.as_ptr().add(x0).cast());
        let h = _mm256_loadu_si256(bot.as_ptr().add(x0).cast());
        let d = _mm256_loadu_si256(mid.as_ptr().add(x0 - 1).cast());
        let e = _mm256_loadu_si256(mid.as_ptr().add(x0).cast());
        let f = _mm256_loadu_si256(mid.as_ptr().add(x0 + 1).cast());

        let r_bd = fz_absdiff(b, d);
        let r_fb = fz_absdiff(f, b);
        let r_hd = fz_absdiff(h, d);
        let r_df = fz_absdiff(d, f);
        let r_hb = fz_absdiff(h, b);
        let r_fh = fz_absdiff(f, h);

        let (m_bd_l, m_bd_h) = fz_membership(r_bd);
        let (m_fb_l, m_fb_h) = fz_membership(r_fb);
        let (m_hd_l, m_hd_h) = fz_membership(r_hd);
        let (m_df_l, m_df_h) = fz_membership(r_df);
        let (m_hb_l, m_hb_h) = fz_membership(r_hb);
        let (m_fh_l, m_fh_h) = fz_membership(r_fh);

        let c255 = _mm256_set1_epi16(255);
        let n_fb_l = _mm256_sub_epi16(c255, m_fb_l);
        let n_fb_h = _mm256_sub_epi16(c255, m_fb_h);
        let n_hd_l = _mm256_sub_epi16(c255, m_hd_l);
        let n_hd_h = _mm256_sub_epi16(c255, m_hd_h);
        let n_bd_l = _mm256_sub_epi16(c255, m_bd_l);
        let n_bd_h = _mm256_sub_epi16(c255, m_bd_h);
        let n_df_l = _mm256_sub_epi16(c255, m_df_l);
        let n_df_h = _mm256_sub_epi16(c255, m_df_h);
        let n_fh_l = _mm256_sub_epi16(c255, m_fh_l);
        let n_fh_h = _mm256_sub_epi16(c255, m_fh_h);
        let n_hb_l = _mm256_sub_epi16(c255, m_hb_l);
        let n_hb_h = _mm256_sub_epi16(c255, m_hb_h);

        // w0 = m(BD)*m'(FB)*m'(HD); w1 = m(FB)*m'(DF)*m'(HB)
        // w2 = m(HD)*m'(BD)*m'(FH); w3 = m(FH)*m'(DF)*m'(HB)
        let (w0l, w0h) = fz_weight(m_bd_l, m_bd_h, n_fb_l, n_fb_h, n_hd_l, n_hd_h);
        let (w1l, w1h) = fz_weight(m_fb_l, m_fb_h, n_df_l, n_df_h, n_hb_l, n_hb_h);
        let (w2l, w2h) = fz_weight(m_hd_l, m_hd_h, n_bd_l, n_bd_h, n_fh_l, n_fh_h);
        let (w3l, w3h) = fz_weight(m_fh_l, m_fh_h, n_df_l, n_df_h, n_hb_l, n_hb_h);

        let e0 = fz_blend(b, e, w0l, w0h);
        let e1 = fz_blend(f, e, w1l, w1h);
        let e2 = fz_blend(d, e, w2l, w2h);
        let e3 = fz_blend(f, e, w3l, w3h);

        store_e0e3(out0, out1, x0, e0, e1, e2, e3);
    }

    /// AVX2 fuzzy, one source row -> two output rows. Bit-exact with
    /// `rules::scale2x_fuzzy`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn fuzzy_rows(top: &[u8], mid: &[u8], bot: &[u8], out0: &mut [u8], out1: &mut [u8]) {
        let w = mid.len();
        {
            let (b, d, e, f, h) = (top[0], mid[0], mid[0], mid[1.min(w - 1)], bot[0]);
            let picks = crate::rules::fz_pick4(b, d, e, f, h);
            out0[0] = picks[0];
            out0[1] = picks[1];
            out1[0] = picks[2];
            out1[1] = picks[3];
        }
        let mut x = 1;
        while x + 32 < w {
            fz_block32(top, mid, bot, out0, out1, x);
            x += 32;
        }
        while x < w {
            let (b, d, e, f, h) = (top[x], mid[x - 1], mid[x], mid[(x + 1).min(w - 1)], bot[x]);
            let picks = crate::rules::fz_pick4(b, d, e, f, h);
            out0[2 * x] = picks[0];
            out0[2 * x + 1] = picks[1];
            out1[2 * x] = picks[2];
            out1[2 * x + 1] = picks[3];
            x += 1;
        }
    }
}

// --------------------------------------------------------------------------
// row-level dispatch used by rules.rs (called only when has_avx2() is true)
// --------------------------------------------------------------------------

/// AVX2 fuzzy: `mid` row -> two output rows. SAFETY: requires AVX2 (checked by
/// the caller via `has_avx2()`).
///
/// # Safety
/// Must only be called when the CPU supports AVX2.
#[cfg(target_arch = "x86_64")]
pub unsafe fn fuzzy_rows_avx2(
    top: &[u8],
    mid: &[u8],
    bot: &[u8],
    out0: &mut [u8],
    out1: &mut [u8],
) {
    x86::fuzzy_rows(top, mid, bot, out0, out1)
}

/// AVX2 EPX: `mid` row -> two output rows.
///
/// # Safety
/// Must only be called when the CPU supports AVX2.
#[cfg(target_arch = "x86_64")]
pub unsafe fn epx_rows_avx2(top: &[u8], mid: &[u8], bot: &[u8], out0: &mut [u8], out1: &mut [u8]) {
    x86::epx_rows(top, mid, bot, out0, out1)
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use crate::grid::Grid;
    use crate::rules::{epx_scalar_ref, fuzzy_scalar_ref, scale2x_fuzzy, scale2x_scalar};
    use crate::simd::has_avx2;

    /// Deterministic LCG so failures are reproducible.
    fn lcg_grid(w: usize, h: usize, seed: u64) -> Grid {
        let mut s = seed;
        let data = (0..w * h)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 33) as u8
            })
            .collect();
        Grid::from_vec(w, h, data)
    }

    #[test]
    fn simd_bit_exact_fuzzy() {
        if !has_avx2() {
            eprintln!("skipped: no AVX2 on this CPU");
            return;
        }
        // widths around every 32-block boundary, heights around row edges
        for w in [
            1, 2, 3, 31, 32, 33, 34, 63, 64, 65, 95, 96, 97, 127, 128, 129, 255, 320,
        ] {
            for h in [1, 2, 3, 5, 8, 17, 64] {
                let g = lcg_grid(w, h, 0x5EED + w as u64 * 31 + h as u64);
                let scalar = fuzzy_scalar_ref(&g);
                let auto = scale2x_fuzzy(&g); // AVX2 path on this CPU
                assert_eq!(scalar.data, auto.data, "fuzzy mismatch at {w}x{h}");
            }
        }
    }

    #[test]
    fn simd_bit_exact_epx() {
        if !has_avx2() {
            eprintln!("skipped: no AVX2 on this CPU");
            return;
        }
        for w in [1, 2, 31, 32, 33, 64, 65, 97, 129] {
            for h in [1, 2, 5, 9, 33] {
                let g = lcg_grid(w, h, 42 + w as u64);
                let scalar = epx_scalar_ref(&g);
                let auto = scale2x_scalar(&g);
                assert_eq!(scalar.data, auto.data, "epx mismatch at {w}x{h}");
            }
        }
    }

    #[test]
    fn e0e3_interleave_matches_scalar_row_pair() {
        // one explicit row-pair check: e0/e1 land on the even output row,
        // e2/e3 on the odd one
        if !has_avx2() {
            return;
        }
        let w = 40;
        let mut g = lcg_grid(w, 3, 7);
        for v in g.data[..w].iter_mut() {
            *v = 10;
        }
        let scalar = fuzzy_scalar_ref(&g);
        let auto = scale2x_fuzzy(&g);
        assert_eq!(scalar.data, auto.data);
    }
}
