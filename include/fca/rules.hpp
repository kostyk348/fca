#pragma once
// libfca — scale2x (EPX) cellular rule, scalar reference + AVX2.
// Rule: 3x3 neighborhood -> 4 subpixels, edge-directed picks, tolerance equality.
// This is the same rule as fca_scale2x.glsl (GPU demo), now on CPU in SIMD.

#include "fca/grid.hpp"
#include <cstdint>
#include <cstddef>

namespace fca {
namespace rule {

// tolerance for "cell state match" (~0.08 in 0..1 == 20 in 0..255, matches GLSL EPS)
static constexpr int kTol = 20;

inline bool eq_tol(uint8_t a, uint8_t b) {
    int d = a > b ? a - b : b - a;
    return d <= kTol;
}

// --- clamped neighbor access (out-of-range clamped, like GLSL clamped sampler) ---
inline uint8_t px(const Grid& g, long x, long y) {
    if (x < 0) x = 0;
    if (x >= (long)g.w) x = (long)g.w - 1;
    if (y < 0) y = 0;
    if (y >= (long)g.h) y = (long)g.h - 1;
    return g.data[(size_t)y * g.w + (size_t)x];
}

// ---------------------------------------------------------------- fuzzy variant (EAFAR S5)
// Instead of boolean pick3 we blend by membership degree. Membership is a
// triangle: 255 at d==0, linear to 0 at d>=kFuzzyW (EAFAR mf::triangle(0,W)).
// Fixed-point, deterministic, no cmath. kFuzzyW = 2*tol gives a soft zone
// around the old hard threshold — kills staircase on diagonals and
// frame-to-frame flicker at tolerance boundaries.
static constexpr int kFuzzyW = 40;

inline int m_fuzzy(int d) {
    if (d >= kFuzzyW) return 0;
    return 255 - ((d * 51) >> 3); // (d*51)/8 == d*6.375 == d*(255/40)
}

// smooth pick: out = E + (w*(a-E))/256 via half-shift.
// ((w+1)/2)*(a-E)>>7 is bit-exact-compatible with the AVX2 path
// (max error vs exact /255 is <=1 level, deterministic).
inline uint8_t fz_pick(int w, uint8_t a, uint8_t e) {
    int t = (((w + 1) >> 1) * ((int)a - (int)e)) >> 7;
    int v = (int)e + t;
    if (v < 0) v = 0;
    else if (v > 255) v = 255;
    return (uint8_t)v;
}

// fuzzy version of the EPX e0..e3 picks. Per output: candidate `a` on the
// diagonal line (a,c), contrast perpendicular (b1 vs a, b2 vs c).
// w = m(a~c) * (1-m(b1~a)) * (1-m(b2~c))  — smooth, deterministic.
inline void fz_pick4(uint8_t B, uint8_t D, uint8_t E, uint8_t F, uint8_t H,
                     uint8_t& e0, uint8_t& e1, uint8_t& e2, uint8_t& e3) {
    auto ad = [](int x, int y) { return x > y ? x - y : y - x; };
    // e0: a=B c=D, b1=F b2=H
    int w0 = (m_fuzzy(ad(B, D)) * (255 - m_fuzzy(ad(F, B))) * (255 - m_fuzzy(ad(H, D)))) >> 16;
    // e1: a=F c=B, b1=D b2=H
    int w1 = (m_fuzzy(ad(F, B)) * (255 - m_fuzzy(ad(D, F))) * (255 - m_fuzzy(ad(H, B)))) >> 16;
    // e2: a=D c=H, b1=B b2=F
    int w2 = (m_fuzzy(ad(D, H)) * (255 - m_fuzzy(ad(B, D))) * (255 - m_fuzzy(ad(F, H)))) >> 16;
    // e3: a=F c=H, b1=D b2=B
    int w3 = (m_fuzzy(ad(F, H)) * (255 - m_fuzzy(ad(D, F))) * (255 - m_fuzzy(ad(B, H)))) >> 16;
    e0 = fz_pick(w0, B, E);
    e1 = fz_pick(w1, F, E);
    e2 = fz_pick(w2, D, E);
    e3 = fz_pick(w3, F, E);
}

void scale2x_fuzzy(const Grid& in, Grid& out) {
    out = Grid(in.w * 2, in.h * 2);
#pragma omp parallel for schedule(static)
    for (size_t y = 0; y < in.h; ++y) {
        for (size_t x = 0; x < in.w; ++x) {
            uint8_t B = px(in, (long)x, (long)y - 1);
            uint8_t D = px(in, (long)x - 1, (long)y);
            uint8_t E = in.data[y * in.w + x];
            uint8_t F = px(in, (long)x + 1, (long)y);
            uint8_t H = px(in, (long)x, (long)y + 1);
            uint8_t e0, e1, e2, e3;
            fz_pick4(B, D, E, F, H, e0, e1, e2, e3);
            out.data[(2 * y) * out.w + 2 * x]     = e0;
            out.data[(2 * y) * out.w + 2 * x + 1] = e1;
            out.data[(2 * y + 1) * out.w + 2 * x] = e2;
            out.data[(2 * y + 1) * out.w + 2 * x + 1] = e3;
        }
    }
}

// ------------------------------------------------- xBR-style rule (fuzzy, 8 neighbors)
// Upgrade over scale2x_fuzzy: each subpixel blends its pair (p,q) weighted by
// the diagonal corner `r` closeness, instead of picking one candidate.
//   strength = m(p~q) * (1-m(perp1~p)) * (1-m(perp2~q))   (same as fuzzy weights)
//   blend    = (w_p*p + w_q*q)/(w_p+w_q),  w = m(r~.)+1
//   out      = mix(E, blend, strength)
// The corner disambiguates thin diagonal lines (1px lines) that the plain
// cross rule breaks. Results stay inside the 3x3 min/max => no ringing by
// construction. Deterministic, integer-only.
inline uint8_t fz_blend2(int p, int q, int wp, int wq) {
    int num = p * wp + q * wq;
    int den = wp + wq;
    return (uint8_t)((num + den / 2) / den);
}

inline void xbr_pick4(uint8_t A, uint8_t B, uint8_t C,
                      uint8_t D, uint8_t E, uint8_t F,
                      uint8_t G, uint8_t H, uint8_t I,
                      uint8_t& e0, uint8_t& e1, uint8_t& e2, uint8_t& e3) {
    auto ad = [](int x, int y) { return x > y ? x - y : y - x; };
    // e0: p=B q=D r=A, perp1=F perp2=H
    {
        int s = (m_fuzzy(ad(B, D)) * (255 - m_fuzzy(ad(F, B))) * (255 - m_fuzzy(ad(H, D)))) >> 16;
        uint8_t b = fz_blend2(B, D, m_fuzzy(ad(A, B)) + 1, m_fuzzy(ad(A, D)) + 1);
        e0 = fz_pick(s, b, E);
    }
    // e1: p=F q=B r=C, perp1=D perp2=H
    {
        int s = (m_fuzzy(ad(F, B)) * (255 - m_fuzzy(ad(D, F))) * (255 - m_fuzzy(ad(H, B)))) >> 16;
        uint8_t b = fz_blend2(F, B, m_fuzzy(ad(C, F)) + 1, m_fuzzy(ad(C, B)) + 1);
        e1 = fz_pick(s, b, E);
    }
    // e2: p=D q=H r=G, perp1=B perp2=F
    {
        int s = (m_fuzzy(ad(D, H)) * (255 - m_fuzzy(ad(B, D))) * (255 - m_fuzzy(ad(F, H)))) >> 16;
        uint8_t b = fz_blend2(D, H, m_fuzzy(ad(G, D)) + 1, m_fuzzy(ad(G, H)) + 1);
        e2 = fz_pick(s, b, E);
    }
    // e3: p=F q=H r=I, perp1=D perp2=B
    {
        int s = (m_fuzzy(ad(F, H)) * (255 - m_fuzzy(ad(D, F))) * (255 - m_fuzzy(ad(B, H)))) >> 16;
        uint8_t b = fz_blend2(F, H, m_fuzzy(ad(I, F)) + 1, m_fuzzy(ad(I, H)) + 1);
        e3 = fz_pick(s, b, E);
    }
}

void scale2x_xbr(const Grid& in, Grid& out) {
    out = Grid(in.w * 2, in.h * 2);
#pragma omp parallel for schedule(static)
    for (size_t y = 0; y < in.h; ++y) {
        for (size_t x = 0; x < in.w; ++x) {
            uint8_t A = px(in, (long)x - 1, (long)y - 1);
            uint8_t B = px(in, (long)x,     (long)y - 1);
            uint8_t C = px(in, (long)x + 1, (long)y - 1);
            uint8_t D = px(in, (long)x - 1, (long)y);
            uint8_t E = in.data[y * in.w + x];
            uint8_t F = px(in, (long)x + 1, (long)y);
            uint8_t G = px(in, (long)x - 1, (long)y + 1);
            uint8_t H = px(in, (long)x,     (long)y + 1);
            uint8_t I = px(in, (long)x + 1, (long)y + 1);
            uint8_t e0, e1, e2, e3;
            xbr_pick4(A, B, C, D, E, F, G, H, I, e0, e1, e2, e3);
            out.data[(2 * y) * out.w + 2 * x]     = e0;
            out.data[(2 * y) * out.w + 2 * x + 1] = e1;
            out.data[(2 * y + 1) * out.w + 2 * x] = e2;
            out.data[(2 * y + 1) * out.w + 2 * x + 1] = e3;
        }
    }
}

// ---------------------------------------------------------------- scalar reference
void scale2x_scalar(const Grid& in, Grid& out) {
    out = Grid(in.w * 2, in.h * 2);
#pragma omp parallel for schedule(static)
    for (size_t y = 0; y < in.h; ++y) {
        for (size_t x = 0; x < in.w; ++x) {
            uint8_t B = px(in, (long)x, (long)y - 1);
            uint8_t D = px(in, (long)x - 1, (long)y);
            uint8_t E = in.data[y * in.w + x];
            uint8_t F = px(in, (long)x + 1, (long)y);
            uint8_t H = px(in, (long)x, (long)y + 1);

            uint8_t e0, e1, e2, e3;
            if (!eq_tol(B, H) && !eq_tol(D, F)) {
                // EPX picks with strict priority
                e0 = eq_tol(B, D) ? B : (eq_tol(B, F) ? F : (eq_tol(D, F) ? D : E));
                e1 = eq_tol(B, F) ? F : (eq_tol(B, D) ? B : (eq_tol(F, D) ? D : E));
                e2 = eq_tol(H, D) ? D : (eq_tol(H, F) ? F : (eq_tol(D, F) ? D : E));
                e3 = eq_tol(H, F) ? F : (eq_tol(H, D) ? D : (eq_tol(F, D) ? D : E));
            } else {
                e0 = e1 = e2 = e3 = E;
            }
            out.data[(2 * y) * out.w + 2 * x]     = e0;
            out.data[(2 * y) * out.w + 2 * x + 1] = e1;
            out.data[(2 * y + 1) * out.w + 2 * x] = e2;
            out.data[(2 * y + 1) * out.w + 2 * x + 1] = e3;
        }
    }
}

// ------------------------------------------------------------------- AVX2 variant
#if defined(__AVX2__)

#include <immintrin.h>

namespace avx2 {

static inline __m256i ones_v()   { return _mm256_set1_epi8(0xFF); }
static inline __m256i zero_v()   { return _mm256_setzero_si256(); }
static inline __m256i tol_v()    { return _mm256_set1_epi8((char)kTol); }

// tolerance-equality mask: 0xFF where |a-b| <= T
static inline __m256i eq_tol_v(__m256i a, __m256i b) {
    __m256i d1 = _mm256_subs_epu8(a, b);
    __m256i d2 = _mm256_subs_epu8(b, a);
    __m256i absd = _mm256_or_si256(d1, d2);          // |a-b| (saturated)
    __m256i over = _mm256_subs_epu8(absd, tol_v());  // >0 where |a-b|>T
    return _mm256_cmpeq_epi8(over, zero_v());        // 0xFF where |a-b|<=T
}

// priority pick: m1 ? v1 : (m2 ? v2 : (m3 ? v3 : e)), masks from cmpeq (0xFF/0x00)
static inline __m256i pick3(__m256i m1, __m256i v1,
                            __m256i m2, __m256i v2,
                            __m256i m3, __m256i v3,
                            __m256i e) {
    __m256i m2p = _mm256_andnot_si256(m1, m2);
    __m256i m3p = _mm256_andnot_si256(_mm256_or_si256(m1, m2), m3);
    __m256i t = _mm256_blendv_epi8(e, v1, m1);
    t = _mm256_blendv_epi8(t, v2, m2p);
    t = _mm256_blendv_epi8(t, v3, m3p);
    return t;
}

// scale 32 pixels at (y, x0..x0+31); x0 in [1, w-32] guaranteed; vertical clamp applied
static inline void scale2x_block32(Grid& out, size_t y, size_t x0,
                                   const uint8_t* top, const uint8_t* mid, const uint8_t* bot) {
    __m256i B = _mm256_loadu_si256((const __m256i*)(top + x0));
    __m256i H = _mm256_loadu_si256((const __m256i*)(bot + x0));
    __m256i D = _mm256_loadu_si256((const __m256i*)(mid + x0 - 1));
    __m256i E = _mm256_loadu_si256((const __m256i*)(mid + x0));
    __m256i F = _mm256_loadu_si256((const __m256i*)(mid + x0 + 1));

    __m256i c_bh = eq_tol_v(B, H);
    __m256i c_df = eq_tol_v(D, F);
    // condition: !eq(B,H) && !eq(D,F)
    __m256i cond = _mm256_andnot_si256(_mm256_or_si256(c_bh, c_df), ones_v());

    __m256i m_BD = eq_tol_v(B, D), m_BF = eq_tol_v(B, F), m_DF = eq_tol_v(D, F);
    __m256i m_HD = eq_tol_v(H, D), m_HF = eq_tol_v(H, F), m_FD = eq_tol_v(F, D);

    __m256i e0 = pick3(m_BD, B, m_BF, F, m_DF, D, E);
    __m256i e1 = pick3(m_BF, F, m_BD, B, m_FD, D, E);
    __m256i e2 = pick3(m_HD, D, m_HF, F, m_DF, D, E);
    __m256i e3 = pick3(m_HF, F, m_HD, D, m_FD, D, E);

    e0 = _mm256_blendv_epi8(E, e0, cond);
    e1 = _mm256_blendv_epi8(E, e1, cond);
    e2 = _mm256_blendv_epi8(E, e2, cond);
    e3 = _mm256_blendv_epi8(E, e3, cond);

    uint8_t b0[32], b1[32], b2[32], b3[32];
    _mm256_storeu_si256((__m256i*)b0, e0);
    _mm256_storeu_si256((__m256i*)b1, e1);
    _mm256_storeu_si256((__m256i*)b2, e2);
    _mm256_storeu_si256((__m256i*)b3, e3);

    uint8_t* o0 = out.row(2 * y) + 2 * x0;
    uint8_t* o1 = out.row(2 * y + 1) + 2 * x0;
    for (int i = 0; i < 32; ++i) {
        o0[2 * i]     = b0[i];
        o0[2 * i + 1] = b1[i];
        o1[2 * i]     = b2[i];
        o1[2 * i + 1] = b3[i];
    }
}

void scale2x_avx2(const Grid& in, Grid& out) {
    out = Grid(in.w * 2, in.h * 2);
    if (in.w == 0 || in.h == 0) return;

#pragma omp parallel for schedule(static)
    for (size_t y = 0; y < in.h; ++y) {
        size_t yt = y == 0 ? 0 : y - 1;
        size_t yb = y + 1 < in.h ? y + 1 : in.h - 1;
        const uint8_t* top = in.row(yt);
        const uint8_t* mid = in.row(y);
        const uint8_t* bot = in.row(yb);

        // edge x=0 scalar (clamped)
        {
            uint8_t B = px(in, 0, (long)y - 1);
            uint8_t D = px(in, -1, (long)y);
            uint8_t E = in.data[y * in.w];
            uint8_t F = px(in, 1, (long)y);
            uint8_t H = px(in, 0, (long)y + 1);
            uint8_t e0, e1, e2, e3;
            if (!eq_tol(B, H) && !eq_tol(D, F)) {
                e0 = eq_tol(B, D) ? B : (eq_tol(B, F) ? F : (eq_tol(D, F) ? D : E));
                e1 = eq_tol(B, F) ? F : (eq_tol(B, D) ? B : (eq_tol(F, D) ? D : E));
                e2 = eq_tol(H, D) ? D : (eq_tol(H, F) ? F : (eq_tol(D, F) ? D : E));
                e3 = eq_tol(H, F) ? F : (eq_tol(H, D) ? D : (eq_tol(F, D) ? D : E));
            } else { e0 = e1 = e2 = e3 = E; }
            out.data[(2 * y) * out.w]     = e0;
            out.data[(2 * y) * out.w + 1] = e1;
            out.data[(2 * y + 1) * out.w] = e2;
            out.data[(2 * y + 1) * out.w + 1] = e3;
        }

        size_t x = 1;
        for (; x + 31 < in.w; x += 32)
            scale2x_block32(out, y, x, top, mid, bot);

        // tail scalar
        for (; x < in.w; ++x) {
            uint8_t B = px(in, (long)x, (long)y - 1);
            uint8_t D = px(in, (long)x - 1, (long)y);
            uint8_t E = in.data[y * in.w + x];
            uint8_t F = px(in, (long)x + 1, (long)y);
            uint8_t H = px(in, (long)x, (long)y + 1);
            uint8_t e0, e1, e2, e3;
            if (!eq_tol(B, H) && !eq_tol(D, F)) {
                e0 = eq_tol(B, D) ? B : (eq_tol(B, F) ? F : (eq_tol(D, F) ? D : E));
                e1 = eq_tol(B, F) ? F : (eq_tol(B, D) ? B : (eq_tol(F, D) ? D : E));
                e2 = eq_tol(H, D) ? D : (eq_tol(H, F) ? F : (eq_tol(D, F) ? D : E));
                e3 = eq_tol(H, F) ? F : (eq_tol(H, D) ? D : (eq_tol(F, D) ? D : E));
            } else { e0 = e1 = e2 = e3 = E; }
            out.data[(2 * y) * out.w + 2 * x]     = e0;
            out.data[(2 * y) * out.w + 2 * x + 1] = e1;
            out.data[(2 * y + 1) * out.w + 2 * x] = e2;
            out.data[(2 * y + 1) * out.w + 2 * x + 1] = e3;
        }
    }
}

// ------------------------------------------------- fuzzy AVX2 (bit-exact vs scale2x_fuzzy)
static inline __m256i fz_absdiff(__m256i a, __m256i b) {
    return _mm256_or_si256(_mm256_subs_epu8(a, b), _mm256_subs_epu8(b, a));
}

// m16 = 255 - min(d8*51>>3, 255), unpacked into lo/hi 16-bit lanes
static inline void fz_membership(__m256i d8, __m256i& lo, __m256i& hi) {
    __m256i z = _mm256_setzero_si256();
    __m256i w51 = _mm256_set1_epi16(51);
    __m256i c255 = _mm256_set1_epi16(255);
    lo = _mm256_mullo_epi16(_mm256_unpacklo_epi8(d8, z), w51);
    hi = _mm256_mullo_epi16(_mm256_unpackhi_epi8(d8, z), w51);
    lo = _mm256_min_epu16(_mm256_srli_epi16(lo, 3), c255);
    hi = _mm256_min_epu16(_mm256_srli_epi16(hi, 3), c255);
    lo = _mm256_sub_epi16(c255, lo);
    hi = _mm256_sub_epi16(c255, hi);
}

// w = (m1*m2*m3)>>16 in lo/hi lanes (m1*m2 < 2^16, so mullo holds full product)
static inline void fz_weight(__m256i m1lo, __m256i m1hi, __m256i m2lo, __m256i m2hi,
                             __m256i m3lo, __m256i m3hi, __m256i& wlo, __m256i& whi) {
    wlo = _mm256_mullo_epi16(m1lo, m2lo);
    whi = _mm256_mullo_epi16(m1hi, m2hi);
    wlo = _mm256_mulhi_epu16(wlo, m3lo);
    whi = _mm256_mulhi_epu16(whi, m3hi);
}

// out8 = sat(E + ((w+1)/2)*(a-E)>>7) — same arithmetic as scalar fz_pick
static inline __m256i fz_blend(__m256i a8, __m256i e8, __m256i wlo, __m256i whi) {
    __m256i z = _mm256_setzero_si256();
    __m256i alo = _mm256_unpacklo_epi8(a8, z), ahi = _mm256_unpackhi_epi8(a8, z);
    __m256i elo = _mm256_unpacklo_epi8(e8, z), ehi = _mm256_unpackhi_epi8(e8, z);
    __m256i dal = _mm256_sub_epi16(alo, elo), dah = _mm256_sub_epi16(ahi, ehi);
    __m256i w2l = _mm256_srli_epi16(_mm256_add_epi16(wlo, _mm256_set1_epi16(1)), 1);
    __m256i w2h = _mm256_srli_epi16(_mm256_add_epi16(whi, _mm256_set1_epi16(1)), 1);
    __m256i plo = _mm256_add_epi16(elo, _mm256_srai_epi16(_mm256_mullo_epi16(w2l, dal), 7));
    __m256i phi = _mm256_add_epi16(ehi, _mm256_srai_epi16(_mm256_mullo_epi16(w2h, dah), 7));
    return _mm256_packus_epi16(plo, phi);
}

// fuzzy scale 32 pixels at (y, x0..x0+31); same block layout as scale2x_block32
static inline void fz_block32(Grid& out, size_t y, size_t x0,
                              const uint8_t* top, const uint8_t* mid, const uint8_t* bot) {
    __m256i B = _mm256_loadu_si256((const __m256i*)(top + x0));
    __m256i H = _mm256_loadu_si256((const __m256i*)(bot + x0));
    __m256i D = _mm256_loadu_si256((const __m256i*)(mid + x0 - 1));
    __m256i E = _mm256_loadu_si256((const __m256i*)(mid + x0));
    __m256i F = _mm256_loadu_si256((const __m256i*)(mid + x0 + 1));

    __m256i rBD = fz_absdiff(B, D), rFB = fz_absdiff(F, B), rHD = fz_absdiff(H, D);
    __m256i rDF = fz_absdiff(D, F), rHB = fz_absdiff(H, B), rFH = fz_absdiff(F, H);

    __m256i mBDl, mBDh, mFBl, mFBh, mHDl, mHDh, mDFl, mDFh, mHBl, mHBh, mFHl, mFHh;
    fz_membership(rBD, mBDl, mBDh);
    fz_membership(rFB, mFBl, mFBh);
    fz_membership(rHD, mHDl, mHDh);
    fz_membership(rDF, mDFl, mDFh);
    fz_membership(rHB, mHBl, mHBh);
    fz_membership(rFH, mFHl, mFHh);

    __m256i c255 = _mm256_set1_epi16(255);
    // complements: 255 - m
    __m256i nFBl = _mm256_sub_epi16(c255, mFBl), nFBh = _mm256_sub_epi16(c255, mFBh);
    __m256i nHDl = _mm256_sub_epi16(c255, mHDl), nHDh = _mm256_sub_epi16(c255, mHDh);
    __m256i nBDl = _mm256_sub_epi16(c255, mBDl), nBDh = _mm256_sub_epi16(c255, mBDh);
    __m256i nDFl = _mm256_sub_epi16(c255, mDFl), nDFh = _mm256_sub_epi16(c255, mDFh);
    __m256i nFHl = _mm256_sub_epi16(c255, mFHl), nFHh = _mm256_sub_epi16(c255, mFHh);
    __m256i nHBl = _mm256_sub_epi16(c255, mHBl), nHBh = _mm256_sub_epi16(c255, mHBh);

    // w0 = m(BD)*m'(FB)*m'(HD); w1 = m(FB)*m'(DF)*m'(HB)
    // w2 = m(HD)*m'(BD)*m'(FH); w3 = m(FH)*m'(DF)*m'(HB)
    __m256i w0l, w0h, w1l, w1h, w2l, w2h, w3l, w3h;
    fz_weight(mBDl, mBDh, nFBl, nFBh, nHDl, nHDh, w0l, w0h);
    fz_weight(mFBl, mFBh, nDFl, nDFh, nHBl, nHBh, w1l, w1h);
    fz_weight(mHDl, mHDh, nBDl, nBDh, nFHl, nFHh, w2l, w2h);
    fz_weight(mFHl, mFHh, nDFl, nDFh, nHBl, nHBh, w3l, w3h);

    __m256i e0 = fz_blend(B, E, w0l, w0h);
    __m256i e1 = fz_blend(F, E, w1l, w1h);
    __m256i e2 = fz_blend(D, E, w2l, w2h);
    __m256i e3 = fz_blend(F, E, w3l, w3h);

    uint8_t b0[32], b1[32], b2[32], b3[32];
    _mm256_storeu_si256((__m256i*)b0, e0);
    _mm256_storeu_si256((__m256i*)b1, e1);
    _mm256_storeu_si256((__m256i*)b2, e2);
    _mm256_storeu_si256((__m256i*)b3, e3);

    uint8_t* o0 = out.row(2 * y) + 2 * x0;
    uint8_t* o1 = out.row(2 * y + 1) + 2 * x0;
    for (int i = 0; i < 32; ++i) {
        o0[2 * i]     = b0[i];
        o0[2 * i + 1] = b1[i];
        o1[2 * i]     = b2[i];
        o1[2 * i + 1] = b3[i];
    }
}

void scale2x_fuzzy_avx2(const Grid& in, Grid& out) {
    out = Grid(in.w * 2, in.h * 2);
    if (in.w == 0 || in.h == 0) return;

#pragma omp parallel for schedule(static)
    for (size_t y = 0; y < in.h; ++y) {
        size_t yt = y == 0 ? 0 : y - 1;
        size_t yb = y + 1 < in.h ? y + 1 : in.h - 1;
        const uint8_t* top = in.row(yt);
        const uint8_t* mid = in.row(y);
        const uint8_t* bot = in.row(yb);

        // edge x=0 scalar (clamped)
        {
            uint8_t B = px(in, 0, (long)y - 1);
            uint8_t D = px(in, -1, (long)y);
            uint8_t E = in.data[y * in.w];
            uint8_t F = px(in, 1, (long)y);
            uint8_t H = px(in, 0, (long)y + 1);
            uint8_t e0, e1, e2, e3;
            fz_pick4(B, D, E, F, H, e0, e1, e2, e3);
            out.data[(2 * y) * out.w]     = e0;
            out.data[(2 * y) * out.w + 1] = e1;
            out.data[(2 * y + 1) * out.w] = e2;
            out.data[(2 * y + 1) * out.w + 1] = e3;
        }

        size_t x = 1;
        for (; x + 31 < in.w; x += 32)
            fz_block32(out, y, x, top, mid, bot);

        // tail scalar
        for (; x < in.w; ++x) {
            uint8_t B = px(in, (long)x, (long)y - 1);
            uint8_t D = px(in, (long)x - 1, (long)y);
            uint8_t E = in.data[y * in.w + x];
            uint8_t F = px(in, (long)x + 1, (long)y);
            uint8_t H = px(in, (long)x, (long)y + 1);
            uint8_t e0, e1, e2, e3;
            fz_pick4(B, D, E, F, H, e0, e1, e2, e3);
            out.data[(2 * y) * out.w + 2 * x]     = e0;
            out.data[(2 * y) * out.w + 2 * x + 1] = e1;
            out.data[(2 * y + 1) * out.w + 2 * x] = e2;
            out.data[(2 * y + 1) * out.w + 2 * x + 1] = e3;
        }
    }
}

} // namespace avx2
#endif // __AVX2__

} // namespace rule
} // namespace fca
