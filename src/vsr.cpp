// libfca VSR — implementation. Integer-only, 8.8 fixed point, deterministic.

#include "fca/vsr.hpp"
#include <algorithm>
#include <array>

namespace fca {
namespace vsr {

static inline int clamp_i(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// ---- downscale by 4 (2x2 average) into a w4*h4 buffer ----
static void down4(const uint8_t* src, int w, int w4, int h4, std::vector<uint8_t>& dst) {
    dst.resize((size_t)w4 * h4);
    for (int y = 0; y < h4; ++y) {
        for (int x = 0; x < w4; ++x) {
            int s = 0;
            for (int dy = 0; dy < 4; ++dy)
                for (int dx = 0; dx < 4; ++dx)
                    s += src[(size_t)(4 * y + dy) * w + 4 * x + dx];
            dst[(size_t)y * w4 + x] = (uint8_t)(s >> 4);
        }
    }
}

// ---- sum of squared differences with integer shift (dx,dy) ----
static long long sse(const uint8_t* a, const uint8_t* b, int w, int h, int dx, int dy) {
    long long s = 0;
    int x0 = dx < 0 ? -dx : 0;
    int x1 = w - (dx > 0 ? dx : 0);
    int y0 = dy < 0 ? -dy : 0;
    int y1 = h - (dy > 0 ? dy : 0);
    for (int y = y0; y < y1; ++y) {
        const uint8_t* ra = a + (size_t)y * w;
        const uint8_t* rb = b + (size_t)(y + dy) * w;
        for (int x = x0; x < x1; ++x) {
            int d = (int)ra[x] - (int)rb[x + dx];
            s += (long long)d * d;
        }
    }
    return s;
}
Shift estimate_shift(const uint8_t* a, const uint8_t* b, int w, int h) {
    Shift g{0, 0};
    if (w < 8 || h < 8) return g;
    int w4 = w / 4, h4 = h / 4;
    if (w4 < 2 || h4 < 2) return g;

    std::vector<uint8_t> da, db;
    down4(a, w, w4, h4, da);
    down4(b, w, w4, h4, db);

    // coarse search on the 4x-downsampled grid: +/-3 (== +/-12 px full res).
    // Tighter than +/-5 to avoid aliasing onto periodic texture (period-20 px
    // patterns would match a +5 shift on the downsampled grid).
    long long best = -1;
    int bdx = 0, bdy = 0;
    for (int dy = -3; dy <= 3; ++dy) {
        for (int dx = -3; dx <= 3; ++dx) {
            long long s = sse(da.data(), db.data(), w4, h4, dx, dy);
            if (best < 0 || s < best) { best = s; bdx = dx; bdy = dy; }
        }
    }

    // refine on full resolution around 4*best, +/-2 px
    long long fbest = -1;
    int fdx = 4 * bdx, fdy = 4 * bdy;
    for (int dy = 4 * bdy - 2; dy <= 4 * bdy + 2; ++dy) {
        for (int dx = 4 * bdx - 2; dx <= 4 * bdx + 2; ++dx) {
            long long s = sse(a, b, w, h, dx, dy);
            if (fbest < 0 || s < fbest) { fbest = s; fdx = dx; fdy = dy; }
        }
    }

    // ---- subpixel refinement: direct search on a fractional grid.
    // Integer SSE cannot distinguish complementary (even/odd) samplings, so we
    // evaluate the cost with bilinear interpolation of b at fractional dx/dy.
    // Every 2nd pixel is used (subsampling, NOT averaging) to keep the exact
    // phase parity while cutting the cost 4x.
    auto sse_sub = [&](int dx8, int dy8) -> long long {
        long long s = 0;
        int fx = dx8 & 0xFF, fy = dy8 & 0xFF;
        int dx = dx8 >> 8, dy = dy8 >> 8;
        for (int yy = 2; yy < h - 2; yy += 2) {
            int byy = yy + dy;
            if (byy < 0) byy = 0; if (byy > h - 2) byy = h - 2;
            const uint8_t* ra = a + (size_t)yy * w;
            const uint8_t* rb = b + (size_t)byy * w;
            for (int xx = 2; xx < w - 2; xx += 2) {
                int bxx = xx + dx;
                if (bxx < 0) bxx = 0; if (bxx > w - 2) bxx = w - 2;
                int p00 = rb[bxx], p10 = rb[bxx + 1];
                int p01 = rb[bxx + w], p11 = rb[bxx + w + 1];
                int bv = (p00 * (256 - fx) * (256 - fy) + p10 * fx * (256 - fy) +
                          p01 * (256 - fx) * fy + p11 * fx * fy + 32768) >> 16;
                int d = (int)ra[xx] - bv;
                s += (long long)d * d;
            }
        }
        return s;
    };
    // grid: integer base + {-3,-2,-1,0,1,2,3} * 64 (i.e. +/-0.75px in 0.25 steps)
    long long bbest = -1; int bdx8 = 0, bdy8 = 0;
    for (int sy = -3; sy <= 3; ++sy) {
        for (int sx = -3; sx <= 3; ++sx) {
            long long s = sse_sub(fdx * 256 + sx * 64, fdy * 256 + sy * 64);
            if (bbest < 0 || s < bbest) { bbest = s; bdx8 = sx * 64; bdy8 = sy * 64; }
        }
    }
    // second pass at 0.125px resolution around the winner
    long long abest = bbest; int adx8 = bdx8, ady8 = bdy8;
    for (int sy = -1; sy <= 1; ++sy) {
        for (int sx = -1; sx <= 1; ++sx) {
            long long s = sse_sub(fdx * 256 + bdx8 + sx * 32, fdy * 256 + bdy8 + sy * 32);
            if (s < abest) { abest = s; adx8 = bdx8 + sx * 32; ady8 = bdy8 + sy * 32; }
        }
    }
    g.gx = fdx * 256 + adx8;
    g.gy = fdy * 256 + ady8;
    return g;
}

// ---- Catmull-Rom (bicubic) sample, 8.8 fixed point, edge-clamped ----
static inline uint8_t bicubic8(const uint8_t* p, int w, int h, int x8, int y8) {
    int x = x8 >> 8, y = y8 >> 8;
    int fx = x8 & 0xFF, fy = y8 & 0xFF;
    if (x < 1) { x = 1; fx = 0; }
    if (x > w - 2) { x = w - 2; fx = 0; }
    if (y < 1) { y = 1; fy = 0; }
    if (y > h - 2) { y = h - 2; fy = 0; }
    auto cr = [](int t) -> std::array<int, 4> { // Catmull-Rom weights for offset t (0..255), sum=256
        int t2 = (t * t) >> 8, t3 = (t2 * t) >> 8;
        return std::array<int, 4>{
            (-t3 + 2 * t2 - t) >> 1,
            (3 * t3 - 5 * t2 + 2 * 256) >> 1,
            (-3 * t3 + 4 * t2 + t) >> 1,
            (t3 - t2) >> 1};
    }(fx);
    auto cr_y = [](int t) -> std::array<int, 4> {
        int t2 = (t * t) >> 8, t3 = (t2 * t) >> 8;
        return std::array<int, 4>{
            (-t3 + 2 * t2 - t) >> 1,
            (3 * t3 - 5 * t2 + 2 * 256) >> 1,
            (-3 * t3 + 4 * t2 + t) >> 1,
            (t3 - t2) >> 1};
    }(fy);
    long long acc = 0;
    for (int j = 0; j < 4; ++j) {
        const uint8_t* r = p + (size_t)(y - 1 + j) * w;
        int hsum = 0;
        for (int i = 0; i < 4; ++i) hsum += (int)r[x - 1 + i] * cr[i];
        acc += (long long)hsum * cr_y[j];
    }
    int val = (int)(acc / 65536);
    return (uint8_t)(val < 0 ? 0 : (val > 255 ? 255 : val));
}

// ---- bilinear sample, 8.8 fixed point, edge-clamped ----
static inline uint8_t bilinear8(const uint8_t* p, int w, int h, int x8, int y8) {
    int x = x8 >> 8, y = y8 >> 8;
    int fx = x8 & 0xFF, fy = y8 & 0xFF;
    if (x < 0) { x = 0; fx = 0; }
    if (x > w - 2) { x = w - 2; fx = 0; }
    if (y < 0) { y = 0; fy = 0; }
    if (y > h - 2) { y = h - 2; fy = 0; }
    int p00 = p[(size_t)y * w + x];
    int p10 = p[(size_t)y * w + x + 1];
    int p01 = p[(size_t)(y + 1) * w + x];
    int p11 = p[(size_t)(y + 1) * w + x + 1];
    int w00 = (256 - fx) * (256 - fy);
    int w10 = fx * (256 - fy);
    int w01 = (256 - fx) * fy;
    int w11 = fx * fy;
    return (uint8_t)((p00 * w00 + p10 * w10 + p01 * w01 + p11 * w11 + 32768) >> 16);
}

void fusion2x(const uint8_t* cur, const std::vector<const uint8_t*>& hist,
              int w, int h, Shift g, uint8_t* out) {
    const int W2 = 2 * w;
    const int kConsist = 48; // luma gate
    const int kMaxShift = 640; // 2.5 px: beyond this, motion model is unsafe
    // If the estimated shift is implausibly large (scene cut / fast motion),
    // fall back to no shift: the fusion becomes a temporal median denoise of
    // the same location (safe, never worse than a single frame).
    if (g.gx > kMaxShift || g.gx < -kMaxShift ||
        g.gy > kMaxShift || g.gy < -kMaxShift) {
        g.gx = 0; g.gy = 0;
    }
    // accumulated shifts per history frame (constant velocity model):
    // hist[0] oldest -> largest accumulated shift
    // output 2x pixel (X,Y)=(2x+ox, 2y+oy) lies at 1x position (x+ox/2, y+oy/2):
    // phases 0.0 for even, 0.5 for odd (in 8.8: 0 and 128)
    const int phase[2] = {0, 128};
#pragma omp parallel for schedule(static)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int oy = 0; oy < 2; ++oy) {
                for (int ox = 0; ox < 2; ++ox) {
                    int cx8 = x * 256 + phase[ox];
                    int cy8 = y * 256 + phase[oy];
                    int votes[16];
                    int nv = 0;
                    int v0 = bicubic8(cur, w, h, cx8, cy8);
                    votes[nv++] = v0;
                    for (size_t k = 0; k < hist.size() && nv < 16; ++k) {
                        int d = (int)(hist.size() - k); // 1..N accumulated steps
                        // estimate_shift(a=hist[last], b=cur) gives dx with
                        // hist[last][x] ~= cur[x + dx], so the scene at cur[p]
                        // sits in hist[k] (d frames back) at p - d*dx.
                        int sx8 = cx8 - d * g.gx;
                        int sy8 = cy8 - d * g.gy;
                        int v = bicubic8(hist[k], w, h, sx8, sy8);
                        if (v < v0 - kConsist || v > v0 + kConsist) continue; // moving object
                        votes[nv++] = v;
                    }
                    // median of 1..5 votes
                    std::sort(votes, votes + nv);
                    out[(size_t)(2 * y + oy) * W2 + 2 * x + ox] = (uint8_t)votes[nv / 2];
                }
            }
        }
    }
}

} // namespace vsr
} // namespace fca
