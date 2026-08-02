// libfca post-processing effects — implementation.
// Integer-only, deterministic. Tables precomputed once at init.

#include "fca/postfx.hpp"
#include <vector>

namespace fca {
namespace postfx {

// ---------------------------------------------------------------- bicubic 2x
// taps for the two 2x phases: [i-1, i, i+1, i+2], x8 fixed point
static const int16_t kTap0[4] = { 0, 256, 0, 0 };     // phase 0: copy
static const int16_t kTap1[4] = { -16, 144, 144, -16 }; // phase 0.5: Catmull-Rom

static inline int clamp8(int v) {
    if (v < 0) return 0;
    if (v > 255) return 255;
    return v;
}

void bicubic2x(const uint8_t* src, int w, int h, uint8_t* dst) {
    if (w <= 0 || h <= 0) return;
    // horizontal pass: src (w*h) -> tmp (2w*h)
    std::vector<int16_t> tmp((size_t)2 * w * h);
    for (int y = 0; y < h; ++y) {
        const uint8_t* row = src + (size_t)y * w;
        for (int ox = 0; ox < 2 * w; ++ox) {
            int i = ox >> 1;
            const int16_t* tap = (ox & 1) ? kTap1 : kTap0;
            int acc = 0;
            for (int k = 0; k < 4; ++k) {
                int ix = i + k - 1;
                if (ix < 0) ix = 0;
                if (ix >= w) ix = w - 1;
                acc += tap[k] * row[ix];
            }
            tmp[(size_t)y * 2 * w + ox] = (int16_t)clamp8(acc >> 8);
        }
    }
    // vertical pass: tmp (2w*h) -> dst (2w*2h)
    const int W2 = 2 * w;
    for (int oy = 0; oy < 2 * h; ++oy) {
        int i = oy >> 1;
        const int16_t* tap = (oy & 1) ? kTap1 : kTap0;
        for (int x = 0; x < W2; ++x) {
            int acc = 0;
            for (int k = 0; k < 4; ++k) {
                int iy = i + k - 1;
                if (iy < 0) iy = 0;
                if (iy >= h) iy = h - 1;
                acc += tap[k] * tmp[(size_t)iy * W2 + x];
            }
            dst[(size_t)oy * W2 + x] = (uint8_t)clamp8(acc >> 8);
        }
    }
}

// ---------------------------------------------------------------- CAS sharpen
void cas_sharpen(uint8_t* plane, int w, int h, int strength) {
    if (strength <= 0 || w <= 0 || h <= 0) return;
    std::vector<uint8_t> in(plane, plane + (size_t)w * h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const uint8_t* c = &in[(size_t)y * w + x];
            int A = c[0];
            int B = c[-w];
            int D = c[-1];
            int E = c[+1];
            int H = c[+w];
            int mn = A, mx = A;
            mn = mn < B ? mn : B; mx = mx > B ? mx : B;
            mn = mn < D ? mn : D; mx = mx > D ? mx : D;
            mn = mn < E ? mn : E; mx = mx > E ? mx : E;
            mn = mn < H ? mn : H; mx = mx > H ? mx : H;
            int cross = (B + D + E + H + 2) >> 2;
            int range = mx - mn;
            int gain = (strength * range) >> 9; // <= ~0.5 * local range (mild)
            if (gain == 0) continue;
            int diff = A - cross;
            int out = A + ((diff * gain) >> 8);
            if (out < mn) out = mn;
            if (out > mx) out = mx;
            plane[(size_t)y * w + x] = (uint8_t)out;
        }
    }
}

// ------------------------------------------------------------------ contrast
void contrast_s(uint8_t* plane, int w, int h, int amount) {
    if (amount <= 0 || w <= 0 || h <= 0) return;
    int K = 256 + ((amount * 90) >> 8); // up to ~1.35x
    for (size_t i = 0, n = (size_t)w * h; i < n; ++i) {
        int v = 128 + (((int)plane[i] - 128) * K >> 8);
        plane[i] = (uint8_t)clamp8(v);
    }
}

// ------------------------------------------------------------------ vibrance
void vibrance(uint8_t* cb, uint8_t* cr, int w, int h, int amount) {
    if (amount <= 0 || w <= 0 || h <= 0) return;
    int G = 256 + ((amount * 100) >> 8); // up to ~1.39x
    size_t n = (size_t)w * h;
    for (size_t i = 0; i < n; ++i) {
        int b = (int)cb[i] - 128;
        int r = (int)cr[i] - 128;
        cb[i] = (uint8_t)clamp8(128 + (b * G >> 8));
        cr[i] = (uint8_t)clamp8(128 + (r * G >> 8));
    }
}

// ------------------------------------------------------------------ deband
void deband(uint8_t* plane, int w, int h, unsigned frame) {
    if (w <= 0 || h <= 0) return;
    std::vector<uint8_t> in(plane, plane + (size_t)w * h);
    const int kFlat = 20;
    const uint32_t seed = frame * 83492791u;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const uint8_t* c = &in[(size_t)y * w + x];
            int B = c[-w], D = c[-1], E = c[+1], H = c[+w];
            int mn = B, mx = B;
            int vals[4] = { B, D, E, H };
            for (int k = 0; k < 4; ++k) {
                mn = mn < vals[k] ? mn : vals[k];
                mx = mx > vals[k] ? mx : vals[k];
            }
            int range = mx - mn;
            if (range < kFlat) {
                int box = (B + D + E + H + 2) >> 2;
                int mixed = (c[0] * 3 + box + 2) >> 2;
                uint32_t hsh = (seed + (uint32_t)(x * 73856093u)) ^ (uint32_t)(y * 19349663u);
                int dith = ((int)((hsh >> 16) & 3u)) - 1; // -1..2
                plane[(size_t)y * w + x] = (uint8_t)clamp8(mixed + dith);
            }
        }
    }
}

} // namespace postfx
} // namespace fca
