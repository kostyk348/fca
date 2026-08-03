// libfca post-processing effects — implementation.
// Integer-only, deterministic. Tables precomputed once at init.

#include "fca/postfx.hpp"
#include <cstring>
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
#pragma omp parallel for schedule(static)
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
#pragma omp parallel for schedule(static)
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

// ---------------------------------------------------------------- edge-aware denoise
// 3x3 bilateral-style: weight = table[|c - n|], table decays 64,48,32,16,8,4,2,1,0.
// Normalized by the sum of weights -> the result stays inside the 3x3 min/max,
// so edges survive. Flat zones (all weights high) become a clean box average.
// Division-free: precomputed 4096/wsum lookup (wsum = 9..576).
static const int kDenW[16] = {64, 56, 48, 40, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1, 0, 0};
static int kDenInv[577];
static bool kDenInit = [] {
    for (int s = 0; s <= 576; ++s) {
        int d = s < 9 ? 9 : s; // min weight sum = 9 (all w=1)
        kDenInv[s] = (int)(((4096ll << 16) / d) + 1) >> 16;
    }
    return true;
}();

void denoise_edge(uint8_t* plane, int w, int h, int amount) {
    if (amount <= 0 || w <= 0 || h <= 0) return;
    std::vector<uint8_t> in(plane, plane + (size_t)w * h);
    const int a = amount;
    (void)kDenInit;
#pragma omp parallel for schedule(static)
    for (int y = 1; y < h - 1; ++y) {
        const uint8_t* c = &in[(size_t)y * w];
        uint8_t* o = &plane[(size_t)y * w];
        for (int x = 1; x < w - 1; ++x) {
            int cv = c[x];
            const uint8_t* n9[9] = {
                &c[x - w - 1], &c[x - w], &c[x - w + 1],
                &c[x - 1], &c[x], &c[x + 1],
                &c[x + w - 1], &c[x + w], &c[x + w + 1]};
            long long sum = 0;
            int wsum = 0;
            for (int k = 0; k < 9; ++k) {
                int d = *n9[k] - cv;
                if (d < 0) d = -d;
                int wgt = kDenW[d > 15 ? 15 : d];
                sum += (long long)(*n9[k]) * wgt;
                wsum += wgt;
            }
            // avg = sum / wsum  (division-free via lookup)
            int avg = (int)((sum * (long long)kDenInv[wsum]) >> 16);
            int mn = 255, mx = 0;
            for (int k = 0; k < 9; ++k) {
                int v = *n9[k];
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            // weight toward original on edges: edge pixels are left ~untouched
            int den_w = (a * 64) >> 8;            // 0..64
            int edge_w = (mx - mn) * 32 >> 8;     // 0..32 (>=255 range clamps)
            if (edge_w > 32) edge_w = 32;
            int w_den = den_w - edge_w;           // less denoise on hard edges
            if (w_den < 0) w_den = 0;
            int out = (cv * (64 - w_den) + avg * w_den) >> 6;
            if (out < mn) out = mn;
            if (out > mx) out = mx;
            o[x] = (uint8_t)out;
        }
    }
    // borders: leave as-is (edges of frame, 1px — negligible)
}

// ---------------------------------------------------------------- dehalo
void dehalo(uint8_t* plane, int w, int h) {
    if (w <= 0 || h <= 0) return;
    std::vector<uint8_t> in(plane, plane + (size_t)w * h);
    const int kEdge = 96;    // cross range that counts as a strong edge
    const int kKeep = 7;     // keep top 1/8 of the local range as legit peak
#pragma omp parallel for schedule(static)
    for (int y = 1; y < h - 1; ++y) {
        const uint8_t* c = &in[(size_t)y * w];
        uint8_t* o = &plane[(size_t)y * w];
        for (int x = 1; x < w - 1; ++x) {
            int b = c[x - w], d = c[x - 1], e = c[x + 1], hh = c[x + w];
            int mn = b < d ? b : d; mn = mn < e ? mn : e; mn = mn < hh ? mn : hh;
            int mx = b > d ? b : d; mx = mx > e ? mx : e; mx = mx > hh ? mx : hh;
            int range = mx - mn;
            if (range < kEdge) { o[x] = c[x]; continue; }
            int lim = mn + ((range * kKeep) >> 3); // m + 7/8*(M-m)
            int v = c[x];
            o[x] = v > lim ? (uint8_t)lim : (uint8_t)v;
        }
    }
}

// ---------------------------------------------------------------- CAS sharpen
void cas_sharpen(uint8_t* plane, int w, int h, int strength) {
    if (strength <= 0 || w <= 0 || h <= 0) return;
    std::vector<uint8_t> in(plane, plane + (size_t)w * h);
    // borders are copied unchanged (1px ring — no 3x3 data there anyway)
    if (h > 1) std::memcpy(plane, in.data(), (size_t)w);
    if (h > 1) std::memcpy(plane + (size_t)(h - 1) * w, in.data() + (size_t)(h - 1) * w, (size_t)w);
#pragma omp parallel for schedule(static)
    for (int y = 1; y < h - 1; ++y) {
        const uint8_t* c = &in[(size_t)y * w];
        uint8_t* o = &plane[(size_t)y * w];
        o[0] = c[0]; // x=0 border
        for (int x = 1; x < w - 1; ++x) {
            int A = c[x];
            int B = c[x - w];
            int D = c[x - 1];
            int E = c[x + 1];
            int H = c[x + w];
            int mn = A, mx = A;
            mn = mn < B ? mn : B; mx = mx > B ? mx : B;
            mn = mn < D ? mn : D; mx = mx > D ? mx : D;
            mn = mn < E ? mn : E; mx = mx > E ? mx : E;
            mn = mn < H ? mn : H; mx = mx > H ? mx : H;
            int cross = (B + D + E + H + 2) >> 2;
            int range = mx - mn;
            int gain = (strength * range) >> 9; // <= ~0.5 * local range (mild)
            if (gain == 0) { o[x] = (uint8_t)A; continue; }
            int diff = A - cross;
            int out = A + ((diff * gain) >> 8);
            if (out < mn) out = mn;
            if (out > mx) out = mx;
            o[x] = (uint8_t)out;
        }
        o[w - 1] = c[w - 1]; // x=w-1 border
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
    // borders: copy unchanged (no cross data on the 1px ring)
    if (h > 1) std::memcpy(plane, in.data(), (size_t)w);
    if (h > 1) std::memcpy(plane + (size_t)(h - 1) * w, in.data() + (size_t)(h - 1) * w, (size_t)w);
#pragma omp parallel for schedule(static)
    for (int y = 1; y < h - 1; ++y) {
        const uint8_t* c = &in[(size_t)y * w];
        uint8_t* o = &plane[(size_t)y * w];
        o[0] = c[0];
        for (int x = 1; x < w - 1; ++x) {
            int B = c[x - w], D = c[x - 1], E = c[x + 1], H = c[x + w];
            int mn = B, mx = B;
            int vals[4] = { B, D, E, H };
            for (int k = 0; k < 4; ++k) {
                mn = mn < vals[k] ? mn : vals[k];
                mx = mx > vals[k] ? mx : vals[k];
            }
            int range = mx - mn;
            if (range < kFlat) {
                int box = (B + D + E + H + 2) >> 2;
                int mixed = (c[x] * 3 + box + 2) >> 2;
                uint32_t hsh = (seed + (uint32_t)(x * 73856093u)) ^ (uint32_t)(y * 19349663u);
                int dith = ((int)((hsh >> 16) & 3u)) - 1; // -1..2
                o[x] = (uint8_t)clamp8(mixed + dith);
            } else o[x] = c[x];
        }
        o[w - 1] = c[w - 1];
    }
}

} // namespace postfx
} // namespace fca
