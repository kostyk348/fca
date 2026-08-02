#pragma once
// libfca — S3-inspired temporal tile cache.
//
// The EAFAR thesis applied to video upscaling: an unchanged tile must not be
// recomputed. The frame is partitioned into tile_ x tile_ regions; a tile
// either SLEEPS (identical to previous frame -> previous upscale is reused,
// zero work) or WAKES (changed -> local fuzzy rule re-runs on that tile only,
// with a 1px halo read from the full input frame).
//
// Reusing the previous upscale for unchanged tiles is not just a speed win:
// it kills frame-to-frame flicker of static backgrounds (the upscaled output
// literally does not change there), which is the main visual artifact of
// single-frame edge-directed rules on compressed content.
//
// Determinism: tiles iterate in ascending (ty,tx) order; same input stream ->
// identical output. Stats are exposed for measurement.

#include "fca/grid.hpp"
#include "fca/rules.hpp"

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>

namespace fca {
namespace temporal {

struct TileStats {
    std::uint64_t total = 0;      // tiles seen
    std::uint64_t recomputed = 0; // woke (changed) -> rule re-run
    std::uint64_t reused = 0;     // slept (identical) -> cache reuse
};

class TemporalUpscaler {
public:
    TemporalUpscaler(std::uint32_t w, std::uint32_t h, std::uint32_t tile = 32)
        : w_(w), h_(h), tile_(tile)
        , prev_((std::size_t)w * h, 0)
        , cache_((std::size_t)4 * w * h, 0) {
        tx_ = (w + tile - 1) / tile;
        ty_ = (h + tile - 1) / tile;
    }

    std::uint32_t tiles_x() const noexcept { return tx_; }
    std::uint32_t tiles_y() const noexcept { return ty_; }
    const TileStats& stats() const noexcept { return stats_; }

    // in: w*h bytes (one gray plane), out: 2w*2h bytes (fuzzy rule result).
    void process(const std::uint8_t* in, std::uint8_t* out) {
        if (first_) {
            Grid g(w_, h_);
            std::memcpy(g.data.data(), in, (std::size_t)w_ * h_);
            Grid o;
            rule::scale2x_fuzzy(g, o);
            std::memcpy(cache_.data(), o.data.data(), o.data.size());
            stats_.total += tx_ * ty_;
            stats_.recomputed += tx_ * ty_;
            first_ = false;
        } else {
            for (std::uint32_t ty = 0; ty < ty_; ++ty) {
                for (std::uint32_t tx = 0; tx < tx_; ++tx) {
                    std::uint32_t x0 = tx * tile_, y0 = ty * tile_;
                    std::uint32_t x1 = std::min<std::uint32_t>(x0 + tile_, w_);
                    std::uint32_t y1 = std::min<std::uint32_t>(y0 + tile_, h_);
                    bool same = true;
                    for (std::uint32_t y = y0; y < y1 && same; ++y) {
                        const std::uint8_t* a = in + (std::size_t)y * w_ + x0;
                        const std::uint8_t* b = prev_.data() + (std::size_t)y * w_ + x0;
                        if (std::memcmp(a, b, x1 - x0) != 0) same = false;
                    }
                    stats_.total++;
                    if (same) { stats_.reused++; continue; }
                    stats_.recomputed++;
                    fz_tile_region(in, x0, y0, x1, y1);
                }
            }
            std::memcpy(prev_.data(), in, (std::size_t)w_ * h_);
        }
        std::memcpy(out, cache_.data(), cache_.size());
    }

private:
    // fuzzy-scale input tile [x0,x1)x[y0,y1) into the 2x cache_ buffer.
    // 3x3 window reads are clamped at world edges (same as rule::px).
    void fz_tile_region(const std::uint8_t* in, std::uint32_t x0, std::uint32_t y0,
                        std::uint32_t x1, std::uint32_t y1) {
        const std::size_t ow = 2 * (std::size_t)w_;
        for (std::uint32_t y = y0; y < y1; ++y) {
            for (std::uint32_t x = x0; x < x1; ++x) {
                long X = (long)x, Y = (long)y;
                std::uint8_t B = px_clamp(in, X, Y - 1);
                std::uint8_t D = px_clamp(in, X - 1, Y);
                std::uint8_t E = in[(std::size_t)Y * w_ + X];
                std::uint8_t F = px_clamp(in, X + 1, Y);
                std::uint8_t H = px_clamp(in, X, Y + 1);
                std::uint8_t e0, e1, e2, e3;
                rule::fz_pick4(B, D, E, F, H, e0, e1, e2, e3);
                std::size_t ox = 2 * (std::size_t)X, oy = 2 * (std::size_t)Y;
                cache_[oy * ow + ox]         = e0;
                cache_[oy * ow + ox + 1]     = e1;
                cache_[(oy + 1) * ow + ox]   = e2;
                cache_[(oy + 1) * ow + ox + 1] = e3;
            }
        }
    }

    std::uint8_t px_clamp(const std::uint8_t* in, long x, long y) const {
        if (x < 0) x = 0;
        if (x >= (long)w_) x = (long)w_ - 1;
        if (y < 0) y = 0;
        if (y >= (long)h_) y = (long)h_ - 1;
        return in[(std::size_t)y * w_ + (std::size_t)x];
    }

    std::uint32_t w_, h_, tile_, tx_ = 0, ty_ = 0;
    bool first_ = true;
    std::vector<std::uint8_t> prev_, cache_;
    TileStats stats_;
};

} // namespace temporal
} // namespace fca
