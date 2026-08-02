#pragma once
// libfca — S7-inspired pre-filter: median 3x3 (edge-preserving denoise).
// Removes blocky compression noise / salt before upscaling; keeps edges
// (unlike box blur). Scalar for now — SIMD median via sorting network later.

#include "fca/grid.hpp"
#include "fca/rules.hpp"

#include <algorithm>
#include <cstdint>
#include <cstddef>

namespace fca {
namespace denoise {

inline std::uint8_t median9(std::uint8_t v[9]) {
    // full sort of 9 elements (small, clear); median = middle
    std::sort(v, v + 9);
    return v[4];
}

void median3x3(const Grid& in, Grid& out) {
    out = Grid(in.w, in.h);
    for (size_t y = 0; y < in.h; ++y) {
        for (size_t x = 0; x < in.w; ++x) {
            std::uint8_t v[9];
            int k = 0;
            for (long dy = -1; dy <= 1; ++dy)
                for (long dx = -1; dx <= 1; ++dx)
                    v[k++] = rule::px(in, (long)x + dx, (long)y + dy);
            out.data[y * in.w + x] = median9(v);
        }
    }
}

} // namespace denoise
} // namespace fca
