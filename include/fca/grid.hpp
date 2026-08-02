#pragma once
// libfca — fast cellular engine core.
// uint8 quantized lattice field (grid), row-major, per-plane.
// No deps, C++20.

#include <cstdint>
#include <cstddef>
#include <vector>

namespace fca {

struct Grid {
    size_t w = 0, h = 0;
    std::vector<uint8_t> data;

    Grid() = default;
    Grid(size_t w_, size_t h_) : w(w_), h(h_), data(w_ * h_) {}

    uint8_t*       row(size_t y)       { return data.data() + y * w; }
    const uint8_t* row(size_t y) const { return data.data() + y * w; }
    size_t pitch() const { return w; }
};

} // namespace fca
