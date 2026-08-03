#pragma once
// libfca — VSR (video super-resolution) via multi-frame subpixel fusion.
//   estimate_shift : global subpixel motion between two frames (pan/static)
//   fusion2x       : 2x shift-and-add fusion of current + history frames
// Deterministic, integer-only (8.8 fixed point), no external deps.
//
// Idea: on static/panning scenes, history frames observe the same content at
// subpixel offsets. Fusing them (with local consistency gating) recovers real
// detail above the source and kills encoder noise — without hallucinating.

#include <cstdint>
#include <cstddef>
#include <vector>

namespace fca {
namespace vsr {

// subpixel shift in 8.8 fixed point (gx,gy = 256 * real shift)
struct Shift { int gx; int gy; };

// estimate global shift: `a(x,y) ~ b(x - gx/256, y - gy/256)`.
// Coarse search on 4x-downsampled, refine on full res, subpixel via parabola.
Shift estimate_shift(const uint8_t* a, const uint8_t* b, int w, int h);

// 2x fusion. `cur` = current frame (w*h), `hist` = older frames (w*h each).
// Every output pixel gathers votes from cur + hist frames sampled at
// subpixel offsets (history k assumes accumulated shift (k+1)*g — constant
// velocity approximation, good for pans), gated by local consistency.
// out = 2w*2h.
void fusion2x(const uint8_t* cur, const std::vector<const uint8_t*>& hist,
              int w, int h, Shift g, uint8_t* out);

} // namespace vsr
} // namespace fca
