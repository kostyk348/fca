#pragma once
// libfca — post-processing effects for the "rich/vibrant" look.
//   bicubic 2x  (Catmull-Rom, chroma planes)
//   cas_sharpen (contrast-adaptive sharpening, FidelityFX spirit, anti-ringing)
//   contrast_s  (luma contrast)
//   vibrance    (chroma saturation gain)
//   deband      (gradient dither for banding removal)
// Integer-only, deterministic, no cmath (tables precomputed at init).

#include <cstdint>
#include <cstddef>

namespace fca {
namespace postfx {

// ------------------------------------------------------------------ bicubic 2x
// Separable Catmull-Rom (a = -0.5) resize  w*h -> 2w*2h  (single plane).
// For 2x only two phases exist (0 and 0.5) -> constant taps:
//   phase 0: [0, 256, 0, 0]     (exact copy)
//   phase 1: [-16, 144, 144, -16]  (9/16, 9/16, -1/16, -1/16)
void bicubic2x(const uint8_t* src, int w, int h, uint8_t* dst);

// ------------------------------------------------------------------ CAS sharpen
// Contrast-adaptive sharpening on a plane (in place).
//   cross   = (B+C+D+E+2)>>2
//   range   = max3x3 - min3x3            (local contrast)
//   gain    = (strength * range) >> 8    -> 0 in flat areas (no grain), max on edges
//   out     = clamp(p + ((p - cross) * gain) >> 8, min3x3, max3x3)   (no ringing)
// strength: 0 (off) .. 255 (max).
void cas_sharpen(uint8_t* plane, int w, int h, int strength);

// ------------------------------------------------------------------ contrast
// S-curve-ish luma contrast: out = 128 + (p-128) * K >> 8.
//   amount: 0 (off) .. 255 -> K from 256 to ~1.35x.
void contrast_s(uint8_t* plane, int w, int h, int amount);

// ------------------------------------------------------------------ vibrance
// Chroma saturation gain applied to Cb,Cr planes (in place).
//   cv' = cv * (256 + amount*100/255) >> 8   (amount 0..255 -> up to ~1.39x)
void vibrance(uint8_t* cb, uint8_t* cr, int w, int h, int amount);

// ------------------------------------------------------------------ deband
// Gradient dither: in flat zones (range < kFlat) blend 3/4 pixel + 1/4 box and
// add a deterministic per-frame dither of +/-2. Breaks visible banding steps
// into fine grain without touching edges. frame = 0-based frame index.
void deband(uint8_t* plane, int w, int h, unsigned frame);

} // namespace postfx
} // namespace fca
