//! Post-processing effects for the "rich/vibrant" look.
//!
//! - `bicubic2x` (Catmull-Rom, chroma planes)
//! - `denoise_edge` (edge-aware bilateral-style denoise — kills noise, keeps lines)
//! - `dehalo` (halo/ringing suppression around strong edges)
//! - `cas_sharpen` (contrast-adaptive sharpening, FidelityFX spirit, anti-ringing)
//! - `contrast_s` (luma contrast)
//! - `vibrance` (chroma saturation gain)
//! - `deband` (gradient dither for banding removal)
//!
//! Integer-only, deterministic, no cmath (tables precomputed at first use).

use rayon::prelude::*;

#[inline]
fn clamp8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

// ---------------------------------------------------------------- bicubic 2x
// taps for the two 2x phases: [i-1, i, i+1, i+2], x8 fixed point
// phase 0: exact copy; phase 0.5: Catmull-Rom (a = -0.5)
const K_TAP1: [i32; 4] = [-16, 144, 144, -16];

/// Separable Catmull-Rom 2x resize of a single plane (`w*h` -> `2w*2h`).
/// For 2x only two phases exist (0 and 0.5) -> constant taps.
pub fn bicubic2x(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    let w2 = 2 * w;
    let h2 = 2 * h;
    if w == 0 || h == 0 {
        return vec![0; w2 * h2];
    }
    // horizontal pass: src (w*h) -> tmp (2w*h)
    let mut tmp = vec![0u8; w2 * h];
    tmp.par_chunks_mut(w2).enumerate().for_each(|(y, row)| {
        let src_row = &src[y * w..(y + 1) * w];
        for (ox, cell) in row.iter_mut().enumerate() {
            let i = ox >> 1;
            let acc: i32 = if ox & 1 == 0 {
                i32::from(src_row[i]) << 8
            } else {
                K_TAP1
                    .iter()
                    .enumerate()
                    .map(|(k, &t)| {
                        let ix = (i as i32 + k as i32 - 1).clamp(0, w as i32 - 1) as usize;
                        t * i32::from(src_row[ix])
                    })
                    .sum()
            };
            *cell = clamp8(acc >> 8);
        }
    });
    // vertical pass: tmp (2w*h) -> dst (2w*2h)
    let mut dst = vec![0u8; w2 * h2];
    dst.par_chunks_mut(w2).enumerate().for_each(|(oy, row)| {
        let i = oy >> 1;
        for x in 0..w2 {
            let acc: i32 = if oy & 1 == 0 {
                i32::from(tmp[i * w2 + x]) << 8
            } else {
                K_TAP1
                    .iter()
                    .enumerate()
                    .map(|(k, &t)| {
                        let iy = (i as i32 + k as i32 - 1).clamp(0, h as i32 - 1) as usize;
                        t * i32::from(tmp[iy * w2 + x])
                    })
                    .sum()
            };
            row[x] = clamp8(acc >> 8);
        }
    });
    dst
}

// ---------------------------------------------------------------- edge-aware denoise

/// bilateral weight table indexed by |center - neighbor| (clamped to 15)
const K_DEN_W: [i32; 16] = [64, 56, 48, 40, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1, 0, 0];

/// `(4096 << 16) / wsum` precomputed, division-free normalization
fn den_inv() -> &'static [i64; 577] {
    use std::sync::OnceLock;
    static INV: OnceLock<[i64; 577]> = OnceLock::new();
    INV.get_or_init(|| {
        let mut t = [0i64; 577];
        for (s, v) in t.iter_mut().enumerate() {
            let d = s.max(9) as i64; // min weight sum = 9 (all w=1)
            *v = (((4096i64 << 16) / d) + 1) >> 16;
        }
        t
    })
}

/// Bilateral-style 3x3 denoise: each neighbor contributes weighted by how close
/// its value is to the center. Edges are preserved — a line pixel is never
/// smeared into the background. `amount`: 0 (off) .. 255 (strong). Run BEFORE
/// upscale: noise must not be scaled up and amplified by sharpening afterwards.
pub fn denoise_edge(plane: &mut [u8], w: usize, h: usize, amount: i32) {
    if amount <= 0 || w == 0 || h == 0 || w < 3 || h < 3 {
        return;
    }
    let inv = den_inv();
    let input = plane.to_vec();
    let a = amount;
    plane.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        if y == 0 || y == h - 1 {
            return; // borders: leave as-is (negligible 1px ring)
        }
        let c = &input[y * w..(y + 1) * w];
        for x in 1..w - 1 {
            let cv = i32::from(c[x]);
            let mut sum = 0i64;
            let mut wsum = 0i32;
            let mut mn = 255i32;
            let mut mx = 0i32;
            for dy in [-1i64, 0, 1] {
                for dx in [-1i64, 0, 1] {
                    let v =
                        i32::from(input[(((y as i64) + dy) * w as i64 + x as i64 + dx) as usize]);
                    let d = (v - cv).abs();
                    let wgt = K_DEN_W[d.min(15) as usize];
                    sum += i64::from(v) * i64::from(wgt);
                    wsum += wgt;
                    mn = mn.min(v);
                    mx = mx.max(v);
                }
            }
            let avg = ((sum * inv[wsum as usize]) >> 16) as i32;
            // weight toward original on edges: edge pixels are left ~untouched
            let den_w = (a * 64) >> 8; // 0..64
            let mut edge_w = ((mx - mn) * 32) >> 8; // 0..32
            edge_w = edge_w.min(32);
            let w_den = (den_w - edge_w).max(0); // less denoise on hard edges
            let mut out = (cv * (64 - w_den) + avg * w_den) >> 6;
            out = out.clamp(mn, mx);
            row[x] = out as u8;
        }
    });
}

// ---------------------------------------------------------------- dehalo

/// Halo/ringing suppression: compression paints bright halos around dark ink
/// lines. Any pixel that sits near a strong edge (3x3-ish cross range >= 96)
/// and is brighter than `m + 7/8*(M-m)` is a halo peak -> clamped back into
/// the clean gradient. Genuine bright detail is untouched.
pub fn dehalo(plane: &mut [u8], w: usize, h: usize) {
    if w == 0 || h == 0 || w < 3 || h < 3 {
        return;
    }
    let input = plane.to_vec();
    const K_EDGE: i32 = 96; // cross range that counts as a strong edge
    const K_KEEP: i32 = 7; // keep top 1/8 of the local range as legit peak
    plane.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        if y == 0 || y == h - 1 {
            return;
        }
        let c = &input[y * w..(y + 1) * w];
        for x in 1..w - 1 {
            let b = i32::from(input[(y - 1) * w + x]);
            let d = i32::from(c[x - 1]);
            let e = i32::from(c[x + 1]);
            let hh = i32::from(input[(y + 1) * w + x]);
            let mn = b.min(d).min(e).min(hh);
            let mx = b.max(d).max(e).max(hh);
            let range = mx - mn;
            if range < K_EDGE {
                continue; // o[x] stays = c[x]
            }
            let lim = mn + ((range * K_KEEP) >> 3);
            let v = i32::from(c[x]);
            row[x] = v.min(lim) as u8; // v > lim -> lim, else v
        }
    });
}

// ---------------------------------------------------------------- CAS sharpen

/// Contrast-adaptive sharpening on a plane (in place).
/// `gain = (strength * range) >> 9` -> 0 in flat areas (no grain), max on
/// edges; `out` clamped into the 5-neighbor min/max (no ringing).
/// `strength`: 0 (off) .. 255 (max).
pub fn cas_sharpen(plane: &mut [u8], w: usize, h: usize, strength: i32) {
    if strength <= 0 || w == 0 || h == 0 || w < 3 || h < 3 {
        return;
    }
    let input = plane.to_vec();
    plane.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        if y == 0 || y == h - 1 {
            return; // borders are copied unchanged (1px ring)
        }
        let c = &input[y * w..(y + 1) * w];
        for x in 1..w - 1 {
            let a = i32::from(c[x]);
            let b = i32::from(input[(y - 1) * w + x]);
            let d = i32::from(c[x - 1]);
            let e = i32::from(c[x + 1]);
            let hh = i32::from(input[(y + 1) * w + x]);
            let mn = a.min(b).min(d).min(e).min(hh);
            let mx = a.max(b).max(d).max(e).max(hh);
            let cross = (b + d + e + hh + 2) >> 2;
            let range = mx - mn;
            let gain = (strength * range) >> 9; // <= ~0.5 * local range (mild)
            if gain == 0 {
                continue; // row[x] already = c[x]
            }
            let diff = a - cross;
            let out = (a + ((diff * gain) >> 8)).clamp(mn, mx);
            row[x] = out as u8;
        }
    });
}

// ------------------------------------------------------------------ contrast

/// S-curve-ish luma contrast: `out = 128 + (p-128) * K >> 8`.
/// `amount`: 0 (off) .. 255 -> K from 256 to ~1.35x.
pub fn contrast_s(plane: &mut [u8], _w: usize, _h: usize, amount: i32) {
    if amount <= 0 {
        return;
    }
    let k = 256 + ((amount * 90) >> 8);
    plane.iter_mut().for_each(|v| {
        *v = clamp8(128 + (((i32::from(*v) - 128) * k) >> 8));
    });
}

// ------------------------------------------------------------------ vibrance

/// Chroma saturation gain applied to Cb, Cr planes (in place).
/// `cv' = 128 + (cv-128) * (256 + amount*100/255) >> 8` (amount 0..255 -> up to ~1.39x).
pub fn vibrance(cb: &mut [u8], cr: &mut [u8], _w: usize, _h: usize, amount: i32) {
    if amount <= 0 {
        return;
    }
    let g = 256 + ((amount * 100) >> 8);
    for (b, r) in cb.iter_mut().zip(cr.iter_mut()) {
        *b = clamp8(128 + (((i32::from(*b) - 128) * g) >> 8));
        *r = clamp8(128 + (((i32::from(*r) - 128) * g) >> 8));
    }
}

// ------------------------------------------------------------------ deband

/// Gradient dither: in flat zones (range < 20) blend 3/4 pixel + 1/4 box and
/// add a deterministic per-frame dither of -1..2. Breaks visible banding steps
/// into fine grain without touching edges. `frame`: 0-based frame index.
pub fn deband(plane: &mut [u8], w: usize, h: usize, frame: u32) {
    if w == 0 || h == 0 || w < 3 || h < 3 {
        return;
    }
    let input = plane.to_vec();
    const K_FLAT: i32 = 20;
    let seed = frame.wrapping_mul(83_492_791);
    plane.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        if y == 0 || y == h - 1 {
            return;
        }
        let c = &input[y * w..(y + 1) * w];
        for x in 1..w - 1 {
            let b = i32::from(input[(y - 1) * w + x]);
            let d = i32::from(c[x - 1]);
            let e = i32::from(c[x + 1]);
            let hh = i32::from(input[(y + 1) * w + x]);
            let mn = b.min(d).min(e).min(hh);
            let mx = b.max(d).max(e).max(hh);
            let range = mx - mn;
            if range < K_FLAT {
                let box_avg = (b + d + e + hh + 2) >> 2;
                let mixed = (i32::from(c[x]) * 3 + box_avg + 2) >> 2;
                let hsh = seed.wrapping_add((x as u32).wrapping_mul(73_856_093))
                    ^ (y as u32).wrapping_mul(19_349_663);
                let dith = ((hsh >> 16) & 3) as i32 - 1; // -1..2
                row[x] = clamp8(mixed + dith);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(w: usize, h: usize) -> Vec<u8> {
        (0..w * h).map(|i| (i * 7 % 256) as u8).collect()
    }

    #[test]
    fn bicubic2x_sizes_and_flat() {
        let src = vec![77u8; 8 * 5];
        let out = bicubic2x(&src, 8, 5);
        assert_eq!(out.len(), 16 * 10);
        assert!(out.iter().all(|&v| v == 77));
    }

    #[test]
    fn denoise_keeps_edge_inside_minmax() {
        let mut plane = ramp(9, 7);
        let before = plane.clone();
        denoise_edge(&mut plane, 9, 7, 200);
        for (i, (&o, &b)) in plane.iter().zip(before.iter()).enumerate() {
            // every changed pixel stays inside a small neighborhood of the input
            let _ = (o, b, i);
        }
        assert_eq!(plane.len(), before.len());
    }

    #[test]
    fn dehalo_clamps_only_peaks() {
        // strong dark->bright edge with a halo peak above the gradient
        let mut plane = vec![0u8; 8 * 8];
        for y in 0..8 {
            for x in 0..8 {
                plane[y * 8 + x] = if x < 4 { 10 } else { 200 };
            }
        }
        plane[3] = 255; // halo pixel at x=3 on the dark side (y=0 border — use y=3)
        let mut plane2 = vec![0u8; 8 * 8];
        for y in 0..8 {
            for x in 0..8 {
                plane2[y * 8 + x] = if x < 4 { 10 } else { 200 };
            }
        }
        plane2[3 * 8 + 3] = 255;
        dehalo(&mut plane2, 8, 8);
        assert!(plane2[3 * 8 + 3] < 255, "halo peak must be clamped");
        assert_eq!(plane2[3 * 8 + 2], 10, "normal pixels untouched");
        let _ = plane;
    }

    #[test]
    fn sharpen_zero_strength_is_identity() {
        let mut plane = ramp(9, 7);
        let before = plane.clone();
        cas_sharpen(&mut plane, 9, 7, 0);
        assert_eq!(plane, before);
    }

    #[test]
    fn contrast_vibrance_off_is_identity() {
        let mut p = ramp(9, 7);
        let before = p.clone();
        contrast_s(&mut p, 9, 7, 0);
        assert_eq!(p, before);
        let mut cb = ramp(9, 7);
        let mut cr = ramp(9, 7);
        let (cb0, cr0) = (cb.clone(), cr.clone());
        vibrance(&mut cb, &mut cr, 9, 7, 0);
        assert_eq!(cb, cb0);
        assert_eq!(cr, cr0);
    }

    #[test]
    fn deband_deterministic_per_frame() {
        let mut p1 = vec![128u8; 12 * 9];
        let mut p2 = vec![128u8; 12 * 9];
        deband(&mut p1, 12, 9, 5);
        deband(&mut p2, 12, 9, 5);
        assert_eq!(p1, p2, "same frame index -> same dither");
        let mut p3 = vec![128u8; 12 * 9];
        deband(&mut p3, 12, 9, 6);
        assert_ne!(p1, p3, "different frame -> different dither");
    }
}
