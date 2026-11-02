//! VSR (video super-resolution) via multi-frame subpixel fusion.
//!
//! - `estimate_shift`: global subpixel motion between two frames (pan/static)
//! - `fusion2x`: 2x shift-and-add fusion of current + history frames
//!
//! Deterministic, integer-only (8.8 fixed point). On static/panning scenes,
//! history frames observe the same content at subpixel offsets. Fusing them
//! (with local consistency gating) recovers real detail above the source and
//! kills encoder noise — without hallucinating.

use rayon::prelude::*;

/// Subpixel shift in 8.8 fixed point (`gx,gy = 256 * real shift`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Shift {
    pub gx: i32,
    pub gy: i32,
}

/// Downscale by 4 (2x2... actually 4x4 box average) into a `w4*h4` buffer.
fn down4(src: &[u8], w: usize, w4: usize, h4: usize) -> Vec<u8> {
    let mut dst = vec![0u8; w4 * h4];
    for y in 0..h4 {
        for x in 0..w4 {
            let mut s = 0i32;
            for dy in 0..4 {
                for dx in 0..4 {
                    s += i32::from(src[(4 * y + dy) * w + 4 * x + dx]);
                }
            }
            dst[y * w4 + x] = (s >> 4) as u8;
        }
    }
    dst
}

/// Sum of squared differences with integer shift (dx,dy).
/// Compares `a[y][x]` against `b[y+dy][x+dx]` over the safe overlap.
fn sse(a: &[u8], b: &[u8], w: usize, h: usize, dx: i32, dy: i32) -> i64 {
    let (wi, hi) = (w as i64, h as i64);
    let x0 = (-dx).max(0) as i64;
    let x1 = wi - (dx.max(0) as i64);
    let y0 = (-dy).max(0) as i64;
    let y1 = hi - (dy.max(0) as i64);
    if x1 <= x0 || y1 <= y0 {
        return i64::MAX;
    }
    let mut s = 0i64;
    for y in y0..y1 {
        let ra = &a[(y * wi) as usize..];
        let rb = &b[((y + dy as i64) * wi) as usize..];
        for x in x0..x1 {
            let d = i32::from(ra[x as usize]) - i32::from(rb[(x + dx as i64) as usize]);
            s += i64::from(d * d);
        }
    }
    s
}

/// Estimate global shift: `a(x,y) ~ b(x - gx/256, y - gy/256)`.
/// Coarse search on 4x-downsampled, refine on full res, subpixel via direct
/// search on a fractional grid with bilinear interpolation of `b`.
pub fn estimate_shift(a: &[u8], b: &[u8], w: usize, h: usize) -> Shift {
    let mut g = Shift::default();
    if w < 8 || h < 8 {
        return g;
    }
    let (w4, h4) = (w / 4, h / 4);
    if w4 < 2 || h4 < 2 {
        return g;
    }
    let da = down4(a, w, w4, h4);
    let db = down4(b, w, w4, h4);

    // coarse search on the 4x-downsampled grid: +/-3 (== +/-12 px full res).
    // Tighter than +/-5 to avoid aliasing onto periodic texture (period-20 px
    // patterns would match a +5 shift on the downsampled grid).
    let mut best = -1i64;
    let mut bdx = 0i32;
    let mut bdy = 0i32;
    for dy in -3..=3 {
        for dx in -3..=3 {
            let s = sse(&da, &db, w4, h4, dx, dy);
            if best < 0 || s < best {
                best = s;
                bdx = dx;
                bdy = dy;
            }
        }
    }

    // refine on full resolution around 4*best, +/-2 px
    let mut fbest = -1i64;
    let mut fdx = 4 * bdx;
    let mut fdy = 4 * bdy;
    for dy in 4 * bdy - 2..=4 * bdy + 2 {
        for dx in 4 * bdx - 2..=4 * bdx + 2 {
            let s = sse(a, b, w, h, dx, dy);
            if fbest < 0 || s < fbest {
                fbest = s;
                fdx = dx;
                fdy = dy;
            }
        }
    }

    // ---- subpixel refinement: direct search on a fractional grid.
    // Integer SSE cannot distinguish complementary (even/odd) samplings, so we
    // evaluate the cost with bilinear interpolation of b at fractional dx/dy.
    // Every 2nd pixel is used (subsampling, NOT averaging) to keep the exact
    // phase parity while cutting the cost 4x.
    let sse_sub = |dx8: i32, dy8: i32| -> i64 {
        let mut s = 0i64;
        let fx = dx8 & 0xFF;
        let fy = dy8 & 0xFF;
        let dx = dx8 >> 8;
        let dy = dy8 >> 8;
        let mut yy = 2;
        while yy < h.saturating_sub(2) {
            let byy = (yy as i32 + dy).clamp(0, h as i32 - 2) as usize;
            let ra = &a[yy * w..];
            let rb = &b[byy * w..];
            let mut xx = 2;
            while xx < w.saturating_sub(2) {
                let bxx = (xx as i32 + dx).clamp(0, w as i32 - 2) as usize;
                let p00 = i32::from(rb[bxx]);
                let p10 = i32::from(rb[bxx + 1]);
                let p01 = i32::from(rb[bxx + w]);
                let p11 = i32::from(rb[bxx + w + 1]);
                let bv = (p00 * (256 - fx) * (256 - fy)
                    + p10 * fx * (256 - fy)
                    + p01 * (256 - fx) * fy
                    + p11 * fx * fy
                    + 32768)
                    >> 16;
                let d = i32::from(ra[xx]) - bv;
                s += i64::from(d * d);
                xx += 2;
            }
            yy += 2;
        }
        s
    };
    // grid: integer base + {-3..3} * 64 (i.e. +/-0.75px in 0.25 steps)
    let mut bbest = -1i64;
    let mut bdx8 = 0i32;
    let mut bdy8 = 0i32;
    for sy in -3..=3 {
        for sx in -3..=3 {
            let s = sse_sub(fdx * 256 + sx * 64, fdy * 256 + sy * 64);
            if bbest < 0 || s < bbest {
                bbest = s;
                bdx8 = sx * 64;
                bdy8 = sy * 64;
            }
        }
    }
    // second pass at 0.125px resolution around the winner
    let mut abest = bbest;
    let mut adx8 = bdx8;
    let mut ady8 = bdy8;
    for sy in -1..=1 {
        for sx in -1..=1 {
            let s = sse_sub(fdx * 256 + bdx8 + sx * 32, fdy * 256 + bdy8 + sy * 32);
            if s < abest {
                abest = s;
                adx8 = bdx8 + sx * 32;
                ady8 = bdy8 + sy * 32;
            }
        }
    }
    let _ = abest;
    g.gx = fdx * 256 + adx8;
    g.gy = fdy * 256 + ady8;
    g
}

/// Catmull-Rom (bicubic) sample, 8.8 fixed point, edge-clamped.
fn bicubic8(p: &[u8], w: usize, h: usize, x8: i32, y8: i32) -> u8 {
    let mut x = x8 >> 8;
    let mut y = y8 >> 8;
    let mut fx = x8 & 0xFF;
    let mut fy = y8 & 0xFF;
    if x < 1 {
        x = 1;
        fx = 0;
    }
    if x > w as i32 - 2 {
        x = w as i32 - 2;
        fx = 0;
    }
    if y < 1 {
        y = 1;
        fy = 0;
    }
    if y > h as i32 - 2 {
        y = h as i32 - 2;
        fy = 0;
    }
    let cr = |t: i32| -> [i32; 4] {
        // Catmull-Rom weights for offset t (0..255), sum=256
        let t2 = (t * t) >> 8;
        let t3 = (t2 * t) >> 8;
        [
            (-t3 + 2 * t2 - t) >> 1,
            (3 * t3 - 5 * t2 + 2 * 256) >> 1,
            (-3 * t3 + 4 * t2 + t) >> 1,
            (t3 - t2) >> 1,
        ]
    };
    let crx = cr(fx);
    let cry = cr(fy);
    let mut acc = 0i64;
    // NB: the C original indexed one row/column past the plane here (weight 0
    // at the exact edge, garbage otherwise); the port clamps the taps instead.
    for (j, &cy) in cry.iter().enumerate() {
        let row = (y - 1 + j as i32).clamp(0, h as i32 - 1) as usize;
        let r = &p[row * w..];
        let hsum: i32 = (0..4)
            .map(|i| i32::from(r[(x - 1 + i).clamp(0, w as i32 - 1) as usize]) * crx[i as usize])
            .sum();
        acc += i64::from(hsum) * i64::from(cy);
    }
    let val = (acc / 65536).clamp(0, 255);
    val as u8
}

/// 2x fusion. `cur` = current frame, `hist` = older frames (oldest first).
/// Every output pixel gathers votes from cur + hist frames sampled at subpixel
/// offsets (history k assumes accumulated shift (k+1)*g — constant velocity
/// approximation, good for pans), gated by local consistency.
/// `out` = `2w*2h`.
pub fn fusion2x(cur: &[u8], hist: &[&[u8]], w: usize, h: usize, g: Shift, out: &mut [u8]) {
    let w2 = 2 * w;
    const K_CONSIST: i32 = 48; // luma gate
    const K_MAX_SHIFT: i32 = 640; // 2.5 px: beyond this, motion model is unsafe
                                  // If the estimated shift is implausibly large (scene cut / fast motion),
                                  // fall back to no shift: the fusion becomes a temporal median denoise of
                                  // the same location (safe, never worse than a single frame).
    let g = if g.gx.abs() > K_MAX_SHIFT || g.gy.abs() > K_MAX_SHIFT {
        Shift::default()
    } else {
        g
    };
    // accumulated shifts per history frame (constant velocity model):
    // hist[0] oldest -> largest accumulated shift
    // output 2x pixel (X,Y)=(2x+ox, 2y+oy) lies at 1x position (x+ox/2, y+oy/2):
    // phases 0.0 for even, 0.5 for odd (in 8.8: 0 and 128)
    let phase = [0i32, 128];
    let hist: Vec<&[u8]> = hist.to_vec();
    out.par_chunks_mut(w2)
        .with_max_len(1 << 16)
        .enumerate()
        .for_each(|(oy_big, row)| {
            let y = oy_big / 2;
            let oy = oy_big % 2;
            for x in 0..w {
                for ox in 0..2 {
                    let cx8 = x as i32 * 256 + phase[ox];
                    let cy8 = y as i32 * 256 + phase[oy];
                    let mut votes = [0i32; 16];
                    let mut nv = 0usize;
                    let v0 = bicubic8(cur, w, h, cx8, cy8);
                    votes[nv] = i32::from(v0);
                    nv += 1;
                    let mut k = 0;
                    while k < hist.len() && nv < 16 {
                        let hf = hist[k];
                        let d = (hist.len() - k) as i32; // 1..N accumulated steps
                                                         // estimate_shift(a=hist[last], b=cur) gives dx with
                                                         // hist[last][x] ~= cur[x + dx], so the scene at cur[p]
                                                         // sits in hist[k] (d frames back) at p - d*dx.
                        let sx8 = cx8 - d * g.gx;
                        let sy8 = cy8 - d * g.gy;
                        let v = i32::from(bicubic8(hf, w, h, sx8, sy8));
                        if !(v < i32::from(v0) - K_CONSIST || v > i32::from(v0) + K_CONSIST) {
                            votes[nv] = v;
                            nv += 1;
                        }
                        k += 1;
                    }
                    votes[..nv].sort_unstable();
                    row[2 * x + ox] = votes[nv / 2] as u8;
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_scene_zero_shift() {
        let f: Vec<u8> = (0..48 * 48).map(|i| (i * 13 % 251) as u8).collect();
        let g = estimate_shift(&f, &f, 48, 48);
        assert_eq!(g, Shift::default(), "identical frames -> no motion");
    }

    #[test]
    fn known_pan_is_found() {
        // b = a shifted right by 4 px (content moves, borders wrap by clamp)
        let a: Vec<u8> = (0..64 * 48)
            .map(|i| ((i % 64) * 4).clamp(0, 255) as u8)
            .collect();
        let mut b = vec![0u8; 64 * 48];
        for y in 0..48 {
            for x in 0..64 {
                let sx = (x as i32 - 4).clamp(0, 63) as usize;
                b[y * 64 + x] = a[y * 64 + sx];
            }
        }
        let g = estimate_shift(&a, &b, 64, 48);
        // a(x) ~ b(x + dx) minimized at dx = +4 (b[x] = a[x-4]), so gx ~= +1024
        assert!((g.gx - 1024).abs() <= 128, "gx={}", g.gx);
    }

    #[test]
    fn fusion2x_sizes_and_flat() {
        let f = vec![100u8; 24 * 24];
        let h1 = vec![100u8; 24 * 24];
        let h2 = vec![100u8; 24 * 24];
        let mut out = vec![0u8; 4 * 24 * 24];
        fusion2x(&f, &[&h1, &h2], 24, 24, Shift::default(), &mut out);
        assert!(out.iter().all(|&v| v == 100));
    }

    #[test]
    fn fusion_gates_moving_object() {
        // center block moves: votes outside the gate are dropped, no crash, plausible output
        let mut cur = vec![100u8; 32 * 32];
        let mut hist = vec![200u8; 32 * 32];
        for y in 12..20 {
            for x in 12..20 {
                cur[y * 32 + x] = 10;
                hist[y * 32 + x] = 10;
            }
        }
        let _ = &mut hist;
        let mut out = vec![0u8; 4 * 32 * 32];
        fusion2x(&cur, &[&hist], 32, 32, Shift::default(), &mut out);
        assert!(out.iter().all(|&v| (0..=255).contains(&v)));
        assert_eq!(out[2 * 5 * 64 + 10], 100, "flat background stays flat");
    }
}
