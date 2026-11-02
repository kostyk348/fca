//! S3-inspired temporal tile cache.
//!
//! The EAFAR thesis applied to video upscaling: an unchanged tile must not be
//! recomputed. The frame is partitioned into `tile x tile` regions; a tile
//! either SLEEPS (identical to previous frame -> previous upscale is reused,
//! zero work) or WAKES (changed -> local rule re-runs on that tile only, with
//! clamped 3x3 reads from the full input frame).
//!
//! Reusing the previous upscale for unchanged tiles is not just a speed win:
//! it kills frame-to-frame flicker of static backgrounds, which is the main
//! visual artifact of single-frame edge-directed rules on compressed content.
//!
//! Determinism: tiles iterate in ascending (ty,tx) order; the same input
//! stream produces identical output.

use crate::grid::Grid;
use crate::rules::{fz_pick4, px, scale2x_fuzzy, scale2x_xbr, xbr_pick4};

/// Which cellular rule the cache uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rule {
    Fuzzy,
    Xbr,
}

/// Sleep/wake statistics, exposed for measurement.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TileStats {
    /// tiles seen
    pub total: u64,
    /// woke (changed) -> rule re-run
    pub recomputed: u64,
    /// slept (identical) -> cache reuse
    pub reused: u64,
}

/// Frame-to-frame temporal upscaler with a per-tile cache.
pub struct TemporalUpscaler {
    w: usize,
    h: usize,
    tile: usize,
    tx: usize,
    ty: usize,
    rule: Rule,
    first: bool,
    prev: Vec<u8>,
    cache: Vec<u8>,
    stats: TileStats,
}

impl TemporalUpscaler {
    /// Cache for `w x h` gray frames with the given tile size (default 32).
    pub fn new(w: usize, h: usize, tile: usize, rule: Rule) -> TemporalUpscaler {
        let tile = tile.max(1);
        TemporalUpscaler {
            w,
            h,
            tile,
            tx: w.div_ceil(tile),
            ty: h.div_ceil(tile),
            rule,
            first: true,
            prev: vec![0; w * h],
            cache: vec![0; 4 * w * h],
            stats: TileStats::default(),
        }
    }

    pub fn tiles_x(&self) -> usize {
        self.tx
    }

    pub fn tiles_y(&self) -> usize {
        self.ty
    }

    pub fn stats(&self) -> TileStats {
        self.stats
    }

    /// Process one frame: `in` = `w*h` gray bytes, `out` = `2w*2h` bytes.
    pub fn process(&mut self, input: &[u8], out: &mut [u8]) {
        if self.first {
            let g = Grid::from_vec(self.w, self.h, input.to_vec());
            let o = match self.rule {
                Rule::Xbr => scale2x_xbr(&g),
                Rule::Fuzzy => scale2x_fuzzy(&g),
            };
            self.cache.copy_from_slice(&o.data);
            // FIX vs the C original: seed the previous-frame buffer here too.
            // The C `process()` never fills `prev_` on the first frame, so its
            // second frame compared against zeros and could never sleep.
            self.prev.copy_from_slice(input);
            self.stats.total += (self.tx * self.ty) as u64;
            self.stats.recomputed += (self.tx * self.ty) as u64;
            self.first = false;
        } else {
            for ty in 0..self.ty {
                for tx in 0..self.tx {
                    let x0 = tx * self.tile;
                    let y0 = ty * self.tile;
                    let x1 = (x0 + self.tile).min(self.w);
                    let y1 = (y0 + self.tile).min(self.h);
                    let same = (y0..y1).all(|y| {
                        input[y * self.w + x0..y * self.w + x1]
                            == self.prev[y * self.w + x0..y * self.w + x1]
                    });
                    self.stats.total += 1;
                    if same {
                        self.stats.reused += 1;
                        continue;
                    }
                    self.stats.recomputed += 1;
                    match self.rule {
                        Rule::Xbr => self.xbr_tile_region(input, x0, y0, x1, y1),
                        Rule::Fuzzy => self.fz_tile_region(input, x0, y0, x1, y1),
                    }
                }
            }
            self.prev.copy_from_slice(input);
        }
        out.copy_from_slice(&self.cache);
    }

    fn set2x(&mut self, x: usize, y: usize, picks: [u8; 4]) {
        let ow = 2 * self.w;
        let (ox, oy) = (2 * x, 2 * y);
        self.cache[oy * ow + ox] = picks[0];
        self.cache[oy * ow + ox + 1] = picks[1];
        self.cache[(oy + 1) * ow + ox] = picks[2];
        self.cache[(oy + 1) * ow + ox + 1] = picks[3];
    }

    fn xbr_tile_region(&mut self, input: &[u8], x0: usize, y0: usize, x1: usize, y1: usize) {
        let g = Grid {
            w: self.w,
            h: self.h,
            data: input.to_vec(),
        };
        for y in y0..y1 {
            for x in x0..x1 {
                let picks = xbr_pick4(
                    px(&g, x as isize - 1, y as isize - 1),
                    px(&g, x as isize, y as isize - 1),
                    px(&g, x as isize + 1, y as isize - 1),
                    px(&g, x as isize - 1, y as isize),
                    g.data[y * self.w + x],
                    px(&g, x as isize + 1, y as isize),
                    px(&g, x as isize - 1, y as isize + 1),
                    px(&g, x as isize, y as isize + 1),
                    px(&g, x as isize + 1, y as isize + 1),
                );
                self.set2x(x, y, picks);
            }
        }
    }

    fn fz_tile_region(&mut self, input: &[u8], x0: usize, y0: usize, x1: usize, y1: usize) {
        let g = Grid {
            w: self.w,
            h: self.h,
            data: input.to_vec(),
        };
        for y in y0..y1 {
            for x in x0..x1 {
                let picks = fz_pick4(
                    px(&g, x as isize, y as isize - 1),
                    px(&g, x as isize - 1, y as isize),
                    g.data[y * self.w + x],
                    px(&g, x as isize + 1, y as isize),
                    px(&g, x as isize, y as isize + 1),
                );
                self.set2x(x, y, picks);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(w: usize, h: usize, seed: u8) -> Vec<u8> {
        (0..w * h)
            .map(|i| (i as u8).wrapping_mul(seed).wrapping_add(seed))
            .collect()
    }

    #[test]
    fn first_frame_recomputes_everything() {
        let mut tu = TemporalUpscaler::new(8, 6, 4, Rule::Fuzzy);
        let f = frame(8, 6, 1);
        let mut out = vec![0u8; 4 * 8 * 6];
        tu.process(&f, &mut out);
        let s = tu.stats();
        assert_eq!(s.total, 4);
        assert_eq!(s.recomputed, 4);
        assert_eq!(s.reused, 0);
    }

    #[test]
    fn static_frame_sleeps() {
        let mut tu = TemporalUpscaler::new(8, 6, 4, Rule::Fuzzy);
        let f = frame(8, 6, 3);
        let mut o1 = vec![0u8; 4 * 8 * 6];
        tu.process(&f, &mut o1);
        let mut o2 = vec![0u8; 4 * 8 * 6];
        tu.process(&f, &mut o2);
        assert_eq!(
            o1, o2,
            "static background: output literally does not change"
        );
        let s = tu.stats();
        assert_eq!(s.reused, 4);
        assert_eq!(s.recomputed, 4, "only the first frame woke all tiles");
    }

    #[test]
    fn partial_change_wakes_one_tile() {
        let mut tu = TemporalUpscaler::new(8, 6, 4, Rule::Fuzzy);
        let mut f = frame(8, 6, 5);
        let mut out = vec![0u8; 4 * 8 * 6];
        tu.process(&f, &mut out);
        f[0] = f[0].wrapping_add(37); // change one pixel in tile (0,0)
        tu.process(&f, &mut out);
        let s = tu.stats();
        assert_eq!(s.reused, 3);
        assert_eq!(s.recomputed, 5);
    }

    #[test]
    fn tile_output_matches_full_rule() {
        let mut tu = TemporalUpscaler::new(8, 6, 4, Rule::Xbr);
        let f = frame(8, 6, 7);
        let mut out = vec![0u8; 4 * 8 * 6];
        tu.process(&f, &mut out);
        let g = Grid::from_vec(8, 6, f);
        let full = scale2x_xbr(&g);
        assert_eq!(out, full.data);
    }
}
