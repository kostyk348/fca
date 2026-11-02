//! S7-inspired pre-filter: median 3x3 (edge-preserving denoise).
//!
//! Removes blocky compression noise / salt before upscaling; keeps edges
//! (unlike box blur).

use crate::grid::Grid;
use crate::rules::px;
use rayon::prelude::*;

/// Median of 9 bytes.
pub fn median9(mut v: [u8; 9]) -> u8 {
    v.sort_unstable();
    v[4]
}

/// Median 3x3 over the whole plane (borders use clamped samples).
pub fn median3x3(input: &Grid) -> Grid {
    let w = input.w;
    let mut out = Grid::new(w, input.h);
    out.data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, cell) in row.iter_mut().enumerate() {
            let mut v = [0u8; 9];
            let mut k = 0;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    v[k] = px(input, (x as i32 + dx) as isize, (y as i32 + dy) as isize);
                    k += 1;
                }
            }
            *cell = median9(v);
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_of_nine() {
        assert_eq!(median9([9, 1, 8, 2, 7, 3, 6, 4, 5]), 5);
        assert_eq!(median9([0; 9]), 0);
        assert_eq!(median9([255; 9]), 255);
    }

    #[test]
    fn salt_noise_is_removed() {
        let mut g = Grid::new(5, 5);
        g.data.iter_mut().for_each(|v| *v = 100);
        g.data[12] = 0; // single black speckle in the center
        let o = median3x3(&g);
        assert_eq!(o.data[12], 100, "median kills the isolated speckle");
    }
}
