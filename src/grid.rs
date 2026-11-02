//! The uint8 quantized lattice field: row-major, single plane.

/// A single 8-bit plane.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct Grid {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

impl Grid {
    /// Zero-filled grid of `w * h`.
    pub fn new(w: usize, h: usize) -> Grid {
        Grid {
            w,
            h,
            data: vec![0; w * h],
        }
    }

    /// Grid from raw bytes (length must be `w * h`).
    ///
    /// # Panics
    /// If `data.len() != w * h`.
    pub fn from_vec(w: usize, h: usize, data: Vec<u8>) -> Grid {
        assert_eq!(data.len(), w * h, "grid size mismatch");
        Grid { w, h, data }
    }

    /// Borrow one row.
    pub fn row(&self, y: usize) -> &[u8] {
        &self.data[y * self.w..(y + 1) * self.w]
    }

    /// Mutably borrow one row.
    pub fn row_mut(&mut self, y: usize) -> &mut [u8] {
        let w = self.w;
        &mut self.data[y * w..(y + 1) * w]
    }
}

impl std::fmt::Debug for Grid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Grid({}x{})", self.w, self.h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_and_sizes() {
        let mut g = Grid::new(4, 3);
        assert_eq!(g.data.len(), 12);
        g.row_mut(1).copy_from_slice(&[1, 2, 3, 4]);
        assert_eq!(g.row(1), &[1, 2, 3, 4]);
        assert_eq!(g.row(0), &[0, 0, 0, 0]);
    }

    #[test]
    #[should_panic(expected = "grid size mismatch")]
    fn from_vec_checks_size() {
        let _ = Grid::from_vec(4, 4, vec![0; 3]);
    }
}
