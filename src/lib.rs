//! # libfca — fuzzy cellular-rule upscaling
//!
//! CPU integer 2x upscaler for old video / anime: rule-based edge-directed
//! scaling with **fuzzy membership**, **temporal tile cache**, multi-frame VSR
//! fusion and integer-only post FX. No ML, no GPU, no float math —
//! deterministic, integer-only, fast on any CPU.
//!
//! Built on the EAFAR paradigm: S5 fuzzy membership instead of boolean
//! decisions, S3 sleep/wake region scheduling for temporal stability, S7
//! diffusion-style pre-filtering. The upscale rule itself is a cellular
//! automaton over the pixel grid.
//!
//! Module map:
//! - [`grid`] — the uint8 lattice field
//! - [`rules`] — `scale2x` (EPX) / `fuzzy` / `xbr` cellular rules (the quality ladder)
//! - [`denoise`] — median 3x3 pre-filter
//! - [`postfx`] — bicubic2x, edge-aware denoise, dehalo, CAS sharpen, contrast, vibrance, deband
//! - [`temporal`] — sleep/wake tile cache (unchanged tiles are never recomputed)
//! - [`vsr`] — multi-frame subpixel fusion (video super-resolution without hallucination)
//! - [`ppm`] — PPM P6 I/O for the CLI
//!
//! Determinism: every function is integer-only with fixed-point arithmetic;
//! the same input always produces the same output on any platform.

pub mod denoise;
pub mod grid;
pub mod postfx;
pub mod ppm;
pub mod rules;
pub mod temporal;
pub mod vsr;

pub use grid::Grid;
pub use temporal::{Rule, TemporalUpscaler};
pub use vsr::Shift;
