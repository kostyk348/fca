<div align="center">

# libfca — fuzzy cellular-rule upscaling

**CPU integer 2x upscaler for old video / anime: rule-based edge-directed
scaling with fuzzy membership, a temporal tile cache and VSR fusion.
No ML, no GPU, no float math — deterministic, integer-only, fast on any CPU.**

[![Release](https://img.shields.io/github/v/release/kostyk348/fca?sort=semver&color=0ea5e9&label=release)](https://github.com/kostyk348/fca/releases)
[![CI](https://img.shields.io/github/actions/workflow/status/kostyk348/fca/ci.yml?branch=main&label=CI)](https://github.com/kostyk348/fca/actions/workflows/ci.yml)
[![tests](https://img.shields.io/badge/tests-30%20passing-22c55e)](#development)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange)](https://www.rust-lang.org)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](CONTRIBUTING.md)

[English](README.md) · [Русский](README.ru.md)

</div>

---

Built on the EAFAR paradigm (`kostyk348/eafar`): S5 fuzzy membership instead of
boolean decisions, S3 sleep/wake region scheduling for temporal stability, S7
diffusion-style pre-filtering. The upscale rule itself is a cellular automaton
over the pixel grid (AVX-FCA concept).

## Rules (quality ladder)

| rule | idea |
|---|---|
| `scale2x` | classic EPX: 3x3 rule, hard tolerance picks (baseline) |
| `fuzzy` | soft blend by triangle membership `m(d) = 255 - d*51>>3` — anti-aliased diagonals, no frame-to-frame flicker at thresholds |
| `xbr` | + diagonal corners (A,C,G,I) and weighted pair blend: thin 1px diagonal lines survive |

All rules share the same skeleton: `out = mix(E, candidate, strength)`,
`strength = m(p~q) * (1-m(perp1~p)) * (1-m(perp2~q))`. Results stay inside the
3x3 min/max — no ringing by construction.

## The Rust port (v0.3)

The prototype was C++20 with a hand-written AVX2 path kept bit-exact against a
scalar reference. The port keeps **the scalar arithmetic as the single source
of truth** (it was the spec anyway), parallelizes rows with rayon instead of
OpenMP, and drops every C dependency. The port is verified **bit-exact against
the C reference**: 5 golden test sets (rules, the whole post FX chain, VSR,
temporal cache) are generated from the original headers and embedded in
`tests/golden.rs`.

| | C++ prototype | Rust 0.3 |
|---|---|---|
| SIMD | hand-written AVX2 + scalar reference | scalar reference only (auto-vectorized), rayon rows |
| deps | C++20 + AVX2, OpenMP | Rust 1.80+, no C deps |
| tests | bench-only | 30 (incl. C-golden bit-exactness) |
| temporal cache | 2nd frame never sleeps (bug) | fixed, golden-verified |
| `bicubic8` edge taps | read past the plane | clamped (documented fix) |

Benchmarks (Rust release build, sandbox x86-64): 640x360 → 1280x720 per plane —
scalar scale2x **2.2 ms**, fuzzy **4.5 ms**, xbr **6.1 ms** (C prototype,
Ryzen 780M, `-O2 -march=native`: 24.2 / 1.75 / scalar-only ms at 1280x720 —
see the original measurements below; different CPU, treat as orientation).

<details>
<summary><b>C prototype benchmarks and SSIM tables (unchanged content)</b></summary>

x86-64 AVX2 (AMD Ryzen 780M, g++ 16, `-O2 -march=native`), 1280x720 → 2560x1440:

| operation | time |
|---|---|
| scalar scale2x | 24.2 ms |
| AVX2 scale2x | 1.75 ms (x13.8) |
| AVX2 fuzzy | 1.26 ms |

Quality (SSIM vs lanczos reference, closeness to lanczos):

| content | lanczos | fuzzy | xbr |
|---|---|---|---|
| text / subtitles | 0.9519 | **0.9771** | **0.9771** |
| diagonals / lines (testsrc2) | 0.9158 | 0.9488 | **0.9749** |
| fine texture (mandelbrot) | **0.9707** | 0.9675 | — |
| live footage (series) | **0.9619** | 0.9599 | 0.9606 |

vs Neural upscalers (Anime4K), Ergo Proxy OP 640x360 → 2560x1440, SSIM vs
lanczos-4x reference:

| method | SSIM |
|---|---|
| Anime4K GAN_x4_UUL | 0.870 |
| Anime4K Restore_CNN_M + CNN_x2 x2 | 0.880 |
| **fca xbr 4x (clean)** | **0.845** |
| **fca xbr 4x + FX (mild)** | **0.899** |

VSR synthetic ground-truth test (Bird 960x540, 0.5 px/frame diagonal pan):

| method | SSIM | PSNR |
|---|---|---|
| bicubic 4x (single frame) | 0.9199 | 21.22 dB |
| **fca VSR fusion (4 frames)** | **0.9418** | **22.67 dB** |
| fusion with *known* shift (upper bound) | 0.9409 | 22.52 dB |

</details>

Takeaway: lanczos wins on smooth photo-like textures, libfca wins on anything
with lines, text, diagonals, or pixel art — which is exactly old anime/DVD.

## Rich-look pipeline (yuv444, 4x, FX)

`fca_video` color mode upsamples luma with the cellular rule and chroma with
bicubic, then applies optional "rich look" post FX — all CPU, deterministic:

```
ffmpeg -i in.mp4 -f rawvideo -pix_fmt yuv444p - \
  | fca_video 640 360 xbr --yuv444 --4x --sharpen --vibrance 30 --contrast 10 --deband \
  | ffmpeg -f rawvideo -pix_fmt yuv444p -s 2560x1440 -i - out.mp4
```

| step | what it does |
|---|---|
| `--vsr` | **video super-resolution (temporal fusion)**: estimates global sub-pixel motion between the current frame and a 4-frame history (coarse ±12px on down4, refine ±2px, bilinear sub-pixel grid search ±0.75/0.125px), then fuses 4 frames at Catmull-Rom (bicubic8) taps shifted by the motion, with a consistency gate and median. Only active at 0.25..2.5 px/frame motion (pan/scroll); static and cut frames fall back to the plain path untouched. Deterministic, no ML. |
| `--dn N` | **edge-aware denoise** (bilateral-style, 3x3, division-free): kills compression noise/grain BEFORE it is scaled up and amplified by sharpening. Lines survive, flat zones become clean. 0..255. |
| `--dehalo` | **halo/ringing suppression**: DVD/BD compression paints bright halos around ink lines. Any pixel near a strong edge (cross range >= 96) brighter than m+7/8*(M-m) is a halo peak and is clamped back into the clean gradient. Applied at the 2x stage in 4x mode (4x cheaper; the rule cannot recreate peaks). |
| `--16bit` | **two-band 16-bit**: hi = v>>8 -> cellular rule, lo = v&255 -> bicubic; adaptive per-pixel mix (rule only where 3x3 contrast is high). Edges stay sharp, gradients stay smooth, **banding disappears** |
| `--4x` | rule applied twice (480 -> 960 -> 1920); chroma bicubic x2 |
| `--sharpen` | contrast-adaptive sharpen (FidelityFX CAS spirit): gain ∝ local contrast, clamped to 3x3 min/max — no ringing, no grain in flat areas |
| `--vibrance N` | chroma saturation gain (0..255) |
| `--contrast N` | luma contrast (0..255) |
| `--deband` | gradient dither in flat zones — breaks banding steps on skies |

### VSR (temporal fusion) — honest deterministic super-resolution

No neural net, but the same trick neural VSR uses: on a slow pan the camera
sub-pixel shift carries *new* sampling information in every frame, so fusing
several frames at motion-compensated sub-pixel taps reconstructs detail that no
single-frame upscaler can see.

> **Full story — [`docs/vsr.md`](docs/vsr.md).** Comparison files:
> `docs/vsr/pan_compare.mp4`, `docs/vsr/sbs_fr27.png`, `docs/vsr/sbs_still.png`.

## Temporal cache (S3)

`TemporalUpscaler` partitions the frame into tiles; a tile identical to the
previous frame **sleeps** (previous upscale reused, zero work), a changed tile
**wakes** (local rule re-run with clamped 3x3 reads). Deterministic, output is
bit-identical to single-frame fuzzy. Kills background flicker on static scenes
and costs almost nothing there. Stats are printed to stderr by `fca_video`.

## Build

```bash
cargo build --release     # fca_upscale, fca_video
```

Requires a Rust toolchain (1.80+). No C libraries, no OpenMP, no AVX2 flag —
the binary runs on any 64-bit CPU.

## Usage

### Single image (PPM P6)

```
fca_upscale in.ppm out.ppm --rule scale2x   # EPX baseline
fca_upscale in.ppm out.ppm --rule fuzzy     # soft blend (default)
fca_upscale in.ppm out.ppm --rule xbr       # diagonals survive
fca_upscale in.ppm out.ppm --bench          # print per-rule timings
```

### Video via ffmpeg pipe (gray rawvideo)

```
ffmpeg -i in.mp4 -f rawvideo -pix_fmt gray - \
  | fca_video 480 270 fuzzy --denoise \
  | ffmpeg -f rawvideo -pix_fmt gray -s 960x540 -i - out.mp4
```

modes: `scale2x | fuzzy | xbr | temporal`, flags: `--denoise --tile N --rule xbr`
(temporal with xbr rule), color + FX: `--yuv444 --16bit --4x --dn N --dehalo
--sharpen --vibrance N --contrast N --deband --vsr`.

Quality pipeline order: denoise (pre-upscale) -> upscale -> dehalo (2x stage) ->
sharpen -> contrast -> deband.

### mpv shaders (real-time GPU)

```
mpv --glsl-shaders="~~/.config/mpv/shaders/fca_xbr2.glsl" --window-scale=2 anime.mkv
```

- `shaders/fca_scale2x.glsl` — EPX, hard rule
- `shaders/fca_fuzzy_scale2x.glsl` — fuzzy blend
- `shaders/fca_xbr2.glsl` — xBR-style with diagonal corners (recommended)

## Development

```bash
cargo fmt --check                          # formatting
cargo clippy --all-targets -- -D warnings  # lints, zero tolerance
cargo test                                 # 30 tests, no GPU/SIMD requirements
cargo build --release
```

The golden tests (`tests/golden.rs`) embed bit-exact outputs of the original C
implementation, so any deviation from the reference arithmetic is caught
immediately. The golden generator lives in the maintainer's scripts
(`fca_golden.cpp` + `fca_make_golden_tests.py`); regenerating it requires a C++
compiler.

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md).

## Layout

```
src/grid.rs              uint8 field (row-major plane)
src/rules.rs             scale2x / fuzzy / xbr cellular rules
src/temporal.rs          S3 tile cache (sleep/wake)
src/denoise.rs           median 3x3 pre-filter
src/postfx.rs            bicubic 2x (chroma), CAS sharpen, contrast, vibrance, deband, denoise_edge, dehalo
src/vsr.rs               shift estimator + multi-frame fusion (temporal VSR)
src/ppm.rs               PPM P6 I/O
src/bin/fca_upscale.rs   PPM CLI
src/bin/fca_video.rs     rawvideo pipe CLI
tests/golden.rs          bit-exact C reference outputs
shaders/                 mpv GLSL versions
docs/                    comparison images
docs/vsr/                VSR comparisons (side-by-side video, crops) — see docs/vsr.md
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
