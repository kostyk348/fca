# libfca — fuzzy cellular-rule upscaling (fca-video)

CPU integer 2x upscaler for old video / anime: rule-based edge-directed scaling
with **fuzzy membership**, **AVX2**, **temporal tile cache**, optional median
denoise. No ML, no GPU, no float-heavy math — deterministic, integer-only,
fast on any CPU with AVX2.

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
3x3 min/max — no ringing by construction. Scalar reference is **bit-exact**
with the AVX2 path (same fixed-point arithmetic).

## Benchmarks

x86-64 AVX2 (AMD Ryzen 780M, g++ 16, `-O2 -march=native`):

| operation | 1280x720 -> 2560x1440 |
|---|---|
| scalar scale2x | 24.2 ms |
| AVX2 scale2x | 1.75 ms (x13.8) |
| AVX2 fuzzy | 1.26 ms |
| AVX2 xbr | scalar-only for now |

Real video pipeline (1080p source -> 480p -> 960p, 193 frames, gray):

| mode | ms/frame |
|---|---|
| fuzzy | 0.70 |
| xbr | 2.98 (scalar) |
| temporal (fuzzy) | 1.69 (9% tiles reused — static scenes sleep much more) |
| temporal --rule xbr | 2.71 (10% reused; output bit-identical to xbr) |
| fuzzy + median denoise | 8.5 (offline use only) |

## Quality (SSIM vs lanczos reference)

Reference is built by lanczos downscale (480p) then upscaled back (960p) and
compared to the lanczos-downscaled original — i.e. *closeness to lanczos*.

| content | lanczos | fuzzy | xbr |
|---|---|---|---|
| text / subtitles | 0.9519 | **0.9771** | **0.9771** |
| diagonals / lines (testsrc2) | 0.9158 | 0.9488 | **0.9749** |
| fine texture (mandelbrot) | **0.9707** | 0.9675 | — |
| live footage (series) | **0.9619** | 0.9599 | 0.9606 |

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
| `--16bit` | **two-band 16-bit**: hi = v>>8 -> cellular rule, lo = v&255 -> bicubic; adaptive per-pixel mix (rule only where 3x3 contrast is high). Edges stay sharp, gradients stay smooth, **banding disappears** |
| `--4x` | rule applied twice (480 -> 960 -> 1920); chroma bicubic x2 |
| `--sharpen` | contrast-adaptive sharpen (FidelityFX CAS spirit): gain ∝ local contrast, clamped to 3x3 min/max — no ringing, no grain in flat areas |
| `--vibrance N` | chroma saturation gain (0..255) |
| `--contrast N` | luma contrast (0..255) |
| `--deband` | gradient dither in flat zones — breaks banding steps on skies |

### vs Neural upscalers (Anime4K), Ergo Proxy OP 640x360 -> 2560x1440

SSIM vs lanczos-4x reference (higher = closer to the smooth reference; neural
upscalers intentionally deviate from it):

| method | SSIM |
|---|---|
| Anime4K GAN_x4_UUL | 0.870 |
| Anime4K Restore_CNN_M + CNN_x2 x2 | 0.880 |
| **fca xbr 4x (clean)** | **0.845** |
| **fca xbr 4x + FX (mild)** | **0.899** |

Reference images: `docs/anime4k_cmp_full.png`, `docs/anime4k_cmp_crop.png`
(lanczos | fca clean | fca+FX | Anime4K GAN | Anime4K CNN, center crops below).

## Temporal cache (S3)

`TemporalUpscaler` partitions the frame into 32x32 tiles; a tile identical to
the previous frame **sleeps** (previous upscale reused, zero work), a changed
tile **wakes** (local rule re-run with 1px halo). Deterministic, output is
bit-identical to single-frame fuzzy. Kills background flicker on static
scenes and costs almost nothing there.

## Build

```
make            # fca_upscale, fca_bench, fca_video
./fca_bench 1280 720
```

Requires a C++20 compiler with AVX2 (`-march=native`). No external deps.

## Usage

### Single image (PPM P6)

```
./fca_upscale in.ppm out.ppm --scalar   # reference
./fca_upscale in.ppm out.ppm --fuzzy    # AVX2 fuzzy (default)
./fca_upscale in.ppm out.ppm --check    # verify AVX2 == scalar
```

### Video via ffmpeg pipe (gray rawvideo)

```
ffmpeg -i in.mp4 -f rawvideo -pix_fmt gray - \
  | ./fca_video 480 270 fuzzy --denoise \
  | ffmpeg -f rawvideo -pix_fmt gray -s 960x540 -i - out.mp4
```

modes: `scale2x | fuzzy | xbr | temporal`, flags: `--denoise --tile N --rule xbr`
(temporal with xbr rule), color + FX: `--yuv444 --16bit --4x --sharpen --vibrance N
--contrast N --deband`.

### mpv shaders (real-time GPU)

```
mpv --glsl-shaders="~~/.config/mpv/shaders/fca_xbr2.glsl" --window-scale=2 anime.mkv
```

- `shaders/fca_scale2x.glsl` — EPX, hard rule
- `shaders/fca_fuzzy_scale2x.glsl` — fuzzy blend
- `shaders/fca_xbr2.glsl` — xBR-style with diagonal corners (recommended)

## Layout

```
include/fca/grid.hpp     uint8 field (SoA plane)
include/fca/rules.hpp    scale2x / fuzzy / xbr, scalar + AVX2
include/fca/temporal.hpp S3 tile cache (sleep/wake)
include/fca/denoise.hpp  median 3x3 pre-filter
include/fca/postfx.hpp   bicubic 2x (chroma), CAS sharpen, contrast, vibrance, deband
src/postfx.cpp           FX implementation
src/fca_upscale.cpp      PPM CLI
src/fca_video.cpp        rawvideo pipe CLI
bench/bench.cpp          benchmark + correctness check
shaders/                 mpv GLSL versions
docs/                    comparison images
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
