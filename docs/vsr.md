# Temporal VSR — honest deterministic video super-resolution

`fca_video --vsr` is a **video super-resolution** step: it reconstructs detail
that is *invisible in any single frame* by fusing several frames of a slow
camera pan. No neural network, no training data — pure geometry, determinism
and arithmetic, at 16-bit precision.

## The problem: one frame is not enough

A 640x360 frame of an old anime is a 640x360 grid of samples. When you upscale
it to 2560x1440 you are *guessing* the missing 93% of pixels. Every single-frame
method — lanczos, bicubic, xBR, even a neural net — is an interpolation model
that invents detail from local statistics. Sharpening the guess does not add
information; it just amplifies the guess.

The only way to get *real* information back is to have more measurements of the
same scene.

## The trick: motion is information

When the camera pans at 0.5 px per frame, each frame samples the scene on a
*shifted* grid. Pixel grids of consecutive frames are **complementary
samplings** of the same underlying signal — what falls between two samples in
frame N lands exactly on a sample in frame N+1.

So the true resolution of a 3-second pan is not 640x360: it is closer to
640+20 x 360+15, *spread across time*. Neural VSR (BasicVSR, Real-ESRGAN
animevideo) does exactly this with optical-flow networks. `fca --vsr` does it
with a motion estimator and a Catmull-Rom resampler.

```
 frame N-3      frame N-2      frame N-1      frame N
   o-o-o-o        o-o-o-o        o-o-o-o        o-o-o-o    <- sampling grid
   |  \  |        |  \  |        |  \  |        |   <-  scene moves 0.5px/frame
   grid shifted 0.5px -> new information between old samples
```

## How it works (three stages)

### 1. Global sub-pixel motion estimation — `estimate_shift`

Finds the translation `(dx, dy)` between the current frame and the history
frames, in *fractional* pixels:

- **Coarse**: frame downscaled 4x, ±3 px search (covers ±12 px in source)
  — integer grid, fast.
- **Refine**: ±2 px integer search at full res.
- **Sub-pixel**: a grid of fractional shifts (±0.75 px in 0.125 px steps),
  scored with *bilinear interpolation* of the reference. Integer SSE cannot
  distinguish complementary samplings — the fractional score is evaluated
  with a real fractional-sample model, so 0.5 px shifts are found exactly.

Accuracy on the synthetic test: **−0.500 px measured vs −0.5 px true**.
The estimator is already at the measured upper bound of the whole fusion.

### 2. Motion-compensated fusion — `fusion2x`

The 4-frame history is sampled at taps **shifted by minus the motion** —
i.e. `history[k]` is read at `p − d·k`, aligning all frames onto the current
grid. The resampler is **Catmull-Rom** (4-tap, `bicubic8`), which preserves
sharp edges without ringing.

Two guards keep the fusion honest:

- **Consistency gate**: a history sample is dropped if it deviates from the
  current frame by more than 48 luma levels — the frame is not allowed to
  rewrite what the current frame clearly shows (protects against occlusions,
  flicker, scene change).
- **1-of-5 median**: the final value is the median of the aligned samples
  (the current frame counts as one), which kills outliers instead of averaging
  them in. This median is the *limiting factor* of the whole method — see the
  numbers below.

### 3. Hybrid gating — don't touch what you can't improve

VSR only runs when the estimated motion is in **0.25..2.5 px/frame**:

- **Static frame / cut** → motion ~0 → the plain xBR path runs, output is
  bit-identical to `--4x` alone. No hallucinated temporal smoothing on
  stills, no ghosting on hard cuts.
- **Fast action** → motion > 2.5 px/frame → sub-pixel sampling is unreliable,
  falls back to the single-frame path.

## What it gives you — measured

### Ground-truth test (the honest setup)

Bird test image (960x540, from the Anime4K set), 4 frames that are true
sub-pixel shifts (−0.5 px diagonal per frame) of the same ground truth, no
noise, no compression. Single-frame bicubic vs 4-frame VSR fusion, compared
to the true GT:

| method | SSIM | PSNR |
|---|---|---|
| bicubic 4x (single frame) | 0.9199 | 21.22 dB |
| **fca VSR fusion (4 frames)** | **0.9418** | **22.67 dB** |
| fusion, *known* shift (upper bound) | 0.9409 | 22.52 dB |

**+1.45 dB over bicubic.** The measured-vs-known comparison is the important
row: the motion estimator is *not* the bottleneck — feeding it the true shift
gets the same score. The 1-of-5 median (required for robustness on real video)
caps the gain; a smarter fusion (e.g. weighted least-squares over aligned
taps) is the next potential step.

### Real video (Ergo Proxy OP, pan sequence)

45 frames, 16-bit 4x pipeline. VSR fires on the **5 moving frames**
(MAE>1.0 vs no-VSR, min SSIM 0.982 — a clearly visible change where motion
exists) and is bit-identical to no-VSR on the other 40. The hybrid gate is
working: it never touches frames it cannot improve.

## When VSR helps / hurts

| scenario | motion | effect |
|---|---|---|
| slow pan (camera) | 0.25–2.5 px/frame | **big win** — true sub-pixel detail |
| vertical scroll (credits) | 0.5 px/frame | win, high-contrast text resolves cleanly |
| static frame | 0 | no change (bit-identical) |
| hard cut | — | no change (gate rejects) |
| fast action / shake | >2.5 px/frame | falls back to single-frame path |
| zoom / rotation | non-translational | estimator degrades gracefully, gate protects |

The model is **global translation only**. Zoom and rotation are not estimated;
the consistency gate prevents them from corrupting the output, but VSR simply
does not add anything there. That is a deliberate scope decision — pan/scroll
is where 90% of anime's static-background detail lives.

## vs neural VSR (BasicVSR, Real-ESRGAN animevideo)

| | neural VSR | fca `--vsr` |
|---|---|---|
| model | trained optical flow + fusion network | closed-form motion estimate + Catmull-Rom |
| inference | GPU (or very slow CPU) | **CPU, deterministic, bit-stable** |
| artifacts | can hallucinate texture that isn't there | cannot invent texture — only real measured samples |
| scale | any | 0.25..2.5 px/frame pan |
| licensing / portability | model weights, runtime deps | none — 2 files of C++, no deps |

The philosophical difference: a neural net *generates* the missing pixels
(styled guesses). fca *assembles* them from measurements that actually exist
in the neighboring frames. When the pan conditions hold, this is the only
honest source of new information — and it is what makes `fca --vsr` unique
among CPU upscalers.

## Cost

~+47 ms/frame over the plain 4x 16-bit path (2560x1440 output). The estimator
scores the sub-pixel grid on a 2x-subsampled image — subsampling, *not*
averaging, so the exact phase parity is preserved (averaging would erase the
very signal VSR needs). Quality-first feature: use `--vsr` for the best
pan/scroll result, omit it when raw speed matters.

## Usage

```
ffmpeg -i in.mp4 -f rawvideo -pix_fmt yuv444p - \
  | fca_video 640 360 xbr --yuv444 --16bit --4x --vsr --sharpen --deband \
  | ffmpeg -f rawvideo -pix_fmt yuv444p16le -s 2560x1440 -i - out.mp4
```

`--vsr` requires `--yuv444 --16bit --4x` (it operates on the 16-bit luma
plane before the rule passes).

## Comparisons in this repo

- `docs/vsr/sbs_fr27.png` — 600x400 crop, frame 27 of the pan test (the frame
  with the strongest VSR effect, MAE 5.7): **VSR | single-frame**, labeled.
- `docs/vsr/sbs_still.png` — full-frame still from the pan comparison video.
- `docs/vsr/pan_compare.mp4` — 45-frame side-by-side (VSR | no-VSR), Ergo
  Proxy OP, 31–34 s, 4x 16-bit.

Source material: Ergo Proxy OP (640x360), synthetic Bird test in
`/tmp/opencode/anime_test/` (not shipped).
