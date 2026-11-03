#!/usr/bin/env python3
"""Real-anime demo + benchmark for libfca.

Generates the README comparison images from REAL footage (Ergo Proxy,
Manglobe 2006 — 854x480 capture) and benchmarks the Rust port end-to-end:
ffmpeg rawvideo pipe -> fca_video -> /dev/null, against swscale lanczos
and the decode-only baseline.

Usage:
    python3 scripts/fca_real_anime.py <video.mp4> [--outdir docs/rust]
                                      [--frames 2700,900,600,1150] [--bench-frames 2400]

Requires: ffmpeg (in PATH), PIL, matplotlib, the crate built in release mode
(target/release/fca_upscale, target/release/fca_video).
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(REPO, "docs", "rust")
FCA_UPSCALE = os.path.join(REPO, "target", "release", "fca_upscale")
FCA_VIDEO = os.path.join(REPO, "target", "release", "fca_video")

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_SIZE = 15

# (frame_number, slug, human label)
SCENES = [
    (900, "dome_walk", "outside the dome"),
    (600, "veranda", "wood / grass textures"),
    (1150, "staircase", "dark gradients + smoke"),
    (2700, "night_silhouette", "night silhouette"),
]

GREEN = (34, 197, 94)
DARK = (10, 12, 16)


def sh(cmd, **kw):
    r = subprocess.run(cmd, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed: {cmd}")


def frame_to_ppm(png_path, ppm_path):
    Image.open(png_path).convert("RGB").save(ppm_path, format="PPM")


def ppm_to_png(ppm_path, png_path):
    Image.open(ppm_path).save(png_path, format="PNG")


def extract_frame(video, n, png_path, w=854, h=480):
    sh(["ffmpeg", "-v", "error", "-i", video, "-vf", f"select=eq(n\\,{n})",
        "-frames:v", "1", "-s", f"{w}x{h}", png_path, "-y"])
    assert os.path.exists(png_path), f"frame {n} not extracted"


def resample(src, dst, mode):
    im = Image.open(src).convert("RGB")
    out = im.resize((im.width * 2, im.height * 2), getattr(Image.Resampling, mode.upper()))
    out.save(dst, format="PNG")


def fca_up(src_png, dst_png, rule, tmp):
    src = os.path.join(tmp, "in.ppm")
    dst = os.path.join(tmp, "out.ppm")
    frame_to_ppm(src_png, src)
    sh([FCA_UPSCALE, "--rule", rule, src, dst])
    ppm_to_png(dst, dst_png)


def fca_pipeline(src_png, dst_png, w, h, extra):
    """ffmpeg png -> yuv444p[16le] raw -> fca_video -> raw -> png (2x)."""
    sixteen = "--16bit" in extra
    in_fmt = "yuv444p16le" if sixteen else "yuv444p"
    flags = [f for f in extra if f != "--yuv444"]  # --16bit implies yuv444
    cmd = (
        f"set -o pipefail; "
        f"ffmpeg -v error -i {src_png} -f rawvideo -pix_fmt {in_fmt} - | "
        f"{FCA_VIDEO} {w} {h} xbr {' '.join(flags)} | "
        f"ffmpeg -v error -f rawvideo -pix_fmt {in_fmt} -s {w*2}x{h*2} -i - -frames:v 1 {dst_png} -y"
    )
    sh(["bash", "-c", cmd])


def label(im, text):
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, FONT_SIZE)
    tw = d.textbbox((0, 0), text, font=f)[2]
    d.rectangle([0, 0, tw + 16, FONT_SIZE + 12], fill=(0, 0, 0))
    d.text((8, 5), text, font=f, fill=(255, 255, 255))
    return im


def montage(panels, cols, pad=6, bg=DARK):
    """panels: list of (PIL.Image, caption). All same size."""
    w, h = panels[0][0].size
    rows = (len(panels) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * w + (cols + 1) * pad, rows * h + (rows + 1) * pad), bg)
    for i, (im, cap) in enumerate(panels):
        r, c = divmod(i, cols)
        x = pad + c * (w + pad)
        y = pad + r * (h + pad)
        canvas.paste(label(im.copy(), cap), (x, y))
    return canvas


def crop2x(im, box):
    """Crop box on the 2x-scaled image (coordinates in source space)."""
    x0, y0, x1, y1 = box
    return im.crop((x0 * 2, y0 * 2, x1 * 2, y1 * 2))


def make_comparisons(video, frames, tmp):
    variants = [
        ("source (854x480)", "source"),
        ("nearest x2", "nearest"),
        ("bicubic x2", "bilinear"),
        ("lanczos x2", "lanczos"),
        ("fca fuzzy x2", "fuzzy"),
        ("fca xbr x2", "xbr"),
    ]
    results = {}  # slug -> {variant_key: PIL.Image}
    for n, slug, _ in SCENES:
        src = os.path.join(tmp, f"{slug}.png")
        extract_frame(video, n, src)
        d = {"source": Image.open(src).convert("RGB")}
        for caption, key in variants[1:]:
            if key in ("nearest", "bilinear", "lanczos"):
                dst = os.path.join(tmp, f"{slug}_{key}.png")
                resample(src, dst, "nearest" if key == "nearest" else key)
            else:
                dst = os.path.join(tmp, f"{slug}_{key}.png")
                fca_up(src, dst, key, tmp)
            d[key] = Image.open(dst).convert("RGB")
        # full video pipeline on the hero frame only (fast enough everywhere)
        if slug == "dome_walk":
            dst = os.path.join(tmp, f"{slug}_pipe.png")
            fca_pipeline(src, dst, 854, 480, ["--sharpen", "--dehalo", "--dn", "8"])
            d["pipe"] = Image.open(dst).convert("RGB")
        results[slug] = d

    # 1) hero frame: full 2x grid
    d = results["dome_walk"]
    panels = [
        (d["source"].resize((1708, 960), Image.Resampling.NEAREST), "source 854x480 (nearest blowup)"),
        (d["nearest"], "nearest x2"),
        (d["bilinear"], "bicubic x2"),
        (d["lanczos"], "lanczos x2"),
        (d["fuzzy"], "fca fuzzy x2"),
        (d["xbr"], "fca xbr x2"),
    ]
    montage(panels, 3).save(os.path.join(OUT, "real_cmp_full.png"))

    # 2) silhouette zoom (6x): nearest/bicubic/lanczos/fuzzy/xbr/pipeline
    face = (325, 215, 455, 385)  # Re-l silhouette, source coords
    zooms = [
        (crop2x(d["nearest"], face).resize(((face[2]-face[0])*4, (face[3]-face[1])*4), Image.Resampling.NEAREST), "nearest x6"),
        (crop2x(d["bilinear"], face).resize(((face[2]-face[0])*4, (face[3]-face[1])*4), Image.Resampling.NEAREST), "bicubic x6"),
        (crop2x(d["lanczos"], face).resize(((face[2]-face[0])*4, (face[3]-face[1])*4), Image.Resampling.NEAREST), "lanczos x6"),
        (crop2x(d["fuzzy"], face).resize(((face[2]-face[0])*4, (face[3]-face[1])*4), Image.Resampling.NEAREST), "fca fuzzy x6"),
        (crop2x(d["xbr"], face).resize(((face[2]-face[0])*4, (face[3]-face[1])*4), Image.Resampling.NEAREST), "fca xbr x6"),
    ]
    if "pipe" in d:
        zooms.append((crop2x(d["pipe"], face).resize(((face[2]-face[0])*4, (face[3]-face[1])*4), Image.Resampling.NEAREST),
                      "fca video pipeline x6 (xbr+CAS+dehalo)"))
    montage(zooms, 3).save(os.path.join(OUT, "real_cmp_zoom_silhouette.png"))

    # 3) thin-lines zoom: porch architecture (roof edge, railings)
    porch = (640, 100, 854, 300)
    gz = 4
    lines = [
        (crop2x(d["nearest"], porch).resize(((porch[2]-porch[0])*gz, (porch[3]-porch[1])*gz), Image.Resampling.NEAREST), "nearest x8"),
        (crop2x(d["lanczos"], porch).resize(((porch[2]-porch[0])*gz, (porch[3]-porch[1])*gz), Image.Resampling.NEAREST), "lanczos x8"),
        (crop2x(d["fuzzy"], porch).resize(((porch[2]-porch[0])*gz, (porch[3]-porch[1])*gz), Image.Resampling.NEAREST), "fca fuzzy x8"),
        (crop2x(d["xbr"], porch).resize(((porch[2]-porch[0])*gz, (porch[3]-porch[1])*gz), Image.Resampling.NEAREST), "fca xbr x8"),
    ]
    montage(lines, 2).save(os.path.join(OUT, "real_cmp_zoom_porch.png"))

    # 4) four scenes, 3 columns: source blowup / lanczos / fca xbr
    panels = []
    for n, slug, human in SCENES:
        d2 = results[slug]
        panels.append((d2["source"].resize((854, 480), Image.Resampling.NEAREST), f"{human} — source 480p"))
        panels.append((d2["lanczos"].resize((854, 480), Image.Resampling.LANCZOS), "lanczos x2 (shown 1:2)"))
        panels.append((d2["xbr"].resize((854, 480), Image.Resampling.LANCZOS), "fca xbr x2 (shown 1:2)"))
    montage(panels, 3).save(os.path.join(OUT, "real_scenes_grid.png"))

    # 5) dark scene gradient: staircase, lanczos vs 16-bit pipeline
    d3 = results["staircase"]
    src = os.path.join(tmp, "staircase.png")
    dst16 = os.path.join(tmp, "staircase_16.png")
    fca_pipeline(src, dst16, 854, 480, ["--16bit", "--sharpen"])
    stairs = (300, 120, 620, 330)
    z = 3
    panels = [
        (crop2x(d3["source"].resize((1708, 960), Image.Resampling.NEAREST), stairs).resize(((stairs[2]-stairs[0])*z, (stairs[3]-stairs[1])*z), Image.Resampling.NEAREST), "source x6"),
        (crop2x(d3["lanczos"], stairs).resize(((stairs[2]-stairs[0])*z, (stairs[3]-stairs[1])*z), Image.Resampling.NEAREST), "lanczos x6"),
        (crop2x(d3["xbr"], stairs).resize(((stairs[2]-stairs[0])*z, (stairs[3]-stairs[1])*z), Image.Resampling.NEAREST), "fca xbr (8-bit) x6"),
        (crop2x(Image.open(dst16).convert("RGB"), stairs).resize(((stairs[2]-stairs[0])*z, (stairs[3]-stairs[1])*z), Image.Resampling.NEAREST), "fca 16-bit two-band + CAS x6"),
    ]
    montage(panels, 2).save(os.path.join(OUT, "real_dark_16bit.png"))
    return results


def bench(video, bench_frames):
    """End-to-end throughput: decode -> rawvideo pipe -> tool -> null."""
    n = bench_frames
    probes = []

    def run(name, cmd):
        # warmup (page cache, codegen paths) then measure the best of 2
        times = []
        for i in range(3):
            t0 = time.perf_counter()
            sh(["bash", "-c", cmd])
            times.append(time.perf_counter() - t0)
            if i == 0:
                continue  # first run may be cold; keep all 3, report min
        best = min(times)
        fps = n / best
        probes.append((name, best, fps))
        print(f"{name:38s} {best:7.2f}s  {fps:8.1f} fps  ({fps/24:.2f}x realtime @24fps)")
        return fps

    pre = f"ffmpeg -v error -i {video} -frames:v {n} "
    run("decode only (baseline)",
        f"{pre}-f rawvideo -pix_fmt yuv444p - > /dev/null")
    run("ffmpeg swscale lanczos x2 (reference)",
        f"{pre}-vf scale=1708:960:flags=lanczos -f rawvideo -pix_fmt yuv444p - > /dev/null")
    run("fca_video fuzzy x2 (--yuv444)",
        f"{pre}-f rawvideo -pix_fmt yuv444p - | {FCA_VIDEO} 854 480 fuzzy --yuv444 > /dev/null")
    run("fca_video xbr x2 (--yuv444)",
        f"{pre}-f rawvideo -pix_fmt yuv444p - | {FCA_VIDEO} 854 480 xbr --yuv444 > /dev/null")
    run("fca_video temporal x2 (--yuv444)",
        f"{pre}-f rawvideo -pix_fmt yuv444p - | {FCA_VIDEO} 854 480 temporal --yuv444 > /dev/null")
    run("fca_video 16-bit two-band + CAS + dehalo",
        f"{pre}-f rawvideo -pix_fmt yuv444p - | {FCA_VIDEO} 854 480 xbr --yuv444 --16bit --sharpen --dehalo > /dev/null")
    return probes


def chart(probes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

    names = [p[0] for p in probes]
    fps = [p[2] for p in probes]
    colors = ["#64748b", "#0ea5e9", "#22c55e", "#16a34a", "#15803d", "#166534"]
    fig, ax = plt.subplots(figsize=(9.5, 4.2), constrained_layout=True)
    bars = ax.barh(names[::-1], fps[::-1], color=colors[: len(names)][::-1])
    ax.axvline(24, color="#ef4444", ls="--", lw=1.4)
    ax.text(24.6, len(names) - 0.45, "realtime @ 24 fps", color="#ef4444", fontsize=9)
    for b, v in zip(bars, fps[::-1]):
        ax.text(b.get_width() + 1.2, b.get_y() + b.get_height() / 2, f"{v:.0f} fps",
                va="center", fontsize=9)
    ax.set_xlabel("854x480 -> 1708x960, frames per second (best of 3)")
    ax.set_title("libfca on real anime footage (Ergo Proxy ep.1)\nend-to-end pipe: ffmpeg decode -> fca_video -> null")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(os.path.join(OUT, "real_bench.png"), dpi=150)
    plt.close(fig)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--outdir", default=OUT)
    ap.add_argument("--bench-frames", type=int, default=2400)
    ap.add_argument("--no-images", action="store_true",
                    help="skip comparisons, run the benchmark only")
    args = ap.parse_args()
    OUT = args.outdir
    os.makedirs(OUT, exist_ok=True)
    assert shutil.which("ffmpeg"), "ffmpeg required"
    assert os.path.exists(FCA_UPSCALE) and os.path.exists(FCA_VIDEO), \
        "build first: cargo build --release"

    probes = None
    if not args.no_images:
        with tempfile.TemporaryDirectory(prefix="fca_real_") as tmp:
            print("== comparisons ==")
            make_comparisons(args.video, SCENES, tmp)
    print("== benchmark ==")
    probes = bench(args.video, args.bench_frames)
    chart(probes)
    for f in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, f)
        print(f"{f}: {os.path.getsize(p)//1024} KB")


if __name__ == "__main__":
    main()
