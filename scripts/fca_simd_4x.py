#!/usr/bin/env python3
"""AVX2 + 4x max-quality showcase for libfca.

Generates the README images for the SIMD return and the 4x HQ preset:
  - docs/rust/simd_bench.png   — end-to-end benchmark, scalar vs AVX2 vs 4x HQ
  - docs/rust/real_4x_full.png — 4x full-frame: lanczos vs fca --hq --4x
  - docs/rust/real_4x_zoom.png — 4x zoom crops (silhouette / thin lines)

Usage:
    python3 scripts/fca_simd_4x.py <video.mp4> [--outdir docs/rust]

Requires: ffmpeg, PIL, matplotlib, the crate built in release mode.
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fca_real_anime import (  # noqa: E402
    FCA_VIDEO,
    OUT,
    SCENES,
    DARK,
    extract_frame,
    label,
    montage,
    sh,
)


def fca_run(src_png, dst_png, w, h, extra, simd=True):
    """png -> yuv444p16le raw -> fca_video (any scale) -> png."""
    fmt = "yuv444p16le" if ("--16bit" in extra or "--hq" in extra) else "yuv444p"
    # output scale: --4x doubles twice, default 2x
    scale = 4 if ("--4x" in extra) else 2
    ow, oh = w * scale, h * scale
    env = dict(os.environ)
    if not simd:
        env["FCA_SIMD"] = "0"
    cmd = (
        f"set -o pipefail; "
        f"ffmpeg -v error -i {src_png} -f rawvideo -pix_fmt {fmt} - | "
        f"{FCA_VIDEO} {w} {h} xbr {' '.join(extra)} | "
        f"ffmpeg -v error -f rawvideo -pix_fmt {fmt} -s {ow}x{oh} -i - -frames:v 1 {dst_png} -y"
    )
    r = subprocess.run(["bash", "-c", cmd], env=env)
    if r.returncode != 0:
        raise RuntimeError(f"fca_run failed: {extra}")


def lanczos4x(src_png, dst_png):
    sh(["ffmpeg", "-v", "error", "-i", src_png, "-vf", "scale=3416:1920:flags=lanczos",
        "-frames:v", "1", dst_png, "-y"])


def cropN(im, box, n):
    """Crop box on an n-times-scaled image (coords in source space)."""
    x0, y0, x1, y1 = box
    return im.crop((x0 * n, y0 * n, x1 * n, y1 * n))


def bench(video):
    """End-to-end throughput: AVX2 vs scalar pairs + the 4x HQ preset."""
    rows = [
        # (name, frames, cmd_template, simd_env)
        ("decode only (baseline)", 1500, "{pre}-f rawvideo -pix_fmt yuv444p - > /dev/null", None),
        ("ffmpeg swscale lanczos x2 (reference)", 1500,
         "{pre}-vf scale=1708:960:flags=lanczos -f rawvideo -pix_fmt yuv444p - > /dev/null", None),
        ("fca fuzzy x2 — scalar (FCA_SIMD=0)", 1200,
         "{pre}-f rawvideo -pix_fmt yuv444p - | {FV} 854 480 fuzzy --yuv444 > /dev/null", "0"),
        ("fca fuzzy x2 — AVX2", 1200,
         "{pre}-f rawvideo -pix_fmt yuv444p - | {FV} 854 480 fuzzy --yuv444 > /dev/null", "1"),
        ("fca xbr x2 (scalar kernel)", 1200,
         "{pre}-f rawvideo -pix_fmt yuv444p - | {FV} 854 480 xbr --yuv444 > /dev/null", "1"),
        ("fca temporal x2 — AVX2", 1200,
         "{pre}-f rawvideo -pix_fmt yuv444p - | {FV} 854 480 temporal --yuv444 > /dev/null", "1"),
        ("fca 16-bit two-band + CAS + dehalo — scalar", 400,
         "{pre}-f rawvideo -pix_fmt yuv444p - | {FV} 854 480 xbr --yuv444 --16bit --sharpen --dehalo > /dev/null", "0"),
        ("fca 16-bit two-band + CAS + dehalo — AVX2", 400,
         "{pre}-f rawvideo -pix_fmt yuv444p - | {FV} 854 480 xbr --yuv444 --16bit --sharpen --dehalo > /dev/null", "1"),
        ("fca --hq --4x  -> 3416x1920 (AVX2)", 150,
         "{pre}-f rawvideo -pix_fmt yuv444p16le - | {FV} 854 480 xbr --hq --4x > /dev/null", "1"),
    ]
    probes = []
    for name, n, tmpl, simd in rows:
        pre = f"ffmpeg -v error -i {video} -frames:v {n} "
        cmd = tmpl.format(pre=pre, FV=FCA_VIDEO)
        env = dict(os.environ)
        if simd is not None:
            env["FCA_SIMD"] = simd
        times = []
        for i in range(3 if n >= 400 else 2):
            t0 = time.perf_counter()
            r = subprocess.run(["bash", "-c", cmd], env=env)
            if r.returncode != 0:
                raise RuntimeError(f"bench failed: {name}")
            times.append(time.perf_counter() - t0)
        best = min(times)
        fps = n / best
        tag = "" if simd is None else ("  AVX2" if simd == "1" else "  scalar")
        print(f"{name:46s} {best:7.2f}s  {fps:8.1f} fps  ({fps/24:.2f}x rt){tag}")
        probes.append((name, best, fps))
    return probes


def chart(probes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

    def color(name):
        if "baseline" in name:
            return "#64748b"
        if "lanczos" in name:
            return "#0ea5e9"
        if "scalar" in name:
            return "#f97316"
        if "--hq" in name:
            return "#eab308"
        return "#22c55e"

    names = [p[0] for p in probes]
    fps = [p[2] for p in probes]
    colors = [color(n) for n in names]
    fig, ax = plt.subplots(figsize=(10.5, 4.9), constrained_layout=True)
    bars = ax.barh(names[::-1], fps[::-1], color=colors[::-1])
    ax.axvline(24, color="#ef4444", ls="--", lw=1.4)
    ax.text(25.5, len(names) - 0.38, "realtime @ 24 fps", color="#ef4444", fontsize=9)
    for b, v in zip(bars, fps[::-1]):
        ax.text(b.get_width() * 1.05 + 1.2, b.get_y() + b.get_height() / 2, f"{v:.0f} fps",
                va="center", fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks([10, 30, 100, 300, 1000, 3000])
    ax.set_xticklabels(["10", "30", "100", "300", "1000", "3000"])
    ax.set_xlabel("854x480 source, frames per second, log scale (best of 3)")
    ax.set_title("libfca AVX2 vs scalar on real anime footage (Ergo Proxy ep.1)\n"
                 "end-to-end pipe: ffmpeg decode -> fca_video -> null")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(os.path.join(OUT, "simd_bench.png"), dpi=150)
    plt.close(fig)


def make_4x(video, tmp):
    n_hero, slug, _ = SCENES[0]  # dome_walk
    n_dark, slug2, _ = SCENES[2]  # staircase
    src = os.path.join(tmp, f"{slug}.png")
    extract_frame(video, n_hero, src)

    lan = os.path.join(tmp, "hero_lanczos4x.png")
    hq = os.path.join(tmp, "hero_hq4x.png")
    lanczos4x(src, lan)
    fca_run(src, hq, 854, 480, ["--hq", "--4x"])

    # 160x160 source-space boxes (equal size so the montage grid aligns):
    # the dark silhouette on the mound and the statue door lines
    box_a = (270, 80, 430, 240)
    box_b = (560, 140, 720, 300)
    zoom = 6  # 4x output shown at 1.5x more

    # full frames (shown 1:2.67)
    panels = [
        (Image.open(src).convert("RGB").resize((1280, 720), Image.Resampling.NEAREST),
         "source 854x480 (nearest blowup)"),
        (Image.open(lan).convert("RGB").resize((1280, 720), Image.Resampling.LANCZOS),
         "lanczos x4 (3416x1920, shown 1:2.7)"),
        (Image.open(hq).convert("RGB").resize((1280, 720), Image.Resampling.LANCZOS),
         "fca --hq --4x (two-band + CAS x2 + dehalo + line dark)"),
    ]
    montage(panels, 1).save(os.path.join(OUT, "real_4x_full.png"))

    # zoom crops at native 4x pixels; the source is blown up 4x (nearest) first
    src4 = Image.open(src).convert("RGB").resize((3416, 1920), Image.Resampling.NEAREST)
    lan4 = Image.open(lan).convert("RGB")
    hq4 = Image.open(hq).convert("RGB")
    side = (box_a[2] - box_a[0]) * zoom
    zp = [
        (cropN(src4, box_a, 4).resize((side, side), Image.Resampling.NEAREST),
         "source x6 (nearest) — silhouette"),
        (cropN(lan4, box_a, 4).resize((side, side), Image.Resampling.NEAREST),
         "lanczos x4, x6 detail — silhouette"),
        (cropN(hq4, box_a, 4).resize((side, side), Image.Resampling.NEAREST),
         "fca --hq --4x, x6 detail — silhouette"),
        (cropN(src4, box_b, 4).resize((side, side), Image.Resampling.NEAREST),
         "source x6 (nearest) — statue door"),
        (cropN(lan4, box_b, 4).resize((side, side), Image.Resampling.NEAREST),
         "lanczos x4, x6 detail — statue door"),
        (cropN(hq4, box_b, 4).resize((side, side), Image.Resampling.NEAREST),
         "fca --hq --4x, x6 detail — statue door"),
    ]
    montage(zp, 3).save(os.path.join(OUT, "real_4x_zoom.png"))


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--outdir", default=OUT)
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()
    OUT = args.outdir
    os.makedirs(OUT, exist_ok=True)
    assert os.path.exists(FCA_VIDEO), "build first: cargo build --release"

    print("== 4x comparisons ==")
    with tempfile.TemporaryDirectory(prefix="fca_simd_") as tmp:
        make_4x(args.video, tmp)
    if not args.no_bench:
        print("== benchmark ==")
        probes = bench(args.video)
        chart(probes)
    for f in sorted(os.listdir(OUT)):
        if f.startswith(("simd_", "real_4x")):
            print(f"{f}: {os.path.getsize(os.path.join(OUT, f)) // 1024} KB")


if __name__ == "__main__":
    main()
