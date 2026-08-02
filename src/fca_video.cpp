// fca_video — rawvideo gray pipe CLI: 2x upscale + optional denoise/temporal.
//
// Usage:
//   fca_video <W> <H> <mode> [--denoise] [--tile N]
//     mode: scale2x | fuzzy | temporal
//
// Reads w*h 8-bit gray rawvideo from stdin, writes 2w*2h rawvideo to stdout.
// In ffmpeg pipelines:
//   ffmpeg -i in.mp4 -f rawvideo -pix_fmt gray - | fca_video 640 360 fuzzy --denoise |
//   ffmpeg -f rawvideo -pix_fmt gray -s 1280x720 -i - out.mp4
//
// `temporal` mode reuses unchanged tiles (S3 sleep/wake) — stability on
// static backgrounds + near-zero cost there. `--denoise` applies a median
// 3x3 pre-filter (S7) before upscaling.

#include <chrono>
#include <cstdio>
#include <cstring> // must precede <cstdlib> (GCC16+glibc include-order constraint)
#include <cstdlib>
#include <string>
#include <vector>

#include "fca/denoise.hpp"
#include "fca/grid.hpp"
#include "fca/rules.hpp"
#include "fca/temporal.hpp"

namespace {

int usage() {
    fprintf(stderr,
            "usage: fca_video <W> <H> <scale2x|fuzzy|xbr|temporal> [--denoise] [--tile N]\n"
            "  reads  W*H gray rawvideo on stdin, writes 2W*2H gray rawvideo on stdout\n");
    return 1;
}

// dispatch: scale one gray frame (WxH) into out (2Wx2H) using AVX2 when available
void upscale(const fca::Grid& in, fca::Grid& out, const char* mode) {
    if (std::strcmp(mode, "xbr") == 0) { fca::rule::scale2x_xbr(in, out); return; }
#if defined(__AVX2__)
    if (std::strcmp(mode, "scale2x") == 0) fca::rule::avx2::scale2x_avx2(in, out);
    else fca::rule::avx2::scale2x_fuzzy_avx2(in, out);
#else
    (void)mode;
    if (std::strcmp(mode, "scale2x") == 0) fca::rule::scale2x_scalar(in, out);
    else fca::rule::scale2x_fuzzy(in, out);
#endif
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) return usage();

    int W = std::atoi(argv[1]);
    int H = std::atoi(argv[2]);
    std::string mode = argv[3];
    if (W <= 0 || H <= 0 ||
        (mode != "scale2x" && mode != "fuzzy" && mode != "temporal" && mode != "xbr"))
        return usage();

    bool do_denoise = false;
    std::uint32_t tile = 32;
    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--denoise") do_denoise = true;
        else if (a == "--tile" && i + 1 < argc) tile = (std::uint32_t)std::atoi(argv[++i]);
    }
    if (tile == 0) tile = 32;

    const std::size_t frame_sz = (std::size_t)W * H;
    const std::size_t out_sz = (std::size_t)4 * W * H;

    std::vector<uint8_t> frame(frame_sz), out(out_sz), den(frame_sz);
    std::vector<uint8_t> tmp_den(frame_sz);
    fca::temporal::TemporalUpscaler tu((std::uint32_t)W, (std::uint32_t)H, tile);

    // reusable grids (avoid realloc per frame)
    fca::Grid g((std::uint32_t)W, (std::uint32_t)H), o;

    auto t0 = std::chrono::steady_clock::now();
    std::uint64_t frames = 0;

    while (std::fread(frame.data(), 1, frame_sz, stdin) == frame_sz) {
        const std::uint8_t* src = frame.data();

        if (do_denoise) {
            g.data = frame;
            fca::denoise::median3x3(g, o);
            tmp_den = o.data;
            src = tmp_den.data();
        }

        if (mode == "temporal") {
            tu.process(src, out.data());
        } else {
            g.data.assign(src, src + frame_sz);
            upscale(g, o, mode.c_str());
            std::memcpy(out.data(), o.data.data(), out_sz);
        }

        std::fwrite(out.data(), 1, out_sz, stdout);
        ++frames;

        if (frames % 30 == 0) {
            const auto& st = tu.stats();
            if (st.total > 0) {
                double pct = 100.0 * st.recomputed / (double)st.total;
                fprintf(stderr, "[fca_video] frame %llu, temporal reuse: %.1f%% recomputed\n",
                        (unsigned long long)frames, pct);
            }
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const auto& st = tu.stats();
    fprintf(stderr,
            "[fca_video] frames=%llu  total=%.1f ms (%.2f ms/frame)  "
            "tiles: %llu total, %llu recomputed (%.1f%%), %llu reused\n",
            (unsigned long long)frames, ms,
            frames ? ms / frames : 0.0,
            (unsigned long long)st.total, (unsigned long long)st.recomputed,
            st.total ? 100.0 * st.recomputed / (double)st.total : 0.0,
            (unsigned long long)st.reused);
    return 0;
}
