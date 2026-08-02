// fca_video — rawvideo pipe CLI: 2x/4x upscale + optional denoise/temporal
// + "rich look" post FX (sharpen / contrast / vibrance / deband).
//
// Usage:
//   fca_video <W> <H> <scale2x|fuzzy|xbr|temporal> [flags]
//     flags: --denoise  median 3x3 pre-filter
//            --tile N   temporal tile size (default 32)
//            --rule xbr temporal uses the xbr rule (default fuzzy)
//            --yuv444   yuv444p in/out (color pipeline; chroma = bicubic 2x)
//            --4x       double pass -> 4x (Y: rule twice, chroma: bicubic x2)
//            --sharpen  CAS contrast-adaptive sharpen on Y
//            --contrast N  luma contrast 0..255
//            --vibrance N  chroma saturation gain 0..255
//            --deband   gradient dither on Y
//
// Gray mode: reads W*H gray rawvideo, writes 2W*2H (or 4W*4H) gray.
// Color mode: reads Y+U+V (each W*H), writes 2W*2H (or 4W*4H) yuv444p.
// In ffmpeg pipelines:
//   ffmpeg -i in.mp4 -f rawvideo -pix_fmt gray - | fca_video 640 360 fuzzy |
//   ffmpeg -f rawvideo -pix_fmt gray -s 1280x720 -i - out.mp4
//   ffmpeg -i in.mp4 -f rawvideo -pix_fmt yuv444p - | fca_video 640 360 xbr --4x --sharpen --vibrance 40 |
//   ffmpeg -f rawvideo -pix_fmt yuv444p -s 2560x1440 -i - out.mp4

#include <chrono>
#include <cstdio>
#include <cstring> // must precede <cstdlib> (GCC16+glibc include-order constraint)
#include <cstdlib>
#include <string>
#include <vector>

#include "fca/denoise.hpp"
#include "fca/grid.hpp"
#include "fca/postfx.hpp"
#include "fca/rules.hpp"
#include "fca/temporal.hpp"

namespace {

int usage() {
    fprintf(stderr,
            "usage: fca_video <W> <H> <scale2x|fuzzy|xbr|temporal> [flags]\n"
            "  gray:   reads W*H gray rawvideo, writes 2W*2H gray rawvideo\n"
            "  color:  --yuv444 -> reads/writes yuv444p (Y,U,V planes)\n"
            "  flags:  --denoise --tile N --rule xbr --yuv444 --4x\n"
            "          --sharpen --contrast N --vibrance N --deband\n");
    return 1;
}

// dispatch: scale one plane (WxH) into out (2Wx2H) using the rule
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

    bool do_denoise = false, yuv444 = false, do_4x = false;
    bool do_sharpen = false, do_deband = false;
    int contrast = 0, vibrance = 0;
    std::uint32_t tile = 32;
    bool temporal_xbr = false;
    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--denoise") do_denoise = true;
        else if (a == "--yuv444") yuv444 = true;
        else if (a == "--4x") do_4x = true;
        else if (a == "--sharpen") do_sharpen = true;
        else if (a == "--deband") do_deband = true;
        else if (a == "--contrast" && i + 1 < argc) contrast = std::atoi(argv[++i]);
        else if (a == "--vibrance" && i + 1 < argc) vibrance = std::atoi(argv[++i]);
        else if (a == "--rule" && i + 1 < argc) temporal_xbr = (std::string(argv[++i]) == "xbr");
        else if (a == "--tile" && i + 1 < argc) tile = (std::uint32_t)std::atoi(argv[++i]);
    }
    if (tile == 0) tile = 32;
    if (contrast < 0) contrast = 0;
    if (vibrance < 0) vibrance = 0;

    const std::size_t plane_sz = (std::size_t)W * H;
    const std::size_t frame_sz = yuv444 ? 3 * plane_sz : plane_sz;
    const std::size_t scale = do_4x ? 4 : 2;
    const std::size_t out_sz = scale * scale * plane_sz; // per-plane output size

    std::vector<uint8_t> frame(frame_sz), tmp_den(plane_sz);
    std::vector<uint8_t> Y(plane_sz), U(plane_sz), V(plane_sz);
    std::vector<uint8_t> out_Y(out_sz), out_U(out_sz), out_V(out_sz);
    std::vector<uint8_t> mid_Y(4 * plane_sz), mid_U(4 * plane_sz), mid_V(4 * plane_sz);
    std::vector<uint8_t> out(yuv444 ? 3 * out_sz : out_sz);

    fca::temporal::TemporalUpscaler tu((std::uint32_t)W, (std::uint32_t)H, tile,
                                       temporal_xbr ? fca::temporal::TemporalUpscaler::Rule::Xbr
                                                    : fca::temporal::TemporalUpscaler::Rule::Fuzzy);

    // reusable grids
    fca::Grid g((std::uint32_t)W, (std::uint32_t)H), o, o2;

    auto t0 = std::chrono::steady_clock::now();
    std::uint64_t frames = 0;

    while (std::fread(frame.data(), 1, frame_sz, stdin) == frame_sz) {
        const uint8_t* src = frame.data();
        if (yuv444) {
            std::memcpy(Y.data(), src, plane_sz);
            std::memcpy(U.data(), src + plane_sz, plane_sz);
            std::memcpy(V.data(), src + 2 * plane_sz, plane_sz);
            src = Y.data();
        }

        // ---- luma: rule upscale (2x or 4x) ----
        const uint8_t* y_in = src;
        if (do_denoise) {
            g.data.assign(src, src + plane_sz);
            fca::denoise::median3x3(g, o);
            tmp_den = o.data;
            y_in = tmp_den.data();
        }

        uint8_t* y_out;
        if (do_4x) {
            if (mode == "temporal") tu.process(y_in, mid_Y.data());
            else { g.data.assign(y_in, y_in + plane_sz); upscale(g, o, mode.c_str()); std::memcpy(mid_Y.data(), o.data.data(), 4 * plane_sz); }
            fca::Grid g2((std::uint32_t)(2 * W), (std::uint32_t)(2 * H));
            g2.data = mid_Y;
            upscale(g2, o2, mode.c_str());
            std::memcpy(out_Y.data(), o2.data.data(), out_sz);
            y_out = out_Y.data();
        } else {
            if (mode == "temporal") tu.process(y_in, out_Y.data());
            else { g.data.assign(y_in, y_in + plane_sz); upscale(g, o, mode.c_str()); std::memcpy(out_Y.data(), o.data.data(), out_sz); }
            y_out = out_Y.data();
        }

        // ---- chroma (color only): bicubic 2x / 4x ----
        if (yuv444) {
            if (do_4x) {
                fca::postfx::bicubic2x(U.data(), W, H, mid_U.data());
                fca::postfx::bicubic2x(mid_U.data(), 2 * W, 2 * H, out_U.data());
                fca::postfx::bicubic2x(V.data(), W, H, mid_V.data());
                fca::postfx::bicubic2x(mid_V.data(), 2 * W, 2 * H, out_V.data());
            } else {
                fca::postfx::bicubic2x(U.data(), W, H, out_U.data());
                fca::postfx::bicubic2x(V.data(), W, H, out_V.data());
            }
            if (vibrance > 0)
                fca::postfx::vibrance(out_U.data(), out_V.data(), (int)(scale * W), (int)(scale * H), vibrance);
        }

        // ---- post FX on luma ----
        if (do_sharpen) fca::postfx::cas_sharpen(y_out, (int)(scale * W), (int)(scale * H), 255);
        if (contrast > 0) fca::postfx::contrast_s(y_out, (int)(scale * W), (int)(scale * H), contrast);
        if (do_deband) fca::postfx::deband(y_out, (int)(scale * W), (int)(scale * H), (unsigned)frames);

        if (yuv444) {
            std::memcpy(out.data(), y_out, out_sz);
            std::memcpy(out.data() + out_sz, out_U.data(), out_sz);
            std::memcpy(out.data() + 2 * out_sz, out_V.data(), out_sz);
        } else {
            std::memcpy(out.data(), y_out, out_sz);
        }

        std::fwrite(out.data(), 1, out.size(), stdout);
        ++frames;
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
