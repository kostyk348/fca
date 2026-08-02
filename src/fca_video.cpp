// fca_video — rawvideo pipe CLI: 2x/4x upscale + optional denoise/temporal
// + "rich look" post FX (sharpen / contrast / vibrance / deband)
// + 16-bit two-band upscale (hi bits via rule, lo bits via bicubic).
//
// Usage:
//   fca_video <W> <H> <scale2x|fuzzy|xbr|temporal> [flags]
//     flags: --denoise  median 3x3 pre-filter
//            --tile N   temporal tile size (default 32)
//            --rule xbr temporal uses the xbr rule (default fuzzy)
//            --yuv444   yuv444p in/out (color pipeline; chroma = bicubic 2x)
//            --16bit    yuv444p16le in/out (two-band: hi->rule, lo->bicubic)
//            --4x       double pass -> 4x
//            --sharpen  CAS contrast-adaptive sharpen on Y (hi band)
//            --contrast N  luma contrast 0..255
//            --vibrance N  chroma saturation gain 0..255
//            --deband   gradient dither (8-bit gray mode only; 16-bit is band-free)
//
// Gray 8-bit:  reads W*H gray, writes 2W*2H (or 4W*4H).
// Color 8-bit: --yuv444 reads Y+U+V (each W*H), writes yuv444p at 2x/4x.
// Color 16-bit: --16bit reads/writes yuv444p16le, two-band upscale:
//   hi = v>>8  -> cellular rule (edges stay sharp)
//   lo = v&255 -> bicubic (gradients stay smooth, banding disappears)
//   out = mix(bicubic, rule, w) where w grows with local 3x3 contrast:
//   rule only where edges exist, bicubic on flat/gradient areas.

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
            "  gray:     reads W*H gray rawvideo, writes 2W*2H gray rawvideo\n"
            "  color:    --yuv444 -> reads/writes yuv444p (Y,U,V planes)\n"
            "  16-bit:   --16bit  -> reads/writes yuv444p16le (two-band upscale)\n"
            "  flags:  --denoise --tile N --rule xbr --yuv444 --16bit --4x\n"
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

void rule2x(const uint8_t* src, int w, int h, uint8_t* dst, const char* mode) {
    fca::Grid g((std::uint32_t)w, (std::uint32_t)h), o;
    g.data.assign(src, src + (size_t)w * h);
    upscale(g, o, mode);
    std::memcpy(dst, o.data.data(), (size_t)4 * w * h);
}

// adaptive per-pixel mix: w = smoothstep(24, 96, range3x3 on input hi)
// out = bc + (rl - bc) * w/255   (bc/rl/out are 2w*2h, hi is w*h)
void adaptive_mix(const uint8_t* hi, const uint8_t* bc, const uint8_t* rl,
                  int w, int h, uint8_t* out) {
    const int W2 = 2 * w;
    for (int y = 0; y < h; ++y) {
        const uint8_t* row = hi + (size_t)y * w;
        for (int x = 0; x < w; ++x) {
            int xl = x == 0 ? 0 : x - 1;
            int xr = x + 1 < w ? x + 1 : w - 1;
            int mn = row[xl], mx = row[xl];
            for (int yy = -1; yy <= 1; ++yy) {
                int yy2 = y + yy < 0 ? 0 : (y + yy >= h ? h - 1 : y + yy);
                const uint8_t* r = hi + (size_t)yy2 * w;
                int v0 = r[xl], v1 = r[x], v2 = r[xr];
                if (v0 < mn) mn = v0;
                if (v0 > mx) mx = v0;
                if (v1 < mn) mn = v1;
                if (v1 > mx) mx = v1;
                if (v2 < mn) mn = v2;
                if (v2 > mx) mx = v2;
            }
            int range = mx - mn;
            int w8 = range <= 24 ? 0 : (range >= 96 ? 255 : (range - 24) * 255 / 72);
            int ws = (w8 * w8) >> 8;                       // smoothstep
            int wf = (ws * (768 - 2 * w8)) >> 8;           // 0..255
            const uint8_t* b = bc + (size_t)(2 * y) * W2 + 2 * x;
            const uint8_t* r = rl + (size_t)(2 * y) * W2 + 2 * x;
            uint8_t* o = out + (size_t)(2 * y) * W2 + 2 * x;
            for (int dy = 0; dy < 2; ++dy) {
                for (int dx = 0; dx < 2; ++dx) {
                    int i = dy * W2 + dx;
                    int v = b[i] + (((int)r[i] - (int)b[i]) * wf >> 8);
                    o[i] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
                }
            }
        }
    }
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

    bool do_denoise = false, yuv444 = false, bit16 = false, do_4x = false;
    bool do_sharpen = false, do_deband = false;
    int contrast = 0, vibrance = 0;
    std::uint32_t tile = 32;
    bool temporal_xbr = false;
    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--denoise") do_denoise = true;
        else if (a == "--yuv444") yuv444 = true;
        else if (a == "--16bit") bit16 = true;
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
    if (bit16) yuv444 = true; // 16-bit implies color

    const std::size_t plane_sz = (std::size_t)W * H;
    const std::size_t bpp = bit16 ? 2 : 1;
    const std::size_t frame_sz = (yuv444 ? 3 : 1) * plane_sz * bpp;
    const std::size_t scale = do_4x ? 4 : 2;
    const std::size_t out_sz = scale * scale * plane_sz; // per-plane output size (8-bit units)

    // ---- buffers ----
    std::vector<uint8_t> frame(frame_sz);
    std::vector<uint16_t> Y16(plane_sz), U16(plane_sz), V16(plane_sz);
    std::vector<uint8_t> Y(plane_sz), U(plane_sz), V(plane_sz);         // hi bands
    std::vector<uint8_t> YL(plane_sz), UL(plane_sz), VL(plane_sz);      // lo bands
    std::vector<uint8_t> out_Y(out_sz), out_U(out_sz), out_V(out_sz);   // hi out
    std::vector<uint8_t> out_YL(out_sz), out_UL(out_sz), out_VL(out_sz);// lo out
    std::vector<uint8_t> mid_Y(4 * plane_sz), mid_YL(4 * plane_sz);
    std::vector<uint8_t> bc_Y(out_sz), rl_Y(out_sz);                    // bicubic/rule hi (2x or 4x)
    std::vector<uint8_t> tmp_den(plane_sz);
    std::vector<uint8_t> out(bit16 ? 6 * out_sz : (yuv444 ? 3 * out_sz : out_sz));

    fca::temporal::TemporalUpscaler tu((std::uint32_t)W, (std::uint32_t)H, tile,
                                       temporal_xbr ? fca::temporal::TemporalUpscaler::Rule::Xbr
                                                    : fca::temporal::TemporalUpscaler::Rule::Fuzzy);
    fca::Grid g((std::uint32_t)W, (std::uint32_t)H), o;

    auto t0 = std::chrono::steady_clock::now();
    std::uint64_t frames = 0;

    while (std::fread(frame.data(), 1, frame_sz, stdin) == frame_sz) {
        const uint8_t* src = frame.data();
        if (bit16) {
            // split planes into hi/lo 8-bit bands
            for (size_t i = 0; i < plane_sz; ++i) {
                uint16_t y = (uint16_t)(src[2 * i]) | ((uint16_t)src[2 * i + 1] << 8);
                uint16_t u = (uint16_t)(src[2 * (plane_sz + i)]) | ((uint16_t)src[2 * (plane_sz + i) + 1] << 8);
                uint16_t v = (uint16_t)(src[2 * (2 * plane_sz + i)]) | ((uint16_t)src[2 * (2 * plane_sz + i) + 1] << 8);
                Y16[i] = y; U16[i] = u; V16[i] = v;
                Y[i] = (uint8_t)(y >> 8); YL[i] = (uint8_t)(y & 0xFF);
                U[i] = (uint8_t)(u >> 8); UL[i] = (uint8_t)(u & 0xFF);
                V[i] = (uint8_t)(v >> 8); VL[i] = (uint8_t)(v & 0xFF);
            }
        } else if (yuv444) {
            std::memcpy(Y.data(), src, plane_sz);
            std::memcpy(U.data(), src + plane_sz, plane_sz);
            std::memcpy(V.data(), src + 2 * plane_sz, plane_sz);
        }

        const uint8_t* y_in = bit16 ? Y.data() : (yuv444 ? Y.data() : frame.data());
        if (do_denoise) {
            g.data.assign(y_in, y_in + plane_sz);
            fca::denoise::median3x3(g, o);
            tmp_den = o.data;
            y_in = tmp_den.data();
        }

        // ================= luma upscale =================
        if (bit16) {
            // two-band: hi -> adaptive mix(rule, bicubic), lo -> bicubic
            if (do_4x) {
                fca::postfx::bicubic2x(y_in, W, H, bc_Y.data());
                rule2x(y_in, W, H, rl_Y.data(), mode.c_str());
                adaptive_mix(y_in, bc_Y.data(), rl_Y.data(), W, H, mid_Y.data());
                fca::postfx::bicubic2x(YL.data(), W, H, mid_YL.data());
                // second pass: two-band again on the 2x result
                fca::postfx::bicubic2x(mid_Y.data(), 2 * W, 2 * H, bc_Y.data());
                rule2x(mid_Y.data(), 2 * W, 2 * H, rl_Y.data(), mode.c_str());
                adaptive_mix(mid_Y.data(), bc_Y.data(), rl_Y.data(), 2 * W, 2 * H, out_Y.data());
                fca::postfx::bicubic2x(mid_YL.data(), 2 * W, 2 * H, out_YL.data());
            } else {
                fca::postfx::bicubic2x(y_in, W, H, bc_Y.data());
                rule2x(y_in, W, H, rl_Y.data(), mode.c_str());
                adaptive_mix(y_in, bc_Y.data(), rl_Y.data(), W, H, out_Y.data());
                fca::postfx::bicubic2x(YL.data(), W, H, out_YL.data());
            }
            // FX on hi band
            if (do_sharpen) fca::postfx::cas_sharpen(out_Y.data(), (int)(scale * W), (int)(scale * H), 255);
            if (contrast > 0) fca::postfx::contrast_s(out_Y.data(), (int)(scale * W), (int)(scale * H), contrast);
        } else {
            // 8-bit: rule-only (or temporal)
            uint8_t* y_out;
            if (do_4x) {
                if (mode == "temporal") tu.process(y_in, mid_Y.data());
                else rule2x(y_in, W, H, mid_Y.data(), mode.c_str());
                fca::Grid g2((std::uint32_t)(2 * W), (std::uint32_t)(2 * H));
                g2.data = mid_Y;
                upscale(g2, o, mode.c_str());
                std::memcpy(out_Y.data(), o.data.data(), out_sz);
                y_out = out_Y.data();
            } else {
                if (mode == "temporal") tu.process(y_in, out_Y.data());
                else rule2x(y_in, W, H, out_Y.data(), mode.c_str());
                y_out = out_Y.data();
            }
            if (do_sharpen) fca::postfx::cas_sharpen(y_out, (int)(scale * W), (int)(scale * H), 255);
            if (contrast > 0) fca::postfx::contrast_s(y_out, (int)(scale * W), (int)(scale * H), contrast);
            if (do_deband) fca::postfx::deband(y_out, (int)(scale * W), (int)(scale * H), (unsigned)frames);
        }

        // ================= chroma =================
        if (yuv444) {
            if (bit16) {
                // chroma: pure bicubic on hi and lo bands
                const uint8_t* uh = U.data();
                const uint8_t* ul = UL.data();
                const uint8_t* vh = V.data();
                const uint8_t* vl = VL.data();
                uint8_t* out_uh = out_U.data();
                uint8_t* out_ul = out_UL.data();
                uint8_t* out_vh = out_V.data();
                uint8_t* out_vl = out_VL.data();
                if (do_4x) {
                    std::vector<uint8_t> tU(4 * plane_sz), tUl(4 * plane_sz), tV(4 * plane_sz), tVl(4 * plane_sz);
                    fca::postfx::bicubic2x(uh, W, H, tU.data());
                    fca::postfx::bicubic2x(tU.data(), 2 * W, 2 * H, out_uh);
                    fca::postfx::bicubic2x(ul, W, H, tUl.data());
                    fca::postfx::bicubic2x(tUl.data(), 2 * W, 2 * H, out_ul);
                    fca::postfx::bicubic2x(vh, W, H, tV.data());
                    fca::postfx::bicubic2x(tV.data(), 2 * W, 2 * H, out_vh);
                    fca::postfx::bicubic2x(vl, W, H, tVl.data());
                    fca::postfx::bicubic2x(tVl.data(), 2 * W, 2 * H, out_vl);
                } else {
                    fca::postfx::bicubic2x(uh, W, H, out_uh);
                    fca::postfx::bicubic2x(ul, W, H, out_ul);
                    fca::postfx::bicubic2x(vh, W, H, out_vh);
                    fca::postfx::bicubic2x(vl, W, H, out_vl);
                }
            } else {
                const uint8_t* uh = U.data();
                const uint8_t* vh = V.data();
                if (do_4x) {
                    std::vector<uint8_t> tU(4 * plane_sz), tV(4 * plane_sz);
                    fca::postfx::bicubic2x(uh, W, H, tU.data());
                    fca::postfx::bicubic2x(tU.data(), 2 * W, 2 * H, out_U.data());
                    fca::postfx::bicubic2x(vh, W, H, tV.data());
                    fca::postfx::bicubic2x(tV.data(), 2 * W, 2 * H, out_V.data());
                } else {
                    fca::postfx::bicubic2x(uh, W, H, out_U.data());
                    fca::postfx::bicubic2x(vh, W, H, out_V.data());
                }
                if (vibrance > 0)
                    fca::postfx::vibrance(out_U.data(), out_V.data(), (int)(scale * W), (int)(scale * H), vibrance);
            }
        }

        // ================= write =================
        if (bit16) {
            // assemble yuv444p16le: out[2i] = lo, out[2i+1] = hi (little-endian)
            const size_t n = out_sz;
            uint8_t* op = out.data();
            auto put = [&](const uint8_t* hi8, const uint8_t* lo8) {
                for (size_t i = 0; i < n; ++i) { op[2 * i] = lo8[i]; op[2 * i + 1] = hi8[i]; }
            };
            put(out_Y.data(), out_YL.data()); op += 2 * n;
            put(out_U.data(), out_UL.data()); op += 2 * n;
            put(out_V.data(), out_VL.data());
            if (vibrance > 0) {
                // vibrance on assembled 16-bit chroma
                int G = 256 + ((vibrance * 100) >> 8);
                for (size_t i = 0; i < n; ++i) {
                    uint8_t* bu = out.data() + 2 * n + 2 * i;
                    uint8_t* bv = bu + 2 * n;
                    int ub = ((int)bu[0] | ((int)bu[1] << 8)) - 32768;
                    int vb = ((int)bv[0] | ((int)bv[1] << 8)) - 32768;
                    int un = 32768 + (ub * G >> 8);
                    int vn = 32768 + (vb * G >> 8);
                    if (un < 0) un = 0;
                    if (un > 65535) un = 65535;
                    if (vn < 0) vn = 0;
                    if (vn > 65535) vn = 65535;
                    bu[0] = (uint8_t)(un & 0xFF); bu[1] = (uint8_t)((un >> 8) & 0xFF);
                    bv[0] = (uint8_t)(vn & 0xFF); bv[1] = (uint8_t)((vn >> 8) & 0xFF);
                }
            }
        } else if (yuv444) {
            std::memcpy(out.data(), out_Y.data(), out_sz);
            std::memcpy(out.data() + out_sz, out_U.data(), out_sz);
            std::memcpy(out.data() + 2 * out_sz, out_V.data(), out_sz);
        } else {
            std::memcpy(out.data(), out_Y.data(), out_sz);
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
