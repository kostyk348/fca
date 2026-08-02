// fca_upscale — CLI: PPM in -> scale2x -> PPM out (per channel).
// Usage:
//   fca_upscale in.ppm out.ppm            (AVX2 if compiled, else scalar)
//   fca_upscale in.ppm out.ppm --scalar
//   fca_upscale in.ppm out.ppm --check    (compare scalar vs AVX2 outputs)

#include <chrono>
#include <cstdio>
#include <cstring> // must precede <cstdlib> (GCC16+glibc include-order constraint)
#include <cstdlib>
#include <string>
#include <vector>

#include "fca/rules.hpp"

namespace {

struct Image {
    int w = 0, h = 0;
    std::vector<std::vector<uint8_t>> planes; // per-channel 8-bit planes
};

bool read_ppm(const char* path, Image& img) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return false; }
    char magic[3] = {};
    if (fscanf(f, "%2s", magic) != 1 || magic[0] != 'P' || magic[1] != '6') {
        fprintf(stderr, "not a PPM P6: %s\n", path); fclose(f); return false;
    }
    // skip comments
    auto skip_ws = [&]() {
        int c;
        do {
            c = fgetc(f);
            if (c == '#') while ((c = fgetc(f)) != '\n' && c != EOF) {}
        } while (c == ' ' || c == '\t' || c == '\n' || c == '\r');
        if (c != EOF) ungetc(c, f);
    };
    skip_ws();
    if (fscanf(f, "%d", &img.w) != 1) { fclose(f); return false; }
    skip_ws();
    if (fscanf(f, "%d", &img.h) != 1) { fclose(f); return false; }
    skip_ws();
    int maxv = 0;
    if (fscanf(f, "%d", &maxv) != 1 || maxv != 255) { fclose(f); return false; }
    fgetc(f); // single whitespace after maxval
    img.planes.assign(3, std::vector<uint8_t>((size_t)img.w * img.h));
    size_t n = (size_t)img.w * img.h;
    std::vector<uint8_t> buf(n * 3);
    if (fread(buf.data(), 1, n * 3, f) != n * 3) { fclose(f); return false; }
    fclose(f);
    for (size_t i = 0; i < n; ++i)
        for (int c = 0; c < 3; ++c)
            img.planes[c][i] = buf[i * 3 + c];
    return true;
}

bool write_ppm(const char* path, const Image& img) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    fprintf(f, "P6\n%d %d\n255\n", img.w, img.h);
    size_t n = (size_t)img.w * img.h;
    std::vector<uint8_t> buf(n * 3);
    for (size_t i = 0; i < n; ++i)
        for (int c = 0; c < 3; ++c)
            buf[i * 3 + c] = img.planes[c][i];
    fwrite(buf.data(), 1, n * 3, f);
    fclose(f);
    return true;
}

double bench_scale2x(const fca::Grid& in, const char* which, fca::Grid& out) {
    auto t0 = std::chrono::steady_clock::now();
    const int reps = 3;
    for (int r = 0; r < reps; ++r) {
        if (std::strcmp(which, "scalar") == 0)
            fca::rule::scale2x_scalar(in, out);
        else if (std::strcmp(which, "fuzzy") == 0)
            fca::rule::scale2x_fuzzy(in, out);
        else if (std::strcmp(which, "fuzzy_avx2") == 0)
            fca::rule::avx2::scale2x_fuzzy_avx2(in, out);
#if defined(__AVX2__)
        else
            fca::rule::avx2::scale2x_avx2(in, out);
#endif
    }
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
}

} // namespace

int main(int argc, char** argv) {
    bool use_scalar = false, do_check = false, use_fuzzy = false;
    std::string in_path, out_path;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--scalar") use_scalar = true;
        else if (a == "--check") do_check = true;
        else if (a == "--fuzzy") use_fuzzy = true;
        else if (in_path.empty()) in_path = a;
        else if (out_path.empty()) out_path = a;
    }
    if (in_path.empty() || out_path.empty()) {
        fprintf(stderr, "usage: fca_upscale in.ppm out.ppm [--scalar|--check|--fuzzy]\n");
        return 1;
    }

    Image img;
    if (!read_ppm(in_path.c_str(), img)) return 1;

    printf("input : %dx%d\n", img.w, img.h);

    Image out;
    out.w = img.w * 2;
    out.h = img.h * 2;
    out.planes.assign(3, std::vector<uint8_t>((size_t)out.w * out.h));

    for (int c = 0; c < 3; ++c) {
        fca::Grid in_g(img.w, img.h);
        in_g.data.swap(img.planes[c]);
        fca::Grid out_g;
        if (use_fuzzy) {
#if defined(__AVX2__)
            if (use_scalar) {
                fca::rule::scale2x_fuzzy(in_g, out_g);
            } else {
                fca::rule::avx2::scale2x_fuzzy_avx2(in_g, out_g);
            }
            if (do_check) {
                fca::Grid ref;
                fca::rule::scale2x_fuzzy(in_g, ref);
                size_t bad = 0;
                for (size_t i = 0; i < out_g.data.size(); ++i)
                    if (ref.data[i] != out_g.data[i]) ++bad;
                printf("plane %d: fuzzy AVX2 vs fuzzy scalar mismatch bytes: %zu / %zu\n",
                       c, bad, out_g.data.size());
            }
#else
            (void)use_scalar;
            fca::rule::scale2x_fuzzy(in_g, out_g);
#endif
        } else {
#if defined(__AVX2__)
            if (use_scalar) {
                fca::rule::scale2x_scalar(in_g, out_g);
            } else {
                fca::rule::avx2::scale2x_avx2(in_g, out_g);
            }
            if (do_check) {
                fca::Grid ref;
                fca::rule::scale2x_scalar(in_g, ref);
                size_t bad = 0;
                for (size_t i = 0; i < out_g.data.size(); ++i)
                    if (ref.data[i] != out_g.data[i]) ++bad;
                printf("plane %d: AVX2 vs scalar mismatch bytes: %zu / %zu\n",
                       c, bad, out_g.data.size());
            }
#else
            (void)use_scalar;
            fca::rule::scale2x_scalar(in_g, out_g);
#endif
        }
        out.planes[c].swap(out_g.data);
        img.planes[c].swap(in_g.data);
    }

    if (!write_ppm(out_path.c_str(), out)) { fprintf(stderr, "cannot write %s\n", out_path.c_str()); return 1; }
    printf("output: %dx%d -> %s\n", out.w, out.h, out_path.c_str());

    // bench on plane 0
    {
        fca::Grid in_g(img.w, img.h);
        in_g.data = img.planes[0];
        fca::Grid out_g;
        double t_scalar = bench_scale2x(in_g, "scalar", out_g);
        printf("scalar: %6.2f ms\n", t_scalar);
        double t_fuzzy = bench_scale2x(in_g, "fuzzy", out_g);
        printf("fuzzy : %6.2f ms   (x%.2f vs scalar)\n", t_fuzzy, t_scalar / t_fuzzy);
#if defined(__AVX2__)
        double t_avx2 = bench_scale2x(in_g, "avx2", out_g);
        printf("avx2  : %6.2f ms   (x%.2f vs scalar)\n", t_avx2, t_scalar / t_avx2);
        double t_favx2 = bench_scale2x(in_g, "fuzzy_avx2", out_g);
        printf("fuzzy+avx2: %4.2f ms   (x%.2f vs fuzzy)\n", t_favx2, t_fuzzy / t_favx2);
#endif
    }
    return 0;
}
