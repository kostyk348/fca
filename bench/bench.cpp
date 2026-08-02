// bench — libfca scale2x: scalar vs AVX2 on synthetic + real-ish content.
// Self-contained: generates test grids, prints ms and speedup.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring> // must precede <cstdlib> (GCC16+glibc include-order constraint)
#include <cstdlib>
#include <random>

#include "fca/rules.hpp"

namespace {

fca::Grid make_synthetic(size_t w, size_t h) {
    fca::Grid g(w, h);
    std::mt19937 rng(42);
    // half structured (checkerboard + gradient), half noise
    for (size_t y = 0; y < h; ++y)
        for (size_t x = 0; x < w; ++x) {
            uint8_t v;
            if ((x + y) % 3 == 0)
                v = (uint8_t)((x * 31 + y * 17) & 0xFF);            // pattern
            else if ((x * y) % 5 == 0)
                v = (uint8_t)(x & 0xFF);                              // gradient
            else
                v = (uint8_t)rng();                                   // noise
            g.data[y * w + x] = v;
        }
    return g;
}

double time_it(const fca::Grid& in, void (*fn)(const fca::Grid&, fca::Grid&), fca::Grid& out) {
    const int reps = 5;
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) fn(in, out);
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
}

} // namespace

int main(int argc, char** argv) {
    size_t w = 1280, h = 720;
    if (argc >= 3) { w = (size_t)std::strtoull(argv[1], nullptr, 10); h = (size_t)std::strtoull(argv[2], nullptr, 10); }

    fca::Grid in = make_synthetic(w, h);
    fca::Grid out;

    double t_sc = time_it(in, fca::rule::scale2x_scalar, out);
    printf("grid  : %zux%zu -> %zux%zu\n", w, h, w * 2, h * 2);

#if defined(__AVX2__)
    fca::Grid ref;
    fca::rule::scale2x_scalar(in, ref);
    fca::Grid sim;
    fca::rule::avx2::scale2x_avx2(in, sim);
    size_t bad = 0;
    for (size_t i = 0; i < ref.data.size(); ++i)
        if (ref.data[i] != sim.data[i]) ++bad;

    double t_av = time_it(in, fca::rule::avx2::scale2x_avx2, sim);
    printf("scalar: %7.2f ms   (%.1f Mpx/s)\n", t_sc, (double)(w * h) / t_sc / 1000.0);
    printf("avx2  : %7.2f ms   (%.1f Mpx/s)\n", t_av, (double)(w * h) / t_av / 1000.0);
    printf("speedup: x%.2f\n", t_sc / t_av);
    printf("correctness: AVX2 == scalar bytes: %s (%zu mismatches)\n",
           bad == 0 ? "IDENTICAL" : "MISMATCH", bad);
#else
    printf("(no AVX2 build; scalar only)\n");
    printf("scalar: %7.2f ms\n", t_sc);
#endif
    return 0;
}
