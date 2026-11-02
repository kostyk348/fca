//! Benchmarks for the cellular upscale rules (criterion).
//!
//! Run with `cargo bench`. The plane is a deterministic pseudo-random 640x360
//! image — worst case for the tile cache (everything changes every frame),
//! which is exactly what a single-frame benchmark should measure.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fca::rules;

fn bench_plane() -> fca::Grid {
    let (w, h) = (640usize, 360usize);
    let mut s: u32 = 0x1234_5678;
    let data = (0..w * h)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s & 0xFF) as u8
        })
        .collect();
    fca::Grid::from_vec(w, h, data)
}

fn bench_rules(c: &mut Criterion) {
    let g = bench_plane();
    let mut group = c.benchmark_group("upscale 640x360 -> 1280x720 (per plane)");
    group.throughput(criterion::Throughput::Elements((g.w * g.h) as u64));

    group.bench_function("scale2x", |b| {
        b.iter(|| rules::scale2x_scalar(black_box(&g)))
    });
    group.bench_function("fuzzy", |b| b.iter(|| rules::scale2x_fuzzy(black_box(&g))));
    group.bench_function("xbr", |b| b.iter(|| rules::scale2x_xbr(black_box(&g))));
    group.finish();
}

criterion_group!(benches, bench_rules);
criterion_main!(benches);
