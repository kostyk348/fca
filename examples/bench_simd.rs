use fca::grid::Grid;
use fca::rules;
use std::time::Instant;

fn bench(name: &str, f: impl Fn(&Grid) -> Grid, g: &Grid, n: usize) -> f64 {
    let _ = f(g); // warmup
    let t = Instant::now();
    let mut acc = 0u8;
    for _ in 0..n {
        acc = acc.wrapping_add(f(g).data[0]);
    }
    let dt = t.elapsed().as_secs_f64() / n as f64;
    println!("{name:28} {:8.2} ms/frame  (acc={acc})", dt * 1000.0);
    dt
}

fn main() {
    let (w, h) = (854, 480);
    let g = Grid::from_vec(w, h, (0..w * h).map(|i| (i * 37 % 251) as u8).collect());
    println!("frame {w}x{h} -> {}x{}", 2 * w, 2 * h);
    let auto = rules::scale2x_fuzzy(&g);
    let scalar = rules::fuzzy_scalar_ref(&g);
    assert_eq!(auto.data, scalar.data, "bit-exactness broken!");
    println!("bit-exact: OK");
    let n = 20;
    bench("fuzzy AVX2 (auto)", rules::scale2x_fuzzy, &g, n);
    bench("fuzzy scalar (ref)", rules::fuzzy_scalar_ref, &g, n);
    let auto = rules::scale2x_scalar(&g);
    let scalar = rules::epx_scalar_ref(&g);
    assert_eq!(auto.data, scalar.data, "epx bit-exactness broken!");
    println!("epx bit-exact: OK");
    bench("epx AVX2 (auto)", rules::scale2x_scalar, &g, n);
    bench("epx scalar (ref)", rules::epx_scalar_ref, &g, n);
}
