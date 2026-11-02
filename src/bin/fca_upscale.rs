//! fca_upscale — CLI: PPM in -> cellular rule 2x -> PPM out (per channel).
//!
//! Usage:
//!   fca_upscale in.ppm out.ppm [--rule fuzzy|xbr|scale2x] [--scalar] [--bench]

use clap::Parser;
use fca::{ppm, rules};
use rayon::prelude::*;
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "fca_upscale",
    version,
    about = "libfca — fuzzy cellular-rule 2x upscaler (PPM in, PPM out)"
)]
struct Cli {
    /// Input PPM (P6, maxval 255)
    input: String,
    /// Output PPM (P6, 2x size)
    output: String,
    /// Cellular rule to use
    #[arg(long, value_enum, default_value = "fuzzy")]
    rule: RuleArg,
    /// Kept for CLI compatibility with the C tool (the scalar path is the only
    /// implementation in the Rust port — it is the bit-exact AVX2 reference)
    #[arg(long)]
    scalar: bool,
    /// Benchmark the rules on plane 0 and print timings
    #[arg(long)]
    bench: bool,
}

#[derive(clap::ValueEnum, Clone, Copy)]
enum RuleArg {
    Scale2x,
    Fuzzy,
    Xbr,
}

fn upscale_plane(input: &fca::Grid, rule: RuleArg) -> fca::Grid {
    match rule {
        RuleArg::Scale2x => rules::scale2x_scalar(input),
        RuleArg::Fuzzy => rules::scale2x_fuzzy(input),
        RuleArg::Xbr => rules::scale2x_xbr(input),
    }
}

fn bench(name: &str, input: &fca::Grid, rule: RuleArg, reps: usize) -> (String, f64, f64) {
    // warm + measure; returns (name, ms, speedup vs scale2x computed by caller)
    let t0 = Instant::now();
    for _ in 0..reps {
        let _ = upscale_plane(input, rule);
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;
    (name.to_string(), ms, ms)
}

fn main() {
    let cli = Cli::parse();
    let img = match ppm::read_ppm(&cli.input) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    println!("input : {}x{}", img.w, img.h);

    let n = img.w * img.h;
    let out: Vec<Vec<u8>> = img
        .planes
        .par_iter()
        .map(|plane| {
            let g = fca::Grid::from_vec(img.w, img.h, plane.clone());
            upscale_plane(&g, cli.rule).data
        })
        .collect();

    let out_img = ppm::Image {
        w: img.w * 2,
        h: img.h * 2,
        planes: [out[0].clone(), out[1].clone(), out[2].clone()],
    };
    let _ = n;
    if let Err(e) = ppm::write_ppm(&cli.output, &out_img) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
    println!("output: {}x{} -> {}", out_img.w, out_img.h, cli.output);

    if cli.bench {
        let g = fca::Grid::from_vec(img.w, img.h, img.planes[0].clone());
        let reps = 5;
        let (n0, t_scalar, _) = bench("scalar", &g, RuleArg::Scale2x, reps);
        let (n1, t_fuzzy, _) = bench("fuzzy", &g, RuleArg::Fuzzy, reps);
        let (n2, t_xbr, _) = bench("xbr", &g, RuleArg::Xbr, reps);
        println!("{n0}: {t_scalar:6.2} ms");
        println!(
            "{n1}: {t_fuzzy:6.2} ms   (x{:.2} vs scalar)",
            t_scalar / t_fuzzy
        );
        println!(
            "{n2}: {t_xbr:6.2} ms   (x{:.2} vs scalar)",
            t_scalar / t_xbr
        );
    }
}
