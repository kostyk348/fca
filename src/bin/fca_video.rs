//! fca_video — rawvideo pipe CLI: 2x/4x upscale + optional denoise/temporal
//! + "rich look" post FX (sharpen / contrast / vibrance / deband)
//! + 16-bit two-band upscale (hi bits via rule, lo bits via bicubic)
//! + VSR: multi-frame subpixel fusion (recovers real detail on pans/static).
//!
//! Usage:
//!   fca_video <W> <H> <scale2x|fuzzy|xbr|temporal> [flags]
//!
//! Gray 8-bit:  reads W*H gray, writes 2W*2H (or 4W*4H).
//! Color 8-bit: --yuv444 reads Y+U+V (each W*H), writes yuv444p at 2x/4x.
//! Color 16-bit: --16bit reads/writes yuv444p16le, two-band upscale.
//! VSR: --vsr replaces the rule in the first pass with 4-frame shift-and-add
//!   fusion (needs history; first frames of a stream are fallback).

use clap::Parser;
use fca::{denoise, postfx, rules, temporal, vsr, Grid};
use std::io::{Read, Write};
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "fca_video",
    version,
    about = "libfca rawvideo pipe: 2x/4x cellular upscale + denoise + temporal cache + post FX + VSR"
)]
struct Cli {
    /// Frame width
    width: usize,
    /// Frame height
    height: usize,
    /// Upscale mode
    mode: Mode,
    /// Median 3x3 pre-filter
    #[arg(long)]
    denoise: bool,
    /// Temporal tile size (temporal mode)
    #[arg(long, default_value_t = 32)]
    tile: usize,
    /// Temporal mode uses the xbr rule (default fuzzy)
    #[arg(long)]
    rule: Option<String>,
    /// yuv444p in/out (color pipeline; chroma = bicubic 2x)
    #[arg(long)]
    yuv444: bool,
    /// yuv444p16le in/out (two-band: hi->rule, lo->bicubic; implies --yuv444)
    #[arg(long = "16bit")]
    sixteen_bit: bool,
    /// VSR fusion as the first upscale pass (4-frame history)
    #[arg(long)]
    vsr: bool,
    /// Double pass -> 4x
    #[arg(long = "4x")]
    four_x: bool,
    /// Maximum-quality preset: 16-bit two-band + CAS (also between the 4x
    /// passes) + dehalo + edge-aware denoise 40 + vibrance 24 + line darkening.
    /// Implies --16bit --sharpen --dehalo; individual flags can only add on top.
    #[arg(long)]
    hq: bool,
    /// CAS contrast-adaptive sharpen on Y (hi band)
    #[arg(long)]
    sharpen: bool,
    /// Halo/ringing suppression
    #[arg(long)]
    dehalo: bool,
    /// Edge-aware denoise amount 0..255 (pre-upscale)
    #[arg(long = "dn", default_value_t = 0)]
    dn: i32,
    /// Ink-line darkening amount 0..255 (restores anime stroke weight)
    #[arg(long = "ld", default_value_t = 0)]
    ld: i32,
    /// Luma contrast 0..255
    #[arg(long, default_value_t = 0)]
    contrast: i32,
    /// Chroma saturation gain 0..255
    #[arg(long, default_value_t = 0)]
    vibrance: i32,
    /// Gradient dither (8-bit gray mode only; 16-bit is band-free)
    #[arg(long)]
    deband: bool,
}

#[derive(clap::ValueEnum, Clone, Copy, PartialEq)]
enum Mode {
    Scale2x,
    Fuzzy,
    Xbr,
    Temporal,
}

fn rule2x(src: &[u8], w: usize, h: usize, mode: Mode) -> Vec<u8> {
    let g = Grid::from_vec(w, h, src.to_vec());
    match mode {
        Mode::Scale2x => rules::scale2x_scalar(&g).data,
        Mode::Fuzzy | Mode::Temporal => rules::scale2x_fuzzy(&g).data,
        Mode::Xbr => rules::scale2x_xbr(&g).data,
    }
}

/// adaptive per-pixel mix: w = smoothstep(24, 96, range3x3 on input hi)
/// out = bc + (rl - bc) * w/255 (bc/rl/out are 2w*2h, hi is w*h)
fn adaptive_mix(hi: &[u8], bc: &[u8], rl: &[u8], w: usize, h: usize) -> Vec<u8> {
    let w2 = 2 * w;
    let mut out = vec![0u8; w2 * 2 * h];
    for y in 0..h {
        let row = &hi[y * w..(y + 1) * w];
        for x in 0..w {
            let xl = if x == 0 { 0 } else { x - 1 };
            let xr = if x + 1 < w { x + 1 } else { w - 1 };
            let mut mn = i32::from(row[xl]);
            let mut mx = mn;
            for dy in -1i32..=1 {
                let yy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                let r = &hi[yy * w..(yy + 1) * w];
                for v in [r[xl], r[x], r[xr]] {
                    let v = i32::from(v);
                    mn = mn.min(v);
                    mx = mx.max(v);
                }
            }
            let range = mx - mn;
            let w8 = if range <= 24 {
                0
            } else if range >= 96 {
                255
            } else {
                (range - 24) * 255 / 72
            };
            let ws = (w8 * w8) >> 8; // smoothstep
            let wf = (ws * (768 - 2 * w8)) >> 8; // 0..255
            for dy in 0..2 {
                for dx in 0..2 {
                    let i = (2 * y + dy) * w2 + 2 * x + dx;
                    let v = i32::from(bc[i]) + (((i32::from(rl[i]) - i32::from(bc[i])) * wf) >> 8);
                    out[i] = v.clamp(0, 255) as u8;
                }
            }
        }
    }
    out
}

struct Buffers {
    y: Vec<u8>,
    yl: Vec<u8>,
    u: Vec<u8>,
    ul: Vec<u8>,
    v: Vec<u8>,
    vl: Vec<u8>,
    mid_y: Vec<u8>,
    mid_yl: Vec<u8>,
    bc_y: Vec<u8>,
    rl_y: Vec<u8>,
    out_y: Vec<u8>,
    out_yl: Vec<u8>,
    out_u: Vec<u8>,
    out_ul: Vec<u8>,
    out_v: Vec<u8>,
    out_vl: Vec<u8>,
    hist: Vec<Vec<u8>>,
}

fn main() {
    std::process::exit(run());
}

fn run() -> i32 {
    let mut cli = Cli::parse();
    // --hq expands to the max chain before anything else reads the flags
    if cli.hq {
        cli.sixteen_bit = true;
        cli.sharpen = true;
        cli.dehalo = true;
        cli.dn = cli.dn.max(40);
        cli.vibrance = cli.vibrance.max(24);
        cli.ld = cli.ld.max(24);
    }
    let (w, h) = (cli.width, cli.height);
    if w == 0 || h == 0 {
        eprintln!("error: W and H must be positive");
        return 1;
    }
    let mode = cli.mode;
    let yuv444 = cli.yuv444 || cli.sixteen_bit;
    let bit16 = cli.sixteen_bit;
    let plane_sz = w * h;
    let bpp = if bit16 { 2 } else { 1 };
    let frame_sz = if yuv444 { 3 } else { 1 } * plane_sz * bpp;
    let scale = if cli.four_x { 4 } else { 2 };
    let out_sz = scale * scale * plane_sz; // per-plane output size (8-bit units)

    let temporal_rule = match cli.rule.as_deref() {
        Some("xbr") => temporal::Rule::Xbr,
        _ => temporal::Rule::Fuzzy,
    };
    let mut tu = temporal::TemporalUpscaler::new(w, h, cli.tile, temporal_rule);

    let mut bufs = Buffers {
        y: vec![0; plane_sz],
        yl: vec![0; plane_sz],
        u: vec![0; plane_sz],
        ul: vec![0; plane_sz],
        v: vec![0; plane_sz],
        vl: vec![0; plane_sz],
        mid_y: vec![0; out_sz],
        mid_yl: vec![0; out_sz],
        bc_y: vec![0; out_sz],
        rl_y: vec![0; out_sz],
        out_y: vec![0; out_sz],
        out_yl: vec![0; out_sz],
        out_u: vec![0; out_sz],
        out_ul: vec![0; out_sz],
        out_v: vec![0; out_sz],
        out_vl: vec![0; out_sz],
        hist: vec![vec![0; plane_sz]; 4],
    };

    let t0 = Instant::now();
    let mut frames = 0u64;
    let mut vsr_frames = 0i64;
    let mut frame = vec![0u8; frame_sz];
    let mut stdin = std::io::stdin().lock();
    let mut stdout = std::io::stdout().lock();

    loop {
        if stdin.read_exact(&mut frame).is_err() {
            break;
        }
        let src = frame.as_slice();
        if bit16 {
            for i in 0..plane_sz {
                let y = u16::from(src[2 * i]) | (u16::from(src[2 * i + 1]) << 8);
                let u = u16::from(src[2 * (plane_sz + i)])
                    | (u16::from(src[2 * (plane_sz + i) + 1]) << 8);
                let v = u16::from(src[2 * (2 * plane_sz + i)])
                    | (u16::from(src[2 * (2 * plane_sz + i) + 1]) << 8);
                bufs.y[i] = (y >> 8) as u8;
                bufs.yl[i] = (y & 0xFF) as u8;
                bufs.u[i] = (u >> 8) as u8;
                bufs.ul[i] = (u & 0xFF) as u8;
                bufs.v[i] = (v >> 8) as u8;
                bufs.vl[i] = (v & 0xFF) as u8;
            }
        } else if yuv444 {
            bufs.y.copy_from_slice(&src[..plane_sz]);
            bufs.u.copy_from_slice(&src[plane_sz..2 * plane_sz]);
            bufs.v.copy_from_slice(&src[2 * plane_sz..3 * plane_sz]);
        }

        let mut y_in: Vec<u8> = if yuv444 { bufs.y.clone() } else { src.to_vec() };
        if cli.denoise {
            let g = Grid::from_vec(w, h, y_in.clone());
            y_in = denoise::median3x3(&g).data;
        }
        // edge-aware denoise: kills compression noise BEFORE it gets scaled up
        // and amplified by sharpening; lines survive (bilateral weights + edge guard)
        if cli.dn > 0 {
            postfx::denoise_edge(&mut y_in, w, h, cli.dn);
        }

        // ================= luma upscale =================
        let mut use_vsr = false;
        let mut vsr_g = vsr::Shift::default();
        if cli.vsr && vsr_frames >= 3 {
            vsr_g = vsr::estimate_shift(&bufs.hist[3], &y_in, w, h);
            let gmag = vsr_g.gx.abs().max(vsr_g.gy.abs());
            // fuse only for real, gentle global motion (0.25..2.5 px);
            // static scenes and scene cuts keep the sharp rule/bicubic path.
            use_vsr = (64..=640).contains(&gmag);
        }
        if use_vsr {
            // ---- VSR pass: 4-frame fusion (2x) into mid_y ----
            let hp: Vec<&[u8]> = bufs.hist.iter().map(Vec::as_slice).collect();
            vsr::fusion2x(&y_in, &hp, w, h, vsr_g, &mut bufs.mid_y);
            if cli.four_x {
                if bit16 {
                    // second pass: two-band on the 2x fused result
                    if cli.dehalo {
                        postfx::dehalo(&mut bufs.mid_y, 2 * w, 2 * h);
                    }
                    if cli.hq {
                        // restore micro-contrast between the two passes
                        postfx::cas_sharpen(&mut bufs.mid_y, 2 * w, 2 * h, 160);
                    }
                    bufs.mid_yl = postfx::bicubic2x(&bufs.yl, w, h);
                    bufs.bc_y = postfx::bicubic2x(&bufs.mid_y, 2 * w, 2 * h);
                    bufs.rl_y = rule2x(&bufs.mid_y, 2 * w, 2 * h, mode);
                    bufs.out_y = adaptive_mix(&bufs.mid_y, &bufs.bc_y, &bufs.rl_y, 2 * w, 2 * h);
                    bufs.out_yl = postfx::bicubic2x(&bufs.mid_yl, 2 * w, 2 * h);
                } else {
                    if cli.dehalo {
                        postfx::dehalo(&mut bufs.mid_y, 2 * w, 2 * h);
                    }
                    bufs.out_y = rule2x(&bufs.mid_y, 2 * w, 2 * h, mode);
                }
            } else {
                bufs.out_y.copy_from_slice(&bufs.mid_y);
                if bit16 {
                    bufs.out_yl = postfx::bicubic2x(&bufs.yl, w, h);
                }
            }
        } else if bit16 {
            // two-band: hi -> adaptive mix(rule, bicubic), lo -> bicubic
            bufs.bc_y = postfx::bicubic2x(&y_in, w, h);
            bufs.rl_y = rule2x(&y_in, w, h, mode);
            let mixed = adaptive_mix(&y_in, &bufs.bc_y, &bufs.rl_y, w, h);
            bufs.mid_yl = postfx::bicubic2x(&bufs.yl, w, h);
            if cli.four_x {
                // second pass: two-band again on the 2x result
                let mut mid2 = mixed;
                if cli.dehalo {
                    postfx::dehalo(&mut mid2, 2 * w, 2 * h);
                }
                if cli.hq {
                    // restore micro-contrast between the two passes
                    postfx::cas_sharpen(&mut mid2, 2 * w, 2 * h, 160);
                }
                bufs.bc_y = postfx::bicubic2x(&mid2, 2 * w, 2 * h);
                bufs.rl_y = rule2x(&mid2, 2 * w, 2 * h, mode);
                bufs.out_y = adaptive_mix(&mid2, &bufs.bc_y, &bufs.rl_y, 2 * w, 2 * h);
                bufs.out_yl = postfx::bicubic2x(&bufs.mid_yl, 2 * w, 2 * h);
            } else {
                bufs.out_y = mixed;
                bufs.out_yl = bufs.mid_yl.clone();
            }
            // FX on hi band
            if cli.dehalo && !cli.four_x {
                postfx::dehalo(&mut bufs.out_y, scale * w, scale * h);
            }
            if cli.sharpen {
                postfx::cas_sharpen(&mut bufs.out_y, scale * w, scale * h, 255);
            }
            if cli.ld > 0 {
                postfx::line_dark(&mut bufs.out_y, scale * w, scale * h, cli.ld);
            }
            if cli.contrast > 0 {
                postfx::contrast_s(&mut bufs.out_y, scale * w, scale * h, cli.contrast);
            }
        } else {
            // 8-bit: rule-only (or temporal)
            if cli.four_x {
                if mode == Mode::Temporal {
                    tu.process(&y_in, &mut bufs.mid_y);
                } else {
                    bufs.mid_y = rule2x(&y_in, w, h, mode);
                }
                if cli.dehalo {
                    postfx::dehalo(&mut bufs.mid_y, 2 * w, 2 * h);
                }
                bufs.out_y = rule2x(&bufs.mid_y, 2 * w, 2 * h, mode);
            } else {
                if mode == Mode::Temporal {
                    tu.process(&y_in, &mut bufs.out_y);
                } else {
                    bufs.out_y = rule2x(&y_in, w, h, mode);
                }
            }
            if cli.dehalo && !cli.four_x {
                postfx::dehalo(&mut bufs.out_y, scale * w, scale * h);
            }
            if cli.sharpen {
                postfx::cas_sharpen(&mut bufs.out_y, scale * w, scale * h, 255);
            }
            if cli.ld > 0 {
                postfx::line_dark(&mut bufs.out_y, scale * w, scale * h, cli.ld);
            }
            if cli.contrast > 0 {
                postfx::contrast_s(&mut bufs.out_y, scale * w, scale * h, cli.contrast);
            }
            if cli.deband {
                postfx::deband(&mut bufs.out_y, scale * w, scale * h, frames as u32);
            }
        }

        // ================= chroma =================
        if yuv444 {
            if bit16 {
                // chroma: pure bicubic on hi and lo bands
                if cli.four_x {
                    let t_u = postfx::bicubic2x(&bufs.u, w, h);
                    bufs.out_u = postfx::bicubic2x(&t_u, 2 * w, 2 * h);
                    let t_ul = postfx::bicubic2x(&bufs.ul, w, h);
                    bufs.out_ul = postfx::bicubic2x(&t_ul, 2 * w, 2 * h);
                    let t_v = postfx::bicubic2x(&bufs.v, w, h);
                    bufs.out_v = postfx::bicubic2x(&t_v, 2 * w, 2 * h);
                    let t_vl = postfx::bicubic2x(&bufs.vl, w, h);
                    bufs.out_vl = postfx::bicubic2x(&t_vl, 2 * w, 2 * h);
                } else {
                    bufs.out_u = postfx::bicubic2x(&bufs.u, w, h);
                    bufs.out_ul = postfx::bicubic2x(&bufs.ul, w, h);
                    bufs.out_v = postfx::bicubic2x(&bufs.v, w, h);
                    bufs.out_vl = postfx::bicubic2x(&bufs.vl, w, h);
                }
            } else {
                if cli.four_x {
                    let t_u = postfx::bicubic2x(&bufs.u, w, h);
                    bufs.out_u = postfx::bicubic2x(&t_u, 2 * w, 2 * h);
                    let t_v = postfx::bicubic2x(&bufs.v, w, h);
                    bufs.out_v = postfx::bicubic2x(&t_v, 2 * w, 2 * h);
                } else {
                    bufs.out_u = postfx::bicubic2x(&bufs.u, w, h);
                    bufs.out_v = postfx::bicubic2x(&bufs.v, w, h);
                }
                if cli.vibrance > 0 {
                    postfx::vibrance(
                        &mut bufs.out_u,
                        &mut bufs.out_v,
                        scale * w,
                        scale * h,
                        cli.vibrance,
                    );
                }
            }
        }

        // ================= write =================
        let mut out_bytes: Vec<u8>;
        if bit16 {
            let n = out_sz;
            // yuv444p16le packed as u16 (lo byte first), dumped once as LE bytes
            let mut planes16 = vec![0u16; 3 * n];
            let plane_pairs = [
                (&bufs.out_y, &bufs.out_yl, 0),
                (&bufs.out_u, &bufs.out_ul, 1),
                (&bufs.out_v, &bufs.out_vl, 2),
            ];
            for (hi, lo, k) in plane_pairs {
                let base = k * n;
                for i in 0..n {
                    planes16[base + i] = (u16::from(hi[i]) << 8) | u16::from(lo[i]);
                }
            }
            if cli.vibrance > 0 {
                let g = (256 + ((cli.vibrance * 100) >> 8)) as u32;
                for k in 1..3 {
                    let base = k * n;
                    for cell in &mut planes16[base..base + n] {
                        let c = i32::from(*cell) - 32768;
                        *cell = (32768 + ((c * g as i32) >> 8)).clamp(0, 65535) as u16;
                    }
                }
            }
            // u16 slice -> LE byte view (u16 is 2-aligned, length is even)
            let bytes =
                unsafe { std::slice::from_raw_parts(planes16.as_ptr().cast::<u8>(), 3 * n * 2) };
            out_bytes = bytes.to_vec();
        } else if yuv444 {
            out_bytes = Vec::with_capacity(3 * out_sz);
            out_bytes.extend_from_slice(&bufs.out_y);
            out_bytes.extend_from_slice(&bufs.out_u);
            out_bytes.extend_from_slice(&bufs.out_v);
        } else {
            out_bytes = bufs.out_y.clone();
        }
        if stdout.write_all(&out_bytes).is_err() {
            break;
        }

        // update VSR history
        if cli.vsr {
            bufs.hist.rotate_left(1);
            bufs.hist[3].copy_from_slice(&y_in);
            if vsr_frames < 1_000_000_000 {
                vsr_frames += 1;
            }
        }
        frames += 1;
    }

    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let st = tu.stats();
    eprintln!(
        "[fca_video] frames={frames}  total={ms:.1} ms ({:.2} ms/frame)  tiles: {} total, {} recomputed ({:.1}%), {} reused  vsr_frames={vsr_frames}",
        if frames > 0 { ms / frames as f64 } else { 0.0 },
        st.total,
        st.recomputed,
        if st.total > 0 { 100.0 * st.recomputed as f64 / st.total as f64 } else { 0.0 },
        st.reused,
    );
    0
}
