// fca_scale2x — rule-based 2x edge-directed upscaler (Scale2x/EPX).
// No CNN, no ML, no float-heavy math: 3x3 neighborhood "cellular rule".
// This is the AVX-FCA concept in GLSL: neighbor-pattern rule over a cell grid.
// Usage: mpv --glsl-shaders="~~/.config/mpv/shaders/fca_scale2x.glsl" file.mp4

//!DESC fca-scale2x: rule-based 2x upscale (CA-style, no CNN)
//!HOOK MAIN
//!BIND HOOKED
//!WIDTH HOOKED.w 2 *
//!HEIGHT HOOKED.h 2 *

#define FCA_EPS 0.08

// color equality with tolerance (quantized "cell state match")
bool fca_eq(vec4 a, vec4 b) {
    return distance(a.rgb, b.rgb) < FCA_EPS;
}

// pick first candidate, else second, else third, else center
vec4 fca_pick(bool p1, vec4 v1, bool p2, vec4 v2, bool p3, vec4 v3, vec4 e) {
    return p1 ? v1 : (p2 ? v2 : (p3 ? v3 : e));
}

vec4 hook() {
    // subpixel phase inside the source pixel: which of the 4 output pixels are we
    vec2 f = fract(HOOKED_pos * HOOKED_size);

    // 3x3 neighborhood of cells
    vec4 A = HOOKED_texOff(vec2(-1.0, -1.0));
    vec4 B = HOOKED_texOff(vec2( 0.0, -1.0));
    vec4 C = HOOKED_texOff(vec2( 1.0, -1.0));
    vec4 D = HOOKED_texOff(vec2(-1.0,  0.0));
    vec4 E = HOOKED_tex(HOOKED_pos);
    vec4 F = HOOKED_texOff(vec2( 1.0,  0.0));
    vec4 G = HOOKED_texOff(vec2(-1.0,  1.0));
    vec4 H = HOOKED_texOff(vec2( 0.0,  1.0));
    vec4 I = HOOKED_texOff(vec2( 1.0,  1.0));

    // EPX / Scale2x rule (shared condition, 4 subpixels)
    vec4 e0, e1, e2, e3;
    if (!fca_eq(B, H) && !fca_eq(D, F)) {
        e0 = fca_pick(fca_eq(B, D), B, fca_eq(B, F), F, fca_eq(D, F), D, E);
        e1 = fca_pick(fca_eq(B, F), F, fca_eq(B, D), B, fca_eq(F, D), D, E);
        e2 = fca_pick(fca_eq(H, D), D, fca_eq(H, F), F, fca_eq(D, F), D, E);
        e3 = fca_pick(fca_eq(H, F), F, fca_eq(H, D), D, fca_eq(F, D), D, E);
    } else {
        e0 = e1 = e2 = e3 = E;
    }

    bool left = f.x < 0.5;
    bool top  = f.y < 0.5;

    vec4 result = top ? (left ? e0 : e1) : (left ? e2 : e3);
    return vec4(result.rgb, E.a);
}
