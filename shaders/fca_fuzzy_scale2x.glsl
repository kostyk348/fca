// fca_fuzzy_scale2x — fuzzy membership 2x upscaler (EAFAR S5 idea in GLSL).
// Same rule family as fca_scale2x.glsl, but instead of a hard pick3 with
// tolerance, each subpixel blends by membership degree (triangle shape):
//   w = m(a~c) * (1-m(b1~a)) * (1-m(b2~c)),  out = mix(E, a, w)
// Result: anti-aliased diagonals, no hard threshold -> no frame flicker.
// The CPU twin (fca::rule::scale2x_fuzzy) is bit-exact in integer math.
// Usage: mpv --glsl-shaders="~~/.config/mpv/shaders/fca_fuzzy_scale2x.glsl" file.mp4

//!DESC fca-fuzzy-scale2x: fuzzy-membership 2x upscale (soft blend, no flicker)
//!HOOK MAIN
//!BIND HOOKED
//!WIDTH HOOKED.w 2 *
//!HEIGHT HOOKED.h 2 *

#define FCA_W 0.16   // membership width (40/255, matches kFuzzyW)

// mean abs channel difference
float fca_d(vec3 a, vec3 b) {
    return (abs(a.r - b.r) + abs(a.g - b.g) + abs(a.b - b.b)) / 3.0;
}

// triangle membership: 1 at d==0, 0 at d>=FCA_W
float fca_m(float d) {
    return clamp(1.0 - d / FCA_W, 0.0, 1.0);
}

vec3 fca_blend(vec3 a, vec3 e, float w) {
    return mix(e, a, w);
}

vec4 hook() {
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

    // six unique pairwise memberships
    float mBD = fca_m(fca_d(B.rgb, D.rgb));
    float mFB = fca_m(fca_d(F.rgb, B.rgb));
    float mHD = fca_m(fca_d(H.rgb, D.rgb));
    float mDF = fca_m(fca_d(D.rgb, F.rgb));
    float mHB = fca_m(fca_d(H.rgb, B.rgb));
    float mFH = fca_m(fca_d(F.rgb, H.rgb));

    // per-subpixel weights (same topology as the CPU rule)
    float w0 = mBD * (1.0 - mFB) * (1.0 - mHD);
    float w1 = mFB * (1.0 - mDF) * (1.0 - mHB);
    float w2 = mHD * (1.0 - mBD) * (1.0 - mFH);
    float w3 = mFH * (1.0 - mDF) * (1.0 - mHB);

    vec3 e0 = fca_blend(B.rgb, E.rgb, w0);
    vec3 e1 = fca_blend(F.rgb, E.rgb, w1);
    vec3 e2 = fca_blend(D.rgb, E.rgb, w2);
    vec3 e3 = fca_blend(F.rgb, E.rgb, w3);

    bool left = f.x < 0.5;
    bool top  = f.y < 0.5;

    vec3 result = top ? (left ? e0 : e1) : (left ? e2 : e3);
    return vec4(result, E.a);
}
