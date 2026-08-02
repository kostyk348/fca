// fca_xbr2 — xBR-style fuzzy upscaler with diagonal corners (A,C,G,I).
// Upgrade over fca_fuzzy_scale2x: each subpixel blends its pair (p,q)
// weighted by the diagonal corner closeness, instead of picking one candidate:
//   strength = m(p~q)*(1-m(perp1~p))*(1-m(perp2~q))
//   blend    = (w_p*p + w_q*q)/(w_p+w_q),  w = m(corner~.) 
//   out      = mix(E, blend, strength)
// Thin 1px diagonal lines survive; anti-ringing holds by construction
// (blends stay inside the 3x3 min/max). CPU twin: fca::rule::scale2x_xbr.
// Usage: mpv --glsl-shaders="~~/.config/mpv/shaders/fca_xbr2.glsl" file.mp4

//!DESC fca-xbr2: xBR-style fuzzy 2x upscale (diagonal corners, soft blend)
//!HOOK MAIN
//!BIND HOOKED
//!WIDTH HOOKED.w 2 *
//!HEIGHT HOOKED.h 2 *

#define FCA_W 0.16   // membership width (40/255, matches kFuzzyW)

float fca_d(vec3 a, vec3 b) {
    return (abs(a.r - b.r) + abs(a.g - b.g) + abs(a.b - b.b)) / 3.0;
}

float fca_m(float d) {
    return clamp(1.0 - d / FCA_W, 0.0, 1.0);
}

// weighted blend of two candidates by corner closeness
vec3 fca_blend2(vec3 p, vec3 q, float wp, float wq) {
    return (p * wp + q * wq) / max(wp + wq, 1e-6);
}

vec4 hook() {
    vec2 f = fract(HOOKED_pos * HOOKED_size);

    vec4 A = HOOKED_texOff(vec2(-1.0, -1.0));
    vec4 B = HOOKED_texOff(vec2( 0.0, -1.0));
    vec4 C = HOOKED_texOff(vec2( 1.0, -1.0));
    vec4 D = HOOKED_texOff(vec2(-1.0,  0.0));
    vec4 E = HOOKED_tex(HOOKED_pos);
    vec4 F = HOOKED_texOff(vec2( 1.0,  0.0));
    vec4 G = HOOKED_texOff(vec2(-1.0,  1.0));
    vec4 H = HOOKED_texOff(vec2( 0.0,  1.0));
    vec4 I = HOOKED_texOff(vec2( 1.0,  1.0));

    vec3 e0, e1, e2, e3;
    // subpixel e0 (TL): pair B,D corner A; perps F,H
    {
        float s = fca_m(fca_d(B.rgb, D.rgb)) * (1.0 - fca_m(fca_d(F.rgb, B.rgb))) * (1.0 - fca_m(fca_d(H.rgb, D.rgb)));
        vec3 b = fca_blend2(B.rgb, D.rgb, fca_m(fca_d(A.rgb, B.rgb)), fca_m(fca_d(A.rgb, D.rgb)));
        e0 = mix(E.rgb, b, s);
    }
    // e1 (TR): pair F,B corner C; perps D,H
    {
        float s = fca_m(fca_d(F.rgb, B.rgb)) * (1.0 - fca_m(fca_d(D.rgb, F.rgb))) * (1.0 - fca_m(fca_d(H.rgb, B.rgb)));
        vec3 b = fca_blend2(F.rgb, B.rgb, fca_m(fca_d(C.rgb, F.rgb)), fca_m(fca_d(C.rgb, B.rgb)));
        e1 = mix(E.rgb, b, s);
    }
    // e2 (BL): pair D,H corner G; perps B,F
    {
        float s = fca_m(fca_d(D.rgb, H.rgb)) * (1.0 - fca_m(fca_d(B.rgb, D.rgb))) * (1.0 - fca_m(fca_d(F.rgb, H.rgb)));
        vec3 b = fca_blend2(D.rgb, H.rgb, fca_m(fca_d(G.rgb, D.rgb)), fca_m(fca_d(G.rgb, H.rgb)));
        e2 = mix(E.rgb, b, s);
    }
    // e3 (BR): pair F,H corner I; perps D,B
    {
        float s = fca_m(fca_d(F.rgb, H.rgb)) * (1.0 - fca_m(fca_d(D.rgb, F.rgb))) * (1.0 - fca_m(fca_d(B.rgb, H.rgb)));
        vec3 b = fca_blend2(F.rgb, H.rgb, fca_m(fca_d(I.rgb, F.rgb)), fca_m(fca_d(I.rgb, H.rgb)));
        e3 = mix(E.rgb, b, s);
    }

    bool left = f.x < 0.5;
    bool top  = f.y < 0.5;
    vec3 result = top ? (left ? e0 : e1) : (left ? e2 : e3);
    return vec4(result, E.a);
}
