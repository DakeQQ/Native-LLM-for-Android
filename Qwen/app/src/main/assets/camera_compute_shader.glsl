#version 310 es
#extension GL_OES_EGL_image_external_essl3 : require
precision lowp float;
precision highp int;
layout(local_size_x = 16, local_size_y = 16) in;

uniform samplerExternalOES uCamera;
uniform mat2  uM2;
uniform vec2  uT;
uniform ivec2 uOutSize;   // (W, H)
uniform vec2  uInvOut;    // (1/W, 1/H)
// Letterbox fit: the fraction (fx, fy) of the output that the aspect-preserved camera frame occupies,
// centered. (1,1) = fill (no letterbox). Native computes this from the camera buffer aspect vs the
// target aspect so out-of-fit pixels become neutral gray 128, replicating load_image_letterbox.
uniform vec2  uFit;

// Output 0: interleaved RGBA8, one uint per pixel (0xFFBBGGRR little-endian => bytes R,G,B,0xFF).
// Consumed directly by the on-screen preview (Bitmap.copyPixelsFromBuffer, ARGB_8888).
layout(std430, binding = 0) buffer OutPacked {
    uint rgba[];
};
// Output 1: planar CHW uint8, tightly packed [R plane | G plane | B plane], 4 pixels per uint.
// This is the inference pixel_values layout ([3,H,W] uint8 == rrr..ggg..bbb..). Each invocation packs
// 4 horizontally-adjacent pixels into one uint per plane, so it REQUIRES W % 4 == 0 (the exported
// input_image_size is always a multiple of 16). The transpose is done here on the GPU (no CPU pass).
layout(std430, binding = 1) buffer OutPlanar {
    uint chw[];
};

uvec3 sampleRGB(uint px, uint py, bool mirror) {
    vec2 ndc = (vec2(float(px), float(py)) + 0.5) * uInvOut;   // output-normalized [0,1]
    vec2 fitOffset = 0.5 * (vec2(1.0) - uFit);                 // centered letterbox origin
    vec2 local = (ndc - fitOffset) / uFit;                     // map into the fit rectangle
    if (local.x < 0.0 || local.x > 1.0 || local.y < 0.0 || local.y > 1.0) {
        return uvec3(128u);                                    // gray pad (near-zero after normalization)
    }
    vec2 uv = uM2 * local + uT;
    if (mirror) {
        uv.x = 1.0 - uv.x;
    }
    vec3 rgb = texture(uCamera, uv).rgb;
    return uvec3(clamp(rgb, 0.0, 1.0) * 255.0 + 0.5);
}

void main() {
    uint W = uint(uOutSize.x);
    uint H = uint(uOutSize.y);
    uint x0 = gl_GlobalInvocationID.x * 4u;   // this invocation owns the 4 pixels x0..x0+3 on row y
    uint y  = gl_GlobalInvocationID.y;
    if (x0 >= W || y >= H) return;

    // --- interleaved RGBA for the preview (mirrored, selfie-style; letterboxed to match inference) ---
    uvec3 r0 = sampleRGB(x0,      y, true);
    uvec3 r1 = sampleRGB(x0 + 1u, y, true);
    uvec3 r2 = sampleRGB(x0 + 2u, y, true);
    uvec3 r3 = sampleRGB(x0 + 3u, y, true);
    uint p = y * W + x0;
    rgba[p]      = 0xFF000000u | (r0.b << 16) | (r0.g << 8) | r0.r;
    rgba[p + 1u] = 0xFF000000u | (r1.b << 16) | (r1.g << 8) | r1.r;
    rgba[p + 2u] = 0xFF000000u | (r2.b << 16) | (r2.g << 8) | r2.r;
    rgba[p + 3u] = 0xFF000000u | (r3.b << 16) | (r3.g << 8) | r3.r;

    // --- planar CHW (rrrr.. / gggg.. / bbbb..), un-mirrored, letterboxed == load_image_letterbox ---
    // Produce model rows top-first at the camera GPU source; display-only paths handle their own view transform.
    uint chwY = H - 1u - y;
    uvec3 c0 = sampleRGB(x0,      chwY, false);
    uvec3 c1 = sampleRGB(x0 + 1u, chwY, false);
    uvec3 c2 = sampleRGB(x0 + 2u, chwY, false);
    uvec3 c3 = sampleRGB(x0 + 3u, chwY, false);
    uint planeWords = (W * H) >> 2;   // bytes-per-plane / 4
    uint w = p >> 2;                  // (y*W + x0) / 4  (x0 is a multiple of 4)
    chw[w]                   = c0.r | (c1.r << 8) | (c2.r << 16) | (c3.r << 24);
    chw[planeWords + w]      = c0.g | (c1.g << 8) | (c2.g << 16) | (c3.g << 24);
    chw[2u * planeWords + w] = c0.b | (c1.b << 8) | (c2.b << 16) | (c3.b << 24);
}
