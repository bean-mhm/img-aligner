#version 450

// push constants
layout(push_constant, std430) uniform pc {
    layout(offset = 0) float target_img_mul;
};

// uniforms
layout(binding = 0) uniform sampler2D warped_img;
layout(binding = 1) uniform sampler2D target_img;

// output from fragment shader
layout(location = 0) out vec4 out_col;

// unsigned logarithmic difference between two RGB triplets, preferably in
// Linear BT.709 I-D65.
float rgb_log_diff_unsigned(vec3 a, vec3 b)
{
    // clip negative values and add a tiny offset to avoid infinity
    a = max(a, 0.) + .001;
    b = max(b, 0.) + .001;
    
    // logarithmic difference per channel
    vec3 d = log(a / b);
    
    // take to the power of 2 and average for all channels
    return dot(d * d, vec3(1. / 3.));
}

// signed logarithmic difference between two RGB triplets, preferably in Linear
// BT.709 I-D65.
float rgb_log_diff(vec3 a, vec3 b)
{
    // clip negative values and add a tiny offset to avoid infinity
    a = max(a, 0.) + .001;
    b = max(b, 0.) + .001;
    
    // logarithmic difference per channel
    vec3 d = log(a / b);
    
    // take to the power of 2 and average for all channels
    float log_diff = dot(d * d, vec3(1. / 3.));
    
    // use luminance to determine which triplet is brighter (the sign)
    const vec3 LUMINANCE_WEIGHTS = vec3(.3, .5, .2);
    float lum_a = dot(a, LUMINANCE_WEIGHTS);
    float lum_b = dot(b, LUMINANCE_WEIGHTS);
    return lum_a > lum_b ? log_diff : -log_diff;
}

void main()
{
    ivec2 intermediate_res = textureSize(warped_img, 0);
    
    // remember that the resolution of the difference image is the upper power
    // of 2 of the intermediate resolution's largest axis (width or height) and
    // it's a square. this is because we use mipmapping to calculate the
    // average difference (error) value for all pixels. the actual difference
    // image will be rendered to the bottom-left corner so we need to make sure
    // that out-of-bounds regions are black so that they don't skew the average
    // difference (error) value.
    if (any(greaterThan(gl_FragCoord.xy, vec2(intermediate_res - 1) + .51)))
    {
        out_col = vec4(0);
    }
    else
    {
        vec2 texcoord = gl_FragCoord.xy / vec2(intermediate_res);

        vec3 warped_col = texture(warped_img, texcoord).rgb;
        vec3 target_col = texture(target_img, texcoord).rgb;

        // apply multiplier to the RGB channels in the target image
        target_col *= target_img_mul;

        // remember that the difference image has a single channel so the other
        // ones don't matter.
        out_col = vec4(rgb_log_diff_unsigned(warped_col, target_col));
    }
}
