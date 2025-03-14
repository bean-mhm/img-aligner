#version 450

// push constants
layout(push_constant, std430) uniform pc {
    layout(offset = 0) ivec2 cost_res;
};

// uniforms
layout(binding = 0) uniform sampler2D difference_img;

// output from fragment shader
layout(location = 0) out vec4 out_col;

float fetch(ivec2 icoord)
{
    return texelFetch(difference_img, icoord, 0).r;
}

// render smooth downscaled (at cost_res) version of difference_img
void main()
{
    // bottom left and top right corners of the current pixel in the pixel
    // space of the downscaled image (bottom left, top right, etc.).
    vec2 bl = gl_FragCoord.xy - .5;
    vec2 tr = gl_FragCoord.xy + .5;
    
    // transform the corners to the pixel space at the original resolution
    vec2 scale = vec2(textureSize(difference_img, 0)) / vec2(cost_res);
    bl *= scale;
    tr *= scale;
    
    // average out pixels in the AABB (axis-aligned bounding box) formed
    // by these corner points.
    float v = 0.;
    float sum_weights = 0.;
    for (int y = int(floor(bl.y)); y <= int(floor(tr.y)); y++)
    {
        for (int x = int(floor(bl.x)); x <= int(floor(tr.x)); x++)
        {
            // corners of the current pixel
            vec2 curr_bl = vec2(x, y);
            vec2 curr_tr = curr_bl + 1.;
            
            // intersect this pixel's AABB with the AABB formed by the
            // corner points from before.
            vec2 intr_bl = max(bl, curr_bl);
            vec2 intr_tr = min(tr, curr_tr);
        
            // sample weight = area of intersection / area of the pixel (1)
            vec2 intr_diagonal = max(intr_tr - intr_bl, 0.);
            float intr_area = intr_diagonal.x * intr_diagonal.y;
            
            // accumulate
            v += intr_area * fetch(ivec2(x, y));
            sum_weights += intr_area;
        }
    }
    v /= sum_weights;;

    // output
    out_col = vec4(v);
}
