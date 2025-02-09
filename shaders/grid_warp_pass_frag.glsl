#version 450

// push constants
layout(push_constant, std430) uniform pc {
    layout(offset = 0) float base_img_mul;
};

// uniforms
layout(binding = 0) uniform sampler2D base_img;

// output from vertex shader
layout(location = 0) in vec2 v_texcoord;

// output from fragment shader
layout(location = 0) out vec4 out_col;

void main()
{
    // sample the image
    out_col = texture(base_img, v_texcoord);

    // apply multiplier to the RGB channels in the base image
    out_col.rgb *= base_img_mul;
}
