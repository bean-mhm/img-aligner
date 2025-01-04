#version 450

// push constants
layout(push_constant, std430) uniform pc {
    layout(offset = 0) int enable_tint;
};

// uniforms
layout(binding = 1) uniform sampler2D tex;

// output from vertex shader
layout(location = 0) in vec2 v_texcoord;
layout(location = 1) in vec3 v_col;

// output from fragment shader
layout(location = 0) out vec4 out_col;

vec3 view_transform(vec3 col)
{
    // eliminate negative values before using power functions
    col = max(col, 0.);

    // OETF (Linear BT.709 I-D65 to sRGB 2.2)
    col = pow(col, vec3(1. / 2.2));

    return col;
}

void main()
{
    vec3 col = texture(tex, v_texcoord).rgb;
    col *= v_col;

    // use push constant
    if (enable_tint != 0)
    {
        col = mix(col, vec3(0, .1, .5), .001);
    }

    out_col = vec4(view_transform(col), 1);
}
