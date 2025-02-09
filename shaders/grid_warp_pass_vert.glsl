#version 450

// input from vertex buffer
layout(location = 0) in vec2 warped_pos;
layout(location = 1) in vec2 orig_pos;

// output from vertex shader
layout(location = 0) out vec2 v_texcoord;

void main()
{
    gl_Position = vec4(warped_pos * 2. - 1., 0, 1);
    v_texcoord = orig_pos;
}
