#version 450

const vec2 fullscreen_quad_texcoords[6] = vec2[](
    vec2(0, 0),
    vec2(1, 0),
    vec2(1, 1),
    vec2(0, 0),
    vec2(1, 1),
    vec2(0, 1)
);

void main()
{
    vec2 texcoord = fullscreen_quad_texcoords[gl_VertexIndex];
    gl_Position = vec4(texcoord * 2. - 1., 0, 1);
}
