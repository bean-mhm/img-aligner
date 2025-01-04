#version 450

// uniforms
layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

// input from vertex buffer
layout(location = 0) in vec3 pos;
layout(location = 1) in vec2 texcoord;

// output from vertex shader
layout(location = 0) out vec2 v_texcoord;
layout(location = 1) out vec3 v_col;

void main()
{
    gl_Position =
        ubo.proj * ubo.view * ubo.model * vec4(pos, 1);

    v_col = vec3(1);
    v_texcoord = texcoord;
}
