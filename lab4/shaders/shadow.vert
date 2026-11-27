#version 450

layout(location = 0) in vec3 in_position;
layout(push_constant) uniform ShadowView {
	mat4 light_view_proj;
} shadow;
layout(std140, set = 1, binding = 0) uniform ModelData {
	mat4 model;
} object;

void main() {
	gl_Position = shadow.light_view_proj * object.model * vec4(in_position, 1.0);
}
