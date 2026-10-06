#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

layout(set = 0, binding = 0) uniform ObjectData {
	mat4 model;
	mat4 view;
	mat4 proj;
	vec4 tint;   // цвет из интерфейса
	vec4 params; // 1 - если включены цвета вершин
} u;

layout(location = 0) out vec3 v_color;

void main() {
	gl_Position = u.proj * u.view * u.model * vec4(in_position, 1.0);
	v_color = mix(vec3(1.0), in_color, u.params.x) * u.tint.rgb;
}
