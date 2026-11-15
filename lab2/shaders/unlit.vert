#version 450

layout(location = 0) in vec3 in_position;

layout(push_constant) uniform Push {
	mat4 mvp;
	vec4 color;
} push;

void main() {
	gl_Position = push.mvp * vec4(in_position, 1.0);
}
