#version 450

layout(push_constant) uniform Push {
	mat4 mvp;
	vec4 color;
} push;

layout(location = 0) out vec4 out_color;

void main() {
	out_color = vec4(push.color.rgb, 1.0);
}
