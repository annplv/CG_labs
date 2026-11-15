#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_color;

layout(std140, set = 0, binding = 0) uniform SceneData {
	mat4 view_proj;
	vec4 camera_position;  
	vec4 camera_forward;   
	vec4 ambient;          
	vec4 light_direction;  
	vec4 light_radiance;   
	uvec4 light_counts;    
} scene;

layout(std140, set = 1, binding = 0) uniform ModelData {
	mat4 model;
	mat4 normal_matrix;    
	vec4 albedo;           
	vec4 specular;         
} object;

layout(location = 0) out vec3 v_world_position;
layout(location = 1) out vec3 v_normal;
layout(location = 2) out vec3 v_color;

void main() {
	const vec4 world = object.model * vec4(in_position, 1.0);
	gl_Position = scene.view_proj * world;
	v_world_position = world.xyz;
	v_normal = mat3(object.normal_matrix) * in_normal;
	v_color = mix(vec3(1.0), in_color, object.albedo.w); 
}
