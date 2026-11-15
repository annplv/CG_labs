#version 450



layout(location = 0) in vec3 v_world_position;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec3 v_color;

layout(location = 0) out vec4 out_color;

layout(std140, set = 0, binding = 0) uniform SceneData {
	mat4 view_proj;
	vec4 camera_position;
	vec4 camera_forward;
	vec4 ambient;
	vec4 light_direction;
	vec4 light_radiance;
	uvec4 light_counts;
} scene;

struct PointLight {
	vec4 position;  
	vec4 radiance;  
};

struct SpotLight {
	vec4 position;  
	vec4 direction; 
	vec4 radiance;  
	vec4 cutoff;    
};

layout(std430, set = 0, binding = 1) readonly buffer PointLights {
	PointLight point_lights[];
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
	SpotLight spot_lights[];
};

layout(std140, set = 1, binding = 0) uniform ModelData {
	mat4 model;
	mat4 normal_matrix;
	vec4 albedo;
	vec4 specular;
} object;


vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 albedo) {
	const float n_dot_l = dot(N, L);
	if (n_dot_l <= 0.0) {
		return vec3(0.0); 
	}
	const vec3 H = normalize(L + V); 
	const float highlight = pow(max(dot(N, H), 0.0), object.specular.w);
	return radiance * (albedo * n_dot_l + object.specular.rgb * highlight);
}

void main() {
	const vec3 N = normalize(v_normal);
	const vec3 V = scene.camera_position.w > 0.5 ? normalize(scene.camera_position.xyz - v_world_position)
	                                             : -scene.camera_forward.xyz;
	const vec3 albedo = object.albedo.rgb * v_color;

	// рассеянный свет
	vec3 color = scene.ambient.rgb * albedo;

	// направленный источник
	color += blinnPhong(N, V, normalize(-scene.light_direction.xyz), scene.light_radiance.rgb, albedo);

	// точечные источники
	for (uint i = 0u; i < scene.light_counts.x; ++i) {
		const vec3 to_light = point_lights[i].position.xyz - v_world_position;
		const float d2 = max(dot(to_light, to_light), 1e-4); 
		const vec3 L = to_light / sqrt(d2);
		const float attenuation = 1.0 / d2; 
		color += blinnPhong(N, V, L, point_lights[i].radiance.rgb * attenuation, albedo);
	}

	// прожекторы
	for (uint i = 0u; i < scene.light_counts.y; ++i) {
		const vec3 to_light = spot_lights[i].position.xyz - v_world_position;
		const float d2 = max(dot(to_light, to_light), 1e-4);
		const vec3 L = to_light / sqrt(d2);
		const float attenuation = 1.0 / d2;

		const float cos_theta = dot(-L, spot_lights[i].direction.xyz);
		const float cos_inner = spot_lights[i].cutoff.x;
		const float cos_outer = spot_lights[i].cutoff.y;

		float spot;
		if (cos_inner > cos_outer) {
			spot = clamp((cos_theta - cos_outer) / (cos_inner - cos_outer), 0.0, 1.0);
		} else {
			spot = cos_theta >= cos_outer ? 1.0 : 0.0;
		}

		color += blinnPhong(N, V, L, spot_lights[i].radiance.rgb * attenuation * spot, albedo);
	}

	out_color = vec4(color, 1.0);
}
