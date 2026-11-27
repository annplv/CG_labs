#version 450

layout(location = 0) in vec3 v_world_position;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec2 v_uv;
layout(location = 3) in vec4 v_sun_clip;
layout(location = 4) in vec4 v_spot_clip;

layout(location = 0) out vec4 out_color;

layout(std140, set = 0, binding = 0) uniform SceneData {
	mat4 view_proj;
	vec4 camera_position;
	vec4 camera_forward;
	vec4 ambient;
	vec4 light_direction;
	vec4 light_radiance;
	uvec4 light_counts;
	mat4 sun_view_proj;
	mat4 spot_view_proj;
} scene;

layout(set = 0, binding = 3) uniform sampler2DShadow sun_shadow;
layout(set = 0, binding = 4) uniform sampler2DShadow spot_shadow;

// В Vulkan глубина после деления на w уже лежит в [0, 1]. Перенос нужен только для XY.
float visibility(sampler2DShadow map, vec4 clip) {
	if (clip.w <= 0.0) return 1.0;
	vec3 ndc = clip.xyz / clip.w;
	if (any(greaterThan(abs(ndc.xy), vec2(1.0))) || ndc.z < 0.0 || ndc.z >= 1.0) return 1.0;
	return texture(map, vec3(ndc.xy * 0.5 + 0.5, ndc.z));
}

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
	vec4 specular; // rgb — цвет блика, w — коэффициент блика
} object;

// текстуры материала (у каждого материала свой набор дескрипторов 1)
layout(set = 1, binding = 1) uniform sampler2D albedo_map;   // основной цвет
layout(set = 1, binding = 2) uniform sampler2D specular_map; // где поверхность блестит
layout(set = 1, binding = 3) uniform sampler2D emissive_map; // где поверхность светится сама

// push-константы включают нетривиальное сэмплирование
layout(push_constant) uniform Push {
	float time; // секунды с запуска программы
	uint flags;
} push;

const uint FLAG_ANIMATED_SAMPLING = 1u;

// Нетривиальное сэмплирование: каналы цвета читаются из одной текстуры с разными координатами
vec3 sampleAnimated(vec2 uv, float t) {
	const vec2 offset = 0.03 * vec2(cos(t), sin(t));
	const float r = texture(albedo_map, uv + offset).r;
	const float g = texture(albedo_map, uv).g;
	const float b = texture(albedo_map, uv - offset).b;

	// модуляция цвета: оттенок плавно переходит от тёплого к холодному и обратно
	const vec3 tint = mix(vec3(1.0, 0.8, 0.6), vec3(0.6, 0.8, 1.0), 0.5 + 0.5 * sin(0.5 * t));
	return vec3(r, g, b) * tint;
}

vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 albedo, vec3 specular) {
	const float n_dot_l = dot(N, L);
	if (n_dot_l <= 0.0) {
		return vec3(0.0);
	}
	const vec3 H = normalize(L + V);
	const float highlight = pow(max(dot(N, H), 0.0), object.specular.w);
	return radiance * (albedo * n_dot_l + specular * highlight);
}

void main() {
	const vec3 N = normalize(v_normal);
	const vec3 V = scene.camera_position.w > 0.5 ? normalize(scene.camera_position.xyz - v_world_position)
	                                             : -scene.camera_forward.xyz;

	// лаба 3: альбедо из текстуры; способ чтения выбирает флаг из push-констант
	const vec3 albedo = (push.flags & FLAG_ANIMATED_SAMPLING) != 0u ? sampleAnimated(v_uv, push.time)
	                                                                 : texture(albedo_map, v_uv).rgb;
	// цвет блика умножается на карту блика: где тексель чёрный, блика нет
	const vec3 specular = object.specular.rgb * texture(specular_map, v_uv).rgb;

	// рассеянный свет
	vec3 color = scene.ambient.rgb * albedo;

	// направленный источник
	color += visibility(sun_shadow, v_sun_clip) * blinnPhong(N, V, normalize(-scene.light_direction.xyz), scene.light_radiance.rgb, albedo, specular);

	// точечные источники
	for (uint i = 0u; i < scene.light_counts.x; ++i) {
		const vec3 to_light = point_lights[i].position.xyz - v_world_position;
		const float d2 = max(dot(to_light, to_light), 1e-4);
		const vec3 L = to_light / sqrt(d2);
		const float attenuation = 1.0 / d2;
		color += blinnPhong(N, V, L, point_lights[i].radiance.rgb * attenuation, albedo, specular);
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

		if (i == 0u) spot *= visibility(spot_shadow, v_spot_clip);
		color += blinnPhong(N, V, L, spot_lights[i].radiance.rgb * attenuation * spot, albedo, specular);
	}

	// свечение прибавляется после Блинна-Фонга и от освещения не зависит
	color += texture(emissive_map, v_uv).rgb;

	out_color = vec4(color, 1.0);
}
