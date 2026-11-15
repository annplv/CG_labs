#pragma once

#include <array>
#include <vector>

#include <glm/glm.hpp>

// Состояние сцены и расчёты над ним. От Vulkan и ImGui не зависит.
namespace scene {

enum class Projection { Perspective, Orthographic };

// Лаба 2: камера задаётся позицией и углами поворота.
// yaw — поворот вокруг оси Y, pitch — наклон вверх/вниз; yaw = 0, pitch = 0 — взгляд вдоль -Z.
struct Camera {
	Projection projection = Projection::Perspective;
	glm::vec3 position = { 0.0f, 3.0f, 8.0f };
	float yaw_degrees = 0.0f;
	float pitch_degrees = -20.0f;
	float fov_degrees = 60.0f; // для ортографической проекции задаёт высоту видимой области
	float z_near = 0.1f;
	float z_far = 100.0f;
};

// Лаба 2: материал по модели Блинна-Фонга
struct Material {
	glm::vec3 albedo = glm::vec3(1.0f);   // цвет диффузного отражения (бывший цвет фигуры из лабы 1)
	glm::vec3 specular = glm::vec3(0.5f); // цвет блика
	float shininess = 32.0f;              // коэффициент блика
};

// Пирамида летает по волнистому кругу: p(a) = (R cos a, H sin(k a), R sin a), где k — целое число волн за оборот.
struct Motion {
	bool playing = true;
	float speed = 0.6f;               // скорость угла a, рад/с
	float spin_speed_degrees = 50.0f; // собственное вращение вокруг оси Y, градусы/с
	float radius = 2.0f;
	float height = 0.8f;
	int waves = 3;
	float phase = 0.0f;               // стартовый угол объекта на траектории
	float angle = 0.0f;               // накопленный угол траектории, рад
	float spin = 0.0f;                // накопленный угол собственного вращения, рад
};

struct Object {
	glm::vec3 position = glm::vec3(0.0f);
	glm::vec3 rotation_degrees = glm::vec3(0.0f);
	glm::vec3 scale = glm::vec3(1.0f);
	Material material;
	bool vertex_colors = true; // альбедо умножается на цвета вершин
	Motion motion;
};

// Лаба 2: источники света
struct AmbientLight {
	glm::vec3 color = glm::vec3(1.0f);
	float intensity = 0.1f;
};

struct DirectionalLight {
	float yaw_degrees = 30.0f;
	float pitch_degrees = -45.0f; // отрицательный — свет падает сверху
	glm::vec3 color = glm::vec3(1.0f);
	float intensity = 1.0f;
};

struct PointLight {
	glm::vec3 position = { 0.0f, 1.5f, 0.0f };
	glm::vec3 color = glm::vec3(1.0f);
	float intensity = 3.0f; // затухание: intensity / d^2
};

struct SpotLight {
	glm::vec3 position = { 0.0f, 3.0f, 0.0f };
	float yaw_degrees = 0.0f;
	float pitch_degrees = -90.0f; // смотрит вниз
	glm::vec3 color = glm::vec3(1.0f);
	float intensity = 8.0f;
	float inner_degrees = 20.0f; // внутри — полная яркость
	float outer_degrees = 30.0f; // снаружи — темно, между углами плавный переход
};

constexpr int max_objects = 8;

struct Scene {
	Camera camera;
	std::array<Object, max_objects> objects = {};
	int count = 1;
	int selected = 0;

	AmbientLight ambient;
	DirectionalLight directional;
	std::vector<PointLight> point_lights;
	std::vector<SpotLight> spot_lights;
	bool show_lights = true; // визуализация источников
};

glm::vec3 orbitPoint(const Motion& motion);
void step(Motion& motion, float dt);

// направление по двум углам (для камеры и источников)
glm::vec3 directionFromAngles(float yaw_degrees, float pitch_degrees);
glm::vec3 cameraForward(const Camera& camera);

// T(позиция + точка траектории) * Ry(собственное вращение) * Rz * Ry * Rx * S
glm::mat4 modelMatrix(const Object& object);
glm::mat4 viewMatrix(const Camera& camera);
glm::mat4 projectionMatrix(const Camera& camera, float aspect);

Object makeObject(int index);
void addObject(Scene& scene);
void removeSelected(Scene& scene);

} // namespace scene
