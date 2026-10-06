#pragma once

#include <array>

#include <glm/glm.hpp>

// Состояние сцены и расчёты над ним. От Vulkan и ImGui не зависит.
namespace scene {

enum class Projection { Perspective, Orthographic };

struct Camera {
	Projection projection = Projection::Perspective;
	float distance = 8.0f;     // камера стоит в (0, 0, distance) и смотрит в начало координат
	float fov_degrees = 60.0f; // для ортографической проекции задаёт высоту видимой области
	float z_near = 0.1f;
	float z_far = 100.0f;
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
	glm::vec3 color = glm::vec3(1.0f);
	bool vertex_colors = true;
	Motion motion;
};

constexpr int max_objects = 8;

struct Scene {
	Camera camera;
	std::array<Object, max_objects> objects = {};
	int count = 1;
	int selected = 0;
};

glm::vec3 orbitPoint(const Motion& motion);
void step(Motion& motion, float dt);

// T(позиция + точка траектории) * Ry(собственное вращение) * Rz * Ry * Rx * S
glm::mat4 modelMatrix(const Object& object);
glm::mat4 viewMatrix(const Camera& camera);
glm::mat4 projectionMatrix(const Camera& camera, float aspect);

Object makeObject(int index);
void addObject(Scene& scene);
void removeSelected(Scene& scene);

} // namespace scene
