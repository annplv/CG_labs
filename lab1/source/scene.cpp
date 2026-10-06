#include "scene.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

#include "transform.hpp"

namespace scene {

namespace {

constexpr float two_pi = 2.0f * std::numbers::pi_v<float>;

const glm::vec3 palette[max_objects] = {
	{ 1.0f, 1.0f, 1.0f }, { 1.0f, 0.3f, 0.3f }, { 0.3f, 1.0f, 0.3f }, { 0.35f, 0.5f, 1.0f },
	{ 1.0f, 0.85f, 0.2f }, { 0.8f, 0.35f, 1.0f }, { 0.2f, 1.0f, 1.0f }, { 1.0f, 0.55f, 0.2f },
};

} // namespace

glm::vec3 orbitPoint(const Motion& m) {
	const float a = m.angle + m.phase;  // накопленный угол + стартовый сдвиг
	return { m.radius * std::cos(a), m.height * std::sin(float(m.waves) * a), m.radius * std::sin(a) };  // траектория - волнистый круг
}

void step(Motion& m, float dt) {
	if (!m.playing) {
		return;
	}
	m.angle = std::fmod(m.angle + dt * m.speed, two_pi);  // угол траектории (в пределах 2п для точности float)
	m.spin = std::fmod(m.spin + dt * glm::radians(m.spin_speed_degrees), two_pi); // угол собственного вращения
}

glm::mat4 modelMatrix(const Object& o) {
	const glm::vec3 r = glm::radians(o.rotation_degrees);
	const glm::mat4 rotation = xform::rotationZ(r.z) * xform::rotationY(r.y) * xform::rotationX(r.x);
	return xform::translation(o.position + orbitPoint(o.motion)) * xform::rotationY(o.motion.spin) * rotation *
		xform::scaling(o.scale); // растяжение -> повороты на углы -> вращение вокруг вертик оси -> перенос в точку
}

glm::mat4 viewMatrix(const Camera& c) {
	return xform::lookAt(glm::vec3(0.0f, 0.0f, c.distance), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
}

glm::mat4 projectionMatrix(const Camera& c, float aspect) {
	const float fov = glm::radians(c.fov_degrees);
	if (c.projection == Projection::Perspective) {  // перспективная
		return xform::perspective(fov, aspect, c.z_near, c.z_far);
	}
	const float h = c.distance * std::tan(fov * 0.5f);  // ортографическая (высота области как у перспективной в плоскости z = 0)
	const float w = h * aspect;
	return xform::orthographic(-w, w, -h, h, c.z_near, c.z_far);
}

Object makeObject(int index) {
	Object o;
	o.color = palette[index % max_objects];
	o.motion.phase = two_pi * float(index) / float(max_objects);
	return o;
}

void addObject(Scene& s) {
	if (s.count >= max_objects) {
		return;
	}
	s.objects[s.count] = makeObject(s.count);
	s.selected = s.count++;
}

void removeSelected(Scene& s) {
	if (s.count <= 1) {
		return;
	}
	std::move(s.objects.begin() + s.selected + 1, s.objects.begin() + s.count, s.objects.begin() + s.selected);
	--s.count;
	s.selected = std::min(s.selected, s.count - 1);
}

} // namespace scene
