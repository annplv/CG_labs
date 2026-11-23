#include "transform.hpp"

#include <cmath>

namespace xform {

glm::mat4 translation(const glm::vec3& offset) {
	glm::mat4 m(1.0f);
	m[3] = glm::vec4(offset, 1.0f);
	return m;
}

glm::mat4 scaling(const glm::vec3& factors) {
	glm::mat4 m(1.0f);
	m[0][0] = factors.x;
	m[1][1] = factors.y;
	m[2][2] = factors.z;
	return m;
}

glm::mat4 rotationX(float angle) {
	const float c = std::cos(angle), s = std::sin(angle);
	glm::mat4 m(1.0f);
	m[1][1] = c;  m[2][1] = -s;
	m[1][2] = s;  m[2][2] = c;
	return m;
}

glm::mat4 rotationY(float angle) {
	const float c = std::cos(angle), s = std::sin(angle);
	glm::mat4 m(1.0f);
	m[0][0] = c;  m[2][0] = s;
	m[0][2] = -s; m[2][2] = c;
	return m;
}

glm::mat4 rotationZ(float angle) {
	const float c = std::cos(angle), s = std::sin(angle);
	glm::mat4 m(1.0f);
	m[0][0] = c;  m[1][0] = -s;
	m[0][1] = s;  m[1][1] = c;
	return m;
}

glm::mat4 lookAt(const glm::vec3& eye, const glm::vec3& target, const glm::vec3& up) {
	const glm::vec3 f = glm::normalize(target - eye);
	const glm::vec3 r = glm::normalize(glm::cross(f, up));
	const glm::vec3 u = glm::cross(r, f);

	glm::mat4 m(1.0f);
	m[0][0] = r.x;  m[1][0] = r.y;  m[2][0] = r.z;  m[3][0] = -glm::dot(r, eye);
	m[0][1] = u.x;  m[1][1] = u.y;  m[2][1] = u.z;  m[3][1] = -glm::dot(u, eye);
	m[0][2] = -f.x; m[1][2] = -f.y; m[2][2] = -f.z; m[3][2] = glm::dot(f, eye);
	return m;
}

glm::mat4 perspective(float fov_y, float aspect, float z_near, float z_far) {
	const float f = 1.0f / std::tan(fov_y * 0.5f);  // зум: чем больше угол, тем меньше фигура

	glm::mat4 m(0.0f);
	m[0][0] = f / aspect;
	m[1][1] = -f;
	m[2][2] = z_far / (z_near - z_far);
	m[2][3] = -1.0f;
	m[3][2] = z_near * z_far / (z_near - z_far);
	return m;
}

glm::mat4 orthographic(float left, float right, float bottom, float top, float z_near, float z_far) {
	glm::mat4 m(1.0f);
	m[0][0] = 2.0f / (right - left);
	m[3][0] = -(right + left) / (right - left);
	m[1][1] = -2.0f / (top - bottom);
	m[3][1] = (top + bottom) / (top - bottom);
	m[2][2] = 1.0f / (z_near - z_far);
	m[3][2] = z_near / (z_near - z_far);
	return m;
}

} // namespace xform
