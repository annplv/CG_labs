#pragma once

#include <glm/glm.hpp>

// m[столбец][строка], углы в радианах
namespace xform {

glm::mat4 translation(const glm::vec3& offset);
glm::mat4 scaling(const glm::vec3& factors);

glm::mat4 rotationX(float angle);
glm::mat4 rotationY(float angle);
glm::mat4 rotationZ(float angle);

glm::mat4 lookAt(const glm::vec3& eye, const glm::vec3& target, const glm::vec3& up);

glm::mat4 perspective(float fov_y, float aspect, float z_near, float z_far);
glm::mat4 orthographic(float left, float right, float bottom, float top, float z_near, float z_far);

} // namespace xform
