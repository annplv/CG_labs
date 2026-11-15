#pragma once

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

namespace geometry {

struct Vertex {
	glm::vec3 position;
	glm::vec3 normal; // лаба 2: нормаль грани
	glm::vec3 color;  // лаба 1: цвет вершины по её положению
};

struct Mesh {
	std::vector<Vertex> vertices;
	std::vector<uint16_t> indices;
};

// Пирамида из лабы 1
Mesh makePyramid();
// Куб для визуализации источников света
Mesh makeCube();

} // namespace geometry
