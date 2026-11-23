#pragma once

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

namespace geometry {

struct Vertex {
	glm::vec3 position;
	glm::vec3 normal; // лаба 2: нормаль грани
	glm::vec2 uv;     // лаба 3: текстурные координаты (заменили цвет вершины из лабы 1)
};

struct Mesh {
	std::vector<Vertex> vertices;
	std::vector<uint16_t> indices;
};

// Пирамида из лабы 1
Mesh makePyramid();
// Куб для визуализации источников света
Mesh makeCube();
// Лаба 3: конус прожектора из линий (вершина в начале координат, ось +Z, длина 1, радиус основания 1).
// Раньше линии пересчитывались каждый кадр, теперь меш один и лежит в VRAM, а конус задаётся матрицей.
Mesh makeSpotCone();

} // namespace geometry
